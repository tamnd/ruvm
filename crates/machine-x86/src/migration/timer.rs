// SPDX-License-Identifier: GPL-2.0-or-later

//! The `timer` section, `vmstate_timers` from system/cpu-timers.c.
//!
//! It goes out while the machine is stopped, when `cpu_ticks_offset` holds `cpu_get_ticks()`
//! and `cpu_clock_offset` holds `QEMU_CLOCK_VIRTUAL`. Loading it sets the virtual clock, so the
//! timer deadlines in the device sections that follow keep their distance from now, and keeps
//! the ticks for the vCPUs: a vCPU's TSC is `cpu_get_ticks()` plus its `tsc_offset`, and the
//! destination's ticks count from somewhere else.
//!
//! The icount subsections are parsed and dropped: ruvm does not run with icount.

use std::sync::{Arc, LazyLock, Mutex, PoisonError};
use std::time::Instant;

use ruvm_base::Result;
use ruvm_hw_core::Clock;
use ruvm_migration::SaveVm;
use ruvm_vmstate::{VmStateDescription, VmStateField};

/// The part of `TimersState` that goes on the wire.
#[derive(Debug, Clone, Default)]
pub(crate) struct TimersState {
    cpu_ticks_offset: i64,
    cpu_clock_offset: i64,
}

type Vmsd = VmStateDescription<TimersState>;

/// A description of `size` bytes at `version`, for the icount state ruvm drops.
fn blob(name: &'static str, version: i32, size: usize) -> Vmsd {
    VmStateDescription::new(name)
        .version_id(version)
        .minimum_version_id(version)
        .needed(|_| false)
        .field(VmStateField::unused(size))
}

/// `icount_vmstate_warp_timer`: `icount_rt_timer` and `icount_vm_timer`.
static ICOUNT_WARP: LazyLock<Vmsd> = LazyLock::new(|| blob("timer/icount/warp_timer", 1, 16));
/// `icount_vmstate_adjust_timers`: `icount_warp_timer` and `icount_rt_timer`.
static ICOUNT_TIMERS: LazyLock<Vmsd> = LazyLock::new(|| blob("timer/icount/timers", 1, 16));
/// `icount_vmstate_shift`: `icount_time_shift` and `last_delta`.
static ICOUNT_SHIFT: LazyLock<Vmsd> = LazyLock::new(|| blob("timer/icount/shift", 2, 10));
/// `icount_vmstate_timers`: `qemu_icount_bias` and `qemu_icount`.
static ICOUNT: LazyLock<Vmsd> = LazyLock::new(|| {
    blob("timer/icount", 1, 16)
        .subsection(&ICOUNT_WARP)
        .subsection(&ICOUNT_TIMERS)
        .subsection(&ICOUNT_SHIFT)
});

/// `vmstate_timers`.
pub(crate) static VMSTATE_TIMERS: LazyLock<Vmsd> = LazyLock::new(|| {
    VmStateDescription::new("timer")
        .version_id(2)
        .minimum_version_id(1)
        .fields([
            VmStateField::scalar("cpu_ticks_offset", |s: &mut TimersState| &mut s.cpu_ticks_offset),
            VmStateField::unused(8),
            VmStateField::scalar("cpu_clock_offset", |s: &mut TimersState| &mut s.cpu_clock_offset)
                .version(2),
        ])
        .subsection(&ICOUNT)
});

/// `cpu_get_ticks()` of a TCG machine and what an incoming stream said it was.
#[derive(Debug)]
pub(crate) struct Ticks {
    base: Instant,
    incoming: Mutex<Option<i64>>,
}

impl Ticks {
    pub(crate) fn new(base: Instant) -> Arc<Self> {
        Arc::new(Ticks { base, incoming: Mutex::new(None) })
    }

    /// `cpu_get_ticks()`: nanoseconds since the base, as the vCPUs' `rdtsc` counts.
    pub(crate) fn now(&self) -> i64 {
        self.base.elapsed().as_nanos() as i64
    }

    /// What to add to the `tsc_offset` an incoming vCPU section carries, so its TSC carries on
    /// from the source's: the source's ticks minus ours. 0 when no `timer` section came in.
    pub(crate) fn tsc_adjust(&self) -> i64 {
        let incoming = *self.incoming.lock().unwrap_or_else(PoisonError::into_inner);
        incoming.map_or(0, |t| t.wrapping_sub(self.now()))
    }

    fn set_incoming(&self, ticks: i64) {
        *self.incoming.lock().unwrap_or_else(PoisonError::into_inner) = Some(ticks);
    }
}

/// Registers the `timer` section over `clock`, `QEMU_CLOCK_VIRTUAL`, and `ticks`. QEMU
/// registers it first, so it gets section id 0.
pub(crate) fn register(savevm: &mut SaveVm, clock: &Arc<Clock>, ticks: &Arc<Ticks>) {
    let (get_clock, put_clock) = (Arc::clone(clock), Arc::clone(clock));
    let (get_ticks, put_ticks) = (Arc::clone(ticks), Arc::clone(ticks));
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_TIMERS,
        move || -> Result<TimersState> {
            Ok(TimersState {
                cpu_ticks_offset: get_ticks.now(),
                cpu_clock_offset: get_clock.get_ns(),
            })
        },
        move |s| {
            put_clock.set_ns(s.cpu_clock_offset);
            put_ticks.set_incoming(s.cpu_ticks_offset);
            Ok(())
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvm_base::ClockType;
    use ruvm_hw_core::timer::TimeSource;
    use ruvm_vmstate::{StreamReader, StreamWriter};

    #[test]
    fn timer_section_layout() {
        let mut s = TimersState { cpu_ticks_offset: 0x1122, cpu_clock_offset: 0x3344 };
        let mut f = StreamWriter::new();
        VMSTATE_TIMERS.save(&mut f, &mut s).unwrap();
        let bytes = f.into_inner();
        // Two i64 and 8 unused bytes; no icount subsection.
        assert_eq!(bytes.len(), 24);
        assert_eq!(bytes[..8], 0x1122i64.to_be_bytes());
        assert_eq!(bytes[16..], 0x3344i64.to_be_bytes());

        let mut back = TimersState::default();
        VMSTATE_TIMERS.load(&mut StreamReader::new(&bytes), &mut back, 2).unwrap();
        assert_eq!((back.cpu_ticks_offset, back.cpu_clock_offset), (0x1122, 0x3344));
    }

    #[test]
    fn loading_sets_the_clock_and_the_ticks() {
        let clock = Clock::new(ClockType::Virtual, TimeSource::Monotonic(Instant::now()));
        clock.stop();
        let ticks = Ticks::new(Instant::now());
        assert_eq!(ticks.tsc_adjust(), 0);
        let mut s = TimersState { cpu_ticks_offset: 1 << 40, cpu_clock_offset: 1 << 41 };
        let mut f = StreamWriter::new();
        VMSTATE_TIMERS.save(&mut f, &mut s).unwrap();
        let bytes = f.into_inner();
        // What register()'s put does with a loaded section.
        let mut back = TimersState::default();
        VMSTATE_TIMERS.load(&mut StreamReader::new(&bytes), &mut back, 2).unwrap();
        clock.set_ns(back.cpu_clock_offset);
        ticks.set_incoming(back.cpu_ticks_offset);
        assert_eq!(clock.get_ns(), 1 << 41);
        let adjust = ticks.tsc_adjust();
        assert!(adjust <= 1 << 40 && adjust > (1 << 40) - 1_000_000_000);
    }
}
