// SPDX-License-Identifier: GPL-2.0-or-later

//! The `timer` section, `vmstate_timers` from system/cpu-timers.c.
//!
//! It goes out while the machine is stopped, when `cpu_ticks_offset` holds `cpu_get_ticks()`
//! and `cpu_clock_offset` holds `QEMU_CLOCK_VIRTUAL`. Loading it sets the virtual clock, so the
//! timer deadlines in the device sections that follow keep their distance from now, and sets
//! `cpu_get_ticks()`: a vCPU's TSC is that plus its `tsc_offset`, so the guest's TSC goes on
//! from where it stopped on the source once the destination starts.

use std::sync::Arc;

use ruvm_base::Result;
use ruvm_hw_core::Clock;
use ruvm_migration::SaveVm;
use ruvm_migration::timers::{TimersState, VMSTATE_TIMERS};
use ruvm_sys::hostticks::Ticks;

/// Registers the `timer` section over `clock`, `QEMU_CLOCK_VIRTUAL`, and `ticks`,
/// `cpu_get_ticks()`. QEMU registers it first, so it gets section id 0. Both are stopped while
/// the section is saved or loaded, so the guest's TSC goes on from the count it carries.
pub(crate) fn register(savevm: &mut SaveVm, clock: &Arc<Clock>, ticks: &Arc<Ticks>) {
    let (get_clock, put_clock) = (Arc::clone(clock), Arc::clone(clock));
    let (get_ticks, put_ticks) = (Arc::clone(ticks), Arc::clone(ticks));
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_TIMERS,
        move || -> Result<TimersState> {
            Ok(TimersState {
                cpu_ticks_offset: get_ticks.get() as i64,
                cpu_clock_offset: get_clock.get_ns(),
            })
        },
        move |s| {
            put_clock.set_ns(s.cpu_clock_offset);
            put_ticks.set(s.cpu_ticks_offset as u64);
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
    use std::time::Instant;

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
        let ticks = Ticks::new();
        let mut s = TimersState { cpu_ticks_offset: 1 << 40, cpu_clock_offset: 1 << 41 };
        let mut f = StreamWriter::new();
        VMSTATE_TIMERS.save(&mut f, &mut s).unwrap();
        let bytes = f.into_inner();
        // What register()'s put does with a loaded section.
        let mut back = TimersState::default();
        VMSTATE_TIMERS.load(&mut StreamReader::new(&bytes), &mut back, 2).unwrap();
        clock.set_ns(back.cpu_clock_offset);
        ticks.set(back.cpu_ticks_offset as u64);
        assert_eq!(clock.get_ns(), 1 << 41);
        // The count stays put until the machine starts, and then goes on from there.
        assert_eq!(ticks.get(), 1 << 40);
        ticks.enable();
        let now = ticks.get();
        assert!(((1 << 40)..(1 << 40) + (1 << 36)).contains(&now), "{now}");
    }
}
