// SPDX-License-Identifier: GPL-2.0-or-later

//! The `timer` section, `vmstate_timers` from system/cpu-timers.c, which every board sends
//! first. The boards register it over their own clocks.
//!
//! The icount subsections are parsed and dropped: ruvm does not run with icount.

use std::sync::LazyLock;

use ruvm_vmstate::{VmStateDescription, VmStateField};

/// The part of `TimersState` that goes on the wire.
#[derive(Debug, Clone, Default)]
pub struct TimersState {
    /// `cpu_get_ticks()`.
    pub cpu_ticks_offset: i64,
    /// `QEMU_CLOCK_VIRTUAL`.
    pub cpu_clock_offset: i64,
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
pub static VMSTATE_TIMERS: LazyLock<Vmsd> = LazyLock::new(|| {
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
