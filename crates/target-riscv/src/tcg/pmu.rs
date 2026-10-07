// SPDX-License-Identifier: GPL-2.0-or-later

//! The performance monitoring unit, a port of QEMU's `target/riscv/tcg/pmu.c` with the
//! counter parts of `csr.c` (`riscv_pmu_read_ctr()`, `riscv_pmu_write_ctr()`,
//! `riscv_pmu_ctr_get_fixed_counters_val()` and `write_mcountinhibit()`).
//!
//! `mcycle` and `minstret` count host ticks, as QEMU does without icount. With a filter in
//! `mcyclecfg`, `minstretcfg` (Smcntrpmf) or the inhibit bits of `mhpmevent`, a counter
//! only adds the ticks spent in the privilege levels it does not inhibit, which
//! [`update_fixed_ctrs`] tracks at every mode change. An `mhpmevent` that selects the cycle
//! or instruction event makes its counter count the same way, and the TLB miss events count
//! the calls of the page walk. With Sscofpmf a counter that counts ticks arms the PMU timer
//! for the tick it overflows at; the callback sets the OF bit of `mhpmevent` and raises
//! LCOFIP, as `riscv_pmu_timer_cb()` does.
//!
//! As in QEMU, the event of an `mhpmevent` value is everything but bit 63
//! (`MHPMEVENT_IDX_MASK`), so a value with an inhibit bit set selects no event the map
//! knows. An event keeps the counter it was first given until that counter's `mhpmevent`
//! is written with event 0.
//!
//! Differences from QEMU:
//!
//! - Without icount QEMU keeps separate per mode counts for cycles and instructions that
//!   both read `cpu_get_host_ticks()`; here one set serves both, with the same values.
//! - QEMU splits an overflow more than `INT64_MAX` nanoseconds away into a timer at
//!   `INT64_MAX` plus a remainder (`irq_overflow_left`). Here the board clamps the
//!   deadline instead; the callback checks the counter and arms the timer again if it has
//!   not overflowed, so only a deadline centuries away differs.

use crate::cpu::{
    COUNTEREN_CY, COUNTEREN_IR, CpuRiscvState, PRV_M, PRV_S, PRV_U, RVH, RVS, RVU, RiscvCfg,
};

/// `RISCV_PMU_EVENT_HW_CPU_CYCLES`.
pub(crate) const EVENT_HW_CPU_CYCLES: u64 = 0x01;
/// `RISCV_PMU_EVENT_HW_INSTRUCTIONS`.
pub(crate) const EVENT_HW_INSTRUCTIONS: u64 = 0x02;
/// `RISCV_PMU_EVENT_CACHE_DTLB_READ_MISS`.
pub(crate) const EVENT_CACHE_DTLB_READ_MISS: u64 = 0x10019;
/// `RISCV_PMU_EVENT_CACHE_DTLB_WRITE_MISS`.
pub(crate) const EVENT_CACHE_DTLB_WRITE_MISS: u64 = 0x1001b;
/// `RISCV_PMU_EVENT_CACHE_ITLB_PREFETCH_MISS`.
pub(crate) const EVENT_CACHE_ITLB_PREFETCH_MISS: u64 = 0x10021;

/// The events `riscv_pmu_update_event_map()` accepts, in the order of
/// `CpuRiscvState::pmu_event_ctr`.
pub(crate) const EVENTS: [u64; 5] = [
    EVENT_HW_CPU_CYCLES,
    EVENT_HW_INSTRUCTIONS,
    EVENT_CACHE_DTLB_READ_MISS,
    EVENT_CACHE_DTLB_WRITE_MISS,
    EVENT_CACHE_ITLB_PREFETCH_MISS,
];

/// `MHPMEVENT_BIT_OF`: the counter overflowed.
pub(crate) const MHPMEVENT_OF: u64 = 1 << 63;
/// `MHPMEVENT_BIT_MINH`, also `MCYCLECFG_BIT_MINH` and `MINSTRETCFG_BIT_MINH`.
pub(crate) const MHPMEVENT_MINH: u64 = 1 << 62;
/// `MHPMEVENT_BIT_SINH`.
pub(crate) const MHPMEVENT_SINH: u64 = 1 << 61;
/// `MHPMEVENT_BIT_UINH`.
pub(crate) const MHPMEVENT_UINH: u64 = 1 << 60;
/// `MHPMEVENT_BIT_VSINH`.
pub(crate) const MHPMEVENT_VSINH: u64 = 1 << 59;
/// `MHPMEVENT_BIT_VUINH`.
pub(crate) const MHPMEVENT_VUINH: u64 = 1 << 58;
/// `MHPMEVENT_FILTER_MASK`.
pub(crate) const MHPMEVENT_FILTER_MASK: u64 =
    MHPMEVENT_MINH | MHPMEVENT_SINH | MHPMEVENT_UINH | MHPMEVENT_VSINH | MHPMEVENT_VUINH;
/// `MHPMEVENT_IDX_MASK`. QEMU builds `MHPMEVENT_SSCOF_MASK` with
/// `MAKE_64BIT_MASK(63, 56)`, which is bit 63 alone.
const MHPMEVENT_IDX_MASK: u64 = !MHPMEVENT_OF;

/// The bits a write to `mhpmevent`, `mcyclecfg` or `minstretcfg` keeps: everything but
/// the inhibit bits of the privilege levels the hart does not have.
pub(crate) fn inh_avail_mask(st: &CpuRiscvState) -> u64 {
    let has = |ext| st.misa & ext != 0;
    let mut m = !MHPMEVENT_FILTER_MASK | MHPMEVENT_MINH;
    if has(RVU) {
        m |= MHPMEVENT_UINH;
    }
    if has(RVS) {
        m |= MHPMEVENT_SINH;
    }
    if has(RVH) && has(RVU) {
        m |= MHPMEVENT_VUINH;
    }
    if has(RVH) && has(RVS) {
        m |= MHPMEVENT_VSINH;
    }
    m
}

/// `riscv_pmu_counter_valid()`.
fn counter_valid(cfg: &RiscvCfg, idx: usize) -> bool {
    (3..32).contains(&idx) && cfg.pmu_mask & (1 << idx) != 0
}

/// `riscv_pmu_counter_enabled()`.
fn counter_enabled(st: &CpuRiscvState, cfg: &RiscvCfg, idx: usize) -> bool {
    counter_valid(cfg, idx) && st.mcountinhibit & (1 << idx) == 0
}

/// The counter `event` counts in, 0 for none: `g_hash_table_lookup(pmu_event_ctr_map)`.
fn event_ctr(st: &CpuRiscvState, event: u64) -> usize {
    match EVENTS.iter().position(|&e| e == event) {
        Some(i) => st.pmu_event_ctr[i] as usize,
        None => 0,
    }
}

/// `riscv_pmu_update_fixed_ctrs()`: add the ticks since the current mode was entered to its
/// count and note `now` as the time `newpriv` (in V mode with `new_virt`) is entered.
pub(crate) fn update_fixed_ctrs(st: &mut CpuRiscvState, now: u64, newpriv: u64, new_virt: bool) {
    let p = st.priv_lvl as usize;
    // The new mode can be the old one, so take the delta before the new snapshot.
    let delta = if st.virt() {
        now.wrapping_sub(st.pmu_counter_virt_prev[p])
    } else {
        now.wrapping_sub(st.pmu_counter_prev[p])
    };
    if new_virt {
        st.pmu_counter_virt_prev[newpriv as usize] = now;
    } else {
        st.pmu_counter_prev[newpriv as usize] = now;
    }
    if st.virt() {
        st.pmu_counter_virt[p] = st.pmu_counter_virt[p].wrapping_add(delta);
    } else {
        st.pmu_counter[p] = st.pmu_counter[p].wrapping_add(delta);
    }
}

/// `riscv_pmu_ctr_monitor_instructions()`.
pub(crate) fn monitor_instructions(st: &CpuRiscvState, idx: usize) -> bool {
    if idx == 2 {
        return true;
    }
    let ctr = event_ctr(st, EVENT_HW_INSTRUCTIONS);
    ctr != 0 && ctr == idx
}

/// `riscv_pmu_ctr_monitor_cycles()`.
pub(crate) fn monitor_cycles(st: &CpuRiscvState, idx: usize) -> bool {
    if idx == 0 {
        return true;
    }
    let ctr = event_ctr(st, EVENT_HW_CPU_CYCLES);
    ctr != 0 && ctr == idx
}

/// Whether counter `idx` counts ticks.
fn monitors(st: &CpuRiscvState, idx: usize) -> bool {
    monitor_cycles(st, idx) || monitor_instructions(st, idx)
}

/// `riscv_pmu_ctr_get_fixed_counters_val()`: the free running count of counter `idx`, the
/// host ticks, or with a mode filter the ticks of the modes it does not inhibit. QEMU first
/// calls `riscv_pmu_update_fixed_ctrs()` for the current mode, which moves the ticks since
/// the mode was entered into its count; adding them here instead gives the same sum and
/// leaves `st` alone.
pub(crate) fn fixed_counters_val(st: &CpuRiscvState, now: u64, idx: usize) -> u64 {
    let cfg_val = match idx {
        0 => st.mcyclecfg,
        2 => st.minstretcfg,
        _ => st.mhpmevent[idx] & MHPMEVENT_FILTER_MASK,
    };
    if cfg_val == 0 {
        return now;
    }
    let mut counter = st.pmu_counter;
    let mut counter_virt = st.pmu_counter_virt;
    let p = st.priv_lvl as usize;
    if st.virt() {
        let delta = now.wrapping_sub(st.pmu_counter_virt_prev[p]);
        counter_virt[p] = counter_virt[p].wrapping_add(delta);
    } else {
        counter[p] = counter[p].wrapping_add(now.wrapping_sub(st.pmu_counter_prev[p]));
    }
    let parts = [
        (MHPMEVENT_MINH, counter[PRV_M as usize]),
        (MHPMEVENT_SINH, counter[PRV_S as usize]),
        (MHPMEVENT_UINH, counter[PRV_U as usize]),
        (MHPMEVENT_VSINH, counter_virt[PRV_S as usize]),
        (MHPMEVENT_VUINH, counter_virt[PRV_U as usize]),
    ];
    let mut v = 0u64;
    for (bit, count) in parts {
        if cfg_val & bit == 0 {
            v = v.wrapping_add(count);
        }
    }
    v
}

/// `riscv_pmu_read_ctr()`.
pub(crate) fn read_ctr(st: &CpuRiscvState, now: u64, idx: usize) -> u64 {
    let val = st.mhpmcounter_val[idx];
    if st.mcountinhibit & (1 << idx) != 0 || !monitors(st, idx) {
        return val;
    }
    let prev = st.mhpmcounter_prev[idx];
    fixed_counters_val(st, now, idx).wrapping_sub(prev).wrapping_add(val)
}

/// `riscv_pmu_write_ctr()`. `arm` gets the delay of the overflow timer, if one is set up.
pub(crate) fn write_ctr(
    st: &mut CpuRiscvState,
    cfg: &RiscvCfg,
    now: u64,
    idx: usize,
    val: u64,
    arm: &mut dyn FnMut(u64),
) {
    st.mhpmcounter_val[idx] = val;
    if st.mcountinhibit & (1 << idx) == 0 && monitors(st, idx) {
        st.mhpmcounter_prev[idx] = fixed_counters_val(st, now, idx);
        if idx > 2 {
            setup_timer(st, cfg, val, idx, arm);
        }
    } else {
        // Other counters keep incrementing from the given value.
        st.mhpmcounter_prev[idx] = val;
    }
}

/// `write_mcountinhibit()`: stopping a counter that counts ticks folds the ticks it
/// counted into its value; starting it again restarts the count from now.
pub(crate) fn write_mcountinhibit(
    st: &mut CpuRiscvState,
    cfg: &RiscvCfg,
    now: u64,
    val: u64,
    arm: &mut dyn FnMut(u64),
) {
    let present = u64::from(cfg.pmu_mask) | COUNTEREN_CY | COUNTEREN_IR;
    let updated = (st.mcountinhibit ^ val) & present;
    st.mcountinhibit = val & present;
    for idx in 0..32 {
        if updated & (1 << idx) == 0 || !monitors(st, idx) {
            continue;
        }
        if st.mcountinhibit & (1 << idx) == 0 {
            st.mhpmcounter_prev[idx] = fixed_counters_val(st, now, idx);
            if idx > 2 {
                setup_timer(st, cfg, st.mhpmcounter_val[idx], idx, arm);
            }
        } else {
            let curr = fixed_counters_val(st, now, idx);
            let prev = st.mhpmcounter_prev[idx];
            st.mhpmcounter_val[idx] = curr.wrapping_sub(prev).wrapping_add(st.mhpmcounter_val[idx]);
        }
    }
}

/// `riscv_pmu_update_event_map()`: let counter `idx` count the event of its new
/// `mhpmevent` value `value`.
pub(crate) fn update_event_map(st: &mut CpuRiscvState, cfg: &RiscvCfg, value: u64, idx: usize) {
    if !counter_valid(cfg, idx) {
        return;
    }
    // Event 0, the reset value, drops the events of the counter.
    if value & MHPMEVENT_IDX_MASK == 0 {
        for c in &mut st.pmu_event_ctr {
            if *c == idx as u64 {
                *c = 0;
            }
        }
        return;
    }
    // QEMU keeps the event in a uint32_t, so the filter bits below bit 63 (and anything else
    // above bit 31) are dropped here.
    let event = value & MHPMEVENT_IDX_MASK & u64::from(u32::MAX);
    // An event that already has a counter keeps it, and raw events are not supported.
    if let Some(i) = EVENTS.iter().position(|&e| e == event) {
        if st.pmu_event_ctr[i] == 0 {
            st.pmu_event_ctr[i] = idx as u64;
        }
    }
}

/// `riscv_pmu_setup_timer()`: give `arm` the ticks (nanoseconds at QEMU's 1 GHz
/// `RISCV_TIMEBASE_FREQ`) until counter `idx`, now at `value`, overflows.
fn setup_timer(
    st: &CpuRiscvState,
    cfg: &RiscvCfg,
    value: u64,
    idx: usize,
    arm: &mut dyn FnMut(u64),
) {
    // No timer if LCOFI is disabled when OF is set.
    if !counter_valid(cfg, idx) || !cfg.ext_sscofpmf || st.mhpmevent[idx] & MHPMEVENT_OF != 0 {
        return;
    }
    if !monitors(st, idx) {
        return;
    }
    let overflow_delta = if value != 0 { u64::MAX - value + 1 } else { u64::MAX };
    arm(overflow_delta);
}

/// `riscv_pmu_incr_ctr()` for `event`: count one in its counter unless the counter is
/// inhibited or filters out the current mode. Gives whether the counter wrapped with its
/// OF bit clear, which raises LCOFIP.
pub(crate) fn incr_ctr(st: &mut CpuRiscvState, cfg: &RiscvCfg, event: u64) -> Option<bool> {
    if cfg.pmu_mask == 0 {
        return None;
    }
    let idx = event_ctr(st, event);
    if idx == 0 || !counter_enabled(st, cfg, idx) {
        return None;
    }
    let ev = st.mhpmevent[idx];
    let virt = st.virt();
    let inhibited = match st.priv_lvl {
        PRV_M => ev & MHPMEVENT_MINH != 0,
        PRV_S if virt => ev & MHPMEVENT_VSINH != 0,
        PRV_U if virt => ev & MHPMEVENT_VUINH != 0,
        PRV_S => ev & MHPMEVENT_SINH != 0,
        PRV_U => ev & MHPMEVENT_UINH != 0,
        _ => false,
    };
    if inhibited {
        return None;
    }
    if st.mhpmcounter_val[idx] == u64::MAX {
        st.mhpmcounter_val[idx] = 0;
        // Raise the interrupt only if OF is clear.
        if ev & MHPMEVENT_OF == 0 {
            st.mhpmevent[idx] |= MHPMEVENT_OF;
            return Some(true);
        }
    } else {
        st.mhpmcounter_val[idx] += 1;
    }
    Some(false)
}

/// `riscv_pmu_timer_cb()`: the PMU timer fired. Each counter of the cycle and instruction
/// events that has overflowed gets its OF bit, and one that has not yet arms the timer
/// again through `arm`. Gives whether LCOFIP is to be raised.
pub(crate) fn timer_cb(
    st: &mut CpuRiscvState,
    cfg: &RiscvCfg,
    now: u64,
    arm: &mut dyn FnMut(u64),
) -> bool {
    let mut raise = false;
    for event in [EVENT_HW_CPU_CYCLES, EVENT_HW_INSTRUCTIONS] {
        let idx = event_ctr(st, event);
        if !counter_enabled(st, cfg, idx) || st.mhpmevent[idx] & MHPMEVENT_OF != 0 {
            continue;
        }
        let curr = read_ctr(st, now, idx);
        // The timer cannot allow for inhibited modes, so check that the counter really
        // wrapped past the value software wrote.
        if curr >= st.mhpmcounter_val[idx] {
            setup_timer(st, cfg, curr, idx, arm);
            continue;
        }
        st.mhpmevent[idx] |= MHPMEVENT_OF;
        raise = true;
    }
    raise
}

/// `read_scountovf()`: the OF bits of `mhpmevent3` to `mhpmevent31`, below M mode only of
/// the counters `mcounteren` (and in V mode `hcounteren`) gives.
pub(crate) fn scountovf(st: &CpuRiscvState) -> u64 {
    let mut v = 0;
    for i in 3..32 {
        if st.priv_lvl < PRV_M {
            if st.mcounteren & (1 << i) == 0 {
                continue;
            }
            if st.virt() && st.hcounteren & (1 << i) == 0 {
                continue;
            }
        }
        if st.mhpmevent[i] & MHPMEVENT_OF != 0 {
            v |= 1 << i;
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st() -> CpuRiscvState {
        CpuRiscvState::reset_cfg(0, 0x1000, &RiscvCfg::max())
    }

    #[test]
    fn mode_filtered_counts() {
        let mut s = st();
        // 200 ticks in M, then 50 in S, then 30 in VU.
        update_fixed_ctrs(&mut s, 200, PRV_S, false);
        s.priv_lvl = PRV_S;
        update_fixed_ctrs(&mut s, 250, PRV_U, true);
        s.priv_lvl = PRV_U;
        s.virt_enabled = 1;
        s.mcyclecfg = MHPMEVENT_MINH;
        assert_eq!(fixed_counters_val(&s, 280, 0), 50 + 30);
        s.mcyclecfg = MHPMEVENT_VUINH;
        assert_eq!(fixed_counters_val(&s, 280, 0), 200 + 50);
        s.mcyclecfg = 0;
        assert_eq!(fixed_counters_val(&s, 280, 0), 280, "no filter reads the ticks");
    }

    #[test]
    fn event_map_and_counting() {
        let cfg = RiscvCfg::max();
        let mut s = st();
        // Raw events are not supported.
        update_event_map(&mut s, &cfg, 7, 3);
        assert!(!monitor_cycles(&s, 3));
        // The event is the low 32 bits, so the filter bits do not change it.
        update_event_map(&mut s, &cfg, MHPMEVENT_MINH | EVENT_HW_CPU_CYCLES, 3);
        assert!(monitor_cycles(&s, 3));
        // The event keeps its first counter.
        update_event_map(&mut s, &cfg, EVENT_HW_CPU_CYCLES, 4);
        assert!(!monitor_cycles(&s, 4));
        update_event_map(&mut s, &cfg, 0, 3);
        assert!(!monitor_cycles(&s, 3));
        // Counter 19 is not in the PMU mask.
        update_event_map(&mut s, &cfg, EVENT_HW_CPU_CYCLES, 19);
        assert!(!monitor_cycles(&s, 19));

        let mut armed = Vec::new();
        update_event_map(&mut s, &cfg, EVENT_HW_CPU_CYCLES, 5);
        s.mhpmevent[5] = EVENT_HW_CPU_CYCLES;
        write_ctr(&mut s, &cfg, 1000, 5, u64::MAX - 9, &mut |d| armed.push(d));
        assert_eq!(armed, [10]);
        assert_eq!(read_ctr(&s, 1004, 5), u64::MAX - 5);
        // Not yet wrapped: the timer is armed again for the rest.
        assert!(!timer_cb(&mut s, &cfg, 1004, &mut |d| armed.push(d)));
        assert_eq!(armed, [10, 6]);
        assert!(timer_cb(&mut s, &cfg, 1012, &mut |_| unreachable!()));
        assert_eq!(s.mhpmevent[5] & MHPMEVENT_OF, MHPMEVENT_OF);
        s.mcounteren = 1 << 5;
        s.priv_lvl = PRV_S;
        assert_eq!(scountovf(&s), 1 << 5);
        s.mcounteren = 0;
        assert_eq!(scountovf(&s), 0);

        // A TLB miss event counts in S mode but not with SINH.
        update_event_map(&mut s, &cfg, EVENT_CACHE_DTLB_READ_MISS, 6);
        s.mhpmevent[6] = EVENT_CACHE_DTLB_READ_MISS;
        s.mhpmcounter_val[6] = u64::MAX;
        assert_eq!(incr_ctr(&mut s, &cfg, EVENT_CACHE_DTLB_READ_MISS), Some(true));
        assert_eq!(s.mhpmcounter_val[6], 0);
        assert_eq!(incr_ctr(&mut s, &cfg, EVENT_CACHE_DTLB_READ_MISS), Some(false));
        s.mhpmevent[6] |= MHPMEVENT_SINH;
        assert_eq!(incr_ctr(&mut s, &cfg, EVENT_CACHE_DTLB_READ_MISS), None);
        assert_eq!(s.mhpmcounter_val[6], 1);
        assert_eq!(incr_ctr(&mut s, &cfg, EVENT_CACHE_ITLB_PREFETCH_MISS), None);
    }

    #[test]
    fn inhibit_folds_the_count() {
        let cfg = RiscvCfg::max();
        let mut s = st();
        let mut none = |_| unreachable!();
        write_ctr(&mut s, &cfg, 10, 0, 5, &mut none);
        assert_eq!(read_ctr(&s, 40, 0), 35);
        write_mcountinhibit(&mut s, &cfg, 40, 1, &mut none);
        assert_eq!(read_ctr(&s, 90, 0), 35);
        write_mcountinhibit(&mut s, &cfg, 100, 0, &mut none);
        assert_eq!(read_ctr(&s, 110, 0), 45);
    }
}
