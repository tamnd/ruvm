// SPDX-License-Identifier: GPL-2.0-or-later

//! The generic timers: the port of `gt_recalc_timer()`, `gt_update_irq()`, the CTL, CVAL
//! and TVAL accessors, `gt_*_redir_*` and the timer access checks from QEMU's
//! `target/arm/helper.c`.
//!
//! The five timers are indexed by `GTIMER_*`. Their outputs and the deadlines at which
//! they need recalculating go to the board through [`ArmBoard::gt_timer_update`]; the board
//! wires the outputs to its interrupt controller (the virt board's GIC PPIs) and calls
//! [`Arm::gt_timer_expired`] at the deadline, as QEMU's `gt_timer[]` QEMU timers do.
//!
//! Differences from QEMU: there is no FEAT_ECV (CNTPOFF_EL2, the CNTHCTL_EL2 mask bits) and
//! no event stream. The counter is host time since the [`Arm`] was made, scaled to
//! CNTFRQ_EL0; it is not stopped while the VM is paused. CNTV_TVAL_EL02 and the other
//! EL02 aliases use the offset of the EL1 view of the timer.
//!
//! [`ArmBoard::gt_timer_update`]: super::ArmBoard::gt_timer_update

use ruvm_jit::Cpu;

use super::Arm;
use super::sysreg::Access;
use crate::cpu::{
    ArmFeatures, CpuArmState, GTIMER_HYP, GTIMER_HYPVIRT, GTIMER_PHYS, GTIMER_VIRT, HCR_E2H,
    HCR_TGE, MMU_IDX_E20_0, MMU_IDX_E20_2, MMU_IDX_E20_2_PAN, SCR_ST,
};

/// The timer names, indexed by `GTIMER_*`, as QEMU names the timer outputs.
pub const GTIMER_NAMES: [&str; 5] = ["phys", "virt", "hyp", "sec", "hypvirt"];

/// The physical count.
fn count(arm: &Arm, st: &CpuArmState) -> u64 {
    arm.counter(st.cntfrq_el0)
}

/// `gt_indirect_access_timer_offset()`: the offset the timer itself compares against.
fn indirect_offset(st: &CpuArmState, timer: usize) -> u64 {
    if timer == GTIMER_VIRT { st.cntvoff_el2 } else { 0 }
}

/// `gt_virt_cnt_offset()`.
fn virt_cnt_offset(f: &ArmFeatures, st: &CpuArmState) -> u64 {
    let hcr = st.hcr_el2_eff(f);
    match st.current_el() {
        2 if hcr & HCR_E2H != 0 => 0,
        0 if hcr & (HCR_E2H | HCR_TGE) == HCR_E2H | HCR_TGE => 0,
        _ => st.cntvoff_el2,
    }
}

/// `gt_direct_access_timer_offset()`: the offset of a register access at the current EL.
fn direct_offset(f: &ArmFeatures, st: &CpuArmState, timer: usize) -> u64 {
    if timer == GTIMER_VIRT { virt_cnt_offset(f, st) } else { 0 }
}

/// `gt_get_countervalue()` minus `gt_virt_cnt_offset()`: CNTVCT_EL0.
pub(crate) fn virt_count(arm: &Arm, st: &CpuArmState) -> u64 {
    count(arm, st).wrapping_sub(virt_cnt_offset(arm.features(), st))
}

/// The physical count, CNTPCT_EL0.
pub(crate) fn phys_count(arm: &Arm, st: &CpuArmState) -> u64 {
    count(arm, st)
}

/// `gt_phys_redir_timeridx()` and `gt_virt_redir_timeridx()`: in the EL2&0 regime the EL0
/// timer registers access the EL2 timers.
pub(crate) fn redirect(f: &ArmFeatures, st: &CpuArmState, timer: usize) -> usize {
    match st.mmu_idx(f) {
        MMU_IDX_E20_0 | MMU_IDX_E20_2 | MMU_IDX_E20_2_PAN => match timer {
            GTIMER_PHYS => GTIMER_HYP,
            GTIMER_VIRT => GTIMER_HYPVIRT,
            t => t,
        },
        _ => timer,
    }
}

/// `gt_recalc_timer()` followed by `gt_update_irq()`: update ISTATUS, then tell the board
/// the output level and the next deadline.
pub(crate) fn recalc(arm: &Arm, cpu: &Cpu<'_>, st: &mut CpuArmState, timer: usize) {
    let ctl = st.gt_ctl[timer];
    let mut deadline = None;
    if ctl & 1 != 0 {
        // Timer enabled: calculate and set current ISTATUS, irq, and reset timer to when
        // ISTATUS next has to change.
        let offset = indirect_offset(st, timer);
        let count = count(arm, st);
        let cval = st.gt_cval[timer];
        let istatus = count.wrapping_sub(offset) >= cval;
        st.gt_ctl[timer] = (ctl & !4) | (u64::from(istatus) << 2);
        let nexttick = if istatus {
            // Next transition is when (count - offset) rolls back over to 0. If offset >
            // count then this is when count == offset; if offset <= count then this is
            // when count wraps around 2^64, which is never.
            if offset > count { Some(offset) } else { None }
        } else {
            // Next transition is when (count - offset) == cval, i.e. when count == (cval
            // + offset). If that would overflow, then again we set up the next timer for
            // the distant future.
            cval.checked_add(offset)
        };
        deadline = nexttick.and_then(|t| arm.counter_deadline(st.cntfrq_el0, t));
    } else {
        // Timer disabled: ISTATUS and timer output always clear.
        st.gt_ctl[timer] = ctl & !4;
    }
    update_irq(arm, cpu, st, timer, deadline);
}

/// `gt_update_irq()`: the output is ISTATUS and ENABLE with IMASK clear.
fn update_irq(
    arm: &Arm,
    cpu: &Cpu<'_>,
    st: &CpuArmState,
    timer: usize,
    deadline: Option<std::time::Instant>,
) {
    let level = st.gt_ctl[timer] & 7 == 5;
    if let Some(board) = &arm.board {
        board.gt_timer_update(cpu.core.shared(), timer, level, deadline);
    }
}

/// `gt_ctl_write()`.
pub(crate) fn ctl_write(arm: &Arm, cpu: &Cpu<'_>, st: &mut CpuArmState, timer: usize, v: u64) {
    let oldval = st.gt_ctl[timer];
    // ISTATUS is read only.
    st.gt_ctl[timer] = (oldval & !3) | (v & 3);
    if (oldval ^ v) & 1 != 0 {
        // Enable toggled.
        recalc(arm, cpu, st, timer);
    } else if (oldval ^ v) & 2 != 0 {
        // IMASK toggled, don't need to recalculate, just set the interrupt line based on
        // ISTATUS.
        update_irq(arm, cpu, st, timer, None);
    }
}

/// `gt_cval_write()`.
pub(crate) fn cval_write(arm: &Arm, cpu: &Cpu<'_>, st: &mut CpuArmState, timer: usize, v: u64) {
    st.gt_cval[timer] = v;
    recalc(arm, cpu, st, timer);
}

/// `gt_tval_read()` at the current EL; `direct` is false for the EL02 aliases.
pub(crate) fn tval_read(arm: &Arm, st: &CpuArmState, timer: usize, direct: bool) -> u64 {
    let offset =
        if direct { direct_offset(arm.features(), st, timer) } else { indirect_offset(st, timer) };
    u64::from(st.gt_cval[timer].wrapping_sub(count(arm, st).wrapping_sub(offset)) as u32)
}

/// `gt_tval_write()`.
pub(crate) fn tval_write(
    arm: &Arm,
    cpu: &Cpu<'_>,
    st: &mut CpuArmState,
    timer: usize,
    direct: bool,
    v: u64,
) {
    let offset =
        if direct { direct_offset(arm.features(), st, timer) } else { indirect_offset(st, timer) };
    st.gt_cval[timer] =
        count(arm, st).wrapping_sub(offset).wrapping_add(v as u32 as i32 as i64 as u64);
    recalc(arm, cpu, st, timer);
}

/// `gt_cntvoff_write()`.
pub(crate) fn cntvoff_write(arm: &Arm, cpu: &Cpu<'_>, st: &mut CpuArmState, v: u64) {
    st.cntvoff_el2 = v;
    recalc(arm, cpu, st, GTIMER_VIRT);
}

/// The CNTKCTL_EL1 view that controls EL0: CNTHCTL_EL2 in the EL2&0 regime.
fn el0_ctl(f: &ArmFeatures, st: &CpuArmState) -> (u64, Access) {
    let hcr = st.hcr_el2_eff(f);
    if hcr & (HCR_E2H | HCR_TGE) == HCR_E2H | HCR_TGE {
        (st.cnthctl_el2, Access::TrapEl2)
    } else {
        (st.cntkctl_el1, Access::TrapEl1)
    }
}

/// `gt_cntfrq_access()`.
pub(crate) fn cntfrq_access(f: &ArmFeatures, st: &CpuArmState, isread: bool) -> Access {
    let el = st.current_el();
    if el == 0 {
        let (ctl, _) = el0_ctl(f, st);
        if ctl & 3 == 0 {
            return Access::TrapEl1;
        }
    }
    let highest = if f.el3 {
        3
    } else if f.el2 {
        2
    } else {
        1
    };
    if !isread && el < highest {
        return Access::Undefined;
    }
    Access::Ok
}

/// `gt_counter_access()`: CNTPCT (`timer` PHYS) and CNTVCT (VIRT).
pub(crate) fn counter_access(f: &ArmFeatures, st: &CpuArmState, timer: usize) -> Access {
    let el = st.current_el();
    if el == 0 {
        let (ctl, trap) = el0_ctl(f, st);
        if ctl & (1 << timer) == 0 {
            return trap;
        }
    }
    if el < 2 && st.is_el2_enabled(f) && timer == GTIMER_PHYS {
        // CNTHCTL_EL2.EL1PCTEN (bit 0, or bit 10 with E2H) gates EL0 and EL1 accesses.
        let hcr = st.hcr_el2_eff(f);
        let bit = if hcr & HCR_E2H != 0 { 10 } else { 0 };
        if hcr & (HCR_E2H | HCR_TGE) != HCR_E2H | HCR_TGE && st.cnthctl_el2 & (1 << bit) == 0 {
            return Access::TrapEl2;
        }
    }
    Access::Ok
}

/// `gt_timer_access()` for the physical (`GTIMER_PHYS`) and virtual (`GTIMER_VIRT`) EL0
/// timer registers.
pub(crate) fn timer_access(f: &ArmFeatures, st: &CpuArmState, timer: usize) -> Access {
    let el = st.current_el();
    if el == 0 {
        let (ctl, trap) = el0_ctl(f, st);
        // EL0PTEN is bit 9 and EL0VTEN bit 8.
        if ctl & (1 << (9 - timer)) == 0 {
            return trap;
        }
    }
    if el < 2 && st.is_el2_enabled(f) && timer == GTIMER_PHYS {
        // CNTHCTL_EL2.EL1PCEN (bit 1, or bit 11 with E2H).
        let hcr = st.hcr_el2_eff(f);
        let bit = if hcr & HCR_E2H != 0 { 11 } else { 1 };
        if hcr & (HCR_E2H | HCR_TGE) != HCR_E2H | HCR_TGE && st.cnthctl_el2 & (1 << bit) == 0 {
            return Access::TrapEl2;
        }
    }
    Access::Ok
}

/// `gt_stimer_access()`: the Secure EL1 physical timer.
pub(crate) fn stimer_access(f: &ArmFeatures, st: &CpuArmState) -> Access {
    match st.current_el() {
        0 => Access::Undefined,
        1 if !st.is_secure_below_el3(f) => Access::Undefined,
        1 if st.scr_el3 & SCR_ST == 0 => Access::TrapEl3,
        _ => Access::Ok,
    }
}
