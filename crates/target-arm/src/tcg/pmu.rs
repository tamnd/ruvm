// SPDX-License-Identifier: GPL-2.0-or-later

//! The PMUv3 performance monitors: the port of the AArch64 parts of QEMU's
//! `target/arm/cpregs-pmu.c` and the PMU bracketing of the MDCR_EL2 and MDCR_EL3 writes in
//! `helper.c`.
//!
//! The counters work as in QEMU: while a counter runs, its register holds the count it had
//! when it was last looked at and its delta holds the underlying count it is measured
//! from. [`op_start`] brings the registers up to date, so that an access or a change of
//! the settings that decide whether a counter runs can work on the guest visible value,
//! and [`op_finish`] turns them back into deltas and arms the overflow timer. Exception
//! entry and return call the pair around the EL change, as QEMU's `pmu_pre_el_change()`
//! and `pmu_post_el_change()` hooks do. The overflow interrupt goes to the board through
//! [`super::ArmBoard::pmu_set_level`], and the board calls [`super::Arm::pmu_timer_expired`]
//! at the deadline it is given through [`super::ArmBoard::pmu_timer_anticipate`].
//!
//! Differences from QEMU:
//!
//! - The cycle counter counts host nanoseconds since the CPU was made, at QEMU's 1 GHz
//!   `ARM_CPU_FREQ`, where QEMU counts `QEMU_CLOCK_VIRTUAL`.
//! - INST_RETIRED needs precise icount in QEMU, which this port does not have, so it is
//!   never supported, as in QEMU without icount.
//! - Only the AArch64 views exist, and there are no fine grained traps.

use std::time::{Duration, Instant};

use ruvm_jit::CpuShared;

use super::sysreg::{Access, key};
use super::{Arm, is_secure};
use crate::cpu::{ArmFeatures, CpuArmState};

const PMCR_N_MASK: u64 = 0xf800;
const PMCR_N_SHIFT: u32 = 11;
const PMCR_LP: u64 = 0x80;
const PMCR_LC: u64 = 0x40;
const PMCR_DP: u64 = 0x20;
const PMCR_X: u64 = 0x10;
const PMCR_D: u64 = 0x8;
const PMCR_C: u64 = 0x4;
const PMCR_P: u64 = 0x2;
const PMCR_E: u64 = 0x1;
const PMCR_WRITABLE_MASK: u64 = PMCR_LP | PMCR_LC | PMCR_DP | PMCR_X | PMCR_D | PMCR_E;

const PMXEVTYPER_P: u64 = 0x8000_0000;
const PMXEVTYPER_U: u64 = 0x4000_0000;
const PMXEVTYPER_NSK: u64 = 0x2000_0000;
const PMXEVTYPER_NSU: u64 = 0x1000_0000;
const PMXEVTYPER_NSH: u64 = 0x0800_0000;
const PMXEVTYPER_M: u64 = 0x0400_0000;
const PMXEVTYPER_MT: u64 = 0x0200_0000;
const PMXEVTYPER_EVTCOUNT: u64 = 0xffff;
const PMXEVTYPER_MASK: u64 = PMXEVTYPER_P
    | PMXEVTYPER_U
    | PMXEVTYPER_NSK
    | PMXEVTYPER_NSU
    | PMXEVTYPER_NSH
    | PMXEVTYPER_M
    | PMXEVTYPER_MT
    | PMXEVTYPER_EVTCOUNT;
const PMCCFILTR_EL0: u64 = 0xf800_0000 | PMXEVTYPER_M;

const MDCR_HPMN: u64 = 0x1f;
const MDCR_TPMCR: u64 = 1 << 5;
const MDCR_TPM: u64 = 1 << 6;
const MDCR_HPME: u64 = 1 << 7;
/// MDCR_EL2.HPMD, the same bit as MDCR_EL3.SPME.
const MDCR_HPMD: u64 = 1 << 17;
const MDCR_SPME: u64 = 1 << 17;
/// MDCR_EL2.HCCD, the same bit as MDCR_EL3.SCCD.
const MDCR_HCCD: u64 = 1 << 23;
const MDCR_SCCD: u64 = 1 << 23;
const MDCR_HLP: u64 = 1 << 26;
/// The MDCR_EL2 bits that decide whether counters run, so that a write changing them is
/// bracketed by [`op_start`] and [`op_finish`].
const MDCR_EL2_PMU_ENABLE_BITS: u64 = MDCR_HPME | MDCR_HPMD | MDCR_HPMN | MDCR_HCCD | MDCR_HLP;
/// The same for MDCR_EL3.
const MDCR_EL3_PMU_ENABLE_BITS: u64 = MDCR_SPME | MDCR_SCCD;

/// The cycle counter's index.
const CCNT: usize = 31;

/// The SW_INCR event.
const EV_SW_INCR: u64 = 0x00;
/// The CPU_CYCLES event.
const EV_CPU_CYCLES: u64 = 0x11;
/// STALL_FRONTEND, which never fires here.
const EV_STALL_FRONTEND: u64 = 0x23;
/// STALL_BACKEND, which never fires here.
const EV_STALL_BACKEND: u64 = 0x24;
/// STALL, which never fires here.
const EV_STALL: u64 = 0x3c;

/// `ARM_CPU_FREQ`, the cycle counter's rate.
const CPU_FREQ: u64 = 1_000_000_000;

pub(crate) const PMCR_EL0: u32 = key(3, 3, 9, 12, 0);
pub(crate) const PMCNTENSET_EL0: u32 = key(3, 3, 9, 12, 1);
pub(crate) const PMCNTENCLR_EL0: u32 = key(3, 3, 9, 12, 2);
pub(crate) const PMOVSCLR_EL0: u32 = key(3, 3, 9, 12, 3);
pub(crate) const PMSWINC_EL0: u32 = key(3, 3, 9, 12, 4);
pub(crate) const PMSELR_EL0: u32 = key(3, 3, 9, 12, 5);
pub(crate) const PMCCNTR_EL0: u32 = key(3, 3, 9, 13, 0);
pub(crate) const PMXEVTYPER_EL0: u32 = key(3, 3, 9, 13, 1);
pub(crate) const PMXEVCNTR_EL0: u32 = key(3, 3, 9, 13, 2);
pub(crate) const PMUSERENR_EL0: u32 = key(3, 3, 9, 14, 0);
pub(crate) const PMINTENSET_EL1: u32 = key(3, 0, 9, 14, 1);
pub(crate) const PMINTENCLR_EL1: u32 = key(3, 0, 9, 14, 2);
pub(crate) const PMOVSSET_EL0: u32 = key(3, 3, 9, 14, 3);
pub(crate) const PMCCFILTR_EL0_KEY: u32 = key(3, 3, 14, 15, 7);
pub(crate) const MDCR_EL2: u32 = key(3, 4, 1, 1, 1);
pub(crate) const MDCR_EL3: u32 = key(3, 6, 1, 3, 1);

/// `pmu_num_counters()`: the number of event counters, PMCR_EL0.N of the model.
pub(crate) fn num_counters(f: &ArmFeatures) -> usize {
    usize::from(f.pmu_counters)
}

/// `pmu_counter_mask()`: the bits of PMCNTEN*, PMINTEN* and PMOVS* that exist.
fn counter_mask(f: &ArmFeatures) -> u64 {
    (1 << 31) | ((1 << num_counters(f)) - 1)
}

/// `arm_mdcr_el2_eff()`.
fn mdcr_el2_eff(f: &ArmFeatures, st: &CpuArmState) -> u64 {
    if st.is_el2_enabled(f) { st.mdcr_el2 } else { 0 }
}

/// `isar_feature_any_pmuv3p1()`.
fn pmuv3p1(f: &ArmFeatures) -> bool {
    f.pmu >= 4 && f.pmu != 0xf
}

/// `isar_feature_any_pmuv3p4()`.
pub(crate) fn pmuv3p4(f: &ArmFeatures) -> bool {
    f.pmu >= 5 && f.pmu != 0xf
}

/// `isar_feature_any_pmuv3p5()`.
fn pmuv3p5(f: &ArmFeatures) -> bool {
    f.pmu >= 6 && f.pmu != 0xf
}

/// `event_supported()`: whether `pm_events[]` has `event` on this CPU.
fn event_supported(f: &ArmFeatures, event: u64) -> bool {
    match event {
        EV_SW_INCR | EV_CPU_CYCLES => true,
        EV_STALL_FRONTEND | EV_STALL_BACKEND => pmuv3p1(f),
        EV_STALL => pmuv3p4(f),
        _ => false,
    }
}

/// The `get_count` of a supported event: the underlying count. Only the cycles move.
fn event_count(arm: &Arm, event: u64) -> u64 {
    if event == EV_CPU_CYCLES { arm.counter(CPU_FREQ) } else { 0 }
}

/// The `ns_per_count` of a supported event: how long `count` events take, or a negative
/// number when they never happen.
fn event_ns_per(event: u64, count: u64) -> i64 {
    if event == EV_CPU_CYCLES { (CPU_FREQ / 1_000_000_000).wrapping_mul(count) as i64 } else { -1 }
}

/// `pmu_init()`: PMCEID0_EL0 and PMCEID1_EL0, the supported events below 0x40.
pub(crate) fn pmceid(f: &ArmFeatures) -> (u64, u64) {
    let (mut id0, mut id1) = (0, 0);
    for ev in [EV_SW_INCR, EV_CPU_CYCLES, EV_STALL_FRONTEND, EV_STALL_BACKEND, EV_STALL] {
        if event_supported(f, ev) {
            if ev & 0x20 != 0 {
                id1 |= 1 << (ev & 0x1f);
            } else {
                id0 |= 1 << (ev & 0x1f);
            }
        }
    }
    (id0, id1)
}

/// The access checks of the PMU registers, the `accessfn`s.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PmuTrap {
    /// `pmreg_access()`.
    Reg,
    /// `pmreg_access_pmcr()`.
    Pmcr,
    /// `pmreg_access_xevcntr()`: PMUSERENR_EL0.ER lets EL0 read.
    Xevcntr,
    /// `pmreg_access_swinc()`: PMUSERENR_EL0.SW lets EL0 write.
    Swinc,
    /// `pmreg_access_selr()`: PMUSERENR_EL0.ER lets EL0 in.
    Selr,
    /// `pmreg_access_ccntr()`: PMUSERENR_EL0.CR lets EL0 read.
    Ccntr,
    /// `access_tpm()`: MDCR_EL2.TPM and MDCR_EL3.TPM only.
    Tpm,
}

/// `do_pmreg_access()`.
fn pmreg_access(f: &ArmFeatures, st: &CpuArmState, is_pmcr: bool) -> Access {
    let el = st.current_el();
    if el == 0 && st.pmuserenr & 1 == 0 {
        return Access::TrapEl1;
    }
    if el < 2 {
        let mdcr_el2 = mdcr_el2_eff(f, st);
        if mdcr_el2 & MDCR_TPM != 0 || (is_pmcr && mdcr_el2 & MDCR_TPMCR != 0) {
            return Access::TrapEl2;
        }
    }
    if el < 3 && st.mdcr_el3 & MDCR_TPM != 0 {
        return Access::TrapEl3;
    }
    Access::Ok
}

impl PmuTrap {
    /// Run the check.
    pub(crate) fn check(self, f: &ArmFeatures, st: &CpuArmState, isread: bool) -> Access {
        let el = st.current_el();
        let user = |bit: u32, ok: bool| el == 0 && ok && st.pmuserenr & (1 << bit) != 0;
        match self {
            PmuTrap::Reg => pmreg_access(f, st, false),
            PmuTrap::Pmcr => pmreg_access(f, st, true),
            PmuTrap::Xevcntr if user(3, isread) => Access::Ok,
            PmuTrap::Swinc if user(1, !isread) => Access::Ok,
            PmuTrap::Selr if user(3, true) => Access::Ok,
            PmuTrap::Ccntr if user(2, isread) => Access::Ok,
            PmuTrap::Xevcntr | PmuTrap::Swinc | PmuTrap::Selr | PmuTrap::Ccntr => {
                pmreg_access(f, st, false)
            }
            PmuTrap::Tpm => {
                if el < 2 && mdcr_el2_eff(f, st) & MDCR_TPM != 0 {
                    Access::TrapEl2
                } else if el < 3 && st.mdcr_el3 & MDCR_TPM != 0 {
                    Access::TrapEl3
                } else {
                    Access::Ok
                }
            }
        }
    }
}

/// `pmu_counter_enabled()`: whether `counter` (31 for the cycle counter) counts at the
/// current EL and Security state.
fn counter_enabled(f: &ArmFeatures, st: &CpuArmState, counter: usize) -> bool {
    if f.pmu == 0 {
        return false;
    }
    let secure = is_secure(f, st);
    let el = st.current_el();
    let mdcr_el2 = mdcr_el2_eff(f, st);
    let hpmn = (mdcr_el2 & MDCR_HPMN) as usize;
    let low = counter < hpmn || counter == CCNT;

    let e = if !f.el2 || low { st.pmcr & PMCR_E != 0 } else { mdcr_el2 & MDCR_HPME != 0 };
    let enabled = e && st.pmcnten & (1 << counter) != 0;

    // Is event counting prohibited?
    let mut prohibited = el == 2 && low && mdcr_el2 & MDCR_HPMD != 0;
    if secure {
        prohibited = prohibited || st.mdcr_el3 & MDCR_SPME == 0;
    }
    if counter == CCNT {
        // The cycle counter defaults to running. PMCR.DP says "disable the cycle counter
        // when event counting is prohibited". Some MDCR bits disable the cycle counter
        // specifically.
        prohibited = prohibited && st.pmcr & PMCR_DP != 0;
        if pmuv3p5(f) {
            if secure {
                prohibited = prohibited || st.mdcr_el3 & MDCR_SCCD != 0;
            }
            if el == 2 {
                prohibited = prohibited || mdcr_el2 & MDCR_HCCD != 0;
            }
        }
    }

    let filter = if counter == CCNT { st.pmccfiltr } else { st.pmevtyper[counter] };
    let p = filter & PMXEVTYPER_P != 0;
    let u = filter & PMXEVTYPER_U != 0;
    let nsk = f.el3 && filter & PMXEVTYPER_NSK != 0;
    let nsu = f.el3 && filter & PMXEVTYPER_NSU != 0;
    let nsh = f.el2 && filter & PMXEVTYPER_NSH != 0;
    // EL1 is always AArch64 here.
    let m = f.el3 && filter & PMXEVTYPER_M != 0;
    let filtered = match el {
        0 => {
            if secure {
                u
            } else {
                u != nsu
            }
        }
        1 => {
            if secure {
                p
            } else {
                p != nsk
            }
        }
        2 => !nsh,
        _ => m != p,
    };

    // If not checking PMCCNTR, ensure the counter is set up to an event we support.
    if counter != CCNT && !event_supported(f, filter & PMXEVTYPER_EVTCOUNT) {
        return false;
    }
    enabled && !prohibited && !filtered
}

/// A vCPU's PMU as the register accesses and EL changes see it.
pub(crate) struct Pmu<'a> {
    arm: &'a Arm,
    shared: &'a CpuShared,
}

impl<'a> Pmu<'a> {
    /// The PMU of the vCPU `shared` of `arm`.
    pub(crate) fn new(arm: &'a Arm, shared: &'a CpuShared) -> Pmu<'a> {
        Pmu { arm, shared }
    }

    fn f(&self) -> &ArmFeatures {
        self.arm.features()
    }

    /// `pmu_update_irq()`.
    fn update_irq(&self, st: &CpuArmState) {
        let level = st.pmcr & PMCR_E != 0 && st.pminten & st.pmovsr != 0;
        if let Some(board) = &self.arm.board {
            board.pmu_set_level(self.shared, level);
        }
    }

    /// `timer_mod_anticipate_ns(cpu->pmu_timer, now + overflow_in)` when `overflow_in` is
    /// positive.
    fn anticipate(&self, overflow_in: i64) {
        if overflow_in <= 0 {
            return;
        }
        let Some(at) = Instant::now().checked_add(Duration::from_nanos(overflow_in as u64)) else {
            return;
        };
        if let Some(board) = &self.arm.board {
            board.pmu_timer_anticipate(self.shared, at);
        }
    }

    /// `pmccntr_clockdiv_enabled()`: PMCR.D divides by 64 unless PMCR.LC is set.
    fn clockdiv(st: &CpuArmState) -> bool {
        st.pmcr & (PMCR_D | PMCR_LC) == PMCR_D
    }

    /// `pmevcntr_is_64_bit()`.
    fn evcntr_is_64_bit(&self, st: &CpuArmState, counter: usize) -> bool {
        let f = self.f();
        if !pmuv3p5(f) {
            return false;
        }
        if f.el2 {
            // MDCR_EL2.HLP still applies even when EL2 is disabled in the current security
            // state, so this does not use the effective MDCR_EL2.
            let hpmn = (st.mdcr_el2 & MDCR_HPMN) as usize;
            if counter >= hpmn {
                return st.mdcr_el2 & MDCR_HLP != 0;
            }
        }
        st.pmcr & PMCR_LP != 0
    }

    /// `pmccntr_op_start()`: make `ccnt` the guest visible count.
    fn ccntr_op_start(&self, st: &mut CpuArmState) {
        let cycles = event_count(self.arm, EV_CPU_CYCLES);
        if counter_enabled(self.f(), st, CCNT) {
            let eff_cycles = if Self::clockdiv(st) { cycles / 64 } else { cycles };
            let new = eff_cycles.wrapping_sub(st.ccnt_delta);
            let overflow_mask = if st.pmcr & PMCR_LC != 0 { 1 << 63 } else { 1 << 31 };
            if st.ccnt & !new & overflow_mask != 0 {
                st.pmovsr |= 1 << 31;
                self.update_irq(st);
            }
            st.ccnt = new;
        }
        st.ccnt_delta = cycles;
    }

    /// `pmccntr_op_finish()`: if the cycle counter runs, arm the overflow timer and turn
    /// `ccnt_delta` back into the difference between the clock and the count.
    fn ccntr_op_finish(&self, st: &mut CpuArmState) {
        if counter_enabled(self.f(), st, CCNT) {
            // Calculate when the counter will next overflow.
            let mut remaining = st.ccnt.wrapping_neg();
            if st.pmcr & PMCR_LC == 0 {
                remaining &= 0xffff_ffff;
            }
            self.anticipate(event_ns_per(EV_CPU_CYCLES, remaining));
            let prev = if Self::clockdiv(st) { st.ccnt_delta / 64 } else { st.ccnt_delta };
            st.ccnt_delta = prev.wrapping_sub(st.ccnt);
        }
    }

    /// `pmevcntr_op_start()`.
    fn evcntr_op_start(&self, st: &mut CpuArmState, counter: usize) {
        let f = self.f();
        let event = st.pmevtyper[counter] & PMXEVTYPER_EVTCOUNT;
        let count = if event_supported(f, event) { event_count(self.arm, event) } else { 0 };
        if counter_enabled(f, st, counter) {
            let new = count.wrapping_sub(st.pmevcntr_delta[counter]);
            let overflow_mask = if self.evcntr_is_64_bit(st, counter) { 1 << 63 } else { 1 << 31 };
            if st.pmevcntr[counter] & !new & overflow_mask != 0 {
                st.pmovsr |= 1 << counter;
                self.update_irq(st);
            }
            st.pmevcntr[counter] = new;
        }
        st.pmevcntr_delta[counter] = count;
    }

    /// `pmevcntr_op_finish()`.
    fn evcntr_op_finish(&self, st: &mut CpuArmState, counter: usize) {
        if counter_enabled(self.f(), st, counter) {
            let event = st.pmevtyper[counter] & PMXEVTYPER_EVTCOUNT;
            let mut delta = st.pmevcntr[counter].wrapping_add(1).wrapping_neg();
            if !self.evcntr_is_64_bit(st, counter) {
                delta &= 0xffff_ffff;
            }
            self.anticipate(event_ns_per(event, delta));
            st.pmevcntr_delta[counter] =
                st.pmevcntr_delta[counter].wrapping_sub(st.pmevcntr[counter]);
        }
    }

    /// `pmu_op_start()`: bring every counter register up to date.
    pub(crate) fn op_start(&self, st: &mut CpuArmState) {
        if self.f().pmu == 0 {
            return;
        }
        self.ccntr_op_start(st);
        for i in 0..num_counters(self.f()) {
            self.evcntr_op_start(st, i);
        }
    }

    /// `pmu_op_finish()`: let the counters run again from their registers.
    pub(crate) fn op_finish(&self, st: &mut CpuArmState) {
        if self.f().pmu == 0 {
            return;
        }
        self.ccntr_op_finish(st);
        for i in 0..num_counters(self.f()) {
            self.evcntr_op_finish(st, i);
        }
    }

    /// `pmevtyper_read()`: 31 is PMCCFILTR_EL0, counters that do not exist read as zero.
    fn evtyper_read(&self, st: &CpuArmState, counter: usize) -> u64 {
        if counter == CCNT {
            st.pmccfiltr
        } else if counter < num_counters(self.f()) {
            st.pmevtyper[counter]
        } else {
            0
        }
    }

    /// `pmevtyper_write()`.
    fn evtyper_write(&self, st: &mut CpuArmState, counter: usize, value: u64) {
        if counter == CCNT {
            self.ccntr_op_start(st);
            st.pmccfiltr = value & PMCCFILTR_EL0;
            self.ccntr_op_finish(st);
        } else if counter < num_counters(self.f()) {
            self.evcntr_op_start(st, counter);
            // If this counter's event type is changing, store the current underlying count
            // for the new type in the delta so op_finish has the correct baseline when it
            // converts back to a delta.
            let old_event = st.pmevtyper[counter] & PMXEVTYPER_EVTCOUNT;
            let new_event = value & PMXEVTYPER_EVTCOUNT;
            if old_event != new_event {
                st.pmevcntr_delta[counter] = if event_supported(self.f(), new_event) {
                    event_count(self.arm, new_event)
                } else {
                    0
                };
            }
            st.pmevtyper[counter] = value & PMXEVTYPER_MASK;
            self.evcntr_op_finish(st, counter);
        }
        // Accesses to counters that do not exist are CONSTRAINED UNPREDICTABLE; they are
        // RAZ/WI, as in QEMU.
    }

    /// `pmevcntr_read()`.
    fn evcntr_read(&self, st: &mut CpuArmState, counter: usize) -> u64 {
        if counter >= num_counters(self.f()) {
            return 0;
        }
        self.evcntr_op_start(st, counter);
        let mut ret = st.pmevcntr[counter];
        self.evcntr_op_finish(st, counter);
        if !pmuv3p5(self.f()) {
            // Before FEAT_PMUv3p5, the top 32 bits of the event counters are RES0.
            ret &= 0xffff_ffff;
        }
        ret
    }

    /// `pmevcntr_write()`.
    fn evcntr_write(&self, st: &mut CpuArmState, counter: usize, mut value: u64) {
        if !pmuv3p5(self.f()) {
            value &= 0xffff_ffff;
        }
        if counter < num_counters(self.f()) {
            self.evcntr_op_start(st, counter);
            st.pmevcntr[counter] = value;
            self.evcntr_op_finish(st, counter);
        }
    }

    /// Read the PMU register (or MDCR) with encoding `key_`.
    pub(crate) fn read(&self, st: &mut CpuArmState, key_: u32) -> u64 {
        match key_ {
            PMCR_EL0 => {
                // pmcr_read(): if EL2 is implemented and enabled for the current security
                // state, reads of PMCR.N from EL1 or EL0 return MDCR_EL2.HPMN.
                let mut pmcr = st.pmcr;
                if st.current_el() <= 1 && st.is_el2_enabled(self.f()) {
                    pmcr = (pmcr & !PMCR_N_MASK) | ((st.mdcr_el2 & MDCR_HPMN) << PMCR_N_SHIFT);
                }
                pmcr
            }
            PMCNTENSET_EL0 | PMCNTENCLR_EL0 => st.pmcnten,
            PMOVSCLR_EL0 | PMOVSSET_EL0 => st.pmovsr,
            PMINTENSET_EL1 | PMINTENCLR_EL1 => st.pminten,
            PMUSERENR_EL0 => st.pmuserenr,
            PMSELR_EL0 => st.pmselr,
            PMCCFILTR_EL0_KEY => st.pmccfiltr,
            PMCCNTR_EL0 => {
                self.ccntr_op_start(st);
                let ret = st.ccnt;
                self.ccntr_op_finish(st);
                ret
            }
            PMXEVTYPER_EL0 => self.evtyper_read(st, (st.pmselr & 31) as usize),
            PMXEVCNTR_EL0 => self.evcntr_read(st, (st.pmselr & 31) as usize),
            MDCR_EL2 => st.mdcr_el2,
            MDCR_EL3 => st.mdcr_el3,
            _ => match evreg(key_) {
                Some((false, n)) => self.evcntr_read(st, n),
                Some((true, n)) => self.evtyper_read(st, n),
                None => panic!("no PMU register with key 0x{key_:x}"),
            },
        }
    }

    /// Write the PMU register (or MDCR) with encoding `key_`.
    pub(crate) fn write(&self, st: &mut CpuArmState, key_: u32, value: u64) {
        let mask = counter_mask(self.f());
        match key_ {
            PMCR_EL0 => {
                self.op_start(st);
                if value & PMCR_C != 0 {
                    // The counter has been reset.
                    st.ccnt = 0;
                }
                if value & PMCR_P != 0 {
                    let n = num_counters(self.f());
                    st.pmevcntr[..n].fill(0);
                }
                st.pmcr = (st.pmcr & !PMCR_WRITABLE_MASK) | (value & PMCR_WRITABLE_MASK);
                self.op_finish(st);
            }
            PMCNTENSET_EL0 | PMCNTENCLR_EL0 => {
                self.op_start(st);
                if key_ == PMCNTENSET_EL0 {
                    st.pmcnten |= value & mask;
                } else {
                    st.pmcnten &= !(value & mask);
                }
                self.op_finish(st);
            }
            PMOVSCLR_EL0 => {
                st.pmovsr &= !(value & mask);
                self.update_irq(st);
            }
            PMOVSSET_EL0 => {
                st.pmovsr |= value & mask;
                self.update_irq(st);
            }
            PMINTENSET_EL1 => {
                st.pminten |= value & mask;
                self.update_irq(st);
            }
            PMINTENCLR_EL1 => {
                st.pminten &= !(value & mask);
                self.update_irq(st);
            }
            PMUSERENR_EL0 => st.pmuserenr = value & 0xf,
            // PMSELR.SEL is checked when PMXEVTYPER and PMXEVCNTR are accessed.
            PMSELR_EL0 => st.pmselr = value & 0x1f,
            PMSWINC_EL0 => self.swinc_write(st, value),
            PMCCNTR_EL0 => {
                self.ccntr_op_start(st);
                st.ccnt = value;
                self.ccntr_op_finish(st);
            }
            PMCCFILTR_EL0_KEY => self.evtyper_write(st, CCNT, value),
            PMXEVTYPER_EL0 => self.evtyper_write(st, (st.pmselr & 31) as usize, value),
            PMXEVCNTR_EL0 => self.evcntr_write(st, (st.pmselr & 31) as usize, value),
            MDCR_EL2 | MDCR_EL3 => {
                // mdcr_el2_write() and mdcr_el3_write(): some MDCR bits affect whether
                // counters are running, so a change of those is bracketed.
                let (slot, bits) = if key_ == MDCR_EL2 {
                    (st.mdcr_el2, MDCR_EL2_PMU_ENABLE_BITS)
                } else {
                    (st.mdcr_el3, MDCR_EL3_PMU_ENABLE_BITS)
                };
                let pmu_op = (slot ^ value) & bits != 0;
                if pmu_op {
                    self.op_start(st);
                }
                if key_ == MDCR_EL2 {
                    st.mdcr_el2 = value;
                } else {
                    st.mdcr_el3 = value;
                }
                if pmu_op {
                    self.op_finish(st);
                }
            }
            _ => match evreg(key_) {
                Some((false, n)) => self.evcntr_write(st, n, value),
                Some((true, n)) => self.evtyper_write(st, n, value),
                None => panic!("no PMU register with key 0x{key_:x}"),
            },
        }
    }

    /// `pmswinc_write()`: count one SW_INCR event on each counter in `value` that counts
    /// them.
    fn swinc_write(&self, st: &mut CpuArmState, value: u64) {
        for i in 0..num_counters(self.f()) {
            if value & (1 << i) != 0
                && counter_enabled(self.f(), st, i)
                && st.pmevtyper[i] & PMXEVTYPER_EVTCOUNT == EV_SW_INCR
            {
                self.evcntr_op_start(st, i);
                // Detect if this write causes an overflow since we can't predict PMSWINC
                // overflows like we can for other events.
                let new = st.pmevcntr[i].wrapping_add(1);
                let overflow_mask = if self.evcntr_is_64_bit(st, i) { 1 << 63 } else { 1 << 31 };
                if st.pmevcntr[i] & !new & overflow_mask != 0 {
                    st.pmovsr |= 1 << i;
                    self.update_irq(st);
                }
                st.pmevcntr[i] = new;
                self.evcntr_op_finish(st, i);
            }
        }
    }
}

/// Whether `key_` is a register [`Pmu::read`] and [`Pmu::write`] handle.
pub(crate) fn is_pmu_key(key_: u32) -> bool {
    matches!(
        key_,
        PMCR_EL0
            | PMCNTENSET_EL0
            | PMCNTENCLR_EL0
            | PMOVSCLR_EL0
            | PMSWINC_EL0
            | PMSELR_EL0
            | PMCCNTR_EL0
            | PMXEVTYPER_EL0
            | PMXEVCNTR_EL0
            | PMUSERENR_EL0
            | PMINTENSET_EL1
            | PMINTENCLR_EL1
            | PMOVSSET_EL0
            | PMCCFILTR_EL0_KEY
            | MDCR_EL2
            | MDCR_EL3
    ) || evreg(key_).is_some()
}

/// PMEVCNTR<n>_EL0 (`(false, n)`) or PMEVTYPER<n>_EL0 (`(true, n)`) for `key_`, whether or
/// not the counter exists. PMCCFILTR_EL0 is not one of them.
pub(crate) fn evreg(key_: u32) -> Option<(bool, usize)> {
    let op0 = key_ >> 14;
    let op1 = (key_ >> 11) & 7;
    let crn = (key_ >> 7) & 0xf;
    let crm = (key_ >> 3) & 0xf;
    let op2 = key_ & 7;
    if op0 != 3 || op1 != 3 || crn != 14 || crm < 8 || key_ == PMCCFILTR_EL0_KEY {
        return None;
    }
    let n = (((crm & 3) << 3) | op2) as usize;
    Some((crm >= 12, n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::ArmCpuModel;

    #[test]
    fn counters_and_events() {
        for m in [
            ArmCpuModel::cortex_a57(),
            ArmCpuModel::cortex_a72(),
            ArmCpuModel::cortex_a76(),
            ArmCpuModel::neoverse_n1(),
            ArmCpuModel::max(),
        ] {
            let n = (m.reset_pmcr_el0 & PMCR_N_MASK) >> PMCR_N_SHIFT;
            assert_eq!(num_counters(&m.features) as u64, n, "{}", m.name);
        }
        let a57 = ArmCpuModel::cortex_a57();
        assert_eq!(counter_mask(&a57.features), 0x8000_003f);
        // SW_INCR and CPU_CYCLES only before PMUv3p1.
        assert_eq!(pmceid(&a57.features), (0x2_0001, 0));
        let a76 = ArmCpuModel::cortex_a76();
        // STALL_FRONTEND and STALL_BACKEND with PMUv3p1.
        assert_eq!(pmceid(&a76.features), (0x2_0001, 0x18));
        let mut f = a76.features;
        f.pmu = 6;
        // And STALL with PMUv3p4.
        assert_eq!(pmceid(&f), (0x2_0001, 0x1000_0018));
    }

    #[test]
    fn event_register_keys() {
        assert_eq!(evreg(key(3, 3, 14, 8, 0)), Some((false, 0)));
        assert_eq!(evreg(key(3, 3, 14, 11, 6)), Some((false, 30)));
        assert_eq!(evreg(key(3, 3, 14, 12, 5)), Some((true, 5)));
        assert_eq!(evreg(key(3, 3, 14, 15, 6)), Some((true, 30)));
        assert_eq!(evreg(PMCCFILTR_EL0_KEY), None);
        assert_eq!(evreg(key(3, 3, 14, 0, 1)), None);
        assert!(is_pmu_key(PMCR_EL0) && is_pmu_key(MDCR_EL3));
    }

    #[test]
    fn el0_access() {
        let model = ArmCpuModel::cortex_a76();
        let f = &model.features;
        let mut st = CpuArmState::reset(&model);
        st.pstate &= !0xc; // EL0
        assert_eq!(st.current_el(), 0);
        assert_eq!(PmuTrap::Pmcr.check(f, &st, true), Access::TrapEl1);
        st.pmuserenr = 1 << 2; // CR
        assert_eq!(PmuTrap::Ccntr.check(f, &st, true), Access::Ok);
        assert_eq!(PmuTrap::Ccntr.check(f, &st, false), Access::TrapEl1);
        st.pmuserenr = 1;
        assert_eq!(PmuTrap::Pmcr.check(f, &st, false), Access::Ok);
        assert_eq!(PmuTrap::Tpm.check(f, &st, false), Access::Ok);
    }

    #[test]
    fn cycle_counter_filter() {
        let model = ArmCpuModel::cortex_a76();
        let f = &model.features;
        let mut st = CpuArmState::reset(&model);
        assert!(!counter_enabled(f, &st, CCNT));
        st.pmcr |= PMCR_E;
        st.pmcnten = 1 << 31;
        assert!(counter_enabled(f, &st, CCNT));
        // PMCCFILTR_EL0.P filters EL1.
        st.pmccfiltr = PMXEVTYPER_P;
        assert_eq!(st.current_el(), 1);
        assert!(!counter_enabled(f, &st, CCNT));
        // An event counter on an unsupported event never counts.
        st.pmcnten |= 1;
        st.pmevtyper[0] = 0x08;
        assert!(!counter_enabled(f, &st, 0));
        st.pmevtyper[0] = EV_CPU_CYCLES;
        assert!(counter_enabled(f, &st, 0));
    }
}
