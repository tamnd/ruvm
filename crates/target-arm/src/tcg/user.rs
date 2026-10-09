// SPDX-License-Identifier: GPL-2.0-or-later

//! The `CONFIG_USER_ONLY` parts of the AArch64 front end: the state `arm_cpu_reset_hold()`
//! leaves a user mode CPU in, `arm_cpu_record_sigsegv()`, and the system registers a program
//! sees at EL0 under Linux.
//!
//! Under user mode emulation the program runs at EL0 and the emulator is its kernel. SVC and
//! every exception leave the vCPU with `exception_index` set and the syndrome recorded, and the
//! emulator's cpu loop turns them into a system call or a signal. An access to a page the
//! program has not mapped becomes a level 3 translation or permission fault.

use ruvm_jit::{Cpu, CpuLoopExit, MmuAccessType, Ra};

use super::{Arm, ptw};
use crate::cpu::{
    ArmCpuModel, CpuArmState, PSTATE_MODE_EL0T, SCTLR_ATA0, SCTLR_DZE, SCTLR_ENDA, SCTLR_ENDB,
    SCTLR_ENIA, SCTLR_ENIB, SCTLR_UCI, SCTLR_UCT,
};
use crate::syndrome::fsc;

/// `SCTLR_BT0`.
const SCTLR_BT0: u64 = 1 << 35;
/// `SCTLR_TSCXT`.
const SCTLR_TSCXT: u64 = 1 << 20;
/// `SCTLR_MSCEN`.
const SCTLR_MSCEN: u64 = 1 << 33;
/// The SVE vector length a program starts with, QEMU's `sve-default-vector-length` of 64
/// bytes.
const SVE_DEFAULT_VQ: u32 = 4;

/// The registers of a user mode CPU after `arm_cpu_reset_hold()`: EL0t with DAIF clear, EL0
/// access to the cache maintenance, DC ZVA, CTR_EL0, the FP and SVE instructions, the PAC
/// keys and MTE tags, and TBI0 for a 48-bit address space.
pub fn user_reset(model: &ArmCpuModel) -> CpuArmState {
    let f = &model.features;
    let mut s = CpuArmState::reset(model);
    s.pstate_write(PSTATE_MODE_EL0T);
    // Userspace expects access to DC ZVA, CTL_EL0 and the cache ops.
    s.sctlr_el[1] |= SCTLR_UCT | SCTLR_UCI | SCTLR_DZE;
    // Enable all PAC keys.
    s.sctlr_el[1] |= SCTLR_ENIA | SCTLR_ENIB | SCTLR_ENDA | SCTLR_ENDB;
    // Trap on btype=3 for PACIxSP.
    s.sctlr_el[1] |= SCTLR_BT0;
    // And to the FP/Neon instructions, CPACR_EL1.FPEN.
    s.cpacr_el1 |= 3 << 20;
    // And to the SVE instructions, with the default vector length.
    if f.sve {
        s.cpacr_el1 |= 3 << 16;
        s.zcr_el[1] = u64::from(SVE_DEFAULT_VQ.min(f.sve_max_vq.max(1)) - 1);
    }
    // Enable 48-bit address space. Enable TBI0 but not TBI1.
    s.tcr_el[1] = 5 | (1 << 37);
    if f.mte >= 2 {
        // Enable tag access, but leave TCF0 as No Effect (0), and exclude all tags, so that
        // tag 0 is always used.
        s.sctlr_el[1] |= SCTLR_ATA0;
        s.gcr_el1 = 0x1ffff;
    }
    // Disable access to SCXTNUM_EL0, and to the Debug Communication Channel.
    s.sctlr_el[1] |= SCTLR_TSCXT;
    s.mdscr_el1 |= 1 << 12;
    // Enable FEAT_MOPS.
    s.sctlr_el[1] |= SCTLR_MSCEN;
    s.rebuild_hflags(f);
    s
}

/// `arm_cpu_record_sigsegv()`: an access the program's mappings do not allow becomes a level
/// 3 translation fault at `addr` when nothing is mapped there, a permission fault otherwise.
pub fn record_sigsegv(
    arm: &Arm,
    cpu: &mut Cpu<'_>,
    addr: u64,
    access_type: MmuAccessType,
    maperr: bool,
    ra: Ra,
) -> CpuLoopExit {
    let code = if maperr { fsc::translation(3) } else { fsc::permission(3) };
    ptw::deliver_fault(arm, cpu, addr, access_type, ptw::Fault::new(code), ra)
}

/// Exported ID register fields, `ID_AA64*` under `modify_arm_cp_regs()`: what Linux lets
/// EL0 read of each register, as `(mask, fixed)`.
fn id_export(crm: u32, op2: u32) -> (u64, u64) {
    match (crm, op2) {
        // ID_AA64PFR0_EL1: FP, AdvSIMD, SVE and DIT; EL0 and EL1 AArch64 only.
        (4, 0) => (0x000f_000f_00ff_0000, 0x11),
        // ID_AA64PFR1_EL1: BT, SSBS, MTE and SME.
        (4, 1) => (0x0f00_0fff, 0),
        // ID_AA64ZFR0_EL1.
        (4, 4) => (0x0ff0_ff0f_0fff_00ff, 0),
        // ID_AA64DFR0_EL1: DebugVer 6.
        (5, 0) => (0, 6),
        // ID_AA64ISAR0_EL1, all but TLB.
        (6, 0) => (0xf0ff_ffff_f0ff_fff0, 0),
        // ID_AA64ISAR1_EL1, all but XS and SPECRES.
        (6, 1) => (0x00ff_f0ff_ffff_ffff, 0),
        // ID_AA64ISAR2_EL1.
        (6, 2) => (0x00ff_0000_00ff_ffff, 0),
        // ID_AA64MMFR0_EL1: ECV; TGran64 and TGran4 not supported.
        (7, 0) => (0xf << 60, 0xff00_0000),
        // ID_AA64MMFR1_EL1: AFP.
        (7, 1) => (0xf << 44, 0),
        // ID_AA64MMFR2_EL1: AT.
        (7, 2) => (0xf << 32, 0),
        _ => (0, 0),
    }
}

/// The ID register at CRm `crm` and op2 `op2` as EL0 reads it under user mode emulation.
pub(crate) fn user_id_reg(model: &ArmCpuModel, crm: u32, op2: u32) -> u64 {
    let v = match (crm, op2) {
        (4, 0) => model.id_aa64pfr0,
        (4, 1) => model.id_aa64pfr1,
        (4, 4) => model.id_aa64zfr0,
        (5, 0) => model.id_aa64dfr0,
        (6, 0) => model.id_aa64isar0,
        (6, 1) => model.id_aa64isar1,
        (6, 2) => model.id_aa64isar2,
        (7, 0) => model.id_aa64mmfr0,
        (7, 1) => model.id_aa64mmfr1,
        (7, 2) => model.id_aa64mmfr2,
        _ => 0,
    };
    let (mask, fixed) = id_export(crm, op2);
    (v & mask) | fixed
}

/// CNTVCT_EL0 under user mode emulation, `gt_virt_cnt_read()`: the host's realtime clock in
/// ticks of `cntfrq`.
pub(crate) fn user_cntvct(cntfrq: u64) -> u64 {
    let ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    // gt_cntfrq_period_ns().
    let period = if cntfrq == 0 || cntfrq >= 1_000_000_000 { 1 } else { 1_000_000_000 / cntfrq };
    ns / period
}

/// The SVE vector length in quadwords at EL0, `sve_vq()`, when the model has SVE.
pub fn sve_vq(cpu: &Cpu<'_>, st: &CpuArmState) -> Option<u32> {
    let ops = cpu.ops();
    let f = super::arm_of(&ops).features();
    f.sve.then(|| super::sve_vqm1_for_el(f, st, 0) + 1)
}

/// Store `st` into `cpu` with its TB flags rebuilt, after a signal frame changes it.
pub fn commit(cpu: &mut Cpu<'_>, st: &mut CpuArmState) {
    super::commit(cpu, st);
}

/// `vfp_set_fpcr()`, as `restore_fpsimd_context()` writes FPCR.
pub fn set_fpcr(cpu: &Cpu<'_>, st: &mut CpuArmState, val: u32) {
    let ops = cpu.ops();
    super::vfp::set_fpcr(st, val, super::arm_of(&ops).features());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_is_el0_with_user_access() {
        let m = ArmCpuModel::max();
        let s = user_reset(&m);
        assert_eq!(s.current_el(), 0);
        assert_eq!(s.daif, 0);
        assert_ne!(s.sctlr_el[1] & SCTLR_UCT, 0);
        assert_eq!(s.cpacr_el1 & (3 << 20), 3 << 20);
        assert_eq!(s.zcr_el[1], 3);
        assert_eq!(s.tcr_el[1], 5 | (1 << 37));
    }

    #[test]
    fn id_regs_show_exported_fields() {
        let m = ArmCpuModel::max();
        assert_eq!(user_id_reg(&m, 5, 0), 6);
        assert_eq!(user_id_reg(&m, 4, 0) & 0xff, 0x11);
        assert_eq!(user_id_reg(&m, 7, 0) & 0xff, 0);
        assert_eq!(user_id_reg(&m, 6, 0) & (0xf << 56), 0);
        assert_eq!(user_id_reg(&m, 4, 2), 0);
    }
}
