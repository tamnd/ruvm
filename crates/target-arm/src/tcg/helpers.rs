// SPDX-License-Identifier: GPL-2.0-or-later

//! The helpers called from generated code: the integer and system parts of QEMU's
//! `helper-a64.c`, `op_helper.c` and `crc32_helper` code.
//!
//! Every helper is described by a [`Def`] with the name, flags and signature the translator
//! declares, so the declaration and the registration cannot drift apart. Helpers that touch
//! the CPU take `env` as their first argument, so their own arguments start at `args[1]`.
//! The translator writes the PC of the instruction to `env` before calling a helper that can
//! raise an exception, so those helpers raise with [`Ra::None`]; DC ZVA, whose faults come
//! from the softmmu, unwinds with [`Ra::Tb`] instead.

use std::sync::atomic::Ordering;

use ruvm_jit::cputlb::cpu_st_mmu;
use ruvm_jit::{Cpu, CpuLoopExit, Ra, excp};
use ruvm_jit_core::types::call_flags::NO_RWG_SE;
use ruvm_jit_core::{HelperInfo, HelperType, MemOp, MemOpIdx};
use ruvm_jit_interp::{HelperEnv, HelperRegistry, Unwind};

use super::{GicAccess, arm_of, exception_target_el, psci, sysreg};
use crate::cpu::{
    CpuArmState, EXCP_HVC, EXCP_HYP_TRAP, EXCP_PREFETCH_ABORT, EXCP_SMC, EXCP_UDEF, HCR_HCD,
    HCR_NV, HCR_TGE, HCR_TSC, HCR_TWI, HFLAGS, PSTATE_DAIF, PSTATE_IL, PSTATE_NRW, PSTATE_NZCV,
    PSTATE_PAN, PSTATE_SS, PSTATE_UAO, SCR_HCE, SCR_SMD, SCR_TWI, SCTLR_NTWI, SCTLR_UMA,
};
use crate::syndrome::{
    EC_ADVSIMDFPACCESSTRAP, syn_aa64_sysregtrap, syn_get_ec, syn_pcalignment, syn_uncategorized,
    syn_wfx,
};

type R<T> = Result<T, CpuLoopExit>;

/// A helper's declaration and implementation.
pub(crate) struct Def {
    /// The name.
    pub(crate) name: &'static str,
    /// `call_flags`.
    pub(crate) flags: u32,
    /// The return type.
    pub(crate) ret: HelperType,
    /// The argument types.
    pub(crate) args: &'static [HelperType],
    pub(crate) f: ruvm_jit_interp::HelperFn,
}

impl Def {
    /// The [`HelperInfo`] the translator declares.
    pub(crate) fn info(&self) -> HelperInfo {
        HelperInfo::new(self.name, self.flags, self.ret, self.args)
    }
}

use HelperType::{I32, I64, Ptr, Void};

macro_rules! def {
    ($id:ident, $name:literal, $flags:expr, $ret:expr, [$($a:expr),*], $f:expr) => {
        pub(crate) const $id: Def =
            Def { name: $name, flags: $flags, ret: $ret, args: &[$($a),*], f: $f };
    };
}
pub(crate) use def;

/// Run `f` on the vCPU behind `h`, turning a guest exception into an [`Unwind`].
pub(crate) fn run(
    h: &mut HelperEnv<'_>,
    f: impl FnOnce(&mut Cpu<'_>) -> R<u64>,
) -> Result<u128, Unwind> {
    let mut cpu = Cpu::from_helper_env(h).expect("arm helpers run under the runtime");
    match f(&mut cpu) {
        Ok(v) => Ok(u128::from(v)),
        Err(e) => Err(cpu.unwind(e)),
    }
}

/// `raise_exception()`: take `excp` with `syndrome` to `target_el`, or to EL2 instead of EL1
/// when HCR_EL2.TGE is set.
pub(crate) fn raise_exception(
    cpu: &mut Cpu<'_>,
    excp: i32,
    mut syndrome: u32,
    mut target_el: u32,
    ra: Ra,
) -> CpuLoopExit {
    let mut st = CpuArmState::load(cpu.env);
    let ops = cpu.ops();
    if target_el == 1 && st.hcr_el2_eff(arm_of(&ops).features()) & HCR_TGE != 0 {
        // Redirect NS EL1 exceptions to NS EL2. These are reported with their original
        // syndrome register value, with the exception of SIMD/FP access traps, which are
        // reported as uncategorized exceptions.
        target_el = 2;
        if syn_get_ec(syndrome) == EC_ADVSIMDFPACCESSTRAP {
            syndrome = syn_uncategorized();
        }
    }
    st.exception_syndrome = syndrome;
    st.exception_target_el = target_el;
    super::commit(cpu, &mut st);
    cpu.raise_exception(excp, ra)
}

def!(
    EXCEPTION_WITH_SYNDROME_EL,
    "exception_with_syndrome_el",
    0,
    Void,
    [Ptr, I32, I32, I32],
    h_exception_with_syndrome_el
);
def!(EXCEPTION_INTERNAL, "exception_internal", 0, Void, [Ptr, I32], h_exception_internal);
def!(
    EXCEPTION_PC_ALIGNMENT,
    "exception_pc_alignment",
    0,
    Void,
    [Ptr, I64],
    h_exception_pc_alignment
);
def!(WFI, "wfi", 0, Void, [Ptr, I32], h_wfi);
def!(EXCEPTION_RETURN, "exception_return", 0, Void, [Ptr, I64], h_exception_return);
def!(MSR_I_DAIFSET, "msr_i_daifset", 0, Void, [Ptr, I32], h_msr_i_daifset);
def!(MSR_I_DAIFCLEAR, "msr_i_daifclear", 0, Void, [Ptr, I32], h_msr_i_daifclear);
def!(MSR_I_SPSEL, "msr_i_spsel", 0, Void, [Ptr, I32], h_msr_i_spsel);
def!(
    ACCESS_CHECK_CP_REG,
    "access_check_cp_reg",
    0,
    Void,
    [Ptr, I32, I32, I32],
    h_access_check_cp_reg
);
def!(GET_SYSREG, "get_sysreg", 0, I64, [Ptr, I32], h_get_sysreg);
def!(SET_SYSREG, "set_sysreg", 0, Void, [Ptr, I32, I64], h_set_sysreg);
def!(DC_ZVA, "dc_zva", 0, Void, [Ptr, I64], h_dc_zva);
def!(PRE_HVC, "pre_hvc", 0, Void, [Ptr], h_pre_hvc);
def!(PRE_SMC, "pre_smc", 0, Void, [Ptr, I32], h_pre_smc);
def!(CRC32_64, "crc32_64", NO_RWG_SE, I64, [I64, I64, I32], h_crc32_64);
def!(CRC32C_64, "crc32c_64", NO_RWG_SE, I64, [I64, I64, I32], h_crc32c_64);
def!(RBIT64, "rbit64", NO_RWG_SE, I64, [I64], h_rbit64);
def!(SDIV64, "sdiv64", NO_RWG_SE, I64, [I64, I64], h_sdiv64);
def!(REBUILD_HFLAGS, "rebuild_hflags_a64", 0, Void, [Ptr], h_rebuild_hflags);
def!(UDIV64, "udiv64", NO_RWG_SE, I64, [I64, I64], h_udiv64);

/// Every AArch64 helper.
pub(crate) const ALL: &[Def] = &[
    EXCEPTION_WITH_SYNDROME_EL,
    EXCEPTION_INTERNAL,
    EXCEPTION_PC_ALIGNMENT,
    WFI,
    EXCEPTION_RETURN,
    MSR_I_DAIFSET,
    MSR_I_DAIFCLEAR,
    MSR_I_SPSEL,
    ACCESS_CHECK_CP_REG,
    GET_SYSREG,
    SET_SYSREG,
    DC_ZVA,
    PRE_HVC,
    PRE_SMC,
    CRC32_64,
    CRC32C_64,
    RBIT64,
    SDIV64,
    UDIV64,
    REBUILD_HFLAGS,
];

/// Register every helper in `r`.
pub(crate) fn register(r: &mut HelperRegistry) {
    let lists = [
        ALL,
        super::vfp::ALL,
        super::vec_helper::ALL,
        super::crypto::ALL,
        super::sve_helper::ALL,
        super::pauth::ALL,
        super::mte::ALL,
    ];
    for d in lists.iter().copied().flatten() {
        r.register_info(&d.info(), d.f);
    }
}

/// `HELPER(rebuild_hflags_a64)`: after generated code stored a register the TB flags
/// depend on.
fn h_rebuild_hflags(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let ops = cpu.ops();
        let st = CpuArmState::load_system(cpu.env);
        let flags = super::tb_flags(arm_of(&ops).features(), &st);
        cpu.env[HFLAGS..HFLAGS + 4].copy_from_slice(&flags.to_le_bytes());
        Ok(0)
    })
}

/// `HELPER(exception_with_syndrome_el)`.
fn h_exception_with_syndrome_el(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| Err(raise_exception(cpu, a[1] as i32, a[2] as u32, a[3] as u32, Ra::None)))
}

/// `HELPER(exception_internal)`: raise an exception that QEMU handles itself, such as
/// `EXCP_SEMIHOST`, without a syndrome or target EL.
fn h_exception_internal(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| Err(cpu.raise_exception(a[1] as i32, Ra::None)))
}

/// `HELPER(exception_pc_alignment)`.
fn h_exception_pc_alignment(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let mut st = CpuArmState::load(cpu.env);
        let target_el = exception_target_el(&st);
        st.exception_vaddress = a[1];
        super::commit(cpu, &mut st);
        Err(raise_exception(cpu, EXCP_PREFETCH_ABORT, syn_pcalignment(), target_el, Ra::None))
    })
}

/// `HELPER(wfi)`. The PC already points past the WFI.
fn h_wfi(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let insn_len = a[1];
        let mut st = CpuArmState::load(cpu.env);
        let ops = cpu.ops();
        let target_el = check_wfx_trap(arm_of(&ops).features(), &st);
        if cpu.has_work() {
            // Don't bother to go into our "low power state" if we would just wake up
            // immediately.
            return Ok(0);
        }
        if target_el != 0 {
            st.pc = st.pc.wrapping_sub(insn_len);
            super::commit(cpu, &mut st);
            return Err(raise_exception(cpu, EXCP_UDEF, syn_wfx(1, 0xe, 0), target_el, Ra::None));
        }
        cpu.core.exception_index = excp::HLT;
        cpu.shared().halted.store(1, Ordering::Release);
        Err(cpu.cpu_loop_exit())
    })
}

/// `check_wfx_trap()` for WFI: the EL a WFI traps to, or 0 if it does not trap.
fn check_wfx_trap(f: &crate::cpu::ArmFeatures, st: &CpuArmState) -> u32 {
    let cur_el = st.current_el();
    // If we are currently in EL0 then we need to check if SCTLR is set up for WFx
    // instructions being trapped to EL1. These trap bits don't exist in v7 EL1 or in v7 AArch32
    // mode.
    if cur_el < 1 && sysreg::sctlr_el0(f, st) & SCTLR_NTWI == 0 {
        return exception_target_el(st);
    }
    // We are not trapping to EL1; trap to EL2 if HCR_EL2 requires it. No need for ARM_FEATURE
    // check as if HCR_EL2 doesn't exist the bits will be zero.
    if cur_el < 2 && st.hcr_el2_eff(f) & HCR_TWI != 0 {
        return 2;
    }
    // We are not trapping to EL1 or EL2; trap to EL3 if SCR_EL3 requires it.
    if f.el3 && cur_el < 3 && st.scr_el3 & SCR_TWI != 0 {
        return 3;
    }
    0
}

/// `el_from_spsr()` for a return to AArch64; AArch32 is not implemented, so a return to it
/// is illegal too.
fn el_from_spsr(spsr: u64) -> Option<u32> {
    if spsr & u64::from(PSTATE_NRW) != 0 {
        return None;
    }
    if (spsr >> 1) & 1 != 0 {
        // Return with reserved M[1] bit set.
        return None;
    }
    if spsr & 0xf == 1 {
        // Return to EL0 with M[0] bit set.
        return None;
    }
    Some(((spsr >> 2) & 3) as u32)
}

/// `aarch64_pstate_valid_mask()` for the features of the model.
fn pstate_valid_mask(f: &crate::cpu::ArmFeatures) -> u32 {
    let mut valid = PSTATE_M_ALL | PSTATE_DAIF | PSTATE_IL | PSTATE_SS | PSTATE_NZCV;
    if f.pan {
        valid |= PSTATE_PAN;
    }
    if f.uao {
        valid |= PSTATE_UAO;
    }
    if f.mte >= 2 {
        valid |= crate::cpu::PSTATE_TCO;
    }
    valid
}

/// `PSTATE_M`, the mode bits kept by an exception return.
const PSTATE_M_ALL: u32 = 0xf;

/// `HELPER(exception_return)`.
fn h_exception_return(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let ops = cpu.ops();
        let feat = *arm_of(&ops).features();
        let mut new_pc = a[1];
        let mut st = CpuArmState::load(cpu.env);
        let cur_el = st.current_el();
        let spsr = st.spsr_el[cur_el as usize];

        st.save_sp(cur_el);
        // arm_clear_exclusive().
        st.exclusive_addr = u64::MAX;

        let hcr = st.hcr_el2_eff(&feat);
        let legal = |el: u32| {
            // No AArch32, so the register width check of QEMU is the NRW check in
            // el_from_spsr(). A return to EL2 needs EL2 enabled, and a return to EL1 is
            // illegal while HCR_EL2.TGE is set.
            el <= cur_el
                && !(el == 2 && !st.is_el2_enabled(&feat))
                && !(el == 1 && hcr & HCR_TGE != 0)
        };
        let target = el_from_spsr(spsr).filter(|&el| legal(el));
        match target {
            Some(new_el) => {
                let spsr = spsr as u32 & pstate_valid_mask(&feat);
                st.pstate_write(spsr);
                // Single step is never active, so PSTATE.SS is always cleared.
                st.pstate &= !PSTATE_SS;
                st.restore_sp(new_el);
                // Apply TBI to the exception return address, using the TBII bits of the EL
                // being returned to.
                let tbii = (super::tb_flags(&feat, &st) >> super::TB_TBII_SHIFT) & 3;
                if (tbii >> ((new_pc >> 55) & 1)) & 1 != 0 {
                    // TBI is enabled.
                    if super::regime_has_2_ranges(st.mmu_idx(&feat)) {
                        new_pc = ((new_pc << 8) as i64 >> 8) as u64;
                    } else {
                        new_pc &= (1 << 56) - 1;
                    }
                }
                st.pc = new_pc;
                super::sve_change_el(&feat, &mut st, cur_el, new_el);
            }
            None => {
                // Illegal return events of various kinds have architecturally mandated
                // behaviour: restore NZCV and DAIF from SPSR_ELx, set PSTATE.IL, restore PC
                // from ELR_ELx, and no change to exception level, execution state or stack
                // pointer.
                st.pstate |= PSTATE_IL;
                st.pc = new_pc;
                let spsr = (spsr as u32 & (PSTATE_NZCV | PSTATE_DAIF))
                    | (st.pstate_read() & !(PSTATE_NZCV | PSTATE_DAIF));
                st.pstate_write(spsr);
                st.pstate &= !PSTATE_SS;
                st.restore_sp(cur_el);
            }
        }
        super::commit(cpu, &mut st);
        if target.is_some() {
            // arm_call_el_change_hook().
            arm_of(&ops).gic_el_change(cpu.core.shared().cpu_index, &st);
        }
        Ok(0)
    })
}

/// `daif_check()`: a DAIF update from EL0 is allowed only if SCTLR.UMA is set.
fn daif_check(cpu: &mut Cpu<'_>, op: u32, imm: u32) -> R<()> {
    let st = CpuArmState::load(cpu.env);
    let ops = cpu.ops();
    if st.current_el() == 0 && sysreg::sctlr_el0(arm_of(&ops).features(), &st) & SCTLR_UMA == 0 {
        let syn = syn_aa64_sysregtrap(0, op & 7, (op >> 3) & 7, 4, imm, 0x1f, false);
        let target_el = exception_target_el(&st);
        return Err(raise_exception(cpu, EXCP_UDEF, syn, target_el, Ra::None));
    }
    Ok(())
}

/// `HELPER(msr_i_daifset)`.
fn h_msr_i_daifset(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let imm = a[1] as u32;
        daif_check(cpu, 0x1e, imm)?;
        let mut st = CpuArmState::load(cpu.env);
        st.daif |= (imm << 6) & PSTATE_DAIF;
        super::commit(cpu, &mut st);
        Ok(0)
    })
}

/// `HELPER(msr_i_daifclear)`.
fn h_msr_i_daifclear(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let imm = a[1] as u32;
        daif_check(cpu, 0x1f, imm)?;
        let mut st = CpuArmState::load(cpu.env);
        st.daif &= !((imm << 6) & PSTATE_DAIF);
        super::commit(cpu, &mut st);
        Ok(0)
    })
}

/// `HELPER(msr_i_spsel)`.
fn h_msr_i_spsel(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let mut st = CpuArmState::load(cpu.env);
        st.update_spsel(a[1] as u32);
        super::commit(cpu, &mut st);
        Ok(0)
    })
}

/// `HELPER(access_check_cp_reg)`: the run time check of a register with an access function.
fn h_access_check_cp_reg(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let ops = cpu.ops();
        let feat = arm_of(&ops).features();
        let key = a[1] as u32;
        let syndrome = a[2] as u32;
        let isread = a[3] != 0;
        let ri = sysreg::lookup(key, feat).expect("the translator checked the register");
        let st = CpuArmState::load(cpu.env);
        let access = if ri.trap == sysreg::Trap::Gic {
            let index = cpu.core.shared().cpu_index;
            match arm_of(&ops).gic_access(index, key, &st, isread) {
                GicAccess::Ok => sysreg::Access::Ok,
                GicAccess::TrapEl1 => sysreg::Access::TrapEl1,
                GicAccess::TrapEl2 => sysreg::Access::TrapEl2,
                GicAccess::TrapEl3 => sysreg::Access::TrapEl3,
                GicAccess::Undefined => sysreg::Access::Undefined,
            }
        } else {
            ri.trap.check(feat, &st, isread)
        };
        let (syn, target_el) = match access {
            sysreg::Access::Ok => return Ok(0),
            sysreg::Access::TrapEl1 => (syndrome, 1),
            sysreg::Access::TrapEl2 => (syndrome, 2),
            sysreg::Access::TrapEl3 => (syndrome, 3),
            sysreg::Access::Undefined => (syn_uncategorized(), exception_target_el(&st)),
        };
        Err(raise_exception(cpu, EXCP_UDEF, syn, target_el, Ra::None))
    })
}

/// Read a system register that has no plain storage.
fn h_get_sysreg(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| Ok(sysreg::read(cpu, a[1] as u32)))
}

/// Write a system register that has no plain storage, or run a system operation.
fn h_set_sysreg(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        sysreg::write(cpu, a[1] as u32, a[2]);
        Ok(0)
    })
}

/// `HELPER(dc_zva)`: zero the block of `4 << DCZID_EL0.BS` bytes holding the address.
/// QEMU writes the block through a host pointer when it can; this port always stores it
/// eight bytes at a time through the softmmu.
fn h_dc_zva(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let ops = cpu.ops();
        let bs = arm_of(&ops).model().dczid & 0xf;
        let blocklen = 4u64 << bs;
        let vaddr_in = a[1];
        let vaddr = vaddr_in & !(blocklen - 1);
        let mmu_idx = CpuArmState::load(cpu.env).mmu_idx(arm_of(&ops).features()) as u32;
        // Fault on the original address first, as QEMU's probe_write() of it does, so that
        // FAR reports it.
        cpu_st_mmu(cpu, vaddr_in, 0, MemOpIdx::new(MemOp::UB, mmu_idx), Ra::Tb)?;
        let oi = MemOpIdx::new(MemOp::LEUQ, mmu_idx);
        for i in (0..blocklen).step_by(8) {
            cpu_st_mmu(cpu, vaddr + i, 0, oi, Ra::Tb)?;
        }
        Ok(0)
    })
}

/// `HELPER(pre_hvc)`: the checks before an HVC is taken as an exception.
fn h_pre_hvc(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let ops = cpu.ops();
        let arm = arm_of(&ops);
        let f = arm.features();
        if psci::is_psci_call(arm, EXCP_HVC) {
            // If PSCI is enabled and this looks like a valid PSCI call then that overrides
            // the architecturally mandated HVC behaviour.
            return Ok(0);
        }
        let st = CpuArmState::load(cpu.env);
        let mut undef = if !f.el2 {
            // If EL2 doesn't exist, HVC always UNDEFs.
            true
        } else if f.el3 {
            // EL3.HCE has priority over EL2.HCD.
            st.scr_el3 & SCR_HCE == 0
        } else {
            st.hcr_el2 & HCR_HCD != 0
        };
        // HVC is UNDEFINED at Secure EL1 (there is no Secure EL2 here). We've already trapped
        // HVC from EL0 at translation time.
        if st.is_secure_below_el3(f) && st.current_el() == 1 {
            undef = true;
        }
        if undef {
            let target_el = exception_target_el(&st);
            return Err(raise_exception(cpu, EXCP_UDEF, syn_uncategorized(), target_el, Ra::None));
        }
        Ok(0)
    })
}

/// `HELPER(pre_smc)`: the checks before an SMC is taken as an exception.
fn h_pre_smc(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let ops = cpu.ops();
        let arm = arm_of(&ops);
        let f = arm.features();
        let syndrome = a[1] as u32;
        let st = CpuArmState::load(cpu.env);
        let cur_el = st.current_el();
        let hcr = st.hcr_el2_eff(f);
        let smd = st.scr_el3 & SCR_SMD != 0;
        let undef = |cpu: &mut Cpu<'_>| {
            let target_el = exception_target_el(&st);
            raise_exception(cpu, EXCP_UDEF, syn_uncategorized(), target_el, Ra::None)
        };
        if !f.el3 && hcr & HCR_NV == 0 && arm.psci_conduit() != super::PsciConduit::Smc {
            // If we have no EL3 then traditionally SMC always UNDEFs and can't be trapped to
            // EL2. PSCI-via-SMC is a sort of ersatz EL3 firmware, and we want an EL2 guest
            // to be able to forbid its EL1 from making PSCI calls via HCR.TSC, so for these
            // purposes treat PSCI-via-SMC as implying an EL3.
            return Err(undef(cpu));
        }
        if cur_el == 1 && hcr & HCR_TSC != 0 {
            // In NS EL1, HCR controlled routing to EL2 has priority over SMD.
            return Err(raise_exception(cpu, EXCP_HYP_TRAP, syndrome, 2, Ra::None));
        }
        // If PSCI is enabled and this looks like a valid PSCI call then suppress the UNDEF
        // that a set SCR.SMD or a missing EL3 would cause.
        if !psci::is_psci_call(arm, EXCP_SMC) && (smd || !f.el3) {
            return Err(undef(cpu));
        }
        Ok(0)
    })
}

/// A raw reflected CRC over the low `bytes` bytes of `val`, little endian first, starting from
/// `acc`, with no inversion of the input or output; this is what the CRC32 instructions do.
fn crc_update(poly: u32, acc: u32, val: u64, bytes: u32) -> u32 {
    let mut crc = acc;
    for i in 0..bytes {
        crc ^= u32::from((val >> (8 * i)) as u8);
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ poly } else { crc >> 1 };
        }
    }
    crc
}

/// `HELPER(crc32_64)`.
fn h_crc32_64(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok(u128::from(crc_update(0xedb8_8320, a[0] as u32, a[1], a[2] as u32)))
}

/// `HELPER(crc32c_64)`.
fn h_crc32c_64(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok(u128::from(crc_update(0x82f6_3b78, a[0] as u32, a[1], a[2] as u32)))
}

/// `HELPER(rbit64)`.
fn h_rbit64(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok(u128::from(a[0].reverse_bits()))
}

/// `HELPER(sdiv64)`: division by zero gives zero, and the one overflowing case gives
/// `INT64_MIN`.
fn h_sdiv64(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    let (num, den) = (a[0] as i64, a[1] as i64);
    let q = if den == 0 { 0 } else { num.wrapping_div(den) };
    Ok(u128::from(q as u64))
}

/// `HELPER(udiv64)`: division by zero gives zero.
fn h_udiv64(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok(u128::from(a[0].checked_div(a[1]).unwrap_or(0)))
}

#[cfg(test)]
mod tests {
    use super::crc_update;

    #[test]
    fn crc32_check_values() {
        // The standard check values: CRC-32 and CRC-32C of "123456789", with the usual
        // inversion done outside the raw update as the guest does it.
        let data = b"123456789";
        let crc = |poly| {
            let mut c = 0xffff_ffffu32;
            for &b in data {
                c = crc_update(poly, c, u64::from(b), 1);
            }
            !c
        };
        assert_eq!(crc(0xedb8_8320), 0xcbf4_3926);
        assert_eq!(crc(0x82f6_3b78), 0xe306_9283);
    }
}
