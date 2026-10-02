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

use super::{arm_of, exception_target_el, sysreg};
use crate::cpu::{
    CpuArmState, EXCP_PREFETCH_ABORT, EXCP_UDEF, PSTATE_DAIF, PSTATE_IL, PSTATE_NRW, PSTATE_NZCV,
    PSTATE_PAN, PSTATE_SS, PSTATE_UAO, SCTLR_NTWI, SCTLR_UMA,
};
use crate::syndrome::{syn_aa64_sysregtrap, syn_pcalignment, syn_uncategorized, syn_wfx};

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
fn run(h: &mut HelperEnv<'_>, f: impl FnOnce(&mut Cpu<'_>) -> R<u64>) -> Result<u128, Unwind> {
    let mut cpu = Cpu::from_helper_env(h).expect("arm helpers run under the runtime");
    match f(&mut cpu) {
        Ok(v) => Ok(u128::from(v)),
        Err(e) => Err(cpu.unwind(e)),
    }
}

/// `raise_exception()`: take `excp` with `syndrome` to `target_el`.
pub(crate) fn raise_exception(
    cpu: &mut Cpu<'_>,
    excp: i32,
    syndrome: u32,
    target_el: u32,
    ra: Ra,
) -> CpuLoopExit {
    let mut st = CpuArmState::load(cpu.env);
    st.exception_syndrome = syndrome;
    st.exception_target_el = target_el;
    st.store(cpu.env);
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
def!(CRC32_64, "crc32_64", NO_RWG_SE, I64, [I64, I64, I32], h_crc32_64);
def!(CRC32C_64, "crc32c_64", NO_RWG_SE, I64, [I64, I64, I32], h_crc32c_64);
def!(RBIT64, "rbit64", NO_RWG_SE, I64, [I64], h_rbit64);
def!(SDIV64, "sdiv64", NO_RWG_SE, I64, [I64, I64], h_sdiv64);
def!(UDIV64, "udiv64", NO_RWG_SE, I64, [I64, I64], h_udiv64);

/// Every AArch64 helper.
pub(crate) const ALL: &[Def] = &[
    EXCEPTION_WITH_SYNDROME_EL,
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
    CRC32_64,
    CRC32C_64,
    RBIT64,
    SDIV64,
    UDIV64,
];

/// Register every helper in `r`.
pub(crate) fn register(r: &mut HelperRegistry) {
    let lists = [ALL, super::vfp::ALL, super::vec_helper::ALL, super::crypto::ALL];
    for d in lists.iter().copied().flatten() {
        r.register_info(&d.info(), d.f);
    }
}

/// `HELPER(exception_with_syndrome_el)`.
fn h_exception_with_syndrome_el(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| Err(raise_exception(cpu, a[1] as i32, a[2] as u32, a[3] as u32, Ra::None)))
}

/// `HELPER(exception_pc_alignment)`.
fn h_exception_pc_alignment(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let mut st = CpuArmState::load(cpu.env);
        let target_el = exception_target_el(&st);
        st.exception_vaddress = a[1];
        st.store(cpu.env);
        Err(raise_exception(cpu, EXCP_PREFETCH_ABORT, syn_pcalignment(), target_el, Ra::None))
    })
}

/// `HELPER(wfi)`. The PC already points past the WFI.
fn h_wfi(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let insn_len = a[1];
        let mut st = CpuArmState::load(cpu.env);
        // check_wfx_trap(): only the EL0 trap controlled by SCTLR_EL1.nTWI exists here.
        let trap = st.current_el() == 0 && st.sctlr_el[1] & SCTLR_NTWI == 0;
        if cpu.has_work() {
            // Don't bother to go into our "low power state" if we would just wake up
            // immediately.
            return Ok(0);
        }
        if trap {
            st.pc = st.pc.wrapping_sub(insn_len);
            st.store(cpu.env);
            let target_el = exception_target_el(&st);
            return Err(raise_exception(cpu, EXCP_UDEF, syn_wfx(1, 0xe, 0), target_el, Ra::None));
        }
        cpu.core.exception_index = excp::HLT;
        cpu.shared().halted.store(1, Ordering::Release);
        Err(cpu.cpu_loop_exit())
    })
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
fn pstate_valid_mask(pan: bool, uao: bool) -> u32 {
    let mut valid = PSTATE_M_ALL | PSTATE_DAIF | PSTATE_IL | PSTATE_SS | PSTATE_NZCV;
    if pan {
        valid |= PSTATE_PAN;
    }
    if uao {
        valid |= PSTATE_UAO;
    }
    valid
}

/// `PSTATE_M`, the mode bits kept by an exception return.
const PSTATE_M_ALL: u32 = 0xf;

/// `HELPER(exception_return)`.
fn h_exception_return(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let ops = cpu.ops();
        let feat = arm_of(&ops).model().features;
        let mut new_pc = a[1];
        let mut st = CpuArmState::load(cpu.env);
        let cur_el = st.current_el();
        let spsr = st.spsr_el[cur_el as usize];

        st.save_sp(cur_el);
        // arm_clear_exclusive().
        st.exclusive_addr = u64::MAX;

        match el_from_spsr(spsr).filter(|&el| el <= cur_el) {
            Some(new_el) => {
                let spsr = spsr as u32 & pstate_valid_mask(feat.pan, feat.uao);
                st.pstate_write(spsr);
                // Single step is never active, so PSTATE.SS is always cleared.
                st.pstate &= !PSTATE_SS;
                st.restore_sp(new_el);
                // Apply TBI to the exception return address, using the TBII bits of the EL
                // being returned to.
                let tbii = (super::tb_flags(&st) >> super::TB_TBII_SHIFT) & 3;
                if (tbii >> ((new_pc >> 55) & 1)) & 1 != 0 {
                    // The EL1&0 regime has two ranges.
                    new_pc = ((new_pc << 8) as i64 >> 8) as u64;
                }
                st.pc = new_pc;
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
        st.store(cpu.env);
        Ok(0)
    })
}

/// `daif_check()`: a DAIF update from EL0 is allowed only if SCTLR_EL1.UMA is set.
fn daif_check(cpu: &mut Cpu<'_>, op: u32, imm: u32) -> R<()> {
    let st = CpuArmState::load(cpu.env);
    if st.current_el() == 0 && st.sctlr_el[1] & SCTLR_UMA == 0 {
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
        st.store(cpu.env);
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
        st.store(cpu.env);
        Ok(0)
    })
}

/// `HELPER(msr_i_spsel)`.
fn h_msr_i_spsel(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let mut st = CpuArmState::load(cpu.env);
        st.update_spsel(a[1] as u32);
        st.store(cpu.env);
        Ok(0)
    })
}

/// `HELPER(access_check_cp_reg)`: the run time check of a register with an access function.
fn h_access_check_cp_reg(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let ops = cpu.ops();
        let feat = arm_of(&ops).model().features;
        let key = a[1] as u32;
        let syndrome = a[2] as u32;
        let isread = a[3] != 0;
        let ri = sysreg::lookup(key, &feat).expect("the translator checked the register");
        let st = CpuArmState::load(cpu.env);
        // The checks of this slice do not depend on the direction; the argument is kept to
        // match QEMU's helper.
        let _ = isread;
        let syn = match ri.trap.check(&st) {
            sysreg::Access::Ok => return Ok(0),
            sysreg::Access::TrapEl1 => syndrome,
            sysreg::Access::Undefined => syn_uncategorized(),
        };
        let target_el = exception_target_el(&st);
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
        let mmu_idx = CpuArmState::load(cpu.env).mmu_idx() as u32;
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
