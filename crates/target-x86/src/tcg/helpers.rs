// SPDX-License-Identifier: GPL-2.0-or-later

//! The helpers called from generated code: the port of `int_helper.c`, `misc_helper.c`,
//! `bpt_helper.c` and `cc_helper.c`, plus thin wrappers around the functions of `seg.rs`.
//!
//! Every helper is described by a [`Def`] with the name, flags and signature the translator
//! declares, so the declaration and the registration cannot drift apart. Helpers that touch
//! the CPU take `env` as their first argument, so their own arguments start at `args[1]`.
//! A helper that raises a guest exception records it on the vCPU with [`Cpu::unwind`] and
//! returns the [`Unwind`] that makes the backend leave the block.

use std::sync::atomic::Ordering;

use ruvm_jit::cputlb::{tlb_flush, tlb_flush_page};
use ruvm_jit::{Cpu, CpuLoopExit, Ra, excp};
use ruvm_jit_core::types::call_flags::NO_RWG_SE;
use ruvm_jit_core::{HelperInfo, HelperType, MemOp};
use ruvm_jit_interp::{HelperEnv, HelperRegistry, Unwind};
use ruvm_mem::{Endian, MemTxAttrs};

use super::cc::{CC_OP_EFLAGS, cc_op_has_eflags, cc_op_size, compute_all, compute_c, parity};
use super::env::{
    APIC_BASE, CC_A, CC_C, CC_O, CC_OP, CC_SRC, CC_Z, CR8, CSTAR, EFER, EFLAGS, EIP, FMASK,
    HF_AVX_EN_MASK, HF_INHIBIT_IRQ_MASK, HF_OSFXSR_MASK, HF_SMAP_MASK, HF_UMIP_MASK, HFLAGS,
    KERNELGSBASE, LSTAR, MISC_ENABLE, PAT, RF_MASK, SEG_BASE, STAR, SYSENTER_CS, SYSENTER_EIP,
    SYSENTER_ESP, TF_MASK, TSC_AUX, TSC_OFFSET, XCR0, avx_enabled, cc_compute_all, compute_eflags,
    cr, dr, ld32, ld64, load_eflags, reg, seg, st32, st64,
};
use super::seg::{self as sh, raise_exception_err_ra, raise_exception_ra, raise_interrupt2};
use super::{
    EXCP00_DIVZ, EXCP0D_GPF, EXCP01_DB, EXCP04_INTO, EXCP05_BOUND, EXCP06_ILLOP, mmu_index_pl,
    x86_of,
};
use crate::msr::{
    MSR_CSTAR, MSR_EFER, MSR_FMASK, MSR_FSBASE, MSR_GSBASE, MSR_IA32_APICBASE,
    MSR_IA32_MISC_ENABLE, MSR_IA32_SYSENTER_CS, MSR_IA32_SYSENTER_EIP, MSR_IA32_SYSENTER_ESP,
    MSR_IA32_TSC, MSR_KERNELGSBASE, MSR_LSTAR, MSR_PAT, MSR_STAR, MSR_TSC_AUX,
};
use crate::state::{
    CR0_ET_MASK, CR0_PE_MASK, CR0_PG_MASK, CR0_TS_MASK, CR0_WP_MASK, CR4_CET_MASK, CR4_DE_MASK,
    CR4_FRED_MASK, CR4_FSGSBASE_MASK, CR4_LA57_MASK, CR4_LAM_SUP_MASK, CR4_MCE_MASK,
    CR4_OSFXSR_MASK, CR4_OSXMMEXCPT_MASK, CR4_OSXSAVE_MASK, CR4_PAE_MASK, CR4_PCE_MASK,
    CR4_PCIDE_MASK, CR4_PGE_MASK, CR4_PKE_MASK, CR4_PKS_MASK, CR4_PSE_MASK, CR4_PVI_MASK,
    CR4_SMAP_MASK, CR4_SMEP_MASK, CR4_TSD_MASK, CR4_UMIP_MASK, CR4_VME_MASK, DR6_FIXED_1,
    DR7_FIXED_1, HF_ADDSEG_MASK, HF_CPL_MASK, HF_CS64_MASK, HF_EM_MASK, HF_LMA_MASK, HF_MP_MASK,
    HF_PE_MASK, HF_TS_MASK, MSR_EFER_FFXSR, MSR_EFER_LMA, MSR_EFER_LME, MSR_EFER_NXE, MSR_EFER_SCE,
    MSR_EFER_SVME, R_EAX, R_EBX, R_ECX, R_EDX, R_FS, R_GS,
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
    f: ruvm_jit_interp::HelperFn,
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

pub(crate) mod fpu;
pub(crate) mod vec;

/// Run `f` on the vCPU behind `h`, turning a guest exception into an [`Unwind`].
fn run(h: &mut HelperEnv<'_>, f: impl FnOnce(&mut Cpu<'_>) -> R<u64>) -> Result<u128, Unwind> {
    let mut cpu = Cpu::from_helper_env(h).expect("x86 helpers run under the runtime");
    match f(&mut cpu) {
        Ok(v) => Ok(u128::from(v)),
        Err(e) => Err(cpu.unwind(e)),
    }
}

fn a32(args: &[u64], i: usize) -> u32 {
    args[i] as u32
}

fn regv(cpu: &Cpu<'_>, r: usize) -> u64 {
    ld64(cpu.env, reg(r))
}

fn set_reg(cpu: &mut Cpu<'_>, r: usize, v: u64) {
    st64(cpu.env, reg(r), v);
}

fn hflags(cpu: &Cpu<'_>) -> u32 {
    ld32(cpu.env, HFLAGS)
}

fn set_hflags(cpu: &mut Cpu<'_>, v: u32) {
    st32(cpu.env, HFLAGS, v);
}

fn cpl(cpu: &Cpu<'_>) -> u32 {
    hflags(cpu) & HF_CPL_MASK
}

const TB: Ra = Ra::Tb;

// Lazy flags.

def!(CC_COMPUTE_ALL, "cc_compute_all", NO_RWG_SE, I64, [I64, I64, I64, I32], h_cc_compute_all);
def!(CC_COMPUTE_C, "cc_compute_c", NO_RWG_SE, I64, [I64, I64, I64, I32], h_cc_compute_c);

fn h_cc_compute_all(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok(u128::from(compute_all(a[0], a[1], a[2], a32(a, 3))))
}

fn h_cc_compute_c(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok(u128::from(compute_c(a[0], a[1], a[2], a32(a, 3))))
}

def!(CC_COMPUTE_NZ, "cc_compute_nz", NO_RWG_SE, I64, [I64, I64, I32], h_cc_compute_nz);

/// `helper_cc_compute_nz()`: a value that is zero exactly when ZF is set.
fn h_cc_compute_nz(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    let op = a32(a, 2);
    let v = if cc_op_has_eflags(op) {
        !a[1] & u64::from(CC_Z)
    } else {
        let bits = 8u32 << cc_op_size(op);
        let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
        a[0] & mask
    };
    Ok(u128::from(v))
}

// Exceptions.

def!(RAISE_EXCEPTION, "x86_raise_exception", 0, Void, [Ptr, I32], h_raise_exception);
def!(
    RAISE_EXCEPTION_ERR,
    "x86_raise_exception_err",
    0,
    Void,
    [Ptr, I32, I32],
    h_raise_exception_err
);
def!(RAISE_INTERRUPT, "x86_raise_interrupt", 0, Void, [Ptr, I32, I32], h_raise_interrupt);
def!(INTO, "x86_into", 0, Void, [Ptr, I32], h_into);
def!(SINGLE_STEP, "x86_single_step", 0, Void, [Ptr], h_single_step);
def!(HLT, "x86_hlt", 0, Void, [Ptr], h_hlt);
def!(ICEBP, "x86_icebp", 0, Void, [Ptr], h_icebp);
def!(
    RECHECKING_SINGLE_STEP,
    "x86_rechecking_single_step",
    0,
    Void,
    [Ptr],
    h_rechecking_single_step
);

/// `raise_exception(env, intno)`, with EIP restored to the faulting instruction.
fn h_raise_exception(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| Err(raise_exception_ra(cpu, a32(a, 1) as i32, TB)))
}

fn h_raise_exception_err(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| Err(raise_exception_err_ra(cpu, a32(a, 1) as i32, a32(a, 2), TB)))
}

/// `helper_raise_interrupt()`: INT n. EIP holds the instruction's address.
fn h_raise_interrupt(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        Err(raise_interrupt2(cpu, a32(a, 1) as i32, true, 0, u64::from(a32(a, 2)), Ra::None))
    })
}

/// `helper_into()`.
fn h_into(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        if cc_compute_all(cpu.env) & CC_O != 0 {
            return Err(raise_interrupt2(
                cpu,
                EXCP04_INTO,
                true,
                0,
                u64::from(a32(a, 1)),
                Ra::None,
            ));
        }
        Ok(0)
    })
}

/// `helper_single_step()`: the #DB after an instruction run with TF set.
fn h_single_step(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        const DR6_BS: u64 = 1 << 14;
        let d6 = ld64(cpu.env, dr(6)) | DR6_BS;
        st64(cpu.env, dr(6), d6);
        Err(raise_exception_ra(cpu, EXCP01_DB, Ra::None))
    })
}

/// `helper_rechecking_single_step()`: the #DB after an instruction that may have changed TF.
fn h_rechecking_single_step(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    if ld64(h.env, EFLAGS) & u64::from(TF_MASK) != 0 {
        return h_single_step(h, a);
    }
    Ok(0)
}

/// `helper_hlt()`. EIP already points past the instruction.
fn h_hlt(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        do_end_instruction(cpu);
        cpu.shared().halted.store(1, Ordering::Release);
        cpu.core.exception_index = excp::HLT;
        Err(cpu.cpu_loop_exit())
    })
}

/// `helper_icebp()`: INT1. EIP already points past the instruction.
fn h_icebp(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        do_end_instruction(cpu);
        Err(raise_interrupt2(cpu, EXCP01_DB, false, 0, 0, Ra::None))
    })
}

/// `do_end_instruction()`.
fn do_end_instruction(cpu: &mut Cpu<'_>) {
    let hf = hflags(cpu) & !HF_INHIBIT_IRQ_MASK;
    set_hflags(cpu, hf);
    let fl = ld64(cpu.env, EFLAGS) & !u64::from(RF_MASK);
    st64(cpu.env, EFLAGS, fl);
}

// Division.

def!(DIVB, "x86_divb_AL", 0, Void, [Ptr, I64], h_divb);
def!(IDIVB, "x86_idivb_AL", 0, Void, [Ptr, I64], h_idivb);
def!(DIVW, "x86_divw_AX", 0, Void, [Ptr, I64], h_divw);
def!(IDIVW, "x86_idivw_AX", 0, Void, [Ptr, I64], h_idivw);
def!(DIVL, "x86_divl_EAX", 0, Void, [Ptr, I64], h_divl);
def!(IDIVL, "x86_idivl_EAX", 0, Void, [Ptr, I64], h_idivl);
def!(DIVQ, "x86_divq_EAX", 0, Void, [Ptr, I64], h_divq);
def!(IDIVQ, "x86_idivq_EAX", 0, Void, [Ptr, I64], h_idivq);

fn de<T>(cpu: &mut Cpu<'_>) -> R<T> {
    Err(raise_exception_ra(cpu, EXCP00_DIVZ, TB))
}

fn h_divb(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let eax = regv(cpu, R_EAX);
        let num = eax & 0xffff;
        let den = a[1] & 0xff;
        if den == 0 {
            return de(cpu);
        }
        let q = num / den;
        if q > 0xff {
            return de(cpu);
        }
        let r = num % den;
        set_reg(cpu, R_EAX, (eax & !0xffff) | (r << 8) | q);
        Ok(0)
    })
}

fn h_idivb(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let eax = regv(cpu, R_EAX);
        let num = i32::from(eax as i16);
        let den = i32::from(a[1] as i8);
        if den == 0 {
            return de(cpu);
        }
        let q = num / den;
        if q != i32::from(q as i8) {
            return de(cpu);
        }
        let r = num % den;
        let v = u64::from(q as u8) | (u64::from(r as u8) << 8);
        set_reg(cpu, R_EAX, (eax & !0xffff) | v);
        Ok(0)
    })
}

fn h_divw(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let eax = regv(cpu, R_EAX);
        let edx = regv(cpu, R_EDX);
        let num = (eax & 0xffff) | ((edx & 0xffff) << 16);
        let den = a[1] & 0xffff;
        if den == 0 {
            return de(cpu);
        }
        let q = num / den;
        if q > 0xffff {
            return de(cpu);
        }
        let r = num % den;
        set_reg(cpu, R_EAX, (eax & !0xffff) | q);
        set_reg(cpu, R_EDX, (edx & !0xffff) | r);
        Ok(0)
    })
}

fn h_idivw(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let eax = regv(cpu, R_EAX);
        let edx = regv(cpu, R_EDX);
        let num = ((eax & 0xffff) | ((edx & 0xffff) << 16)) as u32 as i32 as i64;
        let den = i64::from(a[1] as i16);
        if den == 0 {
            return de(cpu);
        }
        let q = num / den;
        if q != i64::from(q as i16) {
            return de(cpu);
        }
        let r = num % den;
        set_reg(cpu, R_EAX, (eax & !0xffff) | u64::from(q as u16));
        set_reg(cpu, R_EDX, (edx & !0xffff) | u64::from(r as u16));
        Ok(0)
    })
}

fn h_divl(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let num = (regv(cpu, R_EAX) & 0xffff_ffff) | (regv(cpu, R_EDX) << 32);
        let den = a[1] & 0xffff_ffff;
        if den == 0 {
            return de(cpu);
        }
        let q = num / den;
        if q > 0xffff_ffff {
            return de(cpu);
        }
        let r = num % den;
        set_reg(cpu, R_EAX, q);
        set_reg(cpu, R_EDX, r);
        Ok(0)
    })
}

fn h_idivl(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let num = ((regv(cpu, R_EAX) & 0xffff_ffff) | (regv(cpu, R_EDX) << 32)) as i64;
        let den = i64::from(a[1] as i32);
        if den == 0 {
            return de(cpu);
        }
        // i64::MIN / -1 overflows i64 and the i32 check alike.
        let (q, r) = match (num.checked_div(den), num.checked_rem(den)) {
            (Some(q), Some(r)) => (q, r),
            _ => return de(cpu),
        };
        if q != i64::from(q as i32) {
            return de(cpu);
        }
        set_reg(cpu, R_EAX, u64::from(q as u32));
        set_reg(cpu, R_EDX, u64::from(r as u32));
        Ok(0)
    })
}

fn h_divq(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let num = u128::from(regv(cpu, R_EAX)) | (u128::from(regv(cpu, R_EDX)) << 64);
        let den = u128::from(a[1]);
        if den == 0 {
            return de(cpu);
        }
        let q = num / den;
        if q > u128::from(u64::MAX) {
            return de(cpu);
        }
        let r = num % den;
        set_reg(cpu, R_EAX, q as u64);
        set_reg(cpu, R_EDX, r as u64);
        Ok(0)
    })
}

fn h_idivq(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let num = (u128::from(regv(cpu, R_EAX)) | (u128::from(regv(cpu, R_EDX)) << 64)) as i128;
        let den = i128::from(a[1] as i64);
        if den == 0 {
            return de(cpu);
        }
        let (q, r) = match (num.checked_div(den), num.checked_rem(den)) {
            (Some(q), Some(r)) => (q, r),
            _ => return de(cpu),
        };
        if q != i128::from(q as i64) {
            return de(cpu);
        }
        set_reg(cpu, R_EAX, q as u64);
        set_reg(cpu, R_EDX, r as u64);
        Ok(0)
    })
}

// BCD.

def!(AAA, "x86_aaa", 0, Void, [Ptr], h_aaa);
def!(AAS, "x86_aas", 0, Void, [Ptr], h_aas);
def!(DAA, "x86_daa", 0, Void, [Ptr], h_daa);
def!(DAS, "x86_das", 0, Void, [Ptr], h_das);
def!(AAM, "x86_aam", NO_RWG_SE, I64, [I64, I64], h_aam);
def!(AAD, "x86_aad", NO_RWG_SE, I64, [I64, I64], h_aad);

/// `helper_aam()`: AL divided by `base`, the quotient in AH and the remainder in AL. The
/// translator raises #DE for a zero base before calling it.
fn h_aam(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    let al = a[0] & 0xff;
    let base = a[1];
    let ah = al / base;
    let al = al % base;
    Ok(u128::from(al | (ah << 8)))
}

/// `helper_aad()`: `AH * base + AL`, cut to a byte.
fn h_aad(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    let al = a[0] & 0xff;
    let ah = (a[0] >> 8) & 0xff;
    Ok(u128::from(ah.wrapping_mul(a[1]).wrapping_add(al) & 0xff))
}

fn set_flags_eflags(cpu: &mut Cpu<'_>, fl: u32) {
    st64(cpu.env, CC_SRC, u64::from(fl));
    st32(cpu.env, CC_OP, CC_OP_EFLAGS);
}

fn h_aaa(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let mut fl = cc_compute_all(cpu.env);
        let eax = regv(cpu, R_EAX);
        let mut al = (eax & 0xff) as u32;
        let mut ah = ((eax >> 8) & 0xff) as u32;
        let icarry = u32::from(al > 0xf9);
        if (al & 0x0f) > 9 || fl & CC_A != 0 {
            al = (al + 6) & 0x0f;
            ah = (ah + 1 + icarry) & 0xff;
            fl |= CC_C | CC_A;
        } else {
            fl &= !(CC_C | CC_A);
            al &= 0x0f;
        }
        set_reg(cpu, R_EAX, (eax & !0xffff) | u64::from(al | (ah << 8)));
        set_flags_eflags(cpu, fl);
        Ok(0)
    })
}

fn h_aas(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let mut fl = cc_compute_all(cpu.env);
        let eax = regv(cpu, R_EAX);
        let mut al = (eax & 0xff) as u32;
        let mut ah = ((eax >> 8) & 0xff) as u32;
        let icarry = u32::from(al < 6);
        if (al & 0x0f) > 9 || fl & CC_A != 0 {
            al = al.wrapping_sub(6) & 0x0f;
            ah = ah.wrapping_sub(1 + icarry) & 0xff;
            fl |= CC_C | CC_A;
        } else {
            fl &= !(CC_C | CC_A);
            al &= 0x0f;
        }
        set_reg(cpu, R_EAX, (eax & !0xffff) | u64::from(al | (ah << 8)));
        set_flags_eflags(cpu, fl);
        Ok(0)
    })
}

fn zps(al: u32) -> u32 {
    (u32::from(al == 0) << 6) | parity(u64::from(al)) | (al & 0x80)
}

fn h_daa(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let old = cc_compute_all(cpu.env);
        let eax = regv(cpu, R_EAX);
        let old_al = (eax & 0xff) as u32;
        let mut al = old_al;
        let mut fl = 0;
        if (al & 0x0f) > 9 || old & CC_A != 0 {
            al = (al + 6) & 0xff;
            fl |= CC_A;
        }
        if old_al > 0x99 || old & CC_C != 0 {
            al = (al + 0x60) & 0xff;
            fl |= CC_C;
        }
        set_reg(cpu, R_EAX, (eax & !0xff) | u64::from(al));
        set_flags_eflags(cpu, fl | zps(al));
        Ok(0)
    })
}

fn h_das(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let old = cc_compute_all(cpu.env);
        let cf = old & CC_C != 0;
        let eax = regv(cpu, R_EAX);
        let mut al = (eax & 0xff) as u32;
        let al1 = al;
        let mut fl = 0;
        if (al & 0x0f) > 9 || old & CC_A != 0 {
            fl |= CC_A;
            if al < 6 || cf {
                fl |= CC_C;
            }
            al = al.wrapping_sub(6) & 0xff;
        }
        if al1 > 0x99 || cf {
            al = al.wrapping_sub(0x60) & 0xff;
            fl |= CC_C;
        }
        set_reg(cpu, R_EAX, (eax & !0xff) | u64::from(al));
        set_flags_eflags(cpu, fl | zps(al));
        Ok(0)
    })
}

// CPUID and the time stamp counter.

def!(CPUID, "x86_cpuid", 0, Void, [Ptr], h_cpuid);
def!(RDTSC, "x86_rdtsc", 0, Void, [Ptr], h_rdtsc);
def!(RDTSCP, "x86_rdtscp", 0, Void, [Ptr], h_rdtscp);
def!(RDPMC, "x86_rdpmc", 0, Void, [Ptr], h_rdpmc);

fn h_cpuid(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let ops = cpu.ops();
        let x = x86_of(&ops);
        let r = x.model().cpuid_with(
            ld64(cpu.env, cr(4)),
            hflags(cpu),
            ld64(cpu.env, XCR0),
            regv(cpu, R_EAX) as u32,
            regv(cpu, R_ECX) as u32,
        );
        set_reg(cpu, R_EAX, u64::from(r[0]));
        set_reg(cpu, R_EBX, u64::from(r[1]));
        set_reg(cpu, R_ECX, u64::from(r[2]));
        set_reg(cpu, R_EDX, u64::from(r[3]));
        Ok(0)
    })
}

fn tsc(cpu: &Cpu<'_>) -> u64 {
    let ops = cpu.ops();
    x86_of(&ops).host_tsc().wrapping_add(ld64(cpu.env, TSC_OFFSET))
}

fn rdtsc(cpu: &mut Cpu<'_>) -> R<()> {
    if ld64(cpu.env, cr(4)) & CR4_TSD_MASK != 0 && cpl(cpu) != 0 {
        return Err(raise_exception_ra(cpu, EXCP0D_GPF, TB));
    }
    let v = tsc(cpu);
    set_reg(cpu, R_EAX, v & 0xffff_ffff);
    set_reg(cpu, R_EDX, v >> 32);
    Ok(())
}

fn h_rdtsc(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| rdtsc(cpu).map(|()| 0))
}

fn h_rdtscp(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        rdtsc(cpu)?;
        let aux = ld64(cpu.env, TSC_AUX) & 0xffff_ffff;
        set_reg(cpu, R_ECX, aux);
        Ok(0)
    })
}

/// `helper_rdpmc()`: performance counters are not implemented, so it is #UD after the
/// privilege check, as in QEMU.
fn h_rdpmc(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        if ld64(cpu.env, cr(4)) & CR4_PCE_MASK == 0 && cpl(cpu) != 0 {
            return Err(raise_exception_ra(cpu, EXCP0D_GPF, TB));
        }
        Err(raise_exception_err_ra(cpu, EXCP06_ILLOP, 0, TB))
    })
}

// Port I/O.

def!(INB, "x86_inb", 0, I64, [Ptr, I32], h_inb);
def!(INW, "x86_inw", 0, I64, [Ptr, I32], h_inw);
def!(INL, "x86_inl", 0, I64, [Ptr, I32], h_inl);
def!(OUTB, "x86_outb", 0, Void, [Ptr, I32, I32], h_outb);
def!(OUTW, "x86_outw", 0, Void, [Ptr, I32, I32], h_outw);
def!(OUTL, "x86_outl", 0, Void, [Ptr, I32, I32], h_outl);
def!(CHECK_IO, "x86_check_io", 0, Void, [Ptr, I32, I32], h_check_io);

fn port_in(h: &mut HelperEnv<'_>, port: u32, size: u32) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let ops = cpu.ops();
        let mask = if size == 4 { 0xffff_ffff } else { (1u64 << (size * 8)) - 1 };
        Ok(match x86_of(&ops).io() {
            Some(io) => {
                let (v, _) = io.load(
                    u64::from(port & 0xffff),
                    size,
                    Endian::Little,
                    MemTxAttrs::UNSPECIFIED,
                );
                v & mask
            }
            None => mask,
        })
    })
}

fn port_out(h: &mut HelperEnv<'_>, port: u32, val: u32, size: u32) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let ops = cpu.ops();
        if let Some(io) = x86_of(&ops).io() {
            let _ = io.store(
                u64::from(port & 0xffff),
                size,
                u64::from(val),
                Endian::Little,
                MemTxAttrs::UNSPECIFIED,
            );
        }
        Ok(0)
    })
}

fn h_inb(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    port_in(h, a32(a, 1), 1)
}

fn h_inw(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    port_in(h, a32(a, 1), 2)
}

fn h_inl(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    port_in(h, a32(a, 1), 4)
}

fn h_outb(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    port_out(h, a32(a, 1), a32(a, 2) & 0xff, 1)
}

fn h_outw(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    port_out(h, a32(a, 1), a32(a, 2) & 0xffff, 2)
}

fn h_outl(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    port_out(h, a32(a, 1), a32(a, 2), 4)
}

fn h_check_io(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| sh::helper_check_io(cpu, a32(a, 1), a32(a, 2), TB).map(|()| 0))
}

// MSRs.

def!(RDMSR, "x86_rdmsr", 0, Void, [Ptr], h_rdmsr);
def!(WRMSR, "x86_wrmsr", 0, Void, [Ptr], h_wrmsr);

/// `cpu_load_efer()`.
fn load_efer(cpu: &mut Cpu<'_>, val: u64) {
    st64(cpu.env, EFER, val);
    let mut hf = hflags(cpu) & !HF_LMA_MASK;
    if val & MSR_EFER_LMA != 0 {
        hf |= HF_LMA_MASK;
    }
    set_hflags(cpu, hf);
}

fn h_rdmsr(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let env = &*cpu.env;
        let val = match regv(cpu, R_ECX) as u32 {
            MSR_IA32_SYSENTER_CS => ld64(env, SYSENTER_CS),
            MSR_IA32_SYSENTER_ESP => ld64(env, SYSENTER_ESP),
            MSR_IA32_SYSENTER_EIP => ld64(env, SYSENTER_EIP),
            MSR_IA32_APICBASE => ld64(env, APIC_BASE),
            MSR_EFER => ld64(env, EFER),
            MSR_STAR => ld64(env, STAR),
            MSR_PAT => ld64(env, PAT),
            MSR_LSTAR => ld64(env, LSTAR),
            MSR_CSTAR => ld64(env, CSTAR),
            MSR_FMASK => ld64(env, FMASK),
            MSR_FSBASE => ld64(env, seg(R_FS) + SEG_BASE),
            MSR_GSBASE => ld64(env, seg(R_GS) + SEG_BASE),
            MSR_KERNELGSBASE => ld64(env, KERNELGSBASE),
            MSR_TSC_AUX => ld64(env, TSC_AUX),
            MSR_IA32_TSC => tsc(cpu),
            MSR_IA32_MISC_ENABLE => ld64(env, MISC_ENABLE),
            // QEMU reads the MSRs it does not know as zero.
            _ => 0,
        };
        set_reg(cpu, R_EAX, val & 0xffff_ffff);
        set_reg(cpu, R_EDX, val >> 32);
        Ok(0)
    })
}

fn h_wrmsr(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let val = (regv(cpu, R_EAX) & 0xffff_ffff) | (regv(cpu, R_EDX) << 32);
        match regv(cpu, R_ECX) as u32 {
            MSR_IA32_SYSENTER_CS => st64(cpu.env, SYSENTER_CS, val & 0xffff),
            MSR_IA32_SYSENTER_ESP => st64(cpu.env, SYSENTER_ESP, val),
            MSR_IA32_SYSENTER_EIP => st64(cpu.env, SYSENTER_EIP, val),
            MSR_IA32_APICBASE => st64(cpu.env, APIC_BASE, val),
            MSR_EFER => {
                let ops = cpu.ops();
                let m = x86_of(&ops).model();
                let mut update_mask = 0;
                if m.has_feature("syscall") {
                    update_mask |= MSR_EFER_SCE;
                }
                if m.has_feature("lm") {
                    update_mask |= MSR_EFER_LME;
                }
                if m.has_feature("fxsr-opt") {
                    update_mask |= MSR_EFER_FFXSR;
                }
                if m.has_feature("nx") {
                    update_mask |= MSR_EFER_NXE;
                }
                if m.has_feature("svm") {
                    update_mask |= MSR_EFER_SVME;
                }
                let efer = (ld64(cpu.env, EFER) & !update_mask) | (val & update_mask);
                load_efer(cpu, efer);
            }
            MSR_STAR => st64(cpu.env, STAR, val),
            MSR_PAT => st64(cpu.env, PAT, val),
            MSR_LSTAR => st64(cpu.env, LSTAR, val),
            MSR_CSTAR => st64(cpu.env, CSTAR, val),
            MSR_FMASK => st64(cpu.env, FMASK, val),
            MSR_FSBASE => st64(cpu.env, seg(R_FS) + SEG_BASE, val),
            MSR_GSBASE => st64(cpu.env, seg(R_GS) + SEG_BASE, val),
            MSR_KERNELGSBASE => st64(cpu.env, KERNELGSBASE, val),
            MSR_TSC_AUX => st64(cpu.env, TSC_AUX, val & 0xffff_ffff),
            MSR_IA32_TSC => {
                let ops = cpu.ops();
                let host = x86_of(&ops).host_tsc();
                st64(cpu.env, TSC_OFFSET, val.wrapping_sub(host));
            }
            MSR_IA32_MISC_ENABLE => st64(cpu.env, MISC_ENABLE, val),
            // QEMU ignores writes to the MSRs it does not know.
            _ => {}
        }
        Ok(0)
    })
}

// Control and debug registers.

def!(READ_CRN, "x86_read_crN", 0, I64, [Ptr, I32], h_read_crn);
def!(WRITE_CRN, "x86_write_crN", 0, Void, [Ptr, I32, I64], h_write_crn);
def!(GET_DR, "x86_get_dr", 0, I64, [Ptr, I32], h_get_dr);
def!(SET_DR, "x86_set_dr", 0, Void, [Ptr, I32, I64], h_set_dr);
def!(INVLPG, "x86_invlpg", 0, Void, [Ptr, I64], h_invlpg);
def!(LMSW, "x86_lmsw", 0, Void, [Ptr, I64], h_lmsw);
def!(CLTS, "x86_clts", 0, Void, [Ptr], h_clts);

fn h_read_crn(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        Ok(match a32(a, 1) {
            8 => ld64(cpu.env, CR8),
            n => ld64(cpu.env, cr(n as usize)),
        })
    })
}

/// `cpu_x86_update_cr0()`.
fn update_cr0(cpu: &mut Cpu<'_>, new_cr0: u64) {
    let new_cr0 = new_cr0 & 0xffff_ffff;
    let old = ld64(cpu.env, cr(0));
    let m = CR0_PG_MASK | CR0_WP_MASK | CR0_PE_MASK;
    if new_cr0 & m != old & m {
        tlb_flush(cpu);
    }
    let efer = ld64(cpu.env, EFER);
    if old & CR0_PG_MASK == 0 && new_cr0 & CR0_PG_MASK != 0 && efer & MSR_EFER_LME != 0 {
        // Enter long mode. QEMU leaves CR0 alone when PAE is off.
        if ld64(cpu.env, cr(4)) & CR4_PAE_MASK == 0 {
            return;
        }
        st64(cpu.env, EFER, efer | MSR_EFER_LMA);
        let hf = hflags(cpu) | HF_LMA_MASK;
        set_hflags(cpu, hf);
    } else if old & CR0_PG_MASK != 0 && new_cr0 & CR0_PG_MASK == 0 && efer & MSR_EFER_LMA != 0 {
        // Leave long mode.
        st64(cpu.env, EFER, efer & !MSR_EFER_LMA);
        let hf = hflags(cpu) & !(HF_LMA_MASK | HF_CS64_MASK);
        set_hflags(cpu, hf);
        let eip = ld64(cpu.env, EIP) & 0xffff_ffff;
        st64(cpu.env, EIP, eip);
    }
    let cr0 = new_cr0 | CR0_ET_MASK;
    st64(cpu.env, cr(0), cr0);
    let pe = cr0 & CR0_PE_MASK != 0;
    let mut hf = hflags(cpu) & !HF_PE_MASK;
    if pe {
        hf |= HF_PE_MASK;
    } else {
        // ADDSEG is always set in real mode.
        hf |= HF_ADDSEG_MASK;
    }
    hf &= !(HF_MP_MASK | HF_EM_MASK | HF_TS_MASK);
    // CR0.MP, EM and TS (bits 1 to 3) go to hflags bits 9 to 11.
    hf |= ((cr0 as u32) << 8) & (HF_MP_MASK | HF_EM_MASK | HF_TS_MASK);
    set_hflags(cpu, hf);
}

/// `cpu_x86_update_cr3()`.
fn update_cr3(cpu: &mut Cpu<'_>, v: u64) {
    st64(cpu.env, cr(3), v);
    if ld64(cpu.env, cr(0)) & CR0_PG_MASK != 0 {
        tlb_flush(cpu);
    }
}

/// `cr4_reserved_bits()`.
fn cr4_reserved_bits(cpu: &Cpu<'_>) -> u64 {
    let known = CR4_VME_MASK
        | CR4_PVI_MASK
        | CR4_TSD_MASK
        | CR4_DE_MASK
        | CR4_PSE_MASK
        | CR4_PAE_MASK
        | CR4_MCE_MASK
        | CR4_PGE_MASK
        | CR4_PCE_MASK
        | CR4_OSFXSR_MASK
        | CR4_OSXMMEXCPT_MASK
        | CR4_UMIP_MASK
        | CR4_LA57_MASK
        | CR4_FSGSBASE_MASK
        | CR4_PCIDE_MASK
        | CR4_OSXSAVE_MASK
        | CR4_SMEP_MASK
        | CR4_SMAP_MASK
        | CR4_PKE_MASK
        | CR4_CET_MASK
        | CR4_PKS_MASK
        | CR4_LAM_SUP_MASK
        | CR4_FRED_MASK;
    let ops = cpu.ops();
    let m = x86_of(&ops).model();
    let mut r = !known;
    for (feat, bit) in [
        ("xsave", CR4_OSXSAVE_MASK),
        ("smep", CR4_SMEP_MASK),
        ("smap", CR4_SMAP_MASK),
        ("fsgsbase", CR4_FSGSBASE_MASK),
        ("pku", CR4_PKE_MASK),
        ("la57", CR4_LA57_MASK),
        ("umip", CR4_UMIP_MASK),
        ("pks", CR4_PKS_MASK),
        ("lam", CR4_LAM_SUP_MASK),
        ("fred", CR4_FRED_MASK),
    ] {
        if !m.has_feature(feat) {
            r |= bit;
        }
    }
    if !m.has_feature("shstk") && !m.has_feature("ibt") {
        r |= CR4_CET_MASK;
    }
    r
}

/// `cpu_x86_update_cr4()`.
fn update_cr4(cpu: &mut Cpu<'_>, new_cr4: u64) {
    let old = ld64(cpu.env, cr(4));
    let flush =
        CR4_PGE_MASK | CR4_PAE_MASK | CR4_PSE_MASK | CR4_SMEP_MASK | CR4_SMAP_MASK | CR4_LA57_MASK;
    if (new_cr4 ^ old) & flush != 0 {
        tlb_flush(cpu);
    }
    let ops = cpu.ops();
    let m = x86_of(&ops).model();
    let mut v = new_cr4;
    let mut hf = hflags(cpu) & !(HF_OSFXSR_MASK | HF_SMAP_MASK | HF_UMIP_MASK | HF_AVX_EN_MASK);
    if !m.has_feature("sse") {
        v &= !CR4_OSFXSR_MASK;
    }
    if v & CR4_OSFXSR_MASK != 0 {
        hf |= HF_OSFXSR_MASK;
    }
    if !m.has_feature("smap") {
        v &= !CR4_SMAP_MASK;
    }
    if v & CR4_SMAP_MASK != 0 {
        hf |= HF_SMAP_MASK;
    }
    if !m.has_feature("umip") {
        v &= !CR4_UMIP_MASK;
    }
    if v & CR4_UMIP_MASK != 0 {
        hf |= HF_UMIP_MASK;
    }
    if !m.has_feature("pku") {
        v &= !CR4_PKE_MASK;
    }
    if !m.has_feature("pks") {
        v &= !CR4_PKS_MASK;
    }
    if !m.has_feature("lam") {
        v &= !CR4_LAM_SUP_MASK;
    }
    if avx_enabled(v, ld64(cpu.env, XCR0)) {
        hf |= HF_AVX_EN_MASK;
    }
    st64(cpu.env, cr(4), v);
    set_hflags(cpu, hf);
}

fn h_write_crn(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let t0 = a[2];
        match a32(a, 1) {
            0 => update_cr0(cpu, t0),
            3 => {
                let v = if ld64(cpu.env, EFER) & MSR_EFER_LMA == 0 { t0 & 0xffff_ffff } else { t0 };
                update_cr3(cpu, v);
            }
            4 => {
                // QEMU makes an SVM exit here; without SVM the reserved bits fault.
                if t0 & cr4_reserved_bits(cpu) != 0 {
                    return Err(raise_exception_err_ra(cpu, EXCP0D_GPF, 0, TB));
                }
                if (t0 ^ ld64(cpu.env, cr(4))) & CR4_LA57_MASK != 0
                    && hflags(cpu) & HF_CS64_MASK != 0
                {
                    return Err(raise_exception_ra(cpu, EXCP0D_GPF, TB));
                }
                update_cr4(cpu, t0);
            }
            8 => st64(cpu.env, CR8, t0 & 0xf),
            2 => st64(cpu.env, cr(2), t0),
            _ => {}
        }
        Ok(0)
    })
}

const DR7_GD: u64 = 1 << 13;
const DR6_BD: u64 = 1 << 13;
const DR_RESERVED_MASK: u64 = 0xffff_ffff_0000_0000;

/// Resolve DR4 and DR5 and check DR7.GD, as `helper_get_dr()` and `helper_set_dr()` do.
fn dr_index(cpu: &mut Cpu<'_>, n: u32) -> R<usize> {
    let mut n = n as usize;
    if (4..6).contains(&n) {
        if ld64(cpu.env, cr(4)) & CR4_DE_MASK != 0 {
            return Err(raise_exception_ra(cpu, EXCP06_ILLOP, TB));
        }
        n += 2;
    }
    let d7 = ld64(cpu.env, dr(7));
    if d7 & DR7_GD != 0 {
        st64(cpu.env, dr(7), d7 & !DR7_GD);
        let d6 = ld64(cpu.env, dr(6)) | DR6_BD;
        st64(cpu.env, dr(6), d6);
        return Err(raise_exception_ra(cpu, EXCP01_DB, TB));
    }
    Ok(n)
}

fn h_get_dr(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let n = dr_index(cpu, a32(a, 1))?;
        Ok(ld64(cpu.env, dr(n)))
    })
}

fn h_set_dr(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let n = dr_index(cpu, a32(a, 1))?;
        let t0 = a[2];
        if n < 4 {
            st64(cpu.env, dr(n), t0);
        } else {
            if t0 & DR_RESERVED_MASK != 0 {
                return Err(raise_exception_err_ra(cpu, EXCP0D_GPF, 0, TB));
            }
            let fixed = if n == 6 { DR6_FIXED_1 } else { DR7_FIXED_1 };
            st64(cpu.env, dr(n), t0 | fixed);
        }
        Ok(0)
    })
}

fn h_invlpg(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        tlb_flush_page(cpu, a[1]);
        Ok(0)
    })
}

/// `helper_lmsw()`: only the low four bits of CR0 change, and PE cannot be cleared.
fn h_lmsw(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let t0 = (ld64(cpu.env, cr(0)) & !0xe) | (a[1] & 0xf);
        update_cr0(cpu, t0);
        Ok(0)
    })
}

fn h_clts(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let c = ld64(cpu.env, cr(0)) & !CR0_TS_MASK;
        st64(cpu.env, cr(0), c);
        let hf = hflags(cpu) & !HF_TS_MASK;
        set_hflags(cpu, hf);
        Ok(0)
    })
}

// EFLAGS.

def!(READ_EFLAGS, "x86_read_eflags", 0, I64, [Ptr], h_read_eflags);
def!(WRITE_EFLAGS, "x86_write_eflags", 0, Void, [Ptr, I64, I32], h_write_eflags);

/// `helper_read_eflags()`: what PUSHF sees, without VM and RF.
fn h_read_eflags(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let crate_vm_rf = super::env::VM_MASK | RF_MASK;
        Ok(u64::from(compute_eflags(cpu.env) & !crate_vm_rf))
    })
}

fn h_write_eflags(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        load_eflags(cpu.env, a[1] as u32, a32(a, 2));
        Ok(0)
    })
}

// Segments and far transfers.

def!(LOAD_SEG, "x86_load_seg", 0, Void, [Ptr, I32, I32], h_load_seg);
def!(LJMP_PROTECTED, "x86_ljmp_protected", 0, Void, [Ptr, I32, I64], h_ljmp_protected);
def!(LCALL_REAL, "x86_lcall_real", 0, Void, [Ptr, I32, I64, I32, I64], h_lcall_real);
def!(LCALL_PROTECTED, "x86_lcall_protected", 0, Void, [Ptr, I32, I64, I32, I64], h_lcall_protected);
def!(IRET_REAL, "x86_iret_real", 0, Void, [Ptr, I32], h_iret_real);
def!(IRET_PROTECTED, "x86_iret_protected", 0, Void, [Ptr, I32], h_iret_protected);
def!(LRET_PROTECTED, "x86_lret_protected", 0, Void, [Ptr, I32, I64], h_lret_protected);
def!(SYSCALL, "x86_syscall", 0, Void, [Ptr, I32], h_syscall);
def!(SYSRET, "x86_sysret", 0, Void, [Ptr, I32], h_sysret);
def!(SYSENTER, "x86_sysenter", 0, Void, [Ptr], h_sysenter);
def!(SYSEXIT, "x86_sysexit", 0, Void, [Ptr, I32], h_sysexit);
def!(LLDT, "x86_lldt", 0, Void, [Ptr, I32], h_lldt);
def!(LTR, "x86_ltr", 0, Void, [Ptr, I32], h_ltr);
def!(LAR, "x86_lar", 0, I64, [Ptr, I32], h_lar);
def!(LSL, "x86_lsl", 0, I64, [Ptr, I32], h_lsl);
def!(VERR, "x86_verr", 0, Void, [Ptr, I32], h_verr);
def!(VERW, "x86_verw", 0, Void, [Ptr, I32], h_verw);
def!(BOUNDW, "x86_boundw", 0, Void, [Ptr, I64, I32], h_boundw);
def!(BOUNDL, "x86_boundl", 0, Void, [Ptr, I64, I32], h_boundl);

fn unit(r: R<()>) -> R<u64> {
    r.map(|()| 0)
}

fn h_load_seg(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| unit(sh::helper_load_seg(cpu, a32(a, 1) as usize, a32(a, 2), TB)))
}

fn h_ljmp_protected(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| unit(sh::helper_ljmp_protected(cpu, a32(a, 1), a[2], TB)))
}

fn h_lcall_real(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| unit(sh::helper_lcall_real(cpu, a32(a, 1), a[2], a32(a, 3), a[4], TB)))
}

fn h_lcall_protected(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| unit(sh::helper_lcall_protected(cpu, a32(a, 1), a[2], a32(a, 3), a[4], TB)))
}

fn h_iret_real(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| unit(sh::helper_iret_real(cpu, a32(a, 1), TB)))
}

fn h_iret_protected(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| unit(sh::helper_iret_protected(cpu, a32(a, 1), TB)))
}

fn h_lret_protected(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| unit(sh::helper_lret_protected(cpu, a32(a, 1), a[2], TB)))
}

fn h_syscall(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| unit(sh::helper_syscall(cpu, u64::from(a32(a, 1)), TB)))
}

fn h_sysret(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| unit(sh::helper_sysret(cpu, a32(a, 1), TB)))
}

fn h_sysenter(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| unit(sh::helper_sysenter(cpu, TB)))
}

fn h_sysexit(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| unit(sh::helper_sysexit(cpu, a32(a, 1), TB)))
}

fn h_lldt(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| unit(sh::helper_lldt(cpu, a32(a, 1), TB)))
}

fn h_ltr(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| unit(sh::helper_ltr(cpu, a32(a, 1), TB)))
}

fn h_lar(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| sh::helper_lar_lsl(cpu, a32(a, 1), false, TB))
}

fn h_lsl(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| sh::helper_lar_lsl(cpu, a32(a, 1), true, TB))
}

fn h_verr(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| unit(sh::helper_verr_verw(cpu, a32(a, 1), false, TB)))
}

fn h_verw(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| unit(sh::helper_verr_verw(cpu, a32(a, 1), true, TB)))
}

/// `helper_boundw()` and `helper_boundl()`: `addr` is the linear address of the bounds.
fn bound(h: &mut HelperEnv<'_>, addr: u64, v: u32, wide: bool) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let idx = mmu_index_pl(cpu.env, cpl(cpu));
        let (lo, hi, v) = if wide {
            let lo = sh::ld(cpu, addr, MemOp::LEUL, idx, TB)? as u32 as i32;
            let hi = sh::ld(cpu, addr.wrapping_add(4) & 0xffff_ffff, MemOp::LEUL, idx, TB)? as u32
                as i32;
            (lo, hi, v as i32)
        } else {
            let lo = i32::from(sh::ld(cpu, addr, MemOp::LEUW, idx, TB)? as u16 as i16);
            let hi =
                i32::from(sh::ld(cpu, addr.wrapping_add(2) & 0xffff_ffff, MemOp::LEUW, idx, TB)?
                    as u16 as i16);
            (lo, hi, i32::from(v as u16 as i16))
        };
        if v < lo || v > hi {
            return Err(raise_exception_ra(cpu, EXCP05_BOUND, TB));
        }
        Ok(0)
    })
}

fn h_boundw(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    bound(h, a[1], a32(a, 2), false)
}

fn h_boundl(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    bound(h, a[1], a32(a, 2), true)
}

/// Every helper.
const ALL: &[&Def] = &[
    &CC_COMPUTE_ALL,
    &CC_COMPUTE_C,
    &CC_COMPUTE_NZ,
    &RAISE_EXCEPTION,
    &RAISE_EXCEPTION_ERR,
    &RAISE_INTERRUPT,
    &INTO,
    &SINGLE_STEP,
    &HLT,
    &ICEBP,
    &RECHECKING_SINGLE_STEP,
    &DIVB,
    &IDIVB,
    &DIVW,
    &IDIVW,
    &DIVL,
    &IDIVL,
    &DIVQ,
    &IDIVQ,
    &AAA,
    &AAS,
    &DAA,
    &DAS,
    &AAM,
    &AAD,
    &CPUID,
    &RDTSC,
    &RDTSCP,
    &RDPMC,
    &INB,
    &INW,
    &INL,
    &OUTB,
    &OUTW,
    &OUTL,
    &CHECK_IO,
    &RDMSR,
    &WRMSR,
    &READ_CRN,
    &WRITE_CRN,
    &GET_DR,
    &SET_DR,
    &INVLPG,
    &LMSW,
    &CLTS,
    &READ_EFLAGS,
    &WRITE_EFLAGS,
    &LOAD_SEG,
    &LJMP_PROTECTED,
    &LCALL_REAL,
    &LCALL_PROTECTED,
    &IRET_REAL,
    &IRET_PROTECTED,
    &LRET_PROTECTED,
    &SYSCALL,
    &SYSRET,
    &SYSENTER,
    &SYSEXIT,
    &LLDT,
    &LTR,
    &LAR,
    &LSL,
    &VERR,
    &VERW,
    &BOUNDW,
    &BOUNDL,
];

/// Register every x86 helper in `r`.
pub(crate) fn register(r: &mut HelperRegistry) {
    for d in ALL.iter().chain(vec::ALL).chain(fpu::ALL) {
        r.register_info(&d.info(), d.f);
    }
}
