// SPDX-License-Identifier: GPL-2.0-or-later

//! The x87 helpers and the FXSAVE, XSAVE and XRSTOR family: a port of the x87 and save area
//! parts of `fpu_helper.c`.
//!
//! QEMU has one helper per x87 operation. Here one helper, [`X87`], takes the operation and
//! an ST index (or, for the environment instructions, the operand size) packed into one word
//! by [`x87_op`], plus a value or an address, and returns whatever the operation stores. Like
//! QEMU's x87 helpers it is declared with no call flags, so the translator syncs `cc_op`
//! before it and the helper may read and write any global.
//!
//! QEMU keeps an x87 `float_status` in `env` whose rounding mode and precision follow FPUC.
//! Each call here builds a fresh status from FPUC instead. QEMU only folds the flags raised
//! during a helper into FPUS (`save_exception_flags()` and `merge_exception_flags()`), and
//! nothing else reads the accumulated x87 flags, so FPUS ends up the same.
//!
//! Differences from QEMU:
//!
//! - The `X86Access` probes are done exactly as `access_prepare()` does them, but the
//!   accesses that follow go through the softmmu load and store paths rather than host
//!   pointers. The probes have already raised any fault, so the guest sees the same
//!   exceptions in the same order.
//! - FERR# and IGNNE# are not modeled yet: FWAIT with an unmasked exception pending and
//!   CR0.NE clear does nothing, and FLDENV, FRSTOR, FXRSTOR and XRSTOR do not call
//!   `cpu_clear_ignne()`.
//! - The MPX components of XSAVE and XRSTOR are not handled. TCG never reports MPX in CPUID,
//!   so XSETBV cannot enable them and XCR0 never has those bits set.
//! - FSIN, FCOS, FSINCOS and FPTAN go through the host `sin`, `cos` and `tan` on doubles, as
//!   in QEMU, so the low bits of their results can differ between host C libraries.
//! - QEMU's F2XM1 returns the table value of `t` rather than `2^t - 1` when the argument is
//!   an exact multiple of 1/32 (other than -1, 0 and 1). That is kept.
//! - QEMU passes `dflag - 1` as the operand size of FSTENV, FLDENV, FSAVE and FRSTOR, so with
//!   REX.W it is 2 and the probe sizes and register offsets are scaled by 4 rather than 2,
//!   while the layout written is the 32-bit one. That is kept.
//! - XSAVEOPT has no helper of its own: the translator calls the XSAVE helper, as QEMU does,
//!   so it always writes every requested component.

use ruvm_jit::cputlb::{probe_access, tlb_flush};
use ruvm_jit::{Cpu, MmuAccessType};
use ruvm_jit_core::MemOp;
use ruvm_jit_interp::{HelperEnv, Unwind};
use ruvm_softfloat::{
    Float32, Float64, FloatRelation, FloatStatus, FloatX80, FloatX80RoundPrec as Prec, RoundMode,
    flags as ff,
};

use super::super::cc::CC_OP_EFLAGS;
use super::super::env::{
    CC_C, CC_OP, CC_P, CC_SRC, CC_Z, EFER, FPCS, FPDP, FPDS, FPIP, FPSTT, FPTAGS, FPUC, FPUS, FT0,
    MXCSR, PKRU, XCR0, cc_compute_all, cr, fpreg, ld32, ld64, st32, st64, zmm,
};
use super::super::{EXCP0D_GPF, EXCP06_ILLOP, EXCP10_COPR, mmu_index_pl, x86_of};
use super::{Def, I32, I64, Ptr, R, TB, Void, a32, hflags, run, sh};
use crate::cpuid::xsave::xsave_area_size;
use crate::state::{
    CR0_NE_MASK, CR4_OSFXSR_MASK, CR4_OSXSAVE_MASK, HF_CPL_MASK, HF_CS64_MASK, HF_LMA_MASK,
    MSR_EFER_FFXSR,
};

/// The FPUS exception and summary bits.
const FPUS_IE: u32 = 1 << 0;
const FPUS_DE: u32 = 1 << 1;
const FPUS_ZE: u32 = 1 << 2;
const FPUS_OE: u32 = 1 << 3;
const FPUS_UE: u32 = 1 << 4;
const FPUS_PE: u32 = 1 << 5;
const FPUS_SE: u32 = 1 << 7;
const FPUS_B: u32 = 1 << 15;
/// The exception mask bits of FPUC.
const FPUC_EM: u32 = 0x3f;
/// The rounding control field of FPUC and its values.
const FPU_RC_MASK: u32 = 0xc00;
const FPU_RC_DOWN: u32 = 0x400;
const FPU_RC_UP: u32 = 0x800;
const FPU_RC_CHOP: u32 = 0xc00;

/// The XSAVE state components the helpers know about.
const XSTATE_FP: u64 = 1;
const XSTATE_SSE: u64 = 2;
const XSTATE_YMM: u64 = 4;
const XSTATE_BNDREGS: u64 = 8;
const XSTATE_PKRU: u64 = 1 << 9;

/// Offsets in the standard XSAVE layout (`X86XSaveArea`).
const XO_FCW: u64 = 0;
const XO_FSW: u64 = 2;
const XO_FTW: u64 = 4;
const XO_FPIP: u64 = 8;
const XO_FPDP: u64 = 16;
const XO_MXCSR: u64 = 24;
const XO_MXCSR_MASK: u64 = 28;
const XO_FPREGS: u64 = 32;
const XO_XMM: u64 = 160;
const XO_XSTATE_BV: u64 = 512;
const XO_XCOMP_BV: u64 = 520;
const XO_RESERVE0: u64 = 528;
const XO_AVX: u64 = 576;
const XO_PKRU: u64 = 2688;
/// `sizeof(X86LegacyXSaveArea)` and that plus `sizeof(X86XSaveHeader)`.
const LEGACY_SIZE: usize = 512;
const LEGACY_HEADER_SIZE: usize = 576;

/// `make_floatx80()`.
const fn fx(high: u16, low: u64) -> FloatX80 {
    FloatX80::new(high, low)
}

const ZERO: FloatX80 = fx(0, 0);
const ONE: FloatX80 = fx(0x3fff, 0x8000000000000000);
const L2T: FloatX80 = fx(0x4000, 0xd49a784bcd1b8afe);
const L2T_U: FloatX80 = fx(0x4000, 0xd49a784bcd1b8aff);
const L2E: FloatX80 = fx(0x3fff, 0xb8aa3b295c17f0bc);
const L2E_D: FloatX80 = fx(0x3fff, 0xb8aa3b295c17f0bb);
const PI: FloatX80 = fx(0x4000, 0xc90fdaa22168c235);
const PI_D: FloatX80 = fx(0x4000, 0xc90fdaa22168c234);
const LG2: FloatX80 = fx(0x3ffd, 0x9a209a84fbcff799);
const LG2_D: FloatX80 = fx(0x3ffd, 0x9a209a84fbcff798);
const LN2: FloatX80 = fx(0x3ffe, 0xb17217f7d1cf79ac);
const LN2_D: FloatX80 = fx(0x3ffe, 0xb17217f7d1cf79ab);

/// `MAXTAN`: the range limit of FPTAN, FSIN, FCOS and FSINCOS.
const MAXTAN: f64 = 9223372036854775808.0;

macro_rules! ops {
    ($($v:ident),* $(,)?) => {
        /// The x87 operations of [`X87`], one per QEMU helper.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(u8)]
        #[allow(clippy::enum_variant_names, reason = "the names of the QEMU helpers")]
        pub(crate) enum F {
            $($v),*
        }

        impl F {
            const ALL: &[F] = &[$(F::$v),*];

            fn from_u8(v: u8) -> Option<F> {
                Self::ALL.get(usize::from(v)).copied()
            }
        }
    };
}

ops! {
    FldsFt0, FldlFt0, FildlFt0, FldsSt0, FldlSt0, FildlSt0, FildllSt0,
    FstsSt0, FstlSt0, FistSt0, FistlSt0, FistllSt0, FisttSt0, FisttlSt0, FisttllSt0,
    FldtSt0, FsttSt0, FbldSt0, FbstSt0,
    Fpush, Fpop, Fdecstp, Fincstp, FfreeStn,
    FmovSt0Ft0, FmovFt0Stn, FmovSt0Stn, FmovStnSt0, FxchgSt0Stn,
    FcomSt0Ft0, FucomSt0Ft0, FcomiSt0Ft0, FucomiSt0Ft0,
    FaddSt0Ft0, FmulSt0Ft0, FsubSt0Ft0, FsubrSt0Ft0, FdivSt0Ft0, FdivrSt0Ft0,
    FaddStnSt0, FmulStnSt0, FsubStnSt0, FsubrStnSt0, FdivStnSt0, FdivrStnSt0,
    Fchs, Fabs, Fld1, Fldl2t, Fldl2e, Fldpi, Fldlg2, Fldln2, FldzSt0, FldzFt0,
    Fnstsw, Fnstcw, Fldcw, Fclex, Fwait, Fninit,
    F2xm1, Fyl2x, Fptan, Fpatan, Fxtract, Fprem1, Fprem, Fyl2xp1, Fsqrt, Fsincos, Frndint,
    Fscale, Fsin, Fcos, Fxam,
    Fstenv, Fldenv, Fsave, Frstor,
}

/// The second operand of [`X87`] for operation `f` on ST(`n`), or with `n` set to QEMU's
/// `data32` argument (`dflag - 1`) for FSTENV, FLDENV, FSAVE and FRSTOR.
pub(crate) fn x87_op(f: F, n: u32) -> i32 {
    (f as u32 | (n & 7) << 8) as i32
}

/// `gen_helper_fp_arith_ST0_FT0()`: the ST0, FT0 operation for each `/reg` of D8 and DC.
pub(crate) const ARITH_ST0_FT0: [F; 8] = [
    F::FaddSt0Ft0,
    F::FmulSt0Ft0,
    F::FcomSt0Ft0,
    F::FcomSt0Ft0,
    F::FsubSt0Ft0,
    F::FsubrSt0Ft0,
    F::FdivSt0Ft0,
    F::FdivrSt0Ft0,
];

/// `gen_helper_fp_arith_STN_ST0()`: the STn, ST0 operation for each `/reg` of DC and DE.
/// Entries 2 and 3 are never used.
pub(crate) const ARITH_STN_ST0: [F; 8] = [
    F::FaddStnSt0,
    F::FmulStnSt0,
    F::FaddStnSt0,
    F::FaddStnSt0,
    F::FsubrStnSt0,
    F::FsubStnSt0,
    F::FdivrStnSt0,
    F::FdivStnSt0,
];

def!(X87, "x86_x87", 0, I64, [Ptr, I32, I64], h_x87);
def!(FXSAVE, "x86_fxsave", 0, Void, [Ptr, I64], h_fxsave);
def!(FXRSTOR, "x86_fxrstor", 0, Void, [Ptr, I64], h_fxrstor);
def!(XSAVE, "x86_xsave", 0, Void, [Ptr, I64, I64], h_xsave);
def!(XRSTOR, "x86_xrstor", 0, Void, [Ptr, I64, I64], h_xrstor);

pub(super) const ALL: &[&Def] = &[&X87, &FXSAVE, &FXRSTOR, &XSAVE, &XRSTOR];

// The x87 state in env.

/// The x87 float status for the current FPUC (`update_fp_status()`), with no flags raised.
fn status(e: &[u8]) -> FloatStatus {
    let fpuc = ld32(e, FPUC);
    let mut s = FloatStatus::x87();
    s.rounding_mode = RoundMode::from_u8(((fpuc >> 10) & 3) as u8).unwrap_or_default();
    s.floatx80_rounding_precision = match (fpuc >> 8) & 3 {
        0 => Prec::S,
        2 => Prec::D,
        _ => Prec::X,
    };
    s
}

/// `fpu_set_exception()`.
fn set_exception(e: &mut [u8], mask: u32) {
    let mut fpus = ld32(e, FPUS) | mask;
    if fpus & (!ld32(e, FPUC) & FPUC_EM) != 0 {
        fpus |= FPUS_SE | FPUS_B;
    }
    st32(e, FPUS, fpus);
}

/// `merge_exception_flags()` for the flags `f` raised by one helper.
fn merge(e: &mut [u8], f: u16) {
    let bits = [
        (ff::INVALID, FPUS_IE),
        (ff::DIVBYZERO, FPUS_ZE),
        (ff::OVERFLOW, FPUS_OE),
        (ff::UNDERFLOW, FPUS_UE),
        (ff::INEXACT, FPUS_PE),
        (ff::INPUT_DENORMAL_USED, FPUS_DE),
    ];
    let mask = bits.iter().filter(|(fl, _)| f & fl != 0).fold(0, |m, (_, b)| m | b);
    set_exception(e, mask);
}

fn top(e: &[u8]) -> usize {
    (ld32(e, FPSTT) & 7) as usize
}

fn ldx(e: &[u8], off: usize) -> FloatX80 {
    FloatX80::new(ld64(e, off + 8) as u16, ld64(e, off))
}

fn stx(e: &mut [u8], off: usize, v: FloatX80) {
    st64(e, off, v.low);
    st64(e, off + 8, u64::from(v.high));
}

/// `ST(n)`.
fn get(e: &[u8], n: usize) -> FloatX80 {
    ldx(e, fpreg((top(e) + n) & 7))
}

fn set(e: &mut [u8], n: usize, v: FloatX80) {
    let off = fpreg((top(e) + n) & 7);
    stx(e, off, v);
}

fn fpush(e: &mut [u8]) {
    let t = (top(e) + 7) & 7;
    st32(e, FPSTT, t as u32);
    e[FPTAGS + t] = 0;
}

fn fpop(e: &mut [u8]) {
    let t = top(e);
    e[FPTAGS + t] = 1;
    st32(e, FPSTT, ((t + 1) & 7) as u32);
}

fn and_fpus(e: &mut [u8], m: u32) {
    st32(e, FPUS, ld32(e, FPUS) & m);
}

fn or_fpus(e: &mut [u8], m: u32) {
    st32(e, FPUS, ld32(e, FPUS) | m);
}

/// `cpu_set_fpus()`.
fn set_fpus(e: &mut [u8], fpus: u32) {
    st32(e, FPSTT, (fpus >> 11) & 7);
    let mut v = fpus & 0xffff & !0x3800 & !FPUS_B;
    if v & FPUS_SE != 0 {
        v |= FPUS_B;
    }
    st32(e, FPUS, v);
}

/// `do_fninit()`.
fn fninit(e: &mut [u8]) {
    st32(e, FPUS, 0);
    st32(e, FPSTT, 0);
    st32(e, FPCS, 0);
    st32(e, FPDS, 0);
    st64(e, FPIP, 0);
    st64(e, FPDP, 0);
    st32(e, FPUC, 0x37f);
    e[FPTAGS..FPTAGS + 8].fill(1);
}

/// The status word as FNSTSW and FSTENV store it.
fn fnstsw(e: &[u8]) -> u32 {
    (ld32(e, FPUS) & !0x3800) | (top(e) as u32) << 11
}

fn exp_of(x: FloatX80) -> i32 {
    i32::from(x.high & 0x7fff)
}

fn sign_of(x: FloatX80) -> bool {
    x.high & 0x8000 != 0
}

/// The fraction, exponent and sign of `x`.
fn parts(x: FloatX80) -> (u64, i32, bool) {
    (x.low, exp_of(x), sign_of(x))
}

fn rel_idx(r: FloatRelation) -> usize {
    match r {
        FloatRelation::Less => 0,
        FloatRelation::Equal => 1,
        FloatRelation::Greater => 2,
        FloatRelation::Unordered => 3,
    }
}

// Private copies of the softfloat-macros.h helpers the transcendental functions use. Shift
// counts use wrapping shifts so that out of range counts behave as on an x86 host instead
// of panicking.

/// `shift128RightJamming()`.
fn shr_jam(a0: u64, a1: u64, count: i32) -> (u64, u64) {
    let neg = (count.wrapping_neg() & 63) as u32;
    let c = count as u32;
    if count == 0 {
        (a0, a1)
    } else if count < 64 {
        let z1 = a0.wrapping_shl(neg) | a1.wrapping_shr(c) | u64::from(a1.wrapping_shl(neg) != 0);
        (a0.wrapping_shr(c), z1)
    } else if count == 64 {
        (0, a0 | u64::from(a1 != 0))
    } else if count < 128 {
        (0, a0.wrapping_shr(c & 63) | u64::from((a0.wrapping_shl(neg) | a1) != 0))
    } else {
        (0, u64::from((a0 | a1) != 0))
    }
}

/// `shift128Right()`.
fn shr128(a0: u64, a1: u64, count: i32) -> (u64, u64) {
    let neg = (count.wrapping_neg() & 63) as u32;
    let c = count as u32;
    if count == 0 {
        (a0, a1)
    } else if count < 64 {
        (a0.wrapping_shr(c), a0.wrapping_shl(neg) | a1.wrapping_shr(c))
    } else if count < 128 {
        (0, a0.wrapping_shr(c & 63))
    } else {
        (0, 0)
    }
}

/// `shift128Left()`.
fn shl128(a0: u64, a1: u64, count: i32) -> (u64, u64) {
    let c = count as u32;
    if count < 64 {
        let z0 = if count == 0 {
            a0
        } else {
            a0.wrapping_shl(c) | a1.wrapping_shr(c.wrapping_neg() & 63)
        };
        (z0, a1.wrapping_shl(c))
    } else {
        (a1.wrapping_shl(c - 64), 0)
    }
}

/// `add128()`.
fn add128(a0: u64, a1: u64, b0: u64, b1: u64) -> (u64, u64) {
    let z1 = a1.wrapping_add(b1);
    (a0.wrapping_add(b0).wrapping_add(u64::from(z1 < a1)), z1)
}

/// `sub128()`.
fn sub128(a0: u64, a1: u64, b0: u64, b1: u64) -> (u64, u64) {
    (a0.wrapping_sub(b0).wrapping_sub(u64::from(a1 < b1)), a1.wrapping_sub(b1))
}

/// `add192()`.
fn add192(a0: u64, a1: u64, a2: u64, b0: u64, b1: u64, b2: u64) -> (u64, u64, u64) {
    let z2 = a2.wrapping_add(b2);
    let carry1 = u64::from(z2 < a2);
    let mut z1 = a1.wrapping_add(b1);
    let carry0 = u64::from(z1 < a1);
    let mut z0 = a0.wrapping_add(b0);
    z1 = z1.wrapping_add(carry1);
    z0 = z0.wrapping_add(u64::from(z1 < carry1));
    z0 = z0.wrapping_add(carry0);
    (z0, z1, z2)
}

/// `sub192()`.
fn sub192(a0: u64, a1: u64, a2: u64, b0: u64, b1: u64, b2: u64) -> (u64, u64, u64) {
    let z2 = a2.wrapping_sub(b2);
    let borrow1 = u64::from(a2 < b2);
    let mut z1 = a1.wrapping_sub(b1);
    let borrow0 = u64::from(a1 < b1);
    let mut z0 = a0.wrapping_sub(b0);
    z0 = z0.wrapping_sub(u64::from(z1 < borrow1));
    z1 = z1.wrapping_sub(borrow1);
    z0 = z0.wrapping_sub(borrow0);
    (z0, z1, z2)
}

/// `mul64To128()`.
fn mul64(a: u64, b: u64) -> (u64, u64) {
    let p = u128::from(a) * u128::from(b);
    ((p >> 64) as u64, p as u64)
}

/// `mul128By64To192()`.
fn mul128_by64(a0: u64, a1: u64, b: u64) -> (u64, u64, u64) {
    let (z1, z2) = mul64(a1, b);
    let (z0, more1) = mul64(a0, b);
    let (z0, z1) = add128(z0, more1, 0, z1);
    (z0, z1, z2)
}

/// `mul128To256()`.
fn mul128_to256(a0: u64, a1: u64, b0: u64, b1: u64) -> (u64, u64, u64, u64) {
    let (m1, m2) = mul64(a1, b0);
    let (n1, n2) = mul64(a0, b1);
    let (z2, z3) = mul64(a1, b1);
    let (z0, z1) = mul64(a0, b0);
    let (m0, m1, m2) = add192(0, m1, m2, 0, n1, n2);
    let (z0, z1, z2) = add192(m0, m1, m2, z0, z1, z2);
    (z0, z1, z2, z3)
}

/// `estimateDiv128To64()`.
fn est_div(a0: u64, a1: u64, b: u64) -> u64 {
    if b <= a0 {
        return u64::MAX;
    }
    let b0 = b >> 32;
    let mut z = if b0 << 32 <= a0 { 0xFFFF_FFFF_0000_0000 } else { (a0 / b0) << 32 };
    let (t0, t1) = mul64(b, z);
    let (mut rem0, mut rem1) = sub128(a0, a1, t0, t1);
    while (rem0 as i64) < 0 {
        z = z.wrapping_sub(0x1_0000_0000);
        let b1 = b << 32;
        (rem0, rem1) = add128(rem0, rem1, b0, b1);
    }
    rem0 = (rem0 << 32) | (rem1 >> 32);
    z | if b0 << 32 <= rem0 { 0xFFFF_FFFF } else { rem0 / b0 }
}

/// `normalizeFloatx80Subnormal()`: the exponent and significand of a nonzero subnormal.
fn norm_sub(sig: u64) -> (i32, u64) {
    let shift = sig.leading_zeros();
    (1 - shift as i32, sig.wrapping_shl(shift))
}

/// `clz64()` as an `i32`.
fn clz(v: u64) -> i32 {
    v.leading_zeros() as i32
}

/// The x87 helper: runs operation `op` (see [`x87_op`]) with value or address `a[2]`.
fn h_x87(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    let op = a32(a, 1);
    let v = a[2];
    run(h, |cpu| x87(cpu, op, v))
}

fn x87(cpu: &mut Cpu<'_>, op: u32, v: u64) -> R<u64> {
    let f = F::from_u8(op as u8).expect("a valid x87 operation");
    let n = ((op >> 8) & 7) as usize;
    let mut s = status(cpu.env);
    let (ret, merged) = match f {
        F::FldtSt0 => {
            let ac = Access::prepare(cpu, v, 10, MmuAccessType::DataLoad)?;
            let x = ac.ldt(cpu, v)?;
            fpush(cpu.env);
            set(cpu.env, 0, x);
            (0, false)
        }
        F::FsttSt0 => {
            let ac = Access::prepare(cpu, v, 10, MmuAccessType::DataStore)?;
            let x = get(cpu.env, 0);
            ac.stt(cpu, v, x)?;
            (0, false)
        }
        F::FbldSt0 => {
            fbld(cpu, v, &mut s)?;
            (0, false)
        }
        F::FbstSt0 => {
            fbst(cpu, v, &mut s)?;
            (0, true)
        }
        F::Fwait => {
            if ld32(cpu.env, FPUS) & FPUS_SE != 0 && ld64(cpu.env, cr(0)) & CR0_NE_MASK != 0 {
                return Err(sh::raise_exception_ra(cpu, EXCP10_COPR, TB));
            }
            (0, false)
        }
        F::Fstenv => {
            let ac = Access::prepare(cpu, v, 14 << n, MmuAccessType::DataStore)?;
            fstenv(cpu, &ac, v, n)?;
            (0, false)
        }
        F::Fldenv => {
            // QEMU probes FLDENV for a store.
            let ac = Access::prepare(cpu, v, 14 << n, MmuAccessType::DataStore)?;
            fldenv(cpu, &ac, v, n)?;
            (0, false)
        }
        F::Fsave => {
            let size = (14 << n) + 80;
            let ac = Access::prepare(cpu, v, size, MmuAccessType::DataStore)?;
            fstenv(cpu, &ac, v, n)?;
            let mut p = v.wrapping_add(14 << n);
            for i in 0..8 {
                let x = get(cpu.env, i);
                ac.stt(cpu, p, x)?;
                p = p.wrapping_add(10);
            }
            fninit(cpu.env);
            (0, false)
        }
        F::Frstor => {
            let size = (14 << n) + 80;
            let ac = Access::prepare(cpu, v, size, MmuAccessType::DataLoad)?;
            fldenv(cpu, &ac, v, n)?;
            let mut p = v.wrapping_add(14 << n);
            for i in 0..8 {
                let x = ac.ldt(cpu, p)?;
                set(cpu.env, i, x);
                p = p.wrapping_add(10);
            }
            (0, false)
        }
        _ => reg_op(cpu.env, f, n, v, &mut s),
    };
    if merged {
        merge(cpu.env, s.flags());
    }
    Ok(ret)
}

/// The operations that only touch `env`. Returns the result and whether QEMU's helper
/// merges the float flags into FPUS.
fn reg_op(e: &mut [u8], f: F, n: usize, v: u64, s: &mut FloatStatus) -> (u64, bool) {
    let mut ret = 0;
    let merged = match f {
        F::FldsFt0 => {
            stx(e, FT0, Float32(v as u32).to_floatx80(s));
            true
        }
        F::FldlFt0 => {
            stx(e, FT0, Float64(v).to_floatx80(s));
            true
        }
        F::FildlFt0 => {
            stx(e, FT0, FloatX80::from_i32(v as i32, s));
            false
        }
        F::FldsSt0 => {
            let x = Float32(v as u32).to_floatx80(s);
            fpush(e);
            set(e, 0, x);
            true
        }
        F::FldlSt0 => {
            let x = Float64(v).to_floatx80(s);
            fpush(e);
            set(e, 0, x);
            true
        }
        F::FildlSt0 | F::FildllSt0 => {
            s.floatx80_rounding_precision = Prec::X;
            let x = if f == F::FildlSt0 {
                FloatX80::from_i32(v as i32, s)
            } else {
                FloatX80::from_i64(v as i64, s)
            };
            fpush(e);
            set(e, 0, x);
            false
        }
        F::FstsSt0 => {
            ret = u64::from(get(e, 0).to_float32(s).0);
            true
        }
        F::FstlSt0 => {
            ret = get(e, 0).to_float64(s).0;
            true
        }
        F::FistSt0 | F::FisttSt0 => {
            let x = get(e, 0);
            let mut val = if f == F::FistSt0 { x.to_i32(s) } else { x.to_i32_round_to_zero(s) };
            if val != i32::from(val as i16) {
                s.exception_flags = ff::INVALID;
                val = -32768;
            }
            ret = u64::from(val as u32);
            true
        }
        F::FistlSt0 | F::FisttlSt0 => {
            let x = get(e, 0);
            let val = if f == F::FistlSt0 { x.to_i32(s) } else { x.to_i32_round_to_zero(s) };
            ret = if s.flags() & ff::INVALID != 0 { 0x8000_0000 } else { u64::from(val as u32) };
            true
        }
        F::FistllSt0 | F::FisttllSt0 => {
            let x = get(e, 0);
            let val = if f == F::FistllSt0 { x.to_i64(s) } else { x.to_i64_round_to_zero(s) };
            ret = if s.flags() & ff::INVALID != 0 { 1 << 63 } else { val as u64 };
            true
        }
        F::Fpush => {
            fpush(e);
            false
        }
        F::Fpop => {
            fpop(e);
            false
        }
        F::Fdecstp | F::Fincstp => {
            let d = if f == F::Fdecstp { 7 } else { 1 };
            st32(e, FPSTT, ((top(e) + d) & 7) as u32);
            and_fpus(e, !0x4700);
            false
        }
        F::FfreeStn => {
            e[FPTAGS + ((top(e) + n) & 7)] = 1;
            false
        }
        F::FmovSt0Ft0 => {
            let x = ldx(e, FT0);
            set(e, 0, x);
            false
        }
        F::FmovFt0Stn => {
            let x = get(e, n);
            stx(e, FT0, x);
            false
        }
        F::FmovSt0Stn => {
            let x = get(e, n);
            set(e, 0, x);
            false
        }
        F::FmovStnSt0 => {
            let x = get(e, 0);
            set(e, n, x);
            false
        }
        F::FxchgSt0Stn => {
            let (a, b) = (get(e, 0), get(e, n));
            set(e, n, a);
            set(e, 0, b);
            false
        }
        F::FcomSt0Ft0 | F::FucomSt0Ft0 => {
            const FCOM_CCVAL: [u32; 4] = [0x0100, 0x4000, 0x0000, 0x4500];
            let (a, b) = (get(e, 0), ldx(e, FT0));
            let r = if f == F::FcomSt0Ft0 { a.compare(b, s) } else { a.compare_quiet(b, s) };
            st32(e, FPUS, (ld32(e, FPUS) & !0x4500) | FCOM_CCVAL[rel_idx(r)]);
            true
        }
        F::FcomiSt0Ft0 | F::FucomiSt0Ft0 => {
            const FCOMI_CCVAL: [u32; 4] = [CC_C, CC_Z, 0, CC_Z | CC_P | CC_C];
            let (a, b) = (get(e, 0), ldx(e, FT0));
            let r = if f == F::FcomiSt0Ft0 { a.compare(b, s) } else { a.compare_quiet(b, s) };
            let eflags = cc_compute_all(e) & !(CC_Z | CC_P | CC_C);
            st64(e, CC_SRC, u64::from(eflags | FCOMI_CCVAL[rel_idx(r)]));
            st32(e, CC_OP, CC_OP_EFLAGS);
            true
        }
        F::FaddSt0Ft0
        | F::FmulSt0Ft0
        | F::FsubSt0Ft0
        | F::FsubrSt0Ft0
        | F::FdivSt0Ft0
        | F::FdivrSt0Ft0 => {
            let (a, b) = (get(e, 0), ldx(e, FT0));
            let r = match f {
                F::FaddSt0Ft0 => a.add(b, s),
                F::FmulSt0Ft0 => a.mul(b, s),
                F::FsubSt0Ft0 => a.sub(b, s),
                F::FsubrSt0Ft0 => b.sub(a, s),
                F::FdivSt0Ft0 => a.div(b, s),
                _ => b.div(a, s),
            };
            set(e, 0, r);
            true
        }
        F::FaddStnSt0
        | F::FmulStnSt0
        | F::FsubStnSt0
        | F::FsubrStnSt0
        | F::FdivStnSt0
        | F::FdivrStnSt0 => {
            let (a, b) = (get(e, n), get(e, 0));
            let r = match f {
                F::FaddStnSt0 => a.add(b, s),
                F::FmulStnSt0 => a.mul(b, s),
                F::FsubStnSt0 => a.sub(b, s),
                F::FsubrStnSt0 => b.sub(a, s),
                F::FdivStnSt0 => a.div(b, s),
                _ => b.div(a, s),
            };
            set(e, n, r);
            true
        }
        F::Fchs => {
            let x = get(e, 0).chs();
            set(e, 0, x);
            false
        }
        F::Fabs => {
            let x = get(e, 0).abs();
            set(e, 0, x);
            false
        }
        F::Fld1 | F::Fldl2t | F::Fldl2e | F::Fldpi | F::Fldlg2 | F::Fldln2 | F::FldzSt0 => {
            let rc = ld32(e, FPUC) & FPU_RC_MASK;
            let down = rc == FPU_RC_DOWN || rc == FPU_RC_CHOP;
            let pick = |d: FloatX80, x: FloatX80| if down { d } else { x };
            let x = match f {
                F::Fld1 => ONE,
                F::Fldl2t => {
                    if rc == FPU_RC_UP {
                        L2T_U
                    } else {
                        L2T
                    }
                }
                F::Fldl2e => pick(L2E_D, L2E),
                F::Fldpi => pick(PI_D, PI),
                F::Fldlg2 => pick(LG2_D, LG2),
                F::Fldln2 => pick(LN2_D, LN2),
                _ => ZERO,
            };
            set(e, 0, x);
            false
        }
        F::FldzFt0 => {
            stx(e, FT0, ZERO);
            false
        }
        F::Fnstsw => {
            ret = u64::from(fnstsw(e));
            false
        }
        F::Fnstcw => {
            ret = u64::from(ld32(e, FPUC));
            false
        }
        F::Fldcw => {
            st32(e, FPUC, v as u32 & 0xffff);
            false
        }
        F::Fclex => {
            and_fpus(e, 0x7f00);
            false
        }
        F::Fninit => {
            fninit(e);
            false
        }
        F::F2xm1 => {
            f2xm1(e, s);
            true
        }
        F::Fyl2x => {
            fyl2x(e, s);
            true
        }
        F::Fyl2xp1 => {
            fyl2xp1(e, s);
            true
        }
        F::Fpatan => {
            fpatan(e, s);
            true
        }
        F::Fptan | F::Fsin | F::Fcos | F::Fsincos => {
            let t = f64::from_bits(get(e, 0).to_float64(s).0);
            if t > MAXTAN || t < -MAXTAN {
                or_fpus(e, 0x400);
            } else {
                let d = |x: f64, s: &mut FloatStatus| Float64(x.to_bits()).to_floatx80(s);
                match f {
                    F::Fptan => {
                        let x = d(t.tan(), s);
                        set(e, 0, x);
                        fpush(e);
                        set(e, 0, ONE);
                    }
                    F::Fsin => {
                        let x = d(t.sin(), s);
                        set(e, 0, x);
                    }
                    F::Fcos => {
                        let x = d(t.cos(), s);
                        set(e, 0, x);
                    }
                    _ => {
                        let x = d(t.sin(), s);
                        set(e, 0, x);
                        fpush(e);
                        let x = d(t.cos(), s);
                        set(e, 0, x);
                    }
                }
                and_fpus(e, !0x400);
            }
            false
        }
        F::Fxtract => {
            fxtract(e, s);
            true
        }
        F::Fprem1 | F::Fprem => {
            fprem(e, f == F::Fprem, s);
            true
        }
        F::Fsqrt => {
            let x = get(e, 0);
            if x.is_neg() {
                and_fpus(e, !0x4700);
                or_fpus(e, 0x400);
            }
            set(e, 0, x.sqrt(s));
            true
        }
        F::Frndint => {
            let x = get(e, 0).round_to_int(s);
            set(e, 0, x);
            true
        }
        F::Fscale => {
            fscale(e, s);
            true
        }
        F::Fxam => {
            fxam(e);
            false
        }
        F::FldtSt0
        | F::FsttSt0
        | F::FbldSt0
        | F::FbstSt0
        | F::Fwait
        | F::Fstenv
        | F::Fldenv
        | F::Fsave
        | F::Frstor => unreachable!("memory operations are handled by x87()"),
    };
    (ret, merged)
}

/// `helper_fbld_ST0()`.
fn fbld(cpu: &mut Cpu<'_>, ptr: u64, s: &mut FloatStatus) -> R<()> {
    let ac = Access::prepare(cpu, ptr, 10, MmuAccessType::DataLoad)?;
    let mut val: u64 = 0;
    for i in (0..9).rev() {
        let v = ac.ld(cpu, ptr.wrapping_add(i), MemOp::UB)?;
        val = val.wrapping_mul(100).wrapping_add((v >> 4) * 10).wrapping_add(v & 0xf);
    }
    let mut tmp = FloatX80::from_i64(val as i64, s);
    if ac.ld(cpu, ptr.wrapping_add(9), MemOp::UB)? & 0x80 != 0 {
        tmp = tmp.chs();
    }
    fpush(cpu.env);
    set(cpu.env, 0, tmp);
    Ok(())
}

/// `helper_fbst_ST0()`; the caller merges the flags.
fn fbst(cpu: &mut Cpu<'_>, ptr: u64, s: &mut FloatStatus) -> R<()> {
    let ac = Access::prepare(cpu, ptr, 10, MmuAccessType::DataStore)?;
    let x = get(cpu.env, 0);
    let mut val = x.to_i64(s);
    let at = |i: u64| ptr.wrapping_add(i);
    if !(-999_999_999_999_999_999..=999_999_999_999_999_999).contains(&val) {
        s.exception_flags = ff::INVALID;
        for i in 0..7 {
            ac.st(cpu, at(i), 0, MemOp::UB)?;
        }
        ac.st(cpu, at(7), 0xc0, MemOp::UB)?;
        ac.st(cpu, at(8), 0xff, MemOp::UB)?;
        ac.st(cpu, at(9), 0xff, MemOp::UB)?;
        return Ok(());
    }
    if sign_of(x) {
        ac.st(cpu, at(9), 0x80, MemOp::UB)?;
        val = -val;
    } else {
        ac.st(cpu, at(9), 0x00, MemOp::UB)?;
    }
    let mut i = 0;
    while i < 9 && val != 0 {
        let v = val % 100;
        val /= 100;
        ac.st(cpu, at(i), (((v / 10) << 4) | (v % 10)) as u64, MemOp::UB)?;
        i += 1;
    }
    while i < 9 {
        ac.st(cpu, at(i), 0, MemOp::UB)?;
        i += 1;
    }
    Ok(())
}

/// `do_fstenv()`.
fn fstenv(cpu: &mut Cpu<'_>, ac: &Access, ptr: u64, data32: usize) -> R<()> {
    let e = &*cpu.env;
    let mut tag = 0u32;
    for i in (0..8).rev() {
        tag <<= 2;
        if e[FPTAGS + i] != 0 {
            tag |= 3;
        } else {
            let x = ldx(e, fpreg(i));
            let exp = exp_of(x);
            if exp == 0 && x.low == 0 {
                tag |= 1;
            } else if exp == 0 || exp == 0x7fff || x.low >> 63 == 0 {
                tag |= 2;
            }
        }
    }
    let vals = [
        ld32(e, FPUC),
        fnstsw(e),
        tag,
        ld64(e, FPIP) as u32,
        ld32(e, FPCS),
        ld64(e, FPDP) as u32,
        ld32(e, FPDS),
    ];
    let (step, mop) = if data32 != 0 { (4, MemOp::LEUL) } else { (2, MemOp::LEUW) };
    let mut p = ptr;
    for v in vals {
        ac.st(cpu, p, u64::from(v), mop)?;
        p = p.wrapping_add(step);
    }
    Ok(())
}

/// `do_fldenv()`.
fn fldenv(cpu: &mut Cpu<'_>, ac: &Access, ptr: u64, data32: usize) -> R<()> {
    let fpuc = ac.ld(cpu, ptr, MemOp::LEUW)?;
    st32(cpu.env, FPUC, fpuc as u32);
    let fpus = ac.ld(cpu, ptr.wrapping_add(2 << data32), MemOp::LEUW)?;
    let tag = ac.ld(cpu, ptr.wrapping_add(4 << data32), MemOp::LEUW)?;
    set_fpus(cpu.env, fpus as u32);
    for i in 0..8 {
        cpu.env[FPTAGS + i] = u8::from((tag >> (2 * i)) & 3 == 3);
    }
    Ok(())
}

/// `helper_fxam_ST0()`.
fn fxam(e: &mut [u8]) {
    let x = get(e, 0);
    and_fpus(e, !0x4700);
    if sign_of(x) {
        or_fpus(e, 0x200);
    }
    if e[FPTAGS + top(e)] != 0 {
        or_fpus(e, 0x4100);
        return;
    }
    let exp = exp_of(x);
    let mant = x.low;
    if exp == 0x7fff {
        if mant == 1 << 63 {
            or_fpus(e, 0x500);
        } else if mant >> 63 != 0 {
            or_fpus(e, 0x100);
        }
    } else if exp == 0 {
        or_fpus(e, if mant == 0 { 0x4000 } else { 0x4400 });
    } else if mant >> 63 != 0 {
        or_fpus(e, 0x400);
    }
}

/// `helper_fxtract()`.
fn fxtract(e: &mut [u8], s: &mut FloatStatus) {
    let st0 = get(e, 0);
    if st0.is_zero() {
        // An easy way to make -inf and raise the division by zero exception.
        let x = ONE.chs().div(ZERO, s);
        set(e, 0, x);
        fpush(e);
        set(e, 0, st0);
    } else if st0.invalid_encoding(s) {
        s.raise(ff::INVALID);
        set(e, 0, FloatX80::default_nan(s));
        fpush(e);
        let x = get(e, 1);
        set(e, 0, x);
    } else if st0.is_any_nan() {
        if st0.is_signaling_nan(s) {
            s.raise(ff::INVALID);
            set(e, 0, st0.silence_nan(s));
        }
        fpush(e);
        let x = get(e, 1);
        set(e, 0, x);
    } else if st0.is_infinity(s) {
        fpush(e);
        let x = get(e, 1);
        set(e, 0, x);
        set(e, 1, FloatX80::default_inf(false, s));
    } else {
        let mut temp = st0;
        let expdif = if exp_of(temp) == 0 {
            let shift = clz(temp.low);
            temp.low = temp.low.wrapping_shl(shift as u32);
            s.raise(ff::INPUT_DENORMAL_FLUSHED);
            1 - 16383 - shift
        } else {
            exp_of(temp) - 16383
        };
        set(e, 0, FloatX80::from_i32(expdif, s));
        fpush(e);
        temp.high = (temp.high & 0x8000) | 16383;
        set(e, 0, temp);
    }
}

/// `helper_fprem_common()`.
fn fprem(e: &mut [u8], modulo: bool, s: &mut FloatStatus) {
    let (st0, st1) = (get(e, 0), get(e, 1));
    let (mut exp0, mut exp1) = (exp_of(st0), exp_of(st1));
    and_fpus(e, !0x4700);
    if st0.is_zero()
        || st1.is_zero()
        || exp0 == 0x7fff
        || exp1 == 0x7fff
        || st0.invalid_encoding(s)
        || st1.invalid_encoding(s)
    {
        let (r, _) = st0.modrem(st1, modulo, s);
        set(e, 0, r);
        return;
    }
    if exp0 == 0 {
        exp0 = 1 - clz(st0.low);
    }
    if exp1 == 0 {
        exp1 = 1 - clz(st1.low);
    }
    let expdiff = exp0 - exp1;
    if expdiff < 64 {
        let (r, q) = st0.modrem(st1, modulo, s);
        set(e, 0, r);
        let cc = ((q & 4) << 6) | ((q & 2) << 13) | ((q & 1) << 9);
        or_fpus(e, cc as u32);
    } else {
        // A partial remainder, as AMD documents and Intel does too.
        let n = 32 + expdiff % 32;
        let t1 = st1.scalbn(expdiff - n, s);
        let r = st0.modulo(t1, s);
        set(e, 0, r);
        or_fpus(e, 0x400);
    }
}

/// `helper_fscale()`.
fn fscale(e: &mut [u8], s: &mut FloatStatus) {
    let (st0, st1) = (get(e, 0), get(e, 1));
    let r = if st1.invalid_encoding(s) || st0.invalid_encoding(s) {
        s.raise(ff::INVALID);
        FloatX80::default_nan(s)
    } else if st1.is_any_nan() {
        if st0.is_signaling_nan(s) {
            s.raise(ff::INVALID);
        }
        if st1.is_signaling_nan(s) {
            s.raise(ff::INVALID);
            st1.silence_nan(s)
        } else {
            st1
        }
    } else if st1.is_infinity(s) && !st0.invalid_encoding(s) && !st0.is_any_nan() {
        if st1.is_neg() {
            if st0.is_infinity(s) {
                s.raise(ff::INVALID);
                FloatX80::default_nan(s)
            } else if st0.is_neg() {
                ZERO.chs()
            } else {
                ZERO
            }
        } else if st0.is_zero() {
            s.raise(ff::INVALID);
            FloatX80::default_nan(s)
        } else {
            FloatX80::default_inf(st0.is_neg(), s)
        }
    } else {
        let save = s.floatx80_rounding_precision;
        let save_flags = s.take_flags();
        let n = st1.to_i32_round_to_zero(s);
        s.exception_flags = save_flags;
        s.floatx80_rounding_precision = Prec::X;
        let r = st0.scalbn(n, s);
        s.floatx80_rounding_precision = save;
        r
    };
    set(e, 0, r);
}

// F2XM1.

/// `f2xm1_coeff_*`: a polynomial approximation of (2^x - 1) / x on [-1/64, 1/64].
const F2XM1_C0: FloatX80 = fx(0x3ffe, 0xb17217f7d1cf79ac);
const F2XM1_C0_LOW: FloatX80 = fx(0xbfbc, 0xd87edabf495b3762);
const F2XM1_C1: FloatX80 = fx(0x3ffc, 0xf5fdeffc162c7543);
const F2XM1_C2: FloatX80 = fx(0x3ffa, 0xe35846b82505fcc7);
const F2XM1_C3: FloatX80 = fx(0x3ff8, 0x9d955b7dd273b899);
const F2XM1_C4: FloatX80 = fx(0x3ff5, 0xaec3ff3c4ef4ac0c);
const F2XM1_C5: FloatX80 = fx(0x3ff2, 0xa184897c3a7f0de9);
const F2XM1_C6: FloatX80 = fx(0x3fee, 0xffe634d0ec30d504);
const F2XM1_C7: FloatX80 = fx(0x3feb, 0xb160111d2db515e4);
/// The 128 bit significand of ln(2).
const LN2_SIG_HIGH: u64 = 0xb17217f7d1cf79ab;
const LN2_SIG_LOW: u64 = 0xc9e3b39803f2f6af;

/// `f2xm1_table`: t, 2^t and 2^t - 1 for t = n/32 - 1, n in 0..=64.
const F2XM1_TABLE: [[FloatX80; 3]; 65] = [
    [
        fx(0xbfff, 0x8000000000000000),
        fx(0x3ffe, 0x8000000000000000),
        fx(0xbffe, 0x8000000000000000),
    ],
    [
        fx(0xbffe, 0xf800000000002e7e),
        fx(0x3ffe, 0x82cd8698ac2b9160),
        fx(0xbffd, 0xfa64f2cea7a8dd40),
    ],
    [
        fx(0xbffe, 0xefffffffffffe960),
        fx(0x3ffe, 0x85aac367cc488345),
        fx(0xbffd, 0xf4aa7930676ef976),
    ],
    [
        fx(0xbffe, 0xe800000000006f10),
        fx(0x3ffe, 0x88980e8092da5c14),
        fx(0xbffd, 0xeecfe2feda4b47d8),
    ],
    [
        fx(0xbffe, 0xe000000000008a45),
        fx(0x3ffe, 0x8b95c1e3ea8ba2a5),
        fx(0xbffd, 0xe8d47c382ae8bab6),
    ],
    [
        fx(0xbffe, 0xd7ffffffffff8a9e),
        fx(0x3ffe, 0x8ea4398b45cd8116),
        fx(0xbffd, 0xe2b78ce97464fdd4),
    ],
    [
        fx(0xbffe, 0xd0000000000019a0),
        fx(0x3ffe, 0x91c3d373ab11b919),
        fx(0xbffd, 0xdc785918a9dc8dce),
    ],
    [
        fx(0xbffe, 0xc7ffffffffff14df),
        fx(0x3ffe, 0x94f4efa8fef76836),
        fx(0xbffd, 0xd61620ae02112f94),
    ],
    [
        fx(0xbffe, 0xc000000000006530),
        fx(0x3ffe, 0x9837f0518db87fbb),
        fx(0xbffd, 0xcf901f5ce48f008a),
    ],
    [
        fx(0xbffe, 0xb7ffffffffff1723),
        fx(0x3ffe, 0x9b8d39b9d54eb74c),
        fx(0xbffd, 0xc8e58c8c55629168),
    ],
    [
        fx(0xbffe, 0xb00000000000b5e1),
        fx(0x3ffe, 0x9ef5326091a0c366),
        fx(0xbffd, 0xc2159b3edcbe7934),
    ],
    [
        fx(0xbffe, 0xa800000000006f8a),
        fx(0x3ffe, 0xa27043030c49370a),
        fx(0xbffd, 0xbb1f79f9e76d91ec),
    ],
    [
        fx(0xbffe, 0x9fffffffffff816a),
        fx(0x3ffe, 0xa5fed6a9b15171cf),
        fx(0xbffd, 0xb40252ac9d5d1c62),
    ],
    [
        fx(0xbffe, 0x97ffffffffffb621),
        fx(0x3ffe, 0xa9a15ab4ea7c30e6),
        fx(0xbffd, 0xacbd4a962b079e34),
    ],
    [
        fx(0xbffe, 0x8fffffffffff162b),
        fx(0x3ffe, 0xad583eea42a1b886),
        fx(0xbffd, 0xa54f822b7abc8ef4),
    ],
    [
        fx(0xbffe, 0x87ffffffffff4d34),
        fx(0x3ffe, 0xb123f581d2ac7b51),
        fx(0xbffd, 0x9db814fc5aa7095e),
    ],
    [
        fx(0xbffe, 0x800000000000227d),
        fx(0x3ffe, 0xb504f333f9de539d),
        fx(0xbffd, 0x95f619980c4358c6),
    ],
    [
        fx(0xbffd, 0xefffffffffff3978),
        fx(0x3ffe, 0xb8fbaf4762fbd0a1),
        fx(0xbffd, 0x8e08a1713a085ebe),
    ],
    [
        fx(0xbffd, 0xe00000000000df81),
        fx(0x3ffe, 0xbd08a39f580bfd8c),
        fx(0xbffd, 0x85eeb8c14fe804e8),
    ],
    [
        fx(0xbffd, 0xd00000000000bccf),
        fx(0x3ffe, 0xc12c4cca667062f6),
        fx(0xbffc, 0xfb4eccd6663e7428),
    ],
    [
        fx(0xbffd, 0xc00000000000eff0),
        fx(0x3ffe, 0xc5672a1155069abe),
        fx(0xbffc, 0xea6357baabe59508),
    ],
    [
        fx(0xbffd, 0xb000000000000fe6),
        fx(0x3ffe, 0xc9b9bd866e2f234b),
        fx(0xbffc, 0xd91909e6474372d4),
    ],
    [
        fx(0xbffd, 0x9fffffffffff2172),
        fx(0x3ffe, 0xce248c151f84bf00),
        fx(0xbffc, 0xc76dcfab81ed0400),
    ],
    [
        fx(0xbffd, 0x8fffffffffffafff),
        fx(0x3ffe, 0xd2a81d91f12afb2b),
        fx(0xbffc, 0xb55f89b83b541354),
    ],
    [
        fx(0xbffc, 0xffffffffffff81a3),
        fx(0x3ffe, 0xd744fccad69d7d5e),
        fx(0xbffc, 0xa2ec0cd4a58a0a88),
    ],
    [
        fx(0xbffc, 0xdfffffffffff1568),
        fx(0x3ffe, 0xdbfbb797daf25a44),
        fx(0xbffc, 0x901121a0943696f0),
    ],
    [
        fx(0xbffc, 0xbfffffffffff68da),
        fx(0x3ffe, 0xe0ccdeec2a94f811),
        fx(0xbffb, 0xf999089eab583f78),
    ],
    [
        fx(0xbffc, 0x9fffffffffff4690),
        fx(0x3ffe, 0xe5b906e77c83657e),
        fx(0xbffb, 0xd237c8c41be4d410),
    ],
    [
        fx(0xbffb, 0xffffffffffff8aee),
        fx(0x3ffe, 0xeac0c6e7dd24427c),
        fx(0xbffb, 0xa9f9c8c116ddec20),
    ],
    [
        fx(0xbffb, 0xbfffffffffff2d18),
        fx(0x3ffe, 0xefe4b99bdcdb06eb),
        fx(0xbffb, 0x80da33211927c8a8),
    ],
    [
        fx(0xbffa, 0xffffffffffff8ccb),
        fx(0x3ffe, 0xf5257d152486d0f4),
        fx(0xbffa, 0xada82eadb792f0c0),
    ],
    [
        fx(0xbff9, 0xffffffffffff11fe),
        fx(0x3ffe, 0xfa83b2db722a0846),
        fx(0xbff9, 0xaf89a491babef740),
    ],
    [
        fx(0x0000, 0x0000000000000000),
        fx(0x3fff, 0x8000000000000000),
        fx(0x0000, 0x0000000000000000),
    ],
    [
        fx(0x3ff9, 0xffffffffffff2680),
        fx(0x3fff, 0x82cd8698ac2b9f6f),
        fx(0x3ff9, 0xb361a62b0ae7dbc0),
    ],
    [
        fx(0x3ffb, 0x800000000000b500),
        fx(0x3fff, 0x85aac367cc488345),
        fx(0x3ffa, 0xb5586cf9891068a0),
    ],
    [
        fx(0x3ffb, 0xbfffffffffff4b67),
        fx(0x3fff, 0x88980e8092da7cce),
        fx(0x3ffb, 0x8980e8092da7cce0),
    ],
    [
        fx(0x3ffb, 0xffffffffffffff57),
        fx(0x3fff, 0x8b95c1e3ea8bd6df),
        fx(0x3ffb, 0xb95c1e3ea8bd6df0),
    ],
    [
        fx(0x3ffc, 0x9fffffffffff811f),
        fx(0x3fff, 0x8ea4398b45cd4780),
        fx(0x3ffb, 0xea4398b45cd47800),
    ],
    [
        fx(0x3ffc, 0xbfffffffffff9980),
        fx(0x3fff, 0x91c3d373ab11b919),
        fx(0x3ffc, 0x8e1e9b9d588dc8c8),
    ],
    [
        fx(0x3ffc, 0xdffffffffffff631),
        fx(0x3fff, 0x94f4efa8fef70864),
        fx(0x3ffc, 0xa7a77d47f7b84320),
    ],
    [
        fx(0x3ffc, 0xffffffffffff2499),
        fx(0x3fff, 0x9837f0518db892d4),
        fx(0x3ffc, 0xc1bf828c6dc496a0),
    ],
    [
        fx(0x3ffd, 0x8fffffffffff80fb),
        fx(0x3fff, 0x9b8d39b9d54e3a79),
        fx(0x3ffc, 0xdc69cdceaa71d3c8),
    ],
    [
        fx(0x3ffd, 0x9fffffffffffbc23),
        fx(0x3fff, 0x9ef5326091a10313),
        fx(0x3ffc, 0xf7a993048d081898),
    ],
    [
        fx(0x3ffd, 0xafffffffffff20ec),
        fx(0x3fff, 0xa27043030c49370a),
        fx(0x3ffd, 0x89c10c0c3124dc28),
    ],
    [
        fx(0x3ffd, 0xc00000000000fd2c),
        fx(0x3fff, 0xa5fed6a9b15171cf),
        fx(0x3ffd, 0x97fb5aa6c545c73c),
    ],
    [
        fx(0x3ffd, 0xd0000000000093be),
        fx(0x3fff, 0xa9a15ab4ea7c30e6),
        fx(0x3ffd, 0xa6856ad3a9f0c398),
    ],
    [
        fx(0x3ffd, 0xe00000000000c2ae),
        fx(0x3fff, 0xad583eea42a17876),
        fx(0x3ffd, 0xb560fba90a85e1d8),
    ],
    [
        fx(0x3ffd, 0xefffffffffff1e3f),
        fx(0x3fff, 0xb123f581d2abef6c),
        fx(0x3ffd, 0xc48fd6074aafbdb0),
    ],
    [
        fx(0x3ffd, 0xffffffffffff1c23),
        fx(0x3fff, 0xb504f333f9de2cad),
        fx(0x3ffd, 0xd413cccfe778b2b4),
    ],
    [
        fx(0x3ffe, 0x8800000000006344),
        fx(0x3fff, 0xb8fbaf4762fbd0a1),
        fx(0x3ffd, 0xe3eebd1d8bef4284),
    ],
    [
        fx(0x3ffe, 0x9000000000005d67),
        fx(0x3fff, 0xbd08a39f580c668d),
        fx(0x3ffd, 0xf4228e7d60319a34),
    ],
    [
        fx(0x3ffe, 0x9800000000009127),
        fx(0x3fff, 0xc12c4cca6670e042),
        fx(0x3ffe, 0x82589994cce1c084),
    ],
    [
        fx(0x3ffe, 0x9fffffffffff06f9),
        fx(0x3fff, 0xc5672a11550655c3),
        fx(0x3ffe, 0x8ace5422aa0cab86),
    ],
    [
        fx(0x3ffe, 0xa7fffffffffff80d),
        fx(0x3fff, 0xc9b9bd866e2f234b),
        fx(0x3ffe, 0x93737b0cdc5e4696),
    ],
    [
        fx(0x3ffe, 0xafffffffffff1470),
        fx(0x3fff, 0xce248c151f83fd69),
        fx(0x3ffe, 0x9c49182a3f07fad2),
    ],
    [
        fx(0x3ffe, 0xb800000000000e0a),
        fx(0x3fff, 0xd2a81d91f12aec5c),
        fx(0x3ffe, 0xa5503b23e255d8b8),
    ],
    [
        fx(0x3ffe, 0xc00000000000b7fa),
        fx(0x3fff, 0xd744fccad69dd630),
        fx(0x3ffe, 0xae89f995ad3bac60),
    ],
    [
        fx(0x3ffe, 0xc800000000003aa6),
        fx(0x3fff, 0xdbfbb797daf25a44),
        fx(0x3ffe, 0xb7f76f2fb5e4b488),
    ],
    [
        fx(0x3ffe, 0xd00000000000a6ae),
        fx(0x3fff, 0xe0ccdeec2a954685),
        fx(0x3ffe, 0xc199bdd8552a8d0a),
    ],
    [
        fx(0x3ffe, 0xd800000000004165),
        fx(0x3fff, 0xe5b906e77c837155),
        fx(0x3ffe, 0xcb720dcef906e2aa),
    ],
    [
        fx(0x3ffe, 0xe00000000000582c),
        fx(0x3fff, 0xeac0c6e7dd24713a),
        fx(0x3ffe, 0xd5818dcfba48e274),
    ],
    [
        fx(0x3ffe, 0xe800000000001a5d),
        fx(0x3fff, 0xefe4b99bdcdb06eb),
        fx(0x3ffe, 0xdfc97337b9b60dd6),
    ],
    [
        fx(0x3ffe, 0xefffffffffffc1ef),
        fx(0x3fff, 0xf5257d152486a2fa),
        fx(0x3ffe, 0xea4afa2a490d45f4),
    ],
    [
        fx(0x3ffe, 0xf800000000001069),
        fx(0x3fff, 0xfa83b2db722a0e5c),
        fx(0x3ffe, 0xf50765b6e4541cb8),
    ],
    [
        fx(0x3fff, 0x8000000000000000),
        fx(0x4000, 0x8000000000000000),
        fx(0x3fff, 0x8000000000000000),
    ],
];

/// `helper_f2xm1()`.
fn f2xm1(e: &mut [u8], s: &mut FloatStatus) {
    let st0 = get(e, 0);
    let (mut sig, mut exp, sign) = parts(st0);
    if st0.invalid_encoding(s) {
        s.raise(ff::INVALID);
        set(e, 0, FloatX80::default_nan(s));
    } else if st0.is_any_nan() {
        if st0.is_signaling_nan(s) {
            s.raise(ff::INVALID);
            set(e, 0, st0.silence_nan(s));
        }
    } else if exp > 0x3fff || (exp == 0x3fff && sig != 1 << 63) {
        // Out of range for the instruction, treat as invalid.
        s.raise(ff::INVALID);
        set(e, 0, FloatX80::default_nan(s));
    } else if exp == 0x3fff {
        // Argument 1 or -1, exact result 1 or -0.5.
        if sign {
            set(e, 0, fx(0xbffe, 1 << 63));
        }
    } else if exp < 0x3fb0 {
        if !st0.is_zero() {
            // Multiplying the argument by an extra precision version of ln(2) is precise
            // enough. Zero arguments are returned unchanged.
            if exp == 0 {
                (exp, sig) = norm_sub(sig);
            }
            let (sig0, sig1, _) = mul128_by64(LN2_SIG_HIGH, LN2_SIG_LOW, sig);
            let r = FloatX80::normalize_round_and_pack(Prec::X, sign, exp, sig0, sig1 | 1, s);
            set(e, 0, r);
        }
    } else {
        let save_mode = s.rounding_mode;
        let save_prec = s.floatx80_rounding_precision;
        s.rounding_mode = RoundMode::NearestEven;
        s.floatx80_rounding_precision = Prec::X;

        // Find the nearest multiple of 1/32 to the argument.
        let tmp = st0.scalbn(5, s);
        let n = (32 + tmp.to_i32(s)) as usize;
        let [t, exp2, exp2m1] = F2XM1_TABLE[n];
        let y = st0.sub(t, s);

        if y.is_zero() {
            // QEMU stores t here, although its comment says 2^t - 1 is meant.
            set(e, 0, t);
            s.exception_flags = ff::INEXACT;
            s.rounding_mode = save_mode;
        } else {
            // The lower parts of a polynomial expansion of (2^y - 1) / y.
            let mut accum = F2XM1_C7.mul(y, s);
            for c in [F2XM1_C6, F2XM1_C5, F2XM1_C4, F2XM1_C3, F2XM1_C2, F2XM1_C1] {
                accum = c.add(accum, s);
                accum = accum.mul(y, s);
            }
            accum = F2XM1_C0_LOW.add(accum, s);

            // The full expansion is F2XM1_C0 + accum, and accum is much smaller, so the
            // addition cannot carry out.
            let mut aexp = exp_of(F2XM1_C0);
            let mut asign = sign_of(F2XM1_C0);
            let (mut asig0, mut asig1) = shr_jam(accum.low, 0, aexp - exp_of(accum));
            (asig0, asig1) = if asign == sign_of(accum) {
                add128(F2XM1_C0.low, 0, asig0, asig1)
            } else {
                sub128(F2XM1_C0.low, 0, asig0, asig1)
            };
            // And so an approximation to 2^y - 1.
            (asig0, asig1, _) = mul128_by64(asig0, asig1, y.low);
            aexp += exp_of(y) - 0x3ffe;
            asign ^= sign_of(y);
            if n != 32 {
                // Multiply by 2^t and add 2^t - 1.
                (asig0, asig1, _) = mul128_by64(asig0, asig1, exp2.low);
                aexp += exp_of(exp2) - 0x3ffe;
                let bexp = exp_of(exp2m1);
                let (mut bsig0, mut bsig1) = (exp2m1.low, 0);
                if bexp < aexp {
                    (bsig0, bsig1) = shr_jam(bsig0, bsig1, aexp - bexp);
                } else if aexp < bexp {
                    (asig0, asig1) = shr_jam(asig0, asig1, bexp - aexp);
                    aexp = bexp;
                }
                // The sign of 2^t - 1 is always that of the result.
                let bsign = sign_of(exp2m1);
                if asign == bsign {
                    // Avoid a carry out of the addition.
                    (asig0, asig1) = shr_jam(asig0, asig1, 1);
                    (bsig0, bsig1) = shr_jam(bsig0, bsig1, 1);
                    aexp += 1;
                    (asig0, asig1) = add128(asig0, asig1, bsig0, bsig1);
                } else {
                    (asig0, asig1) = sub128(bsig0, bsig1, asig0, asig1);
                    asign = bsign;
                }
            }
            s.rounding_mode = save_mode;
            // The result is inexact.
            let r = FloatX80::normalize_round_and_pack(Prec::X, asign, aexp, asig0, asig1 | 1, s);
            set(e, 0, r);
        }
        s.floatx80_rounding_precision = save_prec;
    }
}

// FPATAN.

/// pi/4, pi/2, 3pi/4 and pi to 128 bits, as exponent and significand.
const PI_4: (i32, u64, u64) = (0x3ffe, 0xc90fdaa22168c234, 0xc4c6628b80dc1cd1);
const PI_2: (i32, u64, u64) = (0x3fff, 0xc90fdaa22168c234, 0xc4c6628b80dc1cd1);
const PI_34: (i32, u64, u64) = (0x4000, 0x96cbe3f9990e91a7, 0x9394c9e8a0a5159d);
const PI_128: (i32, u64, u64) = (0x4000, 0xc90fdaa22168c234, 0xc4c6628b80dc1cd1);

/// `fpatan_coeff_*`: odd powers of an approximation to atan(x) on [-1/16, 1/16].
const FPATAN_C0: FloatX80 = fx(0x3fff, 0x8000000000000000);
const FPATAN_C1: FloatX80 = fx(0xbffd, 0xaaaaaaaaaaaaaa43);
const FPATAN_C2: FloatX80 = fx(0x3ffc, 0xccccccccccbfe4f8);
const FPATAN_C3: FloatX80 = fx(0xbffc, 0x92492491fbab2e66);
const FPATAN_C4: FloatX80 = fx(0x3ffb, 0xe38e372881ea1e0b);
const FPATAN_C5: FloatX80 = fx(0xbffb, 0xba2c0104bbdd0615);
const FPATAN_C6: FloatX80 = fx(0x3ffb, 0x9baf7ebf898b42ef);

/// `fpatan_table`: the high and low parts of atan(n/8), n in 0..=8.
#[rustfmt::skip]
const FPATAN_TABLE: [[FloatX80; 2]; 9] = [
    [fx(0x0000, 0x0000000000000000), fx(0x0000, 0x0000000000000000)],
    [fx(0x3ffb, 0xfeadd4d5617b6e33), fx(0xbfb9, 0xdda19d8305ddc420)],
    [fx(0x3ffc, 0xfadbafc96406eb15), fx(0x3fbb, 0xdb8f3debef442fcc)],
    [fx(0x3ffd, 0xb7b0ca0f26f78474), fx(0xbfbc, 0xeab9bdba460376fa)],
    [fx(0x3ffd, 0xed63382b0dda7b45), fx(0x3fbc, 0xdfc88bd978751a06)],
    [fx(0x3ffe, 0x8f005d5ef7f59f9b), fx(0x3fbd, 0xb906bc2ccb886e90)],
    [fx(0x3ffe, 0xa4bc7d1934f70924), fx(0x3fbb, 0xcd43f9522bed64f8)],
    [fx(0x3ffe, 0xb8053e2bc2319e74), fx(0xbfbc, 0xd3496ab7bd6eef0c)],
    [fx(0x3ffe, 0xc90fdaa22168c235), fx(0xbfbc, 0xece675d1fc8f8cbc)],
];

/// `helper_fpatan()`.
fn fpatan(e: &mut [u8], s: &mut FloatStatus) {
    let (st0, st1) = (get(e, 0), get(e, 1));
    let (_, arg0_exp, arg0_sign) = parts(st0);
    let (_, arg1_exp, arg1_sign) = parts(st1);
    if st0.invalid_encoding(s) || st1.invalid_encoding(s) {
        s.raise(ff::INVALID);
        set(e, 1, FloatX80::default_nan(s));
    } else if st0.is_signaling_nan(s) {
        s.raise(ff::INVALID);
        set(e, 1, st0.silence_nan(s));
    } else if st1.is_signaling_nan(s) {
        s.raise(ff::INVALID);
        set(e, 1, st1.silence_nan(s));
    } else if st0.is_any_nan() {
        set(e, 1, st0);
    } else if st1.is_any_nan() || (st1.is_zero() && !arg0_sign) {
        // Pass this NaN or zero through.
    } else if ((st0.is_infinity(s) && !st1.is_infinity(s)) || arg0_exp - arg1_exp >= 80)
        && !arg0_sign
    {
        // Dividing ST1 by ST0 gives the correct result up to rounding, but if a finite
        // nonzero result of the division is exact, the result of fpatan is still inexact
        // (and underflowing where appropriate).
        let save_prec = s.floatx80_rounding_precision;
        s.floatx80_rounding_precision = Prec::X;
        let mut r = st1.div(st0, s);
        s.floatx80_rounding_precision = save_prec;
        if !r.is_zero() && s.flags() & ff::INEXACT == 0 {
            // The mathematical result is very slightly closer to zero than this.
            let (mut sig, mut exp, sign) = parts(r);
            if exp == 0 {
                (exp, sig) = norm_sub(sig);
            }
            r = FloatX80::normalize_round_and_pack(
                Prec::X,
                sign,
                exp,
                sig.wrapping_sub(1),
                u64::MAX,
                s,
            );
        }
        set(e, 1, r);
    } else {
        // The result is inexact.
        let (rexp, rsig0, rsig1) = if st1.is_zero() {
            // ST0 is negative. The result is pi with the sign of ST1.
            PI_128
        } else if st1.is_infinity(s) {
            if st0.is_infinity(s) { if arg0_sign { PI_34 } else { PI_4 } } else { PI_2 }
        } else if st0.is_zero() || arg1_exp - arg0_exp >= 80 {
            PI_2
        } else if st0.is_infinity(s) || arg0_exp - arg1_exp >= 80 {
            // ST0 is negative.
            PI_128
        } else {
            fpatan_finite(st0, st1, s)
        };
        let r = FloatX80::normalize_round_and_pack(Prec::X, arg1_sign, rexp, rsig0, rsig1 | 1, s);
        set(e, 1, r);
    }
    fpop(e);
}

/// The general case of `helper_fpatan()`: ST0 and ST1 are finite, nonzero and with
/// exponents not too far apart. Returns the exponent and 128 bit significand of the result.
fn fpatan_finite(st0: FloatX80, st1: FloatX80, s: &mut FloatStatus) -> (i32, u64, u64) {
    let (mut arg0_sig, mut arg0_exp, arg0_sign) = parts(st0);
    let (mut arg1_sig, mut arg1_exp, _) = parts(st1);
    let save_mode = s.rounding_mode;
    let save_prec = s.floatx80_rounding_precision;
    s.rounding_mode = RoundMode::NearestEven;
    s.floatx80_rounding_precision = Prec::X;

    if arg0_exp == 0 {
        (arg0_exp, arg0_sig) = norm_sub(arg0_sig);
    }
    if arg1_exp == 0 {
        (arg1_exp, arg1_sig) = norm_sub(arg1_sig);
    }
    let (num_exp, num_sig, den_sig, adj, adj_sub);
    let den_exp;
    if arg0_exp > arg1_exp || (arg0_exp == arg1_exp && arg0_sig >= arg1_sig) {
        // Work with abs(ST1) / abs(ST0).
        (num_exp, num_sig, den_exp, den_sig) = (arg1_exp, arg1_sig, arg0_exp, arg0_sig);
        if arg0_sign {
            // The result is subtracted from pi.
            (adj, adj_sub) = (PI_128, true);
        } else {
            // The result is used as is.
            (adj, adj_sub) = ((0, 0, 0), false);
        }
    } else {
        // Work with abs(ST0) / abs(ST1).
        (num_exp, num_sig, den_exp, den_sig) = (arg0_exp, arg0_sig, arg1_exp, arg1_sig);
        // The result is added to or subtracted from pi/2.
        (adj, adj_sub) = (PI_2, !arg0_sign);
    }

    // Compute x = num/den, where 0 < x <= 1 and x is not too small.
    let mut xexp = num_exp - den_exp + 0x3ffe;
    let (mut remsig0, mut remsig1) = (num_sig, 0);
    if den_sig <= remsig0 {
        (remsig0, remsig1) = shr128(remsig0, remsig1, 1);
        xexp += 1;
    }
    let mut xsig0 = est_div(remsig0, remsig1, den_sig);
    let (msig0, msig1) = mul64(den_sig, xsig0);
    (remsig0, remsig1) = sub128(remsig0, remsig1, msig0, msig1);
    while (remsig0 as i64) < 0 {
        xsig0 = xsig0.wrapping_sub(1);
        (remsig0, remsig1) = add128(remsig0, remsig1, 0, den_sig);
    }
    // No need to correct any estimation error in xsig1; it is accurate enough.
    let xsig1 = est_div(remsig1, 0, den_sig);

    // Split x as x = t + y, where t = n/8 is the nearest multiple of 1/8 to x.
    let x8 = FloatX80::normalize_round_and_pack(Prec::X, false, xexp + 3, xsig0, xsig1, s);
    let n = x8.to_i32(s);
    let (ysign, yexp, ysig0, ysig1, texp, tsig);
    if n == 0 {
        (ysign, yexp, ysig0, ysig1, texp, tsig) = (false, xexp, xsig0, xsig1, 0, 0);
    } else {
        let shift = (n as u32).leading_zeros() as i32 + 32;
        texp = 0x403b - shift;
        tsig = (n as u64).wrapping_shl(shift as u32);
        if texp == xexp {
            let (y0, y1) = sub128(xsig0, xsig1, tsig, 0);
            if (y0 as i64) >= 0 {
                ysign = false;
                if y0 == 0 {
                    if y1 == 0 {
                        (yexp, ysig0, ysig1) = (0, y0, y1);
                    } else {
                        let shift = clz(y1) + 64;
                        yexp = xexp - shift;
                        (ysig0, ysig1) = shl128(y0, y1, shift);
                    }
                } else {
                    let shift = clz(y0);
                    yexp = xexp - shift;
                    (ysig0, ysig1) = shl128(y0, y1, shift);
                }
            } else {
                ysign = true;
                let (y0, y1) = sub128(0, 0, y0, y1);
                let shift = if y0 == 0 { clz(y1) + 64 } else { clz(y0) };
                yexp = xexp - shift;
                (ysig0, ysig1) = shl128(y0, y1, shift);
            }
        } else {
            // t's exponent must be greater than x's because t is positive and the nearest
            // multiple of 1/8 to x, and if x has a greater exponent, the power of 2 with
            // that exponent is also a multiple of 1/8.
            let (u0, u1) = shr_jam(xsig0, xsig1, texp - xexp);
            ysign = true;
            let (y0, y1) = sub128(tsig, 0, u0, u1);
            let shift = if y0 == 0 { clz(y1) + 64 } else { clz(y0) };
            yexp = texp - shift;
            (ysig0, ysig1) = shl128(y0, y1, shift);
        }
    }

    // Compute z = y/(1+tx), so arctan(x) = arctan(t) + arctan(z).
    let zsign = ysign;
    let (zexp, zsig0, zsig1);
    if texp == 0 || yexp == 0 {
        (zexp, zsig0, zsig1) = (yexp, ysig0, ysig1);
    } else {
        // t <= 1, x <= 1 and if both are 1 then y is 0, so tx < 1.
        let dexp = texp + xexp - 0x3ffe;
        let (d0, d1, _) = mul128_by64(xsig0, xsig1, tsig);
        // dexp <= 0x3fff (and if equal, d0 has a leading 0 bit). Add 1 to make the
        // denominator 1+tx.
        let (mut d0, d1) = shr_jam(d0, d1, 0x3fff - dexp);
        d0 |= 1 << 63;
        let mut ze = yexp - 1;
        let (mut r0, mut r1, mut r2) = (ysig0, ysig1, 0);
        if d0 <= r0 {
            (r0, r1) = shr128(r0, r1, 1);
            ze += 1;
        }
        let mut z0 = est_div(r0, r1, d0);
        let (m0, m1, m2) = mul128_by64(d0, d1, z0);
        (r0, r1, r2) = sub192(r0, r1, r2, m0, m1, m2);
        while (r0 as i64) < 0 {
            z0 = z0.wrapping_sub(1);
            (r0, r1, r2) = add192(r0, r1, r2, 0, d0, d1);
        }
        // No need to correct any estimation error in the low half.
        (zexp, zsig0, zsig1) = (ze, z0, est_div(r1, r2, d0));
    }

    let (azexp, azsig0, azsig1);
    if zexp == 0 {
        (azexp, azsig0, azsig1) = (0, 0, 0);
    } else {
        // Compute z^2.
        let (z2sig0, z2sig1, _, _) = mul128_to256(zsig0, zsig1, zsig0, zsig1);
        let z2 = FloatX80::normalize_round_and_pack(
            Prec::X,
            false,
            zexp + zexp - 0x3ffe,
            z2sig0,
            z2sig1,
            s,
        );
        // Compute the lower parts of the polynomial expansion.
        let mut accum = FPATAN_C6.mul(z2, s);
        for c in [FPATAN_C5, FPATAN_C4, FPATAN_C3, FPATAN_C2, FPATAN_C1] {
            accum = c.add(accum, s);
            accum = accum.mul(z2, s);
        }
        // The full expansion is z*(FPATAN_C0 + accum). FPATAN_C0 is 1, and accum is
        // negative and much smaller.
        let aexp = exp_of(FPATAN_C0);
        let (a0, a1) = shr_jam(accum.low, 0, aexp - exp_of(accum));
        let (a0, a1) = sub128(FPATAN_C0.low, 0, a0, a1);
        // Multiply by z to compute arctan(z).
        azexp = aexp + zexp - 0x3ffe;
        (azsig0, azsig1, _, _) = mul128_to256(a0, a1, zsig0, zsig1);
    }

    // Add arctan(t) (positive or zero) and arctan(z) (sign zsign).
    let (mut axexp, mut axsig0, mut axsig1);
    if texp == 0 {
        // z is positive.
        (axexp, axsig0, axsig1) = (azexp, azsig0, azsig1);
    } else {
        let [hi, lo] = FPATAN_TABLE[n as usize];
        let (l0, l1) = shr_jam(lo.low, 0, exp_of(hi) - exp_of(lo));
        axexp = exp_of(hi);
        (axsig0, axsig1) =
            if sign_of(lo) { sub128(hi.low, 0, l0, l1) } else { add128(hi.low, 0, l0, l1) };
        let (mut az0, mut az1) = (azsig0, azsig1);
        if azexp >= axexp {
            (axsig0, axsig1) = shr_jam(axsig0, axsig1, azexp - axexp + 1);
            axexp = azexp + 1;
            (az0, az1) = shr_jam(az0, az1, 1);
        } else {
            (axsig0, axsig1) = shr_jam(axsig0, axsig1, 1);
            (az0, az1) = shr_jam(az0, az1, axexp - azexp + 1);
            axexp += 1;
        }
        (axsig0, axsig1) =
            if zsign { sub128(axsig0, axsig1, az0, az1) } else { add128(axsig0, axsig1, az0, az1) };
    }

    let r = if adj.0 == 0 {
        (axexp, axsig0, axsig1)
    } else {
        // Add or subtract arctan(x) (positive, not necessarily normalized) to the adjustment.
        let (adj_exp, mut ad0, mut ad1) = adj;
        let rexp;
        if adj_exp >= axexp {
            (axsig0, axsig1) = shr_jam(axsig0, axsig1, adj_exp - axexp + 1);
            rexp = adj_exp + 1;
            (ad0, ad1) = shr_jam(ad0, ad1, 1);
        } else {
            (axsig0, axsig1) = shr_jam(axsig0, axsig1, 1);
            (ad0, ad1) = shr_jam(ad0, ad1, axexp - adj_exp + 1);
            rexp = axexp + 1;
        }
        let (r0, r1) = if adj_sub {
            sub128(ad0, ad1, axsig0, axsig1)
        } else {
            add128(ad0, ad1, axsig0, axsig1)
        };
        (rexp, r0, r1)
    };
    s.rounding_mode = save_mode;
    s.floatx80_rounding_precision = save_prec;
    r
}

// FYL2X and FYL2XP1.

/// The 128 bit significand of log2(e).
const LOG2_E_SIG_HIGH: u64 = 0xb8aa3b295c17f0bb;
const LOG2_E_SIG_LOW: u64 = 0xbe87fed0691d3e89;

/// `fyl2x_coeff_*`: odd powers of an approximation to log2((1+x)/(1-x)) on
/// [2*sqrt(2)-3, 3-2*sqrt(2)].
const FYL2X_C0: FloatX80 = fx(0x4000, 0xb8aa3b295c17f0bc);
const FYL2X_C0_LOW: FloatX80 = fx(0xbfbf, 0x834972fe2d7bab1b);
const FYL2X_C1: FloatX80 = fx(0x3ffe, 0xf6384ee1d01febb8);
const FYL2X_C2: FloatX80 = fx(0x3ffe, 0x93bb62877cdfa2e3);
const FYL2X_C3: FloatX80 = fx(0x3ffd, 0xd30bb153d808f269);
const FYL2X_C4: FloatX80 = fx(0x3ffd, 0xa42589eaf451499e);
const FYL2X_C5: FloatX80 = fx(0x3ffd, 0x864d42c0f8f17517);
const FYL2X_C6: FloatX80 = fx(0x3ffc, 0xe3476578adf26272);
const FYL2X_C7: FloatX80 = fx(0x3ffc, 0xc506c5f874e6d80f);
const FYL2X_C8: FloatX80 = fx(0x3ffc, 0xac5cf50cc57d6372);
const FYL2X_C9: FloatX80 = fx(0x3ffc, 0xb1ed0066d971a103);

/// `helper_fyl2x_common()`: an approximation of log2(1+arg) for 1+arg in
/// [sqrt(2)/2, sqrt(2)], with round to nearest and 80 bit precision in effect. `arg` must
/// not be zero or so close to zero that underflow might occur.
fn fyl2x_common(arg: FloatX80, s: &mut FloatStatus) -> (i32, u64, u64) {
    let (arg0_sig, arg0_exp, arg0_sign) = parts(arg);

    // Compute arg/(2+arg) with extra precision, as the argument to the polynomial.
    let (dexp, dsig0, dsig1);
    if arg0_sign {
        dexp = 0x3fff;
        let (a, b) = shr_jam(arg0_sig, 0, dexp - arg0_exp);
        (dsig0, dsig1) = sub128(0, 0, a, b);
    } else {
        dexp = 0x4000;
        let (a, b) = shr_jam(arg0_sig, 0, dexp - arg0_exp);
        (dsig0, dsig1) = (a | 1 << 63, b);
    }
    let mut texp = arg0_exp - dexp + 0x3ffe;
    let (mut r0, mut r1, mut r2) = (arg0_sig, 0, 0);
    if dsig0 <= r0 {
        (r0, r1) = shr128(r0, r1, 1);
        texp += 1;
    }
    let mut tsig0 = est_div(r0, r1, dsig0);
    let (m0, m1, m2) = mul128_by64(dsig0, dsig1, tsig0);
    (r0, r1, r2) = sub192(r0, r1, r2, m0, m1, m2);
    while (r0 as i64) < 0 {
        tsig0 = tsig0.wrapping_sub(1);
        (r0, r1, r2) = add192(r0, r1, r2, 0, dsig0, dsig1);
    }
    // No need to correct any estimation error in tsig1. Now square the approximation.
    let tsig1 = est_div(r1, r2, dsig0);
    let (t2sig0, t2sig1, _, _) = mul128_to256(tsig0, tsig1, tsig0, tsig1);
    let t2 =
        FloatX80::normalize_round_and_pack(Prec::X, false, texp + texp - 0x3ffe, t2sig0, t2sig1, s);

    // Compute the lower parts of the polynomial expansion.
    let mut accum = FYL2X_C9.mul(t2, s);
    for c in [FYL2X_C8, FYL2X_C7, FYL2X_C6, FYL2X_C5, FYL2X_C4, FYL2X_C3, FYL2X_C2, FYL2X_C1] {
        accum = c.add(accum, s);
        accum = accum.mul(t2, s);
    }
    accum = FYL2X_C0_LOW.add(accum, s);

    // The full expansion is FYL2X_C0 + accum (which cannot carry out), times t.
    let mut aexp = exp_of(FYL2X_C0);
    let asign = sign_of(FYL2X_C0);
    let (a0, a1) = shr_jam(accum.low, 0, aexp - exp_of(accum));
    let (a0, a1) = if asign == sign_of(accum) {
        add128(FYL2X_C0.low, 0, a0, a1)
    } else {
        sub128(FYL2X_C0.low, 0, a0, a1)
    };
    let (a0, a1, _, _) = mul128_to256(a0, a1, tsig0, tsig1);
    aexp += texp - 0x3ffe;
    (aexp, a0, a1)
}

/// The NaN and invalid encoding cases shared by FPATAN, FYL2X and FYL2XP1. Returns true if
/// one of them applied.
fn two_arg_nan(e: &mut [u8], st0: FloatX80, st1: FloatX80, s: &mut FloatStatus) -> bool {
    if st0.invalid_encoding(s) || st1.invalid_encoding(s) {
        s.raise(ff::INVALID);
        set(e, 1, FloatX80::default_nan(s));
    } else if st0.is_signaling_nan(s) {
        s.raise(ff::INVALID);
        set(e, 1, st0.silence_nan(s));
    } else if st1.is_signaling_nan(s) {
        s.raise(ff::INVALID);
        set(e, 1, st1.silence_nan(s));
    } else if st0.is_any_nan() {
        set(e, 1, st0);
    } else if st1.is_any_nan() {
        // Pass this NaN through.
    } else {
        return false;
    }
    true
}

/// `helper_fyl2xp1()`.
fn fyl2xp1(e: &mut [u8], s: &mut FloatStatus) {
    let (st0, st1) = (get(e, 0), get(e, 1));
    let (mut arg0_sig, mut arg0_exp, arg0_sign) = parts(st0);
    let (mut arg1_sig, mut arg1_exp, arg1_sign) = parts(st1);
    let limit = if arg0_sign { 0x95f619980c4336f7 } else { 0xd413cccfe7799211 };
    if two_arg_nan(e, st0, st1, s) {
    } else if arg0_exp > 0x3ffd || (arg0_exp == 0x3ffd && arg0_sig > limit) {
        // Out of range for the instruction (Intel allows |ST0| < 1 - sqrt(2)/2, AMD allows
        // sqrt(2)/2 - 1 to sqrt(2) - 1, which is what is accepted here), treat as invalid.
        s.raise(ff::INVALID);
        set(e, 1, FloatX80::default_nan(s));
    } else if st0.is_zero() || st1.is_zero() || arg1_exp == 0x7fff {
        // One argument is zero, or multiplying by infinity; the exact result is the product.
        set(e, 1, st0.mul(st1, s));
    } else if arg0_exp < 0x3fb0 {
        // Multiplying both arguments and an extra precision version of log2(e) is precise
        // enough.
        if arg0_exp == 0 {
            (arg0_exp, arg0_sig) = norm_sub(arg0_sig);
        }
        if arg1_exp == 0 {
            (arg1_exp, arg1_sig) = norm_sub(arg1_sig);
        }
        let (sig0, sig1, _) = mul128_by64(LOG2_E_SIG_HIGH, LOG2_E_SIG_LOW, arg0_sig);
        let mut exp = arg0_exp + 1;
        let (sig0, sig1, _) = mul128_by64(sig0, sig1, arg1_sig);
        exp += arg1_exp - 0x3ffe;
        // The result is inexact.
        let sign = arg0_sign ^ arg1_sign;
        set(e, 1, FloatX80::normalize_round_and_pack(Prec::X, sign, exp, sig0, sig1 | 1, s));
    } else {
        let save_mode = s.rounding_mode;
        let save_prec = s.floatx80_rounding_precision;
        s.rounding_mode = RoundMode::NearestEven;
        s.floatx80_rounding_precision = Prec::X;

        let (mut aexp, asig0, asig1) = fyl2x_common(st0, s);
        // Multiply by the second argument.
        if arg1_exp == 0 {
            (arg1_exp, arg1_sig) = norm_sub(arg1_sig);
        }
        let (asig0, asig1, _) = mul128_by64(asig0, asig1, arg1_sig);
        aexp += arg1_exp - 0x3ffe;
        // The result is inexact.
        s.rounding_mode = save_mode;
        let sign = arg0_sign ^ arg1_sign;
        set(e, 1, FloatX80::normalize_round_and_pack(Prec::X, sign, aexp, asig0, asig1 | 1, s));
        s.floatx80_rounding_precision = save_prec;
    }
    fpop(e);
}

/// `helper_fyl2x()`.
fn fyl2x(e: &mut [u8], s: &mut FloatStatus) {
    let (st0, st1) = (get(e, 0), get(e, 1));
    let (mut arg0_sig, mut arg0_exp, arg0_sign) = parts(st0);
    let (mut arg1_sig, mut arg1_exp, arg1_sign) = parts(st1);
    if two_arg_nan(e, st0, st1, s) {
    } else if arg0_sign && !st0.is_zero() {
        s.raise(ff::INVALID);
        set(e, 1, FloatX80::default_nan(s));
    } else if st1.is_infinity(s) {
        match st0.compare(ONE, s) {
            FloatRelation::Less => set(e, 1, st1.chs()),
            // The result is infinity of the same sign as ST1.
            FloatRelation::Greater => {}
            _ => {
                s.raise(ff::INVALID);
                set(e, 1, FloatX80::default_nan(s));
            }
        }
    } else if st0.is_infinity(s) {
        if st1.is_zero() {
            s.raise(ff::INVALID);
            set(e, 1, FloatX80::default_nan(s));
        } else if arg1_sign {
            set(e, 1, st0.chs());
        } else {
            set(e, 1, st0);
        }
    } else if st0.is_zero() {
        if st1.is_zero() {
            s.raise(ff::INVALID);
            set(e, 1, FloatX80::default_nan(s));
        } else {
            // The result is infinity with the opposite sign to ST1.
            s.raise(ff::DIVBYZERO);
            set(e, 1, fx(if arg1_sign { 0x7fff } else { 0xffff }, 1 << 63));
        }
    } else if st1.is_zero() {
        if st0.lt(ONE, s) {
            set(e, 1, st1.chs());
        }
        // Otherwise ST1 is already the correct result.
    } else if st0.eq(ONE, s) {
        set(e, 1, if arg1_sign { ZERO.chs() } else { ZERO });
    } else {
        let save_mode = s.rounding_mode;
        let save_prec = s.floatx80_rounding_precision;
        s.rounding_mode = RoundMode::NearestEven;
        s.floatx80_rounding_precision = Prec::X;

        if arg0_exp == 0 {
            (arg0_exp, arg0_sig) = norm_sub(arg0_sig);
        }
        if arg1_exp == 0 {
            (arg1_exp, arg1_sig) = norm_sub(arg1_sig);
        }
        let mut int_exp = arg0_exp - 0x3fff;
        if arg0_sig > 0xb504f333f9de6484 {
            int_exp += 1;
        }
        let arg0_m1 = st0.scalbn(-int_exp, s).sub(ONE, s);
        if arg0_m1.is_zero() {
            // An exact power of 2; multiply by ST1.
            s.rounding_mode = save_mode;
            let r = FloatX80::from_i32(int_exp, s).mul(st1, s);
            set(e, 1, r);
        } else {
            let mut asign = sign_of(arg0_m1);
            let (mut aexp, mut asig0, mut asig1) = fyl2x_common(arg0_m1, s);
            if int_exp != 0 {
                let isign = int_exp < 0;
                let iabs = int_exp.unsigned_abs();
                let shift = iabs.leading_zeros() as i32 + 32;
                let isig = u64::from(iabs).wrapping_shl(shift as u32);
                let iexp = 0x403e - shift;
                (asig0, asig1) = shr_jam(asig0, asig1, iexp - aexp);
                (asig0, asig1) = if asign == isign {
                    add128(isig, 0, asig0, asig1)
                } else {
                    sub128(isig, 0, asig0, asig1)
                };
                aexp = iexp;
                asign = isign;
            }
            // Multiply by the second argument.
            if arg1_exp == 0 {
                (arg1_exp, arg1_sig) = norm_sub(arg1_sig);
            }
            (asig0, asig1, _) = mul128_by64(asig0, asig1, arg1_sig);
            aexp += arg1_exp - 0x3ffe;
            // The result is inexact.
            s.rounding_mode = save_mode;
            let sign = asign ^ arg1_sign;
            let r = FloatX80::normalize_round_and_pack(Prec::X, sign, aexp, asig0, asig1 | 1, s);
            set(e, 1, r);
        }
        s.floatx80_rounding_precision = save_prec;
    }
    fpop(e);
}

// Guest memory access, `X86Access`.

/// A probed range of guest memory (`X86Access`).
struct Access {
    idx: usize,
}

impl Access {
    /// `access_prepare()`: probe `size` bytes at `ptr` (at most two pages) for `at` with the
    /// current MMU index, raising any fault now.
    fn prepare(cpu: &mut Cpu<'_>, ptr: u64, size: usize, at: MmuAccessType) -> R<Self> {
        let idx = mmu_index_pl(cpu.env, hflags(cpu) & HF_CPL_MASK);
        let size1 = size.min(0x1000 - (ptr & 0xfff) as usize);
        probe_access(cpu, ptr, size1, at, idx, TB)?;
        if size > size1 {
            probe_access(cpu, ptr.wrapping_add(size1 as u64), size - size1, at, idx, TB)?;
        }
        Ok(Access { idx })
    }

    fn ld(&self, cpu: &mut Cpu<'_>, addr: u64, mop: MemOp) -> R<u64> {
        sh::ld(cpu, addr, mop, self.idx, TB)
    }

    fn st(&self, cpu: &mut Cpu<'_>, addr: u64, v: u64, mop: MemOp) -> R<()> {
        sh::st(cpu, addr, v, mop, self.idx, TB)
    }

    /// `do_fldt()`.
    fn ldt(&self, cpu: &mut Cpu<'_>, addr: u64) -> R<FloatX80> {
        let low = self.ld(cpu, addr, MemOp::LEUQ)?;
        let high = self.ld(cpu, addr.wrapping_add(8), MemOp::LEUW)?;
        Ok(FloatX80::new(high as u16, low))
    }

    /// `do_fstt()`.
    fn stt(&self, cpu: &mut Cpu<'_>, addr: u64, x: FloatX80) -> R<()> {
        self.st(cpu, addr, x.low, MemOp::LEUQ)?;
        self.st(cpu, addr.wrapping_add(8), u64::from(x.high), MemOp::LEUW)
    }
}

// FXSAVE, FXRSTOR, XSAVE and XRSTOR.

/// The number of XMM registers the save area instructions cover.
fn nb_xmm_regs(cpu: &Cpu<'_>) -> usize {
    if hflags(cpu) & HF_CS64_MASK != 0 { 16 } else { 8 }
}

/// Whether FXSAVE and FXRSTOR skip the XMM registers (fast FXSAVE, `MSR_EFER_FFXSR`).
fn ffxsr_skips_sse(cpu: &Cpu<'_>) -> bool {
    let hf = hflags(cpu);
    ld64(cpu.env, EFER) & MSR_EFER_FFXSR != 0 && hf & HF_CPL_MASK == 0 && hf & HF_LMA_MASK != 0
}

/// `do_xsave_fpu()`.
fn xsave_fpu(cpu: &mut Cpu<'_>, ac: &Access, ptr: u64) -> R<()> {
    let e = &*cpu.env;
    let fpus = fnstsw(e);
    let tag = (0..8).fold(0u32, |t, i| t | u32::from(e[FPTAGS + i]) << i);
    let fpuc = ld32(e, FPUC);
    ac.st(cpu, ptr.wrapping_add(XO_FCW), u64::from(fpuc), MemOp::LEUW)?;
    ac.st(cpu, ptr.wrapping_add(XO_FSW), u64::from(fpus), MemOp::LEUW)?;
    ac.st(cpu, ptr.wrapping_add(XO_FTW), u64::from(tag ^ 0xff), MemOp::LEUW)?;
    // In 32 bit mode this is eip, sel, dp, sel and in 64 bit mode rip, rdp, but QEMU
    // writes zeros either way.
    ac.st(cpu, ptr.wrapping_add(XO_FPIP), 0, MemOp::LEUQ)?;
    ac.st(cpu, ptr.wrapping_add(XO_FPDP), 0, MemOp::LEUQ)?;
    for i in 0..8 {
        let x = get(cpu.env, i);
        ac.stt(cpu, ptr.wrapping_add(XO_FPREGS + 16 * i as u64), x)?;
    }
    Ok(())
}

/// `do_xsave_mxcsr()`.
fn xsave_mxcsr(cpu: &mut Cpu<'_>, ac: &Access, ptr: u64) -> R<()> {
    let mxcsr = ld32(cpu.env, MXCSR);
    ac.st(cpu, ptr.wrapping_add(XO_MXCSR), u64::from(mxcsr), MemOp::LEUL)?;
    ac.st(cpu, ptr.wrapping_add(XO_MXCSR_MASK), 0x0000_ffff, MemOp::LEUL)
}

/// `do_xsave_sse()` for `half` 0 and `do_xsave_ymmh()` for `half` 1, with `ptr` at the
/// start of the component.
fn xsave_xmm(cpu: &mut Cpu<'_>, ac: &Access, ptr: u64, half: usize) -> R<()> {
    for i in 0..nb_xmm_regs(cpu) {
        let q0 = ld64(cpu.env, zmm(i) + 16 * half);
        let q1 = ld64(cpu.env, zmm(i) + 16 * half + 8);
        let p = ptr.wrapping_add(16 * i as u64);
        ac.st(cpu, p, q0, MemOp::LEUQ)?;
        ac.st(cpu, p.wrapping_add(8), q1, MemOp::LEUQ)?;
    }
    Ok(())
}

/// `do_xrstor_sse()` and `do_xrstor_ymmh()`, the reverse of [`xsave_xmm`].
fn xrstor_xmm(cpu: &mut Cpu<'_>, ac: &Access, ptr: u64, half: usize) -> R<()> {
    for i in 0..nb_xmm_regs(cpu) {
        let p = ptr.wrapping_add(16 * i as u64);
        let q0 = ac.ld(cpu, p, MemOp::LEUQ)?;
        let q1 = ac.ld(cpu, p.wrapping_add(8), MemOp::LEUQ)?;
        st64(cpu.env, zmm(i) + 16 * half, q0);
        st64(cpu.env, zmm(i) + 16 * half + 8, q1);
    }
    Ok(())
}

/// `do_clear_sse()` and `do_clear_ymmh()`.
fn clear_xmm(cpu: &mut Cpu<'_>, half: usize) {
    for i in 0..nb_xmm_regs(cpu) {
        st64(cpu.env, zmm(i) + 16 * half, 0);
        st64(cpu.env, zmm(i) + 16 * half + 8, 0);
    }
}

/// `do_xrstor_fpu()`.
fn xrstor_fpu(cpu: &mut Cpu<'_>, ac: &Access, ptr: u64) -> R<()> {
    let fpuc = ac.ld(cpu, ptr.wrapping_add(XO_FCW), MemOp::LEUW)?;
    let fpus = ac.ld(cpu, ptr.wrapping_add(XO_FSW), MemOp::LEUW)?;
    let tag = ac.ld(cpu, ptr.wrapping_add(XO_FTW), MemOp::LEUW)? ^ 0xff;
    st32(cpu.env, FPUC, fpuc as u32);
    set_fpus(cpu.env, fpus as u32);
    for i in 0..8 {
        cpu.env[FPTAGS + i] = ((tag >> i) & 1) as u8;
    }
    for i in 0..8 {
        let x = ac.ldt(cpu, ptr.wrapping_add(XO_FPREGS + 16 * i as u64))?;
        set(cpu.env, i as usize, x);
    }
    Ok(())
}

fn h_fxsave(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    let ptr = a[1];
    run(h, |cpu| {
        if ptr & 0xf != 0 {
            return Err(sh::raise_exception_ra(cpu, EXCP0D_GPF, TB));
        }
        let ac = Access::prepare(cpu, ptr, LEGACY_SIZE, MmuAccessType::DataStore)?;
        xsave_fpu(cpu, &ac, ptr)?;
        if ld64(cpu.env, cr(4)) & CR4_OSFXSR_MASK != 0 {
            xsave_mxcsr(cpu, &ac, ptr)?;
            if !ffxsr_skips_sse(cpu) {
                xsave_xmm(cpu, &ac, ptr.wrapping_add(XO_XMM), 0)?;
            }
        }
        Ok(0)
    })
}

fn h_fxrstor(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    let ptr = a[1];
    run(h, |cpu| {
        if ptr & 0xf != 0 {
            return Err(sh::raise_exception_ra(cpu, EXCP0D_GPF, TB));
        }
        let ac = Access::prepare(cpu, ptr, LEGACY_SIZE, MmuAccessType::DataLoad)?;
        xrstor_fpu(cpu, &ac, ptr)?;
        if ld64(cpu.env, cr(4)) & CR4_OSFXSR_MASK != 0 {
            let mxcsr = ac.ld(cpu, ptr.wrapping_add(XO_MXCSR), MemOp::LEUL)?;
            st32(cpu.env, MXCSR, mxcsr as u32);
            if !ffxsr_skips_sse(cpu) {
                xrstor_xmm(cpu, &ac, ptr.wrapping_add(XO_XMM), 0)?;
            }
        }
        Ok(0)
    })
}

/// `get_xinuse()`: every component but BNDREGS, whose use is not tracked without MPX.
fn xinuse() -> u64 {
    !XSTATE_BNDREGS
}

/// `do_xsave_chk()`.
fn xsave_chk(cpu: &mut Cpu<'_>, ptr: u64) -> R<()> {
    if ld64(cpu.env, cr(4)) & CR4_OSXSAVE_MASK == 0 {
        return Err(sh::raise_exception_ra(cpu, EXCP06_ILLOP, TB));
    }
    if ptr & 63 != 0 {
        return Err(sh::raise_exception_ra(cpu, EXCP0D_GPF, TB));
    }
    Ok(())
}

/// `xsave_area_size()` for this CPU model in the standard layout.
fn area_size(cpu: &Cpu<'_>, mask: u64) -> usize {
    let ops = cpu.ops();
    xsave_area_size(x86_of(&ops).model().ext_save_areas(), mask, false) as usize
}

/// `do_xsave()`.
fn xsave(cpu: &mut Cpu<'_>, ptr: u64, rfbm: u64, inuse: u64, opt: u64) -> R<()> {
    xsave_chk(cpu, ptr)?;
    let rfbm = rfbm & ld64(cpu.env, XCR0);
    let opt = opt & rfbm;
    let size = area_size(cpu, opt);
    let ac = Access::prepare(cpu, ptr, size, MmuAccessType::DataStore)?;
    if opt & XSTATE_FP != 0 {
        xsave_fpu(cpu, &ac, ptr)?;
    }
    if rfbm & XSTATE_SSE != 0 {
        xsave_mxcsr(cpu, &ac, ptr)?;
    }
    if opt & XSTATE_SSE != 0 {
        xsave_xmm(cpu, &ac, ptr.wrapping_add(XO_XMM), 0)?;
    }
    if opt & XSTATE_YMM != 0 {
        xsave_xmm(cpu, &ac, ptr.wrapping_add(XO_AVX), 1)?;
    }
    if opt & XSTATE_PKRU != 0 {
        let pkru = ld32(cpu.env, PKRU);
        ac.st(cpu, ptr.wrapping_add(XO_PKRU), u64::from(pkru), MemOp::LEUQ)?;
    }
    let old_bv = ac.ld(cpu, ptr.wrapping_add(XO_XSTATE_BV), MemOp::LEUQ)?;
    let new_bv = (old_bv & !rfbm) | (inuse & rfbm);
    ac.st(cpu, ptr.wrapping_add(XO_XSTATE_BV), new_bv, MemOp::LEUQ)
}

fn h_xsave(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    let (ptr, rfbm) = (a[1], a[2]);
    run(h, |cpu| xsave(cpu, ptr, rfbm, xinuse(), rfbm).map(|()| 0))
}

fn h_xrstor(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    let (ptr, rfbm) = (a[1], a[2]);
    run(h, |cpu| {
        xsave_chk(cpu, ptr)?;
        // Begin with just the minimum size to validate the header.
        let size = LEGACY_HEADER_SIZE;
        let mut ac = Access::prepare(cpu, ptr, size, MmuAccessType::DataLoad)?;
        let xstate_bv = ac.ld(cpu, ptr.wrapping_add(XO_XSTATE_BV), MemOp::LEUQ)?;
        let xcomp_bv = ac.ld(cpu, ptr.wrapping_add(XO_XCOMP_BV), MemOp::LEUQ)?;
        let reserve0 = ac.ld(cpu, ptr.wrapping_add(XO_RESERVE0), MemOp::LEUQ)?;
        let xcr0 = ld64(cpu.env, XCR0);
        if xcomp_bv != 0 || reserve0 != 0 || xstate_bv & !xcr0 != 0 {
            return Err(sh::raise_exception_ra(cpu, EXCP0D_GPF, TB));
        }
        let rfbm = rfbm & xcr0;
        let size_ext = area_size(cpu, rfbm & xstate_bv);
        if size < size_ext {
            ac = Access::prepare(cpu, ptr, size_ext, MmuAccessType::DataLoad)?;
        }

        if rfbm & XSTATE_FP != 0 {
            if xstate_bv & XSTATE_FP != 0 {
                xrstor_fpu(cpu, &ac, ptr)?;
            } else {
                fninit(cpu.env);
                for i in 0..8 {
                    stx(cpu.env, fpreg(i), ZERO);
                }
            }
        }
        if rfbm & XSTATE_SSE != 0 {
            // The standard form of XRSTOR loads MXCSR whether or not XSTATE_BV has SSE.
            let mxcsr = ac.ld(cpu, ptr.wrapping_add(XO_MXCSR), MemOp::LEUL)?;
            st32(cpu.env, MXCSR, mxcsr as u32);
            if xstate_bv & XSTATE_SSE != 0 {
                xrstor_xmm(cpu, &ac, ptr.wrapping_add(XO_XMM), 0)?;
            } else {
                clear_xmm(cpu, 0);
            }
        }
        if rfbm & XSTATE_YMM != 0 {
            if xstate_bv & XSTATE_YMM != 0 {
                xrstor_xmm(cpu, &ac, ptr.wrapping_add(XO_AVX), 1)?;
            } else {
                clear_xmm(cpu, 1);
            }
        }
        if rfbm & XSTATE_PKRU != 0 {
            let old = ld32(cpu.env, PKRU);
            let new = if xstate_bv & XSTATE_PKRU != 0 {
                ac.ld(cpu, ptr.wrapping_add(XO_PKRU), MemOp::LEUQ)? as u32
            } else {
                0
            };
            st32(cpu.env, PKRU, new);
            if new != old {
                tlb_flush(cpu);
            }
        }
        Ok(0)
    })
}
