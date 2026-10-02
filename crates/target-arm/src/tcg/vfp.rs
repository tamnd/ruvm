// SPDX-License-Identifier: GPL-2.0-or-later

//! Scalar floating point and the AdvSIMD floating point helpers: the A64 parts of QEMU's
//! `vfp_helper.c`, `vfp_fpscr.c`, `helper-a64.c` and the floating point parts of
//! `vec_helper.c`.
//!
//! Every operation goes through `ruvm-softfloat` with the `float_status` QEMU would use, so
//! rounding, flushing, default NaNs and the cumulative flags come out bit for bit the same.
//! The two A64 statuses QEMU uses without FEAT_AFP, `FPST_A64` and `FPST_A64_F16`, are kept
//! packed in `CpuArmState::fp_status` (see [`pack_status`]): the rest of `float_status` is
//! the same for both and fixed by `arm_set_default_fp_behaviours()`, which is
//! [`FloatStatus::arm`].
//!
//! The differences from QEMU:
//!
//! - QEMU has a helper for each operation, size and shape; here one helper, `a64_fp`, takes
//!   the register numbers and a descriptor, reads the operands out of `env` and writes the
//!   whole destination register back. The order of the softfloat calls, and therefore the
//!   flags and the NaN chosen, is the one of QEMU's helpers.
//! - FPCR.AH, FIZ and NEP are always zero (no FEAT_AFP), as on the CPUs QEMU models here, so
//!   the `FPST_AH` statuses and the AH variants of the helpers do not exist.

use ruvm_jit_core::HelperType::{I32, I64, Ptr, Void};
use ruvm_jit_interp::{HelperEnv, Unwind};
use ruvm_softfloat::{Float16, Float32, Float64, FloatRelation, FloatStatus, RoundMode, flags};

use super::helpers::{Def, def};
use super::vec_helper::{Desc, RMODE_FPCR, Regs, V, clear_tail, get, set, vload, vstore};
use crate::cpu::{ArmFeatures, CpuArmState, FPCR, FPST_A64};

/// FPCR.AHP, alternative half precision.
pub(crate) const FPCR_AHP: u32 = 1 << 26;
/// FPCR.DN, default NaN.
pub(crate) const FPCR_DN: u32 = 1 << 25;
/// FPCR.FZ, flush to zero.
pub(crate) const FPCR_FZ: u32 = 1 << 24;
/// FPCR.RMode.
pub(crate) const FPCR_RMODE_MASK: u32 = 3 << 22;
/// FPCR.Stride.
pub(crate) const FPCR_STRIDE_MASK: u32 = 3 << 20;
/// FPCR.FZ16, flush half precision to zero.
pub(crate) const FPCR_FZ16: u32 = 1 << 19;
/// FPCR.Len.
pub(crate) const FPCR_LEN_MASK: u32 = 7 << 16;

/// FPSR.QC, cumulative saturation.
pub(crate) const FPSR_QC: u32 = 1 << 27;
const FPSR_NZCV_MASK: u32 = 0xf000_0000;
const FPSR_CEXC_MASK: u32 = 0x9f;
const FPSR_IOC: u32 = 1;
const FPSR_DZC: u32 = 2;
const FPSR_OFC: u32 = 4;
const FPSR_UFC: u32 = 8;
const FPSR_IXC: u32 = 0x10;
const FPSR_IDC: u32 = 0x80;

/// Unpack a status kept in `CpuArmState::fp_status`.
///
/// Bits 0 to 7 hold the rounding mode, bit 8 `flush_to_zero`, bit 9
/// `flush_inputs_to_zero`, bit 10 `default_nan_mode` and bits 16 to 31 the exception flags.
/// Zero is the reset state.
pub(crate) fn unpack_status(p: u64) -> FloatStatus {
    let mut s = FloatStatus::arm();
    s.rounding_mode = RoundMode::from_u8(p as u8).unwrap_or(RoundMode::NearestEven);
    s.flush_to_zero = p & (1 << 8) != 0;
    s.flush_inputs_to_zero = p & (1 << 9) != 0;
    s.default_nan_mode = p & (1 << 10) != 0;
    s.exception_flags = (p >> 16) as u16;
    s
}

/// Pack a status for `CpuArmState::fp_status`, see [`unpack_status`].
pub(crate) fn pack_status(s: &FloatStatus) -> u64 {
    u64::from(s.rounding_mode as u8)
        | u64::from(s.flush_to_zero) << 8
        | u64::from(s.flush_inputs_to_zero) << 9
        | u64::from(s.default_nan_mode) << 10
        | u64::from(s.exception_flags) << 16
}

fn load_status(env: &[u8], idx: usize) -> FloatStatus {
    let o = FPST_A64 + 8 * idx;
    unpack_status(u64::from_le_bytes(env[o..o + 8].try_into().unwrap()))
}

fn store_status(env: &mut [u8], idx: usize, s: &FloatStatus) {
    let o = FPST_A64 + 8 * idx;
    env[o..o + 8].copy_from_slice(&pack_status(s).to_le_bytes());
}

fn env_fpcr(env: &[u8]) -> u32 {
    u32::from_le_bytes(env[FPCR..FPCR + 4].try_into().unwrap())
}

/// `vfp_exceptbits_from_host()` with AH clear.
fn exceptbits_from_host(f: u16) -> u32 {
    let mut t = 0;
    if f & flags::INVALID != 0 {
        t |= FPSR_IOC;
    }
    if f & flags::DIVBYZERO != 0 {
        t |= FPSR_DZC;
    }
    if f & flags::OVERFLOW != 0 {
        t |= FPSR_OFC;
    }
    if f & (flags::UNDERFLOW | flags::OUTPUT_DENORMAL_FLUSHED) != 0 {
        t |= FPSR_UFC;
    }
    if f & flags::INEXACT != 0 {
        t |= FPSR_IXC;
    }
    if f & flags::INPUT_DENORMAL_FLUSHED != 0 {
        t |= FPSR_IDC;
    }
    t
}

/// `vfp_get_fpsr_from_host()`.
fn fpsr_from_host(st: &CpuArmState) -> u32 {
    let mut a64 = unpack_status(st.fp_status[0]).exception_flags;
    a64 |= unpack_status(st.fp_status[1]).exception_flags
        & !(flags::INPUT_DENORMAL_FLUSHED | flags::INPUT_DENORMAL_USED);
    // AH is always clear, so IDC needs FZ.
    if st.fpcr & FPCR_FZ == 0 {
        a64 &= !flags::INPUT_DENORMAL_FLUSHED;
    }
    exceptbits_from_host(a64)
}

fn clear_status_flags(st: &mut CpuArmState) {
    for p in &mut st.fp_status {
        *p &= 0xffff;
    }
}

/// `vfp_get_fpcr()`.
pub(crate) fn get_fpcr(st: &CpuArmState) -> u32 {
    st.fpcr
}

/// `vfp_get_fpsr()`.
pub(crate) fn get_fpsr(st: &CpuArmState) -> u32 {
    let mut fpsr = st.fpsr | fpsr_from_host(st);
    if st.qc[0] | st.qc[1] != 0 {
        fpsr |= FPSR_QC;
    }
    fpsr
}

/// `vfp_set_fpsr()`.
pub(crate) fn set_fpsr(st: &mut CpuArmState, val: u32) {
    st.qc = [u64::from(val & FPSR_QC), 0];
    st.fpsr = val & (FPSR_NZCV_MASK | FPSR_CEXC_MASK);
    clear_status_flags(st);
}

/// `vfp_set_fpcr()`, `vfp_set_fpcr_masked()` and `vfp_set_fpcr_to_host()` for a CPU
/// without FEAT_AFP and FEAT_EBF16.
pub(crate) fn set_fpcr(st: &mut CpuArmState, val: u32, feat: &ArmFeatures) {
    let mut val = val;
    if !feat.fp16 {
        val &= !FPCR_FZ16;
    }
    let changed = st.fpcr ^ val;
    // Fold the flags into FPSR under the old regime before the IDC rule changes.
    if changed & FPCR_FZ != 0 {
        st.fpsr |= fpsr_from_host(st);
        clear_status_flags(st);
    }
    let mut a64 = unpack_status(st.fp_status[0]);
    let mut f16 = unpack_status(st.fp_status[1]);
    let rm = match (val >> 22) & 3 {
        0 => RoundMode::NearestEven,
        1 => RoundMode::Up,
        2 => RoundMode::Down,
        _ => RoundMode::ToZero,
    };
    a64.rounding_mode = rm;
    f16.rounding_mode = rm;
    f16.flush_to_zero = val & FPCR_FZ16 != 0;
    f16.flush_inputs_to_zero = val & FPCR_FZ16 != 0;
    a64.flush_to_zero = val & FPCR_FZ != 0;
    a64.flush_inputs_to_zero = val & FPCR_FZ != 0;
    a64.default_nan_mode = val & FPCR_DN != 0;
    f16.default_nan_mode = val & FPCR_DN != 0;
    st.fp_status = [pack_status(&a64), pack_status(&f16)];
    // QEMU keeps Len and Stride in vec_len and vec_stride and reads them back.
    st.fpcr = val
        & (FPCR_AHP
            | FPCR_DN
            | FPCR_FZ
            | FPCR_RMODE_MASK
            | FPCR_FZ16
            | FPCR_LEN_MASK
            | FPCR_STRIDE_MASK);
}

/// The operations shared by the three formats, so the helpers can be written once.
pub(crate) trait Fp: Copy {
    /// log2 of the size in bytes.
    const ESZ: u32;
    /// The number of fraction bits.
    const FRAC: u32;
    /// The number of exponent bits.
    const EXPB: u32;
    fn from_bits(x: u64) -> Self;
    fn bits(self) -> u64;
    fn add(self, b: Self, s: &mut FloatStatus) -> Self;
    fn sub(self, b: Self, s: &mut FloatStatus) -> Self;
    fn mul(self, b: Self, s: &mut FloatStatus) -> Self;
    fn div(self, b: Self, s: &mut FloatStatus) -> Self;
    fn max(self, b: Self, s: &mut FloatStatus) -> Self;
    fn min(self, b: Self, s: &mut FloatStatus) -> Self;
    fn maxnum(self, b: Self, s: &mut FloatStatus) -> Self;
    fn minnum(self, b: Self, s: &mut FloatStatus) -> Self;
    fn muladd(self, b: Self, c: Self, fl: u32, s: &mut FloatStatus) -> Self;
    fn muladd_scalbn(self, b: Self, c: Self, sc: i32, fl: u32, s: &mut FloatStatus) -> Self;
    fn sqrt(self, s: &mut FloatStatus) -> Self;
    fn round_to_int(self, s: &mut FloatStatus) -> Self;
    fn compare(self, b: Self, s: &mut FloatStatus) -> FloatRelation;
    fn compare_quiet(self, b: Self, s: &mut FloatStatus) -> FloatRelation;
    fn is_any_nan(self) -> bool;
    fn is_signaling_nan(self, s: &FloatStatus) -> bool;
    fn silence_nan(self, s: &FloatStatus) -> Self;
    fn default_nan(s: &FloatStatus) -> Self;
    fn squash_input_denormal(self, s: &mut FloatStatus) -> Self;
    fn to_int(self, bits: u32, signed: bool, rm: RoundMode, sc: i32, s: &mut FloatStatus) -> u64;
    fn from_int(x: u64, signed: bool, sc: i32, s: &mut FloatStatus) -> Self;

    fn sign_bit() -> u64 {
        1 << (Self::FRAC + Self::EXPB)
    }
    fn exp_mask() -> u64 {
        (1 << Self::EXPB) - 1
    }
    fn frac_mask() -> u64 {
        (1 << Self::FRAC) - 1
    }
    fn is_neg(self) -> bool {
        self.bits() & Self::sign_bit() != 0
    }
    fn is_zero(self) -> bool {
        self.bits() & !Self::sign_bit() == 0
    }
    fn is_infinity(self) -> bool {
        self.bits() & !Self::sign_bit() == Self::exp_mask() << Self::FRAC
    }
    fn abs(self) -> Self {
        Self::from_bits(self.bits() & !Self::sign_bit())
    }
    fn chs(self) -> Self {
        Self::from_bits(self.bits() ^ Self::sign_bit())
    }
    fn signed(self, x: u64) -> Self {
        Self::from_bits(x | (self.bits() & Self::sign_bit()))
    }
    fn zero() -> Self {
        Self::from_bits(0)
    }
    fn infinity() -> Self {
        Self::from_bits(Self::exp_mask() << Self::FRAC)
    }
    fn maxnorm() -> u64 {
        ((Self::exp_mask() - 1) << Self::FRAC) | Self::frac_mask()
    }
    /// The value with exponent `bias + e` and a zero fraction.
    fn pow2(e: i32) -> Self {
        let bias = (1i32 << (Self::EXPB - 1)) - 1;
        Self::from_bits(((bias + e) as u64) << Self::FRAC)
    }
    fn two() -> Self {
        Self::pow2(1)
    }
    fn three() -> Self {
        Self::from_bits(Self::pow2(1).bits() | 1 << (Self::FRAC - 1))
    }
    fn one_point_five() -> Self {
        Self::from_bits(Self::pow2(0).bits() | 1 << (Self::FRAC - 1))
    }
}

macro_rules! impl_fp {
    ($t:ident, $raw:ty, $esz:expr, $frac:expr, $expb:expr) => {
        impl Fp for $t {
            const ESZ: u32 = $esz;
            const FRAC: u32 = $frac;
            const EXPB: u32 = $expb;
            fn from_bits(x: u64) -> Self {
                $t(x as $raw)
            }
            fn bits(self) -> u64 {
                u64::from(self.0)
            }
            fn add(self, b: Self, s: &mut FloatStatus) -> Self {
                $t::add(self, b, s)
            }
            fn sub(self, b: Self, s: &mut FloatStatus) -> Self {
                $t::sub(self, b, s)
            }
            fn mul(self, b: Self, s: &mut FloatStatus) -> Self {
                $t::mul(self, b, s)
            }
            fn div(self, b: Self, s: &mut FloatStatus) -> Self {
                $t::div(self, b, s)
            }
            fn max(self, b: Self, s: &mut FloatStatus) -> Self {
                $t::max(self, b, s)
            }
            fn min(self, b: Self, s: &mut FloatStatus) -> Self {
                $t::min(self, b, s)
            }
            fn maxnum(self, b: Self, s: &mut FloatStatus) -> Self {
                $t::maxnum(self, b, s)
            }
            fn minnum(self, b: Self, s: &mut FloatStatus) -> Self {
                $t::minnum(self, b, s)
            }
            fn muladd(self, b: Self, c: Self, fl: u32, s: &mut FloatStatus) -> Self {
                $t::muladd(self, b, c, fl, s)
            }
            fn muladd_scalbn(
                self,
                b: Self,
                c: Self,
                sc: i32,
                fl: u32,
                s: &mut FloatStatus,
            ) -> Self {
                $t::muladd_scalbn(self, b, c, sc, fl, s)
            }
            fn sqrt(self, s: &mut FloatStatus) -> Self {
                $t::sqrt(self, s)
            }
            fn round_to_int(self, s: &mut FloatStatus) -> Self {
                $t::round_to_int(self, s)
            }
            fn compare(self, b: Self, s: &mut FloatStatus) -> FloatRelation {
                $t::compare(self, b, s)
            }
            fn compare_quiet(self, b: Self, s: &mut FloatStatus) -> FloatRelation {
                $t::compare_quiet(self, b, s)
            }
            fn is_any_nan(self) -> bool {
                $t::is_any_nan(self)
            }
            fn is_signaling_nan(self, s: &FloatStatus) -> bool {
                $t::is_signaling_nan(self, s)
            }
            fn silence_nan(self, s: &FloatStatus) -> Self {
                $t::silence_nan(self, s)
            }
            fn default_nan(s: &FloatStatus) -> Self {
                $t::default_nan(s)
            }
            fn squash_input_denormal(self, s: &mut FloatStatus) -> Self {
                $t::squash_input_denormal(self, s)
            }
            fn to_int(
                self,
                bits: u32,
                signed: bool,
                rm: RoundMode,
                sc: i32,
                s: &mut FloatStatus,
            ) -> u64 {
                match (bits, signed) {
                    (16, true) => self.to_i16_scalbn(rm, sc, s) as u64,
                    (32, true) => self.to_i32_scalbn(rm, sc, s) as u64,
                    (64, true) => self.to_i64_scalbn(rm, sc, s) as u64,
                    (16, false) => u64::from(self.to_u16_scalbn(rm, sc, s)),
                    (32, false) => u64::from(self.to_u32_scalbn(rm, sc, s)),
                    _ => self.to_u64_scalbn(rm, sc, s),
                }
            }
            fn from_int(x: u64, signed: bool, sc: i32, s: &mut FloatStatus) -> Self {
                if signed {
                    $t::from_i64_scalbn(x as i64, sc, s)
                } else {
                    $t::from_u64_scalbn(x, sc, s)
                }
            }
        }
    };
}

impl_fp!(Float16, u16, 1, 10, 5);
impl_fp!(Float32, u32, 2, 23, 8);
impl_fp!(Float64, u64, 3, 52, 11);

/// The floating point operations of `a64_fp`, `a64_fp_cmp` and `a64_fp_gpr`.
#[allow(missing_docs)]
pub(crate) mod op {
    // d[i] = f(n[i], m[i]), or m[idx] for the indexed forms.
    pub(crate) const ADD: u32 = 1;
    pub(crate) const SUB: u32 = 2;
    pub(crate) const MUL: u32 = 3;
    pub(crate) const DIV: u32 = 4;
    pub(crate) const MAX: u32 = 5;
    pub(crate) const MIN: u32 = 6;
    pub(crate) const MAXNM: u32 = 7;
    pub(crate) const MINNM: u32 = 8;
    pub(crate) const MULX: u32 = 9;
    pub(crate) const ABD: u32 = 10;
    pub(crate) const NMUL: u32 = 11;
    pub(crate) const RECPS: u32 = 12;
    pub(crate) const RSQRTS: u32 = 13;
    pub(crate) const CEQ: u32 = 14;
    pub(crate) const CGE: u32 = 15;
    pub(crate) const CGT: u32 = 16;
    pub(crate) const ACGE: u32 = 17;
    pub(crate) const ACGT: u32 = 18;
    pub(crate) const MLA: u32 = 19;
    pub(crate) const MLS: u32 = 20;
    // Scalar fused multiply add with Ra.
    pub(crate) const MADD: u32 = 21;
    pub(crate) const MSUB: u32 = 22;
    pub(crate) const NMADD: u32 = 23;
    pub(crate) const NMSUB: u32 = 24;
    // Pairwise.
    pub(crate) const ADDP: u32 = 25;
    pub(crate) const MAXP: u32 = 26;
    pub(crate) const MINP: u32 = 27;
    pub(crate) const MAXNMP: u32 = 28;
    pub(crate) const MINNMP: u32 = 29;
    // Across lanes.
    pub(crate) const MAXV: u32 = 30;
    pub(crate) const MINV: u32 = 31;
    pub(crate) const MAXNMV: u32 = 32;
    pub(crate) const MINNMV: u32 = 33;
    // One source.
    pub(crate) const ABS: u32 = 40;
    pub(crate) const NEG: u32 = 41;
    pub(crate) const SQRT: u32 = 42;
    /// Round to integral, inexact suppressed.
    pub(crate) const RINT: u32 = 43;
    /// Round to integral, inexact raised (FRINTX).
    pub(crate) const RINTX: u32 = 44;
    pub(crate) const RECPE: u32 = 45;
    pub(crate) const RSQRTE: u32 = 46;
    pub(crate) const RECPX: u32 = 47;
    pub(crate) const CEQ0: u32 = 48;
    pub(crate) const CGE0: u32 = 49;
    pub(crate) const CGT0: u32 = 50;
    pub(crate) const CLE0: u32 = 51;
    pub(crate) const CLT0: u32 = 52;
    pub(crate) const MOV: u32 = 53;
    pub(crate) const URECPE: u32 = 54;
    pub(crate) const URSQRTE: u32 = 55;
    // Conversions; imm is the fixed point shift.
    pub(crate) const TOSINT: u32 = 60;
    pub(crate) const TOUINT: u32 = 61;
    pub(crate) const SCVTF: u32 = 62;
    pub(crate) const UCVTF: u32 = 63;
    /// Scalar precision conversion: esz is the source size, imm the destination size.
    pub(crate) const FCVT: u32 = 64;
    /// esz is the destination size.
    pub(crate) const FCVTL: u32 = 65;
    /// esz is the destination size.
    pub(crate) const FCVTN: u32 = 66;
    pub(crate) const FCVTXN: u32 = 67;
    // a64_fp_cmp.
    pub(crate) const CMP: u32 = 70;
    pub(crate) const CMPE: u32 = 71;
}

def!(FP, "a64_fp", 0, Void, [Ptr, I32, I32], h_fp);
def!(FP_CMP, "a64_fp_cmp", 0, I32, [Ptr, I32, I32], h_fp_cmp);
def!(FP_GPR, "a64_fp_gpr", 0, I64, [Ptr, I64, I32, I32], h_fp_gpr);

/// The helpers of this module.
pub(crate) const ALL: &[Def] = &[FP, FP_CMP, FP_GPR];

/// Run `f` with the status QEMU picks for an operation on elements of `esz` (`FPST_A64_F16`
/// for half precision), with the rounding mode of `desc` in force.
fn with_status<R>(
    env: &mut [u8],
    idx: usize,
    desc: &Desc,
    f: impl FnOnce(&mut FloatStatus, &mut [u8]) -> R,
) -> R {
    let mut s = load_status(env, idx);
    let saved = s.rounding_mode;
    if desc.rmode != RMODE_FPCR {
        s.rounding_mode = RoundMode::from_u8(desc.rmode as u8).unwrap_or(saved);
    }
    let r = f(&mut s, env);
    s.rounding_mode = saved;
    store_status(env, idx, &s);
    r
}

fn h_fp(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    let r = Regs::unpack(a[1]);
    let desc = Desc::unpack(a[2]);
    let env = &mut *h.env;
    let out = match desc.op {
        op::FCVT | op::FCVTL | op::FCVTN | op::FCVTXN => fcvt(env, r, &desc),
        op::URECPE | op::URSQRTE => {
            let n = vload(env, r.n);
            let mut out = [0u8; 16];
            for i in 0..desc.oprsz / 4 {
                let x = get(&n, 2, i) as u32;
                let y = if desc.op == op::URECPE { recpe_u32(x) } else { rsqrte_u32(x) };
                set(&mut out, 2, i, u64::from(y));
            }
            clear_tail(&mut out, desc.oprsz);
            out
        }
        _ => {
            let idx = usize::from(desc.esz == 1);
            with_status(env, idx, &desc, |s, env| match desc.esz {
                1 => fp_op::<Float16>(env, r, &desc, s),
                2 => fp_op::<Float32>(env, r, &desc, s),
                _ => fp_op::<Float64>(env, r, &desc, s),
            })
        }
    };
    vstore(env, r.d, &out);
    Ok(0)
}

fn mask_of(b: bool, esz: u32) -> u64 {
    if b { u64::MAX >> (64 - (8 << esz)) } else { 0 }
}

/// `float*_ceq`, `_cge`, `_cgt`, `_acge` and `_acgt` from `neon_helper.c`.
fn fcmp<F: Fp>(op: u32, a: F, b: F, s: &mut FloatStatus) -> bool {
    let rel = |x: F, y: F, s: &mut FloatStatus| x.compare(y, s);
    match op {
        op::CEQ => a.compare_quiet(b, s) == FloatRelation::Equal,
        op::CGE => matches!(rel(b, a, s), FloatRelation::Less | FloatRelation::Equal),
        op::CGT => rel(b, a, s) == FloatRelation::Less,
        op::ACGE => {
            matches!(rel(b.abs(), a.abs(), s), FloatRelation::Less | FloatRelation::Equal)
        }
        _ => rel(b.abs(), a.abs(), s) == FloatRelation::Less,
    }
}

/// One two-operand element operation; `d` is the accumulator of FMLA and FMLS.
fn binop<F: Fp>(op: u32, a: F, b: F, d: F, s: &mut FloatStatus) -> u64 {
    let r = match op {
        op::ADD | op::ADDP => a.add(b, s),
        op::SUB => a.sub(b, s),
        op::MUL => a.mul(b, s),
        op::DIV => a.div(b, s),
        op::MAX | op::MAXP | op::MAXV => a.max(b, s),
        op::MIN | op::MINP | op::MINV => a.min(b, s),
        op::MAXNM | op::MAXNMP | op::MAXNMV => a.maxnum(b, s),
        op::MINNM | op::MINNMP | op::MINNMV => a.minnum(b, s),
        op::MULX => {
            let a = a.squash_input_denormal(s);
            let b = b.squash_input_denormal(s);
            if (a.is_zero() && b.is_infinity()) || (a.is_infinity() && b.is_zero()) {
                // 2.0 with the sign of a XOR b.
                F::from_bits(F::two().bits() | ((a.bits() ^ b.bits()) & F::sign_bit()))
            } else {
                a.mul(b, s)
            }
        }
        op::ABD => a.sub(b, s).abs(),
        op::NMUL => a.mul(b, s).chs(),
        op::RECPS | op::RSQRTS => {
            let a = a.squash_input_denormal(s).chs();
            let b = b.squash_input_denormal(s);
            let special = (a.is_infinity() && b.is_zero()) || (b.is_infinity() && a.is_zero());
            if op == op::RECPS {
                if special { F::two() } else { a.muladd(b, F::two(), 0, s) }
            } else if special {
                F::one_point_five()
            } else {
                a.muladd_scalbn(b, F::three(), -1, 0, s)
            }
        }
        op::CEQ | op::CGE | op::CGT | op::ACGE | op::ACGT => {
            return mask_of(fcmp(op, a, b, s), F::ESZ);
        }
        op::MLA => a.muladd(b, d, 0, s),
        op::MLS => a.chs().muladd(b, d, 0, s),
        _ => unreachable!("bad a64_fp binary op {op}"),
    };
    r.bits()
}

/// `round_to_inf()` from `vfp_helper.c`.
fn round_to_inf(s: &FloatStatus, sign: bool) -> bool {
    match s.rounding_mode {
        RoundMode::NearestEven => true,
        RoundMode::Up => !sign,
        RoundMode::Down => sign,
        _ => false,
    }
}

/// The NaN result of the estimate helpers.
fn estimate_nan<F: Fp>(f: F, s: &mut FloatStatus) -> F {
    let mut nan = f;
    if f.is_signaling_nan(s) {
        s.raise(flags::INVALID);
        if !s.default_nan_mode {
            nan = f.silence_nan(s);
        }
    }
    if s.default_nan_mode {
        nan = F::default_nan(s);
    }
    nan
}

/// `recip_estimate()`.
fn recip_estimate(input: u32) -> u32 {
    let a = input * 2 + 1;
    let b = (1 << 19) / a;
    (b + 1) >> 1
}

/// `call_recip_estimate()` without increased precision.
fn call_recip_estimate(exp: &mut i32, exp_off: i32, frac: u64) -> u64 {
    let mut frac = frac;
    if *exp == 0 {
        if (frac >> 51) & 1 == 0 {
            *exp = -1;
            frac <<= 2;
        } else {
            frac <<= 1;
        }
    }
    let scaled = (1 << 8) | ((frac >> 44) & 0xff) as u32;
    let estimate = recip_estimate(scaled);
    let mut result_exp = exp_off - *exp;
    let mut result_frac = u64::from(estimate & 0xff) << 44;
    if result_exp == 0 {
        result_frac = (result_frac >> 1) | 1 << 51;
    } else if result_exp == -1 {
        result_frac = ((result_frac >> 2) & !(3 << 50)) | 1 << 50;
        result_exp = 0;
    }
    *exp = result_exp;
    result_frac
}

/// `HELPER(recpe_f16)`, `do_recpe_f32()` without FEAT_RPRES and `HELPER(recpe_f64)`.
fn recpe<F: Fp>(input: F, s: &mut FloatStatus) -> F {
    let f = input.squash_input_denormal(s);
    let v = f.bits();
    let sign = f.is_neg();
    let mut exp = ((v >> F::FRAC) & F::exp_mask()) as i32;
    let frac = v & F::frac_mask();
    let bias = (1i32 << (F::EXPB - 1)) - 1;
    let off = 2 * bias - 1;
    if f.is_any_nan() {
        return estimate_nan(f, s);
    } else if f.is_infinity() {
        return f.signed(0);
    } else if f.is_zero() {
        s.raise(flags::DIVBYZERO);
        return f.signed(F::infinity().bits());
    } else if v & !F::sign_bit() < 1 << (F::FRAC - 2) {
        s.raise(flags::OVERFLOW | flags::INEXACT);
        return if round_to_inf(s, sign) {
            f.signed(F::infinity().bits())
        } else {
            f.signed(F::maxnorm())
        };
    } else if exp >= off && s.flush_to_zero {
        s.raise(flags::UNDERFLOW);
        return f.signed(0);
    }
    let f64_frac = call_recip_estimate(&mut exp, off, frac << (52 - F::FRAC));
    let r =
        ((exp as u64) & F::exp_mask()) << F::FRAC | ((f64_frac >> (52 - F::FRAC)) & F::frac_mask());
    f.signed(r)
}

/// `do_recip_sqrt_estimate()`.
fn recip_sqrt_estimate_int(a: u32) -> u32 {
    let mut a = a;
    if a < 256 {
        a = a * 2 + 1;
    } else {
        a = (a >> 1) << 1;
        a = (a + 1) * 2;
    }
    let mut b: u32 = 512;
    while a * (b + 1) * (b + 1) < (1 << 28) {
        b += 1;
    }
    b.div_ceil(2)
}

/// `recip_sqrt_estimate()` without increased precision.
fn recip_sqrt_estimate(exp: &mut i32, exp_off: i32, frac: u64) -> u64 {
    let mut frac = frac;
    if *exp == 0 {
        while (frac >> 51) & 1 == 0 {
            frac <<= 1;
            *exp -= 1;
        }
        frac = (frac & ((1 << 51) - 1)) << 1;
    }
    let scaled = if *exp & 1 != 0 {
        (1 << 7) | ((frac >> 45) & 0x7f) as u32
    } else {
        (1 << 8) | ((frac >> 44) & 0xff) as u32
    };
    let estimate = recip_sqrt_estimate_int(scaled);
    *exp = (exp_off - *exp) / 2;
    u64::from(estimate & 0xff) << 44
}

/// `HELPER(rsqrte_f16)`, `do_rsqrte_f32()` without FEAT_RPRES and `HELPER(rsqrte_f64)`.
fn rsqrte<F: Fp>(input: F, s: &mut FloatStatus) -> F {
    let f = input.squash_input_denormal(s);
    let v = f.bits();
    let mut exp = ((v >> F::FRAC) & F::exp_mask()) as i32;
    let frac = v & F::frac_mask();
    let bias = (1i32 << (F::EXPB - 1)) - 1;
    if f.is_any_nan() {
        return estimate_nan(f, s);
    } else if f.is_zero() {
        s.raise(flags::DIVBYZERO);
        return f.signed(F::infinity().bits());
    } else if f.is_neg() {
        s.raise(flags::INVALID);
        return F::default_nan(s);
    } else if f.is_infinity() {
        return F::zero();
    }
    let f64_frac = recip_sqrt_estimate(&mut exp, 3 * bias - 1, frac << (52 - F::FRAC));
    // The sign is known to be clear here.
    F::from_bits(
        ((exp as u64) & F::exp_mask()) << F::FRAC | ((f64_frac >> 44) & 0xff) << (F::FRAC - 8),
    )
}

/// `HELPER(frecpx_f16)`, `_f32` and `_f64`.
fn recpx<F: Fp>(a: F, s: &mut FloatStatus) -> F {
    if a.is_any_nan() {
        return estimate_nan(a, s);
    }
    let a = a.squash_input_denormal(s);
    let exp = (a.bits() >> F::FRAC) & F::exp_mask();
    let e = if exp == 0 { F::exp_mask() - 1 } else { !exp & F::exp_mask() };
    a.signed(e << F::FRAC)
}

/// `HELPER(recpe_u32)`.
pub(crate) fn recpe_u32(a: u32) -> u32 {
    if a & 0x8000_0000 == 0 {
        return 0xffff_ffff;
    }
    recip_estimate((a >> 23) & 0x1ff) << 23
}

/// `HELPER(rsqrte_u32)`.
pub(crate) fn rsqrte_u32(a: u32) -> u32 {
    if a & 0xc000_0000 == 0 {
        return 0xffff_ffff;
    }
    recip_sqrt_estimate_int((a >> 23) & 0x1ff) << 23
}

/// `HELPER(rints)`: round in the status rounding mode, without raising inexact.
fn rint<F: Fp>(a: F, exact: bool, s: &mut FloatStatus) -> F {
    let old = s.exception_flags;
    let r = a.round_to_int(s);
    if !exact && old & flags::INEXACT == 0 {
        s.exception_flags &= !flags::INEXACT;
    }
    r
}

/// The float to integer conversions of `VFP_CONV_FLOAT_FIX_ROUND`: a NaN raises invalid and
/// gives 0.
fn to_int<F: Fp>(a: F, bits: u32, signed: bool, shift: i32, s: &mut FloatStatus) -> u64 {
    if a.is_any_nan() {
        s.raise(flags::INVALID);
        return 0;
    }
    let rm = s.rounding_mode;
    a.to_int(bits, signed, rm, shift, s)
}

/// One one-operand element operation.
fn unop<F: Fp>(op: u32, a: F, shift: i32, s: &mut FloatStatus) -> u64 {
    let bits = 8 << F::ESZ;
    let z = F::zero();
    let r = match op {
        op::ABS => a.abs(),
        op::NEG => a.chs(),
        op::MOV => a,
        op::SQRT => a.sqrt(s),
        op::RINT => rint(a, false, s),
        op::RINTX => rint(a, true, s),
        op::RECPE => recpe(a, s),
        op::RSQRTE => rsqrte(a, s),
        op::RECPX => recpx(a, s),
        op::CEQ0 => return mask_of(fcmp(op::CEQ, a, z, s), F::ESZ),
        op::CGE0 => return mask_of(fcmp(op::CGE, a, z, s), F::ESZ),
        op::CGT0 => return mask_of(fcmp(op::CGT, a, z, s), F::ESZ),
        op::CLE0 => return mask_of(fcmp(op::CGE, z, a, s), F::ESZ),
        op::CLT0 => return mask_of(fcmp(op::CGT, z, a, s), F::ESZ),
        op::TOSINT => return to_int(a, bits, true, shift, s),
        op::TOUINT => return to_int(a, bits, false, shift, s),
        op::SCVTF => {
            let x = super::vec_helper::sext(a.bits(), F::ESZ) as u64;
            F::from_int(x, true, -shift, s)
        }
        op::UCVTF => F::from_int(a.bits(), false, -shift, s),
        _ => unreachable!("bad a64_fp unary op {op}"),
    };
    r.bits()
}

/// `do_reduction_op()`: combine the halves recursively.
fn reduce<F: Fp>(op: u32, v: &V, base: usize, count: usize, s: &mut FloatStatus) -> F {
    if count == 1 {
        return F::from_bits(get(v, F::ESZ, base));
    }
    let half = count / 2;
    let lo = reduce::<F>(op, v, base, half, s);
    let hi = reduce::<F>(op, v, base + half, half, s);
    F::from_bits(binop(op, lo, hi, lo, s))
}

/// The operations on elements of one format.
fn fp_op<F: Fp>(env: &[u8], r: Regs, desc: &Desc, s: &mut FloatStatus) -> V {
    let esz = F::ESZ;
    let n = vload(env, r.n);
    let m = vload(env, r.m);
    let d = vload(env, r.d);
    let elems = desc.oprsz >> esz;
    let el = |v: &V, i: usize| F::from_bits(get(v, esz, i));
    let mut out = [0u8; 16];
    match desc.op {
        op::ADD..=op::MLS => {
            for i in 0..elems {
                let b = if desc.idx { el(&m, desc.imm as usize) } else { el(&m, i) };
                set(&mut out, esz, i, binop(desc.op, el(&n, i), b, el(&d, i), s));
            }
        }
        op::MADD..=op::NMSUB => {
            let a = el(&vload(env, r.a), 0);
            let (nn, aa) = match desc.op {
                op::MADD => (el(&n, 0), a),
                op::MSUB => (el(&n, 0).chs(), a),
                op::NMADD => (el(&n, 0).chs(), a.chs()),
                _ => (el(&n, 0), a.chs()),
            };
            set(&mut out, esz, 0, nn.muladd(el(&m, 0), aa, 0, s).bits());
        }
        op::ADDP..=op::MINNMP => {
            if elems == 1 {
                set(&mut out, esz, 0, binop(desc.op, el(&n, 0), el(&n, 1), el(&n, 0), s));
            } else {
                let half = elems / 2;
                for i in 0..elems {
                    let (src, j) = if i < half { (&n, 2 * i) } else { (&m, 2 * (i - half)) };
                    let x = binop(desc.op, el(src, j), el(src, j + 1), el(src, j), s);
                    set(&mut out, esz, i, x);
                }
            }
        }
        op::MAXV..=op::MINNMV => {
            set(&mut out, esz, 0, reduce::<F>(desc.op, &n, 0, elems, s).bits());
            clear_tail(&mut out, 1 << esz);
            return out;
        }
        op::ABS..=op::RECPX | op::CEQ0..=op::MOV | op::TOSINT..=op::UCVTF => {
            for i in 0..elems {
                set(&mut out, esz, i, unop(desc.op, el(&n, i), desc.imm as i32, s));
            }
        }
        _ => unreachable!("bad a64_fp op {}", desc.op),
    }
    clear_tail(&mut out, desc.oprsz);
    out
}

/// The status QEMU uses for a conversion and whether to use IEEE half precision.
fn fcvt(env: &mut [u8], r: Regs, desc: &Desc) -> V {
    let ieee = env_fpcr(env) & FPCR_AHP == 0;
    let n = vload(env, r.n);
    let d = vload(env, r.d);
    let mut out = [0u8; 16];
    match desc.op {
        op::FCVT => {
            let (from, to) = (desc.esz, desc.imm);
            let idx = usize::from(from == 1);
            with_status(env, idx, desc, |s, _| {
                let x = get(&n, from, 0);
                let y = cvt(x, from, to, ieee, s);
                set(&mut out, to, 0, y);
            });
            clear_tail(&mut out, 1 << desc.imm);
        }
        op::FCVTL => {
            let to = desc.esz;
            let from = to - 1;
            // The 8 source bytes hold 16 >> to elements.
            let count = 16 >> to;
            let base = if desc.hi { count } else { 0 };
            let idx = usize::from(from == 1);
            with_status(env, idx, desc, |s, _| {
                for i in 0..count {
                    let y = cvt(get(&n, from, base + i), from, to, ieee, s);
                    set(&mut out, to, i, y);
                }
            });
        }
        op::FCVTN | op::FCVTXN => {
            let to = desc.esz;
            let from = to + 1;
            let count = if desc.oprsz < 8 { 1 } else { 8 >> to };
            let mut lo = [0u8; 16];
            with_status(env, 0, desc, |s, _| {
                for i in 0..count {
                    let x = get(&n, from, i);
                    let y = if desc.op == op::FCVTXN {
                        let old = s.rounding_mode;
                        s.rounding_mode = RoundMode::ToOdd;
                        let y = Float64(x).to_float32(s);
                        s.rounding_mode = old;
                        u64::from(y.0)
                    } else {
                        cvt(x, from, to, ieee, s)
                    };
                    set(&mut lo, to, i, y);
                }
            });
            if desc.hi {
                out[..8].copy_from_slice(&d[..8]);
                out[8..].copy_from_slice(&lo[..8]);
            } else {
                out[..8].copy_from_slice(&lo[..8]);
                clear_tail(&mut out, if desc.oprsz < 8 { 4 } else { 8 });
            }
        }
        _ => unreachable!(),
    }
    out
}

/// One precision conversion as QEMU's helpers do it: the half precision conversions turn
/// flushing off for the duration, on the input side when widening and the output side
/// when narrowing.
fn cvt(x: u64, from: u32, to: u32, ieee: bool, s: &mut FloatStatus) -> u64 {
    match (from, to) {
        (2, 3) => Float32(x as u32).to_float64(s).0,
        (3, 2) => u64::from(Float64(x).to_float32(s).0),
        (1, _) => {
            let save = s.flush_inputs_to_zero;
            s.flush_inputs_to_zero = false;
            let y = if to == 2 {
                u64::from(Float16(x as u16).to_float32(ieee, s).0)
            } else {
                Float16(x as u16).to_float64(ieee, s).0
            };
            s.flush_inputs_to_zero = save;
            y
        }
        (_, 1) => {
            let save = s.flush_to_zero;
            s.flush_to_zero = false;
            let y = if from == 2 {
                Float32(x as u32).to_float16(ieee, s)
            } else {
                Float64(x).to_float16(ieee, s)
            };
            s.flush_to_zero = save;
            u64::from(y.0)
        }
        _ => unreachable!("bad fcvt {from} {to}"),
    }
}

/// `HELPER(vfp_cmp*_a64)`: FCMP and FCMPE, returning NZCV in bits 31 to 28. Bit 0 of the
/// immediate compares with +0.0 instead of Rm.
fn h_fp_cmp(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    let r = Regs::unpack(a[1]);
    let desc = Desc::unpack(a[2]);
    let env = &mut *h.env;
    let n = get(&vload(env, r.n), desc.esz, 0);
    let m = if desc.imm & 1 != 0 { 0 } else { get(&vload(env, r.m), desc.esz, 0) };
    let sig = desc.op == op::CMPE;
    let idx = usize::from(desc.esz == 1);
    let rel = with_status(env, idx, &desc, |s, _| match desc.esz {
        1 => cmp_rel(Float16(n as u16), Float16(m as u16), sig, s),
        2 => cmp_rel(Float32(n as u32), Float32(m as u32), sig, s),
        _ => cmp_rel(Float64(n), Float64(m), sig, s),
    });
    let nzcv: u32 = match rel {
        FloatRelation::Equal => 0x6,
        FloatRelation::Less => 0x8,
        FloatRelation::Greater => 0x2,
        FloatRelation::Unordered => 0x3,
    };
    Ok(u128::from(nzcv << 28))
}

fn cmp_rel<F: Fp>(a: F, b: F, sig: bool, s: &mut FloatStatus) -> FloatRelation {
    if sig { a.compare(b, s) } else { a.compare_quiet(b, s) }
}

/// The conversions between a general register and a floating point register, with a fixed
/// point shift in the immediate. `hi` says the general register is 64 bits.
///
/// TOSINT and TOUINT read element 0 of Rn and return the integer; SCVTF and UCVTF convert
/// the argument and write Rd, clearing the rest of it.
fn h_fp_gpr(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    let x = a[1];
    let r = Regs::unpack(a[2]);
    let desc = Desc::unpack(a[3]);
    let env = &mut *h.env;
    let esz = desc.esz;
    let bits = if desc.hi { 64 } else { 32 };
    let shift = desc.imm as i32;
    let idx = usize::from(esz == 1);
    match desc.op {
        op::TOSINT | op::TOUINT => {
            let v = get(&vload(env, r.n), esz, 0);
            let signed = desc.op == op::TOSINT;
            let y = with_status(env, idx, &desc, |s, _| match esz {
                1 => to_int(Float16(v as u16), bits, signed, shift, s),
                2 => to_int(Float32(v as u32), bits, signed, shift, s),
                _ => to_int(Float64(v), bits, signed, shift, s),
            });
            Ok(u128::from(if desc.hi { y } else { y & 0xffff_ffff }))
        }
        _ => {
            let signed = desc.op == op::SCVTF;
            let x = match (desc.hi, signed) {
                (true, _) => x,
                (false, true) => x as i32 as i64 as u64,
                (false, false) => x & 0xffff_ffff,
            };
            let y = with_status(env, idx, &desc, |s, _| match esz {
                1 => Float16::from_int(x, signed, -shift, s).bits(),
                2 => Float32::from_int(x, signed, -shift, s).bits(),
                _ => Float64::from_int(x, signed, -shift, s).bits(),
            });
            let mut out = [0u8; 16];
            set(&mut out, esz, 0, y);
            vstore(env, r.d, &out);
            Ok(0)
        }
    }
}

/// The value of an 8 bit FMOV immediate, `VFPExpandImm()`.
pub(crate) fn vfp_expand_imm(esz: u32, imm8: u32) -> u64 {
    let sign = u64::from(imm8 >> 7);
    let b6 = u64::from((imm8 >> 6) & 1);
    let lo6 = u64::from(imm8 & 0x3f);
    match esz {
        1 => sign << 15 | (b6 ^ 1) << 14 | (if b6 != 0 { 0x3 } else { 0 }) << 12 | lo6 << 6,
        2 => sign << 31 | (b6 ^ 1) << 30 | (if b6 != 0 { 0x1f } else { 0 }) << 25 | lo6 << 19,
        _ => sign << 63 | (b6 ^ 1) << 62 | (if b6 != 0 { 0xff } else { 0 }) << 54 | lo6 << 48,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_round_trips() {
        let mut s = FloatStatus::arm();
        s.rounding_mode = RoundMode::ToZero;
        s.flush_to_zero = true;
        s.default_nan_mode = true;
        s.exception_flags = flags::INEXACT | flags::INVALID;
        let t = unpack_status(pack_status(&s));
        assert_eq!(t.rounding_mode, RoundMode::ToZero);
        assert!(t.flush_to_zero && !t.flush_inputs_to_zero && t.default_nan_mode);
        assert_eq!(t.exception_flags, s.exception_flags);
        assert_eq!(pack_status(&FloatStatus::arm()), 0);
    }

    #[test]
    fn expand_imm() {
        assert_eq!(vfp_expand_imm(3, 0x70), 0x3ff0_0000_0000_0000);
        assert_eq!(vfp_expand_imm(2, 0x00), 0x4000_0000);
        assert_eq!(vfp_expand_imm(1, 0x70), 0x3c00);
    }

    #[test]
    fn constants() {
        assert_eq!(Float32::two().0, 0x4000_0000);
        assert_eq!(Float32::three().0, 0x4040_0000);
        assert_eq!(Float64::one_point_five().0, 0x3ff8_0000_0000_0000);
        assert_eq!(Float16::maxnorm(), 0x7bff);
        assert_eq!(Float16::three().0, 0x4200);
    }

    #[test]
    fn estimates() {
        let mut s = FloatStatus::arm();
        // FRECPE 1.0 = 0.998046875, FRSQRTE 1.0 = 0.998046875, FRSQRTE 4.0 = 0.4990234375.
        assert_eq!(recpe(Float32(0x3f80_0000), &mut s).0, 0x3f7f_8000);
        assert_eq!(rsqrte(Float32(0x3f80_0000), &mut s).0, 0x3f7f_8000);
        assert_eq!(rsqrte(Float32(0x4080_0000), &mut s).0, 0x3eff_8000);
        assert_eq!(recpx(Float64(0x3ff0_0000_0000_0000), &mut s).0, 0x4000_0000_0000_0000);
    }
}
