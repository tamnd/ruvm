// SPDX-License-Identifier: GPL-2.0-or-later

//! The vector floating point helpers, the floating point parts of QEMU's
//! `target/riscv/tcg/vector_helper.c` from "Vector Single-Width Floating-Point Add/Subtract
//! Instructions" to "Vector Widening Floating-Point Reduction Instructions", with the
//! Zvfbfmin conversions and the Zvfbfwma multiply-add.
//!
//! As elsewhere, a helper serves every SEW and gets its registers in the descriptor, and a
//! `.vf` helper is the `.vv` helper with [`Desc::scalar`] set: the NaN boxed scalar is the
//! third argument, which the `.vv` forms ignore. Every helper takes `env`, the descriptor
//! and that scalar.
//!
//! The helpers that round or raise flags start from the `fp_status` of `env` with no flags
//! and accrue what they raise into `fflags` at the end with `riscv_cpu_check_fflags()`,
//! which also makes `mstatus.FS` dirty when a flag is new ([`fpu::check_fflags`]).
//!
//! Deliberate differences from QEMU:
//!
//! - QEMU's `riscv_cpu_check_fflags()` compares every softfloat flag, including the ones
//!   that are not RISC-V flags, such as input denormal; here only the five `fflags` count,
//!   so `mstatus.FS` is not dirtied by a flag RISC-V cannot see.
//! - Zvfbfa is off, so `vtype.altfmt` is never set and the `_bf16` arithmetic helpers of
//!   QEMU are not here.
//! - `vfmv.v.f` uses [`VFMERGE`] with `vm` set, which writes what QEMU's `vmv_v_x`
//!   helpers and its inline splat write.

use ruvm_jit_core::HelperType::{I32, I64, Ptr, Void};
use ruvm_jit_core::types::call_flags::NO_RWG;
use ruvm_softfloat::{
    BFloat16, Float16, Float32, Float64, FloatRelation, FloatStatus, RoundMode, flags, muladd,
};

use super::fpu;
use super::helpers::Def;
use super::vector::{
    Desc, HR, Shape, def, for_each, set_1s, set_vstart, sext, src1, total_elems, vget, vl, vmask,
    vset, vset_mask, vstart,
};
use crate::cpu::VLENB;

// The formats.

/// A floating point format with its raw bits.
trait Fp: Copy {
    /// The width of the exponent.
    const EXP: u32;
    /// The width of the fraction.
    const FRAC: u32;
    /// The value with the low bits of `v`.
    fn of(v: u64) -> Self;
    /// The raw bits, zero extended.
    fn bits(self) -> u64;
}

macro_rules! fp_impl {
    ($t:ident, $raw:ty, $exp:literal, $frac:literal) => {
        impl Fp for $t {
            const EXP: u32 = $exp;
            const FRAC: u32 = $frac;
            fn of(v: u64) -> Self {
                $t(v as $raw)
            }
            fn bits(self) -> u64 {
                u64::from(self.0)
            }
        }
    };
}

fp_impl!(Float16, u16, 5, 10);
fp_impl!(BFloat16, u16, 8, 7);
fp_impl!(Float32, u32, 8, 23);
fp_impl!(Float64, u64, 11, 52);

/// `$e` with `$t` the format of `1 << log2` bytes: half, single or double precision.
macro_rules! per_sew {
    ($log2:expr, $t:ident => $e:expr) => {
        match $log2 {
            1 => {
                type $t = Float16;
                $e
            }
            2 => {
                type $t = Float32;
                $e
            }
            _ => {
                type $t = Float64;
                $e
            }
        }
    };
}

// The element operations. `sew` is log2 of the size in bytes of the operands.

fn fadd(a: u64, b: u64, sew: u32, s: &mut FloatStatus) -> u64 {
    per_sew!(sew, T => T::of(a).add(T::of(b), s).bits())
}

fn fsub(a: u64, b: u64, sew: u32, s: &mut FloatStatus) -> u64 {
    per_sew!(sew, T => T::of(a).sub(T::of(b), s).bits())
}

fn fmul(a: u64, b: u64, sew: u32, s: &mut FloatStatus) -> u64 {
    per_sew!(sew, T => T::of(a).mul(T::of(b), s).bits())
}

fn fdiv(a: u64, b: u64, sew: u32, s: &mut FloatStatus) -> u64 {
    per_sew!(sew, T => T::of(a).div(T::of(b), s).bits())
}

fn fmin(a: u64, b: u64, sew: u32, s: &mut FloatStatus) -> u64 {
    per_sew!(sew, T => T::of(a).minimum_number(T::of(b), s).bits())
}

fn fmax(a: u64, b: u64, sew: u32, s: &mut FloatStatus) -> u64 {
    per_sew!(sew, T => T::of(a).maximum_number(T::of(b), s).bits())
}

/// `a * b + c` with the `float_muladd_*` flags `fl`.
fn fma(a: u64, b: u64, c: u64, sew: u32, fl: u32, s: &mut FloatStatus) -> u64 {
    per_sew!(sew, T => T::of(a).muladd(T::of(b), T::of(c), fl, s).bits())
}

fn fsqrt(a: u64, sew: u32, s: &mut FloatStatus) -> u64 {
    per_sew!(sew, T => T::of(a).sqrt(s).bits())
}

/// `float16_to_float32(a, true)` or `float32_to_float64()`: a SEW float at 2*SEW.
fn widen(a: u64, sew: u32, s: &mut FloatStatus) -> u64 {
    if sew == 1 {
        Float16::of(a).to_float32(true, s).bits()
    } else {
        Float32::of(a).to_float64(s).bits()
    }
}

/// `float32_to_float16(a, true)` or `float64_to_float32()`: a 2*SEW float at SEW `dsew`.
fn narrow(a: u64, dsew: u32, s: &mut FloatStatus) -> u64 {
    if dsew == 1 {
        Float32::of(a).to_float16(true, s).bits()
    } else {
        Float64::of(a).to_float32(s).bits()
    }
}

/// The float of `1 << flog2` bytes `a` as an integer of `1 << ilog2` bytes, with the
/// rounding mode of `s`.
fn ftoi(a: u64, flog2: u32, ilog2: u32, signed: bool, s: &mut FloatStatus) -> u64 {
    per_sew!(flog2, T => {
        let f = T::of(a);
        match (ilog2, signed) {
            (0, false) => u64::from(f.to_u8(s)),
            (0, true) => f.to_i8(s) as u64,
            (1, false) => u64::from(f.to_u16(s)),
            (1, true) => f.to_i16(s) as u64,
            (2, false) => u64::from(f.to_u32(s)),
            (2, true) => f.to_i32(s) as u64,
            (_, false) => f.to_u64(s),
            (_, true) => f.to_i64(s) as u64,
        }
    })
}

/// The integer of `1 << ilog2` bytes `a` as a float of `1 << flog2` bytes.
fn itof(a: u64, ilog2: u32, flog2: u32, signed: bool, s: &mut FloatStatus) -> u64 {
    per_sew!(flog2, T => {
        if signed { T::from_i64(sext(a, ilog2), s).bits() } else { T::from_u64(a, s).bits() }
    })
}

/// The mask of the low `n` bits.
const fn mask(n: u32) -> u64 {
    if n >= 64 { u64::MAX } else { (1 << n) - 1 }
}

/// `deposit64()` of the sign, exponent and fraction fields.
fn pack(sign: u64, exp: u64, frac: u64, exp_size: u32, frac_size: u32) -> u64 {
    (frac & mask(frac_size))
        | ((exp & mask(exp_size)) << frac_size)
        | (sign << (exp_size + frac_size))
}

/// The table of `frsqrt7()`.
const RSQRT7_TABLE: [u8; 128] = [
    52, 51, 50, 48, 47, 46, 44, 43, 42, 41, 40, 39, 38, 36, 35, 34, 33, 32, 31, 30, 30, 29, 28, 27,
    26, 25, 24, 23, 23, 22, 21, 20, 19, 19, 18, 17, 16, 16, 15, 14, 14, 13, 12, 12, 11, 10, 10, 9,
    9, 8, 7, 7, 6, 6, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 127, 125, 123, 121, 119, 118, 116, 114, 113,
    111, 109, 108, 106, 105, 103, 102, 100, 99, 97, 96, 95, 93, 92, 91, 90, 88, 87, 86, 85, 84, 83,
    82, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 70, 69, 68, 67, 66, 65, 64, 63, 63, 62, 61, 60,
    59, 59, 58, 57, 56, 56, 55, 54, 53,
];

/// The table of `frec7()`.
const REC7_TABLE: [u8; 128] = [
    127, 125, 123, 121, 119, 117, 116, 114, 112, 110, 109, 107, 105, 104, 102, 100, 99, 97, 96, 94,
    93, 91, 90, 88, 87, 85, 84, 83, 81, 80, 79, 77, 76, 75, 74, 72, 71, 70, 69, 68, 66, 65, 64, 63,
    62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 40,
    39, 38, 37, 36, 35, 35, 34, 33, 32, 31, 31, 30, 29, 28, 28, 27, 26, 25, 25, 24, 23, 23, 22, 21,
    21, 20, 19, 19, 18, 17, 17, 16, 15, 15, 14, 14, 13, 12, 12, 11, 11, 10, 9, 9, 8, 8, 7, 7, 6, 5,
    5, 4, 4, 3, 3, 2, 2, 1, 1, 0,
];

/// The sign, exponent and fraction of `f`, a subnormal normalized with an exponent that
/// may be negative (wrapped to 64 bits) and its leading one dropped, as `frsqrt7()` and
/// `frec7()` start.
fn unpack7(f: u64, exp_size: u32, frac_size: u32) -> (u64, u64, u64, bool) {
    let sign = (f >> (frac_size + exp_size)) & 1;
    let mut exp = (f >> frac_size) & mask(exp_size);
    let mut frac = f & mask(frac_size);
    let sub = exp == 0 && frac != 0;
    if sub {
        while (frac >> (frac_size - 1)) & 1 == 0 {
            exp = exp.wrapping_sub(1);
            frac <<= 1;
        }
        frac = (frac << 1) & mask(frac_size);
    }
    (sign, exp, frac, sub)
}

/// `frsqrt7()` of a positive normal or subnormal.
fn frsqrt7(f: u64, exp_size: u32, frac_size: u32) -> u64 {
    let (sign, exp, frac, _) = unpack7(f, exp_size, frac_size);
    let idx = ((exp & 1) << 6) | (frac >> (frac_size - 6));
    let out_frac = u64::from(RSQRT7_TABLE[idx as usize]) << (frac_size - 7);
    let out_exp = 3u64.wrapping_mul(mask(exp_size - 1)).wrapping_add(!exp) / 2;
    pack(sign, out_exp, out_frac, exp_size, frac_size)
}

/// `frec7()` of a normal or subnormal.
fn frec7(f: u64, exp_size: u32, frac_size: u32, s: &mut FloatStatus) -> u64 {
    let (sign, exp, frac, sub) = unpack7(f, exp_size, frac_size);
    if sub && exp != 0 && exp != u64::MAX {
        // Overflow to inf or the largest finite value of the same sign, depending on the
        // sign and the rounding mode.
        s.raise(flags::INEXACT | flags::OVERFLOW);
        let inf = mask(exp_size) << frac_size;
        let rm = s.rounding_mode;
        let finite = rm == RoundMode::ToZero
            || (rm == RoundMode::Down && sign == 0)
            || (rm == RoundMode::Up && sign != 0);
        return (sign << (exp_size + frac_size)) | if finite { inf - 1 } else { inf };
    }
    let idx = frac >> (frac_size - 7);
    let mut out_frac = u64::from(REC7_TABLE[idx as usize]) << (frac_size - 7);
    let mut out_exp = 2u64.wrapping_mul(mask(exp_size - 1)).wrapping_add(!exp);
    if out_exp == 0 || out_exp == u64::MAX {
        // The result is subnormal, but there is no underflow as no precision is lost.
        out_frac = (out_frac >> 1) | (1 << (frac_size - 1));
        if out_exp == u64::MAX {
            out_frac >>= 1;
            out_exp = 0;
        }
    }
    pack(sign, out_exp, out_frac, exp_size, frac_size)
}

/// `frsqrt7_h()`, `frsqrt7_s()` and `frsqrt7_d()`.
fn rsqrt7(a: u64, sew: u32, s: &mut FloatStatus) -> u64 {
    per_sew!(sew, T => {
        let f = T::of(a);
        let sign = f.is_neg();
        if f.is_signaling_nan(s)
            || (f.is_infinity() && sign)
            || (f.is_normal() && sign)
            || (f.is_zero_or_denormal() && !f.is_zero() && sign)
        {
            // sNaN, -inf, -normal and -subnormal give the canonical NaN.
            s.raise(flags::INVALID);
            T::default_nan(s).bits()
        } else if f.is_quiet_nan(s) {
            T::default_nan(s).bits()
        } else if f.is_zero() {
            s.raise(flags::DIVBYZERO);
            T::of(mask(T::EXP) << T::FRAC).set_sign(sign).bits()
        } else if f.is_infinity() {
            0
        } else {
            frsqrt7(a, T::EXP, T::FRAC)
        }
    })
}

/// `frec7_h()`, `frec7_s()` and `frec7_d()`.
fn rec7(a: u64, sew: u32, s: &mut FloatStatus) -> u64 {
    per_sew!(sew, T => {
        let f = T::of(a);
        let sign = f.is_neg();
        if f.is_infinity() {
            T::of(0).set_sign(sign).bits()
        } else if f.is_zero() {
            s.raise(flags::DIVBYZERO);
            T::of(mask(T::EXP) << T::FRAC).set_sign(sign).bits()
        } else if f.is_signaling_nan(s) {
            s.raise(flags::INVALID);
            T::default_nan(s).bits()
        } else if f.is_quiet_nan(s) {
            T::default_nan(s).bits()
        } else {
            frec7(a, T::EXP, T::FRAC, s)
        }
    })
}

/// `fclass_h()`, `fclass_s()` and `fclass_d()`.
fn fclass(a: u64, sew: u32) -> u64 {
    per_sew!(sew, T => {
        let f = T::of(a);
        let sign = f.is_neg();
        if f.is_infinity() {
            if sign { 1 << 0 } else { 1 << 7 }
        } else if f.is_zero() {
            if sign { 1 << 3 } else { 1 << 4 }
        } else if f.is_zero_or_denormal() {
            if sign { 1 << 2 } else { 1 << 5 }
        } else if f.is_any_nan() {
            // A default status for snan_bit_is_one.
            if f.is_quiet_nan(&FloatStatus::default()) { 1 << 9 } else { 1 << 8 }
        } else if sign {
            1 << 1
        } else {
            1 << 6
        }
    })
}

/// The sign bit of a SEW float.
fn sign_bit(sew: u32) -> u64 {
    1 << ((8 << sew) - 1)
}

// The loops.

/// The descriptor and the scalar of a helper call.
fn args(a: &[u64]) -> (Desc, u64) {
    (Desc::decode(a[1] as u32), a[2])
}

/// `do_vext_vv()` with a float status: `body(env, i, s)` gives element `i` of `1 << dl`
/// bytes of `vd`, then the flags are accrued.
fn fp_each(
    env: &mut [u8],
    d: &Desc,
    dl: u32,
    mut body: impl FnMut(&[u8], usize, &mut FloatStatus) -> u64,
) {
    let mut s = fpu::status(env);
    for_each(env, d, dl, |env, i| {
        let v = body(env, i, &mut s);
        vset(env, d.vd, i, dl, v);
    });
    fpu::check_fflags(env, &s);
}

/// `OPFVV2` and `OPFVF2`: `vd[i] = op(vs2[i], s1, sew)`, the scalar or `vs1[i]` as `s1`,
/// with the sizes of `shape`.
fn fop2(env: &mut [u8], a: &[u64], shape: Shape, op: fn(u64, u64, u32, &mut FloatStatus) -> u64) {
    let (d, scalar) = args(a);
    let sew = d.esz;
    let (dl, l2, l1) = shape.log2(sew);
    fp_each(env, &d, dl, |env, i, s| {
        let s2 = vget(env, d.vs2, i, l2);
        let s1 = src1(env, &d, i, l1, scalar);
        op(s2, s1, sew, s)
    });
}

/// `OPFVV3` and `OPFVF3`: `vd[i] = op(vs2[i], s1, vd[i], sew)`, with SEW sources and a
/// 2*SEW `vd` when `wide`.
fn fop3(
    env: &mut [u8],
    a: &[u64],
    wide: bool,
    op: fn(u64, u64, u64, u32, &mut FloatStatus) -> u64,
) {
    let (d, scalar) = args(a);
    let sew = d.esz;
    let dl = if wide { sew + 1 } else { sew };
    fp_each(env, &d, dl, |env, i, s| {
        let s2 = vget(env, d.vs2, i, sew);
        let s1 = src1(env, &d, i, sew, scalar);
        let vd = vget(env, d.vd, i, dl);
        op(s2, s1, vd, sew, s)
    });
}

/// `OPFVV1`: `vd[i] = op(vs2[i], sew)` with the destination and source sizes of `shape`.
fn fop1(env: &mut [u8], a: &[u64], shape: Shape, op: fn(u64, u32, &mut FloatStatus) -> u64) {
    let (d, _) = args(a);
    let sew = d.esz;
    let (dl, l2, _) = shape.log2(sew);
    fp_each(env, &d, dl, |env, i, s| op(vget(env, d.vs2, i, l2), sew, s));
}

/// `GEN_VEXT_CMP_VV_ENV()` and `GEN_VEXT_CMP_VF()`: mask bit `i` of `vd` is
/// `op(vs2[i], s1)`.
fn fcmp(env: &mut [u8], a: &[u64], op: fn(u64, u64, u32, &mut FloatStatus) -> bool) {
    let (d, scalar) = args(a);
    let sew = d.esz;
    let vl = vl(env);
    let mut s = fpu::status(env);
    let start = vstart(env);
    if start >= vl {
        // VSTART_CHECK_EARLY_EXIT().
        set_vstart(env, 0);
        return;
    }
    for i in start..vl {
        if !d.vm && !vmask(env, 0, i) {
            // Set the masked off elements to ones.
            if d.vma {
                vset_mask(env, d.vd, i, true);
            }
            continue;
        }
        let s2 = vget(env, d.vs2, i, sew);
        let s1 = src1(env, &d, i, sew, scalar);
        let r = op(s2, s1, sew, &mut s);
        vset_mask(env, d.vd, i, r);
    }
    set_vstart(env, 0);
    // A mask destination is always tail agnostic.
    if d.vta_all_1s {
        for i in vl..VLENB * 8 {
            vset_mask(env, d.vd, i, true);
        }
    }
    fpu::check_fflags(env, &s);
}

/// `GEN_VEXT_FRED()`: `vd[0] = op(... op(vs1[0], vs2[a]) ..., vs2[b])` over the active
/// elements in order, with a 2*SEW `vd[0]` and `vs1[0]` when `wide`.
fn fred(env: &mut [u8], a: &[u64], wide: bool, op: fn(u64, u64, u32, &mut FloatStatus) -> u64) {
    let (d, _) = args(a);
    let sew = d.esz;
    let dl = if wide { sew + 1 } else { sew };
    let vl = vl(env);
    let mut s = fpu::status(env);
    let mut acc = vget(env, d.vs1, 0, dl);
    let start = vstart(env);
    if start >= vl {
        set_vstart(env, 0);
        return;
    }
    for i in start..vl {
        if !d.vm && !vmask(env, 0, i) {
            continue;
        }
        acc = op(acc, vget(env, d.vs2, i, sew), sew, &mut s);
    }
    vset(env, d.vd, 0, dl, acc);
    set_vstart(env, 0);
    // Set the tail elements to ones.
    set_1s(env, d.vd, d.vta, 1 << dl, VLENB);
    fpu::check_fflags(env, &s);
}

/// `GEN_VFMERGE_VF()`: `vd[i]` is the scalar, or `vs2[i]` when masked off. It rounds
/// nothing and raises no flags.
fn merge(env: &mut [u8], a: &[u64]) {
    let (d, scalar) = args(a);
    let sew = d.esz;
    let vl = vl(env);
    let start = vstart(env);
    if start >= vl {
        set_vstart(env, 0);
        return;
    }
    for i in start..vl {
        let v = if !d.vm && !vmask(env, 0, i) { vget(env, d.vs2, i, sew) } else { scalar };
        vset(env, d.vd, i, sew, v);
    }
    set_vstart(env, 0);
    let total = total_elems(env, &d, sew);
    set_1s(env, d.vd, d.vta, vl << sew, total << sew);
}

/// `vfclass.v`, which raises no flags: `GEN_VEXT_V()`.
fn class(env: &mut [u8], a: &[u64]) {
    let (d, _) = args(a);
    let sew = d.esz;
    for_each(env, &d, sew, |env, i| {
        let v = fclass(vget(env, d.vs2, i, sew), sew);
        vset(env, d.vd, i, sew, v);
    });
}

/// A helper that runs `$f(env, args)`: `vdef!(ID, "name", f)`.
macro_rules! vdef {
    ($id:ident, $name:literal, $f:expr) => {
        def!($id, $name, NO_RWG, Void, [Ptr, I32, I64], {
            fn h(e: &mut ruvm_jit_interp::HelperEnv<'_>, a: &[u64]) -> HR {
                let f: fn(&mut [u8], &[u64]) = $f;
                f(&mut *e.env, a);
                Ok(0)
            }
            h
        });
    };
}

// Single-width add, subtract, multiply, divide, min, max and sign injection.

vdef!(VFADD, "vfadd", |e, a| fop2(e, a, Shape::Single, fadd));
vdef!(VFSUB, "vfsub", |e, a| fop2(e, a, Shape::Single, fsub));
vdef!(VFRSUB, "vfrsub", |e, a| fop2(e, a, Shape::Single, |a, b, w, s| fsub(b, a, w, s)));
vdef!(VFMUL, "vfmul", |e, a| fop2(e, a, Shape::Single, fmul));
vdef!(VFDIV, "vfdiv", |e, a| fop2(e, a, Shape::Single, fdiv));
vdef!(VFRDIV, "vfrdiv", |e, a| fop2(e, a, Shape::Single, |a, b, w, s| fdiv(b, a, w, s)));
vdef!(VFMIN, "vfmin", |e, a| fop2(e, a, Shape::Single, fmin));
vdef!(VFMAX, "vfmax", |e, a| fop2(e, a, Shape::Single, fmax));
vdef!(VFSGNJ, "vfsgnj", |e, a| {
    fop2(e, a, Shape::Single, |a, b, w, _| (b & sign_bit(w)) | (a & !sign_bit(w)))
});
vdef!(VFSGNJN, "vfsgnjn", |e, a| {
    fop2(e, a, Shape::Single, |a, b, w, _| (!b & sign_bit(w)) | (a & !sign_bit(w)))
});
vdef!(VFSGNJX, "vfsgnjx", |e, a| fop2(e, a, Shape::Single, |a, b, w, _| a ^ (b & sign_bit(w))));

// Widening add, subtract and multiply.

vdef!(VFWADD, "vfwadd", |e, a| {
    fop2(e, a, Shape::Widen, |a, b, w, s| {
        let (a, b) = (widen(a, w, s), widen(b, w, s));
        fadd(a, b, w + 1, s)
    })
});
vdef!(VFWSUB, "vfwsub", |e, a| {
    fop2(e, a, Shape::Widen, |a, b, w, s| {
        let (a, b) = (widen(a, w, s), widen(b, w, s));
        fsub(a, b, w + 1, s)
    })
});
vdef!(VFWADD_W, "vfwadd_w", |e, a| {
    fop2(e, a, Shape::WidenW, |a, b, w, s| {
        let b = widen(b, w, s);
        fadd(a, b, w + 1, s)
    })
});
vdef!(VFWSUB_W, "vfwsub_w", |e, a| {
    fop2(e, a, Shape::WidenW, |a, b, w, s| {
        let b = widen(b, w, s);
        fsub(a, b, w + 1, s)
    })
});
vdef!(VFWMUL, "vfwmul", |e, a| {
    fop2(e, a, Shape::Widen, |a, b, w, s| {
        let (a, b) = (widen(a, w, s), widen(b, w, s));
        fmul(a, b, w + 1, s)
    })
});

// Fused multiply-add, `OP(vs2, s1, vd)`.

const NC: u32 = muladd::NEGATE_C;
const NP: u32 = muladd::NEGATE_PRODUCT;

vdef!(VFMACC, "vfmacc", |e, a| fop3(e, a, false, |a, b, d, w, s| fma(a, b, d, w, 0, s)));
vdef!(VFNMACC, "vfnmacc", |e, a| fop3(e, a, false, |a, b, d, w, s| fma(a, b, d, w, NC | NP, s)));
vdef!(VFMSAC, "vfmsac", |e, a| fop3(e, a, false, |a, b, d, w, s| fma(a, b, d, w, NC, s)));
vdef!(VFNMSAC, "vfnmsac", |e, a| fop3(e, a, false, |a, b, d, w, s| fma(a, b, d, w, NP, s)));
vdef!(VFMADD, "vfmadd", |e, a| fop3(e, a, false, |a, b, d, w, s| fma(d, b, a, w, 0, s)));
vdef!(VFNMADD, "vfnmadd", |e, a| fop3(e, a, false, |a, b, d, w, s| fma(d, b, a, w, NC | NP, s)));
vdef!(VFMSUB, "vfmsub", |e, a| fop3(e, a, false, |a, b, d, w, s| fma(d, b, a, w, NC, s)));
vdef!(VFNMSUB, "vfnmsub", |e, a| fop3(e, a, false, |a, b, d, w, s| fma(d, b, a, w, NP, s)));

/// The widening multiply-add: `vs2` and `s1` widened, then `fma(vs2, s1, vd, fl)`.
fn fwma(a: u64, b: u64, d: u64, w: u32, fl: u32, s: &mut FloatStatus) -> u64 {
    let (a, b) = (widen(a, w, s), widen(b, w, s));
    fma(a, b, d, w + 1, fl, s)
}

vdef!(VFWMACC, "vfwmacc", |e, a| fop3(e, a, true, |a, b, d, w, s| fwma(a, b, d, w, 0, s)));
vdef!(VFWNMACC, "vfwnmacc", |e, a| fop3(e, a, true, |a, b, d, w, s| fwma(a, b, d, w, NC | NP, s)));
vdef!(VFWMSAC, "vfwmsac", |e, a| fop3(e, a, true, |a, b, d, w, s| fwma(a, b, d, w, NC, s)));
vdef!(VFWNMSAC, "vfwnmsac", |e, a| fop3(e, a, true, |a, b, d, w, s| fwma(a, b, d, w, NP, s)));
vdef!(VFWMACCBF16, "vfwmaccbf16", |e, a| {
    fop3(e, a, true, |a, b, d, _, s| {
        let a = BFloat16::of(a).to_float32(s);
        let b = BFloat16::of(b).to_float32(s);
        a.muladd(b, Float32::of(d), 0, s).bits()
    })
});

// Square root, estimates and classify.

vdef!(VFSQRT, "vfsqrt", |e, a| fop1(e, a, Shape::Single, fsqrt));
vdef!(VFRSQRT7, "vfrsqrt7", |e, a| fop1(e, a, Shape::Single, rsqrt7));
vdef!(VFREC7, "vfrec7", |e, a| fop1(e, a, Shape::Single, rec7));
vdef!(VFCLASS, "vfclass", class);

// Conversions. `w` is SEW; the rounding mode is in the status.

vdef!(VFCVT_XU_F, "vfcvt_xu_f", |e, a| fop1(e, a, Shape::Single, |x, w, s| ftoi(
    x, w, w, false, s
)));
vdef!(VFCVT_X_F, "vfcvt_x_f", |e, a| fop1(e, a, Shape::Single, |x, w, s| ftoi(x, w, w, true, s)));
vdef!(VFCVT_F_XU, "vfcvt_f_xu", |e, a| fop1(e, a, Shape::Single, |x, w, s| itof(
    x, w, w, false, s
)));
vdef!(VFCVT_F_X, "vfcvt_f_x", |e, a| fop1(e, a, Shape::Single, |x, w, s| itof(x, w, w, true, s)));
vdef!(VFWCVT_XU_F, "vfwcvt_xu_f", |e, a| {
    fop1(e, a, Shape::Widen, |x, w, s| ftoi(x, w, w + 1, false, s))
});
vdef!(VFWCVT_X_F, "vfwcvt_x_f", |e, a| fop1(e, a, Shape::Widen, |x, w, s| ftoi(
    x,
    w,
    w + 1,
    true,
    s
)));
vdef!(VFWCVT_F_XU, "vfwcvt_f_xu", |e, a| {
    fop1(e, a, Shape::Widen, |x, w, s| itof(x, w, w + 1, false, s))
});
vdef!(VFWCVT_F_X, "vfwcvt_f_x", |e, a| fop1(e, a, Shape::Widen, |x, w, s| itof(
    x,
    w,
    w + 1,
    true,
    s
)));
vdef!(VFWCVT_F_F, "vfwcvt_f_f", |e, a| fop1(e, a, Shape::Widen, widen));
vdef!(VFNCVT_XU_F, "vfncvt_xu_f", |e, a| {
    fop1(e, a, Shape::Narrow, |x, w, s| ftoi(x, w + 1, w, false, s))
});
vdef!(VFNCVT_X_F, "vfncvt_x_f", |e, a| fop1(e, a, Shape::Narrow, |x, w, s| ftoi(
    x,
    w + 1,
    w,
    true,
    s
)));
vdef!(VFNCVT_F_XU, "vfncvt_f_xu", |e, a| {
    fop1(e, a, Shape::Narrow, |x, w, s| itof(x, w + 1, w, false, s))
});
vdef!(VFNCVT_F_X, "vfncvt_f_x", |e, a| fop1(e, a, Shape::Narrow, |x, w, s| itof(
    x,
    w + 1,
    w,
    true,
    s
)));
vdef!(VFNCVT_F_F, "vfncvt_f_f", |e, a| fop1(e, a, Shape::Narrow, narrow));
vdef!(VFNCVTBF16_F_F, "vfncvtbf16_f_f", |e, a| {
    fop1(e, a, Shape::Narrow, |x, _, s| Float32::of(x).to_bfloat16(s).bits())
});
vdef!(VFWCVTBF16_F_F, "vfwcvtbf16_f_f", |e, a| {
    fop1(e, a, Shape::Widen, |x, _, s| BFloat16::of(x).to_float32(s).bits())
});

// Compares, `OP(vs2, s1)`.

vdef!(VMFEQ, "vmfeq", |e, a| fcmp(
    e,
    a,
    |a, b, w, s| per_sew!(w, T => T::of(a).eq_quiet(T::of(b), s))
));
vdef!(VMFNE, "vmfne", |e, a| {
    fcmp(
        e,
        a,
        |a, b, w, s| per_sew!(w, T => T::of(a).compare_quiet(T::of(b), s) != FloatRelation::Equal),
    )
});
vdef!(VMFLT, "vmflt", |e, a| fcmp(e, a, |a, b, w, s| per_sew!(w, T => T::of(a).lt(T::of(b), s))));
vdef!(VMFLE, "vmfle", |e, a| fcmp(e, a, |a, b, w, s| per_sew!(w, T => T::of(a).le(T::of(b), s))));
vdef!(VMFGT, "vmfgt", |e, a| {
    fcmp(
        e,
        a,
        |a, b, w, s| per_sew!(w, T => T::of(a).compare(T::of(b), s) == FloatRelation::Greater),
    )
});
vdef!(VMFGE, "vmfge", |e, a| {
    fcmp(e, a, |a, b, w, s| {
        per_sew!(w, T => matches!(
            T::of(a).compare(T::of(b), s),
            FloatRelation::Greater | FloatRelation::Equal
        ))
    })
});

// Merge and move.

vdef!(VFMERGE, "vfmerge", merge);

// Reductions, `OP(acc, vs2[i])`. The unordered sum adds in order, as QEMU does.

vdef!(VFREDUSUM, "vfredusum", |e, a| fred(e, a, false, fadd));
vdef!(VFREDOSUM, "vfredosum", |e, a| fred(e, a, false, fadd));
vdef!(VFREDMAX, "vfredmax", |e, a| fred(e, a, false, fmax));
vdef!(VFREDMIN, "vfredmin", |e, a| fred(e, a, false, fmin));

/// `fwadd16()` and `fwadd32()`: `acc + widen(b)`.
fn fwadd(acc: u64, b: u64, w: u32, s: &mut FloatStatus) -> u64 {
    let b = widen(b, w, s);
    fadd(acc, b, w + 1, s)
}

vdef!(VFWREDUSUM, "vfwredusum", |e, a| fred(e, a, true, fwadd));
vdef!(VFWREDOSUM, "vfwredosum", |e, a| fred(e, a, true, fwadd));

/// The vector floating point helpers.
pub(crate) const ALL: &[Def] = &[
    VFADD,
    VFSUB,
    VFRSUB,
    VFMUL,
    VFDIV,
    VFRDIV,
    VFMIN,
    VFMAX,
    VFSGNJ,
    VFSGNJN,
    VFSGNJX,
    VFWADD,
    VFWSUB,
    VFWADD_W,
    VFWSUB_W,
    VFWMUL,
    VFMACC,
    VFNMACC,
    VFMSAC,
    VFNMSAC,
    VFMADD,
    VFNMADD,
    VFMSUB,
    VFNMSUB,
    VFWMACC,
    VFWNMACC,
    VFWMSAC,
    VFWNMSAC,
    VFWMACCBF16,
    VFSQRT,
    VFRSQRT7,
    VFREC7,
    VFCLASS,
    VFCVT_XU_F,
    VFCVT_X_F,
    VFCVT_F_XU,
    VFCVT_F_X,
    VFWCVT_XU_F,
    VFWCVT_X_F,
    VFWCVT_F_XU,
    VFWCVT_F_X,
    VFWCVT_F_F,
    VFNCVT_XU_F,
    VFNCVT_X_F,
    VFNCVT_F_XU,
    VFNCVT_F_X,
    VFNCVT_F_F,
    VFNCVTBF16_F_F,
    VFWCVTBF16_F_F,
    VMFEQ,
    VMFNE,
    VMFLT,
    VMFLE,
    VMFGT,
    VMFGE,
    VFMERGE,
    VFREDUSUM,
    VFREDOSUM,
    VFREDMAX,
    VFREDMIN,
    VFWREDUSUM,
    VFWREDOSUM,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{
        ENV_SIZE, FFLAGS, FP_ROUND, MSTATUS, MSTATUS_FS, RISCV_FRM_DYN, RISCV_FRM_RDN,
        RISCV_FRM_RNE, RISCV_FRM_ROD, RISCV_FRM_RTZ, RISCV_FRM_RUP, VL, VSTART, VTYPE,
    };
    use crate::tcg::vector::voff;
    use crate::tcg::{ld64, st64};
    use ruvm_jit_interp::{HelperEnv, NoMemory};

    const NX: u64 = 0x01;
    const OF: u64 = 0x04;
    const DZ: u64 = 0x08;
    const NV: u64 = 0x10;

    const S_ONE: u64 = 0x3f80_0000;
    const S_QNAN: u64 = 0x7fc0_0000;
    const H_ONE: u64 = 0x3c00;
    const D_ONE: u64 = 0x3ff0_0000_0000_0000;

    /// An `env` with `vtype` SEW `1 << sew` bytes, LMUL 1, `vl` and rounding mode `rm`.
    fn env(sew: u32, vl: u64, rm: u64) -> Vec<u8> {
        let mut env = vec![0u8; ENV_SIZE];
        st64(&mut env, VTYPE, u64::from(sew) << 3);
        st64(&mut env, VL, vl);
        st64(&mut env, FP_ROUND, rm);
        env
    }

    /// Call helper `h` with descriptor `d` and the scalar `f`.
    fn call(h: &Def, env: &mut [u8], d: Desc, f: u64) {
        let a = [0, u64::from(d.encode()), f];
        let mut mem = NoMemory;
        let mut he = HelperEnv { env, mem: &mut mem };
        (h.f)(&mut he, &a).expect("no unwind");
    }

    fn fill(env: &mut [u8], reg: u32, log2: u32, vals: &[u64]) {
        for (i, &v) in vals.iter().enumerate() {
            vset(env, reg, i, log2, v);
        }
    }

    fn get(env: &[u8], reg: u32, log2: u32, n: usize) -> Vec<u64> {
        (0..n).map(|i| vget(env, reg, i, log2)).collect()
    }

    /// The descriptor of `vd` v1, `vs1` v2 and `vs2` v3, unmasked, SEW `1 << esz` bytes.
    fn desc(esz: u32) -> Desc {
        Desc { vm: true, vd: 1, vs1: 2, vs2: 3, esz, ..Desc::default() }
    }

    fn fflags(env: &[u8]) -> u64 {
        ld64(env, FFLAGS)
    }

    fn fs_dirty(env: &[u8]) -> bool {
        ld64(env, MSTATUS) & MSTATUS_FS == MSTATUS_FS
    }

    #[test]
    fn vfadd_rounding_and_flags() {
        // 1 + 2^-30 is inexact in single precision.
        let tiny = 0x3080_0000;
        let mut e = env(2, 2, RISCV_FRM_RNE);
        fill(&mut e, 3, 2, &[S_ONE, S_ONE]);
        fill(&mut e, 2, 2, &[tiny, 0]);
        call(&VFADD, &mut e, desc(2), 0);
        assert_eq!(get(&e, 1, 2, 2), [S_ONE, S_ONE]);
        assert_eq!(fflags(&e), NX);
        assert!(fs_dirty(&e));
        // Rounding up gives the next float; the flag is already set, so FS stays clean.
        let mut e = env(2, 2, RISCV_FRM_RUP);
        st64(&mut e, FFLAGS, NX);
        fill(&mut e, 3, 2, &[S_ONE, S_ONE]);
        fill(&mut e, 2, 2, &[tiny, 0]);
        call(&VFADD, &mut e, desc(2), 0);
        assert_eq!(get(&e, 1, 2, 2), [S_ONE + 1, S_ONE]);
        assert_eq!(fflags(&e), NX);
        assert!(!fs_dirty(&e));
        // The .vf form with the scalar, and vfrsub swaps the operands.
        let mut e = env(2, 2, RISCV_FRM_RNE);
        fill(&mut e, 3, 2, &[S_ONE, 0x4000_0000]);
        call(&VFSUB, &mut e, Desc { scalar: true, ..desc(2) }, 0x4040_0000);
        assert_eq!(get(&e, 1, 2, 2), [0xc000_0000, 0xbf80_0000]);
        call(&VFRSUB, &mut e, Desc { scalar: true, ..desc(2) }, 0x4040_0000);
        assert_eq!(get(&e, 1, 2, 2), [0x4000_0000, S_ONE]);
        assert_eq!(fflags(&e), 0);
        // Invalid: inf - inf gives the canonical NaN.
        fill(&mut e, 3, 2, &[0x7f80_0000]);
        fill(&mut e, 2, 2, &[0x7f80_0000]);
        call(&VFSUB, &mut e, desc(2), 0);
        assert_eq!(vget(&e, 1, 0, 2), S_QNAN);
        assert_eq!(fflags(&e), NV);
        // Masked off elements are kept, or set to ones with vma; the tail with vta.
        let mut e = env(1, 3, RISCV_FRM_RNE);
        fill(&mut e, 1, 1, &[7, 7, 7, 7]);
        fill(&mut e, 3, 1, &[H_ONE, H_ONE, H_ONE]);
        fill(&mut e, 2, 1, &[H_ONE, H_ONE, H_ONE]);
        e[voff(0, 0)] = 0b101;
        call(&VFADD, &mut e, Desc { vm: false, vta: true, ..desc(1) }, 0);
        assert_eq!(get(&e, 1, 1, 4), [0x4000, 7, 0x4000, 0xffff]);
        call(&VFADD, &mut e, Desc { vm: false, vma: true, ..desc(1) }, 0);
        assert_eq!(get(&e, 1, 1, 3), [0x4000, 0xffff, 0x4000]);
    }

    #[test]
    fn widening() {
        // vfwadd.vv: 1.0h + 2.0h = 3.0f; vfwmul.vf by 2.0h.
        let mut e = env(1, 2, RISCV_FRM_RNE);
        fill(&mut e, 3, 1, &[H_ONE, 0x4000]);
        fill(&mut e, 2, 1, &[0x4000, 0xbc00]);
        call(&VFWADD, &mut e, desc(1), 0);
        assert_eq!(get(&e, 1, 2, 2), [0x4040_0000, S_ONE]);
        call(&VFWMUL, &mut e, Desc { scalar: true, ..desc(1) }, 0x4000);
        assert_eq!(get(&e, 1, 2, 2), [0x4000_0000, 0x4080_0000]);
        // vfwsub.wf: a single precision vs2 minus a widened 1.0h.
        fill(&mut e, 3, 2, &[0x4040_0000, S_ONE]);
        call(&VFWSUB_W, &mut e, Desc { scalar: true, ..desc(1) }, H_ONE);
        assert_eq!(get(&e, 1, 2, 2), [0x4000_0000, 0]);
        // vfwmacc.vv: vd += vs2 * vs1, single to double.
        let mut e = env(2, 1, RISCV_FRM_RNE);
        fill(&mut e, 3, 2, &[0x4000_0000]);
        fill(&mut e, 2, 2, &[0x4040_0000]);
        fill(&mut e, 1, 3, &[D_ONE]);
        call(&VFWMACC, &mut e, desc(2), 0);
        assert_eq!(vget(&e, 1, 0, 3), 7f64.to_bits());
        call(&VFWNMSAC, &mut e, desc(2), 0);
        assert_eq!(vget(&e, 1, 0, 3), 1f64.to_bits());
        assert_eq!(fflags(&e), 0);
        // vfwcvt.f.f.v and vfwcvt.f.x.v from 8 bits.
        let mut e = env(1, 1, RISCV_FRM_RNE);
        fill(&mut e, 3, 1, &[0x3555]);
        call(&VFWCVT_F_F, &mut e, desc(1), 0);
        assert_eq!(vget(&e, 1, 0, 2), u64::from((0.333_251_95f32).to_bits()));
        let mut e = env(0, 2, RISCV_FRM_RNE);
        fill(&mut e, 3, 0, &[0xff, 100]);
        call(&VFWCVT_F_X, &mut e, desc(0), 0);
        assert_eq!(get(&e, 1, 1, 2), [0xbc00, 0x5640]);
        call(&VFWCVT_F_XU, &mut e, desc(0), 0);
        assert_eq!(get(&e, 1, 1, 2), [0x5bf8, 0x5640]);
    }

    #[test]
    fn single_width_fma() {
        let mut e = env(3, 1, RISCV_FRM_RNE);
        let (two, three, d) = (2f64.to_bits(), 3f64.to_bits(), 10f64.to_bits());
        let run = |e: &mut Vec<u8>, h: &Def| {
            fill(e, 3, 3, &[two]);
            fill(e, 2, 3, &[three]);
            fill(e, 1, 3, &[d]);
            call(h, e, desc(3), 0);
            f64::from_bits(vget(e, 1, 0, 3))
        };
        assert_eq!(run(&mut e, &VFMACC), 16.0);
        assert_eq!(run(&mut e, &VFNMACC), -16.0);
        assert_eq!(run(&mut e, &VFMSAC), -4.0);
        assert_eq!(run(&mut e, &VFNMSAC), 4.0);
        // madd: vd * vs1 + vs2.
        assert_eq!(run(&mut e, &VFMADD), 32.0);
        assert_eq!(run(&mut e, &VFNMADD), -32.0);
        assert_eq!(run(&mut e, &VFMSUB), 28.0);
        assert_eq!(run(&mut e, &VFNMSUB), -28.0);
    }

    #[test]
    fn narrowing_rod() {
        // 1 + 2^-30 in double precision.
        let x = 0x3ff0_0000_0040_0000;
        let mut e = env(2, 2, RISCV_FRM_RNE);
        fill(&mut e, 3, 3, &[x, D_ONE]);
        call(&VFNCVT_F_F, &mut e, desc(2), 0);
        assert_eq!(get(&e, 1, 2, 2), [S_ONE, S_ONE]);
        assert_eq!(fflags(&e), NX);
        // Round to odd sets the low bit of an inexact result.
        let mut e = env(2, 2, RISCV_FRM_ROD);
        fill(&mut e, 3, 3, &[x, D_ONE]);
        call(&VFNCVT_F_F, &mut e, desc(2), 0);
        assert_eq!(get(&e, 1, 2, 2), [S_ONE | 1, S_ONE]);
        assert_eq!(fflags(&e), NX);
        // vfncvt.x.f.w to 8 bits saturates and raises invalid; rtz truncates.
        let mut e = env(0, 3, RISCV_FRM_RTZ);
        fill(&mut e, 3, 1, &[0x5bf8, 0x3e00, 0xc100]);
        call(&VFNCVT_X_F, &mut e, desc(0), 0);
        assert_eq!(get(&e, 1, 0, 3), [0x7f, 1, 0xfe]);
        assert_eq!(fflags(&e), NV | NX);
        let mut e = env(0, 3, RISCV_FRM_RDN);
        fill(&mut e, 3, 1, &[0x5bf8, 0x3e00, 0xc100]);
        call(&VFNCVT_XU_F, &mut e, desc(0), 0);
        assert_eq!(get(&e, 1, 0, 3), [0xff, 1, 0]);
        // vfncvt.f.xu.w: 2^24 + 1 is inexact in single precision.
        let mut e = env(2, 1, RISCV_FRM_RUP);
        fill(&mut e, 3, 3, &[(1 << 24) + 1]);
        call(&VFNCVT_F_XU, &mut e, desc(2), 0);
        assert_eq!(vget(&e, 1, 0, 2), 0x4b80_0001);
        assert_eq!(fflags(&e), NX);
    }

    #[test]
    fn single_width_conversions() {
        let mut e = env(2, 3, RISCV_FRM_RNE);
        fill(&mut e, 3, 2, &[0x3fc0_0000, 0xc020_0000, S_QNAN]);
        call(&VFCVT_X_F, &mut e, desc(2), 0);
        // 1.5 and -2.5 round to even; NaN is the largest integer.
        assert_eq!(get(&e, 1, 2, 3), [2, 0xffff_fffe, 0x7fff_ffff]);
        assert_eq!(fflags(&e), NV | NX);
        let mut e = env(2, 1, RISCV_FRM_RNE);
        fill(&mut e, 3, 2, &[0xffff_fffe]);
        call(&VFCVT_F_X, &mut e, desc(2), 0);
        assert_eq!(vget(&e, 1, 0, 2), 0xc000_0000);
        call(&VFCVT_F_XU, &mut e, desc(2), 0);
        assert_eq!(vget(&e, 1, 0, 2), 0x4f80_0000);
    }

    #[test]
    fn rec7_rsqrt7_tables() {
        let mut s = fpu::status_rm(RISCV_FRM_RNE);
        // Single precision.
        assert_eq!(rsqrt7(S_ONE, 2, &mut s), 0x3f7f_0000);
        assert_eq!(rsqrt7(0x4000_0000, 2, &mut s), 0x3f34_0000);
        assert_eq!(rsqrt7(0x4080_0000, 2, &mut s), 0x3eff_0000);
        assert_eq!(rec7(S_ONE, 2, &mut s), 0x3f7f_0000);
        assert_eq!(rec7(0x4000_0000, 2, &mut s), 0x3eff_0000);
        assert_eq!(rec7(0xbf80_0000, 2, &mut s), 0xbf7f_0000);
        // The largest finite float has a subnormal estimate.
        assert_eq!(rec7(0x7f7f_ffff, 2, &mut s), 0x0020_0000);
        // A subnormal input normalizes before the lookup.
        assert_eq!(rsqrt7(0x0000_0001, 2, &mut s), 0x64b4_0000);
        assert_eq!(s.flags(), 0);
        // Half and double precision.
        assert_eq!(rec7(H_ONE, 1, &mut s), 0x3bf8);
        assert_eq!(rsqrt7(H_ONE, 1, &mut s), 0x3bf8);
        assert_eq!(rec7(D_ONE, 3, &mut s), 0x3fef_e000_0000_0000);
        assert_eq!(rsqrt7(D_ONE, 3, &mut s), 0x3fef_e000_0000_0000);
        assert_eq!(s.flags(), 0);
        // The special cases.
        let flags = |f: &dyn Fn(&mut FloatStatus) -> u64| {
            let mut s = fpu::status_rm(RISCV_FRM_RNE);
            let r = f(&mut s);
            (r, fpu::riscv_flags(s.flags()))
        };
        assert_eq!(flags(&|s| rsqrt7(0xbf80_0000, 2, s)), (S_QNAN, NV));
        assert_eq!(flags(&|s| rsqrt7(0xff80_0000, 2, s)), (S_QNAN, NV));
        assert_eq!(flags(&|s| rsqrt7(0x7f80_0001, 2, s)), (S_QNAN, NV));
        assert_eq!(flags(&|s| rsqrt7(0xffc0_0000, 2, s)), (S_QNAN, 0));
        assert_eq!(flags(&|s| rsqrt7(0x8000_0000, 2, s)), (0xff80_0000, DZ));
        assert_eq!(flags(&|s| rsqrt7(0x7f80_0000, 2, s)), (0, 0));
        assert_eq!(flags(&|s| rec7(0xff80_0000, 2, s)), (0x8000_0000, 0));
        assert_eq!(flags(&|s| rec7(0, 2, s)), (0x7f80_0000, DZ));
        assert_eq!(flags(&|s| rec7(0x7f80_0001, 2, s)), (S_QNAN, NV));
        // The smallest subnormal overflows to inf, or to the largest finite value when
        // rounding towards zero.
        assert_eq!(flags(&|s| rec7(1, 2, s)), (0x7f80_0000, OF | NX));
        let mut s = fpu::status_rm(RISCV_FRM_RTZ);
        assert_eq!(rec7(1, 2, &mut s), 0x7f7f_ffff);
        let mut s = fpu::status_rm(RISCV_FRM_RDN);
        assert_eq!(rec7(0x8000_0001, 2, &mut s), 0xff80_0000);
        assert_eq!(rec7(1, 2, &mut s), 0x7f7f_ffff);
        // Through the helper.
        let mut e = env(2, 2, RISCV_FRM_RNE);
        fill(&mut e, 3, 2, &[S_ONE, 0]);
        call(&VFREC7, &mut e, desc(2), 0);
        assert_eq!(get(&e, 1, 2, 2), [0x3f7f_0000, 0x7f80_0000]);
        assert_eq!(fflags(&e), DZ);
    }

    #[test]
    fn reductions_ordered_and_unordered() {
        // 1 + 2^24 - 2^24 in order is 0 in single precision: both sums add in order.
        let big = 0x4b80_0000;
        for h in [&VFREDUSUM, &VFREDOSUM] {
            let mut e = env(2, 3, RISCV_FRM_RNE);
            fill(&mut e, 2, 2, &[S_ONE]);
            fill(&mut e, 3, 2, &[big, big | 1 << 31, 0x4000_0000]);
            fill(&mut e, 1, 2, &[0, 9, 9, 9]);
            e[voff(0, 0)] = 0b011;
            call(h, &mut e, Desc { vm: false, vta: true, ..desc(2) }, 0);
            // Element 2 is masked off; the tail of vd is set to ones.
            assert_eq!(get(&e, 1, 2, 4), [0, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff]);
            assert_eq!(fflags(&e), NX);
            assert_eq!(ld64(&e, VSTART), 0);
        }
        let mut e = env(2, 3, RISCV_FRM_RNE);
        fill(&mut e, 2, 2, &[S_ONE]);
        fill(&mut e, 3, 2, &[0xc000_0000, S_QNAN, 0x4000_0000]);
        call(&VFREDMAX, &mut e, desc(2), 0);
        assert_eq!(vget(&e, 1, 0, 2), 0x4000_0000);
        call(&VFREDMIN, &mut e, desc(2), 0);
        assert_eq!(vget(&e, 1, 0, 2), 0xc000_0000);
        assert_eq!(fflags(&e), 0);
        // Widening: half precision elements into a single precision sum.
        let mut e = env(1, 3, RISCV_FRM_RNE);
        fill(&mut e, 2, 2, &[0x4120_0000]);
        fill(&mut e, 3, 1, &[H_ONE, 0x4000, 0x4200]);
        call(&VFWREDOSUM, &mut e, desc(1), 0);
        assert_eq!(vget(&e, 1, 0, 2), 0x4180_0000);
        // vl 0 leaves vd alone.
        st64(&mut e, VL, 0);
        call(&VFWREDUSUM, &mut e, desc(1), 0);
        assert_eq!(vget(&e, 1, 0, 2), 0x4180_0000);
    }

    #[test]
    fn classify() {
        let mut e = env(1, 10, RISCV_FRM_DYN);
        let vals = [0xfc00, 0xbc00, 0x8001, 0x8000, 0, 1, H_ONE, 0x7c00, 0x7c01, 0x7e00];
        fill(&mut e, 3, 1, &vals);
        call(&VFCLASS, &mut e, desc(1), 0);
        let want: Vec<u64> = (0..10).map(|i| 1 << i).collect();
        assert_eq!(get(&e, 1, 1, 10), want);
        let mut e = env(3, 3, RISCV_FRM_DYN);
        fill(&mut e, 3, 3, &[0xfff0_0000_0000_0000, 0x7ff4_0000_0000_0000, 1]);
        call(&VFCLASS, &mut e, desc(3), 0);
        assert_eq!(get(&e, 1, 3, 3), [1, 1 << 8, 1 << 5]);
        assert_eq!(fflags(&e), 0);
    }

    #[test]
    fn compares() {
        let mut e = env(2, 4, RISCV_FRM_RNE);
        fill(&mut e, 3, 2, &[S_ONE, S_QNAN, 0x4000_0000, S_ONE]);
        fill(&mut e, 2, 2, &[S_ONE, S_ONE, S_ONE, 0x4000_0000]);
        let d = Desc { vta_all_1s: true, ..desc(2) };
        call(&VMFEQ, &mut e, d, 0);
        assert_eq!(e[voff(1, 0)], 0b1111_0001);
        assert_eq!(fflags(&e), 0, "quiet compare");
        call(&VMFNE, &mut e, d, 0);
        assert_eq!(e[voff(1, 0)], 0b1111_1110);
        call(&VMFLT, &mut e, d, 0);
        assert_eq!(e[voff(1, 0)], 0b1111_1000);
        assert_eq!(fflags(&e), NV, "signalling compare of a quiet NaN");
        call(&VMFLE, &mut e, d, 0);
        assert_eq!(e[voff(1, 0)], 0b1111_1001);
        call(&VMFGT, &mut e, Desc { scalar: true, ..d }, S_ONE);
        assert_eq!(e[voff(1, 0)], 0b1111_0100);
        call(&VMFGE, &mut e, Desc { scalar: true, ..d }, S_ONE);
        assert_eq!(e[voff(1, 0)], 0b1111_1101);
        // Masked off bits are kept, or set with vma; without vta_all_1s the tail is kept.
        e[voff(0, 0)] = 0b0001;
        e[voff(1, 0)] = 0;
        call(&VMFEQ, &mut e, Desc { vm: false, vta_all_1s: false, ..d }, 0);
        assert_eq!(e[voff(1, 0)], 0b0001);
        call(&VMFEQ, &mut e, Desc { vm: false, vma: true, vta_all_1s: false, ..d }, 0);
        assert_eq!(e[voff(1, 0)], 0b1111);
    }

    #[test]
    fn merge_and_move() {
        let mut e = env(1, 3, RISCV_FRM_RNE);
        fill(&mut e, 3, 1, &[1, 2, 3]);
        fill(&mut e, 1, 1, &[0, 0, 0, 5]);
        e[voff(0, 0)] = 0b010;
        call(&VFMERGE, &mut e, Desc { vm: false, ..desc(1) }, H_ONE);
        assert_eq!(get(&e, 1, 1, 4), [1, H_ONE, 3, 5]);
        // vfmv.v.f is the merge with vm set.
        call(&VFMERGE, &mut e, Desc { vta: true, ..desc(1) }, 0x4000);
        assert_eq!(get(&e, 1, 1, 4), [0x4000, 0x4000, 0x4000, 0xffff]);
        // From vstart.
        st64(&mut e, VSTART, 2);
        call(&VFMERGE, &mut e, desc(1), H_ONE);
        assert_eq!(get(&e, 1, 1, 3), [0x4000, 0x4000, H_ONE]);
        assert_eq!(ld64(&e, VSTART), 0);
    }

    #[test]
    fn sign_injection_and_min_max() {
        let mut e = env(2, 2, RISCV_FRM_RNE);
        fill(&mut e, 3, 2, &[S_ONE, 0xc000_0000]);
        fill(&mut e, 2, 2, &[0x8000_0000, 0x8000_0000]);
        call(&VFSGNJ, &mut e, desc(2), 0);
        assert_eq!(get(&e, 1, 2, 2), [0xbf80_0000, 0xc000_0000]);
        call(&VFSGNJN, &mut e, desc(2), 0);
        assert_eq!(get(&e, 1, 2, 2), [S_ONE, 0x4000_0000]);
        call(&VFSGNJX, &mut e, desc(2), 0);
        assert_eq!(get(&e, 1, 2, 2), [0xbf80_0000, 0x4000_0000]);
        // minimumNumber: a quiet NaN loses, -0 is below +0.
        fill(&mut e, 3, 2, &[S_QNAN, 0]);
        call(&VFMIN, &mut e, desc(2), 0);
        assert_eq!(get(&e, 1, 2, 2), [0x8000_0000, 0x8000_0000]);
        call(&VFMAX, &mut e, desc(2), 0);
        assert_eq!(get(&e, 1, 2, 2), [0x8000_0000, 0]);
        assert_eq!(fflags(&e), 0);
    }

    #[test]
    fn bf16() {
        // vfwcvtbf16.f.f.v is exact.
        let mut e = env(1, 2, RISCV_FRM_RNE);
        fill(&mut e, 3, 1, &[0x3f80, 0xc049]);
        call(&VFWCVTBF16_F_F, &mut e, desc(1), 0);
        assert_eq!(get(&e, 1, 2, 2), [S_ONE, 0xc049_0000]);
        assert_eq!(fflags(&e), 0);
        // vfncvtbf16.f.f.w rounds: ties to even, then up.
        let mut e = env(1, 3, RISCV_FRM_RNE);
        fill(&mut e, 3, 2, &[0x3f80_8000, 0x3f81_8000, 0x7f80_0001]);
        call(&VFNCVTBF16_F_F, &mut e, desc(1), 0);
        assert_eq!(get(&e, 1, 1, 3), [0x3f80, 0x3f82, 0x7fc0]);
        assert_eq!(fflags(&e), NX | NV);
        let mut e = env(1, 1, RISCV_FRM_RUP);
        fill(&mut e, 3, 2, &[0x3f80_0001]);
        call(&VFNCVTBF16_F_F, &mut e, desc(1), 0);
        assert_eq!(vget(&e, 1, 0, 1), 0x3f81);
        // vfwmaccbf16: 1.0 + 2.0 * 3.0 in single precision, .vv and .vf.
        let mut e = env(1, 2, RISCV_FRM_RNE);
        fill(&mut e, 3, 1, &[0x4000, 0x4000]);
        fill(&mut e, 2, 1, &[0x4040, 0x3f80]);
        fill(&mut e, 1, 2, &[S_ONE, S_ONE]);
        call(&VFWMACCBF16, &mut e, desc(1), 0);
        assert_eq!(get(&e, 1, 2, 2), [0x40e0_0000, 0x4040_0000]);
        call(&VFWMACCBF16, &mut e, Desc { scalar: true, ..desc(1) }, 0xbf80);
        assert_eq!(get(&e, 1, 2, 2), [0x40a0_0000, S_ONE]);
        assert_eq!(fflags(&e), 0);
    }

    #[test]
    fn helper_names_are_unique() {
        let mut names: Vec<_> = ALL.iter().map(|d| d.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), ALL.len());
        assert!(ALL.iter().all(|d| d.args == [Ptr, I32, I64]));
    }
}
