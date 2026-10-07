// SPDX-License-Identifier: GPL-2.0-or-later

//! The floating point helpers, a port of QEMU's `target/riscv/tcg/fpu_helper.c` for the F,
//! D, Zfh and Zfa instructions and the conversions of Zfhmin and Zfbfmin, with
//! `fclass_h()`, `fclass_s()` and `fclass_d()` from `vector_helper.c`.
//!
//! QEMU keeps a `float_status` in `env`. Here every helper builds one from `env`: the
//! rounding mode comes from `fp_round`, which `set_rounding_mode` leaves there as a RISC-V
//! `rm` value, and the rest is what `riscv_cpu_reset_hold()` sets: default NaN mode with
//! the pattern of `0x7fc00000`, tininess after rounding and no flush to zero. The flags an
//! operation raises are mapped to the RISC-V order and or'ed into `fflags`, which is what
//! `riscv_cpu_get_fflags()` would read back.
//!
//! The model is QEMU's `rv64` with `priv_spec` 1.12, so `fmin` and `fmax` are IEEE 754-2019
//! minimumNumber and maximumNumber, and Zfinx and Zdinx are off, so single and half
//! precision values are always NaN boxed. The half precision helpers are those of Zfh,
//! Zfhmin and Zfbfmin.
//!
//! Deliberate differences from QEMU:
//!
//! - QEMU's `riscv_cpu_check_fflags()` sets `mstatus.FS` when a compare or a conversion to
//!   an integer raises a softfloat flag that was not set yet, including the internal detail
//!   flags (`float_flag_invalid_cvti`, `float_flag_input_denormal_used` and so on) that
//!   never show in `fflags`. Those detail flags are not kept here, so `mstatus.FS` is set
//!   when one of the five `fflags` bits is new. The values of `fflags` are the same.

use ruvm_jit::Ra;
use ruvm_jit_core::HelperType::{I32, I64, Ptr, Void};
use ruvm_jit_core::types::call_flags::{NO_RWG, NO_RWG_SE, NO_WG};
use ruvm_jit_interp::{HelperEnv, Unwind};
use ruvm_softfloat::{BFloat16, Float16, Float32, Float64, FloatStatus, RoundMode, flags, muladd};

use super::helpers::{Def, run};
use super::{ld64, st64};
use crate::cpu::{
    EXCP_ILLEGAL_INST, FFLAGS, FFLAGS_MASK, FP_ROUND, FRM, MSTATUS, MSTATUS_FS, RISCV_FRM_DYN,
    RISCV_FRM_RDN, RISCV_FRM_RMM, RISCV_FRM_ROD, RISCV_FRM_RTZ, RISCV_FRM_RUP,
};

type HR = Result<u128, Unwind>;

macro_rules! def {
    ($id:ident, $name:literal, $flags:expr, $ret:expr, [$($a:expr),*], $f:expr) => {
        pub(crate) const $id: Def =
            Def { name: $name, flags: $flags, ret: $ret, args: &[$($a),*], f: $f };
    };
}

/// A helper that takes `env` and `$n` 64-bit arguments and returns 64 bits: `$f` applied to
/// the arguments with the float status of `env`, its flags accrued by `$acc`.
macro_rules! fp_def {
    ($id:ident, $name:literal, $flags:expr, $acc:ident, $f:path, 1) => {
        def!($id, $name, $flags, I64, [Ptr, I64], {
            fn h(e: &mut HelperEnv<'_>, a: &[u64]) -> HR {
                $acc(e, |s| $f(a[1], s))
            }
            h
        });
    };
    ($id:ident, $name:literal, $flags:expr, $acc:ident, $f:path, 2) => {
        def!($id, $name, $flags, I64, [Ptr, I64, I64], {
            fn h(e: &mut HelperEnv<'_>, a: &[u64]) -> HR {
                $acc(e, |s| $f(a[1], a[2], s))
            }
            h
        });
    };
    ($id:ident, $name:literal, $flags:expr, $acc:ident, $f:path, 3) => {
        def!($id, $name, $flags, I64, [Ptr, I64, I64, I64], {
            fn h(e: &mut HelperEnv<'_>, a: &[u64]) -> HR {
                $acc(e, |s| $f(a[1], a[2], a[3], s))
            }
            h
        });
    };
}

// The status and the flags.

/// `FPEXC_NX`.
const FPEXC_NX: u64 = 0x01;
/// `FPEXC_UF`.
const FPEXC_UF: u64 = 0x02;
/// `FPEXC_OF`.
const FPEXC_OF: u64 = 0x04;
/// `FPEXC_DZ`.
const FPEXC_DZ: u64 = 0x08;
/// `FPEXC_NV`.
const FPEXC_NV: u64 = 0x10;

/// The softfloat rounding mode of RISC-V rounding mode `rm`, as `set_rounding_mode` has
/// checked it.
fn round_mode(rm: u64) -> RoundMode {
    match rm {
        RISCV_FRM_RTZ => RoundMode::ToZero,
        RISCV_FRM_RDN => RoundMode::Down,
        RISCV_FRM_RUP => RoundMode::Up,
        RISCV_FRM_RMM => RoundMode::TiesAway,
        RISCV_FRM_ROD => RoundMode::ToOdd,
        _ => RoundMode::NearestEven,
    }
}

/// The `fp_status` of RISC-V with rounding mode `rm` and no flags.
pub(super) fn status_rm(rm: u64) -> FloatStatus {
    FloatStatus {
        rounding_mode: round_mode(rm),
        default_nan_mode: true,
        default_nan_pattern: 0b0100_0000,
        ..FloatStatus::default()
    }
}

/// The `fp_status` of `env` with no flags.
pub(super) fn status(env: &[u8]) -> FloatStatus {
    status_rm(ld64(env, FP_ROUND))
}

/// `riscv_cpu_get_fflags()`: softfloat flags `soft` in the RISC-V order.
pub(super) fn riscv_flags(soft: u16) -> u64 {
    let mut hard = 0;
    if soft & flags::INEXACT != 0 {
        hard |= FPEXC_NX;
    }
    if soft & flags::UNDERFLOW != 0 {
        hard |= FPEXC_UF;
    }
    if soft & flags::OVERFLOW != 0 {
        hard |= FPEXC_OF;
    }
    if soft & flags::DIVBYZERO != 0 {
        hard |= FPEXC_DZ;
    }
    if soft & flags::INVALID != 0 {
        hard |= FPEXC_NV;
    }
    hard
}

/// Run `f` with the float status of `env` and accrue the flags it raises into `fflags`.
fn accrue(e: &mut HelperEnv<'_>, f: impl FnOnce(&mut FloatStatus) -> u64) -> HR {
    let mut s = status(e.env);
    let r = f(&mut s);
    let new = riscv_flags(s.flags());
    if new != 0 {
        let old = ld64(e.env, FFLAGS);
        st64(e.env, FFLAGS, old | new);
    }
    Ok(u128::from(r))
}

/// Accrue the flags of `s` into `fflags` with `riscv_cpu_check_fflags()`: `mstatus.FS`
/// becomes dirty when a flag is new.
pub(super) fn check_fflags(env: &mut [u8], s: &FloatStatus) {
    let new = riscv_flags(s.flags());
    let old = ld64(env, FFLAGS);
    if new & !(old & FFLAGS_MASK) != 0 {
        st64(env, FFLAGS, old | new);
        let ms = ld64(env, MSTATUS);
        st64(env, MSTATUS, ms | MSTATUS_FS);
    }
}

/// [`accrue`] for the helpers that write an integer register, with
/// `riscv_cpu_check_fflags()`: `mstatus.FS` becomes dirty when a flag is new, as the
/// translator does not mark it.
fn accrue_check(e: &mut HelperEnv<'_>, f: impl FnOnce(&mut FloatStatus) -> u64) -> HR {
    let mut s = status(e.env);
    let r = f(&mut s);
    check_fflags(e.env, &s);
    Ok(u128::from(r))
}

// NaN boxing.

/// The canonical single precision NaN.
const DEFAULT_NAN_S: u32 = 0x7fc0_0000;
/// The upper half of a NaN boxed single precision value.
const NANBOX_S: u64 = 0xffff_ffff_0000_0000;

/// `nanbox_s()`.
pub(super) fn nanbox_s(f: Float32) -> u64 {
    u64::from(f.0) | NANBOX_S
}

/// `check_nanbox_s()`: a single precision value that is not NaN boxed reads as the
/// canonical NaN.
pub(super) fn unbox_s(f: u64) -> Float32 {
    if f & NANBOX_S == NANBOX_S { Float32(f as u32) } else { Float32(DEFAULT_NAN_S) }
}

/// The upper bits of a NaN boxed half precision or bf16 value.
const NANBOX_H: u64 = 0xffff_ffff_ffff_0000;

/// `nanbox_h()`.
fn nanbox_h(f: u16) -> u64 {
    u64::from(f) | NANBOX_H
}

/// `check_nanbox_h()`: a half precision value that is not NaN boxed reads as the canonical
/// NaN.
fn unbox_h(f: u64) -> Float16 {
    if f & NANBOX_H == NANBOX_H { Float16(f as u16) } else { Float16(0x7e00) }
}

/// `check_nanbox_bf16()`: a bf16 value that is not NaN boxed reads as the canonical NaN.
fn unbox_bf16(f: u64) -> BFloat16 {
    if f & NANBOX_H == NANBOX_H { BFloat16(f as u16) } else { BFloat16(0x7fc0) }
}

// The rounding mode.

def!(SET_ROUNDING_MODE, "set_rounding_mode", NO_WG, Void, [Ptr, I32], h_set_rounding_mode);

/// `HELPER(set_rounding_mode)`: the rounding mode `rm` of an instruction, or `frm` for
/// `DYN`, becomes the one of the helpers that follow. A reserved one is illegal.
fn h_set_rounding_mode(e: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let mut rm = u64::from(a[1] as u32);
    if rm == RISCV_FRM_DYN {
        rm = ld64(e.env, FRM);
    }
    if rm > RISCV_FRM_RMM {
        return run(e, |cpu| Err(cpu.raise_exception(EXCP_ILLEGAL_INST, Ra::Tb)));
    }
    st64(e.env, FP_ROUND, rm);
    Ok(0)
}

def!(
    SET_ROUNDING_MODE_CHKFRM,
    "set_rounding_mode_chkfrm",
    NO_WG,
    Void,
    [Ptr, I32],
    h_set_rounding_mode_chkfrm
);

/// `HELPER(set_rounding_mode_chkfrm)`: [`SET_ROUNDING_MODE`] for the vector conversions
/// with a static rounding mode, which may be round to odd. `frm` is checked even when it
/// is not used.
fn h_set_rounding_mode_chkfrm(e: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let mut rm = u64::from(a[1] as u32);
    let frm = ld64(e.env, FRM);
    if frm >= 5 {
        return run(e, |cpu| Err(cpu.raise_exception(EXCP_ILLEGAL_INST, Ra::Tb)));
    }
    if rm == RISCV_FRM_DYN {
        rm = frm;
    }
    if rm > RISCV_FRM_RMM && rm != RISCV_FRM_ROD {
        return run(e, |cpu| Err(cpu.raise_exception(EXCP_ILLEGAL_INST, Ra::Tb)));
    }
    st64(e.env, FP_ROUND, rm);
    Ok(0)
}

// Single precision.

macro_rules! s_fma {
    ($($f:ident => $fl:expr),* $(,)?) => {
        $(
            fn $f(x: u64, y: u64, z: u64, s: &mut FloatStatus) -> u64 {
                nanbox_s(unbox_s(x).muladd(unbox_s(y), unbox_s(z), $fl, s))
            }
        )*
    };
}

s_fma! {
    fmadd_s => 0,
    fmsub_s => muladd::NEGATE_C,
    fnmsub_s => muladd::NEGATE_PRODUCT,
    fnmadd_s => muladd::NEGATE_C | muladd::NEGATE_PRODUCT,
}

macro_rules! s_bin {
    ($($f:ident => $m:ident),* $(,)?) => {
        $(
            fn $f(x: u64, y: u64, s: &mut FloatStatus) -> u64 {
                nanbox_s(unbox_s(x).$m(unbox_s(y), s))
            }
        )*
    };
}

s_bin! {
    fadd_s => add,
    fsub_s => sub,
    fmul_s => mul,
    fdiv_s => div,
    fmin_s => minimum_number,
    fminm_s => min,
    fmax_s => maximum_number,
    fmaxm_s => max,
}

macro_rules! s_cmp {
    ($($f:ident => $m:ident),* $(,)?) => {
        $(
            fn $f(x: u64, y: u64, s: &mut FloatStatus) -> u64 {
                u64::from(unbox_s(x).$m(unbox_s(y), s))
            }
        )*
    };
}

s_cmp! {
    fle_s => le,
    fleq_s => le_quiet,
    flt_s => lt,
    fltq_s => lt_quiet,
    feq_s => eq_quiet,
}

fn fsqrt_s(x: u64, s: &mut FloatStatus) -> u64 {
    nanbox_s(unbox_s(x).sqrt(s))
}

/// `fcvt.w.s`: the result is sign extended.
fn fcvt_w_s(x: u64, s: &mut FloatStatus) -> u64 {
    i64::from(unbox_s(x).to_i32(s)) as u64
}

/// `fcvt.wu.s`: the result is sign extended too.
fn fcvt_wu_s(x: u64, s: &mut FloatStatus) -> u64 {
    i64::from(unbox_s(x).to_u32(s) as i32) as u64
}

fn fcvt_l_s(x: u64, s: &mut FloatStatus) -> u64 {
    unbox_s(x).to_i64(s) as u64
}

fn fcvt_lu_s(x: u64, s: &mut FloatStatus) -> u64 {
    unbox_s(x).to_u64(s)
}

fn fcvt_s_w(x: u64, s: &mut FloatStatus) -> u64 {
    nanbox_s(Float32::from_i32(x as i32, s))
}

fn fcvt_s_wu(x: u64, s: &mut FloatStatus) -> u64 {
    nanbox_s(Float32::from_u32(x as u32, s))
}

fn fcvt_s_l(x: u64, s: &mut FloatStatus) -> u64 {
    nanbox_s(Float32::from_i64(x as i64, s))
}

fn fcvt_s_lu(x: u64, s: &mut FloatStatus) -> u64 {
    nanbox_s(Float32::from_u64(x, s))
}

/// `fclass_s()`.
fn fclass_s(f: Float32) -> u64 {
    let sign = f.is_neg();
    if f.is_infinity() {
        if sign { 1 << 0 } else { 1 << 7 }
    } else if f.is_zero() {
        if sign { 1 << 3 } else { 1 << 4 }
    } else if f.is_zero_or_denormal() {
        if sign { 1 << 2 } else { 1 << 5 }
    } else if f.is_any_nan() {
        if f.is_quiet_nan(&FloatStatus::default()) { 1 << 9 } else { 1 << 8 }
    } else if sign {
        1 << 1
    } else {
        1 << 6
    }
}

/// `HELPER(fclass_s)`: raises no flags.
fn fclass_s_op(x: u64, _s: &mut FloatStatus) -> u64 {
    fclass_s(unbox_s(x))
}

/// `fround.s`: the inexact flag stays as it was.
fn fround_s(x: u64, s: &mut FloatStatus) -> u64 {
    let r = unbox_s(x).round_to_int(s);
    s.exception_flags &= !flags::INEXACT;
    nanbox_s(r)
}

fn froundnx_s(x: u64, s: &mut FloatStatus) -> u64 {
    nanbox_s(unbox_s(x).round_to_int(s))
}

// Double precision.

macro_rules! d_fma {
    ($($f:ident => $fl:expr),* $(,)?) => {
        $(
            fn $f(x: u64, y: u64, z: u64, s: &mut FloatStatus) -> u64 {
                Float64(x).muladd(Float64(y), Float64(z), $fl, s).0
            }
        )*
    };
}

d_fma! {
    fmadd_d => 0,
    fmsub_d => muladd::NEGATE_C,
    fnmsub_d => muladd::NEGATE_PRODUCT,
    fnmadd_d => muladd::NEGATE_C | muladd::NEGATE_PRODUCT,
}

macro_rules! d_bin {
    ($($f:ident => $m:ident),* $(,)?) => {
        $(
            fn $f(x: u64, y: u64, s: &mut FloatStatus) -> u64 {
                Float64(x).$m(Float64(y), s).0
            }
        )*
    };
}

d_bin! {
    fadd_d => add,
    fsub_d => sub,
    fmul_d => mul,
    fdiv_d => div,
    fmin_d => minimum_number,
    fminm_d => min,
    fmax_d => maximum_number,
    fmaxm_d => max,
}

macro_rules! d_cmp {
    ($($f:ident => $m:ident),* $(,)?) => {
        $(
            fn $f(x: u64, y: u64, s: &mut FloatStatus) -> u64 {
                u64::from(Float64(x).$m(Float64(y), s))
            }
        )*
    };
}

d_cmp! {
    fle_d => le,
    fleq_d => le_quiet,
    flt_d => lt,
    fltq_d => lt_quiet,
    feq_d => eq_quiet,
}

fn fsqrt_d(x: u64, s: &mut FloatStatus) -> u64 {
    Float64(x).sqrt(s).0
}

fn fcvt_s_d(x: u64, s: &mut FloatStatus) -> u64 {
    nanbox_s(Float64(x).to_float32(s))
}

fn fcvt_d_s(x: u64, s: &mut FloatStatus) -> u64 {
    unbox_s(x).to_float64(s).0
}

/// `fcvt.w.d`: the result is sign extended.
fn fcvt_w_d(x: u64, s: &mut FloatStatus) -> u64 {
    i64::from(Float64(x).to_i32(s)) as u64
}

/// `fcvtmod.w.d`: truncate, then wrap modulo 2^32 and sign extend.
fn fcvtmod_w_d(x: u64, s: &mut FloatStatus) -> u64 {
    i64::from(Float64(x).to_i32_modulo(RoundMode::ToZero, s)) as u64
}

/// `fcvt.wu.d`: the result is sign extended too.
fn fcvt_wu_d(x: u64, s: &mut FloatStatus) -> u64 {
    i64::from(Float64(x).to_u32(s) as i32) as u64
}

fn fcvt_l_d(x: u64, s: &mut FloatStatus) -> u64 {
    Float64(x).to_i64(s) as u64
}

fn fcvt_lu_d(x: u64, s: &mut FloatStatus) -> u64 {
    Float64(x).to_u64(s)
}

fn fcvt_d_w(x: u64, s: &mut FloatStatus) -> u64 {
    Float64::from_i32(x as i32, s).0
}

fn fcvt_d_wu(x: u64, s: &mut FloatStatus) -> u64 {
    Float64::from_u32(x as u32, s).0
}

fn fcvt_d_l(x: u64, s: &mut FloatStatus) -> u64 {
    Float64::from_i64(x as i64, s).0
}

fn fcvt_d_lu(x: u64, s: &mut FloatStatus) -> u64 {
    Float64::from_u64(x, s).0
}

/// `fclass_d()`.
fn fclass_d(f: Float64) -> u64 {
    let sign = f.is_neg();
    if f.is_infinity() {
        if sign { 1 << 0 } else { 1 << 7 }
    } else if f.is_zero() {
        if sign { 1 << 3 } else { 1 << 4 }
    } else if f.is_zero_or_denormal() {
        if sign { 1 << 2 } else { 1 << 5 }
    } else if f.is_any_nan() {
        if f.is_quiet_nan(&FloatStatus::default()) { 1 << 9 } else { 1 << 8 }
    } else if sign {
        1 << 1
    } else {
        1 << 6
    }
}

/// `HELPER(fclass_d)`, which takes no `env`.
fn h_fclass_d(_e: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    Ok(u128::from(fclass_d(Float64(a[0]))))
}

/// `fround.d`: the inexact flag stays as it was.
fn fround_d(x: u64, s: &mut FloatStatus) -> u64 {
    let r = Float64(x).round_to_int(s);
    s.exception_flags &= !flags::INEXACT;
    r.0
}

fn froundnx_d(x: u64, s: &mut FloatStatus) -> u64 {
    Float64(x).round_to_int(s).0
}

// The declarations, with the flags of QEMU's `helper.h`.

// Zfhmin and Zfbfmin.

fn fcvt_h_s(x: u64, s: &mut FloatStatus) -> u64 {
    nanbox_h(unbox_s(x).to_float16(true, s).0)
}

fn fcvt_s_h(x: u64, s: &mut FloatStatus) -> u64 {
    nanbox_s(unbox_h(x).to_float32(true, s))
}

fn fcvt_h_d(x: u64, s: &mut FloatStatus) -> u64 {
    nanbox_h(Float64(x).to_float16(true, s).0)
}

fn fcvt_d_h(x: u64, s: &mut FloatStatus) -> u64 {
    unbox_h(x).to_float64(true, s).0
}

fn fcvt_bf16_s(x: u64, s: &mut FloatStatus) -> u64 {
    nanbox_h(unbox_s(x).to_bfloat16(s).0)
}

fn fcvt_s_bf16(x: u64, s: &mut FloatStatus) -> u64 {
    nanbox_s(unbox_bf16(x).to_float32(s))
}

// Half precision (Zfh).

macro_rules! h_fma {
    ($($f:ident => $fl:expr),* $(,)?) => {
        $(
            fn $f(x: u64, y: u64, z: u64, s: &mut FloatStatus) -> u64 {
                nanbox_h(unbox_h(x).muladd(unbox_h(y), unbox_h(z), $fl, s).0)
            }
        )*
    };
}

h_fma! {
    fmadd_h => 0,
    fmsub_h => muladd::NEGATE_C,
    fnmsub_h => muladd::NEGATE_PRODUCT,
    fnmadd_h => muladd::NEGATE_C | muladd::NEGATE_PRODUCT,
}

macro_rules! h_bin {
    ($($f:ident => $m:ident),* $(,)?) => {
        $(
            fn $f(x: u64, y: u64, s: &mut FloatStatus) -> u64 {
                nanbox_h(unbox_h(x).$m(unbox_h(y), s).0)
            }
        )*
    };
}

h_bin! {
    fadd_h => add,
    fsub_h => sub,
    fmul_h => mul,
    fdiv_h => div,
    fmin_h => minimum_number,
    fminm_h => min,
    fmax_h => maximum_number,
    fmaxm_h => max,
}

macro_rules! h_cmp {
    ($($f:ident => $m:ident),* $(,)?) => {
        $(
            fn $f(x: u64, y: u64, s: &mut FloatStatus) -> u64 {
                u64::from(unbox_h(x).$m(unbox_h(y), s))
            }
        )*
    };
}

h_cmp! {
    fle_h => le,
    fleq_h => le_quiet,
    flt_h => lt,
    fltq_h => lt_quiet,
    feq_h => eq_quiet,
}

fn fsqrt_h(x: u64, s: &mut FloatStatus) -> u64 {
    nanbox_h(unbox_h(x).sqrt(s).0)
}

/// `fcvt.w.h`: the result is sign extended.
fn fcvt_w_h(x: u64, s: &mut FloatStatus) -> u64 {
    i64::from(unbox_h(x).to_i32(s)) as u64
}

/// `fcvt.wu.h`: the result is sign extended too.
fn fcvt_wu_h(x: u64, s: &mut FloatStatus) -> u64 {
    i64::from(unbox_h(x).to_u32(s) as i32) as u64
}

fn fcvt_l_h(x: u64, s: &mut FloatStatus) -> u64 {
    unbox_h(x).to_i64(s) as u64
}

fn fcvt_lu_h(x: u64, s: &mut FloatStatus) -> u64 {
    unbox_h(x).to_u64(s)
}

fn fcvt_h_w(x: u64, s: &mut FloatStatus) -> u64 {
    nanbox_h(Float16::from_i32(x as i32, s).0)
}

fn fcvt_h_wu(x: u64, s: &mut FloatStatus) -> u64 {
    nanbox_h(Float16::from_u32(x as u32, s).0)
}

fn fcvt_h_l(x: u64, s: &mut FloatStatus) -> u64 {
    nanbox_h(Float16::from_i64(x as i64, s).0)
}

fn fcvt_h_lu(x: u64, s: &mut FloatStatus) -> u64 {
    nanbox_h(Float16::from_u64(x, s).0)
}

/// `HELPER(fclass_h)`, `fclass_h()` of the NaN unboxed value: raises no flags.
fn fclass_h_op(x: u64, _s: &mut FloatStatus) -> u64 {
    let f = unbox_h(x);
    let sign = f.is_neg();
    if f.is_infinity() {
        if sign { 1 << 0 } else { 1 << 7 }
    } else if f.is_zero() {
        if sign { 1 << 3 } else { 1 << 4 }
    } else if f.is_zero_or_denormal() {
        if sign { 1 << 2 } else { 1 << 5 }
    } else if f.is_any_nan() {
        if f.is_quiet_nan(&FloatStatus::default()) { 1 << 9 } else { 1 << 8 }
    } else if sign {
        1 << 1
    } else {
        1 << 6
    }
}

/// `fround.h`: the inexact flag stays as it was.
fn fround_h(x: u64, s: &mut FloatStatus) -> u64 {
    let r = unbox_h(x).round_to_int(s);
    s.exception_flags &= !flags::INEXACT;
    nanbox_h(r.0)
}

fn froundnx_h(x: u64, s: &mut FloatStatus) -> u64 {
    nanbox_h(unbox_h(x).round_to_int(s).0)
}

fp_def!(FMADD_S, "fmadd_s", NO_RWG, accrue, fmadd_s, 3);
fp_def!(FMSUB_S, "fmsub_s", NO_RWG, accrue, fmsub_s, 3);
fp_def!(FNMSUB_S, "fnmsub_s", NO_RWG, accrue, fnmsub_s, 3);
fp_def!(FNMADD_S, "fnmadd_s", NO_RWG, accrue, fnmadd_s, 3);
fp_def!(FADD_S, "fadd_s", NO_RWG, accrue, fadd_s, 2);
fp_def!(FSUB_S, "fsub_s", NO_RWG, accrue, fsub_s, 2);
fp_def!(FMUL_S, "fmul_s", NO_RWG, accrue, fmul_s, 2);
fp_def!(FDIV_S, "fdiv_s", NO_RWG, accrue, fdiv_s, 2);
fp_def!(FMIN_S, "fmin_s", NO_RWG, accrue, fmin_s, 2);
fp_def!(FMINM_S, "fminm_s", NO_RWG, accrue, fminm_s, 2);
fp_def!(FMAX_S, "fmax_s", NO_RWG, accrue, fmax_s, 2);
fp_def!(FMAXM_S, "fmaxm_s", NO_RWG, accrue, fmaxm_s, 2);
fp_def!(FSQRT_S, "fsqrt_s", NO_RWG, accrue, fsqrt_s, 1);
fp_def!(FLE_S, "fle_s", NO_RWG, accrue_check, fle_s, 2);
fp_def!(FLEQ_S, "fleq_s", NO_RWG, accrue_check, fleq_s, 2);
fp_def!(FLT_S, "flt_s", NO_RWG, accrue_check, flt_s, 2);
fp_def!(FLTQ_S, "fltq_s", NO_RWG, accrue_check, fltq_s, 2);
fp_def!(FEQ_S, "feq_s", NO_RWG, accrue_check, feq_s, 2);
fp_def!(FCVT_W_S, "fcvt_w_s", NO_RWG, accrue_check, fcvt_w_s, 1);
fp_def!(FCVT_WU_S, "fcvt_wu_s", NO_RWG, accrue_check, fcvt_wu_s, 1);
fp_def!(FCVT_L_S, "fcvt_l_s", NO_RWG, accrue_check, fcvt_l_s, 1);
fp_def!(FCVT_LU_S, "fcvt_lu_s", NO_RWG, accrue_check, fcvt_lu_s, 1);
fp_def!(FCVT_S_W, "fcvt_s_w", NO_RWG, accrue, fcvt_s_w, 1);
fp_def!(FCVT_S_WU, "fcvt_s_wu", NO_RWG, accrue, fcvt_s_wu, 1);
fp_def!(FCVT_S_L, "fcvt_s_l", NO_RWG, accrue, fcvt_s_l, 1);
fp_def!(FCVT_S_LU, "fcvt_s_lu", NO_RWG, accrue, fcvt_s_lu, 1);
fp_def!(FCLASS_S, "fclass_s", NO_RWG_SE, accrue, fclass_s_op, 1);
fp_def!(FROUND_S, "fround_s", NO_RWG_SE, accrue, fround_s, 1);
fp_def!(FROUNDNX_S, "froundnx_s", NO_RWG_SE, accrue, froundnx_s, 1);

fp_def!(FMADD_D, "fmadd_d", NO_RWG, accrue, fmadd_d, 3);
fp_def!(FMSUB_D, "fmsub_d", NO_RWG, accrue, fmsub_d, 3);
fp_def!(FNMSUB_D, "fnmsub_d", NO_RWG, accrue, fnmsub_d, 3);
fp_def!(FNMADD_D, "fnmadd_d", NO_RWG, accrue, fnmadd_d, 3);
fp_def!(FADD_D, "fadd_d", NO_RWG, accrue, fadd_d, 2);
fp_def!(FSUB_D, "fsub_d", NO_RWG, accrue, fsub_d, 2);
fp_def!(FMUL_D, "fmul_d", NO_RWG, accrue, fmul_d, 2);
fp_def!(FDIV_D, "fdiv_d", NO_RWG, accrue, fdiv_d, 2);
fp_def!(FMIN_D, "fmin_d", NO_RWG, accrue, fmin_d, 2);
fp_def!(FMINM_D, "fminm_d", NO_RWG, accrue, fminm_d, 2);
fp_def!(FMAX_D, "fmax_d", NO_RWG, accrue, fmax_d, 2);
fp_def!(FMAXM_D, "fmaxm_d", NO_RWG, accrue, fmaxm_d, 2);
fp_def!(FCVT_S_D, "fcvt_s_d", NO_RWG, accrue, fcvt_s_d, 1);
fp_def!(FCVT_D_S, "fcvt_d_s", NO_RWG, accrue, fcvt_d_s, 1);
fp_def!(FSQRT_D, "fsqrt_d", NO_RWG, accrue, fsqrt_d, 1);
fp_def!(FLE_D, "fle_d", NO_RWG, accrue_check, fle_d, 2);
fp_def!(FLEQ_D, "fleq_d", NO_RWG, accrue_check, fleq_d, 2);
fp_def!(FLT_D, "flt_d", NO_RWG, accrue_check, flt_d, 2);
fp_def!(FLTQ_D, "fltq_d", NO_RWG, accrue_check, fltq_d, 2);
fp_def!(FEQ_D, "feq_d", NO_RWG, accrue_check, feq_d, 2);
fp_def!(FCVT_W_D, "fcvt_w_d", NO_RWG, accrue_check, fcvt_w_d, 1);
fp_def!(FCVTMOD_W_D, "fcvtmod_w_d", NO_RWG, accrue_check, fcvtmod_w_d, 1);
fp_def!(FCVT_WU_D, "fcvt_wu_d", NO_RWG, accrue_check, fcvt_wu_d, 1);
fp_def!(FCVT_L_D, "fcvt_l_d", NO_RWG, accrue_check, fcvt_l_d, 1);
fp_def!(FCVT_LU_D, "fcvt_lu_d", NO_RWG, accrue_check, fcvt_lu_d, 1);
fp_def!(FCVT_D_W, "fcvt_d_w", NO_RWG, accrue, fcvt_d_w, 1);
fp_def!(FCVT_D_WU, "fcvt_d_wu", NO_RWG, accrue, fcvt_d_wu, 1);
fp_def!(FCVT_D_L, "fcvt_d_l", NO_RWG, accrue, fcvt_d_l, 1);
fp_def!(FCVT_D_LU, "fcvt_d_lu", NO_RWG, accrue, fcvt_d_lu, 1);
def!(FCLASS_D, "fclass_d", NO_RWG_SE, I64, [I64], h_fclass_d);
fp_def!(FROUND_D, "fround_d", NO_RWG_SE, accrue, fround_d, 1);
fp_def!(FROUNDNX_D, "froundnx_d", NO_RWG_SE, accrue, froundnx_d, 1);

fp_def!(FCVT_H_S, "fcvt_h_s", NO_RWG, accrue, fcvt_h_s, 1);
fp_def!(FCVT_S_H, "fcvt_s_h", NO_RWG, accrue, fcvt_s_h, 1);
fp_def!(FCVT_H_D, "fcvt_h_d", NO_RWG, accrue, fcvt_h_d, 1);
fp_def!(FCVT_D_H, "fcvt_d_h", NO_RWG, accrue, fcvt_d_h, 1);
fp_def!(FCVT_BF16_S, "fcvt_bf16_s", NO_RWG, accrue, fcvt_bf16_s, 1);
fp_def!(FCVT_S_BF16, "fcvt_s_bf16", NO_RWG, accrue, fcvt_s_bf16, 1);

fp_def!(FMADD_H, "fmadd_h", NO_RWG, accrue, fmadd_h, 3);
fp_def!(FMSUB_H, "fmsub_h", NO_RWG, accrue, fmsub_h, 3);
fp_def!(FNMSUB_H, "fnmsub_h", NO_RWG, accrue, fnmsub_h, 3);
fp_def!(FNMADD_H, "fnmadd_h", NO_RWG, accrue, fnmadd_h, 3);
fp_def!(FADD_H, "fadd_h", NO_RWG, accrue, fadd_h, 2);
fp_def!(FSUB_H, "fsub_h", NO_RWG, accrue, fsub_h, 2);
fp_def!(FMUL_H, "fmul_h", NO_RWG, accrue, fmul_h, 2);
fp_def!(FDIV_H, "fdiv_h", NO_RWG, accrue, fdiv_h, 2);
fp_def!(FMIN_H, "fmin_h", NO_RWG, accrue, fmin_h, 2);
fp_def!(FMINM_H, "fminm_h", NO_RWG, accrue, fminm_h, 2);
fp_def!(FMAX_H, "fmax_h", NO_RWG, accrue, fmax_h, 2);
fp_def!(FMAXM_H, "fmaxm_h", NO_RWG, accrue, fmaxm_h, 2);
fp_def!(FSQRT_H, "fsqrt_h", NO_RWG, accrue, fsqrt_h, 1);
fp_def!(FLE_H, "fle_h", NO_RWG, accrue_check, fle_h, 2);
fp_def!(FLEQ_H, "fleq_h", NO_RWG, accrue_check, fleq_h, 2);
fp_def!(FLT_H, "flt_h", NO_RWG, accrue_check, flt_h, 2);
fp_def!(FLTQ_H, "fltq_h", NO_RWG, accrue_check, fltq_h, 2);
fp_def!(FEQ_H, "feq_h", NO_RWG, accrue_check, feq_h, 2);
fp_def!(FCVT_W_H, "fcvt_w_h", NO_RWG, accrue_check, fcvt_w_h, 1);
fp_def!(FCVT_WU_H, "fcvt_wu_h", NO_RWG, accrue_check, fcvt_wu_h, 1);
fp_def!(FCVT_L_H, "fcvt_l_h", NO_RWG, accrue_check, fcvt_l_h, 1);
fp_def!(FCVT_LU_H, "fcvt_lu_h", NO_RWG, accrue_check, fcvt_lu_h, 1);
fp_def!(FCVT_H_W, "fcvt_h_w", NO_RWG, accrue, fcvt_h_w, 1);
fp_def!(FCVT_H_WU, "fcvt_h_wu", NO_RWG, accrue, fcvt_h_wu, 1);
fp_def!(FCVT_H_L, "fcvt_h_l", NO_RWG, accrue, fcvt_h_l, 1);
fp_def!(FCVT_H_LU, "fcvt_h_lu", NO_RWG, accrue, fcvt_h_lu, 1);
fp_def!(FCLASS_H, "fclass_h", NO_RWG_SE, accrue, fclass_h_op, 1);
fp_def!(FROUND_H, "fround_h", NO_RWG_SE, accrue, fround_h, 1);
fp_def!(FROUNDNX_H, "froundnx_h", NO_RWG_SE, accrue, froundnx_h, 1);

/// The helpers of this module.
pub(crate) const ALL: &[Def] = &[
    SET_ROUNDING_MODE,
    SET_ROUNDING_MODE_CHKFRM,
    FMADD_S,
    FMSUB_S,
    FNMSUB_S,
    FNMADD_S,
    FADD_S,
    FSUB_S,
    FMUL_S,
    FDIV_S,
    FMIN_S,
    FMINM_S,
    FMAX_S,
    FMAXM_S,
    FSQRT_S,
    FLE_S,
    FLEQ_S,
    FLT_S,
    FLTQ_S,
    FEQ_S,
    FCVT_W_S,
    FCVT_WU_S,
    FCVT_L_S,
    FCVT_LU_S,
    FCVT_S_W,
    FCVT_S_WU,
    FCVT_S_L,
    FCVT_S_LU,
    FCLASS_S,
    FROUND_S,
    FROUNDNX_S,
    FMADD_D,
    FMSUB_D,
    FNMSUB_D,
    FNMADD_D,
    FADD_D,
    FSUB_D,
    FMUL_D,
    FDIV_D,
    FMIN_D,
    FMINM_D,
    FMAX_D,
    FMAXM_D,
    FCVT_S_D,
    FCVT_D_S,
    FSQRT_D,
    FLE_D,
    FLEQ_D,
    FLT_D,
    FLTQ_D,
    FEQ_D,
    FCVT_W_D,
    FCVTMOD_W_D,
    FCVT_WU_D,
    FCVT_L_D,
    FCVT_LU_D,
    FCVT_D_W,
    FCVT_D_WU,
    FCVT_D_L,
    FCVT_D_LU,
    FCLASS_D,
    FROUND_D,
    FROUNDNX_D,
    FCVT_H_S,
    FCVT_S_H,
    FCVT_H_D,
    FCVT_D_H,
    FCVT_BF16_S,
    FCVT_S_BF16,
    FMADD_H,
    FMSUB_H,
    FNMSUB_H,
    FNMADD_H,
    FADD_H,
    FSUB_H,
    FMUL_H,
    FDIV_H,
    FMIN_H,
    FMINM_H,
    FMAX_H,
    FMAXM_H,
    FSQRT_H,
    FLE_H,
    FLEQ_H,
    FLT_H,
    FLTQ_H,
    FEQ_H,
    FCVT_W_H,
    FCVT_WU_H,
    FCVT_L_H,
    FCVT_LU_H,
    FCVT_H_W,
    FCVT_H_WU,
    FCVT_H_L,
    FCVT_H_LU,
    FCLASS_H,
    FROUND_H,
    FROUNDNX_H,
];

#[cfg(test)]
mod tests {
    use super::*;

    const S_QNAN: u64 = NANBOX_S | 0x7fc0_0000;
    const S_SNAN: u64 = NANBOX_S | 0x7f80_0001;
    const S_ONE: u64 = NANBOX_S | 0x3f80_0000;
    const S_TWO: u64 = NANBOX_S | 0x4000_0000;
    const D_QNAN: u64 = 0x7ff8_0000_0000_0000;
    const D_SNAN: u64 = 0x7ff0_0000_0000_0001;
    const D_ONE: u64 = 0x3ff0_0000_0000_0000;

    fn rne() -> FloatStatus {
        status_rm(0)
    }

    #[test]
    fn fclass_bits() {
        let s = |v: u32| fclass_s_op(NANBOX_S | u64::from(v), &mut rne());
        assert_eq!(s(0xff80_0000), 1 << 0);
        assert_eq!(s(0xbf80_0000), 1 << 1);
        assert_eq!(s(0x8000_0001), 1 << 2);
        assert_eq!(s(0x8000_0000), 1 << 3);
        assert_eq!(s(0x0000_0000), 1 << 4);
        assert_eq!(s(0x0000_0001), 1 << 5);
        assert_eq!(s(0x3f80_0000), 1 << 6);
        assert_eq!(s(0x7f80_0000), 1 << 7);
        assert_eq!(s(0x7f80_0001), 1 << 8);
        assert_eq!(s(0x7fc0_0000), 1 << 9);
        // A value that is not NaN boxed is the canonical NaN.
        assert_eq!(fclass_s_op(0x3f80_0000, &mut rne()), 1 << 9);

        let d = |v: u64| fclass_d(Float64(v));
        assert_eq!(d(0xfff0_0000_0000_0000), 1 << 0);
        assert_eq!(d(0xbff0_0000_0000_0000), 1 << 1);
        assert_eq!(d(0x8000_0000_0000_0001), 1 << 2);
        assert_eq!(d(0x8000_0000_0000_0000), 1 << 3);
        assert_eq!(d(0), 1 << 4);
        assert_eq!(d(1), 1 << 5);
        assert_eq!(d(D_ONE), 1 << 6);
        assert_eq!(d(0x7ff0_0000_0000_0000), 1 << 7);
        assert_eq!(d(D_SNAN), 1 << 8);
        assert_eq!(d(D_QNAN), 1 << 9);
    }

    #[test]
    fn fcvt_saturates() {
        let mut s = rne();
        // NaN converts to the largest value, with invalid.
        assert_eq!(fcvt_w_s(S_QNAN, &mut s), 0x7fff_ffff);
        assert_eq!(riscv_flags(s.flags()), FPEXC_NV);
        let mut s = rne();
        assert_eq!(fcvt_wu_s(S_QNAN, &mut s), u64::MAX, "sign extended");
        assert_eq!(fcvt_l_s(S_QNAN, &mut s), i64::MAX as u64);
        assert_eq!(fcvt_lu_s(S_QNAN, &mut s), u64::MAX);
        assert_eq!(fcvt_w_d(D_QNAN, &mut s), 0x7fff_ffff);
        assert_eq!(fcvt_l_d(D_QNAN, &mut s), i64::MAX as u64);

        // Out of range values saturate.
        let neg_inf_s = NANBOX_S | 0xff80_0000;
        assert_eq!(fcvt_w_s(neg_inf_s, &mut s), i64::from(i32::MIN) as u64);
        assert_eq!(fcvt_wu_s(neg_inf_s, &mut s), 0);
        assert_eq!(fcvt_lu_s(neg_inf_s, &mut s), 0);
        let big_d = 0x41f0_0000_0000_0000; // 2^32
        assert_eq!(fcvt_wu_d(big_d, &mut s), u64::MAX, "u32::MAX sign extended");
        assert_eq!(fcvt_w_d(big_d, &mut s), 0x7fff_ffff);
        let mut s = rne();
        assert_eq!(fcvt_lu_d(0xbff0_0000_0000_0000, &mut s), 0, "-1.0");
        assert_eq!(riscv_flags(s.flags()), FPEXC_NV);

        // In range: rounding follows the mode, inexact is raised.
        let mut s = rne();
        assert_eq!(fcvt_w_d(0x3ff8_0000_0000_0000, &mut s), 2, "1.5 to even");
        assert_eq!(riscv_flags(s.flags()), FPEXC_NX);
        let mut s = status_rm(RISCV_FRM_RTZ);
        assert_eq!(fcvt_w_d(0xbff8_0000_0000_0000, &mut s), (-1i64) as u64, "-1.5");
        let mut s = status_rm(RISCV_FRM_RMM);
        assert_eq!(fcvt_w_s(NANBOX_S | 0x4020_0000, &mut s), 3, "2.5 away");

        // fcvtmod.w.d wraps modulo 2^32 and raises invalid out of range.
        let mut s = status_rm(RISCV_FRM_RTZ);
        assert_eq!(fcvtmod_w_d(big_d, &mut s), 0);
        assert_eq!(riscv_flags(s.flags()), FPEXC_NV);
        let mut s = status_rm(RISCV_FRM_RTZ);
        assert_eq!(fcvtmod_w_d(0x41e0_0000_0000_0000, &mut s), i64::from(i32::MIN) as u64);
        let mut s = status_rm(RISCV_FRM_RTZ);
        assert_eq!(fcvtmod_w_d(D_QNAN, &mut s), 0);
        assert_eq!(riscv_flags(s.flags()), FPEXC_NV);
    }

    #[test]
    fn fmin_fmax_nans() {
        let mut s = rne();
        // minimumNumber: a quiet NaN loses without a flag.
        assert_eq!(fmin_s(S_QNAN, S_ONE, &mut s), S_ONE);
        assert_eq!(fmax_s(S_TWO, S_QNAN, &mut s), S_TWO);
        assert_eq!(s.flags(), 0);
        // A signaling NaN loses too, but raises invalid.
        assert_eq!(fmin_s(S_SNAN, S_ONE, &mut s), S_ONE);
        assert_eq!(riscv_flags(s.flags()), FPEXC_NV);
        // Two NaNs give the canonical NaN.
        let mut s = rne();
        assert_eq!(fmax_s(S_SNAN, S_QNAN, &mut s), S_QNAN);
        assert_eq!(fmin_d(D_QNAN, D_SNAN, &mut s), D_QNAN);
        // A value that is not NaN boxed is a NaN.
        assert_eq!(fmin_s(0x3f80_0000, S_TWO, &mut s), S_TWO);
        // -0 is below +0.
        let mut s = rne();
        assert_eq!(fmin_d(0, 1 << 63, &mut s), 1 << 63);
        assert_eq!(fmax_d(1 << 63, 0, &mut s), 0);

        // fminm and fmaxm (Zfa) propagate NaN as the canonical NaN.
        let mut s = rne();
        assert_eq!(fminm_s(S_QNAN, S_ONE, &mut s), S_QNAN);
        assert_eq!(s.flags(), 0);
        assert_eq!(fmaxm_d(D_ONE, D_SNAN, &mut s), D_QNAN);
        assert_eq!(riscv_flags(s.flags()), FPEXC_NV);
        let mut s = rne();
        assert_eq!(fmaxm_s(S_ONE, S_TWO, &mut s), S_TWO);
    }

    #[test]
    fn arith_nan_boxing_and_flags() {
        let mut s = rne();
        assert_eq!(fadd_s(S_ONE, S_ONE, &mut s), S_TWO);
        // Any NaN result is the canonical NaN.
        assert_eq!(fadd_s(S_SNAN, S_ONE, &mut s), S_QNAN);
        assert_eq!(fadd_d(0x7ff8_0000_0000_1234, D_ONE, &mut s), D_QNAN);
        let mut s = rne();
        assert_eq!(fdiv_d(D_ONE, 0, &mut s), 0x7ff0_0000_0000_0000);
        assert_eq!(riscv_flags(s.flags()), FPEXC_DZ);
        // inf * 0 + c is invalid.
        let mut s = rne();
        assert_eq!(fmadd_d(0x7ff0_0000_0000_0000, 0, D_ONE, &mut s), D_QNAN);
        assert_eq!(riscv_flags(s.flags()), FPEXC_NV);
        // fcvt.s.d of a NaN is the canonical NaN, NaN boxed.
        let mut s = rne();
        assert_eq!(fcvt_s_d(D_SNAN, &mut s), S_QNAN);
        assert_eq!(fcvt_d_s(S_ONE, &mut s), D_ONE);
        // Comparisons: flt signals on quiet NaNs, feq and fltq do not.
        let mut s = rne();
        assert_eq!(feq_s(S_QNAN, S_ONE, &mut s), 0);
        assert_eq!(fltq_d(D_QNAN, D_ONE, &mut s), 0);
        assert_eq!(s.flags(), 0);
        assert_eq!(flt_s(S_QNAN, S_ONE, &mut s), 0);
        assert_eq!(riscv_flags(s.flags()), FPEXC_NV);
        let mut s = rne();
        assert_eq!(fle_d(D_ONE, D_ONE, &mut s), 1);
    }

    #[test]
    fn fround_keeps_nx() {
        let mut s = rne();
        assert_eq!(fround_d(0x3ff8_0000_0000_0000, &mut s), 0x4000_0000_0000_0000);
        assert_eq!(s.flags(), 0);
        let mut s = rne();
        assert_eq!(froundnx_s(NANBOX_S | 0x3fc0_0000, &mut s), S_TWO);
        assert_eq!(riscv_flags(s.flags()), FPEXC_NX);
    }

    #[test]
    fn flag_mapping() {
        assert_eq!(riscv_flags(flags::INEXACT), FPEXC_NX);
        assert_eq!(riscv_flags(flags::UNDERFLOW), FPEXC_UF);
        assert_eq!(riscv_flags(flags::OVERFLOW), FPEXC_OF);
        assert_eq!(riscv_flags(flags::DIVBYZERO), FPEXC_DZ);
        assert_eq!(riscv_flags(flags::INVALID | flags::INVALID_CVTI), FPEXC_NV);
        // Tininess after rounding: the largest subnormal plus a bit that rounds up to the
        // smallest normal is not an underflow.
        let mut s = rne();
        let x = 0x000f_ffff_ffff_ffff; // largest subnormal
        let r = fmul_d(x, 0x3ff0_0000_0000_0001, &mut s); // 1 + ulp
        assert_eq!(r, 0x0010_0000_0000_0000);
        assert_eq!(riscv_flags(s.flags()), FPEXC_NX);
    }

    #[test]
    fn half_conversions_nan_box() {
        let mut s = rne();
        // 1.0 in half precision and bf16.
        assert_eq!(fcvt_h_s(S_ONE, &mut s), NANBOX_H | 0x3c00);
        assert_eq!(fcvt_s_h(NANBOX_H | 0x3c00, &mut s), S_ONE);
        assert_eq!(fcvt_d_h(NANBOX_H | 0x3c00, &mut s), D_ONE);
        assert_eq!(fcvt_h_d(D_ONE, &mut s), NANBOX_H | 0x3c00);
        assert_eq!(fcvt_bf16_s(S_ONE, &mut s), NANBOX_H | 0x3f80);
        assert_eq!(fcvt_s_bf16(NANBOX_H | 0x3f80, &mut s), S_ONE);
        assert_eq!(s.flags(), 0);
        // A half precision value that is not NaN boxed is the canonical NaN, which
        // converts quietly.
        assert_eq!(fcvt_s_h(0x0000_ffff_0000_3c00, &mut s), S_QNAN);
        assert_eq!(fcvt_d_h(0x3c00, &mut s), D_QNAN);
        assert_eq!(fcvt_s_bf16(0x3f80, &mut s), S_QNAN);
        // A single precision value that is not NaN boxed narrows to the canonical NaN.
        assert_eq!(fcvt_h_s(0x3f80_0000, &mut s), NANBOX_H | 0x7e00);
        assert_eq!(fcvt_bf16_s(0x3f80_0000, &mut s), NANBOX_H | 0x7fc0);
        assert_eq!(s.flags(), 0);
        // A signalling NaN is invalid and gives the canonical NaN.
        assert_eq!(fcvt_h_s(S_SNAN, &mut s), NANBOX_H | 0x7e00);
        assert_eq!(riscv_flags(s.flags()), FPEXC_NV);
        // 65520 overflows half precision; 1 + 2^-11 is a tie that rounds to even.
        let mut s = rne();
        assert_eq!(fcvt_h_s(NANBOX_S | 0x477f_f000, &mut s), NANBOX_H | 0x7c00);
        assert_eq!(riscv_flags(s.flags()), FPEXC_OF | FPEXC_NX);
        let mut s = rne();
        assert_eq!(fcvt_h_d(0x3ff0_0200_0000_0000, &mut s), NANBOX_H | 0x3c00);
        assert_eq!(riscv_flags(s.flags()), FPEXC_NX);
        let mut s = status_rm(RISCV_FRM_RUP);
        assert_eq!(fcvt_h_d(0x3ff0_0200_0000_0000, &mut s), NANBOX_H | 0x3c01);
    }

    #[test]
    fn helper_signatures() {
        for d in ALL {
            assert!(
                d.name.starts_with('f') || d.name.starts_with("set_rounding_mode"),
                "{}",
                d.name
            );
            let env = d.args.first().copied() == Some(Ptr);
            assert_eq!(env, d.name != "fclass_d", "{}", d.name);
        }
    }
}
