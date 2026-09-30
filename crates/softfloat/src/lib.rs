// SPDX-License-Identifier: GPL-2.0-or-later

//! A bit exact port of QEMU 11.1's softfloat (`fpu/softfloat.c`, `fpu/softfloat-parts.c.inc`,
//! `fpu/softfloat-specialize.c.inc`, `include/fpu/softfloat-macros.h` and
//! `include/fpu/softfloat*.h`).
//!
//! # Shape
//!
//! Each format is a plain wrapper around its bits: [`Float16`], [`BFloat16`], [`Float32`],
//! [`Float64`], [`FloatX80`] and [`Float128`]. Operations are methods that take a
//! [`FloatStatus`] by `&mut`, which carries the rounding mode, the accrued exception
//! [`flags`], and every target specific rule QEMU keeps in `float_status`. The doc comment of
//! each method names the QEMU function it ports, so `Float32::muladd` is `float32_muladd`.
//!
//! Internally every operation unpacks to QEMU's decomposed `FloatParts` form, a generic
//! `Parts` over a 64, 128 or 256 bit fraction, runs the shared `partsN_*`
//! algorithm and rounds back, exactly as QEMU does.
//!
//! The crate is `no_std`, has no dependencies, no `unsafe` and never allocates.
//!
//! # Differences from QEMU
//!
//! - QEMU's hardfloat fast paths (using the host FPU for float32 and float64 add, sub, mul,
//!   div, muladd and sqrt when the inexact flag is already set and the rounding mode is
//!   nearest even) do not use the host FPU here. They mostly give the same result and flags
//!   as the soft path, but not always, and the two cases where they differ are reproduced:
//!   on overflow the fast path ignores `rebias_overflow` (giving infinity and only the
//!   overflow flag), and `float64_muladd` ignores `float_muladd_suppress_add_product_zero`
//!   (adding the zero product to the addend). Both only happen when QEMU would take the fast
//!   path, that is with inexact already set, nearest even rounding and zero or normal
//!   inputs (after input flushing).
//! - The API is a superset in places: every format has the 8, 16, 32 and 64 bit signed and
//!   unsigned integer conversions and `_scalbn` forms, where QEMU only defines some of them
//!   (for example it has no `float32_to_int8` or `floatx80_to_uint32`). The extra ones use
//!   the same `partsN_float_to_sint` and `partsN_float_to_uint` code, so they behave as
//!   QEMU would if it had them. The same holds for `float16`/`bfloat16` `muladd_scalbn` and
//!   `floatx80` integer to float with a scale.
//! - QEMU's `floatx80_modrem` writes the quotient through a pointer; [`FloatX80::modrem`]
//!   returns it. `floatx80_mod` is [`FloatX80::modulo`].
//! - Where QEMU calls `g_assert_not_reached()` or `abort()` (an unset NaN rule, a zero
//!   default NaN pattern, `roundAndPackFloatx80` with a round to odd mode, silencing a NaN
//!   with `float_snan_never`) this crate panics.
//! - [`FloatStatus::x87`] fills in the fused multiply add NaN rules that QEMU leaves unset
//!   for the x87 status, so that calling those operations does not panic.
//! - Not ported: the float8 (e4m3, e5m2) and float4 formats, the PowerPC `float64r32`
//!   operations and `float32_exp2`.

#![no_std]
#![forbid(unsafe_code)]

#[macro_use]
mod macros;

mod convert;
mod float128;
mod floatx80;
mod formats;
mod frac;
mod pack;
mod parts;
mod status;

pub use float128::Float128;
pub use floatx80::FloatX80;
pub use formats::{BFloat16, Float16, Float32, Float64};
pub use status::{
    Float2NanPropRule, Float3NanPropRule, FloatRelation, FloatStatus, FloatX80Behaviour,
    FloatX80RoundPrec, InfZeroNanRule, RoundMode, SnanRule, flags, muladd,
};

/// The `float_minmax_*` flags for the `minmax` methods.
pub mod minmax {
    /// `float_minmax_ismin`: return the minimum rather than the maximum.
    pub const ISMIN: u32 = crate::parts::MINMAX_ISMIN;
    /// `float_minmax_isnum`: IEEE 754-2008 minNum and maxNum, a quiet NaN loses.
    pub const ISNUM: u32 = crate::parts::MINMAX_ISNUM;
    /// `float_minmax_ismag`: compare magnitudes first.
    pub const ISMAG: u32 = crate::parts::MINMAX_ISMAG;
    /// `float_minmax_isnumber`: IEEE 754-2019 minimumNumber and maximumNumber, any NaN loses.
    pub const ISNUMBER: u32 = crate::parts::MINMAX_ISNUMBER;
}
