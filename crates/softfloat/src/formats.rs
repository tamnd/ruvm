// SPDX-License-Identifier: GPL-2.0-or-later

//! The formats that fit in 64 bits: float16, bfloat16, float32 and float64.

use crate::pack::{pack_raw64, round_pack64, unpack_canonical64, unpack_raw64};
use crate::parts::{
    BFLOAT16_PARAMS, FLOAT16_PARAMS, FLOAT32_PARAMS, FLOAT64_PARAMS, FloatFmt, MINMAX_ISMAG,
    MINMAX_ISMIN, MINMAX_ISNUM, MINMAX_ISNUMBER, Parts64, frac_msb_is_snan, parts_silence_nan_frac,
};
use crate::status::{FloatRelation, FloatStatus, RoundMode, flags};

/// IEEE 754 binary16, QEMU's `float16`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Float16(pub u16);

/// The bfloat16 format, QEMU's `bfloat16`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BFloat16(pub u16);

/// IEEE 754 binary32, QEMU's `float32`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Float32(pub u32);

/// IEEE 754 binary64, QEMU's `float64`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Float64(pub u64);

small_format!(Float16, u16, "float16", FLOAT16_PARAMS, false, true);
small_format!(BFloat16, u16, "bfloat16", BFLOAT16_PARAMS, false, true);
small_format!(Float32, u32, "float32", FLOAT32_PARAMS, true, true);
small_format!(Float64, u64, "float64", FLOAT64_PARAMS, true, false);

impl Float32 {
    /// `float32_rem`: the IEEE remainder.
    pub fn rem(self, b: Self, s: &mut FloatStatus) -> Self {
        let mut pa = self.canon(s);
        let pb = b.canon(s);
        Parts64::modrem(&mut pa, &pb, None, s);
        Self::from_parts(&mut pa, s)
    }

    /// `float32_log2`.
    pub fn log2(self, s: &mut FloatStatus) -> Self {
        let mut p = self.canon(s);
        p.log2(s, &FLOAT32_PARAMS);
        Self::from_parts(&mut p, s)
    }
}

impl Float64 {
    /// `float64_rem`: the IEEE remainder.
    pub fn rem(self, b: Self, s: &mut FloatStatus) -> Self {
        let mut pa = self.canon(s);
        let pb = b.canon(s);
        Parts64::modrem(&mut pa, &pb, None, s);
        Self::from_parts(&mut pa, s)
    }

    /// `float64_log2`.
    pub fn log2(self, s: &mut FloatStatus) -> Self {
        let mut p = self.canon(s);
        p.log2(s, &FLOAT64_PARAMS);
        Self::from_parts(&mut p, s)
    }

    /// `float64_to_int32_modulo`: convert, wrapping modulo 2**32 instead of saturating, as
    /// the Arm FJCVTZS instruction does. Overflow still raises invalid.
    pub fn to_i32_modulo(self, rmode: RoundMode, s: &mut FloatStatus) -> i32 {
        let mut p = self.canon(s);
        p.float_to_sint_modulo(rmode, 31, s) as i32
    }

    /// `float64_to_int64_modulo`: like [`Float64::to_i32_modulo`] for 64 bits.
    pub fn to_i64_modulo(self, rmode: RoundMode, s: &mut FloatStatus) -> i64 {
        let mut p = self.canon(s);
        p.float_to_sint_modulo(rmode, 63, s)
    }
}
