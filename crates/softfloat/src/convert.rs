// SPDX-License-Identifier: GPL-2.0-or-later

//! Conversions between the float formats, the "Float to Float conversions" section of
//! `fpu/softfloat.c`.

use crate::float128::Float128;
use crate::floatx80::FloatX80;
use crate::formats::{BFloat16, Float16, Float32, Float64};
use crate::pack::{
    pack_raw64, parts_float_to_ahp, parts_float_to_float, parts64_to_parts128, parts128_to_parts64,
    unpack_canonical64,
};
use crate::parts::{
    BFLOAT16_PARAMS, FLOAT16_PARAMS, FLOAT16_PARAMS_AHP, FLOAT32_PARAMS, FLOAT64_PARAMS, Parts64,
    Parts128,
};
use crate::status::FloatStatus;

/// `float16a_unpack_canonical` with the IEEE or the Arm alternative layout.
fn f16_parts(a: Float16, ieee: bool, s: &mut FloatStatus) -> Parts64 {
    let fmt = if ieee { &FLOAT16_PARAMS } else { &FLOAT16_PARAMS_AHP };
    unpack_canonical64(u64::from(a.0), &FLOAT16_PARAMS, fmt, s)
}

/// `float32_to_float16` and `float64_to_float16`: the shared tail.
fn to_f16(mut p: Parts64, ieee: bool, s: &mut FloatStatus) -> Float16 {
    let fmt = if ieee {
        parts_float_to_float(&mut p, s);
        &FLOAT16_PARAMS
    } else {
        parts_float_to_ahp(&mut p, s);
        &FLOAT16_PARAMS_AHP
    };
    p.uncanon(s, fmt);
    Float16(pack_raw64(&p, &FLOAT16_PARAMS) as u16)
}

fn p64(raw: u64, fmt: &crate::parts::FloatFmt, s: &mut FloatStatus) -> Parts64 {
    unpack_canonical64(raw, fmt, fmt, s)
}

fn pack64(mut p: Parts64, fmt: &crate::parts::FloatFmt, s: &mut FloatStatus) -> u64 {
    crate::pack::round_pack64(&mut p, s, fmt, fmt)
}

/// `parts128_to_parts64` of an unpacked floatx80, or `parts64_default_nan` for an invalid
/// encoding.
fn x80_to_p64(a: FloatX80, s: &mut FloatStatus) -> Parts64 {
    match a.parts(s) {
        Some(p) => parts128_to_parts64(&p, s),
        None => Parts64::default_nan(s),
    }
}

impl Float16 {
    /// `float16_to_float32`: `ieee` false reads the Arm alternative half precision format.
    pub fn to_float32(self, ieee: bool, s: &mut FloatStatus) -> Float32 {
        let mut p = f16_parts(self, ieee, s);
        parts_float_to_float(&mut p, s);
        Float32(pack64(p, &FLOAT32_PARAMS, s) as u32)
    }

    /// `float16_to_float64`: `ieee` false reads the Arm alternative half precision format.
    pub fn to_float64(self, ieee: bool, s: &mut FloatStatus) -> Float64 {
        let mut p = f16_parts(self, ieee, s);
        parts_float_to_float(&mut p, s);
        Float64(pack64(p, &FLOAT64_PARAMS, s))
    }
}

impl BFloat16 {
    /// `bfloat16_to_float32`.
    pub fn to_float32(self, s: &mut FloatStatus) -> Float32 {
        let mut p = p64(u64::from(self.0), &BFLOAT16_PARAMS, s);
        parts_float_to_float(&mut p, s);
        Float32(pack64(p, &FLOAT32_PARAMS, s) as u32)
    }

    /// `bfloat16_to_float64`.
    pub fn to_float64(self, s: &mut FloatStatus) -> Float64 {
        let mut p = p64(u64::from(self.0), &BFLOAT16_PARAMS, s);
        parts_float_to_float(&mut p, s);
        Float64(pack64(p, &FLOAT64_PARAMS, s))
    }
}

impl Float32 {
    /// `float32_to_float16`: `ieee` false produces the Arm alternative half precision
    /// format, which has no infinities or NaNs.
    pub fn to_float16(self, ieee: bool, s: &mut FloatStatus) -> Float16 {
        let p = p64(u64::from(self.0), &FLOAT32_PARAMS, s);
        to_f16(p, ieee, s)
    }

    /// `float32_to_bfloat16`.
    pub fn to_bfloat16(self, s: &mut FloatStatus) -> BFloat16 {
        let mut p = p64(u64::from(self.0), &FLOAT32_PARAMS, s);
        parts_float_to_float(&mut p, s);
        BFloat16(pack64(p, &BFLOAT16_PARAMS, s) as u16)
    }

    /// `float32_to_float64`.
    pub fn to_float64(self, s: &mut FloatStatus) -> Float64 {
        let mut p = p64(u64::from(self.0), &FLOAT32_PARAMS, s);
        parts_float_to_float(&mut p, s);
        Float64(pack64(p, &FLOAT64_PARAMS, s))
    }

    /// `float32_to_floatx80`.
    pub fn to_floatx80(self, s: &mut FloatStatus) -> FloatX80 {
        let p = p64(u64::from(self.0), &FLOAT32_PARAMS, s);
        let mut q = parts64_to_parts128(&p, s);
        FloatX80::from_parts(&mut q, s)
    }

    /// `float32_to_float128`.
    pub fn to_float128(self, s: &mut FloatStatus) -> Float128 {
        let p = p64(u64::from(self.0), &FLOAT32_PARAMS, s);
        let mut q = parts64_to_parts128(&p, s);
        Float128::from_parts(&mut q, s)
    }
}

impl Float64 {
    /// `float64_to_float16`: `ieee` false produces the Arm alternative half precision
    /// format.
    pub fn to_float16(self, ieee: bool, s: &mut FloatStatus) -> Float16 {
        let p = p64(self.0, &FLOAT64_PARAMS, s);
        to_f16(p, ieee, s)
    }

    /// `float64_to_bfloat16`.
    pub fn to_bfloat16(self, s: &mut FloatStatus) -> BFloat16 {
        let mut p = p64(self.0, &FLOAT64_PARAMS, s);
        parts_float_to_float(&mut p, s);
        BFloat16(pack64(p, &BFLOAT16_PARAMS, s) as u16)
    }

    /// `float64_to_float32`.
    pub fn to_float32(self, s: &mut FloatStatus) -> Float32 {
        let mut p = p64(self.0, &FLOAT64_PARAMS, s);
        parts_float_to_float(&mut p, s);
        Float32(pack64(p, &FLOAT32_PARAMS, s) as u32)
    }

    /// `float64_to_floatx80`.
    pub fn to_floatx80(self, s: &mut FloatStatus) -> FloatX80 {
        let p = p64(self.0, &FLOAT64_PARAMS, s);
        let mut q = parts64_to_parts128(&p, s);
        FloatX80::from_parts(&mut q, s)
    }

    /// `float64_to_float128`.
    pub fn to_float128(self, s: &mut FloatStatus) -> Float128 {
        let p = p64(self.0, &FLOAT64_PARAMS, s);
        let mut q = parts64_to_parts128(&p, s);
        Float128::from_parts(&mut q, s)
    }
}

impl FloatX80 {
    /// `floatx80_to_float32`.
    pub fn to_float32(self, s: &mut FloatStatus) -> Float32 {
        let p = x80_to_p64(self, s);
        Float32(pack64(p, &FLOAT32_PARAMS, s) as u32)
    }

    /// `floatx80_to_float64`.
    pub fn to_float64(self, s: &mut FloatStatus) -> Float64 {
        let p = x80_to_p64(self, s);
        Float64(pack64(p, &FLOAT64_PARAMS, s))
    }

    /// `floatx80_to_float128`.
    pub fn to_float128(self, s: &mut FloatStatus) -> Float128 {
        let mut p = match self.parts(s) {
            Some(mut p) => {
                parts_float_to_float(&mut p, s);
                p
            }
            None => Parts128::default_nan(s),
        };
        Float128::from_parts(&mut p, s)
    }
}

impl Float128 {
    /// `float128_to_float32`.
    pub fn to_float32(self, s: &mut FloatStatus) -> Float32 {
        let p = self.parts(s);
        let q = parts128_to_parts64(&p, s);
        Float32(pack64(q, &FLOAT32_PARAMS, s) as u32)
    }

    /// `float128_to_float64`.
    pub fn to_float64(self, s: &mut FloatStatus) -> Float64 {
        let p = self.parts(s);
        let q = parts128_to_parts64(&p, s);
        Float64(pack64(q, &FLOAT64_PARAMS, s))
    }

    /// `float128_to_floatx80`.
    pub fn to_floatx80(self, s: &mut FloatStatus) -> FloatX80 {
        let mut p = self.parts(s);
        parts_float_to_float(&mut p, s);
        FloatX80::from_parts(&mut p, s)
    }
}
