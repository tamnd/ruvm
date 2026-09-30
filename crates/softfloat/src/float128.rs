// SPDX-License-Identifier: GPL-2.0-or-later

//! IEEE 754 binary128, QEMU's `float128`.

use crate::frac::{Frac, Frac128};
use crate::pack::{
    float128_pack_raw, float128_round_pack, float128_unpack_canonical, float128_unpack_raw,
};
use crate::parts::{
    FLOAT128_PARAMS, FloatClass, MINMAX_ISMAG, MINMAX_ISMIN, MINMAX_ISNUM, MINMAX_ISNUMBER, Parts,
    Parts128, frac_msb_is_snan, parts_silence_nan_frac,
};
use crate::status::{FloatRelation, FloatStatus, RoundMode, flags};

/// IEEE 754 binary128, QEMU's `float128`, as its high and low 64 bit halves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Float128 {
    /// The sign, exponent and top 48 fraction bits.
    pub high: u64,
    /// The low 64 fraction bits.
    pub low: u64,
}

impl Float128 {
    const SIGN: u64 = 1 << 63;
    /// QEMU has no hardfloat paths for float128.
    const HARDFLOAT: bool = false;
    const HARDFLOAT_FMA_CHECKS_SUPPRESS: bool = true;

    /// `make_float128`.
    pub const fn new(high: u64, low: u64) -> Self {
        Float128 { high, low }
    }

    /// The bits as one 128 bit integer.
    pub const fn to_bits(self) -> u128 {
        ((self.high as u128) << 64) | self.low as u128
    }

    /// Build from one 128 bit integer.
    pub const fn from_bits(v: u128) -> Self {
        Float128 { high: (v >> 64) as u64, low: v as u64 }
    }

    /// Unpack and canonicalize.
    #[inline]
    fn canon(self, s: &mut FloatStatus) -> Parts128 {
        float128_unpack_canonical(self.high, self.low, s)
    }

    /// Pack already rounded parts.
    #[inline]
    fn pack_parts(p: &Parts128) -> Self {
        let (high, low) = float128_pack_raw(p);
        Float128 { high, low }
    }

    /// Round and pack.
    #[inline]
    pub(crate) fn from_parts(p: &mut Parts128, s: &mut FloatStatus) -> Self {
        let (high, low) = float128_round_pack(p, s);
        Float128 { high, low }
    }

    /// Unpack for the conversions to other formats.
    #[inline]
    pub(crate) fn parts(self, s: &mut FloatStatus) -> Parts128 {
        self.canon(s)
    }

    common_ops!("float128", Parts128, FLOAT128_PARAMS);

    /// `float128_rem`: the IEEE remainder.
    pub fn rem(self, b: Self, s: &mut FloatStatus) -> Self {
        let mut pa = self.canon(s);
        let pb = b.canon(s);
        Parts128::modrem(&mut pa, &pb, None, s);
        Self::from_parts(&mut pa, s)
    }

    /// `float128_to_int128_scalbn`.
    pub fn to_i128_scalbn(self, rmode: RoundMode, scale: i32, s: &mut FloatStatus) -> i128 {
        let mut fl: u16 = 0;
        let mut p = self.canon(s);
        let r: i128 = match p.cls {
            FloatClass::SNan | FloatClass::QNan => {
                if p.cls == FloatClass::SNan {
                    fl |= flags::INVALID_SNAN;
                }
                fl |= flags::INVALID;
                -1
            }
            FloatClass::Inf => {
                fl = flags::INVALID | flags::INVALID_CVTI;
                if p.sign { i128::MIN } else { i128::MAX }
            }
            FloatClass::Zero => return 0,
            FloatClass::Normal | FloatClass::Denormal => {
                if p.round_to_int_normal(rmode, scale, 128 - 2) {
                    fl = flags::INEXACT;
                }
                if p.exp < 127 {
                    let v =
                        ((u128::from(p.frac.hi) << 64) | u128::from(p.frac.lo)) >> (127 - p.exp);
                    let v = v as i128;
                    if p.sign { v.wrapping_neg() } else { v }
                } else if p.exp == 127 && p.sign && p.frac.lo == 0 && p.frac.hi == 1 << 63 {
                    i128::MIN
                } else {
                    fl = flags::INVALID | flags::INVALID_CVTI;
                    if p.sign { i128::MIN } else { i128::MAX }
                }
            }
            FloatClass::Unclassified => unreachable!(),
        };
        s.raise(fl);
        r
    }

    /// `float128_to_int128`.
    pub fn to_i128(self, s: &mut FloatStatus) -> i128 {
        let rm = s.rounding_mode;
        self.to_i128_scalbn(rm, 0, s)
    }

    /// `float128_to_int128_round_to_zero`.
    pub fn to_i128_round_to_zero(self, s: &mut FloatStatus) -> i128 {
        self.to_i128_scalbn(RoundMode::ToZero, 0, s)
    }

    /// `float128_to_uint128_scalbn`.
    pub fn to_u128_scalbn(self, rmode: RoundMode, scale: i32, s: &mut FloatStatus) -> u128 {
        let mut fl: u16 = 0;
        let mut p = self.canon(s);
        let r: u128 = match p.cls {
            FloatClass::SNan | FloatClass::QNan => {
                if p.cls == FloatClass::SNan {
                    fl |= flags::INVALID_SNAN;
                }
                fl |= flags::INVALID;
                u128::MAX
            }
            FloatClass::Inf => {
                fl = flags::INVALID | flags::INVALID_CVTI;
                if p.sign { 0 } else { u128::MAX }
            }
            FloatClass::Zero => return 0,
            FloatClass::Normal | FloatClass::Denormal => {
                let inexact = p.round_to_int_normal(rmode, scale, 128 - 2);
                if inexact {
                    fl = flags::INEXACT;
                }
                if inexact && p.cls == FloatClass::Zero {
                    0
                } else if p.sign {
                    fl = flags::INVALID | flags::INVALID_CVTI;
                    0
                } else if p.exp <= 127 {
                    ((u128::from(p.frac.hi) << 64) | u128::from(p.frac.lo)) >> (127 - p.exp)
                } else {
                    fl = flags::INVALID | flags::INVALID_CVTI;
                    u128::MAX
                }
            }
            FloatClass::Unclassified => unreachable!(),
        };
        s.raise(fl);
        r
    }

    /// `float128_to_uint128`.
    pub fn to_u128(self, s: &mut FloatStatus) -> u128 {
        let rm = s.rounding_mode;
        self.to_u128_scalbn(rm, 0, s)
    }

    /// `float128_to_uint128_round_to_zero`.
    pub fn to_u128_round_to_zero(self, s: &mut FloatStatus) -> u128 {
        self.to_u128_scalbn(RoundMode::ToZero, 0, s)
    }

    /// `int128_to_float128`.
    pub fn from_i128(a: i128, s: &mut FloatStatus) -> Self {
        let mut p = Parts128::of_class(FloatClass::Zero, false);
        if a != 0 {
            p.cls = FloatClass::Normal;
            p.sign = a < 0;
            Self::set_u128(&mut p, a.unsigned_abs());
        }
        Self::from_parts(&mut p, s)
    }

    /// `uint128_to_float128`.
    pub fn from_u128(a: u128, s: &mut FloatStatus) -> Self {
        let mut p = Parts128::of_class(FloatClass::Zero, false);
        if a != 0 {
            p.cls = FloatClass::Normal;
            Self::set_u128(&mut p, a);
        }
        Self::from_parts(&mut p, s)
    }

    fn set_u128(p: &mut Parts128, a: u128) {
        let shift = a.leading_zeros() as i32;
        p.exp = 127 - shift;
        let a = a << shift;
        p.frac = Frac128 { hi: (a >> 64) as u64, lo: a as u64 };
    }

    /// `float128_is_any_nan`.
    pub const fn is_any_nan(self) -> bool {
        (self.high >> 48) & 0x7fff == 0x7fff && (self.high & 0xffff_ffff_ffff | self.low) != 0
    }
    /// `float128_is_infinity`.
    pub const fn is_infinity(self) -> bool {
        (self.high & !Self::SIGN) == 0x7fff_0000_0000_0000 && self.low == 0
    }
    /// `float128_is_zero`.
    pub const fn is_zero(self) -> bool {
        (self.high & !Self::SIGN) | self.low == 0
    }
    /// `float128_is_neg`.
    pub const fn is_neg(self) -> bool {
        self.high >> 63 != 0
    }
    /// `float128_is_zero_or_denormal`.
    pub const fn is_zero_or_denormal(self) -> bool {
        self.high & 0x7fff_0000_0000_0000 == 0
    }
    /// `float128_is_denormal`.
    pub const fn is_denormal(self) -> bool {
        self.is_zero_or_denormal() && !self.is_zero()
    }
    /// `float128_is_normal`.
    pub const fn is_normal(self) -> bool {
        (((self.high >> 48) + 1) & 0x7fff) >= 2
    }
    /// `float128_abs`.
    pub const fn abs(self) -> Self {
        Float128 { high: self.high & !Self::SIGN, low: self.low }
    }
    /// `float128_chs`.
    pub const fn chs(self) -> Self {
        Float128 { high: self.high ^ Self::SIGN, low: self.low }
    }
    /// `float128_is_signaling_nan`.
    pub fn is_signaling_nan(self, s: &FloatStatus) -> bool {
        self.is_any_nan() && frac_msb_is_snan((self.high >> 47) & 1 != 0, s)
    }
    /// `float128_is_quiet_nan`.
    pub fn is_quiet_nan(self, s: &FloatStatus) -> bool {
        self.is_any_nan() && !frac_msb_is_snan((self.high >> 47) & 1 != 0, s)
    }
    /// `float128_default_nan`.
    pub fn default_nan(s: &FloatStatus) -> Self {
        let mut p = Parts128::default_nan(s);
        p.frac.shr(FLOAT128_PARAMS.frac_shift);
        Self::pack_parts(&p)
    }
    /// `float128_silence_nan`.
    pub fn silence_nan(self, s: &FloatStatus) -> Self {
        let mut p: Parts<Frac128> = float128_unpack_raw(self.high, self.low);
        p.frac.shl(FLOAT128_PARAMS.frac_shift);
        p.frac.hi = parts_silence_nan_frac(p.frac.hi, s);
        p.frac.shr(FLOAT128_PARAMS.frac_shift);
        Self::pack_parts(&p)
    }
}
