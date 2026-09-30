// SPDX-License-Identifier: GPL-2.0-or-later

//! The x87 80 bit extended format, QEMU's `floatx80`.
//!
//! Every operation first checks `floatx80_invalid_encoding` and returns the default NaN
//! after raising invalid for pseudo NaNs, pseudo infinities and unnormals that the
//! [`FloatX80Behaviour`](crate::FloatX80Behaviour) does not allow. Results are rounded to the
//! [`FloatX80RoundPrec`](crate::FloatX80RoundPrec) in the status.

use crate::frac::shift128_left;
use crate::pack::{
    floatx80_invalid_encoding, floatx80_round_pack, floatx80_unpack_canonical, pack_floatx80,
};
use crate::parts::{FLOATX80_PARAMS, Parts128, default_nan_frac64, frac_msb_is_snan};
use crate::status::{
    FloatRelation, FloatStatus, FloatX80Behaviour, FloatX80RoundPrec, RoundMode, SnanRule, flags,
};

/// The x87 extended format, QEMU's `floatx80`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FloatX80 {
    /// The sign and 15 bit exponent.
    pub high: u16,
    /// The 64 bit significand, with its explicit integer bit at the top.
    pub low: u64,
}

impl FloatX80 {
    /// `make_floatx80`.
    pub const fn new(high: u16, low: u64) -> Self {
        FloatX80 { high, low }
    }

    /// `packFloatx80`: the exponent is added to the sign bit, as in QEMU.
    pub fn pack(sign: bool, exp: i32, sig: u64) -> Self {
        let (high, low) = pack_floatx80(sign, exp, sig);
        FloatX80 { high, low }
    }

    /// `floatx80_unpack_canonical`.
    #[inline]
    fn canon(self, s: &mut FloatStatus) -> Option<Parts128> {
        floatx80_unpack_canonical(self.high, self.low, s)
    }

    /// Unpack, substituting `parts128_default_nan` for an invalid encoding, as the integer
    /// conversions do.
    #[inline]
    fn canon_or_dnan(self, s: &mut FloatStatus) -> Parts128 {
        match self.canon(s) {
            Some(p) => p,
            None => Parts128::default_nan(s),
        }
    }

    /// Unpack for the conversions to other formats.
    #[inline]
    pub(crate) fn parts(self, s: &mut FloatStatus) -> Option<Parts128> {
        self.canon(s)
    }

    /// `floatx80_round_pack_canonical`.
    #[inline]
    pub(crate) fn from_parts(p: &mut Parts128, s: &mut FloatStatus) -> Self {
        let (high, low) = floatx80_round_pack(p, s);
        FloatX80 { high, low }
    }

    fn binop(
        self,
        b: Self,
        s: &mut FloatStatus,
        op: impl FnOnce(&Parts128, &Parts128, &mut FloatStatus) -> Parts128,
    ) -> Self {
        let Some(pa) = self.canon(s) else {
            return Self::default_nan(s);
        };
        let Some(pb) = b.canon(s) else {
            return Self::default_nan(s);
        };
        let mut r = op(&pa, &pb, s);
        Self::from_parts(&mut r, s)
    }

    /// `floatx80_add`.
    pub fn add(self, b: Self, s: &mut FloatStatus) -> Self {
        self.binop(b, s, |a, b, s| Parts128::addsub(a, b, s, false))
    }

    /// `floatx80_sub`.
    pub fn sub(self, b: Self, s: &mut FloatStatus) -> Self {
        self.binop(b, s, |a, b, s| Parts128::addsub(a, b, s, true))
    }

    /// `floatx80_mul`.
    pub fn mul(self, b: Self, s: &mut FloatStatus) -> Self {
        self.binop(b, s, Parts128::mul)
    }

    /// `floatx80_div`.
    pub fn div(self, b: Self, s: &mut FloatStatus) -> Self {
        self.binop(b, s, Parts128::div)
    }

    /// `floatx80_modrem`: with `modulo` false the IEEE remainder, with it true the
    /// remainder of the quotient truncated toward zero, whose low 64 bits are returned too.
    pub fn modrem(self, b: Self, modulo: bool, s: &mut FloatStatus) -> (Self, u64) {
        let mut quotient = 0u64;
        let Some(mut pa) = self.canon(s) else {
            return (Self::default_nan(s), quotient);
        };
        let Some(pb) = b.canon(s) else {
            return (Self::default_nan(s), quotient);
        };
        Parts128::modrem(&mut pa, &pb, if modulo { Some(&mut quotient) } else { None }, s);
        (Self::from_parts(&mut pa, s), quotient)
    }

    /// `floatx80_rem`: the IEEE remainder.
    pub fn rem(self, b: Self, s: &mut FloatStatus) -> Self {
        self.modrem(b, false, s).0
    }

    /// `floatx80_mod`: the remainder of the truncated quotient.
    pub fn modulo(self, b: Self, s: &mut FloatStatus) -> Self {
        self.modrem(b, true, s).0
    }

    /// `floatx80_sqrt`.
    pub fn sqrt(self, s: &mut FloatStatus) -> Self {
        let Some(mut p) = self.canon(s) else {
            return Self::default_nan(s);
        };
        let fmt = FLOATX80_PARAMS[s.floatx80_rounding_precision as usize];
        p.sqrt(s, &fmt);
        Self::from_parts(&mut p, s)
    }

    /// `floatx80_scalbn`.
    pub fn scalbn(self, n: i32, s: &mut FloatStatus) -> Self {
        let Some(p) = self.canon(s) else {
            return Self::default_nan(s);
        };
        let mut r = p.scalbn(n, s);
        Self::from_parts(&mut r, s)
    }

    /// `floatx80_round_to_int`.
    pub fn round_to_int(self, s: &mut FloatStatus) -> Self {
        let Some(p) = self.canon(s) else {
            return Self::default_nan(s);
        };
        let fmt = FLOATX80_PARAMS[s.floatx80_rounding_precision as usize];
        let rm = s.rounding_mode;
        let mut r = p.round_to_int(rm, 0, s, &fmt);
        Self::from_parts(&mut r, s)
    }

    /// `floatx80_round`: round to the precision in the status.
    pub fn round(self, s: &mut FloatStatus) -> Self {
        let Some(mut p) = self.canon(s) else {
            return Self::default_nan(s);
        };
        Self::from_parts(&mut p, s)
    }

    fn do_compare(self, b: Self, s: &mut FloatStatus, quiet: bool) -> FloatRelation {
        let Some(pa) = self.canon(s) else {
            return FloatRelation::Unordered;
        };
        let Some(pb) = b.canon(s) else {
            return FloatRelation::Unordered;
        };
        Parts128::compare(&pa, &pb, s, quiet)
    }

    /// `floatx80_compare`.
    pub fn compare(self, b: Self, s: &mut FloatStatus) -> FloatRelation {
        self.do_compare(b, s, false)
    }

    /// `floatx80_compare_quiet`.
    pub fn compare_quiet(self, b: Self, s: &mut FloatStatus) -> FloatRelation {
        self.do_compare(b, s, true)
    }

    compare_helpers!("floatx80");
    int_conversions!("floatx80", Parts128);

    /// `propagateFloatx80NaN`: pick the NaN result of an operation on `self` and `b`.
    pub fn propagate_nan(self, b: Self, s: &mut FloatStatus) -> Self {
        self.binop(b, s, Parts128::pick_nan)
    }

    /// `floatx80_default_nan`.
    pub fn default_nan(s: &FloatStatus) -> Self {
        let (sign, frac) = default_nan_frac64(s);
        FloatX80 { high: 0x7fff | (u16::from(sign) << 15), low: (1 << 63) | frac }
    }

    /// `floatx80_default_inf`.
    pub fn default_inf(sign: bool, s: &FloatStatus) -> Self {
        let z = s.floatx80_behaviour.0 & FloatX80Behaviour::DEFAULT_INF_INT_BIT_IS_ZERO != 0;
        Self::pack(sign, 0x7fff, if z { 0 } else { 1 << 63 })
    }

    /// `floatx80_silence_nan`. Like QEMU, only the bit is zero snan rule is supported.
    pub fn silence_nan(self, s: &FloatStatus) -> Self {
        assert!(s.snan_rule == SnanRule::BitIsZero);
        FloatX80 { high: self.high, low: self.low | 0xc000_0000_0000_0000 }
    }

    /// `floatx80_invalid_encoding`.
    pub fn invalid_encoding(self, s: &FloatStatus) -> bool {
        floatx80_invalid_encoding(self.high, self.low, s)
    }

    /// `floatx80_is_any_nan`.
    pub const fn is_any_nan(self) -> bool {
        self.high & 0x7fff == 0x7fff && (self.low << 1) != 0
    }
    /// `floatx80_is_signaling_nan`.
    pub fn is_signaling_nan(self, s: &FloatStatus) -> bool {
        self.is_any_nan() && frac_msb_is_snan((self.low >> 62) & 1 != 0, s)
    }
    /// `floatx80_is_quiet_nan`.
    pub fn is_quiet_nan(self, s: &FloatStatus) -> bool {
        self.is_any_nan() && !frac_msb_is_snan((self.low >> 62) & 1 != 0, s)
    }
    /// `floatx80_is_infinity`: whether a clear integer bit is allowed depends on the status.
    pub fn is_infinity(self, s: &FloatStatus) -> bool {
        let intbit = self.low >> 63 != 0;
        if !intbit && s.floatx80_behaviour.0 & FloatX80Behaviour::PSEUDO_INF_VALID == 0 {
            return false;
        }
        self.high & 0x7fff == 0x7fff && (self.low << 1) == 0
    }
    /// `floatx80_is_neg`.
    pub const fn is_neg(self) -> bool {
        self.high >> 15 != 0
    }
    /// `floatx80_is_zero`.
    pub const fn is_zero(self) -> bool {
        self.high & 0x7fff == 0 && self.low == 0
    }
    /// `floatx80_is_zero_or_denormal`.
    pub const fn is_zero_or_denormal(self) -> bool {
        self.high & 0x7fff == 0
    }
    /// `floatx80_abs`.
    pub const fn abs(self) -> Self {
        FloatX80 { high: self.high & 0x7fff, low: self.low }
    }
    /// `floatx80_chs`.
    pub const fn chs(self) -> Self {
        FloatX80 { high: self.high ^ 0x8000, low: self.low }
    }

    /// `roundAndPackFloatx80`: the legacy rounding helper the x87 and m68k helpers call
    /// directly. The significand `sig0:sig1` must be normalized, or `exp` must be 0. Only the
    /// nearest even, ties away, to zero, up and down rounding modes are supported, as in
    /// QEMU, which aborts on the others.
    pub fn round_and_pack(
        prec: FloatX80RoundPrec,
        sign: bool,
        mut exp: i32,
        mut sig0: u64,
        mut sig1: u64,
        s: &mut FloatStatus,
    ) -> Self {
        let rm = s.rounding_mode;
        let nearest_even = rm == RoundMode::NearestEven;

        if prec != FloatX80RoundPrec::X {
            let (mut inc, mut mask): (u64, u64) = match prec {
                FloatX80RoundPrec::D => (0x400, 0x7ff),
                _ => (0x80_0000_0000, 0xff_ffff_ffff),
            };
            sig0 |= u64::from(sig1 != 0);
            match rm {
                RoundMode::NearestEven | RoundMode::TiesAway => {}
                RoundMode::ToZero => inc = 0,
                RoundMode::Up => inc = if sign { 0 } else { mask },
                RoundMode::Down => inc = if sign { mask } else { 0 },
                _ => unsupported_rounding(rm),
            }
            let mut bits = sig0 & mask;
            if 0x7ffd <= (exp.wrapping_sub(1)) as u32 {
                if 0x7ffe < exp || (exp == 0x7ffe && sig0.wrapping_add(inc) < sig0) {
                    return Self::overflow(sign, mask, rm, s);
                }
                if exp <= 0 {
                    if s.flush_to_zero {
                        s.raise(flags::OUTPUT_DENORMAL_FLUSHED);
                        return Self::pack(sign, 0, 0);
                    }
                    let is_tiny =
                        s.tininess_before_rounding || exp < 0 || sig0 <= sig0.wrapping_add(inc);
                    sig0 = shift64_right_jamming(sig0, 1 - exp);
                    exp = 0;
                    bits = sig0 & mask;
                    if is_tiny && bits != 0 {
                        s.raise(flags::UNDERFLOW);
                    }
                    if bits != 0 {
                        s.raise(flags::INEXACT);
                    }
                    sig0 = sig0.wrapping_add(inc);
                    if (sig0 as i64) < 0 {
                        exp = 1;
                    }
                    let inc2 = mask.wrapping_add(1);
                    if nearest_even && bits << 1 == inc2 {
                        mask |= inc2;
                    }
                    sig0 &= !mask;
                    return Self::pack(sign, exp, sig0);
                }
            }
            if bits != 0 {
                s.raise(flags::INEXACT);
            }
            sig0 = sig0.wrapping_add(inc);
            if sig0 < inc {
                exp += 1;
                sig0 = 1 << 63;
            }
            let inc2 = mask.wrapping_add(1);
            if nearest_even && bits << 1 == inc2 {
                mask |= inc2;
            }
            sig0 &= !mask;
            if sig0 == 0 {
                exp = 0;
            }
            return Self::pack(sign, exp, sig0);
        }

        let increment_for = |sig1: u64| -> bool {
            match rm {
                RoundMode::NearestEven | RoundMode::TiesAway => (sig1 as i64) < 0,
                RoundMode::ToZero => false,
                RoundMode::Up => !sign && sig1 != 0,
                RoundMode::Down => sign && sig1 != 0,
                _ => unsupported_rounding(rm),
            }
        };
        let mut increment = increment_for(sig1);
        if 0x7ffd <= (exp.wrapping_sub(1)) as u32 {
            if 0x7ffe < exp || (exp == 0x7ffe && sig0 == u64::MAX && increment) {
                return Self::overflow(sign, 0, rm, s);
            }
            if exp <= 0 {
                let is_tiny =
                    s.tininess_before_rounding || exp < 0 || !increment || sig0 < u64::MAX;
                (sig0, sig1) = shift64_extra_right_jamming(sig0, sig1, 1 - exp);
                exp = 0;
                if is_tiny && sig1 != 0 {
                    s.raise(flags::UNDERFLOW);
                }
                if sig1 != 0 {
                    s.raise(flags::INEXACT);
                }
                increment = increment_for(sig1);
                if increment {
                    sig0 = sig0.wrapping_add(1);
                    if (sig1 << 1) == 0 && nearest_even {
                        sig0 &= !1;
                    }
                    if (sig0 as i64) < 0 {
                        exp = 1;
                    }
                }
                return Self::pack(sign, exp, sig0);
            }
        }
        if sig1 != 0 {
            s.raise(flags::INEXACT);
        }
        if increment {
            sig0 = sig0.wrapping_add(1);
            if sig0 == 0 {
                exp += 1;
                sig0 = 1 << 63;
            } else if (sig1 << 1) == 0 && nearest_even {
                sig0 &= !1;
            }
        } else if sig0 == 0 {
            exp = 0;
        }
        Self::pack(sign, exp, sig0)
    }

    /// The overflow exit of `roundAndPackFloatx80`.
    fn overflow(sign: bool, round_mask: u64, rm: RoundMode, s: &mut FloatStatus) -> Self {
        s.raise(flags::OVERFLOW | flags::INEXACT);
        if rm == RoundMode::ToZero
            || (sign && rm == RoundMode::Up)
            || (!sign && rm == RoundMode::Down)
        {
            return Self::pack(sign, 0x7ffe, !round_mask);
        }
        Self::default_inf(sign, s)
    }

    /// `normalizeRoundAndPackFloatx80`: like [`FloatX80::round_and_pack`] for a significand
    /// that need not be normalized.
    pub fn normalize_round_and_pack(
        prec: FloatX80RoundPrec,
        sign: bool,
        mut exp: i32,
        mut sig0: u64,
        mut sig1: u64,
        s: &mut FloatStatus,
    ) -> Self {
        if sig0 == 0 {
            sig0 = sig1;
            sig1 = 0;
            exp -= 64;
        }
        let shift = sig0.leading_zeros();
        if shift != 0 {
            (sig0, sig1) = shift128_left(sig0, sig1, shift);
        }
        exp -= shift as i32;
        Self::round_and_pack(prec, sign, exp, sig0, sig1, s)
    }
}

/// `roundAndPackFloatx80` aborts on the rounding modes it does not know.
fn unsupported_rounding(rm: RoundMode) -> ! {
    panic!("roundAndPackFloatx80 with rounding mode {rm:?}")
}

/// `shift64RightJamming`.
fn shift64_right_jamming(a: u64, count: i32) -> u64 {
    if count == 0 {
        a
    } else if count < 64 {
        (a >> count) | u64::from(a << ((-count) & 63) != 0)
    } else {
        u64::from(a != 0)
    }
}

/// `shift64ExtraRightJamming`.
fn shift64_extra_right_jamming(a0: u64, a1: u64, count: i32) -> (u64, u64) {
    let neg = (-count) & 63;
    if count == 0 {
        (a0, a1)
    } else if count < 64 {
        (a0 >> count, (a0 << neg) | u64::from(a1 != 0))
    } else if count == 64 {
        (0, a0 | u64::from(a1 != 0))
    } else {
        (0, u64::from((a0 | a1) != 0))
    }
}
