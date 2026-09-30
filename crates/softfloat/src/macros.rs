// SPDX-License-Identifier: GPL-2.0-or-later

//! Macros that stamp out the per-format wrappers QEMU writes out by hand for every format.
//!
//! Every format type has two private helpers, `canon`, which unpacks and canonicalizes (the
//! `floatN_unpack_canonical` functions), and `from_parts`, which rounds and packs (the
//! `floatN_round_pack_canonical` functions). The macros build the public API on top of those.

/// Float to integer conversions: `floatN_to_intM_scalbn`, `floatN_to_intM` and
/// `floatN_to_intM_round_to_zero`.
macro_rules! to_sint {
    ($q:literal, $t:ty, $qn:literal, $sc:ident, $plain:ident, $rz:ident) => {
        #[doc = concat!("`", $q, "_to_", $qn, "_scalbn`: convert to a signed integer after ")]
        #[doc = "scaling by 2 to the power `scale`, rounding with `rmode`. Out of range values "]
        #[doc = "and NaNs raise invalid and saturate."]
        pub fn $sc(self, rmode: RoundMode, scale: i32, s: &mut FloatStatus) -> $t {
            let mut p = self.canon_or_dnan(s);
            p.float_to_sint(rmode, scale, <$t>::MIN.into(), <$t>::MAX.into(), s) as $t
        }
        #[doc = concat!("`", $q, "_to_", $qn, "`: convert with the current rounding mode.")]
        pub fn $plain(self, s: &mut FloatStatus) -> $t {
            let rm = s.rounding_mode;
            self.$sc(rm, 0, s)
        }
        #[doc = concat!("`", $q, "_to_", $qn, "_round_to_zero`: convert, truncating.")]
        pub fn $rz(self, s: &mut FloatStatus) -> $t {
            self.$sc(RoundMode::ToZero, 0, s)
        }
    };
}

/// Float to unsigned integer conversions: `floatN_to_uintM_scalbn`, `floatN_to_uintM` and
/// `floatN_to_uintM_round_to_zero`.
macro_rules! to_uint {
    ($q:literal, $t:ty, $qn:literal, $sc:ident, $plain:ident, $rz:ident) => {
        #[doc = concat!("`", $q, "_to_", $qn, "_scalbn`: convert to an unsigned integer after ")]
        #[doc = "scaling by 2 to the power `scale`, rounding with `rmode`. Out of range values "]
        #[doc = "and NaNs raise invalid and saturate."]
        pub fn $sc(self, rmode: RoundMode, scale: i32, s: &mut FloatStatus) -> $t {
            let mut p = self.canon_or_dnan(s);
            p.float_to_uint(rmode, scale, <$t>::MAX.into(), s) as $t
        }
        #[doc = concat!("`", $q, "_to_", $qn, "`: convert with the current rounding mode.")]
        pub fn $plain(self, s: &mut FloatStatus) -> $t {
            let rm = s.rounding_mode;
            self.$sc(rm, 0, s)
        }
        #[doc = concat!("`", $q, "_to_", $qn, "_round_to_zero`: convert, truncating.")]
        pub fn $rz(self, s: &mut FloatStatus) -> $t {
            self.$sc(RoundMode::ToZero, 0, s)
        }
    };
}

/// All the 8 to 64 bit integer conversions in both directions.
macro_rules! int_conversions {
    ($q:literal, $parts:ty) => {
        to_sint!($q, i8, "int8", to_i8_scalbn, to_i8, to_i8_round_to_zero);
        to_sint!($q, i16, "int16", to_i16_scalbn, to_i16, to_i16_round_to_zero);
        to_sint!($q, i32, "int32", to_i32_scalbn, to_i32, to_i32_round_to_zero);
        to_sint!($q, i64, "int64", to_i64_scalbn, to_i64, to_i64_round_to_zero);
        to_uint!($q, u8, "uint8", to_u8_scalbn, to_u8, to_u8_round_to_zero);
        to_uint!($q, u16, "uint16", to_u16_scalbn, to_u16, to_u16_round_to_zero);
        to_uint!($q, u32, "uint32", to_u32_scalbn, to_u32, to_u32_round_to_zero);
        to_uint!($q, u64, "uint64", to_u64_scalbn, to_u64, to_u64_round_to_zero);

        #[doc = concat!("`int64_to_", $q, "_scalbn`: convert `a` times 2 to the power `scale`.")]
        pub fn from_i64_scalbn(a: i64, scale: i32, s: &mut FloatStatus) -> Self {
            let mut p = <$parts>::sint_to_float(a, scale);
            Self::from_parts(&mut p, s)
        }
        #[doc = concat!("`int64_to_", $q, "`.")]
        pub fn from_i64(a: i64, s: &mut FloatStatus) -> Self {
            Self::from_i64_scalbn(a, 0, s)
        }
        #[doc = concat!("`int32_to_", $q, "`.")]
        pub fn from_i32(a: i32, s: &mut FloatStatus) -> Self {
            Self::from_i64_scalbn(a.into(), 0, s)
        }
        #[doc = concat!("`int16_to_", $q, "`.")]
        pub fn from_i16(a: i16, s: &mut FloatStatus) -> Self {
            Self::from_i64_scalbn(a.into(), 0, s)
        }
        #[doc = concat!("`int8_to_", $q, "`.")]
        pub fn from_i8(a: i8, s: &mut FloatStatus) -> Self {
            Self::from_i64_scalbn(a.into(), 0, s)
        }
        #[doc = concat!("`uint64_to_", $q, "_scalbn`: convert `a` times 2 to the power `scale`.")]
        pub fn from_u64_scalbn(a: u64, scale: i32, s: &mut FloatStatus) -> Self {
            let mut p = <$parts>::uint_to_float(a, scale);
            Self::from_parts(&mut p, s)
        }
        #[doc = concat!("`uint64_to_", $q, "`.")]
        pub fn from_u64(a: u64, s: &mut FloatStatus) -> Self {
            Self::from_u64_scalbn(a, 0, s)
        }
        #[doc = concat!("`uint32_to_", $q, "`.")]
        pub fn from_u32(a: u32, s: &mut FloatStatus) -> Self {
            Self::from_u64_scalbn(a.into(), 0, s)
        }
        #[doc = concat!("`uint16_to_", $q, "`.")]
        pub fn from_u16(a: u16, s: &mut FloatStatus) -> Self {
            Self::from_u64_scalbn(a.into(), 0, s)
        }
        #[doc = concat!("`uint8_to_", $q, "`.")]
        pub fn from_u8(a: u8, s: &mut FloatStatus) -> Self {
            Self::from_u64_scalbn(a.into(), 0, s)
        }
    };
}

/// The predicates built on compare: `floatN_eq`, `floatN_le`, `floatN_lt`,
/// `floatN_unordered` and their `_quiet` forms.
macro_rules! compare_helpers {
    ($q:literal) => {
        #[doc = concat!("`", $q, "_eq`: equal, signalling on any NaN.")]
        pub fn eq(self, b: Self, s: &mut FloatStatus) -> bool {
            self.compare(b, s) == FloatRelation::Equal
        }
        #[doc = concat!("`", $q, "_le`: less or equal, signalling on any NaN.")]
        pub fn le(self, b: Self, s: &mut FloatStatus) -> bool {
            matches!(self.compare(b, s), FloatRelation::Less | FloatRelation::Equal)
        }
        #[doc = concat!("`", $q, "_lt`: less, signalling on any NaN.")]
        pub fn lt(self, b: Self, s: &mut FloatStatus) -> bool {
            self.compare(b, s) == FloatRelation::Less
        }
        #[doc = concat!("`", $q, "_unordered`: either is a NaN, signalling on any NaN.")]
        pub fn unordered(self, b: Self, s: &mut FloatStatus) -> bool {
            self.compare(b, s) == FloatRelation::Unordered
        }
        #[doc = concat!("`", $q, "_eq_quiet`: equal, signalling only on signalling NaNs.")]
        pub fn eq_quiet(self, b: Self, s: &mut FloatStatus) -> bool {
            self.compare_quiet(b, s) == FloatRelation::Equal
        }
        #[doc = concat!("`", $q, "_le_quiet`.")]
        pub fn le_quiet(self, b: Self, s: &mut FloatStatus) -> bool {
            matches!(self.compare_quiet(b, s), FloatRelation::Less | FloatRelation::Equal)
        }
        #[doc = concat!("`", $q, "_lt_quiet`.")]
        pub fn lt_quiet(self, b: Self, s: &mut FloatStatus) -> bool {
            self.compare_quiet(b, s) == FloatRelation::Less
        }
        #[doc = concat!("`", $q, "_unordered_quiet`.")]
        pub fn unordered_quiet(self, b: Self, s: &mut FloatStatus) -> bool {
            self.compare_quiet(b, s) == FloatRelation::Unordered
        }
    };
}

/// The min and max family on top of `minmax`.
macro_rules! minmax_helpers {
    ($q:literal) => {
        #[doc = concat!("`", $q, "_min`: IEEE 754-2008 style minimum with NaN propagation.")]
        pub fn min(self, b: Self, s: &mut FloatStatus) -> Self {
            self.minmax(b, s, MINMAX_ISMIN)
        }
        #[doc = concat!("`", $q, "_max`.")]
        pub fn max(self, b: Self, s: &mut FloatStatus) -> Self {
            self.minmax(b, s, 0)
        }
        #[doc = concat!("`", $q, "_minnum`: IEEE 754-2008 minNum, a quiet NaN loses.")]
        pub fn minnum(self, b: Self, s: &mut FloatStatus) -> Self {
            self.minmax(b, s, MINMAX_ISMIN | MINMAX_ISNUM)
        }
        #[doc = concat!("`", $q, "_maxnum`.")]
        pub fn maxnum(self, b: Self, s: &mut FloatStatus) -> Self {
            self.minmax(b, s, MINMAX_ISNUM)
        }
        #[doc = concat!("`", $q, "_minnummag`: minNumMag.")]
        pub fn minnummag(self, b: Self, s: &mut FloatStatus) -> Self {
            self.minmax(b, s, MINMAX_ISMIN | MINMAX_ISNUM | MINMAX_ISMAG)
        }
        #[doc = concat!("`", $q, "_maxnummag`: maxNumMag.")]
        pub fn maxnummag(self, b: Self, s: &mut FloatStatus) -> Self {
            self.minmax(b, s, MINMAX_ISNUM | MINMAX_ISMAG)
        }
        #[doc = concat!("`", $q, "_minimum_number`: IEEE 754-2019 minimumNumber, any NaN loses.")]
        pub fn minimum_number(self, b: Self, s: &mut FloatStatus) -> Self {
            self.minmax(b, s, MINMAX_ISMIN | MINMAX_ISNUMBER)
        }
        #[doc = concat!("`", $q, "_maximum_number`: IEEE 754-2019 maximumNumber.")]
        pub fn maximum_number(self, b: Self, s: &mut FloatStatus) -> Self {
            self.minmax(b, s, MINMAX_ISNUMBER)
        }
    };
}

/// The arithmetic every format whose unpacking cannot fail shares: everything except
/// floatx80, which has to handle invalid encodings first.
macro_rules! common_ops {
    ($q:literal, $parts:ty, $fmt:expr) => {
        #[doc = concat!("`", $q, "_add`.")]
        pub fn add(self, b: Self, s: &mut FloatStatus) -> Self {
            let hf = Self::HARDFLOAT && self.hf_zon(s) && b.hf_zon(s);
            crate::parts::hardfloat_rebias(s, hf, |s| {
                let pa = self.canon(s);
                let pb = b.canon(s);
                let mut r = <$parts>::addsub(&pa, &pb, s, false);
                Self::from_parts(&mut r, s)
            })
        }
        #[doc = concat!("`", $q, "_sub`.")]
        pub fn sub(self, b: Self, s: &mut FloatStatus) -> Self {
            let hf = Self::HARDFLOAT && self.hf_zon(s) && b.hf_zon(s);
            crate::parts::hardfloat_rebias(s, hf, |s| {
                let pa = self.canon(s);
                let pb = b.canon(s);
                let mut r = <$parts>::addsub(&pa, &pb, s, true);
                Self::from_parts(&mut r, s)
            })
        }
        #[doc = concat!("`", $q, "_mul`.")]
        pub fn mul(self, b: Self, s: &mut FloatStatus) -> Self {
            let hf = Self::HARDFLOAT && self.hf_zon(s) && b.hf_zon(s);
            crate::parts::hardfloat_rebias(s, hf, |s| {
                let pa = self.canon(s);
                let pb = b.canon(s);
                let mut r = <$parts>::mul(&pa, &pb, s);
                Self::from_parts(&mut r, s)
            })
        }
        #[doc = concat!("`", $q, "_div`.")]
        pub fn div(self, b: Self, s: &mut FloatStatus) -> Self {
            let hf = Self::HARDFLOAT && self.hf_zon(s) && b.is_normal();
            crate::parts::hardfloat_rebias(s, hf, |s| {
                let pa = self.canon(s);
                let pb = b.canon(s);
                let mut r = <$parts>::div(&pa, &pb, s);
                Self::from_parts(&mut r, s)
            })
        }
        #[doc = concat!("`", $q, "_muladd_scalbn`: `(self * b + c) * 2**scale` with one ")]
        #[doc = "rounding. `flags` is a mask of the [`muladd`](crate::muladd) constants."]
        pub fn muladd_scalbn(
            self,
            b: Self,
            c: Self,
            scale: i32,
            fl: u32,
            s: &mut FloatStatus,
        ) -> Self {
            let pa = self.canon(s);
            let pb = b.canon(s);
            let pc = c.canon(s);
            let mut r = <$parts>::muladd(&pa, &pb, &pc, fl, s);
            if scale != 0 {
                r = r.scalbn(scale, s);
            }
            r.uncanon(s, &$fmt);
            if fl & crate::status::muladd::NEGATE_RESULT != 0 && !r.cls.is_nan() {
                r.sign = !r.sign;
            }
            Self::pack_parts(&r)
        }
        #[doc = concat!("`", $q, "_muladd`: `self * b + c` with one rounding.")]
        pub fn muladd(self, b: Self, c: Self, fl: u32, s: &mut FloatStatus) -> Self {
            use crate::status::muladd::SUPPRESS_ADD_PRODUCT_ZERO;
            let mut hf = Self::HARDFLOAT
                && crate::parts::hardfloat_active(s)
                && self.hf_zon(s)
                && b.hf_zon(s)
                && c.hf_zon(s);
            let mut fl = fl;
            if hf && fl & SUPPRESS_ADD_PRODUCT_ZERO != 0 {
                if Self::HARDFLOAT_FMA_CHECKS_SUPPRESS {
                    // float32_muladd leaves the fast path for this flag.
                    hf = false;
                } else {
                    // float64_muladd does not check the flag, so the host adds the zero
                    // product: the flag has no effect.
                    fl &= !SUPPRESS_ADD_PRODUCT_ZERO;
                }
            }
            crate::parts::hardfloat_rebias(s, hf, |s| self.muladd_scalbn(b, c, 0, fl, s))
        }
        /// Whether QEMU's hardfloat path would accept this input: zero or normal once
        /// flushed (`floatN_input_flush1` then `fN_is_zon`).
        #[inline]
        fn hf_zon(self, s: &FloatStatus) -> bool {
            self.is_zero()
                || self.is_normal()
                || (s.flush_inputs_to_zero && self.is_zero_or_denormal())
        }
        #[doc = concat!("`", $q, "_sqrt`.")]
        pub fn sqrt(self, s: &mut FloatStatus) -> Self {
            let mut p = self.canon(s);
            p.sqrt(s, &$fmt);
            Self::from_parts(&mut p, s)
        }
        #[doc = concat!("`", $q, "_scalbn`: multiply by 2 to the power `n`.")]
        pub fn scalbn(self, n: i32, s: &mut FloatStatus) -> Self {
            let p = self.canon(s);
            let mut r = p.scalbn(n, s);
            Self::from_parts(&mut r, s)
        }
        #[doc = concat!("`", $q, "_round_to_int`: round to an integral value in the current ")]
        #[doc = "rounding mode."]
        pub fn round_to_int(self, s: &mut FloatStatus) -> Self {
            let p = self.canon(s);
            let rm = s.rounding_mode;
            let mut r = p.round_to_int(rm, 0, s, &$fmt);
            Self::from_parts(&mut r, s)
        }
        #[doc = concat!("`", $q, "_minmax`: the shared min and max with `float_minmax_*` ")]
        #[doc = "flags, see [`minmax`](crate::minmax)."]
        pub fn minmax(self, b: Self, s: &mut FloatStatus, fl: u32) -> Self {
            let pa = self.canon(s);
            let pb = b.canon(s);
            let mut r = <$parts>::minmax(&pa, &pb, s, fl);
            Self::from_parts(&mut r, s)
        }
        #[doc = concat!("`", $q, "_compare`: signals invalid on any NaN.")]
        pub fn compare(self, b: Self, s: &mut FloatStatus) -> FloatRelation {
            let pa = self.canon(s);
            let pb = b.canon(s);
            <$parts>::compare(&pa, &pb, s, false)
        }
        #[doc = concat!("`", $q, "_compare_quiet`: signals invalid only on signalling NaNs.")]
        pub fn compare_quiet(self, b: Self, s: &mut FloatStatus) -> FloatRelation {
            let pa = self.canon(s);
            let pb = b.canon(s);
            <$parts>::compare(&pa, &pb, s, true)
        }
        /// The same as `canon`, since this format has no invalid encodings.
        #[inline]
        fn canon_or_dnan(self, s: &mut FloatStatus) -> $parts {
            self.canon(s)
        }
        minmax_helpers!($q);
        compare_helpers!($q);
        int_conversions!($q, $parts);
    };
}

/// A format of 16 to 64 bits that decomposes into [`Parts64`](crate::parts::Parts64).
macro_rules! small_format {
    ($t:ident, $raw:ty, $q:literal, $fmt:expr, $hard:expr, $hard_sup:expr) => {
        impl $t {
            const FMT: FloatFmt = $fmt;
            /// Whether QEMU has hardfloat paths for this format.
            const HARDFLOAT: bool = $hard;
            /// Whether the hardfloat muladd leaves the fast path for
            /// `float_muladd_suppress_add_product_zero`.
            const HARDFLOAT_FMA_CHECKS_SUPPRESS: bool = $hard_sup;
            const SIGN: $raw = 1 << (Self::FMT.frac_size + Self::FMT.exp_size);
            const EXP_MASK: $raw = ((1 << Self::FMT.exp_size) - 1) << Self::FMT.frac_size;

            /// Unpack and canonicalize.
            #[inline]
            fn canon(self, s: &mut FloatStatus) -> Parts64 {
                unpack_canonical64(u64::from(self.0), &Self::FMT, &Self::FMT, s)
            }

            /// Pack already rounded parts.
            #[inline]
            fn pack_parts(p: &Parts64) -> Self {
                $t(pack_raw64(p, &Self::FMT) as $raw)
            }

            /// Round and pack.
            #[inline]
            fn from_parts(p: &mut Parts64, s: &mut FloatStatus) -> Self {
                $t(round_pack64(p, s, &Self::FMT, &Self::FMT) as $raw)
            }

            common_ops!($q, Parts64, Self::FMT);

            #[doc = concat!("`", $q, "_is_any_nan`.")]
            pub const fn is_any_nan(self) -> bool {
                self.0 & !Self::SIGN > Self::EXP_MASK
            }
            #[doc = concat!("`", $q, "_is_infinity`.")]
            pub const fn is_infinity(self) -> bool {
                self.0 & !Self::SIGN == Self::EXP_MASK
            }
            #[doc = concat!("`", $q, "_is_zero`.")]
            pub const fn is_zero(self) -> bool {
                self.0 & !Self::SIGN == 0
            }
            #[doc = concat!("`", $q, "_is_neg`.")]
            pub const fn is_neg(self) -> bool {
                self.0 & Self::SIGN != 0
            }
            #[doc = concat!("`", $q, "_is_zero_or_denormal`.")]
            pub const fn is_zero_or_denormal(self) -> bool {
                self.0 & Self::EXP_MASK == 0
            }
            #[doc = concat!("`", $q, "_is_denormal`.")]
            pub const fn is_denormal(self) -> bool {
                self.is_zero_or_denormal() && !self.is_zero()
            }
            #[doc = concat!("`", $q, "_is_normal`.")]
            pub const fn is_normal(self) -> bool {
                let e = (self.0 & Self::EXP_MASK) >> Self::FMT.frac_size;
                let emax = Self::EXP_MASK >> Self::FMT.frac_size;
                ((e + 1) & emax) >= 2
            }
            #[doc = concat!("`", $q, "_is_zero_or_normal`.")]
            pub const fn is_zero_or_normal(self) -> bool {
                self.is_normal() || self.is_zero()
            }
            #[doc = concat!("`", $q, "_abs`.")]
            pub const fn abs(self) -> Self {
                $t(self.0 & !Self::SIGN)
            }
            #[doc = concat!("`", $q, "_chs`.")]
            pub const fn chs(self) -> Self {
                $t(self.0 ^ Self::SIGN)
            }
            #[doc = concat!("`", $q, "_set_sign`.")]
            pub const fn set_sign(self, sign: bool) -> Self {
                $t((self.0 & !Self::SIGN) | if sign { Self::SIGN } else { 0 })
            }
            #[doc = concat!("`", $q, "_is_signaling_nan`.")]
            pub fn is_signaling_nan(self, s: &FloatStatus) -> bool {
                self.is_any_nan()
                    && frac_msb_is_snan((self.0 >> (Self::FMT.frac_size - 1)) & 1 != 0, s)
            }
            #[doc = concat!("`", $q, "_is_quiet_nan`.")]
            pub fn is_quiet_nan(self, s: &FloatStatus) -> bool {
                self.is_any_nan()
                    && !frac_msb_is_snan((self.0 >> (Self::FMT.frac_size - 1)) & 1 != 0, s)
            }
            #[doc = concat!("`", $q, "_default_nan`.")]
            pub fn default_nan(s: &FloatStatus) -> Self {
                let mut p = Parts64::default_nan(s);
                p.frac.0 >>= Self::FMT.frac_shift;
                Self::pack_parts(&p)
            }
            #[doc = concat!("`", $q, "_silence_nan`: quiet a signalling NaN.")]
            pub fn silence_nan(self, s: &FloatStatus) -> Self {
                let mut p = unpack_raw64(&Self::FMT, u64::from(self.0));
                p.frac.0 = parts_silence_nan_frac(p.frac.0 << Self::FMT.frac_shift, s)
                    >> Self::FMT.frac_shift;
                Self::pack_parts(&p)
            }
            #[doc = concat!("`", $q, "_squash_input_denormal`: with flush_inputs_to_zero, ")]
            #[doc = "replace a denormal with a zero of the same sign and raise input_denormal_flushed."]
            pub fn squash_input_denormal(self, s: &mut FloatStatus) -> Self {
                if s.flush_inputs_to_zero && self.is_denormal() {
                    s.raise(flags::INPUT_DENORMAL_FLUSHED);
                    return $t(self.0 & Self::SIGN);
                }
                self
            }
        }
    };
}
