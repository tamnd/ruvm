// SPDX-License-Identifier: GPL-2.0-or-later

//! The decomposed form every operation works on, a port of `fpu/softfloat-parts.c.inc`,
//! `fpu/softfloat-parts-addsub.c.inc` and the NaN parts of `fpu/softfloat-specialize.c.inc`.
//!
//! QEMU includes the parts file twice, once for 64 bit and once for 128 bit fractions, and
//! the add and subtract helpers a third time for 256 bits. Here that is one generic
//! [`Parts`] over the [`Frac`] size, so each `partsN_*` function exists once.

use crate::frac::{
    Frac, Frac64, Frac128, FracN, IMPLICIT_BIT, add128, mul64_to_128, mul128_to_256, shift128_left,
    shift128_right, sub128,
};
use crate::status::{
    Float2NanPropRule, Float3NanPropRule, FloatRelation, FloatStatus, FloatX80Behaviour,
    InfZeroNanRule, RoundMode, SnanRule, flags, muladd,
};

/// `DECOMPOSED_BINARY_POINT`.
pub(crate) const BINARY_POINT: i32 = 63;

/// `SCALBN_EXP_MAX`.
const SCALBN_EXP_MAX: i32 = 0x0fff_ffff;
/// `SCALBN_EXP_MIN`.
const SCALBN_EXP_MIN: i32 = -SCALBN_EXP_MAX;

/// `FloatClass`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FloatClass {
    Unclassified = 0,
    Zero = 1,
    Normal = 2,
    Denormal = 3,
    Inf = 4,
    QNan = 5,
    SNan = 6,
}

impl FloatClass {
    /// `float_cmask`.
    #[inline]
    pub(crate) fn cmask(self) -> u32 {
        1 << (self as u32)
    }
    /// `is_nan`.
    #[inline]
    pub(crate) fn is_nan(self) -> bool {
        matches!(self, FloatClass::QNan | FloatClass::SNan)
    }
}

pub(crate) const CMASK_ZERO: u32 = 1 << 1;
pub(crate) const CMASK_NORMAL: u32 = 1 << 2;
pub(crate) const CMASK_DENORMAL: u32 = 1 << 3;
pub(crate) const CMASK_INF: u32 = 1 << 4;
pub(crate) const CMASK_QNAN: u32 = 1 << 5;
pub(crate) const CMASK_SNAN: u32 = 1 << 6;
pub(crate) const CMASK_INFZERO: u32 = CMASK_ZERO | CMASK_INF;
pub(crate) const CMASK_ANYNAN: u32 = CMASK_QNAN | CMASK_SNAN;
pub(crate) const CMASK_ANYNORM: u32 = CMASK_NORMAL | CMASK_DENORMAL;

/// `cmask_is_only_normals`.
#[inline]
fn only_normals(m: u32) -> bool {
    m & !CMASK_ANYNORM == 0
}

/// `record_denormals_used`.
#[inline]
fn record_denormals_used(mask: u32, s: &mut FloatStatus) {
    if mask & CMASK_DENORMAL != 0 {
        s.raise(flags::INPUT_DENORMAL_USED);
    }
}

/// `FloatFmt`: the layout and rounding parameters of a format.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FloatFmt {
    pub exp_size: i32,
    pub exp_bias: i32,
    pub exp_re_bias: i32,
    pub exp_max: i32,
    pub frac_size: i32,
    pub frac_shift: i32,
    pub round_mask: u64,
    /// `exp_max_kind == float_expmax_normal`: the top exponent holds normal numbers.
    pub expmax_normal: bool,
    pub has_explicit_bit: bool,
    pub overflow_raises_invalid: bool,
}

/// `FLOAT_PARAMS(E, F)`.
const fn float_params(e: i32, f: i32) -> FloatFmt {
    let shift = (-f - 1) & 63;
    FloatFmt {
        exp_size: e,
        exp_bias: ((1 << e) - 1) >> 1,
        exp_re_bias: (1 << (e - 1)) + (1 << (e - 2)),
        exp_max: (1 << e) - 1,
        frac_size: f,
        frac_shift: shift,
        round_mask: (1u64 << shift) - 1,
        expmax_normal: false,
        has_explicit_bit: false,
        overflow_raises_invalid: false,
    }
}

/// `FLOATX80_PARAMS(R)`.
const fn floatx80_params(r: i32) -> FloatFmt {
    let mut p = float_params(15, 1);
    p.frac_size = if r == 64 { 63 } else { r };
    p.frac_shift = 0;
    p.round_mask = if r == 64 { u64::MAX } else { (1u64 << ((-r - 1) & 63)) - 1 };
    p
}

/// `float16_params`.
pub(crate) const FLOAT16_PARAMS: FloatFmt = float_params(5, 10);
/// `float16_params_ahp`: the Arm alternative half precision layout.
pub(crate) const FLOAT16_PARAMS_AHP: FloatFmt = {
    let mut p = float_params(5, 10);
    p.expmax_normal = true;
    p.overflow_raises_invalid = true;
    p
};
/// `bfloat16_params`.
pub(crate) const BFLOAT16_PARAMS: FloatFmt = float_params(8, 7);
/// `float32_params`.
pub(crate) const FLOAT32_PARAMS: FloatFmt = float_params(8, 23);
/// `float64_params`.
pub(crate) const FLOAT64_PARAMS: FloatFmt = float_params(11, 52);
/// `float128_params`.
pub(crate) const FLOAT128_PARAMS: FloatFmt = float_params(15, 112);
/// `floatx80_params`, indexed by `FloatX80RoundPrec`.
pub(crate) const FLOATX80_PARAMS: [FloatFmt; 3] = [
    {
        let mut p = floatx80_params(64);
        p.has_explicit_bit = true;
        p
    },
    floatx80_params(52),
    floatx80_params(23),
];

/// QEMU's `can_use_fpu`: the hardfloat paths are only taken when inexact is already set and
/// the rounding mode is nearest even.
#[inline]
pub(crate) fn hardfloat_active(s: &FloatStatus) -> bool {
    s.exception_flags & flags::INEXACT != 0 && s.rounding_mode == RoundMode::NearestEven
}

/// Reproduce the visible effect of QEMU's hardfloat fast paths.
///
/// QEMU runs float32 and float64 add, sub, mul, div and muladd on the host FPU when the
/// inexact flag is already set, the rounding mode is nearest even and the inputs are zero or
/// normal (`can_use_fpu` and the `pre` checks). Tiny results go back to the soft path, and in
/// every other case the host gives the same result and flags as softfloat, with one
/// exception: on overflow the host returns infinity and ignores `rebias_overflow`. `eligible`
/// says the inputs pass the checks; `f` runs the soft operation, here with `rebias_overflow`
/// cleared when the hardfloat path would have been taken.
#[inline]
pub(crate) fn hardfloat_rebias<R>(
    s: &mut FloatStatus,
    eligible: bool,
    f: impl FnOnce(&mut FloatStatus) -> R,
) -> R {
    if eligible && s.rebias_overflow && hardfloat_active(s) {
        s.rebias_overflow = false;
        let r = f(s);
        s.rebias_overflow = true;
        r
    } else {
        f(s)
    }
}

/// `exp_scalbn`.
pub(crate) fn exp_scalbn(exp: i32, scale: i32) -> i32 {
    if exp >= SCALBN_EXP_MAX {
        assert!(scale >= 0);
    } else if exp <= SCALBN_EXP_MIN {
        assert!(scale <= 0);
    }
    match exp.checked_add(scale) {
        None => {
            if scale < 0 {
                SCALBN_EXP_MIN
            } else {
                SCALBN_EXP_MAX
            }
        }
        Some(e) => e.clamp(SCALBN_EXP_MIN, SCALBN_EXP_MAX),
    }
}

/// `frac_msb_is_snan`.
#[inline]
pub(crate) fn frac_msb_is_snan(msb: bool, s: &FloatStatus) -> bool {
    match s.snan_rule {
        SnanRule::Never => false,
        SnanRule::BitIsOne => msb,
        SnanRule::BitIsZero => !msb,
    }
}

/// `parts_is_snan_frac`.
#[inline]
pub(crate) fn parts_is_snan_frac(frac: u64, s: &FloatStatus) -> bool {
    frac_msb_is_snan((frac >> (BINARY_POINT - 1)) & 1 != 0, s)
}

/// `parts_silence_nan_frac`.
pub(crate) fn parts_silence_nan_frac(frac: u64, s: &FloatStatus) -> u64 {
    match s.snan_rule {
        SnanRule::BitIsZero => frac | (1 << (BINARY_POINT - 1)),
        SnanRule::BitIsOne => (frac & !(1u64 << (BINARY_POINT - 1))) | (1u64 << (BINARY_POINT - 2)),
        SnanRule::Never => panic!("silencing a NaN with float_snan_never"),
    }
}

/// The fraction of `parts64_default_nan`.
pub(crate) fn default_nan_frac64(s: &FloatStatus) -> (bool, u64) {
    let pattern = s.default_nan_pattern;
    assert!(pattern != 0, "default_nan_pattern is not set");
    let sign = pattern >> 7 != 0;
    let mut frac = u64::from(pattern & 0x7f) << (BINARY_POINT - 7);
    if pattern & 1 != 0 {
        frac |= (1u64 << (BINARY_POINT - 7)) - 1;
    }
    (sign, frac)
}

/// `FloatPartsN`: a decomposed number.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Parts<F> {
    pub cls: FloatClass,
    pub sign: bool,
    pub exp: i32,
    pub frac: F,
}

pub(crate) type Parts64 = Parts<Frac64>;
pub(crate) type Parts128 = Parts<Frac128>;

impl<F: Frac> Parts<F> {
    /// A number of class `cls` with zero exponent and fraction.
    #[inline]
    pub(crate) fn of_class(cls: FloatClass, sign: bool) -> Self {
        Parts { cls, sign, exp: 0, frac: F::default() }
    }

    /// `partsN_default_nan`.
    pub(crate) fn default_nan(s: &FloatStatus) -> Self {
        let (sign, frac) = default_nan_frac64(s);
        Parts { cls: FloatClass::QNan, sign, exp: i32::MAX, frac: F::from_dnan(frac) }
    }

    /// `partsN_silence_nan`.
    pub(crate) fn silence_nan(&self, s: &FloatStatus) -> Self {
        let mut r = *self;
        r.frac.set_hi(parts_silence_nan_frac(r.frac.hi(), s));
        r.cls = FloatClass::QNan;
        r
    }

    /// `partsN_return_nan`.
    pub(crate) fn return_nan(&self, s: &mut FloatStatus) -> Self {
        match self.cls {
            FloatClass::SNan => {
                s.raise(flags::INVALID | flags::INVALID_SNAN);
                if s.default_nan_mode { Self::default_nan(s) } else { self.silence_nan(s) }
            }
            FloatClass::QNan => {
                if s.default_nan_mode {
                    Self::default_nan(s)
                } else {
                    *self
                }
            }
            _ => unreachable!("return_nan on a number"),
        }
    }

    /// `partsN_pick_nan`.
    pub(crate) fn pick_nan(a: &Self, b: &Self, s: &mut FloatStatus) -> Self {
        let mut have_snan = false;
        if a.cls == FloatClass::SNan || b.cls == FloatClass::SNan {
            s.raise(flags::INVALID | flags::INVALID_SNAN);
            have_snan = true;
        }
        if s.default_nan_mode {
            return Self::default_nan(s);
        }
        let ret = match s.float_2nan_prop_rule {
            Float2NanPropRule::SAb if have_snan => {
                if a.cls == FloatClass::SNan {
                    a
                } else {
                    b
                }
            }
            Float2NanPropRule::SAb | Float2NanPropRule::Ab => {
                if a.cls.is_nan() {
                    a
                } else {
                    b
                }
            }
            Float2NanPropRule::SBa if have_snan => {
                if b.cls == FloatClass::SNan {
                    b
                } else {
                    a
                }
            }
            Float2NanPropRule::SBa | Float2NanPropRule::Ba => {
                if b.cls.is_nan() {
                    b
                } else {
                    a
                }
            }
            Float2NanPropRule::X87 => {
                let early = if a.cls == FloatClass::SNan {
                    if b.cls != FloatClass::SNan {
                        Some(if b.cls == FloatClass::QNan { b } else { a })
                    } else {
                        None
                    }
                } else if a.cls == FloatClass::QNan {
                    if b.cls == FloatClass::SNan || b.cls != FloatClass::QNan {
                        Some(a)
                    } else {
                        None
                    }
                } else {
                    Some(b)
                };
                match early {
                    Some(r) => r,
                    None => {
                        let mut cmp = a.frac.cmp(&b.frac);
                        if cmp == 0 {
                            cmp = i32::from(!a.sign & b.sign);
                        }
                        if cmp > 0 { a } else { b }
                    }
                }
            }
            Float2NanPropRule::None => panic!("float_2nan_prop_rule is not set"),
        };
        if ret.cls == FloatClass::SNan { ret.silence_nan(s) } else { *ret }
    }

    /// `partsN_pick_nan_muladd`.
    fn pick_nan_muladd(
        a: &Self,
        b: &Self,
        c: &Self,
        s: &mut FloatStatus,
        ab_mask: u32,
        abc_mask: u32,
    ) -> Self {
        let infzero = ab_mask == CMASK_INFZERO;
        let have_snan = abc_mask & CMASK_SNAN != 0;
        let izn_rule = s.float_infzeronan_rule.0;

        if have_snan {
            s.raise(flags::INVALID | flags::INVALID_SNAN);
        }
        if infzero && izn_rule & InfZeroNanRule::SUPPRESS_INVALID == 0 {
            s.raise(flags::INVALID | flags::INVALID_IMZ);
        }
        if s.default_nan_mode {
            return Self::default_nan(s);
        }
        let ret = if infzero {
            match InfZeroNanRule(izn_rule & !InfZeroNanRule::SUPPRESS_INVALID) {
                InfZeroNanRule::DNAN_NEVER => {}
                InfZeroNanRule::DNAN_ALWAYS => return Self::default_nan(s),
                InfZeroNanRule::DNAN_IF_QNAN => {
                    if c.cls == FloatClass::QNan {
                        return Self::default_nan(s);
                    }
                }
                _ => panic!("float_infzeronan_rule is not set"),
            }
            c
        } else {
            let val = [a, b, c, c];
            let mut rule = s.float_3nan_prop_rule.0;
            assert!(rule != Float3NanPropRule::NONE.0, "float_3nan_prop_rule is not set");
            let prefer_snan = have_snan && rule & Float3NanPropRule::SNAN_MASK != 0;
            loop {
                let r = val[usize::from(rule & 3)];
                rule >>= 2;
                if prefer_snan {
                    if r.cls == FloatClass::SNan {
                        break r;
                    }
                } else if r.cls.is_nan() {
                    break r;
                }
            }
        };
        if ret.cls == FloatClass::SNan { ret.silence_nan(s) } else { *ret }
    }

    /// `partsN_canonicalize`.
    pub(crate) fn canonicalize(&mut self, s: &mut FloatStatus, fmt: &FloatFmt) {
        let has_pseudo_denormals = fmt.has_explicit_bit
            && s.floatx80_behaviour.0 & FloatX80Behaviour::PSEUDO_DENORMAL_VALID != 0;

        if self.exp == 0 {
            if self.frac.eqz() {
                self.cls = FloatClass::Zero;
            } else if s.flush_inputs_to_zero {
                s.raise(flags::INPUT_DENORMAL_FLUSHED);
                self.cls = FloatClass::Zero;
                self.frac = F::default();
            } else {
                let shift = self.frac.normalize();
                self.cls = FloatClass::Denormal;
                self.exp = fmt.frac_shift - fmt.exp_bias - shift + i32::from(!has_pseudo_denormals);
            }
            return;
        }
        if self.exp == fmt.exp_max && !fmt.expmax_normal {
            if self.frac.eqz() {
                self.cls = FloatClass::Inf;
            } else {
                self.frac.shl(fmt.frac_shift);
                self.cls = if parts_is_snan_frac(self.frac.hi(), s) {
                    FloatClass::SNan
                } else {
                    FloatClass::QNan
                };
            }
            return;
        }
        self.cls = FloatClass::Normal;
        self.exp -= fmt.exp_bias;
        self.frac.shl(fmt.frac_shift);
        self.frac.set_hi(self.frac.hi() | IMPLICIT_BIT);
    }

    /// `partsN_uncanon_normal`.
    pub(crate) fn uncanon_normal(&mut self, s: &mut FloatStatus, fmt: &FloatFmt) {
        let exp_max = fmt.exp_max;
        let frac_shift = fmt.frac_shift;
        let round_mask = fmt.round_mask;
        let frac_lsb = round_mask.wrapping_add(1);
        let frac_lsbm1 = round_mask ^ (round_mask >> 1);
        let roundeven_mask = round_mask | frac_lsb;
        let mut overflow_norm = false;
        let mut fl: u16 = 0;
        let wide_lsb = F::N > 64 && frac_lsb == 0;

        let rne = |p: &Self| -> u64 {
            if wide_lsb {
                if (p.frac.hi() & 1) != 0 || (p.frac.lo() & round_mask) != frac_lsbm1 {
                    frac_lsbm1
                } else {
                    0
                }
            } else if (p.frac.lo() & roundeven_mask) != frac_lsbm1 {
                frac_lsbm1
            } else {
                0
            }
        };
        let rodd = |p: &Self| -> u64 {
            if wide_lsb {
                if p.frac.hi() & 1 != 0 { 0 } else { round_mask }
            } else if p.frac.lo() & frac_lsb != 0 {
                0
            } else {
                round_mask
            }
        };

        let mut inc = match s.rounding_mode {
            RoundMode::NearestEvenMax => {
                overflow_norm = true;
                rne(self)
            }
            RoundMode::NearestEven => rne(self),
            RoundMode::TiesAway => frac_lsbm1,
            RoundMode::ToZero => {
                overflow_norm = true;
                0
            }
            RoundMode::Up => {
                overflow_norm |= self.sign;
                if self.sign { 0 } else { round_mask }
            }
            RoundMode::Down => {
                overflow_norm |= !self.sign;
                if self.sign { round_mask } else { 0 }
            }
            RoundMode::ToOdd => {
                overflow_norm = true;
                rodd(self)
            }
            RoundMode::ToOddInf => rodd(self),
        };

        let mut exp = self.exp + fmt.exp_bias;
        if exp > 0 {
            if self.frac.lo() & round_mask != 0 {
                fl |= flags::INEXACT;
                let (r, carry) = self.frac.addi(inc);
                self.frac = r;
                if carry {
                    self.frac.shr(1);
                    self.frac.set_hi(self.frac.hi() | IMPLICIT_BIT);
                    exp += 1;
                }
                self.frac.set_lo(self.frac.lo() & !round_mask);
            }

            if exp >= exp_max {
                if !fmt.expmax_normal {
                    fl |= flags::OVERFLOW;
                    if s.rebias_overflow {
                        exp -= fmt.exp_re_bias;
                    } else if overflow_norm {
                        fl |= flags::INEXACT;
                        exp = exp_max - 1;
                        self.frac.allones();
                        self.frac.set_lo(self.frac.lo() & !round_mask);
                    } else {
                        fl |= flags::INEXACT;
                        self.cls = FloatClass::Inf;
                        exp = exp_max;
                        self.frac = F::default();
                    }
                } else if exp > exp_max {
                    fl = if fmt.overflow_raises_invalid {
                        flags::INVALID
                    } else {
                        flags::OVERFLOW | flags::INEXACT
                    };
                    exp = exp_max;
                    self.frac.allones();
                    self.frac.set_lo(self.frac.lo() & !round_mask);
                }
            }
            self.frac.shr(frac_shift);
        } else if s.rebias_underflow {
            fl |= flags::UNDERFLOW;
            exp += fmt.exp_re_bias;
            if self.frac.lo() & round_mask != 0 {
                fl |= flags::INEXACT;
                let (r, carry) = self.frac.addi(inc);
                self.frac = r;
                if carry {
                    self.frac.shr(1);
                    self.frac.set_hi(self.frac.hi() | IMPLICIT_BIT);
                    exp += 1;
                }
                self.frac.set_lo(self.frac.lo() & !round_mask);
            }
            self.frac.shr(frac_shift);
        } else if s.flush_to_zero && s.ftz_before_rounding {
            fl |= flags::OUTPUT_DENORMAL_FLUSHED;
            self.cls = FloatClass::Zero;
            exp = 0;
            self.frac = F::default();
        } else {
            let mut is_tiny = s.tininess_before_rounding || exp < 0;
            let has_pseudo_denormals = fmt.has_explicit_bit
                && s.floatx80_behaviour.0 & FloatX80Behaviour::PSEUDO_DENORMAL_VALID != 0;

            if !is_tiny {
                let (_, carry) = self.frac.addi(inc);
                is_tiny = !carry;
            }

            self.frac.shrjam(i32::from(!has_pseudo_denormals) - exp);

            if self.frac.lo() & round_mask != 0 {
                match s.rounding_mode {
                    RoundMode::NearestEven | RoundMode::NearestEvenMax => inc = rne(self),
                    RoundMode::ToOdd | RoundMode::ToOddInf => inc = rodd(self),
                    _ => {}
                }
                fl |= flags::INEXACT;
                let (r, _) = self.frac.addi(inc);
                self.frac = r;
                self.frac.set_lo(self.frac.lo() & !round_mask);
            }

            exp = i32::from((self.frac.hi() & IMPLICIT_BIT) != 0 && !has_pseudo_denormals);
            self.frac.shr(frac_shift);

            if is_tiny {
                if s.flush_to_zero {
                    assert!(!s.ftz_before_rounding);
                    fl |= flags::OUTPUT_DENORMAL_FLUSHED;
                    self.cls = FloatClass::Zero;
                    exp = 0;
                    self.frac = F::default();
                } else if fl & flags::INEXACT != 0 {
                    fl |= flags::UNDERFLOW;
                }
                if exp == 0 && self.frac.eqz() {
                    self.cls = FloatClass::Zero;
                }
            }
        }
        self.exp = exp;
        s.raise(fl);
    }

    /// `partsN_uncanon`.
    pub(crate) fn uncanon(&mut self, s: &mut FloatStatus, fmt: &FloatFmt) {
        match self.cls {
            FloatClass::Normal | FloatClass::Denormal => self.uncanon_normal(s, fmt),
            FloatClass::Zero => {
                self.exp = 0;
                self.frac = F::default();
            }
            FloatClass::Inf => {
                assert!(!fmt.expmax_normal);
                self.exp = fmt.exp_max;
                self.frac = F::default();
            }
            FloatClass::QNan | FloatClass::SNan => {
                assert!(!fmt.expmax_normal);
                self.exp = fmt.exp_max;
                self.frac.shr(fmt.frac_shift);
            }
            FloatClass::Unclassified => unreachable!(),
        }
    }

    /// `partsN_add_normal`.
    pub(crate) fn add_normal(a: &mut Self, b: &mut Self) {
        let exp_diff = a.exp - b.exp;
        if exp_diff > 0 {
            b.frac.shrjam(exp_diff);
        } else if exp_diff < 0 {
            a.frac.shrjam(-exp_diff);
            a.exp = b.exp;
        }
        let (r, carry) = a.frac.add(&b.frac);
        a.frac = r;
        if carry {
            a.frac.shrjam(1);
            a.frac.set_hi(a.frac.hi() | IMPLICIT_BIT);
            a.exp += 1;
        }
    }

    /// `partsN_sub_normal`.
    pub(crate) fn sub_normal(a: &mut Self, b: &mut Self) -> bool {
        let exp_diff = a.exp - b.exp;
        if exp_diff > 0 {
            b.frac.shrjam(exp_diff);
            a.frac = a.frac.sub(&b.frac).0;
        } else if exp_diff < 0 {
            a.exp = b.exp;
            a.sign = !a.sign;
            a.frac.shrjam(-exp_diff);
            a.frac = b.frac.sub(&a.frac).0;
        } else {
            let (r, borrow) = a.frac.sub(&b.frac);
            a.frac = r;
            if borrow {
                a.frac.neg();
                a.sign = !a.sign;
            }
        }
        let shift = a.frac.normalize();
        if shift < F::N {
            a.exp -= shift;
            return true;
        }
        a.cls = FloatClass::Zero;
        false
    }

    /// `partsN_addsub`.
    pub(crate) fn addsub(
        a_orig: &Self,
        b_orig: &Self,
        s: &mut FloatStatus,
        subtract: bool,
    ) -> Self {
        let mut ab_mask = a_orig.cls.cmask() | b_orig.cls.cmask();
        if ab_mask & CMASK_ANYNAN != 0 {
            return Self::pick_nan(a_orig, b_orig, s);
        }
        record_denormals_used(ab_mask, s);

        let mut a = *a_orig;
        let mut b = *b_orig;
        b.sign ^= subtract;

        if a.sign != b.sign {
            if only_normals(ab_mask) {
                if Self::sub_normal(&mut a, &mut b) {
                    return a;
                }
                ab_mask = CMASK_ZERO;
            }
            if ab_mask == CMASK_ZERO {
                a.sign = s.rounding_mode == RoundMode::Down;
                return a;
            }
            if ab_mask & CMASK_INF != 0 {
                if a.cls != FloatClass::Inf {
                    return b;
                }
                if b.cls != FloatClass::Inf {
                    return a;
                }
                s.raise(flags::INVALID | flags::INVALID_ISI);
                return Self::default_nan(s);
            }
        } else {
            if only_normals(ab_mask) {
                Self::add_normal(&mut a, &mut b);
                return a;
            }
            if ab_mask == CMASK_ZERO {
                return a;
            }
            if ab_mask & CMASK_INF != 0 {
                a.cls = FloatClass::Inf;
                return a;
            }
        }
        if b.cls == FloatClass::Zero { a } else { b }
    }

    /// `partsN_round_to_int_normal`: round in place, returning true if inexact.
    pub(crate) fn round_to_int_normal(
        &mut self,
        rmode: RoundMode,
        scale: i32,
        frac_size: i32,
    ) -> bool {
        self.exp = exp_scalbn(self.exp, scale);

        if self.exp < 0 {
            let one = match rmode {
                RoundMode::NearestEven | RoundMode::NearestEvenMax => {
                    if self.exp == -1 {
                        let (tmp, _) = self.frac.add(&self.frac);
                        !tmp.eqz()
                    } else {
                        false
                    }
                }
                RoundMode::TiesAway => self.exp == -1,
                RoundMode::ToZero => false,
                RoundMode::Up => !self.sign,
                RoundMode::Down => self.sign,
                RoundMode::ToOdd | RoundMode::ToOddInf => true,
            };
            self.frac = F::default();
            self.exp = 0;
            if one {
                self.frac.set_hi(IMPLICIT_BIT);
            } else {
                self.cls = FloatClass::Zero;
            }
            return true;
        }

        let shift_adj;
        let frac_lsb: u64;
        if F::N > 64 && self.exp < F::N - 64 {
            shift_adj = (F::N - 1) - (self.exp + 2);
            self.frac.shrjam(shift_adj);
            frac_lsb = 1 << 2;
        } else {
            shift_adj = 0;
            frac_lsb = IMPLICIT_BIT >> (self.exp.min(frac_size) & 63);
        }

        let frac_lsbm1 = frac_lsb >> 1;
        let rnd_mask = frac_lsb.wrapping_sub(1);
        let rnd_even_mask = rnd_mask | frac_lsb;

        if self.frac.lo() & rnd_mask == 0 {
            self.frac.shl(shift_adj);
            return false;
        }

        let inc = match rmode {
            RoundMode::NearestEven | RoundMode::NearestEvenMax => {
                if (self.frac.lo() & rnd_even_mask) != frac_lsbm1 { frac_lsbm1 } else { 0 }
            }
            RoundMode::TiesAway => frac_lsbm1,
            RoundMode::ToZero => 0,
            RoundMode::Up => {
                if self.sign {
                    0
                } else {
                    rnd_mask
                }
            }
            RoundMode::Down => {
                if self.sign {
                    rnd_mask
                } else {
                    0
                }
            }
            RoundMode::ToOdd | RoundMode::ToOddInf => {
                if self.frac.lo() & frac_lsb != 0 {
                    0
                } else {
                    rnd_mask
                }
            }
        };

        if shift_adj == 0 {
            let (r, carry) = self.frac.addi(inc);
            self.frac = r;
            if carry {
                self.frac.shr(1);
                self.frac.set_hi(self.frac.hi() | IMPLICIT_BIT);
                self.exp += 1;
            }
            self.frac.set_lo(self.frac.lo() & !rnd_mask);
        } else {
            self.frac = self.frac.addi(inc).0;
            self.frac.set_lo(self.frac.lo() & !rnd_mask);
            self.frac.shl(shift_adj - 1);
            if self.frac.hi() & IMPLICIT_BIT != 0 {
                self.exp += 1;
            } else {
                self.frac = self.frac.add(&self.frac).0;
            }
        }
        true
    }

    /// `partsN_round_to_int`.
    pub(crate) fn round_to_int(
        &self,
        rmode: RoundMode,
        scale: i32,
        s: &mut FloatStatus,
        fmt: &FloatFmt,
    ) -> Self {
        match self.cls {
            FloatClass::QNan | FloatClass::SNan => self.return_nan(s),
            FloatClass::Zero | FloatClass::Inf => *self,
            FloatClass::Normal | FloatClass::Denormal => {
                let mut r = *self;
                if r.round_to_int_normal(rmode, scale, fmt.frac_size) {
                    s.raise(flags::INEXACT);
                }
                r
            }
            FloatClass::Unclassified => unreachable!(),
        }
    }

    /// `partsN_float_to_sint`.
    pub(crate) fn float_to_sint(
        &mut self,
        rmode: RoundMode,
        scale: i32,
        min: i64,
        max: i64,
        s: &mut FloatStatus,
    ) -> i64 {
        let mut fl: u16 = 0;
        let r: u64;
        match self.cls {
            FloatClass::SNan | FloatClass::QNan => {
                if self.cls == FloatClass::SNan {
                    fl |= flags::INVALID_SNAN;
                }
                fl |= flags::INVALID;
                r = max as u64;
            }
            FloatClass::Inf => {
                fl = flags::INVALID | flags::INVALID_CVTI;
                r = if self.sign { min as u64 } else { max as u64 };
            }
            FloatClass::Zero => return 0,
            FloatClass::Normal | FloatClass::Denormal => {
                if self.round_to_int_normal(rmode, scale, F::N - 2) {
                    fl = flags::INEXACT;
                }
                let mut v = if self.exp <= BINARY_POINT {
                    self.frac.hi() >> (BINARY_POINT - self.exp)
                } else {
                    u64::MAX
                };
                if self.sign {
                    if v <= (min as u64).wrapping_neg() {
                        v = v.wrapping_neg();
                    } else {
                        fl = flags::INVALID | flags::INVALID_CVTI;
                        v = min as u64;
                    }
                } else if v > max as u64 {
                    fl = flags::INVALID | flags::INVALID_CVTI;
                    v = max as u64;
                }
                r = v;
            }
            FloatClass::Unclassified => unreachable!(),
        }
        s.raise(fl);
        r as i64
    }

    /// `partsN_float_to_uint`.
    pub(crate) fn float_to_uint(
        &mut self,
        rmode: RoundMode,
        scale: i32,
        max: u64,
        s: &mut FloatStatus,
    ) -> u64 {
        let mut fl: u16 = 0;
        let r: u64;
        match self.cls {
            FloatClass::SNan | FloatClass::QNan => {
                if self.cls == FloatClass::SNan {
                    fl |= flags::INVALID_SNAN;
                }
                fl |= flags::INVALID;
                r = max;
            }
            FloatClass::Inf => {
                fl = flags::INVALID | flags::INVALID_CVTI;
                r = if self.sign { 0 } else { max };
            }
            FloatClass::Zero => return 0,
            FloatClass::Normal | FloatClass::Denormal => {
                let inexact = self.round_to_int_normal(rmode, scale, F::N - 2);
                if inexact {
                    fl = flags::INEXACT;
                }
                if inexact && self.cls == FloatClass::Zero {
                    r = 0;
                } else if self.sign {
                    fl = flags::INVALID | flags::INVALID_CVTI;
                    r = 0;
                } else if self.exp > BINARY_POINT {
                    fl = flags::INVALID | flags::INVALID_CVTI;
                    r = max;
                } else {
                    let v = self.frac.hi() >> (BINARY_POINT - self.exp);
                    if v > max {
                        fl = flags::INVALID | flags::INVALID_CVTI;
                        r = max;
                    } else {
                        r = v;
                    }
                }
            }
            FloatClass::Unclassified => unreachable!(),
        }
        s.raise(fl);
        r
    }

    /// `partsN_sint_to_float`.
    pub(crate) fn sint_to_float(a: i64, scale: i32) -> Self {
        let mut p = Self::of_class(FloatClass::Zero, false);
        if a == 0 {
            return p;
        }
        p.cls = FloatClass::Normal;
        let mut f = a as u64;
        if a < 0 {
            f = f.wrapping_neg();
            p.sign = true;
        }
        let shift = f.leading_zeros() as i32;
        let scale = scale.clamp(-0x10000, 0x10000);
        p.exp = BINARY_POINT - shift + scale;
        p.frac.set_hi(f << shift);
        p
    }

    /// `partsN_uint_to_float`.
    pub(crate) fn uint_to_float(a: u64, scale: i32) -> Self {
        let mut p = Self::of_class(FloatClass::Zero, false);
        if a != 0 {
            let shift = a.leading_zeros() as i32;
            let scale = scale.clamp(-0x10000, 0x10000);
            p.cls = FloatClass::Normal;
            p.exp = BINARY_POINT - shift + scale;
            p.frac.set_hi(a << shift);
        }
        p
    }

    /// `partsN_minmax`.
    pub(crate) fn minmax(a: &Self, b: &Self, s: &mut FloatStatus, mm: u32) -> Self {
        let ab_mask = a.cls.cmask() | b.cls.cmask();
        if ab_mask & CMASK_ANYNAN != 0 {
            if mm & (MINMAX_ISNUM | MINMAX_ISNUMBER) != 0
                && ab_mask & CMASK_SNAN == 0
                && ab_mask & !CMASK_QNAN != 0
            {
                record_denormals_used(ab_mask, s);
                return if a.cls.is_nan() { *b } else { *a };
            }
            if mm & MINMAX_ISNUMBER != 0
                && ab_mask & CMASK_SNAN != 0
                && ab_mask & !CMASK_ANYNAN != 0
            {
                s.raise(flags::INVALID);
                return if a.cls.is_nan() { *b } else { *a };
            }
            return Self::pick_nan(a, b, s);
        }

        record_denormals_used(ab_mask, s);

        let mut a_exp = a.exp;
        let mut b_exp = b.exp;
        if !only_normals(ab_mask) {
            match a.cls {
                FloatClass::Inf => a_exp = i32::from(i16::MAX),
                FloatClass::Zero => a_exp = i32::from(i16::MIN),
                _ => {}
            }
            match b.cls {
                FloatClass::Inf => b_exp = i32::from(i16::MAX),
                FloatClass::Zero => b_exp = i32::from(i16::MIN),
                _ => {}
            }
        }

        let mut cmp = a_exp - b_exp;
        if cmp == 0 {
            cmp = a.frac.cmp(&b.frac);
        }
        if mm & MINMAX_ISMAG == 0 || cmp == 0 {
            if a.sign != b.sign {
                cmp = if a.sign { -1 } else { 1 };
            } else if a.sign {
                cmp = -cmp;
            }
        }
        if mm & MINMAX_ISMIN != 0 {
            cmp = -cmp;
        }
        if cmp < 0 { *b } else { *a }
    }

    /// `partsN_compare`.
    pub(crate) fn compare(
        a: &Self,
        b: &Self,
        s: &mut FloatStatus,
        is_quiet: bool,
    ) -> FloatRelation {
        let ab_mask = a.cls.cmask() | b.cls.cmask();
        let a_sign = || if a.sign { FloatRelation::Less } else { FloatRelation::Greater };
        let b_sign = || if b.sign { FloatRelation::Greater } else { FloatRelation::Less };

        if only_normals(ab_mask) {
            record_denormals_used(ab_mask, s);
            if a.sign != b.sign {
                return a_sign();
            }
            let mut cmp = if a.exp == b.exp {
                a.frac.cmp(&b.frac)
            } else if a.exp < b.exp {
                -1
            } else {
                1
            };
            if a.sign {
                cmp = -cmp;
            }
            return match cmp {
                -1 => FloatRelation::Less,
                0 => FloatRelation::Equal,
                _ => FloatRelation::Greater,
            };
        }

        if ab_mask & CMASK_ANYNAN != 0 {
            if ab_mask & CMASK_SNAN != 0 {
                s.raise(flags::INVALID | flags::INVALID_SNAN);
            } else if !is_quiet {
                s.raise(flags::INVALID);
            }
            return FloatRelation::Unordered;
        }

        record_denormals_used(ab_mask, s);

        if ab_mask & CMASK_ZERO != 0 {
            return if ab_mask == CMASK_ZERO {
                FloatRelation::Equal
            } else if a.cls == FloatClass::Zero {
                b_sign()
            } else {
                a_sign()
            };
        }
        if ab_mask == CMASK_INF {
            if a.sign == b.sign {
                return FloatRelation::Equal;
            }
        } else if b.cls == FloatClass::Inf {
            return b_sign();
        }
        a_sign()
    }

    /// `partsN_scalbn`.
    pub(crate) fn scalbn(&self, n: i32, s: &mut FloatStatus) -> Self {
        match self.cls {
            FloatClass::QNan | FloatClass::SNan => self.return_nan(s),
            FloatClass::Zero | FloatClass::Inf => *self,
            FloatClass::Normal | FloatClass::Denormal => {
                if self.cls == FloatClass::Denormal {
                    s.raise(flags::INPUT_DENORMAL_USED);
                }
                let mut r = *self;
                r.exp = exp_scalbn(r.exp, n);
                r
            }
            FloatClass::Unclassified => unreachable!(),
        }
    }

    /// `partsN_sqrt`.
    pub(crate) fn sqrt(&mut self, s: &mut FloatStatus, fmt: &FloatFmt) {
        const THREE32: u32 = 3 << 30;
        const THREE64: u64 = 3 << 62;

        match self.cls {
            FloatClass::Normal => {}
            FloatClass::Denormal => {
                if !self.sign {
                    s.raise(flags::INPUT_DENORMAL_USED);
                }
            }
            FloatClass::SNan | FloatClass::QNan => {
                *self = self.return_nan(s);
                return;
            }
            FloatClass::Zero => return,
            FloatClass::Inf => {
                if self.sign {
                    s.raise(flags::INVALID | flags::INVALID_SQRT);
                    *self = Self::default_nan(s);
                }
                return;
            }
            FloatClass::Unclassified => unreachable!(),
        }
        if self.sign {
            s.raise(flags::INVALID | flags::INVALID_SQRT);
            *self = Self::default_nan(s);
            return;
        }

        let exp_odd = self.exp & 1 != 0;
        let index = (((self.frac.hi() >> 57) & 0x3f) | (u64::from(!exp_odd) << 6)) as usize;
        if !exp_odd {
            self.frac.shr(1);
        }

        let m64 = self.frac.hi();
        let m32 = (m64 >> 32) as u32;
        let mut r32 = u32::from(RSQRT_TAB[index]) << 16;
        let mut s32 = ((u64::from(m32) * u64::from(r32)) >> 32) as u32;
        let mut d32 = ((u64::from(s32) * u64::from(r32)) >> 32) as u32;
        let mut u32_ = THREE32.wrapping_sub(d32);

        if F::N == 64 {
            r32 = ((u64::from(r32) * u64::from(u32_)) >> 31) as u32;
            s32 = ((u64::from(m32) * u64::from(r32)) >> 32) as u32;
            d32 = ((u64::from(s32) * u64::from(r32)) >> 32) as u32;
            u32_ = THREE32.wrapping_sub(d32);

            if fmt.frac_size <= 23 {
                s32 = ((u64::from(s32) * u64::from(u32_)) >> 32) as u32;
                s32 = s32.wrapping_sub(1) >> 6;
                let d0 = (m32 << 16).wrapping_sub(s32.wrapping_mul(s32));
                let d1 = s32.wrapping_sub(d0);
                let d2 = d1.wrapping_add(s32).wrapping_add(1);
                s32 = s32.wrapping_add(d1 >> 31);
                let mut hi = u64::from(s32) << (64 - 25);
                if d2 != 0 {
                    hi = if ((d1 ^ d2) as i32) < 0 {
                        hi.wrapping_sub(1)
                    } else {
                        hi.wrapping_add(1)
                    };
                }
                self.frac.set_hi(hi);
            } else {
                let r64 = u64::from(r32).wrapping_mul(u64::from(u32_)).wrapping_mul(2);
                let mut s64 = mul64_to_128(m64, r64).0;
                let d64 = mul64_to_128(s64, r64).0;
                let u64_ = THREE64.wrapping_sub(d64);
                s64 = mul64_to_128(s64, u64_).0;
                s64 = s64.wrapping_sub(2) >> 9;
                let d0 = (m64 << 42).wrapping_sub(s64.wrapping_mul(s64));
                let d1 = s64.wrapping_sub(d0);
                let d2 = d1.wrapping_add(s64).wrapping_add(1);
                s64 = s64.wrapping_add(d1 >> 63);
                let mut hi = s64 << (64 - 54);
                if d2 != 0 {
                    hi = if ((d1 ^ d2) as i64) < 0 {
                        hi.wrapping_sub(1)
                    } else {
                        hi.wrapping_add(1)
                    };
                }
                self.frac.set_hi(hi);
            }
        } else {
            let mut r64 = u64::from(r32).wrapping_mul(u64::from(u32_)).wrapping_mul(2);
            let mut s64 = mul64_to_128(m64, r64).0;
            let mut d64 = mul64_to_128(s64, r64).0;
            let mut u64_ = THREE64.wrapping_sub(d64);
            r64 = mul64_to_128(u64_, r64).0;
            r64 <<= 1;

            s64 = mul64_to_128(m64, r64).0;
            d64 = mul64_to_128(s64, r64).0;
            u64_ = THREE64.wrapping_sub(d64);
            let (mut rh, mut rl) = mul64_to_128(u64_, r64);
            (rh, rl) = add128(rh, rl, rh, rl);

            let (fh, fl) = (self.frac.hi(), self.frac.lo());
            let (mut sh, mut sl, _, _) = mul128_to_256(fh, fl, rh, rl);
            let (dh, dl, _, _) = mul128_to_256(sh, sl, rh, rl);
            let (uh, ul) = sub128(THREE64, 0, dh, dl);
            (sh, sl, _, _) = mul128_to_256(uh, ul, sh, sl);

            (sh, sl) = sub128(sh, sl, 0, 4);
            (sh, sl) = shift128_right(sh, sl, 13);

            let (mut d0h, d0l) = mul64_to_128(sl, sl);
            d0h = d0h.wrapping_add(2u64.wrapping_mul(sh).wrapping_mul(sl));
            let (d0h, d0l) = sub128(fl << 34, 0, d0h, d0l);
            let (d1h, d1l) = sub128(sh, sl, d0h, d0l);
            let (d2h, d2l) = add128(sh, sl, 0, 1);
            let (d2h, d2l) = add128(d2h, d2l, d1h, d1l);
            (sh, sl) = add128(sh, sl, 0, d1h >> 63);
            (sh, sl) = shift128_left(sh, sl, 128 - 114);

            if (d2h | d2l) != 0 {
                if ((d1h ^ d2h) as i64) < 0 {
                    (sh, sl) = sub128(sh, sl, 0, 1);
                } else {
                    (sh, sl) = add128(sh, sl, 0, 1);
                }
            }
            self.frac.set_lo(sl);
            self.frac.set_hi(sh);
        }

        self.exp >>= 1;
        if self.frac.hi() & IMPLICIT_BIT == 0 {
            self.frac = self.frac.add(&self.frac).0;
        } else {
            self.exp += 1;
        }
    }
}

/// `float_minmax_ismin`.
pub(crate) const MINMAX_ISMIN: u32 = 1;
/// `float_minmax_isnum`.
pub(crate) const MINMAX_ISNUM: u32 = 2;
/// `float_minmax_ismag`.
pub(crate) const MINMAX_ISMAG: u32 = 4;
/// `float_minmax_isnumber`.
pub(crate) const MINMAX_ISNUMBER: u32 = 8;

impl<F: FracN> Parts<F> {
    /// `partsN_mul`.
    pub(crate) fn mul(a: &Self, b: &Self, s: &mut FloatStatus) -> Self {
        let ab_mask = a.cls.cmask() | b.cls.cmask();
        let sign = a.sign ^ b.sign;

        if only_normals(ab_mask) {
            record_denormals_used(ab_mask, s);
            let tmp = a.frac.mulw(&b.frac);
            let mut r = Parts {
                cls: FloatClass::Normal,
                sign,
                exp: a.exp + b.exp + 1,
                frac: F::truncjam(&tmp),
            };
            if r.frac.hi() & IMPLICIT_BIT == 0 {
                r.frac = r.frac.add(&r.frac).0;
                r.exp -= 1;
            }
            return r;
        }
        if ab_mask == CMASK_INFZERO {
            s.raise(flags::INVALID | flags::INVALID_IMZ);
            return Self::default_nan(s);
        }
        if ab_mask & CMASK_ANYNAN != 0 {
            return Self::pick_nan(a, b, s);
        }
        record_denormals_used(ab_mask, s);
        if ab_mask & CMASK_INF != 0 {
            return Self::of_class(FloatClass::Inf, sign);
        }
        Self::of_class(FloatClass::Zero, sign)
    }

    /// `partsN_muladd`, without the result negation, which the caller applies after rounding.
    pub(crate) fn muladd(a: &Self, b: &Self, c: &Self, fl: u32, s: &mut FloatStatus) -> Self {
        let ab_mask = a.cls.cmask() | b.cls.cmask();
        let c_mask = c.cls.cmask();
        let abc_mask = ab_mask | c_mask;
        let c_sign = c.sign ^ (fl & muladd::NEGATE_C != 0);
        let p_sign = a.sign ^ b.sign ^ (fl & muladd::NEGATE_PRODUCT != 0);

        let likely_mask = ab_mask | (c_mask & !CMASK_ZERO);
        if only_normals(likely_mask) {
            record_denormals_used(abc_mask, s);

            let mut p_widen = Parts {
                cls: FloatClass::Unclassified,
                sign: p_sign,
                exp: a.exp + b.exp + 1,
                frac: a.frac.mulw(&b.frac),
            };
            if p_widen.frac.hi() & IMPLICIT_BIT == 0 {
                p_widen.frac = p_widen.frac.add(&p_widen.frac).0;
                p_widen.exp -= 1;
            }

            if c_mask & CMASK_ZERO == 0 {
                let mut c_widen = Parts {
                    cls: FloatClass::Unclassified,
                    sign: c_sign,
                    exp: c.exp,
                    frac: c.frac.widen(),
                };
                if p_sign == c_sign {
                    Parts::add_normal(&mut p_widen, &mut c_widen);
                } else if !Parts::sub_normal(&mut p_widen, &mut c_widen) {
                    return Self::of_class(FloatClass::Zero, s.rounding_mode == RoundMode::Down);
                }
            }

            return Parts {
                cls: FloatClass::Normal,
                sign: p_widen.sign,
                exp: p_widen.exp,
                frac: F::truncjam(&p_widen.frac),
            };
        }

        if abc_mask & CMASK_ANYNAN != 0 {
            return Self::pick_nan_muladd(a, b, c, s, ab_mask, abc_mask);
        }
        if ab_mask == CMASK_INFZERO {
            s.raise(flags::INVALID | flags::INVALID_IMZ);
            return Self::default_nan(s);
        }
        if ab_mask & CMASK_INF != 0 {
            if c_mask & CMASK_INF != 0 && p_sign != c_sign {
                s.raise(flags::INVALID | flags::INVALID_ISI);
                return Self::default_nan(s);
            }
            record_denormals_used(abc_mask, s);
            return Self::of_class(FloatClass::Inf, p_sign);
        }
        record_denormals_used(abc_mask, s);

        if c_mask & CMASK_ZERO == 0
            || p_sign == c_sign
            || fl & muladd::SUPPRESS_ADD_PRODUCT_ZERO != 0
        {
            let mut r = *c;
            r.sign = c_sign;
            return r;
        }
        Self::of_class(FloatClass::Zero, s.rounding_mode == RoundMode::Down)
    }

    /// `partsN_div`.
    pub(crate) fn div(a: &Self, b: &Self, s: &mut FloatStatus) -> Self {
        let ab_mask = a.cls.cmask() | b.cls.cmask();
        let mut r = *a;
        r.sign ^= b.sign;
        r.exp = r.exp.wrapping_sub(b.exp);

        if only_normals(ab_mask) {
            record_denormals_used(ab_mask, s);
            if r.frac.div(&b.frac) {
                r.exp -= 1;
            }
            return r;
        }
        if ab_mask == CMASK_ZERO {
            s.raise(flags::INVALID | flags::INVALID_ZDZ);
            return Self::default_nan(s);
        }
        if ab_mask == CMASK_INF {
            s.raise(flags::INVALID | flags::INVALID_IDI);
            return Self::default_nan(s);
        }
        if ab_mask & CMASK_ANYNAN != 0 {
            return Self::pick_nan(a, b, s);
        }
        if b.cls != FloatClass::Zero {
            record_denormals_used(ab_mask, s);
        }
        if r.cls == FloatClass::Inf || r.cls == FloatClass::Zero {
            return r;
        }
        if b.cls == FloatClass::Inf {
            r.cls = FloatClass::Zero;
            return r;
        }
        s.raise(flags::DIVBYZERO);
        r.cls = FloatClass::Inf;
        r
    }
}

/// The fraction specific part of `partsN_modrem`.
pub(crate) trait ModRem: Sized {
    /// `fracN_modrem`.
    fn frac_modrem(a: &mut Parts<Self>, b: &Parts<Self>, mod_quot: Option<&mut u64>);
}

impl ModRem for Frac64 {
    fn frac_modrem(a: &mut Parts<Self>, b: &Parts<Self>, mod_quot: Option<&mut u64>) {
        if crate::frac::frac64_modrem(
            &mut a.sign,
            &mut a.exp,
            &mut a.frac.0,
            b.exp,
            b.frac.0,
            mod_quot,
        ) {
            a.cls = FloatClass::Zero;
        }
    }
}

impl ModRem for Frac128 {
    fn frac_modrem(a: &mut Parts<Self>, b: &Parts<Self>, mod_quot: Option<&mut u64>) {
        if crate::frac::frac128_modrem(
            &mut a.sign,
            &mut a.exp,
            &mut a.frac,
            b.exp,
            b.frac,
            mod_quot,
        ) {
            a.cls = FloatClass::Zero;
        }
    }
}

impl<F: Frac + ModRem> Parts<F> {
    /// `partsN_modrem`.
    pub(crate) fn modrem(a: &mut Self, b: &Self, mod_quot: Option<&mut u64>, s: &mut FloatStatus) {
        let ab_mask = a.cls.cmask() | b.cls.cmask();
        if only_normals(ab_mask) {
            record_denormals_used(ab_mask, s);
            F::frac_modrem(a, b, mod_quot);
            return;
        }
        if let Some(m) = mod_quot {
            *m = 0;
        }
        if ab_mask & CMASK_ANYNAN != 0 {
            *a = Self::pick_nan(a, b, s);
            return;
        }
        if a.cls == FloatClass::Inf || b.cls == FloatClass::Zero {
            s.raise(flags::INVALID);
            *a = Self::default_nan(s);
            return;
        }
        record_denormals_used(ab_mask, s);
    }
}

impl Parts64 {
    /// `parts64_float_to_sint_modulo`: the rounded two's complement result modulo
    /// 2**(bitsm1 + 1), raising invalid when it does not fit.
    pub(crate) fn float_to_sint_modulo(
        &mut self,
        rmode: RoundMode,
        bitsm1: i32,
        s: &mut FloatStatus,
    ) -> i64 {
        let mut fl: u16 = 0;
        let mut overflow = false;
        let r: u64;
        match self.cls {
            FloatClass::SNan | FloatClass::QNan => {
                if self.cls == FloatClass::SNan {
                    fl |= flags::INVALID_SNAN;
                }
                fl |= flags::INVALID;
                r = 0;
            }
            FloatClass::Inf => {
                overflow = true;
                r = 0;
            }
            FloatClass::Zero => return 0,
            FloatClass::Normal | FloatClass::Denormal => {
                if self.round_to_int_normal(rmode, 0, 64 - 2) {
                    fl = flags::INEXACT;
                }
                let mut v;
                if self.exp <= BINARY_POINT {
                    v = self.frac.0 >> (BINARY_POINT - self.exp);
                    if self.exp == bitsm1 {
                        overflow = !self.sign || self.frac.0 != IMPLICIT_BIT;
                    } else if self.exp > bitsm1 {
                        overflow = true;
                    }
                } else {
                    let shl = self.exp - BINARY_POINT;
                    v = if shl < 64 { self.frac.0 << shl } else { 0 };
                    overflow = true;
                }
                if self.sign {
                    v = v.wrapping_neg();
                }
                r = v;
            }
            FloatClass::Unclassified => unreachable!(),
        }
        if overflow {
            fl = flags::INVALID | flags::INVALID_CVTI;
        }
        s.raise(fl);
        r as i64
    }

    /// `parts64_log2`.
    pub(crate) fn log2(&mut self, s: &mut FloatStatus, fmt: &FloatFmt) {
        match self.cls {
            FloatClass::Normal => {}
            FloatClass::Denormal => {
                if !self.sign {
                    s.raise(flags::INPUT_DENORMAL_USED);
                }
            }
            FloatClass::SNan | FloatClass::QNan => {
                *self = self.return_nan(s);
                return;
            }
            FloatClass::Zero => {
                s.raise(flags::DIVBYZERO);
                self.cls = FloatClass::Inf;
                self.sign = true;
                return;
            }
            FloatClass::Inf => {
                if self.sign {
                    s.raise(flags::INVALID);
                    *self = Self::default_nan(s);
                }
                return;
            }
            FloatClass::Unclassified => unreachable!(),
        }
        if self.sign {
            s.raise(flags::INVALID);
            *self = Self::default_nan(s);
            return;
        }

        let a_exp = self.exp;
        let mut f_exp: i32 = -1;
        let mut r: u64 = 0;
        let mut t: u64 = IMPLICIT_BIT;
        let mut a0 = self.frac.0;
        let mut a1: u64 = 0;

        let mut n = fmt.frac_size + 2;
        if a_exp == -1 {
            n = (fmt.frac_size * 2 + 2).min(62);
        }

        let mut exact = false;
        let mut i = 0;
        while i < n {
            if a1 != 0 {
                let (z0, z1, _, _) = mul128_to_256(a0, a1, a0, a1);
                a0 = z0;
                a1 = z1;
            } else if a0 & 0xffff_ffff != 0 {
                (a0, a1) = mul64_to_128(a0, a0);
            } else if a0 & !IMPLICIT_BIT != 0 {
                a0 >>= 32;
                a0 *= a0;
            } else {
                exact = true;
                break;
            }

            if a0 & IMPLICIT_BIT != 0 {
                if a_exp == 0 && r == 0 {
                    f_exp -= i;
                    r = IMPLICIT_BIT;
                    t = IMPLICIT_BIT;
                    i = 0;
                } else {
                    r |= t;
                }
            } else {
                (a0, a1) = add128(a0, a1, a0, a1);
            }
            t >>= 1;
            i += 1;
        }

        if !exact {
            r |= u64::from(a1 != 0 || a0 & !IMPLICIT_BIT != 0);
        }

        *self = Self::sint_to_float(i64::from(a_exp), 0);
        if r != 0 {
            let mut f = Parts { cls: FloatClass::Normal, sign: false, exp: 0, frac: Frac64(r) };
            f.exp = f_exp - f.frac.normalize();
            if a_exp < 0 {
                Self::sub_normal(self, &mut f);
            } else if a_exp > 0 {
                Self::add_normal(self, &mut f);
            } else {
                *self = f;
            }
        }
    }
}

/// `rsqrt_tab`: the reciprocal square root estimate table, 1 bit of exponent and 6 bits of
/// fraction, from musl.
static RSQRT_TAB: [u16; 128] = [
    0xb451, 0xb2f0, 0xb196, 0xb044, 0xaef9, 0xadb6, 0xac79, 0xab43, 0xaa14, 0xa8eb, 0xa7c8, 0xa6aa,
    0xa592, 0xa480, 0xa373, 0xa26b, 0xa168, 0xa06a, 0x9f70, 0x9e7b, 0x9d8a, 0x9c9d, 0x9bb5, 0x9ad1,
    0x99f0, 0x9913, 0x983a, 0x9765, 0x9693, 0x95c4, 0x94f8, 0x9430, 0x936b, 0x92a9, 0x91ea, 0x912e,
    0x9075, 0x8fbe, 0x8f0a, 0x8e59, 0x8daa, 0x8cfe, 0x8c54, 0x8bac, 0x8b07, 0x8a64, 0x89c4, 0x8925,
    0x8889, 0x87ee, 0x8756, 0x86c0, 0x862b, 0x8599, 0x8508, 0x8479, 0x83ec, 0x8361, 0x82d8, 0x8250,
    0x81c9, 0x8145, 0x80c2, 0x8040, 0xff02, 0xfd0e, 0xfb25, 0xf947, 0xf773, 0xf5aa, 0xf3ea, 0xf234,
    0xf087, 0xeee3, 0xed47, 0xebb3, 0xea27, 0xe8a3, 0xe727, 0xe5b2, 0xe443, 0xe2dc, 0xe17a, 0xe020,
    0xdecb, 0xdd7d, 0xdc34, 0xdaf1, 0xd9b3, 0xd87b, 0xd748, 0xd61a, 0xd4f1, 0xd3cd, 0xd2ad, 0xd192,
    0xd07b, 0xcf69, 0xce5b, 0xcd51, 0xcc4a, 0xcb48, 0xca4a, 0xc94f, 0xc858, 0xc764, 0xc674, 0xc587,
    0xc49d, 0xc3b7, 0xc2d4, 0xc1f4, 0xc116, 0xc03c, 0xbf65, 0xbe90, 0xbdbe, 0xbcef, 0xbc23, 0xbb59,
    0xba91, 0xb9cc, 0xb90a, 0xb84a, 0xb78c, 0xb6d0, 0xb617, 0xb560,
];
