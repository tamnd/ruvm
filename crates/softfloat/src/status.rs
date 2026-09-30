// SPDX-License-Identifier: GPL-2.0-or-later

//! The floating point status: rounding, exception flags and the target rules.
//!
//! This is QEMU's `float_status` from `include/fpu/softfloat-types.h`, with the same fields and
//! the same numeric values, so a target helper can be ported line by line. The presets at the
//! bottom collect what QEMU's target code sets up for the x86 and Arm units.

/// The rounding mode, QEMU's `FloatRoundMode`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RoundMode {
    /// `float_round_nearest_even`.
    #[default]
    NearestEven = 0,
    /// `float_round_down`, toward negative infinity.
    Down = 1,
    /// `float_round_up`, toward positive infinity.
    Up = 2,
    /// `float_round_to_zero`.
    ToZero = 3,
    /// `float_round_ties_away`.
    TiesAway = 4,
    /// `float_round_to_odd`: not directly representable results round to the odd neighbour, and
    /// overflow gives the largest finite number.
    ToOdd = 5,
    /// `float_round_to_odd_inf`: like `ToOdd`, but overflow gives infinity.
    ToOddInf = 6,
    /// `float_round_nearest_even_max`: nearest even, but overflow gives the largest finite
    /// number.
    NearestEvenMax = 7,
}

impl RoundMode {
    /// Every mode, in QEMU's numeric order.
    pub const ALL: [RoundMode; 8] = [
        RoundMode::NearestEven,
        RoundMode::Down,
        RoundMode::Up,
        RoundMode::ToZero,
        RoundMode::TiesAway,
        RoundMode::ToOdd,
        RoundMode::ToOddInf,
        RoundMode::NearestEvenMax,
    ];

    /// The mode with QEMU's numeric value `v`, if there is one.
    pub const fn from_u8(v: u8) -> Option<RoundMode> {
        if (v as usize) < Self::ALL.len() { Some(Self::ALL[v as usize]) } else { None }
    }
}

/// The x87 rounding precision, QEMU's `FloatX80RoundPrec`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FloatX80RoundPrec {
    /// `floatx80_precision_x`, the full 64 bit significand.
    #[default]
    X = 0,
    /// `floatx80_precision_d`, a 53 bit significand.
    D = 1,
    /// `floatx80_precision_s`, a 24 bit significand.
    S = 2,
}

/// How the most significant fraction bit tells a signaling NaN from a quiet one, QEMU's
/// `FloatSNaNRule`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SnanRule {
    /// `float_snan_bit_is_zero`: a clear bit means signaling. This is IEEE 754-2008 and every
    /// modern target.
    #[default]
    BitIsZero = 0,
    /// `float_snan_bit_is_one`: a set bit means signaling (older MIPS, HPPA).
    BitIsOne = 1,
    /// `float_snan_never`: there are no signaling NaNs.
    Never = 2,
}

/// Which input NaN a two operand operation returns, QEMU's `Float2NaNPropRule`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Float2NanPropRule {
    /// `float_2nan_prop_none`: not set. Picking a NaN with this rule panics, as QEMU asserts.
    #[default]
    None = 0,
    /// `float_2nan_prop_s_ab`: prefer a signaling NaN, then A over B (Arm).
    SAb = 1,
    /// `float_2nan_prop_s_ba`: prefer a signaling NaN, then B over A.
    SBa = 2,
    /// `float_2nan_prop_ab`: A if it is a NaN, else B (Arm with FPCR.AH set).
    Ab = 3,
    /// `float_2nan_prop_ba`: B if it is a NaN, else A.
    Ba = 4,
    /// `float_2nan_prop_x87`: the x87 rule, larger significand wins.
    X87 = 5,
}

/// Which input NaN a fused multiply add returns, QEMU's `Float3NaNPropRule`.
///
/// The value packs three two bit operand indexes (0 is A, 1 is B, 2 is C) in order of
/// preference, plus a flag at bit 6 that says signaling NaNs are preferred first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Float3NanPropRule(pub u8);

const fn proprule(x: u8, y: u8, z: u8) -> u8 {
    x | (y << 2) | (z << 4)
}

impl Float3NanPropRule {
    /// The flag at bit 6: prefer a signaling NaN.
    pub const SNAN_MASK: u8 = 0x40;
    /// `float_3nan_prop_none`: not set. Picking a NaN with this rule panics.
    pub const NONE: Self = Self(0);
    /// `float_3nan_prop_abc`.
    pub const ABC: Self = Self(proprule(0, 1, 2));
    /// `float_3nan_prop_acb`.
    pub const ACB: Self = Self(proprule(0, 2, 1));
    /// `float_3nan_prop_bac`.
    pub const BAC: Self = Self(proprule(1, 0, 2));
    /// `float_3nan_prop_bca`.
    pub const BCA: Self = Self(proprule(1, 2, 0));
    /// `float_3nan_prop_cab`.
    pub const CAB: Self = Self(proprule(2, 0, 1));
    /// `float_3nan_prop_cba`.
    pub const CBA: Self = Self(proprule(2, 1, 0));
    /// `float_3nan_prop_s_abc`.
    pub const S_ABC: Self = Self(Self::SNAN_MASK | proprule(0, 1, 2));
    /// `float_3nan_prop_s_acb`.
    pub const S_ACB: Self = Self(Self::SNAN_MASK | proprule(0, 2, 1));
    /// `float_3nan_prop_s_bac`.
    pub const S_BAC: Self = Self(Self::SNAN_MASK | proprule(1, 0, 2));
    /// `float_3nan_prop_s_bca`.
    pub const S_BCA: Self = Self(Self::SNAN_MASK | proprule(1, 2, 0));
    /// `float_3nan_prop_s_cab`.
    pub const S_CAB: Self = Self(Self::SNAN_MASK | proprule(2, 0, 1));
    /// `float_3nan_prop_s_cba`.
    pub const S_CBA: Self = Self(Self::SNAN_MASK | proprule(2, 1, 0));
}

/// What `0 * Inf + NaN` returns, QEMU's `FloatInfZeroNaNRule`.
///
/// The low two bits pick the result and `SUPPRESS_INVALID` can be or-ed in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct InfZeroNanRule(pub u8);

impl InfZeroNanRule {
    /// `float_infzeronan_none`: not set. Hitting the case with this rule panics.
    pub const NONE: Self = Self(0);
    /// `float_infzeronan_dnan_never`: return the input NaN.
    pub const DNAN_NEVER: Self = Self(1);
    /// `float_infzeronan_dnan_always`: return the default NaN.
    pub const DNAN_ALWAYS: Self = Self(2);
    /// `float_infzeronan_dnan_if_qnan`: return the default NaN if the input NaN is quiet.
    pub const DNAN_IF_QNAN: Self = Self(3);
    /// `float_infzeronan_suppress_invalid`: do not raise invalid for this case.
    pub const SUPPRESS_INVALID: u8 = 1 << 2;

    /// This rule with invalid suppressed.
    pub const fn suppress_invalid(self) -> Self {
        Self(self.0 | Self::SUPPRESS_INVALID)
    }
}

/// The floatx80 behaviour flags, QEMU's `FloatX80Behaviour`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FloatX80Behaviour(pub u8);

impl FloatX80Behaviour {
    /// `floatx80_default_inf_int_bit_is_zero`: the default infinity has a clear integer bit
    /// (m68k).
    pub const DEFAULT_INF_INT_BIT_IS_ZERO: u8 = 1;
    /// `floatx80_pseudo_inf_valid`: pseudo infinities are valid inputs.
    pub const PSEUDO_INF_VALID: u8 = 2;
    /// `floatx80_pseudo_nan_valid`: pseudo NaNs are valid inputs.
    pub const PSEUDO_NAN_VALID: u8 = 4;
    /// `floatx80_unnormal_valid`: unnormals are valid inputs.
    pub const UNNORMAL_VALID: u8 = 8;
    /// `floatx80_pseudo_denormal_valid`: pseudo denormals are valid and may be produced (m68k).
    pub const PSEUDO_DENORMAL_VALID: u8 = 16;
}

/// The exception flags, QEMU's `float_flag_*` values.
pub mod flags {
    /// `float_flag_invalid`.
    pub const INVALID: u16 = 0x0001;
    /// `float_flag_divbyzero`.
    pub const DIVBYZERO: u16 = 0x0002;
    /// `float_flag_overflow`.
    pub const OVERFLOW: u16 = 0x0004;
    /// `float_flag_underflow`.
    pub const UNDERFLOW: u16 = 0x0008;
    /// `float_flag_inexact`.
    pub const INEXACT: u16 = 0x0010;
    /// `float_flag_input_denormal_flushed`: an input denormal was flushed to zero.
    pub const INPUT_DENORMAL_FLUSHED: u16 = 0x0020;
    /// `float_flag_output_denormal_flushed`: a denormal result was flushed to zero.
    pub const OUTPUT_DENORMAL_FLUSHED: u16 = 0x0040;
    /// `float_flag_invalid_isi`: infinity minus infinity.
    pub const INVALID_ISI: u16 = 0x0080;
    /// `float_flag_invalid_imz`: infinity times zero.
    pub const INVALID_IMZ: u16 = 0x0100;
    /// `float_flag_invalid_idi`: infinity divided by infinity.
    pub const INVALID_IDI: u16 = 0x0200;
    /// `float_flag_invalid_zdz`: zero divided by zero.
    pub const INVALID_ZDZ: u16 = 0x0400;
    /// `float_flag_invalid_sqrt`: square root of a negative number.
    pub const INVALID_SQRT: u16 = 0x0800;
    /// `float_flag_invalid_cvti`: a float to integer conversion out of range.
    pub const INVALID_CVTI: u16 = 0x1000;
    /// `float_flag_invalid_snan`: a signaling NaN input.
    pub const INVALID_SNAN: u16 = 0x2000;
    /// `float_flag_input_denormal_used`: an input denormal was used without being flushed.
    pub const INPUT_DENORMAL_USED: u16 = 0x4000;
}

/// The muladd flags, QEMU's `float_muladd_*`.
pub mod muladd {
    /// `float_muladd_negate_c`.
    pub const NEGATE_C: u32 = 1;
    /// `float_muladd_negate_product`.
    pub const NEGATE_PRODUCT: u32 = 2;
    /// `float_muladd_negate_result`, applied after rounding and never to a NaN.
    pub const NEGATE_RESULT: u32 = 4;
    /// `float_muladd_suppress_add_product_zero`: a zero product plus C returns C as is.
    pub const SUPPRESS_ADD_PRODUCT_ZERO: u32 = 8;
}

/// The result of a comparison, QEMU's `FloatRelation`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i8)]
pub enum FloatRelation {
    /// `float_relation_less`.
    Less = -1,
    /// `float_relation_equal`.
    Equal = 0,
    /// `float_relation_greater`.
    Greater = 1,
    /// `float_relation_unordered`.
    Unordered = 2,
}

/// The floating point environment, QEMU's `float_status`.
///
/// A zeroed status (the `Default`) is what QEMU gets from a zeroed struct: nearest even, no
/// flags, and none of the NaN rules set. Operations that need a rule that is not set panic,
/// where QEMU asserts. Use one of the presets to get a status a target would really use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FloatStatus {
    /// `float_exception_flags`, an or of [`flags`] values. Operations only ever set bits.
    pub exception_flags: u16,
    /// `float_rounding_mode`.
    pub rounding_mode: RoundMode,
    /// `floatx80_rounding_precision`.
    pub floatx80_rounding_precision: FloatX80RoundPrec,
    /// `flush_to_zero`: denormal results become zero.
    pub flush_to_zero: bool,
    /// `flush_inputs_to_zero`: denormal inputs are treated as zero.
    pub flush_inputs_to_zero: bool,
    /// `default_nan_mode`: every NaN result is the default NaN.
    pub default_nan_mode: bool,
    /// `rebias_overflow`: an overflowing result is rebiased into range (PowerPC).
    pub rebias_overflow: bool,
    /// `rebias_underflow`: an underflowing result is rebiased into range (PowerPC).
    pub rebias_underflow: bool,
    /// `tininess_before_rounding`: underflow is detected before rounding.
    pub tininess_before_rounding: bool,
    /// `ftz_before_rounding`: flush to zero looks at the result before rounding.
    pub ftz_before_rounding: bool,
    /// `float_snan_rule`.
    pub snan_rule: SnanRule,
    /// `float_2nan_prop_rule`.
    pub float_2nan_prop_rule: Float2NanPropRule,
    /// `float_3nan_prop_rule`.
    pub float_3nan_prop_rule: Float3NanPropRule,
    /// `float_infzeronan_rule`.
    pub float_infzeronan_rule: InfZeroNanRule,
    /// `floatx80_behaviour`.
    pub floatx80_behaviour: FloatX80Behaviour,
    /// `default_nan_pattern`: bit 7 is the sign, bits 6 to 0 are the top of the fraction, and
    /// bit 0 is copied into every lower fraction bit. It must not be zero when a default NaN is
    /// made.
    pub default_nan_pattern: u8,
}

impl FloatStatus {
    /// `float_raise`: or `f` into the exception flags.
    #[inline]
    pub fn raise(&mut self, f: u16) {
        self.exception_flags |= f;
    }

    /// `get_float_exception_flags`.
    #[inline]
    pub fn flags(&self) -> u16 {
        self.exception_flags
    }

    /// `set_float_exception_flags(0, s)`: clear the flags and return what they were.
    #[inline]
    pub fn take_flags(&mut self) -> u16 {
        core::mem::take(&mut self.exception_flags)
    }

    /// The status QEMU's x86 target uses for SSE (`env->sse_status` in `cpu_init_fp_statuses`),
    /// with MXCSR at its reset value: nearest even, DAZ and FZ clear.
    pub const fn x86_sse() -> Self {
        FloatStatus {
            exception_flags: 0,
            rounding_mode: RoundMode::NearestEven,
            floatx80_rounding_precision: FloatX80RoundPrec::X,
            flush_to_zero: false,
            flush_inputs_to_zero: false,
            default_nan_mode: false,
            rebias_overflow: false,
            rebias_underflow: false,
            tininess_before_rounding: false,
            ftz_before_rounding: false,
            snan_rule: SnanRule::BitIsZero,
            float_2nan_prop_rule: Float2NanPropRule::X87,
            float_3nan_prop_rule: Float3NanPropRule::ABC,
            float_infzeronan_rule: InfZeroNanRule::DNAN_NEVER.suppress_invalid(),
            floatx80_behaviour: FloatX80Behaviour(0),
            default_nan_pattern: 0b1100_0000,
        }
    }

    /// The status QEMU's x86 target uses for the x87 unit (`env->fp_status`). The three NaN
    /// and inf-zero rules are left unset there, as in QEMU, since x87 has no fused multiply
    /// add; this preset copies the SSE ones so that every operation works.
    pub const fn x87() -> Self {
        Self::x86_sse()
    }

    /// The status Arm uses with FPCR.AH clear (`arm_set_default_fp_behaviours`), with FPCR at
    /// zero: nearest even, no flush to zero, no default NaN mode.
    pub const fn arm() -> Self {
        FloatStatus {
            exception_flags: 0,
            rounding_mode: RoundMode::NearestEven,
            floatx80_rounding_precision: FloatX80RoundPrec::X,
            flush_to_zero: false,
            flush_inputs_to_zero: false,
            default_nan_mode: false,
            rebias_overflow: false,
            rebias_underflow: false,
            tininess_before_rounding: true,
            ftz_before_rounding: true,
            snan_rule: SnanRule::BitIsZero,
            float_2nan_prop_rule: Float2NanPropRule::SAb,
            float_3nan_prop_rule: Float3NanPropRule::S_CAB,
            float_infzeronan_rule: InfZeroNanRule::DNAN_IF_QNAN,
            floatx80_behaviour: FloatX80Behaviour(0),
            default_nan_pattern: 0b0100_0000,
        }
    }

    /// The status Arm uses with FPCR.AH set (`arm_set_ah_fp_behaviours`).
    pub const fn arm_ah() -> Self {
        FloatStatus {
            exception_flags: 0,
            rounding_mode: RoundMode::NearestEven,
            floatx80_rounding_precision: FloatX80RoundPrec::X,
            flush_to_zero: false,
            flush_inputs_to_zero: false,
            default_nan_mode: false,
            rebias_overflow: false,
            rebias_underflow: false,
            tininess_before_rounding: false,
            ftz_before_rounding: false,
            snan_rule: SnanRule::BitIsZero,
            float_2nan_prop_rule: Float2NanPropRule::Ab,
            float_3nan_prop_rule: Float3NanPropRule::ABC,
            float_infzeronan_rule: InfZeroNanRule::DNAN_NEVER.suppress_invalid(),
            floatx80_behaviour: FloatX80Behaviour(0),
            default_nan_pattern: 0b1100_0000,
        }
    }

    /// The Arm "standard FPSCR value" status (`FPST_STD`), used by Neon on AArch32: flush to
    /// zero, flush inputs to zero and default NaN mode, over the default Arm rules.
    pub const fn arm_standard() -> Self {
        let mut s = Self::arm();
        s.flush_to_zero = true;
        s.flush_inputs_to_zero = true;
        s.default_nan_mode = true;
        s
    }
}
