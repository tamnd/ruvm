// SPDX-License-Identifier: GPL-2.0-or-later

//! Unit tests for special cases, with expected values worked out from IEEE 754 and QEMU's
//! documented target behaviour rather than taken from the implementation.

use ruvm_softfloat::{
    BFloat16, Float16, Float32, Float64, Float128, FloatRelation, FloatStatus, FloatX80,
    FloatX80RoundPrec, RoundMode, flags, minmax, muladd,
};

const F32_ONE: Float32 = Float32(0x3f80_0000);
const F32_TWO: Float32 = Float32(0x4000_0000);
const F32_INF: Float32 = Float32(0x7f80_0000);
const F32_SNAN: Float32 = Float32(0x7f80_0001);
const F32_QNAN_A: Float32 = Float32(0x7fc0_0001);
const F32_QNAN_B: Float32 = Float32(0x7fc0_0002);
const F32_MIN_NORMAL: Float32 = Float32(0x0080_0000);
const F32_MIN_DENORMAL: Float32 = Float32(0x0000_0001);
const F32_MAX: Float32 = Float32(0x7f7f_ffff);

#[test]
fn default_nans() {
    // x86 has a negative default NaN, Arm a positive one.
    assert_eq!(Float32::default_nan(&FloatStatus::x86_sse()), Float32(0xffc0_0000));
    assert_eq!(Float32::default_nan(&FloatStatus::arm()), Float32(0x7fc0_0000));
    assert_eq!(Float64::default_nan(&FloatStatus::x86_sse()), Float64(0xfff8_0000_0000_0000));
    assert_eq!(Float64::default_nan(&FloatStatus::arm()), Float64(0x7ff8_0000_0000_0000));
    assert_eq!(Float16::default_nan(&FloatStatus::arm()), Float16(0x7e00));
    assert_eq!(BFloat16::default_nan(&FloatStatus::arm()), BFloat16(0x7fc0));
    assert_eq!(
        FloatX80::default_nan(&FloatStatus::x87()),
        FloatX80::new(0xffff, 0xc000_0000_0000_0000)
    );
    assert_eq!(
        Float128::default_nan(&FloatStatus::arm()),
        Float128::new(0x7fff_8000_0000_0000, 0)
    );
}

#[test]
fn invalid_operations_give_the_default_nan() {
    for (s, dnan) in [(FloatStatus::x86_sse(), 0xffc0_0000), (FloatStatus::arm(), 0x7fc0_0000)] {
        let mut st = s;
        assert_eq!(F32_INF.sub(F32_INF, &mut st), Float32(dnan));
        assert_eq!(st.flags(), flags::INVALID | flags::INVALID_ISI);

        let mut st = s;
        assert_eq!(Float32(0).div(Float32(0), &mut st), Float32(dnan));
        assert_eq!(st.flags(), flags::INVALID | flags::INVALID_ZDZ);

        let mut st = s;
        assert_eq!(F32_ONE.chs().sqrt(&mut st), Float32(dnan));
        assert_eq!(st.flags(), flags::INVALID | flags::INVALID_SQRT);

        let mut st = s;
        assert_eq!(F32_INF.mul(Float32(0), &mut st), Float32(dnan));
        assert_eq!(st.flags(), flags::INVALID | flags::INVALID_IMZ);
    }
}

#[test]
fn two_nan_propagation() {
    // x86: the x87 rule, larger significand wins, quieted.
    let mut s = FloatStatus::x86_sse();
    assert_eq!(F32_QNAN_A.add(F32_QNAN_B, &mut s), F32_QNAN_B);
    assert_eq!(F32_QNAN_B.add(F32_QNAN_A, &mut s), F32_QNAN_B);
    assert_eq!(s.flags(), 0);

    // Arm: a signaling NaN first, then A.
    let mut s = FloatStatus::arm();
    assert_eq!(F32_QNAN_A.add(F32_QNAN_B, &mut s), F32_QNAN_A);
    assert_eq!(F32_QNAN_B.add(F32_QNAN_A, &mut s), F32_QNAN_B);
    assert_eq!(s.flags(), 0);
    assert_eq!(F32_QNAN_A.add(F32_SNAN, &mut s), Float32(0x7fc0_0001));
    let mut s = FloatStatus::arm();
    assert_eq!(F32_QNAN_B.mul(F32_SNAN, &mut s), Float32(0x7fc0_0001));
    assert_eq!(s.flags(), flags::INVALID | flags::INVALID_SNAN);

    // Arm with FPCR.AH: A if it is a NaN, whatever its kind.
    let mut s = FloatStatus::arm_ah();
    assert_eq!(F32_QNAN_B.mul(F32_SNAN, &mut s), F32_QNAN_B);
    assert_eq!(s.flags(), flags::INVALID | flags::INVALID_SNAN);

    // Default NaN mode ignores the inputs.
    let mut s = FloatStatus::arm_standard();
    assert_eq!(F32_QNAN_A.add(F32_ONE, &mut s), Float32(0x7fc0_0000));
}

#[test]
fn fma_nan_rules() {
    let zero = Float32(0);
    // Arm: 0 * inf + qnan is the default NaN and raises invalid.
    let mut s = FloatStatus::arm();
    assert_eq!(zero.muladd(F32_INF, F32_QNAN_A, 0, &mut s), Float32(0x7fc0_0000));
    assert_eq!(s.flags(), flags::INVALID | flags::INVALID_IMZ);

    // x86: the input NaN, and no invalid.
    let mut s = FloatStatus::x86_sse();
    assert_eq!(zero.muladd(F32_INF, F32_QNAN_A, 0, &mut s), F32_QNAN_A);
    assert_eq!(s.flags(), 0);

    // Arm picks C first among quiet NaNs, x86 picks A.
    let mut s = FloatStatus::arm();
    assert_eq!(F32_QNAN_A.muladd(F32_ONE, F32_QNAN_B, 0, &mut s), F32_QNAN_B);
    let mut s = FloatStatus::x86_sse();
    assert_eq!(F32_QNAN_A.muladd(F32_ONE, F32_QNAN_B, 0, &mut s), F32_QNAN_A);
}

#[test]
fn muladd_flags() {
    let mut s = FloatStatus::arm();
    let three = Float32(0x4040_0000);
    // 1 * 2 + 3 = 5, with each negation.
    assert_eq!(F32_ONE.muladd(F32_TWO, three, 0, &mut s), Float32(0x40a0_0000));
    assert_eq!(F32_ONE.muladd(F32_TWO, three, muladd::NEGATE_C, &mut s), Float32(0xbf80_0000));
    assert_eq!(F32_ONE.muladd(F32_TWO, three, muladd::NEGATE_PRODUCT, &mut s), F32_ONE);
    assert_eq!(F32_ONE.muladd(F32_TWO, three, muladd::NEGATE_RESULT, &mut s), Float32(0xc0a0_0000));
    // Negate result is applied after rounding, so x - x = +0 becomes -0.
    let r = F32_ONE.muladd(F32_ONE, F32_ONE, muladd::NEGATE_C | muladd::NEGATE_RESULT, &mut s);
    assert_eq!(r, Float32(0x8000_0000));
    // The product 0 * 1 = +0 with C = -0 is +0, unless the addition is suppressed.
    let neg_zero = Float32(0x8000_0000);
    assert_eq!(Float32(0).muladd(F32_ONE, neg_zero, 0, &mut s), Float32(0));
    let r = Float32(0).muladd(F32_ONE, neg_zero, muladd::SUPPRESS_ADD_PRODUCT_ZERO, &mut s);
    assert_eq!(r, neg_zero);
    // The scaled form scales before rounding.
    assert_eq!(F32_ONE.muladd_scalbn(F32_ONE, F32_ONE, 3, 0, &mut s), Float32(0x4180_0000));
    assert_eq!(s.flags(), 0);
}

#[test]
fn rounding_modes() {
    // 1 + 2^-24 is halfway between 1 and the next float32.
    let half_ulp = Float32(0x3380_0000);
    let cases = [
        (RoundMode::NearestEven, 0x3f80_0000),
        (RoundMode::TiesAway, 0x3f80_0001),
        (RoundMode::Up, 0x3f80_0001),
        (RoundMode::Down, 0x3f80_0000),
        (RoundMode::ToZero, 0x3f80_0000),
        (RoundMode::ToOdd, 0x3f80_0001),
        (RoundMode::ToOddInf, 0x3f80_0001),
        (RoundMode::NearestEvenMax, 0x3f80_0000),
    ];
    for (rm, want) in cases {
        let mut s = FloatStatus::arm();
        s.rounding_mode = rm;
        assert_eq!(F32_ONE.add(half_ulp, &mut s), Float32(want), "{rm:?}");
        assert_eq!(s.flags(), flags::INEXACT, "{rm:?}");
    }
}

#[test]
fn overflow_by_rounding_mode() {
    let cases = [
        (RoundMode::NearestEven, F32_INF),
        (RoundMode::Up, F32_INF),
        (RoundMode::Down, F32_MAX),
        (RoundMode::ToZero, F32_MAX),
        (RoundMode::ToOdd, F32_MAX),
        (RoundMode::ToOddInf, F32_INF),
        (RoundMode::NearestEvenMax, F32_MAX),
    ];
    for (rm, want) in cases {
        let mut s = FloatStatus::x86_sse();
        s.rounding_mode = rm;
        assert_eq!(F32_MAX.mul(F32_TWO, &mut s), want, "{rm:?}");
        assert_eq!(s.flags(), flags::OVERFLOW | flags::INEXACT, "{rm:?}");
    }
}

#[test]
fn tininess_before_and_after_rounding() {
    // 2^-126 * (1 - 2^-25) rounds up to MIN_NORMAL: tiny before rounding, not after.
    let v = Float64(0x380f_ffff_f000_0000);
    let mut arm = FloatStatus::arm();
    assert!(arm.tininess_before_rounding);
    assert_eq!(v.to_float32(&mut arm), F32_MIN_NORMAL);
    assert_eq!(arm.flags(), flags::UNDERFLOW | flags::INEXACT);
    let mut x86 = FloatStatus::x86_sse();
    assert!(!x86.tininess_before_rounding);
    assert_eq!(v.to_float32(&mut x86), F32_MIN_NORMAL);
    assert_eq!(x86.flags(), flags::INEXACT);
}

#[test]
fn flush_to_zero() {
    // Output flushing: a denormal result becomes a signed zero.
    let mut s = FloatStatus::arm();
    s.flush_to_zero = true;
    let half = Float32(0x3f00_0000);
    // QEMU raises only the flushed flag, and the target maps it to its underflow bit.
    assert_eq!(F32_MIN_NORMAL.mul(half, &mut s), Float32(0));
    assert_eq!(s.flags(), flags::OUTPUT_DENORMAL_FLUSHED);

    // Input flushing: a denormal input is zero.
    let mut s = FloatStatus::arm();
    s.flush_inputs_to_zero = true;
    assert_eq!(F32_MIN_DENORMAL.add(F32_MIN_DENORMAL, &mut s), Float32(0));
    assert_eq!(s.flags(), flags::INPUT_DENORMAL_FLUSHED);

    // Without flushing the denormal is used and reported.
    let mut s = FloatStatus::arm();
    assert_eq!(F32_MIN_DENORMAL.add(F32_MIN_DENORMAL, &mut s), Float32(2));
    assert_eq!(s.flags(), flags::INPUT_DENORMAL_USED);
}

#[test]
fn signaling_nan_bit() {
    let s = FloatStatus::arm();
    assert!(F32_SNAN.is_signaling_nan(&s));
    assert!(!F32_SNAN.is_quiet_nan(&s));
    assert!(F32_QNAN_A.is_quiet_nan(&s));
    assert_eq!(F32_SNAN.silence_nan(&s), Float32(0x7fc0_0001));

    let mut mips = FloatStatus::arm();
    mips.snan_rule = ruvm_softfloat::SnanRule::BitIsOne;
    assert!(F32_QNAN_A.is_signaling_nan(&mips));
    assert!(F32_SNAN.is_quiet_nan(&mips));
}

#[test]
fn compare() {
    let mut s = FloatStatus::x86_sse();
    assert_eq!(F32_ONE.compare(F32_TWO, &mut s), FloatRelation::Less);
    assert_eq!(F32_TWO.compare(F32_ONE, &mut s), FloatRelation::Greater);
    assert_eq!(Float32(0).compare(Float32(0x8000_0000), &mut s), FloatRelation::Equal);
    assert_eq!(s.flags(), 0);
    assert_eq!(F32_QNAN_A.compare_quiet(F32_ONE, &mut s), FloatRelation::Unordered);
    assert_eq!(s.flags(), 0);
    assert_eq!(F32_QNAN_A.compare(F32_ONE, &mut s), FloatRelation::Unordered);
    assert_eq!(s.take_flags(), flags::INVALID);
    assert_eq!(F32_SNAN.compare_quiet(F32_ONE, &mut s), FloatRelation::Unordered);
    assert_eq!(s.take_flags(), flags::INVALID | flags::INVALID_SNAN);
    assert!(F32_ONE.lt(F32_TWO, &mut s));
    assert!(F32_ONE.le(F32_ONE, &mut s));
    assert!(!F32_QNAN_A.eq_quiet(F32_QNAN_A, &mut s));
    assert!(F32_QNAN_A.unordered_quiet(F32_ONE, &mut s));
}

#[test]
fn min_max_family() {
    let mut s = FloatStatus::arm();
    let neg_two = F32_TWO.chs();
    assert_eq!(F32_ONE.min(neg_two, &mut s), neg_two);
    assert_eq!(F32_ONE.max(neg_two, &mut s), F32_ONE);
    assert_eq!(F32_ONE.minnummag(neg_two, &mut s), F32_ONE);
    assert_eq!(F32_ONE.maxnummag(neg_two, &mut s), neg_two);
    // -0 is less than +0.
    let (pz, nz) = (Float32(0), Float32(0x8000_0000));
    assert_eq!(pz.min(nz, &mut s), nz);
    assert_eq!(nz.max(pz, &mut s), pz);
    assert_eq!(s.flags(), 0);

    // min and max propagate a quiet NaN, minnum and maxnum drop it.
    assert_eq!(F32_ONE.min(F32_QNAN_A, &mut s), F32_QNAN_A);
    assert_eq!(F32_ONE.minnum(F32_QNAN_A, &mut s), F32_ONE);
    assert_eq!(F32_QNAN_A.maxnum(F32_ONE, &mut s), F32_ONE);
    assert_eq!(s.flags(), 0);

    // minnum returns a NaN for a signaling input, minimumNumber does not.
    assert_eq!(F32_ONE.minnum(F32_SNAN, &mut s), Float32(0x7fc0_0001));
    assert_eq!(s.take_flags(), flags::INVALID | flags::INVALID_SNAN);
    assert_eq!(F32_ONE.minimum_number(F32_SNAN, &mut s), F32_ONE);
    assert_eq!(s.take_flags(), flags::INVALID);
    assert_eq!(F32_SNAN.maximum_number(F32_ONE, &mut s), F32_ONE);

    // The generic entry point with flags.
    assert_eq!(F32_ONE.minmax(neg_two, &mut s, minmax::ISMIN | minmax::ISNUM | minmax::ISMAG), F32_ONE);
}

#[test]
fn integer_conversions() {
    let mut s = FloatStatus::x86_sse();
    let two_and_half = Float64(0x4004_0000_0000_0000);
    assert_eq!(two_and_half.to_i32(&mut s), 2);
    assert_eq!(s.take_flags(), flags::INEXACT);
    assert_eq!(two_and_half.to_i32_scalbn(RoundMode::TiesAway, 0, &mut s), 3);
    assert_eq!(two_and_half.to_i32_scalbn(RoundMode::Down, 0, &mut s), 2);
    assert_eq!(two_and_half.chs().to_i32_scalbn(RoundMode::Down, 0, &mut s), -3);
    assert_eq!(two_and_half.to_i32_scalbn(RoundMode::ToOdd, 0, &mut s), 3);
    assert_eq!(two_and_half.to_i32_scalbn(RoundMode::ToZero, 2, &mut s), 10);
    s.take_flags();

    // Saturation and invalid.
    let big = Float64(0x41f0_0000_0000_0000); // 2^32
    assert_eq!(big.to_i32(&mut s), i32::MAX);
    assert_eq!(s.take_flags(), flags::INVALID | flags::INVALID_CVTI);
    assert_eq!(big.to_u32(&mut s), u32::MAX);
    assert_eq!(s.take_flags(), flags::INVALID | flags::INVALID_CVTI);
    assert_eq!(big.to_i64(&mut s), 1 << 32);
    assert_eq!(s.take_flags(), 0);
    assert_eq!(F32_ONE.chs().to_u32(&mut s), 0);
    assert_eq!(s.take_flags(), flags::INVALID | flags::INVALID_CVTI);
    // -0.5 rounds to -0, which is fine for unsigned.
    assert_eq!(Float32(0xbf00_0000).to_u32(&mut s), 0);
    assert_eq!(s.take_flags(), flags::INEXACT);
    // NaN gives the maximum and invalid, without the cvti flag.
    assert_eq!(F32_QNAN_A.to_i32(&mut s), i32::MAX);
    assert_eq!(s.take_flags(), flags::INVALID);

    // JavaScript style modulo conversion.
    let v = Float64(0x41f0_0000_0000_0005); // 2^32 + 5 * 2^-20, truncates to 2^32
    assert_eq!(v.to_i32_modulo(RoundMode::ToZero, &mut s), 0);
    assert_eq!(s.take_flags(), flags::INVALID | flags::INVALID_CVTI);

    // From integers.
    assert_eq!(Float32::from_i64(-1, &mut s), Float32(0xbf80_0000));
    assert_eq!(Float32::from_u64(u64::MAX, &mut s), Float32(0x5f80_0000));
    assert_eq!(s.take_flags(), flags::INEXACT);
    assert_eq!(Float64::from_i64_scalbn(3, -1, &mut s), Float64(0x3ff8_0000_0000_0000));
    assert_eq!(Float16::from_i32(65520, &mut s), Float16(0x7c00));
    assert_eq!(s.take_flags(), flags::OVERFLOW | flags::INEXACT);
}

#[test]
fn int128_conversions() {
    let mut s = FloatStatus::arm();
    let v = Float128::from_i128(i128::MIN, &mut s);
    assert_eq!(v, Float128::new(0xc07e_0000_0000_0000, 0));
    assert_eq!(v.to_i128(&mut s), i128::MIN);
    assert_eq!(s.take_flags(), 0);
    let v = Float128::from_u128(u128::MAX, &mut s);
    assert_eq!(v, Float128::new(0x407f_0000_0000_0000, 0));
    assert_eq!(s.take_flags(), flags::INEXACT);
    assert_eq!(v.to_u128(&mut s), u128::MAX);
    assert_eq!(s.take_flags(), flags::INVALID | flags::INVALID_CVTI);
}

#[test]
fn half_precision_conversions() {
    let mut s = FloatStatus::arm();
    // 65504 is the largest float16. The Arm alternative format goes to 131008.
    let f = Float32(0x4780_0000); // 65536
    assert_eq!(f.to_float16(true, &mut s), Float16(0x7c00));
    assert_eq!(s.take_flags(), flags::OVERFLOW | flags::INEXACT);
    assert_eq!(f.to_float16(false, &mut s), Float16(0x7c00));
    assert_eq!(s.take_flags(), 0);
    // An infinity has no alternative encoding: it saturates and is invalid.
    assert_eq!(F32_INF.to_float16(false, &mut s), Float16(0x7fff));
    assert_eq!(s.take_flags(), flags::INVALID);
    // A NaN becomes zero, and is invalid.
    assert_eq!(F32_QNAN_A.to_float16(false, &mut s), Float16(0));
    assert_eq!(s.take_flags(), flags::INVALID);
    // Reading back: 0x7c00 is 65536 in the alternative format and infinity in IEEE.
    assert_eq!(Float16(0x7c00).to_float32(false, &mut s), f);
    assert_eq!(Float16(0x7c00).to_float32(true, &mut s), F32_INF);
    assert_eq!(s.take_flags(), 0);
    // A signaling NaN is quieted with invalid.
    assert_eq!(Float16(0x7c01).to_float32(true, &mut s), Float32(0x7fc0_2000));
    assert_eq!(s.take_flags(), flags::INVALID | flags::INVALID_SNAN);
}

#[test]
fn widening_and_narrowing() {
    let mut s = FloatStatus::x86_sse();
    assert_eq!(F32_ONE.to_float64(&mut s), Float64(0x3ff0_0000_0000_0000));
    assert_eq!(F32_ONE.to_floatx80(&mut s), FloatX80::new(0x3fff, 1 << 63));
    assert_eq!(F32_ONE.to_float128(&mut s), Float128::new(0x3fff_0000_0000_0000, 0));
    assert_eq!(F32_ONE.to_bfloat16(&mut s), BFloat16(0x3f80));
    assert_eq!(BFloat16(0x3f80).to_float32(&mut s), F32_ONE);
    assert_eq!(s.flags(), 0);
    // 1 + 2^-52 narrows inexactly.
    assert_eq!(Float64(0x3ff0_0000_0000_0001).to_float32(&mut s), F32_ONE);
    assert_eq!(s.take_flags(), flags::INEXACT);
}

#[test]
fn floatx80_encodings_and_precision() {
    let mut s = FloatStatus::x87();
    let one = FloatX80::new(0x3fff, 1 << 63);
    let unnormal = FloatX80::new(0x3fff, 1 << 62);
    assert!(unnormal.invalid_encoding(&s));
    assert!(!one.invalid_encoding(&s));
    // An invalid encoding is an invalid operation giving the default NaN.
    assert_eq!(unnormal.add(one, &mut s), FloatX80::default_nan(&s));
    assert_eq!(s.take_flags(), flags::INVALID);

    // Rounding precision: 1 + 2^-60 is exact at 64 bits but not at 24 or 53.
    let tiny = FloatX80::new(0x3fff - 60, 1 << 63);
    assert_eq!(one.add(tiny, &mut s), FloatX80::new(0x3fff, (1 << 63) | (1 << 3)));
    assert_eq!(s.take_flags(), 0);
    s.floatx80_rounding_precision = FloatX80RoundPrec::S;
    assert_eq!(one.add(tiny, &mut s), one);
    assert_eq!(s.take_flags(), flags::INEXACT);
    s.floatx80_rounding_precision = FloatX80RoundPrec::D;
    assert_eq!(one.add(tiny, &mut s), one);
    assert_eq!(s.take_flags(), flags::INEXACT);

    // floatx80_round rounds to the current precision.
    let v = FloatX80::new(0x3fff, (1 << 63) | (1 << 23));
    s.floatx80_rounding_precision = FloatX80RoundPrec::S;
    assert_eq!(v.round(&mut s), one);
    assert_eq!(s.take_flags(), flags::INEXACT);

    // The x87 remainder returns the quotient bits.
    let mut s = FloatStatus::x87();
    let seven = FloatX80::new(0x4001, 0xe000_0000_0000_0000);
    let two = FloatX80::new(0x4000, 1 << 63);
    let (r, q) = seven.modrem(two, true, &mut s);
    assert_eq!((r, q), (one, 3));
    let (r, q) = seven.modrem(two, false, &mut s);
    // FPREM1 style: QEMU only reports the quotient for the modulo form.
    assert_eq!((r, q), (one.chs(), 0));
    assert_eq!(s.flags(), 0);
}

#[test]
fn remainder_sqrt_scalbn_log2() {
    let mut s = FloatStatus::arm();
    let five = Float64(0x4014_0000_0000_0000);
    let three = Float64(0x4008_0000_0000_0000);
    // IEEE remainder: 5 - 2 * 3 = -1.
    assert_eq!(five.rem(three, &mut s), Float64(0xbff0_0000_0000_0000));
    let four = Float64(0x4010_0000_0000_0000);
    assert_eq!(four.sqrt(&mut s), Float64(0x4000_0000_0000_0000));
    assert_eq!(s.flags(), 0);
    assert_eq!(Float64(0x3ff0_0000_0000_0000).scalbn(-1075, &mut s), Float64(0));
    assert_eq!(s.take_flags(), flags::UNDERFLOW | flags::INEXACT);
    assert_eq!(Float32(0x4100_0000).log2(&mut s), Float32(0x4040_0000));
    assert_eq!(Float32(0).log2(&mut s), Float32(0xff80_0000));
    assert_eq!(s.take_flags(), flags::DIVBYZERO);
    assert_eq!(F32_ONE.chs().log2(&mut s), Float32(0x7fc0_0000));
    assert_eq!(s.take_flags(), flags::INVALID);
}

#[test]
fn round_to_int() {
    let mut s = FloatStatus::arm();
    let v = Float32(0x3fc0_0000); // 1.5
    for (rm, want) in [
        (RoundMode::NearestEven, 0x4000_0000),
        (RoundMode::Down, 0x3f80_0000),
        (RoundMode::Up, 0x4000_0000),
        (RoundMode::ToZero, 0x3f80_0000),
        (RoundMode::TiesAway, 0x4000_0000),
        (RoundMode::ToOdd, 0x3f80_0000),
    ] {
        s.rounding_mode = rm;
        assert_eq!(v.round_to_int(&mut s), Float32(want), "{rm:?}");
        assert_eq!(s.take_flags(), flags::INEXACT);
    }
    // -0.5 rounds to -0.
    s.rounding_mode = RoundMode::NearestEven;
    assert_eq!(Float32(0xbf00_0000).round_to_int(&mut s), Float32(0x8000_0000));
}

#[test]
fn predicates() {
    let s = FloatStatus::arm();
    assert!(F32_INF.is_infinity() && !F32_INF.is_any_nan());
    assert!(F32_MIN_DENORMAL.is_denormal() && F32_MIN_DENORMAL.is_zero_or_denormal());
    assert!(F32_MIN_NORMAL.is_normal());
    assert!(Float32(0x8000_0000).is_zero() && Float32(0x8000_0000).is_neg());
    assert!(F32_SNAN.is_any_nan() && F32_SNAN.is_signaling_nan(&s));
    assert_eq!(F32_ONE.chs().abs(), F32_ONE);
    let x = FloatX80::new(0x7fff, 1 << 63);
    assert!(x.is_infinity(&s) && !x.is_any_nan());
    let q = Float128::new(0x7fff_8000_0000_0000, 0);
    assert!(q.is_quiet_nan(&s));
}

#[test]
fn hardfloat_quirks() {
    // With inexact already set and nearest even, QEMU runs float64 arithmetic on the host,
    // which ignores rebias_overflow.
    let mut s = FloatStatus::arm();
    s.rebias_overflow = true;
    let big = Float64(0x7fe0_0000_0000_0000);
    let two = Float64(0x4000_0000_0000_0000);
    let r = big.mul(two, &mut s);
    assert_eq!(r, Float64(0x1ff0_0000_0000_0000));
    assert_eq!(s.take_flags(), flags::OVERFLOW);
    s.raise(flags::INEXACT);
    assert_eq!(big.mul(two, &mut s), Float64(0x7ff0_0000_0000_0000));

    // float64_muladd on the host ignores suppress_add_product_zero, float32_muladd does not.
    let mut s = FloatStatus::arm();
    s.raise(flags::INEXACT);
    let fl = muladd::SUPPRESS_ADD_PRODUCT_ZERO;
    let r = Float64(0).muladd(Float64(0x3ff0_0000_0000_0000), Float64(1 << 63), fl, &mut s);
    assert_eq!(r, Float64(0));
    let r = Float32(0).muladd(F32_ONE, Float32(0x8000_0000), fl, &mut s);
    assert_eq!(r, Float32(0x8000_0000));
}
