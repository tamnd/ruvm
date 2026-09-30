// SPDX-License-Identifier: GPL-2.0-or-later

//! Moving between the packed formats and [`Parts`], the `*_unpack_raw`, `*_pack_raw`,
//! `*_unpack_canonical` and `*_round_pack_canonical` functions of `fpu/softfloat.c`.

use crate::frac::{Frac, Frac64, Frac128, FracN};
use crate::parts::{
    FLOAT128_PARAMS, FLOATX80_PARAMS, FloatClass, FloatFmt, Parts, Parts64, Parts128,
    parts_is_snan_frac,
};
use crate::status::{FloatStatus, FloatX80Behaviour, FloatX80RoundPrec, flags};

/// `unpack_raw64`.
#[inline]
pub(crate) fn unpack_raw64(fmt: &FloatFmt, raw: u64) -> Parts64 {
    let f = fmt.frac_size;
    let e = fmt.exp_size;
    Parts {
        cls: FloatClass::Unclassified,
        sign: (raw >> (f + e)) & 1 != 0,
        exp: ((raw >> f) & ((1u64 << e) - 1)) as i32,
        frac: Frac64(raw & ((1u64 << f) - 1)),
    }
}

/// `pack_raw64`.
#[inline]
pub(crate) fn pack_raw64(p: &Parts64, fmt: &FloatFmt) -> u64 {
    let f = fmt.frac_size;
    let e = fmt.exp_size;
    let emask = (1u64 << e) - 1;
    let fmask = (1u64 << f) - 1;
    (u64::from(p.sign) << (f + e)) | (((p.exp as u64) & emask) << f) | (p.frac.0 & fmask)
}

/// `float16a_unpack_canonical` and the `floatN_unpack_canonical` functions: unpack with the
/// `raw` layout and canonicalize with `canon`, which differ only for Arm alternative half
/// precision.
#[inline]
pub(crate) fn unpack_canonical64(
    raw: u64,
    raw_fmt: &FloatFmt,
    canon: &FloatFmt,
    s: &mut FloatStatus,
) -> Parts64 {
    let mut p = unpack_raw64(raw_fmt, raw);
    p.canonicalize(s, canon);
    p
}

/// `float16a_round_pack_canonical` and the `floatN_round_pack_canonical` functions.
#[inline]
pub(crate) fn round_pack64(
    p: &mut Parts64,
    s: &mut FloatStatus,
    round: &FloatFmt,
    raw_fmt: &FloatFmt,
) -> u64 {
    p.uncanon(s, round);
    pack_raw64(p, raw_fmt)
}

/// `float128_unpack_raw`.
#[inline]
pub(crate) fn float128_unpack_raw(hi: u64, lo: u64) -> Parts128 {
    let f = FLOAT128_PARAMS.frac_size - 64;
    let e = FLOAT128_PARAMS.exp_size;
    Parts {
        cls: FloatClass::Unclassified,
        sign: (hi >> (f + e)) & 1 != 0,
        exp: ((hi >> f) & ((1u64 << e) - 1)) as i32,
        frac: Frac128 { hi: hi & ((1u64 << f) - 1), lo },
    }
}

/// `float128_pack_raw`, as (high, low).
#[inline]
pub(crate) fn float128_pack_raw(p: &Parts128) -> (u64, u64) {
    let f = FLOAT128_PARAMS.frac_size - 64;
    let e = FLOAT128_PARAMS.exp_size;
    let hi = (u64::from(p.sign) << (f + e))
        | (((p.exp as u64) & ((1u64 << e) - 1)) << f)
        | (p.frac.hi & ((1u64 << f) - 1));
    (hi, p.frac.lo)
}

/// `float128_unpack_canonical`.
#[inline]
pub(crate) fn float128_unpack_canonical(hi: u64, lo: u64, s: &mut FloatStatus) -> Parts128 {
    let mut p = float128_unpack_raw(hi, lo);
    p.canonicalize(s, &FLOAT128_PARAMS);
    p
}

/// `float128_round_pack_canonical`.
#[inline]
pub(crate) fn float128_round_pack(p: &mut Parts128, s: &mut FloatStatus) -> (u64, u64) {
    p.uncanon(s, &FLOAT128_PARAMS);
    float128_pack_raw(p)
}

/// `floatx80_invalid_encoding`.
#[inline]
pub(crate) fn floatx80_invalid_encoding(high: u16, low: u64, s: &FloatStatus) -> bool {
    let rule = s.floatx80_behaviour.0;
    if low >> 63 != 0 || high & 0x7fff == 0 {
        return false;
    }
    if high & 0x7fff == 0x7fff {
        if low != 0 {
            rule & FloatX80Behaviour::PSEUDO_NAN_VALID == 0
        } else {
            rule & FloatX80Behaviour::PSEUDO_INF_VALID == 0
        }
    } else {
        rule & FloatX80Behaviour::UNNORMAL_VALID == 0
    }
}

/// `floatx80_unpack_canonical`: `None` for an invalid encoding, after raising invalid.
pub(crate) fn floatx80_unpack_canonical(
    high: u16,
    low: u64,
    s: &mut FloatStatus,
) -> Option<Parts128> {
    if floatx80_invalid_encoding(high, low, s) {
        s.raise(flags::INVALID);
        return None;
    }
    let fmt = &FLOATX80_PARAMS[FloatX80RoundPrec::X as usize];
    let mut p = Parts {
        cls: FloatClass::Unclassified,
        sign: (high >> 15) & 1 != 0,
        exp: i32::from(high & 0x7fff),
        frac: Frac128 { hi: low, lo: 0 },
    };
    if p.exp != fmt.exp_max {
        p.canonicalize(s, fmt);
    } else {
        p.frac.hi &= (1u64 << 63) - 1;
        p.cls = if p.frac.hi == 0 {
            FloatClass::Inf
        } else if parts_is_snan_frac(p.frac.hi, s) {
            FloatClass::SNan
        } else {
            FloatClass::QNan
        };
    }
    Some(p)
}

/// `packFloatx80`: the exponent is added, not or-ed, as in QEMU.
#[inline]
pub(crate) fn pack_floatx80(sign: bool, exp: i32, sig: u64) -> (u16, u64) {
    ((u16::from(sign) << 15).wrapping_add(exp as u16), sig)
}

/// `floatx80_round_pack_canonical`.
pub(crate) fn floatx80_round_pack(p: &mut Parts128, s: &mut FloatStatus) -> (u16, u64) {
    let prec = s.floatx80_rounding_precision;
    let fmt = &FLOATX80_PARAMS[prec as usize];
    let inf_frac = if s.floatx80_behaviour.0 & FloatX80Behaviour::DEFAULT_INF_INT_BIT_IS_ZERO != 0 {
        0
    } else {
        1u64 << 63
    };
    let (frac, exp) = match p.cls {
        FloatClass::Normal | FloatClass::Denormal => {
            let (frac, exp) = if prec == FloatX80RoundPrec::X {
                p.uncanon_normal(s, fmt);
                (p.frac.hi, p.exp)
            } else {
                let mut p64 = Parts {
                    cls: p.cls,
                    sign: p.sign,
                    exp: p.exp,
                    frac: <Frac64 as FracN>::truncjam(&p.frac),
                };
                p64.uncanon_normal(s, fmt);
                (p64.frac.0, p64.exp)
            };
            if exp != fmt.exp_max { (frac, exp) } else { (inf_frac, fmt.exp_max) }
        }
        FloatClass::Inf => (inf_frac, fmt.exp_max),
        FloatClass::Zero => (0, 0),
        FloatClass::SNan | FloatClass::QNan => (p.frac.hi | (1u64 << 63), fmt.exp_max),
        FloatClass::Unclassified => unreachable!(),
    };
    pack_floatx80(p.sign, exp, frac)
}

/// `parts128_to_parts64`.
pub(crate) fn parts128_to_parts64(b: &Parts128, s: &mut FloatStatus) -> Parts64 {
    let mut r = Parts { cls: b.cls, sign: b.sign, exp: b.exp, frac: Frac64(0) };
    match r.cls {
        FloatClass::Denormal | FloatClass::Normal => {
            if r.cls == FloatClass::Denormal {
                s.raise(flags::INPUT_DENORMAL_USED);
            }
            r.frac = <Frac64 as FracN>::truncjam(&b.frac);
        }
        FloatClass::SNan | FloatClass::QNan => {
            r.frac = Frac64(b.frac.hi);
            r = r.return_nan(s);
        }
        _ => {}
    }
    r
}

/// `parts64_to_parts128`.
pub(crate) fn parts64_to_parts128(b: &Parts64, s: &mut FloatStatus) -> Parts128 {
    let mut r =
        Parts { cls: b.cls, sign: b.sign, exp: b.exp, frac: Frac128 { hi: b.frac.0, lo: 0 } };
    match r.cls {
        FloatClass::QNan | FloatClass::SNan => r = r.return_nan(s),
        FloatClass::Denormal => s.raise(flags::INPUT_DENORMAL_USED),
        _ => {}
    }
    r
}

/// `partsN_float_to_float`.
pub(crate) fn parts_float_to_float<F: Frac>(a: &mut Parts<F>, s: &mut FloatStatus) {
    if a.cls.is_nan() {
        *a = a.return_nan(s);
    }
    if a.cls == FloatClass::Denormal {
        s.raise(flags::INPUT_DENORMAL_USED);
    }
}

/// `parts_float_to_ahp`.
pub(crate) fn parts_float_to_ahp(a: &mut Parts64, s: &mut FloatStatus) {
    use crate::parts::FLOAT16_PARAMS_AHP as AHP;
    match a.cls {
        FloatClass::SNan | FloatClass::QNan => {
            if a.cls == FloatClass::SNan {
                s.raise(flags::INVALID_SNAN);
            }
            s.raise(flags::INVALID);
            a.cls = FloatClass::Zero;
        }
        FloatClass::Inf => {
            s.raise(flags::INVALID);
            a.cls = FloatClass::Normal;
            a.exp = AHP.exp_max;
            let len = AHP.frac_size + 1;
            a.frac = Frac64(((1u64 << len) - 1) << AHP.frac_shift);
        }
        FloatClass::Denormal => s.raise(flags::INPUT_DENORMAL_USED),
        FloatClass::Normal | FloatClass::Zero => {}
        FloatClass::Unclassified => unreachable!(),
    }
}
