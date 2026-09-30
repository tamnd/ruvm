// SPDX-License-Identifier: GPL-2.0-or-later

//! The shared half of the differential tests: the request and result records that
//! `tests/c/driver.c` speaks, the evaluation of a request through ruvm-softfloat, and the
//! deterministic generator of random and edge case requests.
//!
//! The record layout and the operation numbers must match `driver.c` exactly.

#![allow(dead_code)]

use ruvm_softfloat::{
    BFloat16, Float2NanPropRule, Float3NanPropRule, Float16, Float32, Float64, Float128,
    FloatRelation, FloatStatus, FloatX80, FloatX80Behaviour, FloatX80RoundPrec, InfZeroNanRule,
    RoundMode, SnanRule, flags, minmax,
};

/// The size of a request record.
pub(crate) const REQ_LEN: usize = 80;
/// The size of a result record.
pub(crate) const RES_LEN: usize = 32;

/// The formats, the high part of an op number.
pub(crate) mod fmt {
    pub(crate) const F16: u16 = 0;
    pub(crate) const BF16: u16 = 1;
    pub(crate) const F32: u16 = 2;
    pub(crate) const F64: u16 = 3;
    pub(crate) const X80: u16 = 4;
    pub(crate) const F128: u16 = 5;
    pub(crate) const FMT_NAMES: [&str; 6] =
        ["float16", "bfloat16", "float32", "float64", "floatx80", "float128"];
}

/// The operations, the low part of an op number, in `driver.c` order.
pub(crate) mod op {
    pub(crate) const ADD: u16 = 0;
    pub(crate) const SUB: u16 = 1;
    pub(crate) const MUL: u16 = 2;
    pub(crate) const DIV: u16 = 3;
    pub(crate) const MULADD: u16 = 4;
    pub(crate) const SQRT: u16 = 5;
    pub(crate) const REM: u16 = 6;
    pub(crate) const SCALBN: u16 = 7;
    pub(crate) const MINMAX: u16 = 8;
    pub(crate) const COMPARE: u16 = 9;
    pub(crate) const COMPARE_QUIET: u16 = 10;
    pub(crate) const ROUND_TO_INT: u16 = 11;
    pub(crate) const LOG2: u16 = 12;
    pub(crate) const TO_I8: u16 = 13;
    pub(crate) const TO_I16: u16 = 14;
    pub(crate) const TO_I32: u16 = 15;
    pub(crate) const TO_I64: u16 = 16;
    pub(crate) const TO_U8: u16 = 17;
    pub(crate) const TO_U16: u16 = 18;
    pub(crate) const TO_U32: u16 = 19;
    pub(crate) const TO_U64: u16 = 20;
    pub(crate) const FROM_I64: u16 = 21;
    pub(crate) const FROM_U64: u16 = 22;
    pub(crate) const TO_I128: u16 = 23;
    pub(crate) const TO_U128: u16 = 24;
    pub(crate) const FROM_I128: u16 = 25;
    pub(crate) const FROM_U128: u16 = 26;
    pub(crate) const TO_F16: u16 = 27;
    pub(crate) const TO_BF16: u16 = 28;
    pub(crate) const TO_F32: u16 = 29;
    pub(crate) const TO_F64: u16 = 30;
    pub(crate) const TO_X80: u16 = 31;
    pub(crate) const TO_F128: u16 = 32;
    pub(crate) const MODULO_I32: u16 = 33;
    pub(crate) const MODULO_I64: u16 = 34;
    pub(crate) const X80_MOD: u16 = 35;
    pub(crate) const X80_ROUND: u16 = 36;
    pub(crate) const X80_ROUND_AND_PACK: u16 = 37;
    pub(crate) const X80_NORM_ROUND_AND_PACK: u16 = 38;
    pub(crate) const PREDICATES: u16 = 39;
    pub(crate) const DEFAULT_NAN: u16 = 40;
    pub(crate) const SILENCE_NAN: u16 = 41;

    pub(crate) const OP_NAMES: [&str; 42] = [
        "add",
        "sub",
        "mul",
        "div",
        "muladd",
        "sqrt",
        "rem",
        "scalbn",
        "minmax",
        "compare",
        "compare_quiet",
        "round_to_int",
        "log2",
        "to_i8",
        "to_i16",
        "to_i32",
        "to_i64",
        "to_u8",
        "to_u16",
        "to_u32",
        "to_u64",
        "from_i64",
        "from_u64",
        "to_i128",
        "to_u128",
        "from_i128",
        "from_u128",
        "to_f16",
        "to_bf16",
        "to_f32",
        "to_f64",
        "to_x80",
        "to_f128",
        "modulo_i32",
        "modulo_i64",
        "x80_mod",
        "x80_round",
        "x80_round_and_pack",
        "x80_norm_round_and_pack",
        "predicates",
        "default_nan",
        "silence_nan",
    ];
}

use fmt::*;
use op::*;

/// Every (format, operation) pair that QEMU has and the harness compares.
pub(crate) fn all_ops() -> Vec<u16> {
    let mut v = Vec::new();
    let mut add = |f: u16, ops: &[u16]| {
        for &o in ops {
            v.push(f * 64 + o);
        }
    };
    let basic = [ADD, SUB, MUL, DIV, SQRT, SCALBN, COMPARE, COMPARE_QUIET, ROUND_TO_INT];
    for f in [F16, BF16, F32, F64, X80, F128] {
        add(f, &basic);
        add(f, &[PREDICATES, DEFAULT_NAN, SILENCE_NAN]);
    }
    let small_int =
        [TO_I8, TO_I16, TO_I32, TO_I64, TO_U8, TO_U16, TO_U32, TO_U64, FROM_I64, FROM_U64];
    add(F16, &small_int);
    add(F16, &[MULADD, MINMAX, TO_F32, TO_F64]);
    add(BF16, &small_int);
    add(BF16, &[MULADD, MINMAX, TO_F32, TO_F64]);
    let wide_int = [TO_I16, TO_I32, TO_I64, TO_U16, TO_U32, TO_U64, FROM_I64, FROM_U64];
    add(F32, &wide_int);
    add(F32, &[MULADD, REM, MINMAX, LOG2, TO_F16, TO_BF16, TO_F64, TO_X80, TO_F128]);
    add(F64, &wide_int);
    add(F64, &[MULADD, REM, MINMAX, LOG2, TO_F16, TO_BF16, TO_F32, TO_X80, TO_F128]);
    add(F64, &[MODULO_I32, MODULO_I64]);
    add(X80, &[REM, X80_MOD, TO_I32, TO_I64, FROM_I64, TO_F32, TO_F64, TO_F128]);
    add(X80, &[X80_ROUND, X80_ROUND_AND_PACK, X80_NORM_ROUND_AND_PACK]);
    add(F128, &[MULADD, REM, MINMAX, TO_I32, TO_I64, TO_U32, TO_U64, TO_I128, TO_U128]);
    add(F128, &[FROM_I64, FROM_U64, FROM_I128, FROM_U128, TO_F32, TO_F64, TO_X80]);
    v
}

/// The name of an op number, for messages.
pub(crate) fn op_name(opn: u16) -> String {
    format!("{}_{}", FMT_NAMES[usize::from(opn / 64)], OP_NAMES[usize::from(opn % 64)])
}

/// One request record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Req {
    pub(crate) op: u16,
    pub(crate) rm: u8,
    pub(crate) x80_prec: u8,
    pub(crate) ftz: u8,
    pub(crate) daz: u8,
    pub(crate) dnan: u8,
    pub(crate) tininess_before: u8,
    pub(crate) ftz_before: u8,
    pub(crate) snan_rule: u8,
    pub(crate) nan2: u8,
    pub(crate) nan3: u8,
    pub(crate) infzeronan: u8,
    pub(crate) x80_behaviour: u8,
    pub(crate) dnan_pattern: u8,
    pub(crate) oprm: u8,
    pub(crate) flags: u16,
    pub(crate) rebias_overflow: u8,
    pub(crate) rebias_underflow: u8,
    pub(crate) imm: i32,
    pub(crate) imm2: i32,
    pub(crate) a: u128,
    pub(crate) b: u128,
    pub(crate) c: u128,
}

/// One result record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Res {
    pub(crate) lo: u64,
    pub(crate) hi: u64,
    pub(crate) extra: u64,
    pub(crate) flags: u16,
}

impl Req {
    pub(crate) fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.op.to_le_bytes());
        out.extend_from_slice(&[
            self.rm,
            self.x80_prec,
            self.ftz,
            self.daz,
            self.dnan,
            self.tininess_before,
            self.ftz_before,
            self.snan_rule,
            self.nan2,
            self.nan3,
            self.infzeronan,
            self.x80_behaviour,
            self.dnan_pattern,
            self.oprm,
        ]);
        out.extend_from_slice(&self.flags.to_le_bytes());
        out.extend_from_slice(&[self.rebias_overflow, self.rebias_underflow]);
        out.extend_from_slice(&self.imm.to_le_bytes());
        out.extend_from_slice(&self.imm2.to_le_bytes());
        out.extend_from_slice(&[0; 4]);
        for v in [self.a, self.b, self.c] {
            out.extend_from_slice(&(v as u64).to_le_bytes());
            out.extend_from_slice(&((v >> 64) as u64).to_le_bytes());
        }
    }

    /// The `FloatStatus` this request describes.
    pub(crate) fn status(&self) -> FloatStatus {
        FloatStatus {
            exception_flags: self.flags,
            rounding_mode: RoundMode::from_u8(self.rm).expect("rounding mode"),
            floatx80_rounding_precision: match self.x80_prec {
                0 => FloatX80RoundPrec::X,
                1 => FloatX80RoundPrec::D,
                _ => FloatX80RoundPrec::S,
            },
            flush_to_zero: self.ftz != 0,
            flush_inputs_to_zero: self.daz != 0,
            default_nan_mode: self.dnan != 0,
            rebias_overflow: self.rebias_overflow != 0,
            rebias_underflow: self.rebias_underflow != 0,
            tininess_before_rounding: self.tininess_before != 0,
            ftz_before_rounding: self.ftz_before != 0,
            snan_rule: match self.snan_rule {
                0 => SnanRule::BitIsZero,
                1 => SnanRule::BitIsOne,
                _ => SnanRule::Never,
            },
            float_2nan_prop_rule: match self.nan2 {
                0 => Float2NanPropRule::None,
                1 => Float2NanPropRule::SAb,
                2 => Float2NanPropRule::SBa,
                3 => Float2NanPropRule::Ab,
                4 => Float2NanPropRule::Ba,
                _ => Float2NanPropRule::X87,
            },
            float_3nan_prop_rule: Float3NanPropRule(self.nan3),
            float_infzeronan_rule: InfZeroNanRule(self.infzeronan),
            floatx80_behaviour: FloatX80Behaviour(self.x80_behaviour),
            default_nan_pattern: self.dnan_pattern,
        }
    }

    /// Load the rule fields of a preset.
    pub(crate) fn set_status(&mut self, s: &FloatStatus) {
        self.rm = s.rounding_mode as u8;
        self.x80_prec = s.floatx80_rounding_precision as u8;
        self.ftz = s.flush_to_zero.into();
        self.daz = s.flush_inputs_to_zero.into();
        self.dnan = s.default_nan_mode.into();
        self.rebias_overflow = s.rebias_overflow.into();
        self.rebias_underflow = s.rebias_underflow.into();
        self.tininess_before = s.tininess_before_rounding.into();
        self.ftz_before = s.ftz_before_rounding.into();
        self.snan_rule = s.snan_rule as u8;
        self.nan2 = s.float_2nan_prop_rule as u8;
        self.nan3 = s.float_3nan_prop_rule.0;
        self.infzeronan = s.float_infzeronan_rule.0;
        self.x80_behaviour = s.floatx80_behaviour.0;
        self.dnan_pattern = s.default_nan_pattern;
        self.flags = s.exception_flags;
    }
}

impl Res {
    pub(crate) fn decode(b: &[u8]) -> Res {
        let u = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        Res { lo: u(0), hi: u(8), extra: u(16), flags: u16::from_le_bytes([b[24], b[25]]) }
    }

    pub(crate) fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.lo.to_le_bytes());
        out.extend_from_slice(&self.hi.to_le_bytes());
        out.extend_from_slice(&self.extra.to_le_bytes());
        out.extend_from_slice(&self.flags.to_le_bytes());
        out.extend_from_slice(&[0; 6]);
    }

    /// A 64 bit FNV-1a hash of the record, what the checked in vectors store.
    pub(crate) fn hash(&self) -> u64 {
        let mut v = Vec::with_capacity(RES_LEN);
        self.encode(&mut v);
        fnv(&v)
    }
}

/// 64 bit FNV-1a.
pub(crate) fn fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn rel(r: FloatRelation) -> u64 {
    r as i8 as i64 as u64
}

fn x80(v: u128) -> FloatX80 {
    FloatX80::new((v >> 64) as u16, v as u64)
}

fn f128(v: u128) -> Float128 {
    Float128::from_bits(v)
}

trait LoHi {
    fn lohi(self) -> (u64, u64);
}

impl LoHi for Float16 {
    fn lohi(self) -> (u64, u64) {
        (u64::from(self.0), 0)
    }
}
impl LoHi for BFloat16 {
    fn lohi(self) -> (u64, u64) {
        (u64::from(self.0), 0)
    }
}
impl LoHi for Float32 {
    fn lohi(self) -> (u64, u64) {
        (u64::from(self.0), 0)
    }
}
impl LoHi for Float64 {
    fn lohi(self) -> (u64, u64) {
        (self.0, 0)
    }
}
impl LoHi for FloatX80 {
    fn lohi(self) -> (u64, u64) {
        (self.low, u64::from(self.high))
    }
}
impl LoHi for Float128 {
    fn lohi(self) -> (u64, u64) {
        (self.low, self.high)
    }
}

fn l<T: LoHi>(v: T) -> (u64, u64) {
    v.lohi()
}

fn si(v: i64) -> (u64, u64) {
    (v as u64, 0)
}

fn ui(v: u64) -> (u64, u64) {
    (v, 0)
}

/// The operations every format has, or `None`.
macro_rules! basic_ops {
    ($o:expr, $a:expr, $b:expr, $r:expr, $s:expr) => {{
        let (a, b, s) = ($a, $b, &mut *$s);
        match $o {
            ADD => Some(l(a.add(b, s))),
            SUB => Some(l(a.sub(b, s))),
            MUL => Some(l(a.mul(b, s))),
            DIV => Some(l(a.div(b, s))),
            SQRT => Some(l(a.sqrt(s))),
            SCALBN => Some(l(a.scalbn($r.imm, s))),
            ROUND_TO_INT => Some(l(a.round_to_int(s))),
            COMPARE => Some((rel(a.compare(b, s)), 0)),
            COMPARE_QUIET => Some((rel(a.compare_quiet(b, s)), 0)),
            _ => None,
        }
    }};
}

/// The float to integer conversions with a scale, or `None`.
macro_rules! to_int_ops {
    ($o:expr, $a:expr, $rm:expr, $r:expr, $s:expr) => {{
        let (a, rm, sc, s) = ($a, $rm, $r.imm, &mut *$s);
        match $o {
            TO_I8 => Some(si(i64::from(a.to_i8_scalbn(rm, sc, s)))),
            TO_I16 => Some(si(i64::from(a.to_i16_scalbn(rm, sc, s)))),
            TO_I32 => Some(si(i64::from(a.to_i32_scalbn(rm, sc, s)))),
            TO_I64 => Some(si(a.to_i64_scalbn(rm, sc, s))),
            TO_U8 => Some(ui(u64::from(a.to_u8_scalbn(rm, sc, s)))),
            TO_U16 => Some(ui(u64::from(a.to_u16_scalbn(rm, sc, s)))),
            TO_U32 => Some(ui(u64::from(a.to_u32_scalbn(rm, sc, s)))),
            TO_U64 => Some(ui(a.to_u64_scalbn(rm, sc, s))),
            _ => None,
        }
    }};
}

/// Predicate bits, as `driver.c` packs them.
macro_rules! preds {
    ($a:expr, $s:expr) => {{
        let a = $a;
        u64::from(a.is_any_nan())
            | u64::from(a.is_quiet_nan($s)) << 1
            | u64::from(a.is_signaling_nan($s)) << 2
            | u64::from(a.is_zero()) << 4
            | u64::from(a.is_neg()) << 5
            | u64::from(a.is_zero_or_denormal()) << 6
    }};
}

/// Run a request through ruvm-softfloat.
pub(crate) fn eval(r: &Req) -> Res {
    let mut st = r.status();
    let s = &mut st;
    let rm = RoundMode::from_u8(r.oprm).expect("operand rounding mode");
    let o = r.op % 64;
    let mut extra = 0;
    let ieee = r.imm & 1 != 0;
    let fl = r.imm as u32;

    let (lo, hi): (u64, u64) = match r.op / 64 {
        F16 => {
            let (a, b, c) = (Float16(r.a as u16), Float16(r.b as u16), Float16(r.c as u16));
            if let Some(v) = basic_ops!(o, a, b, r, s) {
                v
            } else if let Some(v) = to_int_ops!(o, a, rm, r, s) {
                v
            } else {
                match o {
                    MULADD if r.imm2 == 0 => l(a.muladd(b, c, fl, s)),
                    MULADD => l(a.muladd_scalbn(b, c, r.imm2, fl, s)),
                    MINMAX => l(a.minmax(b, s, fl)),
                    FROM_I64 => l(Float16::from_i64_scalbn(r.a as i64, r.imm, s)),
                    FROM_U64 => l(Float16::from_u64_scalbn(r.a as u64, r.imm, s)),
                    TO_F32 => l(a.to_float32(ieee, s)),
                    TO_F64 => l(a.to_float64(ieee, s)),
                    PREDICATES => ui(preds!(a, s)
                        | u64::from(a.is_infinity()) << 3
                        | u64::from(a.is_normal()) << 7),
                    DEFAULT_NAN => l(Float16::default_nan(s)),
                    SILENCE_NAN => l(a.silence_nan(s)),
                    _ => panic!("unsupported {}", op_name(r.op)),
                }
            }
        }
        BF16 => {
            let (a, b, c) = (BFloat16(r.a as u16), BFloat16(r.b as u16), BFloat16(r.c as u16));
            if let Some(v) = basic_ops!(o, a, b, r, s) {
                v
            } else if let Some(v) = to_int_ops!(o, a, rm, r, s) {
                v
            } else {
                match o {
                    MULADD => l(a.muladd(b, c, fl, s)),
                    MINMAX => l(a.minmax(b, s, fl)),
                    FROM_I64 => l(BFloat16::from_i64_scalbn(r.a as i64, r.imm, s)),
                    FROM_U64 => l(BFloat16::from_u64_scalbn(r.a as u64, r.imm, s)),
                    TO_F32 => l(a.to_float32(s)),
                    TO_F64 => l(a.to_float64(s)),
                    PREDICATES => ui(preds!(a, s)
                        | u64::from(a.is_infinity()) << 3
                        | u64::from(a.is_normal()) << 7),
                    DEFAULT_NAN => l(BFloat16::default_nan(s)),
                    SILENCE_NAN => l(a.silence_nan(s)),
                    _ => panic!("unsupported {}", op_name(r.op)),
                }
            }
        }
        F32 => {
            let (a, b, c) = (Float32(r.a as u32), Float32(r.b as u32), Float32(r.c as u32));
            if let Some(v) = basic_ops!(o, a, b, r, s) {
                v
            } else if let Some(v) = to_int_ops!(o, a, rm, r, s) {
                v
            } else {
                match o {
                    MULADD if r.imm2 == 0 => l(a.muladd(b, c, fl, s)),
                    MULADD => l(a.muladd_scalbn(b, c, r.imm2, fl, s)),
                    REM => l(a.rem(b, s)),
                    MINMAX => l(a.minmax(b, s, fl)),
                    LOG2 => l(a.log2(s)),
                    FROM_I64 => l(Float32::from_i64_scalbn(r.a as i64, r.imm, s)),
                    FROM_U64 => l(Float32::from_u64_scalbn(r.a as u64, r.imm, s)),
                    TO_F16 => l(a.to_float16(ieee, s)),
                    TO_BF16 => l(a.to_bfloat16(s)),
                    TO_F64 => l(a.to_float64(s)),
                    TO_X80 => l(a.to_floatx80(s)),
                    TO_F128 => l(a.to_float128(s)),
                    PREDICATES => ui(preds!(a, s)
                        | u64::from(a.is_infinity()) << 3
                        | u64::from(a.is_normal()) << 7
                        | u64::from(a.is_denormal()) << 8),
                    DEFAULT_NAN => l(Float32::default_nan(s)),
                    SILENCE_NAN => l(a.silence_nan(s)),
                    _ => panic!("unsupported {}", op_name(r.op)),
                }
            }
        }
        F64 => {
            let (a, b, c) = (Float64(r.a as u64), Float64(r.b as u64), Float64(r.c as u64));
            if let Some(v) = basic_ops!(o, a, b, r, s) {
                v
            } else if let Some(v) = to_int_ops!(o, a, rm, r, s) {
                v
            } else {
                match o {
                    MULADD if r.imm2 == 0 => l(a.muladd(b, c, fl, s)),
                    MULADD => l(a.muladd_scalbn(b, c, r.imm2, fl, s)),
                    REM => l(a.rem(b, s)),
                    MINMAX => l(a.minmax(b, s, fl)),
                    LOG2 => l(a.log2(s)),
                    FROM_I64 => l(Float64::from_i64_scalbn(r.a as i64, r.imm, s)),
                    FROM_U64 => l(Float64::from_u64_scalbn(r.a as u64, r.imm, s)),
                    TO_F16 => l(a.to_float16(ieee, s)),
                    TO_BF16 => l(a.to_bfloat16(s)),
                    TO_F32 => l(a.to_float32(s)),
                    TO_X80 => l(a.to_floatx80(s)),
                    TO_F128 => l(a.to_float128(s)),
                    MODULO_I32 => si(i64::from(a.to_i32_modulo(rm, s))),
                    MODULO_I64 => si(a.to_i64_modulo(rm, s)),
                    PREDICATES => ui(preds!(a, s)
                        | u64::from(a.is_infinity()) << 3
                        | u64::from(a.is_normal()) << 7
                        | u64::from(a.is_denormal()) << 8),
                    DEFAULT_NAN => l(Float64::default_nan(s)),
                    SILENCE_NAN => l(a.silence_nan(s)),
                    _ => panic!("unsupported {}", op_name(r.op)),
                }
            }
        }
        X80 => {
            let (a, b) = (x80(r.a), x80(r.b));
            if let Some(v) = basic_ops!(o, a, b, r, s) {
                v
            } else {
                match o {
                    REM => l(a.rem(b, s)),
                    X80_MOD => {
                        let (v, q) = a.modrem(b, r.imm & 1 == 0, s);
                        extra = q;
                        l(v)
                    }
                    TO_I32 => si(i64::from(a.to_i32_scalbn(rm, r.imm, s))),
                    TO_I64 => si(a.to_i64_scalbn(rm, r.imm, s)),
                    FROM_I64 => l(FloatX80::from_i64(r.a as i64, s)),
                    TO_F32 => l(a.to_float32(s)),
                    TO_F64 => l(a.to_float64(s)),
                    TO_F128 => l(a.to_float128(s)),
                    X80_ROUND => l(a.round(s)),
                    X80_ROUND_AND_PACK | X80_NORM_ROUND_AND_PACK => {
                        let prec = s.floatx80_rounding_precision;
                        let sign = r.imm2 & 1 != 0;
                        let (sig0, sig1) = (r.a as u64, (r.a >> 64) as u64);
                        if o == X80_ROUND_AND_PACK {
                            l(FloatX80::round_and_pack(prec, sign, r.imm, sig0, sig1, s))
                        } else {
                            l(FloatX80::normalize_round_and_pack(prec, sign, r.imm, sig0, sig1, s))
                        }
                    }
                    PREDICATES => ui(preds!(a, s)
                        | u64::from(a.is_infinity(s)) << 3
                        | u64::from(a.invalid_encoding(s)) << 9),
                    DEFAULT_NAN => l(FloatX80::default_nan(s)),
                    SILENCE_NAN => l(a.silence_nan(s)),
                    _ => panic!("unsupported {}", op_name(r.op)),
                }
            }
        }
        F128 => {
            let (a, b, c) = (f128(r.a), f128(r.b), f128(r.c));
            if let Some(v) = basic_ops!(o, a, b, r, s) {
                v
            } else {
                let (rmo, sc) = (rm, r.imm);
                match o {
                    MULADD => l(a.muladd(b, c, fl, s)),
                    REM => l(a.rem(b, s)),
                    MINMAX => l(a.minmax(b, s, fl)),
                    TO_I32 => si(i64::from(a.to_i32_scalbn(rmo, sc, s))),
                    TO_I64 => si(a.to_i64_scalbn(rmo, sc, s)),
                    TO_U32 => ui(u64::from(a.to_u32_scalbn(rmo, sc, s))),
                    TO_U64 => ui(a.to_u64_scalbn(rmo, sc, s)),
                    TO_I128 => {
                        let v = a.to_i128_scalbn(rmo, sc, s) as u128;
                        (v as u64, (v >> 64) as u64)
                    }
                    TO_U128 => {
                        let v = a.to_u128_scalbn(rmo, sc, s);
                        (v as u64, (v >> 64) as u64)
                    }
                    FROM_I64 => l(Float128::from_i64(r.a as i64, s)),
                    FROM_U64 => l(Float128::from_u64(r.a as u64, s)),
                    FROM_I128 => l(Float128::from_i128(r.a as i128, s)),
                    FROM_U128 => l(Float128::from_u128(r.a, s)),
                    TO_F32 => l(a.to_float32(s)),
                    TO_F64 => l(a.to_float64(s)),
                    TO_X80 => l(a.to_floatx80(s)),
                    PREDICATES => ui(preds!(a, s)
                        | u64::from(a.is_infinity()) << 3
                        | u64::from(a.is_normal()) << 7
                        | u64::from(a.is_denormal()) << 8),
                    DEFAULT_NAN => l(Float128::default_nan(s)),
                    SILENCE_NAN => l(a.silence_nan(s)),
                    _ => panic!("unsupported {}", op_name(r.op)),
                }
            }
        }
        _ => panic!("bad format in op {}", r.op),
    };
    Res { lo, hi, extra, flags: st.exception_flags }
}

/// A small, fast, deterministic generator (SplitMix64).
#[derive(Clone, Debug)]
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub(crate) fn wide(&mut self) -> u128 {
        (u128::from(self.next()) << 64) | u128::from(self.next())
    }

    /// Uniform in `0..n`.
    pub(crate) fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// Uniform in `lo..=hi`.
    pub(crate) fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo + 1) as u64) as i64
    }

    pub(crate) fn bit(&mut self) -> bool {
        self.next() & 1 != 0
    }

    /// True with probability `1 / n`.
    pub(crate) fn one_in(&mut self, n: u64) -> bool {
        self.below(n) == 0
    }
}

/// The layout of an IEEE style format.
#[derive(Clone, Copy, Debug)]
struct Layout {
    ebits: u32,
    fbits: u32,
}

const F16_L: Layout = Layout { ebits: 5, fbits: 10 };
const BF16_L: Layout = Layout { ebits: 8, fbits: 7 };
const F32_L: Layout = Layout { ebits: 8, fbits: 23 };
const F64_L: Layout = Layout { ebits: 11, fbits: 52 };
const F128_L: Layout = Layout { ebits: 15, fbits: 112 };

fn layout(f: u16) -> Layout {
    match f {
        F16 => F16_L,
        BF16 => BF16_L,
        F32 => F32_L,
        F64 => F64_L,
        F128 => F128_L,
        _ => unreachable!(),
    }
}

fn mask(bits: u32) -> u128 {
    if bits >= 128 { u128::MAX } else { (1u128 << bits) - 1 }
}

/// A fraction of `bits` bits with one of several shapes that stress rounding.
fn gen_frac(rng: &mut Rng, bits: u32) -> u128 {
    let m = mask(bits);
    let k = rng.below(u64::from(bits) + 1) as u32;
    match rng.below(10) {
        0 => 0,
        1 => m,
        2 => (1u128 << rng.below(u64::from(bits))) & m,
        3 => m & !(mask(bits - k)),
        4 => m >> k,
        5 => rng.wide() & mask(k.min(8)),
        6 => {
            // A halfway or near halfway pattern at a random position.
            let pos = rng.below(u64::from(bits)) as u32;
            let top = rng.wide() & !mask(pos + 1) & m;
            let half = 1u128 << pos;
            match rng.below(3) {
                0 => top | half,
                1 => top | (half - 1),
                _ => top | half | 1,
            }
        }
        7 => (rng.wide() & m) | (m >> k),
        _ => rng.wide() & m,
    }
}

/// An exponent field value for a layout, biased toward the interesting places.
fn gen_exp(rng: &mut Rng, lay: Layout) -> i64 {
    let emax = (1i64 << lay.ebits) - 1;
    let bias = emax >> 1;
    let f = i64::from(lay.fbits);
    let e = match rng.below(16) {
        0 => 0,
        1 => emax,
        2 => rng.range(1, 3),
        3 => emax - rng.range(1, 3),
        4..=9 => bias + rng.range(-8, 8),
        10 | 11 => bias + rng.range(-f - 8, f + 8),
        12 => bias + rng.range(-2, 66),
        _ => rng.range(0, emax),
    };
    e.clamp(0, emax)
}

fn pack_ieee(lay: Layout, sign: bool, exp: i64, frac: u128) -> u128 {
    (u128::from(sign) << (lay.ebits + lay.fbits))
        | ((exp as u128) << lay.fbits)
        | (frac & mask(lay.fbits))
}

fn exp_of(lay: Layout, v: u128) -> i64 {
    ((v >> lay.fbits) & mask(lay.ebits)) as i64
}

fn with_exp(lay: Layout, v: u128, e: i64) -> u128 {
    let emax = (1i64 << lay.ebits) - 1;
    let e = e.clamp(0, emax) as u128;
    (v & !(mask(lay.ebits) << lay.fbits)) | (e << lay.fbits)
}

/// A special value of an IEEE style layout.
fn special_ieee(rng: &mut Rng, lay: Layout) -> u128 {
    let emax = (1i64 << lay.ebits) - 1;
    let bias = emax >> 1;
    let fm = mask(lay.fbits);
    let qbit = 1u128 << (lay.fbits - 1);
    let sign = rng.bit();
    let (e, f) = match rng.below(14) {
        0 => (0, 0),
        1 => (emax, 0),
        2 => (emax, qbit),
        3 => (emax, 1),
        4 => (emax, qbit | (rng.wide() & fm)),
        5 => (emax, (rng.wide() & (qbit - 1)) | 1),
        6 => (0, 1),
        7 => (0, fm),
        8 => (1, 0),
        9 => (emax - 1, fm),
        10 => (bias, 0),
        11 => (bias - 1, 0),
        12 => (bias + i64::from(lay.fbits), 0),
        _ => (bias + i64::from(lay.fbits), fm),
    };
    pack_ieee(lay, sign, e, f)
}

/// A random operand of an IEEE style layout.
fn gen_ieee(rng: &mut Rng, lay: Layout) -> u128 {
    match rng.below(12) {
        0 => rng.wide() & mask(1 + lay.ebits + lay.fbits),
        1 | 2 => special_ieee(rng, lay),
        _ => {
            let e = gen_exp(rng, lay);
            pack_ieee(lay, rng.bit(), e, gen_frac(rng, lay.fbits))
        }
    }
}

/// A random floatx80 operand, including the unnormal and pseudo encodings.
fn gen_x80(rng: &mut Rng) -> u128 {
    let lay = Layout { ebits: 15, fbits: 63 };
    let v = if rng.one_in(3) {
        special_ieee(rng, lay)
    } else if rng.one_in(10) {
        rng.wide() & mask(80)
    } else {
        pack_ieee(lay, rng.bit(), gen_exp(rng, lay), gen_frac(rng, 63))
    };
    let e = exp_of(lay, v);
    let low = v & mask(63);
    let sign_exp = v >> 63;
    let mut int_bit = u128::from(e != 0);
    if rng.one_in(8) {
        int_bit ^= 1;
    }
    (sign_exp << 64) | (int_bit << 63) | low
}

/// A random operand for a format.
pub(crate) fn gen_operand(rng: &mut Rng, f: u16) -> u128 {
    if f == X80 { gen_x80(rng) } else { gen_ieee(rng, layout(f)) }
}

/// A second operand related to the first: same or nearby exponent, equal, negated, one ulp off.
fn gen_related(rng: &mut Rng, f: u16, a: u128) -> u128 {
    let (lay, sign_bit) = if f == X80 {
        (Layout { ebits: 15, fbits: 64 }, 79)
    } else {
        let lay = layout(f);
        (lay, lay.ebits + lay.fbits)
    };
    match rng.below(10) {
        0 => a,
        1 => a ^ (1u128 << sign_bit),
        2 => a.wrapping_add(1) & mask(sign_bit + 1),
        3 => a.wrapping_sub(1) & mask(sign_bit + 1),
        4 => a ^ (u128::from(rng.bit()) << sign_bit) ^ (1u128 << rng.below(u64::from(lay.fbits))),
        _ => {
            let b = gen_operand(rng, f);
            let d = match rng.below(3) {
                0 => rng.range(-3, 3),
                1 => rng.range(-(i64::from(lay.fbits) + 3), i64::from(lay.fbits) + 3),
                _ => rng.range(-200, 200),
            };
            with_exp(lay, b, exp_of(lay, a) + d)
        }
    }
}

/// A float operand whose value is near the range of an `bits` bit integer.
fn gen_near_int(rng: &mut Rng, f: u16, bits: i64) -> u128 {
    let a = gen_operand(rng, f);
    if rng.one_in(3) {
        return a;
    }
    let (lay, bias) = if f == X80 {
        (Layout { ebits: 15, fbits: 64 }, 0x3fff)
    } else {
        let lay = layout(f);
        (lay, (1i64 << (lay.ebits - 1)) - 1)
    };
    let e = match rng.below(3) {
        0 => bias + rng.range(-2, 3),
        1 => bias + bits - 1 + rng.range(-2, 2),
        _ => bias + rng.range(-2, bits + 2),
    };
    with_exp(lay, a, e)
}

/// A random 128 bit integer with a random magnitude and bit pattern.
fn gen_int(rng: &mut Rng) -> u128 {
    let v = match rng.below(6) {
        0 => rng.wide() >> rng.below(128),
        1 => {
            let hi = rng.below(128) as u32;
            let lo = rng.below(u64::from(hi) + 1) as u32;
            (1u128 << hi) | (1u128 << lo)
        }
        2 => mask(rng.below(129) as u32),
        3 => u128::from(rng.below(8)),
        _ => rng.wide() >> rng.below(128) | (1u128 << rng.below(128)),
    };
    if rng.one_in(3) { v.wrapping_neg() } else { v }
}

/// A scale factor for the `_scalbn` operations.
fn gen_scale(rng: &mut Rng, zero_bias: bool) -> i32 {
    if zero_bias && rng.below(10) < 6 {
        return 0;
    }
    match rng.below(8) {
        0 => [i32::MIN, i32::MAX, 0x10000, -0x10000, 0x10001, -0x10001, 0x7fff, -0x7fff]
            [rng.below(8) as usize],
        1 | 2 => rng.range(-200, 200) as i32,
        3 => rng.range(-40000, 40000) as i32,
        _ => rng.range(-20, 20) as i32,
    }
}

/// The valid `float_minmax_*` combinations.
const MINMAX_FLAGS: [u32; 10] = [
    0,
    minmax::ISMIN,
    minmax::ISNUM,
    minmax::ISNUM | minmax::ISMIN,
    minmax::ISNUM | minmax::ISMAG,
    minmax::ISNUM | minmax::ISMAG | minmax::ISMIN,
    minmax::ISNUMBER,
    minmax::ISNUMBER | minmax::ISMIN,
    minmax::ISMAG,
    minmax::ISMAG | minmax::ISMIN,
];

/// The target configurations the harness runs under.
pub(crate) const CONFIG_NAMES: [&str; 7] =
    ["x86_sse", "arm", "arm_ah", "arm_standard", "mips_legacy", "ppc", "m68k_like"];

fn config(i: u64) -> FloatStatus {
    match i {
        0 => FloatStatus::x86_sse(),
        1 => FloatStatus::arm(),
        2 => FloatStatus::arm_ah(),
        3 => FloatStatus::arm_standard(),
        4 => {
            // MIPS before NaN2008: the snan bit is one.
            let mut s = FloatStatus::arm();
            s.snan_rule = SnanRule::BitIsOne;
            s.float_2nan_prop_rule = Float2NanPropRule::SAb;
            s.float_3nan_prop_rule = Float3NanPropRule::S_ABC;
            s.float_infzeronan_rule = InfZeroNanRule::DNAN_ALWAYS;
            s.default_nan_pattern = 0b0011_1111;
            s.tininess_before_rounding = false;
            s
        }
        5 => {
            let mut s = FloatStatus::arm();
            s.float_2nan_prop_rule = Float2NanPropRule::Ab;
            s.float_3nan_prop_rule = Float3NanPropRule::ACB;
            s.float_infzeronan_rule = InfZeroNanRule::DNAN_NEVER;
            s.tininess_before_rounding = true;
            s
        }
        _ => {
            let mut s = FloatStatus::x87();
            s.floatx80_behaviour = FloatX80Behaviour(31);
            s.float_2nan_prop_rule = Float2NanPropRule::Ab;
            s.default_nan_pattern = 0b0111_1111;
            s
        }
    }
}

/// A random status: a target preset, a random rounding mode and precision, and occasional
/// flips of the flush, default NaN and tininess settings, plus random initial flags (the
/// inexact flag in particular switches QEMU onto its hardfloat paths).
fn gen_status(rng: &mut Rng) -> FloatStatus {
    let mut s = config(rng.below(CONFIG_NAMES.len() as u64));
    s.rounding_mode = RoundMode::from_u8(rng.below(8) as u8).unwrap();
    s.floatx80_rounding_precision = match rng.below(3) {
        0 => FloatX80RoundPrec::X,
        1 => FloatX80RoundPrec::D,
        _ => FloatX80RoundPrec::S,
    };
    if rng.one_in(5) {
        s.flush_to_zero ^= true;
    }
    if rng.one_in(5) {
        s.flush_inputs_to_zero ^= true;
    }
    if rng.one_in(6) {
        s.default_nan_mode ^= true;
    }
    if rng.one_in(8) {
        s.tininess_before_rounding ^= true;
    }
    if rng.one_in(8) {
        s.ftz_before_rounding ^= true;
    }
    if rng.one_in(32) {
        s.rebias_overflow = true;
    }
    if rng.one_in(32) {
        s.rebias_underflow = true;
    }
    if rng.one_in(10) {
        s.floatx80_behaviour = FloatX80Behaviour(rng.below(32) as u8);
    }
    s.exception_flags = match rng.below(10) {
        0..=3 => 0,
        4..=7 => flags::INEXACT,
        _ => rng.next() as u16 & 0x7fff,
    };
    s
}

/// Generate one request for the op number `opn`.
pub(crate) fn gen_request(rng: &mut Rng, opn: u16) -> Req {
    let (f, o) = (opn / 64, opn % 64);
    let mut s = gen_status(rng);
    if o == X80_ROUND_AND_PACK || o == X80_NORM_ROUND_AND_PACK {
        // roundAndPackFloatx80 aborts on the round to odd modes.
        s.rounding_mode = RoundMode::from_u8(rng.below(5) as u8).unwrap();
    }
    if f == X80 && o == SILENCE_NAN && s.snan_rule != SnanRule::BitIsZero {
        s.snan_rule = SnanRule::BitIsZero;
        s.default_nan_pattern = 0b1100_0000;
    }
    let mut r = Req { op: opn, ..Req::default() };
    r.set_status(&s);
    r.oprm = rng.below(8) as u8;

    let a = gen_operand(rng, f);
    r.a = a;
    r.b = if rng.below(10) < 4 { gen_related(rng, f, a) } else { gen_operand(rng, f) };
    r.c = gen_operand(rng, f);

    match o {
        SCALBN => r.imm = gen_scale(rng, false),
        MULADD => {
            r.imm = rng.below(16) as i32;
            r.imm2 = if matches!(f, F16 | F32 | F64) { gen_scale(rng, true) } else { 0 };
            if rng.below(3) == 0 {
                // Aim C at the product so that cancellation happens.
                let lay = if f == X80 { Layout { ebits: 15, fbits: 64 } } else { layout(f) };
                let bias = (1i64 << (lay.ebits - 1)) - 1;
                let e = exp_of(lay, r.a) + exp_of(lay, r.b) - bias + rng.range(-3, 3);
                r.c = with_exp(lay, r.c, e);
            }
        }
        X80_MOD => r.imm = rng.below(2) as i32,
        MINMAX => r.imm = MINMAX_FLAGS[rng.below(MINMAX_FLAGS.len() as u64) as usize] as i32,
        TO_I8 | TO_U8 => r.a = gen_near_int(rng, f, 8),
        TO_I16 | TO_U16 => r.a = gen_near_int(rng, f, 16),
        TO_I32 | TO_U32 | MODULO_I32 => r.a = gen_near_int(rng, f, 32),
        TO_I64 | TO_U64 | MODULO_I64 => r.a = gen_near_int(rng, f, 64),
        TO_I128 | TO_U128 => r.a = gen_near_int(rng, f, 128),
        FROM_I64 | FROM_U64 | FROM_I128 | FROM_U128 => r.a = gen_int(rng),
        ROUND_TO_INT => {
            let bits = rng.range(1, 70);
            r.a = gen_near_int(rng, f, bits);
        }
        TO_F16 | TO_F32 | TO_F64 => r.imm = rng.below(2) as i32,
        SILENCE_NAN => {
            if rng.below(4) != 0 {
                let lay = if f == X80 { Layout { ebits: 15, fbits: 64 } } else { layout(f) };
                r.a |= mask(lay.ebits) << lay.fbits;
                if f == X80 {
                    r.a |= 1u128 << 63;
                }
                if r.a & mask(lay.fbits - 1) == 0 {
                    r.a |= 1;
                }
            }
        }
        X80_ROUND_AND_PACK | X80_NORM_ROUND_AND_PACK => {
            r.imm = match rng.below(4) {
                0 => rng.range(-80, 5) as i32,
                1 => rng.range(0x7ff0, 0x8002) as i32,
                2 => rng.range(0x3f00, 0x4100) as i32,
                _ => rng.range(0, 0x7fff) as i32,
            };
            r.imm2 = rng.below(2) as i32;
            let mut sig0 = gen_frac(rng, 64) as u64;
            let sig1 = gen_frac(rng, 64) as u64;
            if o == X80_ROUND_AND_PACK {
                if r.imm == 0 && rng.bit() {
                    sig0 &= !(1u64 << 63);
                } else {
                    sig0 |= 1u64 << 63;
                }
            } else if rng.one_in(4) {
                sig0 = 0;
            }
            r.a = (u128::from(sig1) << 64) | u128::from(sig0);
        }
        _ => {}
    }
    match o {
        TO_I8 | TO_I16 | TO_I32 | TO_I64 | TO_U8 | TO_U16 | TO_U32 | TO_U64 | TO_I128 | TO_U128
        | FROM_I64 | FROM_U64 => {
            // QEMU has no scale for the float128 and floatx80 integer conversions from an
            // integer, and uses one for the rest.
            let scaled = !(o == FROM_I64 || o == FROM_U64) || matches!(f, F16 | BF16 | F32 | F64);
            r.imm = if scaled { gen_scale(rng, true) } else { 0 };
        }
        _ => {}
    }
    r
}

/// A deterministic stream of `count` requests spread evenly over every op.
pub(crate) fn gen_stream(seed: u64, count: usize) -> Vec<Req> {
    let ops = all_ops();
    let mut rng = Rng::new(seed);
    (0..count).map(|i| gen_request(&mut rng, ops[i % ops.len()])).collect()
}

/// Describe a mismatch.
pub(crate) fn describe(r: &Req, want: &Res, got: &Res) -> String {
    format!(
        "{} cfg rm={} prec={} ftz={} daz={} dnan={} tbr={} ftzbr={} snan={} 2nan={} 3nan={:#x} izn={:#x} x80b={:#x} pat={:#010b} rebias={}{} flags_in={:#06x} oprm={} imm={} imm2={}\n  a={:#034x} b={:#034x} c={:#034x}\n  qemu: lo={:#018x} hi={:#018x} extra={:#x} flags={:#06x}\n  ruvm: lo={:#018x} hi={:#018x} extra={:#x} flags={:#06x}",
        op_name(r.op),
        r.rm,
        r.x80_prec,
        r.ftz,
        r.daz,
        r.dnan,
        r.tininess_before,
        r.ftz_before,
        r.snan_rule,
        r.nan2,
        r.nan3,
        r.infzeronan,
        r.x80_behaviour,
        r.dnan_pattern,
        r.rebias_overflow,
        r.rebias_underflow,
        r.flags,
        r.oprm,
        r.imm,
        r.imm2,
        r.a,
        r.b,
        r.c,
        want.lo,
        want.hi,
        want.extra,
        want.flags,
        got.lo,
        got.hi,
        got.extra,
        got.flags
    )
}

/// The checked in vector file: a header, then one result hash per request of
/// `gen_stream(VECTOR_SEED, VECTOR_COUNT)`.
pub(crate) const VECTOR_MAGIC: &[u8; 8] = b"RSFVEC01";
/// The seed of the checked in vectors.
pub(crate) const VECTOR_SEED: u64 = 0x5eed_50f7_f10a_7001;
/// How many checked in vectors there are.
pub(crate) const VECTOR_COUNT: usize = 40_000;

/// The hash of a request stream, stored in the vector header so that a changed generator is
/// reported as stale vectors rather than as mismatches.
pub(crate) fn stream_hash(reqs: &[Req]) -> u64 {
    let mut v = Vec::with_capacity(reqs.len() * REQ_LEN);
    for r in reqs {
        r.encode(&mut v);
    }
    fnv(&v)
}
