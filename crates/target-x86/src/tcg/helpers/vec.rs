// SPDX-License-Identifier: GPL-2.0-or-later

//! The MMX, SSE and AVX helpers: a port of `ops_sse.h` and of the SSE parts of `fpu_helper.c`
//! and `int_helper.c`.
//!
//! QEMU has one helper per operation and vector length. Here one kernel, [`SSE`], takes the
//! operation, a variant (element size or floating point format), the vector length and an
//! immediate packed into one word, and the `env` offsets of up to four registers packed into
//! another. It reads the source registers into local copies, so a destination that overlaps a
//! source behaves as in QEMU, which also works on temporaries where it matters.
//!
//! QEMU keeps the SSE `float_status` in `env` and folds its flags into MXCSR when MXCSR is read.
//! The kernel builds a fresh status from MXCSR for every call and ors the new flags into MXCSR
//! at once, which leaves the same MXCSR behind.

use ruvm_jit_core::types::call_flags::NO_RWG;
use ruvm_jit_interp::{HelperEnv, Unwind};
use ruvm_softfloat::{
    Float16, Float32, Float64, FloatRelation, FloatStatus, RoundMode, flags as ff,
};

use super::super::cc::CC_OP_EFLAGS;
use super::super::env::{
    CC_C, CC_OP, CC_P, CC_SRC, CC_Z, FPSTT, FPTAGS, HF_AVX_EN_MASK, MXCSR, TSC_AUX, XCR0,
    avx_enabled, cr, ld32, ld64, st32, st64,
};
use super::super::{EXCP0D_GPF, EXCP06_ILLOP, x86_of};
use super::{Def, I32, I64, Ptr, TB, Void, a32, hflags, run, set_hflags, sh};
use crate::state::CR4_OSXSAVE_MASK;

/// The MXCSR exception flag bits.
const MXCSR_IE: u32 = 0x01;
const MXCSR_DE: u32 = 0x02;
const MXCSR_ZE: u32 = 0x04;
const MXCSR_OE: u32 = 0x08;
const MXCSR_UE: u32 = 0x10;
const MXCSR_PE: u32 = 0x20;
const MXCSR_DAZ: u32 = 0x40;
const MXCSR_FZ: u32 = 0x8000;

/// `update_mxcsr_status()`: the SSE float status for `mxcsr`, with no flags raised.
fn sse_status(mxcsr: u32) -> FloatStatus {
    let mut s = FloatStatus::x86_sse();
    s.rounding_mode = RoundMode::from_u8(((mxcsr >> 13) & 3) as u8).unwrap_or_default();
    s.flush_inputs_to_zero = mxcsr & MXCSR_DAZ != 0;
    s.flush_to_zero = mxcsr & MXCSR_FZ != 0;
    s
}

/// `update_mxcsr_from_sse_status()`: the MXCSR bits for the flags in `s`.
fn mxcsr_flags(f: u16) -> u32 {
    let mut m = 0;
    if f & ff::INVALID != 0 {
        m |= MXCSR_IE;
    }
    if f & ff::INPUT_DENORMAL_USED != 0 {
        m |= MXCSR_DE;
    }
    if f & ff::DIVBYZERO != 0 {
        m |= MXCSR_ZE;
    }
    if f & ff::OVERFLOW != 0 {
        m |= MXCSR_OE;
    }
    if f & ff::UNDERFLOW != 0 {
        m |= MXCSR_UE;
    }
    if f & ff::INEXACT != 0 {
        m |= MXCSR_PE;
    }
    if f & ff::OUTPUT_DENORMAL_FLUSHED != 0 {
        m |= MXCSR_UE | MXCSR_PE;
    }
    m
}

/// `set_x86_rounding_mode()`.
fn x86_rmode(m: u32) -> RoundMode {
    RoundMode::from_u8((m & 3) as u8).unwrap_or_default()
}

macro_rules! kernels {
    ($($k:ident),* $(,)?) => {
        /// The operations of [`SSE`].
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(u8)]
        pub(crate) enum K { $($k),* }

        const KS: &[K] = &[$(K::$k),*];
    };
}

kernels!(
    // Integer, the variant is log2 of the element size.
    Add, Sub, AddS, AddUS, SubS, SubUS, CmpEq, CmpGt, MinS, MaxS, MinU, MaxU, And, AndN, Or, Xor,
    Avg, MulL, MulHS, MulHU, MulHRS, Abs, Sign, MaddWd, MaddUbsw, SadBw, MulUdq, MulDq, PackSS,
    PackUS, UnpckL, UnpckH, ShufB, HAdd, HSub, HAddS, HSubS, AlignR, ShlR, ShrR, SarR, ShlI, ShrI,
    SarI, BslI, BsrI, ShlV, ShrV, SarV, PshufD, PshufHW, PshufLW, PshufW, BlendW, BlendV, Blend,
    MovSx, MovZx, Bcast, Ptest, MovMsk, Extr, Insr, Phminposuw, Mpsadbw, Pclmul, PermD, PermQ,
    Perm2, PermilV, PermilI, ShufP, MovSlDup, MovShDup, MovDDup, MaskLd, InsertPs, Extract128,
    Insert128, Copy,
    // Floating point, the variant is 0 for ps, 1 for pd, 2 for ss and 3 for sd.
    FAdd, FSub, FMul, FDiv, FMin, FMax, FSqrt, FRsqrt, FRcp, FHAdd, FHSub, FAddSub, FCmp, Comi,
    Ucomi, Round, Dpp, Fma, // Conversions.
    CvtDq2Ps, CvtPs2Dq, CvttPs2Dq, CvtDq2Pd, CvtPd2Dq, CvttPd2Dq, CvtPs2Pd, CvtPd2Ps, CvtSs2Sd,
    CvtSd2Ss, CvtSi2S, CvtS2Si, CvtPi2Ps, CvtPi2Pd, CvtPs2Pi, CvtPd2Pi, CvtPh2Ps, CvtPs2Ph,
);

impl K {
    fn from_u8(v: u8) -> K {
        KS[usize::from(v)]
    }
}

/// Pack the first argument of [`SSE`].
pub(crate) fn sse_op(k: K, var: u32, len: u32, imm: u32) -> i32 {
    let lenc = match len {
        8 => 0,
        16 => 1,
        _ => 2,
    };
    (k as u32 | (var << 8) | (lenc << 16) | ((imm & 0xff) << 24)) as i32
}

/// Pack the `env` offsets for [`SSE`]: destination, then up to three sources.
pub(crate) fn sse_offs(d: usize, a: usize, b: usize, c: usize) -> i64 {
    (d as u64 | (a as u64) << 16 | (b as u64) << 32 | (c as u64) << 48) as i64
}

def!(SSE, "x86_sse", NO_RWG, I64, [Ptr, I32, I64, I64], h_sse);

type V = [u8; 32];

fn rd(env: &[u8], off: usize) -> V {
    let mut v = [0; 32];
    let n = 32.min(env.len().saturating_sub(off));
    v[..n].copy_from_slice(&env[off..off + n]);
    v
}

/// Element `i` of size `1 << esz` bytes, zero extended.
fn ge(v: &V, esz: u32, i: usize) -> u64 {
    let n = 1usize << esz;
    let mut x = 0u64;
    for k in 0..n {
        x |= u64::from(v[i * n + k]) << (8 * k);
    }
    x
}

fn pe(v: &mut V, esz: u32, i: usize, x: u64) {
    let n = 1usize << esz;
    for k in 0..n {
        v[i * n + k] = (x >> (8 * k)) as u8;
    }
}

/// Sign extend an element.
fn sx(x: u64, esz: u32) -> i64 {
    let sh = 64 - (8u32 << esz);
    ((x << sh) as i64) >> sh
}

fn emask(esz: u32) -> u64 {
    if esz >= 3 { u64::MAX } else { (1u64 << (8u32 << esz)) - 1 }
}

fn sat_s(x: i64, esz: u32) -> u64 {
    let bits = 8u32 << esz;
    let max = (1i64 << (bits - 1)) - 1;
    let min = -(1i64 << (bits - 1));
    (x.clamp(min, max) as u64) & emask(esz)
}

fn sat_u(x: i64, esz: u32) -> u64 {
    let max = emask(esz) as i64;
    x.clamp(0, max) as u64
}

fn f32_of(v: &V, i: usize) -> Float32 {
    Float32(ge(v, 2, i) as u32)
}

fn f64_of(v: &V, i: usize) -> Float64 {
    Float64(ge(v, 3, i))
}

/// x86_float32_to_int32 and friends: an invalid conversion gives the integer indefinite.
fn cvt_i(f: impl FnOnce(&mut FloatStatus) -> i64, big: i64, s: &mut FloatStatus) -> i64 {
    let old = s.take_flags();
    let r = f(s);
    let new = s.flags();
    s.exception_flags = new | old;
    if new & ff::INVALID != 0 { big } else { r }
}

fn f32_to_i32(a: Float32, trunc: bool, s: &mut FloatStatus) -> i64 {
    cvt_i(
        |s| i64::from(if trunc { a.to_i32_round_to_zero(s) } else { a.to_i32(s) }),
        i64::from(i32::MIN),
        s,
    )
}

fn f64_to_i32(a: Float64, trunc: bool, s: &mut FloatStatus) -> i64 {
    cvt_i(
        |s| i64::from(if trunc { a.to_i32_round_to_zero(s) } else { a.to_i32(s) }),
        i64::from(i32::MIN),
        s,
    )
}

fn f32_to_i64(a: Float32, trunc: bool, s: &mut FloatStatus) -> i64 {
    cvt_i(|s| if trunc { a.to_i64_round_to_zero(s) } else { a.to_i64(s) }, i64::MIN, s)
}

fn f64_to_i64(a: Float64, trunc: bool, s: &mut FloatStatus) -> i64 {
    cvt_i(|s| if trunc { a.to_i64_round_to_zero(s) } else { a.to_i64(s) }, i64::MIN, s)
}

/// `comis_eflags[]`.
fn comis_eflags(r: FloatRelation) -> u64 {
    u64::from(match r {
        FloatRelation::Less => CC_C,
        FloatRelation::Equal => CC_Z,
        FloatRelation::Greater => 0,
        FloatRelation::Unordered => CC_Z | CC_P | CC_C,
    })
}

/// The predicate of CMPPS and friends for immediate `p` (0 to 31): whether the comparison is
/// signalling and the result for a relation.
fn cmp_pred(p: u32, r: FloatRelation) -> bool {
    use FloatRelation::{Equal, Greater, Less, Unordered};
    let eq = r == Equal;
    let lt = r == Less;
    let le = lt || eq;
    let gt = r == Greater;
    let un = r == Unordered;
    let equ = eq || un;
    let ge_ = eq || gt;
    match p & 15 {
        0 => eq,
        1 => lt,
        2 => le,
        3 => un,
        4 => !eq,
        5 => !lt,
        6 => !le,
        7 => !un,
        8 => equ,
        9 => !ge_,
        10 => !gt,
        11 => false,
        12 => !equ,
        13 => ge_,
        14 => gt,
        _ => true,
    }
}

/// Whether predicate `p` uses the signalling comparison.
fn cmp_signals(p: u32) -> bool {
    // FPU_CMPS for lt, le, nlt, nle, nge, ngt, ge, gt in the first 16, and the opposite
    // choice for the second 16.
    let s = matches!(p & 15, 1 | 2 | 5 | 6 | 9 | 10 | 13 | 14);
    if p & 16 != 0 { !s } else { s }
}

/// Binary floating point operations on one element.
#[derive(Clone, Copy)]
enum Bin {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}

fn bin32(op: Bin, a: Float32, b: Float32, s: &mut FloatStatus) -> Float32 {
    match op {
        Bin::Add => a.add(b, s),
        Bin::Sub => a.sub(b, s),
        Bin::Mul => a.mul(b, s),
        Bin::Div => a.div(b, s),
        Bin::Min => {
            if a.lt(b, s) {
                a
            } else {
                b
            }
        }
        Bin::Max => {
            if b.lt(a, s) {
                a
            } else {
                b
            }
        }
    }
}

fn bin64(op: Bin, a: Float64, b: Float64, s: &mut FloatStatus) -> Float64 {
    match op {
        Bin::Add => a.add(b, s),
        Bin::Sub => a.sub(b, s),
        Bin::Mul => a.mul(b, s),
        Bin::Div => a.div(b, s),
        Bin::Min => {
            if a.lt(b, s) {
                a
            } else {
                b
            }
        }
        Bin::Max => {
            if b.lt(a, s) {
                a
            } else {
                b
            }
        }
    }
}

/// The inputs and output of one kernel call.
struct Kx {
    d: V,
    a: V,
    b: V,
    c: V,
    x: u64,
    var: u32,
    len: usize,
    imm: u32,
    /// The bytes in one 128-bit lane (8 for MMX).
    lane: usize,
}

impl Kx {
    fn n(&self, esz: u32) -> usize {
        self.len >> esz
    }

    /// Elements of size `esz` per lane.
    fn nl(&self, esz: u32) -> usize {
        self.lane >> esz
    }
}

/// Floating point, element-wise binary: ps, pd, ss or sd.
fn fp_bin(k: &Kx, op: Bin, s: &mut FloatStatus) -> V {
    let mut r = k.a;
    let n = match k.var {
        0 => k.n(2),
        1 => k.n(3),
        _ => 1,
    };
    for i in 0..n {
        if k.var & 1 == 0 {
            pe(&mut r, 2, i, u64::from(bin32(op, f32_of(&k.a, i), f32_of(&k.b, i), s).0));
        } else {
            pe(&mut r, 3, i, bin64(op, f64_of(&k.a, i), f64_of(&k.b, i), s).0);
        }
    }
    r
}

fn int_bin(k: &Kx, kk: K) -> V {
    let esz = k.var;
    let mut r = [0; 32];
    for i in 0..k.n(esz) {
        let (x, y) = (ge(&k.a, esz, i), ge(&k.b, esz, i));
        let (sx_, sy) = (sx(x, esz), sx(y, esz));
        let m = emask(esz);
        let bits = 8u32 << esz;
        let v = match kk {
            K::Add => x.wrapping_add(y),
            K::Sub => x.wrapping_sub(y),
            K::AddS => sat_s(sx_ + sy, esz),
            K::AddUS => sat_u(x as i64 + y as i64, esz),
            K::SubS => sat_s(sx_ - sy, esz),
            K::SubUS => sat_u(x as i64 - y as i64, esz),
            K::CmpEq => {
                if x == y {
                    m
                } else {
                    0
                }
            }
            K::CmpGt => {
                if sx_ > sy {
                    m
                } else {
                    0
                }
            }
            K::MinS => sx_.min(sy) as u64,
            K::MaxS => sx_.max(sy) as u64,
            K::MinU => x.min(y),
            K::MaxU => x.max(y),
            K::And => x & y,
            K::AndN => !x & y,
            K::Or => x | y,
            K::Xor => x ^ y,
            K::Avg => (x + y + 1) >> 1,
            K::MulL => (sx_.wrapping_mul(sy)) as u64,
            K::MulHS => ((sx_ * sy) >> bits) as u64,
            K::MulHU => (x * y) >> bits,
            K::MulHRS => ((((sx_ * sy) >> 14) + 1) >> 1) as u64,
            K::Abs => sy.unsigned_abs(),
            K::Sign => {
                if sy < 0 {
                    (sx_.wrapping_neg()) as u64
                } else if sy == 0 {
                    0
                } else {
                    x
                }
            }
            K::ShlV => {
                if y >= u64::from(bits) {
                    0
                } else {
                    x << y
                }
            }
            K::ShrV => {
                if y >= u64::from(bits) {
                    0
                } else {
                    x >> y
                }
            }
            K::SarV => (sx_ >> y.min(u64::from(bits) - 1)) as u64,
            _ => unreachable!(),
        };
        pe(&mut r, esz, i, v & m);
    }
    r
}

/// The shifts by a count: from the low quadword of `b`, or the immediate.
fn shift(k: &Kx, kk: K) -> V {
    let esz = k.var;
    let bits = 8u64 << esz;
    let cnt = match kk {
        K::ShlR | K::ShrR | K::SarR => ge(&k.b, 3, 0),
        _ => u64::from(k.imm),
    };
    let mut r = [0; 32];
    for i in 0..k.n(esz) {
        let x = ge(&k.a, esz, i);
        let v = match kk {
            K::ShlR | K::ShlI => {
                if cnt >= bits {
                    0
                } else {
                    x << cnt
                }
            }
            K::ShrR | K::ShrI => {
                if cnt >= bits {
                    0
                } else {
                    x >> cnt
                }
            }
            _ => (sx(x, esz) >> cnt.min(bits - 1)) as u64,
        };
        pe(&mut r, esz, i, v & emask(esz));
    }
    r
}

/// The element that MOVMSK and BLENDV look at: the sign bit of each element of size `esz`.
fn sign_bit(v: &V, esz: u32, i: usize) -> bool {
    v[(i << esz) + (1 << esz) - 1] & 0x80 != 0
}

fn clmul(a: u64, b: u64) -> u128 {
    let mut r = 0u128;
    for i in 0..64 {
        if (b >> i) & 1 != 0 {
            r ^= u128::from(a) << i;
        }
    }
    r
}

#[allow(clippy::too_many_lines)]
fn kernel(kk: K, k: &Kx, s: &mut FloatStatus) -> (V, u64) {
    let mut r = [0u8; 32];
    let mut ret = 0u64;
    let lanes = k.len / k.lane;
    match kk {
        K::Add
        | K::Sub
        | K::AddS
        | K::AddUS
        | K::SubS
        | K::SubUS
        | K::CmpEq
        | K::CmpGt
        | K::MinS
        | K::MaxS
        | K::MinU
        | K::MaxU
        | K::And
        | K::AndN
        | K::Or
        | K::Xor
        | K::Avg
        | K::MulL
        | K::MulHS
        | K::MulHU
        | K::MulHRS
        | K::Abs
        | K::Sign
        | K::ShlV
        | K::ShrV
        | K::SarV => r = int_bin(k, kk),
        K::ShlR | K::ShrR | K::SarR | K::ShlI | K::ShrI | K::SarI => r = shift(k, kk),
        K::BslI | K::BsrI => {
            let n = k.imm.min(16) as usize;
            for l in 0..lanes {
                let o = l * 16;
                for i in 0..16 {
                    r[o + i] = if kk == K::BslI {
                        if i >= n { k.a[o + i - n] } else { 0 }
                    } else if i + n < 16 {
                        k.a[o + i + n]
                    } else {
                        0
                    };
                }
            }
        }
        K::MaddWd => {
            for i in 0..k.n(2) {
                let p0 = sx(ge(&k.a, 1, 2 * i), 1) * sx(ge(&k.b, 1, 2 * i), 1);
                let p1 = sx(ge(&k.a, 1, 2 * i + 1), 1) * sx(ge(&k.b, 1, 2 * i + 1), 1);
                pe(&mut r, 2, i, (p0 + p1) as u64 & 0xffff_ffff);
            }
        }
        K::MaddUbsw => {
            for i in 0..k.n(1) {
                let p0 = ge(&k.a, 0, 2 * i) as i64 * sx(ge(&k.b, 0, 2 * i), 0);
                let p1 = ge(&k.a, 0, 2 * i + 1) as i64 * sx(ge(&k.b, 0, 2 * i + 1), 0);
                pe(&mut r, 1, i, sat_s(p0 + p1, 1));
            }
        }
        K::SadBw => {
            for i in 0..k.n(3) {
                let mut sum = 0u64;
                for j in 0..8 {
                    sum += (i64::from(k.a[i * 8 + j]) - i64::from(k.b[i * 8 + j])).unsigned_abs();
                }
                pe(&mut r, 3, i, sum);
            }
        }
        K::MulUdq | K::MulDq => {
            for i in 0..k.n(3) {
                let (x, y) = (ge(&k.a, 2, 2 * i), ge(&k.b, 2, 2 * i));
                let v =
                    if kk == K::MulUdq { x * y } else { (sx(x, 2).wrapping_mul(sx(y, 2))) as u64 };
                pe(&mut r, 3, i, v);
            }
        }
        K::PackSS | K::PackUS => {
            // The variant is the destination element size.
            let de = k.var;
            let half = k.nl(de) / 2;
            for l in 0..lanes {
                for i in 0..half {
                    for (src, o) in [(&k.a, 0), (&k.b, half)] {
                        let x = sx(ge(src, de + 1, l * half + i), de + 1);
                        let v = if kk == K::PackSS { sat_s(x, de) } else { sat_u(x, de) };
                        pe(&mut r, de, l * 2 * half + o + i, v);
                    }
                }
            }
        }
        K::UnpckL | K::UnpckH => {
            let esz = k.var;
            let half = k.nl(esz) / 2;
            for l in 0..lanes {
                let base = l * 2 * half + if kk == K::UnpckH { half } else { 0 };
                for i in 0..half {
                    pe(&mut r, esz, l * 2 * half + 2 * i, ge(&k.a, esz, base + i));
                    pe(&mut r, esz, l * 2 * half + 2 * i + 1, ge(&k.b, esz, base + i));
                }
            }
        }
        K::ShufB => {
            for l in 0..lanes {
                let o = l * k.lane;
                for i in 0..k.lane {
                    let sel = k.b[o + i];
                    r[o + i] =
                        if sel & 0x80 != 0 { 0 } else { k.a[o + (sel as usize & (k.lane - 1))] };
                }
            }
        }
        K::HAdd | K::HSub | K::HAddS | K::HSubS => {
            let esz = k.var;
            let half = k.nl(esz) / 2;
            for l in 0..lanes {
                for (src, o) in [(&k.a, 0), (&k.b, half)] {
                    for i in 0..half {
                        let x = sx(ge(src, esz, l * 2 * half + 2 * i), esz);
                        let y = sx(ge(src, esz, l * 2 * half + 2 * i + 1), esz);
                        let v = match kk {
                            K::HAdd => x.wrapping_add(y) as u64,
                            K::HSub => x.wrapping_sub(y) as u64,
                            K::HAddS => sat_s(x + y, esz),
                            _ => sat_s(x - y, esz),
                        };
                        pe(&mut r, esz, l * 2 * half + o + i, v & emask(esz));
                    }
                }
            }
        }
        K::AlignR => {
            let n = k.imm as usize;
            for l in 0..lanes {
                let o = l * k.lane;
                for i in 0..k.lane {
                    let j = i + n;
                    r[o + i] = if j < k.lane {
                        k.b[o + j]
                    } else if j < 2 * k.lane {
                        k.a[o + j - k.lane]
                    } else {
                        0
                    };
                }
            }
        }
        K::PshufD => {
            for i in 0..k.n(2) {
                let sel = (k.imm >> (2 * (i % 4))) & 3;
                pe(&mut r, 2, i, ge(&k.a, 2, (i & !3) + sel as usize));
            }
        }
        K::PshufHW | K::PshufLW | K::PshufW => {
            r = k.a;
            for i in 0..k.n(1) {
                let q = i % 8;
                let hi = q >= 4;
                let touch = match kk {
                    K::PshufHW => hi,
                    K::PshufLW => !hi,
                    _ => true,
                };
                if touch {
                    let sel = (k.imm >> (2 * (q % 4))) & 3;
                    pe(&mut r, 1, i, ge(&k.a, 1, (i & !3) + sel as usize));
                }
            }
        }
        K::BlendW => {
            for i in 0..k.n(1) {
                let src = if (k.imm >> (i % 8)) & 1 != 0 { &k.b } else { &k.a };
                pe(&mut r, 1, i, ge(src, 1, i));
            }
        }
        K::Blend => {
            let esz = k.var;
            for i in 0..k.n(esz) {
                let src = if (k.imm >> i) & 1 != 0 { &k.b } else { &k.a };
                pe(&mut r, esz, i, ge(src, esz, i));
            }
        }
        K::BlendV => {
            let esz = k.var;
            for i in 0..k.n(esz) {
                let src = if sign_bit(&k.c, esz, i) { &k.b } else { &k.a };
                pe(&mut r, esz, i, ge(src, esz, i));
            }
        }
        K::MovSx | K::MovZx => {
            let (se, de) = (k.var & 3, k.var >> 2);
            for i in 0..k.n(de) {
                let x = ge(&k.b, se, i);
                let v = if kk == K::MovSx { sx(x, se) as u64 } else { x };
                pe(&mut r, de, i, v & emask(de));
            }
        }
        K::Bcast => {
            let n = 1usize << k.var;
            for i in 0..k.len / n {
                r[i * n..(i + 1) * n].copy_from_slice(&k.b[..n]);
            }
        }
        K::Ptest => {
            // The variant is 0 for PTEST, 2 for VTESTPS and 3 for VTESTPD.
            let (mut zf, mut cf) = (true, true);
            for i in 0..k.len {
                let m = if k.var == 0 {
                    0xff
                } else if (i + 1) % (1 << k.var) == 0 {
                    0x80
                } else {
                    0
                };
                zf &= k.a[i] & k.b[i] & m == 0;
                cf &= !k.a[i] & k.b[i] & m == 0;
            }
            ret = u64::from(if zf { CC_Z } else { 0 } | if cf { CC_C } else { 0 });
        }
        K::MovMsk => {
            for i in 0..k.n(k.var) {
                if sign_bit(&k.b, k.var, i) {
                    ret |= 1 << i;
                }
            }
        }
        K::Extr => {
            let esz = k.var;
            ret = ge(&k.a, esz, k.imm as usize & (16usize >> esz).wrapping_sub(1));
        }
        K::Insr => {
            let esz = k.var;
            r = k.a;
            let n = if k.len == 8 { 4 } else { 16usize >> esz };
            pe(&mut r, esz, k.imm as usize & (n - 1), k.x & emask(esz));
        }
        K::Phminposuw => {
            let mut idx = 0;
            for i in 1..8 {
                if ge(&k.b, 1, i) < ge(&k.b, 1, idx) {
                    idx = i;
                }
            }
            pe(&mut r, 1, 0, ge(&k.b, 1, idx));
            pe(&mut r, 1, 1, idx as u64);
        }
        K::Mpsadbw => {
            for l in 0..lanes {
                let ctl = k.imm >> (3 * l);
                let so = (ctl & 3) as usize * 4;
                let dof = (ctl & 4) as usize;
                for i in 0..8 {
                    let mut sum = 0u64;
                    for j in 0..4 {
                        let x = i64::from(k.a[l * 16 + dof + i + j]);
                        let y = i64::from(k.b[l * 16 + so + j]);
                        sum += (x - y).unsigned_abs();
                    }
                    pe(&mut r, 1, l * 8 + i, sum);
                }
            }
        }
        K::Pclmul => {
            for l in 0..lanes {
                let x = ge(&k.a, 3, 2 * l + (k.imm & 1) as usize);
                let y = ge(&k.b, 3, 2 * l + ((k.imm >> 4) & 1) as usize);
                let p = clmul(x, y);
                pe(&mut r, 3, 2 * l, p as u64);
                pe(&mut r, 3, 2 * l + 1, (p >> 64) as u64);
            }
        }
        K::PermD => {
            for i in 0..8 {
                pe(&mut r, 2, i, ge(&k.b, 2, ge(&k.a, 2, i) as usize & 7));
            }
        }
        K::PermQ => {
            for i in 0..4 {
                pe(&mut r, 3, i, ge(&k.a, 3, ((k.imm >> (2 * i)) & 3) as usize));
            }
        }
        K::Perm2 => {
            for l in 0..2 {
                let ctl = k.imm >> (4 * l);
                if ctl & 8 == 0 {
                    let sel = (ctl & 3) as usize;
                    let src = if sel < 2 { &k.a } else { &k.b };
                    let o = (sel & 1) * 16;
                    r[l * 16..l * 16 + 16].copy_from_slice(&src[o..o + 16]);
                }
            }
        }
        K::PermilV | K::PermilI => {
            if k.var == 2 {
                for i in 0..k.n(2) {
                    let sel = if kk == K::PermilV {
                        ge(&k.b, 2, i) & 3
                    } else {
                        u64::from((k.imm >> (2 * (i % 4))) & 3)
                    };
                    pe(&mut r, 2, i, ge(&k.a, 2, (i & !3) + sel as usize));
                }
            } else {
                for i in 0..k.n(3) {
                    let sel = if kk == K::PermilV {
                        (ge(&k.b, 3, i) >> 1) & 1
                    } else {
                        u64::from((k.imm >> i) & 1)
                    };
                    pe(&mut r, 3, i, ge(&k.a, 3, (i & !1) + sel as usize));
                }
            }
        }
        K::ShufP => {
            if k.var == 2 {
                for l in 0..lanes {
                    let b = l * 4;
                    for i in 0..4 {
                        let src = if i < 2 { &k.a } else { &k.b };
                        let sel = ((k.imm >> (2 * i)) & 3) as usize;
                        pe(&mut r, 2, b + i, ge(src, 2, b + sel));
                    }
                }
            } else {
                for l in 0..lanes {
                    let b = l * 2;
                    let s0 = ((k.imm >> (2 * l)) & 1) as usize;
                    let s1 = ((k.imm >> (2 * l + 1)) & 1) as usize;
                    pe(&mut r, 3, b, ge(&k.a, 3, b + s0));
                    pe(&mut r, 3, b + 1, ge(&k.b, 3, b + s1));
                }
            }
        }
        K::MovSlDup | K::MovShDup => {
            let o = usize::from(kk == K::MovShDup);
            for i in 0..k.n(3) {
                let v = ge(&k.b, 2, 2 * i + o);
                pe(&mut r, 2, 2 * i, v);
                pe(&mut r, 2, 2 * i + 1, v);
            }
        }
        K::MovDDup => {
            for i in 0..k.n(4) {
                let v = ge(&k.b, 3, 2 * i);
                pe(&mut r, 3, 2 * i, v);
                pe(&mut r, 3, 2 * i + 1, v);
            }
        }
        K::MaskLd => {
            let esz = k.var;
            for i in 0..k.n(esz) {
                if sign_bit(&k.a, esz, i) {
                    pe(&mut r, esz, i, ge(&k.b, esz, i));
                }
            }
        }
        K::InsertPs => {
            // The variant is 1 when the source comes from memory, in `x`.
            let val = k.imm;
            r = k.a;
            let tmp = if k.var == 1 { k.x & 0xffff_ffff } else { ge(&k.b, 2, (val >> 6) as usize) };
            pe(&mut r, 2, ((val >> 4) & 3) as usize, tmp);
            for i in 0..4 {
                if (val >> i) & 1 != 0 {
                    pe(&mut r, 2, i, 0);
                }
            }
        }
        K::Extract128 => {
            let o = (k.imm as usize & 1) * 16;
            r[..16].copy_from_slice(&k.a[o..o + 16]);
        }
        K::Insert128 => {
            r = k.a;
            let o = (k.imm as usize & 1) * 16;
            r[o..o + 16].copy_from_slice(&k.b[..16]);
        }
        K::Copy => r = k.b,
        K::FAdd => r = fp_bin(k, Bin::Add, s),
        K::FSub => r = fp_bin(k, Bin::Sub, s),
        K::FMul => r = fp_bin(k, Bin::Mul, s),
        K::FDiv => r = fp_bin(k, Bin::Div, s),
        K::FMin => r = fp_bin(k, Bin::Min, s),
        K::FMax => r = fp_bin(k, Bin::Max, s),
        K::FSqrt | K::FRsqrt | K::FRcp => {
            // Scalar forms keep the upper part of the first source; packed ones read `b`.
            r = k.a;
            let n = match k.var {
                0 => k.n(2),
                1 => k.n(3),
                _ => 1,
            };
            // QEMU's RSQRT and RCP are exact and leave the flags alone.
            let mut tmp = *s;
            let st = if kk == K::FSqrt { &mut *s } else { &mut tmp };
            for i in 0..n {
                if k.var & 1 == 0 {
                    let x = f32_of(&k.b, i);
                    let one = Float32(0x3f80_0000);
                    let v = match kk {
                        K::FSqrt => x.sqrt(st),
                        K::FRsqrt => one.div(x.sqrt(st), st),
                        _ => one.div(x, st),
                    };
                    pe(&mut r, 2, i, u64::from(v.0));
                } else {
                    pe(&mut r, 3, i, f64_of(&k.b, i).sqrt(st).0);
                }
            }
        }
        K::FHAdd | K::FHSub => {
            let esz = 2 + (k.var & 1);
            let half = k.nl(esz) / 2;
            for l in 0..lanes {
                for (src, o) in [(&k.a, 0), (&k.b, half)] {
                    for i in 0..half {
                        let j = l * 2 * half + 2 * i;
                        let v = if esz == 2 {
                            let (x, y) = (f32_of(src, j), f32_of(src, j + 1));
                            u64::from(if kk == K::FHAdd { x.add(y, s) } else { x.sub(y, s) }.0)
                        } else {
                            let (x, y) = (f64_of(src, j), f64_of(src, j + 1));
                            if kk == K::FHAdd { x.add(y, s) } else { x.sub(y, s) }.0
                        };
                        pe(&mut r, esz, l * 2 * half + o + i, v);
                    }
                }
            }
        }
        K::FAddSub => {
            let esz = 2 + (k.var & 1);
            for i in 0..k.n(esz) {
                let add = i & 1 != 0;
                let v = if esz == 2 {
                    let (x, y) = (f32_of(&k.a, i), f32_of(&k.b, i));
                    u64::from(if add { x.add(y, s) } else { x.sub(y, s) }.0)
                } else {
                    let (x, y) = (f64_of(&k.a, i), f64_of(&k.b, i));
                    if add { x.add(y, s) } else { x.sub(y, s) }.0
                };
                pe(&mut r, esz, i, v);
            }
        }
        K::FCmp => {
            r = k.a;
            let esz = 2 + (k.var & 1);
            let n = if k.var >= 2 { 1 } else { k.n(esz) };
            let sig = cmp_signals(k.imm);
            for i in 0..n {
                let rel = if esz == 2 {
                    let (x, y) = (f32_of(&k.a, i), f32_of(&k.b, i));
                    if sig { x.compare(y, s) } else { x.compare_quiet(y, s) }
                } else {
                    let (x, y) = (f64_of(&k.a, i), f64_of(&k.b, i));
                    if sig { x.compare(y, s) } else { x.compare_quiet(y, s) }
                };
                pe(&mut r, esz, i, if cmp_pred(k.imm, rel) { emask(esz) } else { 0 });
            }
        }
        K::Comi | K::Ucomi => {
            let q = kk == K::Ucomi;
            let rel = if k.var & 1 == 0 {
                let (x, y) = (f32_of(&k.a, 0), f32_of(&k.b, 0));
                if q { x.compare_quiet(y, s) } else { x.compare(y, s) }
            } else {
                let (x, y) = (f64_of(&k.a, 0), f64_of(&k.b, 0));
                if q { x.compare_quiet(y, s) } else { x.compare(y, s) }
            };
            ret = comis_eflags(rel);
        }
        K::Round => {
            // Packed forms read `b`; scalar ones merge into `a`.
            r = k.a;
            if k.imm & 4 == 0 {
                s.rounding_mode = x86_rmode(k.imm);
            }
            let n = match k.var {
                0 => k.n(2),
                1 => k.n(3),
                _ => 1,
            };
            for i in 0..n {
                if k.var & 1 == 0 {
                    pe(&mut r, 2, i, u64::from(f32_of(&k.b, i).round_to_int(s).0));
                } else {
                    pe(&mut r, 3, i, f64_of(&k.b, i).round_to_int(s).0);
                }
            }
            if k.imm & 8 != 0 {
                s.exception_flags &= !ff::INEXACT;
            }
        }
        K::Dpp => {
            let m = k.imm;
            if k.var == 0 {
                for l in 0..lanes {
                    let o = l * 4;
                    let p = |i: usize, s: &mut FloatStatus| {
                        if (m >> (4 + i)) & 1 != 0 {
                            f32_of(&k.a, o + i).mul(f32_of(&k.b, o + i), s)
                        } else {
                            Float32(0)
                        }
                    };
                    let (p0, p1) = (p(0, s), p(1, s));
                    let t2 = p0.add(p1, s);
                    let (p2, p3) = (p(2, s), p(3, s));
                    let t3 = p2.add(p3, s);
                    let t4 = t2.add(t3, s);
                    for i in 0..4 {
                        let v = if (m >> i) & 1 != 0 { t4.0 } else { 0 };
                        pe(&mut r, 2, o + i, u64::from(v));
                    }
                }
            } else {
                let p = |i: usize, s: &mut FloatStatus| {
                    if (m >> (4 + i)) & 1 != 0 {
                        f64_of(&k.a, i).mul(f64_of(&k.b, i), s)
                    } else {
                        Float64(0)
                    }
                };
                let (p0, p1) = (p(0, s), p(1, s));
                let t = p0.add(p1, s);
                for i in 0..2 {
                    pe(&mut r, 3, i, if (m >> i) & 1 != 0 { t.0 } else { 0 });
                }
            }
        }
        K::Fma => {
            // even | odd << 3 | pd << 6 | scalar << 7.
            r = k.d;
            let (even, odd) = (k.var & 7, (k.var >> 3) & 7);
            let pd = k.var & 0x40 != 0;
            let esz = if pd { 3 } else { 2 };
            let n = if k.var & 0x80 != 0 { 1 } else { k.n(esz) };
            for i in 0..n {
                let fl = if i & 1 != 0 { odd } else { even };
                if pd {
                    let v = f64_of(&k.a, i).muladd(f64_of(&k.b, i), f64_of(&k.c, i), fl, s);
                    pe(&mut r, 3, i, v.0);
                } else {
                    let v = f32_of(&k.a, i).muladd(f32_of(&k.b, i), f32_of(&k.c, i), fl, s);
                    pe(&mut r, 2, i, u64::from(v.0));
                }
            }
        }
        K::CvtDq2Ps => {
            for i in 0..k.n(2) {
                let v = Float32::from_i32(ge(&k.b, 2, i) as u32 as i32, s);
                pe(&mut r, 2, i, u64::from(v.0));
            }
        }
        K::CvtPs2Dq | K::CvttPs2Dq => {
            for i in 0..k.n(2) {
                let v = f32_to_i32(f32_of(&k.b, i), kk == K::CvttPs2Dq, s);
                pe(&mut r, 2, i, v as u64 & 0xffff_ffff);
            }
        }
        K::CvtDq2Pd => {
            for i in 0..k.n(3) {
                pe(&mut r, 3, i, Float64::from_i32(ge(&k.b, 2, i) as u32 as i32, s).0);
            }
        }
        K::CvtPd2Dq | K::CvttPd2Dq => {
            for i in 0..k.n(3) {
                let v = f64_to_i32(f64_of(&k.b, i), kk == K::CvttPd2Dq, s);
                pe(&mut r, 2, i, v as u64 & 0xffff_ffff);
            }
        }
        K::CvtPs2Pd => {
            for i in 0..k.n(3) {
                pe(&mut r, 3, i, f32_of(&k.b, i).to_float64(s).0);
            }
        }
        K::CvtPd2Ps => {
            for i in 0..k.n(3) {
                pe(&mut r, 2, i, u64::from(f64_of(&k.b, i).to_float32(s).0));
            }
        }
        K::CvtSs2Sd => {
            r = k.a;
            pe(&mut r, 3, 0, f32_of(&k.b, 0).to_float64(s).0);
        }
        K::CvtSd2Ss => {
            r = k.a;
            pe(&mut r, 2, 0, u64::from(f64_of(&k.b, 0).to_float32(s).0));
        }
        K::CvtSi2S => {
            // bit 0: double, bit 1: 64-bit source.
            r = k.a;
            let x = if k.var & 2 != 0 { k.x as i64 } else { i64::from(k.x as u32 as i32) };
            if k.var & 1 != 0 {
                pe(&mut r, 3, 0, Float64::from_i64(x, s).0);
            } else {
                pe(&mut r, 2, 0, u64::from(Float32::from_i64(x, s).0));
            }
        }
        K::CvtS2Si => {
            // bit 0: double, bit 1: 64-bit result, bit 2: truncate.
            let t = k.var & 4 != 0;
            ret = match k.var & 3 {
                0 => f32_to_i32(f32_of(&k.b, 0), t, s) as u64 & 0xffff_ffff,
                1 => f64_to_i32(f64_of(&k.b, 0), t, s) as u64 & 0xffff_ffff,
                2 => f32_to_i64(f32_of(&k.b, 0), t, s) as u64,
                _ => f64_to_i64(f64_of(&k.b, 0), t, s) as u64,
            };
        }
        K::CvtPi2Ps => {
            r = k.d;
            for i in 0..2 {
                let v = Float32::from_i32(ge(&k.b, 2, i) as u32 as i32, s);
                pe(&mut r, 2, i, u64::from(v.0));
            }
        }
        K::CvtPi2Pd => {
            for i in 0..2 {
                pe(&mut r, 3, i, Float64::from_i32(ge(&k.b, 2, i) as u32 as i32, s).0);
            }
        }
        K::CvtPs2Pi | K::CvtPd2Pi => {
            let t = k.var & 4 != 0;
            for i in 0..2 {
                let v = if kk == K::CvtPs2Pi {
                    f32_to_i32(f32_of(&k.b, i), t, s)
                } else {
                    f64_to_i32(f64_of(&k.b, i), t, s)
                };
                pe(&mut r, 2, i, v as u64 & 0xffff_ffff);
            }
        }
        K::CvtPh2Ps => {
            for i in 0..k.n(2) {
                let h = Float16(ge(&k.b, 1, i) as u16);
                pe(&mut r, 2, i, u64::from(h.to_float32(true, s).0));
            }
        }
        K::CvtPs2Ph => {
            if k.imm & 4 == 0 {
                s.rounding_mode = x86_rmode(k.imm);
            }
            for i in 0..k.n(2) {
                pe(&mut r, 1, i, u64::from(f32_of(&k.a, i).to_float16(true, s).0));
            }
        }
    }
    (r, ret)
}

fn h_sse(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let op = a32(a, 1);
        let kk = K::from_u8(op as u8);
        let var = (op >> 8) & 0xff;
        let len = 8usize << ((op >> 16) & 3);
        let imm = op >> 24;
        let offs = a[2];
        let off = |i: u32| ((offs >> (16 * i)) & 0xffff) as usize;
        let env = &mut *cpu.env;
        let k = Kx {
            d: rd(env, off(0)),
            a: rd(env, off(1)),
            b: rd(env, off(2)),
            c: rd(env, off(3)),
            x: a[3],
            var,
            len,
            imm,
            lane: len.min(16),
        };
        let mxcsr = ld32(env, MXCSR);
        let mut s = sse_status(mxcsr);
        let (r, ret) = kernel(kk, &k, &mut s);
        let d = off(0);
        let n = len.min(env.len().saturating_sub(d));
        env[d..d + n].copy_from_slice(&r[..n]);
        let fl = mxcsr_flags(s.flags());
        if fl != 0 {
            st32(env, MXCSR, mxcsr | fl);
        }
        Ok(ret)
    })
}

// MMX state.

def!(EMMS, "x86_emms", NO_RWG, Void, [Ptr], h_emms);
def!(ENTER_MMX, "x86_enter_mmx", NO_RWG, Void, [Ptr], h_enter_mmx);

fn h_emms(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        st64(cpu.env, FPTAGS, 0x0101_0101_0101_0101);
        Ok(0)
    })
}

fn h_enter_mmx(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        st32(cpu.env, FPSTT, 0);
        st64(cpu.env, FPTAGS, 0);
        Ok(0)
    })
}

// XCR0.

/// `XSTATE_FP_MASK`, `XSTATE_SSE_MASK`, `XSTATE_YMM_MASK` and the MPX masks.
const XSTATE_FP: u64 = 1;
const XSTATE_SSE: u64 = 2;
const XSTATE_YMM: u64 = 4;
const XSTATE_BNDREGS: u64 = 8;
const XSTATE_BNDCSR: u64 = 16;

def!(XGETBV, "x86_xgetbv", 0, I64, [Ptr, I32], h_xgetbv);
def!(XSETBV, "x86_xsetbv", 0, Void, [Ptr, I32, I64], h_xsetbv);

fn h_xgetbv(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        if ld64(cpu.env, cr(4)) & CR4_OSXSAVE_MASK == 0 {
            return Err(sh::raise_exception_ra(cpu, EXCP06_ILLOP, TB));
        }
        match a32(a, 1) {
            0 => return Ok(ld64(cpu.env, XCR0)),
            1 => {
                let ops = cpu.ops();
                let m = x86_of(&ops).model();
                if m.has_feature("xgetbv1") {
                    // get_xinuse(): everything but BNDREGS, which is not tracked.
                    return Ok(ld64(cpu.env, XCR0) & !XSTATE_BNDREGS);
                }
            }
            _ => {}
        }
        Err(sh::raise_exception_ra(cpu, EXCP0D_GPF, TB))
    })
}

fn h_xsetbv(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let cr4 = ld64(cpu.env, cr(4));
        if cr4 & CR4_OSXSAVE_MASK == 0 {
            return Err(sh::raise_exception_ra(cpu, EXCP06_ILLOP, TB));
        }
        let mask = a[2];
        let ena = {
            let ops = cpu.ops();
            let r = x86_of(&ops).model().cpuid(0x0d, 0);
            u64::from(r[0]) | u64::from(r[3]) << 32
        };
        let bad = a32(a, 1) != 0
            || mask & XSTATE_FP == 0
            || mask & (XSTATE_SSE | XSTATE_YMM) == XSTATE_YMM
            || mask & !ena != 0
            || (mask ^ (mask.wrapping_mul(XSTATE_BNDCSR / XSTATE_BNDREGS))) & XSTATE_BNDCSR != 0;
        if bad {
            return Err(sh::raise_exception_ra(cpu, EXCP0D_GPF, TB));
        }
        st64(cpu.env, XCR0, mask);
        // cpu_sync_avx_hflag().
        let mut hf = hflags(cpu) & !HF_AVX_EN_MASK;
        if avx_enabled(cr4, mask) {
            hf |= HF_AVX_EN_MASK;
        }
        set_hflags(cpu, hf);
        Ok(0)
    })
}

// Integer leftovers.

def!(RDRAND, "x86_rdrand", 0, I64, [Ptr], h_rdrand);
def!(RDPID, "x86_rdpid", NO_RWG, I64, [Ptr], h_rdpid);
def!(CRC32, "x86_crc32", NO_RWG, I64, [I32, I64, I32], h_crc32);
def!(PDEP, "x86_pdep", NO_RWG, I64, [I64, I64], h_pdep);
def!(PEXT, "x86_pext", NO_RWG, I64, [I64, I64], h_pext);
def!(CR4_TESTBIT, "x86_cr4_testbit", 0, Void, [Ptr, I32], h_cr4_testbit);

/// A random 64-bit value from the host, `qemu_guest_getrandom()`.
fn guest_random() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    static CTR: AtomicU64 = AtomicU64::new(0);
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(CTR.fetch_add(1, Ordering::Relaxed));
    h.finish()
}

fn h_rdrand(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        // Success sets CF and clears the other flags.
        st64(cpu.env, CC_SRC, u64::from(CC_C));
        st32(cpu.env, CC_OP, CC_OP_EFLAGS);
        Ok(guest_random())
    })
}

fn h_rdpid(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| Ok(ld64(cpu.env, TSC_AUX)))
}

/// `helper_crc32()`: CRC-32C of the low `len` bits of `msg`.
fn h_crc32(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    let len = a32(a, 2);
    let mut crc = (a[1] & (u64::MAX >> (64 - len))) ^ u64::from(a32(a, 0));
    for _ in 0..len {
        crc = (crc >> 1) ^ if crc & 1 != 0 { 0x82f6_3b78 } else { 0 };
    }
    Ok(u128::from(crc))
}

fn h_pdep(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    let (mut src, mut mask) = (a[0], a[1]);
    let mut dest = 0u64;
    while mask != 0 {
        let low = mask & mask.wrapping_neg();
        if src & 1 != 0 {
            dest |= low;
        }
        src >>= 1;
        mask &= mask - 1;
    }
    Ok(u128::from(dest))
}

fn h_pext(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    let (src, mut mask) = (a[0], a[1]);
    let mut dest = 0u64;
    let mut i = 0;
    while mask != 0 {
        let low = mask & mask.wrapping_neg();
        if src & low != 0 {
            dest |= 1 << i;
        }
        i += 1;
        mask &= mask - 1;
    }
    Ok(u128::from(dest))
}

fn h_cr4_testbit(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        if ld64(cpu.env, cr(4)) & u64::from(a32(a, 1)) == 0 {
            return Err(sh::raise_exception_ra(cpu, EXCP06_ILLOP, TB));
        }
        Ok(0)
    })
}

/// The helpers of this module, for [`super::register`].
pub(super) const ALL: &[&Def] = &[
    &SSE,
    &EMMS,
    &ENTER_MMX,
    &XGETBV,
    &XSETBV,
    &RDRAND,
    &RDPID,
    &CRC32,
    &PDEP,
    &PEXT,
    &CR4_TESTBIT,
];
