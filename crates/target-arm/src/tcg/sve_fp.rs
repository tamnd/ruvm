// SPDX-License-Identifier: GPL-2.0-or-later

//! The SVE and SVE2 floating point helpers: the floating point parts of QEMU's
//! `sve_helper.c` and the SVE users of `vec_helper.c` (the indexed multiplies, FCMLA by
//! element, FTSMUL, FRECPS and FRSQRTS), with the FEAT_BF16 operations (BFCVT, BFCVTNT,
//! BFDOT, BFMMLA, BFMLALB and BFMLALT, which follow `is_ebf()` with FPCR.EBF clear as
//! FEAT_EBF16 is not implemented) and FMLALB, FMLALT, FMLSLB and FMLSLT.
//!
//! They are one family of the `sve` helper ([`super::sve_helper::fam::FP`]). The descriptor
//! names the operation ([`f`] or a [`super::vfp::op`] number), the element size (the
//! container size for the conversions) and the registers. Its data field holds the shape of
//! the operation ([`sh`]) in bits 0 to 3, the status in bit 4 (`FPST_A64_F16` when set,
//! else `FPST_A64`), a rounding mode that replaces the FPCR one for the duration in bits 5
//! to 7 (zero for none, else the [`RoundMode`] plus one) and operation specific bits from
//! bit 8 up. Every element operation goes through `ruvm-softfloat` with the status QEMU
//! would pick, so the results, the default NaNs and the cumulative FPSR flags are those of
//! QEMU bit for bit.
//!
//! Differences from QEMU:
//!
//! - FPCR.AH is always zero in this port (no FEAT_AFP), so the `ah_` variants of the helpers
//!   (and the AH forms of FRECPE and FRSQRTE with FEAT_RPRES) do not exist.
//! - The BFloat16 forms (element size 0 of the arithmetic, FMLA and indexed groups, which
//!   need FEAT_SVE_B16B16) are not implemented; the translator treats them as unallocated,
//!   which is what the architecture says without FEAT_SVE_B16B16. QEMU 11.1 accepts BFMUL
//!   (indexed) without checking FEAT_SVE_B16B16.

#![allow(clippy::needless_range_loop)]

use ruvm_softfloat::{Float16, Float32, Float64, FloatRelation, FloatStatus, RoundMode, flags};

use super::sve_helper::{Dsc, P, Z, act, get, pload, pstore, set, zload, zstore};
use super::vfp::{Fp, binop, cvt, fcmp, load_status, op, recpe, recpx, rint, rsqrte, store_status};
use crate::cpu::{PREG_SIZE, ZREG_SIZE};

/// The shapes of [`super::sve_helper::fam::FP`], in data bits 0 to 3.
#[allow(missing_docs)]
pub(crate) mod sh {
    /// Predicated, merging: `d[i] = op(n[i], m[i])`.
    pub(crate) const ZPZZ: u32 = 0;
    /// Predicated, merging: `d[i] = op(n[i], x)`.
    pub(crate) const ZPZS: u32 = 1;
    /// Unpredicated: `d[i] = op(n[i], m[i])`.
    pub(crate) const ZZZ: u32 = 2;
    /// SVE2 pairwise, predicated.
    pub(crate) const PAIR: u32 = 3;
    /// Predicated multiply-add with Za; extra bit 0 negates Zn, bit 1 Za.
    pub(crate) const MLA: u32 = 4;
    /// Compares into Pd; extra bit 0 compares with zero, bit 1 swaps the operands.
    pub(crate) const CMP: u32 = 5;
    /// Reductions to Vd.
    pub(crate) const RED: u32 = 6;
    /// Predicated unary, merging, on containers of the descriptor's element size.
    pub(crate) const UN: u32 = 7;
    /// Unpredicated unary.
    pub(crate) const UNU: u32 = 8;
    /// FCADD and FCMLA (vectors); extra holds the rotation.
    pub(crate) const CPLX: u32 = 9;
    /// The indexed forms; extra holds the index (and the FCMLA rotation in bits 0 and 1).
    pub(crate) const IDX: u32 = 10;
    /// FTMAD, FTSSEL and FMMLA.
    pub(crate) const MISC: u32 = 11;
}

/// The SVE specific operations of [`super::sve_helper::fam::FP`]. The others are
/// [`super::vfp::op`] numbers.
#[allow(missing_docs)]
pub(crate) mod f {
    pub(crate) const SCALE: u32 = 100;
    pub(crate) const SUBR: u32 = 101;
    pub(crate) const TSMUL: u32 = 102;
    pub(crate) const CNE: u32 = 103;
    pub(crate) const CUO: u32 = 104;
    pub(crate) const ADDA: u32 = 105;
    pub(crate) const TSSEL: u32 = 106;
    pub(crate) const EXPA: u32 = 107;
    pub(crate) const TMAD: u32 = 108;
    pub(crate) const MMLA: u32 = 109;
    pub(crate) const CADD: u32 = 110;
    pub(crate) const CMLA: u32 = 111;
    /// Precision conversion; extra bits 0 and 1 hold the source size, 2 and 3 the result's.
    pub(crate) const CVT: u32 = 112;
    /// Float to integer toward zero; the sizes as for CVT.
    pub(crate) const TOSINT: u32 = 113;
    pub(crate) const TOUINT: u32 = 114;
    /// Integer to float; the sizes as for CVT.
    pub(crate) const SCVTF: u32 = 115;
    pub(crate) const UCVTF: u32 = 116;
    pub(crate) const LOGB: u32 = 117;
    /// Narrow into the top half of each container.
    pub(crate) const CVTNT: u32 = 118;
    /// Widen the top half of each container.
    pub(crate) const CVTLT: u32 = 119;
    /// BFCVT and BFCVTNT (extra bit 0) on 32-bit containers.
    pub(crate) const BFCVT: u32 = 120;
    /// FMLALB, FMLALT, FMLSLB and FMLSLT: extra bit 0 subtracts, bit 1 picks the top
    /// halves, bit 2 asks for the indexed form and bits 3 to 5 hold the index.
    pub(crate) const FMLAL: u32 = 121;
    /// BFDOT: extra bit 0 asks for the indexed form, bits 1 and 2 hold the index.
    pub(crate) const BFDOT: u32 = 122;
    pub(crate) const BFMMLA: u32 = 123;
    /// BFMLALB and BFMLALT: extra bit 0 picks the top halves, bit 1 asks for the indexed
    /// form and bits 2 to 4 hold the index.
    pub(crate) const BFMLAL: u32 = 124;
}

/// The data field of a call: shape, status (from the element size, as QEMU picks it for
/// most instructions) and extra bits.
pub(crate) fn data(shape: u32, f16: bool, rmode: Option<RoundMode>, extra: u32) -> u32 {
    let rm = rmode.map_or(0, |r| r as u32 + 1);
    shape | u32::from(f16) << 4 | rm << 5 | extra << 8
}

/// Run the `FP` family of the `sve` helper; `x` is the scalar operand of the ZPZS shape.
pub(crate) fn fp(env: &mut [u8], d: &Dsc, x: u64) {
    let idx = ((d.data >> 4) & 1) as usize;
    let rm = (d.data >> 5) & 7;
    let mut s = load_status(env, idx);
    let saved = s.rounding_mode;
    if rm != 0 {
        s.rounding_mode = RoundMode::from_u8((rm - 1) as u8).unwrap_or(saved);
    }
    let conv_op =
        matches!(d.op, f::CVT | f::TOSINT | f::TOUINT | f::SCVTF | f::UCVTF | f::CVTNT | f::CVTLT);
    if conv_op {
        conv(env, d, &mut s);
    } else if d.op >= f::BFCVT {
        widen(env, d, &mut s);
    } else {
        match d.esz {
            1 => run::<Float16>(env, d, x, &mut s),
            2 => run::<Float32>(env, d, x, &mut s),
            _ => run::<Float64>(env, d, x, &mut s),
        }
    }
    s.rounding_mode = saved;
    store_status(env, idx, &s);
}

fn el<F: Fp>(z: &Z, i: usize) -> F {
    F::from_bits(get(z, F::ESZ, i))
}

/// The element operations of the ZPZZ, ZPZS and ZZZ shapes; `b` is the raw second operand.
fn op2<F: Fp>(op: u32, a: F, b: u64, s: &mut FloatStatus) -> u64 {
    match op {
        f::SCALE => {
            // The second operand is a signed integer of the element size; FSCALE on doubles
            // clamps it to the int range first, as `scalbn_d()` does.
            let k = match F::ESZ {
                1 => i32::from(b as u16 as i16),
                2 => b as u32 as i32,
                _ => (b as i64).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32,
            };
            a.scalbn(k, s).bits()
        }
        f::SUBR => F::from_bits(b).sub(a, s).bits(),
        f::TSMUL => {
            let r = a.mul(a, s);
            if r.is_any_nan() {
                r.bits()
            } else {
                (r.bits() & !F::sign_bit()) | if b & 1 != 0 { F::sign_bit() } else { 0 }
            }
        }
        f::TSSEL => {
            let mut r = if b & 1 != 0 { F::pow2(0).bits() } else { a.bits() };
            if b & 2 != 0 {
                r ^= F::sign_bit();
            }
            r
        }
        _ => binop(op, a, F::from_bits(b), F::zero(), s),
    }
}

/// The recursive pairwise reduction of `DO_REDUCE`.
fn tree<F: Fp>(v: &[F], op: u32, s: &mut FloatStatus) -> F {
    if v.len() == 1 {
        return v[0];
    }
    let half = v.len() / 2;
    let lo = tree(&v[..half], op, s);
    let hi = tree(&v[half..], op, s);
    F::from_bits(binop(op, lo, hi, F::zero(), s))
}

/// `do_float*_logb_as_int()`.
fn logb<F: Fp>(a: F, s: &mut FloatStatus) -> u64 {
    let bits = a.bits();
    let frac = bits & F::frac_mask();
    let exp = (bits >> F::FRAC) & F::exp_mask();
    let bias = (1i64 << (F::EXPB - 1)) - 1;
    let w = 8u32 << F::ESZ;
    if exp == 0 {
        if frac != 0 {
            if !s.flush_inputs_to_zero {
                s.raise(flags::INPUT_DENORMAL_USED);
                let clz = i64::from(frac.leading_zeros()) - i64::from(64 - F::FRAC);
                return (-bias - clz) as u64;
            }
            s.raise(flags::INPUT_DENORMAL_FLUSHED);
        }
    } else if exp == F::exp_mask() {
        if frac == 0 {
            return (1u64 << (w - 1)) - 1;
        }
    } else {
        return (exp as i64 - bias) as u64;
    }
    s.raise(flags::INVALID);
    1u64 << (w - 1)
}

/// The coefficient `i` of FTMAD (`sve_ftmad_*`'s `coeff[]`).
fn tmad_coeff(esz: u32, i: usize) -> u64 {
    const H: [u16; 16] =
        [0x3c00, 0xb155, 0x2030, 0, 0, 0, 0, 0, 0x3c00, 0xb800, 0x293a, 0, 0, 0, 0, 0];
    const S: [u32; 16] = [
        0x3f80_0000,
        0xbe2a_aaab,
        0x3c08_8886,
        0xb950_08b9,
        0x3636_9d6d,
        0,
        0,
        0,
        0x3f80_0000,
        0xbf00_0000,
        0x3d2a_aaa6,
        0xbab6_0705,
        0x37cd_37cc,
        0,
        0,
        0,
    ];
    const D: [u64; 16] = [
        0x3ff0_0000_0000_0000,
        0xbfc5_5555_5555_5543,
        0x3f81_1111_1110_f30c,
        0xbf2a_01a0_19b9_2fc6,
        0x3ec7_1de3_51f3_d22b,
        0xbe5a_e5e2_b60f_7b91,
        0x3de5_d840_8868_552f,
        0,
        0x3ff0_0000_0000_0000,
        0xbfe0_0000_0000_0000,
        0x3fa5_5555_5555_5536,
        0xbf56_c16c_16c1_3a0b,
        0x3efa_01a0_19b1_e8d8,
        0xbe92_7e4f_7282_f468,
        0x3e21_ee96_d264_1b13,
        0xbda8_f763_80fb_b401,
    ];
    match esz {
        1 => u64::from(H[i]),
        2 => u64::from(S[i]),
        _ => D[i],
    }
}

/// `sve_fexpa_*`.
fn fexpa(esz: u32, n: u64) -> u64 {
    match esz {
        1 => u64::from(FEXPA_H[(n & 31) as usize]) | ((n >> 5) & 31) << 10,
        2 => u64::from(FEXPA_S[(n & 63) as usize]) | ((n >> 6) & 0xff) << 23,
        _ => FEXPA_D[(n & 63) as usize] | ((n >> 6) & 0x7ff) << 52,
    }
}

fn run<F: Fp>(env: &mut [u8], d: &Dsc, x: u64, s: &mut FloatStatus) {
    let e = F::ESZ;
    let elems = d.vl >> e;
    let shape = d.data & 15;
    let extra = d.data >> 8;
    let n = zload(env, d.n);
    let m = zload(env, d.m);
    let g = pload(env, d.g);
    let sign = F::sign_bit();
    let neg = |b: bool| if b { sign } else { 0 };
    if shape == sh::CMP {
        let mut p: P = [0; PREG_SIZE / 8];
        for i in 0..elems {
            if !act(&g, e, i) {
                continue;
            }
            let a: F = el(&n, i);
            let b = if extra & 1 != 0 { F::zero() } else { el(&m, i) };
            let (a, b) = if extra & 2 != 0 { (b, a) } else { (a, b) };
            let r = match d.op {
                f::CNE => a.compare_quiet(b, s) != FloatRelation::Equal,
                f::CUO => a.compare_quiet(b, s) == FloatRelation::Unordered,
                o => fcmp(o, a, b, s),
            };
            if r {
                let bit = i << e;
                p[bit / 64] |= 1 << (bit % 64);
            }
        }
        pstore(env, d.d, &p, d.vl);
        return;
    }
    if shape == sh::RED {
        let r = if d.op == f::ADDA {
            let mut r: F = el(&n, 0);
            for i in 0..elems {
                if act(&g, e, i) {
                    r = r.add(el(&m, i), s);
                }
            }
            r
        } else {
            let ident = match d.op {
                op::ADD => F::zero(),
                op::MAXNM | op::MINNM => F::default_nan(s),
                op::MIN => F::infinity(),
                _ => F::infinity().chs(),
            };
            let total = d.vl.next_power_of_two() >> e;
            let mut v = [ident; ZREG_SIZE / 2];
            for i in 0..elems {
                if act(&g, e, i) {
                    v[i] = el(&n, i);
                }
            }
            tree(&v[..total], d.op, s)
        };
        let mut out = [0u8; ZREG_SIZE];
        set(&mut out, e, 0, r.bits());
        zstore(env, d.d, &out, d.vl);
        return;
    }
    let za = zload(env, d.a);
    let mut out = zload(env, d.d);
    match shape {
        sh::ZPZZ | sh::ZPZS | sh::ZZZ => {
            for i in 0..elems {
                if shape != sh::ZZZ && !act(&g, e, i) {
                    continue;
                }
                let b = if shape == sh::ZPZS { x } else { get(&m, e, i) };
                set(&mut out, e, i, op2::<F>(d.op, el(&n, i), b, s));
            }
        }
        sh::PAIR => {
            for i in (0..elems).step_by(2) {
                let (n0, n1, m0, m1): (F, F, F, F) =
                    (el(&n, i), el(&n, i + 1), el(&m, i), el(&m, i + 1));
                if act(&g, e, i) {
                    set(&mut out, e, i, binop(d.op, n0, n1, F::zero(), s));
                }
                if act(&g, e, i + 1) {
                    set(&mut out, e, i + 1, binop(d.op, m0, m1, F::zero(), s));
                }
            }
        }
        sh::MLA => {
            for i in 0..elems {
                if act(&g, e, i) {
                    let a = F::from_bits(get(&n, e, i) ^ neg(extra & 1 != 0));
                    let c = F::from_bits(get(&za, e, i) ^ neg(extra & 2 != 0));
                    set(&mut out, e, i, a.muladd(el(&m, i), c, 0, s).bits());
                }
            }
        }
        sh::UN => {
            for i in 0..elems {
                if !act(&g, e, i) {
                    continue;
                }
                let a: F = el(&n, i);
                let r = match d.op {
                    op::SQRT => a.sqrt(s).bits(),
                    op::RINT => rint(a, false, s).bits(),
                    op::RINTX => rint(a, true, s).bits(),
                    op::RECPX => recpx(a, s).bits(),
                    _ => logb(a, s),
                };
                set(&mut out, e, i, r);
            }
        }
        sh::UNU => {
            for i in 0..elems {
                let a: F = el(&n, i);
                let r = match d.op {
                    op::RECPE => recpe(a, s).bits(),
                    op::RSQRTE => rsqrte(a, s).bits(),
                    _ => fexpa(e, a.bits()),
                };
                set(&mut out, e, i, r);
            }
        }
        sh::CPLX => {
            for i in (0..elems).step_by(2) {
                let j = i + 1;
                let (nr, ni, mr, mi): (F, F, F, F) = (el(&n, i), el(&n, j), el(&m, i), el(&m, j));
                let (ai, aj): (F, F) = (el(&za, i), el(&za, j));
                if d.op == f::CADD {
                    let (e1, e3) = if extra & 1 != 0 { (mi, mr.chs()) } else { (mi.chs(), mr) };
                    if act(&g, e, i) {
                        set(&mut out, e, i, nr.add(e1, s).bits());
                    }
                    if act(&g, e, j) {
                        set(&mut out, e, j, ni.add(e3, s).bits());
                    }
                } else {
                    let flip = extra & 1 != 0;
                    let negi = extra & 2 != 0;
                    let e2 = if flip { ni } else { nr };
                    let e1 = F::from_bits((if flip { mi } else { mr }).bits() ^ neg(flip ^ negi));
                    let e3 = F::from_bits((if flip { mr } else { mi }).bits() ^ neg(negi));
                    if act(&g, e, i) {
                        set(&mut out, e, i, e2.muladd(e1, ai, 0, s).bits());
                    }
                    if act(&g, e, j) {
                        set(&mut out, e, j, e2.muladd(e3, aj, 0, s).bits());
                    }
                }
            }
        }
        sh::IDX => {
            let seg = 16 >> e;
            for base in (0..elems).step_by(seg) {
                if d.op == f::CMLA {
                    let idx = (extra >> 2) as usize;
                    let flip = extra & 1 != 0;
                    let negi = extra & 2 != 0;
                    let (mr, mi) = (get(&m, e, base + 2 * idx), get(&m, e, base + 2 * idx + 1));
                    let e1 = F::from_bits(if flip { mi } else { mr } ^ neg(flip ^ negi));
                    let e3 = F::from_bits(if flip { mr } else { mi } ^ neg(negi));
                    for j in (base..base + seg).step_by(2) {
                        let e2: F = el(&n, j + usize::from(flip));
                        let r0 = e2.muladd(e1, el(&za, j), 0, s);
                        let r1 = e2.muladd(e3, el(&za, j + 1), 0, s);
                        set(&mut out, e, j, r0.bits());
                        set(&mut out, e, j + 1, r1.bits());
                    }
                } else {
                    let mm: F = el(&m, base + extra as usize);
                    for j in base..base + seg {
                        let r = match d.op {
                            op::MUL => el::<F>(&n, j).mul(mm, s),
                            o => {
                                let a = F::from_bits(get(&n, e, j) ^ neg(o == op::MLS));
                                a.muladd(mm, el(&za, j), 0, s)
                            }
                        };
                        set(&mut out, e, j, r.bits());
                    }
                }
            }
        }
        _ => {
            if d.op == f::TMAD {
                for i in 0..elems {
                    let mut mm: F = el(&m, i);
                    let mut xx = extra as usize;
                    if mm.is_neg() {
                        mm = mm.abs();
                        xx += 8;
                    }
                    let c = F::from_bits(tmad_coeff(e, xx));
                    set(&mut out, e, i, el::<F>(&n, i).muladd(mm, c, 0, s).bits());
                }
            } else {
                // FMMLA: each group of four elements is a 2x2 matrix, row major. A vector
                // length that is not a multiple of the group leaves the rest of Zd alone,
                // as QEMU's helper does.
                for blk in 0..elems / 4 {
                    let b = 4 * blk;
                    for k in 0..4 {
                        let (r, c) = (k / 2, k % 2);
                        let p0 = el::<F>(&n, b + 2 * r).mul(el(&m, b + 2 * c), s);
                        let p1 = el::<F>(&n, b + 2 * r + 1).mul(el(&m, b + 2 * c + 1), s);
                        let sum = p0.add(p1, s);
                        set(&mut out, e, b + k, el::<F>(&za, b + k).add(sum, s).bits());
                    }
                }
            }
        }
    }
    zstore(env, d.d, &out, d.vl);
}

/// `float16_to_float32_by_bits()`: the exact widening FMLAL uses, which raises nothing and
/// flushes denormal inputs to zero when `fz16` is set.
fn f16_to_f32_by_bits(x: u16, fz16: bool) -> Float32 {
    let x = u32::from(x);
    let sign = x >> 15;
    let mut exp = (x >> 10) & 0x1f;
    let mut frac = x & 0x3ff;
    if exp == 0x1f {
        exp = 0xff;
    } else if exp == 0 {
        if frac != 0 {
            if fz16 {
                frac = 0;
            } else {
                // Normalize the denormal.
                let shift = frac.leading_zeros() - 21;
                frac = (frac << shift) & 0x3ff;
                exp = 127 - 15 - shift + 1;
            }
        }
    } else {
        exp += 127 - 15;
    }
    Float32(sign << 31 | exp << 23 | frac << 13)
}

/// `bfdotadd()`: `sum + (e1.lo * e2.lo + e1.hi * e2.hi)` on the BFloat16 pairs of `e1` and
/// `e2`, in the status `is_ebf()` makes with FPCR.EBF clear.
fn bfdotadd(sum: u32, e1: u32, e2: u32, s: &mut FloatStatus) -> u32 {
    let t1 = Float32(e1 << 16).mul(Float32(e2 << 16), s);
    let t2 = Float32(e1 & 0xffff_0000).mul(Float32(e2 & 0xffff_0000), s);
    Float32(sum).add(t1.add(t2, s), s).0
}

/// The widening operations of FEAT_BF16 and SVE2's FMLAL group, in `FPST_A64`.
fn widen(env: &mut [u8], d: &Dsc, s: &mut FloatStatus) {
    let extra = d.data >> 8;
    let n = zload(env, d.n);
    let m = zload(env, d.m);
    let a = zload(env, d.a);
    let mut out = zload(env, d.d);
    let words = d.vl / 4;
    let h = |z: &Z, i: usize| get(z, 1, i) as u16;
    let wd = |z: &Z, i: usize| get(z, 2, i) as u32;
    match d.op {
        f::BFCVT => {
            let g = pload(env, d.g);
            for i in 0..words {
                if act(&g, 2, i) {
                    let r = u64::from(Float32(wd(&n, i)).to_bfloat16(s).0);
                    if extra & 1 != 0 {
                        set(&mut out, 1, 2 * i + 1, r);
                    } else {
                        set(&mut out, 2, i, r);
                    }
                }
            }
        }
        f::FMLAL => {
            let fz16 = load_status(env, 1).flush_inputs_to_zero;
            let negx = if extra & 1 != 0 { 0x8000 } else { 0 };
            let sel = ((extra >> 1) & 1) as usize;
            let idx = ((extra >> 3) & 7) as usize;
            for i in 0..words {
                let nn = f16_to_f32_by_bits(h(&n, 2 * i + sel) ^ negx, fz16);
                let mi = if extra & 4 != 0 { (i / 4) * 8 + idx } else { 2 * i + sel };
                let mm = f16_to_f32_by_bits(h(&m, mi), fz16);
                set(&mut out, 2, i, u64::from(nn.muladd(mm, Float32(wd(&a, i)), 0, s).0));
            }
        }
        f::BFMLAL => {
            let sel = (extra & 1) as usize;
            let idx = ((extra >> 2) & 7) as usize;
            for i in 0..words {
                let nn = Float32(u32::from(h(&n, 2 * i + sel)) << 16);
                let mi = if extra & 2 != 0 { 2 * (i / 4) * 4 + idx } else { 2 * i + sel };
                let mm = Float32(u32::from(h(&m, mi)) << 16);
                set(&mut out, 2, i, u64::from(nn.muladd(mm, Float32(wd(&a, i)), 0, s).0));
            }
        }
        _ => {
            // BFDOT and BFMMLA ignore the cumulative flags and use round to odd with
            // denormals flushed, as `is_ebf()` sets them up with FPCR.EBF clear.
            let mut st = *s;
            st.default_nan_mode = true;
            st.flush_to_zero = true;
            st.flush_inputs_to_zero = true;
            st.rounding_mode = RoundMode::ToOddInf;
            let st = &mut st;
            if d.op == f::BFDOT {
                let idx = ((extra >> 1) & 3) as usize;
                for i in 0..words {
                    let mi = if extra & 1 != 0 { (i / 4) * 4 + idx } else { i };
                    let r = bfdotadd(wd(&a, i), wd(&n, i), wd(&m, mi), st);
                    set(&mut out, 2, i, u64::from(r));
                }
            } else {
                for b in (0..words).step_by(4) {
                    let nw = |k: usize| wd(&n, b + k);
                    let mw = |k: usize| wd(&m, b + k);
                    for k in 0..4 {
                        let (r, c) = (k / 2, k % 2);
                        let t = bfdotadd(wd(&a, b + k), nw(2 * r), mw(2 * c), st);
                        let t = bfdotadd(t, nw(2 * r + 1), mw(2 * c + 1), st);
                        set(&mut out, 2, b + k, u64::from(t));
                    }
                }
            }
        }
    }
    zstore(env, d.d, &out, d.vl);
}

/// Float to integer toward zero; a NaN raises Invalid Operation and gives 0.
fn fto<F: Fp>(a: F, bits: u32, signed: bool, s: &mut FloatStatus) -> u64 {
    if a.is_any_nan() {
        s.raise(flags::INVALID);
        return 0;
    }
    a.to_int(bits, signed, RoundMode::ToZero, 0, s)
}

/// The conversions, on containers of the descriptor's element size.
fn conv(env: &mut [u8], d: &Dsc, s: &mut FloatStatus) {
    let c = d.esz;
    let extra = d.data >> 8;
    let (from, to) = (extra & 3, (extra >> 2) & 3);
    let n = zload(env, d.n);
    let g = pload(env, d.g);
    let mut out = zload(env, d.d);
    let mask = |x: u64, e: u32| if e >= 3 { x } else { x & ((1u64 << (8 << e)) - 1) };
    for i in 0..d.vl >> c {
        if !act(&g, c, i) {
            continue;
        }
        let x = get(&n, c, i);
        match d.op {
            f::CVTNT => set(&mut out, to, 2 * i + 1, cvt(x, from, to, true, s)),
            f::CVTLT => {
                let v = cvt(get(&n, from, 2 * i + 1), from, to, true, s);
                set(&mut out, c, i, v);
            }
            f::CVT => set(&mut out, c, i, cvt(mask(x, from), from, to, true, s)),
            f::TOSINT | f::TOUINT => {
                let sg = d.op == f::TOSINT;
                let bits = 8 << to;
                let v = match from {
                    1 => fto(Float16(x as u16), bits, sg, s),
                    2 => fto(Float32(x as u32), bits, sg, s),
                    _ => fto(Float64(x), bits, sg, s),
                };
                set(&mut out, c, i, v);
            }
            _ => {
                let sg = d.op == f::SCVTF;
                let sh = 64 - (8 << from);
                let x = if sg { (((x << sh) as i64) >> sh) as u64 } else { mask(x, from) };
                let v = match to {
                    1 => Float16::from_int(x, sg, 0, s).bits(),
                    2 => Float32::from_int(x, sg, 0, s).bits(),
                    _ => Float64::from_int(x, sg, 0, s).bits(),
                };
                set(&mut out, c, i, v);
            }
        }
    }
    zstore(env, d.d, &out, d.vl);
}

const FEXPA_H: [u16; 32] = [
    0x0000, 0x0016, 0x002d, 0x0045, 0x005d, 0x0075, 0x008e, 0x00a8, 0x00c2, 0x00dc, 0x00f8, 0x0114,
    0x0130, 0x014d, 0x016b, 0x0189, 0x01a8, 0x01c8, 0x01e8, 0x0209, 0x022b, 0x024e, 0x0271, 0x0295,
    0x02ba, 0x02e0, 0x0306, 0x032e, 0x0356, 0x037f, 0x03a9, 0x03d4,
];

const FEXPA_S: [u32; 64] = [
    0x0000_0000,
    0x0001_64d2,
    0x0002_cd87,
    0x0004_3a29,
    0x0005_aac3,
    0x0007_1f62,
    0x0008_980f,
    0x000a_14d5,
    0x000b_95c2,
    0x000d_1adf,
    0x000e_a43a,
    0x0010_31dc,
    0x0011_c3d3,
    0x0013_5a2b,
    0x0014_f4f0,
    0x0016_942d,
    0x0018_37f0,
    0x0019_e046,
    0x001b_8d3a,
    0x001d_3eda,
    0x001e_f532,
    0x0020_b051,
    0x0022_7043,
    0x0024_3516,
    0x0025_fed7,
    0x0027_cd94,
    0x0029_a15b,
    0x002b_7a3a,
    0x002d_583f,
    0x002f_3b79,
    0x0031_23f6,
    0x0033_11c4,
    0x0035_04f3,
    0x0036_fd92,
    0x0038_fbaf,
    0x003a_ff5b,
    0x003d_08a4,
    0x003f_179a,
    0x0041_2c4d,
    0x0043_46cd,
    0x0045_672a,
    0x0047_8d75,
    0x0049_b9be,
    0x004b_ec15,
    0x004e_248c,
    0x0050_6334,
    0x0052_a81e,
    0x0054_f35b,
    0x0057_44fd,
    0x0059_9d16,
    0x005b_fbb8,
    0x005e_60f5,
    0x0060_ccdf,
    0x0063_3f89,
    0x0065_b907,
    0x0068_396a,
    0x006a_c0c7,
    0x006d_4f30,
    0x006f_e4ba,
    0x0072_8177,
    0x0075_257d,
    0x0077_d0df,
    0x007a_83b3,
    0x007d_3e0c,
];

const FEXPA_D: [u64; 64] = [
    0x0000_0000_0000_0000,
    0x0000_2c9a_3e77_8061,
    0x0000_59b0_d315_8574,
    0x0000_8745_1875_9bc8,
    0x0000_b558_6cf9_890f,
    0x0000_e3ec_32d3_d1a2,
    0x0001_1301_d012_5b51,
    0x0001_429a_aea9_2de0,
    0x0001_72b8_3c7d_517b,
    0x0001_a35b_eb6f_cb75,
    0x0001_d487_3168_b9aa,
    0x0002_063b_8862_8cd6,
    0x0002_387a_6e75_6238,
    0x0002_6b45_65e2_7cdd,
    0x0002_9e9d_f51f_dee1,
    0x0002_d285_a6e4_030b,
    0x0003_06fe_0a31_b715,
    0x0003_3c08_b264_16ff,
    0x0003_71a7_373a_a9cb,
    0x0003_a7db_34e5_9ff7,
    0x0003_dea6_4c12_3422,
    0x0004_160a_21f7_2e2a,
    0x0004_4e08_6061_892d,
    0x0004_86a2_b5c1_3cd0,
    0x0004_bfda_d536_2a27,
    0x0004_f9b2_769d_2ca7,
    0x0005_342b_569d_4f82,
    0x0005_6f47_36b5_27da,
    0x0005_ab07_dd48_5429,
    0x0005_e76f_15ad_2148,
    0x0006_247e_b03a_5585,
    0x0006_6238_8255_2225,
    0x0006_a09e_667f_3bcd,
    0x0006_dfb2_3c65_1a2f,
    0x0007_1f75_e8ec_5f74,
    0x0007_5feb_5642_67c9,
    0x0007_a114_73eb_0187,
    0x0007_e2f3_36cf_4e62,
    0x0008_2589_994c_ce13,
    0x0008_68d9_9b44_92ed,
    0x0008_ace5_422a_a0db,
    0x0008_f1ae_9915_7736,
    0x0009_3737_b0cd_c5e5,
    0x0009_7d82_9fde_4e50,
    0x0009_c491_82a3_f090,
    0x000a_0c66_7b5d_e565,
    0x000a_5503_b23e_255d,
    0x000a_9e6b_5579_fdbf,
    0x000a_e89f_995a_d3ad,
    0x000b_33a2_b84f_15fb,
    0x000b_7f76_f2fb_5e47,
    0x000b_cc1e_904b_c1d2,
    0x000c_199b_dd85_529c,
    0x000c_67f1_2e57_d14b,
    0x000c_b720_dcef_9069,
    0x000d_072d_4a07_897c,
    0x000d_5818_dcfb_a487,
    0x000d_a9e6_03db_3285,
    0x000d_fc97_337b_9b5f,
    0x000e_502e_e78b_3ff6,
    0x000e_a4af_a2a4_90da,
    0x000e_fa1b_ee61_5a27,
    0x000f_5076_5b6e_4540,
    0x000f_a7c1_819e_90d8,
];
