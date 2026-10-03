// SPDX-License-Identifier: GPL-2.0-or-later

//! The SVE2 integer helpers for the widening, narrowing, long and pairwise groups, the
//! complex integer operations, the integer multiplies and dot products by element, the
//! integer matrix multiplies, the quadword permutes of FEAT_F64MM and DUPQ: the parts of
//! QEMU's `sve_helper.c` and `vec_helper.c` they come from (`sve2_*`, `gvec_*dot*`,
//! `gvec_*mmla_b`, `sve2_zip_q` and the rest).
//!
//! They are the `INT2` family of the `sve` helper ([`super::sve_helper::fam::INT2`]). The
//! descriptor names the operation ([`w`]), an element size and the registers, and its data
//! field holds the operation specific bits [`w`] describes. For the widening and narrowing
//! operations the element size is that of the wide elements; Zn, Zm and Za are read and the
//! result is computed into a copy of Zd before any of it is written, so a source that is
//! also the destination is read as QEMU's helpers read it.
//!
//! The results are those of the QEMU helpers bit for bit, with one difference:
//!
//! - ZIP1, ZIP2, TRN1 and TRN2 on quadwords at a vector length that is an odd number of
//!   quadwords write a quadword past the end of the vector in QEMU (into the part of the
//!   register beyond the current length, which it then keeps). This port writes the vector
//!   length only, so those bits stay as they were.

#![allow(clippy::needless_range_loop)]

use super::sve_helper::{
    Dsc, Z, act, get, gets, mask, pload, set, sext, sqrdmlah, ssat, usat, zload, zstore,
};
use crate::cpu::ZREG_SIZE;

/// The operations of [`super::sve_helper::fam::INT2`].
#[allow(missing_docs)]
pub(crate) mod w {
    /// The long operations, from SADDL to SQDMLSL: `d[i] = op(n[2i + sel1], m[2i + sel2])`
    /// (with Za for the accumulating ones) on the wide elements of the descriptor's size.
    /// Data bit 0 is sel1, bit 1 sel2, bit 2 asks for the indexed form, where m is element
    /// `index` (bits 3 to 5) of each 128-bit segment of Zm and sel1 picks the n element.
    pub(crate) const SADDL: u32 = 1;
    pub(crate) const UADDL: u32 = 2;
    pub(crate) const SSUBL: u32 = 3;
    pub(crate) const USUBL: u32 = 4;
    pub(crate) const SABDL: u32 = 5;
    pub(crate) const UABDL: u32 = 6;
    pub(crate) const SMULL: u32 = 7;
    pub(crate) const UMULL: u32 = 8;
    pub(crate) const SQDMULL: u32 = 9;
    pub(crate) const SABAL: u32 = 10;
    pub(crate) const UABAL: u32 = 11;
    pub(crate) const SMLAL: u32 = 12;
    pub(crate) const UMLAL: u32 = 13;
    pub(crate) const SMLSL: u32 = 14;
    pub(crate) const UMLSL: u32 = 15;
    pub(crate) const SQDMLAL: u32 = 16;
    pub(crate) const SQDMLSL: u32 = 17;
    /// The wide operations: `d[i] = n[i] op m[2i + sel]`, data bit 0 is sel.
    pub(crate) const SADDW: u32 = 20;
    pub(crate) const UADDW: u32 = 21;
    pub(crate) const SSUBW: u32 = 22;
    pub(crate) const USUBW: u32 = 23;
    /// EORBT and EORTB: data bit 0 is sel1, bit 1 sel2.
    pub(crate) const EORBT: u32 = 24;
    /// ADCLB, ADCLT, SBCLB and SBCLT on pairs of elements of the descriptor's size (32 or
    /// 64 bits). Data bit 0 is sel, bit 1 inverts n (the SBCL forms).
    pub(crate) const ADCL: u32 = 25;
    /// CADD and SQCADD: data 0 rotates by 90 degrees, 1 by 270.
    pub(crate) const CADD: u32 = 26;
    pub(crate) const SQCADD: u32 = 27;
    /// SSHLL and USHLL: data bit 0 is sel, bits 1 up the shift.
    pub(crate) const SSHLL: u32 = 28;
    pub(crate) const USHLL: u32 = 29;
    /// The saturating extract narrows: data bit 0 writes the top half.
    pub(crate) const SQXTN: u32 = 30;
    pub(crate) const UQXTN: u32 = 31;
    pub(crate) const SQXTUN: u32 = 32;
    /// The shift right narrows: data bit 0 writes the top half, bits 1 up hold the shift.
    pub(crate) const SHRN: u32 = 33;
    pub(crate) const RSHRN: u32 = 34;
    pub(crate) const SQSHRUN: u32 = 35;
    pub(crate) const SQRSHRUN: u32 = 36;
    pub(crate) const SQSHRN: u32 = 37;
    pub(crate) const SQRSHRN: u32 = 38;
    pub(crate) const UQSHRN: u32 = 39;
    pub(crate) const UQRSHRN: u32 = 40;
    /// The high half narrows: data bit 0 writes the top half.
    pub(crate) const ADDHN: u32 = 41;
    pub(crate) const RADDHN: u32 = 42;
    pub(crate) const SUBHN: u32 = 43;
    pub(crate) const RSUBHN: u32 = 44;
    /// The predicated pairwise operations.
    pub(crate) const ADDP: u32 = 45;
    pub(crate) const SMAXP: u32 = 46;
    pub(crate) const UMAXP: u32 = 47;
    pub(crate) const SMINP: u32 = 48;
    pub(crate) const UMINP: u32 = 49;
    /// CMLA, SQRDCMLAH and CDOT: data bits 0 and 1 hold the rotation, bit 2 asks for the
    /// indexed form, bits 3 up hold the index.
    pub(crate) const CMLA: u32 = 50;
    pub(crate) const SQRDCMLAH: u32 = 51;
    pub(crate) const CDOT: u32 = 52;
    /// The multiplies by element: data holds the index.
    pub(crate) const MUL_X: u32 = 53;
    pub(crate) const MLA_X: u32 = 54;
    pub(crate) const MLS_X: u32 = 55;
    pub(crate) const SQDMULH_X: u32 = 56;
    pub(crate) const SQRDMULH_X: u32 = 57;
    pub(crate) const SQRDMLAH_X: u32 = 58;
    pub(crate) const SQRDMLSH_X: u32 = 59;
    /// The four-way dot products: data bit 0 makes n signed, bit 1 m, bit 2 asks for the
    /// indexed form, bits 3 up hold the index.
    pub(crate) const DOT: u32 = 60;
    /// SMMLA, UMMLA and USMMLA: data bit 0 makes n signed, bit 1 m.
    pub(crate) const MMLA: u32 = 61;
    /// The quadword permutes: data holds the byte offset of the odd form.
    pub(crate) const ZIPQ: u32 = 62;
    pub(crate) const UZPQ: u32 = 63;
    pub(crate) const TRNQ: u32 = 64;
    /// DUPQ: data holds the element index.
    pub(crate) const DUPQ: u32 = 65;
}

/// Element `i` of `z`, sign or zero extended.
fn ext(z: &Z, esz: u32, i: usize, signed: bool) -> i128 {
    if signed { i128::from(gets(z, esz, i)) } else { i128::from(get(z, esz, i)) }
}

/// Run the `INT2` family of the `sve` helper.
pub(crate) fn int2(env: &mut [u8], d: &Dsc) {
    let n = zload(env, d.n);
    let m = zload(env, d.m);
    let a = zload(env, d.a);
    let mut out = zload(env, d.d);
    let e = d.esz;
    let elems = d.vl >> e;
    // The elements of a 128-bit segment.
    let seg = 16usize >> e;
    let data = d.data;
    match d.op {
        w::SADDL..=w::SQDMLSL => {
            let h = e - 1;
            let sel1 = (data & 1) as usize;
            let sel2 = ((data >> 1) & 1) as usize;
            let idx = ((data >> 3) & 7) as usize;
            let s = matches!(
                d.op,
                w::SADDL
                    | w::SSUBL
                    | w::SABDL
                    | w::SMULL
                    | w::SQDMULL
                    | w::SABAL
                    | w::SMLAL
                    | w::SMLSL
                    | w::SQDMLAL
                    | w::SQDMLSL
            );
            for i in 0..elems {
                let nn = ext(&n, h, 2 * i + sel1, s);
                let mm = if data & 4 != 0 {
                    ext(&m, h, (i / seg) * seg * 2 + idx, s)
                } else {
                    ext(&m, h, 2 * i + sel2, s)
                };
                set(&mut out, e, i, long(d.op, e, nn, mm, get(&a, e, i)));
            }
        }
        w::SADDW..=w::USUBW => {
            let h = e - 1;
            let sel = (data & 1) as usize;
            let s = matches!(d.op, w::SADDW | w::SSUBW);
            for i in 0..elems {
                let nn = ext(&n, e, i, s);
                let mm = ext(&m, h, 2 * i + sel, s);
                let r = if matches!(d.op, w::SADDW | w::UADDW) { nn + mm } else { nn - mm };
                set(&mut out, e, i, r as u64);
            }
        }
        w::EORBT => {
            let sel1 = (data & 1) as usize;
            let sel2 = ((data >> 1) & 1) as usize;
            for i in (0..elems).step_by(2) {
                set(&mut out, e, i + sel1, get(&n, e, i + sel1) ^ get(&m, e, i + sel2));
            }
        }
        w::ADCL => {
            let sel = (data & 1) as usize;
            let inv = if data & 2 != 0 { mask(e) } else { 0 };
            for i in (0..elems).step_by(2) {
                let f = u128::from(get(&a, e, i))
                    + u128::from(get(&n, e, i + sel) ^ inv)
                    + u128::from(get(&m, e, i + 1) & 1);
                set(&mut out, e, i, f as u64);
                set(&mut out, e, i + 1, (f >> (8 << e)) as u64);
            }
        }
        w::CADD | w::SQCADD => {
            for i in (0..elems).step_by(2) {
                let (nr, ni) = (ext(&n, e, i, true), ext(&n, e, i + 1, true));
                let (mr, mi) = (ext(&m, e, i, true), ext(&m, e, i + 1, true));
                let (r, im) = if data == 0 { (nr - mi, ni + mr) } else { (nr + mi, ni - mr) };
                if d.op == w::SQCADD {
                    set(&mut out, e, i, ssat(r, e));
                    set(&mut out, e, i + 1, ssat(im, e));
                } else {
                    set(&mut out, e, i, r as u64);
                    set(&mut out, e, i + 1, im as u64);
                }
            }
        }
        w::SSHLL | w::USHLL => {
            let sel = (data & 1) as usize;
            let sh = data >> 1;
            for i in 0..elems {
                let x = ext(&n, e - 1, 2 * i + sel, d.op == w::SSHLL) as u64;
                set(&mut out, e, i, x << sh);
            }
        }
        w::SQXTN..=w::RSUBHN => {
            let h = e - 1;
            let top = data & 1 != 0;
            let sh = data >> 1;
            for i in 0..elems {
                let r = narrow(d.op, e, sh, get(&n, e, i), get(&m, e, i));
                if top {
                    set(&mut out, h, 2 * i + 1, r);
                } else {
                    set(&mut out, e, i, r);
                }
            }
        }
        w::ADDP..=w::UMINP => {
            let g = pload(env, d.g);
            for i in (0..elems).step_by(2) {
                if act(&g, e, i) {
                    set(&mut out, e, i, pair(d.op, e, get(&n, e, i), get(&n, e, i + 1)));
                }
                if act(&g, e, i + 1) {
                    set(&mut out, e, i + 1, pair(d.op, e, get(&m, e, i), get(&m, e, i + 1)));
                }
            }
        }
        w::CMLA | w::SQRDCMLAH => {
            let rot = data & 3;
            let idx = (data >> 3) as usize;
            let sel_a = (rot & 1) as usize;
            let sel_b = sel_a ^ 1;
            let sub_r = rot == 1 || rot == 2;
            let sub_i = rot >= 2;
            for i in (0..elems).step_by(2) {
                let (ma, mb) = if data & 4 != 0 {
                    let s0 = (i / seg) * seg + idx * 2;
                    (s0 + sel_a, s0 + sel_b)
                } else {
                    (i + sel_a, i + sel_b)
                };
                let nn = gets(&n, e, i + sel_a);
                let r = cmla(d.op, e, nn, gets(&m, e, ma), gets(&a, e, i), sub_r);
                let im = cmla(d.op, e, nn, gets(&m, e, mb), gets(&a, e, i + 1), sub_i);
                set(&mut out, e, i, r);
                set(&mut out, e, i + 1, im);
            }
        }
        w::CDOT => {
            let p = e - 2;
            let rot = data & 3;
            let idx = (data >> 3) as usize;
            let sel_a = (rot & 1) as usize;
            let sel_b = sel_a ^ 1;
            let sub_i: i128 = if rot == 0 || rot == 3 { -1 } else { 1 };
            for i in 0..elems {
                let mi = if data & 4 != 0 { (i / seg) * seg + idx } else { i };
                let mut acc = i128::from(gets(&a, e, i));
                for k in 0..2 {
                    let r = ext(&n, p, 4 * i + 2 * k, true);
                    let im = ext(&n, p, 4 * i + 2 * k + 1, true);
                    let ma = ext(&m, p, 4 * mi + 2 * k + sel_a, true);
                    let mb = ext(&m, p, 4 * mi + 2 * k + sel_b, true);
                    acc += r * ma + im * mb * sub_i;
                }
                set(&mut out, e, i, acc as u64);
            }
        }
        w::MUL_X..=w::SQRDMLSH_X => {
            let idx = data as usize;
            for i in 0..elems {
                let mi = (i / seg) * seg + idx;
                let r = by_elem(d.op, e, get(&n, e, i), get(&m, e, mi), get(&a, e, i));
                set(&mut out, e, i, r);
            }
        }
        w::DOT => {
            let p = e - 2;
            let ns = data & 1 != 0;
            let ms = data & 2 != 0;
            let idx = (data >> 3) as usize;
            for i in 0..elems {
                let mi = if data & 4 != 0 { (i / seg) * seg + idx } else { i };
                let mut acc = i128::from(get(&a, e, i));
                for k in 0..4 {
                    acc += ext(&n, p, 4 * i + k, ns) * ext(&m, p, 4 * mi + k, ms);
                }
                set(&mut out, e, i, acc as u64);
            }
        }
        w::MMLA => {
            let ns = data & 1 != 0;
            let ms = data & 2 != 0;
            for s in (0..d.vl).step_by(16) {
                for (k, (no, mo)) in [(0, 0), (0, 8), (8, 0), (8, 8)].into_iter().enumerate() {
                    let mut acc = i128::from(get(&a, 2, s / 4 + k));
                    for j in 0..8 {
                        acc += ext(&n, 0, s + no + j, ns) * ext(&m, 0, s + mo + j, ms);
                    }
                    set(&mut out, 2, s / 4 + k, acc as u64);
                }
            }
        }
        w::ZIPQ | w::UZPQ | w::TRNQ => out = permute_q(d.op, d.vl, data as usize, &n, &m),
        w::DUPQ => {
            let idx = data as usize;
            for s in (0..elems).step_by(seg) {
                let v = get(&n, e, s + idx);
                for j in s..s + seg {
                    set(&mut out, e, j, v);
                }
            }
        }
        _ => unreachable!("bad sve2 integer operation {}", d.op),
    }
    zstore(env, d.d, &out, d.vl);
}

/// The long operations on the extended narrow elements; `aa` is the wide accumulator.
fn long(op: u32, e: u32, nn: i128, mm: i128, aa: u64) -> u64 {
    let sa = i128::from(sext(aa, e));
    let r = match op {
        w::SADDL | w::UADDL => nn + mm,
        w::SSUBL | w::USUBL => nn - mm,
        w::SABDL | w::UABDL => (nn - mm).abs(),
        w::SMULL | w::UMULL => nn * mm,
        w::SQDMULL => return ssat(2 * nn * mm, e),
        w::SABAL | w::UABAL => sa + (nn - mm).abs(),
        w::SMLAL | w::UMLAL => sa + nn * mm,
        w::SMLSL | w::UMLSL => sa - nn * mm,
        w::SQDMLAL => return ssat(sa + i128::from(sext(ssat(2 * nn * mm, e), e)), e),
        _ => return ssat(sa - i128::from(sext(ssat(2 * nn * mm, e), e)), e),
    };
    r as u64
}

/// `(x >> sh) + ((x >> (sh - 1)) & 1)`, the rounding shift right of `do_[su]rshr()`.
fn rshr(x: i128, sh: u32) -> i128 {
    (x >> sh) + ((x >> (sh - 1)) & 1)
}

/// The narrowing operations on wide elements `x` and `y` of size `e`, returning the narrow
/// result zero extended.
fn narrow(op: u32, e: u32, sh: u32, x: u64, y: u64) -> u64 {
    let h = e - 1;
    let ux = i128::from(x);
    let sx = i128::from(sext(x, e));
    let hb = 8 << h;
    let rnd = 1u64 << (hb - 1);
    let hi = |v: u64| ((v & mask(e)) >> hb) & mask(h);
    match op {
        w::SQXTN => ssat(sx, h),
        w::UQXTN => usat(ux, h),
        w::SQXTUN => usat(sx, h),
        w::SHRN => (ux >> sh) as u64 & mask(h),
        w::RSHRN => rshr(ux, sh) as u64 & mask(h),
        w::SQSHRUN => usat(sx >> sh, h),
        w::SQRSHRUN => usat(rshr(sx, sh), h),
        w::SQSHRN => ssat(sx >> sh, h),
        w::SQRSHRN => ssat(rshr(sx, sh), h),
        w::UQSHRN => usat(ux >> sh, h),
        w::UQRSHRN => usat(rshr(ux, sh), h),
        w::ADDHN => hi(x.wrapping_add(y)),
        w::RADDHN => hi(x.wrapping_add(y).wrapping_add(rnd)),
        w::SUBHN => hi(x.wrapping_sub(y)),
        _ => hi(x.wrapping_sub(y).wrapping_add(rnd)),
    }
}

/// The pairwise operations.
fn pair(op: u32, e: u32, x: u64, y: u64) -> u64 {
    let (sx, sy) = (sext(x, e), sext(y, e));
    match op {
        w::ADDP => x.wrapping_add(y),
        w::SMAXP => sx.max(sy) as u64,
        w::UMAXP => x.max(y),
        w::SMINP => sx.min(sy) as u64,
        _ => x.min(y),
    }
}

/// `DO_CMLA` and `DO_SQRDMLAH_*`: `a + n * m` or `a - n * m`, wrapping or saturating.
fn cmla(op: u32, e: u32, n: i64, m: i64, a: i64, sub: bool) -> u64 {
    if op == w::SQRDCMLAH {
        return sqrdmlah(n, m, a, sub, true, e);
    }
    let p = i128::from(n) * i128::from(m);
    (i128::from(a) + if sub { -p } else { p }) as u64
}

/// The multiplies by element `m`.
fn by_elem(op: u32, e: u32, n: u64, m: u64, a: u64) -> u64 {
    let (sn, sm, sa) = (sext(n, e), sext(m, e), sext(a, e));
    match op {
        w::MUL_X => n.wrapping_mul(m),
        w::MLA_X => a.wrapping_add(n.wrapping_mul(m)),
        w::MLS_X => a.wrapping_sub(n.wrapping_mul(m)),
        w::SQDMULH_X => sqrdmlah(sn, sm, 0, false, false, e),
        w::SQRDMULH_X => sqrdmlah(sn, sm, 0, false, true, e),
        w::SQRDMLAH_X => sqrdmlah(sn, sm, sa, false, true, e),
        _ => sqrdmlah(sn, sm, sa, true, true, e),
    }
}

/// `sve2_zip_q`, `sve2_uzp_q` and `sve2_trn_q` with `odd` the byte offset of the second
/// forms, into a whole register image.
fn permute_q(op: u32, vl: usize, odd: usize, n: &Z, m: &Z) -> Z {
    let mut d = [0u8; ZREG_SIZE];
    let q = |d: &mut Z, at: usize, src: &Z, from: usize| {
        // A quadword past the end of the register reads as zero: QEMU reads the bytes that
        // follow in its register file, but no vector length reaches there.
        for k in 0..16 {
            if at + k < ZREG_SIZE {
                d[at + k] = src.get(from + k).copied().unwrap_or(0);
            }
        }
    };
    match op {
        w::ZIPQ => {
            for i in (0..vl / 2).step_by(16) {
                q(&mut d, 2 * i, n, odd + i);
                q(&mut d, 2 * i + 16, m, odd + i);
            }
            if vl & 16 != 0 {
                d[vl - 16..vl].fill(0);
            }
        }
        w::UZPQ => {
            let mut i = 0;
            let mut p = odd;
            loop {
                q(&mut d, i, n, p);
                i += 16;
                p += 32;
                if p >= vl {
                    break;
                }
            }
            p -= vl;
            loop {
                q(&mut d, i, m, p);
                i += 16;
                p += 32;
                if p >= vl {
                    break;
                }
            }
        }
        _ => {
            for i in (0..vl).step_by(32) {
                q(&mut d, i, n, i + odd);
                q(&mut d, i + 16, m, i + odd);
            }
            if vl & 16 != 0 {
                d[vl - 16..vl].fill(0);
            }
        }
    }
    d
}
