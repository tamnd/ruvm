// SPDX-License-Identifier: GPL-2.0-or-later

//! The AES-NI, SHA and SSE4.2 string compare operations of `ops_sse.h`, on one 128-bit lane,
//! with the AES round functions of `crypto/aes.c` that they call.

use super::super::env::{CC_C, CC_O, CC_S, CC_Z};

type L = [u8; 16];

/// Multiply by x in GF(2^8) with the AES polynomial.
const fn xtime(a: u8) -> u8 {
    (a << 1) ^ if a & 0x80 != 0 { 0x1b } else { 0 }
}

const fn gmul(a: u8, b: u8) -> u8 {
    let (mut a, mut b, mut r) = (a, b, 0u8);
    while b != 0 {
        if b & 1 != 0 {
            r ^= a;
        }
        a = xtime(a);
        b >>= 1;
    }
    r
}

/// `AES_sbox`: the multiplicative inverse followed by the affine map.
const SBOX: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        // The inverse is a^254, and 0 for 0.
        let a = i as u8;
        let mut inv = 1u8;
        let mut k = 0;
        while k < 254 {
            inv = gmul(inv, a);
            k += 1;
        }
        if a == 0 {
            inv = 0;
        }
        let x = inv;
        t[i] = x ^ x.rotate_left(1) ^ x.rotate_left(2) ^ x.rotate_left(3) ^ x.rotate_left(4) ^ 0x63;
        i += 1;
    }
    t
};

/// `AES_isbox`.
const ISBOX: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        t[SBOX[i] as usize] = i as u8;
        i += 1;
    }
    t
};

/// `AES_shifts` and `AES_ishifts`: the source byte of each byte after ShiftRows.
const SHIFTS: [usize; 16] = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11];
const ISHIFTS: [usize; 16] = [0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3];

fn mix_columns(st: &L, inverse: bool) -> L {
    let k: [u8; 4] = if inverse { [14, 11, 13, 9] } else { [2, 3, 1, 1] };
    let mut out = [0u8; 16];
    for c in 0..4 {
        let col = &st[4 * c..4 * c + 4];
        for r in 0..4 {
            let mut x = 0;
            for (j, &kj) in k.iter().enumerate() {
                x ^= gmul(col[(r + j) % 4], kj);
            }
            out[4 * c + r] = x;
        }
    }
    out
}

/// The AES rounds: 0 is AESENC, 1 AESENCLAST, 2 AESDEC and 3 AESDECLAST.
pub(super) fn aes_round(op: u32, st: &L, rk: &L) -> L {
    let dec = op & 2 != 0;
    let last = op & 1 != 0;
    // SubBytes and ShiftRows, or their inverses, in one step.
    let mut t: L = core::array::from_fn(|i| {
        if dec { ISBOX[st[ISHIFTS[i]] as usize] } else { SBOX[st[SHIFTS[i]] as usize] }
    });
    if !last {
        t = mix_columns(&t, dec);
    }
    core::array::from_fn(|i| t[i] ^ rk[i])
}

/// AESIMC: `aesdec_IMC()`.
pub(super) fn aes_imc(st: &L) -> L {
    mix_columns(st, true)
}

/// `helper_aeskeygenassist`.
pub(super) fn aes_keygen(s: &L, rcon: u32) -> L {
    let mut d = [0u8; 16];
    for i in 0..4 {
        d[i] = SBOX[s[i + 4] as usize];
        d[i + 8] = SBOX[s[i + 12] as usize];
    }
    let w0 = u32::from_le_bytes([d[0], d[1], d[2], d[3]]);
    let w2 = u32::from_le_bytes([d[8], d[9], d[10], d[11]]);
    d[4..8].copy_from_slice(&(w0.rotate_right(8) ^ rcon).to_le_bytes());
    d[12..16].copy_from_slice(&(w2.rotate_right(8) ^ rcon).to_le_bytes());
    d
}

fn w(v: &L) -> [u32; 4] {
    core::array::from_fn(|i| {
        u32::from_le_bytes([v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]])
    })
}

fn unw(x: [u32; 4]) -> L {
    let mut d = [0u8; 16];
    for (i, v) in x.iter().enumerate() {
        d[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
    }
    d
}

/// The SHA operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ShaOp {
    /// SHA1RNDS4 with the function and constant of the immediate.
    Sha1Rnds4(u32),
    Sha1Nexte,
    Sha1Msg1,
    Sha1Msg2,
    /// SHA256RNDS2 with the two message words of XMM0.
    Sha256Rnds2(u32, u32),
    Sha256Msg1,
    Sha256Msg2,
}

const fn s256_rnds0(w: u32) -> u32 {
    w.rotate_right(2) ^ w.rotate_right(13) ^ w.rotate_right(22)
}

const fn s256_rnds1(w: u32) -> u32 {
    w.rotate_right(6) ^ w.rotate_right(11) ^ w.rotate_right(25)
}

const fn s256_msgs0(w: u32) -> u32 {
    w.rotate_right(7) ^ w.rotate_right(18) ^ (w >> 3)
}

const fn s256_msgs1(w: u32) -> u32 {
    w.rotate_right(17) ^ w.rotate_right(19) ^ (w >> 10)
}

const fn ch(e: u32, f: u32, g: u32) -> u32 {
    (e & f) ^ (!e & g)
}

const fn maj(a: u32, b: u32, c: u32) -> u32 {
    (a & b) ^ (a & c) ^ (b & c)
}

/// The SHA helpers of `ops_sse.h`, on `a` (the first source) and `b` (the second).
pub(super) fn sha(op: ShaOp, a: &L, b: &L) -> L {
    let (a, b) = (w(a), w(b));
    let mut d = [0u32; 4];
    match op {
        ShaOp::Sha1Rnds4(f) => {
            let (fk, k): (fn(u32, u32, u32) -> u32, u32) = match f & 3 {
                0 => (ch, 0x5a82_7999),
                1 => (|b, c, d| b ^ c ^ d, 0x6ed9_eba1),
                2 => (maj, 0x8f1b_bcdc),
                _ => (|b, c, d| b ^ c ^ d, 0xca62_c1d6),
            };
            let (mut aa, mut bb, mut cc, mut dd, mut ee) = (a[3], a[2], a[1], a[0], 0u32);
            for i in 0..4 {
                let t = fk(bb, cc, dd)
                    .wrapping_add(aa.rotate_left(5))
                    .wrapping_add(b[3 - i])
                    .wrapping_add(ee)
                    .wrapping_add(k);
                ee = dd;
                dd = cc;
                cc = bb.rotate_left(30);
                bb = aa;
                aa = t;
            }
            d = [dd, cc, bb, aa];
        }
        ShaOp::Sha1Nexte => {
            d = b;
            d[3] = b[3].wrapping_add(a[3].rotate_left(30));
        }
        ShaOp::Sha1Msg1 => {
            d = [a[0] ^ b[2], a[1] ^ b[3], a[2] ^ a[0], a[3] ^ a[1]];
        }
        ShaOp::Sha1Msg2 => {
            d[3] = (a[3] ^ b[2]).rotate_left(1);
            d[2] = (a[2] ^ b[1]).rotate_left(1);
            d[1] = (a[1] ^ b[0]).rotate_left(1);
            d[0] = (a[0] ^ d[3]).rotate_left(1);
        }
        ShaOp::Sha256Rnds2(wk0, wk1) => {
            let (mut sa, mut sb, mut sc, mut sd) = (b[3], b[2], a[3], a[2]);
            let (mut se, mut sf, mut sg, mut sh) = (b[1], b[0], a[1], a[0]);
            for (i, wk) in [wk0, wk1].into_iter().enumerate() {
                let t =
                    ch(se, sf, sg).wrapping_add(s256_rnds1(se)).wrapping_add(wk).wrapping_add(sh);
                let aa = t.wrapping_add(maj(sa, sb, sc)).wrapping_add(s256_rnds0(sa));
                let ee = t.wrapping_add(sd);
                // The even round gives B and F of the result, the odd round A and E.
                d[2 + i] = aa;
                d[i] = ee;
                (sd, sc, sb, sa) = (sc, sb, sa, aa);
                (sh, sg, sf, se) = (sg, sf, se, ee);
            }
        }
        ShaOp::Sha256Msg1 => {
            d = [
                a[0].wrapping_add(s256_msgs0(a[1])),
                a[1].wrapping_add(s256_msgs0(a[2])),
                a[2].wrapping_add(s256_msgs0(a[3])),
                a[3].wrapping_add(s256_msgs0(b[0])),
            ];
        }
        ShaOp::Sha256Msg2 => {
            d[0] = a[0].wrapping_add(s256_msgs1(b[2]));
            d[1] = a[1].wrapping_add(s256_msgs1(b[3]));
            d[2] = a[2].wrapping_add(s256_msgs1(d[0]));
            d[3] = a[3].wrapping_add(s256_msgs1(d[1]));
        }
    }
    unw(d)
}

/// `pcmp_elen()`: the explicit length in `val` (RAX or RDX), as 64 bits with REX.W and as
/// the low 32 bits sign extended without.
pub(super) fn pcmp_elen(val: u64, ctrl: u32, rex_w: bool) -> i32 {
    let v = if rex_w { val as i64 } else { i64::from(val as i32) };
    let limit = if ctrl & 1 != 0 { 8 } else { 16 };
    if v > limit || v < -limit { limit as i32 } else { v.unsigned_abs() as i32 }
}

/// `pcmp_ilen()`: the implicit length, up to the first zero element.
pub(super) fn pcmp_ilen(r: &L, ctrl: u32) -> i32 {
    let words = ctrl & 1 != 0;
    let n = if words { 8 } else { 16 };
    (0..n).find(|&i| pcmp_val(r, ctrl & 1, i) == 0).unwrap_or(n) as i32
}

/// `pcmp_val()`: element `i` with the format of bits 1:0 of `ctrl`.
fn pcmp_val(r: &L, ctrl: u32, i: usize) -> i32 {
    match ctrl & 3 {
        0 => i32::from(r[i]),
        1 => i32::from(u16::from_le_bytes([r[2 * i], r[2 * i + 1]])),
        2 => i32::from(r[i] as i8),
        _ => i32::from(i16::from_le_bytes([r[2 * i], r[2 * i + 1]])),
    }
}

/// `pcmpxstrx()`: the result bits and the flags, with `valids` the length of the second
/// operand `s` and `validd` the length of the first operand `d`.
pub(super) fn pcmpxstrx(d: &L, s: &L, ctrl: u32, valids: i32, validd: i32) -> (u32, u32) {
    let upper: i32 = if ctrl & 1 != 0 { 7 } else { 15 };
    let valids = valids - 1;
    let validd = validd - 1;
    let mut flags = if valids < upper { CC_Z } else { 0 } | if validd < upper { CC_S } else { 0 };
    let val = |r: &L, i: i32| pcmp_val(r, ctrl, i as usize);
    let mut res: u32 = 0;
    match (ctrl >> 2) & 3 {
        0 => {
            for j in (0..=valids).rev() {
                res <<= 1;
                let v = val(s, j);
                for i in (0..=validd).rev() {
                    res |= u32::from(v == val(d, i));
                }
            }
        }
        1 => {
            for j in (0..=valids).rev() {
                res <<= 1;
                let v = val(s, j);
                // The ranges are pairs of elements; `i` is always odd, so `i - 1` is valid.
                let mut i = (validd - 1) | 1;
                while i >= 0 {
                    res |= u32::from(val(d, i) >= v && val(d, i - 1) <= v);
                    i -= 2;
                }
            }
        }
        2 => {
            let (mx, mn) = (valids.max(validd), valids.min(validd));
            res = (1u32 << (upper - mx)).wrapping_sub(1);
            res <<= mx - mn;
            for i in (0..=mn).rev() {
                res <<= 1;
                let v = val(s, i);
                res |= u32::from(v == val(d, i));
            }
        }
        _ => {
            if validd == -1 {
                res = (2u32 << upper) - 1;
            } else {
                let start = if valids == upper { valids } else { valids - validd };
                for j in (0..=start).rev() {
                    res <<= 1;
                    let mut v = true;
                    for i in (0..=(valids - j).min(validd)).rev() {
                        v &= val(s, i + j) == val(d, i);
                    }
                    res |= u32::from(v);
                }
            }
        }
    }
    match (ctrl >> 4) & 3 {
        1 => res ^= (2u32 << upper) - 1,
        3 => res ^= (1u32 << (valids + 1)) - 1,
        _ => {}
    }
    if res != 0 {
        flags |= CC_C;
    }
    if res & 1 != 0 {
        flags |= CC_O;
    }
    (res, flags)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> L {
        let v: Vec<u8> =
            (0..16).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect();
        v.try_into().unwrap()
    }

    #[test]
    fn sbox() {
        assert_eq!(SBOX[0], 0x63);
        assert_eq!(SBOX[0x53], 0xed);
        assert_eq!(ISBOX[0x63], 0);
    }

    #[test]
    fn aes128_fips197() {
        // FIPS-197 appendix C.1: AES-128 with the key schedule from AESKEYGENASSIST.
        let key = hex("000102030405060708090a0b0c0d0e0f");
        let pt = hex("00112233445566778899aabbccddeeff");
        let rcon = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];
        let mut keys = vec![key];
        for &rc in &rcon {
            let prev = *keys.last().unwrap();
            let t = aes_keygen(&prev, rc);
            let mut kw = w(&prev);
            let x = w(&t)[3];
            kw[0] ^= x;
            kw[1] ^= kw[0];
            kw[2] ^= kw[1];
            kw[3] ^= kw[2];
            keys.push(unw(kw));
        }
        let mut st: L = core::array::from_fn(|i| pt[i] ^ keys[0][i]);
        for k in &keys[1..10] {
            st = aes_round(0, &st, k);
        }
        st = aes_round(1, &st, &keys[10]);
        assert_eq!(st, hex("69c4e0d86a7b0430d8cdb78070b4c55a"));
        // And back with the equivalent inverse cipher.
        let mut st2: L = core::array::from_fn(|i| st[i] ^ keys[10][i]);
        for k in keys[1..10].iter().rev() {
            st2 = aes_round(2, &st2, &aes_imc(k));
        }
        st2 = aes_round(3, &st2, &keys[0]);
        assert_eq!(st2, pt);
    }

    #[test]
    fn pcmpistri_equal_any() {
        // Find the first of "lo" in "hello".
        let mut d = [0u8; 16];
        d[..2].copy_from_slice(b"lo");
        let mut s = [0u8; 16];
        s[..5].copy_from_slice(b"hello");
        let (res, flags) = pcmpxstrx(&d, &s, 0, pcmp_ilen(&s, 0), pcmp_ilen(&d, 0));
        assert_eq!(res, 0b11100);
        assert_eq!(flags, CC_Z | CC_S | CC_C);
    }
}
