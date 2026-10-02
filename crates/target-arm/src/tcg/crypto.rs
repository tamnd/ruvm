// SPDX-License-Identifier: GPL-2.0-or-later

//! The AES, SHA-1 and SHA-256 instructions: QEMU's `crypto_helper.c` with the AES rounds of
//! `crypto/aes.c`. PMULL is in [`super::vec_helper`].
//!
//! QEMU calls one helper per instruction with pointers to the registers; here one helper,
//! `a64_crypto`, takes the register numbers and the operation. The results are the same.

use ruvm_jit_core::HelperType::{I32, Ptr, Void};
use ruvm_jit_interp::{HelperEnv, Unwind};

use super::helpers::{Def, def};
use super::vec_helper::{Desc, Regs, V, vload, vstore};

/// The operations of `a64_crypto`.
#[allow(missing_docs)]
pub(crate) mod op {
    pub(crate) const AESE: u32 = 1;
    pub(crate) const AESD: u32 = 2;
    pub(crate) const AESMC: u32 = 3;
    pub(crate) const AESIMC: u32 = 4;
    pub(crate) const SHA1C: u32 = 5;
    pub(crate) const SHA1P: u32 = 6;
    pub(crate) const SHA1M: u32 = 7;
    pub(crate) const SHA1SU0: u32 = 8;
    pub(crate) const SHA1H: u32 = 9;
    pub(crate) const SHA1SU1: u32 = 10;
    pub(crate) const SHA256H: u32 = 11;
    pub(crate) const SHA256H2: u32 = 12;
    pub(crate) const SHA256SU0: u32 = 13;
    pub(crate) const SHA256SU1: u32 = 14;
}

def!(CRYPTO, "a64_crypto", 0, Void, [Ptr, I32, I32], h_crypto);

/// The helpers of this module.
pub(crate) const ALL: &[Def] = &[CRYPTO];

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

/// The AES S-box, `AES_sbox`: the multiplicative inverse followed by the affine map.
const SBOX: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        // The inverse is a^254.
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

fn mix_columns(st: &V, inverse: bool) -> V {
    let k: [u8; 4] = if inverse { [14, 11, 13, 9] } else { [2, 3, 1, 1] };
    let mut out = [0u8; 16];
    for c in 0..4 {
        let col = &st[4 * c..4 * c + 4];
        for r in 0..4 {
            let mut x = 0;
            for j in 0..4 {
                x ^= gmul(col[(r + j) % 4], k[j]);
            }
            out[4 * c + r] = x;
        }
    }
    out
}

fn words(v: &V) -> [u32; 4] {
    core::array::from_fn(|i| u32::from_le_bytes(v[4 * i..4 * i + 4].try_into().unwrap()))
}

fn cho(x: u32, y: u32, z: u32) -> u32 {
    (x & (y ^ z)) ^ z
}

fn par(x: u32, y: u32, z: u32) -> u32 {
    x ^ y ^ z
}

fn maj(x: u32, y: u32, z: u32) -> u32 {
    (x & y) | ((x | y) & z)
}

#[allow(non_snake_case)]
fn S0(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}

#[allow(non_snake_case)]
fn S1(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}

fn s0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}

fn s1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}

/// Compute `op` on the registers. For the two register forms the source is `n`.
pub(crate) fn crypto(op: u32, d: &V, n: &V, m: &V) -> V {
    let mut out = *d;
    match op {
        op::AESE | op::AESD => {
            let mut t = [0u8; 16];
            for i in 0..16 {
                t[i] = d[i] ^ n[i];
            }
            for i in 0..16 {
                out[i] = if op == op::AESE {
                    SBOX[t[SHIFTS[i]] as usize]
                } else {
                    ISBOX[t[ISHIFTS[i]] as usize]
                };
            }
        }
        op::AESMC => out = mix_columns(n, false),
        op::AESIMC => out = mix_columns(n, true),
        _ => {
            let mut w = words(d);
            let nw = words(n);
            let mw = words(m);
            sha(op, &mut w, nw, mw);
            for (i, x) in w.iter().enumerate() {
                out[4 * i..4 * i + 4].copy_from_slice(&x.to_le_bytes());
            }
        }
    }
    out
}

/// The SHA-1 and SHA-256 operations on the words of the registers.
fn sha(op: u32, d: &mut [u32; 4], n: [u32; 4], m: [u32; 4]) {
    let mut n = n;
    match op {
        op::SHA1C | op::SHA1P | op::SHA1M => {
            for mi in m {
                let f = match op {
                    op::SHA1C => cho(d[1], d[2], d[3]),
                    op::SHA1P => par(d[1], d[2], d[3]),
                    _ => maj(d[1], d[2], d[3]),
                };
                let t = f.wrapping_add(d[0].rotate_left(5)).wrapping_add(n[0]).wrapping_add(mi);
                n[0] = d[3];
                d[3] = d[2];
                d[2] = d[1].rotate_right(2);
                d[1] = d[0];
                d[0] = t;
            }
        }
        op::SHA1SU0 => {
            // d0 = d[1] ^ d[0] ^ m[0], d1 = n[0] ^ d[1] ^ m[1] on 64 bit halves.
            let (d0, d1) = (
                [d[2] ^ d[0] ^ m[0], d[3] ^ d[1] ^ m[1]],
                [n[0] ^ d[2] ^ m[2], n[1] ^ d[3] ^ m[3]],
            );
            *d = [d0[0], d0[1], d1[0], d1[1]];
        }
        op::SHA1H => *d = [n[0].rotate_right(2), 0, 0, 0],
        op::SHA1SU1 => {
            d[0] = (d[0] ^ n[1]).rotate_left(1);
            d[1] = (d[1] ^ n[2]).rotate_left(1);
            d[2] = (d[2] ^ n[3]).rotate_left(1);
            d[3] = (d[3] ^ d[0]).rotate_left(1);
        }
        op::SHA256H => {
            for mi in m {
                let mut t = cho(n[0], n[1], n[2])
                    .wrapping_add(n[3])
                    .wrapping_add(S1(n[0]))
                    .wrapping_add(mi);
                n[3] = n[2];
                n[2] = n[1];
                n[1] = n[0];
                n[0] = d[3].wrapping_add(t);
                t = t.wrapping_add(maj(d[0], d[1], d[2])).wrapping_add(S0(d[0]));
                d[3] = d[2];
                d[2] = d[1];
                d[1] = d[0];
                d[0] = t;
            }
        }
        op::SHA256H2 => {
            for i in 0..4 {
                let t = cho(d[0], d[1], d[2])
                    .wrapping_add(d[3])
                    .wrapping_add(S1(d[0]))
                    .wrapping_add(m[i]);
                d[3] = d[2];
                d[2] = d[1];
                d[1] = d[0];
                d[0] = n[3 - i].wrapping_add(t);
            }
        }
        op::SHA256SU0 => {
            d[0] = d[0].wrapping_add(s0(d[1]));
            d[1] = d[1].wrapping_add(s0(d[2]));
            d[2] = d[2].wrapping_add(s0(d[3]));
            d[3] = d[3].wrapping_add(s0(n[0]));
        }
        op::SHA256SU1 => {
            d[0] = d[0].wrapping_add(s1(m[2])).wrapping_add(n[1]);
            d[1] = d[1].wrapping_add(s1(m[3])).wrapping_add(n[2]);
            d[2] = d[2].wrapping_add(s1(d[0])).wrapping_add(n[3]);
            d[3] = d[3].wrapping_add(s1(d[1])).wrapping_add(m[0]);
        }
        _ => unreachable!("bad a64_crypto op {op}"),
    }
}

fn h_crypto(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    let r = Regs::unpack(a[1]);
    let desc = Desc::unpack(a[2]);
    let env = &mut *h.env;
    let out = crypto(desc.op, &vload(env, r.d), &vload(env, r.n), &vload(env, r.m));
    vstore(env, r.d, &out);
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sbox() {
        assert_eq!(SBOX[0], 0x63);
        assert_eq!(SBOX[0x53], 0xed);
        assert_eq!(ISBOX[0x63], 0);
    }

    #[test]
    fn mix_columns_inverts() {
        let st: V = core::array::from_fn(|i| (i * 37 + 5) as u8);
        assert_eq!(mix_columns(&mix_columns(&st, false), true), st);
        // The FIPS-197 example column db 13 53 45 becomes 8e 4d a1 bc.
        let mut c = [0u8; 16];
        c[..4].copy_from_slice(&[0xdb, 0x13, 0x53, 0x45]);
        assert_eq!(mix_columns(&c, false)[..4], [0x8e, 0x4d, 0xa1, 0xbc]);
    }
}
