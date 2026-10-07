// SPDX-License-Identifier: GPL-2.0-or-later

//! The scalar crypto helpers of Zknd, Zkne and Zksed, a port of the RV64 part of QEMU's
//! `target/riscv/tcg/crypto_helper.c`.
//!
//! The AES helpers build QEMU's `AESState` with `rs1` in its low eight bytes and `rs2` in
//! the high eight, as QEMU does on a little endian host, and return the low eight bytes. They
//! use the AES steps of the vector crypto helpers in [`super::vcrypto`]. The SHA-2 and SM3
//! instructions of Zknh and Zksh are generated inline by the translator, as in QEMU, and the
//! RV32 only instructions (`aes32*`, `sha512*r`, `sha512*l`, `sha512*h`) are not here.

use ruvm_jit_core::HelperType::I64;
use ruvm_jit_core::types::call_flags::NO_RWG_SE;
use ruvm_jit_interp::{HelperEnv, Unwind};

use super::helpers::Def;
use super::vcrypto::{AES_RCON, SM4_SBOX, aes_imc, aes_isb_isr, aes_mc, aes_sb_sr, aes_subword};

macro_rules! def {
    ($id:ident, $name:literal, [$($a:expr),*], $f:expr) => {
        pub(crate) const $id: Def =
            Def { name: $name, flags: NO_RWG_SE, ret: I64, args: &[$($a),*], f: $f };
    };
}

def!(AES64ES, "aes64es", [I64, I64], h_aes64es);
def!(AES64ESM, "aes64esm", [I64, I64], h_aes64esm);
def!(AES64DS, "aes64ds", [I64, I64], h_aes64ds);
def!(AES64DSM, "aes64dsm", [I64, I64], h_aes64dsm);
def!(AES64KS2, "aes64ks2", [I64, I64], h_aes64ks2);
def!(AES64KS1I, "aes64ks1i", [I64, I64], h_aes64ks1i);
def!(AES64IM, "aes64im", [I64], h_aes64im);
def!(SM4ED, "sm4ed", [I64, I64, I64], h_sm4ed);
def!(SM4KS, "sm4ks", [I64, I64, I64], h_sm4ks);

/// The helpers of this module.
pub(crate) const ALL: &[Def] =
    &[AES64ES, AES64ESM, AES64DS, AES64DSM, AES64KS2, AES64KS1I, AES64IM, SM4ED, SM4KS];

/// The `AESState` with `lo` in its first eight bytes and `hi` in the last eight.
fn state(lo: u64, hi: u64) -> [u8; 16] {
    let mut t = [0u8; 16];
    t[..8].copy_from_slice(&lo.to_le_bytes());
    t[8..].copy_from_slice(&hi.to_le_bytes());
    t
}

/// The first eight bytes of an `AESState`, `t.d[0]`.
fn low(t: &[u8; 16]) -> u128 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&t[..8]);
    u128::from(u64::from_le_bytes(b))
}

/// `aes64es`: SubBytes and ShiftRows.
pub(super) fn aes64es(rs1: u64, rs2: u64) -> u64 {
    low(&aes_sb_sr(&state(rs1, rs2))) as u64
}

/// `aes64esm`: SubBytes, ShiftRows and MixColumns.
pub(super) fn aes64esm(rs1: u64, rs2: u64) -> u64 {
    let mut t = aes_sb_sr(&state(rs1, rs2));
    aes_mc(&mut t);
    low(&t) as u64
}

/// `aes64ds`: InvSubBytes and InvShiftRows.
pub(super) fn aes64ds(rs1: u64, rs2: u64) -> u64 {
    low(&aes_isb_isr(&state(rs1, rs2))) as u64
}

/// `aes64dsm`: InvSubBytes, InvShiftRows and InvMixColumns. The instruction does not
/// include a round key, so QEMU supplies a zero one.
pub(super) fn aes64dsm(rs1: u64, rs2: u64) -> u64 {
    let mut t = aes_isb_isr(&state(rs1, rs2));
    aes_imc(&mut t);
    low(&t) as u64
}

/// `aes64ks2`: the second half of a key schedule step.
pub(super) fn aes64ks2(rs1: u64, rs2: u64) -> u64 {
    let rs1_hi = (rs1 >> 32) as u32;
    let rs2_lo = rs2 as u32;
    let rs2_hi = (rs2 >> 32) as u32;
    let r_lo = rs1_hi ^ rs2_lo;
    let r_hi = rs1_hi ^ rs2_lo ^ rs2_hi;
    (u64::from(r_hi) << 32) | u64::from(r_lo)
}

/// `aes64ks1i`: SubWord of the high word of `rs1`, rotated and XORed with the round
/// constant unless `rnum` is 0xa, in both words of the result. ShiftRows of a state whose
/// four columns are equal does nothing, so only SubBytes is left of QEMU's
/// `aesenc_SB_SR_AK()`.
pub(super) fn aes64ks1i(rs1: u64, rnum: u64) -> u64 {
    let mut temp = (rs1 >> 32) as u32;
    let mut rc = 0;
    if rnum != 0xa {
        temp = temp.rotate_right(8);
        rc = AES_RCON[rnum as usize];
    }
    let w = u64::from(aes_subword(temp) ^ rc);
    (w << 32) | w
}

/// `aes64im`: InvMixColumns of `rs1`.
pub(super) fn aes64im(rs1: u64) -> u64 {
    let mut t = state(rs1, 0);
    aes_imc(&mut t);
    low(&t) as u64
}

/// `sext32_xlen()`.
fn sext32(x: u32) -> u64 {
    i64::from(x as i32) as u64
}

/// `sm4ed`: one byte of the SM4 round function, the S-box output through the linear
/// transform L, rotated to byte `shamt / 8` and XORed into `rs1`.
pub(super) fn sm4ed(rs1: u64, rs2: u64, shamt: u64) -> u64 {
    let sb = u32::from(SM4_SBOX[usize::from((rs2 >> shamt) as u8)]);
    let x = sb ^ (sb << 8) ^ (sb << 2) ^ (sb << 18) ^ ((sb & 0x3f) << 26) ^ ((sb & 0xc0) << 10);
    sext32(x.rotate_left(shamt as u32) ^ rs1 as u32)
}

/// `sm4ks`: the same with the transform L' of the key schedule.
pub(super) fn sm4ks(rs1: u64, rs2: u64, shamt: u64) -> u64 {
    let sb = u32::from(SM4_SBOX[usize::from((rs2 >> shamt) as u8)]);
    let x =
        sb ^ ((sb & 0x07) << 29) ^ ((sb & 0xfe) << 7) ^ ((sb & 0x01) << 23) ^ ((sb & 0xf8) << 13);
    sext32(x.rotate_left(shamt as u32) ^ rs1 as u32)
}

fn h_aes64es(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok(u128::from(aes64es(a[0], a[1])))
}

fn h_aes64esm(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok(u128::from(aes64esm(a[0], a[1])))
}

fn h_aes64ds(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok(u128::from(aes64ds(a[0], a[1])))
}

fn h_aes64dsm(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok(u128::from(aes64dsm(a[0], a[1])))
}

fn h_aes64ks2(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok(u128::from(aes64ks2(a[0], a[1])))
}

fn h_aes64ks1i(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok(u128::from(aes64ks1i(a[0], a[1])))
}

fn h_aes64im(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok(u128::from(aes64im(a[0])))
}

fn h_sm4ed(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok(u128::from(sm4ed(a[0], a[1], a[2])))
}

fn h_sm4ks(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok(u128::from(sm4ks(a[0], a[1], a[2])))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FIPS-197 appendix B: the first round of AES-128 with the key
    /// 2b7e1516 28aed2a6 abf71588 09cf4f3c on the input 3243f6a8 885a308d 313198a2 e0370734.
    /// The state after the initial AddRoundKey is 193de3be a0f4e22b 9ac68d2a e9f84808 and
    /// after SubBytes, ShiftRows and MixColumns it is 046681e5 e0cb199a 48f8d37a 2806264c.
    #[test]
    fn aes64esm_is_a_fips197_round() {
        let (s_lo, s_hi) = (0x2be2_f4a0_bee3_3d19, 0x0848_f8e9_2a8d_c69a);
        // The high half is the low half of the state with its halves swapped.
        let lo = aes64esm(s_lo, s_hi);
        let hi = aes64esm(s_hi, s_lo);
        assert_eq!(lo, 0x9a19_cbe0_e581_6604);
        assert_eq!(hi, 0x4c26_0628_7ad3_f848);
        // The decryption round undoes it.
        let back_lo = aes64ds(aes64im(lo), aes64im(hi));
        let back_hi = aes64ds(aes64im(hi), aes64im(lo));
        assert_eq!((back_lo, back_hi), (s_lo, s_hi));
        assert_eq!(aes64dsm(aes64im(lo), aes64im(hi)), aes64im(aes64ds(aes64im(lo), aes64im(hi))));
    }

    /// The AES-128 key expansion of FIPS-197 appendix A.1: round key 1 of
    /// 2b7e1516 28aed2a6 abf71588 09cf4f3c is a0fafe17 88542cb1 23a33939 2a6c7605.
    #[test]
    fn aes64ks_expands_the_fips197_key() {
        let k0 = u64::from_le_bytes([0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6]);
        let k1 = u64::from_le_bytes([0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c]);
        let t = aes64ks1i(k1, 0);
        let n0 = aes64ks2(t, k0);
        let n1 = aes64ks2(n0, k1);
        assert_eq!(n0.to_le_bytes(), [0xa0, 0xfa, 0xfe, 0x17, 0x88, 0x54, 0x2c, 0xb1]);
        assert_eq!(n1.to_le_bytes(), [0x23, 0xa3, 0x39, 0x39, 0x2a, 0x6c, 0x76, 0x05]);
        // Round 10 of AES-256 skips the rotation and the round constant.
        assert_eq!(aes64ks1i(0x0100_0000_0000_0000, 0xa), 0x7c63_6363_7c63_6363);
    }
}
