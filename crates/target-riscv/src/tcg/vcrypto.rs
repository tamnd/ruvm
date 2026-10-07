// SPDX-License-Identifier: GPL-2.0-or-later

//! The vector crypto helpers, a port of QEMU's `target/riscv/tcg/vcrypto_helper.c` with the
//! parts of `crypto/aes.c`, `crypto/sm4.c` and `crypto/clmul.c` they use.
//!
//! The element-wise Zvbc, Zvkb and Zvbb operations are [`vop_def!`] helpers, one for every
//! SEW and for both the `.vv` and the `.vx` (`.vi`) forms. The element group operations
//! (Zvkned, Zvknh, Zvksh, Zvkg, Zvksed) share the loop of [`group`]; the registers keep the
//! bytes in memory order, so an AES or GHASH element group is its 16 bytes, and a 32-bit
//! element is the little endian word QEMU reads on a little endian host.
//!
//! Deliberate differences from QEMU:
//!
//! - QEMU checks that `vl` and `vstart` are multiples of the element group size in the
//!   separate helper `egs_check`, called before the operation when the translator cannot
//!   prove it. Here every element group helper starts with that check, which raises the
//!   same illegal instruction exception before anything changes.
//! - QEMU has a helper per SEW for `vsha2ch` and `vsha2cl` (`vsha2ch32_vv`,
//!   `vsha2ch64_vv`); here one helper takes SEW from the descriptor.
//! - The AES rounds are computed a byte at a time from the S-boxes where QEMU uses its
//!   T-tables; the results are the same.

use ruvm_jit::{Cpu, CpuLoopExit, Ra};
use ruvm_jit_core::HelperType::{I32, I64, Ptr, Void};
use ruvm_jit_core::types::call_flags::NO_WG;
use ruvm_jit_interp::HelperEnv;

use super::helpers::{Def, run};
use super::vector::{
    Desc, HR, Shape, def, set_1s, set_vstart, total_elems, vget, vl, voff, vop_def, vset, vsew,
    vstart,
};
use crate::cpu::EXCP_ILLEGAL_INST;

// The element-wise operations. Each gets `vs2`, the first source (`vs1` or the scalar) and
// log2 of SEW in bytes; the operands are zero extended and the result is truncated to SEW
// by the caller.

/// The bits in an element of `1 << sew` bytes.
#[inline]
fn bits(sew: u32) -> u32 {
    8 << sew
}

/// `clmul64()`: the low 64 bits of the carry-less product.
pub(super) fn clmul64(y: u64, x: u64) -> u64 {
    let mut result = 0;
    for j in (0..64).rev() {
        if (y >> j) & 1 != 0 {
            result ^= x << j;
        }
    }
    result
}

/// `clmulh64()`: the high 64 bits of the carry-less product.
pub(super) fn clmulh64(y: u64, x: u64) -> u64 {
    let mut result = 0;
    for j in (1..64).rev() {
        if (y >> j) & 1 != 0 {
            result ^= x >> (64 - j);
        }
    }
    result
}

/// `rol8()` to `rol64()`: rotate the SEW-bit `x` left by `n` modulo SEW.
pub(super) fn rol(x: u64, n: u64, sew: u32) -> u64 {
    let w = bits(sew);
    let n = (n as u32) & (w - 1);
    if n == 0 {
        x
    } else {
        let mask = if w == 64 { u64::MAX } else { (1u64 << w) - 1 };
        ((x << n) | (x >> (w - n))) & mask
    }
}

/// `ror8()` to `ror64()`.
pub(super) fn ror(x: u64, n: u64, sew: u32) -> u64 {
    let w = u64::from(bits(sew));
    rol(x, w - (n & (w - 1)), sew)
}

/// `brev8()`: reverse the bits of every byte.
pub(super) fn brev8(mut val: u64) -> u64 {
    val = ((val & 0x5555_5555_5555_5555) << 1) | ((val & 0xaaaa_aaaa_aaaa_aaaa) >> 1);
    val = ((val & 0x3333_3333_3333_3333) << 2) | ((val & 0xcccc_cccc_cccc_cccc) >> 2);
    ((val & 0x0f0f_0f0f_0f0f_0f0f) << 4) | ((val & 0xf0f0_f0f0_f0f0_f0f0) >> 4)
}

/// `bswap16()` to `bswap64()` of a SEW-bit value; the identity for bytes.
pub(super) fn rev8(x: u64, sew: u32) -> u64 {
    x.swap_bytes() >> (64 - bits(sew))
}

/// `revbit8()` to `revbit64()`.
pub(super) fn brev(x: u64, sew: u32) -> u64 {
    x.reverse_bits() >> (64 - bits(sew))
}

/// `clz8()` to `clz64()`: SEW for zero.
pub(super) fn clz(x: u64, sew: u32) -> u64 {
    if x == 0 { u64::from(bits(sew)) } else { u64::from(x.leading_zeros() - (64 - bits(sew))) }
}

/// `ctz8()` to `ctz64()`: SEW for zero.
pub(super) fn ctz(x: u64, sew: u32) -> u64 {
    if x == 0 { u64::from(bits(sew)) } else { u64::from(x.trailing_zeros()) }
}

/// `DO_SLL()` of `vwsll`: the 2*SEW-bit shift of the zero extended `vs2` element by the low
/// `log2(2 * SEW)` bits of the first source.
pub(super) fn wsll(s2: u64, s1: u64, sew: u32) -> u64 {
    s2 << (s1 & u64::from(2 * bits(sew) - 1))
}

vop_def!(VCLMUL, "vclmul", Shape::Single, |s2, s1, _| clmul64(s2, s1));
vop_def!(VCLMULH, "vclmulh", Shape::Single, |s2, s1, _| clmulh64(s2, s1));
vop_def!(VROR, "vror", Shape::Single, ror);
vop_def!(VROL, "vrol", Shape::Single, rol);
vop_def!(VANDN, "vandn", Shape::Single, |s2, s1, _| s2 & !s1);
vop_def!(VBREV8, "vbrev8", Shape::Single, |s2, _, _| brev8(s2));
vop_def!(VREV8, "vrev8", Shape::Single, |s2, _, sew| rev8(s2, sew));
vop_def!(VBREV, "vbrev", Shape::Single, |s2, _, sew| brev(s2, sew));
vop_def!(VCLZ, "vclz", Shape::Single, |s2, _, sew| clz(s2, sew));
vop_def!(VCTZ, "vctz", Shape::Single, |s2, _, sew| ctz(s2, sew));
vop_def!(VCPOP_V, "vcpop_v", Shape::Single, |s2, _, _| u64::from(s2.count_ones()));
vop_def!(VWSLL, "vwsll", Shape::Widen, wsll);

// The element group operations.

/// `HELPER(egs_check)`: `vl` and `vstart` must be multiples of the element group size.
fn egs_check(cpu: &mut Cpu<'_>, egs: usize) -> Result<(), CpuLoopExit> {
    if vl(cpu.env) % egs != 0 || vstart(cpu.env) % egs != 0 {
        return Err(cpu.raise_exception(EXCP_ILLEGAL_INST, Ra::Tb));
    }
    Ok(())
}

/// An operation on element group `i`, with the descriptor and the immediate.
type GroupOp = fn(&mut [u8], &Desc, usize, u64);

/// The loop of the element group helpers: `op` on every group from `vstart / egs` to
/// `vl / egs`, then `vstart` cleared and the tail set to ones when tail agnostic.
pub(super) fn group(env: &mut [u8], d: &Desc, egs: usize, uimm: u64, op: GroupOp) {
    let vl = vl(env);
    let start = vstart(env);
    if start >= vl {
        // VSTART_CHECK_EARLY_EXIT().
        set_vstart(env, 0);
        return;
    }
    for i in start / egs..vl / egs {
        op(env, d, i, uimm);
    }
    set_vstart(env, 0);
    let log2 = vsew(env);
    let total = total_elems(env, d, log2);
    set_1s(env, d.vd, d.vta, vl << log2, total << log2);
}

/// A helper of an element group operation: `group_def!(ID, "name", egs, op)`, with the
/// descriptor and the immediate as arguments.
macro_rules! group_def {
    ($id:ident, $name:literal, $egs:expr, $op:expr) => {
        def!($id, $name, NO_WG, Void, [Ptr, I32, I64], {
            fn h(e: &mut HelperEnv<'_>, a: &[u64]) -> HR {
                let d = Desc::decode(a[1] as u32);
                let uimm = a[2];
                run(e, |cpu| {
                    egs_check(cpu, $egs)?;
                    group(cpu.env, &d, $egs, uimm, $op);
                    Ok(0)
                })
            }
            h
        });
    };
}

/// The words of element group `i` of `N` SEW-32 elements of register group `reg`.
fn words<const N: usize>(env: &[u8], reg: u32, i: usize) -> [u32; N] {
    let mut w = [0u32; N];
    for (j, x) in w.iter_mut().enumerate() {
        *x = vget(env, reg, i * N + j, 2) as u32;
    }
    w
}

/// Set element group `i` of `N` SEW-32 elements of register group `reg` to `w`.
fn set_words<const N: usize>(env: &mut [u8], reg: u32, i: usize, w: &[u32; N]) {
    for (j, x) in w.iter().enumerate() {
        vset(env, reg, i * N + j, 2, u64::from(*x));
    }
}

/// The SEW-64 elements of element group `i` of four elements of register group `reg`.
fn dwords(env: &[u8], reg: u32, i: usize) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (j, x) in w.iter_mut().enumerate() {
        *x = vget(env, reg, i * 4 + j, 3);
    }
    w
}

/// Set the SEW-64 elements of element group `i` of four elements of register group `reg`.
fn set_dwords(env: &mut [u8], reg: u32, i: usize, w: &[u64; 4]) {
    for (j, x) in w.iter().enumerate() {
        vset(env, reg, i * 4 + j, 3, *x);
    }
}

/// The 16 bytes of 128-bit element group `i` of register group `reg`.
fn block(env: &[u8], reg: u32, i: usize) -> [u8; 16] {
    let mut b = [0u8; 16];
    for (k, x) in b.iter_mut().enumerate() {
        *x = env[voff(reg, i * 16 + k)];
    }
    b
}

/// Set 128-bit element group `i` of register group `reg` to the bytes `b`.
fn set_block(env: &mut [u8], reg: u32, i: usize, b: &[u8; 16]) {
    for (k, x) in b.iter().enumerate() {
        env[voff(reg, i * 16 + k)] = *x;
    }
}

// AES.

/// `AES_sbox`.
const AES_SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

/// `AES_isbox`, the inverse of [`AES_SBOX`].
const AES_ISBOX: [u8; 256] = {
    let mut inv = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        inv[AES_SBOX[i] as usize] = i as u8;
        i += 1;
    }
    inv
};

/// `AES_SH()`: the source byte of byte `x` in ShiftRows.
const fn aes_sh(x: usize) -> usize {
    (x * 5) & 15
}

/// `AES_ISH()`: the source byte of byte `x` in InvShiftRows.
const fn aes_ish(x: usize) -> usize {
    (x * 13) & 15
}

/// Multiplication by x in GF(2^8).
#[inline]
fn xtime(b: u8) -> u8 {
    (b << 1) ^ if b & 0x80 != 0 { 0x1b } else { 0 }
}

/// Multiplication in GF(2^8).
fn gmul(mut a: u8, mut b: u8) -> u8 {
    let mut p = 0;
    while b != 0 {
        if b & 1 != 0 {
            p ^= a;
        }
        a = xtime(a);
        b >>= 1;
    }
    p
}

/// SubBytes and ShiftRows.
pub(super) fn aes_sb_sr(st: &[u8; 16]) -> [u8; 16] {
    let mut t = [0u8; 16];
    for (k, x) in t.iter_mut().enumerate() {
        *x = AES_SBOX[st[aes_sh(k)] as usize];
    }
    t
}

/// InvSubBytes and InvShiftRows.
pub(super) fn aes_isb_isr(st: &[u8; 16]) -> [u8; 16] {
    let mut t = [0u8; 16];
    for (k, x) in t.iter_mut().enumerate() {
        *x = AES_ISBOX[st[aes_ish(k)] as usize];
    }
    t
}

/// MixColumns.
pub(super) fn aes_mc(st: &mut [u8; 16]) {
    for c in st.chunks_exact_mut(4) {
        let [a0, a1, a2, a3] = [c[0], c[1], c[2], c[3]];
        c[0] = xtime(a0) ^ xtime(a1) ^ a1 ^ a2 ^ a3;
        c[1] = a0 ^ xtime(a1) ^ xtime(a2) ^ a2 ^ a3;
        c[2] = a0 ^ a1 ^ xtime(a2) ^ xtime(a3) ^ a3;
        c[3] = xtime(a0) ^ a0 ^ a1 ^ a2 ^ xtime(a3);
    }
}

/// InvMixColumns.
pub(super) fn aes_imc(st: &mut [u8; 16]) {
    for c in st.chunks_exact_mut(4) {
        let [a0, a1, a2, a3] = [c[0], c[1], c[2], c[3]];
        c[0] = gmul(a0, 14) ^ gmul(a1, 11) ^ gmul(a2, 13) ^ gmul(a3, 9);
        c[1] = gmul(a0, 9) ^ gmul(a1, 14) ^ gmul(a2, 11) ^ gmul(a3, 13);
        c[2] = gmul(a0, 13) ^ gmul(a1, 9) ^ gmul(a2, 14) ^ gmul(a3, 11);
        c[3] = gmul(a0, 11) ^ gmul(a1, 13) ^ gmul(a2, 9) ^ gmul(a3, 14);
    }
}

/// `xor_round_key()`, AddRoundKey.
fn aes_ak(st: &mut [u8; 16], key: &[u8; 16]) {
    for (s, k) in st.iter_mut().zip(key) {
        *s ^= *k;
    }
}

/// The AES rounds of the Zvkned instructions.
#[derive(Clone, Copy)]
enum AesRound {
    /// `aesenc_SB_SR_AK()`: `vaesef`.
    EncFinal,
    /// `aesenc_SB_SR_MC_AK()`: `vaesem`.
    EncMiddle,
    /// `aesdec_ISB_ISR_AK()`: `vaesdf`.
    DecFinal,
    /// `aesdec_ISB_ISR_AK_IMC()`: `vaesdm`.
    DecMiddle,
    /// `xor_round_key()`: `vaesz`.
    Zero,
}

/// One AES round of `round` on the state `st` with the round key `key`.
fn aes_round(round: AesRound, st: &[u8; 16], key: &[u8; 16]) -> [u8; 16] {
    let mut t = match round {
        AesRound::EncFinal | AesRound::EncMiddle => aes_sb_sr(st),
        AesRound::DecFinal | AesRound::DecMiddle => aes_isb_isr(st),
        AesRound::Zero => *st,
    };
    if let AesRound::EncMiddle = round {
        aes_mc(&mut t);
    }
    aes_ak(&mut t, key);
    if let AesRound::DecMiddle = round {
        aes_imc(&mut t);
    }
    t
}

/// `GEN_ZVKNED_HELPER_VV()` (`vs = false`) and `GEN_ZVKNED_HELPER_VS()` (`vs = true`):
/// round `round` on group `i` of `vd` with the round key in group `i` of `vs2`, or in group
/// 0 of `vs2` for the `.vs` forms.
fn aes_group(env: &mut [u8], d: &Desc, i: usize, round: AesRound, vs: bool) {
    let key = block(env, d.vs2, if vs { 0 } else { i });
    let st = block(env, d.vd, i);
    let r = aes_round(round, &st, &key);
    set_block(env, d.vd, i, &r);
}

group_def!(VAESEF_VV, "vaesef_vv", 4, |e, d, i, _| aes_group(e, d, i, AesRound::EncFinal, false));
group_def!(VAESEF_VS, "vaesef_vs", 4, |e, d, i, _| aes_group(e, d, i, AesRound::EncFinal, true));
group_def!(VAESDF_VV, "vaesdf_vv", 4, |e, d, i, _| aes_group(e, d, i, AesRound::DecFinal, false));
group_def!(VAESDF_VS, "vaesdf_vs", 4, |e, d, i, _| aes_group(e, d, i, AesRound::DecFinal, true));
group_def!(VAESEM_VV, "vaesem_vv", 4, |e, d, i, _| aes_group(e, d, i, AesRound::EncMiddle, false));
group_def!(VAESEM_VS, "vaesem_vs", 4, |e, d, i, _| aes_group(e, d, i, AesRound::EncMiddle, true));
group_def!(VAESDM_VV, "vaesdm_vv", 4, |e, d, i, _| aes_group(e, d, i, AesRound::DecMiddle, false));
group_def!(VAESDM_VS, "vaesdm_vs", 4, |e, d, i, _| aes_group(e, d, i, AesRound::DecMiddle, true));
group_def!(VAESZ_VS, "vaesz_vs", 4, |e, d, i, _| aes_group(e, d, i, AesRound::Zero, true));
group_def!(VAESKF1_VI, "vaeskf1_vi", 4, aeskf1);
group_def!(VAESKF2_VI, "vaeskf2_vi", 4, aeskf2);

/// The AES round constants of the key schedules.
pub(super) const AES_RCON: [u32; 10] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];

/// SubWord of the AES key schedule.
pub(super) fn aes_subword(w: u32) -> u32 {
    u32::from_le_bytes(w.to_le_bytes().map(|b| AES_SBOX[b as usize]))
}

/// `HELPER(vaeskf1_vi)`: the next AES-128 round key, round `uimm`.
fn aeskf1(env: &mut [u8], d: &Desc, i: usize, uimm: u64) {
    let mut uimm = uimm as u32 & 0b1111;
    if uimm > 10 || uimm == 0 {
        uimm ^= 0b1000;
    }
    let rk: [u32; 4] = words(env, d.vs2, i);
    let tmp = rk[3].rotate_right(8);
    let mut out = [0u32; 4];
    out[0] = rk[0] ^ aes_subword(tmp) ^ AES_RCON[(uimm - 1) as usize];
    out[1] = rk[1] ^ out[0];
    out[2] = rk[2] ^ out[1];
    out[3] = rk[3] ^ out[2];
    set_words(env, d.vd, i, &out);
}

/// `HELPER(vaeskf2_vi)`: the next AES-256 round key, round `uimm`, from the two before it
/// in `vd` and `vs2`.
fn aeskf2(env: &mut [u8], d: &Desc, i: usize, uimm: u64) {
    let mut uimm = uimm as u32 & 0b1111;
    if !(2..=14).contains(&uimm) {
        uimm ^= 0b1000;
    }
    let rk0: [u32; 4] = words(env, d.vd, i);
    let rk1: [u32; 4] = words(env, d.vs2, i);
    let mut out = [0u32; 4];
    out[0] = if uimm % 2 == 0 {
        rk0[0] ^ aes_subword(rk1[3].rotate_right(8)) ^ AES_RCON[((uimm - 1) / 2) as usize]
    } else {
        rk0[0] ^ aes_subword(rk1[3])
    };
    out[1] = rk0[1] ^ out[0];
    out[2] = rk0[2] ^ out[1];
    out[3] = rk0[3] ^ out[2];
    set_words(env, d.vd, i, &out);
}

// SHA-2.

/// The SHA-2 word operations of a word size.
trait Sha2Word: Copy {
    fn wadd(self, o: Self) -> Self;
    fn sig0(self) -> Self;
    fn sig1(self) -> Self;
    fn sum0(self) -> Self;
    fn sum1(self) -> Self;
    fn and(self, o: Self) -> Self;
    fn xor(self, o: Self) -> Self;
    fn not(self) -> Self;
}

impl Sha2Word for u32 {
    fn wadd(self, o: u32) -> u32 {
        self.wrapping_add(o)
    }
    fn sig0(self) -> u32 {
        self.rotate_right(7) ^ self.rotate_right(18) ^ (self >> 3)
    }
    fn sig1(self) -> u32 {
        self.rotate_right(17) ^ self.rotate_right(19) ^ (self >> 10)
    }
    fn sum0(self) -> u32 {
        self.rotate_right(2) ^ self.rotate_right(13) ^ self.rotate_right(22)
    }
    fn sum1(self) -> u32 {
        self.rotate_right(6) ^ self.rotate_right(11) ^ self.rotate_right(25)
    }
    fn and(self, o: u32) -> u32 {
        self & o
    }
    fn xor(self, o: u32) -> u32 {
        self ^ o
    }
    fn not(self) -> u32 {
        !self
    }
}

impl Sha2Word for u64 {
    fn wadd(self, o: u64) -> u64 {
        self.wrapping_add(o)
    }
    fn sig0(self) -> u64 {
        self.rotate_right(1) ^ self.rotate_right(8) ^ (self >> 7)
    }
    fn sig1(self) -> u64 {
        self.rotate_right(19) ^ self.rotate_right(61) ^ (self >> 6)
    }
    fn sum0(self) -> u64 {
        self.rotate_right(28) ^ self.rotate_right(34) ^ self.rotate_right(39)
    }
    fn sum1(self) -> u64 {
        self.rotate_right(14) ^ self.rotate_right(18) ^ self.rotate_right(41)
    }
    fn and(self, o: u64) -> u64 {
        self & o
    }
    fn xor(self, o: u64) -> u64 {
        self ^ o
    }
    fn not(self) -> u64 {
        !self
    }
}

/// `vsha2ms_e32()` and `vsha2ms_e64()`: four words of the message schedule.
fn sha2ms<W: Sha2Word>(vd: &mut [W; 4], vs1: &[W; 4], vs2: &[W; 4]) {
    let r0 = vs1[2].sig1().wadd(vs2[1]).wadd(vd[1].sig0()).wadd(vd[0]);
    let r1 = vs1[3].sig1().wadd(vs2[2]).wadd(vd[2].sig0()).wadd(vd[1]);
    let r2 = r0.sig1().wadd(vs2[3]).wadd(vd[3].sig0()).wadd(vd[2]);
    let r3 = r1.sig1().wadd(vs1[0]).wadd(vs2[0].sig0()).wadd(vd[3]);
    *vd = [r0, r1, r2, r3];
}

/// `vsha2c_32()` and `vsha2c_64()`: two rounds with the message words plus constants
/// `w0` and `w1`.
fn sha2c<W: Sha2Word>(vs2: &[W; 4], vd: &mut [W; 4], w0: W, w1: W) {
    let ch = |x: W, y: W, z: W| x.and(y).xor(x.not().and(z));
    let maj = |x: W, y: W, z: W| x.and(y).xor(x.and(z)).xor(y.and(z));
    let (mut a, mut b, mut e, mut f) = (vs2[3], vs2[2], vs2[1], vs2[0]);
    let (mut c, mut dd, mut g, mut h) = (vd[3], vd[2], vd[1], vd[0]);
    for w in [w0, w1] {
        let t1 = h.wadd(e.sum1()).wadd(ch(e, f, g)).wadd(w);
        let t2 = a.sum0().wadd(maj(a, b, c));
        h = g;
        g = f;
        f = e;
        e = dd.wadd(t1);
        dd = c;
        c = b;
        b = a;
        a = t1.wadd(t2);
    }
    *vd = [f, e, b, a];
}

/// `HELPER(vsha2ms_vv)`.
fn sha2ms_group(env: &mut [u8], d: &Desc, i: usize, _: u64) {
    if d.esz == 2 {
        let mut vd: [u32; 4] = words(env, d.vd, i);
        sha2ms(&mut vd, &words(env, d.vs1, i), &words(env, d.vs2, i));
        set_words(env, d.vd, i, &vd);
    } else {
        let mut vd = dwords(env, d.vd, i);
        sha2ms(&mut vd, &dwords(env, d.vs1, i), &dwords(env, d.vs2, i));
        set_dwords(env, d.vd, i, &vd);
    }
}

/// `HELPER(vsha2ch32_vv)` to `HELPER(vsha2cl64_vv)`: two rounds with elements `off` and
/// `off + 1` of the `vs1` group, 2 for `vsha2ch`, 0 for `vsha2cl`.
fn sha2c_group(env: &mut [u8], d: &Desc, i: usize, off: usize) {
    if d.esz == 2 {
        let vs1: [u32; 4] = words(env, d.vs1, i);
        let mut vd: [u32; 4] = words(env, d.vd, i);
        sha2c(&words(env, d.vs2, i), &mut vd, vs1[off], vs1[off + 1]);
        set_words(env, d.vd, i, &vd);
    } else {
        let vs1 = dwords(env, d.vs1, i);
        let mut vd = dwords(env, d.vd, i);
        sha2c(&dwords(env, d.vs2, i), &mut vd, vs1[off], vs1[off + 1]);
        set_dwords(env, d.vd, i, &vd);
    }
}

group_def!(VSHA2MS_VV, "vsha2ms_vv", 4, sha2ms_group);
group_def!(VSHA2CH_VV, "vsha2ch_vv", 4, |e, d, i, _| sha2c_group(e, d, i, 2));
group_def!(VSHA2CL_VV, "vsha2cl_vv", 4, |e, d, i, _| sha2c_group(e, d, i, 0));

// SM3.

/// `p1()`.
fn sm3_p1(x: u32) -> u32 {
    x ^ x.rotate_left(15) ^ x.rotate_left(23)
}

/// `zvksh_w()`: a word of the SM3 message expansion.
fn zvksh_w(m16: u32, m9: u32, m3: u32, m13: u32, m6: u32) -> u32 {
    sm3_p1(m16 ^ m9 ^ m3.rotate_left(15)) ^ m13.rotate_left(7) ^ m6
}

/// `HELPER(vsm3me_vv)`: eight words of the SM3 message expansion.
fn sm3me_group(env: &mut [u8], d: &Desc, i: usize, _: u64) {
    let mut w = [0u32; 24];
    let w1: [u32; 8] = words(env, d.vs1, i);
    let w2: [u32; 8] = words(env, d.vs2, i);
    for j in 0..8 {
        w[j] = w1[j].swap_bytes();
        w[j + 8] = w2[j].swap_bytes();
    }
    for j in 0..8 {
        w[j + 16] = zvksh_w(w[j], w[j + 7], w[j + 13], w[j + 3], w[j + 10]);
    }
    let mut out = [0u32; 8];
    for (j, x) in out.iter_mut().enumerate() {
        *x = w[j + 16].swap_bytes();
    }
    set_words(env, d.vd, i, &out);
}

/// `ff_j()`.
fn sm3_ff(x: u32, y: u32, z: u32, j: u32) -> u32 {
    if j <= 15 { x ^ y ^ z } else { (x & y) | (x & z) | (y & z) }
}

/// `gg_j()`.
fn sm3_gg(x: u32, y: u32, z: u32, j: u32) -> u32 {
    if j <= 15 { x ^ y ^ z } else { (x & y) | (!x & z) }
}

/// `t_j()`.
fn sm3_t(j: u32) -> u32 {
    if j <= 15 { 0x79cc_4519 } else { 0x7a87_9d8a }
}

/// `p_0()`.
fn sm3_p0(x: u32) -> u32 {
    x ^ x.rotate_left(9) ^ x.rotate_left(17)
}

/// `sm3c()`: rounds `2 * uimm` and `2 * uimm + 1` of the SM3 compression on the state
/// `vs1` with the message words `vs2`.
fn sm3c(vd: &mut [u32; 8], vs1: &mut [u32; 8], vs2: &[u32; 8], uimm: u32) {
    let x0 = vs2[0] ^ vs2[4];
    let x1 = vs2[1] ^ vs2[5];
    let mut j = 2 * uimm;
    let mut ss1 =
        vs1[0].rotate_left(12).wrapping_add(vs1[4]).wrapping_add(sm3_t(j).rotate_left(j % 32));
    ss1 = ss1.rotate_left(7);
    let mut ss2 = ss1 ^ vs1[0].rotate_left(12);
    let mut tt1 =
        sm3_ff(vs1[0], vs1[1], vs1[2], j).wrapping_add(vs1[3]).wrapping_add(ss2).wrapping_add(x0);
    let mut tt2 = sm3_gg(vs1[4], vs1[5], vs1[6], j)
        .wrapping_add(vs1[7])
        .wrapping_add(ss1)
        .wrapping_add(vs2[0]);
    vs1[3] = vs1[2];
    vd[3] = vs1[1].rotate_left(9);
    vs1[1] = vs1[0];
    vd[1] = tt1;
    vs1[7] = vs1[6];
    vd[7] = vs1[5].rotate_left(19);
    vs1[5] = vs1[4];
    vd[5] = sm3_p0(tt2);
    j = 2 * uimm + 1;
    ss1 = vd[1].rotate_left(12).wrapping_add(vd[5]).wrapping_add(sm3_t(j).rotate_left(j % 32));
    ss1 = ss1.rotate_left(7);
    ss2 = ss1 ^ vd[1].rotate_left(12);
    tt1 = sm3_ff(vd[1], vs1[1], vd[3], j).wrapping_add(vs1[3]).wrapping_add(ss2).wrapping_add(x1);
    tt2 =
        sm3_gg(vd[5], vs1[5], vd[7], j).wrapping_add(vs1[7]).wrapping_add(ss1).wrapping_add(vs2[1]);
    vd[2] = vs1[1].rotate_left(9);
    vd[0] = tt1;
    vd[6] = vs1[5].rotate_left(19);
    vd[4] = sm3_p0(tt2);
}

/// `HELPER(vsm3c_vi)`.
fn sm3c_group(env: &mut [u8], d: &Desc, i: usize, uimm: u64) {
    let mut v2: [u32; 8] = words(env, d.vd, i).map(u32::swap_bytes);
    let v3: [u32; 8] = words(env, d.vs2, i).map(u32::swap_bytes);
    let mut v1 = [0u32; 8];
    sm3c(&mut v1, &mut v2, &v3, uimm as u32);
    set_words(env, d.vd, i, &v1.map(u32::swap_bytes));
}

group_def!(VSM3ME_VV, "vsm3me_vv", 8, sm3me_group);
group_def!(VSM3C_VI, "vsm3c_vi", 8, sm3c_group);

// GHASH.

/// The GHASH multiplication of the bit-reversed `y` by the bit-reversed `h` in GF(2^128).
fn gf128_mul(y: [u64; 2], mut h: [u64; 2]) -> [u64; 2] {
    let mut z = [0u64; 2];
    for j in 0..128 {
        if (y[j / 64] >> (j % 64)) & 1 != 0 {
            z[0] ^= h[0];
            z[1] ^= h[1];
        }
        let reduce = (h[1] >> 63) & 1 != 0;
        h[1] = (h[1] << 1) | (h[0] >> 63);
        h[0] <<= 1;
        if reduce {
            h[0] ^= 0x87;
        }
    }
    z
}

/// `HELPER(vghsh_vv)`: `vd = (vd ^ vs1) * vs2`.
fn ghsh_group(env: &mut [u8], d: &Desc, i: usize, _: u64) {
    let get = |env: &[u8], reg| [vget(env, reg, i * 2, 3), vget(env, reg, i * 2 + 1, 3)];
    let y = get(env, d.vd);
    let x = get(env, d.vs1);
    let h = get(env, d.vs2).map(brev8);
    let s = [brev8(y[0] ^ x[0]), brev8(y[1] ^ x[1])];
    let z = gf128_mul(s, h);
    vset(env, d.vd, i * 2, 3, brev8(z[0]));
    vset(env, d.vd, i * 2 + 1, 3, brev8(z[1]));
}

/// `HELPER(vgmul_vv)`: `vd = vd * vs2`.
fn gmul_group(env: &mut [u8], d: &Desc, i: usize, _: u64) {
    let get = |env: &[u8], reg| [vget(env, reg, i * 2, 3), vget(env, reg, i * 2 + 1, 3)];
    let y = get(env, d.vd).map(brev8);
    let h = get(env, d.vs2).map(brev8);
    let z = gf128_mul(y, h);
    vset(env, d.vd, i * 2, 3, brev8(z[0]));
    vset(env, d.vd, i * 2 + 1, 3, brev8(z[1]));
}

group_def!(VGHSH_VV, "vghsh_vv", 4, ghsh_group);
group_def!(VGMUL_VV, "vgmul_vv", 4, gmul_group);

// SM4.

/// `sm4_sbox`.
pub(super) const SM4_SBOX: [u8; 256] = [
    0xd6, 0x90, 0xe9, 0xfe, 0xcc, 0xe1, 0x3d, 0xb7, 0x16, 0xb6, 0x14, 0xc2, 0x28, 0xfb, 0x2c, 0x05,
    0x2b, 0x67, 0x9a, 0x76, 0x2a, 0xbe, 0x04, 0xc3, 0xaa, 0x44, 0x13, 0x26, 0x49, 0x86, 0x06, 0x99,
    0x9c, 0x42, 0x50, 0xf4, 0x91, 0xef, 0x98, 0x7a, 0x33, 0x54, 0x0b, 0x43, 0xed, 0xcf, 0xac, 0x62,
    0xe4, 0xb3, 0x1c, 0xa9, 0xc9, 0x08, 0xe8, 0x95, 0x80, 0xdf, 0x94, 0xfa, 0x75, 0x8f, 0x3f, 0xa6,
    0x47, 0x07, 0xa7, 0xfc, 0xf3, 0x73, 0x17, 0xba, 0x83, 0x59, 0x3c, 0x19, 0xe6, 0x85, 0x4f, 0xa8,
    0x68, 0x6b, 0x81, 0xb2, 0x71, 0x64, 0xda, 0x8b, 0xf8, 0xeb, 0x0f, 0x4b, 0x70, 0x56, 0x9d, 0x35,
    0x1e, 0x24, 0x0e, 0x5e, 0x63, 0x58, 0xd1, 0xa2, 0x25, 0x22, 0x7c, 0x3b, 0x01, 0x21, 0x78, 0x87,
    0xd4, 0x00, 0x46, 0x57, 0x9f, 0xd3, 0x27, 0x52, 0x4c, 0x36, 0x02, 0xe7, 0xa0, 0xc4, 0xc8, 0x9e,
    0xea, 0xbf, 0x8a, 0xd2, 0x40, 0xc7, 0x38, 0xb5, 0xa3, 0xf7, 0xf2, 0xce, 0xf9, 0x61, 0x15, 0xa1,
    0xe0, 0xae, 0x5d, 0xa4, 0x9b, 0x34, 0x1a, 0x55, 0xad, 0x93, 0x32, 0x30, 0xf5, 0x8c, 0xb1, 0xe3,
    0x1d, 0xf6, 0xe2, 0x2e, 0x82, 0x66, 0xca, 0x60, 0xc0, 0x29, 0x23, 0xab, 0x0d, 0x53, 0x4e, 0x6f,
    0xd5, 0xdb, 0x37, 0x45, 0xde, 0xfd, 0x8e, 0x2f, 0x03, 0xff, 0x6a, 0x72, 0x6d, 0x6c, 0x5b, 0x51,
    0x8d, 0x1b, 0xaf, 0x92, 0xbb, 0xdd, 0xbc, 0x7f, 0x11, 0xd9, 0x5c, 0x41, 0x1f, 0x10, 0x5a, 0xd8,
    0x0a, 0xc1, 0x31, 0x88, 0xa5, 0xcd, 0x7b, 0xbd, 0x2d, 0x74, 0xd0, 0x12, 0xb8, 0xe5, 0xb4, 0xb0,
    0x89, 0x69, 0x97, 0x4a, 0x0c, 0x96, 0x77, 0x7e, 0x65, 0xb9, 0xf1, 0x09, 0xc5, 0x6e, 0xc6, 0x84,
    0x18, 0xf0, 0x7d, 0xec, 0x3a, 0xdc, 0x4d, 0x20, 0x79, 0xee, 0x5f, 0x3e, 0xd7, 0xcb, 0x39, 0x48,
];

/// `sm4_ck`.
const SM4_CK: [u32; 32] = [
    0x0007_0e15,
    0x1c23_2a31,
    0x383f_464d,
    0x545b_6269,
    0x7077_7e85,
    0x8c93_9aa1,
    0xa8af_b6bd,
    0xc4cb_d2d9,
    0xe0e7_eef5,
    0xfc03_0a11,
    0x181f_262d,
    0x343b_4249,
    0x5057_5e65,
    0x6c73_7a81,
    0x888f_969d,
    0xa4ab_b2b9,
    0xc0c7_ced5,
    0xdce3_eaf1,
    0xf8ff_060d,
    0x141b_2229,
    0x3037_3e45,
    0x4c53_5a61,
    0x686f_767d,
    0x848b_9299,
    0xa0a7_aeb5,
    0xbcc3_cad1,
    0xd8df_e6ed,
    0xf4fb_0209,
    0x1017_1e25,
    0x2c33_3a41,
    0x484f_565d,
    0x646b_7279,
];

/// `sm4_subword()`.
fn sm4_subword(w: u32) -> u32 {
    u32::from_le_bytes(w.to_le_bytes().map(|b| SM4_SBOX[b as usize]))
}

/// `HELPER(vsm4k_vi)`: the four SM4 round keys of round group `uimm & 7`.
fn sm4k_group(env: &mut [u8], d: &Desc, i: usize, uimm: u64) {
    let rnd = (uimm & 7) as usize;
    let rk: [u32; 4] = words(env, d.vs2, i);
    let mut tmp = [0u32; 8];
    tmp[..4].copy_from_slice(&rk);
    for j in 0..4 {
        let b = tmp[j + 1] ^ tmp[j + 2] ^ tmp[j + 3] ^ SM4_CK[rnd * 4 + j];
        let s = sm4_subword(b);
        tmp[j + 4] = tmp[j] ^ (s ^ s.rotate_left(13) ^ s.rotate_left(23));
    }
    set_words(env, d.vd, i, &[tmp[4], tmp[5], tmp[6], tmp[7]]);
}

/// `do_sm4_round()`: four SM4 rounds of the state `buf[0..4]` into `buf[4..8]`.
fn sm4_round(rk: &[u32; 4], buf: &mut [u32; 8]) {
    for j in 4..8 {
        let b = buf[j - 3] ^ buf[j - 2] ^ buf[j - 1] ^ rk[j - 4];
        let s = sm4_subword(b);
        buf[j] = buf[j - 4]
            ^ (s ^ s.rotate_left(2) ^ s.rotate_left(10) ^ s.rotate_left(18) ^ s.rotate_left(24));
    }
}

/// `HELPER(vsm4r_vv)` (`vs = false`) and `HELPER(vsm4r_vs)` (`vs = true`).
fn sm4r_group(env: &mut [u8], d: &Desc, i: usize, vs: bool) {
    let rk: [u32; 4] = words(env, d.vs2, if vs { 0 } else { i });
    let st: [u32; 4] = words(env, d.vd, i);
    let mut buf = [0u32; 8];
    buf[..4].copy_from_slice(&st);
    sm4_round(&rk, &mut buf);
    set_words(env, d.vd, i, &[buf[4], buf[5], buf[6], buf[7]]);
}

group_def!(VSM4K_VI, "vsm4k_vi", 4, sm4k_group);
group_def!(VSM4R_VV, "vsm4r_vv", 4, |e, d, i, _| sm4r_group(e, d, i, false));
group_def!(VSM4R_VS, "vsm4r_vs", 4, |e, d, i, _| sm4r_group(e, d, i, true));

/// The helpers of this module.
pub(crate) const ALL: &[Def] = &[
    VCLMUL, VCLMULH, VROR, VROL, VANDN, VBREV8, VREV8, VBREV, VCLZ, VCTZ, VCPOP_V, VWSLL,
    VAESEF_VV, VAESEF_VS, VAESDF_VV, VAESDF_VS, VAESEM_VV, VAESEM_VS, VAESDM_VV, VAESDM_VS,
    VAESZ_VS, VAESKF1_VI, VAESKF2_VI, VSHA2MS_VV, VSHA2CH_VV, VSHA2CL_VV, VSM3ME_VV, VSM3C_VI,
    VGHSH_VV, VGMUL_VV, VSM4K_VI, VSM4R_VV, VSM4R_VS,
];

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use super::*;
    use crate::cpu::{ENV_SIZE, VL, VSTART, VTYPE};
    use crate::tcg::{ld64, st64};

    /// An `env` with SEW `8 << sew`, LMUL `1 << lmul` and `vl`.
    fn env(sew: u32, lmul: i32, vl: usize) -> Vec<u8> {
        let mut e = vec![0u8; ENV_SIZE];
        st64(&mut e, VTYPE, u64::from(sew) << 3 | (lmul as u64 & 7));
        st64(&mut e, VL, vl as u64);
        e
    }

    /// The descriptor of an unmasked instruction.
    fn desc(vd: u32, vs1: u32, vs2: u32, esz: u32, lmul: i32) -> Desc {
        Desc { vm: true, lmul, vd, vs1, vs2, esz, ..Desc::default() }
    }

    fn put(env: &mut [u8], reg: u32, log2: u32, v: &[u64]) {
        for (i, x) in v.iter().enumerate() {
            vset(env, reg, i, log2, *x);
        }
    }

    fn get(env: &[u8], reg: u32, log2: u32, n: usize) -> Vec<u64> {
        (0..n).map(|i| vget(env, reg, i, log2)).collect()
    }

    fn put_bytes(env: &mut [u8], reg: u32, b: &[u8]) {
        for (i, x) in b.iter().enumerate() {
            env[voff(reg, i)] = *x;
        }
    }

    fn get_bytes(env: &[u8], reg: u32, n: usize) -> Vec<u8> {
        (0..n).map(|i| env[voff(reg, i)]).collect()
    }

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    fn to_hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    /// Run the element-wise helper body of `op` as the helper would.
    fn run_vop(env: &mut [u8], d: &Desc, scalar: u64, shape: Shape, op: fn(u64, u64, u32) -> u64) {
        super::super::vector::vop(env, &[0, u64::from(d.encode()), scalar], shape, op);
    }

    #[test]
    fn element_ops() {
        assert_eq!(clmul64(3, 3), 5);
        assert_eq!(clmul64(u64::MAX, 2), u64::MAX << 1);
        assert_eq!(clmulh64(2, 1 << 63), 1);
        assert_eq!(clmulh64(u64::MAX, u64::MAX), 0x5555_5555_5555_5555);
        assert_eq!(rol(0x81, 1, 0), 0x03);
        assert_eq!(rol(0x81, 9, 0), 0x03);
        assert_eq!(ror(0x01, 1, 0), 0x80);
        assert_eq!(ror(0x1234, 0, 1), 0x1234);
        assert_eq!(ror(1, 65, 3), 1 << 63);
        assert_eq!(rol(0x8000_0001, 4, 2), 0x18);
        assert_eq!(brev8(0x0102_0380), 0x8040_c001);
        assert_eq!(rev8(0x1234, 1), 0x3412);
        assert_eq!(rev8(0x12, 0), 0x12);
        assert_eq!(rev8(0x0102_0304_0506_0708, 3), 0x0807_0605_0403_0201);
        assert_eq!(brev(1, 0), 0x80);
        assert_eq!(brev(1, 2), 0x8000_0000);
        assert_eq!(clz(1, 0), 7);
        assert_eq!(clz(0, 1), 16);
        assert_eq!(clz(1 << 63, 3), 0);
        assert_eq!(ctz(8, 2), 3);
        assert_eq!(ctz(0, 3), 64);
        assert_eq!(wsll(0xff, 12, 0), 0xff000);
        assert_eq!(wsll(0xff, 17, 0), 0x1fe);
        assert_eq!(wsll(1, 63, 2), 1 << 63);
    }

    #[test]
    fn element_helpers() {
        // vclmulh.vv at SEW 64, two elements.
        let mut e = env(3, 0, 2);
        put(&mut e, 1, 3, &[1 << 63, u64::MAX]);
        put(&mut e, 2, 3, &[2, u64::MAX]);
        run_vop(&mut e, &desc(3, 1, 2, 3, 0), 0, Shape::Single, |s2, s1, _| clmulh64(s2, s1));
        assert_eq!(get(&e, 3, 3, 2), [1, 0x5555_5555_5555_5555]);

        // vrol.vx at SEW 8: the scalar 9 rotates by 1.
        let mut e = env(0, 0, 3);
        put(&mut e, 2, 0, &[0x81, 0x01, 0xf0]);
        run_vop(&mut e, &Desc { scalar: true, ..desc(3, 0, 2, 0, 0) }, 9, Shape::Single, rol);
        assert_eq!(get(&e, 3, 0, 3), [0x03, 0x02, 0xe1]);

        // vandn.vv at SEW 16.
        let mut e = env(1, 0, 2);
        put(&mut e, 1, 1, &[0x00ff, 0xf0f0]);
        put(&mut e, 2, 1, &[0x1234, 0xffff]);
        run_vop(&mut e, &desc(3, 1, 2, 1, 0), 0, Shape::Single, |s2, s1, _| s2 & !s1);
        assert_eq!(get(&e, 3, 1, 2), [0x1200, 0x0f0f]);

        // vcpop.v and vctz.v at SEW 32.
        let mut e = env(2, 0, 2);
        put(&mut e, 2, 2, &[0xffff_ffff, 0]);
        run_vop(&mut e, &desc(3, 0, 2, 2, 0), 0, Shape::Single, |s2, _, _| {
            u64::from(s2.count_ones())
        });
        assert_eq!(get(&e, 3, 2, 2), [32, 0]);
        run_vop(&mut e, &desc(3, 0, 2, 2, 0), 0, Shape::Single, |s2, _, sew| ctz(s2, sew));
        assert_eq!(get(&e, 3, 2, 2), [0, 32]);

        // vwsll.vx at SEW 8: 16-bit results, the shift taken modulo 16.
        let mut e = env(0, 0, 2);
        put(&mut e, 2, 0, &[0xff, 0x81]);
        run_vop(&mut e, &Desc { scalar: true, ..desc(4, 0, 2, 0, 0) }, 0x1c, Shape::Widen, wsll);
        assert_eq!(get(&e, 4, 1, 2), [0xf000, 0x1000]);
    }

    /// The AES-128 round keys of `key` with `vaeskf1`.
    fn aes128_keys(key: &[u8]) -> Vec<[u8; 16]> {
        let mut e = env(2, 0, 4);
        let d = desc(3, 0, 2, 2, 0);
        let mut keys = vec![<[u8; 16]>::try_from(key).unwrap()];
        for i in 1..=10 {
            put_bytes(&mut e, 2, &keys[i - 1]);
            group(&mut e, &d, 4, i as u64, aeskf1);
            keys.push(block(&e, 3, 0));
        }
        keys
    }

    /// The AES-256 round keys of `key` with `vaeskf2`.
    fn aes256_keys(key: &[u8]) -> Vec<[u8; 16]> {
        let mut e = env(2, 0, 4);
        let d = desc(3, 0, 2, 2, 0);
        let mut keys = vec![
            <[u8; 16]>::try_from(&key[..16]).unwrap(),
            <[u8; 16]>::try_from(&key[16..]).unwrap(),
        ];
        for i in 2..=14 {
            put_bytes(&mut e, 3, &keys[i - 2]);
            put_bytes(&mut e, 2, &keys[i - 1]);
            group(&mut e, &d, 4, i as u64, aeskf2);
            keys.push(block(&e, 3, 0));
        }
        keys
    }

    /// One AES round on `v3` with the key in `v2`.
    fn aes_step(e: &mut [u8], key: &[u8; 16], op: GroupOp) {
        put_bytes(e, 2, key);
        group(e, &desc(3, 0, 2, 2, 0), 4, 0, op);
    }

    fn aes_encrypt(keys: &[[u8; 16]], pt: &[u8]) -> Vec<u8> {
        let n = keys.len() - 1;
        let mut e = env(2, 0, 4);
        put_bytes(&mut e, 3, pt);
        aes_step(&mut e, &keys[0], |e, d, i, _| aes_group(e, d, i, AesRound::Zero, true));
        for k in &keys[1..n] {
            aes_step(&mut e, k, |e, d, i, _| aes_group(e, d, i, AesRound::EncMiddle, false));
        }
        aes_step(&mut e, &keys[n], |e, d, i, _| aes_group(e, d, i, AesRound::EncFinal, true));
        get_bytes(&e, 3, 16)
    }

    fn aes_decrypt(keys: &[[u8; 16]], ct: &[u8]) -> Vec<u8> {
        let n = keys.len() - 1;
        let mut e = env(2, 0, 4);
        put_bytes(&mut e, 3, ct);
        aes_step(&mut e, &keys[n], |e, d, i, _| aes_group(e, d, i, AesRound::Zero, true));
        for k in keys[1..n].iter().rev() {
            aes_step(&mut e, k, |e, d, i, _| aes_group(e, d, i, AesRound::DecMiddle, true));
        }
        aes_step(&mut e, &keys[0], |e, d, i, _| aes_group(e, d, i, AesRound::DecFinal, false));
        get_bytes(&e, 3, 16)
    }

    #[test]
    fn aes128_fips197_c1() {
        let keys = aes128_keys(&hex("000102030405060708090a0b0c0d0e0f"));
        assert_eq!(to_hex(&keys[10]), "13111d7fe3944a17f307a78b4d2b30c5");
        let pt = hex("00112233445566778899aabbccddeeff");
        let ct = aes_encrypt(&keys, &pt);
        assert_eq!(to_hex(&ct), "69c4e0d86a7b0430d8cdb78070b4c55a");
        assert_eq!(aes_decrypt(&keys, &ct), pt);
    }

    #[test]
    fn aes256_fips197_c3() {
        let key = hex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f");
        let keys = aes256_keys(&key);
        let pt = hex("00112233445566778899aabbccddeeff");
        let ct = aes_encrypt(&keys, &pt);
        assert_eq!(to_hex(&ct), "8ea2b7ca516745bfeafc49904b496089");
        assert_eq!(aes_decrypt(&keys, &ct), pt);
    }

    #[test]
    fn aeskf_round_number_wraps() {
        // vaeskf1 treats rounds 0 and 11 to 15 as round ^ 8, vaeskf2 rounds 0, 1 and 15.
        let key = hex("000102030405060708090a0b0c0d0e0f");
        let mut e = env(2, 0, 4);
        let d = desc(3, 0, 2, 2, 0);
        let out = |e: &mut Vec<u8>, uimm: u64, op: GroupOp| {
            put_bytes(e, 2, &key);
            put_bytes(e, 3, &key);
            group(e, &d, 4, uimm, op);
            get_bytes(e, 3, 16)
        };
        assert_eq!(out(&mut e, 0, aeskf1), out(&mut e, 8, aeskf1));
        assert_eq!(out(&mut e, 11, aeskf1), out(&mut e, 3, aeskf1));
        assert_eq!(out(&mut e, 31, aeskf1), out(&mut e, 7, aeskf1));
        assert_eq!(out(&mut e, 1, aeskf2), out(&mut e, 9, aeskf2));
        assert_eq!(out(&mut e, 15, aeskf2), out(&mut e, 7, aeskf2));
    }

    #[test]
    fn group_loop_vstart_and_tail() {
        // Two element groups, vstart at the second: the first is left alone.
        let key = [0x5au8; 16];
        let mut e = env(2, 1, 8);
        st64(&mut e, VSTART, 4);
        put_bytes(&mut e, 2, &key);
        put_bytes(&mut e, 4, &[1u8; 32]);
        let d = Desc { vta: true, ..desc(4, 0, 2, 2, 1) };
        group(&mut e, &d, 4, 0, |e, d, i, _| aes_group(e, d, i, AesRound::Zero, true));
        assert_eq!(ld64(&e, VSTART), 0);
        assert_eq!(get_bytes(&e, 4, 16), [1u8; 16]);
        assert_eq!(get_bytes(&e, 4, 32)[16..], [0x5bu8; 16]);

        // One element group of two registers: the tail is set to ones when tail agnostic.
        st64(&mut e, VL, 4);
        group(&mut e, &d, 4, 0, |e, d, i, _| aes_group(e, d, i, AesRound::Zero, true));
        assert_eq!(get_bytes(&e, 4, 16), [0x5bu8; 16]);
        assert_eq!(get_bytes(&e, 4, 32)[16..], [0xffu8; 16]);

        // vl 8: both groups get the key of group 0 for the .vs form.
        let mut e = env(2, 1, 8);
        put_bytes(&mut e, 2, &[0x5a; 16]);
        put_bytes(&mut e, 3, &[0xa5; 16]);
        put_bytes(&mut e, 4, &[0x0f; 32]);
        group(&mut e, &d, 4, 0, |e, d, i, _| aes_group(e, d, i, AesRound::Zero, true));
        assert_eq!(get_bytes(&e, 4, 32), [0x55u8; 32]);
        group(&mut e, &d, 4, 0, |e, d, i, _| aes_group(e, d, i, AesRound::Zero, false));
        let mut want = [0x0fu8; 32];
        want[16..].fill(0x55 ^ 0xa5);
        assert_eq!(get_bytes(&e, 4, 32), want);

        // vstart past vl: nothing but vstart cleared.
        let mut e = env(2, 0, 4);
        st64(&mut e, VSTART, 4);
        put_bytes(&mut e, 3, &[7; 16]);
        group(&mut e, &desc(3, 0, 2, 2, 0), 4, 0, |e, d, i, _| {
            aes_group(e, d, i, AesRound::Zero, true)
        });
        assert_eq!(get_bytes(&e, 3, 16), [7u8; 16]);
        assert_eq!(ld64(&e, VSTART), 0);
    }

    // The SHA-2 constants, from the roots of the primes.

    fn mul(a: &[u64], b: &[u64]) -> Vec<u64> {
        let mut r = vec![0u64; a.len() + b.len()];
        for (i, &x) in a.iter().enumerate() {
            let mut carry = 0u128;
            for (j, &y) in b.iter().enumerate() {
                let t = u128::from(x) * u128::from(y) + u128::from(r[i + j]) + carry;
                r[i + j] = t as u64;
                carry = t >> 64;
            }
            r[i + b.len()] = carry as u64;
        }
        r
    }

    fn cmp(a: &[u64], b: &[u64]) -> Ordering {
        for i in (0..a.len().max(b.len())).rev() {
            let x = a.get(i).copied().unwrap_or(0);
            let y = b.get(i).copied().unwrap_or(0);
            if x != y {
                return x.cmp(&y);
            }
        }
        Ordering::Equal
    }

    /// The first `bits` fractional bits of the `k`th root of `p`.
    fn root_frac(p: u64, k: u32, bits: u32) -> u64 {
        let shift = (k * bits) as usize;
        let mut n = vec![0u64; shift / 64 + 2];
        n[shift / 64] = p << (shift % 64);
        if shift % 64 != 0 {
            n[shift / 64 + 1] = p >> (64 - shift % 64);
        }
        let (mut lo, mut hi) = (0u128, 1u128 << (bits + 4));
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            let m = [mid as u64, (mid >> 64) as u64];
            let mut pow = m.to_vec();
            for _ in 1..k {
                pow = mul(&pow, &m);
            }
            if cmp(&pow, &n) == Ordering::Greater {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let mask = if bits == 64 { u64::MAX } else { (1 << bits) - 1 };
        lo as u64 & mask
    }

    fn primes(n: usize) -> Vec<u64> {
        let mut v = Vec::new();
        let mut x = 2;
        while v.len() < n {
            if v.iter().all(|p| x % p != 0) {
                v.push(x);
            }
            x += 1;
        }
        v
    }

    fn sha2c_cl(e: &mut [u8], d: &Desc, i: usize, _: u64) {
        sha2c_group(e, d, i, 0);
    }

    fn sha2c_ch(e: &mut [u8], d: &Desc, i: usize, _: u64) {
        sha2c_group(e, d, i, 2);
    }

    /// The hash of the padded one block message `msg` with `vsha2ms`, `vsha2cl` and
    /// `vsha2ch` at SEW `8 << sew`.
    fn sha2_block(sew: u32, msg: &[u64; 16]) -> Vec<u64> {
        let bits = 8 << sew;
        let rounds = if sew == 2 { 64 } else { 80 };
        let p = primes(rounds);
        let k: Vec<u64> = p.iter().map(|&p| root_frac(p, 3, bits)).collect();
        let h0: Vec<u64> = p[..8].iter().map(|&p| root_frac(p, 2, bits)).collect();
        if sew == 2 {
            assert_eq!((k[0], k[63], h0[0]), (0x428a_2f98, 0xc671_78f2, 0x6a09_e667));
        } else {
            assert_eq!(k[0], 0x428a_2f98_d728_ae22);
            assert_eq!(h0[7], 0x5be0_cd19_137e_2179);
        }
        let mask = if sew == 2 { 0xffff_ffff } else { u64::MAX };

        // v4 is vd, v6 vs1 and v2 vs2, four elements of SEW, so LMUL 2.
        let mut e = env(sew, 1, 4);
        let d = desc(4, 6, 2, sew, 1);
        let mut w = msg.to_vec();
        for t in 0..rounds / 4 - 4 {
            let b = 4 * t;
            put(&mut e, 4, sew, &w[b..b + 4]);
            put(&mut e, 2, sew, &[w[b + 4], w[b + 9], w[b + 10], w[b + 11]]);
            put(&mut e, 6, sew, &w[b + 12..b + 16]);
            group(&mut e, &d, 4, 0, sha2ms_group);
            w.extend(get(&e, 4, sew, 4));
        }
        let mut abef = vec![h0[5], h0[4], h0[1], h0[0]];
        let mut cdgh = vec![h0[7], h0[6], h0[3], h0[2]];
        for t in 0..rounds / 4 {
            let wk: Vec<u64> = (4 * t..4 * t + 4).map(|j| w[j].wrapping_add(k[j]) & mask).collect();
            for op in [sha2c_cl as GroupOp, sha2c_ch] {
                put(&mut e, 6, sew, &wk);
                put(&mut e, 2, sew, &abef);
                put(&mut e, 4, sew, &cdgh);
                group(&mut e, &d, 4, 0, op);
                cdgh = abef;
                abef = get(&e, 4, sew, 4);
            }
        }
        let out = [abef[3], abef[2], cdgh[3], cdgh[2], abef[1], abef[0], cdgh[1], cdgh[0]];
        out.iter().zip(&h0).map(|(x, h)| x.wrapping_add(*h) & mask).collect()
    }

    #[test]
    fn sha256_abc() {
        let mut msg = [0u64; 16];
        msg[0] = 0x6162_6380;
        msg[15] = 24;
        let h = sha2_block(2, &msg);
        let s: String = h.iter().map(|x| format!("{x:08x}")).collect();
        assert_eq!(s, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn sha512_abc() {
        let mut msg = [0u64; 16];
        msg[0] = 0x6162_6380_0000_0000;
        msg[15] = 24;
        let h = sha2_block(3, &msg);
        let s: String = h.iter().map(|x| format!("{x:016x}")).collect();
        assert_eq!(
            s,
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a\
             2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
        );
    }

    #[test]
    fn sm4_sample() {
        // GB/T 32907 example 1. The elements are the words of the standard.
        let mk = [0x0123_4567u64, 0x89ab_cdef, 0xfedc_ba98, 0x7654_3210];
        let fk = [0xa3b1_bac6u64, 0x56aa_3350, 0x677d_9197, 0xb270_22dc];
        let mut e = env(2, 0, 4);
        let d = desc(3, 0, 2, 2, 0);
        let mut k: Vec<u64> = mk.iter().zip(&fk).map(|(m, f)| m ^ f).collect();
        let mut rks = Vec::new();
        for rnd in 0..8 {
            put(&mut e, 2, 2, &k);
            group(&mut e, &d, 4, rnd, sm4k_group);
            k = get(&e, 3, 2, 4);
            rks.push(k.clone());
        }
        assert_eq!(rks[0][0], 0xf121_86f9);
        assert_eq!(rks[7][3], 0x9124_a012);
        put(&mut e, 3, 2, &mk);
        for (n, rk) in rks.iter().enumerate() {
            put(&mut e, 2, 2, rk);
            let vs = n % 2 == 0;
            let op: GroupOp = if vs {
                |e, d, i, _| sm4r_group(e, d, i, true)
            } else {
                |e, d, i, _| sm4r_group(e, d, i, false)
            };
            group(&mut e, &d, 4, 0, op);
        }
        let x = get(&e, 3, 2, 4);
        let ct = [x[3], x[2], x[1], x[0]];
        assert_eq!(ct, [0x681e_df34, 0xd206_965e, 0x86b3_e94f, 0x536e_4246]);
    }

    #[test]
    fn sm3_abc() {
        // The padded block of "abc" in memory order, the way vle32.v loads it.
        let mut block = [0u8; 64];
        block[..4].copy_from_slice(b"abc\x80");
        block[63] = 24;
        let iv: [u32; 8] = [
            0x7380_166f,
            0x4914_b2b9,
            0x1724_42d7,
            0xda8a_0600,
            0xa96f_30bc,
            0x1631_38aa,
            0xe38d_ee4d,
            0xb0fb_0e4e,
        ];
        // Eight elements a group: LMUL 2.
        let mut e = env(2, 1, 8);
        let mut w: Vec<u8> = block.to_vec();
        let me = desc(4, 6, 2, 2, 1);
        for t in 0..7 {
            put_bytes(&mut e, 6, &w[32 * t..32 * t + 32]);
            put_bytes(&mut e, 2, &w[32 * t + 32..32 * t + 64]);
            group(&mut e, &me, 8, 0, sm3me_group);
            w.extend(get_bytes(&e, 4, 32));
        }
        // W16 of "abc", P1(0x61626380), in memory order.
        assert_eq!(&w[64..68], &[0x90, 0x92, 0xe2, 0x00]);
        let mut state = Vec::new();
        for x in iv {
            state.extend(x.to_be_bytes());
        }
        put_bytes(&mut e, 4, &state);
        let c = desc(4, 0, 2, 2, 1);
        for r in 0..32 {
            put_bytes(&mut e, 2, &w[8 * r..8 * r + 32]);
            group(&mut e, &c, 8, r as u64, sm3c_group);
        }
        let out = get_bytes(&e, 4, 32);
        let v: Vec<u8> = out.iter().zip(&state).map(|(x, s)| x ^ s).collect();
        assert_eq!(to_hex(&v), "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0");
    }

    /// The GCM multiplication of the specification, bit 0 the leftmost.
    fn gf128_ref(x: &[u8], y: &[u8]) -> Vec<u8> {
        let x = u128::from_be_bytes(x.try_into().unwrap());
        let mut v = u128::from_be_bytes(y.try_into().unwrap());
        let mut z = 0u128;
        for i in 0..128 {
            if (x >> (127 - i)) & 1 != 0 {
                z ^= v;
            }
            v = if v & 1 != 0 { (v >> 1) ^ (0xe1 << 120) } else { v >> 1 };
        }
        z.to_be_bytes().to_vec()
    }

    #[test]
    fn ghash_gcm_test_case_2() {
        // AES-128 GCM with the zero key, IV and a 16 byte zero plaintext.
        let h = hex("66e94bd4ef8a2c3b884cfa59ca342b2e");
        let c = hex("0388dace60b6a392f328c2b971b2fe78");
        let mut len = [0u8; 16];
        len[15] = 0x80;
        let ek_y0 = hex("58e2fccefa7e3061367f1d57a4e7455a");
        let mut e = env(2, 0, 4);
        let d = desc(3, 1, 2, 2, 0);
        put_bytes(&mut e, 2, &h);
        put_bytes(&mut e, 1, &c);
        group(&mut e, &d, 4, 0, ghsh_group);
        let y1 = get_bytes(&e, 3, 16);
        assert_eq!(y1, gf128_ref(&c, &h));
        put_bytes(&mut e, 1, &len);
        group(&mut e, &d, 4, 0, ghsh_group);
        let y2 = get_bytes(&e, 3, 16);
        let tag: Vec<u8> = y2.iter().zip(&ek_y0).map(|(a, b)| a ^ b).collect();
        assert_eq!(to_hex(&tag), "ab6e47d42cec13bdf53a67b21257bddf");

        // vgmul of (Y1 ^ len) and H is the same second step.
        let x: Vec<u8> = y1.iter().zip(&len).map(|(a, b)| a ^ b).collect();
        put_bytes(&mut e, 3, &x);
        group(&mut e, &d, 4, 0, gmul_group);
        assert_eq!(get_bytes(&e, 3, 16), y2);
        assert_eq!(y2, gf128_ref(&x, &h));
    }
}
