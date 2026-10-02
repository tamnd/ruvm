// SPDX-License-Identifier: GPL-2.0-or-later

//! The AdvSIMD integer helpers: the A64 parts of QEMU's `vec_helper.c`, `neon_helper.c` and
//! the integer parts of `helper-a64.c`, with the inline functions of `vec_internal.h`.
//!
//! QEMU has one helper per operation and element size, and passes pointers to the vector
//! registers. This port has one helper, `a64_neon`, that takes the register numbers and a
//! descriptor ([`Desc`]) naming the operation, the element size, the operation size and an
//! immediate. It reads the source registers out of `env`, computes the whole result, clears
//! the bytes above the operation size as `clear_tail()` does, and writes the destination
//! back. Saturation sets `vfp.qc`, so FPSR.QC reads as 1, just as QEMU's helpers do.
//!
//! The results are those of the QEMU helpers bit for bit: the shifts by register follow
//! `do_sqrshl_bhs()`, `do_uqrshl_bhs()` and their 64-bit forms, the doubling multiplies
//! follow `do_sqrdmulh_h()` and `do_sqrdmulh_s()`, and the saturating accumulations follow
//! `neon_addl_saturate_s32()` and `_s64()`.

use ruvm_jit_core::HelperType::{I32, Ptr, Void};
use ruvm_jit_interp::{HelperEnv, Unwind};

use super::helpers::{Def, def};
use crate::cpu::{QC, vreg_off};

/// A vector register image, V0 to V31 as QEMU lays out `zregs` (little endian).
pub(crate) type V = [u8; 16];

/// Read Vn out of `env`.
pub(crate) fn vload(env: &[u8], n: u32) -> V {
    let o = vreg_off(n as usize & 31);
    let mut v = [0; 16];
    v.copy_from_slice(&env[o..o + 16]);
    v
}

/// Write Vd into `env`, clearing the rest of the SVE register Zd as `clear_tail()` does up
/// to the vector length (the bytes above it are always zero).
pub(crate) fn vstore(env: &mut [u8], d: u32, v: &V) {
    let o = vreg_off(d as usize & 31);
    env[o..o + 16].copy_from_slice(v);
    env[o + 16..o + crate::cpu::ZREG_SIZE].fill(0);
}

/// Element `i` of size `1 << esz` bytes, zero extended.
pub(crate) fn get(v: &V, esz: u32, i: usize) -> u64 {
    let n = 1usize << esz;
    let mut b = [0u8; 8];
    b[..n].copy_from_slice(&v[i * n..i * n + n]);
    u64::from_le_bytes(b)
}

/// Element `i` of size `1 << esz` bytes, sign extended.
pub(crate) fn gets(v: &V, esz: u32, i: usize) -> i64 {
    sext(get(v, esz, i), esz)
}

/// Set element `i` of size `1 << esz` bytes to the low bits of `x`.
pub(crate) fn set(v: &mut V, esz: u32, i: usize, x: u64) {
    let n = 1usize << esz;
    v[i * n..i * n + n].copy_from_slice(&x.to_le_bytes()[..n]);
}

/// Sign extend the low `8 << esz` bits of `x`.
pub(crate) fn sext(x: u64, esz: u32) -> i64 {
    let sh = 64 - (8u32 << esz);
    ((x << sh) as i64) >> sh
}

/// All ones in the low `8 << esz` bits.
pub(crate) fn mask(esz: u32) -> u64 {
    u64::MAX >> (64 - (8u32 << esz))
}

/// `clear_tail()`: zero the bytes of `v` from `oprsz` on.
pub(crate) fn clear_tail(v: &mut V, oprsz: usize) {
    for b in &mut v[oprsz.min(16)..] {
        *b = 0;
    }
}

/// The register numbers of a helper call: Rd, Rn, Rm and Ra, five bits each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Regs {
    pub(crate) d: u32,
    pub(crate) n: u32,
    pub(crate) m: u32,
    pub(crate) a: u32,
}

impl Regs {
    /// Pack the four register numbers.
    pub(crate) fn pack(d: i32, n: i32, m: i32, a: i32) -> u32 {
        (d as u32 & 31) | (n as u32 & 31) << 5 | (m as u32 & 31) << 10 | (a as u32 & 31) << 15
    }

    pub(crate) fn unpack(x: u64) -> Regs {
        let x = x as u32;
        Regs { d: x & 31, n: (x >> 5) & 31, m: (x >> 10) & 31, a: (x >> 15) & 31 }
    }
}

/// The descriptor of a SIMD helper call, QEMU's `simd_desc()` cut down to what A64 needs.
///
/// Bits 0 to 4 hold the operation size in bytes (1 to 16), bits 5 and 6 the element size,
/// bit 7 selects the half precision float status, bits 8 to 15 the operation, bits 16 to 23
/// an immediate (a shift, an element index or a table length), bit 24 says that the upper
/// half of the sources is used or written (the "2" forms), bit 25 that the operation is
/// indexed, and bits 26 to 29 hold a rounding mode, 15 for the one in FPCR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Desc {
    pub(crate) oprsz: usize,
    pub(crate) esz: u32,
    pub(crate) f16: bool,
    pub(crate) op: u32,
    pub(crate) imm: u32,
    pub(crate) hi: bool,
    pub(crate) idx: bool,
    pub(crate) rmode: u32,
}

impl Desc {
    /// A descriptor with only the operation, element size and operation size set.
    pub(crate) fn new(op: u32, esz: i32, oprsz: usize) -> Desc {
        Desc {
            oprsz,
            esz: esz as u32,
            f16: false,
            op,
            imm: 0,
            hi: false,
            idx: false,
            rmode: RMODE_FPCR,
        }
    }

    pub(crate) fn imm(mut self, imm: i32) -> Desc {
        self.imm = imm as u32 & 0xff;
        self
    }

    pub(crate) fn hi(mut self, hi: bool) -> Desc {
        self.hi = hi;
        self
    }

    pub(crate) fn indexed(mut self, idx: i32) -> Desc {
        self.idx = true;
        self.imm = idx as u32 & 0xff;
        self
    }

    pub(crate) fn rmode(mut self, rmode: u32) -> Desc {
        self.rmode = rmode & 15;
        self
    }

    pub(crate) fn pack(self) -> u32 {
        debug_assert!(self.oprsz >= 1 && self.oprsz <= 16);
        (self.oprsz as u32 & 31)
            | (self.esz & 3) << 5
            | u32::from(self.f16) << 7
            | (self.op & 0xff) << 8
            | (self.imm & 0xff) << 16
            | u32::from(self.hi) << 24
            | u32::from(self.idx) << 25
            | (self.rmode & 15) << 26
    }

    pub(crate) fn unpack(x: u64) -> Desc {
        let x = x as u32;
        Desc {
            oprsz: (x & 31) as usize,
            esz: (x >> 5) & 3,
            f16: x & (1 << 7) != 0,
            op: (x >> 8) & 0xff,
            imm: (x >> 16) & 0xff,
            hi: x & (1 << 24) != 0,
            idx: x & (1 << 25) != 0,
            rmode: (x >> 26) & 15,
        }
    }
}

/// The rounding mode field value that means "the mode in FPCR".
pub(crate) const RMODE_FPCR: u32 = 15;

/// Set `vfp.qc` to show that a saturation happened.
pub(crate) fn set_qc(env: &mut [u8]) {
    env[QC..QC + 8].copy_from_slice(&1u64.to_le_bytes());
}

/// The integer operations of `a64_neon`.
#[allow(missing_docs)]
pub(crate) mod op {
    // Three registers, same element size: d[i] = f(n[i], m[i]) (and d[i] for the
    // accumulating forms).
    pub(crate) const ADD: u32 = 1;
    pub(crate) const SUB: u32 = 2;
    pub(crate) const MUL: u32 = 3;
    pub(crate) const MLA: u32 = 4;
    pub(crate) const MLS: u32 = 5;
    pub(crate) const SMAX: u32 = 6;
    pub(crate) const SMIN: u32 = 7;
    pub(crate) const UMAX: u32 = 8;
    pub(crate) const UMIN: u32 = 9;
    pub(crate) const SABD: u32 = 10;
    pub(crate) const UABD: u32 = 11;
    pub(crate) const SABA: u32 = 12;
    pub(crate) const UABA: u32 = 13;
    pub(crate) const SHADD: u32 = 14;
    pub(crate) const UHADD: u32 = 15;
    pub(crate) const SRHADD: u32 = 16;
    pub(crate) const URHADD: u32 = 17;
    pub(crate) const SHSUB: u32 = 18;
    pub(crate) const UHSUB: u32 = 19;
    pub(crate) const SQADD: u32 = 20;
    pub(crate) const UQADD: u32 = 21;
    pub(crate) const SQSUB: u32 = 22;
    pub(crate) const UQSUB: u32 = 23;
    pub(crate) const SUQADD: u32 = 24;
    pub(crate) const USQADD: u32 = 25;
    pub(crate) const SSHL: u32 = 26;
    pub(crate) const USHL: u32 = 27;
    pub(crate) const SRSHL: u32 = 28;
    pub(crate) const URSHL: u32 = 29;
    pub(crate) const SQSHL: u32 = 30;
    pub(crate) const UQSHL: u32 = 31;
    pub(crate) const SQRSHL: u32 = 32;
    pub(crate) const UQRSHL: u32 = 33;
    pub(crate) const PMUL: u32 = 34;
    pub(crate) const SQDMULH: u32 = 35;
    pub(crate) const SQRDMULH: u32 = 36;
    pub(crate) const SQRDMLAH: u32 = 37;
    pub(crate) const SQRDMLSH: u32 = 38;
    pub(crate) const CMTST: u32 = 39;
    pub(crate) const CMEQ: u32 = 40;
    pub(crate) const CMGE: u32 = 41;
    pub(crate) const CMGT: u32 = 42;
    pub(crate) const CMHI: u32 = 43;
    pub(crate) const CMHS: u32 = 44;
    pub(crate) const AND: u32 = 45;
    pub(crate) const BIC: u32 = 46;
    pub(crate) const ORR: u32 = 47;
    pub(crate) const ORN: u32 = 48;
    pub(crate) const EOR: u32 = 49;
    pub(crate) const BSL: u32 = 50;
    pub(crate) const BIT: u32 = 51;
    pub(crate) const BIF: u32 = 52;
    // Pairwise: the pairs of n, then the pairs of m.
    pub(crate) const ADDP: u32 = 53;
    pub(crate) const SMAXP: u32 = 54;
    pub(crate) const SMINP: u32 = 55;
    pub(crate) const UMAXP: u32 = 56;
    pub(crate) const UMINP: u32 = 57;
    // Two registers.
    pub(crate) const ABS: u32 = 60;
    pub(crate) const NEG: u32 = 61;
    pub(crate) const NOT: u32 = 62;
    pub(crate) const CLS: u32 = 63;
    pub(crate) const CLZ: u32 = 64;
    pub(crate) const CNT: u32 = 65;
    pub(crate) const RBIT: u32 = 66;
    pub(crate) const REV16: u32 = 67;
    pub(crate) const REV32: u32 = 68;
    pub(crate) const REV64: u32 = 69;
    pub(crate) const SQABS: u32 = 70;
    pub(crate) const SQNEG: u32 = 71;
    pub(crate) const CMEQ0: u32 = 72;
    pub(crate) const CMGE0: u32 = 73;
    pub(crate) const CMGT0: u32 = 74;
    pub(crate) const CMLE0: u32 = 75;
    pub(crate) const CMLT0: u32 = 76;
    pub(crate) const SADDLP: u32 = 77;
    pub(crate) const UADDLP: u32 = 78;
    pub(crate) const SADALP: u32 = 79;
    pub(crate) const UADALP: u32 = 80;
    // Narrowing: esz is the destination element size.
    pub(crate) const XTN: u32 = 81;
    pub(crate) const SQXTN: u32 = 82;
    pub(crate) const UQXTN: u32 = 83;
    pub(crate) const SQXTUN: u32 = 84;
    // Widening shift: esz is the source element size.
    pub(crate) const SHLL: u32 = 85;
    // Across lanes: esz is the source element size.
    pub(crate) const ADDV: u32 = 86;
    pub(crate) const SADDLV: u32 = 87;
    pub(crate) const UADDLV: u32 = 88;
    pub(crate) const SMAXV: u32 = 89;
    pub(crate) const SMINV: u32 = 90;
    pub(crate) const UMAXV: u32 = 91;
    pub(crate) const UMINV: u32 = 92;
    // Shifts by immediate.
    pub(crate) const SSHR: u32 = 93;
    pub(crate) const USHR: u32 = 94;
    pub(crate) const SSRA: u32 = 95;
    pub(crate) const USRA: u32 = 96;
    pub(crate) const SRSHR: u32 = 97;
    pub(crate) const URSHR: u32 = 98;
    pub(crate) const SRSRA: u32 = 99;
    pub(crate) const URSRA: u32 = 100;
    pub(crate) const SRI: u32 = 101;
    pub(crate) const SHL: u32 = 102;
    pub(crate) const SLI: u32 = 103;
    pub(crate) const SQSHLI: u32 = 104;
    pub(crate) const UQSHLI: u32 = 105;
    pub(crate) const SQSHLUI: u32 = 106;
    // Widening shift by immediate: esz is the source element size.
    pub(crate) const SSHLL: u32 = 107;
    pub(crate) const USHLL: u32 = 108;
    // Narrowing shifts by immediate: esz is the destination element size.
    pub(crate) const SHRN: u32 = 109;
    pub(crate) const RSHRN: u32 = 110;
    pub(crate) const SQSHRN: u32 = 111;
    pub(crate) const UQSHRN: u32 = 112;
    pub(crate) const SQRSHRN: u32 = 113;
    pub(crate) const UQRSHRN: u32 = 114;
    pub(crate) const SQSHRUN: u32 = 115;
    pub(crate) const SQRSHRUN: u32 = 116;
    // Three registers, different sizes: esz is the narrow element size.
    pub(crate) const SADDL: u32 = 117;
    pub(crate) const UADDL: u32 = 118;
    pub(crate) const SSUBL: u32 = 119;
    pub(crate) const USUBL: u32 = 120;
    pub(crate) const SADDW: u32 = 121;
    pub(crate) const UADDW: u32 = 122;
    pub(crate) const SSUBW: u32 = 123;
    pub(crate) const USUBW: u32 = 124;
    pub(crate) const SABAL: u32 = 125;
    pub(crate) const UABAL: u32 = 126;
    pub(crate) const SABDL: u32 = 127;
    pub(crate) const UABDL: u32 = 128;
    pub(crate) const SMLAL: u32 = 129;
    pub(crate) const UMLAL: u32 = 130;
    pub(crate) const SMLSL: u32 = 131;
    pub(crate) const UMLSL: u32 = 132;
    pub(crate) const SMULL: u32 = 133;
    pub(crate) const UMULL: u32 = 134;
    pub(crate) const SQDMULL: u32 = 135;
    pub(crate) const SQDMLAL: u32 = 136;
    pub(crate) const SQDMLSL: u32 = 137;
    pub(crate) const PMULL: u32 = 138;
    pub(crate) const ADDHN: u32 = 139;
    pub(crate) const SUBHN: u32 = 140;
    pub(crate) const RADDHN: u32 = 141;
    pub(crate) const RSUBHN: u32 = 142;
    // Permutes, the table lookups and EXT.
    pub(crate) const UZP1: u32 = 143;
    pub(crate) const UZP2: u32 = 144;
    pub(crate) const ZIP1: u32 = 145;
    pub(crate) const ZIP2: u32 = 146;
    pub(crate) const TRN1: u32 = 147;
    pub(crate) const TRN2: u32 = 148;
    pub(crate) const EXT: u32 = 149;
    pub(crate) const TBL: u32 = 150;
    pub(crate) const TBX: u32 = 151;
    // Dot products: esz is 2.
    pub(crate) const SDOT: u32 = 152;
    pub(crate) const UDOT: u32 = 153;
}

def!(NEON, "a64_neon", 0, Void, [Ptr, I32, I32], h_neon);

/// The helpers of this module.
pub(crate) const ALL: &[Def] = &[NEON];

/// `MIN`/`MAX` helpers for the signed and unsigned element ranges.
fn smax_of(esz: u32) -> i128 {
    (1i128 << ((8 << esz) - 1)) - 1
}

fn smin_of(esz: u32) -> i128 {
    -(1i128 << ((8 << esz) - 1))
}

fn umax_of(esz: u32) -> i128 {
    (1i128 << (8 << esz)) - 1
}

/// Saturate `x` to the signed range of `esz`, noting a saturation in `sat`.
fn ssat(x: i128, esz: u32, sat: &mut bool) -> u64 {
    if x > smax_of(esz) {
        *sat = true;
        smax_of(esz) as u64
    } else if x < smin_of(esz) {
        *sat = true;
        smin_of(esz) as u64
    } else {
        x as u64
    }
}

/// Saturate `x` to the unsigned range of `esz`, noting a saturation in `sat`.
fn usat(x: i128, esz: u32, sat: &mut bool) -> u64 {
    if x > umax_of(esz) {
        *sat = true;
        umax_of(esz) as u64
    } else if x < 0 {
        *sat = true;
        0
    } else {
        x as u64
    }
}

/// `do_sqrshl_bhs()` and `do_sqrshl_d()`: shift the signed `src` of `bits` bits left by
/// `shift`, right when negative, rounding if asked, saturating if `sat` is given.
pub(crate) fn sqrshl(src: i64, shift: i32, bits: i32, round: bool, sat: Option<&mut bool>) -> i64 {
    let src = i128::from(src);
    if shift <= -bits {
        // Rounding the sign bit always produces 0.
        return if round { 0 } else { (src >> 63) as i64 };
    } else if shift < 0 {
        if round {
            let s = src >> (-shift - 1);
            return ((s >> 1) + (s & 1)) as i64;
        }
        return (src >> -shift) as i64;
    } else if shift < bits {
        let val = src << shift;
        let ext = sext(val as u64, bits_esz(bits));
        if sat.is_none() || i128::from(ext) == val {
            return ext;
        }
    } else if sat.is_none() || src == 0 {
        return 0;
    }
    if let Some(s) = sat {
        *s = true;
    }
    if src >= 0 { smax_of(bits_esz(bits)) as i64 } else { smin_of(bits_esz(bits)) as i64 }
}

/// `do_uqrshl_bhs()` and `do_uqrshl_d()`: the unsigned form of [`sqrshl`].
pub(crate) fn uqrshl(src: u64, shift: i32, bits: i32, round: bool, sat: Option<&mut bool>) -> u64 {
    let src = u128::from(src);
    if shift <= -(bits + i32::from(round)) {
        return 0;
    } else if shift < 0 {
        if round {
            let s = src >> (-shift - 1);
            return ((s >> 1) + (s & 1)) as u64;
        }
        return (src >> -shift) as u64;
    } else if shift < bits {
        let val = src << shift;
        let ext = val as u64 & mask(bits_esz(bits));
        if sat.is_none() || u128::from(ext) == val {
            return ext;
        }
    } else if sat.is_none() || src == 0 {
        return 0;
    }
    if let Some(s) = sat {
        *s = true;
    }
    mask(bits_esz(bits))
}

/// `do_suqrshl_bhs()` and `do_suqrshl_d()`: signed source, unsigned saturated result.
pub(crate) fn suqrshl(src: i64, shift: i32, bits: i32, round: bool, sat: &mut bool) -> u64 {
    if src < 0 {
        *sat = true;
        return 0;
    }
    uqrshl(src as u64, shift, bits, round, Some(sat))
}

fn bits_esz(bits: i32) -> u32 {
    match bits {
        8 => 0,
        16 => 1,
        32 => 2,
        _ => 3,
    }
}

/// `do_sqrdmulh_h()` and `do_sqrdmulh_s()`: the saturating rounding doubling multiply
/// returning the high half, with the accumulator `a` (zero for SQDMULH and SQRDMULH).
fn sqrdmulh(n: i64, m: i64, a: i64, neg: bool, round: bool, esz: u32, sat: &mut bool) -> u64 {
    let bits = 8 << esz;
    let mut ret = i128::from(n) * i128::from(m);
    if neg {
        ret = -ret;
    }
    ret += (i128::from(a) << (bits - 1)) + (i128::from(round) << (bits - 2));
    ret >>= bits - 1;
    ssat(ret, esz, sat)
}

/// Polynomial multiply of the low `bits` of `a` and `b`, without truncation.
fn pmul(a: u64, b: u64, bits: u32) -> u128 {
    let mut r = 0u128;
    for i in 0..bits {
        if b >> i & 1 != 0 {
            r ^= u128::from(a) << i;
        }
    }
    r
}

/// The result of one integer operation on one element, for the elementwise forms. `d` is
/// the old destination element, for the accumulating forms.
fn elem3(op: u32, esz: u32, n: u64, m: u64, d: u64, sat: &mut bool) -> u64 {
    let bits = 8i32 << esz;
    let sn = i128::from(sext(n, esz));
    let sm = i128::from(sext(m, esz));
    let un = i128::from(n & mask(esz));
    let um = i128::from(m & mask(esz));
    let shift = i32::from(m as i8);
    match op {
        op::ADD => n.wrapping_add(m),
        op::SUB => n.wrapping_sub(m),
        op::MUL => n.wrapping_mul(m),
        op::MLA => d.wrapping_add(n.wrapping_mul(m)),
        op::MLS => d.wrapping_sub(n.wrapping_mul(m)),
        op::SMAX => sn.max(sm) as u64,
        op::SMIN => sn.min(sm) as u64,
        op::UMAX => un.max(um) as u64,
        op::UMIN => un.min(um) as u64,
        op::SABD => (sn - sm).unsigned_abs() as u64,
        op::UABD => (un - um).unsigned_abs() as u64,
        op::SABA => d.wrapping_add((sn - sm).unsigned_abs() as u64),
        op::UABA => d.wrapping_add((un - um).unsigned_abs() as u64),
        op::SHADD => ((sn + sm) >> 1) as u64,
        op::UHADD => ((un + um) >> 1) as u64,
        op::SRHADD => ((sn + sm + 1) >> 1) as u64,
        op::URHADD => ((un + um + 1) >> 1) as u64,
        op::SHSUB => ((sn - sm) >> 1) as u64,
        op::UHSUB => ((un - um) >> 1) as u64,
        op::SQADD => ssat(sn + sm, esz, sat),
        op::UQADD => usat(un + um, esz, sat),
        op::SQSUB => ssat(sn - sm, esz, sat),
        op::UQSUB => usat(un - um, esz, sat),
        // SUQADD: signed accumulator d plus unsigned n. USQADD: unsigned d plus signed n.
        op::SUQADD => ssat(i128::from(sext(d, esz)) + un, esz, sat),
        op::USQADD => usat(i128::from(d & mask(esz)) + sn, esz, sat),
        op::SSHL => sqrshl(sn as i64, shift, bits, false, None) as u64,
        op::USHL => uqrshl(un as u64, shift, bits, false, None),
        op::SRSHL => sqrshl(sn as i64, shift, bits, true, None) as u64,
        op::URSHL => uqrshl(un as u64, shift, bits, true, None),
        op::SQSHL => sqrshl(sn as i64, shift, bits, false, Some(sat)) as u64,
        op::UQSHL => uqrshl(un as u64, shift, bits, false, Some(sat)),
        op::SQRSHL => sqrshl(sn as i64, shift, bits, true, Some(sat)) as u64,
        op::UQRSHL => uqrshl(un as u64, shift, bits, true, Some(sat)),
        op::PMUL => pmul(n, m, 8) as u64,
        op::SQDMULH => sqrdmulh(sn as i64, sm as i64, 0, false, false, esz, sat),
        op::SQRDMULH => sqrdmulh(sn as i64, sm as i64, 0, false, true, esz, sat),
        op::SQRDMLAH => sqrdmulh(sn as i64, sm as i64, sext(d, esz), false, true, esz, sat),
        op::SQRDMLSH => sqrdmulh(sn as i64, sm as i64, sext(d, esz), true, true, esz, sat),
        op::CMTST => cmask(n & m & mask(esz) != 0),
        op::CMEQ => cmask(un == um),
        op::CMGE => cmask(sn >= sm),
        op::CMGT => cmask(sn > sm),
        op::CMHI => cmask(un > um),
        op::CMHS => cmask(un >= um),
        op::AND => n & m,
        op::BIC => n & !m,
        op::ORR => n | m,
        op::ORN => n | !m,
        op::EOR => n ^ m,
        // BSL: d selects; BIT: m selects n into d; BIF: !m selects n into d.
        op::BSL => (n & d) | (m & !d),
        op::BIT => (n & m) | (d & !m),
        op::BIF => (n & !m) | (d & m),
        op::ADDP => n.wrapping_add(m),
        op::SMAXP => sn.max(sm) as u64,
        op::SMINP => sn.min(sm) as u64,
        op::UMAXP => un.max(um) as u64,
        op::UMINP => un.min(um) as u64,
        _ => unreachable!("bad a64_neon three-same op {op}"),
    }
}

fn cmask(b: bool) -> u64 {
    if b { u64::MAX } else { 0 }
}

/// The result of one two-register operation on one element.
fn elem2(op: u32, esz: u32, n: u64, sat: &mut bool) -> u64 {
    let bits = 8u32 << esz;
    let sn = i128::from(sext(n, esz));
    let un = n & mask(esz);
    match op {
        op::ABS => sn.unsigned_abs() as u64,
        op::NEG => (-sn) as u64,
        op::NOT => !n,
        op::CLS => {
            // The number of bits after the sign bit that equal it.
            let x = if sn < 0 { !un & mask(esz) } else { un };
            u64::from((x << (64 - bits)).leading_zeros().min(bits)) - 1
        }
        op::CLZ => u64::from((un << (64 - bits)).leading_zeros().min(bits)),
        op::CNT => u64::from(un.count_ones()),
        op::RBIT => un.reverse_bits() >> (64 - bits),
        op::SQABS => ssat(sn.abs(), esz, sat),
        op::SQNEG => ssat(-sn, esz, sat),
        op::CMEQ0 => cmask(sn == 0),
        op::CMGE0 => cmask(sn >= 0),
        op::CMGT0 => cmask(sn > 0),
        op::CMLE0 => cmask(sn <= 0),
        op::CMLT0 => cmask(sn < 0),
        _ => unreachable!("bad a64_neon two-register op {op}"),
    }
}

/// The source element `i` of a widening operation on the half of `v` that `hi` selects.
fn half(v: &V, esz: u32, i: usize, hi: bool, oprsz_elems: usize) -> u64 {
    get(v, esz, i + if hi { oprsz_elems } else { 0 })
}

fn h_neon(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    let r = Regs::unpack(a[1]);
    let desc = Desc::unpack(a[2]);
    let env = &mut *h.env;
    let n = vload(env, r.n);
    let m = vload(env, r.m);
    let d = vload(env, r.d);
    let mut sat = false;
    let out = neon(&desc, r, env, &n, &m, &d, &mut sat);
    if sat {
        set_qc(env);
    }
    vstore(env, r.d, &out);
    Ok(0)
}

/// Compute the result of the integer operation `desc` on the sources.
#[allow(clippy::too_many_lines)]
fn neon(desc: &Desc, r: Regs, env: &[u8], n: &V, m: &V, d: &V, sat: &mut bool) -> V {
    let esz = desc.esz;
    let oprsz = desc.oprsz;
    let elems = oprsz >> esz;
    let bits = 8i32 << esz;
    let shift = desc.imm as i32;
    let mut out = [0u8; 16];
    match desc.op {
        op::ADD..=op::BIF => {
            for i in 0..elems {
                let mm = if desc.idx { get(m, esz, desc.imm as usize) } else { get(m, esz, i) };
                let x = elem3(desc.op, esz, get(n, esz, i), mm, get(d, esz, i), sat);
                set(&mut out, esz, i, x);
            }
        }
        op::ADDP..=op::UMINP => {
            let half_e = elems / 2;
            for i in 0..elems {
                let (src, j) = if i < half_e { (n, 2 * i) } else { (m, 2 * (i - half_e)) };
                // Scalar ADDP: one result from the two elements of n.
                let src = if elems == 1 { n } else { src };
                let x = elem3(desc.op, esz, get(src, esz, j), get(src, esz, j + 1), 0, sat);
                set(&mut out, esz, i, x);
            }
        }
        op::REV16 | op::REV32 | op::REV64 => {
            let cont = match desc.op {
                op::REV16 => 1,
                op::REV32 => 2,
                _ => 3,
            };
            let per = 1usize << (cont - esz);
            for i in 0..elems {
                let j = i ^ (per - 1);
                set(&mut out, esz, i, get(n, esz, j));
            }
        }
        op::ABS..=op::CMLT0 => {
            for i in 0..elems {
                set(&mut out, esz, i, elem2(desc.op, esz, get(n, esz, i), sat));
            }
        }
        op::SADDLP | op::UADDLP | op::SADALP | op::UADALP => {
            let signed = matches!(desc.op, op::SADDLP | op::SADALP);
            let acc = matches!(desc.op, op::SADALP | op::UADALP);
            for i in 0..elems / 2 {
                let e = |k| if signed { gets(n, esz, k) as u64 } else { get(n, esz, k) };
                let mut x = e(2 * i).wrapping_add(e(2 * i + 1));
                if acc {
                    x = x.wrapping_add(get(d, esz + 1, i));
                }
                set(&mut out, esz + 1, i, x);
            }
        }
        op::XTN | op::SQXTN | op::UQXTN | op::SQXTUN => {
            // esz is the narrow size; the source has elems of esz + 1 in all of n.
            let count = if oprsz < 8 { 1 } else { 8 >> esz };
            let mut lo = [0u8; 16];
            for i in 0..count {
                let s = i128::from(gets(n, esz + 1, i));
                let u = i128::from(get(n, esz + 1, i));
                let x = match desc.op {
                    op::XTN => u as u64,
                    op::SQXTN => ssat(s, esz, sat),
                    op::UQXTN => usat(u, esz, sat),
                    _ => usat(s, esz, sat),
                };
                set(&mut lo, esz, i, x);
            }
            out = narrow_out(&lo, d, desc.hi, oprsz);
        }
        op::SHLL => {
            for i in 0..8 >> esz {
                let x = half(n, esz, i, desc.hi, 8 >> esz) << bits;
                set(&mut out, esz + 1, i, x);
            }
        }
        op::ADDV..=op::UMINV => {
            let wide = matches!(desc.op, op::SADDLV | op::UADDLV);
            let mut acc: i128 = match desc.op {
                op::SMAXV | op::SMINV => i128::from(gets(n, esz, 0)),
                op::UMAXV | op::UMINV => i128::from(get(n, esz, 0)),
                _ => 0,
            };
            for i in 0..elems {
                let s = i128::from(gets(n, esz, i));
                let u = i128::from(get(n, esz, i));
                acc = match desc.op {
                    op::ADDV | op::UADDLV => acc + u,
                    op::SADDLV => acc + s,
                    op::SMAXV => acc.max(s),
                    op::SMINV => acc.min(s),
                    op::UMAXV => acc.max(u),
                    _ => acc.min(u),
                };
            }
            let rsz = if wide { esz + 1 } else { esz };
            set(&mut out, rsz, 0, acc as u64);
        }
        op::SSHR..=op::SQSHLUI => {
            for i in 0..elems {
                let s = gets(n, esz, i);
                let u = get(n, esz, i);
                let dd = get(d, esz, i);
                let x = match desc.op {
                    op::SSHR => sqrshl(s, -shift, bits, false, None) as u64,
                    op::USHR => uqrshl(u, -shift, bits, false, None),
                    op::SSRA => dd.wrapping_add(sqrshl(s, -shift, bits, false, None) as u64),
                    op::USRA => dd.wrapping_add(uqrshl(u, -shift, bits, false, None)),
                    op::SRSHR => sqrshl(s, -shift, bits, true, None) as u64,
                    op::URSHR => uqrshl(u, -shift, bits, true, None),
                    op::SRSRA => dd.wrapping_add(sqrshl(s, -shift, bits, true, None) as u64),
                    op::URSRA => dd.wrapping_add(uqrshl(u, -shift, bits, true, None)),
                    op::SRI => {
                        if shift >= bits {
                            dd
                        } else {
                            let mk = mask(esz) >> shift;
                            (dd & !mk) | (u >> shift)
                        }
                    }
                    op::SHL => u << shift,
                    op::SLI => {
                        let mk = mask(esz) << shift;
                        (dd & !mk) | (u << shift)
                    }
                    op::SQSHLI => sqrshl(s, shift, bits, false, Some(sat)) as u64,
                    op::UQSHLI => uqrshl(u, shift, bits, false, Some(sat)),
                    _ => suqrshl(s, shift, bits, false, sat),
                };
                set(&mut out, esz, i, x);
            }
        }
        op::SSHLL | op::USHLL => {
            for i in 0..8 >> esz {
                let x = if desc.op == op::SSHLL {
                    sext(half(n, esz, i, desc.hi, 8 >> esz), esz) as u64
                } else {
                    half(n, esz, i, desc.hi, 8 >> esz)
                };
                set(&mut out, esz + 1, i, x << shift);
            }
        }
        op::SHRN..=op::SQRSHRUN => {
            // esz is the narrow size, the source elements are esz + 1, shift is 1 to bits.
            let count = if oprsz < 8 { 1 } else { 8 >> esz };
            let wbits = bits * 2;
            let mut lo = [0u8; 16];
            for i in 0..count {
                let s = gets(n, esz + 1, i);
                let u = get(n, esz + 1, i);
                let x = match desc.op {
                    op::SHRN => u >> shift,
                    op::RSHRN => uqrshl(u, -shift, wbits, true, None),
                    op::SQSHRN => ssat(i128::from(sqrshl(s, -shift, wbits, false, None)), esz, sat),
                    op::SQRSHRN => ssat(i128::from(sqrshl(s, -shift, wbits, true, None)), esz, sat),
                    op::UQSHRN => usat(i128::from(uqrshl(u, -shift, wbits, false, None)), esz, sat),
                    op::UQRSHRN => usat(i128::from(uqrshl(u, -shift, wbits, true, None)), esz, sat),
                    op::SQSHRUN => {
                        usat(i128::from(sqrshl(s, -shift, wbits, false, None)), esz, sat)
                    }
                    _ => usat(i128::from(sqrshl(s, -shift, wbits, true, None)), esz, sat),
                };
                set(&mut lo, esz, i, x);
            }
            out = narrow_out(&lo, d, desc.hi, oprsz);
        }
        op::PMULL if esz == 3 => {
            let k = usize::from(desc.hi);
            let p = pmul(get(n, 3, k), get(m, 3, k), 64);
            set(&mut out, 3, 0, p as u64);
            set(&mut out, 3, 1, (p >> 64) as u64);
            return out;
        }
        op::SADDL..=op::PMULL => {
            // Widening: esz is the narrow size, the result elements are esz + 1. A scalar
            // (oprsz below 8) has one element.
            let count = if oprsz < 8 { 1 } else { 8 >> esz };
            let we = esz + 1;
            for i in 0..count {
                let wide_n = matches!(desc.op, op::SADDW | op::UADDW | op::SSUBW | op::USUBW);
                let (sn, un) = if wide_n {
                    (i128::from(gets(n, we, i)), i128::from(get(n, we, i)))
                } else {
                    let x = half(n, esz, i, desc.hi, 8 >> esz);
                    (i128::from(sext(x, esz)), i128::from(x))
                };
                let mx = if desc.idx {
                    get(m, esz, desc.imm as usize)
                } else {
                    half(m, esz, i, desc.hi, 8 >> esz)
                };
                let sm = i128::from(sext(mx, esz));
                let um = i128::from(mx);
                let dd = get(d, we, i);
                let sd = i128::from(gets(d, we, i));
                let x = match desc.op {
                    op::SADDL | op::SADDW => (sn + sm) as u64,
                    op::UADDL | op::UADDW => (un + um) as u64,
                    op::SSUBL | op::SSUBW => (sn - sm) as u64,
                    op::USUBL | op::USUBW => (un - um) as u64,
                    op::SABAL => dd.wrapping_add((sn - sm).unsigned_abs() as u64),
                    op::UABAL => dd.wrapping_add((un - um).unsigned_abs() as u64),
                    op::SABDL => (sn - sm).unsigned_abs() as u64,
                    op::UABDL => (un - um).unsigned_abs() as u64,
                    op::SMLAL => dd.wrapping_add((sn * sm) as u64),
                    op::UMLAL => dd.wrapping_add((un * um) as u64),
                    op::SMLSL => dd.wrapping_sub((sn * sm) as u64),
                    op::UMLSL => dd.wrapping_sub((un * um) as u64),
                    op::SMULL => (sn * sm) as u64,
                    op::UMULL => (un * um) as u64,
                    op::SQDMULL => ssat(2 * sn * sm, we, sat),
                    op::SQDMLAL | op::SQDMLSL => {
                        // neon_addl_saturate: the doubled product saturates, then the sum.
                        let p = i128::from(sext(ssat(2 * sn * sm, we, sat), we));
                        let p = if desc.op == op::SQDMLSL { -p } else { p };
                        ssat(sd + p, we, sat)
                    }
                    _ => pmul(un as u64, um as u64, 8 << esz) as u64,
                };
                set(&mut out, we, i, x);
            }
            // A scalar result fills one element of the wide size.
            if oprsz < 8 {
                clear_tail(&mut out, 1 << we);
            }
            return out;
        }
        op::ADDHN..=op::RSUBHN => {
            let count = 8 >> esz;
            let we = esz + 1;
            let mut lo = [0u8; 16];
            for i in 0..count {
                let a = u128::from(get(n, we, i));
                let b = u128::from(get(m, we, i));
                let wm = u128::from(mask(we));
                let x = match desc.op {
                    op::ADDHN => a.wrapping_add(b) & wm,
                    op::SUBHN => a.wrapping_sub(b) & wm,
                    op::RADDHN => (a.wrapping_add(b) & wm).wrapping_add(1 << (bits - 1)) & wm,
                    _ => (a.wrapping_sub(b) & wm).wrapping_add(1 << (bits - 1)) & wm,
                };
                set(&mut lo, esz, i, (x >> bits) as u64);
            }
            out = narrow_out(&lo, d, desc.hi, 16);
            clear_tail(&mut out, if desc.hi { 16 } else { 8 });
            return out;
        }
        op::UZP1 | op::UZP2 => {
            let odd = usize::from(desc.op == op::UZP2);
            let half_e = elems / 2;
            for i in 0..elems {
                let (src, j) =
                    if i < half_e { (n, 2 * i + odd) } else { (m, 2 * (i - half_e) + odd) };
                set(&mut out, esz, i, get(src, esz, j));
            }
        }
        op::ZIP1 | op::ZIP2 => {
            let base = if desc.op == op::ZIP2 { elems / 2 } else { 0 };
            for i in 0..elems / 2 {
                set(&mut out, esz, 2 * i, get(n, esz, base + i));
                set(&mut out, esz, 2 * i + 1, get(m, esz, base + i));
            }
        }
        op::TRN1 | op::TRN2 => {
            let odd = usize::from(desc.op == op::TRN2);
            for i in (0..elems).step_by(2) {
                set(&mut out, esz, i, get(n, esz, i + odd));
                set(&mut out, esz, i + 1, get(m, esz, i + odd));
            }
        }
        op::EXT => {
            let mut cat = [0u8; 32];
            cat[..oprsz].copy_from_slice(&n[..oprsz]);
            cat[oprsz..2 * oprsz].copy_from_slice(&m[..oprsz]);
            let p = desc.imm as usize;
            out[..oprsz].copy_from_slice(&cat[p..p + oprsz]);
        }
        op::TBL | op::TBX => {
            // imm holds the number of table registers minus one; the table starts at Rn and
            // wraps from V31 to V0.
            let len = desc.imm as usize + 1;
            let mut table = [0u8; 64];
            for k in 0..len {
                let t = vload(env, (r.n + k as u32) & 31);
                table[16 * k..16 * k + 16].copy_from_slice(&t);
            }
            for i in 0..oprsz {
                let ix = m[i] as usize;
                out[i] = if ix < 16 * len {
                    table[ix]
                } else if desc.op == op::TBX {
                    d[i]
                } else {
                    0
                };
            }
        }
        op::SDOT | op::UDOT => {
            for i in 0..oprsz / 4 {
                let mut acc = get(d, 2, i) as u32;
                let mi = if desc.idx { (i & !3) + desc.imm as usize } else { i };
                for j in 0..4 {
                    let (a, b) = if desc.op == op::SDOT {
                        (i32::from(n[4 * i + j] as i8), i32::from(m[4 * mi + j] as i8))
                    } else {
                        (i32::from(n[4 * i + j]), i32::from(m[4 * mi + j]))
                    };
                    acc = acc.wrapping_add((a * b) as u32);
                }
                set(&mut out, 2, i, u64::from(acc));
            }
        }
        _ => unreachable!("bad a64_neon op {}", desc.op),
    }
    clear_tail(&mut out, oprsz);
    out
}

/// The destination of a narrowing operation: the 8 bytes in `lo` go to the low half with
/// the high half cleared, or to the high half with the low half of `d` kept.
fn narrow_out(lo: &V, d: &V, hi: bool, oprsz: usize) -> V {
    let mut out = [0u8; 16];
    if hi {
        out[..8].copy_from_slice(&d[..8]);
        out[8..].copy_from_slice(&lo[..8]);
    } else {
        out[..8].copy_from_slice(&lo[..8]);
        // A scalar narrowing writes one element and clears the rest.
        clear_tail(&mut out, oprsz.min(8));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_shifts_match_vec_internal() {
        let mut sat = false;
        // SRSHR #8 of a byte always rounds to 0.
        assert_eq!(sqrshl(-128, -8, 8, true, None), 0);
        assert_eq!(sqrshl(127, -8, 8, true, None), 0);
        // URSHR #8 gives the top bit.
        assert_eq!(uqrshl(0x80, -8, 8, true, None), 1);
        assert_eq!(uqrshl(0x7f, -8, 8, true, None), 0);
        // SQSHL saturates.
        assert_eq!(sqrshl(0x40, 1, 8, false, Some(&mut sat)), 0x7f);
        assert!(sat);
        // SSHL by a large negative shift fills with the sign.
        assert_eq!(sqrshl(-5, -100, 8, false, None), -1);
        assert_eq!(uqrshl(u64::MAX, -64, 64, true, None), 1);
    }

    #[test]
    fn doubling_multiplies() {
        let mut sat = false;
        assert_eq!(sqrdmulh(-0x8000, -0x8000, 0, false, false, 1, &mut sat), 0x7fff);
        assert!(sat);
        sat = false;
        assert_eq!(sqrdmulh(0x4000, 0x4000, 0, false, true, 1, &mut sat), 0x2000);
        assert!(!sat);
    }
}
