// SPDX-License-Identifier: GPL-2.0-or-later

//! The SVE and SVE2 helpers: QEMU's `sve_helper.c`, with the SVE2 parts of `vec_helper.c`
//! and the inline functions of `vec_internal.h` they use.
//!
//! QEMU has one helper per operation and element size and passes pointers to the vector and
//! predicate registers. This port has two helpers. `sve` does every data processing
//! instruction and `sve_mem` every load and store. Both take a descriptor ([`Dsc`]) naming
//! the family, the operation, the element size, the vector length and up to five register
//! numbers, and two 64-bit operands (an immediate, a general register or an address). They
//! read the registers out of `env`, compute the whole result and write it back, so a source
//! that is also the destination is read before it is written, as QEMU's helpers arrange. The
//! instructions that set NZCV write the flags into `env` themselves. The value returned is
//! the scalar result of the instructions that write a general register. The floating point
//! family is in [`super::sve_fp`]; the SVE2 crypto family (AESE, AESD, AESMC, AESIMC, SM4E,
//! SM4EKEY, RAX1 and PMULLB/PMULLT) uses the AdvSIMD ones of [`super::crypto`] per 128-bit
//! segment.
//!
//! The results are those of the QEMU helpers bit for bit. Saturating SVE operations do not
//! set FPSR.QC, as in QEMU (the architecture has no QC for SVE).
//!
//! Differences from QEMU:
//!
//! - Loads and stores go through the softmmu one element at a time ([`cpu_ld_mmu`]) instead
//!   of through a host pointer to the page. Every page an operation touches is checked first,
//!   in the order QEMU checks them, so a fault leaves the registers and memory as QEMU does.
//! - Watchpoints and MTE tag checks are not done.
//! - First fault and non fault loads treat a page that is not RAM as a fault, as QEMU does,
//!   and also decline the elements on the second page as QEMU does.
//! - QEMU 11.1's `sve_ldnfff1_r()` reads the governing predicate at the wrong offset when
//!   the first active element is not at a multiple of 64 bytes into the vector (it loads
//!   the 64-bit word at byte `reg_off / 8` and shifts it by `reg_off % 64`), so with such a
//!   predicate its first fault and non fault loads pick the wrong elements. This port uses
//!   the predicate as the architecture says.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use ruvm_jit::cputlb::{cpu_ld_mmu, cpu_st_mmu, probe_access, probe_access_nonfault};
use ruvm_jit::{Cpu, CpuLoopExit, MmuAccessType, Ra};
use ruvm_jit_core::HelperType::{I64, Ptr};
use ruvm_jit_core::{MemOp, MemOpIdx};
use ruvm_jit_interp::{HelperEnv, Unwind};

use super::helpers::{Def, def, run};
use super::vec_helper::{sqrshl, suqrshl, uqrshl};
use crate::cpu::{CF, FFR, NF, PREG_SIZE, VF, ZF, ZREG_SIZE, preg_off, vreg_off};

/// A Z register image, as `env` holds it.
pub(crate) type Z = [u8; ZREG_SIZE];
/// A predicate register image as four little endian words.
pub(crate) type P = [u64; PREG_SIZE / 8];

/// The descriptor of a call of `sve` or `sve_mem`.
///
/// Bits 0 to 4 hold the family, 5 to 12 the operation in the family, 13 and 14 the element
/// size, 15 to 18 the vector length in quadwords minus one, 19 to 43 the register numbers
/// Rd, Rn, Rm, Ra and Pg (five bits each), and 44 to 63 operation specific data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Dsc {
    pub(crate) fam: u32,
    pub(crate) op: u32,
    pub(crate) esz: u32,
    /// The vector length in bytes.
    pub(crate) vl: usize,
    pub(crate) d: usize,
    pub(crate) n: usize,
    pub(crate) m: usize,
    pub(crate) a: usize,
    pub(crate) g: usize,
    pub(crate) data: u32,
}

impl Dsc {
    /// Pack a descriptor. `r` holds Rd, Rn, Rm, Ra and Pg.
    pub(crate) fn pack(fam: u32, op: u32, esz: i32, vl: u32, r: [i32; 5], data: u32) -> u64 {
        debug_assert!(fam < 32 && op < 256 && data < (1 << 20));
        debug_assert!(vl >= 16 && vl <= ZREG_SIZE as u32 && vl % 16 == 0);
        let mut x = u64::from(fam)
            | u64::from(op) << 5
            | (esz as u64 & 3) << 13
            | u64::from(vl / 16 - 1) << 15;
        for (i, reg) in r.iter().enumerate() {
            x |= (*reg as u64 & 31) << (19 + 5 * i);
        }
        x | u64::from(data) << 44
    }

    pub(crate) fn unpack(x: u64) -> Dsc {
        let reg = |i: u32| ((x >> (19 + 5 * i)) & 31) as usize;
        Dsc {
            fam: (x & 31) as u32,
            op: ((x >> 5) & 0xff) as u32,
            esz: ((x >> 13) & 3) as u32,
            vl: (((x >> 15) & 15) as usize + 1) * 16,
            d: reg(0),
            n: reg(1),
            m: reg(2),
            a: reg(3),
            g: reg(4),
            data: (x >> 44) as u32,
        }
    }

    /// The number of elements in a vector.
    fn elems(&self) -> usize {
        self.vl >> self.esz
    }
}

/// The families of [`Dsc::fam`].
#[allow(missing_docs)]
pub(crate) mod fam {
    /// Unpredicated `d[i] = bin(n[i], m[i])`; data bit 0: m is the 64-bit column (ZZW).
    pub(crate) const ZZZ: u32 = 0;
    /// Predicated `d[i] = bin(n[i], m[i])` for the active elements; data bit 0 as for ZZZ.
    pub(crate) const ZPZZ: u32 = 1;
    /// Predicated `d[i] = bin(n[i], x)` for the active elements.
    pub(crate) const ZPZI: u32 = 2;
    /// Unpredicated `d[i] = bin(n[i], x)`.
    pub(crate) const ZZI: u32 = 3;
    /// Predicated unary, merging: `d[i] = un(n[i])` for the active elements.
    pub(crate) const ZPZ: u32 = 4;
    /// Reductions to a scalar in Vd.
    pub(crate) const RED: u32 = 5;
    /// Integer compares into a predicate; data bit 0: wide, bit 1: m is the immediate.
    pub(crate) const CMP: u32 = 6;
    /// Predicate logic; data bit 0: set the flags.
    pub(crate) const PPPP: u32 = 7;
    /// Permutes, moves and the other whole vector operations ([`super::pm`]).
    pub(crate) const PERM: u32 = 8;
    /// Predicate operations ([`super::pr`]).
    pub(crate) const PRED: u32 = 9;
    /// Floating point ([`super::super::sve_fp`]).
    pub(crate) const FP: u32 = 10;
    /// The SVE2 crypto instructions and PMULLB and PMULLT ([`super::cr`]).
    pub(crate) const CRYPTO: u32 = 11;
}

/// The operations of the CRYPTO family. They work on each 128-bit segment (AES and SM4), on
/// 64-bit elements (RAX1) or on the element size (PMULL, with data bit 0 selecting the odd
/// source elements).
#[allow(missing_docs)]
pub(crate) mod cr {
    pub(crate) const AESE: u32 = 1;
    pub(crate) const AESD: u32 = 2;
    pub(crate) const AESMC: u32 = 3;
    pub(crate) const AESIMC: u32 = 4;
    pub(crate) const SM4E: u32 = 5;
    pub(crate) const SM4EKEY: u32 = 6;
    pub(crate) const RAX1: u32 = 7;
    pub(crate) const PMULL: u32 = 8;
}

/// The element-wise binary operations of the ZZZ, ZPZZ, ZPZI and ZZI families. The
/// accumulating ones also read Zd.
#[allow(missing_docs)]
pub(crate) mod b {
    pub(crate) const ADD: u32 = 1;
    pub(crate) const SUB: u32 = 2;
    pub(crate) const SUBR: u32 = 3;
    pub(crate) const SQADD: u32 = 4;
    pub(crate) const UQADD: u32 = 5;
    pub(crate) const SQSUB: u32 = 6;
    pub(crate) const UQSUB: u32 = 7;
    pub(crate) const AND: u32 = 8;
    pub(crate) const ORR: u32 = 9;
    pub(crate) const EOR: u32 = 10;
    pub(crate) const BIC: u32 = 11;
    pub(crate) const MUL: u32 = 12;
    pub(crate) const SMULH: u32 = 13;
    pub(crate) const UMULH: u32 = 14;
    pub(crate) const PMUL: u32 = 15;
    pub(crate) const SQDMULH: u32 = 16;
    pub(crate) const SQRDMULH: u32 = 17;
    pub(crate) const SMAX: u32 = 18;
    pub(crate) const UMAX: u32 = 19;
    pub(crate) const SMIN: u32 = 20;
    pub(crate) const UMIN: u32 = 21;
    pub(crate) const SABD: u32 = 22;
    pub(crate) const UABD: u32 = 23;
    pub(crate) const SDIV: u32 = 24;
    pub(crate) const UDIV: u32 = 25;
    pub(crate) const ASR: u32 = 26;
    pub(crate) const LSR: u32 = 27;
    pub(crate) const LSL: u32 = 28;
    pub(crate) const SRSHL: u32 = 29;
    pub(crate) const URSHL: u32 = 30;
    pub(crate) const SQSHL: u32 = 31;
    pub(crate) const UQSHL: u32 = 32;
    pub(crate) const SQRSHL: u32 = 33;
    pub(crate) const UQRSHL: u32 = 34;
    pub(crate) const SHADD: u32 = 35;
    pub(crate) const UHADD: u32 = 36;
    pub(crate) const SHSUB: u32 = 37;
    pub(crate) const UHSUB: u32 = 38;
    pub(crate) const SRHADD: u32 = 39;
    pub(crate) const URHADD: u32 = 40;
    pub(crate) const SUQADD: u32 = 41;
    pub(crate) const USQADD: u32 = 42;
    /// Shift right for divide, by the immediate.
    pub(crate) const ASRD: u32 = 43;
    /// Rounding shifts right by the immediate.
    pub(crate) const SRSHR: u32 = 44;
    pub(crate) const URSHR: u32 = 45;
    /// Saturating shift left of a signed value to an unsigned result, by the immediate.
    pub(crate) const SQSHLU: u32 = 46;
    pub(crate) const SABA: u32 = 47;
    pub(crate) const UABA: u32 = 48;
    /// Shift right by the immediate and accumulate.
    pub(crate) const SSRA: u32 = 49;
    pub(crate) const USRA: u32 = 50;
    pub(crate) const SRSRA: u32 = 51;
    pub(crate) const URSRA: u32 = 52;
    /// Shift by the immediate and insert.
    pub(crate) const SRI: u32 = 53;
    pub(crate) const SLI: u32 = 54;
    pub(crate) const BEXT: u32 = 55;
    pub(crate) const BDEP: u32 = 56;
    pub(crate) const BGRP: u32 = 57;
    /// Shift right by the immediate (`ASR`, `LSR` by more than the element size allowed).
    pub(crate) const ASR_I: u32 = 58;
    pub(crate) const LSR_I: u32 = 59;
    /// The first operand: CPY, MOV.
    pub(crate) const MOVM: u32 = 60;
    /// SADALP and UADALP: `m + n.lo + n.hi`, with m the accumulator.
    pub(crate) const SADALP: u32 = 61;
    pub(crate) const UADALP: u32 = 62;
    /// SCLAMP and UCLAMP: `min(max(d, n), m)`.
    pub(crate) const SCLAMP: u32 = 63;
    pub(crate) const UCLAMP: u32 = 64;
    /// The second operand.
    pub(crate) const MOV2: u32 = 65;
    /// SVE2 saturating doubling multiply-add high: `sqrdmlah(n, m, d)` and friends.
    pub(crate) const SQRDMLAH: u32 = 66;
    pub(crate) const SQRDMLSH: u32 = 67;
    pub(crate) const MLA: u32 = 68;
    pub(crate) const MLS: u32 = 69;
    /// `sve_sqaddi_*`, `sve_uqaddi_*`: add the signed 64-bit `m` (unsigned for 64-bit
    /// `UQADDI`) and saturate; `sve_uqsubi_d`: subtract the unsigned `m` and saturate.
    pub(crate) const SQADDI: u32 = 70;
    pub(crate) const UQADDI: u32 = 71;
    pub(crate) const UQSUBI: u32 = 72;
}

/// The unary operations of the ZPZ family.
#[allow(missing_docs)]
pub(crate) mod u {
    pub(crate) const CLS: u32 = 1;
    pub(crate) const CLZ: u32 = 2;
    pub(crate) const CNT: u32 = 3;
    pub(crate) const CNOT: u32 = 4;
    pub(crate) const NOT: u32 = 5;
    pub(crate) const FABS: u32 = 6;
    pub(crate) const FNEG: u32 = 7;
    pub(crate) const ABS: u32 = 8;
    pub(crate) const NEG: u32 = 9;
    pub(crate) const SXTB: u32 = 10;
    pub(crate) const UXTB: u32 = 11;
    pub(crate) const SXTH: u32 = 12;
    pub(crate) const UXTH: u32 = 13;
    pub(crate) const SXTW: u32 = 14;
    pub(crate) const UXTW: u32 = 15;
    pub(crate) const REVB: u32 = 16;
    pub(crate) const REVH: u32 = 17;
    pub(crate) const REVW: u32 = 18;
    pub(crate) const RBIT: u32 = 19;
    pub(crate) const SQABS: u32 = 20;
    pub(crate) const SQNEG: u32 = 21;
    pub(crate) const URECPE: u32 = 22;
    pub(crate) const URSQRTE: u32 = 23;
    pub(crate) const MOV: u32 = 24;
}

/// The reductions of the RED family.
#[allow(missing_docs)]
pub(crate) mod r {
    pub(crate) const ORV: u32 = 1;
    pub(crate) const EORV: u32 = 2;
    pub(crate) const ANDV: u32 = 3;
    pub(crate) const UADDV: u32 = 4;
    pub(crate) const SADDV: u32 = 5;
    pub(crate) const SMAXV: u32 = 6;
    pub(crate) const UMAXV: u32 = 7;
    pub(crate) const SMINV: u32 = 8;
    pub(crate) const UMINV: u32 = 9;
}

/// The conditions of the CMP family.
#[allow(missing_docs)]
pub(crate) mod c {
    pub(crate) const EQ: u32 = 0;
    pub(crate) const NE: u32 = 1;
    pub(crate) const GE: u32 = 2;
    pub(crate) const GT: u32 = 3;
    pub(crate) const LT: u32 = 4;
    pub(crate) const LE: u32 = 5;
    pub(crate) const HS: u32 = 6;
    pub(crate) const HI: u32 = 7;
    pub(crate) const LO: u32 = 8;
    pub(crate) const LS: u32 = 9;
}

/// The predicate logic operations of the PPPP family.
#[allow(missing_docs)]
pub(crate) mod pl {
    pub(crate) const AND: u32 = 0;
    pub(crate) const BIC: u32 = 1;
    pub(crate) const EOR: u32 = 2;
    pub(crate) const SEL: u32 = 3;
    pub(crate) const ORR: u32 = 4;
    pub(crate) const ORN: u32 = 5;
    pub(crate) const NOR: u32 = 6;
    pub(crate) const NAND: u32 = 7;
}

/// The operations of the PERM family.
#[allow(missing_docs)]
pub(crate) mod pm {
    /// Broadcast x to every element.
    pub(crate) const DUP: u32 = 1;
    /// Broadcast element x of Zn (zero if out of range).
    pub(crate) const DUPX: u32 = 2;
    /// `d[i] = x + i * y`.
    pub(crate) const INDEX: u32 = 3;
    /// Shift up one element and insert x; data bit 0: x is element 0 of Vm.
    pub(crate) const INSR: u32 = 4;
    pub(crate) const REV: u32 = 5;
    pub(crate) const TBL: u32 = 6;
    pub(crate) const TBL2: u32 = 7;
    pub(crate) const TBX: u32 = 8;
    pub(crate) const ZIP1: u32 = 9;
    pub(crate) const ZIP2: u32 = 10;
    pub(crate) const UZP1: u32 = 11;
    pub(crate) const UZP2: u32 = 12;
    pub(crate) const TRN1: u32 = 13;
    pub(crate) const TRN2: u32 = 14;
    /// Destructive EXT, byte offset x.
    pub(crate) const EXT: u32 = 15;
    /// Constructive EXT of Zn and Zn+1.
    pub(crate) const EXT2: u32 = 16;
    pub(crate) const SPLICE: u32 = 17;
    pub(crate) const SPLICE2: u32 = 18;
    pub(crate) const COMPACT: u32 = 19;
    pub(crate) const SEL: u32 = 20;
    /// Unpack; data bit 0: high half, bit 1: unsigned.
    pub(crate) const UNPK: u32 = 21;
    pub(crate) const CLASTA_Z: u32 = 22;
    pub(crate) const CLASTB_Z: u32 = 23;
    /// The scalar CLAST and LAST forms; data bit 0: to Vd (else returned), bit 1: B.
    pub(crate) const CLAST: u32 = 24;
    pub(crate) const LAST: u32 = 25;
    /// Merging copy of x (CPY Zd, Pg/M); data bit 0: x is element 0 of Vn.
    pub(crate) const CPY_M: u32 = 26;
    /// Zeroing copy of x (CPY Zd, Pg/Z).
    pub(crate) const CPY_Z: u32 = 27;
    /// ADR; data: 0 for sxtw, 1 for uxtw, 2 for same size, x the shift.
    pub(crate) const ADR: u32 = 28;
    pub(crate) const MOVPRFX: u32 = 29;
    pub(crate) const MOVPRFX_Z: u32 = 30;
    pub(crate) const MOVPRFX_M: u32 = 31;
    /// The bitwise EOR3, BCAX and the BSL forms, with Rm and Ra as the other operands.
    pub(crate) const EOR3: u32 = 32;
    pub(crate) const BCAX: u32 = 33;
    pub(crate) const BSL: u32 = 34;
    pub(crate) const BSL1N: u32 = 35;
    pub(crate) const BSL2N: u32 = 36;
    pub(crate) const NBSL: u32 = 37;
    /// Unpredicated `d = bin(op = data, n, rotr(m))` for XAR.
    pub(crate) const XAR: u32 = 38;
    /// HISTSEG and HISTCNT.
    pub(crate) const HISTSEG: u32 = 39;
    pub(crate) const HISTCNT: u32 = 40;
    /// MATCH and NMATCH into Pd, with flags.
    pub(crate) const MATCH: u32 = 41;
    pub(crate) const NMATCH: u32 = 42;
}

/// The operations of the PRED family.
#[allow(missing_docs)]
pub(crate) mod pr {
    /// PTRUE with the pattern in x; data bit 0: set the flags.
    pub(crate) const PTRUE: u32 = 1;
    pub(crate) const PTEST: u32 = 2;
    /// Pd = FFR (& Pg if data bit 1); data bit 0: set the flags.
    pub(crate) const RDFFR: u32 = 3;
    pub(crate) const WRFFR: u32 = 4;
    pub(crate) const SETFFR: u32 = 5;
    pub(crate) const PFIRST: u32 = 6;
    pub(crate) const PNEXT: u32 = 7;
    /// BRKA and BRKB; data bit 0: flags, bit 1: merging, bit 2: B.
    pub(crate) const BRK: u32 = 8;
    /// BRKPA and BRKPB; data bit 0: flags, bit 2: B.
    pub(crate) const BRKP: u32 = 9;
    pub(crate) const BRKN: u32 = 10;
    /// Return the number of active elements of Pn under Pg.
    pub(crate) const CNTP: u32 = 11;
    /// WHILELT and friends: Pd gets the low x (or high x, data bit 0) elements set.
    pub(crate) const WHILE: u32 = 12;
    /// CTERMEQ and CTERMNE on x and y; data bit 0: NE.
    pub(crate) const CTERM: u32 = 13;
    /// PUNPKLO and PUNPKHI; data bit 0: high.
    pub(crate) const PUNPK: u32 = 14;
    pub(crate) const REV: u32 = 15;
    pub(crate) const ZIP1: u32 = 16;
    pub(crate) const ZIP2: u32 = 17;
    pub(crate) const UZP1: u32 = 18;
    pub(crate) const UZP2: u32 = 19;
    pub(crate) const TRN1: u32 = 20;
    pub(crate) const TRN2: u32 = 21;
    pub(crate) const PFALSE: u32 = 22;
    /// `do_sat_addsub_32()` and `do_sat_addsub_64()`: the saturating add of `y` to (data bit
    /// 1: subtract from) the general register value `x`; bit 0: unsigned, bit 2: 64-bit.
    pub(crate) const SATR: u32 = 23;
}

def!(SVE, "sve", 0, I64, [Ptr, I64, I64, I64], h_sve);
def!(SVE_MEM, "sve_mem", 0, I64, [Ptr, I64, I64, I64], h_sve_mem);

/// The helpers of this module.
pub(crate) const ALL: &[Def] = &[SVE, SVE_MEM];

// Element and register access.

fn mask(esz: u32) -> u64 {
    if esz >= 3 { u64::MAX } else { (1u64 << (8 << esz)) - 1 }
}

fn sext(x: u64, esz: u32) -> i64 {
    let sh = 64 - (8 << esz);
    ((x << sh) as i64) >> sh
}

/// Element `i` of `z`, zero extended.
pub(crate) fn get(z: &[u8], esz: u32, i: usize) -> u64 {
    let n = 1usize << esz;
    let o = i << esz;
    let mut b = [0u8; 8];
    b[..n].copy_from_slice(&z[o..o + n]);
    u64::from_le_bytes(b)
}

/// Element `i` of `z`, sign extended.
fn gets(z: &[u8], esz: u32, i: usize) -> i64 {
    sext(get(z, esz, i), esz)
}

/// Set element `i` of `z` to the low bits of `v`.
pub(crate) fn set(z: &mut [u8], esz: u32, i: usize, v: u64) {
    let n = 1usize << esz;
    let o = i << esz;
    z[o..o + n].copy_from_slice(&v.to_le_bytes()[..n]);
}

pub(crate) fn zload(env: &[u8], r: usize) -> Z {
    let o = vreg_off(r & 31);
    let mut z = [0u8; ZREG_SIZE];
    z.copy_from_slice(&env[o..o + ZREG_SIZE]);
    z
}

/// Write the first `vl` bytes of `z` to Zr. The rest of the register is left zero.
pub(crate) fn zstore(env: &mut [u8], r: usize, z: &Z, vl: usize) {
    let o = vreg_off(r & 31);
    env[o..o + vl].copy_from_slice(&z[..vl]);
}

pub(crate) fn pload(env: &[u8], r: usize) -> P {
    let o = preg_off(r);
    let mut p = [0u64; PREG_SIZE / 8];
    for (i, w) in p.iter_mut().enumerate() {
        let mut b = [0u8; 8];
        b.copy_from_slice(&env[o + 8 * i..o + 8 * i + 8]);
        *w = u64::from_le_bytes(b);
    }
    p
}

/// Write the first `vl / 8` bytes of `p` to predicate register `r`.
pub(crate) fn pstore(env: &mut [u8], r: usize, p: &P, vl: usize) {
    let o = preg_off(r);
    let mut b = [0u8; PREG_SIZE];
    for (i, w) in p.iter().enumerate() {
        b[8 * i..8 * i + 8].copy_from_slice(&w.to_le_bytes());
    }
    env[o..o + vl / 8].copy_from_slice(&b[..vl / 8]);
}

fn st32(env: &mut [u8], off: usize, v: u32) {
    env[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn rd32(env: &[u8], off: usize) -> u32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&env[off..off + 4]);
    u32::from_le_bytes(b)
}

/// Bit `b` of a predicate.
pub(crate) fn pbit(p: &P, b: usize) -> bool {
    (p[b / 64] >> (b % 64)) & 1 != 0
}

fn pset(p: &mut P, b: usize, v: bool) {
    if v {
        p[b / 64] |= 1 << (b % 64);
    } else {
        p[b / 64] &= !(1 << (b % 64));
    }
}

/// Whether element `i` of size `esz` is active in `p`.
pub(crate) fn act(p: &P, esz: u32, i: usize) -> bool {
    pbit(p, i << esz)
}

/// `pred_esz_masks[]`.
fn esz_mask(esz: u32) -> u64 {
    [u64::MAX, 0x5555_5555_5555_5555, 0x1111_1111_1111_1111, 0x0101_0101_0101_0101][esz as usize]
}

/// The number of predicate words that hold `vl / 8` bits.
fn pwords(vl: usize) -> usize {
    vl.div_ceil(64)
}

/// The predicate with every bit of an element of size `esz` set, for `vl` bytes.
fn ptrue(vl: usize, esz: u32) -> P {
    let mut p = [0u64; PREG_SIZE / 8];
    for i in 0..vl {
        if i % (1 << esz) == 0 {
            pset(&mut p, i, true);
        }
    }
    p
}

/// `iter_predtest_fwd()` over the words of `d` and `g`.
fn predtest(d: &P, g: &P, words: usize) -> u32 {
    let mut flags = 1u32;
    for i in 0..words {
        let (d, g) = (d[i], g[i]);
        if g != 0 {
            if flags & 4 == 0 {
                flags |= u32::from(d & (g & g.wrapping_neg()) != 0) << 31;
                flags |= 4;
            }
            flags |= u32::from(d & g != 0) << 1;
            let top = 1u64 << (63 - g.leading_zeros());
            flags = (flags & !1) | u32::from(d & top == 0);
        }
    }
    flags
}

/// `do_pred_flags()`: set NZCV from the result of [`predtest`].
fn set_pred_flags(env: &mut [u8], t: u32) {
    st32(env, NF, t);
    st32(env, ZF, t & 2);
    st32(env, CF, t & 1);
    st32(env, VF, 0);
}

// Arithmetic.

fn smax_of(esz: u32) -> i128 {
    i128::from(i64::MAX >> (64 - (8 << esz)))
}

fn smin_of(esz: u32) -> i128 {
    -smax_of(esz) - 1
}

fn ssat(x: i128, esz: u32) -> u64 {
    x.clamp(smin_of(esz), smax_of(esz)) as u64 & mask(esz)
}

fn usat(x: i128, esz: u32) -> u64 {
    x.clamp(0, i128::from(mask(esz))) as u64
}

/// A shift count from a signed element, clamped to a range that gives the same results.
fn shv(x: i64) -> i32 {
    x.clamp(-1000, 1000) as i32
}

/// `do_sqrdmlah_[bhsd]()`: `(a << (bits - 1) +/- n * m + round) >> (bits - 1)`, saturated.
pub(crate) fn sqrdmlah(n: i64, m: i64, a: i64, neg: bool, round: bool, esz: u32) -> u64 {
    let bits = 8 << esz;
    let mut r = i128::from(n) * i128::from(m);
    if neg {
        r = -r;
    }
    r += (i128::from(a) << (bits - 1)) + (i128::from(round) << (bits - 2));
    ssat(r >> (bits - 1), esz)
}

/// Carry-less multiply of the low `bits` bits of `a` and `b`.
fn clmul(a: u64, b: u64, bits: u32) -> u128 {
    let mut r = 0u128;
    for i in 0..bits {
        if (b >> i) & 1 != 0 {
            r ^= u128::from(a) << i;
        }
    }
    r
}

/// `bitextract()`: gather the bits of `data` selected by `m` to the bottom.
fn bitextract(data: u64, m: u64, bits: u32) -> u64 {
    let mut res = 0;
    let mut rb = 0;
    for db in 0..bits {
        if (m >> db) & 1 != 0 {
            res |= ((data >> db) & 1) << rb;
            rb += 1;
        }
    }
    res
}

/// `bitdeposit()`: scatter the bottom bits of `data` to the bits selected by `m`.
fn bitdeposit(data: u64, m: u64, bits: u32) -> u64 {
    let mut res = 0;
    let mut db = 0;
    for rb in 0..bits {
        if (m >> rb) & 1 != 0 {
            res |= ((data >> db) & 1) << rb;
            db += 1;
        }
    }
    res
}

/// `bitgroup()`: the bits of `data` selected by `m` at the bottom, the rest above them.
fn bitgroup(data: u64, m: u64, bits: u32) -> u64 {
    let msk = if bits == 64 { u64::MAX } else { (1 << bits) - 1 };
    let lo = bitextract(data, m, bits);
    let hi = bitextract(data, !m & msk, bits);
    let n = (m & msk).count_ones();
    if n >= 64 { lo } else { lo | (hi << n) }
}

/// The element-wise binary operation `op` on elements of size `esz`; `d` is the old value
/// of the destination, for the accumulating forms.
pub(crate) fn bin(op: u32, esz: u32, n: u64, m: u64, d: u64) -> u64 {
    let bits = 8u32 << esz;
    let msk = mask(esz);
    let raw = m;
    let (n, m, d) = (n & msk, m & msk, d & msk);
    let (sn, sm) = (sext(n, esz), sext(m, esz));
    let (wn, wm) = (i128::from(sn), i128::from(sm));
    let (un, um) = (i128::from(n), i128::from(m));
    let mut dummy = false;
    let r = match op {
        b::ADD => n.wrapping_add(m),
        b::SUB => n.wrapping_sub(m),
        b::SUBR => m.wrapping_sub(n),
        b::SQADD => ssat(wn + wm, esz),
        b::UQADD => usat(un + um, esz),
        b::SQSUB => ssat(wn - wm, esz),
        b::UQSUB => usat(un - um, esz),
        b::AND => n & m,
        b::ORR => n | m,
        b::EOR => n ^ m,
        b::BIC => n & !m,
        b::MUL => n.wrapping_mul(m),
        b::SMULH => ((wn * wm) >> bits) as u64,
        b::UMULH => ((u128::from(n) * u128::from(m)) >> bits) as u64,
        b::PMUL => clmul(n, m, bits) as u64,
        b::SQDMULH => sqrdmlah(sn, sm, 0, false, false, esz),
        b::SQRDMULH => sqrdmlah(sn, sm, 0, false, true, esz),
        b::SQRDMLAH => sqrdmlah(sn, sm, sext(d, esz), false, true, esz),
        b::SQRDMLSH => sqrdmlah(sn, sm, sext(d, esz), true, true, esz),
        b::MLA => d.wrapping_add(n.wrapping_mul(m)),
        b::MLS => d.wrapping_sub(n.wrapping_mul(m)),
        b::SMAX => sn.max(sm) as u64,
        b::UMAX => n.max(m),
        b::SMIN => sn.min(sm) as u64,
        b::UMIN => n.min(m),
        b::SABD => (wn - wm).unsigned_abs() as u64,
        b::UABD => n.abs_diff(m),
        b::SDIV => {
            if m == 0 {
                0
            } else {
                sn.wrapping_div(sm) as u64
            }
        }
        b::UDIV => n.checked_div(m).unwrap_or(0),
        b::ASR | b::ASR_I => (sn >> m.min(u64::from(bits) - 1)) as u64,
        b::LSR | b::LSR_I => {
            if m >= u64::from(bits) {
                0
            } else {
                n >> m
            }
        }
        b::LSL => {
            if m >= u64::from(bits) {
                0
            } else {
                n << m
            }
        }
        b::SRSHL => sqrshl(sn, shv(sm), bits as i32, true, None) as u64,
        b::URSHL => uqrshl(n, shv(sm), bits as i32, true, None),
        b::SQSHL => sqrshl(sn, shv(sm), bits as i32, false, Some(&mut dummy)) as u64,
        b::UQSHL => uqrshl(n, shv(sm), bits as i32, false, Some(&mut dummy)),
        b::SQRSHL => sqrshl(sn, shv(sm), bits as i32, true, Some(&mut dummy)) as u64,
        b::UQRSHL => uqrshl(n, shv(sm), bits as i32, true, Some(&mut dummy)),
        b::SHADD => ((wn + wm) >> 1) as u64,
        b::UHADD => ((un + um) >> 1) as u64,
        b::SHSUB => ((wn - wm) >> 1) as u64,
        b::UHSUB => ((un - um) >> 1) as u64,
        b::SRHADD => ((wn + wm + 1) >> 1) as u64,
        b::URHADD => ((un + um + 1) >> 1) as u64,
        b::SUQADD => ssat(wn + um, esz),
        b::USQADD => usat(un + wm, esz),
        b::ASRD => {
            let sh = m.min(u64::from(bits)) as u32;
            let x = if sn < 0 { wn + (1i128 << sh) - 1 } else { wn };
            (x >> sh) as u64
        }
        b::SRSHR => sqrshl(sn, -(m as i32), bits as i32, true, None) as u64,
        b::URSHR => uqrshl(n, -(m as i32), bits as i32, true, None),
        b::SQSHLU => suqrshl(sn, m as i32, bits as i32, false, &mut dummy),
        b::SABA => d.wrapping_add((wn - wm).unsigned_abs() as u64),
        b::UABA => d.wrapping_add(n.abs_diff(m)),
        b::SSRA => d.wrapping_add((sn >> m.min(u64::from(bits) - 1)) as u64),
        b::USRA => d.wrapping_add(if m >= u64::from(bits) { 0 } else { n >> m }),
        b::SRSRA => d.wrapping_add(sqrshl(sn, -(m as i32), bits as i32, true, None) as u64),
        b::URSRA => d.wrapping_add(uqrshl(n, -(m as i32), bits as i32, true, None)),
        b::SRI => {
            if m >= u64::from(bits) {
                d
            } else {
                (d & !(msk >> m)) | (n >> m)
            }
        }
        b::SLI => (d & !(msk << m)) | (n << m),
        b::BEXT => bitextract(n, m, bits),
        b::BDEP => bitdeposit(n, m, bits),
        b::BGRP => bitgroup(n, m, bits),
        b::MOVM => n,
        b::MOV2 => m,
        b::SADALP => {
            let h = esz - 1;
            let lo = sext(n & mask(h), h);
            let hi = sext((n >> (bits / 2)) & mask(h), h);
            m.wrapping_add(lo as u64).wrapping_add(hi as u64)
        }
        b::UADALP => {
            let h = esz - 1;
            m.wrapping_add(n & mask(h)).wrapping_add((n >> (bits / 2)) & mask(h))
        }
        b::SQADDI => ssat(wn + i128::from(raw as i64), esz),
        b::UQADDI if esz == 3 => usat(un + i128::from(raw), esz),
        b::UQADDI => usat(un + i128::from(raw as i64), esz),
        b::UQSUBI => usat(un - i128::from(raw), esz),
        b::SCLAMP => sext(d, esz).max(sn).min(sm) as u64,
        b::UCLAMP => d.max(n).min(m),
        _ => unreachable!("bad sve binary op {op}"),
    };
    r & msk
}

/// The unary operation `op` on an element of size `esz`.
fn un(op: u32, esz: u32, n: u64) -> u64 {
    let bits = 8u32 << esz;
    let msk = mask(esz);
    let n = n & msk;
    let sn = sext(n, esz);
    let r = match op {
        u::CLS => {
            let v = if sn < 0 { !n & msk } else { n };
            u64::from(v.leading_zeros() - (64 - bits) - 1)
        }
        u::CLZ => u64::from(n.leading_zeros() - (64 - bits)),
        u::CNT => u64::from(n.count_ones()),
        u::CNOT => u64::from(n == 0),
        u::NOT => !n,
        u::FABS => n & (msk >> 1),
        u::FNEG => n ^ !(msk >> 1),
        u::ABS => sn.wrapping_abs() as u64,
        u::NEG => sn.wrapping_neg() as u64,
        u::SXTB => sext(n, 0) as u64,
        u::UXTB => n & 0xff,
        u::SXTH => sext(n, 1) as u64,
        u::UXTH => n & 0xffff,
        u::SXTW => sext(n, 2) as u64,
        u::UXTW => n & 0xffff_ffff,
        u::REVB => n.swap_bytes() >> (64 - bits),
        u::REVH => {
            let mut r = 0;
            for i in 0..bits / 16 {
                r |= ((n >> (16 * i)) & 0xffff) << (bits - 16 - 16 * i);
            }
            r
        }
        u::REVW => n.rotate_left(32),
        u::RBIT => n.reverse_bits() >> (64 - bits),
        u::SQABS => ssat(i128::from(sn).abs(), esz),
        u::SQNEG => ssat(-i128::from(sn), esz),
        u::URECPE => u64::from(super::vfp::recpe_u32(n as u32)),
        u::URSQRTE => u64::from(super::vfp::rsqrte_u32(n as u32)),
        u::MOV => n,
        _ => unreachable!("bad sve unary op {op}"),
    };
    r & msk
}

/// The reduction `op` of the active elements of `n`.
fn reduce(op: u32, esz: u32, n: &Z, g: &P, elems: usize) -> u64 {
    let msk = mask(esz);
    let mut acc: i128 = match op {
        r::ANDV => i128::from(msk),
        r::SMAXV => smin_of(esz),
        r::UMAXV | r::ORV | r::EORV | r::UADDV | r::SADDV => 0,
        r::SMINV => smax_of(esz),
        r::UMINV => i128::from(msk),
        _ => unreachable!("bad sve reduction {op}"),
    };
    for i in 0..elems {
        if !act(g, esz, i) {
            continue;
        }
        let x = get(n, esz, i);
        let s = i128::from(sext(x, esz));
        let x = i128::from(x);
        acc = match op {
            r::ORV => acc | x,
            r::EORV => acc ^ x,
            r::ANDV => acc & x,
            r::UADDV => acc + x,
            r::SADDV => acc + s,
            r::SMAXV => acc.max(s),
            r::UMAXV => acc.max(x),
            r::SMINV => acc.min(s),
            _ => acc.min(x),
        };
    }
    if matches!(op, r::UADDV | r::SADDV) { acc as u64 } else { acc as u64 & msk }
}

fn h_sve(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    let d = Dsc::unpack(a[1]);
    let env = &mut *h.env;
    let r = match d.fam {
        fam::ZZZ | fam::ZPZZ | fam::ZPZI | fam::ZZI => zbin(env, &d, a[2]),
        fam::ZPZ => {
            let n = zload(env, d.n);
            let g = pload(env, d.g);
            let mut out = zload(env, d.d);
            for i in 0..d.elems() {
                if act(&g, d.esz, i) {
                    set(&mut out, d.esz, i, un(d.op, d.esz, get(&n, d.esz, i)));
                }
            }
            zstore(env, d.d, &out, d.vl);
            0
        }
        fam::RED => {
            let n = zload(env, d.n);
            let g = pload(env, d.g);
            let v = reduce(d.op, d.esz, &n, &g, d.elems());
            let mut out = [0u8; ZREG_SIZE];
            out[..8].copy_from_slice(&v.to_le_bytes());
            zstore(env, d.d, &out, d.vl);
            0
        }
        fam::CMP => {
            cmp(env, &d, a[2]);
            0
        }
        fam::PPPP => {
            pppp(env, &d);
            0
        }
        fam::PERM => perm(env, &d, a[2], a[3]),
        fam::PRED => pred(env, &d, a[2], a[3]),
        fam::FP => {
            super::sve_fp::fp(env, &d, a[2]);
            0
        }
        fam::CRYPTO => {
            crypto(env, &d);
            0
        }
        _ => unreachable!("bad sve family {}", d.fam),
    };
    Ok(u128::from(r))
}

/// The CRYPTO family.
fn crypto(env: &mut [u8], d: &Dsc) {
    let n = zload(env, d.n);
    let m = zload(env, d.m);
    let mut out = zload(env, d.d);
    let seg = |z: &Z, s: usize| -> [u8; 16] { z[16 * s..16 * s + 16].try_into().unwrap() };
    let zero = [0u8; 16];
    match d.op {
        cr::PMULL => {
            let sel = (d.data & 1) as usize;
            match d.esz {
                0 => {
                    for s in 0..d.vl / 16 {
                        let r = clmul(get(&n, 3, 2 * s + sel), get(&m, 3, 2 * s + sel), 64);
                        set(&mut out, 3, 2 * s, r as u64);
                        set(&mut out, 3, 2 * s + 1, (r >> 64) as u64);
                    }
                }
                1 => {
                    for i in 0..d.vl / 2 {
                        let r = clmul(get(&n, 0, 2 * i + sel), get(&m, 0, 2 * i + sel), 8);
                        set(&mut out, 1, i, r as u64);
                    }
                }
                _ => {
                    for i in 0..d.vl / 8 {
                        let r = clmul(get(&n, 2, 2 * i + sel), get(&m, 2, 2 * i + sel), 32);
                        set(&mut out, 3, i, r as u64);
                    }
                }
            }
        }
        cr::RAX1 => {
            for i in 0..d.vl / 8 {
                set(&mut out, 3, i, get(&n, 3, i) ^ get(&m, 3, i).rotate_left(1));
            }
        }
        op => {
            for s in 0..d.vl / 16 {
                let (sn, sm) = (seg(&n, s), seg(&m, s));
                let r = match op {
                    cr::AESE => super::crypto::crypto(super::crypto::op::AESE, &sn, &sm, &zero),
                    cr::AESD => super::crypto::crypto(super::crypto::op::AESD, &sn, &sm, &zero),
                    cr::AESMC => super::crypto::crypto(super::crypto::op::AESMC, &zero, &sn, &zero),
                    cr::AESIMC => {
                        super::crypto::crypto(super::crypto::op::AESIMC, &zero, &sn, &zero)
                    }
                    _ => super::crypto::sm4(op == cr::SM4EKEY, &sn, &sm),
                };
                out[16 * s..16 * s + 16].copy_from_slice(&r);
            }
        }
    }
    zstore(env, d.d, &out, d.vl);
}

/// The ZZZ, ZPZZ, ZPZI and ZZI families.
fn zbin(env: &mut [u8], d: &Dsc, x: u64) -> u64 {
    let n = zload(env, d.n);
    let m = zload(env, d.m);
    let g = pload(env, d.g);
    let mut out = zload(env, d.d);
    // The accumulating forms with a separate addend (MAD and MSB) read it from Za.
    let acc = if d.data & 2 != 0 { zload(env, d.a) } else { out };
    let pred = matches!(d.fam, fam::ZPZZ | fam::ZPZI);
    let imm = matches!(d.fam, fam::ZPZI | fam::ZZI);
    let wide = d.data & 1 != 0;
    for i in 0..d.elems() {
        if pred && !act(&g, d.esz, i) {
            continue;
        }
        let mm = if imm {
            x
        } else if wide {
            get(&m, 3, (i << d.esz) / 8)
        } else {
            get(&m, d.esz, i)
        };
        let v = if wide {
            // The shifts by a 64-bit element see the whole of it.
            let bits = 8u64 << d.esz;
            let nn = get(&n, d.esz, i);
            match d.op {
                b::ASR => (gets(&n, d.esz, i) >> mm.min(bits - 1)) as u64,
                b::LSR => {
                    if mm >= bits {
                        0
                    } else {
                        nn >> mm
                    }
                }
                _ => {
                    if mm >= bits {
                        0
                    } else {
                        nn << mm
                    }
                }
            }
        } else {
            bin(d.op, d.esz, get(&n, d.esz, i), mm, get(&acc, d.esz, i))
        };
        set(&mut out, d.esz, i, v);
    }
    zstore(env, d.d, &out, d.vl);
    0
}

/// The CMP family: compare the active elements of Zn with Zm, the 64-bit column of Zm or x.
fn cmp(env: &mut [u8], d: &Dsc, x: u64) {
    let n = zload(env, d.n);
    let m = zload(env, d.m);
    let g = pload(env, d.g);
    let wide = d.data & 1 != 0;
    let imm = d.data & 2 != 0;
    let mut out = [0u64; PREG_SIZE / 8];
    let signed = matches!(d.op, c::GE | c::GT | c::LT | c::LE);
    for i in 0..d.elems() {
        if !act(&g, d.esz, i) {
            continue;
        }
        let (nn, mm) = if wide {
            let mm = get(&m, 3, (i << d.esz) / 8);
            if signed {
                (gets(&n, d.esz, i) as i128, i128::from(mm as i64))
            } else {
                (i128::from(get(&n, d.esz, i)), i128::from(mm))
            }
        } else {
            let mm = if imm { x & mask(d.esz) } else { get(&m, d.esz, i) };
            if signed {
                (i128::from(gets(&n, d.esz, i)), i128::from(sext(mm, d.esz)))
            } else {
                (i128::from(get(&n, d.esz, i)), i128::from(mm))
            }
        };
        let t = match d.op {
            c::EQ => nn == mm,
            c::NE => nn != mm,
            c::GE | c::HS => nn >= mm,
            c::GT | c::HI => nn > mm,
            c::LT | c::LO => nn < mm,
            _ => nn <= mm,
        };
        pset(&mut out, i << d.esz, t);
    }
    pstore(env, d.d, &out, d.vl);
    let mut gm = g;
    for w in &mut gm {
        *w &= esz_mask(d.esz);
    }
    let t = predtest(&out, &gm, pwords(d.vl));
    set_pred_flags(env, t);
}

/// The PPPP family.
fn pppp(env: &mut [u8], d: &Dsc) {
    let n = pload(env, d.n);
    let m = pload(env, d.m);
    let g = pload(env, d.g);
    let mut out = [0u64; PREG_SIZE / 8];
    for i in 0..out.len() {
        let (n, m, g) = (n[i], m[i], g[i]);
        out[i] = match d.op {
            pl::AND => n & m & g,
            pl::BIC => n & !m & g,
            pl::EOR => (n ^ m) & g,
            pl::SEL => (n & g) | (m & !g),
            pl::ORR => (n | m) & g,
            pl::ORN => (n | !m) & g,
            pl::NOR => !(n | m) & g,
            _ => !(n & m) & g,
        };
    }
    trim(&mut out, d.vl);
    pstore(env, d.d, &out, d.vl);
    if d.data & 1 != 0 {
        let t = predtest(&out, &g, pwords(d.vl));
        set_pred_flags(env, t);
    }
}

/// Clear the bits of `p` above `vl / 8`.
fn trim(p: &mut P, vl: usize) {
    for b in vl..PREG_SIZE * 8 {
        pset(p, b, false);
    }
}

/// The index of the last active element of `g`, if any.
fn last_active(g: &P, esz: u32, elems: usize) -> Option<usize> {
    (0..elems).rev().find(|&i| act(g, esz, i))
}

/// The index of the first active element of `g`, if any.
fn first_active(g: &P, esz: u32, elems: usize) -> Option<usize> {
    (0..elems).find(|&i| act(g, esz, i))
}

/// The PERM family.
fn perm(env: &mut [u8], d: &Dsc, x: u64, y: u64) -> u64 {
    let esz = d.esz;
    let elems = d.elems();
    let vl = d.vl;
    let n = zload(env, d.n);
    let m = zload(env, d.m);
    let g = pload(env, d.g);
    let old = zload(env, d.d);
    let mut out = [0u8; ZREG_SIZE];
    match d.op {
        pm::DUP => {
            for i in 0..elems {
                set(&mut out, esz, i, x);
            }
        }
        pm::DUPX => {
            let idx = x as usize;
            let v = if idx < elems { get(&n, esz, idx) } else { 0 };
            for i in 0..elems {
                set(&mut out, esz, i, v);
            }
        }
        pm::INDEX => {
            for i in 0..elems {
                set(&mut out, esz, i, x.wrapping_add((i as u64).wrapping_mul(y)));
            }
        }
        pm::INSR => {
            let v = if d.data & 1 != 0 { get(&m, esz, 0) } else { x };
            set(&mut out, esz, 0, v);
            for i in 1..elems {
                set(&mut out, esz, i, get(&n, esz, i - 1));
            }
        }
        pm::REV => {
            for i in 0..elems {
                set(&mut out, esz, i, get(&n, esz, elems - 1 - i));
            }
        }
        pm::TBL | pm::TBL2 | pm::TBX => {
            let n2 = zload(env, (d.n + 1) & 31);
            for i in 0..elems {
                let idx = get(&m, esz, i);
                let v = if idx < elems as u64 {
                    get(&n, esz, idx as usize)
                } else if d.op == pm::TBL2 && idx < 2 * elems as u64 {
                    get(&n2, esz, idx as usize - elems)
                } else if d.op == pm::TBX {
                    get(&old, esz, i)
                } else {
                    0
                };
                set(&mut out, esz, i, v);
            }
        }
        pm::ZIP1 | pm::ZIP2 | pm::UZP1 | pm::UZP2 | pm::TRN1 | pm::TRN2 => {
            for i in 0..elems {
                let v = permute(d.op - pm::ZIP1, elems, i, |src, j| {
                    get(if src { &m } else { &n }, esz, j)
                });
                set(&mut out, esz, i, v);
            }
        }
        pm::EXT | pm::EXT2 => {
            let second = if d.op == pm::EXT { m } else { zload(env, (d.n + 1) & 31) };
            let ofs = if x as usize >= vl { 0 } else { x as usize };
            for i in 0..vl {
                let j = i + ofs;
                out[i] = if j < vl { n[j] } else { second[j - vl] };
            }
        }
        pm::SPLICE | pm::SPLICE2 => {
            let second = if d.op == pm::SPLICE { m } else { zload(env, (d.n + 1) & 31) };
            let mut k = 0;
            if let (Some(f), Some(l)) = (first_active(&g, esz, elems), last_active(&g, esz, elems))
            {
                for i in f..=l {
                    set(&mut out, esz, k, get(&n, esz, i));
                    k += 1;
                }
            }
            for j in 0..elems - k {
                set(&mut out, esz, k + j, get(&second, esz, j));
            }
        }
        pm::COMPACT => {
            let mut k = 0;
            for i in 0..elems {
                if act(&g, esz, i) {
                    set(&mut out, esz, k, get(&n, esz, i));
                    k += 1;
                }
            }
        }
        pm::SEL => {
            for i in 0..elems {
                let v = if act(&g, esz, i) { get(&n, esz, i) } else { get(&m, esz, i) };
                set(&mut out, esz, i, v);
            }
        }
        pm::UNPK => {
            let h = esz - 1;
            let base = if d.data & 1 != 0 { elems } else { 0 };
            for i in 0..elems {
                let v = get(&n, h, base + i);
                let v = if d.data & 2 != 0 { v } else { sext(v, h) as u64 };
                set(&mut out, esz, i, v);
            }
        }
        pm::CLASTA_Z | pm::CLASTB_Z => {
            out = n;
            if let Some(l) = last_active(&g, esz, elems) {
                let idx = if d.op == pm::CLASTB_Z { l } else { (l + 1) % elems };
                let v = get(&m, esz, idx);
                for i in 0..elems {
                    set(&mut out, esz, i, v);
                }
            }
        }
        pm::CLAST | pm::LAST => {
            let before = d.data & 2 != 0;
            let to_v = d.data & 1 != 0;
            let last = last_active(&g, esz, elems);
            let v = match (d.op, last) {
                (pm::CLAST, None) => {
                    if to_v {
                        get(&old, esz, 0)
                    } else {
                        x & mask(esz)
                    }
                }
                (pm::CLAST, Some(l)) | (_, Some(l)) => {
                    get(&n, esz, if before { l } else { (l + 1) % elems })
                }
                (_, None) => get(&n, esz, if before { elems - 1 } else { 0 }),
            };
            if !to_v {
                return v;
            }
            out[..8].copy_from_slice(&v.to_le_bytes());
        }
        pm::CPY_M | pm::CPY_Z => {
            let v = if d.data & 1 != 0 { get(&n, esz, 0) } else { x };
            for i in 0..elems {
                let e = if act(&g, esz, i) {
                    v
                } else if d.op == pm::CPY_M {
                    get(&old, esz, i)
                } else {
                    0
                };
                set(&mut out, esz, i, e);
            }
        }
        pm::ADR => {
            for i in 0..elems {
                let off = match d.data {
                    0 => get(&m, 3, i) as i32 as i64 as u64,
                    1 => get(&m, 3, i) & 0xffff_ffff,
                    _ => get(&m, esz, i),
                };
                set(&mut out, esz, i, get(&n, esz, i).wrapping_add(off << x));
            }
        }
        pm::MOVPRFX => out = n,
        pm::MOVPRFX_Z | pm::MOVPRFX_M => {
            for i in 0..elems {
                let v = if act(&g, esz, i) {
                    get(&n, esz, i)
                } else if d.op == pm::MOVPRFX_M {
                    get(&old, esz, i)
                } else {
                    0
                };
                set(&mut out, esz, i, v);
            }
        }
        pm::EOR3 | pm::BCAX | pm::BSL | pm::BSL1N | pm::BSL2N | pm::NBSL => {
            let k = zload(env, d.a);
            for i in 0..vl {
                let (n, m, k) = (n[i], m[i], k[i]);
                out[i] = match d.op {
                    pm::EOR3 => n ^ m ^ k,
                    pm::BCAX => n ^ (m & !k),
                    pm::BSL => (n & k) | (m & !k),
                    pm::BSL1N => (!n & k) | (m & !k),
                    pm::BSL2N => (n & k) | (!m & !k),
                    _ => !((n & k) | (m & !k)),
                };
            }
        }
        pm::XAR => {
            let bits = 8u32 << esz;
            for i in 0..elems {
                let v = get(&n, esz, i) ^ get(&m, esz, i);
                let sh = x as u32 % bits;
                let r = if sh == 0 { v } else { (v >> sh) | (v << (bits - sh)) };
                set(&mut out, esz, i, r);
            }
        }
        pm::HISTSEG => {
            for i in 0..vl {
                let seg = i & !15;
                let cnt = (0..16).filter(|&j| m[seg + j] == n[i]).count();
                out[i] = cnt as u8;
            }
        }
        pm::HISTCNT => {
            for i in 0..elems {
                if !act(&g, esz, i) {
                    continue;
                }
                let nn = get(&n, esz, i);
                let cnt = (0..=i).filter(|&j| act(&g, esz, j) && get(&m, esz, j) == nn).count();
                set(&mut out, esz, i, cnt as u64);
            }
        }
        pm::MATCH | pm::NMATCH => {
            let mut p = [0u64; PREG_SIZE / 8];
            let per = 16 >> esz;
            for i in 0..elems {
                if !act(&g, esz, i) {
                    continue;
                }
                let seg = i / per * per;
                let nn = get(&n, esz, i);
                let hit = (seg..seg + per).any(|j| get(&m, esz, j) == nn);
                pset(&mut p, i << esz, hit != (d.op == pm::NMATCH));
            }
            pstore(env, d.d, &p, vl);
            let mut gm = g;
            for w in &mut gm {
                *w &= esz_mask(esz);
            }
            let t = predtest(&p, &gm, pwords(vl));
            set_pred_flags(env, t);
            return 0;
        }
        _ => unreachable!("bad sve permute op {}", d.op),
    }
    zstore(env, d.d, &out, vl);
    0
}

/// Element `i` of the result of ZIP1 (`k` 0), ZIP2, UZP1, UZP2, TRN1 and TRN2 (`k` 5), on
/// vectors of `elems` elements; `src(true, j)` reads element `j` of the second operand.
fn permute(k: u32, elems: usize, i: usize, src: impl Fn(bool, usize) -> u64) -> u64 {
    let half = elems / 2;
    match k {
        0 | 1 => {
            let base = if k == 1 { half } else { 0 };
            src(i & 1 != 0, base + i / 2)
        }
        2 | 3 => {
            let j = 2 * i + (k as usize - 2);
            if j < elems { src(false, j) } else { src(true, j - elems) }
        }
        _ => {
            let odd = (k - 4) as usize;
            src(i & 1 != 0, (i & !1) + odd)
        }
    }
}

/// `decode_pred_count()`: the number of elements the predicate pattern `pat` selects.
pub(crate) fn pred_count(vl: usize, pat: u32, esz: u32) -> usize {
    let elements = vl >> esz;
    let bound = match pat {
        0 => return 1usize << (usize::BITS - 1 - elements.leading_zeros()),
        1..=8 => pat as usize,
        9..=13 => 16usize << (pat - 9),
        29 => return elements - elements % 4,
        30 => return elements - elements % 3,
        31 => return elements,
        _ => return 0,
    };
    if bound <= elements { bound } else { 0 }
}

/// Element `i` of size `1 << esz` bits of a predicate.
fn pget(p: &P, esz: u32, i: usize) -> u64 {
    let w = 1usize << esz;
    let mut v = 0;
    for k in 0..w {
        v |= u64::from(pbit(p, i * w + k)) << k;
    }
    v
}

fn pput(p: &mut P, esz: u32, i: usize, v: u64) {
    let w = 1usize << esz;
    for k in 0..w {
        pset(p, i * w + k, (v >> k) & 1 != 0);
    }
}

/// `last_active_pred()`: whether the last active element of `g` is true in `n`.
fn last_active_pred(n: &P, g: &P, vl: usize) -> bool {
    (0..vl).rev().find(|&b| pbit(g, b)).is_some_and(|b| pbit(n, b))
}

/// `compute_brk()` over the whole predicate: the mask that is true for all of G up to and
/// including (if `after`) or excluding the first G & N.
fn brk_mask(n: &P, g: &P, vl: usize, after: bool) -> P {
    let mut b = [0u64; PREG_SIZE / 8];
    let mut brk = false;
    for i in 0..vl {
        if !pbit(g, i) {
            continue;
        }
        if brk {
            break;
        }
        if pbit(n, i) {
            brk = true;
            if !after {
                break;
            }
        }
        pset(&mut b, i, true);
    }
    b
}

/// The PRED family.
fn pred(env: &mut [u8], d: &Dsc, x: u64, y: u64) -> u64 {
    let esz = d.esz;
    let vl = d.vl;
    let words = pwords(vl);
    let elems = d.elems();
    let n = pload(env, d.n);
    let m = pload(env, d.m);
    let g = pload(env, d.g);
    let old = pload(env, d.d);
    // WHILE always sets the flags; its data bit 0 is the greater-than form.
    let flags = d.data & 1 != 0 || d.op == pr::WHILE;
    let mut out = [0u64; PREG_SIZE / 8];
    // The governing predicate of the flags, if any.
    let mut fg: Option<P> = None;
    match d.op {
        pr::PTRUE => {
            let cnt = pred_count(vl, x as u32, esz);
            for i in 0..cnt {
                pset(&mut out, i << esz, true);
            }
            if flags {
                // QEMU's do_predset() sets C only for an empty result, not as PredTest does.
                let none = cnt == 0;
                st32(env, NF, if none { 0 } else { 0x8000_0000 });
                st32(env, ZF, u32::from(!none));
                st32(env, CF, u32::from(none));
                st32(env, VF, 0);
            }
        }
        pr::PFALSE => {}
        pr::PTEST => {
            let t = predtest(&n, &g, words);
            set_pred_flags(env, t);
            return 0;
        }
        pr::RDFFR => {
            out = pload(env, FFR);
            if d.data & 2 != 0 {
                for i in 0..out.len() {
                    out[i] &= g[i];
                }
            }
            fg = Some(g);
        }
        pr::WRFFR => {
            pstore(env, FFR, &n, vl);
            return 0;
        }
        pr::SETFFR => {
            pstore(env, FFR, &ptrue(vl, 0), vl);
            return 0;
        }
        pr::PFIRST => {
            out = old;
            if let Some(b) = (0..vl).find(|&b| pbit(&g, b)) {
                pset(&mut out, b, true);
            }
            fg = Some(g);
        }
        pr::PNEXT => {
            let mut gm = g;
            for w in &mut gm {
                *w &= esz_mask(esz);
            }
            let start = (0..vl).rev().find(|&b| pbit(&old, b) && b % (1 << esz) == 0);
            let from = start.map_or(0, |b| b + (1 << esz));
            if let Some(b) = (from..vl).find(|&b| pbit(&gm, b)) {
                pset(&mut out, b, true);
            }
            fg = Some(gm);
        }
        pr::BRK => {
            let after = d.data & 4 == 0;
            let b = brk_mask(&n, &g, vl, after);
            for i in 0..out.len() {
                out[i] =
                    if d.data & 2 != 0 { (b[i] & g[i]) | (old[i] & !g[i]) } else { b[i] & g[i] };
            }
            fg = Some(g);
        }
        pr::BRKP => {
            if last_active_pred(&n, &g, vl) {
                let b = brk_mask(&m, &g, vl, d.data & 4 == 0);
                for i in 0..out.len() {
                    out[i] = b[i] & g[i];
                }
            }
            fg = Some(g);
        }
        pr::BRKN => {
            if last_active_pred(&n, &g, vl) {
                out = old;
            }
            fg = Some(ptrue(vl, 0));
        }
        pr::CNTP => {
            let mut cnt = 0;
            for i in 0..elems {
                if act(&g, esz, i) && act(&n, esz, i) {
                    cnt += 1;
                }
            }
            return cnt;
        }
        pr::WHILE => {
            let cnt = while_count(d, elems, x, y);
            if d.data & 1 != 0 {
                for i in elems - cnt..elems {
                    pset(&mut out, i << esz, true);
                }
            } else {
                for i in 0..cnt {
                    pset(&mut out, i << esz, true);
                }
            }
            fg = Some(ptrue(vl, esz));
        }
        pr::CTERM => {
            let sf = d.data & 2 != 0;
            let (a, b) = if sf { (x, y) } else { (x & 0xffff_ffff, y & 0xffff_ffff) };
            let t = (a == b) != (d.data & 1 != 0);
            let cf = rd32(env, CF);
            let vf = !t && cf == 0;
            st32(env, NF, if t { 0x8000_0000 } else { 0 });
            st32(env, VF, if vf { 0x8000_0000 } else { 0 });
            return 0;
        }
        pr::SATR => return sat_addsub(x, y, d.data & 1 != 0, d.data & 2 != 0, d.data & 4 != 0),
        pr::PUNPK => {
            let base = if d.data & 1 != 0 { vl / 2 } else { 0 };
            for i in 0..vl / 2 {
                pset(&mut out, 2 * i, pbit(&n, base + i));
            }
        }
        pr::REV => {
            for i in 0..elems {
                pput(&mut out, esz, i, pget(&n, esz, elems - 1 - i));
            }
        }
        pr::ZIP1..=pr::TRN2 => {
            for i in 0..elems {
                let v = permute(d.op - pr::ZIP1, elems, i, |src, j| {
                    pget(if src { &m } else { &n }, esz, j)
                });
                pput(&mut out, esz, i, v);
            }
        }
        _ => unreachable!("bad sve predicate op {}", d.op),
    }
    trim(&mut out, vl);
    pstore(env, d.d, &out, vl);
    if flags {
        if let Some(fg) = fg {
            let t = predtest(&out, &fg, words);
            set_pred_flags(env, t);
        }
    }
    0
}

/// `do_sat_addsub_32()` and `do_sat_addsub_64()`, for a non negative `val`.
fn sat_addsub(reg: u64, val: u64, u: bool, d: bool, sf: bool) -> u64 {
    let val = i128::from(val);
    let reg = match (sf, u) {
        (true, true) => i128::from(reg),
        (true, false) => i128::from(reg as i64),
        (false, true) => i128::from(reg as u32),
        (false, false) => i128::from(reg as i32),
    };
    let r = if d { reg - val } else { reg + val };
    let (lo, hi) = match (sf, u) {
        (true, true) => (0, i128::from(u64::MAX)),
        (true, false) => (i128::from(i64::MIN), i128::from(i64::MAX)),
        (false, true) => (0, i128::from(u32::MAX)),
        (false, false) => (i128::from(i32::MIN), i128::from(i32::MAX)),
    };
    r.clamp(lo, hi) as u64
}

/// The number of elements a WHILE sets, from `do_WHILE()` and `trans_WHILE_ptr()`. Data bit
/// 0: greater-than form, bit 1: unsigned, bit 2: equal, bit 3: 64-bit, bit 4: WHILEWR or
/// WHILERW, bit 5: WHILERW.
fn while_count(d: &Dsc, elems: usize, op0: u64, op1: u64) -> usize {
    let tmax = elems as u64;
    if d.data & 16 != 0 {
        let diff = if d.data & 32 != 0 {
            // WHILERW: abs(op1 - op0) / esize, all true if equal.
            if op0 == op1 { tmax } else { op0.abs_diff(op1) >> d.esz }
        } else if op0 >= op1 {
            tmax
        } else {
            (op1 - op0) >> d.esz
        };
        return diff.min(tmax) as usize;
    }
    let lt = d.data & 1 == 0;
    let u = d.data & 2 != 0;
    let eq = d.data & 4 != 0;
    let sf = d.data & 8 != 0;
    let (op0, op1) = if sf {
        (op0, op1)
    } else if u {
        (op0 & 0xffff_ffff, op1 & 0xffff_ffff)
    } else {
        (op0 as i32 as i64 as u64, op1 as i32 as i64 as u64)
    };
    let (mut t0, maxval) = if lt {
        let max = match (u, sf) {
            (true, true) => u64::MAX,
            (true, false) => u64::from(u32::MAX),
            (false, true) => i64::MAX as u64,
            (false, false) => i32::MAX as u64,
        };
        (op1.wrapping_sub(op0), max)
    } else {
        let max = match (u, sf) {
            (true, _) => 0,
            (false, true) => i64::MIN as u64,
            (false, false) => i32::MIN as i64 as u64,
        };
        (op0.wrapping_sub(op1), max)
    };
    if eq {
        t0 = t0.wrapping_add(1);
        if op1 == maxval {
            t0 = tmax;
        }
    }
    t0 = t0.min(tmax);
    let cond = match (lt, u, eq) {
        (true, true, true) => op0 <= op1,
        (true, true, false) => op0 < op1,
        (true, false, true) => (op0 as i64) <= (op1 as i64),
        (true, false, false) => (op0 as i64) < (op1 as i64),
        (false, true, true) => op0 >= op1,
        (false, true, false) => op0 > op1,
        (false, false, true) => (op0 as i64) >= (op1 as i64),
        (false, false, false) => (op0 as i64) > (op1 as i64),
    };
    if cond { t0 as usize } else { 0 }
}

/// The operations of `sve_mem`. The data bits 0 and 1 hold the memory element size, bit 2
/// whether it is sign extended and bits 16 to 19 the MMU index; `x` is the address.
#[allow(missing_docs)]
pub(crate) mod mm {
    /// Contiguous loads of Zd to Zd+nreg-1: data bits 3 and 4 hold nreg-1, bits 5 and 6 the
    /// fault kind ([`FF`](self::FF) or [`NF`](self::NF)).
    pub(crate) const LD: u32 = 1;
    /// Contiguous stores, data as for LD.
    pub(crate) const ST: u32 = 2;
    /// LD1R: load one element and broadcast it to the active elements.
    pub(crate) const LD1R: u32 = 3;
    /// LD1RQ: a contiguous load of 16 bytes, replicated.
    pub(crate) const LD1RQ: u32 = 4;
    /// LDR and STR of a Z register (data bit 3: predicate register).
    pub(crate) const LDR: u32 = 5;
    pub(crate) const STR: u32 = 6;
    /// Gather loads; data bit 3: first fault, bits 4 and 5 the addressing kind ([`OFF_UXTW`]
    /// and on), bit 6: scale the offsets by the memory element size.
    pub(crate) const GATHER: u32 = 7;
    /// Scatter stores, data as for GATHER.
    pub(crate) const SCATTER: u32 = 8;
    /// The fault kinds of LD.
    pub(crate) const FF: u32 = 1;
    pub(crate) const NF: u32 = 2;
    /// The addressing kinds of GATHER and SCATTER: `x + (offset << scale)` with the offset
    /// zero extended from 32 bits, sign extended from 32 bits or 64 bits, or `zn + x`.
    pub(crate) const OFF_UXTW: u32 = 0;
    pub(crate) const OFF_SXTW: u32 = 1;
    pub(crate) const OFF_D: u32 = 2;
    pub(crate) const OFF_VEC: u32 = 3;
}

/// `TARGET_PAGE_SIZE`.
const PAGE: u64 = 4096;

/// `sve_cont_ldst_elements()`, by element index rather than byte offset.
#[derive(Debug)]
struct Cont {
    first: usize,
    last: usize,
    /// The last element wholly on the first page.
    last0: Option<usize>,
    /// The byte offset of the page boundary, if the operation crosses one.
    page_split: Option<u64>,
    /// The active element that crosses the page boundary.
    split: Option<usize>,
    /// The first active element on the second page.
    first1: Option<usize>,
}

fn cont_elements(addr: u64, g: &P, esz: u32, elems: usize, msize: u64) -> Option<Cont> {
    let first = first_active(g, esz, elems)?;
    let last = last_active(g, esz, elems)?;
    let page_split = PAGE - (addr & (PAGE - 1));
    let mut c =
        Cont { first, last, last0: Some(last), page_split: None, split: None, first1: None };
    if last as u64 * msize + msize <= page_split {
        return Some(c);
    }
    c.page_split = Some(page_split);
    let elt_split = (page_split / msize) as usize;
    c.last0 = elt_split.checked_sub(1);
    let mut next = elt_split;
    if page_split % msize != 0 {
        if act(g, esz, elt_split) {
            c.split = Some(elt_split);
            if elt_split == last {
                return Some(c);
            }
        }
        next += 1;
    }
    c.first1 = (next..elems).find(|&i| act(g, esz, i));
    Some(c)
}

/// `record_fault()`: clear the bits of FFR from predicate bit `i`.
fn record_fault(env: &mut [u8], i: usize, vl: usize) {
    let mut ffr = pload(env, FFR);
    for b in i..vl {
        pset(&mut ffr, b, false);
    }
    pstore(env, FFR, &ffr, vl);
}

/// The fields of a `sve_mem` descriptor's data.
struct Mem {
    msz: u32,
    oi: MemOpIdx,
    mmu_idx: usize,
}

impl Mem {
    fn new(d: &Dsc) -> Mem {
        let msz = d.data & 3;
        let mmu_idx = (d.data >> 16) & 15;
        let mop = MemOp(msz | ((d.data >> 2) & 1) << 3);
        Mem { msz, oi: MemOpIdx::new(mop, mmu_idx), mmu_idx: mmu_idx as usize }
    }

    fn msize(&self) -> u64 {
        1 << self.msz
    }
}

fn h_sve_mem(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    let d = Dsc::unpack(a[1]);
    let x = a[2];
    run(h, |cpu| {
        match d.op {
            mm::LD | mm::LD1RQ => cont_ld(cpu, &d, x)?,
            mm::ST => cont_st(cpu, &d, x)?,
            mm::LD1R => ld1r(cpu, &d, x)?,
            mm::LDR | mm::STR => ldr_str(cpu, &d, x)?,
            mm::GATHER => gather(cpu, &d, x)?,
            mm::SCATTER => scatter(cpu, &d, x)?,
            _ => unreachable!("bad sve memory op {}", d.op),
        }
        Ok(0)
    })
}

type R<T> = Result<T, CpuLoopExit>;

/// `sve_cont_ldst_pages()` for the loads and stores that may fault anywhere: probe the first
/// active element, then the second page.
fn cont_probe(
    cpu: &mut Cpu<'_>,
    addr: u64,
    c: &Cont,
    msize: u64,
    at: MmuAccessType,
    mmu_idx: usize,
) -> R<()> {
    probe_access(cpu, addr.wrapping_add(c.first as u64 * msize), 0, at, mmu_idx, Ra::Tb)?;
    if let Some(ps) = c.page_split {
        let off = if c.split.is_some() { ps } else { c.first1.map_or(ps, |i| i as u64 * msize) };
        probe_access(cpu, addr.wrapping_add(off), 0, at, mmu_idx, Ra::Tb)?;
    }
    Ok(())
}

/// The contiguous loads, `sve_ldN_r()` and `sve_ldnfff1_r()`, and LD1RQ.
fn cont_ld(cpu: &mut Cpu<'_>, d: &Dsc, addr: u64) -> R<()> {
    let mem = Mem::new(d);
    let nreg = if d.op == mm::LD { ((d.data >> 3) & 3) as usize + 1 } else { 1 };
    let fault = (d.data >> 5) & 3;
    // LD1RQ loads one quadword under the low bits of the predicate.
    let vl = if d.op == mm::LD1RQ { 16 } else { d.vl };
    let esz = d.esz;
    let elems = vl >> esz;
    let g = pload(cpu.env, d.g);
    let msize = mem.msize() * nreg as u64;
    let mut out = [[0u8; ZREG_SIZE]; 4];
    let elt = |cpu: &mut Cpu<'_>, out: &mut [Z; 4], i: usize| -> R<()> {
        for k in 0..nreg {
            let ea = addr.wrapping_add(i as u64 * msize + ((k as u64) << mem.msz));
            let v = cpu_ld_mmu(cpu, ea, mem.oi, Ra::Tb)?;
            set(&mut out[k], esz, i, v);
        }
        Ok(())
    };
    let commit = |env: &mut [u8], out: &[Z; 4]| {
        for k in 0..nreg {
            let mut z = out[k];
            if d.op == mm::LD1RQ {
                for b in 16..d.vl {
                    z[b] = z[b % 16];
                }
            }
            zstore(env, (d.d + k) & 31, &z, d.vl);
        }
    };
    let Some(c) = cont_elements(addr, &g, esz, elems, msize) else {
        // The entire predicate was false; no load occurs.
        commit(cpu.env, &out);
        return Ok(());
    };
    let at = MmuAccessType::DataLoad;
    if fault == 0 {
        cont_probe(cpu, addr, &c, msize, at, mem.mmu_idx)?;
        for i in c.first..=c.last {
            if act(&g, esz, i) {
                elt(cpu, &mut out, i)?;
            }
        }
        commit(cpu.env, &out);
        return Ok(());
    }
    let first_addr = addr.wrapping_add(c.first as u64 * msize);
    let is_split = c.split == Some(c.first);
    // The element at which FFR is cleared, if any.
    let fault_at: Option<usize>;
    let ram0 = if fault == mm::FF {
        // A size of 0 only raises the fault; whether the page is RAM comes from the
        // non-faulting probe, as QEMU takes both from one probe_access_flags().
        probe_access(cpu, first_addr, 0, at, mem.mmu_idx, Ra::Tb)?;
        let ram0 = probe_access_nonfault(cpu, first_addr, at, mem.mmu_idx, Ra::Tb)? == Some(true);
        if is_split {
            let ps = c.page_split.unwrap_or(0);
            probe_access(cpu, addr.wrapping_add(ps), 0, at, mem.mmu_idx, Ra::Tb)?;
        }
        Some(ram0)
    } else {
        probe_access_nonfault(cpu, first_addr, at, mem.mmu_idx, Ra::Tb)?
    };
    match ram0 {
        None => fault_at = Some(c.first),
        Some(ram0) => {
            if fault == mm::FF && (!ram0 || is_split) {
                // The first active element goes the slow way, and may fault.
                elt(cpu, &mut out, c.first)?;
                fault_at = if is_split { c.first1 } else { Some(c.first + 1) };
            } else if is_split {
                let ps = c.page_split.unwrap_or(0);
                let ram1 =
                    probe_access_nonfault(cpu, addr.wrapping_add(ps), at, mem.mmu_idx, Ra::Tb)?;
                if ram0 && ram1 == Some(true) {
                    elt(cpu, &mut out, c.first)?;
                    fault_at = c.first1;
                } else {
                    fault_at = Some(c.first);
                }
            } else if !ram0 {
                fault_at = Some(c.first);
            } else {
                for i in c.first..=c.last0.unwrap_or(c.first) {
                    if act(&g, esz, i) {
                        elt(cpu, &mut out, i)?;
                    }
                }
                fault_at = c.split.or(c.first1);
            }
        }
    }
    commit(cpu.env, &out);
    if let Some(i) = fault_at {
        record_fault(cpu.env, (i << esz).min(vl), vl);
    }
    Ok(())
}

/// The contiguous stores, `sve_stN_r()`.
fn cont_st(cpu: &mut Cpu<'_>, d: &Dsc, addr: u64) -> R<()> {
    let mem = Mem::new(d);
    let nreg = ((d.data >> 3) & 3) as usize + 1;
    let esz = d.esz;
    let g = pload(cpu.env, d.g);
    let msize = mem.msize() * nreg as u64;
    let regs: Vec<Z> = (0..nreg).map(|k| zload(cpu.env, (d.d + k) & 31)).collect();
    let Some(c) = cont_elements(addr, &g, esz, d.elems(), msize) else {
        return Ok(());
    };
    cont_probe(cpu, addr, &c, msize, MmuAccessType::DataStore, mem.mmu_idx)?;
    for i in c.first..=c.last {
        if !act(&g, esz, i) {
            continue;
        }
        for (k, z) in regs.iter().enumerate() {
            let ea = addr.wrapping_add(i as u64 * msize + ((k as u64) << mem.msz));
            cpu_st_mmu(cpu, ea, get(z, esz, i), mem.oi, Ra::Tb)?;
        }
    }
    Ok(())
}

/// LD1R: if any element is active, load one element and copy it to the active elements.
fn ld1r(cpu: &mut Cpu<'_>, d: &Dsc, addr: u64) -> R<()> {
    let mem = Mem::new(d);
    let g = pload(cpu.env, d.g);
    let mut out = [0u8; ZREG_SIZE];
    if first_active(&g, d.esz, d.elems()).is_some() {
        let v = cpu_ld_mmu(cpu, addr, mem.oi, Ra::Tb)?;
        for i in 0..d.elems() {
            if act(&g, d.esz, i) {
                set(&mut out, d.esz, i, v);
            }
        }
    }
    zstore(cpu.env, d.d, &out, d.vl);
    Ok(())
}

/// `gen_sve_ldr()` and `gen_sve_str()`: the register goes to or from memory 8 bytes at a
/// time, then 4, 2 and 2 for the tail of a predicate register, updating the register as it
/// goes as QEMU's inline code does.
fn ldr_str(cpu: &mut Cpu<'_>, d: &Dsc, addr: u64) -> R<()> {
    let mmu_idx = (d.data >> 16) & 15;
    let (base, len) =
        if d.data & 8 != 0 { (preg_off(d.d), d.vl / 8) } else { (vreg_off(d.d), d.vl) };
    let mut off = 0;
    while off < len {
        let rem = len - off;
        let n = if rem >= 8 {
            8
        } else if rem >= 4 {
            4
        } else {
            2
        };
        let size = (n as u32).trailing_zeros();
        let oi = MemOpIdx::new(MemOp(size), mmu_idx);
        let ea = addr.wrapping_add(off as u64);
        if d.op == mm::LDR {
            let v = cpu_ld_mmu(cpu, ea, oi, Ra::Tb)?;
            cpu.env[base + off..base + off + n].copy_from_slice(&v.to_le_bytes()[..n]);
        } else {
            let mut b = [0u8; 8];
            b[..n].copy_from_slice(&cpu.env[base + off..base + off + n]);
            cpu_st_mmu(cpu, ea, u64::from_le_bytes(b), oi, Ra::Tb)?;
        }
        off += n;
    }
    Ok(())
}

/// The address of element `i` of a gather or scatter.
fn gather_addr(d: &Dsc, mem: &Mem, m: &Z, i: usize, x: u64) -> u64 {
    let raw = get(m, d.esz, i);
    let shift = if d.data & 64 != 0 { mem.msz } else { 0 };
    let off = match (d.data >> 4) & 3 {
        mm::OFF_UXTW => raw & 0xffff_ffff,
        mm::OFF_SXTW => raw as i32 as i64 as u64,
        mm::OFF_D => raw,
        _ => return raw.wrapping_add(x),
    };
    x.wrapping_add(off << shift)
}

/// `sve_ld1_z()` and `sve_ldff1_z()`.
fn gather(cpu: &mut Cpu<'_>, d: &Dsc, x: u64) -> R<()> {
    let mem = Mem::new(d);
    let g = pload(cpu.env, d.g);
    let m = zload(cpu.env, d.m);
    let elems = d.elems();
    let mut out = [0u8; ZREG_SIZE];
    let ff = d.data & 8 != 0;
    let Some(first) = first_active(&g, d.esz, elems) else {
        zstore(cpu.env, d.d, &out, d.vl);
        return Ok(());
    };
    if !ff {
        for i in first..elems {
            if act(&g, d.esz, i) {
                let v = cpu_ld_mmu(cpu, gather_addr(d, &mem, &m, i, x), mem.oi, Ra::Tb)?;
                set(&mut out, d.esz, i, v);
            }
        }
        zstore(cpu.env, d.d, &out, d.vl);
        return Ok(());
    }
    // The first active element may fault; the others are probed without faulting.
    let v = cpu_ld_mmu(cpu, gather_addr(d, &mem, &m, first, x), mem.oi, Ra::Tb)?;
    set(&mut out, d.esz, first, v);
    let mut fault_at = None;
    for i in first + 1..elems {
        if !act(&g, d.esz, i) {
            continue;
        }
        let ea = gather_addr(d, &mem, &m, i, x);
        let in_page = PAGE - (ea & (PAGE - 1));
        let ok = in_page >= mem.msize()
            && probe_access_nonfault(cpu, ea, MmuAccessType::DataLoad, mem.mmu_idx, Ra::Tb)?
                == Some(true);
        if !ok {
            fault_at = Some(i);
            break;
        }
        let v = cpu_ld_mmu(cpu, ea, mem.oi, Ra::Tb)?;
        set(&mut out, d.esz, i, v);
    }
    zstore(cpu.env, d.d, &out, d.vl);
    if let Some(i) = fault_at {
        record_fault(cpu.env, i << d.esz, d.vl);
    }
    Ok(())
}

/// `sve_st1_z()`: probe every active element, then store them.
fn scatter(cpu: &mut Cpu<'_>, d: &Dsc, x: u64) -> R<()> {
    let mem = Mem::new(d);
    let g = pload(cpu.env, d.g);
    let m = zload(cpu.env, d.m);
    let z = zload(cpu.env, d.d);
    let at = MmuAccessType::DataStore;
    let active: Vec<usize> = (0..d.elems()).filter(|&i| act(&g, d.esz, i)).collect();
    for &i in &active {
        let ea = gather_addr(d, &mem, &m, i, x);
        probe_access(cpu, ea, 0, at, mem.mmu_idx, Ra::Tb)?;
        let in_page = PAGE - (ea & (PAGE - 1));
        if in_page < mem.msize() {
            probe_access(cpu, ea.wrapping_add(in_page), 0, at, mem.mmu_idx, Ra::Tb)?;
        }
    }
    for &i in &active {
        let ea = gather_addr(d, &mem, &m, i, x);
        cpu_st_mmu(cpu, ea, get(&z, d.esz, i), mem.oi, Ra::Tb)?;
    }
    Ok(())
}
