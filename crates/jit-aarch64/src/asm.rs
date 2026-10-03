// SPDX-License-Identifier: GPL-2.0-or-later

//! The A64 encoder: instruction words, labels, relocations and the constant pool.
//!
//! This is the instruction emitting half of QEMU's `tcg/aarch64/tcg-target.c.inc`: the
//! `tcg_out_insn_*` format helpers, `tcg_out_movi`, `tcg_out_logicali`, `tcg_out_dupi_vec`,
//! `tcg_out_ldst` and friends, plus the relocation and pool machinery from `tcg/tcg.c` and
//! `tcg/tcg-pool.c.inc`. Code is assembled into a vector of words whose final address is known
//! up front, so PC relative forms (ADR, ADRP, direct B and BL) can be chosen while emitting.

use ruvm_jit_core::types::{Cond, Type};

/// A host register: 0 to 30 are x0 to x30, 31 is sp or xzr depending on the instruction, 32 to
/// 63 are v0 to v31.
pub(crate) type Reg = u8;

pub(crate) const X0: Reg = 0;
pub(crate) const X1: Reg = 1;
pub(crate) const X2: Reg = 2;
pub(crate) const X3: Reg = 3;
pub(crate) const X16: Reg = 16;
pub(crate) const X17: Reg = 17;
pub(crate) const X18: Reg = 18;
pub(crate) const X19: Reg = 19;
pub(crate) const X28: Reg = 28;
pub(crate) const FP: Reg = 29;
pub(crate) const LR: Reg = 30;
pub(crate) const SP: Reg = 31;
pub(crate) const XZR: Reg = 31;
/// The first scratch register, `TCG_REG_TMP0`.
pub(crate) const TMP0: Reg = X16;
/// `TCG_REG_TMP1`.
pub(crate) const TMP1: Reg = X17;
/// `TCG_REG_TMP2`.
pub(crate) const TMP2: Reg = LR;
/// The CPU state pointer, `TCG_AREG0`.
pub(crate) const AREG0: Reg = X19;
/// The guest base in user mode, `TCG_REG_GUEST_BASE`.
pub(crate) const GUEST_BASE: Reg = X28;
/// v31, `TCG_VEC_TMP0`.
pub(crate) const VTMP0: Reg = 63;
/// v30, a second vector scratch this port reserves for its own expansions.
pub(crate) const VTMP1: Reg = 62;

/// The vector register `n`.
pub(crate) const fn v(n: u8) -> Reg {
    32 + n
}

/// A64 condition codes, `enum aarch64_cond_code`.
pub(crate) mod cc {
    pub(crate) const EQ: u32 = 0x0;
    pub(crate) const NE: u32 = 0x1;
    pub(crate) const HS: u32 = 0x2;
    pub(crate) const LO: u32 = 0x3;
    pub(crate) const HI: u32 = 0x8;
    pub(crate) const LS: u32 = 0x9;
    pub(crate) const GE: u32 = 0xa;
    pub(crate) const LT: u32 = 0xb;
    pub(crate) const GT: u32 = 0xc;
    pub(crate) const LE: u32 = 0xd;
    pub(crate) const AL: u32 = 0xe;
}

/// `tcg_cond_to_aarch64`. `Never` and `Always` have no A64 code and are handled by the callers;
/// mapping them here would pick `NV`, which behaves like `AL`.
pub(crate) fn cond_code(c: Cond) -> u32 {
    match c {
        Cond::Eq | Cond::TstEq => cc::EQ,
        Cond::Ne | Cond::TstNe => cc::NE,
        Cond::Lt => cc::LT,
        Cond::Ge => cc::GE,
        Cond::Le => cc::LE,
        Cond::Gt => cc::GT,
        Cond::Ltu => cc::LO,
        Cond::Gtu => cc::HI,
        Cond::Geu => cc::HS,
        Cond::Leu => cc::LS,
        Cond::Always => cc::AL,
        Cond::Never => 0xf,
    }
}

/// The instruction words, `AArch64Insn`.
#[allow(dead_code)]
pub(crate) mod i {
    pub(crate) const CBZ: u32 = 0x34000000;
    pub(crate) const CBNZ: u32 = 0x35000000;
    pub(crate) const B_C: u32 = 0x54000000;
    pub(crate) const TBZ: u32 = 0x36000000;
    pub(crate) const TBNZ: u32 = 0x37000000;
    pub(crate) const B: u32 = 0x14000000;
    pub(crate) const BL: u32 = 0x94000000;
    pub(crate) const BR: u32 = 0xd61f0000;
    pub(crate) const BLR: u32 = 0xd63f0000;
    pub(crate) const RET: u32 = 0xd65f0000;
    pub(crate) const LD1R: u32 = 0x0d40c000;
    pub(crate) const LDR_LIT: u32 = 0x58000000;
    pub(crate) const LDR_V64_LIT: u32 = 0x5c000000;
    pub(crate) const LDR_V128_LIT: u32 = 0x9c000000;
    pub(crate) const LDXP: u32 = 0xc8600000;
    pub(crate) const STXP: u32 = 0xc8200000;

    const LDST: u32 = 0x38000000;
    pub(crate) const STRB: u32 = LDST;
    pub(crate) const STRH: u32 = LDST | 1 << 30;
    pub(crate) const STRW: u32 = LDST | 2 << 30;
    pub(crate) const STRX: u32 = LDST | 3 << 30;
    pub(crate) const LDRB: u32 = LDST | 1 << 22;
    pub(crate) const LDRH: u32 = LDST | 1 << 22 | 1 << 30;
    pub(crate) const LDRW: u32 = LDST | 1 << 22 | 2 << 30;
    pub(crate) const LDRX: u32 = LDST | 1 << 22 | 3 << 30;
    pub(crate) const LDRSBW: u32 = LDST | 3 << 22;
    pub(crate) const LDRSHW: u32 = LDST | 3 << 22 | 1 << 30;
    pub(crate) const LDRSBX: u32 = LDST | 2 << 22;
    pub(crate) const LDRSHX: u32 = LDST | 2 << 22 | 1 << 30;
    pub(crate) const LDRSWX: u32 = LDST | 2 << 22 | 2 << 30;
    pub(crate) const LDRVS: u32 = 0x3c000000 | 1 << 22 | 2 << 30;
    pub(crate) const STRVS: u32 = 0x3c000000 | 2 << 30;
    pub(crate) const LDRVD: u32 = 0x3c000000 | 1 << 22 | 3 << 30;
    pub(crate) const STRVD: u32 = 0x3c000000 | 3 << 30;
    pub(crate) const LDRVQ: u32 = 0x3c000000 | 3 << 22;
    pub(crate) const STRVQ: u32 = 0x3c000000 | 2 << 22;
    pub(crate) const LDST_TO_REG: u32 = 0x00200800;
    pub(crate) const LDST_TO_UIMM: u32 = 0x01000000;

    pub(crate) const LDP: u32 = 0x28400000;
    pub(crate) const STP: u32 = 0x28000000;

    pub(crate) const ADDI: u32 = 0x11000000;
    pub(crate) const ADDSI: u32 = 0x31000000;
    pub(crate) const SUBI: u32 = 0x51000000;
    pub(crate) const SUBSI: u32 = 0x71000000;

    pub(crate) const BFM: u32 = 0x33000000;
    pub(crate) const SBFM: u32 = 0x13000000;
    pub(crate) const UBFM: u32 = 0x53000000;
    pub(crate) const EXTR: u32 = 0x13800000;

    pub(crate) const ANDI: u32 = 0x12000000;
    pub(crate) const ORRI: u32 = 0x32000000;
    pub(crate) const EORI: u32 = 0x52000000;
    pub(crate) const ANDSI: u32 = 0x72000000;

    pub(crate) const MOVN: u32 = 0x12800000;
    pub(crate) const MOVZ: u32 = 0x52800000;
    pub(crate) const MOVK: u32 = 0x72800000;

    pub(crate) const ADR: u32 = 0x10000000;
    pub(crate) const ADRP: u32 = 0x90000000;

    pub(crate) const ADD_EXT: u32 = 0x0b200000;
    pub(crate) const ADD: u32 = 0x0b000000;
    pub(crate) const ADDS: u32 = 0x2b000000;
    pub(crate) const SUB: u32 = 0x4b000000;
    pub(crate) const SUBS: u32 = 0x6b000000;

    pub(crate) const ADC: u32 = 0x1a000000;
    pub(crate) const ADCS: u32 = 0x3a000000;
    pub(crate) const SBC: u32 = 0x5a000000;
    pub(crate) const SBCS: u32 = 0x7a000000;

    pub(crate) const CSEL: u32 = 0x1a800000;
    pub(crate) const CSINC: u32 = 0x1a800400;
    pub(crate) const CSINV: u32 = 0x5a800000;
    pub(crate) const CSNEG: u32 = 0x5a800400;

    pub(crate) const CLZ: u32 = 0x5ac01000;
    pub(crate) const RBIT: u32 = 0x5ac00000;
    pub(crate) const REV: u32 = 0x5ac00000;

    pub(crate) const LSLV: u32 = 0x1ac02000;
    pub(crate) const LSRV: u32 = 0x1ac02400;
    pub(crate) const ASRV: u32 = 0x1ac02800;
    pub(crate) const RORV: u32 = 0x1ac02c00;
    pub(crate) const SMULH: u32 = 0x9b407c00;
    pub(crate) const UMULH: u32 = 0x9bc07c00;
    pub(crate) const UDIV: u32 = 0x1ac00800;
    pub(crate) const SDIV: u32 = 0x1ac00c00;
    pub(crate) const MADD: u32 = 0x1b000000;
    pub(crate) const MSUB: u32 = 0x1b008000;
    pub(crate) const SMADDL: u32 = 0x9b200000;
    pub(crate) const UMADDL: u32 = 0x9ba00000;

    pub(crate) const AND: u32 = 0x0a000000;
    pub(crate) const BIC: u32 = 0x0a200000;
    pub(crate) const ORR: u32 = 0x2a000000;
    pub(crate) const ORN: u32 = 0x2a200000;
    pub(crate) const EOR: u32 = 0x4a000000;
    pub(crate) const EON: u32 = 0x4a200000;
    pub(crate) const ANDS: u32 = 0x6a000000;
    pub(crate) const AND_LSR: u32 = AND | 1 << 22;

    pub(crate) const DUP: u32 = 0x0e000400;
    pub(crate) const INS: u32 = 0x4e001c00;
    pub(crate) const UMOV: u32 = 0x0e003c00;

    pub(crate) const MOVI: u32 = 0x0f000400;
    pub(crate) const MVNI: u32 = 0x2f000400;
    pub(crate) const BIC_IMM: u32 = 0x2f001400;
    pub(crate) const ORR_IMM: u32 = 0x0f001400;

    pub(crate) const Q_SSHR: u32 = 0x5f000400;
    pub(crate) const Q_SHL: u32 = 0x5f005400;
    pub(crate) const Q_USHR: u32 = 0x7f000400;
    pub(crate) const Q_SLI: u32 = 0x7f005400;

    pub(crate) const E_SQADD: u32 = 0x5e200c00;
    pub(crate) const E_SQSUB: u32 = 0x5e202c00;
    pub(crate) const E_CMGT: u32 = 0x5e203400;
    pub(crate) const E_CMGE: u32 = 0x5e203c00;
    pub(crate) const E_SSHL: u32 = 0x5e204400;
    pub(crate) const E_ADD: u32 = 0x5e208400;
    pub(crate) const E_CMTST: u32 = 0x5e208c00;
    pub(crate) const E_UQADD: u32 = 0x7e200c00;
    pub(crate) const E_UQSUB: u32 = 0x7e202c00;
    pub(crate) const E_CMHI: u32 = 0x7e203400;
    pub(crate) const E_CMHS: u32 = 0x7e203c00;
    pub(crate) const E_USHL: u32 = 0x7e204400;
    pub(crate) const E_SUB: u32 = 0x7e208400;
    pub(crate) const E_CMEQ: u32 = 0x7e208c00;

    pub(crate) const S_CMGT0: u32 = 0x5e208800;
    pub(crate) const S_CMEQ0: u32 = 0x5e209800;
    pub(crate) const S_CMLT0: u32 = 0x5e20a800;
    pub(crate) const S_ABS: u32 = 0x5e20b800;
    pub(crate) const S_CMGE0: u32 = 0x7e208800;
    pub(crate) const S_CMLE0: u32 = 0x7e209800;
    pub(crate) const S_NEG: u32 = 0x7e20b800;

    pub(crate) const V_SSHR: u32 = 0x0f000400;
    pub(crate) const V_SHL: u32 = 0x0f005400;
    pub(crate) const V_SLI: u32 = 0x2f005400;
    pub(crate) const V_USHR: u32 = 0x2f000400;

    pub(crate) const Q_ADD: u32 = 0x0e208400;
    pub(crate) const Q_AND: u32 = 0x0e201c00;
    pub(crate) const Q_BIC: u32 = 0x0e601c00;
    pub(crate) const Q_BIF: u32 = 0x2ee01c00;
    pub(crate) const Q_BIT: u32 = 0x2ea01c00;
    pub(crate) const Q_BSL: u32 = 0x2e601c00;
    pub(crate) const Q_EOR: u32 = 0x2e201c00;
    pub(crate) const Q_MUL: u32 = 0x0e209c00;
    pub(crate) const Q_ORR: u32 = 0x0ea01c00;
    pub(crate) const Q_ORN: u32 = 0x0ee01c00;
    pub(crate) const Q_SUB: u32 = 0x2e208400;
    pub(crate) const Q_CMGT: u32 = 0x0e203400;
    pub(crate) const Q_CMGE: u32 = 0x0e203c00;
    pub(crate) const Q_CMTST: u32 = 0x0e208c00;
    pub(crate) const Q_CMHI: u32 = 0x2e203400;
    pub(crate) const Q_CMHS: u32 = 0x2e203c00;
    pub(crate) const Q_CMEQ: u32 = 0x2e208c00;
    pub(crate) const Q_SMAX: u32 = 0x0e206400;
    pub(crate) const Q_SMIN: u32 = 0x0e206c00;
    pub(crate) const Q_SSHL: u32 = 0x0e204400;
    pub(crate) const Q_SQADD: u32 = 0x0e200c00;
    pub(crate) const Q_SQSUB: u32 = 0x0e202c00;
    pub(crate) const Q_UMAX: u32 = 0x2e206400;
    pub(crate) const Q_UMIN: u32 = 0x2e206c00;
    pub(crate) const Q_UQADD: u32 = 0x2e200c00;
    pub(crate) const Q_UQSUB: u32 = 0x2e202c00;
    pub(crate) const Q_USHL: u32 = 0x2e204400;
    pub(crate) const Q_CMGT0: u32 = 0x0e208800;
    pub(crate) const Q_CMEQ0: u32 = 0x0e209800;
    pub(crate) const Q_CMLT0: u32 = 0x0e20a800;
    pub(crate) const Q_CMGE0: u32 = 0x2e208800;
    pub(crate) const Q_CMLE0: u32 = 0x2e209800;
    pub(crate) const Q_NOT: u32 = 0x2e205800;
    pub(crate) const Q_ABS: u32 = 0x0e20b800;
    pub(crate) const Q_NEG: u32 = 0x2e20b800;
    /// CNT, count bits per byte. Not in QEMU's table; used for `ctpop`.
    pub(crate) const Q_CNT: u32 = 0x0e205800;
    /// ADDV, add across lanes. Not in QEMU's table; used for `ctpop`.
    pub(crate) const Q_ADDV: u32 = 0x0e31b800;

    pub(crate) const NOP: u32 = 0xd503201f;
    pub(crate) const DMB_ISH: u32 = 0xd50338bf;
    pub(crate) const DMB_LD: u32 = 0x00000100;
    pub(crate) const DMB_ST: u32 = 0x00000200;

    // Load-acquire and store-release, with the size in bits 30 and 31. Not in QEMU's table;
    // used for the acquire and release guest accesses of `crate::memory_order`.
    /// LDAPR, load-acquire RCpc (FEAT_LRCPC).
    pub(crate) const LDAPR: u32 = 0x38bfc000;
    /// LDAR, load-acquire RCsc.
    pub(crate) const LDAR: u32 = 0x08dffc00;
    /// STLR, store-release.
    pub(crate) const STLR: u32 = 0x089ffc00;
    /// LDAPUR, load-acquire RCpc with an unscaled offset (FEAT_LRCPC2).
    pub(crate) const LDAPUR: u32 = 0x19400000;
    /// LDAPURS with a 64-bit destination, sign extending (FEAT_LRCPC2).
    pub(crate) const LDAPURS_X: u32 = 0x19800000;
    /// LDAPURS with a 32-bit destination, sign extending (FEAT_LRCPC2).
    pub(crate) const LDAPURS_W: u32 = 0x19c00000;
    /// STLUR, store-release with an unscaled offset (FEAT_LRCPC2).
    pub(crate) const STLUR: u32 = 0x19000000;
}

/// Is `val` usable as an add or subtract immediate, `is_aimm`.
pub(crate) fn is_aimm(val: u64) -> bool {
    val & !0xfff == 0 || val & !0xfff000 == 0
}

/// Is `val` a logical immediate of the simplified forms QEMU matches, `is_limm`.
pub(crate) fn is_limm(val: u64) -> bool {
    let mut val = val;
    if (val as i64) < 0 {
        val = !val;
    }
    if val == 0 {
        return false;
    }
    val = val.wrapping_add(val & val.wrapping_neg());
    val & val.wrapping_sub(1) == 0
}

/// `is_shimm16`: returns `(cmode, imm8)`.
pub(crate) fn is_shimm16(v16: u16) -> Option<(u32, u32)> {
    if v16 == v16 & 0xff {
        Some((0x8, (v16 & 0xff) as u32))
    } else if v16 == v16 & 0xff00 {
        Some((0xa, (v16 >> 8) as u32))
    } else {
        None
    }
}

/// `is_shimm32`.
pub(crate) fn is_shimm32(v32: u32) -> Option<(u32, u32)> {
    if v32 == v32 & 0xff {
        Some((0x0, v32 & 0xff))
    } else if v32 == v32 & 0xff00 {
        Some((0x2, (v32 >> 8) & 0xff))
    } else if v32 == v32 & 0xff0000 {
        Some((0x4, (v32 >> 16) & 0xff))
    } else if v32 == v32 & 0xff000000 {
        Some((0x6, v32 >> 24))
    } else {
        None
    }
}

/// `is_soimm32`.
pub(crate) fn is_soimm32(v32: u32) -> Option<(u32, u32)> {
    if v32 & 0xffff00ff == 0xff {
        Some((0xc, (v32 >> 8) & 0xff))
    } else if v32 & 0xff00ffff == 0xffff {
        Some((0xd, (v32 >> 16) & 0xff))
    } else {
        None
    }
}

fn ex32(v: u32, pos: u32, len: u32) -> u32 {
    (v >> pos) & ((1u32 << len) - 1)
}

fn ex64(v: u64, pos: u32, len: u32) -> u64 {
    (v >> pos) & ((1u64 << len) - 1)
}

/// `is_fimm32`.
pub(crate) fn is_fimm32(v32: u32) -> Option<(u32, u32)> {
    if ex32(v32, 0, 19) == 0 && (ex32(v32, 25, 6) == 0x20 || ex32(v32, 25, 6) == 0x1f) {
        Some((0xf, (ex32(v32, 31, 1) << 7) | (ex32(v32, 25, 1) << 6) | ex32(v32, 19, 6)))
    } else {
        None
    }
}

/// `is_fimm64`.
pub(crate) fn is_fimm64(v64: u64) -> Option<(u32, u32)> {
    if ex64(v64, 0, 48) == 0 && (ex64(v64, 54, 9) == 0x100 || ex64(v64, 54, 9) == 0x0ff) {
        let imm8 = (ex64(v64, 63, 1) << 7) | (ex64(v64, 54, 1) << 6) | ex64(v64, 48, 6);
        Some((0xf, imm8 as u32))
    } else {
        None
    }
}

/// `is_shimm32_pair`: the MOVI parameters and the cmode for the ORR, or `None`.
pub(crate) fn is_shimm32_pair(v32: u32) -> Option<(u32, u32, u32)> {
    let mut i = 6u32;
    while i > 0 {
        let tmp = v32 & !(0xffu32 << (i * 4));
        if let Some((cmode, imm8)) = is_shimm32(tmp).or_else(|| is_soimm32(tmp)) {
            return Some((cmode, imm8, i));
        }
        i -= 2;
    }
    None
}

/// `is_shimm1632`.
pub(crate) fn is_shimm1632(v32: u32) -> Option<(u32, u32)> {
    if v32 == (v32 & 0xffff) | (v32 << 16) { is_shimm16(v32 as u16) } else { is_shimm32(v32) }
}

/// Relocation kinds, as in QEMU's `patch_reloc`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reloc {
    /// `R_AARCH64_CONDBR19`: B.cond, CBZ, CBNZ and LDR (literal).
    Condbr19,
    /// `R_AARCH64_JUMP26`: B and BL.
    Jump26,
    /// `R_AARCH64_TSTBR14`: TBZ and TBNZ.
    Tstbr14,
    /// ADR, used for the return address of slow paths.
    Adr21,
}

/// Something went wrong while assembling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum AsmError {
    /// A branch displacement does not fit its field.
    OutOfRange,
}

#[derive(Clone, Copy, Debug)]
struct Fixup {
    at: usize,
    kind: Reloc,
    label: usize,
}

#[derive(Clone, Copy, Debug)]
struct PoolRef {
    at: usize,
    words: usize,
    val: [u64; 2],
}

/// An assembler for one block of code whose first word will live at `base`.
#[derive(Debug)]
pub(crate) struct Asm {
    pub(crate) code: Vec<u32>,
    base: u64,
    labels: Vec<Option<usize>>,
    fixups: Vec<Fixup>,
    pool: Vec<PoolRef>,
}

/// The result of [`Asm::finish`].
#[derive(Debug)]
pub(crate) struct Assembled {
    pub(crate) bytes: Vec<u8>,
    /// Byte offset of the first word after the instructions and pool.
    pub(crate) end: usize,
}

impl Asm {
    pub(crate) fn new(base: u64) -> Asm {
        Asm { code: Vec::new(), base, labels: Vec::new(), fixups: Vec::new(), pool: Vec::new() }
    }

    /// The index of the next word.
    pub(crate) fn pos(&self) -> usize {
        self.code.len()
    }

    /// The address of word `idx`.
    pub(crate) fn addr_of(&self, idx: usize) -> u64 {
        self.base + 4 * idx as u64
    }

    /// The address of the next word.
    pub(crate) fn here(&self) -> u64 {
        self.addr_of(self.pos())
    }

    pub(crate) fn emit(&mut self, w: u32) {
        self.code.push(w);
    }

    pub(crate) fn new_label(&mut self) -> usize {
        self.labels.push(None);
        self.labels.len() - 1
    }

    pub(crate) fn bind(&mut self, l: usize) {
        self.labels[l] = Some(self.pos());
    }

    pub(crate) fn is_bound(&self, l: usize) -> bool {
        self.labels[l].is_some()
    }

    /// Record that the word about to be emitted refers to `label`.
    pub(crate) fn reloc_here(&mut self, kind: Reloc, label: usize) {
        let at = self.pos();
        self.fixups.push(Fixup { at, kind, label });
    }

    /// Record that the word at `at` refers to `label`.
    pub(crate) fn reloc_at(&mut self, at: usize, kind: Reloc, label: usize) {
        self.fixups.push(Fixup { at, kind, label });
    }

    /// Record that the word about to be emitted is an LDR literal of a pool value.
    fn pool_here(&mut self, words: usize, val: [u64; 2]) {
        let at = self.pos();
        self.pool.push(PoolRef { at, words, val });
    }

    /// Patch one field. `disp` is in words from the instruction.
    fn patch(w: u32, kind: Reloc, disp: i64) -> Result<u32, AsmError> {
        let fits = |bits: u32| disp == (disp << (64 - bits)) >> (64 - bits);
        match kind {
            Reloc::Condbr19 => {
                if !fits(19) {
                    return Err(AsmError::OutOfRange);
                }
                Ok((w & !(0x7ffff << 5)) | ((disp as u32 & 0x7ffff) << 5))
            }
            Reloc::Jump26 => {
                if !fits(26) {
                    return Err(AsmError::OutOfRange);
                }
                Ok((w & !0x03ff_ffff) | (disp as u32 & 0x03ff_ffff))
            }
            Reloc::Tstbr14 => {
                if !fits(14) {
                    return Err(AsmError::OutOfRange);
                }
                Ok((w & !(0x3fff << 5)) | ((disp as u32 & 0x3fff) << 5))
            }
            Reloc::Adr21 => {
                let bytes = disp * 4;
                if bytes != (bytes << 43) >> 43 {
                    return Err(AsmError::OutOfRange);
                }
                let b = bytes as u32;
                Ok((w & !(3 << 29 | 0x7ffff << 5)) | (b & 3) << 29 | ((b >> 2) & 0x7ffff) << 5)
            }
        }
    }

    /// Resolve labels, lay out the pool and return the bytes. The pool is 16-byte aligned so
    /// that 128-bit literal loads stay naturally aligned.
    pub(crate) fn finish(mut self) -> Result<Assembled, AsmError> {
        for f in std::mem::take(&mut self.fixups) {
            let target = self.labels[f.label].expect("branch to a label that was never bound");
            let disp = target as i64 - f.at as i64;
            self.code[f.at] = Self::patch(self.code[f.at], f.kind, disp)?;
        }
        if !self.pool.is_empty() {
            while self.code.len() % 4 != 0 {
                self.code.push(0);
            }
            // Two word entries first, so every entry stays aligned to its size.
            let mut entries: Vec<([u64; 2], usize)> = Vec::new();
            let mut order: Vec<PoolRef> = self.pool.clone();
            order.sort_by_key(|p| std::cmp::Reverse(p.words));
            let mut placed: Vec<(usize, [u64; 2], usize)> = Vec::new();
            for p in &order {
                if placed.iter().any(|&(w, v, _)| w == p.words && v == p.val) {
                    continue;
                }
                let at = self.code.len();
                placed.push((p.words, p.val, at));
                for k in 0..p.words {
                    let v = p.val[k];
                    self.code.push(v as u32);
                    self.code.push((v >> 32) as u32);
                }
                entries.push((p.val, at));
            }
            for p in &self.pool {
                let &(_, _, at) = placed
                    .iter()
                    .find(|&&(w, v, _)| w == p.words && v == p.val)
                    .expect("pool entry placed");
                let disp = at as i64 - p.at as i64;
                self.code[p.at] = Self::patch(self.code[p.at], Reloc::Condbr19, disp)?;
            }
        }
        let end = self.code.len() * 4;
        let mut bytes = Vec::with_capacity(end);
        for w in &self.code {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        Ok(Assembled { bytes, end })
    }

    // Instruction formats, `tcg_out_insn_*`.

    pub(crate) fn ldlit(&mut self, insn: u32, imm19: i32, rt: Reg) {
        self.emit(insn | ((imm19 as u32) & 0x7ffff) << 5 | (rt as u32 & 0x1f));
    }

    pub(crate) fn stxp(&mut self, insn: u32, rs: Reg, rt: Reg, rt2: Reg, rn: Reg) {
        self.emit(insn | (rs as u32) << 16 | (rt2 as u32) << 10 | (rn as u32) << 5 | rt as u32);
    }

    pub(crate) fn cbz(&mut self, insn: u32, ext: bool, rt: Reg, imm19: i32) {
        self.emit(insn | (ext as u32) << 31 | ((imm19 as u32) & 0x7ffff) << 5 | rt as u32);
    }

    pub(crate) fn bcond(&mut self, code: u32, imm19: i32) {
        self.emit(i::B_C | code | ((imm19 as u32) & 0x7ffff) << 5);
    }

    pub(crate) fn tbz(&mut self, insn: u32, rt: Reg, bit: u32, imm14: i32) {
        let w = insn | (bit & 0x20) << (31 - 5) | (bit & 0x1f) << 19;
        self.emit(w | ((imm14 as u32) & 0x3fff) << 5 | rt as u32);
    }

    pub(crate) fn branch(&mut self, insn: u32, imm26: i32) {
        self.emit(insn | (imm26 as u32 & 0x03ff_ffff));
    }

    pub(crate) fn breg(&mut self, insn: u32, rn: Reg) {
        self.emit(insn | (rn as u32) << 5);
    }

    #[allow(clippy::too_many_arguments, reason = "one argument per field, as `tcg_out_insn_3314`")]
    pub(crate) fn ldstpair(
        &mut self,
        insn: u32,
        r1: Reg,
        r2: Reg,
        rn: Reg,
        ofs: i64,
        pre: bool,
        w: bool,
    ) {
        debug_assert!((-0x200..0x200).contains(&ofs) && ofs & 7 == 0);
        let insn = insn | 1 << 31 | (pre as u32) << 24 | (w as u32) << 23;
        let insn = insn | ((ofs as u32) & (0x7f << 3)) << (15 - 3);
        self.emit(insn | (r2 as u32) << 10 | (rn as u32) << 5 | r1 as u32);
    }

    pub(crate) fn addsub_imm(&mut self, insn: u32, ext: bool, rd: Reg, rn: Reg, aimm: u64) {
        let mut aimm = aimm;
        if aimm > 0xfff {
            debug_assert!(aimm & 0xfff == 0);
            aimm >>= 12;
            debug_assert!(aimm <= 0xfff);
            aimm |= 1 << 12;
        }
        let w = insn | (ext as u32) << 31 | (aimm as u32) << 10;
        self.emit(w | (rn as u32) << 5 | rd as u32);
    }

    #[allow(clippy::too_many_arguments, reason = "one argument per field, as `tcg_out_insn_3402`")]
    pub(crate) fn bitfield(
        &mut self,
        insn: u32,
        ext: bool,
        rd: Reg,
        rn: Reg,
        n: u32,
        immr: u32,
        imms: u32,
    ) {
        let w = insn | (ext as u32) << 31 | n << 22 | immr << 16 | imms << 10;
        self.emit(w | (rn as u32) << 5 | rd as u32);
    }

    pub(crate) fn extract(&mut self, ext: bool, rd: Reg, rn: Reg, rm: Reg, imms: u32) {
        let e = ext as u32;
        let w = i::EXTR | e << 31 | e << 22 | (rm as u32) << 16 | imms << 10;
        self.emit(w | (rn as u32) << 5 | rd as u32);
    }

    pub(crate) fn movw(&mut self, insn: u32, ext: bool, rd: Reg, half: u16, shift: u32) {
        debug_assert!(shift & !0x30 == 0);
        self.emit(insn | (ext as u32) << 31 | shift << (21 - 4) | (half as u32) << 5 | rd as u32);
    }

    pub(crate) fn pcrel(&mut self, insn: u32, rd: Reg, disp: i64) {
        let d = disp as u32;
        self.emit(insn | (d & 3) << 29 | (d & 0x1ffffc) << (5 - 2) | rd as u32);
    }

    pub(crate) fn addsub_ext(&mut self, sf: bool, rd: Reg, rn: Reg, rm: Reg, opt: u32, imm3: u32) {
        let w = i::ADD_EXT | (sf as u32) << 31 | (rm as u32) << 16 | opt << 13 | imm3 << 10;
        self.emit(w | (rn as u32) << 5 | rd as u32);
    }

    pub(crate) fn realshift(&mut self, insn: u32, ext: bool, rd: Reg, rn: Reg, rm: Reg, imm6: u32) {
        let w = insn | (ext as u32) << 31 | (rm as u32) << 16 | imm6 << 10;
        self.emit(w | (rn as u32) << 5 | rd as u32);
    }

    /// `addsub_shift`, `rrr_sf`, `rrr` and `logic_shift`.
    pub(crate) fn rrr(&mut self, insn: u32, ext: bool, rd: Reg, rn: Reg, rm: Reg) {
        let w = insn | (ext as u32) << 31 | (rm as u32) << 16;
        self.emit(w | (rn as u32) << 5 | rd as u32);
    }

    pub(crate) fn csel(&mut self, insn: u32, ext: bool, rd: Reg, rn: Reg, rm: Reg, code: u32) {
        let w = insn | (ext as u32) << 31 | (rm as u32) << 16 | code << 12;
        self.emit(w | (rn as u32) << 5 | rd as u32);
    }

    pub(crate) fn rr_sf(&mut self, insn: u32, ext: bool, rd: Reg, rn: Reg) {
        self.emit(insn | (ext as u32) << 31 | (rn as u32) << 5 | rd as u32);
    }

    pub(crate) fn rrrr(&mut self, insn: u32, ext: bool, rd: Reg, rn: Reg, rm: Reg, ra: Reg) {
        let w = insn | (ext as u32) << 31 | (rm as u32) << 16 | (ra as u32) << 10;
        self.emit(w | (rn as u32) << 5 | rd as u32);
    }

    /// `simd_copy`. Bit 11 set means a general register input.
    pub(crate) fn simd_copy(
        &mut self,
        insn: u32,
        q: bool,
        rd: Reg,
        rn: Reg,
        dst_idx: u32,
        src_idx: u32,
    ) {
        let w = insn | (q as u32) << 30 | dst_idx << 16 | src_idx << 11;
        let rn = rn as u32;
        self.emit(w | (rd as u32 & 0x1f) | (!rn & 0x20) << 6 | (rn & 0x1f) << 5);
    }

    pub(crate) fn simd_imm(
        &mut self,
        insn: u32,
        q: bool,
        rd: Reg,
        op: bool,
        cmode: u32,
        imm8: u32,
    ) {
        let w = insn | (q as u32) << 30 | (op as u32) << 29 | cmode << 12 | (rd as u32 & 0x1f);
        self.emit(w | (imm8 & 0xe0) << (16 - 5) | (imm8 & 0x1f) << 5);
    }

    pub(crate) fn q_shift(&mut self, insn: u32, rd: Reg, rn: Reg, immhb: u32) {
        self.emit(insn | immhb << 16 | (rn as u32 & 0x1f) << 5 | (rd as u32 & 0x1f));
    }

    pub(crate) fn rrr_e(&mut self, insn: u32, size: u32, rd: Reg, rn: Reg, rm: Reg) {
        let w = insn | size << 22 | (rm as u32 & 0x1f) << 16;
        self.emit(w | (rn as u32 & 0x1f) << 5 | (rd as u32 & 0x1f));
    }

    pub(crate) fn simd_rr(&mut self, insn: u32, size: u32, rd: Reg, rn: Reg) {
        self.emit(insn | size << 22 | (rn as u32 & 0x1f) << 5 | (rd as u32 & 0x1f));
    }

    pub(crate) fn simd_shift_imm(&mut self, insn: u32, q: bool, rd: Reg, rn: Reg, immhb: u32) {
        let w = insn | (q as u32) << 30 | immhb << 16;
        self.emit(w | (rn as u32 & 0x1f) << 5 | (rd as u32 & 0x1f));
    }

    pub(crate) fn qrrr_e(&mut self, insn: u32, q: bool, size: u32, rd: Reg, rn: Reg, rm: Reg) {
        let w = insn | (q as u32) << 30 | size << 22 | (rm as u32 & 0x1f) << 16;
        self.emit(w | (rn as u32 & 0x1f) << 5 | (rd as u32 & 0x1f));
    }

    pub(crate) fn qrr_e(&mut self, insn: u32, q: bool, size: u32, rd: Reg, rn: Reg) {
        let w = insn | (q as u32) << 30 | size << 22;
        self.emit(w | (rn as u32 & 0x1f) << 5 | (rd as u32 & 0x1f));
    }

    pub(crate) fn loadrep(&mut self, q: bool, rt: Reg, rn: Reg, size: u32) {
        self.emit(i::LD1R | (rt as u32 & 0x1f) | (rn as u32) << 5 | size << 10 | (q as u32) << 30);
    }

    pub(crate) fn ldst_reg(&mut self, insn: u32, rd: Reg, base: Reg, ext: bool, regoff: Reg) {
        let w = insn | i::LDST_TO_REG | (regoff as u32) << 16 | 0x4000 | (ext as u32) << 13;
        self.emit(w | (base as u32) << 5 | (rd as u32 & 0x1f));
    }

    /// LDAPR, LDAR or STLR of `1 << size` bytes at `[rn]`.
    pub(crate) fn ldst_ordered(&mut self, insn: u32, size: u32, rt: Reg, rn: Reg) {
        self.emit(insn | size << 30 | (rn as u32) << 5 | (rt as u32 & 0x1f));
    }

    /// LDAPUR, LDAPURS or STLUR of `1 << size` bytes at `[rn, #imm9]`.
    pub(crate) fn ldst_rcpc_imm(&mut self, insn: u32, size: u32, rt: Reg, rn: Reg, imm9: i32) {
        let w = insn | size << 30 | ((imm9 as u32) & 0x1ff) << 12;
        self.emit(w | (rn as u32) << 5 | (rt as u32 & 0x1f));
    }

    pub(crate) fn ldst_imm(&mut self, insn: u32, rd: Reg, rn: Reg, offset: i64) {
        self.emit(insn | ((offset as u32) & 0x1ff) << 12 | (rn as u32) << 5 | (rd as u32 & 0x1f));
    }

    pub(crate) fn ldst_uimm(&mut self, insn: u32, rd: Reg, rn: Reg, scaled: u64) {
        let w = insn | i::LDST_TO_UIMM | (scaled as u32) << 10;
        self.emit(w | (rn as u32) << 5 | (rd as u32 & 0x1f));
    }

    // Higher level helpers.

    /// `tcg_out_movr`.
    pub(crate) fn movr(&mut self, ext: bool, rd: Reg, rm: Reg) {
        self.rrr(i::ORR, ext, rd, XZR, rm);
    }

    /// `tcg_out_movr_sp`.
    pub(crate) fn movr_sp(&mut self, ext: bool, rd: Reg, rn: Reg) {
        self.addsub_imm(i::ADDI, ext, rd, rn, 0);
    }

    /// `tcg_out_logicali`.
    pub(crate) fn logicali(&mut self, insn: u32, ext: bool, rd: Reg, rn: Reg, limm: u64) {
        debug_assert!(is_limm(limm));
        let h = limm.leading_zeros();
        let l = limm.trailing_zeros();
        let (mut r, mut c);
        if l == 0 {
            r = 0;
            c = (!limm).trailing_zeros().wrapping_sub(1);
            if h == 0 {
                r = (!limm).leading_zeros();
                c = c.wrapping_add(r);
            }
        } else {
            r = 64 - l;
            c = r.wrapping_sub(h).wrapping_sub(1);
        }
        if !ext {
            r &= 31;
            c &= 31;
        }
        self.bitfield(insn, ext, rd, rn, ext as u32, r, c);
    }

    /// `tcg_out_movi` for an integer register.
    pub(crate) fn movi(&mut self, ty: Type, rd: Reg, value: u64) {
        debug_assert!(rd < 32);
        let mut value = value;
        let mut svalue = value;
        let mut ivalue = !value;
        let mut ext = ty == Type::I64;
        if !ext || value & !0xffff_ffffu64 == 0 {
            svalue = value as i32 as i64 as u64;
            value = value as u32 as u64;
            ivalue = ivalue as u32 as u64;
            ext = false;
        }
        if value & !0xffffu64 == 0 {
            self.movw(i::MOVZ, ext, rd, value as u16, 0);
            return;
        } else if ivalue & !0xffffu64 == 0 {
            self.movw(i::MOVN, ext, rd, ivalue as u16, 0);
            return;
        }
        if is_limm(svalue) {
            self.logicali(i::ORRI, ext, rd, XZR, svalue);
            return;
        }
        if ext {
            let src = self.here() as i64;
            let disp = (value as i64).wrapping_sub(src);
            if disp == (disp << 43) >> 43 {
                self.pcrel(i::ADR, rd, disp);
                return;
            }
            let disp = ((value as i64) >> 12).wrapping_sub(src >> 12);
            if disp == (disp << 43) >> 43 {
                self.pcrel(i::ADRP, rd, disp);
                if value & 0xfff != 0 {
                    self.addsub_imm(i::ADDI, ext, rd, rd, value & 0xfff);
                }
                return;
            }
        }
        let (t0, opc) = if value.count_ones() >= 32 { (ivalue, i::MOVN) } else { (value, i::MOVZ) };
        let s0 = t0.trailing_zeros() & (63 & !15);
        let t1 = t0 & !(0xffffu64.wrapping_shl(s0));
        let s1 = t1.trailing_zeros() & (63 & !15);
        let t2 = t1 & !(0xffffu64.wrapping_shl(s1));
        if t2 == 0 {
            self.movw(opc, ext, rd, (t0 >> s0) as u16, s0);
            if t1 != 0 {
                self.movw(i::MOVK, ext, rd, (value >> s1) as u16, s1);
            }
            return;
        }
        self.pool_here(1, [value, 0]);
        self.ldlit(i::LDR_LIT, 0, rd);
    }

    /// `tcg_out_dupi_vec`.
    pub(crate) fn dupi_vec(&mut self, ty: Type, vece: u32, rd: Reg, v64: u64) {
        let q = ty == Type::V128;
        if vece == 0 {
            self.simd_imm(i::MOVI, q, rd, false, 0xe, v64 as u8 as u32);
            return;
        }
        let mut imm8 = 0u32;
        let mut ok = true;
        for k in 0..8 {
            let byte = (v64 >> (k * 8)) as u8;
            if byte == 0xff {
                imm8 |= 1 << k;
            } else if byte != 0 {
                ok = false;
                break;
            }
        }
        if ok {
            self.simd_imm(i::MOVI, q, rd, true, 0xe, imm8);
            return;
        }
        if vece == 1 {
            let v16 = v64 as u16;
            if let Some((cmode, imm8)) = is_shimm16(v16) {
                self.simd_imm(i::MOVI, q, rd, false, cmode, imm8);
                return;
            }
            if let Some((cmode, imm8)) = is_shimm16(!v16) {
                self.simd_imm(i::MVNI, q, rd, false, cmode, imm8);
                return;
            }
            self.simd_imm(i::MOVI, q, rd, false, 0x8, (v16 & 0xff) as u32);
            self.simd_imm(i::ORR_IMM, q, rd, false, 0xa, (v16 >> 8) as u32);
            return;
        } else if vece == 2 {
            let v32 = v64 as u32;
            let n32 = !v32;
            if let Some((cmode, imm8)) =
                is_shimm32(v32).or_else(|| is_soimm32(v32)).or_else(|| is_fimm32(v32))
            {
                self.simd_imm(i::MOVI, q, rd, false, cmode, imm8);
                return;
            }
            if let Some((cmode, imm8)) = is_shimm32(n32).or_else(|| is_soimm32(n32)) {
                self.simd_imm(i::MVNI, q, rd, false, cmode, imm8);
                return;
            }
            if let Some((cmode, imm8, k)) = is_shimm32_pair(v32) {
                self.simd_imm(i::MOVI, q, rd, false, cmode, imm8);
                self.simd_imm(i::ORR_IMM, q, rd, false, k, ex32(v32, k * 4, 8));
                return;
            }
            if let Some((cmode, imm8, k)) = is_shimm32_pair(n32) {
                self.simd_imm(i::MVNI, q, rd, false, cmode, imm8);
                self.simd_imm(i::BIC_IMM, q, rd, false, k, ex32(n32, k * 4, 8));
                return;
            }
        } else if let Some((cmode, imm8)) = is_fimm64(v64) {
            self.simd_imm(i::MOVI, q, rd, true, cmode, imm8);
            return;
        }
        if q {
            self.pool_here(2, [v64, v64]);
            self.ldlit(i::LDR_V128_LIT, 0, rd);
        } else {
            self.pool_here(1, [v64, 0]);
            self.ldlit(i::LDR_V64_LIT, 0, rd);
        }
    }

    /// `tcg_out_ldst`.
    pub(crate) fn ldst(&mut self, insn: u32, rd: Reg, rn: Reg, offset: i64, lgsize: u32) {
        if offset >= 0 && offset & ((1 << lgsize) - 1) == 0 {
            let scaled = (offset >> lgsize) as u64;
            if scaled <= 0xfff {
                self.ldst_uimm(insn, rd, rn, scaled);
                return;
            }
        }
        if (-256..256).contains(&offset) {
            self.ldst_imm(insn, rd, rn, offset);
            return;
        }
        self.movi(Type::I64, TMP0, offset as u64);
        self.ldst_reg(insn, rd, rn, true, TMP0);
    }

    /// `tcg_out_mov`.
    pub(crate) fn mov(&mut self, ty: Type, ret: Reg, arg: Reg) {
        if ret == arg {
            return;
        }
        match ty {
            Type::I32 | Type::I64 if ret < 32 && arg < 32 => {
                self.movr(ty == Type::I64, ret, arg);
            }
            Type::I32 | Type::I64 if ret < 32 => {
                // QEMU passes an element index of 0 here, which is not a valid UMOV; use the
                // lowest element of the right size.
                let is64 = ty == Type::I64;
                self.simd_copy(i::UMOV, is64, ret, arg, 4 << is64 as u32, 0);
            }
            Type::I32 | Type::I64 if arg < 32 => {
                self.simd_copy(i::INS, false, ret, arg, 4 << (ty == Type::I64) as u32, 0);
            }
            Type::V128 => self.qrrr_e(i::Q_ORR, true, 0, ret, arg, arg),
            _ => self.qrrr_e(i::Q_ORR, false, 0, ret, arg, arg),
        }
    }

    /// `tcg_out_ld`.
    pub(crate) fn ld(&mut self, ty: Type, ret: Reg, base: Reg, ofs: i64) {
        let (insn, lg) = match ty {
            Type::I32 => (if ret < 32 { i::LDRW } else { i::LDRVS }, 2),
            Type::I64 => (if ret < 32 { i::LDRX } else { i::LDRVD }, 3),
            Type::V64 => (i::LDRVD, 3),
            _ => (i::LDRVQ, 4),
        };
        self.ldst(insn, ret, base, ofs, lg);
    }

    /// `tcg_out_st`.
    pub(crate) fn st(&mut self, ty: Type, src: Reg, base: Reg, ofs: i64) {
        let (insn, lg) = match ty {
            Type::I32 => (if src < 32 { i::STRW } else { i::STRVS }, 2),
            Type::I64 => (if src < 32 { i::STRX } else { i::STRVD }, 3),
            Type::V64 => (i::STRVD, 3),
            _ => (i::STRVQ, 4),
        };
        self.ldst(insn, src, base, ofs, lg);
    }

    /// A direct branch to an absolute address, or an indirect one through x16 when it is out
    /// of range. `link` selects BL/BLR.
    pub(crate) fn jump_abs(&mut self, target: u64, link: bool) {
        let disp = (target as i64).wrapping_sub(self.here() as i64) >> 2;
        if disp == (disp << 38) >> 38 {
            self.branch(if link { i::BL } else { i::B }, disp as i32);
        } else {
            self.movi(Type::I64, TMP0, target);
            self.breg(if link { i::BLR } else { i::BR }, TMP0);
        }
    }

    /// B.cond to a label.
    pub(crate) fn bcond_label(&mut self, code: u32, l: usize) {
        self.reloc_here(Reloc::Condbr19, l);
        self.bcond(code, 0);
    }

    /// B to a label.
    pub(crate) fn b_label(&mut self, l: usize) {
        self.reloc_here(Reloc::Jump26, l);
        self.branch(i::B, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(f: impl FnOnce(&mut Asm)) -> Vec<u32> {
        let mut a = Asm::new(0x10_0000);
        f(&mut a);
        a.code
    }

    #[test]
    fn encodings_match_gnu_as() {
        // Reference words from `aarch64-linux-gnu-as`.
        assert_eq!(one(|a| a.rrr(i::ADD, true, 0, 1, 2)), [0x8b020020]);
        assert_eq!(one(|a| a.rrr(i::SUB, false, 3, 4, 5)), [0x4b050083]);
        assert_eq!(one(|a| a.movr(true, 1, 2)), [0xaa0203e1]);
        assert_eq!(one(|a| a.addsub_imm(i::ADDI, true, 0, 1, 0x10)), [0x91004020]);
        assert_eq!(one(|a| a.addsub_imm(i::SUBI, true, 31, 31, 0x1000)), [0xd14007ff]);
        assert_eq!(one(|a| a.breg(i::RET, LR)), [0xd65f03c0]);
        assert_eq!(one(|a| a.breg(i::BR, X16)), [0xd61f0200]);
        assert_eq!(one(|a| a.ldstpair(i::STP, FP, LR, SP, -96, true, true)), [0xa9ba7bfd]);
        assert_eq!(one(|a| a.ldstpair(i::LDP, FP, LR, SP, 96, false, true)), [0xa8c67bfd]);
        assert_eq!(one(|a| a.csel(i::CSINC, false, 0, XZR, XZR, cc::NE)), [0x1a9f17e0]);
        assert_eq!(one(|a| a.rr_sf(i::CLZ, true, 0, 1)), [0xdac01020]);
        assert_eq!(one(|a| a.qrrr_e(i::Q_ADD, true, 2, v(0), v(1), v(2))), [0x4ea28420]);
        assert_eq!(one(|a| a.ldst(i::LDRX, 0, X19, 8, 3)), [0xf9400660]);
        assert_eq!(one(|a| a.ldst(i::STRW, 1, SP, -4, 2)), [0xb81fc3e1]);
    }

    #[test]
    fn movi_forms() {
        assert_eq!(one(|a| a.movi(Type::I64, 0, 0x1234)), [0x52824680]);
        // mov w0, #-1 (MOVN) for a 32-bit all ones value.
        assert_eq!(one(|a| a.movi(Type::I32, 0, 0xffff_ffff)), [0x12800000]);
        // mov x0, #-1
        assert_eq!(one(|a| a.movi(Type::I64, 0, u64::MAX)), [0x92800000]);
        // orr x0, xzr, #0xff00
        assert_eq!(one(|a| a.movi(Type::I64, 0, 0xff00)), [0x52800000 | 0xff00 << 5]);
        assert_eq!(one(|a| a.movi(Type::I64, 1, 0x0000_00ff_0000_0000)), [0xb2601fe1]);
        let w = one(|a| a.movi(Type::I64, 2, 0x1234_0000_5678_0000));
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn limm_matches_qemu() {
        assert!(is_limm(0xff));
        assert!(is_limm(0xff00));
        assert!(is_limm(!0xff00));
        assert!(!is_limm(0));
        assert!(!is_limm(0x5555));
        assert!(is_aimm(0xfff));
        assert!(is_aimm(0xabc000));
        assert!(!is_aimm(0x1001));
    }

    #[test]
    fn labels_and_pool() {
        let mut a = Asm::new(0);
        let l = a.new_label();
        a.b_label(l);
        a.movi(Type::I64, 0, 0x1234_5678_9abc_def0);
        a.bind(l);
        a.emit(i::NOP);
        let out = a.finish().unwrap();
        let w0 = u32::from_le_bytes(out.bytes[0..4].try_into().unwrap());
        assert_eq!(w0, i::B | 2);
        let w1 = u32::from_le_bytes(out.bytes[4..8].try_into().unwrap());
        // LDR x0, literal 16 bytes ahead (pool aligned to 16 bytes).
        assert_eq!(w1, i::LDR_LIT | (3 << 5));
        let v = u64::from_le_bytes(out.bytes[16..24].try_into().unwrap());
        assert_eq!(v, 0x1234_5678_9abc_def0);
    }

    #[test]
    fn acquire_release_encodings() {
        // Reference words from Apple clang with `.arch armv8.4-a+rcpc`.
        assert_eq!(one(|a| a.ldst_ordered(i::LDAPR, 3, 0, X16)), [0xf8bfc200]);
        assert_eq!(one(|a| a.ldst_ordered(i::LDAPR, 2, 3, X16)), [0xb8bfc203]);
        assert_eq!(one(|a| a.ldst_ordered(i::LDAPR, 1, 5, X16)), [0x78bfc205]);
        assert_eq!(one(|a| a.ldst_ordered(i::LDAPR, 0, 7, X16)), [0x38bfc207]);
        assert_eq!(one(|a| a.ldst_ordered(i::LDAR, 3, 0, X16)), [0xc8dffe00]);
        assert_eq!(one(|a| a.ldst_ordered(i::LDAR, 0, 1, X16)), [0x08dffe01]);
        assert_eq!(one(|a| a.ldst_ordered(i::STLR, 3, 0, X16)), [0xc89ffe00]);
        assert_eq!(one(|a| a.ldst_ordered(i::STLR, 2, 2, X16)), [0x889ffe02]);
        assert_eq!(one(|a| a.ldst_ordered(i::STLR, 1, 2, X16)), [0x489ffe02]);
        assert_eq!(one(|a| a.ldst_ordered(i::STLR, 0, XZR, X16)), [0x089ffe1f]);
        assert_eq!(one(|a| a.ldst_ordered(i::STLR, 3, XZR, X16)), [0xc89ffe1f]);
        assert_eq!(one(|a| a.ldst_rcpc_imm(i::LDAPUR, 3, 0, X16, 0)), [0xd9400200]);
        assert_eq!(one(|a| a.ldst_rcpc_imm(i::LDAPURS_X, 0, 1, X16, 0)), [0x19800201]);
        assert_eq!(one(|a| a.ldst_rcpc_imm(i::LDAPURS_W, 0, 1, X16, 0)), [0x19c00201]);
        assert_eq!(one(|a| a.ldst_rcpc_imm(i::LDAPURS_X, 1, 2, X16, 0)), [0x59800202]);
        assert_eq!(one(|a| a.ldst_rcpc_imm(i::LDAPURS_W, 1, 2, X16, 0)), [0x59c00202]);
        assert_eq!(one(|a| a.ldst_rcpc_imm(i::LDAPURS_X, 2, 3, X16, 0)), [0x99800203]);
        assert_eq!(one(|a| a.ldst_rcpc_imm(i::STLUR, 3, 4, X16, -8)), [0xd91f8204]);
        assert_eq!(one(|a| a.emit(i::DMB_ISH | i::DMB_LD)), [0xd50339bf]);
        assert_eq!(one(|a| a.emit(i::DMB_ISH | i::DMB_ST)), [0xd5033abf]);
        assert_eq!(one(|a| a.emit(i::DMB_ISH | i::DMB_LD | i::DMB_ST)), [0xd5033bbf]);
    }
}
