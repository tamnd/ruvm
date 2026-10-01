// SPDX-License-Identifier: GPL-2.0-or-later

//! The x86-64 encoder: prefixes, REX, VEX, ModRM and SIB bytes, labels, relocations and the
//! constant pool.
//!
//! This is the instruction emitting half of QEMU's `tcg/x86_64/tcg-target.c.inc`:
//! `tcg_out_opc`, `tcg_out_vex_opc`, `tcg_out_sib_offset` and the `tcg_out_modrm*` family,
//! plus the `OPC_*` and `P_*` constants, the relocation handling from `tcg/tcg.c` and the
//! constant pool of `tcg/tcg-pool.c.inc`. Code is assembled into a byte vector; branches to
//! labels are patched by [`Asm::finish`], which also appends the pool after the code.

use ruvm_jit_core::types::Cond;

/// A host register: 0 to 15 are the general registers in hardware order (rax, rcx, rdx, rbx,
/// rsp, rbp, rsi, rdi, r8 to r15), 16 to 31 are xmm0 to xmm15.
pub(crate) type Reg = u8;

pub(crate) const RAX: Reg = 0;
pub(crate) const RCX: Reg = 1;
pub(crate) const RDX: Reg = 2;
pub(crate) const RBX: Reg = 3;
pub(crate) const RSP: Reg = 4;
pub(crate) const RBP: Reg = 5;
pub(crate) const RSI: Reg = 6;
pub(crate) const RDI: Reg = 7;
pub(crate) const R8: Reg = 8;
pub(crate) const R9: Reg = 9;
pub(crate) const R10: Reg = 10;
pub(crate) const R11: Reg = 11;
pub(crate) const R12: Reg = 12;
pub(crate) const R13: Reg = 13;
pub(crate) const R14: Reg = 14;
pub(crate) const R15: Reg = 15;

/// The vector register xmm`n` (ymm`n` in 256-bit ops).
pub(crate) const fn xmm(n: u8) -> Reg {
    16 + n
}

/// x86 condition codes, `JCC_*`.
pub(crate) mod cc {
    pub(crate) const O: u8 = 0x0;
    pub(crate) const B: u8 = 0x2;
    pub(crate) const AE: u8 = 0x3;
    pub(crate) const E: u8 = 0x4;
    pub(crate) const NE: u8 = 0x5;
    pub(crate) const BE: u8 = 0x6;
    pub(crate) const A: u8 = 0x7;
    pub(crate) const S: u8 = 0x8;
    pub(crate) const NS: u8 = 0x9;
    pub(crate) const L: u8 = 0xc;
    pub(crate) const GE: u8 = 0xd;
    pub(crate) const LE: u8 = 0xe;
    pub(crate) const G: u8 = 0xf;
}

/// `tcg_cond_to_jcc`. `Never` and `Always` have no code and are handled by the callers.
pub(crate) fn cond_code(c: Cond) -> u8 {
    match c {
        Cond::Eq | Cond::TstEq => cc::E,
        Cond::Ne | Cond::TstNe => cc::NE,
        Cond::Lt => cc::L,
        Cond::Ge => cc::GE,
        Cond::Le => cc::LE,
        Cond::Gt => cc::G,
        Cond::Ltu => cc::B,
        Cond::Geu => cc::AE,
        Cond::Leu => cc::BE,
        Cond::Gtu => cc::A,
        Cond::Never | Cond::Always => cc::E,
    }
}

/// The 0x0f opcode prefix.
pub(crate) const P_EXT: u32 = 0x100;
/// The 0x0f 0x38 opcode prefix.
pub(crate) const P_EXT38: u32 = 0x200;
/// The 0x66 opcode prefix.
pub(crate) const P_DATA16: u32 = 0x400;
/// Set VEX.W.
pub(crate) const P_VEXW: u32 = 0x1000;
/// Set REX.W; the same bit as VEX.W.
pub(crate) const P_REXW: u32 = P_VEXW;
/// The REG field is a byte register.
pub(crate) const P_REXB_R: u32 = 0x2000;
/// The R/M field is a byte register.
pub(crate) const P_REXB_RM: u32 = 0x4000;
/// The 0x0f 0x3a opcode prefix.
pub(crate) const P_EXT3A: u32 = 0x10000;
/// The 0xf3 opcode prefix.
pub(crate) const P_SIMDF3: u32 = 0x20000;
/// The 0xf2 opcode prefix.
pub(crate) const P_SIMDF2: u32 = 0x40000;
/// Set VEX.L.
pub(crate) const P_VEXL: u32 = 0x80000;

/// The `OPC_*` opcodes of `tcg-target.c.inc`.
pub(crate) mod op {
    use super::*;

    pub(crate) const ARITH_EV_IZ: u32 = 0x81;
    pub(crate) const ARITH_EV_IB: u32 = 0x83;
    pub(crate) const ARITH_GV_EV: u32 = 0x03;
    pub(crate) const ANDN: u32 = 0xf2 | P_EXT38;
    pub(crate) const BSF: u32 = 0xbc | P_EXT;
    pub(crate) const BSR: u32 = 0xbd | P_EXT;
    pub(crate) const BSWAP: u32 = 0xc8 | P_EXT;
    pub(crate) const CMOVCC: u32 = 0x40 | P_EXT;
    pub(crate) const IMUL_GV_EV: u32 = 0xaf | P_EXT;
    pub(crate) const IMUL_GV_EV_IB: u32 = 0x6b;
    pub(crate) const IMUL_GV_EV_IZ: u32 = 0x69;
    pub(crate) const JCC_LONG: u32 = 0x80 | P_EXT;
    pub(crate) const JCC_SHORT: u32 = 0x70;
    pub(crate) const JMP_LONG: u32 = 0xe9;
    pub(crate) const JMP_SHORT: u32 = 0xeb;
    pub(crate) const LEA: u32 = 0x8d;
    pub(crate) const LZCNT: u32 = 0xbd | P_EXT | P_SIMDF3;
    pub(crate) const MOVB_EV_GV: u32 = 0x88;
    pub(crate) const MOVL_EV_GV: u32 = 0x89;
    pub(crate) const MOVL_GV_EV: u32 = 0x8b;
    pub(crate) const MOVB_EV_IZ: u32 = 0xc6;
    pub(crate) const MOVL_EV_IZ: u32 = 0xc7;
    pub(crate) const MOVL_IV: u32 = 0xb8;
    pub(crate) const MOVD_VY_EY: u32 = 0x6e | P_EXT | P_DATA16;
    pub(crate) const MOVD_EY_VY: u32 = 0x7e | P_EXT | P_DATA16;
    pub(crate) const MOVDQA_VX_WX: u32 = 0x6f | P_EXT | P_DATA16;
    pub(crate) const MOVDQU_VX_WX: u32 = 0x6f | P_EXT | P_SIMDF3;
    pub(crate) const MOVDQU_WX_VX: u32 = 0x7f | P_EXT | P_SIMDF3;
    pub(crate) const MOVQ_VQ_WQ: u32 = 0x7e | P_EXT | P_SIMDF3;
    pub(crate) const MOVQ_WQ_VQ: u32 = 0xd6 | P_EXT | P_DATA16;
    pub(crate) const MOVSBL: u32 = 0xbe | P_EXT;
    pub(crate) const MOVSWL: u32 = 0xbf | P_EXT;
    pub(crate) const MOVSLQ: u32 = 0x63 | P_REXW;
    pub(crate) const MOVZBL: u32 = 0xb6 | P_EXT;
    pub(crate) const MOVZWL: u32 = 0xb7 | P_EXT;
    pub(crate) const PABSB: u32 = 0x1c | P_EXT38 | P_DATA16;
    pub(crate) const PABSW: u32 = 0x1d | P_EXT38 | P_DATA16;
    pub(crate) const PABSD: u32 = 0x1e | P_EXT38 | P_DATA16;
    pub(crate) const PADDB: u32 = 0xfc | P_EXT | P_DATA16;
    pub(crate) const PADDW: u32 = 0xfd | P_EXT | P_DATA16;
    pub(crate) const PADDD: u32 = 0xfe | P_EXT | P_DATA16;
    pub(crate) const PADDQ: u32 = 0xd4 | P_EXT | P_DATA16;
    pub(crate) const PADDSB: u32 = 0xec | P_EXT | P_DATA16;
    pub(crate) const PADDSW: u32 = 0xed | P_EXT | P_DATA16;
    pub(crate) const PADDUB: u32 = 0xdc | P_EXT | P_DATA16;
    pub(crate) const PADDUW: u32 = 0xdd | P_EXT | P_DATA16;
    pub(crate) const PAND: u32 = 0xdb | P_EXT | P_DATA16;
    pub(crate) const PANDN: u32 = 0xdf | P_EXT | P_DATA16;
    pub(crate) const PCMPEQB: u32 = 0x74 | P_EXT | P_DATA16;
    pub(crate) const PCMPEQW: u32 = 0x75 | P_EXT | P_DATA16;
    pub(crate) const PCMPEQD: u32 = 0x76 | P_EXT | P_DATA16;
    pub(crate) const PCMPEQQ: u32 = 0x29 | P_EXT38 | P_DATA16;
    pub(crate) const PCMPGTB: u32 = 0x64 | P_EXT | P_DATA16;
    pub(crate) const PCMPGTW: u32 = 0x65 | P_EXT | P_DATA16;
    pub(crate) const PCMPGTD: u32 = 0x66 | P_EXT | P_DATA16;
    pub(crate) const PCMPGTQ: u32 = 0x37 | P_EXT38 | P_DATA16;
    pub(crate) const PMAXSB: u32 = 0x3c | P_EXT38 | P_DATA16;
    pub(crate) const PMAXSW: u32 = 0xee | P_EXT | P_DATA16;
    pub(crate) const PMAXSD: u32 = 0x3d | P_EXT38 | P_DATA16;
    pub(crate) const PMAXUB: u32 = 0xde | P_EXT | P_DATA16;
    pub(crate) const PMAXUW: u32 = 0x3e | P_EXT38 | P_DATA16;
    pub(crate) const PMAXUD: u32 = 0x3f | P_EXT38 | P_DATA16;
    pub(crate) const PMINSB: u32 = 0x38 | P_EXT38 | P_DATA16;
    pub(crate) const PMINSW: u32 = 0xea | P_EXT | P_DATA16;
    pub(crate) const PMINSD: u32 = 0x39 | P_EXT38 | P_DATA16;
    pub(crate) const PMINUB: u32 = 0xda | P_EXT | P_DATA16;
    pub(crate) const PMINUW: u32 = 0x3a | P_EXT38 | P_DATA16;
    pub(crate) const PMINUD: u32 = 0x3b | P_EXT38 | P_DATA16;
    pub(crate) const PMULLW: u32 = 0xd5 | P_EXT | P_DATA16;
    pub(crate) const PMULLD: u32 = 0x40 | P_EXT38 | P_DATA16;
    pub(crate) const POR: u32 = 0xeb | P_EXT | P_DATA16;
    pub(crate) const PSHUFD: u32 = 0x70 | P_EXT | P_DATA16;
    pub(crate) const PSHIFTW_IB: u32 = 0x71 | P_EXT | P_DATA16;
    pub(crate) const PSHIFTD_IB: u32 = 0x72 | P_EXT | P_DATA16;
    pub(crate) const PSHIFTQ_IB: u32 = 0x73 | P_EXT | P_DATA16;
    pub(crate) const PSLLW: u32 = 0xf1 | P_EXT | P_DATA16;
    pub(crate) const PSLLD: u32 = 0xf2 | P_EXT | P_DATA16;
    pub(crate) const PSLLQ: u32 = 0xf3 | P_EXT | P_DATA16;
    pub(crate) const PSRAW: u32 = 0xe1 | P_EXT | P_DATA16;
    pub(crate) const PSRAD: u32 = 0xe2 | P_EXT | P_DATA16;
    pub(crate) const PSRLW: u32 = 0xd1 | P_EXT | P_DATA16;
    pub(crate) const PSRLD: u32 = 0xd2 | P_EXT | P_DATA16;
    pub(crate) const PSRLQ: u32 = 0xd3 | P_EXT | P_DATA16;
    pub(crate) const PSUBB: u32 = 0xf8 | P_EXT | P_DATA16;
    pub(crate) const PSUBW: u32 = 0xf9 | P_EXT | P_DATA16;
    pub(crate) const PSUBD: u32 = 0xfa | P_EXT | P_DATA16;
    pub(crate) const PSUBQ: u32 = 0xfb | P_EXT | P_DATA16;
    pub(crate) const PSUBSB: u32 = 0xe8 | P_EXT | P_DATA16;
    pub(crate) const PSUBSW: u32 = 0xe9 | P_EXT | P_DATA16;
    pub(crate) const PSUBUB: u32 = 0xd8 | P_EXT | P_DATA16;
    pub(crate) const PSUBUW: u32 = 0xd9 | P_EXT | P_DATA16;
    pub(crate) const PUNPCKLBW: u32 = 0x60 | P_EXT | P_DATA16;
    pub(crate) const PUNPCKLWD: u32 = 0x61 | P_EXT | P_DATA16;
    pub(crate) const PUNPCKLQDQ: u32 = 0x6c | P_EXT | P_DATA16;
    pub(crate) const PXOR: u32 = 0xef | P_EXT | P_DATA16;
    pub(crate) const POP_R32: u32 = 0x58;
    pub(crate) const POPCNT: u32 = 0xb8 | P_EXT | P_SIMDF3;
    pub(crate) const PUSH_R32: u32 = 0x50;
    pub(crate) const RET: u32 = 0xc3;
    pub(crate) const SETCC: u32 = 0x90 | P_EXT | P_REXB_RM;
    pub(crate) const SHIFT_1: u32 = 0xd1;
    pub(crate) const SHIFT_IB: u32 = 0xc1;
    pub(crate) const SHIFT_CL: u32 = 0xd3;
    pub(crate) const SARX: u32 = 0xf7 | P_EXT38 | P_SIMDF3;
    pub(crate) const SHLX: u32 = 0xf7 | P_EXT38 | P_DATA16;
    pub(crate) const SHRX: u32 = 0xf7 | P_EXT38 | P_SIMDF2;
    pub(crate) const SHRD_IB: u32 = 0xac | P_EXT;
    pub(crate) const STC: u32 = 0xf9;
    pub(crate) const TESTL: u32 = 0x85;
    pub(crate) const TZCNT: u32 = 0xbc | P_EXT | P_SIMDF3;
    pub(crate) const VPBROADCASTB: u32 = 0x78 | P_EXT38 | P_DATA16;
    pub(crate) const VPBROADCASTW: u32 = 0x79 | P_EXT38 | P_DATA16;
    pub(crate) const VPBROADCASTD: u32 = 0x58 | P_EXT38 | P_DATA16;
    pub(crate) const VPBROADCASTQ: u32 = 0x59 | P_EXT38 | P_DATA16;
    pub(crate) const VPSLLVD: u32 = 0x47 | P_EXT38 | P_DATA16;
    pub(crate) const VPSLLVQ: u32 = 0x47 | P_EXT38 | P_DATA16 | P_VEXW;
    pub(crate) const VPSRAVD: u32 = 0x46 | P_EXT38 | P_DATA16;
    pub(crate) const VPSRLVD: u32 = 0x45 | P_EXT38 | P_DATA16;
    pub(crate) const VPSRLVQ: u32 = 0x45 | P_EXT38 | P_DATA16 | P_VEXW;
    pub(crate) const VZEROUPPER: u32 = 0x77 | P_EXT;
    pub(crate) const GRP3_EV: u32 = 0xf7;
    pub(crate) const GRP5: u32 = 0xff;
    pub(crate) const GRPBT: u32 = 0xba | P_EXT;
}

/// The `ARITH_*` group 1 operations.
pub(crate) mod arith {
    pub(crate) const ADD: u8 = 0;
    pub(crate) const OR: u8 = 1;
    pub(crate) const ADC: u8 = 2;
    pub(crate) const SBB: u8 = 3;
    pub(crate) const AND: u8 = 4;
    pub(crate) const SUB: u8 = 5;
    pub(crate) const XOR: u8 = 6;
    pub(crate) const CMP: u8 = 7;
}

/// The `SHIFT_*` group 2 operations.
pub(crate) mod shift {
    pub(crate) const ROL: u8 = 0;
    pub(crate) const ROR: u8 = 1;
    pub(crate) const SHL: u8 = 4;
    pub(crate) const SHR: u8 = 5;
    pub(crate) const SAR: u8 = 7;
}

/// The `EXT3_*` group 3 operations.
pub(crate) mod ext3 {
    pub(crate) const TESTI: u8 = 0;
    pub(crate) const NOT: u8 = 2;
    pub(crate) const NEG: u8 = 3;
    pub(crate) const MUL: u8 = 4;
    pub(crate) const IMUL: u8 = 5;
    pub(crate) const DIV: u8 = 6;
    pub(crate) const IDIV: u8 = 7;
}

/// The `EXT5_*` group 5 operations.
pub(crate) mod ext5 {
    pub(crate) const CALLN_EV: u8 = 2;
}

/// The `OPC_GRPBT_*` operations.
pub(crate) const GRPBT_BT: u8 = 4;

/// A memory operand: a base register or rip, an optional index, and a displacement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mem {
    /// `disp(base)`.
    Base(Reg, i32),
    /// `disp(base, index, 1)`.
    Index(Reg, Reg, i32),
    /// The constant pool entry with this index, rip relative.
    Pool(usize),
}

/// A branch or address to patch once its label is bound.
#[derive(Clone, Copy, Debug)]
enum Fixup {
    /// A rel8 displacement byte at this offset.
    Rel8(usize, usize),
    /// A rel32 displacement at this offset.
    Rel32(usize, usize),
}

/// Assembling failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AsmError {
    /// A short branch does not reach its label.
    OutOfRange,
    /// A branch to a label that was never bound.
    Unbound,
}

/// Bytes being assembled.
#[derive(Debug, Default)]
pub(crate) struct Asm {
    pub(crate) code: Vec<u8>,
    labels: Vec<Option<usize>>,
    fixups: Vec<Fixup>,
    pool: Vec<[u8; 32]>,
    pool_refs: Vec<(usize, usize)>,
}

const fn low(r: Reg) -> u8 {
    r & 7
}

impl Asm {
    pub(crate) fn new() -> Asm {
        Asm::default()
    }

    /// The offset of the next byte.
    pub(crate) fn pos(&self) -> usize {
        self.code.len()
    }

    pub(crate) fn b8(&mut self, v: u8) {
        self.code.push(v);
    }

    pub(crate) fn b32(&mut self, v: u32) {
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    pub(crate) fn b64(&mut self, v: u64) {
        self.code.extend_from_slice(&v.to_le_bytes());
    }

    pub(crate) fn new_label(&mut self) -> usize {
        self.labels.push(None);
        self.labels.len() - 1
    }

    pub(crate) fn bind(&mut self, l: usize) {
        self.labels[l] = Some(self.code.len());
    }

    pub(crate) fn is_bound(&self, l: usize) -> bool {
        self.labels[l].is_some()
    }

    /// The index of a pool entry holding `data`, shared with an equal one.
    pub(crate) fn pool_entry(&mut self, data: [u8; 32]) -> usize {
        match self.pool.iter().position(|d| *d == data) {
            Some(k) => k,
            None => {
                self.pool.push(data);
                self.pool.len() - 1
            }
        }
    }

    /// `tcg_out_opc`: legacy prefixes, REX and the opcode bytes.
    pub(crate) fn opc(&mut self, opc: u32, r: Reg, rm: Reg, x: Reg) {
        if opc & P_DATA16 != 0 {
            self.b8(0x66);
        }
        if opc & P_SIMDF3 != 0 {
            self.b8(0xf3);
        } else if opc & P_SIMDF2 != 0 {
            self.b8(0xf2);
        }
        let mut rex = 0u32;
        rex |= if opc & P_REXW != 0 { 8 } else { 0 };
        rex |= ((r & 8) >> 1) as u32;
        rex |= ((x & 8) >> 2) as u32;
        rex |= ((rm & 8) >> 3) as u32;
        // A byte register from spl to dil needs a REX prefix, even an empty one, or the
        // encoding means ah to bh.
        if opc & P_REXB_R != 0 && (r & 15) >= 4 {
            rex |= 0x100;
        }
        if opc & P_REXB_RM != 0 && (rm & 15) >= 4 {
            rex |= 0x100;
        }
        if rex != 0 {
            self.b8(0x40 | (rex & 0xf) as u8);
        }
        if opc & (P_EXT | P_EXT38 | P_EXT3A) != 0 {
            self.b8(0x0f);
            if opc & P_EXT38 != 0 {
                self.b8(0x38);
            } else if opc & P_EXT3A != 0 {
                self.b8(0x3a);
            }
        }
        self.b8(opc as u8);
    }

    /// `tcg_out_modrm`: a register to register form.
    pub(crate) fn modrm(&mut self, opc: u32, r: Reg, rm: Reg) {
        self.opc(opc, r, rm, 0);
        self.b8(0xc0 | low(r) << 3 | low(rm));
    }

    /// `tcg_out_vex_opc`.
    pub(crate) fn vex_opc(&mut self, opc: u32, r: Reg, v: Reg, rm: Reg, index: Reg) {
        let mut tmp: u8;
        if opc & (P_EXT | P_EXT38 | P_EXT3A | P_VEXW) == P_EXT && ((rm | index) & 8) == 0 {
            self.b8(0xc5);
            tmp = if r & 8 != 0 { 0 } else { 0x80 };
        } else {
            self.b8(0xc4);
            tmp = if opc & P_EXT3A != 0 {
                3
            } else if opc & P_EXT38 != 0 {
                2
            } else {
                1
            };
            tmp |= if r & 8 != 0 { 0 } else { 0x80 };
            tmp |= if index & 8 != 0 { 0 } else { 0x40 };
            tmp |= if rm & 8 != 0 { 0 } else { 0x20 };
            self.b8(tmp);
            tmp = if opc & P_VEXW != 0 { 0x80 } else { 0 };
        }
        tmp |= if opc & P_VEXL != 0 { 0x04 } else { 0 };
        if opc & P_DATA16 != 0 {
            tmp |= 1;
        } else if opc & P_SIMDF3 != 0 {
            tmp |= 2;
        } else if opc & P_SIMDF2 != 0 {
            tmp |= 3;
        }
        tmp |= (!v & 15) << 3;
        self.b8(tmp);
        self.b8(opc as u8);
    }

    /// `tcg_out_vex_modrm`. `v` is the extra source register, or 0 when the form has none.
    pub(crate) fn vex_modrm(&mut self, opc: u32, r: Reg, v: Reg, rm: Reg) {
        self.vex_opc(opc, r, v, rm, 0);
        self.b8(0xc0 | low(r) << 3 | low(rm));
    }

    /// `tcg_out_sib_offset`: the ModRM, SIB and displacement bytes of a memory operand. `tail`
    /// is the number of immediate bytes that follow, for rip relative addressing.
    fn sib_offset(&mut self, r: Reg, m: Mem, tail: usize) {
        let (rm, index, offset) = match m {
            Mem::Pool(k) => {
                self.b8(low(r) << 3 | 5);
                self.pool_refs.push((self.pos(), k));
                self.b32(tail as u32);
                return;
            }
            Mem::Base(b, d) => (b, None, d),
            Mem::Index(b, i, d) => (b, Some(i), d),
        };
        let (md, len) = if offset == 0 && low(rm) != RBP {
            (0u8, 0)
        } else if offset == offset as i8 as i32 {
            (0x40, 1)
        } else {
            (0x80, 4)
        };
        match index {
            None if low(rm) != RSP => self.b8(md | low(r) << 3 | low(rm)),
            _ => {
                let index = index.unwrap_or(RSP);
                self.b8(md | low(r) << 3 | 4);
                self.b8(low(index) << 3 | low(rm));
            }
        }
        if len == 1 {
            self.b8(offset as u8);
        } else if len == 4 {
            self.b32(offset as u32);
        }
    }

    fn mem_regs(m: Mem) -> (Reg, Reg) {
        match m {
            Mem::Base(b, _) => (b, 0),
            Mem::Index(b, i, _) => (b, i),
            Mem::Pool(_) => (0, 0),
        }
    }

    /// `tcg_out_modrm_sib_offset` for a legacy encoded op. `tail` counts immediate bytes after
    /// the operand.
    pub(crate) fn modrm_mem_tail(&mut self, opc: u32, r: Reg, m: Mem, tail: usize) {
        let (rm, x) = Asm::mem_regs(m);
        self.opc(opc, r, rm, x);
        self.sib_offset(r, m, tail);
    }

    pub(crate) fn modrm_mem(&mut self, opc: u32, r: Reg, m: Mem) {
        self.modrm_mem_tail(opc, r, m, 0);
    }

    /// `tcg_out_vex_modrm_sib_offset`.
    pub(crate) fn vex_modrm_mem(&mut self, opc: u32, r: Reg, v: Reg, m: Mem) {
        let (rm, x) = Asm::mem_regs(m);
        self.vex_opc(opc, r, v, rm, x);
        self.sib_offset(r, m, 0);
    }

    /// A label relative branch: `jmp` for `code` `None`, else `jcc`. Short forms are used
    /// only when asked for, and checked when the code is finished.
    pub(crate) fn jump(&mut self, code: Option<u8>, l: usize, short: bool) {
        match (code, short) {
            (None, true) => self.b8(op::JMP_SHORT as u8),
            (Some(c), true) => self.b8(op::JCC_SHORT as u8 + c),
            (None, false) => self.b8(op::JMP_LONG as u8),
            (Some(c), false) => {
                self.b8(0x0f);
                self.b8(op::JCC_LONG as u8 + c);
            }
        }
        let at = self.pos();
        if short {
            self.fixups.push(Fixup::Rel8(at, l));
            self.b8(0);
        } else {
            self.fixups.push(Fixup::Rel32(at, l));
            self.b32(0);
        }
    }

    /// Resolve every branch and append the constant pool.
    pub(crate) fn finish(mut self) -> Result<Vec<u8>, AsmError> {
        for f in std::mem::take(&mut self.fixups) {
            match f {
                Fixup::Rel8(at, l) => {
                    let to = self.labels[l].ok_or(AsmError::Unbound)?;
                    let d = to as i64 - (at as i64 + 1);
                    if d != d as i8 as i64 {
                        return Err(AsmError::OutOfRange);
                    }
                    self.code[at] = d as u8;
                }
                Fixup::Rel32(at, l) => {
                    let to = self.labels[l].ok_or(AsmError::Unbound)?;
                    let d = to as i64 - (at as i64 + 4);
                    self.code[at..at + 4].copy_from_slice(&(d as i32).to_le_bytes());
                }
            }
        }
        if !self.pool.is_empty() {
            while self.code.len() % 32 != 0 {
                self.code.push(0xcc);
            }
            let start = self.code.len();
            for d in &self.pool {
                self.code.extend_from_slice(d);
            }
            for &(at, k) in &self.pool_refs {
                let tail = u32::from_le_bytes([
                    self.code[at],
                    self.code[at + 1],
                    self.code[at + 2],
                    self.code[at + 3],
                ]) as i64;
                let next = at as i64 + 4 + tail;
                let d = (start + 32 * k) as i64 - next;
                self.code[at..at + 4].copy_from_slice(&(d as i32).to_le_bytes());
            }
        }
        Ok(self.code)
    }

    // Scalar instructions.

    /// `tcg_out_mov` between general registers.
    pub(crate) fn mov(&mut self, rexw: u32, d: Reg, s: Reg) {
        if d != s {
            self.modrm(op::MOVL_GV_EV | rexw, d, s);
        }
    }

    /// `tcg_out_movi` for a general register. `keep_flags` avoids the `xor` form for zero.
    pub(crate) fn movi(&mut self, is64: bool, d: Reg, v: u64, keep_flags: bool) {
        let v = if is64 { v } else { v as u32 as u64 };
        if v == 0 && !keep_flags {
            self.modrm(arith::XOR as u32 * 8 + op::ARITH_GV_EV, d, d);
        } else if v == v as u32 as u64 {
            self.opc(op::MOVL_IV + low(d) as u32, 0, d, 0);
            self.b32(v as u32);
        } else if v as i64 == v as i32 as i64 {
            self.modrm(op::MOVL_EV_IZ | P_REXW, 0, d);
            self.b32(v as u32);
        } else {
            self.movabs(d, v);
        }
    }

    /// `movabs $v, d`, always ten bytes; the immediate starts two bytes in.
    pub(crate) fn movabs(&mut self, d: Reg, v: u64) {
        self.opc((op::MOVL_IV + low(d) as u32) | P_REXW, 0, d, 0);
        self.b64(v);
    }

    /// `tgen_arithr`: `d = d op s`.
    pub(crate) fn arith(&mut self, code: u8, rexw: u32, d: Reg, s: Reg) {
        self.modrm((op::ARITH_GV_EV + (code as u32) * 8) | rexw, d, s);
    }

    /// `tgen_arithi` for an immediate that fits in 32 bits sign extended.
    pub(crate) fn arithi(&mut self, code: u8, rexw: u32, d: Reg, v: i64) {
        if v == v as i8 as i64 {
            self.modrm(op::ARITH_EV_IB | rexw, code, d);
            self.b8(v as u8);
        } else {
            self.modrm(op::ARITH_EV_IZ | rexw, code, d);
            self.b32(v as u32);
        }
    }

    /// A group 1 op with an immediate on a memory operand.
    pub(crate) fn arithi_mem(&mut self, code: u8, rexw: u32, m: Mem, v: i64) {
        if v == v as i8 as i64 {
            self.modrm_mem_tail(op::ARITH_EV_IB | rexw, code, m, 1);
            self.b8(v as u8);
        } else {
            self.modrm_mem_tail(op::ARITH_EV_IZ | rexw, code, m, 4);
            self.b32(v as u32);
        }
    }

    /// `cmp s, (mem)`: flags from `r - mem`.
    pub(crate) fn cmp_mem(&mut self, rexw: u32, r: Reg, m: Mem) {
        self.modrm_mem((op::ARITH_GV_EV + (arith::CMP as u32) * 8) | rexw, r, m);
    }

    /// `tcg_out_shifti`.
    pub(crate) fn shifti(&mut self, code: u8, rexw: u32, d: Reg, n: u32) {
        if n == 1 {
            self.modrm(op::SHIFT_1 | rexw, code, d);
        } else {
            self.modrm(op::SHIFT_IB | rexw, code, d);
            self.b8(n as u8);
        }
    }

    /// A shift of `d` by `cl`.
    pub(crate) fn shift_cl(&mut self, code: u8, rexw: u32, d: Reg) {
        self.modrm(op::SHIFT_CL | rexw, code, d);
    }

    /// A group 3 op on a register: not, neg, mul, imul, div, idiv.
    pub(crate) fn ext3(&mut self, code: u8, rexw: u32, r: Reg) {
        self.modrm(op::GRP3_EV | rexw, code, r);
    }

    /// `test a, b`.
    pub(crate) fn test(&mut self, rexw: u32, a: Reg, b: Reg) {
        self.modrm(op::TESTL | rexw, b, a);
    }

    /// `test $v, r`, with a 32-bit immediate sign extended for a 64-bit test.
    pub(crate) fn testi(&mut self, rexw: u32, r: Reg, v: u32) {
        self.modrm(op::GRP3_EV | rexw, ext3::TESTI, r);
        self.b32(v);
    }

    /// `bt $bit, r`.
    pub(crate) fn bti(&mut self, rexw: u32, r: Reg, bit: u32) {
        self.modrm(op::GRPBT | rexw, GRPBT_BT, r);
        self.b8(bit as u8);
    }

    /// `bt $bit, mem`.
    pub(crate) fn bti_mem(&mut self, rexw: u32, m: Mem, bit: u32) {
        self.modrm_mem_tail(op::GRPBT | rexw, GRPBT_BT, m, 1);
        self.b8(bit as u8);
    }

    /// `setcc` of the byte register `d`.
    pub(crate) fn setcc(&mut self, code: u8, d: Reg) {
        self.modrm(op::SETCC + code as u32, 0, d);
    }

    /// `setcc` to a byte in memory.
    pub(crate) fn setcc_mem(&mut self, code: u8, m: Mem) {
        self.modrm_mem(op::SETCC + code as u32, 0, m);
    }

    /// `cmovcc s, d`.
    pub(crate) fn cmov(&mut self, code: u8, rexw: u32, d: Reg, s: Reg) {
        self.modrm((op::CMOVCC + code as u32) | rexw, d, s);
    }

    /// `lea disp(base[, index]), d`.
    pub(crate) fn lea(&mut self, rexw: u32, d: Reg, m: Mem) {
        self.modrm_mem(op::LEA | rexw, d, m);
    }

    /// `push r`.
    pub(crate) fn push(&mut self, r: Reg) {
        self.opc(op::PUSH_R32 + low(r) as u32, 0, r, 0);
    }

    /// `pop r`.
    pub(crate) fn pop(&mut self, r: Reg) {
        self.opc(op::POP_R32 + low(r) as u32, 0, r, 0);
    }

    /// `bswap r`.
    pub(crate) fn bswap(&mut self, rexw: u32, r: Reg) {
        self.opc((op::BSWAP + low(r) as u32) | rexw, 0, r, 0);
    }

    /// `rolw $8, r`: swap the two low bytes.
    pub(crate) fn rolw8(&mut self, r: Reg) {
        self.modrm(op::SHIFT_IB | P_DATA16, shift::ROL, r);
        self.b8(8);
    }

    /// `call *r`.
    pub(crate) fn call_reg(&mut self, r: Reg) {
        self.modrm(op::GRP5, ext5::CALLN_EV, r);
    }

    /// `lock orl $0, (%rsp)`, `tcg_out_mb`.
    pub(crate) fn mb(&mut self) {
        self.b8(0xf0);
        self.modrm_mem_tail(op::ARITH_EV_IB, arith::OR, Mem::Base(RSP, 0), 1);
        self.b8(0);
    }

    /// Load `size` bytes from `m` into `d`, zero or sign extending to 32 or 64 bits.
    pub(crate) fn load(&mut self, d: Reg, m: Mem, size: u32, signed: bool, rexw: u32) {
        match (size, signed) {
            (1, false) => self.modrm_mem(op::MOVZBL, d, m),
            (1, true) => self.modrm_mem(op::MOVSBL | rexw, d, m),
            (2, false) => self.modrm_mem(op::MOVZWL, d, m),
            (2, true) => self.modrm_mem(op::MOVSWL | rexw, d, m),
            (4, true) if rexw != 0 => self.modrm_mem(op::MOVSLQ, d, m),
            (4, _) => self.modrm_mem(op::MOVL_GV_EV, d, m),
            _ => self.modrm_mem(op::MOVL_GV_EV | P_REXW, d, m),
        }
    }

    /// Store the low `size` bytes of `s` to `m`.
    pub(crate) fn store(&mut self, s: Reg, m: Mem, size: u32) {
        match size {
            1 => self.modrm_mem(op::MOVB_EV_GV | P_REXB_R, s, m),
            2 => self.modrm_mem(op::MOVL_EV_GV | P_DATA16, s, m),
            4 => self.modrm_mem(op::MOVL_EV_GV, s, m),
            _ => self.modrm_mem(op::MOVL_EV_GV | P_REXW, s, m),
        }
    }

    /// Store the constant `v` in `size` bytes at `m`; for 8 bytes it must fit 32 bits signed.
    pub(crate) fn store_imm(&mut self, v: u64, m: Mem, size: u32) {
        match size {
            1 => {
                self.modrm_mem_tail(op::MOVB_EV_IZ, 0, m, 1);
                self.b8(v as u8);
            }
            2 => {
                self.modrm_mem_tail(op::MOVL_EV_IZ | P_DATA16, 0, m, 2);
                self.code.extend_from_slice(&(v as u16).to_le_bytes());
            }
            4 => {
                self.modrm_mem_tail(op::MOVL_EV_IZ, 0, m, 4);
                self.b32(v as u32);
            }
            _ => {
                self.modrm_mem_tail(op::MOVL_EV_IZ | P_REXW, 0, m, 4);
                self.b32(v as u32);
            }
        }
    }

    /// A one byte instruction.
    pub(crate) fn raw(&mut self, opc: u32) {
        self.b8(opc as u8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc(f: impl FnOnce(&mut Asm)) -> Vec<u8> {
        let mut a = Asm::new();
        f(&mut a);
        a.finish().unwrap()
    }

    #[test]
    fn encodings_match_the_manual() {
        // mov %rbx, %rax
        assert_eq!(enc(|a| a.mov(P_REXW, RAX, RBX)), [0x48, 0x8b, 0xc3]);
        // mov %r9d, %r12d
        assert_eq!(enc(|a| a.mov(0, R12, R9)), [0x45, 0x8b, 0xe1]);
        // add $1, %r13
        assert_eq!(enc(|a| a.arithi(arith::ADD, P_REXW, R13, 1)), [0x49, 0x83, 0xc5, 0x01]);
        // mov 0x10(%rbp), %rax
        assert_eq!(
            enc(|a| a.load(RAX, Mem::Base(RBP, 0x10), 8, false, P_REXW)),
            [0x48, 0x8b, 0x45, 0x10]
        );
        // mov (%rbp,%r11,1), %ecx
        assert_eq!(
            enc(|a| a.load(RCX, Mem::Index(RBP, R11, 0), 4, false, 0)),
            [0x42, 0x8b, 0x4c, 0x1d, 0x00]
        );
        // mov %rax, (%r12)
        assert_eq!(enc(|a| a.store(RAX, Mem::Base(R12, 0), 8)), [0x49, 0x89, 0x04, 0x24]);
        // movb %sil, (%r14)
        assert_eq!(enc(|a| a.store(RSI, Mem::Base(R14, 0), 1)), [0x41, 0x88, 0x36]);
        // sete %dil
        assert_eq!(enc(|a| a.setcc(cc::E, RDI)), [0x40, 0x0f, 0x94, 0xc7]);
        // movabs $0x1122334455667788, %r11
        assert_eq!(
            enc(|a| a.movabs(R11, 0x1122_3344_5566_7788)),
            [0x49, 0xbb, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11]
        );
        // vpaddd %xmm2, %xmm1, %xmm0
        assert_eq!(
            enc(|a| a.vex_modrm(op::PADDD, xmm(0), xmm(1), xmm(2))),
            [0xc5, 0xf1, 0xfe, 0xc2]
        );
        // vpaddq %ymm10, %ymm9, %ymm8
        assert_eq!(
            enc(|a| a.vex_modrm(op::PADDQ | P_VEXL, xmm(8), xmm(9), xmm(10))),
            [0xc4, 0x41, 0x35, 0xd4, 0xc2]
        );
        // pcmpgtq %xmm1, %xmm0
        assert_eq!(enc(|a| a.modrm(op::PCMPGTQ, xmm(0), xmm(1))), [0x66, 0x0f, 0x38, 0x37, 0xc1]);
        // shlx %rcx, %rax, %rdx
        assert_eq!(
            enc(|a| a.vex_modrm(op::SHLX | P_REXW, RDX, RCX, RAX)),
            [0xc4, 0xe2, 0xf1, 0xf7, 0xd0]
        );
        // lock orl $0, (%rsp)
        assert_eq!(enc(|a| a.mb()), [0xf0, 0x83, 0x0c, 0x24, 0x00]);
    }

    #[test]
    fn branches_and_pool_are_patched() {
        let mut a = Asm::new();
        let l = a.new_label();
        a.jump(Some(cc::E), l, true);
        a.jump(None, l, false);
        a.bind(l);
        let k = a.pool_entry([7; 32]);
        a.modrm_mem(op::MOVDQU_VX_WX, xmm(1), Mem::Pool(k));
        let code = a.finish().unwrap();
        assert_eq!(&code[..7], &[0x74, 0x05, 0xe9, 0, 0, 0, 0]);
        // movdqu disp32(%rip), %xmm1 at offset 7: f3 0f 6f 0d disp32, ending at 15.
        let disp = i32::from_le_bytes([code[11], code[12], code[13], code[14]]);
        assert_eq!(15 + disp as usize, 32);
        assert_eq!(code[32..64], [7; 32]);
    }
}
