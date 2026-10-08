// SPDX-License-Identifier: GPL-2.0-or-later

//! The RV64 encoder: instruction words, labels, relocations and the constant pool.
//!
//! This is the instruction emitting half of QEMU's `tcg/riscv64/tcg-target.c.inc`: the
//! `RISCVInsn` opcodes, the `encode_*` format helpers, `tcg_out_movi`, `tcg_out_ldst` and the
//! extensions, plus the relocation and pool machinery from `tcg/tcg.c` and
//! `tcg/tcg-pool.c.inc`. Code is assembled into a vector of words whose final address is known
//! up front, so PC relative forms (`auipc` pairs) can be chosen while emitting. No compressed
//! instructions are emitted, as in QEMU.

use ruvm_jit_core::types::{Cond, Type};

use crate::features::HostFeatures;

/// A host register: 0 to 31 are x0 to x31.
pub(crate) type Reg = u8;

pub(crate) const ZERO: Reg = 0;
pub(crate) const RA: Reg = 1;
pub(crate) const SP: Reg = 2;
pub(crate) const A0: Reg = 10;
pub(crate) const A1: Reg = 11;
pub(crate) const A2: Reg = 12;
/// `TCG_REG_TMP0`, t6.
pub(crate) const TMP0: Reg = 31;
/// `TCG_REG_TMP1`, t5.
pub(crate) const TMP1: Reg = 30;
/// `TCG_REG_TMP2`, t4, which [`Asm::ldst`] uses for an offset that does not fit 12 bits.
pub(crate) const TMP2: Reg = 29;
/// t3, a fourth scratch this port reserves for its own expansions.
pub(crate) const TMP3: Reg = 28;

/// `RISCVInsn`: the fixed bits of each instruction.
pub(crate) mod opc {
    pub(crate) const ADD: u32 = 0x33;
    pub(crate) const ADDI: u32 = 0x13;
    pub(crate) const AND: u32 = 0x7033;
    pub(crate) const ANDI: u32 = 0x7013;
    pub(crate) const AUIPC: u32 = 0x17;
    pub(crate) const BEQ: u32 = 0x63;
    pub(crate) const BGE: u32 = 0x5063;
    pub(crate) const BGEU: u32 = 0x7063;
    pub(crate) const BLT: u32 = 0x4063;
    pub(crate) const BLTU: u32 = 0x6063;
    pub(crate) const BNE: u32 = 0x1063;
    pub(crate) const DIV: u32 = 0x2004033;
    pub(crate) const DIVU: u32 = 0x2005033;
    pub(crate) const JAL: u32 = 0x6f;
    pub(crate) const JALR: u32 = 0x67;
    pub(crate) const LB: u32 = 0x3;
    pub(crate) const LBU: u32 = 0x4003;
    pub(crate) const LD: u32 = 0x3003;
    pub(crate) const LH: u32 = 0x1003;
    pub(crate) const LHU: u32 = 0x5003;
    pub(crate) const LUI: u32 = 0x37;
    pub(crate) const LW: u32 = 0x2003;
    pub(crate) const LWU: u32 = 0x6003;
    pub(crate) const MUL: u32 = 0x2000033;
    pub(crate) const MULH: u32 = 0x2001033;
    pub(crate) const MULHSU: u32 = 0x2002033;
    pub(crate) const MULHU: u32 = 0x2003033;
    pub(crate) const OR: u32 = 0x6033;
    pub(crate) const ORI: u32 = 0x6013;
    pub(crate) const REM: u32 = 0x2006033;
    pub(crate) const REMU: u32 = 0x2007033;
    pub(crate) const SB: u32 = 0x23;
    pub(crate) const SD: u32 = 0x3023;
    pub(crate) const SH: u32 = 0x1023;
    pub(crate) const SLL: u32 = 0x1033;
    pub(crate) const SLLI: u32 = 0x1013;
    pub(crate) const SLT: u32 = 0x2033;
    pub(crate) const SLTI: u32 = 0x2013;
    pub(crate) const SLTIU: u32 = 0x3013;
    pub(crate) const SLTU: u32 = 0x3033;
    pub(crate) const SRA: u32 = 0x40005033;
    pub(crate) const SRAI: u32 = 0x40005013;
    pub(crate) const SRL: u32 = 0x5033;
    pub(crate) const SRLI: u32 = 0x5013;
    pub(crate) const SUB: u32 = 0x40000033;
    pub(crate) const SW: u32 = 0x2023;
    pub(crate) const XOR: u32 = 0x4033;
    pub(crate) const XORI: u32 = 0x4013;

    pub(crate) const ADDIW: u32 = 0x1b;
    pub(crate) const ADDW: u32 = 0x3b;
    pub(crate) const DIVUW: u32 = 0x200503b;
    pub(crate) const DIVW: u32 = 0x200403b;
    pub(crate) const MULW: u32 = 0x200003b;
    pub(crate) const REMUW: u32 = 0x200703b;
    pub(crate) const REMW: u32 = 0x200603b;
    pub(crate) const SLLIW: u32 = 0x101b;
    pub(crate) const SLLW: u32 = 0x103b;
    pub(crate) const SRAIW: u32 = 0x4000501b;
    pub(crate) const SRAW: u32 = 0x4000503b;
    pub(crate) const SRLIW: u32 = 0x501b;
    pub(crate) const SRLW: u32 = 0x503b;
    pub(crate) const SUBW: u32 = 0x4000003b;

    pub(crate) const FENCE: u32 = 0x0000000f;
    pub(crate) const NOP: u32 = ADDI;

    // Zba: bit manipulation extension, address generation.
    pub(crate) const ADD_UW: u32 = 0x0800003b;

    // Zbb: bit manipulation extension, basic bit manipulation.
    pub(crate) const ANDN: u32 = 0x40007033;
    pub(crate) const CLZ: u32 = 0x60001013;
    pub(crate) const CLZW: u32 = 0x6000101b;
    pub(crate) const CPOP: u32 = 0x60201013;
    pub(crate) const CPOPW: u32 = 0x6020101b;
    pub(crate) const CTZ: u32 = 0x60101013;
    pub(crate) const CTZW: u32 = 0x6010101b;
    pub(crate) const ORN: u32 = 0x40006033;
    pub(crate) const REV8: u32 = 0x6b805013;
    pub(crate) const ROL: u32 = 0x60001033;
    pub(crate) const ROLW: u32 = 0x6000103b;
    pub(crate) const ROR: u32 = 0x60005033;
    pub(crate) const RORW: u32 = 0x6000503b;
    pub(crate) const RORI: u32 = 0x60005013;
    pub(crate) const RORIW: u32 = 0x6000501b;
    pub(crate) const SEXT_B: u32 = 0x60401013;
    pub(crate) const SEXT_H: u32 = 0x60501013;
    pub(crate) const XNOR: u32 = 0x40004033;
    pub(crate) const ZEXT_H: u32 = 0x0800403b;

    // Zbs: bit manipulation extension, single bit instructions.
    pub(crate) const BEXTI: u32 = 0x48005013;

    // Zicond: integer conditional operations.
    pub(crate) const CZERO_EQZ: u32 = 0x0e005033;
    pub(crate) const CZERO_NEZ: u32 = 0x0e007033;
}

/// The `fence` bits of each ordering, `tcg_out_mb`.
pub(crate) mod fence {
    pub(crate) const LD_LD: u32 = 0x02200000;
    pub(crate) const ST_LD: u32 = 0x01200000;
    pub(crate) const LD_ST: u32 = 0x02100000;
    pub(crate) const ST_ST: u32 = 0x01100000;
}

/// `sextreg(v, 0, bits)`.
pub(crate) fn sext(v: i64, bits: u32) -> i64 {
    (v << (64 - bits)) >> (64 - bits)
}

/// Whether `v` is a 12-bit signed immediate.
pub(crate) fn is_imm12(v: i64) -> bool {
    v == sext(v, 12)
}

/// `encode_r`.
pub(crate) fn encode_r(op: u32, rd: Reg, rs1: Reg, rs2: Reg) -> u32 {
    op | (rd as u32 & 0x1f) << 7 | (rs1 as u32 & 0x1f) << 15 | (rs2 as u32 & 0x1f) << 20
}

/// `encode_i`.
pub(crate) fn encode_i(op: u32, rd: Reg, rs1: Reg, imm: i32) -> u32 {
    op | (rd as u32 & 0x1f) << 7 | (rs1 as u32 & 0x1f) << 15 | (imm as u32 & 0xfff) << 20
}

/// `encode_s`.
pub(crate) fn encode_s(op: u32, rs1: Reg, rs2: Reg, imm: i32) -> u32 {
    let imm = imm as u32;
    op | (rs1 as u32 & 0x1f) << 15
        | (rs2 as u32 & 0x1f) << 20
        | (imm & 0xfe0) << 20
        | (imm & 0x1f) << 7
}

/// `encode_sbimm12`.
fn encode_sbimm12(imm: u32) -> u32 {
    (imm & 0x1000) << 19 | (imm & 0x7e0) << 20 | (imm & 0x1e) << 7 | (imm & 0x800) >> 4
}

/// `encode_sb`.
pub(crate) fn encode_sb(op: u32, rs1: Reg, rs2: Reg, imm: i32) -> u32 {
    op | (rs1 as u32 & 0x1f) << 15 | (rs2 as u32 & 0x1f) << 20 | encode_sbimm12(imm as u32)
}

/// `encode_u`: `imm` holds the value of the upper 20 bits in place.
pub(crate) fn encode_u(op: u32, rd: Reg, imm: i32) -> u32 {
    op | (rd as u32 & 0x1f) << 7 | (imm as u32 & 0xfffff000)
}

/// `encode_ujimm20`.
fn encode_ujimm20(imm: u32) -> u32 {
    (imm & 0x0007fe) << 20 | (imm & 0x000800) << 9 | (imm & 0x0ff000) | (imm & 0x100000) << 11
}

/// `encode_uj`.
pub(crate) fn encode_uj(op: u32, rd: Reg, imm: i32) -> u32 {
    op | (rd as u32 & 0x1f) << 7 | encode_ujimm20(imm as u32)
}

/// `jal zero` over `disp` bytes, if it reaches, for patching `goto_tb`.
pub(crate) fn jal_word(disp: i64) -> Option<u32> {
    (disp == sext(disp, 21) && disp & 1 == 0).then(|| encode_uj(opc::JAL, ZERO, disp as i32))
}

/// `tcg_brcond_to_riscv`: the branch for `c` and whether its operands are swapped. `Never`,
/// `Always` and the test conditions have none.
pub(crate) fn brcond_insn(c: Cond) -> Option<(u32, bool)> {
    Some(match c {
        Cond::Eq => (opc::BEQ, false),
        Cond::Ne => (opc::BNE, false),
        Cond::Lt => (opc::BLT, false),
        Cond::Ge => (opc::BGE, false),
        Cond::Le => (opc::BGE, true),
        Cond::Gt => (opc::BLT, true),
        Cond::Ltu => (opc::BLTU, false),
        Cond::Geu => (opc::BGEU, false),
        Cond::Leu => (opc::BGEU, true),
        Cond::Gtu => (opc::BLTU, true),
        _ => return None,
    })
}

/// The branch taken when `b` is not: the low bit of `funct3` inverts every RISC-V branch.
pub(crate) const fn invert_branch(b: u32) -> u32 {
    b ^ 0x1000
}

/// Relocation kinds, as in QEMU's `patch_reloc`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reloc {
    /// `R_RISCV_BRANCH`: a conditional branch, 13-bit byte displacement.
    Branch,
    /// `R_RISCV_JAL`: `jal`, 21-bit byte displacement.
    Jal,
    /// `R_RISCV_CALL`: an `auipc` and the I-type instruction after it.
    Call,
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
    /// The `auipc` of the `auipc` and `ld` pair that loads the entry.
    at: usize,
    val: u64,
    /// The handle of an entry that is never shared, so that it can be patched alone.
    unique: Option<usize>,
}

/// An assembler for one block of code whose first word will live at `base`.
#[derive(Debug)]
pub(crate) struct Asm {
    pub(crate) code: Vec<u32>,
    base: u64,
    labels: Vec<Option<usize>>,
    fixups: Vec<Fixup>,
    pool: Vec<PoolRef>,
    uniques: usize,
    /// The extensions the code may use.
    pub(crate) feat: HostFeatures,
}

/// The result of [`Asm::finish`].
#[derive(Debug)]
pub(crate) struct Assembled {
    pub(crate) bytes: Vec<u8>,
    /// Byte offset of the first word after the instructions and pool.
    pub(crate) end: usize,
    /// For each [`Asm::pool_unique`] entry, by handle, its byte offset.
    pub(crate) uniques: Vec<usize>,
}

impl Asm {
    pub(crate) fn new(base: u64, feat: HostFeatures) -> Asm {
        Asm {
            code: Vec::new(),
            base,
            labels: Vec::new(),
            fixups: Vec::new(),
            pool: Vec::new(),
            uniques: 0,
            feat,
        }
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

    /// Patch the field of `kind` at `at` for a displacement of `disp` bytes, `patch_reloc`.
    fn patch(code: &mut [u32], at: usize, kind: Reloc, disp: i64) -> Result<(), AsmError> {
        match kind {
            Reloc::Branch => {
                if disp != sext(disp, 13) {
                    return Err(AsmError::OutOfRange);
                }
                code[at] |= encode_sbimm12(disp as u32);
            }
            Reloc::Jal => {
                if disp != sext(disp, 21) {
                    return Err(AsmError::OutOfRange);
                }
                code[at] |= encode_ujimm20(disp as u32);
            }
            Reloc::Call => {
                let lo = sext(disp, 12);
                let hi = disp - lo;
                if hi != sext(hi, 32) {
                    return Err(AsmError::OutOfRange);
                }
                code[at] |= hi as u32 & 0xfffff000;
                code[at + 1] |= (lo as u32 & 0xfff) << 20;
            }
        }
        Ok(())
    }

    /// Resolve labels, lay out the pool after the code, 8-byte aligned, and return the bytes.
    pub(crate) fn finish(mut self) -> Result<Assembled, AsmError> {
        for f in std::mem::take(&mut self.fixups) {
            let target = self.labels[f.label].expect("branch to a label that was never bound");
            let disp = (target as i64 - f.at as i64) * 4;
            Self::patch(&mut self.code, f.at, f.kind, disp)?;
        }
        let mut uniques = vec![0; self.uniques];
        if !self.pool.is_empty() {
            if self.addr_of(self.code.len()) % 8 != 0 {
                self.code.push(opc::NOP);
            }
            let mut placed: Vec<(u64, usize)> = Vec::new();
            for p in std::mem::take(&mut self.pool) {
                let shared = match p.unique {
                    None => placed.iter().find(|e| e.0 == p.val).map(|e| e.1),
                    Some(_) => None,
                };
                let at = match shared {
                    Some(at) => at,
                    None => {
                        let at = self.code.len();
                        self.code.push(p.val as u32);
                        self.code.push((p.val >> 32) as u32);
                        if p.unique.is_none() {
                            placed.push((p.val, at));
                        }
                        at
                    }
                };
                if let Some(h) = p.unique {
                    uniques[h] = at * 4;
                }
                let disp = (at as i64 - p.at as i64) * 4;
                Self::patch(&mut self.code, p.at, Reloc::Call, disp)?;
            }
        }
        let end = self.code.len() * 4;
        let mut bytes = Vec::with_capacity(end);
        for w in &self.code {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        Ok(Assembled { bytes, end, uniques })
    }

    // Instruction formats, `tcg_out_opc_*`.

    /// `tcg_out_opc_reg`.
    pub(crate) fn r(&mut self, op: u32, rd: Reg, rs1: Reg, rs2: Reg) {
        self.emit(encode_r(op, rd, rs1, rs2));
    }

    /// `tcg_out_opc_imm`.
    pub(crate) fn i(&mut self, op: u32, rd: Reg, rs1: Reg, imm: i64) {
        debug_assert!(is_imm12(imm) || op & 0x7f == 0x13 || op & 0x7f == 0x1b);
        self.emit(encode_i(op, rd, rs1, imm as i32));
    }

    /// `tcg_out_opc_store`.
    pub(crate) fn s(&mut self, op: u32, base: Reg, data: Reg, imm: i64) {
        self.emit(encode_s(op, base, data, imm as i32));
    }

    /// `tcg_out_opc_upper`.
    pub(crate) fn u(&mut self, op: u32, rd: Reg, imm: i64) {
        self.emit(encode_u(op, rd, imm as i32));
    }

    /// A conditional branch to `l`, which must be within 4 KiB.
    pub(crate) fn b_label(&mut self, op: u32, rs1: Reg, rs2: Reg, l: usize) {
        self.reloc_here(Reloc::Branch, l);
        self.emit(encode_sb(op, rs1, rs2, 0));
    }

    /// `jal zero` to `l`, `tcg_out_br`.
    pub(crate) fn j_label(&mut self, l: usize) {
        self.reloc_here(Reloc::Jal, l);
        self.emit(encode_uj(opc::JAL, ZERO, 0));
    }

    /// A conditional branch to `l` anywhere in the block: the inverted branch over a `jal`.
    pub(crate) fn far_b_label(&mut self, op: u32, rs1: Reg, rs2: Reg, l: usize) {
        self.emit(encode_sb(invert_branch(op), rs1, rs2, 8));
        self.j_label(l);
    }

    /// `jalr zero, 0(rs)`.
    pub(crate) fn jr(&mut self, rs: Reg) {
        self.i(opc::JALR, ZERO, rs, 0);
    }

    /// A call to the absolute address `addr`, `tcg_out_call_int`: `jal` when in reach, else
    /// an `auipc` pair, else the address in TMP0.
    pub(crate) fn call_abs(&mut self, addr: u64) {
        let disp = addr.wrapping_sub(self.here()) as i64;
        if let Some(w) = jal_word(disp) {
            self.emit(w | (RA as u32) << 7);
            return;
        }
        let lo = sext(disp, 12);
        let hi = disp.wrapping_sub(lo);
        if hi == sext(hi, 32) {
            self.u(opc::AUIPC, TMP0, hi);
            self.i(opc::JALR, RA, TMP0, lo);
        } else {
            self.movi(Type::I64, TMP0, addr as i64);
            self.i(opc::JALR, RA, TMP0, 0);
        }
    }

    /// `tcg_out_mov` for general registers.
    pub(crate) fn mov(&mut self, rd: Reg, rs: Reg) {
        if rd != rs {
            self.i(opc::ADDI, rd, rs, 0);
        }
    }

    /// `tcg_out_movi`.
    pub(crate) fn movi(&mut self, ty: Type, rd: Reg, val: i64) {
        let val = if ty == Type::I32 { val as i32 as i64 } else { val };
        let lo = sext(val, 12);
        if val == lo {
            self.i(opc::ADDI, rd, ZERO, lo);
            return;
        }
        let hi = val.wrapping_sub(lo);
        if val == val as i32 as i64 {
            self.u(opc::LUI, rd, hi);
            if lo != 0 {
                self.i(opc::ADDIW, rd, rd, lo);
            }
            return;
        }
        let pc = (val as u64).wrapping_sub(self.here()) as i64;
        let plo = sext(pc, 12);
        let phi = pc.wrapping_sub(plo);
        if pc == pc as i32 as i64 && phi == sext(phi, 32) {
            self.u(opc::AUIPC, rd, phi);
            self.i(opc::ADDI, rd, rd, plo);
            return;
        }
        // Look for a single 20-bit section.
        let shift = val.trailing_zeros();
        let tmp = val >> shift;
        if tmp == sext(tmp, 20) {
            self.u(opc::LUI, rd, tmp << 12);
            if shift > 12 {
                self.i(opc::SLLI, rd, rd, (shift - 12) as i64);
            } else {
                self.i(opc::SRAI, rd, rd, (12 - shift) as i64);
            }
            return;
        }
        // Look for a few high zero bits, with lots of bits set in the middle.
        let shift = val.leading_zeros();
        let tmp = val << shift;
        if tmp == sext(tmp >> 12, 20) << 12 {
            self.u(opc::LUI, rd, tmp);
            self.i(opc::SRLI, rd, rd, shift as i64);
            return;
        } else if tmp == sext(tmp, 12) {
            self.i(opc::ADDI, rd, ZERO, tmp);
            self.i(opc::SRLI, rd, rd, shift as i64);
            return;
        }
        // Drop into the constant pool.
        self.pool_load(rd, val as u64);
    }

    /// Load the 64-bit pool entry holding `val` into `rd`.
    pub(crate) fn pool_load(&mut self, rd: Reg, val: u64) {
        let at = self.pos();
        self.pool.push(PoolRef { at, val, unique: None });
        self.u(opc::AUIPC, rd, 0);
        self.i(opc::LD, rd, rd, 0);
    }

    /// Load a pool entry of its own into `rd`, whose value can be changed with
    /// [`Asm::set_unique`] until the block is finished and whose offset [`Asm::finish`]
    /// reports. Returns its handle.
    pub(crate) fn pool_unique(&mut self, rd: Reg, val: u64) -> usize {
        let h = self.uniques;
        self.uniques += 1;
        let at = self.pos();
        self.pool.push(PoolRef { at, val, unique: Some(h) });
        self.u(opc::AUIPC, rd, 0);
        self.i(opc::LD, rd, rd, 0);
        h
    }

    /// Change the value of the unique pool entry `h`.
    pub(crate) fn set_unique(&mut self, h: usize, val: u64) {
        for p in &mut self.pool {
            if p.unique == Some(h) {
                p.val = val;
            }
        }
    }

    /// `tcg_out_ldst`: a load or store of `data` at `base + offset`. An offset that does not
    /// fit 12 bits goes through TMP2.
    pub(crate) fn ldst(&mut self, op: u32, data: Reg, base: Reg, offset: i64) {
        let imm12 = sext(offset, 12);
        let mut addr = base;
        if offset != imm12 {
            self.movi(Type::I64, TMP2, offset - imm12);
            if base != ZERO {
                self.r(opc::ADD, TMP2, TMP2, base);
            }
            addr = TMP2;
        }
        if op & 0x7f == 0x23 {
            self.s(op, addr, data, imm12);
        } else {
            self.i(op, data, addr, imm12);
        }
    }

    /// `tcg_out_ld` for general registers.
    pub(crate) fn ld(&mut self, ty: Type, rd: Reg, base: Reg, off: i64) {
        self.ldst(if ty == Type::I32 { opc::LW } else { opc::LD }, rd, base, off);
    }

    /// `tcg_out_st` for general registers.
    pub(crate) fn st(&mut self, ty: Type, rs: Reg, base: Reg, off: i64) {
        self.ldst(if ty == Type::I32 { opc::SW } else { opc::SD }, rs, base, off);
    }

    /// `rd = rs + v` in 64 bits. Uses TMP2 for a `v` that does not fit 12 bits.
    pub(crate) fn addi(&mut self, rd: Reg, rs: Reg, v: i64) {
        if is_imm12(v) {
            if v != 0 || rd != rs {
                self.i(opc::ADDI, rd, rs, v);
            }
        } else {
            self.movi(Type::I64, TMP2, v);
            self.r(opc::ADD, rd, rs, TMP2);
        }
    }

    /// `tcg_out_ext8u`.
    pub(crate) fn ext8u(&mut self, rd: Reg, rs: Reg) {
        self.i(opc::ANDI, rd, rs, 0xff);
    }

    /// `tcg_out_ext16u`.
    pub(crate) fn ext16u(&mut self, rd: Reg, rs: Reg) {
        if self.feat.zbb {
            self.r(opc::ZEXT_H, rd, rs, ZERO);
        } else {
            self.i(opc::SLLIW, rd, rs, 16);
            self.i(opc::SRLIW, rd, rd, 16);
        }
    }

    /// `tcg_out_ext32u`.
    pub(crate) fn ext32u(&mut self, rd: Reg, rs: Reg) {
        if self.feat.zba {
            self.r(opc::ADD_UW, rd, rs, ZERO);
        } else {
            self.i(opc::SLLI, rd, rs, 32);
            self.i(opc::SRLI, rd, rd, 32);
        }
    }

    /// `tcg_out_ext8s`.
    pub(crate) fn ext8s(&mut self, rd: Reg, rs: Reg) {
        if self.feat.zbb {
            self.i(opc::SEXT_B, rd, rs, 0);
        } else {
            self.i(opc::SLLIW, rd, rs, 24);
            self.i(opc::SRAIW, rd, rd, 24);
        }
    }

    /// `tcg_out_ext16s`.
    pub(crate) fn ext16s(&mut self, rd: Reg, rs: Reg) {
        if self.feat.zbb {
            self.i(opc::SEXT_H, rd, rs, 0);
        } else {
            self.i(opc::SLLIW, rd, rs, 16);
            self.i(opc::SRAIW, rd, rd, 16);
        }
    }

    /// `tcg_out_ext32s`.
    pub(crate) fn ext32s(&mut self, rd: Reg, rs: Reg) {
        self.i(opc::ADDIW, rd, rs, 0);
    }

    /// `tcg_out_mb` for the `TCG_MO_*` bits of `mo`.
    pub(crate) fn mb(&mut self, mo: u32) {
        use ruvm_jit_core::types::mo;
        let mut insn = opc::FENCE;
        if mo & mo::LD_LD != 0 {
            insn |= fence::LD_LD;
        }
        if mo & mo::ST_LD != 0 {
            insn |= fence::ST_LD;
        }
        if mo & mo::LD_ST != 0 {
            insn |= fence::LD_ST;
        }
        if mo & mo::ST_ST != 0 {
            insn |= fence::ST_ST;
        }
        self.emit(insn);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(f: impl FnOnce(&mut Asm)) -> Vec<u32> {
        let mut a = Asm::new(0x10_0000, HostFeatures::ALL);
        f(&mut a);
        a.code
    }

    #[test]
    fn encodings_match_gnu_as() {
        // Reference words from `riscv64-linux-gnu-as -march=rv64gc_zba_zbb_zbs_zicond`.
        assert_eq!(one(|a| a.r(opc::ADD, 10, 11, 12)), [0x00c58533]);
        assert_eq!(one(|a| a.i(opc::ADDI, 10, 11, -1)), [0xfff58513]);
        assert_eq!(one(|a| a.ext32s(10, 10)), [0x0005051b]);
        assert_eq!(one(|a| a.s(opc::SD, SP, RA, 104)), [0x06113423]);
        assert_eq!(one(|a| a.ld(Type::I64, 8, 18, 8)), [0x00893403]);
        assert_eq!(one(|a| a.ld(Type::I32, 10, 8, -4)), [0xffc42503]);
        assert_eq!(one(|a| a.emit(encode_sb(opc::BEQ, 10, 11, 16))), [0x00b50863]);
        assert_eq!(one(|a| a.emit(encode_uj(opc::JAL, ZERO, -8))), [0xff9ff06f]);
        assert_eq!(one(|a| a.u(opc::LUI, 10, 0x12345 << 12)), [0x12345537]);
        assert_eq!(one(|a| a.u(opc::AUIPC, TMP0, 0)), [0x00000f97]);
        assert_eq!(one(|a| a.jr(RA)), [0x00008067]);
        assert_eq!(one(|a| a.i(opc::SLLI, 10, 11, 63)), [0x03f59513]);
        assert_eq!(one(|a| a.i(opc::SRAI, 10, 11, 32)), [0x4205d513]);
        assert_eq!(one(|a| a.i(opc::SRAIW, 10, 11, 31)), [0x41f5d51b]);
        assert_eq!(one(|a| a.ext32u(10, 11)), [0x0805853b]);
        assert_eq!(one(|a| a.r(opc::ANDN, 10, 11, 12)), [0x40c5f533]);
        assert_eq!(one(|a| a.i(opc::REV8, 10, 11, 0)), [0x6b85d513]);
        assert_eq!(one(|a| a.i(opc::RORI, 10, 11, 5)), [0x6055d513]);
        assert_eq!(one(|a| a.i(opc::RORIW, 10, 11, 5)), [0x6055d51b]);
        assert_eq!(one(|a| a.r(opc::CZERO_EQZ, 10, 11, 12)), [0x0ec5d533]);
        assert_eq!(one(|a| a.i(opc::BEXTI, 10, 11, 63)), [0x4bf5d513]);
        assert_eq!(one(|a| a.mb(0xf)), [0x0330000f]);
        assert_eq!(one(|a| a.ext8s(10, 11)), [0x60459513]);
        assert_eq!(one(|a| a.ext16u(10, 11)), [0x0805c53b]);
        assert_eq!(one(|a| a.i(opc::CLZW, 10, 11, 0)), [0x6005951b]);
        assert_eq!(one(|a| a.i(opc::CPOP, 10, 11, 0)), [0x60259513]);
        assert_eq!(one(|a| a.r(opc::MULHU, 10, 11, 12)), [0x02c5b533]);
        assert_eq!(one(|a| a.r(opc::REMUW, 10, 11, 12)), [0x02c5f53b]);
        assert_eq!(one(|a| a.i(opc::SLTIU, 10, 11, 1)), [0x0015b513]);
        assert_eq!(one(|a| a.s(opc::SB, TMP0, ZERO, 2047)), [0x7e0f8fa3]);
        assert_eq!(one(|a| a.emit(encode_sb(opc::BLTU, 9, TMP0, -4096))), [0x81f4e063]);
        assert_eq!(jal_word(1048574), Some(0x7ffff06f));
        assert_eq!(jal_word(1048576), None);
    }

    #[test]
    fn fences() {
        use ruvm_jit_core::types::mo;
        // `fence r,rw` and `fence rw,w`.
        assert_eq!(one(|a| a.mb(mo::LD_LD | mo::LD_ST)), [0x0230000f]);
        assert_eq!(one(|a| a.mb(mo::LD_ST | mo::ST_ST)), [0x0310000f]);
    }

    /// Run the words `movi` produced on a tiny interpreter of the instructions it uses.
    fn eval_movi(base: u64, val: i64, ty: Type) -> (u64, usize) {
        let mut a = Asm::new(base, HostFeatures::BASELINE);
        a.movi(ty, 10, val);
        let n = a.code.len();
        let out = a.finish().unwrap();
        let word = |i: usize| u32::from_le_bytes(out.bytes[4 * i..4 * i + 4].try_into().unwrap());
        let mut x = [0u64; 32];
        for k in 0..n {
            let w = word(k);
            let rd = ((w >> 7) & 31) as usize;
            let rs1 = ((w >> 15) & 31) as usize;
            let imm_i = (w as i32 >> 20) as i64 as u64;
            let pc = base + 4 * k as u64;
            let v = match w & 0x707f {
                _ if w & 0x7f == opc::LUI => (w & 0xfffff000) as i32 as i64 as u64,
                _ if w & 0x7f == opc::AUIPC => {
                    pc.wrapping_add((w & 0xfffff000) as i32 as i64 as u64)
                }
                0x13 => x[rs1].wrapping_add(imm_i),
                0x1b => (x[rs1].wrapping_add(imm_i) as i32) as i64 as u64,
                0x1013 => x[rs1] << (imm_i & 63),
                0x5013 if w >> 30 == 1 => ((x[rs1] as i64) >> (imm_i & 63)) as u64,
                0x5013 => x[rs1] >> (imm_i & 63),
                0x3003 => {
                    let at = (x[rs1].wrapping_add(imm_i) - base) as usize;
                    u64::from_le_bytes(out.bytes[at..at + 8].try_into().unwrap())
                }
                _ => panic!("unexpected word {w:#x}"),
            };
            x[rd] = v;
        }
        (x[10], n)
    }

    #[test]
    fn movi_forms() {
        let base = 0x7f00_0000_1000;
        let cases: [(i64, usize); 10] = [
            (0, 1),
            (-2048, 1),
            (0x7ff, 1),
            (0x12345678, 2),
            (-0x8000_0000, 1),
            (0x1234_5000_0000_0000, 2),
            (0x0000_ffff_ffff_f000, 2),
            (base as i64 + 0x1234, 2),
            (0x1234_5678_9abc_def0, 2),
            (0xff, 1),
        ];
        for (v, n) in cases {
            let (got, len) = eval_movi(base, v, Type::I64);
            assert_eq!(got, v as u64, "{v:#x}");
            assert_eq!(len, n, "{v:#x}");
        }
        assert_eq!(eval_movi(base, 0xffff_ffff, Type::I32).0, u64::MAX);
        assert_eq!(eval_movi(base, 0x8000_0000, Type::I32).0, 0xffff_ffff_8000_0000);
    }

    #[test]
    fn far_branches_and_labels() {
        let mut a = Asm::new(0, HostFeatures::BASELINE);
        let l = a.new_label();
        a.far_b_label(opc::BEQ, 10, 11, l);
        a.emit(opc::NOP);
        a.bind(l);
        a.b_label(opc::BLTU, 9, 31, l);
        let out = a.finish().unwrap();
        let w: Vec<u32> =
            out.bytes.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(w[0], encode_sb(opc::BNE, 10, 11, 8));
        assert_eq!(w[1], encode_uj(opc::JAL, ZERO, 8));
        assert_eq!(w[3], encode_sb(opc::BLTU, 9, 31, 0));
    }

    #[test]
    fn pool_dedups_shared_entries_only() {
        let mut a = Asm::new(0x1000, HostFeatures::BASELINE);
        a.pool_load(10, 0x1234_5678_9abc_def0);
        a.pool_load(11, 0x1234_5678_9abc_def0);
        let h = a.pool_unique(12, 7);
        a.set_unique(h, 0x55);
        let out = a.finish().unwrap();
        // Six words of code, then two entries.
        assert_eq!(out.end, 24 + 16);
        assert_eq!(out.uniques, vec![32]);
        assert_eq!(u64::from_le_bytes(out.bytes[32..40].try_into().unwrap()), 0x55);
    }
}
