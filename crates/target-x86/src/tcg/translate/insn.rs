// SPDX-License-Identifier: GPL-2.0-or-later

//! The opcode decoder and the instruction emitters: the integer and system part of QEMU's
//! `disas_insn()`, with the code each instruction generates taken from `emit.c.inc`.

use ruvm_jit_core::ir::TempI128;
use ruvm_jit_core::tcg_op_ldst::AtomicOp;
use ruvm_jit_core::types::{bswap, mo as mb};

use super::super::cc::{CC_OP_ADCB, CC_OP_SBBB};
use super::super::env::{
    AC_MASK, CC_A, CR, DF, GDT, HF_UMIP_MASK, ID_MASK, IDT, IF_MASK, IOPL_MASK, KERNELGSBASE, LDT,
    NT_MASK, SEG_LIMIT, TR, cr,
};
use super::super::{EXCP00_DIVZ, EXCP03_INT3, EXCP07_PREX};
use super::*;
use crate::state::{
    HF_EM_MASK, HF_MP_MASK, HF_TS_MASK, R_CS, R_DS, R_EAX, R_EBP, R_EBX, R_EDI, R_EDX, R_ES, R_ESI,
    R_FS, R_GS,
};

const OP_ADD: u32 = 0;
const OP_OR: u32 = 1;
const OP_ADC: u32 = 2;
const OP_SBB: u32 = 3;
const OP_AND: u32 = 4;
const OP_SUB: u32 = 5;
const OP_XOR: u32 = 6;
const OP_CMP: u32 = 7;

/// `mo_b_d()`: byte for even opcodes, the operand size for odd ones.
fn mo_b_d(b: u32, ot: u32) -> u32 {
    if b & 1 != 0 { ot } else { OT8 }
}

/// `mo_b_d32()`: like [`mo_b_d`] but at most 32 bits.
fn mo_b_d32(b: u32, ot: u32) -> u32 {
    if b & 1 == 0 {
        OT8
    } else if ot == OT16 {
        OT16
    } else {
        OT32
    }
}

/// The opcodes that accept a LOCK prefix (with a memory destination).
fn lockable(b: u32) -> bool {
    match b {
        0x00..=0x3f => b & 7 < 2 && (b >> 3) & 7 != 7,
        0x80..=0x83 | 0x86 | 0x87 | 0xf6 | 0xf7 | 0xfe | 0xff => true,
        0x1ab | 0x1b3 | 0x1bb | 0x1ba | 0x1b0 | 0x1b1 | 0x1c0 | 0x1c1 | 0x1c7 => true,
        _ => false,
    }
}

/// A shift count: an immediate or CL.
#[derive(Clone, Copy)]
enum Count {
    Imm(u64),
    Cl,
}

/// A string instruction body, named by its even opcode.
type StrOp = u32;

impl S<'_, '_, '_> {
    fn lock(&self) -> bool {
        self.d.prefix & PREFIX_LOCK != 0
    }

    /// Raise #UD for a LOCK prefix where `bad` says it is not allowed.
    fn bad_lock(&mut self, bad: bool) -> bool {
        if self.lock() && bad {
            self.gen_illegal_opcode();
            true
        } else {
            false
        }
    }

    /// Raise #UD for an instruction that does not exist in 64-bit mode.
    fn inv64(&mut self) -> bool {
        if self.code64() {
            self.gen_illegal_opcode();
            true
        } else {
            false
        }
    }

    pub(super) fn modrm(&mut self) -> R<(u32, u32, usize, usize)> {
        let m = self.ldub()? as u32;
        let (md, reg, rm) = self.split_modrm(m);
        Ok((m, md, reg, rm))
    }

    fn split_modrm(&self, m: u32) -> (u32, usize, usize) {
        (m >> 6, ((m >> 3) & 7) as usize | self.d.rex_r, (m & 7) as usize | self.d.rex_b)
    }

    /// Near branches use a 64-bit operand in 64-bit code; AMD honours a 66 prefix.
    fn near_branch_ot(&mut self) {
        if self.code64() {
            self.d.dflag =
                if self.d.prefix & PREFIX_DATA != 0 && !self.d.feat.intel { OT16 } else { OT64 };
        }
    }

    /// Decode and translate one instruction.
    pub(super) fn disas_insn(&mut self) -> R {
        self.d.prefix = 0;
        self.d.override_seg = -1;
        self.d.rex_r = 0;
        self.d.rex_x = 0;
        self.d.rex_b = 0;
        self.d.rex_w = false;
        self.d.vex_l = false;
        self.d.vex_v = 0;
        self.d.vex_w = false;
        self.d.rip_offset = 0;
        self.d.popl_esp_hack = 0;
        let mut b;
        loop {
            b = self.ldub()? as u32;
            if self.code64() && (0x40..=0x4f).contains(&b) {
                self.d.prefix |= PREFIX_REX;
                self.d.rex_w = b & 8 != 0;
                self.d.rex_r = ((b & 4) << 1) as usize;
                self.d.rex_x = ((b & 2) << 2) as usize;
                self.d.rex_b = ((b & 1) << 3) as usize;
                continue;
            }
            match b {
                0xf3 => self.d.prefix = (self.d.prefix | PREFIX_REPZ) & !PREFIX_REPNZ,
                0xf2 => self.d.prefix = (self.d.prefix | PREFIX_REPNZ) & !PREFIX_REPZ,
                0xf0 => self.d.prefix |= PREFIX_LOCK,
                0x26 => self.d.override_seg = R_ES as i32,
                0x2e => self.d.override_seg = R_CS as i32,
                0x36 => self.d.override_seg = R_SS as i32,
                0x3e => self.d.override_seg = R_DS as i32,
                0x64 => self.d.override_seg = R_FS as i32,
                0x65 => self.d.override_seg = R_GS as i32,
                0x66 => self.d.prefix |= PREFIX_DATA,
                0x67 => self.d.prefix |= PREFIX_ADR,
                _ => break,
            }
            // A REX prefix only counts right before the opcode.
            self.d.prefix &= !PREFIX_REX;
            self.d.rex_r = 0;
            self.d.rex_x = 0;
            self.d.rex_b = 0;
            self.d.rex_w = false;
        }
        let mut vex_map = 0;
        if b == 0xc4 || b == 0xc5 {
            match self.vex_prefix(b)? {
                ext::Vex::No => {}
                ext::Vex::Map(m) => vex_map = m,
                ext::Vex::Done => return Ok(()),
            }
        }
        if self.code64() {
            // ES, CS, SS and DS overrides are ignored in 64-bit mode.
            if self.d.override_seg >= 0 && self.d.override_seg < R_FS as i32 {
                self.d.override_seg = -1;
            }
            self.d.dflag = if self.d.rex_w {
                OT64
            } else if self.d.prefix & PREFIX_DATA != 0 {
                OT16
            } else {
                OT32
            };
            self.d.aflag = if self.d.prefix & PREFIX_ADR != 0 { OT32 } else { OT64 };
        } else {
            let data = self.d.prefix & PREFIX_DATA != 0;
            let adr = self.d.prefix & PREFIX_ADR != 0;
            self.d.dflag = if self.d.code32 ^ data { OT32 } else { OT16 };
            self.d.aflag = if self.d.code32 ^ adr { OT32 } else { OT16 };
        }
        if vex_map != 0 {
            return self.vex_insn(vex_map);
        }
        if b == 0x0f {
            b = 0x100 | self.ldub()? as u32;
        }
        if self.lock() && !lockable(b) {
            self.gen_illegal_opcode();
            return Ok(());
        }
        self.dispatch(b)
    }

    fn dispatch(&mut self, b: u32) -> R {
        let g = self.g;
        match b {
            0x00..=0x3f if b & 7 < 6 => self.alu(b)?,
            0x06 | 0x0e | 0x16 | 0x1e | 0x1a0 | 0x1a8 => {
                if b < 0x100 && self.inv64() {
                    return Ok(());
                }
                let s = ((b >> 3) & 7) as usize;
                let t = self.new64();
                let env = g.env;
                self.f().gen_ld32u_i64(t, env, (seg(s) + SEG_SELECTOR) as i64);
                self.gen_push_v(t);
            }
            0x07 | 0x17 | 0x1f | 0x1a1 | 0x1a9 => {
                if b < 0x100 && self.inv64() {
                    return Ok(());
                }
                let s = ((b >> 3) & 7) as usize;
                let ot = self.gen_pop_t0();
                self.gen_movl_seg(s, g.t0);
                self.gen_pop_update(ot);
            }
            0x27 | 0x2f | 0x37 | 0x3f => {
                if self.inv64() {
                    return Ok(());
                }
                let h = match b {
                    0x27 => &helpers::DAA,
                    0x2f => &helpers::DAS,
                    0x37 => &helpers::AAA,
                    _ => &helpers::AAS,
                };
                self.env_call(h, None, &[]);
                self.assume_cc_op(CC_OP_EFLAGS);
            }
            0x40..=0x4f => {
                let ot = self.d.dflag;
                let r = (b & 7) as usize;
                self.gen_inc(ot, Some(r), if b < 0x48 { 1 } else { -1 });
            }
            0x50..=0x57 => {
                let r = (b & 7) as usize | self.d.rex_b;
                let t = self.new64();
                self.f().gen_mov_i64(t, g.regs[r]);
                self.gen_push_v(t);
            }
            0x58..=0x5f => {
                let r = (b & 7) as usize | self.d.rex_b;
                let ot = self.gen_pop_t0();
                // The order matters for POP %SP.
                self.gen_pop_update(ot);
                self.mov_reg_v(ot, r, g.t0);
            }
            0x60 | 0x61 => {
                if self.inv64() {
                    return Ok(());
                }
                if b == 0x60 { self.gen_pusha() } else { self.gen_popa() }
            }
            0x62 => self.bound()?,
            0x63 => self.movsxd_arpl()?,
            0x68 | 0x6a => {
                let ot = self.mo_pushpop(self.d.dflag);
                let v = if b == 0x68 {
                    self.insn_get_signed(ot)?
                } else {
                    self.ldub()? as u8 as i8 as i64
                };
                let t = self.c64(v);
                self.gen_push_v(t);
            }
            0x69 | 0x6b | 0x1af => self.imul_rm(b)?,
            0x6c..=0x6f => self.ins_outs(b),
            0x70..=0x7f | 0x180..=0x18f => {
                self.near_branch_ot();
                let diff = if b < 0x100 {
                    self.ldub()? as u8 as i8 as i64
                } else if self.d.dflag == OT16 {
                    self.lduw()? as u16 as i16 as i64
                } else {
                    self.ldl()? as u32 as i32 as i64
                };
                let l1 = self.label();
                self.gen_jcc1(b & 0xf, l1);
                self.gen_jmp_rel_csize(0, 1);
                self.set_label(l1);
                let ot = self.d.dflag;
                self.gen_jmp_rel(ot, diff, 0);
            }
            0x80..=0x83 => self.grp1(b)?,
            0x84 | 0x85 | 0xa8 | 0xa9 => {
                let ot = mo_b_d(b, self.d.dflag);
                if b < 0xa8 {
                    let (m, _, reg, _) = self.modrm()?;
                    self.gen_ldst_modrm(m, ot, None, false)?;
                    self.mov_v_reg(ot, g.t1, reg);
                } else {
                    self.mov_v_reg(ot, g.t0, R_EAX);
                    let v = self.insn_get_signed(ot)?;
                    self.f().gen_movi_i64(g.t1, v);
                }
                let f = self.f();
                f.gen_and_i64(g.cc_dst, g.t0, g.t1);
                self.set_cc_op(CC_OP_LOGICB + ot);
            }
            0x86 | 0x87 => {
                let ot = mo_b_d(b, self.d.dflag);
                let (m, md, reg, rm) = self.modrm()?;
                if self.bad_lock(md == 3) {
                    return Ok(());
                }
                if md == 3 {
                    self.mov_v_reg(ot, g.t0, reg);
                    self.mov_v_reg(ot, g.t1, rm);
                    self.mov_reg_v(ot, rm, g.t0);
                    self.mov_reg_v(ot, reg, g.t1);
                } else {
                    self.gen_lea_modrm(m)?;
                    self.mov_v_reg(ot, g.t0, reg);
                    // XCHG with memory is always atomic.
                    let idx = self.d.mem_index;
                    self.f().gen_atomic_op_i64(AtomicOp::Xchg, g.t1, g.a0, g.t0, idx, mo(ot));
                    self.mov_reg_v(ot, reg, g.t1);
                }
            }
            0x88 | 0x89 => {
                let ot = mo_b_d(b, self.d.dflag);
                let (m, _, reg, _) = self.modrm()?;
                self.gen_ldst_modrm(m, ot, Some(reg), true)?;
            }
            0x8a | 0x8b => {
                let ot = mo_b_d(b, self.d.dflag);
                let (m, _, reg, _) = self.modrm()?;
                self.gen_ldst_modrm(m, ot, None, false)?;
                self.mov_reg_v(ot, reg, g.t0);
            }
            0x8c => {
                let (m, md, _, _) = self.modrm()?;
                let s = ((m >> 3) & 7) as usize;
                if s > 5 {
                    self.gen_illegal_opcode();
                    return Ok(());
                }
                let env = g.env;
                self.f().gen_ld32u_i64(g.t0, env, (seg(s) + SEG_SELECTOR) as i64);
                let ot = if md == 3 { self.d.dflag } else { OT16 };
                self.gen_ldst_modrm(m, ot, None, true)?;
            }
            0x8d => {
                let (m, md, reg, _) = self.modrm()?;
                if md == 3 {
                    self.gen_illegal_opcode();
                    return Ok(());
                }
                let a = self.lea_modrm_0(m)?;
                let ea = self.lea_modrm_1(a);
                let aflag = self.d.aflag;
                self.lea_v_seg(aflag, ea, -1, -1);
                let ot = self.d.dflag;
                self.mov_reg_v(ot, reg, g.a0);
            }
            0x8e => {
                let (m, _, _, _) = self.modrm()?;
                let s = ((m >> 3) & 7) as usize;
                if s >= 6 || s == R_CS {
                    self.gen_illegal_opcode();
                    return Ok(());
                }
                self.gen_ldst_modrm(m, OT16, None, false)?;
                self.gen_movl_seg(s, g.t0);
            }
            0x8f => {
                let (m, md, _, rm) = self.modrm()?;
                if (m >> 3) & 7 != 0 {
                    // XOP prefix, or an undefined encoding.
                    self.gen_illegal_opcode();
                    return Ok(());
                }
                let ot = self.gen_pop_t0();
                if md == 3 {
                    // The order matters for POP %SP.
                    self.gen_pop_update(ot);
                    self.mov_reg_v(ot, rm, g.t0);
                } else {
                    // The order matters for MMU exceptions too.
                    self.d.popl_esp_hack = 1 << ot;
                    self.gen_ldst_modrm(m, ot, None, true)?;
                    self.d.popl_esp_hack = 0;
                    self.gen_pop_update(ot);
                }
            }
            0x90..=0x97 => {
                let r = (b & 7) as usize | self.d.rex_b;
                if r == 0 {
                    // NOP, or PAUSE with F3.
                    if self.d.prefix & PREFIX_REPZ != 0 {
                        self.b.is_jmp = DISAS_EOB_NEXT;
                    }
                } else {
                    let ot = self.d.dflag;
                    self.mov_v_reg(ot, g.t0, r);
                    self.mov_v_reg(ot, g.t1, R_EAX);
                    self.mov_reg_v(ot, R_EAX, g.t0);
                    self.mov_reg_v(ot, r, g.t1);
                }
            }
            0x98 => {
                let rax = g.regs[R_EAX];
                match self.d.dflag {
                    OT64 => self.f().gen_ext32s_i64(rax, rax),
                    OT32 => {
                        self.f().gen_ext16s_i64(g.t0, rax);
                        self.mov_reg_v(OT32, R_EAX, g.t0);
                    }
                    _ => {
                        self.f().gen_ext8s_i64(g.t0, rax);
                        self.mov_reg_v(OT16, R_EAX, g.t0);
                    }
                }
            }
            0x99 => {
                let (rax, rdx) = (g.regs[R_EAX], g.regs[R_EDX]);
                match self.d.dflag {
                    OT64 => self.f().gen_sari_i64(rdx, rax, 63),
                    OT32 => {
                        let f = self.f();
                        f.gen_ext32s_i64(g.t0, rax);
                        f.gen_sari_i64(g.t0, g.t0, 31);
                        self.mov_reg_v(OT32, R_EDX, g.t0);
                    }
                    _ => {
                        let f = self.f();
                        f.gen_ext16s_i64(g.t0, rax);
                        f.gen_sari_i64(g.t0, g.t0, 15);
                        self.mov_reg_v(OT16, R_EDX, g.t0);
                    }
                }
            }
            0x9a | 0xea => {
                if self.inv64() {
                    return Ok(());
                }
                let ot = self.d.dflag;
                let off = self.insn_get(ot)?;
                let sel = self.lduw()?;
                let f = self.f();
                f.gen_movi_i64(g.t0, sel as i64);
                f.gen_movi_i64(g.t1, off as i64);
                if b == 0x9a { self.do_lcall() } else { self.do_ljmp() }
            }
            0x9b => {
                if self.d.flags & (HF_MP_MASK | HF_TS_MASK) == HF_MP_MASK | HF_TS_MASK {
                    self.gen_exception(EXCP07_PREX);
                }
            }
            0x9c => {
                if self.check_vm86_iopl() {
                    self.env_call(&helpers::READ_EFLAGS, Some(g.t0.into()), &[]);
                    self.gen_push_v(g.t0);
                }
            }
            0x9d => {
                if self.check_vm86_iopl() {
                    let mut mask = TF_MASK | AC_MASK | ID_MASK | NT_MASK;
                    if self.d.cpl == 0 {
                        mask |= IF_MASK | IOPL_MASK;
                    } else if self.d.cpl <= self.d.iopl {
                        mask |= IF_MASK;
                    }
                    if self.d.dflag == OT16 {
                        mask &= 0xffff;
                    }
                    let ot = self.gen_pop_t0();
                    let m = self.c32(mask as i32);
                    self.env_call(&helpers::WRITE_EFLAGS, None, &[g.t0.into(), m.into()]);
                    self.gen_pop_update(ot);
                    self.assume_cc_op(CC_OP_EFLAGS);
                    // TF and AC may have changed.
                    self.b.is_jmp = DISAS_EOB_NEXT;
                }
            }
            0x9e | 0x9f => {
                if self.code64() && !self.d.feat.lahf_lm {
                    self.gen_illegal_opcode();
                    return Ok(());
                }
                self.gen_compute_eflags();
                let rax = g.regs[R_EAX];
                let f = self.f();
                if b == 0x9e {
                    f.gen_extract_i64(g.t0, rax, 8, 8);
                    f.gen_andi_i64(g.cc_src, g.cc_src, i64::from(CC_O));
                    f.gen_andi_i64(g.t0, g.t0, i64::from(CC_S | CC_Z | CC_A | CC_P | CC_C));
                    f.gen_or_i64(g.cc_src, g.cc_src, g.t0);
                } else {
                    f.gen_ori_i64(g.t0, g.cc_src, 0x02);
                    f.gen_deposit_i64(rax, rax, g.t0, 8, 8);
                }
            }
            0xa0..=0xa3 => {
                let ot = mo_b_d(b, self.d.dflag);
                let addr = match self.d.aflag {
                    OT64 => self.ldq()?,
                    OT32 => self.ldl()?,
                    _ => self.lduw()?,
                };
                self.f().gen_movi_i64(g.a0, addr as i64);
                let (aflag, ovr) = (self.d.aflag, self.d.override_seg);
                self.lea_v_seg(aflag, g.a0, R_DS as i32, ovr);
                if b & 2 == 0 {
                    self.ld_v(ot, g.t0, g.a0);
                    self.mov_reg_v(ot, R_EAX, g.t0);
                } else {
                    self.mov_v_reg(ot, g.t0, R_EAX);
                    self.st_v(ot, g.t0, g.a0);
                }
            }
            0xa4..=0xa7 | 0xaa..=0xaf => self.string(b),
            0xb0..=0xb7 => {
                let v = self.ldub()?;
                let r = (b & 7) as usize | self.d.rex_b;
                self.f().gen_movi_i64(g.t0, v as i64);
                self.mov_reg_v(OT8, r, g.t0);
            }
            0xb8..=0xbf => {
                let ot = self.d.dflag;
                let v = if ot == OT64 { self.ldq()? } else { self.insn_get(ot)? };
                let r = (b & 7) as usize | self.d.rex_b;
                self.f().gen_movi_i64(g.t0, v as i64);
                self.mov_reg_v(ot, r, g.t0);
            }
            0xc0 | 0xc1 | 0xd0..=0xd3 => {
                let ot = mo_b_d(b, self.d.dflag);
                let (m, md, _, rm) = self.modrm()?;
                let op = (m >> 3) & 7;
                let d = if md != 3 {
                    if b < 0xd0 {
                        self.d.rip_offset = 1;
                    }
                    self.gen_lea_modrm(m)?;
                    None
                } else {
                    Some(rm)
                };
                let count = match b {
                    0xc0 | 0xc1 => Count::Imm(self.ldub()?),
                    0xd0 | 0xd1 => Count::Imm(1),
                    _ => Count::Cl,
                };
                self.gen_shift(ot, op, d, count);
            }
            0xc2 | 0xc3 => {
                self.near_branch_ot();
                let val = if b == 0xc2 { self.lduw()? as u16 as i16 as i64 } else { 0 };
                let ot = self.gen_pop_t0();
                self.gen_stack_update(val + (1 << ot));
                // gen_pop_t0() zero extends.
                self.gen_op_jmp_v(g.t0);
                self.b.is_jmp = DISAS_JUMP;
            }
            0xc4 | 0xc5 => {
                if self.inv64() {
                    return Ok(());
                }
                // A VEX prefix has been handled by disas_insn().
                let m = self.ldub()? as u32;
                let s = if b == 0xc4 { R_ES } else { R_DS };
                self.gen_lxs(s, m)?;
            }
            0xc6 | 0xc7 => {
                let ot = mo_b_d(b, self.d.dflag);
                let (m, md, _, rm) = self.modrm()?;
                if (m >> 3) & 7 != 0 {
                    self.gen_illegal_opcode();
                    return Ok(());
                }
                if md != 3 {
                    self.d.rip_offset = Self::insn_const_size(ot);
                    self.gen_lea_modrm(m)?;
                }
                let v = self.insn_get_signed(ot)?;
                self.f().gen_movi_i64(g.t0, v);
                if md != 3 {
                    self.st_v(ot, g.t0, g.a0);
                } else {
                    self.mov_reg_v(ot, rm, g.t0);
                }
            }
            0xc8 => {
                let addend = self.lduw()? as i64;
                let level = self.ldub()? as i64;
                self.gen_enter(addend, level);
            }
            0xc9 => {
                let d_ot = self.mo_pushpop(self.d.dflag);
                let a_ot = self.mo_stacksize();
                self.lea_v_seg(a_ot, g.regs[R_EBP], R_SS as i32, -1);
                self.ld_v(d_ot, g.t0, g.a0);
                self.f().gen_addi_i64(g.t1, g.regs[R_EBP], 1 << d_ot);
                self.mov_reg_v(d_ot, R_EBP, g.t0);
                self.mov_reg_v(a_ot, R_ESP, g.t1);
            }
            0xca | 0xcb => {
                let val = if b == 0xca { self.lduw()? as u16 as i16 as i64 } else { 0 };
                self.gen_lret(val);
            }
            0xcc => self.gen_interrupt(EXCP03_INT3),
            0xcd => {
                let v = self.ldub()?;
                if self.check_vm86_iopl() {
                    self.gen_interrupt(v as i32);
                }
            }
            0xce => {
                if self.inv64() {
                    return Ok(());
                }
                self.gen_update_cc_op();
                self.gen_update_eip_cur();
                let len = self.cur_insn_len() as i32;
                let l = self.c32(len);
                self.env_call(&helpers::INTO, None, &[l.into()]);
            }
            0xcf => {
                let shift = self.c32(self.d.dflag as i32 - 1);
                if !self.d.pe || self.d.vm86 {
                    if !self.check_vm86_iopl() {
                        return Ok(());
                    }
                    self.env_call(&helpers::IRET_REAL, None, &[shift.into()]);
                } else {
                    self.env_call(&helpers::IRET_PROTECTED, None, &[shift.into()]);
                }
                self.assume_cc_op(CC_OP_EFLAGS);
                self.b.is_jmp = DISAS_EOB_ONLY;
            }
            0xd4 | 0xd5 => {
                if self.inv64() {
                    return Ok(());
                }
                let base = self.ldub()?;
                if b == 0xd4 && base == 0 {
                    self.gen_exception(EXCP00_DIVZ);
                    return Ok(());
                }
                let bt = self.c64(base as i64);
                let h = if b == 0xd4 { &helpers::AAM } else { &helpers::AAD };
                self.call(h, Some(g.t0.into()), &[g.regs[R_EAX].into(), bt.into()]);
                self.mov_reg_v(OT16, R_EAX, g.t0);
                self.f().gen_ext8u_i64(g.cc_dst, g.t0);
                self.set_cc_op(CC_OP_LOGICB);
            }
            0xd6 => {
                if self.inv64() {
                    return Ok(());
                }
                self.gen_compute_eflags_c(g.t0);
                self.f().gen_neg_i64(g.t0, g.t0);
                self.mov_reg_v(OT8, R_EAX, g.t0);
            }
            0xd7 => {
                let f = self.f();
                f.gen_ext8u_i64(g.t0, g.regs[R_EAX]);
                f.gen_add_i64(g.a0, g.regs[R_EBX], g.t0);
                let (aflag, ovr) = (self.d.aflag, self.d.override_seg);
                self.lea_v_seg(aflag, g.a0, R_DS as i32, ovr);
                self.ld_v(OT8, g.t0, g.a0);
                self.mov_reg_v(OT8, R_EAX, g.t0);
            }
            0xd8..=0xdf => {
                // x87 is not implemented.
                if self.d.flags & (HF_EM_MASK | HF_TS_MASK) != 0 {
                    self.gen_exception(EXCP07_PREX);
                } else {
                    self.gen_illegal_opcode();
                }
            }
            0xe0..=0xe3 => {
                self.near_branch_ot();
                let diff = self.ldub()? as u8 as i8 as i64;
                let taken = self.label();
                let not_taken = self.label();
                self.gen_update_cc_op();
                let aflag = self.d.aflag;
                match b {
                    0xe0 | 0xe1 => {
                        self.add_reg_im(aflag, R_ECX, -1);
                        self.jcc_ecx(Cond::Eq, not_taken);
                        self.gen_jcc1((JCC_Z << 1) | u32::from(b == 0xe0), taken);
                    }
                    0xe2 => {
                        self.add_reg_im(aflag, R_ECX, -1);
                        self.jcc_ecx(Cond::Ne, taken);
                    }
                    _ => self.jcc_ecx(Cond::Eq, taken),
                }
                self.set_label(not_taken);
                self.gen_jmp_rel_csize(0, 1);
                self.set_label(taken);
                let ot = self.d.dflag;
                self.gen_jmp_rel(ot, diff, 0);
            }
            0xe4..=0xe7 | 0xec..=0xef => self.in_out(b)?,
            0xe8 => {
                self.near_branch_ot();
                let diff = if self.d.dflag == OT16 {
                    self.lduw()? as u16 as i16 as i64
                } else {
                    self.ldl()? as u32 as i32 as i64
                };
                let next = self.eip_next();
                let t = self.c64(next as i64);
                self.gen_push_v(t);
                let ot = self.d.dflag;
                self.gen_jmp_rel(ot, diff, 0);
            }
            0xe9 | 0xeb => {
                self.near_branch_ot();
                let diff = if b == 0xeb {
                    self.ldub()? as u8 as i8 as i64
                } else if self.d.dflag == OT16 {
                    self.lduw()? as u16 as i16 as i64
                } else {
                    self.ldl()? as u32 as i32 as i64
                };
                let ot = self.d.dflag;
                self.gen_jmp_rel(ot, diff, 0);
            }
            0xf1 => {
                self.gen_update_cc_op();
                self.gen_update_eip_next();
                self.env_call(&helpers::ICEBP, None, &[]);
                self.b.is_jmp = DisasJumpType::NoReturn;
            }
            0xf4 => {
                if self.check_cpl0() {
                    self.gen_update_cc_op();
                    self.gen_update_eip_next();
                    self.env_call(&helpers::HLT, None, &[]);
                    self.b.is_jmp = DisasJumpType::NoReturn;
                }
            }
            0xf5 | 0xf8 | 0xf9 => {
                self.gen_compute_eflags();
                let f = self.f();
                let c = i64::from(CC_C);
                match b {
                    0xf5 => f.gen_xori_i64(g.cc_src, g.cc_src, c),
                    0xf8 => f.gen_andi_i64(g.cc_src, g.cc_src, !c),
                    _ => f.gen_ori_i64(g.cc_src, g.cc_src, c),
                }
            }
            0xf6 | 0xf7 => self.grp3(b)?,
            0xfa => {
                if self.check_iopl() {
                    self.gen_reset_eflags(IF_MASK);
                }
            }
            0xfb => {
                if self.check_iopl() {
                    self.gen_set_eflags(IF_MASK);
                    // Interrupts are enabled only after the next instruction.
                    self.b.is_jmp = DISAS_EOB_INHIBIT_IRQ;
                }
            }
            0xfc | 0xfd => {
                let t = self.c64(if b == 0xfc { 1 } else { -1 });
                self.st_env32(t, DF);
            }
            0xfe | 0xff => self.grp45(b)?,
            0x100 => self.grp6()?,
            0x101 => self.grp7()?,
            0x102 | 0x103 => self.lar_lsl(b)?,
            0x105 => {
                self.gen_update_cc_op();
                self.gen_update_eip_cur();
                let len = self.cur_insn_len() as i32;
                let l = self.c32(len);
                self.env_call(&helpers::SYSCALL, None, &[l.into()]);
                if self.d.lma {
                    self.assume_cc_op(CC_OP_EFLAGS);
                }
                // TF is checked after SYSCALL completes.
                self.b.is_jmp = DISAS_EOB_RECHECK_TF;
            }
            0x106 => {
                if self.check_cpl0() {
                    self.env_call(&helpers::CLTS, None, &[]);
                    // The static CPU state changed.
                    self.b.is_jmp = DISAS_EOB_NEXT;
                }
            }
            0x107 => {
                if !self.d.pe || self.d.cpl != 0 {
                    self.gen_exception_gpf();
                    return Ok(());
                }
                let t = self.c32(self.d.dflag as i32 - 1);
                self.env_call(&helpers::SYSRET, None, &[t.into()]);
                // The condition codes change only in long mode.
                if self.d.lma {
                    self.assume_cc_op(CC_OP_EFLAGS);
                }
                self.b.is_jmp = DISAS_EOB_RECHECK_TF;
            }
            0x108 | 0x109 => {
                // INVD and WBINVD: nothing is cached.
                self.check_cpl0();
            }
            0x10d | 0x118..=0x11f => {
                // Prefetches and hint NOPs: decode the operand and do nothing.
                let (m, _, _, _) = self.modrm()?;
                self.lea_modrm_0(m)?;
            }
            0x120..=0x123 => self.mov_cr_dr(b)?,
            0x130 | 0x132 => {
                if self.check_cpl0() {
                    if b == 0x130 {
                        self.env_call(&helpers::WRMSR, None, &[]);
                        self.b.is_jmp = DISAS_EOB_NEXT;
                    } else {
                        self.env_call(&helpers::RDMSR, None, &[]);
                    }
                }
            }
            0x131 => {
                self.gen_update_cc_op();
                self.gen_update_eip_cur();
                self.b.translator_io_start();
                self.env_call(&helpers::RDTSC, None, &[]);
            }
            0x133 => {
                self.gen_update_cc_op();
                self.gen_update_eip_cur();
                self.env_call(&helpers::RDPMC, None, &[]);
            }
            0x134 => {
                if self.code64() && !self.d.feat.intel {
                    self.gen_illegal_opcode();
                } else if !self.d.pe {
                    self.gen_exception_gpf();
                } else {
                    self.env_call(&helpers::SYSENTER, None, &[]);
                    self.b.is_jmp = DISAS_EOB_ONLY;
                }
            }
            0x135 => {
                if self.code64() && !self.d.feat.intel {
                    self.gen_illegal_opcode();
                } else if !self.d.pe || self.d.cpl != 0 {
                    self.gen_exception_gpf();
                } else {
                    let t = self.c32(self.d.dflag as i32 - 1);
                    self.env_call(&helpers::SYSEXIT, None, &[t.into()]);
                    self.b.is_jmp = DISAS_EOB_ONLY;
                }
            }
            0x140..=0x14f => {
                let ot = self.d.dflag;
                let (m, _, reg, _) = self.modrm()?;
                self.gen_ldst_modrm(m, ot, None, false)?;
                let cc = self.prepare_cc(b & 0xf);
                let old = g.regs[reg];
                match cc.cond {
                    Cond::Always => {}
                    Cond::Never => self.f().gen_mov_i64(g.t0, old),
                    c => {
                        let r2 = self.cc_reg2(&cc);
                        self.f().gen_movcond_i64(c, g.t0, cc.reg, r2, g.t0, old);
                    }
                }
                self.mov_reg_v(ot, reg, g.t0);
            }
            0x190..=0x19f => {
                let (m, _, _, _) = self.modrm()?;
                self.gen_setcc(b & 0xf, g.t0);
                self.gen_ldst_modrm(m, OT8, None, true)?;
            }
            0x1a2 => {
                self.gen_update_cc_op();
                self.gen_update_eip_cur();
                self.env_call(&helpers::CPUID, None, &[]);
            }
            0x1a3 | 0x1ab | 0x1b3 | 0x1bb | 0x1ba => self.bt(b)?,
            0x1a4 | 0x1a5 | 0x1ac | 0x1ad => {
                let ot = self.d.dflag;
                let (m, md, reg, rm) = self.modrm()?;
                let d = if md != 3 {
                    if b & 1 == 0 {
                        self.d.rip_offset = 1;
                    }
                    self.gen_lea_modrm(m)?;
                    None
                } else {
                    Some(rm)
                };
                self.mov_v_reg(ot, g.t1, reg);
                let count = if b & 1 == 0 { Count::Imm(self.ldub()?) } else { Count::Cl };
                self.gen_shiftd(ot, d, b & 8 != 0, count);
            }
            0x1ae => self.grp15()?,
            0x1b0 | 0x1b1 => self.cmpxchg(b)?,
            0x1b2 | 0x1b4 | 0x1b5 => {
                let m = self.ldub()? as u32;
                let s = (b & 7) as usize;
                self.gen_lxs(s, m)?;
            }
            0x1b6 | 0x1b7 | 0x1be | 0x1bf => {
                let d_ot = self.d.dflag;
                let s_ot = if b & 1 != 0 { OT16 } else { OT8 };
                let (m, _, reg, _) = self.modrm()?;
                self.gen_ldst_modrm(m, s_ot, None, false)?;
                self.ext(s_ot, g.t0, g.t0, b & 8 != 0);
                self.mov_reg_v(d_ot, reg, g.t0);
            }
            0x1b8 => {
                if self.d.prefix & PREFIX_REPZ == 0 || !self.d.feat.popcnt {
                    self.gen_illegal_opcode();
                    return Ok(());
                }
                let ot = self.d.dflag;
                let (m, _, reg, _) = self.modrm()?;
                self.gen_ldst_modrm(m, ot, None, false)?;
                let f = self.f();
                f.gen_mov_i64(g.cc_dst, g.t0);
                f.gen_ctpop_i64(g.t0, g.t0);
                self.mov_reg_v(ot, reg, g.t0);
                self.set_cc_op(CC_OP_POPCNT);
            }
            0x1bc | 0x1bd => self.bsf_bsr(b)?,
            0x1c0 | 0x1c1 => {
                let ot = mo_b_d(b, self.d.dflag);
                let (m, md, reg, rm) = self.modrm()?;
                if self.bad_lock(md == 3) {
                    return Ok(());
                }
                self.mov_v_reg(ot, g.t0, reg);
                if md == 3 {
                    self.mov_v_reg(ot, g.t1, rm);
                    self.f().gen_add_i64(g.t0, g.t0, g.t1);
                    self.mov_reg_v(ot, reg, g.t1);
                    self.mov_reg_v(ot, rm, g.t0);
                } else {
                    self.gen_lea_modrm(m)?;
                    let idx = self.d.mem_index;
                    if self.lock() {
                        let f = self.f();
                        f.gen_atomic_op_i64(AtomicOp::FetchAdd, g.t1, g.a0, g.t0, idx, mo(ot));
                        f.gen_add_i64(g.t0, g.t0, g.t1);
                    } else {
                        self.ld_v(ot, g.t1, g.a0);
                        self.f().gen_add_i64(g.t0, g.t0, g.t1);
                        self.st_v(ot, g.t0, g.a0);
                    }
                    self.mov_reg_v(ot, reg, g.t1);
                }
                let f = self.f();
                f.gen_mov_i64(g.cc_src, g.t1);
                f.gen_mov_i64(g.cc_dst, g.t0);
                self.set_cc_op(CC_OP_ADDB + ot);
            }
            0x1c7 => self.grp9()?,
            0x1c3 => self.movnti()?,
            0x138 => self.op_0f38()?,
            0x1c8..=0x1cf => {
                let r = (b & 7) as usize | self.d.rex_b;
                let ot = self.d.dflag;
                self.mov_v_reg(ot, g.t0, r);
                if ot == OT64 {
                    self.f().gen_bswap64_i64(g.t0, g.t0);
                } else {
                    self.f().gen_bswap32_i64(g.t0, g.t0, bswap::OZ);
                }
                self.mov_reg_v(ot, r, g.t0);
            }
            // Everything else: x87, MMX, SSE, AVX, the 0F 3A map, UD0, UD1, UD2
            // and the instructions listed as missing in the module documentation.
            _ => self.gen_illegal_opcode(),
        }
        Ok(())
    }

    // Arithmetic.

    fn alu(&mut self, b: u32) -> R {
        let g = self.g;
        let op = (b >> 3) & 7;
        let ot = mo_b_d(b, self.d.dflag);
        match b & 7 {
            0 | 1 => {
                let (m, md, reg, rm) = self.modrm()?;
                if self.bad_lock(md == 3) {
                    return Ok(());
                }
                if md != 3 {
                    self.gen_lea_modrm(m)?;
                    self.mov_v_reg(ot, g.t1, reg);
                    self.gen_op(ot, op, None);
                } else if op == OP_XOR && rm == reg {
                    // xor reg, reg: the result and the flags are known.
                    self.f().gen_movi_i64(g.t0, 0);
                    self.mov_reg_v(ot, reg, g.t0);
                    self.set_cc_op(CC_OP_CLR);
                } else {
                    self.mov_v_reg(ot, g.t1, reg);
                    self.gen_op(ot, op, Some(rm));
                }
            }
            2 | 3 => {
                let (m, md, reg, rm) = self.modrm()?;
                if md != 3 {
                    self.gen_lea_modrm(m)?;
                    self.ld_v(ot, g.t1, g.a0);
                } else {
                    self.mov_v_reg(ot, g.t1, rm);
                }
                self.gen_op(ot, op, Some(reg));
            }
            _ => {
                let v = self.insn_get_signed(ot)?;
                self.f().gen_movi_i64(g.t1, v);
                self.gen_op(ot, op, Some(R_EAX));
            }
        }
        Ok(())
    }

    fn grp1(&mut self, b: u32) -> R {
        if b == 0x82 && self.inv64() {
            return Ok(());
        }
        let g = self.g;
        let ot = mo_b_d(b, self.d.dflag);
        let (m, md, _, rm) = self.modrm()?;
        let op = (m >> 3) & 7;
        if self.bad_lock(md == 3 || op == OP_CMP) {
            return Ok(());
        }
        let d = if md != 3 {
            self.d.rip_offset = if b == 0x83 { 1 } else { Self::insn_const_size(ot) };
            self.gen_lea_modrm(m)?;
            None
        } else {
            Some(rm)
        };
        let v = if b == 0x83 { self.ldub()? as u8 as i8 as i64 } else { self.insn_get_signed(ot)? };
        self.f().gen_movi_i64(g.t1, v);
        self.gen_op(ot, op, d);
        Ok(())
    }

    /// `gen_op()`: T0 = `d` (a register, or memory at A0) op T1, with the flags.
    fn gen_op(&mut self, ot: u32, op: u32, d: Option<usize>) {
        let g = self.g;
        let lock = d.is_none() && self.lock();
        let idx = self.d.mem_index;
        match d {
            Some(r) => self.mov_v_reg(ot, g.t0, r),
            None if !lock => self.ld_v(ot, g.t0, g.a0),
            None => {}
        }
        match op {
            OP_ADC | OP_SBB => {
                let c = self.new64();
                self.gen_compute_eflags_c(c);
                if lock {
                    let v = self.new64();
                    let f = self.f();
                    f.gen_add_i64(v, g.t1, c);
                    if op == OP_SBB {
                        f.gen_neg_i64(v, v);
                    }
                    f.gen_atomic_op_i64(AtomicOp::AddFetch, g.t0, g.a0, v, idx, mo(ot));
                } else {
                    let f = self.f();
                    if op == OP_ADC {
                        f.gen_add_i64(g.t0, g.t0, g.t1);
                        f.gen_add_i64(g.t0, g.t0, c);
                    } else {
                        f.gen_sub_i64(g.t0, g.t0, g.t1);
                        f.gen_sub_i64(g.t0, g.t0, c);
                    }
                    self.st_rm_t0(ot, d);
                }
                let f = self.f();
                f.gen_mov_i64(g.cc_src2, c);
                f.gen_mov_i64(g.cc_src, g.t1);
                f.gen_mov_i64(g.cc_dst, g.t0);
                let base = if op == OP_ADC { CC_OP_ADCB } else { CC_OP_SBBB };
                self.set_cc_op(base + ot);
            }
            OP_ADD => {
                if lock {
                    self.f().gen_atomic_op_i64(AtomicOp::AddFetch, g.t0, g.a0, g.t1, idx, mo(ot));
                } else {
                    self.f().gen_add_i64(g.t0, g.t0, g.t1);
                    self.st_rm_t0(ot, d);
                }
                let f = self.f();
                f.gen_mov_i64(g.cc_src, g.t1);
                f.gen_mov_i64(g.cc_dst, g.t0);
                self.set_cc_op(CC_OP_ADDB + ot);
            }
            OP_SUB => {
                if lock {
                    let v = self.new64();
                    let f = self.f();
                    f.gen_neg_i64(v, g.t1);
                    f.gen_atomic_op_i64(AtomicOp::FetchAdd, g.cc_srct, g.a0, v, idx, mo(ot));
                    f.gen_sub_i64(g.t0, g.cc_srct, g.t1);
                } else {
                    let f = self.f();
                    f.gen_mov_i64(g.cc_srct, g.t0);
                    f.gen_sub_i64(g.t0, g.t0, g.t1);
                    self.st_rm_t0(ot, d);
                }
                let f = self.f();
                f.gen_mov_i64(g.cc_src, g.t1);
                f.gen_mov_i64(g.cc_dst, g.t0);
                self.set_cc_op(CC_OP_SUBB + ot);
            }
            OP_AND | OP_OR | OP_XOR => {
                if lock {
                    let aop = match op {
                        OP_AND => AtomicOp::AndFetch,
                        OP_OR => AtomicOp::OrFetch,
                        _ => AtomicOp::XorFetch,
                    };
                    self.f().gen_atomic_op_i64(aop, g.t0, g.a0, g.t1, idx, mo(ot));
                } else {
                    let f = self.f();
                    match op {
                        OP_AND => f.gen_and_i64(g.t0, g.t0, g.t1),
                        OP_OR => f.gen_or_i64(g.t0, g.t0, g.t1),
                        _ => f.gen_xor_i64(g.t0, g.t0, g.t1),
                    }
                    self.st_rm_t0(ot, d);
                }
                self.f().gen_mov_i64(g.cc_dst, g.t0);
                self.set_cc_op(CC_OP_LOGICB + ot);
            }
            _ => {
                let f = self.f();
                f.gen_mov_i64(g.cc_src, g.t1);
                f.gen_mov_i64(g.cc_srct, g.t0);
                f.gen_sub_i64(g.cc_dst, g.t0, g.t1);
                self.set_cc_op(CC_OP_SUBB + ot);
            }
        }
    }

    /// `gen_inc()`: INC or DEC of `d` (a register, or memory at A0).
    fn gen_inc(&mut self, ot: u32, d: Option<usize>, c: i64) {
        let g = self.g;
        let lock = d.is_none() && self.lock();
        if lock {
            let v = self.c64(c);
            let idx = self.d.mem_index;
            self.f().gen_atomic_op_i64(AtomicOp::AddFetch, g.t0, g.a0, v, idx, mo(ot));
        } else {
            match d {
                Some(r) => self.mov_v_reg(ot, g.t0, r),
                None => self.ld_v(ot, g.t0, g.a0),
            }
            self.f().gen_addi_i64(g.t0, g.t0, c);
            self.st_rm_t0(ot, d);
        }
        let cf = self.new64();
        self.gen_compute_eflags_c(cf);
        let f = self.f();
        f.gen_mov_i64(g.cc_src, cf);
        f.gen_mov_i64(g.cc_dst, g.t0);
        self.set_cc_op(if c > 0 { CC_OP_INCB } else { CC_OP_DECB } + ot);
    }

    fn grp3(&mut self, b: u32) -> R {
        let g = self.g;
        let ot = mo_b_d(b, self.d.dflag);
        let (m, md, _, rm) = self.modrm()?;
        let op = (m >> 3) & 7;
        if self.bad_lock(md == 3 || !(op == 2 || op == 3)) {
            return Ok(());
        }
        let lock = self.lock();
        let d = if md != 3 {
            if op <= 1 {
                self.d.rip_offset = Self::insn_const_size(ot);
            }
            self.gen_lea_modrm(m)?;
            if !(lock && op == 2) {
                self.ld_v(ot, g.t0, g.a0);
            }
            None
        } else {
            self.mov_v_reg(ot, g.t0, rm);
            Some(rm)
        };
        let idx = self.d.mem_index;
        match op {
            0 | 1 => {
                let v = self.insn_get_signed(ot)?;
                self.f().gen_andi_i64(g.cc_dst, g.t0, v);
                self.set_cc_op(CC_OP_LOGICB + ot);
            }
            2 => {
                if lock {
                    let all = self.c64(-1);
                    self.f().gen_atomic_op_i64(AtomicOp::XorFetch, g.t0, g.a0, all, idx, mo(ot));
                } else {
                    self.f().gen_not_i64(g.t0, g.t0);
                    self.st_rm_t0(ot, d);
                }
            }
            3 => {
                if lock {
                    let l = self.label();
                    let old = self.new64();
                    let cmp = self.new64();
                    let neg = self.new64();
                    self.f().gen_mov_i64(old, g.t0);
                    self.set_label(l);
                    let f = self.f();
                    f.gen_mov_i64(cmp, old);
                    f.gen_neg_i64(neg, old);
                    f.gen_atomic_cmpxchg_i64(old, g.a0, cmp, neg, idx, mo(ot));
                    let x = self.ext_new(ot, cmp, false);
                    let f = self.f();
                    f.gen_brcond_i64(Cond::Ne, old, x, l);
                    f.gen_neg_i64(g.t0, old);
                } else {
                    self.f().gen_neg_i64(g.t0, g.t0);
                    self.st_rm_t0(ot, d);
                }
                let f = self.f();
                f.gen_mov_i64(g.cc_dst, g.t0);
                f.gen_neg_i64(g.cc_src, g.t0);
                f.gen_movi_i64(g.cc_srct, 0);
                self.set_cc_op(CC_OP_SUBB + ot);
            }
            4 => self.gen_mul(ot, false),
            5 => self.gen_mul(ot, true),
            _ => {
                let h = match (ot, op == 7) {
                    (OT8, false) => &helpers::DIVB,
                    (OT8, true) => &helpers::IDIVB,
                    (OT16, false) => &helpers::DIVW,
                    (OT16, true) => &helpers::IDIVW,
                    (OT32, false) => &helpers::DIVL,
                    (OT32, true) => &helpers::IDIVL,
                    (_, false) => &helpers::DIVQ,
                    (_, true) => &helpers::IDIVQ,
                };
                self.env_call(h, None, &[g.t0.into()]);
            }
        }
        Ok(())
    }

    /// MUL and IMUL with the accumulator.
    fn gen_mul(&mut self, ot: u32, signed: bool) {
        let g = self.g;
        let (rax, rdx) = (g.regs[R_EAX], g.regs[R_EDX]);
        match ot {
            OT8 | OT16 => {
                self.ext(ot, g.t0, g.t0, signed);
                self.ext(ot, g.t1, rax, signed);
                self.f().gen_mul_i64(g.t0, g.t0, g.t1);
                self.mov_reg_v(OT16, R_EAX, g.t0);
                self.f().gen_mov_i64(g.cc_dst, g.t0);
                if signed {
                    let t = self.ext_new(ot, g.t0, true);
                    self.f().gen_sub_i64(g.cc_src, g.t0, t);
                } else if ot == OT8 {
                    self.f().gen_andi_i64(g.cc_src, g.t0, 0xff00);
                }
                if ot == OT16 {
                    self.f().gen_shri_i64(g.t1, g.t0, 16);
                    self.mov_reg_v(OT16, R_EDX, g.t1);
                    if !signed {
                        self.f().gen_mov_i64(g.cc_src, g.t1);
                    }
                }
            }
            OT32 => {
                self.ext(OT32, g.t0, g.t0, signed);
                self.ext(OT32, g.t1, rax, signed);
                self.f().gen_mul_i64(g.t0, g.t0, g.t1);
                self.mov_reg_v(OT32, R_EAX, g.t0);
                self.f().gen_shri_i64(g.t1, g.t0, 32);
                self.mov_reg_v(OT32, R_EDX, g.t1);
                self.f().gen_mov_i64(g.cc_dst, rax);
                if signed {
                    let t = self.ext_new(OT32, g.t0, true);
                    self.f().gen_sub_i64(g.cc_src, g.t0, t);
                } else {
                    self.f().gen_mov_i64(g.cc_src, rdx);
                }
            }
            _ => {
                let f = self.f();
                if signed {
                    f.gen_muls2_i64(rax, rdx, g.t0, rax);
                    f.gen_mov_i64(g.cc_dst, rax);
                    f.gen_sari_i64(g.cc_src, rax, 63);
                    f.gen_sub_i64(g.cc_src, g.cc_src, rdx);
                } else {
                    f.gen_mulu2_i64(rax, rdx, g.t0, rax);
                    f.gen_mov_i64(g.cc_dst, rax);
                    f.gen_mov_i64(g.cc_src, rdx);
                }
            }
        }
        self.set_cc_op(CC_OP_MULB + ot);
    }

    /// IMUL Gv, Ev (with an optional immediate).
    fn imul_rm(&mut self, b: u32) -> R {
        let g = self.g;
        let ot = self.d.dflag;
        let (m, _, reg, _) = self.modrm()?;
        if b == 0x69 {
            self.d.rip_offset = Self::insn_const_size(ot);
        } else if b == 0x6b {
            self.d.rip_offset = 1;
        }
        self.gen_ldst_modrm(m, ot, None, false)?;
        match b {
            0x69 => {
                let v = self.insn_get_signed(ot)?;
                self.f().gen_movi_i64(g.t1, v);
            }
            0x6b => {
                let v = self.ldub()? as u8 as i8 as i64;
                self.f().gen_movi_i64(g.t1, v);
            }
            _ => self.mov_v_reg(ot, g.t1, reg),
        }
        if ot == OT64 {
            let r = g.regs[reg];
            let hi = self.new64();
            let f = self.f();
            f.gen_muls2_i64(g.t0, hi, g.t0, g.t1);
            f.gen_mov_i64(r, g.t0);
            f.gen_mov_i64(g.cc_dst, g.t0);
            f.gen_sari_i64(g.cc_src, g.t0, 63);
            f.gen_sub_i64(g.cc_src, g.cc_src, hi);
        } else {
            self.ext(ot, g.t0, g.t0, true);
            self.ext(ot, g.t1, g.t1, true);
            self.f().gen_mul_i64(g.t0, g.t0, g.t1);
            let t = self.ext_new(ot, g.t0, true);
            self.f().gen_sub_i64(g.cc_src, g.t0, t);
            self.ext(ot, g.cc_dst, g.t0, false);
            self.mov_reg_v(ot, reg, g.t0);
        }
        self.set_cc_op(CC_OP_MULB + ot);
        Ok(())
    }

    // Shifts and rotates.

    fn shift_mask(ot: u32) -> i64 {
        if ot == OT64 { 0x3f } else { 0x1f }
    }

    fn count_temp(&mut self, ot: u32, count: Count) -> TempI64 {
        let t = self.new64();
        let mask = Self::shift_mask(ot);
        match count {
            Count::Imm(n) => self.f().gen_movi_i64(t, n as i64 & mask),
            Count::Cl => {
                let ecx = self.g.regs[R_ECX];
                self.f().gen_andi_i64(t, ecx, mask);
            }
        }
        t
    }

    /// Group 2: `op` on `d` (a register, or memory at A0) by `count`.
    fn gen_shift(&mut self, ot: u32, op: u32, d: Option<usize>, count: Count) {
        let g = self.g;
        match d {
            Some(r) => self.mov_v_reg(ot, g.t0, r),
            None => self.ld_v(ot, g.t0, g.a0),
        }
        match op {
            0 | 1 => self.gen_rot(ot, d, count, op == 1),
            2 | 3 => self.gen_rotc(ot, d, count, op == 3),
            _ => {
                let is_right = op == 5 || op == 7;
                let is_arith = op == 7;
                if is_right {
                    self.ext(ot, g.t0, g.t0, is_arith);
                }
                match count {
                    Count::Imm(n) => {
                        let n = n as i64 & Self::shift_mask(ot);
                        if n != 0 {
                            let f = self.f();
                            match (is_right, is_arith) {
                                (true, true) => {
                                    f.gen_sari_i64(g.tmp4, g.t0, n - 1);
                                    f.gen_sari_i64(g.t0, g.t0, n);
                                }
                                (true, false) => {
                                    f.gen_shri_i64(g.tmp4, g.t0, n - 1);
                                    f.gen_shri_i64(g.t0, g.t0, n);
                                }
                                _ => {
                                    f.gen_shli_i64(g.tmp4, g.t0, n - 1);
                                    f.gen_shli_i64(g.t0, g.t0, n);
                                }
                            }
                        }
                        self.st_rm_t0(ot, d);
                        if n != 0 {
                            let f = self.f();
                            f.gen_mov_i64(g.cc_src, g.tmp4);
                            f.gen_mov_i64(g.cc_dst, g.t0);
                            let base = if is_right { CC_OP_SARB } else { CC_OP_SHLB };
                            self.set_cc_op(base + ot);
                        }
                    }
                    Count::Cl => {
                        let c = self.count_temp(ot, count);
                        let f = self.f();
                        f.gen_subi_i64(g.tmp0, c, 1);
                        match (is_right, is_arith) {
                            (true, true) => {
                                f.gen_sar_i64(g.tmp0, g.t0, g.tmp0);
                                f.gen_sar_i64(g.t0, g.t0, c);
                            }
                            (true, false) => {
                                f.gen_shr_i64(g.tmp0, g.t0, g.tmp0);
                                f.gen_shr_i64(g.t0, g.t0, c);
                            }
                            _ => {
                                f.gen_shl_i64(g.tmp0, g.t0, g.tmp0);
                                f.gen_shl_i64(g.t0, g.t0, c);
                            }
                        }
                        self.st_rm_t0(ot, d);
                        self.gen_shift_flags(ot, g.t0, g.tmp0, c, is_right);
                    }
                }
            }
        }
    }

    /// `gen_shift_flags()`: the flags of a shift by a count that may be zero, in which case
    /// they do not change.
    fn gen_shift_flags(
        &mut self,
        ot: u32,
        result: TempI64,
        shm1: TempI64,
        count: TempI64,
        is_right: bool,
    ) {
        let g = self.g;
        // The cc_op global must hold the current value before it is selected below.
        self.gen_update_cc_op();
        let z = self.c64(0);
        let base = if is_right { CC_OP_SARB } else { CC_OP_SHLB };
        let newop = self.c32((base + ot) as i32);
        let z32 = self.c32(0);
        let s32 = self.trunc32(count);
        let f = self.f();
        f.gen_movcond_i64(Cond::Ne, g.cc_dst, count, z, result, g.cc_dst);
        f.gen_movcond_i64(Cond::Ne, g.cc_src, count, z, shm1, g.cc_src);
        f.gen_movcond_i32(Cond::Ne, g.cc_op, s32, z32, newop, g.cc_op);
        self.cc_op_now_dynamic();
    }

    /// All the arithmetic flags into a new temp, without changing the lazy flags state.
    fn eflags_value(&mut self) -> TempI64 {
        let g = self.g;
        let t = self.new64();
        match self.d.cc_op {
            CC_OP_EFLAGS => self.f().gen_mov_i64(t, g.cc_src),
            CC_OP_CLR => self.f().gen_movi_i64(t, i64::from(CC_Z | CC_P)),
            _ => {
                let op = self.cc_op_value();
                self.call(
                    &helpers::CC_COMPUTE_ALL,
                    Some(t.into()),
                    &[g.cc_dst.into(), g.cc_src.into(), g.cc_src2.into(), op.into()],
                );
            }
        }
        t
    }

    /// ROL and ROR.
    fn gen_rot(&mut self, ot: u32, d: Option<usize>, count: Count, is_right: bool) {
        let g = self.g;
        let c = self.count_temp(ot, count);
        let mask = Self::shift_mask(ot);
        if ot == OT64 {
            let f = self.f();
            if is_right { f.gen_rotr_i64(g.t0, g.t0, c) } else { f.gen_rotl_i64(g.t0, g.t0, c) }
        } else {
            // Replicate 8 and 16-bit inputs so that a 32-bit rotate works.
            let f = self.f();
            match ot {
                OT8 => {
                    f.gen_ext8u_i64(g.t0, g.t0);
                    f.gen_muli_i64(g.t0, g.t0, 0x0101_0101);
                }
                OT16 => f.gen_deposit_i64(g.t0, g.t0, g.t0, 16, 16),
                _ => {}
            }
            let a = self.trunc32(g.t0);
            let n = self.trunc32(c);
            let f = self.f();
            if is_right {
                f.gen_rotr_i32(a, a, n)
            } else {
                f.gen_rotl_i32(a, a, n)
            }
            f.gen_extu_i32_i64(g.t0, a);
        }
        self.st_rm_t0(ot, d);
        if let Count::Imm(n) = count {
            if n as i64 & mask == 0 {
                return;
            }
        }
        let fl = self.eflags_value();
        let cf = self.new64();
        let of = self.new64();
        let f = self.f();
        if is_right {
            f.gen_shri_i64(of, g.t0, mask - 1);
            f.gen_shri_i64(cf, g.t0, mask);
            f.gen_andi_i64(cf, cf, 1);
        } else {
            f.gen_shri_i64(of, g.t0, mask);
            f.gen_andi_i64(cf, g.t0, 1);
        }
        f.gen_andi_i64(of, of, 1);
        f.gen_xor_i64(of, of, cf);
        f.gen_mov_i64(g.cc_src, fl);
        f.gen_mov_i64(g.cc_dst, cf);
        f.gen_mov_i64(g.cc_src2, of);
        match count {
            Count::Imm(_) => {
                self.d.cc_op_dirty = true;
                self.d.cc_op = CC_OP_ADCOX;
            }
            Count::Cl => {
                // A zero count keeps the flags: EFLAGS in cc_src.
                let z32 = self.c32(0);
                let s32 = self.trunc32(c);
                let adcox = self.c32(CC_OP_ADCOX as i32);
                let efl = self.c32(CC_OP_EFLAGS as i32);
                self.f().gen_movcond_i32(Cond::Ne, g.cc_op, s32, z32, adcox, efl);
                self.cc_op_now_dynamic();
            }
        }
    }

    /// RCL and RCR, computed inline where QEMU calls `helper_rcl*` and `helper_rcr*`.
    fn gen_rotc(&mut self, ot: u32, d: Option<usize>, count: Count, is_right: bool) {
        let g = self.g;
        let w = 8i64 << ot;
        let n = self.count_temp(ot, count);
        if ot == OT8 || ot == OT16 {
            let m = self.c64(w + 1);
            self.f().gen_remu_i64(n, n, m);
        }
        let fl = self.eflags_value();
        let x = self.ext_new(ot, g.t0, false);
        let cf = self.new64();
        let nm1 = self.new64();
        let wn = self.new64();
        let a = self.new64();
        let bb = self.new64();
        let c = self.new64();
        let newcf = self.new64();
        let f = self.f();
        f.gen_andi_i64(cf, fl, 1);
        f.gen_subi_i64(nm1, n, 1);
        f.gen_subfi_i64(wn, w, n);
        if is_right {
            f.gen_shr_i64(a, x, n);
            f.gen_shl_i64(bb, cf, wn);
            f.gen_shl_i64(c, x, wn);
            f.gen_shli_i64(c, c, 1);
            f.gen_shr_i64(newcf, x, nm1);
        } else {
            f.gen_shl_i64(a, x, n);
            f.gen_shl_i64(bb, cf, nm1);
            f.gen_shr_i64(c, x, wn);
            f.gen_shri_i64(c, c, 1);
            f.gen_shr_i64(newcf, x, wn);
        }
        f.gen_andi_i64(newcf, newcf, 1);
        f.gen_or_i64(a, a, bb);
        f.gen_or_i64(a, a, c);
        self.ext(ot, a, a, false);
        let of = self.new64();
        let nf = self.new64();
        let z = self.c64(0);
        let f = self.f();
        f.gen_xor_i64(of, x, a);
        f.gen_shri_i64(of, of, w - 1);
        f.gen_andi_i64(of, of, 1);
        f.gen_shli_i64(of, of, 11);
        f.gen_andi_i64(nf, fl, !i64::from(CC_C | CC_O));
        f.gen_or_i64(nf, nf, newcf);
        f.gen_or_i64(nf, nf, of);
        f.gen_movcond_i64(Cond::Ne, g.t0, n, z, a, x);
        f.gen_movcond_i64(Cond::Ne, nf, n, z, nf, fl);
        self.st_rm_t0(ot, d);
        self.f().gen_mov_i64(g.cc_src, nf);
        self.set_cc_op(CC_OP_EFLAGS);
        // cc_src now holds EFLAGS whatever the old state was.
        self.d.cc_op_dirty = true;
    }

    /// SHLD and SHRD: T1 is the source register.
    fn gen_shiftd(&mut self, ot: u32, d: Option<usize>, is_right: bool, count: Count) {
        let g = self.g;
        match d {
            Some(r) => self.mov_v_reg(ot, g.t0, r),
            None => self.ld_v(ot, g.t0, g.a0),
        }
        let c = self.count_temp(ot, count);
        let f = self.f();
        if ot == OT16 || ot == OT32 {
            if ot == OT16 {
                // The Intel behaviour for counts above 16: "shrdw C, B, A" shifts A:B:A >> C.
                if is_right {
                    f.gen_deposit_i64(g.tmp0, g.t0, g.t1, 16, 16);
                    f.gen_mov_i64(g.t1, g.t0);
                    f.gen_mov_i64(g.t0, g.tmp0);
                } else {
                    f.gen_deposit_i64(g.t1, g.t0, g.t1, 16, 16);
                }
            }
            // Concatenate the two 32-bit values and use a 64-bit shift.
            f.gen_subi_i64(g.tmp0, c, 1);
            if is_right {
                f.gen_concat32_i64(g.t0, g.t0, g.t1);
                f.gen_shr_i64(g.tmp0, g.t0, g.tmp0);
                f.gen_shr_i64(g.t0, g.t0, c);
            } else {
                f.gen_concat32_i64(g.t0, g.t1, g.t0);
                f.gen_shl_i64(g.tmp0, g.t0, g.tmp0);
                f.gen_shl_i64(g.t0, g.t0, c);
                f.gen_shri_i64(g.tmp0, g.tmp0, 32);
                f.gen_shri_i64(g.t0, g.t0, 32);
            }
        } else {
            f.gen_subi_i64(g.tmp0, c, 1);
            if is_right {
                f.gen_shr_i64(g.tmp0, g.t0, g.tmp0);
                f.gen_subfi_i64(g.tmp4, 64, c);
                f.gen_shr_i64(g.t0, g.t0, c);
                f.gen_shl_i64(g.t1, g.t1, g.tmp4);
            } else {
                f.gen_shl_i64(g.tmp0, g.t0, g.tmp0);
                f.gen_subfi_i64(g.tmp4, 64, c);
                f.gen_shl_i64(g.t0, g.t0, c);
                f.gen_shr_i64(g.t1, g.t1, g.tmp4);
            }
            f.gen_movi_i64(g.tmp4, 0);
            f.gen_movcond_i64(Cond::Eq, g.t1, c, g.tmp4, g.tmp4, g.t1);
            f.gen_or_i64(g.t0, g.t0, g.t1);
        }
        self.st_rm_t0(ot, d);
        self.gen_shift_flags(ot, g.t0, g.tmp0, c, is_right);
    }

    // Bit instructions.

    fn bt(&mut self, b: u32) -> R {
        let g = self.g;
        let ot = self.d.dflag;
        let (m, md, reg, rm) = self.modrm()?;
        let op;
        if b == 0x1ba {
            op = (m >> 3) & 7;
            if op < 4 || self.bad_lock(md == 3 || op == 4) {
                if op < 4 {
                    self.gen_illegal_opcode();
                }
                return Ok(());
            }
            if md != 3 {
                self.d.rip_offset = 1;
                self.gen_lea_modrm(m)?;
                if !self.lock() {
                    self.ld_v(ot, g.t0, g.a0);
                }
            } else {
                self.mov_v_reg(ot, g.t0, rm);
            }
            let v = self.ldub()?;
            self.f().gen_movi_i64(g.t1, v as i64);
        } else {
            op = match b {
                0x1a3 => 4,
                0x1ab => 5,
                0x1b3 => 6,
                _ => 7,
            };
            if self.bad_lock(md == 3) {
                return Ok(());
            }
            self.mov_v_reg(ot, g.t1, reg);
            if md != 3 {
                // The bit offset can reach outside the addressed operand.
                let a = self.lea_modrm_0(m)?;
                self.ext(ot, g.t1, g.t1, true);
                let f = self.f();
                f.gen_sari_i64(g.tmp0, g.t1, 3 + i64::from(ot));
                f.gen_shli_i64(g.tmp0, g.tmp0, i64::from(ot));
                let ea = self.lea_modrm_1(a);
                self.f().gen_add_i64(g.a0, ea, g.tmp0);
                let (aflag, ovr) = (self.d.aflag, self.d.override_seg);
                self.lea_v_seg(aflag, g.a0, a.def_seg, ovr);
                if !self.lock() {
                    self.ld_v(ot, g.t0, g.a0);
                }
            } else {
                self.mov_v_reg(ot, g.t0, rm);
            }
        }
        let op = op - 4;
        let idx = self.d.mem_index;
        let lock = self.lock();
        let f = self.f();
        f.gen_andi_i64(g.t1, g.t1, (1i64 << (3 + ot)) - 1);
        f.gen_movi_i64(g.tmp0, 1);
        f.gen_shl_i64(g.tmp0, g.tmp0, g.t1);
        if lock {
            match op {
                1 => f.gen_atomic_op_i64(AtomicOp::FetchOr, g.t0, g.a0, g.tmp0, idx, mo(ot)),
                2 => {
                    f.gen_not_i64(g.tmp0, g.tmp0);
                    f.gen_atomic_op_i64(AtomicOp::FetchAnd, g.t0, g.a0, g.tmp0, idx, mo(ot));
                }
                _ => f.gen_atomic_op_i64(AtomicOp::FetchXor, g.t0, g.a0, g.tmp0, idx, mo(ot)),
            }
            f.gen_shr_i64(g.tmp4, g.t0, g.t1);
        } else {
            f.gen_shr_i64(g.tmp4, g.t0, g.t1);
            match op {
                0 => {}
                1 => f.gen_or_i64(g.t0, g.t0, g.tmp0),
                2 => f.gen_andc_i64(g.t0, g.t0, g.tmp0),
                _ => f.gen_xor_i64(g.t0, g.t0, g.tmp0),
            }
            if op != 0 {
                let d = if md == 3 { Some(rm) } else { None };
                self.st_rm_t0(ot, d);
            }
        }
        // Delay the flags until after the store. C is the tested bit, Z is unchanged and
        // the others are undefined.
        let cc_op = self.d.cc_op;
        if (CC_OP_MULB..CC_OP_BMILGB).contains(&cc_op) {
            // Z comes from cc_dst; a SAR op of the same width keeps it and takes C from
            // cc_src.
            let f = self.f();
            f.gen_andi_i64(g.cc_src, g.tmp4, 1);
            self.set_cc_op(((cc_op - CC_OP_MULB) & 3) + CC_OP_SARB);
        } else {
            self.gen_compute_eflags();
            self.f().gen_deposit_i64(g.cc_src, g.cc_src, g.tmp4, 0, 1);
        }
        Ok(())
    }

    fn bsf_bsr(&mut self, b: u32) -> R {
        let g = self.g;
        let ot = self.d.dflag;
        let (m, _, reg, _) = self.modrm()?;
        self.gen_ldst_modrm(m, ot, None, false)?;
        self.ext(ot, g.t0, g.t0, false);
        let is_bsr = b & 1 != 0;
        let size = 8i64 << ot;
        let has = if is_bsr { self.d.feat.abm } else { self.d.feat.bmi1 };
        let repz = self.d.prefix & PREFIX_REPZ != 0;
        let f = self.f();
        if repz && has {
            // LZCNT and TZCNT: C is about the input, Z about the result.
            f.gen_mov_i64(g.cc_src, g.t0);
            if is_bsr {
                f.gen_clzi_i64(g.t0, g.t0, 64);
                f.gen_subi_i64(g.t0, g.t0, 64 - size);
            } else {
                f.gen_ctzi_i64(g.t0, g.t0, size);
            }
            f.gen_mov_i64(g.cc_dst, g.t0);
            self.set_cc_op(CC_OP_BMILGB + ot);
        } else {
            // BSF and BSR: only Z is defined, from the input. A zero input leaves the
            // destination unchanged, as on real hardware.
            f.gen_mov_i64(g.cc_dst, g.t0);
            let r = g.regs[reg];
            if is_bsr {
                f.gen_xori_i64(g.t1, r, 63);
                f.gen_clz_i64(g.t0, g.t0, g.t1);
                f.gen_xori_i64(g.t0, g.t0, 63);
            } else {
                f.gen_ctz_i64(g.t0, g.t0, r);
            }
            self.set_cc_op(CC_OP_LOGICB + ot);
        }
        self.mov_reg_v(ot, reg, g.t0);
        Ok(())
    }

    // Exchanges.

    fn cmpxchg(&mut self, b: u32) -> R {
        let g = self.g;
        let ot = mo_b_d(b, self.d.dflag);
        let (m, md, reg, rm) = self.modrm()?;
        if self.bad_lock(md == 3) {
            return Ok(());
        }
        let oldv = self.new64();
        let newv = self.new64();
        let cmpv = self.ext_new(ot, g.regs[R_EAX], false);
        self.mov_v_reg(ot, newv, reg);
        if self.lock() {
            self.gen_lea_modrm(m)?;
            let idx = self.d.mem_index;
            self.f().gen_atomic_cmpxchg_i64(oldv, g.a0, cmpv, newv, idx, mo(ot));
        } else if md == 3 {
            self.mov_v_reg(ot, oldv, rm);
            self.f().gen_movcond_i64(Cond::Eq, newv, oldv, cmpv, newv, oldv);
            self.mov_reg_v(ot, rm, newv);
        } else {
            self.gen_lea_modrm(m)?;
            self.ld_v(ot, oldv, g.a0);
            self.f().gen_movcond_i64(Cond::Eq, newv, oldv, cmpv, newv, oldv);
            // The store happens whether the compare succeeds or not.
            self.st_v(ot, newv, g.a0);
        }
        // EAX is written only when the compare fails; that matters for the zero extension
        // of a 32-bit operand.
        let rax = g.regs[R_EAX];
        if ot == OT32 {
            let t = self.new64();
            let f = self.f();
            f.gen_movcond_i64(Cond::Eq, t, oldv, cmpv, rax, oldv);
            f.gen_mov_i64(rax, t);
        } else {
            self.mov_reg_v(ot, R_EAX, oldv);
        }
        let f = self.f();
        f.gen_mov_i64(g.cc_src, oldv);
        f.gen_mov_i64(g.cc_srct, cmpv);
        f.gen_sub_i64(g.cc_dst, cmpv, oldv);
        self.set_cc_op(CC_OP_SUBB + ot);
        Ok(())
    }

    /// Group 9: CMPXCHG8B and CMPXCHG16B.
    fn grp9(&mut self) -> R {
        let g = self.g;
        let (m, md, _, _) = self.modrm()?;
        let op = (m >> 3) & 7;
        if op == 6 || op == 7 {
            let rm = (m & 7) as usize | self.d.rex_b;
            self.rdrand(md, op, rm);
            return Ok(());
        }
        let wide = self.code64() && self.d.rex_w;
        let ok = op == 1 && md != 3 && if wide { self.d.feat.cx16 } else { self.d.feat.cx8 };
        if !ok {
            self.gen_illegal_opcode();
            return Ok(());
        }
        self.gen_lea_modrm(m)?;
        let idx = self.d.mem_index;
        let lock = self.lock();
        let (rax, rdx, rbx, rcx) = (g.regs[R_EAX], g.regs[R_EDX], g.regs[R_EBX], g.regs[R_ECX]);
        if wide {
            let mop = MemOp::MO_128 | MemOp::ALIGN;
            let cmp: TempI128 = self.f().temp_new_i128();
            let val: TempI128 = self.f().temp_new_i128();
            let t0 = self.new64();
            let t1 = self.new64();
            let f = self.f();
            f.gen_concat_i64_i128(cmp, rax, rdx);
            f.gen_concat_i64_i128(val, rbx, rcx);
            if lock {
                f.gen_atomic_cmpxchg_i128(val, g.a0, cmp, val, idx, mop);
            } else {
                f.gen_nonatomic_cmpxchg_i128(val, g.a0, cmp, val, idx, mop);
            }
            f.gen_extr_i128_i64(g.t0, g.t1, val);
            // Success is determined after the fact.
            f.gen_xor_i64(t0, g.t0, rax);
            f.gen_xor_i64(t1, g.t1, rdx);
            f.gen_or_i64(t0, t0, t1);
            self.gen_compute_eflags();
            let f = self.f();
            f.gen_setcondi_i64(Cond::Eq, t0, t0, 0);
            f.gen_deposit_i64(g.cc_src, g.cc_src, t0, 6, 1);
            // On success the old value equals RDX:RAX, so this is unconditional.
            f.gen_mov_i64(rax, g.t0);
            f.gen_mov_i64(rdx, g.t1);
        } else {
            let cmp = self.new64();
            let val = self.new64();
            let old = self.new64();
            let z = self.new64();
            let zero = self.c64(0);
            let f = self.f();
            f.gen_concat32_i64(cmp, rax, rdx);
            f.gen_concat32_i64(val, rbx, rcx);
            if lock {
                f.gen_atomic_cmpxchg_i64(old, g.a0, cmp, val, idx, MemOp::MO_64);
            } else {
                f.gen_nonatomic_cmpxchg_i64(old, g.a0, cmp, val, idx, MemOp::MO_64);
            }
            f.gen_setcond_i64(Cond::Eq, z, old, cmp);
            // Leave the registers alone on success and zero extend them on failure.
            f.gen_extr32_i64(g.t0, g.t1, old);
            f.gen_movcond_i64(Cond::Eq, rax, z, zero, g.t0, rax);
            f.gen_movcond_i64(Cond::Eq, rdx, z, zero, g.t1, rdx);
            self.gen_compute_eflags();
            self.f().gen_deposit_i64(g.cc_src, g.cc_src, z, 6, 1);
        }
        Ok(())
    }

    // Stack frames.

    fn gen_pusha(&mut self) {
        let g = self.g;
        let d_ot = self.d.dflag;
        let a_ot = self.mo_stacksize();
        let size = 1i64 << d_ot;
        for i in 0..8 {
            self.f().gen_addi_i64(g.a0, g.regs[R_ESP], (i - 8) * size);
            self.lea_v_seg(a_ot, g.a0, R_SS as i32, -1);
            self.st_v(d_ot, g.regs[7 - i as usize], g.a0);
        }
        self.gen_stack_update(-8 * size);
    }

    fn gen_popa(&mut self) {
        let g = self.g;
        let d_ot = self.d.dflag;
        let a_ot = self.mo_stacksize();
        let size = 1i64 << d_ot;
        for i in 0..8 {
            // ESP is not reloaded.
            if 7 - i == R_ESP as i64 {
                continue;
            }
            self.f().gen_addi_i64(g.a0, g.regs[R_ESP], i * size);
            self.lea_v_seg(a_ot, g.a0, R_SS as i32, -1);
            self.ld_v(d_ot, g.t0, g.a0);
            self.mov_reg_v(d_ot, 7 - i as usize, g.t0);
        }
        self.gen_stack_update(8 * size);
    }

    /// `gen_lea_ss_ofs()`.
    fn lea_ss_ofs(&mut self, src: TempI64, ofs: i64) {
        let a_ot = self.mo_stacksize();
        let t = self.new64();
        self.f().gen_addi_i64(t, src, ofs);
        self.lea_v_seg(a_ot, t, R_SS as i32, -1);
    }

    fn gen_enter(&mut self, esp_addend: i64, level: i64) {
        let g = self.g;
        let d_ot = self.mo_pushpop(self.d.dflag);
        let a_ot = self.mo_stacksize();
        let size = 1i64 << d_ot;
        // Push BP and compute FrameTemp into T1.
        self.f().gen_subi_i64(g.t1, g.regs[R_ESP], size);
        self.lea_v_seg(a_ot, g.t1, R_SS as i32, -1);
        self.st_v(d_ot, g.regs[R_EBP], g.a0);
        let level = level & 31;
        if level != 0 {
            // Copy level-1 pointers from the previous frame.
            let fp = self.new64();
            for i in 1..level {
                self.lea_ss_ofs(g.regs[R_EBP], -size * i);
                self.ld_v(d_ot, fp, g.a0);
                self.lea_ss_ofs(g.t1, -size * i);
                self.st_v(d_ot, fp, g.a0);
            }
            // Push the current FrameTemp as the last level.
            self.lea_ss_ofs(g.t1, -size * level);
            self.st_v(d_ot, g.t1, g.a0);
        }
        // Copy the FrameTemp value to EBP.
        self.mov_reg_v(d_ot, R_EBP, g.t1);
        // Compute the final value of ESP.
        self.f().gen_subi_i64(g.t1, g.t1, esp_addend + size * level);
        self.mov_reg_v(a_ot, R_ESP, g.t1);
    }

    // Far transfers.

    /// Far CALL to T0:T1.
    fn do_lcall(&mut self) {
        let g = self.g;
        let sel = self.trunc32(g.t0);
        let shift = self.c32(self.d.dflag as i32 - 1);
        let next = self.eip_next();
        let n = self.c64(next as i64);
        let h = if self.d.pe && !self.d.vm86 {
            &helpers::LCALL_PROTECTED
        } else {
            &helpers::LCALL_REAL
        };
        self.env_call(h, None, &[sel.into(), g.t1.into(), shift.into(), n.into()]);
        self.b.is_jmp = DISAS_JUMP;
    }

    /// Far JMP to T0:T1.
    fn do_ljmp(&mut self) {
        let g = self.g;
        if self.d.pe && !self.d.vm86 {
            let sel = self.trunc32(g.t0);
            self.env_call(&helpers::LJMP_PROTECTED, None, &[sel.into(), g.t1.into()]);
        } else {
            self.movl_seg_real(R_CS, g.t0);
            self.gen_op_jmp_v(g.t1);
        }
        self.b.is_jmp = DISAS_JUMP;
    }

    fn gen_lret(&mut self, val: i64) {
        let g = self.g;
        let dflag = self.d.dflag;
        if self.d.pe && !self.d.vm86 {
            self.gen_update_cc_op();
            self.gen_update_eip_cur();
            let shift = self.c32(dflag as i32 - 1);
            let add = self.c64(val);
            self.env_call(&helpers::LRET_PROTECTED, None, &[shift.into(), add.into()]);
        } else {
            self.gen_stack_a0();
            // Pop the offset. Keeping EIP updated is fine on an exception.
            self.ld_v(dflag, g.t0, g.a0);
            self.gen_op_jmp_v(g.t0);
            // Pop the selector.
            self.add_a0_im(1 << dflag);
            self.ld_v(dflag, g.t0, g.a0);
            self.movl_seg_real(R_CS, g.t0);
            self.gen_stack_update(val + (2 << dflag));
        }
        self.b.is_jmp = DISAS_EOB_ONLY;
    }

    /// LDS, LES, LSS, LFS and LGS.
    fn gen_lxs(&mut self, s: usize, m: u32) -> R {
        let g = self.g;
        let (md, reg, _) = self.split_modrm(m);
        if md == 3 {
            self.gen_illegal_opcode();
            return Ok(());
        }
        let ot = match self.d.dflag {
            OT16 => OT16,
            OT64 if self.d.feat.intel => OT64,
            _ => OT32,
        };
        self.gen_lea_modrm(m)?;
        self.ld_v(ot, g.t1, g.a0);
        self.add_a0_im(1 << ot);
        // Load the segment first to handle exceptions properly.
        self.ld_v(OT16, g.t0, g.a0);
        self.gen_movl_seg(s, g.t0);
        self.mov_reg_v(ot, reg, g.t1);
        Ok(())
    }

    fn grp45(&mut self, b: u32) -> R {
        let g = self.g;
        let (m, md, _, rm) = self.modrm()?;
        let op = (m >> 3) & 7;
        if (b == 0xfe && op > 1) || op == 7 {
            self.gen_illegal_opcode();
            return Ok(());
        }
        if self.bad_lock(md == 3 || op > 1) {
            return Ok(());
        }
        let mut ot = mo_b_d(b, self.d.dflag);
        if op == 2 || op == 4 {
            self.near_branch_ot();
            ot = self.d.dflag;
        } else if self.code64() && (op == 3 || op == 5) {
            ot = if self.d.dflag != OT16 { OT32 + u32::from(self.d.rex_w) } else { OT16 };
        } else if op == 6 {
            ot = self.mo_pushpop(self.d.dflag);
        }
        if md != 3 {
            self.gen_lea_modrm(m)?;
            if op >= 2 && op != 3 && op != 5 {
                self.ld_v(ot, g.t0, g.a0);
            }
        } else if op >= 2 {
            self.mov_v_reg(ot, g.t0, rm);
        }
        let d = if md == 3 { Some(rm) } else { None };
        match op {
            0 | 1 => self.gen_inc(ot, d, if op == 0 { 1 } else { -1 }),
            2 | 4 => {
                if self.d.dflag == OT16 {
                    self.f().gen_ext16u_i64(g.t0, g.t0);
                }
                if op == 2 {
                    let next = self.eip_next();
                    let t = self.c64(next as i64);
                    self.gen_push_v(t);
                }
                self.gen_op_jmp_v(g.t0);
                self.b.is_jmp = DISAS_JUMP;
            }
            3 | 5 => {
                if md == 3 {
                    self.gen_illegal_opcode();
                    return Ok(());
                }
                self.ld_v(ot, g.t1, g.a0);
                self.add_a0_im(1 << ot);
                self.ld_v(OT16, g.t0, g.a0);
                if op == 3 { self.do_lcall() } else { self.do_ljmp() }
            }
            _ => self.gen_push_v(g.t0),
        }
        Ok(())
    }

    fn bound(&mut self) -> R {
        if self.inv64() {
            return Ok(());
        }
        let g = self.g;
        let ot = self.d.dflag;
        let (m, md, reg, _) = self.modrm()?;
        if md == 3 {
            // EVEX prefix, or an invalid form.
            self.gen_illegal_opcode();
            return Ok(());
        }
        self.mov_v_reg(ot, g.t0, reg);
        self.gen_lea_modrm(m)?;
        let v = self.trunc32(g.t0);
        let h = if ot == OT16 { &helpers::BOUNDW } else { &helpers::BOUNDL };
        self.env_call(h, None, &[g.a0.into(), v.into()]);
        Ok(())
    }

    fn movsxd_arpl(&mut self) -> R {
        let g = self.g;
        let (m, md, reg, rm) = self.modrm()?;
        if self.code64() {
            // MOVSXD.
            self.gen_ldst_modrm(m, OT32, None, false)?;
            let ot = self.d.dflag;
            if ot == OT64 {
                self.f().gen_ext32s_i64(g.t0, g.t0);
            }
            self.mov_reg_v(ot, reg, g.t0);
            return Ok(());
        }
        if !self.d.pe || self.d.vm86 {
            self.gen_illegal_opcode();
            return Ok(());
        }
        // ARPL.
        let reg = (m >> 3) as usize & 7;
        let rm = rm & 7;
        if md != 3 {
            self.gen_lea_modrm(m)?;
            self.ld_v(OT16, g.t0, g.a0);
        } else {
            self.mov_v_reg(OT16, g.t0, rm);
        }
        self.mov_v_reg(OT16, g.t1, reg);
        let rpl = self.new64();
        let adj = self.new64();
        let z = self.new64();
        let zero = self.c64(0);
        let czf = self.c64(i64::from(CC_Z));
        let f = self.f();
        f.gen_andi_i64(rpl, g.t0, 3);
        f.gen_andi_i64(g.t1, g.t1, 3);
        f.gen_andi_i64(adj, g.t0, !3);
        f.gen_or_i64(adj, adj, g.t1);
        f.gen_movcond_i64(Cond::Lt, g.t0, rpl, g.t1, adj, g.t0);
        f.gen_movcond_i64(Cond::Lt, z, rpl, g.t1, czf, zero);
        if md != 3 {
            self.st_v(OT16, g.t0, g.a0);
        } else {
            self.mov_reg_v(OT16, rm, g.t0);
        }
        self.gen_compute_eflags();
        let f = self.f();
        f.gen_andi_i64(g.cc_src, g.cc_src, !i64::from(CC_Z));
        f.gen_or_i64(g.cc_src, g.cc_src, z);
        Ok(())
    }

    // Strings.

    fn string_a0_esi(&mut self) {
        let esi = self.g.regs[R_ESI];
        let (aflag, ovr) = (self.d.aflag, self.d.override_seg);
        self.lea_v_seg(aflag, esi, R_DS as i32, ovr);
    }

    fn string_a0_edi(&mut self) {
        let edi = self.g.regs[R_EDI];
        let aflag = self.d.aflag;
        self.lea_v_seg(aflag, edi, R_ES as i32, -1);
    }

    /// `gen_op_movl_T0_Dshift()`: the step, from DF.
    fn dshift(&mut self, ot: u32) -> TempI64 {
        let t = self.new64();
        let env = self.g.env;
        let f = self.f();
        f.gen_ld32s_i64(t, env, DF as i64);
        f.gen_shli_i64(t, t, i64::from(ot));
        t
    }

    fn step(&mut self, ot: u32, r: usize) {
        let d = self.dshift(ot);
        let aflag = self.d.aflag;
        self.add_reg(aflag, r, d);
    }

    fn gen_movs(&mut self, ot: u32) {
        let g = self.g;
        self.string_a0_esi();
        self.ld_v(ot, g.t0, g.a0);
        self.string_a0_edi();
        self.st_v(ot, g.t0, g.a0);
        self.step(ot, R_ESI);
        self.step(ot, R_EDI);
    }

    fn gen_stos(&mut self, ot: u32) {
        let g = self.g;
        self.mov_v_reg(ot, g.t0, R_EAX);
        self.string_a0_edi();
        self.st_v(ot, g.t0, g.a0);
        self.step(ot, R_EDI);
    }

    fn gen_lods(&mut self, ot: u32) {
        let g = self.g;
        self.string_a0_esi();
        self.ld_v(ot, g.t0, g.a0);
        self.mov_reg_v(ot, R_EAX, g.t0);
        self.step(ot, R_ESI);
    }

    fn gen_scas(&mut self, ot: u32) {
        let g = self.g;
        self.string_a0_edi();
        self.ld_v(ot, g.t1, g.a0);
        self.gen_op(ot, OP_CMP, Some(R_EAX));
        self.step(ot, R_EDI);
    }

    fn gen_cmps(&mut self, ot: u32) {
        let g = self.g;
        self.string_a0_edi();
        self.ld_v(ot, g.t1, g.a0);
        self.string_a0_esi();
        self.gen_op(ot, OP_CMP, None);
        self.step(ot, R_ESI);
        self.step(ot, R_EDI);
    }

    fn port_dx(&mut self) -> TempI32 {
        let t = self.new64();
        let edx = self.g.regs[R_EDX];
        self.f().gen_ext16u_i64(t, edx);
        self.trunc32(t)
    }

    fn gen_ins(&mut self, ot: u32) {
        let g = self.g;
        self.string_a0_edi();
        // A dummy write first, so that a page fault leaves the port alone.
        self.f().gen_movi_i64(g.t0, 0);
        self.st_v(ot, g.t0, g.a0);
        let port = self.port_dx();
        self.gen_in(ot, port);
        self.st_v(ot, g.t0, g.a0);
        self.step(ot, R_EDI);
    }

    fn gen_outs(&mut self, ot: u32) {
        let g = self.g;
        self.string_a0_esi();
        self.ld_v(ot, g.t0, g.a0);
        let port = self.port_dx();
        let v = self.trunc32(g.t0);
        self.gen_out(ot, port, v);
        self.step(ot, R_ESI);
    }

    fn jcc_ecx(&mut self, cond: Cond, l: Label) {
        let aflag = self.d.aflag;
        let ecx = self.g.regs[R_ECX];
        let t = self.ext_new(aflag, ecx, false);
        self.f().gen_brcondi_i64(cond, t, 0, l);
    }

    /// `gen_repz()`: run `op` once per block execution, jumping back to the instruction
    /// until ECX runs out (or the condition of CMPS and SCAS fails).
    fn gen_repz(&mut self, ot: u32, op: StrOp, cond: Option<bool>) {
        let l2 = self.label();
        self.gen_update_cc_op();
        // A fault can come from any iteration, after the flags of the previous one.
        if let Some(is) = self.b.insn_start {
            self.f().set_insn_start_param(is, 1, u64::from(CC_OP_DYNAMIC));
        }
        self.jcc_ecx(Cond::Eq, l2);
        self.str_op(op, ot);
        let aflag = self.d.aflag;
        self.add_reg_im(aflag, R_ECX, -1);
        if let Some(repnz) = cond {
            self.gen_jcc1((JCC_Z << 1) | u32::from(!repnz), l2);
        }
        if self.d.repz_opt {
            self.jcc_ecx(Cond::Eq, l2);
        }
        let len = self.cur_insn_len();
        self.gen_jmp_rel_csize(-len, 0);
        self.set_label(l2);
        self.gen_jmp_rel_csize(0, 1);
    }

    fn string(&mut self, b: u32) {
        let ot = mo_b_d(b, self.d.dflag);
        let op: StrOp = b & !1;
        let is_cond = op == 0xa6 || op == 0xae;
        let p = self.d.prefix;
        if p & (PREFIX_REPZ | PREFIX_REPNZ) != 0 {
            let cond = if is_cond { Some(p & PREFIX_REPNZ != 0) } else { None };
            self.gen_repz(ot, op, cond);
        } else {
            self.str_op(op, ot);
        }
    }

    fn str_op(&mut self, op: StrOp, ot: u32) {
        match op {
            0x6c => self.gen_ins(ot),
            0x6e => self.gen_outs(ot),
            0xa4 => self.gen_movs(ot),
            0xa6 => self.gen_cmps(ot),
            0xaa => self.gen_stos(ot),
            0xac => self.gen_lods(ot),
            _ => self.gen_scas(ot),
        }
    }

    // I/O.

    /// `gen_check_io()`: the I/O permission check.
    fn gen_check_io(&mut self, ot: u32, port: TempI32) {
        if self.d.pe && (self.d.cpl > self.d.iopl || self.d.vm86) {
            let size = self.c32(1 << ot);
            self.env_call(&helpers::CHECK_IO, None, &[port.into(), size.into()]);
        }
    }

    fn gen_in(&mut self, ot: u32, port: TempI32) {
        let t0 = self.g.t0;
        let h = match ot {
            OT8 => &helpers::INB,
            OT16 => &helpers::INW,
            _ => &helpers::INL,
        };
        self.env_call(h, Some(t0.into()), &[port.into()]);
    }

    fn gen_out(&mut self, ot: u32, port: TempI32, v: TempI32) {
        let h = match ot {
            OT8 => &helpers::OUTB,
            OT16 => &helpers::OUTW,
            _ => &helpers::OUTL,
        };
        self.env_call(h, None, &[port.into(), v.into()]);
    }

    fn in_out(&mut self, b: u32) -> R {
        let g = self.g;
        let ot = mo_b_d32(b, self.d.dflag);
        let port = if b < 0xe8 {
            let p = self.ldub()?;
            self.c32(p as i32)
        } else {
            self.port_dx()
        };
        self.gen_check_io(ot, port);
        self.b.translator_io_start();
        if b & 2 == 0 {
            self.gen_in(ot, port);
            self.mov_reg_v(ot, R_EAX, g.t0);
        } else {
            self.mov_v_reg(ot, g.t1, R_EAX);
            let v = self.trunc32(g.t1);
            self.gen_out(ot, port, v);
        }
        Ok(())
    }

    fn ins_outs(&mut self, b: u32) {
        let ot = mo_b_d32(b, self.d.dflag);
        let port = self.port_dx();
        self.gen_check_io(ot, port);
        self.b.translator_io_start();
        let op: StrOp = b & !1;
        if self.d.prefix & (PREFIX_REPZ | PREFIX_REPNZ) != 0 {
            self.gen_repz(ot, op, None);
        } else {
            self.str_op(op, ot);
        }
    }

    // System instructions.

    fn mov_cr_dr(&mut self, b: u32) -> R {
        let g = self.g;
        if !self.check_cpl0() {
            return Ok(());
        }
        // The mod bits are ignored and taken as 3.
        let (_, _, reg, rm) = self.modrm()?;
        let ot = if self.code64() { OT64 } else { OT32 };
        let is_cr = b & 1 == 0;
        if is_cr && !matches!(reg, 0 | 2 | 3 | 4 | 8) || !is_cr && reg >= 8 {
            self.gen_illegal_opcode();
            return Ok(());
        }
        let n = self.c32(reg as i32);
        if is_cr {
            self.b.translator_io_start();
        }
        if b & 2 != 0 {
            self.mov_v_reg(ot, g.t0, rm);
            let h = if is_cr { &helpers::WRITE_CRN } else { &helpers::SET_DR };
            self.env_call(h, None, &[n.into(), g.t0.into()]);
            self.b.is_jmp = DISAS_EOB_NEXT;
        } else {
            let h = if is_cr { &helpers::READ_CRN } else { &helpers::GET_DR };
            self.env_call(h, Some(g.t0.into()), &[n.into()]);
            self.mov_reg_v(ot, rm, g.t0);
        }
        Ok(())
    }

    /// Group 6: SLDT, STR, LLDT, LTR, VERR and VERW.
    fn grp6(&mut self) -> R {
        let g = self.g;
        let (m, md, _, _) = self.modrm()?;
        let op = (m >> 3) & 7;
        if !self.d.pe || self.d.vm86 || op >= 6 {
            self.gen_illegal_opcode();
            return Ok(());
        }
        match op {
            0 | 1 => {
                if self.d.flags & HF_UMIP_MASK != 0 && !self.check_cpl0() {
                    return Ok(());
                }
                let off = if op == 0 { LDT } else { TR } + SEG_SELECTOR;
                let env = g.env;
                self.f().gen_ld32u_i64(g.t0, env, off as i64);
                let ot = if md == 3 { self.d.dflag } else { OT16 };
                self.gen_ldst_modrm(m, ot, None, true)?;
            }
            2 | 3 => {
                if self.check_cpl0() {
                    self.gen_ldst_modrm(m, OT16, None, false)?;
                    let t = self.trunc32(g.t0);
                    let h = if op == 2 { &helpers::LLDT } else { &helpers::LTR };
                    self.env_call(h, None, &[t.into()]);
                }
            }
            _ => {
                self.gen_ldst_modrm(m, OT16, None, false)?;
                let t = self.trunc32(g.t0);
                let h = if op == 4 { &helpers::VERR } else { &helpers::VERW };
                self.env_call(h, None, &[t.into()]);
                self.assume_cc_op(CC_OP_EFLAGS);
            }
        }
        Ok(())
    }

    /// Group 7: the descriptor table registers, SMSW, LMSW, INVLPG and friends.
    fn grp7(&mut self) -> R {
        let g = self.g;
        let (m, md, _, _) = self.modrm()?;
        let op = (m >> 3) & 7;
        match m {
            0xd0 | 0xd1 => {
                self.xgetbv_xsetbv(m);
                return Ok(());
            }
            0xca | 0xcb => {
                // CLAC and STAC.
                if !self.d.feat.smap || !self.check_cpl0() {
                    if !self.d.feat.smap {
                        self.gen_illegal_opcode();
                    }
                    return Ok(());
                }
                if m == 0xca {
                    self.gen_reset_eflags(AC_MASK);
                } else {
                    self.gen_set_eflags(AC_MASK);
                }
                self.b.is_jmp = DISAS_EOB_NEXT;
                return Ok(());
            }
            0xf8 => {
                // SWAPGS.
                if !self.code64() {
                    self.gen_illegal_opcode();
                } else if self.check_cpl0() {
                    let gs = g.seg_base[R_GS];
                    let f = self.f();
                    f.gen_mov_i64(g.t0, gs);
                    f.gen_ld_i64(gs, g.env, KERNELGSBASE as i64);
                    f.gen_st_i64(g.t0, g.env, KERNELGSBASE as i64);
                }
                return Ok(());
            }
            0xf9 => {
                // RDTSCP.
                if !self.d.feat.rdtscp {
                    self.gen_illegal_opcode();
                    return Ok(());
                }
                self.gen_update_cc_op();
                self.gen_update_eip_cur();
                self.b.translator_io_start();
                self.env_call(&helpers::RDTSCP, None, &[]);
                return Ok(());
            }
            _ => {}
        }
        match op {
            0 | 1 if md != 3 => {
                // SGDT and SIDT.
                if self.d.flags & HF_UMIP_MASK != 0 && !self.check_cpl0() {
                    return Ok(());
                }
                let base = if op == 0 { GDT } else { IDT };
                self.gen_lea_modrm(m)?;
                let t = self.ld_env32(base + SEG_LIMIT);
                self.st_v(OT16, t, g.a0);
                self.add_a0_im(2);
                let t = self.ld_env64(base + SEG_BASE);
                // All 32 bits are written whatever the operand size.
                let ot = if self.code64() { OT64 } else { OT32 };
                self.st_v(ot, t, g.a0);
            }
            2 | 3 if md != 3 => {
                // LGDT and LIDT.
                if !self.check_cpl0() {
                    return Ok(());
                }
                let base = if op == 2 { GDT } else { IDT };
                self.gen_lea_modrm(m)?;
                self.ld_v(OT16, g.t1, g.a0);
                self.add_a0_im(2);
                let ot = if self.code64() { OT64 } else { OT32 };
                self.ld_v(ot, g.t0, g.a0);
                if self.d.dflag == OT16 {
                    self.f().gen_andi_i64(g.t0, g.t0, 0xff_ffff);
                }
                self.st_env64(g.t0, base + SEG_BASE);
                self.st_env32(g.t1, base + SEG_LIMIT);
            }
            4 => {
                // SMSW.
                if self.d.flags & HF_UMIP_MASK != 0 && !self.check_cpl0() {
                    return Ok(());
                }
                let env = g.env;
                self.f().gen_ld_i64(g.t0, env, cr(0) as i64);
                let ot = if md != 3 { OT16 } else { self.d.dflag };
                self.gen_ldst_modrm(m, ot, None, true)?;
            }
            6 => {
                // LMSW.
                if !self.check_cpl0() {
                    return Ok(());
                }
                self.gen_ldst_modrm(m, OT16, None, false)?;
                self.env_call(&helpers::LMSW, None, &[g.t0.into()]);
                self.b.is_jmp = DISAS_EOB_NEXT;
            }
            7 if md != 3 => {
                // INVLPG.
                if !self.check_cpl0() {
                    return Ok(());
                }
                self.gen_lea_modrm(m)?;
                self.env_call(&helpers::INVLPG, None, &[g.a0.into()]);
                self.b.is_jmp = DISAS_EOB_NEXT;
            }
            _ => self.gen_illegal_opcode(),
        }
        let _ = CR;
        Ok(())
    }

    fn lar_lsl(&mut self, b: u32) -> R {
        let g = self.g;
        if !self.d.pe || self.d.vm86 {
            self.gen_illegal_opcode();
            return Ok(());
        }
        let ot = if self.d.dflag != OT16 { OT32 } else { OT16 };
        let (m, _, reg, _) = self.modrm()?;
        self.gen_ldst_modrm(m, OT16, None, false)?;
        let sel = self.trunc32(g.t0);
        let t = self.new64();
        let h = if b == 0x102 { &helpers::LAR } else { &helpers::LSL };
        self.env_call(h, Some(t.into()), &[sel.into()]);
        self.assume_cc_op(CC_OP_EFLAGS);
        let l = self.label();
        let f = self.f();
        f.gen_andi_i64(g.tmp0, g.cc_src, i64::from(CC_Z));
        f.gen_brcondi_i64(Cond::Eq, g.tmp0, 0, l);
        self.mov_reg_v(ot, reg, t);
        self.set_label(l);
        Ok(())
    }

    /// Group 15: fences and CLFLUSH; the rest needs SSE or XSAVE state.
    fn grp15(&mut self) -> R {
        let (m, md, _, _) = self.modrm()?;
        let op = (m >> 3) & 7;
        let rep = self.d.prefix & (PREFIX_REPZ | PREFIX_REPNZ | PREFIX_DATA);
        if md != 3 && (op == 2 || op == 3) && rep == 0 {
            return self.ldst_mxcsr(m);
        }
        if md == 3 && op < 4 && rep == PREFIX_REPZ {
            self.fsgsbase(m);
            return Ok(());
        }
        if md != 3 {
            if op == 7 && self.d.prefix & (PREFIX_REPZ | PREFIX_REPNZ) == 0 {
                // CLFLUSH and CLFLUSHOPT: nothing is cached.
                self.lea_modrm_0(m)?;
            } else {
                self.gen_illegal_opcode();
            }
            return Ok(());
        }
        if self.d.prefix & (PREFIX_REPZ | PREFIX_REPNZ | PREFIX_DATA) != 0 {
            self.gen_illegal_opcode();
            return Ok(());
        }
        match op {
            5 => self.f().gen_mb(mb::LD_LD | mb::BAR_SC),
            6 => self.f().gen_mb(mb::ALL | mb::BAR_SC),
            7 => self.f().gen_mb(mb::ST_ST | mb::BAR_SC),
            _ => self.gen_illegal_opcode(),
        }
        Ok(())
    }
}
