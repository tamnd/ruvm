// SPDX-License-Identifier: GPL-2.0-or-later

//! The A32 and T32 translator: the integer part of QEMU's `target/arm/tcg/translate.c`.
//!
//! The decoders come from a32.decode, a32-uncond.decode, t32.decode, t16.decode, vfp.decode
//! and vfp-uncond.decode (see `build.rs`). [`S`] implements their `trans_` hooks for the
//! integer instructions, the barriers and the cp15 thread ID registers; the `fp` module has
//! the VFP ones. Every pattern left out returns false, which raises an Undefined Instruction
//! exception as QEMU's `unallocated_encoding()` does.
//!
//! As in QEMU, R0 to R14, the PC and the NZCV flags are TCG globals. R15 is the PC field,
//! which holds the zero extended 32-bit PC in AArch32 state.
//!
//! Not here yet, and so UNDEF: Neon, the BF16 VCVT, VJCVT, the M profile, SWP, and what
//! only AArch32 system mode needs (exception returns, banked registers, SPSR, RFE, SRS,
//! CPS, LDM and STM of the User registers, and the coprocessor registers other than
//! TPIDRURW, TPIDRURO and the CP15 barriers). Only the user mode emulator runs AArch32 code so far, so the
//! unprivileged loads and stores use the normal MMU index.

use ruvm_jit::{Cpu, CpuLoopExit, DisasContextBase, DisasJumpType, TranslatorOps};
use ruvm_jit_core::ir::{TempI32, TempI64, TempPtr};
use ruvm_jit_core::types::{Cond, bswap, mo};
use ruvm_jit_core::{Func, Label, MemOp, Temp};
use ruvm_mem::Endian;

mod fp;

use super::a32_helper as h;
use super::helpers::{self, Def};
use super::{TB_ALIGN_MEM, TB_BE_DATA, TB_EL_MASK, TB_MMUIDX_SHIFT, TB_PSTATE_IL};
use crate::cpu::{
    ArmCpuModel, CF, CONDEXEC_BITS, CPSR_EXEC, CPSR_NZCV, CPSR_USER, CpuArmState, EXCLUSIVE_ADDR,
    EXCLUSIVE_VAL, EXCP_BKPT, EXCP_KERNEL_TRAP, EXCP_SWI, EXCP_UDEF, GE, NF, PC, QF, THUMB, VF, ZF,
    env_off, reg_off,
};
use crate::syndrome::{syn_aa32_bkpt, syn_aa32_svc, syn_illegalstate, syn_uncategorized};

#[allow(missing_docs, unreachable_pub, clippy::pedantic, clippy::nursery)]
mod a32 {
    include!(concat!(env!("OUT_DIR"), "/a32_decode.rs"));

    /// Decode and translate one A32 instruction.
    pub(super) fn disas(ctx: &mut impl DisasA32, insn: u32) -> bool {
        disas_a32(ctx, insn)
    }
}

#[allow(missing_docs, unreachable_pub, clippy::pedantic, clippy::nursery)]
mod a32_uncond {
    use super::a32::*;

    include!(concat!(env!("OUT_DIR"), "/a32_uncond_decode.rs"));

    /// Decode and translate one A32 instruction with the condition 0b1111.
    pub(super) fn disas(ctx: &mut impl DisasA32Uncond, insn: u32) -> bool {
        disas_a32_uncond(ctx, insn)
    }
}

#[allow(missing_docs, unreachable_pub, clippy::pedantic, clippy::nursery)]
mod t32 {
    use super::a32::*;
    use super::a32_uncond::*;

    include!(concat!(env!("OUT_DIR"), "/t32_decode.rs"));

    /// Decode and translate one 32-bit T32 instruction.
    pub(super) fn disas(ctx: &mut impl DisasT32, insn: u32) -> bool {
        disas_t32(ctx, insn)
    }
}

#[allow(missing_docs, unreachable_pub, clippy::pedantic, clippy::nursery)]
mod t16 {
    use super::a32::*;
    use super::a32_uncond::*;
    use super::t32::*;

    include!(concat!(env!("OUT_DIR"), "/t16_decode.rs"));

    /// Decode and translate one 16-bit T32 instruction.
    pub(super) fn disas(ctx: &mut impl DisasT16, insn: u16) -> bool {
        disas_t16(ctx, insn)
    }
}

use a32::{
    DisasA32, arg_bfi, arg_bfx, arg_empty, arg_i, arg_ldrex, arg_ldst_block, arg_ldst_ri,
    arg_ldst_rr, arg_mcr, arg_mrs_reg, arg_msr_i, arg_msr_reg, arg_pkh, arg_r, arg_ri, arg_rr,
    arg_rrr, arg_rrr_rot, arg_rrrr, arg_s_rri_rot, arg_s_rrr_shi, arg_s_rrr_shr, arg_s_rrrr,
    arg_sat, arg_strex,
};
use a32_uncond::{DisasA32Uncond, arg_cps, arg_setend};
use t16::{DisasT16, arg_disas_t1617, arg_disas_t1618};
use t32::{DisasT32, arg_ci, arg_ldst_ri2, arg_tbranch};

/// `DISAS_JUMP`: the PC (and maybe the Thumb bit) is written; look the next block up.
const DISAS_JUMP: DisasJumpType = DisasJumpType::Target(0);
/// `DISAS_EXIT`: the PC is written; go back to the main loop.
const DISAS_EXIT: DisasJumpType = DisasJumpType::Target(1);
/// `DISAS_UPDATE_EXIT`: write the PC of the next instruction and go back to the main loop.
const DISAS_UPDATE_EXIT: DisasJumpType = DisasJumpType::Target(2);
/// `DISAS_SWI`: raise the SVC exception after the instruction.
const DISAS_SWI: DisasJumpType = DisasJumpType::Target(3);

/// The size of a target page, for the Thumb page crossing checks.
const PAGE_SIZE: u64 = 0x1000;

/// The target part of the disassembly context, `DisasContext`.
pub(crate) struct DisasContext {
    model: ArmCpuModel,
    pc_curr: u64,
    /// The start of the page of the first instruction.
    page_start: u64,
    thumb: bool,
    /// The IT block state: the mask shifted left by one, and the condition.
    condexec_mask: u32,
    condexec_cond: u32,
    /// The instruction is conditional and `condlabel` is where its skip path goes.
    condjmp: bool,
    condlabel: Option<Label>,
    /// `MemOp::BE` when CPSR.E is set, else `MemOp::LE`.
    be_data: MemOp,
    align_mem: bool,
    mmu_idx: u32,
    current_el: u32,
    pstate_il: bool,
    /// The immediate of the SVC that ends the block.
    svc_imm: u32,
    /// The TCG globals of the block, made by `init_disas_context`.
    g: Option<G>,
}

/// The TCG globals: QEMU's `cpu_R`, with R15 the 64-bit PC, and `cpu_NF` to `cpu_VF`.
#[derive(Clone, Copy)]
struct G {
    r: [TempI32; 15],
    pc: TempI64,
    nf: TempI32,
    zf: TempI32,
    cf: TempI32,
    vf: TempI32,
}

/// The names QEMU gives the register globals.
const REG_NAMES: [&str; 15] =
    ["r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12", "r13", "r14"];

impl DisasContext {
    /// A context for the CPU `model`; the rest is filled in from the TB flags.
    pub(crate) fn new(model: &ArmCpuModel) -> DisasContext {
        DisasContext {
            model: model.clone(),
            pc_curr: 0,
            page_start: 0,
            thumb: false,
            condexec_mask: 0,
            condexec_cond: 0,
            condjmp: false,
            condlabel: None,
            be_data: MemOp::LE,
            align_mem: false,
            mmu_idx: 0,
            current_el: 0,
            pstate_il: false,
            svc_imm: 0,
            g: None,
        }
    }
}

/// One instruction being translated: the target context and the generic one together, so the
/// `trans_` hooks can reach both.
struct S<'a, 'b> {
    d: &'a mut DisasContext,
    b: &'a mut DisasContextBase<'b>,
}

/// The data processing operations of [`S::alu`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum Alu {
    And,
    Eor,
    Orr,
    Orn,
    Bic,
    Rsb,
    Adc,
    Sbc,
    Rsc,
    Add,
    Sub,
    Tst,
    Teq,
    Cmn,
    Cmp,
    Mov,
    Mvn,
}

/// The second operand of a data processing instruction.
#[derive(Clone, Copy)]
enum Operand {
    /// `rm` shifted by an immediate: the register, the shift type and the amount.
    Shi(i32, i32, i32),
    /// `rm` shifted by a register: the register, the shift type and `rs`.
    Shr(i32, i32, i32),
    /// A rotated immediate: the value and the rotation.
    Rot(i32, i32),
}

/// The extends of [`S::xta`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum Xt {
    Sb,
    Sh,
    Sb16,
    Ub,
    Uh,
    Ub16,
}

/// `t32_expandimm_imm()`: the T32 modified immediate before its rotation.
fn t32_expandimm_imm(x: i32) -> i32 {
    let imm = x & 0xff;
    match (x >> 8) & 0xf {
        // XY
        0 => imm,
        // 00XY00XY
        1 => imm.wrapping_mul(0x0001_0001),
        // XY00XY00
        2 => imm.wrapping_mul(0x0100_0100),
        // XYXYXYXY
        3 => imm.wrapping_mul(0x0101_0101),
        // Rotated constant.
        _ => imm | 0x80,
    }
}

/// `t32_branch24()`: convert J1:J2 at bits 22:21 to I2:I1 (I = J ^ ~S) and append the final
/// zero.
fn t32_branch24(x: i32) -> i32 {
    let x = if x < 0 { x } else { x ^ (3 << 21) };
    x << 1
}

/// `thumb_insn_is_unconditional()`: BKPT and HLT ignore the IT block state.
fn thumb_insn_is_unconditional(insn: u32) -> bool {
    insn & 0xffff_ff00 == 0xbe00 || insn & 0xffff_ffc0 == 0xba80
}

impl S<'_, '_> {
    fn f(&mut self) -> &mut Func {
        &mut self.b.tb.f
    }

    fn g(&self) -> G {
        self.d.g.expect("globals are made in init_disas_context")
    }

    fn env(&mut self) -> TempPtr {
        self.f().env()
    }

    fn new32(&mut self) -> TempI32 {
        self.f().temp_new_i32()
    }

    fn new64(&mut self) -> TempI64 {
        self.f().temp_new_i64()
    }

    fn c32(&mut self, v: i32) -> TempI32 {
        self.f().constant_i32(v)
    }

    fn c64(&mut self, v: i64) -> TempI64 {
        self.f().constant_i64(v)
    }

    /// Declare and call a helper.
    fn call(&mut self, d: &Def, ret: Option<Temp>, args: &[Temp]) {
        let f = self.f();
        let h = f.helper(d.info());
        f.gen_call(h, ret, args);
    }

    /// `IS_USER()`.
    fn is_user(&self) -> bool {
        self.d.current_el == 0
    }

    /// `curr_insn_len()`.
    fn insn_len(&self) -> i64 {
        (self.b.pc_next - self.d.pc_curr) as i64
    }

    /// `jmp_diff()`: the offset from `pc_curr` of a branch to `read_pc() + diff`.
    fn jmp_diff(&self, diff: i32) -> i64 {
        i64::from(diff) + if self.d.thumb { 4 } else { 8 }
    }

    /// `pc_curr + diff` as an AArch32 address.
    fn pc_plus(&self, diff: i64) -> u32 {
        (self.d.pc_curr as u32).wrapping_add(diff as u32)
    }

    fn ld_env32(&mut self, off: usize) -> TempI32 {
        let t = self.new32();
        let env = self.env();
        self.f().gen_ld_i32(t, env, off as i64);
        t
    }

    fn st_env32(&mut self, t: TempI32, off: usize) {
        let env = self.env();
        self.f().gen_st_i32(t, env, off as i64);
    }

    fn ld_env64(&mut self, off: usize) -> TempI64 {
        let t = self.new64();
        let env = self.env();
        self.f().gen_ld_i64(t, env, off as i64);
        t
    }

    fn st_env64(&mut self, t: TempI64, off: usize) {
        let env = self.env();
        self.f().gen_st_i64(t, env, off as i64);
    }

    // Registers.

    /// `load_reg()`: register `r` in a fresh temp; R15 reads as the PC of this instruction
    /// plus 8, or plus 4 for Thumb.
    fn load_reg(&mut self, r: i32) -> TempI32 {
        let t = self.new32();
        if r == 15 {
            let pc = self.pc_plus(self.jmp_diff(0));
            self.f().gen_movi_i32(t, pc as i32);
        } else {
            let g = self.g().r[r as usize];
            self.f().gen_mov_i32(t, g);
        }
        t
    }

    /// `store_reg()`: a write to R15 is a simple branch, which ignores bit 0 in Thumb state
    /// and bits 1:0 in Arm state.
    fn store_reg(&mut self, r: i32, v: TempI32) {
        let g = self.g();
        if r == 15 {
            let t = self.new32();
            let mask = if self.d.thumb { !1 } else { !3 };
            let f = self.f();
            f.gen_andi_i32(t, v, mask);
            f.gen_extu_i32_i64(g.pc, t);
            self.b.is_jmp = DISAS_JUMP;
        } else {
            self.f().gen_mov_i32(g.r[r as usize], v);
        }
    }

    /// `gen_bx()`: branch to `v`, with bit 0 selecting Thumb state.
    fn gen_bx(&mut self, v: TempI32) {
        let g = self.g();
        let t = self.new32();
        let f = self.f();
        f.gen_andi_i32(t, v, !1);
        f.gen_extu_i32_i64(g.pc, t);
        let thumb = f.temp_new_i32();
        f.gen_andi_i32(thumb, v, 1);
        self.st_env32(thumb, THUMB);
        self.b.is_jmp = DISAS_JUMP;
    }

    /// `store_reg_bx()`: a write to R15 interworks (v7).
    fn store_reg_bx(&mut self, r: i32, v: TempI32) {
        if r == 15 {
            self.gen_bx(v);
        } else {
            self.store_reg(r, v);
        }
    }

    /// `store_reg_from_load()`: a load to R15 interworks (v5).
    fn store_reg_from_load(&mut self, r: i32, v: TempI32) {
        self.store_reg_bx(r, v);
    }

    /// `gen_update_pc()`: write `pc_curr + diff` to the PC.
    fn update_pc(&mut self, diff: i64) {
        let pc = self.pc_plus(diff);
        let g = self.g();
        self.f().gen_movi_i64(g.pc, i64::from(pc));
    }

    /// `add_reg_for_lit()`: register `reg` plus `ofs`, where R15 reads as the aligned PC.
    fn add_reg_for_lit(&mut self, reg: i32, ofs: i32) -> TempI32 {
        let t = self.new32();
        if reg == 15 {
            let pc = self.pc_plus(self.jmp_diff(0)) & !3;
            self.f().gen_movi_i32(t, pc.wrapping_add(ofs as u32) as i32);
        } else {
            let r = self.g().r[reg as usize];
            self.f().gen_addi_i32(t, r, ofs);
        }
        t
    }

    // Memory.

    /// `finalize_memop()`.
    fn finalize_memop(&self, op: MemOp) -> MemOp {
        let mut op = op;
        if self.d.align_mem && op.0 & MemOp::AMASK.0 == 0 {
            op = op | MemOp::ALIGN;
        }
        op | self.d.be_data
    }

    /// `gen_aa32_addr()`: the 32-bit address `a` zero extended.
    fn aa32_addr(&mut self, a: TempI32) -> TempI64 {
        let t = self.new64();
        self.f().gen_extu_i32_i64(t, a);
        t
    }

    /// `gen_aa32_ld_i32()`.
    fn ld32(&mut self, val: TempI32, addr: TempI32, idx: u32, op: MemOp) {
        let op = self.finalize_memop(op);
        let a = self.aa32_addr(addr);
        self.f().gen_qemu_ld_i32(val, a, idx, op);
    }

    /// `gen_aa32_st_i32()`.
    fn st32(&mut self, val: TempI32, addr: TempI32, idx: u32, op: MemOp) {
        let op = self.finalize_memop(op);
        let a = self.aa32_addr(addr);
        self.f().gen_qemu_st_i32(val, a, idx, op);
    }

    // Conditions and exceptions.

    /// `arm_test_cc()`: the comparison against zero that holds when condition `cc` passes.
    fn test_cc(&mut self, cc: u32) -> (Cond, TempI32) {
        let g = self.g();
        let (cond, value) = match cc {
            // eq: Z
            0 | 1 => (Cond::Eq, g.zf),
            // cs: C
            2 | 3 => (Cond::Ne, g.cf),
            // mi: N
            4 | 5 => (Cond::Lt, g.nf),
            // vs: V
            6 | 7 => (Cond::Lt, g.vf),
            8 | 9 => {
                // hi: C && !Z. -CF is an all-bits-set mask for C; ZF is non-zero for !Z.
                let t = self.new32();
                let f = self.f();
                f.gen_neg_i32(t, g.cf);
                f.gen_and_i32(t, t, g.zf);
                (Cond::Ne, t)
            }
            10 | 11 => {
                // ge: N == V, the sign bit of N ^ V is clear.
                let t = self.new32();
                self.f().gen_xor_i32(t, g.vf, g.nf);
                (Cond::Ge, t)
            }
            12 | 13 => {
                // gt: !Z && N == V. N == V is the sign bit of ~(NF ^ VF); propagate it and
                // AND with ZF.
                let t = self.new32();
                let f = self.f();
                f.gen_xor_i32(t, g.vf, g.nf);
                f.gen_sari_i32(t, t, 31);
                f.gen_andc_i32(t, g.zf, t);
                (Cond::Ne, t)
            }
            _ => return (Cond::Always, g.zf),
        };
        if cc & 1 != 0 { (cond.invert(), value) } else { (cond, value) }
    }

    /// `arm_gen_condlabel()`: the label the skip path of this instruction goes to.
    fn condlabel(&mut self) -> Label {
        if let (true, Some(l)) = (self.d.condjmp, self.d.condlabel) {
            return l;
        }
        let l = self.f().new_label();
        self.d.condlabel = Some(l);
        self.d.condjmp = true;
        l
    }

    /// `arm_skip_unless()`: skip the instruction unless condition `cond` passes.
    fn skip_unless(&mut self, cond: u32) {
        let l = self.condlabel();
        let (c, v) = self.test_cc(cond ^ 1);
        self.f().gen_brcondi_i32(c, v, 0, l);
    }

    /// `gen_set_condexec()`: write the IT state of this instruction back to the CPU.
    fn set_condexec(&mut self) {
        if self.d.condexec_mask != 0 {
            let val = (self.d.condexec_cond << 4) | (self.d.condexec_mask >> 1);
            let t = self.c32(val as i32);
            self.st_env32(t, CONDEXEC_BITS);
        }
    }

    /// Raise `excp` to `target_el` through `helper_exception_with_syndrome_el()`.
    fn gen_exception_el(&mut self, excp: i32, syndrome: u32, target_el: u32) {
        let env = self.env();
        let e = self.c32(excp);
        let syn = self.c32(syndrome as i32);
        let el = self.c32(target_el as i32);
        self.call(
            &helpers::EXCEPTION_WITH_SYNDROME_EL,
            None,
            &[env.into(), e.into(), syn.into(), el.into()],
        );
    }

    /// `gen_exception_insn()`: raise `excp` at `pc_curr + diff` to the default EL.
    fn gen_exception_insn(&mut self, diff: i64, excp: i32, syndrome: u32) {
        self.set_condexec();
        self.update_pc(diff);
        let el = self.d.current_el.max(1);
        self.gen_exception_el(excp, syndrome, el);
        self.b.is_jmp = DisasJumpType::NoReturn;
    }

    /// `unallocated_encoding()`.
    fn unallocated_encoding(&mut self) {
        self.gen_exception_insn(0, EXCP_UDEF, syn_uncategorized());
    }

    /// [`S::unallocated_encoding`] for a `trans_` hook that has decoded the instruction.
    fn undef(&mut self) -> bool {
        self.unallocated_encoding();
        true
    }

    /// `gen_goto_tb()`: go to `pc_curr + diff`, chaining when possible.
    fn gen_goto_tb(&mut self, n: u64, diff: i64) {
        let dest = u64::from(self.pc_plus(diff));
        if self.b.translator_use_goto_tb(dest) {
            self.f().gen_goto_tb(n);
            self.update_pc(diff);
            let id = self.b.tb.id;
            self.f().gen_exit_tb(id, n);
        } else {
            self.update_pc(diff);
            self.f().gen_lookup_and_goto_ptr();
        }
        self.b.is_jmp = DisasJumpType::NoReturn;
    }

    /// `gen_jmp()`: branch to `pc_curr + diff`.
    fn gen_jmp(&mut self, diff: i64) {
        match self.b.is_jmp {
            DisasJumpType::Next | DisasJumpType::TooMany | DisasJumpType::NoReturn => {
                self.gen_goto_tb(0, diff);
            }
            _ => {
                // DISAS_UPDATE_EXIT: we are leaving the block for another reason, so don't
                // chain.
                self.update_pc(diff);
                self.f().gen_lookup_and_goto_ptr();
                self.b.is_jmp = DisasJumpType::NoReturn;
            }
        }
    }

    /// `gen_lookup_tb()`: end the block after an instruction that changes the CPU state.
    fn gen_lookup_tb(&mut self) {
        self.update_pc(self.insn_len());
        self.b.is_jmp = DISAS_EXIT;
    }

    // Flags.

    /// `gen_logic_CC()`.
    fn logic_cc(&mut self, v: TempI32) {
        let g = self.g();
        let f = self.f();
        f.gen_mov_i32(g.nf, v);
        f.gen_mov_i32(g.zf, v);
    }

    /// `gen_add_CC()`: `dest = t0 + t1`, setting NZCV.
    fn add_cc(&mut self, dest: TempI32, t0: TempI32, t1: TempI32) {
        let g = self.g();
        let f = self.f();
        let tmp = f.temp_new_i32();
        f.gen_movi_i32(tmp, 0);
        f.gen_add2_i32(g.nf, g.cf, t0, tmp, t1, tmp);
        f.gen_mov_i32(g.zf, g.nf);
        f.gen_xor_i32(g.vf, g.nf, t0);
        f.gen_xor_i32(tmp, t0, t1);
        f.gen_andc_i32(g.vf, g.vf, tmp);
        f.gen_mov_i32(dest, g.nf);
    }

    /// `gen_adc_CC()`: `dest = t0 + t1 + CF`, setting NZCV.
    fn adc_cc(&mut self, dest: TempI32, t0: TempI32, t1: TempI32) {
        let g = self.g();
        let f = self.f();
        let tmp = f.temp_new_i32();
        f.gen_addcio_i32(g.nf, g.cf, t0, t1, g.cf);
        f.gen_mov_i32(g.zf, g.nf);
        f.gen_xor_i32(g.vf, g.nf, t0);
        f.gen_xor_i32(tmp, t0, t1);
        f.gen_andc_i32(g.vf, g.vf, tmp);
        f.gen_mov_i32(dest, g.nf);
    }

    /// `gen_sub_CC()`: `dest = t0 - t1`, setting NZCV.
    fn sub_cc(&mut self, dest: TempI32, t0: TempI32, t1: TempI32) {
        let g = self.g();
        let f = self.f();
        f.gen_sub_i32(g.nf, t0, t1);
        f.gen_mov_i32(g.zf, g.nf);
        f.gen_setcond_i32(Cond::Geu, g.cf, t0, t1);
        f.gen_xor_i32(g.vf, g.nf, t0);
        let tmp = f.temp_new_i32();
        f.gen_xor_i32(tmp, t0, t1);
        f.gen_and_i32(g.vf, g.vf, tmp);
        f.gen_mov_i32(dest, g.nf);
    }

    /// `gen_sbc_CC()`: `dest = t0 + ~t1 + CF`, setting NZCV.
    fn sbc_cc(&mut self, dest: TempI32, t0: TempI32, t1: TempI32) {
        let tmp = self.new32();
        self.f().gen_not_i32(tmp, t1);
        self.adc_cc(dest, t0, tmp);
    }

    /// `gen_add_carry()`: `dest = t0 + t1 + CF`.
    fn add_carry(&mut self, dest: TempI32, t0: TempI32, t1: TempI32) {
        let cf = self.g().cf;
        let f = self.f();
        f.gen_add_i32(dest, t0, t1);
        f.gen_add_i32(dest, dest, cf);
    }

    /// `gen_sub_carry()`: `dest = t0 - t1 + CF - 1`.
    fn sub_carry(&mut self, dest: TempI32, t0: TempI32, t1: TempI32) {
        let cf = self.g().cf;
        let f = self.f();
        f.gen_sub_i32(dest, t0, t1);
        f.gen_add_i32(dest, dest, cf);
        f.gen_subi_i32(dest, dest, 1);
    }

    /// `gen_logicq_cc()`: N and Z from the 64-bit result `hi:lo`.
    fn logicq_cc(&mut self, lo: TempI32, hi: TempI32) {
        let g = self.g();
        let f = self.f();
        f.gen_mov_i32(g.nf, hi);
        f.gen_or_i32(g.zf, lo, hi);
    }

    // Shifts.

    /// `shifter_out_im()`: CF = bit `shift` of `var`.
    fn shifter_out_im(&mut self, var: TempI32, shift: i32) {
        let cf = self.g().cf;
        self.f().gen_extract_i32(cf, var, shift as u32, 1);
    }

    /// `gen_arm_shift_im()`: shift `var` by an immediate, setting CF if `flags`.
    fn shift_im(&mut self, var: TempI32, shiftop: i32, shift: i32, flags: bool) {
        match shiftop {
            // LSL
            0 => {
                if shift != 0 {
                    if flags {
                        self.shifter_out_im(var, 32 - shift);
                    }
                    self.f().gen_shli_i32(var, var, shift);
                }
            }
            // LSR
            1 => {
                if shift == 0 {
                    if flags {
                        let cf = self.g().cf;
                        self.f().gen_shri_i32(cf, var, 31);
                    }
                    self.f().gen_movi_i32(var, 0);
                } else {
                    if flags {
                        self.shifter_out_im(var, shift - 1);
                    }
                    self.f().gen_shri_i32(var, var, shift);
                }
            }
            // ASR
            2 => {
                let shift = if shift == 0 { 32 } else { shift };
                if flags {
                    self.shifter_out_im(var, shift - 1);
                }
                self.f().gen_sari_i32(var, var, shift.min(31));
            }
            // ROR, and RRX for an amount of 0.
            _ => {
                if shift != 0 {
                    if flags {
                        self.shifter_out_im(var, shift - 1);
                    }
                    self.f().gen_rotri_i32(var, var, shift);
                } else {
                    let cf = self.g().cf;
                    let tmp = self.new32();
                    self.f().gen_shli_i32(tmp, cf, 31);
                    if flags {
                        self.shifter_out_im(var, 0);
                    }
                    let f = self.f();
                    f.gen_shri_i32(var, var, 1);
                    f.gen_or_i32(var, var, tmp);
                }
            }
        }
    }

    /// `gen_arm_shift_reg()`: shift `var` by the bottom byte of `shift`, setting CF if
    /// `flags`.
    fn shift_reg(&mut self, var: TempI32, shiftop: i32, shift: TempI32, flags: bool) {
        if flags {
            let d = match shiftop {
                0 => &h::SHL_CC,
                1 => &h::SHR_CC,
                2 => &h::SAR_CC,
                _ => &h::ROR_CC,
            };
            let env = self.env();
            self.call(d, Some(var.into()), &[env.into(), var.into(), shift.into()]);
            return;
        }
        let f = self.f();
        match shiftop {
            0 | 1 => {
                // gen_shl() and gen_shr(): amounts of 32 and up give 0.
                let tmpd = f.temp_new_i32();
                let tmp1 = f.temp_new_i32();
                let zero = f.constant_i32(0);
                f.gen_andi_i32(tmp1, shift, 0x1f);
                if shiftop == 0 {
                    f.gen_shl_i32(tmpd, var, tmp1);
                } else {
                    f.gen_shr_i32(tmpd, var, tmp1);
                }
                f.gen_andi_i32(tmp1, shift, 0xe0);
                f.gen_movcond_i32(Cond::Ne, var, tmp1, zero, zero, tmpd);
            }
            2 => {
                // gen_sar(): amounts of 32 and up shift by 31.
                let tmp1 = f.temp_new_i32();
                let max = f.constant_i32(31);
                f.gen_andi_i32(tmp1, shift, 0xff);
                f.gen_umin_i32(tmp1, tmp1, max);
                f.gen_sar_i32(var, var, tmp1);
            }
            _ => {
                let tmp1 = f.temp_new_i32();
                f.gen_andi_i32(tmp1, shift, 0x1f);
                f.gen_rotr_i32(var, var, tmp1);
            }
        }
    }

    // Data processing.

    /// `op_s_rrr_shi()`, `op_s_rrr_shr()`, `op_s_rri_rot()` and their unary and compare
    /// forms, with QEMU's `StoreRegKind` choice per operation.
    fn alu(&mut self, op: Alu, s: bool, rd: i32, rn: i32, src: Operand) -> bool {
        use Alu::{
            Adc, Add, And, Bic, Cmn, Cmp, Eor, Mov, Mvn, Orn, Orr, Rsb, Rsc, Sbc, Sub, Teq, Tst,
        };

        if matches!(op, Sub | Mov) && rd == 15 && s {
            // ALUExceptionReturn: UNPREDICTABLE in User mode, where we UNDEF. Exception
            // returns from other modes are not implemented.
            return self.undef();
        }
        let logic_cc = match op {
            Tst | Teq => true,
            And | Eor | Orr | Orn | Bic | Mov | Mvn => s,
            _ => false,
        };
        let t2 = match src {
            Operand::Shi(rm, shty, shim) => {
                let t = self.load_reg(rm);
                self.shift_im(t, shty, shim, logic_cc);
                t
            }
            Operand::Shr(rm, shty, rs) => {
                let sh = self.load_reg(rs);
                let t = self.load_reg(rm);
                self.shift_reg(t, shty, sh, logic_cc);
                t
            }
            Operand::Rot(imm, rot) => {
                let imm = (imm as u32).rotate_right(rot as u32);
                if logic_cc && rot != 0 {
                    let cf = self.g().cf;
                    self.f().gen_movi_i32(cf, (imm >> 31) as i32);
                }
                self.c32(imm as i32)
            }
        };
        let t1 = if matches!(op, Mov | Mvn) { t2 } else { self.load_reg(rn) };
        let d = self.new32();
        match op {
            And | Tst => self.f().gen_and_i32(d, t1, t2),
            Eor | Teq => self.f().gen_xor_i32(d, t1, t2),
            Orr => self.f().gen_or_i32(d, t1, t2),
            Orn => self.f().gen_orc_i32(d, t1, t2),
            Bic => self.f().gen_andc_i32(d, t1, t2),
            Mov => self.f().gen_mov_i32(d, t2),
            Mvn => self.f().gen_not_i32(d, t2),
            Add | Cmn if s || op == Cmn => self.add_cc(d, t1, t2),
            Add => self.f().gen_add_i32(d, t1, t2),
            Sub | Cmp if s || op == Cmp => self.sub_cc(d, t1, t2),
            Sub => self.f().gen_sub_i32(d, t1, t2),
            Rsb if s => self.sub_cc(d, t2, t1),
            Rsb => self.f().gen_sub_i32(d, t2, t1),
            Adc if s => self.adc_cc(d, t1, t2),
            Adc => self.add_carry(d, t1, t2),
            Sbc if s => self.sbc_cc(d, t1, t2),
            Sbc => self.sub_carry(d, t1, t2),
            Rsc if s => self.sbc_cc(d, t2, t1),
            Rsc => self.sub_carry(d, t2, t1),
            Cmn | Cmp => unreachable!(),
        }
        if logic_cc {
            self.logic_cc(d);
        }
        if !matches!(op, Tst | Teq | Cmn | Cmp) {
            // STREG_NORMAL (STREG_SP_CHECK is the same outside the M profile): see
            // ALUWritePC, only A32 interworks.
            if self.d.thumb {
                self.store_reg(rd, d);
            } else {
                self.store_reg_bx(rd, d);
            }
        }
        true
    }

    // Multiplies.

    /// `op_mla()`.
    fn mla(&mut self, a: &arg_s_rrrr, add: bool) -> bool {
        let t1 = self.load_reg(a.rn);
        let t2 = self.load_reg(a.rm);
        self.f().gen_mul_i32(t1, t1, t2);
        if add {
            let t2 = self.load_reg(a.ra);
            self.f().gen_add_i32(t1, t1, t2);
        }
        if a.s != 0 {
            self.logic_cc(t1);
        }
        self.store_reg(a.rd, t1);
        true
    }

    /// `op_mlal()`: the long multiplies, with `ra` the low and `rd` the high word.
    fn mlal(&mut self, a: &arg_s_rrrr, uns: bool, add: bool) -> bool {
        let t0 = self.load_reg(a.rm);
        let t1 = self.load_reg(a.rn);
        if uns {
            self.f().gen_mulu2_i32(t0, t1, t0, t1);
        } else {
            self.f().gen_muls2_i32(t0, t1, t0, t1);
        }
        if add {
            let t2 = self.load_reg(a.ra);
            let t3 = self.load_reg(a.rd);
            self.f().gen_add2_i32(t0, t1, t0, t1, t2, t3);
        }
        if a.s != 0 {
            self.logicq_cc(t0, t1);
        }
        self.store_reg(a.ra, t0);
        self.store_reg(a.rd, t1);
        true
    }

    /// `op_qaddsub()`.
    fn qaddsub(&mut self, a: &arg_rrr, add: bool, doub: bool) -> bool {
        let t0 = self.load_reg(a.rm);
        let t1 = self.load_reg(a.rn);
        let env = self.env();
        if doub {
            self.call(&h::ADD_SATURATE, Some(t1.into()), &[env.into(), t1.into(), t1.into()]);
        }
        let d = if add { &h::ADD_SATURATE } else { &h::SUB_SATURATE };
        self.call(d, Some(t0.into()), &[env.into(), t0.into(), t1.into()]);
        self.store_reg(a.rd, t0);
        true
    }

    /// `gen_mulxy()`: the product of the halves of `t0` and `t1` that `x` and `y` select,
    /// in `t0`.
    fn mulxy(&mut self, t0: TempI32, t1: TempI32, x: bool, y: bool) {
        let f = self.f();
        for (t, top) in [(t0, x), (t1, y)] {
            if top {
                f.gen_sari_i32(t, t, 16);
            } else {
                f.gen_ext16s_i32(t, t);
            }
        }
        f.gen_mul_i32(t0, t0, t1);
    }

    /// `op_smlaxxx()`: `add_long` is 0 for SMUL, 1 for SMLA and 2 for SMLAL.
    fn smlaxxx(&mut self, a: &arg_rrrr, add_long: u32, nt: bool, mt: bool) -> bool {
        let t0 = self.load_reg(a.rn);
        let t1 = self.load_reg(a.rm);
        self.mulxy(t0, t1, nt, mt);
        match add_long {
            0 => self.store_reg(a.rd, t0),
            1 => {
                let t1 = self.load_reg(a.ra);
                let env = self.env();
                self.call(&h::ADD_SETQ, Some(t0.into()), &[env.into(), t0.into(), t1.into()]);
                self.store_reg(a.rd, t0);
            }
            _ => {
                let tl = self.load_reg(a.ra);
                let th = self.load_reg(a.rd);
                // Sign-extend the 32-bit product to 64 bits.
                let t1 = self.new32();
                let f = self.f();
                f.gen_sari_i32(t1, t0, 31);
                f.gen_add2_i32(tl, th, tl, th, t0, t1);
                self.store_reg(a.ra, tl);
                self.store_reg(a.rd, th);
            }
        }
        true
    }

    /// `op_smlawx()`.
    fn smlawx(&mut self, a: &arg_rrrr, add: bool, mt: bool) -> bool {
        let t0 = self.load_reg(a.rn);
        let t1 = self.load_reg(a.rm);
        // Since the nominal result is product<47:16>, shift the 16-bit input up by 16 bits,
        // so that the result is at product<63:32>.
        let f = self.f();
        if mt {
            f.gen_andi_i32(t1, t1, 0xffff_0000_u32 as i32);
        } else {
            f.gen_shli_i32(t1, t1, 16);
        }
        f.gen_muls2_i32(t0, t1, t0, t1);
        if add {
            let t0 = self.load_reg(a.ra);
            let env = self.env();
            self.call(&h::ADD_SETQ, Some(t1.into()), &[env.into(), t1.into(), t0.into()]);
        }
        self.store_reg(a.rd, t1);
        true
    }

    /// `gen_smul_dual()`: `a = lo(a) * lo(b)`, `b = hi(a) * hi(b)`, signed.
    fn smul_dual(&mut self, a: TempI32, b: TempI32) {
        let f = self.f();
        let tmp1 = f.temp_new_i32();
        let tmp2 = f.temp_new_i32();
        f.gen_ext16s_i32(tmp1, a);
        f.gen_ext16s_i32(tmp2, b);
        f.gen_mul_i32(tmp1, tmp1, tmp2);
        f.gen_sari_i32(a, a, 16);
        f.gen_sari_i32(b, b, 16);
        f.gen_mul_i32(b, b, a);
        f.gen_mov_i32(a, tmp1);
    }

    /// `op_smlad()`.
    fn smlad(&mut self, a: &arg_rrrr, m_swap: bool, sub: bool) -> bool {
        let t1 = self.load_reg(a.rn);
        let t2 = self.load_reg(a.rm);
        if m_swap {
            self.f().gen_rotri_i32(t2, t2, 16);
        }
        self.smul_dual(t1, t2);
        let env = self.env();
        if sub {
            // This subtraction cannot overflow, so we can do a simple 32-bit subtraction
            // and then a possible 32-bit saturating addition of Ra.
            self.f().gen_sub_i32(t1, t1, t2);
            if a.ra != 15 {
                let t2 = self.load_reg(a.ra);
                self.call(&h::ADD_SETQ, Some(t1.into()), &[env.into(), t1.into(), t2.into()]);
            }
        } else if a.ra == 15 {
            // Single saturation-checking addition.
            self.call(&h::ADD_SETQ, Some(t1.into()), &[env.into(), t1.into(), t2.into()]);
        } else {
            // Add the products and Ra together at 64 bits and then check for overflow:
            // two separate add-and-check steps would set Q for cases like
            // (-32768 * -32768) + (-32768 * -32768) + -1.
            let ra = self.load_reg(a.ra);
            let qf = self.ld_env32(QF);
            let f = self.f();
            let p64 = f.temp_new_i64();
            let q64 = f.temp_new_i64();
            f.gen_ext_i32_i64(p64, t1);
            f.gen_ext_i32_i64(q64, t2);
            f.gen_add_i64(p64, p64, q64);
            f.gen_ext_i32_i64(q64, ra);
            f.gen_add_i64(p64, p64, q64);
            f.gen_extr_i64_i32(t1, t2, p64);
            // t1 is the low half of the result, which goes into Rd. It overflowed and must
            // set Q if the high half (t2) differs from the sign extension of t1.
            let t3 = f.temp_new_i32();
            f.gen_sari_i32(t3, t1, 31);
            let one = f.constant_i32(1);
            f.gen_movcond_i32(Cond::Ne, qf, t2, t3, one, qf);
            self.st_env32(qf, QF);
        }
        self.store_reg(a.rd, t1);
        true
    }

    /// `op_smlald()`.
    fn smlald(&mut self, a: &arg_rrrr, m_swap: bool, sub: bool) -> bool {
        let t1 = self.load_reg(a.rn);
        let t2 = self.load_reg(a.rm);
        if m_swap {
            self.f().gen_rotri_i32(t2, t2, 16);
        }
        self.smul_dual(t1, t2);
        let lo = self.load_reg(a.ra);
        let hi = self.load_reg(a.rd);
        let f = self.f();
        let l1 = f.temp_new_i64();
        let l2 = f.temp_new_i64();
        f.gen_ext_i32_i64(l1, t1);
        f.gen_ext_i32_i64(l2, t2);
        if sub {
            f.gen_sub_i64(l1, l1, l2);
        } else {
            f.gen_add_i64(l1, l1, l2);
        }
        // gen_addq() and gen_storeq_reg().
        f.gen_concat_i32_i64(l2, lo, hi);
        f.gen_add_i64(l1, l1, l2);
        f.gen_extr_i64_i32(lo, hi, l1);
        self.store_reg(a.ra, lo);
        self.store_reg(a.rd, hi);
        true
    }

    /// `op_smmla()`.
    fn smmla(&mut self, a: &arg_rrrr, round: bool, sub: bool) -> bool {
        let t1 = self.load_reg(a.rn);
        let t2 = self.load_reg(a.rm);
        self.f().gen_muls2_i32(t2, t1, t1, t2);
        if a.ra != 15 {
            let t3 = self.load_reg(a.ra);
            let f = self.f();
            if sub {
                // For SMMLS, we need a 64-bit subtract. Borrow caused by a non-zero
                // multiplicand lowpart, and the correct result lowpart for rounding.
                let zero = f.constant_i32(0);
                f.gen_sub2_i32(t2, t1, zero, t3, t2, t1);
            } else {
                f.gen_add_i32(t1, t1, t3);
            }
        }
        if round {
            // Adding 0x80000000 to the 64-bit quantity means that we have carry in to the
            // high word when the low word has the msb set.
            let f = self.f();
            f.gen_shri_i32(t2, t2, 31);
            f.gen_add_i32(t1, t1, t2);
        }
        self.store_reg(a.rd, t1);
        true
    }

    /// `op_div()`.
    fn div(&mut self, a: &arg_rrr, d: &Def) -> bool {
        let t1 = self.load_reg(a.rn);
        let t2 = self.load_reg(a.rm);
        let env = self.env();
        self.call(d, Some(t1.into()), &[env.into(), t1.into(), t2.into()]);
        self.store_reg(a.rd, t1);
        true
    }

    // Media instructions.

    /// `op_crc32()` for `1 << sz` bytes.
    fn crc32(&mut self, a: &arg_rrr, c: bool, sz: u32) -> bool {
        let t1 = self.load_reg(a.rn);
        let t2 = self.load_reg(a.rm);
        match sz {
            0 => self.f().gen_ext8u_i32(t2, t2),
            1 => self.f().gen_ext16u_i32(t2, t2),
            _ => {}
        }
        let t3 = self.c32(1 << sz);
        let d = if c { &h::CRC32C } else { &h::CRC32 };
        self.call(d, Some(t1.into()), &[t1.into(), t2.into(), t3.into()]);
        self.store_reg(a.rd, t1);
        true
    }

    /// `op_par_addsub()` and `op_par_addsub_ge()`: `ge` helpers take `env` and write GE.
    fn par_addsub(&mut self, a: &arg_rrr, d: &Def, ge: bool) -> bool {
        let t0 = self.load_reg(a.rn);
        let t1 = self.load_reg(a.rm);
        if ge {
            let env = self.env();
            self.call(d, Some(t0.into()), &[env.into(), t0.into(), t1.into()]);
        } else {
            self.call(d, Some(t0.into()), &[t0.into(), t1.into()]);
        }
        self.store_reg(a.rd, t0);
        true
    }

    /// `op_sat()`.
    fn sat(&mut self, a: &arg_sat, d: &Def) -> bool {
        let t = self.load_reg(a.rn);
        if a.sh != 0 {
            self.f().gen_sari_i32(t, t, if a.imm != 0 { a.imm } else { 31 });
        } else {
            self.f().gen_shli_i32(t, t, a.imm);
        }
        let env = self.env();
        let sat = self.c32(a.satimm);
        self.call(d, Some(t.into()), &[env.into(), t.into(), sat.into()]);
        self.store_reg(a.rd, t);
        true
    }

    /// `gen_add16()`: add the halfwords of `t0` and `t1` separately.
    fn add16(&mut self, d: TempI32, t0: TempI32, t1: TempI32) {
        let f = self.f();
        let tmp = f.temp_new_i32();
        f.gen_xor_i32(tmp, t0, t1);
        f.gen_andi_i32(tmp, tmp, 0x8000);
        f.gen_andi_i32(t0, t0, !0x8000);
        f.gen_andi_i32(t1, t1, !0x8000);
        f.gen_add_i32(t0, t0, t1);
        f.gen_xor_i32(d, t0, tmp);
    }

    /// `op_xta()`: extend the rotated `rm` and add `rn` unless it is 15.
    fn xta(&mut self, a: &arg_rrr_rot, op: Xt) -> bool {
        let t = self.load_reg(a.rm);
        self.f().gen_rotri_i32(t, t, a.rot * 8);
        match op {
            Xt::Sb => self.f().gen_ext8s_i32(t, t),
            Xt::Sh => self.f().gen_ext16s_i32(t, t),
            Xt::Ub => self.f().gen_ext8u_i32(t, t),
            Xt::Uh => self.f().gen_ext16u_i32(t, t),
            Xt::Sb16 => self.call(&h::SXTB16, Some(t.into()), &[t.into()]),
            Xt::Ub16 => self.call(&h::UXTB16, Some(t.into()), &[t.into()]),
        }
        if a.rn != 15 {
            let t2 = self.load_reg(a.rn);
            if matches!(op, Xt::Sb16 | Xt::Ub16) {
                self.add16(t, t, t2);
            } else {
                self.f().gen_add_i32(t, t, t2);
            }
        }
        self.store_reg(a.rd, t);
        true
    }

    /// `op_rr()` for REV, REV16, REVSH and RBIT.
    fn rev(&mut self, a: &arg_rr, kind: u32) -> bool {
        let t = self.load_reg(a.rm);
        match kind {
            0 => self.f().gen_bswap32_i32(t, t),
            1 => {
                // gen_rev16()
                let f = self.f();
                let tmp = f.temp_new_i32();
                let mask = f.constant_i32(0x00ff_00ff);
                f.gen_shri_i32(tmp, t, 8);
                f.gen_and_i32(tmp, tmp, mask);
                f.gen_and_i32(t, t, mask);
                f.gen_shli_i32(t, t, 8);
                f.gen_or_i32(t, t, tmp);
            }
            2 => self.f().gen_bswap16_i32(t, t, bswap::OS),
            _ => self.call(&h::RBIT, Some(t.into()), &[t.into()]),
        }
        self.store_reg(a.rd, t);
        true
    }

    // Loads and stores.

    /// The register offset of `a` applied to `addr`.
    fn rr_offset(&mut self, a: &arg_ldst_rr, addr: TempI32) {
        let ofs = self.load_reg(a.rm);
        self.shift_im(ofs, a.shtype, a.shimm, false);
        if a.u != 0 {
            self.f().gen_add_i32(addr, addr, ofs);
        } else {
            self.f().gen_sub_i32(addr, addr, ofs);
        }
    }

    /// `op_addr_rr_pre()`.
    fn addr_rr_pre(&mut self, a: &arg_ldst_rr) -> TempI32 {
        let addr = self.load_reg(a.rn);
        if a.p != 0 {
            self.rr_offset(a, addr);
        }
        addr
    }

    /// `op_addr_rr_post()`.
    fn addr_rr_post(&mut self, a: &arg_ldst_rr, addr: TempI32) {
        if a.p == 0 {
            self.rr_offset(a, addr);
        } else if a.w == 0 {
            return;
        }
        self.store_reg(a.rn, addr);
    }

    /// `op_load_rr()` and `op_store_rr()`.
    fn ldst_rr(&mut self, a: &arg_ldst_rr, mop: MemOp, store: bool) -> bool {
        // In Thumb encodings of stores Rn=1111 is UNDEF; for Arm it is either UNPREDICTABLE
        // or has defined behaviour.
        if store && self.d.thumb && a.rn == 15 {
            return false;
        }
        let idx = self.d.mmu_idx;
        let addr = self.addr_rr_pre(a);
        if store {
            let t = self.load_reg(a.rt);
            self.st32(t, addr, idx, mop);
            self.addr_rr_post(a, addr);
        } else {
            let t = self.new32();
            self.ld32(t, addr, idx, mop);
            // Perform base writeback before the loaded value to ensure correct behavior
            // with overlapping index registers.
            self.addr_rr_post(a, addr);
            self.store_reg_from_load(a.rt, t);
        }
        true
    }

    /// `op_addr_ri_pre()`.
    fn addr_ri_pre(&mut self, a: &arg_ldst_ri) -> TempI32 {
        let ofs = if a.u != 0 { a.imm } else { a.imm.wrapping_neg() };
        self.add_reg_for_lit(a.rn, if a.p != 0 { ofs } else { 0 })
    }

    /// `op_addr_ri_post()`.
    fn addr_ri_post(&mut self, a: &arg_ldst_ri, addr: TempI32) {
        let ofs = if a.p == 0 {
            if a.u != 0 { a.imm } else { a.imm.wrapping_neg() }
        } else if a.w == 0 {
            return;
        } else {
            0
        };
        self.f().gen_addi_i32(addr, addr, ofs);
        self.store_reg(a.rn, addr);
    }

    /// `op_load_ri()` and `op_store_ri()`.
    fn ldst_ri(&mut self, a: &arg_ldst_ri, mop: MemOp, store: bool) -> bool {
        if store && self.d.thumb && a.rn == 15 {
            return false;
        }
        let idx = self.d.mmu_idx;
        let addr = self.addr_ri_pre(a);
        if store {
            let t = self.load_reg(a.rt);
            self.st32(t, addr, idx, mop);
            self.addr_ri_post(a, addr);
        } else {
            let t = self.new32();
            self.ld32(t, addr, idx, mop);
            self.addr_ri_post(a, addr);
            self.store_reg_from_load(a.rt, t);
        }
        true
    }

    /// The MemOp of LDRD and STRD: a 64-bit access that is two atomic 32-bit ones if only
    /// 4-aligned.
    fn ldrd_memop(&self) -> MemOp {
        MemOp::MO_64 | MemOp::ALIGN_4 | MemOp::ATOM_SUBALIGN | self.d.be_data
    }

    /// `do_ldrd_load()`: Rt is always the word from the lower address, regardless of
    /// endianness, and Rt is not written if the load of Rt2 faults.
    fn ldrd_load(&mut self, addr: TempI32, rt: i32, rt2: i32) {
        let opc = self.ldrd_memop();
        let idx = self.d.mmu_idx;
        let taddr = self.aa32_addr(addr);
        let be = self.d.be_data == MemOp::BE;
        let f = self.f();
        let t64 = f.temp_new_i64();
        let t = f.temp_new_i32();
        let t2 = f.temp_new_i32();
        f.gen_qemu_ld_i64(t64, taddr, idx, opc);
        if be {
            f.gen_extr_i64_i32(t2, t, t64);
        } else {
            f.gen_extr_i64_i32(t, t2, t64);
        }
        self.store_reg(rt, t);
        self.store_reg(rt2, t2);
    }

    /// `do_strd_store()`.
    fn strd_store(&mut self, addr: TempI32, rt: i32, rt2: i32) {
        let opc = self.ldrd_memop();
        let idx = self.d.mmu_idx;
        let taddr = self.aa32_addr(addr);
        let t1 = self.load_reg(rt);
        let t2 = self.load_reg(rt2);
        let be = self.d.be_data == MemOp::BE;
        let f = self.f();
        let t64 = f.temp_new_i64();
        if be {
            f.gen_concat_i32_i64(t64, t2, t1);
        } else {
            f.gen_concat_i32_i64(t64, t1, t2);
        }
        f.gen_qemu_st_i64(t64, taddr, idx, opc);
    }

    /// `trans_LDRD_rr()` and `trans_STRD_rr()`.
    fn ldrd_rr(&mut self, a: &arg_ldst_rr, store: bool) -> bool {
        if a.rt & 1 != 0 {
            return self.undef();
        }
        let addr = self.addr_rr_pre(a);
        if store {
            self.strd_store(addr, a.rt, a.rt + 1);
        } else {
            self.ldrd_load(addr, a.rt, a.rt + 1);
        }
        self.addr_rr_post(a, addr);
        true
    }

    /// `op_ldrd_ri()` and `op_strd_ri()`.
    fn ldrd_ri(&mut self, a: &arg_ldst_ri, rt2: i32, store: bool) -> bool {
        let addr = self.addr_ri_pre(a);
        if store {
            self.strd_store(addr, a.rt, rt2);
        } else {
            self.ldrd_load(addr, a.rt, rt2);
        }
        self.addr_ri_post(a, addr);
        true
    }

    /// `trans_LDRD_ri_t32()` and `trans_STRD_ri_t32()`.
    fn ldrd_ri_t32(&mut self, a: &arg_ldst_ri2, store: bool) -> bool {
        let b = arg_ldst_ri { p: a.p, w: a.w, u: a.u, rn: a.rn, rt: a.rt, imm: a.imm };
        self.ldrd_ri(&b, a.rt2, store)
    }

    /// `gen_load_exclusive()`.
    fn load_exclusive(&mut self, rt: i32, rt2: i32, addr: TempI32, size: u32) {
        let opc = MemOp(size) | MemOp::ALIGN | self.d.be_data;
        let idx = self.d.mmu_idx;
        let t = self.new32();
        let val = self.new64();
        if size == 3 {
            // The word at the lowest address is always Rt and the one at addr+4 is Rt2, even
            // if the CPU is big-endian: do a 64-bit access and split the two halves.
            let taddr = self.aa32_addr(addr);
            let be = self.d.be_data == MemOp::BE;
            let f = self.f();
            let t2 = f.temp_new_i32();
            f.gen_qemu_ld_i64(val, taddr, idx, opc);
            if be {
                f.gen_extr_i64_i32(t2, t, val);
            } else {
                f.gen_extr_i64_i32(t, t2, val);
            }
            self.store_reg(rt2, t2);
        } else {
            self.ld32(t, addr, idx, opc);
            self.f().gen_extu_i32_i64(val, t);
        }
        self.st_env64(val, EXCLUSIVE_VAL);
        self.store_reg(rt, t);
        let ea = self.aa32_addr(addr);
        self.st_env64(ea, EXCLUSIVE_ADDR);
    }

    /// `gen_store_exclusive()`.
    fn store_exclusive(&mut self, rd: i32, rt: i32, rt2: i32, addr: TempI32, size: u32) {
        let opc = MemOp(size) | MemOp::ALIGN | self.d.be_data;
        let idx = self.d.mmu_idx;
        let g = self.g();
        // if (env->exclusive_addr == addr && env->exclusive_val == [addr]) {
        //     [addr] = {Rt};
        //     {Rd} = 0;
        // } else {
        //     {Rd} = 1;
        // }
        let fail_label = self.f().new_label();
        let done_label = self.f().new_label();
        let extaddr = self.aa32_addr(addr);
        let excl_addr = self.ld_env64(EXCLUSIVE_ADDR);
        self.f().gen_brcond_i64(Cond::Ne, extaddr, excl_addr, fail_label);

        let excl_val = self.ld_env64(EXCLUSIVE_VAL);
        let t0 = self.new32();
        let t1 = self.load_reg(rt);
        if size == 3 {
            let t2 = self.load_reg(rt2);
            let be = self.d.be_data == MemOp::BE;
            let f = self.f();
            let o64 = f.temp_new_i64();
            let n64 = f.temp_new_i64();
            // Rt goes to the lower address: as a single 64-bit BE store, put the two halves
            // in the opposite order.
            if be {
                f.gen_concat_i32_i64(n64, t2, t1);
            } else {
                f.gen_concat_i32_i64(n64, t1, t2);
            }
            f.gen_atomic_cmpxchg_i64(o64, extaddr, excl_val, n64, idx, opc);
            f.gen_setcond_i64(Cond::Ne, o64, o64, excl_val);
            f.gen_extrl_i64_i32(t0, o64);
        } else {
            let f = self.f();
            let t2 = f.temp_new_i32();
            f.gen_extrl_i64_i32(t2, excl_val);
            f.gen_atomic_cmpxchg_i32(t0, extaddr, t2, t1, idx, opc);
            f.gen_setcond_i32(Cond::Ne, t0, t0, t2);
        }
        let f = self.f();
        f.gen_mov_i32(g.r[rd as usize], t0);
        f.gen_br(done_label);

        f.gen_set_label(fail_label);
        f.gen_movi_i32(g.r[rd as usize], 1);
        f.gen_set_label(done_label);
        let clear = f.constant_i64(-1);
        self.st_env64(clear, EXCLUSIVE_ADDR);
    }

    /// `op_strex()`.
    fn strex(&mut self, a: &arg_strex, size: u32, rel: bool) -> bool {
        // We UNDEF for these UNPREDICTABLE cases.
        if a.rd == 15
            || a.rn == 15
            || a.rt == 15
            || a.rd == a.rn
            || a.rd == a.rt
            || (size == 3 && (a.rt2 == 15 || a.rd == a.rt2))
        {
            return self.undef();
        }
        if rel {
            self.f().gen_mb(mo::ALL | mo::BAR_STRL);
        }
        let addr = self.load_reg(a.rn);
        self.f().gen_addi_i32(addr, addr, a.imm);
        self.store_exclusive(a.rd, a.rt, a.rt2, addr, size);
        true
    }

    /// `trans_STREXD_a32()` and `trans_STLEXD_a32()`.
    fn strexd_a32(&mut self, a: &arg_strex, rel: bool) -> bool {
        if a.rt & 1 != 0 {
            return self.undef();
        }
        let b = arg_strex { rt2: a.rt + 1, ..*a };
        self.strex(&b, 3, rel)
    }

    /// `op_ldrex()`.
    fn ldrex(&mut self, a: &arg_ldrex, size: u32, acq: bool) -> bool {
        // We UNDEF for these UNPREDICTABLE cases.
        if a.rn == 15 || a.rt == 15 || (size == 3 && (a.rt2 == 15 || a.rt == a.rt2)) {
            return self.undef();
        }
        let addr = self.load_reg(a.rn);
        self.f().gen_addi_i32(addr, addr, a.imm);
        self.load_exclusive(a.rt, a.rt2, addr, size);
        if acq {
            self.f().gen_mb(mo::ALL | mo::BAR_LDAQ);
        }
        true
    }

    /// `trans_LDREXD_a32()` and `trans_LDAEXD_a32()`.
    fn ldrexd_a32(&mut self, a: &arg_ldrex, acq: bool) -> bool {
        if a.rt & 1 != 0 {
            return self.undef();
        }
        let b = arg_ldrex { rt2: a.rt + 1, ..*a };
        self.ldrex(&b, 3, acq)
    }

    /// `op_stl()`.
    fn stl(&mut self, a: &arg_ldrex, mop: MemOp) -> bool {
        // We UNDEF for these UNPREDICTABLE cases.
        if a.rn == 15 || a.rt == 15 {
            return self.undef();
        }
        let addr = self.load_reg(a.rn);
        let t = self.load_reg(a.rt);
        self.f().gen_mb(mo::ALL | mo::BAR_STRL);
        let idx = self.d.mmu_idx;
        self.st32(t, addr, idx, mop | MemOp::ALIGN);
        true
    }

    /// `op_lda()`.
    fn lda(&mut self, a: &arg_ldrex, mop: MemOp) -> bool {
        // We UNDEF for these UNPREDICTABLE cases.
        if a.rn == 15 || a.rt == 15 {
            return self.undef();
        }
        let addr = self.load_reg(a.rn);
        let t = self.new32();
        let idx = self.d.mmu_idx;
        self.ld32(t, addr, idx, mop | MemOp::ALIGN);
        self.store_reg(a.rt, t);
        self.f().gen_mb(mo::ALL | mo::BAR_LDAQ);
        true
    }

    /// `op_addr_block_pre()`.
    fn addr_block_pre(&mut self, a: &arg_ldst_block, n: i32) -> TempI32 {
        let addr = self.load_reg(a.rn);
        let f = self.f();
        if a.b != 0 {
            if a.i != 0 {
                // pre increment
                f.gen_addi_i32(addr, addr, 4);
            } else {
                // pre decrement
                f.gen_addi_i32(addr, addr, -(n * 4));
            }
        } else if a.i == 0 && n != 1 {
            // post decrement
            f.gen_addi_i32(addr, addr, -((n - 1) * 4));
        }
        addr
    }

    /// `op_addr_block_post()`.
    fn addr_block_post(&mut self, a: &arg_ldst_block, addr: TempI32, n: i32) {
        if a.w == 0 {
            return;
        }
        let f = self.f();
        if a.b == 0 {
            if a.i != 0 {
                // post increment
                f.gen_addi_i32(addr, addr, 4);
            } else {
                // post decrement
                f.gen_addi_i32(addr, addr, -(n * 4));
            }
        } else if a.i == 0 && n != 1 {
            // pre decrement
            f.gen_addi_i32(addr, addr, -((n - 1) * 4));
        }
        self.store_reg(a.rn, addr);
    }

    /// `op_stm()`.
    fn stm(&mut self, a: &arg_ldst_block) -> bool {
        // STM (user) is only usable in supervisor mode, and not implemented there.
        if a.u != 0 {
            return self.undef();
        }
        let n = (a.list & 0xffff).count_ones() as i32;
        // UNPREDICTABLE for n < 1 in all encodings, and we choose to UNDEF. In the T32
        // encoding n == 1 is also UNPREDICTABLE, but hardware implements the single register
        // store and some software relies on it.
        if n < 1 || a.rn == 15 {
            return self.undef();
        }
        let addr = self.addr_block_pre(a, n);
        let idx = self.d.mmu_idx;
        let mut j = 0;
        for i in 0..16 {
            if a.list & (1 << i) == 0 {
                continue;
            }
            let t = self.load_reg(i);
            self.st32(t, addr, idx, MemOp::UL | MemOp::ALIGN);
            // No need to add after the last transfer.
            j += 1;
            if j != n {
                self.f().gen_addi_i32(addr, addr, 4);
            }
        }
        self.addr_block_post(a, addr, n);
        true
    }

    /// `do_ldm()`.
    fn ldm(&mut self, a: &arg_ldst_block) -> bool {
        // LDM (user) and LDM (exception return) are only usable in supervisor mode, and not
        // implemented there.
        if a.u != 0 {
            return self.undef();
        }
        let n = (a.list & 0xffff).count_ones() as i32;
        if n < 1 || a.rn == 15 {
            return self.undef();
        }
        let addr = self.addr_block_pre(a, n);
        let idx = self.d.mmu_idx;
        let mut loaded_base = None;
        let mut j = 0;
        for i in 0..16 {
            if a.list & (1 << i) == 0 {
                continue;
            }
            let t = self.new32();
            self.ld32(t, addr, idx, MemOp::UL | MemOp::ALIGN);
            if i == a.rn {
                loaded_base = Some(t);
            } else {
                self.store_reg_from_load(i, t);
            }
            // No need to add after the last transfer.
            j += 1;
            if j != n {
                self.f().gen_addi_i32(addr, addr, 4);
            }
        }
        self.addr_block_post(a, addr, n);
        if let Some(t) = loaded_base {
            // Note that we reject base == pc above.
            self.store_reg(a.rn, t);
        }
        true
    }

    // Branches and system instructions.

    /// `gen_set_cpsr()`: write the bits of `mask` from `t` through `helper_cpsr_write()`.
    fn set_cpsr(&mut self, t: TempI32, mask: u32) {
        let env = self.env();
        let m = self.c32(mask as i32);
        self.call(&h::CPSR_WRITE, None, &[env.into(), t.into(), m.into()]);
    }

    /// `msr_mask()`.
    fn msr_mask(&self, flags: i32, spsr: bool) -> u32 {
        let mut mask = 0;
        for i in 0..4 {
            if flags & (1 << i) != 0 {
                mask |= 0xff << (8 * i);
            }
        }
        // Mask out execution state.
        if !spsr {
            mask &= !CPSR_EXEC;
        }
        // Mask out privileged bits.
        if self.is_user() {
            mask &= CPSR_USER;
        }
        mask
    }

    /// `gen_set_psr()`, which is false where QEMU's is nonzero: access to the SPSR, which
    /// is UNDEF in User mode and not implemented in the other modes.
    fn set_psr(&mut self, mask: u32, spsr: bool, t: TempI32) -> bool {
        if spsr {
            return false;
        }
        self.set_cpsr(t, mask);
        self.gen_lookup_tb();
        true
    }

    /// The cp15 registers of `do_coproc_insn()` that user mode needs.
    fn coproc(&mut self, a: &arg_mcr, isread: bool) -> bool {
        // TPIDRURW and TPIDRURO are the low halves of TPIDR_EL0 and TPIDRRO_EL0.
        let tpidr = env_off(std::mem::offset_of!(CpuArmState, tpidr_el));
        let tpidrro = env_off(std::mem::offset_of!(CpuArmState, tpidrro_el0));
        if a.cp != 15 {
            return false;
        }
        match (a.opc1, a.crn, a.crm, a.opc2) {
            (0, 13, 0, 2 | 3) => {
                let ro = a.opc2 == 3;
                let off = if ro { tpidrro } else { tpidr };
                if isread {
                    let v = self.ld_env64(off);
                    let t = self.new32();
                    self.f().gen_extrl_i64_i32(t, v);
                    if a.rt == 15 {
                        // A read to R15 sets NZCV from the top four bits.
                        self.set_cpsr(t, CPSR_NZCV);
                    } else {
                        self.store_reg(a.rt, t);
                    }
                    return true;
                }
                if ro && self.is_user() {
                    return self.undef();
                }
                let t = self.load_reg(a.rt);
                let v = self.new64();
                self.f().gen_extu_i32_i64(v, t);
                self.st_env64(v, off);
            }
            // CP15DSB and CP15DMB: write only.
            (0, 7, 10, 4 | 5) if !isread => {
                self.f().gen_mb(mo::ALL | mo::BAR_SC);
                return true;
            }
            // CP15ISB: write only.
            (0, 7, 5, 4) if !isread => {}
            _ => return self.undef(),
        }
        // We end the block on a coprocessor register write.
        self.gen_lookup_tb();
        true
    }
}

/// The data processing `trans_` hooks: the shifted register, register shifted register and
/// immediate forms of each operation.
macro_rules! dp {
    ($($rrri:ident, $rrrr:ident, $rri:ident => $op:ident;)*) => {$(
        fn $rrri(&mut self, a: &mut arg_s_rrr_shi) -> bool {
            self.alu(Alu::$op, a.s != 0, a.rd, a.rn, Operand::Shi(a.rm, a.shty, a.shim))
        }
        fn $rrrr(&mut self, a: &mut arg_s_rrr_shr) -> bool {
            self.alu(Alu::$op, a.s != 0, a.rd, a.rn, Operand::Shr(a.rm, a.shty, a.rs))
        }
        fn $rri(&mut self, a: &mut arg_s_rri_rot) -> bool {
            self.alu(Alu::$op, a.s != 0, a.rd, a.rn, Operand::Rot(a.imm, a.rot))
        }
    )*};
}

/// The load and store `trans_` hooks: the immediate and register offset forms, normal and
/// unprivileged.
macro_rules! ldst {
    ($($ri:ident, $rr:ident, $rit:ident, $rrt:ident => $mop:expr, $store:expr;)*) => {$(
        fn $ri(&mut self, a: &mut arg_ldst_ri) -> bool {
            self.ldst_ri(a, $mop, $store)
        }
        fn $rr(&mut self, a: &mut arg_ldst_rr) -> bool {
            self.ldst_rr(a, $mop, $store)
        }
        fn $rit(&mut self, a: &mut arg_ldst_ri) -> bool {
            self.ldst_ri(a, $mop, $store)
        }
        fn $rrt(&mut self, a: &mut arg_ldst_rr) -> bool {
            self.ldst_rr(a, $mop, $store)
        }
    )*};
}

/// The parallel add and subtract `trans_` hooks; `true` for those that set GE.
macro_rules! par {
    ($($name:ident => $h:ident, $ge:expr;)*) => {$(
        fn $name(&mut self, a: &mut arg_rrr) -> bool {
            self.par_addsub(a, &h::$h, $ge)
        }
    )*};
}

/// `trans_` hooks that call one of the methods of [`S`] with fixed arguments.
macro_rules! op {
    ($($name:ident($arg:ty) => $f:ident($($x:expr),*);)*) => {$(
        fn $name(&mut self, a: &mut $arg) -> bool {
            self.$f(a, $($x),*)
        }
    )*};
}

/// Hints that are NOPs here.
macro_rules! nop {
    ($($name:ident;)*) => {$(
        fn $name(&mut self, _a: &mut arg_empty) -> bool {
            true
        }
    )*};
}

impl DisasA32 for S<'_, '_> {
    fn times_2(&mut self, x: i32) -> i32 {
        x * 2
    }

    fn times_4(&mut self, x: i32) -> i32 {
        x * 4
    }

    dp! {
        trans_AND_rrri, trans_AND_rrrr, trans_AND_rri => And;
        trans_EOR_rrri, trans_EOR_rrrr, trans_EOR_rri => Eor;
        trans_SUB_rrri, trans_SUB_rrrr, trans_SUB_rri => Sub;
        trans_RSB_rrri, trans_RSB_rrrr, trans_RSB_rri => Rsb;
        trans_ADD_rrri, trans_ADD_rrrr, trans_ADD_rri => Add;
        trans_ADC_rrri, trans_ADC_rrrr, trans_ADC_rri => Adc;
        trans_SBC_rrri, trans_SBC_rrrr, trans_SBC_rri => Sbc;
        trans_RSC_rrri, trans_RSC_rrrr, trans_RSC_rri => Rsc;
        trans_TST_xrri, trans_TST_xrrr, trans_TST_xri => Tst;
        trans_TEQ_xrri, trans_TEQ_xrrr, trans_TEQ_xri => Teq;
        trans_CMP_xrri, trans_CMP_xrrr, trans_CMP_xri => Cmp;
        trans_CMN_xrri, trans_CMN_xrrr, trans_CMN_xri => Cmn;
        trans_ORR_rrri, trans_ORR_rrrr, trans_ORR_rri => Orr;
        trans_MOV_rxri, trans_MOV_rxrr, trans_MOV_rxi => Mov;
        trans_BIC_rrri, trans_BIC_rrrr, trans_BIC_rri => Bic;
        trans_MVN_rxri, trans_MVN_rxrr, trans_MVN_rxi => Mvn;
    }

    fn trans_MOVW(&mut self, a: &mut arg_ri) -> bool {
        let t = self.c32(a.imm);
        self.store_reg(a.rd, t);
        true
    }

    fn trans_MOVT(&mut self, a: &mut arg_ri) -> bool {
        let t = self.load_reg(a.rd);
        let f = self.f();
        f.gen_ext16u_i32(t, t);
        f.gen_ori_i32(t, t, a.imm << 16);
        self.store_reg(a.rd, t);
        true
    }

    op! {
        trans_MUL(arg_s_rrrr) => mla(false);
        trans_MLA(arg_s_rrrr) => mla(true);
        trans_UMULL(arg_s_rrrr) => mlal(true, false);
        trans_SMULL(arg_s_rrrr) => mlal(false, false);
        trans_UMLAL(arg_s_rrrr) => mlal(true, true);
        trans_SMLAL(arg_s_rrrr) => mlal(false, true);
        trans_QADD(arg_rrr) => qaddsub(true, false);
        trans_QSUB(arg_rrr) => qaddsub(false, false);
        trans_QDADD(arg_rrr) => qaddsub(true, true);
        trans_QDSUB(arg_rrr) => qaddsub(false, true);
        trans_SMULBB(arg_rrrr) => smlaxxx(0, false, false);
        trans_SMULBT(arg_rrrr) => smlaxxx(0, false, true);
        trans_SMULTB(arg_rrrr) => smlaxxx(0, true, false);
        trans_SMULTT(arg_rrrr) => smlaxxx(0, true, true);
        trans_SMLABB(arg_rrrr) => smlaxxx(1, false, false);
        trans_SMLABT(arg_rrrr) => smlaxxx(1, false, true);
        trans_SMLATB(arg_rrrr) => smlaxxx(1, true, false);
        trans_SMLATT(arg_rrrr) => smlaxxx(1, true, true);
        trans_SMLALBB(arg_rrrr) => smlaxxx(2, false, false);
        trans_SMLALBT(arg_rrrr) => smlaxxx(2, false, true);
        trans_SMLALTB(arg_rrrr) => smlaxxx(2, true, false);
        trans_SMLALTT(arg_rrrr) => smlaxxx(2, true, true);
        trans_SMULWB(arg_rrrr) => smlawx(false, false);
        trans_SMLAWB(arg_rrrr) => smlawx(true, false);
        trans_SMULWT(arg_rrrr) => smlawx(false, true);
        trans_SMLAWT(arg_rrrr) => smlawx(true, true);
        trans_SMLAD(arg_rrrr) => smlad(false, false);
        trans_SMLADX(arg_rrrr) => smlad(true, false);
        trans_SMLSD(arg_rrrr) => smlad(false, true);
        trans_SMLSDX(arg_rrrr) => smlad(true, true);
        trans_SMLALD(arg_rrrr) => smlald(false, false);
        trans_SMLALDX(arg_rrrr) => smlald(true, false);
        trans_SMLSLD(arg_rrrr) => smlald(false, true);
        trans_SMLSLDX(arg_rrrr) => smlald(true, true);
        trans_SMMLA(arg_rrrr) => smmla(false, false);
        trans_SMMLAR(arg_rrrr) => smmla(true, false);
        trans_SMMLS(arg_rrrr) => smmla(false, true);
        trans_SMMLSR(arg_rrrr) => smmla(true, true);
        trans_CRC32B(arg_rrr) => crc32(false, 0);
        trans_CRC32H(arg_rrr) => crc32(false, 1);
        trans_CRC32W(arg_rrr) => crc32(false, 2);
        trans_CRC32CB(arg_rrr) => crc32(true, 0);
        trans_CRC32CH(arg_rrr) => crc32(true, 1);
        trans_CRC32CW(arg_rrr) => crc32(true, 2);
        trans_SXTAB(arg_rrr_rot) => xta(Xt::Sb);
        trans_SXTAH(arg_rrr_rot) => xta(Xt::Sh);
        trans_SXTAB16(arg_rrr_rot) => xta(Xt::Sb16);
        trans_UXTAB(arg_rrr_rot) => xta(Xt::Ub);
        trans_UXTAH(arg_rrr_rot) => xta(Xt::Uh);
        trans_UXTAB16(arg_rrr_rot) => xta(Xt::Ub16);
        trans_REV(arg_rr) => rev(0);
        trans_REV16(arg_rr) => rev(1);
        trans_REVSH(arg_rr) => rev(2);
        trans_RBIT(arg_rr) => rev(3);
        trans_SSAT(arg_sat) => sat(&h::SSAT);
        trans_USAT(arg_sat) => sat(&h::USAT);
        trans_SSAT16(arg_sat) => sat(&h::SSAT16);
        trans_USAT16(arg_sat) => sat(&h::USAT16);
        trans_SDIV(arg_rrr) => div(&h::SDIV);
        trans_UDIV(arg_rrr) => div(&h::UDIV);
        trans_LDRD_rr(arg_ldst_rr) => ldrd_rr(false);
        trans_STRD_rr(arg_ldst_rr) => ldrd_rr(true);
        trans_STREX(arg_strex) => strex(2, false);
        trans_STREXB(arg_strex) => strex(0, false);
        trans_STREXH(arg_strex) => strex(1, false);
        trans_STLEX(arg_strex) => strex(2, true);
        trans_STLEXB(arg_strex) => strex(0, true);
        trans_STLEXH(arg_strex) => strex(1, true);
        trans_STREXD_a32(arg_strex) => strexd_a32(false);
        trans_STLEXD_a32(arg_strex) => strexd_a32(true);
        trans_LDREX(arg_ldrex) => ldrex(2, false);
        trans_LDREXB(arg_ldrex) => ldrex(0, false);
        trans_LDREXH(arg_ldrex) => ldrex(1, false);
        trans_LDAEX(arg_ldrex) => ldrex(2, true);
        trans_LDAEXB(arg_ldrex) => ldrex(0, true);
        trans_LDAEXH(arg_ldrex) => ldrex(1, true);
        trans_LDREXD_a32(arg_ldrex) => ldrexd_a32(false);
        trans_LDAEXD_a32(arg_ldrex) => ldrexd_a32(true);
        trans_STL(arg_ldrex) => stl(MemOp::UL);
        trans_STLB(arg_ldrex) => stl(MemOp::UB);
        trans_STLH(arg_ldrex) => stl(MemOp::UW);
        trans_LDA(arg_ldrex) => lda(MemOp::UL);
        trans_LDAB(arg_ldrex) => lda(MemOp::UB);
        trans_LDAH(arg_ldrex) => lda(MemOp::UW);
    }

    fn trans_MLS(&mut self, a: &mut arg_rrrr) -> bool {
        let t1 = self.load_reg(a.rn);
        let t2 = self.load_reg(a.rm);
        self.f().gen_mul_i32(t1, t1, t2);
        let t2 = self.load_reg(a.ra);
        self.f().gen_sub_i32(t1, t2, t1);
        self.store_reg(a.rd, t1);
        true
    }

    fn trans_UMAAL(&mut self, a: &mut arg_rrrr) -> bool {
        let t0 = self.load_reg(a.rm);
        let t1 = self.load_reg(a.rn);
        self.f().gen_mulu2_i32(t0, t1, t0, t1);
        let zero = self.c32(0);
        let t2 = self.load_reg(a.ra);
        self.f().gen_add2_i32(t0, t1, t0, t1, t2, zero);
        let t2 = self.load_reg(a.rd);
        self.f().gen_add2_i32(t0, t1, t0, t1, t2, zero);
        self.store_reg(a.ra, t0);
        self.store_reg(a.rd, t1);
        true
    }

    nop! {
        trans_YIELD;
        trans_WFE;
        trans_WFI;
        trans_SEV;
        trans_SEVL;
        trans_ESB;
        trans_NOP;
    }

    fn trans_MSR_imm(&mut self, a: &mut arg_msr_i) -> bool {
        let val = (a.imm as u32).rotate_right(a.rot as u32);
        let mask = self.msr_mask(a.mask, a.r != 0);
        let t = self.c32(val as i32);
        if !self.set_psr(mask, a.r != 0, t) {
            self.unallocated_encoding();
        }
        true
    }

    fn trans_MSR_reg(&mut self, a: &mut arg_msr_reg) -> bool {
        let t = self.load_reg(a.rn);
        let mask = self.msr_mask(a.mask, a.r != 0);
        if !self.set_psr(mask, a.r != 0, t) {
            self.unallocated_encoding();
        }
        true
    }

    fn trans_MRS_reg(&mut self, a: &mut arg_mrs_reg) -> bool {
        if a.r != 0 {
            // The SPSR: UNDEF in User mode, not implemented in the others.
            return self.undef();
        }
        let t = self.new32();
        let env = self.env();
        self.call(&h::CPSR_READ, Some(t.into()), &[env.into()]);
        self.store_reg(a.rd, t);
        true
    }

    fn trans_BX(&mut self, a: &mut arg_r) -> bool {
        let t = self.load_reg(a.rm);
        self.gen_bx(t);
        true
    }

    fn trans_BXJ(&mut self, a: &mut arg_r) -> bool {
        // Trivial implementation equivalent to bx.
        let t = self.load_reg(a.rm);
        self.gen_bx(t);
        true
    }

    fn trans_BLX_r(&mut self, a: &mut arg_r) -> bool {
        let t = self.load_reg(a.rm);
        let lr = self.pc_plus(self.insn_len() | i64::from(self.d.thumb));
        let lr = self.c32(lr as i32);
        self.store_reg(14, lr);
        self.gen_bx(t);
        true
    }

    fn trans_CLZ(&mut self, a: &mut arg_rr) -> bool {
        let t = self.load_reg(a.rm);
        self.f().gen_clzi_i32(t, t, 32);
        self.store_reg(a.rd, t);
        true
    }

    fn trans_BKPT(&mut self, a: &mut arg_i) -> bool {
        // gen_exception_bkpt_insn(). FAR is UNKNOWN, so helper_exception_bkpt_insn() clears
        // vaddress rather than leak a stale one.
        let zero = self.c64(0);
        let off = env_off(std::mem::offset_of!(CpuArmState, exception_vaddress));
        self.st_env64(zero, off);
        self.gen_exception_insn(0, EXCP_BKPT, syn_aa32_bkpt(a.imm as u32, false));
        true
    }

    ldst! {
        trans_LDR_ri, trans_LDR_rr, trans_LDRT_ri, trans_LDRT_rr => MemOp::UL, false;
        trans_LDRB_ri, trans_LDRB_rr, trans_LDRBT_ri, trans_LDRBT_rr => MemOp::UB, false;
        trans_LDRH_ri, trans_LDRH_rr, trans_LDRHT_ri, trans_LDRHT_rr => MemOp::UW, false;
        trans_LDRSB_ri, trans_LDRSB_rr, trans_LDRSBT_ri, trans_LDRSBT_rr => MemOp::SB, false;
        trans_LDRSH_ri, trans_LDRSH_rr, trans_LDRSHT_ri, trans_LDRSHT_rr => MemOp::SW, false;
        trans_STR_ri, trans_STR_rr, trans_STRT_ri, trans_STRT_rr => MemOp::UL, true;
        trans_STRB_ri, trans_STRB_rr, trans_STRBT_ri, trans_STRBT_rr => MemOp::UB, true;
        trans_STRH_ri, trans_STRH_rr, trans_STRHT_ri, trans_STRHT_rr => MemOp::UW, true;
    }

    fn trans_LDRD_ri_a32(&mut self, a: &mut arg_ldst_ri) -> bool {
        if a.rt & 1 != 0 {
            return self.undef();
        }
        self.ldrd_ri(a, a.rt + 1, false)
    }

    fn trans_STRD_ri_a32(&mut self, a: &mut arg_ldst_ri) -> bool {
        if a.rt & 1 != 0 {
            return self.undef();
        }
        self.ldrd_ri(a, a.rt + 1, true)
    }

    fn trans_USADA8(&mut self, a: &mut arg_rrrr) -> bool {
        let t1 = self.load_reg(a.rn);
        let t2 = self.load_reg(a.rm);
        self.call(&h::USAD8, Some(t1.into()), &[t1.into(), t2.into()]);
        if a.ra != 15 {
            let t2 = self.load_reg(a.ra);
            self.f().gen_add_i32(t1, t1, t2);
        }
        self.store_reg(a.rd, t1);
        true
    }

    fn trans_SBFX(&mut self, a: &mut arg_bfx) -> bool {
        let width = a.widthm1 + 1;
        if a.lsb + width > 32 {
            // UNPREDICTABLE; we choose to UNDEF.
            return self.undef();
        }
        let t = self.load_reg(a.rn);
        self.f().gen_sextract_i32(t, t, a.lsb as u32, width as u32);
        self.store_reg(a.rd, t);
        true
    }

    fn trans_UBFX(&mut self, a: &mut arg_bfx) -> bool {
        let width = a.widthm1 + 1;
        if a.lsb + width > 32 {
            // UNPREDICTABLE; we choose to UNDEF.
            return self.undef();
        }
        let t = self.load_reg(a.rn);
        self.f().gen_extract_i32(t, t, a.lsb as u32, width as u32);
        self.store_reg(a.rd, t);
        true
    }

    fn trans_BFCI(&mut self, a: &mut arg_bfi) -> bool {
        if a.msb < a.lsb {
            // UNPREDICTABLE; we choose to UNDEF.
            return self.undef();
        }
        let width = a.msb + 1 - a.lsb;
        // BFC when rn is 15, else BFI.
        let t_in = if a.rn == 15 { self.c32(0) } else { self.load_reg(a.rn) };
        let t_rd = self.load_reg(a.rd);
        self.f().gen_deposit_i32(t_rd, t_rd, t_in, a.lsb as u32, width as u32);
        self.store_reg(a.rd, t_rd);
        true
    }

    fn trans_UDF(&mut self, _a: &mut arg_empty) -> bool {
        self.undef()
    }

    par! {
        trans_SADD16 => SADD16, true;
        trans_SASX => SASX, true;
        trans_SSAX => SSAX, true;
        trans_SSUB16 => SSUB16, true;
        trans_SADD8 => SADD8, true;
        trans_SSUB8 => SSUB8, true;
        trans_UADD16 => UADD16, true;
        trans_UASX => UASX, true;
        trans_USAX => USAX, true;
        trans_USUB16 => USUB16, true;
        trans_UADD8 => UADD8, true;
        trans_USUB8 => USUB8, true;
        trans_QADD16 => QADD16, false;
        trans_QASX => QASX, false;
        trans_QSAX => QSAX, false;
        trans_QSUB16 => QSUB16, false;
        trans_QADD8 => QADD8, false;
        trans_QSUB8 => QSUB8, false;
        trans_SHADD16 => SHADD16, false;
        trans_SHASX => SHASX, false;
        trans_SHSAX => SHSAX, false;
        trans_SHSUB16 => SHSUB16, false;
        trans_SHADD8 => SHADD8, false;
        trans_SHSUB8 => SHSUB8, false;
        trans_UQADD16 => UQADD16, false;
        trans_UQASX => UQASX, false;
        trans_UQSAX => UQSAX, false;
        trans_UQSUB16 => UQSUB16, false;
        trans_UQADD8 => UQADD8, false;
        trans_UQSUB8 => UQSUB8, false;
        trans_UHADD16 => UHADD16, false;
        trans_UHASX => UHASX, false;
        trans_UHSAX => UHSAX, false;
        trans_UHSUB16 => UHSUB16, false;
        trans_UHADD8 => UHADD8, false;
        trans_UHSUB8 => UHSUB8, false;
    }

    fn trans_PKH(&mut self, a: &mut arg_pkh) -> bool {
        let tn = self.load_reg(a.rn);
        let tm = self.load_reg(a.rm);
        let f = self.f();
        if a.tb != 0 {
            // PKHTB
            f.gen_sari_i32(tm, tm, if a.imm == 0 { 31 } else { a.imm });
            f.gen_deposit_i32(tn, tn, tm, 0, 16);
        } else {
            // PKHBT
            f.gen_shli_i32(tm, tm, a.imm);
            f.gen_deposit_i32(tn, tm, tn, 0, 16);
        }
        self.store_reg(a.rd, tn);
        true
    }

    fn trans_SEL(&mut self, a: &mut arg_rrr) -> bool {
        let t1 = self.load_reg(a.rn);
        let t2 = self.load_reg(a.rm);
        let ge = self.ld_env32(GE);
        self.call(&h::SEL_FLAGS, Some(t1.into()), &[ge.into(), t1.into(), t2.into()]);
        self.store_reg(a.rd, t1);
        true
    }

    fn trans_STM(&mut self, a: &mut arg_ldst_block) -> bool {
        self.stm(a)
    }

    fn trans_LDM_a32(&mut self, a: &mut arg_ldst_block) -> bool {
        // Writeback register in register list is UNPREDICTABLE for ArchVersion() >= 7.
        if a.w != 0 && a.list & (1 << a.rn) != 0 {
            return self.undef();
        }
        self.ldm(a)
    }

    fn trans_B(&mut self, a: &mut arg_i) -> bool {
        self.gen_jmp(self.jmp_diff(a.imm));
        true
    }

    fn trans_BL(&mut self, a: &mut arg_i) -> bool {
        let lr = self.pc_plus(self.insn_len() | i64::from(self.d.thumb));
        let lr = self.c32(lr as i32);
        self.store_reg(14, lr);
        self.gen_jmp(self.jmp_diff(a.imm));
        true
    }

    fn trans_MCR(&mut self, a: &mut arg_mcr) -> bool {
        self.coproc(a, false)
    }

    fn trans_MRC(&mut self, a: &mut arg_mcr) -> bool {
        self.coproc(a, true)
    }

    fn trans_SVC(&mut self, a: &mut arg_i) -> bool {
        self.update_pc(self.insn_len());
        self.d.svc_imm = a.imm as u32;
        self.b.is_jmp = DISAS_SWI;
        true
    }
}

impl DisasA32Uncond for S<'_, '_> {
    fn trans_BLX_i(&mut self, a: &mut arg_i) -> bool {
        if self.d.thumb && a.imm & 2 != 0 {
            return false;
        }
        let lr = self.pc_plus(self.insn_len() | i64::from(self.d.thumb));
        let lr = self.c32(lr as i32);
        self.store_reg(14, lr);
        let thumb = self.c32(i32::from(!self.d.thumb));
        self.st_env32(thumb, THUMB);
        // This jump is computed from an aligned PC: subtract off the low bits.
        let diff = self.jmp_diff(a.imm - (self.d.pc_curr & 3) as i32);
        self.gen_jmp(diff);
        true
    }

    fn trans_CPS(&mut self, _a: &mut arg_cps) -> bool {
        // A NOP in User mode; the other modes are not implemented.
        self.is_user()
    }

    fn trans_CLREX(&mut self, _a: &mut arg_empty) -> bool {
        let t = self.c64(-1);
        self.st_env64(t, EXCLUSIVE_ADDR);
        true
    }

    fn trans_DSB(&mut self, _a: &mut arg_empty) -> bool {
        self.f().gen_mb(mo::ALL | mo::BAR_SC);
        true
    }

    fn trans_DMB(&mut self, _a: &mut arg_empty) -> bool {
        self.f().gen_mb(mo::ALL | mo::BAR_SC);
        true
    }

    fn trans_ISB(&mut self, _a: &mut arg_empty) -> bool {
        // End the block to execute self-modifying code correctly and take any pending
        // interrupts immediately.
        self.b.is_jmp = DisasJumpType::TooMany;
        true
    }

    fn trans_SB(&mut self, _a: &mut arg_empty) -> bool {
        // There is no speculation barrier opcode for TCG; MB and end the block instead.
        self.f().gen_mb(mo::ALL | mo::BAR_SC);
        self.b.is_jmp = DisasJumpType::TooMany;
        true
    }

    fn trans_SETEND(&mut self, a: &mut arg_setend) -> bool {
        if (a.E != 0) != (self.d.be_data == MemOp::BE) {
            // helper_setend()
            let t = self.c32(if a.E != 0 { crate::cpu::CPSR_E as i32 } else { 0 });
            self.set_cpsr(t, crate::cpu::CPSR_E);
            self.b.is_jmp = DISAS_UPDATE_EXIT;
        }
        true
    }

    nop! {
        trans_PLD;
        trans_PLDW;
        trans_PLI;
    }
}

impl DisasT32 for S<'_, '_> {
    fn t32_expandimm_imm(&mut self, x: i32) -> i32 {
        t32_expandimm_imm(x)
    }

    fn t32_expandimm_rot(&mut self, x: i32) -> i32 {
        if x & 0xc00 != 0 { (x >> 7) & 0x1f } else { 0 }
    }

    fn t32_branch24(&mut self, x: i32) -> i32 {
        t32_branch24(x)
    }

    fn negate(&mut self, x: i32) -> i32 {
        x.wrapping_neg()
    }

    fn times_2_plus_1(&mut self, x: i32) -> i32 {
        x * 2 + 1
    }

    // ORN is only available with T32, so there is no register-shifted-register form.
    fn trans_ORN_rrri(&mut self, a: &mut arg_s_rrr_shi) -> bool {
        self.alu(Alu::Orn, a.s != 0, a.rd, a.rn, Operand::Shi(a.rm, a.shty, a.shim))
    }

    fn trans_ORN_rri(&mut self, a: &mut arg_s_rri_rot) -> bool {
        self.alu(Alu::Orn, a.s != 0, a.rd, a.rn, Operand::Rot(a.imm, a.rot))
    }

    fn trans_ADR(&mut self, a: &mut arg_ri) -> bool {
        let t = self.add_reg_for_lit(15, a.imm);
        self.store_reg_bx(a.rd, t);
        true
    }

    fn trans_B_cond_thumb(&mut self, a: &mut arg_ci) -> bool {
        // This has cond from encoding, required to be outside IT block.
        if a.cond >= 0xe {
            return false;
        }
        if self.d.condexec_mask != 0 {
            return self.undef();
        }
        self.skip_unless(a.cond as u32);
        self.gen_jmp(self.jmp_diff(a.imm));
        true
    }

    fn trans_LDRD_ri_t32(&mut self, a: &mut arg_ldst_ri2) -> bool {
        self.ldrd_ri_t32(a, false)
    }

    fn trans_STRD_ri_t32(&mut self, a: &mut arg_ldst_ri2) -> bool {
        self.ldrd_ri_t32(a, true)
    }

    fn trans_STREXD_t32(&mut self, a: &mut arg_strex) -> bool {
        self.strex(a, 3, false)
    }

    fn trans_STLEXD_t32(&mut self, a: &mut arg_strex) -> bool {
        self.strex(a, 3, true)
    }

    fn trans_LDREXD_t32(&mut self, a: &mut arg_ldrex) -> bool {
        self.ldrex(a, 3, false)
    }

    fn trans_LDAEXD_t32(&mut self, a: &mut arg_ldrex) -> bool {
        self.ldrex(a, 3, true)
    }

    fn trans_TBB(&mut self, a: &mut arg_tbranch) -> bool {
        self.tbranch(a, false)
    }

    fn trans_TBH(&mut self, a: &mut arg_tbranch) -> bool {
        self.tbranch(a, true)
    }

    fn trans_STM_t32(&mut self, a: &mut arg_ldst_block) -> bool {
        // Writeback register in register list is UNPREDICTABLE for T32.
        if a.w != 0 && a.list & (1 << a.rn) != 0 {
            return self.undef();
        }
        self.stm(a)
    }

    fn trans_LDM_t32(&mut self, a: &mut arg_ldst_block) -> bool {
        // Writeback register in register list is UNPREDICTABLE for T32.
        if a.w != 0 && a.list & (1 << a.rn) != 0 {
            return self.undef();
        }
        self.ldm(a)
    }
}

impl S<'_, '_> {
    /// `op_tbranch()`: TBB and TBH.
    fn tbranch(&mut self, a: &arg_tbranch, half: bool) -> bool {
        let t = self.load_reg(a.rm);
        if half {
            self.f().gen_add_i32(t, t, t);
        }
        let addr = self.load_reg(a.rn);
        self.f().gen_add_i32(addr, addr, t);
        let idx = self.d.mmu_idx;
        self.ld32(t, addr, idx, if half { MemOp::UW } else { MemOp::UB });
        let pc = self.pc_plus(self.jmp_diff(0));
        let f = self.f();
        f.gen_add_i32(t, t, t);
        f.gen_addi_i32(t, t, pc as i32);
        self.store_reg(15, t);
        true
    }
}

impl DisasT16 for S<'_, '_> {
    fn t16_setflags(&mut self) -> i32 {
        i32::from(self.d.condexec_mask == 0)
    }

    fn plus_2(&mut self, x: i32) -> i32 {
        x + 2
    }

    fn t16_push_list(&mut self, x: i32) -> i32 {
        // LR is bit 8 of the encoding.
        (x & 0xff) | ((x & 0x100) << (14 - 8))
    }

    fn t16_pop_list(&mut self, x: i32) -> i32 {
        // PC is bit 8 of the encoding.
        (x & 0xff) | ((x & 0x100) << (15 - 8))
    }

    fn trans_LDM_t16(&mut self, a: &mut arg_ldst_block) -> bool {
        // Writeback is conditional on the base register not being loaded.
        a.w = i32::from(a.list & (1 << a.rn) == 0);
        self.ldm(a)
    }

    fn trans_IT(&mut self, a: &mut arg_disas_t1617) -> bool {
        // No code, just the state. Combinations of firstcond and mask which set up an
        // 0b1111 condition are UNPREDICTABLE; we treat 0b1111 the same as 0b1110, both
        // meaning "execute always".
        self.d.condexec_cond = ((a.cond_mask >> 4) & 0xe) as u32;
        self.d.condexec_mask = (a.cond_mask & 0x1f) as u32;
        true
    }

    fn trans_CBZ(&mut self, a: &mut arg_disas_t1618) -> bool {
        let t = self.load_reg(a.rn);
        let l = self.condlabel();
        let cond = if a.nz != 0 { Cond::Eq } else { Cond::Ne };
        self.f().gen_brcondi_i32(cond, t, 0, l);
        self.gen_jmp(self.jmp_diff(a.imm));
        true
    }
}

impl S<'_, '_> {
    /// `arm_check_kernelpage()`: in user mode, a jump to the kernel helper page raises
    /// `EXCP_KERNEL_TRAP`.
    fn check_kernelpage(&mut self) -> bool {
        if !self.d.model.features.user_only || self.b.pc_next < 0xffff_0000 {
            return false;
        }
        // We always get here through a jump, so are not in an IT block.
        self.d.pc_curr = self.b.pc_next;
        self.update_pc(0);
        let env = self.env();
        let e = self.c32(EXCP_KERNEL_TRAP);
        self.call(&helpers::EXCEPTION_INTERNAL, None, &[env.into(), e.into()]);
        self.b.is_jmp = DisasJumpType::NoReturn;
        true
    }

    /// `disas_arm_insn()`.
    fn disas_arm_insn(&mut self, insn: u32) {
        let cond = insn >> 28;
        if self.d.pstate_il {
            // Illegal execution state. This has priority over BTI exceptions, but comes
            // after instruction abort exceptions.
            self.gen_exception_insn(0, EXCP_UDEF, syn_illegalstate());
            return;
        }
        if cond == 0xf {
            // Unconditional instructions.
            if !a32_uncond::disas(self, insn) && !fp::disas_uncond(self, insn) {
                self.unallocated_encoding();
            }
            return;
        }
        if cond != 0xe {
            // Not always executed: branch to the next instruction unless the condition
            // passes.
            self.skip_unless(cond);
        }
        if !a32::disas(self, insn) && !fp::disas(self, insn) {
            self.unallocated_encoding();
        }
    }

    /// `arm_post_translate_insn()`.
    fn post_translate_insn(&mut self) {
        if self.d.condjmp && self.b.is_jmp == DisasJumpType::Next {
            if let Some(l) = self.d.condlabel {
                self.f().gen_set_label(l);
            }
            self.d.condjmp = false;
        }
    }
}

impl DisasContext {
    /// `thumb_tr_translate_insn()`.
    fn thumb_translate_insn(
        &mut self,
        db: &mut DisasContextBase<'_>,
        cpu: &mut Cpu<'_>,
    ) -> Result<(), CpuLoopExit> {
        let pc = db.pc_next;
        if (S { d: self, b: db }).check_kernelpage() {
            db.pc_next = pc + 2;
            return Ok(());
        }
        self.pc_curr = pc;
        let hw = db.translator_lduw(cpu, pc, Endian::Little)?;
        // thumb_insn_is_16bit(): Thumb-2 cores always treat 32-bit insns as 32-bit.
        let is_16bit = (hw >> 11) < 0x1d;
        let insn = if is_16bit {
            db.pc_next = pc + 2;
            u32::from(hw)
        } else {
            let hw2 = db.translator_lduw(cpu, pc + 2, Endian::Little)?;
            db.pc_next = pc + 4;
            (u32::from(hw) << 16) | u32::from(hw2)
        };

        let mut s = S { d: self, b: db };
        if s.d.pstate_il {
            // Illegal execution state. This has priority over BTI exceptions, but comes
            // after instruction abort exceptions.
            s.gen_exception_insn(0, EXCP_UDEF, syn_illegalstate());
            return Ok(());
        }
        if s.d.condexec_mask != 0 && !thumb_insn_is_unconditional(insn) {
            // Conditionally skip the insn. Both 0xe and 0xf mean "always".
            let cond = s.d.condexec_cond;
            if cond < 0xe {
                s.skip_unless(cond);
            }
        }
        let ok = if is_16bit {
            t16::disas(&mut s, insn as u16)
        } else {
            t32::disas(&mut s, insn)
                || fp::disas_uncond(&mut s, insn)
                || (insn >> 28 == 0xe && fp::disas(&mut s, insn))
        };
        if !ok {
            s.unallocated_encoding();
        }

        // Advance the Thumb condexec condition.
        if s.d.condexec_mask != 0 {
            s.d.condexec_cond = (s.d.condexec_cond & 0xe) | ((s.d.condexec_mask >> 4) & 1);
            s.d.condexec_mask = (s.d.condexec_mask << 1) & 0x1f;
            if s.d.condexec_mask == 0 {
                s.d.condexec_cond = 0;
            }
        }
        s.post_translate_insn();

        // Thumb is a variable-length ISA. Stop translation when the next insn will touch a
        // new page, so that prefetch aborts occur at the right place: if we are at the last
        // halfword of the page, end the block unless it is a 16-bit insn.
        if db.is_jmp == DisasJumpType::Next {
            let used = db.pc_next - self.page_start;
            if used >= PAGE_SIZE {
                db.is_jmp = DisasJumpType::TooMany;
            } else if used >= PAGE_SIZE - 3 {
                let next = db.translator_lduw(cpu, db.pc_next, Endian::Little)?;
                if (next >> 11) >= 0x1d {
                    db.is_jmp = DisasJumpType::TooMany;
                }
            }
        }
        Ok(())
    }
}

impl TranslatorOps for DisasContext {
    fn init_disas_context(&mut self, db: &mut DisasContextBase<'_>, _cpu: &mut Cpu<'_>) {
        let f = &mut db.tb.f;
        let env = f.env();
        let r = std::array::from_fn(|n| f.global_mem_new_i32(env, reg_off(n) as i64, REG_NAMES[n]));
        self.g = Some(G {
            r,
            pc: f.global_mem_new_i64(env, PC as i64, "pc"),
            nf: f.global_mem_new_i32(env, NF as i64, "NF"),
            zf: f.global_mem_new_i32(env, ZF as i64, "ZF"),
            cf: f.global_mem_new_i32(env, CF as i64, "CF"),
            vf: f.global_mem_new_i32(env, VF as i64, "VF"),
        });
        self.condjmp = false;
        self.condlabel = None;

        let flags = db.tb.flags;
        self.current_el = flags & TB_EL_MASK;
        self.mmu_idx = (flags >> TB_MMUIDX_SHIFT) & 0xf;
        self.pstate_il = flags & TB_PSTATE_IL != 0;
        self.align_mem = flags & TB_ALIGN_MEM != 0;
        self.be_data = if flags & TB_BE_DATA != 0 { MemOp::BE } else { MemOp::LE };
        // The Thumb bit and the IT state come in `cs_base`, see `get_tb_cpu_state()`.
        let cs_base = db.tb.cs_base;
        self.thumb = cs_base & 1 != 0;
        let condexec = ((cs_base >> 1) & 0xff) as u32;
        self.condexec_mask = (condexec & 0xf) << 1;
        self.condexec_cond = condexec >> 4;
        self.page_start = db.pc_first & !(PAGE_SIZE - 1);

        // ARM is a fixed-length ISA. Bound the number of insns to execute to those left on
        // the page.
        if !self.thumb {
            let bound = (db.pc_first | !(PAGE_SIZE - 1)).wrapping_neg() / 4;
            db.max_insns = db.max_insns.min(bound as u32);
        }
    }

    fn tb_start(&mut self, db: &mut DisasContextBase<'_>, _cpu: &mut Cpu<'_>) {
        // Reset the conditional execution bits immediately. This avoids complications
        // trying to do it at the end of the block.
        if self.condexec_mask != 0 || self.condexec_cond != 0 {
            let f = &mut db.tb.f;
            let env = f.env();
            let zero = f.constant_i32(0);
            f.gen_st_i32(zero, env, CONDEXEC_BITS as i64);
        }
    }

    fn insn_start(&mut self, db: &mut DisasContextBase<'_>, _cpu: &mut Cpu<'_>) {
        let condexec = (self.condexec_cond << 4) | (self.condexec_mask >> 1);
        let pc = db.pc_next;
        db.tb.f.gen_insn_start(&[pc, u64::from(condexec)]);
    }

    fn translate_insn(
        &mut self,
        db: &mut DisasContextBase<'_>,
        cpu: &mut Cpu<'_>,
    ) -> Result<(), CpuLoopExit> {
        if self.thumb {
            return self.thumb_translate_insn(db, cpu);
        }
        let pc = db.pc_next;
        if pc & 3 != 0 {
            // PC alignment fault. This has priority over the instruction abort that we
            // would receive from a translation fault, and can only happen after an indirect
            // branch, at the start of the block.
            let mut s = S { d: self, b: db };
            s.d.pc_curr = pc;
            s.update_pc(0);
            let env = s.env();
            let t = s.c64(pc as i64);
            s.call(&helpers::EXCEPTION_PC_ALIGNMENT, None, &[env.into(), t.into()]);
            s.b.is_jmp = DisasJumpType::NoReturn;
            s.b.pc_next = (pc + 3) & !3;
            return Ok(());
        }
        if (S { d: self, b: db }).check_kernelpage() {
            db.pc_next = pc + 4;
            return Ok(());
        }
        self.pc_curr = pc;
        let insn = db.translator_ldl(cpu, pc, Endian::Little)?;
        db.pc_next = pc + 4;
        let mut s = S { d: self, b: db };
        s.disas_arm_insn(insn);
        s.post_translate_insn();
        Ok(())
    }

    fn tb_stop(&mut self, db: &mut DisasContextBase<'_>, _cpu: &mut Cpu<'_>) {
        let mut s = S { d: self, b: db };
        // At this stage condjmp is only set when the skipped instruction was a conditional
        // branch or trap, and the PC has already been written.
        s.set_condexec();
        let len = s.insn_len();
        match s.b.is_jmp {
            DisasJumpType::Next | DisasJumpType::TooMany => s.gen_goto_tb(1, len),
            DisasJumpType::NoReturn => {}
            DISAS_JUMP => s.f().gen_lookup_and_goto_ptr(),
            DISAS_UPDATE_EXIT => {
                s.update_pc(len);
                s.f().gen_exit_tb(0, 0);
            }
            DISAS_SWI => {
                let syn = syn_aa32_svc(s.d.svc_imm, s.d.thumb);
                let el = s.d.current_el.max(1);
                s.gen_exception_el(EXCP_SWI, syn, el);
            }
            // DISAS_EXIT: the hash table must be used to find the next block.
            _ => s.f().gen_exit_tb(0, 0),
        }
        if s.d.condjmp {
            // "Condition failed" instruction codepath for the branch/trap insn.
            if let Some(l) = s.d.condlabel {
                s.f().gen_set_label(l);
            }
            s.set_condexec();
            s.gen_goto_tb(1, len);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{t32_branch24, t32_expandimm_imm};

    #[test]
    fn expandimm() {
        assert_eq!(t32_expandimm_imm(0x0ab), 0xab);
        assert_eq!(t32_expandimm_imm(0x1ab), 0x00ab_00ab);
        assert_eq!(t32_expandimm_imm(0x2ab), 0xab00_ab00_u32 as i32);
        assert_eq!(t32_expandimm_imm(0x3ab), 0xabab_abab_u32 as i32);
        // Rotated: the top bit of the 8-bit value is forced on.
        assert_eq!(t32_expandimm_imm(0x42b), 0xab);
    }

    #[test]
    fn branch24() {
        // S = 0 with J1 = J2 = 1 is I1 = I2 = 0: a small forward offset.
        assert_eq!(t32_branch24(3 << 21 | 0x10), 0x20);
        // S = 1 with J1 = J2 = 1 is I1 = I2 = 1: a small backward offset.
        assert_eq!(t32_branch24(-1), -2);
    }
}
