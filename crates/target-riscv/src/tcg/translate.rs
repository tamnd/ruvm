// SPDX-License-Identifier: GPL-2.0-or-later

//! The RV64 translator: QEMU's `target/riscv/tcg/translate.c` with the integer parts of
//! `insn_trans/` (`trans_rvi`, `trans_rvm`, `trans_rva`, `trans_rvzicsr`, `trans_privileged`,
//! `trans_rvb`, `trans_rvzicbo`, `trans_rvzawrs`, `trans_rvzicond` is not on by default) and
//! `trans_xlrbr`. The floating point instructions are in `translate_fp` and the hypervisor
//! ones in `translate_rvh`.
//!
//! The front end is RV64 only. Conditional branches may continue the block with their
//! fall-through path (a superblock, not in QEMU), the same way as the arm front end.

use ruvm_jit::cputlb::cpu_ld_code;
use ruvm_jit::{Cpu, CpuLoopExit, DisasContextBase, DisasJumpType, Ra, TranslatorOps, cf};
use ruvm_jit_core::ir::{TempI32, TempI64, TempPtr};
use ruvm_jit_core::tcg_op_ldst::AtomicOp;
use ruvm_jit_core::types::{Cond, mo};
use ruvm_jit_core::{Func, Label, MemOp, Temp};
use ruvm_mem::Endian;

use super::crypto;
use super::helpers::{self, Def};
use super::pm::PointerMask;
use super::translate_rvv::Ldst;
use super::{
    TB_FS_SHIFT, TB_LMUL_SHIFT, TB_MEM_IDX_MASK, TB_PRIV_SHIFT, TB_SEW_SHIFT, TB_VILL, TB_VIRT,
    TB_VMA, TB_VS_SHIFT, TB_VSTART_EQ_ZERO, TB_VTA,
};
use super::{
    translate_fp, translate_rvh, translate_rvv, translate_rvv_fp, translate_rvv_int,
    translate_rvv_perm, translate_rvvk,
};
use crate::cpu::{
    BADADDR, BINS, EXCP_BREAKPOINT, EXCP_ILLEGAL_INST, EXCP_INST_ADDR_MIS, EXCP_SEMIHOST,
    EXCP_U_ECALL, EXT_STATUS_DIRTY, LOAD_RES, LOAD_VAL, MSTATUS, MSTATUS_FS, MSTATUS_HS,
    MSTATUS_VS, PC, PRV_U, RVA, RVC, RVM, RVS, RiscvCfg, UW2_ALWAYS_STORE_AMO, fpr_off, gpr_off,
};
use crate::decode::insn16::{DecodeInsn16, arg_c_mop_n, decode16};
use crate::decode::insn32::*;
use crate::decode::xlrbr::{self, DecodeXlrbr};

/// The ABI names of the integer registers, QEMU's `riscv_int_regnames`.
const GPR_NAMES: [&str; 32] = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4",
    "a5", "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4",
    "t5", "t6",
];

/// The ABI names of the floating point registers, QEMU's `riscv_fpr_regnames`.
const FPR_NAMES: [&str; 32] = [
    "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "fs0", "fs1", "fa0", "fa1", "fa2",
    "fa3", "fa4", "fa5", "fa6", "fa7", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7", "fs8", "fs9",
    "fs10", "fs11", "ft8", "ft9", "ft10", "ft11",
];

/// The semihosting trap sequence around an `ebreak`: `slli zero, zero, 0x1f` before it and
/// `srai zero, zero, 7` after it.
const SEMIHOST_PRE: u32 = 0x01f0_1013;
const EBREAK: u32 = 0x0010_0073;
const SEMIHOST_POST: u32 = 0x4070_5013;

/// The most conditional branches one block goes past, see [`SideExit`].
const MAX_SIDE_EXITS: usize = 8;

/// The globals of the block, QEMU's `cpu_gpr`, `cpu_fpr`, `cpu_pc`, `load_res` and
/// `load_val`. `x[0]` exists but is never read or written.
pub(super) struct G {
    pub(super) x: [TempI64; 32],
    pub(super) f: [TempI64; 32],
    pub(super) pc: TempI64,
    pub(super) load_res: TempI64,
    pub(super) load_val: TempI64,
}

/// A conditional branch whose fall-through path the block goes on with (a superblock, not in
/// QEMU). Its exit is emitted out of line when the block ends.
#[derive(Clone, Copy)]
struct SideExit {
    label: Label,
    /// The branch target.
    dest: u64,
    /// The `goto_tb` slot kept for this exit.
    slot: Option<u64>,
}

/// The riscv `DisasContext`.
pub(crate) struct DisasContext {
    /// `Some(userspace)` when semihosting is on: whether U mode may use it too.
    semihosting: Option<bool>,
    /// Whether the XLRBR vendor extension is on.
    xlrbr: bool,
    /// The extensions of the CPU, `cfg_ptr`.
    pub(super) cfg: RiscvCfg,
    /// `mstatus.VS` as the block found it, updated by `mark_vs_dirty()`.
    pub(super) mstatus_vs: u64,
    /// `vill`.
    pub(super) vill: bool,
    /// `vtype.vsew`: log2 of SEW / 8.
    pub(super) sew: i32,
    /// `vtype.vlmul` sign extended: log2 of LMUL, from -3 to 3 (-4 is reserved).
    pub(super) lmul: i32,
    /// `vtype.vta && cfg.rvv_ta_all_1s`.
    pub(super) vta: bool,
    /// `vtype.vma && cfg.rvv_ma_all_1s`.
    pub(super) vma: bool,
    /// `cfg.rvv_ta_all_1s`.
    pub(super) cfg_vta_all_1s: bool,
    /// Whether `vstart` is 0, updated by `finalize_rvv_inst()`.
    pub(super) vstart_eq_zero: bool,
    /// The privilege level of the block.
    pub(super) priv_lvl: u64,
    /// Whether the block runs with V=1, `virt_enabled`.
    pub(super) virt_enabled: bool,
    /// The MMU index of data accesses.
    pub(super) mem_idx: u32,
    /// The pointer mask of data accesses, `addr_xl` and `addr_signed`.
    pm: PointerMask,
    /// `mstatus.FS` as the block found it, updated by `mark_fs_dirty()`.
    pub(super) mstatus_fs: u64,
    /// The rounding mode `gen_set_rm()` last set in this block, or -1.
    pub(super) frm: i32,
    /// Whether `frm` was checked to be valid in this block.
    pub(super) frm_valid: bool,
    /// The PC of the current instruction.
    pub(super) pc_curr: u64,
    /// The length of the current instruction, 2 or 4.
    pub(super) cur_insn_len: u64,
    /// The bits of the current instruction.
    pub(super) opcode: u64,
    /// Whether the semihosting sequence surrounds the current `ebreak`.
    semihost_seq: bool,
    pub(super) g: Option<G>,
    /// The `goto_tb` slots used so far, one bit each.
    goto_tb_used: u64,
    side_exits: Vec<SideExit>,
}

impl DisasContext {
    pub(crate) fn new(semihosting: Option<bool>, xlrbr: bool, cfg: RiscvCfg) -> DisasContext {
        DisasContext {
            semihosting,
            xlrbr,
            cfg,
            mstatus_vs: 0,
            vill: true,
            sew: 0,
            lmul: 0,
            vta: false,
            vma: false,
            cfg_vta_all_1s: false,
            vstart_eq_zero: true,
            priv_lvl: 0,
            virt_enabled: false,
            mem_idx: 0,
            pm: PointerMask::default(),
            mstatus_fs: 0,
            frm: -1,
            frm_valid: false,
            pc_curr: 0,
            cur_insn_len: 4,
            opcode: 0,
            semihost_seq: false,
            g: None,
            goto_tb_used: 0,
            side_exits: Vec::new(),
        }
    }
}

/// The translator state while one instruction is translated: the riscv and the generic
/// halves of QEMU's `DisasContext`.
pub(super) struct S<'a, 'b> {
    pub(super) d: &'a mut DisasContext,
    pub(super) b: &'a mut DisasContextBase<'b>,
}

/// `insn_len()`.
fn insn_len(first_half: u16) -> u64 {
    if first_half & 3 == 3 { 4 } else { 2 }
}

impl S<'_, '_> {
    pub(super) fn f(&mut self) -> &mut Func {
        &mut self.b.tb.f
    }

    pub(super) fn env(&mut self) -> TempPtr {
        self.f().env()
    }

    pub(super) fn new64(&mut self) -> TempI64 {
        self.f().temp_new_i64()
    }

    pub(super) fn new32(&mut self) -> TempI32 {
        self.f().temp_new_i32()
    }

    pub(super) fn c64(&mut self, v: i64) -> TempI64 {
        self.f().constant_i64(v)
    }

    pub(super) fn c32(&mut self, v: i32) -> TempI32 {
        self.f().constant_i32(v)
    }

    pub(super) fn call(&mut self, d: &Def, ret: Option<Temp>, args: &[Temp]) {
        let f = self.f();
        let h = f.helper(d.info());
        f.gen_call(h, ret, args);
    }

    fn g(&self) -> &G {
        self.d.g.as_ref().expect("globals exist while translating")
    }

    // Registers.

    /// `get_gpr(ctx, reg, EXT_NONE)`: x0 reads as a zero constant. The result must not be
    /// written.
    pub(super) fn gpr(&mut self, r: i32) -> TempI64 {
        if r == 0 { self.c64(0) } else { self.g().x[r as usize] }
    }

    /// `get_gpr(ctx, reg, EXT_SIGN)` for 32-bit operations, in a fresh temp.
    fn gpr_s32(&mut self, r: i32) -> TempI64 {
        let s = self.gpr(r);
        let t = self.new64();
        self.f().gen_ext32s_i64(t, s);
        t
    }

    /// `get_gpr(ctx, reg, EXT_ZERO)` for 32-bit operations, in a fresh temp.
    fn gpr_u32(&mut self, r: i32) -> TempI64 {
        let s = self.gpr(r);
        let t = self.new64();
        self.f().gen_ext32u_i64(t, s);
        t
    }

    /// `gen_set_gpr()`: writes to x0 are dropped.
    pub(super) fn set_gpr(&mut self, r: i32, v: TempI64) {
        if r != 0 {
            let x = self.g().x[r as usize];
            self.f().gen_mov_i64(x, v);
        }
    }

    /// `gen_set_gpr()` with `ctx->ol == MXL_RV32`: the low 32 bits, sign extended.
    fn set_gpr_w(&mut self, r: i32, v: TempI64) {
        if r != 0 {
            let x = self.g().x[r as usize];
            self.f().gen_ext32s_i64(x, v);
        }
    }

    /// `gen_set_gpri()`.
    fn set_gpri(&mut self, r: i32, imm: i64) {
        if r != 0 {
            let x = self.g().x[r as usize];
            self.f().gen_movi_i64(x, imm);
        }
    }

    /// The global of floating point register `r`, `cpu_fpr[r]`.
    pub(super) fn fpr(&self, r: i32) -> TempI64 {
        self.g().f[r as usize]
    }

    /// `get_address()`: `rs1 + imm` in a fresh temp, with the high bits pointer masking
    /// ignores sign or zero extended from the rest.
    pub(super) fn address(&mut self, rs1: i32, imm: i64) -> TempI64 {
        let s = self.gpr(rs1);
        let t = self.new64();
        self.f().gen_addi_i64(t, s, imm);
        let pm = self.d.pm;
        let pmlen = pm.pmlen();
        if pmlen != 0 {
            if pm.signext {
                self.f().gen_sextract_i64(t, t, 0, 64 - pmlen);
            } else {
                self.f().gen_extract_i64(t, t, 0, 64 - pmlen);
            }
        }
        t
    }

    // Instruction start, exceptions and block ends.

    /// `decode_save_opc()`: record the instruction bits and `excp_uw2` for unwinding.
    pub(super) fn decode_save_opc(&mut self, excp_uw2: u64) {
        let op = self.b.insn_start.expect("instruction started");
        let opcode = self.d.opcode;
        self.f().set_insn_start_param(op, 1, opcode);
        self.f().set_insn_start_param(op, 2, excp_uw2);
    }

    /// `gen_update_pc()`: `pc = pc_curr + diff`.
    pub(super) fn update_pc(&mut self, diff: i64) {
        let dest = self.d.pc_curr.wrapping_add(diff as u64);
        let pc = self.g().pc;
        self.f().gen_movi_i64(pc, dest as i64);
    }

    /// `generate_exception()`.
    pub(super) fn generate_exception(&mut self, excp: i32) {
        self.update_pc(0);
        let env = self.env();
        let e = self.c32(excp);
        self.call(&helpers::RAISE_EXCEPTION, None, &[env.into(), e.into()]);
        self.b.is_jmp = DisasJumpType::NoReturn;
    }

    /// `gen_exception_inst_addr_mis()`: a misaligned jump to `target`.
    fn gen_exception_inst_addr_mis(&mut self, target: TempI64) {
        let env = self.env();
        self.f().gen_st_i64(target, env, BADADDR as i64);
        self.generate_exception(EXCP_INST_ADDR_MIS);
    }

    /// `gen_exception_illegal()`.
    pub(super) fn gen_exception_illegal(&mut self) {
        let op = self.c64(self.d.opcode as i64);
        let env = self.env();
        self.f().gen_st_i64(op, env, BINS as i64);
        self.generate_exception(EXCP_ILLEGAL_INST);
    }

    /// `exit_tb()` after `pc` is written: back to the main loop.
    pub(super) fn exit_tb(&mut self) {
        self.f().gen_exit_tb(0, 0);
        self.b.is_jmp = DisasJumpType::NoReturn;
    }

    /// `gen_update_pc(cur_insn_len)` and `exit_tb()`: end the block after this
    /// instruction, so that the main loop sees what it changed.
    pub(super) fn end_tb_next(&mut self) {
        self.update_pc(self.d.cur_insn_len as i64);
        self.exit_tb();
    }

    /// `gen_goto_tb()`: go to `pc_curr + diff`, chaining when possible.
    fn gen_goto_tb(&mut self, n: u64, diff: i64) {
        let dest = self.d.pc_curr.wrapping_add(diff as u64);
        let slot = self.goto_tb_slot(n);
        self.gen_goto_dest(slot, dest);
    }

    /// The `goto_tb` slot to use for an exit that would like slot `n`: that one if it is
    /// free, else the other, else none.
    fn goto_tb_slot(&self, n: u64) -> Option<u64> {
        [n, n ^ 1].into_iter().find(|&k| self.d.goto_tb_used & (1 << k) == 0)
    }

    /// Go to `dest`, chaining through slot `slot` when there is one and `dest` is on the
    /// page of the block, else through the inline cache (not in QEMU, which uses
    /// `lookup_and_goto_ptr`).
    fn gen_goto_dest(&mut self, slot: Option<u64>, dest: u64) {
        let pc = self.g().pc;
        if let (true, Some(n)) = (self.b.translator_use_goto_tb(dest), slot) {
            self.d.goto_tb_used |= 1 << n;
            self.f().gen_goto_tb(n);
            self.f().gen_movi_i64(pc, dest as i64);
            let id = self.b.tb.id;
            self.f().gen_exit_tb(id, n);
        } else {
            self.f().gen_movi_i64(pc, dest as i64);
            let t = self.c64(dest as i64);
            self.f().gen_lookup_and_goto_ptr_ic(t);
        }
        self.b.is_jmp = DisasJumpType::NoReturn;
    }

    /// Whether a conditional branch here may continue the block with its fall-through path
    /// rather than end it (a superblock, not in QEMU). Not with icount, which charges a
    /// whole block's instructions on entry, or single step.
    fn can_inline_branch(&self) -> bool {
        self.b.tb.cflags & (cf::USE_ICOUNT | cf::SINGLE_STEP) == 0
            && self.d.side_exits.len() < MAX_SIDE_EXITS
    }

    /// The rest of a conditional branch to `pc_curr + imm` once the code branches to `label`
    /// when it is taken. When it can, the block goes on with the fall-through path and the
    /// taken edge becomes a [`SideExit`]; else the block ends with both edges, as in QEMU.
    fn branch_to(&mut self, label: Label, imm: i64) {
        if !self.can_inline_branch() {
            self.gen_goto_tb(1, self.d.cur_insn_len as i64);
            self.f().gen_set_label(label);
            self.gen_goto_tb(0, imm);
            return;
        }
        let dest = self.d.pc_curr.wrapping_add(imm as u64);
        // The earlier exits get the direct slots: every run of the block passes their
        // branch, not all reach the later ones.
        let mut slot = None;
        if self.b.translator_use_goto_tb(dest) {
            slot = self.goto_tb_slot(0);
            if let Some(n) = slot {
                self.d.goto_tb_used |= 1 << n;
            }
        }
        self.d.side_exits.push(SideExit { label, dest, slot });
    }

    /// Emit the exits of the conditional branches the block went past, after its last
    /// instruction.
    fn gen_side_exits(&mut self) {
        let exits = std::mem::take(&mut self.d.side_exits);
        for e in &exits {
            self.f().gen_set_label(e.label);
            if let Some(n) = e.slot {
                self.d.goto_tb_used &= !(1 << n);
            }
            self.gen_goto_dest(e.slot, e.dest);
        }
        self.d.side_exits = exits;
        self.d.side_exits.clear();
    }

    /// `lookup_and_goto_ptr()` once `pc` holds the target.
    pub(super) fn lookup_and_goto_ptr(&mut self) {
        let pc = self.g().pc;
        self.f().gen_lookup_and_goto_ptr_ic(pc);
        self.b.is_jmp = DisasJumpType::NoReturn;
    }

    /// `mark_fs_dirty()`.
    pub(super) fn mark_fs_dirty(&mut self) {
        if self.d.mstatus_fs != EXT_STATUS_DIRTY {
            // Remember the state change for the rest of the TB.
            self.d.mstatus_fs = EXT_STATUS_DIRTY;
            let env = self.env();
            let t = self.new64();
            let f = self.f();
            f.gen_ld_i64(t, env, MSTATUS as i64);
            f.gen_ori_i64(t, t, MSTATUS_FS as i64);
            f.gen_st_i64(t, env, MSTATUS as i64);
            if self.d.virt_enabled {
                let f = self.f();
                f.gen_ld_i64(t, env, MSTATUS_HS as i64);
                f.gen_ori_i64(t, t, MSTATUS_FS as i64);
                f.gen_st_i64(t, env, MSTATUS_HS as i64);
            }
        }
    }

    /// `mark_vs_dirty()`.
    pub(super) fn mark_vs_dirty(&mut self) {
        if self.d.mstatus_vs != EXT_STATUS_DIRTY {
            // Remember the state change for the rest of the TB.
            self.d.mstatus_vs = EXT_STATUS_DIRTY;
            let env = self.env();
            let t = self.new64();
            let f = self.f();
            f.gen_ld_i64(t, env, MSTATUS as i64);
            f.gen_ori_i64(t, t, MSTATUS_VS as i64);
            f.gen_st_i64(t, env, MSTATUS as i64);
            if self.d.virt_enabled {
                let f = self.f();
                f.gen_ld_i64(t, env, MSTATUS_HS as i64);
                f.gen_ori_i64(t, t, MSTATUS_VS as i64);
                f.gen_st_i64(t, env, MSTATUS_HS as i64);
            }
        }
    }

    /// `finalize_rvv_inst()`: a vector instruction leaves the state dirty and `vstart` 0.
    pub(super) fn finalize_rvv_inst(&mut self) {
        self.mark_vs_dirty();
        self.d.vstart_eq_zero = true;
    }

    // Integer operations.

    /// `rd = op(rs1, rs2)` on 64 bits.
    fn arith(&mut self, a: &arg_r, op: fn(&mut Func, TempI64, TempI64, TempI64)) -> bool {
        let s1 = self.gpr(a.rs1);
        let s2 = self.gpr(a.rs2);
        let d = self.new64();
        op(self.f(), d, s1, s2);
        self.set_gpr(a.rd, d);
        true
    }

    /// `rd = sext32(op(rs1, rs2))`.
    fn arith_w(&mut self, a: &arg_r, op: fn(&mut Func, TempI64, TempI64, TempI64)) -> bool {
        let s1 = self.gpr(a.rs1);
        let s2 = self.gpr(a.rs2);
        let d = self.new64();
        op(self.f(), d, s1, s2);
        self.set_gpr_w(a.rd, d);
        true
    }

    /// `rd = op(rs1, imm)`.
    fn arith_imm(&mut self, a: &arg_i, op: fn(&mut Func, TempI64, TempI64, i64)) -> bool {
        let s1 = self.gpr(a.rs1);
        let d = self.new64();
        op(self.f(), d, s1, i64::from(a.imm));
        self.set_gpr(a.rd, d);
        true
    }

    /// `rd = op(rs1)`.
    fn unary(&mut self, a: &arg_r2, op: fn(&mut Func, TempI64, TempI64)) -> bool {
        let s1 = self.gpr(a.rs1);
        let d = self.new64();
        op(self.f(), d, s1);
        self.set_gpr(a.rd, d);
        true
    }

    /// `gen_shift()` on 64 bits: the shift amount is `rs2 & 63`.
    fn shift(&mut self, a: &arg_r, op: fn(&mut Func, TempI64, TempI64, TempI64)) -> bool {
        let s1 = self.gpr(a.rs1);
        let s2 = self.gpr(a.rs2);
        let amt = self.new64();
        let d = self.new64();
        let f = self.f();
        f.gen_andi_i64(amt, s2, 63);
        op(f, d, s1, amt);
        self.set_gpr(a.rd, d);
        true
    }

    /// `gen_shift()` on 32 bits: `src1` is the extended operand, the shift amount is
    /// `rs2 & 31`, and the result is sign extended.
    fn shift_w(
        &mut self,
        a: &arg_r,
        src1: TempI64,
        op: fn(&mut Func, TempI64, TempI64, TempI64),
    ) -> bool {
        let s2 = self.gpr(a.rs2);
        let amt = self.new64();
        let d = self.new64();
        let f = self.f();
        f.gen_andi_i64(amt, s2, 31);
        op(f, d, src1, amt);
        self.set_gpr_w(a.rd, d);
        true
    }

    /// `rd = op(rs1, shamt)` for the 64-bit immediate shifts.
    fn shift_imm(&mut self, a: &arg_shift, op: fn(&mut Func, TempI64, TempI64, i64)) -> bool {
        if a.shamt >= 64 {
            return false;
        }
        let s1 = self.gpr(a.rs1);
        let d = self.new64();
        op(self.f(), d, s1, i64::from(a.shamt));
        self.set_gpr(a.rd, d);
        true
    }

    /// `rd = sext32(op(rs1, shamt))` for the 32-bit immediate shifts.
    fn shift_imm_w(&mut self, a: &arg_shift, op: fn(&mut Func, TempI64, TempI64, u32)) -> bool {
        if a.shamt >= 32 {
            return false;
        }
        let s1 = self.gpr(a.rs1);
        let d = self.new64();
        op(self.f(), d, s1, a.shamt as u32);
        self.set_gpr_w(a.rd, d);
        true
    }

    /// The atomicity Zama16b gives the loads and stores that are not AMOs: atomic when
    /// the access does not cross a 16-byte boundary.
    pub(super) fn zama16b(&self) -> MemOp {
        if self.d.cfg.ext_zama16b { MemOp::ATOM_WITHIN16 } else { MemOp::ATOM_IFALIGN }
    }

    /// `gen_load()`.
    fn load(&mut self, a: &arg_i, memop: MemOp) -> bool {
        let memop = memop | self.zama16b();
        self.decode_save_opc(0);
        let addr = self.address(a.rs1, i64::from(a.imm));
        let d = self.new64();
        let idx = self.d.mem_idx;
        self.f().gen_qemu_ld_i64(d, addr, idx, memop);
        self.set_gpr(a.rd, d);
        if self.d.cfg.ext_ztso {
            self.f().gen_mb(mo::ALL | mo::BAR_LDAQ);
        }
        true
    }

    /// `gen_store()`.
    fn store(&mut self, a: &arg_s, memop: MemOp) -> bool {
        let memop = memop | self.zama16b();
        self.decode_save_opc(0);
        let addr = self.address(a.rs1, i64::from(a.imm));
        let v = self.gpr(a.rs2);
        if self.d.cfg.ext_ztso {
            self.f().gen_mb(mo::ALL | mo::BAR_STRL);
        }
        let idx = self.d.mem_idx;
        self.f().gen_qemu_st_i64(v, addr, idx, memop);
        true
    }

    /// `gen_branch()`.
    fn branch(&mut self, a: &arg_b, cond: Cond) -> bool {
        let s1 = self.gpr(a.rs1);
        let s2 = self.gpr(a.rs2);
        let l = self.f().new_label();
        self.f().gen_brcond_i64(cond, s1, s2, l);
        if !self.d.cfg.allow_16bit_insn() && a.imm & 3 != 0 {
            // The taken path is a misaligned jump, as in QEMU the block ends here.
            self.gen_goto_tb(1, self.d.cur_insn_len as i64);
            self.f().gen_set_label(l);
            let dest = self.d.pc_curr.wrapping_add(i64::from(a.imm) as u64);
            let t = self.c64(dest as i64);
            self.gen_exception_inst_addr_mis(t);
            return true;
        }
        self.branch_to(l, i64::from(a.imm));
        true
    }

    /// `gen_div()` and `gen_rem()`: signed division with the RISC-V results for a zero
    /// divisor (all ones, and the dividend) and for overflow (the dividend, and zero).
    fn div_rem(&mut self, d: TempI64, s1: TempI64, s2: TempI64, rem: bool) {
        let t1 = self.new64();
        let t2 = self.new64();
        let zero = self.c64(0);
        let one = self.c64(1);
        let mone = self.c64(-1);
        let f = self.f();
        // If overflow, set the divisor to 1, giving the dividend and a zero remainder.
        f.gen_setcondi_i64(Cond::Eq, t1, s1, i64::MIN);
        f.gen_setcondi_i64(Cond::Eq, t2, s2, -1);
        f.gen_and_i64(t1, t1, t2);
        f.gen_movcond_i64(Cond::Ne, t2, t1, zero, one, s2);
        if rem {
            // If div by zero, set the divisor to 1 and take the dividend.
            f.gen_movcond_i64(Cond::Eq, t2, s2, zero, one, t2);
            f.gen_rem_i64(t1, s1, t2);
            f.gen_movcond_i64(Cond::Eq, d, s2, zero, s1, t1);
        } else {
            // If div by zero, divide -1 by 1.
            f.gen_movcond_i64(Cond::Eq, t1, s2, zero, mone, s1);
            f.gen_movcond_i64(Cond::Eq, t2, s2, zero, one, t2);
            f.gen_div_i64(d, t1, t2);
        }
    }

    /// `gen_divu()` and `gen_remu()`.
    fn divu_remu(&mut self, d: TempI64, s1: TempI64, s2: TempI64, rem: bool) {
        let t1 = self.new64();
        let t2 = self.new64();
        let zero = self.c64(0);
        let one = self.c64(1);
        let mone = self.c64(-1);
        let f = self.f();
        f.gen_movcond_i64(Cond::Eq, t2, s2, zero, one, s2);
        if rem {
            f.gen_remu_i64(t1, s1, t2);
            f.gen_movcond_i64(Cond::Eq, d, s2, zero, s1, t1);
        } else {
            f.gen_movcond_i64(Cond::Eq, t1, s2, zero, mone, s1);
            f.gen_divu_i64(d, t1, t2);
        }
    }

    /// The M extension division instructions; `w` takes 32-bit operands extended the way
    /// QEMU does (sign for the signed ones, zero for the others) and sign extends the
    /// result.
    fn div_op(&mut self, a: &arg_r, signed: bool, rem: bool, w: bool) -> bool {
        let (s1, s2) = match (w, signed) {
            (false, _) => (self.gpr(a.rs1), self.gpr(a.rs2)),
            (true, true) => (self.gpr_s32(a.rs1), self.gpr_s32(a.rs2)),
            (true, false) => (self.gpr_u32(a.rs1), self.gpr_u32(a.rs2)),
        };
        let d = self.new64();
        if signed {
            self.div_rem(d, s1, s2, rem);
        } else {
            self.divu_remu(d, s1, s2, rem);
        }
        if w {
            self.set_gpr_w(a.rd, d);
        } else {
            self.set_gpr(a.rd, d);
        }
        true
    }

    /// `rd = (zext32(rs1) << sh) + rs2` (`shNadd.uw`), or `(rs1 << sh) + rs2` (`shNadd`).
    fn shadd(&mut self, a: &arg_r, sh: i64, uw: bool) -> bool {
        let s1 = if uw { self.gpr_u32(a.rs1) } else { self.gpr(a.rs1) };
        let s2 = self.gpr(a.rs2);
        let d = self.new64();
        let f = self.f();
        f.gen_shli_i64(d, s1, sh);
        f.gen_add_i64(d, d, s2);
        self.set_gpr(a.rd, d);
        true
    }

    /// The Zbs register forms: `op(rs1, 1 << (rs2 & 63))`, or `(rs1 >> (rs2 & 63)) & 1`.
    fn bit_op(&mut self, a: &arg_r, op: BitOp) -> bool {
        let s1 = self.gpr(a.rs1);
        let s2 = self.gpr(a.rs2);
        let amt = self.new64();
        self.f().gen_andi_i64(amt, s2, 63);
        let d = self.bit_apply(s1, amt, op);
        self.set_gpr(a.rd, d);
        true
    }

    /// The Zbs immediate forms.
    fn bit_op_imm(&mut self, a: &arg_shift, op: BitOp) -> bool {
        if a.shamt >= 64 {
            return false;
        }
        let s1 = self.gpr(a.rs1);
        let amt = self.c64(i64::from(a.shamt));
        let d = self.bit_apply(s1, amt, op);
        self.set_gpr(a.rd, d);
        true
    }

    fn bit_apply(&mut self, s1: TempI64, amt: TempI64, op: BitOp) -> TempI64 {
        let d = self.new64();
        let one = self.c64(1);
        let bit = self.new64();
        let f = self.f();
        if op == BitOp::Ext {
            f.gen_shr_i64(d, s1, amt);
            f.gen_andi_i64(d, d, 1);
            return d;
        }
        f.gen_shl_i64(bit, one, amt);
        match op {
            BitOp::Clr => f.gen_andc_i64(d, s1, bit),
            BitOp::Inv => f.gen_xor_i64(d, s1, bit),
            BitOp::Set => f.gen_or_i64(d, s1, bit),
            BitOp::Ext => unreachable!(),
        }
        d
    }

    /// `rolw` and `rorw`, or `roriw` with a constant amount: rotate the low 32 bits and sign
    /// extend.
    fn rot_w(&mut self, rd: i32, rs1: i32, amt: TempI64, left: bool) -> bool {
        let s1 = self.gpr(rs1);
        let a32 = self.new32();
        let n32 = self.new32();
        let d = self.new64();
        let f = self.f();
        f.gen_extrl_i64_i32(a32, s1);
        f.gen_extrl_i64_i32(n32, amt);
        f.gen_andi_i32(n32, n32, 31);
        if left {
            f.gen_rotl_i32(a32, a32, n32);
        } else {
            f.gen_rotr_i32(a32, a32, n32);
        }
        f.gen_ext_i32_i64(d, a32);
        self.set_gpr(rd, d);
        true
    }

    /// `gen_lr()`.
    fn lr(&mut self, a: &arg_atomic, mop: MemOp) -> bool {
        let mop = mop | MemOp::ALIGN;
        self.decode_save_opc(0);
        let src1 = self.address(a.rs1, 0);
        if a.rl != 0 {
            self.f().gen_mb(mo::ALL | mo::BAR_STRL);
        }
        let (load_val, load_res) = (self.g().load_val, self.g().load_res);
        let idx = self.d.mem_idx;
        self.f().gen_qemu_ld_i64(load_val, src1, idx, mop);
        // TSO defines AMOs as acquire+release-RCsc, but does not define LR/SC as AMOs.
        // Instead treat them like loads.
        if a.aq != 0 || self.d.cfg.ext_ztso {
            self.f().gen_mb(mo::ALL | mo::BAR_LDAQ);
        }
        // Put addr in load_res, data in load_val.
        self.f().gen_mov_i64(load_res, src1);
        self.set_gpr(a.rd, load_val);
        true
    }

    /// `gen_sc()`.
    fn sc(&mut self, a: &arg_atomic, mop: MemOp) -> bool {
        let mop = mop | MemOp::ALIGN;
        let l1 = self.f().new_label();
        let l2 = self.f().new_label();
        self.decode_save_opc(0);
        let src1 = self.address(a.rs1, 0);
        let (load_val, load_res) = (self.g().load_val, self.g().load_res);
        self.f().gen_brcond_i64(Cond::Ne, load_res, src1, l1);

        // Note that the TCG atomic primitives are SC, so we can ignore AQ/RL along this
        // path.
        let d = self.new64();
        let s2 = self.gpr(a.rs2);
        let idx = self.d.mem_idx;
        let f = self.f();
        f.gen_atomic_cmpxchg_i64(d, load_res, load_val, s2, idx, mop);
        f.gen_setcond_i64(Cond::Ne, d, d, load_val);
        self.set_gpr(a.rd, d);
        self.f().gen_br(l2);

        self.f().gen_set_label(l1);
        // Address comparison failure. However, we still need to provide the memory barrier
        // implied by AQ/RL/TSO.
        let bar = mo::ALL
            | if a.aq != 0 { mo::BAR_LDAQ } else { 0 }
            | if a.rl != 0 || self.d.cfg.ext_ztso { mo::BAR_STRL } else { 0 };
        self.f().gen_mb(bar);
        // "For the purposes of memory protection, a failed SC.W may be treated like a
        // store." so let's check the write access permissions.
        let env = self.env();
        let size = self.c32(1 << mop.size());
        self.call(&helpers::SC_PROBE_WRITE, None, &[env.into(), src1.into(), size.into()]);
        self.set_gpri(a.rd, 1);

        self.f().gen_set_label(l2);
        // Clear the load reservation, since an SC must fail if there is an SC to any
        // address, in between an LR and SC pair.
        let load_res = self.g().load_res;
        self.f().gen_movi_i64(load_res, -1);
        true
    }

    /// `gen_amo()`. With Zama16b a word or doubleword AMO only has to stay within 16 bytes.
    fn amo(&mut self, a: &arg_atomic, op: AtomicOp, mop: MemOp) -> bool {
        let mop = if self.d.cfg.ext_zama16b && mop.size() >= 2 {
            mop | MemOp::ATOM_WITHIN16
        } else {
            mop | MemOp::ALIGN
        };
        let s2 = self.gpr(a.rs2);
        self.decode_save_opc(UW2_ALWAYS_STORE_AMO);
        let src1 = self.address(a.rs1, 0);
        let d = self.new64();
        let idx = self.d.mem_idx;
        self.f().gen_atomic_op_i64(op, d, src1, s2, idx, mop);
        self.set_gpr(a.rd, d);
        true
    }

    /// `gen_cmpxchg()` of Zacas and Zabha: compare `rd` with memory at `rs1` and store `rs2`
    /// if they are equal, `rd` getting the old value either way.
    fn cmpxchg(&mut self, a: &arg_atomic, mop: MemOp) -> bool {
        let cmpv = self.gpr(a.rd);
        let src1 = self.address(a.rs1, 0);
        let s2 = self.gpr(a.rs2);
        self.decode_save_opc(UW2_ALWAYS_STORE_AMO);
        let d = self.new64();
        let idx = self.d.mem_idx;
        self.f().gen_atomic_cmpxchg_i64(d, src1, cmpv, s2, idx, mop);
        self.set_gpr(a.rd, d);
        true
    }

    /// `trans_amocas_q()`: a 128-bit compare and swap on the register pairs `rd` and `rs2`.
    fn cmpxchg_q(&mut self, a: &arg_atomic) -> bool {
        // Encodings with odd numbered registers specified in rs2 and rd are reserved.
        if (a.rs2 | a.rd) & 1 != 0 {
            return false;
        }
        let src1 = self.address(a.rs1, 0);
        let s2l = self.gpr(a.rs2);
        let s2h = self.gpr(if a.rs2 == 0 { 0 } else { a.rs2 + 1 });
        let dl = self.gpr(a.rd);
        let dh = self.gpr(if a.rd == 0 { 0 } else { a.rd + 1 });
        let f = self.f();
        let dest = f.temp_new_i128();
        let src2 = f.temp_new_i128();
        f.gen_concat_i64_i128(src2, s2l, s2h);
        f.gen_concat_i64_i128(dest, dl, dh);
        self.decode_save_opc(UW2_ALWAYS_STORE_AMO);
        let idx = self.d.mem_idx;
        let rl = self.new64();
        let rh = self.new64();
        let f = self.f();
        f.gen_atomic_cmpxchg_i128(dest, src1, dest, src2, idx, MemOp::ALIGN | MemOp::LEUO);
        f.gen_extr_i128_i64(rl, rh, dest);
        if a.rd != 0 {
            self.set_gpr(a.rd, rl);
            self.set_gpr(a.rd + 1, rh);
        }
        true
    }

    /// `gen_load_acquire()` of Zalasr.
    fn load_acquire(&mut self, a: &arg_atomic, memop: MemOp) -> bool {
        // Check that AQ is set, as this is mandatory.
        if a.aq == 0 {
            return false;
        }
        self.decode_save_opc(0);
        let addr = self.address(a.rs1, 0);
        let memop = memop | MemOp::ALIGN | self.zama16b();
        let d = self.new64();
        let idx = self.d.mem_idx;
        self.f().gen_qemu_ld_i64(d, addr, idx, memop);
        self.set_gpr(a.rd, d);
        // Add a memory barrier implied by AQ (mandatory) and RL (optional).
        let bar = if a.rl != 0 { mo::BAR_STRL } else { 0 };
        self.f().gen_mb(mo::ALL | mo::BAR_LDAQ | bar);
        true
    }

    /// `gen_store_release()` of Zalasr.
    fn store_release(&mut self, a: &arg_atomic, memop: MemOp) -> bool {
        // Check that RL is set, as this is mandatory.
        if a.rl == 0 {
            return false;
        }
        self.decode_save_opc(0);
        let addr = self.address(a.rs1, 0);
        let data = self.gpr(a.rs2);
        let memop = memop | MemOp::ALIGN | self.zama16b();
        // Add a memory barrier implied by RL (mandatory) and AQ (optional).
        let bar = if a.aq != 0 { mo::BAR_LDAQ } else { 0 };
        self.f().gen_mb(mo::ALL | mo::BAR_STRL | bar);
        let idx = self.d.mem_idx;
        self.f().gen_qemu_st_i64(data, addr, idx, memop);
        true
    }

    /// `gen_czero()` of Zicond: `rd = rs2 cond 0 ? 0 : rs1`.
    fn czero(&mut self, a: &arg_r, cond: Cond) -> bool {
        let s1 = self.gpr(a.rs1);
        let s2 = self.gpr(a.rs2);
        let zero = self.c64(0);
        let d = self.new64();
        self.f().gen_movcond_i64(cond, d, s2, zero, zero, s1);
        self.set_gpr(a.rd, d);
        true
    }

    /// `gen_sha256()`: two rotations and `op` of the low 32 bits, XORed and sign extended.
    fn sha256(
        &mut self,
        a: &arg_r2,
        op: fn(&mut Func, TempI32, TempI32, i32),
        n: [i32; 3],
    ) -> bool {
        let s1 = self.gpr(a.rs1);
        let t0 = self.new32();
        let t1 = self.new32();
        let t2 = self.new32();
        let d = self.new64();
        let f = self.f();
        f.gen_extrl_i64_i32(t0, s1);
        f.gen_rotri_i32(t1, t0, n[0]);
        f.gen_rotri_i32(t2, t0, n[1]);
        f.gen_xor_i32(t1, t1, t2);
        op(f, t2, t0, n[2]);
        f.gen_xor_i32(t1, t1, t2);
        f.gen_ext_i32_i64(d, t1);
        self.set_gpr(a.rd, d);
        true
    }

    /// `gen_sha512_rv64()`: two rotations and `op` of `rs1`, XORed.
    fn sha512(
        &mut self,
        a: &arg_r2,
        op: fn(&mut Func, TempI64, TempI64, i64),
        n: [i64; 3],
    ) -> bool {
        let s1 = self.gpr(a.rs1);
        let t1 = self.new64();
        let t2 = self.new64();
        let f = self.f();
        f.gen_rotri_i64(t1, s1, n[0]);
        f.gen_rotri_i64(t2, s1, n[1]);
        f.gen_xor_i64(t1, t1, t2);
        op(f, t2, s1, n[2]);
        f.gen_xor_i64(t1, t1, t2);
        self.set_gpr(a.rd, t1);
        true
    }

    /// `gen_sm3()`: `x ^ rol(x, b) ^ rol(x, c)` of the low 32 bits, sign extended.
    fn sm3(&mut self, a: &arg_r2, b: i32, c: i32) -> bool {
        let s1 = self.gpr(a.rs1);
        let t0 = self.new32();
        let t1 = self.new32();
        let d = self.new64();
        let f = self.f();
        f.gen_extrl_i64_i32(t0, s1);
        f.gen_rotli_i32(t1, t0, b);
        f.gen_xor_i32(t1, t0, t1);
        f.gen_rotli_i32(t0, t0, c);
        f.gen_xor_i32(t1, t1, t0);
        f.gen_ext_i32_i64(d, t1);
        self.set_gpr(a.rd, d);
        true
    }

    /// A helper `rd = f(rs1)` without `env`.
    fn helper_r2(&mut self, a: &arg_r2, d: &Def) -> bool {
        let s1 = self.gpr(a.rs1);
        let dst = self.new64();
        self.call(d, Some(dst.into()), &[s1.into()]);
        self.set_gpr(a.rd, dst);
        true
    }

    /// A helper `rd = f(rs1, rs2, imm)` without `env`, `gen_aes32_sm4()`.
    fn helper_rri(&mut self, rd: i32, rs1: i32, rs2: i32, imm: i64, d: &Def) -> bool {
        let s1 = self.gpr(rs1);
        let s2 = self.gpr(rs2);
        let c = self.c64(imm);
        let dst = self.new64();
        self.call(d, Some(dst.into()), &[s1.into(), s2.into(), c.into()]);
        self.set_gpr(rd, dst);
        true
    }

    /// `sinval.vma`, `hinval.vvma` and `hinval.gvma` of Svinval: the TLB flush `d` when the
    /// extension `ext` (S or H) is present and the hart is not in U mode.
    fn svinval(&mut self, ext: bool, d: Option<&Def>) -> bool {
        if !self.d.cfg.ext_svinval || !ext || self.d.priv_lvl == PRV_U {
            return false;
        }
        if let Some(d) = d {
            self.decode_save_opc(0);
            let env = self.env();
            self.call(d, None, &[env.into()]);
        }
        true
    }

    /// `do_csr_post()`: the CSR access may have changed what the block was translated
    /// for, so end it.
    fn csr_post(&mut self) -> bool {
        self.end_tb_next();
        true
    }

    /// `do_csrr()`.
    fn do_csrr(&mut self, rd: i32, csr: i32) -> bool {
        self.decode_save_opc(0);
        let d = self.new64();
        let env = self.env();
        let c = self.c32(csr);
        self.call(&helpers::CSRR, Some(d.into()), &[env.into(), c.into()]);
        self.set_gpr(rd, d);
        self.csr_post()
    }

    /// `do_csrw()`.
    fn do_csrw(&mut self, csr: i32, src: TempI64) -> bool {
        self.decode_save_opc(0);
        let env = self.env();
        let c = self.c32(csr);
        self.call(&helpers::CSRW, None, &[env.into(), c.into(), src.into()]);
        self.csr_post()
    }

    /// `do_csrrw()`.
    fn do_csrrw(&mut self, rd: i32, csr: i32, src: TempI64, mask: TempI64) -> bool {
        self.decode_save_opc(0);
        let d = self.new64();
        let env = self.env();
        let c = self.c32(csr);
        self.call(
            &helpers::CSRRW,
            Some(d.into()),
            &[env.into(), c.into(), src.into(), mask.into()],
        );
        self.set_gpr(rd, d);
        self.csr_post()
    }

    /// A helper that takes `env` and the address in `rs1`, for the cache block operations.
    fn cbo(&mut self, a: &arg_decode_insn3225, d: &Def) -> bool {
        let src = self.address(a.rs1, 0);
        self.decode_save_opc(0);
        let env = self.env();
        self.call(d, None, &[env.into(), src.into()]);
        true
    }

    /// `trans_wrs_impl()`.
    fn wrs(&mut self) -> bool {
        // Clear the load reservation (if any).
        let load_res = self.g().load_res;
        self.f().gen_movi_i64(load_res, -1);
        self.end_tb_next();
        true
    }

    /// `gen_crc()` of XLRBR.
    fn crc(&mut self, a: &arg_r2, d: &Def, size: i32) -> bool {
        let s1 = self.gpr(a.rs1);
        let dst = self.new64();
        let sz = self.c32(size);
        self.call(d, Some(dst.into()), &[s1.into(), sz.into()]);
        self.set_gpr(a.rd, dst);
        true
    }

    /// A helper `rd = f(rs1, rs2)` without `env`.
    fn helper_rr(&mut self, a: &arg_r, d: &Def) -> bool {
        let s1 = self.gpr(a.rs1);
        let s2 = self.gpr(a.rs2);
        let dst = self.new64();
        self.call(d, Some(dst.into()), &[s1.into(), s2.into()]);
        self.set_gpr(a.rd, dst);
        true
    }
}

/// The Zbs operations.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BitOp {
    Clr,
    Inv,
    Set,
    Ext,
}

/// `gen_orc_b()`: each byte becomes all ones if it was non-zero, else zero.
fn gen_orc_b(f: &mut Func, d: TempI64, s: TempI64) {
    const LOW7: i64 = 0x7f7f_7f7f_7f7f_7f7f;
    let t = f.temp_new_i64();
    // Set the msb in each byte if the byte was non-zero.
    f.gen_andi_i64(t, s, LOW7);
    f.gen_addi_i64(t, t, LOW7);
    f.gen_or_i64(t, t, s);
    f.gen_andi_i64(t, t, !LOW7);
    // Extract the msb to the lsb in each byte, and spread it over the byte.
    f.gen_shri_i64(t, t, 7);
    f.gen_muli_i64(d, t, 0xff);
}

/// `gen_clzw()`.
fn gen_clzw(f: &mut Func, d: TempI64, s: TempI64) {
    let t = f.temp_new_i64();
    f.gen_shli_i64(t, s, 32);
    f.gen_clzi_i64(d, t, 32);
}

/// `gen_ctzw()`.
fn gen_ctzw(f: &mut Func, d: TempI64, s: TempI64) {
    f.gen_ori_i64(d, s, (0xffff_ffffu64 << 32) as i64);
    f.gen_ctzi_i64(d, d, 64);
}

/// `gen_cpopw()`.
fn gen_cpopw(f: &mut Func, d: TempI64, s: TempI64) {
    f.gen_ext32u_i64(d, s);
    f.gen_ctpop_i64(d, d);
}

fn gen_clz(f: &mut Func, d: TempI64, s: TempI64) {
    f.gen_clzi_i64(d, s, 64);
}

fn gen_ctz(f: &mut Func, d: TempI64, s: TempI64) {
    f.gen_ctzi_i64(d, s, 64);
}

fn gen_mulh(f: &mut Func, d: TempI64, a: TempI64, b: TempI64) {
    let lo = f.temp_new_i64();
    f.gen_muls2_i64(lo, d, a, b);
}

fn gen_mulhu(f: &mut Func, d: TempI64, a: TempI64, b: TempI64) {
    let lo = f.temp_new_i64();
    f.gen_mulu2_i64(lo, d, a, b);
}

fn gen_mulhsu(f: &mut Func, d: TempI64, a: TempI64, b: TempI64) {
    let lo = f.temp_new_i64();
    f.gen_mulsu2_i64(lo, d, a, b);
}

fn gen_slt(f: &mut Func, d: TempI64, a: TempI64, b: TempI64) {
    f.gen_setcond_i64(Cond::Lt, d, a, b);
}

fn gen_sltu(f: &mut Func, d: TempI64, a: TempI64, b: TempI64) {
    f.gen_setcond_i64(Cond::Ltu, d, a, b);
}

fn gen_slti(f: &mut Func, d: TempI64, a: TempI64, imm: i64) {
    f.gen_setcondi_i64(Cond::Lt, d, a, imm);
}

fn gen_sltiu(f: &mut Func, d: TempI64, a: TempI64, imm: i64) {
    f.gen_setcondi_i64(Cond::Ltu, d, a, imm);
}

fn gen_srliw(f: &mut Func, d: TempI64, a: TempI64, sh: u32) {
    f.gen_extract_i64(d, a, sh, 32 - sh);
}

fn gen_sraiw(f: &mut Func, d: TempI64, a: TempI64, sh: u32) {
    f.gen_sextract_i64(d, a, sh, 32 - sh);
}

fn gen_slliw(f: &mut Func, d: TempI64, a: TempI64, sh: u32) {
    f.gen_shli_i64(d, a, i64::from(sh));
}

impl DecodeInsn32 for S<'_, '_> {
    fn ex_shift_1(&mut self, x: i32) -> i32 {
        x << 1
    }

    fn ex_shift_3(&mut self, x: i32) -> i32 {
        x << 3
    }

    fn ex_plus_1(&mut self, x: i32) -> i32 {
        x + 1
    }

    fn ex_shift_12(&mut self, x: i32) -> i32 {
        x << 12
    }

    // RV64I.

    fn trans_lui(&mut self, a: &mut arg_lui) -> bool {
        self.set_gpri(a.rd, i64::from(a.imm));
        true
    }

    fn trans_auipc(&mut self, a: &mut arg_auipc) -> bool {
        let v = self.d.pc_curr.wrapping_add(i64::from(a.imm) as u64);
        self.set_gpri(a.rd, v as i64);
        true
    }

    fn trans_jal(&mut self, a: &mut arg_jal) -> bool {
        if !self.d.cfg.allow_16bit_insn() && a.imm & 3 != 0 {
            let dest = self.d.pc_curr.wrapping_add(i64::from(a.imm) as u64);
            let t = self.c64(dest as i64);
            self.gen_exception_inst_addr_mis(t);
            return true;
        }
        let succ = self.d.pc_curr.wrapping_add(self.d.cur_insn_len);
        self.set_gpri(a.rd, succ as i64);
        self.gen_goto_tb(0, i64::from(a.imm));
        true
    }

    fn trans_jalr(&mut self, a: &mut arg_jalr) -> bool {
        // A plain add: pointer masking does not apply to the jump target.
        let rs1 = self.gpr(a.rs1);
        let target = self.new64();
        self.f().gen_addi_i64(target, rs1, i64::from(a.imm));
        self.f().gen_andi_i64(target, target, -2);
        let mut misaligned = None;
        if !self.d.cfg.allow_16bit_insn() {
            let l = self.f().new_label();
            let t0 = self.new64();
            self.f().gen_andi_i64(t0, target, 2);
            self.f().gen_brcondi_i64(Cond::Ne, t0, 0, l);
            misaligned = Some(l);
        }
        let succ = self.d.pc_curr.wrapping_add(self.d.cur_insn_len);
        self.set_gpri(a.rd, succ as i64);
        let pc = self.g().pc;
        self.f().gen_mov_i64(pc, target);
        self.lookup_and_goto_ptr();
        if let Some(l) = misaligned {
            self.f().gen_set_label(l);
            self.gen_exception_inst_addr_mis(target);
        }
        true
    }

    fn trans_beq(&mut self, a: &mut arg_beq) -> bool {
        self.branch(a, Cond::Eq)
    }

    fn trans_bne(&mut self, a: &mut arg_bne) -> bool {
        self.branch(a, Cond::Ne)
    }

    fn trans_blt(&mut self, a: &mut arg_blt) -> bool {
        self.branch(a, Cond::Lt)
    }

    fn trans_bge(&mut self, a: &mut arg_bge) -> bool {
        self.branch(a, Cond::Ge)
    }

    fn trans_bltu(&mut self, a: &mut arg_bltu) -> bool {
        self.branch(a, Cond::Ltu)
    }

    fn trans_bgeu(&mut self, a: &mut arg_bgeu) -> bool {
        self.branch(a, Cond::Geu)
    }

    fn trans_lb(&mut self, a: &mut arg_lb) -> bool {
        self.load(a, MemOp::SB)
    }

    fn trans_lh(&mut self, a: &mut arg_lh) -> bool {
        self.load(a, MemOp::LESW)
    }

    fn trans_lw(&mut self, a: &mut arg_lw) -> bool {
        self.load(a, MemOp::LESL)
    }

    fn trans_ld(&mut self, a: &mut arg_ld) -> bool {
        self.load(a, MemOp::LEUQ)
    }

    fn trans_lbu(&mut self, a: &mut arg_lbu) -> bool {
        self.load(a, MemOp::UB)
    }

    fn trans_lhu(&mut self, a: &mut arg_lhu) -> bool {
        self.load(a, MemOp::LEUW)
    }

    fn trans_lwu(&mut self, a: &mut arg_lwu) -> bool {
        self.load(a, MemOp::LEUL)
    }

    fn trans_sb(&mut self, a: &mut arg_sb) -> bool {
        self.store(a, MemOp::UB)
    }

    fn trans_sh(&mut self, a: &mut arg_sh) -> bool {
        self.store(a, MemOp::LEUW)
    }

    fn trans_sw(&mut self, a: &mut arg_sw) -> bool {
        self.store(a, MemOp::LEUL)
    }

    fn trans_sd(&mut self, a: &mut arg_sd) -> bool {
        self.store(a, MemOp::LEUQ)
    }

    fn trans_addi(&mut self, a: &mut arg_addi) -> bool {
        self.arith_imm(a, Func::gen_addi_i64)
    }

    fn trans_slti(&mut self, a: &mut arg_slti) -> bool {
        self.arith_imm(a, gen_slti)
    }

    fn trans_sltiu(&mut self, a: &mut arg_sltiu) -> bool {
        self.arith_imm(a, gen_sltiu)
    }

    fn trans_xori(&mut self, a: &mut arg_xori) -> bool {
        self.arith_imm(a, Func::gen_xori_i64)
    }

    fn trans_ori(&mut self, a: &mut arg_ori) -> bool {
        self.arith_imm(a, Func::gen_ori_i64)
    }

    fn trans_andi(&mut self, a: &mut arg_andi) -> bool {
        self.arith_imm(a, Func::gen_andi_i64)
    }

    fn trans_slli(&mut self, a: &mut arg_slli) -> bool {
        self.shift_imm(a, Func::gen_shli_i64)
    }

    fn trans_srli(&mut self, a: &mut arg_srli) -> bool {
        self.shift_imm(a, Func::gen_shri_i64)
    }

    fn trans_srai(&mut self, a: &mut arg_srai) -> bool {
        self.shift_imm(a, Func::gen_sari_i64)
    }

    fn trans_add(&mut self, a: &mut arg_add) -> bool {
        self.arith(a, Func::gen_add_i64)
    }

    fn trans_sub(&mut self, a: &mut arg_sub) -> bool {
        self.arith(a, Func::gen_sub_i64)
    }

    fn trans_sll(&mut self, a: &mut arg_sll) -> bool {
        self.shift(a, Func::gen_shl_i64)
    }

    fn trans_slt(&mut self, a: &mut arg_slt) -> bool {
        self.arith(a, gen_slt)
    }

    fn trans_sltu(&mut self, a: &mut arg_sltu) -> bool {
        self.arith(a, gen_sltu)
    }

    fn trans_xor(&mut self, a: &mut arg_xor) -> bool {
        self.arith(a, Func::gen_xor_i64)
    }

    fn trans_srl(&mut self, a: &mut arg_srl) -> bool {
        self.shift(a, Func::gen_shr_i64)
    }

    fn trans_sra(&mut self, a: &mut arg_sra) -> bool {
        self.shift(a, Func::gen_sar_i64)
    }

    fn trans_or(&mut self, a: &mut arg_or) -> bool {
        self.arith(a, Func::gen_or_i64)
    }

    fn trans_and(&mut self, a: &mut arg_and) -> bool {
        self.arith(a, Func::gen_and_i64)
    }

    fn trans_addiw(&mut self, a: &mut arg_addiw) -> bool {
        let s1 = self.gpr(a.rs1);
        let d = self.new64();
        self.f().gen_addi_i64(d, s1, i64::from(a.imm));
        self.set_gpr_w(a.rd, d);
        true
    }

    fn trans_slliw(&mut self, a: &mut arg_slliw) -> bool {
        self.shift_imm_w(a, gen_slliw)
    }

    fn trans_srliw(&mut self, a: &mut arg_srliw) -> bool {
        self.shift_imm_w(a, gen_srliw)
    }

    fn trans_sraiw(&mut self, a: &mut arg_sraiw) -> bool {
        self.shift_imm_w(a, gen_sraiw)
    }

    fn trans_addw(&mut self, a: &mut arg_addw) -> bool {
        self.arith_w(a, Func::gen_add_i64)
    }

    fn trans_subw(&mut self, a: &mut arg_subw) -> bool {
        self.arith_w(a, Func::gen_sub_i64)
    }

    fn trans_sllw(&mut self, a: &mut arg_sllw) -> bool {
        let s1 = self.gpr(a.rs1);
        self.shift_w(a, s1, Func::gen_shl_i64)
    }

    fn trans_srlw(&mut self, a: &mut arg_srlw) -> bool {
        let s1 = self.gpr_u32(a.rs1);
        self.shift_w(a, s1, Func::gen_shr_i64)
    }

    fn trans_sraw(&mut self, a: &mut arg_sraw) -> bool {
        let s1 = self.gpr_s32(a.rs1);
        self.shift_w(a, s1, Func::gen_sar_i64)
    }

    fn trans_pause(&mut self, _a: &mut arg_pause) -> bool {
        if !self.d.cfg.ext_zihintpause {
            return false;
        }
        // PAUSE is a no-op in QEMU, end the TB and return to main loop.
        self.end_tb_next();
        true
    }

    fn trans_fence(&mut self, _a: &mut arg_fence) -> bool {
        // FENCE is a full memory barrier.
        self.f().gen_mb(mo::ALL | mo::BAR_SC);
        true
    }

    fn trans_fence_i(&mut self, _a: &mut arg_fence_i) -> bool {
        if !self.d.cfg.ext_zifencei {
            return false;
        }
        // FENCE_I is a no-op in QEMU, however we need to end the translation block.
        self.end_tb_next();
        true
    }

    // Zicsr.

    fn trans_csrrw(&mut self, a: &mut arg_csrrw) -> bool {
        let src = self.gpr(a.rs1);
        // If rd == 0, the insn shall not read the csr, nor cause any of the side effects
        // that might occur on a csr read.
        if a.rd == 0 {
            return self.do_csrw(a.csr, src);
        }
        let mask = self.c64(-1);
        self.do_csrrw(a.rd, a.csr, src, mask)
    }

    fn trans_csrrs(&mut self, a: &mut arg_csrrs) -> bool {
        // If rs1 == 0, the insn shall not write to the csr at all, nor cause any of the
        // side effects that might occur on a csr write. Note that if rs1 specifies a
        // register other than x0, holding a zero value, the instruction will still attempt
        // to write the unmodified value back to the csr and will cause side effects.
        if a.rs1 == 0 {
            return self.do_csrr(a.rd, a.csr);
        }
        let ones = self.c64(-1);
        let mask = self.gpr(a.rs1);
        self.do_csrrw(a.rd, a.csr, ones, mask)
    }

    fn trans_csrrc(&mut self, a: &mut arg_csrrc) -> bool {
        if a.rs1 == 0 {
            return self.do_csrr(a.rd, a.csr);
        }
        let zero = self.c64(0);
        let mask = self.gpr(a.rs1);
        self.do_csrrw(a.rd, a.csr, zero, mask)
    }

    fn trans_csrrwi(&mut self, a: &mut arg_csrrwi) -> bool {
        let src = self.c64(i64::from(a.rs1));
        if a.rd == 0 {
            return self.do_csrw(a.csr, src);
        }
        let mask = self.c64(-1);
        self.do_csrrw(a.rd, a.csr, src, mask)
    }

    fn trans_csrrsi(&mut self, a: &mut arg_csrrsi) -> bool {
        if a.rs1 == 0 {
            return self.do_csrr(a.rd, a.csr);
        }
        let ones = self.c64(-1);
        let mask = self.c64(i64::from(a.rs1));
        self.do_csrrw(a.rd, a.csr, ones, mask)
    }

    fn trans_csrrci(&mut self, a: &mut arg_csrrci) -> bool {
        if a.rs1 == 0 {
            return self.do_csrr(a.rd, a.csr);
        }
        let zero = self.c64(0);
        let mask = self.c64(i64::from(a.rs1));
        self.do_csrrw(a.rd, a.csr, zero, mask)
    }

    // Privileged instructions.

    fn trans_ecall(&mut self, _a: &mut arg_ecall) -> bool {
        // always generates U-level ECALL, fixed in do_interrupt handler
        self.generate_exception(EXCP_U_ECALL);
        true
    }

    fn trans_ebreak(&mut self, _a: &mut arg_ebreak) -> bool {
        if self.d.semihost_seq {
            self.generate_exception(EXCP_SEMIHOST);
        } else {
            let pc = self.c64(self.d.pc_curr as i64);
            let env = self.env();
            self.f().gen_st_i64(pc, env, BADADDR as i64);
            self.generate_exception(EXCP_BREAKPOINT);
        }
        true
    }

    fn trans_sret(&mut self, _a: &mut arg_sret) -> bool {
        if !self.d.cfg.has(RVS) {
            return false;
        }
        self.decode_save_opc(0);
        self.update_pc(0);
        let pc = self.g().pc;
        let env = self.env();
        self.call(&helpers::SRET, Some(pc.into()), &[env.into()]);
        // no chaining
        self.exit_tb();
        true
    }

    fn trans_mret(&mut self, _a: &mut arg_mret) -> bool {
        self.decode_save_opc(0);
        self.update_pc(0);
        let pc = self.g().pc;
        let env = self.env();
        self.call(&helpers::MRET, Some(pc.into()), &[env.into()]);
        self.exit_tb();
        true
    }

    fn trans_wfi(&mut self, _a: &mut arg_wfi) -> bool {
        self.decode_save_opc(0);
        self.update_pc(self.d.cur_insn_len as i64);
        let env = self.env();
        self.call(&helpers::WFI, None, &[env.into()]);
        true
    }

    fn trans_sfence_vma(&mut self, _a: &mut arg_sfence_vma) -> bool {
        self.decode_save_opc(0);
        let env = self.env();
        self.call(&helpers::TLB_FLUSH, None, &[env.into()]);
        true
    }

    // M.

    fn trans_mul(&mut self, a: &mut arg_mul) -> bool {
        if !(self.d.cfg.ext_zmmul || self.d.cfg.has(RVM)) {
            return false;
        }
        self.arith(a, Func::gen_mul_i64)
    }

    fn trans_mulh(&mut self, a: &mut arg_mulh) -> bool {
        if !(self.d.cfg.ext_zmmul || self.d.cfg.has(RVM)) {
            return false;
        }
        self.arith(a, gen_mulh)
    }

    fn trans_mulhsu(&mut self, a: &mut arg_mulhsu) -> bool {
        if !(self.d.cfg.ext_zmmul || self.d.cfg.has(RVM)) {
            return false;
        }
        self.arith(a, gen_mulhsu)
    }

    fn trans_mulhu(&mut self, a: &mut arg_mulhu) -> bool {
        if !(self.d.cfg.ext_zmmul || self.d.cfg.has(RVM)) {
            return false;
        }
        self.arith(a, gen_mulhu)
    }

    fn trans_div(&mut self, a: &mut arg_div) -> bool {
        if !self.d.cfg.has(RVM) {
            return false;
        }
        self.div_op(a, true, false, false)
    }

    fn trans_divu(&mut self, a: &mut arg_divu) -> bool {
        if !self.d.cfg.has(RVM) {
            return false;
        }
        self.div_op(a, false, false, false)
    }

    fn trans_rem(&mut self, a: &mut arg_rem) -> bool {
        if !self.d.cfg.has(RVM) {
            return false;
        }
        self.div_op(a, true, true, false)
    }

    fn trans_remu(&mut self, a: &mut arg_remu) -> bool {
        if !self.d.cfg.has(RVM) {
            return false;
        }
        self.div_op(a, false, true, false)
    }

    fn trans_mulw(&mut self, a: &mut arg_mulw) -> bool {
        if !(self.d.cfg.ext_zmmul || self.d.cfg.has(RVM)) {
            return false;
        }
        self.arith_w(a, Func::gen_mul_i64)
    }

    fn trans_divw(&mut self, a: &mut arg_divw) -> bool {
        if !self.d.cfg.has(RVM) {
            return false;
        }
        self.div_op(a, true, false, true)
    }

    fn trans_divuw(&mut self, a: &mut arg_divuw) -> bool {
        if !self.d.cfg.has(RVM) {
            return false;
        }
        self.div_op(a, false, false, true)
    }

    fn trans_remw(&mut self, a: &mut arg_remw) -> bool {
        if !self.d.cfg.has(RVM) {
            return false;
        }
        self.div_op(a, true, true, true)
    }

    fn trans_remuw(&mut self, a: &mut arg_remuw) -> bool {
        if !self.d.cfg.has(RVM) {
            return false;
        }
        self.div_op(a, false, true, true)
    }

    // A.

    fn trans_lr_w(&mut self, a: &mut arg_lr_w) -> bool {
        if !(self.d.cfg.ext_zalrsc || self.d.cfg.has(RVA)) {
            return false;
        }
        self.lr(a, MemOp::LESL)
    }

    fn trans_sc_w(&mut self, a: &mut arg_sc_w) -> bool {
        if !(self.d.cfg.ext_zalrsc || self.d.cfg.has(RVA)) {
            return false;
        }
        self.sc(a, MemOp::LESL)
    }

    fn trans_lr_d(&mut self, a: &mut arg_lr_d) -> bool {
        if !(self.d.cfg.ext_zalrsc || self.d.cfg.has(RVA)) {
            return false;
        }
        self.lr(a, MemOp::LEUQ)
    }

    fn trans_sc_d(&mut self, a: &mut arg_sc_d) -> bool {
        if !(self.d.cfg.ext_zalrsc || self.d.cfg.has(RVA)) {
            return false;
        }
        self.sc(a, MemOp::LEUQ)
    }

    fn trans_amoswap_w(&mut self, a: &mut arg_amoswap_w) -> bool {
        if !(self.d.cfg.ext_zaamo || self.d.cfg.has(RVA)) {
            return false;
        }
        self.amo(a, AtomicOp::Xchg, MemOp::LESL)
    }

    fn trans_amoadd_w(&mut self, a: &mut arg_amoadd_w) -> bool {
        if !(self.d.cfg.ext_zaamo || self.d.cfg.has(RVA)) {
            return false;
        }
        self.amo(a, AtomicOp::FetchAdd, MemOp::LESL)
    }

    fn trans_amoxor_w(&mut self, a: &mut arg_amoxor_w) -> bool {
        if !(self.d.cfg.ext_zaamo || self.d.cfg.has(RVA)) {
            return false;
        }
        self.amo(a, AtomicOp::FetchXor, MemOp::LESL)
    }

    fn trans_amoand_w(&mut self, a: &mut arg_amoand_w) -> bool {
        if !(self.d.cfg.ext_zaamo || self.d.cfg.has(RVA)) {
            return false;
        }
        self.amo(a, AtomicOp::FetchAnd, MemOp::LESL)
    }

    fn trans_amoor_w(&mut self, a: &mut arg_amoor_w) -> bool {
        if !(self.d.cfg.ext_zaamo || self.d.cfg.has(RVA)) {
            return false;
        }
        self.amo(a, AtomicOp::FetchOr, MemOp::LESL)
    }

    fn trans_amomin_w(&mut self, a: &mut arg_amomin_w) -> bool {
        if !(self.d.cfg.ext_zaamo || self.d.cfg.has(RVA)) {
            return false;
        }
        self.amo(a, AtomicOp::FetchSmin, MemOp::LESL)
    }

    fn trans_amomax_w(&mut self, a: &mut arg_amomax_w) -> bool {
        if !(self.d.cfg.ext_zaamo || self.d.cfg.has(RVA)) {
            return false;
        }
        self.amo(a, AtomicOp::FetchSmax, MemOp::LESL)
    }

    fn trans_amominu_w(&mut self, a: &mut arg_amominu_w) -> bool {
        if !(self.d.cfg.ext_zaamo || self.d.cfg.has(RVA)) {
            return false;
        }
        self.amo(a, AtomicOp::FetchUmin, MemOp::LESL)
    }

    fn trans_amomaxu_w(&mut self, a: &mut arg_amomaxu_w) -> bool {
        if !(self.d.cfg.ext_zaamo || self.d.cfg.has(RVA)) {
            return false;
        }
        self.amo(a, AtomicOp::FetchUmax, MemOp::LESL)
    }

    fn trans_amoswap_d(&mut self, a: &mut arg_amoswap_d) -> bool {
        if !(self.d.cfg.ext_zaamo || self.d.cfg.has(RVA)) {
            return false;
        }
        self.amo(a, AtomicOp::Xchg, MemOp::LEUQ)
    }

    fn trans_amoadd_d(&mut self, a: &mut arg_amoadd_d) -> bool {
        if !(self.d.cfg.ext_zaamo || self.d.cfg.has(RVA)) {
            return false;
        }
        self.amo(a, AtomicOp::FetchAdd, MemOp::LEUQ)
    }

    fn trans_amoxor_d(&mut self, a: &mut arg_amoxor_d) -> bool {
        if !(self.d.cfg.ext_zaamo || self.d.cfg.has(RVA)) {
            return false;
        }
        self.amo(a, AtomicOp::FetchXor, MemOp::LEUQ)
    }

    fn trans_amoand_d(&mut self, a: &mut arg_amoand_d) -> bool {
        if !(self.d.cfg.ext_zaamo || self.d.cfg.has(RVA)) {
            return false;
        }
        self.amo(a, AtomicOp::FetchAnd, MemOp::LEUQ)
    }

    fn trans_amoor_d(&mut self, a: &mut arg_amoor_d) -> bool {
        if !(self.d.cfg.ext_zaamo || self.d.cfg.has(RVA)) {
            return false;
        }
        self.amo(a, AtomicOp::FetchOr, MemOp::LEUQ)
    }

    fn trans_amomin_d(&mut self, a: &mut arg_amomin_d) -> bool {
        if !(self.d.cfg.ext_zaamo || self.d.cfg.has(RVA)) {
            return false;
        }
        self.amo(a, AtomicOp::FetchSmin, MemOp::LEUQ)
    }

    fn trans_amomax_d(&mut self, a: &mut arg_amomax_d) -> bool {
        if !(self.d.cfg.ext_zaamo || self.d.cfg.has(RVA)) {
            return false;
        }
        self.amo(a, AtomicOp::FetchSmax, MemOp::LEUQ)
    }

    fn trans_amominu_d(&mut self, a: &mut arg_amominu_d) -> bool {
        if !(self.d.cfg.ext_zaamo || self.d.cfg.has(RVA)) {
            return false;
        }
        self.amo(a, AtomicOp::FetchUmin, MemOp::LEUQ)
    }

    fn trans_amomaxu_d(&mut self, a: &mut arg_amomaxu_d) -> bool {
        if !(self.d.cfg.ext_zaamo || self.d.cfg.has(RVA)) {
            return false;
        }
        self.amo(a, AtomicOp::FetchUmax, MemOp::LEUQ)
    }

    // Zawrs.

    fn trans_wrs_nto(&mut self, _a: &mut arg_wrs_nto) -> bool {
        if !self.d.cfg.ext_zawrs {
            return false;
        }
        // We only get here when wrs.nto is not reached by an interrupt; it may still be
        // illegal for the privilege level.
        self.decode_save_opc(0);
        let env = self.env();
        self.call(&helpers::WRS_NTO, None, &[env.into()]);
        self.wrs()
    }

    fn trans_wrs_sto(&mut self, _a: &mut arg_wrs_sto) -> bool {
        if !self.d.cfg.ext_zawrs {
            return false;
        }
        self.wrs()
    }

    // Zicbom and Zicboz.

    fn trans_cbo_clean(&mut self, a: &mut arg_cbo_clean) -> bool {
        if !self.d.cfg.ext_zicbom {
            return false;
        }
        self.cbo(a, &helpers::CBO_CLEAN_FLUSH)
    }

    fn trans_cbo_flush(&mut self, a: &mut arg_cbo_flush) -> bool {
        if !self.d.cfg.ext_zicbom {
            return false;
        }
        self.cbo(a, &helpers::CBO_CLEAN_FLUSH)
    }

    fn trans_cbo_inval(&mut self, a: &mut arg_cbo_inval) -> bool {
        if !self.d.cfg.ext_zicbom {
            return false;
        }
        self.cbo(a, &helpers::CBO_INVAL)
    }

    fn trans_cbo_zero(&mut self, a: &mut arg_cbo_zero) -> bool {
        if !self.d.cfg.ext_zicboz {
            return false;
        }
        self.cbo(a, &helpers::CBO_ZERO)
    }

    // Zba.

    fn trans_sh1add(&mut self, a: &mut arg_sh1add) -> bool {
        if !self.d.cfg.ext_zba {
            return false;
        }
        self.shadd(a, 1, false)
    }

    fn trans_sh2add(&mut self, a: &mut arg_sh2add) -> bool {
        if !self.d.cfg.ext_zba {
            return false;
        }
        self.shadd(a, 2, false)
    }

    fn trans_sh3add(&mut self, a: &mut arg_sh3add) -> bool {
        if !self.d.cfg.ext_zba {
            return false;
        }
        self.shadd(a, 3, false)
    }

    fn trans_add_uw(&mut self, a: &mut arg_add_uw) -> bool {
        if !self.d.cfg.ext_zba {
            return false;
        }
        self.shadd(a, 0, true)
    }

    fn trans_sh1add_uw(&mut self, a: &mut arg_sh1add_uw) -> bool {
        if !self.d.cfg.ext_zba {
            return false;
        }
        self.shadd(a, 1, true)
    }

    fn trans_sh2add_uw(&mut self, a: &mut arg_sh2add_uw) -> bool {
        if !self.d.cfg.ext_zba {
            return false;
        }
        self.shadd(a, 2, true)
    }

    fn trans_sh3add_uw(&mut self, a: &mut arg_sh3add_uw) -> bool {
        if !self.d.cfg.ext_zba {
            return false;
        }
        self.shadd(a, 3, true)
    }

    fn trans_slli_uw(&mut self, a: &mut arg_slli_uw) -> bool {
        if !self.d.cfg.ext_zba {
            return false;
        }
        if a.shamt >= 64 {
            return false;
        }
        let s1 = self.gpr(a.rs1);
        let d = self.new64();
        let f = self.f();
        if a.shamt < 32 {
            f.gen_deposit_z_i64(d, s1, a.shamt as u32, 32);
        } else {
            f.gen_shli_i64(d, s1, i64::from(a.shamt));
        }
        self.set_gpr(a.rd, d);
        true
    }

    // Zbb.

    fn trans_andn(&mut self, a: &mut arg_andn) -> bool {
        if !(self.d.cfg.ext_zbb || self.d.cfg.ext_zbkb) {
            return false;
        }
        self.arith(a, Func::gen_andc_i64)
    }

    fn trans_orn(&mut self, a: &mut arg_orn) -> bool {
        if !(self.d.cfg.ext_zbb || self.d.cfg.ext_zbkb) {
            return false;
        }
        self.arith(a, Func::gen_orc_i64)
    }

    fn trans_xnor(&mut self, a: &mut arg_xnor) -> bool {
        if !(self.d.cfg.ext_zbb || self.d.cfg.ext_zbkb) {
            return false;
        }
        self.arith(a, Func::gen_eqv_i64)
    }

    fn trans_clz(&mut self, a: &mut arg_clz) -> bool {
        if !self.d.cfg.ext_zbb {
            return false;
        }
        self.unary(a, gen_clz)
    }

    fn trans_ctz(&mut self, a: &mut arg_ctz) -> bool {
        if !self.d.cfg.ext_zbb {
            return false;
        }
        self.unary(a, gen_ctz)
    }

    fn trans_cpop(&mut self, a: &mut arg_cpop) -> bool {
        if !self.d.cfg.ext_zbb {
            return false;
        }
        self.unary(a, Func::gen_ctpop_i64)
    }

    fn trans_clzw(&mut self, a: &mut arg_clzw) -> bool {
        if !self.d.cfg.ext_zbb {
            return false;
        }
        self.unary(a, gen_clzw)
    }

    fn trans_ctzw(&mut self, a: &mut arg_ctzw) -> bool {
        if !self.d.cfg.ext_zbb {
            return false;
        }
        self.unary(a, gen_ctzw)
    }

    fn trans_cpopw(&mut self, a: &mut arg_cpopw) -> bool {
        if !self.d.cfg.ext_zbb {
            return false;
        }
        self.unary(a, gen_cpopw)
    }

    fn trans_max(&mut self, a: &mut arg_max) -> bool {
        if !self.d.cfg.ext_zbb {
            return false;
        }
        self.arith(a, Func::gen_smax_i64)
    }

    fn trans_maxu(&mut self, a: &mut arg_maxu) -> bool {
        if !self.d.cfg.ext_zbb {
            return false;
        }
        self.arith(a, Func::gen_umax_i64)
    }

    fn trans_min(&mut self, a: &mut arg_min) -> bool {
        if !self.d.cfg.ext_zbb {
            return false;
        }
        self.arith(a, Func::gen_smin_i64)
    }

    fn trans_minu(&mut self, a: &mut arg_minu) -> bool {
        if !self.d.cfg.ext_zbb {
            return false;
        }
        self.arith(a, Func::gen_umin_i64)
    }

    fn trans_sext_b(&mut self, a: &mut arg_sext_b) -> bool {
        if !self.d.cfg.ext_zbb {
            return false;
        }
        self.unary(a, Func::gen_ext8s_i64)
    }

    fn trans_sext_h(&mut self, a: &mut arg_sext_h) -> bool {
        if !self.d.cfg.ext_zbb {
            return false;
        }
        self.unary(a, Func::gen_ext16s_i64)
    }

    fn trans_zext_h_64(&mut self, a: &mut arg_zext_h_64) -> bool {
        if !self.d.cfg.ext_zbb {
            return false;
        }
        self.unary(a, Func::gen_ext16u_i64)
    }

    fn trans_rev8_64(&mut self, a: &mut arg_rev8_64) -> bool {
        if !(self.d.cfg.ext_zbb || self.d.cfg.ext_zbkb) {
            return false;
        }
        self.unary(a, Func::gen_bswap64_i64)
    }

    fn trans_orc_b(&mut self, a: &mut arg_orc_b) -> bool {
        if !self.d.cfg.ext_zbb {
            return false;
        }
        self.unary(a, gen_orc_b)
    }

    fn trans_rol(&mut self, a: &mut arg_rol) -> bool {
        if !(self.d.cfg.ext_zbb || self.d.cfg.ext_zbkb) {
            return false;
        }
        self.shift(a, Func::gen_rotl_i64)
    }

    fn trans_ror(&mut self, a: &mut arg_ror) -> bool {
        if !(self.d.cfg.ext_zbb || self.d.cfg.ext_zbkb) {
            return false;
        }
        self.shift(a, Func::gen_rotr_i64)
    }

    fn trans_rori(&mut self, a: &mut arg_rori) -> bool {
        if !(self.d.cfg.ext_zbb || self.d.cfg.ext_zbkb) {
            return false;
        }
        self.shift_imm(a, Func::gen_rotri_i64)
    }

    fn trans_rolw(&mut self, a: &mut arg_rolw) -> bool {
        if !(self.d.cfg.ext_zbb || self.d.cfg.ext_zbkb) {
            return false;
        }
        let amt = self.gpr(a.rs2);
        self.rot_w(a.rd, a.rs1, amt, true)
    }

    fn trans_rorw(&mut self, a: &mut arg_rorw) -> bool {
        if !(self.d.cfg.ext_zbb || self.d.cfg.ext_zbkb) {
            return false;
        }
        let amt = self.gpr(a.rs2);
        self.rot_w(a.rd, a.rs1, amt, false)
    }

    fn trans_roriw(&mut self, a: &mut arg_roriw) -> bool {
        if !(self.d.cfg.ext_zbb || self.d.cfg.ext_zbkb) {
            return false;
        }
        if a.shamt >= 32 {
            return false;
        }
        let amt = self.c64(i64::from(a.shamt));
        self.rot_w(a.rd, a.rs1, amt, false)
    }

    // Zbc.

    fn trans_clmul(&mut self, a: &mut arg_clmul) -> bool {
        if !(self.d.cfg.ext_zbc || self.d.cfg.ext_zbkc) {
            return false;
        }
        self.helper_rr(a, &helpers::CLMUL)
    }

    fn trans_clmulh(&mut self, a: &mut arg_clmulh) -> bool {
        if !(self.d.cfg.ext_zbc || self.d.cfg.ext_zbkc) {
            return false;
        }
        // clmulh is clmulr shifted right by one.
        let s1 = self.gpr(a.rs1);
        let s2 = self.gpr(a.rs2);
        let d = self.new64();
        self.call(&helpers::CLMULR, Some(d.into()), &[s1.into(), s2.into()]);
        self.f().gen_shri_i64(d, d, 1);
        self.set_gpr(a.rd, d);
        true
    }

    fn trans_clmulr(&mut self, a: &mut arg_clmulr) -> bool {
        if !self.d.cfg.ext_zbc {
            return false;
        }
        self.helper_rr(a, &helpers::CLMULR)
    }

    // Zbs.

    fn trans_bclr(&mut self, a: &mut arg_bclr) -> bool {
        if !self.d.cfg.ext_zbs {
            return false;
        }
        self.bit_op(a, BitOp::Clr)
    }

    fn trans_bclri(&mut self, a: &mut arg_bclri) -> bool {
        if !self.d.cfg.ext_zbs {
            return false;
        }
        self.bit_op_imm(a, BitOp::Clr)
    }

    fn trans_bext(&mut self, a: &mut arg_bext) -> bool {
        if !self.d.cfg.ext_zbs {
            return false;
        }
        self.bit_op(a, BitOp::Ext)
    }

    fn trans_bexti(&mut self, a: &mut arg_bexti) -> bool {
        if !self.d.cfg.ext_zbs {
            return false;
        }
        self.bit_op_imm(a, BitOp::Ext)
    }

    fn trans_binv(&mut self, a: &mut arg_binv) -> bool {
        if !self.d.cfg.ext_zbs {
            return false;
        }
        self.bit_op(a, BitOp::Inv)
    }

    fn trans_binvi(&mut self, a: &mut arg_binvi) -> bool {
        if !self.d.cfg.ext_zbs {
            return false;
        }
        self.bit_op_imm(a, BitOp::Inv)
    }

    fn trans_bset(&mut self, a: &mut arg_bset) -> bool {
        if !self.d.cfg.ext_zbs {
            return false;
        }
        self.bit_op(a, BitOp::Set)
    }

    fn trans_bseti(&mut self, a: &mut arg_bseti) -> bool {
        if !self.d.cfg.ext_zbs {
            return false;
        }
        self.bit_op_imm(a, BitOp::Set)
    }

    // Zbkb.

    fn trans_pack(&mut self, a: &mut arg_pack) -> bool {
        if !self.d.cfg.ext_zbkb {
            return false;
        }
        self.arith(a, |f, d, a, b| f.gen_deposit_i64(d, a, b, 32, 32))
    }

    fn trans_packh(&mut self, a: &mut arg_packh) -> bool {
        if !self.d.cfg.ext_zbkb {
            return false;
        }
        self.arith(a, |f, d, a, b| {
            let t = f.temp_new_i64();
            f.gen_ext8u_i64(t, b);
            f.gen_deposit_i64(d, a, t, 8, 56);
        })
    }

    fn trans_packw(&mut self, a: &mut arg_packw) -> bool {
        if !self.d.cfg.ext_zbkb {
            return false;
        }
        self.arith(a, |f, d, a, b| {
            let t = f.temp_new_i64();
            f.gen_ext16s_i64(t, b);
            f.gen_deposit_i64(d, a, t, 16, 48);
        })
    }

    fn trans_brev8(&mut self, a: &mut arg_brev8) -> bool {
        if !self.d.cfg.ext_zbkb {
            return false;
        }
        self.helper_r2(a, &helpers::BREV8)
    }

    // Zbkx.

    fn trans_xperm4(&mut self, a: &mut arg_xperm4) -> bool {
        if !self.d.cfg.ext_zbkx {
            return false;
        }
        self.helper_rr(a, &helpers::XPERM4)
    }

    fn trans_xperm8(&mut self, a: &mut arg_xperm8) -> bool {
        if !self.d.cfg.ext_zbkx {
            return false;
        }
        self.helper_rr(a, &helpers::XPERM8)
    }

    // Zknd, Zkne, Zknh, Zksed and Zksh. The RV32 only instructions keep the default.

    fn trans_aes64es(&mut self, a: &mut arg_aes64es) -> bool {
        if !self.d.cfg.ext_zkne {
            return false;
        }
        self.helper_rr(a, &crypto::AES64ES)
    }

    fn trans_aes64esm(&mut self, a: &mut arg_aes64esm) -> bool {
        if !self.d.cfg.ext_zkne {
            return false;
        }
        self.helper_rr(a, &crypto::AES64ESM)
    }

    fn trans_aes64ds(&mut self, a: &mut arg_aes64ds) -> bool {
        if !self.d.cfg.ext_zknd {
            return false;
        }
        self.helper_rr(a, &crypto::AES64DS)
    }

    fn trans_aes64dsm(&mut self, a: &mut arg_aes64dsm) -> bool {
        if !self.d.cfg.ext_zknd {
            return false;
        }
        self.helper_rr(a, &crypto::AES64DSM)
    }

    fn trans_aes64ks2(&mut self, a: &mut arg_aes64ks2) -> bool {
        if !(self.d.cfg.ext_zknd || self.d.cfg.ext_zkne) {
            return false;
        }
        self.helper_rr(a, &crypto::AES64KS2)
    }

    fn trans_aes64ks1i(&mut self, a: &mut arg_aes64ks1i) -> bool {
        if !(self.d.cfg.ext_zknd || self.d.cfg.ext_zkne) {
            return false;
        }
        if a.imm > 0xa {
            return false;
        }
        let s1 = self.gpr(a.rs1);
        let c = self.c64(i64::from(a.imm));
        let dst = self.new64();
        self.call(&crypto::AES64KS1I, Some(dst.into()), &[s1.into(), c.into()]);
        self.set_gpr(a.rd, dst);
        true
    }

    fn trans_aes64im(&mut self, a: &mut arg_aes64im) -> bool {
        if !self.d.cfg.ext_zknd {
            return false;
        }
        self.helper_r2(a, &crypto::AES64IM)
    }

    fn trans_sha256sig0(&mut self, a: &mut arg_sha256sig0) -> bool {
        if !self.d.cfg.ext_zknh {
            return false;
        }
        self.sha256(a, Func::gen_shri_i32, [7, 18, 3])
    }

    fn trans_sha256sig1(&mut self, a: &mut arg_sha256sig1) -> bool {
        if !self.d.cfg.ext_zknh {
            return false;
        }
        self.sha256(a, Func::gen_shri_i32, [17, 19, 10])
    }

    fn trans_sha256sum0(&mut self, a: &mut arg_sha256sum0) -> bool {
        if !self.d.cfg.ext_zknh {
            return false;
        }
        self.sha256(a, Func::gen_rotri_i32, [2, 13, 22])
    }

    fn trans_sha256sum1(&mut self, a: &mut arg_sha256sum1) -> bool {
        if !self.d.cfg.ext_zknh {
            return false;
        }
        self.sha256(a, Func::gen_rotri_i32, [6, 11, 25])
    }

    fn trans_sha512sig0(&mut self, a: &mut arg_sha512sig0) -> bool {
        if !self.d.cfg.ext_zknh {
            return false;
        }
        self.sha512(a, Func::gen_shri_i64, [1, 8, 7])
    }

    fn trans_sha512sig1(&mut self, a: &mut arg_sha512sig1) -> bool {
        if !self.d.cfg.ext_zknh {
            return false;
        }
        self.sha512(a, Func::gen_shri_i64, [19, 61, 6])
    }

    fn trans_sha512sum0(&mut self, a: &mut arg_sha512sum0) -> bool {
        if !self.d.cfg.ext_zknh {
            return false;
        }
        self.sha512(a, Func::gen_rotri_i64, [28, 34, 39])
    }

    fn trans_sha512sum1(&mut self, a: &mut arg_sha512sum1) -> bool {
        if !self.d.cfg.ext_zknh {
            return false;
        }
        self.sha512(a, Func::gen_rotri_i64, [14, 18, 41])
    }

    fn trans_sm3p0(&mut self, a: &mut arg_sm3p0) -> bool {
        if !self.d.cfg.ext_zksh {
            return false;
        }
        self.sm3(a, 9, 17)
    }

    fn trans_sm3p1(&mut self, a: &mut arg_sm3p1) -> bool {
        if !self.d.cfg.ext_zksh {
            return false;
        }
        self.sm3(a, 15, 23)
    }

    fn trans_sm4ed(&mut self, a: &mut arg_sm4ed) -> bool {
        if !self.d.cfg.ext_zksed {
            return false;
        }
        self.helper_rri(a.rd, a.rs1, a.rs2, i64::from(a.shamt), &crypto::SM4ED)
    }

    fn trans_sm4ks(&mut self, a: &mut arg_sm4ks) -> bool {
        if !self.d.cfg.ext_zksed {
            return false;
        }
        self.helper_rri(a.rd, a.rs1, a.rs2, i64::from(a.shamt), &crypto::SM4KS)
    }

    // Zicond.

    fn trans_czero_eqz(&mut self, a: &mut arg_czero_eqz) -> bool {
        if !self.d.cfg.ext_zicond {
            return false;
        }
        self.czero(a, Cond::Eq)
    }

    fn trans_czero_nez(&mut self, a: &mut arg_czero_nez) -> bool {
        if !self.d.cfg.ext_zicond {
            return false;
        }
        self.czero(a, Cond::Ne)
    }

    // Zimop.

    fn trans_mop_r_n(&mut self, a: &mut arg_mop_r_n) -> bool {
        if !self.d.cfg.ext_zimop {
            return false;
        }
        self.set_gpri(a.rd, 0);
        true
    }

    fn trans_mop_rr_n(&mut self, a: &mut arg_mop_rr_n) -> bool {
        if !self.d.cfg.ext_zimop {
            return false;
        }
        self.set_gpri(a.rd, 0);
        true
    }

    // Svinval.

    fn trans_sinval_vma(&mut self, _a: &mut arg_sinval_vma) -> bool {
        // Do the same as sfence.vma currently.
        let rvs = self.d.cfg.has(RVS);
        self.svinval(rvs, Some(&helpers::TLB_FLUSH))
    }

    fn trans_sfence_w_inval(&mut self, _a: &mut arg_sfence_w_inval) -> bool {
        // Do nothing currently.
        let rvs = self.d.cfg.has(RVS);
        self.svinval(rvs, None)
    }

    fn trans_sfence_inval_ir(&mut self, _a: &mut arg_sfence_inval_ir) -> bool {
        // Do nothing currently.
        let rvs = self.d.cfg.has(RVS);
        self.svinval(rvs, None)
    }

    fn trans_hinval_vvma(&mut self, _a: &mut arg_hinval_vvma) -> bool {
        // Do the same as hfence.vvma currently.
        let rvh = self.d.cfg.ext_h();
        self.svinval(rvh, Some(&helpers::HYP_TLB_FLUSH))
    }

    fn trans_hinval_gvma(&mut self, _a: &mut arg_hinval_gvma) -> bool {
        // Do the same as hfence.gvma currently.
        let rvh = self.d.cfg.ext_h();
        self.svinval(rvh, Some(&helpers::HYP_GVMA_TLB_FLUSH))
    }

    // Zacas.

    fn trans_amocas_w(&mut self, a: &mut arg_amocas_w) -> bool {
        if !self.d.cfg.ext_zacas {
            return false;
        }
        self.cmpxchg(a, MemOp::ALIGN | MemOp::LESL)
    }

    fn trans_amocas_d(&mut self, a: &mut arg_amocas_d) -> bool {
        if !self.d.cfg.ext_zacas {
            return false;
        }
        self.cmpxchg(a, MemOp::ALIGN | MemOp::LEUQ)
    }

    fn trans_amocas_q(&mut self, a: &mut arg_amocas_q) -> bool {
        if !self.d.cfg.ext_zacas {
            return false;
        }
        self.cmpxchg_q(a)
    }

    // Zabha.

    fn trans_amoswap_b(&mut self, a: &mut arg_amoswap_b) -> bool {
        self.d.cfg.ext_zabha && self.amo(a, AtomicOp::Xchg, MemOp::SB)
    }

    fn trans_amoadd_b(&mut self, a: &mut arg_amoadd_b) -> bool {
        self.d.cfg.ext_zabha && self.amo(a, AtomicOp::FetchAdd, MemOp::SB)
    }

    fn trans_amoxor_b(&mut self, a: &mut arg_amoxor_b) -> bool {
        self.d.cfg.ext_zabha && self.amo(a, AtomicOp::FetchXor, MemOp::SB)
    }

    fn trans_amoand_b(&mut self, a: &mut arg_amoand_b) -> bool {
        self.d.cfg.ext_zabha && self.amo(a, AtomicOp::FetchAnd, MemOp::SB)
    }

    fn trans_amoor_b(&mut self, a: &mut arg_amoor_b) -> bool {
        self.d.cfg.ext_zabha && self.amo(a, AtomicOp::FetchOr, MemOp::SB)
    }

    fn trans_amomin_b(&mut self, a: &mut arg_amomin_b) -> bool {
        self.d.cfg.ext_zabha && self.amo(a, AtomicOp::FetchSmin, MemOp::SB)
    }

    fn trans_amomax_b(&mut self, a: &mut arg_amomax_b) -> bool {
        self.d.cfg.ext_zabha && self.amo(a, AtomicOp::FetchSmax, MemOp::SB)
    }

    fn trans_amominu_b(&mut self, a: &mut arg_amominu_b) -> bool {
        self.d.cfg.ext_zabha && self.amo(a, AtomicOp::FetchUmin, MemOp::SB)
    }

    fn trans_amomaxu_b(&mut self, a: &mut arg_amomaxu_b) -> bool {
        self.d.cfg.ext_zabha && self.amo(a, AtomicOp::FetchUmax, MemOp::SB)
    }

    fn trans_amoswap_h(&mut self, a: &mut arg_amoswap_h) -> bool {
        self.d.cfg.ext_zabha && self.amo(a, AtomicOp::Xchg, MemOp::LESW)
    }

    fn trans_amoadd_h(&mut self, a: &mut arg_amoadd_h) -> bool {
        self.d.cfg.ext_zabha && self.amo(a, AtomicOp::FetchAdd, MemOp::LESW)
    }

    fn trans_amoxor_h(&mut self, a: &mut arg_amoxor_h) -> bool {
        self.d.cfg.ext_zabha && self.amo(a, AtomicOp::FetchXor, MemOp::LESW)
    }

    fn trans_amoand_h(&mut self, a: &mut arg_amoand_h) -> bool {
        self.d.cfg.ext_zabha && self.amo(a, AtomicOp::FetchAnd, MemOp::LESW)
    }

    fn trans_amoor_h(&mut self, a: &mut arg_amoor_h) -> bool {
        self.d.cfg.ext_zabha && self.amo(a, AtomicOp::FetchOr, MemOp::LESW)
    }

    fn trans_amomin_h(&mut self, a: &mut arg_amomin_h) -> bool {
        self.d.cfg.ext_zabha && self.amo(a, AtomicOp::FetchSmin, MemOp::LESW)
    }

    fn trans_amomax_h(&mut self, a: &mut arg_amomax_h) -> bool {
        self.d.cfg.ext_zabha && self.amo(a, AtomicOp::FetchSmax, MemOp::LESW)
    }

    fn trans_amominu_h(&mut self, a: &mut arg_amominu_h) -> bool {
        self.d.cfg.ext_zabha && self.amo(a, AtomicOp::FetchUmin, MemOp::LESW)
    }

    fn trans_amomaxu_h(&mut self, a: &mut arg_amomaxu_h) -> bool {
        self.d.cfg.ext_zabha && self.amo(a, AtomicOp::FetchUmax, MemOp::LESW)
    }

    fn trans_amocas_b(&mut self, a: &mut arg_amocas_b) -> bool {
        if !(self.d.cfg.ext_zacas && self.d.cfg.ext_zabha) {
            return false;
        }
        self.cmpxchg(a, MemOp::SB)
    }

    fn trans_amocas_h(&mut self, a: &mut arg_amocas_h) -> bool {
        if !(self.d.cfg.ext_zacas && self.d.cfg.ext_zabha) {
            return false;
        }
        self.cmpxchg(a, MemOp::ALIGN | MemOp::LESW)
    }

    // Zalasr.

    fn trans_lb_aqrl(&mut self, a: &mut arg_lb_aqrl) -> bool {
        self.d.cfg.ext_zalasr && self.load_acquire(a, MemOp::SB)
    }

    fn trans_lh_aqrl(&mut self, a: &mut arg_lh_aqrl) -> bool {
        self.d.cfg.ext_zalasr && self.load_acquire(a, MemOp::LESW)
    }

    fn trans_lw_aqrl(&mut self, a: &mut arg_lw_aqrl) -> bool {
        self.d.cfg.ext_zalasr && self.load_acquire(a, MemOp::LESL)
    }

    fn trans_ld_aqrl(&mut self, a: &mut arg_ld_aqrl) -> bool {
        self.d.cfg.ext_zalasr && self.load_acquire(a, MemOp::LEUQ)
    }

    fn trans_sb_aqrl(&mut self, a: &mut arg_sb_aqrl) -> bool {
        self.d.cfg.ext_zalasr && self.store_release(a, MemOp::SB)
    }

    fn trans_sh_aqrl(&mut self, a: &mut arg_sh_aqrl) -> bool {
        self.d.cfg.ext_zalasr && self.store_release(a, MemOp::LESW)
    }

    fn trans_sw_aqrl(&mut self, a: &mut arg_sw_aqrl) -> bool {
        self.d.cfg.ext_zalasr && self.store_release(a, MemOp::LESL)
    }

    fn trans_sd_aqrl(&mut self, a: &mut arg_sd_aqrl) -> bool {
        self.d.cfg.ext_zalasr && self.store_release(a, MemOp::LEUQ)
    }

    translate_fp::fp_trans32!();
    translate_rvv::rvv_trans32!();
    translate_rvh::rvh_trans32!();
    translate_rvv_int::rvv_int_trans32!();
    translate_rvv_fp::rvv_fp_trans32!();
    translate_rvv_perm::rvv_perm_trans32!();
    translate_rvvk::rvvk_trans32!();
}

impl DecodeInsn16 for S<'_, '_> {
    fn ex_shift_4(&mut self, x: i32) -> i32 {
        x << 4
    }

    fn ex_shift_2(&mut self, x: i32) -> i32 {
        x << 2
    }

    fn ex_rvc_register(&mut self, x: i32) -> i32 {
        x + 8
    }

    fn ex_rvc_shiftri(&mut self, x: i32) -> i32 {
        x
    }

    fn ex_rvc_shiftli(&mut self, x: i32) -> i32 {
        x
    }

    fn ex_sreg_register(&mut self, x: i32) -> i32 {
        // `sreg_register()` of Zcmp: s0, s1, then s2 to s7.
        if x < 2 { x + 8 } else { x + 16 }
    }

    fn trans_illegal(&mut self, _a: &mut arg_empty) -> bool {
        self.gen_exception_illegal();
        true
    }

    fn trans_c64_illegal(&mut self, _a: &mut arg_empty) -> bool {
        self.gen_exception_illegal();
        true
    }

    // Zcmop.

    fn trans_c_mop_n(&mut self, _a: &mut arg_c_mop_n) -> bool {
        self.d.cfg.ext_zcmop
    }

    // Zcb.

    fn trans_c_zext_b(&mut self, a: &mut arg_r2) -> bool {
        self.d.cfg.ext_zcb && self.unary(a, Func::gen_ext8u_i64)
    }

    fn trans_c_zext_h(&mut self, a: &mut arg_r2) -> bool {
        self.d.cfg.ext_zcb && self.d.cfg.ext_zbb && self.unary(a, Func::gen_ext16u_i64)
    }

    fn trans_c_sext_b(&mut self, a: &mut arg_r2) -> bool {
        self.d.cfg.ext_zcb && self.d.cfg.ext_zbb && self.unary(a, Func::gen_ext8s_i64)
    }

    fn trans_c_sext_h(&mut self, a: &mut arg_r2) -> bool {
        self.d.cfg.ext_zcb && self.d.cfg.ext_zbb && self.unary(a, Func::gen_ext16s_i64)
    }

    fn trans_c_zext_w(&mut self, a: &mut arg_r2) -> bool {
        self.d.cfg.ext_zcb && self.d.cfg.ext_zba && self.unary(a, Func::gen_ext32u_i64)
    }

    fn trans_c_not(&mut self, a: &mut arg_r2) -> bool {
        self.d.cfg.ext_zcb && self.unary(a, Func::gen_not_i64)
    }

    fn trans_c_mul(&mut self, a: &mut arg_r) -> bool {
        self.d.cfg.ext_zcb
            && (self.d.cfg.has(RVM) || self.d.cfg.ext_zmmul)
            && self.arith(a, Func::gen_mul_i64)
    }

    fn trans_c_lbu(&mut self, a: &mut arg_i) -> bool {
        self.d.cfg.ext_zcb && self.load(a, MemOp::UB)
    }

    fn trans_c_lhu(&mut self, a: &mut arg_i) -> bool {
        self.d.cfg.ext_zcb && self.load(a, MemOp::LEUW)
    }

    fn trans_c_lh(&mut self, a: &mut arg_i) -> bool {
        self.d.cfg.ext_zcb && self.load(a, MemOp::LESW)
    }

    fn trans_c_sb(&mut self, a: &mut arg_s) -> bool {
        self.d.cfg.ext_zcb && self.store(a, MemOp::UB)
    }

    fn trans_c_sh(&mut self, a: &mut arg_s) -> bool {
        self.d.cfg.ext_zcb && self.store(a, MemOp::LEUW)
    }

    translate_fp::fp_trans16!();
}

impl DecodeXlrbr for S<'_, '_> {
    fn trans_crc32_b(&mut self, a: &mut arg_r2) -> bool {
        self.crc(a, &helpers::CRC32, 1)
    }

    fn trans_crc32_h(&mut self, a: &mut arg_r2) -> bool {
        self.crc(a, &helpers::CRC32, 2)
    }

    fn trans_crc32_w(&mut self, a: &mut arg_r2) -> bool {
        self.crc(a, &helpers::CRC32, 4)
    }

    fn trans_crc32_d(&mut self, a: &mut arg_r2) -> bool {
        self.crc(a, &helpers::CRC32, 8)
    }

    fn trans_crc32c_b(&mut self, a: &mut arg_r2) -> bool {
        self.crc(a, &helpers::CRC32C, 1)
    }

    fn trans_crc32c_h(&mut self, a: &mut arg_r2) -> bool {
        self.crc(a, &helpers::CRC32C, 2)
    }

    fn trans_crc32c_w(&mut self, a: &mut arg_r2) -> bool {
        self.crc(a, &helpers::CRC32C, 4)
    }

    fn trans_crc32c_d(&mut self, a: &mut arg_r2) -> bool {
        self.crc(a, &helpers::CRC32C, 8)
    }
}

impl DisasContext {
    /// Whether the semihosting sequence surrounds the `ebreak` at `pc`, as QEMU's
    /// `trans_ebreak()` checks: semihosting is on for this privilege level and the
    /// sequence is within one page.
    fn semihost_sequence(&self, cpu: &mut Cpu<'_>, pc: u64) -> bool {
        let enabled = match self.semihosting {
            Some(userspace) => self.priv_lvl != PRV_U || userspace,
            None => false,
        };
        let pre = pc.wrapping_sub(4);
        let post = pc.wrapping_add(4);
        if !enabled || pre & !0xfff != post & !0xfff {
            return false;
        }
        let mut word = |addr: u64| {
            let mut b = [0u8; 4];
            let ops = cpu.ops();
            let idx = ops.mmu_index(cpu, true);
            cpu_ld_code(cpu, addr, &mut b, idx, Ra::None).ok().map(|()| u32::from_le_bytes(b))
        };
        word(pre) == Some(SEMIHOST_PRE) && word(post) == Some(SEMIHOST_POST)
    }
}

impl TranslatorOps for DisasContext {
    fn init_disas_context(&mut self, db: &mut DisasContextBase<'_>, _cpu: &mut Cpu<'_>) {
        let f = &mut db.tb.f;
        let env = f.env();
        let x = std::array::from_fn(|r| f.global_mem_new_i64(env, gpr_off(r) as i64, GPR_NAMES[r]));
        let fr =
            std::array::from_fn(|r| f.global_mem_new_i64(env, fpr_off(r) as i64, FPR_NAMES[r]));
        self.g = Some(G {
            x,
            f: fr,
            pc: f.global_mem_new_i64(env, PC as i64, "pc"),
            load_res: f.global_mem_new_i64(env, LOAD_RES as i64, "load_res"),
            load_val: f.global_mem_new_i64(env, LOAD_VAL as i64, "load_val"),
        });
        self.goto_tb_used = 0;
        self.side_exits.clear();

        let flags = db.tb.flags;
        self.mem_idx = flags & TB_MEM_IDX_MASK;
        self.priv_lvl = u64::from((flags >> TB_PRIV_SHIFT) & 3);
        self.virt_enabled = flags & TB_VIRT != 0;
        self.pm = PointerMask::from_tb_flags(flags);
        self.mstatus_fs = u64::from((flags >> TB_FS_SHIFT) & 3);
        self.mstatus_vs = u64::from((flags >> TB_VS_SHIFT) & 3);
        self.vill = flags & TB_VILL != 0;
        self.sew = ((flags >> TB_SEW_SHIFT) & 7) as i32;
        // sextract32(lmul, 0, 3).
        self.lmul = ((((flags >> TB_LMUL_SHIFT) & 7) as i32) << 29) >> 29;
        self.vta = flags & TB_VTA != 0 && self.cfg.rvv_ta_all_1s;
        self.vma = flags & TB_VMA != 0 && self.cfg.rvv_ma_all_1s;
        self.cfg_vta_all_1s = self.cfg.rvv_ta_all_1s;
        self.vstart_eq_zero = flags & TB_VSTART_EQ_ZERO != 0;
        self.frm = -1;
        self.frm_valid = false;
    }

    fn insn_start(&mut self, db: &mut DisasContextBase<'_>, _cpu: &mut Cpu<'_>) {
        let pc = db.pc_next;
        db.tb.f.gen_insn_start(&[pc, 0, 0]);
    }

    fn translate_insn(
        &mut self,
        db: &mut DisasContextBase<'_>,
        cpu: &mut Cpu<'_>,
    ) -> Result<(), CpuLoopExit> {
        let pc = db.pc_next;
        self.pc_curr = pc;
        let opcode16 = db.translator_lduw(cpu, pc, Endian::Little)?;

        // decode_opc().
        self.cur_insn_len = insn_len(opcode16);
        if self.cur_insn_len == 2 {
            self.opcode = u64::from(opcode16);
            self.semihost_seq = false;
            // Zca is the C extension without the floating point loads and stores.
            let c = self.cfg.has(RVC) || self.cfg.ext_zca;
            let mut s = S { d: self, b: db };
            if !(c && decode16(&mut s, opcode16)) {
                s.gen_exception_illegal();
            }
        } else {
            let hi = db.translator_lduw(cpu, pc + 2, Endian::Little)?;
            let opcode32 = u32::from(opcode16) | (u32::from(hi) << 16);
            self.opcode = u64::from(opcode32);
            self.semihost_seq = opcode32 == EBREAK && self.semihost_sequence(cpu, pc);
            let xlrbr = self.xlrbr;
            let mut s = S { d: self, b: db };
            if !decode32(&mut s, opcode32) && !(xlrbr && xlrbr::decode(&mut s, opcode32)) {
                s.gen_exception_illegal();
            }
        }
        db.pc_next = pc.wrapping_add(self.cur_insn_len);

        // Only the first insn within a TB is allowed to cross a page boundary.
        if db.is_jmp == DisasJumpType::Next {
            if !db.is_same_page(db.pc_next) {
                db.is_jmp = DisasJumpType::TooMany;
            } else {
                let page_ofs = db.pc_next & 0xfff;
                if page_ofs > 4096 - 4 {
                    let next = db.translator_lduw(cpu, db.pc_next, Endian::Little)?;
                    let len = insn_len(next);
                    if !db.is_same_page(db.pc_next + len - 1) {
                        db.is_jmp = DisasJumpType::TooMany;
                    }
                }
            }
        }
        Ok(())
    }

    fn tb_stop(&mut self, db: &mut DisasContextBase<'_>, _cpu: &mut Cpu<'_>) {
        let mut s = S { d: self, b: db };
        match s.b.is_jmp {
            DisasJumpType::Next | DisasJumpType::TooMany => {
                let dest = s.b.pc_next;
                let slot = s.goto_tb_slot(1);
                s.gen_goto_dest(slot, dest);
            }
            DisasJumpType::NoReturn | DisasJumpType::Target(_) => {}
        }
        s.gen_side_exits();
    }
}

#[cfg(test)]
mod tests {
    use super::insn_len;

    #[test]
    fn lengths() {
        assert_eq!(insn_len(0x0001), 2);
        assert_eq!(insn_len(0x0013), 4);
        assert_eq!(insn_len(0x4502), 2);
    }
}
