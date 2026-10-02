// SPDX-License-Identifier: GPL-2.0-or-later

//! The A64 translator: QEMU's `target/arm/tcg/translate-a64.c`.
//!
//! The decoder comes from `a64.decode` (see `build.rs`); [`S`] implements its `trans_` hooks
//! for the base integer instructions here and for scalar FP and AdvSIMD in `translate_simd.rs`,
//! and every pattern left out returns false, which raises an Undefined Instruction exception
//! just as QEMU's `unallocated_encoding()` does.
//!
//! QEMU keeps the general registers, the PC, the flags and the exclusive monitor in TCG
//! globals. This port has no globals: every read loads from `env` and every write stores to it
//! at once, which gives the same result because the generated code is the only writer while
//! a block runs, and helpers only see `env` after the store. The PC is written to `env`
//! before every helper that can raise an exception, so those helpers raise without unwinding.

use ruvm_jit::{Cpu, CpuLoopExit, DisasContextBase, DisasJumpType, TranslatorOps};
use ruvm_jit_core::ir::{TempI32, TempI64, TempPtr};
use ruvm_jit_core::tcg_op_ldst::AtomicOp;
use ruvm_jit_core::types::{Cond, mo};
use ruvm_jit_core::{Func, MemOp, Temp};
use ruvm_mem::Endian;

use super::helpers::{self, Def};
use super::sysreg::{self, Kind};
use super::{
    TB_ALIGN_MEM, TB_E2H, TB_EL_MASK, TB_FPEXC_EL_SHIFT, TB_MMUIDX_SHIFT, TB_PSTATE_IL,
    TB_SVEEXC_EL_SHIFT, TB_TBID_SHIFT, TB_TBII_SHIFT, TB_UNPRIV, TB_VL_SHIFT, regime_has_2_ranges,
};
use crate::cpu::{
    ArmCpuModel, CF, EXCLUSIVE_ADDR, EXCLUSIVE_HIGH, EXCLUSIVE_VAL, EXCP_BKPT, EXCP_HVC, EXCP_SMC,
    EXCP_SWI, EXCP_UDEF, MMU_IDX_E10_0, MMU_IDX_E10_1, MMU_IDX_E10_1_PAN, MMU_IDX_E20_0,
    MMU_IDX_E20_2, MMU_IDX_E20_2_PAN, NF, PC, PSTATE, PSTATE_PAN, PSTATE_SP, PSTATE_UAO, VF, ZF,
    xreg_off,
};
use crate::syndrome::{
    syn_aa64_bkpt, syn_aa64_hvc, syn_aa64_smc, syn_aa64_svc, syn_aa64_sysregtrap, syn_illegalstate,
    syn_uncategorized,
};

#[allow(missing_docs, unreachable_pub, clippy::pedantic, clippy::nursery)]
mod decode {
    include!(concat!(env!("OUT_DIR"), "/a64_decode.rs"));

    /// Decode and translate one instruction.
    pub(super) fn disas(ctx: &mut impl DisasA64, insn: u32) -> bool {
        disas_a64(ctx, insn)
    }
}

#[path = "translate_simd.rs"]
mod simd;

#[path = "translate_sve.rs"]
mod sve;

use simd::{Chk, Feat, Mov, RA, RF, RM, RN, RP, RZ, cop, fop, nop};

use decode::{
    DisasA64, arg_addsub_ext, arg_addsub_shift, arg_atomic, arg_bitfield, arg_cbz, arg_disas_a6426,
    arg_disas_a6432, arg_disas_a6434, arg_disas_a6436, arg_disas_a6439, arg_disas_a6445,
    arg_disas_a6461, arg_disas_a6462, arg_extract, arg_i, arg_ldlit, arg_ldst, arg_ldst_imm,
    arg_ldstpair, arg_logic_shift, arg_movw, arg_r, arg_ri, arg_rr, arg_rr_sf, arg_rri_log,
    arg_rri_sf, arg_rrr, arg_rrr_e, arg_rrr_sf, arg_rrrr, arg_stlr, arg_stxr, arg_tbz,
};

/// `DISAS_EXIT`: exit to the main loop without touching the PC.
const DISAS_EXIT: DisasJumpType = DisasJumpType::Target(0);
/// `DISAS_JUMP`: the PC is in `env`; look up the next block.
const DISAS_JUMP: DisasJumpType = DisasJumpType::Target(1);
/// `DISAS_UPDATE_EXIT`: write the PC of the next instruction and exit to the main loop.
const DISAS_UPDATE_EXIT: DisasJumpType = DisasJumpType::Target(2);
/// `DISAS_WFI`: write the PC of the next instruction and call the WFI helper.
const DISAS_WFI: DisasJumpType = DisasJumpType::Target(3);

/// The A64 shift types.
const SHIFT_LSL: i32 = 0;
const SHIFT_LSR: i32 = 1;
const SHIFT_ASR: i32 = 2;

/// The target part of the disassembly context, `DisasContext`: what the TB flags say.
pub(crate) struct DisasContext {
    model: ArmCpuModel,
    pc_curr: u64,
    current_el: u32,
    pstate_il: bool,
    /// LDTR and STTR use the EL0 MMU index.
    unpriv: bool,
    align_mem: bool,
    tbii: u32,
    tbid: u32,
    mmu_idx: u32,
    /// HCR_EL2.E2H, for the VHE register redirections.
    e2h: bool,
    /// The EL that FP and AdvSIMD instructions trap to, or 0 if they do not trap.
    fp_excp_el: u32,
    /// The EL that SVE instructions trap to, or 0 if they do not trap.
    sve_excp_el: u32,
    /// The SVE vector length in bytes.
    vl: u32,
}

impl DisasContext {
    /// A context for the CPU `model`; the rest is filled in from the TB flags.
    pub(crate) fn new(model: &ArmCpuModel) -> DisasContext {
        DisasContext {
            model: model.clone(),
            pc_curr: 0,
            current_el: 0,
            pstate_il: false,
            unpriv: false,
            align_mem: false,
            tbii: 0,
            tbid: 0,
            mmu_idx: 0,
            e2h: false,
            fp_excp_el: 0,
            sve_excp_el: 0,
            vl: 16,
        }
    }
}

/// One instruction being translated: the target context and the generic one together, so the
/// `trans_` hooks can reach both.
struct S<'a, 'b> {
    d: &'a mut DisasContext,
    b: &'a mut DisasContextBase<'b>,
}

/// How [`S::arith`] combines its operands.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Arith {
    Add,
    Sub,
    AddCc,
    SubCc,
}

/// The logical operations of [`S::logic_reg`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum Logic {
    And,
    Or,
    Xor,
}

impl S<'_, '_> {
    fn f(&mut self) -> &mut Func {
        &mut self.b.tb.f
    }

    fn env(&mut self) -> TempPtr {
        self.f().env()
    }

    fn new64(&mut self) -> TempI64 {
        self.f().temp_new_i64()
    }

    fn new32(&mut self) -> TempI32 {
        self.f().temp_new_i32()
    }

    fn c64(&mut self, v: i64) -> TempI64 {
        self.f().constant_i64(v)
    }

    fn c32(&mut self, v: i32) -> TempI32 {
        self.f().constant_i32(v)
    }

    fn feat(&self) -> crate::cpu::ArmFeatures {
        self.d.model.features
    }

    fn get_mem_index(&self) -> u32 {
        self.d.mmu_idx
    }

    /// `get_a64_user_mem_index()`.
    fn user_mem_index(&self, unpriv: bool) -> u32 {
        if !(unpriv && self.d.unpriv) {
            return self.d.mmu_idx;
        }
        match self.d.mmu_idx as usize {
            MMU_IDX_E10_1 | MMU_IDX_E10_1_PAN => MMU_IDX_E10_0 as u32,
            MMU_IDX_E20_2 | MMU_IDX_E20_2_PAN => MMU_IDX_E20_0 as u32,
            idx => idx as u32,
        }
    }

    // Registers and flags.

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

    /// `cpu_reg()`: register `r`, with 31 reading as zero. The result is a fresh temp.
    fn reg(&mut self, r: i32) -> TempI64 {
        if r == 31 {
            let t = self.new64();
            self.f().gen_movi_i64(t, 0);
            t
        } else {
            self.ld_env64(xreg_off(r as usize))
        }
    }

    /// `cpu_reg_sp()`: register `r`, with 31 being SP.
    fn reg_sp(&mut self, r: i32) -> TempI64 {
        self.ld_env64(xreg_off(r as usize))
    }

    /// `read_cpu_reg()`.
    fn read_cpu_reg(&mut self, r: i32, sf: bool) -> TempI64 {
        let t = self.reg(r);
        if !sf {
            self.f().gen_ext32u_i64(t, t);
        }
        t
    }

    /// `read_cpu_reg_sp()`.
    fn read_cpu_reg_sp(&mut self, r: i32, sf: bool) -> TempI64 {
        let t = self.reg_sp(r);
        if !sf {
            self.f().gen_ext32u_i64(t, t);
        }
        t
    }

    /// Write register `r`; writes to 31 (XZR) are discarded.
    fn set_reg(&mut self, r: i32, t: TempI64) {
        if r != 31 {
            self.st_env64(t, xreg_off(r as usize));
        }
    }

    /// Write register `r`, with 31 being SP.
    fn set_reg_sp(&mut self, r: i32, t: TempI64) {
        self.st_env64(t, xreg_off(r as usize));
    }

    /// Write `t` to register `r`, zero extending from 32 bits unless `sf`.
    fn set_reg_sf(&mut self, r: i32, t: TempI64, sf: bool) {
        if !sf {
            self.f().gen_ext32u_i64(t, t);
        }
        self.set_reg(r, t);
    }

    /// `gen_a64_update_pc()`: write `pc_curr + diff` to the PC.
    fn update_pc(&mut self, diff: i64) {
        let pc = self.d.pc_curr.wrapping_add(diff as u64);
        let t = self.c64(pc as i64);
        self.st_env64(t, PC);
    }

    /// `gen_top_byte_ignore()`.
    fn top_byte_ignore(&mut self, dst: TempI64, src: TempI64, tbi: u32) {
        if tbi == 0 {
            // Load unmodified address.
            self.f().gen_mov_i64(dst, src);
        } else {
            if !regime_has_2_ranges(self.d.mmu_idx as usize) {
                // Force tag byte to all zero.
                self.f().gen_extract_i64(dst, src, 0, 56);
                return;
            }
            let f = self.f();
            // Sign-extend from bit 55.
            f.gen_sextract_i64(dst, src, 0, 56);
            match tbi {
                1 => f.gen_and_i64(dst, dst, src),
                2 => f.gen_or_i64(dst, dst, src),
                // tbi == 3: tbi enabled for both ranges, sign extension is all we need.
                _ => {}
            }
        }
    }

    /// `gen_a64_set_pc()`: jump to the address in `src`, with TBI applied.
    fn set_pc(&mut self, src: TempI64) {
        let t = self.new64();
        let tbii = self.d.tbii;
        self.top_byte_ignore(t, src, tbii);
        self.st_env64(t, PC);
    }

    /// `clean_data_tbi()`.
    fn clean_data_tbi(&mut self, addr: TempI64) -> TempI64 {
        let t = self.new64();
        let tbid = self.d.tbid;
        self.top_byte_ignore(t, addr, tbid);
        t
    }

    /// Declare and call a helper.
    fn call(&mut self, d: &Def, ret: Option<Temp>, args: &[Temp]) {
        let f = self.f();
        let h = f.helper(d.info());
        f.gen_call(h, ret, args);
    }

    /// `gen_exception_insn()`: raise `excp` at `pc_curr + diff` to the default EL.
    fn gen_exception_insn(&mut self, diff: i64, excp: i32, syndrome: u32) {
        // default_exception_el(): EL1 unless already higher.
        self.gen_exception_insn_el(diff, excp, syndrome, self.d.current_el.max(1));
    }

    /// `gen_exception_insn_el()`: raise `excp` at `pc_curr + diff` to `target_el`.
    fn gen_exception_insn_el(&mut self, diff: i64, excp: i32, syndrome: u32, target_el: u32) {
        self.update_pc(diff);
        let env = self.env();
        let e = self.c32(excp);
        let syn = self.c32(syndrome as i32);
        let el = self.c32(target_el as i32);
        self.call(
            &helpers::EXCEPTION_WITH_SYNDROME_EL,
            None,
            &[env.into(), e.into(), syn.into(), el.into()],
        );
        self.b.is_jmp = DisasJumpType::NoReturn;
    }

    /// `unallocated_encoding()`.
    fn unallocated_encoding(&mut self) {
        self.gen_exception_insn(0, EXCP_UDEF, syn_uncategorized());
    }

    /// `gen_goto_tb()`: go to `pc_curr + diff`, chaining when possible.
    fn gen_goto_tb(&mut self, n: u64, diff: i64) {
        let dest = self.d.pc_curr.wrapping_add(diff as u64);
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

    // Flags.

    /// `gen_set_NZ64()`.
    fn set_nz64(&mut self, result: TempI64) {
        let nf = self.new32();
        let zf = self.new32();
        let f = self.f();
        f.gen_extr_i64_i32(zf, nf, result);
        f.gen_or_i32(zf, zf, nf);
        self.st_env32(nf, NF);
        self.st_env32(zf, ZF);
    }

    /// `gen_logic_CC()`.
    fn logic_cc(&mut self, sf: bool, result: TempI64) {
        if sf {
            self.set_nz64(result);
        } else {
            let nf = self.new32();
            self.f().gen_extrl_i64_i32(nf, result);
            self.st_env32(nf, NF);
            self.st_env32(nf, ZF);
        }
        let zero = self.c32(0);
        self.st_env32(zero, CF);
        self.st_env32(zero, VF);
    }

    /// `gen_add64_CC()` and `gen_add32_CC()`.
    fn add_cc(&mut self, sf: bool, dest: TempI64, t0: TempI64, t1: TempI64) {
        if sf {
            let result = self.new64();
            let flag = self.new64();
            let tmp = self.new64();
            let cf = self.new32();
            let vf = self.new32();
            let f = self.f();
            f.gen_movi_i64(tmp, 0);
            f.gen_add2_i64(result, flag, t0, tmp, t1, tmp);
            f.gen_extrl_i64_i32(cf, flag);
            self.st_env32(cf, CF);
            self.set_nz64(result);
            let f = self.f();
            f.gen_xor_i64(flag, result, t0);
            f.gen_xor_i64(tmp, t0, t1);
            f.gen_andc_i64(flag, flag, tmp);
            f.gen_extrh_i64_i32(vf, flag);
            f.gen_mov_i64(dest, result);
            self.st_env32(vf, VF);
        } else {
            let t0_32 = self.new32();
            let t1_32 = self.new32();
            let tmp = self.new32();
            let nf = self.new32();
            let cf = self.new32();
            let vf = self.new32();
            let f = self.f();
            f.gen_movi_i32(tmp, 0);
            f.gen_extrl_i64_i32(t0_32, t0);
            f.gen_extrl_i64_i32(t1_32, t1);
            f.gen_add2_i32(nf, cf, t0_32, tmp, t1_32, tmp);
            f.gen_xor_i32(vf, nf, t0_32);
            f.gen_xor_i32(tmp, t0_32, t1_32);
            f.gen_andc_i32(vf, vf, tmp);
            f.gen_extu_i32_i64(dest, nf);
            self.st_env32(nf, NF);
            self.st_env32(nf, ZF);
            self.st_env32(cf, CF);
            self.st_env32(vf, VF);
        }
    }

    /// `gen_sub64_CC()` and `gen_sub32_CC()`.
    fn sub_cc(&mut self, sf: bool, dest: TempI64, t0: TempI64, t1: TempI64) {
        if sf {
            // 64 bit arithmetic
            let result = self.new64();
            let flag = self.new64();
            let tmp = self.new64();
            let cf = self.new32();
            let vf = self.new32();
            self.f().gen_sub_i64(result, t0, t1);
            self.set_nz64(result);
            let f = self.f();
            f.gen_setcond_i64(Cond::Geu, flag, t0, t1);
            f.gen_extrl_i64_i32(cf, flag);
            f.gen_xor_i64(flag, result, t0);
            f.gen_xor_i64(tmp, t0, t1);
            f.gen_and_i64(flag, flag, tmp);
            f.gen_extrh_i64_i32(vf, flag);
            f.gen_mov_i64(dest, result);
            self.st_env32(cf, CF);
            self.st_env32(vf, VF);
        } else {
            // 32 bit arithmetic
            let t0_32 = self.new32();
            let t1_32 = self.new32();
            let tmp = self.new32();
            let nf = self.new32();
            let cf = self.new32();
            let vf = self.new32();
            let f = self.f();
            f.gen_extrl_i64_i32(t0_32, t0);
            f.gen_extrl_i64_i32(t1_32, t1);
            f.gen_sub_i32(nf, t0_32, t1_32);
            f.gen_setcond_i32(Cond::Geu, cf, t0_32, t1_32);
            f.gen_xor_i32(vf, nf, t0_32);
            f.gen_xor_i32(tmp, t0_32, t1_32);
            f.gen_and_i32(vf, vf, tmp);
            f.gen_extu_i32_i64(dest, nf);
            self.st_env32(nf, NF);
            self.st_env32(nf, ZF);
            self.st_env32(cf, CF);
            self.st_env32(vf, VF);
        }
    }

    /// `gen_adc()`: dest = T0 + T1 + CF; do not compute flags.
    fn adc(&mut self, sf: bool, dest: TempI64, t0: TempI64, t1: TempI64) {
        let cf = self.ld_env32(CF);
        let tmp = self.new64();
        let f = self.f();
        f.gen_add_i64(dest, t0, t1);
        f.gen_extu_i32_i64(tmp, cf);
        f.gen_add_i64(dest, dest, tmp);
        if !sf {
            f.gen_ext32u_i64(dest, dest);
        }
    }

    /// `gen_adc_CC()`: dest = T0 + T1 + CF; compute C, N, V and Z flags.
    fn adc_cc(&mut self, sf: bool, dest: TempI64, t0: TempI64, t1: TempI64) {
        let cf = self.ld_env32(CF);
        if sf {
            let result = self.new64();
            let cf_64 = self.new64();
            let vf_64 = self.new64();
            let tmp = self.new64();
            let vf = self.new32();
            let f = self.f();
            f.gen_extu_i32_i64(cf_64, cf);
            f.gen_addcio_i64(result, cf_64, t0, t1, cf_64);
            f.gen_extrl_i64_i32(cf, cf_64);
            self.st_env32(cf, CF);
            self.set_nz64(result);
            let f = self.f();
            f.gen_xor_i64(vf_64, result, t0);
            f.gen_xor_i64(tmp, t0, t1);
            f.gen_andc_i64(vf_64, vf_64, tmp);
            f.gen_extrh_i64_i32(vf, vf_64);
            f.gen_mov_i64(dest, result);
            self.st_env32(vf, VF);
        } else {
            let t0_32 = self.new32();
            let t1_32 = self.new32();
            let tmp = self.new32();
            let nf = self.new32();
            let vf = self.new32();
            let f = self.f();
            f.gen_extrl_i64_i32(t0_32, t0);
            f.gen_extrl_i64_i32(t1_32, t1);
            f.gen_addcio_i32(nf, cf, t0_32, t1_32, cf);
            f.gen_xor_i32(vf, nf, t0_32);
            f.gen_xor_i32(tmp, t0_32, t1_32);
            f.gen_andc_i32(vf, vf, tmp);
            f.gen_extu_i32_i64(dest, nf);
            self.st_env32(nf, NF);
            self.st_env32(nf, ZF);
            self.st_env32(cf, CF);
            self.st_env32(vf, VF);
        }
    }

    /// `dest = t0 op t1`, setting the flags for the CC forms; the result is zero extended
    /// from 32 bits unless `sf`.
    fn arith(&mut self, sf: bool, op: Arith, dest: TempI64, t0: TempI64, t1: TempI64) {
        match op {
            Arith::Add => self.f().gen_add_i64(dest, t0, t1),
            Arith::Sub => self.f().gen_sub_i64(dest, t0, t1),
            Arith::AddCc => self.add_cc(sf, dest, t0, t1),
            Arith::SubCc => self.sub_cc(sf, dest, t0, t1),
        }
        if !sf {
            self.f().gen_ext32u_i64(dest, dest);
        }
    }

    /// `arm_test_cc()`: the comparison against zero of the returned value that is true when
    /// condition `cc` holds.
    fn test_cc(&mut self, cc: i32) -> (Cond, TempI32) {
        let (cond, value) = match cc >> 1 {
            // eq: Z; ne: !Z
            0 => (Cond::Eq, self.ld_env32(ZF)),
            // cs: C; cc: !C
            1 => (Cond::Ne, self.ld_env32(CF)),
            // mi: N; pl: !N
            2 => (Cond::Lt, self.ld_env32(NF)),
            // vs: V; vc: !V
            3 => (Cond::Lt, self.ld_env32(VF)),
            4 => {
                // hi: C && !Z; ls: !C || Z
                let cf = self.ld_env32(CF);
                let zf = self.ld_env32(ZF);
                let v = self.new32();
                let f = self.f();
                f.gen_neg_i32(v, cf);
                f.gen_and_i32(v, v, zf);
                (Cond::Ne, v)
            }
            5 => {
                // ge: N == V -> N ^ V == 0; lt: N != V -> N ^ V != 0
                let vf = self.ld_env32(VF);
                let nf = self.ld_env32(NF);
                let v = self.new32();
                self.f().gen_xor_i32(v, vf, nf);
                (Cond::Ge, v)
            }
            6 => {
                // gt: !Z && N == V; le: Z || N != V
                let vf = self.ld_env32(VF);
                let nf = self.ld_env32(NF);
                let zf = self.ld_env32(ZF);
                let v = self.new32();
                let f = self.f();
                f.gen_xor_i32(v, vf, nf);
                f.gen_sari_i32(v, v, 31);
                f.gen_andc_i32(v, zf, v);
                (Cond::Ne, v)
            }
            // Use the ALWAYS condition, which will fold early. It doesn't matter what we use
            // for the value.
            _ => (Cond::Always, self.ld_env32(ZF)),
        };
        if cc & 1 != 0 && cond != Cond::Always { (cond.invert(), value) } else { (cond, value) }
    }

    /// `a64_test_cc()`: [`S::test_cc`] with the value sign extended to 64 bits.
    fn test_cc64(&mut self, cc: i32) -> (Cond, TempI64) {
        let (cond, v32) = self.test_cc(cc);
        let v = self.new64();
        self.f().gen_ext_i32_i64(v, v32);
        (cond, v)
    }

    // Shifts.

    /// `shift_reg()`.
    fn shift_reg(&mut self, dst: TempI64, src: TempI64, sf: bool, st: i32, amount: TempI64) {
        match st {
            SHIFT_LSL => self.f().gen_shl_i64(dst, src, amount),
            SHIFT_LSR => self.f().gen_shr_i64(dst, src, amount),
            SHIFT_ASR => {
                let f = self.f();
                if sf {
                    f.gen_sar_i64(dst, src, amount);
                } else {
                    f.gen_ext32s_i64(dst, src);
                    f.gen_sar_i64(dst, dst, amount);
                }
            }
            _ => {
                if sf {
                    self.f().gen_rotr_i64(dst, src, amount);
                } else {
                    let t0 = self.new32();
                    let t1 = self.new32();
                    let f = self.f();
                    f.gen_extrl_i64_i32(t0, src);
                    f.gen_extrl_i64_i32(t1, amount);
                    f.gen_rotr_i32(t0, t0, t1);
                    f.gen_extu_i32_i64(dst, t0);
                }
            }
        }
        if !sf {
            // zero extend final result
            self.f().gen_ext32u_i64(dst, dst);
        }
    }

    /// `shift_reg_imm()`.
    fn shift_reg_imm(&mut self, dst: TempI64, src: TempI64, sf: bool, st: i32, shift: i32) {
        if shift == 0 {
            self.f().gen_mov_i64(dst, src);
        } else {
            let amount = self.c64(i64::from(shift));
            self.shift_reg(dst, src, sf, st, amount);
        }
    }

    /// `ext_and_shift_reg()`.
    fn ext_and_shift_reg(&mut self, out: TempI64, input: TempI64, option: i32, shift: i32) {
        let extsize = (option & 3) as u32;
        let is_signed = option & 4 != 0;
        let mop = MemOp(extsize | if is_signed { MemOp::SIGN.0 } else { 0 });
        let f = self.f();
        f.gen_ext_i64(out, input, mop);
        f.gen_shli_i64(out, out, i64::from(shift));
    }

    // Memory.

    /// `finalize_memop()`: alignment from SCTLR.A, single copy atomicity if aligned, little
    /// endian.
    fn finalize_memop(&self, op: MemOp) -> MemOp {
        self.finalize_memop_atom(op, MemOp::ATOM_IFALIGN)
    }

    /// `finalize_memop_atom()`. QEMU also adds `MO_ALIGN | MO_ALIGN_TLB_ONLY` when SCTLR.A is
    /// clear, for Device memory; this port leaves that out (see the `tcg` module doc).
    fn finalize_memop_atom(&self, op: MemOp, atom: MemOp) -> MemOp {
        let mut op = op;
        if op.0 & MemOp::AMASK.0 == 0 && self.d.align_mem {
            op = op | MemOp::ALIGN;
        }
        op | atom | MemOp::LE
    }

    /// `check_atomic_align()` without FEAT_LSE2.
    fn check_atomic_align(&self, op: MemOp) -> MemOp {
        match op.0 & MemOp::SIZE.0 {
            0 => op,
            4 => self.finalize_memop_atom(MemOp::MO_128 | MemOp::ALIGN, MemOp::ATOM_IFALIGN_PAIR),
            _ => self.finalize_memop(op | MemOp::ALIGN),
        }
    }

    /// `check_ordered_align()` without FEAT_LSE2.
    fn check_ordered_align(&self, op: MemOp) -> MemOp {
        self.check_atomic_align(op)
    }

    /// `do_gpr_ld()`: load into a fresh temp, zero extending a sign extended load to 32 bits
    /// when `extend`.
    fn gpr_ld(&mut self, addr: TempI64, memop: MemOp, extend: bool, memidx: u32) -> TempI64 {
        let t = self.new64();
        let f = self.f();
        f.gen_qemu_ld_i64(t, addr, memidx, memop);
        if extend && memop.0 & MemOp::SIGN.0 != 0 {
            f.gen_ext32u_i64(t, t);
        }
        t
    }

    /// `op_addr_ldst_imm_pre()`: the dirty and clean addresses of an immediate offset access.
    fn addr_imm_pre(&mut self, rn: i32, imm: i64, p: bool) -> (TempI64, TempI64) {
        let dirty = self.read_cpu_reg_sp(rn, true);
        if !p {
            self.f().gen_addi_i64(dirty, dirty, imm);
        }
        let clean = self.clean_data_tbi(dirty);
        (dirty, clean)
    }

    /// `op_addr_ldst_imm_post()` and `op_addr_ldstpair_post()`: the base register writeback.
    fn addr_imm_post(&mut self, rn: i32, dirty: TempI64, imm: i64, w: bool, p: bool) {
        if w {
            if p {
                self.f().gen_addi_i64(dirty, dirty, imm);
            }
            self.set_reg_sp(rn, dirty);
        }
    }

    /// `op_addr_ldst_pre()`: the clean address of a register offset access.
    fn addr_reg(&mut self, a: &arg_ldst) -> TempI64 {
        let dirty = self.read_cpu_reg_sp(a.rn, true);
        let rm = self.read_cpu_reg(a.rm, true);
        self.ext_and_shift_reg(rm, rm, a.opt, if a.s != 0 { a.sz } else { 0 });
        self.f().gen_add_i64(dirty, dirty, rm);
        self.clean_data_tbi(dirty)
    }

    /// `gen_load_exclusive()`.
    fn load_exclusive(&mut self, rt: i32, rt2: i32, rn: i32, size: i32, is_pair: bool) {
        let idx = self.get_mem_index();
        let memop = self.check_atomic_align(MemOp((size + i32::from(is_pair)) as u32));
        let dirty = self.reg_sp(rn);
        let clean = self.clean_data_tbi(dirty);
        let val = self.new64();
        if is_pair {
            if size == 2 {
                let t1 = self.new64();
                let t2 = self.new64();
                let f = self.f();
                f.gen_qemu_ld_i64(val, clean, idx, memop);
                f.gen_extract_i64(t1, val, 0, 32);
                f.gen_extract_i64(t2, val, 32, 32);
                self.set_reg(rt, t1);
                self.set_reg(rt2, t2);
            } else {
                let t16 = self.f().temp_new_i128();
                let high = self.new64();
                let f = self.f();
                f.gen_qemu_ld_i128(t16, clean, idx, memop);
                f.gen_extr_i128_i64(val, high, t16);
                self.st_env64(high, EXCLUSIVE_HIGH);
                self.set_reg(rt, val);
                self.set_reg(rt2, high);
            }
        } else {
            self.f().gen_qemu_ld_i64(val, clean, idx, memop);
            self.set_reg(rt, val);
        }
        self.st_env64(val, EXCLUSIVE_VAL);
        self.st_env64(clean, EXCLUSIVE_ADDR);
    }

    /// `gen_store_exclusive()`.
    fn store_exclusive(&mut self, rd: i32, rt: i32, rt2: i32, rn: i32, size: i32, is_pair: bool) {
        let fail_label = self.f().new_label();
        let done_label = self.f().new_label();
        let dirty = self.reg_sp(rn);
        let clean = self.clean_data_tbi(dirty);
        let excl_addr = self.ld_env64(EXCLUSIVE_ADDR);
        self.f().gen_brcond_i64(Cond::Ne, clean, excl_addr, fail_label);
        // Without FEAT_LSE2 every size needs alignment.
        let memop = self.finalize_memop(MemOp((size + i32::from(is_pair)) as u32) | MemOp::ALIGN);
        let idx = self.get_mem_index();
        let excl_val = self.ld_env64(EXCLUSIVE_VAL);
        let tmp = self.new64();
        if is_pair {
            let t1 = self.reg(rt);
            let t2 = self.reg(rt2);
            if size == 2 {
                let f = self.f();
                f.gen_concat32_i64(tmp, t1, t2);
                f.gen_atomic_cmpxchg_i64(tmp, excl_addr, excl_val, tmp, idx, memop);
                f.gen_setcond_i64(Cond::Ne, tmp, tmp, excl_val);
            } else {
                let excl_high = self.ld_env64(EXCLUSIVE_HIGH);
                let a = self.new64();
                let b = self.new64();
                let f = self.f();
                let t16 = f.temp_new_i128();
                let c16 = f.temp_new_i128();
                f.gen_concat_i64_i128(t16, t1, t2);
                f.gen_concat_i64_i128(c16, excl_val, excl_high);
                f.gen_atomic_cmpxchg_i128(t16, excl_addr, c16, t16, idx, memop);
                f.gen_extr_i128_i64(a, b, t16);
                f.gen_xor_i64(a, a, excl_val);
                f.gen_xor_i64(b, b, excl_high);
                f.gen_or_i64(tmp, a, b);
                f.gen_setcondi_i64(Cond::Ne, tmp, tmp, 0);
            }
        } else {
            let t = self.reg(rt);
            let f = self.f();
            f.gen_atomic_cmpxchg_i64(tmp, excl_addr, excl_val, t, idx, memop);
            f.gen_setcond_i64(Cond::Ne, tmp, tmp, excl_val);
        }
        self.set_reg(rd, tmp);
        self.f().gen_br(done_label);
        self.f().gen_set_label(fail_label);
        let one = self.c64(1);
        self.set_reg(rd, one);
        self.f().gen_set_label(done_label);
        let m1 = self.c64(-1);
        self.st_env64(m1, EXCLUSIVE_ADDR);
    }

    /// `do_atomic_ld()`.
    fn atomic_ld(&mut self, a: &arg_atomic, op: AtomicOp, sign: bool, invert: bool) -> bool {
        if !self.feat().lse {
            return false;
        }
        let mop = MemOp(a.sz as u32 | if sign { MemOp::SIGN.0 } else { 0 });
        let mop = self.check_atomic_align(mop);
        let dirty = self.reg_sp(a.rn);
        let clean = self.clean_data_tbi(dirty);
        let rs = self.read_cpu_reg(a.rs, true);
        let rt = self.new64();
        let idx = self.get_mem_index();
        let f = self.f();
        if invert {
            f.gen_not_i64(rs, rs);
        }
        // The atomic primitives are all full barriers. Therefore we can ignore the Acquire
        // and Release bits of this instruction.
        f.gen_atomic_op_i64(op, rt, clean, rs, idx, mop);
        if sign {
            match a.sz {
                0 => f.gen_ext8u_i64(rt, rt),
                1 => f.gen_ext16u_i64(rt, rt),
                2 => f.gen_ext32u_i64(rt, rt),
                _ => {}
            }
        }
        self.set_reg(a.rt, rt);
        true
    }

    /// `do_muladd()`.
    fn muladd(&mut self, a: &arg_rrrr, sf: bool, is_sub: bool, mop: Option<MemOp>) -> bool {
        let op1 = self.reg(a.rn);
        let op2 = self.reg(a.rm);
        if let Some(m) = mop {
            let f = self.f();
            f.gen_ext_i64(op1, op1, m);
            f.gen_ext_i64(op2, op2, m);
        }
        let rd = self.new64();
        if a.ra == 31 && !is_sub {
            // Special-case MADD with rA == XZR; it is the standard MUL alias.
            self.f().gen_mul_i64(rd, op1, op2);
        } else {
            let ra = self.reg(a.ra);
            let tmp = self.new64();
            let f = self.f();
            f.gen_mul_i64(tmp, op1, op2);
            if is_sub {
                f.gen_sub_i64(rd, ra, tmp);
            } else {
                f.gen_add_i64(rd, ra, tmp);
            }
        }
        self.set_reg_sf(a.rd, rd, sf);
        true
    }

    /// `gen_rri()`.
    fn rri(&mut self, a: &arg_rri_sf, rd_sp: bool, op: Arith) -> bool {
        let rn = self.reg_sp(a.rn);
        let imm = self.c64(i64::from(a.imm));
        let rd = self.new64();
        self.arith(a.sf != 0, op, rd, rn, imm);
        if rd_sp {
            self.set_reg_sp(a.rd, rd);
        } else {
            self.set_reg(a.rd, rd);
        }
        true
    }

    /// `gen_rri_log()`.
    fn rri_log(&mut self, a: &arg_rri_log, set_cc: bool, op: Logic) -> bool {
        let dbm = a.dbm as u32;
        // Some immediate field values are reserved.
        let Some(mut imm) = logic_imm_decode_wmask((dbm >> 12) & 1, dbm & 0x3f, (dbm >> 6) & 0x3f)
        else {
            return false;
        };
        let sf = a.sf != 0;
        if !sf {
            imm &= 0xffff_ffff;
        }
        let rn = self.reg(a.rn);
        let rd = self.new64();
        let f = self.f();
        match op {
            Logic::And => f.gen_andi_i64(rd, rn, imm as i64),
            Logic::Or => f.gen_ori_i64(rd, rn, imm as i64),
            Logic::Xor => f.gen_xori_i64(rd, rn, imm as i64),
        }
        if set_cc {
            self.logic_cc(sf, rd);
        }
        if !sf {
            self.f().gen_ext32u_i64(rd, rd);
        }
        if set_cc {
            self.set_reg(a.rd, rd);
        } else {
            self.set_reg_sp(a.rd, rd);
        }
        true
    }

    /// `do_logic_reg()`.
    fn logic_reg(&mut self, a: &arg_logic_shift, op: Logic, setflags: bool) -> bool {
        let sf = a.sf != 0;
        if !sf && a.sa & (1 << 5) != 0 {
            return false;
        }
        let rn = self.reg(a.rn);
        let rm = self.read_cpu_reg(a.rm, sf);
        if a.sa != 0 {
            self.shift_reg_imm(rm, rm, sf, a.st, a.sa);
        }
        let rd = self.new64();
        let inv = a.n != 0;
        let f = self.f();
        match (op, inv) {
            (Logic::And, false) => f.gen_and_i64(rd, rn, rm),
            (Logic::And, true) => f.gen_andc_i64(rd, rn, rm),
            (Logic::Or, false) => f.gen_or_i64(rd, rn, rm),
            (Logic::Or, true) => f.gen_orc_i64(rd, rn, rm),
            (Logic::Xor, false) => f.gen_xor_i64(rd, rn, rm),
            (Logic::Xor, true) => f.gen_eqv_i64(rd, rn, rm),
        }
        if !sf {
            f.gen_ext32u_i64(rd, rd);
        }
        if setflags {
            self.logic_cc(sf, rd);
        }
        self.set_reg(a.rd, rd);
        true
    }

    /// `do_addsub_ext()`.
    fn addsub_ext(&mut self, a: &arg_addsub_ext, sub_op: bool, setflags: bool) -> bool {
        if a.sa > 4 {
            return false;
        }
        let sf = a.sf != 0;
        let rn = self.read_cpu_reg_sp(a.rn, sf);
        let rm = self.read_cpu_reg(a.rm, sf);
        self.ext_and_shift_reg(rm, rm, a.st, a.sa);
        let result = self.new64();
        let op = match (sub_op, setflags) {
            (false, false) => Arith::Add,
            (true, false) => Arith::Sub,
            (false, true) => Arith::AddCc,
            (true, true) => Arith::SubCc,
        };
        self.arith(sf, op, result, rn, rm);
        // non-flag setting ops may use SP
        if setflags {
            self.set_reg(a.rd, result);
        } else {
            self.set_reg_sp(a.rd, result);
        }
        true
    }

    /// `do_addsub_reg()`.
    fn addsub_reg(&mut self, a: &arg_addsub_shift, sub_op: bool, setflags: bool) -> bool {
        let sf = a.sf != 0;
        if a.st == 3 || (!sf && a.sa & 32 != 0) {
            return false;
        }
        let rn = self.read_cpu_reg(a.rn, sf);
        let rm = self.read_cpu_reg(a.rm, sf);
        self.shift_reg_imm(rm, rm, sf, a.st, a.sa);
        let result = self.new64();
        let op = match (sub_op, setflags) {
            (false, false) => Arith::Add,
            (true, false) => Arith::Sub,
            (false, true) => Arith::AddCc,
            (true, true) => Arith::SubCc,
        };
        self.arith(sf, op, result, rn, rm);
        self.set_reg(a.rd, result);
        true
    }

    /// `do_adc_sbc()`.
    fn adc_sbc(&mut self, a: &arg_rrr_sf, is_sub: bool, setflags: bool) -> bool {
        let sf = a.sf != 0;
        let rn = self.reg(a.rn);
        let y = self.reg(a.rm);
        if is_sub {
            self.f().gen_not_i64(y, y);
        }
        let rd = self.new64();
        if setflags {
            self.adc_cc(sf, rd, rn, y);
        } else {
            self.adc(sf, rd, rn, y);
        }
        self.set_reg(a.rd, rd);
        true
    }

    /// `handle_shift_reg()`.
    fn shift_by_reg(&mut self, a: &arg_rrr_sf, st: i32) -> bool {
        let sf = a.sf != 0;
        let shift = self.reg(a.rm);
        self.f().gen_andi_i64(shift, shift, if sf { 63 } else { 31 });
        let rn = self.read_cpu_reg(a.rn, sf);
        let rd = self.new64();
        self.shift_reg(rd, rn, sf, st, shift);
        self.set_reg(a.rd, rd);
        true
    }

    /// `do_div()`.
    fn div(&mut self, a: &arg_rrr_sf, is_signed: bool) -> bool {
        let sf = a.sf != 0;
        let (n, m) = if !sf && is_signed {
            let n = self.reg(a.rn);
            let m = self.reg(a.rm);
            let f = self.f();
            f.gen_ext32s_i64(n, n);
            f.gen_ext32s_i64(m, m);
            (n, m)
        } else {
            (self.read_cpu_reg(a.rn, sf), self.read_cpu_reg(a.rm, sf))
        };
        let rd = self.new64();
        let h = if is_signed { &helpers::SDIV64 } else { &helpers::UDIV64 };
        self.call(h, Some(rd.into()), &[n.into(), m.into()]);
        self.set_reg_sf(a.rd, rd, sf);
        true
    }

    /// `do_crc32()`.
    fn crc32(&mut self, a: &arg_rrr_e, crc32c: bool) -> bool {
        if !self.feat().crc32 {
            return false;
        }
        let val = self.reg(a.rm);
        if a.esz < 3 {
            self.f().gen_extract_i64(val, val, 0, 8 << a.esz);
        }
        let acc = self.reg(a.rn);
        let bytes = self.c32(1 << a.esz);
        let rd = self.new64();
        let h = if crc32c { &helpers::CRC32C_64 } else { &helpers::CRC32_64 };
        self.call(h, Some(rd.into()), &[acc.into(), val.into(), bytes.into()]);
        self.set_reg(a.rd, rd);
        true
    }

    /// The conditional branch of CBZ, TBZ and B.cond: go to `pc_curr + imm` when `cond`
    /// holds of `value` against zero, else to the next instruction.
    fn cond_branch64(&mut self, cond: Cond, value: TempI64, imm: i32) {
        let label = self.f().new_label();
        self.f().gen_brcondi_i64(cond, value, 0, label);
        self.gen_goto_tb(0, 4);
        self.f().gen_set_label(label);
        self.gen_goto_tb(1, i64::from(imm));
    }

    /// `handle_sys()`: MRS, MSR (register) and SYS.
    #[allow(clippy::too_many_arguments)]
    fn handle_sys(
        &mut self,
        isread: bool,
        op0: u32,
        op1: u32,
        op2: u32,
        crn: u32,
        crm: u32,
        rt: i32,
    ) {
        let feat = self.feat();
        let el = self.d.current_el;
        // The VHE redirections and the _EL12 and _EL02 aliases.
        let Some(key) =
            sysreg::resolve(sysreg::key(op0, op1, crn, crm, op2), el, self.d.e2h, &feat)
        else {
            self.unallocated_encoding();
            return;
        };
        let Some(ri) = sysreg::lookup(key, &feat) else {
            // Unknown register; this might be a guest error or a QEMU unimplemented feature.
            // Without FEAT_IDST this is an uncategorized UNDEF.
            self.unallocated_encoding();
            return;
        };
        // Check access permissions
        if !sysreg::access_ok(ri.access, el, isread) {
            self.unallocated_encoding();
            return;
        }
        if ri.trap.applies(el) {
            // Emit code to perform further access permissions checks at runtime; this may
            // result in an exception.
            self.update_pc(0);
            let syndrome = syn_aa64_sysregtrap(op0, op1, op2, crn, crm, rt as u32, isread);
            let env = self.env();
            let k = self.c32(key as i32);
            let syn = self.c32(syndrome as i32);
            let r = self.c32(i32::from(isread));
            self.call(
                &helpers::ACCESS_CHECK_CP_REG,
                None,
                &[env.into(), k.into(), syn.into(), r.into()],
            );
        }
        // FPCR and FPSR are ARM_CP_FPU registers.
        let is_fpu = key == sysreg::FPCR || key == sysreg::FPSR;
        let is_sve = matches!(key, sysreg::ZCR_EL1 | sysreg::ZCR_EL2 | sysreg::ZCR_EL3);
        let denied =
            if is_fpu { !self.fp_access_check() } else { is_sve && !self.sve_access_check() };
        if denied {
            return;
        }
        match ri.kind {
            Kind::Nop => return,
            Kind::CurrentEl => {
                // Reads as current EL value from pstate, which is guaranteed to be constant
                // by the tb flags.
                let t = self.c64(i64::from(el << 2));
                self.set_reg(rt, t);
                return;
            }
            Kind::DcZva => {
                // Writes clear the aligned block of memory which rt points into.
                let v = self.reg(rt);
                let addr = self.clean_data_tbi(v);
                let env = self.env();
                self.call(&helpers::DC_ZVA, None, &[env.into(), addr.into()]);
                return;
            }
            Kind::Model(get) => {
                if isread {
                    let t = self.c64(get(&self.d.model) as i64);
                    self.set_reg(rt, t);
                }
            }
            Kind::Zero => {
                if isread {
                    let t = self.c64(0);
                    self.set_reg(rt, t);
                }
            }
            Kind::Field { off, mask } => {
                if isread {
                    let t = self.ld_env64(off);
                    self.set_reg(rt, t);
                } else {
                    let t = self.reg(rt);
                    if mask != u64::MAX {
                        self.f().gen_andi_i64(t, t, mask as i64);
                    }
                    self.st_env64(t, off);
                }
            }
            Kind::Special => {
                // The accessors may look at the PC, and the AT instructions walk the page
                // tables, so synchronize the CPU state first.
                self.update_pc(0);
                let env = self.env();
                let k = self.c32(key as i32);
                if isread {
                    let t = self.new64();
                    self.call(&helpers::GET_SYSREG, Some(t.into()), &[env.into(), k.into()]);
                    self.set_reg(rt, t);
                } else {
                    let v = self.reg(rt);
                    self.call(&helpers::SET_SYSREG, None, &[env.into(), k.into(), v.into()]);
                }
            }
        }
        if !isread && key != sysreg::FPSR {
            // We default to ending the TB on a coprocessor register write; FPSR has
            // ARM_CP_SUPPRESS_TB_END.
            self.b.is_jmp = DISAS_UPDATE_EXIT;
        }
    }

    /// Set or clear `bit` of PSTATE, as `set_pstate_bits()` and `clear_pstate_bits()` do.
    fn pstate_bit(&mut self, bit: u32, set: bool) {
        let t = self.ld_env32(PSTATE);
        let f = self.f();
        if set {
            f.gen_ori_i32(t, t, bit as i32);
        } else {
            f.gen_andi_i32(t, t, !bit as i32);
        }
        self.st_env32(t, PSTATE);
    }

    /// The helper call of the MSR (immediate) forms that go through one.
    fn msr_i_helper(&mut self, d: &Def, imm: i32) {
        self.update_pc(0);
        let env = self.env();
        let i = self.c32(imm);
        self.call(d, None, &[env.into(), i.into()]);
    }
}

/// `logic_imm_decode_wmask()`: the bitmask immediate of the logical instructions, or `None`
/// for the reserved encodings.
fn logic_imm_decode_wmask(immn: u32, imms: u32, immr: u32) -> Option<u64> {
    // First determine the element size
    let x = (immn << 6) | (!imms & 0x3f);
    if x == 0 {
        return None;
    }
    let len = 31 - x.leading_zeros();
    if len < 1 {
        // This is the immn == 0, imms == 0x11111x case
        return None;
    }
    let e = 1u32 << len;
    let levels = e - 1;
    let s = imms & levels;
    let r = immr & levels;
    if s == levels {
        // <length of run - 1> mustn't be all-ones.
        return None;
    }
    // Create the value of one element: s+1 set bits rotated by r within the element (which
    // is e bits wide)...
    let emask = if e == 64 { u64::MAX } else { (1u64 << e) - 1 };
    let mut mask = if s + 1 == 64 { u64::MAX } else { (1u64 << (s + 1)) - 1 };
    if r != 0 {
        mask = ((mask >> r) | (mask << (e - r))) & emask;
    }
    // ...then replicate the element over the whole 64 bit value
    let mut size = e;
    while size < 64 {
        mask |= mask << size;
        size *= 2;
    }
    Some(mask)
}

impl DisasA64 for S<'_, '_> {
    simd::simd_trans!();

    fn shl_12(&mut self, x: i32) -> i32 {
        x << 12
    }

    fn times_4(&mut self, x: i32) -> i32 {
        x.wrapping_mul(4)
    }

    fn xor_2(&mut self, x: i32) -> i32 {
        x ^ 2
    }

    fn plus_2(&mut self, x: i32) -> i32 {
        x + 2
    }

    fn times_8(&mut self, x: i32) -> i32 {
        x.wrapping_mul(8)
    }

    fn plus_1(&mut self, x: i32) -> i32 {
        x + 1
    }

    fn rsub_32(&mut self, x: i32) -> i32 {
        32 - x
    }

    fn rsub_64(&mut self, x: i32) -> i32 {
        64 - x
    }

    fn rsub_16(&mut self, x: i32) -> i32 {
        16 - x
    }

    fn rsub_8(&mut self, x: i32) -> i32 {
        8 - x
    }

    fn scale_by_log2_tag_granule(&mut self, x: i32) -> i32 {
        x << 4
    }

    fn uimm_scaled(&mut self, x: i32) -> i32 {
        let imm = (x as u32) >> 3;
        let scale = x as u32 & 7;
        (imm << scale) as i32
    }

    // PC-rel. addressing

    fn trans_ADR(&mut self, a: &mut arg_ri) -> bool {
        let pc = self.d.pc_curr.wrapping_add(i64::from(a.imm) as u64);
        let t = self.c64(pc as i64);
        self.set_reg(a.rd, t);
        true
    }

    fn trans_ADRP(&mut self, a: &mut arg_ri) -> bool {
        let offset = (i64::from(a.imm) << 12) - (self.d.pc_curr & 0xfff) as i64;
        let pc = self.d.pc_curr.wrapping_add(offset as u64);
        let t = self.c64(pc as i64);
        self.set_reg(a.rd, t);
        true
    }

    // Add/subtract (immediate)

    fn trans_ADD_i(&mut self, a: &mut arg_rri_sf) -> bool {
        self.rri(a, true, Arith::Add)
    }

    fn trans_SUB_i(&mut self, a: &mut arg_rri_sf) -> bool {
        self.rri(a, true, Arith::Sub)
    }

    fn trans_ADDS_i(&mut self, a: &mut arg_rri_sf) -> bool {
        self.rri(a, false, Arith::AddCc)
    }

    fn trans_SUBS_i(&mut self, a: &mut arg_rri_sf) -> bool {
        self.rri(a, false, Arith::SubCc)
    }

    // Logical (immediate)

    fn trans_AND_i(&mut self, a: &mut arg_rri_log) -> bool {
        self.rri_log(a, false, Logic::And)
    }

    fn trans_ORR_i(&mut self, a: &mut arg_rri_log) -> bool {
        self.rri_log(a, false, Logic::Or)
    }

    fn trans_EOR_i(&mut self, a: &mut arg_rri_log) -> bool {
        self.rri_log(a, false, Logic::Xor)
    }

    fn trans_ANDS_i(&mut self, a: &mut arg_rri_log) -> bool {
        self.rri_log(a, true, Logic::And)
    }

    // Move wide (immediate)

    fn trans_MOVZ(&mut self, a: &mut arg_movw) -> bool {
        let pos = a.hw << 4;
        let t = self.c64(((a.imm as u64) << pos) as i64);
        self.set_reg(a.rd, t);
        true
    }

    fn trans_MOVN(&mut self, a: &mut arg_movw) -> bool {
        let pos = a.hw << 4;
        let mut imm = !((a.imm as u64) << pos);
        if a.sf == 0 {
            imm = u64::from(imm as u32);
        }
        let t = self.c64(imm as i64);
        self.set_reg(a.rd, t);
        true
    }

    fn trans_MOVK(&mut self, a: &mut arg_movw) -> bool {
        let pos = (a.hw << 4) as u32;
        let rd = self.reg(a.rd);
        let imm = self.c64(i64::from(a.imm));
        self.f().gen_deposit_i64(rd, rd, imm, pos, 16);
        self.set_reg_sf(a.rd, rd, a.sf != 0);
        true
    }

    // Bitfield

    fn trans_SBFM(&mut self, a: &mut arg_bitfield) -> bool {
        let rd = self.new64();
        let tmp = self.read_cpu_reg(a.rn, true);
        let bitsize: u32 = if a.sf != 0 { 64 } else { 32 };
        let ri = a.immr as u32;
        let si = a.imms as u32;
        let f = self.f();
        if si >= ri {
            // Wd<s-r:0> = Wn<s:r>
            let len = (si - ri) + 1;
            f.gen_sextract_i64(rd, tmp, ri, len);
            if a.sf == 0 {
                f.gen_ext32u_i64(rd, rd);
            }
        } else {
            // Wd<32+s-r,32-r> = Wn<s:0>
            let mut len = si + 1;
            let pos = (bitsize - ri) & (bitsize - 1);
            if len < ri {
                // Sign extend the destination field from len to fill the balance of the
                // word. Let the deposit below insert all of those sign bits.
                f.gen_sextract_i64(tmp, tmp, 0, len);
                len = ri;
            }
            // We start with zero, and we haven't modified any bits outside bitsize,
            // therefore no final zero-extension is unneeded for !sf.
            f.gen_deposit_z_i64(rd, tmp, pos, len);
        }
        self.set_reg(a.rd, rd);
        true
    }

    fn trans_UBFM(&mut self, a: &mut arg_bitfield) -> bool {
        let rd = self.new64();
        let tmp = self.read_cpu_reg(a.rn, true);
        let bitsize: u32 = if a.sf != 0 { 64 } else { 32 };
        let ri = a.immr as u32;
        let si = a.imms as u32;
        let f = self.f();
        if si >= ri {
            // Wd<s-r:0> = Wn<s:r>
            let len = (si - ri) + 1;
            f.gen_extract_i64(rd, tmp, ri, len);
        } else {
            // Wd<32+s-r,32-r> = Wn<s:0>
            let len = si + 1;
            let pos = (bitsize - ri) & (bitsize - 1);
            f.gen_deposit_z_i64(rd, tmp, pos, len);
        }
        self.set_reg(a.rd, rd);
        true
    }

    fn trans_BFM(&mut self, a: &mut arg_bitfield) -> bool {
        let rd = self.reg(a.rd);
        let tmp = self.read_cpu_reg(a.rn, true);
        let bitsize: u32 = if a.sf != 0 { 64 } else { 32 };
        let ri = a.immr as u32;
        let si = a.imms as u32;
        let f = self.f();
        let (pos, len) = if si >= ri {
            // Wd<s-r:0> = Wn<s:r>
            f.gen_shri_i64(tmp, tmp, i64::from(ri));
            (0, (si - ri) + 1)
        } else {
            // Wd<32+s-r,32-r> = Wn<s:0>
            ((bitsize - ri) & (bitsize - 1), si + 1)
        };
        f.gen_deposit_i64(rd, rd, tmp, pos, len);
        self.set_reg_sf(a.rd, rd, a.sf != 0);
        true
    }

    // Extract

    fn trans_EXTR(&mut self, a: &mut arg_extract) -> bool {
        let rd = self.new64();
        let sf = a.sf != 0;
        if a.imm == 0 {
            // shl_i32/shl_i64 is undefined for 32/64 bit shifts, so an extract from bit 0 is
            // a special case.
            let rm = self.reg(a.rm);
            self.set_reg_sf(a.rd, rm, sf);
            return true;
        }
        let rm = self.reg(a.rm);
        let rn = self.reg(a.rn);
        if sf {
            // Specialization to ROR happens in EXTRACT2.
            self.f().gen_extract2_i64(rd, rm, rn, a.imm as u32);
        } else {
            let t0 = self.new32();
            let t1 = self.new32();
            let f = self.f();
            f.gen_extrl_i64_i32(t0, rm);
            if a.rm == a.rn {
                f.gen_rotri_i32(t0, t0, a.imm);
            } else {
                f.gen_extrl_i64_i32(t1, rn);
                f.gen_extract2_i32(t0, t0, t1, a.imm as u32);
            }
            f.gen_extu_i32_i64(rd, t0);
        }
        self.set_reg(a.rd, rd);
        true
    }

    // Branches

    fn trans_B(&mut self, a: &mut arg_i) -> bool {
        self.gen_goto_tb(0, i64::from(a.imm));
        true
    }

    fn trans_BL(&mut self, a: &mut arg_i) -> bool {
        let lr = self.c64(self.d.pc_curr.wrapping_add(4) as i64);
        self.set_reg(30, lr);
        self.gen_goto_tb(0, i64::from(a.imm));
        true
    }

    fn trans_CBZ(&mut self, a: &mut arg_cbz) -> bool {
        let cmp = self.read_cpu_reg(a.rt, a.sf != 0);
        let cond = if a.nz != 0 { Cond::Ne } else { Cond::Eq };
        self.cond_branch64(cond, cmp, a.imm);
        true
    }

    fn trans_TBZ(&mut self, a: &mut arg_tbz) -> bool {
        let cmp = self.reg(a.rt);
        self.f().gen_andi_i64(cmp, cmp, (1u64 << a.bitpos) as i64);
        let cond = if a.nz != 0 { Cond::Ne } else { Cond::Eq };
        self.cond_branch64(cond, cmp, a.imm);
        true
    }

    fn trans_B_cond(&mut self, a: &mut arg_disas_a6426) -> bool {
        // BC.cond is only present with FEAT_HBC
        if a.c != 0 {
            return false;
        }
        if a.cond < 0x0e {
            // genuinely conditional branches
            let (cond, value) = self.test_cc(a.cond);
            let label = self.f().new_label();
            self.f().gen_brcondi_i32(cond, value, 0, label);
            self.gen_goto_tb(0, 4);
            self.f().gen_set_label(label);
            self.gen_goto_tb(1, i64::from(a.imm));
        } else {
            // 0xe and 0xf are both "always" conditions
            self.gen_goto_tb(0, i64::from(a.imm));
        }
        true
    }

    fn trans_BR(&mut self, a: &mut arg_r) -> bool {
        let dst = self.reg(a.rn);
        self.set_pc(dst);
        self.b.is_jmp = DISAS_JUMP;
        true
    }

    fn trans_BLR(&mut self, a: &mut arg_r) -> bool {
        // Read the target before writing the link register, which may be the same.
        let dst = self.reg(a.rn);
        let lr = self.c64(self.d.pc_curr.wrapping_add(4) as i64);
        self.set_reg(30, lr);
        self.set_pc(dst);
        self.b.is_jmp = DISAS_JUMP;
        true
    }

    fn trans_RET(&mut self, a: &mut arg_r) -> bool {
        let dst = self.reg(a.rn);
        self.set_pc(dst);
        self.b.is_jmp = DISAS_JUMP;
        true
    }

    fn trans_ERET(&mut self, _a: &mut arg_disas_a6432) -> bool {
        let el = self.d.current_el;
        if el == 0 {
            return false;
        }
        let elr = crate::cpu::env_off(std::mem::offset_of!(crate::cpu::CpuArmState, elr_el))
            + 8 * el as usize;
        let dst = self.ld_env64(elr);
        self.update_pc(0);
        let env = self.env();
        self.call(&helpers::EXCEPTION_RETURN, None, &[env.into(), dst.into()]);
        // Must exit loop to check un-masked IRQs
        self.b.is_jmp = DISAS_EXIT;
        true
    }

    // Hints

    fn trans_NOP(&mut self, _a: &mut arg_disas_a6432) -> bool {
        true
    }

    fn trans_YIELD(&mut self, _a: &mut arg_disas_a6432) -> bool {
        // A no-op here; see the `tcg` module doc.
        true
    }

    fn trans_WFE(&mut self, _a: &mut arg_disas_a6432) -> bool {
        // A no-op here; see the `tcg` module doc.
        true
    }

    fn trans_SEV(&mut self, _a: &mut arg_disas_a6432) -> bool {
        true
    }

    fn trans_SEVL(&mut self, _a: &mut arg_disas_a6432) -> bool {
        true
    }

    fn trans_ESB(&mut self, _a: &mut arg_disas_a6432) -> bool {
        // Without RAS, we must implement this as NOP.
        true
    }

    fn trans_CHKFEAT(&mut self, _a: &mut arg_disas_a6432) -> bool {
        // No feature that CHKFEAT reports is enabled, so X16 is unchanged.
        true
    }

    fn trans_GCSB(&mut self, _a: &mut arg_disas_a6432) -> bool {
        // Without FEAT_GCS this is a NOP.
        true
    }

    fn trans_XPACLRI(&mut self, _a: &mut arg_disas_a6432) -> bool {
        true
    }

    fn trans_PACIA1716(&mut self, _a: &mut arg_disas_a6432) -> bool {
        true
    }

    fn trans_PACIB1716(&mut self, _a: &mut arg_disas_a6432) -> bool {
        true
    }

    fn trans_AUTIA1716(&mut self, _a: &mut arg_disas_a6432) -> bool {
        true
    }

    fn trans_AUTIB1716(&mut self, _a: &mut arg_disas_a6432) -> bool {
        true
    }

    fn trans_PACIAZ(&mut self, _a: &mut arg_disas_a6432) -> bool {
        true
    }

    fn trans_PACIASP(&mut self, _a: &mut arg_disas_a6432) -> bool {
        true
    }

    fn trans_PACIBZ(&mut self, _a: &mut arg_disas_a6432) -> bool {
        true
    }

    fn trans_PACIBSP(&mut self, _a: &mut arg_disas_a6432) -> bool {
        true
    }

    fn trans_AUTIAZ(&mut self, _a: &mut arg_disas_a6432) -> bool {
        true
    }

    fn trans_AUTIASP(&mut self, _a: &mut arg_disas_a6432) -> bool {
        true
    }

    fn trans_AUTIBZ(&mut self, _a: &mut arg_disas_a6432) -> bool {
        true
    }

    fn trans_AUTIBSP(&mut self, _a: &mut arg_disas_a6432) -> bool {
        true
    }

    fn trans_WFI(&mut self, _a: &mut arg_disas_a6432) -> bool {
        self.b.is_jmp = DISAS_WFI;
        true
    }

    // Barriers

    fn trans_CLREX(&mut self, _a: &mut arg_disas_a6432) -> bool {
        let m1 = self.c64(-1);
        self.st_env64(m1, EXCLUSIVE_ADDR);
        true
    }

    fn trans_DSB_DMB(&mut self, a: &mut arg_disas_a6434) -> bool {
        // We handle DSB and DMB the same way
        let bar = match a.types {
            // MBReqTypes_Reads
            1 => mo::BAR_SC | mo::LD_LD | mo::LD_ST,
            // MBReqTypes_Writes
            2 => mo::BAR_SC | mo::ST_ST,
            // MBReqTypes_All
            _ => mo::BAR_SC | mo::ALL,
        };
        self.f().gen_mb(bar);
        true
    }

    fn trans_ISB(&mut self, _a: &mut arg_disas_a6432) -> bool {
        // We need to break the TB after this insn to execute self-modifying code correctly
        // and also to take any pending interrupts immediately.
        self.gen_goto_tb(0, 4);
        true
    }

    // MSR (immediate)

    fn trans_MSR_i_UAO(&mut self, a: &mut arg_i) -> bool {
        if !self.feat().uao || self.d.current_el == 0 {
            return false;
        }
        self.pstate_bit(PSTATE_UAO, a.imm & 1 != 0);
        self.b.is_jmp = DisasJumpType::TooMany;
        true
    }

    fn trans_MSR_i_PAN(&mut self, a: &mut arg_i) -> bool {
        if !self.feat().pan || self.d.current_el == 0 {
            return false;
        }
        self.pstate_bit(PSTATE_PAN, a.imm & 1 != 0);
        self.b.is_jmp = DisasJumpType::TooMany;
        true
    }

    fn trans_MSR_i_SPSEL(&mut self, a: &mut arg_i) -> bool {
        if self.d.current_el == 0 {
            return false;
        }
        self.msr_i_helper(&helpers::MSR_I_SPSEL, a.imm & PSTATE_SP as i32);
        self.b.is_jmp = DisasJumpType::TooMany;
        true
    }

    fn trans_MSR_i_DAIFSET(&mut self, a: &mut arg_i) -> bool {
        self.msr_i_helper(&helpers::MSR_I_DAIFSET, a.imm);
        self.b.is_jmp = DisasJumpType::TooMany;
        true
    }

    fn trans_MSR_i_DAIFCLEAR(&mut self, a: &mut arg_i) -> bool {
        self.msr_i_helper(&helpers::MSR_I_DAIFCLEAR, a.imm);
        // Exit the cpu loop to re-evaluate pending IRQs.
        self.b.is_jmp = DISAS_UPDATE_EXIT;
        true
    }

    // System

    fn trans_SYS(&mut self, a: &mut arg_disas_a6436) -> bool {
        self.handle_sys(
            a.l != 0,
            a.op0 as u32,
            a.op1 as u32,
            a.op2 as u32,
            a.crn as u32,
            a.crm as u32,
            a.rt,
        );
        true
    }

    // Exception generation

    fn trans_SVC(&mut self, a: &mut arg_i) -> bool {
        self.gen_exception_insn(4, EXCP_SWI, syn_aa64_svc(a.imm as u32));
        true
    }

    fn trans_BRK(&mut self, a: &mut arg_i) -> bool {
        // gen_exception_bkpt_insn(); the debug target EL is EL1.
        self.gen_exception_insn(0, EXCP_BKPT, syn_aa64_bkpt(a.imm as u32));
        true
    }

    fn trans_HVC(&mut self, a: &mut arg_i) -> bool {
        let target_el = if self.d.current_el == 3 { 3 } else { 2 };
        if self.d.current_el == 0 {
            self.unallocated_encoding();
            return true;
        }
        // The pre HVC helper handles cases when HVC gets trapped as an undefined insn by
        // runtime configuration.
        self.update_pc(0);
        let env = self.env();
        self.call(&helpers::PRE_HVC, None, &[env.into()]);
        self.gen_exception_insn_el(4, EXCP_HVC, syn_aa64_hvc(a.imm as u32), target_el);
        true
    }

    fn trans_SMC(&mut self, a: &mut arg_i) -> bool {
        if self.d.current_el == 0 {
            self.unallocated_encoding();
            return true;
        }
        self.update_pc(0);
        let env = self.env();
        let syn = self.c32(syn_aa64_smc(a.imm as u32) as i32);
        self.call(&helpers::PRE_SMC, None, &[env.into(), syn.into()]);
        self.gen_exception_insn_el(4, EXCP_SMC, syn_aa64_smc(a.imm as u32), 3);
        true
    }

    // HLT is UNDEFINED without semihosting: it keeps the default false.

    // Load/store exclusive

    fn trans_STXR(&mut self, a: &mut arg_stxr) -> bool {
        if a.lasr != 0 {
            self.f().gen_mb(mo::ALL | mo::BAR_STRL);
        }
        self.store_exclusive(a.rs, a.rt, a.rt2, a.rn, a.sz, false);
        true
    }

    fn trans_LDXR(&mut self, a: &mut arg_stxr) -> bool {
        self.load_exclusive(a.rt, a.rt2, a.rn, a.sz, false);
        if a.lasr != 0 {
            self.f().gen_mb(mo::ALL | mo::BAR_LDAQ);
        }
        true
    }

    fn trans_STXP(&mut self, a: &mut arg_stxr) -> bool {
        if a.lasr != 0 {
            self.f().gen_mb(mo::ALL | mo::BAR_STRL);
        }
        self.store_exclusive(a.rs, a.rt, a.rt2, a.rn, a.sz, true);
        true
    }

    fn trans_LDXP(&mut self, a: &mut arg_stxr) -> bool {
        self.load_exclusive(a.rt, a.rt2, a.rn, a.sz, true);
        if a.lasr != 0 {
            self.f().gen_mb(mo::ALL | mo::BAR_LDAQ);
        }
        true
    }

    fn trans_STLR(&mut self, a: &mut arg_stlr) -> bool {
        // StoreLORelease is the same as Store-Release for QEMU, but needs the feature-test.
        if a.lasr == 0 && !self.feat().lor {
            return false;
        }
        self.f().gen_mb(mo::ALL | mo::BAR_STRL);
        let memop = self.check_ordered_align(MemOp(a.sz as u32));
        let dirty = self.reg_sp(a.rn);
        let clean = self.clean_data_tbi(dirty);
        let rt = self.reg(a.rt);
        let idx = self.get_mem_index();
        self.f().gen_qemu_st_i64(rt, clean, idx, memop);
        true
    }

    fn trans_LDAR(&mut self, a: &mut arg_stlr) -> bool {
        // LoadLOAcquire is the same as Load-Acquire for QEMU.
        if a.lasr == 0 && !self.feat().lor {
            return false;
        }
        let memop = self.check_ordered_align(MemOp(a.sz as u32));
        let dirty = self.reg_sp(a.rn);
        let clean = self.clean_data_tbi(dirty);
        let idx = self.get_mem_index();
        let t = self.gpr_ld(clean, memop, false, idx);
        self.set_reg(a.rt, t);
        self.f().gen_mb(mo::ALL | mo::BAR_LDAQ);
        true
    }

    // Compare and swap

    fn trans_CAS(&mut self, a: &mut arg_disas_a6439) -> bool {
        if !self.feat().lse {
            return false;
        }
        // gen_compare_and_swap()
        let rs = self.reg(a.rs);
        let rt = self.reg(a.rt);
        let memop = self.check_atomic_align(MemOp(a.sz as u32));
        let dirty = self.reg_sp(a.rn);
        let clean = self.clean_data_tbi(dirty);
        let idx = self.get_mem_index();
        self.f().gen_atomic_cmpxchg_i64(rs, clean, rs, rt, idx, memop);
        self.set_reg(a.rs, rs);
        true
    }

    fn trans_CASP(&mut self, a: &mut arg_disas_a6439) -> bool {
        if !self.feat().lse {
            return false;
        }
        if (a.rt | a.rs) & 1 != 0 {
            return false;
        }
        // gen_compare_and_swap_pair()
        let s1 = self.reg(a.rs);
        let s2 = self.reg(a.rs + 1);
        let t1 = self.reg(a.rt);
        let t2 = self.reg(a.rt + 1);
        let memop = self.check_atomic_align(MemOp((a.sz + 1) as u32));
        let dirty = self.reg_sp(a.rn);
        let clean = self.clean_data_tbi(dirty);
        let idx = self.get_mem_index();
        if a.sz == 2 {
            let cmp = self.new64();
            let val = self.new64();
            let f = self.f();
            f.gen_concat32_i64(val, t1, t2);
            f.gen_concat32_i64(cmp, s1, s2);
            f.gen_atomic_cmpxchg_i64(cmp, clean, cmp, val, idx, memop);
            f.gen_extr32_i64(s1, s2, cmp);
        } else {
            let f = self.f();
            let cmp = f.temp_new_i128();
            let val = f.temp_new_i128();
            f.gen_concat_i64_i128(val, t1, t2);
            f.gen_concat_i64_i128(cmp, s1, s2);
            f.gen_atomic_cmpxchg_i128(cmp, clean, cmp, val, idx, memop);
            f.gen_extr_i128_i64(s1, s2, cmp);
        }
        self.set_reg(a.rs, s1);
        self.set_reg(a.rs + 1, s2);
        true
    }

    // Load register (literal)

    fn trans_LD_lit(&mut self, a: &mut arg_ldlit) -> bool {
        let memop = self.finalize_memop(MemOp((a.sz + 8 * a.sign) as u32));
        let addr = self.c64(self.d.pc_curr.wrapping_add(i64::from(a.imm) as u64) as i64);
        let idx = self.get_mem_index();
        let t = self.gpr_ld(addr, memop, false, idx);
        self.set_reg(a.rt, t);
        true
    }

    // Load/store pair

    fn trans_STP(&mut self, a: &mut arg_ldstpair) -> bool {
        let offset = i64::from(a.imm) << a.sz;
        let (dirty, clean) = self.addr_imm_pre(a.rn, offset, a.p != 0);
        let rt = self.reg(a.rt);
        let rt2 = self.reg(a.rt2);
        // The single paired access, aligned to the element size when SCTLR.A is set.
        let mut mop = MemOp((a.sz + 1) as u32);
        if self.d.align_mem {
            mop = mop | if a.sz == 2 { MemOp::ALIGN_4 } else { MemOp::ALIGN_8 };
        }
        let mop = self.finalize_memop_atom(mop, MemOp::ATOM_IFALIGN_PAIR);
        let idx = self.get_mem_index();
        if a.sz == 2 {
            let tmp = self.new64();
            let f = self.f();
            f.gen_concat32_i64(tmp, rt, rt2);
            f.gen_qemu_st_i64(tmp, clean, idx, mop);
        } else {
            let f = self.f();
            let tmp = f.temp_new_i128();
            f.gen_concat_i64_i128(tmp, rt, rt2);
            f.gen_qemu_st_i128(tmp, clean, idx, mop);
        }
        self.addr_imm_post(a.rn, dirty, offset, a.w != 0, a.p != 0);
        true
    }

    fn trans_LDP(&mut self, a: &mut arg_ldstpair) -> bool {
        let offset = i64::from(a.imm) << a.sz;
        let (dirty, clean) = self.addr_imm_pre(a.rn, offset, a.p != 0);
        // This treats sign-extending loads like zero-extending loads, since that reuses the
        // most code below.
        let mut mop = MemOp((a.sz + 1) as u32);
        if self.d.align_mem {
            mop = mop | if a.sz == 2 { MemOp::ALIGN_4 } else { MemOp::ALIGN_8 };
        }
        let mop = self.finalize_memop_atom(mop, MemOp::ATOM_IFALIGN_PAIR);
        let idx = self.get_mem_index();
        let rt = self.new64();
        let rt2 = self.new64();
        if a.sz == 2 {
            let f = self.f();
            f.gen_qemu_ld_i64(rt, clean, idx, mop);
            if a.sign != 0 {
                f.gen_sextract_i64(rt2, rt, 32, 32);
                f.gen_sextract_i64(rt, rt, 0, 32);
            } else {
                f.gen_extract_i64(rt2, rt, 32, 32);
                f.gen_extract_i64(rt, rt, 0, 32);
            }
        } else {
            let f = self.f();
            let tmp = f.temp_new_i128();
            f.gen_qemu_ld_i128(tmp, clean, idx, mop);
            f.gen_extr_i128_i64(rt, rt2, tmp);
        }
        self.set_reg(a.rt, rt);
        self.set_reg(a.rt2, rt2);
        self.addr_imm_post(a.rn, dirty, offset, a.w != 0, a.p != 0);
        true
    }

    // Load/store register (immediate)

    fn trans_STR_i(&mut self, a: &mut arg_ldst_imm) -> bool {
        let mop = self.finalize_memop(MemOp(a.sz as u32));
        let memidx = self.user_mem_index(a.unpriv != 0);
        let imm = i64::from(a.imm);
        let (dirty, clean) = self.addr_imm_pre(a.rn, imm, a.p != 0);
        let rt = self.reg(a.rt);
        self.f().gen_qemu_st_i64(rt, clean, memidx, mop);
        self.addr_imm_post(a.rn, dirty, imm, a.w != 0, a.p != 0);
        true
    }

    fn trans_LDR_i(&mut self, a: &mut arg_ldst_imm) -> bool {
        let mop = self.finalize_memop(MemOp((a.sz + 8 * a.sign) as u32));
        let memidx = self.user_mem_index(a.unpriv != 0);
        let imm = i64::from(a.imm);
        let (dirty, clean) = self.addr_imm_pre(a.rn, imm, a.p != 0);
        let t = self.gpr_ld(clean, mop, a.ext != 0, memidx);
        self.set_reg(a.rt, t);
        self.addr_imm_post(a.rn, dirty, imm, a.w != 0, a.p != 0);
        true
    }

    // Load/store register (register offset)

    fn trans_STR(&mut self, a: &mut arg_ldst) -> bool {
        if a.opt & 2 == 0 {
            return false;
        }
        let mop = self.finalize_memop(MemOp(a.sz as u32));
        let clean = self.addr_reg(a);
        let rt = self.reg(a.rt);
        let idx = self.get_mem_index();
        self.f().gen_qemu_st_i64(rt, clean, idx, mop);
        true
    }

    fn trans_LDR(&mut self, a: &mut arg_ldst) -> bool {
        if a.opt & 2 == 0 {
            return false;
        }
        let mop = self.finalize_memop(MemOp((a.sz + 8 * a.sign) as u32));
        let clean = self.addr_reg(a);
        let idx = self.get_mem_index();
        let t = self.gpr_ld(clean, mop, a.ext != 0, idx);
        self.set_reg(a.rt, t);
        true
    }

    // Atomic memory operations

    fn trans_LDADD(&mut self, a: &mut arg_atomic) -> bool {
        self.atomic_ld(a, AtomicOp::FetchAdd, false, false)
    }

    fn trans_LDCLR(&mut self, a: &mut arg_atomic) -> bool {
        self.atomic_ld(a, AtomicOp::FetchAnd, false, true)
    }

    fn trans_LDEOR(&mut self, a: &mut arg_atomic) -> bool {
        self.atomic_ld(a, AtomicOp::FetchXor, false, false)
    }

    fn trans_LDSET(&mut self, a: &mut arg_atomic) -> bool {
        self.atomic_ld(a, AtomicOp::FetchOr, false, false)
    }

    fn trans_LDSMAX(&mut self, a: &mut arg_atomic) -> bool {
        self.atomic_ld(a, AtomicOp::FetchSmax, true, false)
    }

    fn trans_LDSMIN(&mut self, a: &mut arg_atomic) -> bool {
        self.atomic_ld(a, AtomicOp::FetchSmin, true, false)
    }

    fn trans_LDUMAX(&mut self, a: &mut arg_atomic) -> bool {
        self.atomic_ld(a, AtomicOp::FetchUmax, false, false)
    }

    fn trans_LDUMIN(&mut self, a: &mut arg_atomic) -> bool {
        self.atomic_ld(a, AtomicOp::FetchUmin, false, false)
    }

    fn trans_SWP(&mut self, a: &mut arg_atomic) -> bool {
        self.atomic_ld(a, AtomicOp::Xchg, false, false)
    }

    fn trans_LDAPR(&mut self, a: &mut arg_disas_a6445) -> bool {
        let feat = self.feat();
        if !feat.lse || !feat.rcpc {
            return false;
        }
        let mop = self.check_ordered_align(MemOp(a.sz as u32));
        let dirty = self.reg_sp(a.rn);
        let clean = self.clean_data_tbi(dirty);
        let idx = self.get_mem_index();
        // LDAPR* are a special case because they are a simple load, not a
        // fetch-and-do-something op. The architectural consistency requirements here are
        // weaker than full load-acquire (we only need "load-acquire processor consistent"),
        // but we choose to implement them as full LDAQ.
        let t = self.gpr_ld(clean, mop, false, idx);
        self.set_reg(a.rt, t);
        self.f().gen_mb(mo::ALL | mo::BAR_LDAQ);
        true
    }

    // Data-processing (2 source)

    fn trans_UDIV(&mut self, a: &mut arg_rrr_sf) -> bool {
        self.div(a, false)
    }

    fn trans_SDIV(&mut self, a: &mut arg_rrr_sf) -> bool {
        self.div(a, true)
    }

    fn trans_LSLV(&mut self, a: &mut arg_rrr_sf) -> bool {
        self.shift_by_reg(a, SHIFT_LSL)
    }

    fn trans_LSRV(&mut self, a: &mut arg_rrr_sf) -> bool {
        self.shift_by_reg(a, SHIFT_LSR)
    }

    fn trans_ASRV(&mut self, a: &mut arg_rrr_sf) -> bool {
        self.shift_by_reg(a, SHIFT_ASR)
    }

    fn trans_RORV(&mut self, a: &mut arg_rrr_sf) -> bool {
        self.shift_by_reg(a, 3)
    }

    fn trans_CRC32(&mut self, a: &mut arg_rrr_e) -> bool {
        self.crc32(a, false)
    }

    fn trans_CRC32C(&mut self, a: &mut arg_rrr_e) -> bool {
        self.crc32(a, true)
    }

    // Data-processing (1 source)

    fn trans_RBIT(&mut self, a: &mut arg_rr_sf) -> bool {
        let rn = self.reg(a.rn);
        let rd = self.new64();
        self.call(&helpers::RBIT64, Some(rd.into()), &[rn.into()]);
        if a.sf == 0 {
            // gen_rbit32(): the reversed low word is the high word of the 64 bit reversal.
            self.f().gen_shri_i64(rd, rd, 32);
        }
        self.set_reg(a.rd, rd);
        true
    }

    fn trans_REV16(&mut self, a: &mut arg_rr_sf) -> bool {
        // gen_rev16()
        let sf = a.sf != 0;
        let rn = self.read_cpu_reg(a.rn, sf);
        let tmp = self.new64();
        let rd = self.new64();
        let mask = self.c64(if sf { 0x00ff_00ff_00ff_00ff } else { 0x00ff_00ff });
        let f = self.f();
        f.gen_shri_i64(tmp, rn, 8);
        f.gen_and_i64(rd, rn, mask);
        f.gen_and_i64(tmp, tmp, mask);
        f.gen_shli_i64(rd, rd, 8);
        f.gen_or_i64(rd, rd, tmp);
        self.set_reg(a.rd, rd);
        true
    }

    fn trans_REV32(&mut self, a: &mut arg_rr_sf) -> bool {
        let rn = self.reg(a.rn);
        let rd = self.new64();
        let f = self.f();
        if a.sf != 0 {
            // gen_rev32()
            f.gen_bswap64_i64(rd, rn);
            f.gen_rotri_i64(rd, rd, 32);
        } else {
            f.gen_bswap32_i64(rd, rn, ruvm_jit_core::types::bswap::OZ);
        }
        self.set_reg(a.rd, rd);
        true
    }

    fn trans_REV64(&mut self, a: &mut arg_rr) -> bool {
        let rn = self.reg(a.rn);
        let rd = self.new64();
        self.f().gen_bswap64_i64(rd, rn);
        self.set_reg(a.rd, rd);
        true
    }

    fn trans_CLZ(&mut self, a: &mut arg_rr_sf) -> bool {
        let rn = self.reg(a.rn);
        let rd = self.new64();
        if a.sf != 0 {
            self.f().gen_clzi_i64(rd, rn, 64);
        } else {
            let t = self.new32();
            let f = self.f();
            f.gen_extrl_i64_i32(t, rn);
            f.gen_clzi_i32(t, t, 32);
            f.gen_extu_i32_i64(rd, t);
        }
        self.set_reg(a.rd, rd);
        true
    }

    fn trans_CLS(&mut self, a: &mut arg_rr_sf) -> bool {
        let rn = self.reg(a.rn);
        let rd = self.new64();
        if a.sf != 0 {
            self.f().gen_clrsb_i64(rd, rn);
        } else {
            let t = self.new32();
            let f = self.f();
            f.gen_extrl_i64_i32(t, rn);
            f.gen_clrsb_i32(t, t);
            f.gen_extu_i32_i64(rd, t);
        }
        self.set_reg(a.rd, rd);
        true
    }

    // Logical (shifted register)

    fn trans_AND_r(&mut self, a: &mut arg_logic_shift) -> bool {
        self.logic_reg(a, Logic::And, false)
    }

    fn trans_ANDS_r(&mut self, a: &mut arg_logic_shift) -> bool {
        self.logic_reg(a, Logic::And, true)
    }

    fn trans_ORR_r(&mut self, a: &mut arg_logic_shift) -> bool {
        self.logic_reg(a, Logic::Or, false)
    }

    fn trans_EOR_r(&mut self, a: &mut arg_logic_shift) -> bool {
        self.logic_reg(a, Logic::Xor, false)
    }

    // Add/subtract (shifted and extended register)

    fn trans_ADD_r(&mut self, a: &mut arg_addsub_shift) -> bool {
        self.addsub_reg(a, false, false)
    }

    fn trans_SUB_r(&mut self, a: &mut arg_addsub_shift) -> bool {
        self.addsub_reg(a, true, false)
    }

    fn trans_ADDS_r(&mut self, a: &mut arg_addsub_shift) -> bool {
        self.addsub_reg(a, false, true)
    }

    fn trans_SUBS_r(&mut self, a: &mut arg_addsub_shift) -> bool {
        self.addsub_reg(a, true, true)
    }

    fn trans_ADD_ext(&mut self, a: &mut arg_addsub_ext) -> bool {
        self.addsub_ext(a, false, false)
    }

    fn trans_SUB_ext(&mut self, a: &mut arg_addsub_ext) -> bool {
        self.addsub_ext(a, true, false)
    }

    fn trans_ADDS_ext(&mut self, a: &mut arg_addsub_ext) -> bool {
        self.addsub_ext(a, false, true)
    }

    fn trans_SUBS_ext(&mut self, a: &mut arg_addsub_ext) -> bool {
        self.addsub_ext(a, true, true)
    }

    // Add/subtract (with carry)

    fn trans_ADC(&mut self, a: &mut arg_rrr_sf) -> bool {
        self.adc_sbc(a, false, false)
    }

    fn trans_ADCS(&mut self, a: &mut arg_rrr_sf) -> bool {
        self.adc_sbc(a, false, true)
    }

    fn trans_SBC(&mut self, a: &mut arg_rrr_sf) -> bool {
        self.adc_sbc(a, true, false)
    }

    fn trans_SBCS(&mut self, a: &mut arg_rrr_sf) -> bool {
        self.adc_sbc(a, true, true)
    }

    // Conditional compare

    fn trans_CCMP(&mut self, a: &mut arg_disas_a6461) -> bool {
        let t0 = self.new32();
        let t1 = self.new32();
        let tmp = self.new64();

        // Set T0 = !COND.
        let (cond, value) = self.test_cc(a.cond);
        self.f().gen_setcondi_i32(cond.invert(), t0, value, 0);

        // Load the arguments for the new comparison.
        let y = if a.imm != 0 { self.c64(i64::from(a.y)) } else { self.reg(a.y) };
        let rn = self.reg(a.rn);

        // Set the flags for the new comparison.
        let sf = a.sf != 0;
        if a.op != 0 {
            self.sub_cc(sf, tmp, rn, y);
        } else {
            self.add_cc(sf, tmp, rn, y);
        }

        // If COND was false, force the flags to #nzcv. Compute a mask to help with this:
        // T1 = (COND ? 0 : -1).
        self.f().gen_neg_i32(t1, t0);
        let nzcv = a.nzcv;
        let nf = self.ld_env32(NF);
        let zf = self.ld_env32(ZF);
        let cf = self.ld_env32(CF);
        let vf = self.ld_env32(VF);
        let f = self.f();
        if nzcv & 8 != 0 {
            // N
            f.gen_or_i32(nf, nf, t1);
        } else {
            f.gen_andc_i32(nf, nf, t1);
        }
        if nzcv & 4 != 0 {
            // Z
            f.gen_andc_i32(zf, zf, t1);
        } else {
            f.gen_or_i32(zf, zf, t0);
        }
        if nzcv & 2 != 0 {
            // C
            f.gen_or_i32(cf, cf, t0);
        } else {
            f.gen_andc_i32(cf, cf, t1);
        }
        if nzcv & 1 != 0 {
            // V
            f.gen_or_i32(vf, vf, t1);
        } else {
            f.gen_andc_i32(vf, vf, t1);
        }
        self.st_env32(nf, NF);
        self.st_env32(zf, ZF);
        self.st_env32(cf, CF);
        self.st_env32(vf, VF);
        true
    }

    // Conditional select

    fn trans_CSEL(&mut self, a: &mut arg_disas_a6462) -> bool {
        let rd = self.new64();
        let zero = self.c64(0);
        let (cond, value) = self.test_cc64(a.cond);
        if a.rn == 31 && a.rm == 31 && (a.else_inc ^ a.else_inv) != 0 {
            // CSET & CSETM.
            let f = self.f();
            if a.else_inv != 0 {
                f.gen_negsetcond_i64(cond.invert(), rd, value, zero);
            } else {
                f.gen_setcond_i64(cond.invert(), rd, value, zero);
            }
        } else {
            let t_true = self.reg(a.rn);
            let t_false = self.read_cpu_reg(a.rm, true);
            let f = self.f();
            if a.else_inv != 0 && a.else_inc != 0 {
                f.gen_neg_i64(t_false, t_false);
            } else if a.else_inv != 0 {
                f.gen_not_i64(t_false, t_false);
            } else if a.else_inc != 0 {
                f.gen_addi_i64(t_false, t_false, 1);
            }
            f.gen_movcond_i64(cond, rd, value, zero, t_true, t_false);
        }
        self.set_reg_sf(a.rd, rd, a.sf != 0);
        true
    }

    // Data-processing (3 source)

    fn trans_MADD_w(&mut self, a: &mut arg_rrrr) -> bool {
        self.muladd(a, false, false, None)
    }

    fn trans_MSUB_w(&mut self, a: &mut arg_rrrr) -> bool {
        self.muladd(a, false, true, None)
    }

    fn trans_MADD_x(&mut self, a: &mut arg_rrrr) -> bool {
        self.muladd(a, true, false, None)
    }

    fn trans_MSUB_x(&mut self, a: &mut arg_rrrr) -> bool {
        self.muladd(a, true, true, None)
    }

    fn trans_SMADDL(&mut self, a: &mut arg_rrrr) -> bool {
        self.muladd(a, true, false, Some(MemOp::SL))
    }

    fn trans_SMSUBL(&mut self, a: &mut arg_rrrr) -> bool {
        self.muladd(a, true, true, Some(MemOp::SL))
    }

    fn trans_UMADDL(&mut self, a: &mut arg_rrrr) -> bool {
        self.muladd(a, true, false, Some(MemOp::UL))
    }

    fn trans_UMSUBL(&mut self, a: &mut arg_rrrr) -> bool {
        self.muladd(a, true, true, Some(MemOp::UL))
    }

    fn trans_SMULH(&mut self, a: &mut arg_rrr) -> bool {
        let rn = self.reg(a.rn);
        let rm = self.reg(a.rm);
        let discard = self.new64();
        let rd = self.new64();
        self.f().gen_muls2_i64(discard, rd, rn, rm);
        self.set_reg(a.rd, rd);
        true
    }

    fn trans_UMULH(&mut self, a: &mut arg_rrr) -> bool {
        let rn = self.reg(a.rn);
        let rm = self.reg(a.rm);
        let discard = self.new64();
        let rd = self.new64();
        self.f().gen_mulu2_i64(discard, rd, rn, rm);
        self.set_reg(a.rd, rd);
        true
    }
}

impl TranslatorOps for DisasContext {
    fn init_disas_context(&mut self, db: &mut DisasContextBase<'_>, _cpu: &mut Cpu<'_>) {
        let flags = db.tb.flags;
        self.current_el = flags & TB_EL_MASK;
        self.pstate_il = flags & TB_PSTATE_IL != 0;
        self.unpriv = flags & TB_UNPRIV != 0;
        self.align_mem = flags & TB_ALIGN_MEM != 0;
        self.tbii = (flags >> TB_TBII_SHIFT) & 3;
        self.tbid = (flags >> TB_TBID_SHIFT) & 3;
        self.fp_excp_el = (flags >> TB_FPEXC_EL_SHIFT) & 3;
        self.mmu_idx = (flags >> TB_MMUIDX_SHIFT) & 0xf;
        self.e2h = flags & TB_E2H != 0;
        self.sve_excp_el = (flags >> TB_SVEEXC_EL_SHIFT) & 3;
        self.vl = (((flags >> TB_VL_SHIFT) & 0xf) + 1) * 16;

        // Bound the number of insns to execute to those left on the page.
        let bound = (db.pc_first | !0xfff).wrapping_neg() / 4;
        db.max_insns = db.max_insns.min(bound as u32);
    }

    fn insn_start(&mut self, db: &mut DisasContextBase<'_>, _cpu: &mut Cpu<'_>) {
        let pc = db.pc_next;
        db.tb.f.gen_insn_start(&[pc]);
    }

    fn translate_insn(
        &mut self,
        db: &mut DisasContextBase<'_>,
        cpu: &mut Cpu<'_>,
    ) -> Result<(), CpuLoopExit> {
        let pc = db.pc_next;
        self.pc_curr = pc;
        if pc & 3 != 0 {
            // We can't instruction fetch from a misaligned PC: raise the PC alignment fault
            // without fetching.
            let mut s = S { d: self, b: db };
            s.update_pc(0);
            let env = s.env();
            let t = s.c64(pc as i64);
            s.call(&helpers::EXCEPTION_PC_ALIGNMENT, None, &[env.into(), t.into()]);
            s.b.is_jmp = DisasJumpType::NoReturn;
            s.b.pc_next = (pc + 3) & !3;
            return Ok(());
        }

        let insn = db.translator_ldl(cpu, pc, Endian::Little)?;
        db.pc_next = pc + 4;
        let mut s = S { d: self, b: db };

        // Illegal execution state. This has priority over BTI exceptions, but comes after
        // instruction abort exceptions.
        if s.d.pstate_il {
            s.gen_exception_insn(0, EXCP_UDEF, syn_illegalstate());
            return Ok(());
        }

        if !decode::disas(&mut s, insn) && !sve::disas(&mut s, insn) {
            s.unallocated_encoding();
        }
        Ok(())
    }

    fn tb_stop(&mut self, db: &mut DisasContextBase<'_>, _cpu: &mut Cpu<'_>) {
        let mut s = S { d: self, b: db };
        match s.b.is_jmp {
            DisasJumpType::Next | DisasJumpType::TooMany => s.gen_goto_tb(1, 4),
            DisasJumpType::NoReturn => {}
            DISAS_JUMP => s.f().gen_lookup_and_goto_ptr(),
            DISAS_EXIT => s.f().gen_exit_tb(0, 0),
            DISAS_WFI => {
                s.update_pc(4);
                let env = s.env();
                let len = s.c32(4);
                s.call(&helpers::WFI, None, &[env.into(), len.into()]);
                // The helper doesn't necessarily throw an exception, but we must go back to
                // the main loop to check for interrupts anyway.
                s.f().gen_exit_tb(0, 0);
            }
            // DISAS_UPDATE_EXIT and anything else.
            _ => {
                s.update_pc(4);
                s.f().gen_exit_tb(0, 0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::logic_imm_decode_wmask;

    #[test]
    fn logic_imm() {
        // 64 bit element, 1 bit set, no rotation.
        assert_eq!(logic_imm_decode_wmask(1, 0, 0), Some(1));
        // 0x5555...: 2 bit elements with one bit set.
        assert_eq!(logic_imm_decode_wmask(0, 0b111100, 0), Some(0x5555_5555_5555_5555));
        // 0xff00ff00...: 16 bit elements with 8 bits set rotated by 8.
        assert_eq!(logic_imm_decode_wmask(0, 0b100111, 8), Some(0xff00_ff00_ff00_ff00));
        // 32 bit elements, 31 bits set rotated by 1: 0x7fffffff rotated is 0xbfffffff.
        assert_eq!(logic_imm_decode_wmask(0, 30, 1), Some(0xbfff_ffff_bfff_ffff));
        // Reserved: all ones run, and immn == 0 with imms == 11111x.
        assert_eq!(logic_imm_decode_wmask(1, 63, 0), None);
        assert_eq!(logic_imm_decode_wmask(0, 0b111110, 0), None);
    }
}
