// SPDX-License-Identifier: GPL-2.0-or-later

//! The Advanced SIMD instructions: the integer part of QEMU's `translate-neon.c`, with the
//! decoders of neon-dp.decode and neon-ls.decode.
//!
//! The data processing instructions go through the one `a32_neon` helper, which runs the
//! A64 element code on D register images; the moves, the immediates and the loads and stores
//! are inline. As for VFP, the access checks always pass and every register D0 to D31
//! exists.
//!
//! Not here yet, and so UNDEF: the floating point operations (including VRECPE, VRSQRTE,
//! the conversions and VRINT), the AES and SHA instructions, and neon-shared.decode.

use ruvm_jit_core::MemOp;
use ruvm_jit_core::ir::{TempI32, TempI64};

use super::S;
use super::fp::vfp_reg_offset;
use crate::tcg::a32_neon::{self, OUT_Q, PAIR};
use crate::tcg::translate::asimd_imm_const;
use crate::tcg::vec_helper::{Desc, Regs, op};

#[allow(missing_docs, unreachable_pub, clippy::pedantic, clippy::nursery)]
mod dp {
    include!(concat!(env!("OUT_DIR"), "/neon_dp_decode.rs"));
}

#[allow(missing_docs, unreachable_pub, clippy::pedantic, clippy::nursery)]
mod ls {
    include!(concat!(env!("OUT_DIR"), "/neon_ls_decode.rs"));
}

use dp::*;
use ls::*;

/// `disas_neon_dp()`.
pub(super) fn disas_dp(s: &mut S<'_, '_>, insn: u32) -> bool {
    disas_neon_dp(s, insn)
}

/// `disas_neon_ls()`.
pub(super) fn disas_ls(s: &mut S<'_, '_>, insn: u32) -> bool {
    disas_neon_ls(s, insn)
}

/// The `env` offset of D`reg`.
fn d_off(reg: i32) -> usize {
    vfp_reg_offset(true, reg)
}

/// `pow2_align()`.
const fn pow2_align(i: i32) -> MemOp {
    MemOp((i as u32) << 5)
}

/// The `neon_ls_element_type[]` table: the registers, interleave and spacing of each type of
/// VLDn and VSTn (multiple structures).
const ELEMENT_TYPE: [(i32, i32, i32); 11] = [
    (1, 4, 1),
    (1, 4, 2),
    (4, 1, 1),
    (2, 2, 2),
    (1, 3, 1),
    (1, 3, 2),
    (3, 1, 1),
    (1, 1, 1),
    (1, 2, 1),
    (1, 2, 2),
    (2, 1, 1),
];

/// The [`OUT_Q`] flag for a Q form.
const fn q_flag(q: i32) -> u32 {
    if q != 0 { OUT_Q } else { 0 }
}

/// The operation size of a D or Q form.
const fn vec_size(q: i32) -> usize {
    if q != 0 { 16 } else { 8 }
}

/// `neon_get_scalar()`: the D register and element index of the scalar operand `vm` of
/// size `size`.
const fn scalar(size: i32, vm: i32) -> (i32, i32) {
    if size == 1 { (vm & 7, ((vm >> 4) << 1) | ((vm >> 3) & 1)) } else { (vm & 15, vm >> 4) }
}

impl S<'_, '_> {
    /// Call `a32_neon` for `desc` on D`n`, D`m` and D`d`, with the flags of `a32_neon`.
    fn neon_op(&mut self, desc: Desc, d: i32, n: i32, m: i32, flags: u32) {
        let regs = self.c32((Regs::pack(d, n, m, 0) | flags) as i32);
        let desc = self.c32(desc.pack() as i32);
        let env = self.env();
        self.call(&a32_neon::NEON, None, &[env.into(), regs.into(), desc.into()]);
    }

    /// `do_3same()`, with `no64` for the forms that UNDEF on 64-bit elements.
    fn same3(&mut self, a: &arg_3same, op: u32, no64: bool) -> bool {
        if (a.vn | a.vm | a.vd) & a.q != 0 || (no64 && a.size == 3) {
            return false;
        }
        self.neon_op(Desc::new(op, a.size, vec_size(a.q)), a.vd, a.vn, a.vm, q_flag(a.q));
        true
    }

    /// `do_vector_2sh()`.
    fn shift2(&mut self, a: &arg_2reg_shift, op: u32) -> bool {
        if (a.vm | a.vd) & a.q != 0 {
            return false;
        }
        let desc = Desc::new(op, a.size, vec_size(a.q)).imm(a.shift);
        self.neon_op(desc, a.vd, a.vm, a.vm, q_flag(a.q));
        true
    }

    /// `do_2shift_narrow_32()` and `do_2shift_narrow_64()`: `size` is the source element
    /// size.
    fn shift2_narrow(&mut self, a: &arg_2reg_shift, op: u32) -> bool {
        if a.vm & 1 != 0 {
            return false;
        }
        let desc = Desc::new(op, a.size - 1, 8).imm(a.shift);
        self.neon_op(desc, a.vd, a.vm, a.vm, 0);
        true
    }

    /// `do_vshll_2sh()`.
    fn shift2_long(&mut self, a: &arg_2reg_shift, op: u32) -> bool {
        if a.vd & 1 != 0 {
            return false;
        }
        let desc = Desc::new(op, a.size, 16).imm(a.shift);
        self.neon_op(desc, a.vd, a.vm, a.vm, OUT_Q);
        true
    }

    /// The three registers of different lengths forms: `wide_n` for the VADDW and VSUBW
    /// forms, whose Vn is a Q register, `narrow` for the forms with a D result.
    fn diff3(&mut self, a: &arg_3diff, op: u32, wide_n: bool, narrow: bool) -> bool {
        // Size 3 is a different group, tried first.
        if a.size == 3 {
            return false;
        }
        let bad = if narrow {
            (a.vn | a.vm) & 1 != 0
        } else {
            a.vd & 1 != 0 || (wide_n && a.vn & 1 != 0)
        };
        if bad {
            return false;
        }
        let flags = if narrow { 0 } else { OUT_Q };
        self.neon_op(Desc::new(op, a.size, 16), a.vd, a.vn, a.vm, flags);
        true
    }

    /// The VQDMULL, VQDMLAL and VQDMLSL forms, which only have 16 and 32-bit elements.
    fn diff3_qd(&mut self, a: &arg_3diff, op: u32) -> bool {
        a.size != 0 && self.diff3(a, op, false, false)
    }

    /// `do_2scalar()` and `do_2scalar_long()`.
    fn scalar2(&mut self, a: &arg_2scalar, op: u32, long: bool) -> bool {
        // Size 0 has no scalar form, size 3 is a different group.
        if a.size == 0 || a.size == 3 {
            return false;
        }
        let bad = if long { a.vd & 1 != 0 } else { a.q != 0 && (a.vd | a.vn) & 1 != 0 };
        if bad {
            return false;
        }
        let (vm, idx) = scalar(a.size, a.vm);
        if long {
            self.neon_op(Desc::new(op, a.size, 16).indexed(idx), a.vd, a.vn, vm, OUT_Q);
        } else {
            let desc = Desc::new(op, a.size, vec_size(a.q)).indexed(idx);
            self.neon_op(desc, a.vd, a.vn, vm, q_flag(a.q));
        }
        true
    }

    /// `do_2misc_vec()` and friends: `ok` says the size is allowed.
    fn misc2(&mut self, a: &arg_2misc, op: u32, ok: bool) -> bool {
        if !ok || (a.vd | a.vm) & a.q != 0 {
            return false;
        }
        self.neon_op(Desc::new(op, a.size, vec_size(a.q)), a.vd, a.vm, a.vm, q_flag(a.q));
        true
    }

    /// VTRN, VUZP and VZIP: operation `op` gives the new Vd and `op + 1` the new Vm.
    fn misc2_pair(&mut self, a: &arg_2misc, op: u32, ok: bool) -> bool {
        if !ok || (a.vd | a.vm) & a.q != 0 {
            return false;
        }
        let desc = Desc::new(op, a.size, vec_size(a.q));
        self.neon_op(desc, a.vd, a.vd, a.vm, q_flag(a.q) | PAIR);
        true
    }

    /// `do_vmovn()`.
    fn movn(&mut self, a: &arg_2misc, op: u32) -> bool {
        if a.vm & 1 != 0 || a.size == 3 {
            return false;
        }
        self.neon_op(Desc::new(op, a.size, 8), a.vd, a.vm, a.vm, 0);
        true
    }

    /// `t` (the low `8 << size` bits) repeated over 64 bits.
    pub(super) fn dup64(&mut self, t: TempI32, size: i32) -> TempI64 {
        let x = self.new64();
        let f = self.f();
        f.gen_extu_i32_i64(x, t);
        match size {
            0 => {
                f.gen_andi_i64(x, x, 0xff);
                f.gen_muli_i64(x, x, 0x0101_0101_0101_0101);
            }
            1 => {
                f.gen_andi_i64(x, x, 0xffff);
                f.gen_muli_i64(x, x, 0x0001_0001_0001_0001);
            }
            _ => f.gen_muli_i64(x, x, 0x1_0000_0001),
        }
        x
    }

    /// Write `x` to D`reg`, and to D`reg+1` too if `q`.
    pub(super) fn st_dreg_q(&mut self, x: TempI64, reg: i32, q: bool) {
        self.st_env64(x, d_off(reg));
        if q {
            self.st_env64(x, d_off(reg + 1));
        }
    }

    /// Load the element at `off` of `size` into the zero extended `t`.
    fn ld_elem64(&mut self, t: TempI64, off: usize, size: i32) {
        let env = self.env();
        let off = off as i64;
        let f = self.f();
        match size {
            0 => f.gen_ld8u_i64(t, env, off),
            1 => f.gen_ld16u_i64(t, env, off),
            2 => f.gen_ld32u_i64(t, env, off),
            _ => f.gen_ld_i64(t, env, off),
        }
    }

    /// Store the low `8 << size` bits of `t` to the element at `off`.
    fn st_elem64(&mut self, t: TempI64, off: usize, size: i32) {
        let env = self.env();
        let off = off as i64;
        let f = self.f();
        match size {
            0 => f.gen_st8_i64(t, env, off),
            1 => f.gen_st16_i64(t, env, off),
            2 => f.gen_st32_i64(t, env, off),
            _ => f.gen_st_i64(t, env, off),
        }
    }

    /// Load the element at `off` of `size` (up to 32 bits) into the zero extended `t`.
    fn ld_elem32(&mut self, t: TempI32, off: usize, size: i32) {
        let env = self.env();
        let off = off as i64;
        let f = self.f();
        match size {
            0 => f.gen_ld8u_i32(t, env, off),
            1 => f.gen_ld16u_i32(t, env, off),
            _ => f.gen_ld_i32(t, env, off),
        }
    }

    /// Store the low `8 << size` bits (up to 32) of `t` to the element at `off`.
    fn st_elem32(&mut self, t: TempI32, off: usize, size: i32) {
        let env = self.env();
        let off = off as i64;
        let f = self.f();
        match size {
            0 => f.gen_st8_i32(t, env, off),
            1 => f.gen_st16_i32(t, env, off),
            _ => f.gen_st_i32(t, env, off),
        }
    }

    /// `gen_neon_ldst_base_update()`.
    fn ldst_base_update(&mut self, rm: i32, rn: i32, stride: i32) {
        if rm != 15 {
            let base = self.load_reg(rn);
            if rm == 13 {
                self.f().gen_addi_i32(base, base, stride);
            } else {
                let index = self.load_reg(rm);
                self.f().gen_add_i32(base, base, index);
            }
            self.store_reg(rn, base);
        }
    }
}

macro_rules! same3 {
    ($($name:ident => $op:expr, $no64:expr;)*) => {
        $(fn $name(&mut self, a: &mut arg_3same) -> bool {
            self.same3(a, $op, $no64)
        })*
    };
}

macro_rules! shift2 {
    ($f:ident: $($name:ident => $op:expr;)*) => {
        $(fn $name(&mut self, a: &mut arg_2reg_shift) -> bool {
            self.$f(a, $op)
        })*
    };
}

macro_rules! diff3 {
    ($($name:ident => $op:expr, $wide_n:expr, $narrow:expr;)*) => {
        $(fn $name(&mut self, a: &mut arg_3diff) -> bool {
            self.diff3(a, $op, $wide_n, $narrow)
        })*
    };
}

macro_rules! scalar2 {
    ($($name:ident => $op:expr, $long:expr;)*) => {
        $(fn $name(&mut self, a: &mut arg_2scalar) -> bool {
            self.scalar2(a, $op, $long)
        })*
    };
}

macro_rules! misc2 {
    ($f:ident: $($name:ident => $op:expr, |$s:ident, $q:ident| $ok:expr;)*) => {
        $(fn $name(&mut self, a: &mut arg_2misc) -> bool {
            let ($s, $q) = (a.size, a.q);
            let _ = $q;
            self.$f(a, $op, $ok)
        })*
    };
}

impl DisasNeonDp for S<'_, '_> {
    fn rsub_8(&mut self, x: i32) -> i32 {
        8 - x
    }

    fn rsub_16(&mut self, x: i32) -> i32 {
        16 - x
    }

    fn rsub_32(&mut self, x: i32) -> i32 {
        32 - x
    }

    fn rsub_64(&mut self, x: i32) -> i32 {
        64 - x
    }

    fn neon_3same_fp_size(&mut self, x: i32) -> i32 {
        // 0 is single precision and 1 half precision.
        2 - x
    }

    same3! {
        trans_VADD_3s => op::ADD, false;
        trans_VSUB_3s => op::SUB, false;
        trans_VAND_3s => op::AND, false;
        trans_VBIC_3s => op::BIC, false;
        trans_VORR_3s => op::ORR, false;
        trans_VORN_3s => op::ORN, false;
        trans_VEOR_3s => op::EOR, false;
        trans_VBSL_3s => op::BSL, false;
        trans_VBIT_3s => op::BIT, false;
        trans_VBIF_3s => op::BIF, false;
        trans_VSHL_S_3s => op::SSHL, false;
        trans_VSHL_U_3s => op::USHL, false;
        trans_VQADD_S_3s => op::SQADD, false;
        trans_VQADD_U_3s => op::UQADD, false;
        trans_VQSUB_S_3s => op::SQSUB, false;
        trans_VQSUB_U_3s => op::UQSUB, false;
        trans_VRSHL_S_3s => op::SRSHL, false;
        trans_VRSHL_U_3s => op::URSHL, false;
        trans_VQSHL_S_3s => op::SQSHL, false;
        trans_VQSHL_U_3s => op::UQSHL, false;
        trans_VQRSHL_S_3s => op::SQRSHL, false;
        trans_VQRSHL_U_3s => op::UQRSHL, false;
        trans_VMAX_S_3s => op::SMAX, true;
        trans_VMAX_U_3s => op::UMAX, true;
        trans_VMIN_S_3s => op::SMIN, true;
        trans_VMIN_U_3s => op::UMIN, true;
        trans_VMUL_3s => op::MUL, true;
        trans_VMLA_3s => op::MLA, true;
        trans_VMLS_3s => op::MLS, true;
        trans_VTST_3s => op::CMTST, true;
        trans_VABD_S_3s => op::SABD, true;
        trans_VABA_S_3s => op::SABA, true;
        trans_VABD_U_3s => op::UABD, true;
        trans_VABA_U_3s => op::UABA, true;
        trans_VPADD_3s => op::ADDP, true;
        trans_VPMAX_S_3s => op::SMAXP, true;
        trans_VPMIN_S_3s => op::SMINP, true;
        trans_VPMAX_U_3s => op::UMAXP, true;
        trans_VPMIN_U_3s => op::UMINP, true;
        trans_VHADD_S_3s => op::SHADD, true;
        trans_VHADD_U_3s => op::UHADD, true;
        trans_VHSUB_S_3s => op::SHSUB, true;
        trans_VHSUB_U_3s => op::UHSUB, true;
        trans_VRHADD_S_3s => op::SRHADD, true;
        trans_VRHADD_U_3s => op::URHADD, true;
        trans_VCGT_S_3s => op::CMGT, true;
        trans_VCGT_U_3s => op::CMHI, true;
        trans_VCGE_S_3s => op::CMGE, true;
        trans_VCGE_U_3s => op::CMHS, true;
        trans_VCEQ_3s => op::CMEQ, true;
    }

    fn trans_VMUL_p_3s(&mut self, a: &mut arg_3same) -> bool {
        a.size == 0 && self.same3(a, op::PMUL, false)
    }

    fn trans_VQDMULH_3s(&mut self, a: &mut arg_3same) -> bool {
        (1..=2).contains(&a.size) && self.same3(a, op::SQDMULH, false)
    }

    fn trans_VQRDMULH_3s(&mut self, a: &mut arg_3same) -> bool {
        (1..=2).contains(&a.size) && self.same3(a, op::SQRDMULH, false)
    }

    fn trans_VQRDMLAH_3s(&mut self, a: &mut arg_3same) -> bool {
        (1..=2).contains(&a.size) && self.same3(a, op::SQRDMLAH, false)
    }

    fn trans_VQRDMLSH_3s(&mut self, a: &mut arg_3same) -> bool {
        (1..=2).contains(&a.size) && self.same3(a, op::SQRDMLSH, false)
    }

    shift2! { shift2:
        trans_VSHL_2sh => op::SHL;
        trans_VSLI_2sh => op::SLI;
        trans_VSRI_2sh => op::SRI;
        trans_VSRA_S_2sh => op::SSRA;
        trans_VSRA_U_2sh => op::USRA;
        trans_VRSHR_S_2sh => op::SRSHR;
        trans_VRSHR_U_2sh => op::URSHR;
        trans_VRSRA_S_2sh => op::SRSRA;
        trans_VRSRA_U_2sh => op::URSRA;
        trans_VSHR_S_2sh => op::SSHR;
        trans_VSHR_U_2sh => op::USHR;
        trans_VQSHLU_2sh => op::SQSHLUI;
        trans_VQSHL_U_2sh => op::UQSHLI;
        trans_VQSHL_S_2sh => op::SQSHLI;
    }

    shift2! { shift2_narrow:
        trans_VSHRN_64_2sh => op::SHRN;
        trans_VSHRN_32_2sh => op::SHRN;
        trans_VSHRN_16_2sh => op::SHRN;
        trans_VRSHRN_64_2sh => op::RSHRN;
        trans_VRSHRN_32_2sh => op::RSHRN;
        trans_VRSHRN_16_2sh => op::RSHRN;
        trans_VQSHRUN_64_2sh => op::SQSHRUN;
        trans_VQSHRUN_32_2sh => op::SQSHRUN;
        trans_VQSHRUN_16_2sh => op::SQSHRUN;
        trans_VQRSHRUN_64_2sh => op::SQRSHRUN;
        trans_VQRSHRUN_32_2sh => op::SQRSHRUN;
        trans_VQRSHRUN_16_2sh => op::SQRSHRUN;
        trans_VQSHRN_S64_2sh => op::SQSHRN;
        trans_VQSHRN_S32_2sh => op::SQSHRN;
        trans_VQSHRN_S16_2sh => op::SQSHRN;
        trans_VQRSHRN_S64_2sh => op::SQRSHRN;
        trans_VQRSHRN_S32_2sh => op::SQRSHRN;
        trans_VQRSHRN_S16_2sh => op::SQRSHRN;
        trans_VQSHRN_U64_2sh => op::UQSHRN;
        trans_VQSHRN_U32_2sh => op::UQSHRN;
        trans_VQSHRN_U16_2sh => op::UQSHRN;
        trans_VQRSHRN_U64_2sh => op::UQRSHRN;
        trans_VQRSHRN_U32_2sh => op::UQRSHRN;
        trans_VQRSHRN_U16_2sh => op::UQRSHRN;
    }

    shift2! { shift2_long:
        trans_VSHLL_S_2sh => op::SSHLL;
        trans_VSHLL_U_2sh => op::USHLL;
    }

    fn trans_Vimm_1r(&mut self, a: &mut arg_1reg_imm) -> bool {
        // For op=1 the immediate is inverted, so VBIC becomes an AND.
        let logic = a.cmode & 1 != 0 && a.cmode < 12;
        if (!logic && a.cmode == 15 && a.op == 1) || a.vd & a.q != 0 {
            return false;
        }
        let imm = asimd_imm_const(a.imm as u32, a.cmode, a.op) as i64;
        for i in 0..=a.q {
            let off = d_off(a.vd + i);
            if logic {
                let t = self.ld_env64(off);
                if a.op != 0 {
                    self.f().gen_andi_i64(t, t, imm);
                } else {
                    self.f().gen_ori_i64(t, t, imm);
                }
                self.st_env64(t, off);
            } else {
                let t = self.c64(imm);
                self.st_env64(t, off);
            }
        }
        true
    }

    diff3! {
        trans_VADDL_S_3d => op::SADDL, false, false;
        trans_VADDL_U_3d => op::UADDL, false, false;
        trans_VSUBL_S_3d => op::SSUBL, false, false;
        trans_VSUBL_U_3d => op::USUBL, false, false;
        trans_VADDW_S_3d => op::SADDW, true, false;
        trans_VADDW_U_3d => op::UADDW, true, false;
        trans_VSUBW_S_3d => op::SSUBW, true, false;
        trans_VSUBW_U_3d => op::USUBW, true, false;
        trans_VADDHN_3d => op::ADDHN, false, true;
        trans_VRADDHN_3d => op::RADDHN, false, true;
        trans_VSUBHN_3d => op::SUBHN, false, true;
        trans_VRSUBHN_3d => op::RSUBHN, false, true;
        trans_VABAL_S_3d => op::SABAL, false, false;
        trans_VABAL_U_3d => op::UABAL, false, false;
        trans_VABDL_S_3d => op::SABDL, false, false;
        trans_VABDL_U_3d => op::UABDL, false, false;
        trans_VMLAL_S_3d => op::SMLAL, false, false;
        trans_VMLAL_U_3d => op::UMLAL, false, false;
        trans_VMLSL_S_3d => op::SMLSL, false, false;
        trans_VMLSL_U_3d => op::UMLSL, false, false;
        trans_VMULL_S_3d => op::SMULL, false, false;
        trans_VMULL_U_3d => op::UMULL, false, false;
    }

    fn trans_VQDMULL_3d(&mut self, a: &mut arg_3diff) -> bool {
        self.diff3_qd(a, op::SQDMULL)
    }

    fn trans_VQDMLAL_3d(&mut self, a: &mut arg_3diff) -> bool {
        self.diff3_qd(a, op::SQDMLAL)
    }

    fn trans_VQDMLSL_3d(&mut self, a: &mut arg_3diff) -> bool {
        self.diff3_qd(a, op::SQDMLSL)
    }

    fn trans_VMULL_P_3d(&mut self, a: &mut arg_3diff) -> bool {
        // 8 x 8 bits, or 64 x 64 bits (FEAT_PMULL) for size 2.
        if a.vd & 1 != 0 || (a.size != 0 && a.size != 2) {
            return false;
        }
        let esz = if a.size == 2 { 3 } else { 0 };
        self.neon_op(Desc::new(op::PMULL, esz, 16), a.vd, a.vn, a.vm, OUT_Q);
        true
    }

    scalar2! {
        trans_VMUL_2sc => op::MUL, false;
        trans_VMLA_2sc => op::MLA, false;
        trans_VMLS_2sc => op::MLS, false;
        trans_VQDMULH_2sc => op::SQDMULH, false;
        trans_VQRDMULH_2sc => op::SQRDMULH, false;
        trans_VQRDMLAH_2sc => op::SQRDMLAH, false;
        trans_VQRDMLSH_2sc => op::SQRDMLSH, false;
        trans_VMULL_S_2sc => op::SMULL, true;
        trans_VMULL_U_2sc => op::UMULL, true;
        trans_VMLAL_S_2sc => op::SMLAL, true;
        trans_VMLAL_U_2sc => op::UMLAL, true;
        trans_VMLSL_S_2sc => op::SMLSL, true;
        trans_VMLSL_U_2sc => op::UMLSL, true;
        trans_VQDMULL_2sc => op::SQDMULL, true;
        trans_VQDMLAL_2sc => op::SQDMLAL, true;
        trans_VQDMLSL_2sc => op::SQDMLSL, true;
    }

    fn trans_VEXT(&mut self, a: &mut arg_VEXT) -> bool {
        if (a.vn | a.vm | a.vd) & a.q != 0 || (a.imm > 7 && a.q == 0) {
            return false;
        }
        let desc = Desc::new(op::EXT, 0, vec_size(a.q)).imm(a.imm);
        self.neon_op(desc, a.vd, a.vn, a.vm, q_flag(a.q));
        true
    }

    fn trans_VTBL(&mut self, a: &mut arg_VTBL) -> bool {
        // Running off the end of the register file is UNPREDICTABLE; QEMU UNDEFs.
        if a.vn + a.len + 1 > 32 {
            return false;
        }
        let op = if a.op != 0 { op::TBX } else { op::TBL };
        self.neon_op(Desc::new(op, 0, 8).imm(a.len), a.vd, a.vn, a.vm, 0);
        true
    }

    fn trans_VDUP_scalar(&mut self, a: &mut arg_VDUP_scalar) -> bool {
        if a.vd & a.q != 0 {
            return false;
        }
        let t = self.new32();
        self.ld_elem32(t, d_off(a.vm) + ((a.index as usize) << a.size), a.size);
        let x = self.dup64(t, a.size);
        self.st_dreg_q(x, a.vd, a.q != 0);
        true
    }

    misc2! { misc2:
        trans_VNEG => op::NEG, |s, q| s != 3;
        trans_VABS => op::ABS, |s, q| s != 3;
        trans_VCEQ0 => op::CMEQ0, |s, q| s != 3;
        trans_VCGT0 => op::CMGT0, |s, q| s != 3;
        trans_VCLE0 => op::CMLE0, |s, q| s != 3;
        trans_VCGE0 => op::CMGE0, |s, q| s != 3;
        trans_VCLT0 => op::CMLT0, |s, q| s != 3;
        trans_VCLS => op::CLS, |s, q| s != 3;
        trans_VCLZ => op::CLZ, |s, q| s != 3;
        trans_VREV64 => op::REV64, |s, q| s != 3;
        trans_VPADDL_S => op::SADDLP, |s, q| s != 3;
        trans_VPADDL_U => op::UADDLP, |s, q| s != 3;
        trans_VPADAL_S => op::SADALP, |s, q| s != 3;
        trans_VPADAL_U => op::UADALP, |s, q| s != 3;
        trans_VQABS => op::SQABS, |s, q| s != 3;
        trans_VQNEG => op::SQNEG, |s, q| s != 3;
        trans_VMVN => op::NOT, |s, q| s == 0;
        trans_VCNT => op::CNT, |s, q| s == 0;
        trans_VREV16 => op::REV16, |s, q| s == 0;
        trans_VREV32 => op::REV32, |s, q| s <= 1;
    }

    misc2! { misc2_pair:
        trans_VTRN => op::TRN1, |s, q| s != 3;
        // The D forms only have 8 and 16-bit elements.
        trans_VUZP => op::UZP1, |s, q| s <= 1 || (q != 0 && s == 2);
        trans_VZIP => op::ZIP1, |s, q| s <= 1 || (q != 0 && s == 2);
    }

    fn trans_VSWP(&mut self, a: &mut arg_2misc) -> bool {
        if a.size != 0 || (a.vd | a.vm) & a.q != 0 {
            return false;
        }
        for i in 0..=a.q {
            let m = self.ld_env64(d_off(a.vm + i));
            let d = self.ld_env64(d_off(a.vd + i));
            self.st_env64(m, d_off(a.vd + i));
            self.st_env64(d, d_off(a.vm + i));
        }
        true
    }

    fn trans_VMOVN(&mut self, a: &mut arg_2misc) -> bool {
        self.movn(a, op::XTN)
    }

    fn trans_VQMOVUN(&mut self, a: &mut arg_2misc) -> bool {
        self.movn(a, op::SQXTUN)
    }

    fn trans_VQMOVN_S(&mut self, a: &mut arg_2misc) -> bool {
        self.movn(a, op::SQXTN)
    }

    fn trans_VQMOVN_U(&mut self, a: &mut arg_2misc) -> bool {
        self.movn(a, op::UQXTN)
    }

    fn trans_VSHLL(&mut self, a: &mut arg_2misc) -> bool {
        if a.vd & 1 != 0 || a.size == 3 {
            return false;
        }
        self.neon_op(Desc::new(op::SHLL, a.size, 16), a.vd, a.vm, a.vm, OUT_Q);
        true
    }
}

impl DisasNeonLs for S<'_, '_> {
    fn plus_1(&mut self, x: i32) -> i32 {
        x + 1
    }

    fn trans_VLDST_multiple(&mut self, a: &mut arg_VLDST_multiple) -> bool {
        if a.itype > 10 {
            return false;
        }
        // The bad values of the align field.
        match a.itype & 0xc {
            4 if a.align >= 2 => return false,
            8 if a.align == 3 => return false,
            _ => {}
        }
        let (nregs, interleave, spacing) = ELEMENT_TYPE[a.itype as usize];
        let mut size = a.size;
        if size == 3 && (interleave | spacing) != 1 {
            return false;
        }
        // Bytes are always little endian.
        let endian = if size == 0 { MemOp::LE } else { self.d.be_data };
        // With SCTLR.A clear QEMU asks for MO_ALIGN | MO_ALIGN_TLB_ONLY, which the user mode
        // emulator never checks.
        let mut align = if a.align != 0 {
            pow2_align(a.align + 2)
        } else if self.d.align_mem {
            MemOp::ALIGN
        } else {
            MemOp(0)
        };
        // Consecutive little endian elements from one register become 64-bit accesses,
        // keeping any natural alignment.
        if interleave == 1 && endian == MemOp::LE {
            if align == MemOp::ALIGN {
                align = pow2_align(size);
            }
            size = 3;
        }
        let idx = self.d.mmu_idx;
        let t = self.new64();
        let addr = self.load_reg(a.rn);
        let mut mop = endian | MemOp(size as u32) | align;
        for reg in 0..nregs {
            for n in 0..8 >> size {
                for xs in 0..interleave {
                    let off = d_off(a.vd + reg + spacing * xs) + ((n as usize) << size);
                    let va = self.aa32_addr(addr);
                    if a.l != 0 {
                        self.f().gen_qemu_ld_i64(t, va, idx, mop);
                        self.st_elem64(t, off, size);
                    } else {
                        self.ld_elem64(t, off, size);
                        self.f().gen_qemu_st_i64(t, va, idx, mop);
                    }
                    self.f().gen_addi_i32(addr, addr, 1 << size);
                    // Later accesses inherit the alignment.
                    mop = mop.without(MemOp::AMASK);
                }
            }
        }
        self.ldst_base_update(a.rm, a.rn, nregs * interleave * 8);
        true
    }

    fn trans_VLD_all_lanes(&mut self, a: &mut arg_VLD_all_lanes) -> bool {
        let nregs = a.n + 1;
        let mut size = a.size;
        let align = if size == 3 {
            // VLD4 with size 3 and a set is 32 bits at 16 byte alignment.
            if nregs != 4 || a.a == 0 {
                return false;
            }
            size = 2;
            MemOp::ALIGN_16
        } else if a.a != 0 {
            match nregs {
                1 if size == 0 => return false,
                1 => MemOp::ALIGN,
                2 => pow2_align(size + 1),
                3 => return false,
                _ if size == 2 => pow2_align(3),
                _ => pow2_align(size + 2),
            }
        } else {
            MemOp(0)
        };
        // VLD1 to all lanes: T says how many D registers to write. VLD2, 3 and 4: T is the
        // register stride.
        let stride = if a.t != 0 { 2 } else { 1 };
        let q = nregs == 1 && stride == 2;
        let idx = self.d.mmu_idx;
        let mut mop = MemOp(size as u32) | align;
        let t = self.new32();
        let addr = self.load_reg(a.rn);
        let mut vd = a.vd;
        for _ in 0..nregs {
            self.ld32(t, addr, idx, mop);
            let x = self.dup64(t, size);
            self.st_dreg_q(x, vd, q);
            self.f().gen_addi_i32(addr, addr, 1 << size);
            vd += stride;
            // Later accesses inherit the alignment.
            mop = mop.without(MemOp::AMASK);
        }
        self.ldst_base_update(a.rm, a.rn, (1 << size) * nregs);
        true
    }

    fn trans_VLDST_single(&mut self, a: &mut arg_VLDST_single) -> bool {
        let nregs = a.n + 1;
        let size = a.size;
        let bad = match nregs {
            1 => {
                a.stride != 1
                    || a.align & (1 << size) != 0
                    || (size == 2 && (a.align == 1 || a.align == 2))
            }
            2 => size == 2 && a.align & 2 != 0,
            3 => a.align != 0,
            _ => size == 2 && a.align == 3,
        };
        // Writing off the end of the register file is UNPREDICTABLE; QEMU UNDEFs.
        if bad || a.vd + a.stride * (nregs - 1) > 31 {
            return false;
        }
        let mut mop = self.finalize_memop(MemOp(size as u32));
        // With SCTLR.A clear QEMU keeps the MO_ALIGN_TLB_ONLY of finalize_memop(), and the
        // user mode emulator does not check the alignment.
        if a.align != 0 && self.d.align_mem {
            let align = match nregs {
                1 => MemOp::ALIGN,
                2 => pow2_align(size + 1),
                // VLD4.32: align 1 is double alignment, 2 quad alignment.
                _ if size == 2 => pow2_align(size + a.align),
                _ => pow2_align(size + 2),
            };
            mop = mop.without(MemOp::AMASK) | align;
        }
        let idx = self.d.mmu_idx;
        let t = self.new32();
        let addr = self.load_reg(a.rn);
        let mut vd = a.vd;
        for _ in 0..nregs {
            let off = d_off(vd) + ((a.reg_idx as usize) << size);
            let va = self.aa32_addr(addr);
            if a.l != 0 {
                self.f().gen_qemu_ld_i32(t, va, idx, mop);
                self.st_elem32(t, off, size);
            } else {
                self.ld_elem32(t, off, size);
                self.f().gen_qemu_st_i32(t, va, idx, mop);
            }
            vd += a.stride;
            self.f().gen_addi_i32(addr, addr, 1 << size);
            // Later accesses inherit the alignment.
            mop = mop.without(MemOp::AMASK);
        }
        self.ldst_base_update(a.rm, a.rn, (1 << size) * nregs);
        true
    }
}
