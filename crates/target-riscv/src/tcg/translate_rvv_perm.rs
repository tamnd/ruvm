// SPDX-License-Identifier: GPL-2.0-or-later

//! The vector integer reduction, mask and permutation instructions, a port of the
//! "Vector Reduction Operations" (the integer ones), "Vector Mask Operations" and
//! "Vector Permutation Instructions" parts of QEMU's `trans_rvv.c.inc`, with the floating
//! point scalar moves and slides.
//!
//! Deliberate differences from QEMU:
//!
//! - QEMU expands `vrgather.vx` and `vrgather.vi` inline as a splat when `vl` is VLMAX and
//!   the instruction is unmasked, and `vmv<nr>r.v` as a gvec move when `vstart` is 0. Here
//!   they always call their helper, which writes the same registers.

use ruvm_jit_core::ir::TempI64;
use ruvm_jit_core::types::Cond;

use super::helpers::Def;
use super::translate::S;
use super::translate_rvv::{ImmMode, VArgs, is_overlapped, require_align, require_vm};
use super::vector_perm as vp;
use crate::cpu::RISCV_FRM_DYN;

/// The first source operand of a permutation helper.
#[derive(Clone, Copy)]
pub(super) enum PSrc {
    /// None or `vs1`.
    None,
    /// `x[rs1]`.
    X,
    /// The zero extended immediate in the `rs1` field.
    Imm,
}

impl S<'_, '_> {
    /// Call the permutation helper `h` with the `VDATA` of the context and [`Desc::x`]
    /// `x`, then `finalize_rvv_inst()`.
    ///
    /// [`Desc::x`]: super::vector::Desc::x
    fn perm_call(&mut self, h: &Def, a: VArgs, x: u32, src: PSrc) -> bool {
        let mut desc = self.vdesc(a.rd, a.rs1, a.rs2, a.vm);
        desc.x = x;
        match src {
            PSrc::None => self.vcall(h, None, desc, &[]),
            PSrc::X => {
                let s1 = self.gpr(a.rs1);
                self.vcall(h, None, desc, &[s1.into()]);
            }
            PSrc::Imm => {
                let imm = self.extract_imm(a.rs1, ImmMode::Zx);
                let s1 = self.c64(imm);
                self.vcall(h, None, desc, &[s1.into()]);
            }
        }
        self.finalize_rvv_inst();
        true
    }

    /// `do_nanbox()`: `f[rs1]` as a SEW float, the canonical NaN unless it is NaN boxed.
    fn vnanbox(&mut self, rs1: i32) -> TempI64 {
        let f = self.fpr(rs1);
        let (max, nan) = match self.d.sew {
            1 => (0xffff_ffff_ffff_0000u64, 0xffff_ffff_ffff_7e00u64),
            2 => (0xffff_ffff_0000_0000, 0xffff_ffff_7fc0_0000),
            _ => return f,
        };
        let t = self.new64();
        let t_max = self.c64(max as i64);
        let t_nan = self.c64(nan as i64);
        self.f().gen_movcond_i64(Cond::Geu, t, f, t_max, f, t_nan);
        t
    }

    // Reductions.

    /// `reduction_check()`.
    fn reduction_check(&self, a: VArgs) -> bool {
        self.require_rvv()
            && self.vext_check_isa_ill()
            && require_vm(a.vm, a.rs1)
            && require_vm(a.vm, a.rs2)
            && self.vext_check_reduction(a.rs2)
    }

    /// The single width integer reductions, `vred<op>.vs`.
    pub(super) fn vred(&mut self, a: VArgs, op: u32) -> bool {
        self.reduction_check(a) && self.perm_call(&vp::VRED, a, op, PSrc::None)
    }

    /// `vwredsum.vs` and `vwredsumu.vs`: `reduction_widen_check()`.
    pub(super) fn vwred(&mut self, a: VArgs, signed: bool) -> bool {
        let (sew, lmul) = (self.d.sew, self.d.lmul);
        self.reduction_check(a)
            && sew < 3
            && !is_overlapped(a.rs1, 1, a.rs2, 1 << lmul.max(0))
            && sew < (self.d.cfg.elen >> 4) as i32
            && self.perm_call(&vp::VWRED, a, u32::from(!signed), PSrc::None)
    }

    // Mask operations.

    /// `GEN_MM_TRANS()`.
    pub(super) fn vmask_mm(&mut self, rd: i32, rs1: i32, rs2: i32, op: u32) -> bool {
        if !self.require_rvv() || !self.vext_check_isa_ill() {
            return false;
        }
        let a = VArgs { rd, rs1, rs2, vm: 0 };
        self.perm_call(&vp::VMASK_MM, a, op, PSrc::None)
    }

    /// `trans_vcpop_m()` and `trans_vfirst_m()`: `x[rd]` from a helper. They do not end
    /// with `finalize_rvv_inst()`, as in QEMU.
    pub(super) fn vmask_x(&mut self, rd: i32, rs2: i32, vm: i32, h: &Def) -> bool {
        if !self.require_rvv() || !self.vext_check_isa_ill() || !self.d.vstart_eq_zero {
            return false;
        }
        let desc = self.vdesc(0, 0, rs2, vm);
        let dst = self.new64();
        self.vcall(h, Some(dst.into()), desc, &[]);
        self.set_gpr(rd, dst);
        true
    }

    /// `GEN_M_TRANS()`: `vmsbf.m`, `vmsif.m` and `vmsof.m`.
    pub(super) fn vmsetm(&mut self, rd: i32, rs2: i32, vm: i32, ty: u32) -> bool {
        if !self.require_rvv()
            || !self.vext_check_isa_ill()
            || !require_vm(vm, rd)
            || rd == rs2
            || !self.d.vstart_eq_zero
        {
            return false;
        }
        let a = VArgs { rd, rs1: 0, rs2, vm };
        self.perm_call(&vp::VMSETM, a, ty, PSrc::None)
    }

    /// `trans_viota_m()`.
    pub(super) fn viota(&mut self, rd: i32, rs2: i32, vm: i32) -> bool {
        let lmul = self.d.lmul;
        if !self.require_rvv()
            || !self.vext_check_isa_ill()
            || is_overlapped(rd, 1 << lmul.max(0), rs2, 1)
            || !require_vm(vm, rd)
            || !require_align(rd, lmul)
            || !self.d.vstart_eq_zero
        {
            return false;
        }
        let a = VArgs { rd, rs1: 0, rs2, vm };
        self.perm_call(&vp::VIOTA_M, a, 0, PSrc::None)
    }

    /// `trans_vid_v()`.
    pub(super) fn vid(&mut self, rd: i32, vm: i32) -> bool {
        if !self.require_rvv()
            || !self.vext_check_isa_ill()
            || !require_align(rd, self.d.lmul)
            || !require_vm(vm, rd)
        {
            return false;
        }
        let a = VArgs { rd, rs1: 0, rs2: 0, vm };
        self.perm_call(&vp::VID_V, a, 0, PSrc::None)
    }

    // Scalar moves.

    /// `trans_vmv_x_s()`.
    pub(super) fn vmv_x_s(&mut self, rd: i32, rs2: i32) -> bool {
        if !self.require_rvv() || !self.vext_check_isa_ill() {
            return false;
        }
        let desc = self.vdesc(0, 0, rs2, 1);
        let dst = self.new64();
        self.vcall(&vp::VMV_X_S, Some(dst.into()), desc, &[]);
        self.set_gpr(rd, dst);
        self.finalize_rvv_inst();
        true
    }

    /// `trans_vmv_s_x()`.
    pub(super) fn vmv_s_x(&mut self, rd: i32, rs1: i32) -> bool {
        if !self.require_rvv() || !self.vext_check_isa_ill() {
            return false;
        }
        let s1 = self.gpr(rs1);
        let desc = self.vdesc(rd, 0, 0, 1);
        self.vcall(&vp::VMV_S_X, None, desc, &[s1.into()]);
        self.finalize_rvv_inst();
        true
    }

    /// `trans_vfmv_f_s()`.
    pub(super) fn vfmv_f_s(&mut self, rd: i32, rs2: i32) -> bool {
        if !self.require_rvv() || !self.require_rvf() || !self.vext_check_isa_ill() {
            return false;
        }
        self.gen_set_rm(RISCV_FRM_DYN as i32);
        let mut desc = self.vdesc(0, 0, rs2, 1);
        desc.x = 1;
        let t = self.new64();
        self.vcall(&vp::VMV_X_S, Some(t.into()), desc, &[]);
        let f = self.fpr(rd);
        self.f().gen_mov_i64(f, t);
        self.mark_fs_dirty();
        self.finalize_rvv_inst();
        true
    }

    /// `trans_vfmv_s_f()`.
    pub(super) fn vfmv_s_f(&mut self, rd: i32, rs1: i32) -> bool {
        if !self.require_rvv() || !self.require_rvf() || !self.vext_check_isa_ill() {
            return false;
        }
        self.gen_set_rm(RISCV_FRM_DYN as i32);
        let t = self.vnanbox(rs1);
        let desc = self.vdesc(rd, 0, 0, 1);
        self.vcall(&vp::VMV_S_X, None, desc, &[t.into()]);
        self.finalize_rvv_inst();
        true
    }

    // Slides.

    /// `slideup_check()`.
    fn slideup_check(&self, a: VArgs) -> bool {
        self.require_rvv()
            && self.vext_check_isa_ill()
            && self.vext_check_slide(a.rd, a.rs2, a.vm, true)
    }

    /// `slidedown_check()`.
    fn slidedown_check(&self, a: VArgs) -> bool {
        self.require_rvv()
            && self.vext_check_isa_ill()
            && self.vext_check_slide(a.rd, a.rs2, a.vm, false)
    }

    /// `vslideup.vx`, `vslideup.vi`, `vslide1up.vx` and the down forms: `h` with `src`.
    pub(super) fn vslide(&mut self, a: VArgs, up: bool, h: &Def, src: PSrc) -> bool {
        let ok = if up { self.slideup_check(a) } else { self.slidedown_check(a) };
        ok && self.perm_call(h, a, 0, src)
    }

    /// `vfslide1up.vf` and `vfslide1down.vf`: `fslideup_check()` or `fslidedown_check()`,
    /// then the slide of `f[rs1]` NaN boxed.
    pub(super) fn vfslide1(&mut self, a: VArgs, up: bool) -> bool {
        let ok = if up { self.slideup_check(a) } else { self.slidedown_check(a) };
        if !ok || !self.require_rvf() {
            return false;
        }
        self.gen_set_rm(RISCV_FRM_DYN as i32);
        let t = self.vnanbox(a.rs1);
        let desc = self.vdesc(a.rd, a.rs1, a.rs2, a.vm);
        let h = if up { &vp::VSLIDE1UP } else { &vp::VSLIDE1DOWN };
        self.vcall(h, None, desc, &[t.into()]);
        self.finalize_rvv_inst();
        true
    }

    // Gather, compress and whole register moves.

    /// `vrgather_vv_check()`, then `vrgather.vv`.
    pub(super) fn vrgather_vv(&mut self, a: VArgs) -> bool {
        let (sew, lmul) = (self.d.sew, self.d.lmul);
        self.require_rvv()
            && self.vext_check_isa_ill()
            && self.vext_check_input_eew(a.rs1, sew, a.rs2, sew, a.vm)
            && require_align(a.rd, lmul)
            && require_align(a.rs1, lmul)
            && require_align(a.rs2, lmul)
            && a.rd != a.rs2
            && a.rd != a.rs1
            && require_vm(a.vm, a.rd)
            && self.perm_call(&vp::VRGATHER_VV, a, 0, PSrc::None)
    }

    /// `vrgatherei16_vv_check()`, then `vrgatherei16.vv`.
    pub(super) fn vrgatherei16_vv(&mut self, a: VArgs) -> bool {
        let (sew, lmul) = (self.d.sew, self.d.lmul);
        let emul = 1 - sew + lmul;
        self.require_rvv()
            && self.vext_check_isa_ill()
            && self.vext_check_input_eew(a.rs1, 1, a.rs2, sew, a.vm)
            && (-3..=3).contains(&emul)
            && require_align(a.rd, lmul)
            && require_align(a.rs1, emul)
            && require_align(a.rs2, lmul)
            && a.rd != a.rs2
            && a.rd != a.rs1
            && !is_overlapped(a.rd, 1 << lmul.max(0), a.rs1, 1 << emul.max(0))
            && !is_overlapped(a.rd, 1 << lmul.max(0), a.rs2, 1 << lmul.max(0))
            && require_vm(a.vm, a.rd)
            && self.perm_call(&vp::VRGATHER_VV, a, 1, PSrc::None)
    }

    /// `vrgather_vx_check()`, then `vrgather.vx` or `vrgather.vi`.
    pub(super) fn vrgather_x(&mut self, a: VArgs, src: PSrc) -> bool {
        let lmul = self.d.lmul;
        self.require_rvv()
            && self.vext_check_isa_ill()
            && self.vext_check_input_eew(-1, 3, a.rs2, self.d.sew, a.vm)
            && require_align(a.rd, lmul)
            && require_align(a.rs2, lmul)
            && a.rd != a.rs2
            && require_vm(a.vm, a.rd)
            && self.perm_call(&vp::VRGATHER_VX, a, 0, src)
    }

    /// `trans_vcompress_vm()`.
    pub(super) fn vcompress(&mut self, rd: i32, rs1: i32, rs2: i32) -> bool {
        let lmul = self.d.lmul;
        self.require_rvv()
            && self.vext_check_isa_ill()
            && require_align(rd, lmul)
            && require_align(rs2, lmul)
            && rd != rs2
            && !is_overlapped(rd, 1 << lmul.max(0), rs1, 1)
            && self.d.vstart_eq_zero
            && self.perm_call(&vp::VCOMPRESS_VM, VArgs { rd, rs1, rs2, vm: 1 }, 0, PSrc::None)
    }

    /// `GEN_VMV_WHOLE_TRANS()`: `vmv<len>r.v`.
    pub(super) fn vmvr(&mut self, rd: i32, rs2: i32, len: i32) -> bool {
        if !self.require_rvv() || !self.vext_check_isa_ill() || rd % len != 0 || rs2 % len != 0 {
            return false;
        }
        let mut desc = self.vdesc(rd, 0, rs2, 1);
        desc.nf = len as u32;
        self.vcall(&vp::VMVR_V, None, desc, &[]);
        self.finalize_rvv_inst();
        true
    }
}

/// The `trans_*` methods of this file for `DecodeInsn32`.
macro_rules! rvv_perm_trans32 {
    () => {
        fn trans_vredsum_vs(&mut self, a: &mut arg_vredsum_vs) -> bool {
            self.vred($crate::tcg::translate_rvv::vargs(a), $crate::tcg::vector_perm::RED_SUM)
        }
        fn trans_vredmaxu_vs(&mut self, a: &mut arg_vredmaxu_vs) -> bool {
            self.vred($crate::tcg::translate_rvv::vargs(a), $crate::tcg::vector_perm::RED_MAXU)
        }
        fn trans_vredmax_vs(&mut self, a: &mut arg_vredmax_vs) -> bool {
            self.vred($crate::tcg::translate_rvv::vargs(a), $crate::tcg::vector_perm::RED_MAX)
        }
        fn trans_vredminu_vs(&mut self, a: &mut arg_vredminu_vs) -> bool {
            self.vred($crate::tcg::translate_rvv::vargs(a), $crate::tcg::vector_perm::RED_MINU)
        }
        fn trans_vredmin_vs(&mut self, a: &mut arg_vredmin_vs) -> bool {
            self.vred($crate::tcg::translate_rvv::vargs(a), $crate::tcg::vector_perm::RED_MIN)
        }
        fn trans_vredand_vs(&mut self, a: &mut arg_vredand_vs) -> bool {
            self.vred($crate::tcg::translate_rvv::vargs(a), $crate::tcg::vector_perm::RED_AND)
        }
        fn trans_vredor_vs(&mut self, a: &mut arg_vredor_vs) -> bool {
            self.vred($crate::tcg::translate_rvv::vargs(a), $crate::tcg::vector_perm::RED_OR)
        }
        fn trans_vredxor_vs(&mut self, a: &mut arg_vredxor_vs) -> bool {
            self.vred($crate::tcg::translate_rvv::vargs(a), $crate::tcg::vector_perm::RED_XOR)
        }
        fn trans_vwredsum_vs(&mut self, a: &mut arg_vwredsum_vs) -> bool {
            self.vwred($crate::tcg::translate_rvv::vargs(a), true)
        }
        fn trans_vwredsumu_vs(&mut self, a: &mut arg_vwredsumu_vs) -> bool {
            self.vwred($crate::tcg::translate_rvv::vargs(a), false)
        }

        fn trans_vmand_mm(&mut self, a: &mut arg_vmand_mm) -> bool {
            self.vmask_mm(a.rd, a.rs1, a.rs2, $crate::tcg::vector_perm::MM_AND)
        }
        fn trans_vmnand_mm(&mut self, a: &mut arg_vmnand_mm) -> bool {
            self.vmask_mm(a.rd, a.rs1, a.rs2, $crate::tcg::vector_perm::MM_NAND)
        }
        fn trans_vmandn_mm(&mut self, a: &mut arg_vmandn_mm) -> bool {
            self.vmask_mm(a.rd, a.rs1, a.rs2, $crate::tcg::vector_perm::MM_ANDN)
        }
        fn trans_vmxor_mm(&mut self, a: &mut arg_vmxor_mm) -> bool {
            self.vmask_mm(a.rd, a.rs1, a.rs2, $crate::tcg::vector_perm::MM_XOR)
        }
        fn trans_vmor_mm(&mut self, a: &mut arg_vmor_mm) -> bool {
            self.vmask_mm(a.rd, a.rs1, a.rs2, $crate::tcg::vector_perm::MM_OR)
        }
        fn trans_vmnor_mm(&mut self, a: &mut arg_vmnor_mm) -> bool {
            self.vmask_mm(a.rd, a.rs1, a.rs2, $crate::tcg::vector_perm::MM_NOR)
        }
        fn trans_vmorn_mm(&mut self, a: &mut arg_vmorn_mm) -> bool {
            self.vmask_mm(a.rd, a.rs1, a.rs2, $crate::tcg::vector_perm::MM_ORN)
        }
        fn trans_vmxnor_mm(&mut self, a: &mut arg_vmxnor_mm) -> bool {
            self.vmask_mm(a.rd, a.rs1, a.rs2, $crate::tcg::vector_perm::MM_XNOR)
        }
        fn trans_vcpop_m(&mut self, a: &mut arg_vcpop_m) -> bool {
            self.vmask_x(a.rd, a.rs2, a.vm, &$crate::tcg::vector_perm::VCPOP_M)
        }
        fn trans_vfirst_m(&mut self, a: &mut arg_vfirst_m) -> bool {
            self.vmask_x(a.rd, a.rs2, a.vm, &$crate::tcg::vector_perm::VFIRST_M)
        }
        fn trans_vmsbf_m(&mut self, a: &mut arg_vmsbf_m) -> bool {
            self.vmsetm(a.rd, a.rs2, a.vm, $crate::tcg::vector_perm::BEFORE_FIRST)
        }
        fn trans_vmsif_m(&mut self, a: &mut arg_vmsif_m) -> bool {
            self.vmsetm(a.rd, a.rs2, a.vm, $crate::tcg::vector_perm::INCLUDE_FIRST)
        }
        fn trans_vmsof_m(&mut self, a: &mut arg_vmsof_m) -> bool {
            self.vmsetm(a.rd, a.rs2, a.vm, $crate::tcg::vector_perm::ONLY_FIRST)
        }
        fn trans_viota_m(&mut self, a: &mut arg_viota_m) -> bool {
            self.viota(a.rd, a.rs2, a.vm)
        }
        fn trans_vid_v(&mut self, a: &mut arg_vid_v) -> bool {
            self.vid(a.rd, a.vm)
        }

        fn trans_vmv_x_s(&mut self, a: &mut arg_vmv_x_s) -> bool {
            self.vmv_x_s(a.rd, a.rs2)
        }
        fn trans_vmv_s_x(&mut self, a: &mut arg_vmv_s_x) -> bool {
            self.vmv_s_x(a.rd, a.rs1)
        }
        fn trans_vfmv_f_s(&mut self, a: &mut arg_vfmv_f_s) -> bool {
            self.vfmv_f_s(a.rd, a.rs2)
        }
        fn trans_vfmv_s_f(&mut self, a: &mut arg_vfmv_s_f) -> bool {
            self.vfmv_s_f(a.rd, a.rs1)
        }

        fn trans_vslideup_vx(&mut self, a: &mut arg_vslideup_vx) -> bool {
            let a = $crate::tcg::translate_rvv::vargs(a);
            self.vslide(
                a,
                true,
                &$crate::tcg::vector_perm::VSLIDEUP,
                $crate::tcg::translate_rvv_perm::PSrc::X,
            )
        }
        fn trans_vslideup_vi(&mut self, a: &mut arg_vslideup_vi) -> bool {
            let a = $crate::tcg::translate_rvv::vargs(a);
            self.vslide(
                a,
                true,
                &$crate::tcg::vector_perm::VSLIDEUP,
                $crate::tcg::translate_rvv_perm::PSrc::Imm,
            )
        }
        fn trans_vslidedown_vx(&mut self, a: &mut arg_vslidedown_vx) -> bool {
            let a = $crate::tcg::translate_rvv::vargs(a);
            self.vslide(
                a,
                false,
                &$crate::tcg::vector_perm::VSLIDEDOWN,
                $crate::tcg::translate_rvv_perm::PSrc::X,
            )
        }
        fn trans_vslidedown_vi(&mut self, a: &mut arg_vslidedown_vi) -> bool {
            let a = $crate::tcg::translate_rvv::vargs(a);
            self.vslide(
                a,
                false,
                &$crate::tcg::vector_perm::VSLIDEDOWN,
                $crate::tcg::translate_rvv_perm::PSrc::Imm,
            )
        }
        fn trans_vslide1up_vx(&mut self, a: &mut arg_vslide1up_vx) -> bool {
            let a = $crate::tcg::translate_rvv::vargs(a);
            self.vslide(
                a,
                true,
                &$crate::tcg::vector_perm::VSLIDE1UP,
                $crate::tcg::translate_rvv_perm::PSrc::X,
            )
        }
        fn trans_vslide1down_vx(&mut self, a: &mut arg_vslide1down_vx) -> bool {
            let a = $crate::tcg::translate_rvv::vargs(a);
            self.vslide(
                a,
                false,
                &$crate::tcg::vector_perm::VSLIDE1DOWN,
                $crate::tcg::translate_rvv_perm::PSrc::X,
            )
        }
        fn trans_vfslide1up_vf(&mut self, a: &mut arg_vfslide1up_vf) -> bool {
            self.vfslide1($crate::tcg::translate_rvv::vargs(a), true)
        }
        fn trans_vfslide1down_vf(&mut self, a: &mut arg_vfslide1down_vf) -> bool {
            self.vfslide1($crate::tcg::translate_rvv::vargs(a), false)
        }

        fn trans_vrgather_vv(&mut self, a: &mut arg_vrgather_vv) -> bool {
            self.vrgather_vv($crate::tcg::translate_rvv::vargs(a))
        }
        fn trans_vrgatherei16_vv(&mut self, a: &mut arg_vrgatherei16_vv) -> bool {
            self.vrgatherei16_vv($crate::tcg::translate_rvv::vargs(a))
        }
        fn trans_vrgather_vx(&mut self, a: &mut arg_vrgather_vx) -> bool {
            self.vrgather_x(
                $crate::tcg::translate_rvv::vargs(a),
                $crate::tcg::translate_rvv_perm::PSrc::X,
            )
        }
        fn trans_vrgather_vi(&mut self, a: &mut arg_vrgather_vi) -> bool {
            self.vrgather_x(
                $crate::tcg::translate_rvv::vargs(a),
                $crate::tcg::translate_rvv_perm::PSrc::Imm,
            )
        }
        fn trans_vcompress_vm(&mut self, a: &mut arg_vcompress_vm) -> bool {
            self.vcompress(a.rd, a.rs1, a.rs2)
        }
        fn trans_vmv1r_v(&mut self, a: &mut arg_vmv1r_v) -> bool {
            self.vmvr(a.rd, a.rs2, 1)
        }
        fn trans_vmv2r_v(&mut self, a: &mut arg_vmv2r_v) -> bool {
            self.vmvr(a.rd, a.rs2, 2)
        }
        fn trans_vmv4r_v(&mut self, a: &mut arg_vmv4r_v) -> bool {
            self.vmvr(a.rd, a.rs2, 4)
        }
        fn trans_vmv8r_v(&mut self, a: &mut arg_vmv8r_v) -> bool {
            self.vmvr(a.rd, a.rs2, 8)
        }
    };
}
pub(super) use rvv_perm_trans32;
