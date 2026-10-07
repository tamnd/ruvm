// SPDX-License-Identifier: GPL-2.0-or-later

//! The vector floating point instructions, a port of the floating point parts of QEMU's
//! `trans_rvv.c.inc` and of the vector parts of `trans_rvbf16.c.inc` (Zvfbfmin and
//! Zvfbfwma). The scalar moves and slides are with the permutations.
//!
//! A `.vf` instruction calls the helper of its `.vv` form with the NaN boxed scalar
//! (`do_nanbox()`) and [`Desc::scalar`] set; the helpers are in `vector_fp`.
//!
//! Deliberate differences from QEMU:
//!
//! - Zvfbfa is off, so `vtype.altfmt` is never set and `vext_check_altfmt()` always passes.
//! - QEMU expands `vfmv.v.f` inline as a splat when `vl` is VLMAX and the tail is not
//!   agnostic with a fractional LMUL. Here it always calls [`VFMERGE`] with `vm` set, which
//!   writes the same registers.
//!
//! [`Desc::scalar`]: super::vector::Desc::scalar
//! [`VFMERGE`]: super::vector_fp::VFMERGE

use ruvm_jit_core::ir::TempI64;
use ruvm_jit_core::types::Cond;

use super::helpers::Def;
use super::translate::S;
use super::translate_rvv::{VArgs, VSrc, is_overlapped, require_align, require_vm};
use super::vector_fp as vfp;
use crate::cpu::{RISCV_FRM_DYN, RISCV_FRM_ROD};
use crate::decode::insn32::arg_rmr;

/// The checks of an instruction with two sources, a `.vv` or `.vf` form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FCheck {
    /// `opfvv_check()` and `opfvf_check()`.
    Single,
    /// `opfvv_cmp_check()` and `opfvf_cmp_check()`.
    Cmp,
    /// `opfvv_widen_check()` and `opfvf_widen_check()`.
    Widen,
    /// `opfvv_overwrite_widen_check()` and `opfvf_overwrite_widen_check()`: the widening
    /// multiply-adds.
    WidenAcc,
    /// `opfwv_widen_check()` and `opfwf_widen_check()`.
    WidenW,
    /// `freduction_check()`.
    Red,
    /// `freduction_widen_check()`.
    RedWiden,
}

/// The checks of an instruction with one vector source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FvCheck {
    /// `opfv_check()`: square root, estimates, classify and single width conversions.
    Opfv,
    /// `opxfv_widen_check()`: `vfwcvt.xu.f.v` and `vfwcvt.x.f.v`.
    WidenXF,
    /// `opffv_widen_check()`: `vfwcvt.f.f.v`.
    WidenFF,
    /// `opfxv_widen_check()`: `vfwcvt.f.xu.v` and `vfwcvt.f.x.v`.
    WidenFX,
    /// `opfxv_narrow_check()`: `vfncvt.f.xu.w` and `vfncvt.f.x.w`.
    NarrowFX,
    /// `opffv_narrow_check()`: `vfncvt.f.f.w`.
    NarrowFF,
    /// `opffv_rod_narrow_check()`: `vfncvt.rod.f.f.w`.
    NarrowRod,
    /// `opxfv_narrow_check()`: `vfncvt.xu.f.w` and `vfncvt.x.f.w`.
    NarrowXF,
    /// `vfncvtbf16.f.f.w`.
    Bf16Narrow,
    /// `vfwcvtbf16.f.f.v`.
    Bf16Widen,
}

/// The [`VArgs`] of an `arg_rmr`, which has no `vs1`.
pub(super) fn rargs(a: &arg_rmr) -> VArgs {
    VArgs { rd: a.rd, rs1: 0, rs2: a.rs2, vm: a.vm }
}

impl S<'_, '_> {
    /// `do_nanbox()`: `f[rs1]` as a SEW float, the canonical NaN unless it is NaN boxed.
    fn vf_nanbox(&mut self, rs1: i32) -> TempI64 {
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

    /// `reduction_check()`.
    fn freduction_base(&self, a: VArgs) -> bool {
        self.require_rvv()
            && self.vext_check_isa_ill()
            && require_vm(a.vm, a.rs1)
            && require_vm(a.vm, a.rs2)
            && self.vext_check_reduction(a.rs2)
    }

    /// The checks `k` of a `.vv` form, or of a `.vf` form when `vf`.
    fn fcheck(&self, a: VArgs, k: FCheck, vf: bool) -> bool {
        let (sew, lmul) = (self.d.sew, self.d.lmul);
        let (rd, rs1, rs2, vm) = (a.rd, a.rs1, a.rs2, a.vm);
        let base = self.require_rvv() && self.require_rvf();
        match k {
            FCheck::Single => {
                base && self.vext_check_isa_ill()
                    && if vf {
                        self.vext_check_ss(rd, rs2, vm)
                    } else {
                        self.vext_check_sss(rd, rs1, rs2, vm)
                    }
            }
            FCheck::Cmp => {
                base && self.vext_check_isa_ill()
                    && if vf {
                        self.vext_check_ms(rd, rs2)
                    } else {
                        self.vext_check_mss(rd, rs1, rs2)
                    }
            }
            FCheck::Widen | FCheck::WidenAcc => {
                let ok = base
                    && self.require_scale_rvf()
                    && self.vext_check_isa_ill()
                    && if vf {
                        self.vext_check_ds(rd, rs2, vm)
                    } else {
                        self.vext_check_dss(rd, rs1, rs2, vm)
                    };
                if k == FCheck::Widen {
                    return ok;
                }
                ok && (vf || self.vext_check_input_eew(rd, sew + 1, rs1, sew, vm))
                    && self.vext_check_input_eew(rd, sew + 1, rs2, sew, vm)
            }
            FCheck::WidenW => {
                base && self.require_scale_rvf()
                    && self.vext_check_isa_ill()
                    && if vf {
                        self.vext_check_dd(rd, rs2, vm)
                    } else {
                        self.vext_check_dds(rd, rs1, rs2, vm)
                    }
            }
            FCheck::Red => self.freduction_base(a) && self.require_rvf(),
            FCheck::RedWiden => {
                self.freduction_base(a)
                    && sew < 3
                    && !is_overlapped(rs1, 1, rs2, 1 << lmul.max(0))
                    && sew < (self.d.cfg.elen >> 4) as i32
                    && self.require_rvf()
                    && self.require_scale_rvf()
            }
        }
    }

    /// `GEN_OPFVV_TRANS()` and its kind: helper `h` on `vd`, `vs1` and `vs2` with the
    /// checks `k`.
    pub(super) fn opfvv(&mut self, h: &Def, a: VArgs, k: FCheck) -> bool {
        if !self.fcheck(a, k, false) {
            return false;
        }
        self.gen_set_rm(RISCV_FRM_DYN as i32);
        self.gen_vop(h, a, VSrc::V)
    }

    /// `GEN_OPFVF_TRANS()` and its kind: helper `h` on `vd`, `vs2` and the NaN boxed
    /// `f[rs1]` with the checks `k`.
    pub(super) fn opfvf(&mut self, h: &Def, a: VArgs, k: FCheck) -> bool {
        if !self.fcheck(a, k, true) {
            return false;
        }
        self.gen_set_rm(RISCV_FRM_DYN as i32);
        let t = self.vf_nanbox(a.rs1);
        self.gen_vop(h, a, VSrc::T(t))
    }

    /// The checks `k` of an instruction with one vector source.
    fn fvcheck(&self, a: VArgs, k: FvCheck) -> bool {
        let (rd, rs2, vm) = (a.rd, a.rs2, a.vm);
        let sew = self.d.sew;
        let rvv = self.require_rvv() && self.vext_check_isa_ill();
        match k {
            FvCheck::Opfv => rvv && self.require_rvf() && self.vext_check_ss(rd, rs2, vm),
            FvCheck::WidenXF => rvv && self.vext_check_ds(rd, rs2, vm) && self.require_rvf(),
            FvCheck::WidenFF => {
                rvv && self.vext_check_ds(rd, rs2, vm)
                    && self.require_rvfmin()
                    && self.require_scale_rvfmin()
            }
            FvCheck::WidenFX => rvv && self.require_scale_rvf() && self.vext_check_ds(rd, rs2, vm),
            FvCheck::NarrowFX => {
                rvv && self.vext_check_sd(rd, rs2, vm) && self.require_rvf() && sew != 3
            }
            FvCheck::NarrowFF => {
                rvv && self.vext_check_sd(rd, rs2, vm)
                    && self.require_rvfmin()
                    && self.require_scale_rvfmin()
            }
            FvCheck::NarrowRod => {
                rvv && self.vext_check_sd(rd, rs2, vm)
                    && self.require_rvf()
                    && self.require_scale_rvf()
            }
            FvCheck::NarrowXF => rvv && self.require_scale_rvf() && self.vext_check_sd(rd, rs2, vm),
            FvCheck::Bf16Narrow => {
                !self.fpu_off()
                    && self.d.cfg.ext_zvfbfmin
                    && rvv
                    && self.vext_check_sd(rd, rs2, vm)
                    && sew == 1
            }
            FvCheck::Bf16Widen => {
                !self.fpu_off()
                    && self.d.cfg.ext_zvfbfmin
                    && rvv
                    && self.vext_check_ds(rd, rs2, vm)
                    && sew == 1
            }
        }
    }

    /// `do_opfv()` and the conversions: helper `h` on `vd` and `vs2` with the checks `k`
    /// and the rounding mode `rm`, checked against `frm` (`gen_set_rm_chkfrm()`) except
    /// for the conversions from integers to wider floats, which QEMU sets with
    /// `gen_set_rm()`.
    pub(super) fn opfv(&mut self, h: &Def, a: VArgs, k: FvCheck, rm: u64) -> bool {
        if !self.fvcheck(a, k) {
            return false;
        }
        if k == FvCheck::WidenFX {
            self.gen_set_rm(rm as i32);
        } else {
            self.gen_set_rm_chkfrm(rm as i32);
        }
        self.gen_vop(h, a, VSrc::V)
    }

    /// `vfncvt.rod.f.f.w`.
    pub(super) fn opfv_rod(&mut self, a: VArgs) -> bool {
        self.opfv(&vfp::VFNCVT_F_F, a, FvCheck::NarrowRod, RISCV_FRM_ROD)
    }

    /// `vfmv.v.f`: every element from `vstart` to `vl` is the NaN boxed `f[rs1]`.
    pub(super) fn vfmv_v_f(&mut self, rd: i32, rs1: i32) -> bool {
        if !(self.require_rvv()
            && self.require_rvf()
            && self.vext_check_isa_ill()
            && require_align(rd, self.d.lmul))
        {
            return false;
        }
        self.gen_set_rm(RISCV_FRM_DYN as i32);
        let t = self.vf_nanbox(rs1);
        let a = VArgs { rd, rs1, rs2: 0, vm: 1 };
        self.gen_vop(&vfp::VFMERGE, a, VSrc::T(t))
    }

    /// `vfwmaccbf16.vv`, or `vfwmaccbf16.vf` when `vf`.
    pub(super) fn vfwmaccbf16(&mut self, a: VArgs, vf: bool) -> bool {
        if self.fpu_off() || !self.d.cfg.ext_zvfbfwma {
            return false;
        }
        let sew = self.d.sew;
        let (rd, rs1, rs2, vm) = (a.rd, a.rs1, a.rs2, a.vm);
        let ok = self.require_rvv()
            && self.vext_check_isa_ill()
            && sew == 1
            && if vf {
                self.vext_check_ds(rd, rs2, vm)
            } else {
                self.vext_check_dss(rd, rs1, rs2, vm)
                    && self.vext_check_input_eew(rd, sew + 1, rs1, sew, vm)
            }
            && self.vext_check_input_eew(rd, sew + 1, rs2, sew, vm);
        if !ok {
            return false;
        }
        if vf {
            self.gen_set_rm(RISCV_FRM_DYN as i32);
            let t = self.vf_nanbox(rs1);
            self.gen_vop(&vfp::VFWMACCBF16, a, VSrc::T(t))
        } else {
            self.gen_set_rm_chkfrm(RISCV_FRM_DYN as i32);
            self.gen_vop(&vfp::VFWMACCBF16, a, VSrc::V)
        }
    }
}

/// A `.vv` `trans_*` method: `fvv!(trans_x, HELPER, Check)`.
macro_rules! fvv {
    ($t:ident, $h:ident, $k:ident) => {
        fn $t(&mut self, a: &mut $crate::decode::insn32::arg_rmrr) -> bool {
            self.opfvv(
                &$crate::tcg::vector_fp::$h,
                $crate::tcg::translate_rvv::vargs(a),
                $crate::tcg::translate_rvv_fp::FCheck::$k,
            )
        }
    };
}
pub(super) use fvv;

/// A `.vf` `trans_*` method: `fvf!(trans_x, HELPER, Check)`.
macro_rules! fvf {
    ($t:ident, $h:ident, $k:ident) => {
        fn $t(&mut self, a: &mut $crate::decode::insn32::arg_rmrr) -> bool {
            self.opfvf(
                &$crate::tcg::vector_fp::$h,
                $crate::tcg::translate_rvv::vargs(a),
                $crate::tcg::translate_rvv_fp::FCheck::$k,
            )
        }
    };
}
pub(super) use fvf;

/// A `trans_*` method with one vector source: `fv!(trans_x, HELPER, Check, RISCV_FRM_x)`.
macro_rules! fv {
    ($t:ident, $h:ident, $k:ident, $rm:ident) => {
        fn $t(&mut self, a: &mut $crate::decode::insn32::arg_rmr) -> bool {
            self.opfv(
                &$crate::tcg::vector_fp::$h,
                $crate::tcg::translate_rvv_fp::rargs(a),
                $crate::tcg::translate_rvv_fp::FvCheck::$k,
                $crate::cpu::$rm,
            )
        }
    };
}
pub(super) use fv;

/// The vector floating point `trans_*` methods of `DecodeInsn32`.
macro_rules! rvv_fp_trans32 {
    () => {
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfadd_vv, VFADD, Single);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfadd_vf, VFADD, Single);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfsub_vv, VFSUB, Single);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfsub_vf, VFSUB, Single);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfrsub_vf, VFRSUB, Single);

        $crate::tcg::translate_rvv_fp::fvv!(trans_vfwadd_vv, VFWADD, Widen);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfwadd_vf, VFWADD, Widen);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfwsub_vv, VFWSUB, Widen);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfwsub_vf, VFWSUB, Widen);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfwadd_wv, VFWADD_W, WidenW);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfwadd_wf, VFWADD_W, WidenW);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfwsub_wv, VFWSUB_W, WidenW);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfwsub_wf, VFWSUB_W, WidenW);

        $crate::tcg::translate_rvv_fp::fvv!(trans_vfmul_vv, VFMUL, Single);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfmul_vf, VFMUL, Single);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfdiv_vv, VFDIV, Single);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfdiv_vf, VFDIV, Single);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfrdiv_vf, VFRDIV, Single);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfwmul_vv, VFWMUL, Widen);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfwmul_vf, VFWMUL, Widen);

        $crate::tcg::translate_rvv_fp::fvv!(trans_vfmacc_vv, VFMACC, Single);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfmacc_vf, VFMACC, Single);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfnmacc_vv, VFNMACC, Single);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfnmacc_vf, VFNMACC, Single);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfmsac_vv, VFMSAC, Single);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfmsac_vf, VFMSAC, Single);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfnmsac_vv, VFNMSAC, Single);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfnmsac_vf, VFNMSAC, Single);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfmadd_vv, VFMADD, Single);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfmadd_vf, VFMADD, Single);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfnmadd_vv, VFNMADD, Single);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfnmadd_vf, VFNMADD, Single);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfmsub_vv, VFMSUB, Single);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfmsub_vf, VFMSUB, Single);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfnmsub_vv, VFNMSUB, Single);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfnmsub_vf, VFNMSUB, Single);

        $crate::tcg::translate_rvv_fp::fvv!(trans_vfwmacc_vv, VFWMACC, WidenAcc);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfwmacc_vf, VFWMACC, WidenAcc);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfwnmacc_vv, VFWNMACC, WidenAcc);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfwnmacc_vf, VFWNMACC, WidenAcc);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfwmsac_vv, VFWMSAC, WidenAcc);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfwmsac_vf, VFWMSAC, WidenAcc);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfwnmsac_vv, VFWNMSAC, WidenAcc);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfwnmsac_vf, VFWNMSAC, WidenAcc);

        $crate::tcg::translate_rvv_fp::fv!(trans_vfsqrt_v, VFSQRT, Opfv, RISCV_FRM_DYN);
        $crate::tcg::translate_rvv_fp::fv!(trans_vfrsqrt7_v, VFRSQRT7, Opfv, RISCV_FRM_DYN);
        $crate::tcg::translate_rvv_fp::fv!(trans_vfrec7_v, VFREC7, Opfv, RISCV_FRM_DYN);

        $crate::tcg::translate_rvv_fp::fvv!(trans_vfmin_vv, VFMIN, Single);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfmin_vf, VFMIN, Single);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfmax_vv, VFMAX, Single);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfmax_vf, VFMAX, Single);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfsgnj_vv, VFSGNJ, Single);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfsgnj_vf, VFSGNJ, Single);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfsgnjn_vv, VFSGNJN, Single);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfsgnjn_vf, VFSGNJN, Single);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfsgnjx_vv, VFSGNJX, Single);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfsgnjx_vf, VFSGNJX, Single);

        $crate::tcg::translate_rvv_fp::fvv!(trans_vmfeq_vv, VMFEQ, Cmp);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vmfeq_vf, VMFEQ, Cmp);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vmfne_vv, VMFNE, Cmp);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vmfne_vf, VMFNE, Cmp);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vmflt_vv, VMFLT, Cmp);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vmflt_vf, VMFLT, Cmp);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vmfle_vv, VMFLE, Cmp);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vmfle_vf, VMFLE, Cmp);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vmfgt_vf, VMFGT, Cmp);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vmfge_vf, VMFGE, Cmp);

        $crate::tcg::translate_rvv_fp::fv!(trans_vfclass_v, VFCLASS, Opfv, RISCV_FRM_DYN);
        $crate::tcg::translate_rvv_fp::fvf!(trans_vfmerge_vfm, VFMERGE, Single);

        fn trans_vfmv_v_f(&mut self, a: &mut arg_vfmv_v_f) -> bool {
            self.vfmv_v_f(a.rd, a.rs1)
        }

        $crate::tcg::translate_rvv_fp::fv!(trans_vfcvt_xu_f_v, VFCVT_XU_F, Opfv, RISCV_FRM_DYN);
        $crate::tcg::translate_rvv_fp::fv!(trans_vfcvt_x_f_v, VFCVT_X_F, Opfv, RISCV_FRM_DYN);
        $crate::tcg::translate_rvv_fp::fv!(trans_vfcvt_f_xu_v, VFCVT_F_XU, Opfv, RISCV_FRM_DYN);
        $crate::tcg::translate_rvv_fp::fv!(trans_vfcvt_f_x_v, VFCVT_F_X, Opfv, RISCV_FRM_DYN);
        $crate::tcg::translate_rvv_fp::fv!(trans_vfcvt_rtz_xu_f_v, VFCVT_XU_F, Opfv, RISCV_FRM_RTZ);
        $crate::tcg::translate_rvv_fp::fv!(trans_vfcvt_rtz_x_f_v, VFCVT_X_F, Opfv, RISCV_FRM_RTZ);

        $crate::tcg::translate_rvv_fp::fv!(
            trans_vfwcvt_xu_f_v,
            VFWCVT_XU_F,
            WidenXF,
            RISCV_FRM_DYN
        );
        $crate::tcg::translate_rvv_fp::fv!(trans_vfwcvt_x_f_v, VFWCVT_X_F, WidenXF, RISCV_FRM_DYN);
        $crate::tcg::translate_rvv_fp::fv!(trans_vfwcvt_f_f_v, VFWCVT_F_F, WidenFF, RISCV_FRM_DYN);
        $crate::tcg::translate_rvv_fp::fv!(
            trans_vfwcvt_rtz_xu_f_v,
            VFWCVT_XU_F,
            WidenXF,
            RISCV_FRM_RTZ
        );
        $crate::tcg::translate_rvv_fp::fv!(
            trans_vfwcvt_rtz_x_f_v,
            VFWCVT_X_F,
            WidenXF,
            RISCV_FRM_RTZ
        );
        $crate::tcg::translate_rvv_fp::fv!(
            trans_vfwcvt_f_xu_v,
            VFWCVT_F_XU,
            WidenFX,
            RISCV_FRM_DYN
        );
        $crate::tcg::translate_rvv_fp::fv!(trans_vfwcvt_f_x_v, VFWCVT_F_X, WidenFX, RISCV_FRM_DYN);

        $crate::tcg::translate_rvv_fp::fv!(
            trans_vfncvt_f_xu_w,
            VFNCVT_F_XU,
            NarrowFX,
            RISCV_FRM_DYN
        );
        $crate::tcg::translate_rvv_fp::fv!(trans_vfncvt_f_x_w, VFNCVT_F_X, NarrowFX, RISCV_FRM_DYN);
        $crate::tcg::translate_rvv_fp::fv!(trans_vfncvt_f_f_w, VFNCVT_F_F, NarrowFF, RISCV_FRM_DYN);

        fn trans_vfncvt_rod_f_f_w(&mut self, a: &mut arg_vfncvt_rod_f_f_w) -> bool {
            self.opfv_rod($crate::tcg::translate_rvv_fp::rargs(a))
        }

        $crate::tcg::translate_rvv_fp::fv!(
            trans_vfncvt_xu_f_w,
            VFNCVT_XU_F,
            NarrowXF,
            RISCV_FRM_DYN
        );
        $crate::tcg::translate_rvv_fp::fv!(trans_vfncvt_x_f_w, VFNCVT_X_F, NarrowXF, RISCV_FRM_DYN);
        $crate::tcg::translate_rvv_fp::fv!(
            trans_vfncvt_rtz_xu_f_w,
            VFNCVT_XU_F,
            NarrowXF,
            RISCV_FRM_RTZ
        );
        $crate::tcg::translate_rvv_fp::fv!(
            trans_vfncvt_rtz_x_f_w,
            VFNCVT_X_F,
            NarrowXF,
            RISCV_FRM_RTZ
        );

        $crate::tcg::translate_rvv_fp::fvv!(trans_vfredusum_vs, VFREDUSUM, Red);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfredosum_vs, VFREDOSUM, Red);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfredmax_vs, VFREDMAX, Red);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfredmin_vs, VFREDMIN, Red);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfwredusum_vs, VFWREDUSUM, RedWiden);
        $crate::tcg::translate_rvv_fp::fvv!(trans_vfwredosum_vs, VFWREDOSUM, RedWiden);

        // Zvfbfmin and Zvfbfwma.

        $crate::tcg::translate_rvv_fp::fv!(
            trans_vfncvtbf16_f_f_w,
            VFNCVTBF16_F_F,
            Bf16Narrow,
            RISCV_FRM_DYN
        );
        $crate::tcg::translate_rvv_fp::fv!(
            trans_vfwcvtbf16_f_f_v,
            VFWCVTBF16_F_F,
            Bf16Widen,
            RISCV_FRM_DYN
        );

        fn trans_vfwmaccbf16_vv(&mut self, a: &mut arg_vfwmaccbf16_vv) -> bool {
            self.vfwmaccbf16($crate::tcg::translate_rvv::vargs(a), false)
        }

        fn trans_vfwmaccbf16_vf(&mut self, a: &mut arg_vfwmaccbf16_vf) -> bool {
            self.vfwmaccbf16($crate::tcg::translate_rvv::vargs(a), true)
        }
    };
}
pub(super) use rvv_fp_trans32;
