// SPDX-License-Identifier: GPL-2.0-or-later

//! The vector integer and fixed point instructions, a port of the integer and fixed point
//! parts of QEMU's `trans_rvv.c.inc`: "Vector Integer Arithmetic Instructions" to "Vector
//! Fixed-Point Arithmetic Instructions", and the integer extensions.
//!
//! Each `trans_*` method runs the check QEMU's runs ([`Chk`]) and calls the helper of
//! [`super::vector_int`] that serves every SEW.
//!
//! Deliberate differences from QEMU:
//!
//! - QEMU expands some instructions inline with gvec when `vm` is set and `vl` is VLMAX
//!   (`vadd`, `vand`, `vsll`, `vmul`, `vmin`, `vmv.v.*` and others). Here they always call
//!   the helper, which leaves the same registers: with `vl` equal to VLMAX and `vstart` 0
//!   every element is written, and the tail past VLMAX only exists for a fractional LMUL,
//!   where QEMU uses the helper too when the tail is agnostic.

use super::helpers::Def;
use super::translate::S;
use super::translate_rvv::{VArgs, VSrc, require_align, require_noover, require_vm, vargs};
use crate::decode::insn32::{arg_r2, arg_rmr, arg_rmrr};

/// The checks of the integer and fixed point instructions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Chk {
    /// `opivv_check()`.
    Vv,
    /// `opivx_check()`.
    Vx,
    /// `opivv_widen_check()`.
    WidenVv,
    /// `opivx_widen_check()`.
    WidenVx,
    /// `opivv_overwrite_widen_check()`.
    OverWidenVv,
    /// `opivx_overwrite_widen_check()`.
    OverWidenVx,
    /// `opiwv_widen_check()`.
    Wv,
    /// `opiwx_widen_check()`.
    Wx,
    /// `opivv_vadc_check()`.
    AdcVv,
    /// `opivx_vadc_check()`.
    AdcVx,
    /// `opivv_vmadc_check()` and `opivv_cmp_check()`.
    MaskVv,
    /// `opivx_vmadc_check()` and `opivx_cmp_check()`.
    MaskVx,
    /// `opiwv_narrow_check()`.
    NarrowWv,
    /// `opiwx_narrow_check()`.
    NarrowWx,
    /// `vmulh_vv_check()` and `vsmul_vv_check()`.
    MulhVv,
    /// `vmulh_vx_check()` and `vsmul_vx_check()`.
    MulhVx,
}

impl S<'_, '_> {
    /// The check `c` of an integer or fixed point instruction.
    pub(super) fn int_check(&self, a: &arg_rmrr, c: Chk) -> bool {
        if !self.require_rvv() || !self.vext_check_isa_ill() {
            return false;
        }
        let (rd, rs1, rs2, vm) = (a.rd, a.rs1, a.rs2, a.vm);
        let sew = self.d.sew;
        // Zve64* has no vmulh* and vsmul for EEW 64.
        let mulh_ok = self.d.cfg.ext_v() || sew != 3;
        match c {
            Chk::Vv => self.vext_check_sss(rd, rs1, rs2, vm),
            Chk::Vx => self.vext_check_ss(rd, rs2, vm),
            Chk::WidenVv => self.vext_check_dss(rd, rs1, rs2, vm),
            Chk::WidenVx => self.vext_check_ds(rd, rs2, vm),
            Chk::OverWidenVv => {
                self.vext_check_dss(rd, rs1, rs2, vm)
                    && self.vext_check_input_eew(rd, sew + 1, rs1, sew, vm)
                    && self.vext_check_input_eew(rd, sew + 1, rs2, sew, vm)
            }
            Chk::OverWidenVx => {
                self.vext_check_ds(rd, rs2, vm)
                    && self.vext_check_input_eew(rd, sew + 1, rs2, sew, vm)
            }
            Chk::Wv => self.vext_check_dds(rd, rs1, rs2, vm),
            Chk::Wx => self.vext_check_dd(rd, rs2, vm),
            // vd cannot be v0, which holds the carries.
            Chk::AdcVv => rd != 0 && self.vext_check_sss(rd, rs1, rs2, vm),
            Chk::AdcVx => rd != 0 && self.vext_check_ss(rd, rs2, vm),
            Chk::MaskVv => self.vext_check_mss(rd, rs1, rs2),
            Chk::MaskVx => self.vext_check_ms(rd, rs2),
            Chk::NarrowWv => self.vext_check_sds(rd, rs1, rs2, vm),
            Chk::NarrowWx => self.vext_check_sd(rd, rs2, vm),
            Chk::MulhVv => self.vext_check_sss(rd, rs1, rs2, vm) && mulh_ok,
            Chk::MulhVx => self.vext_check_ss(rd, rs2, vm) && mulh_ok,
        }
    }

    /// `GEN_OPIVV_TRANS()`, `GEN_OPIVX_TRANS()`, `GEN_OPIVI_TRANS()` and their widening and
    /// narrowing kin: check `c`, then call `h` with the first source `src`.
    pub(super) fn int_op(&mut self, h: &Def, a: &arg_rmrr, c: Chk, src: VSrc) -> bool {
        self.int_check(a, c) && self.gen_vop(h, vargs(a), src)
    }

    /// `trans_vmv_v_v()`, `trans_vmv_v_x()` and `trans_vmv_v_i()`: `rs2` is 0 and `vm` 1.
    pub(super) fn int_vmv_v(&mut self, a: &arg_r2, src: VSrc) -> bool {
        let ok = self.require_rvv()
            && self.vext_check_isa_ill()
            && match src {
                VSrc::V => self.vext_check_sss(a.rd, a.rs1, 0, 1),
                _ => self.vext_check_ss(a.rd, 0, 1),
            };
        ok && self.gen_vop(
            &super::vector_int::VMV_V,
            VArgs { rd: a.rd, rs1: a.rs1, rs2: 0, vm: 1 },
            src,
        )
    }

    /// `int_ext_check()` and `int_ext_op()`: `vzext.vf<2^div>` or `vsext.vf<2^div>` with
    /// helper `h`.
    pub(super) fn int_ext(&mut self, h: &Def, a: &arg_rmr, div: i32) -> bool {
        let (sew, lmul) = (self.d.sew, self.d.lmul);
        let from = sew + 3 - div;
        let ok = self.require_rvv()
            && (3..=8).contains(&from)
            && a.rd != a.rs2
            && require_align(a.rd, lmul)
            && require_align(a.rs2, lmul - div)
            && require_vm(a.vm, a.rd)
            && require_noover(a.rd, lmul, a.rs2, lmul - div)
            && self.vext_check_input_eew(-1, 0, a.rs2, sew, a.vm);
        // QEMU has no helper for a source narrower than 8 bits.
        if !ok || sew < div {
            return false;
        }
        let mut desc = self.vdesc(a.rd, 0, a.rs2, a.vm);
        desc.x = div as u32;
        let zero = self.c64(0);
        self.vcall(h, None, desc, &[zero.into()]);
        self.finalize_rvv_inst();
        true
    }
}

/// `trans_a(arg_a) => HELPER, Check, Src;` makes a `trans_a` that calls [`S::int_op`] with
/// `vector_int::HELPER`, `Chk::Check` and `VSrc::Src`, where `Src` is `V`, `X` or
/// `I(ImmMode)`.
macro_rules! int_trans {
    ($($t:ident($arg:ident) => $h:ident, $c:ident, $src:ident $(($m:ident))?;)*) => {
        $(
            fn $t(&mut self, a: &mut $arg) -> bool {
                self.int_op(
                    &$crate::tcg::vector_int::$h,
                    a,
                    $crate::tcg::translate_rvv_int::Chk::$c,
                    $crate::tcg::translate_rvv::VSrc::$src
                        $(($crate::tcg::translate_rvv::ImmMode::$m))?,
                )
            }
        )*
    };
}
pub(super) use int_trans;

/// The `trans_*` methods of this file for `DecodeInsn32`.
macro_rules! rvv_int_trans32 {
    () => {
        $crate::tcg::translate_rvv_int::int_trans! {
            // Vector Single-Width Integer Add and Subtract.
            trans_vadd_vv(arg_vadd_vv) => VADD, Vv, V;
            trans_vadd_vx(arg_vadd_vx) => VADD, Vx, X;
            trans_vadd_vi(arg_vadd_vi) => VADD, Vx, I(Sx);
            trans_vsub_vv(arg_vsub_vv) => VSUB, Vv, V;
            trans_vsub_vx(arg_vsub_vx) => VSUB, Vx, X;
            trans_vrsub_vx(arg_vrsub_vx) => VRSUB, Vx, X;
            trans_vrsub_vi(arg_vrsub_vi) => VRSUB, Vx, I(Sx);

            // Vector Widening Integer Add/Subtract.
            trans_vwaddu_vv(arg_vwaddu_vv) => VWADDU, WidenVv, V;
            trans_vwaddu_vx(arg_vwaddu_vx) => VWADDU, WidenVx, X;
            trans_vwadd_vv(arg_vwadd_vv) => VWADD, WidenVv, V;
            trans_vwadd_vx(arg_vwadd_vx) => VWADD, WidenVx, X;
            trans_vwsubu_vv(arg_vwsubu_vv) => VWSUBU, WidenVv, V;
            trans_vwsubu_vx(arg_vwsubu_vx) => VWSUBU, WidenVx, X;
            trans_vwsub_vv(arg_vwsub_vv) => VWSUB, WidenVv, V;
            trans_vwsub_vx(arg_vwsub_vx) => VWSUB, WidenVx, X;
            trans_vwaddu_wv(arg_vwaddu_wv) => VWADDU_W, Wv, V;
            trans_vwaddu_wx(arg_vwaddu_wx) => VWADDU_W, Wx, X;
            trans_vwadd_wv(arg_vwadd_wv) => VWADD_W, Wv, V;
            trans_vwadd_wx(arg_vwadd_wx) => VWADD_W, Wx, X;
            trans_vwsubu_wv(arg_vwsubu_wv) => VWSUBU_W, Wv, V;
            trans_vwsubu_wx(arg_vwsubu_wx) => VWSUBU_W, Wx, X;
            trans_vwsub_wv(arg_vwsub_wv) => VWSUB_W, Wv, V;
            trans_vwsub_wx(arg_vwsub_wx) => VWSUB_W, Wx, X;

            // Vector Integer Add-with-Carry / Subtract-with-Borrow.
            trans_vadc_vvm(arg_vadc_vvm) => VADC, AdcVv, V;
            trans_vadc_vxm(arg_vadc_vxm) => VADC, AdcVx, X;
            trans_vadc_vim(arg_vadc_vim) => VADC, AdcVx, I(Sx);
            trans_vmadc_vvm(arg_vmadc_vvm) => VMADC, MaskVv, V;
            trans_vmadc_vxm(arg_vmadc_vxm) => VMADC, MaskVx, X;
            trans_vmadc_vim(arg_vmadc_vim) => VMADC, MaskVx, I(Sx);
            trans_vsbc_vvm(arg_vsbc_vvm) => VSBC, AdcVv, V;
            trans_vsbc_vxm(arg_vsbc_vxm) => VSBC, AdcVx, X;
            trans_vmsbc_vvm(arg_vmsbc_vvm) => VMSBC, MaskVv, V;
            trans_vmsbc_vxm(arg_vmsbc_vxm) => VMSBC, MaskVx, X;

            // Vector Bitwise Logical.
            trans_vand_vv(arg_vand_vv) => VAND, Vv, V;
            trans_vand_vx(arg_vand_vx) => VAND, Vx, X;
            trans_vand_vi(arg_vand_vi) => VAND, Vx, I(Sx);
            trans_vor_vv(arg_vor_vv) => VOR, Vv, V;
            trans_vor_vx(arg_vor_vx) => VOR, Vx, X;
            trans_vor_vi(arg_vor_vi) => VOR, Vx, I(Sx);
            trans_vxor_vv(arg_vxor_vv) => VXOR, Vv, V;
            trans_vxor_vx(arg_vxor_vx) => VXOR, Vx, X;
            trans_vxor_vi(arg_vxor_vi) => VXOR, Vx, I(Sx);

            // Vector Single-Width Bit Shift.
            trans_vsll_vv(arg_vsll_vv) => VSLL, Vv, V;
            trans_vsll_vx(arg_vsll_vx) => VSLL, Vx, X;
            trans_vsll_vi(arg_vsll_vi) => VSLL, Vx, I(TruncSew);
            trans_vsrl_vv(arg_vsrl_vv) => VSRL, Vv, V;
            trans_vsrl_vx(arg_vsrl_vx) => VSRL, Vx, X;
            trans_vsrl_vi(arg_vsrl_vi) => VSRL, Vx, I(TruncSew);
            trans_vsra_vv(arg_vsra_vv) => VSRA, Vv, V;
            trans_vsra_vx(arg_vsra_vx) => VSRA, Vx, X;
            trans_vsra_vi(arg_vsra_vi) => VSRA, Vx, I(TruncSew);

            // Vector Narrowing Integer Right Shift.
            trans_vnsrl_wv(arg_vnsrl_wv) => VNSRL, NarrowWv, V;
            trans_vnsrl_wx(arg_vnsrl_wx) => VNSRL, NarrowWx, X;
            trans_vnsrl_wi(arg_vnsrl_wi) => VNSRL, NarrowWx, I(Zx);
            trans_vnsra_wv(arg_vnsra_wv) => VNSRA, NarrowWv, V;
            trans_vnsra_wx(arg_vnsra_wx) => VNSRA, NarrowWx, X;
            trans_vnsra_wi(arg_vnsra_wi) => VNSRA, NarrowWx, I(Zx);

            // Vector Integer Comparison.
            trans_vmseq_vv(arg_vmseq_vv) => VMSEQ, MaskVv, V;
            trans_vmseq_vx(arg_vmseq_vx) => VMSEQ, MaskVx, X;
            trans_vmseq_vi(arg_vmseq_vi) => VMSEQ, MaskVx, I(Sx);
            trans_vmsne_vv(arg_vmsne_vv) => VMSNE, MaskVv, V;
            trans_vmsne_vx(arg_vmsne_vx) => VMSNE, MaskVx, X;
            trans_vmsne_vi(arg_vmsne_vi) => VMSNE, MaskVx, I(Sx);
            trans_vmsltu_vv(arg_vmsltu_vv) => VMSLTU, MaskVv, V;
            trans_vmsltu_vx(arg_vmsltu_vx) => VMSLTU, MaskVx, X;
            trans_vmslt_vv(arg_vmslt_vv) => VMSLT, MaskVv, V;
            trans_vmslt_vx(arg_vmslt_vx) => VMSLT, MaskVx, X;
            trans_vmsleu_vv(arg_vmsleu_vv) => VMSLEU, MaskVv, V;
            trans_vmsleu_vx(arg_vmsleu_vx) => VMSLEU, MaskVx, X;
            trans_vmsleu_vi(arg_vmsleu_vi) => VMSLEU, MaskVx, I(Sx);
            trans_vmsle_vv(arg_vmsle_vv) => VMSLE, MaskVv, V;
            trans_vmsle_vx(arg_vmsle_vx) => VMSLE, MaskVx, X;
            trans_vmsle_vi(arg_vmsle_vi) => VMSLE, MaskVx, I(Sx);
            trans_vmsgtu_vx(arg_vmsgtu_vx) => VMSGTU, MaskVx, X;
            trans_vmsgtu_vi(arg_vmsgtu_vi) => VMSGTU, MaskVx, I(Sx);
            trans_vmsgt_vx(arg_vmsgt_vx) => VMSGT, MaskVx, X;
            trans_vmsgt_vi(arg_vmsgt_vi) => VMSGT, MaskVx, I(Sx);

            // Vector Integer Min/Max.
            trans_vminu_vv(arg_vminu_vv) => VMINU, Vv, V;
            trans_vminu_vx(arg_vminu_vx) => VMINU, Vx, X;
            trans_vmin_vv(arg_vmin_vv) => VMIN, Vv, V;
            trans_vmin_vx(arg_vmin_vx) => VMIN, Vx, X;
            trans_vmaxu_vv(arg_vmaxu_vv) => VMAXU, Vv, V;
            trans_vmaxu_vx(arg_vmaxu_vx) => VMAXU, Vx, X;
            trans_vmax_vv(arg_vmax_vv) => VMAX, Vv, V;
            trans_vmax_vx(arg_vmax_vx) => VMAX, Vx, X;

            // Vector Single-Width Integer Multiply.
            trans_vmul_vv(arg_vmul_vv) => VMUL, Vv, V;
            trans_vmul_vx(arg_vmul_vx) => VMUL, Vx, X;
            trans_vmulh_vv(arg_vmulh_vv) => VMULH, MulhVv, V;
            trans_vmulh_vx(arg_vmulh_vx) => VMULH, MulhVx, X;
            trans_vmulhu_vv(arg_vmulhu_vv) => VMULHU, MulhVv, V;
            trans_vmulhu_vx(arg_vmulhu_vx) => VMULHU, MulhVx, X;
            trans_vmulhsu_vv(arg_vmulhsu_vv) => VMULHSU, MulhVv, V;
            trans_vmulhsu_vx(arg_vmulhsu_vx) => VMULHSU, MulhVx, X;

            // Vector Integer Divide.
            trans_vdivu_vv(arg_vdivu_vv) => VDIVU, Vv, V;
            trans_vdivu_vx(arg_vdivu_vx) => VDIVU, Vx, X;
            trans_vdiv_vv(arg_vdiv_vv) => VDIV, Vv, V;
            trans_vdiv_vx(arg_vdiv_vx) => VDIV, Vx, X;
            trans_vremu_vv(arg_vremu_vv) => VREMU, Vv, V;
            trans_vremu_vx(arg_vremu_vx) => VREMU, Vx, X;
            trans_vrem_vv(arg_vrem_vv) => VREM, Vv, V;
            trans_vrem_vx(arg_vrem_vx) => VREM, Vx, X;

            // Vector Widening Integer Multiply.
            trans_vwmul_vv(arg_vwmul_vv) => VWMUL, WidenVv, V;
            trans_vwmul_vx(arg_vwmul_vx) => VWMUL, WidenVx, X;
            trans_vwmulu_vv(arg_vwmulu_vv) => VWMULU, WidenVv, V;
            trans_vwmulu_vx(arg_vwmulu_vx) => VWMULU, WidenVx, X;
            trans_vwmulsu_vv(arg_vwmulsu_vv) => VWMULSU, WidenVv, V;
            trans_vwmulsu_vx(arg_vwmulsu_vx) => VWMULSU, WidenVx, X;

            // Vector Single-Width Integer Multiply-Add.
            trans_vmacc_vv(arg_vmacc_vv) => VMACC, Vv, V;
            trans_vmacc_vx(arg_vmacc_vx) => VMACC, Vx, X;
            trans_vnmsac_vv(arg_vnmsac_vv) => VNMSAC, Vv, V;
            trans_vnmsac_vx(arg_vnmsac_vx) => VNMSAC, Vx, X;
            trans_vmadd_vv(arg_vmadd_vv) => VMADD, Vv, V;
            trans_vmadd_vx(arg_vmadd_vx) => VMADD, Vx, X;
            trans_vnmsub_vv(arg_vnmsub_vv) => VNMSUB, Vv, V;
            trans_vnmsub_vx(arg_vnmsub_vx) => VNMSUB, Vx, X;

            // Vector Widening Integer Multiply-Add.
            trans_vwmaccu_vv(arg_vwmaccu_vv) => VWMACCU, OverWidenVv, V;
            trans_vwmaccu_vx(arg_vwmaccu_vx) => VWMACCU, OverWidenVx, X;
            trans_vwmacc_vv(arg_vwmacc_vv) => VWMACC, OverWidenVv, V;
            trans_vwmacc_vx(arg_vwmacc_vx) => VWMACC, OverWidenVx, X;
            trans_vwmaccsu_vv(arg_vwmaccsu_vv) => VWMACCSU, OverWidenVv, V;
            trans_vwmaccsu_vx(arg_vwmaccsu_vx) => VWMACCSU, OverWidenVx, X;
            trans_vwmaccus_vx(arg_vwmaccus_vx) => VWMACCUS, OverWidenVx, X;

            // Vector Integer Merge.
            trans_vmerge_vvm(arg_vmerge_vvm) => VMERGE, AdcVv, V;
            trans_vmerge_vxm(arg_vmerge_vxm) => VMERGE, AdcVx, X;
            trans_vmerge_vim(arg_vmerge_vim) => VMERGE, AdcVx, I(Sx);

            // Vector Single-Width Saturating Add and Subtract.
            trans_vsaddu_vv(arg_vsaddu_vv) => VSADDU, Vv, V;
            trans_vsaddu_vx(arg_vsaddu_vx) => VSADDU, Vx, X;
            trans_vsaddu_vi(arg_vsaddu_vi) => VSADDU, Vx, I(Sx);
            trans_vsadd_vv(arg_vsadd_vv) => VSADD, Vv, V;
            trans_vsadd_vx(arg_vsadd_vx) => VSADD, Vx, X;
            trans_vsadd_vi(arg_vsadd_vi) => VSADD, Vx, I(Sx);
            trans_vssubu_vv(arg_vssubu_vv) => VSSUBU, Vv, V;
            trans_vssubu_vx(arg_vssubu_vx) => VSSUBU, Vx, X;
            trans_vssub_vv(arg_vssub_vv) => VSSUB, Vv, V;
            trans_vssub_vx(arg_vssub_vx) => VSSUB, Vx, X;

            // Vector Single-Width Averaging Add and Subtract.
            trans_vaadd_vv(arg_vaadd_vv) => VAADD, Vv, V;
            trans_vaadd_vx(arg_vaadd_vx) => VAADD, Vx, X;
            trans_vaaddu_vv(arg_vaaddu_vv) => VAADDU, Vv, V;
            trans_vaaddu_vx(arg_vaaddu_vx) => VAADDU, Vx, X;
            trans_vasub_vv(arg_vasub_vv) => VASUB, Vv, V;
            trans_vasub_vx(arg_vasub_vx) => VASUB, Vx, X;
            trans_vasubu_vv(arg_vasubu_vv) => VASUBU, Vv, V;
            trans_vasubu_vx(arg_vasubu_vx) => VASUBU, Vx, X;

            // Vector Single-Width Fractional Multiply with Rounding and Saturation.
            trans_vsmul_vv(arg_vsmul_vv) => VSMUL, MulhVv, V;
            trans_vsmul_vx(arg_vsmul_vx) => VSMUL, MulhVx, X;

            // Vector Single-Width Scaling Shift.
            trans_vssrl_vv(arg_vssrl_vv) => VSSRL, Vv, V;
            trans_vssrl_vx(arg_vssrl_vx) => VSSRL, Vx, X;
            trans_vssrl_vi(arg_vssrl_vi) => VSSRL, Vx, I(TruncSew);
            trans_vssra_vv(arg_vssra_vv) => VSSRA, Vv, V;
            trans_vssra_vx(arg_vssra_vx) => VSSRA, Vx, X;
            trans_vssra_vi(arg_vssra_vi) => VSSRA, Vx, I(TruncSew);

            // Vector Narrowing Fixed-Point Clip.
            trans_vnclipu_wv(arg_vnclipu_wv) => VNCLIPU, NarrowWv, V;
            trans_vnclipu_wx(arg_vnclipu_wx) => VNCLIPU, NarrowWx, X;
            trans_vnclipu_wi(arg_vnclipu_wi) => VNCLIPU, NarrowWx, I(Zx);
            trans_vnclip_wv(arg_vnclip_wv) => VNCLIP, NarrowWv, V;
            trans_vnclip_wx(arg_vnclip_wx) => VNCLIP, NarrowWx, X;
            trans_vnclip_wi(arg_vnclip_wi) => VNCLIP, NarrowWx, I(Zx);
        }

        // Vector Integer Move.
        fn trans_vmv_v_v(&mut self, a: &mut arg_vmv_v_v) -> bool {
            self.int_vmv_v(a, $crate::tcg::translate_rvv::VSrc::V)
        }
        fn trans_vmv_v_x(&mut self, a: &mut arg_vmv_v_x) -> bool {
            self.int_vmv_v(a, $crate::tcg::translate_rvv::VSrc::X)
        }
        fn trans_vmv_v_i(&mut self, a: &mut arg_vmv_v_i) -> bool {
            self.int_vmv_v(
                a,
                $crate::tcg::translate_rvv::VSrc::I($crate::tcg::translate_rvv::ImmMode::Sx),
            )
        }

        // Vector Integer Extension.
        fn trans_vzext_vf2(&mut self, a: &mut arg_vzext_vf2) -> bool {
            self.int_ext(&$crate::tcg::vector_int::VZEXT, a, 1)
        }
        fn trans_vzext_vf4(&mut self, a: &mut arg_vzext_vf4) -> bool {
            self.int_ext(&$crate::tcg::vector_int::VZEXT, a, 2)
        }
        fn trans_vzext_vf8(&mut self, a: &mut arg_vzext_vf8) -> bool {
            self.int_ext(&$crate::tcg::vector_int::VZEXT, a, 3)
        }
        fn trans_vsext_vf2(&mut self, a: &mut arg_vsext_vf2) -> bool {
            self.int_ext(&$crate::tcg::vector_int::VSEXT, a, 1)
        }
        fn trans_vsext_vf4(&mut self, a: &mut arg_vsext_vf4) -> bool {
            self.int_ext(&$crate::tcg::vector_int::VSEXT, a, 2)
        }
        fn trans_vsext_vf8(&mut self, a: &mut arg_vsext_vf8) -> bool {
            self.int_ext(&$crate::tcg::vector_int::VSEXT, a, 3)
        }
    };
}
pub(super) use rvv_int_trans32;
