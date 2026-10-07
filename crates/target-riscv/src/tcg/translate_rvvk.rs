// SPDX-License-Identifier: GPL-2.0-or-later

//! The vector crypto instructions, a port of QEMU's `trans_rvvk.c.inc`: Zvbc, Zvkb, Zvbb,
//! Zvkned, Zvknha, Zvknhb, Zvksh, Zvkg and Zvksed (Zvkt has no instructions). The helpers
//! are in [`super::vcrypto`].
//!
//! The checks of QEMU's `opivv_check()` family are private copies here, with a `zvk_`
//! prefix, so that they do not clash with the methods of the other vector files.
//!
//! Deliberate differences from QEMU:
//!
//! - QEMU expands `vrol`, `vror` and `vandn` inline with gvec when `vl` is VLMAX. Here they
//!   always call the helper, which gives the same result.
//! - QEMU only calls `egs_check` for the element group instructions when it cannot tell at
//!   translation time that `vstart` is zero and `vl` is VLMAX. There is no `vl_eq_vlmax`
//!   flag here, so the opcode is always saved and every element group helper does the
//!   check itself. When `vl` is VLMAX it is a multiple of the element group size, so the
//!   check passes in the cases QEMU skips it.

use super::helpers::Def;
use super::translate::S;
use super::translate_rvv::{VArgs, VSrc, is_overlapped, require_align, vargs};
use super::vcrypto;
use crate::cpu::VLENB;
use crate::decode::insn32::{arg_rmr, arg_rmrr};

/// `ZVKNED_EGS`, `ZVKNH_EGS`, `ZVKG_EGS` and `ZVKSED_EGS`.
const EGS4: i32 = 4;
/// `ZVKSH_EGS`.
const ZVKSH_EGS: i32 = 8;

/// `MO_32`.
const MO_32: i32 = 2;
/// `MO_64`.
const MO_64: i32 = 3;

// The checks.
impl S<'_, '_> {
    /// `opivv_check()`.
    fn zvk_opivv_check(&self, a: &arg_rmrr) -> bool {
        self.require_rvv()
            && self.vext_check_isa_ill()
            && self.vext_check_sss(a.rd, a.rs1, a.rs2, a.vm)
    }

    /// `opivx_check()`.
    fn zvk_opivx_check(&self, a: &arg_rmrr) -> bool {
        self.require_rvv() && self.vext_check_isa_ill() && self.vext_check_ss(a.rd, a.rs2, a.vm)
    }

    /// `opivv_check()` for `src` `VSrc::V`, `opivx_check()` for the others.
    fn zvk_opiv_check(&self, a: &arg_rmrr, src: VSrc) -> bool {
        match src {
            VSrc::V => self.zvk_opivv_check(a),
            _ => self.zvk_opivx_check(a),
        }
    }

    /// `MAXSZ()`: the bytes of a register group.
    fn zvk_maxsz(&self) -> i32 {
        (VLENB as i32 * 8) >> (3 - self.d.lmul)
    }

    /// `MAXSZ(s) >= egw_bytes` for element groups of `egs` elements.
    fn zvk_egw_fits(&self, egs: i32) -> bool {
        self.zvk_maxsz() >= egs << self.d.sew
    }

    /// `1 << MAX(s->lmul, 0)`: the registers of a group.
    fn zvk_mult(&self) -> i32 {
        1 << self.d.lmul.max(0)
    }

    /// `vaes_check_vv()`, `vaeskf1_check()`, `vaeskf2_check()`, `vsm4k_vi_check()` and
    /// `vsm4r_vv_check()` without their extension: `vd` and `vs2` aligned to LMUL.
    fn zvk_check_vv(&self, rd: i32, rs2: i32) -> bool {
        self.require_rvv()
            && self.vext_check_isa_ill()
            && self.zvk_egw_fits(EGS4)
            && require_align(rd, self.d.lmul)
            && require_align(rs2, self.d.lmul)
            && self.d.sew == MO_32
    }

    /// `vaes_check_vs()`.
    fn zvk_aes_check_vs(&self, a: &arg_rmr) -> bool {
        // vaes_check_overlap().
        let op_size = if self.d.lmul <= 0 { 1 } else { 1 << self.d.lmul };
        !is_overlapped(a.rd, op_size, a.rs2, 1)
            && self.zvk_egw_fits(EGS4)
            && self.d.cfg.ext_zvkned
            && self.require_rvv()
            && self.vext_check_isa_ill()
            && require_align(a.rd, self.d.lmul)
            && self.d.sew == MO_32
    }

    /// `vsha_check()`.
    fn zvk_sha_check(&self, a: &arg_rmrr) -> bool {
        let cfg = &self.d.cfg;
        let sew = self.d.sew;
        let mult = self.zvk_mult();
        self.zvk_opivv_check(a)
            // vsha_check_sew().
            && ((cfg.ext_zvknha && sew == MO_32)
                || (cfg.ext_zvknhb && (sew == MO_32 || sew == MO_64)))
            && self.zvk_egw_fits(EGS4)
            && !is_overlapped(a.rd, mult, a.rs1, mult)
            && !is_overlapped(a.rd, mult, a.rs2, mult)
    }

    /// `vsm3_check()`.
    fn zvk_sm3_check(&self, a: &arg_rmrr) -> bool {
        let mult = self.zvk_mult();
        self.d.cfg.ext_zvksh
            && self.require_rvv()
            && self.vext_check_isa_ill()
            && !is_overlapped(a.rd, mult, a.rs2, mult)
            && self.zvk_egw_fits(ZVKSH_EGS)
            && self.d.sew == MO_32
    }

    /// `zvksed_check()`.
    fn zvk_sm4_check(&self) -> bool {
        self.d.cfg.ext_zvksed
            && self.require_rvv()
            && self.vext_check_isa_ill()
            && self.zvk_egw_fits(EGS4)
            && self.d.sew == MO_32
    }
}

// The translation.
impl S<'_, '_> {
    /// `GEN_V_UNMASKED_TRANS()`, `GEN_VI_UNMASKED_TRANS()` and `GEN_VV_UNMASKED_TRANS()`:
    /// call the element group helper `h` on `vd`, `vs1`, `vs2` and the immediate `uimm`.
    fn zvk_group(&mut self, h: &Def, a: VArgs, uimm: i32) -> bool {
        // Save the opcode for the illegal instruction exception of the element group size
        // check in the helper.
        self.decode_save_opc(0);
        let desc = self.vdesc(a.rd, a.rs1, a.rs2, a.vm);
        let imm = self.c64(i64::from(uimm));
        self.vcall(h, None, desc, &[imm.into()]);
        self.finalize_rvv_inst();
        true
    }

    /// `vclmul_vv_check()` and `vclmul_vx_check()`, then the helper.
    pub(super) fn zvk_clmul(&mut self, h: &Def, a: &arg_rmrr, src: VSrc) -> bool {
        self.zvk_opiv_check(a, src)
            && self.d.cfg.ext_zvbc
            && self.d.sew == MO_64
            && self.gen_vop(h, vargs(a), src)
    }

    /// `zvkb_vv_check()` and `zvkb_vx_check()`, then the helper: `vrol`, `vror`, `vandn`.
    pub(super) fn zvk_kb(&mut self, h: &Def, a: &arg_rmrr, src: VSrc) -> bool {
        self.zvk_opiv_check(a, src)
            && (self.d.cfg.ext_zvbb || self.d.cfg.ext_zvkb)
            && self.gen_vop(h, vargs(a), src)
    }

    /// `GEN_OPIV_TRANS()` with `zvkb_opiv_check()` (`zvkb`) or `zvbb_opiv_check()`.
    pub(super) fn zvk_unary(&mut self, h: &Def, a: &arg_rmr, zvkb: bool) -> bool {
        let ext = self.d.cfg.ext_zvbb || (zvkb && self.d.cfg.ext_zvkb);
        ext && self.require_rvv()
            && self.vext_check_isa_ill()
            && self.vext_check_ss(a.rd, a.rs2, a.vm)
            && self.gen_vop(h, VArgs { rd: a.rd, rs1: 0, rs2: a.rs2, vm: a.vm }, VSrc::V)
    }

    /// `vwsll_vv_check()` and `vwsll_vx_check()`, then the helper.
    pub(super) fn zvk_vwsll(&mut self, a: &arg_rmrr, src: VSrc) -> bool {
        let ok = self.d.cfg.ext_zvbb
            && match src {
                // opivv_widen_check().
                VSrc::V => {
                    self.require_rvv()
                        && self.vext_check_isa_ill()
                        && self.vext_check_dss(a.rd, a.rs1, a.rs2, a.vm)
                }
                // opivx_widen_check().
                _ => {
                    self.require_rvv()
                        && self.vext_check_isa_ill()
                        && self.vext_check_ds(a.rd, a.rs2, a.vm)
                }
            };
        ok && self.gen_vop(&vcrypto::VWSLL, vargs(a), src)
    }

    /// The `.vv` forms of `vaesef`, `vaesdf`, `vaesem` and `vaesdm`: `vaes_check_vv()`.
    pub(super) fn zvk_aes_vv(&mut self, h: &Def, a: &arg_rmr) -> bool {
        self.d.cfg.ext_zvkned
            && self.zvk_check_vv(a.rd, a.rs2)
            && self.zvk_group(h, VArgs { rd: a.rd, rs1: 0, rs2: a.rs2, vm: a.vm }, 0)
    }

    /// The `.vs` forms and `vaesz.vs`: `vaes_check_vs()`.
    pub(super) fn zvk_aes_vs(&mut self, h: &Def, a: &arg_rmr) -> bool {
        self.zvk_aes_check_vs(a)
            && self.zvk_group(h, VArgs { rd: a.rd, rs1: 0, rs2: a.rs2, vm: a.vm }, 0)
    }

    /// `vaeskf1.vi` and `vaeskf2.vi`: `vaeskf1_check()` and `vaeskf2_check()`.
    pub(super) fn zvk_aeskf(&mut self, h: &Def, a: &arg_rmrr) -> bool {
        self.d.cfg.ext_zvkned
            && self.zvk_check_vv(a.rd, a.rs2)
            && self.zvk_group(h, VArgs { rs1: 0, ..vargs(a) }, a.rs1)
    }

    /// `vsha2ms.vv`, `vsha2ch.vv` and `vsha2cl.vv`.
    pub(super) fn zvk_sha(&mut self, h: &Def, a: &arg_rmrr) -> bool {
        self.zvk_sha_check(a) && self.zvk_group(h, vargs(a), 0)
    }

    /// `vsm3me.vv`: `vsm3me_check()`.
    pub(super) fn zvk_sm3me(&mut self, a: &arg_rmrr) -> bool {
        self.zvk_sm3_check(a)
            && self.vext_check_sss(a.rd, a.rs1, a.rs2, a.vm)
            && self.zvk_group(&vcrypto::VSM3ME_VV, vargs(a), 0)
    }

    /// `vsm3c.vi`: `vsm3c_check()`.
    pub(super) fn zvk_sm3c(&mut self, a: &arg_rmrr) -> bool {
        self.zvk_sm3_check(a)
            && self.vext_check_ss(a.rd, a.rs2, a.vm)
            && self.zvk_group(&vcrypto::VSM3C_VI, VArgs { rs1: 0, ..vargs(a) }, a.rs1)
    }

    /// `vgmul.vv`: `vgmul_check()`.
    pub(super) fn zvk_vgmul(&mut self, a: &arg_rmr) -> bool {
        self.d.cfg.ext_zvkg
            && self.vext_check_isa_ill()
            && self.require_rvv()
            && self.zvk_egw_fits(EGS4)
            && self.vext_check_ss(a.rd, a.rs2, a.vm)
            && self.d.sew == MO_32
            && self.zvk_group(
                &vcrypto::VGMUL_VV,
                VArgs { rd: a.rd, rs1: 0, rs2: a.rs2, vm: a.vm },
                0,
            )
    }

    /// `vghsh.vv`: `vghsh_check()`.
    pub(super) fn zvk_vghsh(&mut self, a: &arg_rmrr) -> bool {
        self.d.cfg.ext_zvkg
            && self.zvk_opivv_check(a)
            && self.zvk_egw_fits(EGS4)
            && self.d.sew == MO_32
            && self.zvk_group(&vcrypto::VGHSH_VV, vargs(a), 0)
    }

    /// `vsm4k.vi`: `vsm4k_vi_check()`.
    pub(super) fn zvk_sm4k(&mut self, a: &arg_rmrr) -> bool {
        self.zvk_sm4_check()
            && require_align(a.rd, self.d.lmul)
            && require_align(a.rs2, self.d.lmul)
            && self.zvk_group(&vcrypto::VSM4K_VI, VArgs { rs1: 0, ..vargs(a) }, a.rs1)
    }

    /// `vsm4r.vv`: `vsm4r_vv_check()`.
    pub(super) fn zvk_sm4r_vv(&mut self, a: &arg_rmr) -> bool {
        self.zvk_sm4_check()
            && require_align(a.rd, self.d.lmul)
            && require_align(a.rs2, self.d.lmul)
            && self.zvk_group(
                &vcrypto::VSM4R_VV,
                VArgs { rd: a.rd, rs1: 0, rs2: a.rs2, vm: a.vm },
                0,
            )
    }

    /// `vsm4r.vs`: `vsm4r_vs_check()`.
    pub(super) fn zvk_sm4r_vs(&mut self, a: &arg_rmr) -> bool {
        self.zvk_sm4_check()
            && !is_overlapped(a.rd, self.zvk_mult(), a.rs2, 1)
            && require_align(a.rd, self.d.lmul)
            && self.zvk_group(
                &vcrypto::VSM4R_VS,
                VArgs { rd: a.rd, rs1: 0, rs2: a.rs2, vm: a.vm },
                0,
            )
    }
}

/// Generate `trans_*` methods that call a method of this file with a helper and a form:
/// `zvk_trans!(method(before; after) => trans_a(arg_a))` calls `method(before, a, after)`,
/// `zvk_trans!(method(before) => trans_a(arg_a))` calls `method(before, a)`.
macro_rules! zvk_trans {
    ($m:ident($($e:expr),*; $($x:expr),*) => $t:ident($arg:ty)) => {
        fn $t(&mut self, a: &mut $arg) -> bool {
            self.$m($($e,)* a, $($x),*)
        }
    };
    ($m:ident($($e:expr),*) => $t:ident($arg:ty)) => {
        fn $t(&mut self, a: &mut $arg) -> bool {
            self.$m($($e,)* a)
        }
    };
}
pub(super) use zvk_trans;

/// The vector crypto `trans_*` methods of `DecodeInsn32`.
macro_rules! rvvk_trans32 {
    () => {
        // Zvbc.
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_clmul(&$crate::tcg::vcrypto::VCLMUL;
            $crate::tcg::translate_rvv::VSrc::V) => trans_vclmul_vv(arg_vclmul_vv));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_clmul(&$crate::tcg::vcrypto::VCLMUL;
            $crate::tcg::translate_rvv::VSrc::X) => trans_vclmul_vx(arg_vclmul_vx));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_clmul(&$crate::tcg::vcrypto::VCLMULH;
            $crate::tcg::translate_rvv::VSrc::V) => trans_vclmulh_vv(arg_vclmulh_vv));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_clmul(&$crate::tcg::vcrypto::VCLMULH;
            $crate::tcg::translate_rvv::VSrc::X) => trans_vclmulh_vx(arg_vclmulh_vx));

        // Zvkb.
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_kb(&$crate::tcg::vcrypto::VROL;
            $crate::tcg::translate_rvv::VSrc::V) => trans_vrol_vv(arg_vrol_vv));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_kb(&$crate::tcg::vcrypto::VROL;
            $crate::tcg::translate_rvv::VSrc::X) => trans_vrol_vx(arg_vrol_vx));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_kb(&$crate::tcg::vcrypto::VROR;
            $crate::tcg::translate_rvv::VSrc::V) => trans_vror_vv(arg_vror_vv));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_kb(&$crate::tcg::vcrypto::VROR;
            $crate::tcg::translate_rvv::VSrc::X) => trans_vror_vx(arg_vror_vx));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_kb(&$crate::tcg::vcrypto::VROR;
            $crate::tcg::translate_rvv::VSrc::I($crate::tcg::translate_rvv::ImmMode::TruncSew))
            => trans_vror_vi(arg_vror_vi));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_kb(&$crate::tcg::vcrypto::VANDN;
            $crate::tcg::translate_rvv::VSrc::V) => trans_vandn_vv(arg_vandn_vv));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_kb(&$crate::tcg::vcrypto::VANDN;
            $crate::tcg::translate_rvv::VSrc::X) => trans_vandn_vx(arg_vandn_vx));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_unary(&$crate::tcg::vcrypto::VBREV8; true)
            => trans_vbrev8_v(arg_vbrev8_v));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_unary(&$crate::tcg::vcrypto::VREV8; true)
            => trans_vrev8_v(arg_vrev8_v));

        // Zvbb.
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_unary(&$crate::tcg::vcrypto::VBREV; false)
            => trans_vbrev_v(arg_vbrev_v));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_unary(&$crate::tcg::vcrypto::VCLZ; false)
            => trans_vclz_v(arg_vclz_v));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_unary(&$crate::tcg::vcrypto::VCTZ; false)
            => trans_vctz_v(arg_vctz_v));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_unary(&$crate::tcg::vcrypto::VCPOP_V; false)
            => trans_vcpop_v(arg_vcpop_v));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_vwsll(; $crate::tcg::translate_rvv::VSrc::V)
            => trans_vwsll_vv(arg_vwsll_vv));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_vwsll(; $crate::tcg::translate_rvv::VSrc::X)
            => trans_vwsll_vx(arg_vwsll_vx));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_vwsll(;
            $crate::tcg::translate_rvv::VSrc::I($crate::tcg::translate_rvv::ImmMode::Zx))
            => trans_vwsll_vi(arg_vwsll_vi));

        // Zvkned.
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_aes_vv(&$crate::tcg::vcrypto::VAESEF_VV)
            => trans_vaesef_vv(arg_vaesef_vv));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_aes_vs(&$crate::tcg::vcrypto::VAESEF_VS)
            => trans_vaesef_vs(arg_vaesef_vs));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_aes_vv(&$crate::tcg::vcrypto::VAESDF_VV)
            => trans_vaesdf_vv(arg_vaesdf_vv));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_aes_vs(&$crate::tcg::vcrypto::VAESDF_VS)
            => trans_vaesdf_vs(arg_vaesdf_vs));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_aes_vv(&$crate::tcg::vcrypto::VAESDM_VV)
            => trans_vaesdm_vv(arg_vaesdm_vv));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_aes_vs(&$crate::tcg::vcrypto::VAESDM_VS)
            => trans_vaesdm_vs(arg_vaesdm_vs));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_aes_vs(&$crate::tcg::vcrypto::VAESZ_VS)
            => trans_vaesz_vs(arg_vaesz_vs));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_aes_vv(&$crate::tcg::vcrypto::VAESEM_VV)
            => trans_vaesem_vv(arg_vaesem_vv));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_aes_vs(&$crate::tcg::vcrypto::VAESEM_VS)
            => trans_vaesem_vs(arg_vaesem_vs));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_aeskf(&$crate::tcg::vcrypto::VAESKF1_VI)
            => trans_vaeskf1_vi(arg_vaeskf1_vi));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_aeskf(&$crate::tcg::vcrypto::VAESKF2_VI)
            => trans_vaeskf2_vi(arg_vaeskf2_vi));

        // Zvknha and Zvknhb.
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_sha(&$crate::tcg::vcrypto::VSHA2MS_VV)
            => trans_vsha2ms_vv(arg_vsha2ms_vv));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_sha(&$crate::tcg::vcrypto::VSHA2CH_VV)
            => trans_vsha2ch_vv(arg_vsha2ch_vv));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_sha(&$crate::tcg::vcrypto::VSHA2CL_VV)
            => trans_vsha2cl_vv(arg_vsha2cl_vv));

        // Zvksh.
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_sm3me() => trans_vsm3me_vv(arg_vsm3me_vv));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_sm3c() => trans_vsm3c_vi(arg_vsm3c_vi));

        // Zvkg.
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_vgmul() => trans_vgmul_vv(arg_vgmul_vv));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_vghsh() => trans_vghsh_vv(arg_vghsh_vv));

        // Zvksed.
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_sm4k() => trans_vsm4k_vi(arg_vsm4k_vi));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_sm4r_vv() => trans_vsm4r_vv(arg_vsm4r_vv));
        $crate::tcg::translate_rvvk::zvk_trans!(zvk_sm4r_vs() => trans_vsm4r_vs(arg_vsm4r_vs));
    };
}
pub(super) use rvvk_trans32;
