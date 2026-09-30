// SPDX-License-Identifier: MIT OR Apache-2.0

//! Vector ops, a port of `tcg/tcg-op-vec.c`.
//!
//! The host is assumed to support every vector op for every element size, so each method emits
//! the op itself and the expansions QEMU falls back on are not ported. The generic vector
//! expanders of `tcg-op-gvec.c` are not ported either.

use crate::ir::{Func, TempI32, TempI64, TempPtr, TempVec};
use crate::opcode::Opcode;
use crate::types::{Cond, Type};

macro_rules! vec_op3 {
    ($($(#[$m:meta])* $name:ident => $opc:ident;)*) => {$(
        $(#[$m])*
        pub fn $name(&mut self, vece: u32, r: TempVec, a: TempVec, b: TempVec) {
            self.vec_gen_op3(Opcode::$opc, vece, r, a, b)
        }
    )*};
}

macro_rules! vec_logic3 {
    ($($(#[$m:meta])* $name:ident => $opc:ident;)*) => {$(
        $(#[$m])*
        pub fn $name(&mut self, _vece: u32, r: TempVec, a: TempVec, b: TempVec) {
            self.vec_gen_op3(Opcode::$opc, 0, r, a, b)
        }
    )*};
}

macro_rules! vec_shifti {
    ($($(#[$m:meta])* $name:ident => $opc:ident;)*) => {$(
        $(#[$m])*
        pub fn $name(&mut self, vece: u32, r: TempVec, a: TempVec, i: i64) {
            self.vec_shifti(Opcode::$opc, vece, r, a, i)
        }
    )*};
}

macro_rules! vec_shifts {
    ($($(#[$m:meta])* $name:ident => $opc:ident;)*) => {$(
        $(#[$m])*
        pub fn $name(&mut self, vece: u32, r: TempVec, a: TempVec, s: TempI32) {
            let ty = self.vec_type(r);
            self.vec_gen(Opcode::$opc, ty, vece, &[r.arg(), a.arg(), s.arg()]);
        }
    )*};
}

impl Func {
    fn vec_type(&self, r: TempVec) -> Type {
        self.temp(r).base_type
    }

    fn vec_gen(&mut self, opc: Opcode, ty: Type, vece: u32, args: &[u64]) {
        assert!(vece <= 3, "bad element size");
        let op = self.emit_op(opc, ty, args);
        self.op_mut(op).vece = vece as u8;
    }

    fn vec_gen_op2(&mut self, opc: Opcode, vece: u32, r: TempVec, a: TempVec) {
        let ty = self.vec_type(r);
        assert!(self.vec_type(a) >= ty, "input vector too small");
        self.vec_gen(opc, ty, vece, &[r.arg(), a.arg()]);
    }

    fn vec_gen_op3(&mut self, opc: Opcode, vece: u32, r: TempVec, a: TempVec, b: TempVec) {
        let ty = self.vec_type(r);
        assert!(self.vec_type(a) >= ty && self.vec_type(b) >= ty, "input vector too small");
        self.vec_gen(opc, ty, vece, &[r.arg(), a.arg(), b.arg()]);
    }

    fn vec_shifti(&mut self, opc: Opcode, vece: u32, r: TempVec, a: TempVec, i: i64) {
        let ty = self.vec_type(r);
        assert_eq!(self.vec_type(a), ty, "shift input has a different type");
        assert!(i >= 0 && i < (8i64 << vece), "shift count out of range");
        if i == 0 {
            self.gen_mov_vec(r, a);
        } else {
            self.vec_gen(opc, ty, vece, &[r.arg(), a.arg(), i as u64]);
        }
    }

    /// `tcg_gen_mov_vec`.
    pub fn gen_mov_vec(&mut self, r: TempVec, a: TempVec) {
        if r != a {
            self.vec_gen_op2(Opcode::MovVec, 0, r, a);
        }
    }

    /// `tcg_gen_dupi_vec`.
    pub fn gen_dupi_vec(&mut self, vece: u32, r: TempVec, a: u64) {
        let ty = self.vec_type(r);
        let c = self.constant_vec(ty, vece, a as i64);
        self.gen_mov_vec(r, c);
    }

    /// `tcg_gen_dup_i64_vec`.
    pub fn gen_dup_i64_vec(&mut self, vece: u32, r: TempVec, a: TempI64) {
        let ty = self.vec_type(r);
        self.vec_gen(Opcode::DupVec, ty, vece, &[r.arg(), a.arg()]);
    }

    /// `tcg_gen_dup_i32_vec`.
    pub fn gen_dup_i32_vec(&mut self, vece: u32, r: TempVec, a: TempI32) {
        let ty = self.vec_type(r);
        self.vec_gen(Opcode::DupVec, ty, vece, &[r.arg(), a.arg()]);
    }

    /// `tcg_gen_dup_mem_vec`.
    pub fn gen_dup_mem_vec(&mut self, vece: u32, r: TempVec, b: TempPtr, ofs: i64) {
        let ty = self.vec_type(r);
        self.vec_gen(Opcode::DupmVec, ty, vece, &[r.arg(), b.arg(), ofs as u64]);
    }

    /// `tcg_gen_ld_vec`.
    pub fn gen_ld_vec(&mut self, r: TempVec, b: TempPtr, ofs: i64) {
        let ty = self.vec_type(r);
        self.vec_gen(Opcode::LdVec, ty, 0, &[r.arg(), b.arg(), ofs as u64]);
    }

    /// `tcg_gen_st_vec`.
    pub fn gen_st_vec(&mut self, r: TempVec, b: TempPtr, ofs: i64) {
        let ty = self.vec_type(r);
        self.vec_gen(Opcode::StVec, ty, 0, &[r.arg(), b.arg(), ofs as u64]);
    }

    /// `tcg_gen_stl_vec`: store only the low `low_type` part.
    pub fn gen_stl_vec(&mut self, r: TempVec, b: TempPtr, ofs: i64, low_type: Type) {
        let ty = self.vec_type(r);
        assert!(low_type >= Type::V64 && low_type <= ty, "bad low type");
        self.vec_gen(Opcode::StVec, low_type, 0, &[r.arg(), b.arg(), ofs as u64]);
    }

    vec_logic3! {
        /// `tcg_gen_and_vec`.
        gen_and_vec => AndVec;
        /// `tcg_gen_or_vec`.
        gen_or_vec => OrVec;
        /// `tcg_gen_xor_vec`.
        gen_xor_vec => XorVec;
        /// `tcg_gen_andc_vec`.
        gen_andc_vec => AndcVec;
        /// `tcg_gen_orc_vec`.
        gen_orc_vec => OrcVec;
        /// `tcg_gen_nand_vec`.
        gen_nand_vec => NandVec;
        /// `tcg_gen_nor_vec`.
        gen_nor_vec => NorVec;
        /// `tcg_gen_eqv_vec`.
        gen_eqv_vec => EqvVec;
    }

    vec_op3! {
        /// `tcg_gen_add_vec`.
        gen_add_vec => AddVec;
        /// `tcg_gen_sub_vec`.
        gen_sub_vec => SubVec;
        /// `tcg_gen_mul_vec`.
        gen_mul_vec => MulVec;
        /// `tcg_gen_ssadd_vec`.
        gen_ssadd_vec => SsaddVec;
        /// `tcg_gen_usadd_vec`.
        gen_usadd_vec => UsaddVec;
        /// `tcg_gen_sssub_vec`.
        gen_sssub_vec => SssubVec;
        /// `tcg_gen_ussub_vec`.
        gen_ussub_vec => UssubVec;
        /// `tcg_gen_smin_vec`.
        gen_smin_vec => SminVec;
        /// `tcg_gen_umin_vec`.
        gen_umin_vec => UminVec;
        /// `tcg_gen_smax_vec`.
        gen_smax_vec => SmaxVec;
        /// `tcg_gen_umax_vec`.
        gen_umax_vec => UmaxVec;
        /// `tcg_gen_shlv_vec`.
        gen_shlv_vec => ShlvVec;
        /// `tcg_gen_shrv_vec`.
        gen_shrv_vec => ShrvVec;
        /// `tcg_gen_sarv_vec`.
        gen_sarv_vec => SarvVec;
        /// `tcg_gen_rotlv_vec`.
        gen_rotlv_vec => RotlvVec;
        /// `tcg_gen_rotrv_vec`.
        gen_rotrv_vec => RotrvVec;
    }

    vec_shifti! {
        /// `tcg_gen_shli_vec`.
        gen_shli_vec => ShliVec;
        /// `tcg_gen_shri_vec`.
        gen_shri_vec => ShriVec;
        /// `tcg_gen_sari_vec`.
        gen_sari_vec => SariVec;
        /// `tcg_gen_rotli_vec`.
        gen_rotli_vec => RotliVec;
    }

    vec_shifts! {
        /// `tcg_gen_shls_vec`.
        gen_shls_vec => ShlsVec;
        /// `tcg_gen_shrs_vec`.
        gen_shrs_vec => ShrsVec;
        /// `tcg_gen_sars_vec`.
        gen_sars_vec => SarsVec;
        /// `tcg_gen_rotls_vec`.
        gen_rotls_vec => RotlsVec;
    }

    /// `tcg_gen_rotri_vec`, emitted as a left rotate.
    pub fn gen_rotri_vec(&mut self, vece: u32, r: TempVec, a: TempVec, i: i64) {
        let bits = 8i64 << vece;
        assert!(i >= 0 && i < bits, "rotate count out of range");
        self.vec_shifti(Opcode::RotliVec, vece, r, a, i.wrapping_neg() & (bits - 1));
    }

    /// `tcg_gen_not_vec`.
    pub fn gen_not_vec(&mut self, _vece: u32, r: TempVec, a: TempVec) {
        self.vec_gen_op2(Opcode::NotVec, 0, r, a);
    }

    /// `tcg_gen_neg_vec`.
    pub fn gen_neg_vec(&mut self, vece: u32, r: TempVec, a: TempVec) {
        self.vec_gen_op2(Opcode::NegVec, vece, r, a);
    }

    /// `tcg_gen_abs_vec`.
    pub fn gen_abs_vec(&mut self, vece: u32, r: TempVec, a: TempVec) {
        self.vec_gen_op2(Opcode::AbsVec, vece, r, a);
    }

    /// `tcg_gen_cmp_vec`: each element becomes all ones if the condition holds, else zero.
    pub fn gen_cmp_vec(&mut self, cond: Cond, vece: u32, r: TempVec, a: TempVec, b: TempVec) {
        let ty = self.vec_type(r);
        self.vec_gen(Opcode::CmpVec, ty, vece, &[r.arg(), a.arg(), b.arg(), cond as u64]);
    }

    /// `tcg_gen_bitsel_vec`: `r = (a & b) | (~a & c)`.
    pub fn gen_bitsel_vec(&mut self, _vece: u32, r: TempVec, a: TempVec, b: TempVec, c: TempVec) {
        let ty = self.vec_type(r);
        self.vec_gen(Opcode::BitselVec, ty, 0, &[r.arg(), a.arg(), b.arg(), c.arg()]);
    }

    /// `tcg_gen_cmpsel_vec`: `r = cond(a, b) ? c : d` per element.
    #[allow(clippy::too_many_arguments)]
    pub fn gen_cmpsel_vec(
        &mut self,
        cond: Cond,
        vece: u32,
        r: TempVec,
        a: TempVec,
        b: TempVec,
        c: TempVec,
        d: TempVec,
    ) {
        let ty = self.vec_type(r);
        self.vec_gen(
            Opcode::CmpselVec,
            ty,
            vece,
            &[r.arg(), a.arg(), b.arg(), c.arg(), d.arg(), cond as u64],
        );
    }
}
