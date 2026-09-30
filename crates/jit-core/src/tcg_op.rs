// SPDX-License-Identifier: MIT OR Apache-2.0

//! The op builder, a port of `tcg/tcg-op.c` and the inline parts of `include/tcg/tcg-op.h`.
//!
//! The method names follow QEMU with the `tcg_` prefix dropped, so `tcg_gen_addi_i32` is
//! [`Func::gen_addi_i32`]. Each method emits exactly the ops QEMU emits on a host whose backend
//! supports every optional opcode, including `extract`, `sextract` and `deposit` for every field
//! and the carry ops. The fallback expansions QEMU uses on smaller hosts are not ported because
//! they are never taken on such a host. This matters for dumps: QEMU's `-d op` output depends on
//! the host, and the output here matches a host like aarch64 or x86_64.
//!
//! The 32-bit and 64-bit variants share one implementation that takes the type as a parameter,
//! just as QEMU 11 shares one opcode for both widths.

use crate::ir::{Func, HelperId, Label, OpId, Temp, TempI32, TempI64, TempI128, TempPtr};
use crate::opcode::Opcode;
use crate::types::{Cond, INSN_START_WORDS, PluginFrom, Type, bswap, tb_exit};

/// Sign extend a 32-bit value the way QEMU stores `TCG_TYPE_I32` constants.
pub(crate) const fn norm(ty: Type, v: i64) -> i64 {
    match ty {
        Type::I32 => v as i32 as i64,
        _ => v,
    }
}

/// The mask of the bits that matter in a value of this integer type.
pub(crate) const fn type_mask(ty: Type) -> u64 {
    match ty {
        Type::I32 => 0xffff_ffff,
        _ => u64::MAX,
    }
}

macro_rules! wrap_unary {
    ($($(#[$m:meta])* $n32:ident, $n64:ident => $g:ident;)*) => {$(
        $(#[$m])*
        pub fn $n32(&mut self, ret: TempI32, a: TempI32) {
            self.$g(Type::I32, ret.0, a.0)
        }
        $(#[$m])*
        pub fn $n64(&mut self, ret: TempI64, a: TempI64) {
            self.$g(Type::I64, ret.0, a.0)
        }
    )*};
}

macro_rules! wrap_binop {
    ($($(#[$m:meta])* $n32:ident, $n64:ident => $opc:ident;)*) => {$(
        $(#[$m])*
        pub fn $n32(&mut self, ret: TempI32, a: TempI32, b: TempI32) {
            self.op3_t(Opcode::$opc, Type::I32, ret.0, a.0, b.0)
        }
        $(#[$m])*
        pub fn $n64(&mut self, ret: TempI64, a: TempI64, b: TempI64) {
            self.op3_t(Opcode::$opc, Type::I64, ret.0, a.0, b.0)
        }
    )*};
}

macro_rules! wrap_binfn {
    ($($(#[$m:meta])* $n32:ident, $n64:ident => $g:ident;)*) => {$(
        $(#[$m])*
        pub fn $n32(&mut self, ret: TempI32, a: TempI32, b: TempI32) {
            self.$g(Type::I32, ret.0, a.0, b.0)
        }
        $(#[$m])*
        pub fn $n64(&mut self, ret: TempI64, a: TempI64, b: TempI64) {
            self.$g(Type::I64, ret.0, a.0, b.0)
        }
    )*};
}

macro_rules! wrap_imm {
    ($($(#[$m:meta])* $n32:ident, $n64:ident => $g:ident;)*) => {$(
        $(#[$m])*
        pub fn $n32(&mut self, ret: TempI32, a: TempI32, imm: i32) {
            self.$g(Type::I32, ret.0, a.0, imm as i64)
        }
        $(#[$m])*
        pub fn $n64(&mut self, ret: TempI64, a: TempI64, imm: i64) {
            self.$g(Type::I64, ret.0, a.0, imm)
        }
    )*};
}

macro_rules! wrap_cond {
    ($($(#[$m:meta])* $n32:ident, $n64:ident, $ni32:ident, $ni64:ident => $g:ident, $gi:ident;)*) => {$(
        $(#[$m])*
        pub fn $n32(&mut self, cond: Cond, ret: TempI32, a: TempI32, b: TempI32) {
            self.$g(Type::I32, cond, ret.0, a.0, b.0)
        }
        $(#[$m])*
        pub fn $n64(&mut self, cond: Cond, ret: TempI64, a: TempI64, b: TempI64) {
            self.$g(Type::I64, cond, ret.0, a.0, b.0)
        }
        $(#[$m])*
        pub fn $ni32(&mut self, cond: Cond, ret: TempI32, a: TempI32, b: i32) {
            self.$gi(Type::I32, cond, ret.0, a.0, b as i64)
        }
        $(#[$m])*
        pub fn $ni64(&mut self, cond: Cond, ret: TempI64, a: TempI64, b: i64) {
            self.$gi(Type::I64, cond, ret.0, a.0, b)
        }
    )*};
}

macro_rules! wrap_ld {
    ($($(#[$m:meta])* $name:ident, $ty:ident, $tt:ident => $opc:ident;)*) => {$(
        $(#[$m])*
        pub fn $name(&mut self, ret: $tt, base: TempPtr, offset: i64) {
            self.emit_op(Opcode::$opc, Type::$ty, &[ret.arg(), base.arg(), offset as u64]);
        }
    )*};
}

impl Func {
    /// The constant temp of this type, with I32 values sign extended.
    pub(crate) fn cst(&mut self, ty: Type, v: i64) -> Temp {
        self.constant_internal(ty, norm(ty, v))
    }

    pub(crate) fn op2_t(&mut self, opc: Opcode, ty: Type, a: Temp, b: Temp) {
        self.emit_op(opc, ty, &[a.arg(), b.arg()]);
    }

    pub(crate) fn op3_t(&mut self, opc: Opcode, ty: Type, a: Temp, b: Temp, c: Temp) {
        self.emit_op(opc, ty, &[a.arg(), b.arg(), c.arg()]);
    }

    // Generic implementations.

    pub(crate) fn mov_t(&mut self, ty: Type, r: Temp, a: Temp) {
        if r != a {
            self.op2_t(Opcode::Mov, ty, r, a);
        }
    }

    pub(crate) fn movi_t(&mut self, ty: Type, r: Temp, v: i64) {
        let c = self.cst(ty, v);
        self.mov_t(ty, r, c);
    }

    fn discard_t(&mut self, ty: Type, a: Temp) {
        self.emit_op(Opcode::Discard, ty, &[a.arg()]);
    }

    fn neg_t(&mut self, ty: Type, r: Temp, a: Temp) {
        self.op2_t(Opcode::Neg, ty, r, a);
    }

    fn not_t(&mut self, ty: Type, r: Temp, a: Temp) {
        self.op2_t(Opcode::Not, ty, r, a);
    }

    fn ctpop_t(&mut self, ty: Type, r: Temp, a: Temp) {
        self.op2_t(Opcode::Ctpop, ty, r, a);
    }

    pub(crate) fn addi_t(&mut self, ty: Type, r: Temp, a: Temp, v: i64) {
        if norm(ty, v) == 0 {
            self.mov_t(ty, r, a);
        } else {
            let c = self.cst(ty, v);
            self.op3_t(Opcode::Add, ty, r, a, c);
        }
    }

    fn subi_t(&mut self, ty: Type, r: Temp, a: Temp, v: i64) {
        self.addi_t(ty, r, a, v.wrapping_neg());
    }

    fn subfi_t(&mut self, ty: Type, r: Temp, v: i64, a: Temp) {
        if norm(ty, v) == 0 {
            self.neg_t(ty, r, a);
        } else {
            let c = self.cst(ty, v);
            self.op3_t(Opcode::Sub, ty, r, c, a);
        }
    }

    pub(crate) fn andi_t(&mut self, ty: Type, r: Temp, a: Temp, v: i64) {
        let v = norm(ty, v);
        match v {
            0 => return self.movi_t(ty, r, 0),
            -1 => return self.mov_t(ty, r, a),
            _ => {}
        }
        let m = v as u64 & type_mask(ty);
        if m & m.wrapping_add(1) == 0 {
            let len = (!m & type_mask(ty)).trailing_zeros();
            return self.extract_t(ty, r, a, 0, len);
        }
        let c = self.cst(ty, v);
        self.op3_t(Opcode::And, ty, r, a, c);
    }

    fn ori_t(&mut self, ty: Type, r: Temp, a: Temp, v: i64) {
        let v = norm(ty, v);
        if v == -1 {
            self.movi_t(ty, r, -1);
        } else if v == 0 {
            self.mov_t(ty, r, a);
        } else {
            let c = self.cst(ty, v);
            self.op3_t(Opcode::Or, ty, r, a, c);
        }
    }

    fn xori_t(&mut self, ty: Type, r: Temp, a: Temp, v: i64) {
        let v = norm(ty, v);
        if v == 0 {
            self.mov_t(ty, r, a);
        } else if v == -1 {
            self.op2_t(Opcode::Not, ty, r, a);
        } else {
            let c = self.cst(ty, v);
            self.op3_t(Opcode::Xor, ty, r, a, c);
        }
    }

    fn shifti_t(&mut self, opc: Opcode, ty: Type, r: Temp, a: Temp, v: i64) {
        assert!(v >= 0 && v < ty.bits() as i64, "shift count out of range");
        if v == 0 {
            self.mov_t(ty, r, a);
        } else {
            let c = self.cst(ty, v);
            self.op3_t(opc, ty, r, a, c);
        }
    }

    pub(crate) fn shli_t(&mut self, ty: Type, r: Temp, a: Temp, v: i64) {
        self.shifti_t(Opcode::Shl, ty, r, a, v);
    }

    pub(crate) fn shri_t(&mut self, ty: Type, r: Temp, a: Temp, v: i64) {
        self.shifti_t(Opcode::Shr, ty, r, a, v);
    }

    pub(crate) fn sari_t(&mut self, ty: Type, r: Temp, a: Temp, v: i64) {
        self.shifti_t(Opcode::Sar, ty, r, a, v);
    }

    fn rotli_t(&mut self, ty: Type, r: Temp, a: Temp, v: i64) {
        self.shifti_t(Opcode::Rotl, ty, r, a, v);
    }

    fn rotri_t(&mut self, ty: Type, r: Temp, a: Temp, v: i64) {
        self.shifti_t(Opcode::Rotr, ty, r, a, v);
    }

    fn muli_t(&mut self, ty: Type, r: Temp, a: Temp, v: i64) {
        let v = norm(ty, v);
        if v == 0 {
            self.movi_t(ty, r, 0);
        } else if (v as u64).is_power_of_two() {
            self.shli_t(ty, r, a, (v as u64).trailing_zeros() as i64);
        } else {
            let c = self.cst(ty, v);
            self.op3_t(Opcode::Mul, ty, r, a, c);
        }
    }

    fn clzi_t(&mut self, ty: Type, r: Temp, a: Temp, v: i64) {
        let c = self.cst(ty, v);
        self.op3_t(Opcode::Clz, ty, r, a, c);
    }

    fn ctzi_t(&mut self, ty: Type, r: Temp, a: Temp, v: i64) {
        let c = self.cst(ty, v);
        self.op3_t(Opcode::Ctz, ty, r, a, c);
    }

    fn clrsb_t(&mut self, ty: Type, r: Temp, a: Temp) {
        let t = self.temp_new_internal(ty, crate::types::TempKind::Ebb);
        let bits = ty.bits() as i64;
        self.sari_t(ty, t, a, bits - 1);
        self.op3_t(Opcode::Xor, ty, t, t, a);
        self.clzi_t(ty, t, t, bits);
        self.subi_t(ty, r, t, 1);
        self.temp_free(t);
    }

    fn check_field(ty: Type, ofs: u32, len: u32) {
        let bits = ty.bits();
        assert!(ofs < bits && len > 0 && len <= bits && ofs + len <= bits, "bad bit field");
    }

    fn deposit_t(&mut self, ty: Type, r: Temp, a: Temp, b: Temp, ofs: u32, len: u32) {
        Self::check_field(ty, ofs, len);
        if len == ty.bits() {
            self.mov_t(ty, r, b);
        } else {
            self.emit_op(Opcode::Deposit, ty, &[r.arg(), a.arg(), b.arg(), ofs as u64, len as u64]);
        }
    }

    fn deposit_z_t(&mut self, ty: Type, r: Temp, a: Temp, ofs: u32, len: u32) {
        Self::check_field(ty, ofs, len);
        if ofs + len == ty.bits() {
            self.shli_t(ty, r, a, ofs as i64);
        } else if ofs == 0 {
            if ty == Type::I32 {
                self.extract_t(ty, r, a, 0, len);
            } else {
                self.andi_t(ty, r, a, ((1u64 << len) - 1) as i64);
            }
        } else {
            let zero = self.cst(ty, 0);
            self.emit_op(
                Opcode::Deposit,
                ty,
                &[r.arg(), zero.arg(), a.arg(), ofs as u64, len as u64],
            );
        }
    }

    pub(crate) fn extract_t(&mut self, ty: Type, r: Temp, a: Temp, ofs: u32, len: u32) {
        Self::check_field(ty, ofs, len);
        if ofs + len == ty.bits() {
            self.shri_t(ty, r, a, (ty.bits() - len) as i64);
        } else {
            self.emit_op(Opcode::Extract, ty, &[r.arg(), a.arg(), ofs as u64, len as u64]);
        }
    }

    pub(crate) fn sextract_t(&mut self, ty: Type, r: Temp, a: Temp, ofs: u32, len: u32) {
        Self::check_field(ty, ofs, len);
        if ofs + len == ty.bits() {
            self.sari_t(ty, r, a, (ty.bits() - len) as i64);
        } else {
            self.emit_op(Opcode::Sextract, ty, &[r.arg(), a.arg(), ofs as u64, len as u64]);
        }
    }

    fn extract2_t(&mut self, ty: Type, r: Temp, al: Temp, ah: Temp, ofs: u32) {
        assert!(ofs <= ty.bits(), "bad extract2 offset");
        if ofs == 0 {
            self.mov_t(ty, r, al);
        } else if ofs == ty.bits() {
            self.mov_t(ty, r, ah);
        } else if al == ah {
            self.rotri_t(ty, r, al, ofs as i64);
        } else {
            self.emit_op(Opcode::Extract2, ty, &[r.arg(), al.arg(), ah.arg(), ofs as u64]);
        }
    }

    fn brcond_t(&mut self, ty: Type, cond: Cond, a: Temp, b: Temp, l: Label) {
        match cond {
            Cond::Always => self.gen_br(l),
            Cond::Never => {}
            _ => {
                let op =
                    self.emit_op(Opcode::Brcond, ty, &[a.arg(), b.arg(), cond as u64, l.arg()]);
                self.add_label_use(l, op);
            }
        }
    }

    fn setcond_t(&mut self, ty: Type, cond: Cond, r: Temp, a: Temp, b: Temp) {
        match cond {
            Cond::Always => self.movi_t(ty, r, 1),
            Cond::Never => self.movi_t(ty, r, 0),
            _ => {
                self.emit_op(Opcode::Setcond, ty, &[r.arg(), a.arg(), b.arg(), cond as u64]);
            }
        }
    }

    fn setcondi_t(&mut self, ty: Type, cond: Cond, r: Temp, a: Temp, v: i64) {
        let c = self.cst(ty, v);
        self.setcond_t(ty, cond, r, a, c);
    }

    fn negsetcond_t(&mut self, ty: Type, cond: Cond, r: Temp, a: Temp, b: Temp) {
        match cond {
            Cond::Always => self.movi_t(ty, r, -1),
            Cond::Never => self.movi_t(ty, r, 0),
            _ => {
                self.emit_op(Opcode::Negsetcond, ty, &[r.arg(), a.arg(), b.arg(), cond as u64]);
            }
        }
    }

    fn negsetcondi_t(&mut self, ty: Type, cond: Cond, r: Temp, a: Temp, v: i64) {
        let c = self.cst(ty, v);
        self.negsetcond_t(ty, cond, r, a, c);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn movcond_t(
        &mut self,
        ty: Type,
        cond: Cond,
        r: Temp,
        c1: Temp,
        c2: Temp,
        v1: Temp,
        v2: Temp,
    ) {
        match cond {
            Cond::Always => self.mov_t(ty, r, v1),
            Cond::Never => self.mov_t(ty, r, v2),
            _ => {
                self.emit_op(
                    Opcode::Movcond,
                    ty,
                    &[r.arg(), c1.arg(), c2.arg(), v1.arg(), v2.arg(), cond as u64],
                );
            }
        }
    }

    fn smin_t(&mut self, ty: Type, r: Temp, a: Temp, b: Temp) {
        self.movcond_t(ty, Cond::Lt, r, a, b, a, b);
    }

    fn umin_t(&mut self, ty: Type, r: Temp, a: Temp, b: Temp) {
        self.movcond_t(ty, Cond::Ltu, r, a, b, a, b);
    }

    fn smax_t(&mut self, ty: Type, r: Temp, a: Temp, b: Temp) {
        self.movcond_t(ty, Cond::Lt, r, a, b, b, a);
    }

    fn umax_t(&mut self, ty: Type, r: Temp, a: Temp, b: Temp) {
        self.movcond_t(ty, Cond::Ltu, r, a, b, b, a);
    }

    fn abs_t(&mut self, ty: Type, r: Temp, a: Temp) {
        let t = self.temp_new_internal(ty, crate::types::TempKind::Ebb);
        self.sari_t(ty, t, a, ty.bits() as i64 - 1);
        self.op3_t(Opcode::Xor, ty, r, a, t);
        self.op3_t(Opcode::Sub, ty, r, r, t);
        self.temp_free(t);
    }

    #[allow(clippy::too_many_arguments)]
    fn add2_t(&mut self, ty: Type, rl: Temp, rh: Temp, al: Temp, ah: Temp, bl: Temp, bh: Temp) {
        let t0 = self.temp_new_internal(ty, crate::types::TempKind::Ebb);
        self.op3_t(Opcode::Addco, ty, t0, al, bl);
        self.op3_t(Opcode::Addci, ty, rh, ah, bh);
        self.mov_t(ty, rl, t0);
        self.temp_free(t0);
    }

    #[allow(clippy::too_many_arguments)]
    fn sub2_t(&mut self, ty: Type, rl: Temp, rh: Temp, al: Temp, ah: Temp, bl: Temp, bh: Temp) {
        let t0 = self.temp_new_internal(ty, crate::types::TempKind::Ebb);
        self.op3_t(Opcode::Subbo, ty, t0, al, bl);
        self.op3_t(Opcode::Subbi, ty, rh, ah, bh);
        self.mov_t(ty, rl, t0);
        self.temp_free(t0);
    }

    fn addcio_t(&mut self, ty: Type, r: Temp, co: Temp, a: Temp, b: Temp, ci: Temp) {
        let t0 = self.temp_new_internal(ty, crate::types::TempKind::Ebb);
        let zero = self.cst(ty, 0);
        let mone = self.cst(ty, -1);
        self.op3_t(Opcode::Addco, ty, t0, ci, mone);
        self.op3_t(Opcode::Addcio, ty, r, a, b);
        self.op3_t(Opcode::Addci, ty, co, zero, zero);
        self.temp_free(t0);
    }

    fn mul2_t(&mut self, opc: Opcode, ty: Type, rl: Temp, rh: Temp, a: Temp, b: Temp) {
        self.emit_op(opc, ty, &[rl.arg(), rh.arg(), a.arg(), b.arg()]);
    }

    fn bswap_t(&mut self, opc: Opcode, ty: Type, r: Temp, a: Temp, flags: u32) {
        assert!(
            flags & bswap::OS == 0 || flags & bswap::OZ == 0,
            "only one extension flag may be present"
        );
        self.emit_op(opc, ty, &[r.arg(), a.arg(), flags as u64]);
    }

    fn ext8s_t(&mut self, ty: Type, r: Temp, a: Temp) {
        self.sextract_t(ty, r, a, 0, 8);
    }

    fn ext16s_t(&mut self, ty: Type, r: Temp, a: Temp) {
        self.sextract_t(ty, r, a, 0, 16);
    }

    fn ext8u_t(&mut self, ty: Type, r: Temp, a: Temp) {
        self.extract_t(ty, r, a, 0, 8);
    }

    fn ext16u_t(&mut self, ty: Type, r: Temp, a: Temp) {
        self.extract_t(ty, r, a, 0, 16);
    }

    fn hswap_i32_t(&mut self, _ty: Type, r: Temp, a: Temp) {
        self.rotli_t(Type::I32, r, a, 16);
    }

    // Public API: control flow and markers.

    /// `gen_set_label`.
    pub fn gen_set_label(&mut self, l: Label) {
        self.labels[l.0 as usize].present = true;
        self.emit_op(Opcode::SetLabel, Type::I32, &[l.arg()]);
    }

    /// `tcg_gen_br`.
    pub fn gen_br(&mut self, l: Label) {
        let op = self.emit_op(Opcode::Br, Type::I32, &[l.arg()]);
        self.add_label_use(l, op);
    }

    /// `tcg_gen_mb`. In user mode the barrier is only emitted for a parallel block.
    pub fn gen_mb(&mut self, mb_type: u32) {
        let parallel = if self.config.user_only { self.config.parallel } else { true };
        if parallel {
            self.emit_op(Opcode::Mb, Type::I32, &[mb_type as u64]);
        }
    }

    /// `tcg_gen_plugin_cb`. The op is a marker; nothing is instrumented.
    pub fn gen_plugin_cb(&mut self, from: PluginFrom) {
        self.emit_op(Opcode::PluginCb, Type::I32, &[from as u64]);
    }

    /// `tcg_gen_plugin_mem_cb`. The op is a marker; nothing is instrumented.
    pub fn gen_plugin_mem_cb(&mut self, addr: TempI64, meminfo: u32) {
        self.emit_op(Opcode::PluginMemCb, Type::I32, &[addr.arg(), meminfo as u64]);
    }

    /// `tcg_gen_insn_start`. Missing words are zero.
    pub fn gen_insn_start(&mut self, words: &[u64]) -> OpId {
        assert!(words.len() <= INSN_START_WORDS, "too many insn_start words");
        let mut args = [0u64; INSN_START_WORDS];
        args[..words.len()].copy_from_slice(words);
        let op = self.emit_op(Opcode::InsnStart, Type::I32, &args);
        self.last_insn_start = Some(op);
        self.num_insns += 1;
        op
    }

    /// `tcg_gen_exit_tb`. `tb` stands for the translation block pointer; zero means NULL. The
    /// op carries `tb + idx`, which is also what the interpreter returns.
    pub fn gen_exit_tb(&mut self, tb: u64, idx: u64) {
        if tb == 0 {
            assert_eq!(idx, 0, "exit_tb without a block must use index 0");
        } else {
            assert!(idx <= tb_exit::IDXMAX || idx == tb_exit::REQUESTED, "bad exit_tb index");
        }
        self.emit_op(Opcode::ExitTb, Type::I32, &[tb.wrapping_add(idx)]);
    }

    /// `tcg_gen_goto_tb`.
    pub fn gen_goto_tb(&mut self, idx: u64) {
        assert!(idx <= tb_exit::IDXMAX, "only two chained exits are supported");
        self.emit_op(Opcode::GotoTb, Type::I32, &[idx]);
    }

    /// `tcg_gen_lookup_and_goto_ptr`.
    pub fn gen_lookup_and_goto_ptr(&mut self) {
        if self.config.no_goto_ptr {
            self.gen_exit_tb(0, 0);
            return;
        }
        let ptr = self.temp_ebb_new_ptr();
        let h = self.helper(crate::helpers::lookup_tb_ptr());
        let env = self.env();
        self.gen_call(h, Some(ptr.0), &[env.0]);
        self.emit_op(Opcode::GotoPtr, Type::PTR, &[ptr.arg()]);
        self.temp_free(ptr);
    }

    /// `tcg_gen_callN`: call a helper. `ret` must be given exactly when the helper returns a
    /// value; an I128 value names the low half of its pair.
    pub fn gen_call(&mut self, helper: HelperId, ret: Option<Temp>, args: &[Temp]) -> OpId {
        let info = self.helper_info(helper).clone();
        assert_eq!(info.args.len(), args.len(), "wrong number of arguments to {}", info.name);
        let mut v: Vec<u64> = Vec::with_capacity(info.nr_out() + info.nr_in() + 2);
        match (info.nr_out(), ret) {
            (0, None) => {}
            (1, Some(r)) => v.push(r.arg()),
            (2, Some(r)) => {
                assert_eq!(self.temp(r).subindex, 0, "an I128 result must name the low half");
                v.push(r.arg());
                v.push(r.arg() + 1);
            }
            _ => panic!("result does not match the declaration of {}", info.name),
        }
        for (a, t) in args.iter().zip(&info.args) {
            v.push(a.arg());
            if t.slots() == 2 {
                v.push(a.arg() + 1);
            }
        }
        v.push(helper.0 as u64);
        v.push(helper.0 as u64);
        let op = self.emit_op(Opcode::Call, Type::I32, &v);
        let o = self.op_mut(op);
        o.callo = info.nr_out() as u8;
        o.calli = info.nr_in() as u8;
        op
    }

    // Public API: typed integer ops.

    /// `tcg_gen_discard_i32`.
    pub fn gen_discard_i32(&mut self, a: TempI32) {
        self.discard_t(Type::I32, a.0);
    }

    /// `tcg_gen_discard_i64`.
    pub fn gen_discard_i64(&mut self, a: TempI64) {
        self.discard_t(Type::I64, a.0);
    }

    /// `tcg_gen_movi_i32`.
    pub fn gen_movi_i32(&mut self, ret: TempI32, v: i32) {
        self.movi_t(Type::I32, ret.0, v as i64);
    }

    /// `tcg_gen_movi_i64`.
    pub fn gen_movi_i64(&mut self, ret: TempI64, v: i64) {
        self.movi_t(Type::I64, ret.0, v);
    }

    /// `tcg_gen_subfi_i32`.
    pub fn gen_subfi_i32(&mut self, ret: TempI32, v: i32, a: TempI32) {
        self.subfi_t(Type::I32, ret.0, v as i64, a.0);
    }

    /// `tcg_gen_subfi_i64`.
    pub fn gen_subfi_i64(&mut self, ret: TempI64, v: i64, a: TempI64) {
        self.subfi_t(Type::I64, ret.0, v, a.0);
    }

    wrap_unary! {
        /// `tcg_gen_mov`.
        gen_mov_i32, gen_mov_i64 => mov_t;
        /// `tcg_gen_neg`.
        gen_neg_i32, gen_neg_i64 => neg_t;
        /// `tcg_gen_not`.
        gen_not_i32, gen_not_i64 => not_t;
        /// `tcg_gen_ctpop`.
        gen_ctpop_i32, gen_ctpop_i64 => ctpop_t;
        /// `tcg_gen_clrsb`.
        gen_clrsb_i32, gen_clrsb_i64 => clrsb_t;
        /// `tcg_gen_abs`.
        gen_abs_i32, gen_abs_i64 => abs_t;
        /// `tcg_gen_ext8s`.
        gen_ext8s_i32, gen_ext8s_i64 => ext8s_t;
        /// `tcg_gen_ext16s`.
        gen_ext16s_i32, gen_ext16s_i64 => ext16s_t;
        /// `tcg_gen_ext8u`.
        gen_ext8u_i32, gen_ext8u_i64 => ext8u_t;
        /// `tcg_gen_ext16u`.
        gen_ext16u_i32, gen_ext16u_i64 => ext16u_t;
    }

    wrap_binop! {
        /// `tcg_gen_add`.
        gen_add_i32, gen_add_i64 => Add;
        /// `tcg_gen_sub`.
        gen_sub_i32, gen_sub_i64 => Sub;
        /// `tcg_gen_and`.
        gen_and_i32, gen_and_i64 => And;
        /// `tcg_gen_or`.
        gen_or_i32, gen_or_i64 => Or;
        /// `tcg_gen_xor`.
        gen_xor_i32, gen_xor_i64 => Xor;
        /// `tcg_gen_andc`.
        gen_andc_i32, gen_andc_i64 => Andc;
        /// `tcg_gen_eqv`.
        gen_eqv_i32, gen_eqv_i64 => Eqv;
        /// `tcg_gen_nand`.
        gen_nand_i32, gen_nand_i64 => Nand;
        /// `tcg_gen_nor`.
        gen_nor_i32, gen_nor_i64 => Nor;
        /// `tcg_gen_orc`.
        gen_orc_i32, gen_orc_i64 => Orc;
        /// `tcg_gen_shl`.
        gen_shl_i32, gen_shl_i64 => Shl;
        /// `tcg_gen_shr`.
        gen_shr_i32, gen_shr_i64 => Shr;
        /// `tcg_gen_sar`.
        gen_sar_i32, gen_sar_i64 => Sar;
        /// `tcg_gen_rotl`.
        gen_rotl_i32, gen_rotl_i64 => Rotl;
        /// `tcg_gen_rotr`.
        gen_rotr_i32, gen_rotr_i64 => Rotr;
        /// `tcg_gen_mul`.
        gen_mul_i32, gen_mul_i64 => Mul;
        /// `tcg_gen_div`.
        gen_div_i32, gen_div_i64 => Divs;
        /// `tcg_gen_rem`.
        gen_rem_i32, gen_rem_i64 => Rems;
        /// `tcg_gen_divu`.
        gen_divu_i32, gen_divu_i64 => Divu;
        /// `tcg_gen_remu`.
        gen_remu_i32, gen_remu_i64 => Remu;
        /// The `mulsh` op: the high half of a signed product.
        gen_mulsh_i32, gen_mulsh_i64 => Mulsh;
        /// The `muluh` op: the high half of an unsigned product.
        gen_muluh_i32, gen_muluh_i64 => Muluh;
        /// `tcg_gen_clz`.
        gen_clz_i32, gen_clz_i64 => Clz;
        /// `tcg_gen_ctz`.
        gen_ctz_i32, gen_ctz_i64 => Ctz;
    }

    wrap_binfn! {
        /// `tcg_gen_smin`.
        gen_smin_i32, gen_smin_i64 => smin_t;
        /// `tcg_gen_umin`.
        gen_umin_i32, gen_umin_i64 => umin_t;
        /// `tcg_gen_smax`.
        gen_smax_i32, gen_smax_i64 => smax_t;
        /// `tcg_gen_umax`.
        gen_umax_i32, gen_umax_i64 => umax_t;
    }

    wrap_imm! {
        /// `tcg_gen_addi`.
        gen_addi_i32, gen_addi_i64 => addi_t;
        /// `tcg_gen_subi`.
        gen_subi_i32, gen_subi_i64 => subi_t;
        /// `tcg_gen_andi`.
        gen_andi_i32, gen_andi_i64 => andi_t;
        /// `tcg_gen_ori`.
        gen_ori_i32, gen_ori_i64 => ori_t;
        /// `tcg_gen_xori`.
        gen_xori_i32, gen_xori_i64 => xori_t;
        /// `tcg_gen_shli`.
        gen_shli_i32, gen_shli_i64 => shli_t;
        /// `tcg_gen_shri`.
        gen_shri_i32, gen_shri_i64 => shri_t;
        /// `tcg_gen_sari`.
        gen_sari_i32, gen_sari_i64 => sari_t;
        /// `tcg_gen_rotli`.
        gen_rotli_i32, gen_rotli_i64 => rotli_t;
        /// `tcg_gen_rotri`.
        gen_rotri_i32, gen_rotri_i64 => rotri_t;
        /// `tcg_gen_muli`.
        gen_muli_i32, gen_muli_i64 => muli_t;
        /// `tcg_gen_clzi`.
        gen_clzi_i32, gen_clzi_i64 => clzi_t;
        /// `tcg_gen_ctzi`.
        gen_ctzi_i32, gen_ctzi_i64 => ctzi_t;
    }

    wrap_cond! {
        /// `tcg_gen_setcond`.
        gen_setcond_i32, gen_setcond_i64, gen_setcondi_i32, gen_setcondi_i64
            => setcond_t, setcondi_t;
        /// `tcg_gen_negsetcond`.
        gen_negsetcond_i32, gen_negsetcond_i64, gen_negsetcondi_i32, gen_negsetcondi_i64
            => negsetcond_t, negsetcondi_t;
    }

    /// `tcg_gen_brcond_i32`.
    pub fn gen_brcond_i32(&mut self, cond: Cond, a: TempI32, b: TempI32, l: Label) {
        self.brcond_t(Type::I32, cond, a.0, b.0, l);
    }

    /// `tcg_gen_brcond_i64`.
    pub fn gen_brcond_i64(&mut self, cond: Cond, a: TempI64, b: TempI64, l: Label) {
        self.brcond_t(Type::I64, cond, a.0, b.0, l);
    }

    /// `tcg_gen_brcondi_i32`. Unlike the 64-bit version, no constant is created for ALWAYS and
    /// NEVER.
    pub fn gen_brcondi_i32(&mut self, cond: Cond, a: TempI32, b: i32, l: Label) {
        match cond {
            Cond::Always => self.gen_br(l),
            Cond::Never => {}
            _ => {
                let c = self.cst(Type::I32, b as i64);
                self.brcond_t(Type::I32, cond, a.0, c, l);
            }
        }
    }

    /// `tcg_gen_brcondi_i64`.
    pub fn gen_brcondi_i64(&mut self, cond: Cond, a: TempI64, b: i64, l: Label) {
        let c = self.cst(Type::I64, b);
        self.brcond_t(Type::I64, cond, a.0, c, l);
    }

    /// `tcg_gen_movcond_i32`.
    pub fn gen_movcond_i32(
        &mut self,
        cond: Cond,
        ret: TempI32,
        c1: TempI32,
        c2: TempI32,
        v1: TempI32,
        v2: TempI32,
    ) {
        self.movcond_t(Type::I32, cond, ret.0, c1.0, c2.0, v1.0, v2.0);
    }

    /// `tcg_gen_movcond_i64`.
    pub fn gen_movcond_i64(
        &mut self,
        cond: Cond,
        ret: TempI64,
        c1: TempI64,
        c2: TempI64,
        v1: TempI64,
        v2: TempI64,
    ) {
        self.movcond_t(Type::I64, cond, ret.0, c1.0, c2.0, v1.0, v2.0);
    }

    /// `tcg_gen_deposit_i32`.
    pub fn gen_deposit_i32(&mut self, ret: TempI32, a: TempI32, b: TempI32, ofs: u32, len: u32) {
        self.deposit_t(Type::I32, ret.0, a.0, b.0, ofs, len);
    }

    /// `tcg_gen_deposit_i64`.
    pub fn gen_deposit_i64(&mut self, ret: TempI64, a: TempI64, b: TempI64, ofs: u32, len: u32) {
        self.deposit_t(Type::I64, ret.0, a.0, b.0, ofs, len);
    }

    /// `tcg_gen_deposit_z_i32`.
    pub fn gen_deposit_z_i32(&mut self, ret: TempI32, a: TempI32, ofs: u32, len: u32) {
        self.deposit_z_t(Type::I32, ret.0, a.0, ofs, len);
    }

    /// `tcg_gen_deposit_z_i64`.
    pub fn gen_deposit_z_i64(&mut self, ret: TempI64, a: TempI64, ofs: u32, len: u32) {
        self.deposit_z_t(Type::I64, ret.0, a.0, ofs, len);
    }

    /// `tcg_gen_extract_i32`.
    pub fn gen_extract_i32(&mut self, ret: TempI32, a: TempI32, ofs: u32, len: u32) {
        self.extract_t(Type::I32, ret.0, a.0, ofs, len);
    }

    /// `tcg_gen_extract_i64`.
    pub fn gen_extract_i64(&mut self, ret: TempI64, a: TempI64, ofs: u32, len: u32) {
        self.extract_t(Type::I64, ret.0, a.0, ofs, len);
    }

    /// `tcg_gen_sextract_i32`.
    pub fn gen_sextract_i32(&mut self, ret: TempI32, a: TempI32, ofs: u32, len: u32) {
        self.sextract_t(Type::I32, ret.0, a.0, ofs, len);
    }

    /// `tcg_gen_sextract_i64`.
    pub fn gen_sextract_i64(&mut self, ret: TempI64, a: TempI64, ofs: u32, len: u32) {
        self.sextract_t(Type::I64, ret.0, a.0, ofs, len);
    }

    /// `tcg_gen_extract2_i32`.
    pub fn gen_extract2_i32(&mut self, ret: TempI32, al: TempI32, ah: TempI32, ofs: u32) {
        self.extract2_t(Type::I32, ret.0, al.0, ah.0, ofs);
    }

    /// `tcg_gen_extract2_i64`.
    pub fn gen_extract2_i64(&mut self, ret: TempI64, al: TempI64, ah: TempI64, ofs: u32) {
        self.extract2_t(Type::I64, ret.0, al.0, ah.0, ofs);
    }

    /// `tcg_gen_add2_i32`.
    pub fn gen_add2_i32(
        &mut self,
        rl: TempI32,
        rh: TempI32,
        al: TempI32,
        ah: TempI32,
        bl: TempI32,
        bh: TempI32,
    ) {
        self.add2_t(Type::I32, rl.0, rh.0, al.0, ah.0, bl.0, bh.0);
    }

    /// `tcg_gen_add2_i64`.
    pub fn gen_add2_i64(
        &mut self,
        rl: TempI64,
        rh: TempI64,
        al: TempI64,
        ah: TempI64,
        bl: TempI64,
        bh: TempI64,
    ) {
        self.add2_t(Type::I64, rl.0, rh.0, al.0, ah.0, bl.0, bh.0);
    }

    /// `tcg_gen_sub2_i32`.
    pub fn gen_sub2_i32(
        &mut self,
        rl: TempI32,
        rh: TempI32,
        al: TempI32,
        ah: TempI32,
        bl: TempI32,
        bh: TempI32,
    ) {
        self.sub2_t(Type::I32, rl.0, rh.0, al.0, ah.0, bl.0, bh.0);
    }

    /// `tcg_gen_sub2_i64`.
    pub fn gen_sub2_i64(
        &mut self,
        rl: TempI64,
        rh: TempI64,
        al: TempI64,
        ah: TempI64,
        bl: TempI64,
        bh: TempI64,
    ) {
        self.sub2_t(Type::I64, rl.0, rh.0, al.0, ah.0, bl.0, bh.0);
    }

    /// `tcg_gen_addcio_i32`: `r = a + b + ci`, with the carry out in `co`.
    pub fn gen_addcio_i32(&mut self, r: TempI32, co: TempI32, a: TempI32, b: TempI32, ci: TempI32) {
        self.addcio_t(Type::I32, r.0, co.0, a.0, b.0, ci.0);
    }

    /// `tcg_gen_addcio_i64`.
    pub fn gen_addcio_i64(&mut self, r: TempI64, co: TempI64, a: TempI64, b: TempI64, ci: TempI64) {
        self.addcio_t(Type::I64, r.0, co.0, a.0, b.0, ci.0);
    }

    /// `tcg_gen_addN_i64`: a multi-word add, least significant word first.
    pub fn gen_addn_i64(&mut self, r: &[TempI64], a: &[TempI64], b: &[TempI64]) {
        let n = r.len();
        assert!(n > 2 && a.len() == n && b.len() == n, "addN needs at least three words");
        for (i, ri) in r.iter().enumerate() {
            for j in i + 1..n {
                assert!(*ri != a[j] && *ri != b[j], "addN outputs must not overlap inputs");
            }
        }
        let ty = Type::I64;
        self.op3_t(Opcode::Addco, ty, r[0].0, a[0].0, b[0].0);
        for i in 1..n - 1 {
            self.op3_t(Opcode::Addcio, ty, r[i].0, a[i].0, b[i].0);
        }
        self.op3_t(Opcode::Addci, ty, r[n - 1].0, a[n - 1].0, b[n - 1].0);
    }

    /// `tcg_gen_mulu2_i32`.
    pub fn gen_mulu2_i32(&mut self, rl: TempI32, rh: TempI32, a: TempI32, b: TempI32) {
        self.mul2_t(Opcode::Mulu2, Type::I32, rl.0, rh.0, a.0, b.0);
    }

    /// `tcg_gen_mulu2_i64`.
    pub fn gen_mulu2_i64(&mut self, rl: TempI64, rh: TempI64, a: TempI64, b: TempI64) {
        self.mul2_t(Opcode::Mulu2, Type::I64, rl.0, rh.0, a.0, b.0);
    }

    /// `tcg_gen_muls2_i32`.
    pub fn gen_muls2_i32(&mut self, rl: TempI32, rh: TempI32, a: TempI32, b: TempI32) {
        self.mul2_t(Opcode::Muls2, Type::I32, rl.0, rh.0, a.0, b.0);
    }

    /// `tcg_gen_muls2_i64`.
    pub fn gen_muls2_i64(&mut self, rl: TempI64, rh: TempI64, a: TempI64, b: TempI64) {
        self.mul2_t(Opcode::Muls2, Type::I64, rl.0, rh.0, a.0, b.0);
    }

    /// `tcg_gen_mulsu2_i32`: signed `a` times unsigned `b`.
    pub fn gen_mulsu2_i32(&mut self, rl: TempI32, rh: TempI32, a: TempI32, b: TempI32) {
        let t0 = self.temp_ebb_new_i64();
        let t1 = self.temp_ebb_new_i64();
        self.gen_ext_i32_i64(t0, a);
        self.gen_extu_i32_i64(t1, b);
        self.gen_mul_i64(t0, t0, t1);
        self.gen_extr_i64_i32(rl, rh, t0);
        self.temp_free(t0);
        self.temp_free(t1);
    }

    /// `tcg_gen_mulsu2_i64`.
    pub fn gen_mulsu2_i64(&mut self, rl: TempI64, rh: TempI64, a: TempI64, b: TempI64) {
        let t0 = self.temp_ebb_new_i64();
        let t1 = self.temp_ebb_new_i64();
        let t2 = self.temp_ebb_new_i64();
        self.gen_mulu2_i64(t0, t1, a, b);
        self.gen_sari_i64(t2, a, 63);
        self.gen_and_i64(t2, t2, b);
        self.gen_sub_i64(rh, t1, t2);
        self.gen_mov_i64(rl, t0);
        self.temp_free(t0);
        self.temp_free(t1);
        self.temp_free(t2);
    }

    /// `tcg_gen_ext32s_i64`.
    pub fn gen_ext32s_i64(&mut self, ret: TempI64, a: TempI64) {
        self.sextract_t(Type::I64, ret.0, a.0, 0, 32);
    }

    /// `tcg_gen_ext32u_i64`.
    pub fn gen_ext32u_i64(&mut self, ret: TempI64, a: TempI64) {
        self.extract_t(Type::I64, ret.0, a.0, 0, 32);
    }

    /// `tcg_gen_bswap16_i32`.
    pub fn gen_bswap16_i32(&mut self, ret: TempI32, a: TempI32, flags: u32) {
        self.bswap_t(Opcode::Bswap16, Type::I32, ret.0, a.0, flags);
    }

    /// `tcg_gen_bswap32_i32`.
    pub fn gen_bswap32_i32(&mut self, ret: TempI32, a: TempI32) {
        self.bswap_t(Opcode::Bswap32, Type::I32, ret.0, a.0, 0);
    }

    /// `tcg_gen_bswap16_i64`.
    pub fn gen_bswap16_i64(&mut self, ret: TempI64, a: TempI64, flags: u32) {
        self.bswap_t(Opcode::Bswap16, Type::I64, ret.0, a.0, flags);
    }

    /// `tcg_gen_bswap32_i64`.
    pub fn gen_bswap32_i64(&mut self, ret: TempI64, a: TempI64, flags: u32) {
        self.bswap_t(Opcode::Bswap32, Type::I64, ret.0, a.0, flags);
    }

    /// `tcg_gen_bswap64_i64`.
    pub fn gen_bswap64_i64(&mut self, ret: TempI64, a: TempI64) {
        self.bswap_t(Opcode::Bswap64, Type::I64, ret.0, a.0, 0);
    }

    /// `tcg_gen_hswap_i32`: swap the 16-bit halves.
    pub fn gen_hswap_i32(&mut self, ret: TempI32, a: TempI32) {
        self.hswap_i32_t(Type::I32, ret.0, a.0);
    }

    /// `tcg_gen_hswap_i64`: swap the 16-bit halves of each 32-bit word, and the words.
    pub fn gen_hswap_i64(&mut self, ret: TempI64, a: TempI64) {
        let m = 0x0000_ffff_0000_ffffi64;
        let t0 = self.temp_ebb_new_i64();
        let t1 = self.temp_ebb_new_i64();
        self.gen_rotli_i64(t1, a, 32);
        self.gen_andi_i64(t0, t1, m);
        self.gen_shli_i64(t0, t0, 16);
        self.gen_shri_i64(t1, t1, 16);
        self.gen_andi_i64(t1, t1, m);
        self.gen_or_i64(ret, t0, t1);
        self.temp_free(t0);
        self.temp_free(t1);
    }

    /// `tcg_gen_wswap_i64`: swap the 32-bit words.
    pub fn gen_wswap_i64(&mut self, ret: TempI64, a: TempI64) {
        self.gen_rotli_i64(ret, a, 32);
    }

    // Host memory loads and stores relative to a pointer.

    wrap_ld! {
        /// `tcg_gen_ld8u_i32`.
        gen_ld8u_i32, I32, TempI32 => Ld8u;
        /// `tcg_gen_ld8s_i32`.
        gen_ld8s_i32, I32, TempI32 => Ld8s;
        /// `tcg_gen_ld16u_i32`.
        gen_ld16u_i32, I32, TempI32 => Ld16u;
        /// `tcg_gen_ld16s_i32`.
        gen_ld16s_i32, I32, TempI32 => Ld16s;
        /// `tcg_gen_ld_i32`.
        gen_ld_i32, I32, TempI32 => Ld;
        /// `tcg_gen_st8_i32`.
        gen_st8_i32, I32, TempI32 => St8;
        /// `tcg_gen_st16_i32`.
        gen_st16_i32, I32, TempI32 => St16;
        /// `tcg_gen_st_i32`.
        gen_st_i32, I32, TempI32 => St;
        /// `tcg_gen_ld8u_i64`.
        gen_ld8u_i64, I64, TempI64 => Ld8u;
        /// `tcg_gen_ld8s_i64`.
        gen_ld8s_i64, I64, TempI64 => Ld8s;
        /// `tcg_gen_ld16u_i64`.
        gen_ld16u_i64, I64, TempI64 => Ld16u;
        /// `tcg_gen_ld16s_i64`.
        gen_ld16s_i64, I64, TempI64 => Ld16s;
        /// `tcg_gen_ld32u_i64`.
        gen_ld32u_i64, I64, TempI64 => Ld32u;
        /// `tcg_gen_ld32s_i64`.
        gen_ld32s_i64, I64, TempI64 => Ld32s;
        /// `tcg_gen_ld_i64`.
        gen_ld_i64, I64, TempI64 => Ld;
        /// `tcg_gen_st8_i64`.
        gen_st8_i64, I64, TempI64 => St8;
        /// `tcg_gen_st16_i64`.
        gen_st16_i64, I64, TempI64 => St16;
        /// `tcg_gen_st32_i64`.
        gen_st32_i64, I64, TempI64 => St32;
        /// `tcg_gen_st_i64`.
        gen_st_i64, I64, TempI64 => St;
        /// `tcg_gen_ld_ptr`.
        gen_ld_ptr, I64, TempPtr => Ld;
        /// `tcg_gen_st_ptr`.
        gen_st_ptr, I64, TempPtr => St;
    }

    // Size changing operations.

    /// `tcg_gen_extrl_i64_i32`.
    pub fn gen_extrl_i64_i32(&mut self, ret: TempI32, a: TempI64) {
        self.op2_t(Opcode::ExtrlI64I32, Type::I32, ret.0, a.0);
    }

    /// `tcg_gen_extrh_i64_i32`.
    pub fn gen_extrh_i64_i32(&mut self, ret: TempI32, a: TempI64) {
        self.op2_t(Opcode::ExtrhI64I32, Type::I32, ret.0, a.0);
    }

    /// `tcg_gen_extu_i32_i64`.
    pub fn gen_extu_i32_i64(&mut self, ret: TempI64, a: TempI32) {
        self.op2_t(Opcode::ExtuI32I64, Type::I64, ret.0, a.0);
    }

    /// `tcg_gen_ext_i32_i64`.
    pub fn gen_ext_i32_i64(&mut self, ret: TempI64, a: TempI32) {
        self.op2_t(Opcode::ExtI32I64, Type::I64, ret.0, a.0);
    }

    /// `tcg_gen_concat_i32_i64`.
    pub fn gen_concat_i32_i64(&mut self, dest: TempI64, low: TempI32, high: TempI32) {
        let tmp = self.temp_ebb_new_i64();
        self.gen_extu_i32_i64(tmp, high);
        self.gen_extu_i32_i64(dest, low);
        self.gen_deposit_i64(dest, dest, tmp, 32, 32);
        self.temp_free(tmp);
    }

    /// `tcg_gen_extr_i64_i32`.
    pub fn gen_extr_i64_i32(&mut self, lo: TempI32, hi: TempI32, a: TempI64) {
        self.gen_extrl_i64_i32(lo, a);
        self.gen_extrh_i64_i32(hi, a);
    }

    /// `tcg_gen_extr32_i64`.
    pub fn gen_extr32_i64(&mut self, lo: TempI64, hi: TempI64, a: TempI64) {
        self.gen_ext32u_i64(lo, a);
        self.gen_shri_i64(hi, a, 32);
    }

    /// `tcg_gen_concat32_i64`.
    pub fn gen_concat32_i64(&mut self, ret: TempI64, lo: TempI64, hi: TempI64) {
        self.gen_deposit_i64(ret, lo, hi, 32, 32);
    }

    /// `tcg_gen_extr_i128_i64`.
    pub fn gen_extr_i128_i64(&mut self, lo: TempI64, hi: TempI64, a: TempI128) {
        self.gen_mov_i64(lo, a.low());
        self.gen_mov_i64(hi, a.high());
    }

    /// `tcg_gen_concat_i64_i128`.
    pub fn gen_concat_i64_i128(&mut self, ret: TempI128, lo: TempI64, hi: TempI64) {
        self.gen_mov_i64(ret.low(), lo);
        self.gen_mov_i64(ret.high(), hi);
    }

    /// `tcg_zero_i128`: a new TB temp holding zero.
    pub fn zero_i128(&mut self) -> TempI128 {
        let zero = self.constant_i64(0);
        let r = self.temp_new_i128();
        self.gen_concat_i64_i128(r, zero, zero);
        r
    }

    /// `tcg_gen_mov_i128`.
    pub fn gen_mov_i128(&mut self, dst: TempI128, src: TempI128) {
        if dst != src {
            self.gen_mov_i64(dst.low(), src.low());
            self.gen_mov_i64(dst.high(), src.high());
        }
    }

    /// `tcg_gen_ld_i128`: the low half is at `offset`, as on a little-endian host.
    pub fn gen_ld_i128(&mut self, ret: TempI128, base: TempPtr, offset: i64) {
        self.gen_ld_i64(ret.low(), base, offset);
        self.gen_ld_i64(ret.high(), base, offset + 8);
    }

    /// `tcg_gen_st_i128`.
    pub fn gen_st_i128(&mut self, val: TempI128, base: TempPtr, offset: i64) {
        self.gen_st_i64(val.low(), base, offset);
        self.gen_st_i64(val.high(), base, offset + 8);
    }

    // Pointer arithmetic, from tcg-op-common.h.

    /// `tcg_gen_mov_ptr`.
    pub fn gen_mov_ptr(&mut self, ret: TempPtr, a: TempPtr) {
        self.mov_t(Type::PTR, ret.0, a.0);
    }

    /// `tcg_gen_movi_ptr`.
    pub fn gen_movi_ptr(&mut self, ret: TempPtr, v: i64) {
        self.movi_t(Type::PTR, ret.0, v);
    }

    /// `tcg_gen_add_ptr`.
    pub fn gen_add_ptr(&mut self, ret: TempPtr, a: TempPtr, b: TempPtr) {
        self.op3_t(Opcode::Add, Type::PTR, ret.0, a.0, b.0);
    }

    /// `tcg_gen_addi_ptr`.
    pub fn gen_addi_ptr(&mut self, ret: TempPtr, a: TempPtr, v: i64) {
        self.addi_t(Type::PTR, ret.0, a.0, v);
    }
}
