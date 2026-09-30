// SPDX-License-Identifier: MIT OR Apache-2.0

//! Guest memory access and atomics, a port of `tcg/tcg-op-ldst.c`.
//!
//! The host is assumed to swap bytes as part of a memory access and to have 128-bit loads and
//! stores, so no `bswap` ops are added around accesses and `qemu_ld2`/`qemu_st2` are always used
//! for 128-bit values. Plugin memory callbacks are not emitted.

use crate::helpers;
use crate::ir::{Func, Temp, TempI32, TempI64, TempI128};
use crate::opcode::Opcode;
use crate::types::{Cond, MemOp, MemOpIdx, Type, mo};

/// A read-modify-write operation for the `atomic_*` family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtomicOp {
    /// `fetch_add`: returns the old value.
    FetchAdd,
    /// `fetch_and`.
    FetchAnd,
    /// `fetch_or`.
    FetchOr,
    /// `fetch_xor`.
    FetchXor,
    /// `fetch_smin`.
    FetchSmin,
    /// `fetch_umin`.
    FetchUmin,
    /// `fetch_smax`.
    FetchSmax,
    /// `fetch_umax`.
    FetchUmax,
    /// `add_fetch`: returns the new value.
    AddFetch,
    /// `and_fetch`.
    AndFetch,
    /// `or_fetch`.
    OrFetch,
    /// `xor_fetch`.
    XorFetch,
    /// `smin_fetch`.
    SminFetch,
    /// `umin_fetch`.
    UminFetch,
    /// `smax_fetch`.
    SmaxFetch,
    /// `umax_fetch`.
    UmaxFetch,
    /// `xchg`: stores the operand and returns the old value.
    Xchg,
}

/// The binary operation an [`AtomicOp`] applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtomicKind {
    /// Addition.
    Add,
    /// Bitwise and.
    And,
    /// Bitwise or.
    Or,
    /// Bitwise exclusive or.
    Xor,
    /// Signed minimum.
    Smin,
    /// Unsigned minimum.
    Umin,
    /// Signed maximum.
    Smax,
    /// Unsigned maximum.
    Umax,
    /// Take the operand.
    Mov2,
}

impl AtomicOp {
    /// The name used in helper names, such as `fetch_add`.
    pub const fn name(self) -> &'static str {
        match self {
            AtomicOp::FetchAdd => "fetch_add",
            AtomicOp::FetchAnd => "fetch_and",
            AtomicOp::FetchOr => "fetch_or",
            AtomicOp::FetchXor => "fetch_xor",
            AtomicOp::FetchSmin => "fetch_smin",
            AtomicOp::FetchUmin => "fetch_umin",
            AtomicOp::FetchSmax => "fetch_smax",
            AtomicOp::FetchUmax => "fetch_umax",
            AtomicOp::AddFetch => "add_fetch",
            AtomicOp::AndFetch => "and_fetch",
            AtomicOp::OrFetch => "or_fetch",
            AtomicOp::XorFetch => "xor_fetch",
            AtomicOp::SminFetch => "smin_fetch",
            AtomicOp::UminFetch => "umin_fetch",
            AtomicOp::SmaxFetch => "smax_fetch",
            AtomicOp::UmaxFetch => "umax_fetch",
            AtomicOp::Xchg => "xchg",
        }
    }

    /// The operation applied to the old value and the operand.
    pub const fn kind(self) -> AtomicKind {
        match self {
            AtomicOp::FetchAdd | AtomicOp::AddFetch => AtomicKind::Add,
            AtomicOp::FetchAnd | AtomicOp::AndFetch => AtomicKind::And,
            AtomicOp::FetchOr | AtomicOp::OrFetch => AtomicKind::Or,
            AtomicOp::FetchXor | AtomicOp::XorFetch => AtomicKind::Xor,
            AtomicOp::FetchSmin | AtomicOp::SminFetch => AtomicKind::Smin,
            AtomicOp::FetchUmin | AtomicOp::UminFetch => AtomicKind::Umin,
            AtomicOp::FetchSmax | AtomicOp::SmaxFetch => AtomicKind::Smax,
            AtomicOp::FetchUmax | AtomicOp::UmaxFetch => AtomicKind::Umax,
            AtomicOp::Xchg => AtomicKind::Mov2,
        }
    }

    /// True if the result is the new value rather than the old one.
    pub const fn returns_new(self) -> bool {
        matches!(
            self,
            AtomicOp::AddFetch
                | AtomicOp::AndFetch
                | AtomicOp::OrFetch
                | AtomicOp::XorFetch
                | AtomicOp::SminFetch
                | AtomicOp::UminFetch
                | AtomicOp::SmaxFetch
                | AtomicOp::UmaxFetch
        )
    }

    /// True if QEMU provides a 128-bit form, `GEN_ATOMIC_HELPER128`.
    pub const fn has_i128(self) -> bool {
        matches!(self, AtomicOp::FetchAnd | AtomicOp::FetchOr | AtomicOp::Xchg)
    }

    /// Every operation, in QEMU's order.
    pub const ALL: [AtomicOp; 17] = [
        AtomicOp::FetchAdd,
        AtomicOp::FetchAnd,
        AtomicOp::FetchOr,
        AtomicOp::FetchXor,
        AtomicOp::FetchSmin,
        AtomicOp::FetchUmin,
        AtomicOp::FetchSmax,
        AtomicOp::FetchUmax,
        AtomicOp::AddFetch,
        AtomicOp::AndFetch,
        AtomicOp::OrFetch,
        AtomicOp::XorFetch,
        AtomicOp::SminFetch,
        AtomicOp::UminFetch,
        AtomicOp::SmaxFetch,
        AtomicOp::UmaxFetch,
        AtomicOp::Xchg,
    ];
}

macro_rules! named_atomics {
    ($($(#[$m:meta])* $n32:ident, $n64:ident => $op:ident;)*) => {$(
        $(#[$m])*
        pub fn $n32(
            &mut self,
            ret: TempI32,
            addr: impl Into<Temp>,
            val: TempI32,
            idx: u32,
            memop: MemOp,
        ) {
            self.gen_atomic_op_i32(AtomicOp::$op, ret, addr, val, idx, memop)
        }
        $(#[$m])*
        pub fn $n64(
            &mut self,
            ret: TempI64,
            addr: impl Into<Temp>,
            val: TempI64,
            idx: u32,
            memop: MemOp,
        ) {
            self.gen_atomic_op_i64(AtomicOp::$op, ret, addr, val, idx, memop)
        }
    )*};
}

impl Func {
    fn parallel(&self) -> bool {
        self.config.parallel
    }

    fn check_addr(&self, addr: Temp) {
        assert_eq!(self.temp(addr).ty, self.config.addr_type, "address has the wrong type");
    }

    /// `tcg_canonicalize_memop`.
    pub fn canonicalize_memop(&self, op: MemOp, is64: bool, st: bool) -> MemOp {
        let mut op = op.0;
        let a_bits = MemOp(op).alignment_bits();
        if a_bits == op & MemOp::SIZE.0 {
            op = (op & !MemOp::AMASK.0) | MemOp::ALIGN.0;
        }
        match op & MemOp::SIZE.0 {
            0 => op &= !MemOp::BSWAP.0,
            1 => {}
            2 => {
                if !is64 {
                    op &= !MemOp::SIGN.0;
                }
            }
            3 if is64 => op &= !MemOp::SIGN.0,
            _ => panic!("bad memop size"),
        }
        if st {
            op &= !MemOp::SIGN.0;
        }
        if !self.parallel() {
            op = (op & !MemOp::ATOM_MASK.0) | MemOp::ATOM_NONE.0;
        }
        MemOp(op)
    }

    fn req_mo(&mut self, ty: u32) {
        let ty = ty & self.config.guest_mo & !self.config.target_default_mo;
        if ty != 0 {
            self.gen_mb(ty | mo::BAR_SC);
        }
    }

    fn gen_ldst1(&mut self, opc: Opcode, ty: Type, v: Temp, addr: Temp, oi: MemOpIdx) {
        let op = self.emit_op(opc, ty, &[v.arg(), addr.arg(), oi.0 as u64]);
        self.op_mut(op).flags = oi.memop().size() as u8;
    }

    fn gen_ldst2(&mut self, opc: Opcode, ty: Type, vl: Temp, vh: Temp, addr: Temp, oi: MemOpIdx) {
        let op = self.emit_op(opc, ty, &[vl.arg(), vh.arg(), addr.arg(), oi.0 as u64]);
        self.op_mut(op).flags = oi.memop().size() as u8;
    }

    fn qemu_ld_i32_int(&mut self, val: TempI32, addr: Temp, idx: u32, memop: MemOp) {
        self.req_mo(mo::LD_LD | mo::ST_LD);
        let memop = self.canonicalize_memop(memop, false, false);
        self.gen_ldst1(Opcode::QemuLd, Type::I32, val.0, addr, MemOpIdx::new(memop, idx));
    }

    fn qemu_st_i32_int(&mut self, val: TempI32, addr: Temp, idx: u32, memop: MemOp) {
        self.req_mo(mo::LD_ST | mo::ST_ST);
        let memop = self.canonicalize_memop(memop, false, true);
        self.gen_ldst1(Opcode::QemuSt, Type::I32, val.0, addr, MemOpIdx::new(memop, idx));
    }

    fn qemu_ld_i64_int(&mut self, val: TempI64, addr: Temp, idx: u32, memop: MemOp) {
        self.req_mo(mo::LD_LD | mo::ST_LD);
        let memop = self.canonicalize_memop(memop, true, false);
        self.gen_ldst1(Opcode::QemuLd, Type::I64, val.0, addr, MemOpIdx::new(memop, idx));
    }

    fn qemu_st_i64_int(&mut self, val: TempI64, addr: Temp, idx: u32, memop: MemOp) {
        self.req_mo(mo::LD_ST | mo::ST_ST);
        let memop = self.canonicalize_memop(memop, true, true);
        self.gen_ldst1(Opcode::QemuSt, Type::I64, val.0, addr, MemOpIdx::new(memop, idx));
    }

    fn reduce_atom(&self, memop: MemOp) -> MemOp {
        if self.parallel() {
            memop
        } else {
            MemOp((memop.0 & !MemOp::ATOM_MASK.0) | MemOp::ATOM_NONE.0)
        }
    }

    fn qemu_ld_i128_int(&mut self, val: TempI128, addr: Temp, idx: u32, memop: MemOp) {
        self.req_mo(mo::LD_LD | mo::ST_LD);
        let memop = self.reduce_atom(memop);
        let oi = MemOpIdx::new(memop, idx);
        self.gen_ldst2(Opcode::QemuLd2, Type::I128, val.low().0, val.high().0, addr, oi);
    }

    fn qemu_st_i128_int(&mut self, val: TempI128, addr: Temp, idx: u32, memop: MemOp) {
        self.req_mo(mo::ST_LD | mo::ST_ST);
        let memop = self.reduce_atom(memop);
        let oi = MemOpIdx::new(memop, idx);
        self.gen_ldst2(Opcode::QemuSt2, Type::I128, val.low().0, val.high().0, addr, oi);
    }

    /// `tcg_gen_qemu_ld_i32`: load from guest memory through MMU index `idx`.
    pub fn gen_qemu_ld_i32(&mut self, val: TempI32, addr: impl Into<Temp>, idx: u32, memop: MemOp) {
        let addr = addr.into();
        self.check_addr(addr);
        assert!(memop.size() <= 2, "access too large for i32");
        self.qemu_ld_i32_int(val, addr, idx, memop);
    }

    /// `tcg_gen_qemu_st_i32`.
    pub fn gen_qemu_st_i32(&mut self, val: TempI32, addr: impl Into<Temp>, idx: u32, memop: MemOp) {
        let addr = addr.into();
        self.check_addr(addr);
        assert!(memop.size() <= 2, "access too large for i32");
        self.qemu_st_i32_int(val, addr, idx, memop);
    }

    /// `tcg_gen_qemu_ld_i64`.
    pub fn gen_qemu_ld_i64(&mut self, val: TempI64, addr: impl Into<Temp>, idx: u32, memop: MemOp) {
        let addr = addr.into();
        self.check_addr(addr);
        assert!(memop.size() <= 3, "access too large for i64");
        self.qemu_ld_i64_int(val, addr, idx, memop);
    }

    /// `tcg_gen_qemu_st_i64`.
    pub fn gen_qemu_st_i64(&mut self, val: TempI64, addr: impl Into<Temp>, idx: u32, memop: MemOp) {
        let addr = addr.into();
        self.check_addr(addr);
        assert!(memop.size() <= 3, "access too large for i64");
        self.qemu_st_i64_int(val, addr, idx, memop);
    }

    /// `tcg_gen_qemu_ld_i128`.
    pub fn gen_qemu_ld_i128(
        &mut self,
        val: TempI128,
        addr: impl Into<Temp>,
        idx: u32,
        memop: MemOp,
    ) {
        let addr = addr.into();
        self.check_addr(addr);
        assert!(memop.size() == 4 && !memop.is_signed(), "i128 access must be MO_128");
        self.qemu_ld_i128_int(val, addr, idx, memop);
    }

    /// `tcg_gen_qemu_st_i128`.
    pub fn gen_qemu_st_i128(
        &mut self,
        val: TempI128,
        addr: impl Into<Temp>,
        idx: u32,
        memop: MemOp,
    ) {
        let addr = addr.into();
        self.check_addr(addr);
        assert!(memop.size() == 4 && !memop.is_signed(), "i128 access must be MO_128");
        self.qemu_st_i128_int(val, addr, idx, memop);
    }

    /// `tcg_gen_ext_i32`: extend by the size and sign of `opc`.
    pub fn gen_ext_i32(&mut self, ret: TempI32, val: TempI32, opc: MemOp) {
        match opc.0 & MemOp::SSIZE.0 {
            8 => self.gen_ext8s_i32(ret, val),
            0 => self.gen_ext8u_i32(ret, val),
            9 => self.gen_ext16s_i32(ret, val),
            1 => self.gen_ext16u_i32(ret, val),
            2 | 10 => self.gen_mov_i32(ret, val),
            _ => panic!("bad size for ext_i32"),
        }
    }

    /// `tcg_gen_ext_i64`.
    pub fn gen_ext_i64(&mut self, ret: TempI64, val: TempI64, opc: MemOp) {
        match opc.0 & MemOp::SSIZE.0 {
            8 => self.gen_ext8s_i64(ret, val),
            0 => self.gen_ext8u_i64(ret, val),
            9 => self.gen_ext16s_i64(ret, val),
            1 => self.gen_ext16u_i64(ret, val),
            10 => self.gen_ext32s_i64(ret, val),
            2 => self.gen_ext32u_i64(ret, val),
            3 | 11 => self.gen_mov_i64(ret, val),
            _ => panic!("bad size for ext_i64"),
        }
    }

    fn maybe_extend_addr64(&mut self, addr: Temp) -> TempI64 {
        if self.config.addr_type == Type::I32 {
            let a64 = self.temp_ebb_new_i64();
            self.gen_extu_i32_i64(a64, TempI32(addr));
            a64
        } else {
            TempI64(addr)
        }
    }

    fn maybe_free_addr64(&mut self, a64: TempI64) {
        if self.config.addr_type == Type::I32 {
            self.temp_free(a64);
        }
    }

    fn nonatomic_cmpxchg_i32_int(
        &mut self,
        retv: TempI32,
        addr: Temp,
        cmpv: TempI32,
        newv: TempI32,
        idx: u32,
        memop: MemOp,
    ) {
        let t1 = self.temp_ebb_new_i32();
        let t2 = self.temp_ebb_new_i32();
        self.gen_ext_i32(t2, cmpv, memop & MemOp::SIZE);
        self.qemu_ld_i32_int(t1, addr, idx, memop.without(MemOp::SIGN));
        self.gen_movcond_i32(Cond::Eq, t2, t1, t2, newv, t1);
        self.qemu_st_i32_int(t2, addr, idx, memop);
        self.temp_free(t2);
        if memop.is_signed() {
            self.gen_ext_i32(retv, t1, memop);
        } else {
            self.gen_mov_i32(retv, t1);
        }
        self.temp_free(t1);
    }

    fn nonatomic_cmpxchg_i64_int(
        &mut self,
        retv: TempI64,
        addr: Temp,
        cmpv: TempI64,
        newv: TempI64,
        idx: u32,
        memop: MemOp,
    ) {
        let t1 = self.temp_ebb_new_i64();
        let t2 = self.temp_ebb_new_i64();
        self.gen_ext_i64(t2, cmpv, memop & MemOp::SIZE);
        self.qemu_ld_i64_int(t1, addr, idx, memop.without(MemOp::SIGN));
        self.gen_movcond_i64(Cond::Eq, t2, t1, t2, newv, t1);
        self.qemu_st_i64_int(t2, addr, idx, memop);
        self.temp_free(t2);
        if memop.is_signed() {
            self.gen_ext_i64(retv, t1, memop);
        } else {
            self.gen_mov_i64(retv, t1);
        }
        self.temp_free(t1);
    }

    fn nonatomic_cmpxchg_i128_int(
        &mut self,
        retv: TempI128,
        addr: Temp,
        cmpv: TempI128,
        newv: TempI128,
        idx: u32,
        memop: MemOp,
    ) {
        let oldv = self.temp_ebb_new_i128();
        let tmpv = self.temp_ebb_new_i128();
        let t0 = self.temp_ebb_new_i64();
        let t1 = self.temp_ebb_new_i64();
        let z = self.constant_i64(0);
        self.qemu_ld_i128_int(oldv, addr, idx, memop);
        self.gen_xor_i64(t0, oldv.low(), cmpv.low());
        self.gen_xor_i64(t1, oldv.high(), cmpv.high());
        self.gen_or_i64(t0, t0, t1);
        self.gen_movcond_i64(Cond::Eq, tmpv.low(), t0, z, newv.low(), oldv.low());
        self.gen_movcond_i64(Cond::Eq, tmpv.high(), t0, z, newv.high(), oldv.high());
        self.qemu_st_i128_int(tmpv, addr, idx, memop);
        self.gen_mov_i128(retv, oldv);
        self.temp_free(t0);
        self.temp_free(t1);
        self.temp_free(tmpv);
        self.temp_free(oldv);
    }

    fn atomic_cmpxchg_i32_int(
        &mut self,
        retv: TempI32,
        addr: Temp,
        cmpv: TempI32,
        newv: TempI32,
        idx: u32,
        memop: MemOp,
    ) {
        if !self.parallel() {
            return self.nonatomic_cmpxchg_i32_int(retv, addr, cmpv, newv, idx, memop);
        }
        let memop = self.canonicalize_memop(memop, false, false);
        let h = self.helper(helpers::atomic_cmpxchg(memop));
        let oi = MemOpIdx::new(memop.without(MemOp::SIGN), idx);
        let a64 = self.maybe_extend_addr64(addr);
        let env = self.env();
        let oic = self.constant_i32(oi.0 as i32);
        self.gen_call(h, Some(retv.0), &[env.0, a64.0, cmpv.0, newv.0, oic.0]);
        self.maybe_free_addr64(a64);
        if memop.is_signed() {
            self.gen_ext_i32(retv, retv, memop);
        }
    }

    /// `tcg_gen_nonatomic_cmpxchg_i32`.
    pub fn gen_nonatomic_cmpxchg_i32(
        &mut self,
        retv: TempI32,
        addr: impl Into<Temp>,
        cmpv: TempI32,
        newv: TempI32,
        idx: u32,
        memop: MemOp,
    ) {
        let addr = addr.into();
        self.check_addr(addr);
        assert!(memop.size() <= 2);
        self.nonatomic_cmpxchg_i32_int(retv, addr, cmpv, newv, idx, memop);
    }

    /// `tcg_gen_nonatomic_cmpxchg_i64`.
    pub fn gen_nonatomic_cmpxchg_i64(
        &mut self,
        retv: TempI64,
        addr: impl Into<Temp>,
        cmpv: TempI64,
        newv: TempI64,
        idx: u32,
        memop: MemOp,
    ) {
        let addr = addr.into();
        self.check_addr(addr);
        assert!(memop.size() <= 3);
        self.nonatomic_cmpxchg_i64_int(retv, addr, cmpv, newv, idx, memop);
    }

    /// `tcg_gen_nonatomic_cmpxchg_i128`.
    pub fn gen_nonatomic_cmpxchg_i128(
        &mut self,
        retv: TempI128,
        addr: impl Into<Temp>,
        cmpv: TempI128,
        newv: TempI128,
        idx: u32,
        memop: MemOp,
    ) {
        let addr = addr.into();
        self.check_addr(addr);
        assert!(memop.0 & (MemOp::SIZE.0 | MemOp::SIGN.0) == MemOp::MO_128.0);
        self.nonatomic_cmpxchg_i128_int(retv, addr, cmpv, newv, idx, memop);
    }

    /// `tcg_gen_atomic_cmpxchg_i32`.
    pub fn gen_atomic_cmpxchg_i32(
        &mut self,
        retv: TempI32,
        addr: impl Into<Temp>,
        cmpv: TempI32,
        newv: TempI32,
        idx: u32,
        memop: MemOp,
    ) {
        let addr = addr.into();
        self.check_addr(addr);
        assert!(memop.size() <= 2);
        self.atomic_cmpxchg_i32_int(retv, addr, cmpv, newv, idx, memop);
    }

    /// `tcg_gen_atomic_cmpxchg_i64`.
    pub fn gen_atomic_cmpxchg_i64(
        &mut self,
        retv: TempI64,
        addr: impl Into<Temp>,
        cmpv: TempI64,
        newv: TempI64,
        idx: u32,
        memop: MemOp,
    ) {
        let addr = addr.into();
        self.check_addr(addr);
        assert!(memop.size() <= 3);
        if !self.parallel() {
            return self.nonatomic_cmpxchg_i64_int(retv, addr, cmpv, newv, idx, memop);
        }
        if memop.size() == 3 {
            let memop = self.canonicalize_memop(memop, true, false);
            let h = self.helper(helpers::atomic_cmpxchg(memop));
            let oi = MemOpIdx::new(memop, idx);
            let a64 = self.maybe_extend_addr64(addr);
            let env = self.env();
            let oic = self.constant_i32(oi.0 as i32);
            self.gen_call(h, Some(retv.0), &[env.0, a64.0, cmpv.0, newv.0, oic.0]);
            self.maybe_free_addr64(a64);
        } else {
            let c32 = self.temp_ebb_new_i32();
            let n32 = self.temp_ebb_new_i32();
            let r32 = self.temp_ebb_new_i32();
            self.gen_extrl_i64_i32(c32, cmpv);
            self.gen_extrl_i64_i32(n32, newv);
            self.atomic_cmpxchg_i32_int(r32, addr, c32, n32, idx, memop.without(MemOp::SIGN));
            self.temp_free(c32);
            self.temp_free(n32);
            self.gen_extu_i32_i64(retv, r32);
            self.temp_free(r32);
            if memop.is_signed() {
                self.gen_ext_i64(retv, retv, memop);
            }
        }
    }

    /// `tcg_gen_atomic_cmpxchg_i128`.
    pub fn gen_atomic_cmpxchg_i128(
        &mut self,
        retv: TempI128,
        addr: impl Into<Temp>,
        cmpv: TempI128,
        newv: TempI128,
        idx: u32,
        memop: MemOp,
    ) {
        let addr = addr.into();
        self.check_addr(addr);
        assert!(memop.0 & (MemOp::SIZE.0 | MemOp::SIGN.0) == MemOp::MO_128.0);
        if !self.parallel() {
            return self.nonatomic_cmpxchg_i128_int(retv, addr, cmpv, newv, idx, memop);
        }
        let h = self.helper(helpers::atomic_cmpxchg(memop));
        let oi = MemOpIdx::new(memop, idx);
        let a64 = self.maybe_extend_addr64(addr);
        let env = self.env();
        let oic = self.constant_i32(oi.0 as i32);
        self.gen_call(h, Some(retv.0), &[env.0, a64.0, cmpv.0, newv.0, oic.0]);
        self.maybe_free_addr64(a64);
    }

    fn atomic_kind_t(&mut self, k: AtomicKind, ty: Type, r: Temp, a: Temp, b: Temp) {
        match ty {
            Type::I32 => {
                let (r, a, b) = (TempI32(r), TempI32(a), TempI32(b));
                match k {
                    AtomicKind::Add => self.gen_add_i32(r, a, b),
                    AtomicKind::And => self.gen_and_i32(r, a, b),
                    AtomicKind::Or => self.gen_or_i32(r, a, b),
                    AtomicKind::Xor => self.gen_xor_i32(r, a, b),
                    AtomicKind::Smin => self.gen_smin_i32(r, a, b),
                    AtomicKind::Umin => self.gen_umin_i32(r, a, b),
                    AtomicKind::Smax => self.gen_smax_i32(r, a, b),
                    AtomicKind::Umax => self.gen_umax_i32(r, a, b),
                    AtomicKind::Mov2 => self.gen_mov_i32(r, b),
                }
            }
            _ => {
                let (r, a, b) = (TempI64(r), TempI64(a), TempI64(b));
                match k {
                    AtomicKind::Add => self.gen_add_i64(r, a, b),
                    AtomicKind::And => self.gen_and_i64(r, a, b),
                    AtomicKind::Or => self.gen_or_i64(r, a, b),
                    AtomicKind::Xor => self.gen_xor_i64(r, a, b),
                    AtomicKind::Smin => self.gen_smin_i64(r, a, b),
                    AtomicKind::Umin => self.gen_umin_i64(r, a, b),
                    AtomicKind::Smax => self.gen_smax_i64(r, a, b),
                    AtomicKind::Umax => self.gen_umax_i64(r, a, b),
                    AtomicKind::Mov2 => self.gen_mov_i64(r, b),
                }
            }
        }
    }

    fn do_atomic_op_i32(
        &mut self,
        op: AtomicOp,
        ret: TempI32,
        addr: Temp,
        val: TempI32,
        idx: u32,
        memop: MemOp,
    ) {
        let memop = self.canonicalize_memop(memop, false, false);
        let h = self.helper(helpers::atomic_op(op.name(), memop));
        let oi = MemOpIdx::new(memop.without(MemOp::SIGN), idx);
        let a64 = self.maybe_extend_addr64(addr);
        let env = self.env();
        let oic = self.constant_i32(oi.0 as i32);
        self.gen_call(h, Some(ret.0), &[env.0, a64.0, val.0, oic.0]);
        self.maybe_free_addr64(a64);
        if memop.is_signed() {
            self.gen_ext_i32(ret, ret, memop);
        }
    }

    /// `tcg_gen_atomic_<op>_i32`, for any [`AtomicOp`].
    pub fn gen_atomic_op_i32(
        &mut self,
        op: AtomicOp,
        ret: TempI32,
        addr: impl Into<Temp>,
        val: TempI32,
        idx: u32,
        memop: MemOp,
    ) {
        let addr = addr.into();
        self.check_addr(addr);
        assert!(memop.size() <= 2);
        if self.parallel() {
            return self.do_atomic_op_i32(op, ret, addr, val, idx, memop);
        }
        let t1 = self.temp_ebb_new_i32();
        let t2 = self.temp_ebb_new_i32();
        let memop = self.canonicalize_memop(memop, false, false);
        self.qemu_ld_i32_int(t1, addr, idx, memop);
        self.gen_ext_i32(t2, val, memop);
        self.atomic_kind_t(op.kind(), Type::I32, t2.0, t1.0, t2.0);
        self.qemu_st_i32_int(t2, addr, idx, memop);
        self.gen_ext_i32(ret, if op.returns_new() { t2 } else { t1 }, memop);
        self.temp_free(t1);
        self.temp_free(t2);
    }

    /// `tcg_gen_atomic_<op>_i64`, for any [`AtomicOp`].
    pub fn gen_atomic_op_i64(
        &mut self,
        op: AtomicOp,
        ret: TempI64,
        addr: impl Into<Temp>,
        val: TempI64,
        idx: u32,
        memop: MemOp,
    ) {
        let addr = addr.into();
        self.check_addr(addr);
        assert!(memop.size() <= 3);
        if !self.parallel() {
            let t1 = self.temp_ebb_new_i64();
            let t2 = self.temp_ebb_new_i64();
            let memop = self.canonicalize_memop(memop, true, false);
            self.qemu_ld_i64_int(t1, addr, idx, memop);
            self.gen_ext_i64(t2, val, memop);
            self.atomic_kind_t(op.kind(), Type::I64, t2.0, t1.0, t2.0);
            self.qemu_st_i64_int(t2, addr, idx, memop);
            self.gen_ext_i64(ret, if op.returns_new() { t2 } else { t1 }, memop);
            self.temp_free(t1);
            self.temp_free(t2);
            return;
        }
        let memop = self.canonicalize_memop(memop, true, false);
        if memop.size() == 3 {
            let h = self.helper(helpers::atomic_op(op.name(), memop));
            let oi = MemOpIdx::new(memop.without(MemOp::SIGN), idx);
            let a64 = self.maybe_extend_addr64(addr);
            let env = self.env();
            let oic = self.constant_i32(oi.0 as i32);
            self.gen_call(h, Some(ret.0), &[env.0, a64.0, val.0, oic.0]);
            self.maybe_free_addr64(a64);
        } else {
            let v32 = self.temp_ebb_new_i32();
            let r32 = self.temp_ebb_new_i32();
            self.gen_extrl_i64_i32(v32, val);
            self.do_atomic_op_i32(op, r32, addr, v32, idx, memop.without(MemOp::SIGN));
            self.temp_free(v32);
            self.gen_extu_i32_i64(ret, r32);
            self.temp_free(r32);
            if memop.is_signed() {
                self.gen_ext_i64(ret, ret, memop);
            }
        }
    }

    /// `tcg_gen_atomic_<op>_i128`. Only `fetch_and`, `fetch_or` and `xchg` exist in 128 bits.
    pub fn gen_atomic_op_i128(
        &mut self,
        op: AtomicOp,
        ret: TempI128,
        addr: impl Into<Temp>,
        val: TempI128,
        idx: u32,
        memop: MemOp,
    ) {
        let addr = addr.into();
        self.check_addr(addr);
        assert!(op.has_i128(), "no 128-bit form of atomic {}", op.name());
        assert!(memop.size() == 4);
        if self.parallel() {
            let h = self.helper(helpers::atomic_op(op.name(), memop));
            let oi = MemOpIdx::new(memop.without(MemOp::SIGN), idx);
            let a64 = self.maybe_extend_addr64(addr);
            let env = self.env();
            let oic = self.constant_i32(oi.0 as i32);
            self.gen_call(h, Some(ret.0), &[env.0, a64.0, val.0, oic.0]);
            self.maybe_free_addr64(a64);
            return;
        }
        let t = self.temp_ebb_new_i128();
        let r = self.temp_ebb_new_i128();
        self.qemu_ld_i128_int(r, addr, idx, memop);
        let k = op.kind();
        self.atomic_kind_t(k, Type::I64, t.low().0, r.low().0, val.low().0);
        self.atomic_kind_t(k, Type::I64, t.high().0, r.high().0, val.high().0);
        self.qemu_st_i128_int(t, addr, idx, memop);
        self.gen_mov_i128(ret, r);
        self.temp_free(t);
        self.temp_free(r);
    }

    named_atomics! {
        /// `tcg_gen_atomic_fetch_add`.
        gen_atomic_fetch_add_i32, gen_atomic_fetch_add_i64 => FetchAdd;
        /// `tcg_gen_atomic_fetch_and`.
        gen_atomic_fetch_and_i32, gen_atomic_fetch_and_i64 => FetchAnd;
        /// `tcg_gen_atomic_fetch_or`.
        gen_atomic_fetch_or_i32, gen_atomic_fetch_or_i64 => FetchOr;
        /// `tcg_gen_atomic_fetch_xor`.
        gen_atomic_fetch_xor_i32, gen_atomic_fetch_xor_i64 => FetchXor;
        /// `tcg_gen_atomic_fetch_smin`.
        gen_atomic_fetch_smin_i32, gen_atomic_fetch_smin_i64 => FetchSmin;
        /// `tcg_gen_atomic_fetch_umin`.
        gen_atomic_fetch_umin_i32, gen_atomic_fetch_umin_i64 => FetchUmin;
        /// `tcg_gen_atomic_fetch_smax`.
        gen_atomic_fetch_smax_i32, gen_atomic_fetch_smax_i64 => FetchSmax;
        /// `tcg_gen_atomic_fetch_umax`.
        gen_atomic_fetch_umax_i32, gen_atomic_fetch_umax_i64 => FetchUmax;
        /// `tcg_gen_atomic_add_fetch`.
        gen_atomic_add_fetch_i32, gen_atomic_add_fetch_i64 => AddFetch;
        /// `tcg_gen_atomic_and_fetch`.
        gen_atomic_and_fetch_i32, gen_atomic_and_fetch_i64 => AndFetch;
        /// `tcg_gen_atomic_or_fetch`.
        gen_atomic_or_fetch_i32, gen_atomic_or_fetch_i64 => OrFetch;
        /// `tcg_gen_atomic_xor_fetch`.
        gen_atomic_xor_fetch_i32, gen_atomic_xor_fetch_i64 => XorFetch;
        /// `tcg_gen_atomic_smin_fetch`.
        gen_atomic_smin_fetch_i32, gen_atomic_smin_fetch_i64 => SminFetch;
        /// `tcg_gen_atomic_umin_fetch`.
        gen_atomic_umin_fetch_i32, gen_atomic_umin_fetch_i64 => UminFetch;
        /// `tcg_gen_atomic_smax_fetch`.
        gen_atomic_smax_fetch_i32, gen_atomic_smax_fetch_i64 => SmaxFetch;
        /// `tcg_gen_atomic_umax_fetch`.
        gen_atomic_umax_fetch_i32, gen_atomic_umax_fetch_i64 => UmaxFetch;
        /// `tcg_gen_atomic_xchg`.
        gen_atomic_xchg_i32, gen_atomic_xchg_i64 => Xchg;
    }
}
