// SPDX-License-Identifier: MIT OR Apache-2.0

//! The tier-1 optimizer, a port of `tcg/optimize.c`.
//!
//! It walks the ops once, forward, and within each extended basic block it propagates copies
//! and constants, folds constant expressions, tracks known-zero, known-one and sign-repetition
//! bits, folds conditional branches whose outcome is known, and remembers which env slots hold a
//! copy of which temp so that redundant loads and stores go away. Dead code is removed later by
//! [`Func::reachable_code_pass`] and [`Func::liveness_pass_1`].
//!
//! Differences from QEMU:
//!
//! - The host is the virtual host of this crate, which supports every op, `deposit` and
//!   `extract` at every position and length, and the test conditions. The fallback lowerings
//!   QEMU uses for hosts without them are therefore never reached and are not ported.
//! - The env memory copies are kept in a vector in insertion order instead of an interval tree.
//!   Lookups visit entries in the same order as the tree does for equal start offsets, so the
//!   result is the same.

use crate::ir::{Func, OpId, Temp};
use crate::opcode::Opcode;
use crate::types::{Cond, MemOpIdx, TempKind, Type, bswap, dup_const};

const NO_DEST: u64 = u64::MAX;

#[derive(Clone, Copy, Debug)]
struct Info {
    prev_copy: Temp,
    next_copy: Temp,
    z_mask: u64,
    o_mask: u64,
    s_mask: u64,
}

#[derive(Clone, Copy, Debug)]
struct MemCopy {
    start: i64,
    last: i64,
    ty: Type,
    ts: Temp,
}

fn make_mask(shift: u32, len: u32) -> u64 {
    (!0u64 >> (64 - len)) << shift
}

fn clrsb64(x: u64) -> u32 {
    let x = x as i64;
    ((x ^ (x >> 63)) as u64).leading_zeros() - 1
}

fn extract64(v: u64, pos: u32, len: u32) -> u64 {
    (v >> pos) & (!0u64 >> (64 - len))
}

fn sextract64(v: u64, pos: u32, len: u32) -> u64 {
    (((v << (64 - len - pos)) as i64) >> (64 - len)) as u64
}

fn deposit64(v: u64, pos: u32, len: u32, field: u64) -> u64 {
    let mask = make_mask(pos, len);
    (v & !mask) | ((field << pos) & mask)
}

fn mulu64(a: u64, b: u64) -> (u64, u64) {
    let r = a as u128 * b as u128;
    (r as u64, (r >> 64) as u64)
}

fn muls64(a: u64, b: u64) -> (u64, u64) {
    let r = a as i64 as i128 * b as i64 as i128;
    (r as u64, (r >> 64) as u64)
}

fn do_constant_folding_2(op: Opcode, ty: Type, x: u64, y: u64) -> u64 {
    let i32t = ty == Type::I32;
    match op {
        Opcode::Add => x.wrapping_add(y),
        Opcode::Sub => x.wrapping_sub(y),
        Opcode::Mul => x.wrapping_mul(y),
        Opcode::And | Opcode::AndVec => x & y,
        Opcode::Or | Opcode::OrVec => x | y,
        Opcode::Xor | Opcode::XorVec => x ^ y,
        Opcode::Shl => {
            if i32t {
                ((x as u32) << (y & 31)) as u64
            } else {
                x << (y & 63)
            }
        }
        Opcode::Shr => {
            if i32t {
                ((x as u32) >> (y & 31)) as u64
            } else {
                x >> (y & 63)
            }
        }
        Opcode::Sar => {
            if i32t {
                ((x as i32) >> (y & 31)) as i64 as u64
            } else {
                ((x as i64) >> (y & 63)) as u64
            }
        }
        Opcode::Rotr => {
            if i32t {
                (x as u32).rotate_right((y & 31) as u32) as u64
            } else {
                x.rotate_right((y & 63) as u32)
            }
        }
        Opcode::Rotl => {
            if i32t {
                (x as u32).rotate_left((y & 31) as u32) as u64
            } else {
                x.rotate_left((y & 63) as u32)
            }
        }
        Opcode::Not | Opcode::NotVec => !x,
        Opcode::Neg => x.wrapping_neg(),
        Opcode::Andc | Opcode::AndcVec => x & !y,
        Opcode::Orc | Opcode::OrcVec => x | !y,
        Opcode::Eqv | Opcode::EqvVec => !(x ^ y),
        Opcode::Nand | Opcode::NandVec => !(x & y),
        Opcode::Nor | Opcode::NorVec => !(x | y),
        Opcode::Clz => {
            if i32t {
                if x as u32 != 0 { (x as u32).leading_zeros() as u64 } else { y }
            } else if x != 0 {
                x.leading_zeros() as u64
            } else {
                y
            }
        }
        Opcode::Ctz => {
            if i32t {
                if x as u32 != 0 { (x as u32).trailing_zeros() as u64 } else { y }
            } else if x != 0 {
                x.trailing_zeros() as u64
            } else {
                y
            }
        }
        Opcode::Ctpop => {
            if i32t {
                (x as u32).count_ones() as u64
            } else {
                x.count_ones() as u64
            }
        }
        Opcode::Bswap16 => {
            let x = (x as u16).swap_bytes() as u64;
            if y & bswap::OS as u64 != 0 { x as u16 as i16 as i64 as u64 } else { x }
        }
        Opcode::Bswap32 => {
            let x = (x as u32).swap_bytes() as u64;
            if y & bswap::OS as u64 != 0 { x as u32 as i32 as i64 as u64 } else { x }
        }
        Opcode::Bswap64 => x.swap_bytes(),
        Opcode::ExtI32I64 => x as i32 as i64 as u64,
        Opcode::ExtuI32I64 | Opcode::ExtrlI64I32 => x as u32 as u64,
        Opcode::ExtrhI64I32 => x >> 32,
        Opcode::Muluh => {
            if i32t {
                ((x as u32 as u64) * (y as u32 as u64)) >> 32
            } else {
                mulu64(x, y).1
            }
        }
        Opcode::Mulsh => {
            if i32t {
                ((x as i32 as i64 * y as i32 as i64) >> 32) as u64
            } else {
                muls64(x, y).1
            }
        }
        Opcode::Divs => {
            if i32t {
                let d = if y as i32 != 0 { y as i32 } else { 1 };
                (x as i32).wrapping_div(d) as i64 as u64
            } else {
                let d = if y as i64 != 0 { y as i64 } else { 1 };
                (x as i64).wrapping_div(d) as u64
            }
        }
        Opcode::Divu => {
            if i32t {
                let d = if y as u32 != 0 { y as u32 } else { 1 };
                ((x as u32) / d) as u64
            } else {
                x / if y != 0 { y } else { 1 }
            }
        }
        Opcode::Rems => {
            if i32t {
                let d = if y as i32 != 0 { y as i32 } else { 1 };
                (x as i32).wrapping_rem(d) as i64 as u64
            } else {
                let d = if y as i64 != 0 { y as i64 } else { 1 };
                (x as i64).wrapping_rem(d) as u64
            }
        }
        Opcode::Remu => {
            if i32t {
                let d = if y as u32 != 0 { y as u32 } else { 1 };
                ((x as u32) % d) as u64
            } else {
                x % if y != 0 { y } else { 1 }
            }
        }
        _ => unreachable!("no constant folding for {}", op.name()),
    }
}

/// `do_constant_folding`: the result of a foldable op, sign-extended from 32 bits for I32.
pub(crate) fn do_constant_folding(op: Opcode, ty: Type, x: u64, y: u64) -> u64 {
    let r = do_constant_folding_2(op, ty, x, y);
    if ty == Type::I32 { r as i32 as i64 as u64 } else { r }
}

fn do_constant_folding_cond_eq(c: Cond) -> i32 {
    match c {
        Cond::Gt | Cond::Ltu | Cond::Lt | Cond::Gtu | Cond::Ne => 0,
        Cond::Ge | Cond::Geu | Cond::Le | Cond::Leu | Cond::Eq => 1,
        Cond::TstEq | Cond::TstNe => -1,
        Cond::Always | Cond::Never => unreachable!("constant condition"),
    }
}

fn cond_of(v: u64) -> Cond {
    Cond::from_u64(v).expect("bad condition")
}

struct Opt<'a> {
    f: &'a mut Func,
    info: Vec<Info>,
    used: Vec<bool>,
    mem: Vec<MemCopy>,
    prev_mb: Option<OpId>,
    ty: Type,
    carry_state: i32,
}

impl Opt<'_> {
    // Temp info.

    fn grow(&mut self) {
        let n = self.f.nb_temps();
        while self.info.len() < n {
            let t = Temp::from_index(self.info.len());
            self.info.push(Info { prev_copy: t, next_copy: t, z_mask: !0, o_mask: 0, s_mask: 0 });
            self.used.push(false);
        }
    }

    fn ti(&self, t: Temp) -> &Info {
        &self.info[t.index()]
    }

    fn ti_mut(&mut self, t: Temp) -> &mut Info {
        &mut self.info[t.index()]
    }

    fn arg(&self, op: OpId, i: usize) -> u64 {
        self.f.op(op).args[i]
    }

    fn set_arg(&mut self, op: OpId, i: usize, v: u64) {
        self.f.op_mut(op).args[i] = v;
    }

    fn set_opc(&mut self, op: OpId, opc: Opcode) {
        self.f.op_mut(op).opc = opc;
    }

    fn opc(&self, op: OpId) -> Opcode {
        self.f.op(op).opc
    }

    fn ai(&self, a: u64) -> Info {
        *self.ti(Temp::from_arg(a))
    }

    fn ts_is_const(&self, t: Temp) -> bool {
        let i = self.ti(t);
        i.z_mask == i.o_mask
    }

    fn arg_is_const(&self, a: u64) -> bool {
        self.ts_is_const(Temp::from_arg(a))
    }

    fn arg_const_val(&self, a: u64) -> u64 {
        self.ti(Temp::from_arg(a)).z_mask
    }

    fn arg_is_const_val(&self, a: u64, v: u64) -> bool {
        self.arg_is_const(a) && self.arg_const_val(a) == v
    }

    fn ts_is_copy(&self, t: Temp) -> bool {
        self.ti(t).next_copy != t
    }

    fn cmp_better_copy(&self, a: Temp, b: Temp) -> Temp {
        if self.f.temp(a).kind < self.f.temp(b).kind { b } else { a }
    }

    fn init_ts_info(&mut self, t: Temp) {
        self.grow();
        if self.used[t.index()] {
            return;
        }
        self.used[t.index()] = true;
        let td = self.f.temp(t);
        let (kind, val) = (td.kind, td.val as u64);
        let ti = self.ti_mut(t);
        ti.next_copy = t;
        ti.prev_copy = t;
        if kind == TempKind::Const {
            ti.z_mask = val;
            ti.o_mask = val;
            ti.s_mask = (i64::MIN >> clrsb64(val)) as u64;
        } else {
            ti.z_mask = !0;
            ti.o_mask = 0;
            ti.s_mask = 0;
        }
    }

    fn temp_readonly(&self, t: Temp) -> bool {
        matches!(self.f.temp(t).kind, TempKind::Const | TempKind::Fixed)
    }

    fn find_better_copy(&self, t: Temp) -> Temp {
        if self.temp_readonly(t) {
            return t;
        }
        let mut ret = t;
        let mut i = self.ti(t).next_copy;
        while i != t {
            ret = self.cmp_better_copy(ret, i);
            i = self.ti(i).next_copy;
        }
        ret
    }

    fn has_mem_copy(&self, t: Temp) -> bool {
        self.mem.iter().any(|m| m.ts == t)
    }

    fn move_mem_copies(&mut self, dst: Temp, src: Temp) {
        for m in &mut self.mem {
            if m.ts == src {
                m.ts = dst;
            }
        }
    }

    fn reset_ts(&mut self, t: Temp) {
        let ti = *self.ti(t);
        let (pts, nts) = (ti.prev_copy, ti.next_copy);
        self.ti_mut(nts).prev_copy = pts;
        self.ti_mut(pts).next_copy = nts;
        let ti = self.ti_mut(t);
        ti.next_copy = t;
        ti.prev_copy = t;
        ti.z_mask = !0;
        ti.o_mask = 0;
        ti.s_mask = 0;
        if self.has_mem_copy(t) {
            if t == nts {
                self.mem.retain(|m| m.ts != t);
            } else {
                let b = self.find_better_copy(nts);
                self.move_mem_copies(b, t);
            }
        }
    }

    fn reset_temp(&mut self, a: u64) {
        self.reset_ts(Temp::from_arg(a));
    }

    fn remove_mem_copy_in(&mut self, s: i64, l: i64) {
        self.mem.retain(|m| !(m.start <= l && s <= m.last));
    }

    fn remove_mem_copy_all(&mut self) {
        self.mem.clear();
    }

    fn record_mem_copy(&mut self, ty: Type, t: Temp, start: i64, last: i64) {
        let t = self.find_better_copy(t);
        self.mem.push(MemCopy { start, last, ty, ts: t });
    }

    fn ts_are_copies(&self, a: Temp, b: Temp) -> bool {
        if a == b {
            return true;
        }
        if !self.ts_is_copy(a) || !self.ts_is_copy(b) {
            return false;
        }
        let mut i = self.ti(a).next_copy;
        while i != a {
            if i == b {
                return true;
            }
            i = self.ti(i).next_copy;
        }
        false
    }

    fn args_are_copies(&self, a: u64, b: u64) -> bool {
        self.ts_are_copies(Temp::from_arg(a), Temp::from_arg(b))
    }

    fn find_mem_copy_for(&self, ty: Type, s: i64) -> Option<Temp> {
        self.mem.iter().find(|m| m.start == s && m.ty == ty).map(|m| self.find_better_copy(m.ts))
    }

    fn arg_new_constant(&mut self, val: u64) -> u64 {
        let val = if self.ty == Type::I32 { val as i32 as i64 } else { val as i64 };
        let t = self.f.constant_internal(self.ty, val);
        self.init_ts_info(t);
        t.arg()
    }

    fn insert_after(&mut self, op: OpId, opc: Opcode, n: usize) -> OpId {
        self.f.insert_after(op, opc, self.ty, n)
    }

    fn insert_before(&mut self, op: OpId, opc: Opcode, n: usize) -> OpId {
        self.f.insert_before(op, opc, self.ty, n)
    }

    fn gen_mov(&mut self, op: OpId, dst: u64, src: u64) -> bool {
        let (dt, st) = (Temp::from_arg(dst), Temp::from_arg(src));
        if self.ts_are_copies(dt, st) {
            self.f.remove_op(op);
            return true;
        }
        self.reset_ts(dt);
        let new_op = match self.ty {
            Type::I32 | Type::I64 => Opcode::Mov,
            Type::V64 | Type::V128 | Type::V256 => Opcode::MovVec,
            Type::I128 => unreachable!("no I128 moves"),
        };
        {
            let o = self.f.op_mut(op);
            o.opc = new_op;
            o.args[0] = dst;
            o.args[1] = src;
            o.nargs = 2;
        }
        let si = *self.ti(st);
        {
            let di = self.ti_mut(dt);
            di.z_mask = si.z_mask;
            di.o_mask = si.o_mask;
            di.s_mask = si.s_mask;
        }
        let (sty, dty) = (self.f.temp(st).ty, self.f.temp(dt).ty);
        if sty == dty {
            let ni = si.next_copy;
            {
                let di = self.ti_mut(dt);
                di.next_copy = ni;
                di.prev_copy = st;
            }
            self.ti_mut(ni).prev_copy = dt;
            self.ti_mut(st).next_copy = dt;
            if self.has_mem_copy(st) && self.cmp_better_copy(st, dt) == dt {
                self.move_mem_copies(dt, st);
            }
        } else if dty == Type::I32 {
            let di = self.ti_mut(dt);
            di.z_mask = di.z_mask as i32 as i64 as u64;
            di.o_mask = di.o_mask as i32 as i64 as u64;
            di.s_mask |= i32::MIN as i64 as u64;
        } else {
            let di = self.ti_mut(dt);
            di.z_mask |= make_mask(32, 32);
            di.o_mask = di.o_mask as u32 as u64;
            di.s_mask = i64::MIN as u64;
        }
        true
    }

    fn gen_movi(&mut self, op: OpId, dst: u64, val: u64) -> bool {
        let c = self.arg_new_constant(val);
        self.gen_mov(op, dst, c)
    }

    fn do_constant_folding_cond(&self, x: u64, y: u64, c: Cond) -> i32 {
        if self.arg_is_const(x) && self.arg_is_const(y) {
            let (xv, yv) = (self.arg_const_val(x), self.arg_const_val(y));
            match self.ty {
                Type::I32 => {
                    assert!(!matches!(c, Cond::Always | Cond::Never));
                    c.eval_u32(xv as u32, yv as u32) as i32
                }
                Type::I64 => {
                    assert!(!matches!(c, Cond::Always | Cond::Never));
                    c.eval_u64(xv, yv) as i32
                }
                _ => -1,
            }
        } else if self.args_are_copies(x, y) {
            do_constant_folding_cond_eq(c)
        } else if self.arg_is_const_val(y, 0) {
            match c {
                Cond::Ltu | Cond::TstNe => 0,
                Cond::Geu | Cond::TstEq => 1,
                _ => -1,
            }
        } else {
            -1
        }
    }

    fn pref_commutative(&self, a: u64) -> i32 {
        if !self.arg_is_const(a) {
            0
        } else if self.arg_const_val(a) != 0 {
            3
        } else {
            2
        }
    }

    fn swap_commutative(&mut self, dest: u64, op: OpId, i1: usize, i2: usize) -> bool {
        let (a1, a2) = (self.arg(op, i1), self.arg(op, i2));
        let sum = self.pref_commutative(a1) - self.pref_commutative(a2);
        if sum > 0 || (sum == 0 && dest == a2) {
            self.set_arg(op, i1, a2);
            self.set_arg(op, i2, a1);
            return true;
        }
        false
    }

    fn do_constant_folding_cond1(
        &mut self,
        op: OpId,
        dest: u64,
        i1: usize,
        i2: usize,
        icond: usize,
    ) -> i32 {
        let swap = self.swap_commutative(dest, op, i1, i2);
        let mut cond = cond_of(self.arg(op, icond));
        if swap {
            cond = cond.swap();
            self.set_arg(op, icond, cond as u64);
        }
        let (p1, p2) = (self.arg(op, i1), self.arg(op, i2));
        let r = self.do_constant_folding_cond(p1, p2, cond);
        if r >= 0 {
            return r;
        }
        if !cond.is_tst() {
            return -1;
        }
        let t1 = self.ai(p1);
        if self.args_are_copies(p1, p2)
            || (self.arg_is_const(p2) && (t1.z_mask & !self.arg_const_val(p2)) == 0)
        {
            let z = self.arg_new_constant(0);
            self.set_arg(op, i2, z);
            self.set_arg(op, icond, cond.tst_eqne() as u64);
            return -1;
        }
        if self.arg_is_const(p2) && (self.arg_const_val(p2) & !t1.s_mask) == 0 {
            let z = self.arg_new_constant(0);
            self.set_arg(op, i2, z);
            self.set_arg(op, icond, cond.tst_ltge() as u64);
            return -1;
        }
        // The virtual host supports the test conditions, so no AND expansion is needed.
        -1
    }

    fn init_arguments(&mut self, op: OpId, n: usize) {
        for i in 0..n {
            let t = Temp::from_arg(self.arg(op, i));
            self.init_ts_info(t);
        }
    }

    fn copy_propagate(&mut self, op: OpId, nb_oargs: usize, nb_iargs: usize) {
        for i in nb_oargs..nb_oargs + nb_iargs {
            let t = Temp::from_arg(self.arg(op, i));
            if self.ts_is_copy(t) {
                let b = self.find_better_copy(t);
                self.set_arg(op, i, b.arg());
            }
        }
    }

    fn finish_bb(&mut self) {
        self.prev_mb = None;
    }

    fn finish_ebb(&mut self) {
        self.finish_bb();
        self.used.fill(false);
        self.remove_mem_copy_all();
    }

    fn finish_folding(&mut self, op: OpId) -> bool {
        let n = self.opc(op).def().nb_oargs as usize;
        for i in 0..n {
            let a = self.arg(op, i);
            self.reset_temp(a);
        }
        true
    }

    // Folding helpers.

    fn fold_const1(&mut self, op: OpId) -> bool {
        let a1 = self.arg(op, 1);
        if self.arg_is_const(a1) {
            let t = do_constant_folding(self.opc(op), self.ty, self.arg_const_val(a1), 0);
            let d = self.arg(op, 0);
            return self.gen_movi(op, d, t);
        }
        false
    }

    fn fold_const2(&mut self, op: OpId) -> bool {
        let (a1, a2) = (self.arg(op, 1), self.arg(op, 2));
        if self.arg_is_const(a1) && self.arg_is_const(a2) {
            let t = do_constant_folding(
                self.opc(op),
                self.ty,
                self.arg_const_val(a1),
                self.arg_const_val(a2),
            );
            let d = self.arg(op, 0);
            return self.gen_movi(op, d, t);
        }
        false
    }

    fn fold_commutative(&mut self, op: OpId) -> bool {
        let d = self.arg(op, 0);
        self.swap_commutative(d, op, 1, 2);
        false
    }

    fn fold_const2_commutative(&mut self, op: OpId) -> bool {
        let d = self.arg(op, 0);
        self.swap_commutative(d, op, 1, 2);
        self.fold_const2(op)
    }

    fn fold_masks_zosa_int(
        &mut self,
        op: OpId,
        mut z_mask: u64,
        mut o_mask: u64,
        mut s_mask: u64,
        mut a_mask: u64,
    ) -> bool {
        debug_assert_eq!(self.opc(op).def().nb_oargs, 1);
        if self.ty == Type::I32 {
            z_mask = z_mask as i32 as i64 as u64;
            o_mask = o_mask as i32 as i64 as u64;
            s_mask |= i32::MIN as i64 as u64;
            a_mask = a_mask as u32 as u64;
        }
        debug_assert_eq!(o_mask & !z_mask, 0);
        if z_mask == o_mask {
            let d = self.arg(op, 0);
            return self.gen_movi(op, d, o_mask);
        }
        if a_mask == 0 {
            let (d, s) = (self.arg(op, 0), self.arg(op, 1));
            return self.gen_mov(op, d, s);
        }
        let t = Temp::from_arg(self.arg(op, 0));
        self.reset_ts(t);
        let mut rep = (!s_mask).leading_zeros();
        rep = rep.max(z_mask.leading_zeros());
        rep = rep.max((!o_mask).leading_zeros());
        let rep = rep.saturating_sub(1);
        let ti = self.ti_mut(t);
        ti.z_mask = z_mask;
        ti.o_mask = o_mask;
        ti.s_mask = (i64::MIN >> rep) as u64;
        false
    }

    fn fold_masks_zosa(&mut self, op: OpId, z: u64, o: u64, s: u64, a: u64) -> bool {
        self.fold_masks_zosa_int(op, z, o, s, a);
        true
    }

    fn fold_masks_zos(&mut self, op: OpId, z: u64, o: u64, s: u64) -> bool {
        self.fold_masks_zosa(op, z, o, s, !0)
    }

    fn fold_masks_zo(&mut self, op: OpId, z: u64, o: u64) -> bool {
        self.fold_masks_zosa(op, z, o, 0, !0)
    }

    fn fold_masks_zs(&mut self, op: OpId, z: u64, s: u64) -> bool {
        self.fold_masks_zosa(op, z, 0, s, !0)
    }

    fn fold_masks_z(&mut self, op: OpId, z: u64) -> bool {
        self.fold_masks_zosa(op, z, 0, 0, !0)
    }

    fn fold_masks_s(&mut self, op: OpId, s: u64) -> bool {
        self.fold_masks_zosa(op, !0, 0, s, !0)
    }

    fn fold_to_not(&mut self, op: OpId, idx: usize) -> bool {
        let not_op = match self.ty {
            Type::I32 | Type::I64 => Opcode::Not,
            Type::V64 | Type::V128 | Type::V256 => Opcode::NotVec,
            Type::I128 => unreachable!(),
        };
        self.set_opc(op, not_op);
        let a = self.arg(op, idx);
        self.set_arg(op, 1, a);
        self.fold_not(op)
    }

    fn fold_ix_to_i(&mut self, op: OpId, i: u64) -> bool {
        if self.arg_is_const_val(self.arg(op, 1), i) {
            let d = self.arg(op, 0);
            return self.gen_movi(op, d, i);
        }
        false
    }

    fn fold_ix_to_not(&mut self, op: OpId, i: u64) -> bool {
        if self.arg_is_const_val(self.arg(op, 1), i) {
            return self.fold_to_not(op, 2);
        }
        false
    }

    fn fold_xi_to_i(&mut self, op: OpId, i: u64) -> bool {
        if self.arg_is_const_val(self.arg(op, 2), i) {
            let d = self.arg(op, 0);
            return self.gen_movi(op, d, i);
        }
        false
    }

    fn fold_xi_to_x(&mut self, op: OpId, i: u64) -> bool {
        if self.arg_is_const_val(self.arg(op, 2), i) {
            let (d, s) = (self.arg(op, 0), self.arg(op, 1));
            return self.gen_mov(op, d, s);
        }
        false
    }

    fn fold_xi_to_not(&mut self, op: OpId, i: u64) -> bool {
        if self.arg_is_const_val(self.arg(op, 2), i) {
            return self.fold_to_not(op, 1);
        }
        false
    }

    fn fold_xx_to_i(&mut self, op: OpId, i: u64) -> bool {
        if self.args_are_copies(self.arg(op, 1), self.arg(op, 2)) {
            let d = self.arg(op, 0);
            return self.gen_movi(op, d, i);
        }
        false
    }

    fn fold_xx_to_x(&mut self, op: OpId) -> bool {
        if self.args_are_copies(self.arg(op, 1), self.arg(op, 2)) {
            let (d, s) = (self.arg(op, 0), self.arg(op, 1));
            return self.gen_mov(op, d, s);
        }
        false
    }

    // The folders, in QEMU's alphabetical order.

    fn fold_add(&mut self, op: OpId) -> bool {
        if self.fold_const2_commutative(op) || self.fold_xi_to_x(op, 0) {
            return true;
        }
        self.finish_folding(op)
    }

    fn fold_add_vec(&mut self, op: OpId) -> bool {
        if self.fold_commutative(op) || self.fold_xi_to_x(op, 0) {
            return true;
        }
        self.finish_folding(op)
    }

    fn squash_prev_carryout(&mut self, op: OpId) {
        let op = self.f.prev_op(op).expect("carry-in op without a carry-out op before it");
        match self.opc(op) {
            Opcode::Addco => {
                self.set_opc(op, Opcode::Add);
                self.fold_add(op);
            }
            Opcode::Addcio => self.set_opc(op, Opcode::Addci),
            Opcode::Addc1o => {
                self.set_opc(op, Opcode::Add);
                let a2 = self.arg(op, 2);
                if self.arg_is_const(a2) {
                    let c = self.arg_new_constant(self.arg_const_val(a2).wrapping_add(1));
                    self.set_arg(op, 2, c);
                    self.fold_add(op);
                } else {
                    let ret = self.arg(op, 0);
                    let n = self.insert_after(op, Opcode::Add, 3);
                    let c = self.arg_new_constant(1);
                    self.set_arg(n, 0, ret);
                    self.set_arg(n, 1, ret);
                    self.set_arg(n, 2, c);
                }
            }
            o => unreachable!("unexpected carry producer {}", o.name()),
        }
    }

    fn fold_addci(&mut self, op: OpId) -> bool {
        self.fold_commutative(op);
        if self.carry_state < 0 {
            return self.finish_folding(op);
        }
        self.squash_prev_carryout(op);
        self.set_opc(op, Opcode::Add);
        if self.carry_state > 0 {
            let a2 = self.arg(op, 2);
            if self.arg_is_const(a2) {
                let c = self.arg_new_constant(self.arg_const_val(a2).wrapping_add(1));
                self.set_arg(op, 2, c);
            } else {
                let op2 = self.insert_before(op, Opcode::Add, 3);
                for i in 0..3 {
                    let a = self.arg(op, i);
                    self.set_arg(op2, i, a);
                }
                self.fold_add(op2);
                let a0 = self.arg(op, 0);
                self.set_arg(op, 1, a0);
                let c = self.arg_new_constant(1);
                self.set_arg(op, 2, c);
            }
        }
        self.carry_state = -1;
        self.fold_add(op)
    }

    fn fold_addcio(&mut self, op: OpId) -> bool {
        self.fold_commutative(op);
        let t1 = self.ai(self.arg(op, 1));
        let t2 = self.ai(self.arg(op, 2));
        let mut carry_out = -1;
        if let Some(sum) = t1.z_mask.checked_add(t2.z_mask) {
            if sum.checked_add((self.carry_state != 0) as u64).is_some() {
                carry_out = 0;
            }
        }
        if self.carry_state < 0 {
            self.carry_state = carry_out;
            return self.finish_folding(op);
        }
        self.squash_prev_carryout(op);
        if self.carry_state != 0 {
            let max = if self.ty == Type::I32 { u32::MAX as u64 } else { u64::MAX };
            let mut done = false;
            if t2.z_mask == t2.o_mask {
                let v = t2.z_mask & max;
                if v < max {
                    let c = self.arg_new_constant(v + 1);
                    self.set_arg(op, 2, c);
                    done = true;
                } else {
                    carry_out = 1;
                }
            }
            if !done && t1.z_mask == t1.o_mask {
                let v = t1.z_mask & max;
                if v < max {
                    let c = self.arg_new_constant(v + 1);
                    self.set_arg(op, 1, c);
                    done = true;
                } else {
                    carry_out = 1;
                }
            }
            if !done {
                self.set_opc(op, Opcode::Addc1o);
                self.carry_state = carry_out;
                return self.finish_folding(op);
            }
        }
        self.set_opc(op, Opcode::Addco);
        self.fold_addco(op)
    }

    fn fold_addco(&mut self, op: OpId) -> bool {
        self.fold_commutative(op);
        let t1 = self.ai(self.arg(op, 1));
        let t2 = self.ai(self.arg(op, 2));
        let mut carry_out = -1;
        if t2.z_mask == t2.o_mask {
            let v2 = t2.z_mask;
            if t1.z_mask == t1.o_mask {
                carry_out = t1.z_mask.overflowing_add(v2).1 as i32;
            } else if v2 == 0 {
                carry_out = 0;
            }
        } else if t1.z_mask.checked_add(t2.z_mask).is_some() {
            carry_out = 0;
        }
        self.carry_state = carry_out;
        self.finish_folding(op)
    }

    fn fold_and(&mut self, op: OpId) -> bool {
        if self.fold_const2_commutative(op) {
            return true;
        }
        let t1 = self.ai(self.arg(op, 1));
        let t2 = self.ai(self.arg(op, 2));
        let z = t1.z_mask & t2.z_mask;
        let o = t1.o_mask & t2.o_mask;
        let s = t1.s_mask & t2.s_mask;
        let a = t1.z_mask & !t2.o_mask;
        if !self.fold_masks_zosa_int(op, z, o, s, a) {
            if self.opc(op) == Opcode::And && t2.z_mask == t2.o_mask {
                let val = t2.z_mask;
                if val & val.wrapping_add(1) == 0 {
                    // The virtual host accepts extract at every length.
                    let len = (!val).trailing_zeros();
                    self.set_opc(op, Opcode::Extract);
                    self.set_arg(op, 2, 0);
                    self.set_arg(op, 3, len as u64);
                    self.f.op_mut(op).nargs = 4;
                }
            } else {
                self.fold_xx_to_x(op);
            }
        }
        true
    }

    fn fold_andc(&mut self, op: OpId) -> bool {
        if self.fold_const2(op) {
            return true;
        }
        let t1 = self.ai(self.arg(op, 1));
        let t2 = self.ai(self.arg(op, 2));
        if t2.z_mask == t2.o_mask {
            let o = match self.ty {
                Type::I32 | Type::I64 => Opcode::And,
                _ => Opcode::AndVec,
            };
            self.set_opc(op, o);
            let c = self.arg_new_constant(!t2.z_mask);
            self.set_arg(op, 2, c);
            return self.fold_and(op);
        }
        if self.fold_xx_to_i(op, 0) || self.fold_ix_to_not(op, !0) {
            return true;
        }
        let z = t1.z_mask & !t2.o_mask;
        let o = t1.o_mask & !t2.z_mask;
        let s = t1.s_mask & t2.s_mask;
        let a = t1.z_mask & t2.z_mask;
        self.fold_masks_zosa(op, z, o, s, a)
    }

    fn fold_bitsel_vec(&mut self, op: OpId) -> bool {
        let (a1, a2, a3) = (self.arg(op, 1), self.arg(op, 2), self.arg(op, 3));
        if self.args_are_copies(a2, a3) {
            let d = self.arg(op, 0);
            return self.gen_mov(op, d, a2);
        }
        if self.arg_is_const(a2) && self.arg_is_const(a3) {
            let tv = self.arg_const_val(a2);
            let fv = self.arg_const_val(a3);
            if tv == !0 && fv == 0 {
                let d = self.arg(op, 0);
                return self.gen_mov(op, d, a1);
            }
            if tv == 0 && fv == !0 {
                self.set_opc(op, Opcode::NotVec);
                return self.fold_not(op);
            }
        }
        if self.arg_is_const(a2) {
            let tv = self.arg_const_val(a2);
            if tv == !0 {
                self.set_opc(op, Opcode::OrVec);
                self.set_arg(op, 2, a3);
                return self.fold_or(op);
            }
            if tv == 0 {
                self.set_opc(op, Opcode::AndcVec);
                self.set_arg(op, 2, a1);
                self.set_arg(op, 1, a3);
                return self.fold_andc(op);
            }
        }
        if self.arg_is_const(a3) {
            let fv = self.arg_const_val(a3);
            if fv == 0 {
                self.set_opc(op, Opcode::AndVec);
                return self.fold_and(op);
            }
            if fv == !0 {
                self.set_opc(op, Opcode::OrcVec);
                self.set_arg(op, 2, a1);
                self.set_arg(op, 1, a2);
                return self.fold_orc(op);
            }
        }
        self.finish_folding(op)
    }

    fn fold_brcond(&mut self, op: OpId) -> bool {
        let i = self.do_constant_folding_cond1(op, NO_DEST, 0, 1, 2);
        if i == 0 {
            self.f.remove_op(op);
            return true;
        }
        if i > 0 {
            // The label keeps this op as its user, now through argument 0.
            let label = self.arg(op, 3);
            let o = self.f.op_mut(op);
            o.opc = Opcode::Br;
            o.args[0] = label;
            o.nargs = 1;
            self.finish_ebb();
        } else {
            self.finish_bb();
        }
        true
    }

    fn fold_bswap(&mut self, op: OpId) -> bool {
        let t1 = self.ai(self.arg(op, 1));
        let flags = self.arg(op, 2);
        if t1.z_mask == t1.o_mask {
            let v = do_constant_folding(self.opc(op), self.ty, t1.z_mask, flags);
            let d = self.arg(op, 0);
            return self.gen_movi(op, d, v);
        }
        let mut z = t1.z_mask;
        let mut o = t1.o_mask;
        let mut s = 0;
        match self.opc(op) {
            Opcode::Bswap16 => {
                z = (z as u16).swap_bytes() as u64;
                o = (o as u16).swap_bytes() as u64;
                if flags & bswap::OS as u64 != 0 {
                    z = z as u16 as i16 as i64 as u64;
                    o = o as u16 as i16 as i64 as u64;
                    s = i16::MIN as i64 as u64;
                } else if flags & bswap::OZ as u64 == 0 {
                    z |= make_mask(16, 48);
                }
            }
            Opcode::Bswap32 => {
                z = (z as u32).swap_bytes() as u64;
                o = (o as u32).swap_bytes() as u64;
                if flags & bswap::OS as u64 != 0 {
                    z = z as u32 as i32 as i64 as u64;
                    o = o as u32 as i32 as i64 as u64;
                    s = i32::MIN as i64 as u64;
                } else if flags & bswap::OZ as u64 == 0 {
                    z |= make_mask(32, 32);
                }
            }
            Opcode::Bswap64 => {
                z = z.swap_bytes();
                o = o.swap_bytes();
            }
            _ => unreachable!(),
        }
        self.fold_masks_zos(op, z, o, s)
    }

    fn fold_call(&mut self, op: OpId) -> bool {
        let o = *self.f.op(op);
        let (nb_oargs, nb_iargs) = (o.callo as usize, o.calli as usize);
        self.init_arguments(op, nb_oargs + nb_iargs);
        self.copy_propagate(op, nb_oargs, nb_iargs);
        let flags = self.f.helper_info(o.call_helper()).flags;
        use crate::types::call_flags::{NO_READ_GLOBALS, NO_SIDE_EFFECTS, NO_WRITE_GLOBALS};
        if flags & (NO_READ_GLOBALS | NO_WRITE_GLOBALS) == 0 {
            for i in 0..self.f.nb_globals() {
                if self.used[i] {
                    self.reset_ts(Temp::from_index(i));
                }
            }
        }
        if flags & NO_SIDE_EFFECTS == 0 {
            self.remove_mem_copy_all();
        }
        for i in 0..nb_oargs {
            let a = self.arg(op, i);
            self.reset_temp(a);
        }
        self.prev_mb = None;
        true
    }

    fn fold_cmp_vec(&mut self, op: OpId) -> bool {
        if self.swap_commutative(NO_DEST, op, 1, 2) {
            let c = cond_of(self.arg(op, 3)).swap();
            self.set_arg(op, 3, c as u64);
        }
        self.finish_folding(op)
    }

    fn fold_cmpsel_vec(&mut self, op: OpId) -> bool {
        if self.args_are_copies(self.arg(op, 3), self.arg(op, 4)) {
            let (d, s) = (self.arg(op, 0), self.arg(op, 3));
            return self.gen_mov(op, d, s);
        }
        if self.swap_commutative(NO_DEST, op, 1, 2) {
            let c = cond_of(self.arg(op, 5)).swap();
            self.set_arg(op, 5, c as u64);
        }
        let d = self.arg(op, 0);
        if self.swap_commutative(d, op, 4, 3) {
            let c = cond_of(self.arg(op, 5)).invert();
            self.set_arg(op, 5, c as u64);
        }
        self.finish_folding(op)
    }

    fn fold_count_zeros(&mut self, op: OpId) -> bool {
        let t1 = self.ai(self.arg(op, 1));
        let t2 = self.ai(self.arg(op, 2));
        if t1.z_mask == t1.o_mask {
            let t = t1.z_mask;
            let d = self.arg(op, 0);
            if t != 0 {
                let t = do_constant_folding(self.opc(op), self.ty, t, 0);
                return self.gen_movi(op, d, t);
            }
            let s = self.arg(op, 2);
            return self.gen_mov(op, d, s);
        }
        let mut z: u64 = match self.ty {
            Type::I32 => 31,
            Type::I64 => 63,
            _ => unreachable!(),
        };
        let mut s = !z;
        z |= t2.z_mask;
        s &= t2.s_mask;
        self.fold_masks_zs(op, z, s)
    }

    fn fold_ctpop(&mut self, op: OpId) -> bool {
        if self.fold_const1(op) {
            return true;
        }
        let z = match self.ty {
            Type::I32 => 32 | 31,
            Type::I64 => 64 | 63,
            _ => unreachable!(),
        };
        self.fold_masks_z(op, z)
    }

    fn fold_deposit(&mut self, op: OpId) -> bool {
        let (arg1, arg2) = (self.arg(op, 1), self.arg(op, 2));
        let ofs = self.arg(op, 3) as u32;
        let len = self.arg(op, 4) as u32;
        let t1 = self.ai(arg1);
        let t2 = self.ai(arg2);
        if t1.z_mask == t1.o_mask && t2.z_mask == t2.o_mask {
            let d = self.arg(op, 0);
            return self.gen_movi(op, d, deposit64(t1.z_mask, ofs, len, t2.z_mask));
        }
        let width = self.ty.bits();
        let type_mask = make_mask(0, width);
        let len_mask = make_mask(0, len);

        if t2.z_mask & len_mask == 0 {
            self.set_opc(op, Opcode::And);
            let c = self.arg_new_constant(!(len_mask << ofs));
            self.set_arg(op, 2, c);
            self.f.op_mut(op).nargs = 3;
            return self.fold_and(op);
        }
        if t2.o_mask & len_mask == len_mask {
            self.set_opc(op, Opcode::Or);
            let c = self.arg_new_constant(len_mask << ofs);
            self.set_arg(op, 2, c);
            self.f.op_mut(op).nargs = 3;
            return self.fold_or(op);
        }

        // The virtual host accepts deposit at every position and length.
        let z = deposit64(t1.z_mask, ofs, len, t2.z_mask);
        let o = deposit64(t1.o_mask, ofs, len, t2.o_mask);
        let s =
            if ofs + len < width { t1.s_mask & !make_mask(0, ofs + len) } else { t2.s_mask << ofs };

        if t1.z_mask == t1.o_mask && t1.z_mask == 0 {
            if ofs == 0 {
                self.set_opc(op, Opcode::And);
                self.set_arg(op, 1, arg2);
                let c = self.arg_new_constant(len_mask);
                self.set_arg(op, 2, c);
                self.f.op_mut(op).nargs = 3;
                return self.fold_and(op);
            }
            let need_mask = ((t2.z_mask & !len_mask) << ofs) & type_mask;
            if need_mask == 0 {
                self.set_opc(op, Opcode::Shl);
                self.set_arg(op, 1, arg2);
                let c = self.arg_new_constant(ofs as u64);
                self.set_arg(op, 2, c);
                self.f.op_mut(op).nargs = 3;
            }
        }
        self.fold_masks_zos(op, z, o, s)
    }

    fn fold_divide(&mut self, op: OpId) -> bool {
        if self.fold_const2(op) || self.fold_xi_to_x(op, 1) {
            return true;
        }
        self.finish_folding(op)
    }

    fn fold_dup(&mut self, op: OpId) -> bool {
        let a1 = self.arg(op, 1);
        if self.arg_is_const(a1) {
            let t = dup_const(self.f.op(op).vece as u32, self.arg_const_val(a1));
            let d = self.arg(op, 0);
            return self.gen_movi(op, d, t);
        }
        self.finish_folding(op)
    }

    fn fold_eqv(&mut self, op: OpId) -> bool {
        if self.fold_const2_commutative(op) {
            return true;
        }
        let t2 = self.ai(self.arg(op, 2));
        if t2.z_mask == t2.o_mask {
            let o = match self.ty {
                Type::I32 | Type::I64 => Opcode::Xor,
                _ => Opcode::XorVec,
            };
            self.set_opc(op, o);
            let c = self.arg_new_constant(!t2.z_mask);
            self.set_arg(op, 2, c);
            return self.fold_xor(op);
        }
        let t1 = self.ai(self.arg(op, 1));
        let z = (t1.z_mask | !t2.o_mask) & (t2.z_mask | !t1.o_mask);
        let o = !(t1.z_mask | t2.z_mask) | (t1.o_mask & t2.o_mask);
        let s = t1.s_mask & t2.s_mask;
        self.fold_masks_zos(op, z, o, s)
    }

    fn fold_extract(&mut self, op: OpId) -> bool {
        let t1 = self.ai(self.arg(op, 1));
        let pos = self.arg(op, 2) as u32;
        let len = self.arg(op, 3) as u32;
        if t1.z_mask == t1.o_mask {
            let d = self.arg(op, 0);
            return self.gen_movi(op, d, extract64(t1.z_mask, pos, len));
        }
        let z = extract64(t1.z_mask, pos, len);
        let o = extract64(t1.o_mask, pos, len);
        let a = if pos != 0 { !0 } else { t1.z_mask ^ z };
        self.fold_masks_zosa(op, z, o, 0, a)
    }

    fn fold_extract2(&mut self, op: OpId) -> bool {
        let t1 = self.ai(self.arg(op, 1));
        let t2 = self.ai(self.arg(op, 2));
        let (mut z1, mut z2, mut o1, mut o2) = (t1.z_mask, t2.z_mask, t1.o_mask, t2.o_mask);
        let shr = self.arg(op, 3) as u32;
        let shl;
        if self.ty == Type::I32 {
            z1 = ((z1 as u32) >> shr) as u64;
            o1 = ((o1 as u32) >> shr) as u64;
            shl = 32 - shr;
            z2 = ((z2 as i32) << shl) as i64 as u64;
            o2 = ((o2 as i32) << shl) as i64 as u64;
        } else {
            z1 >>= shr;
            o1 >>= shr;
            shl = 64 - shr;
            z2 <<= shl;
            o2 <<= shl;
        }
        let zr = z1 | z2;
        let or = o1 | o2;
        if zr == or {
            let d = self.arg(op, 0);
            return self.gen_movi(op, d, zr);
        }
        if z2 == 0 {
            self.set_opc(op, Opcode::Shr);
            let c = self.arg_new_constant(shr as u64);
            self.set_arg(op, 2, c);
            self.f.op_mut(op).nargs = 3;
        } else if z1 == 0 {
            self.set_opc(op, Opcode::Shl);
            let a2 = self.arg(op, 2);
            self.set_arg(op, 1, a2);
            let c = self.arg_new_constant(shl as u64);
            self.set_arg(op, 2, c);
            self.f.op_mut(op).nargs = 3;
        }
        self.fold_masks_zo(op, zr, or)
    }

    fn fold_exts(&mut self, op: OpId) -> bool {
        if self.fold_const1(op) {
            return true;
        }
        let t1 = self.ai(self.arg(op, 1));
        let s = t1.s_mask | i32::MIN as i64 as u64;
        let z = t1.z_mask as i32 as i64 as u64;
        let o = t1.o_mask as i32 as i64 as u64;
        self.fold_masks_zos(op, z, o, s)
    }

    fn fold_extu(&mut self, op: OpId) -> bool {
        if self.fold_const1(op) {
            return true;
        }
        let t1 = self.ai(self.arg(op, 1));
        let (z, o) = match self.opc(op) {
            Opcode::ExtrlI64I32 | Opcode::ExtuI32I64 => {
                (t1.z_mask as u32 as u64, t1.o_mask as u32 as u64)
            }
            Opcode::ExtrhI64I32 => (t1.z_mask >> 32, t1.o_mask >> 32),
            _ => unreachable!(),
        };
        self.fold_masks_zo(op, z, o)
    }

    fn fold_mb(&mut self, op: OpId) -> bool {
        if let Some(p) = self.prev_mb {
            let a = self.arg(op, 0);
            self.f.op_mut(p).args[0] |= a;
            self.f.remove_op(op);
        } else {
            self.prev_mb = Some(op);
        }
        true
    }

    fn fold_mov(&mut self, op: OpId) -> bool {
        let (d, s) = (self.arg(op, 0), self.arg(op, 1));
        self.gen_mov(op, d, s)
    }

    fn fold_movcond(&mut self, op: OpId) -> bool {
        if self.args_are_copies(self.arg(op, 3), self.arg(op, 4)) {
            let (d, s) = (self.arg(op, 0), self.arg(op, 3));
            return self.gen_mov(op, d, s);
        }
        let d = self.arg(op, 0);
        if self.swap_commutative(d, op, 4, 3) {
            let c = cond_of(self.arg(op, 5)).invert();
            self.set_arg(op, 5, c as u64);
        }
        let i = self.do_constant_folding_cond1(op, NO_DEST, 1, 2, 5);
        if i >= 0 {
            let s = self.arg(op, (4 - i) as usize);
            return self.gen_mov(op, d, s);
        }
        let tt = self.ai(self.arg(op, 3));
        let ft = self.ai(self.arg(op, 4));
        let z = tt.z_mask | ft.z_mask;
        let o = tt.o_mask & ft.o_mask;
        let s = tt.s_mask & ft.s_mask;
        if tt.z_mask == tt.o_mask && ft.z_mask == ft.o_mask {
            let (tv, fv) = (tt.z_mask, ft.z_mask);
            let cond = cond_of(self.arg(op, 5));
            let change = if tv == 1 && fv == 0 {
                Some((Opcode::Setcond, cond))
            } else if fv == 1 && tv == 0 {
                Some((Opcode::Setcond, cond.invert()))
            } else if tv == !0 && fv == 0 {
                Some((Opcode::Negsetcond, cond))
            } else if fv == !0 && tv == 0 {
                Some((Opcode::Negsetcond, cond.invert()))
            } else {
                None
            };
            if let Some((opc, c)) = change {
                self.set_opc(op, opc);
                self.set_arg(op, 3, c as u64);
                self.f.op_mut(op).nargs = 4;
            }
        }
        self.fold_masks_zos(op, z, o, s)
    }

    fn fold_mul(&mut self, op: OpId) -> bool {
        if self.fold_const2(op) || self.fold_xi_to_i(op, 0) || self.fold_xi_to_x(op, 1) {
            return true;
        }
        self.finish_folding(op)
    }

    fn fold_mul_highpart(&mut self, op: OpId) -> bool {
        if self.fold_const2_commutative(op) || self.fold_xi_to_i(op, 0) {
            return true;
        }
        self.finish_folding(op)
    }

    fn fold_multiply2(&mut self, op: OpId) -> bool {
        let d = self.arg(op, 0);
        self.swap_commutative(d, op, 2, 3);
        let a3 = self.arg(op, 3);
        if self.arg_is_const(a3) {
            let b = self.arg_const_val(a3);
            let rl = self.arg(op, 0);
            let rh = self.arg(op, 1);
            let a2 = self.arg(op, 2);
            let opc = self.opc(op);
            if self.arg_is_const(a2) {
                let a = self.arg_const_val(a2);
                let (l, h) = match opc {
                    Opcode::Mulu2 => {
                        if self.ty == Type::I32 {
                            let l = (a as u32 as u64) * (b as u32 as u64);
                            ((l as i32) as i64 as u64, ((l >> 32) as i32) as i64 as u64)
                        } else {
                            mulu64(a, b)
                        }
                    }
                    Opcode::Muls2 => {
                        if self.ty == Type::I32 {
                            let l = (a as i32 as i64) * (b as i32 as i64);
                            ((l as i32) as i64 as u64, (l >> 32) as u64)
                        } else {
                            muls64(a, b)
                        }
                    }
                    _ => unreachable!(),
                };
                let op2 = self.insert_before(op, Opcode::Discard, 2);
                self.gen_movi(op, rl, l);
                self.gen_movi(op2, rh, h);
                return true;
            }
            if b == 0 {
                let op2 = self.insert_before(op, Opcode::Discard, 2);
                self.gen_movi(op2, rl, 0);
                self.gen_movi(op, rh, 0);
                return true;
            }
            if b == 1 {
                let op2 = self.insert_before(op, Opcode::Discard, 2);
                self.gen_mov(op2, rl, a2);
                match opc {
                    Opcode::Mulu2 => {
                        self.gen_movi(op, rh, 0);
                    }
                    Opcode::Muls2 => {
                        self.set_opc(op, Opcode::Sar);
                        self.set_arg(op, 0, rh);
                        self.set_arg(op, 1, rl);
                        let c = self.arg_new_constant(self.ty.bits() as u64 - 1);
                        self.set_arg(op, 2, c);
                        self.f.op_mut(op).nargs = 3;
                    }
                    _ => unreachable!(),
                }
                return true;
            }
        }
        self.finish_folding(op)
    }

    fn fold_nand(&mut self, op: OpId) -> bool {
        if self.fold_const2_commutative(op) || self.fold_xi_to_not(op, !0) {
            return true;
        }
        let t1 = self.ai(self.arg(op, 1));
        let t2 = self.ai(self.arg(op, 2));
        let z = !(t1.o_mask & t2.o_mask);
        let o = !(t1.z_mask & t2.z_mask);
        let s = t1.s_mask & t2.s_mask;
        self.fold_masks_zos(op, z, o, s)
    }

    fn fold_neg_no_const(&mut self, op: OpId) -> bool {
        let z = self.ai(self.arg(op, 1)).z_mask;
        let z = (z & z.wrapping_neg()).wrapping_neg();
        self.fold_masks_z(op, z)
    }

    fn fold_neg(&mut self, op: OpId) -> bool {
        self.fold_const1(op) || self.fold_neg_no_const(op)
    }

    fn fold_nor(&mut self, op: OpId) -> bool {
        if self.fold_const2_commutative(op) || self.fold_xi_to_not(op, 0) {
            return true;
        }
        let t1 = self.ai(self.arg(op, 1));
        let t2 = self.ai(self.arg(op, 2));
        let z = !(t1.o_mask | t2.o_mask);
        let o = !(t1.z_mask | t2.z_mask);
        let s = t1.s_mask & t2.s_mask;
        self.fold_masks_zos(op, z, o, s)
    }

    fn fold_not(&mut self, op: OpId) -> bool {
        if self.fold_const1(op) {
            return true;
        }
        let t1 = self.ai(self.arg(op, 1));
        self.f.op_mut(op).nargs = 2;
        self.fold_masks_zos(op, !t1.o_mask, !t1.z_mask, t1.s_mask)
    }

    fn fold_or(&mut self, op: OpId) -> bool {
        if self.fold_const2_commutative(op) || self.fold_xi_to_x(op, 0) || self.fold_xx_to_x(op) {
            return true;
        }
        let t1 = self.ai(self.arg(op, 1));
        let t2 = self.ai(self.arg(op, 2));
        let z = t1.z_mask | t2.z_mask;
        let o = t1.o_mask | t2.o_mask;
        let s = t1.s_mask & t2.s_mask;
        let a = !t1.o_mask & t2.z_mask;
        self.fold_masks_zosa(op, z, o, s, a)
    }

    fn fold_orc(&mut self, op: OpId) -> bool {
        if self.fold_const2(op) {
            return true;
        }
        let t2 = self.ai(self.arg(op, 2));
        if t2.z_mask == t2.o_mask {
            let o = match self.ty {
                Type::I32 | Type::I64 => Opcode::Or,
                _ => Opcode::OrVec,
            };
            self.set_opc(op, o);
            let c = self.arg_new_constant(!t2.z_mask);
            self.set_arg(op, 2, c);
            return self.fold_or(op);
        }
        if self.fold_xx_to_i(op, !0) || self.fold_ix_to_not(op, 0) {
            return true;
        }
        let t1 = self.ai(self.arg(op, 1));
        let z = t1.z_mask | !t2.o_mask;
        let o = t1.o_mask | !t2.z_mask;
        let s = t1.s_mask & t2.s_mask;
        let a = !t1.o_mask & !t2.o_mask;
        self.fold_masks_zosa(op, z, o, s, a)
    }

    fn fold_qemu_ld_1reg(&mut self, op: OpId) -> bool {
        let def = self.opc(op).def();
        let oi = MemOpIdx(self.arg(op, (def.nb_oargs + def.nb_iargs) as usize) as u32);
        let mop = oi.memop();
        let width = 8 * mop.size_bytes();
        let mut z = !0u64;
        let mut s = 0u64;
        if width < 64 {
            if mop.is_signed() {
                s = make_mask(width - 1, 64 - (width - 1));
            } else {
                z = make_mask(0, width);
            }
        }
        self.prev_mb = None;
        self.fold_masks_zs(op, z, s)
    }

    fn fold_qemu_ld_2reg(&mut self, op: OpId) -> bool {
        self.prev_mb = None;
        self.finish_folding(op)
    }

    fn fold_qemu_st(&mut self) -> bool {
        self.prev_mb = None;
        true
    }

    fn fold_remainder(&mut self, op: OpId) -> bool {
        if self.fold_const2(op) || self.fold_xx_to_i(op, 0) {
            return true;
        }
        self.finish_folding(op)
    }

    /// Returns 1 if finished, -1 if simplified, 0 if unchanged.
    fn fold_setcond_zmask(&mut self, op: OpId, neg: bool) -> i32 {
        let a2 = self.arg(op, 2);
        if !self.arg_is_const(a2) {
            return 0;
        }
        let mut a_zmask = self.ai(self.arg(op, 1)).z_mask;
        let mut b_val = self.arg_const_val(a2);
        let cond = cond_of(self.arg(op, 3));
        if self.ty == Type::I32 {
            a_zmask = a_zmask as u32 as u64;
            b_val = b_val as u32 as u64;
        }
        if a_zmask < b_val {
            let inv = match cond {
                Cond::Ne | Cond::Leu | Cond::Ltu => Some(true),
                Cond::Gtu | Cond::Geu | Cond::Eq => Some(false),
                _ => None,
            };
            if let Some(inv) = inv {
                let d = self.arg(op, 0);
                let v = if neg { (inv as u64).wrapping_neg() } else { inv as u64 };
                return self.gen_movi(op, d, v) as i32;
            }
        }
        if a_zmask <= 1 {
            let (convert, inv) = match cond {
                Cond::Eq => (b_val == 0, true),
                Cond::Ne => (b_val == 0, false),
                Cond::Ltu | Cond::TstEq => (b_val == 1, true),
                Cond::Geu | Cond::TstNe => (b_val == 1, false),
                _ => (false, false),
            };
            if convert {
                if !inv && !neg {
                    let (d, s) = (self.arg(op, 0), self.arg(op, 1));
                    return self.gen_mov(op, d, s) as i32;
                }
                if !inv {
                    self.set_opc(op, Opcode::Neg);
                    self.f.op_mut(op).nargs = 2;
                } else if neg {
                    self.set_opc(op, Opcode::Add);
                    let c = self.arg_new_constant(!0);
                    self.set_arg(op, 2, c);
                    self.f.op_mut(op).nargs = 3;
                } else {
                    self.set_opc(op, Opcode::Xor);
                    let c = self.arg_new_constant(1);
                    self.set_arg(op, 2, c);
                    self.f.op_mut(op).nargs = 3;
                }
                return -1;
            }
        }
        0
    }

    fn fold_setcond_tst_pow2(&mut self, op: OpId, neg: bool) {
        let cond = cond_of(self.arg(op, 3));
        let src2 = self.arg(op, 2);
        if !cond.is_tst() || !self.arg_is_const(src2) {
            return;
        }
        let val = self.arg_const_val(src2);
        if !val.is_power_of_two() {
            return;
        }
        let sh = val.trailing_zeros() as u64;
        let ret = self.arg(op, 0);
        let src1 = self.arg(op, 1);
        let inv = cond == Cond::TstEq;

        // The virtual host accepts extract and sextract of every field.
        if sh != 0 && neg && !inv {
            self.set_opc(op, Opcode::Sextract);
            self.set_arg(op, 1, src1);
            self.set_arg(op, 2, sh);
            self.set_arg(op, 3, 1);
            return;
        } else if sh != 0 {
            self.set_opc(op, Opcode::Extract);
            self.set_arg(op, 1, src1);
            self.set_arg(op, 2, sh);
            self.set_arg(op, 3, 1);
        } else {
            self.set_opc(op, Opcode::And);
            self.set_arg(op, 1, src1);
            let c = self.arg_new_constant(1);
            self.set_arg(op, 2, c);
            self.f.op_mut(op).nargs = 3;
        }
        if neg && inv {
            let op2 = self.insert_after(op, Opcode::Add, 3);
            let c = self.arg_new_constant(!0);
            self.set_arg(op2, 0, ret);
            self.set_arg(op2, 1, ret);
            self.set_arg(op2, 2, c);
        } else if inv {
            let op2 = self.insert_after(op, Opcode::Xor, 3);
            let c = self.arg_new_constant(1);
            self.set_arg(op2, 0, ret);
            self.set_arg(op2, 1, ret);
            self.set_arg(op2, 2, c);
        } else if neg {
            let op2 = self.insert_after(op, Opcode::Neg, 2);
            self.set_arg(op2, 0, ret);
            self.set_arg(op2, 1, ret);
        }
    }

    fn fold_setcond(&mut self, op: OpId) -> bool {
        let d = self.arg(op, 0);
        let i = self.do_constant_folding_cond1(op, d, 1, 2, 3);
        if i >= 0 {
            return self.gen_movi(op, d, i as u64);
        }
        let i = self.fold_setcond_zmask(op, false);
        if i > 0 {
            return true;
        }
        if i == 0 {
            self.fold_setcond_tst_pow2(op, false);
        }
        self.fold_masks_z(op, 1)
    }

    fn fold_negsetcond(&mut self, op: OpId) -> bool {
        let d = self.arg(op, 0);
        let i = self.do_constant_folding_cond1(op, d, 1, 2, 3);
        if i >= 0 {
            return self.gen_movi(op, d, (i as i64).wrapping_neg() as u64);
        }
        let i = self.fold_setcond_zmask(op, true);
        if i > 0 {
            return true;
        }
        if i == 0 {
            self.fold_setcond_tst_pow2(op, true);
        }
        self.fold_masks_s(op, !0)
    }

    fn fold_sextract(&mut self, op: OpId) -> bool {
        let t1 = self.ai(self.arg(op, 1));
        let pos = self.arg(op, 2) as u32;
        let len = self.arg(op, 3) as u32;
        if t1.z_mask == t1.o_mask {
            let d = self.arg(op, 0);
            return self.gen_movi(op, d, sextract64(t1.z_mask, pos, len));
        }
        let mut s = t1.s_mask >> pos;
        s |= !0u64 << (len - 1);
        let a = if pos != 0 { !0 } else { s & !t1.s_mask };
        let z = sextract64(t1.z_mask, pos, len);
        let o = sextract64(t1.o_mask, pos, len);
        self.fold_masks_zosa(op, z, o, s, a)
    }

    fn fold_shift(&mut self, op: OpId) -> bool {
        if self.fold_const2(op) || self.fold_ix_to_i(op, 0) || self.fold_xi_to_x(op, 0) {
            return true;
        }
        let t1 = self.ai(self.arg(op, 1));
        let t2 = self.ai(self.arg(op, 2));
        let (s, z, o) = (t1.s_mask, t1.z_mask, t1.o_mask);
        if t2.z_mask == t2.o_mask {
            // QEMU passes the count through an int, which keeps the low 32 bits.
            let sh = t2.z_mask as i32 as i64 as u64;
            let opc = self.opc(op);
            let z = do_constant_folding(opc, self.ty, z, sh);
            let o = do_constant_folding(opc, self.ty, o, sh);
            let s = do_constant_folding(opc, self.ty, s, sh);
            return self.fold_masks_zos(op, z, o, s);
        }
        match self.opc(op) {
            Opcode::Sar => return self.fold_masks_s(op, s),
            Opcode::Shr if !z & s.wrapping_neg() != 0 => return self.fold_masks_s(op, s),
            _ => {}
        }
        self.finish_folding(op)
    }

    fn fold_sub_to_neg(&mut self, op: OpId) -> bool {
        if !self.arg_is_const_val(self.arg(op, 1), 0) {
            return false;
        }
        let neg_op = match self.ty {
            Type::I32 | Type::I64 => Opcode::Neg,
            Type::V64 | Type::V128 | Type::V256 => Opcode::NegVec,
            Type::I128 => unreachable!(),
        };
        self.set_opc(op, neg_op);
        let a2 = self.arg(op, 2);
        self.set_arg(op, 1, a2);
        self.f.op_mut(op).nargs = 2;
        self.fold_neg_no_const(op)
    }

    fn fold_sub_vec(&mut self, op: OpId) -> bool {
        if self.fold_xx_to_i(op, 0) || self.fold_xi_to_x(op, 0) || self.fold_sub_to_neg(op) {
            return true;
        }
        self.finish_folding(op)
    }

    fn fold_sub(&mut self, op: OpId) -> bool {
        if self.fold_const2(op)
            || self.fold_xx_to_i(op, 0)
            || self.fold_xi_to_x(op, 0)
            || self.fold_sub_to_neg(op)
        {
            return true;
        }
        let a2 = self.arg(op, 2);
        if self.arg_is_const(a2) {
            let val = self.arg_const_val(a2);
            self.set_opc(op, Opcode::Add);
            let c = self.arg_new_constant(val.wrapping_neg());
            self.set_arg(op, 2, c);
        }
        self.finish_folding(op)
    }

    fn squash_prev_borrowout(&mut self, op: OpId) {
        let op = self.f.prev_op(op).expect("borrow-in op without a borrow-out op before it");
        match self.opc(op) {
            Opcode::Subbo => {
                self.set_opc(op, Opcode::Sub);
                self.fold_sub(op);
            }
            Opcode::Subbio => self.set_opc(op, Opcode::Subbi),
            Opcode::Subb1o => {
                let a2 = self.arg(op, 2);
                if self.arg_is_const(a2) {
                    self.set_opc(op, Opcode::Add);
                    let c = self
                        .arg_new_constant(self.arg_const_val(a2).wrapping_add(1).wrapping_neg());
                    self.set_arg(op, 2, c);
                    self.fold_add(op);
                } else {
                    let ret = self.arg(op, 0);
                    self.set_opc(op, Opcode::Sub);
                    let n = self.insert_after(op, Opcode::Add, 3);
                    let c = self.arg_new_constant(!0);
                    self.set_arg(n, 0, ret);
                    self.set_arg(n, 1, ret);
                    self.set_arg(n, 2, c);
                }
            }
            o => unreachable!("unexpected borrow producer {}", o.name()),
        }
    }

    fn fold_subbi(&mut self, op: OpId) -> bool {
        let borrow_in = self.carry_state;
        if borrow_in < 0 {
            return self.finish_folding(op);
        }
        self.carry_state = -1;
        self.squash_prev_borrowout(op);
        if borrow_in == 0 {
            self.set_opc(op, Opcode::Sub);
            return self.fold_sub(op);
        }
        let a2 = self.arg(op, 2);
        if self.arg_is_const(a2) {
            let c = self.arg_new_constant(self.arg_const_val(a2).wrapping_add(1).wrapping_neg());
            self.set_arg(op, 2, c);
        } else {
            let op2 = self.insert_before(op, Opcode::Sub, 3);
            for i in 0..3 {
                let a = self.arg(op, i);
                self.set_arg(op2, i, a);
            }
            self.fold_sub(op2);
            let a0 = self.arg(op, 0);
            self.set_arg(op, 1, a0);
            let c = self.arg_new_constant(!0);
            self.set_arg(op, 2, c);
        }
        self.set_opc(op, Opcode::Add);
        self.fold_add(op)
    }

    fn fold_subbio(&mut self, op: OpId) -> bool {
        if self.carry_state < 0 {
            return self.finish_folding(op);
        }
        self.squash_prev_borrowout(op);
        if self.carry_state != 0 {
            let t1 = self.ai(self.arg(op, 1));
            let t2 = self.ai(self.arg(op, 2));
            let mut borrow_out = -1;
            let mut done = false;
            if t2.z_mask == t2.o_mask {
                let max = if self.ty == Type::I32 { u32::MAX as u64 } else { u64::MAX };
                let v = t2.z_mask & max;
                if v < max {
                    let c = self.arg_new_constant(v + 1);
                    self.set_arg(op, 2, c);
                    done = true;
                } else {
                    borrow_out = 1;
                }
            }
            if !done && t1.z_mask == t1.o_mask {
                let v = t1.z_mask;
                if v != 0 {
                    // QEMU replaces the second operand here, which this port keeps.
                    let c = self.arg_new_constant(v - 1);
                    self.set_arg(op, 2, c);
                    done = true;
                }
            }
            if !done {
                self.set_opc(op, Opcode::Subb1o);
                self.carry_state = borrow_out;
                return self.finish_folding(op);
            }
        }
        self.set_opc(op, Opcode::Subbo);
        self.fold_subbo(op)
    }

    fn fold_subbo(&mut self, op: OpId) -> bool {
        let t1 = self.ai(self.arg(op, 1));
        let t2 = self.ai(self.arg(op, 2));
        let mut borrow_out = -1;
        if t2.z_mask == t2.o_mask {
            let v2 = t2.z_mask;
            if v2 == 0 {
                borrow_out = 0;
            } else if t1.z_mask == t1.o_mask {
                borrow_out = (t1.z_mask < v2) as i32;
            }
        }
        self.carry_state = borrow_out;
        self.finish_folding(op)
    }

    fn fold_tcg_ld(&mut self, op: OpId) -> bool {
        let (z, s) = match self.opc(op) {
            Opcode::Ld8s => (!0, i8::MIN as i64 as u64),
            Opcode::Ld8u => (make_mask(0, 8), 0),
            Opcode::Ld16s => (!0, i16::MIN as i64 as u64),
            Opcode::Ld16u => (make_mask(0, 16), 0),
            Opcode::Ld32s => (!0, i32::MIN as i64 as u64),
            Opcode::Ld32u => (make_mask(0, 32), 0),
            _ => unreachable!(),
        };
        self.fold_masks_zs(op, z, s)
    }

    fn fold_tcg_ld_memcopy(&mut self, op: OpId) -> bool {
        if self.arg(op, 1) != self.f.env().arg() {
            return self.finish_folding(op);
        }
        let ty = self.ty;
        let ofs = self.arg(op, 2) as i64;
        let dst = Temp::from_arg(self.arg(op, 0));
        if let Some(src) = self.find_mem_copy_for(ty, ofs) {
            if self.f.temp(src).base_type == ty {
                return self.gen_mov(op, dst.arg(), src.arg());
            }
        }
        self.reset_ts(dst);
        self.record_mem_copy(ty, dst, ofs, ofs + ty.size() as i64 - 1);
        true
    }

    fn fold_tcg_st(&mut self, op: OpId) -> bool {
        let ofs = self.arg(op, 2) as i64;
        if self.arg(op, 1) != self.f.env().arg() {
            self.remove_mem_copy_all();
            return true;
        }
        let lm1 = match self.opc(op) {
            Opcode::St8 => 0,
            Opcode::St16 => 1,
            Opcode::St32 => 3,
            Opcode::St | Opcode::StVec => self.ty.size() as i64 - 1,
            _ => unreachable!(),
        };
        self.remove_mem_copy_in(ofs, ofs + lm1);
        true
    }

    fn fold_tcg_st_memcopy(&mut self, op: OpId) -> bool {
        if self.arg(op, 1) != self.f.env().arg() {
            return self.fold_tcg_st(op);
        }
        let src = Temp::from_arg(self.arg(op, 0));
        let ofs = self.arg(op, 2) as i64;
        let ty = self.ty;
        if self.ts_is_const(src) {
            let prev = self.find_mem_copy_for(ty, ofs);
            if prev == Some(src) {
                self.f.remove_op(op);
                return true;
            }
        }
        let last = ofs + ty.size() as i64 - 1;
        self.remove_mem_copy_in(ofs, last);
        self.record_mem_copy(ty, src, ofs, last);
        true
    }

    fn fold_xor(&mut self, op: OpId) -> bool {
        if self.fold_const2_commutative(op)
            || self.fold_xx_to_i(op, 0)
            || self.fold_xi_to_x(op, 0)
            || self.fold_xi_to_not(op, !0)
        {
            return true;
        }
        let t1 = self.ai(self.arg(op, 1));
        let t2 = self.ai(self.arg(op, 2));
        let z = (t1.z_mask | t2.z_mask) & !(t1.o_mask & t2.o_mask);
        let o = (t1.o_mask & !t2.z_mask) | (t2.o_mask & !t1.z_mask);
        let s = t1.s_mask & t2.s_mask;
        self.fold_masks_zos(op, z, o, s)
    }

    fn run(&mut self) {
        let mut cur = self.f.first_op();
        while let Some(op) = cur {
            cur = self.f.next_op(op);
            let opc = self.opc(op);
            if opc == Opcode::Call {
                self.fold_call(op);
                continue;
            }
            let def = opc.def();
            self.init_arguments(op, (def.nb_oargs + def.nb_iargs) as usize);
            self.copy_propagate(op, def.nb_oargs as usize, def.nb_iargs as usize);
            self.ty = self.f.op(op).ty;

            let done = match opc {
                Opcode::Add => self.fold_add(op),
                Opcode::AddVec => self.fold_add_vec(op),
                Opcode::Addci => self.fold_addci(op),
                Opcode::Addcio => self.fold_addcio(op),
                Opcode::Addco => self.fold_addco(op),
                Opcode::And | Opcode::AndVec => self.fold_and(op),
                Opcode::Andc | Opcode::AndcVec => self.fold_andc(op),
                Opcode::Brcond => self.fold_brcond(op),
                Opcode::Bswap16 | Opcode::Bswap32 | Opcode::Bswap64 => self.fold_bswap(op),
                Opcode::Clz | Opcode::Ctz => self.fold_count_zeros(op),
                Opcode::Ctpop => self.fold_ctpop(op),
                Opcode::Deposit => self.fold_deposit(op),
                Opcode::Divs | Opcode::Divu => self.fold_divide(op),
                Opcode::DupVec => self.fold_dup(op),
                Opcode::Eqv | Opcode::EqvVec => self.fold_eqv(op),
                Opcode::Extract => self.fold_extract(op),
                Opcode::Extract2 => self.fold_extract2(op),
                Opcode::ExtI32I64 => self.fold_exts(op),
                Opcode::ExtuI32I64 | Opcode::ExtrlI64I32 | Opcode::ExtrhI64I32 => {
                    self.fold_extu(op)
                }
                Opcode::Ld8s
                | Opcode::Ld8u
                | Opcode::Ld16s
                | Opcode::Ld16u
                | Opcode::Ld32s
                | Opcode::Ld32u => self.fold_tcg_ld(op),
                Opcode::Ld | Opcode::LdVec => self.fold_tcg_ld_memcopy(op),
                Opcode::St8 | Opcode::St16 | Opcode::St32 => self.fold_tcg_st(op),
                Opcode::St | Opcode::StVec => self.fold_tcg_st_memcopy(op),
                Opcode::Mb => self.fold_mb(op),
                Opcode::Mov | Opcode::MovVec => self.fold_mov(op),
                Opcode::Movcond => self.fold_movcond(op),
                Opcode::Mul => self.fold_mul(op),
                Opcode::Mulsh | Opcode::Muluh => self.fold_mul_highpart(op),
                Opcode::Muls2 | Opcode::Mulu2 => self.fold_multiply2(op),
                Opcode::Nand | Opcode::NandVec => self.fold_nand(op),
                Opcode::Neg => self.fold_neg(op),
                Opcode::Nor | Opcode::NorVec => self.fold_nor(op),
                Opcode::Not | Opcode::NotVec => self.fold_not(op),
                Opcode::Or | Opcode::OrVec => self.fold_or(op),
                Opcode::Orc | Opcode::OrcVec => self.fold_orc(op),
                Opcode::QemuLd => self.fold_qemu_ld_1reg(op),
                Opcode::QemuLd2 => self.fold_qemu_ld_2reg(op),
                Opcode::QemuSt | Opcode::QemuSt2 => self.fold_qemu_st(),
                Opcode::Rems | Opcode::Remu => self.fold_remainder(op),
                Opcode::Rotl | Opcode::Rotr | Opcode::Sar | Opcode::Shl | Opcode::Shr => {
                    self.fold_shift(op)
                }
                Opcode::Setcond => self.fold_setcond(op),
                Opcode::Negsetcond => self.fold_negsetcond(op),
                Opcode::CmpVec => self.fold_cmp_vec(op),
                Opcode::CmpselVec => self.fold_cmpsel_vec(op),
                Opcode::BitselVec => self.fold_bitsel_vec(op),
                Opcode::Sextract => self.fold_sextract(op),
                Opcode::Sub => self.fold_sub(op),
                Opcode::Subbi => self.fold_subbi(op),
                Opcode::Subbio => self.fold_subbio(op),
                Opcode::Subbo => self.fold_subbo(op),
                Opcode::SubVec => self.fold_sub_vec(op),
                Opcode::Xor | Opcode::XorVec => self.fold_xor(op),
                Opcode::SetLabel
                | Opcode::Br
                | Opcode::ExitTb
                | Opcode::GotoTb
                | Opcode::GotoPtr => {
                    self.finish_ebb();
                    true
                }
                _ => self.finish_folding(op),
            };
            debug_assert!(done);
        }
    }
}

impl Func {
    /// `tcg_optimize`: propagate constants and copies and fold constant expressions.
    pub fn optimize(&mut self) {
        let mut o = Opt {
            f: self,
            info: Vec::new(),
            used: Vec::new(),
            mem: Vec::new(),
            prev_mb: None,
            ty: Type::I32,
            carry_state: -1,
        };
        o.grow();
        o.run();
        // Folding can change an op into one with fewer arguments; keep the counts exact.
        for id in self.op_ids() {
            let op = self.op_mut(id);
            if op.opc != Opcode::Call {
                op.nargs = op.opc.def().nb_args() as u8;
            }
        }
    }
}
