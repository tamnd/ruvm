// SPDX-License-Identifier: MIT OR Apache-2.0

//! The target independent register allocator, a port of the `tcg_reg_alloc_*` half of
//! `tcg/tcg.c`.
//!
//! A backend implements [`Target`]: its registers, its constraint letters, and the `tcg_out_*`
//! hooks that emit moves, loads, stores and ops. [`reg_alloc`] then walks a [`Func`] in order,
//! keeping every temp in one of QEMU's states (`TEMP_VAL_DEAD`, `TEMP_VAL_REG`, `TEMP_VAL_MEM`
//! and `TEMP_VAL_CONST`, here [`Val`]), loading inputs into registers that satisfy each op's
//! constraints, spilling when it runs out, and syncing or freeing temps as `op.life` says.
//!
//! [`liveness`] computes `op.life` the way `liveness_pass_1` does, so a backend can allocate a
//! function that never went through [`Func::gen_code`].
//!
//! Differences from QEMU:
//!
//! - [`liveness`] never removes or rewrites ops. Dead ops are allocated and emitted like any
//!   other, with their outputs freed straight away, and `mov` to a dead, unsynced temp emits
//!   nothing. Register preferences (`output_pref`) are not computed.
//! - A target can add `CALL_CLOBBER` and `SIDE_EFFECTS` to an op ([`Target::extra_op_flags`]),
//!   for ops it implements with a call or ops that can fault, and both [`liveness`] and the
//!   allocator honour them.
//! - `TEMP_FIXED` temps are treated as read only constants holding `val` rather than as values
//!   in a fixed register, since the backends that use this keep `env` as an offset.
//! - Where QEMU only asserts that liveness left globals and TB temps in memory at the end of a
//!   block, before a conditional branch and around calls, this allocator syncs them, so a
//!   function with stale `op.life` still produces correct code, just slower code.
//! - Register pairs (`TCG_CT_PAIR`) are not supported; no target here needs them.
//! - A global that may have been set since the last point every global was in memory, and that
//!   the fall through path of a conditional branch sets again before reading it, is not stored
//!   before the branch as QEMU's `la_global_sync()` asks. [`liveness`] keeps it live up to the
//!   branch, and a target that opts in ([`Target::out_of_line_branches`]) gets the branch
//!   pointed at a stub emitted after the function, which stores it and jumps to the real label.
//!   The fall through path then never stores it. Without the opt in the branch stores it in
//!   line.
//! - Helper calls pass every argument through memory: the target says where each argument
//!   word goes ([`Target::call_arg_home`]) and results come back the same way.

use std::borrow::Cow;

use crate::hash::FastHashMap;
use crate::ir::{DEAD_ARG, Func, Label, MAX_OP_ARGS, Op, OpId, SYNC_ARG, Temp, TempData};
use crate::opcode::Opcode;
use crate::types::{Cond, TempKind, Type, call_flags, dup_const, opf};

/// A host register number, below 64.
pub type Reg = u8;

/// A set of host registers, `TCGRegSet`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct RegSet(pub u64);

impl RegSet {
    /// No registers.
    pub const EMPTY: RegSet = RegSet(0);

    /// Just `r`.
    pub const fn single(r: Reg) -> RegSet {
        RegSet(1 << r)
    }

    /// Is `r` in the set.
    pub const fn contains(self, r: Reg) -> bool {
        (self.0 >> r) & 1 != 0
    }

    /// The set with `r` added.
    pub const fn with(self, r: Reg) -> RegSet {
        RegSet(self.0 | 1 << r)
    }

    /// Every register in either set.
    pub const fn union(self, o: RegSet) -> RegSet {
        RegSet(self.0 | o.0)
    }

    /// The registers in both sets.
    pub const fn and(self, o: RegSet) -> RegSet {
        RegSet(self.0 & o.0)
    }

    /// The registers of `self` not in `o`.
    pub const fn minus(self, o: RegSet) -> RegSet {
        RegSet(self.0 & !o.0)
    }

    /// True for the empty set.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The number of registers.
    pub const fn len(self) -> u32 {
        self.0.count_ones()
    }

    /// The lowest numbered register, if any.
    pub const fn first(self) -> Option<Reg> {
        if self.0 == 0 { None } else { Some(self.0.trailing_zeros() as Reg) }
    }

    /// Iterate over the registers, lowest first.
    pub fn iter(self) -> impl Iterator<Item = Reg> {
        (0..64u8).filter(move |&r| self.contains(r))
    }
}

/// Target independent constraint bits, `TCG_CT_*`. Targets use bits from `0x100` up for their
/// own constant classes.
pub mod ct {
    /// Any constant, `TCG_CT_CONST`.
    pub const CONST: u32 = 1;
    /// A zero constant may use the hardware zero register, `TCG_CT_REG_ZERO`.
    pub const REG_ZERO: u32 = 2;
}

/// The parsed constraint of one op argument, `TCGArgConstraint`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArgConstraint {
    /// [`ct`] and target constant bits.
    pub ct: u32,
    /// The registers the argument may live in.
    pub regs: RegSet,
    /// An input that must share the register of output `alias_index`.
    pub ialias: bool,
    /// An output that shares the register of input `alias_index`.
    pub oalias: bool,
    /// An output that must not overlap any input, `&`.
    pub newreg: bool,
    /// The other side of an alias.
    pub alias_index: u8,
    /// The order the allocator handles arguments in.
    pub sort_index: u8,
}

/// What a target constraint letter means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Letter {
    /// A register class, the `REGS` lines of `tcg-target-con-str.h`.
    Regs(RegSet),
    /// A constant class, the `CONST` lines.
    Const(u32),
}

/// Parse one constraint set, `process_constraint_sets` and `sort_constraints`. `set` has one
/// string per argument, outputs first.
pub fn parse_constraints(
    set: &[&str],
    nb_oargs: usize,
    letter: &dyn Fn(char) -> Option<Letter>,
) -> Result<Vec<ArgConstraint>, String> {
    let mut a = vec![ArgConstraint::default(); set.len()];
    for (i, s) in set.iter().enumerate() {
        let input = i >= nb_oargs;
        let mut chars = s.chars().peekable();
        match chars.peek() {
            Some(&c) if c.is_ascii_digit() => {
                let o = c as usize - '0' as usize;
                if !input || o >= nb_oargs || a[o].oalias || s.len() != 1 {
                    return Err(format!("bad alias constraint {s:?}"));
                }
                a[i] = a[o];
                a[o].oalias = true;
                a[o].alias_index = i as u8;
                a[i].ialias = true;
                a[i].alias_index = o as u8;
                continue;
            }
            Some('&') => {
                if input {
                    return Err(format!("'&' on input constraint {s:?}"));
                }
                a[i].newreg = true;
                chars.next();
            }
            _ => {}
        }
        for c in chars {
            match c {
                'i' => a[i].ct |= ct::CONST,
                'z' => a[i].ct |= ct::REG_ZERO,
                _ => match letter(c) {
                    Some(Letter::Regs(r)) => a[i].regs = a[i].regs.union(r),
                    Some(Letter::Const(k)) => a[i].ct |= k,
                    None => return Err(format!("unknown constraint letter {c:?}")),
                },
            }
        }
    }
    sort_constraints(&mut a, 0, nb_oargs);
    let n = a.len();
    sort_constraints(&mut a, nb_oargs, n - nb_oargs);
    Ok(a)
}

/// `get_constraint_priority`: arguments with fewer choices go first.
fn priority(a: &[ArgConstraint], k: usize) -> i64 {
    let n = a[k].regs.len();
    if n == 1 || a[k].oalias { i64::MAX } else { -(n as i64) }
}

/// `sort_constraints`.
fn sort_constraints(a: &mut [ArgConstraint], start: usize, n: usize) {
    for i in 0..n {
        a[start + i].sort_index = (start + i) as u8;
    }
    if n <= 1 {
        return;
    }
    for i in 0..n - 1 {
        for j in i + 1..n {
            let p1 = priority(a, a[start + i].sort_index as usize);
            let p2 = priority(a, a[start + j].sort_index as usize);
            if p1 < p2 {
                let t = a[start + i].sort_index;
                a[start + i].sort_index = a[start + j].sort_index;
                a[start + j].sort_index = t;
            }
        }
    }
}

/// Where a temp's value is, `TCGTempVal`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Val {
    /// Nothing; the value is not needed.
    Dead,
    /// In this register.
    Reg(Reg),
    /// In the temp's memory home.
    Mem,
    /// This constant, not yet materialized.
    Const(i64),
}

/// What a backend provides to the allocator: `tcg-target.c.inc`'s register description and
/// `tcg_out_*` hooks.
pub trait Target {
    /// The error type of the hooks that can fail.
    type Error;

    /// Turn an allocator complaint about the IR into an error.
    fn bad_ir(&self, msg: String) -> Self::Error;

    /// `tcg_target_reg_alloc_order`.
    fn alloc_order(&self) -> &[Reg];
    /// `tcg_target_available_regs[ty]`.
    fn available_regs(&self, ty: Type) -> RegSet;
    /// `s->reserved_regs`: never allocated.
    fn reserved_regs(&self) -> RegSet;
    /// `tcg_target_call_clobber_regs`.
    fn call_clobber_regs(&self) -> RegSet;
    /// `TCG_REG_ZERO`, if the host has one.
    fn zero_reg(&self) -> Option<Reg>;

    /// The constraint strings of `op`, `tcg_target_op_def`. Fails for an op the target
    /// cannot generate.
    fn op_constraints(&self, f: &Func, op: &Op) -> Result<&'static [&'static str], Self::Error>;
    /// The meaning of a target constraint letter, `tcg-target-con-str.h`.
    fn constraint_letter(&self, c: char) -> Option<Letter>;
    /// `tcg_target_const_match`.
    fn const_match(&self, val: i64, ct: u32, ty: Type, cond: Cond, vece: u32) -> bool;
    /// Flags to add to `op`'s own: `CALL_CLOBBER` for an op the target implements with a
    /// call, `SIDE_EFFECTS` for one that can leave the block.
    fn extra_op_flags(&self, f: &Func, op: &Op) -> u32;

    /// The memory home of a global or TB or EBB temp: a base register and an offset.
    fn temp_home(&mut self, f: &Func, t: Temp) -> (Reg, i64);

    /// `tcg_out_mov`. Returns false for a move between register classes it cannot do.
    fn out_mov(&mut self, ty: Type, dst: Reg, src: Reg) -> bool;
    /// `tcg_out_movi`.
    fn out_movi(&mut self, ty: Type, dst: Reg, val: i64);
    /// `tcg_out_dupi_vec`.
    fn out_dupi_vec(&mut self, ty: Type, vece: u32, dst: Reg, val: u64);
    /// `tcg_out_ld`.
    fn out_ld(&mut self, ty: Type, dst: Reg, base: Reg, off: i64);
    /// `tcg_out_st`.
    fn out_st(&mut self, ty: Type, src: Reg, base: Reg, off: i64);
    /// `tcg_out_sti`: store a constant directly, if the host can.
    fn out_sti(&mut self, ty: Type, val: i64, base: Reg, off: i64) -> bool;
    /// Emit `op` with registers and constants in `args`, `const_args` saying which are
    /// constants. The constant arguments of the op itself follow the temps unchanged.
    fn out_op(
        &mut self,
        f: &Func,
        id: OpId,
        op: &Op,
        args: &[u64],
        const_args: &[bool],
    ) -> Result<(), Self::Error>;

    /// Where argument and result word `idx` of a helper call goes.
    fn call_arg_home(&self, idx: usize) -> (Reg, i64);
    /// Emit the call itself, once the arguments are stored.
    fn out_call(&mut self, f: &Func, op: &Op) -> Result<(), Self::Error>;

    /// Whether the target accepts labels numbered from [`Func::nb_labels`] up in `set_label`,
    /// `br` and `brcond`, so the allocator can send the taken edge of a conditional branch
    /// through a stub that stores the dirty globals. Without it a branch stores them in line.
    fn out_of_line_branches(&self) -> bool {
        false
    }
}

/// The taken edge of a conditional branch, emitted after the function: the stores that bring
/// the globals the branch left dirty back to memory, then a jump to the branch's own label.
#[derive(Debug)]
struct BranchStub {
    /// The label the branch was pointed at instead.
    label: u32,
    /// The branch's label.
    target: Label,
    /// Globals in registers: type, register, and memory home.
    regs: Vec<(Type, Reg, Reg, i64)>,
    /// Globals holding a constant: type, value, and memory home.
    consts: Vec<(Type, i64, Reg, i64)>,
}

/// The allocator state for one function.
#[derive(Debug)]
pub struct RegAlloc<'f> {
    f: &'f Func,
    val: Vec<Val>,
    coherent: Vec<bool>,
    reg_to_temp: [Option<Temp>; 64],
    reserved: RegSet,
    /// Parsed constraint sets, keyed by output count and the address and length of the
    /// target's static strings, so a lookup hashes three words rather than the strings.
    cache: FastHashMap<(usize, usize, usize), usize>,
    /// The sets `cache` points into.
    ct_sets: Vec<Vec<ArgConstraint>>,
    stubs: Vec<BranchStub>,
    next_label: u32,
    /// For each conditional branch, the globals that are dead on its fall through path.
    drops: Drops,
}

fn readonly(td: &TempData) -> bool {
    matches!(td.kind, TempKind::Const | TempKind::Fixed)
}

/// Allocate registers for `f` and emit it through `t`, `tcg_gen_code`'s main loop. `op.life`
/// must be up to date; see [`liveness`].
pub fn reg_alloc<T: Target>(f: &Func, t: &mut T) -> Result<(), T::Error> {
    let mut ra = RegAlloc::new(f, t);
    if t.out_of_line_branches() {
        ra.drops = scan(f, &|f, op| t.extra_op_flags(f, op)).1;
    }
    let mut last = None;
    for (id, op) in f.ops() {
        ra.op(t, id, op)?;
        last = Some(id);
    }
    match last {
        Some(id) => ra.out_stubs(t, id),
        None => Ok(()),
    }
}

/// [`reg_alloc`] with `op.life` taken from `live` rather than from the ops, so a function
/// can be allocated without a copy to write `op.life` into. `live` must come from [`analyze`]
/// on `f` with the target's [`Target::extra_op_flags`].
pub fn reg_alloc_live<T: Target>(f: &Func, live: Liveness, t: &mut T) -> Result<(), T::Error> {
    let mut ra = RegAlloc::new(f, t);
    if t.out_of_line_branches() {
        ra.drops = live.drops;
    }
    let mut last = None;
    for (id, op) in f.ops() {
        let mut op = *op;
        op.life = live.life[id.index()];
        ra.op(t, id, &op)?;
        last = Some(id);
    }
    match last {
        Some(id) => ra.out_stubs(t, id),
        None => Ok(()),
    }
}

/// A `set_label` or `br` op naming label `id`.
fn label_op(opc: Opcode, id: u32) -> Op {
    let mut args = [0; MAX_OP_ARGS];
    args[0] = u64::from(id);
    Op { opc, ty: Type::I64, vece: 0, flags: 0, callo: 0, calli: 0, life: 0, nargs: 1, args }
}

impl<'f> RegAlloc<'f> {
    /// `tcg_reg_alloc_start`.
    pub fn new<T: Target>(f: &'f Func, t: &T) -> RegAlloc<'f> {
        let val = f
            .temps()
            .iter()
            .map(|td| match td.kind {
                TempKind::Fixed | TempKind::Const => Val::Const(td.val),
                TempKind::Global | TempKind::Tb => Val::Mem,
                TempKind::Ebb => Val::Dead,
            })
            .collect();
        RegAlloc {
            f,
            val,
            coherent: vec![true; f.nb_temps()],
            reg_to_temp: [None; 64],
            reserved: t.reserved_regs(),
            cache: FastHashMap::default(),
            ct_sets: Vec::new(),
            stubs: Vec::new(),
            next_label: f.nb_labels() as u32,
            drops: FastHashMap::default(),
        }
    }

    /// The current state of `t`.
    pub fn val(&self, t: Temp) -> Val {
        self.val[t.index()]
    }

    fn td(&self, t: Temp) -> &'f TempData {
        &self.f.temps()[t.index()]
    }

    fn set_reg(&mut self, t: Temp, reg: Reg) {
        if let Val::Reg(old) = self.val[t.index()] {
            if old == reg {
                return;
            }
            self.reg_to_temp[old as usize] = None;
        }
        self.reg_to_temp[reg as usize] = Some(t);
        self.val[t.index()] = Val::Reg(reg);
    }

    fn set_nonreg(&mut self, t: Temp, v: Val) {
        if let Val::Reg(old) = self.val[t.index()] {
            self.reg_to_temp[old as usize] = None;
        }
        self.val[t.index()] = v;
    }

    /// `temp_free_or_dead`: negative frees, positive kills.
    fn free_or_dead(&mut self, t: Temp, free_or_dead: i32) {
        let td = self.td(t);
        let v = match td.kind {
            TempKind::Fixed | TempKind::Const => Val::Const(td.val),
            TempKind::Global | TempKind::Tb => Val::Mem,
            TempKind::Ebb => {
                if free_or_dead < 0 {
                    Val::Mem
                } else {
                    Val::Dead
                }
            }
        };
        self.set_nonreg(t, v);
    }

    /// `temp_dead`.
    fn dead(&mut self, t: Temp) {
        self.free_or_dead(t, 1);
    }

    /// `temp_sync`.
    fn sync<T: Target>(
        &mut self,
        tg: &mut T,
        t: Temp,
        allocated: RegSet,
        preferred: RegSet,
        free_or_dead: i32,
    ) {
        let td = self.td(t);
        if !readonly(td) && !self.coherent[t.index()] {
            match self.val[t.index()] {
                Val::Const(v) => {
                    let (base, off) = tg.temp_home(self.f, t);
                    if free_or_dead == 0 || !tg.out_sti(td.ty, v, base, off) {
                        self.load(tg, t, tg.available_regs(td.ty), allocated, preferred);
                        if let Val::Reg(r) = self.val[t.index()] {
                            tg.out_st(td.ty, r, base, off);
                        }
                    }
                }
                Val::Reg(r) => {
                    let (base, off) = tg.temp_home(self.f, t);
                    tg.out_st(td.ty, r, base, off);
                }
                Val::Mem | Val::Dead => {}
            }
            self.coherent[t.index()] = true;
        }
        if free_or_dead != 0 {
            self.free_or_dead(t, free_or_dead);
        }
    }

    /// `tcg_reg_free`: spill whatever is in `reg`.
    fn reg_free<T: Target>(&mut self, tg: &mut T, reg: Reg, allocated: RegSet) {
        if let Some(t) = self.reg_to_temp[reg as usize] {
            self.sync(tg, t, allocated, RegSet::EMPTY, -1);
        }
    }

    /// `tcg_reg_alloc`: a register in `required` but not `allocated`, spilling if needed.
    fn reg_alloc<T: Target>(
        &mut self,
        tg: &mut T,
        required: RegSet,
        allocated: RegSet,
        preferred: RegSet,
    ) -> Reg {
        let sets = [required.minus(allocated).and(preferred), required.minus(allocated)];
        assert!(!sets[1].is_empty(), "no register left to allocate");
        let first = if sets[0].is_empty() || sets[0] == sets[1] { 1 } else { 0 };
        // A copy, since freeing a register below needs `tg` mutably.
        let mut order_buf = [0 as Reg; 64];
        let n = {
            let order = tg.alloc_order();
            let n = order.len().min(64);
            order_buf[..n].copy_from_slice(&order[..n]);
            n
        };
        let order = &order_buf[..n];
        for set in &sets[first..] {
            if set.len() == 1 {
                let r = set.first().unwrap_or(0);
                if self.reg_to_temp[r as usize].is_none() {
                    return r;
                }
            } else {
                for &r in order {
                    if self.reg_to_temp[r as usize].is_none() && set.contains(r) {
                        return r;
                    }
                }
            }
        }
        for set in &sets[first..] {
            if set.len() == 1 {
                let r = set.first().unwrap_or(0);
                self.reg_free(tg, r, allocated);
                return r;
            }
            for &r in order {
                if set.contains(r) {
                    self.reg_free(tg, r, allocated);
                    return r;
                }
            }
        }
        panic!("the allocation order misses every register of {:#x}", sets[1].0)
    }

    /// `temp_load`: make sure `t` is in a register, from `desired` if it has to move.
    fn load<T: Target>(
        &mut self,
        tg: &mut T,
        t: Temp,
        desired: RegSet,
        allocated: RegSet,
        preferred: RegSet,
    ) {
        let td = self.td(t);
        let reg = match self.val[t.index()] {
            Val::Reg(_) => return,
            Val::Const(v) => {
                let reg = self.reg_alloc(tg, desired, allocated, preferred);
                if td.ty.is_vector() {
                    let u = v as u64;
                    let vece = if u == dup_const(0, u) {
                        0
                    } else if u == dup_const(1, u) {
                        1
                    } else if u == dup_const(2, u) {
                        2
                    } else {
                        3
                    };
                    tg.out_dupi_vec(td.ty, vece, reg, u);
                } else {
                    tg.out_movi(td.ty, reg, v);
                }
                self.coherent[t.index()] = false;
                reg
            }
            // A dead temp is only read by IR that uses a value it never set; give it
            // whatever its home holds rather than failing.
            Val::Mem | Val::Dead => {
                let reg = self.reg_alloc(tg, desired, allocated, preferred);
                let (base, off) = tg.temp_home(self.f, t);
                tg.out_ld(td.ty, reg, base, off);
                self.coherent[t.index()] = true;
                reg
            }
        };
        self.set_reg(t, reg);
    }

    /// Bring every global back to memory and free its register, `save_globals`.
    fn save_globals<T: Target>(&mut self, tg: &mut T, allocated: RegSet) {
        for i in 0..self.f.nb_globals() {
            let t = Temp::from_index(i);
            if self.val[i] != Val::Mem {
                if readonly(self.td(t)) {
                    self.dead(t);
                } else {
                    self.sync(tg, t, allocated, RegSet::EMPTY, -1);
                }
            }
        }
    }

    /// Write every global back to memory, keeping registers, `sync_globals`.
    fn sync_globals<T: Target>(&mut self, tg: &mut T, allocated: RegSet) {
        for i in 0..self.f.nb_globals() {
            self.sync(tg, Temp::from_index(i), allocated, RegSet::EMPTY, 0);
        }
    }

    /// `tcg_reg_alloc_bb_end`.
    fn bb_end<T: Target>(&mut self, tg: &mut T, allocated: RegSet) {
        for i in self.f.nb_globals()..self.f.nb_temps() {
            let t = Temp::from_index(i);
            match self.td(t).kind {
                TempKind::Tb => {
                    if self.val[i] != Val::Mem {
                        self.sync(tg, t, allocated, RegSet::EMPTY, -1);
                    }
                }
                _ => {
                    if matches!(self.val[i], Val::Reg(_)) || self.td(t).kind == TempKind::Ebb {
                        self.dead(t);
                    }
                }
            }
        }
        self.save_globals(tg, allocated);
    }

    /// `tcg_reg_alloc_cbranch`.
    fn cbranch<T: Target>(&mut self, tg: &mut T, allocated: RegSet) {
        self.sync_globals(tg, allocated);
        for i in self.f.nb_globals()..self.f.nb_temps() {
            let t = Temp::from_index(i);
            if self.td(t).kind == TempKind::Tb {
                self.sync(tg, t, allocated, RegSet::EMPTY, 0);
            }
        }
    }

    /// The out of line half of [`Self::cbranch`]: syncs TB temps and any global that is not a
    /// plain integer in line, and returns the label of a stub that stores the rest, if any are
    /// dirty. The fall through path keeps them in registers unless it no longer needs them.
    fn cbranch_stub<T: Target>(
        &mut self,
        tg: &mut T,
        id: OpId,
        allocated: RegSet,
        target: Label,
    ) -> Option<u32> {
        for i in 0..self.f.nb_temps() {
            let t = Temp::from_index(i);
            let td = self.td(t);
            let global = i < self.f.nb_globals();
            if td.kind == TempKind::Tb || (global && !matches!(td.ty, Type::I32 | Type::I64)) {
                self.sync(tg, t, allocated, RegSet::EMPTY, 0);
            }
        }
        // Collect only now: syncing a constant above may have spilled a register.
        let mut regs = Vec::new();
        let mut consts = Vec::new();
        for i in 0..self.f.nb_globals() {
            let t = Temp::from_index(i);
            let td = self.td(t);
            if readonly(td) || self.coherent[i] {
                continue;
            }
            match self.val[i] {
                Val::Reg(r) => {
                    let (base, off) = tg.temp_home(self.f, t);
                    regs.push((td.ty, r, base, off));
                }
                Val::Const(v) => {
                    let (base, off) = tg.temp_home(self.f, t);
                    consts.push((td.ty, v, base, off));
                }
                Val::Mem | Val::Dead => {}
            }
        }
        // A global the fall through path sets again before reading it or needing it in
        // memory gives up its register there; only the stub stores it.
        if let Some(dead) = self.drops.remove(&id) {
            for t in dead {
                self.dead(t);
            }
        }
        if regs.is_empty() && consts.is_empty() {
            return None;
        }
        let label = self.next_label;
        self.next_label += 1;
        self.stubs.push(BranchStub { label, target, regs, consts });
        Some(label)
    }

    /// Emit the [`BranchStub`]s after the last op, behind a jump in case the function falls
    /// off its end.
    fn out_stubs<T: Target>(&mut self, tg: &mut T, id: OpId) -> Result<(), T::Error> {
        if self.stubs.is_empty() {
            return Ok(());
        }
        let end = self.next_label;
        self.next_label += 1;
        let br = label_op(Opcode::Br, end);
        tg.out_op(self.f, id, &br, &br.args[..1], &[true])?;
        let scratch = tg
            .alloc_order()
            .iter()
            .copied()
            .find(|&r| !self.reserved.contains(r) && tg.available_regs(Type::I64).contains(r));
        for s in std::mem::take(&mut self.stubs) {
            let set = label_op(Opcode::SetLabel, s.label);
            tg.out_op(self.f, id, &set, &set.args[..1], &[true])?;
            for &(ty, r, base, off) in &s.regs {
                tg.out_st(ty, r, base, off);
            }
            // The registers are all stored, so any of them can carry a constant.
            for &(ty, v, base, off) in &s.consts {
                if !tg.out_sti(ty, v, base, off) {
                    let Some(r) = scratch else {
                        return Err(tg.bad_ir("no register to store a constant global".into()));
                    };
                    tg.out_movi(ty, r, v);
                    tg.out_st(ty, r, base, off);
                }
            }
            let br = label_op(Opcode::Br, s.target.id());
            tg.out_op(self.f, id, &br, &br.args[..1], &[true])?;
        }
        let set = label_op(Opcode::SetLabel, end);
        tg.out_op(self.f, id, &set, &set.args[..1], &[true])
    }

    fn op<T: Target>(&mut self, tg: &mut T, id: OpId, op: &Op) -> Result<(), T::Error> {
        match op.opc {
            Opcode::Mov | Opcode::MovVec => {
                self.alloc_mov(tg, op);
                Ok(())
            }
            Opcode::Discard => {
                self.dead(op.arg_temp(0));
                Ok(())
            }
            Opcode::Call => self.alloc_call(tg, op),
            Opcode::DupVec if matches!(self.val(op.arg_temp(1)), Val::Const(_)) => {
                self.alloc_dup_const(tg, op);
                Ok(())
            }
            _ => self.alloc_op(tg, id, op),
        }
    }

    /// `tcg_reg_alloc_do_movi`.
    fn do_movi<T: Target>(&mut self, tg: &mut T, ots: Temp, val: i64, op: &Op) {
        self.set_nonreg(ots, Val::Const(val));
        self.coherent[ots.index()] = false;
        if op.need_sync_arg(0) {
            let fd = if op.is_dead_arg(0) { 1 } else { 0 };
            self.sync(tg, ots, self.reserved, RegSet::EMPTY, fd);
        } else if op.is_dead_arg(0) {
            self.dead(ots);
        }
    }

    /// `tcg_reg_alloc_mov`.
    fn alloc_mov<T: Target>(&mut self, tg: &mut T, op: &Op) {
        let ots = op.arg_temp(0);
        let ts = op.arg_temp(1);
        let allocated = self.reserved;
        let otype = self.td(ots).ty;
        let itype = self.td(ts).ty;
        if ts == ots {
            if op.need_sync_arg(0) {
                self.sync(tg, ots, allocated, RegSet::EMPTY, op.is_dead_arg(0) as i32);
            }
            return;
        }
        if op.is_dead_arg(0) && !op.need_sync_arg(0) {
            // Liveness would have removed this op.
            if op.is_dead_arg(1) {
                self.dead(ts);
            }
            self.dead(ots);
            return;
        }
        if let Val::Const(v) = self.val(ts) {
            if op.is_dead_arg(1) {
                self.dead(ts);
            }
            self.do_movi(tg, ots, v, op);
            return;
        }
        if !matches!(self.val(ts), Val::Reg(_)) {
            self.load(tg, ts, tg.available_regs(itype), allocated, RegSet::EMPTY);
        }
        let Val::Reg(ireg) = self.val(ts) else { unreachable!("loaded temp not in a register") };
        if op.is_dead_arg(0) {
            let (base, off) = tg.temp_home(self.f, ots);
            tg.out_st(otype, ireg, base, off);
            if op.is_dead_arg(1) {
                self.dead(ts);
            }
            self.dead(ots);
            return;
        }
        let oreg;
        if op.is_dead_arg(1) && self.td(ts).kind != TempKind::Fixed {
            self.dead(ts);
            oreg = ireg;
        } else {
            oreg = match self.val(ots) {
                Val::Reg(r) => r,
                _ => self.reg_alloc(
                    tg,
                    tg.available_regs(otype),
                    allocated.with(ireg),
                    RegSet::EMPTY,
                ),
            };
            if !tg.out_mov(otype, oreg, ireg) {
                let (base, off) = tg.temp_home(self.f, ots);
                tg.out_st(itype, ireg, base, off);
                self.set_nonreg(ots, Val::Mem);
                self.coherent[ots.index()] = true;
                return;
            }
        }
        self.set_reg(ots, oreg);
        self.coherent[ots.index()] = false;
        if op.need_sync_arg(0) {
            self.sync(tg, ots, allocated, RegSet::EMPTY, 0);
        }
    }

    /// The constant case of `tcg_reg_alloc_dup`; the rest goes through [`Self::alloc_op`].
    fn alloc_dup_const<T: Target>(&mut self, tg: &mut T, op: &Op) {
        let ots = op.arg_temp(0);
        let its = op.arg_temp(1);
        let Val::Const(v) = self.val(its) else { return };
        let v = if self.td(its).ty == Type::I32 { v as u32 as u64 } else { v as u64 };
        let val = dup_const(op.vece as u32, v) as i64;
        if op.is_dead_arg(1) {
            self.dead(its);
        }
        self.do_movi(tg, ots, val, op);
    }

    /// The parsed constraints of `op`, copied out of the cache so the caller can keep them
    /// while it mutates the allocator.
    fn constraints<T: Target>(
        &mut self,
        tg: &T,
        op: &Op,
    ) -> Result<[ArgConstraint; MAX_OP_ARGS], T::Error> {
        let set = tg.op_constraints(self.f, op)?;
        let nb_oargs = op.nb_oargs();
        if set.len() != nb_oargs + op.nb_iargs() || set.len() > MAX_OP_ARGS {
            return Err(tg.bad_ir(format!("{}: constraint set has the wrong size", op.opc.name())));
        }
        let key = (nb_oargs, set.as_ptr() as usize, set.len());
        let idx = match self.cache.get(&key) {
            Some(&i) => i,
            None => {
                let c = parse_constraints(set, nb_oargs, &|ch| tg.constraint_letter(ch))
                    .map_err(|e| tg.bad_ir(format!("{}: {e}", op.opc.name())))?;
                self.ct_sets.push(c);
                self.cache.insert(key, self.ct_sets.len() - 1);
                self.ct_sets.len() - 1
            }
        };
        let c = &self.ct_sets[idx];
        let mut out = [ArgConstraint::default(); MAX_OP_ARGS];
        out[..c.len()].copy_from_slice(c);
        Ok(out)
    }

    /// `tcg_reg_alloc_op`.
    fn alloc_op<T: Target>(&mut self, tg: &mut T, id: OpId, op: &Op) -> Result<(), T::Error> {
        let def = op.opc.def();
        let nb_oargs = def.nb_oargs as usize;
        let nb_iargs = def.nb_iargs as usize;
        let flags = def.flags | tg.extra_op_flags(self.f, op);
        let mut new_args = [0u64; MAX_OP_ARGS];
        let mut const_args = [false; MAX_OP_ARGS];
        let nargs = (op.nargs as usize).max(nb_oargs + nb_iargs);
        new_args[nb_oargs + nb_iargs..nargs].copy_from_slice(&op.args[nb_oargs + nb_iargs..nargs]);

        let mut i_allocated = self.reserved;
        let mut o_allocated = self.reserved;
        let cond_at = match op.opc {
            Opcode::Brcond => Some(2),
            Opcode::Setcond | Opcode::Negsetcond | Opcode::CmpVec => Some(3),
            Opcode::Movcond | Opcode::CmpselVec => Some(5),
            _ => None,
        };
        let op_cond = cond_at.and_then(|k| Cond::from_u64(op.args[k])).unwrap_or(Cond::Always);
        let args_ct = self.constraints(tg, op)?;

        for k in 0..nb_iargs {
            let i = args_ct[nb_oargs + k].sort_index as usize;
            let a = args_ct[i];
            let ts = op.arg_temp(i);
            let td = self.td(ts);
            if let Val::Const(v) = self.val(ts) {
                if let Some(z) = tg.zero_reg() {
                    if v == 0 && a.ct & ct::REG_ZERO != 0 {
                        new_args[i] = z as u64;
                        const_args[i] = false;
                        continue;
                    }
                }
                if tg.const_match(v, a.ct, td.ty, op_cond, op.vece as u32) {
                    new_args[i] = v as u64;
                    const_args[i] = true;
                    continue;
                }
            }
            let mut preferred = RegSet::EMPTY;
            let required = a.regs;
            let mut allocate_new = false;
            if a.ialias {
                preferred = RegSet::EMPTY;
                if readonly(td) || !op.is_dead_arg(i) || args_ct[a.alias_index as usize].newreg {
                    allocate_new = true;
                } else if let Val::Reg(r) = self.val(ts) {
                    allocate_new = i_allocated.contains(r);
                }
            }
            let mut reg = 0;
            if !allocate_new {
                self.load(tg, ts, required, i_allocated, preferred);
                if let Val::Reg(r) = self.val(ts) {
                    reg = r;
                }
                allocate_new = !required.contains(reg);
            }
            if allocate_new {
                self.load(tg, ts, tg.available_regs(td.ty), i_allocated, RegSet::EMPTY);
                reg = self.reg_alloc(tg, required, i_allocated, preferred);
                if let Val::Reg(cur) = self.val(ts) {
                    if !tg.out_mov(td.ty, reg, cur) {
                        self.sync(tg, ts, i_allocated, RegSet::EMPTY, 0);
                        let (base, off) = tg.temp_home(self.f, ts);
                        tg.out_ld(td.ty, reg, base, off);
                    }
                }
            }
            new_args[i] = reg as u64;
            const_args[i] = false;
            i_allocated = i_allocated.with(reg);
        }

        for i in nb_oargs..nb_oargs + nb_iargs {
            if op.is_dead_arg(i) {
                self.dead(op.arg_temp(i));
            }
        }

        let mut stub = None;
        if flags & opf::COND_BRANCH != 0 {
            if op.opc == Opcode::Brcond && tg.out_of_line_branches() {
                stub = self.cbranch_stub(tg, id, i_allocated, op.arg_label(3));
            } else {
                self.cbranch(tg, i_allocated);
            }
        } else if flags & opf::BB_END != 0 {
            self.bb_end(tg, i_allocated);
        } else {
            if flags & opf::CALL_CLOBBER != 0 {
                for r in tg.call_clobber_regs().iter() {
                    self.reg_free(tg, r, i_allocated);
                }
            }
            if flags & opf::SIDE_EFFECTS != 0 {
                self.sync_globals(tg, i_allocated);
            }
            for k in 0..nb_oargs {
                let i = args_ct[k].sort_index as usize;
                let a = args_ct[i];
                let ts = op.arg_temp(i);
                if readonly(self.td(ts)) {
                    return Err(tg.bad_ir(format!("{}: writes a constant", op.opc.name())));
                }
                let reg = if a.oalias && !const_args[a.alias_index as usize] {
                    new_args[a.alias_index as usize] as Reg
                } else if a.newreg {
                    self.reg_alloc(tg, a.regs, i_allocated.union(o_allocated), RegSet::EMPTY)
                } else {
                    self.reg_alloc(tg, a.regs, o_allocated, RegSet::EMPTY)
                };
                o_allocated = o_allocated.with(reg);
                // An aliased input register can still be owned by a temp that is not dead;
                // write it back before the op overwrites it.
                if let Some(other) = self.reg_to_temp[reg as usize] {
                    if other != ts {
                        self.sync(tg, other, i_allocated.union(o_allocated), RegSet::EMPTY, -1);
                    }
                }
                self.set_reg(ts, reg);
                self.coherent[ts.index()] = false;
                new_args[i] = reg as u64;
            }
        }

        match stub {
            Some(l) => {
                let mut op = *op;
                op.args[3] = u64::from(l);
                tg.out_op(self.f, id, &op, &new_args[..nargs], &const_args[..nargs])?;
            }
            None => tg.out_op(self.f, id, op, &new_args[..nargs], &const_args[..nargs])?,
        }

        for i in 0..nb_oargs {
            let ts = op.arg_temp(i);
            if op.need_sync_arg(i) {
                let fd = if op.is_dead_arg(i) { 1 } else { 0 };
                self.sync(tg, ts, o_allocated, RegSet::EMPTY, fd);
            } else if op.is_dead_arg(i) {
                self.dead(ts);
            }
        }
        Ok(())
    }

    /// `tcg_reg_alloc_call`, with every argument and result passed in memory.
    fn alloc_call<T: Target>(&mut self, tg: &mut T, op: &Op) -> Result<(), T::Error> {
        let no = op.callo as usize;
        let ni = op.calli as usize;
        let flags = self.f.helper_info(op.call_helper()).flags;
        let allocated = self.reserved;
        for k in 0..ni {
            let ts = op.arg_temp(no + k);
            let ty = self.td(ts).ty;
            let (base, off) = tg.call_arg_home(k);
            if let Val::Const(v) = self.val(ts) {
                let v = if ty == Type::I32 { v as u32 as i64 } else { v };
                if tg.out_sti(Type::I64, v, base, off) {
                    continue;
                }
            }
            self.load(tg, ts, tg.available_regs(ty), allocated, RegSet::EMPTY);
            if let Val::Reg(r) = self.val(ts) {
                tg.out_st(Type::I64, r, base, off);
            }
        }
        for k in no..no + ni {
            if op.is_dead_arg(k) {
                self.dead(op.arg_temp(k));
            }
        }
        for r in tg.call_clobber_regs().iter() {
            self.reg_free(tg, r, allocated);
        }
        if flags & call_flags::NO_READ_GLOBALS != 0 {
            // Nothing to do.
        } else if flags & call_flags::NO_WRITE_GLOBALS != 0 {
            self.sync_globals(tg, allocated);
        } else {
            self.save_globals(tg, allocated);
        }
        tg.out_call(self.f, op)?;
        let mut o_allocated = allocated;
        for k in 0..no {
            let ts = op.arg_temp(k);
            if readonly(self.td(ts)) {
                return Err(tg.bad_ir("call: writes a constant".into()));
            }
            let ty = self.td(ts).ty;
            let reg = self.reg_alloc(tg, tg.available_regs(ty), o_allocated, RegSet::EMPTY);
            let (base, off) = tg.call_arg_home(k);
            tg.out_ld(ty, reg, base, off);
            o_allocated = o_allocated.with(reg);
            self.set_reg(ts, reg);
            self.coherent[ts.index()] = false;
        }
        for k in 0..no {
            let ts = op.arg_temp(k);
            if op.need_sync_arg(k) {
                let fd = if op.is_dead_arg(k) { 1 } else { 0 };
                self.sync(tg, ts, o_allocated, RegSet::EMPTY, fd);
            } else if op.is_dead_arg(k) {
                self.dead(ts);
            }
        }
        Ok(())
    }
}

const TS_DEAD: u8 = 1;
const TS_MEM: u8 = 2;

/// Compute `op.life` for every op, `liveness_pass_1` without its rewrites: no op is removed
/// or lowered. `extra_flags` adds target flags to an op, as [`Target::extra_op_flags`] does.
pub fn liveness(f: &mut Func, extra_flags: &dyn Fn(&Func, &Op) -> u32) {
    let (lives, _) = scan(f, extra_flags);
    let mut cur = f.first_op();
    while let Some(id) = cur {
        cur = f.next_op(id);
        f.op_mut(id).life = lives[id.index()];
    }
}

/// What [`analyze`] finds: `op.life` for every op, and the globals each conditional branch
/// leaves for its stub to store.
#[derive(Clone, Debug, Default)]
pub struct Liveness {
    /// `op.life`, indexed by [`OpId`].
    life: Vec<u32>,
    drops: Drops,
}

impl Liveness {
    /// The `op.life` [`liveness`] would give `id`.
    pub fn life(&self, id: OpId) -> u32 {
        self.life[id.index()]
    }
}

/// [`liveness`] without writing the result into `f`, for [`reg_alloc_live`].
pub fn analyze(f: &Func, extra_flags: &dyn Fn(&Func, &Op) -> u32) -> Liveness {
    let (life, drops) = scan(f, extra_flags);
    Liveness { life, drops }
}

/// The pass behind [`liveness`]: `op.life` for every op, and for each conditional branch the
/// globals its fall through path neither reads nor needs in memory before setting them again.
/// The globals each conditional branch leaves for its stub to store, which the fall through
/// path frees after the branch.
type Drops = FastHashMap<OpId, Vec<Temp>>;

/// Finds, for each conditional branch, the globals that may have been written since the
/// last point where every global was forced to memory. Only these can be dirty at the
/// branch, so only these need to stay live until it.
fn written_globals(
    f: &Func,
    extra_flags: &dyn Fn(&Func, &Op) -> u32,
) -> FastHashMap<OpId, Vec<usize>> {
    let ng = f.nb_globals();
    let global = |t: Temp| t.index() < ng && f.temp(t).kind == TempKind::Global;
    let mut map = FastHashMap::default();
    let mut written = vec![false; ng];
    let mut cur = f.first_op();
    while let Some(id) = cur {
        cur = f.next_op(id);
        let op = f.op(id);
        let no = match op.opc {
            Opcode::Call => {
                let flags = f.helper_info(op.call_helper()).flags;
                if flags & call_flags::NO_READ_GLOBALS == 0 {
                    written.fill(false);
                }
                op.callo as usize
            }
            Opcode::InsnStart | Opcode::Discard => 0,
            _ => {
                let def = op.opc.def();
                let flags = def.flags | extra_flags(f, op);
                if flags & opf::COND_BRANCH != 0 {
                    let w: Vec<usize> = (0..ng).filter(|&i| written[i]).collect();
                    if !w.is_empty() {
                        map.insert(id, w);
                    }
                } else if flags & (opf::BB_END | opf::BB_EXIT | opf::SIDE_EFFECTS) != 0 {
                    written.fill(false);
                }
                def.nb_oargs as usize
            }
        };
        for i in 0..no {
            let t = op.arg_temp(i);
            if global(t) {
                written[t.index()] = true;
            }
        }
    }
    map
}

fn scan(f: &Func, extra_flags: &dyn Fn(&Func, &Op) -> u32) -> (Vec<u32>, Drops) {
    let mut lives = vec![0u32; f.op_slots()];
    let mut drops = FastHashMap::default();
    let ng = f.nb_globals();
    let kinds: Vec<TempKind> = f.temps().iter().map(|t| t.kind).collect();
    let nt = kinds.len();
    let mut st = vec![0u8; nt];
    let func_end = |st: &mut [u8]| {
        for (i, s) in st.iter_mut().enumerate() {
            *s = if i < ng { TS_DEAD | TS_MEM } else { TS_DEAD };
        }
    };
    let global_sync = |st: &mut [u8]| {
        for s in &mut st[..ng] {
            *s |= TS_MEM;
        }
    };
    func_end(&mut st);
    let written = written_globals(f, extra_flags);

    let mut cur = f.last_op();
    while let Some(id) = cur {
        cur = f.prev_op(id);
        let op = *f.op(id);
        let mut life = 0u32;
        let at = |i: usize| op.arg_temp(i).index();
        match op.opc {
            Opcode::Call => {
                let flags = f.helper_info(op.call_helper()).flags;
                let no = op.callo as usize;
                let ni = op.calli as usize;
                for i in 0..no {
                    let s = &mut st[at(i)];
                    if *s & TS_DEAD != 0 {
                        life |= DEAD_ARG << i;
                    }
                    if *s & TS_MEM != 0 {
                        life |= SYNC_ARG << i;
                    }
                    *s = TS_DEAD;
                }
                if flags & (call_flags::NO_WRITE_GLOBALS | call_flags::NO_READ_GLOBALS) == 0 {
                    for s in &mut st[..ng] {
                        *s = TS_DEAD | TS_MEM;
                    }
                } else if flags & call_flags::NO_READ_GLOBALS == 0 {
                    global_sync(&mut st);
                }
                for i in no..no + ni {
                    if st[at(i)] & TS_DEAD != 0 {
                        life |= DEAD_ARG << i;
                    }
                }
                for i in no..no + ni {
                    st[at(i)] &= !TS_DEAD;
                }
            }
            Opcode::InsnStart => {}
            Opcode::Discard => st[at(0)] = TS_DEAD,
            _ => {
                let def = op.opc.def();
                let flags = def.flags | extra_flags(f, &op);
                let no = def.nb_oargs as usize;
                let ni = def.nb_iargs as usize;
                for i in 0..no {
                    let s = &mut st[at(i)];
                    if *s & TS_DEAD != 0 {
                        life |= DEAD_ARG << i;
                    }
                    if *s & TS_MEM != 0 {
                        life |= SYNC_ARG << i;
                    }
                    *s = TS_DEAD;
                }
                if flags & opf::BB_EXIT != 0 {
                    func_end(&mut st);
                } else if flags & opf::COND_BRANCH != 0 {
                    // A global that may be dirty here and is dead on the fallthrough is left
                    // in its register for the branch's stub to store, saving the store on the
                    // fallthrough. The others must be in memory, as in QEMU.
                    let dirty = written.get(&id).map(Vec::as_slice).unwrap_or_default();
                    let dead: Vec<usize> =
                        dirty.iter().copied().filter(|&i| st[i] == TS_DEAD).collect();
                    global_sync(&mut st);
                    for &i in &dead {
                        st[i] = 0;
                    }
                    if !dead.is_empty() {
                        drops.insert(id, dead.into_iter().map(Temp::from_index).collect());
                    }
                    for (s, k) in st.iter_mut().zip(&kinds).skip(ng) {
                        if *k == TempKind::Tb {
                            *s |= TS_MEM;
                        }
                    }
                } else if flags & opf::BB_END != 0 {
                    for (s, k) in st.iter_mut().zip(&kinds) {
                        *s = match k {
                            TempKind::Fixed | TempKind::Global | TempKind::Tb => TS_DEAD | TS_MEM,
                            TempKind::Ebb | TempKind::Const => TS_DEAD,
                        };
                    }
                } else if flags & opf::SIDE_EFFECTS != 0 {
                    global_sync(&mut st);
                }
                for i in no..no + ni {
                    if st[at(i)] & TS_DEAD != 0 {
                        life |= DEAD_ARG << i;
                    }
                }
                for i in no..no + ni {
                    st[at(i)] &= !TS_DEAD;
                }
            }
        }
        lives[id.index()] = life;
    }
    (lives, drops)
}

/// Get a copy of `f` ready for [`reg_alloc`]: indirect globals lowered to explicit loads and
/// stores (`liveness_pass_2`) and `op.life` computed by [`liveness`].
pub fn prepare(f: &Func, extra_flags: &dyn Fn(&Func, &Op) -> u32) -> Func {
    let mut g = f.clone();
    liveness(&mut g, extra_flags);
    let used_indirect = g.ops().any(|(_, op)| {
        let n = op.nb_oargs() + op.nb_iargs();
        (0..n).any(|i| g.temp(op.arg_temp(i)).indirect_reg)
    });
    if used_indirect && g.liveness_pass_2() {
        liveness(&mut g, extra_flags);
    }
    g
}

/// [`prepare`] for [`reg_alloc_live`]: `f` itself when it uses no indirect global, which is
/// the common case and needs no copy, and the lowered copy otherwise, with its liveness.
pub fn prepare_live<'f>(
    f: &'f Func,
    extra_flags: &dyn Fn(&Func, &Op) -> u32,
) -> (Cow<'f, Func>, Liveness) {
    let used_indirect = f.nb_indirects() > 0
        && f.ops().any(|(_, op)| {
            let n = op.nb_oargs() + op.nb_iargs();
            (0..n).any(|i| f.temp(op.arg_temp(i)).indirect_reg)
        });
    if !used_indirect {
        return (Cow::Borrowed(f), analyze(f, extra_flags));
    }
    let g = prepare(f, extra_flags);
    let live = analyze(&g, extra_flags);
    (Cow::Owned(g), live)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn letters(c: char) -> Option<Letter> {
        match c {
            'r' => Some(Letter::Regs(RegSet(0xffff))),
            'w' => Some(Letter::Regs(RegSet(0xffff_0000))),
            'A' => Some(Letter::Const(0x100)),
            _ => None,
        }
    }

    #[test]
    fn analyze_matches_liveness() {
        use crate::ir::FuncConfig;
        let mut f = Func::new(FuncConfig::default());
        let env = f.env();
        let r0 = f.global_mem_new_i64(env, 0, "r0");
        let r1 = f.global_mem_new_i64(env, 8, "r1");
        let t = f.temp_new_i64();
        let l = f.new_label();
        f.gen_add_i64(t, r0, r1);
        f.gen_addi_i64(r0, t, 1);
        f.gen_brcondi_i64(Cond::Eq, t, 0, l);
        f.gen_addi_i64(r0, r1, 2);
        f.gen_set_label(l);
        f.gen_movi_i64(r1, 3);
        f.gen_exit_tb(0, 0);
        let none = |_: &Func, _: &Op| 0;
        let (g, live) = prepare_live(&f, &none);
        assert!(matches!(g, Cow::Borrowed(_)), "no indirect globals, so no copy");
        let mut want = f.clone();
        liveness(&mut want, &none);
        let mut n = 0;
        for (id, op) in want.ops() {
            assert_eq!(live.life(id), op.life, "{}", op.opc.name());
            n += 1;
        }
        assert_eq!(n, f.nb_ops());
    }

    #[test]
    fn constraints_parse_and_sort() {
        let c = parse_constraints(&["r", "0", "rA"], 1, &letters).unwrap();
        assert!(c[0].oalias && c[0].alias_index == 1);
        assert!(c[1].ialias && c[1].alias_index == 0);
        assert_eq!(c[2].ct, 0x100);
        assert_eq!(c[2].regs, RegSet(0xffff));
        // The aliased input goes first.
        assert_eq!(c[1].sort_index, 1);
        let c = parse_constraints(&["&r", "rz", "w"], 1, &letters).unwrap();
        assert!(c[0].newreg);
        assert_eq!(c[1].ct, ct::REG_ZERO);
        assert!(parse_constraints(&["r", "q"], 1, &letters).is_err());
        assert!(parse_constraints(&["r", "1"], 1, &letters).is_err());
    }

    #[test]
    fn regsets() {
        let s = RegSet::single(3).with(5);
        assert!(s.contains(3) && s.contains(5) && !s.contains(4));
        assert_eq!(s.len(), 2);
        assert_eq!(s.first(), Some(3));
        assert_eq!(s.iter().collect::<Vec<_>>(), [3, 5]);
        assert_eq!(s.minus(RegSet::single(3)), RegSet::single(5));
    }
}
