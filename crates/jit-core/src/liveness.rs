// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reachability and liveness, a port of `reachable_code_pass` and `liveness_pass_0` to
//! `liveness_pass_2` from `tcg/tcg.c`.
//!
//! Register preferences are not computed, since they only guide a real register allocator. The
//! results that matter for the IR are the same as QEMU's: dead ops are removed, `op.life` holds
//! the `SYNC_ARG` and `DEAD_ARG` bits, carry ops whose carry is unused are lowered, and globals
//! reached through another global are replaced by direct temps with explicit loads and stores.

use crate::ir::{DEAD_ARG, Func, OpId, SYNC_ARG, Temp, TempData};
use crate::opcode::Opcode;
use crate::types::{TempKind, call_flags, opf};

const TS_DEAD: u8 = 1;
const TS_MEM: u8 = 2;

struct Live {
    state: Vec<u8>,
    carry_live: bool,
}

impl Live {
    fn st(&mut self, f: &Func, t: Temp) -> &mut u8 {
        if t.index() >= self.state.len() {
            self.state.resize(f.nb_temps(), TS_DEAD);
        }
        &mut self.state[t.index()]
    }

    fn func_end(&mut self, ng: usize) {
        for (i, s) in self.state.iter_mut().enumerate() {
            *s = if i < ng { TS_DEAD | TS_MEM } else { TS_DEAD };
        }
    }

    fn bb_end(&mut self, temps: &[TempData]) {
        for (s, td) in self.state.iter_mut().zip(temps) {
            *s = match td.kind {
                TempKind::Fixed | TempKind::Global | TempKind::Tb => TS_DEAD | TS_MEM,
                TempKind::Ebb | TempKind::Const => TS_DEAD,
            };
        }
    }

    fn global_sync(&mut self, ng: usize) {
        for s in &mut self.state[..ng] {
            *s |= TS_MEM;
        }
    }

    fn bb_sync(&mut self, ng: usize, temps: &[TempData]) {
        self.global_sync(ng);
        for (s, td) in self.state.iter_mut().zip(temps).skip(ng) {
            if td.kind == TempKind::Tb {
                *s |= TS_MEM;
            }
        }
    }

    fn global_kill(&mut self, ng: usize) {
        for s in &mut self.state[..ng] {
            *s = TS_DEAD | TS_MEM;
        }
    }
}

impl Func {
    /// `reachable_code_pass`: remove code after unconditional exits, labels nobody branches to,
    /// and branches to the very next op.
    pub fn reachable_code_pass(&mut self) {
        let mut dead = false;
        let mut cur = self.first_op();
        while let Some(id) = cur {
            cur = self.next_op(id);
            let mut remove = dead;
            let op = *self.op(id);
            match op.opc {
                Opcode::SetLabel => {
                    let label = op.arg_label(0);
                    let mut prev = self.prev_op(id);
                    if let Some(p) = prev {
                        if self.op(p).opc == Opcode::SetLabel {
                            let from = self.op(p).arg_label(0);
                            self.move_label_uses(label, from);
                            self.remove_op(p);
                            prev = self.prev_op(id);
                        }
                    }
                    if let Some(p) = prev {
                        let po = self.op(p);
                        if po.opc == Opcode::Br && po.arg_label(0) == label {
                            self.remove_op(p);
                            dead = false;
                        }
                    }
                    if self.label(label).branches.is_empty() {
                        remove = true;
                    } else {
                        dead = false;
                        remove = false;
                    }
                }
                Opcode::Br | Opcode::ExitTb | Opcode::GotoPtr => dead = true,
                Opcode::Call => {
                    if self.helper_info(op.call_helper()).flags & call_flags::NO_RETURN != 0 {
                        dead = true;
                    }
                }
                Opcode::InsnStart => remove = false,
                _ => {}
            }
            if remove {
                self.remove_op(id);
            }
        }
    }

    /// `liveness_pass_0`: TB temps used within a single extended basic block become EBB temps.
    pub fn liveness_pass_0(&mut self) {
        #[derive(Clone, Copy, PartialEq)]
        enum Use {
            None,
            One(OpId),
            Many,
        }
        let ng = self.nb_globals();
        let mut uses = vec![Use::None; self.nb_temps()];
        let Some(mut ebb) = self.first_op() else { return };
        for (id, op) in self.ops() {
            match op.opc {
                Opcode::SetLabel => {
                    ebb = id;
                    continue;
                }
                Opcode::Discard => continue,
                _ => {}
            }
            for i in 0..op.nb_oargs() + op.nb_iargs() {
                let t = op.arg_temp(i);
                if self.temp(t).kind != TempKind::Tb {
                    continue;
                }
                let u = &mut uses[t.index()];
                match *u {
                    Use::None => *u = Use::One(ebb),
                    Use::One(e) if e != ebb => *u = Use::Many,
                    _ => {}
                }
            }
        }
        for (i, u) in uses.iter().enumerate().skip(ng) {
            if self.temps[i].kind == TempKind::Tb && *u != Use::Many {
                self.temp_mut(Temp::from_index(i)).kind = TempKind::Ebb;
            }
        }
    }

    /// `liveness_pass_1`: compute `op.life`, remove dead ops, and lower carry and double word
    /// ops whose extra result is unused.
    pub fn liveness_pass_1(&mut self) {
        let ng = self.nb_globals();
        let mut lv = Live { state: vec![0; self.nb_temps()], carry_live: false };
        lv.func_end(ng);

        let mut cur = self.last_op();
        while let Some(id) = cur {
            cur = self.prev_op(id);
            let mut arg_life: u32 = 0;
            let mut opc = self.op(id).opc;

            if opc == Opcode::Call {
                debug_assert!(!lv.carry_live, "carry live across a call");
                let op = *self.op(id);
                let flags = self.helper_info(op.call_helper()).flags;
                let nb_oargs = op.callo as usize;
                let nb_iargs = op.calli as usize;
                if flags & call_flags::NO_SIDE_EFFECTS != 0
                    && (0..nb_oargs).all(|i| *lv.st(self, op.arg_temp(i)) == TS_DEAD)
                {
                    self.remove_op(id);
                    continue;
                }
                for i in 0..nb_oargs {
                    let s = lv.st(self, op.arg_temp(i));
                    if *s & TS_DEAD != 0 {
                        arg_life |= DEAD_ARG << i;
                    }
                    if *s & TS_MEM != 0 {
                        arg_life |= SYNC_ARG << i;
                    }
                    *s = TS_DEAD;
                }
                if flags & (call_flags::NO_WRITE_GLOBALS | call_flags::NO_READ_GLOBALS) == 0 {
                    lv.global_kill(ng);
                } else if flags & call_flags::NO_READ_GLOBALS == 0 {
                    lv.global_sync(ng);
                }
                for i in nb_oargs..nb_oargs + nb_iargs {
                    if *lv.st(self, op.arg_temp(i)) & TS_DEAD != 0 {
                        arg_life |= DEAD_ARG << i;
                    }
                }
                for i in nb_oargs..nb_oargs + nb_iargs {
                    *lv.st(self, op.arg_temp(i)) &= !TS_DEAD;
                }
                self.op_mut(id).life = arg_life;
                continue;
            }

            enum Next {
                Default,
                Keep,
                Remove,
            }
            let mut next = Next::Default;
            match opc {
                Opcode::InsnStart => {
                    debug_assert!(!lv.carry_live, "carry live across an instruction");
                    self.op_mut(id).life = 0;
                    continue;
                }
                Opcode::Discard => {
                    let t = self.op(id).arg_temp(0);
                    *lv.st(self, t) = TS_DEAD;
                    self.op_mut(id).life = 0;
                    continue;
                }
                Opcode::Muls2 | Opcode::Mulu2 => {
                    debug_assert!(!lv.carry_live);
                    let (new1, new2) = if opc == Opcode::Muls2 {
                        (Opcode::Mul, Opcode::Mulsh)
                    } else {
                        (Opcode::Mul, Opcode::Muluh)
                    };
                    let op = *self.op(id);
                    let lo_dead = *lv.st(self, op.arg_temp(0)) == TS_DEAD;
                    let hi_dead = *lv.st(self, op.arg_temp(1)) == TS_DEAD;
                    if hi_dead {
                        if lo_dead {
                            next = Next::Remove;
                        } else {
                            let o = self.op_mut(id);
                            o.opc = new1;
                            o.args[1] = o.args[2];
                            o.args[2] = o.args[3];
                            o.nargs = 3;
                            opc = new1;
                            next = Next::Keep;
                        }
                    } else if lo_dead {
                        let o = self.op_mut(id);
                        o.opc = new2;
                        o.args[0] = o.args[1];
                        o.args[1] = o.args[2];
                        o.args[2] = o.args[3];
                        o.nargs = 3;
                        opc = new2;
                        next = Next::Keep;
                    } else {
                        next = Next::Keep;
                    }
                }
                Opcode::Addco | Opcode::Addcio | Opcode::Subbio => {
                    if lv.carry_live {
                        next = Next::Keep;
                    } else {
                        opc = match opc {
                            Opcode::Addco => Opcode::Add,
                            Opcode::Addcio => Opcode::Addci,
                            _ => Opcode::Subbi,
                        };
                        self.op_mut(id).opc = opc;
                    }
                }
                Opcode::Subbo => {
                    if lv.carry_live {
                        next = Next::Keep;
                    } else {
                        opc = Opcode::Sub;
                        self.op_mut(id).opc = opc;
                        let t = self.op(id).arg_temp(2);
                        let td = self.temp(t);
                        if td.kind == TempKind::Const {
                            let (ty, val) = (td.ty, td.val);
                            let n = self.constant_internal(ty, val.wrapping_neg());
                            lv.st(self, n);
                            let o = self.op_mut(id);
                            o.args[2] = n.arg();
                            o.opc = Opcode::Add;
                            opc = Opcode::Add;
                        }
                    }
                }
                Opcode::Addc1o | Opcode::Subb1o => {
                    if lv.carry_live {
                        next = Next::Keep;
                    } else {
                        let (first, c) = if opc == Opcode::Addc1o {
                            (Opcode::Add, 1)
                        } else {
                            (Opcode::Sub, -1)
                        };
                        let op = *self.op(id);
                        let p = self.insert_before(id, first, op.ty, 3);
                        self.op_mut(p).args[..3].copy_from_slice(&op.args[..3]);
                        let ty = self.temp(op.arg_temp(0)).ty;
                        let k = self.constant_internal(ty, c);
                        lv.st(self, k);
                        let o = self.op_mut(id);
                        o.opc = Opcode::Add;
                        o.args[1] = o.args[0];
                        o.args[2] = k.arg();
                        opc = Opcode::Add;
                        cur = Some(p);
                    }
                }
                _ => {}
            }

            if let Next::Default = next {
                let def = opc.def();
                next = Next::Keep;
                if def.flags & opf::SIDE_EFFECTS == 0 && def.nb_oargs != 0 {
                    let op = *self.op(id);
                    if (0..def.nb_oargs as usize).all(|i| *lv.st(self, op.arg_temp(i)) == TS_DEAD) {
                        next = Next::Remove;
                    }
                }
            }
            if let Next::Remove = next {
                self.remove_op(id);
                continue;
            }

            let def = opc.def();
            let nb_oargs = def.nb_oargs as usize;
            let nb_iargs = def.nb_iargs as usize;
            let op = *self.op(id);
            for i in 0..nb_oargs {
                let s = lv.st(self, op.arg_temp(i));
                if *s & TS_DEAD != 0 {
                    arg_life |= DEAD_ARG << i;
                }
                if *s & TS_MEM != 0 {
                    arg_life |= SYNC_ARG << i;
                }
                *s = TS_DEAD;
            }
            // Make sure every temp has a state before the block helpers walk them.
            if lv.state.len() < self.nb_temps() {
                lv.state.resize(self.nb_temps(), TS_DEAD);
            }
            if def.flags & opf::BB_EXIT != 0 {
                debug_assert!(!lv.carry_live);
                lv.func_end(ng);
            } else if def.flags & opf::COND_BRANCH != 0 {
                debug_assert!(!lv.carry_live);
                lv.bb_sync(ng, &self.temps);
            } else if def.flags & opf::BB_END != 0 {
                debug_assert!(!lv.carry_live);
                lv.bb_end(&self.temps);
            } else if def.flags & opf::SIDE_EFFECTS != 0 {
                debug_assert!(!lv.carry_live);
                lv.global_sync(ng);
            }
            for i in nb_oargs..nb_oargs + nb_iargs {
                if *lv.st(self, op.arg_temp(i)) & TS_DEAD != 0 {
                    arg_life |= DEAD_ARG << i;
                }
            }
            if def.flags & opf::CARRY_OUT != 0 {
                lv.carry_live = false;
            }
            for i in nb_oargs..nb_oargs + nb_iargs {
                *lv.st(self, op.arg_temp(i)) &= !TS_DEAD;
            }
            if def.flags & opf::CARRY_IN != 0 {
                lv.carry_live = true;
            }
            self.op_mut(id).life = arg_life;
        }
        debug_assert!(!lv.carry_live, "carry live at the start of the block");
    }

    /// `liveness_pass_2`: replace indirect globals by direct EBB temps with explicit loads and
    /// stores. Returns true if anything changed.
    pub fn liveness_pass_2(&mut self) -> bool {
        let ng = self.nb_globals();
        let mut dir: Vec<Option<Temp>> = vec![None; self.nb_temps()];
        for (i, d) in dir.iter_mut().enumerate().take(ng) {
            let its = &self.temps[i];
            if its.indirect_reg {
                let mut td = TempData::new(TempKind::Ebb, its.ty, its.base_type);
                td.subindex = its.subindex;
                td.allocated = true;
                *d = Some(self.temp_alloc(td));
            }
        }
        let mut state = vec![TS_DEAD; self.nb_temps()];
        let mut changes = false;

        let mut cur = self.first_op();
        while let Some(id) = cur {
            cur = self.next_op(id);
            let op = *self.op(id);
            let opc = op.opc;
            let def = opc.def();
            let arg_life = op.life;
            let (nb_oargs, nb_iargs, cflags) = if opc == Opcode::Call {
                (op.callo as usize, op.calli as usize, self.helper_info(op.call_helper()).flags)
            } else {
                let f = if def.flags & opf::COND_BRANCH != 0 {
                    call_flags::NO_WRITE_GLOBALS
                } else if def.flags & opf::BB_END != 0 {
                    0
                } else if def.flags & opf::SIDE_EFFECTS != 0 {
                    call_flags::NO_WRITE_GLOBALS
                } else {
                    call_flags::NO_READ_GLOBALS | call_flags::NO_WRITE_GLOBALS
                };
                (def.nb_oargs as usize, def.nb_iargs as usize, f)
            };

            for i in nb_oargs..nb_oargs + nb_iargs {
                let t = op.arg_temp(i);
                let Some(Some(d)) = dir.get(t.index()) else { continue };
                if state[t.index()] == TS_DEAD {
                    let its = &self.temps[t.index()];
                    let (ty, base, off) = (its.ty, its.mem_base, its.mem_offset);
                    let base = base.expect("indirect global without a base");
                    let l = self.insert_before(id, Opcode::Ld, ty, 3);
                    let lo = self.op_mut(l);
                    lo.args[0] = d.arg();
                    lo.args[1] = base.arg();
                    lo.args[2] = off as u64;
                    state[t.index()] = TS_MEM;
                }
            }
            for i in nb_oargs..nb_oargs + nb_iargs {
                let t = op.arg_temp(i);
                let Some(Some(d)) = dir.get(t.index()) else { continue };
                self.op_mut(id).args[i] = d.arg();
                changes = true;
                if arg_life & (DEAD_ARG << i) != 0 {
                    state[t.index()] = TS_DEAD;
                }
            }

            if cflags & call_flags::NO_READ_GLOBALS == 0 {
                let need_dead = cflags & call_flags::NO_WRITE_GLOBALS == 0;
                for (i, d) in dir.iter().enumerate().take(ng) {
                    debug_assert!(
                        d.is_none() || if need_dead { state[i] == TS_DEAD } else { state[i] != 0 },
                        "liveness left an indirect global unsynced"
                    );
                }
            }

            if opc == Opcode::Mov {
                let t = op.arg_temp(0);
                if let Some(Some(d)) = dir.get(t.index()) {
                    let d = *d;
                    self.op_mut(id).args[0] = d.arg();
                    changes = true;
                    state[t.index()] = 0;
                    if arg_life & SYNC_ARG != 0 {
                        let its = &self.temps[t.index()];
                        let (ty, base, off) = (its.ty, its.mem_base, its.mem_offset);
                        let base = base.expect("indirect global without a base");
                        let s = self.insert_after(id, Opcode::St, ty, 3);
                        let mut out = d;
                        if arg_life & DEAD_ARG != 0 {
                            out = self.op(id).arg_temp(1);
                            state[t.index()] = TS_DEAD;
                            self.remove_op(id);
                        } else {
                            state[t.index()] = TS_MEM;
                        }
                        let so = self.op_mut(s);
                        so.args[0] = out.arg();
                        so.args[1] = base.arg();
                        so.args[2] = off as u64;
                    }
                }
            } else {
                for i in 0..nb_oargs {
                    let t = op.arg_temp(i);
                    let Some(Some(d)) = dir.get(t.index()) else { continue };
                    let d = *d;
                    self.op_mut(id).args[i] = d.arg();
                    changes = true;
                    state[t.index()] = 0;
                    if arg_life & (SYNC_ARG << i) != 0 {
                        let its = &self.temps[t.index()];
                        let (ty, base, off) = (its.ty, its.mem_base, its.mem_offset);
                        let base = base.expect("indirect global without a base");
                        let s = self.insert_after(id, Opcode::St, ty, 3);
                        let so = self.op_mut(s);
                        so.args[0] = d.arg();
                        so.args[1] = base.arg();
                        so.args[2] = off as u64;
                        state[t.index()] = TS_MEM;
                    }
                    if arg_life & (DEAD_ARG << i) != 0 {
                        state[t.index()] = TS_DEAD;
                    }
                }
            }
        }
        changes
    }
}

/// Which dumps [`Func::gen_code`] produces, like QEMU's `-d op,op_ind,op_opt`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LogMask {
    /// `-d op`: the ops as generated.
    pub op: bool,
    /// `-d op_ind`: the ops before indirect globals are lowered.
    pub op_ind: bool,
    /// `-d op_opt`: the ops after optimization and liveness.
    pub op_opt: bool,
}

impl LogMask {
    /// Every dump.
    pub const ALL: LogMask = LogMask { op: true, op_ind: true, op_opt: true };
}

impl Func {
    /// The middle end of `tcg_gen_code`: optimize if asked, then remove unreachable code and
    /// run liveness. The requested dumps are returned in QEMU's log format.
    pub fn gen_code(&mut self, optimize: bool, log: LogMask) -> String {
        let mut out = String::new();
        if log.op {
            out.push_str("OP:\n");
            out.push_str(&self.dump_ops(false));
            out.push('\n');
        }
        self.temp_ebb_reset_freed();
        if optimize {
            self.optimize();
        }
        self.reachable_code_pass();
        self.liveness_pass_0();
        self.liveness_pass_1();
        if self.nb_indirects() > 0 {
            if log.op_ind {
                out.push_str("OP before indirect lowering:\n");
                out.push_str(&self.dump_ops(false));
                out.push('\n');
            }
            if self.liveness_pass_2() {
                self.liveness_pass_1();
            }
        }
        if log.op_opt {
            out.push_str("OP after optimization and liveness analysis:\n");
            out.push_str(&self.dump_ops(true));
            out.push('\n');
        }
        out
    }
}
