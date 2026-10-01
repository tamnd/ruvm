// SPDX-License-Identifier: MIT OR Apache-2.0

//! A portable interpreter for the ruvm JIT IR, used where no native backend exists and as a
//! reference in tests.
//!
//! [`Machine::run`] executes one [`Func`] against a CPU state byte buffer. Globals live in that
//! buffer at their offsets from `env`, which is offset 0; host pointers (the values of `env`,
//! pointer globals and pointer temps) are offsets into the same buffer, so the host `ld` and `st`
//! ops and indirect globals work on it too. `qemu_ld` and `qemu_st` go through a
//! [`GuestMemory`], and `call` ops through a [`HelperRegistry`]. A run ends at `exit_tb`, at a
//! `goto_tb` whose slot is marked linked, at `goto_ptr`, or when a helper or a memory access
//! unwinds; see [`Exit`].
//!
//! Every op follows QEMU's semantics bit for bit. Where TCG leaves a result unspecified, the
//! interpreter picks the same answer as the constant folder of `tcg/optimize.c`, so that a block
//! gives the same result optimized or not:
//!
//! - I32 shifts and rotates use the count modulo 32, I64 ones modulo 64;
//! - `divs`, `divu`, `rems` and `remu` divide by one when the divisor is zero, and the most
//!   negative value divided by minus one wraps; `divs2` and `divu2` do the same and keep the low
//!   half of the quotient;
//! - `bswap16` and `bswap32` zero extend their result unless `TCG_BSWAP_OS` asks for a sign
//!   extension;
//! - vector shifts by a scalar or per element use the count modulo the element size;
//! - a vector op narrower than its output temp clears the rest of the temp.
//!
//! Differences from a QEMU backend:
//!
//! - Values of globals are never cached: every read and write of a global goes to the CPU state
//!   buffer, which is what QEMU's syncs amount to at every point where they are observable.
//! - The `atomic_*` helpers are not atomic; the interpreter is single threaded.
//! - `goto_ptr` does not look anything up; it returns the pointer, and the built-in
//!   `lookup_tb_ptr` always answers 0 (go back to the main loop).
//! - Plugin ops do nothing.

#![forbid(unsafe_code)]

pub mod helpers;
pub mod mem;
mod vector;

use std::fmt;

use ruvm_jit_core::ir::{Func, Op, Temp};
use ruvm_jit_core::opcode::Opcode;
use ruvm_jit_core::types::{Cond, INSN_START_WORDS, MemOpIdx, TempKind, Type, bswap};

pub use helpers::{HelperEntry, HelperEnv, HelperFn, HelperRegistry, Unwind};
pub use mem::{
    FaultKind, FlatMemory, GuestMemory, MemFault, NoMemory, guest_load, guest_load_env,
    guest_store, guest_store_env,
};

/// How a run of a translation block ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Exit {
    /// `exit_tb` with this value: the block pointer plus the exit index in the low two bits, or 0.
    ExitTb(u64),
    /// `goto_tb` on a slot marked linked in [`Machine::linked`].
    GotoTb(u32),
    /// `goto_ptr` with this pointer.
    GotoPtr(u64),
    /// A helper or a guest memory access left the block early.
    Unwind(Unwind),
}

/// The block could not be run. These are bugs in the IR or in the setup, not guest events.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InterpError {
    /// An access to the CPU state buffer is out of bounds.
    EnvOutOfBounds {
        /// The offset of the access.
        offset: u64,
        /// Its length in bytes.
        len: usize,
    },
    /// A call names a helper that is not registered.
    UnknownHelper(String),
    /// A call's declaration does not match the registered helper.
    HelperSignature(String),
    /// A branch names a label that is not set.
    BadLabel(u32),
    /// An op is malformed; the text says how.
    BadOp(String),
    /// The last op ran without leaving the block.
    FellOffEnd,
    /// More than [`Machine::step_limit`] ops ran.
    StepLimit,
}

impl fmt::Display for InterpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InterpError::EnvOutOfBounds { offset, len } => {
                write!(f, "access of {len} bytes at env offset {offset:#x} is out of bounds")
            }
            InterpError::UnknownHelper(n) => write!(f, "helper {n} is not registered"),
            InterpError::HelperSignature(n) => {
                write!(f, "helper {n} is registered with a different signature")
            }
            InterpError::BadLabel(l) => write!(f, "label $L{l} is not set"),
            InterpError::BadOp(s) => write!(f, "bad op: {s}"),
            InterpError::FellOffEnd => f.write_str("the block ended without an exit"),
            InterpError::StepLimit => f.write_str("the step limit was reached"),
        }
    }
}

impl std::error::Error for InterpError {}

type R<T> = Result<T, InterpError>;

/// An interpreter bound to one CPU state, guest memory and helper set.
pub struct Machine<'a> {
    /// The CPU state buffer. `env` is offset 0.
    pub env: &'a mut [u8],
    /// Guest memory for `qemu_ld` and `qemu_st`.
    pub mem: &'a mut dyn GuestMemory,
    /// The helpers `call` ops may name.
    pub helpers: &'a HelperRegistry,
    /// For each `goto_tb` slot, whether it is chained to another block. An unlinked `goto_tb`
    /// falls through to the next op, as it does in QEMU before the jump is patched.
    pub linked: [bool; 2],
    /// The most ops one run may execute.
    pub step_limit: u64,
    /// The words of the last `insn_start` executed in the last run.
    pub last_insn_start: Option<[u64; INSN_START_WORDS]>,
}

impl fmt::Debug for Machine<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Machine")
            .field("env_len", &self.env.len())
            .field("linked", &self.linked)
            .field("step_limit", &self.step_limit)
            .field("last_insn_start", &self.last_insn_start)
            .finish_non_exhaustive()
    }
}

/// The run time state of one run: temp values and the carry flag.
struct State {
    vals: Vec<[u64; 4]>,
    carry: bool,
}

fn wmask(w: u32) -> u64 {
    if w == 32 { 0xffff_ffff } else { !0 }
}

fn sext(v: u64, bits: u32) -> u64 {
    if bits >= 64 { v } else { (((v << (64 - bits)) as i64) >> (64 - bits)) as u64 }
}

fn field_mask(len: u32) -> u64 {
    if len >= 64 { !0 } else { (1u64 << len) - 1 }
}

fn eval_cond(w: u32, c: Cond, a: u64, b: u64) -> bool {
    if w == 32 { c.eval_u32(a as u32, b as u32) } else { c.eval_u64(a, b) }
}

fn cond_arg(op: &Op, i: usize) -> R<Cond> {
    Cond::from_u64(op.args[i])
        .ok_or_else(|| InterpError::BadOp(format!("{}: bad condition", op.opc.name())))
}

/// Run `f` once with the built-in helpers only, no linked `goto_tb` slots and a large step
/// limit.
pub fn run_tb(f: &Func, env: &mut [u8], mem: &mut dyn GuestMemory) -> Result<Exit, InterpError> {
    let helpers = HelperRegistry::new();
    let mut m = Machine::new(env, mem, &helpers);
    m.run(f)
}

impl<'a> Machine<'a> {
    /// A machine with no linked `goto_tb` slots and a step limit of 2^32.
    pub fn new(
        env: &'a mut [u8],
        mem: &'a mut dyn GuestMemory,
        helpers: &'a HelperRegistry,
    ) -> Machine<'a> {
        Machine {
            env,
            mem,
            helpers,
            linked: [false; 2],
            step_limit: 1 << 32,
            last_insn_start: None,
        }
    }

    fn env_range(&self, offset: u64, len: usize) -> R<std::ops::Range<usize>> {
        let bad = InterpError::EnvOutOfBounds { offset, len };
        let start = usize::try_from(offset).map_err(|_| bad.clone())?;
        let end = start.checked_add(len).ok_or_else(|| bad.clone())?;
        if end > self.env.len() {
            return Err(bad);
        }
        Ok(start..end)
    }

    /// Read `len` (at most 8) bytes, little endian, from the CPU state.
    fn env_read(&self, offset: u64, len: usize) -> R<u64> {
        let r = self.env_range(offset, len)?;
        let mut b = [0u8; 8];
        b[..len].copy_from_slice(&self.env[r]);
        Ok(u64::from_le_bytes(b))
    }

    fn env_write(&mut self, offset: u64, len: usize, v: u64) -> R<()> {
        let r = self.env_range(offset, len)?;
        self.env[r].copy_from_slice(&v.to_le_bytes()[..len]);
        Ok(())
    }

    fn env_read_bytes(&self, offset: u64, out: &mut [u8]) -> R<()> {
        let r = self.env_range(offset, out.len())?;
        out.copy_from_slice(&self.env[r]);
        Ok(())
    }

    fn env_write_bytes(&mut self, offset: u64, data: &[u8]) -> R<()> {
        let r = self.env_range(offset, data.len())?;
        self.env[r].copy_from_slice(data);
        Ok(())
    }

    /// The address of a global in the CPU state.
    fn global_addr(&self, f: &Func, st: &State, t: Temp) -> R<u64> {
        let td = f.temp(t);
        let base = match td.mem_base {
            Some(b) => self.get(f, st, b)?,
            None => return Err(InterpError::BadOp(format!("global {t:?} has no base"))),
        };
        Ok(base.wrapping_add(td.mem_offset as u64))
    }

    /// The value of a scalar temp. I32 values come back zero extended.
    fn get(&self, f: &Func, st: &State, t: Temp) -> R<u64> {
        let td = f.temp(t);
        let m = wmask(td.ty.bits());
        let v = match td.kind {
            TempKind::Const => td.val as u64,
            TempKind::Fixed => 0,
            TempKind::Global => {
                let a = self.global_addr(f, st, t)?;
                self.env_read(a, td.ty.size() as usize)?
            }
            TempKind::Tb | TempKind::Ebb => st.vals[t.index()][0],
        };
        Ok(v & m)
    }

    fn set(&mut self, f: &Func, st: &mut State, t: Temp, v: u64) -> R<()> {
        let td = f.temp(t);
        let v = v & wmask(td.ty.bits());
        match td.kind {
            TempKind::Const | TempKind::Fixed => {
                Err(InterpError::BadOp(format!("write to read-only temp {}", f.temp_name(t))))
            }
            TempKind::Global => {
                let a = self.global_addr(f, st, t)?;
                self.env_write(a, td.ty.size() as usize, v)
            }
            TempKind::Tb | TempKind::Ebb => {
                st.vals[t.index()] = [v, 0, 0, 0];
                Ok(())
            }
        }
    }

    fn getv(&self, f: &Func, st: &State, t: Temp) -> R<[u64; 4]> {
        let td = f.temp(t);
        match td.kind {
            TempKind::Const => Ok([td.val as u64; 4]),
            TempKind::Tb | TempKind::Ebb => Ok(st.vals[t.index()]),
            _ => Err(InterpError::BadOp(format!("{} is not a vector temp", f.temp_name(t)))),
        }
    }

    fn setv(&mut self, f: &Func, st: &mut State, t: Temp, v: [u64; 4]) -> R<()> {
        match f.temp(t).kind {
            TempKind::Tb | TempKind::Ebb => {
                st.vals[t.index()] = v;
                Ok(())
            }
            _ => Err(InterpError::BadOp(format!("{} is not a vector temp", f.temp_name(t)))),
        }
    }

    /// Execute `f` from its first op until it leaves the block.
    pub fn run(&mut self, f: &Func) -> Result<Exit, InterpError> {
        let ops: Vec<Op> = f.ops().map(|(_, o)| *o).collect();
        let mut labels: Vec<Option<usize>> = vec![None; f.nb_labels()];
        for (i, op) in ops.iter().enumerate() {
            if op.opc == Opcode::SetLabel {
                labels[op.arg_label(0).id() as usize] = Some(i);
            }
        }
        let mut st = State { vals: vec![[0; 4]; f.nb_temps()], carry: false };
        self.last_insn_start = None;
        let mut pc = 0usize;
        let mut steps = 0u64;
        while pc < ops.len() {
            steps += 1;
            if steps > self.step_limit {
                return Err(InterpError::StepLimit);
            }
            let op = &ops[pc];
            pc += 1;
            match self.step(f, &mut st, op)? {
                Flow::Next => {}
                Flow::Jump(l) => {
                    pc = labels
                        .get(l as usize)
                        .copied()
                        .flatten()
                        .ok_or(InterpError::BadLabel(l))?;
                }
                Flow::Exit(e) => return Ok(e),
            }
        }
        Err(InterpError::FellOffEnd)
    }

    fn step(&mut self, f: &Func, st: &mut State, op: &Op) -> R<Flow> {
        let w = op.ty.bits();
        let m = wmask(w);
        let a = |i: usize| op.arg_temp(i);
        macro_rules! g {
            ($i:expr) => {
                self.get(f, st, a($i))?
            };
        }
        macro_rules! s {
            ($i:expr, $v:expr) => {{
                let v = $v;
                self.set(f, st, a($i), v)?
            }};
        }
        let sh_mask = (w - 1) as u64;
        match op.opc {
            Opcode::Discard
            | Opcode::SetLabel
            | Opcode::Mb
            | Opcode::PluginCb
            | Opcode::PluginMemCb => {}
            Opcode::InsnStart => {
                let mut words = [0u64; INSN_START_WORDS];
                words.copy_from_slice(&op.args[..INSN_START_WORDS]);
                self.last_insn_start = Some(words);
                self.mem.insn_start(&words);
            }
            Opcode::Br => return Ok(Flow::Jump(op.arg_label(0).id())),
            Opcode::Brcond => {
                let c = cond_arg(op, 2)?;
                if eval_cond(w, c, g!(0), g!(1)) {
                    return Ok(Flow::Jump(op.arg_label(3).id()));
                }
            }
            Opcode::ExitTb => return Ok(Flow::Exit(Exit::ExitTb(op.args[0]))),
            Opcode::GotoTb => {
                let idx = op.args[0] as usize;
                if idx < 2 && self.linked[idx] {
                    return Ok(Flow::Exit(Exit::GotoTb(idx as u32)));
                }
            }
            Opcode::GotoPtr => return Ok(Flow::Exit(Exit::GotoPtr(g!(0)))),
            Opcode::Call => {
                if let Some(u) = self.call(f, st, op)? {
                    return Ok(Flow::Exit(Exit::Unwind(u)));
                }
            }
            Opcode::Mov => {
                let v = g!(1);
                s!(0, v)
            }
            Opcode::Add => s!(0, g!(1).wrapping_add(g!(2))),
            Opcode::Sub => s!(0, g!(1).wrapping_sub(g!(2))),
            Opcode::Mul => s!(0, g!(1).wrapping_mul(g!(2))),
            Opcode::Neg => s!(0, g!(1).wrapping_neg()),
            Opcode::And => s!(0, g!(1) & g!(2)),
            Opcode::Or => s!(0, g!(1) | g!(2)),
            Opcode::Xor => s!(0, g!(1) ^ g!(2)),
            Opcode::Andc => s!(0, g!(1) & !g!(2)),
            Opcode::Orc => s!(0, g!(1) | !g!(2)),
            Opcode::Eqv => s!(0, !(g!(1) ^ g!(2))),
            Opcode::Nand => s!(0, !(g!(1) & g!(2))),
            Opcode::Nor => s!(0, !(g!(1) | g!(2))),
            Opcode::Not => s!(0, !g!(1)),
            Opcode::Shl => s!(0, g!(1) << (g!(2) & sh_mask)),
            Opcode::Shr => s!(0, g!(1) >> (g!(2) & sh_mask)),
            Opcode::Sar => s!(0, ((sext(g!(1), w) as i64) >> (g!(2) & sh_mask)) as u64),
            Opcode::Rotl | Opcode::Rotr => {
                let x = g!(1);
                let mut n = (g!(2) & sh_mask) as u32;
                if op.opc == Opcode::Rotr {
                    n = (w - n) % w;
                }
                let v = if w == 32 { (x as u32).rotate_left(n) as u64 } else { x.rotate_left(n) };
                s!(0, v)
            }
            Opcode::Clz => {
                let x = g!(1);
                let v = if x == 0 { g!(2) } else { (x.leading_zeros() - (64 - w)) as u64 };
                s!(0, v)
            }
            Opcode::Ctz => {
                let x = g!(1);
                let v = if x == 0 { g!(2) } else { x.trailing_zeros() as u64 };
                s!(0, v)
            }
            Opcode::Ctpop => s!(0, g!(1).count_ones() as u64),
            Opcode::Bswap16 | Opcode::Bswap32 | Opcode::Bswap64 => {
                let x = g!(1);
                let flags = op.args[2] as u32;
                let v = match op.opc {
                    Opcode::Bswap16 => {
                        let v = (x as u16).swap_bytes() as u64;
                        if flags & bswap::OS != 0 { sext(v, 16) } else { v }
                    }
                    Opcode::Bswap32 => {
                        let v = (x as u32).swap_bytes() as u64;
                        if flags & bswap::OS != 0 { sext(v, 32) } else { v }
                    }
                    _ => x.swap_bytes(),
                };
                s!(0, v)
            }
            Opcode::Deposit => {
                let (ofs, len) = (op.args[3] as u32, op.args[4] as u32);
                let fm = field_mask(len) << ofs;
                s!(0, (g!(1) & !fm) | ((g!(2) << ofs) & fm))
            }
            Opcode::Extract => {
                let (ofs, len) = (op.args[2] as u32, op.args[3] as u32);
                s!(0, (g!(1) >> ofs) & field_mask(len))
            }
            Opcode::Sextract => {
                let (ofs, len) = (op.args[2] as u32, op.args[3] as u32);
                s!(0, sext(g!(1) >> ofs, len))
            }
            Opcode::Extract2 => {
                let ofs = op.args[3] as u32;
                let (lo, hi) = (g!(1), g!(2));
                let v = if w == 32 {
                    ((hi << 32 | lo) >> ofs) & m
                } else {
                    ((((hi as u128) << 64) | lo as u128) >> ofs) as u64
                };
                s!(0, v)
            }
            Opcode::Muluh | Opcode::Mulsh | Opcode::Mulu2 | Opcode::Muls2 => {
                let (i1, i2) =
                    if matches!(op.opc, Opcode::Mulu2 | Opcode::Muls2) { (2, 3) } else { (1, 2) };
                let (x, y) = (g!(i1), g!(i2));
                let p = if matches!(op.opc, Opcode::Mulsh | Opcode::Muls2) {
                    ((sext(x, w) as i64 as i128) * (sext(y, w) as i64 as i128)) as u128
                } else {
                    x as u128 * y as u128
                };
                let lo = p as u64 & m;
                let hi = (p >> w) as u64 & m;
                if matches!(op.opc, Opcode::Mulu2 | Opcode::Muls2) {
                    s!(0, lo);
                    s!(1, hi);
                } else {
                    s!(0, hi);
                }
            }
            Opcode::Divs | Opcode::Rems => {
                let x = sext(g!(1), w) as i64;
                let mut y = sext(g!(2), w) as i64;
                if y == 0 {
                    y = 1;
                }
                let v = if w == 32 {
                    let (x, y) = (x as i32, y as i32);
                    let v =
                        if op.opc == Opcode::Divs { x.wrapping_div(y) } else { x.wrapping_rem(y) };
                    v as i64
                } else if op.opc == Opcode::Divs {
                    x.wrapping_div(y)
                } else {
                    x.wrapping_rem(y)
                };
                s!(0, v as u64)
            }
            Opcode::Divu | Opcode::Remu => {
                let x = g!(1);
                let y = g!(2).max(1);
                s!(0, if op.opc == Opcode::Divu { x / y } else { x % y })
            }
            Opcode::Divs2 | Opcode::Divu2 => {
                let (lo, hi, d) = (g!(2), g!(3), g!(4));
                let (q, r) = if op.opc == Opcode::Divu2 {
                    let n = (hi as u128) << w | lo as u128;
                    let d = d.max(1) as u128;
                    ((n / d) as u64, (n % d) as u64)
                } else {
                    let n = ((sext(hi, w) as i64 as i128) << w) | lo as i128;
                    let mut d = sext(d, w) as i64 as i128;
                    if d == 0 {
                        d = 1;
                    }
                    (n.wrapping_div(d) as u64, n.wrapping_rem(d) as u64)
                };
                s!(0, q);
                s!(1, r);
            }
            Opcode::Setcond | Opcode::Negsetcond => {
                let c = cond_arg(op, 3)?;
                let t = eval_cond(w, c, g!(1), g!(2)) as u64;
                s!(0, if op.opc == Opcode::Setcond { t } else { t.wrapping_neg() })
            }
            Opcode::Movcond => {
                let c = cond_arg(op, 5)?;
                let v = if eval_cond(w, c, g!(1), g!(2)) { g!(3) } else { g!(4) };
                s!(0, v)
            }
            Opcode::Addco | Opcode::Addci | Opcode::Addcio | Opcode::Addc1o => {
                let (x, y) = (g!(1) as u128, g!(2) as u128);
                let cin = match op.opc {
                    Opcode::Addco => 0,
                    Opcode::Addc1o => 1,
                    _ => st.carry as u128,
                };
                let full = x + y + cin;
                if op.opc != Opcode::Addci {
                    st.carry = full >> w != 0;
                }
                s!(0, full as u64)
            }
            Opcode::Subbo | Opcode::Subbi | Opcode::Subbio | Opcode::Subb1o => {
                let (x, y) = (g!(1) as u128, g!(2) as u128);
                let bin = match op.opc {
                    Opcode::Subbo => 0,
                    Opcode::Subb1o => 1,
                    _ => st.carry as u128,
                };
                if op.opc != Opcode::Subbi {
                    st.carry = x < y + bin;
                }
                s!(0, x.wrapping_sub(y + bin) as u64)
            }
            Opcode::ExtI32I64 => s!(0, sext(g!(1), 32)),
            Opcode::ExtuI32I64 | Opcode::ExtrlI64I32 => s!(0, g!(1) & 0xffff_ffff),
            Opcode::ExtrhI64I32 => s!(0, g!(1) >> 32),
            Opcode::Ld8u
            | Opcode::Ld8s
            | Opcode::Ld16u
            | Opcode::Ld16s
            | Opcode::Ld32u
            | Opcode::Ld32s
            | Opcode::Ld => {
                let addr = g!(1).wrapping_add(op.args[2]);
                let (len, signed) = match op.opc {
                    Opcode::Ld8u => (1, false),
                    Opcode::Ld8s => (1, true),
                    Opcode::Ld16u => (2, false),
                    Opcode::Ld16s => (2, true),
                    Opcode::Ld32u => (4, false),
                    Opcode::Ld32s => (4, true),
                    _ => (op.ty.size() as usize, false),
                };
                let v = self.env_read(addr, len)?;
                s!(0, if signed { sext(v, 8 * len as u32) } else { v })
            }
            Opcode::St8 | Opcode::St16 | Opcode::St32 | Opcode::St => {
                let addr = g!(1).wrapping_add(op.args[2]);
                let len = match op.opc {
                    Opcode::St8 => 1,
                    Opcode::St16 => 2,
                    Opcode::St32 => 4,
                    _ => op.ty.size() as usize,
                };
                let v = g!(0);
                self.env_write(addr, len, v)?;
            }
            Opcode::QemuLd | Opcode::QemuSt | Opcode::QemuLd2 | Opcode::QemuSt2 => {
                let two = matches!(op.opc, Opcode::QemuLd2 | Opcode::QemuSt2);
                let ai = if two { 2 } else { 1 };
                let addr = g!(ai);
                let oi = MemOpIdx(op.args[ai + 1] as u32);
                match op.opc {
                    Opcode::QemuLd | Opcode::QemuLd2 => {
                        match guest_load_env(self.mem, self.env, addr, oi) {
                            Ok(v) => {
                                s!(0, v as u64);
                                if two {
                                    s!(1, (v >> 64) as u64);
                                }
                            }
                            Err(e) => return Ok(Flow::Exit(Exit::Unwind(Unwind::Mem(e)))),
                        }
                    }
                    _ => {
                        let v =
                            if two { g!(0) as u128 | (g!(1) as u128) << 64 } else { g!(0) as u128 };
                        if let Err(e) = guest_store_env(self.mem, self.env, addr, v, oi) {
                            return Ok(Flow::Exit(Exit::Unwind(Unwind::Mem(e))));
                        }
                    }
                }
            }
            _ if op.opc.flags() & ruvm_jit_core::types::opf::VECTOR != 0 => {
                self.step_vec(f, st, op)?;
            }
            _ => return Err(InterpError::BadOp(format!("unhandled op {}", op.opc.name()))),
        }
        Ok(Flow::Next)
    }

    /// Run a call op. Returns the unwind reason if the helper left the block.
    fn call(&mut self, f: &Func, st: &mut State, op: &Op) -> R<Option<Unwind>> {
        let info = f.helper_info(op.call_helper());
        let entry = self
            .helpers
            .get(&info.name)
            .ok_or_else(|| InterpError::UnknownHelper(info.name.clone()))?;
        if entry.ret != info.ret || entry.args != info.args {
            return Err(InterpError::HelperSignature(info.name.clone()));
        }
        let (no, ni) = (op.callo as usize, op.calli as usize);
        let mut args = Vec::with_capacity(ni);
        for i in no..no + ni {
            args.push(self.get(f, st, op.arg_temp(i))?);
        }
        let func = entry.f;
        let r = {
            let mut he = HelperEnv { env: &mut *self.env, mem: &mut *self.mem };
            func(&mut he, &args)
        };
        match r {
            Ok(v) => {
                if no >= 1 {
                    self.set(f, st, op.arg_temp(0), v as u64)?;
                }
                if no >= 2 {
                    self.set(f, st, op.arg_temp(1), (v >> 64) as u64)?;
                }
                Ok(None)
            }
            Err(u) => Ok(Some(u)),
        }
    }

    fn step_vec(&mut self, f: &Func, st: &mut State, op: &Op) -> R<()> {
        let a = |i: usize| op.arg_temp(i);
        let bytes = op.ty.size() as usize;
        let res = match op.opc {
            Opcode::LdVec => {
                let addr = self.get(f, st, a(1))?.wrapping_add(op.args[2]);
                let mut b = [0u8; 32];
                self.env_read_bytes(addr, &mut b[..bytes])?;
                vector::from_bytes(&b)
            }
            Opcode::StVec => {
                let addr = self.get(f, st, a(1))?.wrapping_add(op.args[2]);
                let v = self.getv(f, st, a(0))?;
                let b = vector::to_bytes(&v);
                return self.env_write_bytes(addr, &b[..bytes]);
            }
            Opcode::DupmVec => {
                let addr = self.get(f, st, a(1))?.wrapping_add(op.args[2]);
                let v = self.env_read(addr, 1 << op.vece)?;
                vector::dup(op.vece as u32, v)
            }
            Opcode::DupVec => {
                let v = self.get(f, st, a(1))?;
                vector::dup(op.vece as u32, v)
            }
            Opcode::ShlsVec | Opcode::ShrsVec | Opcode::SarsVec | Opcode::RotlsVec => {
                let x = self.getv(f, st, a(1))?;
                let n = self.get(f, st, a(2))?;
                vector::shift_scalar(op.opc, op.vece as u32, &x, n)
            }
            _ => {
                let def = op.opc.def();
                let mut ins = [[0u64; 4]; 4];
                for (i, v) in ins.iter_mut().enumerate().take(def.nb_iargs as usize) {
                    *v = self.getv(f, st, a(1 + i))?;
                }
                vector::exec(op, &ins).map_err(InterpError::BadOp)?
            }
        };
        let res = vector::clear_above(res, bytes);
        self.setv(f, st, a(0), res)
    }
}

enum Flow {
    Next,
    Jump(u32),
    Exit(Exit),
}

/// Round a value to the width of `ty` the way the interpreter stores it: I32 values are kept
/// zero extended.
pub fn normalize(ty: Type, v: u64) -> u64 {
    v & wmask(ty.bits())
}
