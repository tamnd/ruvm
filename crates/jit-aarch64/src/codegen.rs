// SPDX-License-Identifier: GPL-2.0-or-later

//! Instruction selection and register allocation: one finished [`Func`] in, A64 code out. This
//! is QEMU's `tcg/aarch64/tcg-target.c.inc` (constraints, `tcg_out_op`, `tcg_out_vec_op` and
//! the `tcg_out_*` hooks) plugged into the generic allocator of
//! [`ruvm_jit_core::regalloc`], which drives it the way `tcg_gen_code` does.
//!
//! Register use:
//!
//! - x19 is the address of the CPU state buffer (`TCG_AREG0`), x20 its length, x21 the run
//!   context, x22 the slot array; all four are fixed for the whole block;
//! - x0 to x15 and x23 to x28 hold temps, callee-saved ones first in the allocation order;
//! - x16, x17 and x30 are scratch (`TCG_REG_TMP0`, `TMP1` and `TMP2`); x18 is never touched,
//!   it is the platform register on macOS;
//! - v0 to v7 and v16 to v29 hold vector temps, v30 and v31 are vector scratch.
//!
//! Globals live in the CPU state at their offsets, TB and EBB temps in a slot array, and the
//! allocator moves them into registers and back as `op.life` says.
//!
//! Host pointers are offsets into the CPU state buffer, as in the interpreter: `env` is 0, and a
//! pointer global holds an offset. Every access through such a pointer is bounds checked against
//! x20 and leaves the block with [`ruvm_jit_interp::InterpError::EnvOutOfBounds`] when it would
//! fall outside the buffer. Accesses at constant offsets from `env` are checked once, on entry,
//! against the furthest one in the block.
//!
//! Whatever needs Rust (helper calls, `qemu_ld` and `qemu_st`, the 128 by 64 bit divisions) is
//! a call to one service routine with the index of a [`Request`]; operands go through the
//! argument words of the run context.
//!
//! Differences from QEMU:
//!
//! - `env` is the constant 0, not a register, and every access through a pointer is bounds
//!   checked as described above. Host loads and stores take a constant base (`ri`) so that
//!   accesses through `env` use the static check.
//! - Loads and stores through a pointer that is not `env` can fault, so the allocator syncs
//!   globals before them, as it does for ops with side effects.
//! - Helper calls, guest memory accesses and `divs2`/`divu2` go through the service routine,
//!   with every argument in memory, instead of the host calling convention and the softmmu
//!   fast path.
//! - `insn_start` stores the index of its words in the run context rather than recording them
//!   in a table next to the code; the runtime reports them to the guest memory before each
//!   service request and to the caller at the end.
//! - Ops QEMU's backend does not implement and instead expands (`nand`, `nor`, `rotl`, `ctpop`,
//!   I32 `mulsh` and `muluh`, `muls2` and `mulu2`, and vector `mul`, `smin` and the like at 64
//!   bit elements, `rotli`, the shifts by vector and by scalar, `cmpsel`) are expanded inline
//!   here, using the scratch registers.
//! - The vector immediate forms of `and`, `or`, `andc` and `orc` (`wO`, `wN`, `wV`) are not
//!   used; such constants are loaded into a register.
//! - The add and subtract with carry ops keep the carry in a word of the slot array, and only
//!   pass it in the flags between two adjacent ops of the same family.

use ruvm_jit_core::ir::{Func, HelperType, Op, OpId, Temp};
use ruvm_jit_core::opcode::Opcode;
use ruvm_jit_core::regalloc::{self, Letter, RegSet, Target};
use ruvm_jit_core::types::{
    Cond, INSN_START_WORDS, MemOpIdx, TempKind, Type, bswap, dup_const, opf,
};

use crate::asm::{self, Asm, AsmError, LR, Reg, TMP0, TMP1, VTMP0, VTMP1, XZR, cc, i};

/// Base of the CPU state buffer.
pub(crate) const ENV: Reg = asm::AREG0;
/// Length of the CPU state buffer.
const ENV_LEN: Reg = 20;
/// The run context.
const CTX: Reg = 21;
/// The slot array.
const SLOTS: Reg = 22;

const X0: Reg = 0;
const X1: Reg = 1;
const X2: Reg = 2;
const X3: Reg = 3;

/// Bytes per temp slot: room for a 256-bit vector.
pub(crate) const SLOT_BYTES: usize = 32;
/// Words of the run context used to pass operands to and from the service routine.
pub(crate) const NARGS: usize = 32;
/// Byte offset of the return value word in the run context, right after the argument words.
pub(crate) const RET_OFFSET: i64 = 8 * NARGS as i64;
/// Byte offset of the word holding one more than the index of the last `insn_start` request.
pub(crate) const INSN_OFFSET: i64 = RET_OFFSET + 8;

/// How generated code left, in x0 at the epilogue.
pub(crate) mod kind {
    /// `exit_tb`; the value is in the return word.
    pub(crate) const EXIT_TB: u64 = 0;
    /// A linked `goto_tb`; the slot is in the return word.
    pub(crate) const GOTO_TB: u64 = 1;
    /// `goto_ptr`; the pointer is in the return word.
    pub(crate) const GOTO_PTR: u64 = 2;
    /// The service routine recorded an unwind in the context.
    pub(crate) const UNWIND: u64 = 3;
    /// The service routine recorded an error in the context.
    pub(crate) const ERROR: u64 = 4;
    /// A CPU state access out of bounds: offset in the return word, length in argument 0.
    pub(crate) const BOUNDS: u64 = 5;
    /// The last op ran without leaving the block.
    pub(crate) const FELL_OFF: u64 = 6;
}

/// Work the generated code hands to Rust.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Request {
    /// Call a helper. Inputs are in the argument words, one per 64-bit slot; the result comes
    /// back in words 0 and 1.
    Call {
        /// The helper's name, looked up in the registry at run time.
        name: String,
        /// Its declared return type.
        ret: HelperType,
        /// Its declared argument types.
        args: Vec<HelperType>,
        /// Number of input words.
        nin: usize,
    },
    /// `qemu_ld`: the address is in word 0, the value comes back in words 0 and 1.
    Load(MemOpIdx),
    /// `qemu_st`: the value is in words 0 and 1, the address in word 2.
    Store(MemOpIdx),
    /// The words of an `insn_start`. Never called; the code stores its index in the context.
    InsnStart([u64; INSN_START_WORDS]),
    /// `divs2` or `divu2` at this width: low, high and divisor in words 0 to 2, quotient and
    /// remainder back in words 0 and 1.
    Div2 {
        /// Signed division.
        signed: bool,
        /// Operand width in bits.
        bits: u32,
    },
}

/// Why a block could not be compiled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GenCodeError {
    /// The block uses an op or a type this backend does not generate; the text names it. The
    /// caller can run the block with the interpreter instead.
    Unsupported(String),
    /// The IR is malformed; the text says how.
    BadOp(String),
    /// A branch went out of range, or the block does not fit in the code buffer.
    TooLarge,
}

impl std::fmt::Display for GenCodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenCodeError::Unsupported(s) => write!(f, "not supported by the aarch64 backend: {s}"),
            GenCodeError::BadOp(s) => write!(f, "bad op: {s}"),
            GenCodeError::TooLarge => f.write_str("the block is too large"),
        }
    }
}

impl std::error::Error for GenCodeError {}

impl From<AsmError> for GenCodeError {
    fn from(_: AsmError) -> GenCodeError {
        GenCodeError::TooLarge
    }
}

type R<T> = Result<T, GenCodeError>;

/// The output of [`generate`].
#[derive(Debug)]
pub(crate) struct Generated {
    pub(crate) bytes: Vec<u8>,
    pub(crate) requests: Vec<Request>,
    /// Number of 64-bit words of slot array the code uses.
    pub(crate) slot_words: usize,
    /// For each `goto_tb`: its slot, the byte offset of its patchable word, and the word that
    /// sends it to the exit stub.
    pub(crate) goto_tb: Vec<(u32, usize, u32)>,
}

/// The general registers the allocator may use: x0 to x15 and x23 to x28.
const GPRS: RegSet = RegSet(0xffff | 0x3f << 23);
/// The vector registers the allocator may use: v0 to v7 and v16 to v29.
const VECS: RegSet = RegSet(0xff << 32 | 0x3fff << 48);
/// `tcg_target_reg_alloc_order`: callee-saved registers first, so values survive calls.
const ALLOC_ORDER: [Reg; 44] = [
    23, 24, 25, 26, 27, 28, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, //
    32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
];
/// The registers a call to Rust may change: x0 to x17, v0 to v7 and v16 to v31.
const CALL_CLOBBER: RegSet = RegSet(0x3ffff | 0xff << 32 | 0xffff << 48);
/// Never allocated: scratch, the platform register, the fixed registers, fp, lr, sp, the
/// callee-saved vector registers and the vector scratch.
const RESERVED: RegSet = RegSet(0x7f << 16 | 0x7 << 29 | 0xff << 40 | 0x3 << 62);

/// Target constant classes, `TCG_CT_CONST_*`.
mod ctc {
    /// An add or subtract immediate, `TCG_CT_CONST_AIMM`.
    pub(super) const AIMM: u32 = 0x100;
    /// A logical immediate, `TCG_CT_CONST_LIMM`.
    pub(super) const LIMM: u32 = 0x200;
    /// A comparison immediate, `TCG_CT_CONST_CMP`.
    pub(super) const CMP: u32 = 0x400;
    /// Zero, `TCG_CT_CONST_ZERO`.
    pub(super) const ZERO: u32 = 0x800;
}

/// `TCG_TARGET_HAS_*` style decision: the flags an op gets on top of its definition.
fn extra_flags(f: &Func, op: &Op) -> u32 {
    match op.opc {
        Opcode::Divs2 | Opcode::Divu2 => opf::CALL_CLOBBER,
        Opcode::Ld8u
        | Opcode::Ld8s
        | Opcode::Ld16u
        | Opcode::Ld16s
        | Opcode::Ld32u
        | Opcode::Ld32s
        | Opcode::Ld
        | Opcode::LdVec
        | Opcode::DupmVec => may_fault(f, op, 1),
        Opcode::St8 | Opcode::St16 | Opcode::St32 | Opcode::St | Opcode::StVec => {
            may_fault(f, op, 1)
        }
        _ => 0,
    }
}

/// `SIDE_EFFECTS` unless the access at argument `bi` is to a constant offset that the check on
/// entry covers.
fn may_fault(f: &Func, op: &Op, bi: usize) -> u32 {
    let off = op.args[op.nb_oargs() + op.nb_iargs()] as i64;
    match static_offset(f, op.arg_temp(bi), off) {
        Some(_) => 0,
        None => opf::SIDE_EFFECTS,
    }
}

/// The CPU state offset of `base + off` when `base` is a constant and the result is small
/// enough to check on entry.
fn static_offset(f: &Func, base: Temp, off: i64) -> Option<i64> {
    let td = f.temp(base);
    if !matches!(td.kind, TempKind::Fixed | TempKind::Const) {
        return None;
    }
    let v = td.val.wrapping_add(off);
    if (0..1 << 31).contains(&v) { Some(v) } else { None }
}

/// Compile `f` for code that will live at `base`. `service` is the address of the service
/// routine.
pub(crate) fn generate(f: &Func, base: u64, service: u64) -> R<Generated> {
    check_types(f)?;
    let f = regalloc::prepare(f, &extra_flags);
    let mut g = Gen {
        a: Asm::new(base),
        labels: vec![None; f.nb_labels()],
        requests: Vec::new(),
        goto_tb: Vec::new(),
        service,
        exit: 0,
        bounds: 0,
        static_end: 0,
        static_access: (0, 0),
        static_always_fails: false,
        nb_temps: f.nb_temps(),
        err: None,
    };
    g.exit = g.a.new_label();
    g.bounds = g.a.new_label();
    let static_fail = g.a.new_label();

    // Prologue: a frame record and the callee-saved registers the block uses.
    g.a.ldstpair(i::STP, asm::FP, LR, asm::SP, -96, true, true);
    g.a.movr_sp(true, asm::FP, asm::SP);
    g.a.ldstpair(i::STP, ENV, ENV_LEN, asm::SP, 16, true, false);
    g.a.ldstpair(i::STP, CTX, SLOTS, asm::SP, 32, true, false);
    g.a.ldstpair(i::STP, 23, 24, asm::SP, 48, true, false);
    g.a.ldstpair(i::STP, 25, 26, asm::SP, 64, true, false);
    g.a.ldstpair(i::STP, 27, 28, asm::SP, 80, true, false);
    g.a.movr(true, ENV, X0);
    g.a.movr(true, ENV_LEN, X1);
    g.a.movr(true, CTX, X2);
    g.a.movr(true, SLOTS, X3);
    // The static bounds check; the two immediates are patched once the body is known.
    let check_at = g.a.pos();
    g.a.movw(i::MOVZ, true, TMP0, 0, 0);
    g.a.movw(i::MOVK, true, TMP0, 0, 16);
    g.a.rrr(i::SUBS, true, XZR, ENV_LEN, TMP0);
    g.a.bcond_label(cc::LO, static_fail);

    regalloc::reg_alloc(&f, &mut g)?;
    if let Some(e) = g.err.take() {
        return Err(e);
    }
    g.exit_with(kind::FELL_OFF, None);

    // Exit stubs for linked goto_tb slots.
    let mut goto_tb = Vec::new();
    for (slot, at) in std::mem::take(&mut g.goto_tb) {
        let stub = g.a.pos();
        g.a.movi(Type::I64, X1, slot as u64);
        g.a.movi(Type::I64, X0, kind::GOTO_TB);
        g.a.b_label(g.exit);
        let word = i::B | ((stub as i64 - at as i64) as u32 & 0x03ff_ffff);
        goto_tb.push((slot, at * 4, word));
    }

    // A failed static check reports the access that reaches furthest.
    g.a.bind(static_fail);
    g.a.movi(Type::I64, TMP0, g.static_access.0);
    g.a.movi(Type::I64, LR, g.static_access.1);
    g.a.b_label(g.bounds);

    // A failed bounds check: offset in x16, length in x30.
    g.a.bind(g.bounds);
    g.a.st(Type::I64, LR, CTX, 0);
    g.a.movr(true, X1, TMP0);
    g.a.movi(Type::I64, X0, kind::BOUNDS);

    // The epilogue: x0 is the kind, x1 the return word.
    g.a.bind(g.exit);
    g.a.st(Type::I64, X1, CTX, RET_OFFSET);
    g.a.ldstpair(i::LDP, 27, 28, asm::SP, 80, true, false);
    g.a.ldstpair(i::LDP, 25, 26, asm::SP, 64, true, false);
    g.a.ldstpair(i::LDP, 23, 24, asm::SP, 48, true, false);
    g.a.ldstpair(i::LDP, CTX, SLOTS, asm::SP, 32, true, false);
    g.a.ldstpair(i::LDP, ENV, ENV_LEN, asm::SP, 16, true, false);
    g.a.ldstpair(i::LDP, asm::FP, LR, asm::SP, 96, false, true);
    g.a.breg(i::RET, LR);

    // A branch to a label that is never set is malformed IR; the interpreter reports it when
    // the branch is taken, a compiler has to report it now.
    for (id, l) in g.labels.iter().enumerate() {
        if let Some(l) = *l {
            if !g.a.is_bound(l) {
                return Err(GenCodeError::BadOp(format!("label $L{id} is not set")));
            }
        }
    }
    let end = g.static_end;
    g.a.code[check_at] |= ((end & 0xffff) as u32) << 5;
    g.a.code[check_at + 1] |= (((end >> 16) & 0xffff) as u32) << 5;
    if g.static_always_fails {
        g.a.code[check_at + 3] = (g.a.code[check_at + 3] & !0xf) | cc::AL;
    }

    let slot_words = (f.nb_temps() + 1) * SLOT_BYTES / 8;
    let requests = g.requests;
    let out = g.a.finish()?;
    Ok(Generated { bytes: out.bytes, requests, slot_words, goto_tb })
}

/// Refuse temps of types this backend has no registers for, and calls with more arguments
/// than the run context holds.
fn check_types(f: &Func) -> R<()> {
    for (_, op) in f.ops() {
        let n = op.nb_oargs() + op.nb_iargs();
        for k in 0..n {
            let ty = f.temp(op.arg_temp(k)).ty;
            if !matches!(ty, Type::I32 | Type::I64 | Type::V64 | Type::V128) {
                return Err(GenCodeError::Unsupported(format!("{}: {ty:?} temps", op.opc.name())));
            }
        }
        if op.opc.def().flags & opf::VECTOR != 0 && !matches!(op.ty, Type::V64 | Type::V128) {
            return Err(GenCodeError::Unsupported(format!("{}: {:?}", op.opc.name(), op.ty)));
        }
        if op.opc == Opcode::Call && (op.calli as usize > NARGS || op.callo > 2) {
            return Err(bad(op, "too many call arguments"));
        }
    }
    Ok(())
}

fn bad(op: &Op, what: &str) -> GenCodeError {
    GenCodeError::BadOp(format!("{}: {what}", op.opc.name()))
}

fn cond_arg(op: &Op, i: usize) -> R<Cond> {
    Cond::from_u64(op.args[i]).ok_or_else(|| bad(op, "bad condition"))
}

/// The A64 condition that holds when `code` does not.
const fn invert(code: u32) -> u32 {
    code ^ 1
}

/// A constant operand of an `op.ty` op, sign extended from 32 bits for I32 as QEMU does.
fn norm(ty: Type, v: u64) -> i64 {
    if ty == Type::I32 { v as i32 as i64 } else { v as i64 }
}

/// Where a host memory access goes.
enum Addr {
    /// At this constant offset into the CPU state, covered by the check on entry.
    Static(i64),
    /// At the offset in x16, already bounds checked.
    Dyn,
}

struct Gen {
    a: Asm,
    labels: Vec<Option<usize>>,
    requests: Vec<Request>,
    goto_tb: Vec<(u32, usize)>,
    service: u64,
    exit: usize,
    bounds: usize,
    /// The furthest end of a constant offset CPU state access, and that access.
    static_end: u64,
    static_access: (u64, u64),
    /// A constant offset access is outside what the check on entry can describe.
    static_always_fails: bool,
    nb_temps: usize,
    /// An error from a hook that cannot return one.
    err: Option<GenCodeError>,
}

// Constraint sets, `tcg-target-con-set.h`.
const C_R: &[&str] = &["r"];
const C_R_R: &[&str] = &["r", "r"];
const C_R_RI: &[&str] = &["r", "ri"];
const C_RZ_RI: &[&str] = &["rz", "ri"];
const C_RZ_R: &[&str] = &["rz", "r"];
const C_RZ_RZ_R: &[&str] = &["rz", "rz", "r"];
const C_R_R_R: &[&str] = &["r", "r", "r"];
const C_R_R_RA: &[&str] = &["r", "r", "rA"];
const C_R_R_RL: &[&str] = &["r", "r", "rL"];
const C_R_R_RI: &[&str] = &["r", "r", "ri"];
const C_R_R_RC: &[&str] = &["r", "r", "rC"];
const C_R_RZ_RZ: &[&str] = &["r", "rz", "rz"];
const C_R_0_RZ: &[&str] = &["r", "0", "rz"];
const C_R_R_R_R: &[&str] = &["r", "r", "r", "r"];
const C_R5: &[&str] = &["r", "r", "r", "r", "r"];
const C_R_RC: &[&str] = &["r", "rC"];
const C_MOVCOND: &[&str] = &["r", "r", "rC", "rz", "rz"];
const C_NONE: &[&str] = &[];
const C_W_RI: &[&str] = &["w", "ri"];
const C_W_R: &[&str] = &["w", "r"];
const C_W_W: &[&str] = &["w", "w"];
const C_W_W_W: &[&str] = &["w", "w", "w"];
const C_W_W_R: &[&str] = &["w", "w", "r"];
const C_W_W_WZ: &[&str] = &["w", "w", "wZ"];
const C_W4: &[&str] = &["w", "w", "w", "w"];
const C_W5: &[&str] = &["w", "w", "w", "w", "w"];

impl Gen {
    fn label(&mut self, op: &Op, i: usize) -> R<usize> {
        let id = op.arg_label(i).id() as usize;
        let slot = self.labels.get_mut(id).ok_or_else(|| bad(op, "unknown label"))?;
        Ok(match slot {
            Some(l) => *l,
            None => {
                let l = self.a.new_label();
                *slot = Some(l);
                l
            }
        })
    }

    /// The offset of the carry flag word in the slot array, after every temp's slot.
    fn carry_offset(&self) -> i64 {
        (self.nb_temps * SLOT_BYTES) as i64
    }

    fn exit_with(&mut self, k: u64, value: Option<u64>) {
        if let Some(v) = value {
            self.a.movi(Type::I64, X1, v);
        }
        self.a.movi(Type::I64, X0, k);
        self.a.b_label(self.exit);
    }

    /// `rd = rn + v` in 64 bits. Uses x17 for a large `v`.
    fn add_const(&mut self, rd: Reg, rn: Reg, v: i64) {
        let u = v as u64;
        if asm::is_aimm(u) {
            self.a.addsub_imm(i::ADDI, true, rd, rn, u);
        } else if asm::is_aimm(u.wrapping_neg()) {
            self.a.addsub_imm(i::SUBI, true, rd, rn, u.wrapping_neg());
        } else {
            self.a.movi(Type::I64, TMP1, u);
            self.a.rrr(i::ADD, true, rd, rn, TMP1);
        }
    }

    /// Record an access at a constant CPU state offset for the check on entry.
    fn note_static(&mut self, off: i64, len: u64) {
        if !(0..1 << 31).contains(&off) {
            if !self.static_always_fails {
                self.static_always_fails = true;
                self.static_access = (off as u64, len);
            }
            return;
        }
        let end = off as u64 + len;
        if end > self.static_end && !self.static_always_fails {
            self.static_end = end;
            self.static_access = (off as u64, len);
        }
    }

    /// Check that `len` bytes at the offset in x16 are inside the CPU state.
    fn check_bounds(&mut self, len: u64) {
        self.a.movi(Type::I64, LR, len);
        self.a.rrr(i::ADDS, true, TMP1, TMP0, LR);
        self.a.bcond_label(cc::HS, self.bounds);
        self.a.rrr(i::SUBS, true, XZR, TMP1, ENV_LEN);
        self.a.bcond_label(cc::HI, self.bounds);
    }

    /// Where `len` bytes at `base + off` are. `base` is argument `bi` of `op`, a register or a
    /// constant.
    fn host_addr(
        &mut self,
        f: &Func,
        op: &Op,
        args: &[u64],
        const_args: &[bool],
        bi: usize,
        len: u64,
    ) -> Addr {
        let off = op.args[op.nb_oargs() + op.nb_iargs()] as i64;
        if const_args[bi] {
            if let Some(v) = static_offset(f, op.arg_temp(bi), off) {
                self.note_static(v, len);
                return Addr::Static(v);
            }
            self.a.movi(Type::I64, TMP0, (args[bi] as i64).wrapping_add(off) as u64);
        } else {
            self.add_const(TMP0, args[bi] as Reg, off);
        }
        self.check_bounds(len);
        Addr::Dyn
    }

    /// A host load or store of `rt` with `insn`, `1 << lg` bytes.
    fn host_access(&mut self, addr: Addr, insn: u32, rt: Reg, lg: u32) {
        match addr {
            Addr::Static(off) => self.a.ldst(insn, rt, ENV, off, lg),
            Addr::Dyn => self.a.ldst_reg(insn, rt, ENV, true, TMP0),
        }
    }

    /// Call the service routine for `req`. Leaves the block if it reports an unwind or error.
    fn service(&mut self, req: Request) {
        let idx = self.requests.len();
        self.requests.push(req);
        self.a.movr(true, X0, CTX);
        self.a.movi(Type::I64, X1, idx as u64);
        self.a.movi(Type::I64, TMP0, self.service);
        self.a.breg(i::BLR, TMP0);
        let ok = self.a.new_label();
        self.a.reloc_here(asm::Reloc::Condbr19, ok);
        self.a.cbz(i::CBZ, true, X0, 0);
        self.a.movi(Type::I64, X1, 0);
        self.a.b_label(self.exit);
        self.a.bind(ok);
    }

    /// Set the flags for `c` from `a` and `b`, `tgen_cmp` and `tgen_cmpi`.
    fn compare(&mut self, ext: bool, c: Cond, a: Reg, b: u64, b_const: bool, ty: Type) {
        if !b_const {
            let insn = if c.is_tst() { i::ANDS } else { i::SUBS };
            self.a.rrr(insn, ext, XZR, a, b as Reg);
        } else if c.is_tst() {
            self.a.logicali(i::ANDSI, ext, XZR, a, norm(ty, b) as u64);
        } else {
            let v = norm(ty, b);
            if asm::is_aimm(v as u64) {
                self.a.addsub_imm(i::SUBSI, ext, XZR, a, v as u64);
            } else {
                self.a.addsub_imm(i::ADDSI, ext, XZR, a, v.wrapping_neg() as u64);
            }
        }
    }

    /// `tgen_brcond` and `tgen_brcondi`.
    #[allow(clippy::too_many_arguments, reason = "the operands of brcond, already allocated")]
    fn brcond(&mut self, ty: Type, c: Cond, a: Reg, b: u64, b_const: bool, l: usize) {
        let ext = ty == Type::I64;
        let w = ty.bits();
        if b_const {
            let v = norm(ty, b);
            let mask = if ty == Type::I32 { v as u32 as u64 } else { v as u64 };
            match c {
                Cond::Eq | Cond::Ne if v == 0 => {
                    self.a.reloc_here(asm::Reloc::Condbr19, l);
                    self.a.cbz(if c == Cond::Eq { i::CBZ } else { i::CBNZ }, ext, a, 0);
                    return;
                }
                Cond::Lt | Cond::Ge if v == 0 => {
                    self.a.reloc_here(asm::Reloc::Tstbr14, l);
                    self.a.tbz(if c == Cond::Lt { i::TBNZ } else { i::TBZ }, a, w - 1, 0);
                    return;
                }
                Cond::TstEq | Cond::TstNe if mask == 0xffff_ffff => {
                    self.a.reloc_here(asm::Reloc::Condbr19, l);
                    self.a.cbz(if c == Cond::TstEq { i::CBZ } else { i::CBNZ }, false, a, 0);
                    return;
                }
                Cond::TstEq | Cond::TstNe if mask.is_power_of_two() => {
                    self.a.reloc_here(asm::Reloc::Tstbr14, l);
                    let insn = if c == Cond::TstEq { i::TBZ } else { i::TBNZ };
                    self.a.tbz(insn, a, mask.trailing_zeros(), 0);
                    return;
                }
                _ => {}
            }
        }
        self.compare(ext, c, a, b, b_const, ty);
        self.a.bcond_label(asm::cond_code(c), l);
    }

    /// The add and subtract with carry family. The carry lives in a word after the temp slots,
    /// except between two adjacent ops of the same family, where it stays in the flags.
    fn carry_op(&mut self, f: &Func, id: OpId, op: &Op, args: &[u64]) {
        let ext = op.ty == Type::I64;
        let carry = self.carry_offset();
        let sub = matches!(op.opc, Opcode::Subbo | Opcode::Subbi | Opcode::Subbio | Opcode::Subb1o);
        let family_out: &[Opcode] = if sub {
            &[Opcode::Subbo, Opcode::Subbio, Opcode::Subb1o]
        } else {
            &[Opcode::Addco, Opcode::Addcio, Opcode::Addc1o]
        };
        let family_in: &[Opcode] =
            if sub { &[Opcode::Subbi, Opcode::Subbio] } else { &[Opcode::Addci, Opcode::Addcio] };
        let def = op.opc.def();
        let carry_in = def.flags & opf::CARRY_IN != 0;
        let carry_out = def.flags & opf::CARRY_OUT != 0;
        let (d, a, b) = (args[0] as Reg, args[1] as Reg, args[2] as Reg);
        if carry_in {
            let fused = f.prev_op(id).is_some_and(|p| {
                let p = f.op(p);
                family_out.contains(&p.opc) && p.ty == op.ty
            });
            if !fused {
                self.a.ld(Type::I64, TMP0, SLOTS, carry);
                if sub {
                    // C is "no borrow": set when the borrow word is zero.
                    self.a.rrr(i::SUBS, true, XZR, XZR, TMP0);
                } else {
                    self.a.addsub_imm(i::SUBSI, true, XZR, TMP0, 1);
                }
            }
        }
        match op.opc {
            // 0 - 0 sets C, 0 + 0 clears it.
            Opcode::Addc1o => self.a.rrr(i::SUBS, true, XZR, XZR, XZR),
            Opcode::Subb1o => self.a.rrr(i::ADDS, true, XZR, XZR, XZR),
            _ => {}
        }
        let insn = match (sub, op.opc) {
            (false, Opcode::Addco) => i::ADDS,
            (false, Opcode::Addci) => i::ADC,
            (false, _) => i::ADCS,
            (true, Opcode::Subbo) => i::SUBS,
            (true, Opcode::Subbi) => i::SBC,
            (true, _) => i::SBCS,
        };
        self.a.rrr(insn, ext, d, a, b);
        if carry_out {
            let fused = f.next_op(id).is_some_and(|n| {
                let n = f.op(n);
                family_in.contains(&n.opc) && n.ty == op.ty
            });
            if !fused {
                // Carry out is C for an add and !C for a subtract.
                let when_clear = if sub { cc::HS } else { cc::LO };
                self.a.csel(i::CSINC, true, TMP0, XZR, XZR, when_clear);
                self.a.st(Type::I64, TMP0, SLOTS, carry);
            }
        }
    }

    /// Store `regs` to the argument words, from word 0.
    fn put_args(&mut self, regs: &[u64]) {
        for (k, &r) in regs.iter().enumerate() {
            self.a.st(Type::I64, r as Reg, CTX, 8 * k as i64);
        }
    }

    fn out_scalar(
        &mut self,
        f: &Func,
        id: OpId,
        op: &Op,
        args: &[u64],
        const_args: &[bool],
    ) -> R<()> {
        let ty = op.ty;
        let ext = ty == Type::I64;
        let w = ty.bits();
        let r = |k: usize| args[k] as Reg;
        let (d, a1) = (r(0), if args.len() > 1 { r(1) } else { 0 });
        match op.opc {
            Opcode::ExtI32I64 => self.a.bitfield(i::SBFM, true, d, a1, 1, 0, 31),
            Opcode::ExtuI32I64 | Opcode::ExtrlI64I32 => self.a.movr(false, d, a1),
            Opcode::ExtrhI64I32 => self.a.bitfield(i::UBFM, true, d, a1, 1, 32, 63),
            Opcode::Add | Opcode::Sub => {
                let add = op.opc == Opcode::Add;
                if const_args[2] {
                    let v = norm(ty, args[2]);
                    let (v, add) = if asm::is_aimm(v as u64) { (v, add) } else { (-v, !add) };
                    let insn = if add { i::ADDI } else { i::SUBI };
                    self.a.addsub_imm(insn, ext, d, a1, v as u64);
                } else {
                    self.a.rrr(if add { i::ADD } else { i::SUB }, ext, d, a1, r(2));
                }
            }
            Opcode::And | Opcode::Or | Opcode::Xor | Opcode::Andc | Opcode::Orc | Opcode::Eqv => {
                let (rinsn, iinsn, inv) = match op.opc {
                    Opcode::And => (i::AND, i::ANDI, false),
                    Opcode::Or => (i::ORR, i::ORRI, false),
                    Opcode::Xor => (i::EOR, i::EORI, false),
                    Opcode::Andc => (i::BIC, i::ANDI, true),
                    Opcode::Orc => (i::ORN, i::ORRI, true),
                    _ => (i::EON, i::EORI, true),
                };
                if const_args[2] {
                    let v = norm(ty, args[2]);
                    let v = if inv { !v } else { v };
                    self.a.logicali(iinsn, ext, d, a1, v as u64);
                } else {
                    self.a.rrr(rinsn, ext, d, a1, r(2));
                }
            }
            Opcode::Nand | Opcode::Nor => {
                let insn = if op.opc == Opcode::Nand { i::AND } else { i::ORR };
                self.a.rrr(insn, ext, d, a1, r(2));
                self.a.rrr(i::ORN, ext, d, XZR, d);
            }
            Opcode::Shl | Opcode::Shr | Opcode::Sar | Opcode::Rotl | Opcode::Rotr => {
                if const_args[2] {
                    let n = args[2] as u32 & (w - 1);
                    match op.opc {
                        Opcode::Shl => {
                            self.a.bitfield(i::UBFM, ext, d, a1, ext as u32, (w - n) % w, w - 1 - n)
                        }
                        Opcode::Shr => self.a.bitfield(i::UBFM, ext, d, a1, ext as u32, n, w - 1),
                        Opcode::Sar => self.a.bitfield(i::SBFM, ext, d, a1, ext as u32, n, w - 1),
                        Opcode::Rotr => self.a.extract(ext, d, a1, a1, n),
                        _ => self.a.extract(ext, d, a1, a1, (w - n) & (w - 1)),
                    }
                } else {
                    match op.opc {
                        Opcode::Shl => self.a.rrr(i::LSLV, ext, d, a1, r(2)),
                        Opcode::Shr => self.a.rrr(i::LSRV, ext, d, a1, r(2)),
                        Opcode::Sar => self.a.rrr(i::ASRV, ext, d, a1, r(2)),
                        Opcode::Rotr => self.a.rrr(i::RORV, ext, d, a1, r(2)),
                        _ => {
                            self.a.rrr(i::SUB, ext, TMP0, XZR, r(2));
                            self.a.rrr(i::RORV, ext, d, a1, TMP0);
                        }
                    }
                }
            }
            Opcode::Mul => self.a.rrrr(i::MADD, ext, d, a1, r(2), XZR),
            Opcode::Not => self.a.rrr(i::ORN, ext, d, XZR, a1),
            Opcode::Neg => self.a.rrr(i::SUB, ext, d, XZR, a1),
            Opcode::Clz | Opcode::Ctz => {
                if op.opc == Opcode::Ctz {
                    self.a.rr_sf(i::RBIT, ext, TMP0, a1);
                    self.a.rr_sf(i::CLZ, ext, TMP0, TMP0);
                } else {
                    self.a.rr_sf(i::CLZ, ext, TMP0, a1);
                }
                if const_args[2] && norm(ty, args[2]) as u64 & ty_mask(ty) == w as u64 {
                    self.a.movr(ext, d, TMP0);
                } else {
                    let b = if const_args[2] {
                        self.a.movi(ty, TMP1, args[2]);
                        TMP1
                    } else {
                        r(2)
                    };
                    self.a.addsub_imm(i::SUBSI, ext, XZR, a1, 0);
                    self.a.csel(i::CSEL, ext, d, TMP0, b, cc::NE);
                }
            }
            Opcode::Ctpop => {
                self.a.mov(Type::I64, VTMP0, a1);
                self.a.qrr_e(i::Q_CNT, false, 0, VTMP0, VTMP0);
                self.a.qrr_e(i::Q_ADDV, false, 0, VTMP0, VTMP0);
                self.a.simd_copy(i::UMOV, false, d, VTMP0, 1, 0);
            }
            Opcode::Bswap16 => {
                let os = op.args[2] as u32 & bswap::OS != 0;
                self.a.rr_sf(i::REV | 2 << 10, false, d, a1);
                let insn = if os { i::SBFM } else { i::UBFM };
                self.a.bitfield(insn, ext, d, d, ext as u32, 16, 31);
            }
            Opcode::Bswap32 => {
                let os = op.args[2] as u32 & bswap::OS != 0;
                self.a.rr_sf(i::REV | 2 << 10, false, d, a1);
                if os && ext {
                    self.a.bitfield(i::SBFM, true, d, d, 1, 0, 31);
                }
            }
            Opcode::Bswap64 => self.a.rr_sf(i::REV | 3 << 10, true, d, a1),
            Opcode::Deposit => {
                let (ofs, len) = (op.args[3] as u32, op.args[4] as u32);
                if len == 0 || ofs + len > w {
                    return Err(bad(op, "field out of range"));
                }
                // The output shares the register of the first input.
                self.a.bitfield(i::BFM, ext, d, r(2), ext as u32, (w - ofs) % w, len - 1);
            }
            Opcode::Extract | Opcode::Sextract => {
                let (ofs, len) = (op.args[2] as u32, op.args[3] as u32);
                if len == 0 || ofs + len > w {
                    return Err(bad(op, "field out of range"));
                }
                let insn = if op.opc == Opcode::Extract { i::UBFM } else { i::SBFM };
                self.a.bitfield(insn, ext, d, a1, ext as u32, ofs, ofs + len - 1);
            }
            Opcode::Extract2 => {
                let ofs = op.args[3] as u32;
                if ofs >= w {
                    return Err(bad(op, "shift out of range"));
                }
                self.a.extract(ext, d, r(2), a1, ofs);
            }
            Opcode::Muluh | Opcode::Mulsh => {
                let signed = op.opc == Opcode::Mulsh;
                if ext {
                    self.a.rrr(if signed { i::SMULH } else { i::UMULH }, true, d, a1, r(2));
                } else {
                    let insn = if signed { i::SMADDL } else { i::UMADDL };
                    self.a.rrrr(insn, true, d, a1, r(2), XZR);
                    self.a.bitfield(i::UBFM, true, d, d, 1, 32, 63);
                }
            }
            Opcode::Mulu2 | Opcode::Muls2 => {
                let signed = op.opc == Opcode::Muls2;
                let (lo, hi, a, b) = (r(0), r(1), r(2), r(3));
                if ext {
                    self.a.rrr(if signed { i::SMULH } else { i::UMULH }, true, TMP0, a, b);
                    self.a.rrrr(i::MADD, true, lo, a, b, XZR);
                    self.a.movr(true, hi, TMP0);
                } else {
                    let insn = if signed { i::SMADDL } else { i::UMADDL };
                    self.a.rrrr(insn, true, TMP0, a, b, XZR);
                    self.a.movr(false, lo, TMP0);
                    self.a.bitfield(i::UBFM, true, hi, TMP0, 1, 32, 63);
                }
            }
            Opcode::Divs | Opcode::Divu | Opcode::Rems | Opcode::Remu => {
                let signed = matches!(op.opc, Opcode::Divs | Opcode::Rems);
                let rem = matches!(op.opc, Opcode::Rems | Opcode::Remu);
                // A zero divisor divides by one.
                self.a.addsub_imm(i::SUBSI, ext, XZR, r(2), 0);
                self.a.csel(i::CSINC, ext, TMP1, r(2), XZR, cc::NE);
                let div = if signed { i::SDIV } else { i::UDIV };
                if rem {
                    self.a.rrr(div, ext, TMP0, a1, TMP1);
                    self.a.rrrr(i::MSUB, ext, d, TMP0, TMP1, a1);
                } else {
                    self.a.rrr(div, ext, d, a1, TMP1);
                }
            }
            Opcode::Divs2 | Opcode::Divu2 => {
                self.put_args(&args[2..5]);
                self.service(Request::Div2 { signed: op.opc == Opcode::Divs2, bits: w });
                self.a.ld(ty, r(0), CTX, 0);
                self.a.ld(ty, r(1), CTX, 8);
            }
            Opcode::Setcond | Opcode::Negsetcond => {
                let c = cond_arg(op, 3)?;
                let neg = op.opc == Opcode::Negsetcond;
                match c {
                    Cond::Never => self.a.movi(Type::I64, d, 0),
                    Cond::Always => self.a.movi(ty, d, if neg { u64::MAX } else { 1 }),
                    _ => {
                        self.compare(ext, c, a1, args[2], const_args[2], ty);
                        let insn = if neg { i::CSINV } else { i::CSINC };
                        self.a.csel(insn, ext, d, XZR, XZR, invert(asm::cond_code(c)));
                    }
                }
            }
            Opcode::Movcond => {
                let c = cond_arg(op, 5)?;
                match c {
                    Cond::Never => self.a.movr(ext, d, r(4)),
                    Cond::Always => self.a.movr(ext, d, r(3)),
                    _ => {
                        self.compare(ext, c, a1, args[2], const_args[2], ty);
                        self.a.csel(i::CSEL, ext, d, r(3), r(4), asm::cond_code(c));
                    }
                }
            }
            Opcode::Addco
            | Opcode::Addci
            | Opcode::Addcio
            | Opcode::Addc1o
            | Opcode::Subbo
            | Opcode::Subbi
            | Opcode::Subbio
            | Opcode::Subb1o => self.carry_op(f, id, op, args),
            Opcode::Ld8u
            | Opcode::Ld8s
            | Opcode::Ld16u
            | Opcode::Ld16s
            | Opcode::Ld32u
            | Opcode::Ld32s
            | Opcode::Ld => {
                let (insn, lg) = match op.opc {
                    Opcode::Ld8u => (i::LDRB, 0),
                    Opcode::Ld8s => (if ext { i::LDRSBX } else { i::LDRSBW }, 0),
                    Opcode::Ld16u => (i::LDRH, 1),
                    Opcode::Ld16s => (if ext { i::LDRSHX } else { i::LDRSHW }, 1),
                    Opcode::Ld32u => (i::LDRW, 2),
                    Opcode::Ld32s => (if ext { i::LDRSWX } else { i::LDRW }, 2),
                    _ => (if ext { i::LDRX } else { i::LDRW }, if ext { 3 } else { 2 }),
                };
                let addr = self.host_addr(f, op, args, const_args, 1, 1 << lg);
                self.host_access(addr, insn, d, lg);
            }
            Opcode::St8 | Opcode::St16 | Opcode::St32 | Opcode::St => {
                let (insn, lg) = match op.opc {
                    Opcode::St8 => (i::STRB, 0),
                    Opcode::St16 => (i::STRH, 1),
                    Opcode::St32 => (i::STRW, 2),
                    _ => (if ext { i::STRX } else { i::STRW }, if ext { 3 } else { 2 }),
                };
                let addr = self.host_access_addr(f, op, args, const_args, 1 << lg);
                self.host_access(addr, insn, r(0), lg);
            }
            Opcode::QemuLd | Opcode::QemuLd2 => {
                let two = op.opc == Opcode::QemuLd2;
                let ai = if two { 2 } else { 1 };
                self.put_args(&args[ai..ai + 1]);
                self.service(Request::Load(MemOpIdx(op.args[ai + 1] as u32)));
                if two {
                    self.a.ld(Type::I64, r(0), CTX, 0);
                    self.a.ld(Type::I64, r(1), CTX, 8);
                } else {
                    self.a.ld(ty, r(0), CTX, 0);
                }
            }
            Opcode::QemuSt | Opcode::QemuSt2 => {
                let two = op.opc == Opcode::QemuSt2;
                let ai = if two { 2 } else { 1 };
                if two {
                    self.put_args(&args[..2]);
                } else {
                    self.put_args(&[args[0], XZR as u64]);
                }
                self.a.st(Type::I64, r(ai), CTX, 16);
                self.service(Request::Store(MemOpIdx(op.args[ai + 1] as u32)));
            }
            Opcode::GotoPtr => {
                self.a.movr(true, X1, r(0));
                self.exit_with(kind::GOTO_PTR, None);
            }
            other => return Err(GenCodeError::Unsupported(other.name().to_string())),
        }
        Ok(())
    }

    /// [`Self::host_addr`] for a store, whose base is argument 1.
    fn host_access_addr(
        &mut self,
        f: &Func,
        op: &Op,
        args: &[u64],
        const_args: &[bool],
        len: u64,
    ) -> Addr {
        self.host_addr(f, op, args, const_args, 1, len)
    }

    /// A three operand vector op: the scalar form `e` for a 64-bit element of a V64, else the
    /// vector form `q`.
    #[allow(clippy::too_many_arguments, reason = "operands of tcg_out_vec_op")]
    fn v3(&mut self, ty: Type, vece: u32, q: u32, e: u32, d: Reg, n: Reg, m: Reg) {
        if ty == Type::V64 && vece == 3 {
            self.a.rrr_e(e, 3, d, n, m);
        } else {
            self.a.qrrr_e(q, ty == Type::V128, vece, d, n, m);
        }
    }

    /// A two operand vector op, as [`Self::v3`].
    fn v2(&mut self, ty: Type, vece: u32, q: u32, s: u32, d: Reg, n: Reg) {
        if ty == Type::V64 && vece == 3 {
            self.a.simd_rr(s, 3, d, n);
        } else {
            self.a.qrr_e(q, ty == Type::V128, vece, d, n);
        }
    }

    /// A bitwise op, for which the element size does not matter.
    fn vlogic(&mut self, ty: Type, insn: u32, d: Reg, n: Reg, m: Reg) {
        self.a.qrrr_e(insn, ty == Type::V128, 0, d, n, m);
    }

    fn vnot(&mut self, ty: Type, d: Reg, n: Reg) {
        self.a.qrr_e(i::Q_NOT, ty == Type::V128, 0, d, n);
    }

    /// Shift left by an immediate.
    fn vshl(&mut self, ty: Type, vece: u32, insn: (u32, u32), d: Reg, n: Reg, sh: u32) {
        if ty == Type::V64 && vece == 3 {
            self.a.q_shift(insn.0, d, n, 64 + sh);
        } else {
            self.a.simd_shift_imm(insn.1, ty == Type::V128, d, n, (8 << vece) + sh);
        }
    }

    /// Shift right by an immediate from 1 to the element width.
    fn vshr(&mut self, ty: Type, vece: u32, insn: (u32, u32), d: Reg, n: Reg, sh: u32) {
        if ty == Type::V64 && vece == 3 {
            self.a.q_shift(insn.0, d, n, 128 - sh);
        } else {
            self.a.simd_shift_imm(insn.1, ty == Type::V128, d, n, (16 << vece) - sh);
        }
    }

    /// `dup` of a general register to every element.
    fn vdup(&mut self, ty: Type, vece: u32, d: Reg, n: Reg) {
        // A 64-bit element always uses the 128-bit form: there is no one element DUP.
        let q = ty == Type::V128 || vece == 3;
        self.a.simd_copy(i::DUP, q, d, n, 1 << vece, 0);
    }

    /// `cmp_vec` into `d`. `b` is a register, or the zero constant when `zero` is set.
    #[allow(clippy::too_many_arguments, reason = "operands of tcg_out_vec_op")]
    fn vcmp(&mut self, ty: Type, vece: u32, c: Cond, d: Reg, a: Reg, b: Reg, zero: bool) {
        match c {
            Cond::Never => return self.a.dupi_vec(ty, 0, d, 0),
            Cond::Always => return self.a.dupi_vec(ty, 0, d, u64::MAX),
            _ => {}
        }
        if zero {
            let z = match c {
                Cond::Eq | Cond::Ne => Some((i::Q_CMEQ0, i::S_CMEQ0)),
                Cond::Lt => Some((i::Q_CMLT0, i::S_CMLT0)),
                Cond::Le => Some((i::Q_CMLE0, i::S_CMLE0)),
                Cond::Gt => Some((i::Q_CMGT0, i::S_CMGT0)),
                Cond::Ge => Some((i::Q_CMGE0, i::S_CMGE0)),
                _ => None,
            };
            if let Some((q, s)) = z {
                self.v2(ty, vece, q, s, d, a);
                if c == Cond::Ne {
                    self.vnot(ty, d, d);
                }
                return;
            }
            self.a.dupi_vec(ty, 0, VTMP1, 0);
            return self.vcmp(ty, vece, c, d, a, VTMP1, false);
        }
        let (q, e, swap, not) = match c {
            Cond::Eq => (i::Q_CMEQ, i::E_CMEQ, false, false),
            Cond::Ne => (i::Q_CMEQ, i::E_CMEQ, false, true),
            Cond::Gt => (i::Q_CMGT, i::E_CMGT, false, false),
            Cond::Ge => (i::Q_CMGE, i::E_CMGE, false, false),
            Cond::Lt => (i::Q_CMGT, i::E_CMGT, true, false),
            Cond::Le => (i::Q_CMGE, i::E_CMGE, true, false),
            Cond::Gtu => (i::Q_CMHI, i::E_CMHI, false, false),
            Cond::Geu => (i::Q_CMHS, i::E_CMHS, false, false),
            Cond::Ltu => (i::Q_CMHI, i::E_CMHI, true, false),
            Cond::Leu => (i::Q_CMHS, i::E_CMHS, true, false),
            Cond::TstNe => (i::Q_CMTST, i::E_CMTST, false, false),
            _ => (i::Q_CMTST, i::E_CMTST, false, true),
        };
        let (n, m) = if swap { (b, a) } else { (a, b) };
        self.v3(ty, vece, q, e, d, n, m);
        if not {
            self.vnot(ty, d, d);
        }
    }

    /// Put `dup_const(vece, v)` in `d`.
    fn vdupi(&mut self, ty: Type, vece: u32, d: Reg, v: u64) {
        self.a.dupi_vec(ty, vece, d, dup_const(vece, v));
    }

    fn out_vector(&mut self, f: &Func, op: &Op, args: &[u64], const_args: &[bool]) -> R<()> {
        let ty = op.ty;
        let vece = op.vece as u32;
        let bits = 8u32 << vece;
        let q = ty == Type::V128;
        let r = |k: usize| args[k] as Reg;
        let d = r(0);
        match op.opc {
            Opcode::LdVec | Opcode::StVec => {
                let (insn, lg) = match (op.opc, q) {
                    (Opcode::LdVec, false) => (i::LDRVD, 3),
                    (Opcode::LdVec, true) => (i::LDRVQ, 4),
                    (_, false) => (i::STRVD, 3),
                    _ => (i::STRVQ, 4),
                };
                let addr = self.host_addr(f, op, args, const_args, 1, 1 << lg);
                self.host_access(addr, insn, d, lg);
            }
            Opcode::DupmVec => {
                if let Addr::Static(off) = self.host_addr(f, op, args, const_args, 1, 1 << vece) {
                    self.a.movi(Type::I64, TMP0, off as u64);
                }
                self.a.rrr(i::ADD, true, TMP0, ENV, TMP0);
                self.a.loadrep(q, d, TMP0, vece);
            }
            Opcode::DupVec => self.vdup(ty, vece, d, r(1)),
            Opcode::AddVec => self.v3(ty, vece, i::Q_ADD, i::E_ADD, d, r(1), r(2)),
            Opcode::SubVec => self.v3(ty, vece, i::Q_SUB, i::E_SUB, d, r(1), r(2)),
            Opcode::SsaddVec => self.v3(ty, vece, i::Q_SQADD, i::E_SQADD, d, r(1), r(2)),
            Opcode::UsaddVec => self.v3(ty, vece, i::Q_UQADD, i::E_UQADD, d, r(1), r(2)),
            Opcode::SssubVec => self.v3(ty, vece, i::Q_SQSUB, i::E_SQSUB, d, r(1), r(2)),
            Opcode::UssubVec => self.v3(ty, vece, i::Q_UQSUB, i::E_UQSUB, d, r(1), r(2)),
            Opcode::NegVec => self.v2(ty, vece, i::Q_NEG, i::S_NEG, d, r(1)),
            Opcode::AbsVec => self.v2(ty, vece, i::Q_ABS, i::S_ABS, d, r(1)),
            Opcode::MulVec if vece < 3 => self.a.qrrr_e(i::Q_MUL, q, vece, d, r(1), r(2)),
            Opcode::MulVec => {
                for k in 0..if q { 2 } else { 1 } {
                    let lane = 8 | k << 4;
                    self.a.simd_copy(i::UMOV, true, TMP0, r(1), lane, 0);
                    self.a.simd_copy(i::UMOV, true, TMP1, r(2), lane, 0);
                    self.a.rrrr(i::MADD, true, TMP0, TMP0, TMP1, XZR);
                    self.a.simd_copy(i::INS, false, VTMP0, TMP0, lane, 0);
                }
                self.a.mov(ty, d, VTMP0);
            }
            Opcode::SminVec | Opcode::UminVec | Opcode::SmaxVec | Opcode::UmaxVec if vece < 3 => {
                let insn = match op.opc {
                    Opcode::SminVec => i::Q_SMIN,
                    Opcode::UminVec => i::Q_UMIN,
                    Opcode::SmaxVec => i::Q_SMAX,
                    _ => i::Q_UMAX,
                };
                self.a.qrrr_e(insn, q, vece, d, r(1), r(2));
            }
            Opcode::SminVec | Opcode::UminVec | Opcode::SmaxVec | Opcode::UmaxVec => {
                let signed = matches!(op.opc, Opcode::SminVec | Opcode::SmaxVec);
                let (qi, ei) = if signed { (i::Q_CMGT, i::E_CMGT) } else { (i::Q_CMHI, i::E_CMHI) };
                // The mask is a > b; min takes b there, max takes a.
                self.v3(ty, vece, qi, ei, VTMP0, r(1), r(2));
                let (t, e) = if matches!(op.opc, Opcode::SminVec | Opcode::UminVec) {
                    (r(2), r(1))
                } else {
                    (r(1), r(2))
                };
                self.vlogic(ty, i::Q_BSL, VTMP0, t, e);
                self.a.mov(ty, d, VTMP0);
            }
            Opcode::AndVec => self.vlogic(ty, i::Q_AND, d, r(1), r(2)),
            Opcode::OrVec => self.vlogic(ty, i::Q_ORR, d, r(1), r(2)),
            Opcode::XorVec => self.vlogic(ty, i::Q_EOR, d, r(1), r(2)),
            Opcode::AndcVec => self.vlogic(ty, i::Q_BIC, d, r(1), r(2)),
            Opcode::OrcVec => self.vlogic(ty, i::Q_ORN, d, r(1), r(2)),
            Opcode::NandVec | Opcode::NorVec | Opcode::EqvVec => {
                let insn = match op.opc {
                    Opcode::NandVec => i::Q_AND,
                    Opcode::NorVec => i::Q_ORR,
                    _ => i::Q_EOR,
                };
                self.vlogic(ty, insn, d, r(1), r(2));
                self.vnot(ty, d, d);
            }
            Opcode::NotVec => self.vnot(ty, d, r(1)),
            Opcode::ShliVec => {
                let n = op.args[2] as u32 & (bits - 1);
                self.vshl(ty, vece, (i::Q_SHL, i::V_SHL), d, r(1), n);
            }
            Opcode::ShriVec | Opcode::SariVec => {
                let n = op.args[2] as u32 & (bits - 1);
                if n == 0 {
                    self.a.mov(ty, d, r(1));
                } else if op.opc == Opcode::ShriVec {
                    self.vshr(ty, vece, (i::Q_USHR, i::V_USHR), d, r(1), n);
                } else {
                    self.vshr(ty, vece, (i::Q_SSHR, i::V_SSHR), d, r(1), n);
                }
            }
            Opcode::RotliVec => {
                let n = op.args[2] as u32 & (bits - 1);
                if n == 0 {
                    self.a.mov(ty, d, r(1));
                } else {
                    self.vshr(ty, vece, (i::Q_USHR, i::V_USHR), VTMP0, r(1), bits - n);
                    self.vshl(ty, vece, (i::Q_SLI, i::V_SLI), VTMP0, r(1), n);
                    self.a.mov(ty, d, VTMP0);
                }
            }
            Opcode::ShlsVec | Opcode::ShrsVec | Opcode::SarsVec | Opcode::RotlsVec => {
                self.a.logicali(i::ANDI, false, TMP0, r(2), (bits - 1) as u64);
                if op.opc != Opcode::ShlsVec && op.opc != Opcode::RotlsVec {
                    self.a.rrr(i::SUB, false, TMP0, XZR, TMP0);
                }
                self.vdup(ty, vece, VTMP0, TMP0);
                match op.opc {
                    Opcode::SarsVec => self.v3(ty, vece, i::Q_SSHL, i::E_SSHL, d, r(1), VTMP0),
                    Opcode::RotlsVec => {
                        self.a.addsub_imm(i::SUBI, false, TMP0, TMP0, bits as u64);
                        self.vdup(ty, vece, VTMP1, TMP0);
                        self.v3(ty, vece, i::Q_USHL, i::E_USHL, VTMP0, r(1), VTMP0);
                        self.v3(ty, vece, i::Q_USHL, i::E_USHL, VTMP1, r(1), VTMP1);
                        self.vlogic(ty, i::Q_ORR, d, VTMP0, VTMP1);
                    }
                    _ => self.v3(ty, vece, i::Q_USHL, i::E_USHL, d, r(1), VTMP0),
                }
            }
            Opcode::ShlvVec | Opcode::ShrvVec | Opcode::SarvVec => {
                self.vdupi(ty, vece, VTMP0, (bits - 1) as u64);
                self.vlogic(ty, i::Q_AND, VTMP0, r(2), VTMP0);
                if op.opc != Opcode::ShlvVec {
                    self.v2(ty, vece, i::Q_NEG, i::S_NEG, VTMP0, VTMP0);
                }
                if op.opc == Opcode::SarvVec {
                    self.v3(ty, vece, i::Q_SSHL, i::E_SSHL, d, r(1), VTMP0);
                } else {
                    self.v3(ty, vece, i::Q_USHL, i::E_USHL, d, r(1), VTMP0);
                }
            }
            Opcode::RotlvVec | Opcode::RotrvVec => {
                self.vdupi(ty, vece, VTMP0, (bits - 1) as u64);
                self.vlogic(ty, i::Q_AND, VTMP0, r(2), VTMP0);
                self.vdupi(ty, vece, VTMP1, bits as u64);
                if op.opc == Opcode::RotlvVec {
                    // Left by n, right by bits - n.
                    self.v3(ty, vece, i::Q_SUB, i::E_SUB, VTMP1, VTMP0, VTMP1);
                } else {
                    // Right by n, left by bits - n.
                    self.v3(ty, vece, i::Q_SUB, i::E_SUB, VTMP1, VTMP1, VTMP0);
                    self.v2(ty, vece, i::Q_NEG, i::S_NEG, VTMP0, VTMP0);
                }
                self.v3(ty, vece, i::Q_USHL, i::E_USHL, VTMP0, r(1), VTMP0);
                self.v3(ty, vece, i::Q_USHL, i::E_USHL, VTMP1, r(1), VTMP1);
                self.vlogic(ty, i::Q_ORR, d, VTMP0, VTMP1);
            }
            Opcode::CmpVec => {
                let c = cond_arg(op, 3)?;
                self.vcmp(ty, vece, c, d, r(1), r(2), const_args[2]);
            }
            Opcode::BitselVec => {
                let (a, b, c) = (r(1), r(2), r(3));
                if d == a {
                    self.vlogic(ty, i::Q_BSL, d, b, c);
                } else if d == c {
                    self.vlogic(ty, i::Q_BIT, d, b, a);
                } else if d == b {
                    self.vlogic(ty, i::Q_BIF, d, c, a);
                } else {
                    self.a.mov(ty, d, a);
                    self.vlogic(ty, i::Q_BSL, d, b, c);
                }
            }
            Opcode::CmpselVec => {
                let c = cond_arg(op, 5)?;
                self.vcmp(ty, vece, c, VTMP0, r(1), r(2), false);
                self.vlogic(ty, i::Q_BSL, VTMP0, r(3), r(4));
                self.a.mov(ty, d, VTMP0);
            }
            other => return Err(GenCodeError::Unsupported(other.name().to_string())),
        }
        Ok(())
    }
}

/// All ones in the width of `ty`.
fn ty_mask(ty: Type) -> u64 {
    if ty == Type::I32 { 0xffff_ffff } else { u64::MAX }
}

impl Target for Gen {
    type Error = GenCodeError;

    fn bad_ir(&self, msg: String) -> GenCodeError {
        GenCodeError::BadOp(msg)
    }

    fn alloc_order(&self) -> &[Reg] {
        &ALLOC_ORDER
    }

    fn available_regs(&self, ty: Type) -> RegSet {
        match ty {
            Type::I32 | Type::I64 => GPRS,
            Type::V64 | Type::V128 => VECS,
            _ => RegSet::EMPTY,
        }
    }

    fn reserved_regs(&self) -> RegSet {
        RESERVED
    }

    fn call_clobber_regs(&self) -> RegSet {
        CALL_CLOBBER
    }

    fn zero_reg(&self) -> Option<Reg> {
        Some(XZR)
    }

    fn op_constraints(&self, f: &Func, op: &Op) -> R<&'static [&'static str]> {
        Ok(match op.opc {
            Opcode::SetLabel
            | Opcode::Br
            | Opcode::Mb
            | Opcode::InsnStart
            | Opcode::ExitTb
            | Opcode::GotoTb
            | Opcode::PluginCb => C_NONE,
            Opcode::GotoPtr | Opcode::PluginMemCb => C_R,
            Opcode::Ld8u
            | Opcode::Ld8s
            | Opcode::Ld16u
            | Opcode::Ld16s
            | Opcode::Ld32u
            | Opcode::Ld32s
            | Opcode::Ld => C_R_RI,
            Opcode::St8 | Opcode::St16 | Opcode::St32 | Opcode::St => C_RZ_RI,
            Opcode::Add | Opcode::Sub => C_R_R_RA,
            Opcode::And | Opcode::Or | Opcode::Xor | Opcode::Andc | Opcode::Orc | Opcode::Eqv => {
                C_R_R_RL
            }
            Opcode::Nand
            | Opcode::Nor
            | Opcode::Mul
            | Opcode::Mulsh
            | Opcode::Muluh
            | Opcode::Divs
            | Opcode::Divu
            | Opcode::Rems
            | Opcode::Remu => C_R_R_R,
            Opcode::Muls2 | Opcode::Mulu2 => C_R_R_R_R,
            Opcode::Shl
            | Opcode::Shr
            | Opcode::Sar
            | Opcode::Rotl
            | Opcode::Rotr
            | Opcode::Clz
            | Opcode::Ctz => C_R_R_RI,
            Opcode::Ctpop
            | Opcode::Neg
            | Opcode::Not
            | Opcode::Bswap16
            | Opcode::Bswap32
            | Opcode::Bswap64
            | Opcode::Extract
            | Opcode::Sextract
            | Opcode::ExtI32I64
            | Opcode::ExtuI32I64
            | Opcode::ExtrlI64I32
            | Opcode::ExtrhI64I32 => C_R_R,
            Opcode::Setcond | Opcode::Negsetcond => C_R_R_RC,
            Opcode::Brcond => C_R_RC,
            Opcode::Movcond => C_MOVCOND,
            Opcode::Deposit => C_R_0_RZ,
            Opcode::Extract2
            | Opcode::Addco
            | Opcode::Addci
            | Opcode::Addcio
            | Opcode::Addc1o
            | Opcode::Subbo
            | Opcode::Subbi
            | Opcode::Subbio
            | Opcode::Subb1o => C_R_RZ_RZ,
            Opcode::Divs2 | Opcode::Divu2 => C_R5,
            Opcode::QemuLd => C_R_R,
            Opcode::QemuLd2 => C_R_R_R,
            Opcode::QemuSt => C_RZ_R,
            Opcode::QemuSt2 => C_RZ_RZ_R,
            Opcode::LdVec | Opcode::StVec | Opcode::DupmVec => C_W_RI,
            Opcode::DupVec => {
                if f.temp(op.arg_temp(1)).ty.is_vector() {
                    return Err(GenCodeError::Unsupported("dup_vec of a vector".into()));
                }
                C_W_R
            }
            Opcode::AddVec
            | Opcode::SubVec
            | Opcode::MulVec
            | Opcode::SsaddVec
            | Opcode::UsaddVec
            | Opcode::SssubVec
            | Opcode::UssubVec
            | Opcode::SminVec
            | Opcode::UminVec
            | Opcode::SmaxVec
            | Opcode::UmaxVec
            | Opcode::AndVec
            | Opcode::OrVec
            | Opcode::XorVec
            | Opcode::AndcVec
            | Opcode::OrcVec
            | Opcode::NandVec
            | Opcode::NorVec
            | Opcode::EqvVec
            | Opcode::ShlvVec
            | Opcode::ShrvVec
            | Opcode::SarvVec
            | Opcode::RotlvVec
            | Opcode::RotrvVec => C_W_W_W,
            Opcode::NegVec
            | Opcode::AbsVec
            | Opcode::NotVec
            | Opcode::ShliVec
            | Opcode::ShriVec
            | Opcode::SariVec
            | Opcode::RotliVec => C_W_W,
            Opcode::ShlsVec | Opcode::ShrsVec | Opcode::SarsVec | Opcode::RotlsVec => C_W_W_R,
            Opcode::CmpVec => C_W_W_WZ,
            Opcode::BitselVec => C_W4,
            Opcode::CmpselVec => C_W5,
            other => return Err(GenCodeError::Unsupported(other.name().to_string())),
        })
    }

    fn constraint_letter(&self, c: char) -> Option<Letter> {
        Some(match c {
            'r' => Letter::Regs(GPRS),
            'w' => Letter::Regs(VECS),
            'A' => Letter::Const(ctc::AIMM),
            'L' => Letter::Const(ctc::LIMM),
            'C' => Letter::Const(ctc::CMP),
            'Z' => Letter::Const(ctc::ZERO),
            _ => return None,
        })
    }

    fn const_match(&self, val: i64, ct: u32, ty: Type, cond: Cond, _vece: u32) -> bool {
        if ct & regalloc::ct::CONST != 0 {
            return true;
        }
        let val = if ty == Type::I32 { val as i32 as i64 } else { val };
        let mut ct = ct;
        if ct & ctc::CMP != 0 {
            ct |= if cond.is_tst() { ctc::LIMM } else { ctc::AIMM };
        }
        if ct & ctc::AIMM != 0
            && (asm::is_aimm(val as u64) || asm::is_aimm((val as u64).wrapping_neg()))
        {
            return true;
        }
        if ct & ctc::LIMM != 0 && asm::is_limm(val as u64) {
            return true;
        }
        ct & ctc::ZERO != 0 && val == 0
    }

    fn extra_op_flags(&self, f: &Func, op: &Op) -> u32 {
        extra_flags(f, op)
    }

    fn temp_home(&mut self, f: &Func, t: Temp) -> (Reg, i64) {
        let td = f.temp(t);
        match td.kind {
            TempKind::Tb | TempKind::Ebb => (SLOTS, (t.index() * SLOT_BYTES) as i64),
            TempKind::Global => {
                let fixed = td.mem_base.is_some_and(|b| f.temp(b).kind == TempKind::Fixed);
                if !fixed {
                    self.err.get_or_insert_with(|| {
                        GenCodeError::BadOp(format!("global {t:?} is not at a fixed offset"))
                    });
                    return (SLOTS, 0);
                }
                self.note_static(td.mem_offset, td.ty.size() as u64);
                (ENV, td.mem_offset)
            }
            TempKind::Fixed | TempKind::Const => {
                self.err.get_or_insert_with(|| {
                    GenCodeError::BadOp(format!("constant {t:?} has no storage"))
                });
                (SLOTS, 0)
            }
        }
    }

    fn out_mov(&mut self, ty: Type, dst: Reg, src: Reg) -> bool {
        self.a.mov(ty, dst, src);
        true
    }

    fn out_movi(&mut self, ty: Type, dst: Reg, val: i64) {
        self.a.movi(ty, dst, val as u64);
    }

    fn out_dupi_vec(&mut self, ty: Type, vece: u32, dst: Reg, val: u64) {
        self.a.dupi_vec(ty, vece, dst, val);
    }

    fn out_ld(&mut self, ty: Type, dst: Reg, base: Reg, off: i64) {
        self.a.ld(ty, dst, base, off);
    }

    fn out_st(&mut self, ty: Type, src: Reg, base: Reg, off: i64) {
        self.a.st(ty, src, base, off);
    }

    fn out_sti(&mut self, ty: Type, val: i64, base: Reg, off: i64) -> bool {
        if ty.is_int() && val == 0 {
            self.a.st(ty, XZR, base, off);
            return true;
        }
        false
    }

    fn out_op(&mut self, f: &Func, id: OpId, op: &Op, args: &[u64], const_args: &[bool]) -> R<()> {
        match op.opc {
            Opcode::PluginCb | Opcode::PluginMemCb => {}
            Opcode::SetLabel => {
                let l = self.label(op, 0)?;
                self.a.bind(l);
            }
            Opcode::Br => {
                let l = self.label(op, 0)?;
                self.a.b_label(l);
            }
            Opcode::Mb => self.a.emit(i::DMB_ISH | i::DMB_LD | i::DMB_ST),
            Opcode::InsnStart => {
                let mut words = [0u64; INSN_START_WORDS];
                words.copy_from_slice(&op.args[..INSN_START_WORDS]);
                self.requests.push(Request::InsnStart(words));
                self.a.movi(Type::I64, TMP0, self.requests.len() as u64);
                self.a.st(Type::I64, TMP0, CTX, INSN_OFFSET);
            }
            Opcode::ExitTb => self.exit_with(kind::EXIT_TB, Some(op.args[0])),
            Opcode::GotoTb => {
                let at = self.a.pos();
                self.a.emit(i::NOP);
                self.goto_tb.push((op.args[0] as u32, at));
            }
            Opcode::Brcond => {
                let c = cond_arg(op, 2)?;
                let l = self.label(op, 3)?;
                match c {
                    Cond::Never => {}
                    Cond::Always => self.a.b_label(l),
                    _ => self.brcond(op.ty, c, args[0] as Reg, args[1], const_args[1], l),
                }
            }
            _ if op.opc.def().flags & opf::VECTOR != 0 => {
                self.out_vector(f, op, args, const_args)?
            }
            _ => self.out_scalar(f, id, op, args, const_args)?,
        }
        Ok(())
    }

    fn call_arg_home(&self, idx: usize) -> (Reg, i64) {
        (CTX, 8 * idx as i64)
    }

    fn out_call(&mut self, f: &Func, op: &Op) -> R<()> {
        let info = f.helper_info(op.call_helper()).clone();
        let ni = op.calli as usize;
        self.service(Request::Call { name: info.name, ret: info.ret, args: info.args, nin: ni });
        Ok(())
    }
}
