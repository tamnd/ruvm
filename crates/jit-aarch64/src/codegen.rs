// SPDX-License-Identifier: GPL-2.0-or-later

//! Instruction selection: one finished [`Func`] in, A64 code out. This is the `tcg_out_op` half
//! of QEMU's `tcg/aarch64/tcg-target.c.inc`, driven by the op list the way `tcg_gen_code` drives
//! it, but without a register allocator.
//!
//! Every value lives in memory between ops: globals in the CPU state at their offsets, TB and EBB
//! temps in a slot array, constants in the instruction stream. Each op loads its inputs into
//! scratch registers, computes, and stores its outputs. That is QEMU's code with every temp
//! spilled after every op; the register allocator that removes the loads and stores is tier 2's
//! job (spec/08). It keeps every op's semantics identical to `ruvm-jit-interp`, which is what the
//! tests compare against.
//!
//! Register use, all fixed for the whole block:
//!
//! - x19 is the address of the CPU state buffer (`TCG_AREG0`), x20 its length, x21 the run
//!   context, x22 the slot array;
//! - x0 to x5 hold op operands, x9 to x12 addresses and bounds checks, x16 and x17 are the
//!   assembler's own scratch registers, v31 the vector scratch;
//! - x18 is never touched; it is the platform register on macOS.
//!
//! Host pointers are offsets into the CPU state buffer, as in the interpreter: `env` is 0, and a
//! pointer global holds an offset. Every access through such a pointer is bounds checked against
//! x20 and leaves the block with [`InterpError::EnvOutOfBounds`] when it would fall outside the
//! buffer. Accesses at constant offsets from `env` are checked once, on entry, against the
//! furthest one in the block.
//!
//! Whatever needs Rust (helper calls, `qemu_ld` and `qemu_st`, `insn_start`, the 128 by 64 bit
//! divisions) is a call to one service routine with the index of a [`Request`]; operands go
//! through the argument words of the run context.

use ruvm_jit_core::ir::{Func, HelperType, Op, Temp};
use ruvm_jit_core::opcode::Opcode;
use ruvm_jit_core::types::{Cond, INSN_START_WORDS, MemOpIdx, TempKind, Type, bswap};

use crate::asm::{self, Asm, AsmError, Reg, XZR, cc, i};

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
const X4: Reg = 4;
const X9: Reg = 9;
/// The address being computed for a checked access.
const XADDR: Reg = 10;
/// The length of a checked access.
const XLEN: Reg = 11;
const XEND: Reg = 12;

/// Bytes per temp slot: room for a 256-bit vector, though only scalars are generated today.
pub(crate) const SLOT_BYTES: usize = 32;
/// Words of the run context used to pass operands to and from the service routine.
pub(crate) const NARGS: usize = 32;
/// Byte offset of the return value word in the run context, right after the argument words.
pub(crate) const RET_OFFSET: i64 = 8 * NARGS as i64;

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
    /// `insn_start` with these words.
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
    /// The block uses an op this backend does not generate; the text names it. The caller can
    /// run the block with the interpreter instead.
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

/// Where a scalar lives.
enum Loc {
    /// At a constant offset into the CPU state.
    Env(i64),
    /// At the offset in [`XADDR`], already bounds checked.
    EnvDyn,
    /// In the slot array at this byte offset.
    Slot(i64),
}

struct Gen<'f> {
    f: &'f Func,
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
}

/// Compile `f` for code that will live at `base`. `service` is the address of the service
/// routine.
pub(crate) fn generate(f: &Func, base: u64, service: u64) -> R<Generated> {
    let mut g = Gen {
        f,
        a: Asm::new(base),
        labels: vec![None; f.nb_labels()],
        requests: Vec::new(),
        goto_tb: Vec::new(),
        service,
        exit: 0,
        bounds: 0,
        static_end: 0,
        static_access: (0, 0),
    };
    g.exit = g.a.new_label();
    g.bounds = g.a.new_label();
    let static_fail = g.a.new_label();

    // Prologue: a frame record and the four callee-saved registers the block keeps.
    g.a.ldstpair(i::STP, asm::FP, asm::LR, asm::SP, -48, true, true);
    g.a.movr_sp(true, asm::FP, asm::SP);
    g.a.ldstpair(i::STP, ENV, ENV_LEN, asm::SP, 16, true, false);
    g.a.ldstpair(i::STP, CTX, SLOTS, asm::SP, 32, true, false);
    g.a.movr(true, ENV, X0);
    g.a.movr(true, ENV_LEN, X1);
    g.a.movr(true, CTX, X2);
    g.a.movr(true, SLOTS, X3);
    // The static bounds check; the two immediates are patched once the body is known.
    let check_at = g.a.pos();
    g.a.movw(i::MOVZ, true, X9, 0, 0);
    g.a.movw(i::MOVK, true, X9, 0, 16);
    g.a.rrr(i::SUBS, true, XZR, ENV_LEN, X9);
    g.a.bcond_label(cc::LO, static_fail);

    for (_, op) in f.ops() {
        g.op(op)?;
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
    g.a.movi(Type::I64, XADDR, g.static_access.0);
    g.a.movi(Type::I64, XLEN, g.static_access.1);
    g.a.b_label(g.bounds);

    // A failed bounds check: offset in XADDR, length in XLEN.
    g.a.bind(g.bounds);
    g.a.st(Type::I64, XLEN, CTX, 0);
    g.a.movr(true, X1, XADDR);
    g.a.movi(Type::I64, X0, kind::BOUNDS);

    // The epilogue: x0 is the kind, x1 the return word.
    g.a.bind(g.exit);
    g.a.st(Type::I64, X1, CTX, RET_OFFSET);
    g.a.ldstpair(i::LDP, CTX, SLOTS, asm::SP, 32, true, false);
    g.a.ldstpair(i::LDP, ENV, ENV_LEN, asm::SP, 16, true, false);
    g.a.ldstpair(i::LDP, asm::FP, asm::LR, asm::SP, 48, false, true);
    g.a.breg(i::RET, asm::LR);

    // A branch to a label that is never set is malformed IR; the interpreter reports it when
    // the branch is taken, a compiler has to report it now.
    for (id, l) in g.labels.iter().enumerate() {
        if let Some(l) = *l
            && !g.a.is_bound(l)
        {
            return Err(GenCodeError::BadOp(format!("label $L{id} is not set")));
        }
    }
    let end = g.static_end;
    g.a.code[check_at] |= ((end & 0xffff) as u32) << 5;
    g.a.code[check_at + 1] |= (((end >> 16) & 0xffff) as u32) << 5;

    let slot_words = (f.nb_temps() + 1) * SLOT_BYTES / 8;
    let requests = g.requests;
    let out = g.a.finish()?;
    Ok(Generated { bytes: out.bytes, requests, slot_words, goto_tb })
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

impl Gen<'_> {
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
        (self.f.nb_temps() * SLOT_BYTES) as i64
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
            self.a.movi(Type::I64, asm::TMP1, u);
            self.a.rrr(i::ADD, true, rd, rn, asm::TMP1);
        }
    }

    /// Check that `len` bytes at the offset in [`XADDR`] are inside the CPU state.
    fn check_bounds(&mut self, len: u64) {
        self.a.movi(Type::I64, XLEN, len);
        self.a.rrr(i::ADDS, true, XEND, XADDR, XLEN);
        self.a.bcond_label(cc::HS, self.bounds);
        self.a.rrr(i::SUBS, true, XZR, XEND, ENV_LEN);
        self.a.bcond_label(cc::HI, self.bounds);
    }

    /// Where `len` bytes at `base + off` are, `base` being a pointer temp.
    ///
    /// An access at a small constant offset from `env` is left to the check on entry. Anything
    /// else is computed and checked here, so a wild offset fails at run time, as it does in
    /// the interpreter, rather than failing the compile.
    fn env_loc(&mut self, base: Temp, off: i64, len: u64) -> R<Loc> {
        if self.f.temp(base).kind == TempKind::Fixed && (0..1 << 31).contains(&off) {
            let end = off as u64 + len;
            if end > self.static_end {
                self.static_end = end;
                self.static_access = (off as u64, len);
            }
            return Ok(Loc::Env(off));
        }
        self.load(XADDR, base)?;
        self.add_const(XADDR, XADDR, off);
        self.check_bounds(len);
        Ok(Loc::EnvDyn)
    }

    /// Where the scalar temp `t` is kept.
    fn loc(&mut self, t: Temp) -> R<Loc> {
        let td = self.f.temp(t);
        match td.kind {
            TempKind::Tb | TempKind::Ebb => Ok(Loc::Slot((t.index() * SLOT_BYTES) as i64)),
            TempKind::Global => {
                let base = td
                    .mem_base
                    .ok_or_else(|| GenCodeError::BadOp(format!("global {t:?} has no base")))?;
                let (off, len) = (td.mem_offset, td.ty.size() as u64);
                self.env_loc(base, off, len)
            }
            _ => Err(GenCodeError::BadOp(format!("{t:?} has no storage"))),
        }
    }

    fn check_scalar(&self, t: Temp) -> R<Type> {
        let ty = self.f.temp(t).ty;
        if ty.is_int() { Ok(ty) } else { Err(GenCodeError::Unsupported(format!("{ty:?} temps"))) }
    }

    /// Load the scalar temp `t` into `r`. I32 values come back zero extended.
    fn load(&mut self, r: Reg, t: Temp) -> R<()> {
        let ty = self.check_scalar(t)?;
        let td = self.f.temp(t);
        match td.kind {
            TempKind::Const => self.a.movi(ty, r, td.val as u64),
            TempKind::Fixed => self.a.movi(Type::I64, r, 0),
            _ => {
                let insn = if ty == Type::I32 { i::LDRW } else { i::LDRX };
                self.access(insn, r, t, ty)?;
            }
        }
        Ok(())
    }

    /// Store `r` to the scalar temp `t`, truncated to its type.
    fn store(&mut self, t: Temp, r: Reg) -> R<()> {
        let ty = self.check_scalar(t)?;
        let insn = if ty == Type::I32 { i::STRW } else { i::STRX };
        self.access(insn, r, t, ty)
    }

    fn access(&mut self, insn: u32, r: Reg, t: Temp, ty: Type) -> R<()> {
        let lg = if ty == Type::I32 { 2 } else { 3 };
        match self.loc(t)? {
            Loc::Env(off) => self.a.ldst(insn, r, ENV, off, lg),
            Loc::EnvDyn => self.a.ldst_reg(insn, r, ENV, true, XADDR),
            Loc::Slot(off) => self.a.ldst(insn, r, SLOTS, off, lg),
        }
        Ok(())
    }

    /// Set the flags for `c` from `ra` and `rb`.
    fn compare(&mut self, ext: bool, c: Cond, ra: Reg, rb: Reg) {
        let insn = if c.is_tst() { i::ANDS } else { i::SUBS };
        self.a.rrr(insn, ext, XZR, ra, rb);
    }

    /// Call the service routine for `req`. Leaves the block if it reports an unwind or error.
    fn service(&mut self, req: Request) {
        let idx = self.requests.len();
        self.requests.push(req);
        self.a.movr(true, X0, CTX);
        self.a.movi(Type::I64, X1, idx as u64);
        self.a.movi(Type::I64, X9, self.service);
        self.a.breg(i::BLR, X9);
        let ok = self.a.new_label();
        self.a.reloc_here(asm::Reloc::Condbr19, ok);
        self.a.cbz(i::CBZ, true, X0, 0);
        self.a.movi(Type::I64, X1, 0);
        self.a.b_label(self.exit);
        self.a.bind(ok);
    }

    fn put_arg(&mut self, n: usize, t: Temp) -> R<()> {
        self.load(X0, t)?;
        self.a.st(Type::I64, X0, CTX, 8 * n as i64);
        Ok(())
    }

    fn get_arg(&mut self, t: Temp, n: usize) -> R<()> {
        self.a.ld(Type::I64, X0, CTX, 8 * n as i64);
        self.store(t, X0)
    }

    fn op(&mut self, op: &Op) -> R<()> {
        let ext = op.ty == Type::I64;
        let w = op.ty.bits();
        let t = |i: usize| op.arg_temp(i);
        match op.opc {
            Opcode::Discard | Opcode::PluginCb | Opcode::PluginMemCb => {}
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
                self.service(Request::InsnStart(words));
            }
            Opcode::ExitTb => self.exit_with(kind::EXIT_TB, Some(op.args[0])),
            Opcode::GotoTb => {
                let at = self.a.pos();
                self.a.emit(i::NOP);
                self.goto_tb.push((op.args[0] as u32, at));
            }
            Opcode::GotoPtr => {
                self.load(X1, t(0))?;
                self.exit_with(kind::GOTO_PTR, None);
            }
            Opcode::Call => self.call(op)?,
            Opcode::Mov | Opcode::ExtuI32I64 | Opcode::ExtrlI64I32 => {
                self.load(X0, t(1))?;
                self.store(t(0), X0)?;
            }
            Opcode::ExtI32I64 => {
                self.unary(op, |a, _| a.bitfield(i::SBFM, true, X0, X1, 1, 0, 31))?
            }
            Opcode::ExtrhI64I32 => {
                self.unary(op, |a, _| a.bitfield(i::UBFM, true, X0, X1, 1, 32, 63))?
            }
            Opcode::Add => self.binary(op, i::ADD)?,
            Opcode::Sub => self.binary(op, i::SUB)?,
            Opcode::And => self.binary(op, i::AND)?,
            Opcode::Or => self.binary(op, i::ORR)?,
            Opcode::Xor => self.binary(op, i::EOR)?,
            Opcode::Andc => self.binary(op, i::BIC)?,
            Opcode::Orc => self.binary(op, i::ORN)?,
            Opcode::Eqv => self.binary(op, i::EON)?,
            Opcode::Shl => self.binary(op, i::LSLV)?,
            Opcode::Shr => self.binary(op, i::LSRV)?,
            Opcode::Sar => self.binary(op, i::ASRV)?,
            Opcode::Rotr => self.binary(op, i::RORV)?,
            Opcode::Nand | Opcode::Nor => {
                let insn = if op.opc == Opcode::Nand { i::AND } else { i::ORR };
                self.binary_with(op, |a, ext| {
                    a.rrr(insn, ext, X0, X1, X2);
                    a.rrr(i::ORN, ext, X0, XZR, X0);
                })?
            }
            Opcode::Rotl => self.binary_with(op, |a, ext| {
                a.rrr(i::SUB, ext, X2, XZR, X2);
                a.rrr(i::RORV, ext, X0, X1, X2);
            })?,
            Opcode::Mul => self.binary_with(op, |a, ext| a.rrrr(i::MADD, ext, X0, X1, X2, XZR))?,
            Opcode::Not => self.unary(op, |a, ext| a.rrr(i::ORN, ext, X0, XZR, X1))?,
            Opcode::Neg => self.unary(op, |a, ext| a.rrr(i::SUB, ext, X0, XZR, X1))?,
            Opcode::Clz | Opcode::Ctz => {
                let ctz = op.opc == Opcode::Ctz;
                self.binary_with(op, |a, ext| {
                    if ctz {
                        a.rr_sf(i::RBIT, ext, X0, X1);
                        a.rr_sf(i::CLZ, ext, X0, X0);
                    } else {
                        a.rr_sf(i::CLZ, ext, X0, X1);
                    }
                    a.addsub_imm(i::SUBSI, ext, XZR, X1, 0);
                    a.csel(i::CSEL, ext, X0, X2, X0, cc::EQ);
                })?
            }
            Opcode::Ctpop => self.unary(op, |a, _| {
                a.mov(Type::I64, asm::VTMP0, X1);
                a.qrr_e(i::Q_CNT, false, 0, asm::VTMP0, asm::VTMP0);
                a.qrr_e(i::Q_ADDV, false, 0, asm::VTMP0, asm::VTMP0);
                a.simd_copy(i::UMOV, false, X0, asm::VTMP0, 1, 0);
            })?,
            Opcode::Bswap16 | Opcode::Bswap32 | Opcode::Bswap64 => {
                let os = op.args[2] as u32 & bswap::OS != 0;
                match op.opc {
                    Opcode::Bswap16 => self.unary(op, |a, ext| {
                        a.rr_sf(i::REV | 2 << 10, false, X0, X1);
                        let insn = if os { i::SBFM } else { i::UBFM };
                        a.bitfield(insn, ext, X0, X0, ext as u32, 16, 31);
                    })?,
                    Opcode::Bswap32 => self.unary(op, |a, ext| {
                        a.rr_sf(i::REV | 2 << 10, false, X0, X1);
                        if os && ext {
                            a.bitfield(i::SBFM, true, X0, X0, 1, 0, 31);
                        }
                    })?,
                    _ => self.unary(op, |a, _| a.rr_sf(i::REV | 3 << 10, true, X0, X1))?,
                }
            }
            Opcode::Deposit => {
                let (ofs, len) = (op.args[3] as u32, op.args[4] as u32);
                if len == 0 || ofs + len > w {
                    return Err(bad(op, "field out of range"));
                }
                self.load(X0, t(1))?;
                self.load(X1, t(2))?;
                self.a.bitfield(i::BFM, ext, X0, X1, ext as u32, (w - ofs) % w, len - 1);
                self.store(t(0), X0)?;
            }
            Opcode::Extract | Opcode::Sextract => {
                let (ofs, len) = (op.args[2] as u32, op.args[3] as u32);
                if len == 0 || ofs + len > w {
                    return Err(bad(op, "field out of range"));
                }
                let insn = if op.opc == Opcode::Extract { i::UBFM } else { i::SBFM };
                self.unary(op, |a, ext| {
                    a.bitfield(insn, ext, X0, X1, ext as u32, ofs, ofs + len - 1)
                })?
            }
            Opcode::Extract2 => {
                let ofs = op.args[3] as u32;
                if ofs >= w {
                    return Err(bad(op, "shift out of range"));
                }
                self.binary_with(op, |a, ext| a.extract(ext, X0, X2, X1, ofs))?
            }
            Opcode::Muluh | Opcode::Mulsh => {
                let signed = op.opc == Opcode::Mulsh;
                self.binary_with(op, |a, ext| {
                    if ext {
                        a.rrr(if signed { i::SMULH } else { i::UMULH }, true, X0, X1, X2);
                    } else {
                        a.rrrr(if signed { i::SMADDL } else { i::UMADDL }, true, X0, X1, X2, XZR);
                        a.bitfield(i::UBFM, true, X0, X0, 1, 32, 63);
                    }
                })?
            }
            Opcode::Mulu2 | Opcode::Muls2 => {
                let signed = op.opc == Opcode::Muls2;
                self.load(X1, t(2))?;
                self.load(X2, t(3))?;
                if ext {
                    self.a.rrrr(i::MADD, true, X0, X1, X2, XZR);
                    self.a.rrr(if signed { i::SMULH } else { i::UMULH }, true, X3, X1, X2);
                } else {
                    let insn = if signed { i::SMADDL } else { i::UMADDL };
                    self.a.rrrr(insn, true, X0, X1, X2, XZR);
                    self.a.bitfield(i::UBFM, true, X3, X0, 1, 32, 63);
                }
                self.store(t(0), X0)?;
                self.store(t(1), X3)?;
            }
            Opcode::Divs | Opcode::Divu | Opcode::Rems | Opcode::Remu => {
                let signed = matches!(op.opc, Opcode::Divs | Opcode::Rems);
                let rem = matches!(op.opc, Opcode::Rems | Opcode::Remu);
                self.binary_with(op, |a, ext| {
                    // A zero divisor divides by one.
                    a.addsub_imm(i::SUBSI, ext, XZR, X2, 0);
                    a.csel(i::CSINC, ext, X2, X2, XZR, cc::NE);
                    let div = if signed { i::SDIV } else { i::UDIV };
                    if rem {
                        a.rrr(div, ext, X3, X1, X2);
                        a.rrrr(i::MSUB, ext, X0, X3, X2, X1);
                    } else {
                        a.rrr(div, ext, X0, X1, X2);
                    }
                })?
            }
            Opcode::Divs2 | Opcode::Divu2 => {
                for k in 0..3 {
                    self.put_arg(k, t(2 + k))?;
                }
                self.service(Request::Div2 { signed: op.opc == Opcode::Divs2, bits: w });
                self.get_arg(t(0), 0)?;
                self.get_arg(t(1), 1)?;
            }
            Opcode::Setcond | Opcode::Negsetcond => {
                let c = cond_arg(op, 3)?;
                let neg = op.opc == Opcode::Negsetcond;
                match c {
                    Cond::Never => self.a.movi(Type::I64, X0, 0),
                    Cond::Always => self.a.movi(op.ty, X0, if neg { u64::MAX } else { 1 }),
                    _ => {
                        self.load(X1, t(1))?;
                        self.load(X2, t(2))?;
                        self.compare(ext, c, X1, X2);
                        let insn = if neg { i::CSINV } else { i::CSINC };
                        self.a.csel(insn, ext, X0, XZR, XZR, invert(asm::cond_code(c)));
                    }
                }
                self.store(t(0), X0)?;
            }
            Opcode::Movcond => {
                let c = cond_arg(op, 5)?;
                match c {
                    Cond::Never => self.load(X0, t(4))?,
                    Cond::Always => self.load(X0, t(3))?,
                    _ => {
                        self.load(X1, t(1))?;
                        self.load(X2, t(2))?;
                        self.load(X3, t(3))?;
                        self.load(X4, t(4))?;
                        self.compare(ext, c, X1, X2);
                        self.a.csel(i::CSEL, ext, X0, X3, X4, asm::cond_code(c));
                    }
                }
                self.store(t(0), X0)?;
            }
            Opcode::Brcond => {
                let c = cond_arg(op, 2)?;
                let l = self.label(op, 3)?;
                match c {
                    Cond::Never => {}
                    Cond::Always => self.a.b_label(l),
                    _ => {
                        self.load(X1, t(0))?;
                        self.load(X2, t(1))?;
                        self.compare(ext, c, X1, X2);
                        self.a.bcond_label(asm::cond_code(c), l);
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
            | Opcode::Subb1o => self.carry_op(op)?,
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
                self.host_access(insn, lg, X0, t(1), op.args[2] as i64)?;
                self.store(t(0), X0)?;
            }
            Opcode::St8 | Opcode::St16 | Opcode::St32 | Opcode::St => {
                let (insn, lg) = match op.opc {
                    Opcode::St8 => (i::STRB, 0),
                    Opcode::St16 => (i::STRH, 1),
                    Opcode::St32 => (i::STRW, 2),
                    _ => (if ext { i::STRX } else { i::STRW }, if ext { 3 } else { 2 }),
                };
                // The value goes in a register the address computation leaves alone.
                self.load(X4, t(0))?;
                self.host_access(insn, lg, X4, t(1), op.args[2] as i64)?;
            }
            Opcode::QemuLd | Opcode::QemuLd2 => {
                let two = op.opc == Opcode::QemuLd2;
                let ai = if two { 2 } else { 1 };
                self.put_arg(0, t(ai))?;
                self.service(Request::Load(MemOpIdx(op.args[ai + 1] as u32)));
                self.get_arg(t(0), 0)?;
                if two {
                    self.get_arg(t(1), 1)?;
                }
            }
            Opcode::QemuSt | Opcode::QemuSt2 => {
                let two = op.opc == Opcode::QemuSt2;
                let ai = if two { 2 } else { 1 };
                self.put_arg(0, t(0))?;
                if two {
                    self.put_arg(1, t(1))?;
                } else {
                    self.a.st(Type::I64, XZR, CTX, 8);
                }
                self.put_arg(2, t(ai))?;
                self.service(Request::Store(MemOpIdx(op.args[ai + 1] as u32)));
            }
            other => return Err(GenCodeError::Unsupported(other.name().to_string())),
        }
        Ok(())
    }

    /// An op with one input in x1 and one output from x0.
    fn unary(&mut self, op: &Op, emit: impl FnOnce(&mut Asm, bool)) -> R<()> {
        self.load(X1, op.arg_temp(1))?;
        emit(&mut self.a, op.ty == Type::I64);
        self.store(op.arg_temp(0), X0)
    }

    /// An op with inputs in x1 and x2 and one output from x0.
    fn binary_with(&mut self, op: &Op, emit: impl FnOnce(&mut Asm, bool)) -> R<()> {
        self.load(X1, op.arg_temp(1))?;
        self.load(X2, op.arg_temp(2))?;
        emit(&mut self.a, op.ty == Type::I64);
        self.store(op.arg_temp(0), X0)
    }

    fn binary(&mut self, op: &Op, insn: u32) -> R<()> {
        self.binary_with(op, |a, ext| a.rrr(insn, ext, X0, X1, X2))
    }

    /// A load or store of `rt` at `base + off` in the CPU state.
    fn host_access(&mut self, insn: u32, lg: u32, rt: Reg, base: Temp, off: i64) -> R<()> {
        match self.env_loc(base, off, 1 << lg)? {
            Loc::Env(off) => self.a.ldst(insn, rt, ENV, off, lg),
            _ => self.a.ldst_reg(insn, rt, ENV, true, XADDR),
        }
        Ok(())
    }

    /// The add and subtract with carry family. The carry lives in a word after the temp slots.
    fn carry_op(&mut self, op: &Op) -> R<()> {
        let ext = op.ty == Type::I64;
        let carry = self.carry_offset();
        let sub = matches!(op.opc, Opcode::Subbo | Opcode::Subbi | Opcode::Subbio | Opcode::Subb1o);
        self.load(X1, op.arg_temp(1))?;
        self.load(X2, op.arg_temp(2))?;
        let (carry_in, carry_out) = match op.opc {
            Opcode::Addco | Opcode::Subbo => (false, true),
            Opcode::Addc1o | Opcode::Subb1o => (false, true),
            Opcode::Addci | Opcode::Subbi => (true, false),
            _ => (true, true),
        };
        if carry_in {
            self.a.ld(Type::I64, X3, SLOTS, carry);
            if sub {
                // C is "no borrow": set when the borrow word is zero.
                self.a.rrr(i::SUBS, true, XZR, XZR, X3);
            } else {
                self.a.addsub_imm(i::SUBSI, true, XZR, X3, 1);
            }
        }
        let insn = match (sub, op.opc) {
            (false, Opcode::Addco) => i::ADDS,
            (false, Opcode::Addci) => i::ADC,
            (false, _) => i::ADCS,
            (true, Opcode::Subbo) => i::SUBS,
            (true, Opcode::Subbi) => i::SBC,
            (true, _) => i::SBCS,
        };
        match op.opc {
            // 0 - 0 sets C, 0 + 0 clears it.
            Opcode::Addc1o => self.a.rrr(i::SUBS, true, XZR, XZR, XZR),
            Opcode::Subb1o => self.a.rrr(i::ADDS, true, XZR, XZR, XZR),
            _ => {}
        }
        self.a.rrr(insn, ext, X0, X1, X2);
        if carry_out {
            // Carry out is C for an add and !C for a subtract.
            let when_clear = if sub { cc::HS } else { cc::LO };
            self.a.csel(i::CSINC, true, X3, XZR, XZR, when_clear);
            self.a.st(Type::I64, X3, SLOTS, carry);
        }
        self.store(op.arg_temp(0), X0)
    }

    fn call(&mut self, op: &Op) -> R<()> {
        let info = self.f.helper_info(op.call_helper()).clone();
        let (no, ni) = (op.callo as usize, op.calli as usize);
        if ni > NARGS || no > 2 {
            return Err(bad(op, "too many call arguments"));
        }
        for k in 0..ni {
            self.put_arg(k, op.arg_temp(no + k))?;
        }
        self.service(Request::Call { name: info.name, ret: info.ret, args: info.args, nin: ni });
        for k in 0..no {
            self.get_arg(op.arg_temp(k), k)?;
        }
        Ok(())
    }
}
