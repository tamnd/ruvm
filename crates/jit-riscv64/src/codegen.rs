// SPDX-License-Identifier: GPL-2.0-or-later

//! Instruction selection and register allocation: one finished [`Func`] in, RV64 code out. This
//! is QEMU's `tcg/riscv64/tcg-target.c.inc` (constraints, `tcg_out_op` and the `tcg_out_*`
//! hooks) plugged into the generic allocator of [`ruvm_jit_core::regalloc`], which drives it
//! the way `tcg_gen_code` does.
//!
//! Register use:
//!
//! - s0 is the address of the CPU state buffer (`TCG_AREG0`), s1 its length, s2 the run
//!   context and s3 the slot array; all four are fixed for the whole block;
//! - s4 to s11, t0 to t2 and a0 to a7 hold temps, callee-saved ones first in the allocation
//!   order;
//! - t6, t5 and t4 are scratch (`TCG_REG_TMP0`, `TMP1` and `TMP2`), and t3 is a fourth scratch
//!   for the expansions of this port, and the carry between two adjacent carry ops.
//!
//! I32 values are kept sign extended to 64 bits in registers, as QEMU does on this host: every
//! I32 op leaves its result that way, mostly through the `*w` instructions, so that comparisons
//! and branches can work on the whole register.
//!
//! Globals live in the CPU state at their offsets, TB and EBB temps in a slot array, and the
//! allocator moves them into registers and back as `op.life` says.
//!
//! Host pointers are offsets into the CPU state buffer, as in the interpreter: `env` is 0, and a
//! pointer global holds an offset. Every access through such a pointer is bounds checked against
//! s1 and leaves the block with [`ruvm_jit_interp::InterpError::EnvOutOfBounds`] when it would
//! fall outside the buffer. Accesses at constant offsets from `env` are checked once, on entry,
//! against the furthest one in the block.
//!
//! Whatever needs Rust (helper calls, `qemu_ld` and `qemu_st`, the 128 by 64 bit divisions) is
//! a call to one service routine with the index of a [`Request`]; operands go through the
//! argument words of the run context.
//!
//! Zba, Zbb, Zbs and Zicond are used as [`HostFeatures`] allows, each with the fallback QEMU
//! has, or an inline expansion where QEMU leaves the op to the generic expanders.
//!
//! Differences from QEMU:
//!
//! - `env` is the constant 0, not a register, and every access through a pointer is bounds
//!   checked as described above. Host loads and stores take a constant base (`ri`) so that
//!   accesses through `env` use the static check.
//! - Loads and stores through a pointer that is not `env` can fault, so the allocator syncs
//!   globals before them, as it does for ops with side effects.
//! - Helper calls, guest memory accesses and `divs2`/`divu2` go through the service routine,
//!   with every argument in memory, instead of the host calling convention. The exception is a
//!   `TCG_CALL_NO_SE` helper with a [`ruvm_jit_interp::NativeHelperFn`] in
//!   [`GenOptions::helpers`]: that is a direct call, its arguments loaded from memory into a0
//!   to a3 and its result stored back. I32 arguments are passed zero extended, as the
//!   interpreter holds them.
//! - With [`GenOptions::tlb_page_bits`], `qemu_ld` and `qemu_st` of up to 64 bits look up the
//!   softmmu TLB inline as QEMU's `prepare_host_addr` does, and use the service routine on a
//!   miss. The descriptor is found through the run context rather than at a fixed offset from
//!   `env`, the alignment check is a separate test rather than part of the comparison, and
//!   byte swapped accesses and 128-bit accesses that must be atomic as a whole always take the
//!   slow path. A 128-bit access whose halves need only be atomic each
//!   (`MO_ATOM_IFALIGN_PAIR`) or not at all is two `ld` or `sd` on a hit.
//! - `insn_start` emits no code. Each service request carries the index of the `insn_start`
//!   of its instruction, fixed when the block is compiled, and each exit stores it in the run
//!   context, instead of QEMU's table of host code offsets next to the code.
//! - Ops QEMU leaves to the generic expanders without Zbb (`clz`, `ctz`, `ctpop`, `bswap*`,
//!   `rotl` and `rotr`, `andc`, `orc` and `eqv`), and `nand`, `nor`, `extract2`, `deposit`,
//!   `muls2`, `mulu2` and the I32 `mulsh` and `muluh`, are expanded inline here, using the
//!   scratch registers.
//! - `div` and `rem` divide by one when the divisor is zero, as the interpreter does; RISC-V
//!   does not trap either way, but its results differ.
//! - The add and subtract with carry ops keep the carry in a word of the slot array, and only
//!   pass it in t3 between two adjacent ops of the same family.
//! - `goto_tb` is a `nop` until the block is linked. Linking patches it to a `jal` straight to
//!   the next block, as in QEMU, when that block is in a region this one keeps mapped and
//!   within the 1 MiB reach of `jal`, and to an exit stub that leaves with
//!   [`ruvm_jit_interp::Exit::GotoTb`] otherwise. QEMU uses an indirect jump through a table
//!   when the target is out of reach.
//! - Every block has its own prologue and epilogue, with the same frame, and chained jumps
//!   enter a block after its prologue. QEMU shares one prologue for the whole buffer.
//! - `goto_ptr` jumps to the address `lookup_tb_ptr` returned only if the runtime vouched for
//!   it in the run context, and leaves with [`ruvm_jit_interp::Exit::GotoPtr`] otherwise.
//!   QEMU jumps to whatever the helper returned.
//! - A call to `lookup_tb_ptr_ic` first looks the guest program counter up in an inline cache
//!   of block headers and jumps straight to the block on a hit; see the runtime. Not in QEMU.
//! - A 32-bit load of the `icount_decr` word at the offset the runtime gives reads the shared
//!   atomic through a pointer in the run context, so that exit requests from other threads
//!   are seen without leaving generated code. In QEMU the word is part of the CPU state.
//! - Vector ops are refused, so blocks with them run in the interpreter. QEMU uses RVV.

use ruvm_jit_core::ir::{Func, HelperType, Op, OpId, Temp};
use ruvm_jit_core::memory_model::{FenceMapping, ldst_flags};
use ruvm_jit_core::opcode::Opcode;
use ruvm_jit_core::regalloc::{self, Letter, RegSet, Target};
use ruvm_jit_core::types::{
    Cond, INSN_START_WORDS, MemOp, MemOpIdx, TempKind, Type, bswap, call_flags, mo, opf,
};
use ruvm_jit_interp::HelperRegistry;
use ruvm_jit_interp::fast_tlb::{
    TLB_ADDEND_WORD, TLB_DESC_WORDS, TLB_ENTRY_BITS, TLB_FLAGS_SHIFT, TLB_MAX_MMU_MODES,
};

use crate::asm::{
    self, A0, A1, A2, Asm, AsmError, RA, Reg, SP, TMP0, TMP1, TMP2, TMP3, ZERO, is_imm12, opc,
};
use crate::features::HostFeatures;

/// Base of the CPU state buffer, s0.
const ENV: Reg = 8;
/// Length of the CPU state buffer, s1.
const ENV_LEN: Reg = 9;
/// The run context, s2.
const CTX: Reg = 18;
/// The slot array, s3.
const SLOTS: Reg = 19;

/// Bytes per temp slot: room for a 256-bit vector.
pub(crate) const SLOT_BYTES: usize = 32;
/// Words of the run context used to pass operands to and from the service routine.
pub(crate) const NARGS: usize = 32;
/// Byte offset of the return value word in the run context, right after the argument words.
pub(crate) const RET_OFFSET: i64 = 8 * NARGS as i64;
/// Byte offset of the word holding one more than the index of the last `insn_start` request
/// before the exit generated code left through.
pub(crate) const INSN_OFFSET: i64 = RET_OFFSET + 8;
/// Byte offset of the length of the CPU state buffer.
pub(crate) const ENV_LEN_OFFSET: i64 = INSN_OFFSET + 8;
/// Byte offset of the word where a block stores the address of its [`GenOptions::meta`] when
/// it leaves.
pub(crate) const META_OFFSET: i64 = ENV_LEN_OFFSET + 8;
/// Byte offset of the address of the `icount_decr` word that [`GenOptions::icount_decr`]
/// loads read.
pub(crate) const DECR_OFFSET: i64 = META_OFFSET + 8;
/// Byte offset of the one block entry address `goto_ptr` may jump to, as the service routine
/// last checked it.
pub(crate) const GOTO_PTR_OK_OFFSET: i64 = DECR_OFFSET + 8;
/// Byte offset of the copy of the TLB descriptor the inline softmmu fast path reads, a
/// [`ruvm_jit_interp::FastTlb::desc`], so that a lookup reads the mask and table straight from
/// the run context, as QEMU's reads them from `env`.
pub(crate) const TLB_OFFSET: i64 = GOTO_PTR_OK_OFFSET + 8;

/// Bytes of stack frame: ra and s0 to s11, 16-byte aligned.
const FRAME: i64 = 112;
/// The callee-saved registers the prologue saves, at 8 times their index in the frame.
const SAVED: [Reg; 12] = [8, 9, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27];

/// What [`generate`] needs besides the IR.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GenOptions<'a> {
    /// Passed to the service routine with each request, so that it knows which block's requests
    /// it serves, and stored at [`META_OFFSET`] in the run context when the block leaves, so
    /// that the runtime knows which block left.
    pub(crate) meta: u64,
    /// A 32-bit load at this constant offset from `env` reads the word whose address is at
    /// [`DECR_OFFSET`] instead, so that the `icount_decr` check at the start of a chained block
    /// sees exit requests from other threads.
    pub(crate) icount_decr: Option<i64>,
    /// log2 of the guest page size of the TLB tables at [`TLB_OFFSET`], or `None` to send
    /// every `qemu_ld` and `qemu_st` to the service routine.
    pub(crate) tlb_page_bits: Option<u32>,
    /// The routine calls to `lookup_tb_ptr` go to instead of the service routine, or 0. It
    /// takes the same arguments: the context, the request index with the [`Gen::insn`] of the
    /// call in its upper 32 bits, so that neither routine need look it up, and the metadata.
    pub(crate) lookup: u64,
    /// Calls to helpers with a [`ruvm_jit_interp::NativeHelperFn`] here, and the declared
    /// signature, go straight to it rather than through the service routine.
    pub(crate) helpers: Option<&'a HelperRegistry>,
}

/// How generated code left, in a0 at the epilogue.
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
        /// The helper has `TCG_CALL_NO_SE`: it has no side effects and so cannot raise an
        /// exception or look at where the guest is.
        pure: bool,
    },
    /// `qemu_ld`: the address is in word 0, the value comes back in words 0 and 1.
    Load(MemOpIdx),
    /// `qemu_st`: the value is in words 0 and 1, the address in word 2.
    Store(MemOpIdx),
    /// The words of an `insn_start`. Never called; [`Generated::insn_of`] refers to it.
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
            GenCodeError::Unsupported(s) => write!(f, "not supported by the riscv64 backend: {s}"),
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
    /// For each request, one more than the index of the `insn_start` request before it in the
    /// code, or 0: the instruction a request made by a helper call or a memory access is part
    /// of, as QEMU finds it from the host return address.
    pub(crate) insn_of: Vec<u64>,
    /// Number of 64-bit words of slot array the code uses.
    pub(crate) slot_words: usize,
    /// For each `goto_tb`: its slot, the byte offset of its patchable word, and the word that
    /// sends it to the exit stub.
    pub(crate) goto_tb: Vec<(u32, usize, u32)>,
    /// The byte offset chained jumps enter the block at, after the prologue.
    pub(crate) body: usize,
    /// The byte offset after the static bounds check, where a chained jump may enter when the
    /// run's CPU state is known to be at least [`Generated::env_need`] bytes long.
    pub(crate) fast_body: usize,
    /// The length of CPU state the static bounds check asks for (`u64::MAX` if it always fails).
    pub(crate) env_need: u64,
    /// For each `lookup_tb_ptr_ic` call with an inline cache: the index of its request and the
    /// byte offset of the 8-byte constant pool word that holds the address of its [`IC_WAYS`]
    /// cache words, which is 0 until the caller patches it.
    pub(crate) ic_sites: Vec<(usize, usize)>,
}

/// Entries of the inline cache of a `lookup_tb_ptr_ic` call. Each is the address of the
/// two-word header of a block: the guest program counter it starts at, and the address to
/// jump to, or 0 for none.
pub(crate) const IC_WAYS: usize = 8;

/// The entry of the inline cache for the program counter `pc`.
pub(crate) fn ic_way(pc: u64) -> usize {
    ((pc ^ (pc >> 4)) as usize) & (IC_WAYS - 1)
}

/// The general registers the allocator may use: t0 to t2, a0 to a7 and s4 to s11.
const GPRS: RegSet = RegSet(0x7 << 5 | 0xff << 10 | 0xff << 20);
/// `tcg_target_reg_alloc_order`: callee-saved registers first, so values survive calls.
const ALLOC_ORDER: [Reg; 19] =
    [20, 21, 22, 23, 24, 25, 26, 27, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17];
/// The registers a call to Rust may change: ra, t0 to t6 and a0 to a7.
const CALL_CLOBBER: RegSet = RegSet(1 << 1 | 0x7 << 5 | 0xff << 10 | 0xf << 28);
/// Never allocated: zero, ra, sp, gp, tp, the fixed registers and the scratch registers.
const RESERVED: RegSet = RegSet(0x1f | 0x3 << 8 | 0x3 << 18 | 0xf << 28);

/// Target constant classes, `TCG_CT_CONST_*`.
mod ctc {
    /// A 12-bit signed immediate, `TCG_CT_CONST_S12`.
    pub(super) const S12: u32 = 0x100;
    /// A value whose negation is a 12-bit signed immediate, `TCG_CT_CONST_N12`.
    pub(super) const N12: u32 = 0x400;
    /// A value from -0x7ff to 0x7ff, so that it and its negation are immediates,
    /// `TCG_CT_CONST_M12`.
    pub(super) const M12: u32 = 0x200;
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
        | Opcode::St8
        | Opcode::St16
        | Opcode::St32
        | Opcode::St => may_fault(f, op, 1),
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
pub(crate) fn generate(
    f: &Func,
    base: u64,
    service: u64,
    feat: HostFeatures,
    opts: &GenOptions<'_>,
) -> R<Generated> {
    check_types(f)?;
    let (prepared, live) = regalloc::prepare_live(f, &extra_flags);
    let f: &Func = &prepared;
    let c = &f.config;
    let mut g = Gen {
        mapping: c.fence_mapping.effective(c.guest_mo, c.target_default_mo),
        addr32: c.addr_type == Type::I32,
        a: Asm::new(base, feat.normalized()),
        labels: vec![None; f.nb_labels()],
        requests: Vec::new(),
        insn_of: Vec::new(),
        insn: 0,
        meta: opts.meta,
        goto_tb: Vec::new(),
        service,
        lookup: opts.lookup,
        exit: 0,
        bounds: 0,
        static_end: 0,
        static_access: (0, 0),
        static_always_fails: false,
        nb_temps: f.nb_temps(),
        icount_decr: opts.icount_decr,
        tlb_page_bits: opts.tlb_page_bits,
        err: None,
        ic_sites: Vec::new(),
        helpers: opts.helpers,
        slow_paths: Vec::new(),
    };
    g.exit = g.a.new_label();
    g.bounds = g.a.new_label();
    let static_fail = g.a.new_label();

    // Prologue: save ra and the callee-saved registers, then take the arguments.
    g.a.i(opc::ADDI, SP, SP, -FRAME);
    g.a.s(opc::SD, SP, RA, FRAME - 8);
    for (k, &r) in SAVED.iter().enumerate() {
        g.a.s(opc::SD, SP, r, 8 * k as i64);
    }
    g.a.mov(ENV, A0);
    g.a.mov(CTX, A1);
    g.a.mov(SLOTS, A2);
    g.a.ld(Type::I64, ENV_LEN, CTX, ENV_LEN_OFFSET);
    // Chained jumps from other blocks enter here, with the same frame and fixed registers.
    let body = g.a.pos() * 4;
    // The static bounds check; the pool word is set once the body is known.
    let check = g.a.pool_unique(TMP0, 0);
    let check_at = g.a.pos();
    g.a.emit(asm::encode_sb(opc::BGEU, ENV_LEN, TMP0, 8));
    g.a.j_label(static_fail);
    let fast_body = g.a.pos() * 4;

    regalloc::reg_alloc_live(f, live, &mut g)?;
    if let Some(e) = g.err.take() {
        return Err(e);
    }
    g.exit_with(kind::FELL_OFF, None);

    // The miss paths of inline guest accesses, `tcg_out_ldst_finalize`.
    for sp in std::mem::take(&mut g.slow_paths) {
        g.a.bind(sp.slow);
        g.insn = sp.insn;
        g.ldst_slow(&sp);
        g.a.j_label(sp.done);
    }

    // Exit stubs for linked goto_tb slots.
    let mut goto_tb = Vec::new();
    for (slot, at, insn) in std::mem::take(&mut g.goto_tb) {
        let stub = g.a.pos();
        g.insn = insn;
        g.note_exit();
        g.a.movi(Type::I64, A1, slot as i64);
        g.a.movi(Type::I64, A0, kind::GOTO_TB as i64);
        g.a.j_label(g.exit);
        let word = asm::jal_word((stub as i64 - at as i64) * 4).ok_or(GenCodeError::TooLarge)?;
        goto_tb.push((slot, at * 4, word));
    }

    // A failed static check reports the access that reaches furthest.
    g.a.bind(static_fail);
    g.a.movi(Type::I64, TMP0, g.static_access.0 as i64);
    g.a.movi(Type::I64, TMP1, g.static_access.0.wrapping_add(g.static_access.1) as i64);
    g.a.j_label(g.bounds);

    // A failed bounds check: offset in TMP0, end in TMP1.
    g.a.bind(g.bounds);
    g.a.r(opc::SUB, TMP1, TMP1, TMP0);
    g.a.st(Type::I64, TMP1, CTX, 0);
    g.a.movi(Type::I64, TMP2, g.meta as i64);
    g.a.st(Type::I64, TMP2, CTX, META_OFFSET);
    g.a.mov(A1, TMP0);
    g.a.movi(Type::I64, A0, kind::BOUNDS as i64);

    // The epilogue: a0 is the kind, a1 the return word.
    g.a.bind(g.exit);
    g.a.st(Type::I64, A1, CTX, RET_OFFSET);
    for (k, &r) in SAVED.iter().enumerate() {
        g.a.ld(Type::I64, r, SP, 8 * k as i64);
    }
    g.a.ld(Type::I64, RA, SP, FRAME - 8);
    g.a.i(opc::ADDI, SP, SP, FRAME);
    g.a.jr(RA);

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
    g.a.set_unique(check, end);
    if g.static_always_fails {
        g.a.code[check_at] = opc::NOP;
    }

    let slot_words = (f.nb_temps() + 1) * SLOT_BYTES / 8;
    let (requests, insn_of, ic) = (g.requests, g.insn_of, g.ic_sites);
    let out = g.a.finish()?;
    let ic_sites = ic.into_iter().map(|(req, h)| (req, out.uniques[h])).collect();
    let env_need = if g.static_always_fails { u64::MAX } else { end };
    Ok(Generated {
        bytes: out.bytes,
        requests,
        insn_of,
        slot_words,
        goto_tb,
        body,
        fast_body,
        env_need,
        ic_sites,
    })
}

/// Refuse temps of types this backend has no registers for, vector ops, and calls with more
/// arguments than the run context holds.
fn check_types(f: &Func) -> R<()> {
    for (_, op) in f.ops() {
        let n = op.nb_oargs() + op.nb_iargs();
        for k in 0..n {
            let ty = f.temp(op.arg_temp(k)).ty;
            if !matches!(ty, Type::I32 | Type::I64) {
                return Err(GenCodeError::Unsupported(format!("{}: {ty:?} temps", op.opc.name())));
            }
        }
        if op.opc.def().flags & opf::VECTOR != 0 {
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

/// A constant operand of an `op.ty` op, sign extended from 32 bits for I32 as QEMU does.
fn norm(ty: Type, v: u64) -> i64 {
    if ty == Type::I32 { v as i32 as i64 } else { v as i64 }
}

/// `len` low bits set.
fn field_mask(len: u32) -> u64 {
    if len >= 64 { u64::MAX } else { (1u64 << len) - 1 }
}

/// `tcg_out_setcond_int` returns a register with these flags: the result is inverted, and
/// the register holds zero or nonzero rather than zero or one.
const SETCOND_INV: u32 = 1;
const SETCOND_NEZ: u32 = 2;

/// Where a host memory access goes.
enum Addr {
    /// At this constant offset into the CPU state, covered by the check on entry.
    Static(i64),
    /// At the offset in TMP0, already bounds checked.
    Dyn,
}

struct Gen<'h> {
    /// The fence mapping the block was built with, after `FenceMapping::effective`.
    mapping: FenceMapping,
    /// Guest addresses are 32 bits wide.
    addr32: bool,
    a: Asm,
    labels: Vec<Option<usize>>,
    requests: Vec<Request>,
    /// See [`Generated::insn_of`].
    insn_of: Vec<u64>,
    /// One more than the index of the last `insn_start` request so far, or 0.
    insn: u64,
    /// See [`GenOptions::meta`].
    meta: u64,
    /// For each `goto_tb`: its slot, the word index of its jump, and [`Gen::insn`] there.
    goto_tb: Vec<(u32, usize, u64)>,
    service: u64,
    /// See [`GenOptions::lookup`].
    lookup: u64,
    exit: usize,
    bounds: usize,
    /// The furthest end of a constant offset CPU state access, and that access.
    static_end: u64,
    static_access: (u64, u64),
    /// A constant offset access is outside what the check on entry can describe.
    static_always_fails: bool,
    nb_temps: usize,
    /// See [`GenOptions::icount_decr`].
    icount_decr: Option<i64>,
    /// See [`GenOptions::tlb_page_bits`].
    tlb_page_bits: Option<u32>,
    /// An error from a hook that cannot return one.
    err: Option<GenCodeError>,
    /// For each inline cache: its request index and the handle of its pool word.
    ic_sites: Vec<(usize, usize)>,
    /// See [`GenOptions::helpers`].
    helpers: Option<&'h HelperRegistry>,
    /// The miss paths of guest accesses whose hit path is inline, emitted after the block.
    slow_paths: Vec<SlowPath>,
}

/// The miss path of one `qemu_ld` or `qemu_st`, kept until the end of the block as QEMU's
/// `TCGLabelQemuLdst` is, so that a hit falls straight through.
struct SlowPath {
    /// Where the hit path branches on a miss, and where the miss path returns to.
    slow: usize,
    done: usize,
    /// [`Gen::insn`] at the access.
    insn: u64,
    store: bool,
    /// A 128-bit access in two registers.
    two: bool,
    /// The type of a load's result.
    ty: Type,
    /// The data registers, and the address register.
    data: [Reg; 2],
    addr: Reg,
    oi: MemOpIdx,
}

// Constraint sets, `tcg-target-con-set.h`.
const C_R: &[&str] = &["r"];
const C_R_R: &[&str] = &["r", "r"];
const C_R_RI: &[&str] = &["r", "ri"];
const C_RZ_RI: &[&str] = &["rz", "ri"];
const C_RZ_R: &[&str] = &["rz", "r"];
const C_RZ_RZ: &[&str] = &["rz", "rz"];
const C_RZ_RZ_R: &[&str] = &["rz", "rz", "r"];
const C_R_R_R: &[&str] = &["r", "r", "r"];
const C_R_R_RI_I: &[&str] = &["r", "r", "rI"];
const C_R_RZ_RJ: &[&str] = &["r", "rz", "rJ"];
const C_R_R_RI: &[&str] = &["r", "r", "ri"];
const C_R_RZ_RZ: &[&str] = &["r", "rz", "rz"];
const C_R_0_RZ: &[&str] = &["r", "0", "rz"];
const C_R_R_R_R: &[&str] = &["r", "r", "r", "r"];
const C_R5: &[&str] = &["r", "r", "r", "r", "r"];
const C_MOVCOND: &[&str] = &["r", "r", "rI", "rM", "rM"];
const C_NONE: &[&str] = &[];

impl Gen<'_> {
    fn label(&mut self, op: &Op, i: usize) -> R<usize> {
        let id = op.arg_label(i).id() as usize;
        // The allocator numbers the labels of its branch stubs after the function's own.
        if id >= self.labels.len() {
            self.labels.resize(id + 1, None);
        }
        let slot = &mut self.labels[id];
        Ok(match slot {
            Some(l) => *l,
            None => {
                let l = self.a.new_label();
                *slot = Some(l);
                l
            }
        })
    }

    fn feat(&self) -> HostFeatures {
        self.a.feat
    }

    /// The offset of the carry flag word in the slot array, after every temp's slot.
    fn carry_offset(&self) -> i64 {
        (self.nb_temps * SLOT_BYTES) as i64
    }

    fn exit_with(&mut self, k: u64, value: Option<u64>) {
        self.note_exit();
        if let Some(v) = value {
            self.a.movi(Type::I64, A1, v as i64);
        }
        self.a.movi(Type::I64, A0, k as i64);
        self.a.j_label(self.exit);
    }

    /// Record in the run context that this block left, after the instruction of [`Gen::insn`].
    fn note_exit(&mut self) {
        self.a.movi(Type::I64, TMP0, self.meta as i64);
        self.a.st(Type::I64, TMP0, CTX, META_OFFSET);
        self.a.movi(Type::I64, TMP0, self.insn as i64);
        self.a.st(Type::I64, TMP0, CTX, INSN_OFFSET);
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

    /// Check that `len` bytes at the offset in TMP0 are inside the CPU state, leaving the end
    /// in TMP1 as the bounds exit expects.
    fn check_bounds(&mut self, len: u64) {
        self.a.addi(TMP1, TMP0, len as i64);
        self.a.far_b_label(opc::BLTU, TMP1, TMP0, self.bounds);
        self.a.far_b_label(opc::BLTU, ENV_LEN, TMP1, self.bounds);
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
            self.a.movi(Type::I64, TMP0, (args[bi] as i64).wrapping_add(off));
        } else {
            self.a.addi(TMP0, args[bi] as Reg, off);
        }
        self.check_bounds(len);
        Addr::Dyn
    }

    /// A host load or store of `rt` with `insn`.
    fn host_access(&mut self, addr: Addr, insn: u32, rt: Reg) {
        match addr {
            Addr::Static(off) => self.a.ldst(insn, rt, ENV, off),
            Addr::Dyn => {
                self.a.r(opc::ADD, TMP0, ENV, TMP0);
                self.a.ldst(insn, rt, TMP0, 0);
            }
        }
    }

    /// Call the service routine for `req`. Leaves the block if it reports an unwind or error.
    fn service(&mut self, req: Request) {
        self.service_via(req, None, self.service, self.insn, 0);
    }

    /// [`Self::service`] through the routine at `routine`, with `tag` in the upper half of the
    /// request index and `site` or'ed into the lower half, and `after` emitted right after the
    /// call returns.
    fn service_via(&mut self, req: Request, after: Option<u32>, routine: u64, tag: u64, site: u64) {
        let idx = self.requests.len();
        self.requests.push(req);
        self.insn_of.push(self.insn);
        self.a.mov(A0, CTX);
        self.a.movi(Type::I64, A1, (idx as u64 | site | tag << 32) as i64);
        self.a.movi(Type::I64, A2, self.meta as i64);
        self.a.call_abs(routine);
        if let Some(w) = after {
            self.a.emit(w);
        }
        let ok = self.a.new_label();
        self.a.b_label(opc::BEQ, A0, ZERO, ok);
        self.a.movi(Type::I64, A1, 0);
        self.a.j_label(self.exit);
        self.a.bind(ok);
    }

    /// A direct call to the [`ruvm_jit_interp::NativeHelperFn`] at `addr` of a helper with
    /// `nin` argument words, taking them from and leaving its result in the argument words, as
    /// the service routine would. Such a helper has no side effects, so it cannot raise an
    /// exception and needs no guest state, and as in QEMU the call is all there is to it.
    fn call_native(&mut self, addr: u64, nin: usize, ret: HelperType) {
        for k in 0..nin {
            self.a.ld(Type::I64, A0 + k as Reg, CTX, 8 * k as i64);
        }
        self.a.call_abs(addr);
        if ret != HelperType::Void {
            self.a.st(Type::I64, A0, CTX, 0);
        }
    }

    /// The inline cache of a `lookup_tb_ptr_ic` call, whose arguments are already in the run
    /// context: when the entry for the program counter in argument 1 names a block header with
    /// that program counter and a nonzero address, jump there; otherwise fall through to the
    /// lookup. The address of the cache words is a pool word the runtime patches. The entry is
    /// read before the header it names, an address dependency, so a header published with a
    /// release store before the entry is seen whole.
    fn ic_probe(&mut self) {
        let miss = self.a.new_label();
        self.a.ld(Type::I64, TMP0, CTX, 8);
        // The entry, ic_way(pc), as a byte offset: ((pc ^ pc >> 4) & 7) << 3.
        self.a.i(opc::SRLI, TMP2, TMP0, 4);
        self.a.r(opc::XOR, TMP2, TMP2, TMP0);
        self.a.i(opc::ANDI, TMP2, TMP2, (IC_WAYS - 1) as i64);
        self.a.i(opc::SLLI, TMP2, TMP2, 3);
        let h = self.a.pool_unique(TMP1, 0);
        self.ic_sites.push((self.requests.len(), h));
        self.a.r(opc::ADD, TMP1, TMP1, TMP2);
        self.a.ld(Type::I64, TMP1, TMP1, 0);
        self.a.ld(Type::I64, TMP2, TMP1, 0);
        self.a.b_label(opc::BNE, TMP0, TMP2, miss);
        self.a.ld(Type::I64, TMP1, TMP1, 8);
        self.a.b_label(opc::BEQ, TMP1, ZERO, miss);
        self.a.jr(TMP1);
        self.a.bind(miss);
    }

    /// `tcg_out_setcond_int`: compute `a1 cond a2` into `ret` or another register, and return
    /// that register with [`SETCOND_INV`] and [`SETCOND_NEZ`] in the bits above it.
    fn setcond_int(&mut self, cond: Cond, ret: Reg, a1: Reg, a2: i64, c2: bool) -> u32 {
        let mut flags = 0;
        let (mut cond, mut a1, mut a2, mut c2, mut ret) = (cond, a1, a2, c2, ret);
        if matches!(cond, Cond::Eq | Cond::Ge | Cond::Geu | Cond::Gt | Cond::Gtu | Cond::TstEq) {
            cond = cond.invert();
            flags ^= SETCOND_INV;
        }
        if matches!(cond, Cond::Le | Cond::Leu) {
            if c2 {
                // Add 1 and use LT; LEU against all ones is always true.
                if cond == Cond::Leu {
                    if a2 == -1 {
                        self.a.movi(Type::I64, ret, (flags & SETCOND_INV == 0) as i64);
                        return ret as u32;
                    }
                    cond = Cond::Ltu;
                } else {
                    cond = Cond::Lt;
                }
                a2 += 1;
                if a2 == 0x800 {
                    self.a.movi(Type::I64, TMP0, a2);
                    a2 = TMP0 as i64;
                    c2 = false;
                }
            } else {
                let t = a1;
                a1 = a2 as Reg;
                a2 = t as i64;
                cond = cond.swap().invert();
                flags ^= SETCOND_INV;
            }
        }
        match cond {
            Cond::Ne => {
                flags |= SETCOND_NEZ;
                if !c2 {
                    self.a.r(opc::XOR, ret, a1, a2 as Reg);
                } else if a2 == 0 {
                    ret = a1;
                } else {
                    self.a.i(opc::XORI, ret, a1, a2);
                }
            }
            Cond::TstNe => {
                flags |= SETCOND_NEZ;
                if !c2 {
                    self.a.r(opc::AND, ret, a1, a2 as Reg);
                } else {
                    self.a.i(opc::ANDI, ret, a1, a2);
                }
            }
            Cond::Lt => {
                if c2 {
                    self.a.i(opc::SLTI, ret, a1, a2);
                } else {
                    self.a.r(opc::SLT, ret, a1, a2 as Reg);
                }
            }
            _ => {
                if c2 {
                    self.a.i(opc::SLTIU, ret, a1, a2);
                } else {
                    self.a.r(opc::SLTU, ret, a1, a2 as Reg);
                }
            }
        }
        ret as u32 | flags << 8
    }

    /// `tcg_out_setcond`.
    fn setcond(&mut self, cond: Cond, ret: Reg, a1: Reg, a2: i64, c2: bool) {
        let tf = self.setcond_int(cond, ret, a1, a2, c2);
        let tmp = tf as Reg;
        match tf >> 8 {
            0 => {}
            SETCOND_INV => self.a.i(opc::XORI, ret, tmp, 1),
            SETCOND_NEZ => self.a.r(opc::SLTU, ret, ZERO, tmp),
            _ => self.a.i(opc::SLTIU, ret, tmp, 1),
        }
    }

    /// `tcg_out_negsetcond`.
    fn negsetcond(&mut self, cond: Cond, ret: Reg, a1: Reg, a2: i64, c2: bool) {
        // For LT and GE against 0, replicate the sign bit.
        if c2 && a2 == 0 && matches!(cond, Cond::Lt | Cond::Ge) {
            let mut src = a1;
            if cond == Cond::Ge {
                self.a.i(opc::XORI, ret, a1, -1);
                src = ret;
            }
            self.a.i(opc::SRAI, ret, src, 63);
            return;
        }
        let tf = self.setcond_int(cond, ret, a1, a2, c2);
        let mut tmp = tf as Reg;
        if (tf >> 8) & SETCOND_NEZ != 0 {
            self.a.r(opc::SLTU, ret, ZERO, tmp);
            tmp = ret;
        }
        if (tf >> 8) & SETCOND_INV != 0 {
            self.a.i(opc::ADDI, ret, tmp, -1);
        } else {
            self.a.r(opc::SUB, ret, ZERO, tmp);
        }
    }

    /// `tcg_out_movcond_zicond`: `ret = test_ne != 0 ? v1 : v2`.
    fn movcond_zicond(&mut self, ret: Reg, test_ne: Reg, v1: (i64, bool), v2: (i64, bool)) {
        let zero = |v: (i64, bool)| v.0 == 0;
        if zero(v1) {
            let r2 = self.reg_or_tmp1(v2);
            self.a.r(opc::CZERO_NEZ, ret, r2, test_ne);
            return;
        }
        if zero(v2) {
            let r1 = self.reg_or_tmp1(v1);
            self.a.r(opc::CZERO_EQZ, ret, r1, test_ne);
            return;
        }
        if v2.1 {
            if v1.1 {
                self.a.movi(Type::I64, TMP1, v1.0 - v2.0);
            } else {
                self.a.i(opc::ADDI, TMP1, v1.0 as Reg, -v2.0);
            }
            self.a.r(opc::CZERO_EQZ, ret, TMP1, test_ne);
            self.a.i(opc::ADDI, ret, ret, v2.0);
            return;
        }
        if v1.1 {
            self.a.i(opc::ADDI, TMP1, v2.0 as Reg, -v1.0);
            self.a.r(opc::CZERO_NEZ, ret, TMP1, test_ne);
            self.a.i(opc::ADDI, ret, ret, v1.0);
            return;
        }
        self.a.r(opc::CZERO_NEZ, TMP1, v2.0 as Reg, test_ne);
        self.a.r(opc::CZERO_EQZ, TMP0, v1.0 as Reg, test_ne);
        self.a.r(opc::OR, ret, TMP0, TMP1);
    }

    /// The register of `v`, or the constant in TMP1.
    fn reg_or_tmp1(&mut self, v: (i64, bool)) -> Reg {
        if v.1 {
            self.a.movi(Type::I64, TMP1, v.0);
            TMP1
        } else {
            v.0 as Reg
        }
    }

    /// `tcg_out_movcond_br1`: `ret = val` unless `cmp1 cond cmp2`.
    fn movcond_br1(&mut self, cond: Cond, ret: Reg, cmp1: Reg, cmp2: Reg, val: (i64, bool)) {
        let (op, swap) = asm::brcond_insn(cond).expect("movcond_br1 takes a branch condition");
        let (x, y) = if swap { (cmp2, cmp1) } else { (cmp1, cmp2) };
        self.a.emit(asm::encode_sb(op, x, y, 8));
        if val.1 {
            self.a.i(opc::ADDI, ret, ZERO, val.0);
        } else {
            self.a.i(opc::ADDI, ret, val.0 as Reg, 0);
        }
    }

    /// `tcg_out_movcond_br2`.
    #[allow(clippy::too_many_arguments, reason = "the operands of movcond, already allocated")]
    fn movcond_br2(
        &mut self,
        cond: Cond,
        ret: Reg,
        cmp1: Reg,
        cmp2: Reg,
        v1: (i64, bool),
        v2: (i64, bool),
    ) {
        // The optimizer prefers ret matching v2.
        if !v2.1 && ret == v2.0 as Reg {
            self.movcond_br1(cond.invert(), ret, cmp1, cmp2, v1);
            return;
        }
        if !v1.1 && ret == v1.0 as Reg {
            self.movcond_br1(cond, ret, cmp1, cmp2, v2);
            return;
        }
        let tmp = if ret == cmp1 || ret == cmp2 { TMP1 } else { ret };
        if v1.1 {
            self.a.movi(Type::I64, tmp, v1.0);
        } else {
            self.a.mov(tmp, v1.0 as Reg);
        }
        self.movcond_br1(cond, tmp, cmp1, cmp2, v2);
        self.a.mov(ret, tmp);
    }

    /// `tcg_out_movcond`.
    #[allow(clippy::too_many_arguments, reason = "the operands of movcond, already allocated")]
    fn movcond(
        &mut self,
        cond: Cond,
        ret: Reg,
        cmp1: Reg,
        cmp2: (i64, bool),
        v1: (i64, bool),
        v2: (i64, bool),
    ) {
        let zicond = self.feat().zicond;
        if !zicond && !cond.is_tst() && (!cmp2.1 || cmp2.0 == 0) {
            let c2 = if cmp2.1 { ZERO } else { cmp2.0 as Reg };
            self.movcond_br2(cond, ret, cmp1, c2, v1, v2);
            return;
        }
        let tf = self.setcond_int(cond, TMP0, cmp1, cmp2.0, cmp2.1);
        let t = tf as Reg;
        let inv = (tf >> 8) & SETCOND_INV != 0;
        if zicond {
            if inv {
                self.movcond_zicond(ret, t, v2, v1);
            } else {
                self.movcond_zicond(ret, t, v1, v2);
            }
        } else {
            let c = if inv { Cond::Eq } else { Cond::Ne };
            self.movcond_br2(c, ret, t, ZERO, v1, v2);
        }
    }

    /// `tcg_out_brcond`, to a label anywhere in the block.
    fn brcond(&mut self, c: Cond, a: Reg, b: Reg, l: usize) {
        let (mut c, mut a, mut b) = (c, a, b);
        if c.is_tst() {
            self.a.r(opc::AND, TMP0, a, b);
            a = TMP0;
            b = ZERO;
            c = c.tst_eqne();
        }
        let (op, swap) = asm::brcond_insn(c).expect("a branch condition");
        let (x, y) = if swap { (b, a) } else { (a, b) };
        self.a.far_b_label(op, x, y, l);
    }

    /// The add and subtract with carry family. The carry lives in a word after the temp slots,
    /// except between two adjacent ops of the same family, where it stays in TMP3.
    fn carry_op(&mut self, f: &Func, id: OpId, op: &Op, args: &[u64]) {
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
                self.a.ld(Type::I64, TMP3, SLOTS, carry);
            }
        }
        let cin = carry_in || matches!(op.opc, Opcode::Addc1o | Opcode::Subb1o);
        if matches!(op.opc, Opcode::Addc1o | Opcode::Subb1o) {
            self.a.i(opc::ADDI, TMP3, ZERO, 1);
        }
        if op.ty == Type::I32 {
            // The full sum or difference of the zero extended operands; the carry or borrow is
            // bit 32 of a sum and the sign of a difference.
            self.a.ext32u(TMP0, a);
            self.a.ext32u(TMP1, b);
            self.a.r(if sub { opc::SUB } else { opc::ADD }, TMP0, TMP0, TMP1);
            if cin {
                self.a.r(if sub { opc::SUB } else { opc::ADD }, TMP0, TMP0, TMP3);
            }
            if carry_out {
                self.a.i(opc::SRLI, TMP1, TMP0, if sub { 63 } else { 32 });
            }
            self.a.ext32s(d, TMP0);
        } else if sub {
            self.a.r(opc::SLTU, TMP1, a, b);
            self.a.r(opc::SUB, TMP0, a, b);
            if cin {
                self.a.r(opc::SLTU, TMP2, TMP0, TMP3);
                self.a.r(opc::SUB, TMP0, TMP0, TMP3);
                self.a.r(opc::OR, TMP1, TMP1, TMP2);
            }
            self.a.mov(d, TMP0);
        } else {
            self.a.r(opc::ADD, TMP0, a, b);
            self.a.r(opc::SLTU, TMP1, TMP0, a);
            if cin {
                self.a.r(opc::ADD, TMP0, TMP0, TMP3);
                self.a.r(opc::SLTU, TMP2, TMP0, TMP3);
                self.a.r(opc::OR, TMP1, TMP1, TMP2);
            }
            self.a.mov(d, TMP0);
        }
        if carry_out {
            let fused = f.next_op(id).is_some_and(|n| {
                let n = f.op(n);
                family_in.contains(&n.opc) && n.ty == op.ty
            });
            if fused {
                self.a.mov(TMP3, TMP1);
            } else {
                self.a.st(Type::I64, TMP1, SLOTS, carry);
            }
        }
    }

    /// Store `regs` to the argument words, from word 0.
    fn put_args(&mut self, regs: &[u64]) {
        for (k, &r) in regs.iter().enumerate() {
            self.a.st(Type::I64, r as Reg, CTX, 8 * k as i64);
        }
    }

    /// `d = (a == 0) ? b : TMP0`, the result of `clz` and `ctz` for a zero input.
    fn select_zero(&mut self, d: Reg, a: Reg, b: (i64, bool)) {
        if self.feat().zicond {
            self.a.r(opc::CZERO_EQZ, TMP0, TMP0, a);
            let rb = self.reg_or_tmp1(b);
            self.a.r(opc::CZERO_NEZ, TMP1, rb, a);
            self.a.r(opc::OR, d, TMP0, TMP1);
            return;
        }
        let (lb, done) = (self.a.new_label(), self.a.new_label());
        self.a.b_label(opc::BEQ, a, ZERO, lb);
        self.a.mov(d, TMP0);
        self.a.j_label(done);
        self.a.bind(lb);
        if b.1 {
            self.a.movi(Type::I64, d, b.0);
        } else {
            self.a.mov(d, b.0 as Reg);
        }
        self.a.bind(done);
    }

    /// `clz` of the `w` low bits of `a` into TMP0 without Zbb, a binary search that gives `w`
    /// for zero.
    fn clz_soft(&mut self, w: u32, a: Reg) {
        self.a.i(opc::SLLI, TMP1, a, (64 - w) as i64);
        self.a.mov(TMP0, ZERO);
        let mut s = w / 2;
        while s > 0 {
            self.a.i(opc::SRLI, TMP2, TMP1, (64 - s) as i64);
            self.a.i(opc::SLTIU, TMP2, TMP2, 1);
            self.a.i(opc::SLLI, TMP2, TMP2, s.trailing_zeros() as i64);
            self.a.r(opc::ADD, TMP0, TMP0, TMP2);
            self.a.r(opc::SLL, TMP1, TMP1, TMP2);
            s /= 2;
        }
        self.a.i(opc::SRLI, TMP2, TMP1, 63);
        self.a.i(opc::XORI, TMP2, TMP2, 1);
        self.a.r(opc::ADD, TMP0, TMP0, TMP2);
    }

    /// `ctz` of the `w` low bits of `a` into TMP0 without Zbb, as [`Self::clz_soft`].
    fn ctz_soft(&mut self, w: u32, a: Reg) {
        self.a.mov(TMP1, a);
        self.a.mov(TMP0, ZERO);
        let mut s = w / 2;
        while s > 0 {
            self.a.i(opc::SLLI, TMP2, TMP1, (64 - s) as i64);
            self.a.i(opc::SLTIU, TMP2, TMP2, 1);
            self.a.i(opc::SLLI, TMP2, TMP2, s.trailing_zeros() as i64);
            self.a.r(opc::ADD, TMP0, TMP0, TMP2);
            self.a.r(opc::SRL, TMP1, TMP1, TMP2);
            s /= 2;
        }
        self.a.i(opc::ANDI, TMP2, TMP1, 1);
        self.a.i(opc::XORI, TMP2, TMP2, 1);
        self.a.r(opc::ADD, TMP0, TMP0, TMP2);
    }

    /// `ctpop` of the zero extended value in TMP1 into `d` without Zbb.
    fn ctpop_soft(&mut self, d: Reg) {
        let m1 = 0x5555_5555_5555_5555u64 as i64;
        let m2 = 0x3333_3333_3333_3333u64 as i64;
        let m4 = 0x0f0f_0f0f_0f0f_0f0fu64 as i64;
        let h01 = 0x0101_0101_0101_0101u64 as i64;
        self.a.movi(Type::I64, TMP2, m1);
        self.a.i(opc::SRLI, TMP0, TMP1, 1);
        self.a.r(opc::AND, TMP0, TMP0, TMP2);
        self.a.r(opc::SUB, TMP1, TMP1, TMP0);
        self.a.movi(Type::I64, TMP2, m2);
        self.a.i(opc::SRLI, TMP0, TMP1, 2);
        self.a.r(opc::AND, TMP0, TMP0, TMP2);
        self.a.r(opc::AND, TMP1, TMP1, TMP2);
        self.a.r(opc::ADD, TMP1, TMP1, TMP0);
        self.a.i(opc::SRLI, TMP0, TMP1, 4);
        self.a.r(opc::ADD, TMP1, TMP1, TMP0);
        self.a.movi(Type::I64, TMP2, m4);
        self.a.r(opc::AND, TMP1, TMP1, TMP2);
        self.a.movi(Type::I64, TMP2, h01);
        self.a.r(opc::MUL, TMP1, TMP1, TMP2);
        self.a.i(opc::SRLI, d, TMP1, 56);
    }

    /// Byte swap the low `bytes` bytes of `a` into TMP0 without Zbb, the top byte of the
    /// result sign extended when `sext`.
    fn bswap_soft(&mut self, bytes: u32, a: Reg, sext: bool) {
        // The lowest byte goes to the top of the result.
        let top = 8 * (bytes - 1);
        self.a.i(opc::SLLI, TMP0, a, 56);
        if top < 56 {
            self.a.i(if sext { opc::SRAI } else { opc::SRLI }, TMP0, TMP0, (56 - top) as i64);
        }
        for k in 1..bytes {
            self.a.i(opc::SRLI, TMP1, a, 8 * k as i64);
            if k < 7 {
                self.a.i(opc::ANDI, TMP1, TMP1, 0xff);
            }
            let to = 8 * (bytes - 1 - k);
            if to > 0 {
                self.a.i(opc::SLLI, TMP1, TMP1, to as i64);
            }
            self.a.r(opc::OR, TMP0, TMP0, TMP1);
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
        let w32 = ty == Type::I32;
        let w = ty.bits();
        let feat = self.feat();
        let r = |k: usize| args[k] as Reg;
        let (d, a1) = (r(0), if args.len() > 1 { r(1) } else { 0 });
        // The 32-bit form of an op for I32, else the 64-bit one.
        let pick = |o64: u32, o32: u32| if w32 { o32 } else { o64 };
        match op.opc {
            Opcode::ExtI32I64 | Opcode::ExtrlI64I32 => self.a.ext32s(d, a1),
            Opcode::ExtuI32I64 => self.a.ext32u(d, a1),
            Opcode::ExtrhI64I32 => self.a.i(opc::SRAI, d, a1, 32),
            Opcode::Add => {
                if const_args[2] {
                    self.a.i(pick(opc::ADDI, opc::ADDIW), d, a1, norm(ty, args[2]));
                } else {
                    self.a.r(pick(opc::ADD, opc::ADDW), d, a1, r(2));
                }
            }
            Opcode::Sub => {
                if const_args[2] {
                    self.a.i(pick(opc::ADDI, opc::ADDIW), d, a1, -norm(ty, args[2]));
                } else {
                    self.a.r(pick(opc::SUB, opc::SUBW), d, a1, r(2));
                }
            }
            Opcode::And | Opcode::Or | Opcode::Xor => {
                let (ro, io) = match op.opc {
                    Opcode::And => (opc::AND, opc::ANDI),
                    Opcode::Or => (opc::OR, opc::ORI),
                    _ => (opc::XOR, opc::XORI),
                };
                if const_args[2] {
                    self.a.i(io, d, a1, norm(ty, args[2]));
                } else {
                    self.a.r(ro, d, a1, r(2));
                }
            }
            Opcode::Andc | Opcode::Orc | Opcode::Eqv => {
                let (zo, base) = match op.opc {
                    Opcode::Andc => (opc::ANDN, opc::AND),
                    Opcode::Orc => (opc::ORN, opc::OR),
                    _ => (opc::XNOR, opc::XOR),
                };
                if feat.zbb {
                    self.a.r(zo, d, a1, r(2));
                } else if op.opc == Opcode::Eqv {
                    self.a.r(opc::XOR, d, a1, r(2));
                    self.a.i(opc::XORI, d, d, -1);
                } else {
                    self.a.i(opc::XORI, TMP0, r(2), -1);
                    self.a.r(base, d, a1, TMP0);
                }
            }
            Opcode::Nand | Opcode::Nor => {
                let o = if op.opc == Opcode::Nand { opc::AND } else { opc::OR };
                self.a.r(o, d, a1, r(2));
                self.a.i(opc::XORI, d, d, -1);
            }
            Opcode::Not => self.a.i(opc::XORI, d, a1, -1),
            Opcode::Neg => self.a.r(pick(opc::SUB, opc::SUBW), d, ZERO, a1),
            Opcode::Shl | Opcode::Shr | Opcode::Sar => {
                let (ro, io) = match op.opc {
                    Opcode::Shl => (pick(opc::SLL, opc::SLLW), pick(opc::SLLI, opc::SLLIW)),
                    Opcode::Shr => (pick(opc::SRL, opc::SRLW), pick(opc::SRLI, opc::SRLIW)),
                    _ => (pick(opc::SRA, opc::SRAW), pick(opc::SRAI, opc::SRAIW)),
                };
                if const_args[2] {
                    self.a.i(io, d, a1, (args[2] & (w as u64 - 1)) as i64);
                } else {
                    self.a.r(ro, d, a1, r(2));
                }
            }
            Opcode::Rotl | Opcode::Rotr => {
                let left = op.opc == Opcode::Rotl;
                if const_args[2] {
                    let n = args[2] as u32 & (w - 1);
                    // A right rotate by this much.
                    let rn = if left { (w - n) & (w - 1) } else { n };
                    if feat.zbb {
                        self.a.i(pick(opc::RORI, opc::RORIW), d, a1, rn as i64);
                    } else if rn == 0 {
                        self.a.mov(d, a1);
                    } else {
                        self.a.i(pick(opc::SRLI, opc::SRLIW), TMP0, a1, rn as i64);
                        self.a.i(pick(opc::SLLI, opc::SLLIW), TMP1, a1, (w - rn) as i64);
                        self.a.r(opc::OR, d, TMP0, TMP1);
                    }
                } else if feat.zbb {
                    let o = match left {
                        true => pick(opc::ROL, opc::ROLW),
                        false => pick(opc::ROR, opc::RORW),
                    };
                    self.a.r(o, d, a1, r(2));
                } else {
                    let (first, second) = match left {
                        true => (pick(opc::SLL, opc::SLLW), pick(opc::SRL, opc::SRLW)),
                        false => (pick(opc::SRL, opc::SRLW), pick(opc::SLL, opc::SLLW)),
                    };
                    self.a.r(first, TMP0, a1, r(2));
                    self.a.r(opc::SUB, TMP1, ZERO, r(2));
                    self.a.r(second, TMP1, a1, TMP1);
                    self.a.r(opc::OR, d, TMP0, TMP1);
                }
            }
            Opcode::Clz | Opcode::Ctz => {
                let clz = op.opc == Opcode::Clz;
                if feat.zbb {
                    let o = match clz {
                        true => pick(opc::CLZ, opc::CLZW),
                        false => pick(opc::CTZ, opc::CTZW),
                    };
                    self.a.i(o, TMP0, a1, 0);
                } else if clz {
                    self.clz_soft(w, a1);
                } else {
                    self.ctz_soft(w, a1);
                }
                let b =
                    if const_args[2] { (norm(ty, args[2]), true) } else { (r(2) as i64, false) };
                if b.1 && b.0 == w as i64 {
                    self.a.mov(d, TMP0);
                } else {
                    self.select_zero(d, a1, b);
                }
            }
            Opcode::Ctpop => {
                if feat.zbb {
                    self.a.i(pick(opc::CPOP, opc::CPOPW), d, a1, 0);
                } else {
                    if w32 {
                        self.a.ext32u(TMP1, a1);
                    } else {
                        self.a.mov(TMP1, a1);
                    }
                    self.ctpop_soft(d);
                }
            }
            Opcode::Bswap16 | Opcode::Bswap32 | Opcode::Bswap64 => {
                let bytes = match op.opc {
                    Opcode::Bswap16 => 2,
                    Opcode::Bswap32 => 4,
                    _ => 8,
                };
                let os = bytes < 8 && op.args[2] as u32 & bswap::OS != 0;
                // An I32 result is kept sign extended, so a 32-bit swap always is.
                let sext = os || (w32 && bytes == 4);
                if feat.zbb {
                    if bytes == 8 {
                        self.a.i(opc::REV8, d, a1, 0);
                    } else {
                        self.a.i(opc::REV8, TMP0, a1, 0);
                        let o = if sext { opc::SRAI } else { opc::SRLI };
                        self.a.i(o, d, TMP0, (64 - 8 * bytes) as i64);
                    }
                } else {
                    self.bswap_soft(bytes, a1, sext);
                    self.a.mov(d, TMP0);
                }
            }
            Opcode::Deposit => {
                let (ofs, len) = (op.args[3] as u32, op.args[4] as u32);
                if len == 0 || ofs + len > w {
                    return Err(bad(op, "field out of range"));
                }
                // The output shares the register of the first input. The field, shifted into
                // place, with an I32 field at the top sign extended to 64 bits.
                let top32 = w32 && ofs + len == 32;
                let b = r(2);
                self.a.i(opc::SLLI, TMP0, b, (64 - len) as i64);
                let o = if top32 { opc::SRAI } else { opc::SRLI };
                self.a.i(o, TMP0, TMP0, (64 - len - ofs) as i64);
                let keep = if top32 { field_mask(ofs) } else { !(field_mask(len) << ofs) };
                let keep = keep as i64;
                if is_imm12(keep) {
                    self.a.i(opc::ANDI, d, a1, keep);
                } else if feat.zbb && is_imm12(!keep) {
                    self.a.i(opc::ADDI, TMP1, ZERO, !keep);
                    self.a.r(opc::ANDN, d, a1, TMP1);
                } else {
                    self.a.movi(Type::I64, TMP1, keep);
                    self.a.r(opc::AND, d, a1, TMP1);
                }
                self.a.r(opc::OR, d, d, TMP0);
            }
            Opcode::Extract | Opcode::Sextract => {
                let (ofs, len) = (op.args[2] as u32, op.args[3] as u32);
                if len == 0 || ofs + len > w {
                    return Err(bad(op, "field out of range"));
                }
                let signed = op.opc == Opcode::Sextract;
                match (signed, ofs, len) {
                    (_, 0, 32) if w32 => self.a.mov(d, a1),
                    (false, 0, 8) => self.a.ext8u(d, a1),
                    (false, 0, 16) => self.a.ext16u(d, a1),
                    (false, 0, 32) => self.a.ext32u(d, a1),
                    (true, 0, 8) => self.a.ext8s(d, a1),
                    (true, 0, 16) => self.a.ext16s(d, a1),
                    (true, 0, 32) => self.a.ext32s(d, a1),
                    (false, _, 1) if feat.zbs => self.a.i(opc::BEXTI, d, a1, ofs as i64),
                    _ => {
                        self.a.i(opc::SLLI, TMP0, a1, (64 - ofs - len) as i64);
                        let o = if signed { opc::SRAI } else { opc::SRLI };
                        self.a.i(o, d, TMP0, (64 - len) as i64);
                    }
                }
            }
            Opcode::Extract2 => {
                let ofs = op.args[3] as u32;
                if ofs >= w {
                    return Err(bad(op, "shift out of range"));
                }
                let (lo, hi) = (r(1), r(2));
                if ofs == 0 {
                    self.a.mov(d, lo);
                } else {
                    self.a.i(pick(opc::SRLI, opc::SRLIW), TMP0, lo, ofs as i64);
                    self.a.i(pick(opc::SLLI, opc::SLLIW), TMP1, hi, (w - ofs) as i64);
                    self.a.r(opc::OR, d, TMP0, TMP1);
                }
            }
            Opcode::Mul => self.a.r(pick(opc::MUL, opc::MULW), d, a1, r(2)),
            Opcode::Muluh | Opcode::Mulsh => {
                let signed = op.opc == Opcode::Mulsh;
                if !w32 {
                    self.a.r(if signed { opc::MULH } else { opc::MULHU }, d, a1, r(2));
                } else if signed {
                    self.a.r(opc::MUL, TMP0, a1, r(2));
                    self.a.i(opc::SRAI, d, TMP0, 32);
                } else {
                    self.a.ext32u(TMP0, a1);
                    self.a.ext32u(TMP1, r(2));
                    self.a.r(opc::MUL, TMP0, TMP0, TMP1);
                    self.a.i(opc::SRAI, d, TMP0, 32);
                }
            }
            Opcode::Mulu2 | Opcode::Muls2 => {
                let signed = op.opc == Opcode::Muls2;
                let (lo, hi, a, b) = (r(0), r(1), r(2), r(3));
                if !w32 {
                    self.a.r(if signed { opc::MULH } else { opc::MULHU }, TMP0, a, b);
                    self.a.r(opc::MUL, lo, a, b);
                    self.a.mov(hi, TMP0);
                } else {
                    if signed {
                        self.a.r(opc::MUL, TMP0, a, b);
                    } else {
                        self.a.ext32u(TMP0, a);
                        self.a.ext32u(TMP1, b);
                        self.a.r(opc::MUL, TMP0, TMP0, TMP1);
                    }
                    self.a.ext32s(lo, TMP0);
                    self.a.i(opc::SRAI, hi, TMP0, 32);
                }
            }
            Opcode::Divs | Opcode::Divu | Opcode::Rems | Opcode::Remu => {
                let o = match op.opc {
                    Opcode::Divs => pick(opc::DIV, opc::DIVW),
                    Opcode::Divu => pick(opc::DIVU, opc::DIVUW),
                    Opcode::Rems => pick(opc::REM, opc::REMW),
                    _ => pick(opc::REMU, opc::REMUW),
                };
                // A zero divisor divides by one: b | (b == 0).
                self.a.i(opc::SLTIU, TMP1, r(2), 1);
                self.a.r(opc::OR, TMP1, TMP1, r(2));
                self.a.r(o, d, a1, TMP1);
            }
            Opcode::Divs2 | Opcode::Divu2 => {
                self.put_args(&args[2..5]);
                if w32 {
                    // The service routine takes I32 operands zero extended.
                    for k in 0..3 {
                        self.a.s(opc::SW, CTX, ZERO, 8 * k + 4);
                    }
                }
                self.service(Request::Div2 { signed: op.opc == Opcode::Divs2, bits: w });
                self.a.ld(ty, r(0), CTX, 0);
                self.a.ld(ty, r(1), CTX, 8);
            }
            Opcode::Setcond | Opcode::Negsetcond => {
                let c = cond_arg(op, 3)?;
                let neg = op.opc == Opcode::Negsetcond;
                let a2 = if const_args[2] { norm(ty, args[2]) } else { args[2] as i64 };
                match c {
                    Cond::Never => self.a.movi(Type::I64, d, 0),
                    Cond::Always => self.a.movi(Type::I64, d, if neg { -1 } else { 1 }),
                    _ if neg => self.negsetcond(c, d, a1, a2, const_args[2]),
                    _ => self.setcond(c, d, a1, a2, const_args[2]),
                }
            }
            Opcode::Movcond => {
                let c = cond_arg(op, 5)?;
                let v = |k: usize| {
                    if const_args[k] { (norm(ty, args[k]), true) } else { (args[k] as i64, false) }
                };
                match c {
                    Cond::Never => self.movc(d, v(4)),
                    Cond::Always => self.movc(d, v(3)),
                    _ => self.movcond(c, d, a1, v(2), v(3), v(4)),
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
                let (insn, len) = match op.opc {
                    Opcode::Ld8u => (opc::LBU, 1),
                    Opcode::Ld8s => (opc::LB, 1),
                    Opcode::Ld16u => (opc::LHU, 2),
                    Opcode::Ld16s => (opc::LH, 2),
                    Opcode::Ld32u if !w32 => (opc::LWU, 4),
                    Opcode::Ld32u | Opcode::Ld32s => (opc::LW, 4),
                    _ => (pick(opc::LD, opc::LW), w as u64 / 8),
                };
                let addr = self.host_addr(f, op, args, const_args, 1, len);
                match addr {
                    Addr::Static(off) if len == 4 && Some(off) == self.icount_decr => {
                        // `icount_decr` is read where other threads set it.
                        self.a.ld(Type::I64, TMP1, CTX, DECR_OFFSET);
                        self.a.ldst(insn, d, TMP1, 0);
                    }
                    _ => self.host_access(addr, insn, d),
                }
            }
            Opcode::St8 | Opcode::St16 | Opcode::St32 | Opcode::St => {
                let (insn, len) = match op.opc {
                    Opcode::St8 => (opc::SB, 1),
                    Opcode::St16 => (opc::SH, 2),
                    Opcode::St32 => (opc::SW, 4),
                    _ => (pick(opc::SD, opc::SW), w as u64 / 8),
                };
                let addr = self.host_addr(f, op, args, const_args, 1, len);
                self.host_access(addr, insn, r(0));
            }
            Opcode::QemuLd | Opcode::QemuLd2 => {
                let two = op.opc == Opcode::QemuLd2;
                let ai = if two { 2 } else { 1 };
                let oi = MemOpIdx(op.args[ai + 1] as u32);
                let acquire = op.flags & ldst_flags::ACQUIRE_PC != 0;
                let sp = SlowPath {
                    slow: self.a.new_label(),
                    done: self.a.new_label(),
                    insn: self.insn,
                    store: false,
                    two,
                    ty,
                    data: [r(0), if two { r(1) } else { 0 }],
                    addr: r(ai),
                    oi,
                };
                let inline = if two && self.tlb_pair_fits(oi) {
                    self.tlb_addr(r(ai), oi, false, sp.slow);
                    self.a.ld(Type::I64, r(0), TMP0, 0);
                    self.a.ld(Type::I64, r(1), TMP0, 8);
                    true
                } else if !two && self.tlb_fits(oi) {
                    self.tlb_addr(r(ai), oi, false, sp.slow);
                    let m = oi.memop();
                    let insn = match m.size() {
                        0 if m.is_signed() => opc::LB,
                        0 => opc::LBU,
                        1 if m.is_signed() => opc::LH,
                        1 => opc::LHU,
                        2 if m.is_signed() || w32 => opc::LW,
                        2 => opc::LWU,
                        _ => opc::LD,
                    };
                    self.a.ldst(insn, r(0), TMP0, 0);
                    true
                } else {
                    false
                };
                self.finish_ldst(sp, inline);
                if acquire {
                    self.a.mb(mo::LD_LD | mo::LD_ST);
                }
            }
            Opcode::QemuSt | Opcode::QemuSt2 => {
                let two = op.opc == Opcode::QemuSt2;
                let ai = if two { 2 } else { 1 };
                let oi = MemOpIdx(op.args[ai + 1] as u32);
                if op.flags & ldst_flags::RELEASE != 0 {
                    self.a.mb(mo::LD_ST | mo::ST_ST);
                }
                let sp = SlowPath {
                    slow: self.a.new_label(),
                    done: self.a.new_label(),
                    insn: self.insn,
                    store: true,
                    two,
                    ty,
                    data: [r(0), if two { r(1) } else { ZERO }],
                    addr: r(ai),
                    oi,
                };
                let inline = if two && self.tlb_pair_fits(oi) {
                    self.tlb_addr(r(ai), oi, true, sp.slow);
                    self.a.s(opc::SD, TMP0, r(0), 0);
                    self.a.s(opc::SD, TMP0, r(1), 8);
                    true
                } else if !two && self.tlb_fits(oi) {
                    self.tlb_addr(r(ai), oi, true, sp.slow);
                    let insn = [opc::SB, opc::SH, opc::SW, opc::SD][oi.memop().size() as usize];
                    self.a.s(insn, TMP0, r(0), 0);
                    true
                } else {
                    false
                };
                self.finish_ldst(sp, inline);
            }
            Opcode::GotoPtr => {
                // Jump straight to the block when the service routine vouched for the address
                // (`lookup_tb_ptr` found a block it can enter); leave otherwise. The context
                // holds 0 when no address is vouched for, so 0 itself always leaves.
                let out = self.a.new_label();
                self.a.b_label(opc::BEQ, r(0), ZERO, out);
                self.a.ld(Type::I64, TMP0, CTX, GOTO_PTR_OK_OFFSET);
                self.a.b_label(opc::BNE, r(0), TMP0, out);
                self.a.jr(r(0));
                self.a.bind(out);
                self.a.mov(A1, r(0));
                self.exit_with(kind::GOTO_PTR, None);
            }
            other => return Err(GenCodeError::Unsupported(other.name().to_string())),
        }
        Ok(())
    }

    /// `d = v`, a register or a constant.
    fn movc(&mut self, d: Reg, v: (i64, bool)) {
        if v.1 {
            self.a.movi(Type::I64, d, v.0);
        } else {
            self.a.mov(d, v.0 as Reg);
        }
    }

    /// End a guest access: with an `inline` hit path, the hit falls through and the miss path
    /// waits for the end of the block; without one, the miss path is all there is.
    fn finish_ldst(&mut self, sp: SlowPath, inline: bool) {
        if inline {
            self.a.bind(sp.done);
            self.slow_paths.push(sp);
        } else {
            self.ldst_slow(&sp);
            self.a.bind(sp.done);
        }
    }

    /// The guest address in `addr` as a 64-bit value: zero extended into TMP3 for 32-bit
    /// guests, whose addresses are I32 values and so sign extended in registers.
    fn guest_addr(&mut self, addr: Reg) -> Reg {
        if self.addr32 {
            self.a.ext32u(TMP3, addr);
            TMP3
        } else {
            addr
        }
    }

    /// The service routine call of a guest access, `tcg_out_qemu_ld_slow_path` and
    /// `tcg_out_qemu_st_slow_path`.
    fn ldst_slow(&mut self, sp: &SlowPath) {
        if sp.store {
            self.put_args(&[u64::from(sp.data[0]), u64::from(sp.data[1])]);
            let addr = self.guest_addr(sp.addr);
            self.a.st(Type::I64, addr, CTX, 16);
            self.service(Request::Store(sp.oi));
            return;
        }
        let addr = self.guest_addr(sp.addr);
        self.a.st(Type::I64, addr, CTX, 0);
        self.service(Request::Load(sp.oi));
        if sp.two {
            self.a.ld(Type::I64, sp.data[0], CTX, 0);
            self.a.ld(Type::I64, sp.data[1], CTX, 8);
        } else {
            self.a.ld(sp.ty, sp.data[0], CTX, 0);
        }
    }

    /// True if a guest access with `oi` can be looked up in the TLB inline: there is a TLB, and
    /// the access is at most 64 bits, needs no byte swap, and needs no more low address bits
    /// clear than the comparators keep free of flags.
    fn tlb_fits(&self, oi: MemOpIdx) -> bool {
        let m = oi.memop();
        self.tlb_page_bits.is_some()
            && m.0 & MemOp::BSWAP.0 == 0
            && m.size() <= 3
            && m.alignment_bits() <= TLB_FLAGS_SHIFT
            && (oi.mmu_idx() as usize) < TLB_MAX_MMU_MODES
    }

    /// True if a 128-bit guest access with `oi` can be looked up in the TLB inline and done as
    /// two 64-bit halves: there is a TLB, the access needs no byte swap, and its halves need
    /// only be atomic each (`MO_ATOM_IFALIGN_PAIR`) or not at all. A 128-bit access that must
    /// be atomic as a whole takes the slow path.
    fn tlb_pair_fits(&self, oi: MemOpIdx) -> bool {
        let m = oi.memop();
        let atom = m.0 & MemOp::ATOM_MASK.0;
        self.tlb_page_bits.is_some()
            && m.0 & MemOp::BSWAP.0 == 0
            && m.size() == 4
            && (atom == MemOp::ATOM_IFALIGN_PAIR.0 || atom == MemOp::ATOM_NONE.0)
            && m.alignment_bits() <= TLB_FLAGS_SHIFT
            && (oi.mmu_idx() as usize) < TLB_MAX_MMU_MODES
    }

    /// The inline TLB lookup of the guest access at `addr`, QEMU's `prepare_host_addr`: on a
    /// hit leave the host address in TMP0; on a miss branch to `slow`.
    fn tlb_addr(&mut self, addr: Reg, oi: MemOpIdx, store: bool, slow: usize) {
        let page_bits = self.tlb_page_bits.expect("tlb_fits checked there is a TLB");
        let m = oi.memop();
        let s_mask = (1i64 << m.size()) - 1;
        let a_mask = (1i64 << m.alignment_bits()) - 1;
        let src = self.guest_addr(addr);
        // The mask and the table, from the copy of the descriptor in the run context.
        let desc = TLB_OFFSET + (oi.mmu_idx() as usize * TLB_DESC_WORDS * 8) as i64;
        self.a.ld(Type::I64, TMP0, CTX, desc);
        self.a.ld(Type::I64, TMP1, CTX, desc + 8);
        // The entry: table + ((addr >> (page_bits - TLB_ENTRY_BITS)) & mask).
        self.a.i(opc::SRLI, TMP2, src, (page_bits - TLB_ENTRY_BITS) as i64);
        self.a.r(opc::AND, TMP0, TMP0, TMP2);
        self.a.r(opc::ADD, TMP1, TMP1, TMP0);
        self.a.ld(Type::I64, TMP0, TMP1, if store { 8 } else { 0 });
        if a_mask != 0 {
            self.a.i(opc::ANDI, TMP2, src, a_mask);
            self.a.far_b_label(opc::BNE, TMP2, ZERO, slow);
        }
        // An access less aligned than its size must not cross the page: compare the page of
        // its last byte, as QEMU does.
        let page_src = if a_mask < s_mask {
            self.a.i(opc::ADDI, TMP2, src, s_mask - a_mask);
            TMP2
        } else {
            src
        };
        self.a.i(opc::SRLI, TMP2, page_src, page_bits as i64);
        self.a.i(opc::SLLI, TMP2, TMP2, page_bits as i64);
        self.a.far_b_label(opc::BNE, TMP2, TMP0, slow);
        self.a.ld(Type::I64, TMP1, TMP1, (TLB_ADDEND_WORD * 8) as i64);
        self.a.r(opc::ADD, TMP0, TMP1, src);
    }
}

impl Target for Gen<'_> {
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
        Some(ZERO)
    }

    fn op_constraints(&self, _f: &Func, op: &Op) -> R<&'static [&'static str]> {
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
            Opcode::Add | Opcode::And | Opcode::Or | Opcode::Xor => C_R_R_RI_I,
            Opcode::Sub => C_R_RZ_RJ,
            Opcode::Andc
            | Opcode::Orc
            | Opcode::Eqv
            | Opcode::Nand
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
            Opcode::Setcond | Opcode::Negsetcond => C_R_R_RI_I,
            Opcode::Brcond => C_RZ_RZ,
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
            other => return Err(GenCodeError::Unsupported(other.name().to_string())),
        })
    }

    fn constraint_letter(&self, c: char) -> Option<Letter> {
        Some(match c {
            'r' => Letter::Regs(GPRS),
            'I' => Letter::Const(ctc::S12),
            'J' => Letter::Const(ctc::N12),
            'M' => Letter::Const(ctc::M12),
            _ => return None,
        })
    }

    fn const_match(&self, val: i64, ct: u32, ty: Type, _cond: Cond, _vece: u32) -> bool {
        if ct & regalloc::ct::CONST != 0 {
            return true;
        }
        let val = if ty == Type::I32 { val as i32 as i64 } else { val };
        if ct & ctc::S12 != 0 && is_imm12(val) {
            return true;
        }
        if ct & ctc::N12 != 0 && is_imm12(val.wrapping_neg()) && val != i64::MIN {
            return true;
        }
        ct & ctc::M12 != 0 && (-0x7ff..=0x7ff).contains(&val)
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

    fn out_mov(&mut self, _ty: Type, dst: Reg, src: Reg) -> bool {
        self.a.mov(dst, src);
        true
    }

    fn out_movi(&mut self, ty: Type, dst: Reg, val: i64) {
        self.a.movi(ty, dst, val);
    }

    fn out_dupi_vec(&mut self, _ty: Type, _vece: u32, _dst: Reg, _val: u64) {
        self.err.get_or_insert_with(|| GenCodeError::Unsupported("vector constants".into()));
    }

    fn out_ld(&mut self, ty: Type, dst: Reg, base: Reg, off: i64) {
        self.a.ld(ty, dst, base, off);
    }

    fn out_st(&mut self, ty: Type, src: Reg, base: Reg, off: i64) {
        self.a.st(ty, src, base, off);
    }

    fn out_sti(&mut self, ty: Type, val: i64, base: Reg, off: i64) -> bool {
        if !ty.is_int() {
            return false;
        }
        if val == 0 {
            self.a.st(ty, ZERO, base, off);
        } else {
            self.a.movi(ty, TMP0, val);
            self.a.st(ty, TMP0, base, off);
        }
        true
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
                self.a.j_label(l);
            }
            Opcode::Mb => self.a.mb(op.args[0] as u32),
            Opcode::InsnStart => {
                let mut words = [0u64; INSN_START_WORDS];
                words.copy_from_slice(&op.args[..INSN_START_WORDS]);
                self.requests.push(Request::InsnStart(words));
                self.insn = self.requests.len() as u64;
                self.insn_of.push(self.insn);
            }
            Opcode::ExitTb => self.exit_with(kind::EXIT_TB, Some(op.args[0])),
            Opcode::GotoTb => {
                let at = self.a.pos();
                self.a.emit(opc::NOP);
                self.goto_tb.push((op.args[0] as u32, at, self.insn));
            }
            Opcode::Brcond => {
                let c = cond_arg(op, 2)?;
                let l = self.label(op, 3)?;
                match c {
                    Cond::Never => {}
                    Cond::Always => self.a.j_label(l),
                    _ => self.brcond(c, args[0] as Reg, args[1] as Reg, l),
                }
            }
            _ => self.out_scalar(f, id, op, args, const_args)?,
        }
        Ok(())
    }

    fn call_arg_home(&self, idx: usize) -> (Reg, i64) {
        (CTX, 8 * idx as i64)
    }

    fn out_of_line_branches(&self) -> bool {
        true
    }

    fn out_call(&mut self, f: &Func, op: &Op) -> R<()> {
        let info = f.helper_info(op.call_helper()).clone();
        let ni = op.calli as usize;
        // The argument words of I32 inputs hold them sign extended, as registers do; helpers
        // see them zero extended, as in the interpreter.
        for k in 0..ni {
            if f.temp(op.arg_temp(op.callo as usize + k)).ty == Type::I32 {
                self.a.s(opc::SW, CTX, ZERO, 8 * k as i64 + 4);
            }
        }
        let pure = info.flags & call_flags::NO_SIDE_EFFECTS != 0;
        // A helper that may have side effects may access guest memory: order it as a guest
        // access with the fence mapping, `fence rw,w` before and `fence r,rw` after.
        let fence = self.mapping != FenceMapping::Qemu && !pure;
        if fence {
            self.a.mb(mo::LD_ST | mo::ST_ST);
        }
        let ic = self.lookup != 0 && info.name == crate::runtime::LOOKUP_TB_PTR_IC && ni == 2;
        let lookup = ic || self.lookup != 0 && info.name == crate::runtime::LOOKUP_TB_PTR;
        let native = match self.helpers {
            Some(h) if pure && !lookup => h
                .get(&info.name)
                .filter(|e| e.ret == info.ret && e.args == info.args)
                .and_then(|_| h.native(&info.name)),
            _ => None,
        };
        if let Some(nf) = native {
            self.call_native(nf as usize as u64, ni, info.ret);
            return Ok(());
        }
        let req = Request::Call { name: info.name, ret: info.ret, args: info.args, nin: ni, pure };
        let after = fence.then_some(opc::FENCE | asm::fence::LD_LD | asm::fence::LD_ST);
        if ic {
            self.ic_probe();
        }
        if lookup {
            let site = if ic { crate::runtime::LOOKUP_IC_SITE } else { 0 };
            self.service_via(req, after, self.lookup, self.insn, site);
        } else {
            self.service_via(req, after, self.service, self.insn, 0);
        }
        Ok(())
    }
}
