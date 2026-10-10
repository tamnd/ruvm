// SPDX-License-Identifier: GPL-2.0-or-later

//! Instruction selection and register allocation: one finished [`Func`] in, x86-64 code out.
//! This is QEMU's `tcg/x86_64/tcg-target.c.inc` (the constraints of `tcg-target-con-set.h` and
//! `tcg-target-con-str.h`, `tcg_out_op`, `tcg_out_vec_op` and the `tcg_out_*` hooks) plugged
//! into the generic allocator of [`ruvm_jit_core::regalloc`], which drives it the way
//! `tcg_gen_code` does.
//!
//! Register use:
//!
//! - rbp is the address of the CPU state buffer (`TCG_AREG0`), r15 the run context and r14 the
//!   slot array; all three are fixed for the whole block;
//! - rax, rdx, rbx, rsi, rdi, r8, r9, r12 and r13 hold temps, callee-saved ones first in the
//!   allocation order;
//! - r11, r10 and rcx are scratch, rcx because variable shifts take their count in cl;
//! - xmm0 to xmm11 hold vector temps, xmm12 to xmm15 are vector scratch.
//!
//! Globals live in the CPU state at their offsets, TB and EBB temps in a slot array, and the
//! allocator moves them into registers and back as `op.life` says.
//!
//! Host pointers are offsets into the CPU state buffer, as in the interpreter: `env` is 0, and a
//! pointer global holds an offset. Every access through such a pointer is bounds checked against
//! the length in the run context and leaves the block with
//! [`ruvm_jit_interp::InterpError::EnvOutOfBounds`] when it would fall outside the buffer.
//! Accesses at constant offsets from `env` are checked once, on entry, against the furthest one
//! in the block.
//!
//! Whatever needs Rust (helper calls, `qemu_ld` and `qemu_st`, the 128 by 64 bit divisions) is
//! a call to one service routine with the index of a [`Request`]; operands go through the
//! argument words of the run context.
//!
//! Optional extensions are used as [`HostFeatures`] allows: BMI1 `andn`, BMI2 `shlx`, `shrx`
//! and `sarx`, LZCNT and TZCNT, POPCNT, SSSE3 to SSE4.2 vector ops, VEX encoding with AVX and
//! 256-bit vectors with AVX2. Each has a fallback, as in QEMU.
//!
//! Differences from QEMU:
//!
//! - `env` is the constant 0, not a register, and every access through a pointer is bounds
//!   checked as described above. Host loads and stores take a constant base (`ri`) so that
//!   accesses through `env` use the static check.
//! - Loads and stores through a pointer that is not `env` can fault, so the allocator syncs
//!   globals before them, as it does for ops with side effects.
//! - Helper calls, guest memory accesses and `divs2`/`divu2` go through the service routine,
//!   with every argument in memory, instead of the host calling convention. Only the TLB miss
//!   path of a `qemu_ld` or `qemu_st` does: with [`GenOptions::tlb_page_bits`] the lookup is
//!   inlined as in QEMU, except for byte swapped accesses, which always take the slow path
//!   (QEMU inlines those too). A 128-bit access is inlined as two 64-bit host accesses when its
//!   halves need only be atomic each (`MO_ATOM_IFALIGN_PAIR`, such as an aarch64 `ldp` or
//!   `stp` of two X registers) or not at all. One that must be atomic as a whole when aligned
//!   (`MO_ATOM_IFALIGN`, such as an x86 `movdqu` on a CPU with AVX) is inlined as one
//!   `vmovdqa` or `vmovdqu` through `VT0` when the host makes those atomic, as QEMU does, and
//!   takes the slow path otherwise.
//! - `insn_start` emits no code. Each service request carries the index of the `insn_start`
//!   of its instruction, fixed when the block is compiled, and each exit stores it in the run
//!   context, instead of QEMU's table of host code offsets next to the code; the runtime
//!   reports the words to the guest memory before each service request and to the caller at
//!   the end.
//! - rcx is never allocated, so the shift ops take any register for the count (QEMU's `c`
//!   constraint is not used) and the code moves the count into cl itself.
//! - `div`, `rem`, `muluh` and `mulsh` take any registers and shuffle rax and rdx through the
//!   scratch registers, instead of QEMU's expansion into `div2` and `mul2` with fixed
//!   registers. A zero divisor divides by one and the most negative value divided by -1 gives
//!   itself, as in the interpreter, instead of raising a host exception.
//! - `clz`, `ctz` and `ctpop` without LZCNT, TZCNT or POPCNT, and the I32 forms of `mulsh` and
//!   `muluh`, are expanded inline.
//! - Vector ops that QEMU's backend leaves to the generic expanders (64-bit element multiply,
//!   min, max and arithmetic shifts without AVX-512, saturating arithmetic on 32 and 64-bit
//!   elements, byte shifts and multiplies) are done here, either with a short SIMD sequence or
//!   one element at a time in general registers through two scratch slots.
//! - Without AVX, three operand vector ops are done with legacy SSE encodings and a copy, the
//!   way QEMU does it on hosts without AVX for 64 and 128-bit vectors.
//! - The add and subtract with carry ops keep the carry in a word of the slot array, and only
//!   pass it in the flags between two adjacent ops of the same family.
//! - `goto_tb` is a `jmp` whose displacement is 0 (fall through) until the block is linked.
//!   Linking patches it to jump straight to the next block, as in QEMU, when that block is in
//!   a region this one keeps mapped and within reach of a 32-bit displacement, and to an exit
//!   stub that leaves with [`ruvm_jit_interp::Exit::GotoTb`] otherwise.
//! - Every block has its own prologue and epilogue, with the same frame, and chained jumps
//!   enter a block after its prologue. QEMU shares one prologue for the whole buffer. The
//!   first thing after the prologue stores the address of the block's request table in the
//!   run context, so the service routine knows which block a request comes from.
//! - `goto_ptr` jumps to the address `lookup_tb_ptr` returned only if the runtime vouched for
//!   it in the run context, and leaves with [`ruvm_jit_interp::Exit::GotoPtr`] otherwise.
//!   QEMU jumps to whatever the helper returned.
//! - A call to `lookup_tb_ptr_ic` first looks the guest program counter up in an inline cache
//!   of block headers and jumps straight to the block on a hit; see the runtime. Not in QEMU.
//! - A 32-bit load of the `icount_decr` word at the offset the runtime gives reads the shared
//!   atomic through a pointer in the run context, so that exit requests from other threads
//!   are seen without leaving generated code. In QEMU the word is part of the CPU state.

use ruvm_jit_core::ir::{Func, HelperType, Op, OpId, Temp};
use ruvm_jit_core::opcode::Opcode;
use ruvm_jit_core::regalloc::{self, Letter, RegSet, Target};
use ruvm_jit_core::types::{
    Cond, INSN_START_WORDS, MemOp, MemOpIdx, TempKind, Type, bswap, dup_const, mo, opf,
};
use ruvm_jit_interp::HelperRegistry;
use ruvm_jit_interp::fast_tlb::{
    TLB_ADDEND_WORD, TLB_DESC_WORDS, TLB_ENTRY_BITS, TLB_FLAGS_SHIFT, TLB_MAX_MMU_MODES,
};

use crate::asm::{
    Asm, AsmError, Mem, P_DATA16, P_REXB_R, P_REXB_RM, P_REXW, P_VEXL, R8, R9, R10, R11, R12, R13,
    R14, R15, RAX, RBP, RBX, RCX, RDI, RDX, RSI, RSP, Reg, arith, cc, cond_code, ext3, op, shift,
    xmm,
};
use crate::features::HostFeatures;

/// Base of the CPU state buffer.
const ENV: Reg = RBP;
/// The run context.
const CTX: Reg = R15;
/// The slot array.
const SLOTS: Reg = R14;

/// Scratch registers.
const TMP0: Reg = R11;
const TMP1: Reg = R10;
const TMP2: Reg = RCX;

/// Vector scratch registers.
const VT0: Reg = xmm(15);
const VT1: Reg = xmm(14);
const VT2: Reg = xmm(13);
const VSWAP: Reg = xmm(12);

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
    /// Calls to helpers with a [`ruvm_jit_interp::NativeHelperFn`] here, and the declared signature, go straight
    /// to it rather than through the service routine.
    pub(crate) helpers: Option<&'a HelperRegistry>,
}

/// Whether the host uses the Win64 calling convention rather than System V.
const WIN64: bool = cfg!(windows);

/// How generated code left, in rax at the epilogue.
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
            GenCodeError::Unsupported(s) => write!(f, "not supported by the x86_64 backend: {s}"),
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
    /// The byte offset where another block's chained jump enters, after the prologue.
    pub(crate) body: usize,
    /// The byte offset after the static bounds check, where a chained jump may enter when the
    /// run's CPU state is known to be at least [`Generated::env_need`] bytes long.
    pub(crate) fast_body: usize,
    /// The length of CPU state the static bounds check asks for (`u64::MAX` if it always fails).
    pub(crate) env_need: u64,
    /// For each `goto_tb`: its slot, the byte offset of its 4-byte aligned jump displacement,
    /// and the displacement that sends it to the exit stub.
    pub(crate) goto_tb: Vec<(u32, usize, u32)>,
    /// For each `lookup_tb_ptr_ic` call with an inline cache: the index of its request and the
    /// byte offset of the 8-byte address of its [`IC_WAYS`] cache words, which is 0 until the
    /// caller patches it.
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

const fn bit(r: Reg) -> u64 {
    1 << r
}

/// The general registers the allocator may use.
const GPRS: RegSet = RegSet(
    bit(RAX) | bit(RDX) | bit(RBX) | bit(RSI) | bit(RDI) | bit(R8) | bit(R9) | bit(R12) | bit(R13),
);
/// The vector registers the allocator may use: xmm0 to xmm11.
const VECS: RegSet = RegSet(0xfff << 16);
/// `tcg_target_reg_alloc_order`: callee-saved registers first, so values survive calls.
const ALLOC_ORDER: [Reg; 21] = [
    RBX,
    R12,
    R13,
    R9,
    R8,
    RDX,
    RSI,
    RDI,
    RAX,
    xmm(0),
    xmm(1),
    xmm(2),
    xmm(3),
    xmm(4),
    xmm(5),
    xmm(6),
    xmm(7),
    xmm(8),
    xmm(9),
    xmm(10),
    xmm(11),
];
/// The registers a call to Rust may change. Every vector register counts, because the code
/// clears the upper halves before the call and Win64 only preserves the low halves.
const CALL_CLOBBER: RegSet = RegSet(
    bit(RAX)
        | bit(RCX)
        | bit(RDX)
        | bit(R8)
        | bit(R9)
        | bit(R10)
        | bit(R11)
        | if WIN64 { 0 } else { bit(RSI) | bit(RDI) }
        | 0xffff << 16,
);
/// Never allocated: scratch, the stack pointer, the fixed registers and the vector scratch.
const RESERVED: RegSet = RegSet(!(GPRS.0 | VECS.0) & 0xffff_ffff);

/// Target constant classes, `TCG_CT_CONST_*`.
mod ctc {
    /// A sign extended 32-bit immediate, `TCG_CT_CONST_S32`.
    pub(super) const S32: u32 = 0x100;
    /// A zero extended 32-bit immediate, `TCG_CT_CONST_U32`.
    pub(super) const U32: u32 = 0x200;
    /// The operation width in bits, `TCG_CT_CONST_WSZ`.
    pub(super) const WSZ: u32 = 0x800;
    /// A test immediate, `TCG_CT_CONST_TST`.
    pub(super) const TST: u32 = 0x1000;
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
        | Opcode::DupmVec
        | Opcode::St8
        | Opcode::St16
        | Opcode::St32
        | Opcode::St
        | Opcode::StVec => may_fault(f, op, 1),
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

/// The size of the stack frame below the pushed registers: alignment padding, and on Win64 the
/// 32-byte shadow area plus room to save xmm6 to xmm15.
const FRAME: i32 = if WIN64 { 8 + 32 + 160 } else { 8 };

/// The callee-saved registers the block pushes, in order.
const SAVED: &[Reg] =
    if WIN64 { &[RBP, RBX, R12, R13, R14, R15, RDI, RSI] } else { &[RBP, RBX, R12, R13, R14, R15] };

/// The integer argument registers of the host convention.
const ARGS: [Reg; 3] = if WIN64 { [RCX, RDX, R8] } else { [RDI, RSI, RDX] };

/// Compile `f` for code that will live at `base`. `service` is the address of the service
/// routine.
pub(crate) fn generate(
    f: &Func,
    _base: u64,
    service: u64,
    feat: HostFeatures,
    opts: &GenOptions<'_>,
) -> R<Generated> {
    let v256 = check_types(f, feat)?;
    // Liveness goes alongside `f` rather than into a copy of it, unless it has indirect
    // globals to lower.
    let (prepared, live) = regalloc::prepare_live(f, &extra_flags);
    let f: &Func = &prepared;
    let mut g = Gen {
        a: Asm::new(),
        feat,
        labels: vec![None; f.nb_labels()],
        requests: Vec::new(),
        insn_of: Vec::new(),
        insn: 0,
        meta: opts.meta,
        v256,
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
        carry_live: false,
        err: None,
        tlb_page_bits: opts.tlb_page_bits,
        ldst: Vec::new(),
        ic_sites: Vec::new(),
        helpers: opts.helpers,
    };
    g.exit = g.a.new_label();
    g.bounds = g.a.new_label();
    let static_fail = g.a.new_label();

    // Prologue: the callee-saved registers, the frame, and the fixed registers.
    for &r in SAVED {
        g.a.push(r);
    }
    g.a.arithi(arith::SUB, P_REXW, RSP, FRAME as i64);
    if WIN64 {
        for k in 0..10 {
            g.vst(Type::V128, xmm(6 + k), Mem::Base(RSP, 32 + 16 * k as i32));
        }
    }
    g.a.mov(P_REXW, ENV, ARGS[0]);
    g.a.mov(P_REXW, CTX, ARGS[1]);
    g.a.mov(P_REXW, SLOTS, ARGS[2]);
    // Chained jumps from other blocks enter here, with the fixed registers set up.
    let body = g.a.pos();
    // The static bounds check; the immediate is patched once the body is known.
    let check_at = g.a.pos() + 2;
    g.a.movabs(TMP0, 0);
    g.a.cmp_mem(P_REXW, TMP0, Mem::Base(CTX, ENV_LEN_OFFSET as i32));
    g.a.jump(Some(cc::A), static_fail, false);
    let fast_body = g.a.pos();

    regalloc::reg_alloc_live(f, live, &mut g)?;
    if let Some(e) = g.err.take() {
        return Err(e);
    }
    g.exit_with(kind::FELL_OFF, None);

    // The TLB miss paths.
    for l in std::mem::take(&mut g.ldst) {
        g.a.bind(l.label);
        g.insn = l.insn;
        match (l.val, l.hi) {
            (None, None) => g.qemu_ld_slow(l.ty, l.oi, l.out, l.addr),
            (Some(v), None) => g.qemu_st_slow(l.ty, l.oi, v, l.addr),
            (None, Some((hi, _))) => g.qemu_ld2_slow(l.oi, l.out, hi as Reg, l.addr),
            (Some(v), Some(hi)) => g.qemu_st2_slow(l.oi, [v, hi], l.addr),
        }
        g.a.jump(None, l.back, false);
    }

    // Exit stubs for linked goto_tb slots.
    let mut goto_tb = Vec::new();
    for (slot, at, insn) in std::mem::take(&mut g.goto_tb) {
        let stub = g.a.pos();
        g.insn = insn;
        g.note_exit();
        g.a.movi(false, RDX, slot as u64, false);
        g.a.movi(false, RAX, kind::GOTO_TB, false);
        g.a.jump(None, g.exit, false);
        let disp = (stub as i64 - (at as i64 + 4)) as u32;
        goto_tb.push((slot, at, disp));
    }

    // A failed static check reports the access that reaches furthest.
    g.a.bind(static_fail);
    g.a.movabs(TMP0, g.static_access.0);
    g.a.movabs(TMP1, g.static_access.0.wrapping_add(g.static_access.1));
    g.a.jump(None, g.bounds, false);

    // A failed bounds check: the offset in r11, its end in r10.
    g.a.bind(g.bounds);
    g.a.movabs(TMP2, g.meta);
    g.a.store(TMP2, Mem::Base(CTX, META_OFFSET as i32), 8);
    g.a.mov(P_REXW, RCX, TMP1);
    g.a.arith(arith::SUB, P_REXW, RCX, TMP0);
    g.a.store(RCX, Mem::Base(CTX, 0), 8);
    g.a.mov(P_REXW, RDX, TMP0);
    g.a.movi(false, RAX, kind::BOUNDS, false);

    // The epilogue: rax is the kind, rdx the return word.
    g.a.bind(g.exit);
    g.a.store(RDX, Mem::Base(CTX, RET_OFFSET as i32), 8);
    if feat.avx {
        g.a.vex_opc(op::VZEROUPPER, 0, 0, 0, 0);
    }
    if WIN64 {
        for k in 0..10 {
            g.vld(Type::V128, xmm(6 + k), Mem::Base(RSP, 32 + 16 * k as i32));
        }
    }
    g.a.arithi(arith::ADD, P_REXW, RSP, FRAME as i64);
    for &r in SAVED.iter().rev() {
        g.a.pop(r);
    }
    g.a.raw(op::RET);

    // A branch to a label that is never set is malformed IR; the interpreter reports it when
    // the branch is taken, a compiler has to report it now.
    for (id, l) in g.labels.iter().enumerate() {
        if let Some(l) = *l {
            if !g.a.is_bound(l) {
                return Err(GenCodeError::BadOp(format!("label $L{id} is not set")));
            }
        }
    }
    let end = if g.static_always_fails { u64::MAX } else { g.static_end };
    g.a.code[check_at..check_at + 8].copy_from_slice(&end.to_le_bytes());

    // The temp slots, the carry word, then two 32-byte lanes of vector scratch.
    let slot_words = (f.nb_temps() + 1) * SLOT_BYTES / 8 + 2 * SLOT_BYTES / 8;
    let (requests, insn_of, ic_sites) = (g.requests, g.insn_of, g.ic_sites);
    let bytes = g.a.finish()?;
    Ok(Generated {
        bytes,
        requests,
        insn_of,
        slot_words,
        body,
        fast_body,
        env_need: end,
        goto_tb,
        ic_sites,
    })
}

/// Refuse temps of types this backend has no registers for, and calls with more arguments
/// than the run context holds. Returns whether `f` uses 256-bit vectors.
fn check_types(f: &Func, feat: HostFeatures) -> R<bool> {
    let mut v256 = false;
    let ok = |ty: Type| match ty {
        Type::I32 | Type::I64 | Type::V64 | Type::V128 => true,
        Type::V256 => feat.avx2,
        _ => false,
    };
    for (_, op) in f.ops() {
        let n = op.nb_oargs() + op.nb_iargs();
        for k in 0..n {
            let ty = f.temp(op.arg_temp(k)).ty;
            v256 |= ty == Type::V256;
            if !ok(ty) {
                return Err(GenCodeError::Unsupported(format!("{}: {ty:?} temps", op.opc.name())));
            }
        }
        if op.opc.def().flags & opf::VECTOR != 0 && !(op.ty.is_vector() && ok(op.ty)) {
            return Err(GenCodeError::Unsupported(format!("{}: {:?}", op.opc.name(), op.ty)));
        }
        if op.opc == Opcode::Call && (op.calli as usize > NARGS || op.callo > 2) {
            return Err(bad(op, "too many call arguments"));
        }
        v256 |= op.ty == Type::V256;
    }
    Ok(v256)
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

/// `P_REXW` for a 64-bit op.
fn rexw(ty: Type) -> u32 {
    if ty == Type::I64 { P_REXW } else { 0 }
}

/// `P_VEXL` for a 256-bit op.
fn vexl(ty: Type) -> u32 {
    if ty == Type::V256 { P_VEXL } else { 0 }
}

/// Whether a 128-bit `qemu_ld` or `qemu_st` must be atomic as more than two halves, so that
/// two 64-bit host accesses cannot make it.
fn atomic16(oi: MemOpIdx) -> bool {
    let atom = oi.memop().0 & MemOp::ATOM_MASK.0;
    atom != MemOp::ATOM_IFALIGN_PAIR.0 && atom != MemOp::ATOM_NONE.0
}

fn is_vec_reg(r: Reg) -> bool {
    r >= 16
}

fn fits_i32(v: i64) -> bool {
    v == v as i32 as i64
}

/// A field of `len` low bits.
fn field_mask(len: u32) -> u64 {
    if len >= 64 { u64::MAX } else { (1 << len) - 1 }
}

/// Where a host memory access goes.
enum Addr {
    /// At this constant offset into the CPU state, covered by the check on entry.
    Static(i64),
    /// At the offset in r11, already bounds checked.
    Dyn,
}

impl Addr {
    fn mem(&self) -> Mem {
        match *self {
            Addr::Static(off) => Mem::Base(ENV, off as i32),
            Addr::Dyn => Mem::Index(ENV, TMP0, 0),
        }
    }
}

/// What the element at a time fallback computes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Lane {
    Mul,
    Ssadd,
    Usadd,
    Sssub,
    Ussub,
    Smin,
    Umin,
    Smax,
    Umax,
    Abs,
    /// Shifts by the count already in cl.
    Shl,
    Shr,
    Sar,
    /// Shifts and rotates by the count in the matching element of the second operand.
    Shlv,
    Shrv,
    Sarv,
    Rotlv,
    Rotrv,
    Cmp(Cond),
}

struct Gen<'h> {
    a: Asm,
    feat: HostFeatures,
    labels: Vec<Option<usize>>,
    requests: Vec<Request>,
    /// See [`Generated::insn_of`].
    insn_of: Vec<u64>,
    /// One more than the index of the last `insn_start` request so far, or 0.
    insn: u64,
    /// See [`GenOptions::meta`].
    meta: u64,
    /// The block uses 256-bit vectors, so it leaves the upper vector state dirty.
    v256: bool,
    /// For each `goto_tb`: its slot, the offset of its displacement, and [`Gen::insn`] there.
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
    /// The carry flag holds a carry between two fused ops, so moves must not change the flags.
    carry_live: bool,
    /// An error from a hook that cannot return one.
    err: Option<GenCodeError>,
    /// See [`GenOptions::tlb_page_bits`].
    tlb_page_bits: Option<u32>,
    /// The TLB miss paths, emitted after the body.
    ldst: Vec<LdstSlow>,
    /// See [`Generated::ic_sites`].
    ic_sites: Vec<(usize, usize)>,
    /// See [`GenOptions::helpers`].
    helpers: Option<&'h HelperRegistry>,
}

/// The out of line TLB miss path of one `qemu_ld` or `qemu_st`, QEMU's `TCGLabelQemuLdst`.
struct LdstSlow {
    /// Where the miss path starts.
    label: usize,
    /// Where it goes back to.
    back: usize,
    oi: MemOpIdx,
    ty: Type,
    addr: Reg,
    /// For a load, the register of the value; for a store, the value and whether it is a
    /// constant.
    val: Option<(u64, bool)>,
    out: Reg,
    /// For a 128-bit access, the high half: the register of a load, or the value of a store
    /// and whether it is a constant. `val` and `out` are then the low half.
    hi: Option<(u64, bool)>,
    /// [`Gen::insn`] at the access.
    insn: u64,
}

// Constraint sets, `tcg-target-con-set.h`.
const C_R: &[&str] = &["r"];
const C_R_R: &[&str] = &["r", "r"];
const C_R_0: &[&str] = &["r", "0"];
const C_R_RI: &[&str] = &["r", "ri"];
const C_RE_RI: &[&str] = &["re", "ri"];
const C_RE_R: &[&str] = &["re", "r"];
const C_R_R_RE: &[&str] = &["r", "r", "re"];
const C_R_0_RE: &[&str] = &["r", "0", "re"];
const C_R_0_REZ: &[&str] = &["r", "0", "reZ"];
const C_R_0_R: &[&str] = &["r", "0", "r"];
const C_R_0_RI: &[&str] = &["r", "0", "ri"];
const C_R_R_RI: &[&str] = &["r", "r", "ri"];
const C_R_R_R: &[&str] = &["r", "r", "r"];
const C_R_R_RW: &[&str] = &["r", "r", "rW"];
const C_R_R_RET: &[&str] = &["r", "r", "reT"];
const C_RE_RE_R: &[&str] = &["re", "re", "r"];
const C_MUL2: &[&str] = &["a", "d", "0", "r"];
const C_R5: &[&str] = &["r", "r", "r", "r", "r"];
const C_R_RET: &[&str] = &["r", "reT"];
const C_MOVCOND: &[&str] = &["r", "r", "reT", "r", "0"];
const C_NONE: &[&str] = &[];
const C_X_RI: &[&str] = &["x", "ri"];
const C_X_R: &[&str] = &["x", "r"];
const C_X_X: &[&str] = &["x", "x"];
const C_X_X_X: &[&str] = &["x", "x", "x"];
const C_X_X_R: &[&str] = &["x", "x", "r"];
const C_X4: &[&str] = &["x", "x", "x", "x"];
const C_X5: &[&str] = &["x", "x", "x", "x", "x"];

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

    /// The offset of the carry flag word in the slot array, after every temp's slot.
    fn carry_offset(&self) -> i32 {
        (self.nb_temps * SLOT_BYTES) as i32
    }

    /// The offset of the two vector scratch areas in the slot array, after the carry word.
    fn lane_offset(&self) -> i32 {
        ((self.nb_temps + 1) * SLOT_BYTES) as i32
    }

    fn exit_with(&mut self, k: u64, value: Option<u64>) {
        self.note_exit();
        if let Some(v) = value {
            self.a.movi(true, RDX, v, false);
        }
        self.a.movi(false, RAX, k, false);
        self.a.jump(None, self.exit, false);
    }

    /// Record in the run context that this block left, after the instruction of [`Gen::insn`].
    fn note_exit(&mut self) {
        self.a.movabs(TMP0, self.meta);
        self.a.store(TMP0, Mem::Base(CTX, META_OFFSET as i32), 8);
        self.a.store_imm(self.insn, Mem::Base(CTX, INSN_OFFSET as i32), 8);
    }

    /// `tcg_out_movi`, leaving the flags alone while a carry is live.
    fn movi(&mut self, ty: Type, d: Reg, v: u64) {
        self.a.movi(ty == Type::I64, d, v, self.carry_live);
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

    /// Check that `len` bytes at the offset in r11 are inside the CPU state.
    fn check_bounds(&mut self, len: u64) {
        self.a.mov(P_REXW, TMP1, TMP0);
        self.a.arithi(arith::ADD, P_REXW, TMP1, len as i64);
        self.a.jump(Some(cc::B), self.bounds, false);
        self.a.cmp_mem(P_REXW, TMP1, Mem::Base(CTX, ENV_LEN_OFFSET as i32));
        self.a.jump(Some(cc::A), self.bounds, false);
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
            self.a.movi(true, TMP0, (args[bi] as i64).wrapping_add(off) as u64, false);
        } else if fits_i32(off) {
            self.a.lea(P_REXW, TMP0, Mem::Base(args[bi] as Reg, off as i32));
        } else {
            self.a.movabs(TMP0, off as u64);
            self.a.arith(arith::ADD, P_REXW, TMP0, args[bi] as Reg);
        }
        self.check_bounds(len);
        Addr::Dyn
    }

    /// Call the service routine for `req`. Leaves the block if it reports an unwind or error.
    fn service(&mut self, req: Request) {
        self.service_via(req, self.service, self.insn, 0);
    }

    /// [`Gen::service`] through the routine at `routine`, with `tag` in the upper half of the
    /// request index and `flags` ORed into its lower half.
    fn service_via(&mut self, req: Request, routine: u64, tag: u64, flags: u64) {
        let idx = self.requests.len();
        debug_assert!(idx < 1 << 31);
        self.requests.push(req);
        self.insn_of.push(self.insn);
        // As in QEMU, helpers are called with whatever upper vector state the code left, but
        // a block that uses 256-bit vectors clears it, which costs little next to its work.
        if self.v256 {
            self.a.vex_opc(op::VZEROUPPER, 0, 0, 0, 0);
        }
        self.a.mov(P_REXW, ARGS[0], CTX);
        self.a.movi(true, ARGS[1], idx as u64 | flags | tag << 32, false);
        self.a.movabs(ARGS[2], self.meta);
        self.a.movabs(TMP0, routine);
        self.a.call_reg(TMP0);
        self.a.test(P_REXW, RAX, RAX);
        let ok = self.a.new_label();
        self.a.jump(Some(cc::E), ok, true);
        self.a.movi(false, RDX, 0, false);
        self.a.jump(None, self.exit, false);
        self.a.bind(ok);
    }

    /// A direct call to the [`ruvm_jit_interp::NativeHelperFn`] at `addr` of a helper with `nin` argument words,
    /// taking them from and leaving its result in the argument words, as the service routine
    /// would. Such a helper has no side effects, so it cannot raise an exception and needs no
    /// guest state, and as in QEMU the call is all there is to it.
    fn call_native(&mut self, addr: u64, nin: usize, ret: HelperType) {
        if self.v256 {
            self.a.vex_opc(op::VZEROUPPER, 0, 0, 0, 0);
        }
        let regs = if WIN64 { [RCX, RDX, R8, R9] } else { [RDI, RSI, RDX, RCX] };
        for (k, &r) in regs.iter().enumerate().take(nin) {
            self.a.load(r, Mem::Base(CTX, 8 * k as i32), 8, false, P_REXW);
        }
        self.a.movabs(TMP0, addr);
        self.a.call_reg(TMP0);
        match ret {
            HelperType::Void => {}
            HelperType::I32 => self.a.store(RAX, Mem::Base(CTX, 0), 4),
            _ => self.a.store(RAX, Mem::Base(CTX, 0), 8),
        }
    }

    /// `qemu_ld` through the service routine: the value of `oi` at `addr` into `out`.
    fn qemu_ld_slow(&mut self, ty: Type, oi: MemOpIdx, out: Reg, addr: Reg) {
        self.put_args(&[addr as u64]);
        self.service(Request::Load(oi));
        self.a.load(out, Mem::Base(CTX, 0), ty.size(), false, rexw(ty));
    }

    /// `qemu_ld2` through the service routine: the 128-bit value of `oi` at `addr` into `lo`
    /// and `hi`.
    fn qemu_ld2_slow(&mut self, oi: MemOpIdx, lo: Reg, hi: Reg, addr: Reg) {
        self.put_args(&[addr as u64]);
        self.service(Request::Load(oi));
        self.a.load(lo, Mem::Base(CTX, 0), 8, false, P_REXW);
        self.a.load(hi, Mem::Base(CTX, 8), 8, false, P_REXW);
    }

    /// `qemu_st2` through the service routine: the low and high halves in `val`, each a
    /// register or a constant if its flag is set, stored as `oi` says at `addr`.
    fn qemu_st2_slow(&mut self, oi: MemOpIdx, val: [(u64, bool); 2], addr: Reg) {
        for (k, (v, is_const)) in val.into_iter().enumerate() {
            let at = Mem::Base(CTX, 8 * k as i32);
            if is_const {
                self.a.store_imm(v, at, 8);
            } else {
                self.a.store(v as Reg, at, 8);
            }
        }
        self.a.store(addr, Mem::Base(CTX, 16), 8);
        self.service(Request::Store(oi));
    }

    /// `qemu_st` through the service routine: `val` (a register, or a constant if the flag is
    /// set) stored as `oi` says at `addr`.
    fn qemu_st_slow(&mut self, ty: Type, oi: MemOpIdx, val: (u64, bool), addr: Reg) {
        let (v, is_const) = val;
        if is_const && ty == Type::I32 {
            // Zero extended: the low word, then a zero high word.
            self.a.store_imm(v, Mem::Base(CTX, 0), 4);
            self.a.store_imm(0, Mem::Base(CTX, 4), 4);
        } else if is_const {
            self.a.store_imm(v, Mem::Base(CTX, 0), 8);
        } else {
            self.a.store(v as Reg, Mem::Base(CTX, 0), 8);
        }
        self.a.store_imm(0, Mem::Base(CTX, 8), 8);
        self.a.store(addr, Mem::Base(CTX, 16), 8);
        self.service(Request::Store(oi));
    }

    /// The inline TLB lookup of a `qemu_ld` or `qemu_st`, QEMU's `prepare_host_addr`. On a hit
    /// it falls through with the entry's addend in `TMP1`, so the access is at `(addr, TMP1)`
    /// for the returned address register; on a miss it jumps to the returned label. `None`
    /// when the access always takes the slow path: no TLB to read, a byte swapped access, a
    /// 128-bit access that neither two 64-bit host accesses nor one atomic vector access can
    /// make, or an alignment the compare cannot check.
    fn tlb_fast_path(
        &mut self,
        f: &Func,
        addr: Reg,
        oi: MemOpIdx,
        store: bool,
    ) -> Option<(Reg, usize)> {
        let page_bits = self.tlb_page_bits?;
        let m = oi.memop();
        let mmu_idx = oi.mmu_idx() as usize;
        let (s_bits, a_bits) = (m.size(), m.alignment_bits());
        // Alignment bits must stay below the comparator's flag bits, or a misaligned address
        // could match a flagged comparator.
        // Two 64-bit host accesses make a 128-bit one whose halves need only be atomic each,
        // and one vector access one that must be atomic as a whole when aligned.
        let atom = m.0 & MemOp::ATOM_MASK.0;
        let vec_ok = atom == MemOp::ATOM_IFALIGN.0 && self.feat.avx && self.feat.atomic_vmovdqa;
        if s_bits > 4
            || (s_bits == 4 && atomic16(oi) && !vec_ok)
            || m.is_bswap()
            || a_bits > TLB_FLAGS_SHIFT
            || mmu_idx >= TLB_MAX_MMU_MODES
        {
            return None;
        }
        let (s_mask, a_mask) = ((1u64 << s_bits) - 1, (1u64 << a_bits) - 1);
        let addr = if f.config.addr_type == Type::I32 {
            self.a.mov(0, TMP2, addr);
            TMP2
        } else {
            addr
        };
        let desc = TLB_OFFSET as i32 + (mmu_idx * TLB_DESC_WORDS * 8) as i32;
        self.a.mov(P_REXW, TMP1, addr);
        self.a.shifti(shift::SHR, P_REXW, TMP1, page_bits - TLB_ENTRY_BITS);
        self.a.arith_mem(arith::AND, P_REXW, TMP1, Mem::Base(CTX, desc));
        self.a.arith_mem(arith::ADD, P_REXW, TMP1, Mem::Base(CTX, desc + 8));
        // An access that is less aligned than its size must not cross the page: check the
        // address of its last byte, as QEMU does.
        if a_mask >= s_mask {
            self.a.mov(P_REXW, TMP0, addr);
        } else {
            self.a.lea(P_REXW, TMP0, Mem::Base(addr, (s_mask - a_mask) as i32));
        }
        self.a.arithi(arith::AND, P_REXW, TMP0, ((u64::MAX << page_bits) | a_mask) as i64);
        let cmp = if store { 8 } else { 0 };
        self.a.cmp_mem(P_REXW, TMP0, Mem::Base(TMP1, cmp));
        let miss = self.a.new_label();
        self.a.jump(Some(cc::NE), miss, false);
        self.a.load(TMP1, Mem::Base(TMP1, (TLB_ADDEND_WORD * 8) as i32), 8, false, P_REXW);
        Some((addr, miss))
    }

    /// The vector half of an atomic 128-bit access at `(addr, TMP1)` through `VT0`, QEMU's
    /// `MO_128` case of `tcg_out_qemu_ld_direct` and `tcg_out_qemu_st_direct`: `vmovdqa` when
    /// the op requires 16-byte alignment, `vmovdqu` when the host makes it atomic too, and
    /// otherwise a test of the address that picks `vmovdqa` when it is aligned. The addend is
    /// page aligned, so the guest address tells.
    fn vec_access16(&mut self, oi: MemOpIdx, addr: Reg, store: bool) {
        let host = Mem::Index(addr, TMP1, 0);
        let (aligned, unaligned) = if store {
            (op::MOVDQA_WX_VX, op::MOVDQU_WX_VX)
        } else {
            (op::MOVDQA_VX_WX, op::MOVDQU_VX_WX)
        };
        if oi.memop().alignment_bits() >= 4 {
            self.a.vex_modrm_mem(aligned, VT0, 0, host);
        } else if self.feat.atomic_vmovdqu {
            self.a.vex_modrm_mem(unaligned, VT0, 0, host);
        } else {
            let (other, done) = (self.a.new_label(), self.a.new_label());
            self.a.testi(0, addr, 15);
            self.a.jump(Some(cc::NE), other, true);
            self.a.vex_modrm_mem(aligned, VT0, 0, host);
            self.a.jump(None, done, true);
            self.a.bind(other);
            self.a.vex_modrm_mem(unaligned, VT0, 0, host);
            self.a.bind(done);
        }
    }

    /// Set the flags for `c` from `a` and `b`, `tcg_out_cmp`, and return the x86 condition
    /// that holds when `c` does.
    fn compare(&mut self, ty: Type, c: Cond, a: Reg, b: u64, b_const: bool) -> u8 {
        let w = rexw(ty);
        if !b_const {
            if c.is_tst() {
                self.a.test(w, a, b as Reg);
            } else {
                self.a.arith(arith::CMP, w, a, b as Reg);
            }
            return cond_code(c);
        }
        let v = norm(ty, b);
        if c.is_tst() {
            let m = if ty == Type::I32 { v as u32 as u64 } else { v as u64 };
            if ty == Type::I32 || m <= u32::MAX as u64 {
                if m <= 0xff && a < 4 {
                    // `testb` on al to bl, as QEMU does for small masks.
                    self.a.opc(0xf6, 0, a, 0);
                    self.a.b8(0xc0 | a);
                    self.a.b8(m as u8);
                } else {
                    self.a.testi(0, a, m as u32);
                }
            } else if fits_i32(v) {
                self.a.testi(P_REXW, a, v as u32);
            } else {
                // A single bit above bit 31: `bt` puts it in the carry flag.
                self.a.bti(P_REXW, a, m.trailing_zeros());
                return if c == Cond::TstEq { cc::AE } else { cc::B };
            }
            return cond_code(c);
        }
        if v == 0 {
            self.a.test(w, a, a);
        } else {
            self.a.arithi(arith::CMP, w, a, v);
        }
        cond_code(c)
    }

    /// The add and subtract with carry family. The carry lives in a word after the temp slots,
    /// except between two adjacent ops of the same family, where it stays in the flags.
    fn carry_op(&mut self, f: &Func, id: OpId, op: &Op, args: &[u64]) {
        let w = rexw(op.ty);
        let carry = Mem::Base(SLOTS, self.carry_offset());
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
        let (d, b) = (args[0] as Reg, args[2] as Reg);
        if carry_in {
            let fused = f.prev_op(id).is_some_and(|p| {
                let p = f.op(p);
                family_out.contains(&p.opc) && p.ty == op.ty
            });
            if !fused {
                self.a.bti_mem(P_REXW, carry, 0);
            }
        }
        if matches!(op.opc, Opcode::Addc1o | Opcode::Subb1o) {
            self.a.raw(op::STC);
        }
        let code = match op.opc {
            Opcode::Addco => arith::ADD,
            Opcode::Subbo => arith::SUB,
            _ if sub => arith::SBB,
            _ => arith::ADC,
        };
        self.a.arith(code, w, d, b);
        self.carry_live = false;
        if carry_out {
            let fused = f.next_op(id).is_some_and(|n| {
                let n = f.op(n);
                family_in.contains(&n.opc) && n.ty == op.ty
            });
            if fused {
                self.carry_live = true;
            } else {
                self.a.setcc_mem(cc::B, carry);
            }
        }
    }

    /// Store `regs` to the argument words, from word 0.
    fn put_args(&mut self, regs: &[u64]) {
        for (k, &r) in regs.iter().enumerate() {
            self.a.store(r as Reg, Mem::Base(CTX, 8 * k as i32), 8);
        }
    }

    /// `d = a / b` or `a % b` without host exceptions.
    fn divide(&mut self, ty: Type, signed: bool, rem: bool, d: Reg, a: Reg, b: Reg) {
        let w = rexw(ty);
        // A zero divisor divides by one.
        self.a.mov(w, TMP2, b);
        self.a.movi(false, TMP0, 1, false);
        self.a.test(w, TMP2, TMP2);
        self.a.cmov(cc::E, w, TMP2, TMP0);
        let done = self.a.new_label();
        if signed {
            // Dividing by -1 is a negation, which also covers the one quotient that overflows.
            let normal = self.a.new_label();
            self.a.arithi(arith::CMP, w, TMP2, -1);
            self.a.jump(Some(cc::NE), normal, true);
            if rem {
                self.a.movi(false, d, 0, true);
            } else {
                self.a.mov(w, d, a);
                self.a.ext3(ext3::NEG, w, d);
            }
            self.a.jump(None, done, true);
            self.a.bind(normal);
        }
        self.a.mov(P_REXW, TMP1, RAX);
        self.a.mov(P_REXW, TMP0, RDX);
        self.a.mov(w, RAX, a);
        if signed {
            // cqo or cdq.
            self.a.opc(0x99 | w, 0, 0, 0);
            self.a.ext3(ext3::IDIV, w, TMP2);
        } else {
            self.a.arith(arith::XOR, 0, RDX, RDX);
            self.a.ext3(ext3::DIV, w, TMP2);
        }
        self.a.mov(P_REXW, TMP2, if rem { RDX } else { RAX });
        self.a.mov(P_REXW, RAX, TMP1);
        self.a.mov(P_REXW, RDX, TMP0);
        self.a.mov(w, d, TMP2);
        self.a.bind(done);
    }

    /// The high half of a product.
    fn mul_high(&mut self, ty: Type, signed: bool, d: Reg, a: Reg, b: Reg) {
        if ty == Type::I64 {
            self.a.mov(P_REXW, TMP1, RAX);
            self.a.mov(P_REXW, TMP0, RDX);
            self.a.mov(P_REXW, TMP2, b);
            self.a.mov(P_REXW, RAX, a);
            self.a.ext3(if signed { ext3::IMUL } else { ext3::MUL }, P_REXW, TMP2);
            self.a.mov(P_REXW, TMP2, RDX);
            self.a.mov(P_REXW, RAX, TMP1);
            self.a.mov(P_REXW, RDX, TMP0);
            self.a.mov(P_REXW, d, TMP2);
        } else {
            if signed {
                self.a.modrm(op::MOVSLQ, TMP0, a);
                self.a.modrm(op::MOVSLQ, TMP2, b);
            } else {
                self.a.mov(0, TMP0, a);
                self.a.mov(0, TMP2, b);
            }
            self.a.modrm(op::IMUL_GV_EV | P_REXW, TMP0, TMP2);
            self.a.shifti(shift::SHR, P_REXW, TMP0, 32);
            self.a.mov(0, d, TMP0);
        }
    }

    /// `clz` and `ctz`: the count of `a`, or `b` when `a` is zero.
    fn count_zeros(&mut self, ty: Type, lead: bool, d: Reg, a: Reg, b: u64, b_const: bool) {
        let w = rexw(ty);
        let bits = ty.bits();
        if (lead && self.feat.lzcnt) || (!lead && self.feat.bmi1 && self.feat.lzcnt) {
            // lzcnt and tzcnt give the width for zero and set the carry flag.
            let opc = if lead { op::LZCNT } else { op::TZCNT };
            self.a.modrm(opc | w, TMP0, a);
            if !b_const {
                self.a.cmov(cc::B, w, TMP0, b as Reg);
            }
            self.a.mov(w, d, TMP0);
            return;
        }
        // bsr and bsf set the zero flag for zero and leave the destination undefined.
        if lead {
            if b_const {
                self.a.movi(false, TMP2, (2 * bits - 1) as u64, false);
            } else {
                self.a.mov(w, TMP2, b as Reg);
                self.a.arithi(arith::XOR, w, TMP2, (bits - 1) as i64);
            }
            self.a.modrm(op::BSR | w, TMP0, a);
            self.a.cmov(cc::E, w, TMP0, TMP2);
            self.a.arithi(arith::XOR, w, TMP0, (bits - 1) as i64);
        } else {
            if b_const {
                self.a.movi(false, TMP2, bits as u64, false);
            } else {
                self.a.mov(w, TMP2, b as Reg);
            }
            self.a.modrm(op::BSF | w, TMP0, a);
            self.a.cmov(cc::E, w, TMP0, TMP2);
        }
        self.a.mov(w, d, TMP0);
    }

    /// `ctpop` without POPCNT: the usual bit slicing sum.
    fn popcount(&mut self, ty: Type, d: Reg, a: Reg) {
        let w = rexw(ty);
        let is64 = ty == Type::I64;
        let k = |c: u64| if is64 { c } else { c & 0xffff_ffff };
        self.a.mov(w, TMP0, a);
        self.a.mov(w, TMP2, TMP0);
        self.a.shifti(shift::SHR, w, TMP2, 1);
        self.a.movi(is64, TMP1, k(0x5555_5555_5555_5555), false);
        self.a.arith(arith::AND, w, TMP2, TMP1);
        self.a.arith(arith::SUB, w, TMP0, TMP2);
        self.a.movi(is64, TMP1, k(0x3333_3333_3333_3333), false);
        self.a.mov(w, TMP2, TMP0);
        self.a.arith(arith::AND, w, TMP0, TMP1);
        self.a.shifti(shift::SHR, w, TMP2, 2);
        self.a.arith(arith::AND, w, TMP2, TMP1);
        self.a.arith(arith::ADD, w, TMP0, TMP2);
        self.a.mov(w, TMP2, TMP0);
        self.a.shifti(shift::SHR, w, TMP2, 4);
        self.a.arith(arith::ADD, w, TMP0, TMP2);
        self.a.movi(is64, TMP1, k(0x0f0f_0f0f_0f0f_0f0f), false);
        self.a.arith(arith::AND, w, TMP0, TMP1);
        self.a.movi(is64, TMP1, k(0x0101_0101_0101_0101), false);
        self.a.modrm(op::IMUL_GV_EV | w, TMP0, TMP1);
        self.a.shifti(shift::SHR, w, TMP0, ty.bits() - 8);
        self.a.mov(w, d, TMP0);
    }

    /// `d &= v` for a constant that may need more than a sign extended 32-bit immediate.
    fn and_const(&mut self, ty: Type, d: Reg, v: u64) {
        let w = rexw(ty);
        if ty == Type::I32 || fits_i32(v as i64) {
            self.a.arithi(arith::AND, w, d, v as i32 as i64);
        } else if v <= u32::MAX as u64 {
            // A 32-bit op clears the high half, which the mask clears anyway.
            self.a.arithi(arith::AND, 0, d, v as u32 as i32 as i64);
        } else {
            self.a.movabs(TMP1, v);
            self.a.arith(arith::AND, P_REXW, d, TMP1);
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
        let w = rexw(ty);
        let bits = ty.bits();
        let r = |k: usize| args[k] as Reg;
        let (d, a1) = (r(0), if args.len() > 1 { r(1) } else { 0 });
        match op.opc {
            Opcode::ExtI32I64 => self.a.modrm(op::MOVSLQ, d, a1),
            Opcode::ExtuI32I64 | Opcode::ExtrlI64I32 => self.a.modrm(op::MOVL_GV_EV, d, a1),
            Opcode::ExtrhI64I32 => {
                self.a.mov(P_REXW, d, a1);
                self.a.shifti(shift::SHR, P_REXW, d, 32);
            }
            Opcode::Add => {
                if const_args[2] {
                    let v = norm(ty, args[2]);
                    if d == a1 {
                        if v != 0 {
                            self.a.arithi(arith::ADD, w, d, v);
                        }
                    } else {
                        self.a.lea(w, d, Mem::Base(a1, v as i32));
                    }
                } else if d == a1 {
                    self.a.arith(arith::ADD, w, d, r(2));
                } else if d == r(2) {
                    self.a.arith(arith::ADD, w, d, a1);
                } else {
                    self.a.lea(w, d, Mem::Index(a1, r(2), 0));
                }
            }
            Opcode::Sub | Opcode::And | Opcode::Or | Opcode::Xor => {
                let code = match op.opc {
                    Opcode::Sub => arith::SUB,
                    Opcode::And => arith::AND,
                    Opcode::Or => arith::OR,
                    _ => arith::XOR,
                };
                if !const_args[2] {
                    self.a.arith(code, w, d, r(2));
                } else if op.opc == Opcode::And {
                    self.and_const(ty, d, norm(ty, args[2]) as u64);
                } else {
                    self.a.arithi(code, w, d, norm(ty, args[2]));
                }
            }
            Opcode::Andc => {
                if self.feat.bmi1 {
                    self.a.vex_modrm(op::ANDN | w, d, r(2), a1);
                } else {
                    self.a.mov(w, TMP0, r(2));
                    self.a.ext3(ext3::NOT, w, TMP0);
                    self.a.arith(arith::AND, w, d, TMP0);
                }
            }
            Opcode::Orc | Opcode::Eqv => {
                self.a.mov(w, TMP0, r(2));
                self.a.ext3(ext3::NOT, w, TMP0);
                let code = if op.opc == Opcode::Orc { arith::OR } else { arith::XOR };
                self.a.arith(code, w, d, TMP0);
            }
            Opcode::Nand | Opcode::Nor => {
                let code = if op.opc == Opcode::Nand { arith::AND } else { arith::OR };
                self.a.arith(code, w, d, r(2));
                self.a.ext3(ext3::NOT, w, d);
            }
            Opcode::Shl | Opcode::Shr | Opcode::Sar | Opcode::Rotl | Opcode::Rotr => {
                let code = match op.opc {
                    Opcode::Shl => shift::SHL,
                    Opcode::Shr => shift::SHR,
                    Opcode::Sar => shift::SAR,
                    Opcode::Rotl => shift::ROL,
                    _ => shift::ROR,
                };
                if const_args[2] {
                    self.a.mov(w, d, a1);
                    let n = args[2] as u32 & (bits - 1);
                    if n != 0 {
                        self.a.shifti(code, w, d, n);
                    }
                } else if self.feat.bmi2
                    && matches!(op.opc, Opcode::Shl | Opcode::Shr | Opcode::Sar)
                {
                    let opc = match op.opc {
                        Opcode::Shl => op::SHLX,
                        Opcode::Shr => op::SHRX,
                        _ => op::SARX,
                    };
                    self.a.vex_modrm(opc | w, d, r(2), a1);
                } else {
                    self.a.mov(0, TMP2, r(2));
                    self.a.mov(w, d, a1);
                    self.a.shift_cl(code, w, d);
                }
            }
            Opcode::Mul => {
                if const_args[2] {
                    let v = norm(ty, args[2]);
                    if v == v as i8 as i64 {
                        self.a.modrm(op::IMUL_GV_EV_IB | w, d, d);
                        self.a.b8(v as u8);
                    } else {
                        self.a.modrm(op::IMUL_GV_EV_IZ | w, d, d);
                        self.a.b32(v as u32);
                    }
                } else {
                    self.a.modrm(op::IMUL_GV_EV | w, d, r(2));
                }
            }
            Opcode::Muls2 | Opcode::Mulu2 => {
                let code = if op.opc == Opcode::Muls2 { ext3::IMUL } else { ext3::MUL };
                self.a.ext3(code, w, r(3));
            }
            Opcode::Muluh | Opcode::Mulsh => {
                self.mul_high(ty, op.opc == Opcode::Mulsh, d, a1, r(2));
            }
            Opcode::Divs | Opcode::Divu | Opcode::Rems | Opcode::Remu => {
                let signed = matches!(op.opc, Opcode::Divs | Opcode::Rems);
                let rem = matches!(op.opc, Opcode::Rems | Opcode::Remu);
                self.divide(ty, signed, rem, d, a1, r(2));
            }
            Opcode::Divs2 | Opcode::Divu2 => {
                self.put_args(&args[2..5]);
                self.service(Request::Div2 { signed: op.opc == Opcode::Divs2, bits });
                self.a.load(r(0), Mem::Base(CTX, 0), ty.size(), false, w);
                self.a.load(r(1), Mem::Base(CTX, 8), ty.size(), false, w);
            }
            Opcode::Clz | Opcode::Ctz => {
                self.count_zeros(ty, op.opc == Opcode::Clz, d, a1, args[2], const_args[2]);
            }
            Opcode::Ctpop => {
                if self.feat.popcnt {
                    self.a.modrm(op::POPCNT | w, d, a1);
                } else {
                    self.popcount(ty, d, a1);
                }
            }
            Opcode::Neg => self.a.ext3(ext3::NEG, w, d),
            Opcode::Not => self.a.ext3(ext3::NOT, w, d),
            Opcode::Bswap16 => {
                let os = op.args[2] as u32 & bswap::OS != 0;
                self.a.mov(0, d, a1);
                self.a.rolw8(d);
                if os {
                    self.a.modrm(op::MOVSWL | w, d, d);
                } else {
                    self.a.modrm(op::MOVZWL, d, d);
                }
            }
            Opcode::Bswap32 => {
                let os = op.args[2] as u32 & bswap::OS != 0;
                self.a.mov(0, d, a1);
                self.a.bswap(0, d);
                if os && ty == Type::I64 {
                    self.a.modrm(op::MOVSLQ, d, d);
                }
            }
            Opcode::Bswap64 => {
                self.a.mov(P_REXW, d, a1);
                self.a.bswap(P_REXW, d);
            }
            Opcode::Deposit => {
                let (ofs, len) = (op.args[3] as u32, op.args[4] as u32);
                if len == 0 || ofs + len > bits {
                    return Err(bad(op, "field out of range"));
                }
                if len == bits {
                    self.a.mov(w, d, r(2));
                } else if ofs == 0 && len == 8 {
                    // As tcg/i386 does: a byte move into the low byte, `d` being `a1`.
                    self.a.modrm(op::MOVB_EV_GV | P_REXB_R | P_REXB_RM, r(2), d);
                } else if ofs == 0 && len == 16 {
                    self.a.modrm(op::MOVL_EV_GV | P_DATA16, r(2), d);
                } else {
                    let fm = field_mask(len) << ofs;
                    self.a.mov(w, TMP0, r(2));
                    self.a.shifti(shift::SHL, w, TMP0, bits - len);
                    if bits - len - ofs != 0 {
                        self.a.shifti(shift::SHR, w, TMP0, bits - len - ofs);
                    }
                    let keep = !fm & if ty == Type::I32 { 0xffff_ffff } else { u64::MAX };
                    self.and_const(ty, d, keep);
                    self.a.arith(arith::OR, w, d, TMP0);
                }
            }
            Opcode::Extract | Opcode::Sextract => {
                let (ofs, len) = (op.args[2] as u32, op.args[3] as u32);
                if len == 0 || ofs + len > bits {
                    return Err(bad(op, "field out of range"));
                }
                let signed = op.opc == Opcode::Sextract;
                match (ofs, len, signed) {
                    (0, 8, false) => self.a.modrm(op::MOVZBL | P_REXB_RM, d, a1),
                    (0, 16, false) => self.a.modrm(op::MOVZWL, d, a1),
                    (0, 8, true) => self.a.modrm(op::MOVSBL | P_REXB_RM | w, d, a1),
                    (0, 16, true) => self.a.modrm(op::MOVSWL | w, d, a1),
                    (0, 32, false) if ty == Type::I64 => self.a.modrm(op::MOVL_GV_EV, d, a1),
                    (0, 32, true) if ty == Type::I64 => self.a.modrm(op::MOVSLQ, d, a1),
                    // QEMU's `tcg_gen_extract_*` makes this an `andi` on x86 hosts. A 32-bit
                    // `and` also clears the high half.
                    (0, _, false) if len < 32 => {
                        self.a.mov(0, d, a1);
                        self.a.arithi(arith::AND, 0, d, (1i64 << len) - 1);
                    }
                    _ => {
                        self.a.mov(w, d, a1);
                        if bits - ofs - len != 0 {
                            self.a.shifti(shift::SHL, w, d, bits - ofs - len);
                        }
                        if len != bits {
                            let code = if signed { shift::SAR } else { shift::SHR };
                            self.a.shifti(code, w, d, bits - len);
                        }
                    }
                }
            }
            Opcode::Extract2 => {
                let ofs = op.args[3] as u32;
                if ofs >= bits {
                    return Err(bad(op, "shift out of range"));
                }
                if ofs != 0 {
                    self.a.modrm(op::SHRD_IB | w, r(2), d);
                    self.a.b8(ofs as u8);
                }
            }
            Opcode::Setcond | Opcode::Negsetcond => {
                let c = cond_arg(op, 3)?;
                let neg = op.opc == Opcode::Negsetcond;
                match c {
                    Cond::Never => self.movi(ty, d, 0),
                    Cond::Always => self.movi(ty, d, if neg { u64::MAX } else { 1 }),
                    _ => {
                        let zero_first = d != a1 && (const_args[2] || d != r(2));
                        if zero_first {
                            self.a.arith(arith::XOR, 0, d, d);
                        }
                        let code = self.compare(ty, c, a1, args[2], const_args[2]);
                        self.a.setcc(code, d);
                        if !zero_first {
                            self.a.modrm(op::MOVZBL | P_REXB_RM, d, d);
                        }
                        if neg {
                            self.a.ext3(ext3::NEG, w, d);
                        }
                    }
                }
            }
            Opcode::Movcond => {
                let c = cond_arg(op, 5)?;
                match c {
                    Cond::Never => {}
                    Cond::Always => self.a.mov(w, d, r(3)),
                    _ => {
                        let code = self.compare(ty, c, a1, args[2], const_args[2]);
                        self.a.cmov(code, w, d, r(3));
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
                let (size, signed) = match op.opc {
                    Opcode::Ld8u => (1, false),
                    Opcode::Ld8s => (1, true),
                    Opcode::Ld16u => (2, false),
                    Opcode::Ld16s => (2, true),
                    Opcode::Ld32u => (4, false),
                    Opcode::Ld32s => (4, true),
                    _ => (ty.size(), false),
                };
                let addr = self.host_addr(f, op, args, const_args, 1, size as u64);
                match addr {
                    Addr::Static(off) if size == 4 && Some(off) == self.icount_decr => {
                        self.a.load(TMP0, Mem::Base(CTX, DECR_OFFSET as i32), 8, false, P_REXW);
                        self.a.load(d, Mem::Base(TMP0, 0), size, signed, w);
                    }
                    _ => self.a.load(d, addr.mem(), size, signed, w),
                }
            }
            Opcode::St8 | Opcode::St16 | Opcode::St32 | Opcode::St => {
                let size = match op.opc {
                    Opcode::St8 => 1,
                    Opcode::St16 => 2,
                    Opcode::St32 => 4,
                    _ => ty.size(),
                };
                let addr = self.host_addr(f, op, args, const_args, 1, size as u64);
                if const_args[0] {
                    self.a.store_imm(args[0], addr.mem(), size);
                } else {
                    self.a.store(d, addr.mem(), size);
                }
            }
            Opcode::QemuLd2 => {
                let oi = MemOpIdx(op.args[3] as u32);
                match self.tlb_fast_path(f, r(2), oi, false) {
                    Some((addr, label)) if atomic16(oi) => {
                        self.vec_access16(oi, addr, false);
                        // `tcg_out_vec_to_pair`: `vmovq` and `vpextrq $1`.
                        self.a.vex_modrm(op::MOVD_EY_VY | P_REXW, VT0, 0, r(0));
                        self.a.vex_modrm(op::PEXTRD | P_REXW, VT0, 0, r(1));
                        self.a.b8(1);
                        let back = self.a.new_label();
                        self.a.bind(back);
                        self.ldst.push(LdstSlow {
                            label,
                            back,
                            oi,
                            ty,
                            addr: r(2),
                            val: None,
                            out: r(0),
                            hi: Some((r(1) as u64, false)),
                            insn: self.insn,
                        });
                    }
                    Some((addr, label)) => {
                        // Load the half whose register is the address last.
                        let halves = if r(0) == addr { [(1, 8), (0, 0)] } else { [(0, 0), (1, 8)] };
                        for (k, disp) in halves {
                            self.a.load(r(k), Mem::Index(addr, TMP1, disp), 8, false, P_REXW);
                        }
                        let back = self.a.new_label();
                        self.a.bind(back);
                        self.ldst.push(LdstSlow {
                            label,
                            back,
                            oi,
                            ty,
                            addr: r(2),
                            val: None,
                            out: r(0),
                            hi: Some((r(1) as u64, false)),
                            insn: self.insn,
                        });
                    }
                    None => self.qemu_ld2_slow(oi, r(0), r(1), r(2)),
                }
            }
            Opcode::QemuLd => {
                let oi = MemOpIdx(op.args[2] as u32);
                match self.tlb_fast_path(f, r(1), oi, false) {
                    Some((addr, label)) => {
                        let m = oi.memop();
                        let host = Mem::Index(addr, TMP1, 0);
                        self.a.load(r(0), host, m.size_bytes(), m.is_signed(), w);
                        let back = self.a.new_label();
                        self.a.bind(back);
                        let addr = r(1);
                        self.ldst.push(LdstSlow {
                            label,
                            back,
                            oi,
                            ty,
                            addr,
                            val: None,
                            out: r(0),
                            hi: None,
                            insn: self.insn,
                        });
                    }
                    None => self.qemu_ld_slow(ty, oi, r(0), r(1)),
                }
            }
            Opcode::QemuSt => {
                let oi = MemOpIdx(op.args[2] as u32);
                let val = (args[0], const_args[0]);
                match self.tlb_fast_path(f, r(1), oi, true) {
                    Some((addr, label)) => {
                        let host = Mem::Index(addr, TMP1, 0);
                        let size = oi.memop().size_bytes();
                        if val.1 {
                            self.a.store_imm(val.0, host, size);
                        } else {
                            self.a.store(val.0 as Reg, host, size);
                        }
                        let back = self.a.new_label();
                        self.a.bind(back);
                        let addr = r(1);
                        self.ldst.push(LdstSlow {
                            label,
                            back,
                            oi,
                            ty,
                            addr,
                            val: Some(val),
                            out: 0,
                            hi: None,
                            insn: self.insn,
                        });
                    }
                    None => self.qemu_st_slow(ty, oi, val, r(1)),
                }
            }
            Opcode::QemuSt2 => {
                let oi = MemOpIdx(op.args[3] as u32);
                let val = [(args[0], const_args[0]), (args[1], const_args[1])];
                match self.tlb_fast_path(f, r(2), oi, true) {
                    Some((addr, label)) if atomic16(oi) => {
                        // `tcg_out_pair_to_vec`: `vmovq` and `vpinsrq $1`, with a constant
                        // half put in `TMP0` first.
                        for (k, (v, is_const)) in val.into_iter().enumerate() {
                            let s = if is_const {
                                self.a.movi(true, TMP0, v, false);
                                TMP0
                            } else {
                                v as Reg
                            };
                            if k == 0 {
                                self.a.vex_modrm(op::MOVD_VY_EY | P_REXW, VT0, 0, s);
                            } else {
                                self.a.vex_modrm(op::PINSRD | P_REXW, VT0, VT0, s);
                                self.a.b8(1);
                            }
                        }
                        self.vec_access16(oi, addr, true);
                        let back = self.a.new_label();
                        self.a.bind(back);
                        self.ldst.push(LdstSlow {
                            label,
                            back,
                            oi,
                            ty,
                            addr: r(2),
                            val: Some(val[0]),
                            out: 0,
                            hi: Some(val[1]),
                            insn: self.insn,
                        });
                    }
                    Some((addr, label)) => {
                        for (k, (v, is_const)) in val.into_iter().enumerate() {
                            let host = Mem::Index(addr, TMP1, 8 * k as i32);
                            if is_const {
                                self.a.store_imm(v, host, 8);
                            } else {
                                self.a.store(v as Reg, host, 8);
                            }
                        }
                        let back = self.a.new_label();
                        self.a.bind(back);
                        self.ldst.push(LdstSlow {
                            label,
                            back,
                            oi,
                            ty,
                            addr: r(2),
                            val: Some(val[0]),
                            out: 0,
                            hi: Some(val[1]),
                            insn: self.insn,
                        });
                    }
                    None => self.qemu_st2_slow(oi, val, r(2)),
                }
            }
            Opcode::GotoPtr => {
                // Jump straight to the block when the service routine vouched for the address
                // (`lookup_tb_ptr` found a block it can enter); leave otherwise. The context
                // holds 0 when no address is vouched for, so 0 itself always leaves.
                let out = self.a.new_label();
                self.a.test(P_REXW, r(0), r(0));
                self.a.jump(Some(cc::E), out, true);
                self.a.cmp_mem(P_REXW, r(0), Mem::Base(CTX, GOTO_PTR_OK_OFFSET as i32));
                self.a.jump(Some(cc::NE), out, true);
                self.a.jmp_reg(r(0));
                self.a.bind(out);
                self.a.mov(P_REXW, RDX, r(0));
                self.exit_with(kind::GOTO_PTR, None);
            }
            other => return Err(GenCodeError::Unsupported(other.name().to_string())),
        }
        Ok(())
    }

    // Vector helpers.

    /// Load a whole `ty` vector from memory.
    fn vld(&mut self, ty: Type, d: Reg, m: Mem) {
        let opc = match ty {
            Type::V64 => op::MOVQ_VQ_WQ,
            _ => op::MOVDQU_VX_WX | vexl(ty),
        };
        if self.feat.avx {
            self.a.vex_modrm_mem(opc, d, 0, m);
        } else {
            self.a.modrm_mem(opc, d, m);
        }
    }

    /// Store a whole `ty` vector to memory.
    fn vst(&mut self, ty: Type, s: Reg, m: Mem) {
        let opc = match ty {
            Type::V64 => op::MOVQ_WQ_VQ,
            _ => op::MOVDQU_WX_VX | vexl(ty),
        };
        if self.feat.avx {
            self.a.vex_modrm_mem(opc, s, 0, m);
        } else {
            self.a.modrm_mem(opc, s, m);
        }
    }

    /// Copy one vector register to another.
    fn vmov(&mut self, ty: Type, d: Reg, s: Reg) {
        if d == s {
            return;
        }
        if self.feat.avx {
            self.a.vex_modrm(op::MOVDQA_VX_WX | vexl(ty), d, 0, s);
        } else {
            self.a.modrm(op::MOVDQA_VX_WX, d, s);
        }
    }

    /// `d = a op b`, with the legacy two operand form when there is no AVX.
    #[allow(clippy::too_many_arguments, reason = "operands of tcg_out_vec_op")]
    fn vop3(&mut self, opc: u32, ty: Type, d: Reg, a: Reg, b: Reg, commutative: bool) {
        if self.feat.avx {
            self.a.vex_modrm(opc | vexl(ty), d, a, b);
        } else if d == a {
            self.a.modrm(opc, d, b);
        } else if d == b && commutative {
            self.a.modrm(opc, d, a);
        } else if d == b {
            self.vmov(ty, VSWAP, b);
            self.vmov(ty, d, a);
            self.a.modrm(opc, d, VSWAP);
        } else {
            self.vmov(ty, d, a);
            self.a.modrm(opc, d, b);
        }
    }

    /// `d = op a` for an op that only reads its source.
    fn vop2(&mut self, opc: u32, ty: Type, d: Reg, a: Reg) {
        if self.feat.avx {
            self.a.vex_modrm(opc | vexl(ty), d, 0, a);
        } else {
            self.a.modrm(opc, d, a);
        }
    }

    /// A shift by an immediate, `/ext` of `opc`.
    #[allow(clippy::too_many_arguments, reason = "operands of tcg_out_vec_op")]
    fn vshift_imm(&mut self, opc: u32, ext: u8, ty: Type, d: Reg, a: Reg, n: u32) {
        if self.feat.avx {
            self.a.vex_modrm(opc | vexl(ty), ext, d, a);
        } else {
            self.vmov(ty, d, a);
            self.a.modrm(opc, ext, d);
        }
        self.a.b8(n as u8);
    }

    /// `pshufd`.
    fn pshufd(&mut self, ty: Type, d: Reg, a: Reg, imm: u8) {
        self.vop2(op::PSHUFD, ty, d, a);
        self.a.b8(imm);
    }

    /// `tcg_out_dupi_vec`: every element of `d` set from the 64-bit pattern `v`.
    fn vdupi(&mut self, ty: Type, d: Reg, v: u64) {
        if v == 0 {
            self.vop3(op::PXOR, ty, d, d, d, true);
        } else if v == u64::MAX {
            self.vop3(op::PCMPEQB, ty, d, d, d, true);
        } else {
            let mut data = [0u8; 32];
            for k in 0..4 {
                data[8 * k..8 * k + 8].copy_from_slice(&v.to_le_bytes());
            }
            let k = self.a.pool_entry(data);
            self.vld(ty, d, Mem::Pool(k));
        }
    }

    /// `tcg_out_dup_vec`: every element of `d` set from the general register `s`.
    fn vdup(&mut self, ty: Type, vece: u32, d: Reg, s: Reg) {
        if vece == 3 {
            self.vop2(op::MOVD_VY_EY | P_REXW, Type::V128, d, s);
        } else {
            self.vop2(op::MOVD_VY_EY, Type::V128, d, s);
        }
        if self.feat.avx2 {
            let opc = [op::VPBROADCASTB, op::VPBROADCASTW, op::VPBROADCASTD, op::VPBROADCASTQ]
                [vece as usize];
            self.a.vex_modrm(opc | vexl(ty), d, 0, d);
            return;
        }
        match vece {
            0 => {
                self.vop3(op::PUNPCKLBW, Type::V128, d, d, d, true);
                self.vop3(op::PUNPCKLWD, Type::V128, d, d, d, true);
                self.pshufd(Type::V128, d, d, 0);
            }
            1 => {
                self.vop3(op::PUNPCKLWD, Type::V128, d, d, d, true);
                self.pshufd(Type::V128, d, d, 0);
            }
            2 => self.pshufd(Type::V128, d, d, 0),
            _ => self.vop3(op::PUNPCKLQDQ, Type::V128, d, d, d, true),
        }
    }

    /// All ones in `d`.
    fn vones(&mut self, ty: Type, d: Reg) {
        self.vop3(op::PCMPEQB, ty, d, d, d, true);
    }

    /// Shift every element left by `n`, below the element width. Byte elements shift as words
    /// and mask off the bits that crossed into the next byte.
    fn vshli(&mut self, ty: Type, vece: u32, d: Reg, a: Reg, n: u32) {
        if n == 0 {
            return self.vmov(ty, d, a);
        }
        let opc = [op::PSHIFTW_IB, op::PSHIFTW_IB, op::PSHIFTD_IB, op::PSHIFTQ_IB][vece as usize];
        self.vshift_imm(opc, 6, ty, d, a, n);
        if vece == 0 {
            self.vdupi(ty, VT2, dup_const(0, (0xffu64 << n) & 0xff));
            self.vop3(op::PAND, ty, d, d, VT2, true);
        }
    }

    /// Shift every element right by `n`, below the element width, logically.
    fn vshri(&mut self, ty: Type, vece: u32, d: Reg, a: Reg, n: u32) {
        if n == 0 {
            return self.vmov(ty, d, a);
        }
        let opc = [op::PSHIFTW_IB, op::PSHIFTW_IB, op::PSHIFTD_IB, op::PSHIFTQ_IB][vece as usize];
        self.vshift_imm(opc, 2, ty, d, a, n);
        if vece == 0 {
            self.vdupi(ty, VT2, dup_const(0, 0xff >> n));
            self.vop3(op::PAND, ty, d, d, VT2, true);
        }
    }

    /// The comparison mask of `c` into VT0, `tcg_out_cmp_vec`. Returns false if the host has
    /// no instruction for it.
    #[allow(clippy::too_many_arguments, reason = "operands of tcg_out_vec_op")]
    fn vcmp_mask(&mut self, ty: Type, vece: u32, c: Cond, a: Reg, b: Reg) -> bool {
        match c {
            Cond::Never => {
                self.vdupi(ty, VT0, 0);
                return true;
            }
            Cond::Always => {
                self.vdupi(ty, VT0, u64::MAX);
                return true;
            }
            _ => {}
        }
        let eq = [op::PCMPEQB, op::PCMPEQW, op::PCMPEQD, op::PCMPEQQ][vece as usize];
        let gt = [op::PCMPGTB, op::PCMPGTW, op::PCMPGTD, op::PCMPGTQ][vece as usize];
        let needs_gt = !matches!(c, Cond::Eq | Cond::Ne | Cond::TstEq | Cond::TstNe);
        if vece == 3 && (!self.feat.sse41 || (needs_gt && !self.feat.sse42)) {
            return false;
        }
        let invert = match c {
            Cond::Eq | Cond::TstEq => {
                let (x, y) = if c.is_tst() {
                    self.vop3(op::PAND, ty, VT1, a, b, true);
                    self.vdupi(ty, VT2, 0);
                    (VT1, VT2)
                } else {
                    (a, b)
                };
                self.vop3(eq, ty, VT0, x, y, true);
                false
            }
            Cond::Ne | Cond::TstNe => {
                return self.vcmp_mask(ty, vece, c.invert(), a, b) && {
                    self.vinvert(ty);
                    true
                };
            }
            _ => {
                let (x, y) = if c.is_unsigned() {
                    // Flip the sign bits so that a signed compare orders unsigned values.
                    let sign = dup_const(vece, 1u64 << ((8 << vece) - 1));
                    self.vdupi(ty, VT0, sign);
                    self.vop3(op::PXOR, ty, VT2, a, VT0, true);
                    self.vop3(op::PXOR, ty, VT1, b, VT0, true);
                    (VT2, VT1)
                } else {
                    (a, b)
                };
                // Gt is x > y, Lt is y > x, Le is not Gt and Ge is not Lt.
                let (swap, inv) = match c {
                    Cond::Gt | Cond::Gtu => (false, false),
                    Cond::Lt | Cond::Ltu => (true, false),
                    Cond::Le | Cond::Leu => (false, true),
                    _ => (true, true),
                };
                let (x, y) = if swap { (y, x) } else { (x, y) };
                self.vop3(gt, ty, VT0, x, y, false);
                inv
            }
        };
        if invert {
            self.vinvert(ty);
        }
        true
    }

    /// VT0 = !VT0.
    fn vinvert(&mut self, ty: Type) {
        self.vones(ty, VT1);
        self.vop3(op::PXOR, ty, VT0, VT0, VT1, true);
    }

    /// One element at a time in general registers: `a` and `b` go to the two scratch areas,
    /// each element of the first is replaced by the result, and the area is loaded into `d`.
    #[allow(clippy::too_many_arguments, reason = "operands of tcg_out_vec_op")]
    fn lanes(&mut self, ty: Type, vece: u32, l: Lane, d: Reg, a: Reg, b: Option<Reg>) {
        let base = self.lane_offset();
        let size = 1u32 << vece;
        let bits = 8 * size;
        self.vst(ty, a, Mem::Base(SLOTS, base));
        if let Some(b) = b {
            self.vst(ty, b, Mem::Base(SLOTS, base + 32));
        }
        let signed = match l {
            Lane::Ssadd | Lane::Sssub | Lane::Smin | Lane::Smax | Lane::Abs | Lane::Sar => true,
            Lane::Sarv => true,
            Lane::Cmp(c) => c.is_signed(),
            _ => false,
        };
        let top = 64 - bits;
        for k in 0..ty.size() / size {
            let at = base + (k * size) as i32;
            let at_b = at + 32;
            self.a.load(TMP0, Mem::Base(SLOTS, at), size, signed, P_REXW);
            match l {
                Lane::Shl | Lane::Shr | Lane::Sar | Lane::Abs => {}
                Lane::Shlv | Lane::Shrv | Lane::Sarv | Lane::Rotlv | Lane::Rotrv => {
                    self.a.load(TMP2, Mem::Base(SLOTS, at_b), size, false, P_REXW);
                    self.a.arithi(arith::AND, 0, TMP2, (bits - 1) as i64);
                }
                _ => self.a.load(TMP1, Mem::Base(SLOTS, at_b), size, signed, P_REXW),
            }
            match l {
                Lane::Mul => self.a.modrm(op::IMUL_GV_EV | P_REXW, TMP0, TMP1),
                Lane::Ssadd | Lane::Sssub => {
                    // Work at the top of the register so that the host overflow flag is the
                    // element's, and saturate towards the sign of the first operand.
                    if top != 0 {
                        self.a.shifti(shift::SHL, P_REXW, TMP0, top);
                        self.a.shifti(shift::SHL, P_REXW, TMP1, top);
                    }
                    self.a.mov(P_REXW, TMP2, TMP0);
                    self.a.shifti(shift::SAR, P_REXW, TMP2, 63);
                    self.a.ext3(ext3::NOT, P_REXW, TMP2);
                    self.a.modrm(op::GRPBT | P_REXW, 7, TMP2);
                    self.a.b8(63);
                    let code = if l == Lane::Ssadd { arith::ADD } else { arith::SUB };
                    self.a.arith(code, P_REXW, TMP0, TMP1);
                    self.a.cmov(cc::O, P_REXW, TMP0, TMP2);
                    if top != 0 {
                        self.a.shifti(shift::SAR, P_REXW, TMP0, top);
                    }
                }
                Lane::Usadd | Lane::Ussub => {
                    if top != 0 {
                        self.a.shifti(shift::SHL, P_REXW, TMP0, top);
                        self.a.shifti(shift::SHL, P_REXW, TMP1, top);
                    }
                    if l == Lane::Usadd {
                        self.a.arith(arith::ADD, P_REXW, TMP0, TMP1);
                        self.a.arith(arith::SBB, P_REXW, TMP2, TMP2);
                        self.a.arith(arith::OR, P_REXW, TMP0, TMP2);
                    } else {
                        self.a.arith(arith::SUB, P_REXW, TMP0, TMP1);
                        self.a.arith(arith::SBB, P_REXW, TMP2, TMP2);
                        self.a.ext3(ext3::NOT, P_REXW, TMP2);
                        self.a.arith(arith::AND, P_REXW, TMP0, TMP2);
                    }
                    if top != 0 {
                        self.a.shifti(shift::SHR, P_REXW, TMP0, top);
                    }
                }
                Lane::Smin | Lane::Smax | Lane::Umin | Lane::Umax => {
                    // Keep the first operand unless the second wins.
                    let code = match l {
                        Lane::Smin => cc::G,
                        Lane::Smax => cc::L,
                        Lane::Umin => cc::A,
                        _ => cc::B,
                    };
                    self.a.arith(arith::CMP, P_REXW, TMP0, TMP1);
                    self.a.cmov(code, P_REXW, TMP0, TMP1);
                }
                Lane::Abs => {
                    self.a.mov(P_REXW, TMP1, TMP0);
                    self.a.ext3(ext3::NEG, P_REXW, TMP1);
                    self.a.cmov(cc::NS, P_REXW, TMP0, TMP1);
                }
                Lane::Shl | Lane::Shlv => self.a.shift_cl(shift::SHL, P_REXW, TMP0),
                Lane::Shr | Lane::Shrv => self.a.shift_cl(shift::SHR, P_REXW, TMP0),
                Lane::Sar | Lane::Sarv => self.a.shift_cl(shift::SAR, P_REXW, TMP0),
                Lane::Rotlv | Lane::Rotrv => {
                    let code = if l == Lane::Rotlv { shift::ROL } else { shift::ROR };
                    match size {
                        1 => self.a.modrm(0xd2, code, TMP0),
                        2 => self.a.modrm(op::SHIFT_CL | P_DATA16, code, TMP0),
                        4 => self.a.shift_cl(code, 0, TMP0),
                        _ => self.a.shift_cl(code, P_REXW, TMP0),
                    }
                }
                Lane::Cmp(c) => {
                    if c.is_tst() {
                        self.a.test(P_REXW, TMP0, TMP1);
                    } else {
                        self.a.arith(arith::CMP, P_REXW, TMP0, TMP1);
                    }
                    self.a.setcc(cond_code(c), TMP0);
                    self.a.modrm(op::MOVZBL | P_REXB_RM, TMP0, TMP0);
                    self.a.ext3(ext3::NEG, P_REXW, TMP0);
                }
            }
            self.a.store(TMP0, Mem::Base(SLOTS, at), size);
        }
        self.vld(ty, d, Mem::Base(SLOTS, base));
    }

    fn out_vector(&mut self, f: &Func, op: &Op, args: &[u64], const_args: &[bool]) -> R<()> {
        let ty = op.ty;
        let vece = op.vece as u32;
        let ve = vece as usize;
        let bits = 8u32 << vece;
        let r = |k: usize| args[k] as Reg;
        let d = r(0);
        let feat = self.feat;
        match op.opc {
            Opcode::LdVec => {
                let addr = self.host_addr(f, op, args, const_args, 1, ty.size() as u64);
                self.vld(ty, d, addr.mem());
            }
            Opcode::StVec => {
                let addr = self.host_addr(f, op, args, const_args, 1, ty.size() as u64);
                self.vst(ty, d, addr.mem());
            }
            Opcode::DupmVec => {
                let addr = self.host_addr(f, op, args, const_args, 1, 1 << vece);
                self.a.load(TMP1, addr.mem(), 1 << vece, false, P_REXW);
                self.vdup(ty, vece, d, TMP1);
            }
            Opcode::DupVec => self.vdup(ty, vece, d, r(1)),
            Opcode::AddVec => {
                let opc = [op::PADDB, op::PADDW, op::PADDD, op::PADDQ][ve];
                self.vop3(opc, ty, d, r(1), r(2), true);
            }
            Opcode::SubVec => {
                let opc = [op::PSUBB, op::PSUBW, op::PSUBD, op::PSUBQ][ve];
                self.vop3(opc, ty, d, r(1), r(2), false);
            }
            Opcode::NegVec => {
                let opc = [op::PSUBB, op::PSUBW, op::PSUBD, op::PSUBQ][ve];
                self.vdupi(ty, VT0, 0);
                self.vop3(opc, ty, d, VT0, r(1), false);
            }
            Opcode::MulVec => match vece {
                1 => self.vop3(op::PMULLW, ty, d, r(1), r(2), true),
                2 if feat.sse41 => self.vop3(op::PMULLD, ty, d, r(1), r(2), true),
                _ => self.lanes(ty, vece, Lane::Mul, d, r(1), Some(r(2))),
            },
            Opcode::SsaddVec | Opcode::UsaddVec | Opcode::SssubVec | Opcode::UssubVec => {
                let (simd, lane) = match op.opc {
                    Opcode::SsaddVec => ([op::PADDSB, op::PADDSW], Lane::Ssadd),
                    Opcode::UsaddVec => ([op::PADDUB, op::PADDUW], Lane::Usadd),
                    Opcode::SssubVec => ([op::PSUBSB, op::PSUBSW], Lane::Sssub),
                    _ => ([op::PSUBUB, op::PSUBUW], Lane::Ussub),
                };
                let comm = matches!(op.opc, Opcode::SsaddVec | Opcode::UsaddVec);
                if vece < 2 {
                    self.vop3(simd[ve], ty, d, r(1), r(2), comm);
                } else {
                    self.lanes(ty, vece, lane, d, r(1), Some(r(2)));
                }
            }
            Opcode::SminVec | Opcode::UminVec | Opcode::SmaxVec | Opcode::UmaxVec => {
                let (simd, lane) = match op.opc {
                    Opcode::SminVec => ([op::PMINSB, op::PMINSW, op::PMINSD], Lane::Smin),
                    Opcode::UminVec => ([op::PMINUB, op::PMINUW, op::PMINUD], Lane::Umin),
                    Opcode::SmaxVec => ([op::PMAXSB, op::PMAXSW, op::PMAXSD], Lane::Smax),
                    _ => ([op::PMAXUB, op::PMAXUW, op::PMAXUD], Lane::Umax),
                };
                let signed = matches!(op.opc, Opcode::SminVec | Opcode::SmaxVec);
                // SSE2 has pminub and pminsw and their max forms; the rest are SSE4.1.
                let sse2 = (vece == 0 && !signed) || (vece == 1 && signed);
                if vece < 3 && (sse2 || feat.sse41) {
                    self.vop3(simd[ve], ty, d, r(1), r(2), true);
                } else {
                    self.lanes(ty, vece, lane, d, r(1), Some(r(2)));
                }
            }
            Opcode::AbsVec => {
                if vece < 3 && feat.ssse3 {
                    self.vop2([op::PABSB, op::PABSW, op::PABSD][ve], ty, d, r(1));
                } else {
                    self.lanes(ty, vece, Lane::Abs, d, r(1), None);
                }
            }
            Opcode::AndVec => self.vop3(op::PAND, ty, d, r(1), r(2), true),
            Opcode::OrVec => self.vop3(op::POR, ty, d, r(1), r(2), true),
            Opcode::XorVec => self.vop3(op::PXOR, ty, d, r(1), r(2), true),
            Opcode::AndcVec => self.vop3(op::PANDN, ty, d, r(2), r(1), false),
            Opcode::OrcVec => {
                self.vones(ty, VT0);
                self.vop3(op::PXOR, ty, VT0, VT0, r(2), true);
                self.vop3(op::POR, ty, d, r(1), VT0, true);
            }
            Opcode::NandVec | Opcode::NorVec | Opcode::EqvVec => {
                let opc = match op.opc {
                    Opcode::NandVec => op::PAND,
                    Opcode::NorVec => op::POR,
                    _ => op::PXOR,
                };
                self.vop3(opc, ty, VT1, r(1), r(2), true);
                self.vones(ty, VT0);
                self.vop3(op::PXOR, ty, d, VT1, VT0, true);
            }
            Opcode::NotVec => {
                self.vones(ty, VT0);
                self.vop3(op::PXOR, ty, d, r(1), VT0, true);
            }
            Opcode::ShliVec => {
                let n = op.args[2] as u32 & (bits - 1);
                self.vshli(ty, vece, d, r(1), n);
            }
            Opcode::ShriVec => {
                let n = op.args[2] as u32 & (bits - 1);
                self.vshri(ty, vece, d, r(1), n);
            }
            Opcode::SariVec => {
                let n = op.args[2] as u32 & (bits - 1);
                if n == 0 {
                    self.vmov(ty, d, r(1));
                } else if vece == 1 || vece == 2 {
                    let opc = if vece == 1 { op::PSHIFTW_IB } else { op::PSHIFTD_IB };
                    self.vshift_imm(opc, 4, ty, d, r(1), n);
                } else {
                    self.a.movi(false, TMP2, n as u64, false);
                    self.lanes(ty, vece, Lane::Sar, d, r(1), None);
                }
            }
            Opcode::RotliVec => {
                let n = op.args[2] as u32 & (bits - 1);
                if n == 0 {
                    self.vmov(ty, d, r(1));
                } else {
                    self.vshli(ty, vece, VT0, r(1), n);
                    self.vshri(ty, vece, VT1, r(1), bits - n);
                    self.vop3(op::POR, ty, d, VT0, VT1, true);
                }
            }
            Opcode::ShlsVec | Opcode::ShrsVec | Opcode::SarsVec | Opcode::RotlsVec => {
                self.a.mov(0, TMP2, r(2));
                self.a.arithi(arith::AND, 0, TMP2, (bits - 1) as i64);
                let simd = match op.opc {
                    Opcode::ShlsVec | Opcode::RotlsVec => vece >= 1,
                    Opcode::ShrsVec => vece >= 1,
                    _ => vece == 1 || vece == 2,
                };
                if !simd {
                    let lane = match op.opc {
                        Opcode::ShlsVec => Lane::Shl,
                        Opcode::ShrsVec => Lane::Shr,
                        Opcode::SarsVec => Lane::Sar,
                        _ => {
                            // A byte rotate by a scalar is a rotate by a vector of copies.
                            self.vdup(ty, 0, VT0, TMP2);
                            self.lanes(ty, vece, Lane::Rotlv, d, r(1), Some(VT0));
                            return Ok(());
                        }
                    };
                    self.lanes(ty, vece, lane, d, r(1), None);
                    return Ok(());
                }
                let shl = [0, op::PSLLW, op::PSLLD, op::PSLLQ][ve];
                let shr = [0, op::PSRLW, op::PSRLD, op::PSRLQ][ve];
                let sar = [0, op::PSRAW, op::PSRAD, 0][ve];
                self.vop2(op::MOVD_VY_EY, Type::V128, VT1, TMP2);
                match op.opc {
                    Opcode::ShlsVec => self.vop3(shl, ty, d, r(1), VT1, false),
                    Opcode::ShrsVec => self.vop3(shr, ty, d, r(1), VT1, false),
                    Opcode::SarsVec => self.vop3(sar, ty, d, r(1), VT1, false),
                    _ => {
                        self.vop3(shl, ty, VT0, r(1), VT1, false);
                        self.a.movi(false, TMP1, bits as u64, false);
                        self.a.arith(arith::SUB, 0, TMP1, TMP2);
                        self.vop2(op::MOVD_VY_EY, Type::V128, VT1, TMP1);
                        self.vop3(shr, ty, VT1, r(1), VT1, false);
                        self.vop3(op::POR, ty, d, VT0, VT1, true);
                    }
                }
            }
            Opcode::ShlvVec
            | Opcode::ShrvVec
            | Opcode::SarvVec
            | Opcode::RotlvVec
            | Opcode::RotrvVec => {
                let simd = feat.avx2
                    && match op.opc {
                        Opcode::SarvVec => vece == 2,
                        _ => vece >= 2,
                    };
                if !simd {
                    let lane = match op.opc {
                        Opcode::ShlvVec => Lane::Shlv,
                        Opcode::ShrvVec => Lane::Shrv,
                        Opcode::SarvVec => Lane::Sarv,
                        Opcode::RotlvVec => Lane::Rotlv,
                        _ => Lane::Rotrv,
                    };
                    self.lanes(ty, vece, lane, d, r(1), Some(r(2)));
                    return Ok(());
                }
                let shl = if vece == 2 { op::VPSLLVD } else { op::VPSLLVQ };
                let shr = if vece == 2 { op::VPSRLVD } else { op::VPSRLVQ };
                let sub = if vece == 2 { op::PSUBD } else { op::PSUBQ };
                self.vdupi(ty, VT0, dup_const(vece, (bits - 1) as u64));
                self.vop3(op::PAND, ty, VT0, r(2), VT0, true);
                match op.opc {
                    Opcode::ShlvVec => self.vop3(shl, ty, d, r(1), VT0, false),
                    Opcode::ShrvVec => self.vop3(shr, ty, d, r(1), VT0, false),
                    Opcode::SarvVec => self.vop3(op::VPSRAVD, ty, d, r(1), VT0, false),
                    _ => {
                        // One way by n, the other by bits - n, which shifts everything out
                        // when n is 0.
                        self.vdupi(ty, VT1, dup_const(vece, bits as u64));
                        self.vop3(sub, ty, VT1, VT1, VT0, false);
                        let (first, second) =
                            if op.opc == Opcode::RotlvVec { (shl, shr) } else { (shr, shl) };
                        self.vop3(first, ty, VT0, r(1), VT0, false);
                        self.vop3(second, ty, VT1, r(1), VT1, false);
                        self.vop3(op::POR, ty, d, VT0, VT1, true);
                    }
                }
            }
            Opcode::CmpVec => {
                let c = cond_arg(op, 3)?;
                if self.vcmp_mask(ty, vece, c, r(1), r(2)) {
                    self.vmov(ty, d, VT0);
                } else {
                    self.lanes(ty, vece, Lane::Cmp(c), d, r(1), Some(r(2)));
                }
            }
            Opcode::BitselVec => {
                self.vop3(op::PAND, ty, VT0, r(1), r(2), true);
                self.vop3(op::PANDN, ty, VT1, r(1), r(3), false);
                self.vop3(op::POR, ty, d, VT0, VT1, true);
            }
            Opcode::CmpselVec => {
                let c = cond_arg(op, 5)?;
                if !self.vcmp_mask(ty, vece, c, r(1), r(2)) {
                    self.lanes(ty, vece, Lane::Cmp(c), VT0, r(1), Some(r(2)));
                }
                self.vop3(op::PAND, ty, VT1, r(3), VT0, true);
                self.vop3(op::PANDN, ty, VT2, VT0, r(4), false);
                self.vop3(op::POR, ty, d, VT1, VT2, true);
            }
            other => return Err(GenCodeError::Unsupported(other.name().to_string())),
        }
        Ok(())
    }

    /// The inline cache of a `lookup_tb_ptr_ic` call, whose arguments are already in the run
    /// context: when the entry for the program counter in argument 1 names a block header with
    /// that program counter and a nonzero address, jump there; otherwise fall through to the
    /// lookup. The address of the cache words is patched in by the runtime.
    fn ic_probe(&mut self) {
        let miss = self.a.new_label();
        self.a.load(TMP0, Mem::Base(CTX, 8), 8, false, P_REXW);
        self.a.mov(0, RAX, TMP0);
        self.a.shifti(shift::SHR, 0, RAX, 4);
        self.a.arith(arith::XOR, 0, RAX, TMP0);
        self.a.arithi(arith::AND, 0, RAX, (IC_WAYS - 1) as i64);
        self.a.shifti(shift::SHL, 0, RAX, 3);
        let at = self.a.pos() + 2;
        self.a.movabs(TMP1, 0);
        self.ic_sites.push((self.requests.len(), at));
        self.a.load(TMP1, Mem::Index(TMP1, RAX, 0), 8, false, P_REXW);
        self.a.cmp_mem(P_REXW, TMP0, Mem::Base(TMP1, 0));
        self.a.jump(Some(cc::NE), miss, true);
        self.a.load(RAX, Mem::Base(TMP1, 8), 8, false, P_REXW);
        self.a.test(P_REXW, RAX, RAX);
        self.a.jump(Some(cc::E), miss, true);
        self.a.jmp_reg(RAX);
        self.a.bind(miss);
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
            Type::V64 | Type::V128 | Type::V256 => VECS,
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
        None
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
            Opcode::St8 | Opcode::St16 | Opcode::St32 | Opcode::St => C_RE_RI,
            Opcode::Add => C_R_R_RE,
            Opcode::Sub | Opcode::Or | Opcode::Xor | Opcode::Mul => C_R_0_RE,
            Opcode::And => C_R_0_REZ,
            Opcode::Andc if self.feat.bmi1 => C_R_R_R,
            Opcode::Andc | Opcode::Orc | Opcode::Eqv | Opcode::Nand | Opcode::Nor => C_R_0_R,
            Opcode::Shl | Opcode::Shr | Opcode::Sar if self.feat.bmi2 => C_R_R_RI,
            Opcode::Shl | Opcode::Shr | Opcode::Sar | Opcode::Rotl | Opcode::Rotr => C_R_0_RI,
            Opcode::Mulsh
            | Opcode::Muluh
            | Opcode::Divs
            | Opcode::Divu
            | Opcode::Rems
            | Opcode::Remu => C_R_R_R,
            Opcode::Muls2 | Opcode::Mulu2 => C_MUL2,
            Opcode::Clz | Opcode::Ctz => C_R_R_RW,
            Opcode::Neg | Opcode::Not => C_R_0,
            Opcode::Ctpop
            | Opcode::Bswap16
            | Opcode::Bswap32
            | Opcode::Bswap64
            | Opcode::Extract
            | Opcode::Sextract
            | Opcode::ExtI32I64
            | Opcode::ExtuI32I64
            | Opcode::ExtrlI64I32
            | Opcode::ExtrhI64I32 => C_R_R,
            Opcode::Setcond | Opcode::Negsetcond => C_R_R_RET,
            Opcode::Brcond => C_R_RET,
            Opcode::Movcond => C_MOVCOND,
            Opcode::Deposit
            | Opcode::Extract2
            | Opcode::Addco
            | Opcode::Addci
            | Opcode::Addcio
            | Opcode::Addc1o
            | Opcode::Subbo
            | Opcode::Subbi
            | Opcode::Subbio
            | Opcode::Subb1o => C_R_0_R,
            Opcode::Divs2 | Opcode::Divu2 => C_R5,
            Opcode::QemuLd => C_R_R,
            Opcode::QemuLd2 => C_R_R_R,
            Opcode::QemuSt => C_RE_R,
            Opcode::QemuSt2 => C_RE_RE_R,
            Opcode::LdVec | Opcode::StVec | Opcode::DupmVec => C_X_RI,
            Opcode::DupVec => {
                if f.temp(op.arg_temp(1)).ty.is_vector() {
                    return Err(GenCodeError::Unsupported("dup_vec of a vector".into()));
                }
                C_X_R
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
            | Opcode::RotrvVec
            | Opcode::CmpVec => C_X_X_X,
            Opcode::NegVec
            | Opcode::AbsVec
            | Opcode::NotVec
            | Opcode::ShliVec
            | Opcode::ShriVec
            | Opcode::SariVec
            | Opcode::RotliVec => C_X_X,
            Opcode::ShlsVec | Opcode::ShrsVec | Opcode::SarsVec | Opcode::RotlsVec => C_X_X_R,
            Opcode::BitselVec => C_X4,
            Opcode::CmpselVec => C_X5,
            other => return Err(GenCodeError::Unsupported(other.name().to_string())),
        })
    }

    fn constraint_letter(&self, c: char) -> Option<Letter> {
        Some(match c {
            'r' => Letter::Regs(GPRS),
            'x' => Letter::Regs(VECS),
            'a' => Letter::Regs(RegSet::single(RAX)),
            'd' => Letter::Regs(RegSet::single(RDX)),
            'e' => Letter::Const(ctc::S32),
            'Z' => Letter::Const(ctc::U32),
            'W' => Letter::Const(ctc::WSZ),
            'T' => Letter::Const(ctc::TST),
            _ => return None,
        })
    }

    fn const_match(&self, val: i64, ct: u32, ty: Type, cond: Cond, _vece: u32) -> bool {
        if ct & regalloc::ct::CONST != 0 {
            return true;
        }
        let i32_ty = ty == Type::I32;
        if ct & ctc::S32 != 0 && (i32_ty || fits_i32(val)) {
            return true;
        }
        if ct & ctc::U32 != 0 && (i32_ty || val == val as u32 as i64) {
            return true;
        }
        if ct & ctc::TST != 0
            && cond.is_tst()
            && (i32_ty || val == val as u32 as i64 || fits_i32(val) || val.count_ones() == 1)
        {
            return true;
        }
        if ct & ctc::WSZ != 0 {
            let v = if i32_ty { val as u32 as i64 } else { val };
            return v == ty.bits() as i64;
        }
        false
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
        match (is_vec_reg(dst), is_vec_reg(src)) {
            (false, false) => self.a.mov(rexw(ty), dst, src),
            (true, true) => self.vmov(ty, dst, src),
            (true, false) => self.vop2(op::MOVD_VY_EY | P_REXW, Type::V128, dst, src),
            (false, true) => self.vop2(op::MOVD_EY_VY | P_REXW, Type::V128, src, dst),
        }
        true
    }

    fn out_movi(&mut self, ty: Type, dst: Reg, val: i64) {
        self.movi(ty, dst, val as u64);
    }

    fn out_dupi_vec(&mut self, ty: Type, vece: u32, dst: Reg, val: u64) {
        self.vdupi(ty, dst, dup_const(vece, val));
    }

    fn out_ld(&mut self, ty: Type, dst: Reg, base: Reg, off: i64) {
        // Offsets that do not fit are only in blocks whose check on entry always fails.
        let m = Mem::Base(base, off as i32);
        match ty {
            Type::I32 => self.a.load(dst, m, 4, false, 0),
            Type::I64 => self.a.load(dst, m, 8, false, P_REXW),
            _ => self.vld(ty, dst, m),
        }
    }

    fn out_st(&mut self, ty: Type, src: Reg, base: Reg, off: i64) {
        let m = Mem::Base(base, off as i32);
        match ty {
            Type::I32 => self.a.store(src, m, 4),
            Type::I64 => self.a.store(src, m, 8),
            _ => self.vst(ty, src, m),
        }
    }

    fn out_sti(&mut self, ty: Type, val: i64, base: Reg, off: i64) -> bool {
        let m = Mem::Base(base, off as i32);
        match ty {
            Type::I32 => self.a.store_imm(val as u64, m, 4),
            Type::I64 if fits_i32(val) => self.a.store_imm(val as u64, m, 8),
            _ => return false,
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
                self.a.jump(None, l, false);
            }
            Opcode::Mb => {
                if op.args[0] as u32 & mo::ST_LD != 0 {
                    self.a.mb();
                }
            }
            Opcode::InsnStart => {
                let mut words = [0u64; INSN_START_WORDS];
                words.copy_from_slice(&op.args[..INSN_START_WORDS]);
                self.requests.push(Request::InsnStart(words));
                self.insn = self.requests.len() as u64;
                self.insn_of.push(self.insn);
            }
            Opcode::ExitTb => self.exit_with(kind::EXIT_TB, Some(op.args[0])),
            Opcode::GotoTb => {
                // The displacement must be 4-byte aligned so that linking is one write.
                while (self.a.pos() + 1) % 4 != 0 {
                    self.a.b8(0x90);
                }
                self.a.b8(op::JMP_LONG as u8);
                let at = self.a.pos();
                self.a.b32(0);
                self.goto_tb.push((op.args[0] as u32, at, self.insn));
            }
            Opcode::Brcond => {
                let c = cond_arg(op, 2)?;
                let l = self.label(op, 3)?;
                match c {
                    Cond::Never => {}
                    Cond::Always => self.a.jump(None, l, false),
                    _ => {
                        let code = self.compare(op.ty, c, args[0] as Reg, args[1], const_args[1]);
                        self.a.jump(Some(code), l, false);
                    }
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

    fn out_of_line_branches(&self) -> bool {
        true
    }

    fn out_call(&mut self, f: &Func, op: &Op) -> R<()> {
        let info = f.helper_info(op.call_helper()).clone();
        let ni = op.calli as usize;
        let ic = self.lookup != 0 && info.name == crate::runtime::LOOKUP_TB_PTR_IC && ni == 2;
        let lookup = ic || self.lookup != 0 && info.name == crate::runtime::LOOKUP_TB_PTR;
        let pure = info.flags & ruvm_jit_core::types::call_flags::NO_SE != 0;
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
        if ic {
            self.ic_probe();
        }
        if lookup {
            let site = if ic { crate::runtime::LOOKUP_IC_SITE } else { 0 };
            self.service_via(req, self.lookup, self.insn, site);
        } else {
            self.service(req);
        }
        Ok(())
    }
}
