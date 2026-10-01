// SPDX-License-Identifier: GPL-2.0-or-later

//! Compiled blocks and running them: QEMU's `tcg_gen_code` output placement, `tcg_qemu_tb_exec`
//! and `tb_target_set_jmp_target`, for the code [`crate::codegen`] produces.
//!
//! A [`CodeRegion`] is one code buffer filled from the start, the way a QEMU region is. A
//! [`CompiledTb`] keeps its region alive, so its code cannot be unmapped while it can still run.
//!
//! Generated code is entered as an ordinary C function with the CPU state buffer, its length,
//! a run context and a slot array. Anything it cannot do itself goes through [`service`], a
//! C-ABI Rust function whose address is built into the code. The context is the only channel
//! between the two: argument words in, results and the reason for leaving out.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{Arc, Mutex};

use ruvm_jit_core::ir::{Func, HelperType};
use ruvm_jit_core::types::INSN_START_WORDS;
use ruvm_jit_interp::{
    Exit, GuestMemory, HelperEnv, HelperRegistry, InterpError, Unwind, guest_load_env,
    guest_store_env,
};

use crate::asm::i;
use crate::buffer::{BufferError, CodeBuffer};
use crate::codegen::{self, GenCodeError, INSN_OFFSET, NARGS, RET_OFFSET, Request, kind};

/// One executable buffer that blocks are compiled into, front to back.
#[derive(Debug)]
pub struct CodeRegion {
    buf: CodeBuffer,
    next: Mutex<usize>,
}

impl CodeRegion {
    /// A region of at least `size` bytes of executable memory.
    pub fn new(size: usize) -> Result<Arc<CodeRegion>, BufferError> {
        Ok(Arc::new(CodeRegion { buf: CodeBuffer::new(size)?, next: Mutex::new(0) }))
    }

    /// Bytes of the region in use.
    pub fn used(&self) -> usize {
        *self.next.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The region's size in bytes.
    pub fn size(&self) -> usize {
        self.buf.size()
    }

    /// Compile `f` into the region, `tcg_gen_code`. A block with temps wider than 128 bits is
    /// refused with [`GenCodeError::Unsupported`]; it can be run with the interpreter instead.
    pub fn compile(self: &Arc<Self>, f: &Func) -> Result<CompiledTb, GenCodeError> {
        let mut next = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let offset = next.next_multiple_of(16);
        let base = self.buf.addr() + offset as u64;
        let service_fn: extern "C" fn(&mut RunCtx<'_>, u64) -> u64 = service;
        let g = codegen::generate(f, base, service_fn as usize as u64)?;
        let end = offset.checked_add(g.bytes.len()).ok_or(GenCodeError::TooLarge)?;
        if end > self.buf.size() {
            return Err(GenCodeError::TooLarge);
        }
        self.buf.write(offset, &g.bytes).map_err(|_| GenCodeError::TooLarge)?;
        *next = end;
        Ok(CompiledTb {
            region: Arc::clone(self),
            offset,
            len: g.bytes.len(),
            requests: g.requests,
            slot_words: g.slot_words,
            goto_tb: g.goto_tb,
        })
    }
}

/// One block of generated code.
#[derive(Debug)]
pub struct CompiledTb {
    region: Arc<CodeRegion>,
    offset: usize,
    len: usize,
    requests: Vec<Request>,
    slot_words: usize,
    goto_tb: Vec<(u32, usize, u32)>,
}

impl CompiledTb {
    /// The address of the first instruction.
    pub fn addr(&self) -> u64 {
        self.region.buf.addr() + self.offset as u64
    }

    /// The generated code, instructions and literal pool.
    pub fn code(&self) -> Vec<u8> {
        self.region.buf.read(self.offset, self.len).unwrap_or_default()
    }

    /// Mark `goto_tb` slot `idx` as chained or not, `tb_target_set_jmp_target`. A chained slot
    /// leaves the block with [`Exit::GotoTb`]; an unchained one falls through, as
    /// [`ruvm_jit_interp::Machine::linked`] describes. Returns false if the block has no such
    /// slot.
    pub fn set_goto_tb_linked(&self, idx: u32, linked: bool) -> bool {
        let mut found = false;
        for &(slot, at, word) in &self.goto_tb {
            if slot == idx {
                let w = if linked { word } else { i::NOP };
                found = self.region.buf.patch_u32(self.offset + at, w).is_ok();
            }
        }
        found
    }

    /// Run the block once against `env` and `mem`, as [`ruvm_jit_interp::Machine::run`] does.
    ///
    /// Generated code has no step limit: a block that loops forever does not return.
    /// Accesses at constant offsets from `env` are checked when the block is entered, so a
    /// block with one beyond the end of `env` fails before it does anything.
    pub fn run(
        &self,
        env: &mut [u8],
        mem: &mut dyn GuestMemory,
        helpers: &HelperRegistry,
    ) -> Result<Exit, InterpError> {
        let mut last = None;
        self.run_traced(env, mem, helpers, &mut last)
    }

    /// [`CompiledTb::run`], also reporting the words of the last `insn_start` executed, as
    /// [`ruvm_jit_interp::Machine::last_insn_start`] does.
    pub fn run_traced(
        &self,
        env: &mut [u8],
        mem: &mut dyn GuestMemory,
        helpers: &HelperRegistry,
        last_insn_start: &mut Option<[u64; INSN_START_WORDS]>,
    ) -> Result<Exit, InterpError> {
        let mut slots = vec![0u64; self.slot_words];
        let env_len = env.len();
        let mut ctx = RunCtx {
            args: [0; NARGS],
            ret: 0,
            insn: 0,
            insn_delivered: 0,
            requests: &self.requests,
            env: env.as_mut_ptr(),
            env_len,
            mem,
            helpers,
            insn_start: None,
            unwind: None,
            error: None,
            panic: None,
        };
        let k = enter(self.addr(), ctx.env, env_len, &mut ctx, slots.as_mut_ptr())?;
        deliver_insn_start(&mut ctx);
        *last_insn_start = ctx.insn_start;
        if let Some(p) = ctx.panic.take() {
            resume_unwind(p);
        }
        let ret = ctx.ret;
        match k {
            kind::EXIT_TB => Ok(Exit::ExitTb(ret)),
            kind::GOTO_TB => Ok(Exit::GotoTb(ret as u32)),
            kind::GOTO_PTR => Ok(Exit::GotoPtr(ret)),
            kind::UNWIND => ctx
                .unwind
                .map(Exit::Unwind)
                .ok_or_else(|| InterpError::BadOp("unwind without a reason".into())),
            kind::ERROR => Err(ctx
                .error
                .unwrap_or_else(|| InterpError::BadOp("error without a reason".into()))),
            kind::BOUNDS => {
                Err(InterpError::EnvOutOfBounds { offset: ret, len: ctx.args[0] as usize })
            }
            kind::FELL_OFF => Err(InterpError::FellOffEnd),
            other => Err(InterpError::BadOp(format!("generated code returned kind {other}"))),
        }
    }
}

/// The state shared by generated code and [`service`] during one run. Generated code only
/// touches `args`, `ret` and `insn`, at the offsets [`codegen`] uses.
#[repr(C)]
pub(crate) struct RunCtx<'a> {
    args: [u64; NARGS],
    ret: u64,
    /// One more than the index of the request of the last `insn_start` executed, or 0.
    insn: u64,
    /// The value of `insn` last reported to the guest memory.
    insn_delivered: u64,
    requests: &'a [Request],
    /// The CPU state, as a pointer because generated code writes it between service calls.
    env: *mut u8,
    env_len: usize,
    mem: &'a mut dyn GuestMemory,
    helpers: &'a HelperRegistry,
    insn_start: Option<[u64; INSN_START_WORDS]>,
    unwind: Option<Unwind>,
    error: Option<InterpError>,
    panic: Option<Box<dyn Any + Send>>,
}

const _: () = assert!(std::mem::offset_of!(RunCtx<'static>, ret) == RET_OFFSET as usize);
const _: () = assert!(std::mem::offset_of!(RunCtx<'static>, insn) == INSN_OFFSET as usize);

/// Report the last `insn_start` generated code stored, if it is new, to the context and the
/// guest memory, as the interpreter does when it executes one.
fn deliver_insn_start(ctx: &mut RunCtx<'_>) {
    if ctx.insn == ctx.insn_delivered {
        return;
    }
    ctx.insn_delivered = ctx.insn;
    let at = (ctx.insn as usize).wrapping_sub(1);
    if let Some(Request::InsnStart(w)) = ctx.requests.get(at) {
        ctx.insn_start = Some(*w);
        ctx.mem.insn_start(w);
    }
}

/// Call generated code at `addr`.
#[cfg(all(unix, target_arch = "aarch64"))]
fn enter(
    addr: u64,
    env: *mut u8,
    env_len: usize,
    ctx: &mut RunCtx<'_>,
    slots: *mut u64,
) -> Result<u64, InterpError> {
    type Entry = unsafe extern "C" fn(*mut u8, u64, *mut RunCtx<'_>, *mut u64) -> u64;
    // SAFETY: `addr` is the start of a block that `CodeRegion::compile` generated and wrote,
    // and the region stays mapped because `self` holds it. The block follows the AAPCS64 C
    // calling convention: it saves and restores every callee-saved register it uses and keeps
    // the stack 16-byte aligned. It reads and writes only `env_len` bytes at `env` (every
    // access is bounds checked against that length), the `slot_words` words at `slots` that
    // the caller allocated, and the argument and return words at the start of `ctx`; it calls
    // only `service`, with `ctx`. `env` comes from a `&mut [u8]` the caller holds for the whole
    // call, and nothing else uses these pointers until the block returns.
    let k = unsafe {
        let f = std::mem::transmute::<usize, Entry>(addr as usize);
        f(env, env_len as u64, ctx, slots)
    };
    Ok(k)
}

/// Generated code can only run on an AArch64 host.
#[cfg(not(all(unix, target_arch = "aarch64")))]
fn enter(
    _addr: u64,
    _env: *mut u8,
    _env_len: usize,
    _ctx: &mut RunCtx<'_>,
    _slots: *mut u64,
) -> Result<u64, InterpError> {
    Err(InterpError::BadOp("aarch64 code cannot run on this host".into()))
}

/// The one Rust entry point of generated code. Returns 0 to continue, or the kind to leave
/// the block with.
extern "C" fn service(ctx: &mut RunCtx<'_>, req: u64) -> u64 {
    match catch_unwind(AssertUnwindSafe(|| serve(ctx, req))) {
        Ok(Ok(())) => 0,
        Ok(Err(Leave::Unwind(u))) => {
            ctx.unwind = Some(u);
            kind::UNWIND
        }
        Ok(Err(Leave::Error(e))) => {
            ctx.error = Some(e);
            kind::ERROR
        }
        // A panic must not unwind into generated code; it is resumed once the block returns.
        Err(p) => {
            ctx.panic = Some(p);
            kind::ERROR
        }
    }
}

enum Leave {
    Unwind(Unwind),
    Error(InterpError),
}

fn serve(ctx: &mut RunCtx<'_>, req: u64) -> Result<(), Leave> {
    deliver_insn_start(ctx);
    let r = ctx
        .requests
        .get(req as usize)
        .ok_or_else(|| Leave::Error(InterpError::BadOp(format!("unknown request {req}"))))?;
    // SAFETY: `env` and `env_len` come from the `&mut [u8]` that `run_traced` holds for the
    // whole run. Generated code is suspended in this call and touches the buffer again only
    // after it returns, and this slice does not outlive the call, so it is the only live
    // access to the buffer.
    let env = unsafe { std::slice::from_raw_parts_mut(ctx.env, ctx.env_len) };
    match r {
        Request::InsnStart(words) => {
            ctx.insn_start = Some(*words);
            ctx.mem.insn_start(words);
        }
        Request::Call { name, ret, args, nin } => {
            let entry = ctx
                .helpers
                .get(name)
                .ok_or_else(|| Leave::Error(InterpError::UnknownHelper(name.clone())))?;
            if entry.ret != *ret || entry.args != *args {
                return Err(Leave::Error(InterpError::HelperSignature(name.clone())));
            }
            let inputs: Vec<u64> = ctx.args[..*nin].to_vec();
            let mut he = HelperEnv { env, mem: &mut *ctx.mem };
            let v = (entry.f)(&mut he, &inputs).map_err(Leave::Unwind)?;
            if *ret != HelperType::Void {
                ctx.args[0] = v as u64;
                ctx.args[1] = (v >> 64) as u64;
            }
        }
        Request::Load(oi) => {
            let v = guest_load_env(&mut *ctx.mem, env, ctx.args[0], *oi)
                .map_err(|e| Leave::Unwind(Unwind::Mem(e)))?;
            ctx.args[0] = v as u64;
            ctx.args[1] = (v >> 64) as u64;
        }
        Request::Store(oi) => {
            let v = ctx.args[0] as u128 | (ctx.args[1] as u128) << 64;
            guest_store_env(&mut *ctx.mem, env, ctx.args[2], v, *oi)
                .map_err(|e| Leave::Unwind(Unwind::Mem(e)))?;
        }
        Request::Div2 { signed, bits } => {
            let (q, r) = div2(*signed, *bits, ctx.args[0], ctx.args[1], ctx.args[2]);
            ctx.args[0] = q;
            ctx.args[1] = r;
        }
    }
    Ok(())
}

fn sext(v: u64, bits: u32) -> u64 {
    if bits >= 64 { v } else { (((v << (64 - bits)) as i64) >> (64 - bits)) as u64 }
}

/// `divs2` and `divu2` exactly as the interpreter does them: a zero divisor divides by one,
/// and the results are truncated to `bits` by the stores that follow.
fn div2(signed: bool, bits: u32, lo: u64, hi: u64, d: u64) -> (u64, u64) {
    let m = if bits == 32 { 0xffff_ffff } else { u64::MAX };
    let (lo, hi, d) = (lo & m, hi & m, d & m);
    if signed {
        let n = ((sext(hi, bits) as i64 as i128) << bits) | lo as i128;
        let mut d = sext(d, bits) as i64 as i128;
        if d == 0 {
            d = 1;
        }
        (n.wrapping_div(d) as u64, n.wrapping_rem(d) as u64)
    } else {
        let n = (hi as u128) << bits | lo as u128;
        let d = d.max(1) as u128;
        ((n / d) as u64, (n % d) as u64)
    }
}

impl std::fmt::Debug for RunCtx<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunCtx").field("ret", &self.ret).finish_non_exhaustive()
    }
}
