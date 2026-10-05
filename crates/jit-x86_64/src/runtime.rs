// SPDX-License-Identifier: GPL-2.0-or-later

//! Compiled blocks and running them: QEMU's `tcg_gen_code` output placement, `tcg_qemu_tb_exec`
//! and `tb_target_set_jmp_target`, for the code [`crate::codegen`] produces.
//!
//! A [`CodeRegion`] is one code buffer filled from the start, the way a QEMU region is. A
//! [`CompiledTb`] keeps its region alive, so its code cannot be unmapped while it can still run.
//!
//! Generated code is entered as an ordinary C function with the CPU state buffer, a run
//! context (which also holds the length of the buffer) and a slot array. Anything it cannot do
//! itself goes through [`service`], a C-ABI Rust function whose address is built into the code.
//! The context is the only channel between the two: argument words in, results and the reason
//! for leaving out.
//!
//! [`CompiledTb::run_chained`] runs a block and whatever it chains to, the way
//! `tcg_qemu_tb_exec()` does. Linked `goto_tb` jumps go straight to the next block's code (see
//! [`CompiledTb::set_goto_tb_target`]). After `lookup_tb_ptr` the service routine asks the
//! [`Chain`] for the next block and, when its code is in a region this one may reach, tells
//! the code it may jump there. Each request passes the address of the [`BlockMeta`] of the
//! block that made it, so requests are served against that block and the guest memory is told
//! when another block starts ([`GuestMemory::enter_block`]). A block stores the address in the
//! run context only when it leaves, so that the runtime knows which block left. A
//! `lookup_tb_ptr` request is the exception: like `helper_lookup_tb_ptr()` it only runs at the
//! end of a block, so the guest memory is not told about its block or instruction.
//!
//! A chained jump enters past the static bounds check of the next block when the check is known
//! to pass: for `goto_tb` when the next block needs no more CPU state than the one jumping, for
//! `goto_ptr` when it needs no more than the run has.
//!
//! Helpers named by a block are resolved when it is compiled, so serving a call is an indexed
//! load and an indirect call, the way QEMU's generated code calls the helper directly.
//!
//! As in QEMU, generated code does not record which instruction it is running. The instruction
//! of a request is known when the block is compiled ([`BlockMeta::insn_of`]), the way QEMU
//! finds it from the host return address with the block's search data.
//!
//! Regions form chains. A region started because the one before was full keeps that one
//! mapped, so blocks may jump back into it but never forward. A new chain starts after a flush,
//! and blocks never jump between chains.
//!
//! The slot array is one buffer per thread, [`MAX_SLOT_WORDS`] long, reused by every run, so
//! running a block does not allocate. Blocks do not expect it to be zeroed.
//!
//! Differences from QEMU:
//!
//! - QEMU has one code buffer split into regions and invalidates jumps into it on a flush;
//!   here region chains are separate mappings, freed when the last block of a chain is gone.
//! - A block needing more than [`MAX_SLOT_WORDS`] slot words is refused; QEMU's frame size is
//!   fixed too, at `TCG_STATIC_FRAME_SIZE`, and it refuses such blocks the same way.
//! - Inlined softmmu lookups ([`CompileOptions::tlb_page_bits`]) find the TLB descriptor through
//!   the run context, not at a fixed offset from `env`. A run whose guest memory has no TLB of
//!   the chain's page size gives them a descriptor that never matches, so every access takes
//!   the slow path through the service routine.
//! - `lookup_and_goto_ptr` calls its own lookup routine, which finds the block through a
//!   per-CPU cache that also keeps the address to jump to, so a hit costs no reference count
//!   and no chain check; QEMU's `helper_lookup_tb_ptr` reads `tb->tc.ptr` from its jump cache.
//! - A call to a helper with `TCG_CALL_NO_SE` does not tell the guest memory which block and
//!   instruction it is in, since such a helper cannot raise an exception; the next request
//!   that may raise one, or the end of the run, does. QEMU needs no such step because it
//!   finds the instruction from the host return address only when it unwinds.

use std::any::Any;
use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use ruvm_jit_core::ir::{Func, HelperType};
use ruvm_jit_core::types::INSN_START_WORDS;
use ruvm_jit_interp::{
    Exit, FastTlb, GuestMemory, HelperEnv, HelperFn, HelperRegistry, InterpError, Unwind,
    guest_load_env, guest_store_env,
};

use crate::buffer::{BufferError, CodeBuffer};
use crate::codegen::{
    self, DECR_OFFSET, ENV_LEN_OFFSET, GOTO_PTR_OK_OFFSET, GenCodeError, GenOptions, INSN_OFFSET,
    META_OFFSET, NARGS, RET_OFFSET, Request, TLB_OFFSET, kind,
};
use crate::features::HostFeatures;

/// The most 64-bit words of slot array a block may use, so that every block of a chain fits
/// the one slot array of the thread. A block that needs more is refused with
/// [`GenCodeError::TooLarge`].
pub const MAX_SLOT_WORDS: usize = 1 << 16;

/// The name of the helper whose result `goto_ptr` jumps to.
pub(crate) const LOOKUP_TB_PTR: &str = "lookup_tb_ptr";

/// The blocks a run may chain to and how to find them, for [`CompiledTb::run_chained`]. This
/// is the runtime side of `lookup_and_goto_ptr`.
pub trait Chain {
    /// `helper_lookup_tb_ptr()`: the block to go on with after a `goto_ptr`, as the value given
    /// to [`CompiledTb::set_owner`], or `None` to leave generated code.
    fn lookup_tb_ptr(
        &self,
        he: &mut HelperEnv<'_>,
    ) -> Result<Option<Arc<dyn Any + Send + Sync>>, Unwind>;

    /// The compiled code of a block [`Chain::lookup_tb_ptr`] returned, if it has any.
    fn code<'a>(&self, block: &'a (dyn Any + Send + Sync)) -> Option<&'a CompiledTb>;

    /// [`Chain::lookup_tb_ptr`] and [`Chain::code`] in one, which is what the service routine
    /// calls. `jump` is given the code of the block found and returns the address generated
    /// code may jump to, or `None` if it may not jump there. What it returns only depends on
    /// that code and `key`, so an implementation may remember an address `jump` gave for a
    /// block and `key` and use it again for the same block and key without calling `jump`;
    /// a key of `[0, 0]` must not be remembered. The default calls the other two methods; an
    /// implementation can override it so as not to take a reference to the block when the
    /// code jumps.
    fn lookup_code(
        &self,
        he: &mut HelperEnv<'_>,
        key: [u64; 2],
        jump: &mut dyn FnMut(&CompiledTb) -> Option<u64>,
    ) -> Result<Found, Unwind> {
        let _ = key;
        let Some(block) = self.lookup_tb_ptr(he)? else { return Ok(Found::Leave(None)) };
        match self.code(&*block).and_then(jump) {
            Some(entry) => Ok(Found::Jump(entry)),
            None => Ok(Found::Leave(Some(block))),
        }
    }
}

/// What [`Chain::lookup_code`] found.
#[derive(Debug)]
pub enum Found {
    /// Generated code jumps to this address.
    Jump(u64),
    /// Generated code leaves. The block found, if there was one, runs next.
    Leave(Option<Arc<dyn Any + Send + Sync>>),
}

/// What [`CompiledTb::run_chained`] ended with.
#[derive(Debug)]
pub struct ChainExit {
    /// How the code left.
    pub exit: Exit,
    /// The block that was running when the code left, as given to [`CompiledTb::set_owner`].
    pub block: Option<Arc<dyn Any + Send + Sync>>,
    /// After [`Exit::GotoPtr`] with 0: the block [`Chain::lookup_tb_ptr`] found but generated
    /// code cannot jump to, because it has no compiled code or lives in a region the run may
    /// not reach. The caller should run it next.
    pub next: Option<Arc<dyn Any + Send + Sync>>,
    /// The words of the last `insn_start` executed.
    pub last_insn_start: Option<[u64; INSN_START_WORDS]>,
}

/// Numbers the region chains, so that blocks only link to blocks of their own chain.
static NEXT_CHAIN: AtomicU64 = AtomicU64::new(1);

/// What the service routine needs to know about a block: its requests, with the helpers they
/// call already looked up, and who owns it. A block stores the address of its `BlockMeta` in
/// the run context when it starts. The region owns it, so it lives as long as the code does.
#[derive(Debug, Default)]
pub(crate) struct BlockMeta {
    requests: Vec<Request>,
    /// For each request, one more than the index of the `insn_start` request of its instruction,
    /// or 0.
    insn_of: Vec<u64>,
    /// For each request, what the service routine can do without looking at the request.
    fast: Vec<Fast>,
    owner: OnceLock<Weak<dyn Any + Send + Sync>>,
}

/// How the service routine serves a request, decided when the block is compiled.
#[derive(Clone, Copy, Debug)]
enum Fast {
    /// A call to this helper, found in the registry with the right signature, with `nin`
    /// argument words; `ret` says whether it returns a value and `pure` whether the helper is
    /// `TCG_CALL_NO_SE`.
    Helper { f: HelperFn, nin: usize, ret: bool, pure: bool },
    /// A call to `lookup_tb_ptr`.
    Lookup,
    /// Anything else.
    Other,
}

impl BlockMeta {
    fn owner(&self) -> Option<Arc<dyn Any + Send + Sync>> {
        self.owner.get().and_then(Weak::upgrade)
    }
}

/// One executable buffer that blocks are compiled into, front to back.
#[derive(Debug)]
pub struct CodeRegion {
    buf: CodeBuffer,
    next: Mutex<usize>,
    features: HostFeatures,
    /// The region before this one in its chain, kept mapped as long as this one is, so that
    /// blocks here can jump to blocks there.
    _prev: Option<Arc<CodeRegion>>,
    /// The chain, and the place of this region in it, counting from 0.
    chain: u64,
    seq: u64,
    /// The metadata of every block compiled here.
    metas: Mutex<Vec<Arc<BlockMeta>>>,
    /// The guest page size, as log2, of the TLB that blocks of the chain inline lookups in,
    /// set by the first block compiled with [`CompileOptions::tlb_page_bits`]. Blocks of a
    /// chain jump to each other with one run context, so they all read one TLB.
    tlb_page_bits: Arc<OnceLock<u32>>,
}

/// How to compile a block, for [`CodeRegion::compile_with`].
#[derive(Clone, Copy, Debug, Default)]
pub struct CompileOptions<'a> {
    /// The helpers the block will call. Calls to helpers found here with the right signature
    /// skip the lookup by name when they run.
    pub helpers: Option<&'a HelperRegistry>,
    /// A 32-bit load at this constant offset from `env` reads the `icount_decr` word given to
    /// [`CompiledTb::run_chained`] instead of the CPU state, as the check at the start of each
    /// block must see exit requests from other threads without leaving generated code.
    pub icount_decr_offset: Option<i64>,
    /// Inline the softmmu TLB lookup of `qemu_ld` and `qemu_st`, as QEMU's
    /// `tcg_out_qemu_ld` does, for a TLB of pages of `1 << tlb_page_bits` bytes: the
    /// [`GuestMemory::fast_tlb`] of the memory the block runs with. When the memory has no
    /// such TLB, every access takes the slow path. All blocks of a region chain must give the
    /// same value; a block that gives another is refused with [`GenCodeError::Unsupported`].
    pub tlb_page_bits: Option<u32>,
}

impl CodeRegion {
    /// A region of at least `size` bytes of executable memory, generating code for the
    /// features of the host it runs on, as QEMU's `tcg_target_init` detects them.
    pub fn new(size: usize) -> Result<Arc<CodeRegion>, BufferError> {
        CodeRegion::with_features(size, HostFeatures::detect())
    }

    /// A region that generates code for `features` only. Features the host lacks must not be
    /// asked for: the code would fault with an illegal instruction.
    pub fn with_features(
        size: usize,
        features: HostFeatures,
    ) -> Result<Arc<CodeRegion>, BufferError> {
        Ok(Arc::new(CodeRegion {
            buf: CodeBuffer::new(size)?,
            next: Mutex::new(0),
            features: features.normalized(),
            _prev: None,
            chain: NEXT_CHAIN.fetch_add(1, Ordering::Relaxed),
            seq: 0,
            metas: Mutex::new(Vec::new()),
            tlb_page_bits: Arc::new(OnceLock::new()),
        }))
    }

    /// A region of `size` bytes that continues the chain of this one, for when this one is
    /// full. Blocks compiled into it can chain to blocks of this region and the ones before
    /// it, which it keeps mapped.
    pub fn successor(self: &Arc<Self>, size: usize) -> Result<Arc<CodeRegion>, BufferError> {
        Ok(Arc::new(CodeRegion {
            buf: CodeBuffer::new(size)?,
            next: Mutex::new(0),
            features: self.features,
            _prev: Some(Arc::clone(self)),
            chain: self.chain,
            seq: self.seq + 1,
            metas: Mutex::new(Vec::new()),
            tlb_page_bits: Arc::clone(&self.tlb_page_bits),
        }))
    }

    /// The instruction set extensions the region generates code for.
    pub fn features(&self) -> HostFeatures {
        self.features
    }

    /// Bytes of the region in use.
    pub fn used(&self) -> usize {
        *self.next.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The region's size in bytes.
    pub fn size(&self) -> usize {
        self.buf.size()
    }

    /// Whether code running in this region may jump into `other`: it is this region or one
    /// before it in the chain, which this one keeps mapped.
    fn reaches(&self, other: &CodeRegion) -> bool {
        self.chain == other.chain && other.seq <= self.seq
    }

    /// Compile `f` into the region, `tcg_gen_code`. A block with 256-bit temps is refused with
    /// [`GenCodeError::Unsupported`] when the region does not use AVX2; it can be run with the
    /// interpreter instead.
    pub fn compile(self: &Arc<Self>, f: &Func) -> Result<CompiledTb, GenCodeError> {
        self.compile_with(f, &CompileOptions::default())
    }

    /// [`CodeRegion::compile`] with `opts`.
    pub fn compile_with(
        self: &Arc<Self>,
        f: &Func,
        opts: &CompileOptions<'_>,
    ) -> Result<CompiledTb, GenCodeError> {
        if let Some(bits) = opts.tlb_page_bits {
            if *self.tlb_page_bits.get_or_init(|| bits) != bits {
                return Err(GenCodeError::Unsupported(format!(
                    "TLB page bits {bits} in a chain of {}",
                    self.tlb_page_bits.get().copied().unwrap_or(0)
                )));
            }
        }
        let mut meta = Arc::new(BlockMeta::default());
        let gen_opts = GenOptions {
            meta: Arc::as_ptr(&meta) as usize as u64,
            icount_decr: opts.icount_decr_offset,
            tlb_page_bits: opts.tlb_page_bits,
            lookup: {
                let f: extern "C" fn(&mut RunCtx<'_>, u64, *const BlockMeta) -> u64 =
                    lookup_service;
                f as usize as u64
            },
        };
        let mut next = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let offset = next.next_multiple_of(16);
        let base = self.buf.addr() + offset as u64;
        let service_fn: extern "C" fn(&mut RunCtx<'_>, u64, *const BlockMeta) -> u64 = service;
        let g = codegen::generate(f, base, service_fn as usize as u64, self.features, &gen_opts)?;
        if g.slot_words > MAX_SLOT_WORDS {
            return Err(GenCodeError::TooLarge);
        }
        let end = offset.checked_add(g.bytes.len()).ok_or(GenCodeError::TooLarge)?;
        if end > self.buf.size() {
            return Err(GenCodeError::TooLarge);
        }
        let m = Arc::get_mut(&mut meta).expect("the metadata is not shared yet");
        m.fast = g
            .requests
            .iter()
            .map(|r| match (r, opts.helpers) {
                (Request::Call { name, .. }, _) if name == LOOKUP_TB_PTR => Fast::Lookup,
                (Request::Call { name, ret, args, nin, pure }, Some(reg)) => reg
                    .get(name)
                    .filter(|e| e.ret == *ret && e.args == *args)
                    .map_or(Fast::Other, |e| Fast::Helper {
                        f: e.f,
                        nin: *nin,
                        ret: *ret != HelperType::Void,
                        pure: *pure,
                    }),
                _ => Fast::Other,
            })
            .collect();
        m.requests = g.requests;
        m.insn_of = g.insn_of;
        self.buf.write(offset, &g.bytes).map_err(|_| GenCodeError::TooLarge)?;
        *next = end;
        drop(next);
        self.metas.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::clone(&meta));
        Ok(CompiledTb {
            region: Arc::clone(self),
            offset,
            len: g.bytes.len(),
            meta,
            body: g.body,
            fast_body: g.fast_body,
            env_need: g.env_need,
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
    meta: Arc<BlockMeta>,
    body: usize,
    fast_body: usize,
    env_need: u64,
    goto_tb: Vec<(u32, usize, u32)>,
}

thread_local! {
    /// The slot array of the blocks this thread runs, kept between runs so that running a block
    /// does not allocate. Blocks do not expect it to be zeroed.
    static SLOTS: Cell<Option<Box<[u64]>>> = const { Cell::new(None) };
}

impl CompiledTb {
    /// The address of the first instruction.
    pub fn addr(&self) -> u64 {
        self.region.buf.addr() + self.offset as u64
    }

    /// The address chained jumps enter the block at, after the prologue.
    fn entry(&self) -> u64 {
        self.addr() + self.body as u64
    }

    /// Where a chained jump from code running against `env_len` bytes of CPU state enters: past
    /// the static bounds check when the block is known to pass it.
    fn chain_entry(&self, env_len: u64) -> u64 {
        if self.env_need <= env_len { self.addr() + self.fast_body as u64 } else { self.entry() }
    }

    /// The size of the generated code in bytes.
    pub fn size(&self) -> usize {
        self.len
    }

    /// The generated code, instructions and constant pool.
    pub fn code(&self) -> Vec<u8> {
        self.region.buf.read(self.offset, self.len).unwrap_or_default()
    }

    /// Record who owns this block, the value [`CompiledTb::run_chained`] reports blocks by.
    /// Only the first call has an effect.
    pub fn set_owner(&self, owner: Weak<dyn Any + Send + Sync>) {
        let _ = self.meta.owner.set(owner);
    }

    /// Mark `goto_tb` slot `idx` as chained or not, `tb_target_set_jmp_target`. A chained slot
    /// leaves the block with [`Exit::GotoTb`]; an unchained one falls through, as
    /// [`ruvm_jit_interp::Machine::linked`] describes. Returns false if the block has no such
    /// slot.
    pub fn set_goto_tb_linked(&self, idx: u32, linked: bool) -> bool {
        self.patch_goto_tb(idx, |_, stub| if linked { stub } else { 0 })
    }

    /// Point `goto_tb` slot `idx` at `dest`, `tb_target_set_jmp_target`: the jump goes
    /// straight to the code of `dest` when this block may reach it (same chain of regions, not
    /// a later region, and in range of a 32-bit displacement), and otherwise leaves the block
    /// with [`Exit::GotoTb`]. `None` unlinks the slot, so that it falls through. Returns false
    /// if the block has no such slot.
    pub fn set_goto_tb_target(&self, idx: u32, dest: Option<&CompiledTb>) -> bool {
        let base = self.addr();
        self.patch_goto_tb(idx, |at, stub| match dest {
            None => 0,
            Some(d) if self.region.reaches(&d.region) => {
                let disp = d.chain_entry(self.env_need) as i64 - (base as i64 + at as i64 + 4);
                if disp == disp as i32 as i64 { disp as i32 as u32 } else { stub }
            }
            Some(_) => stub,
        })
    }

    fn patch_goto_tb(&self, idx: u32, word: impl Fn(usize, u32) -> u32) -> bool {
        let mut found = false;
        for &(slot, at, stub) in &self.goto_tb {
            if slot == idx {
                found = self.region.buf.patch_u32(self.offset + at, word(at, stub)).is_ok();
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
        let mut slots = SLOTS.take().unwrap_or_else(|| vec![0; MAX_SLOT_WORDS].into());
        let r = self.run_in(env, mem, helpers, None, &mut slots, last_insn_start);
        SLOTS.set(Some(slots));
        Ok(r?.exit)
    }

    /// Run the block and every block it chains to, without coming back here in between, as
    /// `cpu_tb_exec()` runs `tcg_qemu_tb_exec()`. Linked `goto_tb` slots jump to the next
    /// block's code, and `goto_ptr` jumps to the code of the block `chain` looks up. The
    /// `icount_decr` loads of blocks compiled with
    /// [`CompileOptions::icount_decr_offset`] read `icount_decr`.
    pub fn run_chained(
        &self,
        env: &mut [u8],
        mem: &mut dyn GuestMemory,
        helpers: &HelperRegistry,
        chain: &dyn Chain,
        icount_decr: &AtomicU32,
    ) -> Result<ChainExit, InterpError> {
        let mut slots = SLOTS.take().unwrap_or_else(|| vec![0; MAX_SLOT_WORDS].into());
        let mut last = None;
        let r = self.run_in(env, mem, helpers, Some((chain, icount_decr)), &mut slots, &mut last);
        SLOTS.set(Some(slots));
        r
    }

    fn run_in(
        &self,
        env: &mut [u8],
        mem: &mut dyn GuestMemory,
        helpers: &HelperRegistry,
        chain: Option<(&dyn Chain, &AtomicU32)>,
        slots: &mut [u64],
        last_insn_start: &mut Option<[u64; INSN_START_WORDS]>,
    ) -> Result<ChainExit, InterpError> {
        // Chained blocks share the slots, so there must be room for any block.
        assert!(slots.len() >= MAX_SLOT_WORDS);
        let env_len = env.len();
        // Blocks compiled without `icount_decr_offset` never read this.
        let local_decr = AtomicU32::new(0);
        let decr = chain.map_or(&local_decr, |c| c.1);
        let chain = chain.map(|c| c.0);
        let meta = Arc::as_ptr(&self.meta);
        // Blocks of this chain that inline TLB lookups read the TLB of `mem` when it has pages
        // of the size they were compiled for, and otherwise a descriptor that never matches.
        let bits = self.region.tlb_page_bits.get().copied();
        let fast = mem.fast_tlb().filter(|t| Some(t.page_bits()) == bits);
        let _run = fast.as_ref().map(|t| t.enter_run());
        let tlb = match &fast {
            Some(t) => t.desc().as_ptr(),
            None => FastTlb::miss_desc().as_ptr(),
        };
        let mut ctx = RunCtx {
            args: [0; NARGS],
            ret: 0,
            insn: 0,
            env_len,
            meta,
            decr: decr.as_ptr(),
            goto_ptr_ok: 0,
            tlb,
            jump_key: jump_key(&self.region, env_len as u64),
            meta_seen: meta,
            insn_delivered: 0,
            region: &self.region,
            env: env.as_mut_ptr(),
            mem,
            helpers,
            chain,
            next: None,
            insn_start: None,
            unwind: None,
            error: None,
            panic: None,
        };
        let k = enter(self.addr(), ctx.env, &mut ctx, slots.as_mut_ptr())?;
        deliver_insn_start(&mut ctx);
        *last_insn_start = ctx.insn_start;
        if let Some(p) = ctx.panic.take() {
            resume_unwind(p);
        }
        let ret = ctx.ret;
        let exit = match k {
            kind::EXIT_TB => Exit::ExitTb(ret),
            kind::GOTO_TB => Exit::GotoTb(ret as u32),
            kind::GOTO_PTR => Exit::GotoPtr(ret),
            kind::UNWIND => Exit::Unwind(
                ctx.unwind.ok_or_else(|| InterpError::BadOp("unwind without a reason".into()))?,
            ),
            kind::ERROR => {
                return Err(ctx
                    .error
                    .unwrap_or_else(|| InterpError::BadOp("error without a reason".into())));
            }
            kind::BOUNDS => {
                return Err(InterpError::EnvOutOfBounds { offset: ret, len: ctx.args[0] as usize });
            }
            kind::FELL_OFF => return Err(InterpError::FellOffEnd),
            other => {
                return Err(InterpError::BadOp(format!("generated code returned kind {other}")));
            }
        };
        Ok(ChainExit {
            exit,
            block: ctx.meta_ref().owner(),
            next: ctx.next.take(),
            last_insn_start: ctx.insn_start,
        })
    }
}

/// The state shared by generated code and [`service`] during one run. Generated code only
/// touches the fields up to `tlb`, at the offsets [`codegen`] uses.
#[repr(C)]
pub(crate) struct RunCtx<'a> {
    args: [u64; NARGS],
    ret: u64,
    /// One more than the index of the `insn_start` request of the instruction the last request
    /// or exit was made in, or 0.
    insn: u64,
    /// The length of the CPU state, which generated code compares offsets against.
    env_len: usize,
    /// The metadata of the block that made the last request, or that left.
    meta: *const BlockMeta,
    /// The `icount_decr` word.
    decr: *const u32,
    /// The one address `goto_ptr` may jump to, or 0 for none.
    goto_ptr_ok: u64,
    /// The TLB descriptor inlined lookups read.
    tlb: *const AtomicU64,
    /// The key [`Chain::lookup_code`] is given, see [`jump_key`].
    jump_key: [u64; 2],
    /// The block the service routine last told the guest memory about.
    meta_seen: *const BlockMeta,
    /// The value of `insn` last reported to the guest memory.
    insn_delivered: u64,
    /// The region of the first block. Chained code only reaches it and the regions before it.
    region: &'a CodeRegion,
    /// The CPU state, as a pointer because generated code writes it between service calls.
    env: *mut u8,
    mem: &'a mut dyn GuestMemory,
    helpers: &'a HelperRegistry,
    chain: Option<&'a dyn Chain>,
    /// See [`ChainExit::next`].
    next: Option<Arc<dyn Any + Send + Sync>>,
    insn_start: Option<[u64; INSN_START_WORDS]>,
    unwind: Option<Unwind>,
    error: Option<InterpError>,
    panic: Option<Box<dyn Any + Send>>,
}

const _: () = assert!(std::mem::offset_of!(RunCtx<'static>, ret) == RET_OFFSET as usize);
const _: () = assert!(std::mem::offset_of!(RunCtx<'static>, insn) == INSN_OFFSET as usize);
const _: () = assert!(std::mem::offset_of!(RunCtx<'static>, env_len) == ENV_LEN_OFFSET as usize);
const _: () = assert!(std::mem::offset_of!(RunCtx<'static>, meta) == META_OFFSET as usize);
const _: () = assert!(std::mem::offset_of!(RunCtx<'static>, decr) == DECR_OFFSET as usize);
const _: () =
    assert!(std::mem::offset_of!(RunCtx<'static>, goto_ptr_ok) == GOTO_PTR_OK_OFFSET as usize);
const _: () = assert!(std::mem::offset_of!(RunCtx<'static>, tlb) == TLB_OFFSET as usize);

impl RunCtx<'_> {
    /// The metadata of the running block.
    fn meta_ref(&self) -> &BlockMeta {
        meta_of(self.meta)
    }
}

/// The metadata at `meta`, the address a block passed or stored in [`RunCtx::meta`]. The reference is not
/// tied to the context, so that a request can be used while the context is changed.
fn meta_of<'m>(meta: *const BlockMeta) -> &'m BlockMeta {
    // SAFETY: `meta` is either the metadata of the first block, which `run_in` borrows
    // through its `CompiledTb`, or the address a block that generated code jumped to passed
    // with a request or stored when it left. Generated code only jumps to blocks in the first block's region or
    // the regions before it in its chain (`set_goto_tb_target` and `serve_lookup` check
    // this), every block passes and stores the address of a `BlockMeta` its region holds in
    // `metas`,
    // and the first block's region keeps itself and the regions before it alive for the
    // whole run. The metadata is never written after the block is compiled.
    // The reference is only used during the run, inside `run_in`.
    unsafe { &*meta }
}

/// Report the `insn_start` of the instruction of the last request or exit, if it is new, to the
/// context and the guest memory, as the interpreter does when it executes one. When a chained block is
/// running, tell the guest memory about it first.
fn deliver_insn_start(ctx: &mut RunCtx<'_>) {
    if ctx.meta != ctx.meta_seen {
        ctx.meta_seen = ctx.meta;
        ctx.insn_delivered = 0;
        ctx.insn_start = None;
        let owner = ctx.meta_ref().owner();
        ctx.mem.enter_block(owner);
    }
    if ctx.insn == ctx.insn_delivered {
        return;
    }
    ctx.insn_delivered = ctx.insn;
    let at = (ctx.insn as usize).wrapping_sub(1);
    if let Some(Request::InsnStart(w)) = ctx.meta_ref().requests.get(at) {
        let w = *w;
        ctx.insn_start = Some(w);
        ctx.mem.insn_start(&w);
    }
}

/// Call generated code at `addr`.
#[cfg(target_arch = "x86_64")]
fn enter(
    addr: u64,
    env: *mut u8,
    ctx: &mut RunCtx<'_>,
    slots: *mut u64,
) -> Result<u64, InterpError> {
    type Entry = unsafe extern "C" fn(*mut u8, *mut RunCtx<'_>, *mut u64) -> u64;
    // SAFETY: `addr` is the start of a block that `CodeRegion::compile_with` generated and
    // wrote, and the region stays mapped because the caller's `CompiledTb` holds it. The block
    // follows the host C calling convention (System V, or Win64 on Windows): it saves and
    // restores every callee-saved register it uses, including xmm6 to xmm15 on Windows, keeps
    // the stack 16-byte aligned and clears the upper vector state before it returns. It only
    // uses instruction set extensions the region was created for. It reads and writes only the
    // `env_len` bytes at `env` that `ctx` records (every access is bounds checked against that
    // length), at most `MAX_SLOT_WORDS` words at `slots`, which `run_in` checks the caller
    // provides, the words of `ctx` up to `tlb`, and the `icount_decr` word `ctx` points
    // to, which it only reads. It reads the TLB descriptor `ctx.tlb` points to and the tables it
    // names, which `run_in` keeps alive for the call with `FastTlb::enter_run`, and it accesses
    // host memory directly only at `addr + addend` for an entry whose comparator matched the
    // page of `addr`, which `TlbTables::set` requires to be mapped while the entry is there. It calls only `service`, with `ctx`. It jumps only to other
    // blocks with the same frame layout and the same guarantees: to the targets
    // `set_goto_tb_target` patched in, which are in this region or one before it in its chain,
    // all kept mapped by this region, and to the one address `serve_lookup` checked the same
    // way. `env` comes from a `&mut [u8]` the caller holds for the whole call, and nothing else
    // uses these pointers until the block returns.
    let k = unsafe {
        let f = std::mem::transmute::<usize, Entry>(addr as usize);
        f(env, ctx, slots)
    };
    Ok(k)
}

/// Generated code can only run on an x86-64 host.
#[cfg(not(target_arch = "x86_64"))]
fn enter(
    _addr: u64,
    _env: *mut u8,
    _ctx: &mut RunCtx<'_>,
    _slots: *mut u64,
) -> Result<u64, InterpError> {
    Err(InterpError::BadOp("x86_64 code cannot run on this host".into()))
}

/// The one Rust entry point of generated code. Returns 0 to continue, or the kind to leave
/// the block with.
extern "C" fn service(ctx: &mut RunCtx<'_>, req: u64, meta: *const BlockMeta) -> u64 {
    ctx.meta = meta;
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
    // The request index is in the lower half and its `insn_of` in the upper half, so that
    // the service routine need not look it up.
    let i = (req & u64::from(u32::MAX)) as usize;
    let m = meta_of(ctx.meta);
    let fast = m.fast.get(i).copied().unwrap_or(Fast::Other);
    ctx.insn = req >> 32;
    debug_assert_eq!(m.insn_of.get(i).copied().unwrap_or(0), ctx.insn);
    // SAFETY: `env` and `env_len` come from the `&mut [u8]` that `run_in` holds for the whole
    // run. Generated code is suspended in this call and touches the buffer again only after it
    // returns, and this slice does not outlive the call, so it is the only live access to the
    // buffer.
    let env = unsafe { std::slice::from_raw_parts_mut(ctx.env, ctx.env_len) };
    // A helper without side effects cannot raise an exception, so it needs neither the block
    // nor the instruction to restore the guest state from; the next request that may raise one,
    // or the end of the run, tells the guest memory.
    if !matches!(fast, Fast::Helper { pure: true, .. }) {
        deliver_insn_start(ctx);
    }
    if let Fast::Helper { f, nin, ret, .. } = fast {
        let mut he = HelperEnv { env, mem: &mut *ctx.mem };
        let v = f(&mut he, &ctx.args[..nin]).map_err(Leave::Unwind)?;
        if ret {
            ctx.args[0] = v as u64;
            ctx.args[1] = (v >> 64) as u64;
        }
        return Ok(());
    }
    let r = m
        .requests
        .get(i)
        .ok_or_else(|| Leave::Error(InterpError::BadOp(format!("unknown request {req}"))))?;
    match r {
        Request::InsnStart(words) => {
            ctx.insn_start = Some(*words);
            ctx.mem.insn_start(words);
        }
        Request::Call { name, ret, args, nin, .. } => {
            // Not found when the block was compiled: look it up by name now.
            let entry = ctx
                .helpers
                .get(name)
                .ok_or_else(|| Leave::Error(InterpError::UnknownHelper(name.clone())))?;
            if entry.ret != *ret || entry.args != *args {
                return Err(Leave::Error(InterpError::HelperSignature(name.clone())));
            }
            let f = entry.f;
            let mut he = HelperEnv { env, mem: &mut *ctx.mem };
            let v = f(&mut he, &ctx.args[..*nin]).map_err(Leave::Unwind)?;
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

/// What the entry [`serve_lookup`] finds for a block depends on besides the block's code: the
/// chain of `region`, the region of the run, and its place in it, and the length of the CPU
/// state. `[0, 0]`, which [`Chain::lookup_code`] does not remember, when they do not fit.
fn jump_key(region: &CodeRegion, env_len: u64) -> [u64; 2] {
    if region.seq > u64::from(u32::MAX) || env_len > u64::from(u32::MAX) {
        return [0, 0];
    }
    // Chains are numbered from 1, so a real key is never `[0, 0]`.
    [region.chain, region.seq << 32 | env_len]
}

/// The routine generated code calls for `lookup_tb_ptr` instead of [`service`], see
/// [`GenOptions::lookup`]. In a chained run it serves the call with [`serve_lookup`];
/// otherwise the call is served like any other.
extern "C" fn lookup_service(ctx: &mut RunCtx<'_>, req: u64, meta: *const BlockMeta) -> u64 {
    let Some(chain) = ctx.chain else { return service(ctx, req, meta) };
    ctx.meta = meta;
    // `helper_lookup_tb_ptr()` runs at the end of a block and does not restore the state of an
    // instruction, so the guest memory is not told about the block or instruction.
    ctx.insn = req >> 32;
    match catch_unwind(AssertUnwindSafe(|| serve_lookup(ctx, chain))) {
        Ok(Ok(v)) => {
            ctx.args[0] = v;
            ctx.args[1] = 0;
            0
        }
        Ok(Err(u)) => {
            ctx.unwind = Some(u);
            kind::UNWIND
        }
        // A panic must not unwind into generated code; it is resumed once the block returns.
        Err(p) => {
            ctx.panic = Some(p);
            kind::ERROR
        }
    }
}

/// `lookup_tb_ptr` in a chained run: the entry of the next block when generated code may jump
/// there, which `goto_ptr` then does, or 0 to leave.
fn serve_lookup(ctx: &mut RunCtx<'_>, chain: &dyn Chain) -> Result<u64, Unwind> {
    // SAFETY: as in `serve`: `env` and `env_len` come from the `&mut [u8]` that `run_in` holds
    // for the whole run, generated code is suspended in this call, and the slice does not
    // outlive it.
    let env = unsafe { std::slice::from_raw_parts_mut(ctx.env, ctx.env_len) };
    let region = ctx.region;
    let env_len = env.len() as u64;
    let mut he = HelperEnv { env, mem: &mut *ctx.mem };
    let mut jump = |c: &CompiledTb| region.reaches(&c.region).then(|| c.chain_entry(env_len));
    match chain.lookup_code(&mut he, ctx.jump_key, &mut jump)? {
        Found::Jump(entry) => {
            ctx.goto_ptr_ok = entry;
            Ok(entry)
        }
        Found::Leave(next) => {
            if next.is_some() {
                ctx.next = next;
            }
            Ok(0)
        }
    }
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
