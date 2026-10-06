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
//!
//! [`CompiledTb::run_with_window`] also hands the code a [`HostWindow`], a run of guest memory
//! held as host atomics that code compiled with [`CodegenOptions::guest_window`] loads and
//! stores directly. Several threads can run blocks against the same window at once, which is
//! what the memory ordering litmus tests do.
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
//! - Inlined softmmu lookups ([`CompileOptions::tlb_page_bits`]) read a copy of the TLB
//!   descriptor in the run context, not one at a fixed offset from `env`. The copy is made
//!   again after any call out of generated code that changed the descriptor. A run whose guest
//!   memory has no TLB of the chain's page size gives them a descriptor that never matches, so
//!   every access takes the slow path through the service routine.
//! - `lookup_and_goto_ptr` calls its own lookup routine, which finds the block through a
//!   per-CPU cache that also keeps the address to jump to, so a hit costs no reference count
//!   and no chain check; QEMU's `helper_lookup_tb_ptr` reads `tb->tc.ptr` from its jump cache.
//! - A `lookup_tb_ptr_ic` call has an inline cache of [`IC_WAYS`] entries, direct mapped by
//!   the guest program counter. Each names the header of a block, which holds the program
//!   counter the block starts at and the address to jump to; generated code jumps when the
//!   program counter matches, and otherwise calls the lookup routine, which fills the entry
//!   when the block found was translated for the same CPU state as the calling block
//!   ([`CompiledTb::set_ic_key`]). Invalidating a block clears its header
//!   ([`CompiledTb::clear_ic`]). QEMU has no such cache: every `lookup_and_goto_ptr` calls
//!   `helper_lookup_tb_ptr`. Like a chained `goto_tb`, a hit skips what the lookup does
//!   besides finding the block: the breakpoint check and setting `can_do_io`.
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
use ruvm_jit_interp::fast_tlb::{TLB_DESC_WORDS, TLB_MAX_MMU_MODES};
use ruvm_jit_interp::{
    Exit, FastTlb, GuestMemory, HelperEnv, HelperFn, HelperRegistry, InterpError, Unwind,
    guest_load_env, guest_store_env,
};

use crate::asm::i;
use crate::buffer::{BufferError, CodeBuffer};
use crate::codegen::{
    self, ChainGen, CodegenOptions, DECR_OFFSET, GOTO_PTR_OK_OFFSET, GenCodeError, IC_WAYS,
    INSN_OFFSET, META_OFFSET, NARGS, RET_OFFSET, Request, TLB_OFFSET, WIN_BASE_OFFSET,
    WIN_HOST_OFFSET, WIN_LIMIT_OFFSET, ic_way, kind, patch_ic_addr,
};

/// The most 64-bit words of slot array a block may use, so that every block of a chain fits
/// the one slot array of the thread. A block that needs more is refused with
/// [`GenCodeError::TooLarge`].
pub const MAX_SLOT_WORDS: usize = 1 << 16;

/// The name of the helper whose result `goto_ptr` jumps to.
pub(crate) const LOOKUP_TB_PTR: &str = "lookup_tb_ptr";

/// The name of the `lookup_tb_ptr` variant whose calls get an inline cache, see
/// [`CompiledTb::set_ic_key`].
pub(crate) const LOOKUP_TB_PTR_IC: &str = "lookup_tb_ptr_ic";

/// Set in the request word of a `lookup_tb_ptr_ic` call, so that [`lookup_service`] knows
/// the call has an inline cache without reading the metadata of the block, which is often
/// not in the host cache at the end of a block.
pub(crate) const LOOKUP_IC_SITE: u64 = 1 << 31;

/// The program counter word of a block header that no inline cache may jump through. Its
/// address word is 0 too, so even a guest jump to this program counter does not use it.
const IC_NONE: u64 = u64::MAX;

/// The header inline cache entries name before they are filled.
static IC_EMPTY: [u64; 2] = [IC_NONE, 0];

/// The words of a block header: the program counter and the entry past the bounds check, then
/// the program counter again and the entry before the check. An inline cache names the first
/// half when the calling block's check covers the callee's, and the second half otherwise.
const IC_HEAD: usize = 4;

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
    /// The block header ([`IC_HEAD`] words, see [`CompiledTb::set_ic_key`]), followed by
    /// [`IC_WAYS`] words for each inline cache of the block. Generated code reads it, the
    /// runtime writes it.
    ic: Box<[AtomicU64]>,
    /// The CPU state the block runs with, apart from the program counter, as given to
    /// [`CompiledTb::set_ic_key`].
    ic_key: OnceLock<[u64; 2]>,
    /// The chain of the region of the block and its place in it, see [`CodeRegion::reaches`].
    chain: u64,
    seq: u64,
    /// [`Generated::env_need`](codegen::Generated::env_need).
    env_need: u64,
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
    /// A call to `lookup_tb_ptr_ic` with an inline cache at this word of [`BlockMeta::ic`].
    LookupIc(usize),
    /// Anything else.
    Other,
}

impl BlockMeta {
    /// Make every inline cache that names this block's header miss.
    fn clear_ic(&self) {
        self.ic[0].store(IC_NONE, Ordering::Release);
        self.ic[2].store(IC_NONE, Ordering::Release);
        self.ic[1].store(0, Ordering::Release);
        self.ic[3].store(0, Ordering::Release);
    }

    fn owner(&self) -> Option<Arc<dyn Any + Send + Sync>> {
        self.owner.get().and_then(Weak::upgrade)
    }
}

/// One executable buffer that blocks are compiled into, front to back.
#[derive(Debug)]
pub struct CodeRegion {
    buf: CodeBuffer,
    next: Mutex<usize>,
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
    /// skip the lookup by name when they run. A call to a `TCG_CALL_NO_SE` helper that has a
    /// native entry point (`HelperRegistry::register_native`) is a direct host call that does
    /// not leave the block.
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
    /// A region of at least `size` bytes of executable memory.
    pub fn new(size: usize) -> Result<Arc<CodeRegion>, BufferError> {
        Ok(Arc::new(CodeRegion {
            buf: CodeBuffer::new(size)?,
            next: Mutex::new(0),
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
            _prev: Some(Arc::clone(self)),
            chain: self.chain,
            seq: self.seq + 1,
            metas: Mutex::new(Vec::new()),
            tlb_page_bits: Arc::clone(&self.tlb_page_bits),
        }))
    }

    /// Bytes of the region in use.
    pub fn used(&self) -> usize {
        *self.next.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The region's size in bytes.
    pub fn size(&self) -> usize {
        self.buf.size()
    }

    /// Stop inline caches from jumping to any block of this region and the regions before it,
    /// for when the blocks are all dropped.
    pub fn clear_ic(&self) {
        let mut r = Some(self);
        while let Some(region) = r {
            for m in region.metas.lock().unwrap_or_else(|e| e.into_inner()).iter() {
                m.clear_ic();
            }
            r = region._prev.as_deref();
        }
    }

    /// Whether code running in this region may jump into `other`: it is this region or one
    /// before it in the chain, which this one keeps mapped.
    fn reaches(&self, other: &CodeRegion) -> bool {
        self.chain == other.chain && other.seq <= self.seq
    }

    /// Compile `f` into the region, `tcg_gen_code`. A block with temps wider than 128 bits is
    /// refused with [`GenCodeError::Unsupported`]; it can be run with the interpreter instead.
    pub fn compile(self: &Arc<Self>, f: &Func) -> Result<CompiledTb, GenCodeError> {
        self.compile_with(f, &CodegenOptions::default())
    }

    /// [`CodeRegion::compile`] with `opts`.
    pub fn compile_with(
        self: &Arc<Self>,
        f: &Func,
        opts: &CodegenOptions,
    ) -> Result<CompiledTb, GenCodeError> {
        self.compile_chained(f, opts, &CompileOptions::default())
    }

    /// [`CodeRegion::compile_with`], for running with [`CompiledTb::run_chained`] as `opts`
    /// describes.
    pub fn compile_chained(
        self: &Arc<Self>,
        f: &Func,
        gen_opts: &CodegenOptions,
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
        let chain = ChainGen {
            meta: Arc::as_ptr(&meta) as usize as u64,
            icount_decr: opts.icount_decr_offset,
            tlb_page_bits: opts.tlb_page_bits,
            lookup: {
                let f: extern "C" fn(&mut RunCtx<'_>, u64, *const BlockMeta) -> u64 =
                    lookup_service;
                f as usize as u64
            },
            helpers: opts.helpers,
        };
        let mut next = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let offset = next.next_multiple_of(16);
        let base = self.buf.addr() + offset as u64;
        let service_fn: extern "C" fn(&mut RunCtx<'_>, u64, *const BlockMeta) -> u64 = service;
        let mut g = codegen::generate(f, base, service_fn as usize as u64, gen_opts, &chain)?;
        if g.slot_words > MAX_SLOT_WORDS {
            return Err(GenCodeError::TooLarge);
        }
        let end = offset.checked_add(g.bytes.len()).ok_or(GenCodeError::TooLarge)?;
        if end > self.buf.size() {
            return Err(GenCodeError::TooLarge);
        }
        let m = Arc::get_mut(&mut meta).expect("the metadata is not shared yet");
        // The header, then the cache words of each inline cache, which start out naming the
        // empty header. The code finds them through the address patched into it.
        let empty = IC_EMPTY.as_ptr() as u64;
        m.ic = (0..IC_HEAD + IC_WAYS * g.ic_sites.len())
            .map(|k| AtomicU64::new(if k < IC_HEAD { IC_EMPTY[k % 2] } else { empty }))
            .collect();
        let mut ic_fast = Vec::with_capacity(g.ic_sites.len());
        for (j, &(req, at)) in g.ic_sites.iter().enumerate() {
            let word = IC_HEAD + IC_WAYS * j;
            patch_ic_addr(&mut g.bytes, at, m.ic[word..].as_ptr() as u64);
            ic_fast.push((req, word));
        }
        m.chain = self.chain;
        m.seq = self.seq;
        m.env_need = g.env_need;
        m.fast = g
            .requests
            .iter()
            .map(|r| match (r, opts.helpers) {
                (Request::Call { name, .. }, _)
                    if name == LOOKUP_TB_PTR || name == LOOKUP_TB_PTR_IC =>
                {
                    Fast::Lookup
                }
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
        for (req, word) in ic_fast {
            m.fast[req] = Fast::LookupIc(word);
        }
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

/// Guest memory that generated code accesses directly: guest address `guest_base + k` is byte
/// `k % 8` of `words[k / 8]`, in host (little endian) order.
///
/// Rust code must only touch the words through the atomics. Generated code uses plain,
/// load-acquire and store-release instructions on them, which the hardware makes single-copy
/// atomic for aligned accesses.
#[derive(Clone, Copy, Debug)]
pub struct HostWindow<'a> {
    words: &'a [AtomicU64],
    guest_base: u64,
}

impl<'a> HostWindow<'a> {
    /// The window of `words` at guest address `guest_base`, which must be 8 byte aligned.
    pub fn new(words: &'a [AtomicU64], guest_base: u64) -> HostWindow<'a> {
        assert!(guest_base % 8 == 0, "the window must start 8 byte aligned");
        assert!(
            guest_base.checked_add(8 * words.len() as u64).is_some(),
            "the window wraps the guest address space"
        );
        HostWindow { words, guest_base }
    }

    /// The guest address of the first byte.
    pub fn guest_base(&self) -> u64 {
        self.guest_base
    }

    /// The words.
    pub fn words(&self) -> &'a [AtomicU64] {
        self.words
    }

    /// The window fields of the run context: base, limit and host address.
    fn ctx_fields(&self) -> (u64, u64, u64) {
        let len = 8 * self.words.len() as u64;
        (self.guest_base, len.saturating_sub(7), self.words.as_ptr() as u64)
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

/// What a run has beyond the block: a host window or a chain to follow.
#[derive(Clone, Copy, Default)]
struct Extra<'a> {
    window: Option<&'a HostWindow<'a>>,
    chain: Option<(&'a dyn Chain, &'a AtomicU32)>,
}

/// Run `f` with the slot array of this thread.
fn with_slots<T>(f: impl FnOnce(&mut [u64]) -> T) -> T {
    let mut slots = SLOTS.take().unwrap_or_else(|| vec![0; MAX_SLOT_WORDS].into());
    let r = f(&mut slots);
    SLOTS.set(Some(slots));
    r
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

    /// The generated code, instructions and literal pool.
    pub fn code(&self) -> Vec<u8> {
        self.region.buf.read(self.offset, self.len).unwrap_or_default()
    }

    /// Record who owns this block, the value [`CompiledTb::run_chained`] reports blocks by.
    /// Only the first call has an effect.
    pub fn set_owner(&self, owner: Weak<dyn Any + Send + Sync>) {
        let _ = self.meta.owner.set(owner);
    }

    /// Let the inline caches of `lookup_tb_ptr_ic` calls jump to this block: it starts at the
    /// guest program counter `pc`, and `key` stands for the rest of the CPU state it was
    /// translated for. A block whose `lookup_tb_ptr_ic` call finds this one remembers it in its
    /// cache when both have the same `key`, so the call must only be made where the CPU state
    /// apart from the program counter is the one its block started with. Only the first call
    /// has an effect.
    pub fn set_ic_key(&self, pc: u64, key: [u64; 2]) {
        if self.meta.ic_key.set(key).is_ok() && pc != IC_NONE {
            self.meta.ic[1].store(self.addr() + self.fast_body as u64, Ordering::Release);
            self.meta.ic[3].store(self.entry(), Ordering::Release);
            self.meta.ic[0].store(pc, Ordering::Release);
            self.meta.ic[2].store(pc, Ordering::Release);
        }
    }

    /// Stop inline caches from jumping to this block, for when it is invalidated.
    pub fn clear_ic(&self) {
        self.meta.clear_ic();
    }

    /// Mark `goto_tb` slot `idx` as chained or not, `tb_target_set_jmp_target`. A chained slot
    /// leaves the block with [`Exit::GotoTb`]; an unchained one falls through, as
    /// [`ruvm_jit_interp::Machine::linked`] describes. Returns false if the block has no such
    /// slot.
    pub fn set_goto_tb_linked(&self, idx: u32, linked: bool) -> bool {
        self.patch_goto_tb(idx, |_, stub| if linked { stub } else { i::NOP })
    }

    /// Point `goto_tb` slot `idx` at `dest`, `tb_target_set_jmp_target`: the jump goes
    /// straight to the code of `dest` when this block may reach it (same chain of regions, not
    /// a later region, and within the 128 MiB reach of `B`), and otherwise leaves the block
    /// with [`Exit::GotoTb`]. `None` unlinks the slot, so that it falls through. Returns false
    /// if the block has no such slot.
    pub fn set_goto_tb_target(&self, idx: u32, dest: Option<&CompiledTb>) -> bool {
        let base = self.addr();
        self.patch_goto_tb(idx, |at, stub| match dest {
            None => i::NOP,
            Some(d) if self.region.reaches(&d.region) => {
                // B has a signed 26-bit word displacement, 128 MiB either way.
                let disp = (d.chain_entry(self.env_need) as i64 - (base as i64 + at as i64)) >> 2;
                if disp == (disp << 38) >> 38 { i::B | (disp as u32 & 0x03ff_ffff) } else { stub }
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
        let r = with_slots(|slots| {
            self.run_in(env, Extra::default(), mem, helpers, slots, last_insn_start)
        });
        Ok(r?.exit)
    }

    /// [`CompiledTb::run`] with guest memory `window` that code compiled with
    /// [`CodegenOptions::guest_window`] accesses directly. Accesses the code cannot make there
    /// (outside the window, misaligned, byte swapped or 128 bits wide) go to `mem`, which
    /// should therefore show the same bytes for the window's addresses.
    pub fn run_with_window(
        &self,
        env: &mut [u8],
        window: &HostWindow<'_>,
        mem: &mut dyn GuestMemory,
        helpers: &HelperRegistry,
    ) -> Result<Exit, InterpError> {
        let mut last = None;
        let extra = Extra { window: Some(window), chain: None };
        let r = with_slots(|slots| self.run_in(env, extra, mem, helpers, slots, &mut last));
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
        let mut last = None;
        let extra = Extra { window: None, chain: Some((chain, icount_decr)) };
        with_slots(|slots| self.run_in(env, extra, mem, helpers, slots, &mut last))
    }

    fn run_in(
        &self,
        env: &mut [u8],
        extra: Extra<'_>,
        mem: &mut dyn GuestMemory,
        helpers: &HelperRegistry,
        slots: &mut [u64],
        last_insn_start: &mut Option<[u64; INSN_START_WORDS]>,
    ) -> Result<ChainExit, InterpError> {
        // Chained blocks share the slots, so there must be room for any block.
        assert!(slots.len() >= MAX_SLOT_WORDS);
        let env_len = env.len();
        // Blocks compiled without `icount_decr_offset` never read this.
        let local_decr = AtomicU32::new(0);
        let decr = extra.chain.map_or(&local_decr, |c| c.1);
        let chain = extra.chain.map(|c| c.0);
        let (win_base, win_limit, win_host) =
            extra.window.map_or((0, 0, 0), HostWindow::ctx_fields);
        let meta = Arc::as_ptr(&self.meta);
        // Blocks of this chain that inline TLB lookups read the TLB of `mem` when it has pages
        // of the size they were compiled for, and otherwise a descriptor that never matches.
        let bits = self.region.tlb_page_bits.get().copied();
        let fast = mem.fast_tlb().filter(|t| Some(t.page_bits()) == bits);
        let _run = fast.as_ref().map(|t| t.enter_run());
        let tlb = copy_desc(fast.as_deref().map_or(FastTlb::miss_desc(), FastTlb::desc));
        let mut ctx = RunCtx {
            args: [0; NARGS],
            ret: 0,
            insn: 0,
            win_base,
            win_limit,
            win_host,
            meta,
            decr: decr.as_ptr(),
            goto_ptr_ok: 0,
            tlb,
            fast_tlb: fast.as_deref(),
            tlb_generation: fast.as_ref().map_or(0, |t| t.generation()),
            jump_key: jump_key(&self.region, env_len as u64),
            meta_seen: meta,
            insn_delivered: 0,
            region: &self.region,
            env: env.as_mut_ptr(),
            env_len,
            mem,
            helpers,
            chain,
            next: None,
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
    /// The guest address of the host window.
    win_base: u64,
    /// The window length less 7, or 0 when there is no window: an offset below it has 8 bytes
    /// in the window.
    win_limit: u64,
    /// The host address of the window.
    win_host: u64,
    /// The metadata of the block that made the last request, or that left.
    meta: *const BlockMeta,
    /// The `icount_decr` word.
    decr: *const u32,
    /// The one address `goto_ptr` may jump to, or 0 for none.
    goto_ptr_ok: u64,
    /// A copy of the TLB descriptor, which inlined lookups read. Copied again after any call
    /// out of generated code that changed the descriptor.
    tlb: [u64; TLB_MAX_MMU_MODES * TLB_DESC_WORDS],
    /// The TLB the descriptor is copied from, if any, and its generation at the last copy.
    fast_tlb: Option<&'a FastTlb>,
    tlb_generation: u64,
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
    env_len: usize,
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
const _: () = assert!(std::mem::offset_of!(RunCtx<'static>, win_base) == WIN_BASE_OFFSET as usize);
const _: () =
    assert!(std::mem::offset_of!(RunCtx<'static>, win_limit) == WIN_LIMIT_OFFSET as usize);
const _: () = assert!(std::mem::offset_of!(RunCtx<'static>, win_host) == WIN_HOST_OFFSET as usize);
const _: () = assert!(std::mem::offset_of!(RunCtx<'static>, meta) == META_OFFSET as usize);
const _: () = assert!(std::mem::offset_of!(RunCtx<'static>, decr) == DECR_OFFSET as usize);
const _: () =
    assert!(std::mem::offset_of!(RunCtx<'static>, goto_ptr_ok) == GOTO_PTR_OK_OFFSET as usize);
const _: () = assert!(std::mem::offset_of!(RunCtx<'static>, tlb) == TLB_OFFSET as usize);

/// The descriptor words generated code reads, which [`RunCtx::tlb`] holds a copy of.
fn copy_desc(desc: &[AtomicU64]) -> [u64; TLB_MAX_MMU_MODES * TLB_DESC_WORDS] {
    std::array::from_fn(|i| desc[i].load(Ordering::Acquire))
}

impl RunCtx<'_> {
    /// Copy the TLB descriptor again if it changed since the last copy. Called before going
    /// back to generated code after a call out of it.
    fn refresh_tlb(&mut self) {
        if let Some(t) = self.fast_tlb {
            let g = t.generation();
            if g != self.tlb_generation {
                self.tlb_generation = g;
                self.tlb = copy_desc(t.desc());
            }
        }
    }

    /// The metadata of the running block.
    fn meta_ref(&self) -> &BlockMeta {
        meta_of(self.meta)
    }
}

/// The metadata at `meta`, the address a block passed or stored in [`RunCtx::meta`]. The
/// reference is not tied to the context, so that a request can be used while the context is
/// changed.
fn meta_of<'m>(meta: *const BlockMeta) -> &'m BlockMeta {
    // SAFETY: `meta` is either the metadata of the first block, which `run_in` borrows
    // through its `CompiledTb`, or the address a block that generated code jumped to passed
    // with a request or stored when it left. Generated code only jumps to blocks in the first
    // block's region or the regions before it in its chain (`set_goto_tb_target` and
    // `serve_lookup` check this), every block passes and stores the address of a `BlockMeta`
    // its region holds in `metas`, and the first block's region keeps itself and the regions
    // before it alive for the whole run. The metadata is never written after the block is
    // compiled. The reference is only used during the run, inside `run_in`.
    unsafe { &*meta }
}

/// Report the `insn_start` of the instruction of the last request or exit, if it is new, to the
/// context and the guest memory, as the interpreter does when it executes one. When a chained
/// block is running, tell the guest memory about it first.
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
#[cfg(all(unix, target_arch = "aarch64"))]
fn enter(
    addr: u64,
    env: *mut u8,
    env_len: usize,
    ctx: &mut RunCtx<'_>,
    slots: *mut u64,
) -> Result<u64, InterpError> {
    type Entry = unsafe extern "C" fn(*mut u8, u64, *mut RunCtx<'_>, *mut u64) -> u64;
    // SAFETY: `addr` is the start of a block that `CodeRegion::compile_chained` generated and
    // wrote, and the region stays mapped because the caller's `CompiledTb` holds it. The block
    // follows the AAPCS64 C calling convention: it saves and restores every callee-saved
    // register it uses and keeps the stack 16-byte aligned. It reads and writes only `env_len`
    // bytes at `env` (every access is bounds checked against that length), at most
    // `MAX_SLOT_WORDS` words at `slots`, which `run_in` checks the caller provides, the words
    // of `ctx` up to `tlb`, the `icount_decr` word `ctx` points to, which it only
    // reads, and the host window `ctx` describes, if any. It reads the copy of the TLB
    // descriptor in `ctx.tlb` and the tables it names, which `run_in` keeps alive for the call
    // with `FastTlb::enter_run` and which `RunCtx::refresh_tlb` copies again whenever a call
    // out of generated code changed them, and it accesses host memory directly only at
    // `addr + addend` for an entry whose comparator matched the page of `addr`, which
    // `TlbTables::set` requires to be mapped while the entry is there. It calls only
    // `service`, with `ctx`, and the safe `NativeHelperFn`s of the registry it was compiled
    // with, with words as arguments. It reads the inline cache words of its own metadata and
    // the block headers they name, all owned by regions this one keeps alive. It jumps only to
    // other blocks with the same frame layout and the same guarantees: to the targets
    // `set_goto_tb_target` patched in, which are in this region or one before it in its chain,
    // all kept mapped by this region, to the one address `serve_lookup` checked the same way,
    // and through inline caches to the addresses in block headers, which `ic_fill` only makes
    // them name for blocks in a region the caching block's region keeps mapped, and which hold
    // either 0 (not jumped to) or the entry of the block after the static check, taken only
    // when the block needs no more CPU state than the one jumping, which passed its own check,
    // or the entry before it. `env` comes from a `&mut [u8]` the caller holds for the whole
    // call, and nothing else uses these pointers until the block returns. The window comes from a
    // `&[AtomicU64]` that outlives the call; its accesses are checked against its length, and
    // atomics allow shared mutation, so other threads may use it at the same time.
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
extern "C" fn service(ctx: &mut RunCtx<'_>, req: u64, meta: *const BlockMeta) -> u64 {
    ctx.meta = meta;
    match catch_unwind(AssertUnwindSafe(|| serve(ctx, req))) {
        Ok(Ok(())) => {
            ctx.refresh_tlb();
            0
        }
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
/// [`ChainGen::lookup`]. In a chained run it serves the call with [`serve_lookup`];
/// otherwise the call is served like any other.
extern "C" fn lookup_service(ctx: &mut RunCtx<'_>, req: u64, meta: *const BlockMeta) -> u64 {
    let Some(chain) = ctx.chain else { return service(ctx, req & !LOOKUP_IC_SITE, meta) };
    ctx.meta = meta;
    // `helper_lookup_tb_ptr()` runs at the end of a block and does not restore the state of an
    // instruction, so the guest memory is not told about the block or instruction.
    ctx.insn = req >> 32;
    let ic = req & LOOKUP_IC_SITE != 0;
    let i = (req & u64::from(u32::MAX) & !LOOKUP_IC_SITE) as usize;
    match catch_unwind(AssertUnwindSafe(|| serve_lookup(ctx, chain, i, ic))) {
        Ok(Ok(v)) => {
            ctx.args[0] = v;
            ctx.args[1] = 0;
            ctx.refresh_tlb();
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
/// there, which `goto_ptr` then does, or 0 to leave. `ic` says whether request `i` has an
/// inline cache.
fn serve_lookup(
    ctx: &mut RunCtx<'_>,
    chain: &dyn Chain,
    i: usize,
    ic: bool,
) -> Result<u64, Unwind> {
    let site = if ic {
        match meta_of(ctx.meta).fast.get(i) {
            Some(&Fast::LookupIc(word)) => Some(word),
            _ => None,
        }
    } else {
        None
    };
    let pc = ctx.args[1];
    // SAFETY: as in `serve`: `env` and `env_len` come from the `&mut [u8]` that `run_in` holds
    // for the whole run, generated code is suspended in this call, and the slice does not
    // outlive it.
    let env = unsafe { std::slice::from_raw_parts_mut(ctx.env, ctx.env_len) };
    let region = ctx.region;
    let env_len = env.len() as u64;
    let mut he = HelperEnv { env, mem: &mut *ctx.mem };
    let mut target: Option<Arc<BlockMeta>> = None;
    let mut jump = |c: &CompiledTb| {
        let entry = region.reaches(&c.region).then(|| c.chain_entry(env_len));
        if entry.is_some() && site.is_some() {
            target = Some(Arc::clone(&c.meta));
        }
        entry
    };
    // An inline cache needs the block found, which a remembered entry does not give.
    let key = if site.is_some() { [0, 0] } else { ctx.jump_key };
    match chain.lookup_code(&mut he, key, &mut jump)? {
        Found::Jump(entry) => {
            if let (Some(word), Some(b)) = (site, target) {
                ic_fill(meta_of(ctx.meta), word, &b, pc);
            }
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

/// Remember block `b`, found for the program counter `pc`, in the inline cache at `word` of
/// block `a`, when code jumping there from `a` is right whatever the CPU state: `b` was
/// translated for the state `a` was ([`CompiledTb::set_ic_key`]), which the call promises
/// is the state at the call; it is in a region `a` keeps mapped; and it has not been
/// invalidated. The cache skips the bounds check of `b` only when `a`'s covers it.
fn ic_fill(a: &BlockMeta, word: usize, b: &BlockMeta, pc: u64) {
    let (Some(ka), Some(kb)) = (a.ic_key.get(), b.ic_key.get()) else { return };
    if ka != kb || a.chain != b.chain || b.seq > a.seq {
        return;
    }
    let half = if b.env_need <= a.env_need { 0 } else { 2 };
    if b.ic[half].load(Ordering::Acquire) != pc || b.ic[half + 1].load(Ordering::Acquire) == 0 {
        return;
    }
    if let Some(slot) = a.ic.get(word + ic_way(pc)) {
        slot.store(b.ic[half..].as_ptr() as u64, Ordering::Release);
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
