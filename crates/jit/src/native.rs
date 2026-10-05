// SPDX-License-Identifier: GPL-2.0-or-later

//! Host code generation: a [`Backend`] on top of `ruvm-jit-aarch64` or `ruvm-jit-x86_64`,
//! whichever matches the host, and the choice between it and the interpreter.
//!
//! [`host_backend`] picks the backend for a new runtime. It is the native one when the host has
//! one, the interpreter otherwise. The `RUVM_JIT_BACKEND` environment variable overrides the
//! choice for debugging: `interp` forces the interpreter (QEMU's TCI, which QEMU picks at build
//! time instead), `native` asks for the native backend. With `RUVM_JIT_BACKEND=native` a runtime
//! made with an [`InterpBackend`] also switches to the native backend with the same helpers,
//! which is how the front end test suites run on both backends without changes.
//!
//! Differences from QEMU:
//!
//! - A block the native code generator refuses (an op or type it does not handle) runs with
//!   the interpreter instead. QEMU has no such fallback because its backends handle every op.
//! - Code lives in a chain of regions. When the current one has no room left for a block a new
//!   one is started that keeps the old ones mapped, so that blocks in it can jump to blocks in
//!   them. `tb_flush` starts a new chain, and the old one is unmapped once the blocks in it are
//!   gone. The runtime's own code budget decides when to flush. QEMU flushes when its one
//!   buffer fills.
//! - Native blocks chain as in QEMU: a linked `goto_tb` jumps straight to the next block's code,
//!   and `lookup_and_goto_ptr` jumps to the code `helper_lookup_tb_ptr()` finds without
//!   leaving generated code. Jumps to a block run by the interpreter, or to a block of an older
//!   chain of regions, come back to this loop instead, which runs the block.

use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use ruvm_jit_core::{Func, HelperType};
use ruvm_jit_interp::{Exit, HelperEnv, HelperRegistry, Machine, Unwind};

use crate::backend::{Backend, GenCodeError, InterpBackend, TbRet};
use crate::cpu::{Cpu, CpuLoopExit};
use crate::cpu_exec;
use crate::tb::{Tb, lock};

/// The environment variable that picks the backend: `interp` or `native`.
pub const BACKEND_ENV: &str = "RUVM_JIT_BACKEND";

/// Which backend a runtime should use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// The IR interpreter.
    Interp,
    /// Code for the host CPU.
    Native,
}

impl BackendKind {
    /// The kind named `s`, as `RUVM_JIT_BACKEND` spells it.
    pub fn parse(s: &str) -> Option<BackendKind> {
        match s {
            "interp" | "tci" => Some(BackendKind::Interp),
            "native" => Some(BackendKind::Native),
            _ => None,
        }
    }

    /// What `RUVM_JIT_BACKEND` asks for, if anything.
    pub fn from_env() -> Option<BackendKind> {
        std::env::var(BACKEND_ENV).ok().and_then(|v| BackendKind::parse(v.trim()))
    }
}

/// Whether this host has a native backend.
pub const fn native_available() -> bool {
    host::AVAILABLE
}

/// The name of the host's native backend, such as `aarch64`, or `None` without one.
pub const fn native_name() -> Option<&'static str> {
    if host::AVAILABLE { Some(host::NAME) } else { None }
}

/// The backend for a runtime whose target helpers are `helpers`: the native one when the host
/// has it, unless `RUVM_JIT_BACKEND=interp` asks for the interpreter. `code_size` is the size of
/// each code region, the `tb-size` property.
pub fn host_backend(helpers: HelperRegistry, code_size: usize) -> Arc<dyn Backend> {
    backend_of_kind(BackendKind::from_env().unwrap_or(BackendKind::Native), helpers, code_size)
}

/// The backend of kind `kind`, falling back to the interpreter when the host has no native
/// backend or its code region cannot be mapped.
pub fn backend_of_kind(
    kind: BackendKind,
    helpers: HelperRegistry,
    code_size: usize,
) -> Arc<dyn Backend> {
    if kind == BackendKind::Native {
        if let Some(b) = NativeBackend::with_helpers(helpers.clone(), code_size) {
            return Arc::new(b);
        }
    }
    Arc::new(InterpBackend::with_helpers(helpers))
}

/// `backend` itself, or a native backend with its helpers when `RUVM_JIT_BACKEND=native` (or
/// the `force-native` feature) asks for one and `backend` is an [`InterpBackend`].
pub(crate) fn env_override(backend: Arc<dyn Backend>, code_size: usize) -> Arc<dyn Backend> {
    let forced = if cfg!(feature = "force-native") { Some(BackendKind::Native) } else { None };
    if BackendKind::from_env().or(forced) != Some(BackendKind::Native) {
        return backend;
    }
    let Some(helpers) = backend.interp_helpers() else { return backend };
    match NativeBackend::with_helpers(helpers.clone(), code_size) {
        Some(b) => Arc::new(b),
        None => backend,
    }
}

/// The code of one block.
enum Body {
    Native(host::CompiledTb),
    Interp(Box<Func>),
}

/// What a run of native code ended with.
struct Ran {
    exit: Exit,
    /// The block running when the code left, if the code chained on from the first one.
    block: Option<Arc<Tb>>,
    /// After `goto_ptr` with 0: the block to run next, which native code could not jump to.
    /// After `goto_tb` from an interpreted block: the target of the slot.
    next: Option<Arc<Tb>>,
}

/// `lookup_and_goto_ptr` for native code: the runtime's `helper_lookup_tb_ptr()`.
#[cfg_attr(not(any(all(unix, target_arch = "aarch64"), target_arch = "x86_64")), allow(dead_code))]
struct TbChain;

#[cfg_attr(not(any(all(unix, target_arch = "aarch64"), target_arch = "x86_64")), allow(dead_code))]
impl TbChain {
    fn lookup(he: &mut HelperEnv<'_>) -> Result<Option<Arc<Tb>>, Unwind> {
        let Some(mut cpu) = Cpu::from_helper_env(he) else { return Ok(None) };
        cpu_exec::helper_lookup_tb_ptr(&mut cpu).map_err(|e| cpu.unwind(e))
    }

    fn code(block: &(dyn Any + Send + Sync)) -> Option<&host::CompiledTb> {
        TbChain::native(block.downcast_ref::<Tb>()?)
    }

    fn native(tb: &Tb) -> Option<&host::CompiledTb> {
        match &native_code(tb).body {
            Body::Native(c) => Some(c),
            Body::Interp(_) => None,
        }
    }

    /// `Chain::lookup_code`: the block is only cloned when generated code cannot jump to it.
    #[inline]
    fn lookup_code(
        he: &mut HelperEnv<'_>,
        key: [u64; 2],
        jump: &mut dyn FnMut(&host::CompiledTb) -> Option<u64>,
    ) -> Result<host::Found, Unwind> {
        let Some(mut cpu) = Cpu::from_helper_env(he) else { return Ok(host::Found::Leave(None)) };
        let found = cpu_exec::helper_lookup_tb_ptr_jump(&mut cpu, key, |tb| {
            TbChain::native(tb).and_then(jump)
        });
        match found {
            Ok(Some(Ok(entry))) => Ok(host::Found::Jump(entry)),
            Ok(Some(Err(tb))) => Ok(host::Found::Leave(Some(tb as Arc<dyn Any + Send + Sync>))),
            Ok(None) => Ok(host::Found::Leave(None)),
            Err(e) => Err(cpu.unwind(e)),
        }
    }
}

#[cfg_attr(not(any(all(unix, target_arch = "aarch64"), target_arch = "x86_64")), allow(dead_code))]
fn as_tb(b: Option<Arc<dyn Any + Send + Sync>>) -> Option<Arc<Tb>> {
    b.and_then(|b| b.downcast::<Tb>().ok())
}

/// The code of a block for [`NativeBackend`].
struct NativeCode {
    body: Body,
    targets: Mutex<[Option<Weak<Tb>>; 2]>,
}

/// A backend that generates host code, falling back to the interpreter for blocks it cannot
/// compile.
pub struct NativeBackend {
    helpers: HelperRegistry,
    region: Mutex<Arc<host::CodeRegion>>,
    region_size: usize,
    native_blocks: AtomicU64,
    interp_blocks: AtomicU64,
}

impl fmt::Debug for NativeBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeBackend")
            .field("host", &host::NAME)
            .field("helpers", &self.helpers.len())
            .field("native_blocks", &self.native_blocks())
            .field("interp_blocks", &self.interp_blocks())
            .finish_non_exhaustive()
    }
}

/// The smallest code region, so that a tiny `tb-size` still holds a few blocks.
const MIN_REGION: usize = 1 << 20;

impl NativeBackend {
    /// A native backend for this host with `helpers`, which should include the interpreter's
    /// built-in ones, and code regions of `code_size` bytes. `None` when the host has no native
    /// backend or the region cannot be mapped. `lookup_tb_ptr` is replaced by the runtime's.
    pub fn with_helpers(mut helpers: HelperRegistry, code_size: usize) -> Option<NativeBackend> {
        if !host::AVAILABLE {
            return None;
        }
        let region_size = code_size.max(MIN_REGION);
        let region = host::new_region(region_size)?;
        helpers.register(
            "lookup_tb_ptr",
            HelperType::Ptr,
            &[HelperType::Ptr],
            crate::backend::lookup_tb_ptr,
        );
        crate::plugin::register_helpers(&mut helpers);
        Some(NativeBackend {
            helpers,
            region: Mutex::new(region),
            region_size,
            native_blocks: AtomicU64::new(0),
            interp_blocks: AtomicU64::new(0),
        })
    }

    /// The host the code is for, such as `aarch64`.
    pub fn host(&self) -> &'static str {
        host::NAME
    }

    /// How many blocks were compiled to host code.
    pub fn native_blocks(&self) -> u64 {
        self.native_blocks.load(Ordering::Relaxed)
    }

    /// How many blocks fell back to the interpreter.
    pub fn interp_blocks(&self) -> u64 {
        self.interp_blocks.load(Ordering::Relaxed)
    }

    /// Compile `f`, starting a new region when the current one is full.
    fn compile(&self, f: &Func) -> Result<host::CompiledTb, host::GenCodeError> {
        let region = lock(&self.region).clone();
        match host::compile(&region, f, &self.helpers) {
            Err(host::GenCodeError::TooLarge) if region.used() > 0 => {}
            r => return r,
        }
        // The block may just not fit in what is left; try a fresh region.
        let fresh =
            host::next_region(&region, self.region_size).ok_or(host::GenCodeError::TooLarge)?;
        let c = host::compile(&fresh, f, &self.helpers)?;
        let mut cur = lock(&self.region);
        if Arc::ptr_eq(&cur, &region) {
            *cur = fresh;
        }
        Ok(c)
    }
}

fn native_code(tb: &Tb) -> &NativeCode {
    tb.code().downcast_ref::<NativeCode>().expect("block was not generated by NativeBackend")
}

impl Backend for NativeBackend {
    fn gen_code(
        &self,
        f: Func,
        _id: u64,
    ) -> Result<(Box<dyn Any + Send + Sync>, usize), GenCodeError> {
        let (body, size) = match self.compile(&f) {
            Ok(c) => {
                self.native_blocks.fetch_add(1, Ordering::Relaxed);
                let size = c.size().next_multiple_of(16);
                (Body::Native(c), size)
            }
            Err(host::GenCodeError::TooLarge) => return Err(GenCodeError::TooLarge),
            Err(_) => {
                let size = f.nb_ops() * 16;
                if size > usize::from(u16::MAX) {
                    return Err(GenCodeError::TooLarge);
                }
                self.interp_blocks.fetch_add(1, Ordering::Relaxed);
                (Body::Interp(Box::new(f)), size)
            }
        };
        Ok((Box::new(NativeCode { body, targets: Mutex::new([None, None]) }), size))
    }

    fn set_jmp_target(&self, tb: &Arc<Tb>, n: usize, dest: Option<&Arc<Tb>>) {
        let code = native_code(tb);
        let mut t = lock(&code.targets);
        // Patch under the lock, so that code seen to take the jump finds its target set.
        if let Body::Native(c) = &code.body {
            // An unlinked slot keeps its old target, so that code that left through the jump
            // just before it was unlinked still finds the block it was jumping to.
            if let Some(d) = dest {
                t[n] = Some(Arc::downgrade(d));
            }
            let n = n as u32;
            match dest.map(|d| &native_code(d).body) {
                None => c.set_goto_tb_linked(n, false),
                Some(Body::Native(d)) => host::set_goto_tb_target(c, n, d),
                Some(Body::Interp(_)) => c.set_goto_tb_linked(n, true),
            };
        } else {
            t[n] = dest.map(Arc::downgrade);
        }
    }

    fn tb_created(&self, tb: &Arc<Tb>) {
        if let Body::Native(c) = &native_code(tb).body {
            let owner: Weak<dyn Any + Send + Sync> = Arc::downgrade(tb) as Weak<Tb>;
            host::set_owner(c, owner);
        }
    }

    fn tb_flush(&self) {
        let mut cur = lock(&self.region);
        if cur.used() > 0 {
            if let Some(fresh) = host::new_region(self.region_size) {
                *cur = fresh;
            }
        }
    }

    fn exec(&self, cpu: &mut Cpu<'_>, tb: &Arc<Tb>) -> Result<TbRet, CpuLoopExit> {
        let shared = cpu.core.shared.clone();
        let mut tb = tb.clone();
        let r = loop {
            cpu.core.current_tb = Some(tb.clone());
            cpu.core.cur_insn = None;
            let code = native_code(&tb);
            let ran = match &code.body {
                Body::Native(c) => {
                    host::run(c, &mut *cpu.env, &mut *cpu.core, &self.helpers, &shared.icount_decr)
                }
                Body::Interp(func) => {
                    crate::backend::copy_icount_decr(cpu);
                    let targets = crate::backend::live_targets(&code.targets);
                    let mut m = Machine::new(&mut *cpu.env, &mut *cpu.core, &self.helpers);
                    m.linked = [targets[0].is_some(), targets[1].is_some()];
                    m.run(func).map(|exit| {
                        // The slot is followed as it was linked when the block started.
                        let next = match exit {
                            Exit::GotoTb(n) => targets[n as usize & 1].clone(),
                            _ => None,
                        };
                        Ran { exit, block: None, next }
                    })
                }
            };
            let ran = match ran {
                Ok(r) => r,
                Err(e) => {
                    let pc = cpu.core.current_tb.as_ref().map_or(tb.pc, |t| t.pc);
                    panic!("{} backend error in the block at pc {pc:#x}: {e}", host::NAME)
                }
            };
            // The block that left, which is not the first one when native code chained on.
            let last = ran.block.unwrap_or_else(|| tb.clone());
            match ran.exit {
                Exit::ExitTb(v) => {
                    let id = v & !3;
                    if id == 0 {
                        break Ok(TbRet { last_tb: None, exit: v & 3 });
                    }
                    assert_eq!(id, last.id, "exit_tb names a block that is not running");
                    break Ok(TbRet { last_tb: Some(last), exit: v & 3 });
                }
                Exit::GotoTb(n) => {
                    let n = n as usize & 1;
                    let next = ran.next.or_else(|| {
                        crate::backend::live_targets(&native_code(&last).targets)[n].clone()
                    });
                    tb = next.expect("goto_tb on an unlinked slot");
                }
                Exit::GotoPtr(0) => match ran.next {
                    Some(next) => tb = next,
                    None => break Ok(TbRet { last_tb: None, exit: 0 }),
                },
                Exit::GotoPtr(p) => {
                    let next = cpu.core.goto_ptr_target.take().expect("goto_ptr without a lookup");
                    assert_eq!(next.id, p, "goto_ptr to a block lookup_tb_ptr did not return");
                    tb = next;
                }
                Exit::Unwind(u) => break Err(unwind(cpu, u)),
            }
        };
        cpu.core.current_tb = None;
        cpu.core.goto_ptr_target = None;
        r
    }

    fn target_default_mo(&self) -> u32 {
        host::TARGET_DEFAULT_MO
    }

    fn fence_mapping(&self, guest_mo: u32) -> ruvm_jit_core::FenceMapping {
        host::fence_mapping(guest_mo)
    }

    fn native_stats(&self) -> Option<(u64, u64)> {
        Some((self.native_blocks(), self.interp_blocks()))
    }
}

fn unwind(cpu: &mut Cpu<'_>, u: Unwind) -> CpuLoopExit {
    crate::backend::unwind(cpu, u)
}

#[cfg(all(unix, target_arch = "aarch64"))]
mod host {
    use std::any::Any;
    use std::sync::atomic::AtomicU32;
    use std::sync::{Arc, Weak};

    use ruvm_jit_aarch64::select_fence_mapping;
    use ruvm_jit_aarch64::{Chain, CodegenOptions, CompileOptions, HostFeatures};
    use ruvm_jit_core::{FenceMapping, Func};
    use ruvm_jit_interp::{GuestMemory, HelperEnv, HelperRegistry, InterpError, Unwind};

    pub(super) use ruvm_jit_aarch64::{
        CodeRegion, CompiledTb, Found, GenCodeError, TARGET_DEFAULT_MO,
    };

    use super::{Ran, TbChain, as_tb};
    use crate::ENV_ICOUNT_DECR_OFFSET;

    pub(super) const AVAILABLE: bool = true;
    pub(super) const NAME: &str = "aarch64";

    pub(super) fn new_region(size: usize) -> Option<Arc<CodeRegion>> {
        CodeRegion::new(size).ok()
    }

    pub(super) fn next_region(r: &Arc<CodeRegion>, size: usize) -> Option<Arc<CodeRegion>> {
        r.successor(size).ok()
    }

    pub(super) fn compile(
        r: &Arc<CodeRegion>,
        f: &Func,
        helpers: &HelperRegistry,
    ) -> Result<CompiledTb, GenCodeError> {
        let gen_opts = CodegenOptions { guest_window: false, features: features() };
        let opts = CompileOptions {
            helpers: Some(helpers),
            icount_decr_offset: Some(ENV_ICOUNT_DECR_OFFSET),
            tlb_page_bits: f.config.tlb_page_bits,
        };
        r.compile_chained(f, &gen_opts, &opts)
    }

    pub(super) fn set_goto_tb_target(c: &CompiledTb, n: u32, dest: &CompiledTb) -> bool {
        c.set_goto_tb_target(n, Some(dest))
    }

    pub(super) fn set_owner(c: &CompiledTb, owner: Weak<dyn Any + Send + Sync>) {
        c.set_owner(owner);
    }

    impl Chain for TbChain {
        fn lookup_tb_ptr(
            &self,
            he: &mut HelperEnv<'_>,
        ) -> Result<Option<Arc<dyn Any + Send + Sync>>, Unwind> {
            Ok(TbChain::lookup(he)?.map(|t| t as Arc<dyn Any + Send + Sync>))
        }

        fn code<'a>(&self, block: &'a (dyn Any + Send + Sync)) -> Option<&'a CompiledTb> {
            TbChain::code(block)
        }

        fn lookup_code(
            &self,
            he: &mut HelperEnv<'_>,
            key: [u64; 2],
            jump: &mut dyn FnMut(&CompiledTb) -> Option<u64>,
        ) -> Result<Found, Unwind> {
            TbChain::lookup_code(he, key, jump)
        }
    }

    pub(super) fn run(
        c: &CompiledTb,
        env: &mut [u8],
        mem: &mut dyn GuestMemory,
        helpers: &HelperRegistry,
        icount_decr: &AtomicU32,
    ) -> Result<Ran, InterpError> {
        let x = c.run_chained(env, mem, helpers, &TbChain, icount_decr)?;
        Ok(Ran { exit: x.exit, block: as_tb(x.block), next: as_tb(x.next) })
    }

    fn features() -> HostFeatures {
        static F: std::sync::OnceLock<HostFeatures> = std::sync::OnceLock::new();
        *F.get_or_init(HostFeatures::detect)
    }

    pub(super) fn fence_mapping(guest_mo: u32) -> FenceMapping {
        select_fence_mapping(FenceMapping::preferred(guest_mo, TARGET_DEFAULT_MO), guest_mo)
    }
}

#[cfg(target_arch = "x86_64")]
mod host {
    use std::any::Any;
    use std::sync::atomic::AtomicU32;
    use std::sync::{Arc, Weak};

    use ruvm_jit_core::types::mo;
    use ruvm_jit_core::{FenceMapping, Func};
    use ruvm_jit_interp::{GuestMemory, HelperEnv, HelperRegistry, InterpError, Unwind};
    use ruvm_jit_x86_64::{Chain, CompileOptions};

    pub(super) use ruvm_jit_x86_64::{CodeRegion, CompiledTb, Found, GenCodeError};

    use super::{Ran, TbChain, as_tb};
    use crate::ENV_ICOUNT_DECR_OFFSET;

    pub(super) const AVAILABLE: bool = true;
    pub(super) const NAME: &str = "x86_64";
    /// `TCG_TARGET_DEFAULT_MO` of `tcg/i386`: everything but store then load.
    pub(super) const TARGET_DEFAULT_MO: u32 = mo::ALL & !mo::ST_LD;

    pub(super) fn new_region(size: usize) -> Option<Arc<CodeRegion>> {
        CodeRegion::new(size).ok()
    }

    pub(super) fn next_region(r: &Arc<CodeRegion>, size: usize) -> Option<Arc<CodeRegion>> {
        r.successor(size).ok()
    }

    pub(super) fn compile(
        r: &Arc<CodeRegion>,
        f: &Func,
        helpers: &HelperRegistry,
    ) -> Result<CompiledTb, GenCodeError> {
        let opts = CompileOptions {
            helpers: Some(helpers),
            icount_decr_offset: Some(ENV_ICOUNT_DECR_OFFSET),
            tlb_page_bits: f.config.tlb_page_bits,
        };
        r.compile_with(f, &opts)
    }

    pub(super) fn set_goto_tb_target(c: &CompiledTb, n: u32, dest: &CompiledTb) -> bool {
        c.set_goto_tb_target(n, Some(dest))
    }

    pub(super) fn set_owner(c: &CompiledTb, owner: Weak<dyn Any + Send + Sync>) {
        c.set_owner(owner);
    }

    impl Chain for TbChain {
        fn lookup_tb_ptr(
            &self,
            he: &mut HelperEnv<'_>,
        ) -> Result<Option<Arc<dyn Any + Send + Sync>>, Unwind> {
            Ok(TbChain::lookup(he)?.map(|t| t as Arc<dyn Any + Send + Sync>))
        }

        fn code<'a>(&self, block: &'a (dyn Any + Send + Sync)) -> Option<&'a CompiledTb> {
            TbChain::code(block)
        }

        fn lookup_code(
            &self,
            he: &mut HelperEnv<'_>,
            key: [u64; 2],
            jump: &mut dyn FnMut(&CompiledTb) -> Option<u64>,
        ) -> Result<Found, Unwind> {
            TbChain::lookup_code(he, key, jump)
        }
    }

    pub(super) fn run(
        c: &CompiledTb,
        env: &mut [u8],
        mem: &mut dyn GuestMemory,
        helpers: &HelperRegistry,
        icount_decr: &AtomicU32,
    ) -> Result<Ran, InterpError> {
        let x = c.run_chained(env, mem, helpers, &TbChain, icount_decr)?;
        Ok(Ran { exit: x.exit, block: as_tb(x.block), next: as_tb(x.next) })
    }

    pub(super) fn fence_mapping(_guest_mo: u32) -> FenceMapping {
        FenceMapping::Qemu
    }
}

/// A host without a native backend: nothing here can be made.
#[cfg(not(any(all(unix, target_arch = "aarch64"), target_arch = "x86_64")))]
mod host {
    use std::any::Any;
    use std::sync::atomic::AtomicU32;
    use std::sync::{Arc, Weak};

    use ruvm_jit_core::{FenceMapping, Func};
    use ruvm_jit_interp::{GuestMemory, HelperRegistry, InterpError};

    use super::Ran;

    pub(super) const AVAILABLE: bool = false;
    pub(super) const NAME: &str = "none";
    pub(super) const TARGET_DEFAULT_MO: u32 = 0;

    #[derive(Debug)]
    pub(super) enum GenCodeError {
        TooLarge,
    }

    #[derive(Debug)]
    pub(super) struct CodeRegion;

    impl CodeRegion {
        pub(super) fn used(&self) -> usize {
            0
        }
    }

    #[derive(Debug)]
    pub(super) enum CompiledTb {}

    #[allow(dead_code)]
    pub(super) enum Found {
        Jump(u64),
        Leave(Option<Arc<dyn Any + Send + Sync>>),
    }

    impl CompiledTb {
        pub(super) fn size(&self) -> usize {
            match *self {}
        }

        pub(super) fn set_goto_tb_linked(&self, _idx: u32, _linked: bool) -> bool {
            match *self {}
        }
    }

    pub(super) fn new_region(_size: usize) -> Option<Arc<CodeRegion>> {
        None
    }

    pub(super) fn next_region(_r: &Arc<CodeRegion>, _size: usize) -> Option<Arc<CodeRegion>> {
        None
    }

    pub(super) fn compile(
        _r: &Arc<CodeRegion>,
        _f: &Func,
        _helpers: &HelperRegistry,
    ) -> Result<CompiledTb, GenCodeError> {
        Err(GenCodeError::TooLarge)
    }

    pub(super) fn set_goto_tb_target(c: &CompiledTb, _n: u32, _dest: &CompiledTb) -> bool {
        match *c {}
    }

    pub(super) fn set_owner(c: &CompiledTb, _owner: Weak<dyn Any + Send + Sync>) {
        match *c {}
    }

    pub(super) fn run(
        c: &CompiledTb,
        _env: &mut [u8],
        _mem: &mut dyn GuestMemory,
        _helpers: &HelperRegistry,
        _icount_decr: &AtomicU32,
    ) -> Result<Ran, InterpError> {
        match *c {}
    }

    pub(super) fn fence_mapping(_guest_mo: u32) -> FenceMapping {
        FenceMapping::Qemu
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvm_jit_core::FuncConfig;
    use ruvm_jit_core::types::mo;

    fn trivial() -> Func {
        let mut f = Func::new(FuncConfig::default());
        f.gen_exit_tb(0, 0);
        f.gen_code(true, ruvm_jit_core::liveness::LogMask::default());
        f
    }

    #[test]
    fn kinds_parse() {
        assert_eq!(BackendKind::parse("interp"), Some(BackendKind::Interp));
        assert_eq!(BackendKind::parse("tci"), Some(BackendKind::Interp));
        assert_eq!(BackendKind::parse("native"), Some(BackendKind::Native));
        assert_eq!(BackendKind::parse("llvm"), None);
    }

    #[test]
    fn interp_kind_is_the_interpreter() {
        let b = backend_of_kind(BackendKind::Interp, HelperRegistry::new(), 1 << 20);
        assert!(b.interp_helpers().is_some());
        assert!(b.native_stats().is_none());
    }

    #[test]
    fn native_kind_compiles_on_a_host_with_a_backend() {
        let b = backend_of_kind(BackendKind::Native, HelperRegistry::new(), 1 << 20);
        if !native_available() {
            assert!(b.interp_helpers().is_some());
            return;
        }
        assert!(b.interp_helpers().is_none());
        b.gen_code(trivial(), 4).expect("a trivial block compiles");
        assert_eq!(b.native_stats(), Some((1, 0)));
    }

    #[test]
    fn a_full_region_starts_another() {
        let Some(b) = NativeBackend::with_helpers(HelperRegistry::new(), MIN_REGION) else {
            return;
        };
        let mut total = 0;
        while total < 2 * MIN_REGION {
            let (_, size) = b.gen_code(trivial(), 4).expect("blocks keep compiling");
            total += size;
        }
        assert_eq!(b.interp_blocks(), 0);
    }

    #[test]
    fn fence_mapping_matches_the_host() {
        let x86_tso = mo::ALL & !mo::ST_LD;
        let m = host::fence_mapping(x86_tso);
        #[cfg(all(unix, target_arch = "aarch64"))]
        assert_eq!(
            m,
            ruvm_jit_aarch64::select_fence_mapping(
                ruvm_jit_core::FenceMapping::preferred(x86_tso, 0),
                x86_tso
            )
        );
        #[cfg(not(all(unix, target_arch = "aarch64")))]
        assert_eq!(m, ruvm_jit_core::FenceMapping::Qemu);
    }
}
