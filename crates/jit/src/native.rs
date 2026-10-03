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
//!   one is started, and the old one is unmapped once the blocks in it are gone, which the
//!   runtime's own code budget and `tb_flush` see to. QEMU flushes when its one buffer fills.
//! - Chained `goto_tb` jumps come back to this loop, which runs the next block, rather than
//!   jumping straight to its code.

use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use ruvm_jit_core::{Func, HelperType};
use ruvm_jit_interp::{Exit, HelperRegistry, Machine, Unwind};

use crate::backend::{Backend, GenCodeError, InterpBackend, TbRet};
use crate::cpu::{Cpu, CpuLoopExit};
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
        match host::compile(&region, f) {
            Err(host::GenCodeError::TooLarge) if region.used() > 0 => {}
            r => return r,
        }
        // The block may just not fit in what is left; try a fresh region.
        let fresh = host::new_region(self.region_size).ok_or(host::GenCodeError::TooLarge)?;
        let c = host::compile(&fresh, f)?;
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
        t[n] = dest.map(Arc::downgrade);
        // Patch under the lock, so that code seen to take the jump finds its target set.
        if let Body::Native(c) = &code.body {
            c.set_goto_tb_linked(n as u32, dest.is_some());
        }
    }

    fn exec(&self, cpu: &mut Cpu<'_>, tb: &Arc<Tb>) -> Result<TbRet, CpuLoopExit> {
        let mut tb = tb.clone();
        let r = loop {
            cpu.core.current_tb = Some(tb.clone());
            cpu.core.cur_insn = None;
            crate::backend::copy_icount_decr(cpu);
            let code = native_code(&tb);
            let targets = crate::backend::live_targets(&code.targets);
            let exit = match &code.body {
                Body::Native(c) => {
                    let mut last = None;
                    c.run_traced(&mut *cpu.env, &mut *cpu.core, &self.helpers, &mut last)
                }
                Body::Interp(func) => {
                    let mut m = Machine::new(&mut *cpu.env, &mut *cpu.core, &self.helpers);
                    m.linked = [targets[0].is_some(), targets[1].is_some()];
                    m.run(func)
                }
            };
            match exit {
                Ok(Exit::ExitTb(v)) => {
                    let id = v & !3;
                    if id == 0 {
                        break Ok(TbRet { last_tb: None, exit: v & 3 });
                    }
                    assert_eq!(id, tb.id, "exit_tb names a block that is not running");
                    break Ok(TbRet { last_tb: Some(tb), exit: v & 3 });
                }
                Ok(Exit::GotoTb(n)) => {
                    let n = n as usize & 1;
                    // The jump may have been linked after the targets were read.
                    let next = match &targets[n] {
                        Some(t) => Some(t.clone()),
                        None => crate::backend::live_targets(&code.targets)[n].clone(),
                    };
                    tb = next.expect("goto_tb on an unlinked slot");
                }
                Ok(Exit::GotoPtr(0)) => break Ok(TbRet { last_tb: None, exit: 0 }),
                Ok(Exit::GotoPtr(p)) => {
                    let next = cpu.core.goto_ptr_target.take().expect("goto_ptr without a lookup");
                    assert_eq!(next.id, p, "goto_ptr to a block lookup_tb_ptr did not return");
                    tb = next;
                }
                Ok(Exit::Unwind(u)) => break Err(unwind(cpu, u)),
                Err(e) => {
                    panic!("{} backend error in the block at pc {:#x}: {e}", host::NAME, tb.pc)
                }
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
    use std::sync::Arc;

    use ruvm_jit_aarch64::{CodegenOptions, HostFeatures, select_fence_mapping};
    use ruvm_jit_core::{FenceMapping, Func};

    pub(super) use ruvm_jit_aarch64::{CodeRegion, CompiledTb, GenCodeError, TARGET_DEFAULT_MO};

    pub(super) const AVAILABLE: bool = true;
    pub(super) const NAME: &str = "aarch64";

    pub(super) fn new_region(size: usize) -> Option<Arc<CodeRegion>> {
        CodeRegion::new(size).ok()
    }

    pub(super) fn compile(r: &Arc<CodeRegion>, f: &Func) -> Result<CompiledTb, GenCodeError> {
        let opts = CodegenOptions { guest_window: false, features: features() };
        r.compile_with(f, &opts)
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
    use std::sync::Arc;

    use ruvm_jit_core::types::mo;
    use ruvm_jit_core::{FenceMapping, Func};

    pub(super) use ruvm_jit_x86_64::{CodeRegion, CompiledTb, GenCodeError};

    pub(super) const AVAILABLE: bool = true;
    pub(super) const NAME: &str = "x86_64";
    /// `TCG_TARGET_DEFAULT_MO` of `tcg/i386`: everything but store then load.
    pub(super) const TARGET_DEFAULT_MO: u32 = mo::ALL & !mo::ST_LD;

    pub(super) fn new_region(size: usize) -> Option<Arc<CodeRegion>> {
        CodeRegion::new(size).ok()
    }

    pub(super) fn compile(r: &Arc<CodeRegion>, f: &Func) -> Result<CompiledTb, GenCodeError> {
        r.compile(f)
    }

    pub(super) fn fence_mapping(_guest_mo: u32) -> FenceMapping {
        FenceMapping::Qemu
    }
}

/// A host without a native backend: nothing here can be made.
#[cfg(not(any(all(unix, target_arch = "aarch64"), target_arch = "x86_64")))]
mod host {
    use std::sync::Arc;

    use ruvm_jit_core::{FenceMapping, Func};
    use ruvm_jit_interp::{Exit, GuestMemory, HelperRegistry, InterpError};

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

    impl CompiledTb {
        pub(super) fn size(&self) -> usize {
            match *self {}
        }

        pub(super) fn set_goto_tb_linked(&self, _idx: u32, _linked: bool) -> bool {
            match *self {}
        }

        pub(super) fn run_traced(
            &self,
            _env: &mut [u8],
            _mem: &mut dyn GuestMemory,
            _helpers: &HelperRegistry,
            _last: &mut Option<[u64; ruvm_jit_core::types::INSN_START_WORDS]>,
        ) -> Result<Exit, InterpError> {
            match *self {}
        }
    }

    pub(super) fn new_region(_size: usize) -> Option<Arc<CodeRegion>> {
        None
    }

    pub(super) fn compile(_r: &Arc<CodeRegion>, _f: &Func) -> Result<CompiledTb, GenCodeError> {
        Err(GenCodeError::TooLarge)
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
