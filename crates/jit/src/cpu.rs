// SPDX-License-Identifier: GPL-2.0-or-later

//! The vCPU, QEMU's `CPUState` as far as TCG uses it, with the hooks of `TCGCPUOps`, the work
//! queue and exclusive sections of `cpus-common.c`, `cpu_interrupt()` and `qemu_cpu_kick()`, and
//! the breakpoint and watchpoint lists of `cpu-common.c` and `watchpoint.c`.
//!
//! A vCPU is split in two. [`CpuShared`] is what other threads touch: `interrupt_request`,
//! `exit_request`, the work queue, the TLB and the jump cache. [`CpuCore`] is what only the vCPU
//! thread touches. Together with the `env` buffer the core makes a [`Vcpu`], and a borrowed
//! [`Cpu`] is what every hook and runtime function takes.
//!
//! Differences from QEMU:
//!
//! - `cpu_loop_exit()` returns a [`CpuLoopExit`] that the caller propagates with `?`, instead of
//!   a longjmp.
//! - `exclusive_context_count` is kept per thread, so a thread that is not a vCPU can also start
//!   an exclusive section.
//! - Work items run without a big lock. `run_on_cpu()` must not be called by the vCPU's own
//!   thread; that thread already has the [`Cpu`] and can call the function directly.
//! - A breakpoint or watchpoint is named by its position in the list instead of a pointer.

use std::cell::Cell;
use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::thread::ThreadId;

use ruvm_jit_core::Type;
use ruvm_jit_core::types::{INSN_START_WORDS, MemOpIdx};
use ruvm_jit_interp::{FaultKind, GuestMemory, HelperEnv, MemFault, Unwind};
use ruvm_mem::{AddressSpace, MemTxAttrs, MemTxResult};

use crate::cputlb::{self, CpuTlb};
use crate::jit::Jit;
use crate::tb::{Tb, TbCpuState, lock};
use crate::tb_maint::{JcEntry, JcPending};
use crate::translate::TbBuild;
use crate::{ENV_CAN_DO_IO_OFFSET, TB_JMP_CACHE_SIZE, bp, cf, excp, interrupt};

/// The kind of a memory access, `MMUAccessType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MmuAccessType {
    /// `MMU_DATA_LOAD`.
    DataLoad = 0,
    /// `MMU_DATA_STORE`.
    DataStore = 1,
    /// `MMU_INST_FETCH`.
    InstFetch = 2,
}

/// Where a helper or slow path was called from, QEMU's `retaddr`.
///
/// `Ra::Tb` means the call came from the block that is running, and the guest state is restored
/// to the instruction whose `insn_start` ran last. `Ra::None` is QEMU's 0: not from generated
/// code, nothing to restore.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ra {
    /// Not called from generated code.
    None,
    /// Called from the running block.
    Tb,
}

/// Proof that the CPU state was set up for leaving the execution loop, the Rust form of
/// `cpu_loop_exit()`. Only the `cpu_loop_exit*` functions make one.
#[must_use = "a CpuLoopExit must be returned to the execution loop"]
pub struct CpuLoopExit {
    _private: (),
}

impl fmt::Debug for CpuLoopExit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CpuLoopExit")
    }
}

/// `CPUBreakpoint`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Breakpoint {
    /// The guest PC.
    pub pc: u64,
    /// `BP_*` flags.
    pub flags: u32,
}

/// `CPUWatchpoint`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Watchpoint {
    /// First watched address.
    pub vaddr: u64,
    /// Number of bytes watched.
    pub len: u64,
    /// The address that hit.
    pub hitaddr: u64,
    /// The attributes of the access that hit.
    pub hitattrs: MemTxAttrs,
    /// `BP_*` flags, including the hit bits.
    pub flags: u32,
}

/// The target hooks, `TCGCPUOps` plus the few `CPUClass` methods TCG calls.
pub trait CpuOps: Send + Sync + fmt::Debug {
    /// `translate_code`: translate the block described by `tb`, normally by calling
    /// [`crate::translator_loop`].
    fn translate_code(&self, cpu: &mut Cpu<'_>, tb: &mut TbBuild) -> Result<(), CpuLoopExit>;

    /// `get_tb_cpu_state`. The `cflags` field is ignored.
    fn get_tb_cpu_state(&self, cpu: &Cpu<'_>) -> TbCpuState;

    /// `synchronize_from_tb`: set the state to the start of `tb`. The default sets the PC, as
    /// QEMU does when the hook is missing.
    fn synchronize_from_tb(&self, cpu: &mut Cpu<'_>, tb: &Tb) {
        assert!(tb.cflags() & cf::PCREL == 0, "synchronize_from_tb needed for CF_PCREL");
        self.set_pc(cpu, tb.pc);
    }

    /// `restore_state_to_opc`: set the state to the instruction whose `insn_start` words are
    /// `data`.
    fn restore_state_to_opc(&self, cpu: &mut Cpu<'_>, tb: &Tb, data: &[u64; INSN_START_WORDS]);

    /// `CPUClass::set_pc`.
    fn set_pc(&self, cpu: &mut Cpu<'_>, pc: u64);

    /// `CPUClass::get_pc`.
    fn get_pc(&self, cpu: &Cpu<'_>) -> u64;

    /// `cpu_exec_enter`.
    fn cpu_exec_enter(&self, cpu: &mut Cpu<'_>) {
        let _ = cpu;
    }

    /// `cpu_exec_exit`.
    fn cpu_exec_exit(&self, cpu: &mut Cpu<'_>) {
        let _ = cpu;
    }

    /// `cpu_exec_interrupt`: take an interrupt from `interrupt_request` if one is deliverable.
    fn cpu_exec_interrupt(&self, cpu: &mut Cpu<'_>, interrupt_request: u32) -> bool {
        let _ = (cpu, interrupt_request);
        false
    }

    /// `cpu_exec_halt`: whether a halted CPU should wake up. The default is
    /// [`CpuOps::has_work`].
    fn cpu_exec_halt(&self, cpu: &mut Cpu<'_>) -> bool {
        self.has_work(cpu)
    }

    /// `cpu_exec_reset`, for `CPU_INTERRUPT_RESET`.
    fn cpu_exec_reset(&self, cpu: &mut Cpu<'_>) {
        let _ = cpu;
    }

    /// `do_interrupt`: deliver `exception_index`.
    fn do_interrupt(&self, cpu: &mut Cpu<'_>);

    /// `SysemuCPUOps::has_work`. The default looks for `CPU_INTERRUPT_HARD`.
    fn has_work(&self, cpu: &Cpu<'_>) -> bool {
        cpu.core.shared.interrupt_request() & interrupt::HARD != 0
    }

    /// `tlb_fill`: find the translation of `addr` and install it with
    /// [`Cpu::tlb_set_page_full`] or [`Cpu::tlb_set_page`]. With `probe` a failed translation
    /// returns `Ok(false)`; otherwise it raises the guest exception.
    #[allow(clippy::too_many_arguments)]
    fn tlb_fill(
        &self,
        cpu: &mut Cpu<'_>,
        addr: u64,
        size: usize,
        access_type: MmuAccessType,
        mmu_idx: usize,
        probe: bool,
        ra: Ra,
    ) -> Result<bool, CpuLoopExit>;

    /// `do_unaligned_access`: raise the alignment fault.
    fn do_unaligned_access(
        &self,
        cpu: &mut Cpu<'_>,
        addr: u64,
        access_type: MmuAccessType,
        mmu_idx: usize,
        ra: Ra,
    ) -> CpuLoopExit;

    /// `do_transaction_failed`. The default ignores the failure.
    #[allow(clippy::too_many_arguments)]
    fn do_transaction_failed(
        &self,
        cpu: &mut Cpu<'_>,
        physaddr: u64,
        addr: u64,
        size: usize,
        access_type: MmuAccessType,
        mmu_idx: usize,
        attrs: MemTxAttrs,
        response: MemTxResult,
        ra: Ra,
    ) -> Result<(), CpuLoopExit> {
        let _ = (cpu, physaddr, addr, size, access_type, mmu_idx, attrs, response, ra);
        Ok(())
    }

    /// `cpu_mmu_index`.
    fn mmu_index(&self, cpu: &Cpu<'_>, ifetch: bool) -> usize;

    /// `pointer_wrap`: the address of the second page of an access that crosses a page. The
    /// default does not wrap.
    fn pointer_wrap(&self, cpu: &Cpu<'_>, mmu_idx: usize, result: u64, base: u64) -> u64 {
        let _ = (cpu, mmu_idx, base);
        result
    }

    /// `debug_excp_handler`.
    fn debug_excp_handler(&self, cpu: &mut Cpu<'_>) {
        let _ = cpu;
    }

    /// `debug_check_watchpoint`: whether a CPU watchpoint really hit.
    fn debug_check_watchpoint(&self, cpu: &mut Cpu<'_>, wp: &Watchpoint) -> bool {
        let _ = (cpu, wp);
        true
    }

    /// `debug_check_breakpoint`: whether a CPU breakpoint really hit.
    fn debug_check_breakpoint(&self, cpu: &mut Cpu<'_>) -> bool {
        let _ = cpu;
        true
    }

    /// `adjust_watchpoint_address`.
    fn adjust_watchpoint_address(&self, cpu: &mut Cpu<'_>, addr: u64, len: u64) -> u64 {
        let _ = (cpu, len);
        addr
    }

    /// `guest_default_memory_order`, `TCG_MO_*` bits.
    fn guest_default_memory_order(&self) -> u32 {
        0
    }

    /// The guest address type, `TARGET_LONG_BITS` as an IR type.
    fn addr_type(&self) -> Type {
        Type::I64
    }

    /// `TARGET_HAS_PRECISE_SMC`.
    fn precise_smc(&self) -> bool {
        false
    }

    /// The target hooks as [`std::any::Any`], so that target helpers can reach their own
    /// per-CPU-type data (models, I/O address spaces) through [`Cpu::ops`]. QEMU gets there
    /// with `env_archcpu()`.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }
}

type WorkFn = Box<dyn FnOnce(&mut Cpu<'_>) + Send>;

struct WorkItem {
    f: WorkFn,
    exclusive: bool,
    done: Option<Arc<AtomicBool>>,
}

/// The part of a vCPU that other threads use.
pub struct CpuShared {
    /// `cpu_index`.
    pub cpu_index: usize,
    pub(crate) jit: Weak<Jit>,
    /// `icount_decr`: the high half is the exit request.
    pub(crate) icount_decr: AtomicU32,
    pub(crate) exit_request: AtomicBool,
    interrupt_request: AtomicU32,
    /// `halted`.
    pub halted: AtomicU32,
    pub(crate) running: AtomicBool,
    pub(crate) has_waiter: AtomicBool,
    /// `stop`: the vCPU thread should stop.
    pub stop: AtomicBool,
    /// `stopped`.
    pub stopped: AtomicBool,
    /// `unplug`: the vCPU thread should exit.
    pub unplug: AtomicBool,
    pub(crate) thread_kicked: AtomicBool,
    work: Mutex<VecDeque<WorkItem>>,
    pub(crate) halt: Arc<(Mutex<()>, Condvar)>,
    pub(crate) thread: Mutex<Option<ThreadId>>,
    pub(crate) tlb: Mutex<CpuTlb>,
    /// The part of `tlb` generated code reads.
    pub(crate) fast_tlb: Arc<ruvm_jit_interp::FastTlb>,
    /// Jump cache invalidations for the vCPU to apply, and whether there are any.
    pub(crate) jc_pending: Mutex<JcPending>,
    pub(crate) jc_dirty: AtomicBool,
}

impl fmt::Debug for CpuShared {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuShared")
            .field("cpu_index", &self.cpu_index)
            .field("interrupt_request", &self.interrupt_request())
            .field("halted", &self.halted.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl CpuShared {
    pub(crate) fn new(
        cpu_index: usize,
        jit: Weak<Jit>,
        halt: Arc<(Mutex<()>, Condvar)>,
        tlb: CpuTlb,
    ) -> CpuShared {
        CpuShared {
            cpu_index,
            jit,
            icount_decr: AtomicU32::new(0),
            exit_request: AtomicBool::new(false),
            interrupt_request: AtomicU32::new(0),
            halted: AtomicU32::new(0),
            running: AtomicBool::new(false),
            has_waiter: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            stopped: AtomicBool::new(false),
            unplug: AtomicBool::new(false),
            thread_kicked: AtomicBool::new(false),
            work: Mutex::new(VecDeque::new()),
            halt,
            thread: Mutex::new(None),
            fast_tlb: Arc::clone(tlb.fast()),
            tlb: Mutex::new(tlb),
            jc_pending: Mutex::new(JcPending::default()),
            jc_dirty: AtomicBool::new(false),
        }
    }

    /// `interrupt_request`.
    pub fn interrupt_request(&self) -> u32 {
        self.interrupt_request.load(Ordering::Acquire)
    }

    /// `cpu_test_interrupt()`.
    pub fn test_interrupt(&self, mask: u32) -> bool {
        self.interrupt_request() & mask != 0
    }

    /// `cpu_set_interrupt()`.
    pub fn set_interrupt(&self, mask: u32) {
        self.interrupt_request.fetch_or(mask, Ordering::AcqRel);
    }

    /// `cpu_reset_interrupt()`.
    pub fn reset_interrupt(&self, mask: u32) {
        self.interrupt_request.fetch_and(!mask, Ordering::AcqRel);
    }

    /// `qemu_cpu_is_self()`.
    pub fn is_self(&self) -> bool {
        *lock(&self.thread) == Some(std::thread::current().id())
    }

    /// `cpu_interrupt()` with TCG's `tcg_handle_interrupt()`: raise `mask` and make the vCPU
    /// notice.
    pub fn cpu_interrupt(&self, mask: u32) {
        self.set_interrupt(mask);
        if !self.is_self() {
            self.kick();
        } else {
            self.set_exit_high();
        }
    }

    pub(crate) fn set_exit_high(&self) {
        self.icount_decr.fetch_or(0xffff_0000, Ordering::Release);
    }

    /// `cpu_exit()`: leave the execution loop soon.
    pub fn cpu_exit(&self) {
        self.exit_request.store(true, Ordering::Release);
        self.set_exit_high();
    }

    /// `qemu_cpu_kick()`: wake the vCPU if it waits and make it leave the execution loop.
    pub fn kick(&self) {
        {
            let _g = lock(&self.halt.0);
            self.halt.1.notify_all();
        }
        match self.jit.upgrade() {
            Some(jit) if !jit.config.mttcg => jit.rr_kick_next_cpu(),
            _ => self.cpu_exit(),
        }
    }

    /// `cpu_work_list_empty()`.
    pub fn work_list_empty(&self) -> bool {
        lock(&self.work).is_empty()
    }

    fn queue_work(&self, item: WorkItem) {
        lock(&self.work).push_back(item);
        self.kick();
    }

    /// `async_run_on_cpu()`: run `f` on the vCPU's thread.
    pub fn async_run_on_cpu(&self, f: impl FnOnce(&mut Cpu<'_>) + Send + 'static) {
        self.queue_work(WorkItem { f: Box::new(f), exclusive: false, done: None });
    }

    /// `async_safe_run_on_cpu()`: run `f` on the vCPU's thread with every other vCPU stopped.
    pub fn async_safe_run_on_cpu(&self, f: impl FnOnce(&mut Cpu<'_>) + Send + 'static) {
        self.queue_work(WorkItem { f: Box::new(f), exclusive: true, done: None });
    }

    /// `run_on_cpu()`: run `f` on the vCPU's thread and wait for it.
    ///
    /// # Panics
    ///
    /// If called from the vCPU's own thread, which would wait forever.
    pub fn run_on_cpu(&self, f: impl FnOnce(&mut Cpu<'_>) + Send + 'static) {
        assert!(!self.is_self(), "run_on_cpu called from the vCPU's own thread");
        let jit = self.jit.upgrade().expect("run_on_cpu after the runtime was dropped");
        let done = Arc::new(AtomicBool::new(false));
        self.queue_work(WorkItem { f: Box::new(f), exclusive: false, done: Some(done.clone()) });
        let mut g = lock(&jit.work_lock);
        while !done.load(Ordering::Acquire) {
            g = jit.work_cond.wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }
}

thread_local! {
    static EXCLUSIVE_CONTEXT_COUNT: Cell<u32> = const { Cell::new(0) };
}

/// `cpu_in_exclusive_context()` for the calling thread.
pub fn in_exclusive_context() -> bool {
    EXCLUSIVE_CONTEXT_COUNT.with(|c| c.get() != 0)
}

impl Jit {
    /// `start_exclusive()`: wait until no other vCPU runs generated code and keep them out
    /// until [`Jit::end_exclusive`]. Nests.
    pub fn start_exclusive(&self) {
        let count = EXCLUSIVE_CONTEXT_COUNT.with(Cell::get);
        if count != 0 {
            EXCLUSIVE_CONTEXT_COUNT.with(|c| c.set(count + 1));
            return;
        }
        let mut g = lock(&self.list_lock);
        g = self.exclusive_idle(g);

        // Make all other CPUs stop executing.
        self.pending_cpus.store(1, Ordering::SeqCst);
        let mut running_cpus = 0;
        for other in self.cpu_list() {
            if other.running.load(Ordering::SeqCst) {
                other.has_waiter.store(true, Ordering::SeqCst);
                running_cpus += 1;
                other.kick();
            }
        }
        self.pending_cpus.store(running_cpus + 1, Ordering::SeqCst);
        while self.pending_cpus.load(Ordering::SeqCst) > 1 {
            g = self.exclusive_cond.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        drop(g);
        EXCLUSIVE_CONTEXT_COUNT.with(|c| c.set(1));
    }

    /// `end_exclusive()`.
    pub fn end_exclusive(&self) {
        let count = EXCLUSIVE_CONTEXT_COUNT.with(Cell::get);
        assert!(count > 0, "end_exclusive without start_exclusive");
        EXCLUSIVE_CONTEXT_COUNT.with(|c| c.set(count - 1));
        if count > 1 {
            return;
        }
        let _g = lock(&self.list_lock);
        self.pending_cpus.store(0, Ordering::SeqCst);
        self.exclusive_resume.notify_all();
    }

    fn exclusive_idle<'a>(
        &self,
        mut g: std::sync::MutexGuard<'a, ()>,
    ) -> std::sync::MutexGuard<'a, ()> {
        while self.pending_cpus.load(Ordering::SeqCst) != 0 {
            g = self.exclusive_resume.wait(g).unwrap_or_else(|e| e.into_inner());
        }
        g
    }

    /// `cpu_exec_start()`: wait for exclusive sections to finish and mark the vCPU running.
    pub(crate) fn cpu_exec_start(&self, cpu: &CpuShared) {
        cpu.running.store(true, Ordering::SeqCst);
        if self.pending_cpus.load(Ordering::SeqCst) != 0 {
            let g = lock(&self.list_lock);
            if !cpu.has_waiter.load(Ordering::SeqCst) {
                cpu.running.store(false, Ordering::SeqCst);
                let _g = self.exclusive_idle(g);
                cpu.running.store(true, Ordering::SeqCst);
            }
        }
    }

    /// `cpu_exec_end()`.
    pub(crate) fn cpu_exec_end(&self, cpu: &CpuShared) {
        cpu.running.store(false, Ordering::SeqCst);
        if self.pending_cpus.load(Ordering::SeqCst) != 0 {
            let _g = lock(&self.list_lock);
            if cpu.has_waiter.load(Ordering::SeqCst) {
                cpu.has_waiter.store(false, Ordering::SeqCst);
                let n = self.pending_cpus.fetch_sub(1, Ordering::SeqCst) - 1;
                if n == 1 {
                    self.exclusive_cond.notify_one();
                }
            }
        }
    }
}

/// The part of a vCPU only its own thread uses.
pub struct CpuCore {
    pub(crate) jit: Arc<Jit>,
    pub(crate) shared: Arc<CpuShared>,
    pub(crate) ops: Arc<dyn CpuOps>,
    pub(crate) as_: Arc<AddressSpace>,
    /// `exception_index`, -1 for none.
    pub exception_index: i32,
    /// `cflags_next_tb`, `u32::MAX` for QEMU's -1.
    pub cflags_next_tb: u32,
    /// `tcg_cflags`.
    pub tcg_cflags: u32,
    /// `singlestep_enabled`.
    pub singlestep_enabled: bool,
    /// `ignore_memory_transaction_failures`.
    pub ignore_memory_transaction_failures: bool,
    pub(crate) current_tb: Option<Arc<Tb>>,
    pub(crate) cur_insn: Option<[u64; INSN_START_WORDS]>,
    pub(crate) unwinding: Option<CpuLoopExit>,
    pub(crate) goto_ptr_target: Option<Arc<Tb>>,
    pub(crate) atomic_depth: u32,
    /// `tb_jmp_cache`, which only this vCPU reads or writes.
    pub(crate) jmp_cache: Vec<JcEntry>,
    /// `breakpoints`, GDB ones first.
    pub breakpoints: Vec<Breakpoint>,
    /// `watchpoints`, GDB ones first.
    pub watchpoints: Vec<Watchpoint>,
    /// `watchpoint_hit`, as an index into `watchpoints`.
    pub watchpoint_hit: Option<usize>,
}

/// `cpu_exec_unrealizefn()`: a vCPU leaves the CPU list when it goes away, so its TLB is freed
/// and a runtime that creates vCPUs one after another does not keep every one of them.
impl Drop for CpuCore {
    fn drop(&mut self) {
        self.jit.cpu_list_remove(&self.shared);
    }
}

impl fmt::Debug for CpuCore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuCore")
            .field("cpu_index", &self.shared.cpu_index)
            .field("exception_index", &self.exception_index)
            .field("cflags_next_tb", &self.cflags_next_tb)
            .field("tcg_cflags", &self.tcg_cflags)
            .finish_non_exhaustive()
    }
}

impl CpuCore {
    pub(crate) fn new(
        jit: Arc<Jit>,
        shared: Arc<CpuShared>,
        ops: Arc<dyn CpuOps>,
        as_: Arc<AddressSpace>,
        tcg_cflags: u32,
    ) -> CpuCore {
        CpuCore {
            jit,
            shared,
            ops,
            as_,
            exception_index: -1,
            cflags_next_tb: u32::MAX,
            tcg_cflags,
            singlestep_enabled: false,
            ignore_memory_transaction_failures: false,
            current_tb: None,
            cur_insn: None,
            unwinding: None,
            goto_ptr_target: None,
            atomic_depth: 0,
            jmp_cache: vec![JcEntry::default(); TB_JMP_CACHE_SIZE],
            breakpoints: Vec::new(),
            watchpoints: Vec::new(),
            watchpoint_hit: None,
        }
    }

    /// The runtime.
    pub fn jit(&self) -> &Arc<Jit> {
        &self.jit
    }

    /// The shared half.
    pub fn shared(&self) -> &Arc<CpuShared> {
        &self.shared
    }

    /// The target hooks.
    pub fn ops(&self) -> &Arc<dyn CpuOps> {
        &self.ops
    }

    /// The address space the CPU's physical accesses go to.
    pub fn address_space(&self) -> &Arc<AddressSpace> {
        &self.as_
    }

    /// The block that is running, if any.
    pub fn current_tb(&self) -> Option<&Arc<Tb>> {
        self.current_tb.as_ref()
    }

    fn fault(&mut self, e: CpuLoopExit, addr: u64, write: bool, oi: MemOpIdx) -> MemFault {
        self.unwinding = Some(e);
        MemFault { addr, write, oi, kind: FaultKind::Protection }
    }
}

impl GuestMemory for CpuCore {
    fn read(&mut self, addr: u64, buf: &mut [u8], oi: MemOpIdx) -> Result<(), MemFault> {
        let mut env = [0u8; 0];
        self.read_with_env(&mut env, addr, buf, oi)
    }

    fn write(&mut self, addr: u64, data: &[u8], oi: MemOpIdx) -> Result<(), MemFault> {
        let mut env = [0u8; 0];
        self.write_with_env(&mut env, addr, data, oi)
    }

    fn read_with_env(
        &mut self,
        env: &mut [u8],
        addr: u64,
        buf: &mut [u8],
        oi: MemOpIdx,
    ) -> Result<(), MemFault> {
        let r = {
            let mut cpu = Cpu { env, core: self };
            cputlb::do_ld_bytes(&mut cpu, addr, buf, oi, MmuAccessType::DataLoad, Ra::Tb)
        };
        r.map_err(|e| self.fault(e, addr, false, oi))
    }

    fn write_with_env(
        &mut self,
        env: &mut [u8],
        addr: u64,
        data: &[u8],
        oi: MemOpIdx,
    ) -> Result<(), MemFault> {
        let r = {
            let mut cpu = Cpu { env, core: self };
            cputlb::do_st_bytes(&mut cpu, addr, data, oi, Ra::Tb)
        };
        r.map_err(|e| self.fault(e, addr, true, oi))
    }

    fn atomic_access(
        &mut self,
        env: &mut [u8],
        addr: u64,
        oi: MemOpIdx,
        op: &mut dyn FnMut(&[AtomicU8]),
    ) -> Result<bool, Unwind> {
        let size = oi.memop().size_bytes() as usize;
        let r = {
            let mut cpu = Cpu { env, core: self };
            cputlb::atomic_mmu_lookup(&mut cpu, addr, oi, size, Ra::Tb)
        };
        let (block, off) = r.map_err(|e| Unwind::Mem(self.fault(e, addr, true, oi)))?;
        let bytes = usize::try_from(off)
            .ok()
            .and_then(|o| block.atomic_bytes().get(o..o.checked_add(size)?));
        match bytes {
            Some(b) => op(b),
            None => return Err(Unwind::ExitAtomic),
        }
        Ok(true)
    }

    fn atomic_begin(&mut self) {
        if self.atomic_depth == 0 {
            self.jit.atomic_lock();
        }
        self.atomic_depth += 1;
    }

    fn atomic_end(&mut self) {
        self.atomic_depth -= 1;
        if self.atomic_depth == 0 {
            self.jit.atomic_unlock();
        }
    }

    fn insn_start(&mut self, words: &[u64; INSN_START_WORDS]) {
        self.cur_insn = Some(*words);
    }

    fn enter_block(&mut self, block: Option<Arc<dyn std::any::Any + Send + Sync>>) {
        self.current_tb = block.and_then(|b| b.downcast::<Tb>().ok());
        self.cur_insn = None;
    }

    fn fast_tlb(&self) -> Option<Arc<ruvm_jit_interp::FastTlb>> {
        Some(Arc::clone(&self.shared.fast_tlb))
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

/// A vCPU: its `env` buffer and its core.
#[derive(Debug)]
pub struct Vcpu {
    /// The CPU state buffer generated code runs against.
    pub env: Vec<u8>,
    /// The rest of the vCPU.
    pub core: CpuCore,
}

impl Vcpu {
    /// Borrow as a [`Cpu`].
    pub fn cpu(&mut self) -> Cpu<'_> {
        Cpu { env: &mut self.env, core: &mut self.core }
    }

    /// The shared half.
    pub fn shared(&self) -> &Arc<CpuShared> {
        &self.core.shared
    }
}

/// A borrowed vCPU, what `CPUState *` is in QEMU's TCG code.
#[derive(Debug)]
pub struct Cpu<'a> {
    /// The CPU state buffer.
    pub env: &'a mut [u8],
    /// The rest of the vCPU.
    pub core: &'a mut CpuCore,
}

impl<'a> Cpu<'a> {
    /// Reborrow.
    pub fn rb(&mut self) -> Cpu<'_> {
        Cpu { env: self.env, core: self.core }
    }

    /// The CPU a helper runs on. `None` if the helper was not called by this runtime.
    #[inline]
    pub fn from_helper_env<'b>(h: &'b mut HelperEnv<'_>) -> Option<Cpu<'b>> {
        let core = h.mem.as_any_mut()?.downcast_mut::<CpuCore>()?;
        Some(Cpu { env: &mut *h.env, core })
    }

    /// Hand a [`CpuLoopExit`] from inside a helper back to the runtime. The helper returns the
    /// [`Unwind`] this gives.
    pub fn unwind(&mut self, e: CpuLoopExit) -> Unwind {
        self.core.unwinding = Some(e);
        Unwind::Exception(0)
    }

    /// The runtime.
    pub fn jit(&self) -> Arc<Jit> {
        self.core.jit.clone()
    }

    /// The target hooks.
    pub fn ops(&self) -> Arc<dyn CpuOps> {
        self.core.ops.clone()
    }

    /// The shared half.
    pub fn shared(&self) -> Arc<CpuShared> {
        self.core.shared.clone()
    }

    /// `cpu->neg.can_do_io`.
    pub fn can_do_io(&self) -> bool {
        self.env[ENV_CAN_DO_IO_OFFSET as usize] != 0
    }

    pub(crate) fn set_can_do_io(&mut self, v: bool) {
        self.env[ENV_CAN_DO_IO_OFFSET as usize] = u8::from(v);
    }

    /// `curr_cflags()`.
    pub fn curr_cflags(&self) -> u32 {
        let mut cflags = self.core.tcg_cflags;
        if self.core.singlestep_enabled {
            cflags |= cf::NO_GOTO_TB | cf::NO_GOTO_PTR | cf::SINGLE_STEP | 1;
        } else if self.core.jit.config.one_insn_per_tb {
            cflags |= cf::NO_GOTO_TB | 1;
        } else if self.core.jit.config.nochain {
            cflags |= cf::NO_GOTO_TB;
        }
        cflags
    }

    /// `cpu_in_serial_context()`.
    pub fn in_serial_context(&self) -> bool {
        self.core.tcg_cflags & cf::PARALLEL == 0 || in_exclusive_context()
    }

    /// `cpu_single_stepping()`.
    pub fn single_stepping(&self) -> bool {
        self.core.singlestep_enabled
    }

    /// `cpu_loop_exit()`.
    pub fn cpu_loop_exit(&mut self) -> CpuLoopExit {
        // Undo the setting in cpu_tb_exec.
        self.set_can_do_io(true);
        CpuLoopExit { _private: () }
    }

    /// `cpu_loop_exit_noexc()`.
    pub fn cpu_loop_exit_noexc(&mut self) -> CpuLoopExit {
        self.core.exception_index = -1;
        self.cpu_loop_exit()
    }

    /// `cpu_loop_exit_restore()`.
    pub fn cpu_loop_exit_restore(&mut self, ra: Ra) -> CpuLoopExit {
        if ra != Ra::None {
            self.cpu_restore_state(ra);
        }
        self.cpu_loop_exit()
    }

    /// `cpu_loop_exit_atomic()`.
    pub fn cpu_loop_exit_atomic(&mut self, ra: Ra) -> CpuLoopExit {
        // Prevent looping if already executing in a serial context.
        assert!(!self.in_serial_context(), "cpu_loop_exit_atomic in a serial context");
        self.core.exception_index = excp::ATOMIC;
        self.cpu_loop_exit_restore(ra)
    }

    /// Raise guest exception `excp`, restoring the state to the current instruction.
    pub fn raise_exception(&mut self, excp: i32, ra: Ra) -> CpuLoopExit {
        self.core.exception_index = excp;
        self.cpu_loop_exit_restore(ra)
    }

    /// `cpu_restore_state()`: set the guest state to the instruction that made the call.
    pub fn cpu_restore_state(&mut self, ra: Ra) -> bool {
        if ra == Ra::None {
            return false;
        }
        let (Some(tb), Some(data)) = (self.core.current_tb.clone(), self.core.cur_insn) else {
            return false;
        };
        let ops = self.ops();
        ops.restore_state_to_opc(self, &tb, &data);
        true
    }

    /// `cpu_breakpoint_insert()`.
    pub fn breakpoint_insert(&mut self, pc: u64, flags: u32) {
        let b = Breakpoint { pc, flags };
        if flags & bp::GDB != 0 {
            self.core.breakpoints.insert(0, b);
        } else {
            self.core.breakpoints.push(b);
        }
    }

    /// `cpu_breakpoint_remove()`: `false` (QEMU's -ENOENT) if there is no such breakpoint.
    pub fn breakpoint_remove(&mut self, pc: u64, flags: u32) -> bool {
        match self.core.breakpoints.iter().position(|b| b.pc == pc && b.flags == flags) {
            Some(i) => {
                self.core.breakpoints.remove(i);
                true
            }
            None => false,
        }
    }

    /// `cpu_watchpoint_insert()`: `false` (QEMU's -EINVAL) for an empty range or one that
    /// runs off the end of the address space.
    pub fn watchpoint_insert(&mut self, addr: u64, len: u64, flags: u32) -> bool {
        if len == 0 || addr.wrapping_add(len - 1) < addr {
            eprintln!("tried to set invalid watchpoint at {addr:x}, len={len}");
            return false;
        }
        let wp =
            Watchpoint { vaddr: addr, len, hitaddr: 0, hitattrs: MemTxAttrs::default(), flags };
        if flags & bp::GDB != 0 {
            self.core.watchpoints.insert(0, wp);
            if let Some(h) = self.core.watchpoint_hit.as_mut() {
                *h += 1;
            }
        } else {
            self.core.watchpoints.push(wp);
        }
        let page_mask = self.core.jit.page_mask();
        let in_page = (addr | page_mask).wrapping_neg();
        if len <= in_page {
            cputlb::tlb_flush_page(self, addr);
        } else {
            cputlb::tlb_flush(self);
        }
        true
    }

    /// `cpu_watchpoint_remove()`.
    pub fn watchpoint_remove(&mut self, addr: u64, len: u64, flags: u32) -> bool {
        let pos = self.core.watchpoints.iter().position(|w| {
            w.vaddr == addr && w.len == len && flags == (w.flags & !bp::WATCHPOINT_HIT)
        });
        match pos {
            Some(i) => {
                self.watchpoint_remove_by_ref(i);
                true
            }
            None => false,
        }
    }

    /// `cpu_watchpoint_remove_by_ref()`.
    pub fn watchpoint_remove_by_ref(&mut self, i: usize) {
        let wp = self.core.watchpoints.remove(i);
        match self.core.watchpoint_hit {
            Some(h) if h == i => self.core.watchpoint_hit = None,
            Some(h) if h > i => self.core.watchpoint_hit = Some(h - 1),
            _ => {}
        }
        cputlb::tlb_flush_page(self, wp.vaddr);
    }

    /// `cpu_watchpoint_address_matches()`: the flags of the watchpoints covering the range.
    pub fn watchpoint_address_matches(&self, addr: u64, len: u64) -> u32 {
        self.core
            .watchpoints
            .iter()
            .filter(|w| watchpoint_address_matches(w, addr, len))
            .fold(0, |acc, w| acc | w.flags)
    }

    /// `cpu_check_watchpoint()`: raise a debug exception if a watchpoint covers the access.
    pub fn check_watchpoint(
        &mut self,
        addr: u64,
        len: u64,
        attrs: MemTxAttrs,
        flags: u32,
        ra: Ra,
    ) -> Result<(), CpuLoopExit> {
        if self.core.watchpoint_hit.is_some() {
            // We re-entered the check after replacing the TB. Now raise the debug interrupt
            // so that it will trigger after the current instruction.
            self.core.shared.cpu_interrupt(interrupt::DEBUG);
            return Ok(());
        }
        let ops = self.ops();
        let addr = ops.adjust_watchpoint_address(self, addr, len);
        assert!(flags & !bp::MEM_ACCESS == 0);
        for i in 0..self.core.watchpoints.len() {
            let wp = self.core.watchpoints[i];
            let hit_flags = wp.flags & flags;
            if hit_flags != 0 && watchpoint_address_matches(&wp, addr, len) {
                {
                    let w = &mut self.core.watchpoints[i];
                    w.flags |= hit_flags << bp::HIT_SHIFT;
                    w.hitaddr = addr.max(w.vaddr);
                    w.hitattrs = attrs;
                }
                let w = self.core.watchpoints[i];
                if w.flags & bp::CPU != 0 && !ops.debug_check_watchpoint(self, &w) {
                    self.core.watchpoints[i].flags &= !bp::WATCHPOINT_HIT;
                    continue;
                }
                self.core.watchpoint_hit = Some(i);

                // This call also restores vCPU state.
                crate::translate::tb_check_watchpoint(self, ra);
                if w.flags & bp::STOP_BEFORE_ACCESS != 0 {
                    self.core.exception_index = excp::DEBUG;
                    return Err(self.cpu_loop_exit());
                }
                // Force execution of one insn next time.
                self.core.cflags_next_tb = 1 | cf::NOIRQ | self.curr_cflags();
                return Err(self.cpu_loop_exit_noexc());
            } else {
                self.core.watchpoints[i].flags &= !bp::WATCHPOINT_HIT;
            }
        }
        Ok(())
    }

    /// `process_queued_cpu_work()`.
    pub fn process_queued_cpu_work(&mut self) {
        let shared = self.shared();
        let jit = self.jit();
        let mut any = false;
        loop {
            let item = lock(&shared.work).pop_front();
            let Some(item) = item else { break };
            any = true;
            if item.exclusive {
                jit.start_exclusive();
                (item.f)(self);
                jit.end_exclusive();
            } else {
                (item.f)(self);
            }
            if let Some(done) = item.done {
                let _g = lock(&jit.work_lock);
                done.store(true, Ordering::Release);
            }
        }
        if any {
            let _g = lock(&jit.work_lock);
            jit.work_cond.notify_all();
        }
    }

    /// `cpu_has_work()`.
    pub fn has_work(&self) -> bool {
        self.core.ops.has_work(self)
    }
}

fn watchpoint_address_matches(wp: &Watchpoint, addr: u64, len: u64) -> bool {
    // The lengths are non-zero, but the range may end exactly at the top of the address
    // space, where addr + len wraps round to zero.
    let wpend = wp.vaddr.wrapping_add(wp.len - 1);
    let addrend = addr.wrapping_add(len - 1);
    !(addr > wpend || wp.vaddr > addrend)
}
