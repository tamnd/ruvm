// SPDX-License-Identifier: GPL-2.0-or-later

//! Running the virt board on TCG: the parts of accel/tcg/tcg-accel-ops*.c, system/cpus.c and
//! system/runstate.c that put a [`VirtMachine`] on the vCPUs of the AArch64 front end
//! (`ruvm-target-arm`) running on `ruvm-jit`.
//!
//! [`VirtTcgMachine::new`] takes a board with its devices plugged. It creates the runtime with
//! the `-accel tcg` options ([`TcgOptions`]), finishes the board with `machine_done()`, makes
//! one vCPU per CPU (the secondaries powered off, for PSCI CPU_ON to start) and starts the
//! vCPU threads stopped: `CPU n/TCG` with MTTCG (`thread=multi`, the default, as AArch64
//! supports it) and `ALL CPUs/TCG` in round robin mode (`thread=single`).
//! [`VirtTcgMachine::start`] lets them run. A thread fires the
//! timers of the board clocks (the generic timers and the RTC).
//!
//! PSCI SYSTEM_RESET stops every vCPU, resets the board, drops all translated code, resets each
//! vCPU on its own thread and lets them go again, as `qemu_system_reset()` does from the main
//! loop. With `-no-reboot` it is a shutdown instead. PSCI SYSTEM_OFF is reported to the
//! [`VirtEventHandler`] the caller gives.
//!
//! Deliberate differences from QEMU:
//!
//! - The timers of the board clocks are fired by a thread of their own rather than by the main
//!   loop's poll. It sleeps until the next deadline and is woken when a timer becomes the first
//!   to expire, as `timerlist_notify()` wakes QEMU's main loop, but it also wakes at least every
//!   100 ms (`TIMER_IDLE`) in case a wakeup was missed.
//! - Writes to RAM that do not come from a vCPU (DMA) do not invalidate translated code.
//! - A memory map change (`tcg_commit()`) queues a TLB flush on every vCPU, including the one
//!   that made it, which finishes its current block first; QEMU flushes that one at once.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use ruvm_accel::VcpuControl;
use ruvm_accel::tcg::{TcgOptions, TcgVcpus};
use ruvm_hw_core::Clock;
use ruvm_jit::Jit;
use ruvm_jit::cputlb::tlb_flush;
use ruvm_jit::native::{BackendKind, backend_of_kind};
use ruvm_target_arm::tcg::{helper_registry, jit_config};

use ruvm_mem::MemoryListener;

use crate::virt::{VirtMachine, VirtRequest};

/// The longest the timer thread sleeps. Arming a timer that becomes the first to fire wakes it
/// through the clock's notify hook, as `timerlist_notify()` kicks QEMU's main loop, so this only
/// bounds how long a missed wakeup could delay a timer.
const TIMER_IDLE: Duration = Duration::from_millis(100);

/// AArch64 has `TARGET_SUPPORTS_MTTCG`, so `thread=multi` gives no warning.
const ARM_SUPPORTS_MTTCG: bool = true;

/// Why the guest asked to stop.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ShutdownReason {
    /// PSCI SYSTEM_OFF, `SHUTDOWN_CAUSE_GUEST_SHUTDOWN`.
    GuestShutdown,
    /// PSCI SYSTEM_RESET with `-no-reboot` in effect, `SHUTDOWN_CAUSE_GUEST_RESET`.
    GuestReset,
}

/// What a running virt board tells its owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VirtEvent {
    /// The guest wants the machine to stop.
    Shutdown(ShutdownReason),
    /// The guest reset the machine and it has been reset.
    Reset,
    /// Resetting the board failed. The vCPUs stay stopped.
    InternalError(String),
}

/// Receives the [`VirtEvent`]s, on whatever thread they happen.
pub type VirtEventHandler = Arc<dyn Fn(VirtEvent) + Send + Sync>;

/// Options of the run loop.
#[derive(Clone, Debug, Default)]
pub struct VirtRunConfig {
    /// `-no-reboot`: a guest reset shuts the machine down instead.
    pub no_reboot: bool,
    /// The `-accel tcg` properties.
    pub tcg: TcgOptions,
    /// The code generator to use, for debugging: `None` picks the host's native one when
    /// there is one (or what `RUVM_JIT_BACKEND` asks for), `Some(BackendKind::Interp)` the
    /// IR interpreter.
    pub backend: Option<BackendKind>,
}

#[derive(Debug, Default)]
struct Ctl {
    /// The machine is started, `vm_start()`.
    running: bool,
    /// A reset was requested and not finished yet.
    reset_pending: bool,
    quit: bool,
}

struct Shared {
    ctl: Mutex<Ctl>,
    cv: Condvar,
    handler: VirtEventHandler,
    no_reboot: bool,
    quit: AtomicBool,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Ctl> {
        self.ctl.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `qemu_system_reset_request(SHUTDOWN_CAUSE_GUEST_RESET)`. The control thread does the
    /// work, so this is safe from a vCPU thread.
    fn request_reset(&self) {
        if self.no_reboot {
            (self.handler)(VirtEvent::Shutdown(ShutdownReason::GuestReset));
            return;
        }
        self.lock().reset_pending = true;
        self.cv.notify_all();
    }
}

fn lock_board(board: &Mutex<VirtMachine>) -> MutexGuard<'_, VirtMachine> {
    board.lock().unwrap_or_else(PoisonError::into_inner)
}

fn control_loop(shared: Arc<Shared>, vcpus: Arc<TcgVcpus>, board: Arc<Mutex<VirtMachine>>) {
    loop {
        let mut c = shared.lock();
        while !c.reset_pending && !c.quit {
            c = shared.cv.wait(c).unwrap_or_else(PoisonError::into_inner);
        }
        if c.quit {
            return;
        }
        drop(c);

        // qemu_system_reset(): stop the vCPUs, reset the devices, then every CPU.
        vcpus.pause_all();
        let res = lock_board(&board).system_reset();
        // The ROMs are copied back into RAM; nothing translated before may survive.
        vcpus.jit().tb_flush_exclusive_or_serial();
        let b = Arc::clone(&board);
        vcpus.run_on_each(move |cpu| lock_board(&b).reset_cpu(cpu));
        let ok = res.is_ok();
        {
            let mut c = shared.lock();
            c.reset_pending = false;
            if ok && c.running && !c.quit {
                vcpus.resume_all();
            }
        }
        match res {
            Ok(()) => (shared.handler)(VirtEvent::Reset),
            Err(e) => (shared.handler)(VirtEvent::InternalError(e)),
        }
    }
}

fn timer_loop(shared: Arc<Shared>, clocks: Vec<Arc<Clock>>) {
    for c in &clocks {
        let me = std::thread::current();
        c.set_notify(move || me.unpark());
    }
    while !shared.quit.load(Ordering::Acquire) {
        let mut sleep = TIMER_IDLE;
        for c in &clocks {
            c.run_timers();
            let d = c.deadline_ns();
            if d >= 0 {
                sleep = sleep.min(Duration::from_nanos(d as u64));
            }
        }
        if !sleep.is_zero() {
            std::thread::park_timeout(sleep);
        }
    }
}

/// `tcg_commit()`: a change to the memory map, such as a flash leaving romd mode for a
/// command, flushes every vCPU's TLB so no entry points at the old view. The flush is queued
/// on each vCPU, so the vCPU that made the change runs to the end of its block first; QEMU
/// flushes that one at once.
struct TlbCommit(std::sync::Weak<Jit>);

impl MemoryListener for TlbCommit {
    fn name(&self) -> &str {
        "tcg"
    }

    fn commit(&self) {
        let Some(jit) = self.0.upgrade() else { return };
        for cpu in jit.cpu_list() {
            cpu.async_run_on_cpu(tlb_flush);
        }
    }
}

/// The virt board running on TCG.
pub struct VirtTcgMachine {
    shared: Arc<Shared>,
    board: Arc<Mutex<VirtMachine>>,
    vcpus: Arc<TcgVcpus>,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl fmt::Debug for VirtTcgMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VirtTcgMachine")
            .field("vcpus", &self.vcpus.vcpu_count())
            .field("mttcg", &self.mttcg())
            .finish_non_exhaustive()
    }
}

fn spawn(name: &str, f: impl FnOnce() + Send + 'static) -> Result<JoinHandle<()>, String> {
    std::thread::Builder::new()
        .name(name.to_string())
        .spawn(f)
        .map_err(|e| format!("could not create thread: {e}"))
}

impl VirtTcgMachine {
    /// Creates the runtime for `cfg`, finishes the board, makes its vCPUs and starts their
    /// threads stopped. `clocks` are the clocks the board's timers run on (the board's
    /// [`VirtMachine::clock`] and the RTC clock); a thread fires them. Also gives the warnings
    /// to print, such as QEMU's for `thread=multi` on a guest without MTTCG.
    pub fn new(
        board: VirtMachine,
        clocks: Vec<Arc<Clock>>,
        cfg: &VirtRunConfig,
        handler: VirtEventHandler,
    ) -> Result<(VirtTcgMachine, Vec<String>), String> {
        let mut board = board;
        let (config, warnings) = cfg.tcg.jit_config(jit_config(), ARM_SUPPORTS_MTTCG)?;
        let backend = match cfg.backend {
            Some(kind) => backend_of_kind(kind, helper_registry(), config.code_gen_buffer_size),
            None => TcgOptions::backend(&config, helper_registry()),
        };
        let jit = Jit::new(config, backend);

        let shared = Arc::new(Shared {
            ctl: Mutex::new(Ctl::default()),
            cv: Condvar::new(),
            handler: Arc::clone(&handler),
            no_reboot: cfg.no_reboot,
            quit: AtomicBool::new(false),
        });
        {
            let s = Arc::downgrade(&shared);
            let h = Arc::clone(&handler);
            board.set_request_handler(Some(Arc::new(move |req| match req {
                VirtRequest::Shutdown => h(VirtEvent::Shutdown(ShutdownReason::GuestShutdown)),
                VirtRequest::Reset => {
                    if let Some(s) = s.upgrade() {
                        s.request_reset();
                    }
                }
            })));
        }

        board.machine_done()?;
        let vcpus = board.create_vcpus(&jit)?;
        board
            .memory_system()
            .register_listener(Arc::new(TlbCommit(Arc::downgrade(&jit))), board.memory_as())
            .map_err(|e| e.to_string())?;
        let board = Arc::new(Mutex::new(board));
        let vcpus = Arc::new(TcgVcpus::start(&jit, vcpus));

        let control = {
            let (s, v, b) = (Arc::clone(&shared), Arc::clone(&vcpus), Arc::clone(&board));
            spawn("reset", move || control_loop(s, v, b))?
        };
        let timers = {
            let s = Arc::clone(&shared);
            spawn("timers", move || timer_loop(s, clocks))?
        };
        let machine =
            VirtTcgMachine { shared, board, vcpus, threads: Mutex::new(vec![control, timers]) };
        Ok((machine, warnings))
    }

    /// The board, for the monitor and for device access.
    pub fn board(&self) -> &Arc<Mutex<VirtMachine>> {
        &self.board
    }

    /// The vCPUs.
    pub fn vcpus(&self) -> &Arc<TcgVcpus> {
        &self.vcpus
    }

    /// The number of vCPUs.
    pub fn vcpu_count(&self) -> usize {
        self.vcpus.vcpu_count()
    }

    /// Whether each vCPU has its own thread, `qemu_tcg_mttcg_enabled()`.
    pub fn mttcg(&self) -> bool {
        self.vcpus.jit().config.mttcg
    }

    /// `resume_all_vcpus()`.
    pub fn start(&self) {
        let mut c = self.shared.lock();
        c.running = true;
        if !c.reset_pending && !c.quit {
            self.vcpus.resume_all();
        }
    }

    /// `pause_all_vcpus()`. Waits until every vCPU is out of the guest, unless called from a
    /// vCPU thread, which only asks.
    pub fn pause(&self) {
        self.shared.lock().running = false;
        self.vcpus.pause_all();
    }

    /// `qemu_system_reset_request()` from outside the guest.
    pub fn request_reset(&self) {
        self.shared.request_reset();
    }

    /// Stops every thread and waits for them. Must not be called from a vCPU thread or from
    /// the [`VirtEventHandler`].
    pub fn quit(&self) {
        self.shared.lock().quit = true;
        self.shared.quit.store(true, Ordering::Release);
        self.shared.cv.notify_all();
        // The control thread first: a reset it is in the middle of needs the vCPU threads.
        let threads =
            std::mem::take(&mut *self.threads.lock().unwrap_or_else(PoisonError::into_inner));
        for t in threads {
            // The timer thread may be parked until its next deadline.
            t.thread().unpark();
            let _ = t.join();
        }
        lock_board(&self.board).set_request_handler(None);
        drop(self.vcpus.quit());
    }
}

impl Drop for VirtTcgMachine {
    fn drop(&mut self) {
        self.quit();
    }
}
