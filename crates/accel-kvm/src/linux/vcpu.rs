// SPDX-License-Identifier: GPL-2.0-or-later

//! The vCPU run loop and kicks, `kvm_cpu_exec()` and `kvm_cpu_kick()`.
//!
//! A kick sets the vCPU's exit request and sends SIGUSR1 to its thread. The signal handler
//! stores 1 into `kvm_run.immediate_exit`, so a signal that lands just before `KVM_RUN` still
//! makes the kernel return at once, and one that lands inside `KVM_RUN` interrupts it with
//! `EINTR`. That is the `KVM_CAP_IMMEDIATE_EXIT` scheme; ruvm has no signal mask fallback.

use std::cell::Cell;
use std::io;
use std::os::unix::thread::JoinHandleExt;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Arc, Once};
use std::thread::JoinHandle;

use kvm_bindings::{
    KVM_SYSTEM_EVENT_CRASH, KVM_SYSTEM_EVENT_RESET, KVM_SYSTEM_EVENT_SHUTDOWN, kvm_run,
};
use kvm_ioctls::{VcpuExit, VcpuFd};
use ruvm_mem::{AddressSpace, MemTxAttrs};
use vmm_sys_util::signal::{Killable, register_signal_handler};

use super::os_error;
use crate::{KvmError, vcpu_thread_name};

/// `SIG_IPI`, which is SIGUSR1 in QEMU's include/qemu/osdep.h.
const SIG_IPI: libc::c_int = libc::SIGUSR1;

thread_local! {
    /// The running vCPU's `immediate_exit` byte, set only while [`KvmVcpu::run`] is active.
    static IMMEDIATE_EXIT: Cell<Option<&'static AtomicU8>> = const { Cell::new(None) };
}

/// `kvm_ipi_signal()`. Async signal safe: a TLS read of a const initialized cell and a store.
extern "C" fn ipi_signal(_num: libc::c_int, _info: *mut libc::siginfo_t, _ctx: *mut libc::c_void) {
    let _ = IMMEDIATE_EXIT.try_with(|cell| {
        if let Some(flag) = cell.get() {
            flag.store(1, Ordering::Relaxed);
        }
    });
}

fn install_ipi_handler() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        register_signal_handler(SIG_IPI, ipi_signal).expect("SIGUSR1 is a valid signal");
    });
}

/// Publishes the `immediate_exit` byte to the signal handler for as long as it lives.
struct ImmediateExitBinding;

impl ImmediateExitBinding {
    fn bind(run: &mut kvm_run) -> Self {
        let ptr: *mut u8 = &mut run.immediate_exit;
        // SAFETY: the byte lives in the vCPU's kvm_run mapping, which stays mapped for as long as
        // the VcpuFd does. The binding is dropped before `run` returns, while the caller still
        // holds `&mut KvmVcpu`, so the 'static reference never outlives the mapping. The kernel
        // only reads the byte and every userspace access goes through this atomic or through
        // `set_kvm_immediate_exit` on the same thread, never at the same time.
        let flag: &'static AtomicU8 = unsafe { AtomicU8::from_ptr(ptr) };
        IMMEDIATE_EXIT.with(|cell| cell.set(Some(flag)));
        ImmediateExitBinding
    }
}

impl Drop for ImmediateExitBinding {
    fn drop(&mut self) {
        IMMEDIATE_EXIT.with(|cell| cell.set(None));
    }
}

/// Why [`KvmVcpu::run`] came back to its caller.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VcpuStop {
    /// The guest ran `hlt` with no in-kernel LAPIC to wait for interrupts.
    Halted,
    /// A kick or a signal interrupted the run, `EXCP_INTERRUPT`.
    Kicked,
    /// `KVM_EXIT_IRQ_WINDOW_OPEN`: the guest can take an interrupt now.
    IrqWindowOpen,
    /// A triple fault or a guest shutdown request.
    Shutdown,
    /// `KVM_SYSTEM_EVENT_RESET`.
    Reset,
    /// `KVM_SYSTEM_EVENT_CRASH`.
    Crash,
    /// `KVM_EXIT_INTERNAL_ERROR`.
    InternalError,
    /// `KVM_EXIT_FAIL_ENTRY` with the hardware reason and the host CPU.
    FailEntry { reason: u64, cpu: u32 },
    /// Any exit this slice does not handle yet, by name.
    Unhandled(String),
}

enum Step {
    Continue,
    Io,
    Stop(VcpuStop),
}

/// One KVM vCPU, `CPUState` with its `kvm_fd` and `kvm_run`.
#[derive(Debug)]
pub struct KvmVcpu {
    fd: VcpuFd,
    index: u32,
    exit_request: Arc<AtomicBool>,
}

impl KvmVcpu {
    pub(crate) fn new(fd: VcpuFd, index: u32) -> Self {
        KvmVcpu { fd, index, exit_request: Arc::new(AtomicBool::new(false)) }
    }

    /// The vCPU index, `cpu_index`.
    pub fn index(&self) -> u32 {
        self.index
    }

    /// The vCPU file descriptor, for register access from the CPU code.
    pub fn fd(&self) -> &VcpuFd {
        &self.fd
    }

    /// The vCPU file descriptor, mutably.
    pub fn fd_mut(&mut self) -> &mut VcpuFd {
        &mut self.fd
    }

    /// The exit request flag, `cpu->exit_request`, shared with the [`VcpuKick`] for this vCPU.
    pub fn exit_request(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.exit_request)
    }

    /// Runs the guest until something needs the caller, `kvm_cpu_exec()`. Port I/O goes to
    /// `io` and MMIO to `mem`, both with unspecified attributes, and neither returns here.
    pub fn run(&mut self, io: &AddressSpace, mem: &AddressSpace) -> Result<VcpuStop, KvmError> {
        install_ipi_handler();
        let _binding = ImmediateExitBinding::bind(self.fd.get_kvm_run());
        let attrs = MemTxAttrs::UNSPECIFIED;
        loop {
            if self.exit_request.swap(false, Ordering::AcqRel) {
                // `kvm_cpu_kick_self()`: the next KVM_RUN finishes any pending I/O and returns
                // EINTR without running guest code.
                self.fd.set_kvm_immediate_exit(1);
            }
            let step = match self.fd.run() {
                Ok(VcpuExit::IoIn(..) | VcpuExit::IoOut(..)) => Step::Io,
                Ok(VcpuExit::MmioRead(addr, data)) => {
                    let _ = mem.read(addr, attrs, data);
                    Step::Continue
                }
                Ok(VcpuExit::MmioWrite(addr, data)) => {
                    let _ = mem.write(addr, attrs, data);
                    Step::Continue
                }
                Ok(VcpuExit::Hlt) => Step::Stop(VcpuStop::Halted),
                Ok(VcpuExit::Intr) => Step::Stop(VcpuStop::Kicked),
                Ok(VcpuExit::IrqWindowOpen) => Step::Stop(VcpuStop::IrqWindowOpen),
                Ok(VcpuExit::Shutdown) => Step::Stop(VcpuStop::Shutdown),
                Ok(VcpuExit::SystemEvent(kind, _)) => Step::Stop(match kind {
                    KVM_SYSTEM_EVENT_SHUTDOWN => VcpuStop::Shutdown,
                    KVM_SYSTEM_EVENT_RESET => VcpuStop::Reset,
                    KVM_SYSTEM_EVENT_CRASH => VcpuStop::Crash,
                    other => VcpuStop::Unhandled(format!("system event {other}")),
                }),
                Ok(VcpuExit::InternalError) => Step::Stop(VcpuStop::InternalError),
                Ok(VcpuExit::FailEntry(reason, cpu)) => {
                    Step::Stop(VcpuStop::FailEntry { reason, cpu })
                }
                Ok(other) => Step::Stop(VcpuStop::Unhandled(format!("{other:?}"))),
                Err(e) if e.errno() == libc::EINTR || e.errno() == libc::EAGAIN => {
                    Step::Stop(VcpuStop::Kicked)
                }
                Err(e) => {
                    self.fd.set_kvm_immediate_exit(0);
                    return Err(KvmError::Run(os_error(e)));
                }
            };
            self.fd.set_kvm_immediate_exit(0);
            match step {
                Step::Continue => {}
                Step::Io => self.handle_io(io, attrs),
                Step::Stop(stop) => return Ok(stop),
            }
        }
    }

    /// `kvm_handle_io()`: string I/O arrives as `count` accesses of `size` bytes each, so it is
    /// replayed one access at a time in order, as QEMU does.
    fn handle_io(&mut self, space: &AddressSpace, attrs: MemTxAttrs) {
        let run = self.fd.get_kvm_run();
        let base: *mut u8 = (run as *mut kvm_run).cast();
        // SAFETY: this runs right after KVM_RUN returned KVM_EXIT_IO, so `io` is the live union
        // member. `data_offset` is set by the kernel to a place inside the vCPU mmap area, which
        // kvm-ioctls maps whole, and `size * count` bytes from there belong to this exit. Nothing
        // else touches the mapping until the next KVM_RUN, which needs `&mut self`.
        let (io, data) = unsafe {
            let io = run.__bindgen_anon_1.io;
            let len = usize::from(io.size) * io.count as usize;
            (io, std::slice::from_raw_parts_mut(base.add(io.data_offset as usize), len))
        };
        let size = usize::from(io.size).max(1);
        let port = u64::from(io.port);
        let out = u32::from(io.direction) == kvm_bindings::KVM_EXIT_IO_OUT;
        for chunk in data.chunks_mut(size) {
            let _ =
                if out { space.write(port, attrs, chunk) } else { space.read(port, attrs, chunk) };
        }
    }
}

/// Kicks one vCPU thread out of `KVM_RUN`, `kvm_cpu_kick()` and `qemu_cpu_kick()`.
#[derive(Clone, Debug)]
pub struct VcpuKick {
    thread: libc::pthread_t,
    exit_request: Arc<AtomicBool>,
}

// SAFETY: the handle comes from `JoinHandle::as_pthread_t` on the vCPU thread. Callers kick only
// while that thread runs, the same rule QEMU follows by joining vCPU threads after the last kick.
unsafe impl Killable for VcpuKick {
    fn pthread_handle(&self) -> libc::pthread_t {
        self.thread
    }
}

impl VcpuKick {
    /// A kicker for the vCPU whose exit request flag is `exit_request` and that runs on
    /// `thread`.
    pub fn new<T>(thread: &JoinHandle<T>, exit_request: Arc<AtomicBool>) -> Self {
        VcpuKick { thread: thread.as_pthread_t(), exit_request }
    }

    /// Asks the vCPU to come back from [`KvmVcpu::run`] with [`VcpuStop::Kicked`].
    pub fn kick(&self) -> io::Result<()> {
        self.exit_request.store(true, Ordering::Release);
        self.kill(SIG_IPI).map_err(|e| io::Error::from_raw_os_error(e.errno()))
    }
}

/// Starts the thread for vCPU `index` with QEMU's name for it.
pub fn spawn_vcpu_thread<F, T>(index: u32, f: F) -> io::Result<JoinHandle<T>>
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    install_ipi_handler();
    std::thread::Builder::new().name(vcpu_thread_name(index)).spawn(f)
}
