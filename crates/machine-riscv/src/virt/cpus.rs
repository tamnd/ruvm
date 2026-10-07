// SPDX-License-Identifier: GPL-2.0-or-later

//! How the virt board meets its harts: the ACLINT `mtime` the `time` CSR reads and the Sstc
//! timers behind `stimecmp` ([`CpuHub`], the board side of `RiscvBoard`), the shutdown and
//! reset requests of the SiFive test device, and semihosting ([`VirtSemihost`]).

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, Weak};
use std::time::Instant;

use ruvm_hw_core::{Clock, Timer};
use ruvm_hw_intc::riscv_aclint::RiscvAclintMtimer;
use ruvm_jit::CpuShared;
use ruvm_target_riscv::tcg::{Riscv, RiscvBoard, SemihostingHost};

use super::{VirtRequest, VirtRequestHandler};

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One hart as the board sees it.
struct CpuSlot {
    /// The vCPU, once [`super::VirtMachine::create_vcpus`] has made it.
    shared: Weak<CpuShared>,
    /// `env->stimer`: the timer that raises STIP at the `stimecmp` deadline.
    stimer: Option<Timer>,
}

/// The board side of every hart: `rdtime_fn` on the ACLINT timer, the Sstc timers on the
/// board clock, and the requests of the SiFive test device.
pub(crate) struct CpuHub {
    mtimer: Arc<RiscvAclintMtimer>,
    clock: Arc<Clock>,
    riscv: OnceLock<Weak<Riscv>>,
    slots: Mutex<Vec<CpuSlot>>,
    request: Mutex<Option<VirtRequest>>,
    /// Where requests go instead of [`CpuHub::take_request`], once a run loop has set it.
    handler: Mutex<Option<VirtRequestHandler>>,
}

impl fmt::Debug for CpuHub {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuHub")
            .field("harts", &lock(&self.slots).len())
            .field("request", &*lock(&self.request))
            .finish_non_exhaustive()
    }
}

impl CpuHub {
    pub(crate) fn new(mtimer: Arc<RiscvAclintMtimer>, harts: usize, clock: Arc<Clock>) -> CpuHub {
        let slots = (0..harts).map(|_| CpuSlot { shared: Weak::new(), stimer: None }).collect();
        CpuHub {
            mtimer,
            clock,
            riscv: OnceLock::new(),
            slots: Mutex::new(slots),
            request: Mutex::new(None),
            handler: Mutex::new(None),
        }
    }

    pub(crate) fn set_riscv(&self, riscv: &Arc<Riscv>) {
        let _ = self.riscv.set(Arc::downgrade(riscv));
    }

    /// Record the vCPU with index `cpu`.
    pub(crate) fn register(&self, cpu: usize, shared: &Arc<CpuShared>) {
        if let Some(s) = lock(&self.slots).get_mut(cpu) {
            s.shared = Arc::downgrade(shared);
        }
    }

    /// Stop the Sstc timer of hart `cpu`, as a CPU reset leaves `env->stimer` idle.
    pub(crate) fn reset_timer(&self, cpu: usize) {
        if let Some(t) = lock(&self.slots).get(cpu).and_then(|s| s.stimer.as_ref()) {
            t.del();
        }
    }

    /// Take the pending shutdown or reset request.
    pub(crate) fn take_request(&self) -> Option<VirtRequest> {
        lock(&self.request).take()
    }

    /// Send the requests to `handler` from now on, or keep them for
    /// [`CpuHub::take_request`] again with `None`.
    pub(crate) fn set_request_handler(&self, handler: Option<VirtRequestHandler>) {
        *lock(&self.handler) = handler;
    }

    /// `qemu_system_shutdown_request_with_code()` and `qemu_system_reset_request()`: hand
    /// `req` to the handler if there is one, otherwise record it and kick every vCPU out of
    /// its loop.
    pub(crate) fn request(&self, req: VirtRequest) {
        let handler = lock(&self.handler).clone();
        if let Some(h) = handler {
            h(req);
            return;
        }
        *lock(&self.request) = Some(req);
        let all: Vec<_> = lock(&self.slots).iter().filter_map(|s| s.shared.upgrade()).collect();
        for s in all {
            s.kick();
        }
    }
}

impl RiscvBoard for CpuHub {
    fn rdtime(&self) -> Option<u64> {
        Some(self.mtimer.time())
    }

    fn timebase_freq(&self) -> u64 {
        u64::from(self.mtimer.timebase_freq())
    }

    fn stimer_update(&self, shared: &CpuShared, deadline: Option<Instant>) {
        let mut slots = lock(&self.slots);
        let Some(slot) = slots.get_mut(shared.cpu_index) else {
            return;
        };
        match deadline {
            Some(d) => {
                let wait = d.saturating_duration_since(Instant::now());
                let ns = i64::try_from(wait.as_nanos()).unwrap_or(i64::MAX);
                let when = self.clock.get_ns().saturating_add(ns);
                let weak_cpu = slot.shared.clone();
                let weak_riscv = self.riscv.get().cloned().unwrap_or_default();
                let t = slot.stimer.get_or_insert_with(|| {
                    self.clock.new_timer(move || {
                        if let (Some(s), Some(r)) = (weak_cpu.upgrade(), weak_riscv.upgrade()) {
                            r.stimer_expired(&s);
                        }
                    })
                });
                t.modify(when);
            }
            None => {
                if let Some(t) = &slot.stimer {
                    t.del();
                }
            }
        }
    }
}

/// The board's semihosting: the console and exit go to the host given in the config, the
/// heap comes from the ROM layout at machine_done and the command line falls back to the
/// kernel name and `-append`, as `semihosting_arg_fallback()`.
pub(crate) struct VirtSemihost {
    pub(crate) host: Arc<dyn SemihostingHost>,
    pub(crate) heap: Arc<Mutex<(u64, u64)>>,
    pub(crate) cmdline: String,
}

impl fmt::Debug for VirtSemihost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VirtSemihost")
            .field("heap", &*lock(&self.heap))
            .field("cmdline", &self.cmdline)
            .finish_non_exhaustive()
    }
}

impl SemihostingHost for VirtSemihost {
    fn console_write(&self, buf: &[u8]) -> usize {
        self.host.console_write(buf)
    }

    fn console_read(&self) -> u8 {
        self.host.console_read()
    }

    fn exit(&self, code: u32) {
        self.host.exit(code);
    }

    fn cmdline(&self) -> Option<String> {
        self.host.cmdline().or_else(|| Some(self.cmdline.clone()))
    }

    fn heap_info(&self) -> (u64, u64) {
        *lock(&self.heap)
    }

    fn stdio_write(&self, fd: u32, buf: &[u8]) -> std::io::Result<usize> {
        self.host.stdio_write(fd, buf)
    }

    fn stdio_read(&self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.host.stdio_read(buf)
    }

    fn system(&self, cmd: &str) -> Option<i64> {
        self.host.system(cmd)
    }

    fn unsupported(&self, nr: u32) {
        self.host.unsupported(nr);
    }
}

/// `semihosting_arg_fallback()`: the kernel file name, then the `-append` words, joined
/// with spaces.
pub(crate) fn semihosting_cmdline(kernel: Option<&str>, append: Option<&str>) -> String {
    let Some(k) = kernel else {
        return String::new();
    };
    let mut args = vec![k.to_string()];
    args.extend(append.unwrap_or("").split(' ').filter(|w| !w.is_empty()).map(str::to_string));
    args.join(" ")
}
