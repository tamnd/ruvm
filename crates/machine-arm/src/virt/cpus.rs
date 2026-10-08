// SPDX-License-Identifier: GPL-2.0-or-later

//! How the virt board meets its CPUs: the generic timer outputs and PSCI calls
//! ([`CpuHub`], the board side of `ArmBoard`), the GICv3 CPU interface registers
//! ([`GicCpuIf`]) and semihosting ([`VirtSemihost`]).

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, Weak};
use std::time::Instant;

use ruvm_hw_core::{Clock, IrqLine, Timer};
use ruvm_hw_intc::gicv3::{GicV3, IccAccess, IccCpuCtx, IccReg};
use ruvm_jit::CpuShared;
use ruvm_target_arm::tcg::{
    Arm, ArmBoard, GicAccess, GicCpuInterface, GicCpuState, IccEncoding, PSCI_RET_INVALID_PARAMS,
    SemihostingHost,
};

use super::{VirtRequest, VirtRequestHandler};

/// The number of generic timers, `NUM_GTIMERS`.
pub(crate) const NUM_GTIMERS: usize = 5;

/// The PPI (as an INTID) each generic timer output drives on the virt board, by `GTIMER_*`
/// index: `ARCH_TIMER_NS_EL1_IRQ`, `ARCH_TIMER_VIRT_IRQ`, `ARCH_TIMER_NS_EL2_IRQ`,
/// `ARCH_TIMER_S_EL1_IRQ` and `ARCH_TIMER_NS_EL2_VIRT_IRQ`, each plus 16.
pub(crate) const TIMER_PPIS: [u32; NUM_GTIMERS] = [30, 27, 26, 29, 28];

/// The PPI (as an INTID) the PMU interrupt drives, `VIRTUAL_PMU_IRQ` plus 16, on virt and
/// sbsa-ref alike.
pub(crate) const PMU_PPI: u32 = 23;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One vCPU as the board sees it.
struct CpuSlot {
    /// The vCPU, once [`super::VirtMachine::create_vcpus`] has made it.
    shared: Weak<CpuShared>,
    /// `gt_timer[]`: the timers that call back into the vCPU at the next deadline.
    timers: [Option<Timer>; NUM_GTIMERS],
    /// `pmu_timer`.
    pmu_timer: Option<Timer>,
}

/// The board side of every vCPU: `gt_timer_outputs[]` wired to the GIC PPIs, the timers
/// behind `gt_timer[]` on the board clock, and PSCI.
pub(crate) struct CpuHub {
    mpidrs: Vec<u64>,
    /// `gt_timer_outputs[]` by vCPU, connected to the redistributor PPIs.
    ppis: Vec<[IrqLine; NUM_GTIMERS]>,
    /// `pmu_interrupt` by vCPU, connected to PPI [`PMU_PPI`].
    pmu_ppis: Vec<IrqLine>,
    clock: Arc<Clock>,
    arm: OnceLock<Weak<Arm>>,
    slots: Mutex<Vec<CpuSlot>>,
    request: Mutex<Option<VirtRequest>>,
    /// Where requests go instead of [`CpuHub::request`], once a run loop has set it.
    handler: Mutex<Option<VirtRequestHandler>>,
}

impl fmt::Debug for CpuHub {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CpuHub")
            .field("mpidrs", &self.mpidrs)
            .field("request", &*lock(&self.request))
            .finish_non_exhaustive()
    }
}

impl CpuHub {
    pub(crate) fn new(gic: &Arc<GicV3>, mpidrs: Vec<u64>, clock: Arc<Clock>) -> CpuHub {
        let ppis = (0..mpidrs.len()).map(|cpu| TIMER_PPIS.map(|ppi| gic.ppi(cpu, ppi))).collect();
        let pmu_ppis = (0..mpidrs.len()).map(|cpu| gic.ppi(cpu, PMU_PPI)).collect();
        let slots = (0..mpidrs.len())
            .map(|_| CpuSlot { shared: Weak::new(), timers: Default::default(), pmu_timer: None })
            .collect();
        CpuHub {
            mpidrs,
            ppis,
            pmu_ppis,
            clock,
            arm: OnceLock::new(),
            slots: Mutex::new(slots),
            request: Mutex::new(None),
            handler: Mutex::new(None),
        }
    }

    pub(crate) fn set_arm(&self, arm: &Arc<Arm>) {
        let _ = self.arm.set(Arc::downgrade(arm));
    }

    fn arm(&self) -> Option<Arc<Arm>> {
        self.arm.get().and_then(Weak::upgrade)
    }

    /// Record the vCPU with index `cpu`.
    pub(crate) fn register(&self, cpu: usize, shared: &Arc<CpuShared>) {
        if let Some(s) = lock(&self.slots).get_mut(cpu) {
            s.shared = Arc::downgrade(shared);
        }
    }

    /// The vCPU with index `cpu`.
    pub(crate) fn shared(&self, cpu: usize) -> Option<Arc<CpuShared>> {
        lock(&self.slots).get(cpu).and_then(|s| s.shared.upgrade())
    }

    /// `arm_get_cpu_by_id()`: the index of the vCPU whose MPIDR is `mpidr`.
    fn index_of(&self, mpidr: u64) -> Option<usize> {
        self.mpidrs.iter().position(|&m| m == mpidr)
    }

    /// `gt_timer_reset()` for the timers of vCPU `cpu`, and the PMU timer: stop them.
    pub(crate) fn reset_timers(&self, cpu: usize) {
        if let Some(s) = lock(&self.slots).get_mut(cpu) {
            for t in s.timers.iter().chain(std::iter::once(&s.pmu_timer)).flatten() {
                t.del();
            }
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

    /// `qemu_system_*_request()`: hand `req` to the handler if there is one, otherwise record
    /// it and kick every vCPU out of its loop.
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

impl ArmBoard for CpuHub {
    fn gt_timer_update(
        &self,
        shared: &CpuShared,
        timer: usize,
        level: bool,
        deadline: Option<Instant>,
    ) {
        let cpu = shared.cpu_index;
        if let Some(line) = self.ppis.get(cpu).and_then(|l| l.get(timer)) {
            line.set_bool(level);
        }
        let mut slots = lock(&self.slots);
        let Some(slot) = slots.get_mut(cpu) else {
            return;
        };
        let Some(t) = slot.timers.get_mut(timer) else {
            return;
        };
        match deadline {
            Some(d) => {
                let wait = d.saturating_duration_since(Instant::now());
                let ns = i64::try_from(wait.as_nanos()).unwrap_or(i64::MAX);
                let when = self.clock.get_ns().saturating_add(ns);
                let t = t.get_or_insert_with(|| {
                    let weak = slot.shared.clone();
                    self.clock.new_timer(move || {
                        if let Some(s) = weak.upgrade() {
                            Arm::gt_timer_expired(&s, timer);
                        }
                    })
                });
                t.modify(when);
            }
            None => {
                if let Some(t) = t {
                    t.del();
                }
            }
        }
    }

    fn gt_timer_set_level(&self, shared: &CpuShared, timer: usize, level: bool) {
        if let Some(line) = self.ppis.get(shared.cpu_index).and_then(|l| l.get(timer)) {
            line.set_bool(level);
        }
    }

    fn pmu_set_level(&self, shared: &CpuShared, level: bool) {
        if let Some(line) = self.pmu_ppis.get(shared.cpu_index) {
            line.set_bool(level);
        }
    }

    fn pmu_timer_anticipate(&self, shared: &CpuShared, deadline: Instant) {
        let mut slots = lock(&self.slots);
        let Some(slot) = slots.get_mut(shared.cpu_index) else {
            return;
        };
        let wait = deadline.saturating_duration_since(Instant::now());
        let ns = i64::try_from(wait.as_nanos()).unwrap_or(i64::MAX);
        let when = self.clock.get_ns().saturating_add(ns);
        let weak = slot.shared.clone();
        let t = slot.pmu_timer.get_or_insert_with(|| {
            self.clock.new_timer(move || {
                if let Some(s) = weak.upgrade() {
                    Arm::pmu_timer_expired(&s);
                }
            })
        });
        t.modify_anticipate(when);
    }

    fn psci_cpu_on(&self, mpidr: u64, entry: u64, context_id: u64, target_el: u32) -> i64 {
        let (Some(i), Some(arm)) = (self.index_of(mpidr), self.arm()) else {
            return PSCI_RET_INVALID_PARAMS;
        };
        match self.shared(i) {
            Some(s) => arm.cpu_on(&s, entry, context_id, target_el),
            None => PSCI_RET_INVALID_PARAMS,
        }
    }

    fn psci_power_state(&self, mpidr: u64) -> Option<u32> {
        let i = self.index_of(mpidr)?;
        Some(self.arm()?.power_state(i))
    }

    fn psci_system_off(&self) {
        self.request(VirtRequest::Shutdown);
    }

    fn psci_system_reset(&self) {
        self.request(VirtRequest::Reset);
    }
}

/// The GICv3 CPU interface registers of the virt board's GIC.
#[derive(Debug)]
pub(crate) struct GicCpuIf(pub(crate) Arc<GicV3>);

fn ctx(s: &GicCpuState) -> IccCpuCtx {
    IccCpuCtx {
        el: s.el,
        has_el2: s.has_el2,
        has_el3: s.has_el3,
        secure: s.secure,
        secure_below_el3: s.secure_below_el3,
        hcr_el2: s.hcr_el2,
        scr_el3: s.scr_el3,
    }
}

fn icc_reg(reg: IccEncoding) -> Option<IccReg> {
    IccReg::from_encoding(reg.0, reg.1, reg.2, reg.3, reg.4)
}

impl GicCpuInterface for GicCpuIf {
    fn access(&self, cpu: usize, reg: IccEncoding, state: &GicCpuState, isread: bool) -> GicAccess {
        let Some(r) = icc_reg(reg) else {
            return GicAccess::Undefined;
        };
        match self.0.icc_access(cpu, r, &ctx(state), isread) {
            IccAccess::Ok => GicAccess::Ok,
            IccAccess::TrapEl1 => GicAccess::TrapEl1,
            IccAccess::TrapEl2 => GicAccess::TrapEl2,
            IccAccess::TrapEl3 => GicAccess::TrapEl3,
            IccAccess::Undefined => GicAccess::Undefined,
        }
    }

    fn read(&self, cpu: usize, reg: IccEncoding, state: &GicCpuState) -> u64 {
        icc_reg(reg).map_or(0, |r| self.0.icc_read(cpu, r, &ctx(state)))
    }

    fn write(&self, cpu: usize, reg: IccEncoding, state: &GicCpuState, value: u64) {
        if let Some(r) = icc_reg(reg) {
            self.0.icc_write(cpu, r, &ctx(state), value);
        }
    }

    fn state_changed(&self, cpu: usize, state: &GicCpuState) {
        self.0.cpu_state_changed(cpu, &ctx(state));
    }

    fn reset(&self, cpu: usize) {
        self.0.cpuif_reset(cpu);
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
