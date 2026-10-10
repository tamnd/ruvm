// SPDX-License-Identifier: GPL-2.0-or-later

//! The virt board on the vCPUs of a KVM VM: the Arm parts of accel/kvm/kvm-accel-ops.c,
//! system/cpus.c, hw/arm/virt.c and target/arm/kvm.c that tie a [`VirtMachine`] to a VM.
//!
//! [`prepare_config`] points the board's SPIs at KVM before the board is built.
//! [`KvmVirtMachine::new`] then registers the memory slot listener, creates and initializes one
//! vCPU per CPU (the secondaries powered off when PSCI starts them), creates the in-kernel GIC
//! with its ITS, sets up the PMUs, finishes the board and starts the vCPU threads, named
//! `CPU n/KVM`, paused. [`KvmVirtMachine::start`] lets them run.
//!
//! PSCI is handled by the kernel. Its SYSTEM_OFF and SYSTEM_RESET come back as system events:
//! a reset stops every vCPU, resets the board and the in-kernel GIC, loads each vCPU with its
//! reset state and the boot PC, and lets them go again. With `-no-reboot` it is a shutdown.
//!
//! Not done yet: EL2 for the guest, SVE and pointer authentication, which need the host's ID
//! registers, the steal time area, and migration of the GIC and ITS state.

use std::cell::Cell;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use ruvm_accel_kvm::arm::vcpu::{ArmVcpu, ArmVcpuConfig, PutLevel};
use ruvm_accel_kvm::arm::{GITS_TRANSLATER, GicV2State, GicV3State, ItsState};
use ruvm_accel_kvm::{
    KernelIrqchip, KvmAccel, KvmGicV2, KvmGicV3, KvmIts, KvmVcpu, VcpuKick, VcpuStop,
    spawn_vcpu_thread,
};
use ruvm_hw_core::{Clock, IrqLine};
use ruvm_hw_intc::gicv2::GicV2;
use ruvm_hw_intc::gicv3::{ITS_CONTROL_SIZE, ITS_TRANS_SIZE};
use ruvm_mem::{AccessCtx, AccessSize, AddressSpace, MemResult, MmioOps};

use super::{KvmSpis, boot_pc, check_config, device_irq_ppi, gicv3_config, gicv3_typers};
use crate::tcg_run::{ShutdownReason, VirtEvent, VirtEventHandler, VirtRunConfig};
use crate::virt::boot::BootInfo;
use crate::virt::{
    VIRT_GIC_CPU, VIRT_GIC_DIST, VIRT_GIC_ITS, VIRT_GIC_NUM_IRQ, VirtConfig, VirtGic, VirtMachine,
    err,
};

/// The longest the timer thread sleeps, since arming a timer does not wake it.
const TIMER_SLICE: Duration = Duration::from_millis(1);

/// `KVM_CAP_ARM_USER_IRQ`: KVM reports the timer and PMU outputs to userspace.
const KVM_CAP_ARM_USER_IRQ: u32 = 148;

thread_local! {
    /// The index of the vCPU this thread runs, for the userspace GICv2 and for
    /// [`KvmVirtMachine::pause`], which must not wait for its own thread.
    static KVM_CPU: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Points the SPIs of `cfg` at KVM when `accel` keeps the GIC in the kernel, after the
/// `machvirt_init()` checks for KVM. Pass the result to [`KvmVirtMachine::new`].
pub fn prepare_config(accel: &KvmAccel, cfg: &mut VirtConfig) -> Result<KvmSpis, String> {
    let irqchip = accel.kernel_irqchip();
    check_config(cfg, irqchip)?;
    let spis = KvmSpis::new();
    if irqchip != KernelIrqchip::Off {
        cfg.spi_sink = Some(spis.sink());
    }
    Ok(spis)
}

#[derive(Debug, Default)]
struct Ctl {
    /// The machine is started, `vm_start()`.
    running: bool,
    /// A reset was requested and not finished yet.
    reset_pending: bool,
    quit: bool,
    /// vCPUs waiting in the park loop.
    parked: usize,
    /// Bumped once the board is reset; each vCPU then loads its reset state.
    reset_gen: u64,
    /// vCPUs that loaded the state of `reset_gen`.
    applied: usize,
}

impl Ctl {
    fn must_park(&self) -> bool {
        !self.running || self.reset_pending || self.quit
    }
}

struct Shared {
    ctl: Mutex<Ctl>,
    cv: Condvar,
    kicks: OnceLock<Vec<VcpuKick>>,
    handler: VirtEventHandler,
    no_reboot: bool,
    nr_vcpus: usize,
    quit: AtomicBool,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Ctl> {
        self.ctl.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn wait<'a>(&self, g: MutexGuard<'a, Ctl>) -> MutexGuard<'a, Ctl> {
        self.cv.wait(g).unwrap_or_else(PoisonError::into_inner)
    }

    fn kick_all(&self) {
        for k in self.kicks.get().map(Vec::as_slice).unwrap_or_default() {
            let _ = k.kick();
        }
    }

    /// `qemu_system_reset_request(SHUTDOWN_CAUSE_GUEST_RESET)`.
    fn request_reset(&self) {
        if self.no_reboot {
            (self.handler)(VirtEvent::Shutdown(ShutdownReason::GuestReset));
            return;
        }
        self.lock().reset_pending = true;
        self.cv.notify_all();
        self.kick_all();
    }

    /// Stops the machine without waiting and reports `msg`, for a vCPU that cannot go on.
    fn fail(&self, msg: String) {
        self.lock().running = false;
        self.cv.notify_all();
        self.kick_all();
        (self.handler)(VirtEvent::InternalError(msg));
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The GIC the vCPUs see.
enum KernelGic {
    /// An in-kernel GICv3 and the state ruvm keeps for it, with the in-kernel ITS.
    V3(Arc<Mutex<KvmGicV3>>, GicV3State, Option<Arc<Mutex<KvmIts>>>),
    /// An in-kernel GICv2.
    V2(Arc<Mutex<KvmGicV2>>, GicV2State),
    /// The board's GICv2, with `kernel-irqchip=off`. The board resets it.
    User,
}

impl KernelGic {
    /// The reset of the in-kernel GIC and ITS, `kvm_arm_gicv3_reset_hold()` and friends.
    fn reset(&mut self, irq_reset_nonsecure: bool) -> Result<(), String> {
        match self {
            KernelGic::V3(gic, state, its) => {
                state.reset(false, irq_reset_nonsecure);
                for c in &mut state.cpus {
                    c.icc_reset();
                }
                lock(gic).reset(state).map_err(err)?;
                if let Some(its) = its {
                    if let Some(w) = lock(its).reset(&ItsState::default()).map_err(err)? {
                        eprintln!("ruvm: {w}");
                    }
                }
            }
            KernelGic::V2(gic, state) => {
                state.reset();
                lock(gic).reset(state).map_err(err)?;
            }
            KernelGic::User => {}
        }
        Ok(())
    }

    /// The vm state change handlers: the LPI pending tables and the ITS tables go into guest
    /// memory when the VM stops. QEMU reports a failure and carries on.
    fn vm_stopped(&self) {
        if let KernelGic::V3(gic, _, its) = self {
            if let Err(e) = lock(gic).vm_stopped() {
                eprintln!("ruvm: {e}");
            }
            if let Some(its) = its {
                if let Err(e) = lock(its).vm_stopped() {
                    eprintln!("ruvm: {e}");
                }
            }
        }
    }
}

/// `GITS_TRANSLATER` for writes that do not come from a vCPU, such as a PCI device's MSI: the
/// translation frame of `gicv3_its_trans_ops`, sending the write to the in-kernel ITS with the
/// requester ID as the device ID.
struct ItsDoorbell(Arc<Mutex<KvmIts>>);

impl MmioOps for ItsDoorbell {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(0)
    }

    fn write(&self, cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        if offset == GITS_TRANSLATER && matches!(size.bytes(), 2 | 4) {
            let devid = u32::from(cx.attrs.requester_id());
            let _ = lock(&self.0).send_msi(value as u32, devid);
        }
        Ok(())
    }
}

struct VcpuCtx {
    idx: usize,
    vcpu: KvmVcpu,
    arm: ArmVcpu,
    info: BootInfo,
    mem: Arc<AddressSpace>,
}

impl VcpuCtx {
    /// `kvm_arm_reset_vcpu()`, the entry from `do_cpu_reset()`, then
    /// `kvm_arch_put_registers(KVM_PUT_RESET_STATE)`.
    fn apply_reset(&mut self) -> Result<(), String> {
        self.arm.reset(&mut self.vcpu).map_err(err)?;
        if let Some(pc) = boot_pc(&self.info, self.idx) {
            self.arm.regs.pc = pc;
        }
        self.arm.put_registers(&mut self.vcpu, PutLevel::Reset).map_err(err)
    }

    /// `kvm_arm_vm_state_change()`.
    fn vm_state_change(&mut self, running: bool) -> Result<(), String> {
        self.arm.vm_state_change(&mut self.vcpu, running).map_err(err)
    }
}

fn vcpu_loop(mut v: VcpuCtx, shared: Arc<Shared>) {
    KVM_CPU.with(|c| c.set(Some(v.idx)));
    let mut my_gen = 0u64;
    loop {
        let mut failed = None;
        {
            let mut c = shared.lock();
            if c.must_park() {
                let stopped = !c.running;
                if stopped {
                    if let Err(e) = v.vm_state_change(false) {
                        failed = Some(e);
                    }
                }
                c.parked += 1;
                shared.cv.notify_all();
                loop {
                    if c.quit {
                        c.parked -= 1;
                        return;
                    }
                    if c.reset_gen != my_gen {
                        my_gen = c.reset_gen;
                        if let Err(e) = v.apply_reset() {
                            failed = Some(e);
                        }
                        c.applied += 1;
                        shared.cv.notify_all();
                        continue;
                    }
                    if !c.must_park() {
                        break;
                    }
                    c = shared.wait(c);
                }
                c.parked -= 1;
                if stopped && failed.is_none() {
                    if let Err(e) = v.vm_state_change(true) {
                        failed = Some(e);
                    }
                }
            }
        }
        if let Some(e) = failed {
            shared.fail(e);
            continue;
        }

        let stop = match v.vcpu.run(&v.mem, &v.mem) {
            Ok(stop) => stop,
            Err(e) => {
                shared.fail(e.to_string());
                continue;
            }
        };
        match stop {
            VcpuStop::Kicked | VcpuStop::IrqWindowOpen | VcpuStop::Halted => {}
            // PSCI SYSTEM_OFF and SYSTEM_RESET, which the kernel handles.
            VcpuStop::Shutdown => {
                (shared.handler)(VirtEvent::Shutdown(ShutdownReason::GuestShutdown));
            }
            VcpuStop::Reset => shared.request_reset(),
            VcpuStop::Crash => shared.fail("KVM: the guest crashed".to_string()),
            VcpuStop::InternalError => shared.fail("KVM internal error.".to_string()),
            VcpuStop::FailEntry { reason, cpu } => shared
                .fail(format!("KVM: entry failed, hardware error 0x{reason:x} on host CPU {cpu}")),
            VcpuStop::Unhandled(what) => shared.fail(format!("KVM: unhandled exit {what}")),
        }
    }
}

fn control_loop(
    shared: Arc<Shared>,
    board: Arc<Mutex<VirtMachine>>,
    gic: Arc<Mutex<KernelGic>>,
    irq_reset_nonsecure: bool,
) {
    let n = shared.nr_vcpus;
    loop {
        let mut c = shared.lock();
        while !c.reset_pending && !c.quit {
            c = shared.wait(c);
        }
        if c.quit {
            return;
        }
        drop(c);
        shared.kick_all();
        let mut c = shared.lock();
        while c.parked < n && !c.quit {
            c = shared.wait(c);
        }
        if c.quit {
            return;
        }
        drop(c);

        // qemu_system_reset(): the devices with the GIC, then every CPU.
        let res = lock(&board).system_reset().and_then(|()| lock(&gic).reset(irq_reset_nonsecure));
        let mut c = shared.lock();
        c.reset_gen += 1;
        c.applied = 0;
        shared.cv.notify_all();
        while c.applied < n && !c.quit {
            c = shared.wait(c);
        }
        c.reset_pending = false;
        shared.cv.notify_all();
        drop(c);
        match res {
            Ok(()) => (shared.handler)(VirtEvent::Reset),
            Err(e) => (shared.handler)(VirtEvent::InternalError(e)),
        }
    }
}

fn timer_loop(shared: Arc<Shared>, clocks: Vec<Arc<Clock>>) {
    while !shared.quit.load(Ordering::Acquire) {
        let mut sleep = TIMER_SLICE;
        for c in &clocks {
            c.run_timers();
            let d = c.deadline_ns();
            if d >= 0 {
                sleep = sleep.min(Duration::from_nanos(d as u64));
            }
        }
        if !sleep.is_zero() {
            std::thread::sleep(sleep);
        }
    }
}

fn spawn(name: &str, f: impl FnOnce() + Send + 'static) -> Result<JoinHandle<()>, String> {
    std::thread::Builder::new()
        .name(name.to_string())
        .spawn(f)
        .map_err(|e| format!("could not create thread: {e}"))
}

/// Wires the board's GICv2 to KVM for `kernel-irqchip=off`: its IRQ and FIQ outputs drive the
/// vCPU lines, and it learns the CPU making an access from the vCPU thread.
fn wire_user_gic(gic: &Arc<GicV2>, accel: &Arc<KvmAccel>, smp: usize) {
    gic.set_current_cpu_fn(Some(Arc::new(|| KVM_CPU.with(Cell::get))));
    for i in 0..smp {
        for (pin, fiq) in [(gic.cpu_irq(i), false), (gic.cpu_fiq(i), true)] {
            let accel = Arc::clone(accel);
            pin.connect(IrqLine::from_fn(move |level| {
                let _ = accel.set_cpu_irq(i as u32, fiq, level != 0);
            }));
        }
    }
}

/// A virt board running on KVM.
pub struct KvmVirtMachine {
    shared: Arc<Shared>,
    board: Arc<Mutex<VirtMachine>>,
    gic: Arc<Mutex<KernelGic>>,
    threads: Mutex<Vec<JoinHandle<()>>>,
    _accel: Arc<KvmAccel>,
}

impl fmt::Debug for KvmVirtMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KvmVirtMachine")
            .field("vcpus", &self.shared.nr_vcpus)
            .finish_non_exhaustive()
    }
}

impl KvmVirtMachine {
    /// Puts `board`, built from a config that went through [`prepare_config`], on the VM of
    /// `accel`. `spis` is what [`prepare_config`] gave. `clocks` are the clocks the board's
    /// timers run on; a thread fires them. Only `cfg.no_reboot` is looked at.
    pub fn new(
        accel: KvmAccel,
        board: VirtMachine,
        spis: KvmSpis,
        clocks: Vec<Arc<Clock>>,
        cfg: &VirtRunConfig,
        handler: VirtEventHandler,
    ) -> Result<KvmVirtMachine, String> {
        let accel = Arc::new(accel);
        let mut board = board;
        let mode = accel.kernel_irqchip();
        let smp = board.smp;
        let in_kernel = mode != KernelIrqchip::Off;
        if !in_kernel && accel.kvm().check_extension_raw(u64::from(KVM_CAP_ARM_USER_IRQ)) <= 0 {
            return Err("KVM with user space irqchip only works when the host kernel supports \
                        KVM_CAP_ARM_USER_IRQ"
                .to_string());
        }
        board
            .memory_system()
            .register_listener(accel.slot_listener(), board.memory_as())
            .map_err(err)?;

        // kvm_arch_init_vcpu() for each CPU, before the GIC as in QEMU.
        let target = accel.arm_preferred_target().map_err(err)?;
        let caps = accel.arm_caps();
        let mut vcpus = Vec::with_capacity(smp);
        for i in 0..smp {
            let mut vcpu = accel.create_vcpu(i as u32).map_err(err)?;
            let mut vcfg = ArmVcpuConfig::host(target, &caps);
            vcfg.start_powered_off = i != 0 && board.secondaries_off;
            vcfg.pmu = caps.pmu_v3 && board.model.features.pmu != 0;
            let arm = ArmVcpu::init(&mut vcpu, &vcfg, &caps)
                .map_err(|e| format!("kvm_init_vcpu: kvm_arch_init_vcpu failed ({i}): {e}"))?;
            vcpus.push((vcpu, arm));
        }

        // create_gic() and create_its() for the in-kernel GIC.
        let gic = match (&board.gic, in_kernel) {
            (VirtGic::V3(_), true) => {
                let gcfg = gicv3_config(smp, board.redist2.map(|r| r.base));
                let mut state = GicV3State::new(VIRT_GIC_NUM_IRQ, &gicv3_typers(smp));
                let dev =
                    Arc::new(Mutex::new(KvmGicV3::new(&accel, &gcfg, &mut state).map_err(err)?));
                let d = Arc::clone(&dev);
                spis.connect(Arc::new(move |n, level| {
                    let _ = lock(&d).set_irq(n, level);
                }));
                let its = match board.its {
                    Some(_) => {
                        let its =
                            Arc::new(Mutex::new(KvmIts::new(&accel, VIRT_GIC_ITS).map_err(err)?));
                        let r = board
                            .mem
                            .new_io(
                                "kvm-its-translation",
                                ITS_TRANS_SIZE.into(),
                                Arc::new(ItsDoorbell(Arc::clone(&its))),
                            )
                            .map_err(err)?;
                        board
                            .mem
                            .add_subregion_overlap(
                                board.system,
                                VIRT_GIC_ITS + ITS_CONTROL_SIZE,
                                r,
                                1,
                            )
                            .map_err(err)?;
                        Some(its)
                    }
                    None => None,
                };
                KernelGic::V3(dev, state, its)
            }
            (VirtGic::V2(_), true) => {
                let dev = KvmGicV2::new(
                    &accel,
                    VIRT_GIC_NUM_IRQ,
                    VIRT_GIC_DIST,
                    VIRT_GIC_CPU,
                    false,
                    false,
                )
                .map_err(err)?;
                let dev = Arc::new(Mutex::new(dev));
                let d = Arc::clone(&dev);
                spis.connect(Arc::new(move |n, level| {
                    let _ = lock(&d).set_irq(n, level);
                }));
                KernelGic::V2(dev, GicV2State::new(VIRT_GIC_NUM_IRQ, smp as u32))
            }
            (VirtGic::V2(g), false) => {
                wire_user_gic(g, &accel, smp);
                for (i, (vcpu, _)) in vcpus.iter_mut().enumerate() {
                    let g = Arc::clone(g);
                    vcpu.set_device_irq_hook(move |changes| {
                        for &(irq, level) in &changes.changed {
                            g.ppi(i, device_irq_ppi(irq)).set(i32::from(level));
                        }
                    });
                }
                KernelGic::User
            }
            (VirtGic::V3(_), false) => {
                return Err(
                    "KVM with kernel-irqchip=off does not support GICv3 emulation".to_string()
                );
            }
        };

        // virt_cpu_post_init(): the PMU, which needs the GIC.
        for (vcpu, arm) in &mut vcpus {
            if in_kernel {
                arm.pmu_set_irq(vcpu, super::PMU_PPI).map_err(err)?;
            }
            arm.pmu_init(vcpu).map_err(err)?;
        }

        board.machine_done()?;
        let irq_reset_nonsecure = board.info.is_linux;
        let mut gic = gic;
        gic.reset(irq_reset_nonsecure)?;
        let info = board.info.clone();
        let mem = Arc::clone(board.memory_as());
        let board = Arc::new(Mutex::new(board));
        let gic = Arc::new(Mutex::new(gic));

        let shared = Arc::new(Shared {
            ctl: Mutex::new(Ctl::default()),
            cv: Condvar::new(),
            kicks: OnceLock::new(),
            handler,
            no_reboot: cfg.no_reboot,
            nr_vcpus: smp,
            quit: AtomicBool::new(false),
        });
        let mut threads = Vec::with_capacity(smp + 2);
        let mut kicks = Vec::with_capacity(smp);
        for (idx, (vcpu, arm)) in vcpus.into_iter().enumerate() {
            let mut v = VcpuCtx { idx, vcpu, arm, info: info.clone(), mem: Arc::clone(&mem) };
            v.apply_reset()?;
            let exit = v.vcpu.exit_request();
            let s = Arc::clone(&shared);
            let t = spawn_vcpu_thread(idx as u32, move || vcpu_loop(v, s))
                .map_err(|e| format!("could not create vCPU thread: {e}"))?;
            kicks.push(VcpuKick::new(&t, exit));
            threads.push(t);
        }
        let _ = shared.kicks.set(kicks);
        let control = {
            let (s, b, g) = (Arc::clone(&shared), Arc::clone(&board), Arc::clone(&gic));
            spawn("reset", move || control_loop(s, b, g, irq_reset_nonsecure))?
        };
        let timers = {
            let s = Arc::clone(&shared);
            spawn("timers", move || timer_loop(s, clocks))?
        };
        threads.push(control);
        threads.push(timers);
        Ok(KvmVirtMachine { shared, board, gic, threads: Mutex::new(threads), _accel: accel })
    }

    /// The board, for the monitor and for device access.
    pub fn board(&self) -> &Arc<Mutex<VirtMachine>> {
        &self.board
    }

    /// The number of vCPUs.
    pub fn vcpu_count(&self) -> usize {
        self.shared.nr_vcpus
    }

    /// `resume_all_vcpus()`.
    pub fn start(&self) {
        self.shared.lock().running = true;
        self.shared.cv.notify_all();
    }

    /// `pause_all_vcpus()`, then the GIC's vm state change handler. Waits until every vCPU
    /// is out of the guest, unless called from a vCPU thread, which only asks.
    pub fn pause(&self) {
        self.shared.lock().running = false;
        self.shared.cv.notify_all();
        self.shared.kick_all();
        if KVM_CPU.with(Cell::get).is_some() {
            return;
        }
        let mut c = self.shared.lock();
        while c.parked < self.shared.nr_vcpus && !c.quit {
            c = self.shared.wait(c);
        }
        drop(c);
        lock(&self.gic).vm_stopped();
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
        self.shared.kick_all();
        let threads = std::mem::take(&mut *lock(&self.threads));
        for t in threads {
            let _ = t.join();
        }
    }
}

impl Drop for KvmVirtMachine {
    fn drop(&mut self) {
        self.quit();
    }
}
