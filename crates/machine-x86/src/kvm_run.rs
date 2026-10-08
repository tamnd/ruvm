// SPDX-License-Identifier: GPL-2.0-or-later

//! Running an x86 board on KVM: the parts of accel/kvm/kvm-accel-ops.c, system/cpus.c,
//! hw/i386/kvm/ and target/i386/kvm/kvm.c that tie a [`X86Board`] to a VM.
//!
//! [`KvmMachine::new`] takes an opened [`KvmAccel`] and a board with its devices plugged. It
//! registers the memory slot listener, wires the interrupt lines for the `kernel-irqchip` mode,
//! creates one vCPU per present APIC ID and finishes the board with `machine_done()`. The vCPU
//! threads, named `CPU n/KVM`, start paused; [`KvmMachine::start`] lets them run.
//!
//! Interrupts per mode:
//!
//! - `on`: the PIC, IOAPIC and PIT live in the kernel. The board's GSIs below 24 go straight to
//!   `KVM_IRQ_LINE` through the GSI hook, so the emulated PIC and IOAPIC never see them and an
//!   interrupt is delivered once. The kernel also claims the PIC, PIT and IOAPIC registers, so
//!   the emulated ones are never reached. q35 installs the PC routing table of
//!   `kvm_pc_setup_irq_routing()`; microvm keeps KVM's default table, as in QEMU. MSIs, and
//!   the pins of microvm's second IOAPIC, go through `KVM_SIGNAL_MSI`.
//! - `split`: only the local APICs are in the kernel. The emulated IOAPICs send their messages
//!   with `KVM_SIGNAL_MSI`, and level triggered entries are mirrored into MSI routes so that
//!   KVM reports their EOIs, which are then broadcast to the IOAPICs. The PIC's INTR output
//!   goes to the BSP and is injected with `KVM_INTERRUPT` when the guest can take it, the way
//!   `kvm_arch_pre_run()` does.
//! - `off`: not supported yet.
//!
//! Resets stop every vCPU, reset the board, load each vCPU with its reset state and let them
//! go again. Guest shutdown, `-no-reboot` resets, crashes and internal errors are reported to
//! the [`EventHandler`] the caller gives.

use std::cell::Cell;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use kvm_bindings::{
    KVM_IRQ_ROUTING_IRQCHIP, KVM_IRQ_ROUTING_MSI, KVM_IRQCHIP_IOAPIC, KVM_IRQCHIP_PIC_MASTER,
    KVM_IRQCHIP_PIC_SLAVE, KVMIO, KvmIrqRouting, kvm_interrupt, kvm_irq_routing_entry,
    kvm_irq_routing_entry__bindgen_ty_1, kvm_irq_routing_irqchip, kvm_irq_routing_msi, kvm_msi,
    kvm_pit_config,
};
use kvm_ioctls::{VcpuFd, VmFd};
use ruvm_accel_kvm::{
    KernelIrqchip, KvmAccel, KvmError, KvmOptions, KvmVcpu, VcpuKick, VcpuStop, spawn_vcpu_thread,
};
use ruvm_hw_acpi::SystemRequest;
use ruvm_hw_core::{Clock, IrqLine};
use ruvm_hw_intc::i8259::I8259;
use ruvm_hw_intc::ioapic::{
    IOAPIC_NUM_PINS, IOAPIC_TRIGGER_LEVEL, IoApic, IoApics, ioapic_entry_parse,
};
use ruvm_mem::AddressSpace;
use ruvm_target_x86::cpuid::topo::X86CpuTopoInfo;
use ruvm_target_x86::cpuid::{Accel, X86Cpu};
use ruvm_target_x86::kvm::{X86KvmVcpu, host_cpuid, setup_vcpu};
use ruvm_target_x86::kvm_convert::PutLevel;
use vmm_sys_util::ioctl::ioctl_with_ref;
use vmm_sys_util::ioctl_iow_nr;

use crate::board::X86Board;
use crate::q35::CpuIdent;

// kvm-ioctls has no wrapper for KVM_INTERRUPT, so it is declared here.
ioctl_iow_nr!(KVM_INTERRUPT, KVMIO, 0x86, kvm_interrupt);

/// The GSIs the kernel irqchip covers, `KVM_IOAPIC_NUM_PINS`.
const KERNEL_GSIS: u32 = 24;

/// The trigger mode bit of an MSI data word, `MSI_DATA_TRIGGER_SHIFT`.
const MSI_DATA_LEVEL: u32 = 1 << 15;

/// The longest the timer thread sleeps, since arming a timer does not wake it.
const TIMER_SLICE: Duration = Duration::from_millis(1);

thread_local! {
    /// Set on vCPU threads, which must not wait for all vCPUs to stop.
    static ON_VCPU_THREAD: Cell<bool> = const { Cell::new(false) };
}

pub use crate::run_event::{EventHandler, GuestEvent, ShutdownReason};

/// The lines QEMU prints when `kvm_init()` fails: the reason, then `failed to initialize kvm`
/// with the error number's text.
pub fn init_error_lines(e: &KvmError) -> [String; 2] {
    let errno = match e {
        KvmError::Open(e)
        | KvmError::SplitIrqchip(e)
        | KvmError::CreateIrqchip(e)
        | KvmError::Ioctl(_, e) => strerror(e),
        // kvm_dirty_ring_init() gives -EIO whatever the ioctl said.
        KvmError::DirtyRing(_) | KvmError::DirtyRingBitmap(_) => "Input/output error".to_string(),
        _ => "Invalid argument".to_string(),
    };
    [e.to_string(), format!("failed to initialize kvm: {errno}")]
}

/// Opens KVM for a board whose class has `default_kernel_irqchip_split` set to
/// `default_split`. The error holds the lines of [`init_error_lines`].
pub fn open_accel(opts: &KvmOptions, default_split: bool) -> Result<KvmAccel, [String; 2]> {
    KvmAccel::new(opts, default_split).map_err(|e| init_error_lines(&e))
}

/// `kvm_pit_in_kernel()`: whether a board built for `accel` should leave out its own PIT.
pub fn pit_in_kernel(accel: &KvmAccel) -> bool {
    accel.kernel_irqchip() == KernelIrqchip::On
}

fn strerror(e: &std::io::Error) -> String {
    let s = e.to_string();
    match s.find(" (os error") {
        Some(i) => s[..i].to_string(),
        None => s,
    }
}

/// The `-cpu` model and features, checked against what the host's KVM offers.
#[derive(Debug)]
pub struct CpuModel {
    cpu: X86Cpu,
    phys_bits: u32,
    ident: CpuIdent,
    warnings: Vec<String>,
}

impl CpuModel {
    /// Parses `-cpu model,feat,...` (`qemu64` when `None`) for `accel`. Errors are QEMU's.
    pub fn new(accel: &KvmAccel, arg: Option<&str>) -> Result<CpuModel, String> {
        let arg = arg.unwrap_or("qemu64");
        let (model, features) = match arg.split_once(',') {
            Some((m, f)) => (m, f),
            None => (arg, ""),
        };
        let host = host_cpuid(accel).map_err(|e| e.to_string())?;
        let mut cpu = X86Cpu::new(model, Accel::Kvm(host)).map_err(|e| e.to_string())?;
        if !features.is_empty() {
            cpu.parse_features(features).map_err(|e| e.to_string())?;
        }
        // Realize one copy to learn the derived values and the warnings.
        let mut probe = cpu.clone();
        probe.set_topology(X86CpuTopoInfo::default(), 0);
        probe.realize().map_err(|e| e.to_string())?;
        let ident = CpuIdent::from_cpuid(probe.cpuid(0, 0), probe.cpuid(1, 0));
        Ok(CpuModel {
            cpu,
            phys_bits: probe.phys_bits(),
            ident,
            warnings: probe.warnings().to_vec(),
        })
    }

    /// The vendor and CPUID signature, for the q35 memory map and SMBIOS.
    pub fn ident(&self) -> CpuIdent {
        self.ident
    }

    /// The guest physical address width, what the board uses for its memory map.
    pub fn phys_bits(&self) -> u32 {
        self.phys_bits
    }

    /// The warnings QEMU prints for this model on this host.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    fn instance(&self, apic_id: u32) -> Result<X86Cpu, String> {
        let mut cpu = self.cpu.clone();
        cpu.set_topology(X86CpuTopoInfo::default(), apic_id);
        cpu.realize().map_err(|e| e.to_string())?;
        Ok(cpu)
    }
}

/// Options of the run loop.
#[derive(Clone, Debug, Default)]
pub struct KvmRunConfig {
    /// `-no-reboot`: a guest reset shuts the machine down instead.
    pub no_reboot: bool,
}

#[derive(Debug, Default)]
struct Ctl {
    /// The machine is started, `vm_start()`.
    running: bool,
    /// A reset was requested and not finished yet.
    reset_pending: bool,
    quit: bool,
    /// vCPUs waiting in [`Shared::park`].
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
    handler: EventHandler,
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

    fn kick(&self, index: usize) {
        if let Some(k) = self.kicks.get().and_then(|k| k.get(index)) {
            let _ = k.kick();
        }
    }

    fn kick_all(&self) {
        for k in self.kicks.get().map(Vec::as_slice).unwrap_or_default() {
            let _ = k.kick();
        }
    }

    /// `qemu_system_reset_request(SHUTDOWN_CAUSE_GUEST_RESET)`.
    fn request_reset(&self) {
        if self.no_reboot {
            (self.handler)(GuestEvent::Shutdown(ShutdownReason::GuestReset));
            return;
        }
        self.lock().reset_pending = true;
        self.cv.notify_all();
        self.kick_all();
    }

    /// Stops the machine without waiting, for a vCPU that cannot go on.
    fn stop_from_vcpu(&self) {
        self.lock().running = false;
        self.cv.notify_all();
        self.kick_all();
    }
}

/// The PIC's INTR line as the BSP sees it in split mode.
struct PicInject {
    master: Arc<I8259>,
    level: Arc<AtomicBool>,
}

struct VcpuCtx {
    vcpu: KvmVcpu,
    ctx: X86KvmVcpu,
    cpu: X86Cpu,
    is_bsp: bool,
    io: Arc<AddressSpace>,
    mem: Arc<AddressSpace>,
    pic: Option<PicInject>,
    ioapics: IoApics,
}

/// `KVM_INTERRUPT`: queue `irq` as the external interrupt the vCPU takes next.
#[allow(unsafe_code)]
fn kvm_interrupt_ioctl(fd: &VcpuFd, irq: u8) -> std::io::Result<()> {
    let intr = kvm_interrupt { irq: u32::from(irq) };
    // SAFETY: KVM_INTERRUPT reads one `struct kvm_interrupt` from the pointer, which refers to
    // a live local of exactly that type, and writes nothing back. The fd is an open vCPU.
    let ret = unsafe { ioctl_with_ref(fd, KVM_INTERRUPT(), &intr) };
    if ret < 0 { Err(std::io::Error::last_os_error()) } else { Ok(()) }
}

impl VcpuCtx {
    /// `x86_cpu_reset_hold()` followed by `kvm_arch_put_registers(KVM_PUT_RESET_STATE)`.
    fn apply_reset(&self) -> Result<(), String> {
        let state = self.cpu.new_state(self.is_bsp);
        self.ctx
            .put_registers(self.vcpu.fd(), &state, PutLevel::Reset)
            .map_err(|e| e.to_string())?;
        self.ctx.put_lapic_reset(self.vcpu.fd(), &state).map_err(|e| e.to_string())
    }

    /// The PIC part of `kvm_arch_pre_run()` for a split irqchip.
    fn pre_run(&mut self) {
        let Some(pic) = &self.pic else { return };
        let (ready, if_flag) = {
            let run = self.vcpu.fd_mut().get_kvm_run();
            (run.ready_for_interrupt_injection != 0, run.if_flag != 0)
        };
        if pic.level.load(Ordering::Acquire) && ready && if_flag {
            let irq = pic.master.pic_read_irq();
            let _ = kvm_interrupt_ioctl(self.vcpu.fd(), irq);
        }
        let want = pic.level.load(Ordering::Acquire);
        self.vcpu.fd_mut().get_kvm_run().request_interrupt_window = u8::from(want);
    }
}

/// `ioapic_eoi_broadcast()` for the IOAPIC EOI exit, which the run loop reports by name.
fn ioapic_eoi_vector(exit: &str) -> Option<u8> {
    exit.strip_prefix("IoapicEoi(")?.strip_suffix(')')?.parse().ok()
}

fn vcpu_loop(mut v: VcpuCtx, shared: Arc<Shared>) {
    ON_VCPU_THREAD.with(|c| c.set(true));
    let mut my_gen = 0u64;
    loop {
        let mut failed = None;
        {
            let mut c = shared.lock();
            if c.must_park() {
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
            }
        }
        if let Some(e) = failed {
            shared.stop_from_vcpu();
            (shared.handler)(GuestEvent::InternalError(e));
            continue;
        }

        v.pre_run();
        let stop = match v.vcpu.run(&v.io, &v.mem) {
            Ok(stop) => stop,
            Err(e) => {
                shared.stop_from_vcpu();
                (shared.handler)(GuestEvent::InternalError(e.to_string()));
                continue;
            }
        };
        match stop {
            VcpuStop::Kicked | VcpuStop::IrqWindowOpen | VcpuStop::Halted => {}
            // KVM_EXIT_SHUTDOWN is a triple fault, which resets a PC.
            VcpuStop::Shutdown | VcpuStop::Reset => shared.request_reset(),
            VcpuStop::Crash => {
                shared.stop_from_vcpu();
                (shared.handler)(GuestEvent::Panicked);
            }
            VcpuStop::InternalError => {
                shared.stop_from_vcpu();
                (shared.handler)(GuestEvent::InternalError("KVM internal error.".to_string()));
            }
            VcpuStop::FailEntry { reason, cpu } => {
                shared.stop_from_vcpu();
                (shared.handler)(GuestEvent::InternalError(format!(
                    "KVM: entry failed, hardware error 0x{reason:x} on host CPU {cpu}"
                )));
            }
            VcpuStop::Unhandled(what) => match ioapic_eoi_vector(&what) {
                Some(vector) => v.ioapics.eoi_broadcast(i32::from(vector)),
                None => {
                    shared.stop_from_vcpu();
                    (shared.handler)(GuestEvent::InternalError(format!(
                        "KVM: unhandled exit {what}"
                    )));
                }
            },
        }
    }
}

fn control_loop(shared: Arc<Shared>, board: Arc<Mutex<X86Board>>) {
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

        // qemu_system_reset(): the devices, then every CPU.
        let res = board.lock().unwrap_or_else(PoisonError::into_inner).system_reset();
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
            Ok(()) => (shared.handler)(GuestEvent::Reset),
            Err(e) => (shared.handler)(GuestEvent::InternalError(e)),
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

fn irqchip_route(gsi: u32, chip: u32, pin: u32) -> kvm_irq_routing_entry {
    kvm_irq_routing_entry {
        gsi,
        type_: KVM_IRQ_ROUTING_IRQCHIP,
        u: kvm_irq_routing_entry__bindgen_ty_1 {
            irqchip: kvm_irq_routing_irqchip { irqchip: chip, pin },
        },
        ..Default::default()
    }
}

fn msi_route(gsi: u32, addr: u64, data: u32) -> kvm_irq_routing_entry {
    kvm_irq_routing_entry {
        gsi,
        type_: KVM_IRQ_ROUTING_MSI,
        u: kvm_irq_routing_entry__bindgen_ty_1 {
            msi: kvm_irq_routing_msi {
                address_lo: addr as u32,
                address_hi: (addr >> 32) as u32,
                data,
                ..Default::default()
            },
        },
        ..Default::default()
    }
}

fn routing(entries: &[kvm_irq_routing_entry]) -> Result<KvmIrqRouting, String> {
    let mut r = KvmIrqRouting::new(0).map_err(|e| format!("{e:?}"))?;
    for e in entries {
        r.push(*e).map_err(|e| format!("{e:?}"))?;
    }
    Ok(r)
}

/// `kvm_pc_setup_irq_routing(true)`: the ISA IRQs to the PICs, and GSI 0 to IOAPIC pin 2 and
/// every other GSI but 2 to its own pin.
fn pc_irq_routes() -> Vec<kvm_irq_routing_entry> {
    let mut v = Vec::new();
    for i in 0..8 {
        if i != 2 {
            v.push(irqchip_route(i, KVM_IRQCHIP_PIC_MASTER, i));
        }
    }
    for i in 8..16 {
        v.push(irqchip_route(i, KVM_IRQCHIP_PIC_SLAVE, i - 8));
    }
    for i in 0..KERNEL_GSIS {
        if i == 0 {
            v.push(irqchip_route(i, KVM_IRQCHIP_IOAPIC, 2));
        } else if i != 2 {
            v.push(irqchip_route(i, KVM_IRQCHIP_IOAPIC, i));
        }
    }
    v
}

fn signal_msi(vm: &VmFd, addr: u64, data: u32) {
    let msi = kvm_msi {
        address_lo: addr as u32,
        address_hi: (addr >> 32) as u32,
        data,
        ..Default::default()
    };
    let _ = vm.signal_msi(msi);
}

/// The split irqchip MSI path, `kvm_send_msi()` plus the EOI routes of
/// `ioapic_update_kvm_routes()`.
///
/// KVM only asks userspace about EOIs of vectors it finds in the MSI routes of GSIs 0 to 23.
/// QEMU keeps one route per pin of the first IOAPIC. Here every level triggered entry of every
/// IOAPIC is gathered, duplicates dropped, and the set packed into those 24 routes, so the
/// second IOAPIC of microvm gets its EOIs too. The routes are rebuilt when a level triggered
/// message goes out and the set has changed.
struct SplitMsi {
    vm: Arc<VmFd>,
    ioapics: Vec<Arc<IoApic>>,
    routes: Mutex<Vec<(u32, u32)>>,
}

impl SplitMsi {
    fn deliver(&self, addr: u64, data: u32) {
        if data & MSI_DATA_LEVEL != 0 {
            self.sync_routes();
        }
        signal_msi(&self.vm, addr, data);
    }

    fn level_entries(&self) -> Vec<(u32, u32)> {
        let mut want = Vec::new();
        for io in &self.ioapics {
            for pin in 0..IOAPIC_NUM_PINS {
                let info = ioapic_entry_parse(io.redirection_entry(pin), None);
                if info.trig_mode != IOAPIC_TRIGGER_LEVEL || info.vector < 0x10 {
                    continue;
                }
                let m = (info.addr, info.data);
                if !want.contains(&m) {
                    want.push(m);
                }
            }
        }
        want.truncate(KERNEL_GSIS as usize);
        want
    }

    fn sync_routes(&self) {
        let want = self.level_entries();
        let mut cur = self.routes.lock().unwrap_or_else(PoisonError::into_inner);
        if *cur == want {
            return;
        }
        let entries: Vec<_> = want
            .iter()
            .enumerate()
            .map(|(gsi, &(addr, data))| msi_route(gsi as u32, u64::from(addr), data))
            .collect();
        if let Ok(r) = routing(&entries) {
            if self.vm.set_gsi_routing(&r).is_ok() {
                *cur = want;
            }
        }
    }
}

/// A board running on KVM.
pub struct KvmMachine {
    shared: Arc<Shared>,
    board: Arc<Mutex<X86Board>>,
    threads: Mutex<Vec<JoinHandle<()>>>,
    _accel: KvmAccel,
}

impl fmt::Debug for KvmMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KvmMachine").field("vcpus", &self.shared.nr_vcpus).finish_non_exhaustive()
    }
}

impl KvmMachine {
    /// Puts `board`, with its devices plugged, on the VM of `accel`, creates the vCPUs from
    /// `cpu`, finishes the board and starts the threads paused. `clocks` are the clocks the
    /// board's timers run on; a thread fires them.
    pub fn new(
        accel: KvmAccel,
        board: X86Board,
        cpu: &CpuModel,
        clocks: Vec<Arc<Clock>>,
        cfg: &KvmRunConfig,
        handler: EventHandler,
    ) -> Result<KvmMachine, String> {
        let mode = accel.kernel_irqchip();
        if mode == KernelIrqchip::Off {
            return Err("kernel-irqchip=off is not supported yet".to_string());
        }
        let mut board = board;
        let vm = Arc::clone(accel.vm());
        board
            .memory_system()
            .register_listener(accel.slot_listener(), board.memory_as())
            .map_err(|e| e.to_string())?;

        let apic_ids = board.apic_ids();
        let shared = Arc::new(Shared {
            ctl: Mutex::new(Ctl::default()),
            cv: Condvar::new(),
            kicks: OnceLock::new(),
            handler: Arc::clone(&handler),
            no_reboot: cfg.no_reboot,
            nr_vcpus: apic_ids.len(),
            quit: AtomicBool::new(false),
        });

        let mut bsp_pic = None;
        match mode {
            KernelIrqchip::On => {
                if board.sets_up_irq_routing() {
                    let r = routing(&pc_irq_routes())?;
                    vm.set_gsi_routing(&r)
                        .map_err(|e| format!("KVM_SET_GSI_ROUTING failed: {e}"))?;
                }
                if board.pit_wanted() {
                    vm.create_pit2(kvm_pit_config::default()).map_err(|e| {
                        format!(
                            "Create kernel PIC irqchip failed: {}",
                            strerror(&std::io::Error::from_raw_os_error(e.errno()))
                        )
                    })?;
                }
                let v = Arc::clone(&vm);
                board.set_gsi_hook(Some(Arc::new(move |gsi, level| {
                    if gsi < KERNEL_GSIS {
                        let _ = v.set_irq_line(gsi, level != 0);
                        true
                    } else {
                        false
                    }
                })));
                let v = Arc::clone(&vm);
                board.set_msi_handler(Some(Arc::new(move |addr, data| signal_msi(&v, addr, data))));
            }
            KernelIrqchip::Split => {
                let split = Arc::new(SplitMsi {
                    vm: Arc::clone(&vm),
                    ioapics: board.ioapic_list(),
                    routes: Mutex::new(Vec::new()),
                });
                board.set_msi_handler(Some(Arc::new(move |addr, data| split.deliver(addr, data))));
                if let Some(pic) = board.pic() {
                    let level = Arc::new(AtomicBool::new(false));
                    let l = Arc::clone(&level);
                    let s = Arc::downgrade(&shared);
                    board.pic_output().connect(IrqLine::from_fn(move |lvl| {
                        l.store(lvl != 0, Ordering::Release);
                        if lvl != 0 {
                            if let Some(s) = s.upgrade() {
                                s.kick(0);
                            }
                        }
                    }));
                    bsp_pic = Some(PicInject { master: Arc::clone(&pic.master), level });
                }
            }
            KernelIrqchip::Off => unreachable!("rejected above"),
        }

        {
            let s = Arc::downgrade(&shared);
            let h = Arc::clone(&handler);
            board.set_request_handler(Arc::new(move |req| match req {
                SystemRequest::Shutdown | SystemRequest::SuspendDisk => {
                    h(GuestEvent::Shutdown(ShutdownReason::GuestShutdown));
                }
                SystemRequest::Reset => {
                    if let Some(s) = s.upgrade() {
                        s.request_reset();
                    }
                }
                // S3 is not supported yet; the guest keeps running.
                SystemRequest::Suspend | SystemRequest::Wakeup(_) => {}
            }));
        }

        let io = Arc::clone(board.io_as());
        let mem = Arc::clone(board.memory_as());
        let ioapics = board.ioapics().clone();
        let mut ctxs = Vec::new();
        for (i, &apic_id) in apic_ids.iter().enumerate() {
            let is_bsp = i == 0;
            let x86 = cpu.instance(apic_id)?;
            let vcpu = accel.create_vcpu(i as u32).map_err(|e| e.to_string())?;
            let state = x86.new_state(is_bsp);
            let ctx = setup_vcpu(&accel, vcpu.fd(), &x86, &state).map_err(|e| {
                format!("kvm_init_vcpu: kvm_arch_init_vcpu failed ({apic_id}): {e}")
            })?;
            ctxs.push(VcpuCtx {
                vcpu,
                ctx,
                cpu: x86,
                is_bsp,
                io: Arc::clone(&io),
                mem: Arc::clone(&mem),
                pic: if is_bsp { bsp_pic.take() } else { None },
                ioapics: ioapics.clone(),
            });
        }

        board.machine_done()?;
        let board = Arc::new(Mutex::new(board));

        let mut vcpus = Vec::new();
        let mut kicks = Vec::new();
        for (i, v) in ctxs.into_iter().enumerate() {
            let exit = v.vcpu.exit_request();
            let s = Arc::clone(&shared);
            let t = spawn_vcpu_thread(i as u32, move || vcpu_loop(v, s))
                .map_err(|e| format!("could not create vCPU thread: {e}"))?;
            kicks.push(VcpuKick::new(&t, exit));
            vcpus.push(t);
        }
        let _ = shared.kicks.set(kicks);
        let control = {
            let s = Arc::clone(&shared);
            let b = Arc::clone(&board);
            std::thread::Builder::new()
                .name("reset".to_string())
                .spawn(move || control_loop(s, b))
                .map_err(|e| format!("could not create thread: {e}"))?
        };
        let timers = {
            let s = Arc::clone(&shared);
            std::thread::Builder::new()
                .name("timers".to_string())
                .spawn(move || timer_loop(s, clocks))
                .map_err(|e| format!("could not create thread: {e}"))?
        };
        vcpus.push(control);
        vcpus.push(timers);
        Ok(KvmMachine { shared, board, threads: Mutex::new(vcpus), _accel: accel })
    }

    /// The board, for the monitor and for device access.
    pub fn board(&self) -> &Arc<Mutex<X86Board>> {
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

    /// `pause_all_vcpus()`. Waits until every vCPU is out of the guest, unless called from a
    /// vCPU thread, which only asks.
    pub fn pause(&self) {
        self.shared.lock().running = false;
        self.shared.cv.notify_all();
        self.shared.kick_all();
        if ON_VCPU_THREAD.with(Cell::get) {
            return;
        }
        let mut c = self.shared.lock();
        while c.parked < self.shared.nr_vcpus && !c.quit {
            c = self.shared.wait(c);
        }
    }

    /// `qemu_system_reset_request()` from outside the guest.
    pub fn request_reset(&self) {
        self.shared.request_reset();
    }

    /// Stops every thread and waits for them. Must not be called from a vCPU thread or from
    /// the [`EventHandler`].
    pub fn quit(&self) {
        self.shared.lock().quit = true;
        self.shared.quit.store(true, Ordering::Release);
        self.shared.cv.notify_all();
        self.shared.kick_all();
        let threads =
            std::mem::take(&mut *self.threads.lock().unwrap_or_else(PoisonError::into_inner));
        for t in threads {
            let _ = t.join();
        }
    }
}

impl Drop for KvmMachine {
    fn drop(&mut self) {
        self.quit();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eoi_exit_names() {
        assert_eq!(ioapic_eoi_vector("IoapicEoi(33)"), Some(33));
        assert_eq!(ioapic_eoi_vector("IoapicEoi(x)"), None);
        assert_eq!(ioapic_eoi_vector("Debug"), None);
    }

    #[test]
    fn pc_routes_match_qemu() {
        let r = pc_irq_routes();
        // 7 master pins, 8 slave pins and 23 IOAPIC pins.
        assert_eq!(r.len(), 7 + 8 + 23);
        assert!(r.iter().all(|e| e.type_ == KVM_IRQ_ROUTING_IRQCHIP));
    }

    #[test]
    fn init_errors_have_two_lines() {
        let e = KvmError::Open(std::io::Error::from_raw_os_error(2));
        assert_eq!(
            init_error_lines(&e),
            [
                "Could not access KVM kernel module: No such file or directory".to_string(),
                "failed to initialize kvm: No such file or directory".to_string()
            ]
        );
    }
}
