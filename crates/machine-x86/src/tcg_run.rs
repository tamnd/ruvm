// SPDX-License-Identifier: GPL-2.0-or-later

//! Running an x86 board on TCG: the parts of accel/tcg/tcg-accel-ops*.c, system/cpus.c,
//! hw/i386/x86-cpu.c and hw/i386/pc.c that tie a [`X86Board`] to the vCPUs of the x86 front
//! end (`ruvm-target-x86`) running on `ruvm-jit`.
//!
//! [`TcgMachine::new`] takes a board with its devices plugged. It creates the runtime with the
//! `-accel tcg` options ([`TcgOptions`]), one vCPU per present APIC ID with its reset state
//! (the APs halted), wires the interrupt sources, finishes the board with `machine_done()` and
//! starts the vCPU threads stopped: `CPU n/TCG` with MTTCG, `ALL CPUs/TCG` in round robin
//! mode. [`TcgMachine::start`] lets them run. Guest RAM is reached through the softmmu TLB of
//! the runtime, port I/O through the board's I/O address space and MMIO through dispatch.
//!
//! Interrupts:
//!
//! - Every vCPU has a local APIC ([`ruvm_hw_intc::apic`]) on one [`ApicBus`], as
//!   `x86_cpu_apic_create()` and `x86_cpu_apic_realize()` make them. The register window
//!   (`apic-msi`) is mapped once at 0xfee00000 over system memory with priority 0x1000, and
//!   each vCPU makes its APIC the current one while it runs, so that the window shows the
//!   registers of the vCPU that accesses it.
//! - The PIC's INTR output goes to LINT0 of every APIC that accepts it, or straight to the
//!   BSP as `CPU_INTERRUPT_HARD` while the BSP's APIC is disabled (`pic_irq_request()`).
//! - IOAPIC messages and PCI MSIs go to the APIC bus (`apic_send_msi()`).
//! - A vCPU about to take a hardware interrupt asks its APIC and then the PIC for the vector
//!   (`cpu_get_pic_interrupt()`).
//! - The APIC timers run on the virtual clock, fired by the timer thread like the board's
//!   timers.
//!
//! A change to the memory map, such as SeaBIOS moving the PAM registers to shadow the BIOS
//! in RAM, queues a TLB flush on every vCPU (`tcg_commit()`), so no TLB entry keeps pointing
//! at the old view.
//!
//! Resets stop every vCPU, reset the board, drop all translated code, load each vCPU with its
//! reset state, cold reset the APICs and let them go again. A triple fault requests a reset,
//! as on a PC. Guest shutdown and `-no-reboot` resets are reported to the [`EventHandler`]
//! the caller gives.
//!
//! Deliberate differences from QEMU:
//!
//! - The APIC bus has room for the APIC IDs of the CPUs present at startup, where QEMU sizes
//!   it for every possible CPU (`apic_id_limit`); there is no CPU hotplug here.
//! - The vCPUs keep the CPUID of the `-cpu` model even where it names hardware the front end
//!   does not have yet; see the `ruvm-target-x86` documentation.
//! - The x86 front end has no System Management Mode, so TCG does not offer it
//!   ([`TCG_SMM_AVAILABLE`]): q35's `smm=auto` resolves to off and `smm=on` fails with
//!   "System Management Mode not supported by this hypervisor.", as on KVM without
//!   `KVM_CAP_X86_SMM`. SeaBIOS then skips its SMM setup.
//! - The PC `pc` (i440FX) board does not exist yet, so only microvm and q35 run here.
//! - Writes to RAM that do not come from a vCPU (DMA) do not invalidate translated code.
//! - Timers run on their own thread, woken when a timer becomes the first to expire as
//!   `timerlist_notify()` wakes QEMU's main loop, but it also wakes at least every
//!   100 ms (`TIMER_IDLE`) in case a wakeup was missed.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ruvm_accel::VcpuControl;
use ruvm_accel::tcg::{TcgOptions, TcgVcpus};
use ruvm_base::ClockType;
use ruvm_hw_acpi::SystemRequest;
use ruvm_hw_core::{Clock, IrqLine};
use ruvm_hw_intc::apic::{
    APIC_DEFAULT_ADDRESS, APIC_SPACE_SIZE, Apic, ApicBus, ApicCpu, CpuIrq, set_current_apic,
};
use ruvm_jit::cputlb::tlb_flush;
use ruvm_jit::native::{BackendKind, backend_of_kind};
use ruvm_jit::translate::TbBuild;
use ruvm_jit::{
    Cpu, CpuLoopExit, CpuOps, CpuShared, Jit, MmuAccessType, Ra, Tb, TbCpuState, Vcpu, Watchpoint,
    interrupt,
};
use ruvm_jit_core::Type;
use ruvm_jit_core::types::INSN_START_WORDS;
use ruvm_mem::{MemTxAttrs, MemTxResult, MemoryListener};
use ruvm_target_x86::cpuid::topo::X86CpuTopoInfo;
use ruvm_target_x86::cpuid::{Accel, X86Cpu};
use ruvm_target_x86::tcg::{
    CPU_INTERRUPT_INIT, CPU_INTERRUPT_NMI, CPU_INTERRUPT_POLL, CPU_INTERRUPT_SIPI,
    CPU_INTERRUPT_SMI, X86, X86Platform, env, helper_registry, jit_config,
};

use crate::board::X86Board;
use crate::q35::CpuIdent;
use crate::run_event::{EventHandler, GuestEvent, ShutdownReason};

/// The longest the timer thread sleeps. Arming a timer that becomes the first to fire wakes it
/// through the clock's notify hook, as `timerlist_notify()` kicks QEMU's main loop, so this only
/// bounds how long a missed wakeup could delay a timer.
const TIMER_IDLE: Duration = Duration::from_millis(100);

/// x86 has `TARGET_SUPPORTS_MTTCG`, so `thread=multi` gives no warning.
const X86_SUPPORTS_MTTCG: bool = true;

/// Whether TCG can run SMM, the `smm_available` of the board. QEMU's TCG can; the x86 front
/// end here has no SMM yet.
pub const TCG_SMM_AVAILABLE: bool = false;

/// The `-cpu` model and features for TCG.
#[derive(Debug)]
pub struct TcgCpuModel {
    cpu: X86Cpu,
    phys_bits: u32,
    ident: CpuIdent,
    warnings: Vec<String>,
}

impl TcgCpuModel {
    /// Parses `-cpu model,feat,...` (`qemu64` when `None`) for TCG. Errors are QEMU's.
    pub fn new(arg: Option<&str>) -> Result<TcgCpuModel, String> {
        let arg = arg.unwrap_or("qemu64");
        let (model, features) = match arg.split_once(',') {
            Some((m, f)) => (m, f),
            None => (arg, ""),
        };
        let mut cpu = X86Cpu::new(model, Accel::Tcg).map_err(|e| e.to_string())?;
        if !features.is_empty() {
            cpu.parse_features(features).map_err(|e| e.to_string())?;
        }
        // Realize one copy to learn the derived values and the warnings.
        let mut probe = cpu.clone();
        probe.set_topology(X86CpuTopoInfo::default(), 0);
        probe.realize().map_err(|e| e.to_string())?;
        let ident = CpuIdent::from_cpuid(probe.cpuid(0, 0), probe.cpuid(1, 0));
        Ok(TcgCpuModel {
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

    /// The warnings QEMU prints for this model.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// The realized CPU with APIC ID `apic_id`.
    pub fn instance(&self, apic_id: u32) -> Result<X86Cpu, String> {
        let mut cpu = self.cpu.clone();
        cpu.set_topology(X86CpuTopoInfo::default(), apic_id);
        cpu.realize().map_err(|e| e.to_string())?;
        Ok(cpu)
    }
}

/// Options of the run loop.
#[derive(Clone, Debug, Default)]
pub struct TcgRunConfig {
    /// `-no-reboot`: a guest reset shuts the machine down instead.
    pub no_reboot: bool,
    /// The `-accel tcg` properties.
    pub tcg: TcgOptions,
    /// The code generator to use, for debugging: `None` picks the host's native one when
    /// there is one (or what `RUVM_JIT_BACKEND` asks for), `Some(BackendKind::Interp)` the
    /// IR interpreter.
    pub backend: Option<BackendKind>,
}

/// An x86 vCPU on a PC: the front end's [`X86`], which also makes its local APIC the
/// current one while it runs (`cpu_get_current_apic()` finds it through `current_cpu`).
struct PcCpu {
    x86: Arc<X86>,
    apic: OnceLock<Arc<Apic>>,
}

impl fmt::Debug for PcCpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PcCpu").field("x86", &self.x86).finish_non_exhaustive()
    }
}

impl CpuOps for PcCpu {
    fn translate_code(&self, cpu: &mut Cpu<'_>, tb: &mut TbBuild) -> Result<(), CpuLoopExit> {
        self.x86.translate_code(cpu, tb)
    }

    fn get_tb_cpu_state(&self, cpu: &Cpu<'_>) -> TbCpuState {
        self.x86.get_tb_cpu_state(cpu)
    }

    fn synchronize_from_tb(&self, cpu: &mut Cpu<'_>, tb: &Tb) {
        self.x86.synchronize_from_tb(cpu, tb);
    }

    fn restore_state_to_opc(&self, cpu: &mut Cpu<'_>, tb: &Tb, data: &[u64; INSN_START_WORDS]) {
        self.x86.restore_state_to_opc(cpu, tb, data);
    }

    fn set_pc(&self, cpu: &mut Cpu<'_>, pc: u64) {
        self.x86.set_pc(cpu, pc);
    }

    fn get_pc(&self, cpu: &Cpu<'_>) -> u64 {
        self.x86.get_pc(cpu)
    }

    fn cpu_exec_enter(&self, cpu: &mut Cpu<'_>) {
        set_current_apic(self.apic.get().cloned());
        self.x86.cpu_exec_enter(cpu);
    }

    fn cpu_exec_exit(&self, cpu: &mut Cpu<'_>) {
        self.x86.cpu_exec_exit(cpu);
        set_current_apic(None);
    }

    fn cpu_exec_interrupt(&self, cpu: &mut Cpu<'_>, interrupt_request: u32) -> bool {
        self.x86.cpu_exec_interrupt(cpu, interrupt_request)
    }

    fn cpu_exec_halt(&self, cpu: &mut Cpu<'_>) -> bool {
        self.x86.cpu_exec_halt(cpu)
    }

    fn cpu_exec_reset(&self, cpu: &mut Cpu<'_>) {
        self.x86.cpu_exec_reset(cpu);
    }

    fn do_interrupt(&self, cpu: &mut Cpu<'_>) {
        self.x86.do_interrupt(cpu);
    }

    fn has_work(&self, cpu: &Cpu<'_>) -> bool {
        self.x86.has_work(cpu)
    }

    fn tlb_fill(
        &self,
        cpu: &mut Cpu<'_>,
        addr: u64,
        size: usize,
        access_type: MmuAccessType,
        mmu_idx: usize,
        probe: bool,
        ra: Ra,
    ) -> Result<bool, CpuLoopExit> {
        self.x86.tlb_fill(cpu, addr, size, access_type, mmu_idx, probe, ra)
    }

    fn do_unaligned_access(
        &self,
        cpu: &mut Cpu<'_>,
        addr: u64,
        access_type: MmuAccessType,
        mmu_idx: usize,
        ra: Ra,
    ) -> CpuLoopExit {
        self.x86.do_unaligned_access(cpu, addr, access_type, mmu_idx, ra)
    }

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
        self.x86.do_transaction_failed(
            cpu,
            physaddr,
            addr,
            size,
            access_type,
            mmu_idx,
            attrs,
            response,
            ra,
        )
    }

    fn mmu_index(&self, cpu: &Cpu<'_>, ifetch: bool) -> usize {
        self.x86.mmu_index(cpu, ifetch)
    }

    fn pointer_wrap(&self, cpu: &Cpu<'_>, mmu_idx: usize, result: u64, base: u64) -> u64 {
        self.x86.pointer_wrap(cpu, mmu_idx, result, base)
    }

    fn debug_excp_handler(&self, cpu: &mut Cpu<'_>) {
        self.x86.debug_excp_handler(cpu);
    }

    fn debug_check_watchpoint(&self, cpu: &mut Cpu<'_>, wp: &Watchpoint) -> bool {
        self.x86.debug_check_watchpoint(cpu, wp)
    }

    fn debug_check_breakpoint(&self, cpu: &mut Cpu<'_>) -> bool {
        self.x86.debug_check_breakpoint(cpu)
    }

    fn adjust_watchpoint_address(&self, cpu: &mut Cpu<'_>, addr: u64, len: u64) -> u64 {
        self.x86.adjust_watchpoint_address(cpu, addr, len)
    }

    fn guest_default_memory_order(&self) -> u32 {
        self.x86.guest_default_memory_order()
    }

    fn addr_type(&self) -> Type {
        self.x86.addr_type()
    }

    fn precise_smc(&self) -> bool {
        self.x86.precise_smc()
    }

    /// The front end's helpers find their [`X86`] through this, as they would without the
    /// board around it.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        self.x86.as_any()
    }
}

/// `tcg_commit()`: a change to the memory map flushes every vCPU's TLB so no entry points at
/// the old view. As in QEMU the flush is queued with `async_run_on_cpu()`, which kicks the
/// vCPU out of its translated code.
struct TlbCommit(Weak<Jit>);

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

/// `x86_cpu_reset_hold()` for a vCPU of a [`TcgMachine`]: the model's reset state, nothing
/// pending, an empty TLB. The APICs are cold reset after this (`x86_cpu_after_reset()`).
fn reset_vcpu(cpu: &mut Cpu<'_>) {
    let ops = cpu.ops();
    let Some(x86) = ops.as_any().and_then(|a| a.downcast_ref::<X86>()) else { return };
    let shared = cpu.shared();
    let state = x86.model().new_state(shared.cpu_index == 0);
    env::load_state(cpu.env, &state);
    shared.reset_interrupt(u32::MAX);
    shared.halted.store(u32::from(state.halted), Ordering::Release);
    x86.clear_irqs();
    cpu.core.exception_index = -1;
    tlb_flush(cpu);
}

/// The `CPU_INTERRUPT_*` bit of an APIC request.
fn irq_bit(irq: CpuIrq) -> u32 {
    match irq {
        CpuIrq::Hard => interrupt::HARD,
        CpuIrq::Poll => CPU_INTERRUPT_POLL,
        CpuIrq::Smi => CPU_INTERRUPT_SMI,
        CpuIrq::Nmi => CPU_INTERRUPT_NMI,
        CpuIrq::Init => CPU_INTERRUPT_INIT,
        CpuIrq::Sipi => CPU_INTERRUPT_SIPI,
    }
}

/// The CPU an APIC belongs to, `s->cpu`.
struct ApicLink {
    shared: Arc<CpuShared>,
    x86: Weak<X86>,
    x2apic: bool,
}

impl ApicCpu for ApicLink {
    fn cpu_interrupt(&self, irq: CpuIrq) {
        self.shared.cpu_interrupt(irq_bit(irq));
    }

    fn cpu_reset_interrupt(&self, irq: CpuIrq) {
        self.shared.reset_interrupt(irq_bit(irq));
    }

    fn is_self(&self) -> bool {
        self.shared.is_self()
    }

    fn has_x2apic(&self) -> bool {
        self.x2apic
    }

    fn set_apic_feature(&self, on: bool) {
        if let Some(x) = self.x86.upgrade() {
            x.set_apic_feature(on);
        }
    }
}

/// What a vCPU reaches outside itself: its APIC (`cpu->apic_state`), the 8259 (`isa_pic`)
/// and system reset.
struct PcPlatform {
    apic: Arc<Apic>,
    bus: Arc<ApicBus>,
    machine: Weak<Shared>,
}

impl X86Platform for PcPlatform {
    /// `cpu_get_pic_interrupt()`.
    fn get_pic_interrupt(&self) -> Option<u8> {
        let intno = self.apic.get_interrupt();
        if intno >= 0 {
            return Some(intno as u8);
        }
        // Read the IRQ from the PIC.
        if !self.apic.accept_pic_intr() {
            return None;
        }
        self.bus.pic().map(|p| p.pic_read_irq())
    }

    fn apic_poll_irq(&self) {
        self.apic.poll_irq();
    }

    fn apic_sipi(&self) -> Option<u8> {
        self.apic.sipi()
    }

    fn apic_init_reset(&self) {
        self.apic.init_reset();
    }

    fn apic_base(&self) -> u64 {
        self.apic.apic_base()
    }

    fn set_apic_base(&self, val: u64) -> bool {
        self.apic.set_base(val)
    }

    fn apic_tpr(&self) -> u8 {
        self.apic.tpr()
    }

    fn set_apic_tpr(&self, val: u8) {
        self.apic.set_tpr(val);
    }

    fn apic_msr_read(&self, index: u32) -> Option<u64> {
        self.apic.msr_read(index)
    }

    fn apic_msr_write(&self, index: u32, val: u64) -> bool {
        self.apic.msr_write(index, val)
    }

    fn system_reset_request(&self) {
        if let Some(s) = self.machine.upgrade() {
            s.request_reset();
        }
    }
}

/// `pic_irq_request()`: the INTR output of the 8259 at `level`.
fn pic_irq_request(apics: &[Arc<Apic>], bsp: &CpuShared, level: bool) {
    let Some(first) = apics.first() else { return };
    if first.is_enabled() {
        for a in apics {
            if a.accept_pic_intr() {
                a.deliver_pic_intr(level);
            }
        }
    } else if level {
        bsp.cpu_interrupt(interrupt::HARD);
    } else {
        bsp.reset_interrupt(interrupt::HARD);
    }
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
    handler: EventHandler,
    no_reboot: bool,
    quit: AtomicBool,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Ctl> {
        self.ctl.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `qemu_system_reset_request(SHUTDOWN_CAUSE_GUEST_RESET)`. The control thread does the
    /// work, so this is safe from a vCPU thread and from device callbacks.
    fn request_reset(&self) {
        if self.no_reboot {
            (self.handler)(GuestEvent::Shutdown(ShutdownReason::GuestReset));
            return;
        }
        self.lock().reset_pending = true;
        self.cv.notify_all();
    }
}

fn control_loop(
    shared: Arc<Shared>,
    vcpus: Arc<TcgVcpus>,
    board: Arc<Mutex<X86Board>>,
    apics: Vec<Arc<Apic>>,
) {
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
        let res = board.lock().unwrap_or_else(PoisonError::into_inner).system_reset();
        // The firmware is copied back into RAM; nothing translated before may survive.
        vcpus.jit().tb_flush_exclusive_or_serial();
        vcpus.run_on_each(reset_vcpu);
        for (i, a) in apics.iter().enumerate() {
            a.reset(i == 0);
        }
        {
            let mut c = shared.lock();
            c.reset_pending = false;
            if c.running && !c.quit {
                vcpus.resume_all();
            }
        }
        match res {
            Ok(()) => (shared.handler)(GuestEvent::Reset),
            Err(e) => (shared.handler)(GuestEvent::InternalError(e)),
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

/// A board running on TCG.
pub struct TcgMachine {
    shared: Arc<Shared>,
    board: Arc<Mutex<X86Board>>,
    vcpus: Arc<TcgVcpus>,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl fmt::Debug for TcgMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TcgMachine")
            .field("vcpus", &self.vcpus.vcpu_count())
            .field("mttcg", &self.mttcg())
            .finish_non_exhaustive()
    }
}

impl TcgMachine {
    /// Creates the runtime for `cfg`, the vCPUs from `cpu` with the board's APIC IDs, wires
    /// the interrupts, finishes the board and starts the vCPU threads stopped. `clocks` are
    /// the clocks the board's timers run on; a thread fires them. Also gives the warnings to
    /// print, such as QEMU's for `thread=multi` on a guest without MTTCG.
    pub fn new(
        board: X86Board,
        cpu: &TcgCpuModel,
        clocks: Vec<Arc<Clock>>,
        cfg: &TcgRunConfig,
        handler: EventHandler,
    ) -> Result<(TcgMachine, Vec<String>), String> {
        let mut board = board;
        let (config, warnings) = cfg.tcg.jit_config(jit_config(), X86_SUPPORTS_MTTCG)?;
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

        let io = Arc::clone(board.io_as());
        let mem = Arc::clone(board.memory_as());
        let apic_ids = board.apic_ids();
        let pic = board.pic().map(|p| Arc::clone(&p.master));
        // The APIC timers run on the virtual clock.
        let vclock = clocks
            .iter()
            .find(|c| c.kind() == ClockType::Virtual)
            .or(clocks.first())
            .cloned()
            .ok_or_else(|| "no virtual clock".to_string())?;
        let max_apic_id = apic_ids.iter().copied().max().map_or(1, |m| m + 1);
        let bus = ApicBus::new(max_apic_id, pic, board.ioapics().clone());
        // cpu_get_ticks() is one count for the whole machine.
        let tsc_base = Instant::now();
        let mut vcpus: Vec<Vcpu> = Vec::new();
        let mut apics = Vec::new();
        let mut bsp_shared = None;
        for (i, &apic_id) in apic_ids.iter().enumerate() {
            let is_bsp = i == 0;
            let model = cpu.instance(apic_id)?;
            let state = model.new_state(is_bsp);
            let x2apic = model.has_feature("x2apic");
            let x86 = Arc::new(X86::new(model).with_io(Arc::clone(&io)).with_tsc_base(tsc_base));
            let ops = Arc::new(PcCpu { x86: Arc::clone(&x86), apic: OnceLock::new() });
            let mut v = jit.create_vcpu(ops.clone(), Arc::clone(&mem), env::ENV_SIZE);
            env::load_state(&mut v.env, &state);
            v.shared().halted.store(u32::from(state.halted), Ordering::Release);
            let link =
                ApicLink { shared: Arc::clone(v.shared()), x86: Arc::downgrade(&x86), x2apic };
            let apic = Apic::realize(&bus, &vclock, apic_id, is_bsp, Arc::new(link))
                .map_err(|e| e.to_string())?;
            let _ = ops.apic.set(Arc::clone(&apic));
            x86.set_platform(Arc::new(PcPlatform {
                apic: Arc::clone(&apic),
                bus: Arc::clone(&bus),
                machine: Arc::downgrade(&shared),
            }));
            if is_bsp {
                bsp_shared = Some(Arc::clone(v.shared()));
            }
            apics.push(apic);
            vcpus.push(v);
        }

        // x86_cpu_apic_realize(): the register window, mapped once for every APIC.
        {
            let ms = board.memory_system();
            let r = ms
                .new_io("apic-msi", APIC_SPACE_SIZE.into(), bus.mmio())
                .map_err(|e| e.to_string())?;
            ms.add_subregion_overlap(board.system_memory(), APIC_DEFAULT_ADDRESS, r, 0x1000)
                .map_err(|e| e.to_string())?;
        }
        if let Some(bsp) = bsp_shared {
            let a = apics.clone();
            board.pic_output().connect(IrqLine::from_fn(move |level| {
                pic_irq_request(&a, &bsp, level != 0);
            }));
        }
        {
            let b = Arc::clone(&bus);
            board.set_msi_handler(Some(Arc::new(move |addr, data| b.send_msi(addr, data))));
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

        board.machine_done()?;
        board
            .memory_system()
            .register_listener(Arc::new(TlbCommit(Arc::downgrade(&jit))), board.memory_as())
            .map_err(|e| e.to_string())?;
        let board = Arc::new(Mutex::new(board));
        let vcpus = Arc::new(TcgVcpus::start(&jit, vcpus));

        let control = {
            let s = Arc::clone(&shared);
            let v = Arc::clone(&vcpus);
            let b = Arc::clone(&board);
            std::thread::Builder::new()
                .name("reset".to_string())
                .spawn(move || control_loop(s, v, b, apics))
                .map_err(|e| format!("could not create thread: {e}"))?
        };
        let timers = {
            let s = Arc::clone(&shared);
            std::thread::Builder::new()
                .name("timers".to_string())
                .spawn(move || timer_loop(s, clocks))
                .map_err(|e| format!("could not create thread: {e}"))?
        };
        let machine =
            TcgMachine { shared, board, vcpus, threads: Mutex::new(vec![control, timers]) };
        Ok((machine, warnings))
    }

    /// The board, for the monitor and for device access.
    pub fn board(&self) -> &Arc<Mutex<X86Board>> {
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
    /// the [`EventHandler`].
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
        drop(self.vcpus.quit());
    }
}

impl Drop for TcgMachine {
    fn drop(&mut self) {
        self.quit();
    }
}
