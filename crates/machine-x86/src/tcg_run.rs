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
//! - The PIC's INTR output raises `CPU_INTERRUPT_HARD` on the BSP, as `pic_irq_request()`
//!   does. The vector is read from the PIC (`pic_read_irq()`) only when the BSP is about to
//!   take it, which is what `cpu_get_pic_interrupt()` does in QEMU.
//! - IOAPIC messages (and any other MSI) go to the vCPUs whose APIC ID matches the
//!   destination, as fixed or lowest priority interrupts.
//!
//! Resets stop every vCPU, reset the board, drop all translated code, load each vCPU with its
//! reset state and let them go again. A triple fault requests a reset, as on a PC. Guest
//! shutdown and `-no-reboot` resets are reported to the [`EventHandler`] the caller gives.
//!
//! Deliberate differences from QEMU:
//!
//! - There is no local APIC model, so there are no APIC timers, no IPIs and no INIT or SIPI:
//!   the APs stay halted for good. MSIs and IOAPIC messages are delivered straight to the
//!   vCPU's interrupt queue by destination APIC ID (physical mode) or by bit position in a
//!   flat logical destination; NMI, SMI, INIT and ExtINT messages are dropped. Since nothing
//!   sends an EOI to the IOAPICs, a level triggered IOAPIC entry delivers only once.
//! - The interrupt hook of the x86 front end is a vector queue, so a vector already handed to
//!   a vCPU survives a reset of that vCPU.
//! - The vCPUs keep the CPUID of the `-cpu` model even where it names hardware the front end
//!   does not have yet (x87, SSE, the local APIC); see the `ruvm-target-x86` documentation.
//! - The x86 front end has no System Management Mode, so TCG does not offer it
//!   ([`TCG_SMM_AVAILABLE`]): q35's `smm=auto` resolves to off and `smm=on` fails with
//!   "System Management Mode not supported by this hypervisor.", as on KVM without
//!   `KVM_CAP_X86_SMM`. SeaBIOS then skips its SMM setup.
//! - The PC `pc` (i440FX) board does not exist yet, so only microvm and q35 run here.
//! - Writes to RAM that do not come from a vCPU (DMA) do not invalidate translated code.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};
use std::thread::JoinHandle;
use std::time::Duration;

use ruvm_accel::VcpuControl;
use ruvm_accel::tcg::{TcgOptions, TcgVcpus};
use ruvm_hw_acpi::SystemRequest;
use ruvm_hw_core::{Clock, IrqLine};
use ruvm_hw_intc::i8259::I8259;
use ruvm_jit::cputlb::tlb_flush;
use ruvm_jit::native::{BackendKind, backend_of_kind};
use ruvm_jit::translate::TbBuild;
use ruvm_jit::{
    Cpu, CpuLoopExit, CpuOps, CpuShared, Jit, MmuAccessType, Ra, Tb, TbCpuState, Vcpu, Watchpoint,
    interrupt,
};
use ruvm_jit_core::Type;
use ruvm_jit_core::types::INSN_START_WORDS;
use ruvm_mem::{MemTxAttrs, MemTxResult};
use ruvm_target_x86::cpuid::topo::X86CpuTopoInfo;
use ruvm_target_x86::cpuid::{Accel, X86Cpu};
use ruvm_target_x86::tcg::{X86, env, helper_registry, jit_config};

use crate::board::X86Board;
use crate::q35::CpuIdent;
use crate::run_event::{EventHandler, GuestEvent, ShutdownReason};

/// The longest the timer thread sleeps, since arming a timer does not wake it.
const TIMER_SLICE: Duration = Duration::from_millis(1);

/// x86 has `TARGET_SUPPORTS_MTTCG`, so `thread=` defaults to `multi`.
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

/// An x86 vCPU on a PC: the front end's [`X86`] plus the board's side of
/// `cpu_get_pic_interrupt()` and of the triple fault reset.
struct PcCpu {
    x86: Arc<X86>,
    /// The PIC, on the BSP only.
    pic: Option<Arc<I8259>>,
    /// The triple faults already turned into reset requests.
    triple_faults: AtomicU64,
    shared: Weak<Shared>,
}

impl fmt::Debug for PcCpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PcCpu")
            .field("x86", &self.x86)
            .field("pic", &self.pic.is_some())
            .finish_non_exhaustive()
    }
}

impl PcCpu {
    /// A triple fault halts the vCPU in the front end; on a PC it resets the machine
    /// (`qemu_system_reset_request(SHUTDOWN_CAUSE_GUEST_RESET)` in `check_exception()`).
    fn check_triple_fault(&self) {
        let n = self.x86.triple_faults();
        if self.triple_faults.swap(n, Ordering::AcqRel) != n {
            if let Some(s) = self.shared.upgrade() {
                s.request_reset();
            }
        }
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
        self.x86.cpu_exec_enter(cpu);
    }

    fn cpu_exec_exit(&self, cpu: &mut Cpu<'_>) {
        self.x86.cpu_exec_exit(cpu);
    }

    /// `x86_cpu_exec_interrupt()` with `cpu_get_pic_interrupt()`: when the PIC asks for the
    /// CPU and the CPU can take a hardware interrupt, the vector is acknowledged now and
    /// handed to the front end, which delivers it.
    fn cpu_exec_interrupt(&self, cpu: &mut Cpu<'_>, interrupt_request: u32) -> bool {
        if interrupt_request & interrupt::HARD != 0 {
            if let Some(pic) = &self.pic {
                if pic.pic_get_output() && self.x86.has_work(cpu) {
                    let vector = pic.pic_read_irq();
                    self.x86.raise_irq(&cpu.shared(), vector);
                }
            }
        }
        self.x86.cpu_exec_interrupt(cpu, interrupt_request)
    }

    fn cpu_exec_halt(&self, cpu: &mut Cpu<'_>) -> bool {
        self.check_triple_fault();
        self.x86.cpu_exec_halt(cpu)
    }

    fn cpu_exec_reset(&self, cpu: &mut Cpu<'_>) {
        self.x86.cpu_exec_reset(cpu);
    }

    fn do_interrupt(&self, cpu: &mut Cpu<'_>) {
        self.x86.do_interrupt(cpu);
    }

    fn has_work(&self, cpu: &Cpu<'_>) -> bool {
        self.check_triple_fault();
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

/// `x86_cpu_reset_hold()` for a vCPU of a [`TcgMachine`]: the model's reset state, nothing
/// pending, an empty TLB.
fn reset_vcpu(cpu: &mut Cpu<'_>) {
    let ops = cpu.ops();
    let Some(x86) = ops.as_any().and_then(|a| a.downcast_ref::<X86>()) else { return };
    let shared = cpu.shared();
    let state = x86.model().new_state(shared.cpu_index == 0);
    env::load_state(cpu.env, &state);
    shared.reset_interrupt(u32::MAX);
    shared.halted.store(u32::from(state.halted), Ordering::Release);
    cpu.core.exception_index = -1;
    tlb_flush(cpu);
}

/// One vCPU as the interrupt sources see it.
struct Target {
    apic_id: u32,
    x86: Arc<X86>,
    shared: Arc<CpuShared>,
}

/// The MSI address and data fields of `x86_msi_info` that matter without a local APIC.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Msi {
    dest: u32,
    logical: bool,
    mode: u32,
    vector: u8,
}

/// `MSI_ADDR_DEST_ID`, `MSI_ADDR_DEST_MODE`, `MSI_DATA_DELIVERY_MODE` and `MSI_DATA_VECTOR`.
fn msi_decode(addr: u64, data: u32) -> Msi {
    Msi {
        dest: ((addr >> 12) & 0xff) as u32,
        logical: addr & (1 << 2) != 0,
        mode: (data >> 8) & 7,
        vector: data as u8,
    }
}

/// The delivery modes `apic_deliver_irq()` turns into a vector on the CPU.
const DELIVERY_FIXED: u32 = 0;
const DELIVERY_LOWPRI: u32 = 1;

/// `apic_deliver_msi()` without local APICs: the indexes of the vCPUs a message goes to.
fn msi_targets(msi: Msi, apic_ids: &[u32]) -> Vec<usize> {
    if msi.mode != DELIVERY_FIXED && msi.mode != DELIVERY_LOWPRI {
        return Vec::new();
    }
    let mut out: Vec<usize> = (0..apic_ids.len())
        .filter(|&i| {
            if msi.dest == 0xff {
                true
            } else if msi.logical {
                i < 8 && msi.dest & (1 << i) != 0
            } else {
                apic_ids[i] == msi.dest
            }
        })
        .collect();
    if msi.mode == DELIVERY_LOWPRI {
        out.truncate(1);
    }
    out
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

fn control_loop(shared: Arc<Shared>, vcpus: Arc<TcgVcpus>, board: Arc<Mutex<X86Board>>) {
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
        let mut vcpus: Vec<Vcpu> = Vec::new();
        let mut targets = Vec::new();
        for (i, &apic_id) in apic_ids.iter().enumerate() {
            let is_bsp = i == 0;
            let model = cpu.instance(apic_id)?;
            let state = model.new_state(is_bsp);
            let x86 = Arc::new(X86::new(model).with_io(Arc::clone(&io)));
            let ops = Arc::new(PcCpu {
                x86: Arc::clone(&x86),
                pic: if is_bsp { pic.clone() } else { None },
                triple_faults: AtomicU64::new(0),
                shared: Arc::downgrade(&shared),
            });
            let mut v = jit.create_vcpu(ops, Arc::clone(&mem), env::ENV_SIZE);
            env::load_state(&mut v.env, &state);
            v.shared().halted.store(u32::from(state.halted), Ordering::Release);
            targets.push(Target { apic_id, x86, shared: Arc::clone(v.shared()) });
            vcpus.push(v);
        }
        let targets = Arc::new(targets);

        // pic_irq_request(): the INTR line of the PIC goes to the BSP.
        if let Some(bsp) = targets.first() {
            let bsp_shared = Arc::clone(&bsp.shared);
            board.pic_output().connect(IrqLine::from_fn(move |level| {
                if level != 0 {
                    bsp_shared.cpu_interrupt(interrupt::HARD);
                }
            }));
        }
        {
            let t = Arc::clone(&targets);
            let ids: Vec<u32> = targets.iter().map(|t| t.apic_id).collect();
            board.set_msi_handler(Some(Arc::new(move |addr, data| {
                let msi = msi_decode(addr, data);
                for i in msi_targets(msi, &ids) {
                    t[i].x86.raise_irq(&t[i].shared, msi.vector);
                }
            })));
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
        let board = Arc::new(Mutex::new(board));
        let vcpus = Arc::new(TcgVcpus::start(&jit, vcpus));

        let control = {
            let s = Arc::clone(&shared);
            let v = Arc::clone(&vcpus);
            let b = Arc::clone(&board);
            std::thread::Builder::new()
                .name("reset".to_string())
                .spawn(move || control_loop(s, v, b))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msi_fields() {
        let m = msi_decode(0xfee0_1000, 0x0031);
        assert_eq!(m, Msi { dest: 1, logical: false, mode: 0, vector: 0x31 });
        let m = msi_decode(0xfeef_f004, 0x0141);
        assert_eq!(m, Msi { dest: 0xff, logical: true, mode: 1, vector: 0x41 });
    }

    #[test]
    fn msi_destinations() {
        let ids = [0, 1, 2, 3];
        let fixed = |dest, logical| Msi { dest, logical, mode: DELIVERY_FIXED, vector: 0x20 };
        assert_eq!(msi_targets(fixed(2, false), &ids), vec![2]);
        assert_eq!(msi_targets(fixed(7, false), &ids), Vec::<usize>::new());
        assert_eq!(msi_targets(fixed(0xff, false), &ids), vec![0, 1, 2, 3]);
        assert_eq!(msi_targets(fixed(0b1010, true), &ids), vec![1, 3]);
        let low = Msi { mode: DELIVERY_LOWPRI, ..fixed(0xff, false) };
        assert_eq!(msi_targets(low, &ids), vec![0]);
        // NMI, SMI, INIT and ExtINT are not delivered without a local APIC.
        for mode in [2, 4, 5, 7] {
            assert!(msi_targets(Msi { mode, ..fixed(0, false) }, &ids).is_empty());
        }
    }
}
