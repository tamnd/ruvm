# 06. Accelerators

This document specifies ruvm-accel, the `Accel` and `Vcpu` traits, the vCPU thread and run loop that sit on top of them, and every backend crate: ruvm-accel-kvm, ruvm-accel-hvf, ruvm-accel-whpx, ruvm-accel-mshv, ruvm-accel-nvmm, ruvm-accel-xen, ruvm-accel-nitro, ruvm-accel-qtest, and the JIT (ruvm-jit, documents 07 and 08) in its role as an accelerator. The reference is QEMU 11.1.0: accel/accel-common.c, accel/accel-system.c, system/cpus.c, accel/kvm/kvm-all.c, target/*/kvm, accel/hvf, target/arm/hvf, target/i386/hvf, accel/whpx, target/i386/whpx, accel/mshv, target/i386/mshv, target/i386/nvmm, accel/xen, hw/xen, hw/i386/kvm, accel/nitro, accel/qtest and accel/tcg. Memory slots, dirty bitmaps and guest_memfd are specified in document 05; confidential computing policy in document 19; interrupt controllers in document 12; CPU models, CPUID and MSR tables in document 09.

## Goals

1. Same `-accel` and `-machine accel=` syntax, the same accelerator property names, defaults and error messages, and the same fallback order when several accelerators are listed (`-accel kvm -accel tcg` tries each in turn, as `configure_accelerators()` in system/vl.c does).
2. Same guest-visible behavior per accelerator. A guest running under ruvm with KVM must see the CPUID, MSRs, irqchip and paravirtual interfaces it would see under QEMU with KVM on the same host, because that is what migration and Windows activation depend on.
3. A run loop that handles the common exits (port I/O, MMIO, halt, interrupt window) on the vCPU thread with no global lock, no allocation and no round trip through the main thread.
4. State synchronization that is lazy in the same places QEMU is lazy, so that the number of register get and set ioctls per exit is never higher than QEMU's, and lower where ruvm can track register classes separately.
5. One trait surface that every backend implements, so that the machine and device code never names a specific hypervisor except through capability queries.

## Crates

ruvm-accel (L2, GPL) holds the traits, the vCPU thread, the kick machinery, lazy register sync, the `run_on_cpu` queue, pause and resume, and the accelerator registry. Each backend is its own crate that depends on ruvm-accel, ruvm-mem and the relevant target crates, and registers itself through a `ruvm_qom::register_type!` entry named `kvm-accel`, `hvf-accel`, `whpx-accel`, `mshv-accel`, `nvmm-accel`, `xen-accel`, `nitro-accel`, `qtest-accel` and `tcg-accel`, the same QOM type names as QEMU (`ACCEL_CLASS_NAME()` appends `-accel`). Host FFI lives in ruvm-sys (document 24): kvm-bindings and kvm-ioctls from rust-vmm plus newer ioctls, Hypervisor.framework, WinHvPlatform.dll loaded with the same lazy dispatch table QEMU uses (`whp_dispatch` in target/i386/whpx/whpx-all.c, loaded on first use), `/dev/mshv`, libnvmm, libxenctrl and friends, and `/dev/nitro_enclaves`.

QEMU removed MIPS KVM support, so ruvm-accel-kvm covers x86, Arm, RISC-V, s390x, PowerPC and LoongArch, which is the set QEMU 11.1 builds `target/*/kvm` for. Document 24 lists the same set.

A new sharing decision: the x86 instruction emulator that QEMU keeps in target/i386/emulate (x86_decode.c, x86_emu.c, used by HVF on Intel Macs, by WHPX and by MSHV) becomes a module of ruvm-target-x86 with a small `EmulatorOps` trait for register, segment and memory access. HVF x86, WHPX and MSHV use it. NVMM does not, because QEMU's NVMM backend delegates MMIO and port I/O instruction completion to libnvmm's `nvmm_assist_mem()` and `nvmm_assist_io()`, and ruvm does the same for identical behavior. KVM never needs it on x86 because the kernel completes the instruction.

## Vocabulary

A partition (Microsoft's term) or VM is the host hypervisor object that owns guest physical memory. A vCPU fd or handle is the host object for one virtual CPU. An exit is a return from the host's run call to ruvm. Kick means forcing a vCPU that is in or about to enter the run call to return promptly. State sync means copying architectural register state between the host hypervisor and ruvm's `ArchState` for that vCPU. A put level is how much state a sync writes back: QEMU's `KVM_PUT_RUNTIME_STATE`, `KVM_PUT_RESET_STATE` and `KVM_PUT_FULL_STATE` in include/system/kvm.h.

## The Accel trait

```rust
/// One per VM. Created by `-accel` parsing, initialized when the machine is created.
pub trait Accel: Object + Send + Sync {
    /// QEMU's AccelClass::init_machine. Opens the hypervisor, creates the VM,
    /// checks required capabilities, installs memory listeners.
    fn init_machine(&self, m: &MachineCtx) -> Result<(), AccelError>;

    /// Called after all devices are realized and before the first vCPU runs.
    fn setup_post(&self, m: &MachineCtx) -> Result<(), AccelError> { Ok(()) }

    fn create_vcpu(&self, cpu: &CpuHandle) -> Result<Box<dyn Vcpu>, AccelError>;

    fn caps(&self) -> &AccelCaps;

    /// Memory slot mapping. Called from the MemoryListener in document 05.
    fn slots(&self) -> Option<&dyn SlotOps> { None }
    fn dirty(&self) -> Option<&dyn DirtyLog> { None }
    fn irqchip(&self) -> Option<&dyn IrqChipOps> { None }
    fn ioeventfd(&self) -> Option<&dyn IoEventOps> { None }

    /// Whether the accelerator executes guest code in another context
    /// (Xen, nitro, qtest). If false, vCPU threads are dummies.
    fn runs_vcpus(&self) -> bool { true }

    fn has_memory(&self, asidx: u32, gpa: u64, len: u64) -> bool;
    fn gdbstub_supported_sstep_flags(&self) -> SstepFlags;
}

pub trait SlotOps {
    fn set_slot(&self, s: &SlotUpdate<'_>) -> Result<(), AccelError>;
    fn max_slots(&self) -> u32;
    fn max_slot_size(&self) -> Option<u64>;
}

pub trait DirtyLog {
    fn start(&self, slot: SlotId) -> Result<(), AccelError>;
    fn stop(&self, slot: SlotId) -> Result<(), AccelError>;
    /// Merge dirty pages into the MIGRATION bitmap of the RamBlock (document 05).
    fn sync(&self, slot: SlotId, sink: &mut dyn DirtySink) -> Result<(), AccelError>;
    fn clear(&self, slot: SlotId, first_page: u64, npages: u64) -> Result<(), AccelError>;
    fn sync_global(&self) -> Result<(), AccelError> { Ok(()) }
}

pub trait IrqChipOps {
    fn mode(&self) -> IrqChipMode;            // Off, On, Split
    fn set_irq(&self, gsi: u32, level: bool) -> Result<(), AccelError>;
    fn send_msi(&self, msg: MsiMessage, requester: u16) -> Result<(), AccelError>;
    fn begin_route_changes(&self) -> RouteChange<'_>;
    fn add_irqfd(&self, fd: BorrowedFd<'_>, gsi: u32, resample: Option<BorrowedFd<'_>>)
        -> Result<(), AccelError>;
    fn remove_irqfd(&self, fd: BorrowedFd<'_>, gsi: u32) -> Result<(), AccelError>;
}
```

`AccelCaps` is a plain struct filled once at `init_machine`: irqchip modes available, ioeventfd, irqfd, MSI routing, dirty tracking kind (none, bitmap, ring, write protect), maximum vCPUs (`KVM_CAP_NR_VCPUS` and `KVM_CAP_MAX_VCPUS` for KVM), maximum memslots, read-only memory, coalesced MMIO, guest_memfd, nested, protected state, and guest debug. Device and machine code query `caps()` rather than asking "is this KVM", which replaces QEMU's scattered `kvm_enabled()` checks. Where QEMU's behavior depends on the accelerator identity rather than a capability (for example a machine compat default that applies only under HVF, see the vGIC section), the code asks `accel.type_name()` explicitly, so such cases are greppable.

Route changes are a guard, following QEMU 11.1's `accel_irqchip_begin_route_changes()` which returns an `AccelRouteChange` that `kvm_irqchip_add_msi_route()` fills. The guard batches additions and commits the route table once on drop, so a guest that reprograms 64 MSI-X vectors issues one `KVM_SET_GSI_ROUTING` rather than 64.

## The Vcpu trait

```rust
/// One per vCPU, owned by its vCPU thread. Not Sync.
pub trait Vcpu: Send {
    /// Enter the guest. Returns when the guest exits or the kick fires.
    /// The returned exit may borrow the host's shared run structure.
    fn run(&mut self, kick: &KickState) -> Result<VcpuExit<'_>, AccelError>;

    /// Complete an exit that needs data written back (MMIO read, PIO in,
    /// MSR read, hypercall return). Backends that complete by writing the
    /// shared run page do it here without a syscall.
    fn complete(&mut self, c: Completion) -> Result<(), AccelError>;

    fn get_state(&mut self, set: RegSet, st: &mut ArchState) -> Result<(), AccelError>;
    fn put_state(&mut self, set: RegSet, level: PutLevel, st: &ArchState)
        -> Result<(), AccelError>;

    fn inject(&mut self, ev: InjectEvent) -> Result<(), AccelError>;
    fn set_guest_debug(&mut self, dbg: &GuestDebug) -> Result<(), AccelError>;
    fn kicker(&self) -> Kicker;
    fn state_protected(&self) -> bool { false }
}

pub enum VcpuExit<'a> {
    Io { port: u16, size: u8, dir: Dir, data: &'a mut [u8], count: u32, attrs: MemTxAttrs },
    Mmio { gpa: u64, data: &'a mut [u8], is_write: bool, attrs: MemTxAttrs },
    Halt,
    IrqWindowOpen,
    Interrupted,                  // kick or signal; KVM_RUN returned -EINTR/-EAGAIN
    Shutdown(ShutdownCause),
    SystemEvent(SystemEvent),     // PSCI off/reset, SEV termination, crash
    MemoryFault { gpa: u64, size: u64, private: bool },
    DirtyRingFull,
    Debug(DebugExit),
    Arch(ArchExit<'a>),           // target specific: MSR, CPUID, EOI, hypercall, s390 SIE...
    InternalError(InternalError),
}
```

`VcpuExit<'a>` borrows from the backend's shared run structure. For KVM, `Io.data` is a slice of the mmapped `kvm_run` page at `io.data_offset` and `Mmio.data` is `kvm_run.mmio.data`, so an MMIO read is completed by dispatching straight into that slice; no copy into an intermediate exit struct, as document 21 requires. The borrow ends before the next `run`, which the type system enforces because `run` takes `&mut self`.

`RegSet` is a bit set of register classes per architecture: for x86, GPR, SREGS, FPU/XSAVE, XCRS, MSRS, EVENTS, DEBUGREGS, LAPIC, MP_STATE, NESTED, TSC; for Arm, CORE, SYSREGS, FPSIMD, SVE, SME, PMU, TIMER, MP_STATE; and so on. `ArchState` is the architecture-neutral handle from the canon: an enum over per-target state structs owned by the target crate (document 09), so the same struct serves the JIT, which keeps state there natively.

## vCPU threads

Each vCPU has one OS thread named like QEMU's (`CPU 0/KVM`, `CPU 1/HVF`, `CPU 0/TCG`), because libvirt and administrators find them by name and `query-cpus-fast` reports their `thread-id`. The thread function is shared by every backend and replaces `kvm_vcpu_thread_fn()` in accel/kvm/kvm-accel-ops.c, `hvf_cpu_thread_fn()`, `whpx_cpu_thread_fn()` and their siblings:

```rust
fn vcpu_thread(cpu: CpuHandle, mut v: Box<dyn Vcpu>, ctl: &VcpuControl) {
    rcu::register_thread();
    ctl.signal_created();
    loop {
        ctl.wait_while_stopped();                 // pause_all_vcpus, vm_stop, cpu->stop
        ctl.drain_work(&mut *v, &cpu);            // run_on_cpu / async_run_on_cpu items
        if ctl.should_unplug() { break; }
        if !cpu.can_run() { ctl.wait_for_event(); continue; }   // halted, no work
        match run_until_event(&mut *v, &cpu, ctl) {
            Ok(Event::Halted) => cpu.set_halted(),
            Ok(Event::Debug) => ctl.request_debug_stop(),
            Ok(Event::Interrupted) => {}
            Err(e) => { cpu.dump_state(); ctl.request_internal_error(e); ctl.stop_self(); }
        }
    }
    v.destroy();
    ctl.signal_destroyed();
}
```

`wait_for_event` parks on a per-vCPU futex (a `parking_lot`-style condvar on macOS and Windows), not on QEMU's `qemu_cpu_cond` with the BQL, since there is no BQL. Halt semantics follow the target: on x86 with an in-kernel LAPIC, HLT is handled in the kernel and ruvm never sees it; with a userspace APIC (TCG, HVF x86, WHPX with `kernel-irqchip=off`, NVMM), `VcpuExit::Halt` parks the thread until an interrupt is raised for that CPU, and the interrupt controller's delivery code (document 12) wakes it.

## The run loop and exit dispatch

`run_until_event` is the equivalent of `kvm_cpu_exec()`. The order of operations is QEMU's, because several of them are visible to the guest or to the hypervisor's instruction completion rules:

```rust
fn run_until_event(v: &mut dyn Vcpu, cpu: &CpuHandle, ctl: &VcpuControl)
    -> Result<Event, AccelError>
{
    if cpu.arch().process_async_events(v)? { return Ok(Event::Halted); }   // INIT/SIPI, NMI
    loop {
        cpu.sync().flush_dirty(v, PutLevel::Runtime)?;    // vcpu_dirty -> put RUNTIME
        cpu.arch().pre_run(v)?;                            // pending IRQ injection, TPR
        let kick = ctl.kick_state();
        if kick.exit_requested_acquire() { kick.kick_self(); }  // re-enter once, exit at once
        let exit = v.run(kick)?;
        let attrs = cpu.arch().post_run(v, &exit);          // e.g. x86 TPR, SMM attrs
        match exit {
            VcpuExit::Interrupted => { kick.clear_self(); return Ok(Event::Interrupted) }
            VcpuExit::Io { port, size, dir, data, count, .. } =>
                io::dispatch_pio(cpu.io_as(), port, size, dir, data, count, attrs),
            VcpuExit::Mmio { gpa, data, is_write, .. } =>
                cpu.memory_as().rw(gpa, attrs, data, is_write),
            VcpuExit::IrqWindowOpen => return Ok(Event::Interrupted),
            VcpuExit::Shutdown(c) => { ctl.post_reset_request(c); return Ok(Event::Interrupted) }
            VcpuExit::SystemEvent(e) => if let Some(ev) = handle_system_event(cpu, v, e)? {
                return Ok(ev) },
            VcpuExit::MemoryFault { gpa, size, private } =>
                cpu.machine().convert_memory(gpa, size, private)?,
            VcpuExit::DirtyRingFull => cpu.accel().dirty_ring_reap(Some(cpu))?,
            VcpuExit::Debug(d) => if cpu.arch().debug_exit(v, d)? { return Ok(Event::Debug) },
            VcpuExit::Arch(a) => if let Some(ev) = cpu.arch().handle_exit(v, a)? { return Ok(ev) },
            VcpuExit::InternalError(e) => return Err(e.into()),
            VcpuExit::Halt => return Ok(Event::Halted),
        }
    }
}
```

Three QEMU rules are preserved exactly.

First, after a PIO or MMIO exit the vCPU must re-enter the hypervisor before its state is read or migrated, because KVM completes the emulated instruction on the next `KVM_RUN` (the kernel's api.rst says userspace "should ensure that the operation is completed before performing a live migration" and that re-entering with `immediate_exit` set completes pending operations without running further guest instructions). That is why `kvm_cpu_exec()` checks `exit_request` after `kvm_arch_pre_run()` and kicks itself instead of returning: the next `KVM_RUN` completes the instruction and returns `-EINTR` at once. ruvm keeps the self kick. The same holds for HVF and WHPX in a weaker form, since they rely on ruvm to advance RIP after emulation; the emulator does that before `complete`.

Second, `exit_request` is read with acquire ordering and written with release ordering in `cpu_exit()`, so a kick that races with entry is never lost. ruvm uses `AtomicBool` with the same orderings.

Third, the exit reasons that QEMU handles in generic code and the ones it forwards to `kvm_arch_handle_exit()` stay split the same way, because the arch handlers have side effects that depend on ordering. In kvm-all.c the generic set is `KVM_EXIT_IO`, `KVM_EXIT_MMIO`, `KVM_EXIT_IRQ_WINDOW_OPEN`, `KVM_EXIT_SHUTDOWN`, `KVM_EXIT_UNKNOWN`, `KVM_EXIT_INTERNAL_ERROR`, `KVM_EXIT_DIRTY_RING_FULL`, `KVM_EXIT_SYSTEM_EVENT` (shutdown, reset, crash, SEV termination; others fall through to the arch) and `KVM_EXIT_MEMORY_FAULT`. On x86 `kvm_arch_handle_exit()` in target/i386/kvm/kvm.c handles `KVM_EXIT_HLT`, `SET_TPR`, `TPR_ACCESS`, `FAIL_ENTRY`, `EXCEPTION`, `DEBUG`, `HYPERV`, `IOAPIC_EOI`, `X86_BUS_LOCK`, `NOTIFY`, `X86_RDMSR`, `X86_WRMSR`, `XEN`, `HYPERCALL`, `SYSTEM_EVENT` and `TDX`. ruvm maps each of these to an `ArchExit` variant handled by `ruvm-target-x86::kvm`.

`KVM_RUN` returning `-EFAULT` with `exit_reason == KVM_EXIT_MEMORY_FAULT` is not an error; it is how the kernel asks for a private/shared conversion (document 05). Every other negative return except `-EINTR` and `-EAGAIN` stops the VM with runstate `internal-error` after dumping CPU state with `CPU_DUMP_CODE`, as QEMU does; error strings match (`error: kvm run failed %s`, and on PowerPC the SMT hint).

Port I/O dispatch loops `count` times for string I/O, advancing through `data` by `size`, into the `I/O` address space with the attrs from `post_run`, exactly as `kvm_handle_io()`. MMIO goes to `cpu.memory_as()`, which in QEMU is always `address_space_memory` for KVM exits, while x86 SMM accesses are distinguished by attrs from `kvm_arch_post_run()` selecting the SMM address space through `cpu_asidx_from_attrs()` (document 05). Both run with no lock except the target region's domain.

## Kick

A kick has two parts: set `exit_request` (release), then make sure the target thread leaves or does not enter the run call. The second part is backend specific:

| Backend | Mechanism | QEMU reference |
|---|---|---|
| KVM | `pthread_kill(SIG_IPI)`; the handler sets `kvm_run->immediate_exit = 1` | `kvm_ipi_signal()`, `kvm_cpu_kick()` |
| HVF Arm | `hv_vcpus_exit()` plus the signal for threads parked outside the run call | `hvf_kick_vcpu_thread()`, `cpus_kick_thread()` |
| HVF x86 | `hv_vcpu_interrupt()` plus signal | target/i386/hvf |
| WHPX | `WHvCancelRunVirtualProcessor()` | accel/whpx/whpx-common.c |
| MSHV | SIG_IPI to a thread in `MSHV_RUN_VP`; the kernel returns and the loop exits | accel/mshv/mshv-all.c |
| NVMM | `nvmm_vcpu_stop()` from the signal handler | target/i386/nvmm |
| JIT | set the per-vCPU `icount_decr` high half, checked at every TB entry | accel/tcg/cpu-exec.c |

For KVM, QEMU supports two schemes. With `KVM_CAP_IMMEDIATE_EXIT`, SIG_IPI is unblocked in userspace and its handler writes `immediate_exit`, which the kernel polls once when `KVM_RUN` starts; a signal that arrives while the thread is inside `KVM_RUN` interrupts it with `-EINTR`. Without the capability, QEMU blocks SIG_IPI in userspace and uses `KVM_SET_SIGNAL_MASK` so that it is unblocked only inside `KVM_RUN`, and drains it afterwards with `sigtimedwait()` in `kvm_eat_signals()`. The kernel's documentation says the signal mask approach "has worse scalability". Decision: ruvm-accel-kvm requires `KVM_CAP_IMMEDIATE_EXIT` and has no signal mask path. Every kernel that has the other capabilities ruvm requires (below) has it.

The signal number is SIG_IPI, which is SIGUSR1 in QEMU (include/qemu/osdep.h), and ruvm uses the same so that host tooling that filters signals behaves the same. The signal handler is async-signal-safe: it reads a thread-local pointer to the vCPU's `kvm_run` page and does a single relaxed store.

Coalescing: `Kicker::kick()` swaps `exit_request` first and signals only if the previous value was false and the target is running, so several interrupts in a row cost one signal. QEMU's `qemu_cpu_kick()` signals on every call except under TCG.

With an in-kernel irqchip, device interrupts are not kicks at all: the device's IRQ line writes an irqfd (below) or calls `KVM_IRQ_LINE`, and the kernel IPIs the physical CPU if needed.

## Lazy state synchronization

QEMU keeps one flag per vCPU, `cpu->vcpu_dirty`. It means "the authoritative copy of this vCPU's registers is in QEMU's `CPUArchState`, and must be written back before the next run". The rules, from accel/kvm/kvm-all.c:

- `kvm_cpu_synchronize_state()` runs `do_kvm_cpu_synchronize_state()` on the vCPU's own thread through `run_on_cpu()`. If the vCPU is not dirty and its state is not protected (`guest_state_protected`, set for SEV-ES, SEV-SNP and TDX), it calls `kvm_arch_get_registers()` and sets `vcpu_dirty`.
- `kvm_cpu_synchronize_post_reset()` puts with `KVM_PUT_RESET_STATE` and clears `vcpu_dirty`; `kvm_cpu_synchronize_post_init()` puts with `KVM_PUT_FULL_STATE` and clears it; `kvm_cpu_synchronize_pre_loadvm()` sets it without fetching, since incoming migration will overwrite everything.
- The run loop, if `vcpu_dirty`, puts with `KVM_PUT_RUNTIME_STATE` and clears it.

The three levels exist because some state must not be written on every run. On x86, `kvm_arch_put_registers()` writes the TSC, kvmclock, and a set of MSRs only at `KVM_PUT_RESET_STATE` or above, and certain feature MSRs only at `KVM_PUT_FULL_STATE`, since writing them at runtime would perturb the guest (rewriting the TSC after every `info registers` would make guest time jump). HVF, WHPX, MSHV and NVMM use the same `vcpu_dirty` flag with the same meaning, as their `*_cpu_synchronize_*` functions show.

ruvm keeps the three put levels with the same per-level register content (the tables live with the per-target code in document 09) and changes one thing: dirtiness and validity are tracked per `RegSet` class instead of for the whole vCPU. A consumer asks for what it needs:

```rust
impl CpuHandle {
    /// QEMU's cpu_synchronize_state(): everything. Used by the monitor,
    /// gdbstub register dumps, migration, and any caller that does not say.
    pub fn synchronize_state(&self) { self.synchronize(RegSet::ALL) }

    /// Only what is named. Runs on the vCPU thread; if called there, no queueing.
    pub fn synchronize(&self, set: RegSet) { /* run_on_cpu if not current thread */ }
}

struct SyncState { valid: RegSet, dirty: RegSet }
```

Getting a class sets it valid; writing through `ArchState` marks it dirty; the run loop puts `dirty` at the runtime level and clears `valid` for classes the guest can change. For KVM x86 this turns an HVF-style or gdbstub single register read into one `KVM_GET_REGS` instead of QEMU's full `kvm_arch_get_registers()`, which issues about ten ioctls (regs, xsave, xcrs, sregs, MSRs, MP state, LAPIC, vCPU events, debug registers, nested state). `synchronize_state()` stays the default so that ported code keeps QEMU's semantics, and the subset path is used only by callers that are audited to need less: the x86 instruction emulator on HVF, WHPX and MSHV (GPRs, segment registers, CR0/CR4/EFER), the MMIO decoder on Arm HVF (the data abort syndrome names one register), the gdbstub `g` packet (GPRs plus PC and flags), and hypercall handlers.

`query-cpus-fast` does not sync at all in QEMU, which is the point of it, and ruvm does not either; `info registers`, `query-cpus` (removed from QEMU, not implemented) and `dump-guest-memory` sync fully. Protected vCPUs return an error from `get_state` for encrypted classes, and consumers that print registers print what QEMU prints for them.

## Pause, resume and run_on_cpu

`pause_all_vcpus()` sets `stop` on every vCPU, kicks them, and waits until each reports stopped. There is no BQL to drop while waiting, but the rule from document 03 applies: the caller must hold no device domain, only the control lock. `run_on_cpu(cpu, f)` enqueues `f` on the vCPU's work queue, kicks it, and waits; if called on the vCPU's own thread it runs `f` directly, like QEMU's `do_run_on_cpu()`. `async_run_on_cpu` does not wait. `async_safe_run_on_cpu` (used by the JIT for TB flushes) runs `f` when all vCPUs are outside their run loops, using the exclusive section from `cpu_exec_start()` and `start_exclusive()` in cpu-common.c.

## KVM: initialization and capabilities

`init_machine` opens `/dev/kvm` (or the `device` property, which QEMU 11.1 accepts for `-accel kvm,device=/path`), checks `KVM_GET_API_VERSION` equals 12, determines the VM type (on x86, `KVM_X86_DEFAULT_VM`, `KVM_X86_SEV_VM`, `KVM_X86_SEV_ES_VM`, `KVM_X86_SNP_VM` or `KVM_X86_TDX_VM`, chosen by the confidential-guest-support object as in `kvm_arch_get_default_type()` and target/i386/sev.c), and calls `KVM_CREATE_VM`, retrying on `EINTR`. Then the required capability check, which ruvm copies with the same list and the same error text (`kvm does not support %s`):

- Generic, from `kvm_required_capabilites[]` in kvm-all.c: `KVM_CAP_USER_MEMORY`, `KVM_CAP_DESTROY_MEMORY_REGION_WORKS`, `KVM_CAP_JOIN_MEMORY_REGIONS_WORKS`, `KVM_CAP_INTERNAL_ERROR_DATA`, `KVM_CAP_IOEVENTFD`, `KVM_CAP_IOEVENTFD_ANY_LENGTH`.
- x86, from `kvm_arch_required_capabilities[]` in target/i386/kvm/kvm.c: `KVM_CAP_SET_TSS_ADDR`, `KVM_CAP_EXT_CPUID`, `KVM_CAP_MP_STATE`, `KVM_CAP_SIGNAL_MSI`, `KVM_CAP_IRQ_ROUTING`, `KVM_CAP_DEBUGREGS`, `KVM_CAP_XSAVE`, `KVM_CAP_VCPU_EVENTS`, `KVM_CAP_X86_ROBUST_SINGLESTEP`, `KVM_CAP_MCE`.
- ruvm adds `KVM_CAP_IMMEDIATE_EXIT` and `KVM_CAP_IRQFD` to the generic list. The second is effectively required by QEMU too, since `do_kvm_irqchip_create()` exits with `kvm: irqfd not implemented` when it is missing.

Optional capabilities are probed and recorded in `AccelCaps`: `KVM_CAP_NR_MEMSLOTS`, `KVM_CAP_MULTI_ADDRESS_SPACE` (x86 SMM uses address space 1 for the `kvm-smram` view, `X86ASIdx_SMM`), `KVM_CAP_READONLY_MEM`, `KVM_CAP_COALESCED_MMIO` and `KVM_CAP_COALESCED_PIO`, `KVM_CAP_MANUAL_DIRTY_LOG_PROTECT2`, the dirty ring caps, `KVM_CAP_USER_MEMORY2`, `KVM_CAP_GUEST_MEMFD`, `KVM_CAP_MEMORY_ATTRIBUTES`, `KVM_CAP_PRE_FAULT_MEMORY`, `KVM_CAP_IRQFD_RESAMPLE`, `KVM_CAP_NESTED_STATE`, `KVM_CAP_BINARY_STATS_FD`, `KVM_CAP_SET_GUEST_DEBUG` and the per-arch set. ruvm keeps QEMU's accelerator properties: `kernel-irqchip=on|off|split`, `kvm-shadow-mem`, `dirty-ring-size`, `device`, and on x86 `notify-vmexit`, `notify-window`, `xen-version`, `xen-gnttab-max-frames`, `xen-evtchn-max-pirq`, `rapl`, `rapl-helper-socket` and `honor-guest-pat`, and on Arm `eager-split-size`.

Memory slots are handled by the listener in document 05: one `KVM_SET_USER_MEMORY_REGION2` (or `KVM_SET_USER_MEMORY_REGION` without `KVM_CAP_USER_MEMORY2`) per changed slot. The kernel has no batched slot ioctl. What ruvm does differently is when the calls happen: at machine creation, the whole initial memory map is committed in one transaction before any vCPU exists, so there is no delete and re-add churn as devices realize one by one, and slot registration for guest RAM does not touch the pages (no prefault, no `KVM_PRE_FAULT_MEMORY`) unless `prealloc=on` or a confidential guest needs it. This is the "slots registered in one batch" item in document 21.

## KVM: interrupt controllers and routing

`kernel-irqchip` selects where the interrupt controller lives. `on` puts the whole chip in the kernel (x86: PIC, IOAPIC and LAPIC via `KVM_CREATE_IRQCHIP`; Arm: vGICv3 via `KVM_CREATE_DEVICE` with `KVM_DEV_TYPE_ARM_VGIC_V3`, plus the ITS; RISC-V: the AIA device `KVM_DEV_TYPE_RISCV_AIA`; s390x and PowerPC: their own devices). `off` keeps everything in userspace. `split` (x86 only) keeps the LAPIC in the kernel and puts the IOAPIC, PIC and PIT in userspace, enabled with `KVM_CAP_SPLIT_IRQCHIP`; level-triggered EOIs that need the IOAPIC come back as `KVM_EXIT_IOAPIC_EOI`. Arm rejects split with `-machine kernel_irqchip=split is not supported on ARM.` (target/arm/kvm.c), which ruvm reproduces. When the property is not given, the machine class default decides (`default_kernel_irqchip_split` in `MachineClass`), and some guests force it: TDX requires split and fails with `TDX VM requires kernel_irqchip to be split` (target/i386/kvm/tdx.c), and SEV-ES requires an in-kernel irqchip. RISC-V adds the `riscv-aia` property (`emul`, `hwaccel`, `auto`, default `auto`) in target/riscv/kvm/kvm-cpu.c for the IMSIC mode.

Routing uses a GSI table. `KVM_CAP_IRQ_ROUTING` returns the table size; QEMU keeps a bitmap of used GSIs and a `kvm_irq_routing` array and replaces the whole table with `KVM_SET_GSI_ROUTING` on commit (`kvm_irqchip_commit_routes()`). MSI vectors that are delivered through an irqfd need a route (`kvm_irqchip_add_msi_route()`); MSIs raised by userspace devices without an irqfd go straight through `KVM_SIGNAL_MSI` (`kvm_irqchip_send_msi()`), which QEMU requires on x86. ruvm keeps both, with the route change guard described above batching commits across a whole MSI-X table write.

irqfd connects an eventfd to a GSI (`KVM_IRQFD`) so that a vhost worker, a VFIO interrupt, or a ruvm iothread can inject without a syscall into KVM from the device model's point of view (the eventfd write is the syscall). Level-triggered lines use a resample fd (`KVM_IRQFD_FLAG_RESAMPLE`, `KVM_CAP_IRQFD_RESAMPLE`) that KVM signals on EOI so the device can re-assert. In ruvm, an `IrqLine` (document 12) whose sink is a KVM GSI and whose source is a reactor-owned device holds an irqfd and writes it; there is no lock on that path.

ioeventfd goes the other way: `KVM_IOEVENTFD` binds (address, length, optional datamatch, PIO or MMIO) to an eventfd so a guest write completes in the kernel and signals the fd. Document 05 describes how the listener diffs ioeventfd sets; the KVM backend implements `IoEventOps` directly with `KVM_IOEVENTFD`, including zero-length MMIO eventfds (`KVM_CAP_IOEVENTFD_ANY_LENGTH`, needed for virtio-mmio and virtio-pci notify with `ioeventfd=on`).

Coalesced MMIO (`KVM_REGISTER_COALESCED_MMIO`) buffers writes to registered ranges in a ring shared with userspace. QEMU drains the ring (`kvm_flush_coalesced_mmio_buffer()`) before dispatching to any region marked `flush_coalesced_mmio`; ruvm does the same, and checks head and tail in the mapped page first so an empty ring costs two loads.

## KVM: dirty ring versus dirty log

The KVM backend implements `DirtyLog` in two modes. The data structures and bitmap merging are in document 05; this section covers the accelerator side.

Bitmap mode is the default, as in QEMU. With `KVM_CAP_MANUAL_DIRTY_LOG_PROTECT2` enabled (with `KVM_DIRTY_LOG_INITIALLY_SET` when offered), `KVM_GET_DIRTY_LOG` returns the bitmap without re-protecting and `KVM_CLEAR_DIRTY_LOG` re-protects in 64-page aligned chunks just before the pages are sent. The cost of a sync is proportional to guest memory size, not to the number of dirty pages.

Ring mode is enabled with `-accel kvm,dirty-ring-size=N`, where N is the number of entries per vCPU, a power of two. The kernel documentation recommends at least 4096 entries (64 KiB per vCPU, since `struct kvm_dirty_gfn` is 16 bytes). It must be enabled with `KVM_ENABLE_CAP` before any vCPU is created; ruvm checks this ordering in `init_machine`. `KVM_CAP_DIRTY_LOG_RING_ACQ_REL` is used on hosts with weak memory ordering (arm64), since only that variant is valid there. With `KVM_CAP_DIRTY_LOG_RING_WITH_BITMAP` the backup bitmap is read at the final sync. QEMU runs a reaper thread named `kvm-reaper` that wakes every second (a `sleep(1)` with a TODO about a smarter timeout in `kvm_dirty_ring_reaper_thread()`), takes the BQL and reaps all rings. ruvm keeps a reaper thread with the same name, and makes two changes. The reaper holds no global lock: each vCPU's ring is read with acquire loads of `flags`, harvested into the per-RamBlock bitmap with atomic ORs, then `KVM_RESET_DIRTY_RINGS` is issued once per pass. The wake interval adapts: after a pass that found a ring more than half full, it wakes again after 10 ms; otherwise it doubles its sleep up to one second. On `KVM_EXIT_DIRTY_RING_FULL`, the exiting vCPU reaps only its own ring when the dirty limit feature is active (`dirtylimit_in_service()`) and all rings otherwise, as QEMU does, then applies the dirty-limit sleep. The `calc-dirty-rate` and `set-vcpu-dirty-limit` QMP commands depend on that behavior.

The ring suits large guests with sparse writes (sync cost follows dirty pages); the bitmap suits write-heavy guests (a full ring forces an exit per vCPU). Migration (document 17) uses whichever is configured.

## KVM x86: CPUID, MSRs and exits

CPUID: at vCPU creation ruvm calls `KVM_GET_SUPPORTED_CPUID` once per VM (cached, as `kvm_arch_get_supported_cpuid()` caches), intersects it with the CPU model and `-cpu` flags using the tables and filtering rules in document 09 (the same as `x86_cpu_filter_features()`, with the same warnings for unavailable features), and sets it with `KVM_SET_CPUID2` before the first `KVM_RUN`. The kernel documents that changing CPUID after `KVM_RUN` "may cause guest instability", and ruvm refuses to do it: CPU hotplug creates the new vCPU with the same CPUID as its siblings, modulo APIC ID and topology leaves. Hyper-V enlightenments (`hv-*` flags) use `KVM_GET_SUPPORTED_HV_CPUID` when `KVM_CAP_HYPERV_CPUID` is available.

MSRs: the MSR list comes from `KVM_GET_MSR_INDEX_LIST` and `KVM_GET_MSR_FEATURE_INDEX_LIST`; feature MSRs (for example `IA32_ARCH_CAPABILITIES` and the VMX capability MSRs for nested) are filtered like CPUID. QEMU enables `KVM_CAP_X86_USER_SPACE_MSR` with `KVM_MSR_EXIT_REASON_FILTER` and installs a filter (`kvm_filter_msr()`) for MSRs it emulates in userspace: `MSR_CORE_THREAD_COUNT`, and with `rapl=on` the RAPL MSRs `MSR_RAPL_POWER_UNIT`, `MSR_PKG_POWER_LIMIT`, `MSR_PKG_POWER_INFO` and `MSR_PKG_ENERGY_STATUS`. Accesses come back as `KVM_EXIT_X86_RDMSR` and `KVM_EXIT_X86_WRMSR` and are completed by writing `data` and `error` in the run page. ruvm installs the same filter with the same handlers.

Other x86 VM-wide settings mirror `kvm_arch_init()`: `KVM_SET_TSS_ADDR` and the identity map address, `KVM_CAP_EXCEPTION_PAYLOAD`, `KVM_CAP_X86_TRIPLE_FAULT_EVENT`, `KVM_CAP_X2APIC_API`, the PMU capability (`KVM_CAP_PMU_CAPABILITY` to disable the vPMU when `-cpu ...,pmu=off`), `KVM_CAP_X86_NOTIFY_VMEXIT` from `notify-vmexit`, bus lock detection, and `KVM_CAP_X86_DISABLE_EXITS` with MWAIT, HLT, PAUSE and CSTATE when `-overcommit cpu-pm=on` is given. The SMM address space, `KVM_SMI`, and `KVM_EXIT_HYPERCALL` for `KVM_HC_MAP_GPA_RANGE` (SEV page state changes) are handled in ruvm-target-x86.

Nested virtualization: VMX or SVM is exposed through CPUID and the capability MSRs (document 09). Nested state is saved and restored with `KVM_GET_NESTED_STATE` and `KVM_SET_NESTED_STATE` when `KVM_CAP_NESTED_STATE` is present; the buffer size comes from `kvm_max_nested_state_length()` and QEMU allocates `env->nested_state` only when it is non-zero, as ruvm does. The `NESTED` register class is part of `RegSet::ALL` but is fetched only when the vCPU has VMX or SVM enabled.

## KVM: confidential guests

The confidential computing model is in document 19; the accelerator side is:

- SEV and SEV-ES: `KVM_MEMORY_ENCRYPT_OP` with `KVM_SEV_INIT` or `KVM_SEV_ES_INIT` on legacy VM types, and the `KVM_SEV_LAUNCH_*` sequence (`START`, `UPDATE_DATA`, `UPDATE_VMSA` for ES, `MEASURE`, `SECRET`, `FINISH`), as target/i386/sev.c.
- SEV-SNP: VM type `KVM_X86_SNP_VM`, `KVM_SEV_SNP_LAUNCH_START`, `KVM_SEV_SNP_LAUNCH_UPDATE` with page types normal, zero, unmeasured, secrets and CPUID, and `KVM_SEV_SNP_LAUNCH_FINISH`.
- TDX: VM type `KVM_X86_TDX_VM`, `KVM_TDX_CAPABILITIES`, `KVM_TDX_INIT_VM`, `KVM_TDX_INIT_VCPU`, `KVM_TDX_INIT_MEM_REGION`, `KVM_TDX_FINALIZE_VM`, and `KVM_EXIT_TDX` for TDVMCALLs that KVM forwards (GetQuote is serviced through the quote generation socket, `tdx_handle_get_quote()`).
- All of them with guest_memfd: private memory, `KVM_EXIT_MEMORY_FAULT` driving conversions, and `vcpu.state_protected()` true after launch finish (after SEV-ES VMSA encryption, or TDX finalize), so lazy sync never tries to read encrypted state.
- Reset (QEMU 11.0): SEV-SNP and TDX guests cannot have their state reset in place. QEMU 11.0 recreates the KVM VM and rebinds vCPUs to new vCPU fds on a system reset. In ruvm, `Accel::rebuild_vm()` does the same: every `Vcpu` is dropped on its thread and recreated from the new VM fd inside the reset hold phase, and memory slots are replayed by re-registering the listener.

## KVM on Arm, RISC-V, s390x, PowerPC and LoongArch

Arm (target/arm/kvm.c): vCPU init with `KVM_ARM_PREFERRED_TARGET` and `KVM_ARM_VCPU_INIT` feature bits for PSCI 0.2, PMUv3, SVE (then `KVM_ARM_VCPU_FINALIZE`), pointer authentication, and `KVM_ARM_VCPU_HAS_EL2` for nested (QEMU 10.1, with `-machine virt,virtualization=on`); `hw/arm/virt.c` rejects nested unless the in-kernel GICv3 is used (`KVM EL2 is only supported with in-kernel GICv3`). The IPA size comes from `KVM_CAP_ARM_VM_IPA_SIZE` and is encoded in the `KVM_CREATE_VM` type. Registers are synced with `KVM_GET_ONE_REG` and `KVM_SET_ONE_REG` over the list from `KVM_GET_REG_LIST`, which is where the per-class `RegSet` saves the most, since a full Arm sync costs one ioctl per register. Other caps used: `KVM_CAP_ARM_MTE`, `KVM_CAP_ARM_EL1_32BIT`, `KVM_CAP_ARM_NISV_TO_USER` (data aborts without valid syndrome delivered to userspace), `KVM_CAP_ARM_INJECT_SERROR_ESR`, `KVM_CAP_ARM_INJECT_EXT_DABT`, `KVM_CAP_ARM_IRQ_LINE_LAYOUT_2` and `KVM_CAP_ARM_EAGER_SPLIT_CHUNK_SIZE`. PSCI is handled in the kernel and reaches userspace as `KVM_EXIT_SYSTEM_EVENT`.

RISC-V (target/riscv/kvm/kvm-cpu.c): ISA extensions are probed and set through ONE_REG, QEMU 11.1 added Zicbop and BFloat16; the AIA irqchip modes are above. s390x uses the SIE intercept exits (`KVM_EXIT_S390_SIEIC` and friends) handled in target/s390x/kvm, the FLIC device, and QEMU 11.1's ASTFLE facility 2 for nested. PowerPC KVM (HV and PR) and LoongArch KVM are ported as they are in QEMU, behind the same trait, in milestone M10.

## HVF

Hypervisor.framework has two unrelated implementations sharing accel/hvf/hvf-all.c and accel/hvf/hvf-accel-ops.c.

Arm (target/arm/hvf/hvf.c) is the one that matters. `hv_vm_create` with a config that sets the IPA size (clamped to what `hv_vm_config_get_max_ipa_size` allows and reflected into `ID_AA64MMFR0_EL1.PARange`), `hv_vcpu_create`, and `hv_vcpu_run`, which returns with `HV_EXIT_REASON_EXCEPTION`, `HV_EXIT_REASON_VTIMER_ACTIVATED` or `HV_EXIT_REASON_CANCELED`. Exceptions are decoded from the syndrome: data aborts (`EC_DATAABORT`) become MMIO or, for write-protected RAM, dirty tracking; system register traps (`EC_SYSTEMREGISTERTRAP`) go to the userspace sysreg emulation; `EC_WFX_TRAP` implements WFI in userspace (`hvf_wfi()`), sleeping until the vtimer deadline or a kick; HVC and SMC go to the PSCI implementation in userspace. The vtimer is a QEMU concern under HVF: the framework delivers `VTIMER_ACTIVATED` only inside `hv_vcpu_run`, so ruvm, like `hvf_sync_vtimer()`, masks and unmasks it and raises the timer PPI through the GIC.

QEMU 11.1 added two features by Mohamed Mediouni. The in-kernel vGIC, created with `hv_gic_create()` with the distributor at 0x08000000 and redistributors at 0x080A0000 (the virt machine's GICv3 layout), available from macOS 15; and nested virtualization with `hv_vm_config_set_el2_enabled()`, gated by `hv_vm_config_get_el2_supported()`, which requires macOS 15 and an M3 or later, enabled with `-machine virt,virtualization=on`, and requiring the in-kernel vGIC. The guest gets nVHE EL2 only. For the `virt-11.1` machine type under HVF, the in-kernel vGIC is the default (`get_kernel_irqchip_default()` in hw/arm/virt.c returns true unless `hvf_no_kernel_irqchip_default` is set, which `virt-11.0` and older set). ruvm reproduces this compat default exactly, because it changes the migration stream and the guest-visible GIC implementation. SME2 register state (Z, P, ZA and ZT0) is synchronized when macOS 15.2 or later reports SME support (QEMU 11.0); ruvm disables SME for nested guests as QEMU does. `SME` and `SVE` register classes are separate in `RegSet`, so a normal MMIO exit does not read ZA.

x86 HVF (target/i386/hvf) runs on Intel Macs, uses `hv_vcpu_run_until(HV_DEADLINE_FOREVER)`, VMCS field access, and the shared x86 emulator for MMIO and port I/O. It is maintained for parity and gets no new features.

Neither HVF variant has ioeventfd, irqfd or MSI routing; virtio notify writes are MMIO exits handled by the lockless notify path from document 03. Dirty tracking uses `hv_vm_protect()` to drop write permission on logged RAM; the resulting write fault marks the page dirty and restores write permission for that page (`hvf_unprotect_dirty_range()`), so HVF supports migration and VGA dirty tracking at 4 KiB fault cost per first write.

## WHPX

WHPX (accel/whpx/whpx-common.c, target/i386/whpx/whpx-all.c, and an Arm port in target/arm/whpx) uses `WHvCreatePartition`, partition properties set before `WHvSetupPartition`, `WHvMapGpaRange` for memory, and `WHvRunVirtualProcessor`. x86 exit reasons handled: `MemoryAccess`, `X64IoPortAccess`, `X64InterruptWindow`, `X64ApicEoi`, `X64Halt`, `X64MsrAccess`, `X64Cpuid`, `Exception`, `Canceled`, `UnrecoverableException`, `InvalidVpRegisterValue` and `UnsupportedFeature`. MMIO and port I/O are completed by the shared x86 emulator. Properties: `kernel-irqchip` (on, off, split) and `hyperv` (OnOffAuto). The in-kernel irqchip uses the platform's local APIC emulation (`WHvPartitionPropertyCodeLocalApicEmulationMode`), and nested virtualization is enabled only when the in-kernel irqchip is on and the host reports `NestedVirtSupport`. The Arm port always uses the platform vGICv3.

WHPX has no dirty tracking: its `log_sync` marks the whole section dirty (enough for display refresh) and a migration blocker is registered with QEMU's message (`State blocked due to missing dirty memory tracking support,And some system register/state save-restore`, including the missing space). ruvm reproduces the blocker and the message.

## MSHV

MSHV (accel/mshv, target/i386/mshv, QEMU 10.2) drives the Microsoft Hypervisor from a Linux root partition through `/dev/mshv` (Linux 6.15). It creates a partition with `MSHV_CREATE_PARTITION` and `MSHV_INITIALIZE_PARTITION`, issues hypercalls with `MSHV_ROOT_HVCALL`, maps memory, registers ioeventfds with `MSHV_IOEVENTFD`, handles interrupts and MSI routing in accel/mshv/irq.c, and runs vCPUs with `MSHV_RUN_VP`. Exits are Hyper-V intercept messages; MMIO and port I/O use the shared x86 emulator; the synthetic interrupt controller support is in target/i386/mshv/synic.c. The kick is SIG_IPI with a no-op handler: the kernel sees the pending signal and returns from `MSHV_RUN_VP`. There is no dirty logging in accel/mshv, so ruvm registers a migration blocker. An ARM64 MSHV series was at v5 in July 2026 and is not merged; ruvm tracks it in document 25. Milestone: M10.

## NVMM

NVMM (target/i386/nvmm, NetBSD) uses libnvmm: `nvmm_machine_create`, `nvmm_vcpu_create`, `nvmm_gpa_map` and `nvmm_hva_map`, `nvmm_vcpu_run`, and `nvmm_vcpu_stop` from the SIG_IPI handler as the kick. Exits handled: `MEMORY`, `IO`, `INT_READY`, `NMI_READY`, `TPR_CHANGED`, `HALTED`, `SHUTDOWN`, `RDMSR`, `WRMSR`, `MONITOR` and `MWAIT`. MMIO and port I/O use `nvmm_assist_mem` and `nvmm_assist_io` with callbacks into ruvm's dispatch. The APIC is always in userspace. QEMU registers a migration blocker, and ruvm does too. Milestone: M10.

## Xen

Two unrelated things carry the name.

Real Xen (accel/xen/xen-all.c, hw/xen): QEMU runs as a device model for a domain that Xen schedules. There are no vCPU run loops in QEMU; `runs_vcpus()` is false and the vCPU threads are dummies. QEMU registers an ioreq server, receives port I/O and MMIO requests on event channels, handles them with the normal memory dispatch, and completes them through the shared ioreq page (hw/xen/xen-hvm-common.c). Guest memory is not mapped as a whole: xen-mapcache.c maps buckets of guest frames on demand and invalidates them on request, so `GuestMemory` for Xen has a slow path that maps a bucket, and `GuestPtr` lifetimes (document 05) pin the bucket. PV backends (block, net, console, 9pfs) live on the xen-bus (hw/xen/xen-bus.c) and talk to frontends through grant tables and xenstore; the `xenpv` and `xenpvh` machines host them. PCI passthrough (`xen_pt`) goes through the hypervisor. The accelerator property is `igd-passthru`, and the Xen accelerator also sets global migration properties (`store-global-state`, `send-configuration` and `send-section-footer` off) that ruvm must set identically. hw/xen/xen-operations.c abstracts event channel, grant table, foreign memory and xenstore operations so the same backends work on real Xen and on the emulation below; ruvm keeps this abstraction as a trait, `XenOps`.

Xen emulation under KVM (hw/i386/kvm/xen_*.c): KVM runs an unmodified Xen HVM guest, with KVM handling hypercall pages, event channel delivery, shared info and runstate areas in the kernel (`KVM_XEN_HVM_CONFIG`, `KVM_XEN_HVM_SET_ATTR`, `KVM_XEN_VCPU_SET_ATTR`) and QEMU emulating the rest: event channels (xen_evtchn.c), grant tables (xen_gnttab.c), xenstore (xen_xenstore.c), overlay pages (xen_overlay.c) and the primary console (xen_primary_console.c). Hypercalls KVM does not handle arrive as `KVM_EXIT_XEN`. The invocation is `-accel kvm,xen-version=0x40011,kernel-irqchip=split`; `xen-gnttab-max-frames` defaults to 64 and `xen-evtchn-max-pirq` to 256. The kernel requirements per QEMU's documentation are Linux 5.12 for the base, 5.17 for PIRQ acceleration and 5.19 for PV timers and IPIs. The same PV backends then run inside the same process through `XenOps`. ruvm ports both: emulation first (it rides on the KVM backend and is testable in CI without a Xen host), real Xen in M10.

## Nitro Enclaves

accel/nitro/nitro-accel.c (QEMU 11.0, by Alexander Graf) drives `/dev/nitro_enclaves` on an EC2 parent instance: `NE_CREATE_VM`, `NE_SET_USER_MEMORY_REGION` for each RAM region, `NE_ADD_VCPU` per vCPU, then `NE_START_ENCLAVE` with the enclave CID. The hypervisor runs the enclave on cores donated by the parent; QEMU never runs guest code and never sees exits, so vCPU threads are dummies. Properties are `debug-mode` and `enclave-cid`, the image is an EIF file, and the machine is the dedicated nitro enclaves machine. ruvm ports the driver interface and error mapping (the `NE_ERR_*` codes to QEMU's messages). Milestone: M10.

## qtest

accel/qtest/qtest.c provides an accelerator whose vCPUs never run; time advances only through the qtest protocol (`clock_step`, `clock_set`), and tests read and write guest memory and I/O through the protocol. ruvm-accel-qtest implements it at M1 because every later milestone is tested through qtest (document 23). The virtual clock under qtest is owned by the protocol server, so `QEMU_CLOCK_VIRTUAL` timers fire exactly when the test steps, which is what makes the qtest suites deterministic.

## The JIT as an accelerator

`-accel tcg` selects ruvm-jit, keeping the name `tcg` and the `tcg-accel` type for compatibility. Properties: `thread=single|multi`, `tb-size`, `split-wx`, `one-insn-per-tb`. MTTCG is always on in ruvm (canon), so `thread=single` is accepted and selects the serialized mode used for icount and record/replay (document 03). The JIT's `Vcpu::run` executes translated blocks until an exception, interrupt, or exit request; its kick is the `icount_decr` flag write checked at block entry, which costs no syscall. Its `get_state` and `put_state` are no-ops because the JIT's `ArchState` is the state. Dirty tracking is the `DIRTY_MEMORY_CODE` and `DIRTY_MEMORY_MIGRATION` bitmaps written by the softmmu store path (document 05). Details are in documents 07 and 08.

## Performance

### Exit latency budget

The canon targets are about startup; the exit targets are in document 21 (area 4): a userspace PIO exit with no device at a median of at most 0.85 of QEMU's cycles, and aggregate exits per second at least 4 times QEMU's with 16 vCPUs hitting different devices. The budget for the userspace part of an exit, from `KVM_RUN` return to the next `KVM_RUN`, is structural:

1. Zero syscalls besides `KVM_RUN` itself for PIO, MMIO to a lockless or uncontended region, `IRQ_WINDOW_OPEN`, and MSR filter exits. QEMU meets this except for coalesced MMIO flushes; ruvm checks the coalesced ring's head and tail in the mmapped page and skips the flush call when it is empty.
2. Zero heap allocations. `VcpuExit` borrows the run page; `AccessCtx` is on the stack; the dispatch lookup is a read of an RCU-protected array (document 05).
3. At most one lock, the target region's domain, uncontended in the common case. QEMU takes the BQL here.
4. No shared cache line written except the device's own state. RCU read sections use thread-local epochs (document 03). Statistics counters are per vCPU.
5. No register sync. `post_run` reads only what the run page provides.

The `vmexit` harness in document 21 enforces the ratio in CI; disabled trace points cost one static-key branch.

### No main thread round trips

In QEMU several vCPU paths need the main loop or the BQL: `KVM_EXIT_DIRTY_RING_FULL` takes the BQL to reap, MCE injection after SIGBUS takes the BQL, `KVM_EXIT_SYSTEM_EVENT` crash handling takes the BQL, and every MMIO to a normal region takes the BQL. Guest-initiated reset and shutdown are posted to the main loop and then pause every vCPU. ruvm handles every exit on the vCPU thread with at most one domain lock. Actions that must be global (reset, shutdown, panic, stop on internal error) are posted to the main thread as flags, exactly like QEMU's `qemu_system_*_request()` functions, and the vCPU returns to its loop and parks on the pause that follows; it never waits synchronously for the main thread. Monitor commands that need vCPU state go the other direction through `run_on_cpu`, and the `RegSet` subsets keep that cheap.

### Pinning

QEMU has no vCPU pinning of its own: libvirt pins through `query-cpus-fast` thread ids and cgroups. ruvm is compatible with that and adds an extension for direct users, `-accel kvm,x-vcpu-affinity=0-3:8-11`, a list of CPU sets indexed by vCPU (one set applies to all). The `x-` prefix follows the QEMU convention for unstable properties and is listed as a deviation in document 02. The same `thread-context` objects that bind preallocation threads (document 05) can be referenced with `x-vcpu-thread-context=tc0` so vCPUs, preallocation and NUMA memory binding agree. Affinity is applied by the vCPU thread itself before it enters the guest, so there is no window where the thread runs elsewhere. With `-overcommit cpu-pm=on` and pinned vCPUs, HLT and MWAIT exits disappear on x86 as in QEMU;

## Feature support per accelerator

Rows are backends as ruvm implements them, matching QEMU 11.1 behavior. "User" means the component is emulated in ruvm userspace.

| Accelerator | Hosts | Irqchip modes | ioeventfd | irqfd and MSI routing | Dirty tracking | Nested | Confidential | Migration | ruvm milestone |
|---|---|---|---|---|---|---|---|---|---|
| KVM x86 | Linux | on, off, split | yes | yes | bitmap or ring | VMX, SVM | SEV, SEV-ES, SEV-SNP, TDX | yes | M2 (CoCo M8) |
| KVM Arm | Linux | on (vGICv3, ITS), off | yes | yes | bitmap or ring (ACQ_REL, with bitmap) | EL2, in-kernel GIC only | none in 11.1 | yes | M6 |
| KVM RISC-V | Linux | AIA emul, hwaccel, auto; off | yes | yes | bitmap | no | none | yes | M6 |
| KVM s390x, PowerPC, LoongArch | Linux | arch devices | yes | yes | bitmap | s390x ASTFLE 2; PowerPC HV | s390x PV (document 19) | yes | M10 |
| HVF Arm | macOS | vGIC (macOS 15, default on virt-11.1) or user | no | no | write protect | EL2 nVHE (macOS 15, M3+) | none | no blocker in QEMU | M6 |
| HVF x86 | macOS Intel | user | no | no | write protect | no | none | yes, invtsc blocks | M6 |
| WHPX x86 | Windows | on, off, split | no | no | none | with in-kernel irqchip | none | blocked | M6 |
| WHPX Arm | Windows | platform vGICv3 | no | no | none | no | none | blocked | M10 |
| MSHV x86 | Linux root partition | in hypervisor | yes | yes | none | no | none | blocked | M10 |
| NVMM | NetBSD | user | no | no | none | no | none | blocked | M10 |
| Xen | Linux dom0 | in Xen | n/a (ioreq) | event channels | via Xen | Xen's | Xen's | via Xen | M10 |
| Xen on KVM | Linux | split required | yes | yes | as KVM x86 | as KVM | no | yes | M10 |
| nitro | Linux EC2 parent | n/a | n/a | n/a | n/a | n/a | enclave | no | M10 |
| qtest | all | user | no | no | none | n/a | n/a | test only | M1 |
| JIT (tcg) | all | user | emulated | emulated | bitmaps | per target (for example Arm EL2, x86 SVM) | none | yes | M4 |

Document 09 owns the s390x protected virtualization and PowerPC nested details.

## Failure modes

- Missing required capability: fail at `init_machine` with QEMU's text, then try the next `-accel` in the list.
- `KVM_CREATE_VM` returning `EINVAL` for an IPA size on Arm: QEMU's message about the requested IPA size, then fail.
- vCPU count above `KVM_CAP_MAX_VCPUS`: QEMU's `Number of %s cpus requested (%d) exceeds the maximum cpus supported by KVM (%d)` message.
- Slot registration failure: `kvm_set_phys_mem()` prints `%s: error registering slot: %s` and aborts; ruvm does the same, since the guest's view of memory would otherwise be inconsistent.
- `KVM_EXIT_INTERNAL_ERROR` with emulation failure: print the suberror and the instruction bytes as `kvm_handle_internal_error()` does, and stop with `internal-error`, unless the arch handler reports that it recovered.
- Dirty ring enabled with vCPUs already created: impossible by construction because ruvm enables the ring in `init_machine`.
- HVF nested requested on an unsupported Mac: QEMU's error from `hvf_arm_el2_supported()` returning false.
- WHPX: partition creation failure reports the HRESULT in hex (`hr=%08lx`), as QEMU does.
- A kick sent to a vCPU thread that already exited: the `Kicker` holds a weak reference and does nothing.

## Testing

Each backend runs the QEMU qtest suites that the host supports, kvm-unit-tests (x86 and arm) under KVM, and the boot matrix from document 22. The lazy sync change is tested by a differential mode, `x-verify-sync=on`, which after every subset sync also fetches `RegSet::ALL` and asserts that the classes the subset promised valid match. Kick races are tested with a loom model of `Kicker` and the run loop's acquire/release pair, and with a stress test that kicks a vCPU spinning on `IRQ_WINDOW_OPEN` exits a million times and checks that no kick is lost.

## New decisions in this document

- ruvm-accel-kvm requires `KVM_CAP_IMMEDIATE_EXIT` and `KVM_CAP_IRQFD`; no `KVM_SET_SIGNAL_MASK` path.
- The x86 instruction emulator is a module of ruvm-target-x86 shared by HVF x86, WHPX and MSHV; NVMM uses libnvmm's assists.
- Lazy register sync tracks validity and dirtiness per `RegSet` class, with `synchronize_state()` keeping QEMU's all-registers semantics as the default and a verification mode.
- `VcpuExit` borrows the host run structure; no copy of exit data.
- Kicks coalesce on the `exit_request` swap.
- The dirty ring reaper holds no global lock and adapts its wake interval.
- MSI route changes are batched through a guard, one `KVM_SET_GSI_ROUTING` per batch.
- `x-vcpu-affinity` and `x-vcpu-thread-context` accelerator properties for pinning, listed as extensions in document 02.
- SEV-SNP and TDX reset is `Accel::rebuild_vm()` inside the reset hold phase.
- Document 24 should drop MIPS from ruvm-accel-kvm.

## References

- QEMU 11.1.0 source, https://gitlab.com/qemu-project/qemu (tag v11.1.0): accel/kvm/kvm-all.c, target/i386/kvm/kvm.c, target/i386/kvm/tdx.c, target/i386/sev.c, target/arm/kvm.c, hw/arm/virt.c, target/riscv/kvm/kvm-cpu.c, accel/hvf/hvf-all.c, target/arm/hvf/hvf.c, target/i386/hvf, accel/whpx/whpx-common.c, target/i386/whpx/whpx-all.c, accel/mshv/mshv-all.c, target/i386/nvmm/nvmm-all.c, accel/xen/xen-all.c, accel/nitro/nitro-accel.c, accel/qtest/qtest.c.
- QEMU 11.1.0 release announcement (HVF nested virtualization and vGIC, RISC-V KVM Zicbop and BFloat16, s390x ASTFLE facility 2): https://www.qemu.org/2026/08/11/qemu-11-1-0/
- QEMU 11.0.0 release announcement (nitro accelerator, KVM CET, SEV-SNP and TDX reset, HVF SME2): https://www.qemu.org/2026/04/22/qemu-11-0-0/
- QEMU 10.1.0 release announcement (TDX, Arm KVM nested): https://www.qemu.org/2025/08/26/qemu-10-1-0/
- QEMU Xen emulation documentation: https://www.qemu.org/docs/master/system/i386/xen.html
- Linux KVM API (immediate_exit, dirty ring, MSR filtering, KVM_SET_CPUID2 caveats, guest_memfd): https://docs.kernel.org/virt/kvm/api.html
- Apple Hypervisor framework: https://developer.apple.com/documentation/hypervisor
- Windows Hypervisor Platform API: https://learn.microsoft.com/en-us/virtualization/api/hypervisor-platform/hypervisor-platform
- rust-vmm kvm-ioctls and kvm-bindings: https://github.com/rust-vmm/kvm
