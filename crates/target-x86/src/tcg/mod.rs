// SPDX-License-Identifier: GPL-2.0-or-later

//! The x86 TCG front end: the port of QEMU's `target/i386/tcg`, for 64-bit, 32-bit and 16-bit
//! guest code, run by `ruvm-jit`.
//!
//! [`X86`] is the [`CpuOps`] of an x86 vCPU. Its CPU state lives in the runtime's state
//! buffer with the layout in [`env`](mod@env), so generated code reaches every register by
//! offset. The translator (`translate.rs`) decodes guest instructions and emits IR, keeping the
//! arithmetic flags lazy with the `cc_op` scheme of [`cc`]. Helpers called from generated code
//! are in `helpers.rs` (arithmetic, CPUID, MSRs, control registers) and `seg.rs`
//! (descriptors, far transfers, exceptions and interrupts through the IDT, SYSCALL and
//! friends). The page walk used by `tlb_fill` is in `mmu.rs`.
//!
//! What is covered: every general purpose integer instruction in real, protected (16 and 32 bit)
//! and long mode, including the string instructions with REP, the stack instructions (PUSH, POP,
//! PUSHA, POPA, PUSHF, POPF, CALL, RET, ENTER, LEAVE, far CALL, JMP and RET), Jcc, SETcc, CMOVcc,
//! MUL and DIV (with #DE), the bit instructions (BT, BTS, BTR, BTC, BSF, BSR, and LZCNT, TZCNT and
//! POPCNT when the model has them), BSWAP, XCHG, XADD, CMPXCHG, CMPXCHG8B, CMPXCHG16B and LOCK
//! prefixed read-modify-write instructions done with atomic ops, the BCD instructions, CPUID (from
//! the [`crate::cpuid`] model, `qemu64` by default), RDTSC and RDTSCP, RDMSR and WRMSR, IN and OUT,
//! MOV to and from control and debug registers, the descriptor table instructions, LAR, LSL, VERR,
//! VERW, INT, INT3, INTO, INT1, IRET, SYSCALL, SYSRET, SYSENTER, SYSEXIT, HLT, CLI, STI and INVLPG.
//! Exceptions and interrupts are delivered through the IDT in all three modes, with double and
//! triple fault detection. The page walk handles 32-bit paging (with PSE), PAE paging, and 4 and 5
//! level long mode paging, with NX, WP, SMEP and SMAP, and in long mode the protection keys of
//! PKRU (RDPKRU, WRPKRU) for user pages and `IA32_PKRS` for supervisor pages, as QEMU does.
//!
//! The x87 instructions (D8 to DF and FWAIT, `translate/x87.rs`) run on the 80-bit
//! `floatx80` of `ruvm-softfloat`, with FPUC rounding and precision control and the FPUS
//! exception flags, and FXSAVE, FXRSTOR, XSAVE, XRSTOR and XSAVEOPT save and restore the x87,
//! SSE and AVX state (`helpers/fpu.rs`). MMX, SSE and AVX to AVX2 go through the table driven
//! decoder of `translate/sse.rs` and the kernels of `helpers/vec.rs`, including AES-NI, SHA,
//! the SSE4.2 PCMPESTRI, PCMPESTRM, PCMPISTRI and PCMPISTRM, and the AVX2 gathers
//! (`helpers/crypto.rs` ports the AES, SHA and string compare parts of QEMU's `ops_sse.h`).
//! The EVEX prefix raises #UD, as do XSAVEC, XSAVES, XRSTORS and INVPCID, which QEMU's TCG
//! does not offer either; `-cpu max` does not advertise them, nor PCID. The VEX prefix is
//! decoded in `translate/ext.rs`, along with the general purpose register instructions that
//! come with this part of the instruction set: ANDN, BEXTR, BLSI, BLSMSK, BLSR, BZHI, MULX,
//! PDEP, PEXT, RORX, SARX, SHLX and SHRX (VEX class 13, VEX.L must be 0), ADCX and ADOX (with
//! QEMU's `CC_OP_ADCX`, `CC_OP_ADOX` and `CC_OP_ADCOX` carry chaining), MOVBE, CRC32, RDRAND,
//! RDSEED, RDPID, XGETBV, XSETBV, RDFSBASE, RDGSBASE, WRFSBASE, WRGSBASE (CR4.FSGSBASE is
//! tested at run time, as in QEMU), MOVNTI, LDMXCSR and STMXCSR.
//!
//! Deliberate differences from QEMU:
//!
//! - The decoder is hand written rather than generated from QEMU's decode tables, so the
//!   order in which some invalid encodings are rejected may differ. Results and flags follow
//!   QEMU's `emit.c.inc`.
//! - Task switches are not implemented: a task gate in the IDT, a far JMP or CALL to a TSS or
//!   task gate, and IRET with NT set raise #GP with the selector (or 0 for IRET). Call gates
//!   are not implemented either: a far JMP or CALL through one raises #GP. Virtual 8086 mode
//!   is not supported: IRET to a frame with VM set raises #GP(0).
//! - When the TSS is not present or has the wrong type on a stack switch, QEMU aborts with
//!   "invalid tss"; this port raises #TS with the TR selector instead.
//! - A triple fault halts the vCPU (counted by [`X86::triple_faults`]) and asks the
//!   [`X86Platform`] for a system reset; QEMU leaves the vCPU running until the reset comes.
//!   Without a platform the vCPU just stays halted.
//! - The accessed and dirty bits of page table entries are set with a plain read and write of
//!   physical memory, not with a compare and swap, so another vCPU changing the same entry at
//!   the same moment can lose an update.
//! - The TSC counts host ticks since [`X86`] was made, or while the machine runs when it shares
//!   the machine's count, plus `IA32_TSC` writes. As in QEMU's TCG, that is the host TSC on an
//!   x86 host, so the guest TSC runs at the host TSC rate.
//! - Unknown MSRs read as zero and ignore writes instead of raising #GP. Only the MSRs this
//!   front end uses (EFER, STAR, LSTAR, CSTAR, FMASK, FS and GS base, KERNEL_GS_BASE, the
//!   SYSENTER MSRs, TSC, TSC_AUX, PAT, APIC_BASE, MISC_ENABLE, PKRS) are kept.
//! - Debug registers are stored but hardware breakpoints and watchpoints are not armed.
//!   Single stepping with TF works.
//! - SVM, VMX, SMM, MPX, CET, FRED, LAM and nested paging are not modelled.
//! - The local APIC, the PIC and system reset are reached through the [`X86Platform`] the
//!   machine installs with [`X86::set_platform`], standing in for `cpu->apic_state`,
//!   `isa_pic` and `qemu_system_reset_request()`. Without one, hardware interrupts come from
//!   a small vector queue fed by [`X86::raise_irq`], the APIC base MSR and CR8 are plain
//!   registers and the x2APIC MSRs raise #GP, as QEMU does without an APIC.
//! - `x86_cpu_exec_interrupt()` follows QEMU's priority (POLL, SIPI, SMI, NMI, MCE, HARD), but
//!   an SMI is dropped since there is no SMM, and a hardware interrupt for which the PIC and
//!   APIC have no vector (-1) is not delivered, where QEMU would deliver vector -1.
//! - `x86_cpu_exec_halt()` clears `CPU_INTERRUPT_POLL` before polling the APIC rather than
//!   after, since there is no big lock here to keep another thread from raising it in
//!   between.
//! - PAUSE ends the block like any other instruction rather than leaving the execution loop.
//! - A block that ends with a near RET, a near indirect JMP or CALL, or a direct jump to
//!   another page looks the next block up with `lookup_tb_ptr_ic`, passing the linear EIP, so
//!   that a backend can cache the target where the block jumps. QEMU calls
//!   `helper_lookup_tb_ptr` there. Far transfers keep the plain lookup, since they can change
//!   CS and the block flags.
//! - RCL and RCR are computed inline in generated code instead of calling `helper_rcl*` and
//!   `helper_rcr*`; the results and flags are the same. ROL and ROR with an immediate count
//!   go through the same code as a count in CL.
//! - VEX.X and VEX.B are ignored outside 64-bit mode, as on hardware; QEMU copies them into
//!   `rex_x` and `rex_b` in every mode.
//! - RDRAND and RDSEED return values from the host's `RandomState` hasher instead of
//!   `qemu_guest_getrandom()`; like QEMU they always succeed (CF = 1, the other flags 0).
//! - LDMXCSR does not raise #GP for reserved MXCSR bits, as in QEMU.
//! - BSF and BSR with a zero source leave the destination unchanged, as QEMU and real
//!   hardware do; LZCNT needs ABM and TZCNT needs BMI1 in the model (`qemu64` has neither,
//!   so there they decode as BSR and BSF, as on hardware without them).
//!
//! Not translated yet (they raise #UD, and are left for a follow up): RSM, MONITOR and
//! MWAIT, the SVM and VMX instructions, CMPccXADD, the MPX
//! instructions, the AMD `lock mov cr0` alias for CR8, and the PCREL
//! translation mode. I/O breakpoints (`bpt_io`) are not checked. Virtual 8086 mode decodes
//! like real mode with the IOPL checks, but nothing enters it (see above).

pub mod cc;
pub mod env;
mod helpers;
mod mmu;
mod seg;
mod translate;
pub mod user;

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use ruvm_jit::cputlb::tlb_flush;
use ruvm_jit::translate::TbBuild;
use ruvm_jit::{
    Cpu, CpuLoopExit, CpuOps, CpuShared, InterpBackend, Jit, JitConfig, MmuAccessType, Ra, Tb,
    TbCpuState, Vcpu, excp, interrupt, translator_loop,
};
use ruvm_jit_interp::HelperRegistry;
use ruvm_mem::AddressSpace;
use ruvm_sys::hostticks::Ticks;

use crate::cpuid::X86Cpu;
use crate::state::{
    HF_CPL_MASK, HF_CS64_MASK, HF_LMA_MASK, HF_SMM_MASK, HF2_GIF_MASK, HF2_NMI_MASK, R_CS,
    X86CpuState,
};
use env::{
    AC_MASK, CC_OP, DR, EFLAGS, EIP, ENV_SIZE, HF_INHIBIT_IRQ_MASK, HF_SMAP_MASK, HFLAGS, HFLAGS2,
    IF_MASK, IOPL_MASK, RF_MASK, SEG_BASE, SEG_SELECTOR, STAR, TF_MASK, TSC_OFFSET, VM_MASK, dr,
    ld32, ld64, seg, st32, st64,
};

/// Divide error.
pub const EXCP00_DIVZ: i32 = 0;
/// Debug.
pub const EXCP01_DB: i32 = 1;
/// NMI.
pub const EXCP02_NMI: i32 = 2;
/// Breakpoint.
pub const EXCP03_INT3: i32 = 3;
/// Overflow.
pub const EXCP04_INTO: i32 = 4;
/// BOUND range exceeded.
pub const EXCP05_BOUND: i32 = 5;
/// Invalid opcode.
pub const EXCP06_ILLOP: i32 = 6;
/// Device not available.
pub const EXCP07_PREX: i32 = 7;
/// Double fault.
pub const EXCP08_DBLE: i32 = 8;
/// Coprocessor segment overrun.
pub const EXCP09_XERR: i32 = 9;
/// Invalid TSS.
pub const EXCP0A_TSS: i32 = 10;
/// Segment not present.
pub const EXCP0B_NOSEG: i32 = 11;
/// Stack fault.
pub const EXCP0C_STACK: i32 = 12;
/// General protection.
pub const EXCP0D_GPF: i32 = 13;
/// Page fault.
pub const EXCP0E_PAGE: i32 = 14;
/// x87 floating point error.
pub const EXCP10_COPR: i32 = 16;
/// Alignment check.
pub const EXCP11_ALGN: i32 = 17;
/// Machine check.
pub const EXCP12_MCHK: i32 = 18;
/// `EXCP_SYSCALL`: the guest ran SYSCALL under user mode emulation, which returns to the
/// emulator's cpu loop instead of entering a guest kernel.
pub const EXCP_SYSCALL: i32 = 0x100;

/// `CPU_INTERRUPT_POLL`: the local APIC asks its vCPU to look at it again.
pub const CPU_INTERRUPT_POLL: u32 = 0x0010;
/// `CPU_INTERRUPT_SMI`.
pub const CPU_INTERRUPT_SMI: u32 = 0x0040;
/// `CPU_INTERRUPT_NMI`.
pub const CPU_INTERRUPT_NMI: u32 = 0x0200;
/// `CPU_INTERRUPT_MCE`.
pub const CPU_INTERRUPT_MCE: u32 = 0x1000;
/// `CPU_INTERRUPT_VIRQ`, the SVM virtual interrupt (not used without SVM).
pub const CPU_INTERRUPT_VIRQ: u32 = 0x0100;
/// `CPU_INTERRUPT_SIPI`.
pub const CPU_INTERRUPT_SIPI: u32 = 0x0800;
/// `CPU_INTERRUPT_TPR`.
pub const CPU_INTERRUPT_TPR: u32 = 0x2000;
/// `CPU_INTERRUPT_INIT`, which x86 puts in the place of `CPU_INTERRUPT_RESET`.
pub const CPU_INTERRUPT_INIT: u32 = interrupt::RESET;

/// `DR6_BS`: the single step bit of DR6.
const DR6_BS: u64 = 1 << 14;

/// What an x86 vCPU needs from the machine around it: its local APIC (`cpu->apic_state`),
/// the 8259 (`isa_pic`) and system reset. Every method runs on the vCPU's own thread.
pub trait X86Platform: Send + Sync {
    /// `cpu_get_pic_interrupt()`: acknowledge and return the vector of the interrupt to
    /// take, from the APIC or else the PIC, or `None` when there is none.
    fn get_pic_interrupt(&self) -> Option<u8>;
    /// `apic_poll_irq()`.
    fn apic_poll_irq(&self);
    /// `apic_sipi()`: the startup vector when the APIC was waiting for a SIPI, which it then
    /// stops doing.
    fn apic_sipi(&self) -> Option<u8>;
    /// `apic_init_reset()`.
    fn apic_init_reset(&self);
    /// `cpu_get_apic_base()`.
    fn apic_base(&self) -> u64;
    /// `cpu_set_apic_base()`: false when the new value is refused, which raises #GP.
    fn set_apic_base(&self, val: u64) -> bool;
    /// `cpu_get_apic_tpr()`, the value CR8 reads.
    fn apic_tpr(&self) -> u8;
    /// `cpu_set_apic_tpr()`, for a CR8 write.
    fn set_apic_tpr(&self, val: u8);
    /// `apic_msr_read()` of x2APIC register `index`, `None` when it raises #GP.
    fn apic_msr_read(&self, index: u32) -> Option<u64>;
    /// `apic_msr_write()` of x2APIC register `index`: false when it raises #GP.
    fn apic_msr_write(&self, index: u32, val: u64) -> bool;
    /// `qemu_system_reset_request(SHUTDOWN_CAUSE_GUEST_RESET)`, after a triple fault.
    fn system_reset_request(&self);
}

/// Supervisor access with SMAP enforced, 64-bit addresses.
pub const MMU_KSMAP64_IDX: usize = 0;
/// Supervisor access with SMAP enforced, 32-bit addresses.
pub const MMU_KSMAP32_IDX: usize = 1;
/// User access, 64-bit addresses.
pub const MMU_USER64_IDX: usize = 2;
/// User access, 32-bit addresses.
pub const MMU_USER32_IDX: usize = 3;
/// Supervisor access without SMAP, 64-bit addresses.
pub const MMU_KNOSMAP64_IDX: usize = 4;
/// Supervisor access without SMAP, 32-bit addresses.
pub const MMU_KNOSMAP32_IDX: usize = 5;
/// Physical addresses.
pub const MMU_PHYS_IDX: usize = 6;
/// Nested paging (not used).
pub const MMU_NESTED_IDX: usize = 7;
/// `NB_MMU_MODES`.
pub const NB_MMU_MODES: usize = 8;

/// The x86 CPU: the [`CpuOps`] of x86 vCPUs translated by the TCG front end.
pub struct X86 {
    model: X86Cpu,
    /// The decoder's view of the model's features.
    feat: translate::Feat,
    io: Option<Arc<AddressSpace>>,
    /// `cpu_get_ticks()`, the count the TSC is, see [`ruvm_sys::hostticks`].
    ticks: Arc<Ticks>,
    irqs: Mutex<VecDeque<u8>>,
    triple_faults: AtomicU64,
    platform: OnceLock<Arc<dyn X86Platform>>,
    /// `CPUID_APIC` in `features[FEAT_1_EDX]`, cleared while the APIC is disabled.
    apic_feature: AtomicBool,
    /// Built for user mode emulation, `CONFIG_USER_ONLY`.
    user: bool,
}

impl fmt::Debug for X86 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("X86")
            .field("model", &self.model.typename())
            .field("io", &self.io.is_some())
            .field("triple_faults", &self.triple_faults.load(Ordering::Relaxed))
            .field("platform", &self.platform.get().is_some())
            .finish_non_exhaustive()
    }
}

impl X86 {
    /// An x86 CPU of the given model.
    pub fn new(model: X86Cpu) -> X86 {
        X86 {
            feat: translate::Feat::of(&model),
            model,
            io: None,
            ticks: Arc::new(Ticks::running()),
            irqs: Mutex::new(VecDeque::new()),
            triple_faults: AtomicU64::new(0),
            platform: OnceLock::new(),
            apic_feature: AtomicBool::new(true),
            user: false,
        }
    }

    /// The `qemu64` model.
    pub fn qemu64() -> X86 {
        let mut m = X86Cpu::new("qemu64", crate::cpuid::Accel::Tcg).expect("qemu64 exists");
        // Realizing only filters features against the accelerator; TCG keeps them all.
        let _ = m.realize();
        X86::new(m)
    }

    /// Use `io` for IN and OUT. Without one, port reads return all ones and writes are dropped.
    pub fn with_io(mut self, io: Arc<AddressSpace>) -> X86 {
        self.io = Some(io);
        self
    }

    /// Run as the CPU of a user mode emulator: SYSCALL leaves the vCPU with [`EXCP_SYSCALL`]
    /// instead of entering a guest kernel, as QEMU's `CONFIG_USER_ONLY` helpers do.
    pub fn with_user_mode(mut self) -> X86 {
        self.user = true;
        self
    }

    /// Whether this CPU runs under user mode emulation.
    pub fn is_user_mode(&self) -> bool {
        self.user
    }

    /// Count the TSC with `ticks` instead of from when this [`X86`] was made, so that the vCPUs
    /// of one machine share one `cpu_get_ticks()`, which stops while the machine does.
    pub fn with_ticks(mut self, ticks: Arc<Ticks>) -> X86 {
        self.ticks = ticks;
        self
    }

    /// Connect the machine: the local APIC, the PIC and system reset. Only the first call
    /// counts; it returns false if a platform was already set.
    pub fn set_platform(&self, platform: Arc<dyn X86Platform>) -> bool {
        self.platform.set(platform).is_ok()
    }

    pub(crate) fn platform(&self) -> Option<&Arc<dyn X86Platform>> {
        self.platform.get()
    }

    /// `cpu_set_apic_feature()` and `cpu_clear_apic_feature()`: whether CPUID leaf 1 shows
    /// the APIC, which the APIC turns off while it is disabled.
    pub fn set_apic_feature(&self, on: bool) {
        self.apic_feature.store(on, Ordering::Relaxed);
    }

    pub(crate) fn apic_feature(&self) -> bool {
        self.apic_feature.load(Ordering::Relaxed)
    }

    /// Drop the vectors queued by [`X86::raise_irq`], for a reset.
    pub fn clear_irqs(&self) {
        self.irqs.lock().expect("irq queue").clear();
    }

    /// The CPU model.
    pub fn model(&self) -> &X86Cpu {
        &self.model
    }

    /// Queue hardware interrupt `vector` for the vCPU whose shared half is `shared`, which
    /// takes it when IF is set.
    pub fn raise_irq(&self, shared: &CpuShared, vector: u8) {
        self.irqs.lock().expect("irq queue").push_back(vector);
        shared.cpu_interrupt(interrupt::HARD);
    }

    /// How many triple faults halted a vCPU.
    pub fn triple_faults(&self) -> u64 {
        self.triple_faults.load(Ordering::Relaxed)
    }

    /// The TSC before `tsc_offset` is added, `cpu_get_tsc()`. It counts host ticks, as
    /// `cpu_get_ticks()` does under TCG, so it runs at the host TSC rate on an x86 host.
    pub(crate) fn host_tsc(&self) -> u64 {
        self.ticks.get()
    }

    pub(crate) fn io(&self) -> Option<&Arc<AddressSpace>> {
        self.io.as_ref()
    }

    /// Count a triple fault and ask the platform for a reset.
    pub(crate) fn note_triple_fault(&self) {
        self.triple_faults.fetch_add(1, Ordering::Relaxed);
        if let Some(p) = self.platform() {
            p.system_reset_request();
        }
    }

    /// `x86_cpu_pending_interrupt()`, without SVM's virtual interrupts.
    fn pending_interrupt(&self, cpu: &Cpu<'_>, request: u32) -> u32 {
        if request & CPU_INTERRUPT_POLL != 0 {
            return CPU_INTERRUPT_POLL;
        }
        if request & CPU_INTERRUPT_SIPI != 0 {
            return CPU_INTERRUPT_SIPI;
        }
        let fl = ld64(cpu.env, EFLAGS) as u32;
        let hf = ld32(cpu.env, HFLAGS);
        let hf2 = ld32(cpu.env, HFLAGS2);
        if hf2 & HF2_GIF_MASK == 0 {
            return 0;
        }
        if request & CPU_INTERRUPT_SMI != 0 && hf & HF_SMM_MASK == 0 {
            CPU_INTERRUPT_SMI
        } else if request & CPU_INTERRUPT_NMI != 0 && hf2 & HF2_NMI_MASK == 0 {
            CPU_INTERRUPT_NMI
        } else if request & CPU_INTERRUPT_MCE != 0 {
            CPU_INTERRUPT_MCE
        } else if request & interrupt::HARD != 0
            && fl & IF_MASK != 0
            && hf & HF_INHIBIT_IRQ_MASK == 0
        {
            interrupt::HARD
        } else {
            0
        }
    }

    /// `cpu_get_pic_interrupt()` through the platform, or the next queued vector.
    fn get_pic_interrupt(&self, shared: &CpuShared) -> Option<u8> {
        if let Some(p) = self.platform() {
            return p.get_pic_interrupt();
        }
        let mut q = self.irqs.lock().expect("irq queue");
        let v = q.pop_front();
        if !q.is_empty() {
            shared.set_interrupt(interrupt::HARD);
        }
        v
    }

    /// `do_cpu_sipi()` with `apic_sipi()` and `cpu_x86_load_seg_cache_sipi()`.
    fn do_cpu_sipi(&self, cpu: &mut Cpu<'_>) {
        if ld32(cpu.env, HFLAGS) & HF_SMM_MASK != 0 {
            return;
        }
        let Some(vector) = self.platform().and_then(|p| p.apic_sipi()) else { return };
        st64(cpu.env, EIP, 0);
        st32(cpu.env, seg(R_CS) + SEG_SELECTOR, u32::from(vector) << 8);
        st64(cpu.env, seg(R_CS) + SEG_BASE, u64::from(vector) << 12);
        cpu.shared().halted.store(0, Ordering::Release);
    }
}

/// The [`X86`] behind a vCPU's ops.
pub(crate) fn x86_of(ops: &Arc<dyn CpuOps>) -> &X86 {
    ops.as_any().and_then(|a| a.downcast_ref::<X86>()).expect("the vCPU is an x86 TCG vCPU")
}

/// `x86_mmu_index_pl()`: the MMU index for an access at privilege level `pl`.
pub(crate) fn mmu_index_pl(env: &[u8], pl: u32) -> usize {
    let hf = ld32(env, HFLAGS);
    let fl = ld64(env, EFLAGS) as u32;
    let mmu_index_32 = usize::from(hf & HF_CS64_MASK == 0);
    let base = if pl == 3 {
        MMU_USER64_IDX
    } else if hf & HF_SMAP_MASK == 0 || fl & AC_MASK != 0 {
        MMU_KNOSMAP64_IDX
    } else {
        MMU_KSMAP64_IDX
    };
    base + mmu_index_32
}

/// `x86_mmu_index_kernel_pl()`: the MMU index for an implicit supervisor access, such as a
/// descriptor table read, made while running at privilege level `pl`.
pub(crate) fn mmu_index_kernel_pl(env: &[u8], pl: u32) -> usize {
    let hf = ld32(env, HFLAGS);
    let fl = ld64(env, EFLAGS) as u32;
    let mmu_index_32 = usize::from(hf & HF_LMA_MASK == 0);
    let base = if hf & HF_SMAP_MASK == 0 || (pl < 3 && fl & AC_MASK != 0) {
        MMU_KNOSMAP64_IDX
    } else {
        MMU_KSMAP64_IDX
    };
    base + mmu_index_32
}

/// `cpu_mmu_index_kernel()`.
pub(crate) fn mmu_index_kernel(env: &[u8]) -> usize {
    mmu_index_kernel_pl(env, ld32(env, HFLAGS) & HF_CPL_MASK)
}

/// The linear PC: `eip + cs.base`, cut to 32 bits outside 64-bit code.
pub(crate) fn linear_pc(env: &[u8]) -> u64 {
    let pc = ld64(env, EIP).wrapping_add(ld64(env, seg(R_CS) + SEG_BASE));
    if ld32(env, HFLAGS) & HF_CS64_MASK != 0 { pc } else { pc & 0xffff_ffff }
}

impl CpuOps for X86 {
    fn translate_code(&self, cpu: &mut Cpu<'_>, tb: &mut TbBuild) -> Result<(), CpuLoopExit> {
        let mut dc = translate::DisasContext::new(self.feat);
        translator_loop(cpu, tb, &mut dc)
    }

    fn get_tb_cpu_state(&self, cpu: &Cpu<'_>) -> TbCpuState {
        let env = &*cpu.env;
        let fl = ld64(env, EFLAGS) as u32;
        let flags = ld32(env, HFLAGS) | (fl & (IOPL_MASK | TF_MASK | RF_MASK | VM_MASK | AC_MASK));
        TbCpuState {
            pc: linear_pc(env),
            flags,
            cflags: 0,
            cs_base: ld64(env, seg(R_CS) + SEG_BASE),
        }
    }

    fn synchronize_from_tb(&self, cpu: &mut Cpu<'_>, tb: &Tb) {
        st64(cpu.env, EIP, tb.pc.wrapping_sub(tb.cs_base));
    }

    fn restore_state_to_opc(&self, cpu: &mut Cpu<'_>, tb: &Tb, data: &[u64; 3]) {
        let mut eip = data[0].wrapping_sub(tb.cs_base);
        if tb.flags & HF_CS64_MASK == 0 {
            eip &= 0xffff_ffff;
        }
        st64(cpu.env, EIP, eip);
        let cc_op = data[1] as u32;
        if cc_op != cc::CC_OP_DYNAMIC {
            st32(cpu.env, CC_OP, cc_op);
        }
    }

    /// x86 is TSO: everything is ordered except a store followed by a load, as QEMU's
    /// `TCG_MO_ALL & ~TCG_MO_ST_LD` for i386.
    fn guest_default_memory_order(&self) -> u32 {
        ruvm_jit_core::types::mo::ALL & !ruvm_jit_core::types::mo::ST_LD
    }

    fn set_pc(&self, cpu: &mut Cpu<'_>, pc: u64) {
        let base = ld64(cpu.env, seg(R_CS) + SEG_BASE);
        st64(cpu.env, EIP, pc.wrapping_sub(base));
    }

    fn get_pc(&self, cpu: &Cpu<'_>) -> u64 {
        linear_pc(cpu.env)
    }

    /// `x86_cpu_exec_interrupt()`: take one interrupt request, in priority order.
    fn cpu_exec_interrupt(&self, cpu: &mut Cpu<'_>, interrupt_request: u32) -> bool {
        let pending = self.pending_interrupt(cpu, interrupt_request);
        if pending == 0 {
            return false;
        }
        let shared = cpu.shared();
        match pending {
            CPU_INTERRUPT_POLL => {
                shared.reset_interrupt(CPU_INTERRUPT_POLL);
                if let Some(p) = self.platform() {
                    p.apic_poll_irq();
                }
            }
            CPU_INTERRUPT_SIPI => {
                shared.reset_interrupt(CPU_INTERRUPT_SIPI);
                self.do_cpu_sipi(cpu);
            }
            CPU_INTERRUPT_SMI => {
                // There is no SMM to enter.
                shared.reset_interrupt(CPU_INTERRUPT_SMI);
            }
            CPU_INTERRUPT_NMI => {
                shared.reset_interrupt(CPU_INTERRUPT_NMI);
                let hf2 = ld32(cpu.env, HFLAGS2);
                st32(cpu.env, HFLAGS2, hf2 | HF2_NMI_MASK);
                seg::do_interrupt_x86_hardirq(cpu, self, EXCP02_NMI, true);
            }
            CPU_INTERRUPT_MCE => {
                shared.reset_interrupt(CPU_INTERRUPT_MCE);
                seg::do_interrupt_x86_hardirq(cpu, self, EXCP12_MCHK, false);
            }
            _ => {
                shared.reset_interrupt(interrupt::HARD | CPU_INTERRUPT_VIRQ);
                if let Some(intno) = self.get_pic_interrupt(&shared) {
                    seg::do_interrupt_x86_hardirq(cpu, self, i32::from(intno), true);
                }
            }
        }
        true
    }

    /// `x86_cpu_exec_halt()`.
    fn cpu_exec_halt(&self, cpu: &mut Cpu<'_>) -> bool {
        let shared = cpu.shared();
        if shared.test_interrupt(CPU_INTERRUPT_POLL) {
            shared.reset_interrupt(CPU_INTERRUPT_POLL);
            if let Some(p) = self.platform() {
                p.apic_poll_irq();
            }
        }
        if !self.has_work(cpu) {
            return false;
        }
        // Complete the HLT instruction.
        if ld64(cpu.env, EFLAGS) as u32 & TF_MASK != 0 {
            let d6 = ld64(cpu.env, dr(6)) | DR6_BS;
            st64(cpu.env, dr(6), d6);
            seg::hlt_single_step(cpu, self);
        }
        true
    }

    /// `CPU_INTERRUPT_INIT` in `cpu_handle_interrupt()`: `do_cpu_init()`, then leave the loop
    /// with `EXCP_HALTED`.
    fn cpu_exec_reset(&self, cpu: &mut Cpu<'_>) {
        let shared = cpu.shared();
        let sipi = shared.interrupt_request() & CPU_INTERRUPT_SIPI;
        let save = cpu.env[..ENV_SIZE].to_vec();
        // x86_cpu_reset_hold(): the BSP is hard-wired to the first CPU.
        let is_bsp = shared.cpu_index == 0;
        let state = self.model.new_state(is_bsp);
        env::load_state(cpu.env, &state);
        // The registers between start_init_save and end_init_save survive INIT.
        cpu.env[STAR..DR].copy_from_slice(&save[STAR..DR]);
        cpu.env[TSC_OFFSET..ENV_SIZE].copy_from_slice(&save[TSC_OFFSET..ENV_SIZE]);
        shared.reset_interrupt(!sipi);
        shared.halted.store(u32::from(!is_bsp), Ordering::Release);
        self.clear_irqs();
        tlb_flush(cpu);
        if let Some(p) = self.platform() {
            p.apic_init_reset();
        }
        cpu.core.exception_index = excp::HALTED;
    }

    fn do_interrupt(&self, cpu: &mut Cpu<'_>) {
        seg::x86_cpu_do_interrupt(cpu, self);
    }

    fn fake_user_interrupt(&self, cpu: &mut Cpu<'_>) {
        user::do_interrupt_user(cpu);
    }

    fn has_work(&self, cpu: &Cpu<'_>) -> bool {
        let req = cpu.shared().interrupt_request();
        self.pending_interrupt(cpu, req) != 0
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
        mmu::tlb_fill(cpu, self, addr, size, access_type, mmu_idx, probe, ra)
    }

    fn do_unaligned_access(
        &self,
        cpu: &mut Cpu<'_>,
        _addr: u64,
        _access_type: MmuAccessType,
        _mmu_idx: usize,
        ra: Ra,
    ) -> CpuLoopExit {
        // Only CMPXCHG16B asks for alignment, and it raises #GP(0) when misaligned.
        seg::raise_exception_err_ra(cpu, EXCP0D_GPF, 0, ra)
    }

    fn mmu_index(&self, cpu: &Cpu<'_>, _ifetch: bool) -> usize {
        mmu_index_pl(cpu.env, ld32(cpu.env, HFLAGS) & HF_CPL_MASK)
    }

    fn pointer_wrap(&self, _cpu: &Cpu<'_>, mmu_idx: usize, result: u64, _base: u64) -> u64 {
        match mmu_idx {
            MMU_USER32_IDX | MMU_KSMAP32_IDX | MMU_KNOSMAP32_IDX => result & 0xffff_ffff,
            _ => result,
        }
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

/// The runtime configuration the x86 front end needs: 4 KiB pages and eight MMU modes.
pub fn jit_config() -> JitConfig {
    JitConfig { page_bits: 12, nb_mmu_modes: NB_MMU_MODES, ..JitConfig::default() }
}

/// The runtime's built-in helpers plus every x86 helper.
pub fn helper_registry() -> HelperRegistry {
    let mut r = HelperRegistry::new();
    helpers::register(&mut r);
    r
}

/// An interpreter backend that knows the x86 helpers.
pub fn interp_backend() -> Arc<InterpBackend> {
    Arc::new(InterpBackend::with_helpers(helper_registry()))
}

/// A runtime for x86 guests running on the interpreter backend.
pub fn new_jit() -> Arc<Jit> {
    Jit::new(jit_config(), interp_backend())
}

/// Make a vCPU on `jit` running `ops`, with memory `as_` and registers from `state`.
pub fn create_vcpu(
    jit: &Arc<Jit>,
    ops: Arc<X86>,
    as_: Arc<AddressSpace>,
    state: &X86CpuState,
) -> Vcpu {
    let mut v = jit.create_vcpu(ops, as_, ENV_SIZE);
    env::load_state(&mut v.env, state);
    v
}

/// Copy the registers of `v` back into `state`.
pub fn save_vcpu(v: &Vcpu, state: &mut X86CpuState) {
    env::save_state(&v.env, state);
    state.halted = v.shared().halted.load(Ordering::Acquire) != 0;
}
