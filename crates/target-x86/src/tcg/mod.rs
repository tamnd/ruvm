// SPDX-License-Identifier: GPL-2.0-or-later

//! The x86 TCG front end: the port of QEMU's `target/i386/tcg`, for 64-bit, 32-bit and 16-bit
//! guest code, run by `ruvm-jit`.
//!
//! [`X86`] is the [`CpuOps`] of an x86 vCPU. Its CPU state lives in the runtime's state
//! buffer with the layout in [`env`], so generated code reaches every register by offset. The
//! translator (`translate.rs`) decodes guest instructions and emits IR, keeping the
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
//! level long mode paging, with NX, WP, SMEP and SMAP.
//!
//! **x87, MMX, SSE and AVX are not implemented yet.** Every x87 escape opcode (D8 to DF),
//! every MMX and SSE opcode in the 0F, 0F 38 and 0F 3A maps (including the 66, F2 and F3
//! forms), every vector instruction behind a VEX prefix, and the EVEX prefix raise #UD. FWAIT
//! is a no-op. FXSAVE, FXRSTOR, XSAVE, XRSTOR and XSAVEOPT are #UD too. CPUID still reports
//! what the model has, so a guest that checks CPUID and then uses SSE gets #UD. The VEX prefix
//! itself is decoded (`translate/ext.rs`), and so are the general purpose register
//! instructions that come with this part of the instruction set: ANDN, BEXTR, BLSI, BLSMSK,
//! BLSR, BZHI, MULX, PDEP, PEXT, RORX, SARX, SHLX and SHRX (VEX class 13, VEX.L must be 0),
//! ADCX and ADOX (with QEMU's `CC_OP_ADCX`, `CC_OP_ADOX` and `CC_OP_ADCOX` carry chaining),
//! MOVBE, CRC32, RDRAND, RDSEED, RDPID, XGETBV, XSETBV, RDFSBASE, RDGSBASE, WRFSBASE,
//! WRGSBASE (CR4.FSGSBASE is tested at run time, as in QEMU), MOVNTI, LDMXCSR and STMXCSR.
//! `helpers/vec.rs` already holds the SSE to AVX2 arithmetic kernels (on `ruvm-softfloat`,
//! with MXCSR rounding, DAZ, FZ and flags), but no decoder calls them yet.
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
//! - A triple fault halts the vCPU (counted by [`X86::triple_faults`]) instead of requesting
//!   a system reset, since this crate has no machine to reset.
//! - The accessed and dirty bits of page table entries are set with a plain read and write of
//!   physical memory, not with a compare and swap, so another vCPU changing the same entry at
//!   the same moment can lose an update.
//! - The TSC counts host nanoseconds since [`X86`] was made, plus `IA32_TSC` writes.
//! - Unknown MSRs read as zero and ignore writes instead of raising #GP. Only the MSRs this
//!   front end uses (EFER, STAR, LSTAR, CSTAR, FMASK, FS and GS base, KERNEL_GS_BASE, the
//!   SYSENTER MSRs, TSC, TSC_AUX, PAT, APIC_BASE, MISC_ENABLE) are kept.
//! - Debug registers are stored but hardware breakpoints and watchpoints are not armed.
//!   Single stepping with TF works.
//! - SVM, VMX, SMM, protection keys, MPX, CET, FRED, LAM and nested paging are not modelled.
//! - Hardware interrupts come from a small vector queue fed by [`X86::raise_irq`], standing
//!   in for the PIC and APIC that QEMU asks with `cpu_get_pic_interrupt()`. NMIs are not
//!   modelled.
//! - PAUSE ends the block like any other instruction rather than leaving the execution loop.
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
//! instructions, RDPKRU and WRPKRU, the AMD `lock mov cr0` alias for CR8, and the PCREL
//! translation mode. I/O breakpoints (`bpt_io`) are not checked. Virtual 8086 mode decodes
//! like real mode with the IOPL checks, but nothing enters it (see above).

pub mod cc;
pub mod env;
mod helpers;
mod mmu;
mod seg;
mod translate;

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ruvm_jit::translate::TbBuild;
use ruvm_jit::{
    Cpu, CpuLoopExit, CpuOps, CpuShared, InterpBackend, Jit, JitConfig, MmuAccessType, Ra, Tb,
    TbCpuState, Vcpu, interrupt, translator_loop,
};
use ruvm_jit_interp::HelperRegistry;
use ruvm_mem::AddressSpace;

use crate::cpuid::X86Cpu;
use crate::state::{HF_CPL_MASK, HF_CS64_MASK, HF_LMA_MASK, HF2_GIF_MASK, R_CS, X86CpuState};
use env::{
    AC_MASK, CC_OP, EFLAGS, EIP, HF_INHIBIT_IRQ_MASK, HF_SMAP_MASK, HFLAGS, HFLAGS2, IF_MASK,
    IOPL_MASK, RF_MASK, SEG_BASE, TF_MASK, VM_MASK, ld32, ld64, seg, st32, st64,
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
    io: Option<Arc<AddressSpace>>,
    tsc_base: Instant,
    irqs: Mutex<VecDeque<u8>>,
    triple_faults: AtomicU64,
}

impl fmt::Debug for X86 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("X86")
            .field("model", &self.model.typename())
            .field("io", &self.io.is_some())
            .field("triple_faults", &self.triple_faults.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl X86 {
    /// An x86 CPU of the given model.
    pub fn new(model: X86Cpu) -> X86 {
        X86 {
            model,
            io: None,
            tsc_base: Instant::now(),
            irqs: Mutex::new(VecDeque::new()),
            triple_faults: AtomicU64::new(0),
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

    /// The TSC before `tsc_offset` is added, `cpu_get_tsc()`.
    pub(crate) fn host_tsc(&self) -> u64 {
        self.tsc_base.elapsed().as_nanos() as u64
    }

    pub(crate) fn io(&self) -> Option<&Arc<AddressSpace>> {
        self.io.as_ref()
    }

    pub(crate) fn note_triple_fault(&self) {
        self.triple_faults.fetch_add(1, Ordering::Relaxed);
    }

    /// `x86_cpu_pending_interrupt()` for the interrupts this port models.
    fn pending_interrupt(&self, cpu: &Cpu<'_>, request: u32) -> u32 {
        let fl = ld64(cpu.env, EFLAGS) as u32;
        let hf = ld32(cpu.env, HFLAGS);
        let hf2 = ld32(cpu.env, HFLAGS2);
        if hf2 & HF2_GIF_MASK == 0 {
            return 0;
        }
        if request & interrupt::HARD != 0 && fl & IF_MASK != 0 && hf & HF_INHIBIT_IRQ_MASK == 0 {
            return interrupt::HARD;
        }
        0
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
        let mut dc = translate::DisasContext::new(&self.model);
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

    fn cpu_exec_interrupt(&self, cpu: &mut Cpu<'_>, interrupt_request: u32) -> bool {
        let pending = self.pending_interrupt(cpu, interrupt_request);
        if pending == 0 {
            return false;
        }
        let shared = cpu.shared();
        let vector = {
            let mut q = self.irqs.lock().expect("irq queue");
            let v = q.pop_front();
            if q.is_empty() {
                shared.reset_interrupt(interrupt::HARD);
            }
            v
        };
        let Some(vector) = vector else {
            return false;
        };
        seg::do_interrupt_x86_hardirq(cpu, self, i32::from(vector));
        true
    }

    fn do_interrupt(&self, cpu: &mut Cpu<'_>) {
        seg::x86_cpu_do_interrupt(cpu, self);
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
    let mut v = jit.create_vcpu(ops, as_, env::ENV_SIZE);
    env::load_state(&mut v.env, state);
    v
}

/// Copy the registers of `v` back into `state`.
pub fn save_vcpu(v: &Vcpu, state: &mut X86CpuState) {
    env::save_state(&v.env, state);
    state.halted = v.shared().halted.load(Ordering::Acquire) != 0;
}
