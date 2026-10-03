// SPDX-License-Identifier: GPL-2.0-or-later

//! Architectural state of one x86 vCPU and the reset that fills it in.
//!
//! [`X86CpuState`] is a trimmed `CPUX86State` from `target/i386/cpu.h`. It
//! keeps the registers a hardware accelerator needs to load into a vCPU:
//! general registers, segment caches, control registers, the FPU and SSE
//! control words and the MSR values that `kvm_put_msrs()` writes. Fields that
//! only the TCG translator uses (condition code temporaries, the soft float
//! status, the TLB) are left out.
//!
//! [`X86CpuState::reset`] is a port of `x86_cpu_reset_hold()` from
//! `target/i386/cpu.c`, including the parts of `kvm_arch_reset_vcpu()` and
//! `apic_reset_common()` that change this state.

use crate::msr::{
    APIC_DEFAULT_ADDRESS, MSR_AMD64_TSC_RATIO_DEFAULT, MSR_IA32_APICBASE_BSP,
    MSR_IA32_APICBASE_ENABLE, MSR_PAT_RESET, SGX_LEPUBKEYHASH_DEFAULT, misc_enable_reset,
};

/// Index of RAX in [`X86CpuState::regs`].
pub const R_EAX: usize = 0;
/// Index of RCX in [`X86CpuState::regs`].
pub const R_ECX: usize = 1;
/// Index of RDX in [`X86CpuState::regs`].
pub const R_EDX: usize = 2;
/// Index of RBX in [`X86CpuState::regs`].
pub const R_EBX: usize = 3;
/// Index of RSP in [`X86CpuState::regs`].
pub const R_ESP: usize = 4;
/// Index of RBP in [`X86CpuState::regs`].
pub const R_EBP: usize = 5;
/// Index of RSI in [`X86CpuState::regs`].
pub const R_ESI: usize = 6;
/// Index of RDI in [`X86CpuState::regs`].
pub const R_EDI: usize = 7;
/// Number of general purpose registers in 64-bit mode.
pub const CPU_NB_REGS: usize = 16;

/// Index of ES in [`X86CpuState::segs`].
pub const R_ES: usize = 0;
/// Index of CS in [`X86CpuState::segs`].
pub const R_CS: usize = 1;
/// Index of SS in [`X86CpuState::segs`].
pub const R_SS: usize = 2;
/// Index of DS in [`X86CpuState::segs`].
pub const R_DS: usize = 3;
/// Index of FS in [`X86CpuState::segs`].
pub const R_FS: usize = 4;
/// Index of GS in [`X86CpuState::segs`].
pub const R_GS: usize = 5;

/// Granularity bit of a segment's flags.
pub const DESC_G_MASK: u32 = 1 << 23;
/// Default operand size (the B or D bit).
pub const DESC_B_SHIFT: u32 = 22;
/// Default operand size mask.
pub const DESC_B_MASK: u32 = 1 << DESC_B_SHIFT;
/// 64-bit code segment.
pub const DESC_L_MASK: u32 = 1 << 21;
/// Available for software.
pub const DESC_AVL_MASK: u32 = 1 << 20;
/// Segment present.
pub const DESC_P_MASK: u32 = 1 << 15;
/// Shift of the privilege level field.
pub const DESC_DPL_SHIFT: u32 = 13;
/// Privilege level field.
pub const DESC_DPL_MASK: u32 = 3 << DESC_DPL_SHIFT;
/// Code or data segment (clear for system segments).
pub const DESC_S_MASK: u32 = 1 << 12;
/// Shift of the four bit type field.
pub const DESC_TYPE_SHIFT: u32 = 8;
/// Type field.
pub const DESC_TYPE_MASK: u32 = 15 << DESC_TYPE_SHIFT;
/// Accessed.
pub const DESC_A_MASK: u32 = 1 << 8;
/// Code segment.
pub const DESC_CS_MASK: u32 = 1 << 11;
/// Conforming code segment.
pub const DESC_C_MASK: u32 = 1 << 10;
/// Readable code segment.
pub const DESC_R_MASK: u32 = 1 << 9;
/// Expand-down data segment.
pub const DESC_E_MASK: u32 = 1 << 10;
/// Writable data segment.
pub const DESC_W_MASK: u32 = 1 << 9;

/// Protection enable.
pub const CR0_PE_MASK: u64 = 1 << 0;
/// Monitor coprocessor.
pub const CR0_MP_MASK: u64 = 1 << 1;
/// FPU emulation.
pub const CR0_EM_MASK: u64 = 1 << 2;
/// Task switched.
pub const CR0_TS_MASK: u64 = 1 << 3;
/// Extension type, always set.
pub const CR0_ET_MASK: u64 = 1 << 4;
/// Numeric error.
pub const CR0_NE_MASK: u64 = 1 << 5;
/// Write protect.
pub const CR0_WP_MASK: u64 = 1 << 16;
/// Alignment mask.
pub const CR0_AM_MASK: u64 = 1 << 18;
/// Not write-through.
pub const CR0_NW_MASK: u64 = 1 << 29;
/// Cache disable.
pub const CR0_CD_MASK: u64 = 1 << 30;
/// Paging.
pub const CR0_PG_MASK: u64 = 1 << 31;

/// Virtual 8086 mode extensions.
pub const CR4_VME_MASK: u64 = 1 << 0;
/// Protected mode virtual interrupts.
pub const CR4_PVI_MASK: u64 = 1 << 1;
/// Time stamp disable.
pub const CR4_TSD_MASK: u64 = 1 << 2;
/// Debugging extensions.
pub const CR4_DE_MASK: u64 = 1 << 3;
/// Page size extension.
pub const CR4_PSE_MASK: u64 = 1 << 4;
/// Physical address extension.
pub const CR4_PAE_MASK: u64 = 1 << 5;
/// Machine check enable.
pub const CR4_MCE_MASK: u64 = 1 << 6;
/// Page global enable.
pub const CR4_PGE_MASK: u64 = 1 << 7;
/// Performance counter enable.
pub const CR4_PCE_MASK: u64 = 1 << 8;
/// FXSAVE and FXRSTOR support.
pub const CR4_OSFXSR_MASK: u64 = 1 << 9;
/// Unmasked SIMD exception support.
pub const CR4_OSXMMEXCPT_MASK: u64 = 1 << 10;
/// User mode instruction prevention.
pub const CR4_UMIP_MASK: u64 = 1 << 11;
/// Five level paging.
pub const CR4_LA57_MASK: u64 = 1 << 12;
/// VMX enable.
pub const CR4_VMXE_MASK: u64 = 1 << 13;
/// SMX enable.
pub const CR4_SMXE_MASK: u64 = 1 << 14;
/// FSGSBASE instructions enable.
pub const CR4_FSGSBASE_MASK: u64 = 1 << 16;
/// Process context identifiers.
pub const CR4_PCIDE_MASK: u64 = 1 << 17;
/// XSAVE and processor extended states enable.
pub const CR4_OSXSAVE_MASK: u64 = 1 << 18;
/// Supervisor mode execution prevention.
pub const CR4_SMEP_MASK: u64 = 1 << 20;
/// Supervisor mode access prevention.
pub const CR4_SMAP_MASK: u64 = 1 << 21;
/// Protection keys for user pages.
pub const CR4_PKE_MASK: u64 = 1 << 22;
/// Control flow enforcement.
pub const CR4_CET_MASK: u64 = 1 << 23;
/// Protection keys for supervisor pages.
pub const CR4_PKS_MASK: u64 = 1 << 24;
/// Linear address masking for supervisor pointers.
pub const CR4_LAM_SUP_MASK: u64 = 1 << 28;
/// Flexible return and event delivery.
pub const CR4_FRED_MASK: u64 = 1 << 32;

/// SYSCALL enable.
pub const MSR_EFER_SCE: u64 = 1 << 0;
/// Long mode enable.
pub const MSR_EFER_LME: u64 = 1 << 8;
/// Long mode active.
pub const MSR_EFER_LMA: u64 = 1 << 10;
/// No-execute enable.
pub const MSR_EFER_NXE: u64 = 1 << 11;
/// SVM enable.
pub const MSR_EFER_SVME: u64 = 1 << 12;
/// Fast FXSAVE and FXRSTOR.
pub const MSR_EFER_FFXSR: u64 = 1 << 14;

/// `hflags` bit for protected mode.
pub const HF_PE_MASK: u32 = 1 << 7;
/// `hflags` bit for a 32-bit code segment.
pub const HF_CS32_MASK: u32 = 1 << 4;
/// `hflags` bit for a 32-bit stack segment.
pub const HF_SS32_MASK: u32 = 1 << 5;
/// `hflags` bit set when a segment base may be nonzero.
pub const HF_ADDSEG_MASK: u32 = 1 << 6;
/// `hflags` bits for CR0.MP, CR0.EM and CR0.TS.
pub const HF_MP_MASK: u32 = 1 << 9;
/// See [`HF_MP_MASK`].
pub const HF_EM_MASK: u32 = 1 << 10;
/// See [`HF_MP_MASK`].
pub const HF_TS_MASK: u32 = 1 << 11;
/// `hflags` bit for long mode active.
pub const HF_LMA_MASK: u32 = 1 << 14;
/// `hflags` bit for a 64-bit code segment.
pub const HF_CS64_MASK: u32 = 1 << 15;
/// `hflags` bit for system management mode.
pub const HF_SMM_MASK: u32 = 1 << 19;
/// `hflags` bit for running as an SVM guest.
pub const HF_GUEST_MASK: u32 = 1 << 21;
/// `hflags` field for the current privilege level.
pub const HF_CPL_MASK: u32 = 3;
/// `hflags2` bit: the global interrupt flag.
pub const HF2_GIF_MASK: u32 = 1 << 0;
/// `hflags2` bit: an NMI is being handled.
pub const HF2_NMI_MASK: u32 = 1 << 2;
/// `hflags2` bit: the virtual global interrupt flag.
pub const HF2_VGIF_MASK: u32 = 1 << 8;

/// Bits of DR6 that always read as 1.
pub const DR6_FIXED_1: u64 = 0xffff_0ff0;
/// Bits of DR7 that always read as 1.
pub const DR7_FIXED_1: u64 = 0x0000_0400;

/// Bit 1 of RFLAGS, which is always set.
pub const RFLAGS_FIXED_1: u64 = 0x2;

/// XCR0 bit for x87 state.
pub const XSTATE_FP_MASK: u64 = 1 << 0;

/// The vCPU is runnable (`KVM_MP_STATE_RUNNABLE`).
pub const MP_STATE_RUNNABLE: u32 = 0;
/// The vCPU waits for INIT and SIPI (`KVM_MP_STATE_UNINITIALIZED`).
pub const MP_STATE_UNINITIALIZED: u32 = 1;

/// Number of variable MTRR pairs QEMU models.
pub const MTRR_VAR_COUNT: usize = 8;
/// Number of machine check banks times four registers per bank.
pub const MCE_BANK_REGS: usize = 40;

/// A cached segment descriptor, as `SegmentCache` in QEMU.
///
/// `flags` holds bits 8 to 23 of the high descriptor word in their
/// descriptor positions, so the `DESC_*` masks above apply to it directly.
/// The GDT and IDT use only `base` and `limit`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SegmentCache {
    /// Selector.
    pub selector: u32,
    /// Linear base address.
    pub base: u64,
    /// Limit, already scaled by the granularity bit.
    pub limit: u32,
    /// Attribute bits.
    pub flags: u32,
}

/// One variable range MTRR.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MtrrVar {
    /// `MTRRphysBase`.
    pub base: u64,
    /// `MTRRphysMask`.
    pub mask: u64,
}

/// How the interrupt controller is split between KVM and user space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IrqchipMode {
    /// Everything in user space (`kernel-irqchip=off`).
    Off,
    /// The local APIC in KVM, the IOAPIC and PIC in user space (`split`).
    Split,
    /// Everything in KVM (`kernel-irqchip=on`).
    #[default]
    Full,
}

impl IrqchipMode {
    /// True when the local APIC is emulated by KVM (`kvm_irqchip_in_kernel()`).
    pub fn in_kernel(self) -> bool {
        !matches!(self, IrqchipMode::Off)
    }
}

/// Inputs to [`X86CpuState::reset`] that come from outside `CPUX86State`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResetConfig {
    /// `cs->cpu_index == 0`. QEMU hard wires the BSP to the first CPU.
    pub is_bsp: bool,
    /// Running under KVM (`kvm_enabled()`).
    pub kvm: bool,
    /// Interrupt controller mode, used for the KVM MP state.
    pub irqchip: IrqchipMode,
    /// `CPUID[1].EAX`, loaded into EDX at reset.
    pub cpuid_version: u32,
    /// The guest has `monitor` in `CPUID[1].ECX`.
    pub has_monitor: bool,
}

/// The register state of one x86 vCPU.
///
/// The field names follow `CPUX86State` where one exists. `Default` gives an
/// all-zero state, the same as QEMU's `memset` before the reset code runs;
/// call [`X86CpuState::reset`] to get the power-on values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct X86CpuState {
    /// General registers in `R_EAX` .. `R_EDI`, then R8 to R15.
    pub regs: [u64; CPU_NB_REGS],
    /// Instruction pointer (`eip`).
    pub rip: u64,
    /// Flags register (`eflags`).
    pub rflags: u64,
    /// Hidden flags that summarise the CPU mode (`HF_*`).
    pub hflags: u32,
    /// More hidden flags (`HF2_*`).
    pub hflags2: u32,
    /// ES, CS, SS, DS, FS, GS in that order.
    pub segs: [SegmentCache; 6],
    /// Local descriptor table register.
    pub ldt: SegmentCache,
    /// Task register.
    pub tr: SegmentCache,
    /// Global descriptor table register.
    pub gdt: SegmentCache,
    /// Interrupt descriptor table register.
    pub idt: SegmentCache,
    /// CR0.
    pub cr0: u64,
    /// CR2.
    pub cr2: u64,
    /// CR3.
    pub cr3: u64,
    /// CR4.
    pub cr4: u64,
    /// CR8, the task priority. QEMU keeps it in the APIC; it is 0 at reset.
    pub cr8: u64,
    /// A20 gate mask.
    pub a20_mask: i32,
    /// Extended feature enable register.
    pub efer: u64,
    /// `IA32_APIC_BASE`. QEMU keeps it in the APIC device.
    pub apic_base: u64,
    /// XCR0.
    pub xcr0: u64,
    /// Time stamp counter.
    pub tsc: u64,

    /// x87 top of stack.
    pub fpstt: u32,
    /// x87 status word.
    pub fpus: u16,
    /// x87 control word.
    pub fpuc: u16,
    /// x87 tag bits, 1 meaning empty.
    pub fptags: [u8; 8],
    /// SSE control and status.
    pub mxcsr: u32,
    /// XSAVE header `XSTATE_BV`.
    pub xstate_bv: u64,
    /// PKRU.
    pub pkru: u32,
    /// The x87 registers in physical order, `fpregs[]`: the 64-bit significand, which is
    /// also the MMX register, then the sign and exponent in the low 16 bits of the second
    /// word.
    pub fpregs: [[u64; 2]; 8],
    /// The last x87 opcode, `fpop`.
    pub fpop: u16,
    /// The last x87 instruction pointer, `fpip`.
    pub fpip: u64,
    /// The last x87 data pointer, `fpdp`.
    pub fpdp: u64,
    /// The last x87 code selector, `fpcs`.
    pub fpcs: u16,
    /// The last x87 data selector, `fpds`.
    pub fpds: u16,
    /// The vector registers ZMM0 to ZMM31, eight 64-bit lanes each, lowest first. XMMn is
    /// lanes 0 and 1 and YMMn is lanes 0 to 3.
    pub xmm_regs: [[u64; 8]; 32],

    /// Debug registers. DR4 and DR5 are unused.
    pub dr: [u64; 8],

    /// `IA32_SYSENTER_CS`.
    pub sysenter_cs: u32,
    /// `IA32_SYSENTER_ESP`.
    pub sysenter_esp: u64,
    /// `IA32_SYSENTER_EIP`.
    pub sysenter_eip: u64,
    /// `STAR`.
    pub star: u64,
    /// `VM_HSAVE_PA`.
    pub vm_hsave: u64,
    /// `LSTAR`.
    pub lstar: u64,
    /// `CSTAR`.
    pub cstar: u64,
    /// `FMASK`.
    pub fmask: u64,
    /// `KERNEL_GS_BASE`.
    pub kernelgsbase: u64,
    /// FRED stack pointers for levels 0 to 3.
    pub fred_rsp: [u64; 4],
    /// FRED stack levels.
    pub fred_stklvls: u64,
    /// FRED shadow stack pointers for levels 1 to 3.
    pub fred_ssp: [u64; 3],
    /// FRED configuration.
    pub fred_config: u64,
    /// User mode CET.
    pub u_cet: u64,
    /// Supervisor mode CET.
    pub s_cet: u64,
    /// Shadow stack pointers for rings 0 to 3.
    pub pl_ssp: [u64; 4],
    /// Interrupt shadow stack table.
    pub int_ssp_table: u64,
    /// `IA32_TSC_ADJUST`.
    pub tsc_adjust: u64,
    /// `IA32_TSC_DEADLINE`.
    pub tsc_deadline: u64,
    /// `TSC_AUX`.
    pub tsc_aux: u64,
    /// `IA32_MCG_STATUS`.
    pub mcg_status: u64,
    /// `IA32_MISC_ENABLE`.
    pub msr_ia32_misc_enable: u64,
    /// `IA32_FEATURE_CONTROL`.
    pub msr_ia32_feature_control: u64,
    /// `IA32_SGXLEPUBKEYHASH0` to 3.
    pub msr_ia32_sgxlepubkeyhash: [u64; 4],
    /// `IA32_PAT`.
    pub pat: u64,
    /// SMRAM base.
    pub smbase: u32,
    /// `MSR_SMI_COUNT`.
    pub msr_smi_count: u64,
    /// `IA32_PKRS`.
    pub pkrs: u32,
    /// `IA32_TSX_CTRL`.
    pub tsx_ctrl: u32,
    /// `IA32_SPEC_CTRL`.
    pub spec_ctrl: u64,
    /// `MSR_AMD64_TSC_RATIO`.
    pub amd_tsc_scale_msr: u64,
    /// `MSR_VIRT_SSBD`.
    pub virt_ssbd: u64,
    /// `IA32_BNDCFGS`.
    pub msr_bndcfgs: u64,
    /// kvmclock system time.
    pub system_time_msr: u64,
    /// kvmclock wall clock.
    pub wall_clock_msr: u64,
    /// KVM steal time.
    pub steal_time_msr: u64,
    /// KVM async page fault enable.
    pub async_pf_en_msr: u64,
    /// KVM async page fault interrupt.
    pub async_pf_int_msr: u64,
    /// KVM paravirtual EOI.
    pub pv_eoi_en_msr: u64,
    /// KVM poll control.
    pub poll_control_msr: u64,
    /// `IA32_XFD`.
    pub msr_xfd: u64,
    /// `IA32_XFD_ERR`.
    pub msr_xfd_err: u64,
    /// `MSR_K7_HWCR`.
    pub msr_hwcr: u64,

    /// Exception being delivered, -1 when none.
    pub old_exception: i32,
    /// SVM interrupt control.
    pub int_ctl: u32,
    /// An NMI was injected and not yet delivered.
    pub nmi_injected: bool,
    /// An NMI is pending.
    pub nmi_pending: bool,

    // Fields from here on sit after `end_reset_fields` in QEMU and survive
    // the memset. Reset sets some of them explicitly.
    /// Fixed range MTRRs in [`crate::msr::MTRR_FIXED_MSRS`] order.
    pub mtrr_fixed: [u64; 11],
    /// `MTRRdefType`.
    pub mtrr_deftype: u64,
    /// Variable range MTRRs.
    pub mtrr_var: [MtrrVar; MTRR_VAR_COUNT],
    /// KVM multiprocessing state (`KVM_MP_STATE_*`).
    pub mp_state: u32,
    /// Pending or injected exception vector, -1 when none.
    pub exception_nr: i32,
    /// Injected interrupt vector, -1 when none.
    pub interrupt_injected: i32,
    /// The injected interrupt came from INTn.
    pub soft_interrupt: bool,
    /// An exception is pending.
    pub exception_pending: bool,
    /// An exception was injected.
    pub exception_injected: bool,
    /// The exception has an error code.
    pub has_error_code: bool,
    /// Error code of the pending or injected exception.
    pub error_code: u32,
    /// The exception carries a payload (CR2 or DR6).
    pub exception_has_payload: bool,
    /// Exception payload.
    pub exception_payload: u64,
    /// A triple fault is pending.
    pub triple_fault_pending: bool,
    /// Last SIPI vector.
    pub sipi_vector: u32,
    /// `IA32_MCG_CAP`, set at realize time.
    pub mcg_cap: u64,
    /// `IA32_MCG_CTL`, set at realize time.
    pub mcg_ctl: u64,
    /// `IA32_MCG_EXT_CTL`.
    pub mcg_ext_ctl: u64,
    /// Machine check banks, four registers each.
    pub mce_banks: [u64; MCE_BANK_REGS],
    /// `IA32_XSS`.
    pub xss: u64,
    /// `IA32_UMWAIT_CONTROL`.
    pub umwait: u32,

    /// The vCPU is halted. Lives in `CPUState` in QEMU.
    pub halted: bool,
}

impl Default for X86CpuState {
    fn default() -> Self {
        Self {
            regs: [0; CPU_NB_REGS],
            rip: 0,
            rflags: 0,
            hflags: 0,
            hflags2: 0,
            segs: [SegmentCache::default(); 6],
            ldt: SegmentCache::default(),
            tr: SegmentCache::default(),
            gdt: SegmentCache::default(),
            idt: SegmentCache::default(),
            cr0: 0,
            cr2: 0,
            cr3: 0,
            cr4: 0,
            cr8: 0,
            a20_mask: 0,
            efer: 0,
            apic_base: 0,
            xcr0: 0,
            tsc: 0,
            fpstt: 0,
            fpus: 0,
            fpuc: 0,
            fptags: [0; 8],
            mxcsr: 0,
            xstate_bv: 0,
            pkru: 0,
            fpregs: [[0; 2]; 8],
            fpop: 0,
            fpip: 0,
            fpdp: 0,
            fpcs: 0,
            fpds: 0,
            xmm_regs: [[0; 8]; 32],
            dr: [0; 8],
            sysenter_cs: 0,
            sysenter_esp: 0,
            sysenter_eip: 0,
            star: 0,
            vm_hsave: 0,
            lstar: 0,
            cstar: 0,
            fmask: 0,
            kernelgsbase: 0,
            fred_rsp: [0; 4],
            fred_stklvls: 0,
            fred_ssp: [0; 3],
            fred_config: 0,
            u_cet: 0,
            s_cet: 0,
            pl_ssp: [0; 4],
            int_ssp_table: 0,
            tsc_adjust: 0,
            tsc_deadline: 0,
            tsc_aux: 0,
            mcg_status: 0,
            msr_ia32_misc_enable: 0,
            msr_ia32_feature_control: 0,
            msr_ia32_sgxlepubkeyhash: [0; 4],
            pat: 0,
            smbase: 0,
            msr_smi_count: 0,
            pkrs: 0,
            tsx_ctrl: 0,
            spec_ctrl: 0,
            amd_tsc_scale_msr: 0,
            virt_ssbd: 0,
            msr_bndcfgs: 0,
            system_time_msr: 0,
            wall_clock_msr: 0,
            steal_time_msr: 0,
            async_pf_en_msr: 0,
            async_pf_int_msr: 0,
            pv_eoi_en_msr: 0,
            poll_control_msr: 0,
            msr_xfd: 0,
            msr_xfd_err: 0,
            msr_hwcr: 0,
            old_exception: 0,
            int_ctl: 0,
            nmi_injected: false,
            nmi_pending: false,
            mtrr_fixed: [0; 11],
            mtrr_deftype: 0,
            mtrr_var: [MtrrVar::default(); MTRR_VAR_COUNT],
            mp_state: 0,
            exception_nr: 0,
            interrupt_injected: 0,
            soft_interrupt: false,
            exception_pending: false,
            exception_injected: false,
            has_error_code: false,
            error_code: 0,
            exception_has_payload: false,
            exception_payload: 0,
            triple_fault_pending: false,
            sipi_vector: 0,
            mcg_cap: 0,
            mcg_ctl: 0,
            mcg_ext_ctl: 0,
            mce_banks: [0; MCE_BANK_REGS],
            xss: 0,
            umwait: 0,
            halted: false,
        }
    }
}

impl X86CpuState {
    /// A state with every field at the power-on value.
    pub fn new_reset(cfg: &ResetConfig) -> Self {
        let mut s = Self::default();
        s.reset(cfg);
        s
    }

    /// Clear the fields that QEMU keeps before `end_reset_fields`.
    fn clear_reset_fields(&mut self) {
        let kept = Self {
            mtrr_fixed: self.mtrr_fixed,
            mtrr_deftype: self.mtrr_deftype,
            mtrr_var: self.mtrr_var,
            mp_state: self.mp_state,
            exception_nr: self.exception_nr,
            interrupt_injected: self.interrupt_injected,
            soft_interrupt: self.soft_interrupt,
            exception_pending: self.exception_pending,
            exception_injected: self.exception_injected,
            has_error_code: self.has_error_code,
            exception_has_payload: self.exception_has_payload,
            exception_payload: self.exception_payload,
            triple_fault_pending: self.triple_fault_pending,
            sipi_vector: self.sipi_vector,
            tsc: self.tsc,
            mcg_cap: self.mcg_cap,
            mcg_ctl: self.mcg_ctl,
            mcg_ext_ctl: self.mcg_ext_ctl,
            mce_banks: self.mce_banks,
            xstate_bv: self.xstate_bv,
            xss: self.xss,
            umwait: self.umwait,
            halted: self.halted,
            ..Self::default()
        };
        *self = kept;
    }

    /// Port of `x86_cpu_reset_hold()` for a system emulator build.
    ///
    /// Fields that QEMU keeps across reset (the MTRRs before they are
    /// cleared again, `MCG_CAP`, `MCG_CTL`, the MCE banks, `XSS`) stay as
    /// they were, so the machine check setup done at realize time survives.
    /// `tsc` is also kept under KVM, where a nonzero value becomes 1.
    pub fn reset(&mut self, cfg: &ResetConfig) {
        self.clear_reset_fields();

        self.old_exception = -1;

        self.int_ctl = 0;
        self.hflags2 |= HF2_GIF_MASK;
        self.hflags2 |= HF2_VGIF_MASK;
        self.hflags &= !HF_GUEST_MASK;

        self.update_cr0(0x6000_0010);
        self.a20_mask = !0;
        self.smbase = 0x30000;
        self.msr_smi_count = 0;

        self.idt.limit = 0xffff;
        self.gdt.limit = 0xffff;
        self.ldt.limit = 0xffff;
        self.ldt.flags = DESC_P_MASK | (2 << DESC_TYPE_SHIFT);
        self.tr.limit = 0xffff;
        self.tr.flags = DESC_P_MASK | (11 << DESC_TYPE_SHIFT);

        self.load_seg_cache(
            R_CS,
            0xf000,
            0xffff_0000,
            0xffff,
            DESC_P_MASK | DESC_S_MASK | DESC_CS_MASK | DESC_R_MASK | DESC_A_MASK,
        );
        let data = DESC_P_MASK | DESC_S_MASK | DESC_W_MASK | DESC_A_MASK;
        for seg in [R_DS, R_ES, R_SS, R_FS, R_GS] {
            self.load_seg_cache(seg, 0, 0, 0xffff, data);
        }

        self.rip = 0xfff0;
        self.regs[R_EDX] = u64::from(cfg.cpuid_version);

        self.rflags = RFLAGS_FIXED_1;

        self.fptags = [1; 8];
        self.fpuc = 0x37f;

        self.mxcsr = 0x1f80;
        self.xstate_bv = 0;

        self.pat = MSR_PAT_RESET;

        if cfg.kvm {
            // KVM takes a TSC of 0 to mean a hot-plugged CPU, so a reset of
            // a running CPU writes 1 instead.
            if self.tsc != 0 {
                self.tsc = 1;
            }
        } else {
            self.tsc = 0;
        }

        self.msr_ia32_misc_enable = misc_enable_reset(cfg.has_monitor);

        self.dr = [0; 8];
        self.dr[6] = DR6_FIXED_1;
        self.dr[7] = DR7_FIXED_1;

        self.xcr0 = XSTATE_FP_MASK;
        self.cr4 = 0;

        self.mtrr_deftype = 0;
        self.mtrr_var = [MtrrVar::default(); MTRR_VAR_COUNT];
        self.mtrr_fixed = [0; 11];

        self.interrupt_injected = -1;
        self.exception_nr = -1;
        self.exception_pending = false;
        self.exception_injected = false;
        self.exception_has_payload = false;
        self.exception_payload = 0;
        self.nmi_injected = false;
        self.triple_fault_pending = false;

        // apic_designate_bsp() followed by apic_reset_common(): the base
        // goes back to the default address with the APIC enabled.
        self.apic_base = APIC_DEFAULT_ADDRESS | MSR_IA32_APICBASE_ENABLE;
        if cfg.is_bsp {
            self.apic_base |= MSR_IA32_APICBASE_BSP;
        }
        self.halted = !cfg.is_bsp;

        if cfg.kvm {
            self.kvm_arch_reset_vcpu(cfg);
        }

        self.msr_ia32_sgxlepubkeyhash = SGX_LEPUBKEYHASH_DEFAULT;
        self.amd_tsc_scale_msr = MSR_AMD64_TSC_RATIO_DEFAULT;
    }

    /// The parts of `kvm_arch_reset_vcpu()` that touch this state.
    fn kvm_arch_reset_vcpu(&mut self, cfg: &ResetConfig) {
        self.xcr0 = 1;
        self.mp_state = if cfg.irqchip.in_kernel() && !cfg.is_bsp {
            MP_STATE_UNINITIALIZED
        } else {
            MP_STATE_RUNNABLE
        };
        // Hyper-V SynIC state is not modelled here.
        self.poll_control_msr = 1;
    }

    /// Port of `cpu_x86_update_cr0()` without the TLB flush.
    pub fn update_cr0(&mut self, new_cr0: u64) {
        if self.cr0 & CR0_PG_MASK == 0
            && new_cr0 & CR0_PG_MASK != 0
            && self.efer & MSR_EFER_LME != 0
        {
            if self.cr4 & CR4_PAE_MASK == 0 {
                return;
            }
            self.efer |= MSR_EFER_LMA;
            self.hflags |= HF_LMA_MASK;
        } else if self.cr0 & CR0_PG_MASK != 0
            && new_cr0 & CR0_PG_MASK == 0
            && self.efer & MSR_EFER_LMA != 0
        {
            self.efer &= !MSR_EFER_LMA;
            self.hflags &= !(HF_LMA_MASK | HF_CS64_MASK);
            self.rip &= 0xffff_ffff;
        }
        self.cr0 = new_cr0 | CR0_ET_MASK;

        let pe = (self.cr0 & CR0_PE_MASK) as u32;
        self.hflags = (self.hflags & !HF_PE_MASK) | (pe << 7);
        self.hflags |= (pe ^ 1) << 6;
        let fpu = HF_MP_MASK | HF_EM_MASK | HF_TS_MASK;
        self.hflags = (self.hflags & !fpu) | (((new_cr0 as u32) << 8) & fpu);
    }

    /// Port of `cpu_x86_load_seg_cache()` for ES to GS, including the
    /// `hflags` update. The BNDCS sync is not modelled.
    pub fn load_seg_cache(&mut self, seg: usize, selector: u32, base: u64, limit: u32, flags: u32) {
        self.segs[seg] = SegmentCache { selector, base, limit, flags };

        if seg == R_CS {
            if self.hflags & HF_LMA_MASK != 0 && flags & DESC_L_MASK != 0 {
                self.hflags |= HF_CS32_MASK | HF_SS32_MASK | HF_CS64_MASK;
                self.hflags &= !HF_ADDSEG_MASK;
            } else {
                let cs32 = (self.segs[R_CS].flags & DESC_B_MASK) >> (DESC_B_SHIFT - 4);
                self.hflags = (self.hflags & !(HF_CS32_MASK | HF_CS64_MASK)) | cs32;
            }
        }
        if seg == R_SS {
            let cpl = (flags >> DESC_DPL_SHIFT) & 3;
            self.hflags = (self.hflags & !HF_CPL_MASK) | cpl;
        }
        let mut new_hflags = (self.segs[R_SS].flags & DESC_B_MASK) >> (DESC_B_SHIFT - 5);
        // In long mode DS, ES and SS have a zero base, so ADDSEG stays clear.
        if self.hflags & HF_CS64_MASK == 0
            && (self.cr0 & CR0_PE_MASK == 0
                || self.rflags & (1 << 17) != 0
                || self.hflags & HF_CS32_MASK == 0
                || (self.segs[R_DS].base | self.segs[R_ES].base | self.segs[R_SS].base) != 0)
        {
            new_hflags |= HF_ADDSEG_MASK;
        }
        self.hflags = (self.hflags & !(HF_SS32_MASK | HF_ADDSEG_MASK)) | new_hflags;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::msr::MCG_CAP_DEFAULT;

    fn cfg(is_bsp: bool, kvm: bool) -> ResetConfig {
        ResetConfig {
            is_bsp,
            kvm,
            irqchip: IrqchipMode::Full,
            // qemu64: family 15, model 107, stepping 1.
            cpuid_version: 0x0006_0fb1,
            has_monitor: false,
        }
    }

    #[test]
    fn bsp_reset_values() {
        let s = X86CpuState::new_reset(&cfg(true, false));
        assert_eq!(s.rip, 0xfff0);
        assert_eq!(s.rflags, 2);
        assert_eq!(s.regs[R_EDX], 0x0006_0fb1);
        assert_eq!(s.regs[R_EAX], 0);
        assert_eq!(s.cr0, 0x6000_0010);
        assert_eq!(s.cr2, 0);
        assert_eq!(s.cr3, 0);
        assert_eq!(s.cr4, 0);
        assert_eq!(s.cr8, 0);
        assert_eq!(s.efer, 0);
        assert_eq!(s.xcr0, 1);
        assert_eq!(s.apic_base, 0xfee0_0900);
        assert_eq!(s.a20_mask, -1);
        assert_eq!(s.smbase, 0x30000);

        let cs = s.segs[R_CS];
        assert_eq!(cs.selector, 0xf000);
        assert_eq!(cs.base, 0xffff_0000);
        assert_eq!(cs.limit, 0xffff);
        assert_eq!(cs.flags, 0x9b00);
        for seg in [R_ES, R_SS, R_DS, R_FS, R_GS] {
            assert_eq!(
                s.segs[seg],
                SegmentCache { selector: 0, base: 0, limit: 0xffff, flags: 0x9300 }
            );
        }
        assert_eq!(s.ldt.limit, 0xffff);
        assert_eq!(s.ldt.flags, 0x8200);
        assert_eq!(s.tr.limit, 0xffff);
        assert_eq!(s.tr.flags, 0x8b00);
        assert_eq!((s.gdt.base, s.gdt.limit), (0, 0xffff));
        assert_eq!((s.idt.base, s.idt.limit), (0, 0xffff));

        assert_eq!(s.fptags, [1; 8]);
        assert_eq!(s.fpuc, 0x37f);
        assert_eq!(s.fpus, 0);
        assert_eq!(s.mxcsr, 0x1f80);
        assert_eq!(s.pat, 0x0007_0406_0007_0406);
        assert_eq!(s.msr_ia32_misc_enable, 1);
        assert_eq!(s.dr, [0, 0, 0, 0, 0, 0, 0xffff_0ff0, 0x400]);
        assert_eq!(s.tsc, 0);
        assert_eq!(s.interrupt_injected, -1);
        assert_eq!(s.exception_nr, -1);
        assert_eq!(s.old_exception, -1);
        assert_eq!(s.hflags2, HF2_GIF_MASK | HF2_VGIF_MASK);
        // Real mode: only ADDSEG is set.
        assert_eq!(s.hflags, HF_ADDSEG_MASK);
        assert!(!s.halted);
        assert_eq!(s.amd_tsc_scale_msr, 0x1_0000_0000);
        assert_eq!(s.msr_ia32_sgxlepubkeyhash[0], 0xa605_3e05_1270_b7ac);
        assert_eq!(s.poll_control_msr, 0);
    }

    #[test]
    fn ap_reset_under_kvm() {
        let mut s = X86CpuState {
            tsc: 123_456,
            mcg_cap: MCG_CAP_DEFAULT,
            mcg_ctl: u64::MAX,
            mcg_status: 5,
            mtrr_deftype: 0xc06,
            ..Default::default()
        };
        s.regs[R_EBX] = 7;
        let mut c = cfg(false, true);
        c.has_monitor = true;
        s.reset(&c);
        assert_eq!(s.tsc, 1);
        assert_eq!(s.apic_base, 0xfee0_0800);
        assert!(s.halted);
        assert_eq!(s.mp_state, MP_STATE_UNINITIALIZED);
        assert_eq!(s.poll_control_msr, 1);
        assert_eq!(s.msr_ia32_misc_enable, 0x40001);
        // Preserved across reset.
        assert_eq!(s.mcg_cap, MCG_CAP_DEFAULT);
        assert_eq!(s.mcg_ctl, u64::MAX);
        // Cleared by the memset or explicitly.
        assert_eq!(s.mcg_status, 0);
        assert_eq!(s.mtrr_deftype, 0);
        assert_eq!(s.regs[R_EBX], 0);

        // Without an in-kernel irqchip every vCPU starts runnable.
        c.irqchip = IrqchipMode::Off;
        s.reset(&c);
        assert_eq!(s.mp_state, MP_STATE_RUNNABLE);
    }

    #[test]
    fn fresh_kvm_tsc_stays_zero() {
        let s = X86CpuState::new_reset(&cfg(true, true));
        assert_eq!(s.tsc, 0);
        assert_eq!(s.mp_state, MP_STATE_RUNNABLE);
    }
}
