// SPDX-License-Identifier: GPL-2.0-or-later

//! Model-specific register numbers and the MSR list KVM loads at reset.
//!
//! The indices come from `target/i386/cpu.h` and, for the paravirtual KVM
//! registers, from `include/standard-headers/asm-x86/kvm_para.h`. This module
//! only has constants. The code that moves them in and out of a vCPU lives in
//! `ruvm-accel-kvm`.
//!
//! [`RESET_MSRS`] follows `kvm_put_msrs()` in `target/i386/kvm/kvm.c` when it
//! runs with `KVM_PUT_RESET_STATE` or higher. It keeps QEMU's order and says
//! for each entry when QEMU writes it. The values are what
//! `x86_cpu_reset_hold()` leaves in `CPUX86State`. Hyper-V, PMU counters,
//! Intel PT and arch LBR are not in the list. They depend on host probing and
//! on features the CPU models here do not turn on.

/// Time stamp counter.
pub const MSR_IA32_TSC: u32 = 0x10;
/// Local APIC base address and enable bits.
pub const MSR_IA32_APICBASE: u32 = 0x1b;
/// Set in [`MSR_IA32_APICBASE`] on the bootstrap processor.
pub const MSR_IA32_APICBASE_BSP: u64 = 1 << 8;
/// x2APIC mode enable in [`MSR_IA32_APICBASE`].
pub const MSR_IA32_APICBASE_EXTD: u64 = 1 << 10;
/// Global APIC enable in [`MSR_IA32_APICBASE`].
pub const MSR_IA32_APICBASE_ENABLE: u64 = 1 << 11;
/// Base address field of [`MSR_IA32_APICBASE`].
pub const MSR_IA32_APICBASE_BASE: u64 = 0xfffff << 12;
/// Default physical address of the local APIC (`APIC_DEFAULT_ADDRESS`).
pub const APIC_DEFAULT_ADDRESS: u64 = 0xfee0_0000;
/// Feature control (VMX and SGX locks).
pub const MSR_IA32_FEATURE_CONTROL: u32 = 0x3a;
/// TSC adjust.
pub const MSR_TSC_ADJUST: u32 = 0x3b;
/// Speculation control.
pub const MSR_IA32_SPEC_CTRL: u32 = 0x48;
/// Prediction command.
pub const MSR_IA32_PRED_CMD: u32 = 0x49;
/// SMI counter.
pub const MSR_SMI_COUNT: u32 = 0x34;
/// Core and thread count.
pub const MSR_CORE_THREAD_COUNT: u32 = 0x35;
/// Microcode revision.
pub const MSR_IA32_UCODE_REV: u32 = 0x8b;
/// SGX launch enclave public key hash, first of four.
pub const MSR_IA32_SGXLEPUBKEYHASH0: u32 = 0x8c;
/// SGX launch enclave public key hash, second of four.
pub const MSR_IA32_SGXLEPUBKEYHASH1: u32 = 0x8d;
/// SGX launch enclave public key hash, third of four.
pub const MSR_IA32_SGXLEPUBKEYHASH2: u32 = 0x8e;
/// SGX launch enclave public key hash, last of four.
pub const MSR_IA32_SGXLEPUBKEYHASH3: u32 = 0x8f;
/// SMRAM base.
pub const MSR_IA32_SMBASE: u32 = 0x9e;
/// First general purpose performance counter (Intel).
pub const MSR_P6_PERFCTR0: u32 = 0xc1;
/// Core capabilities.
pub const MSR_IA32_CORE_CAPABILITY: u32 = 0xcf;
/// UMWAIT control.
pub const MSR_IA32_UMWAIT_CONTROL: u32 = 0xe1;
/// MTRR capabilities.
pub const MSR_MTRRCAP: u32 = 0xfe;
/// Number of variable MTRRs QEMU exposes.
pub const MSR_MTRRCAP_VCNT: u32 = 8;
/// Fixed range MTRRs are supported.
pub const MSR_MTRRCAP_FIXRANGE_SUPPORT: u64 = 1 << 8;
/// Write combining memory type is supported.
pub const MSR_MTRRCAP_WC_SUPPORTED: u64 = 1 << 10;
/// MTRR enable bit in [`MSR_MTRRDEFTYPE`].
pub const MSR_MTRR_ENABLE: u64 = 1 << 11;
/// Architectural capabilities.
pub const MSR_IA32_ARCH_CAPABILITIES: u32 = 0x10a;
/// TSX control.
pub const MSR_IA32_TSX_CTRL: u32 = 0x122;
/// SYSENTER code segment.
pub const MSR_IA32_SYSENTER_CS: u32 = 0x174;
/// SYSENTER stack pointer.
pub const MSR_IA32_SYSENTER_ESP: u32 = 0x175;
/// SYSENTER instruction pointer.
pub const MSR_IA32_SYSENTER_EIP: u32 = 0x176;
/// Machine check capabilities.
pub const MSR_MCG_CAP: u32 = 0x179;
/// Machine check status.
pub const MSR_MCG_STATUS: u32 = 0x17a;
/// Machine check control.
pub const MSR_MCG_CTL: u32 = 0x17b;
/// First event select register (Intel).
pub const MSR_P6_EVNTSEL0: u32 = 0x186;
/// Miscellaneous enables.
pub const MSR_IA32_MISC_ENABLE: u32 = 0x1a0;
/// Fast string enable, the value QEMU starts with (`MSR_IA32_MISC_ENABLE_DEFAULT`).
pub const MSR_IA32_MISC_ENABLE_DEFAULT: u64 = 1;
/// MONITOR/MWAIT enable in [`MSR_IA32_MISC_ENABLE`].
pub const MSR_IA32_MISC_ENABLE_MWAIT: u64 = 1 << 18;
/// Extended feature disable.
pub const MSR_IA32_XFD: u32 = 0x1c4;
/// Extended feature disable error.
pub const MSR_IA32_XFD_ERR: u32 = 0x1c5;
/// FRED ring 0 stack pointer.
pub const MSR_IA32_FRED_RSP0: u32 = 0x1cc;
/// FRED stack level 1 stack pointer.
pub const MSR_IA32_FRED_RSP1: u32 = 0x1cd;
/// FRED stack level 2 stack pointer.
pub const MSR_IA32_FRED_RSP2: u32 = 0x1ce;
/// FRED stack level 3 stack pointer.
pub const MSR_IA32_FRED_RSP3: u32 = 0x1cf;
/// FRED exception stack levels.
pub const MSR_IA32_FRED_STKLVLS: u32 = 0x1d0;
/// FRED stack level 1 shadow stack pointer.
pub const MSR_IA32_FRED_SSP1: u32 = 0x1d1;
/// FRED stack level 2 shadow stack pointer.
pub const MSR_IA32_FRED_SSP2: u32 = 0x1d2;
/// FRED stack level 3 shadow stack pointer.
pub const MSR_IA32_FRED_SSP3: u32 = 0x1d3;
/// FRED configuration.
pub const MSR_IA32_FRED_CONFIG: u32 = 0x1d4;

/// Variable MTRR base register `n`.
pub const fn msr_mtrr_phys_base(n: u32) -> u32 {
    0x200 + 2 * n
}

/// Variable MTRR mask register `n`.
pub const fn msr_mtrr_phys_mask(n: u32) -> u32 {
    0x200 + 2 * n + 1
}

/// Fixed MTRR for 0x00000 to 0x7ffff.
pub const MSR_MTRRFIX64K_00000: u32 = 0x250;
/// Fixed MTRR for 0x80000 to 0x9ffff.
pub const MSR_MTRRFIX16K_80000: u32 = 0x258;
/// Fixed MTRR for 0xa0000 to 0xbffff.
pub const MSR_MTRRFIX16K_A0000: u32 = 0x259;
/// Fixed MTRR for 0xc0000 to 0xc7fff.
pub const MSR_MTRRFIX4K_C0000: u32 = 0x268;
/// Fixed MTRR for 0xc8000 to 0xcffff.
pub const MSR_MTRRFIX4K_C8000: u32 = 0x269;
/// Fixed MTRR for 0xd0000 to 0xd7fff.
pub const MSR_MTRRFIX4K_D0000: u32 = 0x26a;
/// Fixed MTRR for 0xd8000 to 0xdffff.
pub const MSR_MTRRFIX4K_D8000: u32 = 0x26b;
/// Fixed MTRR for 0xe0000 to 0xe7fff.
pub const MSR_MTRRFIX4K_E0000: u32 = 0x26c;
/// Fixed MTRR for 0xe8000 to 0xeffff.
pub const MSR_MTRRFIX4K_E8000: u32 = 0x26d;
/// Fixed MTRR for 0xf0000 to 0xf7fff.
pub const MSR_MTRRFIX4K_F0000: u32 = 0x26e;
/// Fixed MTRR for 0xf8000 to 0xfffff.
pub const MSR_MTRRFIX4K_F8000: u32 = 0x26f;
/// The eleven fixed range MTRRs in the order of `env->mtrr_fixed[]`.
pub const MTRR_FIXED_MSRS: [u32; 11] = [
    MSR_MTRRFIX64K_00000,
    MSR_MTRRFIX16K_80000,
    MSR_MTRRFIX16K_A0000,
    MSR_MTRRFIX4K_C0000,
    MSR_MTRRFIX4K_C8000,
    MSR_MTRRFIX4K_D0000,
    MSR_MTRRFIX4K_D8000,
    MSR_MTRRFIX4K_E0000,
    MSR_MTRRFIX4K_E8000,
    MSR_MTRRFIX4K_F0000,
    MSR_MTRRFIX4K_F8000,
];
/// Page attribute table.
pub const MSR_PAT: u32 = 0x277;
/// Reset value of [`MSR_PAT`].
pub const MSR_PAT_RESET: u64 = 0x0007_0406_0007_0406;
/// Default MTRR memory type.
pub const MSR_MTRRDEFTYPE: u32 = 0x2ff;
/// First fixed-function performance counter.
pub const MSR_CORE_PERF_FIXED_CTR0: u32 = 0x309;
/// Performance capabilities.
pub const MSR_IA32_PERF_CAPABILITIES: u32 = 0x345;
/// Fixed-function counter control.
pub const MSR_CORE_PERF_FIXED_CTR_CTRL: u32 = 0x38d;
/// Global performance counter status.
pub const MSR_CORE_PERF_GLOBAL_STATUS: u32 = 0x38e;
/// Global performance counter control.
pub const MSR_CORE_PERF_GLOBAL_CTRL: u32 = 0x38f;
/// Global performance counter overflow control.
pub const MSR_CORE_PERF_GLOBAL_OVF_CTRL: u32 = 0x390;
/// First machine check bank control register.
pub const MSR_MC0_CTL: u32 = 0x400;
/// Machine check status of bank 0.
pub const MSR_MC0_STATUS: u32 = 0x401;
/// Machine check address of bank 0.
pub const MSR_MC0_ADDR: u32 = 0x402;
/// Machine check misc of bank 0.
pub const MSR_MC0_MISC: u32 = 0x403;
/// Extended machine check control.
pub const MSR_MCG_EXT_CTL: u32 = 0x4d0;
/// `MCG_CAP.MCG_CTL_P`: the `MCG_CTL` register exists.
pub const MCG_CTL_P: u64 = 1 << 8;
/// `MCG_CAP.MCG_SER_P`: software error recovery.
pub const MCG_SER_P: u64 = 1 << 24;
/// `MCG_CAP.MCG_LMCE_P`: local machine check.
pub const MCG_LMCE_P: u64 = 1 << 27;
/// Number of machine check banks QEMU gives a guest (`MCE_BANKS_DEF`).
pub const MCE_BANKS_DEF: u64 = 10;
/// `MCG_CAP` that `mce_init()` sets for family 6 and later with MCE and MCA.
///
/// KVM trims this to what the host supports when the vCPU is created.
pub const MCG_CAP_DEFAULT: u64 = MCG_CTL_P | MCG_SER_P | MCE_BANKS_DEF;

/// VMX basic information.
pub const MSR_IA32_VMX_BASIC: u32 = 0x480;
/// VMX pin-based controls.
pub const MSR_IA32_VMX_PINBASED_CTLS: u32 = 0x481;
/// VMX processor-based controls.
pub const MSR_IA32_VMX_PROCBASED_CTLS: u32 = 0x482;
/// VMX exit controls.
pub const MSR_IA32_VMX_EXIT_CTLS: u32 = 0x483;
/// VMX entry controls.
pub const MSR_IA32_VMX_ENTRY_CTLS: u32 = 0x484;
/// VMX miscellaneous data.
pub const MSR_IA32_VMX_MISC: u32 = 0x485;
/// VMX CR0 fixed-0 bits.
pub const MSR_IA32_VMX_CR0_FIXED0: u32 = 0x486;
/// VMX CR0 fixed-1 bits.
pub const MSR_IA32_VMX_CR0_FIXED1: u32 = 0x487;
/// VMX CR4 fixed-0 bits.
pub const MSR_IA32_VMX_CR4_FIXED0: u32 = 0x488;
/// VMX CR4 fixed-1 bits.
pub const MSR_IA32_VMX_CR4_FIXED1: u32 = 0x489;
/// VMCS enumeration.
pub const MSR_IA32_VMX_VMCS_ENUM: u32 = 0x48a;
/// VMX secondary processor-based controls.
pub const MSR_IA32_VMX_PROCBASED_CTLS2: u32 = 0x48b;
/// VMX EPT and VPID capabilities.
pub const MSR_IA32_VMX_EPT_VPID_CAP: u32 = 0x48c;
/// VMX true pin-based controls.
pub const MSR_IA32_VMX_TRUE_PINBASED_CTLS: u32 = 0x48d;
/// VMX true processor-based controls.
pub const MSR_IA32_VMX_TRUE_PROCBASED_CTLS: u32 = 0x48e;
/// VMX true exit controls.
pub const MSR_IA32_VMX_TRUE_EXIT_CTLS: u32 = 0x48f;
/// VMX true entry controls.
pub const MSR_IA32_VMX_TRUE_ENTRY_CTLS: u32 = 0x490;
/// VM functions.
pub const MSR_IA32_VMX_VMFUNC: u32 = 0x491;

/// User mode CET configuration.
pub const MSR_IA32_U_CET: u32 = 0x6a0;
/// Supervisor mode CET configuration.
pub const MSR_IA32_S_CET: u32 = 0x6a2;
/// Ring 0 shadow stack pointer.
pub const MSR_IA32_PL0_SSP: u32 = 0x6a4;
/// Ring 1 shadow stack pointer.
pub const MSR_IA32_PL1_SSP: u32 = 0x6a5;
/// Ring 2 shadow stack pointer.
pub const MSR_IA32_PL2_SSP: u32 = 0x6a6;
/// Ring 3 shadow stack pointer.
pub const MSR_IA32_PL3_SSP: u32 = 0x6a7;
/// Interrupt shadow stack table address.
pub const MSR_IA32_INT_SSP_TAB: u32 = 0x6a8;
/// TSC deadline timer.
pub const MSR_IA32_TSCDEADLINE: u32 = 0x6e0;
/// Protection keys for supervisor pages.
pub const MSR_IA32_PKRS: u32 = 0x6e1;
/// First x2APIC register.
pub const MSR_X2APIC_BASE: u32 = 0x800;
/// Last x2APIC register.
pub const MSR_X2APIC_END: u32 = 0x8ff;
/// MPX bounds configuration.
pub const MSR_IA32_BNDCFGS: u32 = 0xd90;
/// Extended supervisor state mask.
pub const MSR_IA32_XSS: u32 = 0xda0;

/// Extended feature enable register.
pub const MSR_EFER: u32 = 0xc000_0080;
/// SYSCALL target and segments (legacy mode).
pub const MSR_STAR: u32 = 0xc000_0081;
/// SYSCALL target in 64-bit mode.
pub const MSR_LSTAR: u32 = 0xc000_0082;
/// SYSCALL target in compatibility mode.
pub const MSR_CSTAR: u32 = 0xc000_0083;
/// SYSCALL flag mask.
pub const MSR_FMASK: u32 = 0xc000_0084;
/// FS base.
pub const MSR_FSBASE: u32 = 0xc000_0100;
/// GS base.
pub const MSR_GSBASE: u32 = 0xc000_0101;
/// Kernel GS base for SWAPGS.
pub const MSR_KERNELGSBASE: u32 = 0xc000_0102;
/// Auxiliary TSC value for RDTSCP.
pub const MSR_TSC_AUX: u32 = 0xc000_0103;
/// AMD TSC ratio.
pub const MSR_AMD64_TSC_RATIO: u32 = 0xc000_0104;
/// Reset value of [`MSR_AMD64_TSC_RATIO`] (a ratio of 1.0).
pub const MSR_AMD64_TSC_RATIO_DEFAULT: u64 = 0x1_0000_0000;
/// First legacy AMD event select register.
pub const MSR_K7_EVNTSEL0: u32 = 0xc001_0000;
/// First legacy AMD performance counter.
pub const MSR_K7_PERFCTR0: u32 = 0xc001_0004;
/// AMD hardware configuration.
pub const MSR_K7_HWCR: u32 = 0xc001_0015;
/// SVM host save area.
pub const MSR_VM_HSAVE_PA: u32 = 0xc001_0117;
/// AMD virtual speculative store bypass disable.
pub const MSR_VIRT_SSBD: u32 = 0xc001_011f;
/// First family 15h event select register.
pub const MSR_F15H_PERF_CTL0: u32 = 0xc001_0200;
/// First family 15h performance counter.
pub const MSR_F15H_PERF_CTR0: u32 = 0xc001_0201;

/// kvmclock wall clock, old number. QEMU writes this one.
pub const MSR_KVM_WALL_CLOCK: u32 = 0x11;
/// kvmclock system time, old number. QEMU writes this one.
pub const MSR_KVM_SYSTEM_TIME: u32 = 0x12;
/// kvmclock wall clock, new number.
pub const MSR_KVM_WALL_CLOCK_NEW: u32 = 0x4b56_4d00;
/// kvmclock system time, new number.
pub const MSR_KVM_SYSTEM_TIME_NEW: u32 = 0x4b56_4d01;
/// Asynchronous page fault enable.
pub const MSR_KVM_ASYNC_PF_EN: u32 = 0x4b56_4d02;
/// Steal time area.
pub const MSR_KVM_STEAL_TIME: u32 = 0x4b56_4d03;
/// Paravirtual end of interrupt.
pub const MSR_KVM_PV_EOI_EN: u32 = 0x4b56_4d04;
/// Host-side halt polling control.
pub const MSR_KVM_POLL_CONTROL: u32 = 0x4b56_4d05;
/// Asynchronous page fault interrupt vector.
pub const MSR_KVM_ASYNC_PF_INT: u32 = 0x4b56_4d06;
/// Asynchronous page fault acknowledge.
pub const MSR_KVM_ASYNC_PF_ACK: u32 = 0x4b56_4d07;
/// Migration control.
pub const MSR_KVM_MIGRATION_CONTROL: u32 = 0x4b56_4d08;

/// The SGX launch enclave hash that QEMU loads at reset.
///
/// These are the Skylake hardware defaults from `x86_cpu_set_sgxlepubkeyhash()`.
pub const SGX_LEPUBKEYHASH_DEFAULT: [u64; 4] =
    [0xa605_3e05_1270_b7ac, 0x6cfb_e8ba_8b3b_413d, 0xc491_6d99_f2b3_735d, 0xd4f8_c059_09f9_bb3b];

/// When `kvm_put_msrs()` includes an MSR.
///
/// The `Host*` gates are the `has_msr_*` flags QEMU fills in from
/// `KVM_GET_MSR_INDEX_LIST`. The feature gates test the guest CPUID words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsrGate {
    /// Written every time.
    Always,
    /// Written when the host lists this MSR index.
    HostHas(u32),
    /// Written on a host kernel that runs 64-bit guests (`lm_capable_kernel`).
    LongModeKernel,
    /// Long mode kernel and the guest has FRED (`CPUID[7,1].EAX` bit 17).
    Fred,
    /// FRED and no CET shadow stack in the guest, for `MSR_IA32_PL0_SSP`.
    FredWithoutShadowStack,
    /// The guest has `kvmclock` in `CPUID[0x40000001].EAX` (bit 0 or 3).
    KvmClock,
    /// The guest has `kvm-asyncpf-int`.
    KvmAsyncPfInt,
    /// The guest has `kvm-asyncpf`.
    KvmAsyncPf,
    /// The guest has `kvm-pv-eoi`.
    KvmPvEoi,
    /// The guest has `kvm-steal-time`.
    KvmStealTime,
    /// The guest has `kvm-poll-control`.
    KvmPollControl,
    /// The guest has `mtrr` in `CPUID[1].EDX`.
    Mtrr,
    /// The guest has `sgxlc` in `CPUID[7,0].ECX`.
    SgxLc,
    /// The guest has `xfd` in `CPUID[0xd,1].EAX`.
    Xfd,
    /// `MCG_CAP` is nonzero.
    Mce,
    /// `MCG_CAP` is nonzero and the host has `MSR_MCG_EXT_CTL`.
    MceExtCtl,
    /// The guest has CET shadow stack or indirect branch tracking.
    Cet,
    /// The guest has CET shadow stack.
    CetShadowStack,
    /// CET shadow stack on a long mode kernel.
    CetShadowStackLongMode,
}

/// One MSR write in the reset-level `kvm_put_msrs()` sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResetMsr {
    /// The MSR index.
    pub index: u32,
    /// The value after `x86_cpu_reset_hold()`.
    pub value: u64,
    /// When QEMU includes it.
    pub gate: MsrGate,
    /// True when QEMU only writes it at `KVM_PUT_RESET_STATE` or above, false
    /// when it goes out on every register sync.
    pub reset_only: bool,
}

const fn m(index: u32, value: u64, gate: MsrGate, reset_only: bool) -> ResetMsr {
    ResetMsr { index, value, gate, reset_only }
}

/// The MSRs `kvm_put_msrs()` writes at reset level, in QEMU's order.
///
/// A few values depend on the CPU and are given here for the common case:
///
/// - `MSR_IA32_TSC` is 0. Under KVM QEMU writes 1 instead if the TSC was
///   already running, so KVM does not treat the reset as a hotplug.
/// - `MSR_IA32_MISC_ENABLE` is 1, plus [`MSR_IA32_MISC_ENABLE_MWAIT`] when
///   the guest has `monitor`. See [`misc_enable_reset`].
/// - `MSR_MCG_CTL` is all ones because `mce_init()` sets it at realize time
///   and reset does not clear it.
/// - The variable MTRR masks are written after masking with the physical
///   address width, which leaves them at 0.
///
/// The machine check bank registers follow `MSR_MCG_EXT_CTL` in QEMU. They
/// are left out of the table because their number comes from `MCG_CAP`; use
/// [`mce_bank_reset_value`] for them.
pub const RESET_MSRS: &[ResetMsr] = &[
    m(MSR_IA32_SYSENTER_CS, 0, MsrGate::Always, false),
    m(MSR_IA32_SYSENTER_ESP, 0, MsrGate::Always, false),
    m(MSR_IA32_SYSENTER_EIP, 0, MsrGate::Always, false),
    m(MSR_PAT, MSR_PAT_RESET, MsrGate::Always, false),
    m(MSR_STAR, 0, MsrGate::HostHas(MSR_STAR), false),
    m(MSR_VM_HSAVE_PA, 0, MsrGate::HostHas(MSR_VM_HSAVE_PA), false),
    m(MSR_TSC_AUX, 0, MsrGate::HostHas(MSR_TSC_AUX), false),
    m(MSR_TSC_ADJUST, 0, MsrGate::HostHas(MSR_TSC_ADJUST), false),
    m(
        MSR_IA32_MISC_ENABLE,
        MSR_IA32_MISC_ENABLE_DEFAULT,
        MsrGate::HostHas(MSR_IA32_MISC_ENABLE),
        false,
    ),
    m(MSR_IA32_SMBASE, 0x30000, MsrGate::HostHas(MSR_IA32_SMBASE), false),
    m(MSR_SMI_COUNT, 0, MsrGate::HostHas(MSR_SMI_COUNT), false),
    m(MSR_IA32_PKRS, 0, MsrGate::HostHas(MSR_IA32_PKRS), false),
    m(MSR_IA32_BNDCFGS, 0, MsrGate::HostHas(MSR_IA32_BNDCFGS), false),
    m(MSR_IA32_XSS, 0, MsrGate::HostHas(MSR_IA32_XSS), false),
    m(MSR_IA32_UMWAIT_CONTROL, 0, MsrGate::HostHas(MSR_IA32_UMWAIT_CONTROL), false),
    m(MSR_IA32_SPEC_CTRL, 0, MsrGate::HostHas(MSR_IA32_SPEC_CTRL), false),
    m(
        MSR_AMD64_TSC_RATIO,
        MSR_AMD64_TSC_RATIO_DEFAULT,
        MsrGate::HostHas(MSR_AMD64_TSC_RATIO),
        false,
    ),
    m(MSR_IA32_TSX_CTRL, 0, MsrGate::HostHas(MSR_IA32_TSX_CTRL), false),
    m(MSR_VIRT_SSBD, 0, MsrGate::HostHas(MSR_VIRT_SSBD), false),
    m(MSR_K7_HWCR, 0, MsrGate::HostHas(MSR_K7_HWCR), false),
    m(MSR_CSTAR, 0, MsrGate::LongModeKernel, false),
    m(MSR_KERNELGSBASE, 0, MsrGate::LongModeKernel, false),
    m(MSR_FMASK, 0, MsrGate::LongModeKernel, false),
    m(MSR_LSTAR, 0, MsrGate::LongModeKernel, false),
    m(MSR_IA32_FRED_RSP0, 0, MsrGate::Fred, false),
    m(MSR_IA32_FRED_RSP1, 0, MsrGate::Fred, false),
    m(MSR_IA32_FRED_RSP2, 0, MsrGate::Fred, false),
    m(MSR_IA32_FRED_RSP3, 0, MsrGate::Fred, false),
    m(MSR_IA32_FRED_STKLVLS, 0, MsrGate::Fred, false),
    m(MSR_IA32_FRED_SSP1, 0, MsrGate::Fred, false),
    m(MSR_IA32_FRED_SSP2, 0, MsrGate::Fred, false),
    m(MSR_IA32_FRED_SSP3, 0, MsrGate::Fred, false),
    m(MSR_IA32_FRED_CONFIG, 0, MsrGate::Fred, false),
    m(MSR_IA32_PL0_SSP, 0, MsrGate::FredWithoutShadowStack, false),
    // From here on QEMU only writes at KVM_PUT_RESET_STATE and above.
    m(MSR_IA32_TSC, 0, MsrGate::Always, true),
    m(MSR_KVM_SYSTEM_TIME, 0, MsrGate::KvmClock, true),
    m(MSR_KVM_WALL_CLOCK, 0, MsrGate::KvmClock, true),
    m(MSR_KVM_ASYNC_PF_INT, 0, MsrGate::KvmAsyncPfInt, true),
    m(MSR_KVM_ASYNC_PF_EN, 0, MsrGate::KvmAsyncPf, true),
    m(MSR_KVM_PV_EOI_EN, 0, MsrGate::KvmPvEoi, true),
    m(MSR_KVM_STEAL_TIME, 0, MsrGate::KvmStealTime, true),
    // kvm_arch_reset_vcpu() turns host polling on by default.
    m(MSR_KVM_POLL_CONTROL, 1, MsrGate::KvmPollControl, true),
    m(MSR_MTRRDEFTYPE, 0, MsrGate::Mtrr, true),
    m(MSR_MTRRFIX64K_00000, 0, MsrGate::Mtrr, true),
    m(MSR_MTRRFIX16K_80000, 0, MsrGate::Mtrr, true),
    m(MSR_MTRRFIX16K_A0000, 0, MsrGate::Mtrr, true),
    m(MSR_MTRRFIX4K_C0000, 0, MsrGate::Mtrr, true),
    m(MSR_MTRRFIX4K_C8000, 0, MsrGate::Mtrr, true),
    m(MSR_MTRRFIX4K_D0000, 0, MsrGate::Mtrr, true),
    m(MSR_MTRRFIX4K_D8000, 0, MsrGate::Mtrr, true),
    m(MSR_MTRRFIX4K_E0000, 0, MsrGate::Mtrr, true),
    m(MSR_MTRRFIX4K_E8000, 0, MsrGate::Mtrr, true),
    m(MSR_MTRRFIX4K_F0000, 0, MsrGate::Mtrr, true),
    m(MSR_MTRRFIX4K_F8000, 0, MsrGate::Mtrr, true),
    m(msr_mtrr_phys_base(0), 0, MsrGate::Mtrr, true),
    m(msr_mtrr_phys_mask(0), 0, MsrGate::Mtrr, true),
    m(msr_mtrr_phys_base(1), 0, MsrGate::Mtrr, true),
    m(msr_mtrr_phys_mask(1), 0, MsrGate::Mtrr, true),
    m(msr_mtrr_phys_base(2), 0, MsrGate::Mtrr, true),
    m(msr_mtrr_phys_mask(2), 0, MsrGate::Mtrr, true),
    m(msr_mtrr_phys_base(3), 0, MsrGate::Mtrr, true),
    m(msr_mtrr_phys_mask(3), 0, MsrGate::Mtrr, true),
    m(msr_mtrr_phys_base(4), 0, MsrGate::Mtrr, true),
    m(msr_mtrr_phys_mask(4), 0, MsrGate::Mtrr, true),
    m(msr_mtrr_phys_base(5), 0, MsrGate::Mtrr, true),
    m(msr_mtrr_phys_mask(5), 0, MsrGate::Mtrr, true),
    m(msr_mtrr_phys_base(6), 0, MsrGate::Mtrr, true),
    m(msr_mtrr_phys_mask(6), 0, MsrGate::Mtrr, true),
    m(msr_mtrr_phys_base(7), 0, MsrGate::Mtrr, true),
    m(msr_mtrr_phys_mask(7), 0, MsrGate::Mtrr, true),
    m(MSR_IA32_SGXLEPUBKEYHASH0, SGX_LEPUBKEYHASH_DEFAULT[0], MsrGate::SgxLc, true),
    m(MSR_IA32_SGXLEPUBKEYHASH1, SGX_LEPUBKEYHASH_DEFAULT[1], MsrGate::SgxLc, true),
    m(MSR_IA32_SGXLEPUBKEYHASH2, SGX_LEPUBKEYHASH_DEFAULT[2], MsrGate::SgxLc, true),
    m(MSR_IA32_SGXLEPUBKEYHASH3, SGX_LEPUBKEYHASH_DEFAULT[3], MsrGate::SgxLc, true),
    m(MSR_IA32_XFD, 0, MsrGate::Xfd, true),
    m(MSR_IA32_XFD_ERR, 0, MsrGate::Xfd, true),
    // Back to every sync.
    m(MSR_MCG_STATUS, 0, MsrGate::Mce, false),
    m(MSR_MCG_CTL, u64::MAX, MsrGate::Mce, false),
    m(MSR_MCG_EXT_CTL, 0, MsrGate::MceExtCtl, false),
    m(MSR_IA32_U_CET, 0, MsrGate::Cet, false),
    m(MSR_IA32_S_CET, 0, MsrGate::Cet, false),
    m(MSR_IA32_PL0_SSP, 0, MsrGate::CetShadowStack, false),
    m(MSR_IA32_PL1_SSP, 0, MsrGate::CetShadowStack, false),
    m(MSR_IA32_PL2_SSP, 0, MsrGate::CetShadowStack, false),
    m(MSR_IA32_PL3_SSP, 0, MsrGate::CetShadowStack, false),
    m(MSR_IA32_INT_SSP_TAB, 0, MsrGate::CetShadowStackLongMode, false),
];

/// `MSR_IA32_MISC_ENABLE` after reset, as `x86_cpu_reset_hold()` computes it.
pub const fn misc_enable_reset(has_monitor: bool) -> u64 {
    if has_monitor {
        MSR_IA32_MISC_ENABLE_DEFAULT | MSR_IA32_MISC_ENABLE_MWAIT
    } else {
        MSR_IA32_MISC_ENABLE_DEFAULT
    }
}

/// Reset value of machine check bank register `MSR_MC0_CTL + i`.
///
/// Each bank has four registers (CTL, STATUS, ADDR, MISC). `mce_init()` sets
/// every CTL to all ones and leaves the others at 0.
pub const fn mce_bank_reset_value(i: u32) -> u64 {
    if i % 4 == 0 { u64::MAX } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mtrr_numbering_matches_cpu_h() {
        assert_eq!(msr_mtrr_phys_base(0), 0x200);
        assert_eq!(msr_mtrr_phys_mask(0), 0x201);
        assert_eq!(msr_mtrr_phys_base(7), 0x20e);
        assert_eq!(msr_mtrr_phys_mask(7), 0x20f);
    }

    #[test]
    fn reset_list_values() {
        let find = |index: u32| RESET_MSRS.iter().find(|e| e.index == index).copied();
        assert_eq!(find(MSR_PAT).map(|e| e.value), Some(0x0007_0406_0007_0406));
        assert_eq!(find(MSR_IA32_SMBASE).map(|e| e.value), Some(0x30000));
        assert_eq!(find(MSR_AMD64_TSC_RATIO).map(|e| e.value), Some(0x1_0000_0000));
        let tsc = find(MSR_IA32_TSC).unwrap();
        assert!(tsc.reset_only);
        assert_eq!(tsc.value, 0);
        assert_eq!(find(MSR_KVM_POLL_CONTROL).map(|e| e.value), Some(1));
        // Eleven fixed MTRRs, eight base/mask pairs and the default type.
        let mtrr = RESET_MSRS.iter().filter(|e| e.gate == MsrGate::Mtrr).count();
        assert_eq!(mtrr, 11 + 16 + 1);
        assert_eq!(misc_enable_reset(false), 1);
        assert_eq!(misc_enable_reset(true), 0x40001);
        assert_eq!(MCG_CAP_DEFAULT, 0x0100_010a);
        assert_eq!(mce_bank_reset_value(0), u64::MAX);
        assert_eq!(mce_bank_reset_value(1), 0);
    }
}
