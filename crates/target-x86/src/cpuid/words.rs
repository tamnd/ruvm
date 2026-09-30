// SPDX-License-Identifier: GPL-2.0-or-later

//! Feature words: the CPUID registers and feature MSRs that make up a CPU
//! model, with their bit names.
//!
//! This is `FeatureWord` from `target/i386/cpu.h` and `feature_word_info[]`
//! and `feature_dependencies[]` from `target/i386/cpu.c`. The tables below
//! were produced from the QEMU 11.1 source by compiling its initialisers, so
//! the TCG masks and bit names match QEMU exactly.

use crate::msr::{
    MSR_IA32_ARCH_CAPABILITIES, MSR_IA32_CORE_CAPABILITY, MSR_IA32_PERF_CAPABILITIES,
    MSR_IA32_VMX_BASIC, MSR_IA32_VMX_EPT_VPID_CAP, MSR_IA32_VMX_MISC, MSR_IA32_VMX_PROCBASED_CTLS2,
    MSR_IA32_VMX_TRUE_ENTRY_CTLS, MSR_IA32_VMX_TRUE_EXIT_CTLS, MSR_IA32_VMX_TRUE_PINBASED_CTLS,
    MSR_IA32_VMX_TRUE_PROCBASED_CTLS, MSR_IA32_VMX_VMFUNC,
};

/// `CPUID[1].EDX`.
pub const FEAT_1_EDX: usize = 0;
/// `CPUID[1].ECX`.
pub const FEAT_1_ECX: usize = 1;
/// `CPUID[EAX=7,ECX=0].EBX`.
pub const FEAT_7_0_EBX: usize = 2;
/// `CPUID[EAX=7,ECX=0].ECX`.
pub const FEAT_7_0_ECX: usize = 3;
/// `CPUID[EAX=7,ECX=0].EDX`.
pub const FEAT_7_0_EDX: usize = 4;
/// `CPUID[EAX=7,ECX=1].EAX`.
pub const FEAT_7_1_EAX: usize = 5;
/// `CPUID[8000_0001].EDX`.
pub const FEAT_8000_0001_EDX: usize = 6;
/// `CPUID[8000_0001].ECX`.
pub const FEAT_8000_0001_ECX: usize = 7;
/// `CPUID[8000_0007].EBX`.
pub const FEAT_8000_0007_EBX: usize = 8;
/// `CPUID[8000_0007].EDX`.
pub const FEAT_8000_0007_EDX: usize = 9;
/// `CPUID[8000_0008].EBX`.
pub const FEAT_8000_0008_EBX: usize = 10;
/// `CPUID[8000_0021].EAX`.
pub const FEAT_8000_0021_EAX: usize = 11;
/// `CPUID[8000_0021].EBX`.
pub const FEAT_8000_0021_EBX: usize = 12;
/// `CPUID[8000_0021].ECX`.
pub const FEAT_8000_0021_ECX: usize = 13;
/// `CPUID[8000_0022].EAX`.
pub const FEAT_8000_0022_EAX: usize = 14;
/// `CPUID[C000_0001].EDX`.
pub const FEAT_C000_0001_EDX: usize = 15;
/// `CPUID[4000_0001].EAX` (KVM paravirt features).
pub const FEAT_KVM: usize = 16;
/// `CPUID[4000_0001].EDX` (KVM hints).
pub const FEAT_KVM_HINTS: usize = 17;
/// `CPUID[8000_000A].EDX`.
pub const FEAT_SVM: usize = 18;
/// `CPUID[EAX=0xd,ECX=1].EAX`.
pub const FEAT_XSAVE: usize = 19;
/// `CPUID[6].EAX`.
pub const FEAT_6_EAX: usize = 20;
/// `CPUID[EAX=0xd,ECX=0].EAX`.
pub const FEAT_XSAVE_XCR0_LO: usize = 21;
/// `CPUID[EAX=0xd,ECX=0].EDX`.
pub const FEAT_XSAVE_XCR0_HI: usize = 22;
/// `MSR_IA32_ARCH_CAPABILITIES`.
pub const FEAT_ARCH_CAPABILITIES: usize = 23;
/// `MSR_IA32_CORE_CAPABILITY`.
pub const FEAT_CORE_CAPABILITY: usize = 24;
/// `MSR_IA32_PERF_CAPABILITIES`.
pub const FEAT_PERF_CAPABILITIES: usize = 25;
/// `MSR_IA32_VMX_TRUE_PROCBASED_CTLS`.
pub const FEAT_VMX_PROCBASED_CTLS: usize = 26;
/// `MSR_IA32_VMX_PROCBASED_CTLS2`.
pub const FEAT_VMX_SECONDARY_CTLS: usize = 27;
/// `MSR_IA32_VMX_TRUE_PINBASED_CTLS`.
pub const FEAT_VMX_PINBASED_CTLS: usize = 28;
/// `MSR_IA32_VMX_TRUE_EXIT_CTLS`.
pub const FEAT_VMX_EXIT_CTLS: usize = 29;
/// `MSR_IA32_VMX_TRUE_ENTRY_CTLS`.
pub const FEAT_VMX_ENTRY_CTLS: usize = 30;
/// `MSR_IA32_VMX_MISC`.
pub const FEAT_VMX_MISC: usize = 31;
/// `MSR_IA32_VMX_EPT_VPID_CAP`.
pub const FEAT_VMX_EPT_VPID_CAPS: usize = 32;
/// `MSR_IA32_VMX_BASIC`.
pub const FEAT_VMX_BASIC: usize = 33;
/// `MSR_IA32_VMX_VMFUNC`.
pub const FEAT_VMX_VMFUNC: usize = 34;
/// `CPUID[EAX=0x14,ECX=0].ECX`.
pub const FEAT_14_0_ECX: usize = 35;
/// `CPUID[EAX=0x12,ECX=0].EAX`.
pub const FEAT_SGX_12_0_EAX: usize = 36;
/// `CPUID[EAX=0x12,ECX=0].EBX`.
pub const FEAT_SGX_12_0_EBX: usize = 37;
/// `CPUID[EAX=0x12,ECX=1].EAX`.
pub const FEAT_SGX_12_1_EAX: usize = 38;
/// `CPUID[EAX=0xd,ECX=1].ECX`.
pub const FEAT_XSAVE_XSS_LO: usize = 39;
/// `CPUID[EAX=0xd,ECX=1].EDX`.
pub const FEAT_XSAVE_XSS_HI: usize = 40;
/// `CPUID[EAX=7,ECX=1].ECX`.
pub const FEAT_7_1_ECX: usize = 41;
/// `CPUID[EAX=7,ECX=1].EDX`.
pub const FEAT_7_1_EDX: usize = 42;
/// `CPUID[EAX=7,ECX=2].EDX`.
pub const FEAT_7_2_EDX: usize = 43;
/// `CPUID[EAX=0x24,ECX=0].EBX`.
pub const FEAT_24_0_EBX: usize = 44;
/// `CPUID[EAX=0x29,ECX=0].EBX`.
pub const FEAT_29_0_EBX: usize = 45;
/// `CPUID[EAX=0x1e,ECX=1].EAX`.
pub const FEAT_1E_1_EAX: usize = 46;
/// `CPUID[EAX=0x24,ECX=1].ECX`. QEMU's comment says ECX=0, but the table
/// entry uses subleaf 1.
pub const FEAT_24_1_ECX: usize = 47;
/// Number of feature words.
pub const FEATURE_WORDS: usize = 48;

/// One value per feature word, as `FeatureWordArray`.
pub type FeatureWordArray = [u64; FEATURE_WORDS];

/// `KVM_CPUID_FEATURES`, the KVM paravirt feature leaf.
pub const KVM_CPUID_FEATURES: u32 = 0x4000_0001;

/// A CPUID output register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reg {
    /// EAX.
    Eax,
    /// EBX.
    Ebx,
    /// ECX.
    Ecx,
    /// EDX.
    Edx,
}

impl Reg {
    /// The name QEMU prints (`get_register_name_32`).
    pub fn name(self) -> &'static str {
        match self {
            Reg::Eax => "EAX",
            Reg::Ebx => "EBX",
            Reg::Ecx => "ECX",
            Reg::Edx => "EDX",
        }
    }
}

/// Where a feature word lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WordKind {
    /// A CPUID register. `ecx` is `Some` when the leaf takes a subleaf
    /// (`needs_ecx` in QEMU).
    Cpuid {
        /// Leaf.
        eax: u32,
        /// Subleaf, if the leaf uses one.
        ecx: Option<u32>,
        /// Output register.
        reg: Reg,
    },
    /// A feature MSR.
    Msr {
        /// MSR index.
        index: u32,
    },
}

/// Static description of one feature word, as `FeatureWordInfo`.
#[derive(Debug)]
pub struct FeatureWordInfo {
    /// CPUID register or MSR.
    pub kind: WordKind,
    /// Bit names, indexed by bit number. Missing trailing entries are unnamed.
    pub names: &'static [Option<&'static str>],
    /// Bits TCG can emulate.
    pub tcg: u64,
    /// Bits that block migration.
    pub unmigratable: u64,
    /// Unnamed bits that are still migratable.
    pub migratable: u64,
    /// Bits `-cpu max` and `-cpu host` never turn on by themselves.
    pub no_autoenable: u64,
}

impl FeatureWordInfo {
    /// Name of bit `bit`, if it has one.
    pub fn bit_name(&self, bit: u32) -> Option<&'static str> {
        self.names.get(bit as usize).copied().flatten()
    }

    /// The CPUID leaf, or `None` for MSR words.
    pub fn cpuid_leaf(&self) -> Option<u32> {
        match self.kind {
            WordKind::Cpuid { eax, .. } => Some(eax),
            WordKind::Msr { .. } => None,
        }
    }

    /// Text used in warnings, as `feature_word_description()`, for example
    /// `CPUID[eax=01h].ECX` or `MSR(10Ah)`.
    pub fn description(&self) -> String {
        match self.kind {
            WordKind::Cpuid { eax, ecx: None, reg } => {
                format!("CPUID[eax={eax:02X}h].{}", reg.name())
            }
            WordKind::Cpuid { eax, ecx: Some(ecx), reg } => {
                format!("CPUID[eax={eax:02X}h,ecx={ecx:02X}h].{}", reg.name())
            }
            WordKind::Msr { index } => format!("MSR({index:02X}h)"),
        }
    }
}

/// `feature_word_info[]`.
pub static FEATURE_WORD_INFO: [FeatureWordInfo; FEATURE_WORDS] = [
    // FEAT_1_EDX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x1, ecx: None, reg: Reg::Edx },
        names: &[
            Some("fpu"),
            Some("vme"),
            Some("de"),
            Some("pse"),
            Some("tsc"),
            Some("msr"),
            Some("pae"),
            Some("mce"),
            Some("cx8"),
            Some("apic"),
            None,
            Some("sep"),
            Some("mtrr"),
            Some("pge"),
            Some("mca"),
            Some("cmov"),
            Some("pat"),
            Some("pse36"),
            Some("pn"),
            Some("clflush"),
            None,
            Some("ds"),
            Some("acpi"),
            Some("mmx"),
            Some("fxsr"),
            Some("sse"),
            Some("sse2"),
            Some("ss"),
            Some("ht"),
            Some("tm"),
            Some("ia64"),
            Some("pbe"),
        ],
        tcg: 0x1fcb_fbfd,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x1000_0000,
    },
    // FEAT_1_ECX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x1, ecx: None, reg: Reg::Ecx },
        names: &[
            Some("pni"),
            Some("pclmulqdq"),
            Some("dtes64"),
            Some("monitor"),
            Some("ds-cpl"),
            Some("vmx"),
            Some("smx"),
            Some("est"),
            Some("tm2"),
            Some("ssse3"),
            Some("cid"),
            None,
            Some("fma"),
            Some("cx16"),
            Some("xtpr"),
            Some("pdcm"),
            None,
            Some("pcid"),
            Some("dca"),
            Some("sse4.1"),
            Some("sse4.2"),
            Some("x2apic"),
            Some("movbe"),
            Some("popcnt"),
            Some("tsc-deadline"),
            Some("aes"),
            Some("xsave"),
            None,
            Some("avx"),
            Some("f16c"),
            Some("rdrand"),
            Some("hypervisor"),
        ],
        tcg: 0xf6f8_320b,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_7_0_EBX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x7, ecx: Some(0), reg: Reg::Ebx },
        names: &[
            Some("fsgsbase"),
            Some("tsc-adjust"),
            Some("sgx"),
            Some("bmi1"),
            Some("hle"),
            Some("avx2"),
            Some("fdp-excptn-only"),
            Some("smep"),
            Some("bmi2"),
            Some("erms"),
            Some("invpcid"),
            Some("rtm"),
            None,
            Some("zero-fcs-fds"),
            Some("mpx"),
            None,
            Some("avx512f"),
            Some("avx512dq"),
            Some("rdseed"),
            Some("adx"),
            Some("smap"),
            Some("avx512ifma"),
            Some("pcommit"),
            Some("clflushopt"),
            Some("clwb"),
            Some("intel-pt"),
            Some("avx512pf"),
            Some("avx512er"),
            Some("avx512cd"),
            Some("sha-ni"),
            Some("avx512bw"),
            Some("avx512vl"),
        ],
        tcg: 0x219c_43a9,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_7_0_ECX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x7, ecx: Some(0), reg: Reg::Ecx },
        names: &[
            None,
            Some("avx512vbmi"),
            Some("umip"),
            Some("pku"),
            None,
            Some("waitpkg"),
            Some("avx512vbmi2"),
            Some("cet-ss"),
            Some("gfni"),
            Some("vaes"),
            Some("vpclmulqdq"),
            Some("avx512vnni"),
            Some("avx512bitalg"),
            None,
            Some("avx512-vpopcntdq"),
            None,
            Some("la57"),
            None,
            None,
            None,
            None,
            None,
            Some("rdpid"),
            None,
            Some("bus-lock-detect"),
            Some("cldemote"),
            None,
            Some("movdiri"),
            Some("movdir64b"),
            None,
            Some("sgxlc"),
            Some("pks"),
        ],
        tcg: 0x8041_020c,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_7_0_EDX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x7, ecx: Some(0), reg: Reg::Edx },
        names: &[
            None,
            None,
            Some("avx512-4vnniw"),
            Some("avx512-4fmaps"),
            Some("fsrm"),
            None,
            None,
            None,
            Some("avx512-vp2intersect"),
            None,
            Some("md-clear"),
            None,
            None,
            None,
            Some("serialize"),
            None,
            Some("tsx-ldtrk"),
            None,
            None,
            Some("arch-lbr"),
            Some("cet-ibt"),
            None,
            Some("amx-bf16"),
            Some("avx512-fp16"),
            Some("amx-tile"),
            Some("amx-int8"),
            Some("spec-ctrl"),
            Some("stibp"),
            Some("flush-l1d"),
            Some("arch-capabilities"),
            Some("core-capability"),
            Some("ssbd"),
        ],
        tcg: 0x10,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_7_1_EAX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x7, ecx: Some(1), reg: Reg::Eax },
        names: &[
            Some("sha512"),
            Some("sm3"),
            Some("sm4"),
            None,
            Some("avx-vnni"),
            Some("avx512-bf16"),
            None,
            Some("cmpccxadd"),
            None,
            None,
            Some("fzrm"),
            Some("fsrs"),
            Some("fsrc"),
            None,
            None,
            None,
            None,
            Some("fred"),
            Some("lkgs"),
            Some("wrmsrns"),
            None,
            Some("amx-fp16"),
            None,
            Some("avx-ifma"),
            None,
            None,
            Some("lam"),
            None,
            None,
            None,
            None,
            Some("movrs"),
        ],
        tcg: 0x1c80,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_8000_0001_EDX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x8000_0001, ecx: None, reg: Reg::Edx },
        names: &[
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("syscall"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("nx"),
            None,
            Some("mmxext"),
            None,
            None,
            Some("fxsr-opt"),
            Some("pdpe1gb"),
            Some("rdtscp"),
            None,
            Some("lm"),
            Some("3dnowext"),
            Some("3dnow"),
        ],
        tcg: 0xedd3_fbfd,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_8000_0001_ECX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x8000_0001, ecx: None, reg: Reg::Ecx },
        names: &[
            Some("lahf-lm"),
            Some("cmp-legacy"),
            Some("svm"),
            Some("extapic"),
            Some("cr8legacy"),
            Some("abm"),
            Some("sse4a"),
            Some("misalignsse"),
            Some("3dnowprefetch"),
            Some("osvw"),
            Some("ibs"),
            Some("xop"),
            Some("skinit"),
            Some("wdt"),
            None,
            Some("lwp"),
            Some("fma4"),
            Some("tce"),
            None,
            Some("nodeid-msr"),
            None,
            Some("tbm"),
            Some("topoext"),
            Some("perfctr-core"),
            Some("perfctr-nb"),
        ],
        tcg: 0x177,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0040_0000,
    },
    // FEAT_8000_0007_EBX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x8000_0007, ecx: None, reg: Reg::Ebx },
        names: &[Some("overflow-recov"), Some("succor")],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_8000_0007_EDX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x8000_0007, ecx: None, reg: Reg::Edx },
        names: &[None, None, None, None, None, None, None, None, Some("invtsc")],
        tcg: 0x0,
        unmigratable: 0x100,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_8000_0008_EBX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x8000_0008, ecx: None, reg: Reg::Ebx },
        names: &[
            Some("clzero"),
            None,
            Some("xsaveerptr"),
            None,
            None,
            None,
            None,
            None,
            None,
            Some("wbnoinvd"),
            None,
            None,
            Some("ibpb"),
            None,
            Some("ibrs"),
            Some("amd-stibp"),
            None,
            Some("stibp-always-on"),
            None,
            None,
            None,
            None,
            None,
            None,
            Some("amd-ssbd"),
            Some("virt-ssbd"),
            Some("amd-no-ssb"),
            None,
            Some("amd-psfd"),
        ],
        tcg: 0x204,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_8000_0021_EAX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x8000_0021, ecx: None, reg: Reg::Eax },
        names: &[
            Some("no-nested-data-bp"),
            Some("fs-gs-base-ns"),
            Some("lfence-always-serializing"),
            None,
            None,
            Some("verw-clear"),
            Some("null-sel-clr-base"),
            None,
            Some("auto-ibrs"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("prefetchi"),
            None,
            None,
            None,
            Some("eraps"),
            None,
            None,
            Some("sbpb"),
            Some("ibpb-brtype"),
            Some("srso-no"),
            Some("srso-user-kernel-no"),
        ],
        tcg: 0x41,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_8000_0021_EBX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x8000_0021, ecx: None, reg: Reg::Ebx },
        names: &[],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_8000_0021_ECX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x8000_0021, ecx: None, reg: Reg::Ecx },
        names: &[None, Some("tsa-sq-no"), Some("tsa-l1-no")],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_8000_0022_EAX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x8000_0022, ecx: None, reg: Reg::Eax },
        names: &[Some("perfmon-v2")],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_C000_0001_EDX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0xc000_0001, ecx: None, reg: Reg::Edx },
        names: &[
            None,
            None,
            Some("xstore"),
            Some("xstore-en"),
            None,
            None,
            Some("xcrypt"),
            Some("xcrypt-en"),
            Some("ace2"),
            Some("ace2-en"),
            Some("phe"),
            Some("phe-en"),
            Some("pmm"),
            Some("pmm-en"),
        ],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_KVM
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: KVM_CPUID_FEATURES, ecx: None, reg: Reg::Eax },
        names: &[
            Some("kvmclock"),
            Some("kvm-nopiodelay"),
            Some("kvm-mmu"),
            Some("kvmclock"),
            Some("kvm-asyncpf"),
            Some("kvm-steal-time"),
            Some("kvm-pv-eoi"),
            Some("kvm-pv-unhalt"),
            None,
            Some("kvm-pv-tlb-flush"),
            Some("kvm-asyncpf-vmexit"),
            Some("kvm-pv-ipi"),
            Some("kvm-poll-control"),
            Some("kvm-pv-sched-yield"),
            Some("kvm-asyncpf-int"),
            Some("kvm-msi-ext-dest-id"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("kvmclock-stable-bit"),
        ],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_KVM_HINTS
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: KVM_CPUID_FEATURES, ecx: None, reg: Reg::Edx },
        names: &[Some("kvm-hint-dedicated")],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0xffff_ffff,
    },
    // FEAT_SVM
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x8000_000a, ecx: None, reg: Reg::Edx },
        names: &[
            Some("npt"),
            Some("lbrv"),
            Some("svm-lock"),
            Some("nrip-save"),
            Some("tsc-scale"),
            Some("vmcb-clean"),
            Some("flushbyasid"),
            Some("decodeassists"),
            None,
            None,
            Some("pause-filter"),
            None,
            Some("pfthreshold"),
            Some("avic"),
            None,
            Some("v-vmsave-vmload"),
            Some("vgif"),
            Some("gmet"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("vnmi"),
            None,
            None,
            Some("svme-addr-chk"),
        ],
        tcg: 0x1001_0001,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_XSAVE
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0xd, ecx: Some(1), reg: Reg::Eax },
        names: &[Some("xsaveopt"), Some("xsavec"), Some("xgetbv1"), Some("xsaves"), Some("xfd")],
        tcg: 0x5,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_6_EAX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x6, ecx: None, reg: Reg::Eax },
        names: &[None, None, Some("arat")],
        tcg: 0x4,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_XSAVE_XCR0_LO
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0xd, ecx: Some(0), reg: Reg::Eax },
        names: &[],
        tcg: 0x21f,
        unmigratable: 0x0,
        migratable: 0x000e_02ff,
        no_autoenable: 0x0,
    },
    // FEAT_XSAVE_XCR0_HI
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0xd, ecx: Some(0), reg: Reg::Edx },
        names: &[],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_ARCH_CAPABILITIES
    FeatureWordInfo {
        kind: WordKind::Msr { index: MSR_IA32_ARCH_CAPABILITIES },
        names: &[
            Some("rdctl-no"),
            Some("ibrs-all"),
            Some("rsba"),
            Some("skip-l1dfl-vmentry"),
            Some("ssb-no"),
            Some("mds-no"),
            Some("pschange-mc-no"),
            Some("tsx-ctrl"),
            Some("taa-no"),
            None,
            None,
            None,
            None,
            Some("sbdr-ssdp-no"),
            Some("fbsdp-no"),
            Some("psdp-no"),
            None,
            Some("fb-clear"),
            None,
            None,
            Some("bhi-no"),
            None,
            None,
            None,
            Some("pbrsb-no"),
            None,
            Some("gds-no"),
            Some("rfds-no"),
            Some("rfds-clear"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("its-no"),
        ],
        tcg: 0xffff_ffff,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_CORE_CAPABILITY
    FeatureWordInfo {
        kind: WordKind::Msr { index: MSR_IA32_CORE_CAPABILITY },
        names: &[None, None, None, None, None, Some("split-lock-detect")],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_PERF_CAPABILITIES
    FeatureWordInfo {
        kind: WordKind::Msr { index: MSR_IA32_PERF_CAPABILITIES },
        names: &[
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("full-width-write"),
        ],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_VMX_PROCBASED_CTLS
    FeatureWordInfo {
        kind: WordKind::Msr { index: MSR_IA32_VMX_TRUE_PROCBASED_CTLS },
        names: &[
            None,
            None,
            Some("vmx-vintr-pending"),
            Some("vmx-tsc-offset"),
            None,
            None,
            None,
            Some("vmx-hlt-exit"),
            None,
            Some("vmx-invlpg-exit"),
            Some("vmx-mwait-exit"),
            Some("vmx-rdpmc-exit"),
            Some("vmx-rdtsc-exit"),
            None,
            None,
            Some("vmx-cr3-load-noexit"),
            Some("vmx-cr3-store-noexit"),
            None,
            None,
            Some("vmx-cr8-load-exit"),
            Some("vmx-cr8-store-exit"),
            Some("vmx-flexpriority"),
            Some("vmx-vnmi-pending"),
            Some("vmx-movdr-exit"),
            Some("vmx-io-exit"),
            Some("vmx-io-bitmap"),
            None,
            Some("vmx-mtf"),
            Some("vmx-msr-bitmap"),
            Some("vmx-monitor-exit"),
            Some("vmx-pause-exit"),
            Some("vmx-secondary-ctls"),
        ],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_VMX_SECONDARY_CTLS
    FeatureWordInfo {
        kind: WordKind::Msr { index: MSR_IA32_VMX_PROCBASED_CTLS2 },
        names: &[
            Some("vmx-apicv-xapic"),
            Some("vmx-ept"),
            Some("vmx-desc-exit"),
            Some("vmx-rdtscp-exit"),
            Some("vmx-apicv-x2apic"),
            Some("vmx-vpid"),
            Some("vmx-wbinvd-exit"),
            Some("vmx-unrestricted-guest"),
            Some("vmx-apicv-register"),
            Some("vmx-apicv-vid"),
            Some("vmx-ple"),
            Some("vmx-rdrand-exit"),
            Some("vmx-invpcid-exit"),
            Some("vmx-vmfunc"),
            Some("vmx-shadow-vmcs"),
            Some("vmx-encls-exit"),
            Some("vmx-rdseed-exit"),
            Some("vmx-pml"),
            None,
            None,
            Some("vmx-xsaves"),
            None,
            Some("vmx-mbec"),
            None,
            None,
            Some("vmx-tsc-scaling"),
            Some("vmx-enable-user-wait-pause"),
        ],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_VMX_PINBASED_CTLS
    FeatureWordInfo {
        kind: WordKind::Msr { index: MSR_IA32_VMX_TRUE_PINBASED_CTLS },
        names: &[
            Some("vmx-intr-exit"),
            None,
            None,
            Some("vmx-nmi-exit"),
            None,
            Some("vmx-vnmi"),
            Some("vmx-preemption-timer"),
            Some("vmx-posted-intr"),
        ],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_VMX_EXIT_CTLS
    FeatureWordInfo {
        kind: WordKind::Msr { index: MSR_IA32_VMX_TRUE_EXIT_CTLS },
        names: &[
            None,
            None,
            Some("vmx-exit-nosave-debugctl"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("vmx-exit-load-perf-global-ctrl"),
            None,
            None,
            Some("vmx-exit-ack-intr"),
            None,
            None,
            Some("vmx-exit-save-pat"),
            Some("vmx-exit-load-pat"),
            Some("vmx-exit-save-efer"),
            Some("vmx-exit-load-efer"),
            Some("vmx-exit-save-preemption-timer"),
            Some("vmx-exit-clear-bndcfgs"),
            None,
            Some("vmx-exit-clear-rtit-ctl"),
            None,
            None,
            Some("vmx-exit-save-cet"),
            Some("vmx-exit-load-pkrs"),
            None,
            Some("vmx-exit-secondary-ctls"),
        ],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_VMX_ENTRY_CTLS
    FeatureWordInfo {
        kind: WordKind::Msr { index: MSR_IA32_VMX_TRUE_ENTRY_CTLS },
        names: &[
            None,
            None,
            Some("vmx-entry-noload-debugctl"),
            None,
            None,
            None,
            None,
            None,
            None,
            Some("vmx-entry-ia32e-mode"),
            None,
            None,
            None,
            Some("vmx-entry-load-perf-global-ctrl"),
            Some("vmx-entry-load-pat"),
            Some("vmx-entry-load-efer"),
            Some("vmx-entry-load-bndcfgs"),
            None,
            Some("vmx-entry-load-rtit-ctl"),
            None,
            Some("vmx-entry-load-cet"),
            None,
            Some("vmx-entry-load-pkrs"),
            Some("vmx-entry-load-fred"),
        ],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_VMX_MISC
    FeatureWordInfo {
        kind: WordKind::Msr { index: MSR_IA32_VMX_MISC },
        names: &[
            None,
            None,
            None,
            None,
            None,
            Some("vmx-store-lma"),
            Some("vmx-activity-hlt"),
            Some("vmx-activity-shutdown"),
            Some("vmx-activity-wait-sipi"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("vmx-vmwrite-vmexit-fields"),
            Some("vmx-zero-len-inject"),
        ],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_VMX_EPT_VPID_CAPS
    FeatureWordInfo {
        kind: WordKind::Msr { index: MSR_IA32_VMX_EPT_VPID_CAP },
        names: &[
            Some("vmx-ept-execonly"),
            None,
            None,
            None,
            None,
            None,
            Some("vmx-page-walk-4"),
            Some("vmx-page-walk-5"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("vmx-ept-2mb"),
            Some("vmx-ept-1gb"),
            None,
            None,
            Some("vmx-invept"),
            Some("vmx-eptad"),
            Some("vmx-ept-advanced-exitinfo"),
            None,
            None,
            Some("vmx-invept-single-context"),
            Some("vmx-invept-all-context"),
            None,
            None,
            None,
            None,
            None,
            Some("vmx-invvpid"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("vmx-invvpid-single-addr"),
            Some("vmx-invept-single-context"),
            Some("vmx-invvpid-all-context"),
            Some("vmx-invept-single-context-noglobals"),
        ],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_VMX_BASIC
    FeatureWordInfo {
        kind: WordKind::Msr { index: MSR_IA32_VMX_BASIC },
        names: &[
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("vmx-ins-outs"),
            Some("vmx-true-ctls"),
            Some("vmx-any-errcode"),
            None,
            Some("vmx-nested-exception"),
        ],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0002_0000_0000_0000,
    },
    // FEAT_VMX_VMFUNC
    FeatureWordInfo {
        kind: WordKind::Msr { index: MSR_IA32_VMX_VMFUNC },
        names: &[Some("vmx-eptp-switching")],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_14_0_ECX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x14, ecx: Some(0), reg: Reg::Ecx },
        names: &[
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("intel-pt-lip"),
        ],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_SGX_12_0_EAX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x12, ecx: Some(0), reg: Reg::Eax },
        names: &[
            Some("sgx1"),
            Some("sgx2"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("sgx-edeccssa"),
        ],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_SGX_12_0_EBX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x12, ecx: Some(0), reg: Reg::Ebx },
        names: &[Some("sgx-exinfo")],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_SGX_12_1_EAX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x12, ecx: Some(1), reg: Reg::Eax },
        names: &[
            None,
            Some("sgx-debug"),
            Some("sgx-mode64"),
            None,
            Some("sgx-provisionkey"),
            Some("sgx-tokenkey"),
            None,
            Some("sgx-kss"),
            None,
            None,
            Some("sgx-aex-notify"),
        ],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_XSAVE_XSS_LO
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0xd, ecx: Some(1), reg: Reg::Ecx },
        names: &[],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x9800,
        no_autoenable: 0x0,
    },
    // FEAT_XSAVE_XSS_HI
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0xd, ecx: Some(1), reg: Reg::Edx },
        names: &[],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_7_1_ECX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x7, ecx: Some(1), reg: Reg::Ecx },
        names: &[None, None, None, None, None, Some("msr-imm")],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_7_1_EDX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x7, ecx: Some(1), reg: Reg::Edx },
        names: &[
            None,
            None,
            None,
            None,
            Some("avx-vnni-int8"),
            Some("avx-ne-convert"),
            None,
            None,
            Some("amx-complex"),
            None,
            Some("avx-vnni-int16"),
            None,
            None,
            None,
            Some("prefetchiti"),
            None,
            None,
            None,
            None,
            Some("avx10"),
            None,
            Some("apxf"),
        ],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_7_2_EDX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x7, ecx: Some(2), reg: Reg::Edx },
        names: &[
            Some("intel-psfd"),
            Some("ipred-ctrl"),
            Some("rrsba-ctrl"),
            Some("ddpd-u"),
            Some("bhi-ctrl"),
            Some("mcdt-no"),
        ],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_24_0_EBX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x24, ecx: Some(0), reg: Reg::Ebx },
        names: &[
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some("avx10-128"),
            Some("avx10-256"),
            Some("avx10-512"),
        ],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_29_0_EBX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x29, ecx: Some(0), reg: Reg::Ebx },
        names: &[Some("apx-nci-ndd-nf")],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_1E_1_EAX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x1e, ecx: Some(1), reg: Reg::Eax },
        names: &[
            Some("amx-int8-alias"),
            Some("amx-bf16-alias"),
            Some("amx-complex-alias"),
            Some("amx-fp16-alias"),
            Some("amx-fp8"),
            None,
            Some("amx-tf32"),
            Some("amx-avx512"),
            Some("amx-movrs"),
        ],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
    // FEAT_24_1_ECX
    FeatureWordInfo {
        kind: WordKind::Cpuid { eax: 0x24, ecx: Some(1), reg: Reg::Ecx },
        names: &[None, None, Some("avx10-vnni-int")],
        tcg: 0x0,
        unmigratable: 0x0,
        migratable: 0x0,
        no_autoenable: 0x0,
    },
];

/// One entry of `feature_dependencies[]`: the `to` bits need the `from`
/// bits, and are dropped when `from` is clear.
#[derive(Debug, Clone, Copy)]
pub struct FeatureDep {
    /// Word and mask that must be present.
    pub from: (usize, u64),
    /// Word and mask that depend on it.
    pub to: (usize, u64),
}

/// `feature_dependencies[]`.
pub static FEATURE_DEPENDENCIES: &[FeatureDep] = &[
    FeatureDep { from: (FEAT_7_0_EDX, 0x2000_0000), to: (FEAT_ARCH_CAPABILITIES, u64::MAX) },
    FeatureDep { from: (FEAT_7_0_EDX, 0x4000_0000), to: (FEAT_CORE_CAPABILITY, u64::MAX) },
    FeatureDep { from: (FEAT_1_ECX, 0x8000), to: (FEAT_PERF_CAPABILITIES, u64::MAX) },
    FeatureDep { from: (FEAT_1_ECX, 0x20), to: (FEAT_VMX_PROCBASED_CTLS, u64::MAX) },
    FeatureDep { from: (FEAT_1_ECX, 0x20), to: (FEAT_VMX_PINBASED_CTLS, u64::MAX) },
    FeatureDep { from: (FEAT_1_ECX, 0x20), to: (FEAT_VMX_EXIT_CTLS, u64::MAX) },
    FeatureDep { from: (FEAT_1_ECX, 0x20), to: (FEAT_VMX_ENTRY_CTLS, u64::MAX) },
    FeatureDep { from: (FEAT_1_ECX, 0x20), to: (FEAT_VMX_MISC, u64::MAX) },
    FeatureDep { from: (FEAT_1_ECX, 0x20), to: (FEAT_VMX_BASIC, u64::MAX) },
    FeatureDep { from: (FEAT_8000_0001_EDX, 0x2000_0000), to: (FEAT_VMX_ENTRY_CTLS, 0x200) },
    FeatureDep {
        from: (FEAT_VMX_PROCBASED_CTLS, 0x8000_0000),
        to: (FEAT_VMX_SECONDARY_CTLS, u64::MAX),
    },
    FeatureDep { from: (FEAT_XSAVE, 0x8), to: (FEAT_VMX_SECONDARY_CTLS, 0x0010_0000) },
    FeatureDep { from: (FEAT_1_ECX, 0x4000_0000), to: (FEAT_VMX_SECONDARY_CTLS, 0x800) },
    FeatureDep { from: (FEAT_7_0_EBX, 0x400), to: (FEAT_VMX_SECONDARY_CTLS, 0x1000) },
    FeatureDep { from: (FEAT_7_0_EBX, 0x4000), to: (FEAT_VMX_EXIT_CTLS, 0x0080_0000) },
    FeatureDep { from: (FEAT_7_0_EBX, 0x4000), to: (FEAT_VMX_ENTRY_CTLS, 0x0001_0000) },
    FeatureDep { from: (FEAT_7_0_EBX, 0x0004_0000), to: (FEAT_VMX_SECONDARY_CTLS, 0x0001_0000) },
    FeatureDep { from: (FEAT_7_0_EBX, 0x0200_0000), to: (FEAT_14_0_ECX, u64::MAX) },
    FeatureDep { from: (FEAT_8000_0001_EDX, 0x0800_0000), to: (FEAT_VMX_SECONDARY_CTLS, 0x8) },
    FeatureDep { from: (FEAT_VMX_SECONDARY_CTLS, 0x2), to: (FEAT_VMX_EPT_VPID_CAPS, 0xffff_ffff) },
    FeatureDep { from: (FEAT_VMX_SECONDARY_CTLS, 0x2), to: (FEAT_VMX_SECONDARY_CTLS, 0x80) },
    FeatureDep { from: (FEAT_VMX_SECONDARY_CTLS, 0x2), to: (FEAT_VMX_SECONDARY_CTLS, 0x0040_0000) },
    FeatureDep {
        from: (FEAT_VMX_SECONDARY_CTLS, 0x20),
        to: (FEAT_VMX_EPT_VPID_CAPS, 0xffff_ffff_0000_0000),
    },
    FeatureDep { from: (FEAT_VMX_SECONDARY_CTLS, 0x2000), to: (FEAT_VMX_VMFUNC, u64::MAX) },
    FeatureDep { from: (FEAT_8000_0001_ECX, 0x4), to: (FEAT_SVM, u64::MAX) },
    FeatureDep { from: (FEAT_7_0_ECX, 0x20), to: (FEAT_VMX_SECONDARY_CTLS, 0x0400_0000) },
    FeatureDep { from: (FEAT_8000_0001_EDX, 0x2000_0000), to: (FEAT_7_1_EAX, 0x0002_0000) },
    FeatureDep { from: (FEAT_7_1_EAX, 0x0004_0000), to: (FEAT_7_1_EAX, 0x0002_0000) },
    FeatureDep { from: (FEAT_7_0_EBX, 0x4), to: (FEAT_7_0_ECX, 0x4000_0000) },
    FeatureDep { from: (FEAT_7_0_EBX, 0x4), to: (FEAT_SGX_12_0_EAX, u64::MAX) },
    FeatureDep { from: (FEAT_7_0_EBX, 0x4), to: (FEAT_SGX_12_0_EBX, u64::MAX) },
    FeatureDep { from: (FEAT_7_0_EBX, 0x4), to: (FEAT_SGX_12_1_EAX, u64::MAX) },
    FeatureDep { from: (FEAT_24_0_EBX, 0x0001_0000), to: (FEAT_24_0_EBX, 0x0002_0000) },
    FeatureDep { from: (FEAT_24_0_EBX, 0x0002_0000), to: (FEAT_24_0_EBX, 0x0004_0000) },
    FeatureDep { from: (FEAT_24_0_EBX, 0x0007_0000), to: (FEAT_7_1_EDX, 0x0008_0000) },
    FeatureDep { from: (FEAT_7_1_EDX, 0x0008_0000), to: (FEAT_24_0_EBX, u64::MAX) },
    FeatureDep { from: (FEAT_7_1_EDX, 0x0020_0000), to: (FEAT_29_0_EBX, u64::MAX) },
    FeatureDep { from: (FEAT_7_1_EDX, 0x0008_0000), to: (FEAT_24_1_ECX, u64::MAX) },
    FeatureDep { from: (FEAT_7_1_EAX, 0x0002_0000), to: (FEAT_VMX_ENTRY_CTLS, 0x0080_0000) },
];

/// Property aliases that `x86_cpu_initfn()` adds with
/// `object_property_add_alias()`, as (alias, feature name).
///
/// `hv-apicv` and `lbr_fmt` are left out because the Hyper-V and `lbr-fmt`
/// properties are not ported.
pub static FEATURE_ALIASES: &[(&str, &str)] = &[
    ("sse3", "pni"),
    ("pclmuldq", "pclmulqdq"),
    ("sse4-1", "sse4.1"),
    ("sse4-2", "sse4.2"),
    ("xd", "nx"),
    ("ffxsr", "fxsr-opt"),
    ("i64", "lm"),
    ("ds_cpl", "ds-cpl"),
    ("tsc_adjust", "tsc-adjust"),
    ("fxsr_opt", "fxsr-opt"),
    ("lahf_lm", "lahf-lm"),
    ("cmp_legacy", "cmp-legacy"),
    ("nodeid_msr", "nodeid-msr"),
    ("perfctr_core", "perfctr-core"),
    ("perfctr_nb", "perfctr-nb"),
    ("kvm_nopiodelay", "kvm-nopiodelay"),
    ("kvm_mmu", "kvm-mmu"),
    ("kvm_asyncpf", "kvm-asyncpf"),
    ("kvm_asyncpf_int", "kvm-asyncpf-int"),
    ("kvm_steal_time", "kvm-steal-time"),
    ("kvm_pv_eoi", "kvm-pv-eoi"),
    ("kvm_pv_unhalt", "kvm-pv-unhalt"),
    ("kvm_poll_control", "kvm-poll-control"),
    ("svm_lock", "svm-lock"),
    ("nrip_save", "nrip-save"),
    ("tsc_scale", "tsc-scale"),
    ("vmcb_clean", "vmcb-clean"),
    ("pause_filter", "pause-filter"),
    ("sse4_1", "sse4.1"),
    ("sse4_2", "sse4.2"),
];

/// Finds the feature word and mask for a feature property name, following
/// the aliases above. A name used for several bits of one word (`kvmclock`)
/// gives all of them.
pub fn find_feature(name: &str) -> Option<(usize, u64)> {
    let name =
        FEATURE_ALIASES.iter().find(|(alias, _)| *alias == name).map_or(name, |(_, target)| target);
    for (w, info) in FEATURE_WORD_INFO.iter().enumerate() {
        let mut mask = 0u64;
        for (bit, n) in info.names.iter().enumerate() {
            if *n == Some(name) {
                mask |= 1 << bit;
            }
        }
        if mask != 0 {
            return Some((w, mask));
        }
    }
    None
}

// Bit masks used by the CPUID code. The names and values are QEMU's.

/// `CPUID_APIC`.
pub const CPUID_APIC: u64 = 0x200;
/// `CPUID_MCE`.
pub const CPUID_MCE: u64 = 0x80;
/// `CPUID_MCA`.
pub const CPUID_MCA: u64 = 0x4000;
/// `CPUID_MTRR`.
pub const CPUID_MTRR: u64 = 0x1000;
/// `CPUID_PAT`.
pub const CPUID_PAT: u64 = 0x10000;
/// `CPUID_PSE36`.
pub const CPUID_PSE36: u64 = 0x20000;
/// `CPUID_PAE`.
pub const CPUID_PAE: u64 = 0x40;
/// `CPUID_HT`.
pub const CPUID_HT: u64 = 0x10000000;
/// `CPUID_EXT_MONITOR`.
pub const CPUID_EXT_MONITOR: u64 = 0x8;
/// `CPUID_EXT_VMX`.
pub const CPUID_EXT_VMX: u64 = 0x20;
/// `CPUID_EXT_PDCM`.
pub const CPUID_EXT_PDCM: u64 = 0x8000;
/// `CPUID_EXT_X2APIC`.
pub const CPUID_EXT_X2APIC: u64 = 0x200000;
/// `CPUID_EXT_TSC_DEADLINE_TIMER`.
pub const CPUID_EXT_TSC_DEADLINE_TIMER: u64 = 0x1000000;
/// `CPUID_EXT_XSAVE`.
pub const CPUID_EXT_XSAVE: u64 = 0x4000000;
/// `CPUID_EXT_OSXSAVE`.
pub const CPUID_EXT_OSXSAVE: u64 = 0x8000000;
/// `CPUID_EXT_AVX`.
pub const CPUID_EXT_AVX: u64 = 0x10000000;
/// `CPUID_EXT_RDRAND`.
pub const CPUID_EXT_RDRAND: u64 = 0x40000000;
/// `CPUID_EXT_HYPERVISOR`.
pub const CPUID_EXT_HYPERVISOR: u64 = 0x80000000;
/// `CPUID_7_0_EBX_SGX`.
pub const CPUID_7_0_EBX_SGX: u64 = 0x4;
/// `CPUID_7_0_EBX_HLE`.
pub const CPUID_7_0_EBX_HLE: u64 = 0x10;
/// `CPUID_7_0_EBX_ERMS`.
pub const CPUID_7_0_EBX_ERMS: u64 = 0x200;
/// `CPUID_7_0_EBX_INVPCID`.
pub const CPUID_7_0_EBX_INVPCID: u64 = 0x400;
/// `CPUID_7_0_EBX_RTM`.
pub const CPUID_7_0_EBX_RTM: u64 = 0x800;
/// `CPUID_7_0_EBX_MPX`.
pub const CPUID_7_0_EBX_MPX: u64 = 0x4000;
/// `CPUID_7_0_EBX_AVX512F`.
pub const CPUID_7_0_EBX_AVX512F: u64 = 0x10000;
/// `CPUID_7_0_EBX_RDSEED`.
pub const CPUID_7_0_EBX_RDSEED: u64 = 0x40000;
/// `CPUID_7_0_EBX_INTEL_PT`.
pub const CPUID_7_0_EBX_INTEL_PT: u64 = 0x2000000;
/// `CPUID_7_0_ECX_PKU`.
pub const CPUID_7_0_ECX_PKU: u64 = 0x8;
/// `CPUID_7_0_ECX_OSPKE`.
pub const CPUID_7_0_ECX_OSPKE: u64 = 0x10;
/// `CPUID_7_0_ECX_CET_SHSTK`.
pub const CPUID_7_0_ECX_CET_SHSTK: u64 = 0x80;
/// `CPUID_7_0_ECX_LA57`.
pub const CPUID_7_0_ECX_LA57: u64 = 0x10000;
/// `CPUID_7_0_ECX_SGX_LC`.
pub const CPUID_7_0_ECX_SGX_LC: u64 = 0x40000000;
/// `CPUID_7_0_EDX_FSRM`.
pub const CPUID_7_0_EDX_FSRM: u64 = 0x10;
/// `CPUID_7_0_EDX_CET_IBT`.
pub const CPUID_7_0_EDX_CET_IBT: u64 = 0x100000;
/// `CPUID_7_0_EDX_ARCH_LBR`.
pub const CPUID_7_0_EDX_ARCH_LBR: u64 = 0x80000;
/// `CPUID_7_0_EDX_AMX_TILE`.
pub const CPUID_7_0_EDX_AMX_TILE: u64 = 0x1000000;
/// `CPUID_7_0_EDX_ARCH_CAPABILITIES`.
pub const CPUID_7_0_EDX_ARCH_CAPABILITIES: u64 = 0x20000000;
/// `CPUID_7_1_EAX_FZRM`.
pub const CPUID_7_1_EAX_FZRM: u64 = 0x400;
/// `CPUID_7_1_EAX_FSRS`.
pub const CPUID_7_1_EAX_FSRS: u64 = 0x800;
/// `CPUID_7_1_EAX_FSRC`.
pub const CPUID_7_1_EAX_FSRC: u64 = 0x1000;
/// `CPUID_7_1_EDX_AVX10`.
pub const CPUID_7_1_EDX_AVX10: u64 = 0x80000;
/// `CPUID_7_1_EDX_APXF`.
pub const CPUID_7_1_EDX_APXF: u64 = 0x200000;
/// `CPUID_7_2_EDX_MCDT_NO`.
pub const CPUID_7_2_EDX_MCDT_NO: u64 = 0x20;
/// `CPUID_EXT2_SYSCALL`.
pub const CPUID_EXT2_SYSCALL: u64 = 0x800;
/// `CPUID_EXT2_NX`.
pub const CPUID_EXT2_NX: u64 = 0x100000;
/// `CPUID_EXT2_RDTSCP`.
pub const CPUID_EXT2_RDTSCP: u64 = 0x8000000;
/// `CPUID_EXT2_LM`.
pub const CPUID_EXT2_LM: u64 = 0x20000000;
/// `CPUID_EXT2_AMD_ALIASES`.
pub const CPUID_EXT2_AMD_ALIASES: u64 = 0x183f3ff;
/// `CPUID_EXT3_LAHF_LM`.
pub const CPUID_EXT3_LAHF_LM: u64 = 0x1;
/// `CPUID_EXT3_CMP_LEG`.
pub const CPUID_EXT3_CMP_LEG: u64 = 0x2;
/// `CPUID_EXT3_SVM`.
pub const CPUID_EXT3_SVM: u64 = 0x4;
/// `CPUID_EXT3_TOPOEXT`.
pub const CPUID_EXT3_TOPOEXT: u64 = 0x400000;
/// `CPUID_8000_0007_EBX_OVERFLOW_RECOV`.
pub const CPUID_8000_0007_EBX_OVERFLOW_RECOV: u64 = 0x1;
/// `CPUID_8000_0007_EBX_SUCCOR`.
pub const CPUID_8000_0007_EBX_SUCCOR: u64 = 0x2;
/// `CPUID_APM_INVTSC`.
pub const CPUID_APM_INVTSC: u64 = 0x100;
/// `CPUID_XSAVE_XSAVES`.
pub const CPUID_XSAVE_XSAVES: u64 = 0x8;
/// `CPUID_6_EAX_ARAT`.
pub const CPUID_6_EAX_ARAT: u64 = 0x4;
/// `CPUID_XSTATE_XCR0_MASK`.
pub const CPUID_XSTATE_XCR0_MASK: u64 = 0xe02ff;
/// `CPUID_XSTATE_XSS_MASK`.
pub const CPUID_XSTATE_XSS_MASK: u64 = 0x9800;
/// `MSR_VMX_BASIC_DUAL_MONITOR`.
pub const MSR_VMX_BASIC_DUAL_MONITOR: u64 = 0x2000000000000;
/// `VMX_SECONDARY_EXEC_RDTSCP`.
pub const VMX_SECONDARY_EXEC_RDTSCP: u64 = 0x8;
/// `VMX_SECONDARY_EXEC_RDRAND_EXITING`.
pub const VMX_SECONDARY_EXEC_RDRAND_EXITING: u64 = 0x800;
/// `VMX_SECONDARY_EXEC_ENABLE_INVPCID`.
pub const VMX_SECONDARY_EXEC_ENABLE_INVPCID: u64 = 0x1000;
/// `VMX_SECONDARY_EXEC_RDSEED_EXITING`.
pub const VMX_SECONDARY_EXEC_RDSEED_EXITING: u64 = 0x10000;
/// `VMX_SECONDARY_EXEC_XSAVES`.
pub const VMX_SECONDARY_EXEC_XSAVES: u64 = 0x100000;
/// `MSR_ARCH_CAP_MDS_NO`.
pub const MSR_ARCH_CAP_MDS_NO: u64 = 0x20;
/// `MSR_ARCH_CAP_TAA_NO`.
pub const MSR_ARCH_CAP_TAA_NO: u64 = 0x100;
/// `MSR_ARCH_CAP_SBDR_SSDP_NO`.
pub const MSR_ARCH_CAP_SBDR_SSDP_NO: u64 = 0x2000;
/// `MSR_ARCH_CAP_FBSDP_NO`.
pub const MSR_ARCH_CAP_FBSDP_NO: u64 = 0x4000;
/// `MSR_ARCH_CAP_PSDP_NO`.
pub const MSR_ARCH_CAP_PSDP_NO: u64 = 0x8000;
/// `MSR_ARCH_CAP_FB_CLEAR`.
pub const MSR_ARCH_CAP_FB_CLEAR: u64 = 0x20000;
/// `CPUID_B_ECX_TOPO_LEVEL_INVALID`.
pub const CPUID_B_ECX_TOPO_LEVEL_INVALID: u32 = 0x0;
/// `CPUID_B_ECX_TOPO_LEVEL_SMT`.
pub const CPUID_B_ECX_TOPO_LEVEL_SMT: u32 = 0x1;
/// `CPUID_B_ECX_TOPO_LEVEL_CORE`.
pub const CPUID_B_ECX_TOPO_LEVEL_CORE: u32 = 0x2;
/// `CPUID_1F_ECX_TOPO_LEVEL_MODULE`.
pub const CPUID_1F_ECX_TOPO_LEVEL_MODULE: u32 = 0x3;
/// `CPUID_1F_ECX_TOPO_LEVEL_DIE`.
pub const CPUID_1F_ECX_TOPO_LEVEL_DIE: u32 = 0x5;
/// `CPUID_MWAIT_EMX`.
pub const CPUID_MWAIT_EMX: u64 = 0x1;
/// `CPUID_MWAIT_IBE`.
pub const CPUID_MWAIT_IBE: u64 = 0x2;
