// SPDX-License-Identifier: GPL-2.0-or-later

//! Tests for the model, feature and CPUID code. Expected values are worked
//! out by hand from `target/i386/cpu.c` in QEMU 11.1; the comments say
//! where each piece comes from.

use super::host::{CpuidEntry, HostCpuid};
use super::words::{FEAT_1_ECX, FEAT_1_EDX, FEAT_7_0_EBX, KVM_CPUID_FEATURES};
use super::*;
use crate::state::IrqchipMode;

/// "Auth" "enti" "cAMD" as little-endian words (leaf 0 EBX, EDX, ECX).
const AMD_EBX: u32 = 0x6874_7541;
const AMD_EDX: u32 = 0x6974_6e65;
const AMD_ECX: u32 = 0x444d_4163;
/// "Genu" "ineI" "ntel".
const INTEL_EBX: u32 = 0x756e_6547;
const INTEL_EDX: u32 = 0x4965_6e69;
const INTEL_ECX: u32 = 0x6c65_746e;

fn tcg(model: &str) -> X86Cpu {
    let mut cpu = X86Cpu::new(model, Accel::Tcg).unwrap();
    cpu.realize().unwrap();
    cpu
}

/// A GenuineIntel host where KVM supports every bit of every leaf the
/// tests look at, with 48 physical address bits.
fn permissive_host() -> HostCpuid {
    let all = [u32::MAX; 4];
    let mut supported = Vec::new();
    for (f, i) in [
        (1, 0),
        (6, 0),
        (7, 0),
        (7, 1),
        (7, 2),
        (0xd, 0),
        (0xd, 1),
        (0x8000_0001, 0),
        (0x8000_0007, 0),
        (0x8000_0008, 0),
        (0x8000_000a, 0),
        (0xc000_0001, 0),
        (KVM_CPUID_FEATURES, 0),
    ] {
        supported.push(CpuidEntry::new(f, i, all));
    }
    let host = vec![
        CpuidEntry::new(0, 0, [0x16, INTEL_EBX, INTEL_ECX, INTEL_EDX]),
        CpuidEntry::new(1, 0, [0x0005_06e3, 0, 0, 0]),
        CpuidEntry::new(0x8000_0000, 0, [0x8000_0008, 0, 0, 0]),
        CpuidEntry::new(0x8000_0008, 0, [0x3027, 0, 0, 0]),
    ];
    HostCpuid {
        supported,
        host,
        irqchip: IrqchipMode::Full,
        has_tsc_deadline: true,
        has_msr_arch_capabs: true,
        ..Default::default()
    }
}

fn kvm(model: &str, host: HostCpuid) -> X86Cpu {
    let mut cpu = X86Cpu::new(model, Accel::Kvm(host)).unwrap();
    cpu.realize().unwrap();
    cpu
}

// qemu64 under TCG.
//
// builtin_x86_defs[] "qemu64": level 0xd, vendor AuthenticAMD, family 15,
// model 107, stepping 1, xlevel 0x8000000A.
//   FEAT_1_EDX = PPRO_FEATURES | CPUID_MTRR | CPUID_CLFLUSH | CPUID_MCA |
//                CPUID_PSE36 = 0x078bfbfd (no VME, so the TCG default
//                "vme=off" changes nothing)
//   FEAT_1_ECX = CPUID_EXT_SSE3 | CPUID_EXT_CX16 = 0x00002001
//   FEAT_8000_0001_EDX = CPUID_EXT2_LM | CPUID_EXT2_SYSCALL | CPUID_EXT2_NX
//                      = 0x20100800
//   FEAT_8000_0001_ECX = CPUID_EXT3_LAHF_LM | CPUID_EXT3_SVM = 0x5
// x86_cpu_load_model() adds CPUID_EXT_HYPERVISOR (bit 31) to FEAT_1_ECX.
// All of these are in the TCG_* masks, so nothing is filtered.

#[test]
fn qemu64_tcg_leaf0() {
    let cpu = tcg("qemu64");
    // cpu_x86_cpuid() case 0: EAX = cpuid_level, then the vendor words in
    // EBX, EDX, ECX order.
    assert_eq!(cpu.cpuid(0, 0), [0xd, AMD_EBX, AMD_ECX, AMD_EDX]);
    assert!(cpu.warnings().is_empty(), "{:?}", cpu.warnings());
}

#[test]
fn qemu64_tcg_leaf1() {
    let cpu = tcg("qemu64");
    // Version: family 15 -> 0xf << 8; model 107 = 0x6b -> 0xb << 4 and
    // 0x6 << 16; stepping 1. EBX: APIC ID 0 << 24 | CLFLUSH size 8 << 8.
    assert_eq!(cpu.cpuid(1, 0), [0x0006_0fb1, 0x0000_0800, 0x8000_2001, 0x078b_fbfd]);
}

#[test]
fn qemu64_tcg_leaf7() {
    let cpu = tcg("qemu64");
    // No leaf 7 feature words are set, so level_func7 stays 0 and every
    // register is zero. Leaf 7 is below level 0xd, so it is not clamped.
    assert_eq!(cpu.cpuid(7, 0), [0, 0, 0, 0]);
}

#[test]
fn qemu64_tcg_leaf_80000001() {
    let cpu = tcg("qemu64");
    // x86_cpu_realizefn(): for AMD, the CPUID_EXT2_AMD_ALIASES bits of
    // 8000_0001.EDX are copied from 1.EDX:
    //   0x078bfbfd & 0x0183f3ff = 0x0183f3fd
    //   0x20100800 | 0x0183f3fd = 0x2193fbfd
    // EAX is cpuid_version. SYSCALL is only hidden for Intel vendors.
    assert_eq!(cpu.cpuid(0x8000_0001, 0), [0x0006_0fb1, 0, 0x5, 0x2193_fbfd]);
    // SVM raises xlevel to 0x8000000A, which it already is.
    assert_eq!(cpu.cpuid(0x8000_0000, 0), [0x8000_000a, AMD_EBX, AMD_ECX, AMD_EDX]);
    assert_eq!(cpu.levels(), (0xd, 0x8000_000a, 0));
}

#[test]
fn qemu64_tcg_misc() {
    let cpu = tcg("qemu64");
    // Leaf above level is answered as the highest basic leaf.
    assert_eq!(cpu.cpuid(0x20, 0), cpu.cpuid(0xd, 0));
    // No XSAVE in qemu64, so leaf 0xd is all zeros.
    assert_eq!(cpu.cpuid(0xd, 0), [0; 4]);
    // TCG signature at 0x40000000: "TCGTCGTCGTCG".
    assert_eq!(cpu.cpuid(0x4000_0000, 0), [0x4000_0001, 0x5447_4354, 0x4354_4743, 0x4743_5447]);
    // 0x80000008 EAX: phys_bits 40 | 48 << 8 | guest_phys_bits 0 << 16.
    assert_eq!(cpu.cpuid(0x8000_0008, 0)[0], 0x3028);
    // Model id "QEMU Virtual CPU version 2.5+".
    assert_eq!(cpu.cpuid(0x8000_0002, 0)[0], u32::from_le_bytes(*b"QEMU"));
    // AMD ucode default and MCE setup (family 15, MCE and MCA set).
    assert_eq!(cpu.ucode_rev(), 0x0100_0065);
    assert_eq!(cpu.mcg_cap(), MCG_CTL_P | MCG_SER_P | 10);
    let s = cpu.new_state(true);
    assert_eq!(s.mcg_ctl, u64::MAX);
    assert_eq!(s.mce_banks[4], u64::MAX);
    assert_eq!(s.regs[2], 0x0006_0fb1);
}

// Skylake-Client under KVM with a host that supports everything.
//
// builtin_x86_defs[] "Skylake-Client": level 0xd, GenuineIntel, family 6,
// model 94, stepping 3, xlevel 0x80000008.
//   FEAT_1_EDX = 0x078bfbff (qemu64's set plus VME)
//   FEAT_1_ECX = AVX | XSAVE | AES | POPCNT | X2APIC | SSE42 | SSE41 | CX16 |
//                SSSE3 | PCLMULQDQ | SSE3 | TSC_DEADLINE | FMA | MOVBE |
//                PCID | F16C | RDRAND = 0x77fa3203
//   FEAT_7_0_EBX = FSGSBASE | BMI1 | HLE | AVX2 | SMEP | BMI2 | ERMS |
//                  INVPCID | RTM | RDSEED | ADX | SMAP = 0x001c0fb9
//   FEAT_8000_0001_EDX = RDTSCP | LM | NX | SYSCALL = 0x28100800
//   FEAT_8000_0001_ECX = ABM | LAHF_LM | 3DNOWPREFETCH = 0x121

#[test]
fn skylake_kvm_leaf0() {
    let cpu = kvm("Skylake-Client", permissive_host());
    assert_eq!(cpu.cpuid(0, 0), [0xd, INTEL_EBX, INTEL_ECX, INTEL_EDX]);
    assert!(cpu.warnings().is_empty(), "{:?}", cpu.warnings());
}

#[test]
fn skylake_kvm_leaf1() {
    let cpu = kvm("Skylake-Client", permissive_host());
    // Version: 6 << 8 | 0xe << 4 | 0x5 << 16 | 3 = 0x000506e3.
    // ECX adds HYPERVISOR; kvm_default_props turns x2apic on (already set)
    // and monitor off (not set). EDX keeps VME, since the "vme=off"
    // default is TCG only.
    assert_eq!(cpu.cpuid(1, 0), [0x0005_06e3, 0x0000_0800, 0xf7fa_3203, 0x078b_fbff]);
    // With CR4.OSXSAVE set, cpu_x86_cpuid() reports OSXSAVE (bit 27).
    let mut s = cpu.new_state(true);
    s.cr4 |= CR4_OSXSAVE_MASK;
    assert_eq!(cpu.cpuid_for(&s, 1, 0)[2], 0xf7fa_3203 | (1 << 27));
}

#[test]
fn skylake_kvm_leaf7() {
    let cpu = kvm("Skylake-Client", permissive_host());
    // Only subleaf 0 words are set, so level_func7 (EAX) is 0.
    assert_eq!(cpu.cpuid(7, 0), [0, 0x001c_0fb9, 0, 0]);
    assert_eq!(cpu.cpuid(7, 1), [0; 4]);
}

#[test]
fn skylake_kvm_leaf_80000001() {
    let cpu = kvm("Skylake-Client", permissive_host());
    // Intel: no alias copy, and SYSCALL stays under KVM.
    assert_eq!(cpu.cpuid(0x8000_0001, 0), [0x0005_06e3, 0, 0x121, 0x2810_0800]);
    // x-vendor-cpuid-only-v2 hides the vendor in 0x80000000 for Intel.
    assert_eq!(cpu.cpuid(0x8000_0000, 0), [0x8000_0008, 0, 0, 0]);
    // 0x80000008: phys 40, virt 48, guest phys from the host (0 here).
    assert_eq!(cpu.cpuid(0x8000_0008, 0), [0x3028, 0, 0, 0]);
    assert_eq!(cpu.ucode_rev(), 0x1_0000_0000);
    // KVM paravirt features from kvm_default_props: kvmclock (bits 0 and
    // 3), nopiodelay (1), asyncpf (4), steal-time (5), pv-eoi (6),
    // clocksource-stable (24).
    assert_eq!(cpu.cpuid(0x4000_0000, 0), [0; 4]);
    assert_eq!(cpu.features()[FEAT_KVM], 0x0100_007b);
}

#[test]
fn skylake_versions() {
    // v2 (Skylake-Client-IBRS) adds spec-ctrl, CPUID[7].EDX bit 26.
    let cpu = kvm("Skylake-Client-IBRS", permissive_host());
    assert_eq!(cpu.cpuid(7, 0)[3], 1 << 26);
    // v3 also drops hle (bit 4) and rtm (bit 11).
    let cpu = kvm("Skylake-Client-v3", permissive_host());
    assert_eq!(cpu.cpuid(7, 0)[1], 0x001c_0fb9 & !((1 << 4) | (1 << 11)));
    assert!(X86Cpu::new("Skylake-Client-v9", Accel::Tcg).is_err());
}

#[test]
fn skylake_tcg_filters() {
    let cpu = tcg("Skylake-Client");
    // TCG has no TSC deadline timer, x2apic, PCID and friends; each
    // filtered bit gets one warning in feature_word_description() form.
    assert!(
        cpu.warnings().iter().any(|w| w
            == "TCG doesn't support requested feature: CPUID[eax=01h].ECX.tsc-deadline [bit 24]"),
        "{:?}",
        cpu.warnings()
    );
    assert_ne!(cpu.filtered_features()[FEAT_1_ECX], 0);
    assert_eq!(cpu.features()[FEAT_1_ECX] & (1 << 24), 0);
    // "vme=off" is applied by x86_tcg_default_props.
    assert_eq!(cpu.cpuid(1, 0)[3] & 2, 0);
    // SYSCALL is hidden for Intel under TCG outside long mode.
    assert_eq!(cpu.cpuid(0x8000_0001, 0)[3] & 0x800, 0);
    let mut s = cpu.new_state(true);
    s.hflags |= HF_LMA_MASK;
    assert_eq!(cpu.cpuid_for(&s, 0x8000_0001, 0)[3] & 0x800, 0x800);
}

// Feature parsing.

#[test]
fn plus_minus_and_aliases() {
    let mut cpu = X86Cpu::new("qemu64", Accel::Tcg).unwrap();
    cpu.parse_features("+sse4_1,-sse3,lahf_lm=off,+ssse3,sse4-2").unwrap();
    cpu.realize().unwrap();
    let r = cpu.cpuid(1, 0);
    // sse4.1 is bit 19, ssse3 bit 9, sse4.2 bit 20; pni (sse3) bit 0 off.
    assert_eq!(r[2], 0x8000_2001 & !1 | (1 << 19) | (1 << 9) | (1 << 20));
    // lahf_lm=off is the global "lahf-lm"; it clears 8000_0001.ECX bit 0.
    assert_eq!(cpu.cpuid(0x8000_0001, 0)[2], 0x4);
    assert!(cpu.has_feature("sse4.1"));
    assert!(cpu.has_feature("sse4_1"));
    assert!(!cpu.has_feature("pni"));
}

#[test]
fn plus_lahf_lm_alias() {
    let mut cpu = X86Cpu::new("kvm64", Accel::Tcg).unwrap();
    cpu.parse_features("+lahf_lm,+i64").unwrap();
    cpu.realize().unwrap();
    assert_eq!(cpu.cpuid(0x8000_0001, 0)[2] & 1, 1);
}

#[test]
fn unknown_feature_errors() {
    // A key=value pair becomes a global property; QEMU's error when it
    // is applied in object_apply_global_props().
    let mut cpu = X86Cpu::new("qemu64", Accel::Tcg).unwrap();
    let e = cpu.parse_features("foo=on").unwrap_err();
    assert_eq!(
        e.to_string(),
        "can't apply global qemu64-x86_64-cpu.foo=on: Property 'qemu64-x86_64-cpu.foo' not found"
    );
    // A bare name means name=on.
    let e = cpu.parse_features("foo").unwrap_err();
    assert_eq!(
        e.0,
        "can't apply global qemu64-x86_64-cpu.foo=on: Property 'qemu64-x86_64-cpu.foo' not found"
    );
    // +foo is only looked up in x86_cpu_expand_features() at realize.
    let mut cpu = X86Cpu::new("qemu64", Accel::Tcg).unwrap();
    cpu.parse_features("+foo").unwrap();
    assert_eq!(cpu.realize().unwrap_err().0, "Property 'qemu64-x86_64-cpu.foo' not found");
    let e = X86Cpu::new("pentium9", Accel::Tcg).unwrap_err();
    assert_eq!(e.0, "unable to find CPU model 'pentium9'");
}

#[test]
fn property_errors() {
    let mut cpu = X86Cpu::new("qemu64", Accel::Tcg).unwrap();
    assert_eq!(
        cpu.parse_features("sse3=maybe").unwrap_err().0,
        "can't apply global qemu64-x86_64-cpu.sse3=maybe: Parameter 'sse3' expects 'on' or 'off'"
    );
    assert_eq!(
        cpu.set_property("family", "300").unwrap_err().0,
        "parameter 'family' can be at most 270"
    );
    assert_eq!(
        cpu.set_property("vendor", "Intel").unwrap_err().0,
        "value of property 'vendor' must consist of exactly 12 characters"
    );
    cpu.parse_features("family=6,model=0x2a,stepping=7,vendor=GenuineIntel,level=7").unwrap();
    cpu.realize().unwrap();
    // 6 << 8 | 0xa << 4 | 0x2 << 16 | 7.
    assert_eq!(cpu.cpuid_version(), 0x0002_06a7);
    assert_eq!(cpu.cpuid(0, 0), [7, INTEL_EBX, INTEL_ECX, INTEL_EDX]);
    // family above 15 goes to the extended family field.
    let mut cpu = X86Cpu::new("qemu64", Accel::Tcg).unwrap();
    cpu.set_property("family", "23").unwrap();
    assert_eq!(cpu.cpuid_version() & 0x0ff0_0f00, 0x0080_0f00);
}

// Filtering against a fake host.

#[test]
fn filter_against_host_without_avx() {
    let mut host = permissive_host();
    host.supported[0].regs[2] &= !(1 << 28);
    let cpu = kvm("Skylake-Client", host.clone());
    assert_eq!(
        cpu.warnings(),
        ["host doesn't support requested feature: CPUID[eax=01h].ECX.avx [bit 28]"]
    );
    assert_eq!(cpu.filtered_features()[FEAT_1_ECX], 1 << 28);
    assert_eq!(cpu.cpuid(1, 0)[2], 0xf7fa_3203 & !(1 << 28));

    // check=off keeps quiet but still filters.
    let mut cpu = X86Cpu::new("Skylake-Client", Accel::Kvm(host.clone())).unwrap();
    cpu.parse_features("check=off").unwrap();
    cpu.realize().unwrap();
    assert!(cpu.warnings().is_empty());
    assert_eq!(cpu.filtered_features()[FEAT_1_ECX], 1 << 28);

    // enforce turns it into an error, after the warnings.
    let mut cpu = X86Cpu::new("Skylake-Client", Accel::Kvm(host.clone())).unwrap();
    cpu.parse_features("enforce").unwrap();
    assert_eq!(cpu.realize().unwrap_err().0, "Host doesn't support requested features");
    assert_eq!(cpu.warnings().len(), 1);

    // x-force-features keeps the bit in CPUID but still reports it.
    let mut cpu = X86Cpu::new("Skylake-Client", Accel::Kvm(host)).unwrap();
    cpu.parse_features("x-force-features=on").unwrap();
    cpu.realize().unwrap();
    assert_eq!(cpu.cpuid(1, 0)[2] & (1 << 28), 1 << 28);
}

#[test]
fn filter_leaf7_and_dependencies() {
    let mut host = permissive_host();
    // Host without avx2 (7.EBX bit 5) and rdseed (bit 18).
    host.supported[2].regs[1] &= !((1 << 5) | (1 << 18));
    let cpu = kvm("Skylake-Client", host);
    assert_eq!(
        cpu.warnings(),
        [
            "host doesn't support requested feature: CPUID[eax=07h,ecx=00h].EBX.avx2 [bit 5]",
            "host doesn't support requested feature: CPUID[eax=07h,ecx=00h].EBX.rdseed [bit 18]",
        ]
    );
    assert_eq!(cpu.cpuid(7, 0)[1], 0x001c_0fb9 & !((1 << 5) | (1 << 18)));
    assert_eq!(cpu.features()[FEAT_7_0_EBX], u64::from(cpu.cpuid(7, 0)[1]));
    assert_eq!(cpu.features()[FEAT_1_EDX], 0x078b_fbff);

    // feature_dependencies[]: MSR_IA32_ARCH_CAPABILITIES bits need
    // arch-capabilities (CPUID[7].EDX bit 29), which qemu64 lacks, so a
    // user-requested mds-no (bit 5) is dropped with a warning.
    let mut cpu = X86Cpu::new("qemu64", Accel::Tcg).unwrap();
    cpu.parse_features("+mds-no").unwrap();
    cpu.realize().unwrap();
    assert_eq!(
        cpu.warnings(),
        [
            "This feature depends on other features that were not requested: MSR(10Ah).mds-no [bit 5]"
        ]
    );
}

#[test]
fn max_under_tcg() {
    let cpu = tcg("max");
    // max_x86_cpu_initfn(): AuthenticAMD and the TCG model id.
    assert_eq!(cpu.cpuid(0, 0)[1], AMD_EBX);
    // Family is decided before expansion, when LM is not yet set: 6/6/3.
    assert_eq!(cpu.cpuid_version(), 0x0000_0663);
    assert!(cpu.has_feature("lm"));
    assert!(cpu.warnings().is_empty(), "{:?}", cpu.warnings());
    assert!(matches!(
        X86Cpu::new("host", Accel::Tcg).unwrap().realize(),
        Err(CpuError(ref m)) if m == "CPU model 'host' requires KVM or HVF"
    ));
}

#[test]
fn epyc_models() {
    let cpu = tcg("EPYC");
    // family 23 = 0xf + 8 -> 0xf00 | 8 << 20; model 1; stepping 2.
    assert_eq!(cpu.cpuid_version(), 0x0080_0f12);
    assert_eq!(cpu.cpuid(0x8000_0000, 0)[1], AMD_EBX);
    assert!(X86Cpu::new("EPYC-v4", Accel::Tcg).is_ok());
    // qemu64 has no versioned cache info, so legacy-cache=off fails.
    let mut cpu = X86Cpu::new("qemu64", Accel::Tcg).unwrap();
    cpu.set_property("legacy-cache", "off").unwrap();
    assert_eq!(cpu.realize().unwrap_err().0, "CPU model 'qemu64' doesn't support legacy-cache=off");
    // EPYC carries epyc_cache_info, so its L1d in 0x80000005 ECX is
    // 32 KiB, 8-way, 1 line per tag, 64-byte lines: 0x20080140.
    let cpu = tcg("EPYC");
    assert_eq!(cpu.cpuid(0x8000_0005, 0)[2], 0x2008_0140);
}

#[test]
fn topology_leaves() {
    let mut cpu = X86Cpu::new("Skylake-Client", Accel::Kvm(permissive_host())).unwrap();
    let topo = X86CpuTopoInfo {
        dies_per_pkg: 1,
        modules_per_die: 1,
        cores_per_module: 4,
        threads_per_core: 2,
    };
    cpu.set_topology(topo, 3);
    cpu.realize().unwrap();
    // Leaf 1 EBX: APIC 3 << 24, logical count 1 << pkg_offset(3) = 8.
    assert_eq!(cpu.cpuid(1, 0)[1], 0x0308_0800);
    // HT is set because there is more than one thread per package.
    assert_ne!(cpu.cpuid(1, 0)[3] & (1 << 28), 0);
    // Leaf 0xB: SMT level, 2 threads, shift 1; core level, 8, shift 3.
    assert_eq!(cpu.cpuid(0xb, 0), [1, 2, 0x100, 3]);
    assert_eq!(cpu.cpuid(0xb, 1), [3, 8, 0x201, 3]);
    assert_eq!(cpu.cpuid(0xb, 2), [0, 0, 2, 3]);
}
