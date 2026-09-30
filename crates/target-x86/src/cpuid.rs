// SPDX-License-Identifier: GPL-2.0-or-later

//! CPU models, feature words and the CPUID instruction.
//!
//! This is a port of the model and CPUID parts of `target/i386/cpu.c`,
//! together with the accelerator hooks from `target/i386/tcg/tcg-cpu.c`,
//! `target/i386/kvm/kvm-cpu.c` and `target/i386/host-cpu.c` that change
//! what the guest sees.
//!
//! [`X86Cpu`] follows the life of a QEMU `X86CPU` object:
//!
//! 1. [`X86Cpu::new`] is `instance_init`: it loads the model, applies the
//!    model version properties and the accelerator defaults.
//! 2. [`X86Cpu::parse_features`] takes the part of `-cpu` after the model
//!    name. `+feat` and `-feat` are kept for realize, `key=value` pairs are
//!    applied at once, the way QEMU applies them as global properties.
//! 3. [`X86Cpu::realize`] expands and filters the features and fixes the
//!    derived values (CPUID levels, physical address bits, caches).
//! 4. [`X86Cpu::cpuid`] answers CPUID queries, as `cpu_x86_cpuid()`.
//!
//! The KVM accelerator is represented by a [`HostCpuid`] snapshot, since
//! this crate does not talk to `/dev/kvm`.

pub mod cache;
pub mod host;
pub mod models;
pub mod parse;
pub mod topo;
pub mod words;
pub mod xsave;

use std::fmt;

use cache::{
    CpuCaches, LEGACY_AMD_CACHES, LEGACY_INTEL_CACHES, LEGACY_INTEL_CPUID2_CACHES, Regs,
    encode_cpuid2, encode_cpuid4, encode_cpuid8000001d, encode_cpuid80000005, encode_cpuid80000006,
};
use host::{HostCpuid, cpu_family};
use models::{ResolvedModel, find_model};
use parse::{parse_bool, parse_featurestr, parse_i64, parse_u8, parse_u32, parse_u64};
use topo::{TopoLevel, X86CpuTopoInfo};
use words::{
    CPUID_1F_ECX_TOPO_LEVEL_DIE, CPUID_1F_ECX_TOPO_LEVEL_MODULE, CPUID_7_0_EBX_INTEL_PT,
    CPUID_7_0_EBX_MPX, CPUID_7_0_EBX_SGX, CPUID_7_0_ECX_LA57, CPUID_7_0_ECX_OSPKE,
    CPUID_7_0_ECX_PKU, CPUID_7_0_ECX_SGX_LC, CPUID_7_0_EDX_AMX_TILE,
    CPUID_7_0_EDX_ARCH_CAPABILITIES, CPUID_7_0_EDX_ARCH_LBR, CPUID_7_1_EDX_APXF,
    CPUID_7_1_EDX_AVX10, CPUID_APM_INVTSC, CPUID_B_ECX_TOPO_LEVEL_CORE,
    CPUID_B_ECX_TOPO_LEVEL_INVALID, CPUID_B_ECX_TOPO_LEVEL_SMT, CPUID_EXT_MONITOR,
    CPUID_EXT_OSXSAVE, CPUID_EXT_PDCM, CPUID_EXT_XSAVE, CPUID_EXT2_AMD_ALIASES, CPUID_EXT2_LM,
    CPUID_EXT2_SYSCALL, CPUID_EXT3_CMP_LEG, CPUID_EXT3_SVM, CPUID_EXT3_TOPOEXT, CPUID_HT,
    CPUID_MCA, CPUID_MCE, CPUID_MWAIT_EMX, CPUID_MWAIT_IBE, CPUID_PAE, CPUID_PSE36,
    CPUID_XSAVE_XSAVES, CPUID_XSTATE_XCR0_MASK, CPUID_XSTATE_XSS_MASK, FEAT_1_ECX, FEAT_1_EDX,
    FEAT_1E_1_EAX, FEAT_6_EAX, FEAT_7_0_EBX, FEAT_7_0_ECX, FEAT_7_0_EDX, FEAT_7_1_EAX,
    FEAT_7_1_ECX, FEAT_7_1_EDX, FEAT_7_2_EDX, FEAT_14_0_ECX, FEAT_29_0_EBX, FEAT_8000_0001_ECX,
    FEAT_8000_0001_EDX, FEAT_8000_0007_EBX, FEAT_8000_0007_EDX, FEAT_8000_0008_EBX,
    FEAT_8000_0021_EAX, FEAT_8000_0021_EBX, FEAT_8000_0021_ECX, FEAT_8000_0022_EAX,
    FEAT_C000_0001_EDX, FEAT_KVM, FEAT_SVM, FEAT_XSAVE, FEAT_XSAVE_XCR0_HI, FEAT_XSAVE_XCR0_LO,
    FEAT_XSAVE_XSS_HI, FEAT_XSAVE_XSS_LO, FEATURE_DEPENDENCIES, FEATURE_WORD_INFO, FEATURE_WORDS,
    FeatureWordArray, Reg, WordKind, find_feature,
};
use xsave::{
    DEFAULT_EXT_SAVE_AREAS, ExtSaveAreas, XSAVE_STATE_AREA_COUNT, has_xsave_feature,
    kvm_init_offsets, tcg_init_offsets, trim_unsupported, xsave_area_size,
};

use crate::msr::{MCE_BANKS_DEF, MCG_CTL_P, MCG_LMCE_P, MCG_SER_P, MSR_IA32_UCODE_REV};
use crate::state::{
    CR4_OSXSAVE_MASK, CR4_PKE_MASK, HF_LMA_MASK, ResetConfig, X86CpuState, XSTATE_FP_MASK,
};

/// `QEMU_HW_VERSION`, used in the `max` model name.
const QEMU_HW_VERSION: &str = "2.5+";

/// `TCG_PHYS_ADDR_BITS`, the default `phys-bits` for 64-bit CPUs.
const TCG_PHYS_ADDR_BITS: u32 = 40;

/// `CPUID_8000_0022_EAX_PERFMON_V2`.
const CPUID_8000_0022_EAX_PERFMON_V2: u64 = 1 << 0;

/// `CPUID_14_0_ECX_LIP`.
const CPUID_14_0_ECX_LIP: u64 = 1 << 31;

/// `CACHE_NO_INVD_SHARING | CACHE_INCLUSIVE`, the `CPUID[8000_001D].EDX`
/// bits kept with `x-amd-topoext-features-only`.
const CACHE_TOPOEXT_EDX_MASK: u32 = 0x3;

/// Vendor strings QEMU tests for.
const VENDOR_INTEL: &str = "GenuineIntel";
const VENDOR_AMD: &str = "AuthenticAMD";
const VENDOR_ZHAOXIN1: &str = "CentaurHauls";
const VENDOR_ZHAOXIN2: &str = "  Shanghai  ";

/// An error, carrying the message QEMU reports (without the
/// `qemu-system-x86_64:` prefix).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuError(pub String);

impl fmt::Display for CpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CpuError {}

/// The accelerator the CPU runs under.
#[derive(Debug, Clone)]
pub enum Accel {
    /// Software emulation. Supported features are the `tcg` masks of the
    /// feature word table.
    Tcg,
    /// KVM. Supported features come from the snapshot.
    Kvm(HostCpuid),
}

impl Accel {
    fn is_kvm(&self) -> bool {
        matches!(self, Accel::Kvm(_))
    }
}

/// A property value, either parsed from a string (`-cpu` globals, model
/// definitions) or a boolean (`+feat` and `-feat`).
#[derive(Debug, Clone, Copy)]
enum PropValue<'a> {
    Str(&'a str),
    Bool(bool),
}

/// Registers of a vCPU that `cpu_x86_cpuid()` looks at.
#[derive(Debug, Clone, Copy)]
struct GuestRegs {
    cr4: u64,
    hflags: u32,
    xcr0: u64,
}

impl GuestRegs {
    /// The values right after reset.
    const RESET: GuestRegs = GuestRegs { cr4: 0, hflags: 0, xcr0: XSTATE_FP_MASK };
}

fn pack_vendor(s: &str) -> [u32; 3] {
    let b = s.as_bytes();
    let mut v = [0u32; 3];
    for (i, word) in v.iter_mut().enumerate() {
        for j in 0..4 {
            *word |= u32::from(b.get(i * 4 + j).copied().unwrap_or(0)) << (8 * j);
        }
    }
    v
}

fn pack_model_id(s: &str) -> [u32; 12] {
    let b = s.as_bytes();
    let mut m = [0u32; 12];
    for i in 0..48 {
        let c = u32::from(b.get(i).copied().unwrap_or(0));
        m[i >> 2] |= c << (8 * (i & 3));
    }
    m
}

fn reg_of(r: Regs, reg: Reg) -> u32 {
    match reg {
        Reg::Eax => r[0],
        Reg::Ebx => r[1],
        Reg::Ecx => r[2],
        Reg::Edx => r[3],
    }
}

/// An x86 CPU object: model, properties, feature words and the values
/// derived from them at realize time.
#[derive(Debug, Clone)]
pub struct X86Cpu {
    /// Model name as given, used for the QOM type name.
    name: String,
    model: Option<ResolvedModel>,
    accel: Accel,
    max_features: bool,
    host_cpuid_required: bool,
    realized: bool,

    features: FeatureWordArray,
    user_features: FeatureWordArray,
    filtered_features: FeatureWordArray,
    plus_features: Vec<String>,
    minus_features: Vec<String>,

    cpuid_level: u32,
    cpuid_xlevel: u32,
    cpuid_xlevel2: u32,
    cpuid_level_func7: u32,
    cpuid_min_level: u32,
    cpuid_min_xlevel: u32,
    cpuid_min_xlevel2: u32,
    cpuid_min_level_func7: u32,
    vendor: [u32; 3],
    version: u32,
    model_id: [u32; 12],
    tsc_khz: i64,
    user_tsc_khz: i64,

    topo: X86CpuTopoInfo,
    apic_id: u32,
    ext_save_areas: ExtSaveAreas,
    cache_info: CpuCaches,
    legacy_cpuid2_cache: bool,
    legacy_vendor_cache: bool,
    mwait: Regs,
    mcg_cap: u64,

    check_cpuid: bool,
    enforce_cpuid: bool,
    force_features: bool,
    expose_kvm: bool,
    expose_tcg: bool,
    enable_pmu: bool,
    enable_lmce: bool,
    enable_l3_cache: bool,
    enable_cpuid_0xb: bool,
    vendor_cpuid_only: bool,
    vendor_cpuid_only_v2: bool,
    amd_topoext_features_only: bool,
    legacy_cache: bool,
    consistent_cache: bool,
    legacy_multi_node: bool,
    l1_cache_per_core: bool,
    force_cpuid_0x1f: bool,
    arch_cap_always_on: bool,
    pdcm_on_even_without_pmu: bool,
    host_phys_bits: bool,
    migratable: bool,
    phys_bits: u32,
    /// `u32::MAX` stands for QEMU's -1, "not set".
    guest_phys_bits: u32,
    host_phys_bits_limit: u8,
    ucode_rev: u64,

    warnings: Vec<String>,
}

impl X86Cpu {
    /// Creates a CPU of model `model`, as `object_new()` on the model's
    /// class: `x86_cpu_initfn()`, the accelerator `instance_init` hook and,
    /// for `max` and `host`, `max_x86_cpu_initfn()`.
    ///
    /// Models are looked up the way QEMU registers them: the base name
    /// (version 1 on the PC machines), `NAME-vN` and version aliases.
    pub fn new(model: &str, accel: Accel) -> Result<Self, CpuError> {
        let (resolved, max_features, host_cpuid_required) = match model {
            "max" => (None, true, false),
            "host" => (None, true, true),
            _ => match find_model(model) {
                Some(r) => (Some(r), false, false),
                None => {
                    return Err(CpuError(format!("unable to find CPU model '{model}'")));
                }
            },
        };
        let mut cpu = X86Cpu {
            name: model.to_string(),
            model: resolved,
            accel,
            max_features,
            host_cpuid_required,
            realized: false,
            features: [0; FEATURE_WORDS],
            user_features: [0; FEATURE_WORDS],
            filtered_features: [0; FEATURE_WORDS],
            plus_features: Vec::new(),
            minus_features: Vec::new(),
            cpuid_level: u32::MAX,
            cpuid_xlevel: u32::MAX,
            cpuid_xlevel2: u32::MAX,
            cpuid_level_func7: u32::MAX,
            cpuid_min_level: 0,
            cpuid_min_xlevel: 0,
            cpuid_min_xlevel2: 0,
            cpuid_min_level_func7: 0,
            vendor: [0; 3],
            version: 0,
            model_id: [0; 12],
            tsc_khz: 0,
            user_tsc_khz: 0,
            topo: X86CpuTopoInfo::default(),
            apic_id: 0,
            ext_save_areas: DEFAULT_EXT_SAVE_AREAS,
            cache_info: LEGACY_INTEL_CACHES,
            legacy_cpuid2_cache: false,
            legacy_vendor_cache: false,
            mwait: [0; 4],
            mcg_cap: 0,
            check_cpuid: true,
            enforce_cpuid: false,
            force_features: false,
            expose_kvm: true,
            expose_tcg: true,
            enable_pmu: false,
            enable_lmce: false,
            enable_l3_cache: true,
            enable_cpuid_0xb: true,
            vendor_cpuid_only: true,
            vendor_cpuid_only_v2: true,
            amd_topoext_features_only: true,
            legacy_cache: true,
            consistent_cache: true,
            legacy_multi_node: false,
            l1_cache_per_core: true,
            force_cpuid_0x1f: false,
            arch_cap_always_on: false,
            pdcm_on_even_without_pmu: false,
            host_phys_bits: false,
            // Only max and host have the property; the field is zero for
            // the named models.
            migratable: max_features,
            phys_bits: 0,
            guest_phys_bits: u32::MAX,
            host_phys_bits_limit: 0,
            ucode_rev: 0,
            warnings: Vec::new(),
        };
        cpu.instance_init().map_err(CpuError)?;
        Ok(cpu)
    }

    fn instance_init(&mut self) -> Result<(), String> {
        if let Some(m) = self.model {
            self.load_model(m)?;
        }
        self.init_xsave();
        match self.accel.clone() {
            Accel::Tcg => {
                if self.model.is_some() {
                    // x86_tcg_default_props
                    self.set_prop("vme", PropValue::Str("off"))?;
                }
                tcg_init_offsets(&mut self.ext_save_areas);
            }
            Accel::Kvm(host) => {
                // host_cpu_instance_init()
                self.set_prop("vendor", PropValue::Str(&host.vendor()))?;
                if self.max_features {
                    let (family, model, stepping) = host.fms();
                    self.host_phys_bits = true;
                    self.set_family(u64::from(family))?;
                    self.set_model(u64::from(model))?;
                    self.set_stepping(u64::from(stepping))?;
                    self.model_id = pack_model_id(&host.model_id());
                }
                if self.model.is_some() {
                    let in_kernel = host.irqchip.in_kernel();
                    let split = host.irqchip == crate::state::IrqchipMode::Split;
                    let defaults = [
                        ("kvmclock", "on"),
                        ("kvm-nopiodelay", "on"),
                        ("kvm-asyncpf", "on"),
                        ("kvm-steal-time", "on"),
                        ("kvm-pv-eoi", "on"),
                        ("kvmclock-stable-bit", "on"),
                        ("x2apic", if in_kernel { "on" } else { "off" }),
                        ("kvm-msi-ext-dest-id", if split { "on" } else { "off" }),
                        ("acpi", "off"),
                        ("monitor", "off"),
                        ("svm", "off"),
                    ];
                    for (name, value) in defaults {
                        self.set_prop(name, PropValue::Str(value))?;
                    }
                }
                if self.max_features {
                    // kvm_cpu_max_instance_init()
                    self.enable_pmu = true;
                    if host.lmce_supported {
                        self.enable_lmce = true;
                    }
                    self.cpuid_min_level = host.supported_cpuid(0, 0, Reg::Eax);
                    self.cpuid_min_xlevel = host.supported_cpuid(0x8000_0000, 0, Reg::Eax);
                    self.cpuid_min_xlevel2 = host.supported_cpuid(0xC000_0000, 0, Reg::Eax);
                }
                kvm_init_offsets(&mut self.ext_save_areas, |i| {
                    let r = host.host_cpuid(0xd, i);
                    (r[0], r[1], r[2])
                });
            }
        }
        if self.max_features {
            // max_x86_cpu_initfn()
            if self.vendor[0] == 0 {
                self.vendor = pack_vendor(VENDOR_AMD);
            }
            if self.model_id[0] == 0 {
                self.model_id = pack_model_id(&format!("QEMU TCG CPU version {QEMU_HW_VERSION}"));
            }
        }
        Ok(())
    }

    /// `x86_cpu_load_model()`.
    fn load_model(&mut self, m: ResolvedModel) -> Result<(), String> {
        let def = m.def;
        self.cpuid_min_level = def.level;
        self.cpuid_min_xlevel = def.xlevel;
        self.set_family(u64::from(def.family))?;
        self.set_model(u64::from(def.model))?;
        self.set_stepping(u64::from(def.stepping))?;
        self.model_id = pack_model_id(def.model_id);
        self.features = [0; FEATURE_WORDS];
        for &(w, mask) in def.features {
            self.features[w] = mask;
        }
        self.legacy_cache = def.cache_for_version(m.version).is_none();
        self.features[FEAT_1_ECX] |= words::CPUID_EXT_HYPERVISOR;
        self.vendor = pack_vendor(def.vendor);
        // x86_cpu_apply_version_props()
        for v in def.versions {
            for &(name, value) in v.props {
                self.set_prop(name, PropValue::Str(value))?;
            }
            if v.version == m.version {
                break;
            }
        }
        self.user_features = [0; FEATURE_WORDS];
        Ok(())
    }

    /// `x86_cpu_init_xsave()`.
    fn init_xsave(&mut self) {
        let supported = (self.supported_word(false, FEAT_XSAVE_XCR0_HI) << 32)
            | self.supported_word(false, FEAT_XSAVE_XCR0_LO)
            | (self.supported_word(false, FEAT_XSAVE_XSS_HI) << 32)
            | self.supported_word(false, FEAT_XSAVE_XSS_LO);
        trim_unsupported(&mut self.ext_save_areas, supported);
    }

    /// The QOM type name, for example `qemu64-x86_64-cpu`.
    pub fn typename(&self) -> String {
        format!("{}-x86_64-cpu", self.name)
    }

    /// Applies the feature part of `-cpu`, everything after the first
    /// comma, as `x86_cpu_parse_featurestr()` followed by the global
    /// property application at object creation.
    pub fn parse_features(&mut self, features: &str) -> Result<(), CpuError> {
        let parsed = parse_featurestr(features).map_err(CpuError)?;
        self.warnings.extend(parsed.warnings);
        self.plus_features.extend(parsed.plus);
        self.minus_features.extend(parsed.minus);
        for (name, value) in &parsed.globals {
            self.set_prop(name, PropValue::Str(value)).map_err(|e| {
                CpuError(format!(
                    "can't apply global {}.{}={}: {}",
                    self.typename(),
                    name,
                    value,
                    e
                ))
            })?;
        }
        Ok(())
    }

    /// Sets a property from its string form, as `object_property_parse()`.
    ///
    /// Feature names (and their aliases) are boolean properties; the other
    /// names are the `X86CPU` properties this port supports.
    pub fn set_property(&mut self, name: &str, value: &str) -> Result<(), CpuError> {
        self.set_prop(name, PropValue::Str(value)).map_err(CpuError)
    }

    /// Sets the topology and APIC ID, as the PC machine does before realize.
    pub fn set_topology(&mut self, topo: X86CpuTopoInfo, apic_id: u32) {
        self.topo = topo;
        self.apic_id = apic_id;
    }

    fn not_found(&self, name: &str) -> String {
        format!("Property '{}.{}' not found", self.typename(), name)
    }

    fn set_prop(&mut self, name: &str, v: PropValue<'_>) -> Result<(), String> {
        let bool_field = match name {
            "check" => Some(&mut self.check_cpuid),
            "enforce" => Some(&mut self.enforce_cpuid),
            "x-force-features" => Some(&mut self.force_features),
            "kvm" => Some(&mut self.expose_kvm),
            "tcg-cpuid" => Some(&mut self.expose_tcg),
            "pmu" => Some(&mut self.enable_pmu),
            "lmce" => Some(&mut self.enable_lmce),
            "l3-cache" => Some(&mut self.enable_l3_cache),
            "cpuid-0xb" => Some(&mut self.enable_cpuid_0xb),
            "x-vendor-cpuid-only" => Some(&mut self.vendor_cpuid_only),
            "x-vendor-cpuid-only-v2" => Some(&mut self.vendor_cpuid_only_v2),
            "x-amd-topoext-features-only" => Some(&mut self.amd_topoext_features_only),
            "legacy-cache" => Some(&mut self.legacy_cache),
            "x-consistent-cache" => Some(&mut self.consistent_cache),
            "legacy-multi-node" => Some(&mut self.legacy_multi_node),
            "x-l1-cache-per-thread" => Some(&mut self.l1_cache_per_core),
            "x-force-cpuid-0x1f" => Some(&mut self.force_cpuid_0x1f),
            "x-arch-cap-always-on" => Some(&mut self.arch_cap_always_on),
            "x-pdcm-on-even-without-pmu" => Some(&mut self.pdcm_on_even_without_pmu),
            "host-phys-bits" => Some(&mut self.host_phys_bits),
            "migratable" if self.max_features => Some(&mut self.migratable),
            _ => None,
        };
        if let Some(field) = bool_field {
            *field = match v {
                PropValue::Bool(b) => b,
                PropValue::Str(s) => parse_bool(name, s)?,
            };
            return Ok(());
        }

        let integer = |v: PropValue<'_>| -> Result<String, String> {
            match v {
                PropValue::Str(s) => Ok(s.to_string()),
                PropValue::Bool(_) => {
                    Err(format!("Invalid parameter type for '{name}', expected: integer"))
                }
            }
        };
        let u32_field = match name {
            "level" => Some(&mut self.cpuid_level),
            "xlevel" => Some(&mut self.cpuid_xlevel),
            "xlevel2" => Some(&mut self.cpuid_xlevel2),
            "level-func7" => Some(&mut self.cpuid_level_func7),
            "min-level" => Some(&mut self.cpuid_min_level),
            "min-xlevel" => Some(&mut self.cpuid_min_xlevel),
            "min-xlevel2" => Some(&mut self.cpuid_min_xlevel2),
            "phys-bits" => Some(&mut self.phys_bits),
            "guest-phys-bits" => Some(&mut self.guest_phys_bits),
            "apic-id" => Some(&mut self.apic_id),
            _ => None,
        };
        if let Some(field) = u32_field {
            *field = parse_u32(name, &integer(v)?)?;
            return Ok(());
        }
        match name {
            "host-phys-bits-limit" => {
                self.host_phys_bits_limit = parse_u8(name, &integer(v)?)?;
            }
            "ucode-rev" => self.ucode_rev = parse_u64(name, &integer(v)?)?,
            "family" => self.set_family(parse_u64(name, &integer(v)?)?)?,
            "model" => self.set_model(parse_u64(name, &integer(v)?)?)?,
            "stepping" => self.set_stepping(parse_u64(name, &integer(v)?)?)?,
            "tsc-frequency" => {
                let value = parse_i64(name, &integer(v)?)?;
                if value < 0 {
                    return Err(format!("parameter '{name}' can be at most {}", i64::MAX));
                }
                self.tsc_khz = value / 1000;
                self.user_tsc_khz = value / 1000;
            }
            "vendor" | "model-id" => {
                let PropValue::Str(s) = v else {
                    return Err(format!("Invalid parameter type for '{name}', expected: string"));
                };
                if name == "model-id" {
                    self.model_id = pack_model_id(s);
                } else if s.len() != 12 {
                    return Err("value of property 'vendor' must consist of exactly 12 characters"
                        .to_string());
                } else {
                    self.vendor = pack_vendor(s);
                }
            }
            _ => {
                let Some((w, mask)) = find_feature(name) else {
                    return Err(self.not_found(name));
                };
                let on = match v {
                    PropValue::Bool(b) => b,
                    PropValue::Str(s) => parse_bool(name, s)?,
                };
                if on {
                    self.features[w] |= mask;
                } else {
                    self.features[w] &= !mask;
                }
                self.user_features[w] |= mask;
            }
        }
        Ok(())
    }

    fn set_family(&mut self, value: u64) -> Result<(), String> {
        const MAX: u64 = 0xff + 0xf;
        if value > MAX {
            return Err(format!("parameter 'family' can be at most {MAX}"));
        }
        let value = value as u32;
        self.version &= !0x0ff0_0f00;
        if value > 0x0f {
            self.version |= 0xf00 | ((value - 0x0f) << 20);
        } else {
            self.version |= value << 8;
        }
        Ok(())
    }

    fn set_model(&mut self, value: u64) -> Result<(), String> {
        const MAX: u64 = 0xff;
        if value > MAX {
            return Err(format!("parameter 'model' can be at most {MAX}"));
        }
        let value = value as u32;
        self.version &= !0xf00f0;
        self.version |= ((value & 0xf) << 4) | ((value >> 4) << 16);
        Ok(())
    }

    fn set_stepping(&mut self, value: u64) -> Result<(), String> {
        const MAX: u64 = 0xf;
        if value > MAX {
            return Err(format!("parameter 'stepping' can be at most {MAX}"));
        }
        self.version &= !0xf;
        self.version |= value as u32 & 0xf;
        Ok(())
    }

    fn vendor_is(&self, s: &str) -> bool {
        self.vendor == pack_vendor(s)
    }

    fn is_intel(&self) -> bool {
        self.vendor_is(VENDOR_INTEL)
    }

    fn is_amd(&self) -> bool {
        self.vendor_is(VENDOR_AMD)
    }

    fn is_zhaoxin(&self) -> bool {
        self.vendor_is(VENDOR_ZHAOXIN1) || self.vendor_is(VENDOR_ZHAOXIN2)
    }

    /// `x86_cpu_get_migratable_flags()`.
    fn migratable_flags(&self, w: usize) -> u64 {
        let wi = &FEATURE_WORD_INFO[w];
        let mut r = 0u64;
        for i in 0..64 {
            let f = 1u64 << i;
            if wi.migratable & f != 0 || (wi.bit_name(i).is_some() && wi.unmigratable & f == 0) {
                r |= f;
            }
        }
        if w == FEAT_8000_0007_EDX && self.user_tsc_khz != 0 {
            r |= CPUID_APM_INVTSC;
        }
        r
    }

    /// `x86_cpu_get_supported_feature_word()`. `with_cpu` is QEMU's
    /// non-NULL `cpu` argument, which adds the vendor and migration checks.
    fn supported_word(&self, with_cpu: bool, w: usize) -> u64 {
        let wi = &FEATURE_WORD_INFO[w];
        let mut r = match &self.accel {
            Accel::Tcg => wi.tcg,
            Accel::Kvm(h) => match wi.kind {
                WordKind::Cpuid { eax, ecx, reg } => {
                    u64::from(h.supported_cpuid(eax, ecx.unwrap_or(0), reg))
                }
                WordKind::Msr { index } => h.supported_msr_feature(index),
            },
        };
        // check_sgx_support() needs an SGX EPC backend, which is not ported.
        let unavail = match w {
            FEAT_8000_0007_EBX if with_cpu && !self.is_amd() => u64::MAX,
            FEAT_7_0_EBX => CPUID_7_0_EBX_SGX,
            FEAT_7_0_ECX => CPUID_7_0_ECX_SGX_LC,
            FEAT_7_0_EDX if with_cpu && self.is_amd() && !self.arch_cap_always_on => {
                CPUID_7_0_EDX_ARCH_CAPABILITIES
            }
            _ => 0,
        };
        r &= !unavail;
        if with_cpu && self.migratable {
            r &= self.migratable_flags(w);
        }
        r
    }

    fn supported_regs(&self, function: u32, index: u32) -> Regs {
        match &self.accel {
            Accel::Tcg => [0; 4],
            Accel::Kvm(h) => [
                h.supported_cpuid(function, index, Reg::Eax),
                h.supported_cpuid(function, index, Reg::Ebx),
                h.supported_cpuid(function, index, Reg::Ecx),
                h.supported_cpuid(function, index, Reg::Edx),
            ],
        }
    }

    /// `mark_unavailable_features()`.
    fn mark_unavailable(&mut self, w: usize, mask: u64, prefix: Option<&str>) {
        if !self.force_features {
            self.features[w] &= !mask;
        }
        self.filtered_features[w] |= mask;
        let Some(prefix) = prefix else {
            return;
        };
        let wi = &FEATURE_WORD_INFO[w];
        let desc = wi.description();
        for i in 0..64 {
            if mask & (1u64 << i) != 0 {
                let name = wi.bit_name(i);
                self.warnings.push(format!(
                    "{prefix}: {desc}{}{} [bit {i}]",
                    if name.is_some() { "." } else { "" },
                    name.unwrap_or("")
                ));
            }
        }
    }

    fn adjust_level(min: &mut u32, value: u32) {
        if *min < value {
            *min = value;
        }
    }

    /// `x86_cpu_adjust_feat_level()`.
    fn adjust_feat_level(&mut self, w: usize) {
        let WordKind::Cpuid { eax, ecx, .. } = FEATURE_WORD_INFO[w].kind else {
            return;
        };
        if self.features[w] == 0 {
            return;
        }
        match eax & 0xF000_0000 {
            0 => Self::adjust_level(&mut self.cpuid_min_level, eax),
            0x8000_0000 => Self::adjust_level(&mut self.cpuid_min_xlevel, eax),
            0xC000_0000 => Self::adjust_level(&mut self.cpuid_min_xlevel2, eax),
            _ => {}
        }
        if eax == 7 {
            Self::adjust_level(&mut self.cpuid_min_level_func7, ecx.unwrap_or(0));
        }
    }

    /// `x86_has_cpuid_0x1f()`.
    fn has_cpuid_0x1f(&self) -> bool {
        self.force_cpuid_0x1f || self.topo.has_extended_topo()
    }

    fn xcr0_components(&self) -> u64 {
        (self.features[FEAT_XSAVE_XCR0_HI] << 32) | self.features[FEAT_XSAVE_XCR0_LO]
    }

    fn xss_components(&self) -> u64 {
        (self.features[FEAT_XSAVE_XSS_HI] << 32) | self.features[FEAT_XSAVE_XSS_LO]
    }

    /// `x86_cpu_enable_xsave_components()`.
    fn enable_xsave_components(&mut self) {
        if self.features[FEAT_1_ECX] & CPUID_EXT_XSAVE == 0 {
            self.features[FEAT_XSAVE_XCR0_LO] = 0;
            self.features[FEAT_XSAVE_XCR0_HI] = 0;
            self.features[FEAT_XSAVE_XSS_LO] = 0;
            self.features[FEAT_XSAVE_XSS_HI] = 0;
            return;
        }
        let mut mask = 0u64;
        for (i, esa) in self.ext_save_areas.iter().enumerate() {
            if CPUID_XSTATE_XSS_MASK & (1 << i) != 0
                && self.features[FEAT_XSAVE] & CPUID_XSAVE_XSAVES == 0
            {
                continue;
            }
            if has_xsave_feature(&self.features, esa) {
                mask |= 1 << i;
            }
        }
        self.features[FEAT_XSAVE_XCR0_LO] = mask & CPUID_XSTATE_XCR0_MASK & 0xffff_ffff;
        self.features[FEAT_XSAVE_XCR0_HI] = (mask & CPUID_XSTATE_XCR0_MASK) >> 32;
        self.features[FEAT_XSAVE_XSS_LO] = mask & CPUID_XSTATE_XSS_MASK & 0xffff_ffff;
        self.features[FEAT_XSAVE_XSS_HI] = (mask & CPUID_XSTATE_XSS_MASK) >> 32;
    }

    /// `x86_cpu_expand_features()`.
    fn expand_features(&mut self) -> Result<(), String> {
        for name in self.plus_features.clone() {
            self.set_prop(&name, PropValue::Bool(true))?;
        }
        for name in self.minus_features.clone() {
            self.set_prop(&name, PropValue::Bool(false))?;
        }

        if self.max_features {
            for (w, wi) in FEATURE_WORD_INFO.iter().enumerate() {
                self.features[w] |=
                    self.supported_word(true, w) & !self.user_features[w] & !wi.no_autoenable;
            }
        }

        if self.topo.threads_per_pkg() > 1 {
            self.features[FEAT_1_EDX] |= CPUID_HT;
            if !self.is_intel() && !self.is_zhaoxin() {
                self.features[FEAT_8000_0001_ECX] |= CPUID_EXT3_CMP_LEG;
            }
        }

        if !self.enable_pmu {
            if !self.pdcm_on_even_without_pmu {
                self.features[FEAT_1_ECX] &= !CPUID_EXT_PDCM;
            }
            self.features[FEAT_7_0_EDX] &= !CPUID_7_0_EDX_ARCH_LBR;
        }

        for d in FEATURE_DEPENDENCIES {
            if self.features[d.from.0] & d.from.1 == 0 {
                let (w, mask) = d.to;
                let unavailable = self.features[w] & mask;
                self.mark_unavailable(
                    w,
                    unavailable & self.user_features[w],
                    Some("This feature depends on other features that were not requested"),
                );
                self.features[w] &= !unavailable;
            }
        }

        if !self.accel.is_kvm() || !self.expose_kvm {
            self.features[FEAT_KVM] = 0;
        }

        if self.features[FEAT_7_0_EBX] & CPUID_7_0_EBX_MPX != 0
            && self.features[FEAT_7_1_EDX] & CPUID_7_1_EDX_APXF != 0
        {
            self.mark_unavailable(
                FEAT_7_0_EBX,
                CPUID_7_0_EBX_MPX,
                Some("this feature conflicts with APX"),
            );
            self.mark_unavailable(
                FEAT_7_1_EDX,
                CPUID_7_1_EDX_APXF,
                Some("this feature conflicts with MPX"),
            );
        }

        self.enable_xsave_components();

        for w in [
            FEAT_7_0_EBX,
            FEAT_1_EDX,
            FEAT_1_ECX,
            FEAT_6_EAX,
            FEAT_7_0_ECX,
            FEAT_7_1_EAX,
            FEAT_7_1_ECX,
            FEAT_7_1_EDX,
            FEAT_7_2_EDX,
            FEAT_8000_0001_EDX,
            FEAT_8000_0001_ECX,
            FEAT_8000_0007_EDX,
            FEAT_8000_0008_EBX,
            FEAT_C000_0001_EDX,
            FEAT_SVM,
            FEAT_XSAVE,
        ] {
            self.adjust_feat_level(w);
        }

        if self.features[FEAT_7_0_EBX] & CPUID_7_0_EBX_INTEL_PT != 0 {
            Self::adjust_level(&mut self.cpuid_min_level, 0x14);
        }
        if self.has_cpuid_0x1f() && (self.is_intel() || !self.vendor_cpuid_only) {
            Self::adjust_level(&mut self.cpuid_min_level, 0x1F);
        }
        if self.features[FEAT_7_1_EDX] & CPUID_7_1_EDX_AVX10 != 0 {
            Self::adjust_level(&mut self.cpuid_min_level, 0x24);
        }
        if self.features[FEAT_7_1_EDX] & CPUID_7_1_EDX_APXF != 0 {
            Self::adjust_level(&mut self.cpuid_min_level, 0x29);
        }
        if self.features[FEAT_8000_0001_ECX] & CPUID_EXT3_SVM != 0 {
            Self::adjust_level(&mut self.cpuid_min_xlevel, 0x8000_000A);
        }
        if self.features[FEAT_8000_0021_EAX] != 0 {
            Self::adjust_level(&mut self.cpuid_min_xlevel, 0x8000_0021);
        }
        if self.features[FEAT_7_0_EBX] & CPUID_7_0_EBX_SGX != 0 {
            Self::adjust_level(&mut self.cpuid_min_level, 0x12);
        }

        if self.cpuid_level_func7 == u32::MAX {
            self.cpuid_level_func7 = self.cpuid_min_level_func7;
        }
        if self.cpuid_level == u32::MAX {
            self.cpuid_level = self.cpuid_min_level;
        }
        if self.cpuid_xlevel == u32::MAX {
            self.cpuid_xlevel = self.cpuid_min_xlevel;
        }
        if self.cpuid_xlevel2 == u32::MAX {
            self.cpuid_xlevel2 = self.cpuid_min_xlevel2;
        }
        Ok(())
    }

    /// `x86_cpu_filter_features()`. Returns whether anything was filtered.
    fn filter_features(&mut self, verbose: bool) -> bool {
        let prefix = if !verbose {
            None
        } else if self.accel.is_kvm() {
            Some("host doesn't support requested feature")
        } else {
            Some("TCG doesn't support requested feature")
        };
        for w in 0..FEATURE_WORDS {
            let host = self.supported_word(false, w);
            let unavailable = self.features[w] & !host;
            self.mark_unavailable(w, unavailable, prefix);
        }
        self.filtered_features.iter().any(|&f| f != 0)
    }

    /// Realizes the CPU: `max_x86_cpu_realize()` for `max` and `host`, then
    /// `x86_cpu_realizefn()` up to the point where the CPUID data is final,
    /// including the accelerator `cpu_target_realize` hook.
    ///
    /// Warnings are collected in [`X86Cpu::warnings`].
    pub fn realize(&mut self) -> Result<(), CpuError> {
        self.realize_inner().map_err(CpuError)
    }

    fn realize_inner(&mut self) -> Result<(), String> {
        if self.max_features && cpu_family(self.version) == 0 {
            if self.features[FEAT_8000_0001_EDX] & CPUID_EXT2_LM != 0 {
                self.set_family(15)?;
                self.set_model(107)?;
                self.set_stepping(1)?;
            } else {
                self.set_family(6)?;
                self.set_model(6)?;
                self.set_stepping(3)?;
            }
        }

        if !self.vendor_cpuid_only && self.vendor_cpuid_only_v2 {
            return Err(
                "x-vendor-cpuid-only-v2 property depends on x-vendor-cpuid-only".to_string()
            );
        }

        self.expand_features()?;

        if self.filter_features(self.check_cpuid || self.enforce_cpuid) && self.enforce_cpuid {
            return Err(if self.accel.is_kvm() {
                "Host doesn't support requested features".to_string()
            } else {
                "TCG doesn't support requested features".to_string()
            });
        }

        if self.is_amd() {
            self.features[FEAT_8000_0001_EDX] &= !CPUID_EXT2_AMD_ALIASES;
            self.features[FEAT_8000_0001_EDX] |= self.features[FEAT_1_EDX] & CPUID_EXT2_AMD_ALIASES;
        }

        if let Accel::Kvm(host) = self.accel.clone() {
            self.kvm_realize(&host);
        }

        if self.host_cpuid_required && !self.accel.is_kvm() {
            return Err(format!("CPU model '{}' requires KVM or HVF", self.name));
        }

        if self.guest_phys_bits == u32::MAX {
            self.guest_phys_bits = 0;
        }

        if self.ucode_rev == 0 {
            self.ucode_rev = if self.is_amd() { 0x0100_0065 } else { 0x1_0000_0000 };
        }

        self.mwait[2] |= (CPUID_MWAIT_EMX | CPUID_MWAIT_IBE) as u32;

        if self.is_amd()
            && self.features[FEAT_8000_0001_ECX] & CPUID_EXT3_TOPOEXT == 0
            && self.topo.threads_per_core > 1
        {
            self.warnings.push(format!(
                "This family of AMD CPU doesn't support hyperthreading({}). Please configure -smp options properly or try enabling topoext feature.",
                self.topo.threads_per_core
            ));
        }

        if self.features[FEAT_8000_0001_EDX] & CPUID_EXT2_LM != 0 {
            if self.phys_bits != 0 && self.phys_bits < 32 {
                return Err(format!("phys-bits should be at least 32 (but is {})", self.phys_bits));
            }
            if self.phys_bits == 0 {
                self.phys_bits = TCG_PHYS_ADDR_BITS;
            }
            if self.guest_phys_bits != 0
                && (self.guest_phys_bits > self.phys_bits || self.guest_phys_bits < 32)
            {
                return Err(format!(
                    "guest-phys-bits should be between 32 and {}  (but is {})",
                    self.phys_bits, self.guest_phys_bits
                ));
            }
        } else {
            if self.phys_bits != 0 {
                return Err("phys-bits is not user-configurable in 32 bit".to_string());
            }
            if self.guest_phys_bits != 0 {
                return Err("guest-phys-bits is not user-configurable in 32 bit".to_string());
            }
            self.phys_bits =
                if self.features[FEAT_1_EDX] & (CPUID_PSE36 | CPUID_PAE) != 0 { 36 } else { 32 };
        }

        if !self.legacy_cache {
            let cache = self.model.and_then(|m| m.def.cache_for_version(m.version));
            let Some(cache) = cache else {
                return Err(format!("CPU model '{}' doesn't support legacy-cache=off", self.name));
            };
            self.cache_info = cache.caches();
        } else {
            if !self.consistent_cache {
                self.legacy_cpuid2_cache = true;
            }
            if !self.vendor_cpuid_only_v2 {
                self.legacy_vendor_cache = true;
            }
            self.cache_info = if self.is_amd() { LEGACY_AMD_CACHES } else { LEGACY_INTEL_CACHES };
        }

        // mce_init()
        if cpu_family(self.version) >= 6
            && self.features[FEAT_1_EDX] & (CPUID_MCE | CPUID_MCA) == (CPUID_MCE | CPUID_MCA)
        {
            self.mcg_cap = MCG_CTL_P
                | MCG_SER_P
                | MCE_BANKS_DEF
                | if self.enable_lmce { MCG_LMCE_P } else { 0 };
        }

        self.realized = true;
        Ok(())
    }

    /// `kvm_cpu_realizefn()` and `host_cpu_realizefn()`.
    fn kvm_realize(&mut self, host: &HostCpuid) {
        if self.max_features && self.ucode_rev == 0 {
            self.ucode_rev = host.supported_msr_feature(MSR_IA32_UCODE_REV);
        }
        if self.features[FEAT_8000_0001_EDX] & CPUID_EXT2_LM == 0 {
            return;
        }
        // host_cpu_adjust_phys_bits()
        let host_phys_bits = host.phys_bits();
        let mut phys_bits = self.phys_bits;
        if phys_bits != host_phys_bits && phys_bits != 0 {
            self.warnings.push(format!(
                "Host physical bits ({host_phys_bits}) does not match phys-bits property ({phys_bits})"
            ));
        }
        if self.host_phys_bits {
            phys_bits = host_phys_bits;
            let limit = u32::from(self.host_phys_bits_limit);
            if limit != 0 && phys_bits > limit {
                phys_bits = limit;
            }
        }
        self.phys_bits = phys_bits;

        // kvm_set_guest_phys_bits()
        if self.guest_phys_bits == u32::MAX {
            let eax = host.supported_cpuid(0x8000_0008, 0, Reg::Eax);
            let bits = (eax >> 16) & 0xff;
            if bits != 0 {
                self.guest_phys_bits = bits.min(self.phys_bits);
                let limit = u32::from(self.host_phys_bits_limit);
                if self.host_phys_bits && limit != 0 && self.guest_phys_bits > limit {
                    self.guest_phys_bits = limit;
                }
            }
        }
    }

    /// Whether [`X86Cpu::realize`] has succeeded.
    pub fn is_realized(&self) -> bool {
        self.realized
    }

    /// Warnings printed so far, without QEMU's `warning:` prefix.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// The feature words, `env->features`.
    pub fn features(&self) -> &FeatureWordArray {
        &self.features
    }

    /// Bits removed by filtering or by the dependency checks,
    /// `cpu->filtered_features`.
    pub fn filtered_features(&self) -> &FeatureWordArray {
        &self.filtered_features
    }

    /// Whether the feature called `name` (or an alias of it) is enabled.
    pub fn has_feature(&self, name: &str) -> bool {
        find_feature(name).is_some_and(|(w, mask)| self.features[w] & mask == mask)
    }

    /// `CPUID[1].EAX`.
    pub fn cpuid_version(&self) -> u32 {
        self.version
    }

    /// Basic, extended and Centaur CPUID levels.
    pub fn levels(&self) -> (u32, u32, u32) {
        (self.cpuid_level, self.cpuid_xlevel, self.cpuid_xlevel2)
    }

    /// Physical address bits.
    pub fn phys_bits(&self) -> u32 {
        self.phys_bits
    }

    /// Guest physical address bits. `u32::MAX` before realize means unset.
    pub fn guest_phys_bits(&self) -> u32 {
        self.guest_phys_bits
    }

    /// Microcode revision reported in `MSR_IA32_UCODE_REV`.
    pub fn ucode_rev(&self) -> u64 {
        self.ucode_rev
    }

    /// TSC frequency in kHz set with `tsc-frequency`, or 0.
    pub fn tsc_khz(&self) -> i64 {
        self.tsc_khz
    }

    /// The APIC ID.
    pub fn apic_id(&self) -> u32 {
        self.apic_id
    }

    /// `MCG_CAP` as set up by `mce_init()`.
    pub fn mcg_cap(&self) -> u64 {
        self.mcg_cap
    }

    /// Copies the machine check setup of `mce_init()` into `state`.
    pub fn init_mce(&self, state: &mut X86CpuState) {
        if self.mcg_cap == 0 {
            return;
        }
        state.mcg_cap = self.mcg_cap;
        state.mcg_ctl = u64::MAX;
        for bank in 0..MCE_BANKS_DEF as usize {
            state.mce_banks[bank * 4] = u64::MAX;
        }
    }

    /// The reset inputs for this CPU.
    pub fn reset_config(&self, is_bsp: bool) -> ResetConfig {
        let irqchip = match &self.accel {
            Accel::Tcg => crate::state::IrqchipMode::default(),
            Accel::Kvm(h) => h.irqchip,
        };
        ResetConfig {
            is_bsp,
            kvm: self.accel.is_kvm(),
            irqchip,
            cpuid_version: self.version,
            has_monitor: self.features[FEAT_1_ECX] & CPUID_EXT_MONITOR != 0,
        }
    }

    /// A register state as left by realize: machine checks set up, then
    /// reset.
    pub fn new_state(&self, is_bsp: bool) -> X86CpuState {
        let mut s = X86CpuState::default();
        self.init_mce(&mut s);
        s.reset(&self.reset_config(is_bsp));
        s
    }

    /// `cpu_x86_cpuid()` with the register state right after reset
    /// (CR4 clear, not in long mode, XCR0 = x87 only).
    pub fn cpuid(&self, index: u32, count: u32) -> Regs {
        self.cpuid_regs(GuestRegs::RESET, index, count)
    }

    /// `cpu_x86_cpuid()` as seen by a vCPU in `state`, which matters for
    /// OSXSAVE, OSPKE, `CPUID[0xd,0].EBX` and SYSCALL on Intel under TCG.
    pub fn cpuid_for(&self, state: &X86CpuState, index: u32, count: u32) -> Regs {
        let regs = GuestRegs { cr4: state.cr4, hflags: state.hflags, xcr0: state.xcr0 };
        self.cpuid_regs(regs, index, count)
    }

    fn feat(&self, w: usize) -> u32 {
        self.features[w] as u32
    }

    fn cpuid_regs(&self, g: GuestRegs, index: u32, count: u32) -> Regs {
        let topo = &self.topo;
        let threads_per_pkg = topo.threads_per_pkg();
        let limit = if index >= 0xC000_0000 {
            self.cpuid_xlevel2
        } else if index >= 0x8000_0000 {
            self.cpuid_xlevel
        } else if index >= 0x4000_0000 {
            0x4000_0001
        } else {
            self.cpuid_level
        };
        let index = if index > limit { self.cpuid_level } else { index };

        match index {
            0 => [self.cpuid_level, self.vendor[0], self.vendor[2], self.vendor[1]],
            1 => {
                let mut ebx = (self.apic_id << 24) | (8 << 8);
                let mut ecx = self.feat(FEAT_1_ECX);
                if ecx & CPUID_EXT_XSAVE as u32 != 0 && g.cr4 & CR4_OSXSAVE_MASK != 0 {
                    ecx |= CPUID_EXT_OSXSAVE as u32;
                }
                if threads_per_pkg > 1 {
                    let num = if self.is_intel() || self.is_zhaoxin() {
                        1 << topo.pkg_offset()
                    } else {
                        threads_per_pkg
                    };
                    ebx |= num.min(255) << 16;
                }
                if self.pdcm_on_even_without_pmu && !self.enable_pmu {
                    ecx &= !(CPUID_EXT_PDCM as u32);
                }
                [self.version, ebx, ecx, self.feat(FEAT_1_EDX)]
            }
            2 => {
                let caches = if self.legacy_cpuid2_cache {
                    &LEGACY_INTEL_CPUID2_CACHES
                } else if self.legacy_vendor_cache {
                    &LEGACY_INTEL_CACHES
                } else {
                    &self.cache_info
                };
                if self.vendor_cpuid_only && self.is_amd() {
                    return [0; 4];
                }
                encode_cpuid2(
                    caches,
                    self.consistent_cache,
                    self.enable_l3_cache,
                    self.cpuid_min_level,
                )
            }
            4 => {
                let caches =
                    if self.legacy_vendor_cache { &LEGACY_INTEL_CACHES } else { &self.cache_info };
                if self.vendor_cpuid_only && self.is_amd() {
                    return [0; 4];
                }
                let l1_mask = if self.l1_cache_per_core { u32::MAX } else { !(0xfff << 14) };
                match count {
                    0 => {
                        let mut r = encode_cpuid4(&caches.l1d, topo);
                        r[0] &= l1_mask;
                        r
                    }
                    1 => {
                        let mut r = encode_cpuid4(&caches.l1i, topo);
                        r[0] &= l1_mask;
                        r
                    }
                    2 => encode_cpuid4(&caches.l2, topo),
                    3 if self.enable_l3_cache => encode_cpuid4(&caches.l3, topo),
                    _ => [0; 4],
                }
            }
            5 => self.mwait,
            6 => [self.feat(FEAT_6_EAX), 0, 0, 0],
            7 => match count {
                0 => {
                    let mut ecx = self.feat(FEAT_7_0_ECX);
                    if ecx & CPUID_7_0_ECX_PKU as u32 != 0 && g.cr4 & CR4_PKE_MASK != 0 {
                        ecx |= CPUID_7_0_ECX_OSPKE as u32;
                    }
                    [self.cpuid_level_func7, self.feat(FEAT_7_0_EBX), ecx, self.feat(FEAT_7_0_EDX)]
                }
                1 => [self.feat(FEAT_7_1_EAX), 0, self.feat(FEAT_7_1_ECX), self.feat(FEAT_7_1_EDX)],
                2 => [0, 0, 0, self.feat(FEAT_7_2_EDX)],
                _ => [0; 4],
            },
            0xA => {
                if self.enable_pmu {
                    self.supported_regs(0xA, count)
                } else {
                    [0; 4]
                }
            }
            0xB => {
                if !self.enable_cpuid_0xb {
                    return [0; 4];
                }
                let mut ecx = count & 0xff;
                let (eax, ebx) = match count {
                    0 => {
                        ecx |= CPUID_B_ECX_TOPO_LEVEL_SMT << 8;
                        (topo.core_offset(), topo.threads_per_core)
                    }
                    1 => {
                        ecx |= CPUID_B_ECX_TOPO_LEVEL_CORE << 8;
                        (topo.pkg_offset(), threads_per_pkg)
                    }
                    _ => {
                        ecx |= CPUID_B_ECX_TOPO_LEVEL_INVALID << 8;
                        (0, 0)
                    }
                };
                [eax, ebx & 0xffff, ecx, self.apic_id]
            }
            0xD => self.cpuid_xsave(g, count),
            0x14 => self.cpuid_intel_pt(count),
            0x1C => {
                if self.features[FEAT_7_0_EDX] & CPUID_7_0_EDX_ARCH_LBR == 0 {
                    return [0; 4];
                }
                let mut r = self.supported_regs(0x1C, 0);
                r[3] = 0;
                r
            }
            0x1D => {
                if self.features[FEAT_7_0_EDX] & CPUID_7_0_EDX_AMX_TILE == 0 {
                    return [0; 4];
                }
                match count {
                    // INTEL_AMX_TILE_MAX_SUBLEAF
                    0 => [1, 0, 0, 0],
                    // Tile palette 1: total bytes, bytes per tile, bytes per
                    // row, max names, max rows.
                    1 => [0x2000 | (0x400 << 16), 0x40 | (8 << 16), 0x10, 0],
                    _ => [0; 4],
                }
            }
            0x1E => {
                if self.features[FEAT_7_0_EDX] & CPUID_7_0_EDX_AMX_TILE == 0 {
                    return [0; 4];
                }
                match count {
                    // INTEL_AMX_TMUL_MAX_K | INTEL_AMX_TMUL_MAX_N << 8
                    0 => [self.supported_regs(0x1E, 0)[0], 0x10 | (0x40 << 8), 0, 0],
                    1 => [self.feat(FEAT_1E_1_EAX), 0, 0, 0],
                    _ => [0; 4],
                }
            }
            0x1F => {
                if !self.has_cpuid_0x1f() {
                    return [0; 4];
                }
                self.encode_topo_cpuid1f(count)
            }
            0x29 => {
                if self.features[FEAT_7_1_EDX] & CPUID_7_1_EDX_APXF != 0 && count == 0 {
                    [0, self.feat(FEAT_29_0_EBX), 0, 0]
                } else {
                    [0; 4]
                }
            }
            0x4000_0000 => {
                if !self.accel.is_kvm() && self.expose_tcg {
                    let sig = pack_vendor("TCGTCGTCGTCG");
                    [0x4000_0001, sig[0], sig[1], sig[2]]
                } else {
                    [0; 4]
                }
            }
            0x8000_0000 => {
                if self.vendor_cpuid_only_v2 && (self.is_intel() || self.is_zhaoxin()) {
                    [self.cpuid_xlevel, 0, 0, 0]
                } else {
                    [self.cpuid_xlevel, self.vendor[0], self.vendor[2], self.vendor[1]]
                }
            }
            0x8000_0001 => {
                let mut edx = self.feat(FEAT_8000_0001_EDX);
                if !self.accel.is_kvm() && self.is_intel() && g.hflags & HF_LMA_MASK == 0 {
                    edx &= !(CPUID_EXT2_SYSCALL as u32);
                }
                [self.version, 0, self.feat(FEAT_8000_0001_ECX), edx]
            }
            0x8000_0002..=0x8000_0004 => {
                let base = (index - 0x8000_0002) as usize * 4;
                [
                    self.model_id[base],
                    self.model_id[base + 1],
                    self.model_id[base + 2],
                    self.model_id[base + 3],
                ]
            }
            0x8000_0005 => {
                let caches =
                    if self.legacy_vendor_cache { &LEGACY_AMD_CACHES } else { &self.cache_info };
                if self.vendor_cpuid_only_v2 && self.is_intel() {
                    return [0; 4];
                }
                // L1 TLBs: 255 entries, fully associative, for 2M and 4K.
                [
                    0x01ff_01ff,
                    0x01ff_01ff,
                    encode_cpuid80000005(&caches.l1d),
                    encode_cpuid80000005(&caches.l1i),
                ]
            }
            0x8000_0006 => {
                let caches =
                    if self.legacy_vendor_cache { &LEGACY_AMD_CACHES } else { &self.cache_info };
                if self.vendor_cpuid_only_v2 && (self.is_intel() || self.is_zhaoxin()) {
                    let (ecx, edx) = encode_cpuid80000006(&caches.l2, None);
                    return [0, 0, ecx, edx];
                }
                let l3 = if self.enable_l3_cache { Some(&caches.l3) } else { None };
                let (ecx, edx) = encode_cpuid80000006(&caches.l2, l3);
                // L2 TLB: 2M disabled, 4K with 512 entries, 4-way
                // (X86_ENC_ASSOC(4) = 4).
                [0, (4 << 28) | (512 << 16) | (4 << 12) | 512, ecx, edx]
            }
            0x8000_0007 => {
                let ebx = if self.vendor_cpuid_only_v2 && self.is_intel() {
                    0
                } else {
                    self.feat(FEAT_8000_0007_EBX)
                };
                [0, ebx, 0, self.feat(FEAT_8000_0007_EDX)]
            }
            0x8000_0008 => {
                let mut eax = self.phys_bits;
                if self.features[FEAT_8000_0001_EDX] & CPUID_EXT2_LM != 0 {
                    let vaddr =
                        if self.features[FEAT_7_0_ECX] & CPUID_7_0_ECX_LA57 != 0 { 57 } else { 48 };
                    eax |= vaddr << 8;
                    eax |= self.guest_phys_bits << 16;
                }
                let ebx = self.feat(FEAT_8000_0008_EBX);
                if self.vendor_cpuid_only_v2 && (self.is_intel() || self.is_zhaoxin()) {
                    return [eax, ebx, 0, 0];
                }
                let ecx = if threads_per_pkg > 1 {
                    (topo.pkg_offset() << 12) | (threads_per_pkg - 1)
                } else {
                    0
                };
                [eax, ebx, ecx, 0]
            }
            0x8000_000A => {
                if self.features[FEAT_8000_0001_ECX] & CPUID_EXT3_SVM != 0 {
                    [1, 0x10, 0, self.feat(FEAT_SVM)]
                } else {
                    [0; 4]
                }
            }
            0x8000_001D => {
                let c = &self.cache_info;
                let mut r = match count {
                    0 => encode_cpuid8000001d(&c.l1d, topo),
                    1 => encode_cpuid8000001d(&c.l1i, topo),
                    2 => encode_cpuid8000001d(&c.l2, topo),
                    3 => encode_cpuid8000001d(&c.l3, topo),
                    _ => [0; 4],
                };
                if self.amd_topoext_features_only {
                    r[3] &= CACHE_TOPOEXT_EDX_MASK;
                }
                r
            }
            0x8000_001E => {
                let ids = topo.ids_from_apicid(self.apic_id);
                if ids.core_id > 255 {
                    return [0; 4];
                }
                let ebx = ((topo.threads_per_core - 1) << 8) | (ids.core_id & 0xff);
                let ecx = if self.legacy_multi_node {
                    ((topo.dies_per_pkg - 1) << 8)
                        | (self.apic_id.checked_shr(topo.die_offset()).unwrap_or(0) & 0xff)
                } else {
                    self.apic_id.checked_shr(topo.pkg_offset()).unwrap_or(0) & 0xff
                };
                [self.apic_id, ebx, ecx, 0]
            }
            0x8000_0021 => [
                self.feat(FEAT_8000_0021_EAX),
                self.feat(FEAT_8000_0021_EBX),
                self.feat(FEAT_8000_0021_ECX),
                0,
            ],
            0x8000_0022 => {
                if self.accel.is_kvm()
                    && self.enable_pmu
                    && self.features[FEAT_8000_0022_EAX] & CPUID_8000_0022_EAX_PERFMON_V2 != 0
                {
                    let ebx = self.supported_regs(index, count)[1] & 0xf;
                    [CPUID_8000_0022_EAX_PERFMON_V2 as u32, ebx, 0, 0]
                } else {
                    [0; 4]
                }
            }
            0xC000_0000 => [self.cpuid_xlevel2, 0, 0, 0],
            0xC000_0001 => [self.version, 0, 0, self.feat(FEAT_C000_0001_EDX)],
            // 9, 0x12 (SGX is never enabled here), 0x24 (AVX10 version is
            // not ported), 0x40000001, 0x8000001F (no SEV), 0xC0000002 to
            // 0xC0000004 and everything else.
            _ => [0; 4],
        }
    }

    fn cpuid_xsave(&self, g: GuestRegs, count: u32) -> Regs {
        if self.features[FEAT_1_ECX] & CPUID_EXT_XSAVE == 0 {
            return [0; 4];
        }
        let areas = &self.ext_save_areas;
        match count {
            0 => {
                let ecx = xsave_area_size(areas, self.xcr0_components(), false);
                let ebx =
                    if self.accel.is_kvm() { ecx } else { xsave_area_size(areas, g.xcr0, false) };
                [self.feat(FEAT_XSAVE_XCR0_LO), ebx, ecx, self.feat(FEAT_XSAVE_XCR0_HI)]
            }
            1 => {
                let xstate = self.xcr0_components() | self.xss_components();
                [
                    self.feat(FEAT_XSAVE),
                    xsave_area_size(areas, xstate, true),
                    self.feat(FEAT_XSAVE_XSS_LO),
                    self.feat(FEAT_XSAVE_XSS_HI),
                ]
            }
            c if (c as usize) < XSAVE_STATE_AREA_COUNT => {
                let esa = &areas[c as usize];
                [esa.size, esa.offset, esa.ecx, 0]
            }
            _ => [0; 4],
        }
    }

    /// Leaf 0x14. The KVM capability check that QEMU does while filtering
    /// is not ported, so this only reports what the feature bits say.
    fn cpuid_intel_pt(&self, count: u32) -> Regs {
        if self.features[FEAT_7_0_EBX] & CPUID_7_0_EBX_INTEL_PT == 0 || !self.accel.is_kvm() {
            return [0; 4];
        }
        match count {
            0 => {
                // INTEL_PT_MAX_SUBLEAF, INTEL_PT_MINIMAL_EBX, INTEL_PT_MINIMAL_ECX
                let mut ecx = 0x7;
                if self.features[FEAT_14_0_ECX] & CPUID_14_0_ECX_LIP != 0 {
                    ecx |= CPUID_14_0_ECX_LIP as u32;
                }
                [1, 0xf, ecx, 0]
            }
            // MTC bitmap | address ranges, PSB bitmap | cycle bitmap.
            1 => [(0x0249 << 16) | 2, (0x003f << 16) | 0x1fff, 0, 0],
            _ => [0; 4],
        }
    }

    /// `encode_topo_cpuid1f()`.
    fn encode_topo_cpuid1f(&self, count: u32) -> Regs {
        let topo = &self.topo;
        let avail = topo.available_levels();
        // Levels below the package, in order; subleaf N describes the Nth.
        let below: Vec<TopoLevel> =
            avail.iter().copied().filter(|&l| l != TopoLevel::Socket).collect();
        let (level_type, eax, ebx) = match below.get(count as usize) {
            None => (CPUID_B_ECX_TOPO_LEVEL_INVALID, 0, 0),
            Some(&level) => {
                let next = avail.iter().copied().find(|&l| l > level).unwrap_or(TopoLevel::Socket);
                let ty = match level {
                    TopoLevel::Thread => CPUID_B_ECX_TOPO_LEVEL_SMT,
                    TopoLevel::Core => CPUID_B_ECX_TOPO_LEVEL_CORE,
                    TopoLevel::Module => CPUID_1F_ECX_TOPO_LEVEL_MODULE,
                    TopoLevel::Die => CPUID_1F_ECX_TOPO_LEVEL_DIE,
                    TopoLevel::Socket => CPUID_B_ECX_TOPO_LEVEL_INVALID,
                };
                (ty, topo.offset_of(next), topo.threads_at(next) & 0xffff)
            }
        };
        [eax, ebx, (count & 0xff) | (level_type << 8), self.apic_id]
    }

    /// `x86_cpu_family()` of this CPU.
    pub fn family(&self) -> u32 {
        cpu_family(self.version)
    }

    /// The feature word array as `(description, value)` pairs, handy for
    /// debugging and `query-cpu-model-expansion` style output.
    pub fn feature_word_report(&self) -> Vec<(String, u64)> {
        FEATURE_WORD_INFO
            .iter()
            .zip(self.features.iter())
            .map(|(wi, &v)| (wi.description(), v))
            .collect()
    }

    /// Returns a register of a CPUID query, for tests and callers that
    /// only need one value.
    pub fn cpuid_reg(&self, index: u32, count: u32, reg: Reg) -> u32 {
        reg_of(self.cpuid(index, count), reg)
    }
}

#[cfg(test)]
mod tests;
