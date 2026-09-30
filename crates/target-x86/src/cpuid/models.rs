// SPDX-License-Identifier: GPL-2.0-or-later

//! Built-in CPU models: a subset of `builtin_x86_defs[]` from
//! `target/i386/cpu.c`.
//!
//! The feature masks were produced by compiling QEMU's own initialisers, so
//! they are the exact values QEMU 11.1 uses. `max` and `host` have no entry
//! here; they are handled in [`super::X86Cpu::new`].

use super::words::{
    FEAT_1_ECX, FEAT_1_EDX, FEAT_6_EAX, FEAT_7_0_EBX, FEAT_8000_0001_ECX, FEAT_8000_0001_EDX,
    FEAT_SVM, FEAT_VMX_BASIC, FEAT_VMX_ENTRY_CTLS, FEAT_VMX_EPT_VPID_CAPS, FEAT_VMX_EXIT_CTLS,
    FEAT_VMX_MISC, FEAT_VMX_PINBASED_CTLS, FEAT_VMX_PROCBASED_CTLS, FEAT_VMX_SECONDARY_CTLS,
    FEAT_VMX_VMFUNC, FEAT_XSAVE,
};

/// Which cache description a model or model version uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelCache {
    /// `epyc_cache_info`.
    Epyc,
    /// `epyc_v4_cache_info`.
    EpycV4,
    /// `epyc_v5_cache_info`.
    EpycV5,
}

/// One entry of a model's `versions` list, as `X86CPUVersionDefinition`.
#[derive(Debug)]
pub struct X86CpuVersion {
    /// Version number, starting at 1.
    pub version: u32,
    /// Extra model name that selects this version.
    pub alias: Option<&'static str>,
    /// Properties applied on top of the previous version.
    pub props: &'static [(&'static str, &'static str)],
    /// Cache description introduced by this version.
    pub cache: Option<ModelCache>,
}

/// A built-in CPU model, as `X86CPUDefinition`.
#[derive(Debug)]
pub struct X86CpuDefinition {
    /// Base model name.
    pub name: &'static str,
    /// Minimum basic CPUID level.
    pub level: u32,
    /// Minimum extended CPUID level.
    pub xlevel: u32,
    /// Vendor string, 12 characters.
    pub vendor: &'static str,
    /// Family.
    pub family: u32,
    /// Model.
    pub model: u32,
    /// Stepping.
    pub stepping: u32,
    /// Nonzero feature words.
    pub features: &'static [(usize, u64)],
    /// Brand string.
    pub model_id: &'static str,
    /// Cache description for version 1.
    pub cache: Option<ModelCache>,
    /// Explicit versions. Empty means the model only has version 1.
    pub versions: &'static [X86CpuVersion],
}

/// The ported models.
pub static BUILTIN_MODELS: &[X86CpuDefinition] = &[
    X86CpuDefinition {
        name: "qemu64",
        level: 0xd,
        xlevel: 0x8000000a,
        vendor: "AuthenticAMD",
        family: 15,
        model: 107,
        stepping: 1,
        features: &[
            (FEAT_1_EDX, 0x078b_fbfd),
            (FEAT_1_ECX, 0x2001),
            (FEAT_8000_0001_EDX, 0x2010_0800),
            (FEAT_8000_0001_ECX, 0x5),
        ],
        model_id: "QEMU Virtual CPU version 2.5+",
        cache: None,
        versions: &[],
    },
    X86CpuDefinition {
        name: "kvm64",
        level: 0xd,
        xlevel: 0x80000008,
        vendor: "GenuineIntel",
        family: 15,
        model: 6,
        stepping: 1,
        features: &[
            (FEAT_1_EDX, 0x078b_fbff),
            (FEAT_1_ECX, 0x2001),
            (FEAT_8000_0001_EDX, 0x2010_0800),
            (FEAT_VMX_PROCBASED_CTLS, 0x63b8_1e8c),
            (FEAT_VMX_PINBASED_CTLS, 0x9),
            (FEAT_VMX_EXIT_CTLS, 0x8000),
            (FEAT_VMX_ENTRY_CTLS, 0x200),
            (FEAT_VMX_MISC, 0x40),
        ],
        model_id: "Common KVM processor",
        cache: None,
        versions: &[],
    },
    X86CpuDefinition {
        name: "Skylake-Client",
        level: 0xd,
        xlevel: 0x80000008,
        vendor: "GenuineIntel",
        family: 6,
        model: 94,
        stepping: 3,
        features: &[
            (FEAT_1_EDX, 0x078b_fbff),
            (FEAT_1_ECX, 0x77fa_3203),
            (FEAT_7_0_EBX, 0x001c_0fb9),
            (FEAT_8000_0001_EDX, 0x2810_0800),
            (FEAT_8000_0001_ECX, 0x121),
            (FEAT_XSAVE, 0x7),
            (FEAT_6_EAX, 0x4),
            (FEAT_VMX_PROCBASED_CTLS, 0xfbf9_9e8c),
            (FEAT_VMX_SECONDARY_CTLS, 0x0003_78ef),
            (FEAT_VMX_PINBASED_CTLS, 0x69),
            (FEAT_VMX_EXIT_CTLS, 0x007c_9004),
            (FEAT_VMX_ENTRY_CTLS, 0xe204),
            (FEAT_VMX_MISC, 0x2000_0060),
            (FEAT_VMX_EPT_VPID_CAPS, 0x0000_0f01_0633_4041),
            (FEAT_VMX_BASIC, 0x00c0_0000_0000_0000),
            (FEAT_VMX_VMFUNC, 0x1),
        ],
        model_id: "Intel Core Processor (Skylake)",
        cache: None,
        versions: &[
            X86CpuVersion { version: 1, alias: None, props: &[], cache: None },
            X86CpuVersion {
                version: 2,
                alias: Some("Skylake-Client-IBRS"),
                props: &[("spec-ctrl", "on"), ("model-id", "Intel Core Processor (Skylake, IBRS)")],
                cache: None,
            },
            X86CpuVersion {
                version: 3,
                alias: Some("Skylake-Client-noTSX-IBRS"),
                props: &[
                    ("hle", "off"),
                    ("rtm", "off"),
                    ("model-id", "Intel Core Processor (Skylake, IBRS, no TSX)"),
                ],
                cache: None,
            },
            X86CpuVersion {
                version: 4,
                alias: None,
                props: &[("xsaves", "on"), ("vmx-xsaves", "on")],
                cache: None,
            },
        ],
    },
    X86CpuDefinition {
        name: "EPYC",
        level: 0xd,
        xlevel: 0x8000001e,
        vendor: "AuthenticAMD",
        family: 23,
        model: 1,
        stepping: 2,
        features: &[
            (FEAT_1_EDX, 0x078b_fbff),
            (FEAT_1_ECX, 0x76d8_320b),
            (FEAT_7_0_EBX, 0x209c_01a9),
            (FEAT_8000_0001_EDX, 0x2e50_0800),
            (FEAT_8000_0001_ECX, 0x0040_03f5),
            (FEAT_SVM, 0x9),
            (FEAT_XSAVE, 0x7),
            (FEAT_6_EAX, 0x4),
        ],
        model_id: "AMD EPYC Processor",
        cache: Some(ModelCache::Epyc),
        versions: &[
            X86CpuVersion { version: 1, alias: None, props: &[], cache: None },
            X86CpuVersion {
                version: 2,
                alias: Some("EPYC-IBPB"),
                props: &[("ibpb", "on"), ("model-id", "AMD EPYC Processor (with IBPB)")],
                cache: None,
            },
            X86CpuVersion {
                version: 3,
                alias: None,
                props: &[
                    ("ibpb", "on"),
                    ("perfctr-core", "on"),
                    ("clzero", "on"),
                    ("xsaveerptr", "on"),
                    ("xsaves", "on"),
                    ("model-id", "AMD EPYC Processor"),
                ],
                cache: None,
            },
            X86CpuVersion {
                version: 4,
                alias: None,
                props: &[("model-id", "AMD EPYC-v4 Processor")],
                cache: Some(ModelCache::EpycV4),
            },
            X86CpuVersion {
                version: 5,
                alias: None,
                props: &[
                    ("overflow-recov", "on"),
                    ("succor", "on"),
                    ("lbrv", "on"),
                    ("tsc-scale", "on"),
                    ("vmcb-clean", "on"),
                    ("flushbyasid", "on"),
                    ("pause-filter", "on"),
                    ("pfthreshold", "on"),
                    ("v-vmsave-vmload", "on"),
                    ("vgif", "on"),
                    ("model-id", "AMD EPYC-v5 Processor"),
                ],
                cache: Some(ModelCache::EpycV5),
            },
        ],
    },
];

/// A model name resolved to a definition and a version.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedModel {
    /// The definition.
    pub def: &'static X86CpuDefinition,
    /// The version to load.
    pub version: u32,
}

impl X86CpuDefinition {
    /// Highest version number, as `x86_cpu_model_last_version()`.
    pub fn last_version(&self) -> u32 {
        self.versions.last().map_or(1, |v| v.version)
    }

    /// Cache description for `version`: the one set by the newest version
    /// at or below it, falling back to the base definition, as
    /// `x86_cpu_get_versioned_cache_info()`.
    pub fn cache_for_version(&self, version: u32) -> Option<ModelCache> {
        let mut cache = self.cache;
        for v in self.versions.iter().take_while(|v| v.version <= version) {
            if v.cache.is_some() {
                cache = v.cache;
            }
        }
        cache
    }
}

/// Looks up a model by the names QEMU registers: the base name (which
/// means version 1, `default_cpu_version` on the PC machines), `NAME-vN`,
/// and version aliases such as `Skylake-Client-IBRS`.
pub fn find_model(name: &str) -> Option<ResolvedModel> {
    for def in BUILTIN_MODELS {
        if def.name == name {
            return Some(ResolvedModel { def, version: 1 });
        }
        for v in def.versions {
            if v.alias == Some(name) {
                return Some(ResolvedModel { def, version: v.version });
            }
        }
        let suffix = name.strip_prefix(def.name).and_then(|rest| rest.strip_prefix("-v"));
        if let Some(n) = suffix.and_then(|s| s.parse::<u32>().ok()) {
            if n >= 1 && n <= def.last_version() && suffix == Some(n.to_string().as_str()) {
                return Some(ResolvedModel { def, version: n });
            }
        }
    }
    None
}
