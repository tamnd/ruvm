// SPDX-License-Identifier: GPL-2.0-or-later

//! The CPU models of qemu-system-riscv64 and their configuration.
//!
//! This is `RISCVCPUConfig` (target/riscv/cpu_cfg_fields.h.inc), `isa_edata_arr[]`, the
//! implied extension rules, the profiles and the CPU type definitions of target/riscv/cpu.c,
//! and the property setters and `riscv_tcg_cpu_finalize_features()` of
//! target/riscv/tcg/tcg-cpu.c.
//!
//! [`CpuBuilder`] runs the same steps as QEMU. [`CpuBuilder::new`] does what
//! `riscv_cpu_init()` and `riscv_tcg_cpu_instance_init()` do for the model.
//! [`CpuBuilder::set`] sets one `-cpu` property, in the order they are given. Then
//! [`CpuBuilder::finalize`] does `riscv_cpu_finalize_features()`: it applies the implied
//! rules, checks the result and gives the [`RiscvCfg`] the hart runs with.
//!
//! The extensions this port does not implement yet are marked in [`ISA_EXTS`]. `max` leaves
//! them off, and a property that would turn one on is refused, so the guest never sees an
//! extension in the device tree that the translator does not have.

use crate::cpu::{NUM_TRIGGERS, RVA, RVC, RVD, RVF, RVH, RVI, RVM, RVS, RVU, RVV, VLENB, rvx};

/// `RVE`.
pub const RVE: u64 = rvx(b'E');
/// `RVG`.
pub const RVG: u64 = rvx(b'G');
/// `RVB`.
pub const RVB: u64 = rvx(b'B');
/// `RVX`, set when a vendor extension is on.
pub const RVX: u64 = rvx(b'X');

/// `misa_bits[]`: the misa extensions in the order QEMU walks them.
const MISA_BITS: [u64; 13] = [RVI, RVE, RVM, RVA, RVF, RVD, RVV, RVC, RVS, RVU, RVH, RVG, RVB];

/// `riscv_single_letter_exts[]`: the order of the letters in the ISA string.
const SINGLE_LETTER_EXTS: &[u8] = b"IEMAFDQCBPVH";

/// The largest VLEN, `RV_VLEN_MAX`.
const RV_VLEN_MAX: u32 = 1024;

/// `VM_1_10_MBARE`: no translation.
pub const VM_MBARE: i8 = 0;
/// `VM_1_10_SV39`.
pub const VM_SV39: i8 = 8;
/// `VM_1_10_SV48`.
pub const VM_SV48: i8 = 9;
/// `VM_1_10_SV57`.
pub const VM_SV57: i8 = 10;
/// `VM_1_10_SV64`.
pub const VM_SV64: i8 = 11;

/// `valid_vm_1_10_64[]`: the satp modes an RV64 hart can have.
pub const fn satp_mode_valid(mode: i8) -> bool {
    matches!(mode, VM_MBARE | VM_SV39 | VM_SV48 | VM_SV57)
}

/// `satp_mode_str()` for RV64.
pub fn satp_mode_str(mode: i8) -> &'static str {
    match mode {
        VM_SV64 => "sv64",
        VM_SV57 => "sv57",
        VM_SV48 => "sv48",
        VM_SV39 => "sv39",
        _ => "none",
    }
}

/// `satp_mode_from_str()`, for the satp mode properties.
fn satp_mode_from_str(name: &str) -> Option<i8> {
    match name {
        "svbare" => Some(VM_MBARE),
        "sv39" => Some(VM_SV39),
        "sv48" => Some(VM_SV48),
        "sv57" => Some(VM_SV57),
        "sv64" => Some(VM_SV64),
        _ => None,
    }
}

/// A privileged architecture version, `PRIV_VERSION_*`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrivVer {
    /// 1.10.0, the oldest QEMU has.
    #[default]
    V1_10,
    /// 1.11.0.
    V1_11,
    /// 1.12.0.
    V1_12,
    /// 1.13.0, `PRIV_VERSION_LATEST`.
    V1_13,
}

impl PrivVer {
    /// `PRIV_VERSION_LATEST`.
    pub const LATEST: PrivVer = PrivVer::V1_13;

    /// `priv_spec_from_str()`.
    pub fn parse(s: &str) -> Option<PrivVer> {
        match s {
            "v1.10.0" => Some(PrivVer::V1_10),
            "v1.11.0" => Some(PrivVer::V1_11),
            "v1.12.0" => Some(PrivVer::V1_12),
            "v1.13.0" => Some(PrivVer::V1_13),
            _ => None,
        }
    }

    /// `priv_spec_to_str()`.
    pub fn as_str(self) -> &'static str {
        match self {
            PrivVer::V1_10 => "v1.10.0",
            PrivVer::V1_11 => "v1.11.0",
            PrivVer::V1_12 => "v1.12.0",
            PrivVer::V1_13 => "v1.13.0",
        }
    }

    fn rank(self) -> i8 {
        self as i8
    }

    fn from_rank(r: i8) -> PrivVer {
        match r {
            i8::MIN..=0 => PrivVer::V1_10,
            1 => PrivVer::V1_11,
            2 => PrivVer::V1_12,
            _ => PrivVer::V1_13,
        }
    }
}

macro_rules! riscv_cfg {
    ($($field:ident),* $(,)?) => {
        /// The configuration of a hart, QEMU's `RISCVCPUConfig` with the misa extensions and
        /// the privileged architecture version next to it.
        ///
        /// The [`Default`] is what QEMU's default CPU, `rv64`, ends up with.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[allow(clippy::struct_excessive_bools)]
        pub struct RiscvCfg {
            $(
                #[doc = concat!("`", stringify!($field), "`.")]
                pub $field: bool,
            )*
            /// The misa extension letters, `env->misa_ext`.
            pub misa_ext: u64,
            /// The privileged architecture version, `env->priv_ver`.
            pub priv_ver: PrivVer,
            /// `mvendorid`.
            pub mvendorid: u32,
            /// `marchid`.
            pub marchid: u64,
            /// `mimpid`.
            pub mimpid: u64,
            /// `pmu-mask`: the programmable counters, as bits 3 to 31.
            pub pmu_mask: u32,
            /// `vlenb`. Only 16 is supported.
            pub vlenb: u32,
            /// `elen`.
            pub elen: u32,
            /// `cbom_blocksize`.
            pub cbom_blocksize: u16,
            /// `cbop_blocksize`.
            pub cbop_blocksize: u16,
            /// `cboz_blocksize`.
            pub cboz_blocksize: u16,
            /// `num-pmp-regions`.
            pub pmp_regions: u8,
            /// The largest satp mode, `max_satp_mode`, or -1 when the CPU leaves it open.
            pub max_satp_mode: i8,
        }

        impl RiscvCfg {
            /// A configuration with everything off and every number zero.
            const fn zeroed() -> RiscvCfg {
                RiscvCfg {
                    $($field: false,)*
                    misa_ext: 0,
                    priv_ver: PrivVer::V1_10,
                    mvendorid: 0,
                    marchid: 0,
                    mimpid: 0,
                    pmu_mask: 0,
                    vlenb: 0,
                    elen: 0,
                    cbom_blocksize: 0,
                    cbop_blocksize: 0,
                    cboz_blocksize: 0,
                    pmp_regions: 0,
                    max_satp_mode: -1,
                }
            }
        }
    };
}

riscv_cfg! {
    ext_zba, ext_zbb, ext_zbc, ext_zbkb, ext_zbkc, ext_zbkx, ext_zbs,
    ext_zca, ext_zcb, ext_zcd, ext_zce, ext_zcf, ext_zcmp, ext_zcmt, ext_zclsd,
    ext_zk, ext_zkn, ext_zknd, ext_zkne, ext_zknh, ext_zkr, ext_zks, ext_zksed, ext_zksh, ext_zkt,
    ext_zifencei, ext_zicntr, ext_zicsr, ext_zicbom, ext_zicbop, ext_zicboz, ext_zicfilp,
    ext_zicfiss, ext_zicond, ext_zihintntl, ext_zihintpause, ext_zihpm, ext_zilsd, ext_zimop,
    ext_zcmop, ext_ztso, ext_smstateen, ext_sstc, ext_smcdeleg, ext_ssccfg, ext_smcntrpmf,
    ext_smcsrind, ext_sscsrind, ext_ssdbltrp, ext_smdbltrp, ext_svadu, ext_svinval, ext_svnapot,
    ext_svpbmt, ext_smpmpmt, ext_svrsw60t59b, ext_svvptc, ext_svukte, ext_zdinx, ext_zaamo,
    ext_zacas, ext_zama16b, ext_zabha, ext_zalasr, ext_zalrsc, ext_zawrs, ext_zfa, ext_zfbfmin,
    ext_zfh, ext_zfhmin, ext_zfinx, ext_zhinx, ext_zhinxmin,
    ext_zve32f, ext_zve32x, ext_zve64f, ext_zve64d, ext_zve64x,
    ext_zvbb, ext_zvbc, ext_zvkb, ext_zvkg, ext_zvkned, ext_zvknha, ext_zvknhb, ext_zvksed,
    ext_zvksh, ext_zvkt, ext_zvkn, ext_zvknc, ext_zvkng, ext_zvks, ext_zvksc, ext_zvksg,
    ext_zmmul, ext_zvfbfa, ext_zvfbfmin, ext_zvfbfwma, ext_zvfh, ext_zvfhmin,
    ext_smaia, ext_ssaia, ext_smctr, ext_ssctr, ext_sscofpmf, ext_smepmp, ext_smrnmi,
    ext_ssnpm, ext_smnpm, ext_smmpm, ext_sspm, ext_supm,
    rvv_ta_all_1s, rvv_ma_all_1s, rvv_vl_half_avl, rvv_vsetvl_x0_vill,
    ext_svade, ext_zic64b, ext_ssstateen, ext_sha,
    has_priv_1_13, has_priv_1_12, has_priv_1_11,
    ext_ziccrse,
    ext_xlrbr,
    mmu, pmp, debug, short_isa_string,
}

/// Whether an [`IsaExt`] has a property, and under which name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtProp {
    /// A property with the extension's name, `ISA_EXT_DATA_ENTRY`.
    User,
    /// A property named `x-` and the name, `ISA_EXPERIMENTAL_EXT_DATA_ENTRY`.
    Experimental,
    /// No property: a named feature the configuration decides, `ISA_INTERNAL_EXT_DATA_ENTRY`.
    Internal,
}

/// One entry of `isa_edata_arr[]`.
#[derive(Clone, Copy)]
pub struct IsaExt {
    /// The name in the ISA string.
    pub name: &'static str,
    /// The property.
    pub prop: ExtProp,
    /// The oldest privileged version that has it.
    pub min: PrivVer,
    get: fn(&RiscvCfg) -> bool,
    set: fn(&mut RiscvCfg, bool),
    /// Whether this port implements it. `max` leaves the others off.
    pub ruvm: bool,
}

impl std::fmt::Debug for IsaExt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IsaExt")
            .field("name", &self.name)
            .field("prop", &self.prop)
            .field("min", &self.min)
            .field("ruvm", &self.ruvm)
            .finish()
    }
}

impl IsaExt {
    /// Whether `cfg` has it on, `isa_ext_is_enabled()`.
    pub fn enabled(&self, cfg: &RiscvCfg) -> bool {
        (self.get)(cfg)
    }

    /// The name of its property, if it has one.
    pub fn prop_name(&self) -> Option<String> {
        match self.prop {
            ExtProp::User => Some(self.name.to_string()),
            ExtProp::Experimental => Some(format!("x-{}", self.name)),
            ExtProp::Internal => None,
        }
    }

    /// Whether this is a vendor extension, or a property `max` does not turn on.
    fn is_x(&self) -> bool {
        self.name.starts_with('x') || self.prop == ExtProp::Experimental
    }
}

macro_rules! isa_exts {
    ($($kind:ident $name:ident $min:ident $field:ident $ruvm:literal;)*) => {
        /// `isa_edata_arr[]`: every multi-letter extension, in the order of the ISA string.
        pub static ISA_EXTS: &[IsaExt] = &[
            $(IsaExt {
                name: stringify!($name),
                prop: ExtProp::$kind,
                min: PrivVer::$min,
                get: |c| c.$field,
                set: |c, v| c.$field = v,
                ruvm: $ruvm,
            },)*
        ];
    };
}

isa_exts! {
    Internal zic64b V1_12 ext_zic64b true;
    User zicbom V1_12 ext_zicbom true;
    User zicbop V1_12 ext_zicbop true;
    User zicboz V1_12 ext_zicboz true;
    Internal ziccamoa V1_11 has_priv_1_11 true;
    Internal ziccif V1_11 has_priv_1_11 true;
    Internal zicclsm V1_11 has_priv_1_11 true;
    User ziccrse V1_11 ext_ziccrse true;
    User zicfilp V1_12 ext_zicfilp false;
    User zicfiss V1_13 ext_zicfiss false;
    User zicond V1_12 ext_zicond true;
    User zicntr V1_12 ext_zicntr true;
    User zicsr V1_10 ext_zicsr true;
    User zifencei V1_10 ext_zifencei true;
    User zihintntl V1_10 ext_zihintntl true;
    User zihintpause V1_10 ext_zihintpause true;
    User zihpm V1_12 ext_zihpm true;
    User zilsd V1_12 ext_zilsd false;
    User zimop V1_13 ext_zimop true;
    User zmmul V1_12 ext_zmmul true;
    Internal za64rs V1_12 has_priv_1_12 true;
    User zaamo V1_12 ext_zaamo true;
    User zabha V1_13 ext_zabha true;
    User zacas V1_12 ext_zacas true;
    User zalasr V1_12 ext_zalasr true;
    User zalrsc V1_12 ext_zalrsc true;
    User zama16b V1_13 ext_zama16b true;
    User zawrs V1_12 ext_zawrs true;
    User zfa V1_12 ext_zfa true;
    User zfbfmin V1_12 ext_zfbfmin true;
    User zfh V1_11 ext_zfh true;
    User zfhmin V1_11 ext_zfhmin true;
    User zfinx V1_12 ext_zfinx false;
    User zdinx V1_12 ext_zdinx false;
    User zca V1_12 ext_zca true;
    User zcb V1_12 ext_zcb true;
    User zcf V1_12 ext_zcf false;
    User zcd V1_12 ext_zcd true;
    User zce V1_12 ext_zce false;
    User zcmop V1_13 ext_zcmop true;
    User zcmp V1_12 ext_zcmp false;
    User zcmt V1_12 ext_zcmt false;
    User zclsd V1_12 ext_zclsd false;
    User zba V1_12 ext_zba true;
    User zbb V1_12 ext_zbb true;
    User zbc V1_12 ext_zbc true;
    User zbkb V1_12 ext_zbkb true;
    User zbkc V1_12 ext_zbkc true;
    User zbkx V1_12 ext_zbkx true;
    User zbs V1_12 ext_zbs true;
    User zk V1_12 ext_zk true;
    User zkn V1_12 ext_zkn true;
    User zknd V1_12 ext_zknd true;
    User zkne V1_12 ext_zkne true;
    User zknh V1_12 ext_zknh true;
    User zkr V1_12 ext_zkr true;
    User zks V1_12 ext_zks true;
    User zksed V1_12 ext_zksed true;
    User zksh V1_12 ext_zksh true;
    User zkt V1_12 ext_zkt true;
    User ztso V1_12 ext_ztso true;
    User zvbb V1_12 ext_zvbb true;
    User zvbc V1_12 ext_zvbc true;
    User zve32f V1_10 ext_zve32f true;
    User zve32x V1_10 ext_zve32x true;
    User zve64f V1_10 ext_zve64f true;
    User zve64d V1_10 ext_zve64d true;
    User zve64x V1_10 ext_zve64x true;
    User zvfbfa V1_13 ext_zvfbfa false;
    User zvfbfmin V1_12 ext_zvfbfmin true;
    User zvfbfwma V1_12 ext_zvfbfwma true;
    User zvfh V1_12 ext_zvfh true;
    User zvfhmin V1_12 ext_zvfhmin true;
    User zvkb V1_12 ext_zvkb true;
    User zvkg V1_12 ext_zvkg true;
    User zvkn V1_12 ext_zvkn true;
    User zvknc V1_12 ext_zvknc true;
    User zvkned V1_12 ext_zvkned true;
    User zvkng V1_12 ext_zvkng true;
    User zvknha V1_12 ext_zvknha true;
    User zvknhb V1_12 ext_zvknhb true;
    User zvks V1_12 ext_zvks true;
    User zvksc V1_12 ext_zvksc true;
    User zvksed V1_12 ext_zvksed true;
    User zvksg V1_12 ext_zvksg true;
    User zvksh V1_12 ext_zvksh true;
    User zvkt V1_12 ext_zvkt true;
    User zhinx V1_12 ext_zhinx false;
    User zhinxmin V1_12 ext_zhinxmin false;
    User sdtrig V1_12 debug true;
    Internal shcounterenw V1_12 has_priv_1_12 true;
    Internal sha V1_12 ext_sha true;
    Internal shgatpa V1_12 has_priv_1_12 true;
    Internal shtvala V1_12 has_priv_1_12 true;
    Internal shvsatpa V1_12 has_priv_1_12 true;
    Internal shvstvala V1_12 has_priv_1_12 true;
    Internal shvstvecd V1_12 has_priv_1_12 true;
    User smaia V1_12 ext_smaia true;
    User smcdeleg V1_13 ext_smcdeleg true;
    User smcntrpmf V1_12 ext_smcntrpmf true;
    User smcsrind V1_13 ext_smcsrind true;
    User smctr V1_12 ext_smctr false;
    User smdbltrp V1_13 ext_smdbltrp false;
    User smepmp V1_12 ext_smepmp true;
    User smpmpmt V1_12 ext_smpmpmt true;
    User smrnmi V1_12 ext_smrnmi false;
    User smmpm V1_13 ext_smmpm true;
    User smnpm V1_13 ext_smnpm true;
    User smstateen V1_12 ext_smstateen true;
    User ssaia V1_12 ext_ssaia true;
    User ssccfg V1_13 ext_ssccfg true;
    Internal ssccptr V1_11 has_priv_1_11 true;
    User sscofpmf V1_12 ext_sscofpmf true;
    Internal sscounterenw V1_12 has_priv_1_12 true;
    User sscsrind V1_12 ext_sscsrind true;
    User ssctr V1_12 ext_ssctr false;
    User ssdbltrp V1_13 ext_ssdbltrp false;
    User ssnpm V1_13 ext_ssnpm true;
    User sspm V1_13 ext_sspm true;
    Internal ssstateen V1_12 ext_ssstateen true;
    Internal ssstrict V1_12 has_priv_1_12 true;
    User sstc V1_12 ext_sstc true;
    Internal sstvala V1_12 has_priv_1_12 true;
    Internal sstvecd V1_12 has_priv_1_12 true;
    Internal ssu64xl V1_12 has_priv_1_12 true;
    User supm V1_13 ext_supm true;
    User svade V1_11 ext_svade true;
    User svadu V1_12 ext_svadu true;
    User svinval V1_12 ext_svinval true;
    User svnapot V1_12 ext_svnapot true;
    User svpbmt V1_12 ext_svpbmt true;
    User svrsw60t59b V1_13 ext_svrsw60t59b true;
    Experimental svukte V1_13 ext_svukte false;
    User svvptc V1_13 ext_svvptc true;
    User xlrbr V1_13 ext_xlrbr true;
}

/// The extension called `name` in [`ISA_EXTS`].
pub fn isa_ext(name: &str) -> Option<&'static IsaExt> {
    ISA_EXTS.iter().find(|e| e.name == name)
}

fn ext(name: &str) -> &'static IsaExt {
    match isa_ext(name) {
        Some(e) => e,
        None => panic!("no extension {name} in isa_edata_arr"),
    }
}

/// A rule of `riscv_misa_ext_implied_rules[]` or `riscv_multi_ext_implied_rules[]`.
struct Rule {
    /// The misa bit, or 0 for a multi-letter extension.
    misa: u64,
    /// The multi-letter extension, when `misa` is 0.
    name: &'static str,
    implied_misa: u64,
    implied: &'static [&'static str],
}

const fn misa_rule(misa: u64, implied_misa: u64, implied: &'static [&'static str]) -> Rule {
    Rule { misa, name: "", implied_misa, implied }
}

const fn rule(name: &'static str, implied_misa: u64, implied: &'static [&'static str]) -> Rule {
    Rule { misa: 0, name, implied_misa, implied }
}

/// `riscv_misa_ext_implied_rules[]`.
static MISA_RULES: &[Rule] = &[
    misa_rule(RVA, 0, &["zalrsc", "zaamo"]),
    misa_rule(RVD, RVF, &[]),
    misa_rule(RVF, 0, &["zicsr"]),
    misa_rule(RVM, 0, &["zmmul"]),
    misa_rule(RVV, 0, &["zve64d"]),
    misa_rule(RVG, RVI | RVM | RVA | RVF | RVD, &["zicsr", "zifencei"]),
    misa_rule(RVB, 0, &["zba", "zbb", "zbs"]),
];

/// `riscv_multi_ext_implied_rules[]`.
static MULTI_RULES: &[Rule] = &[
    rule("zcb", 0, &["zca"]),
    rule("zcd", RVD, &["zca"]),
    rule("zce", 0, &["zcb", "zcmp", "zcmt"]),
    rule("zcf", RVF, &["zca"]),
    rule("zcmp", 0, &["zca"]),
    rule("zcmt", 0, &["zca", "zicsr"]),
    rule("zdinx", 0, &["zfinx"]),
    rule("zfa", RVF, &[]),
    rule("zfbfmin", RVF, &[]),
    rule("zfh", 0, &["zfhmin"]),
    rule("zfhmin", RVF, &[]),
    rule("zfinx", 0, &["zicsr"]),
    rule("zhinx", 0, &["zhinxmin"]),
    rule("zhinxmin", 0, &["zfinx"]),
    rule("zicntr", 0, &["zicsr"]),
    rule("zihpm", 0, &["zicsr"]),
    rule("zk", 0, &["zkn", "zkr", "zkt"]),
    rule("zkn", 0, &["zbkb", "zbkc", "zbkx", "zkne", "zknd", "zknh"]),
    rule("zks", 0, &["zbkb", "zbkc", "zbkx", "zksed", "zksh"]),
    rule("zvbb", 0, &["zvkb"]),
    rule("zve32f", RVF, &["zve32x"]),
    rule("zve32x", 0, &["zicsr"]),
    rule("zve64d", RVD, &["zve64f"]),
    rule("zve64f", RVF, &["zve32f", "zve64x"]),
    rule("zve64x", 0, &["zve32x"]),
    rule("zvfbfa", 0, &["zve32f", "zfbfmin"]),
    rule("zvfbfmin", 0, &["zve32f"]),
    rule("zvfbfwma", 0, &["zvfbfmin", "zfbfmin"]),
    rule("zvfh", 0, &["zvfhmin", "zfhmin"]),
    rule("zvfhmin", 0, &["zve32f"]),
    rule("zvkn", 0, &["zvkned", "zvknhb", "zvkb", "zvkt"]),
    rule("zvknc", 0, &["zvkn", "zvbc"]),
    rule("zvkng", 0, &["zvkn", "zvkg"]),
    rule("zvknhb", 0, &["zve64x", "zvknha"]),
    rule("zvks", 0, &["zvksed", "zvksh", "zvkb", "zvkt"]),
    rule("zvksc", 0, &["zvks", "zvbc"]),
    rule("zvksg", 0, &["zvks", "zvkg"]),
    rule("sha", RVH, &["smstateen", "ssstateen"]),
    rule("ssccfg", 0, &["smcsrind", "sscsrind", "smcdeleg"]),
    rule("supm", 0, &["ssnpm", "smnpm"]),
    rule("sspm", 0, &["smnpm"]),
    rule("smctr", RVS, &["sscsrind"]),
    rule("ssctr", RVS, &["sscsrind"]),
    rule("ssstateen", 0, &["smstateen"]),
];

/// A `RISCVCPUProfile`.
struct Profile {
    name: &'static str,
    u_parent: Option<usize>,
    s_parent: Option<usize>,
    misa: u64,
    priv_spec: Option<PrivVer>,
    satp_mode: Option<i8>,
    exts: &'static [&'static str],
}

const RVA22U64: usize = 0;
const RVA22S64: usize = 1;
const RVA23U64: usize = 2;
const RVA23S64: usize = 3;

/// `riscv_profiles[]`.
static PROFILES: [Profile; 4] = [
    Profile {
        name: "rva22u64",
        u_parent: None,
        s_parent: None,
        misa: RVI | RVM | RVA | RVF | RVD | RVC | RVB | RVU,
        priv_spec: None,
        satp_mode: None,
        exts: &[
            "zicsr",
            "zihintpause",
            "zba",
            "zbb",
            "zbs",
            "zfhmin",
            "zkt",
            "zicntr",
            "zihpm",
            "zicbom",
            "zicbop",
            "zicboz",
            "zic64b",
        ],
    },
    Profile {
        name: "rva22s64",
        u_parent: Some(RVA22U64),
        s_parent: None,
        misa: RVS,
        priv_spec: Some(PrivVer::V1_12),
        satp_mode: Some(VM_SV39),
        exts: &["zifencei", "svpbmt", "svinval", "svade"],
    },
    Profile {
        name: "rva23u64",
        u_parent: Some(RVA22U64),
        s_parent: None,
        misa: RVV,
        priv_spec: None,
        satp_mode: None,
        exts: &[
            "zvfhmin",
            "zvbb",
            "zvkt",
            "zihintntl",
            "zicond",
            "zimop",
            "zcmop",
            "zcb",
            "zfa",
            "zawrs",
            "supm",
        ],
    },
    Profile {
        name: "rva23s64",
        u_parent: Some(RVA23U64),
        s_parent: Some(RVA22S64),
        misa: RVS,
        priv_spec: Some(PrivVer::V1_13),
        satp_mode: Some(VM_SV39),
        exts: &["svnapot", "sstc", "sscofpmf", "ssnpm", "sha"],
    },
];

/// The kind of a CPU type, its abstract parent in QEMU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// `TYPE_RISCV_DYNAMIC_CPU`: `rv64` and `max`.
    Dynamic,
    /// `TYPE_RISCV_VENDOR_CPU`: a CPU that exists, which takes no new extensions.
    Vendor,
    /// `TYPE_RISCV_BARE_CPU`: `rv64i`, `rv64e` and the profile CPUs.
    Bare,
}

/// The `-cpu` models of qemu-system-riscv64 this port has.
pub const CPU_MODELS: &[&str] = &[
    "max",
    "rv64",
    "rv64e",
    "rv64i",
    "rva22s64",
    "rva22u64",
    "rva23s64",
    "rva23u64",
    "shakti-c",
    "sifive-e51",
    "sifive-u54",
    "xiangshan-nanhu",
];

/// The `-cpu` models of qemu-system-riscv64 this port does not have.
pub const OTHER_CPU_MODELS: &[&str] = &[
    "max32",
    "rv32",
    "x-rv128",
    "rv32i",
    "rv32e",
    "lowrisc-ibex",
    "sifive-e31",
    "sifive-e34",
    "sifive-u34",
    "thead-c906",
    "thead-c908",
    "thead-c908v",
    "veyron-v1",
    "tt-ascalon",
    "xiangshan-kunminghu",
    "mips-p8700",
    "host",
];

/// Why a `-cpu` property could not be set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PropError {
    /// QEMU has no such property.
    NotFound,
    /// QEMU has it, this port does not support the value.
    Unsupported,
    /// QEMU refuses the value, with this message.
    Invalid(String),
    /// QEMU refuses the value, with this message and this hint (`error_append_hint()`).
    Hinted(String, String),
}

/// A CPU model with its `-cpu` properties being set, before `riscv_cpu_finalize_features()`.
#[derive(Clone, Debug)]
pub struct CpuBuilder {
    name: String,
    kind: Kind,
    cfg: RiscvCfg,
    /// `env->priv_ver`, which a profile can set to "unset" (-1) for a moment.
    priv_ver: i8,
    /// `multi_ext_user_opts`: the extensions the user (or a profile) set.
    user_ext: Vec<(&'static str, bool)>,
    /// `misa_ext_user_opts`: the misa bits set.
    user_misa: u64,
    /// `satp_modes.map` and `satp_modes.init`.
    satp_map: u16,
    satp_init: u16,
    /// The profiles: enabled, and set by the user.
    profile_enabled: [bool; 4],
    profile_user_set: [bool; 4],
    /// The warnings the property setters print, once per hart.
    prop_warnings: Vec<String>,
}

impl CpuBuilder {
    /// `riscv_cpu_init()` and `riscv_tcg_cpu_instance_init()` for the model `name`. Gives
    /// `None` when this port does not have the model.
    pub fn new(name: &str) -> Option<CpuBuilder> {
        let mut cfg = RiscvCfg::zeroed();
        let (kind, profile) = match name {
            "rv64" | "max" => (Kind::Dynamic, None),
            "sifive-e51" | "sifive-u54" | "shakti-c" | "xiangshan-nanhu" => (Kind::Vendor, None),
            "rv64i" | "rv64e" => (Kind::Bare, None),
            "rva22u64" => (Kind::Bare, Some(RVA22U64)),
            "rva22s64" => (Kind::Bare, Some(RVA22S64)),
            "rva23u64" => (Kind::Bare, Some(RVA23U64)),
            "rva23s64" => (Kind::Bare, Some(RVA23S64)),
            _ => return None,
        };
        let bare = kind == Kind::Bare;
        // riscv_cpu_init().
        cfg.ext_zicntr = !bare;
        cfg.ext_zihpm = !bare;
        cfg.pmu_mask = 0xffff << 3;
        cfg.vlenb = VLENB as u32;
        cfg.elen = 64;
        cfg.cbom_blocksize = 64;
        cfg.cbop_blocksize = 64;
        cfg.cboz_blocksize = 64;
        cfg.pmp_regions = 16;
        cfg.max_satp_mode = -1;
        cfg.mvendorid = 0;
        cfg.marchid = 42;
        cfg.mimpid = 0;
        // The `debug` property defaults to true.
        cfg.debug = true;
        let mut priv_ver = PrivVer::V1_10.rank();
        // The class definitions, merged from the parent down.
        match kind {
            Kind::Dynamic => {
                cfg.mmu = true;
                cfg.pmp = true;
                priv_ver = PrivVer::LATEST.rank();
                cfg.max_satp_mode = VM_SV57;
            }
            Kind::Bare => {
                cfg.max_satp_mode = VM_SV57;
                cfg.misa_ext = if name == "rv64e" { RVE } else { RVI };
            }
            Kind::Vendor => {}
        }
        match name {
            "rv64" => {
                for e in [
                    "zicbom",
                    "zicbop",
                    "zicboz",
                    "zicntr",
                    "zicsr",
                    "zifencei",
                    "zihintntl",
                    "zihintpause",
                    "zihpm",
                    "zawrs",
                    "zfa",
                    "zba",
                    "zbb",
                    "zbc",
                    "zbs",
                    "sstc",
                    "svadu",
                    "svvptc",
                ] {
                    (ext(e).set)(&mut cfg, true);
                }
            }
            "sifive-e51" => {
                cfg.misa_ext = RVI | RVM | RVA | RVC | RVU;
                cfg.max_satp_mode = VM_MBARE;
                cfg.ext_zifencei = true;
                cfg.ext_zicsr = true;
                cfg.pmp = true;
                cfg.pmp_regions = 8;
            }
            "sifive-u54" | "shakti-c" => {
                cfg.misa_ext = RVI | RVM | RVA | RVF | RVD | RVC | RVS | RVU;
                cfg.max_satp_mode = VM_SV39;
                cfg.ext_zifencei = true;
                cfg.ext_zicsr = true;
                cfg.mmu = true;
                cfg.pmp = true;
                cfg.pmp_regions = 8;
            }
            "xiangshan-nanhu" => {
                cfg.misa_ext = RVG | RVC | RVB | RVS | RVU;
                priv_ver = PrivVer::V1_12.rank();
                for e in [
                    "zbc", "zbkb", "zbkc", "zbkx", "zknd", "zkne", "zknh", "zksed", "zksh",
                    "svinval",
                ] {
                    (ext(e).set)(&mut cfg, true);
                }
                cfg.mmu = true;
                cfg.pmp = true;
                cfg.max_satp_mode = VM_SV39;
            }
            _ => {}
        }
        let mut b = CpuBuilder {
            name: name.to_string(),
            kind,
            cfg,
            priv_ver,
            user_ext: Vec::new(),
            user_misa: 0,
            satp_map: 0,
            satp_init: 0,
            profile_enabled: [false; 4],
            profile_user_set: [false; 4],
            prop_warnings: Vec::new(),
        };
        // riscv_cpu_add_misa_properties(): the generic CPUs take the defaults of
        // misa_ext_cfgs[].
        if kind == Kind::Dynamic {
            b.cfg.misa_ext = RVA | RVC | RVD | RVF | RVI | RVM | RVS | RVU | RVH;
        }
        // riscv_cpu_add_profiles().
        if let Some(p) = profile {
            b.profile_enabled[p] = true;
            b.set_profile(p, true);
        }
        if name == "max" {
            b.init_max_extensions();
        }
        Some(b)
    }

    /// The model name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// `riscv_init_max_cpu_extensions()`, without the extensions this port does not have.
    fn init_max_extensions(&mut self) {
        self.cfg.misa_ext |= RVB | RVG | RVV;
        for e in ISA_EXTS {
            if e.is_x() || !e.ruvm {
                continue;
            }
            (e.set)(&mut self.cfg, true);
        }
        self.cfg.ext_svade = false;
        for e in [
            "zfinx", "zdinx", "zhinx", "zhinxmin", "zce", "zcmp", "zcmt", "zilsd", "zclsd", "zcf",
            "smrnmi", "smdbltrp",
        ] {
            (ext(e).set)(&mut self.cfg, false);
        }
    }

    fn ext_user_set(&self, name: &str) -> bool {
        self.user_ext.iter().any(|(n, _)| *n == name)
    }

    fn add_user_ext(&mut self, name: &'static str, value: bool) {
        if let Some(slot) = self.user_ext.iter_mut().find(|(n, _)| *n == name) {
            slot.1 = value;
        } else {
            self.user_ext.push((name, value));
        }
    }

    fn has(&self, bit: u64) -> bool {
        self.cfg.misa_ext & bit != 0
    }

    /// `cpu_bump_multi_ext_priv_ver()`.
    fn bump_priv_ver(&mut self, e: &IsaExt) {
        if self.priv_ver == PrivVer::LATEST.rank() {
            return;
        }
        if self.priv_ver < e.min.rank() {
            self.priv_ver = e.min.rank();
        }
    }

    /// `riscv_cpu_set_profile()`.
    fn set_profile(&mut self, p: usize, enabled: bool) {
        let profile = &PROFILES[p];
        if let Some(u) = profile.u_parent {
            self.set_profile(u, enabled);
        }
        if let Some(s) = profile.s_parent {
            self.set_profile(s, enabled);
        }
        self.profile_enabled[p] = enabled;
        if enabled {
            self.priv_ver = profile.priv_spec.map_or(-1, PrivVer::rank);
            if let Some(mode) = profile.satp_mode {
                self.cfg.mmu = true;
                self.set_satp(mode, true);
            }
        }
        for bit in MISA_BITS {
            if profile.misa & bit == 0 || (bit == RVI && !enabled) {
                continue;
            }
            self.user_misa |= bit;
            self.write_misa_bit(bit, enabled);
        }
        for name in profile.exts {
            let e = ext(name);
            if enabled {
                self.bump_priv_ver(e);
            }
            self.add_user_ext(e.name, enabled);
            (e.set)(&mut self.cfg, enabled);
        }
    }

    fn write_misa_bit(&mut self, bit: u64, value: bool) {
        if value {
            self.cfg.misa_ext |= bit;
        } else {
            self.cfg.misa_ext &= !bit;
        }
    }

    /// `cpu_riscv_set_satp()`.
    fn set_satp(&mut self, mode: i8, value: bool) {
        let bit = 1u16 << mode;
        if value {
            self.satp_map |= bit;
        } else {
            self.satp_map &= !bit;
        }
        self.satp_init |= bit;
    }

    fn vendor_err(&self, prop: &str) -> PropError {
        PropError::Invalid(format!(
            "CPU '{}' does not allow changing the value of '{prop}'",
            self.name
        ))
    }

    /// The warnings the properties set so far print.
    pub fn prop_warnings(&self) -> &[String] {
        &self.prop_warnings
    }

    /// `cpu_set_prop_err()` with the hint that gives the current value.
    fn vendor_err_hint(&self, prop: &str, current: impl std::fmt::Display) -> PropError {
        match self.vendor_err(prop) {
            PropError::Invalid(msg) => {
                PropError::Hinted(msg, format!("Current '{prop}' val: {current}\n"))
            }
            e => e,
        }
    }

    /// Sets the `-cpu` property `prop` to `value`, as the QOM property setter does.
    pub fn set(&mut self, prop: &str, value: &str) -> Result<(), PropError> {
        let boolean = || parse_bool(prop, value);
        // The misa properties.
        if let Some(bit) = misa_prop(prop) {
            let on = boolean()?;
            return self.set_misa(bit, on);
        }
        if let Some(e) = ISA_EXTS.iter().find(|e| e.prop_name().as_deref() == Some(prop)) {
            let on = boolean()?;
            return self.set_ext(e, on);
        }
        if let Some(p) = PROFILES.iter().position(|p| p.name == prop) {
            let on = boolean()?;
            if self.kind == Kind::Vendor {
                return Err(PropError::Invalid(format!(
                    "Profile {prop} is not available for vendor CPUs"
                )));
            }
            if on && !profile_missing(p).is_empty() {
                return Err(PropError::Unsupported);
            }
            self.profile_user_set[p] = true;
            self.set_profile(p, on);
            return Ok(());
        }
        if let Some(mode) = satp_mode_from_str(prop) {
            let on = boolean()?;
            self.set_satp(mode, on);
            return Ok(());
        }
        match prop {
            "mmu" => {
                let on = boolean()?;
                if on != self.cfg.mmu && self.kind == Kind::Vendor {
                    return Err(self.vendor_err("mmu"));
                }
                self.cfg.mmu = on;
            }
            "pmp" => {
                let on = boolean()?;
                if on != self.cfg.pmp && self.kind == Kind::Vendor {
                    return Err(self.vendor_err("pmp"));
                }
                self.cfg.pmp = on;
            }
            "num-pmp-regions" => {
                let n: u8 = parse_uint(prop, value)?;
                if n != self.cfg.pmp_regions && self.kind == Kind::Vendor {
                    return Err(self.vendor_err(prop));
                }
                let max = if self.priv_ver < PrivVer::V1_12.rank() { 16 } else { 64 };
                if n > max {
                    return Err(PropError::Invalid(
                        "Number of PMP regions exceeds maximum available".into(),
                    ));
                }
                if n > 16 {
                    return Err(PropError::Unsupported);
                }
                self.cfg.pmp_regions = n;
            }
            "priv_spec" => {
                let Some(v) = PrivVer::parse(value) else {
                    return Err(PropError::Invalid(format!(
                        "Unsupported privilege spec version '{value}'"
                    )));
                };
                if v.rank() != self.priv_ver && self.kind == Kind::Vendor {
                    let current = PrivVer::from_rank(self.priv_ver).as_str();
                    return Err(self.vendor_err_hint(prop, current));
                }
                self.priv_ver = v.rank();
            }
            "vext_spec" => {
                if value != "v1.0" {
                    return Err(PropError::Invalid(format!(
                        "Unsupported vector spec version '{value}'"
                    )));
                }
            }
            "debug" => self.cfg.debug = boolean()?,
            "short-isa-string" => self.cfg.short_isa_string = boolean()?,
            "rvv_ta_all_1s" => self.cfg.rvv_ta_all_1s = boolean()?,
            "rvv_ma_all_1s" => self.cfg.rvv_ma_all_1s = boolean()?,
            "rvv_vl_half_avl" => self.cfg.rvv_vl_half_avl = boolean()?,
            "rvv_vsetvl_x0_vill" => self.cfg.rvv_vsetvl_x0_vill = boolean()?,
            "pmu-mask" => {
                let mask: u32 = parse_uint(prop, value)?;
                if mask != self.cfg.pmu_mask && self.kind == Kind::Vendor {
                    return Err(self.vendor_err_hint(prop, format!("{:x}", self.cfg.pmu_mask)));
                }
                if mask.count_ones() > 29 {
                    return Err(PropError::Invalid(
                        "Number of counters exceeds maximum available".into(),
                    ));
                }
                self.cfg.pmu_mask = mask;
            }
            "vlen" => {
                let v: u16 = parse_uint(prop, value)?;
                if !v.is_power_of_two() {
                    return Err(PropError::Invalid(
                        "Vector extension VLEN must be power of 2".into(),
                    ));
                }
                if u32::from(v) != self.cfg.vlenb << 3 && self.kind == Kind::Vendor {
                    return Err(self.vendor_err_hint(prop, self.cfg.vlenb << 3));
                }
                // The value only matters with a vector extension, and finalize() refuses one
                // this port does not have then.
                self.cfg.vlenb = u32::from(v) >> 3;
            }
            "elen" => {
                let v: u16 = parse_uint(prop, value)?;
                if !v.is_power_of_two() {
                    return Err(PropError::Invalid(
                        "Vector extension ELEN must be power of 2".into(),
                    ));
                }
                if u32::from(v) != self.cfg.elen && self.kind == Kind::Vendor {
                    return Err(self.vendor_err_hint(prop, self.cfg.elen));
                }
                self.cfg.elen = u32::from(v);
            }
            "cbom_blocksize" | "cbop_blocksize" | "cboz_blocksize" => {
                let v: u16 = parse_uint(prop, value)?;
                let field = match prop {
                    "cbom_blocksize" => &mut self.cfg.cbom_blocksize,
                    "cbop_blocksize" => &mut self.cfg.cbop_blocksize,
                    _ => &mut self.cfg.cboz_blocksize,
                };
                let current = *field;
                if v != current && self.kind == Kind::Vendor {
                    return Err(self.vendor_err_hint(prop, current));
                }
                // The value only matters with the extension, and finalize() refuses one this
                // port does not have then.
                *field = v;
            }
            "mvendorid" => {
                let v: u32 = parse_uint(prop, value)?;
                if self.kind != Kind::Dynamic && v != self.cfg.mvendorid {
                    return Err(PropError::Invalid(format!(
                        "Unable to change {}-riscv-cpu mvendorid (0x{:x})",
                        self.name, self.cfg.mvendorid
                    )));
                }
                self.cfg.mvendorid = v;
            }
            "mimpid" => {
                let v: u64 = parse_uint(prop, value)?;
                if self.kind != Kind::Dynamic && v != self.cfg.mimpid {
                    return Err(PropError::Invalid(format!(
                        "Unable to change {}-riscv-cpu mimpid (0x{})",
                        self.name, self.cfg.mimpid
                    )));
                }
                self.cfg.mimpid = v;
            }
            "marchid" => {
                let v: u64 = parse_uint(prop, value)?;
                if self.kind != Kind::Dynamic && v != self.cfg.marchid {
                    return Err(PropError::Invalid(format!(
                        "Unable to change {}-riscv-cpu marchid (0x{})",
                        self.name, self.cfg.marchid
                    )));
                }
                if v == 1 << 63 {
                    return Err(PropError::Invalid(
                        "Unable to set marchid with MSB (64) bit set and the remaining bits zero"
                            .into(),
                    ));
                }
                self.cfg.marchid = v;
            }
            "pmu-num" => {
                let n: u8 = parse_uint(prop, value)?;
                let current = self.cfg.pmu_mask.count_ones();
                if u32::from(n) != current && self.kind == Kind::Vendor {
                    return Err(self.vendor_err_hint(prop, current));
                }
                if n > 29 {
                    return Err(PropError::Invalid(
                        "Number of counters exceeds maximum available".into(),
                    ));
                }
                self.cfg.pmu_mask = if n == 0 { 0 } else { ((1u32 << n) - 1) << 3 };
                self.prop_warnings
                    .push("\"pmu-num\" property is deprecated; use \"pmu-mask\"".into());
            }
            // virt sets `resetvec` of each hart after the global properties, and the RNMI
            // vectors only matter with Smrnmi, which this port does not have.
            "resetvec" | "rnmi-interrupt-vector" | "rnmi-exception-vector" => {
                let _: u64 = parse_uint(prop, value)?;
            }
            "pmp-granularity" => {
                let v: u32 = parse_uint(prop, value)?;
                if v < 4 && !v.is_power_of_two() {
                    return Err(PropError::Invalid(
                        "PMP granularity must be a power of 2 and at least 4".into(),
                    ));
                }
                if v != 4 && self.kind == Kind::Vendor {
                    return Err(self.vendor_err(prop));
                }
                if v != 4 {
                    return Err(PropError::Unsupported);
                }
            }
            "num-triggers" => {
                let v: u32 = parse_uint(prop, value)?;
                if v != NUM_TRIGGERS as u32 {
                    return Err(PropError::Unsupported);
                }
            }
            "big-endian" | "x-misa-w" => {
                if boolean()? {
                    return Err(PropError::Unsupported);
                }
            }
            _ => return Err(PropError::NotFound),
        }
        Ok(())
    }

    /// `cpu_set_misa_ext_cfg()`.
    fn set_misa(&mut self, bit: u64, on: bool) -> Result<(), PropError> {
        self.user_misa |= bit;
        if on == self.has(bit) {
            return Ok(());
        }
        if on {
            if self.kind == Kind::Vendor {
                return Err(PropError::Invalid(format!(
                    "'{}' CPU does not allow enabling extensions",
                    self.name
                )));
            }
            if bit == RVH && self.priv_ver < PrivVer::V1_12.rank() {
                self.priv_ver = PrivVer::V1_12.rank();
            }
        }
        self.write_misa_bit(bit, on);
        Ok(())
    }

    /// `cpu_set_multi_ext_cfg()`.
    fn set_ext(&mut self, e: &'static IsaExt, on: bool) -> Result<(), PropError> {
        if on && !e.ruvm {
            return Err(PropError::Unsupported);
        }
        self.add_user_ext(e.name, on);
        if on == e.enabled(&self.cfg) {
            return Ok(());
        }
        if on && self.kind == Kind::Vendor {
            return Err(PropError::Invalid(format!(
                "'{}' CPU does not allow enabling extensions",
                self.name
            )));
        }
        if on {
            self.bump_priv_ver(e);
        }
        (e.set)(&mut self.cfg, on);
        Ok(())
    }

    /// `cpu_cfg_ext_auto_update()`.
    fn auto_update(&mut self, name: &str, value: bool) {
        let e = ext(name);
        if e.enabled(&self.cfg) == value || self.ext_user_set(name) {
            return;
        }
        if value && self.priv_ver != PrivVer::LATEST.rank() && self.priv_ver < e.min.rank() {
            return;
        }
        (e.set)(&mut self.cfg, value);
    }

    /// `cpu_enable_implied_rule()`. `done` is the `rule->enabled` bitmap for this hart.
    fn enable_rule(&mut self, r: &'static Rule, done: &mut Vec<*const Rule>) {
        let key: *const Rule = r;
        if done.contains(&key) {
            return;
        }
        if r.implied_misa != 0 {
            for bit in MISA_BITS {
                if r.implied_misa & bit == 0 {
                    continue;
                }
                if self.user_misa & bit != 0 && !self.has(bit) {
                    continue;
                }
                self.cfg.misa_ext |= bit;
                if let Some(ir) = MISA_RULES.iter().find(|x| x.misa == bit) {
                    self.enable_rule(ir, done);
                }
            }
        }
        for name in r.implied {
            self.auto_update(name, true);
            if let Some(ir) = MULTI_RULES.iter().find(|x| x.name == *name) {
                self.enable_rule(ir, done);
            }
        }
        done.push(key);
    }

    /// `riscv_cpu_enable_implied_rules()`.
    fn enable_implied_rules(&mut self) {
        // cpu_enable_zc_implied_rules(), for RV64.
        if self.cfg.ext_zce {
            for n in ["zca", "zcb", "zcmp", "zcmt"] {
                self.auto_update(n, true);
            }
        }
        if self.has(RVC) && self.priv_ver >= PrivVer::V1_12.rank() {
            self.auto_update("zca", true);
            if self.has(RVD) {
                self.auto_update("zcd", true);
            }
        }
        // cpu_enable_zilsd_implied_rules().
        if self.cfg.ext_zilsd && self.has(RVC) {
            self.auto_update("zclsd", true);
        }
        if self.cfg.ext_zclsd {
            self.auto_update("zca", true);
            self.auto_update("zilsd", true);
        }
        let mut done = Vec::new();
        for r in MISA_RULES {
            if self.has(r.misa) {
                self.enable_rule(r, &mut done);
            }
        }
        for r in MULTI_RULES {
            if ext(r.name).enabled(&self.cfg) {
                self.enable_rule(r, &mut done);
            }
        }
    }

    /// `riscv_cpu_satp_mode_finalize()`.
    fn satp_mode_finalize(&mut self, warn: &mut Vec<String>) -> Result<(), String> {
        let max = self.cfg.max_satp_mode;
        if max == -1 {
            return Ok(());
        }
        let supported: u16 =
            (0..=max).filter(|&i| satp_mode_valid(i)).fold(0, |acc, i| acc | (1 << i));
        if self.satp_map == 0 {
            if self.satp_init == 0 {
                if self.kind == Kind::Bare {
                    warn.push("No satp mode set. Defaulting to 'bare'".into());
                    self.cfg.max_satp_mode = VM_MBARE;
                }
            } else {
                'outer: for i in 1..16 {
                    if self.satp_init & (1 << i) != 0 && supported & (1 << i) != 0 {
                        for j in (0..i).rev() {
                            if supported & (1 << j) != 0 {
                                self.cfg.max_satp_mode = j;
                                break 'outer;
                            }
                        }
                    }
                }
            }
            return Ok(());
        }
        let map_max = (15 - self.satp_map.leading_zeros()) as i8;
        if map_max > max {
            return Err(format!(
                "satp_mode {} is higher than hw max capability {}",
                satp_mode_str(map_max),
                satp_mode_str(max)
            ));
        }
        for i in (0..map_max).rev() {
            if self.satp_map & (1 << i) == 0
                && self.satp_init & (1 << i) != 0
                && supported & (1 << i) != 0
            {
                return Err(format!(
                    "cannot disable {} satp mode if {} is enabled",
                    satp_mode_str(i),
                    satp_mode_str(map_max)
                ));
            }
        }
        self.cfg.max_satp_mode = map_max;
        Ok(())
    }

    /// `riscv_cpu_finalize_features()` for hart `hartid`: the configuration the hart runs
    /// with. The warnings QEMU prints for the hart, from its properties and then from this,
    /// go to `warn`, also when it fails.
    pub fn finalize(mut self, hartid: u64, warn: &mut Vec<String>) -> Result<RiscvCfg, String> {
        warn.append(&mut self.prop_warnings);
        self.satp_mode_finalize(warn)?;
        self.enable_implied_rules();
        self.update_misa_c(warn);
        // riscv_cpu_update_misa_x().
        if ISA_EXTS.iter().any(|e| e.name.starts_with('x') && e.enabled(&self.cfg)) {
            self.cfg.misa_ext |= RVX;
        }
        // riscv_cpu_validate_misa_priv().
        if self.has(RVH) && self.priv_ver < PrivVer::V1_12.rank() {
            return Err("H extension requires priv spec 1.12.0".into());
        }
        self.update_cfg();
        self.validate_profiles(warn);
        if self.cfg.ext_smepmp && !self.cfg.pmp {
            return Err("Invalid configuration: Smepmp requires PMP support".into());
        }
        self.validate_set_extensions(hartid, warn)?;
        // riscv_pmu_init().
        if self.cfg.pmu_mask & 7 != 0 {
            return Err("\"pmu-mask\" contains invalid bits (0-2) set".into());
        }
        // QEMU takes any block size. Here the block is a power of 2 that fits in a page.
        let block_ok = |v: u16| v.is_power_of_two() && (8..=4096).contains(&v);
        for (on, prop, v) in [
            (self.cfg.ext_zicbom, "cbom_blocksize", self.cfg.cbom_blocksize),
            (self.cfg.ext_zicboz, "cboz_blocksize", self.cfg.cboz_blocksize),
        ] {
            if on && !block_ok(v) {
                return Err(format!("CPU property {prop}={v} is not supported by ruvm yet"));
            }
        }
        self.cfg.priv_ver = PrivVer::from_rank(self.priv_ver);
        Ok(self.cfg)
    }

    /// `riscv_cpu_update_misa_c()` for RV64.
    fn update_misa_c(&mut self, warn: &mut Vec<String>) {
        if self.has(RVC) {
            return;
        }
        let set = (self.cfg.ext_zca && !self.has(RVF)) || (self.cfg.ext_zca && self.cfg.ext_zcd);
        if set {
            if self.user_misa & RVC != 0 {
                warn.push("RVC mandated by Zca/Zcf/Zcd extensions".into());
                return;
            }
            self.cfg.misa_ext |= RVC;
        }
    }

    /// `riscv_cpu_update_cfg()`.
    fn update_cfg(&mut self) {
        let c = &mut self.cfg;
        if self.priv_ver >= PrivVer::V1_11.rank() {
            c.has_priv_1_11 = true;
        }
        if self.priv_ver >= PrivVer::V1_12.rank() {
            c.has_priv_1_12 = true;
        }
        if self.priv_ver >= PrivVer::V1_13.rank() {
            c.has_priv_1_13 = true;
        }
        c.ext_zic64b = c.cbom_blocksize == 64
            && c.cbop_blocksize == 64
            && c.cboz_blocksize == 64
            && c.has_priv_1_12;
        c.ext_ssstateen = c.ext_smstateen;
        c.ext_sha = c.misa_ext & RVH != 0 && c.ext_ssstateen;
        c.ext_ziccrse = c.has_priv_1_11;
    }

    /// `riscv_cpu_validate_profiles()`, which only warns.
    fn validate_profiles(&mut self, warn: &mut Vec<String>) {
        let mut present = [false; 4];
        for (i, p) in PROFILES.iter().enumerate() {
            let send_warn = self.profile_user_set[i] && self.profile_enabled[i];
            let mut ok = true;
            if let Some(mode) = p.satp_mode {
                let max = self.cfg.max_satp_mode;
                if mode > max {
                    if send_warn {
                        warn.push(format!(
                            "Profile {} requires satp mode {}, but satp mode {} was set",
                            p.name,
                            satp_mode_str(mode),
                            satp_mode_str(max)
                        ));
                    }
                    ok = false;
                }
            }
            if let Some(v) = p.priv_spec {
                if v.rank() > self.priv_ver {
                    ok = false;
                    if send_warn {
                        warn.push(format!(
                            "Profile {} requires priv spec {}, but priv ver {} was set",
                            p.name,
                            v.as_str(),
                            PrivVer::from_rank(self.priv_ver).as_str()
                        ));
                    }
                }
            }
            for bit in MISA_BITS {
                if p.misa & bit != 0 && !self.has(bit) {
                    ok = false;
                    if send_warn {
                        warn.push(format!(
                            "Profile {} mandates disabled extension {}",
                            p.name,
                            misa_name(bit)
                        ));
                    }
                }
            }
            for name in p.exts {
                if !ext(name).enabled(&self.cfg) {
                    ok = false;
                    if send_warn {
                        warn.push(format!("Profile {} mandates disabled extension {name}", p.name));
                    }
                }
            }
            if ok {
                if let Some(u) = p.u_parent {
                    ok = present[u];
                }
            }
            if ok {
                if let Some(s) = p.s_parent {
                    ok = present[s];
                }
            }
            present[i] = ok;
        }
    }

    /// `riscv_cpu_validate_set_extensions()`.
    fn validate_set_extensions(
        &mut self,
        hartid: u64,
        warn: &mut Vec<String>,
    ) -> Result<(), String> {
        let err = |m: &str| Err(m.to_string());
        if self.has(RVG) {
            let send = self.user_misa & RVG != 0;
            for bit in [RVI, RVM, RVA, RVF, RVD] {
                if !self.has(bit) && send {
                    warn.push(format!("RVG mandates disabled extension {}", misa_name(bit)));
                }
            }
            if !self.cfg.ext_zicsr && send {
                warn.push("RVG mandates disabled extension zicsr".into());
            }
            if !self.cfg.ext_zifencei && send {
                warn.push("RVG mandates disabled extension zifencei".into());
            }
        }
        if self.has(RVB) {
            for (n, on) in
                [("zba", self.cfg.ext_zba), ("zbb", self.cfg.ext_zbb), ("zbs", self.cfg.ext_zbs)]
            {
                if !on {
                    warn.push(format!("RVB mandates disabled extension {n}"));
                }
            }
        }
        let c = self.cfg;
        let has = |bit: u64| c.misa_ext & bit != 0;
        if has(RVI) && has(RVE) {
            return err("I and E extensions are incompatible");
        }
        if !has(RVI) && !has(RVE) {
            return err("Either I or E extension must be set");
        }
        if has(RVS) && !has(RVU) {
            return err("Setting S extension without U extension is illegal");
        }
        if has(RVH) && !has(RVI) {
            return err("H depends on an I base integer ISA with 32 x registers");
        }
        if has(RVH) && !has(RVS) {
            return err("H extension implicitly requires S-mode");
        }
        if has(RVF) && !c.ext_zicsr {
            return err("F extension requires Zicsr");
        }
        if c.ext_zacas && !has(RVA) {
            return err("Zacas extension requires A extension");
        }
        if c.ext_zawrs && !has(RVA) {
            return err("Zawrs extension requires A extension");
        }
        if c.ext_zfa && !has(RVF) {
            return err("Zfa extension requires F extension");
        }
        if c.ext_zfhmin && !has(RVF) {
            return err("Zfh/Zfhmin extensions require F extension");
        }
        if c.ext_zfbfmin && !has(RVF) {
            return err("Zfbfmin extension depends on F extension");
        }
        if has(RVD) && !has(RVF) {
            return err("D extension requires F extension");
        }
        // riscv_cpu_validate_v().
        let min_vlen = if has(RVV) {
            128
        } else if c.ext_zve64x {
            64
        } else if c.ext_zve32x {
            32
        } else {
            0
        };
        if min_vlen != 0 {
            let vlen = c.vlenb << 3;
            if vlen > RV_VLEN_MAX || vlen < min_vlen {
                return Err(format!(
                    "Vector extension implementation only supports VLEN in the range \
                     [{min_vlen}, {RV_VLEN_MAX}]"
                ));
            }
            // The vector registers of this port have one size.
            if vlen as usize != VLENB * 8 {
                return Err(format!("CPU property vlen={vlen} is not supported by ruvm yet"));
            }
            if c.elen > 64 || c.elen < 8 {
                return err(
                    "Vector extension implementation only supports ELEN in the range [8, 64]",
                );
            }
            if vlen < c.elen {
                return err(
                    "Vector extension implementation requires VLEN to be greater than or equal \
                     to ELEN",
                );
            }
        }
        if c.ext_zve64d && !has(RVD) {
            return err("Zve64d/V extensions require D extension");
        }
        if c.ext_zve32f && !has(RVF) {
            return err("Zve32f/Zve64f extensions require F extension");
        }
        if c.ext_zvfhmin && !c.ext_zve32f {
            return err("Zvfh/Zvfhmin extensions require Zve32f extension");
        }
        if c.ext_zvfh && !c.ext_zfhmin {
            return err("Zvfh extensions requires Zfhmin extension");
        }
        if c.ext_zvfbfmin && !c.ext_zve32f {
            return err("Zvfbfmin extension depends on Zve32f extension");
        }
        if c.ext_zvfbfwma && !c.ext_zvfbfmin {
            return err("Zvfbfwma extension depends on Zvfbfmin extension");
        }
        if c.ext_zvfbfa && (!c.ext_zve32f || !c.ext_zfbfmin) {
            return err("Zvfbfa extension requires Zve32f extension and Zfbfmin extension");
        }
        if (c.ext_zdinx || c.ext_zhinxmin) && !c.ext_zfinx {
            return err("Zdinx/Zhinx/Zhinxmin extensions require Zfinx");
        }
        if c.ext_zfinx {
            if !c.ext_zicsr {
                return err("Zfinx extension requires Zicsr");
            }
            if has(RVF) {
                return err("Zfinx cannot be supported together with F extension");
            }
        }
        if c.ext_zcmop && !c.ext_zca {
            return err("Zcmop extensions require Zca");
        }
        if c.ext_zcf {
            return err("Zcf extension is only relevant to RV32");
        }
        if !has(RVD) && c.ext_zcd {
            return err("Zcd extension requires D extension");
        }
        if (c.ext_zcf || c.ext_zcd || c.ext_zcb || c.ext_zcmp || c.ext_zcmt) && !c.ext_zca {
            return err("Zcf/Zcd/Zcb/Zcmp/Zcmt extensions require Zca extension");
        }
        if c.ext_zcd && (c.ext_zcmp || c.ext_zcmt) {
            return err("Zcmp/Zcmt extensions are incompatible with Zcd extension");
        }
        if c.ext_zcmt && !c.ext_zicsr {
            return err("Zcmt extension requires Zicsr extension");
        }
        if (c.ext_zvbb
            || c.ext_zvkb
            || c.ext_zvkg
            || c.ext_zvkned
            || c.ext_zvknha
            || c.ext_zvksed
            || c.ext_zvksh)
            && !c.ext_zve32x
        {
            return err("Vector crypto extensions require V or Zve* extensions");
        }
        if (c.ext_zvbc || c.ext_zvknhb) && !c.ext_zve64x {
            return err("Zvbc and Zvknhb extensions require V or Zve64x extensions");
        }
        if self.cfg.ext_zicntr && !self.cfg.ext_zicsr {
            if self.ext_user_set("zicntr") {
                return err("zicntr requires zicsr");
            }
            self.cfg.ext_zicntr = false;
        }
        if self.cfg.ext_zihpm && !self.cfg.ext_zicsr {
            if self.ext_user_set("zihpm") {
                return err("zihpm requires zicsr");
            }
            self.cfg.ext_zihpm = false;
        }
        let c = self.cfg;
        if c.ext_zicfiss {
            if !c.ext_zicsr {
                return err("zicfiss extension requires zicsr extension");
            }
            if !has(RVA) {
                return err("zicfiss extension requires A extension");
            }
            if !has(RVS) {
                return err("zicfiss extension requires S");
            }
            if !c.ext_zimop {
                return err("zicfiss extension requires zimop extension");
            }
            if c.ext_zca && !c.ext_zcmop {
                return err("zicfiss with zca requires zcmop extension");
            }
        }
        if !c.ext_zihpm {
            self.cfg.pmu_mask = 0;
        }
        if c.ext_zclsd {
            if has(RVC) && has(RVF) {
                return err("Zclsd cannot be supported together with C and F extension");
            }
            if c.ext_zcf {
                return err("Zclsd cannot be supported together with Zcf extension");
            }
        }
        if c.ext_zicfilp && !c.ext_zicsr {
            return err("zicfilp extension requires zicsr extension");
        }
        if (c.ext_smctr || c.ext_ssctr) && (!has(RVS) || !c.ext_sscsrind) {
            if self.ext_user_set("smctr") || self.ext_user_set("ssctr") {
                return err("Smctr and Ssctr require S-mode and Sscsrind");
            }
            self.cfg.ext_smctr = false;
            self.cfg.ext_ssctr = false;
        }
        if c.ext_svrsw60t59b && !c.mmu {
            return err("svrsw60t59b is not supported on RV32 and MMU-less platforms");
        }
        for (name, on) in [("svpbmt", c.ext_svpbmt), ("svnapot", c.ext_svnapot)] {
            if on && c.max_satp_mode < VM_SV39 {
                (ext(name).set)(&mut self.cfg, false);
                if self.ext_user_set(name) {
                    warn.push(format!(
                        "{name} requires at least satp sv39, current satp mode: {}",
                        satp_mode_str(c.max_satp_mode)
                    ));
                }
            }
        }
        // riscv_cpu_disable_priv_spec_isa_exts().
        for e in ISA_EXTS {
            if e.enabled(&self.cfg) && self.priv_ver < e.min.rank() {
                if matches!(e.name, "zicntr" | "zihpm" | "sdtrig") {
                    continue;
                }
                (e.set)(&mut self.cfg, false);
                warn.push(format!(
                    "disabling {} extension for hart 0x{hartid:x} because privilege spec \
                     version does not match",
                    e.name
                ));
            }
        }
        Ok(())
    }
}

/// The extensions profile `p` (with its parents) mandates that this port does not have.
fn profile_missing(p: usize) -> Vec<&'static str> {
    let mut out = Vec::new();
    let mut stack = vec![p];
    while let Some(i) = stack.pop() {
        let prof = &PROFILES[i];
        stack.extend(prof.u_parent);
        stack.extend(prof.s_parent);
        for n in prof.exts {
            if !ext(n).ruvm && !out.contains(n) {
                out.push(*n);
            }
        }
    }
    out
}

/// The extensions a CPU model starts with that this port does not have. A model with any is
/// refused.
pub fn model_missing(name: &str) -> Vec<&'static str> {
    match name {
        "rva22u64" => profile_missing(RVA22U64),
        "rva22s64" => profile_missing(RVA22S64),
        "rva23u64" => profile_missing(RVA23U64),
        "rva23s64" => profile_missing(RVA23S64),
        _ => Vec::new(),
    }
}

/// `riscv_get_misa_ext_name()`.
fn misa_name(bit: u64) -> String {
    let i = bit.trailing_zeros() as u8;
    char::from(b'a' + i).to_string()
}

/// The misa bit of a misa property name, `misa_ext_cfgs[]`.
fn misa_prop(name: &str) -> Option<u64> {
    match name {
        "a" => Some(RVA),
        "c" => Some(RVC),
        "d" => Some(RVD),
        "f" => Some(RVF),
        "i" => Some(RVI),
        "e" => Some(RVE),
        "m" => Some(RVM),
        "s" => Some(RVS),
        "u" => Some(RVU),
        "h" => Some(RVH),
        "v" => Some(RVV),
        "g" => Some(RVG),
        "b" => Some(RVB),
        _ => None,
    }
}

/// `visit_type_bool()` of the keyval input visitor.
fn parse_bool(name: &str, value: &str) -> Result<bool, PropError> {
    match value {
        "on" | "yes" | "true" | "y" => Ok(true),
        "off" | "no" | "false" | "n" => Ok(false),
        _ => Err(PropError::Invalid(format!("Parameter '{name}' expects 'on' or 'off'"))),
    }
}

/// `visit_type_uintN()` of the keyval input visitor.
fn parse_uint<T: TryFrom<u64>>(name: &str, value: &str) -> Result<T, PropError> {
    let n = if let Some(hex) = value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).ok()
    } else if value.len() > 1 && value.starts_with('0') {
        u64::from_str_radix(&value[1..], 8).ok()
    } else {
        value.parse::<u64>().ok()
    };
    let Some(n) = n else {
        return Err(PropError::Invalid(format!(
            "Parameter '{name}' expects a non-negative number below 2^64"
        )));
    };
    T::try_from(n).map_err(|_| {
        PropError::Invalid(format!("Parameter '{name}' expects uint{}", size_of::<T>() * 8))
    })
}

impl Default for RiscvCfg {
    fn default() -> RiscvCfg {
        static RV64: std::sync::OnceLock<RiscvCfg> = std::sync::OnceLock::new();
        *RV64.get_or_init(|| RiscvCfg::model("rv64"))
    }
}

impl RiscvCfg {
    /// The configuration of CPU model `name` with no properties, for tests and defaults.
    ///
    /// # Panics
    ///
    /// When this port does not have the model, or the model does not finalize.
    pub fn model(name: &str) -> RiscvCfg {
        match CpuBuilder::new(name).map(|b| b.finalize(0, &mut Vec::new())) {
            Some(Ok(cfg)) => cfg,
            _ => panic!("CPU model {name} does not finalize"),
        }
    }

    /// `max`.
    pub fn max() -> RiscvCfg {
        RiscvCfg::model("max")
    }

    /// Whether the misa extension `bit` is on, `riscv_has_ext()`.
    pub fn has(&self, bit: u64) -> bool {
        self.misa_ext & bit != 0
    }

    /// The `misa` extension bits.
    pub fn misa_ext(&self) -> u64 {
        self.misa_ext
    }

    /// Whether the H extension is on.
    pub fn ext_h(&self) -> bool {
        self.has(RVH)
    }

    /// Whether the V extension is on.
    pub fn ext_v(&self) -> bool {
        self.has(RVV)
    }

    /// The privileged version is at least `v`.
    pub fn priv_at_least(&self, v: PrivVer) -> bool {
        self.priv_ver >= v
    }

    /// `riscv_cpu_allow_16bit_insn()`: whether jumps may go to 2-byte aligned targets. From
    /// priv 1.12 on that is Zca, which C implies; before, it is C.
    pub fn allow_16bit_insn(&self) -> bool {
        if self.priv_ver >= PrivVer::V1_12 { self.ext_zca } else { self.has(RVC) }
    }

    /// `get_xepc_mask()`: the bits `mepc` and `sepc` keep. With IALIGN=16 (C or any Zc*
    /// extension) only bit 0 is cleared, with IALIGN=32 both low bits are.
    pub fn xepc_mask(&self) -> u64 {
        let ialign16 = self.has(RVC)
            || self.ext_zca
            || self.ext_zcb
            || self.ext_zcd
            || self.ext_zce
            || self.ext_zcf
            || self.ext_zcmp
            || self.ext_zcmt;
        if ialign16 { !1 } else { !3 }
    }

    /// `riscv_isa_string()`: `rv64`, the misa letters, then each multi-letter extension
    /// after an underscore unless `short-isa-string` is set.
    pub fn isa_string(&self) -> String {
        let mut s = String::from("rv64");
        for &l in SINGLE_LETTER_EXTS {
            if self.misa_ext & rvx(l) != 0 {
                s.push(char::from(l.to_ascii_lowercase()));
            }
        }
        if !self.short_isa_string {
            for e in ISA_EXTS {
                if e.enabled(self) {
                    s.push('_');
                    s.push_str(e.name);
                }
            }
        }
        s
    }

    /// `riscv_isa_extensions_list()`: each misa letter, then each multi-letter extension.
    pub fn isa_extensions(&self) -> Vec<String> {
        let mut v: Vec<String> = SINGLE_LETTER_EXTS
            .iter()
            .filter(|&&l| self.misa_ext & rvx(l) != 0)
            .map(|&l| char::from(l.to_ascii_lowercase()).to_string())
            .collect();
        v.extend(ISA_EXTS.iter().filter(|e| e.enabled(self)).map(|e| e.name.to_string()));
        v
    }

    /// The `mmu-type` of the cpu node, or `None` when the CPU leaves the satp mode open.
    pub fn mmu_type(&self) -> Option<String> {
        (self.max_satp_mode != -1).then(|| format!("riscv,{}", satp_mode_str(self.max_satp_mode)))
    }

    /// Whether `satp` mode `mode` can be written, `validate_vm()`.
    pub fn satp_mode_ok(&self, mode: u64) -> bool {
        mode <= self.max_satp_mode.max(0) as u64 && satp_mode_valid(mode as i8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn isa(model: &str, props: &[(&str, &str)]) -> String {
        let mut b = CpuBuilder::new(model).unwrap();
        for (p, v) in props {
            b.set(p, v).unwrap();
        }
        b.finalize(0, &mut Vec::new()).unwrap().isa_string()
    }

    #[test]
    fn rv64_matches_qemu() {
        assert_eq!(
            isa("rv64", &[]),
            "rv64imafdch_zic64b_zicbom_zicbop_zicboz_ziccamoa_ziccif_zicclsm_ziccrse_zicntr_\
             zicsr_zifencei_zihintntl_zihintpause_zihpm_zmmul_za64rs_zaamo_zalrsc_zawrs_zfa_\
             zca_zcd_zba_zbb_zbc_zbs_sdtrig_shcounterenw_shgatpa_shtvala_shvsatpa_shvstvala_\
             shvstvecd_ssccptr_sscounterenw_ssstrict_sstc_sstvala_sstvecd_ssu64xl_svadu_svvptc"
        );
        assert_eq!(RiscvCfg::default().mmu_type().as_deref(), Some("riscv,sv57"));
    }

    #[test]
    fn implemented_extensions_imply_implemented_ones() {
        for r in MISA_RULES.iter().chain(MULTI_RULES) {
            if r.misa == 0 && !ext(r.name).ruvm {
                continue;
            }
            for n in r.implied {
                assert!(ext(n).ruvm, "{} implies {n}, which this port does not have", r.name);
            }
        }
    }

    #[test]
    fn vendor_models_match_qemu() {
        assert_eq!(isa("sifive-u54", &[]), "rv64imafdc_zicntr_zicsr_zifencei_zihpm_sdtrig");
        assert_eq!(isa("shakti-c", &[]), "rv64imafdc_zicntr_zicsr_zifencei_zihpm_sdtrig");
        assert_eq!(isa("sifive-e51", &[]), "rv64imac_zicntr_zicsr_zifencei_zihpm_sdtrig");
        assert_eq!(RiscvCfg::model("sifive-u54").mmu_type().as_deref(), Some("riscv,sv39"));
        assert_eq!(RiscvCfg::model("sifive-e51").mmu_type().as_deref(), Some("riscv,none"));
        let mut b = CpuBuilder::new("sifive-u54").unwrap();
        assert_eq!(
            b.set("v", "on"),
            Err(PropError::Invalid("'sifive-u54' CPU does not allow enabling extensions".into()))
        );
    }

    #[test]
    fn bare_models_match_qemu() {
        assert_eq!(isa("rv64i", &[]), "rv64i_sdtrig");
        assert_eq!(isa("rv64e", &[]), "rv64e_sdtrig");
        let mut warn = Vec::new();
        let cfg = CpuBuilder::new("rv64i").unwrap().finalize(0, &mut warn).unwrap();
        assert_eq!(cfg.mmu_type().as_deref(), Some("riscv,none"));
        assert_eq!(warn, vec!["No satp mode set. Defaulting to 'bare'".to_string()]);
    }

    #[test]
    fn implied_rules() {
        let cfg = {
            let mut b = CpuBuilder::new("rv64").unwrap();
            b.set("v", "on").unwrap();
            b.finalize(0, &mut Vec::new()).unwrap()
        };
        assert!(cfg.ext_zve64d && cfg.ext_zve64f && cfg.ext_zve32f && cfg.ext_zve32x);
        // A property the user set keeps its value.
        let mut b = CpuBuilder::new("rv64").unwrap();
        b.set("zvfh", "on").unwrap();
        b.set("zfhmin", "off").unwrap();
        assert_eq!(
            b.finalize(0, &mut Vec::new()).unwrap_err(),
            "Zvfh extensions requires Zfhmin extension".to_string()
        );
    }

    #[test]
    fn satp_properties() {
        let mut b = CpuBuilder::new("rv64").unwrap();
        b.set("sv48", "off").unwrap();
        assert_eq!(b.finalize(0, &mut Vec::new()).unwrap().max_satp_mode, VM_SV39);
        let mut b = CpuBuilder::new("rv64").unwrap();
        b.set("sv48", "on").unwrap();
        assert_eq!(b.finalize(0, &mut Vec::new()).unwrap().max_satp_mode, VM_SV48);
        let mut b = CpuBuilder::new("sifive-u54").unwrap();
        b.set("sv48", "on").unwrap();
        assert_eq!(
            b.finalize(0, &mut Vec::new()).unwrap_err(),
            "satp_mode sv48 is higher than hw max capability sv39".to_string()
        );
    }

    #[test]
    fn sizes_and_counters() {
        let fin = |props: &[(&str, &str)]| {
            let mut b = CpuBuilder::new("rv64").unwrap();
            for (p, v) in props {
                b.set(p, v).unwrap();
            }
            let mut warn = Vec::new();
            (b.finalize(0, &mut warn), warn)
        };
        // pmu-num warns for each hart and sets pmu-mask.
        let (cfg, warn) = fin(&[("pmu-num", "3")]);
        assert_eq!(cfg.unwrap().pmu_mask, 0x38);
        assert_eq!(warn, vec!["\"pmu-num\" property is deprecated; use \"pmu-mask\"".to_string()]);
        // A VLEN or a block size only matters with its extension.
        assert!(fin(&[("vlen", "256")]).0.is_ok());
        assert_eq!(
            fin(&[("vlen", "256"), ("v", "on")]).0.unwrap_err(),
            "CPU property vlen=256 is not supported by ruvm yet"
        );
        assert_eq!(
            fin(&[("vlen", "64"), ("v", "on")]).0.unwrap_err(),
            "Vector extension implementation only supports VLEN in the range [128, 1024]"
        );
        let cfg = fin(&[("cbom_blocksize", "128")]).0.unwrap();
        assert!(cfg.cbom_blocksize == 128 && !cfg.ext_zic64b);
        assert!(fin(&[("cboz_blocksize", "100"), ("zicboz", "off")]).0.is_ok());
        assert!(fin(&[("cboz_blocksize", "100")]).0.is_err());
        // A vendor CPU keeps its values, and says which they are.
        let mut b = CpuBuilder::new("sifive-u54").unwrap();
        assert_eq!(
            b.set("pmu-num", "3"),
            Err(PropError::Hinted(
                "CPU 'sifive-u54' does not allow changing the value of 'pmu-num'".into(),
                "Current 'pmu-num' val: 16\n".into()
            ))
        );
    }

    #[test]
    fn priv_spec_disables_newer_extensions() {
        let mut b = CpuBuilder::new("rv64").unwrap();
        b.set("h", "off").unwrap();
        b.set("priv_spec", "v1.11.0").unwrap();
        let mut warn = Vec::new();
        let cfg = b.finalize(3, &mut warn).unwrap();
        assert!(!cfg.ext_zba && !cfg.ext_sstc && cfg.ext_zicntr);
        assert!(warn.contains(
            &"disabling zba extension for hart 0x3 because privilege spec version does not match"
                .to_string()
        ));
    }
}
