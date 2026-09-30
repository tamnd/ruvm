// SPDX-License-Identifier: GPL-2.0-or-later

//! SMBIOS tables for x86 machines, ported from hw/smbios/smbios.c and the SMBIOS half of
//! hw/i386/fw_cfg.c.
//!
//! QEMU builds the tables once the machine is done and hands them to the firmware as two fw_cfg
//! files: `etc/smbios/smbios-tables` holds the structures back to back and
//! `etc/smbios/smbios-anchor` holds the entry point. SeaBIOS and OVMF copy the tables somewhere
//! in guest memory, then fill in the table address and both checksums, which is why the anchor
//! QEMU produces has them all zero.
//!
//! QEMU keeps the `-smbios` settings in file scope globals. Here they live in [`SmbiosOptions`],
//! which the command line code fills one option string at a time with [`SmbiosOptions::add`].
//! Everything the machine knows (defaults, CPU topology and ID, RAM size, the RAM ranges of the
//! e820 map) goes in [`SmbiosConfig`], and [`smbios_get_tables`] turns the two into bytes.
//!
//! Only the fw_cfg file interface is here. The legacy `FW_CFG_SMBIOS_ENTRIES` blob used by old
//! pc machine types is not, and neither is the type 38 (IPMI) table, which QEMU only emits when
//! an IPMI device exists.

use std::fmt;
use std::io;

use crate::e820::{E820_RAM, E820Entry};

/// The fw_cfg file with the SMBIOS structures.
pub const SMBIOS_TABLES_FILE: &str = "etc/smbios/smbios-tables";
/// The fw_cfg file with the SMBIOS entry point.
pub const SMBIOS_ANCHOR_FILE: &str = "etc/smbios/smbios-anchor";

/// Highest structure type the `-smbios` option accepts.
pub const SMBIOS_MAX_TYPE: u8 = 127;

/// `mc->smbios_memory_device_size` for current machine types (set in hw/core/machine.c).
/// Old q35 and pc machine types used 2047 TiB and, before that, 16 GiB.
pub const DEFAULT_MEMORY_DEVICE_SIZE: u64 = 2 << 40;

/// Manufacturer string passed to `smbios_set_defaults` by every x86 pc machine.
pub const Q35_MANUFACTURER: &str = "QEMU";
/// `mc->desc` of the q35 machine, used as the product name.
pub const Q35_PRODUCT: &str = "Standard PC (Q35 + ICH9, 2009)";
/// `mc->name` of the current q35 machine type, used as the version string.
pub const Q35_VERSION: &str = "pc-q35-11.1";

/// Size of the SMBIOS 2.1 (32-bit) entry point.
pub const SMBIOS_21_ENTRY_POINT_LEN: usize = 31;
/// Size of the SMBIOS 3.0 (64-bit) entry point.
pub const SMBIOS_30_ENTRY_POINT_LEN: usize = 24;

// The structure table length field of the 2.1 entry point is 16 bits wide.
const SMBIOS_21_MAX_TABLES_LEN: usize = 0xffff;

// SVVP wants both speeds set and nonzero, QEMU has always used 2000 MHz.
const DEFAULT_CPU_SPEED: u64 = 2000;

const T0_BASE: u32 = 0x000;
const T1_BASE: u32 = 0x100;
const T2_BASE: u32 = 0x200;
const T3_BASE: u32 = 0x300;
const T4_BASE: u32 = 0x400;
const T9_BASE: u32 = 0x900;
const T11_BASE: u32 = 0xe00;
const T16_BASE: u32 = 0x1000;
const T17_BASE: u32 = 0x1100;
const T19_BASE: u32 = 0x1300;
const T32_BASE: u32 = 0x2000;
const T41_BASE: u32 = 0x2900;
const T127_BASE: u32 = 0x7f00;

const TYPE_4_LEN_V28: usize = 42;
const TYPE_4_LEN_V30: usize = 48;

const MAX_T16_STD_SZ: u64 = 0x8000_0000;
const MAX_T17_STD_SZ: u64 = 0x7fff;
const MAX_T17_EXT_SZ: u64 = 0x8000_0000;

const KIB: u64 = 1024;
const MIB: u64 = 1024 * 1024;

/// An error from parsing a `-smbios` option or from building the tables. The message is the
/// text QEMU passes to `error_setg`, without the program name prefix `error_report` adds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmbiosError {
    message: String,
}

impl SmbiosError {
    fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }

    /// The error text.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for SmbiosError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SmbiosError {}

/// Which entry point to build, the `smbios-entry-point-type` machine property.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SmbiosEntryPointType {
    /// SMBIOS 2.1 `_SM_` entry point, version 2.8 tables.
    Ep32,
    /// SMBIOS 3.0 `_SM3_` entry point.
    Ep64,
    /// Try 32-bit first and fall back to 64-bit if the tables do not fit. This is the default
    /// for current q35 and pc machine types.
    Auto,
}

impl SmbiosEntryPointType {
    /// Parses the QAPI names `32`, `64` and `auto`.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "32" => Some(Self::Ep32),
            "64" => Some(Self::Ep64),
            "auto" => Some(Self::Auto),
            _ => None,
        }
    }
}

/// One RAM range for a type 19 table, `struct smbios_phys_mem_area`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SmbiosPhysMemArea {
    /// Start of the range.
    pub address: u64,
    /// Length in bytes.
    pub length: u64,
}

/// The RAM entries of an e820 table in table order, which is what `fw_cfg_build_smbios` passes
/// as `mem_array`. Pass the final table, the one published as `etc/e820`.
pub fn mem_array_from_e820(entries: &[E820Entry]) -> Vec<SmbiosPhysMemArea> {
    entries
        .iter()
        .filter(|e| e.kind == E820_RAM)
        .map(|e| SmbiosPhysMemArea { address: e.address, length: e.length })
        .collect()
}

/// The strings `smbios_set_defaults` fills in for any field the user left unset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmbiosDefaults {
    /// Manufacturer for types 1, 2, 3, 4 and 17.
    pub manufacturer: String,
    /// Product name for types 1 and 2, the machine description.
    pub product: String,
    /// Version for types 1, 2, 3 and 4, the machine type name.
    pub version: String,
}

impl SmbiosDefaults {
    /// Defaults from explicit strings.
    pub fn new(manufacturer: &str, product: &str, version: &str) -> Self {
        Self {
            manufacturer: manufacturer.to_owned(),
            product: product.to_owned(),
            version: version.to_owned(),
        }
    }

    /// What `pc-q35-11.1` passes.
    pub fn q35() -> Self {
        Self::new(Q35_MANUFACTURER, Q35_PRODUCT, Q35_VERSION)
    }
}

/// The parts of `ms->smp` that the type 4 tables use. The counts are the full topology from
/// `-smp`, including `maxcpus`, not just the CPUs present at boot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SmbiosTopology {
    /// Number of sockets; one type 4 table is built per socket.
    pub sockets: u32,
    /// Dies per socket.
    pub dies: u32,
    /// Clusters per die.
    pub clusters: u32,
    /// Modules per cluster.
    pub modules: u32,
    /// Cores per module.
    pub cores: u32,
    /// Threads per core.
    pub threads: u32,
}

impl Default for SmbiosTopology {
    fn default() -> Self {
        Self { sockets: 1, dies: 1, clusters: 1, modules: 1, cores: 1, threads: 1 }
    }
}

impl SmbiosTopology {
    /// `machine_topo_get_cores_per_socket`.
    pub fn cores_per_socket(&self) -> u32 {
        self.cores.wrapping_mul(self.modules).wrapping_mul(self.clusters).wrapping_mul(self.dies)
    }

    /// `machine_topo_get_threads_per_socket`.
    pub fn threads_per_socket(&self) -> u32 {
        self.threads.wrapping_mul(self.cores_per_socket())
    }
}

/// Where a PCI device sits, for `-smbios type=41,pcidev=<id>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmbiosPciDevice {
    /// The device's `id=`.
    pub id: String,
    /// Bus number.
    pub bus: u8,
    /// Device and function, `PCIDevice::devfn`.
    pub devfn: u8,
    /// True when the device is on a root bus. QEMU refuses devices behind bridges.
    pub on_root_bus: bool,
}

/// What the machine contributes to the tables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmbiosConfig {
    /// The `smbios_set_defaults` strings, or `None` for a machine with
    /// `pcmc->smbios_defaults` off. Without defaults, only the types the user gave fields for
    /// are built.
    pub defaults: Option<SmbiosDefaults>,
    /// The `-uuid` value in RFC 4122 (big endian) byte order. A `uuid=` in `-smbios type=1`
    /// takes precedence, see [`SmbiosOptions::uuid`].
    pub uuid: Option<[u8; 16]>,
    /// The `smbios-entry-point-type` machine property.
    pub ep_type: SmbiosEntryPointType,
    /// CPU topology.
    pub topology: SmbiosTopology,
    /// `env->cpuid_version` of the first possible CPU, CPUID leaf 1 EAX.
    pub cpuid_version: u32,
    /// `env->features[FEAT_1_EDX]` of the first possible CPU, CPUID leaf 1 EDX.
    pub cpuid_features: u32,
    /// `ms->ram_size`.
    pub ram_size: u64,
    /// `mc->smbios_memory_device_size`, the largest DIMM a type 17 table describes.
    pub memory_device_size: u64,
    /// The RAM ranges for the type 19 tables, see [`mem_array_from_e820`].
    pub mem_array: Vec<SmbiosPhysMemArea>,
    /// PCI devices that `type=41,pcidev=` may name.
    pub pci_devices: Vec<SmbiosPciDevice>,
}

impl SmbiosConfig {
    /// A q35 configuration with one CPU, no RAM and the `auto` entry point. The caller fills in
    /// the rest.
    pub fn q35() -> Self {
        Self {
            defaults: Some(SmbiosDefaults::q35()),
            uuid: None,
            ep_type: SmbiosEntryPointType::Auto,
            topology: SmbiosTopology::default(),
            cpuid_version: 0,
            cpuid_features: 0,
            ram_size: 0,
            memory_device_size: DEFAULT_MEMORY_DEVICE_SIZE,
            mem_array: Vec::new(),
            pci_devices: Vec::new(),
        }
    }
}

/// `-smbios type=0` fields.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SmbiosType0 {
    /// `vendor=`.
    pub vendor: Option<String>,
    /// `version=`.
    pub version: Option<String>,
    /// `date=`.
    pub date: Option<String>,
    /// `release=major.minor`, if given.
    pub release: Option<(u8, u8)>,
    /// `uefi=`, off by default.
    pub uefi: bool,
    /// `vm=`, on by default.
    pub vm: bool,
}

/// `-smbios type=1` fields. The UUID is kept in [`SmbiosOptions::uuid`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SmbiosType1 {
    /// `manufacturer=`.
    pub manufacturer: Option<String>,
    /// `product=`.
    pub product: Option<String>,
    /// `version=`.
    pub version: Option<String>,
    /// `serial=`.
    pub serial: Option<String>,
    /// `sku=`.
    pub sku: Option<String>,
    /// `family=`.
    pub family: Option<String>,
}

/// `-smbios type=2` fields.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SmbiosType2 {
    /// `manufacturer=`.
    pub manufacturer: Option<String>,
    /// `product=`.
    pub product: Option<String>,
    /// `version=`.
    pub version: Option<String>,
    /// `serial=`.
    pub serial: Option<String>,
    /// `asset=`.
    pub asset: Option<String>,
    /// `location=`.
    pub location: Option<String>,
}

/// `-smbios type=3` fields.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SmbiosType3 {
    /// `manufacturer=`.
    pub manufacturer: Option<String>,
    /// `version=`.
    pub version: Option<String>,
    /// `serial=`.
    pub serial: Option<String>,
    /// `asset=`.
    pub asset: Option<String>,
    /// `sku=`.
    pub sku: Option<String>,
}

/// `-smbios type=4` fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmbiosType4 {
    /// `processor-family=`, 0x01 ("Other") by default.
    pub processor_family: u16,
    /// `sock_pfx=`.
    pub sock_pfx: Option<String>,
    /// `manufacturer=`.
    pub manufacturer: Option<String>,
    /// `version=`.
    pub version: Option<String>,
    /// `serial=`.
    pub serial: Option<String>,
    /// `asset=`.
    pub asset: Option<String>,
    /// `part=`.
    pub part: Option<String>,
    /// `max-speed=` in MHz.
    pub max_speed: u64,
    /// `current-speed=` in MHz.
    pub current_speed: u64,
    /// `processor-id=`. Zero means "use the CPUID values from [`SmbiosConfig`]".
    pub processor_id: u64,
}

impl Default for SmbiosType4 {
    fn default() -> Self {
        Self {
            processor_family: 0x01,
            sock_pfx: None,
            manufacturer: None,
            version: None,
            serial: None,
            asset: None,
            part: None,
            max_speed: DEFAULT_CPU_SPEED,
            current_speed: DEFAULT_CPU_SPEED,
            processor_id: 0,
        }
    }
}

/// One `-smbios type=8` instance.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SmbiosType8 {
    /// `internal_reference=`.
    pub internal_reference: Option<String>,
    /// `external_reference=`.
    pub external_reference: Option<String>,
    /// `connector_type=`.
    pub connector_type: u8,
    /// `port_type=`.
    pub port_type: u8,
}

/// One `-smbios type=9` instance.
///
/// QEMU's option table for type 9 lists `pci_device` while the code reads `pcidev`, so the PCI
/// location is never filled in and the bus, device and segment fields are always 0xff. This
/// keeps that behaviour: `pci_device=` is accepted and ignored.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SmbiosType9 {
    /// `slot_designation=`.
    pub slot_designation: Option<String>,
    /// `slot_type=`.
    pub slot_type: u8,
    /// `slot_data_bus_width=`.
    pub slot_data_bus_width: u8,
    /// `current_usage=`.
    pub current_usage: u8,
    /// `slot_length=`.
    pub slot_length: u8,
    /// `slot_id=`.
    pub slot_id: u16,
    /// `slot_characteristics1=`.
    pub slot_characteristics1: u8,
    /// `slot_characteristics2=`.
    pub slot_characteristics2: u8,
}

/// `-smbios type=17` fields.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SmbiosType17 {
    /// `loc_pfx=`.
    pub loc_pfx: Option<String>,
    /// `bank=`.
    pub bank: Option<String>,
    /// `manufacturer=`.
    pub manufacturer: Option<String>,
    /// `serial=`.
    pub serial: Option<String>,
    /// `asset=`.
    pub asset: Option<String>,
    /// `part=`.
    pub part: Option<String>,
    /// `speed=` in MT/s.
    pub speed: u16,
}

/// One `-smbios type=41` instance.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SmbiosType41 {
    /// `designation=`.
    pub designation: Option<String>,
    /// Device type with the "enabled" bit (0x80) set.
    pub kind: u8,
    /// `instance=`, 1 by default.
    pub instance: u8,
    /// `pcidev=`.
    pub pcidev: Option<String>,
}

const TYPE41_KINDS: [&str; 10] =
    ["other", "unknown", "video", "scsi", "ethernet", "tokenring", "sound", "pata", "sata", "sas"];

/// Everything the `-smbios` options set, accumulated over all of them in command line order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmbiosOptions {
    /// Type 0 fields.
    pub type0: SmbiosType0,
    /// Type 1 fields.
    pub type1: SmbiosType1,
    /// `uuid=` from `type=1`, RFC 4122 byte order. In QEMU this and `-uuid` write the same
    /// global, so the later of the two wins; [`smbios_get_tables`] prefers this one when set.
    pub uuid: Option<[u8; 16]>,
    /// Type 2 fields.
    pub type2: SmbiosType2,
    /// Type 3 fields.
    pub type3: SmbiosType3,
    /// Type 4 fields.
    pub type4: SmbiosType4,
    /// Type 8 instances.
    pub type8: Vec<SmbiosType8>,
    /// Type 9 instances.
    pub type9: Vec<SmbiosType9>,
    /// Type 11 OEM strings, from `value=` and the contents of `path=` files.
    pub type11: Vec<Vec<u8>>,
    /// Type 17 fields.
    pub type17: SmbiosType17,
    /// Type 41 instances.
    pub type41: Vec<SmbiosType41>,
    usr_blobs: Vec<u8>,
    usr_table_max: usize,
    usr_table_cnt: usize,
    have_binfile: [bool; SMBIOS_MAX_TYPE as usize + 1],
    have_fields: [bool; SMBIOS_MAX_TYPE as usize + 1],
}

impl Default for SmbiosOptions {
    fn default() -> Self {
        Self {
            type0: SmbiosType0 { vm: true, ..SmbiosType0::default() },
            type1: SmbiosType1::default(),
            uuid: None,
            type2: SmbiosType2::default(),
            type3: SmbiosType3::default(),
            type4: SmbiosType4::default(),
            type8: Vec::new(),
            type9: Vec::new(),
            type11: Vec::new(),
            type17: SmbiosType17::default(),
            type41: Vec::new(),
            usr_blobs: Vec::new(),
            usr_table_max: 0,
            usr_table_cnt: 0,
            have_binfile: [false; SMBIOS_MAX_TYPE as usize + 1],
            have_fields: [false; SMBIOS_MAX_TYPE as usize + 1],
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OptKind {
    Str,
    Bool,
    Num,
}

use OptKind::{Bool, Num, Str};

const FILE_OPTS: &[(&str, OptKind)] = &[("file", Str)];
const TYPE0_OPTS: &[(&str, OptKind)] = &[
    ("type", Num),
    ("vendor", Str),
    ("version", Str),
    ("date", Str),
    ("release", Str),
    ("uefi", Bool),
    ("vm", Bool),
];
const TYPE1_OPTS: &[(&str, OptKind)] = &[
    ("type", Num),
    ("manufacturer", Str),
    ("product", Str),
    ("version", Str),
    ("serial", Str),
    ("uuid", Str),
    ("sku", Str),
    ("family", Str),
];
const TYPE2_OPTS: &[(&str, OptKind)] = &[
    ("type", Num),
    ("manufacturer", Str),
    ("product", Str),
    ("version", Str),
    ("serial", Str),
    ("asset", Str),
    ("location", Str),
];
const TYPE3_OPTS: &[(&str, OptKind)] = &[
    ("type", Num),
    ("manufacturer", Str),
    ("version", Str),
    ("serial", Str),
    ("asset", Str),
    ("sku", Str),
];
const TYPE4_OPTS: &[(&str, OptKind)] = &[
    ("type", Num),
    ("sock_pfx", Str),
    ("manufacturer", Str),
    ("version", Str),
    ("max-speed", Num),
    ("current-speed", Num),
    ("serial", Str),
    ("asset", Str),
    ("part", Str),
    ("processor-family", Num),
    ("processor-id", Num),
];
const TYPE8_OPTS: &[(&str, OptKind)] = &[
    ("type", Num),
    ("internal_reference", Str),
    ("external_reference", Str),
    ("connector_type", Num),
    ("port_type", Num),
];
const TYPE9_OPTS: &[(&str, OptKind)] = &[
    ("type", Num),
    ("slot_designation", Str),
    ("slot_type", Num),
    ("slot_data_bus_width", Num),
    ("current_usage", Num),
    ("slot_length", Num),
    ("slot_id", Num),
    ("slot_characteristics1", Num),
    ("slot_characteristics2", Num),
    ("pci_device", Str),
];
const TYPE11_OPTS: &[(&str, OptKind)] = &[("type", Num), ("value", Str), ("path", Str)];
const TYPE17_OPTS: &[(&str, OptKind)] = &[
    ("type", Num),
    ("loc_pfx", Str),
    ("bank", Str),
    ("manufacturer", Str),
    ("serial", Str),
    ("asset", Str),
    ("part", Str),
    ("speed", Num),
];
const TYPE41_OPTS: &[(&str, OptKind)] =
    &[("type", Num), ("designation", Str), ("kind", Str), ("instance", Num), ("pcidev", Str)];

/// The `name=value` pairs of one option string, in order, as QemuOpts stores them.
struct Opts {
    list: Vec<(String, String)>,
}

impl Opts {
    /// Splits an option string the way `opts_do_parse` does: `,,` is a literal comma inside a
    /// value, a bare `name` means `name=on`, `noname` means `name=off`, and `id=` is dropped.
    fn parse(params: &str) -> Self {
        let mut list = Vec::new();
        let mut rest = params;
        while !rest.is_empty() {
            let len = rest.find(['=', ',']).unwrap_or(rest.len());
            let (name, value);
            if rest.as_bytes().get(len) != Some(&b'=') {
                let flag = &rest[..len];
                if let Some(stripped) = flag.strip_prefix("no") {
                    name = stripped.to_owned();
                    value = "off".to_owned();
                } else {
                    name = flag.to_owned();
                    value = "on".to_owned();
                }
                rest = &rest[len..];
            } else {
                name = rest[..len].to_owned();
                rest = &rest[len + 1..];
                let mut v = String::new();
                loop {
                    match rest.find(',') {
                        None => {
                            v.push_str(rest);
                            rest = "";
                            break;
                        }
                        Some(i) => {
                            v.push_str(&rest[..i]);
                            if rest[i + 1..].starts_with(',') {
                                v.push(',');
                                rest = &rest[i + 2..];
                            } else {
                                rest = &rest[i..];
                                break;
                            }
                        }
                    }
                }
                value = v;
            }
            if let Some(r) = rest.strip_prefix(',') {
                rest = r;
            }
            if name != "id" {
                list.push((name, value));
            }
        }
        Self { list }
    }

    /// `qemu_opt_get`: the last value given for `name`.
    fn get(&self, name: &str) -> Option<&str> {
        self.list.iter().rev().find(|(n, _)| n == name).map(|(_, v)| v.as_str())
    }

    fn get_string(&self, name: &str) -> Option<String> {
        self.get(name).map(str::to_owned)
    }

    /// `save_opt`: overwrite `dest` only when the option is present.
    fn save(&self, dest: &mut Option<String>, name: &str) {
        if let Some(v) = self.get(name) {
            *dest = Some(v.to_owned());
        }
    }

    /// `qemu_opts_validate`: every option must be in `desc` and parse as its type.
    fn validate(&self, desc: &[(&str, OptKind)]) -> Result<(), SmbiosError> {
        for (name, value) in &self.list {
            let Some(&(_, kind)) = desc.iter().find(|(n, _)| n == name) else {
                return Err(SmbiosError::new(format!("Invalid parameter '{name}'")));
            };
            match kind {
                Str => {}
                Bool => {
                    parse_bool(name, value)?;
                }
                Num => {
                    parse_number(name, value)?;
                }
            }
        }
        Ok(())
    }

    /// `qemu_opt_get_bool` after validation.
    fn get_bool(&self, name: &str, def: bool) -> bool {
        self.get(name).and_then(|v| parse_bool(name, v).ok()).unwrap_or(def)
    }

    /// `qemu_opt_get_number` after validation.
    fn get_number(&self, name: &str, def: u64) -> u64 {
        self.get(name).and_then(|v| parse_number(name, v).ok()).unwrap_or(def)
    }
}

/// `qapi_bool_parse`.
fn parse_bool(name: &str, value: &str) -> Result<bool, SmbiosError> {
    match value {
        "on" | "yes" | "true" | "y" => Ok(true),
        "off" | "no" | "false" | "n" => Ok(false),
        _ => Err(SmbiosError::new(format!("Parameter '{name}' expects 'on' or 'off'"))),
    }
}

/// The result of C `strtoull(s, &end, 0)`: the value, how many bytes were consumed (zero when
/// nothing was converted) and whether it overflowed.
fn strtoull(s: &str) -> (u64, usize, bool) {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r') {
        i += 1;
    }
    let mut negative = false;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        negative = b[i] == b'-';
        i += 1;
    }
    let mut radix = 10;
    if i < b.len() && b[i] == b'0' {
        if matches!(b.get(i + 1), Some(b'x' | b'X'))
            && b.get(i + 2).is_some_and(u8::is_ascii_hexdigit)
        {
            radix = 16;
            i += 2;
        } else {
            radix = 8;
        }
    }
    let start = i;
    let mut value: u64 = 0;
    let mut overflow = false;
    while i < b.len() {
        let Some(d) = (b[i] as char).to_digit(radix) else {
            break;
        };
        match value.checked_mul(u64::from(radix)).and_then(|v| v.checked_add(u64::from(d))) {
            Some(v) => value = v,
            None => overflow = true,
        }
        i += 1;
    }
    if i == start {
        return (0, 0, false);
    }
    if overflow {
        return (u64::MAX, i, true);
    }
    (if negative { value.wrapping_neg() } else { value }, i, false)
}

/// `parse_option_number`, which is `qemu_strtou64(value, NULL, 0)`.
fn parse_number(name: &str, value: &str) -> Result<u64, SmbiosError> {
    let (v, end, overflow) = strtoull(value);
    if end == 0 || end != value.len() {
        return Err(SmbiosError::new(format!("Parameter '{name}' expects a number")));
    }
    if overflow {
        return Err(SmbiosError::new(format!(
            "Value '{value}' is too large for parameter '{name}'"
        )));
    }
    Ok(v)
}

/// One `%hhu` conversion of `sscanf`: returns the value truncated to a byte and the rest.
fn scan_hhu(s: &str) -> Option<(u8, &str)> {
    let (v, end, _) = strtoull_base10(s);
    if end == 0 {
        return None;
    }
    Some((v as u8, &s[end..]))
}

fn strtoull_base10(s: &str) -> (u64, usize, bool) {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r') {
        i += 1;
    }
    let mut negative = false;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        negative = b[i] == b'-';
        i += 1;
    }
    let start = i;
    let mut value: u64 = 0;
    let mut overflow = false;
    while i < b.len() && b[i].is_ascii_digit() {
        match value.checked_mul(10).and_then(|v| v.checked_add(u64::from(b[i] - b'0'))) {
            Some(v) => value = v,
            None => overflow = true,
        }
        i += 1;
    }
    if i == start {
        return (0, 0, false);
    }
    if overflow {
        return (u64::MAX, i, true);
    }
    (if negative { value.wrapping_neg() } else { value }, i, false)
}

/// `sscanf(val, "%hhu.%hhu", ...) == 2`.
fn parse_release(s: &str) -> Option<(u8, u8)> {
    let (major, rest) = scan_hhu(s)?;
    let rest = rest.strip_prefix('.')?;
    let (minor, _) = scan_hhu(rest)?;
    Some((major, minor))
}

/// Parses a UUID in the `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` form `qemu_uuid_parse` accepts
/// and returns its bytes in RFC 4122 order.
pub fn parse_uuid(s: &str) -> Option<[u8; 16]> {
    let b = s.as_bytes();
    if b.len() != 36 {
        return None;
    }
    let mut out = [0u8; 16];
    let mut n = 0;
    let mut i = 0;
    while i < 36 {
        if i == 8 || i == 13 || i == 18 || i == 23 {
            if b[i] != b'-' {
                return None;
            }
            i += 1;
            continue;
        }
        let hi = (b[i] as char).to_digit(16)?;
        let lo = (b[i + 1] as char).to_digit(16)?;
        out[n] = (hi << 4 | lo) as u8;
        n += 1;
        i += 2;
    }
    Some(out)
}

/// Turns an I/O error into the strerror text QEMU prints, dropping the "(os error N)" that the
/// Rust formatting adds.
fn strerror(e: &io::Error) -> String {
    let s = e.to_string();
    match s.rfind(" (os error ") {
        Some(i) => s[..i].to_owned(),
        None => s,
    }
}

impl SmbiosOptions {
    /// No options given.
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies one `-smbios` option string, like `smbios_entry_add`. Files named by `file=`
    /// and `path=` are read from the host file system.
    pub fn add(&mut self, arg: &str) -> Result<(), SmbiosError> {
        self.add_with_reader(arg, |p| std::fs::read(p))
    }

    /// [`SmbiosOptions::add`] with a caller supplied file reader, for tests and sandboxes.
    pub fn add_with_reader<F>(&mut self, arg: &str, mut read: F) -> Result<(), SmbiosError>
    where
        F: FnMut(&str) -> io::Result<Vec<u8>>,
    {
        let opts = Opts::parse(arg);

        if let Some(file) = opts.get("file") {
            opts.validate(FILE_OPTS)?;
            let blob = match read(file) {
                Ok(b) if b.len() >= 4 => b,
                _ => return Err(SmbiosError::new(format!("Cannot read SMBIOS file {file}"))),
            };
            let ty = blob[0];
            if ty <= SMBIOS_MAX_TYPE {
                if self.have_fields[usize::from(ty)] {
                    return Err(SmbiosError::new(format!(
                        "can't load type {ty} struct, fields already specified!"
                    )));
                }
                self.have_binfile[usize::from(ty)] = true;
            }
            self.usr_table_max = self.usr_table_max.max(blob.len());
            self.usr_blobs.extend_from_slice(&blob);
            self.usr_table_cnt += 1;
            return Ok(());
        }

        let Some(type_str) = opts.get("type") else {
            return Err(SmbiosError::new("Must specify type= or file="));
        };
        // Plain strtoul here, so garbage reads as type 0 and is caught by the validation below.
        let (ty, _, _) = strtoull(type_str);
        if ty > u64::from(SMBIOS_MAX_TYPE) {
            return Err(SmbiosError::new("out of range!"));
        }
        let ty = ty as usize;
        if self.have_binfile[ty] {
            return Err(SmbiosError::new("can't add fields, binary file already loaded!"));
        }
        self.have_fields[ty] = true;

        match ty {
            0 => {
                opts.validate(TYPE0_OPTS)?;
                let t = &mut self.type0;
                opts.save(&mut t.vendor, "vendor");
                opts.save(&mut t.version, "version");
                opts.save(&mut t.date, "date");
                t.uefi = opts.get_bool("uefi", false);
                t.vm = opts.get_bool("vm", true);
                if let Some(v) = opts.get("release") {
                    match parse_release(v) {
                        Some(r) => t.release = Some(r),
                        None => return Err(SmbiosError::new("Invalid release")),
                    }
                }
            }
            1 => {
                opts.validate(TYPE1_OPTS)?;
                let t = &mut self.type1;
                opts.save(&mut t.manufacturer, "manufacturer");
                opts.save(&mut t.product, "product");
                opts.save(&mut t.version, "version");
                opts.save(&mut t.serial, "serial");
                opts.save(&mut t.sku, "sku");
                opts.save(&mut t.family, "family");
                if let Some(v) = opts.get("uuid") {
                    match parse_uuid(v) {
                        Some(u) => self.uuid = Some(u),
                        None => return Err(SmbiosError::new("Invalid UUID")),
                    }
                }
            }
            2 => {
                opts.validate(TYPE2_OPTS)?;
                let t = &mut self.type2;
                opts.save(&mut t.manufacturer, "manufacturer");
                opts.save(&mut t.product, "product");
                opts.save(&mut t.version, "version");
                opts.save(&mut t.serial, "serial");
                opts.save(&mut t.asset, "asset");
                opts.save(&mut t.location, "location");
            }
            3 => {
                opts.validate(TYPE3_OPTS)?;
                let t = &mut self.type3;
                opts.save(&mut t.manufacturer, "manufacturer");
                opts.save(&mut t.version, "version");
                opts.save(&mut t.serial, "serial");
                opts.save(&mut t.asset, "asset");
                opts.save(&mut t.sku, "sku");
            }
            4 => {
                opts.validate(TYPE4_OPTS)?;
                let t = &mut self.type4;
                opts.save(&mut t.sock_pfx, "sock_pfx");
                t.processor_family = opts.get_number("processor-family", 0x01) as u16;
                opts.save(&mut t.manufacturer, "manufacturer");
                opts.save(&mut t.version, "version");
                opts.save(&mut t.serial, "serial");
                opts.save(&mut t.asset, "asset");
                opts.save(&mut t.part, "part");
                t.processor_id = opts.get_number("processor-id", 0);
                t.max_speed = opts.get_number("max-speed", DEFAULT_CPU_SPEED);
                t.current_speed = opts.get_number("current-speed", DEFAULT_CPU_SPEED);
                if t.max_speed > u64::from(u16::MAX) || t.current_speed > u64::from(u16::MAX) {
                    return Err(SmbiosError::new(format!(
                        "SMBIOS CPU speed is too large (> {})",
                        u16::MAX
                    )));
                }
            }
            8 => {
                opts.validate(TYPE8_OPTS)?;
                self.type8.push(SmbiosType8 {
                    internal_reference: opts.get_string("internal_reference"),
                    external_reference: opts.get_string("external_reference"),
                    connector_type: opts.get_number("connector_type", 0) as u8,
                    port_type: opts.get_number("port_type", 0) as u8,
                });
            }
            9 => {
                opts.validate(TYPE9_OPTS)?;
                self.type9.push(SmbiosType9 {
                    slot_designation: opts.get_string("slot_designation"),
                    slot_type: opts.get_number("slot_type", 0) as u8,
                    slot_data_bus_width: opts.get_number("slot_data_bus_width", 0) as u8,
                    current_usage: opts.get_number("current_usage", 0) as u8,
                    slot_length: opts.get_number("slot_length", 0) as u8,
                    slot_id: opts.get_number("slot_id", 0) as u16,
                    slot_characteristics1: opts.get_number("slot_characteristics1", 0) as u8,
                    slot_characteristics2: opts.get_number("slot_characteristics2", 0) as u8,
                });
            }
            11 => {
                opts.validate(TYPE11_OPTS)?;
                for (name, value) in &opts.list {
                    match name.as_str() {
                        "value" => self.type11.push(value.as_bytes().to_vec()),
                        "path" => {
                            let data = read(value).map_err(|e| {
                                SmbiosError::new(format!(
                                    "Could not open '{value}': {}",
                                    strerror(&e)
                                ))
                            })?;
                            if data.contains(&0) {
                                return Err(SmbiosError::new(format!(
                                    "NUL in OEM strings value in {value}"
                                )));
                            }
                            self.type11.push(data);
                        }
                        _ => {}
                    }
                }
            }
            17 => {
                opts.validate(TYPE17_OPTS)?;
                let t = &mut self.type17;
                opts.save(&mut t.loc_pfx, "loc_pfx");
                opts.save(&mut t.bank, "bank");
                opts.save(&mut t.manufacturer, "manufacturer");
                opts.save(&mut t.serial, "serial");
                opts.save(&mut t.asset, "asset");
                opts.save(&mut t.part, "part");
                t.speed = opts.get_number("speed", 0) as u16;
            }
            41 => {
                opts.validate(TYPE41_OPTS)?;
                let kind = match opts.get("kind") {
                    None => 0,
                    Some(k) => match TYPE41_KINDS.iter().position(|n| *n == k) {
                        Some(i) => i,
                        None => {
                            return Err(SmbiosError::new(format!("invalid parameter value: {k}")));
                        }
                    },
                };
                self.type41.push(SmbiosType41 {
                    designation: opts.get_string("designation"),
                    kind: (kind as u8 + 1) | 0x80,
                    instance: opts.get_number("instance", 1) as u8,
                    pcidev: opts.get_string("pcidev"),
                });
            }
            _ => {
                return Err(SmbiosError::new(format!(
                    "Don't know how to build fields for SMBIOS type {ty}"
                )));
            }
        }
        Ok(())
    }

    /// `smbios_set_defaults`: fill the unset strings from the machine defaults.
    fn apply_defaults(&mut self, d: &SmbiosDefaults) {
        fn set(field: &mut Option<String>, value: &str) {
            if field.is_none() {
                *field = Some(value.to_owned());
            }
        }
        set(&mut self.type1.manufacturer, &d.manufacturer);
        set(&mut self.type1.product, &d.product);
        set(&mut self.type1.version, &d.version);
        set(&mut self.type2.manufacturer, &d.manufacturer);
        set(&mut self.type2.product, &d.product);
        set(&mut self.type2.version, &d.version);
        set(&mut self.type3.manufacturer, &d.manufacturer);
        set(&mut self.type3.version, &d.version);
        set(&mut self.type4.sock_pfx, "CPU");
        set(&mut self.type4.manufacturer, &d.manufacturer);
        set(&mut self.type4.version, &d.version);
        set(&mut self.type17.loc_pfx, "DIMM");
        set(&mut self.type17.manufacturer, &d.manufacturer);
    }
}

/// The two fw_cfg files and the entry point type that was actually used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmbiosTables {
    /// Contents of `etc/smbios/smbios-tables`.
    pub tables: Vec<u8>,
    /// Contents of `etc/smbios/smbios-anchor`.
    pub anchor: Vec<u8>,
    /// [`SmbiosEntryPointType::Ep32`] or [`SmbiosEntryPointType::Ep64`], never `Auto`.
    pub ep_type: SmbiosEntryPointType,
}

/// One structure under construction: the formatted area, then the strings.
struct Table {
    area: Vec<u8>,
    strings: Vec<u8>,
    str_index: u32,
}

impl Table {
    fn new(ty: u8, handle: u32, len: usize) -> Self {
        let mut area = vec![0u8; len];
        area[0] = ty;
        area[1] = len as u8;
        area[2..4].copy_from_slice(&(handle as u16).to_le_bytes());
        Self { area, strings: Vec::new(), str_index: 0 }
    }

    fn u8(&mut self, off: usize, v: u8) {
        self.area[off] = v;
    }

    fn u16(&mut self, off: usize, v: u16) {
        self.area[off..off + 2].copy_from_slice(&v.to_le_bytes());
    }

    fn u32(&mut self, off: usize, v: u32) {
        self.area[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }

    fn u64(&mut self, off: usize, v: u64) {
        self.area[off..off + 8].copy_from_slice(&v.to_le_bytes());
    }

    /// `SMBIOS_TABLE_SET_STR_LIST`: append a string if it is not empty and return its index,
    /// or 0 when nothing was added.
    fn push_str(&mut self, value: Option<&[u8]>) -> u8 {
        match value {
            Some(s) if !s.is_empty() => {
                self.strings.extend_from_slice(s);
                self.strings.push(0);
                self.str_index += 1;
                self.str_index as u8
            }
            _ => 0,
        }
    }

    /// `SMBIOS_TABLE_SET_STR`.
    fn set_str(&mut self, off: usize, value: Option<&str>) {
        let idx = self.push_str(value.map(str::as_bytes));
        self.area[off] = idx;
    }
}

/// The state `smbios_get_tables_ep` keeps in globals.
struct Builder<'a> {
    cfg: &'a SmbiosConfig,
    opts: &'a SmbiosOptions,
    ep_type: SmbiosEntryPointType,
    uuid: Option<[u8; 16]>,
    tables: Vec<u8>,
    table_max: usize,
    table_cnt: usize,
    type4_count: u32,
    type11_freed: &'a mut bool,
}

/// `snprintf` into a 128 byte buffer keeps at most 127 bytes.
fn snprintf128(s: String) -> Vec<u8> {
    let mut b = s.into_bytes();
    b.truncate(127);
    b
}

fn or_null(s: Option<&str>) -> &str {
    s.unwrap_or("(null)")
}

impl Builder<'_> {
    /// `smbios_skip_table`.
    fn skip(&self, ty: u8, required: bool) -> bool {
        let ty = usize::from(ty);
        if self.opts.have_binfile[ty] {
            return true;
        }
        if self.opts.have_fields[ty] {
            return false;
        }
        !(self.cfg.defaults.is_some() && required)
    }

    /// `SMBIOS_BUILD_TABLE_POST`.
    fn finish(&mut self, t: Table) {
        let start = self.tables.len();
        self.tables.extend_from_slice(&t.area);
        self.tables.extend_from_slice(&t.strings);
        let term = if t.str_index == 0 { 2 } else { 1 };
        self.tables.extend(std::iter::repeat_n(0u8, term));
        self.table_max = self.table_max.max(self.tables.len() - start);
        self.table_cnt += 1;
    }

    fn type0(&mut self) {
        if self.skip(0, false) {
            return;
        }
        let o = &self.opts.type0;
        let mut t = Table::new(0, T0_BASE, 24);
        t.set_str(4, o.vendor.as_deref());
        t.set_str(5, o.version.as_deref());
        t.u16(6, 0xe800);
        t.set_str(8, o.date.as_deref());
        t.u8(9, 0);
        t.u64(10, 0x08);
        let mut ext1 = 0x04;
        if o.uefi {
            ext1 |= 0x08;
        }
        if o.vm {
            ext1 |= 0x10;
        }
        t.u8(19, ext1);
        let (major, minor) = o.release.unwrap_or((0, 0));
        t.u8(20, major);
        t.u8(21, minor);
        t.u8(22, 0xff);
        t.u8(23, 0xff);
        self.finish(t);
    }

    fn type1(&mut self) {
        if self.skip(1, true) {
            return;
        }
        let o = &self.opts.type1;
        let mut t = Table::new(1, T1_BASE, 27);
        t.set_str(4, o.manufacturer.as_deref());
        t.set_str(5, o.product.as_deref());
        t.set_str(6, o.version.as_deref());
        t.set_str(7, o.serial.as_deref());
        if let Some(u) = self.uuid {
            // SMBIOS 2.6 stores the first three fields little endian.
            let mut w = u;
            w[0..4].reverse();
            w[4..6].reverse();
            w[6..8].reverse();
            t.area[8..24].copy_from_slice(&w);
        }
        t.u8(24, 0x06);
        t.set_str(25, o.sku.as_deref());
        t.set_str(26, o.family.as_deref());
        self.finish(t);
    }

    fn type2(&mut self) {
        if self.skip(2, false) {
            return;
        }
        let o = &self.opts.type2;
        let mut t = Table::new(2, T2_BASE, 15);
        t.set_str(4, o.manufacturer.as_deref());
        t.set_str(5, o.product.as_deref());
        t.set_str(6, o.version.as_deref());
        t.set_str(7, o.serial.as_deref());
        t.set_str(8, o.asset.as_deref());
        t.u8(9, 0x01);
        t.set_str(10, o.location.as_deref());
        t.u16(11, 0x300);
        t.u8(13, 0x0a);
        t.u8(14, 0);
        self.finish(t);
    }

    fn type3(&mut self) {
        if self.skip(3, true) {
            return;
        }
        let o = &self.opts.type3;
        let mut t = Table::new(3, T3_BASE, 22);
        t.set_str(4, o.manufacturer.as_deref());
        t.u8(5, 0x01);
        t.set_str(6, o.version.as_deref());
        t.set_str(7, o.serial.as_deref());
        t.set_str(8, o.asset.as_deref());
        t.u8(9, 0x03);
        t.u8(10, 0x03);
        t.u8(11, 0x03);
        t.u8(12, 0x02);
        t.set_str(21, o.sku.as_deref());
        self.finish(t);
    }

    fn type4(&mut self, instance: u32) -> Result<(), SmbiosError> {
        if self.skip(4, true) {
            return Ok(());
        }
        let o = &self.opts.type4;
        let v30 = self.ep_type == SmbiosEntryPointType::Ep64;
        let len = if v30 { TYPE_4_LEN_V30 } else { TYPE_4_LEN_V28 };
        let mut t = Table::new(4, T4_BASE + instance, len);

        let sock = snprintf128(format!("{}{:2x}", or_null(o.sock_pfx.as_deref()), instance));
        let idx = t.push_str(Some(&sock));
        t.u8(4, idx);
        t.u8(5, 0x03);
        t.u8(6, 0xfe);
        t.set_str(7, o.manufacturer.as_deref());
        if o.processor_id == 0 {
            t.u32(8, self.cfg.cpuid_version);
            t.u32(12, self.cfg.cpuid_features);
        } else {
            t.u64(8, o.processor_id);
        }
        t.set_str(16, o.version.as_deref());
        t.u16(20, o.max_speed as u16);
        t.u16(22, o.current_speed as u16);
        t.u8(24, 0x41);
        t.u8(25, 0x01);
        t.u16(26, 0xffff);
        t.u16(28, 0xffff);
        t.u16(30, 0xffff);
        t.set_str(32, o.serial.as_deref());
        t.set_str(33, o.asset.as_deref());
        t.set_str(34, o.part.as_deref());

        let threads = self.cfg.topology.threads_per_socket();
        let cores = self.cfg.topology.cores_per_socket();
        let core_count = if cores > 255 { 0xff } else { cores as u8 };
        let thread_count = if threads > 255 { 0xff } else { threads as u8 };
        t.u8(35, core_count);
        t.u8(36, core_count);
        t.u8(37, thread_count);
        t.u16(38, 0x02);
        t.u16(40, o.processor_family);
        if v30 {
            t.u16(42, cores as u16);
            t.u16(44, cores as u16);
            t.u16(46, threads as u16);
        } else if core_count == 0xff || thread_count == 0xff {
            return Err(SmbiosError::new(
                "SMBIOS 2.0 doesn't support number of processor cores/threads more than 255, \
                 use -machine smbios-entry-point-type=64 option to enable SMBIOS 3.0 support",
            ));
        }
        self.finish(t);
        self.type4_count += 1;
        Ok(())
    }

    fn type8(&mut self) {
        for (i, o) in self.opts.type8.iter().enumerate() {
            if self.skip(8, true) {
                return;
            }
            // QEMU really does number these from T0_BASE.
            let mut t = Table::new(8, T0_BASE + i as u32, 9);
            t.set_str(4, o.internal_reference.as_deref());
            t.set_str(6, o.external_reference.as_deref());
            t.u8(5, 0);
            t.u8(7, o.connector_type);
            t.u8(8, o.port_type);
            self.finish(t);
        }
    }

    fn type9(&mut self) {
        for (i, o) in self.opts.type9.iter().enumerate() {
            if self.skip(9, true) {
                return;
            }
            let mut t = Table::new(9, T9_BASE + i as u32, 17);
            t.set_str(4, o.slot_designation.as_deref());
            t.u8(5, o.slot_type);
            t.u8(6, o.slot_data_bus_width);
            t.u8(7, o.current_usage);
            t.u8(8, o.slot_length);
            t.u16(9, o.slot_id);
            t.u8(11, o.slot_characteristics1);
            t.u8(12, o.slot_characteristics2);
            // A 0xff store into the 16-bit segment field, exactly as QEMU writes it.
            t.u16(13, 0xff);
            t.u8(15, 0xff);
            t.u8(16, 0xff);
            self.finish(t);
        }
    }

    fn type11(&mut self) {
        let values = &self.opts.type11;
        if values.is_empty() || self.skip(11, true) {
            return;
        }
        let mut t = Table::new(11, T11_BASE, 5);
        t.u8(4, values.len() as u8);
        // QEMU frees the strings after the first build, so a 64-bit retry after a failed 32-bit
        // attempt gets the count but no strings.
        if !*self.type11_freed {
            for v in values {
                t.push_str(Some(v));
            }
        }
        *self.type11_freed = true;
        self.finish(t);
    }

    fn type16(&mut self, dimm_cnt: u64) {
        if self.skip(16, true) {
            return;
        }
        let ram_size = self.cfg.ram_size;
        let mut t = Table::new(16, T16_BASE, 23);
        t.u8(4, 0x01);
        t.u8(5, 0x03);
        t.u8(6, 0x06);
        let size_kb = ram_size.div_ceil(KIB);
        if size_kb < MAX_T16_STD_SZ {
            t.u32(7, size_kb as u32);
        } else {
            t.u32(7, MAX_T16_STD_SZ as u32);
            t.u64(15, ram_size);
        }
        t.u16(11, 0xfffe);
        t.u16(13, dimm_cnt as u16);
        self.finish(t);
    }

    fn type17(&mut self, instance: u64, size: u64) {
        if self.skip(17, true) {
            return;
        }
        let o = &self.opts.type17;
        let mut t = Table::new(17, T17_BASE + instance as u32, 40);
        t.u16(4, 0x1000);
        t.u16(6, 0xfffe);
        t.u16(8, 0xffff);
        t.u16(10, 0xffff);
        let size_mb = size.div_ceil(MIB);
        if size_mb < MAX_T17_STD_SZ {
            t.u16(12, size_mb as u16);
        } else {
            assert!(size_mb < MAX_T17_EXT_SZ, "SMBIOS memory device too large");
            t.u16(12, MAX_T17_STD_SZ as u16);
            t.u32(28, size_mb as u32);
        }
        t.u8(14, 0x09);
        t.u8(15, 0);
        let loc = snprintf128(format!("{} {}", or_null(o.loc_pfx.as_deref()), instance as i32));
        let idx = t.push_str(Some(&loc));
        t.u8(16, idx);
        t.set_str(17, o.bank.as_deref());
        t.u8(18, 0x07);
        t.u16(19, 0x02);
        t.u16(21, o.speed);
        t.set_str(23, o.manufacturer.as_deref());
        t.set_str(24, o.serial.as_deref());
        t.set_str(25, o.asset.as_deref());
        t.set_str(26, o.part.as_deref());
        t.u8(27, 0);
        t.u16(32, o.speed);
        self.finish(t);
    }

    fn type19(&mut self, instance: u32, offset: u32, start: u64, size: u64) {
        if self.skip(19, true) {
            return;
        }
        let mut t = Table::new(19, T19_BASE + offset + instance, 31);
        let end = start.wrapping_add(size).wrapping_sub(1);
        assert!(end > start, "SMBIOS type 19 range is empty");
        let start_kb = start / KIB;
        let end_kb = end / KIB;
        if start_kb < u64::from(u32::MAX) && end_kb < u64::from(u32::MAX) {
            t.u32(4, start_kb as u32);
            t.u32(8, end_kb as u32);
        } else {
            t.u32(4, u32::MAX);
            t.u32(8, u32::MAX);
            t.u64(15, start);
            t.u64(23, end);
        }
        t.u16(12, 0x1000);
        t.u8(14, 1);
        self.finish(t);
    }

    fn type32(&mut self) {
        if self.skip(32, true) {
            return;
        }
        let t = Table::new(32, T32_BASE, 11);
        self.finish(t);
    }

    fn type41(&mut self) -> Result<(), SmbiosError> {
        for (i, o) in self.opts.type41.iter().enumerate() {
            if self.skip(41, true) {
                return Ok(());
            }
            let mut t = Table::new(41, T41_BASE + i as u32, 11);
            t.set_str(4, o.designation.as_deref());
            t.u8(5, o.kind);
            t.u8(6, o.instance);
            if let Some(id) = o.pcidev.as_deref() {
                let Some(dev) = self.cfg.pci_devices.iter().find(|d| d.id == id) else {
                    return Err(SmbiosError::new(format!(
                        "No PCI device {id} for SMBIOS type 41 entry {}",
                        or_null(o.designation.as_deref())
                    )));
                };
                if !dev.on_root_bus {
                    return Err(SmbiosError::new(format!(
                        "Cannot create type 41 entry for PCI device {id}: not attached to the \
                         root bus"
                    )));
                }
                t.u8(9, dev.bus);
                t.u8(10, dev.devfn);
            }
            self.finish(t);
        }
        Ok(())
    }

    fn type127(&mut self) {
        if self.skip(127, true) {
            return;
        }
        let t = Table::new(127, T127_BASE, 4);
        self.finish(t);
    }

    /// `smbios_entry_point_setup`.
    fn anchor(&self) -> Vec<u8> {
        match self.ep_type {
            SmbiosEntryPointType::Ep64 => {
                let mut a = vec![0u8; SMBIOS_30_ENTRY_POINT_LEN];
                a[0..5].copy_from_slice(b"_SM3_");
                a[6] = SMBIOS_30_ENTRY_POINT_LEN as u8;
                a[7] = 3;
                a[8] = 0;
                a[9] = 0;
                a[10] = 1;
                a[12..16].copy_from_slice(&(self.tables.len() as u32).to_le_bytes());
                a
            }
            _ => {
                let mut a = vec![0u8; SMBIOS_21_ENTRY_POINT_LEN];
                a[0..4].copy_from_slice(b"_SM_");
                a[5] = SMBIOS_21_ENTRY_POINT_LEN as u8;
                a[6] = 2;
                a[7] = 8;
                a[8..10].copy_from_slice(&(self.table_max as u16).to_le_bytes());
                a[16..21].copy_from_slice(b"_DMI_");
                a[22..24].copy_from_slice(&(self.tables.len() as u16).to_le_bytes());
                a[28..30].copy_from_slice(&(self.table_cnt as u16).to_le_bytes());
                a[30] = 0x28;
                a
            }
        }
    }

    /// `smbios_get_tables_ep`.
    fn build(mut self) -> Result<SmbiosTables, SmbiosError> {
        self.type0();
        self.type1();
        self.type2();
        self.type3();

        let sockets = self.cfg.topology.sockets;
        assert!(sockets >= 1, "SMBIOS needs at least one socket");
        for i in 0..sockets {
            self.type4(i)?;
        }

        self.type8();
        self.type9();
        self.type11();

        let dev_size = self.cfg.memory_device_size;
        let ram_size = self.cfg.ram_size;
        let dimm_cnt = ram_size.div_ceil(dev_size);
        let gap = u64::from(T19_BASE - T17_BASE);
        let offset = dimm_cnt.saturating_sub(gap) as u32;

        self.type16(dimm_cnt);
        for i in 0..dimm_cnt {
            let size = if i < dimm_cnt - 1 { dev_size } else { (ram_size - 1) % dev_size + 1 };
            self.type17(i, size);
        }
        let mem_array = &self.cfg.mem_array;
        for (i, m) in mem_array.iter().enumerate() {
            self.type19(i as u32, offset, m.address, m.length);
        }
        assert!(
            (mem_array.len() as u64 + u64::from(offset)) < u64::from(T32_BASE - T19_BASE),
            "SMBIOS type 19 handles overlap type 32"
        );

        self.type32();
        self.type41()?;
        self.type127();

        if self.type4_count != 0 && self.type4_count != sockets {
            return Err(SmbiosError::new(format!(
                "Expected {sockets} SMBIOS Type 4 tables, got {} instead",
                self.type4_count
            )));
        }
        if self.ep_type == SmbiosEntryPointType::Ep32
            && self.tables.len() > SMBIOS_21_MAX_TABLES_LEN
        {
            return Err(SmbiosError::new(format!(
                "SMBIOS 2.1 table length {} exceeds {}",
                self.tables.len(),
                SMBIOS_21_MAX_TABLES_LEN
            )));
        }
        let anchor = self.anchor();
        Ok(SmbiosTables { tables: self.tables, anchor, ep_type: self.ep_type })
    }
}

fn get_tables_ep(
    cfg: &SmbiosConfig,
    opts: &SmbiosOptions,
    ep_type: SmbiosEntryPointType,
    type11_freed: &mut bool,
) -> Result<SmbiosTables, SmbiosError> {
    let builder = Builder {
        cfg,
        opts,
        ep_type,
        uuid: opts.uuid.or(cfg.uuid),
        tables: opts.usr_blobs.clone(),
        table_max: opts.usr_table_max,
        table_cnt: opts.usr_table_cnt,
        type4_count: 0,
        type11_freed,
    };
    builder.build()
}

/// Builds `etc/smbios/smbios-tables` and `etc/smbios/smbios-anchor`, like
/// `fw_cfg_build_smbios` followed by `smbios_get_tables`.
///
/// With [`SmbiosEntryPointType::Auto`] the 32-bit layout is tried first and the 64-bit one is
/// used when the tables are too long or a socket has more than 255 cores or threads.
pub fn smbios_get_tables(
    cfg: &SmbiosConfig,
    options: &SmbiosOptions,
) -> Result<SmbiosTables, SmbiosError> {
    let mut opts = options.clone();
    if let Some(d) = &cfg.defaults {
        opts.apply_defaults(d);
    }
    let mut type11_freed = false;
    match cfg.ep_type {
        SmbiosEntryPointType::Ep64 => {
            get_tables_ep(cfg, &opts, SmbiosEntryPointType::Ep64, &mut type11_freed)
        }
        SmbiosEntryPointType::Ep32 => {
            get_tables_ep(cfg, &opts, SmbiosEntryPointType::Ep32, &mut type11_freed)
        }
        SmbiosEntryPointType::Auto => {
            match get_tables_ep(cfg, &opts, SmbiosEntryPointType::Ep32, &mut type11_freed) {
                Ok(t) => Ok(t),
                Err(_) => get_tables_ep(cfg, &opts, SmbiosEntryPointType::Ep64, &mut type11_freed),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opts_split_like_qemu() {
        let o = Opts::parse("type=1,serial=a,,b,uefi,novm,id=x,serial=c");
        let names: Vec<_> = o.list.iter().map(|(n, v)| format!("{n}={v}")).collect();
        assert_eq!(names, ["type=1", "serial=a,b", "uefi=on", "vm=off", "serial=c"]);
        assert_eq!(o.get("serial"), Some("c"));
    }

    #[test]
    fn numbers_follow_strtoull() {
        assert_eq!(parse_number("n", "0x10").ok(), Some(16));
        assert_eq!(parse_number("n", "010").ok(), Some(8));
        assert_eq!(parse_number("n", " 12").ok(), Some(12));
        assert_eq!(parse_number("n", "-1").ok(), Some(u64::MAX));
        assert!(parse_number("n", "0x").is_err());
        assert!(parse_number("n", "08").is_err());
        assert!(parse_number("n", "").is_err());
        assert!(parse_number("n", "12a").is_err());
        assert_eq!(
            parse_number("n", "99999999999999999999").unwrap_err().message(),
            "Value '99999999999999999999' is too large for parameter 'n'"
        );
    }

    #[test]
    fn release_follows_sscanf() {
        assert_eq!(parse_release("4.5"), Some((4, 5)));
        assert_eq!(parse_release(" 1. 2junk"), Some((1, 2)));
        assert_eq!(parse_release("256.1"), Some((0, 1)));
        assert_eq!(parse_release("1"), None);
        assert_eq!(parse_release("1 .2"), None);
        assert_eq!(parse_release("a.b"), None);
    }

    #[test]
    fn uuid_parse() {
        let u = parse_uuid("12345678-9abc-def0-1234-56789ABCDEF0");
        assert_eq!(
            u,
            Some([
                0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc,
                0xde, 0xf0
            ])
        );
        assert_eq!(parse_uuid("12345678-9abc-def0-1234-56789abcdef"), None);
        assert_eq!(parse_uuid("12345678x9abc-def0-1234-56789abcdef0"), None);
        assert_eq!(parse_uuid("g2345678-9abc-def0-1234-56789abcdef0"), None);
    }
}
