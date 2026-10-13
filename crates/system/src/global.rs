// SPDX-License-Identifier: GPL-2.0-or-later

//! `-global`: the qdev global properties of system/qdev-monitor.c and hw/core/qdev-properties.c.
//!
//! QEMU sets a global on every device that is an instance of its driver, when the device is
//! created and before the device's own properties. ruvm builds its boards without QOM, so tables
//! taken from QEMU 11.1 say what QEMU would create: `global/types.txt` has the device types,
//! `global/boards.txt` the objects each board makes with `-nodefaults`, and `global/props.txt`
//! the properties of those objects. A global that reaches a board object is checked against
//! them and fails with QEMU's error when the property or the value is wrong. A good one that
//! ruvm would not act on is refused rather than dropped. ruvm takes `force-legacy` and
//! `ioeventfd` of `virtio-mmio`, and the properties of the NIC models a board only makes while
//! it lists them. On `-device` and `-drive if=virtio`, the globals go in front of the device's
//! own properties, so the device code sees what `object_apply_global_props()` leaves.
//!
//! The tables follow the default machine options. Options that change which objects a board
//! makes, such as `gic-version` or the microvm `pit`, `pic` and `rtc`, are not taken into
//! account. Globals on CPUs, on the VGA of `-vga`, and on the interrupt controllers and timers
//! that depend on the accelerator are refused.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use ruvm_base::report::{Location, report_error, warn_report};
use ruvm_base::{Error, Result};
use ruvm_hw_virtio::mmio::VIRTIO_MMIO_FORCE_LEGACY_DEFAULT;
use ruvm_qapi::opts::{QemuOptDesc, QemuOptType, QemuOptsList, is_help_option};
use ruvm_qapi::types::{
    FloppyDriveType, LostTickPolicy, MigMode, MultiFDCompression, OnOffAuto, ZeroPageDetection,
};
use ruvm_qapi::visit::{QEnumLookup, StringInputVisitor, Visitor, VisitorExt, qapi_bool_parse};

use crate::x86::Located;

/// The target architectures the tables cover.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Arch {
    X86,
    Arm,
    Riscv,
}

impl Arch {
    /// The architecture of the `qemu-system-*` target `target`.
    pub(crate) fn of_target(target: &str) -> Option<Arch> {
        if crate::x86::is_x86(target) {
            Some(Arch::X86)
        } else if crate::arm::is_arm(target) {
            Some(Arch::Arm)
        } else if crate::riscv::is_riscv(target) {
            Some(Arch::Riscv)
        } else {
            None
        }
    }

    fn letter(self) -> char {
        match self {
            Arch::X86 => 'x',
            Arch::Arm => 'a',
            Arch::Riscv => 'r',
        }
    }

    fn name(self) -> &'static str {
        match self {
            Arch::X86 => "x86",
            Arch::Arm => "arm",
            Arch::Riscv => "riscv",
        }
    }
}

/// `GlobalProperty`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Global {
    driver: String,
    property: String,
    value: String,
    used: bool,
}

impl Global {
    fn new(driver: &str, property: &str, value: &str) -> Global {
        Global {
            driver: driver.to_string(),
            property: property.to_string(),
            value: value.to_string(),
            used: false,
        }
    }

    /// `object_apply_global_props()` puts this in front of the error.
    fn prefix(&self) -> String {
        format!("can't apply global {}.{}={}: ", self.driver, self.property, self.value)
    }

    fn not_found(&self, ty: &str) -> Error {
        Error::generic(format!("Property '{ty}.{}' not found", self.property))
            .prepend(self.prefix())
    }

    fn unsupported(&self, what: &str) -> Error {
        Error::generic(format!("{what} not supported by ruvm yet")).prepend(self.prefix())
    }
}

/// `global_props()`, in command line order.
static GLOBALS: Mutex<Vec<Global>> = Mutex::new(Vec::new());

/// The architecture of the board, once [`apply_board`] has run.
static ARCH: OnceLock<Arch> = OnceLock::new();

fn globals() -> MutexGuard<'static, Vec<Global>> {
    GLOBALS.lock().unwrap_or_else(PoisonError::into_inner)
}

const GLOBAL_OPTS: [QemuOptDesc; 3] = [
    QemuOptDesc::new("driver", QemuOptType::String),
    QemuOptDesc::new("property", QemuOptType::String),
    QemuOptDesc::new("value", QemuOptType::String),
];

/// `qemu_global_option()`. Errors are reported here, and `-global help` lists the options;
/// either way the caller exits with status 1.
pub(crate) fn global_option(s: &str) -> bool {
    match parse_global(s) {
        Ok(Some(g)) => {
            globals().push(g);
            true
        }
        Ok(None) => false,
        Err(e) => {
            report_error(&e);
            false
        }
    }
}

/// `qemu_global_option()` without the side effects: `Ok(None)` when `qemu_opts_parse_noisily()`
/// already reported the problem.
fn parse_global(s: &str) -> Result<Option<Global>> {
    if let Some((driver, property, value)) = split_short(s) {
        return Ok(Some(Global::new(driver, property, value)));
    }
    let mut list = QemuOptsList::new("global", &GLOBAL_OPTS);
    let Some(opts) = list.parse_noisily(s, false) else { return Ok(None) };
    match (opts.get("driver"), opts.get("property"), opts.get("value")) {
        (Some(d), Some(p), Some(v)) => Ok(Some(Global::new(d, p, v))),
        _ => Err(Error::generic("options 'driver', 'property', and 'value' are required")),
    }
}

/// `sscanf(str, "%63[^.=].%63[^=]%n", ...)` with `str[offset] == '='`: `driver.property=value`,
/// where the driver has no dot and the value is everything after the first `=`.
fn split_short(s: &str) -> Option<(&str, &str, &str)> {
    let dot = s.find(['.', '='])?;
    if dot == 0 || dot > 63 || s.as_bytes()[dot] != b'.' {
        return None;
    }
    let rest = &s[dot + 1..];
    let eq = rest.find('=')?;
    if eq == 0 || eq > 63 {
        return None;
    }
    Some((&s[..dot], &rest[..eq], &rest[eq + 1..]))
}

/// A line of `global/types.txt`.
struct TypeEntry {
    parent: &'static str,
    arches: &'static str,
    hotpluggable: bool,
}

fn data_lines(s: &'static str) -> impl Iterator<Item = &'static str> {
    s.lines().filter(|l| !l.is_empty() && !l.starts_with('#'))
}

fn types() -> &'static HashMap<&'static str, TypeEntry> {
    static TYPES: OnceLock<HashMap<&'static str, TypeEntry>> = OnceLock::new();
    TYPES.get_or_init(|| {
        data_lines(include_str!("global/types.txt"))
            .map(|l| {
                let f: Vec<&str> = l.split(' ').collect();
                let e =
                    TypeEntry { parent: f[1], arches: f[2], hotpluggable: f.get(3) != Some(&"n") };
                (f[0], e)
            })
            .collect()
    })
}

/// A device type of QEMU 11.1 for `arch`.
fn device_type(arch: Arch, name: &str) -> Option<&'static TypeEntry> {
    types().get(name).filter(|t| t.arches.contains(arch.letter()))
}

/// `object_dynamic_cast()` on type names: whether `ty` is `ancestor` or derives from it.
fn is_a(ty: &str, ancestor: &str) -> bool {
    let mut cur = ty;
    loop {
        if cur == ancestor {
            return true;
        }
        match types().get(cur) {
            Some(t) => cur = t.parent,
            None => return false,
        }
    }
}

/// `qdev_alias_table` for x86, Arm and RISC-V: the short names `-device` takes.
const ALIASES: &[(&str, &str)] = &[
    ("AC97", "ac97"),
    ("e1000", "e1000-82540em"),
    ("ES1370", "es1370"),
    ("ich9-ahci", "ahci"),
    ("lsi53c895a", "lsi"),
    ("virtio-9p-pci", "virtio-9p"),
    ("virtio-balloon-pci", "virtio-balloon"),
    ("virtio-blk-pci", "virtio-blk"),
    ("virtio-gpu-pci", "virtio-gpu"),
    ("virtio-gpu-gl-pci", "virtio-gpu-gl"),
    ("virtio-gpu-rutabaga-pci", "virtio-gpu-rutabaga"),
    ("virtio-input-host-pci", "virtio-input-host"),
    ("virtio-iommu-pci", "virtio-iommu"),
    ("virtio-keyboard-pci", "virtio-keyboard"),
    ("virtio-mouse-pci", "virtio-mouse"),
    ("virtio-net-pci", "virtio-net"),
    ("virtio-rng-pci", "virtio-rng"),
    ("virtio-scsi-pci", "virtio-scsi"),
    ("virtio-serial-pci", "virtio-serial"),
    ("virtio-sound-pci", "virtio-sound"),
    ("virtio-tablet-pci", "virtio-tablet"),
];

/// `qdev_get_device_class()`: the device type `-device driver=` names, by type name first and
/// then by alias.
fn device_class(arch: Arch, driver: &str) -> Option<&'static str> {
    let name = match types().get_key_value(driver) {
        Some((name, _)) => *name,
        None => ALIASES.iter().find(|(_, a)| *a == driver)?.0,
    };
    (device_type(arch, name).is_some() && is_a(name, "device")).then_some(name)
}

/// The kind of a property, which says which values it takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Bool,
    U8,
    U16,
    U32,
    U64,
    I32,
    I64,
    Size,
    Str,
    Enum(&'static str),
    /// A child, link or property of a kind ruvm does not check.
    Other,
}

impl Kind {
    fn from_name(name: &'static str) -> Kind {
        match name {
            "bool" => Kind::Bool,
            "uint8" => Kind::U8,
            "uint16" => Kind::U16,
            "uint32" => Kind::U32,
            "uint64" => Kind::U64,
            "int32" => Kind::I32,
            "int64" => Kind::I64,
            "size" => Kind::Size,
            "str" => Kind::Str,
            "other" => Kind::Other,
            e if enum_lookup(e).is_some() => Kind::Enum(e),
            _ => Kind::Other,
        }
    }
}

fn enum_lookup(name: &str) -> Option<QEnumLookup> {
    Some(match name {
        "OnOffAuto" => OnOffAuto::LOOKUP,
        "LostTickPolicy" => LostTickPolicy::LOOKUP,
        "FloppyDriveType" => FloppyDriveType::LOOKUP,
        "MigMode" => MigMode::LOOKUP,
        "MultiFDCompression" => MultiFDCompression::LOOKUP,
        "ZeroPageDetection" => ZeroPageDetection::LOOKUP,
        _ => return None,
    })
}

/// `object_property_parse()` of `value` into a property of kind `kind`, without the object:
/// the string input visitor's errors.
fn check_value(kind: Kind, ty: &str, name: &str, value: &str) -> Result<()> {
    let mut v = StringInputVisitor::new(value);
    let field = Some(name);
    match kind {
        Kind::Bool => v.type_bool(field, &mut false),
        Kind::U8 => v.type_uint8(field, &mut 0),
        Kind::U16 => v.type_uint16(field, &mut 0),
        Kind::U32 => v.type_uint32(field, &mut 0),
        Kind::U64 => v.type_uint64(field, &mut 0),
        Kind::I32 => v.type_int32(field, &mut 0),
        Kind::I64 => v.type_int64(field, &mut 0),
        Kind::Size => v.type_size(field, &mut 0),
        Kind::Enum(e) => {
            let lookup = enum_lookup(e).expect("an enum Kind::from_name knows");
            v.type_enum(field, &mut 0, &lookup)
        }
        // set_mac()
        Kind::Str if matches!(name, "mac" | "fp_start_macaddr") && !is_mac(value) => {
            Err(Error::generic(format!("Property '{ty}.{name}' doesn't take value '{value}'")))
        }
        Kind::Str | Kind::Other => Ok(()),
    }
}

/// Six pairs of hex digits, each pair but the last followed by ':' or '-'.
fn is_mac(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 17
        && (0..6).all(|i| b[i * 3].is_ascii_hexdigit() && b[i * 3 + 1].is_ascii_hexdigit())
        && (0..5).all(|i| matches!(b[i * 3 + 2], b':' | b'-'))
}

/// `global/props.txt`: the properties each type adds to those of its parent.
fn props() -> &'static HashMap<&'static str, Vec<(&'static str, Kind)>> {
    static PROPS: OnceLock<HashMap<&'static str, Vec<(&'static str, Kind)>>> = OnceLock::new();
    PROPS.get_or_init(|| {
        data_lines(include_str!("global/props.txt"))
            .map(|l| {
                let mut f = l.split(' ');
                let class = f.next().unwrap_or_default();
                let props = f
                    .filter_map(|p| p.rsplit_once(':'))
                    .map(|(name, kind)| (name, Kind::from_name(kind)))
                    .collect();
                (class, props)
            })
            .collect()
    })
}

/// The kind of property `name` of a board object of type `ty`.
fn prop_kind(ty: &str, name: &str) -> Option<Kind> {
    let mut cur = ty;
    loop {
        if let Some(&(_, kind)) = props().get(cur).and_then(|p| p.iter().find(|(n, _)| *n == name))
        {
            return Some(kind);
        }
        cur = types().get(cur)?.parent;
    }
}

/// An object of a board: its type, and `s` when it comes with a serial port or `t` for a NIC
/// model that only lives while the board lists them.
type BoardObject = (&'static str, Option<char>);

/// `global/boards.txt`: the objects of each board in the order QEMU creates them.
fn boards() -> &'static HashMap<&'static str, Vec<BoardObject>> {
    static BOARDS: OnceLock<HashMap<&'static str, Vec<BoardObject>>> = OnceLock::new();
    BOARDS.get_or_init(|| {
        let mut boards: HashMap<&'static str, Vec<BoardObject>> = HashMap::new();
        let mut section = "";
        for l in data_lines(include_str!("global/boards.txt")) {
            if let Some(name) = l.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
                section = name;
                continue;
            }
            let mut f = l.split(' ').skip(1);
            let object = f.next().unwrap_or_default();
            let flag = f.next().and_then(|s| s.chars().next());
            let objects = boards.entry(section).or_default();
            if !objects.iter().any(|(o, _)| *o == object) {
                objects.push((object, flag));
            }
        }
        boards
    })
}

/// The board properties ruvm acts on.
fn takes(ty: &str, property: &str) -> bool {
    ty == "virtio-mmio" && matches!(property, "force-legacy" | "ioeventfd")
}

/// Devices that a board makes in a form the tables do not follow: which one depends on the
/// accelerator or on machine options.
const VARYING: &[&str] = &[
    "apic",
    "apic-common",
    "kvm-apic",
    "ioapic",
    "ioapic-common",
    "kvm-ioapic",
    "isa-i8259",
    "pic-common",
    "kvm-i8259",
    "isa-pit",
    "pit-common",
    "kvm-pit",
    "kvmclock",
    "kvmvapic",
    "arm_gic",
    "arm_gic_common",
    "kvm-arm-gic",
    "arm-gicv2m",
    "arm-gicv3",
    "arm-gicv3-common",
    "kvm-arm-gicv3",
    "arm-gicv3-its",
    "arm-gicv3-its-common",
    "arm-its-kvm",
];

/// Whether there is any `-global`.
pub(crate) fn any() -> bool {
    !globals().is_empty()
}

/// What QEMU would create for the machine, beyond the tables.
#[derive(Debug)]
pub(crate) struct Board<'a> {
    pub arch: Arch,
    /// `q35`, `microvm`, `virt`, `sbsa-ref` or `none`.
    pub name: &'a str,
    /// Whether the board gets a serial port.
    pub serial: bool,
    /// The CPU type, or `None` when the board has no CPU.
    pub cpu: Option<String>,
    /// Whether `-vga std` makes a VGA.
    pub vga: bool,
    /// No `-nodefaults`, so QEMU would add its default devices.
    pub defaults: bool,
}

/// The globals as the board objects take them, when the board is created. The errors are
/// fatal, as with `error_fatal` in `qdev_prop_set_globals()`.
pub(crate) fn apply_board(board: &Board<'_>) -> Result<()> {
    let _ = ARCH.set(board.arch);
    apply_board_to(&mut globals(), board)
}

fn apply_board_to(globals: &mut [Global], board: &Board<'_>) -> Result<()> {
    if globals.is_empty() {
        return Ok(());
    }
    let key = format!("{} {}", board.arch.name(), board.name);
    let Some(objects) = boards().get(key.as_str()) else {
        return Err(Error::generic("-global is not supported with this machine by ruvm yet"));
    };
    if board.defaults && board.name != "none" {
        return Err(Error::generic("-global without -nodefaults is not supported by ruvm yet"));
    }
    for &(ty, flag) in objects {
        if flag == Some('s') && !board.serial {
            continue;
        }
        for g in globals.iter_mut().filter(|g| is_a(ty, &g.driver)) {
            g.used = true;
            let Some(kind) = prop_kind(ty, &g.property) else { return Err(g.not_found(ty)) };
            check_value(kind, ty, &g.property, &g.value).map_err(|e| e.prepend(g.prefix()))?;
            let looked_up = kind == Kind::Str
                && PARSED_STRINGS.contains(&g.property.as_str())
                && !matches!(g.property.as_str(), "mac" | "fp_start_macaddr");
            if (flag != Some('t') || looked_up) && !takes(ty, &g.property) {
                return Err(g.unsupported(&format!("property '{ty}.{}' is", g.property)));
            }
        }
    }
    let vga = board.vga.then_some("VGA");
    for g in globals.iter_mut().filter(|g| !g.used) {
        // A model the tables do not know, such as an alias, may be any CPU.
        let cpu = board.cpu.as_deref().is_some_and(|c| match device_type(board.arch, c) {
            Some(_) => is_a(c, &g.driver),
            None => is_a(&g.driver, "cpu"),
        });
        let vga = vga.is_some_and(|v| is_a(v, &g.driver));
        let varying = board.name != "none" && VARYING.contains(&g.driver.as_str());
        if cpu || vga || varying {
            g.used = true;
            return Err(g.unsupported(&format!("globals on '{}' are", g.driver)));
        }
    }
    Ok(())
}

/// `force-legacy` of the board's `virtio-mmio` transports: the last global that sets it, as
/// they are applied in order. [`apply_board`] has checked the value.
pub(crate) fn virtio_mmio_force_legacy() -> bool {
    globals()
        .iter()
        .rev()
        .find(|g| g.property == "force-legacy" && is_a("virtio-mmio", &g.driver))
        .and_then(|g| qapi_bool_parse("force-legacy", &g.value).ok())
        .unwrap_or(VIRTIO_MMIO_FORCE_LEGACY_DEFAULT)
}

/// `qdev_prop_check_globals()`, when the machine is created: warns about the globals no device
/// took.
pub(crate) fn check_globals() {
    let globals = globals();
    let Some(&arch) = ARCH.get() else { return };
    for g in globals.iter().filter(|g| !g.used) {
        match device_type(arch, &g.driver).filter(|_| is_a(&g.driver, "device")) {
            None => {
                warn_report(&format!("global {}.{} has invalid class name", g.driver, g.property))
            }
            Some(t) if !t.hotpluggable => {
                warn_report(&format!("global {}.{}={} not used", g.driver, g.property, g.value));
            }
            Some(_) => {}
        }
    }
}

/// The properties of a virtio PCI proxy that its virtio device does not have: those of
/// `pci-device` and `virtio-pci`, and the ones most proxies add.
const PROXY_ONLY: &[&str] = &[
    "virtio-pci-bus-master-bug-migration",
    "modern-pio-notify",
    "page-per-vq",
    "ats",
    "x-ats-page-aligned",
    "x-pcie-pm-no-soft-reset",
    "x-pcie-flr-init",
    "aer",
    "disable-legacy",
    "disable-modern",
    "vectors",
    "ioeventfd",
    "class",
];

/// The virtio device inside a virtio PCI proxy, which `virtio_instance_init_common()` creates
/// first and so takes its globals first.
fn inner_device(ty: &str) -> Option<String> {
    if !is_a(ty, "virtio-pci") {
        return None;
    }
    let base = ty
        .strip_suffix("-non-transitional")
        .or_else(|| ty.strip_suffix("-transitional"))
        .unwrap_or(ty)
        .strip_suffix("-pci")?;
    let inner = format!("{base}-device");
    types().contains_key(inner.as_str()).then_some(inner)
}

/// The globals for a device of type `ty`, each with the type it reaches: the inner virtio
/// device's first, then the device's.
fn pick(globals: &mut [Global], ty: &str) -> Vec<(Global, String)> {
    let mut picked = Vec::new();
    for owner in inner_device(ty).into_iter().chain([ty.to_string()]) {
        for g in globals.iter_mut().filter(|g| is_a(&owner, &g.driver)) {
            g.used = true;
            picked.push((g.clone(), owner.clone()));
        }
    }
    picked
}

/// An option value with its commas doubled.
fn escape(s: &str) -> String {
    s.replace(',', ",,")
}

/// Splits a `-device` argument after its first element, `get_opt_value()` style.
fn split_first(arg: &str) -> (&str, &str) {
    let b = arg.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b',' {
            if b.get(i + 1) == Some(&b',') {
                i += 2;
                continue;
            }
            break;
        }
        i += 1;
    }
    arg.split_at(i)
}

/// `arg` with `globals` in front of its properties.
fn merge(arg: &str, globals: &[(Global, String)]) -> String {
    let props: Vec<String> =
        globals.iter().map(|(g, _)| format!("{}={}", g.property, escape(&g.value))).collect();
    let props = props.join(",");
    let implied = arg.find([',', '=']).is_none_or(|i| arg.as_bytes()[i] != b'=');
    if implied {
        let (first, rest) = split_first(arg);
        format!("{first},{props}{rest}")
    } else {
        format!("{props},{arg}")
    }
}

/// A property no `-device` ruvm plans can take from a global: the option keys `qdev_device_add()`
/// takes itself, and names `QemuOpts` cannot carry.
fn plain_key(property: &str) -> bool {
    !property.is_empty()
        && !property.contains(',')
        && !matches!(property, "driver" | "id" | "bus")
        && !is_help_option(property)
}

/// A key no device has, which makes the planner stop once it has looked at every property.
const END: &str = "x-ruvm-global-end=on";

/// The string properties whose setters look the value up or parse it, so only planning the
/// device tells whether a global's value is good.
const PARSED_STRINGS: &[&str] =
    &["addr", "audiodev", "chardev", "drive", "fp_start_macaddr", "iothread", "mac", "netdev"];

/// `qdev_device_add()` with the globals: plans `arg` with the globals for its device in front of
/// its own properties. A global that fails gets QEMU's "can't apply global" in front of its
/// error. The tables check most values; for the rest `plan` is called with `true` to find
/// which global an error comes from, and must then leave no trace. `fail` turns the plan of
/// `arg` without the globals into the global's error, for a device that reports property
/// errors later.
pub(crate) fn plan_device<T>(
    arg: &str,
    loc: &Option<Location>,
    mut plan: impl FnMut(&str, bool) -> std::result::Result<T, Located>,
    fail: impl FnOnce(T, Located) -> std::result::Result<T, Located>,
) -> std::result::Result<T, Located> {
    let Some(&arch) = ARCH.get() else { return plan(arg, false) };
    let mut list = QemuOptsList::new("device", &[]).with_implied_opt_name("driver");
    let Ok(opts) = list.parse(arg, true) else { return plan(arg, false) };
    let Some(driver) = opts.get("driver").map(str::to_string) else { return plan(arg, false) };
    let bus = opts.get("bus").map(str::to_string);
    let Some(ty) = device_class(arch, &driver) else { return plan(arg, false) };
    let picked = pick(&mut globals(), ty);
    if picked.is_empty() {
        return plan(arg, false);
    }
    let mut head = format!("driver={}", escape(&driver));
    if let Some(bus) = &bus {
        head = format!("{head},bus={}", escape(bus));
    }
    let mut base = None;
    let mut error = None;
    for (g, owner) in &picked {
        let proxy_only = owner != ty && PROXY_ONLY.contains(&g.property.as_str());
        if !plain_key(&g.property) || proxy_only {
            error = Some(Located(loc.clone(), g.not_found(owner)));
            break;
        }
        let measured = props().contains_key(owner.as_str());
        let kind = prop_kind(owner, &g.property);
        match kind {
            None if measured => {
                error = Some(Located(loc.clone(), g.not_found(owner)));
                break;
            }
            Some(kind) => {
                if let Err(e) = check_value(kind, owner, &g.property, &g.value) {
                    error = Some(Located(loc.clone(), e.prepend(g.prefix())));
                    break;
                }
            }
            None => {}
        }
        head = format!("{head},{}={}", g.property, escape(&g.value));
        let parsed = match kind {
            None | Some(Kind::Other) => true,
            Some(Kind::Str) => PARSED_STRINGS.contains(&g.property.as_str()),
            Some(_) => false,
        };
        if !parsed {
            continue;
        }
        let base = base.get_or_insert_with(|| {
            let mut head = format!("driver={}", escape(&driver));
            if let Some(bus) = &bus {
                head = format!("{head},bus={}", escape(bus));
            }
            plan(&format!("{head},{END}"), true).err().map(|e| e.1.message().to_string())
        });
        let Err(Located(l, e)) = plan(&format!("{head},{END}"), true) else { continue };
        if base.as_deref() == Some(e.message()) {
            continue;
        }
        // The proxy has the inner device's properties under its own name.
        let e = if owner == ty {
            e
        } else {
            let msg = e.message().replace(&format!("'{ty}."), &format!("'{owner}."));
            Error::generic(msg.replace(&format!("'{driver}."), &format!("'{owner}.")))
        };
        error = Some(Located(l, e.prepend(g.prefix())));
        break;
    }
    match error {
        // qdev_device_add() finds the bus before it makes the device, so plan the device as
        // given for its bus errors.
        Some(e) => match plan(arg, false) {
            Ok(t) => fail(t, e),
            Err(_) => Err(e),
        },
        // A property QEMU has but the ruvm device model does not parse.
        None => plan(&merge(arg, &picked), false).map_err(|Located(l, e)| {
            let given = |p: &str| opts.get(p).is_some();
            let missing = picked.iter().find(|(g, _)| {
                !given(&g.property)
                    && [&driver, ty]
                        .iter()
                        .any(|t| e.message() == format!("Property '{t}.{}' not found", g.property))
            });
            match missing {
                Some((g, owner)) => {
                    Located(l, g.unsupported(&format!("property '{owner}.{}' is", g.property)))
                }
                None => Located(l, e),
            }
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(d: &str, p: &str, v: &str) -> Global {
        Global::new(d, p, v)
    }

    #[test]
    fn short_form_splits_at_the_first_dot_and_equals() {
        assert_eq!(split_short("a.b=c,d=e"), Some(("a", "b", "c,d=e")));
        assert_eq!(split_short("foo.bar.baz=1"), Some(("foo", "bar.baz", "1")));
        assert_eq!(split_short("a.b="), Some(("a", "b", "")));
        for s in ["foo", ".b=1", "a.=1", "=1", "a=1", "a.b", "a=b.c=1"] {
            assert_eq!(split_short(s), None, "{s}");
        }
        let long = format!("{}.b=1", "d".repeat(64));
        assert_eq!(split_short(&long), None);
        let long = format!("a.{}=1", "p".repeat(64));
        assert_eq!(split_short(&long), None);
        let ok = format!("{}.{}=1", "d".repeat(63), "p".repeat(63));
        assert!(split_short(&ok).is_some());
    }

    #[test]
    fn long_form_needs_all_three() {
        let got = parse_global("driver=virtio-mmio,property=force-legacy,value=off").unwrap();
        assert_eq!(got, Some(g("virtio-mmio", "force-legacy", "off")));
        let got = parse_global("driver=a,driver=b,property=p,value=v").unwrap();
        assert_eq!(got, Some(g("b", "p", "v")));
        let e = parse_global("driver=a,property=p").unwrap_err();
        assert_eq!(e.message(), "options 'driver', 'property', and 'value' are required");
    }

    #[test]
    fn values_get_the_string_input_visitor_errors() {
        let err = |k, v| check_value(k, "t", "p", v).unwrap_err().message().to_string();
        assert_eq!(err(Kind::Bool, "foo"), "Parameter 'p' expects 'on' or 'off'");
        assert!(check_value(Kind::Bool, "t", "p", "yes").is_ok());
        assert!(check_value(Kind::Str, "t", "mac", "52-54-00-12-34:5f").is_ok());
        assert_eq!(
            check_value(Kind::Str, "e1000e", "mac", "52:54:00:12:34:56:").unwrap_err().message(),
            "Property 'e1000e.mac' doesn't take value '52:54:00:12:34:56:'"
        );
        assert_eq!(err(Kind::U8, "x"), "Parameter 'p' expects uint64");
        assert_eq!(err(Kind::U8, "256"), "Parameter 'p' expects uint8_t");
        assert!(check_value(Kind::U8, "t", "p", "0x10").is_ok());
        assert_eq!(err(Kind::U16, "65536"), "Parameter 'p' expects uint16_t");
        assert_eq!(err(Kind::I32, "x"), "Parameter 'p' expects int64");
        assert_eq!(err(Kind::I32, "2147483648"), "Parameter 'p' expects int32_t");
        assert_eq!(err(Kind::Size, "x"), "Parameter 'p' expects a non-negative number below 2^64");
        assert_eq!(err(Kind::Size, "-1"), "Value '-1' is out of range for parameter 'p'");
        assert_eq!(err(Kind::Enum("OnOffAuto"), "x"), "Parameter 'p' does not accept value 'x'");
        assert!(check_value(Kind::Enum("LostTickPolicy"), "t", "p", "discard").is_ok());
    }

    #[test]
    fn types_and_props_follow_the_parents() {
        assert!(is_a("virtio-sound-pci", "virtio-pci"));
        assert!(is_a("virtio-sound-pci", "device"));
        assert!(is_a("virtio-sound-pci", "object"));
        assert!(!is_a("virtio-sound-pci", "virtio-device"));
        assert_eq!(prop_kind("virtio-mmio", "force-legacy"), Some(Kind::Bool));
        assert_eq!(prop_kind("virtio-mmio", "bogus"), None);
        assert_eq!(
            prop_kind("mc146818rtc", "lost_tick_policy"),
            Some(Kind::Enum("LostTickPolicy"))
        );
        assert_eq!(device_class(Arch::X86, "virtio-sound"), Some("virtio-sound-pci"));
        assert_eq!(device_class(Arch::X86, "migration"), Some("migration"));
        assert_eq!(device_class(Arch::X86, "nope"), None);
        assert_eq!(inner_device("virtio-sound-pci").as_deref(), Some("virtio-sound-device"));
        assert_eq!(
            inner_device("virtio-net-pci-transitional").as_deref(),
            Some("virtio-net-device")
        );
        assert_eq!(inner_device("virtio-sound-device"), None);
    }

    fn board(arch: Arch, name: &str) -> Board<'_> {
        Board { arch, name, serial: false, cpu: None, vga: false, defaults: false }
    }

    fn apply(board: &Board<'_>, gs: &[Global]) -> std::result::Result<Vec<Global>, String> {
        let mut gs = gs.to_vec();
        match apply_board_to(&mut gs, board) {
            Ok(()) => Ok(gs),
            Err(e) => Err(e.message().to_string()),
        }
    }

    #[test]
    fn board_objects_take_globals_in_creation_order() {
        let mv = board(Arch::X86, "microvm");
        let gs = apply(&mv, &[g("virtio-mmio", "force-legacy", "off")]).unwrap();
        assert!(gs[0].used);
        assert_eq!(
            apply(&mv, &[g("virtio-mmio", "bogus", "1")]).unwrap_err(),
            "can't apply global virtio-mmio.bogus=1: Property 'virtio-mmio.bogus' not found"
        );
        assert_eq!(
            apply(
                &mv,
                &[g("virtio-mmio", "force-legacy", "foo"), g("virtio-mmio", "force-legacy", "on")]
            )
            .unwrap_err(),
            "can't apply global virtio-mmio.force-legacy=foo: Parameter 'force-legacy' expects 'on' or 'off'"
        );
        assert_eq!(
            apply(&mv, &[g("device", "zz", "1")]).unwrap_err(),
            "can't apply global device.zz=1: Property 'migration.zz' not found"
        );
        assert_eq!(
            apply(&board(Arch::X86, "q35"), &[g("device", "zz", "1")]).unwrap_err(),
            "can't apply global device.zz=1: Property 'cfi.pflash01.zz' not found"
        );
        // The migration object comes before the virtio-mmio transports.
        assert_eq!(
            apply(&mv, &[g("virtio-mmio", "bogus", "1"), g("migration", "zz", "1")]).unwrap_err(),
            "can't apply global migration.zz=1: Property 'migration.zz' not found"
        );
        assert_eq!(
            apply(&board(Arch::X86, "q35"), &[g("e1000e", "bogus", "1")]).unwrap_err(),
            "can't apply global e1000e.bogus=1: Property 'e1000e.bogus' not found"
        );
        assert!(
            apply(&board(Arch::X86, "q35"), &[g("e1000e", "mac", "52:54:00:12:34:56")]).is_ok()
        );
        assert_eq!(
            apply(&board(Arch::X86, "q35"), &[g("e1000e", "mac", "x")]).unwrap_err(),
            "can't apply global e1000e.mac=x: Property 'e1000e.mac' doesn't take value 'x'"
        );
        assert_eq!(
            apply(&mv, &[g("mc146818rtc", "base_year", "2000")]).unwrap_err(),
            "can't apply global mc146818rtc.base_year=2000: property 'mc146818rtc.base_year' is not supported by ruvm yet"
        );
        // Without a serial port there is no isa-serial, and isa-serial is hotpluggable.
        let gs = apply(&mv, &[g("isa-serial", "bogus", "1")]).unwrap();
        assert!(!gs[0].used);
        let serial = Board { serial: true, ..board(Arch::X86, "microvm") };
        assert!(apply(&serial, &[g("isa-serial", "bogus", "1")]).is_err());
    }

    #[test]
    fn cpus_vga_and_irqchips_are_refused() {
        let cpu = Board { cpu: Some("qemu64-x86_64-cpu".into()), ..board(Arch::X86, "q35") };
        assert_eq!(
            apply(&cpu, &[g("x86_64-cpu", "pmu", "on")]).unwrap_err(),
            "can't apply global x86_64-cpu.pmu=on: globals on 'x86_64-cpu' are not supported by ruvm yet"
        );
        let vga = Board { vga: true, ..board(Arch::X86, "q35") };
        assert!(apply(&vga, &[g("VGA", "vgamem_mb", "4")]).is_err());
        assert!(apply(&board(Arch::X86, "q35"), &[g("VGA", "vgamem_mb", "4")]).is_ok());
        assert!(
            apply(&board(Arch::X86, "q35"), &[g("kvm-pit", "lost_tick_policy", "discard")])
                .is_err()
        );
        assert!(
            apply(&board(Arch::X86, "none"), &[g("kvm-pit", "lost_tick_policy", "discard")])
                .is_ok()
        );
        let defaults = Board { defaults: true, ..board(Arch::X86, "q35") };
        assert_eq!(
            apply(&defaults, &[g("e1000e", "mac", "x")]).unwrap_err(),
            "-global without -nodefaults is not supported by ruvm yet"
        );
        let none = Board { defaults: true, ..board(Arch::X86, "none") };
        assert!(apply(&none, &[g("e1000e", "mac", "x")]).is_ok());
    }

    #[test]
    fn device_globals_go_before_the_device_properties() {
        let picked =
            |gs: &[Global]| gs.iter().map(|g| (g.clone(), String::new())).collect::<Vec<_>>();
        let gs = [g("virtio-sound-pci", "streams", "2"), g("virtio-sound-pci", "x", "a,b")];
        assert_eq!(
            merge("virtio-sound-pci,audiodev=a", &picked(&gs)),
            "virtio-sound-pci,streams=2,x=a,,b,audiodev=a"
        );
        assert_eq!(merge("a,,b", &picked(&gs[..1])), "a,,b,streams=2");
        assert_eq!(
            merge("audiodev=a,driver=virtio-sound-pci", &picked(&gs[..1])),
            "streams=2,audiodev=a,driver=virtio-sound-pci"
        );
        let mut gs = vec![
            g("virtio-sound-pci", "streams", "2"),
            g("virtio-device", "event_idx", "off"),
            g("virtio-mmio", "force-legacy", "off"),
        ];
        let got: Vec<String> = pick(&mut gs, "virtio-sound-pci")
            .into_iter()
            .map(|(g, owner)| format!("{}:{owner}", g.driver))
            .collect();
        assert_eq!(got, ["virtio-device:virtio-sound-device", "virtio-sound-pci:virtio-sound-pci"]);
        assert!(gs[0].used && gs[1].used && !gs[2].used);
    }

    #[test]
    fn device_errors_name_the_global() {
        let _ = ARCH.set(Arch::X86);
        *globals() = vec![g("ich9-intel-hda", "bogus", "0")];
        let plan = |arg: &str, _trial: bool| match arg
            .split(',')
            .skip(1)
            .find(|kv| !kv.starts_with("audiodev="))
        {
            Some(kv) => {
                let msg =
                    format!("Property 'ich9-intel-hda.{}' not found", &kv[..kv.find('=').unwrap()]);
                Err(Located(None, Error::generic(msg)))
            }
            None => Ok(()),
        };
        let e = plan_device("ich9-intel-hda", &None, plan, |_, e| Err(e)).unwrap_err();
        *globals() = vec![g("intel-hda-generic", "debug", "1")];
        let debug = plan_device("ich9-intel-hda", &None, plan, |_, e| Err(e)).unwrap_err();
        *globals() = vec![g("intel-hda-generic", "debug", "1")];
        let given = plan_device("ich9-intel-hda,debug=2", &None, plan, |_, e| Err(e)).unwrap_err();
        globals().clear();
        assert_eq!(
            e.1.message(),
            "can't apply global ich9-intel-hda.bogus=0: Property 'ich9-intel-hda.bogus' not found"
        );
        assert_eq!(
            debug.1.message(),
            "can't apply global intel-hda-generic.debug=1: property 'ich9-intel-hda.debug' is not supported by ruvm yet"
        );
        assert_eq!(given.1.message(), "Property 'ich9-intel-hda.debug' not found");
    }
}
