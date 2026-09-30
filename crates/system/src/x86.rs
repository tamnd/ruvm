// SPDX-License-Identifier: GPL-2.0-or-later

//! The x86 boards, microvm and q35, from the command line: what `-machine`, `-m`, `-smp`,
//! `-kernel`, `-drive`, `-device` and `-serial` turn into, and, on Linux x86 hosts, the
//! board running on KVM.
//!
//! The mapping is plain data and works on any host, so it is tested everywhere. Only the part
//! that opens `/dev/kvm` is built for Linux x86_64.
//!
//! Firmware and option ROMs are looked up in the `-L` directories in command line order, then
//! in `/usr/share/qemu`, `/usr/share/seabios` and `/usr/local/share/qemu`, and last in the
//! `share/qemu` directory next to a `qemu-system-x86_64` found on `PATH`. `-L help` prints
//! that list.

// Without KVM nothing builds a board, so the device plan is only used by the tests.
#![cfg_attr(not(all(target_os = "linux", target_arch = "x86_64")), allow(dead_code))]

use std::collections::HashSet;
use std::path::PathBuf;

use ruvm_base::report::{Location, push_location, report_error};
use ruvm_base::{Error, Result};
use ruvm_chardev::opts::{NographicDefaults, nographic_defaults};
use ruvm_firmware::smbios::{SmbiosOptions, SmbiosTopology, parse_uuid};
use ruvm_machine_x86::board::X86_BOARDS;
use ruvm_machine_x86::pflash::raw_block_length;
use ruvm_machine_x86::q35::PflashDrive;
use ruvm_machine_x86::{
    BoardKind, FileBackend, FirmwareSearch, KernelFiles, MicrovmProps, PflashBacking, Q35Props,
};
use ruvm_qapi::opts::{QemuOptsList, is_help_option};
use ruvm_qapi::types::{MemorySizeConfiguration, SMPConfiguration};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit, Visitor};
use ruvm_qapi::{QDict, QValue};

/// Whether `target` is one the x86 boards exist for.
pub(crate) fn is_x86(target: &str) -> bool {
    matches!(target, "x86_64" | "i386")
}

/// The `-machine help` lines of the x86 boards, as (sort key, line) pairs. An alias gets its
/// own line right before the machine, as `machine_help_func()` prints it.
pub(crate) fn machine_help_lines() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for &(name, alias, desc) in X86_BOARDS {
        let mut text = String::new();
        if let Some(alias) = alias {
            text.push_str(&format!("{alias:<20} {desc} (alias of {name})\n"));
        }
        text.push_str(&format!("{name:<20} {desc}\n"));
        out.push((name.to_string(), text));
    }
    out
}

/// An error and the command line option it belongs to.
#[derive(Debug)]
pub(crate) struct Located(pub Option<Location>, pub Error);

impl Located {
    fn new(loc: &Option<Location>, msg: impl Into<String>) -> Located {
        Located(loc.clone(), Error::generic(msg.into()))
    }

    fn bare(msg: impl Into<String>) -> Located {
        Located(None, Error::generic(msg.into()))
    }

    /// Prints the error with its location in front.
    pub(crate) fn report(&self) {
        let _loc = self.0.clone().map(push_location);
        report_error(&self.1);
    }
}

/// The x86 options of the command line, as the option loop gathers them.
#[derive(Debug)]
pub(crate) struct Cmdline {
    /// `-L`, in order.
    pub data_dirs: Vec<PathBuf>,
    /// `-L help`.
    pub list_data_dirs: bool,
    /// `-cpu`.
    pub cpu: Option<String>,
    /// `-serial`, with where each came from.
    pub serials: Vec<(String, Option<Location>)>,
    /// `-drive`.
    pub drives: Vec<(String, Option<Location>)>,
    /// `-device`.
    pub devices: Vec<(String, Option<Location>)>,
    /// `-nographic`.
    pub nographic: bool,
    /// `default_serial`: no `-serial` and no `-nodefaults`.
    pub default_serial: bool,
    /// `default_monitor`: no `-monitor`, `-qmp` or `-mon`, and no `-nodefaults`.
    pub default_monitor: bool,
    /// `has_defaults`: no `-nodefaults`.
    pub has_defaults: bool,
    /// `-no-reboot`.
    pub no_reboot: bool,
    /// `qemu_uuid`: the last of `-uuid` and `-smbios type=1,uuid=`.
    pub uuid: Option<[u8; 16]>,
    /// `-smbios`, parsed as it comes like `smbios_entry_add()` does.
    pub smbios: SmbiosOptions,
}

impl Default for Cmdline {
    fn default() -> Self {
        Cmdline {
            data_dirs: Vec::new(),
            list_data_dirs: false,
            cpu: None,
            serials: Vec::new(),
            drives: Vec::new(),
            devices: Vec::new(),
            nographic: false,
            default_serial: true,
            default_monitor: true,
            has_defaults: true,
            no_reboot: false,
            uuid: None,
            smbios: SmbiosOptions::new(),
        }
    }
}

impl Cmdline {
    /// `-uuid`: sets `qemu_uuid`, which fw_cfg and the SMBIOS type 1 table report.
    pub(crate) fn set_uuid(&mut self, arg: &str) -> std::result::Result<(), String> {
        let uuid =
            parse_uuid(arg).ok_or("failed to parse UUID string: wrong format".to_string())?;
        self.uuid = Some(uuid);
        self.smbios.uuid = Some(uuid);
        Ok(())
    }

    /// `-smbios`: `smbios_entry_add()`, where `type=1,uuid=` sets `qemu_uuid` too.
    pub(crate) fn add_smbios(&mut self, arg: &str) -> std::result::Result<(), String> {
        let before = self.smbios.uuid;
        self.smbios.add(arg).map_err(|e| e.message().to_string())?;
        if self.smbios.uuid != before {
            self.uuid = self.smbios.uuid;
        }
        Ok(())
    }

    /// The firmware search path.
    pub(crate) fn firmware(&self) -> FirmwareSearch {
        FirmwareSearch::new(&self.data_dirs)
    }

    /// `qemu_create_default_devices()` for the serial port and the monitor: the old style
    /// strings the default `-serial` and `-monitor` get, if any. `no_serial` is the machine's
    /// `no_serial`, set for `none`. With `-nographic` both go to stdio, sharing a mux when
    /// both are wanted. Otherwise they would go to a virtual console, and in a build without
    /// one, like this one, the serial port gets `null` and the monitor nothing. ruvm has no
    /// parallel port, so the default one is left out.
    pub(crate) fn default_devices(&self, no_serial: bool) -> NographicDefaults {
        let serial = self.default_serial && !no_serial;
        if self.nographic {
            return nographic_defaults(serial, self.default_monitor, false);
        }
        NographicDefaults { serial: serial.then_some("null"), ..NographicDefaults::default() }
    }
}

/// The generic machine properties an x86 board takes, pulled out of the `-machine` options.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BoardOptions {
    /// `memory.size`, rounded up to 8 KiB like `machine_set_mem()` does.
    pub ram_size: Option<u64>,
    pub cpus: u32,
    pub max_cpus: u32,
    pub kernel: Option<KernelFiles>,
    /// `firmware`, which `-bios` sets.
    pub bios: Option<String>,
    /// `pflash0` and `pflash1` (q35): the names of the drives for the system flashes.
    pub pflash: [Option<String>; 2],
    /// The topology `-smp` resolved to, for the SMBIOS tables.
    pub topology: Option<SmbiosTopology>,
    /// Everything else, for the board, in command line order.
    pub props: Vec<(String, String)>,
}

/// The largest `-smp maxcpus` of each board, `mc->max_cpus`.
fn board_max_cpus(kind: BoardKind) -> u32 {
    match kind {
        BoardKind::Microvm => 288,
        BoardKind::Q35 => 4096,
    }
}

fn board_type_name(kind: BoardKind) -> &'static str {
    match kind {
        BoardKind::Microvm => "microvm",
        BoardKind::Q35 => "pc-q35-11.1",
    }
}

/// Visits `name` of `machine` as a `T`, the way the machine property setter does.
fn visit_member<T: Visit>(machine: &QDict, name: &str) -> Result<Option<T>> {
    let Some(value) = machine.get(name) else { return Ok(None) };
    let mut root = QDict::new();
    root.put(name, value.clone());
    let mut v = QObjectInputVisitor::new_keyval(QValue::Dict(root));
    v.start_struct(None)?;
    let mut t = T::default();
    let r = T::visit(&mut v, Some(name), &mut t).and_then(|()| v.check_struct());
    v.end_struct();
    r.map(|()| Some(t))
}

/// [`parse_smp_topology`] without the topology: (cpus, maxcpus).
#[cfg(test)]
pub(crate) fn parse_smp(kind: BoardKind, config: &SMPConfiguration) -> Result<(u32, u32)> {
    parse_smp_topology(kind, config).map(|(cpus, max, _)| (cpus, max))
}

/// `machine_parse_smp_config()` for the x86 boards, which know dies and modules but not
/// clusters, books or drawers. Gives (cpus, maxcpus, topology).
pub(crate) fn parse_smp_topology(
    kind: BoardKind,
    config: &SMPConfiguration,
) -> Result<(u32, u32, SmbiosTopology)> {
    let explicit = [
        config.cpus,
        config.drawers,
        config.books,
        config.sockets,
        config.dies,
        config.clusters,
        config.modules,
        config.cores,
        config.threads,
        config.maxcpus,
    ];
    if explicit.iter().any(|v| matches!(v, Some(n) if *n <= 0)) {
        return Err(Error::generic(
            "Invalid CPU topology: CPU topology parameters must be greater than zero",
        ));
    }
    let get = |v: Option<i64>| v.map_or(0, |n| u64::try_from(n).unwrap_or(u64::MAX));
    for (value, name) in
        [(config.clusters, "clusters"), (config.books, "books"), (config.drawers, "drawers")]
    {
        if get(value) > 1 {
            return Err(Error::generic(format!(
                "{name} > 1 not supported by this machine's CPU topology"
            )));
        }
    }
    let cpus = get(config.cpus);
    let mut sockets = get(config.sockets);
    let dies = get(config.dies).max(1);
    let modules = get(config.modules).max(1);
    let mut cores = get(config.cores);
    let mut threads = get(config.threads);
    let mut maxcpus = get(config.maxcpus);

    if cpus == 0 && maxcpus == 0 {
        sockets = sockets.max(1);
        cores = cores.max(1);
        threads = threads.max(1);
    } else {
        if maxcpus == 0 {
            maxcpus = cpus;
        }
        // Cores are preferred over sockets since 6.2.
        if cores == 0 {
            sockets = sockets.max(1);
            threads = threads.max(1);
            cores = maxcpus / (sockets * dies * modules * threads);
        } else if sockets == 0 {
            threads = threads.max(1);
            sockets = maxcpus / (dies * modules * cores * threads);
        }
        if threads == 0 {
            threads = maxcpus / (sockets * dies * modules * cores);
        }
    }
    let total = sockets * dies * modules * cores * threads;
    if maxcpus == 0 {
        maxcpus = total;
    }
    let cpus = if cpus == 0 { maxcpus } else { cpus };
    let topo = format!(
        "sockets ({sockets}) * dies ({dies}) * modules ({modules}) * cores ({cores}) * threads ({threads})"
    );
    if total != maxcpus {
        return Err(Error::generic(format!(
            "Invalid CPU topology: product of the hierarchy must match maxcpus: {topo} != maxcpus ({maxcpus})"
        )));
    }
    if maxcpus < cpus {
        return Err(Error::generic(format!(
            "Invalid CPU topology: maxcpus must be equal to or greater than smp: {topo} == maxcpus ({maxcpus}) < smp_cpus ({cpus})"
        )));
    }
    let max = board_max_cpus(kind);
    if maxcpus > u64::from(max) {
        return Err(Error::generic(format!(
            "Invalid SMP CPUs {maxcpus}. The max CPUs supported by machine '{}' is {max}",
            board_type_name(kind)
        )));
    }
    // Every count is at most maxcpus now, which fits a u32.
    let topology = SmbiosTopology {
        sockets: sockets as u32,
        dies: dies as u32,
        clusters: 1,
        modules: modules as u32,
        cores: cores as u32,
        threads: threads as u32,
    };
    Ok((cpus as u32, maxcpus as u32, topology))
}

/// The keyval value of a `-machine` property as a string, the way the property parser sees
/// it.
fn prop_string(name: &str, value: &QValue) -> Result<String> {
    match value {
        QValue::Str(s) => Ok(s.clone()),
        _ => Err(Error::generic(format!("Parameter '{name}' is missing"))),
    }
}

/// `qemu_apply_machine_options()` for an x86 board: takes the generic properties out of
/// `machine` (whose `type`, `accel` and `kernel-irqchip` are gone already) and checks the
/// rest against the board, with the errors setting them would give.
pub(crate) fn take_board_options(kind: BoardKind, machine: &QDict) -> Result<BoardOptions> {
    let mut o = BoardOptions::default();
    if let Some(mem) = visit_member::<MemorySizeConfiguration>(machine, "memory")? {
        if mem.slots.is_some_and(|s| s != 0) || mem.max_size.is_some_and(|m| Some(m) != mem.size) {
            return Err(Error::generic("memory hotplug is not supported by ruvm yet"));
        }
        o.ram_size = mem.size.map(|s| s.next_multiple_of(8192));
    }
    let smp = visit_member::<SMPConfiguration>(machine, "smp")?.unwrap_or_default();
    let (cpus, max_cpus, topology) = parse_smp_topology(kind, &smp)?;
    (o.cpus, o.max_cpus, o.topology) = (cpus, max_cpus, Some(topology));

    let mut kernel = None;
    let mut initrd = None;
    let mut append = None;
    let mut warnings = Vec::new();
    for (name, value) in machine.iter_inserted() {
        match name {
            "memory" | "smp" => {}
            "kernel" => kernel = Some(prop_string(name, value)?),
            "initrd" => initrd = Some(prop_string(name, value)?),
            "append" => append = Some(prop_string(name, value)?),
            "firmware" => o.bios = Some(prop_string(name, value)?),
            // The link properties pc_pflash_create() aliases onto the machine.
            "pflash0" | "pflash1" if kind == BoardKind::Q35 => {
                o.pflash[usize::from(name == "pflash1")] = Some(prop_string(name, value)?);
            }
            // Generic machine properties that change nothing here.
            "dump-guest-core" | "mem-merge" => {}
            // Only q35 has a use for graphics=, but every machine has the property.
            "graphics" if kind == BoardKind::Microvm => {}
            _ => {
                let v = prop_string(name, value)?;
                match kind {
                    BoardKind::Microvm => MicrovmProps::default().set(name, &v),
                    BoardKind::Q35 => Q35Props::default().set(name, &v, &mut warnings),
                }
                .map_err(Error::generic)?;
                o.props.push((name.to_string(), v));
            }
        }
    }
    if let Some(kernel) = kernel.filter(|k| !k.is_empty()) {
        o.kernel = Some(KernelFiles {
            kernel,
            initrd: initrd.filter(|i| !i.is_empty()),
            append: append.unwrap_or_default(),
        });
    }
    Ok(o)
}

/// `-drive if=`, `BlockInterfaceType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DriveIf {
    None,
    Ide,
    Scsi,
    Floppy,
    Pflash,
    Mtd,
    Sd,
    Virtio,
    Xen,
}

impl DriveIf {
    const ALL: [DriveIf; 9] = [
        DriveIf::None,
        DriveIf::Ide,
        DriveIf::Scsi,
        DriveIf::Floppy,
        DriveIf::Pflash,
        DriveIf::Mtd,
        DriveIf::Sd,
        DriveIf::Virtio,
        DriveIf::Xen,
    ];

    /// `if_name[]`.
    pub(crate) fn name(self) -> &'static str {
        match self {
            DriveIf::None => "none",
            DriveIf::Ide => "ide",
            DriveIf::Scsi => "scsi",
            DriveIf::Floppy => "floppy",
            DriveIf::Pflash => "pflash",
            DriveIf::Mtd => "mtd",
            DriveIf::Sd => "sd",
            DriveIf::Virtio => "virtio",
            DriveIf::Xen => "xen",
        }
    }
}

/// `block_default_type` and `units_per_default_bus` of the machine, `None` being the `none`
/// machine.
fn block_defaults(kind: Option<BoardKind>) -> (DriveIf, u32) {
    match kind {
        Some(BoardKind::Q35) => (DriveIf::Ide, 1),
        Some(BoardKind::Microvm) => (DriveIf::None, 1),
        None => (DriveIf::None, 0),
    }
}

/// `if_max_devs[]` after `override_max_devs()`.
fn max_devs(kind: Option<BoardKind>, iface: DriveIf) -> u32 {
    let (default_if, units) = block_defaults(kind);
    if iface == default_if && units > 0 {
        return units;
    }
    match iface {
        DriveIf::Ide => 2,
        DriveIf::Scsi => 7,
        _ => 0,
    }
}

/// One `-drive`, after `drive_new()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Drive {
    pub id: String,
    pub file: Option<String>,
    pub iface: DriveIf,
    pub bus: u32,
    pub unit: u32,
    pub read_only: bool,
    pub cdrom: bool,
    /// No `format=`: QEMU would probe, ruvm takes the file as raw and says so.
    pub probed: bool,
    pub loc: Option<Location>,
}

fn opt_number(opts: &ruvm_qapi::opts::QemuOpts, name: &str) -> Result<Option<u64>> {
    match opts.get(name) {
        None => Ok(None),
        Some(v) => ruvm_qapi::cutils::parse_uint(v, 0, true)
            .map(|(n, _)| Some(n))
            .map_err(|_| Error::generic(format!("Parameter '{name}' expects a number"))),
    }
}

fn opt_bool(opts: &ruvm_qapi::opts::QemuOpts, name: &str) -> Result<Option<bool>> {
    match opts.get(name) {
        None => Ok(None),
        Some("on" | "yes" | "true" | "y") => Ok(Some(true)),
        Some("off" | "no" | "false" | "n") => Ok(Some(false)),
        Some(_) => Err(Error::generic(format!("Parameter '{name}' expects 'on' or 'off'"))),
    }
}

/// The `-drive` options ruvm takes but has no use for.
const DRIVE_IGNORED: &[&str] =
    &["cache", "aio", "discard", "detect-zeroes", "werror", "rerror", "copy-on-read"];

/// `drive_new()`: parses one `-drive` for a machine of kind `kind` (`None` for `none`),
/// given the drives before it.
pub(crate) fn parse_drive(
    list: &mut QemuOptsList,
    arg: &str,
    kind: Option<BoardKind>,
    earlier: &[Drive],
    loc: Option<Location>,
) -> Result<Drive> {
    let opts = list.parse(arg, false)?;
    for (name, _) in opts.iter() {
        let known = matches!(
            name,
            "file"
                | "if"
                | "index"
                | "bus"
                | "unit"
                | "media"
                | "format"
                | "readonly"
                | "read-only"
                | "snapshot"
        ) || DRIVE_IGNORED.contains(&name);
        if !known {
            return Err(Error::generic(format!(
                "Block format 'raw' does not support the option '{name}'"
            )));
        }
    }
    let mut probed = true;
    if let Some(fmt) = opts.get("format") {
        if is_help_option(fmt) {
            return Err(Error::generic("format=help is not supported by ruvm yet"));
        }
        if fmt != "raw" {
            return Err(Error::generic(format!("format '{fmt}' is not supported by ruvm yet")));
        }
        probed = false;
    }
    if opt_bool(opts, "snapshot")? == Some(true) {
        return Err(Error::generic("snapshot=on is not supported by ruvm yet"));
    }
    let mut cdrom = false;
    let mut read_only = false;
    match opts.get("media") {
        None | Some("disk") => {}
        Some("cdrom") => {
            cdrom = true;
            read_only = true;
        }
        Some(v) => return Err(Error::generic(format!("'{v}' invalid media"))),
    }
    read_only |= opt_bool(opts, "read-only")?.or(opt_bool(opts, "readonly")?).unwrap_or(false);

    let iface = match opts.get("if") {
        None => block_defaults(kind).0,
        Some(v) => *DriveIf::ALL
            .iter()
            .find(|i| i.name() == v)
            .ok_or_else(|| Error::generic(format!("unsupported bus type '{v}'")))?,
    };

    let bus_given = opt_number(opts, "bus")?;
    let unit_given = opt_number(opts, "unit")?;
    let index = opt_number(opts, "index")?;
    let max = max_devs(kind, iface);
    let mut bus = bus_given.unwrap_or(0) as u32;
    let mut unit = unit_given.map(|u| u as u32);
    if let Some(index) = index {
        if bus_given.is_some_and(|b| b != 0) || unit_given.is_some() {
            return Err(Error::generic("index cannot be used with bus and unit"));
        }
        let index = index as u32;
        (bus, unit) = match (index.checked_div(max), index.checked_rem(max)) {
            (Some(bus), Some(unit)) => (bus, Some(unit)),
            _ => (0, Some(index)),
        };
    }
    let taken = |bus: u32, unit: u32| {
        earlier.iter().any(|d| d.iface == iface && d.bus == bus && d.unit == unit)
    };
    let unit = match unit {
        Some(u) => u,
        None => {
            let mut u = 0;
            while taken(bus, u) {
                u += 1;
                if max != 0 && u >= max {
                    u -= max;
                    bus += 1;
                }
            }
            u
        }
    };
    if max != 0 && unit >= max {
        return Err(Error::generic(format!("unit {unit} too big (max is {})", max - 1)));
    }
    if taken(bus, unit) {
        let index = index.map_or(-1, |i| i as i64);
        return Err(Error::generic(format!(
            "drive with bus={bus}, unit={unit} (index={index}) exists"
        )));
    }

    let id = match opts.id() {
        Some(id) => id.to_string(),
        None => {
            let media = match iface {
                DriveIf::Ide | DriveIf::Scsi if cdrom => "-cd",
                DriveIf::Ide | DriveIf::Scsi => "-hd",
                _ => "",
            };
            let id = if max != 0 {
                format!("{}{bus}{media}{unit}", iface.name())
            } else {
                format!("{}{media}{unit}", iface.name())
            };
            opts.set_id(Some(id.clone()));
            id
        }
    };
    let file = opts.get("file").filter(|f| !f.is_empty()).map(str::to_string);
    Ok(Drive { id, file, iface, bus, unit, read_only, cdrom, probed, loc })
}

/// The `-drive` option list: anything goes, with an id.
pub(crate) fn drive_opts() -> QemuOptsList {
    QemuOptsList::new("drive", &[])
}

/// The virtio device models `-device` knows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VirtioModel {
    Blk,
    Rng,
    Serial,
}

/// How a virtio device reaches the guest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Transport {
    /// A `virtio-*-device` on a virtio-mmio transport, `virtio-bus`.
    Mmio,
    /// A `virtio-*-pci` function.
    Pci,
}

/// The device types, with the bus they plug into.
const DEVICE_TYPES: &[(&str, VirtioModel, Transport)] = &[
    ("virtio-blk-device", VirtioModel::Blk, Transport::Mmio),
    ("virtio-blk-pci", VirtioModel::Blk, Transport::Pci),
    ("virtio-rng-device", VirtioModel::Rng, Transport::Mmio),
    ("virtio-rng-pci", VirtioModel::Rng, Transport::Pci),
    ("virtio-serial-device", VirtioModel::Serial, Transport::Mmio),
    ("virtio-serial-pci", VirtioModel::Serial, Transport::Pci),
];

/// `qdev_alias_table[]` for x86, where the default virtio transport is PCI.
const DEVICE_ALIASES: &[(&str, &str)] = &[
    ("virtio-blk", "virtio-blk-pci"),
    ("virtio-rng", "virtio-rng-pci"),
    ("virtio-serial", "virtio-serial-pci"),
];

/// A virtio device to plug.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Plug {
    /// The type name, after aliases.
    pub typename: &'static str,
    pub model: VirtioModel,
    pub transport: Transport,
    /// Index into the drives, for virtio-blk.
    pub drive: Option<usize>,
    /// virtio-blk `serial`.
    pub serial: Option<String>,
    pub loc: Option<Location>,
}

/// What the board gets from `-drive` and `-device`, in the order QEMU plugs it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Plan {
    /// (AHCI port, drive index) for the `if=ide` drives the board claims.
    pub ide: Vec<(u32, usize)>,
    /// The empty CD-ROM drive q35 gets by default at IDE index 2.
    pub default_cdrom: Option<u32>,
    /// The virtio devices, `-device` first and then the ones `-drive if=virtio` adds.
    pub virtio: Vec<Plug>,
}

/// The drives of the q35 system flashes: `-machine pflashN=` names a drive by id, and
/// `-drive if=pflash` with unit N is taken the way `pflash_cfi01_legacy_drive()` does. The
/// images are read into the flash and written back with plain file I/O, so the drive only
/// supplies a file name and the read-only flag.
pub(crate) fn pflash_drives(
    kind: BoardKind,
    opts: &BoardOptions,
    drives: &[Drive],
) -> std::result::Result<[Option<PflashDrive>; 2], Located> {
    let mut out: [Option<PflashDrive>; 2] = [None, None];
    if kind != BoardKind::Q35 {
        return Ok(out);
    }
    let open = |d: &Drive| -> std::result::Result<PflashDrive, Located> {
        let Some(file) = &d.file else {
            // No medium: blk_getlength() fails and the size check catches it.
            return Ok(PflashDrive { name: d.id.clone(), size: 0, backing: PflashBacking::None });
        };
        let size =
            FileBackend::open(file, d.read_only).map_err(|e| Located::new(&d.loc, e))?.size();
        Ok(PflashDrive {
            name: d.id.clone(),
            size: raw_block_length(size),
            backing: PflashBacking::File { path: file.into(), read_only: d.read_only },
        })
    };
    for (i, name) in opts.pflash.iter().enumerate() {
        let Some(name) = name else { continue };
        let d = drives.iter().find(|d| &d.id == name).ok_or_else(|| {
            Located::bare(format!("Property 'cfi.pflash01.drive' can't find value '{name}'"))
        })?;
        out[i] = Some(open(d)?);
    }
    for d in drives.iter().filter(|d| d.iface == DriveIf::Pflash && d.bus == 0 && d.unit < 2) {
        let i = d.unit as usize;
        if out[i].is_some() {
            return Err(Located::new(&d.loc, "clashes with -machine"));
        }
        out[i] = Some(open(d)?);
    }
    Ok(out)
}

/// Works out what `-drive` and `-device` plug into a board of kind `kind` (`None` for the
/// `none` machine). The errors are those of `qdev_device_add()` and
/// `drive_check_orphaned()`; for orphans every one is listed.
pub(crate) fn plan(
    kind: Option<BoardKind>,
    drives: &[Drive],
    devices: &[(String, Option<Location>)],
    has_defaults: bool,
) -> std::result::Result<Plan, Vec<Located>> {
    let mut p = Plan::default();
    let mut used: HashSet<usize> = HashSet::new();

    // The board claims its IDE drives while it is built: q35 has one unit on each of the six
    // AHCI ports.
    if kind == Some(BoardKind::Q35) {
        for (i, d) in drives.iter().enumerate() {
            if d.iface == DriveIf::Ide && d.unit == 0 && d.bus < 6 {
                p.ide.push((d.bus, i));
                used.insert(i);
            }
        }
        // pc_system_firmware_init(): drive_get(IF_PFLASH, 0, i) for the two flashes.
        for (i, d) in drives.iter().enumerate() {
            if d.iface == DriveIf::Pflash && d.bus == 0 && d.unit < 2 {
                used.insert(i);
            }
        }
        // default_drive(default_cdrom, ..., IF_IDE, 2, CDROM_OPTS)
        let taken = drives.iter().any(|d| d.iface == DriveIf::Ide && d.bus == 2 && d.unit == 0);
        if has_defaults && !taken {
            p.default_cdrom = Some(2);
        }
    }

    let mut queue: Vec<(String, Option<Location>)> = devices.to_vec();
    // drive_new() adds a -device virtio-blk for every if=virtio drive, after the others.
    for d in drives.iter().filter(|d| d.iface == DriveIf::Virtio) {
        queue.push((format!("virtio-blk,drive={}", d.id), d.loc.clone()));
    }
    for (arg, loc) in &queue {
        let plug = plan_device(kind, drives, &mut used, arg, loc).map_err(|e| vec![e])?;
        p.virtio.push(plug);
    }

    let orphans: Vec<Located> = drives
        .iter()
        .enumerate()
        .filter(|(i, d)| {
            !used.contains(i) && !matches!(d.iface, DriveIf::None | DriveIf::Virtio | DriveIf::Xen)
        })
        .map(|(_, d)| {
            Located::new(
                &d.loc,
                format!(
                    "machine type does not support if={},bus={},unit={}",
                    d.iface.name(),
                    d.bus,
                    d.unit
                ),
            )
        })
        .collect();
    if !orphans.is_empty() {
        return Err(orphans);
    }
    Ok(p)
}

fn plan_device(
    kind: Option<BoardKind>,
    drives: &[Drive],
    used: &mut HashSet<usize>,
    arg: &str,
    loc: &Option<Location>,
) -> std::result::Result<Plug, Located> {
    let mut list = QemuOptsList::new("device", &[]).with_implied_opt_name("driver");
    let opts = list.parse(arg, true).map_err(|e| Located(loc.clone(), e))?;
    let Some(driver) = opts.get("driver") else {
        return Err(Located::new(loc, "Parameter 'driver' is missing"));
    };
    if is_help_option(driver) || opts.has_help_opt() {
        return Err(Located::new(loc, "-device help is not supported by ruvm yet"));
    }
    let alias = DEVICE_ALIASES.iter().find(|(a, _)| *a == driver).map(|(_, t)| *t);
    let name = alias.unwrap_or(driver);
    let Some(&(typename, model, transport)) = DEVICE_TYPES.iter().find(|(t, ..)| *t == name) else {
        return Err(Located::new(loc, format!("'{driver}' is not a valid device model name")));
    };
    let mut drive = None;
    let mut serial = None;
    let mut bus = None;
    for (k, v) in opts.iter() {
        match k {
            "driver" => {}
            "bus" => bus = Some(v.to_string()),
            "drive" if model == VirtioModel::Blk => {
                let Some(i) = drives.iter().position(|d| d.id == v) else {
                    return Err(Located::new(
                        loc,
                        format!("Property '{typename}.drive' can't find value '{v}'"),
                    ));
                };
                drive = Some(i);
            }
            "serial" if model == VirtioModel::Blk => serial = Some(v.to_string()),
            _ => {
                return Err(Located::new(loc, format!("Property '{typename}.{k}' not found")));
            }
        }
    }
    // qdev_device_add() finds the bus before the device is realized.
    let bus_type = match transport {
        Transport::Mmio => "virtio-bus",
        Transport::Pci => "PCI",
    };
    let has_bus = matches!(
        (kind, transport),
        (Some(BoardKind::Microvm), Transport::Mmio) | (Some(BoardKind::Q35), Transport::Pci)
    );
    match bus {
        Some(b)
            if !(kind == Some(BoardKind::Q35) && transport == Transport::Pci && b == "pcie.0") =>
        {
            return Err(Located::new(loc, format!("Bus '{b}' not found")));
        }
        _ => {}
    }
    if !has_bus {
        return Err(Located::new(
            loc,
            format!("No '{bus_type}' bus found for device '{typename}'"),
        ));
    }
    if model == VirtioModel::Blk {
        let Some(i) = drive else {
            return Err(Located::new(loc, "drive property not set"));
        };
        if !used.insert(i) {
            let d = &drives[i];
            let msg = if d.iface == DriveIf::None || d.iface == DriveIf::Virtio {
                format!("Drive '{}' is already in use by another device", d.id)
            } else {
                format!(
                    "Drive '{}' is already in use because it has been automatically connected to another device (did you need 'if=none' in the drive options?)",
                    d.id
                )
            };
            return Err(Located::new(loc, msg));
        }
        if drives[i].file.is_none() {
            return Err(Located::new(loc, "Device needs media, but drive is empty"));
        }
    }
    Ok(Plug { typename, model, transport, drive, serial, loc: loc.clone() })
}

/// QEMU's warning for a drive without `format=`, which it would probe and find raw.
pub(crate) fn probe_warning(file: &str) -> String {
    format!(
        "WARNING: Image format was not specified for '{file}' and probing guessed raw.\n         Automatically detecting the format is dangerous for raw images, write operations on block 0 will be restricted.\n         Specify the 'raw' format explicitly to remove the restrictions.\n"
    )
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub(crate) use kvm::{Running, kvm_init, start_board};

/// The accelerator `configure_accelerators()` picked.
#[derive(Debug)]
pub(crate) enum Accel {
    Qtest,
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    Kvm(Box<ruvm_accel_kvm::KvmAccel>),
}

/// Why an accelerator did not come up.
#[derive(Debug)]
pub(crate) enum AccelInitError {
    /// A bad property, which QEMU treats as fatal.
    Fatal(Error),
    /// `init_machine()` failed; the lines to print before trying the next accelerator.
    #[cfg_attr(not(all(target_os = "linux", target_arch = "x86_64")), allow(dead_code))]
    Failed([String; 2]),
}

/// The KVM side, Linux on x86_64 only.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod kvm {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use ruvm_accel_kvm::{KernelIrqchip, KvmAccel, KvmOptions};
    use ruvm_base::report::warn_report;
    use ruvm_base::{ClockType, Error};
    use ruvm_chardev::{Attachment, Chardev, Connection, Frontend};
    use ruvm_hw_char::serial::{Serial, SerialBackend};
    use ruvm_hw_core::Clock;
    use ruvm_hw_core::timer::TimeSource;
    use ruvm_hw_storage::DriveConfig;
    use ruvm_hw_virtio::{
        RandomFile, VirtioBlk, VirtioBlkConf, VirtioConsole, VirtioDeviceClass, VirtioRng,
        VirtioRngConf,
    };
    use ruvm_machine_x86::kvm_run::{
        CpuModel, GuestEvent, KvmMachine, KvmRunConfig, ShutdownReason, open_accel, pit_in_kernel,
    };
    use ruvm_machine_x86::{BoardKind, BoardSpec, FileBackend, build_board};
    use ruvm_qapi::events::{event_guest_panicked, event_reset};
    use ruvm_qapi::types::{GuestPanicAction, GuestPanickedArg, ResetArg, RunState, ShutdownCause};

    use super::{
        AccelInitError, BoardOptions, Cmdline, Drive, Located, VirtioModel, pflash_drives, plan,
        probe_warning,
    };
    use crate::vl::Vm;

    /// `do_configure_accelerator()` for kvm: the `-accel kvm` properties, with
    /// `-machine kernel-irqchip=` as the default for the property of that name.
    pub(crate) fn kvm_init(
        props: &[(String, String)],
        irqchip_sugar: Option<&str>,
        default_split: bool,
    ) -> Result<KvmAccel, AccelInitError> {
        let mut opts = KvmOptions::default();
        let parse =
            |v: &str| KernelIrqchip::parse(v).map_err(|e| AccelInitError::Fatal(Error::generic(e)));
        if let Some(v) = irqchip_sugar {
            opts.kernel_irqchip = Some(parse(v)?);
        }
        for (k, v) in props {
            match k.as_str() {
                "kernel-irqchip" => opts.kernel_irqchip = Some(parse(v)?),
                "device" => opts.device = Some(v.into()),
                _ => {
                    return Err(AccelInitError::Fatal(Error::generic(format!(
                        "Property 'kvm-accel.{k}' not found"
                    ))));
                }
            }
        }
        open_accel(&opts, default_split).map_err(AccelInitError::Failed)
    }

    /// The x86 board running on KVM, and the chardevs its serial ports are attached to.
    #[derive(Debug)]
    pub(crate) struct Running {
        machine: Arc<KvmMachine>,
        _serials: Vec<Attachment>,
    }

    impl Running {
        /// Stops the vCPU and timer threads.
        pub(crate) fn quit(&self) {
            self.machine.quit();
        }
    }

    /// Feeds `input` to `serial`, as much at a time as the port has room for.
    fn feed(serial: &Serial, mut input: &[u8]) {
        while !input.is_empty() {
            let room = serial.can_receive();
            if room == 0 {
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            let k = room.min(input.len());
            serial.receive(&input[..k]);
            input = &input[k..];
        }
    }

    /// A serial port on its chardev, the `chardev` property of `isa-serial`: what the guest
    /// writes goes to the chardev, what the chardev reads goes to the port.
    struct ChardevSerial {
        serial: Arc<Serial>,
        chr: Arc<Chardev>,
    }

    impl SerialBackend for ChardevSerial {
        fn write(&self, bytes: &[u8]) -> usize {
            // Output nobody can take is dropped, as serial_xmit() does after an error.
            let _ = self.chr.write_all(bytes);
            bytes.len()
        }
    }

    impl Frontend for ChardevSerial {
        fn serve(&self, conn: &mut Connection) -> std::io::Result<()> {
            let mut buf = [0u8; 256];
            loop {
                match conn.recv(&mut buf) {
                    Ok(0) => return Ok(()),
                    Ok(n) => feed(&self.serial, &buf[..n]),
                    Err(e) => return Err(e),
                }
            }
        }
    }

    fn virtio_class(
        plug_model: VirtioModel,
        drive: Option<&Drive>,
        serial: Option<String>,
    ) -> Result<Box<dyn VirtioDeviceClass>, String> {
        Ok(match plug_model {
            VirtioModel::Blk => {
                let d = drive.expect("planned with a drive");
                let file = d.file.as_deref().expect("planned with media");
                let backend = FileBackend::open(file, d.read_only)?;
                if d.probed && !d.read_only {
                    eprint!("{}", probe_warning(file));
                }
                let conf = VirtioBlkConf { serial, ..VirtioBlkConf::default() };
                Box::new(VirtioBlk::new(Box::new(backend), conf))
            }
            VirtioModel::Rng => {
                Box::new(VirtioRng::new(Box::<RandomFile>::default(), VirtioRngConf::default()))
            }
            VirtioModel::Serial => Box::new(VirtioConsole::new(None)),
        })
    }

    /// `qemu_init_board()`, `qemu_create_cli_devices()` and `qemu_machine_creation_done()`
    /// for an x86 board on KVM: builds the board, plugs the devices, connects the serial ports
    /// to `serial_hds` (`serial_hd(i)` by index) and puts it all on the vCPUs, stopped until
    /// `vm_start()`.
    pub(crate) fn start_board(
        vm: &Arc<Vm>,
        accel: KvmAccel,
        kind: BoardKind,
        opts: BoardOptions,
        cmd: &Cmdline,
        drives: &[Drive],
        serial_hds: &[Option<Arc<Chardev>>],
    ) -> Result<Running, Vec<Located>> {
        let one = |e: Located| vec![e];
        let cpu = CpuModel::new(&accel, cmd.cpu.as_deref()).map_err(|e| one(Located::bare(e)))?;
        for w in cpu.warnings() {
            warn_report(w);
        }
        let clock = Clock::new(ClockType::Virtual, TimeSource::Monotonic(Instant::now()));
        let rtc_clock = Clock::new(ClockType::Host, TimeSource::Wall);
        let pflash = pflash_drives(kind, &opts, drives).map_err(one)?;
        let spec = BoardSpec {
            kind,
            props: opts.props,
            ram_size: opts.ram_size,
            cpus: opts.cpus,
            max_cpus: opts.max_cpus,
            kvm: true,
            pit_in_kernel: pit_in_kernel(&accel),
            smm_available: false,
            phys_bits: cpu.phys_bits(),
            cpu: cpu.ident(),
            bios: opts.bios,
            pflash,
            uuid: cmd.uuid,
            smbios: cmd.smbios.clone(),
            topology: opts.topology,
            kernel: opts.kernel,
            firmware: cmd.firmware(),
            serial_hds: serial_hds.iter().map(Option::is_some).collect(),
            clock: Arc::clone(&clock),
            rtc_clock: Arc::clone(&rtc_clock),
        };
        let (mut board, warnings) = build_board(spec).map_err(|e| one(Located::bare(e)))?;
        for w in warnings.iter().chain(board.warnings()) {
            warn_report(w);
        }

        let p = plan(Some(kind), drives, &cmd.devices, cmd.has_defaults)?;
        for &(port, i) in &p.ide {
            let d = &drives[i];
            let blk: Option<Arc<dyn ruvm_hw_storage::BlockBackend>> = match &d.file {
                Some(f) => {
                    let b = FileBackend::open(f, d.read_only)
                        .map_err(|e| one(Located::new(&d.loc, e)))?;
                    if d.probed && !d.read_only {
                        eprint!("{}", probe_warning(f));
                    }
                    Some(Arc::new(b))
                }
                None => None,
            };
            let config = if d.cdrom { DriveConfig::cdrom() } else { DriveConfig::hd() };
            if !d.cdrom && blk.is_none() {
                return Err(one(Located::new(&d.loc, "Device needs media, but drive is empty")));
            }
            attach_ide(&board, port, config, blk).map_err(|e| one(Located::new(&d.loc, e)))?;
        }
        if let Some(port) = p.default_cdrom {
            attach_ide(&board, port, DriveConfig::cdrom(), None)
                .map_err(|e| one(Located::bare(e)))?;
        }
        for plug in p.virtio {
            let drive = plug.drive.map(|i| &drives[i]);
            let class = virtio_class(plug.model, drive, plug.serial)
                .map_err(|e| one(Located::new(drive.map_or(&plug.loc, |d| &d.loc), e)))?;
            board.attach_virtio(class).map_err(|e| {
                let msg = if e.starts_with("No 'virtio-bus' bus found for device") {
                    format!("No 'virtio-bus' bus found for device '{}'", plug.typename)
                } else {
                    e
                };
                one(Located::new(&plug.loc, msg))
            })?;
        }

        // The ports the board made take their chardevs; a microvm leaves the ones after the
        // first unconnected, as QEMU does.
        let mut attachments = Vec::new();
        for (index, chr) in serial_hds.iter().enumerate() {
            let (Some(chr), Some(port)) = (chr, board.serial(index)) else { continue };
            let fe = Arc::new(ChardevSerial { serial: Arc::clone(port), chr: Arc::clone(chr) });
            board.set_serial_backend(index, Some(fe.clone()));
            attachments.push(chr.attach(fe).map_err(|e| vec![Located(None, e)])?);
        }

        let rs = Arc::clone(&vm.runstate);
        let qmp = Arc::clone(&vm.qmp);
        let handler = Arc::new(move |e: GuestEvent| match e {
            GuestEvent::Shutdown(r) => rs.shutdown_request(match r {
                ShutdownReason::GuestShutdown => ShutdownCause::GuestShutdown,
                ShutdownReason::GuestReset => ShutdownCause::GuestReset,
            }),
            GuestEvent::Reset => {
                let arg = ResetArg { guest: true, reason: ShutdownCause::GuestReset };
                if let Some(ev) = event_reset(&qmp.policy(), arg) {
                    qmp.emit_event(ev);
                }
            }
            GuestEvent::Panicked => {
                // The default panic action is shutdown: say so, stop and quit.
                let arg = GuestPanickedArg { action: GuestPanicAction::Poweroff, info: None };
                if let Some(ev) = event_guest_panicked(&qmp.policy(), arg) {
                    qmp.emit_event(ev);
                }
                rs.vm_stop(RunState::GuestPanicked);
                rs.shutdown_request(ShutdownCause::GuestPanic);
            }
            GuestEvent::InternalError(msg) => {
                // kvm_cpu_exec() prints these with fprintf(), without the program name.
                eprintln!("{msg}");
                rs.vm_stop(RunState::InternalError);
            }
        });
        let cfg = KvmRunConfig { no_reboot: cmd.no_reboot };
        let machine = KvmMachine::new(accel, board, &cpu, vec![clock, rtc_clock], &cfg, handler)
            .map_err(|e| one(Located::bare(e)))?;
        let machine = Arc::new(machine);
        let weak = Arc::downgrade(&machine);
        vm.runstate.set_cpu_hook(Some(Arc::new(move |run| {
            if let Some(m) = weak.upgrade() {
                if run {
                    m.start();
                } else {
                    m.pause();
                }
            }
        })));
        Ok(Running { machine, _serials: attachments })
    }

    fn attach_ide(
        board: &ruvm_machine_x86::X86Board,
        port: u32,
        config: DriveConfig,
        blk: Option<Arc<dyn ruvm_hw_storage::BlockBackend>>,
    ) -> Result<(), String> {
        match board {
            ruvm_machine_x86::X86Board::Q35(m, _) => m.attach_drive(port as usize, config, blk),
            ruvm_machine_x86::X86Board::Microvm(_) => {
                Err(format!("machine type does not support if=ide,bus={port},unit=0"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvm_qapi::keyval::keyval_parse;

    fn machine(arg: &str) -> QDict {
        keyval_parse(arg, Some("type"), None).unwrap()
    }

    #[test]
    fn help_lists_the_alias_before_the_board() {
        let lines = machine_help_lines();
        let q35 = lines.iter().find(|(n, _)| n == "pc-q35-11.1").unwrap();
        let mut it = q35.1.lines();
        assert!(it.next().unwrap().starts_with("q35                  "));
        assert!(it.next().unwrap().starts_with("pc-q35-11.1          "));
        assert!(q35.1.contains("(alias of pc-q35-11.1)"));
        assert!(lines.iter().any(|(n, l)| n == "microvm" && l.starts_with("microvm     ")));
    }

    #[test]
    fn machine_options_split_into_generic_and_board_ones() {
        let mut m = machine("microvm,pit=off,x-option-roms=off,graphics=off,dump-guest-core=on");
        m.remove("type");
        let mut mem = QDict::new();
        mem.put("size", "256M");
        m.put("memory", mem);
        let mut smp = QDict::new();
        smp.put("cpus", "2");
        m.put("smp", smp);
        m.put("kernel", "bzImage");
        m.put("append", "console=ttyS0");
        m.put("firmware", "bios.bin");
        let o = take_board_options(BoardKind::Microvm, &m).unwrap();
        assert_eq!(o.ram_size, Some(256 << 20));
        assert_eq!((o.cpus, o.max_cpus), (2, 2));
        assert_eq!(o.bios.as_deref(), Some("bios.bin"));
        let k = o.kernel.unwrap();
        assert_eq!(
            (k.kernel.as_str(), k.initrd, k.append.as_str()),
            ("bzImage", None, "console=ttyS0")
        );
        assert_eq!(
            o.props,
            vec![("pit".into(), "off".into()), ("x-option-roms".into(), "off".into())]
        );

        let o = take_board_options(BoardKind::Q35, &QDict::new()).unwrap();
        let one = Some(SmbiosTopology::default());
        assert_eq!(o, BoardOptions { cpus: 1, max_cpus: 1, topology: one, ..Default::default() });
    }

    #[test]
    fn bad_machine_options_fail_like_qemu() {
        let mut m = QDict::new();
        m.put("bogus", "on");
        let e = take_board_options(BoardKind::Microvm, &m).unwrap_err();
        assert_eq!(e.message(), "Property 'microvm-machine.bogus' not found");
        let e = take_board_options(BoardKind::Q35, &m).unwrap_err();
        assert_eq!(e.message(), "Property 'pc-q35-11.1-machine.bogus' not found");

        let mut m = QDict::new();
        let mut mem = QDict::new();
        mem.put("size", "lots");
        m.put("memory", mem);
        let e = take_board_options(BoardKind::Q35, &m).unwrap_err();
        assert_eq!(e.message(), "Parameter 'memory.size' expects size");
    }

    fn smp(f: impl FnOnce(&mut SMPConfiguration)) -> SMPConfiguration {
        let mut s = SMPConfiguration::default();
        f(&mut s);
        s
    }

    #[test]
    fn smp_follows_machine_parse_smp_config() {
        let k = BoardKind::Q35;
        assert_eq!(parse_smp(k, &smp(|_| {})).unwrap(), (1, 1));
        assert_eq!(parse_smp(k, &smp(|s| s.cpus = Some(4))).unwrap(), (4, 4));
        assert_eq!(
            parse_smp(
                k,
                &smp(|s| {
                    s.cpus = Some(2);
                    s.maxcpus = Some(8);
                })
            )
            .unwrap(),
            (2, 8)
        );
        assert_eq!(
            parse_smp(
                k,
                &smp(|s| {
                    s.sockets = Some(2);
                    s.cores = Some(2);
                    s.threads = Some(2);
                })
            )
            .unwrap(),
            (8, 8)
        );
        let e = parse_smp(k, &smp(|s| s.cpus = Some(0))).unwrap_err();
        assert_eq!(
            e.message(),
            "Invalid CPU topology: CPU topology parameters must be greater than zero"
        );
        let e = parse_smp(
            k,
            &smp(|s| {
                s.cpus = Some(4);
                s.sockets = Some(3);
                s.cores = Some(1);
            }),
        )
        .unwrap_err();
        assert_eq!(
            e.message(),
            "Invalid CPU topology: product of the hierarchy must match maxcpus: sockets (3) * dies (1) * modules (1) * cores (1) * threads (1) != maxcpus (4)"
        );
        let e = parse_smp(BoardKind::Microvm, &smp(|s| s.cpus = Some(300))).unwrap_err();
        assert_eq!(
            e.message(),
            "Invalid SMP CPUs 300. The max CPUs supported by machine 'microvm' is 288"
        );
        let e = parse_smp(k, &smp(|s| s.clusters = Some(2))).unwrap_err();
        assert_eq!(e.message(), "clusters > 1 not supported by this machine's CPU topology");
    }

    #[test]
    fn default_devices_follow_nographic() {
        let d = |c: Cmdline, no_serial| {
            let d = c.default_devices(no_serial);
            (d.serial, d.monitor, d.parallel)
        };
        let ng = || Cmdline { nographic: true, ..Cmdline::default() };
        assert_eq!(d(ng(), false), (Some("mon:stdio"), None, None));
        // The none machine has no serial port, so the monitor has stdio to itself.
        assert_eq!(d(ng(), true), (None, Some("stdio"), None));
        assert_eq!(
            d(Cmdline { default_monitor: false, ..ng() }, false),
            (Some("stdio"), None, None)
        );
        assert_eq!(
            d(Cmdline { default_serial: false, ..ng() }, false),
            (None, Some("stdio"), None)
        );
        assert_eq!(d(Cmdline::default(), false), (Some("null"), None, None));
        assert_eq!(d(Cmdline::default(), true), (None, None, None));
        let nodefaults =
            Cmdline { default_serial: false, default_monitor: false, has_defaults: false, ..ng() };
        assert_eq!(d(nodefaults, false), (None, None, None));
    }

    fn drives(kind: Option<BoardKind>, args: &[&str]) -> Result<Vec<Drive>> {
        let mut list = drive_opts();
        let mut out = Vec::new();
        for a in args {
            let d = parse_drive(&mut list, a, kind, &out, None)?;
            out.push(d);
        }
        Ok(out)
    }

    #[test]
    fn drives_get_qemus_ids_and_units() {
        let q35 = Some(BoardKind::Q35);
        let d =
            drives(q35, &["file=a.img", "file=b.img,if=ide", "file=c.img,if=virtio,format=raw"])
                .unwrap();
        assert_eq!(d[0].id, "ide0-hd0");
        assert_eq!((d[0].bus, d[0].unit, d[0].iface), (0, 0, DriveIf::Ide));
        assert_eq!(d[1].id, "ide1-hd0");
        assert_eq!(d[1].bus, 1);
        assert!(d[1].probed);
        assert_eq!(d[2].id, "virtio0");
        assert!(!d[2].probed);

        let d = drives(Some(BoardKind::Microvm), &["file=a.img", "file=b.img", "id=x,file=c.img"])
            .unwrap();
        assert_eq!(d[0].id, "none00");
        assert_eq!(d[1].id, "none10");
        assert_eq!(d[2].id, "x");
        assert_eq!(d[0].iface, DriveIf::None);

        let d = drives(q35, &["file=a.img,index=2,media=cdrom"]).unwrap();
        assert_eq!((d[0].id.as_str(), d[0].bus, d[0].read_only), ("ide2-cd0", 2, true));
    }

    #[test]
    fn drive_errors_are_qemus() {
        let q35 = Some(BoardKind::Q35);
        let msg = |args: &[&str]| drives(q35, args).unwrap_err().message().to_string();
        assert_eq!(msg(&["file=a,if=usb"]), "unsupported bus type 'usb'");
        assert_eq!(msg(&["file=a,media=tape"]), "'tape' invalid media");
        assert_eq!(msg(&["file=a,index=1,unit=0"]), "index cannot be used with bus and unit");
        assert_eq!(msg(&["file=a,unit=1"]), "unit 1 too big (max is 0)");
        assert_eq!(
            msg(&["file=a,index=0", "file=b,index=0"]),
            "drive with bus=0, unit=0 (index=0) exists"
        );
        assert_eq!(msg(&["file=a,format=qcow2"]), "format 'qcow2' is not supported by ruvm yet");
        assert_eq!(msg(&["file=a,id=d", "file=b,id=d"]), "Duplicate ID 'd' for drive");
        assert_eq!(msg(&["file=a,index=x"]), "Parameter 'index' expects a number");
    }

    fn devices(args: &[&str]) -> Vec<(String, Option<Location>)> {
        args.iter().map(|a| (a.to_string(), None)).collect()
    }

    #[test]
    fn microvm_devices_go_on_virtio_mmio() {
        let k = Some(BoardKind::Microvm);
        let d = drives(k, &["id=d0,file=a.img,if=none,format=raw"]).unwrap();
        let p = plan(
            k,
            &d,
            &devices(&[
                "virtio-blk-device,drive=d0,serial=s",
                "virtio-rng-device",
                "virtio-serial-device",
            ]),
            true,
        )
        .unwrap();
        assert!(p.ide.is_empty());
        assert_eq!(p.default_cdrom, None);
        let got: Vec<_> = p.virtio.iter().map(|v| (v.typename, v.transport, v.drive)).collect();
        assert_eq!(
            got,
            vec![
                ("virtio-blk-device", Transport::Mmio, Some(0)),
                ("virtio-rng-device", Transport::Mmio, None),
                ("virtio-serial-device", Transport::Mmio, None),
            ]
        );
        assert_eq!(p.virtio[0].serial.as_deref(), Some("s"));
    }

    #[test]
    fn q35_devices_go_on_pci_and_ide() {
        let k = Some(BoardKind::Q35);
        let d = drives(k, &["file=a.img,format=raw", "file=b.img,if=virtio,format=raw"]).unwrap();
        let p = plan(k, &d, &devices(&["virtio-rng-pci", "virtio-blk"]), true);
        // virtio-blk is virtio-blk-pci, which needs a drive.
        let e = p.unwrap_err();
        assert_eq!(e[0].1.message(), "drive property not set");

        let p = plan(k, &d, &devices(&["virtio-rng", "virtio-serial"]), true).unwrap();
        assert_eq!(p.ide, vec![(0, 0)]);
        assert_eq!(p.default_cdrom, Some(2));
        let got: Vec<_> = p.virtio.iter().map(|v| (v.typename, v.drive)).collect();
        assert_eq!(
            got,
            vec![
                ("virtio-rng-pci", None),
                ("virtio-serial-pci", None),
                ("virtio-blk-pci", Some(1))
            ]
        );
        let p = plan(k, &d, &[], false).unwrap();
        assert_eq!(p.default_cdrom, None);
    }

    #[test]
    fn device_errors_are_qemus() {
        let m = Some(BoardKind::Microvm);
        let q = Some(BoardKind::Q35);
        let err = |kind, drives: &[Drive], args: &[&str]| {
            plan(kind, drives, &devices(args), true).unwrap_err()[0].1.message().to_string()
        };
        assert_eq!(err(m, &[], &["foo"]), "'foo' is not a valid device model name");
        assert_eq!(
            err(m, &[], &["virtio-rng-pci"]),
            "No 'PCI' bus found for device 'virtio-rng-pci'"
        );
        assert_eq!(err(m, &[], &["virtio-rng"]), "No 'PCI' bus found for device 'virtio-rng-pci'");
        assert_eq!(
            err(q, &[], &["virtio-blk-device"]),
            "No 'virtio-bus' bus found for device 'virtio-blk-device'"
        );
        assert_eq!(
            err(None, &[], &["virtio-rng-device"]),
            "No 'virtio-bus' bus found for device 'virtio-rng-device'"
        );
        assert_eq!(
            err(m, &[], &["virtio-blk-device,drive=nope"]),
            "Property 'virtio-blk-device.drive' can't find value 'nope'"
        );
        assert_eq!(
            err(m, &[], &["virtio-rng-device,foo=1"]),
            "Property 'virtio-rng-device.foo' not found"
        );
        assert_eq!(err(q, &[], &["virtio-rng-pci,bus=pci.1"]), "Bus 'pci.1' not found");

        let d = drives(m, &["id=d,file=a,if=none"]).unwrap();
        assert_eq!(
            err(m, &d, &["virtio-blk-device,drive=d", "virtio-blk-device,drive=d"]),
            "Drive 'd' is already in use by another device"
        );
        let d = drives(q, &["id=d,file=a"]).unwrap();
        assert!(err(q, &d, &["virtio-blk-pci,drive=d"]).starts_with(
            "Drive 'd' is already in use because it has been automatically connected"
        ));
        // if=virtio becomes virtio-blk-pci, which microvm has no bus for.
        let d = drives(m, &["file=a,if=virtio"]).unwrap();
        assert_eq!(err(m, &d, &[]), "No 'PCI' bus found for device 'virtio-blk-pci'");
    }

    #[test]
    fn unclaimed_drives_are_all_reported() {
        let m = Some(BoardKind::Microvm);
        let d = drives(m, &["file=a,if=ide", "file=b,if=scsi", "file=c"]).unwrap();
        let e = plan(m, &d, &[], true).unwrap_err();
        let msgs: Vec<String> = e.iter().map(|l| l.1.message().to_string()).collect();
        assert_eq!(
            msgs,
            vec![
                "machine type does not support if=ide,bus=0,unit=0",
                "machine type does not support if=scsi,bus=0,unit=0",
            ]
        );
        let q = Some(BoardKind::Q35);
        let d = drives(q, &["file=a,index=6"]).unwrap();
        let e = plan(q, &d, &[], true).unwrap_err();
        assert_eq!(e[0].1.message(), "machine type does not support if=ide,bus=6,unit=0");
    }

    #[test]
    fn the_probe_warning_is_qemus() {
        let w = probe_warning("disk.img");
        assert!(w.starts_with(
            "WARNING: Image format was not specified for 'disk.img' and probing guessed raw.\n"
        ));
        assert!(w.ends_with("Specify the 'raw' format explicitly to remove the restrictions.\n"));
    }

    #[test]
    fn smp_topology_is_what_smbios_describes() {
        let k = BoardKind::Q35;
        let (_, _, t) = parse_smp_topology(k, &smp(|s| s.cpus = Some(4))).unwrap();
        assert_eq!(t, SmbiosTopology { cores: 4, ..SmbiosTopology::default() });
        let (_, _, t) = parse_smp_topology(
            k,
            &smp(|s| {
                s.sockets = Some(2);
                s.threads = Some(2);
                s.cpus = Some(8);
            }),
        )
        .unwrap();
        assert_eq!((t.sockets, t.cores, t.threads), (2, 2, 2));
    }

    #[test]
    fn uuid_and_smbios_share_qemu_uuid() {
        let mut c = Cmdline::default();
        assert_eq!(c.set_uuid("nope").unwrap_err(), "failed to parse UUID string: wrong format");
        c.set_uuid("12345678-9abc-def0-1234-56789abcdef0").unwrap();
        assert_eq!(c.uuid.unwrap()[0], 0x12);
        assert_eq!(c.smbios.uuid, c.uuid);
        c.add_smbios("type=1,uuid=00000000-0000-0000-0000-000000000001").unwrap();
        assert_eq!(c.uuid.unwrap()[15], 1);
        // Options that leave the UUID alone keep the one -uuid gave.
        c.set_uuid("12345678-9abc-def0-1234-56789abcdef0").unwrap();
        c.add_smbios("type=1,product=P").unwrap();
        assert_eq!(c.uuid.unwrap()[0], 0x12);
        assert!(c.add_smbios("type=1,bogus=x").is_err());
    }

    #[test]
    fn pflash_comes_from_the_machine_or_if_pflash() {
        let dir = std::env::temp_dir().join(format!("ruvm-pflash-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let code = dir.join("code.fd");
        let vars = dir.join("vars.fd");
        std::fs::write(&code, vec![0u8; 0x1000]).unwrap();
        std::fs::write(&vars, vec![0u8; 1000]).unwrap();
        let (code, vars) = (code.to_str().unwrap(), vars.to_str().unwrap());
        let q35 = BoardKind::Q35;

        let d = drives(
            Some(q35),
            &[
                &format!("if=pflash,format=raw,readonly=on,file={code}"),
                &format!("if=pflash,format=raw,file={vars}"),
            ],
        )
        .unwrap();
        let f = pflash_drives(q35, &BoardOptions::default(), &d).unwrap();
        let f0 = f[0].as_ref().unwrap();
        assert_eq!((f0.name.as_str(), f0.size), ("pflash0", 0x1000));
        assert_eq!(f0.backing, PflashBacking::File { path: code.into(), read_only: true });
        // Raw images are whole 512 byte sectors long.
        assert_eq!(f[1].as_ref().unwrap().size, 1024);
        // The board claims both, so neither is an orphan.
        assert!(plan(Some(q35), &d, &[], false).is_ok());
        assert!(
            pflash_drives(BoardKind::Microvm, &BoardOptions::default(), &d).unwrap()[0].is_none()
        );
        assert!(plan(Some(BoardKind::Microvm), &d, &[], false).is_err());

        let d = drives(Some(q35), &[&format!("if=none,id=code,file={code}")]).unwrap();
        let mut m = QDict::new();
        m.put("pflash0", "code");
        let o = take_board_options(q35, &m).unwrap();
        assert_eq!(o.pflash, [Some("code".to_string()), None]);
        let f = pflash_drives(q35, &o, &d).unwrap();
        assert_eq!(f[0].as_ref().unwrap().name, "code");
        let e = pflash_drives(q35, &o, &[]).unwrap_err();
        assert_eq!(e.1.message(), "Property 'cfi.pflash01.drive' can't find value 'code'");

        let d = drives(
            Some(q35),
            &[&format!("if=none,id=code,file={code}"), &format!("if=pflash,file={code}")],
        )
        .unwrap();
        assert_eq!(pflash_drives(q35, &o, &d).unwrap_err().1.message(), "clashes with -machine");
        let d = drives(Some(q35), &["if=pflash,file=/nonexistent/ruvm.fd"]).unwrap();
        let e = pflash_drives(q35, &BoardOptions::default(), &d).unwrap_err();
        assert_eq!(
            e.1.message(),
            "Could not open '/nonexistent/ruvm.fd': No such file or directory"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
