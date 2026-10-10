// SPDX-License-Identifier: GPL-2.0-or-later

//! The Arm `virt` and `sbsa-ref` boards from the command line: what `-machine virt` or
//! `-machine sbsa-ref`, `-m`, `-smp`, `-cpu`, `-kernel`, `-initrd`, `-append`, `-dtb`, `-bios`,
//! `-drive`, `-device`, `-serial`, `-semihosting` and `-semihosting-config` turn into, and the
//! board running on TCG.
//!
//! `-drive` and `-device` plug virtio devices into the PCIe root bus or the virtio-mmio
//! transports, give the flashes their drives and, on sbsa-ref, give the AHCI ports theirs,
//! see [`devices`].
//!
//! On sbsa-ref, `serial_hd(0)` is the PL011 and `serial_hd(1)` and `serial_hd(2)` the two
//! secure UARTs. Without `-smp` it has four CPUs in one socket, cluster and core, as QEMU's
//! machine defaults give. Its CPU models are `cortex-a57`, `cortex-a72`, `neoverse-n1` and
//! `max`, and the default is `neoverse-n1`: QEMU's default `neoverse-n2` and `neoverse-v1`
//! are not modelled. It has no machine properties of its own, and semihosting fails with
//! "... is not supported by ruvm yet".
//!
//! `-serial` (or `-nographic`) connects `serial_hd(0)` to the PL011, and a second `-serial`
//! connects `serial_hd(1)` to the second PL011 (the secure one with `secure=on`). `-bios`
//! loads the firmware into the first flash. Semihosting writes its
//! console to the `chardev` of `-semihosting-config`, or to standard error without one, as
//! semihosting/console.c does, and SYS_EXIT ends ruvm with the guest's status.
//!
//! Deliberate differences from QEMU:
//!
//! - The default CPU of virt is `cortex-a57`, since the 32-bit `cortex-a15` QEMU picks for TCG
//!   is not modelled. The CPU models are `cortex-a57`, `cortex-a72`, `cortex-a76`, `neoverse-n1`
//!   and `max`; the other
//!   models virt accepts fail with "... is not supported by ruvm yet", and so do the CPU
//!   properties other than `sve-max-vq` and `pmu`.
//! - The machine properties are taken only where their value describes the board that exists:
//!   `gic-version=3`, `its`, `secure`, `virtualization`, `mte`, `ras=off`, `acpi`, `spcr`,
//!   `x-oem-id`, `x-oem-table-id`, `iommu`, `default-bus-bypass-iommu`, `msi` other than
//!   `gicv2m`, 32 virtio-mmio
//!   transports and the `highmem*` properties. Other values fail with "... is not supported by ruvm yet". The
//!   board behaves as with `dtb-randomness=off` whatever that property says.
//! - `-semihosting-config target=gdb` fails, since there is no gdbstub; `auto` and `native`
//!   both mean native.
//! - SYS_EXIT asks the main loop to quit with the guest's status (`shutdown_request` with the
//!   code) rather than calling `exit()` on the vCPU thread, so QMP clients see a SHUTDOWN
//!   event first.
//! - A vCPU waiting in SYS_READC or SYS_READ from the console blocks its thread rather than
//!   halting, so `stop` on the monitor waits until the console has input.
//! - `-bios` takes a path: the name is not looked up in the firmware directories as
//!   `qemu_find_file()` does.
//! - There is no default NIC: QEMU plugs a `virtio-net-pci` with user networking unless
//!   `-nodefaults` or a network option says otherwise.
//! - With `secure=on` the secure-only devices are in the one address space, visible to
//!   non-secure accesses too.

use std::collections::VecDeque;
use std::io::Write as _;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Duration, Instant};

use ruvm_accel::tcg::TcgOptions;
use ruvm_base::report::{Location, warn_report};
use ruvm_base::{ClockType, Error, Result};
use ruvm_chardev::{Attachment, Chardev, Chardevs, Connection, Frontend};
use ruvm_hw_char::pl011::Pl011;
use ruvm_hw_char::serial::SerialBackend;
use ruvm_hw_core::Clock;
use ruvm_hw_core::timer::TimeSource;
use ruvm_machine_arm::sbsa_ref::{SBSA_DEFAULT_CPUS, SbsaRefConfig, SbsaRefMachine};
use ruvm_machine_arm::tcg_run::{
    SbsaRefTcgMachine, ShutdownReason, TcgBoard, TcgMachine, VirtEvent, VirtEventHandler,
    VirtRunConfig, VirtTcgMachine,
};
use ruvm_machine_arm::virt::memmap::check_highmem_mmio_size;
use ruvm_machine_arm::virt::{CpuTopology, Highmem, VirtConfig, VirtIommu, VirtMachine, VirtMsi};
use ruvm_qapi::events::event_reset;
use ruvm_qapi::opts::{QemuOptDesc, QemuOptType, QemuOptsList};
use ruvm_qapi::types::{
    MemorySizeConfiguration, ResetArg, RunState, SMPConfiguration, ShutdownCause,
};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit, Visitor};
use ruvm_qapi::{QDict, QValue};
use ruvm_target_arm::cpu::{ARM_MAX_VQ, ArmCpuModel, PauthAlg};
use ruvm_target_arm::tcg::SemihostingHost;

use crate::runstate::Runstate;
use crate::vl::Vm;
use crate::x86::{Drive, Located};

mod devices;

pub(crate) use devices::parse_drives;

/// Whether `target` is one the Arm boards exist for.
pub(crate) fn is_arm(target: &str) -> bool {
    target == "aarch64"
}

/// The versioned name of virt, `virt-11.1`, which `virt` is an alias of.
const VIRT_NAME: &str = "virt-11.1";

/// `mc->desc` of virt.
const VIRT_DESC: &str = "QEMU 11.1 ARM Virtual Machine";

/// `mc->max_cpus` of virt.
const VIRT_MAX_CPUS: u64 = 512;

/// The name of sbsa-ref.
const SBSA_NAME: &str = "sbsa-ref";

/// `mc->desc` of sbsa-ref.
const SBSA_DESC: &str = "QEMU 'SBSA Reference' ARM Virtual Machine";

/// `mc->max_cpus` of sbsa-ref.
const SBSA_MAX_CPUS: u64 = 512;

/// The Arm boards.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ArmBoard {
    /// `virt`.
    #[default]
    Virt,
    /// `sbsa-ref`.
    SbsaRef,
}

impl ArmBoard {
    /// The machine name the errors give.
    fn name(self) -> &'static str {
        match self {
            ArmBoard::Virt => VIRT_NAME,
            ArmBoard::SbsaRef => SBSA_NAME,
        }
    }

    /// `mc->max_cpus`.
    fn max_cpus(self) -> u64 {
        match self {
            ArmBoard::Virt => VIRT_MAX_CPUS,
            ArmBoard::SbsaRef => SBSA_MAX_CPUS,
        }
    }
}

/// The Arm board `-machine type=` names, if it names one.
pub(crate) fn board_by_name(name: &str) -> Option<ArmBoard> {
    match name {
        "virt" | VIRT_NAME => Some(ArmBoard::Virt),
        SBSA_NAME => Some(ArmBoard::SbsaRef),
        _ => None,
    }
}

/// The `-machine help` lines of the Arm boards, as (sort key, line) pairs, the alias on the
/// line before its machine.
pub(crate) fn machine_help_lines() -> Vec<(String, String)> {
    let text =
        format!("{:<20} {VIRT_DESC} (alias of {VIRT_NAME})\n{VIRT_NAME:<20} {VIRT_DESC}\n", "virt");
    vec![
        (VIRT_NAME.to_string(), text),
        (SBSA_NAME.to_string(), format!("{SBSA_NAME:<20} {SBSA_DESC}\n")),
    ]
}

/// Where semihosting calls go, `SemihostingTarget`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum SemihostingTarget {
    #[default]
    Auto,
    Native,
    Gdb,
}

/// `SemihostingConfig` and `semihost_chardev` of semihosting/config.c, as the option loop
/// fills them.
#[derive(Debug)]
pub(crate) struct Semihosting {
    pub enabled: bool,
    pub userspace: bool,
    pub target: SemihostingTarget,
    pub chardev: Option<String>,
    /// `semihosting.argv`, from `arg=`.
    pub args: Vec<String>,
    /// `qemu_semihosting_config_opts`.
    opts: QemuOptsList,
}

impl Default for Semihosting {
    fn default() -> Self {
        Semihosting {
            enabled: false,
            userspace: false,
            target: SemihostingTarget::Auto,
            chardev: None,
            args: Vec::new(),
            opts: semihosting_config_opts(),
        }
    }
}

/// `qemu_semihosting_config_opts`.
pub(crate) fn semihosting_config_opts() -> QemuOptsList {
    QemuOptsList::new(
        "semihosting-config",
        &[
            QemuOptDesc::new("enable", QemuOptType::Bool),
            QemuOptDesc::new("userspace", QemuOptType::Bool),
            QemuOptDesc::new("target", QemuOptType::String),
            QemuOptDesc::new("chardev", QemuOptType::String),
            QemuOptDesc::new("arg", QemuOptType::String),
        ],
    )
    .with_implied_opt_name("enable")
    .with_merge_lists()
}

impl Semihosting {
    /// `-semihosting`, `qemu_semihosting_enable()`.
    pub(crate) fn enable(&mut self) {
        self.enabled = true;
        self.target = SemihostingTarget::Auto;
    }

    /// `-semihosting-config`, `qemu_semihosting_config_options()`. The error is the message
    /// to report; errors from parsing have been reported already.
    pub(crate) fn config_options(&mut self, optstr: &str) -> std::result::Result<(), String> {
        let unsupported = || format!("unsupported semihosting-config {optstr}");
        self.enabled = true;
        let Some(opts) = self.opts.parse_noisily(optstr, false) else {
            return Err(unsupported());
        };
        self.enabled = opts.get_bool("enable", true);
        self.userspace = opts.get_bool("userspace", false);
        self.chardev = opts.get("chardev").map(str::to_string);
        self.target = match opts.get("target") {
            None | Some("auto") => SemihostingTarget::Auto,
            Some("native") => SemihostingTarget::Native,
            Some("gdb") => SemihostingTarget::Gdb,
            Some(_) => return Err(unsupported()),
        };
        // qemu_opt_foreach() over the merged options adds every arg= again each time.
        let args: Vec<String> = opts.iter_values(Some("arg")).map(str::to_string).collect();
        self.args.extend(args);
        Ok(())
    }
}

/// The generic machine properties and the virt properties, pulled out of the `-machine`
/// options.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BoardOptions {
    /// The board.
    pub board: ArmBoard,
    /// `memory.size`, rounded up to 8 KiB like `machine_set_mem()` does.
    pub ram_size: Option<u64>,
    pub cpus: u32,
    pub max_cpus: u32,
    pub kernel: Option<String>,
    pub initrd: Option<String>,
    pub append: Option<String>,
    pub dtb: Option<String>,
    /// `firmware`, from `-bios`.
    pub firmware: Option<String>,
    /// `secure`.
    pub secure: bool,
    /// `virtualization`.
    pub virtualization: bool,
    /// `mte`.
    pub mte: bool,
    /// `highmem`, `compact-highmem`, `highmem-redists`, `highmem-ecam`, `highmem-mmio` and
    /// `highmem-mmio-size`.
    pub highmem: Highmem,
    /// `msi`, or `its`, whichever came last.
    pub msi: VirtMsi,
    /// `iommu`.
    pub iommu: VirtIommu,
    /// `default-bus-bypass-iommu`.
    pub default_bus_bypass_iommu: bool,
    /// `dumpdtb`: write the device tree there and exit.
    pub dumpdtb: Option<String>,
    /// `acpi=off`: no ACPI tables in fw_cfg and no GED.
    pub acpi_off: bool,
    /// `spcr=off`: no SPCR among the ACPI tables.
    pub spcr_off: bool,
    /// `x-oem-id`.
    pub oem_id: Option<String>,
    /// `x-oem-table-id`.
    pub oem_table_id: Option<String>,
    /// The `-smp` topology.
    pub topology: Option<CpuTopology>,
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

/// A keyval size, as `visit_type_size()` of the keyval input visitor parses it.
fn prop_size(name: &str, value: &str) -> Result<u64> {
    let mut root = QDict::new();
    root.put(name, QValue::Str(value.to_string()));
    let mut v = QObjectInputVisitor::new_keyval(QValue::Dict(root));
    v.start_struct(None)?;
    let mut size = 0;
    let r = v.type_size(Some(name), &mut size);
    v.end_struct();
    r.map(|()| size)
}

/// The keyval value of a `-machine` property as a string.
fn prop_string(name: &str, value: &QValue) -> Result<String> {
    match value {
        QValue::Str(s) => Ok(s.clone()),
        _ => Err(Error::generic(format!("Parameter '{name}' is missing"))),
    }
}

/// A keyval boolean, as `visit_type_bool()` of the keyval input visitor parses it.
fn prop_bool(name: &str, value: &str) -> Result<bool> {
    match value {
        "on" | "yes" | "true" => Ok(true),
        "off" | "no" | "false" => Ok(false),
        _ => Err(Error::generic(format!("Parameter '{name}' expects 'on' or 'off'"))),
    }
}

/// `visit_type_OnOffAuto()` of the keyval input visitor.
fn on_off_auto<'a>(name: &str, value: &'a str) -> Result<&'a str> {
    match value {
        "on" | "off" | "auto" => Ok(value),
        _ => Err(Error::generic(format!("Parameter '{name}' does not accept value '{value}'"))),
    }
}

fn not_supported(name: &str, value: &str) -> Error {
    Error::generic(format!("{name}={value} is not supported by ruvm yet"))
}

/// Checks one virt property against the board that exists.
fn check_virt_prop(name: &str, value: &str) -> Result<()> {
    let want_bool = |want: bool| -> Result<()> {
        if prop_bool(name, value)? == want { Ok(()) } else { Err(not_supported(name, value)) }
    };
    match name {
        "ras" | "usb" => want_bool(false),
        // The board always behaves as with dtb-randomness=off.
        "dtb-randomness" | "dtb-kaslr-seed" => prop_bool(name, value).map(drop),
        "gic-version" => match value {
            "3" => Ok(()),
            "2" | "4" | "5" | "host" | "max" => Err(not_supported(name, value)),
            _ => Err(Error::generic("Invalid gic-version value".to_string())
                .hint("Valid values are 2, 3, 4, 5, host, and max.\n")),
        },
        "virtio-mmio-transports" => match value.parse::<u8>() {
            Ok(32) => Ok(()),
            Ok(_) => Err(not_supported(name, value)),
            Err(_) => Err(Error::generic(format!("Parameter '{name}' expects uint8_t"))),
        },
        _ => Err(Error::generic(format!("Property '{VIRT_NAME}-machine.{name}' not found"))),
    }
}

/// `machine_parse_smp_config()` for `board`, which knows clusters but not dies, modules,
/// books or drawers. Gives (cpus, maxcpus) and the topology.
pub(crate) fn parse_smp(
    board: ArmBoard,
    config: &SMPConfiguration,
) -> Result<(u32, u32, CpuTopology)> {
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
    for (value, name) in [
        (config.modules, "modules"),
        (config.dies, "dies"),
        (config.books, "books"),
        (config.drawers, "drawers"),
    ] {
        if get(value) > 1 {
            return Err(Error::generic(format!(
                "{name} > 1 not supported by this machine's CPU topology"
            )));
        }
    }
    let cpus = get(config.cpus);
    let mut sockets = get(config.sockets);
    let clusters = get(config.clusters).max(1);
    let mut cores = get(config.cores);
    let mut threads = get(config.threads);
    let mut maxcpus = get(config.maxcpus);
    let div = |a: u64, b: u64| a.checked_div(b).unwrap_or(0);

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
            cores = div(maxcpus, sockets * clusters * threads);
        } else if sockets == 0 {
            threads = threads.max(1);
            sockets = div(maxcpus, clusters * cores * threads);
        }
        if threads == 0 {
            threads = div(maxcpus, sockets * clusters * cores);
        }
    }
    let total = sockets * clusters * cores * threads;
    if maxcpus == 0 {
        maxcpus = total;
    }
    let cpus = if cpus == 0 { maxcpus } else { cpus };
    let topo = format!(
        "sockets ({sockets}) * clusters ({clusters}) * cores ({cores}) * threads ({threads})"
    );
    if total != maxcpus {
        return Err(Error::generic(format!(
            "Invalid CPU topology: product of the hierarchy must match maxcpus: {topo} != \
             maxcpus ({maxcpus})"
        )));
    }
    if maxcpus < cpus {
        return Err(Error::generic(format!(
            "Invalid CPU topology: maxcpus must be equal to or greater than smp: {topo} == \
             maxcpus ({maxcpus}) < smp_cpus ({cpus})"
        )));
    }
    if maxcpus > board.max_cpus() {
        return Err(Error::generic(format!(
            "Invalid SMP CPUs {maxcpus}. The max CPUs supported by machine '{}' is {}",
            board.name(),
            board.max_cpus()
        )));
    }
    // All are at most 512 now.
    let topology = CpuTopology {
        sockets: sockets as u32,
        clusters: clusters as u32,
        cores: cores as u32,
        threads: threads as u32,
        has_clusters: config.clusters.is_some(),
    };
    Ok((cpus as u32, maxcpus as u32, topology))
}

/// `qemu_apply_machine_options()` for `board`: takes the generic properties out of `machine`
/// (whose `type`, `accel` and `kernel-irqchip` are gone already) and checks the rest
/// against the board.
pub(crate) fn take_board_options(board: ArmBoard, machine: &QDict) -> Result<BoardOptions> {
    let mut o = BoardOptions { board, ..BoardOptions::default() };
    if let Some(mem) = visit_member::<MemorySizeConfiguration>(machine, "memory")? {
        if mem.slots.is_some_and(|s| s != 0) || mem.max_size.is_some_and(|m| Some(m) != mem.size) {
            return Err(Error::generic("memory hotplug is not supported by ruvm yet"));
        }
        o.ram_size = mem.size.map(|s| s.next_multiple_of(8192));
    }
    match visit_member::<SMPConfiguration>(machine, "smp")? {
        // Without -smp, sbsa-ref has its default_cpus in one socket, cluster, core and thread.
        None if board == ArmBoard::SbsaRef => {
            let n = SBSA_DEFAULT_CPUS as u32;
            (o.cpus, o.max_cpus, o.topology) = (n, n, None);
        }
        smp => {
            let (cpus, max_cpus, topology) = parse_smp(board, &smp.unwrap_or_default())?;
            (o.cpus, o.max_cpus, o.topology) = (cpus, max_cpus, Some(topology));
        }
    }
    for (name, value) in machine.iter_inserted() {
        match name {
            "memory" | "smp" => {}
            "kernel" => o.kernel = Some(prop_string(name, value)?),
            "initrd" => o.initrd = Some(prop_string(name, value)?),
            "append" => o.append = Some(prop_string(name, value)?),
            "dtb" => o.dtb = Some(prop_string(name, value)?),
            "dumpdtb" => o.dumpdtb = Some(prop_string(name, value)?),
            "firmware" => o.firmware = Some(prop_string(name, value)?),
            // Generic machine properties that change nothing here.
            "dump-guest-core" | "mem-merge" | "graphics" | "suppress-vmdesc" => {}
            // sbsa-ref has no properties of its own.
            _ if board == ArmBoard::SbsaRef => {
                return Err(Error::generic(format!(
                    "Property '{SBSA_NAME}-machine.{name}' not found"
                )));
            }
            "secure" => o.secure = prop_bool(name, &prop_string(name, value)?)?,
            "virtualization" => o.virtualization = prop_bool(name, &prop_string(name, value)?)?,
            "mte" => o.mte = prop_bool(name, &prop_string(name, value)?)?,
            "highmem" => o.highmem.highmem = prop_bool(name, &prop_string(name, value)?)?,
            "compact-highmem" => {
                o.highmem.compact = prop_bool(name, &prop_string(name, value)?)?;
            }
            "highmem-redists" => {
                o.highmem.redists = prop_bool(name, &prop_string(name, value)?)?;
            }
            // virt_set_msi() and virt_set_its(). its=off means no MSI controller with a GICv3.
            "msi" => {
                o.msi = match prop_string(name, value)?.as_str() {
                    "auto" => VirtMsi::Auto,
                    "its" => VirtMsi::Its,
                    "gicv2m" => return Err(not_supported(name, "gicv2m")),
                    "off" => VirtMsi::Off,
                    _ => {
                        return Err(Error::generic("Invalid msi value")
                            .hint("Valid values are auto, gicv2m, its, off\n"));
                    }
                };
            }
            "its" => {
                let on = prop_bool(name, &prop_string(name, value)?)?;
                o.msi = if on { VirtMsi::Its } else { VirtMsi::Off };
            }
            // virt_set_iommu().
            "iommu" => {
                o.iommu = match prop_string(name, value)?.as_str() {
                    "smmuv3" => VirtIommu::SmmuV3,
                    "none" => VirtIommu::None,
                    _ => {
                        return Err(Error::generic("Invalid iommu value")
                            .hint("Valid values are none, smmuv3.\n"));
                    }
                };
            }
            "default-bus-bypass-iommu" => {
                o.default_bus_bypass_iommu = prop_bool(name, &prop_string(name, value)?)?;
            }
            "acpi" => o.acpi_off = on_off_auto(name, &prop_string(name, value)?)? == "off",
            "spcr" => o.spcr_off = !prop_bool(name, &prop_string(name, value)?)?,
            // virt_set_oem_id() and virt_set_oem_table_id().
            "x-oem-id" => {
                let v = prop_string(name, value)?;
                if v.len() > 6 {
                    return Err(Error::generic(
                        "User specified oem-id value is bigger than 6 bytes in size",
                    ));
                }
                o.oem_id = Some(v);
            }
            "x-oem-table-id" => {
                let v = prop_string(name, value)?;
                if v.len() > 8 {
                    return Err(Error::generic(
                        "User specified oem-table-id value is bigger than 8 bytes in size",
                    ));
                }
                o.oem_table_id = Some(v);
            }
            "highmem-ecam" => o.highmem.ecam = prop_bool(name, &prop_string(name, value)?)?,
            "highmem-mmio" => o.highmem.mmio = prop_bool(name, &prop_string(name, value)?)?,
            "highmem-mmio-size" => {
                let size = prop_size(name, &prop_string(name, value)?)?;
                check_highmem_mmio_size(size).map_err(Error::generic)?;
                o.highmem.mmio_size = size;
            }
            _ => check_virt_prop(name, &prop_string(name, value)?)?,
        }
    }
    o.kernel = o.kernel.filter(|k| !k.is_empty());
    o.initrd = o.initrd.filter(|i| !i.is_empty());
    o.dtb = o.dtb.filter(|d| !d.is_empty());
    o.firmware = o.firmware.filter(|f| !f.is_empty());
    Ok(o)
}

/// The CPU models virt takes in QEMU that are not modelled here.
const OTHER_VIRT_CPUS: &[&str] = &[
    "cortex-a7",
    "cortex-a15",
    "cortex-a35",
    "cortex-a53",
    "cortex-a55",
    "cortex-a710",
    "a64fx",
    "neoverse-v1",
    "neoverse-n2",
    "host",
];

/// `valid_cpu_types` of sbsa-ref.
const SBSA_CPUS: &[&str] =
    &["cortex-a57", "cortex-a72", "neoverse-n1", "neoverse-v1", "neoverse-n2", "max"];

/// `-cpu model,prop=value,...` for `board` (`cortex-a57` on virt and `neoverse-n1` on sbsa-ref
/// without one).
pub(crate) fn parse_cpu(board: ArmBoard, arg: Option<&str>) -> Result<ArmCpuModel> {
    let arg = arg.unwrap_or(match board {
        ArmBoard::Virt => "cortex-a57",
        ArmBoard::SbsaRef => "neoverse-n1",
    });
    let mut parts = arg.split(',');
    let name = parts.next().unwrap_or_default();
    // machine_run_board_init() checks the model against valid_cpu_types.
    if board == ArmBoard::SbsaRef
        && !SBSA_CPUS.contains(&name)
        && (ArmCpuModel::by_name(name).is_some() || OTHER_VIRT_CPUS.contains(&name))
    {
        return Err(Error::generic(format!("Invalid CPU model: {name}"))
            .hint(format!("The valid models are: {}\n", SBSA_CPUS.join(", "))));
    }
    let Some(mut model) = ArmCpuModel::by_name(name) else {
        if OTHER_VIRT_CPUS.contains(&name) {
            return Err(Error::generic(format!("CPU model '{name}' is not supported by ruvm yet")));
        }
        return Err(Error::generic(format!("unable to find CPU model '{name}'")));
    };
    // arm_cpu_pauth_finalize() runs after every property is set.
    let mut pauth = model.features.pauth != 0;
    let (mut impdef, mut qarma3, mut qarma5) = (false, false, false);
    for feat in parts.filter(|f| !f.is_empty()) {
        let (prop, value) = match feat.split_once('=') {
            Some((p, v)) => (p, v),
            None => match feat.strip_prefix('-') {
                Some(p) => (p, "off"),
                None => (feat.strip_prefix('+').unwrap_or(feat), "on"),
            },
        };
        match prop {
            "sve-max-vq" if name == "max" => {
                let vq: u32 = value.parse().map_err(|_| {
                    Error::generic("Parameter 'sve-max-vq' expects uint32_t".to_string())
                })?;
                if vq == 0 || vq as usize > ARM_MAX_VQ {
                    return Err(Error::generic("unsupported SVE vector length")
                        .hint(format!("Valid sve-max-vq in range [1-{ARM_MAX_VQ}]\n")));
                }
                model = model.with_sve_max_vq(vq);
            }
            "sve" if model.features.sve && prop_bool(prop, value)? => {}
            "aarch64" if prop_bool(prop, value)? => {}
            "pauth" if name == "max" => pauth = prop_bool(prop, value)?,
            // FEAT_RME with FEAT_RME_GPC3; the board takes it away again without EL3.
            "x-rme" if name == "max" => model = model.with_rme(prop_bool(prop, value)?),
            "pauth-impdef" if name == "max" => impdef = prop_bool(prop, value)?,
            "pauth-qarma3" if name == "max" => qarma3 = prop_bool(prop, value)?,
            "pauth-qarma5" if name == "max" => qarma5 = prop_bool(prop, value)?,
            "pmu" => {
                if !prop_bool(prop, value)? {
                    model = model.without_pmu();
                }
            }
            "sve" | "sme" | "sme-fa64" | "pauth" | "pauth-impdef" | "pauth-qarma3"
            | "pauth-qarma5" | "lpa2" | "aarch64" | "reset-cbar" | "rvbar" | "cntfrq"
            | "has_el2" | "has_el3" | "sve128" | "sve256" | "sve512" | "sve1024" | "sve2048"
            | "sme128" | "sme256" | "sme512" | "sme1024" | "sme2048" => {
                return Err(Error::generic(format!(
                    "CPU property {prop}={value} is not supported by ruvm yet"
                )));
            }
            _ => {
                return Err(Error::generic(format!(
                    "can't apply global {name}-arm-cpu.{prop}={value}: Property \
                     '{name}-arm-cpu.{prop}' not found"
                )));
            }
        }
    }
    if name == "max" {
        let alg = if pauth {
            if u8::from(impdef) + u8::from(qarma3) + u8::from(qarma5) > 1 {
                return Err(Error::generic(
                    "cannot enable pauth-impdef, pauth-qarma3 and pauth-qarma5 at the same time",
                ));
            }
            Some(if qarma5 {
                PauthAlg::Qarma5
            } else if qarma3 {
                PauthAlg::Qarma3
            } else {
                PauthAlg::Impdef
            })
        } else {
            if impdef || qarma3 || qarma5 {
                return Err(Error::generic(
                    "cannot enable pauth-impdef, pauth-qarma3 or pauth-qarma5 without pauth",
                )
                .hint("Add pauth=on to the CPU property list.\n"));
            }
            None
        };
        model = model.with_pauth(alg);
    }
    Ok(model)
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The PL011 on its chardev, `qdev_prop_set_chr(dev, "chardev", serial_hd(0))`: what the
/// guest writes goes to the chardev, what the chardev reads goes to the UART.
struct ChardevPl011 {
    uart: Arc<Pl011>,
    chr: Arc<Chardev>,
}

impl SerialBackend for ChardevPl011 {
    fn write(&self, bytes: &[u8]) -> usize {
        // Output nobody can take is dropped, as pl011_write_txdata() ignores errors.
        let _ = self.chr.write_all(bytes);
        bytes.len()
    }
}

impl Frontend for ChardevPl011 {
    fn serve(&self, conn: &mut Connection) -> std::io::Result<()> {
        let mut buf = [0u8; 256];
        loop {
            let n = conn.recv(&mut buf)?;
            if n == 0 {
                return Ok(());
            }
            let mut input = &buf[..n];
            while !input.is_empty() {
                let room = self.uart.can_receive();
                if room == 0 {
                    std::thread::sleep(Duration::from_millis(1));
                    continue;
                }
                let k = room.min(input.len());
                self.uart.receive(&input[..k]);
                input = &input[k..];
            }
        }
    }
}

/// Stands in for the chardev of the second UART until the board exists and it can be
/// connected.
struct Unconnected;

impl SerialBackend for Unconnected {
    fn write(&self, bytes: &[u8]) -> usize {
        bytes.len()
    }
}

/// The size of the console input FIFO, `FIFO_SIZE`.
const SEMI_FIFO_SIZE: usize = 1024;

#[derive(Debug, Default)]
struct Fifo {
    bytes: VecDeque<u8>,
    /// The machine is going away: wake the readers.
    closed: bool,
}

/// The semihosting console, semihosting/console.c, and the exit of SYS_EXIT.
struct SemiConsole {
    chr: Option<Arc<Chardev>>,
    /// `semihosting.argv` from `arg=`; the board falls back to `-kernel` and `-append`.
    args: Vec<String>,
    fifo: Mutex<Fifo>,
    cv: Condvar,
    runstate: Weak<Runstate>,
}

impl std::fmt::Debug for SemiConsole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SemiConsole")
            .field("chr", &self.chr.as_ref().map(|c| c.label().to_string()))
            .field("args", &self.args)
            .finish_non_exhaustive()
    }
}

impl SemiConsole {
    /// Wakes a vCPU waiting for console input, for good.
    fn close(&self) {
        lock(&self.fifo).closed = true;
        self.cv.notify_all();
    }
}

impl SemihostingHost for SemiConsole {
    /// `qemu_semihosting_console_write()`.
    fn console_write(&self, buf: &[u8]) -> usize {
        match &self.chr {
            Some(chr) => chr.write_all(buf).unwrap_or(0),
            None => {
                let mut err = std::io::stderr();
                match err.write_all(buf) {
                    Ok(()) => buf.len(),
                    Err(_) => 0,
                }
            }
        }
    }

    /// `qemu_semihosting_console_read()` for one byte.
    fn console_read(&self) -> u8 {
        let mut f = lock(&self.fifo);
        loop {
            if let Some(b) = f.bytes.pop_front() {
                self.cv.notify_all();
                return b;
            }
            if f.closed {
                return 0;
            }
            f = self.cv.wait(f).unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn exit(&self, code: u32) {
        if let Some(rs) = self.runstate.upgrade() {
            rs.shutdown_request_with_code(ShutdownCause::GuestShutdown, code as i32);
        }
    }

    /// `semihosting_get_cmdline()`.
    fn cmdline(&self) -> Option<String> {
        if self.args.is_empty() { None } else { Some(self.args.join(" ")) }
    }

    /// The board answers this one.
    fn heap_info(&self) -> (u64, u64) {
        (0, 0)
    }
}

/// The chardev side of the semihosting console: `console_can_read()` and `console_read()`.
struct SemiConsoleFrontend(Arc<SemiConsole>);

impl Frontend for SemiConsoleFrontend {
    fn serve(&self, conn: &mut Connection) -> std::io::Result<()> {
        let c = &self.0;
        let mut buf = [0u8; 256];
        loop {
            let n = conn.recv(&mut buf)?;
            if n == 0 {
                return Ok(());
            }
            let mut input = &buf[..n];
            let mut f = lock(&c.fifo);
            while !input.is_empty() {
                if f.closed {
                    return Ok(());
                }
                let room = SEMI_FIFO_SIZE - f.bytes.len();
                if room == 0 {
                    f = c.cv.wait(f).unwrap_or_else(PoisonError::into_inner);
                    continue;
                }
                let k = room.min(input.len());
                f.bytes.extend(&input[..k]);
                input = &input[k..];
                c.cv.notify_all();
            }
        }
    }
}

/// The board on TCG.
#[derive(Debug)]
enum Machine {
    Virt(Arc<VirtTcgMachine>),
    SbsaRef(Arc<SbsaRefTcgMachine>),
}

/// The board running on its vCPUs, and the chardevs its devices are attached to.
#[derive(Debug)]
pub(crate) struct Running {
    machine: Machine,
    console: Option<Arc<SemiConsole>>,
    _attachments: Vec<Attachment>,
}

impl Running {
    /// Lets a vCPU waiting for semihosting console input go, so that the vCPUs can stop.
    pub(crate) fn wake_console(&self) {
        if let Some(c) = &self.console {
            c.close();
        }
    }

    /// Stops the vCPU and timer threads.
    pub(crate) fn quit(&self) {
        self.wake_console();
        match &self.machine {
            Machine::Virt(m) => m.quit(),
            Machine::SbsaRef(m) => m.quit(),
        }
    }
}

/// What the run loop reports, turned into runstate changes and QMP events.
fn event_handler(vm: &Arc<Vm>) -> VirtEventHandler {
    let rs = Arc::clone(&vm.runstate);
    let qmp = Arc::clone(&vm.qmp);
    Arc::new(move |e: VirtEvent| match e {
        VirtEvent::Shutdown(r) => rs.shutdown_request(match r {
            ShutdownReason::GuestShutdown => ShutdownCause::GuestShutdown,
            ShutdownReason::GuestReset => ShutdownCause::GuestReset,
        }),
        VirtEvent::Reset => {
            let arg = ResetArg { guest: true, reason: ShutdownCause::GuestReset };
            if let Some(ev) = event_reset(&qmp.policy(), arg) {
                qmp.emit_event(ev);
            }
        }
        VirtEvent::InternalError(msg) => {
            eprintln!("{msg}");
            rs.vm_stop(RunState::InternalError);
        }
    })
}

/// Lets the runstate start and stop the vCPUs.
fn set_cpu_hook<B: TcgBoard>(vm: &Arc<Vm>, machine: &Arc<TcgMachine<B>>) {
    let weak = Arc::downgrade(machine);
    vm.runstate.set_cpu_hook(Some(Arc::new(move |run| {
        if let Some(m) = weak.upgrade() {
            if run {
                m.start();
            } else {
                m.pause();
            }
        }
    })));
}

/// What `start_board_tcg` needs from the command line besides the machine options.
#[derive(Debug)]
pub(crate) struct ArmArgs<'a> {
    /// `-cpu`.
    pub cpu: Option<&'a str>,
    /// `-no-reboot`.
    pub no_reboot: bool,
    pub semihosting: &'a Semihosting,
    /// `-device`.
    pub devices: &'a [(String, Option<Location>)],
    /// `-drive`, from [`parse_drives`].
    pub drives: &'a [Drive],
}

/// `qemu_semihosting_chardev_init()`: the chardev of `-semihosting-config chardev=`.
fn semihosting_chardev(
    chardevs: &Chardevs,
    semi: &Semihosting,
) -> std::result::Result<Option<Arc<Chardev>>, Located> {
    let Some(id) = &semi.chardev else { return Ok(None) };
    match chardevs.find(id) {
        Some(c) => Ok(Some(c)),
        None => Err(Located(None, Error::generic(format!("semihosting chardev '{id}' not found")))),
    }
}

/// `handle_machine_dumpdtb()`: writes the device tree to `path` and ends the process.
fn dumpdtb(path: &str, fdt: &[u8]) -> std::result::Result<(), Vec<Located>> {
    if let Err(e) = std::fs::write(path, fdt) {
        let e = Error::generic(format!("Error saving FDT to file {path}: {e}"));
        return Err(vec![Located(None, e)]);
    }
    ruvm_chardev::stdio::term_exit();
    std::process::exit(0);
}

/// `qemu_init_board()` and `qemu_machine_creation_done()` for the Arm boards on TCG: builds
/// the board, connects the UARTs to `serial_hds` and semihosting to its chardev, and puts it
/// all on the vCPU threads, stopped until `vm_start()`. A `dumpdtb` ends the process here, as
/// in QEMU.
pub(crate) fn start_board_tcg(
    vm: &Arc<Vm>,
    tcg: TcgOptions,
    opts: BoardOptions,
    args: &ArmArgs<'_>,
    serial_hds: &[Option<Arc<Chardev>>],
) -> std::result::Result<Running, Vec<Located>> {
    if opts.board == ArmBoard::SbsaRef {
        return start_sbsa_ref_tcg(vm, tcg, opts, args, serial_hds);
    }
    let one = |e: Error| vec![Located(None, e)];
    let semi = args.semihosting;
    let console_chr = semihosting_chardev(&vm.chardevs, semi).map_err(|e| vec![e])?;
    if semi.enabled && semi.target == SemihostingTarget::Gdb {
        return Err(one(Error::generic(
            "semihosting-config target=gdb is not supported by ruvm yet",
        )));
    }
    let cpu = parse_cpu(ArmBoard::Virt, args.cpu).map_err(one)?;
    let plan = devices::plan(ArmBoard::Virt, args.drives, args.devices)?;
    let clock = Clock::new(ClockType::Virtual, TimeSource::Monotonic(Instant::now()));
    let rtc_clock = Clock::new(ClockType::Host, TimeSource::Wall);

    let console = semi.enabled.then(|| {
        Arc::new(SemiConsole {
            chr: console_chr.clone(),
            args: semi.args.clone(),
            fifo: Mutex::new(Fifo::default()),
            cv: Condvar::new(),
            runstate: Arc::downgrade(&vm.runstate),
        })
    });
    let mut cfg = VirtConfig::new(cpu);
    cfg.smp = opts.cpus as usize;
    if let Some(size) = opts.ram_size {
        cfg.ram_size = size;
    }
    cfg.kernel = opts.kernel;
    cfg.initrd = opts.initrd;
    cfg.append = opts.append;
    cfg.dtb = opts.dtb;
    cfg.firmware = opts.firmware;
    cfg.secure = opts.secure;
    cfg.virtualization = opts.virtualization;
    cfg.mte = opts.mte;
    cfg.highmem = opts.highmem;
    cfg.msi = opts.msi;
    cfg.iommu = opts.iommu;
    cfg.default_bus_bypass_iommu = opts.default_bus_bypass_iommu;
    // machvirt_init() checks maxcpus against the redistributor space.
    cfg.max_cpus = Some(opts.max_cpus as usize);
    cfg.topology = opts.topology;
    cfg.acpi = !opts.acpi_off;
    cfg.spcr = !opts.spcr_off;
    if let Some(id) = opts.oem_id {
        cfg.oem_id = id;
    }
    if let Some(id) = opts.oem_table_id {
        cfg.oem_table_id = id;
    }
    for (slot, drive) in cfg.pflash.iter_mut().zip(plan.pflash) {
        if let Some(backing) = drive {
            *slot = backing;
        }
    }
    // serial_hd(1) makes the second UART exist; its chardev is connected once it does.
    let serial1 = serial_hds.get(1).cloned().flatten();
    if serial1.is_some() {
        cfg.serial1 = Some(Arc::new(Unconnected));
    }
    cfg.semihosting = console.clone().map(|c| c as Arc<dyn SemihostingHost>);
    cfg.semihosting_userspace = semi.userspace;
    cfg.clock = Some(Arc::clone(&clock));
    cfg.rtc_clock = Some(Arc::clone(&rtc_clock));
    let mut board = VirtMachine::new(cfg).map_err(|e| one(Error::generic(e)))?;
    devices::plug(devices::Target::Virt(&board), &plan.virtio, &plan.display, args.drives)
        .map_err(|e| vec![e])?;

    let mut attachments = Vec::new();
    if let Some(Some(chr)) = serial_hds.first() {
        let fe = Arc::new(ChardevPl011 { uart: Arc::clone(board.uart()), chr: Arc::clone(chr) });
        board.set_serial_backend(Some(fe.clone()));
        attachments.push(chr.attach(fe).map_err(|e| vec![Located(None, e)])?);
    }
    match (serial1, board.uart1()) {
        (Some(chr), Some(uart)) => {
            let fe = Arc::new(ChardevPl011 { uart: Arc::clone(uart), chr: Arc::clone(&chr) });
            board.set_serial1_backend(Some(fe.clone()));
            attachments.push(chr.attach(fe).map_err(|e| vec![Located(None, e)])?);
        }
        // The secure UART without a chardev.
        (None, Some(_)) => board.set_serial1_backend(None),
        _ => {}
    }
    if let (Some(console), Some(chr)) = (&console, &console_chr) {
        let fe = Arc::new(SemiConsoleFrontend(Arc::clone(console)));
        attachments.push(chr.attach(fe).map_err(|e| vec![Located(None, e)])?);
    }

    board.machine_done().map_err(|e| one(Error::generic(e)))?;
    if let Some(path) = &opts.dumpdtb {
        dumpdtb(path, board.fdt().as_bytes())?;
    }

    let cfg = VirtRunConfig { no_reboot: args.no_reboot, tcg, backend: None };
    let (machine, warnings) =
        VirtTcgMachine::new(board, vec![clock, rtc_clock], &cfg, event_handler(vm))
            .map_err(|e| one(Error::generic(e)))?;
    for w in &warnings {
        warn_report(w);
    }
    let machine = Arc::new(machine);
    set_cpu_hook(vm, &machine);
    Ok(Running { machine: Machine::Virt(machine), console, _attachments: attachments })
}

/// [`start_board_tcg`] for sbsa-ref: the UART on `serial_hds[0]` and the secure UARTs on
/// `serial_hds[1]` and `serial_hds[2]`.
fn start_sbsa_ref_tcg(
    vm: &Arc<Vm>,
    tcg: TcgOptions,
    opts: BoardOptions,
    args: &ArmArgs<'_>,
    serial_hds: &[Option<Arc<Chardev>>],
) -> std::result::Result<Running, Vec<Located>> {
    let one = |e: Error| vec![Located(None, e)];
    if args.semihosting.enabled {
        return Err(one(Error::generic(
            "semihosting with machine sbsa-ref is not supported by ruvm yet",
        )));
    }
    let cpu = parse_cpu(ArmBoard::SbsaRef, args.cpu).map_err(one)?;
    let plan = devices::plan(ArmBoard::SbsaRef, args.drives, args.devices)?;
    let clock = Clock::new(ClockType::Virtual, TimeSource::Monotonic(Instant::now()));
    let rtc_clock = Clock::new(ClockType::Host, TimeSource::Wall);

    let mut cfg = SbsaRefConfig::new(cpu);
    cfg.smp = opts.cpus as usize;
    if let Some(size) = opts.ram_size {
        cfg.ram_size = size;
    }
    cfg.kernel = opts.kernel;
    cfg.initrd = opts.initrd;
    cfg.append = opts.append;
    cfg.dtb = opts.dtb;
    cfg.firmware = opts.firmware;
    cfg.max_cpus = Some(opts.max_cpus as usize);
    cfg.topology = opts.topology;
    for (slot, drive) in cfg.pflash.iter_mut().zip(plan.pflash) {
        if let Some(backing) = drive {
            *slot = backing;
        }
    }
    cfg.clock = Some(Arc::clone(&clock));
    cfg.rtc_clock = Some(Arc::clone(&rtc_clock));
    let mut board = SbsaRefMachine::new(cfg).map_err(|e| one(Error::generic(e)))?;
    devices::plug_ahci(&board, &plan.ide, args.drives).map_err(|e| vec![e])?;
    devices::plug(devices::Target::SbsaRef(&board), &plan.virtio, &plan.display, args.drives)
        .map_err(|e| vec![e])?;

    let mut attachments = Vec::new();
    let uarts = [board.uart(), board.secure_uarts()[0], board.secure_uarts()[1]].map(Arc::clone);
    for (n, uart) in uarts.into_iter().enumerate() {
        let Some(Some(chr)) = serial_hds.get(n) else { continue };
        let fe = Arc::new(ChardevPl011 { uart, chr: Arc::clone(chr) });
        match n {
            0 => board.set_serial_backend(Some(fe.clone())),
            _ => board.set_secure_serial_backend(n - 1, Some(fe.clone())),
        }
        attachments.push(chr.attach(fe).map_err(|e| vec![Located(None, e)])?);
    }

    board.machine_done().map_err(|e| one(Error::generic(e)))?;
    if let Some(path) = &opts.dumpdtb {
        dumpdtb(path, board.fdt().as_bytes())?;
    }

    let cfg = VirtRunConfig { no_reboot: args.no_reboot, tcg, backend: None };
    let (machine, warnings) =
        SbsaRefTcgMachine::new(board, vec![clock, rtc_clock], &cfg, event_handler(vm))
            .map_err(|e| one(Error::generic(e)))?;
    for w in &warnings {
        warn_report(w);
    }
    let machine = Arc::new(machine);
    set_cpu_hook(vm, &machine);
    Ok(Running { machine: Machine::SbsaRef(machine), console: None, _attachments: attachments })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn smp_cfg(cpus: Option<i64>, clusters: Option<i64>, cores: Option<i64>) -> SMPConfiguration {
        SMPConfiguration { cpus, clusters, cores, ..SMPConfiguration::default() }
    }

    #[test]
    fn smp_topologies() {
        let topo = |sockets, clusters, cores, threads, has_clusters| CpuTopology {
            sockets,
            clusters,
            cores,
            threads,
            has_clusters,
        };
        let flat = |n| topo(1, 1, n, 1, false);
        assert_eq!(
            parse_smp(ArmBoard::Virt, &SMPConfiguration::default()).unwrap(),
            (1, 1, flat(1))
        );
        assert_eq!(
            parse_smp(ArmBoard::Virt, &smp_cfg(Some(4), None, None)).unwrap(),
            (4, 4, flat(4))
        );
        assert_eq!(
            parse_smp(ArmBoard::Virt, &smp_cfg(None, Some(2), Some(2))).unwrap(),
            (4, 4, topo(1, 2, 2, 1, true))
        );
        let threads = SMPConfiguration {
            cpus: Some(2),
            sockets: Some(2),
            threads: Some(2),
            maxcpus: Some(8),
            ..SMPConfiguration::default()
        };
        assert_eq!(parse_smp(ArmBoard::Virt, &threads).unwrap(), (2, 8, topo(2, 1, 2, 2, false)));
        let e = parse_smp(ArmBoard::Virt, &smp_cfg(Some(3), Some(2), None)).unwrap_err();
        assert_eq!(
            e.message(),
            "Invalid CPU topology: product of the hierarchy must match maxcpus: sockets (1) * \
             clusters (2) * cores (1) * threads (1) != maxcpus (3)"
        );
        let dies = SMPConfiguration { dies: Some(2), ..SMPConfiguration::default() };
        assert_eq!(
            parse_smp(ArmBoard::Virt, &dies).unwrap_err().message(),
            "dies > 1 not supported by this machine's CPU topology"
        );
        let e = parse_smp(ArmBoard::Virt, &smp_cfg(Some(513), None, None)).unwrap_err();
        assert_eq!(
            e.message(),
            "Invalid SMP CPUs 513. The max CPUs supported by machine 'virt-11.1' is 512"
        );
    }

    #[test]
    fn semihosting_config() {
        let mut s = Semihosting::default();
        s.config_options("enable=on,target=native,chardev=output").unwrap();
        assert!(s.enabled);
        assert_eq!(s.target, SemihostingTarget::Native);
        assert_eq!(s.chardev.as_deref(), Some("output"));
        s.config_options("arg=a,arg=b").unwrap();
        assert_eq!(s.args, ["a", "b"]);
        // The lists merge, so the earlier settings stay.
        assert_eq!(s.chardev.as_deref(), Some("output"));
        let mut s = Semihosting::default();
        assert_eq!(
            s.config_options("target=foo").unwrap_err(),
            "unsupported semihosting-config target=foo"
        );
        let mut s = Semihosting::default();
        s.config_options("enable=off").unwrap();
        assert!(!s.enabled);
    }

    #[test]
    fn cpu_models() {
        assert_eq!(parse_cpu(ArmBoard::Virt, None).unwrap().name, "cortex-a57");
        assert_eq!(
            parse_cpu(ArmBoard::Virt, Some("max,sve-max-vq=2")).unwrap().features.sve_max_vq,
            2
        );
        assert_eq!(
            parse_cpu(ArmBoard::Virt, Some("cortex-a53")).unwrap_err().message(),
            "CPU model 'cortex-a53' is not supported by ruvm yet"
        );
        assert_eq!(
            parse_cpu(ArmBoard::Virt, Some("foo")).unwrap_err().message(),
            "unable to find CPU model 'foo'"
        );
        assert_eq!(
            parse_cpu(ArmBoard::Virt, Some("cortex-a57,pauth-qarma5=on")).unwrap_err().message(),
            "CPU property pauth-qarma5=on is not supported by ruvm yet"
        );
        assert_eq!(
            parse_cpu(ArmBoard::Virt, Some("max,sve-max-vq=0")).unwrap_err().message(),
            "unsupported SVE vector length"
        );
    }

    #[test]
    fn machine_properties() {
        let mut m = QDict::new();
        m.put("its", "off");
        m.put("gic-version", "3");
        m.put("kernel", "k");
        m.put("dtb", "d.dtb");
        let o = take_board_options(ArmBoard::Virt, &m).unwrap();
        assert_eq!(o.kernel.as_deref(), Some("k"));
        assert_eq!(o.dtb.as_deref(), Some("d.dtb"));
        assert_eq!(o.msi, VirtMsi::Off);
        // msi and its set the same thing, and the last one wins.
        let msi = |props: &[(&str, &str)]| {
            let mut m = QDict::new();
            for (k, v) in props {
                m.put(*k, *v);
            }
            take_board_options(ArmBoard::Virt, &m).map(|o| o.msi)
        };
        assert_eq!(msi(&[]).unwrap(), VirtMsi::Auto);
        assert_eq!(msi(&[("msi", "off"), ("its", "on")]).unwrap(), VirtMsi::Its);
        assert_eq!(msi(&[("its", "on"), ("msi", "off")]).unwrap(), VirtMsi::Off);
        assert_eq!(msi(&[("msi", "its")]).unwrap(), VirtMsi::Its);
        assert_eq!(msi(&[("msi", "auto")]).unwrap(), VirtMsi::Auto);
        let e = msi(&[("msi", "foo")]).unwrap_err();
        assert_eq!(e.message(), "Invalid msi value");
        assert_eq!(e.hint_text(), Some("Valid values are auto, gicv2m, its, off\n"));
        assert_eq!(
            msi(&[("msi", "gicv2m")]).unwrap_err().message(),
            "msi=gicv2m is not supported by ruvm yet"
        );
        // virt_set_iommu().
        let iommu = |v: &str| {
            let mut m = QDict::new();
            m.put("iommu", v);
            take_board_options(ArmBoard::Virt, &m).map(|o| o.iommu)
        };
        assert_eq!(iommu("smmuv3").unwrap(), VirtIommu::SmmuV3);
        assert_eq!(iommu("none").unwrap(), VirtIommu::None);
        let e = iommu("virtio").unwrap_err();
        assert_eq!(e.message(), "Invalid iommu value");
        assert_eq!(e.hint_text(), Some("Valid values are none, smmuv3.\n"));
        let mut m = QDict::new();
        m.put("default-bus-bypass-iommu", "on");
        assert!(take_board_options(ArmBoard::Virt, &m).unwrap().default_bus_bypass_iommu);
        let mut m = QDict::new();
        m.put("secure", "on");
        m.put("virtualization", "on");
        m.put("firmware", "edk2.fd");
        let o = take_board_options(ArmBoard::Virt, &m).unwrap();
        assert!(o.secure && o.virtualization);
        assert_eq!(o.firmware.as_deref(), Some("edk2.fd"));
        let mut m = QDict::new();
        m.put("mte", "on");
        assert!(take_board_options(ArmBoard::Virt, &m).unwrap().mte);
        let mut m = QDict::new();
        m.put("ras", "on");
        assert_eq!(
            take_board_options(ArmBoard::Virt, &m).unwrap_err().message(),
            "ras=on is not supported by ruvm yet"
        );
        let mut m = QDict::new();
        m.put("highmem-ecam", "off");
        m.put("compact-highmem", "off");
        m.put("highmem-mmio-size", "1T");
        let o = take_board_options(ArmBoard::Virt, &m).unwrap();
        assert!(!o.highmem.ecam && !o.highmem.compact && o.highmem.mmio);
        assert_eq!(o.highmem.mmio_size, 1 << 40);
        let mut m = QDict::new();
        m.put("highmem-mmio-size", "1G");
        assert_eq!(
            take_board_options(ArmBoard::Virt, &m).unwrap_err().message(),
            "highmem-mmio-size cannot be set to a lower value than the default (512 GiB)"
        );
        let mut m = QDict::new();
        m.put("foo", "on");
        assert_eq!(
            take_board_options(ArmBoard::Virt, &m).unwrap_err().message(),
            "Property 'virt-11.1-machine.foo' not found"
        );
    }

    #[test]
    fn acpi_properties() {
        let opts = |props: &[(&str, &str)]| {
            let mut m = QDict::new();
            for (k, v) in props {
                m.put(*k, *v);
            }
            take_board_options(ArmBoard::Virt, &m)
        };
        let o = opts(&[]).unwrap();
        assert!(!o.acpi_off && !o.spcr_off && o.oem_id.is_none() && o.oem_table_id.is_none());
        assert!(opts(&[("acpi", "off")]).unwrap().acpi_off);
        for acpi in ["on", "auto"] {
            let o = opts(&[("acpi", acpi), ("spcr", "off")]).unwrap();
            assert!(!o.acpi_off && o.spcr_off);
        }
        assert_eq!(
            opts(&[("acpi", "maybe")]).unwrap_err().message(),
            "Parameter 'acpi' does not accept value 'maybe'"
        );
        let o = opts(&[("x-oem-id", "RUVM"), ("x-oem-table-id", "RUVMTBL")]).unwrap();
        assert_eq!(o.oem_id.as_deref(), Some("RUVM"));
        assert_eq!(o.oem_table_id.as_deref(), Some("RUVMTBL"));
        assert_eq!(
            opts(&[("x-oem-id", "1234567")]).unwrap_err().message(),
            "User specified oem-id value is bigger than 6 bytes in size"
        );
        assert_eq!(
            opts(&[("x-oem-table-id", "123456789")]).unwrap_err().message(),
            "User specified oem-table-id value is bigger than 8 bytes in size"
        );
    }

    #[test]
    fn sbsa_ref_options() {
        let s = ArmBoard::SbsaRef;
        assert_eq!(board_by_name("sbsa-ref"), Some(s));
        assert_eq!(board_by_name("virt-11.1"), Some(ArmBoard::Virt));
        assert_eq!(board_by_name("sbsa"), None);
        // Four CPUs in one core without -smp, and the usual parsing with it.
        let o = take_board_options(s, &QDict::new()).unwrap();
        assert_eq!((o.board, o.cpus, o.max_cpus, o.topology), (s, 4, 4, None));
        let mut m = QDict::new();
        let mut smp = QDict::new();
        smp.put("cpus", "2");
        m.put("smp", QValue::Dict(smp));
        let o = take_board_options(s, &m).unwrap();
        assert_eq!((o.cpus, o.max_cpus), (2, 2));
        assert_eq!(o.topology.map(|t| t.cores), Some(2));
        let e = parse_smp(s, &smp_cfg(Some(513), None, None)).unwrap_err();
        assert_eq!(
            e.message(),
            "Invalid SMP CPUs 513. The max CPUs supported by machine 'sbsa-ref' is 512"
        );
        let mut m = QDict::new();
        m.put("firmware", "SBSA_FLASH0.fd");
        assert_eq!(take_board_options(s, &m).unwrap().firmware.as_deref(), Some("SBSA_FLASH0.fd"));
        let mut m = QDict::new();
        m.put("secure", "on");
        assert_eq!(
            take_board_options(s, &m).unwrap_err().message(),
            "Property 'sbsa-ref-machine.secure' not found"
        );
        // valid_cpu_types.
        assert_eq!(parse_cpu(s, None).unwrap().name, "neoverse-n1");
        assert_eq!(parse_cpu(s, Some("max")).unwrap().name, "max");
        let e = parse_cpu(s, Some("cortex-a76")).unwrap_err();
        assert_eq!(e.message(), "Invalid CPU model: cortex-a76");
        assert_eq!(
            e.hint_text(),
            Some(
                "The valid models are: cortex-a57, cortex-a72, neoverse-n1, neoverse-v1, neoverse-n2, max\n"
            )
        );
        assert_eq!(
            parse_cpu(s, Some("neoverse-n2")).unwrap_err().message(),
            "CPU model 'neoverse-n2' is not supported by ruvm yet"
        );
        assert!(machine_help_lines().iter().any(|(k, l)| k == "sbsa-ref"
            && l == "sbsa-ref             QEMU 'SBSA Reference' ARM Virtual Machine\n"));
    }
}
