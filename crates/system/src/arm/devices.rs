// SPDX-License-Identifier: GPL-2.0-or-later

//! `-drive` and `-device` for the Arm `virt` and `sbsa-ref` boards, the part of blockdev.c's
//! `drive_new()` and qdev-monitor.c's `qdev_device_add()` that applies to them.
//!
//! A `-drive` without `if=` is a virtio disk on virt, since its `block_default_type` is
//! `IF_VIRTIO`, and every `if=virtio` drive adds a `-device virtio-blk,drive=<id>` after the
//! others. The short virtio names are aliases of the PCI types, as in `qdev_alias_table[]`.
//!
//! On sbsa-ref a `-drive` without `if=` is an IDE drive, and the IDE drive with index `n`
//! (`bus=n/2,unit=n%2`) goes to AHCI port `n` for the first six, as `ahci_ide_create_devs()`
//! takes them. sbsa-ref has no virtio-mmio transports, so only the PCI types plug.
//!
//! `-device` takes `virtio-blk`, `virtio-rng` and `virtio-serial`, each as a PCI function on
//! the root bus `pcie.0` (`virtio-*-pci`) or on a virtio-mmio transport (`virtio-*-device`).
//! The other virtio types and `loader` fail with "... is not supported with this machine by
//! ruvm yet". The PCI types take `addr`, `bus=pcie.0`, `disable-legacy`,
//! `disable-modern` and `vectors`; the virtio-mmio ones take `bus=virtio-mmio-bus.<n>`. Both
//! take `iommu_platform`.
//! On virt, `-device arm-smmuv3,primary-bus=pcie.0` is the SMMUv3 on the platform bus that
//! translates the functions plugged after it.
//! On virt, `-device pcie-root-port` with an `id` puts a root port on `pcie.0`, and the PCI
//! types plug behind it with `bus=<id>`.
//! `-drive if=pflash` with unit 0 or 1 on bus 0 is the drive of the flash of that number, as
//! `pflash_cfi01_legacy_drive()` takes it.

use std::collections::HashSet;
use std::sync::Arc;

use ruvm_base::Error;
use ruvm_base::report::Location;
use ruvm_block::BlockGraph;
use ruvm_hw_iommu::smmuv3::OAS_MODES;
use ruvm_hw_iommu::{SmmuV3Props, VirtioIommuConf};
use ruvm_hw_pci::PcieRootPortConfig;
use ruvm_hw_storage::{BlockBackend, DriveConfig};
use ruvm_hw_virtio::mmio::VIRTIO_MMIO_FORCE_LEGACY_DEFAULT;
use ruvm_hw_virtio::pci::virtio_pci_optimal_num_queues;
use ruvm_hw_virtio::{
    RandomFile, VirtioBalloon, VirtioBlk, VirtioBlkConf, VirtioConsole, VirtioDeviceClass,
    VirtioNet, VirtioNetConf, VirtioPciProps, VirtioRng, VirtioRngConf,
};
use ruvm_machine_arm::sbsa_ref::{NUM_SATA_PORTS, SbsaRefMachine};
use ruvm_machine_arm::virt::VirtMachine;
use ruvm_machine_x86::VirtioHandle;
use ruvm_qapi::opts::{QemuOpts, QemuOptsList, is_help_option};
use ruvm_qapi::visit::{Visitor, VisitorExt};

use super::{ArmBoard, prop_bool};
use crate::drives::{DriveDev, Plugged, device_image, peripheral_path, resize_cb};
use crate::net::Network;
use crate::tpm::TpmModel;
use crate::x86::{Drive, DriveIf, KeepRam, Located, Transport, drive_opts, parse_drive, parse_mac};

/// The virtio device models virt can plug.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Model {
    Blk,
    Rng,
    Serial,
    Iommu,
    Balloon,
    Net,
    /// `arm-smmuv3`, a dynamic sysbus device.
    Smmu,
    /// `tpm-tis-device`, a dynamic sysbus device.
    Tpm,
    /// `pcie-root-port` on the root bus.
    RootPort,
}

/// The device types, with the bus they plug into.
const DEVICE_TYPES: &[(&str, Model, Transport)] = &[
    ("virtio-blk-device", Model::Blk, Transport::Mmio),
    ("virtio-blk-pci", Model::Blk, Transport::Pci),
    ("virtio-rng-device", Model::Rng, Transport::Mmio),
    ("virtio-rng-pci", Model::Rng, Transport::Pci),
    ("virtio-serial-device", Model::Serial, Transport::Mmio),
    ("virtio-serial-pci", Model::Serial, Transport::Pci),
    ("virtio-iommu-device", Model::Iommu, Transport::Mmio),
    ("virtio-iommu-pci", Model::Iommu, Transport::Pci),
    ("virtio-balloon-device", Model::Balloon, Transport::Mmio),
    ("virtio-balloon-pci", Model::Balloon, Transport::Pci),
    ("virtio-net-device", Model::Net, Transport::Mmio),
    ("virtio-net-pci", Model::Net, Transport::Pci),
];

/// `qdev_alias_table[]` for Arm, where the default virtio transport is PCI.
const DEVICE_ALIASES: &[(&str, &str)] = &[
    ("virtio-blk", "virtio-blk-pci"),
    ("virtio-rng", "virtio-rng-pci"),
    ("virtio-serial", "virtio-serial-pci"),
    ("virtio-iommu", "virtio-iommu-pci"),
    ("virtio-net", "virtio-net-pci"),
    ("virtio-scsi", "virtio-scsi-pci"),
    ("virtio-balloon", "virtio-balloon-pci"),
    ("virtio-gpu", "virtio-gpu-pci"),
    ("virtio-input-host", "virtio-input-host-pci"),
    ("virtio-keyboard", "virtio-keyboard-pci"),
    ("virtio-mouse", "virtio-mouse-pci"),
    ("virtio-tablet", "virtio-tablet-pci"),
    ("virtio-9p", "virtio-9p-pci"),
];

/// The root bus of the PCIe host.
const PCIE_ROOT_BUS: &str = "pcie.0";

/// A virtio device to plug.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Plug {
    /// The type name, after aliases.
    pub typename: &'static str,
    model: Model,
    transport: Transport,
    /// Index into the drives, for virtio-blk.
    drive: Option<usize>,
    /// virtio-blk `serial`.
    serial: Option<String>,
    /// `addr` of a PCI function.
    devfn: Option<u8>,
    /// `bus=virtio-mmio-bus.<n>` of a virtio-mmio device.
    mmio_bus: Option<usize>,
    /// `iommu_platform`: the DMA of a PCI function goes through the SMMU. A virtio-mmio
    /// device only offers `VIRTIO_F_IOMMU_PLATFORM`, since no IOMMU is in front of it.
    iommu_platform: bool,
    /// The properties of a virtio-iommu.
    iommu: Option<VirtioIommuConf>,
    /// The properties of an arm-smmuv3, and whether its `primary-bus` is set.
    smmu: Option<(SmmuV3Props, bool)>,
    /// The `tpmdev`, `irq` and `ppi` properties of a tpm-tis-device.
    tpm: Option<(Option<String>, u32, bool)>,
    /// The properties of a pcie-root-port.
    root_port: Option<RootPortConf>,
    /// The `bus` of a PCI function, `None` for the root bus.
    bus: Option<String>,
    /// virtio-net `netdev`.
    netdev: Option<String>,
    /// virtio-net `mac`.
    mac: Option<[u8; 6]>,
    props: VirtioPciProps,
    /// Where `qdev_set_id()` put the device in the QOM tree.
    path: String,
    loc: Option<Location>,
}

/// What virt gets from `-drive` and `-device`, in the order QEMU plugs it.
#[derive(Debug, Default)]
pub(crate) struct Plan {
    /// The virtio devices, `-device` first and then the ones `-drive if=virtio` adds.
    pub virtio: Vec<Plug>,
    /// The drives of the two flashes as indices into the drives, `None` where there is no
    /// `-drive if=pflash`.
    pub pflash: [Option<usize>; 2],
    /// The AHCI ports of sbsa-ref with a drive, as (port, index into the drives).
    pub ide: Vec<(usize, usize)>,
    /// The display devices of `-device` on virt, each with the number of virtio plugs before
    /// it, so they take PCI slots in command line order.
    pub display: Vec<(usize, crate::display::DisplayPlug)>,
}

/// `drive_new()` for each `-drive`, with the default interface of `board`.
pub(crate) fn parse_drives(
    board: ArmBoard,
    args: &[(String, Option<Location>)],
) -> Result<Vec<Drive>, Located> {
    let default_if = match board {
        ArmBoard::Virt => "virtio",
        ArmBoard::SbsaRef => "ide",
    };
    let mut list = drive_opts();
    let mut drives = Vec::new();
    for (arg, loc) in args {
        let fail = |e: Error| Located(loc.clone(), e);
        let has_if = drive_opts().parse(arg, false).map_err(fail)?.get("if").is_some();
        let arg = if has_if { arg.clone() } else { format!("{arg},if={default_if}") };
        let d = parse_drive(&mut list, &arg, None, &drives, loc.clone()).map_err(fail)?;
        drives.push(d);
    }
    Ok(drives)
}

/// `pci_devfn` as a property: `SLOT[.FN]` in hex.
fn parse_devfn(v: &str) -> Option<u8> {
    let (slot, func) = match v.split_once('.') {
        Some((s, f)) => (s, f),
        None => (v, "0"),
    };
    let hex = |s: &str| {
        let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
        if s.is_empty() { None } else { u32::from_str_radix(s, 16).ok() }
    };
    let (slot, func) = (hex(slot)?, hex(func)?);
    (slot <= 31 && func <= 7).then_some((slot << 3 | func) as u8)
}

/// One `-device` that is not `loader`. `mmio` says whether the board has virtio-mmio
/// transports, and `buses` are the names of the buses behind the root ports.
fn plan_virtio(
    drives: &[Drive],
    used: &mut HashSet<usize>,
    driver: &str,
    opts: &QemuOpts,
    loc: &Option<Location>,
    mmio: bool,
    buses: &[String],
) -> Result<Plug, Located> {
    let name = DEVICE_ALIASES.iter().find(|(a, _)| *a == driver).map_or(driver, |(_, t)| *t);
    let Some(&(typename, model, transport)) = DEVICE_TYPES.iter().find(|(t, ..)| *t == name) else {
        if name.starts_with("virtio-") {
            return Err(Located::new(
                loc,
                format!("-device {name} is not supported with this machine by ruvm yet"),
            ));
        }
        return Err(Located::new(loc, format!("'{driver}' is not a valid device model name")));
    };
    let bad_value = |k: &str, v: &str| {
        Located::new(loc, format!("Property '{typename}.{k}' doesn't take value '{v}'"))
    };
    let mut plug = Plug {
        typename,
        model,
        transport,
        drive: None,
        serial: None,
        devfn: None,
        mmio_bus: None,
        iommu_platform: false,
        // The virt compat property sets aw-bits to 48.
        iommu: (model == Model::Iommu)
            .then(|| VirtioIommuConf { aw_bits: 48, ..Default::default() }),
        smmu: None,
        tpm: None,
        root_port: None,
        bus: None,
        netdev: None,
        mac: None,
        props: VirtioPciProps::default(),
        path: String::new(),
        loc: loc.clone(),
    };
    let pci = transport == Transport::Pci;
    if !pci && !mmio {
        let msg = match opts.get("bus") {
            Some(v) => format!("Bus '{v}' not found"),
            None => format!("No 'virtio-bus' bus found for device '{typename}'"),
        };
        return Err(Located::new(loc, msg));
    }
    for (k, v) in opts.iter() {
        if let Some(conf) = plug.iommu.as_mut()
            && let Some(r) = crate::x86::virtio_iommu_prop(conf, k, v, pci, loc)
        {
            r?;
            continue;
        }
        match k {
            "driver" => {}
            "id" => plug.props.id = Some(v.to_string()),
            "bus" => {
                let found = if pci {
                    if v != PCIE_ROOT_BUS {
                        plug.bus = Some(v.to_string());
                    }
                    v == PCIE_ROOT_BUS || buses.iter().any(|b| b == v)
                } else {
                    let n = v.strip_prefix("virtio-mmio-bus.").and_then(|n| n.parse().ok());
                    plug.mmio_bus = n;
                    n.is_some()
                };
                if !found {
                    return Err(Located::new(loc, format!("Bus '{v}' not found")));
                }
            }
            "addr" if pci => plug.devfn = Some(parse_devfn(v).ok_or_else(|| bad_value(k, v))?),
            "disable-legacy" if pci => {
                plug.props.disable_legacy = match v {
                    "auto" => None,
                    _ => Some(prop_bool(k, v).map_err(|_| bad_value(k, v))?),
                };
            }
            "disable-modern" if pci => {
                plug.props.disable_modern = prop_bool(k, v).map_err(|_| bad_value(k, v))?;
            }
            "iommu_platform" => {
                plug.iommu_platform = prop_bool(k, v).map_err(|_| bad_value(k, v))?;
            }
            "vectors" if pci => {
                plug.props.vectors = Some(v.parse().map_err(|_| bad_value(k, v))?);
            }
            "drive" if model == Model::Blk => {
                let Some(i) = drives.iter().position(|d| d.id == v) else {
                    return Err(Located::new(
                        loc,
                        format!("Property '{typename}.drive' can't find value '{v}'"),
                    ));
                };
                plug.drive = Some(i);
            }
            "serial" if model == Model::Blk => plug.serial = Some(v.to_string()),
            "netdev" if model == Model::Net => plug.netdev = Some(v.to_string()),
            "mac" if model == Model::Net => {
                plug.mac = Some(parse_mac(v).ok_or_else(|| bad_value(k, v))?)
            }
            _ => return Err(Located::new(loc, format!("Property '{typename}.{k}' not found"))),
        }
    }
    if model == Model::Blk {
        let Some(i) = plug.drive else {
            return Err(Located::new(loc, "drive property not set"));
        };
        let d = &drives[i];
        if !used.insert(i) {
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
        if d.file.is_none() {
            return Err(Located::new(loc, "Device needs media, but drive is empty"));
        }
    }
    Ok(plug)
}

/// The `arm-smmuv3` type name.
const TYPE_ARM_SMMUV3: &str = "arm-smmuv3";

/// The `arm-smmuv3` properties, with the checks `qdev_device_add()` makes before realize.
fn plan_smmuv3(board: ArmBoard, opts: &QemuOpts, loc: &Option<Location>) -> Result<Plug, Located> {
    if board != ArmBoard::Virt {
        return Err(Located::new(
            loc,
            "Parameter 'driver' expects a dynamic sysbus device type for the machine",
        ));
    }
    let at = |e: Error| Located(loc.clone(), e);
    let not_accepted = |k: &str, v: &str| {
        Located::new(loc, format!("Parameter '{k}' does not accept value '{v}'"))
    };
    let on_off_auto = |k: &str, v: &str| match v {
        "on" => Ok(Some(true)),
        "off" => Ok(Some(false)),
        "auto" => Ok(None),
        _ => Err(not_accepted(k, v)),
    };
    let mut props = SmmuV3Props::default();
    let mut primary_bus = false;
    for (k, v) in opts.iter() {
        match k {
            "driver" | "id" => {}
            "bus" if v == "main-system-bus" => {}
            "bus" => return Err(Located::new(loc, format!("Bus '{v}' not found"))),
            // object_resolve_link(): the root bus is the only PCI bus.
            "primary-bus" if v == PCIE_ROOT_BUS => primary_bus = true,
            "primary-bus" => return Err(Located::new(loc, format!("Device '{v}' not found"))),
            "stage" => props.stage = Some(v.to_string()),
            "accel" | "smmu_per_bus" => {
                let b = crate::display::prop(k, v, |vi, n, out| vi.type_bool(Some(n), out))
                    .map_err(at)?;
                if k == "accel" {
                    props.accel = b;
                }
            }
            "ril" => props.ril = on_off_auto(k, v)?,
            "ats" => props.ats = on_off_auto(k, v)?,
            "cmdqv" => props.cmdqv = on_off_auto(k, v)?,
            "oas" => {
                props.oas = match v {
                    "auto" => None,
                    _ => Some(
                        v.parse()
                            .ok()
                            .filter(|b| OAS_MODES.contains(b))
                            .ok_or_else(|| not_accepted(k, v))?,
                    ),
                }
            }
            "ssidsize" => {
                props.ssidsize = match v {
                    "auto" => None,
                    _ => Some(
                        v.parse()
                            .ok()
                            .filter(|&b: &u8| b <= 20 && b.to_string() == v)
                            .ok_or_else(|| not_accepted(k, v))?,
                    ),
                }
            }
            // virt sets the identifier and the memory itself.
            "identifier" | "bus_num" => {
                crate::display::prop(k, v, |vi, n, out| vi.type_uint8(Some(n), out)).map_err(at)?;
            }
            "msi-gpa" => {
                crate::display::prop(k, v, |vi, n, out| vi.type_uint64(Some(n), out))
                    .map_err(at)?;
            }
            _ => {
                return Err(Located::new(
                    loc,
                    format!("Property '{TYPE_ARM_SMMUV3}.{k}' not found"),
                ));
            }
        }
    }
    Ok(Plug {
        typename: TYPE_ARM_SMMUV3,
        model: Model::Smmu,
        transport: Transport::Mmio,
        drive: None,
        serial: None,
        devfn: None,
        mmio_bus: None,
        iommu_platform: false,
        iommu: None,
        smmu: Some((props, primary_bus)),
        tpm: None,
        root_port: None,
        bus: None,
        netdev: None,
        mac: None,
        props: VirtioPciProps::default(),
        path: String::new(),
        loc: loc.clone(),
    })
}

/// `TYPE_TPM_TIS_SYSBUS`.
const TYPE_TPM_TIS_SYSBUS: &str = "tpm-tis-device";

/// The `tpm-tis-device` properties.
fn plan_tpm_tis_device(
    board: ArmBoard,
    opts: &QemuOpts,
    loc: &Option<Location>,
) -> Result<Plug, Located> {
    if board != ArmBoard::Virt {
        return Err(Located::new(
            loc,
            "Parameter 'driver' expects a dynamic sysbus device type for the machine",
        ));
    }
    let at = |e: Error| Located(loc.clone(), e);
    let (mut tpmdev, mut irq, mut ppi) = (None, 5, true);
    for (k, v) in opts.iter() {
        match k {
            "driver" | "id" => {}
            "bus" if v == "main-system-bus" => {}
            "bus" => {
                let mmio = v.strip_prefix("virtio-mmio-bus.").and_then(|n| n.parse::<u32>().ok());
                let bus_type = match mmio {
                    _ if v == PCIE_ROOT_BUS => "PCIE",
                    Some(n) if n < 32 => "virtio-mmio-bus",
                    _ => return Err(Located::new(loc, format!("Bus '{v}' not found"))),
                };
                let msg = format!("Device '{TYPE_TPM_TIS_SYSBUS}' can't go on {bus_type} bus");
                return Err(Located::new(loc, msg));
            }
            "tpmdev" => tpmdev = Some(v.to_string()),
            "irq" => {
                irq = crate::display::prop(k, v, |vi, n, out| vi.type_uint32(Some(n), out))
                    .map_err(at)?;
            }
            "ppi" => {
                ppi = crate::display::prop(k, v, |vi, n, out| vi.type_bool(Some(n), out))
                    .map_err(at)?;
            }
            _ => {
                return Err(Located::new(
                    loc,
                    format!("Property '{TYPE_TPM_TIS_SYSBUS}.{k}' not found"),
                ));
            }
        }
    }
    Ok(Plug {
        typename: TYPE_TPM_TIS_SYSBUS,
        model: Model::Tpm,
        transport: Transport::Mmio,
        drive: None,
        serial: None,
        devfn: None,
        mmio_bus: None,
        iommu_platform: false,
        iommu: None,
        smmu: None,
        tpm: Some((tpmdev, irq, ppi)),
        root_port: None,
        bus: None,
        netdev: None,
        mac: None,
        props: VirtioPciProps::default(),
        path: String::new(),
        loc: loc.clone(),
    })
}

/// `TYPE_GEN_PCIE_ROOT_PORT`.
const TYPE_GEN_PCIE_ROOT_PORT: &str = "pcie-root-port";

/// The `pcie-root-port` properties ruvm takes. The secondary bus is named after `id`.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RootPortConf {
    id: String,
    chassis: u8,
    slot: u16,
    port: u8,
    hotplug: bool,
    multifunction: bool,
}

/// The `pcie-root-port` properties QEMU has that ruvm does not take yet.
const ROOT_PORT_UNSUPPORTED: &[&str] = &[
    "acpi-index",
    "aer_log_max",
    "bus-reserve",
    "busnr",
    "failover_pair_id",
    "io-reserve",
    "mem-reserve",
    "power_controller_present",
    "pref32-reserve",
    "pref64-reserve",
    "rombar",
    "romfile",
    "romsize",
    "sriov-pf",
    "x-do-not-expose-native-hotplug-cap",
    "x-max-bounce-buffer-size",
    "x-pci-express-writeable-slt-bug",
    "x-pcie-ari-nextfn-1",
    "x-pcie-err-unc-mask",
    "x-pcie-ext-tag",
    "x-pcie-lnksta-dllla",
    "x-speed",
    "x-width",
];

/// The `pcie-root-port` properties, on the root bus of virt. `buses` are the names of the
/// buses behind the root ports planned before it.
fn plan_root_port(
    board: ArmBoard,
    opts: &QemuOpts,
    loc: &Option<Location>,
    buses: &[String],
) -> Result<Plug, Located> {
    if board != ArmBoard::Virt {
        let msg = "-device pcie-root-port is not supported with machine sbsa-ref by ruvm yet";
        return Err(Located::new(loc, msg));
    }
    let at = |e: Error| Located(loc.clone(), e);
    let bad_value = |k: &str, v: &str| {
        Located::new(
            loc,
            format!("Property '{TYPE_GEN_PCIE_ROOT_PORT}.{k}' doesn't take value '{v}'"),
        )
    };
    let mut conf = RootPortConf {
        id: opts.id().unwrap_or_default().to_string(),
        chassis: 0,
        slot: 0,
        port: 0,
        hotplug: true,
        multifunction: false,
    };
    let mut devfn = None;
    for (k, v) in opts.iter() {
        match k {
            "driver" => {}
            "bus" if v == PCIE_ROOT_BUS => {}
            "bus" if buses.iter().any(|b| b == v) => {
                let msg = "-device pcie-root-port behind a root port is not supported by ruvm yet";
                return Err(Located::new(loc, msg));
            }
            "bus" => return Err(Located::new(loc, format!("Bus '{v}' not found"))),
            "addr" => devfn = Some(parse_devfn(v).ok_or_else(|| bad_value(k, v))?),
            "chassis" | "port" => {
                let n = crate::display::prop(k, v, |vi, n, out| vi.type_uint8(Some(n), out))
                    .map_err(at)?;
                if k == "chassis" {
                    conf.chassis = n;
                } else {
                    conf.port = n;
                }
            }
            "slot" => {
                conf.slot = crate::display::prop(k, v, |vi, n, out| vi.type_uint16(Some(n), out))
                    .map_err(at)?;
            }
            "hotplug" | "multifunction" => {
                let b = crate::display::prop(k, v, |vi, n, out| vi.type_bool(Some(n), out))
                    .map_err(at)?;
                if k == "hotplug" {
                    conf.hotplug = b;
                } else {
                    conf.multifunction = b;
                }
            }
            _ if ROOT_PORT_UNSUPPORTED.contains(&k) => {
                let msg = format!(
                    "Property '{TYPE_GEN_PCIE_ROOT_PORT}.{k}' is not supported by ruvm yet"
                );
                return Err(Located::new(loc, msg));
            }
            _ => {
                return Err(Located::new(
                    loc,
                    format!("Property '{TYPE_GEN_PCIE_ROOT_PORT}.{k}' not found"),
                ));
            }
        }
    }
    if conf.id.is_empty() {
        // QEMU names the bus of a root port without an id "pcie.<n>", with an n that depends on
        // the other buses, so nothing could plug behind it.
        let msg = "-device pcie-root-port without id is not supported by ruvm yet";
        return Err(Located::new(loc, msg));
    }
    Ok(Plug {
        typename: TYPE_GEN_PCIE_ROOT_PORT,
        model: Model::RootPort,
        transport: Transport::Pci,
        drive: None,
        serial: None,
        devfn,
        mmio_bus: None,
        iommu_platform: false,
        iommu: None,
        smmu: None,
        tpm: None,
        root_port: Some(conf),
        bus: None,
        netdev: None,
        mac: None,
        props: VirtioPciProps::default(),
        path: String::new(),
        loc: loc.clone(),
    })
}

/// What one `-device` plugs.
enum Planned {
    Display(crate::display::DisplayPlug),
    Virtio(Plug),
}

/// `qdev_device_add()` for one `-device` on `board`, with `-global` merged in.
fn plan_device(
    board: ArmBoard,
    drives: &[Drive],
    used: &mut HashSet<usize>,
    buses: &[String],
    arg: &str,
    loc: &Option<Location>,
) -> Result<Planned, Located> {
    let mut list = QemuOptsList::new("device", &[]).with_implied_opt_name("driver");
    let opts = list.parse(arg, true).map_err(|e| Located(loc.clone(), e))?;
    let Some(driver) = opts.get("driver") else {
        return Err(Located::new(loc, "Parameter 'driver' is missing"));
    };
    if is_help_option(driver) || opts.has_help_opt() {
        return Err(Located::new(loc, "-device help is not supported by ruvm yet"));
    }
    if driver == "loader" {
        let msg = "-device loader is not supported with this machine by ruvm yet";
        return Err(Located::new(loc, msg));
    }
    if let Some(plug) = crate::display::plan_device(driver, opts, loc, true, true) {
        let plug = plug?;
        if board == ArmBoard::SbsaRef {
            return Err(Located::new(loc, sbsa_ref_display(plug.typename())));
        }
        return Ok(Planned::Display(plug));
    }
    if driver == TYPE_ARM_SMMUV3 {
        return plan_smmuv3(board, opts, loc).map(Planned::Virtio);
    }
    if driver == TYPE_TPM_TIS_SYSBUS {
        return plan_tpm_tis_device(board, opts, loc).map(Planned::Virtio);
    }
    if driver == TYPE_GEN_PCIE_ROOT_PORT {
        return plan_root_port(board, opts, loc, buses).map(Planned::Virtio);
    }
    let mmio = board == ArmBoard::Virt;
    plan_virtio(drives, used, driver, opts, loc, mmio, buses).map(Planned::Virtio)
}

/// Works out what `-drive` and `-device` plug into `board`. The errors are those of
/// `qdev_device_add()` and `drive_check_orphaned()`; for orphans every one is listed.
pub(crate) fn plan(
    board: ArmBoard,
    drives: &[Drive],
    devices: &[(String, Option<Location>)],
) -> Result<Plan, Vec<Located>> {
    let mut p = Plan::default();
    let mut used = HashSet::new();
    // create_ahci() takes the IDE drives with index 0 to 5 while the board is built, before
    // any -device.
    if board == ArmBoard::SbsaRef {
        for (i, d) in drives.iter().enumerate() {
            let port = d.bus as usize * 2 + d.unit as usize;
            if d.iface == DriveIf::Ide && port < NUM_SATA_PORTS {
                p.ide.push((port, i));
                used.insert(i);
            }
        }
    }
    let mut queue: Vec<(String, Option<Location>)> = devices.to_vec();
    // drive_new() adds a -device virtio-blk for every if=virtio drive, after the others.
    for d in drives.iter().filter(|d| d.iface == DriveIf::Virtio) {
        queue.push((format!("virtio-blk,drive={}", d.id), d.loc.clone()));
    }
    let mut anon = 0;
    // The secondary buses of the root ports planned so far.
    let mut buses: Vec<String> = Vec::new();
    for (arg, loc) in &queue {
        let path = peripheral_path(arg, &mut anon);
        let plan = |arg: &str, trial: bool| {
            if trial {
                plan_device(board, drives, &mut used.clone(), &buses, arg, loc)
            } else {
                plan_device(board, drives, &mut used, &buses, arg, loc)
            }
        };
        let planned = crate::global::plan_device(arg, loc, plan, |_, e| Err(e));
        match planned.map_err(|e| vec![e])? {
            Planned::Display(plug) => p.display.push((p.virtio.len(), plug)),
            Planned::Virtio(plug) => {
                if let Some(rp) = &plug.root_port {
                    buses.push(rp.id.clone());
                }
                p.virtio.push(Plug { path, ..plug });
            }
        }
    }

    // pflash_cfi01_legacy_drive() for the two flashes, which read their drives through the
    // block layer.
    for (i, d) in drives.iter().enumerate() {
        if d.iface != DriveIf::Pflash || d.bus != 0 || d.unit >= 2 {
            continue;
        }
        used.insert(i);
        p.pflash[d.unit as usize] = Some(i);
    }

    let orphans: Vec<Located> = drives
        .iter()
        .enumerate()
        .filter(|(i, d)| {
            !used.contains(i) && !matches!(d.iface, DriveIf::None | DriveIf::Virtio | DriveIf::Xen)
        })
        .map(|(_, d)| {
            let msg = format!(
                "machine type does not support if={},bus={},unit={}",
                d.iface.name(),
                d.bus,
                d.unit
            );
            Located::new(&d.loc, msg)
        })
        .collect();
    if !orphans.is_empty() {
        return Err(orphans);
    }
    Ok(p)
}

/// The board the virtio devices plug into.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Target<'a> {
    Virt(&'a VirtMachine),
    SbsaRef(&'a SbsaRefMachine),
}

impl Target<'_> {
    fn smp(self) -> usize {
        match self {
            Target::Virt(b) => b.smp(),
            Target::SbsaRef(b) => b.smp(),
        }
    }

    fn ram_size(self) -> u64 {
        match self {
            Target::Virt(b) => b.ram_size(),
            Target::SbsaRef(b) => b.ram_size(),
        }
    }
}

/// The error for a display device on sbsa-ref, whose display devices ruvm does not build yet.
fn sbsa_ref_display(typename: &str) -> String {
    format!("-device {typename} is not supported with machine sbsa-ref by ruvm yet")
}

/// `qdev_device_add()` for the planned devices: builds each one and plugs it into its bus, a
/// virtio-net joined to its netdev in `net`. Gives the drives it put on virtio-blk devices.
pub(crate) fn plug(
    graph: &BlockGraph,
    board: Target<'_>,
    plugs: &[Plug],
    display: &[(usize, crate::display::DisplayPlug)],
    drives: &[Drive],
    net: &Network,
) -> Result<Vec<Plugged>, Located> {
    let realize = |d: &crate::display::DisplayPlug| match board {
        Target::Virt(b) => crate::display::realize_virt(b, d),
        Target::SbsaRef(_) => Err(Located::new(&d.loc, sbsa_ref_display(d.typename()))),
    };
    let mut display = display.iter().peekable();
    let mut plugged = Vec::new();
    for (i, plug) in plugs.iter().enumerate() {
        while let Some((_, d)) = display.next_if(|(at, _)| *at == i) {
            realize(d)?;
        }
        let at = |e: String| Located::new(&plug.loc, e);
        let (mut image, mut nic) = (None, None);
        let class: Box<dyn VirtioDeviceClass> = match plug.model {
            Model::Blk => {
                let drive = plug.drive.expect("planned with a drive");
                let d = &drives[drive];
                // virtio_blk_device_realize()
                let readonly = graph.backend(&d.id).is_some_and(|b| b.is_read_only());
                let backend = device_image(graph, d, readonly, true)
                    .map_err(|e| Located(plug.loc.clone(), e))?;
                // blk_get_attached_dev_id(): the id of a virtio-blk-device if it has one.
                // virtio-blk-pci holds a virtio-blk without an id.
                let qdev = match plug.transport {
                    Transport::Pci => format!("{}/virtio-backend", plug.path),
                    Transport::Mmio => {
                        let id = plug.path.strip_prefix("/machine/peripheral/");
                        id.unwrap_or(&plug.path).to_string()
                    }
                };
                image = Some((drive, qdev));
                // virtio_blk_pci_realize() turns the automatic num-queues into one queue per
                // vCPU. virtio-blk-device keeps one.
                let num_queues = match plug.transport {
                    Transport::Pci => {
                        let cpus = u32::try_from(board.smp()).unwrap_or(u32::MAX);
                        Some(virtio_pci_optimal_num_queues(0, cpus) as u16)
                    }
                    Transport::Mmio => None,
                };
                let conf = VirtioBlkConf {
                    serial: plug.serial.clone(),
                    num_queues,
                    ..VirtioBlkConf::default()
                };
                Box::new(VirtioBlk::new(Box::new(backend), conf))
            }
            Model::Rng => {
                Box::new(VirtioRng::new(Box::<RandomFile>::default(), VirtioRngConf::default()))
            }
            Model::Serial => Box::new(VirtioConsole::new(None)),
            Model::Iommu => {
                let conf = plug.iommu.as_ref().expect("planned with a config");
                match (board, plug.transport) {
                    (Target::Virt(board), Transport::Pci) => {
                        let id = plug.props.id.clone();
                        board.attach_virtio_iommu(conf, plug.devfn, id).map_err(at)?;
                    }
                    (Target::Virt(_), Transport::Mmio) => {
                        return Err(at("VIRTIO-IOMMU is not attached to any PCI bus!".into()));
                    }
                    (Target::SbsaRef(_), _) => {
                        let msg = "is not supported with machine sbsa-ref by ruvm yet";
                        return Err(at(format!("-device {} {msg}", plug.typename)));
                    }
                }
                continue;
            }
            Model::Smmu => {
                let (props, primary_bus) = plug.smmu.as_ref().expect("planned with properties");
                match board {
                    Target::Virt(board) => board.attach_smmuv3(props, *primary_bus).map_err(at)?,
                    Target::SbsaRef(_) => unreachable!("planned only on virt"),
                }
                continue;
            }
            Model::RootPort => {
                let rp = plug.root_port.as_ref().expect("planned with properties");
                let Target::Virt(board) = board else { unreachable!("planned only on virt") };
                let cfg = PcieRootPortConfig {
                    id: Some(rp.id.clone()),
                    sec_bus_name: rp.id.clone(),
                    chassis: rp.chassis,
                    slot: rp.slot,
                    port: rp.port,
                    hotplug: rp.hotplug,
                    multifunction: rp.multifunction,
                    ..PcieRootPortConfig::default()
                };
                board.attach_pcie_root_port(&cfg, plug.devfn).map_err(at)?;
                continue;
            }
            Model::Tpm => {
                let (tpmdev, irq, ppi) = plug.tpm.as_ref().expect("planned with properties");
                let Target::Virt(board) = board else { unreachable!("planned only on virt") };
                let prop = format!("{TYPE_TPM_TIS_SYSBUS}.tpmdev");
                let be =
                    crate::tpm::realize(TpmModel::TpmTis, &prop, tpmdev.as_deref()).map_err(at)?;
                board.attach_tpm_tis_device(be, *irq, *ppi).map_err(at)?;
                continue;
            }
            Model::Balloon => Box::new(VirtioBalloon::new(board.ram_size(), Box::new(KeepRam))),
            Model::Net => {
                let mut mac = plug.mac.unwrap_or_default();
                let id = plug.props.id.as_deref();
                let (peer, port) =
                    net.new_nic(plug.typename, id, plug.netdev.as_deref(), &mut mac).map_err(at)?;
                nic = Some(port);
                let conf = VirtioNetConf { mac, ..VirtioNetConf::default() };
                Box::new(VirtioNet::new(conf, Some(peer)))
            }
        };
        let (devfn, iommu) = (plug.devfn, plug.iommu_platform);
        let handle = match (board, plug.transport, plug.mmio_bus) {
            (Target::Virt(board), Transport::Pci, _) => {
                let bus = plug.bus.as_deref();
                let dev = board
                    .attach_virtio_pci_on(bus, class, devfn, &plug.props, iommu)
                    .map_err(at)?;
                Some(VirtioHandle::Pci(dev))
            }
            (Target::SbsaRef(board), Transport::Pci, _) => {
                let dev =
                    board.attach_virtio_pci_with(class, devfn, &plug.props, iommu).map_err(at)?;
                Some(VirtioHandle::Pci(dev))
            }
            (Target::Virt(board), Transport::Mmio, Some(n)) => {
                board
                    .attach_virtio_at_with(n, class, VIRTIO_MMIO_FORCE_LEGACY_DEFAULT, iommu)
                    .map_err(at)?;
                board.virtio_transport(n).map(VirtioHandle::Mmio)
            }
            (Target::Virt(board), Transport::Mmio, None) => {
                let n = board
                    .attach_virtio_with(class, iommu)
                    .map_err(|e| at(format!("{e} '{}'", plug.typename)))?;
                board.virtio_transport(n).map(VirtioHandle::Mmio)
            }
            (Target::SbsaRef(_), Transport::Mmio, _) => {
                return Err(at(format!(
                    "No 'virtio-bus' bus found for device '{}'",
                    plug.typename
                )));
            }
        };
        if let Some((drive, qdev)) = image {
            let resize_cb = Some(resize_cb(handle.as_ref()));
            plugged.push(Plugged { drive, dev: DriveDev::disk(qdev), resize_cb });
        }
        if let (Some(nic), Some(handle)) = (nic, handle) {
            nic.connect(handle);
        }
    }
    for (_, d) in display {
        realize(d)?;
    }
    Ok(plugged)
}

/// `ahci_ide_create_devs()` for the planned IDE drives of sbsa-ref: an `ide-cd` or `ide-hd`
/// on each AHCI port with a drive. Gives the drives it put on them.
pub(crate) fn plug_ahci(
    graph: &BlockGraph,
    board: &SbsaRefMachine,
    ide: &[(usize, usize)],
    drives: &[Drive],
) -> Result<Vec<Plugged>, Located> {
    let mut plugged = Vec::new();
    for &(port, i) in ide {
        let d = &drives[i];
        // ide_dev_initfn()
        let blk: Option<Arc<dyn BlockBackend>> = match &d.file {
            Some(_) => {
                // The board adds the device, with no location to report.
                let image =
                    device_image(graph, d, d.cdrom, !d.cdrom).map_err(|e| Located(None, e))?;
                Some(Arc::new(image))
            }
            None => None,
        };
        if !d.cdrom && blk.is_none() {
            return Err(Located::new(&d.loc, "Device needs media, but drive is empty"));
        }
        let config = if d.cdrom { DriveConfig::cdrom() } else { DriveConfig::hd() };
        board.attach_drive(port, config, blk).map_err(|e| Located::new(&d.loc, e))?;
        // The IDE devices come after the CPUs and the ten devices sbsa-ref makes before the
        // AHCI controller in /machine/unattached, in the order of their ports.
        let before = ide.iter().filter(|(p, _)| *p < port).count();
        let qdev = format!("/machine/unattached/device[{}]", board.smp() + 10 + before);
        let dev = if d.cdrom {
            DriveDev { qdev, removable: true, iostatus: true }
        } else {
            DriveDev::disk(qdev)
        };
        plugged.push(Plugged { drive: i, dev, resize_cb: None });
    }
    Ok(plugged)
}

#[cfg(test)]
mod tests {
    use ruvm_base::ClockType;
    use ruvm_hw_core::Clock;

    use super::*;
    use crate::net::NetOption;

    const V: ArmBoard = ArmBoard::Virt;
    const S: ArmBoard = ArmBoard::SbsaRef;

    fn dev(arg: &str) -> (String, Option<Location>) {
        (arg.to_string(), None)
    }

    #[test]
    fn drives_default_to_virtio() {
        let d = parse_drives(V, &[dev("file=a.img"), dev("file=b.img,if=none,id=d0")]).unwrap();
        assert_eq!(d[0].iface, DriveIf::Virtio);
        assert_eq!(d[0].id, "virtio0");
        assert_eq!(d[1].iface, DriveIf::None);
        let p = plan(V, &d, &[dev("virtio-blk-pci,drive=d0,addr=3")]).unwrap();
        assert_eq!(p.virtio.len(), 2);
        assert_eq!(p.virtio[0].devfn, Some(3 << 3));
        assert_eq!(p.virtio[0].drive, Some(1));
        assert_eq!(p.virtio[1].typename, "virtio-blk-pci");
        assert_eq!(p.virtio[1].drive, Some(0));
    }

    #[test]
    fn devices() {
        let p = plan(V, &[], &[dev("virtio-rng")]).unwrap();
        assert_eq!(p.virtio[0].typename, "virtio-rng-pci");
        assert_eq!(p.virtio[0].transport, Transport::Pci);
        let p = plan(V, &[], &[dev("virtio-balloon"), dev("virtio-balloon-device")]).unwrap();
        assert_eq!(p.virtio[0].typename, "virtio-balloon-pci");
        assert_eq!(p.virtio[1].transport, Transport::Mmio);
        let p = plan(V, &[], &[dev("virtio-rng-device,bus=virtio-mmio-bus.2")]).unwrap();
        assert_eq!(p.virtio[0].mmio_bus, Some(2));
        let p = plan(V, &[], &[dev("virtio-serial-pci,disable-legacy=on,vectors=0,addr=1f.7")])
            .unwrap();
        assert_eq!(p.virtio[0].props.disable_legacy, Some(true));
        assert_eq!(p.virtio[0].props.vectors, Some(0));
        assert_eq!(p.virtio[0].devfn, Some(0xff));
        let p = plan(V, &[], &[dev("virtio-rng-pci,iommu_platform=on")]).unwrap();
        assert!(p.virtio[0].iommu_platform);
        let p = plan(V, &[], &[dev("virtio-rng-device,iommu_platform=on")]).unwrap();
        assert!(p.virtio[0].iommu_platform);

        let err = |args: &[&str]| {
            let devs: Vec<_> = args.iter().map(|a| dev(a)).collect();
            plan(V, &[], &devs).unwrap_err()[0].1.message().to_string()
        };
        assert_eq!(
            err(&["virtio-rng-pci,addr=20"]),
            "Property 'virtio-rng-pci.addr' doesn't take value '20'"
        );
        assert_eq!(err(&["virtio-rng-pci,bus=pci.0"]), "Bus 'pci.0' not found");
        assert_eq!(
            err(&["virtio-rng-device,addr=1"]),
            "Property 'virtio-rng-device.addr' not found"
        );
        assert_eq!(err(&["virtio-blk-pci"]), "drive property not set");
        assert_eq!(
            err(&["virtio-scsi-device"]),
            "-device virtio-scsi-device is not supported with this machine by ruvm yet"
        );
        assert_eq!(err(&["e1000"]), "'e1000' is not a valid device model name");
        assert_eq!(
            err(&["loader,addr=0x40000000,data=1,data-len=4"]),
            "-device loader is not supported with this machine by ruvm yet"
        );
    }

    #[test]
    fn pcie_root_port() {
        let devs = [
            dev("pcie-root-port,id=rp1,chassis=1,slot=2,port=3,addr=4,hotplug=off"),
            dev("virtio-rng-pci,bus=rp1"),
            dev("virtio-rng-pci,bus=pcie.0"),
        ];
        let p = plan(V, &[], &devs).unwrap();
        let rp = p.virtio[0].root_port.clone().unwrap();
        assert_eq!((rp.id.as_str(), rp.chassis, rp.slot, rp.port), ("rp1", 1, 2, 3));
        assert!(!rp.hotplug);
        assert_eq!(p.virtio[0].devfn, Some(4 << 3));
        assert_eq!(p.virtio[1].bus.as_deref(), Some("rp1"));
        assert_eq!(p.virtio[2].bus, None);

        let err = |board, args: &[&str]| {
            let devs: Vec<_> = args.iter().map(|a| dev(a)).collect();
            plan(board, &[], &devs).unwrap_err()[0].1.message().to_string()
        };
        // A bus is there only once its root port is.
        let e = err(V, &["virtio-rng-pci,bus=rp1", "pcie-root-port,id=rp1"]);
        assert_eq!(e, "Bus 'rp1' not found");
        assert_eq!(err(V, &["pcie-root-port,id=rp1,bus=rp0"]), "Bus 'rp0' not found");
        assert_eq!(
            err(V, &["pcie-root-port,id=rp1", "pcie-root-port,id=rp2,bus=rp1"]),
            "-device pcie-root-port behind a root port is not supported by ruvm yet"
        );
        assert_eq!(
            err(V, &["pcie-root-port,id=rp1,io-reserve=0"]),
            "Property 'pcie-root-port.io-reserve' is not supported by ruvm yet"
        );
        assert_eq!(
            err(V, &["pcie-root-port,id=rp1,foo=1"]),
            "Property 'pcie-root-port.foo' not found"
        );
        assert_eq!(
            err(V, &["pcie-root-port,chassis=1"]),
            "-device pcie-root-port without id is not supported by ruvm yet"
        );
        assert_eq!(
            err(S, &["pcie-root-port,id=rp1"]),
            "-device pcie-root-port is not supported with machine sbsa-ref by ruvm yet"
        );
    }

    #[test]
    fn arm_smmuv3() {
        let p = plan(V, &[], &[dev("arm-smmuv3,primary-bus=pcie.0,stage=nested,oas=44")]);
        let (props, primary) = p.unwrap().virtio[0].smmu.clone().unwrap();
        assert!(primary);
        assert_eq!((props.stage.as_deref(), props.oas), (Some("nested"), Some(44)));
        let p = plan(V, &[], &[dev("arm-smmuv3,ril=off,ssidsize=20,cmdqv=auto,identifier=7")]);
        let (props, primary) = p.unwrap().virtio[0].smmu.clone().unwrap();
        assert!(!primary);
        assert_eq!((props.ril, props.ssidsize, props.cmdqv), (Some(false), Some(20), None));

        let err = |arg: &str| plan(V, &[], &[dev(arg)]).unwrap_err()[0].1.message().to_string();
        assert_eq!(err("arm-smmuv3,primary-bus=pcie.1"), "Device 'pcie.1' not found");
        assert_eq!(err("arm-smmuv3,oas=45"), "Parameter 'oas' does not accept value '45'");
        assert_eq!(
            err("arm-smmuv3,ssidsize=21"),
            "Parameter 'ssidsize' does not accept value '21'"
        );
        assert_eq!(err("arm-smmuv3,ats=yes"), "Parameter 'ats' does not accept value 'yes'");
        assert_eq!(err("arm-smmuv3,accel=2"), "Parameter 'accel' expects 'on' or 'off'");
        assert_eq!(err("arm-smmuv3,identifier=256"), "Parameter 'identifier' expects uint8_t");
        assert_eq!(err("arm-smmuv3,bus=pcie.0"), "Bus 'pcie.0' not found");
        assert_eq!(err("arm-smmuv3,foo=1"), "Property 'arm-smmuv3.foo' not found");
        let e = plan(S, &[], &[dev("arm-smmuv3")]).unwrap_err();
        assert_eq!(
            e[0].1.message(),
            "Parameter 'driver' expects a dynamic sysbus device type for the machine"
        );
    }

    #[test]
    fn tpm_tis_device() {
        let p = plan(V, &[], &[dev("tpm-tis-device,tpmdev=t0,irq=9,ppi=off")]).unwrap();
        assert_eq!(p.virtio[0].tpm, Some((Some("t0".to_string()), 9, false)));
        let p = plan(V, &[], &[dev("tpm-tis-device,bus=main-system-bus")]).unwrap();
        assert_eq!(p.virtio[0].tpm, Some((None, 5, true)));
        let err = |arg: &str| plan(V, &[], &[dev(arg)]).unwrap_err()[0].1.message().to_string();
        assert_eq!(err("tpm-tis-device,ppi=2"), "Parameter 'ppi' expects 'on' or 'off'");
        assert_eq!(err("tpm-tis-device,foo=1"), "Property 'tpm-tis-device.foo' not found");
        let msg = "Device 'tpm-tis-device' can't go on PCIE bus";
        assert_eq!(err("tpm-tis-device,bus=pcie.0"), msg);
        let msg = "Device 'tpm-tis-device' can't go on virtio-mmio-bus bus";
        assert_eq!(err("tpm-tis-device,bus=virtio-mmio-bus.31"), msg);
        assert_eq!(
            err("tpm-tis-device,bus=virtio-mmio-bus.32"),
            "Bus 'virtio-mmio-bus.32' not found"
        );
        assert_eq!(err("tpm-tis-device,bus=pcie.1"), "Bus 'pcie.1' not found");
        let e = plan(S, &[], &[dev("tpm-tis-device")]).unwrap_err();
        assert_eq!(
            e[0].1.message(),
            "Parameter 'driver' expects a dynamic sysbus device type for the machine"
        );
    }

    #[test]
    fn virtio_iommu() {
        let p = plan(V, &[], &[dev("virtio-iommu")]).unwrap();
        assert_eq!(p.virtio[0].typename, "virtio-iommu-pci");
        let conf = p.virtio[0].iommu.clone().unwrap();
        assert_eq!((conf.boot_bypass, conf.aw_bits), (true, 48));
        let p = plan(V, &[], &[dev("virtio-iommu-pci,aw-bits=40,boot-bypass=off,addr=3")]);
        let plug = &p.unwrap().virtio[0];
        let conf = plug.iommu.clone().unwrap();
        assert_eq!((conf.boot_bypass, conf.aw_bits, plug.devfn), (false, 40, Some(3 << 3)));

        let err = |arg: &str| plan(V, &[], &[dev(arg)]).unwrap_err()[0].1.message().to_string();
        assert_eq!(
            err("virtio-iommu-pci,granule=2k"),
            "Parameter 'granule' does not accept value '2k'"
        );
        assert_eq!(
            err("virtio-iommu-pci,reserved-regions=1"),
            "Invalid parameter type for 'reserved-regions', expected: array"
        );
        assert_eq!(
            err("virtio-iommu-device,reserved-regions=1"),
            "Property 'virtio-iommu-device.reserved-regions' not found"
        );
        assert_eq!(err("virtio-iommu-pci,aw-bits=abc"), "Parameter 'aw-bits' expects integer");
    }

    fn no_net() -> Network {
        Network::new(&[], false, &Clock::manual(ClockType::Virtual), None).unwrap()
    }

    #[test]
    fn virtio_net() {
        let plan_one = |a: &str| plan(V, &[], &[dev(a)]).map(|mut p| p.virtio.remove(0));
        let p = plan_one("virtio-net-pci,netdev=n0,mac=52:54:00:12:34:57").unwrap();
        assert_eq!(p.netdev.as_deref(), Some("n0"));
        assert_eq!(p.mac, Some([0x52, 0x54, 0, 0x12, 0x34, 0x57]));
        assert_eq!(plan_one("virtio-net-device").unwrap().transport, Transport::Mmio);
        let e = plan_one("virtio-net-pci,mac=52:54:00").unwrap_err();
        assert_eq!(e[0].1.message(), "Property 'virtio-net-pci.mac' doesn't take value '52:54:00'");

        // The NIC takes its netdev, and a second one on the same netdev is refused.
        let netdevs = [(NetOption::Netdev, "hubport,id=n0,hubid=0".to_string(), None)];
        let net = Network::new(&netdevs, false, &Clock::manual(ClockType::Virtual), None).unwrap();
        let p = plan(V, &[], &[dev("virtio-net-pci,netdev=n0")]).unwrap();
        let board = VirtMachine::new(ruvm_machine_arm::virt::VirtConfig::default()).unwrap();
        plug(&BlockGraph::new(), Target::Virt(&board), &p.virtio, &p.display, &[], &net).unwrap();
        let e = plug(&BlockGraph::new(), Target::Virt(&board), &p.virtio, &p.display, &[], &net)
            .err()
            .unwrap();
        assert_eq!(
            e.1.message(),
            "Property 'virtio-net-pci.netdev' can't take value 'n0', it's in use"
        );
    }

    #[test]
    fn blk_pci_queues_follow_smp() {
        // One queue per vCPU and one MSI-X vector more, as virtio_blk_pci_realize() picks.
        let path = std::env::temp_dir().join(format!("ruvm-arm-blk-{}.img", std::process::id()));
        std::fs::write(&path, vec![0u8; 1 << 16]).unwrap();
        let arg = format!("file={},format=raw", path.display());
        let d = parse_drives(V, &[dev(&arg)]).unwrap();
        let graph = BlockGraph::new();
        crate::drives::open_drives(&graph, &d, &[dev(&arg)]).unwrap();
        let p = plan(V, &d, &[dev("virtio-rng-pci")]).unwrap();
        let cfg = ruvm_machine_arm::virt::VirtConfig {
            smp: 2,
            ..ruvm_machine_arm::virt::VirtConfig::default()
        };
        let board = VirtMachine::new(cfg).unwrap();
        plug(&graph, Target::Virt(&board), &p.virtio, &p.display, &d, &no_net()).unwrap();
        let _ = std::fs::remove_file(&path);
        let vectors: Vec<u32> = board.pci_devices().iter().map(|f| f.nvectors()).collect();
        assert_eq!(vectors, [2, 3]);
    }

    #[test]
    fn pflash_drives() {
        let d = parse_drives(
            V,
            &[
                dev("if=pflash,unit=0,format=raw,file=code.fd,readonly=on"),
                dev("if=pflash,unit=1,format=raw,file=vars.fd"),
            ],
        )
        .unwrap();
        let p = plan(V, &d, &[]).unwrap();
        assert_eq!(p.pflash, [Some(0), Some(1)]);
        let p = plan(V, &[], &[]).unwrap();
        assert_eq!(p.pflash, [None, None]);
    }

    #[test]
    fn orphaned_drives() {
        let d = parse_drives(V, &[dev("file=a.img,if=ide")]).unwrap();
        let e = plan(V, &d, &[]).unwrap_err();
        assert_eq!(e[0].1.message(), "machine type does not support if=ide,bus=0,unit=0");
        let d = parse_drives(V, &[dev("file=a.img,if=pflash,unit=2,format=raw")]).unwrap();
        let e = plan(V, &d, &[]).unwrap_err();
        assert_eq!(e[0].1.message(), "machine type does not support if=pflash,bus=0,unit=2");
        let d = parse_drives(V, &[dev("file=a.img,if=none,id=x")]).unwrap();
        let devs = [dev("virtio-blk,drive=x"), dev("virtio-blk-device,drive=x")];
        assert_eq!(
            plan(V, &d, &devs).unwrap_err()[0].1.message(),
            "Drive 'x' is already in use by another device"
        );
    }

    #[test]
    fn sbsa_ref_drives() {
        // IDE is the default, and index n is AHCI port n.
        let d = parse_drives(
            S,
            &[
                dev("file=a.img"),
                dev("if=ide,media=cdrom"),
                dev("file=c.img,index=5"),
                dev("file=v.img,if=virtio"),
            ],
        )
        .unwrap();
        assert_eq!((d[0].iface, d[0].bus, d[0].unit), (DriveIf::Ide, 0, 0));
        assert_eq!(d[0].id, "ide0-hd0");
        assert_eq!((d[2].bus, d[2].unit), (2, 1));
        let p = plan(S, &d, &[]).unwrap();
        assert_eq!(p.ide, [(0, 0), (1, 1), (5, 2)]);
        assert_eq!(p.virtio.len(), 1);
        assert_eq!(p.virtio[0].typename, "virtio-blk-pci");
        let d = parse_drives(S, &[dev("file=a.img,bus=3,unit=0")]).unwrap();
        assert_eq!(
            plan(S, &d, &[]).unwrap_err()[0].1.message(),
            "machine type does not support if=ide,bus=3,unit=0"
        );
        // No virtio-mmio transports.
        let err = |arg: &str| plan(S, &[], &[dev(arg)]).unwrap_err()[0].1.message().to_string();
        assert_eq!(
            err("virtio-rng-device"),
            "No 'virtio-bus' bus found for device 'virtio-rng-device'"
        );
        assert_eq!(
            err("virtio-rng-device,bus=virtio-mmio-bus.0"),
            "Bus 'virtio-mmio-bus.0' not found"
        );
        assert_eq!(plan(S, &[], &[dev("virtio-rng")]).unwrap().virtio.len(), 1);
        let e = plan(S, &[], &[dev("VGA")]).unwrap_err();
        let msg = "-device VGA is not supported with machine sbsa-ref by ruvm yet";
        assert_eq!(e[0].1.message(), msg);
        let p = plan(V, &[], &[dev("virtio-rng-pci"), dev("ramfb"), dev("VGA")]).unwrap();
        let at: Vec<usize> = p.display.iter().map(|(at, _)| *at).collect();
        assert_eq!((p.virtio.len(), at), (1, vec![1, 1]));
    }

    #[test]
    fn sbsa_ref_ahci_plug() {
        let board = SbsaRefMachine::new(Default::default()).unwrap();
        let d = parse_drives(S, &[dev("if=ide")]).unwrap();
        let p = plan(S, &d, &[]).unwrap();
        let e = plug_ahci(&BlockGraph::new(), &board, &p.ide, &d).map(drop).unwrap_err();
        assert_eq!(e.1.message(), "Device needs media, but drive is empty");
        let d = parse_drives(S, &[dev("if=ide,media=cdrom")]).unwrap();
        let p = plan(S, &d, &[]).unwrap();
        plug_ahci(&BlockGraph::new(), &board, &p.ide, &d).unwrap();
    }
}
