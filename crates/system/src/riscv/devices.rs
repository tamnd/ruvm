// SPDX-License-Identifier: GPL-2.0-or-later

//! `-drive` and `-device` for the RISC-V `virt` board, the part of blockdev.c's `drive_new()`
//! and qdev-monitor.c's `qdev_device_add()` that applies to it.
//!
//! A `-drive` without `if=` is a virtio disk, since virt's `block_default_type` is
//! `IF_VIRTIO`, and every `if=virtio` drive adds a `-device virtio-blk,drive=<id>` after the
//! others. The short virtio names are aliases of the PCI types, as in `qdev_alias_table[]`.
//!
//! `-device` takes `loader`, `riscv-iommu-pci`, `virtio-blk`, `virtio-rng` and
//! `virtio-serial`, the virtio ones as a PCI function on the root bus `pcie.0`
//! (`virtio-*-pci`) or on a virtio-mmio transport (`virtio-*-device`). The other virtio
//! types fail with "... is not supported with this machine by ruvm yet". The PCI types take
//! `addr`, `bus=pcie.0`, `disable-legacy`, `disable-modern`, `iommu_platform` and `vectors`;
//! the virtio-mmio ones take `bus=virtio-mmio-bus.<n>`.
//! `-drive if=pflash` with unit 0 or 1 on bus 0 is the drive of the flash of that number, as
//! `pflash_cfi01_legacy_drive()` takes it.

use std::collections::HashSet;

use ruvm_base::Error;
use ruvm_base::report::Location;
use ruvm_block::BlockGraph;
use ruvm_hw_virtio::mmio::VIRTIO_MMIO_FORCE_LEGACY_DEFAULT;
use ruvm_hw_virtio::pci::virtio_pci_optimal_num_queues;
use ruvm_hw_virtio::{
    RandomFile, VirtioBalloon, VirtioBlk, VirtioBlkConf, VirtioConsole, VirtioDeviceClass,
    VirtioNet, VirtioNetConf, VirtioPciProps, VirtioRng, VirtioRngConf,
};
use ruvm_machine_riscv::virt::{
    GenericLoader, RiscvIommuPciProps, TYPE_RISCV_IOMMU_PCI, VirtMachine,
};
use ruvm_machine_x86::VirtioHandle;
use ruvm_qapi::opts::{QemuOpts, QemuOptsList, is_help_option};
use ruvm_qapi::visit::{Visitor, VisitorExt};

use super::{parse_loader, prop_bool};
use crate::drives::{DriveDev, Plugged, device_image, peripheral_path, resize_cb};
use crate::net::Network;
use crate::x86::{Drive, DriveIf, KeepRam, Located, Transport, drive_opts, parse_drive, parse_mac};

/// The virtio device models virt can plug.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Model {
    Blk,
    Rng,
    Serial,
    Balloon,
    Net,
    /// `riscv-iommu-pci`, which is not a virtio device but plugs in the same order.
    RiscvIommu,
}

/// The device types, with the bus they plug into.
const DEVICE_TYPES: &[(&str, Model, Transport)] = &[
    ("virtio-blk-device", Model::Blk, Transport::Mmio),
    ("virtio-blk-pci", Model::Blk, Transport::Pci),
    ("virtio-rng-device", Model::Rng, Transport::Mmio),
    ("virtio-rng-pci", Model::Rng, Transport::Pci),
    ("virtio-serial-device", Model::Serial, Transport::Mmio),
    ("virtio-serial-pci", Model::Serial, Transport::Pci),
    ("virtio-balloon-device", Model::Balloon, Transport::Mmio),
    ("virtio-balloon-pci", Model::Balloon, Transport::Pci),
    ("virtio-net-device", Model::Net, Transport::Mmio),
    ("virtio-net-pci", Model::Net, Transport::Pci),
];

/// `qdev_alias_table[]` for RISC-V, where the default virtio transport is PCI.
const DEVICE_ALIASES: &[(&str, &str)] = &[
    ("virtio-blk", "virtio-blk-pci"),
    ("virtio-rng", "virtio-rng-pci"),
    ("virtio-serial", "virtio-serial-pci"),
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
    /// virtio-net `netdev`.
    netdev: Option<String>,
    /// virtio-net `mac`.
    mac: Option<[u8; 6]>,
    props: VirtioPciProps,
    /// `iommu_platform` of a PCI function: its DMA goes through the IOMMU.
    iommu_platform: bool,
    /// The properties of a `riscv-iommu-pci`.
    iommu: Box<RiscvIommuPciProps>,
    /// Where `qdev_set_id()` put the device in the QOM tree.
    path: String,
    loc: Option<Location>,
}

/// What virt gets from `-drive` and `-device`, in the order QEMU plugs it.
#[derive(Debug, Default)]
pub(crate) struct Plan {
    pub loaders: Vec<GenericLoader>,
    /// The virtio devices, `-device` first and then the ones `-drive if=virtio` adds.
    pub virtio: Vec<Plug>,
    /// The drives of the two flashes as indices into the drives, `None` where there is no
    /// `-drive if=pflash`.
    pub pflash: [Option<usize>; 2],
}

/// `drive_new()` for each `-drive`, with virt's default interface.
pub(crate) fn parse_drives(args: &[(String, Option<Location>)]) -> Result<Vec<Drive>, Located> {
    let mut list = drive_opts();
    let mut drives = Vec::new();
    for (arg, loc) in args {
        let fail = |e: Error| Located(loc.clone(), e);
        let has_if = drive_opts().parse(arg, false).map_err(fail)?.get("if").is_some();
        let arg = if has_if { arg.clone() } else { format!("{arg},if=virtio") };
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

/// One `-device` that is not `loader`.
fn plan_virtio(
    drives: &[Drive],
    used: &mut HashSet<usize>,
    driver: &str,
    opts: &QemuOpts,
    loc: &Option<Location>,
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
        netdev: None,
        mac: None,
        props: VirtioPciProps::default(),
        iommu_platform: false,
        iommu: Box::default(),
        path: String::new(),
        loc: loc.clone(),
    };
    let pci = transport == Transport::Pci;
    for (k, v) in opts.iter() {
        match k {
            "driver" => {}
            "id" => plug.props.id = Some(v.to_string()),
            "bus" => {
                let found = if pci {
                    v == PCIE_ROOT_BUS
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
            "vectors" if pci => {
                plug.props.vectors = Some(v.parse().map_err(|_| bad_value(k, v))?);
            }
            "iommu_platform" if pci => {
                plug.iommu_platform = prop_bool(k, v).map_err(|_| bad_value(k, v))?;
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

/// A `riscv-iommu-pci` with its own properties and those of `riscv-iommu`. The `bus`
/// property of `riscv-iommu` cannot be given here, since `qdev_device_add()` takes `bus` as
/// the bus to plug into.
fn plan_riscv_iommu(opts: &QemuOpts, loc: &Option<Location>) -> Result<Plug, Located> {
    let at = |e: Error| Located(loc.clone(), e);
    let boolean =
        |k: &str, v: &str| crate::display::prop(k, v, |vi, n, out| vi.type_bool(Some(n), out));
    let u16_prop =
        |k: &str, v: &str| crate::display::prop(k, v, |vi, n, out| vi.type_uint16(Some(n), out));
    let u8_prop =
        |k: &str, v: &str| crate::display::prop(k, v, |vi, n, out| vi.type_uint8(Some(n), out));
    let u32_prop =
        |k: &str, v: &str| crate::display::prop(k, v, |vi, n, out| vi.type_uint32(Some(n), out));
    let mut plug = Plug {
        typename: TYPE_RISCV_IOMMU_PCI,
        model: Model::RiscvIommu,
        transport: Transport::Pci,
        drive: None,
        serial: None,
        devfn: None,
        mmio_bus: None,
        netdev: None,
        mac: None,
        props: VirtioPciProps::default(),
        iommu_platform: false,
        iommu: Box::default(),
        path: String::new(),
        loc: loc.clone(),
    };
    let p = &mut plug.iommu;
    p.id = opts.id().map(str::to_string);
    for (k, v) in opts.iter() {
        match k {
            "driver" => {}
            "bus" if v == PCIE_ROOT_BUS => {}
            "bus" => return Err(Located::new(loc, format!("Bus '{v}' not found"))),
            "addr" => {
                plug.devfn = Some(parse_devfn(v).ok_or_else(|| {
                    Located::new(
                        loc,
                        format!("Property '{TYPE_RISCV_IOMMU_PCI}.{k}' doesn't take value '{v}'"),
                    )
                })?);
            }
            "vendor-id" => p.vendor_id = u16_prop(k, v).map_err(at)?,
            "device-id" => p.device_id = u16_prop(k, v).map_err(at)?,
            "revision" => p.revision = u8_prop(k, v).map_err(at)?,
            "version" => p.conf.version = u32_prop(k, v).map_err(at)?,
            "pas-bits" => p.conf.pas_bits = u32_prop(k, v).map_err(at)?,
            "ioatc-limit" => p.conf.ioatc_limit = u32_prop(k, v).map_err(at)?,
            "intremap" => p.conf.intremap = boolean(k, v).map_err(at)?,
            "ats" => p.conf.ats = boolean(k, v).map_err(at)?,
            "off" => p.conf.off = boolean(k, v).map_err(at)?,
            "s-stage" => p.conf.s_stage = boolean(k, v).map_err(at)?,
            "g-stage" => p.conf.g_stage = boolean(k, v).map_err(at)?,
            "hpm-counters" => p.conf.hpm_counters = u8_prop(k, v).map_err(at)?,
            _ => {
                return Err(Located::new(
                    loc,
                    format!("Property '{TYPE_RISCV_IOMMU_PCI}.{k}' not found"),
                ));
            }
        }
    }
    Ok(plug)
}

/// What one `-device` plugs.
enum Planned {
    Loader(GenericLoader),
    Virtio(Plug),
}

/// `qdev_device_add()` for one `-device`, with `-global` merged in.
fn plan_device(
    drives: &[Drive],
    used: &mut HashSet<usize>,
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
        return parse_loader(opts, loc).map(Planned::Loader);
    }
    if driver == TYPE_RISCV_IOMMU_PCI {
        return plan_riscv_iommu(opts, loc).map(Planned::Virtio);
    }
    plan_virtio(drives, used, driver, opts, loc).map(Planned::Virtio)
}

/// Works out what `-drive` and `-device` plug into virt. The errors are those of
/// `qdev_device_add()` and `drive_check_orphaned()`; for orphans every one is listed.
pub(crate) fn plan(
    drives: &[Drive],
    devices: &[(String, Option<Location>)],
) -> Result<Plan, Vec<Located>> {
    let mut p = Plan::default();
    let mut used = HashSet::new();
    let mut queue: Vec<(String, Option<Location>)> = devices.to_vec();
    // drive_new() adds a -device virtio-blk for every if=virtio drive, after the others.
    for d in drives.iter().filter(|d| d.iface == DriveIf::Virtio) {
        queue.push((format!("virtio-blk,drive={}", d.id), d.loc.clone()));
    }
    let mut anon = 0;
    for (arg, loc) in &queue {
        let path = peripheral_path(arg, &mut anon);
        let plan = |arg: &str, trial: bool| {
            if trial {
                plan_device(drives, &mut used.clone(), arg, loc)
            } else {
                plan_device(drives, &mut used, arg, loc)
            }
        };
        let planned = crate::global::plan_device(arg, loc, plan, |_, e| Err(e));
        match planned.map_err(|e| vec![e])? {
            Planned::Loader(loader) => p.loaders.push(loader),
            Planned::Virtio(plug) => p.virtio.push(Plug { path, ..plug }),
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

/// `qdev_device_add()` for the planned virtio devices: builds each one and plugs it into its
/// bus, a virtio-net joined to its netdev in `net`. Gives the drives it put on virtio-blk
/// devices.
pub(crate) fn plug(
    graph: &BlockGraph,
    board: &VirtMachine,
    plugs: &[Plug],
    drives: &[Drive],
    net: &Network,
) -> Result<Vec<Plugged>, Located> {
    let mut plugged = Vec::new();
    for plug in plugs {
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
            Model::Balloon => Box::new(VirtioBalloon::new(board.ram_size(), Box::new(KeepRam))),
            Model::RiscvIommu => {
                board.attach_riscv_iommu_pci(&plug.iommu, plug.devfn).map_err(at)?;
                continue;
            }
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
        let handle = match (plug.transport, plug.mmio_bus) {
            (Transport::Pci, _) => {
                let dev = board
                    .attach_virtio_pci_with(class, plug.devfn, &plug.props, plug.iommu_platform)
                    .map_err(at)?;
                Some(VirtioHandle::Pci(dev))
            }
            (Transport::Mmio, Some(n)) => {
                board.attach_virtio_at(n, class, VIRTIO_MMIO_FORCE_LEGACY_DEFAULT).map_err(at)?;
                board.virtio_transport(n).map(VirtioHandle::Mmio)
            }
            (Transport::Mmio, None) => {
                let n = board
                    .attach_virtio(class)
                    .map_err(|e| at(format!("{e} '{}'", plug.typename)))?;
                board.virtio_transport(n).map(VirtioHandle::Mmio)
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
    Ok(plugged)
}

#[cfg(test)]
mod tests {
    use ruvm_base::ClockType;
    use ruvm_hw_core::Clock;

    use super::*;
    use crate::net::NetOption;

    fn dev(arg: &str) -> (String, Option<Location>) {
        (arg.to_string(), None)
    }

    #[test]
    fn drives_default_to_virtio() {
        let d = parse_drives(&[dev("file=a.img"), dev("file=b.img,if=none,id=d0")]).unwrap();
        assert_eq!(d[0].iface, DriveIf::Virtio);
        assert_eq!(d[0].id, "virtio0");
        assert_eq!(d[1].iface, DriveIf::None);
        let p = plan(&d, &[dev("virtio-blk-pci,drive=d0,addr=3")]).unwrap();
        assert_eq!(p.virtio.len(), 2);
        assert_eq!(p.virtio[0].devfn, Some(3 << 3));
        assert_eq!(p.virtio[0].drive, Some(1));
        assert_eq!(p.virtio[1].typename, "virtio-blk-pci");
        assert_eq!(p.virtio[1].drive, Some(0));
    }

    #[test]
    fn devices() {
        let p = plan(&[], &[dev("loader,addr=0x80000000,data=1,data-len=4"), dev("virtio-rng")])
            .unwrap();
        assert_eq!(p.loaders.len(), 1);
        assert_eq!(p.loaders[0].addr, 0x8000_0000);
        assert_eq!(p.loaders[0].data, 1);
        assert_eq!(p.loaders[0].data_len, 4);
        assert_eq!(p.virtio[0].typename, "virtio-rng-pci");
        assert_eq!(p.virtio[0].transport, Transport::Pci);
        let p = plan(&[], &[dev("virtio-rng-device,bus=virtio-mmio-bus.2")]).unwrap();
        assert_eq!(p.virtio[0].mmio_bus, Some(2));
        let p = plan(&[], &[dev("virtio-balloon"), dev("virtio-balloon-device")]).unwrap();
        assert_eq!(p.virtio[0].typename, "virtio-balloon-pci");
        assert_eq!(p.virtio[1].transport, Transport::Mmio);
        let p =
            plan(&[], &[dev("virtio-serial-pci,disable-legacy=on,vectors=0,addr=1f.7")]).unwrap();
        assert_eq!(p.virtio[0].props.disable_legacy, Some(true));
        assert_eq!(p.virtio[0].props.vectors, Some(0));
        assert_eq!(p.virtio[0].devfn, Some(0xff));

        let err = |args: &[&str]| {
            let devs: Vec<_> = args.iter().map(|a| dev(a)).collect();
            plan(&[], &devs).unwrap_err()[0].1.message().to_string()
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
        assert_eq!(err(&["loader,foo=1"]), "Property 'loader.foo' not found");
    }

    #[test]
    fn riscv_iommu_pci() {
        let d = parse_drives(&[dev("file=a.img,if=none,id=d0")]).unwrap();
        let p = plan(
            &d,
            &[
                dev("riscv-iommu-pci,addr=1,pas-bits=48,s-stage=off,hpm-counters=4,id=io0"),
                dev("virtio-blk-pci,drive=d0,iommu_platform=on"),
            ],
        )
        .unwrap();
        assert_eq!(p.virtio[0].model, Model::RiscvIommu);
        assert_eq!(p.virtio[0].devfn, Some(1 << 3));
        assert_eq!(p.virtio[0].iommu.id.as_deref(), Some("io0"));
        assert_eq!(p.virtio[0].iommu.device_id, 0x0014);
        let conf = &p.virtio[0].iommu.conf;
        assert_eq!((conf.pas_bits, conf.s_stage, conf.hpm_counters), (48, false, 4));
        assert!(p.virtio[1].iommu_platform);

        let err = |a: &str| plan(&[], &[dev(a)]).unwrap_err()[0].1.message().to_string();
        assert_eq!(err("riscv-iommu-pci,foo=1"), "Property 'riscv-iommu-pci.foo' not found");
        assert_eq!(err("riscv-iommu-pci,bus=pci.0"), "Bus 'pci.0' not found");
        assert_eq!(
            err("virtio-rng-device,iommu_platform=on"),
            "Property 'virtio-rng-device.iommu_platform' not found"
        );

        let board = VirtMachine::new(ruvm_machine_riscv::virt::VirtConfig::default()).unwrap();
        let rng = dev("virtio-rng-pci,iommu_platform=on,disable-legacy=on");
        let p = plan(&[], &[dev("riscv-iommu-pci"), rng]).unwrap();
        plug(&BlockGraph::new(), &board, &p.virtio, &[], &no_net()).unwrap();
        assert_eq!(board.riscv_iommus().len(), 1);
        // A transitional device cannot negotiate VIRTIO_F_IOMMU_PLATFORM.
        let p = plan(&[], &[dev("virtio-rng-pci,iommu_platform=on")]).unwrap();
        let e = plug(&BlockGraph::new(), &board, &p.virtio, &[], &no_net()).err().unwrap();
        assert_eq!(
            e.1.message(),
            "VIRTIO_F_IOMMU_PLATFORM was supported by neither legacy nor transitional device"
        );
    }

    fn no_net() -> Network {
        Network::new(&[], false, &Clock::manual(ClockType::Virtual), None).unwrap()
    }

    #[test]
    fn virtio_net() {
        let plan_one = |a: &str| plan(&[], &[dev(a)]).map(|mut p| p.virtio.remove(0));
        let p = plan_one("virtio-net-pci,netdev=n0,mac=52:54:00:12:34:57").unwrap();
        assert_eq!(p.netdev.as_deref(), Some("n0"));
        assert_eq!(p.mac, Some([0x52, 0x54, 0, 0x12, 0x34, 0x57]));
        assert_eq!(plan_one("virtio-net-device").unwrap().transport, Transport::Mmio);
        let e = plan_one("virtio-net-pci,mac=52:54:00").unwrap_err();
        assert_eq!(e[0].1.message(), "Property 'virtio-net-pci.mac' doesn't take value '52:54:00'");

        // The NIC takes its netdev, and a second one on the same netdev is refused.
        let netdevs = [(NetOption::Netdev, "hubport,id=n0,hubid=0".to_string(), None)];
        let net = Network::new(&netdevs, false, &Clock::manual(ClockType::Virtual), None).unwrap();
        let p = plan(&[], &[dev("virtio-net-pci,netdev=n0")]).unwrap();
        let board = VirtMachine::new(ruvm_machine_riscv::virt::VirtConfig::default()).unwrap();
        plug(&BlockGraph::new(), &board, &p.virtio, &[], &net).unwrap();
        let e = plug(&BlockGraph::new(), &board, &p.virtio, &[], &net).err().unwrap();
        assert_eq!(
            e.1.message(),
            "Property 'virtio-net-pci.netdev' can't take value 'n0', it's in use"
        );
    }

    #[test]
    fn blk_pci_queues_follow_smp() {
        // One queue per vCPU and one MSI-X vector more, as virtio_blk_pci_realize() picks.
        let path = std::env::temp_dir().join(format!("ruvm-rv-blk-{}.img", std::process::id()));
        std::fs::write(&path, vec![0u8; 1 << 16]).unwrap();
        let arg = format!("file={},format=raw", path.display());
        let d = parse_drives(&[dev(&arg)]).unwrap();
        let graph = BlockGraph::new();
        crate::drives::open_drives(&graph, &d, &[dev(&arg)]).unwrap();
        let p = plan(&d, &[dev("virtio-rng-pci")]).unwrap();
        let cfg = ruvm_machine_riscv::virt::VirtConfig {
            smp: 2,
            ..ruvm_machine_riscv::virt::VirtConfig::default()
        };
        let board = VirtMachine::new(cfg).unwrap();
        plug(&graph, &board, &p.virtio, &d, &no_net()).unwrap();
        let _ = std::fs::remove_file(&path);
        let vectors: Vec<u32> = board.pci_devices().iter().map(|f| f.nvectors()).collect();
        assert_eq!(vectors, [2, 3]);
    }

    #[test]
    fn pflash_drives() {
        let d = parse_drives(&[
            dev("if=pflash,unit=0,format=raw,file=code.fd,readonly=on"),
            dev("if=pflash,unit=1,format=raw,file=vars.fd"),
        ])
        .unwrap();
        let p = plan(&d, &[]).unwrap();
        assert_eq!(p.pflash, [Some(0), Some(1)]);
        let p = plan(&[], &[]).unwrap();
        assert_eq!(p.pflash, [None, None]);
    }

    #[test]
    fn orphaned_drives() {
        let d = parse_drives(&[dev("file=a.img,if=ide")]).unwrap();
        let e = plan(&d, &[]).unwrap_err();
        assert_eq!(e[0].1.message(), "machine type does not support if=ide,bus=0,unit=0");
        let d = parse_drives(&[dev("file=a.img,if=pflash,unit=2,format=raw")]).unwrap();
        let e = plan(&d, &[]).unwrap_err();
        assert_eq!(e[0].1.message(), "machine type does not support if=pflash,bus=0,unit=2");
        let d = parse_drives(&[dev("file=a.img,if=none,id=x")]).unwrap();
        let devs = [dev("virtio-blk,drive=x"), dev("virtio-blk-device,drive=x")];
        assert_eq!(
            plan(&d, &devs).unwrap_err()[0].1.message(),
            "Drive 'x' is already in use by another device"
        );
    }
}
