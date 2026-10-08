// SPDX-License-Identifier: GPL-2.0-or-later

//! `-drive` and `-device` for the Arm `virt` board, the part of blockdev.c's `drive_new()` and
//! qdev-monitor.c's `qdev_device_add()` that applies to it.
//!
//! A `-drive` without `if=` is a virtio disk, since virt's `block_default_type` is
//! `IF_VIRTIO`, and every `if=virtio` drive adds a `-device virtio-blk,drive=<id>` after the
//! others. The short virtio names are aliases of the PCI types, as in `qdev_alias_table[]`.
//!
//! `-device` takes `virtio-blk`, `virtio-rng` and `virtio-serial`, each as a PCI function on
//! the root bus `pcie.0` (`virtio-*-pci`) or on a virtio-mmio transport (`virtio-*-device`).
//! The other virtio types and `loader` fail with "... is not supported with this machine by
//! ruvm yet". The PCI types take `addr`, `bus=pcie.0`, `disable-legacy`,
//! `disable-modern` and `vectors`; the virtio-mmio ones take `bus=virtio-mmio-bus.<n>`.
//! `-drive if=pflash` with unit 0 or 1 on bus 0 is the drive of the flash of that number, as
//! `pflash_cfi01_legacy_drive()` takes it.

use std::collections::HashSet;

use ruvm_base::Error;
use ruvm_base::report::Location;
use ruvm_hw_virtio::mmio::VIRTIO_MMIO_FORCE_LEGACY_DEFAULT;
use ruvm_hw_virtio::pci::virtio_pci_optimal_num_queues;
use ruvm_hw_virtio::{
    RandomFile, VirtioBlk, VirtioBlkConf, VirtioConsole, VirtioDeviceClass, VirtioPciProps,
    VirtioRng, VirtioRngConf,
};
use ruvm_machine_arm::pflash::PflashBacking;
use ruvm_machine_arm::virt::VirtMachine;
use ruvm_machine_x86::FileBackend;
use ruvm_qapi::opts::{QemuOpts, QemuOptsList, is_help_option};

use super::prop_bool;
use crate::x86::{Drive, DriveIf, Located, Transport, drive_opts, parse_drive, probe_warning};

/// The virtio device models virt can plug.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Model {
    Blk,
    Rng,
    Serial,
}

/// The device types, with the bus they plug into.
const DEVICE_TYPES: &[(&str, Model, Transport)] = &[
    ("virtio-blk-device", Model::Blk, Transport::Mmio),
    ("virtio-blk-pci", Model::Blk, Transport::Pci),
    ("virtio-rng-device", Model::Rng, Transport::Mmio),
    ("virtio-rng-pci", Model::Rng, Transport::Pci),
    ("virtio-serial-device", Model::Serial, Transport::Mmio),
    ("virtio-serial-pci", Model::Serial, Transport::Pci),
];

/// `qdev_alias_table[]` for Arm, where the default virtio transport is PCI.
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
    props: VirtioPciProps,
    loc: Option<Location>,
}

/// What virt gets from `-drive` and `-device`, in the order QEMU plugs it.
#[derive(Debug, Default)]
pub(crate) struct Plan {
    /// The virtio devices, `-device` first and then the ones `-drive if=virtio` adds.
    pub virtio: Vec<Plug>,
    /// The drives of the two flashes, `None` where there is no `-drive if=pflash`.
    pub pflash: [Option<PflashBacking>; 2],
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
        props: VirtioPciProps::default(),
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
    for (arg, loc) in &queue {
        let mut list = QemuOptsList::new("device", &[]).with_implied_opt_name("driver");
        let opts = list.parse(arg, true).map_err(|e| vec![Located(loc.clone(), e)])?;
        let Some(driver) = opts.get("driver") else {
            return Err(vec![Located::new(loc, "Parameter 'driver' is missing")]);
        };
        if is_help_option(driver) || opts.has_help_opt() {
            return Err(vec![Located::new(loc, "-device help is not supported by ruvm yet")]);
        }
        if driver == "loader" {
            let msg = "-device loader is not supported with this machine by ruvm yet";
            return Err(vec![Located::new(loc, msg)]);
        }
        let plug = plan_virtio(drives, &mut used, driver, opts, loc).map_err(|e| vec![e])?;
        p.virtio.push(plug);
    }

    // pflash_cfi01_legacy_drive() for the two flashes. The flash reads the image itself and
    // checks its size; a drive without a medium leaves the flash empty.
    for (i, d) in drives.iter().enumerate() {
        if d.iface != DriveIf::Pflash || d.bus != 0 || d.unit >= 2 {
            continue;
        }
        used.insert(i);
        let backing = match &d.file {
            Some(file) => {
                if d.probed && !d.read_only {
                    eprint!("{}", probe_warning(file));
                }
                PflashBacking::File { path: file.into(), read_only: d.read_only }
            }
            None => PflashBacking::None,
        };
        p.pflash[d.unit as usize] = Some(backing);
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

/// Opens drive `d` for a device.
fn open_drive(d: &Drive) -> Result<FileBackend, String> {
    let file = d.file.as_deref().expect("planned with media");
    let backend = FileBackend::open(file, d.read_only)?;
    if d.probed && !d.read_only {
        eprint!("{}", probe_warning(file));
    }
    Ok(backend)
}

/// `qdev_device_add()` for the planned virtio devices: builds each one and plugs it into its
/// bus.
pub(crate) fn plug(board: &VirtMachine, plugs: &[Plug], drives: &[Drive]) -> Result<(), Located> {
    for plug in plugs {
        let at = |e: String| Located::new(&plug.loc, e);
        let class: Box<dyn VirtioDeviceClass> = match plug.model {
            Model::Blk => {
                let d = &drives[plug.drive.expect("planned with a drive")];
                let backend = open_drive(d).map_err(|e| Located::new(&d.loc, e))?;
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
        };
        match (plug.transport, plug.mmio_bus) {
            (Transport::Pci, _) => {
                board.attach_virtio_pci(class, plug.devfn, &plug.props).map_err(at)?;
            }
            (Transport::Mmio, Some(n)) => {
                board.attach_virtio_at(n, class, VIRTIO_MMIO_FORCE_LEGACY_DEFAULT).map_err(at)?;
            }
            (Transport::Mmio, None) => {
                board.attach_virtio(class).map_err(|e| at(format!("{e} '{}'", plug.typename)))?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let p = plan(&[], &[dev("virtio-rng")]).unwrap();
        assert_eq!(p.virtio[0].typename, "virtio-rng-pci");
        assert_eq!(p.virtio[0].transport, Transport::Pci);
        let p = plan(&[], &[dev("virtio-rng-device,bus=virtio-mmio-bus.2")]).unwrap();
        assert_eq!(p.virtio[0].mmio_bus, Some(2));
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
            err(&["virtio-net-device"]),
            "-device virtio-net-device is not supported with this machine by ruvm yet"
        );
        assert_eq!(err(&["e1000"]), "'e1000' is not a valid device model name");
        assert_eq!(
            err(&["loader,addr=0x40000000,data=1,data-len=4"]),
            "-device loader is not supported with this machine by ruvm yet"
        );
    }

    #[test]
    fn blk_pci_queues_follow_smp() {
        // One queue per vCPU and one MSI-X vector more, as virtio_blk_pci_realize() picks.
        let path = std::env::temp_dir().join(format!("ruvm-arm-blk-{}.img", std::process::id()));
        std::fs::write(&path, vec![0u8; 1 << 16]).unwrap();
        let arg = format!("file={},format=raw", path.display());
        let d = parse_drives(&[dev(&arg)]).unwrap();
        let p = plan(&d, &[dev("virtio-rng-pci")]).unwrap();
        let cfg = ruvm_machine_arm::virt::VirtConfig {
            smp: 2,
            ..ruvm_machine_arm::virt::VirtConfig::default()
        };
        let board = VirtMachine::new(cfg).unwrap();
        plug(&board, &p.virtio, &d).unwrap();
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
        assert_eq!(
            p.pflash,
            [
                Some(PflashBacking::File { path: "code.fd".into(), read_only: true }),
                Some(PflashBacking::File { path: "vars.fd".into(), read_only: false }),
            ]
        );
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
