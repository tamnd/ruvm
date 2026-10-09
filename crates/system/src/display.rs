// SPDX-License-Identifier: GPL-2.0-or-later

//! The display side of the command line and the monitor: `-vga` (`select_vgahw()` of QEMU's
//! system/vl.c), `pc_vga_init()` of hw/i386/pc.c, `-device VGA`, `bochs-display` and `ramfb`
//! on q35 and Arm virt, and the QMP `screendump` command of ui/ui-qmp-cmds.c.
//!
//! Where this differs from QEMU:
//! - q35 gets a VGA card only with `-vga std` or `-device VGA`. QEMU plugs the std VGA by
//!   default.
//! - `-vga` knows `std` and `none`. The other types fail with "... not available", which is
//!   what a QEMU built without them says. `retrace=` is checked and has no effect.
//! - On Arm virt the display functions get no option ROM, so `romfile` has no effect there.
//! - The HMP `screendump` command is not here.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use ruvm_base::report::{Location, warn_report};
use ruvm_base::{Error, Result};
use ruvm_hw_core::fw_cfg::{DmaMemory, FwCfgState};
use ruvm_hw_display::bochs_display::{BOCHS_DISPLAY_ROMFILE, BochsDisplay, BochsDisplayProps};
use ruvm_hw_display::edid::EdidInfo;
use ruvm_hw_display::ramfb::Ramfb;
use ruvm_hw_display::vga_pci::{VGA_ROMFILE, VgaPci, VgaPciProps};
use ruvm_hw_pci::regs::PCI_ROM_SLOT;
use ruvm_hw_pci::{PciBus, PciDevice};
use ruvm_machine_arm::virt::VirtMachine;
use ruvm_machine_x86::{FirmwareSearch, X86Board};
use ruvm_mem::AddressSpace;
use ruvm_monitor::{Commands, MonitorQmp};
use ruvm_qapi::commands::register_screendump;
use ruvm_qapi::opts::QemuOpts;
use ruvm_qapi::types::ImageFormat;
use ruvm_qapi::visit::{QObjectInputVisitor, Visitor, VisitorExt};
use ruvm_qapi::{QDict, QValue};
use ruvm_ui::console::DisplayState;
use ruvm_ui::screendump;

use crate::x86::Located;

/// `vga_interfaces[]`: the `-vga` name, the description, and whether ruvm has the device.
const VGA_INTERFACES: &[(&str, &str, bool)] = &[
    ("none", "no graphic card", true),
    ("std", "standard VGA", true),
    ("cirrus", "Cirrus VGA", false),
    ("vmware", "VMWare SVGA", false),
    ("virtio", "Virtio VGA", false),
    ("qxl", "QXL VGA", false),
    ("tcx", "TCX framebuffer", false),
    ("cg3", "CG3 framebuffer", false),
];

/// `vga_interface_created`: a board plugged the card `-vga` asked for.
static VGA_INTERFACE_CREATED: AtomicBool = AtomicBool::new(false);

/// What `-vga` selected, `vga_interface_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VgaInterface {
    None,
    Std,
}

/// `select_vgahw()` for the last `-vga`. `Ok(None)` is `help`, after the list is printed. The
/// default is `std` on every machine, as QEMU picks it when there is no Cirrus VGA.
pub(crate) fn select_vgahw(p: &str) -> std::result::Result<Option<VgaInterface>, String> {
    if p == "help" {
        for (name, desc, available) in VGA_INTERFACES {
            if *available {
                let default = if *name == "std" { " (default)" } else { "" };
                println!("{name:<20} {desc}{default}");
            }
        }
        return Ok(None);
    }
    let unknown = || format!("unknown vga type: {p}");
    let Some((i, mut opts)) = VGA_INTERFACES
        .iter()
        .enumerate()
        .find_map(|(i, (name, ..))| p.strip_prefix(name).map(|rest| (i, rest)))
    else {
        return Err(unknown());
    };
    let (_, desc, available) = VGA_INTERFACES[i];
    if !available {
        return Err(format!("{desc} not available"));
    }
    while !opts.is_empty() {
        let rest = opts.strip_prefix(",retrace=").ok_or_else(unknown)?;
        opts = rest
            .strip_prefix("dumb")
            .or_else(|| rest.strip_prefix("precise"))
            .ok_or_else(unknown)?;
    }
    Ok(Some(if i == 0 { VgaInterface::None } else { VgaInterface::Std }))
}

/// The end of `qemu_create_cli_devices()`: warns when `-vga` asked for a card the machine
/// does not plug.
pub(crate) fn check_vga_created(vga: Option<VgaInterface>) {
    if vga == Some(VgaInterface::Std) && !VGA_INTERFACE_CREATED.load(Ordering::Acquire) {
        warn_report(
            "A -vga option was passed but this machine type does not use that option; No VGA device has been created",
        );
    }
}

/// The display types `-device` knows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DisplayModel {
    /// "VGA", the std VGA PCI card.
    Vga,
    /// "bochs-display".
    BochsDisplay,
    /// "ramfb".
    Ramfb,
}

const DISPLAY_TYPES: &[(&str, DisplayModel)] = &[
    ("VGA", DisplayModel::Vga),
    ("bochs-display", DisplayModel::BochsDisplay),
    ("ramfb", DisplayModel::Ramfb),
];

/// A display device to plug, with its properties.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DisplayPlug {
    pub model: DisplayModel,
    pub id: Option<String>,
    /// `addr`.
    devfn: Option<u8>,
    /// `romfile`, `None` for an empty one.
    romfile: Option<String>,
    /// VGA `vgamem_mb`.
    vgamem_mb: u32,
    /// bochs-display `vgamem`, in bytes.
    vgamem: u64,
    mmio: bool,
    qemu_extended_regs: bool,
    edid: bool,
    xres: u32,
    yres: u32,
    xmax: u32,
    ymax: u32,
    refresh_rate: u32,
    big_endian: bool,
    /// ramfb `use-legacy-x86-rom`, which the PC machines' compat properties turn on.
    legacy_rom: Option<bool>,
    pub loc: Option<Location>,
}

impl DisplayPlug {
    fn new(model: DisplayModel, loc: Option<Location>) -> DisplayPlug {
        let romfile = match model {
            DisplayModel::Vga => Some(VGA_ROMFILE.to_string()),
            DisplayModel::BochsDisplay => Some(BOCHS_DISPLAY_ROMFILE.to_string()),
            DisplayModel::Ramfb => None,
        };
        DisplayPlug {
            model,
            id: None,
            devfn: None,
            romfile,
            vgamem_mb: 16,
            vgamem: 16 << 20,
            mmio: true,
            qemu_extended_regs: true,
            edid: true,
            xres: 0,
            yres: 0,
            xmax: 0,
            ymax: 0,
            refresh_rate: 0,
            big_endian: false,
            legacy_rom: None,
            loc,
        }
    }

    pub(crate) fn typename(&self) -> &'static str {
        match self.model {
            DisplayModel::Vga => "VGA",
            DisplayModel::BochsDisplay => "bochs-display",
            DisplayModel::Ramfb => "ramfb",
        }
    }

    fn at(&self, e: impl Into<String>) -> Located {
        Located::new(&self.loc, e)
    }

    fn edid_info(&self) -> EdidInfo {
        EdidInfo {
            prefx: self.xres,
            prefy: self.yres,
            maxx: self.xmax,
            maxy: self.ymax,
            refresh_rate: self.refresh_rate,
            ..EdidInfo::default()
        }
    }
}

/// A qdev property from its command line string, read the way the keyval input visitor reads
/// it, so the errors are QEMU's.
fn prop<T: Default>(
    name: &str,
    value: &str,
    visit: impl FnOnce(&mut QObjectInputVisitor, &str, &mut T) -> Result<()>,
) -> Result<T> {
    let mut d = QDict::new();
    d.put(name, QValue::Str(value.to_string()));
    let mut v = QObjectInputVisitor::new_keyval(QValue::Dict(d));
    v.start_struct(None)?;
    let mut out = T::default();
    visit(&mut v, name, &mut out)?;
    v.end_struct();
    Ok(out)
}

fn prop_u32(name: &str, value: &str) -> Result<u32> {
    prop(name, value, |v, n, out| v.type_uint32(Some(n), out))
}

fn prop_bool(name: &str, value: &str) -> Result<bool> {
    prop(name, value, |v, n, out| v.type_bool(Some(n), out))
}

fn prop_size(name: &str, value: &str) -> Result<u64> {
    prop(name, value, |v, n, out| v.type_size(Some(n), out))
}

/// The `addr` property, `SLOT[.FN]` in hex.
fn parse_devfn(v: &str) -> Option<u8> {
    let (slot, func) = v.split_once('.').unwrap_or((v, "0"));
    let hex = |s: &str| {
        let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")).unwrap_or(s);
        if s.is_empty() { None } else { u32::from_str_radix(s, 16).ok() }
    };
    let (slot, func) = (hex(slot)?, hex(func)?);
    (slot <= 31 && func <= 7).then_some((slot << 3 | func) as u8)
}

/// `qdev_device_add()` up to realize for a display type: the bus, then the properties.
/// `None` when `driver` is not a display type. `pci` says whether the machine has the root
/// bus `pcie.0`, `sysbus` whether it takes a `ramfb`.
pub(crate) fn plan_device(
    driver: &str,
    opts: &QemuOpts,
    loc: &Option<Location>,
    pci: bool,
    sysbus: bool,
) -> Option<std::result::Result<DisplayPlug, Located>> {
    let &(typename, model) = DISPLAY_TYPES.iter().find(|(t, _)| *t == driver)?;
    Some(plan(typename, model, opts, loc, pci, sysbus))
}

fn plan(
    typename: &str,
    model: DisplayModel,
    opts: &QemuOpts,
    loc: &Option<Location>,
    pci: bool,
    sysbus: bool,
) -> std::result::Result<DisplayPlug, Located> {
    let is_pci = model != DisplayModel::Ramfb;
    if let Some(b) = opts.get("bus") {
        // The bus is looked up by name first, then its type is checked.
        let bus_type = match b {
            "pcie.0" if pci => "PCIE",
            "main-system-bus" => "System",
            _ => return Err(Located::new(loc, format!("Bus '{b}' not found"))),
        };
        if (bus_type == "PCIE") != is_pci {
            let msg = format!("Device '{typename}' can't go on {bus_type} bus");
            return Err(Located::new(loc, msg));
        }
    }
    if is_pci && !pci {
        return Err(Located::new(loc, format!("No 'PCI' bus found for device '{typename}'")));
    }
    let mut plug = DisplayPlug::new(model, loc.clone());
    plug.id = opts.id().map(str::to_string);
    let at = |e: Error| Located(loc.clone(), e);
    for (k, v) in opts.iter() {
        let vga = model == DisplayModel::Vga;
        let bochs = model == DisplayModel::BochsDisplay;
        match k {
            "driver" | "bus" => {}
            "id" => plug.id = Some(v.to_string()),
            "addr" if is_pci => {
                plug.devfn = Some(parse_devfn(v).ok_or_else(|| {
                    Located::new(
                        loc,
                        format!("Property '{typename}.addr' doesn't take value '{v}'"),
                    )
                })?);
            }
            "romfile" if is_pci => plug.romfile = (!v.is_empty()).then(|| v.to_string()),
            "vgamem_mb" if vga => plug.vgamem_mb = prop_u32(k, v).map_err(at)?,
            "mmio" if vga => plug.mmio = prop_bool(k, v).map_err(at)?,
            "qemu-extended-regs" if vga => plug.qemu_extended_regs = prop_bool(k, v).map_err(at)?,
            "vgamem" if bochs => plug.vgamem = prop_size(k, v).map_err(at)?,
            "edid" if is_pci => plug.edid = prop_bool(k, v).map_err(at)?,
            "xres" if is_pci => plug.xres = prop_u32(k, v).map_err(at)?,
            "yres" if is_pci => plug.yres = prop_u32(k, v).map_err(at)?,
            "xmax" if is_pci => plug.xmax = prop_u32(k, v).map_err(at)?,
            "ymax" if is_pci => plug.ymax = prop_u32(k, v).map_err(at)?,
            "refresh_rate" if is_pci => plug.refresh_rate = prop_u32(k, v).map_err(at)?,
            "big-endian-framebuffer" if is_pci => plug.big_endian = prop_bool(k, v).map_err(at)?,
            "use-legacy-x86-rom" if !is_pci => {
                plug.legacy_rom = Some(prop_bool(k, v).map_err(at)?);
            }
            // Only what migrates depends on it.
            "x-migrate" if !is_pci => {
                prop_bool(k, v).map_err(at)?;
            }
            _ => return Err(Located::new(loc, format!("Property '{typename}.{k}' not found"))),
        }
    }
    if !is_pci && !sysbus {
        return Err(Located::new(
            loc,
            format!("Option '-device {typename}' cannot be handled by this machine"),
        ));
    }
    Ok(plug)
}

/// Guest memory for ramfb that does not keep the machine alive.
struct WeakMemory(Weak<AddressSpace>);

impl DmaMemory for WeakMemory {
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.0.upgrade().is_some_and(|a| DmaMemory::read(&*a, addr, buf))
    }

    fn write(&self, addr: u64, buf: &[u8]) -> bool {
        self.0.upgrade().is_some_and(|a| DmaMemory::write(&*a, addr, buf))
    }
}

/// `ramfb_realizefn()`, and with `legacy_rom` the "vgaroms/vgabios-ramfb.bin" file that
/// `rom_add_vga()` puts in fw_cfg for SeaBIOS to run.
fn realize_ramfb(
    plug: &DisplayPlug,
    fw_cfg: &FwCfgState,
    memory: &Arc<AddressSpace>,
    legacy_rom: Option<&FirmwareSearch>,
) -> std::result::Result<(), Located> {
    let mem = Arc::new(WeakMemory(Arc::downgrade(memory)));
    Ramfb::realize(plug.id.clone(), Some(fw_cfg), mem, &DisplayState::global())
        .map_err(|e| Located(plug.loc.clone(), e))?;
    if let Some(firmware) = legacy_rom {
        const ROM: &str = "vgabios-ramfb.bin";
        match firmware.load(ROM) {
            Some(data) => {
                fw_cfg
                    .add_file(&format!("vgaroms/{ROM}"), data)
                    .map_err(|e| plug.at(e.to_string()))?;
            }
            // rom_add_file() reports it and the device works without it.
            None => eprintln!(
                "rom: file {ROM:<20}: error Failed to open file \u{201c}{ROM}\u{201d}: No such file or directory"
            ),
        }
    }
    Ok(())
}

/// Realizes a VGA or bochs-display function on `bus`. Gives the function and the name its
/// option ROM takes, `<vmsd name>.rom`.
fn realize_pci(
    bus: &PciBus,
    plug: &DisplayPlug,
    x86: bool,
) -> std::result::Result<(Arc<PciDevice>, &'static str), Located> {
    let ds = DisplayState::global();
    let at = |e: Error| Located(plug.loc.clone(), e);
    match plug.model {
        DisplayModel::Vga => {
            let props = VgaPciProps {
                id: plug.id.clone(),
                vgamem_mb: plug.vgamem_mb,
                mmio: plug.mmio,
                qemu_extended_regs: plug.qemu_extended_regs,
                edid: plug.edid,
                edid_info: plug.edid_info(),
                x86,
                big_endian: plug.big_endian,
            };
            let vga = VgaPci::realize(bus, plug.devfn, &props, &ds).map_err(at)?;
            Ok((Arc::clone(vga.pci_device()), "vga"))
        }
        DisplayModel::BochsDisplay => {
            let props = BochsDisplayProps {
                id: plug.id.clone(),
                vgamem: plug.vgamem,
                edid: plug.edid,
                edid_info: plug.edid_info(),
                big_endian: plug.big_endian,
            };
            // pci_bus_is_express(): both q35's pcie.0 and the GPEX root bus are.
            let dev = BochsDisplay::realize(bus, plug.devfn, true, &props, &ds).map_err(at)?;
            Ok((Arc::clone(dev.pci_device()), "bochs-display"))
        }
        DisplayModel::Ramfb => unreachable!("ramfb is not a PCI device"),
    }
}

/// `pc_vga_init()`: q35 plugs the `-vga std` card before the `-device` functions, so it takes
/// the first free slot, 00:01.0.
pub(crate) fn realize_x86_vga(
    board: &mut X86Board,
    vga: Option<VgaInterface>,
    firmware: &FirmwareSearch,
) -> std::result::Result<(), Located> {
    if vga != Some(VgaInterface::Std) || !matches!(board, X86Board::Q35(..)) {
        return Ok(());
    }
    VGA_INTERFACE_CREATED.store(true, Ordering::Release);
    realize_x86(board, &DisplayPlug::new(DisplayModel::Vga, None), firmware)
}

/// Realizes a planned display device on an x86 board.
pub(crate) fn realize_x86(
    board: &mut X86Board,
    plug: &DisplayPlug,
    firmware: &FirmwareSearch,
) -> std::result::Result<(), Located> {
    if plug.model == DisplayModel::Ramfb {
        let fw_cfg = match board {
            X86Board::Q35(m, _) => Arc::clone(m.fw_cfg()),
            X86Board::Microvm(m) => Arc::clone(m.fw_cfg()),
        };
        // The PC machines' compat properties set use-legacy-x86-rom.
        let legacy = plug.legacy_rom.unwrap_or(true).then_some(firmware);
        return realize_ramfb(plug, &fw_cfg, board.memory_as(), legacy);
    }
    let X86Board::Q35(m, _) = board else {
        return Err(plug.at(format!("No 'PCI' bus found for device '{}'", plug.typename())));
    };
    let (dev, rom_name) = realize_pci(m.pci_bus(), plug, true)?;
    // pci_add_option_rom()
    if let Some(name) = &plug.romfile {
        let Some(data) = firmware.load(name) else {
            return Err(plug.at(format!("failed to find romfile \"{name}\"")));
        };
        if data.is_empty() {
            return Err(plug.at(format!("romfile \"{name}\" is empty")));
        }
        let devfn = dev.devfn();
        let block = format!("0000:00:{:02x}.{:x}/{rom_name}.rom", devfn >> 3, devfn & 7);
        let rom = m.add_device_rom(&block, &data).map_err(|e| plug.at(e))?;
        dev.register_bar(PCI_ROM_SLOT, 0, rom);
    }
    Ok(())
}

/// Realizes a planned display device on Arm virt.
pub(crate) fn realize_virt(
    board: &VirtMachine,
    plug: &DisplayPlug,
) -> std::result::Result<(), Located> {
    if plug.model == DisplayModel::Ramfb {
        return realize_ramfb(plug, board.fw_cfg().state(), board.memory_as(), None);
    }
    realize_pci(board.gpex().bus(), plug, false).map(drop)
}

/// The QMP `screendump` command.
pub(crate) fn register(cmds: &mut Commands) {
    register_screendump(cmds, |_: &MonitorQmp, arg| {
        let format = arg.format.map(|f| match f {
            ImageFormat::Ppm => screendump::ImageFormat::Ppm,
            ImageFormat::Png => screendump::ImageFormat::Png,
        });
        let ds = DisplayState::global();
        screendump::screendump(&ds, &arg.filename, arg.device.as_deref(), arg.head, format)
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvm_qapi::opts::QemuOptsList;

    fn plan_one(arg: &str, pci: bool, sysbus: bool) -> std::result::Result<DisplayPlug, String> {
        let mut list = QemuOptsList::new("device", &[]).with_implied_opt_name("driver");
        let opts = list.parse(arg, true).unwrap();
        let driver = opts.get("driver").unwrap().to_string();
        plan_device(&driver, opts, &None, pci, sysbus)
            .expect("a display type")
            .map_err(|e| e.1.message().to_string())
    }

    #[test]
    fn vga_types() {
        assert_eq!(select_vgahw("std"), Ok(Some(VgaInterface::Std)));
        assert_eq!(select_vgahw("none"), Ok(Some(VgaInterface::None)));
        assert_eq!(select_vgahw("std,retrace=dumb,retrace=precise"), Ok(Some(VgaInterface::Std)));
        assert_eq!(select_vgahw("stdx"), Err("unknown vga type: stdx".to_string()));
        assert_eq!(select_vgahw("std,retrace=x"), Err("unknown vga type: std,retrace=x".into()));
        assert_eq!(select_vgahw("foo"), Err("unknown vga type: foo".to_string()));
        assert_eq!(select_vgahw("cirrus"), Err("Cirrus VGA not available".to_string()));
    }

    #[test]
    fn device_properties() {
        let p = plan_one("VGA,id=v,addr=3,vgamem_mb=32,edid=off,xres=800", true, true).unwrap();
        assert_eq!(p.model, DisplayModel::Vga);
        assert_eq!((p.id.as_deref(), p.devfn, p.vgamem_mb), (Some("v"), Some(0x18), 32));
        assert!(!p.edid);
        assert_eq!(p.xres, 800);
        assert_eq!(p.romfile.as_deref(), Some(VGA_ROMFILE));
        let p = plan_one("bochs-display,vgamem=64M,romfile=", true, true).unwrap();
        assert_eq!((p.vgamem, p.romfile), (64 << 20, None));
        let p = plan_one("ramfb,use-legacy-x86-rom=off", false, true).unwrap();
        assert_eq!(p.legacy_rom, Some(false));

        assert_eq!(plan_one("VGA,foo=1", true, true).unwrap_err(), "Property 'VGA.foo' not found");
        assert_eq!(
            plan_one("bochs-display,vgamem_mb=1", true, true).unwrap_err(),
            "Property 'bochs-display.vgamem_mb' not found"
        );
        assert_eq!(
            plan_one("VGA,addr=20", true, true).unwrap_err(),
            "Property 'VGA.addr' doesn't take value '20'"
        );
        assert_eq!(plan_one("VGA,bus=pci.0", true, true).unwrap_err(), "Bus 'pci.0' not found");
        assert_eq!(
            plan_one("ramfb,bus=pcie.0", true, true).unwrap_err(),
            "Device 'ramfb' can't go on PCIE bus"
        );
        assert_eq!(
            plan_one("VGA,bus=main-system-bus", true, true).unwrap_err(),
            "Device 'VGA' can't go on System bus"
        );
        assert_eq!(
            plan_one("VGA", false, true).unwrap_err(),
            "No 'PCI' bus found for device 'VGA'"
        );
        assert_eq!(
            plan_one("ramfb", false, false).unwrap_err(),
            "Option '-device ramfb' cannot be handled by this machine"
        );
        let mut list = QemuOptsList::new("device", &[]).with_implied_opt_name("driver");
        let opts = list.parse("virtio-rng-pci", true).unwrap();
        assert!(plan_device("virtio-rng-pci", opts, &None, true, true).is_none());
    }
}
