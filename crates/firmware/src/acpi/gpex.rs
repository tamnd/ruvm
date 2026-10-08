// SPDX-License-Identifier: GPL-2.0-or-later

//! The AML of the generic PCIe host bridge, `gpex-pcihost`, from hw/pci-host/gpex-acpi.c:
//! the `PCI0` device with its routing table, the `_CRS` of its windows, `_OSC`, `_DSM` and
//! the `RES0` device that reserves the ECAM, and the four GSI link devices.
//!
//! Only the root bus is described. QEMU also emits a device per PCI expander bridge
//! (`pxb-pcie`) and takes their windows out of `PCI0._CRS`; ruvm has no expander bridges.

use super::aml::{self, Aml, Serialize};
use super::pci::{self, CrsRangeSet};

/// `PCI_SLOT_MAX`.
const PCI_SLOT_MAX: u32 = 32;
/// `PCI_NUM_PINS`.
const PCI_NUM_PINS: u32 = 4;
/// `PCIE_MMCFG_SIZE_MIN`, the ECAM space of one bus.
const PCIE_MMCFG_SIZE_MIN: u64 = 1 << 20;

/// One window of the bridge, `struct GPEXConfig`'s `MemMapEntry` fields.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Window {
    pub base: u64,
    pub size: u64,
}

/// `struct GPEXConfig`: what the AML says about the bridge.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpexAcpi {
    pub ecam: Window,
    pub mmio32: Window,
    pub mmio64: Window,
    /// The I/O port window as the CPU sees it.
    pub pio: Window,
    /// The GSI of INTA of slot 0; INTB to INTD follow it.
    pub irq: u32,
    /// Native PCIe hotplug is offered through `_OSC`.
    pub pci_native_hotplug: bool,
    /// `_DSM` function 5 asks the OS to keep the firmware's resource assignment.
    pub preserve_config: bool,
}

/// `acpi_dsdt_add_pci_route_table()`: `_PRT` on `dev`, and the link devices `LnnX` in `scope`.
fn add_pci_route_table(dev: &mut Aml, irq: u32, scope: &mut Aml, bus_num: u8) {
    // Declare the PCI Routing Table.
    let mut rt_pkg = aml::varpackage(PCI_SLOT_MAX * PCI_NUM_PINS);
    for slot_no in 0..PCI_SLOT_MAX {
        for i in 0..PCI_NUM_PINS {
            let gsi = (i + slot_no) % PCI_NUM_PINS;
            let mut pkg = aml::package(4);
            pkg.append(&aml::int(u64::from(slot_no << 16 | 0xFFFF)));
            pkg.append(&aml::int(i.into()));
            pkg.append(&aml::name(&format!("L{bus_num:02X}{gsi:X}")));
            pkg.append(&aml::int(0));
            rt_pkg.append(&pkg);
        }
    }
    dev.append(&aml::name_decl("_PRT", &rt_pkg));

    // Create GSI link device
    for i in 0..PCI_NUM_PINS {
        let irqs = [irq + i];
        let mut dev_gsi = aml::device(&format!("L{bus_num:02X}{i:X}"));
        dev_gsi.append(&aml::name_decl("_HID", &aml::string("PNP0C0F")));
        dev_gsi.append(&aml::name_decl("_UID", &aml::int(i.into())));
        let crs = || {
            let mut crs = aml::resource_template();
            crs.append(&aml::interrupt(
                aml::ConsumerProducer::Consumer,
                aml::Trigger::Level,
                aml::Polarity::ActiveHigh,
                aml::Shared::Exclusive,
                &irqs,
            ));
            crs
        };
        dev_gsi.append(&aml::name_decl("_PRS", &crs()));
        dev_gsi.append(&aml::name_decl("_CRS", &crs()));
        dev_gsi.append(&aml::method("_SRS", 1, Serialize::NotSerialized));
        scope.append(&dev_gsi);
    }
}

/// `build_pci_host_bridge_dsm_method()`.
fn host_bridge_dsm(preserve_config: bool) -> Aml {
    let mut method = aml::method("_DSM", 4, Serialize::NotSerialized);
    // PCI Firmware Specification 3.0, 4.6.1, _DSM for PCI Express Slot Information.
    let mut ifctx = aml::if_(&aml::equal(&aml::arg(0), &aml::touuid(pci::PCI_DSM_UUID)));
    let mut ifctx1 = aml::if_(&aml::equal(&aml::arg(2), &aml::int(0)));
    // With preserve_config, function 5 is supported as well as function 0.
    let funcs = if preserve_config { 0x21 } else { 0 };
    ifctx1.append(&aml::return_(&aml::buffer(1, Some(&[funcs]))));
    ifctx.append(&ifctx1);
    if preserve_config {
        let mut ifctx2 = aml::if_(&aml::equal(&aml::arg(2), &aml::int(5)));
        // 0: the OS must not ignore the PCI configuration the firmware did at boot.
        ifctx2.append(&aml::return_(&aml::int(0)));
        ifctx.append(&ifctx2);
    }
    method.append(&ifctx);
    method.append(&aml::return_(&aml::buffer(1, Some(&[0]))));
    method
}

/// `acpi_dsdt_add_host_bridge_methods()`.
fn add_host_bridge_methods(dev: &mut Aml, enable_native_pcie_hotplug: bool, preserve: bool) {
    // Declare an _OSC (OS Control Handoff) method
    dev.append(&pci::host_bridge_osc(enable_native_pcie_hotplug));
    dev.append(&host_bridge_dsm(preserve));
}

/// `acpi_dsdt_add_gpex()` for a bridge without expander buses.
pub fn add_gpex(scope: &mut Aml, cfg: &GpexAcpi) {
    let nr_pcie_buses = cfg.ecam.size / PCIE_MMCFG_SIZE_MIN;
    // The windows of the expander buses, which there are none of.
    let mut crs_range_set = CrsRangeSet::default();

    // tables for the main
    let mut dev = aml::device("PCI0");
    dev.append(&aml::name_decl("_HID", &aml::string("PNP0A08")));
    dev.append(&aml::name_decl("_CID", &aml::string("PNP0A03")));
    dev.append(&aml::name_decl("_SEG", &aml::int(0)));
    dev.append(&aml::name_decl("_BBN", &aml::int(0)));
    dev.append(&aml::name_decl("_UID", &aml::int(0)));
    dev.append(&aml::name_decl("_STR", &aml::unicode("PCIe 0 Device")));
    dev.append(&aml::name_decl("_CCA", &aml::int(1)));

    add_pci_route_table(&mut dev, cfg.irq, scope, 0);

    let mut method = aml::method("_CBA", 0, Serialize::NotSerialized);
    method.append(&aml::return_(&aml::int(cfg.ecam.base)));
    dev.append(&method);

    // crs_range_set holds the ranges of the buses other than PCI0, which PCI0._CRS leaves out.
    let mut rbuf = aml::resource_template();
    rbuf.append(&aml::word_bus_number(
        aml::MinFixed::Fixed,
        aml::MaxFixed::Fixed,
        aml::Decode::Pos,
        0x0000,
        0x0000,
        (nr_pcie_buses - 1) as u16,
        0x0000,
        nr_pcie_buses as u16,
    ));
    if cfg.mmio32.size != 0 {
        let ranges = &mut crs_range_set.mem_ranges;
        let end = cfg.mmio32.base + cfg.mmio32.size - 1;
        pci::crs_replace_with_free_ranges(ranges, cfg.mmio32.base, end);
        for r in ranges.iter() {
            rbuf.append(&aml::dword_memory(
                aml::Decode::Pos,
                aml::MinFixed::Fixed,
                aml::MaxFixed::Fixed,
                aml::Cacheable::NonCacheable,
                aml::ReadWrite::ReadWrite,
                0x0000,
                r.base as u32,
                r.limit as u32,
                0x0000,
                (r.limit - r.base + 1) as u32,
            ));
        }
    }
    if cfg.pio.size != 0 {
        let ranges = &mut crs_range_set.io_ranges;
        pci::crs_replace_with_free_ranges(ranges, 0x0000, cfg.pio.size - 1);
        for r in ranges.iter() {
            rbuf.append(&aml::dword_io(
                aml::MinFixed::Fixed,
                aml::MaxFixed::Fixed,
                aml::Decode::Pos,
                aml::IsaRanges::EntireRange,
                0x0000,
                r.base as u32,
                r.limit as u32,
                cfg.pio.base as u32,
                (r.limit - r.base + 1) as u32,
            ));
        }
    }
    if cfg.mmio64.size != 0 {
        let ranges = &mut crs_range_set.mem_64bit_ranges;
        let end = cfg.mmio64.base + cfg.mmio64.size - 1;
        pci::crs_replace_with_free_ranges(ranges, cfg.mmio64.base, end);
        for r in ranges.iter() {
            rbuf.append(&aml::qword_memory(
                aml::Decode::Pos,
                aml::MinFixed::Fixed,
                aml::MaxFixed::Fixed,
                aml::Cacheable::NonCacheable,
                aml::ReadWrite::ReadWrite,
                0x0000,
                r.base,
                r.limit,
                0x0000,
                r.limit - r.base + 1,
            ));
        }
    }
    dev.append(&aml::name_decl("_CRS", &rbuf));

    add_host_bridge_methods(&mut dev, cfg.pci_native_hotplug, cfg.preserve_config);

    let mut dev_res0 = aml::device("RES0");
    dev_res0.append(&aml::name_decl("_HID", &aml::string("PNP0C02")));
    let mut crs = aml::resource_template();
    crs.append(&aml::qword_memory(
        aml::Decode::Pos,
        aml::MinFixed::Fixed,
        aml::MaxFixed::Fixed,
        aml::Cacheable::NonCacheable,
        aml::ReadWrite::ReadWrite,
        0x0000,
        cfg.ecam.base,
        cfg.ecam.base + cfg.ecam.size - 1,
        0x0000,
        cfg.ecam.size,
    ));
    dev_res0.append(&aml::name_decl("_CRS", &crs));
    dev.append(&dev_res0);
    scope.append(&dev);
}
