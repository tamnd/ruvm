// SPDX-License-Identifier: GPL-2.0-or-later

//! AML for individual devices. In QEMU each of these lives next to its device model (the
//! `build_dev_aml` hooks and helpers such as `build_ged_aml()`). They are collected here as plain
//! functions of the few values each one needs, so a machine can emit them without the table code
//! depending on the device crates.

use super::aml::{self, Aml};

/// `FW_CFG_IO_BASE`.
pub const FW_CFG_IO_BASE: u16 = 0x510;
/// `FW_CFG_CTL_SIZE`.
const FW_CFG_CTL_SIZE: u8 = 0x02;

/// `fw_cfg_add_acpi_dsdt()`, the x86 port I/O fw_cfg device. It always has DMA, so the region
/// covers the control and data ports rounded to four bytes plus the 8 byte DMA address register.
pub fn fw_cfg_x86(scope: &mut Aml) {
    let io_size = FW_CFG_CTL_SIZE.next_multiple_of(4) + 8;
    let mut dev = aml::device("FWCF");
    dev.append(&aml::name_decl("_HID", &aml::string("QEMU0002")));
    // Present, functioning, decoding, not shown in UI.
    dev.append(&aml::name_decl("_STA", &aml::int(0xB)));
    let mut crs = aml::resource_template();
    crs.append(&aml::io(aml::IoDecode::Decode16, FW_CFG_IO_BASE, FW_CFG_IO_BASE, 0x01, io_size));
    dev.append(&aml::name_decl("_CRS", &crs));
    scope.append(&dev);
}

/// `serial_isa_build_aml()`. `index` is the zero based `index` property, so COM1 is 0.
pub fn serial_isa(scope: &mut Aml, index: u32, iobase: u16, isairq: u8) {
    let mut crs = aml::resource_template();
    crs.append(&aml::io(aml::IoDecode::Decode16, iobase, iobase, 0x00, 0x08));
    crs.append(&aml::irq(
        isairq,
        aml::Trigger::Level,
        aml::Polarity::ActiveLow,
        aml::Shared::Shared,
    ));
    let mut dev = aml::device(&format!("COM{}", index + 1));
    dev.append(&aml::name_decl("_HID", &aml::eisaid("PNP0501")));
    dev.append(&aml::name_decl("_UID", &aml::int((index + 1).into())));
    dev.append(&aml::name_decl("_STA", &aml::int(0xf)));
    dev.append(&aml::name_decl("_CRS", &crs));
    scope.append(&dev);
}

/// `rtc_build_aml()` for the MC146818. Eight ports are reserved like on real hardware even though
/// only the first two respond.
pub fn rtc_mc146818(scope: &mut Aml, io_base: u16, isairq: u8) {
    let mut crs = aml::resource_template();
    crs.append(&aml::io(aml::IoDecode::Decode16, io_base, io_base, 0x01, 0x08));
    crs.append(&aml::irq_no_flags(isairq));
    let mut dev = aml::device("RTC");
    dev.append(&aml::name_decl("_HID", &aml::eisaid("PNP0B00")));
    dev.append(&aml::name_decl("_CRS", &crs));
    scope.append(&dev);
}

/// `ACPI_POWER_BUTTON_DEVICE`.
pub const POWER_BUTTON_DEVICE: &str = "PWRB";
/// `ACPI_APEI_ERROR_DEVICE`.
pub const APEI_ERROR_DEVICE: &str = "GEDD";
/// `GED_DEVICE`.
pub const GED_DEVICE: &str = "GED";

/// GED event bits, `ACPI_GED_*_EVT`.
pub mod ged_event {
    pub const MEM_HOTPLUG: u32 = 0x1;
    pub const PWR_DOWN: u32 = 0x2;
    pub const NVDIMM_HOTPLUG: u32 = 0x4;
    pub const CPU_HOTPLUG: u32 = 0x8;
    pub const PCI_HOTPLUG: u32 = 0x10;
    pub const ERROR: u32 = 0x20;
}

/// `ACPI_GED_EVT_SEL_OFFSET`.
pub const GED_EVT_SEL_OFFSET: u64 = 0x0;
/// `ACPI_GED_EVT_SEL_LEN`.
pub const GED_EVT_SEL_LEN: u32 = 0x4;

/// `ged_supported_events[]`, in the order the `_EVT` method tests them.
const GED_SUPPORTED_EVENTS: [u32; 6] = [
    ged_event::MEM_HOTPLUG,
    ged_event::PWR_DOWN,
    ged_event::NVDIMM_HOTPLUG,
    ged_event::CPU_HOTPLUG,
    ged_event::PCI_HOTPLUG,
    ged_event::ERROR,
];

/// `build_ged_aml()`. `event_bitmap` is the device's `ged-event` property.
pub fn ged(
    table: &mut Aml,
    name: &str,
    event_bitmap: u32,
    ged_irq: u32,
    rs: aml::RegionSpace,
    ged_base: u64,
) {
    assert_eq!(
        event_bitmap & !GED_SUPPORTED_EVENTS.iter().fold(0, |a, e| a | e),
        0,
        "Unsupported events specified"
    );
    let mut crs = aml::resource_template();
    crs.append(&aml::interrupt(
        aml::ConsumerProducer::Consumer,
        aml::Trigger::Edge,
        aml::Polarity::ActiveHigh,
        aml::Shared::Exclusive,
        &[ged_irq],
    ));
    let mut dev = aml::device(name);
    dev.append(&aml::name_decl("_HID", &aml::string("ACPI0013")));
    dev.append(&aml::name_decl("_UID", &aml::string(GED_DEVICE)));
    dev.append(&aml::name_decl("_CRS", &crs));
    dev.append(&aml::operation_region(
        "EREG",
        rs,
        &aml::int(ged_base + GED_EVT_SEL_OFFSET),
        GED_EVT_SEL_LEN,
    ));
    let mut field = aml::field(
        "EREG",
        aml::AccessType::Dword,
        aml::LockRule::NoLock,
        aml::UpdateRule::WriteAsZeros,
    );
    field.append(&aml::named_field("ESEL", (GED_EVT_SEL_LEN * 8) as usize));
    dev.append(&field);

    // Local0 = ESEL, then one If per event: If ((Local0 & Event) == Event) { handler }.
    let evt_sel = aml::local(0);
    let mut evt = aml::method("_EVT", 1, aml::Serialize::Serialized);
    evt.append(&aml::store(&aml::name("ESEL"), &evt_sel));
    for event in GED_SUPPORTED_EVENTS.into_iter().filter(|e| event_bitmap & e != 0) {
        let ev = aml::int(event.into());
        let mut if_ctx = aml::if_(&aml::equal(&aml::and(&evt_sel, &ev, None), &ev));
        let notify = |target: &str| aml::notify(&aml::name(target), &aml::int(0x80));
        match event {
            ged_event::MEM_HOTPLUG => if_ctx.append(&aml::call("\\_SB.MHPC.MSCN", &[])),
            ged_event::CPU_HOTPLUG => if_ctx.append(&aml::call("\\_SB.GED.CSCN", &[])),
            ged_event::PWR_DOWN => if_ctx.append(&notify(POWER_BUTTON_DEVICE)),
            ged_event::ERROR => if_ctx.append(&notify(APEI_ERROR_DEVICE)),
            ged_event::NVDIMM_HOTPLUG => if_ctx.append(&notify("\\_SB.NVDR")),
            _ => {
                if_ctx.append(&aml::acquire(&aml::name("\\_SB.PCI0.BLCK"), 0xFFFF));
                if_ctx.append(&aml::call("\\_SB.PCI0.PCNT", &[]));
                if_ctx.append(&aml::release(&aml::name("\\_SB.PCI0.BLCK")));
            }
        }
        evt.append(&if_ctx);
    }
    dev.append(&evt);
    table.append(&dev);
}

/// `acpi_dsdt_add_power_button()`.
pub fn power_button(scope: &mut Aml) {
    let mut dev = aml::device(POWER_BUTTON_DEVICE);
    dev.append(&aml::name_decl("_HID", &aml::string("PNP0C0C")));
    dev.append(&aml::name_decl("_UID", &aml::int(0)));
    scope.append(&dev);
}

/// `virtio_acpi_dsdt_add()`: `num` virtio-mmio transports starting at transport `start_index`,
/// each `size` bytes after the previous one with the next interrupt line.
pub fn virtio_mmio(
    scope: &mut Aml,
    base: u64,
    size: u64,
    mmio_irq: u32,
    start_index: u32,
    num: u32,
) {
    let (mut base, mut irq) = (base, mmio_irq);
    for i in start_index..start_index + num {
        let mut dev = aml::device(&format!("VR{i:02}"));
        dev.append(&aml::name_decl("_HID", &aml::string("LNRO0005")));
        dev.append(&aml::name_decl("_UID", &aml::int(i.into())));
        dev.append(&aml::name_decl("_CCA", &aml::int(1)));
        let mut crs = aml::resource_template();
        crs.append(&aml::memory32_fixed(base as u32, size as u32, aml::ReadWrite::ReadWrite));
        crs.append(&aml::interrupt(
            aml::ConsumerProducer::Consumer,
            aml::Trigger::Level,
            aml::Polarity::ActiveHigh,
            aml::Shared::Exclusive,
            &[irq],
        ));
        dev.append(&aml::name_decl("_CRS", &crs));
        scope.append(&dev);
        base += size;
        irq += 1;
    }
}

/// `XHCI_LEN_REGS`.
pub const XHCI_LEN_REGS: u32 = 0x4000;

/// `xhci_sysbus_build_aml()`.
pub fn xhci_sysbus(scope: &mut Aml, mmio: u32, irq: u32) {
    let mut crs = aml::resource_template();
    crs.append(&aml::memory32_fixed(mmio, XHCI_LEN_REGS, aml::ReadWrite::ReadWrite));
    crs.append(&aml::interrupt(
        aml::ConsumerProducer::Consumer,
        aml::Trigger::Level,
        aml::Polarity::ActiveHigh,
        aml::Shared::Exclusive,
        &[irq],
    ));
    let mut dev = aml::device("XHCI");
    dev.append(&aml::name_decl("_HID", &aml::eisaid("PNP0D10")));
    dev.append(&aml::name_decl("_CRS", &crs));
    scope.append(&dev);
}

/// A device on the ISA bus that describes itself in the DSDT, the `AcpiDevAmlIf` devices that
/// `qbus_build_aml()` walks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IsaDevice {
    /// isa-serial, see [`serial_isa`].
    Serial { index: u32, iobase: u16, irq: u8 },
    /// mc146818rtc, see [`rtc_mc146818`].
    Rtc { io_base: u16, irq: u8 },
}

impl IsaDevice {
    /// The device's `build_dev_aml` hook.
    pub fn build_aml(&self, scope: &mut Aml) {
        match *self {
            IsaDevice::Serial { index, iobase, irq } => serial_isa(scope, index, iobase, irq),
            IsaDevice::Rtc { io_base, irq } => rtc_mc146818(scope, io_base, irq),
        }
    }
}
