// SPDX-License-Identifier: GPL-2.0-or-later

//! The microvm tables from hw/i386/acpi-microvm.c. microvm is hardware reduced ACPI: no PM
//! block, no FACS, a Generic Event Device for power button events and sleep and reset registers
//! in MMIO.

use super::BuildTables;
use super::aml::{self, AddressSpace, Aml};
use super::devices::{self, IsaDevice};
use super::linker::BiosLinker;
use super::table::{self, AcpiTable, FadtData, Gas, RsdpData, TABLE_FILE, fadt_flags};
use super::x86::{self, MadtConfig};

/// `VIRTIO_MMIO_BASE`.
pub const VIRTIO_MMIO_BASE: u64 = 0xfeb0_0000;
/// Each virtio-mmio transport takes 512 bytes.
pub const VIRTIO_MMIO_SIZE: u64 = 512;
/// `GED_MMIO_BASE`.
pub const GED_MMIO_BASE: u64 = 0xfea0_0000;
/// `GED_MMIO_BASE_REGS`, the sleep and reset registers.
pub const GED_MMIO_BASE_REGS: u64 = GED_MMIO_BASE + 0x200;
/// `GED_MMIO_IRQ`.
pub const GED_MMIO_IRQ: u32 = 9;
/// `MICROVM_XHCI_BASE`.
pub const MICROVM_XHCI_BASE: u32 = 0xfe90_0000;
/// `MICROVM_XHCI_IRQ`.
pub const MICROVM_XHCI_IRQ: u32 = 10;

/// `ACPI_GED_REG_SLEEP_CTL`.
const GED_REG_SLEEP_CTL: u64 = 0x00;
/// `ACPI_GED_REG_SLEEP_STS`.
const GED_REG_SLEEP_STS: u64 = 0x01;
/// `ACPI_GED_REG_RESET`.
const GED_REG_RESET: u64 = 0x02;
/// `ACPI_GED_RESET_VALUE`.
const GED_RESET_VALUE: u8 = 0x42;
/// `ACPI_GED_SLP_TYP_S5`.
const GED_SLP_TYP_S5: u64 = 0x05;

/// Everything about a microvm instance that shows up in its tables.
#[derive(Clone, Debug)]
pub struct MicrovmAcpi {
    pub oem_id: String,
    pub oem_table_id: String,
    pub madt: MadtConfig,
    /// The ISA devices in bus order.
    pub isa: Vec<IsaDevice>,
    /// The GED's `ged-event` bitmap.
    pub ged_events: u32,
    /// First GSI of the virtio-mmio transports, `virtio_irq_base`.
    pub virtio_irq_base: u32,
    /// The virtio-mmio transports that have a device plugged in.
    pub virtio_transports: Vec<u32>,
    pub usb: bool,
    /// `iapc_boot_arch_8042()`, whether there is an i8042.
    pub i8042: bool,
}

/// `build_dsdt_microvm()`. The PCIe host bridge is not supported yet.
fn build_dsdt(table_data: &mut Vec<u8>, linker: &mut BiosLinker, m: &MicrovmAcpi) {
    let table = AcpiTable::begin("DSDT", 2, &m.oem_id, &m.oem_table_id, table_data);
    let mut dsdt = Aml::new();
    let mut sb = aml::scope("_SB");
    devices::fw_cfg_x86(&mut sb);
    for dev in &m.isa {
        dev.build_aml(&mut sb);
    }
    devices::ged(
        &mut sb,
        devices::GED_DEVICE,
        m.ged_events,
        GED_MMIO_IRQ,
        aml::RegionSpace::SystemMemory,
        GED_MMIO_BASE,
    );
    devices::power_button(&mut sb);
    for &index in &m.virtio_transports {
        let base = VIRTIO_MMIO_BASE + u64::from(index) * VIRTIO_MMIO_SIZE;
        devices::virtio_mmio(&mut sb, base, VIRTIO_MMIO_SIZE, m.virtio_irq_base + index, index, 1);
    }
    if m.usb {
        devices::xhci_sysbus(&mut sb, MICROVM_XHCI_BASE, MICROVM_XHCI_IRQ);
    }
    dsdt.append(&sb);

    // ACPI 5.0, Table 7-209, the System State package.
    let mut scope = aml::scope("\\");
    let mut pkg = aml::package(4);
    pkg.append(&aml::int(GED_SLP_TYP_S5));
    pkg.append(&aml::int(0)); // ignored
    pkg.append(&aml::int(0)); // reserved
    pkg.append(&aml::int(0)); // reserved
    scope.append(&aml::name_decl("_S5", &pkg));
    dsdt.append(&scope);

    table_data.extend_from_slice(dsdt.as_bytes());
    table.end(Some(linker), table_data);
}

/// `acpi_build_microvm()`.
pub fn build(m: &MicrovmAcpi) -> BuildTables {
    let mut t = BuildTables::new();
    let data = &mut t.table_data;
    let linker = &mut t.linker;
    // 64 so the FACS would be aligned, though microvm has none.
    linker.alloc(TABLE_FILE, 64, false);

    let dsdt = data.len() as u32;
    build_dsdt(data, linker, m);

    let reg = |off| Gas::new(AddressSpace::SystemMemory, 8, GED_MMIO_BASE_REGS + off);
    let fadt = FadtData {
        // ACPI 5.0, 4.1 Hardware-Reduced ACPI.
        rev: 5,
        flags: 1 << fadt_flags::HW_REDUCED_ACPI | 1 << fadt_flags::RESET_REG_SUP,
        sleep_ctl: reg(GED_REG_SLEEP_CTL),
        sleep_sts: reg(GED_REG_SLEEP_STS),
        reset_reg: reg(GED_REG_RESET),
        reset_val: GED_RESET_VALUE,
        // ACPI 2.0, Table 5-10, bit 1 is 8042.
        iapc_boot_arch: if m.i8042 { 0x2 } else { 0 },
        dsdt_tbl_offset: Some(dsdt),
        xdsdt_tbl_offset: Some(dsdt),
        ..FadtData::default()
    };
    let mut table_offsets = Vec::new();
    table::add_table(&mut table_offsets, data);
    table::build_fadt(data, linker, &fadt, &m.oem_id, &m.oem_table_id);

    table::add_table(&mut table_offsets, data);
    x86::build_madt(data, linker, &m.madt, &m.oem_id, &m.oem_table_id);

    let xsdt = data.len() as u32;
    table::build_xsdt(data, linker, &table_offsets, &m.oem_id, &m.oem_table_id);

    // The RSDP lives in the F segment, so it is a file of its own. XSDT needs revision 2.
    let rsdp = RsdpData {
        revision: 2,
        oem_id: &m.oem_id,
        xsdt_tbl_offset: Some(xsdt),
        rsdt_tbl_offset: None,
    };
    table::build_rsdp(&mut t.rsdp, linker, &rsdp);
    t
}
