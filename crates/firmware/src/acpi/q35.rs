// SPDX-License-Identifier: GPL-2.0-or-later

//! The pc machine tables from hw/i386/acpi-build.c as the q35 machine builds them. The ICH9 LPC
//! bridge supplies the PM block, so unlike microvm there is a FACS, a real PM timer and the
//! classic fixed feature registers.
//!
//! The DSDT body is passed in by the caller for now. Only the fixed tables around it are built
//! here.

use super::BuildTables;
use super::aml::{AddressSpace, Aml};
use super::linker::BiosLinker;
use super::table::{self, AcpiTable, FadtData, Gas, McfgInfo, RsdpData, TABLE_FILE, fadt_flags};
use super::x86::{self, MadtConfig};

/// `ACPI_PORT_SMI_CMD`, the APM control port.
pub const PORT_SMI_CMD: u32 = 0xb2;
/// `ICH9_RST_CNT_IOPORT`.
pub const ICH9_RST_CNT_IOPORT: u64 = 0xcf9;
/// `ICH9_APM_ACPI_ENABLE`.
pub const ICH9_APM_ACPI_ENABLE: u8 = 0x2;
/// `ICH9_APM_ACPI_DISABLE`.
pub const ICH9_APM_ACPI_DISABLE: u8 = 0x3;
/// `ICH9_PMIO_GPE0_STS`, the GPE0 block offset in the PM I/O space.
pub const ICH9_PMIO_GPE0_STS: u16 = 0x20;
/// `ICH9_PMIO_GPE0_LEN`.
pub const ICH9_PMIO_GPE0_LEN: u8 = 16;
/// `RTC_CENTURY`, the CMOS index of the century byte.
pub const RTC_CENTURY: u8 = 0x32;
/// `ACPI_BUILD_TABLE_SIZE`.
pub const TABLE_SIZE: usize = 0x20000;
/// `ACPI_BUILD_ALIGN_SIZE`.
pub const ALIGN_SIZE: usize = 0x1000;

/// Everything about a q35 instance that shows up in the fixed tables.
#[derive(Clone, Debug)]
pub struct Q35Acpi {
    pub oem_id: String,
    pub oem_table_id: String,
    pub madt: MadtConfig,
    /// `-smp maxcpus`. Above 8 the FADT asks for clustered logical APIC mode.
    pub max_cpus: u32,
    /// SMM enabled, or the LPC `smm-compat` property.
    pub smm: bool,
    /// The LPC `sci-int` property.
    pub sci_int: u16,
    /// The PM I/O base, `ACPI_PM_PROP_PM_IO_BASE`.
    pub pm_io_base: u16,
    /// `iapc_boot_arch_8042()`, whether there is an i8042.
    pub i8042: bool,
    /// Whether an HPET is present.
    pub hpet: bool,
    /// The ECAM window, `None` while the firmware has not mapped it.
    pub mcfg: Option<McfgInfo>,
}

impl Q35Acpi {
    /// `init_common_fadt_data()` plus the ICH9 part of `acpi_get_pm_info()`.
    pub fn fadt(&self) -> FadtData {
        let io = u64::from(self.pm_io_base);
        let sysio = |width: u8, address: u64| Gas::new(AddressSpace::SystemIo, width, address);
        let mut flags = 1 << fadt_flags::WBINVD
            | 1 << fadt_flags::PROC_C1
            | 1 << fadt_flags::SLP_BUTTON
            | 1 << fadt_flags::RTC_S4
            | 1 << fadt_flags::USE_PLATFORM_CLOCK
            | 1 << fadt_flags::RESET_REG_SUP;
        // Flat logical APIC destination mode tops out at 8 CPUs.
        if self.max_cpus > 8 {
            flags |= 1 << fadt_flags::FORCE_APIC_CLUSTER_MODEL;
        }
        FadtData {
            rev: 3,
            flags,
            int_model: 1, // multiple APIC
            rtc_century: RTC_CENTURY,
            plvl2_lat: 0xfff, // no C2
            plvl3_lat: 0xfff, // no C3
            smi_cmd: if self.smm { PORT_SMI_CMD } else { 0 },
            sci_int: self.sci_int,
            acpi_enable_cmd: if self.smm { ICH9_APM_ACPI_ENABLE } else { 0 },
            acpi_disable_cmd: if self.smm { ICH9_APM_ACPI_DISABLE } else { 0 },
            pm1a_evt: sysio(32, io),
            pm1a_cnt: sysio(16, io + 0x04),
            pm_tmr: sysio(32, io + 0x08),
            gpe0_blk: sysio(ICH9_PMIO_GPE0_LEN * 8, io + u64::from(ICH9_PMIO_GPE0_STS)),
            reset_reg: sysio(8, ICH9_RST_CNT_IOPORT),
            reset_val: 0xf,
            iapc_boot_arch: if self.i8042 { 0x2 } else { 0 },
            ..FadtData::default()
        }
    }
}

fn build_dsdt(table_data: &mut Vec<u8>, linker: &mut BiosLinker, q: &Q35Acpi, body: &Aml) {
    let table = AcpiTable::begin("DSDT", 1, &q.oem_id, &q.oem_table_id, table_data);
    table_data.extend_from_slice(body.as_bytes());
    table.end(Some(linker), table_data);
}

/// `acpi_build()` for q35 with `dsdt` as the DSDT contents.
pub fn build(q: &Q35Acpi, dsdt: &Aml) -> BuildTables {
    let mut t = BuildTables::new();
    let data = &mut t.table_data;
    let linker = &mut t.linker;
    linker.alloc(TABLE_FILE, 64, false);

    // The FACS goes first because it is the only table with an alignment requirement.
    let facs = data.len() as u32;
    x86::build_facs(data);

    let dsdt_off = data.len() as u32;
    build_dsdt(data, linker, q, dsdt);

    let mut table_offsets = Vec::new();
    table::add_table(&mut table_offsets, data);
    let fadt = FadtData {
        facs_tbl_offset: Some(facs),
        dsdt_tbl_offset: Some(dsdt_off),
        xdsdt_tbl_offset: Some(dsdt_off),
        ..q.fadt()
    };
    table::build_fadt(data, linker, &fadt, &q.oem_id, &q.oem_table_id);

    table::add_table(&mut table_offsets, data);
    x86::build_madt(data, linker, &q.madt, &q.oem_id, &q.oem_table_id);

    if q.hpet {
        table::add_table(&mut table_offsets, data);
        x86::build_hpet(data, linker, &q.oem_id, &q.oem_table_id);
    }
    if let Some(mcfg) = &q.mcfg {
        table::add_table(&mut table_offsets, data);
        table::build_mcfg(data, linker, mcfg, &q.oem_id, &q.oem_table_id);
    }
    table::add_table(&mut table_offsets, data);
    x86::build_waet(data, linker, &q.oem_id, &q.oem_table_id);

    let rsdt = data.len() as u32;
    table::build_rsdt(data, linker, &table_offsets, &q.oem_id, &q.oem_table_id);

    let rsdp = RsdpData {
        revision: 0,
        oem_id: &q.oem_id,
        xsdt_tbl_offset: None,
        rsdt_tbl_offset: Some(rsdt),
    };
    table::build_rsdp(&mut t.rsdp, linker, &rsdp);

    table::align_size(&mut t.table_data, TABLE_SIZE);
    t.linker.pad_to(ALIGN_SIZE);
    t
}
