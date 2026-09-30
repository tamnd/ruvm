// SPDX-License-Identifier: GPL-2.0-or-later

//! The pc machine tables from hw/i386/acpi-build.c as the q35 machine builds them. The ICH9 LPC
//! bridge supplies the PM block, so unlike microvm there is a FACS, a real PM timer and the
//! classic fixed feature registers.
//!
//! Not covered yet: PCI expander buses, CXL, TPM, SGX EPC, VMBus, NVDIMM, memory hotplug and
//! ACPI hotplug slots below bridges.

use super::BuildTables;
use super::aml::{
    self, AccessType, AddressSpace, Aml, Cacheable, Decode, IoDecode, IsaRanges, LockRule,
    MaxFixed, MinFixed, ReadWrite, RegionSpace, Serialize, UpdateRule,
};
use super::devices::{self, IsaDevice};
use super::linker::BiosLinker;
use super::pci::{self, CrsRange, CrsRangeSet};
use super::table::{self, AcpiTable, FadtData, Gas, McfgInfo, RsdpData, TABLE_FILE, fadt_flags};
use super::x86::{self, HPET_BASE, MadtConfig};
use super::{cpuhp, pcihp};

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

/// `HPET_LEN`.
pub const HPET_LEN: u32 = 0x400;

/// What a PCI device on the root bus adds to its `Device (Sxx)` object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PciDeviceAml {
    /// Nothing beyond `_ADR`.
    Plain,
    /// VGA with its sleep state methods. `qxl` keeps the display powered in S3.
    Vga { qxl: bool },
    /// The ICH9 LPC bridge with its ISA devices in bus order.
    Lpc { isa: Vec<IsaDevice> },
    /// A cold plugged PCI bridge or root port and the devices behind it.
    Bridge { devices: Vec<PciDevice> },
}

/// A device on the root bus.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PciDevice {
    pub devfn: u8,
    /// The `acpi-index` property, only for devices that cannot be hot unplugged. Hotpluggable
    /// devices get theirs from the hotplug `_DSM` instead.
    pub acpi_index: Option<u32>,
    pub aml: PciDeviceAml,
}

/// Everything about a q35 instance that shows up in its tables.
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
    /// `pcmc->pci_root_uid`.
    pub pci_root_uid: u32,
    /// ACPI PCI hotplug on bridges, the LPC `acpi-pci-hotplug-with-bridge-support` property.
    pub pcihp_bridge: bool,
    pub pcihp_io_base: u16,
    pub pcihp_io_len: u16,
    /// The firmware negotiated CPU hotplug and hot unplug over SMI.
    pub smi_on_cpuhp: bool,
    pub smi_on_cpu_unplug: bool,
    pub cpu_hp_io_base: u16,
    pub s3_disabled: bool,
    pub s4_disabled: bool,
    pub s4_val: u8,
    /// The 32 bit PCI hole.
    pub pci_hole: CrsRange,
    /// The 64 bit PCI hole, `None` when it is empty.
    pub pci_hole64: Option<CrsRange>,
    /// The devices on the root bus in devfn order.
    pub pci_devices: Vec<PciDevice>,
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

/// `build_dbg_aml()`: `DBUG` writes a hex dump of its argument to the debug port at 0x402.
fn build_dbg(table: &mut Aml) {
    let mut scope = aml::scope("\\");
    let buf = aml::local(0);
    let len = aml::local(1);
    let idx = aml::local(2);
    scope.append(&aml::operation_region("DBG", RegionSpace::SystemIo, &aml::int(0x0402), 0x01));
    let mut field = aml::field("DBG", AccessType::Byte, LockRule::NoLock, UpdateRule::Preserve);
    field.append(&aml::named_field("DBGB", 8));
    scope.append(&field);

    let mut method = aml::method("DBUG", 1, Serialize::NotSerialized);
    method.append(&aml::to_hexstring(&aml::arg(0), Some(&buf)));
    method.append(&aml::to_buffer(&buf, Some(&buf)));
    method.append(&aml::subtract(&aml::sizeof(&buf), &aml::int(1), Some(&len)));
    method.append(&aml::store(&aml::int(0), &idx));
    let mut while_ctx = aml::while_(&aml::lless(&idx, &len));
    while_ctx.append(&aml::store(&aml::derefof(&aml::index(&buf, &idx)), &aml::name("DBGB")));
    while_ctx.append(&aml::increment(&idx));
    method.append(&while_ctx);
    method.append(&aml::store(&aml::int(0x0A), &aml::name("DBGB")));
    scope.append(&method);
    table.append(&scope);
}

/// `build_q35_dram_controller()`, reserving the ECAM window.
fn dram_controller(mcfg: &McfgInfo) -> Aml {
    let mut dev = aml::device("DRAC");
    dev.append(&aml::name_decl("_HID", &aml::string("PNP0C01")));
    let mut crs = aml::resource_template();
    let (base, size) = (mcfg.base, mcfg.size);
    let last = base + size - 1;
    let (dec, min, max, nc, rw) = (
        Decode::Pos,
        MinFixed::Fixed,
        MaxFixed::Fixed,
        Cacheable::NonCacheable,
        ReadWrite::ReadWrite,
    );
    if last >= 1 << 32 {
        crs.append(&aml::qword_memory(dec, min, max, nc, rw, 0, base, last, 0, size));
    } else {
        crs.append(&aml::dword_memory(
            dec,
            min,
            max,
            nc,
            rw,
            0,
            base as u32,
            last as u32,
            0,
            size as u32,
        ));
    }
    dev.append(&aml::name_decl("_CRS", &crs));
    dev
}

/// `build_link_dev()`, one of the PIRQ links whose routing lives in LPC config space.
fn link_dev(name: &str, uid: u8, reg: &Aml) -> Aml {
    let mut dev = aml::device(name);
    dev.append(&aml::name_decl("_HID", &aml::eisaid("PNP0C0F")));
    dev.append(&aml::name_decl("_UID", &aml::int(uid.into())));
    let mut crs = aml::resource_template();
    crs.append(&level_high_shared(&[5, 10, 11]));
    dev.append(&aml::name_decl("_PRS", &crs));

    let mut method = aml::method("_STA", 0, Serialize::NotSerialized);
    method.append(&aml::return_(&aml::call("IQST", &[reg])));
    dev.append(&method);

    let mut method = aml::method("_DIS", 0, Serialize::NotSerialized);
    method.append(&aml::or(reg, &aml::int(0x80), Some(reg)));
    dev.append(&method);

    let mut method = aml::method("_CRS", 0, Serialize::NotSerialized);
    method.append(&aml::return_(&aml::call("IQCR", &[reg])));
    dev.append(&method);

    let mut method = aml::method("_SRS", 1, Serialize::NotSerialized);
    method.append(&aml::create_dword_field(&aml::arg(0), &aml::int(5), "PRRI"));
    method.append(&aml::store(&aml::name("PRRI"), reg));
    dev.append(&method);
    dev
}

fn level_high_shared(irqs: &[u32]) -> Aml {
    aml::interrupt(
        aml::ConsumerProducer::Consumer,
        aml::Trigger::Level,
        aml::Polarity::ActiveHigh,
        aml::Shared::Shared,
        irqs,
    )
}

/// `build_gsi_link_dev()`, a link hard wired to one GSI for APIC mode.
fn gsi_link_dev(name: &str, uid: u8, gsi: u8) -> Aml {
    let mut dev = aml::device(name);
    dev.append(&aml::name_decl("_HID", &aml::eisaid("PNP0C0F")));
    dev.append(&aml::name_decl("_UID", &aml::int(uid.into())));
    let mut crs = aml::resource_template();
    crs.append(&level_high_shared(&[gsi.into()]));
    dev.append(&aml::name_decl("_PRS", &crs));
    dev.append(&aml::name_decl("_CRS", &crs));
    // The interrupt cannot be disabled or moved, so these do nothing.
    dev.append(&aml::method("_DIS", 0, Serialize::NotSerialized));
    dev.append(&aml::method("_SRS", 1, Serialize::NotSerialized));
    dev
}

/// `build_iqcr_method()` for ICH9, where the low four bits of a PIRQ register are the IRQ.
fn iqcr_method() -> Aml {
    let mut method = aml::method("IQCR", 1, Serialize::Serialized);
    let mut crs = aml::resource_template();
    crs.append(&level_high_shared(&[0]));
    method.append(&aml::name_decl("PRR0", &crs));
    method.append(&aml::create_dword_field(&aml::name("PRR0"), &aml::int(5), "PRRI"));
    method.append(&aml::store(&aml::and(&aml::arg(0), &aml::int(0xF), None), &aml::name("PRRI")));
    method.append(&aml::return_(&aml::name("PRR0")));
    method
}

/// `build_irq_status_method()`: bit 7 of a PIRQ register means the link is disabled.
fn irq_status_method() -> Aml {
    let mut method = aml::method("IQST", 1, Serialize::NotSerialized);
    let mut if_ctx = aml::if_(&aml::and(&aml::int(0x80), &aml::arg(0), None));
    if_ctx.append(&aml::return_(&aml::int(0x09)));
    method.append(&if_ctx);
    method.append(&aml::return_(&aml::int(0x0B)));
    method
}

/// `append_q35_prt_entry()`: the four pins of slot `nr`, rotating through the links of the
/// A to D or E to H group starting at `name`.
fn append_prt_entry(ctx: &mut Aml, nr: u32, prefix: &str, first: u8) {
    let base = if first < b'E' { b'A' } else { b'E' };
    let a_nr = aml::int(u64::from(nr) << 16 | 0xffff);
    let mut head = i32::from(first - base);
    for i in 0..4i32 {
        if head + i > 3 {
            head = -i;
        }
        let link = format!("{prefix}{}", char::from(base + (head + i) as u8));
        let mut pkg = aml::package(4);
        pkg.append(&a_nr);
        pkg.append(&aml::int(i as u64));
        pkg.append(&aml::name(&link));
        pkg.append(&aml::int(0));
        ctx.append(&pkg);
    }
}

/// `build_q35_routing_table()`: slots 0 to 0x17 rotate through PIRQ E to H, 0x18 and the
/// PCIe to PCI bridge at 0x1e start at E, and the rest follow the D<N>IR defaults.
fn routing_table(prefix: &str) -> Aml {
    let mut pkg = aml::package(128);
    for i in 0..0x18u32 {
        append_prt_entry(&mut pkg, i, prefix, b'E' + (i & 3) as u8);
    }
    append_prt_entry(&mut pkg, 0x18, prefix, b'E');
    for i in 0x19..0x1e {
        append_prt_entry(&mut pkg, i, prefix, b'A');
    }
    append_prt_entry(&mut pkg, 0x1e, prefix, b'E');
    append_prt_entry(&mut pkg, 0x1f, prefix, b'A');
    pkg
}

/// `build_q35_pci0_int()`.
fn build_pci0_int(table: &mut Aml) {
    // 0 is PIC mode, 1 is APIC mode.
    table.append(&aml::name_decl("PICF", &aml::int(0)));
    let mut method = aml::method("_PIC", 1, Serialize::NotSerialized);
    method.append(&aml::store(&aml::arg(0), &aml::name("PICF")));
    table.append(&method);

    let mut sb_scope = aml::scope("_SB");
    let mut pci0_scope = aml::scope("PCI0");
    pci0_scope.append(&aml::name_decl("PRTP", &routing_table("LNK")));
    pci0_scope.append(&aml::name_decl("PRTA", &routing_table("GSI")));
    let mut method = aml::method("_PRT", 0, Serialize::NotSerialized);
    let mut if_ctx = aml::if_(&aml::equal(&aml::name("PICF"), &aml::int(0)));
    if_ctx.append(&aml::return_(&aml::name("PRTP")));
    method.append(&if_ctx);
    let mut else_ctx = aml::else_();
    else_ctx.append(&aml::return_(&aml::name("PRTA")));
    method.append(&else_ctx);
    pci0_scope.append(&method);
    sb_scope.append(&pci0_scope);

    sb_scope.append(&irq_status_method());
    sb_scope.append(&iqcr_method());
    for (i, l) in (b'A'..=b'H').enumerate() {
        let l = char::from(l);
        sb_scope.append(&link_dev(&format!("LNK{l}"), i as u8, &aml::name(&format!("PRQ{l}"))));
    }
    for (i, l) in (b'A'..=b'H').enumerate() {
        let gsi = 0x10 + i as u8;
        sb_scope.append(&gsi_link_dev(&format!("GSI{}", char::from(l)), gsi, gsi));
    }
    table.append(&sb_scope);
}

/// `build_hpet_aml()`. `_STA` reports the HPET absent when its registers read back nonsense.
fn build_hpet_aml(table: &mut Aml) {
    let zero = aml::int(0);
    let id = aml::local(0);
    let period = aml::local(1);
    let mut scope = aml::scope("_SB");
    let mut dev = aml::device("HPET");
    dev.append(&aml::name_decl("_HID", &aml::eisaid("PNP0103")));
    dev.append(&aml::name_decl("_UID", &zero));
    dev.append(&aml::operation_region(
        "HPTM",
        RegionSpace::SystemMemory,
        &aml::int(HPET_BASE),
        HPET_LEN,
    ));
    let mut field = aml::field("HPTM", AccessType::Dword, LockRule::Lock, UpdateRule::Preserve);
    field.append(&aml::named_field("VEND", 32));
    field.append(&aml::named_field("PRD", 32));
    dev.append(&field);

    let mut method = aml::method("_STA", 0, Serialize::NotSerialized);
    method.append(&aml::store(&aml::name("VEND"), &id));
    method.append(&aml::store(&aml::name("PRD"), &period));
    method.append(&aml::shiftright(&id, &aml::int(16), Some(&id)));
    let mut if_ctx =
        aml::if_(&aml::lor(&aml::equal(&id, &zero), &aml::equal(&id, &aml::int(0xffff))));
    if_ctx.append(&aml::return_(&zero));
    method.append(&if_ctx);
    let mut if_ctx = aml::if_(&aml::lor(
        &aml::equal(&period, &zero),
        &aml::lgreater(&period, &aml::int(100_000_000)),
    ));
    if_ctx.append(&aml::return_(&zero));
    method.append(&if_ctx);
    method.append(&aml::return_(&aml::int(0x0F)));
    dev.append(&method);

    let mut crs = aml::resource_template();
    crs.append(&aml::memory32_fixed(HPET_BASE as u32, HPET_LEN, ReadWrite::ReadOnly));
    dev.append(&aml::name_decl("_CRS", &crs));
    scope.append(&dev);
    table.append(&scope);
}

/// `build_ich9_isa_aml()`: the PIRQ routing registers, then the ISA devices.
fn lpc_aml(scope: &mut Aml, isa: &[IsaDevice]) {
    scope.append(&aml::operation_region("PIRQ", RegionSpace::PciConfig, &aml::int(0x60), 0x0C));
    let mut sb_scope = aml::scope("\\_SB");
    let mut field =
        aml::field("PCI0.SF8.PIRQ", AccessType::Byte, LockRule::NoLock, UpdateRule::Preserve);
    for l in ["PRQA", "PRQB", "PRQC", "PRQD"] {
        field.append(&aml::named_field(l, 8));
    }
    field.append(&aml::reserved_field(0x20));
    for l in ["PRQE", "PRQF", "PRQG", "PRQH"] {
        field.append(&aml::named_field(l, 8));
    }
    sb_scope.append(&field);
    scope.append(&sb_scope);
    for dev in isa {
        dev.build_aml(scope);
    }
}

/// `build_append_pci_bus_devices()` for the root bus.
fn pci_bus_devices(scope: &mut Aml, devices: &[PciDevice]) {
    for d in devices {
        let adr = u64::from(d.devfn >> 3) << 16 | u64::from(d.devfn & 7);
        let mut dev = aml::device(&format!("S{:02X}", d.devfn));
        dev.append(&aml::name_decl("_ADR", &aml::int(adr)));
        match &d.aml {
            PciDeviceAml::Plain => {}
            PciDeviceAml::Vga { qxl } => devices::vga_pci(&mut dev, *qxl),
            PciDeviceAml::Lpc { isa } => lpc_aml(&mut dev, isa),
            PciDeviceAml::Bridge { devices } => pci_bus_devices(&mut dev, devices),
        }
        if let Some(index) = d.acpi_index {
            dev.append(&pcihp::static_endpoint_dsm(index));
        }
        scope.append(&dev);
    }
}

/// `PCI0._CRS`: the bus numbers, the config ports, all I/O and memory not taken by expander
/// buses or the ECAM window, and the legacy VGA window.
fn pci0_crs(q: &Q35Acpi, ranges: &mut CrsRangeSet) -> Aml {
    let root_bus_limit: u16 = 0xFF;
    let (min, max, dec) = (MinFixed::Fixed, MaxFixed::Fixed, Decode::Pos);
    let mut crs = aml::resource_template();
    crs.append(&aml::word_bus_number(min, max, dec, 0, 0, root_bus_limit, 0, root_bus_limit + 1));
    crs.append(&aml::io(IoDecode::Decode16, 0x0CF8, 0x0CF8, 0x01, 0x08));
    crs.append(&aml::word_io(min, max, dec, IsaRanges::EntireRange, 0, 0, 0x0CF7, 0, 0x0CF8));

    pci::crs_replace_with_free_ranges(&mut ranges.io_ranges, 0x0D00, 0xFFFF);
    for r in &ranges.io_ranges {
        let (base, limit) = (r.base as u16, r.limit as u16);
        crs.append(&aml::word_io(
            min,
            max,
            dec,
            IsaRanges::EntireRange,
            0,
            base,
            limit,
            0,
            limit.wrapping_sub(base).wrapping_add(1),
        ));
    }

    let rw = ReadWrite::ReadWrite;
    crs.append(&aml::dword_memory(
        dec,
        min,
        max,
        Cacheable::Cacheable,
        rw,
        0,
        0x000A_0000,
        0x000B_FFFF,
        0,
        0x0002_0000,
    ));
    pci::crs_replace_with_free_ranges(&mut ranges.mem_ranges, q.pci_hole.base, q.pci_hole.limit);
    for r in &ranges.mem_ranges {
        let (base, limit) = (r.base as u32, r.limit as u32);
        crs.append(&aml::dword_memory(
            dec,
            min,
            max,
            Cacheable::NonCacheable,
            rw,
            0,
            base,
            limit,
            0,
            limit.wrapping_sub(base).wrapping_add(1),
        ));
    }
    if let Some(hole64) = q.pci_hole64 {
        pci::crs_replace_with_free_ranges(&mut ranges.mem_64bit_ranges, hole64.base, hole64.limit);
        for r in &ranges.mem_64bit_ranges {
            crs.append(&aml::qword_memory(
                dec,
                min,
                max,
                Cacheable::Cacheable,
                rw,
                0,
                r.base,
                r.limit,
                0,
                r.limit - r.base + 1,
            ));
        }
    }
    crs
}

/// `build_dsdt()` for q35.
fn build_dsdt(table_data: &mut Vec<u8>, linker: &mut BiosLinker, q: &Q35Acpi) {
    let table = AcpiTable::begin("DSDT", 1, &q.oem_id, &q.oem_table_id, table_data);
    let mut dsdt = Aml::new();
    let fadt = q.fadt();

    build_dbg(&mut dsdt);

    let mut sb_scope = aml::scope("_SB");
    let mut dev = aml::device("PCI0");
    dev.append(&aml::name_decl("_HID", &aml::eisaid("PNP0A08")));
    dev.append(&aml::name_decl("_CID", &aml::eisaid("PNP0A03")));
    dev.append(&aml::name_decl("_UID", &aml::int(q.pci_root_uid.into())));
    dev.append(&pci::host_bridge_osc(!q.pcihp_bridge));
    dev.append(&pci::bridge_edsm());
    sb_scope.append(&dev);
    if let Some(mcfg) = &q.mcfg {
        sb_scope.append(&dram_controller(mcfg));
    }
    if q.smi_on_cpuhp {
        // Reserve the SMI ports 0xB2 and 0xB3.
        let smi = fadt.smi_cmd as u16;
        let mut dev = aml::device("PCI0.SMI0");
        dev.append(&aml::name_decl("_HID", &aml::eisaid("PNP0A06")));
        dev.append(&aml::name_decl("_UID", &aml::string("SMI resources")));
        let mut crs = aml::resource_template();
        crs.append(&aml::io(IoDecode::Decode16, smi, smi, 1, 2));
        dev.append(&aml::name_decl("_CRS", &crs));
        dev.append(&aml::operation_region("SMIR", RegionSpace::SystemIo, &aml::int(smi.into()), 2));
        let mut field =
            aml::field("SMIR", AccessType::Byte, LockRule::NoLock, UpdateRule::WriteAsZeros);
        field.append(&aml::named_field("SMIC", 8));
        field.append(&aml::reserved_field(8));
        dev.append(&field);
        sb_scope.append(&dev);
    }
    dsdt.append(&sb_scope);

    if q.pcihp_bridge {
        pcihp::build_hotplug(&mut dsdt, RegionSpace::SystemIo, q.pcihp_io_base.into());
    }
    build_pci0_int(&mut dsdt);

    if q.hpet {
        build_hpet_aml(&mut dsdt);
    }

    let mut scope = aml::scope("_GPE");
    scope.append(&aml::name_decl("_HID", &aml::string("ACPI0006")));
    dsdt.append(&scope);

    let cpus_opts = cpuhp::Features {
        acpi_1_compatible: true,
        smi_path: q.smi_on_cpuhp.then(|| "\\_SB.PCI0.SMI0.SMIC".to_string()),
        fw_unplugs_cpu: q.smi_on_cpu_unplug,
    };
    cpuhp::build_cpus(
        &mut dsdt,
        &q.madt.cpus,
        &cpus_opts,
        q.cpu_hp_io_base.into(),
        "\\_SB.PCI0",
        "\\_GPE._E02",
        RegionSpace::SystemIo,
    );

    // Leave the ECAM window out of PCI0._CRS.
    let mut ranges = CrsRangeSet::default();
    if let Some(mcfg) = &q.mcfg {
        pci::crs_range_insert(&mut ranges.mem_ranges, mcfg.base, mcfg.base + mcfg.size - 1);
    }
    let mut scope = aml::scope("\\_SB.PCI0");
    scope.append(&aml::name_decl("_CRS", &pci0_crs(q, &mut ranges)));

    // Reserve the GPE0 block.
    let mut dev = aml::device("GPE0");
    dev.append(&aml::name_decl("_HID", &aml::string("PNP0A06")));
    dev.append(&aml::name_decl("_UID", &aml::string("GPE0 resources")));
    // Present, functioning, decoding, not shown in UI.
    dev.append(&aml::name_decl("_STA", &aml::int(0xB)));
    let mut crs = aml::resource_template();
    let gpe0 = fadt.gpe0_blk.address as u16;
    crs.append(&aml::io(IoDecode::Decode16, gpe0, gpe0, 1, fadt.gpe0_blk.bit_width / 8));
    dev.append(&aml::name_decl("_CRS", &crs));
    scope.append(&dev);
    if q.pcihp_io_len != 0 && q.pcihp_bridge {
        pcihp::build_resources(&mut scope, q.pcihp_io_base, q.pcihp_io_len);
    }
    dsdt.append(&scope);

    // The sleep state packages: PM1a_CNT.SLP_TYP, PM1b_CNT.SLP_TYP and two reserved.
    let mut scope = aml::scope("\\");
    let sleep_pkg = |typ: u64| {
        let mut pkg = aml::package(4);
        for v in [typ, typ, 0, 0] {
            pkg.append(&aml::int(v));
        }
        pkg
    };
    if !q.s3_disabled {
        scope.append(&aml::name_decl("_S3", &sleep_pkg(1)));
    }
    if !q.s4_disabled {
        scope.append(&aml::name_decl("_S4", &sleep_pkg(q.s4_val.into())));
    }
    scope.append(&aml::name_decl("_S5", &sleep_pkg(0)));
    dsdt.append(&scope);

    let mut scope = aml::scope("\\_SB.PCI0");
    devices::fw_cfg_x86(&mut scope);
    dsdt.append(&scope);

    let mut sb_scope = aml::scope("\\_SB");
    let mut pci0 = aml::scope("PCI0");
    pci_bus_devices(&mut pci0, &q.pci_devices);
    sb_scope.append(&pci0);
    dsdt.append(&sb_scope);

    if q.pcihp_bridge {
        // No bridges yet, so there is no PCNT to call and the handler is empty.
        let mut scope = aml::scope("_GPE");
        scope.append(&aml::method("_E01", 0, Serialize::NotSerialized));
        dsdt.append(&scope);
    }

    table_data.extend_from_slice(dsdt.as_bytes());
    table.end(Some(linker), table_data);
}

/// `acpi_build()` for q35.
pub fn build(q: &Q35Acpi) -> BuildTables {
    let mut t = BuildTables::new();
    let data = &mut t.table_data;
    let linker = &mut t.linker;
    linker.alloc(TABLE_FILE, 64, false);

    // The FACS goes first because it is the only table with an alignment requirement.
    let facs = data.len() as u32;
    x86::build_facs(data);

    let dsdt_off = data.len() as u32;
    build_dsdt(data, linker, q);

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
