// SPDX-License-Identifier: GPL-2.0-or-later

//! The tables of the Arm `virt` board, hw/arm/virt-acpi-build.c: DSDT, FADT, MADT, PPTT,
//! GTDT, MCFG, SPCR, DBG2, SRAT and SLIT with NUMA nodes, and IORT, behind an XSDT and a
//! revision 2 RSDP.
//!
//! The board is hardware reduced ACPI. The DSDT has the CPUs, the PL011 UARTs, fw_cfg, the
//! virtio-mmio transports, the PCIe host bridge, then the GED (with firmware) or the PL061
//! GPIO controller that raises the power button, the power button, the error device and the
//! functions on the root bus. The MADT describes a GICv3: the distributor, a GICC per CPU,
//! the redistributor regions and the ITS. The IORT has the ITS group and the root complex,
//! whose requester IDs all go to the ITS.
//!
//! Not built: the GICv2 and GICv2m forms of the MADT and the IORT, the SMMUv3 nodes, the
//! cache nodes of the PPTT (`smp-cache`), HEST (`ras=on`), HMAT, the GWDT and WDAT, CEDT,
//! NFIT, TPM2 and VIOT, and the memory hotplug and ACPI PCI hotplug AML, since ruvm has none
//! of those devices.

use super::BuildTables;
use super::aml::{self, AddressSpace, Aml, append_int_noprefix};
use super::devices;
use super::gpex::{self, GpexAcpi, Window};
use super::linker::BiosLinker;
use super::pci;
use super::q35::{self, PciDevice};
use super::table::{
    self, AcpiTable, FadtData, Gas, McfgInfo, RsdpData, SpcrData, TABLE_FILE, fadt_flags,
};

pub use super::riscv_virt::NumaNode;

/// `ACPI_BUILD_TABLE_SIZE`: `etc/acpi/tables` is padded to a multiple of this.
pub const ACPI_BUILD_TABLE_SIZE: usize = 0x20000;

/// `ARCH_TIMER_S_EL1_IRQ`.
pub const ARCH_TIMER_S_EL1_IRQ: u32 = 29;
/// `ARCH_TIMER_NS_EL1_IRQ`.
pub const ARCH_TIMER_NS_EL1_IRQ: u32 = 30;
/// `ARCH_TIMER_VIRT_IRQ`.
pub const ARCH_TIMER_VIRT_IRQ: u32 = 27;
/// `ARCH_TIMER_NS_EL2_IRQ`.
pub const ARCH_TIMER_NS_EL2_IRQ: u32 = 26;
/// `ARCH_TIMER_NS_EL2_VIRT_IRQ`.
pub const ARCH_TIMER_NS_EL2_VIRT_IRQ: u32 = 28;
/// `ARCH_GIC_MAINT_IRQ`, the GIC maintenance interrupt with `virtualization=on`.
pub const ARCH_GIC_MAINT_IRQ: u32 = 25;
/// `VIRTUAL_PMU_IRQ`, the PMU interrupt of CPUs that have a PMU.
pub const VIRTUAL_PMU_IRQ: u32 = 23;

/// `GPIO_PIN_POWER_BUTTON`.
const GPIO_PIN_POWER_BUTTON: u16 = 3;
/// `IORT_NODE_OFFSET`.
const IORT_NODE_OFFSET: u32 = 48;
/// `ROOT_COMPLEX_ENTRY_SIZE`.
const ROOT_COMPLEX_ENTRY_SIZE: u32 = 36;
/// `ID_MAPPING_ENTRY_SIZE`.
const ID_MAPPING_ENTRY_SIZE: u32 = 20;

/// `ACPI_FADT_ARM_PSCI_COMPLIANT`.
const ACPI_FADT_ARM_PSCI_COMPLIANT: u16 = 1 << 0;
/// `ACPI_FADT_ARM_PSCI_USE_HVC`.
const ACPI_FADT_ARM_PSCI_USE_HVC: u16 = 1 << 1;

/// The board's PSCI conduit, `vms->psci_conduit`, as the FADT reports it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PsciConduit {
    /// No PSCI from the board: the firmware provides it.
    Disabled,
    #[default]
    Hvc,
    Smc,
}

/// The parts of `vms->memmap[]` the tables use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VirtMemmap {
    pub uart0: Window,
    /// The second non-secure UART, `second_ns_uart_present`.
    pub uart1: Option<Window>,
    pub fw_cfg: Window,
    /// The first virtio-mmio transport.
    pub virtio: Window,
    /// The ECAM window in use, low or high.
    pub ecam: Window,
    pub pcie_mmio: Window,
    pub pcie_pio: Window,
    /// The high PCIe MMIO window, size 0 without one.
    pub pcie_mmio_high: Window,
    pub gic_dist: u64,
    pub gic_redist: Window,
    /// The high redistributor region, when the CPUs do not fit in the first one.
    pub gic_redist2: Option<Window>,
    /// The ITS, `msi_controller == VIRT_MSI_CTRL_ITS`.
    pub gic_its: Option<u64>,
    pub acpi_ged: u64,
    pub gpio: Window,
    pub mem: u64,
}

/// The interrupts of the devices, as GSIs (SPI number plus 32).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VirtIrqs {
    pub uart0: u32,
    pub uart1: u32,
    /// The first virtio-mmio transport.
    pub virtio: u32,
    /// INTA of the PCIe host bridge.
    pub pcie: u32,
    pub acpi_ged: u32,
    pub gpio: u32,
}

/// One possible CPU, an entry of `possible_cpu_arch_ids()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PossibleCpu {
    pub socket: u32,
    pub cluster: u32,
    pub core: u32,
    pub thread: u32,
    /// `props.node_id`.
    pub node: u32,
}

/// Everything about a `virt` instance that shows up in its tables.
#[derive(Clone, Debug, Default)]
pub struct ArmVirtAcpi {
    pub oem_id: String,
    pub oem_table_id: String,
    pub memmap: VirtMemmap,
    pub irqs: VirtIrqs,
    /// `virtio_transports`.
    pub virtio_count: u32,
    /// The MPIDR of each present CPU, `smp.cpus` of them.
    pub mpidrs: Vec<u64>,
    /// The possible CPUs, `smp.max_cpus` of them, in topology order.
    pub possible_cpus: Vec<PossibleCpu>,
    /// `smp_props.has_clusters`: `-smp` named the clusters.
    pub has_clusters: bool,
    /// `smp.threads`.
    pub threads: u32,
    /// The PMU interrupt of the CPUs, 0 without a PMU.
    pub pmu_irq: u32,
    /// `vms->virt`: the GIC maintenance interrupt is described.
    pub virtualization: bool,
    /// `ns_el2_virt_timer_irq`.
    pub ns_el2_virt_timer: bool,
    pub psci: PsciConduit,
    /// The `ged-event` bitmap of the GED, which exists when firmware boots with ACPI. Without
    /// it the DSDT has the GPIO power button instead.
    pub ged_events: Option<u32>,
    /// `ms->acpi_spcr_enabled`.
    pub spcr: bool,
    /// The functions on the PCIe root bus.
    pub pci_devices: Vec<PciDevice>,
    pub numa: Vec<NumaNode>,
}

/// `acpi_dsdt_add_cpus()`.
fn dsdt_add_cpus(scope: &mut Aml, s: &ArmVirtAcpi) {
    for i in 0..s.mpidrs.len() {
        let mut dev = aml::device(&format!("C{i:03X}"));
        dev.append(&aml::name_decl("_HID", &aml::string("ACPI0007")));
        dev.append(&aml::name_decl("_UID", &aml::int(i as u64)));
        scope.append(&dev);
    }
}

/// `acpi_dsdt_add_uart()`.
fn dsdt_add_uart(scope: &mut Aml, uart: Window, uart_irq: u32, uartidx: u32) {
    let mut dev = aml::device(&format!("COM{uartidx}"));
    dev.append(&aml::name_decl("_HID", &aml::string("ARMH0011")));
    dev.append(&aml::name_decl("_UID", &aml::int(uartidx.into())));

    let mut crs = aml::resource_template();
    crs.append(&aml::memory32_fixed(uart.base as u32, uart.size as u32, aml::ReadWrite::ReadWrite));
    crs.append(&aml::interrupt(
        aml::ConsumerProducer::Consumer,
        aml::Trigger::Level,
        aml::Polarity::ActiveHigh,
        aml::Shared::Exclusive,
        &[uart_irq],
    ));
    dev.append(&aml::name_decl("_CRS", &crs));
    scope.append(&dev);
}

/// `acpi_dsdt_add_gpio()`: the PL061 with the power button on pin 3.
fn dsdt_add_gpio(scope: &mut Aml, gpio: Window, gpio_irq: u32) {
    let mut dev = aml::device("GPO0");
    dev.append(&aml::name_decl("_HID", &aml::string("ARMH0061")));
    dev.append(&aml::name_decl("_UID", &aml::int(0)));

    let mut crs = aml::resource_template();
    crs.append(&aml::memory32_fixed(gpio.base as u32, gpio.size as u32, aml::ReadWrite::ReadWrite));
    crs.append(&aml::interrupt(
        aml::ConsumerProducer::Consumer,
        aml::Trigger::Level,
        aml::Polarity::ActiveHigh,
        aml::Shared::Exclusive,
        &[gpio_irq],
    ));
    dev.append(&aml::name_decl("_CRS", &crs));

    let mut aei = aml::resource_template();
    aei.append(&aml::gpio_int(
        aml::ConsumerProducer::Consumer,
        aml::Trigger::Edge,
        aml::Polarity::ActiveHigh,
        aml::Shared::Exclusive,
        aml::PinConfig::PullUp,
        0,
        &[GPIO_PIN_POWER_BUTTON],
        "GPO0",
        &[],
    ));
    dev.append(&aml::name_decl("_AEI", &aei));

    // _E03 is handle for power button
    let mut method = aml::method("_E03", 0, aml::Serialize::NotSerialized);
    method.append(&aml::notify(&aml::name(devices::POWER_BUTTON_DEVICE), &aml::int(0x80)));
    dev.append(&method);
    scope.append(&dev);
}

/// `build_dsdt()`.
fn build_dsdt(tbl: &mut Vec<u8>, linker: &mut BiosLinker, s: &ArmVirtAcpi) {
    let table = AcpiTable::begin("DSDT", 2, &s.oem_id, &s.oem_table_id, tbl);
    let mut dsdt = Aml::new();
    let memmap = &s.memmap;

    // When booting the VM with UEFI, UEFI takes ownership of the RTC hardware. While UEFI
    // can use libfdt to disable the RTC device node in the DTB that it passes to the OS, it
    // cannot modify AML. Therefore, we won't generate the RTC ACPI device at all when using
    // UEFI.
    let mut scope = aml::scope("\\_SB");
    dsdt_add_cpus(&mut scope, s);
    dsdt_add_uart(&mut scope, memmap.uart0, s.irqs.uart0, 0);
    if let Some(uart1) = memmap.uart1 {
        dsdt_add_uart(&mut scope, uart1, s.irqs.uart1, 1);
    }
    devices::fw_cfg_mmio(&mut scope, memmap.fw_cfg.base, memmap.fw_cfg.size);
    devices::virtio_mmio(
        &mut scope,
        memmap.virtio.base,
        memmap.virtio.size,
        s.irqs.virtio,
        0,
        s.virtio_count,
    );
    let gpex = GpexAcpi {
        ecam: memmap.ecam,
        mmio32: memmap.pcie_mmio,
        mmio64: memmap.pcie_mmio_high,
        pio: memmap.pcie_pio,
        irq: s.irqs.pcie,
        // ACPI PCI hotplug is off, so native PCIe hotplug is offered.
        pci_native_hotplug: true,
        preserve_config: false,
    };
    gpex::add_gpex(&mut scope, &gpex);
    match s.ged_events {
        Some(events) => devices::ged(
            &mut scope,
            "\\_SB.GED",
            events,
            s.irqs.acpi_ged,
            aml::RegionSpace::SystemMemory,
            memmap.acpi_ged,
        ),
        None => dsdt_add_gpio(&mut scope, memmap.gpio, s.irqs.gpio),
    }
    devices::power_button(&mut scope);
    devices::error_device(&mut scope);
    dsdt.append(&scope);

    let mut pci0_scope = aml::scope("\\_SB.PCI0");
    pci0_scope.append(&pci::bridge_edsm());
    q35::pci_bus_devices(&mut pci0_scope, &s.pci_devices);
    dsdt.append(&pci0_scope);

    tbl.extend_from_slice(dsdt.as_bytes());
    table.end(Some(linker), tbl);
}

/// `build_fadt_rev6()`.
fn build_fadt_rev6(tbl: &mut Vec<u8>, linker: &mut BiosLinker, s: &ArmVirtAcpi, dsdt: u32) {
    // ACPI v6.3
    let fadt = FadtData {
        rev: 6,
        minor_ver: 3,
        flags: 1 << fadt_flags::HW_REDUCED_ACPI,
        xdsdt_tbl_offset: Some(dsdt),
        arm_boot_arch: match s.psci {
            PsciConduit::Disabled => 0,
            PsciConduit::Hvc => ACPI_FADT_ARM_PSCI_COMPLIANT | ACPI_FADT_ARM_PSCI_USE_HVC,
            PsciConduit::Smc => ACPI_FADT_ARM_PSCI_COMPLIANT,
        },
        ..FadtData::default()
    };
    table::build_fadt(tbl, linker, &fadt, &s.oem_id, &s.oem_table_id);
}

/// `build_append_gicr()`.
fn append_gicr(tbl: &mut Vec<u8>, w: Window) {
    tbl.push(0xE); // Type
    tbl.push(16); // Length
    append_int_noprefix(tbl, 0, 2); // Reserved
    append_int_noprefix(tbl, w.base, 8); // Discovery Range Base Address
    append_int_noprefix(tbl, w.size, 4); // Discovery Range Length
}

/// `build_madt()` for a GICv3 (ACPI 6.0 Errata A, 5.2.12).
fn build_madt(tbl: &mut Vec<u8>, linker: &mut BiosLinker, s: &ArmVirtAcpi) {
    let memmap = &s.memmap;
    let table = AcpiTable::begin("APIC", 4, &s.oem_id, &s.oem_table_id, tbl);
    append_int_noprefix(tbl, 0, 4); // Local Interrupt Controller Address
    append_int_noprefix(tbl, 0, 4); // Flags

    // 5.2.12.15 GIC Distributor Structure
    tbl.push(0xC); // Type
    tbl.push(24); // Length
    append_int_noprefix(tbl, 0, 2); // Reserved
    append_int_noprefix(tbl, 0, 4); // GIC ID
    append_int_noprefix(tbl, memmap.gic_dist, 8); // Physical Base Address
    append_int_noprefix(tbl, 0, 4); // System Vector Base
    tbl.push(3); // GIC version
    append_int_noprefix(tbl, 0, 3); // Reserved

    let vgic_interrupt = if s.virtualization { ARCH_GIC_MAINT_IRQ } else { 0 };
    for (i, &mpidr) in s.mpidrs.iter().enumerate() {
        // 5.2.12.14 GIC Structure
        tbl.push(0xB); // Type
        tbl.push(80); // Length
        append_int_noprefix(tbl, 0, 2); // Reserved
        append_int_noprefix(tbl, i as u64, 4); // GIC ID
        append_int_noprefix(tbl, i as u64, 4); // ACPI Processor UID
        append_int_noprefix(tbl, 1, 4); // Flags: Enabled
        append_int_noprefix(tbl, 0, 4); // Parking Protocol Version
        append_int_noprefix(tbl, s.pmu_irq.into(), 4); // Performance Interrupt GSIV
        append_int_noprefix(tbl, 0, 8); // Parked Address
        append_int_noprefix(tbl, 0, 8); // Physical Base Address
        append_int_noprefix(tbl, 0, 8); // GICV
        append_int_noprefix(tbl, 0, 8); // GICH
        append_int_noprefix(tbl, vgic_interrupt.into(), 4); // VGIC Maintenance interrupt
        append_int_noprefix(tbl, 0, 8); // GICR Base Address
        append_int_noprefix(tbl, mpidr, 8); // MPIDR
        tbl.push(0); // Processor Power Efficiency Class
        append_int_noprefix(tbl, 0, 3); // Reserved
    }

    append_gicr(tbl, memmap.gic_redist);
    if let Some(redist2) = memmap.gic_redist2 {
        append_gicr(tbl, redist2);
    }
    if let Some(its) = memmap.gic_its {
        // ACPI spec, Revision 6.0 Errata A (original 6.0 definition has invalid Length)
        // 5.2.12.18 GIC ITS Structure
        tbl.push(0xF); // Type
        tbl.push(20); // Length
        append_int_noprefix(tbl, 0, 2); // Reserved
        append_int_noprefix(tbl, 0, 4); // GIC ITS ID
        append_int_noprefix(tbl, its, 8); // Physical Base Address
        append_int_noprefix(tbl, 0, 4); // Reserved
    }
    table.end(Some(linker), tbl);
}

/// `build_processor_hierarchy_node()` without private resources.
fn append_processor_hierarchy_node(tbl: &mut Vec<u8>, flags: u32, parent: u32, id: u32) {
    tbl.push(0); // Type 0 - processor
    tbl.push(20); // Length
    append_int_noprefix(tbl, 0, 2); // Reserved
    append_int_noprefix(tbl, flags.into(), 4); // Flags
    append_int_noprefix(tbl, parent.into(), 4); // Parent
    append_int_noprefix(tbl, id.into(), 4); // ACPI Processor ID
    append_int_noprefix(tbl, 0, 4); // Number of private resources
}

/// `build_pptt()` without cache nodes (ACPI 6.3, 5.2.29).
fn build_pptt(tbl: &mut Vec<u8>, linker: &mut BiosLinker, s: &ArmVirtAcpi) {
    const PHYSICAL_PACKAGE: u32 = 1 << 0;
    const ACPI_ID_VALID: u32 = 1 << 1;
    const THREAD: u32 = 1 << 2;
    const LEAF: u32 = 1 << 3;
    const IDENTICAL: u32 = 1 << 4;

    let table = AcpiTable::begin("PPTT", 2, &s.oem_id, &s.oem_table_id, tbl);
    let pptt_start = table.offset();
    let rel = |tbl: &Vec<u8>| (tbl.len() - pptt_start) as u32;

    // A root node for all the processor nodes, so that an OS can tell that a multi-socket
    // system is homogeneous.
    let root_offset = rel(tbl);
    append_processor_hierarchy_node(tbl, PHYSICAL_PACKAGE | IDENTICAL, 0, 0);

    let (mut socket_id, mut cluster_id, mut core_id) = (None, None, None);
    let (mut socket_offset, mut cluster_offset, mut core_offset) = (0, 0, 0);
    for (n, cpu) in s.possible_cpus.iter().enumerate() {
        if socket_id != Some(cpu.socket) {
            socket_id = Some(cpu.socket);
            cluster_id = None;
            core_id = None;
            socket_offset = rel(tbl);
            append_processor_hierarchy_node(
                tbl,
                PHYSICAL_PACKAGE | IDENTICAL,
                root_offset,
                cpu.socket,
            );
        }
        if s.has_clusters {
            if cluster_id != Some(cpu.cluster) {
                cluster_id = Some(cpu.cluster);
                core_id = None;
                cluster_offset = rel(tbl);
                append_processor_hierarchy_node(tbl, IDENTICAL, socket_offset, cpu.cluster);
            }
        } else {
            cluster_offset = socket_offset;
        }
        if s.threads == 1 {
            append_processor_hierarchy_node(tbl, ACPI_ID_VALID | LEAF, cluster_offset, n as u32);
        } else {
            if core_id != Some(cpu.core) {
                core_id = Some(cpu.core);
                core_offset = rel(tbl);
                append_processor_hierarchy_node(tbl, IDENTICAL, cluster_offset, cpu.core);
            }
            append_processor_hierarchy_node(
                tbl,
                ACPI_ID_VALID | THREAD | LEAF,
                core_offset,
                n as u32,
            );
        }
    }
    table.end(Some(linker), tbl);
}

/// `build_gtdt()` without the watchdog (ACPI 6.5, 5.2.25).
fn build_gtdt(tbl: &mut Vec<u8>, linker: &mut BiosLinker, s: &ArmVirtAcpi) {
    // Only the "Timer interrupt Mode" flag, with the polarity bit as 0: active high.
    let irqflags: u64 = 0; // Interrupt is Level triggered
    let table = AcpiTable::begin("GTDT", 3, &s.oem_id, &s.oem_table_id, tbl);
    append_int_noprefix(tbl, u64::MAX, 8); // CntControlBase Physical Address
    append_int_noprefix(tbl, 0, 4); // Reserved
    append_int_noprefix(tbl, ARCH_TIMER_S_EL1_IRQ.into(), 4); // Secure EL1 timer GSIV
    append_int_noprefix(tbl, irqflags, 4); // Secure EL1 timer Flags
    append_int_noprefix(tbl, ARCH_TIMER_NS_EL1_IRQ.into(), 4); // Non-Secure EL1 timer GSIV
    // Non-Secure EL1 timer Flags, with the Always-on Capability
    append_int_noprefix(tbl, irqflags | 1 << 2, 4);
    append_int_noprefix(tbl, ARCH_TIMER_VIRT_IRQ.into(), 4); // Virtual timer GSIV
    append_int_noprefix(tbl, irqflags, 4); // Virtual Timer Flags
    append_int_noprefix(tbl, ARCH_TIMER_NS_EL2_IRQ.into(), 4); // Non-Secure EL2 timer GSIV
    append_int_noprefix(tbl, irqflags, 4); // Non-Secure EL2 timer Flags
    append_int_noprefix(tbl, u64::MAX, 8); // CntReadBase Physical address
    append_int_noprefix(tbl, 0, 4); // Platform Timer Count
    append_int_noprefix(tbl, 0, 4); // Platform Timer Offset
    if s.ns_el2_virt_timer {
        append_int_noprefix(tbl, ARCH_TIMER_NS_EL2_VIRT_IRQ.into(), 4); // Virtual EL2 Timer GSIV
        append_int_noprefix(tbl, irqflags, 4); // Virtual EL2 Timer Flags
    } else {
        append_int_noprefix(tbl, 0, 4);
        append_int_noprefix(tbl, 0, 4);
    }
    table.end(Some(linker), tbl);
}

/// `spcr_setup()`: SPCR revision 2 for the PL011.
fn spcr_setup(tbl: &mut Vec<u8>, linker: &mut BiosLinker, s: &ArmVirtAcpi) {
    let serial = SpcrData {
        interface_type: 3, // ARM PL011 UART
        base_addr: Gas {
            space_id: AddressSpace::SystemMemory as u8,
            bit_width: 32,
            bit_offset: 0,
            access_width: 3,
            address: s.memmap.uart0.base,
        },
        interrupt_type: 1 << 3, // Bit[3] ARMH GIC interrupt
        pc_interrupt: 0,        // IRQ
        interrupt: s.irqs.uart0,
        baud_rate: 3,         // 9600
        parity: 0,            // No Parity
        stop_bits: 1,         // 1 Stop bit
        flow_control: 1 << 1, // RTS/CTS hardware flow control
        terminal_type: 0,     // VT100
        language: 0,
        pci_device_id: 0xffff, // not a PCI device
        pci_vendor_id: 0xffff, // not a PCI device
        ..SpcrData::default()
    };
    // Revision 2 has no NameSpaceString.
    table::build_spcr(tbl, linker, &serial, 2, &s.oem_id, &s.oem_table_id, &[]);
}

/// `build_dbg2()`: the Debug Port Table 2 with the PL011.
fn build_dbg2(tbl: &mut Vec<u8>, linker: &mut BiosLinker, s: &ArmVirtAcpi) {
    const NAME: &[u8] = b"COM0\0";
    let namespace_length = NAME.len() as u64;
    let table = AcpiTable::begin("DBG2", 0, &s.oem_id, &s.oem_table_id, tbl);

    let dbg2devicelength = 22 // BaseAddressRegister[] offset
        + 12 // BaseAddressRegister[]
        + 4 // AddressSize[]
        + namespace_length; // NamespaceString[]

    append_int_noprefix(tbl, 44, 4); // OffsetDbgDeviceInfo
    append_int_noprefix(tbl, 1, 4); // NumberDbgDeviceInfo

    // Table 2. Debug Device Information structure format
    tbl.push(0); // Revision
    append_int_noprefix(tbl, dbg2devicelength, 2); // Length
    tbl.push(1); // NumberofGenericAddressRegisters
    append_int_noprefix(tbl, namespace_length, 2); // NameSpaceStringLength
    append_int_noprefix(tbl, 38, 2); // NameSpaceStringOffset
    append_int_noprefix(tbl, 0, 2); // OemDataLength
    append_int_noprefix(tbl, 0, 2); // OemDataOffset (0 means no OEM data)
    append_int_noprefix(tbl, 0x8000, 2); // Port Type: Serial
    append_int_noprefix(tbl, 0x3, 2); // Port Subtype: ARM PL011 UART
    append_int_noprefix(tbl, 0, 2); // Reserved
    append_int_noprefix(tbl, 22, 2); // BaseAddressRegisterOffset
    append_int_noprefix(tbl, 34, 2); // AddressSizeOffset

    // BaseAddressRegister[]
    table::append_gas(tbl, AddressSpace::SystemMemory, 32, 0, 3, s.memmap.uart0.base);
    append_int_noprefix(tbl, s.memmap.uart0.size, 4); // AddressSize[]
    tbl.extend_from_slice(NAME); // NamespaceString[]
    table.end(Some(linker), tbl);
}

/// `build_srat()` (ACPI 5.1, 5.2.16).
fn build_srat(tbl: &mut Vec<u8>, linker: &mut BiosLinker, s: &ArmVirtAcpi) {
    let table = AcpiTable::begin("SRAT", 3, &s.oem_id, &s.oem_table_id, tbl);
    append_int_noprefix(tbl, 1, 4); // Reserved
    append_int_noprefix(tbl, 0, 8); // Reserved

    for (i, cpu) in s.possible_cpus.iter().enumerate() {
        // 5.2.16.4 GICC Affinity Structure
        tbl.push(3); // Type
        tbl.push(18); // Length
        append_int_noprefix(tbl, cpu.node.into(), 4); // Proximity Domain
        append_int_noprefix(tbl, i as u64, 4); // ACPI Processor UID
        append_int_noprefix(tbl, 1, 4); // Flags: Enabled
        append_int_noprefix(tbl, 0, 4); // Clock Domain
    }

    let mut mem_base = s.memmap.mem;
    for (i, node) in s.numa.iter().enumerate() {
        if node.mem > 0 {
            table::build_srat_memory(
                tbl,
                mem_base,
                node.mem,
                i as u32,
                table::MEM_AFFINITY_ENABLED,
            );
            mem_base += node.mem;
        }
    }
    table.end(Some(linker), tbl);
}

/// `build_iort_id_mapping()`.
fn append_iort_id_mapping(tbl: &mut Vec<u8>, input_base: u32, id_count: u32, out_ref: u32) {
    // Table 4 ID mapping format
    append_int_noprefix(tbl, input_base.into(), 4); // Input base
    // Number of IDs - The number of IDs in the range minus one
    append_int_noprefix(tbl, (id_count - 1).into(), 4);
    append_int_noprefix(tbl, input_base.into(), 4); // Output base
    append_int_noprefix(tbl, out_ref.into(), 4); // Output Reference
    append_int_noprefix(tbl, 0, 4); // Flags
}

/// `build_iort()` without SMMUv3 nodes (IO Remapping Table, E.b).
fn build_iort(tbl: &mut Vec<u8>, linker: &mut BiosLinker, s: &ArmVirtAcpi) {
    let its = s.memmap.gic_its.is_some();
    let table = AcpiTable::begin("IORT", 5, &s.oem_id, &s.oem_table_id, tbl);
    let mut id = 0u64;

    // RC and ITS with a direct map to the ITS, or the RC alone with no output mapping.
    let (nb_nodes, rc_mapping_count) = if its { (2, 1) } else { (1, 0) };
    append_int_noprefix(tbl, nb_nodes, 4); // Number of IORT Nodes
    append_int_noprefix(tbl, IORT_NODE_OFFSET.into(), 4); // Offset to Array of IORT Nodes
    append_int_noprefix(tbl, 0, 4); // Reserved

    if its {
        // Table 12 ITS Group Format
        tbl.push(0); // Type: ITS Group
        let node_size = 20 + 4; // fixed header size and 1 GIC ITS Identifier
        append_int_noprefix(tbl, node_size, 2); // Length
        tbl.push(1); // Revision
        append_int_noprefix(tbl, id, 4); // Identifier
        id += 1;
        append_int_noprefix(tbl, 0, 4); // Number of ID mappings
        append_int_noprefix(tbl, 0, 4); // Reference to ID Array
        append_int_noprefix(tbl, 1, 4); // Number of ITSs
        append_int_noprefix(tbl, 0, 4); // GIC ITS Identifier Array: MADT translation_id
    }

    // Table 17 Root Complex Node
    tbl.push(2); // Type: Root complex
    let node_size = ROOT_COMPLEX_ENTRY_SIZE + ID_MAPPING_ENTRY_SIZE * rc_mapping_count;
    append_int_noprefix(tbl, node_size.into(), 2); // Length
    tbl.push(3); // Revision
    append_int_noprefix(tbl, id, 4); // Identifier
    append_int_noprefix(tbl, rc_mapping_count.into(), 4); // Number of ID mappings
    append_int_noprefix(tbl, ROOT_COMPLEX_ENTRY_SIZE.into(), 4); // Reference to ID Array

    // Table 14 Memory access properties
    append_int_noprefix(tbl, 1, 4); // CCA: Cache Coherent Attribute, fully coherent
    tbl.push(0); // AH: Note Allocation Hints
    append_int_noprefix(tbl, 0, 2); // Reserved
    tbl.push(0x3); // Table 15 Memory Access Flags: CCA = CPM = DACS = 1
    append_int_noprefix(tbl, 0, 4); // ATS Attribute
    append_int_noprefix(tbl, 0, 4); // MCFG pci_segment: PCI Segment number
    tbl.push(64); // Memory address size limit
    append_int_noprefix(tbl, 0, 3); // Reserved

    // Map all requester IDs to the ITS Group node directly, since there is no SMMU.
    if its {
        append_iort_id_mapping(tbl, 0, 0x10000, IORT_NODE_OFFSET);
    }
    table.end(Some(linker), tbl);
}

/// `virt_acpi_build()`.
pub fn build(s: &ArmVirtAcpi) -> BuildTables {
    let mut t = BuildTables::new();
    let data = &mut t.table_data;
    let linker = &mut t.linker;
    let mut table_offsets = Vec::new();

    linker.alloc(TABLE_FILE, 64, false);

    // DSDT is pointed to by FADT
    let dsdt = data.len() as u32;
    build_dsdt(data, linker, s);

    // FADT MADT PPTT GTDT MCFG SPCR DBG2 pointed to by RSDT
    table::add_table(&mut table_offsets, data);
    build_fadt_rev6(data, linker, s, dsdt);

    table::add_table(&mut table_offsets, data);
    build_madt(data, linker, s);

    // QEMU adds an entry for the WDAT whether or not there is a watchdog, so without one the
    // PPTT gets two.
    table::add_table(&mut table_offsets, data);

    table::add_table(&mut table_offsets, data);
    build_pptt(data, linker, s);

    table::add_table(&mut table_offsets, data);
    build_gtdt(data, linker, s);

    table::add_table(&mut table_offsets, data);
    let mcfg = McfgInfo { base: s.memmap.ecam.base, size: s.memmap.ecam.size };
    table::build_mcfg(data, linker, &mcfg, &s.oem_id, &s.oem_table_id);

    // The same for the SPCR: without it the DBG2 has two entries.
    table::add_table(&mut table_offsets, data);
    if s.spcr {
        spcr_setup(data, linker, s);
    }

    table::add_table(&mut table_offsets, data);
    build_dbg2(data, linker, s);

    if !s.numa.is_empty() {
        table::add_table(&mut table_offsets, data);
        build_srat(data, linker, s);
        if s.numa.iter().any(|n| !n.distance.is_empty()) {
            let distance: Vec<Vec<u8>> = s.numa.iter().map(|n| n.distance.clone()).collect();
            table::add_table(&mut table_offsets, data);
            table::build_slit(data, linker, &distance, &s.oem_id, &s.oem_table_id);
        }
    }

    table::add_table(&mut table_offsets, data);
    build_iort(data, linker, s);

    // XSDT is pointed to by RSDP
    let xsdt = data.len() as u32;
    table::build_xsdt(data, linker, &table_offsets, &s.oem_id, &s.oem_table_id);

    // RSDP is in FSEG memory, so allocate it separately
    let rsdp = RsdpData {
        revision: 2,
        oem_id: &s.oem_id,
        xsdt_tbl_offset: Some(xsdt),
        rsdt_tbl_offset: None,
    };
    table::build_rsdp(&mut t.rsdp, linker, &rsdp);

    // The align size is 128 KiB; warn if 64 KiB is not enough, since then the align size
    // could be resized.
    if t.table_data.len() > ACPI_BUILD_TABLE_SIZE / 2 {
        eprintln!(
            "warning: ACPI table size {} exceeds {} bytes, migration may not work",
            t.table_data.len(),
            ACPI_BUILD_TABLE_SIZE / 2
        );
        eprintln!("Try removing CPUs, NUMA nodes, memory slots or PCI bridges.");
    }
    table::align_size(&mut t.table_data, ACPI_BUILD_TABLE_SIZE);
    t
}

/// The possible CPUs of a `sockets` x `clusters` x `cores` x `threads` topology, as
/// `virt_possible_cpu_arch_ids()` numbers them, all on NUMA node 0.
pub fn possible_cpus(sockets: u32, clusters: u32, cores: u32, threads: u32) -> Vec<PossibleCpu> {
    (0..sockets * clusters * cores * threads)
        .map(|n| PossibleCpu {
            socket: n / (clusters * cores * threads),
            cluster: (n / (cores * threads)) % clusters,
            core: (n / threads) % cores,
            thread: n % threads,
            node: 0,
        })
        .collect()
}
