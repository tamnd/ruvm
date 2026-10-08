// SPDX-License-Identifier: GPL-2.0-or-later

//! The tables of the RISC-V `virt` board, hw/riscv/virt-acpi-build.c: DSDT, FADT, MADT, RHCT,
//! SPCR, MCFG, SRAT and SLIT behind an XSDT and a revision 2 RSDP.
//!
//! The board is hardware reduced ACPI. The DSDT has the harts (with their RINTC as `_MAT`),
//! fw_cfg, the PLIC or the S level APLIC of each socket, the UART, the eight virtio-mmio
//! transports and the PCIe host bridge. The MADT has a RINTC per hart and then the IMSIC,
//! the APLICs or the PLICs, and the RHCT describes the ISA string, the cache block sizes and
//! the MMU of the first hart. SRAT and SLIT are there only with NUMA nodes.
//!
//! RIMT (the IOMMU table) is not built, since ruvm has no RISC-V IOMMU.

use super::BuildTables;
use super::aml::{self, AddressSpace, Aml, append_int_noprefix};
use super::devices;
use super::gpex::{self, GpexAcpi, Window};
use super::linker::BiosLinker;
use super::table::{
    self, AcpiTable, FadtData, Gas, McfgInfo, RsdpData, SpcrData, TABLE_FILE, fadt_flags,
};

/// `ACPI_BUILD_TABLE_SIZE`: `etc/acpi/tables` is padded to a multiple of this.
pub const ACPI_BUILD_TABLE_SIZE: usize = 0x20000;
/// `IMSIC_MMIO_PAGE_SHIFT`.
const IMSIC_MMIO_PAGE_SHIFT: u32 = 12;
/// `IMSIC_MMIO_GROUP_MIN_SHIFT`.
const IMSIC_MMIO_GROUP_MIN_SHIFT: u8 = 24;
/// `VIRT_IMSIC_GROUP_MAX_SIZE`, the IMSIC space of one socket.
const VIRT_IMSIC_GROUP_MAX_SIZE: u64 = 1 << IMSIC_MMIO_GROUP_MIN_SHIFT;
/// `RHCT_NODE_ARRAY_OFFSET`.
const RHCT_NODE_ARRAY_OFFSET: u32 = 56;
/// `RISCV_ACLINT_DEFAULT_TIMEBASE_FREQ`, the RHCT time base under TCG.
pub const RISCV_ACLINT_DEFAULT_TIMEBASE_FREQ: u64 = 10_000_000;

/// `RISCVVirtAIAType`, the interrupt controllers of the board.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VirtAia {
    /// The SiFive PLIC.
    #[default]
    None,
    /// APLIC domains in direct mode.
    Aplic,
    /// APLIC domains in MSI mode and IMSICs.
    AplicImsic,
}

/// One possible hart, an entry of `possible_cpu_arch_ids()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hart {
    /// `arch_id`, the hart ID.
    pub hart_id: u64,
    /// `props.node_id`, its socket (and NUMA node).
    pub socket: u32,
}

/// One socket: `riscv_socket_first_hartid()` and `riscv_socket_hart_count()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Socket {
    pub first_hartid: u64,
    pub num_harts: u32,
}

/// One NUMA node, `NodeInfo`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NumaNode {
    /// `node_mem`.
    pub mem: u64,
    /// `distance[]` to every node, or empty when no distances were given.
    pub distance: Vec<u8>,
}

/// The parts of `virt_memmap[]` the tables use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VirtMemmap {
    pub plic: Window,
    pub aplic_s: Window,
    pub imsic_s: Window,
    pub uart0: Window,
    /// The first virtio-mmio transport.
    pub virtio: Window,
    pub fw_cfg: Window,
    pub pcie_ecam: Window,
    pub pcie_mmio: Window,
    pub pcie_pio: Window,
    /// `virt_high_pcie_memmap`.
    pub pcie_mmio_high: Window,
    pub dram: Window,
}

/// The cache block management of the first hart, for the RHCT CMO node.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cmo {
    /// `cbom_blocksize`, 0 when unknown.
    pub cbom_blocksize: u16,
    /// `cboz_blocksize`, 0 when unknown.
    pub cboz_blocksize: u16,
}

/// The RHCT MMU type: Sv39, Sv48 or Sv57.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MmuType {
    Sv39 = 0,
    Sv48 = 1,
    Sv57 = 2,
}

/// Everything about a `virt` instance that shows up in its tables.
#[derive(Clone, Debug)]
pub struct RiscvVirtAcpi {
    pub oem_id: String,
    pub oem_table_id: String,
    pub memmap: VirtMemmap,
    /// The possible harts in `arch_id` order.
    pub harts: Vec<Hart>,
    /// `ms->smp.cpus`, the number of hart nodes the RHCT header claims.
    pub smp_cpus: u32,
    pub sockets: Vec<Socket>,
    pub aia: VirtAia,
    /// `aia_guests`.
    pub aia_guests: u32,
    /// `num_sources` of each PLIC or APLIC, `VIRT_IRQCHIP_NUM_SOURCES`.
    pub num_sources: u32,
    /// `VIRT_IRQCHIP_NUM_MSIS`.
    pub num_msis: u32,
    pub uart_irq: u32,
    /// The interrupt of the first virtio-mmio transport.
    pub virtio_irq: u32,
    pub virtio_count: u32,
    /// The GSI of INTA of the PCIe host bridge.
    pub pcie_irq: u32,
    /// `riscv_isa_string()` of the first hart.
    pub isa: String,
    /// `Some` when the first hart has Zicbom or Zicboz.
    pub cmo: Option<Cmo>,
    /// `max_satp_mode` of the first hart when it is Sv39 or more.
    pub mmu: Option<MmuType>,
    pub timebase_freq: u64,
    /// `ms->acpi_spcr_enabled`.
    pub spcr: bool,
    /// The NUMA nodes, empty without `-numa`.
    pub numa: Vec<NumaNode>,
}

/// `imsic_num_bits()`: the bits it takes to number `count` things.
fn imsic_num_bits(count: u32) -> u8 {
    let mut ret = 0;
    while (1u64 << ret) < u64::from(count) {
        ret += 1;
    }
    ret
}

/// `IMSIC_HART_SIZE()`.
fn imsic_hart_size(guest_bits: u8) -> u64 {
    (1u64 << guest_bits) << IMSIC_MMIO_PAGE_SHIFT
}

/// `ACPI_BUILD_INTC_ID()`.
fn intc_id(socket: u32, index: u64) -> u64 {
    u64::from(socket) << 24 | index
}

/// `riscv_acpi_madt_add_rintc()`: the RINTC structure of hart `uid`.
fn madt_add_rintc(uid: usize, entry: &mut Vec<u8>, s: &RiscvVirtAcpi) {
    let guest_index_bits = imsic_num_bits(s.aia_guests + 1);
    let hart = s.harts[uid];
    let socket = s.sockets[hart.socket as usize];
    let local_cpu_id = (hart.hart_id - socket.first_hartid) % u64::from(socket.num_harts);
    let imsic_socket_addr =
        s.memmap.imsic_s.base + u64::from(hart.socket) * VIRT_IMSIC_GROUP_MAX_SIZE;
    let imsic_size = imsic_hart_size(guest_index_bits);
    let imsic_addr = imsic_socket_addr + local_cpu_id * imsic_size;
    entry.push(0x18); // Type
    entry.push(36); // Length
    entry.push(1); // Version
    entry.push(0); // Reserved
    append_int_noprefix(entry, 0x1, 4); // Flags
    append_int_noprefix(entry, hart.hart_id, 8);
    append_int_noprefix(entry, uid as u64, 4); // ACPI Processor UID
    // External Interrupt Controller ID. Under TCG each hart has an M and an S context on the
    // PLIC, and the S one is the odd one.
    let ext = match s.aia {
        VirtAia::Aplic => intc_id(hart.socket, local_cpu_id),
        VirtAia::None => intc_id(hart.socket, 2 * local_cpu_id + 1),
        VirtAia::AplicImsic => 0,
    };
    append_int_noprefix(entry, ext, 4);
    if s.aia == VirtAia::AplicImsic {
        append_int_noprefix(entry, imsic_addr, 8);
        append_int_noprefix(entry, imsic_size, 4);
    } else {
        append_int_noprefix(entry, 0, 8);
        append_int_noprefix(entry, 0, 4);
    }
}

/// `acpi_dsdt_add_cpus()`.
fn dsdt_add_cpus(scope: &mut Aml, s: &RiscvVirtAcpi) {
    for (i, hart) in s.harts.iter().enumerate() {
        let mut dev = aml::device(&format!("C{i:03X}"));
        dev.append(&aml::name_decl("_HID", &aml::string("ACPI0007")));
        dev.append(&aml::name_decl("_UID", &aml::int(hart.hart_id)));
        let mut madt_buf = Vec::new();
        madt_add_rintc(i, &mut madt_buf, s);
        dev.append(&aml::name_decl("_MAT", &aml::buffer(madt_buf.len(), Some(&madt_buf))));
        scope.append(&dev);
    }
}

/// `acpi_dsdt_add_plic_aplic()`.
fn dsdt_add_plic_aplic(
    scope: &mut Aml,
    socket_count: u32,
    num_sources: u32,
    mmio: Window,
    hid: &str,
) {
    // The RISC-V Advanced Interrupt Architecture, Chapter 1.2. Limits
    assert!(num_sources <= 1023, "{num_sources} interrupt sources");
    for socket in 0..socket_count {
        let plic_aplic_addr = mmio.base + mmio.size * u64::from(socket);
        let gsi_base = num_sources * socket;
        let mut dev = aml::device(&format!("IC{socket:02X}"));
        dev.append(&aml::name_decl("_HID", &aml::string(hid)));
        dev.append(&aml::name_decl("_UID", &aml::int(socket.into())));
        dev.append(&aml::name_decl("_GSB", &aml::int(gsi_base.into())));
        let mut crs = aml::resource_template();
        crs.append(&aml::memory32_fixed(
            plic_aplic_addr as u32,
            mmio.size as u32,
            aml::ReadWrite::ReadWrite,
        ));
        dev.append(&aml::name_decl("_CRS", &crs));
        scope.append(&dev);
    }
}

/// `acpi_dsdt_add_uart()`.
fn dsdt_add_uart(scope: &mut Aml, uart: Window, uart_irq: u32) {
    let mut dev = aml::device("COM0");
    dev.append(&aml::name_decl("_HID", &aml::string("RSCV0003")));
    dev.append(&aml::name_decl("_UID", &aml::int(0)));

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

    let mut pkg = aml::package(2);
    pkg.append(&aml::string("clock-frequency"));
    pkg.append(&aml::int(3_686_400));
    let mut pkg1 = aml::package(1);
    pkg1.append(&pkg);
    // The Device Properties UUID.
    let mut package = aml::package(2);
    package.append(&aml::touuid("DAFFD814-6EBA-4D8C-8A91-BC9BBF4AA301"));
    package.append(&pkg1);
    dev.append(&aml::name_decl("_DSD", &package));
    scope.append(&dev);
}

/// `spcr_setup()`: a 16550 at the UART's address on its GSI.
fn spcr_setup(tbl: &mut Vec<u8>, linker: &mut BiosLinker, s: &RiscvVirtAcpi) {
    let name = b".\0";
    let serial = SpcrData {
        interface_type: 0x12, // 16550 compatible
        base_addr: Gas {
            space_id: AddressSpace::SystemMemory as u8,
            bit_width: 32,
            bit_offset: 0,
            access_width: 1,
            address: s.memmap.uart0.base,
        },
        interrupt_type: 1 << 4, // Bit 4: RISC-V PLIC or APLIC
        pc_interrupt: 0,
        interrupt: s.uart_irq,
        baud_rate: 7, // 115200
        parity: 0,
        stop_bits: 1,
        flow_control: 0,
        terminal_type: 3, // ANSI
        language: 0,
        pci_device_id: 0xffff, // not a PCI device
        pci_vendor_id: 0xffff, // not a PCI device
        pci_bus: 0,
        pci_device: 0,
        pci_function: 0,
        pci_flags: 0,
        pci_segment: 0,
        uart_clk_freq: 0,
        precise_baudrate: 0,
        namespace_string_length: name.len() as u16,
        namespace_string_offset: 88,
    };
    table::build_spcr(tbl, linker, &serial, 4, &s.oem_id, &s.oem_table_id, name);
}

/// `build_rhct()`.
fn build_rhct(tbl: &mut Vec<u8>, linker: &mut BiosLinker, s: &RiscvVirtAcpi) {
    let table = AcpiTable::begin("RHCT", 1, &s.oem_id, &s.oem_table_id, tbl);
    append_int_noprefix(tbl, 0x0, 4); // Reserved
    append_int_noprefix(tbl, s.timebase_freq, 8); // Time Base Frequency

    // ISA and a node per hart.
    let mut num_rhct_nodes = 1 + s.smp_cpus;
    if s.cmo.is_some() {
        num_rhct_nodes += 1;
    }
    if s.mmu.is_some() {
        num_rhct_nodes += 1;
    }
    append_int_noprefix(tbl, num_rhct_nodes.into(), 4);
    append_int_noprefix(tbl, RHCT_NODE_ARRAY_OFFSET.into(), 4);

    // ISA String Node
    let isa_offset = (tbl.len() - table.offset()) as u64;
    append_int_noprefix(tbl, 0, 2); // Type 0
    let len = 8 + s.isa.len() + 1;
    let aligned_len = len.next_multiple_of(2);
    append_int_noprefix(tbl, aligned_len as u64, 2); // Length
    append_int_noprefix(tbl, 0x1, 2); // Revision
    // ISA string length including NUL
    append_int_noprefix(tbl, (s.isa.len() + 1) as u64, 2);
    tbl.extend_from_slice(s.isa.as_bytes());
    tbl.push(0);
    if aligned_len != len {
        tbl.push(0); // Optional Padding
    }

    // CMO node
    let mut cmo_offset = 0;
    if let Some(cmo) = s.cmo {
        cmo_offset = (tbl.len() - table.offset()) as u64;
        append_int_noprefix(tbl, 1, 2); // Type
        append_int_noprefix(tbl, 10, 2); // Length
        append_int_noprefix(tbl, 0x1, 2); // Revision
        tbl.push(0); // Reserved
        let log2 = |size: u16| if size == 0 { 0 } else { size.trailing_zeros() as u8 };
        tbl.push(log2(cmo.cbom_blocksize)); // CBOM block size
        tbl.push(0); // CBOP block size
        tbl.push(log2(cmo.cboz_blocksize)); // CBOZ block size
    }

    // MMU node structure
    let mut mmu_offset = 0;
    if let Some(mmu) = s.mmu {
        mmu_offset = (tbl.len() - table.offset()) as u64;
        append_int_noprefix(tbl, 2, 2); // Type
        append_int_noprefix(tbl, 8, 2); // Length
        append_int_noprefix(tbl, 0x1, 2); // Revision
        tbl.push(0); // Reserved
        tbl.push(mmu as u8); // MMU Type
    }

    // Hart Info Node
    for i in 0..s.harts.len() {
        let mut len = 16;
        let mut num_offsets = 1;
        append_int_noprefix(tbl, 0xFFFF, 2); // Type
        if cmo_offset != 0 {
            len += 4;
            num_offsets += 1;
        }
        if mmu_offset != 0 {
            len += 4;
            num_offsets += 1;
        }
        append_int_noprefix(tbl, len, 2); // Length
        append_int_noprefix(tbl, 0x1, 2); // Revision
        append_int_noprefix(tbl, num_offsets, 2); // Number of offsets
        append_int_noprefix(tbl, i as u64, 4); // ACPI Processor UID
        append_int_noprefix(tbl, isa_offset, 4);
        if cmo_offset != 0 {
            append_int_noprefix(tbl, cmo_offset, 4);
        }
        if mmu_offset != 0 {
            append_int_noprefix(tbl, mmu_offset, 4);
        }
    }
    table.end(Some(linker), tbl);
}

/// `build_fadt_rev6()`: hardware reduced, with only the 64 bit DSDT pointer.
fn build_fadt_rev6(tbl: &mut Vec<u8>, linker: &mut BiosLinker, s: &RiscvVirtAcpi, dsdt: u32) {
    let fadt = FadtData {
        rev: 6,
        minor_ver: 6,
        flags: 1 << fadt_flags::HW_REDUCED_ACPI,
        xdsdt_tbl_offset: Some(dsdt),
        ..FadtData::default()
    };
    table::build_fadt(tbl, linker, &fadt, &s.oem_id, &s.oem_table_id);
}

/// `build_dsdt()`.
fn build_dsdt(tbl: &mut Vec<u8>, linker: &mut BiosLinker, s: &RiscvVirtAcpi) {
    let table = AcpiTable::begin("DSDT", 2, &s.oem_id, &s.oem_table_id, tbl);
    let mut dsdt = Aml::new();
    let memmap = &s.memmap;

    // When booting with UEFI, UEFI takes ownership of the RTC hardware. It can disable the
    // RTC node in the device tree it passes on but cannot change AML, so there is no RTC
    // device here at all.
    let mut scope = aml::scope("\\_SB");
    dsdt_add_cpus(&mut scope, s);
    devices::fw_cfg_mmio(&mut scope, memmap.fw_cfg.base, memmap.fw_cfg.size);

    let socket_count = s.sockets.len() as u32;
    if s.aia == VirtAia::None {
        dsdt_add_plic_aplic(&mut scope, socket_count, s.num_sources, memmap.plic, "RSCV0001");
    } else {
        dsdt_add_plic_aplic(&mut scope, socket_count, s.num_sources, memmap.aplic_s, "RSCV0002");
    }
    dsdt_add_uart(&mut scope, memmap.uart0, s.uart_irq);

    // The virtio-mmio transports are on the second socket's interrupt controller with two
    // sockets or more, and PCIe on the third with three or more.
    let (virtio_irq, pcie_irq) = match socket_count {
        1 => (s.virtio_irq, s.pcie_irq),
        2 => (s.virtio_irq + s.num_sources, s.pcie_irq + s.num_sources),
        _ => (s.virtio_irq + s.num_sources, s.pcie_irq + s.num_sources * 2),
    };
    devices::virtio_mmio(
        &mut scope,
        memmap.virtio.base,
        memmap.virtio.size,
        virtio_irq,
        0,
        s.virtio_count,
    );
    let gpex = GpexAcpi {
        ecam: memmap.pcie_ecam,
        mmio32: memmap.pcie_mmio,
        mmio64: memmap.pcie_mmio_high,
        pio: memmap.pcie_pio,
        irq: pcie_irq,
        pci_native_hotplug: false,
        preserve_config: false,
    };
    gpex::add_gpex(&mut scope, &gpex);

    dsdt.append(&scope);
    tbl.extend_from_slice(dsdt.as_bytes());
    table.end(Some(linker), tbl);
}

/// `build_madt()`.
fn build_madt(tbl: &mut Vec<u8>, linker: &mut BiosLinker, s: &RiscvVirtAcpi) {
    let socket_count = s.sockets.len() as u32;
    let group_index_bits = imsic_num_bits(socket_count);
    let guest_index_bits = imsic_num_bits(s.aia_guests + 1);
    let imsic_max_hart_per_socket = s.sockets.iter().map(|k| k.num_harts).max().unwrap_or(0);
    let hart_index_bits = imsic_num_bits(imsic_max_hart_per_socket);

    let table = AcpiTable::begin("APIC", 7, &s.oem_id, &s.oem_table_id, tbl);
    append_int_noprefix(tbl, 0, 4); // Local Interrupt Controller Address
    append_int_noprefix(tbl, 0, 4); // MADT Flags

    // RISC-V Local INTC structures per HART
    for i in 0..s.harts.len() {
        madt_add_rintc(i, tbl, s);
    }

    if s.aia == VirtAia::AplicImsic {
        tbl.push(0x19); // Type
        tbl.push(16); // Length
        tbl.push(1); // Version
        tbl.push(0); // Reserved
        append_int_noprefix(tbl, 0, 4); // Flags
        // Number of supervisor mode Interrupt Identities
        append_int_noprefix(tbl, s.num_msis.into(), 2);
        // Number of guest mode Interrupt Identities
        append_int_noprefix(tbl, s.num_msis.into(), 2);
        tbl.push(guest_index_bits);
        tbl.push(hart_index_bits);
        tbl.push(group_index_bits);
        tbl.push(IMSIC_MMIO_GROUP_MIN_SHIFT); // Group Index Shift
    }

    if s.aia != VirtAia::None {
        // APLICs
        for socket in 0..socket_count {
            let aplic_addr = s.memmap.aplic_s.base + s.memmap.aplic_s.size * u64::from(socket);
            let gsi_base = s.num_sources * socket;
            tbl.push(0x1A); // Type
            tbl.push(36); // Length
            tbl.push(1); // Version
            tbl.push(socket as u8); // APLIC ID
            append_int_noprefix(tbl, 0, 4); // Flags
            append_int_noprefix(tbl, 0, 8); // Hardware ID
            // Number of IDCs
            let idcs =
                if s.aia == VirtAia::Aplic { s.sockets[socket as usize].num_harts } else { 0 };
            append_int_noprefix(tbl, idcs.into(), 2);
            // Total External Interrupt Sources Supported
            append_int_noprefix(tbl, s.num_sources.into(), 2);
            append_int_noprefix(tbl, gsi_base.into(), 4); // Global System Interrupt Base
            append_int_noprefix(tbl, aplic_addr, 8);
            append_int_noprefix(tbl, s.memmap.aplic_s.size, 4);
        }
    } else {
        // PLICs
        for socket in 0..socket_count {
            let plic_addr = s.memmap.plic.base + s.memmap.plic.size * u64::from(socket);
            let gsi_base = s.num_sources * socket;
            tbl.push(0x1B); // Type
            tbl.push(36); // Length
            tbl.push(1); // Version
            tbl.push(socket as u8); // PLIC ID
            append_int_noprefix(tbl, 0, 8); // Hardware ID
            // Total External Interrupt Sources Supported
            append_int_noprefix(tbl, (s.num_sources - 1).into(), 2);
            append_int_noprefix(tbl, 0, 2); // Max Priority
            append_int_noprefix(tbl, 0, 4); // Flags
            append_int_noprefix(tbl, s.memmap.plic.size, 4); // PLIC Size
            append_int_noprefix(tbl, plic_addr, 8); // PLIC Address
            append_int_noprefix(tbl, gsi_base.into(), 4); // Global System Interrupt Vector Base
        }
    }
    table.end(Some(linker), tbl);
}

/// `build_srat()`.
fn build_srat(tbl: &mut Vec<u8>, linker: &mut BiosLinker, s: &RiscvVirtAcpi) {
    let table = AcpiTable::begin("SRAT", 3, &s.oem_id, &s.oem_table_id, tbl);
    append_int_noprefix(tbl, 1, 4); // Reserved
    append_int_noprefix(tbl, 0, 8); // Reserved

    for (i, hart) in s.harts.iter().enumerate() {
        // ACPI 6.6, 5.2.16.8 RINTC Affinity Structure
        tbl.push(7); // Type
        tbl.push(20); // Length
        append_int_noprefix(tbl, 0, 2); // Reserved
        append_int_noprefix(tbl, hart.socket.into(), 4); // Proximity Domain
        append_int_noprefix(tbl, i as u64, 4); // ACPI Processor UID
        append_int_noprefix(tbl, 1, 4); // Flags: Enabled
        append_int_noprefix(tbl, 0, 4); // Clock Domain
    }

    let mut mem_base = s.memmap.dram.base;
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

/// `virt_acpi_build()`.
pub fn build(s: &RiscvVirtAcpi) -> BuildTables {
    let mut t = BuildTables::new();
    let data = &mut t.table_data;
    let linker = &mut t.linker;
    let mut table_offsets = Vec::new();

    linker.alloc(TABLE_FILE, 64, false);

    // DSDT is pointed to by FADT
    let dsdt = data.len() as u32;
    build_dsdt(data, linker, s);

    // FADT and others pointed to by XSDT
    table::add_table(&mut table_offsets, data);
    build_fadt_rev6(data, linker, s, dsdt);

    table::add_table(&mut table_offsets, data);
    build_madt(data, linker, s);

    table::add_table(&mut table_offsets, data);
    build_rhct(data, linker, s);

    // QEMU adds an XSDT entry here whether or not there is an SPCR, so without one the
    // entry points at the MCFG, which then has two.
    table::add_table(&mut table_offsets, data);
    if s.spcr {
        spcr_setup(data, linker, s);
    }

    table::add_table(&mut table_offsets, data);
    let mcfg = McfgInfo { base: s.memmap.pcie_ecam.base, size: s.memmap.pcie_ecam.size };
    table::build_mcfg(data, linker, &mcfg, &s.oem_id, &s.oem_table_id);

    if !s.numa.is_empty() {
        table::add_table(&mut table_offsets, data);
        build_srat(data, linker, s);
        if s.numa.iter().any(|n| !n.distance.is_empty()) {
            let distance: Vec<Vec<u8>> = s.numa.iter().map(|n| n.distance.clone()).collect();
            table::add_table(&mut table_offsets, data);
            table::build_slit(data, linker, &distance, &s.oem_id, &s.oem_table_id);
        }
    }

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

    if t.table_data.len() > ACPI_BUILD_TABLE_SIZE / 2 {
        eprintln!(
            "warning: ACPI table size {} exceeds {} bytes, migration may not work",
            t.table_data.len(),
            ACPI_BUILD_TABLE_SIZE / 2
        );
        eprintln!("Try removing some objects.");
    }
    table::align_size(&mut t.table_data, ACPI_BUILD_TABLE_SIZE);
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_bits() {
        assert_eq!(imsic_num_bits(0), 0);
        assert_eq!(imsic_num_bits(1), 0);
        assert_eq!(imsic_num_bits(2), 1);
        assert_eq!(imsic_num_bits(3), 2);
        assert_eq!(imsic_num_bits(8), 3);
        assert_eq!(imsic_hart_size(imsic_num_bits(4)), 0x4000);
    }
}
