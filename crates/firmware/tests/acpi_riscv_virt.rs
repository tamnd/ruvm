// SPDX-License-Identifier: GPL-2.0-or-later

//! The riscv64 `virt` tables against the expected blobs QEMU's bios-tables-test uses. Those
//! come from `-M virt -cpu rva22s64` with one hart, the PLIC and 128 MiB of RAM.

mod common;

use ruvm_firmware::acpi::gpex::Window;
use ruvm_firmware::acpi::linker::Command;
use ruvm_firmware::acpi::riscv_virt::{
    self, Cmo, Hart, MmuType, NumaNode, RISCV_ACLINT_DEFAULT_TIMEBASE_FREQ, RiscvVirtAcpi, Socket,
    VirtAia, VirtMemmap,
};
use ruvm_firmware::acpi::table::{APPNAME6, APPNAME8, RSDP_FILE, TABLE_FILE};

use common::load;

const MACHINE: &str = "virt";

fn w(base: u64, size: u64) -> Window {
    Window { base, size }
}

/// The ISA string of `-cpu rva22s64`, taken from the expected RHCT.
fn rva22s64_isa() -> String {
    let rhct = common::expected_arch("riscv64", MACHINE, "RHCT");
    let len = u16::from_le_bytes([rhct[0x3e], rhct[0x3f]]) as usize;
    String::from_utf8(rhct[0x40..0x40 + len - 1].to_vec()).unwrap()
}

fn base() -> RiscvVirtAcpi {
    RiscvVirtAcpi {
        oem_id: APPNAME6.into(),
        oem_table_id: APPNAME8.into(),
        memmap: VirtMemmap {
            plic: w(0x0c00_0000, 0x60_0000),
            aplic_s: w(0x0d00_0000, 0x8000),
            imsic_s: w(0x2800_0000, 0x400_0000),
            uart0: w(0x1000_0000, 0x100),
            virtio: w(0x1000_1000, 0x1000),
            fw_cfg: w(0x1010_0000, 0x18),
            pcie_ecam: w(0x3000_0000, 0x1000_0000),
            pcie_mmio: w(0x4000_0000, 0x4000_0000),
            pcie_pio: w(0x0300_0000, 0x1_0000),
            pcie_mmio_high: w(0x4_0000_0000, 0x4_0000_0000),
            dram: w(0x8000_0000, 0x800_0000),
        },
        harts: vec![Hart { hart_id: 0, socket: 0 }],
        smp_cpus: 1,
        sockets: vec![Socket { first_hartid: 0, num_harts: 1 }],
        aia: VirtAia::None,
        aia_guests: 0,
        num_sources: 96,
        num_msis: 255,
        uart_irq: 10,
        virtio_irq: 1,
        virtio_count: 8,
        pcie_irq: 0x20,
        isa: rva22s64_isa(),
        cmo: Some(Cmo { cbom_blocksize: 64, cboz_blocksize: 64 }),
        mmu: Some(MmuType::Sv39),
        timebase_freq: RISCV_ACLINT_DEFAULT_TIMEBASE_FREQ,
        spcr: true,
        numa: Vec::new(),
    }
}

fn check(m: &RiscvVirtAcpi, sig: &str, file: &str) {
    common::check_arch(&riscv_virt::build(m), "riscv64", MACHINE, sig, file);
}

#[test]
fn dsdt() {
    check(&base(), "DSDT", "DSDT");
}

#[test]
fn facp() {
    check(&base(), "FACP", "FACP");
}

#[test]
fn apic() {
    check(&base(), "APIC", "APIC");
}

#[test]
fn rhct() {
    check(&base(), "RHCT", "RHCT");
}

#[test]
fn spcr() {
    check(&base(), "SPCR", "SPCR");
}

#[test]
fn mcfg() {
    check(&base(), "MCFG", "MCFG");
}

/// `-object memory-backend-ram,id=ram0,size=128M -numa node,memdev=ram0`.
#[test]
fn srat_numamem() {
    let mut m = base();
    m.numa = vec![NumaNode { mem: 0x800_0000, distance: Vec::new() }];
    let t = riscv_virt::build(&m);
    let sigs: Vec<String> = load(&t).into_iter().map(|(s, _)| s).collect();
    assert!(!sigs.contains(&"SLIT".to_string()), "{sigs:?}");
    common::check_arch(&t, "riscv64", MACHINE, "SRAT", "SRAT.numamem");
}

/// The order of the tables, and the XSDT pointing at all but the DSDT.
#[test]
fn layout() {
    let t = riscv_virt::build(&base());
    let tables = load(&t);
    let sigs: Vec<&str> = tables.iter().map(|(s, _)| s.as_str()).collect();
    assert_eq!(sigs, ["DSDT", "FACP", "APIC", "RHCT", "SPCR", "MCFG", "XSDT"]);
    assert_eq!(t.table_data.len(), 0x20000);
    let xsdt = common::table(&tables, "XSDT");
    assert_eq!(xsdt.len(), 36 + 5 * 8);
    assert_eq!(t.rsdp.len(), 36);
    assert_eq!(&t.rsdp[..8], b"RSD PTR ");
    assert_eq!(t.rsdp[15], 2, "RSDP revision");
    let cmds = t.linker.commands();
    assert!(cmds.iter().any(|c| matches!(c, Command::Allocate { file, .. } if file == TABLE_FILE)));
    assert!(cmds.iter().any(|c| matches!(c, Command::Allocate { file, .. } if file == RSDP_FILE)));
}

/// QEMU reserves the SPCR's XSDT entry even with `spcr=off`, so the MCFG is listed twice.
#[test]
fn no_spcr() {
    let mut m = base();
    m.spcr = false;
    let t = riscv_virt::build(&m);
    let tables = load(&t);
    let sigs: Vec<&str> = tables.iter().map(|(s, _)| s.as_str()).collect();
    assert_eq!(sigs, ["DSDT", "FACP", "APIC", "RHCT", "MCFG", "XSDT"]);
    let xsdt = common::table(&tables, "XSDT");
    let entries: Vec<u64> =
        xsdt[36..].chunks(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(entries.len(), 5);
    assert_eq!(entries[3], entries[4]);
}

/// Two NUMA nodes with distances give an SLIT after the SRAT.
#[test]
fn numa_distances() {
    let mut m = base();
    m.harts = vec![Hart { hart_id: 0, socket: 0 }, Hart { hart_id: 1, socket: 1 }];
    m.smp_cpus = 2;
    m.sockets =
        vec![Socket { first_hartid: 0, num_harts: 1 }, Socket { first_hartid: 1, num_harts: 1 }];
    m.numa = vec![
        NumaNode { mem: 0x400_0000, distance: vec![10, 20] },
        NumaNode { mem: 0x400_0000, distance: vec![20, 10] },
    ];
    let tables = load(&riscv_virt::build(&m));
    let sigs: Vec<&str> = tables.iter().map(|(s, _)| s.as_str()).collect();
    assert_eq!(sigs, ["DSDT", "FACP", "APIC", "RHCT", "SPCR", "MCFG", "SRAT", "SLIT", "XSDT"]);
    let slit = common::table(&tables, "SLIT");
    assert_eq!(&slit[36..], [2, 0, 0, 0, 0, 0, 0, 0, 10, 20, 20, 10]);
    let srat = common::table(&tables, "SRAT");
    // Two RINTC affinity structures, then two memory ranges back to back.
    assert_eq!(srat.len(), 48 + 2 * 20 + 2 * 40);
    let mem1 = &srat[48 + 40 + 40..];
    assert_eq!(u32::from_le_bytes(mem1[2..6].try_into().unwrap()), 1);
    assert_eq!(u32::from_le_bytes(mem1[8..12].try_into().unwrap()), 0x8400_0000);
    // The second socket's PLIC follows the first one with its GSIs after the first's.
    let apic = common::table(&tables, "APIC");
    let plic1 = &apic[44 + 2 * 36 + 36..];
    assert_eq!(plic1[0], 0x1B);
    assert_eq!(plic1[3], 1);
    assert_eq!(u64::from_le_bytes(plic1[24..32].try_into().unwrap()), 0x0c60_0000);
    assert_eq!(u32::from_le_bytes(plic1[32..36].try_into().unwrap()), 96);
}

/// With IMSICs the RINTCs carry their IMSIC page and the MADT has an IMSIC structure and an
/// APLIC with no IDCs.
#[test]
fn madt_aplic_imsic() {
    let mut m = base();
    m.aia = VirtAia::AplicImsic;
    m.aia_guests = 3;
    m.harts = (0..4).map(|i| Hart { hart_id: i, socket: 0 }).collect();
    m.smp_cpus = 4;
    m.sockets = vec![Socket { first_hartid: 0, num_harts: 4 }];
    let tables = load(&riscv_virt::build(&m));
    let apic = common::table(&tables, "APIC");
    assert_eq!(apic.len(), 44 + 4 * 36 + 16 + 36);
    let rintc3 = &apic[44 + 3 * 36..44 + 4 * 36];
    assert_eq!(u32::from_le_bytes(rintc3[20..24].try_into().unwrap()), 0, "ext intc id");
    // Two guest index bits: four 4 KiB pages per hart.
    assert_eq!(u64::from_le_bytes(rintc3[24..32].try_into().unwrap()), 0x2800_0000 + 3 * 0x4000);
    assert_eq!(u32::from_le_bytes(rintc3[32..36].try_into().unwrap()), 0x4000);
    let imsic = &apic[44 + 4 * 36..44 + 4 * 36 + 16];
    assert_eq!(imsic, [0x19, 16, 1, 0, 0, 0, 0, 0, 255, 0, 255, 0, 2, 2, 0, 24]);
    let aplic = &apic[44 + 4 * 36 + 16..];
    assert_eq!(aplic[0], 0x1A);
    assert_eq!(u16::from_le_bytes([aplic[16], aplic[17]]), 0, "IDCs");
    assert_eq!(u64::from_le_bytes(aplic[24..32].try_into().unwrap()), 0x0d00_0000);
}

/// With APLIC direct mode the RINTC names the hart's IDC.
#[test]
fn madt_aplic() {
    let mut m = base();
    m.aia = VirtAia::Aplic;
    m.harts = (0..2).map(|i| Hart { hart_id: i, socket: 0 }).collect();
    m.smp_cpus = 2;
    m.sockets = vec![Socket { first_hartid: 0, num_harts: 2 }];
    let tables = load(&riscv_virt::build(&m));
    let apic = common::table(&tables, "APIC");
    let rintc1 = &apic[44 + 36..44 + 2 * 36];
    assert_eq!(u32::from_le_bytes(rintc1[20..24].try_into().unwrap()), 1);
    let aplic = &apic[44 + 2 * 36..];
    assert_eq!(u16::from_le_bytes([aplic[16], aplic[17]]), 2, "IDCs");
    let dsdt = common::table(&tables, "DSDT");
    assert!(dsdt.windows(8).any(|x| x == b"RSCV0002"));
    assert!(!dsdt.windows(8).any(|x| x == b"RSCV0001"));
}
