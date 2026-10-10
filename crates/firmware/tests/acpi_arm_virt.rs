// SPDX-License-Identifier: GPL-2.0-or-later

//! The aarch64 `virt` tables against the expected blobs QEMU's bios-tables-test uses. Those
//! come from `-M virt -cpu cortex-a57` with EDK2 in the first flash, one CPU, 128 MiB of RAM
//! and the test image on a virtio-blk-pci disk in slot 1. The base test runs a GICv2, so its
//! MADT and IORT are not ruvm's; the GICv3 ones come from the `its_off` variant and from
//! cases built here by hand.

mod common;

use ruvm_firmware::acpi::arm_virt::{
    self, ArmVirtAcpi, GicV2Bases, GicV2mFrame, IortSmmu, NumaNode, PossibleCpu, PsciConduit,
    VIRTUAL_PMU_IRQ, VirtIrqs, VirtMemmap,
};
use ruvm_firmware::acpi::gpex::Window;
use ruvm_firmware::acpi::linker::Command;
use ruvm_firmware::acpi::q35::{PciDevice, PciDeviceAml};
use ruvm_firmware::acpi::table::{APPNAME6, APPNAME8, RSDP_FILE, TABLE_FILE};

use common::load;

const MACHINE: &str = "virt";

fn w(base: u64, size: u64) -> Window {
    Window { base, size }
}

fn pci(devfn: u8) -> PciDevice {
    PciDevice { devfn, acpi_index: None, aml: PciDeviceAml::Plain }
}

fn base() -> ArmVirtAcpi {
    ArmVirtAcpi {
        oem_id: APPNAME6.into(),
        oem_table_id: APPNAME8.into(),
        memmap: VirtMemmap {
            uart0: w(0x0900_0000, 0x1000),
            uart1: None,
            fw_cfg: w(0x0902_0000, 0x18),
            virtio: w(0x0a00_0000, 0x200),
            ecam: w(0x40_1000_0000, 0x1000_0000),
            pcie_mmio: w(0x1000_0000, 0x2eff_0000),
            pcie_pio: w(0x3eff_0000, 0x1_0000),
            pcie_mmio_high: w(0x80_0000_0000, 0x80_0000_0000),
            gic_dist: 0x0800_0000,
            gic_redist: w(0x080a_0000, 0x00f6_0000),
            gic_redist2: None,
            gic_its: Some(0x0808_0000),
            gic_v2: None,
            gic_v2m: None,
            acpi_ged: 0x0908_0000,
            gpio: w(0x0903_0000, 0x1000),
            mem: 0x4000_0000,
        },
        irqs: VirtIrqs { uart0: 33, uart1: 40, virtio: 48, pcie: 35, acpi_ged: 41, gpio: 39 },
        virtio_count: 32,
        mpidrs: vec![0],
        possible_cpus: arm_virt::possible_cpus(1, 1, 1, 1),
        has_clusters: false,
        threads: 1,
        pmu_irq: VIRTUAL_PMU_IRQ,
        virtualization: false,
        ns_el2_virt_timer: false,
        psci: PsciConduit::Hvc,
        ged_events: Some(0x22),
        spcr: true,
        pci_devices: vec![pci(0), pci(8)],
        numa: Vec::new(),
        smmu: None,
    }
}

/// `-smp 4,sockets=2` with the CPUs split over NUMA nodes 0 and 1, and three nodes of
/// 128 MiB each, the `acpihmatvirt` test.
fn hmat() -> ArmVirtAcpi {
    let mut m = base();
    m.mpidrs = vec![0, 1, 2, 3];
    m.possible_cpus = arm_virt::possible_cpus(2, 1, 2, 1);
    for (i, cpu) in m.possible_cpus.iter_mut().enumerate() {
        cpu.node = (i / 2) as u32;
    }
    m.numa = (0..3).map(|_| NumaNode { mem: 0x800_0000, distance: Vec::new() }).collect();
    m
}

fn check(m: &ArmVirtAcpi, sig: &str, file: &str) {
    common::check_arch(&arm_virt::build(m), "aarch64", MACHINE, sig, file);
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
fn gtdt() {
    check(&base(), "GTDT", "GTDT");
}

#[test]
fn mcfg() {
    check(&base(), "MCFG", "MCFG");
}

#[test]
fn spcr() {
    check(&base(), "SPCR", "SPCR");
}

#[test]
fn dbg2() {
    check(&base(), "DBG2", "DBG2");
}

#[test]
fn pptt() {
    check(&base(), "PPTT", "PPTT");
}

/// Without an ITS the IORT has the root complex alone, with no ID mappings. The base test
/// gets this from its GICv2m.
#[test]
fn iort_without_its() {
    let mut m = base();
    m.memmap.gic_its = None;
    check(&m, "IORT", "IORT");
}

/// `-M gic-version=3,its=off`: the MADT of a GICv3 with no ITS structure.
#[test]
fn apic_its_off() {
    let mut m = base();
    m.memmap.gic_its = None;
    check(&m, "APIC", "APIC.its_off");
}

/// `-object memory-backend-ram,id=ram0,size=128M -numa node,memdev=ram0`.
#[test]
fn srat_numamem() {
    let mut m = base();
    m.numa = vec![NumaNode { mem: 0x800_0000, distance: Vec::new() }];
    let t = arm_virt::build(&m);
    let sigs: Vec<String> = load(&t).into_iter().map(|(s, _)| s).collect();
    assert!(!sigs.contains(&"SLIT".to_string()), "{sigs:?}");
    common::check_arch(&t, "aarch64", MACHINE, "SRAT", "SRAT.numamem");
}

#[test]
fn acpihmatvirt() {
    let m = hmat();
    check(&m, "DSDT", "DSDT.acpihmatvirt");
    check(&m, "PPTT", "PPTT.acpihmatvirt");
    check(&m, "SRAT", "SRAT.acpihmatvirt");
}

/// `-smp sockets=1,clusters=2,cores=2,threads=2`: eight CPU devices. The PPTT of that test
/// has cache nodes, which ruvm does not build.
#[test]
fn dsdt_topology() {
    let mut m = base();
    m.mpidrs = (0..8).collect();
    m.possible_cpus = arm_virt::possible_cpus(1, 2, 2, 2);
    m.has_clusters = true;
    m.threads = 2;
    check(&m, "DSDT", "DSDT.topology");
}

/// The processor nodes of `-smp sockets=1,clusters=2,cores=2,threads=2` with no caches: root,
/// socket, then per cluster the cluster node, and per core the core node and its two threads.
#[test]
fn pptt_threads() {
    let mut m = base();
    m.mpidrs = (0..8).collect();
    m.possible_cpus = arm_virt::possible_cpus(1, 2, 2, 2);
    m.has_clusters = true;
    m.threads = 2;
    let pptt = common::table(&load(&arm_virt::build(&m)), "PPTT");
    let nodes: Vec<(u32, u32, u32)> = pptt[36..]
        .chunks(20)
        .map(|n| {
            let r = |o: usize| u32::from_le_bytes(n[o..o + 4].try_into().unwrap());
            (r(4), r(8), r(12))
        })
        .collect();
    let off = |i: u32| 36 + 20 * i;
    assert_eq!(
        nodes,
        [
            (0x11, 0, 0),
            (0x11, off(0), 0),
            (0x10, off(1), 0),
            (0x10, off(2), 0),
            (0xE, off(3), 0),
            (0xE, off(3), 1),
            (0x10, off(2), 1),
            (0xE, off(6), 2),
            (0xE, off(6), 3),
            (0x10, off(1), 1),
            (0x10, off(9), 0),
            (0xE, off(10), 4),
            (0xE, off(10), 5),
            (0x10, off(9), 1),
            (0xE, off(13), 6),
            (0xE, off(13), 7),
        ]
    );
}

/// The order of the tables, and the XSDT with the PPTT listed twice for the missing WDAT.
#[test]
fn layout() {
    let t = arm_virt::build(&base());
    let tables = load(&t);
    let sigs: Vec<&str> = tables.iter().map(|(s, _)| s.as_str()).collect();
    assert_eq!(
        sigs,
        ["DSDT", "FACP", "APIC", "PPTT", "GTDT", "MCFG", "SPCR", "DBG2", "IORT", "XSDT"]
    );
    assert_eq!(t.table_data.len(), 0x20000);
    let xsdt = common::table(&tables, "XSDT");
    let entries: Vec<u64> =
        xsdt[36..].chunks(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(entries.len(), 9);
    assert_eq!(entries[2], entries[3]);
    assert_eq!(t.rsdp.len(), 36);
    assert_eq!(&t.rsdp[..8], b"RSD PTR ");
    assert_eq!(t.rsdp[15], 2, "RSDP revision");
    let cmds = t.linker.commands();
    assert!(cmds.iter().any(|c| matches!(c, Command::Allocate { file, .. } if file == TABLE_FILE)));
    assert!(cmds.iter().any(|c| matches!(c, Command::Allocate { file, .. } if file == RSDP_FILE)));
}

/// With `spcr=off` QEMU still reserves the SPCR's XSDT entry, so the DBG2 is listed twice.
#[test]
fn no_spcr() {
    let mut m = base();
    m.spcr = false;
    let tables = load(&arm_virt::build(&m));
    let sigs: Vec<&str> = tables.iter().map(|(s, _)| s.as_str()).collect();
    assert_eq!(sigs, ["DSDT", "FACP", "APIC", "PPTT", "GTDT", "MCFG", "DBG2", "IORT", "XSDT"]);
    let xsdt = common::table(&tables, "XSDT");
    let entries: Vec<u64> =
        xsdt[36..].chunks(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(entries.len(), 9);
    assert_eq!(entries[6], entries[7]);
}

/// The GICv3 MADT with the ITS, a second redistributor region, the maintenance interrupt and
/// no PMU, and the IORT mapping every requester ID to the ITS group.
#[test]
fn madt_and_iort_with_its() {
    let mut m = base();
    m.mpidrs = vec![0, 1];
    m.possible_cpus = arm_virt::possible_cpus(1, 1, 2, 1);
    m.memmap.gic_redist2 = Some(w(0x40_0000_0000, 0x400_0000));
    m.pmu_irq = 0;
    m.virtualization = true;
    let tables = load(&arm_virt::build(&m));
    let apic = common::table(&tables, "APIC");
    assert_eq!(apic.len(), 44 + 24 + 2 * 80 + 2 * 16 + 20);
    let gicc1 = &apic[44 + 24 + 80..44 + 24 + 160];
    assert_eq!(gicc1[..2], [0xB, 80]);
    assert_eq!(u32::from_le_bytes(gicc1[4..8].try_into().unwrap()), 1);
    assert_eq!(u32::from_le_bytes(gicc1[20..24].try_into().unwrap()), 0, "no PMU");
    assert_eq!(u32::from_le_bytes(gicc1[56..60].try_into().unwrap()), 25, "maintenance");
    assert_eq!(u64::from_le_bytes(gicc1[68..76].try_into().unwrap()), 1, "MPIDR");
    let gicr2 = &apic[44 + 24 + 160 + 16..44 + 24 + 160 + 32];
    assert_eq!(gicr2, [0xE, 16, 0, 0, 0, 0, 0, 0, 0x40, 0, 0, 0, 0, 0, 0, 4]);
    let its = &apic[apic.len() - 20..];
    assert_eq!(its, [0xF, 20, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0]);

    let iort = common::table(&tables, "IORT");
    assert_eq!(iort.len(), 48 + 24 + 36 + 20);
    assert_eq!(u32::from_le_bytes(iort[36..40].try_into().unwrap()), 2, "nodes");
    let group = &iort[48..72];
    assert_eq!(group[..4], [0, 24, 0, 1]);
    let rc = &iort[72..];
    assert_eq!(rc[..4], [2, 56, 0, 3]);
    assert_eq!(u32::from_le_bytes(rc[4..8].try_into().unwrap()), 1, "identifier");
    assert_eq!(u32::from_le_bytes(rc[8..12].try_into().unwrap()), 1, "mappings");
    let map = &rc[36..];
    let r = |o: usize| u32::from_le_bytes(map[o..o + 4].try_into().unwrap());
    assert_eq!([r(0), r(4), r(8), r(12), r(16)], [0, 0xffff, 0, 48, 0]);
}

/// The `gic-version=2` MADT: a version 2 distributor, the CPU interface bases in each GICC,
/// no GICR or ITS, and the GICv2m frame.
#[test]
fn madt_gicv2_with_v2m() {
    let mut m = base();
    m.mpidrs = vec![0, 1];
    m.possible_cpus = arm_virt::possible_cpus(1, 1, 2, 1);
    m.memmap.gic_its = None;
    m.memmap.gic_v2 = Some(GicV2Bases { cpu: 0x0801_0000, vcpu: 0x0804_0000, hyp: 0x0803_0000 });
    m.memmap.gic_v2m = Some(GicV2mFrame { base: 0x0802_0000, spi_base: 80, spi_count: 64 });
    let tables = load(&arm_virt::build(&m));
    let apic = common::table(&tables, "APIC");
    assert_eq!(apic.len(), 44 + 24 + 2 * 80 + 24);
    assert_eq!(apic[44 + 20], 2, "GIC version");
    let gicc1 = &apic[44 + 24 + 80..44 + 24 + 160];
    let q = |o: usize| u64::from_le_bytes(gicc1[o..o + 8].try_into().unwrap());
    assert_eq!([q(32), q(40), q(48)], [0x0801_0000, 0x0804_0000, 0x0803_0000]);
    assert_eq!(q(60), 0, "no GICR base");
    let frame = &apic[apic.len() - 24..];
    assert_eq!(frame[..8], [0xD, 24, 0, 0, 0, 0, 0, 0]);
    assert_eq!(u64::from_le_bytes(frame[8..16].try_into().unwrap()), 0x0802_0000);
    assert_eq!(frame[16..], [1, 0, 0, 0, 64, 0, 80, 0]);
}

/// The `smmuv3-legacy` test: `iommu=smmuv3` with a GICv2, so no ITS, and three host bridges,
/// the root bus with a root port (buses 0 and 1), a pxb-pcie at bus 0x10 and one at 0x20 that
/// bypasses the IOMMU. The SMMU node has no ID mappings and the root complex sends the first
/// two bus ranges to it.
#[test]
fn iort_smmuv3_legacy() {
    let mut m = base();
    m.memmap.gic_its = None;
    m.smmu = Some(IortSmmu {
        base: 0x0905_0000,
        gsi: 74 + 32,
        rc_id_maps: vec![(0, 0x200), (0x1000, 0x100)],
    });
    check(&m, "IORT", "IORT.smmuv3-legacy");
}

/// `iommu=smmuv3` with the ITS: the SMMU maps its IDs to the ITS group, and the root complex
/// sends bus 0 to the SMMU and the rest straight to the ITS.
#[test]
fn iort_smmuv3_with_its() {
    let mut m = base();
    m.smmu = Some(IortSmmu { base: 0x0905_0000, gsi: 106, rc_id_maps: vec![(0, 0x100)] });
    let iort = common::table(&load(&arm_virt::build(&m)), "IORT");
    let r = |o: usize| u32::from_le_bytes(iort[o..o + 4].try_into().unwrap());
    assert_eq!(iort.len(), 48 + 24 + 88 + 36 + 2 * 20);
    assert_eq!(r(36), 3, "nodes");
    let smmu = 72;
    assert_eq!(iort[smmu..smmu + 4], [4, 88, 0, 4]);
    assert_eq!([r(smmu + 8), r(smmu + 12)], [1, 68], "ID mappings");
    assert_eq!(r(smmu + 16), 0x0905_0000);
    assert_eq!([r(smmu + 44), r(smmu + 48), r(smmu + 52), r(smmu + 56)], [106, 107, 109, 108]);
    let map = smmu + 68;
    assert_eq!([r(map), r(map + 4), r(map + 8), r(map + 12)], [0, 0xffff, 0, 48]);
    let rc = smmu + 88;
    assert_eq!(iort[rc..rc + 4], [2, 76, 0, 3]);
    assert_eq!([r(rc + 4), r(rc + 8)], [2, 2], "identifier and mappings");
    let maps = rc + 36;
    assert_eq!([r(maps), r(maps + 4), r(maps + 8), r(maps + 12)], [0, 0xff, 0, 72]);
    assert_eq!(
        [r(maps + 20), r(maps + 24), r(maps + 28), r(maps + 32)],
        [0x100, 0xfeff, 0x100, 48]
    );
}

/// Without the GED, as when a kernel is booted with no firmware, the PL061 raises the power
/// button through an `_AEI` event on pin 3.
#[test]
fn dsdt_gpio_power_button() {
    let mut m = base();
    m.ged_events = None;
    let dsdt = common::table(&load(&arm_virt::build(&m)), "DSDT");
    let find = |needle: &[u8]| dsdt.windows(needle.len()).position(|w| w == needle);
    assert!(find(b"GPO0").is_some());
    assert!(find(b"ARMH0061").is_some());
    assert!(find(b"_E03").is_some());
    assert!(find(b"_AEI").is_some());
    assert!(find(b"GED_").is_none());
    // The GPIO Connection descriptor: interrupt type, edge, active high, exclusive, pull up,
    // pin 3, then the resource source "GPO0".
    let gpio = find(&[0x8C, 0x1B, 0x00, 0x01, 0x00]).expect("GpioInt descriptor");
    let d = &dsdt[gpio..gpio + 30];
    assert_eq!(d[5..9], [1, 0, 1, 0], "consumer, edge");
    assert_eq!(d[9], 1, "pull up");
    assert_eq!(d[23..25], [3, 0], "pin");
    assert_eq!(&d[25..30], b"GPO0\0");
}

/// The FADT's boot flags follow the PSCI conduit.
#[test]
fn fadt_psci() {
    for (psci, flags) in
        [(PsciConduit::Disabled, 0u16), (PsciConduit::Hvc, 3), (PsciConduit::Smc, 1)]
    {
        let mut m = base();
        m.psci = psci;
        let facp = common::table(&load(&arm_virt::build(&m)), "FACP");
        assert_eq!(u16::from_le_bytes([facp[129], facp[130]]), flags, "{psci:?}");
    }
}

/// Two NUMA nodes with distances give an SLIT after the SRAT, before the IORT.
#[test]
fn numa_distances() {
    let mut m = base();
    m.mpidrs = vec![0, 1];
    m.possible_cpus = vec![
        PossibleCpu { socket: 0, node: 0, ..PossibleCpu::default() },
        PossibleCpu { socket: 1, node: 1, ..PossibleCpu::default() },
    ];
    m.numa = vec![
        NumaNode { mem: 0x400_0000, distance: vec![10, 20] },
        NumaNode { mem: 0x400_0000, distance: vec![20, 10] },
    ];
    let tables = load(&arm_virt::build(&m));
    let sigs: Vec<&str> = tables.iter().map(|(s, _)| s.as_str()).collect();
    assert_eq!(
        sigs,
        [
            "DSDT", "FACP", "APIC", "PPTT", "GTDT", "MCFG", "SPCR", "DBG2", "SRAT", "SLIT", "IORT",
            "XSDT"
        ]
    );
    let slit = common::table(&tables, "SLIT");
    assert_eq!(&slit[36..], [2, 0, 0, 0, 0, 0, 0, 0, 10, 20, 20, 10]);
    let srat = common::table(&tables, "SRAT");
    assert_eq!(srat.len(), 48 + 2 * 18 + 2 * 40);
    let gicc1 = &srat[48 + 18..48 + 36];
    assert_eq!(gicc1[..6], [3, 18, 1, 0, 0, 0]);
    let mem1 = &srat[48 + 36 + 40..];
    assert_eq!(u32::from_le_bytes(mem1[2..6].try_into().unwrap()), 1);
    assert_eq!(u32::from_le_bytes(mem1[8..12].try_into().unwrap()), 0x4400_0000);
}
