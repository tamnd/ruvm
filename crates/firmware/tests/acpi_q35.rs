// SPDX-License-Identifier: GPL-2.0-or-later

//! The fixed q35 tables against the expected blobs QEMU's bios-tables-test uses. The DSDT is
//! not built yet, so these tests pass an empty one.

mod common;

use ruvm_firmware::acpi::BuildTables;
use ruvm_firmware::acpi::aml::Aml;
use ruvm_firmware::acpi::linker::Command;
use ruvm_firmware::acpi::q35::{self, Q35Acpi};
use ruvm_firmware::acpi::table::{APPNAME6, APPNAME8, McfgInfo, TABLE_FILE};
use ruvm_firmware::acpi::x86::{MadtConfig, PossibleCpu};

use common::load;

/// `-machine q35` with the defaults bios-tables-test uses.
fn base() -> Q35Acpi {
    Q35Acpi {
        oem_id: APPNAME6.into(),
        oem_table_id: APPNAME8.into(),
        madt: MadtConfig {
            cpus: vec![PossibleCpu { arch_id: 0, present: true }],
            pic: true,
            ioapic2: false,
            apic_xrupt_override: true,
            // PIRQ E to H are routed to GSIs 16 to 23, A to D sit on IRQs 5, 9, 10 and 11.
            pci_irq_mask: 1 << 5 | 1 << 9 | 1 << 10 | 1 << 11,
        },
        max_cpus: 1,
        smm: true,
        sci_int: 9,
        pm_io_base: 0x600,
        i8042: true,
        hpet: true,
        mcfg: Some(McfgInfo { base: 0xb000_0000, size: 256 << 20 }),
    }
}

fn build(q: &Q35Acpi) -> BuildTables {
    q35::build(q, &Aml::new())
}

fn check(q: &Q35Acpi, sig: &str, file: &str) {
    common::check(&build(q), "q35", sig, file);
}

#[test]
fn facs() {
    check(&base(), "FACS", "FACS");
}

#[test]
fn facp() {
    check(&base(), "FACP", "FACP");
}

#[test]
fn facp_nosmm() {
    let mut q = base();
    q.smm = false;
    check(&q, "FACP", "FACP.nosmm");
}

#[test]
fn apic() {
    check(&base(), "APIC", "APIC");
}

#[test]
fn hpet() {
    check(&base(), "HPET", "HPET");
}

#[test]
fn mcfg() {
    check(&base(), "MCFG", "MCFG");
}

#[test]
fn waet() {
    check(&base(), "WAET", "WAET");
}

#[test]
fn table_order_and_sizes() {
    let t = build(&base());
    let sigs: Vec<String> = load(&t).into_iter().map(|(s, _)| s).collect();
    assert_eq!(sigs, ["FACS", "DSDT", "FACP", "APIC", "HPET", "MCFG", "WAET", "RSDT"]);
    assert_eq!(t.table_data.len(), q35::TABLE_SIZE);
    assert_eq!(t.linker.cmd_blob().len() % q35::ALIGN_SIZE, 0);
    assert_eq!(t.rsdp.len(), 20);
    let cmds = t.linker.commands();
    assert!(
        matches!(&cmds[1], Command::Allocate { file, align: 64, fseg: false } if file == TABLE_FILE)
    );
}

#[test]
fn no_hpet_no_mcfg() {
    let mut q = base();
    q.hpet = false;
    q.mcfg = None;
    let sigs: Vec<String> = load(&build(&q)).into_iter().map(|(s, _)| s).collect();
    assert_eq!(sigs, ["FACS", "DSDT", "FACP", "APIC", "WAET", "RSDT"]);
}
