// SPDX-License-Identifier: GPL-2.0-or-later

//! The microvm tables against the expected blobs QEMU's bios-tables-test uses.
//!
//! The test loads the tables the way firmware would, runs the checksum commands from the loader
//! script and cleans up the FADT pointers the same way bios-tables-test does before it compares.

use std::path::PathBuf;

use ruvm_firmware::acpi::BuildTables;
use ruvm_firmware::acpi::devices::{IsaDevice, ged_event};
use ruvm_firmware::acpi::linker::Command;
use ruvm_firmware::acpi::microvm::{self, MicrovmAcpi};
use ruvm_firmware::acpi::table::{APPNAME6, APPNAME8, TABLE_FILE, checksum};
use ruvm_firmware::acpi::x86::{MadtConfig, PossibleCpu};

fn expected(name: &str) -> Vec<u8> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../vendor-qemu/acpi-expected/x86/microvm");
    std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// `-machine microvm,acpi=on,ioapic2=off,rtc=off` with one virtio-blk-device, which lands on the
/// last of the eight transports.
fn base() -> MicrovmAcpi {
    MicrovmAcpi {
        oem_id: APPNAME6.into(),
        oem_table_id: APPNAME8.into(),
        madt: MadtConfig {
            cpus: vec![PossibleCpu { arch_id: 0, present: true }],
            pic: true,
            ioapic2: false,
            apic_xrupt_override: false,
            pci_irq_mask: 0,
        },
        isa: vec![IsaDevice::Serial { index: 0, iobase: 0x3f8, irq: 4 }],
        ged_events: ged_event::PWR_DOWN,
        virtio_irq_base: 16,
        virtio_transports: vec![7],
        usb: false,
        i8042: false,
    }
}

/// Runs the checksum commands and splits `etc/acpi/tables` into its tables.
fn load(t: &BuildTables) -> Vec<(String, Vec<u8>)> {
    let mut data = t.table_data.clone();
    for cmd in t.linker.commands() {
        if let Command::AddChecksum { file, offset, start, length } = cmd {
            if file == TABLE_FILE {
                let (s, o) = (start as usize, offset as usize);
                data[o] = 0;
                data[o] = checksum(&data[s..s + length as usize]);
            }
        }
    }
    let mut tables = Vec::new();
    let mut at = 0;
    while at < data.len() {
        let len = u32::from_le_bytes(data[at + 4..at + 8].try_into().unwrap()) as usize;
        let sig = String::from_utf8(data[at..at + 4].to_vec()).unwrap();
        let mut table = data[at..at + len].to_vec();
        assert_eq!(checksum(&table), 0, "{sig} checksum");
        if sig == "FACP" {
            // test_acpi_fadt_table() zeroes the pointers the firmware filled in and redoes the sum.
            table[36..44].fill(0);
            if table[8] >= 3 {
                table[132..148].fill(0);
            }
            table[9] = 0;
            table[9] = checksum(&table);
        }
        tables.push((sig, table));
        at += len;
    }
    tables
}

fn table(tables: &[(String, Vec<u8>)], sig: &str) -> Vec<u8> {
    tables.iter().find(|(s, _)| s == sig).unwrap_or_else(|| panic!("no {sig}")).1.clone()
}

fn hex(b: &[u8]) -> String {
    b.chunks(16)
        .enumerate()
        .map(|(i, c)| {
            let bytes: Vec<String> = c.iter().map(|x| format!("{x:02x}")).collect();
            format!("{:04x}: {}\n", i * 16, bytes.join(" "))
        })
        .collect()
}

fn check(m: &MicrovmAcpi, sig: &str, file: &str) {
    let got = table(&load(&microvm::build(m)), sig);
    let want = expected(file);
    assert!(got == want, "{file} differs\ngot:\n{}\nwant:\n{}", hex(&got), hex(&want));
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
fn dsdt_rtc() {
    let mut m = base();
    m.isa.push(IsaDevice::Rtc { io_base: 0x70, irq: 8 });
    check(&m, "DSDT", "DSDT.rtc");
}

#[test]
fn dsdt_usb() {
    let mut m = base();
    m.usb = true;
    check(&m, "DSDT", "DSDT.usb");
}

/// With the second IOAPIC the virtio-mmio transports move to it and there are 24 of them.
fn ioapic2() -> MicrovmAcpi {
    let mut m = base();
    m.madt.ioapic2 = true;
    m.virtio_irq_base = 24;
    m.virtio_transports = vec![23];
    m
}

#[test]
fn dsdt_ioapic2() {
    check(&ioapic2(), "DSDT", "DSDT.ioapic2");
}

#[test]
fn apic_ioapic2() {
    check(&ioapic2(), "APIC", "APIC.ioapic2");
}

#[test]
fn table_order_and_loader_script() {
    let t = microvm::build(&base());
    let sigs: Vec<String> = load(&t).into_iter().map(|(s, _)| s).collect();
    assert_eq!(sigs, ["DSDT", "FACP", "APIC", "XSDT"]);
    let cmds = t.linker.commands();
    assert!(
        matches!(&cmds[0], Command::Allocate { file, fseg: true, .. } if file == "etc/acpi/rsdp")
    );
    assert!(
        matches!(&cmds[1], Command::Allocate { file, align: 64, fseg: false } if file == TABLE_FILE)
    );
    assert_eq!(t.rsdp.len(), 36);
}
