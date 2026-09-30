// SPDX-License-Identifier: GPL-2.0-or-later

//! The microvm tables against the expected blobs QEMU's bios-tables-test uses.

mod common;

use ruvm_firmware::acpi::devices::{IsaDevice, ged_event};
use ruvm_firmware::acpi::linker::Command;
use ruvm_firmware::acpi::microvm::{self, MicrovmAcpi};
use ruvm_firmware::acpi::table::{APPNAME6, APPNAME8, TABLE_FILE};
use ruvm_firmware::acpi::x86::{MadtConfig, PossibleCpu};

use common::load;

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

fn check(m: &MicrovmAcpi, sig: &str, file: &str) {
    common::check(&microvm::build(m), "microvm", sig, file);
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
