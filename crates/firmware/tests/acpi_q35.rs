// SPDX-License-Identifier: GPL-2.0-or-later

//! The q35 tables against the expected blobs QEMU's bios-tables-test uses.

mod common;

use ruvm_firmware::acpi::BuildTables;
use ruvm_firmware::acpi::devices::IsaDevice;
use ruvm_firmware::acpi::linker::Command;
use ruvm_firmware::acpi::pci::CrsRange;
use ruvm_firmware::acpi::q35::{self, PciDevice, PciDeviceAml, Q35Acpi};
use ruvm_firmware::acpi::table::{APPNAME6, APPNAME8, McfgInfo, TABLE_FILE};
use ruvm_firmware::acpi::tpm::{TpmInfo, TpmInterface, TpmVersion};
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
        pci_root_uid: 0,
        pcihp_bridge: true,
        pcihp_io_base: 0x0cc0,
        pcihp_io_len: 0x18,
        smi_on_cpuhp: false,
        smi_on_cpu_unplug: false,
        cpu_hp_io_base: 0x0cd8,
        s3_disabled: false,
        s4_disabled: false,
        s4_val: 2,
        // 128 MiB of RAM, so the hole starts right above it and ends below the IOAPIC.
        pci_hole: CrsRange { base: 0x0800_0000, limit: 0xfebf_ffff },
        pci_hole64: Some(CrsRange { base: 1 << 32, limit: (1 << 32) + (32 << 30) - 1 }),
        pci_devices: vec![
            plain(0),
            PciDevice { devfn: 0x08, acpi_index: None, aml: PciDeviceAml::Vga { qxl: false } },
            PciDevice {
                devfn: 0xf8,
                acpi_index: None,
                aml: PciDeviceAml::Lpc {
                    // qbus_build_aml() walks the children newest first.
                    isa: vec![
                        IsaDevice::I8042 { kbd_irq: 1, mouse_irq: 12 },
                        IsaDevice::Parallel { index: 0, iobase: 0x378, irq: 7 },
                        IsaDevice::Serial { index: 0, iobase: 0x3f8, irq: 4 },
                        IsaDevice::Rtc { io_base: 0x70, irq: 8 },
                    ],
                },
            },
            plain(0xfa),
            plain(0xfb),
        ],
        ivrs: None,
        dmar: None,
        viot: None,
        tpm: None,
    }
}

fn plain(devfn: u8) -> PciDevice {
    PciDevice { devfn, acpi_index: None, aml: PciDeviceAml::Plain }
}

fn indexed(devfn: u8, index: u32) -> PciDevice {
    PciDevice { devfn, acpi_index: Some(index), aml: PciDeviceAml::Plain }
}

fn bridge(devfn: u8, devices: Vec<PciDevice>) -> PciDevice {
    PciDevice { devfn, acpi_index: None, aml: PciDeviceAml::Bridge { devices } }
}

fn build(q: &Q35Acpi) -> BuildTables {
    q35::build(q)
}

fn check(q: &Q35Acpi, sig: &str, file: &str) {
    common::check(&build(q), "q35", sig, file);
}

#[test]
fn facs() {
    check(&base(), "FACS", "FACS");
}

#[test]
fn dsdt() {
    check(&base(), "DSDT", "DSDT");
}

#[test]
fn dsdt_nohpet() {
    let mut q = base();
    q.hpet = false;
    check(&q, "DSDT", "DSDT.nohpet");
}

#[test]
fn dsdt_noacpihp() {
    let mut q = base();
    q.pcihp_bridge = false;
    // Only devices that cannot be unplugged get a static _DSM: those on the root bus, behind
    // the bridge without SHPC and behind the root ports with hotplug off.
    let extra = [
        indexed(0x18, 101),
        bridge(0x20, vec![plain(0x08)]),
        bridge(0x28, vec![indexed(0x08, 301)]),
        bridge(0x30, vec![plain(0)]),
        bridge(0x38, vec![indexed(0, 501)]),
        bridge(0x40, vec![indexed(0x01, 601), bridge(0x02, vec![plain(0)])]),
    ];
    q.pci_devices.splice(2..2, extra);
    check(&q, "DSDT", "DSDT.noacpihp");
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

/// `-device tpm-tis,tpmdev=dev` on top of [`base`], as bios-tables-test runs it.
fn tis(version: TpmVersion) -> Q35Acpi {
    let mut q = base();
    let info = TpmInfo { version, interface: TpmInterface::TisIsa, ppi: true };
    q.tpm = Some(info);
    for d in &mut q.pci_devices {
        if let PciDeviceAml::Lpc { isa } = &mut d.aml {
            isa.insert(0, IsaDevice::TpmTis(info));
        }
    }
    q
}

/// Checks a TPM table after pointing its log area where SeaBIOS put `etc/tpm/log` when
/// the expected blobs were made, right below the 128 MiB of RAM.
fn check_tpm(q: &Q35Acpi, sig: &str, file: &str) {
    let mut t = build(q);
    for cmd in t.linker.commands() {
        if let Command::AddPointer { dest_file, src_file, offset, size: 8 } = cmd {
            if dest_file == TABLE_FILE && src_file == "etc/tpm/log" {
                let o = offset as usize;
                let v = u64::from_le_bytes(t.table_data[o..o + 8].try_into().unwrap());
                t.table_data[o..o + 8].copy_from_slice(&(v + 0x07ff_0000).to_le_bytes());
            }
        }
    }
    common::check(&t, "q35", sig, file);
}

#[test]
fn dsdt_tis_tpm2() {
    check(&tis(TpmVersion::V2_0), "DSDT", "DSDT.tis.tpm2");
}

#[test]
fn tpm2_tis() {
    check_tpm(&tis(TpmVersion::V2_0), "TPM2", "TPM2.tis.tpm2");
}

#[test]
fn dsdt_tis_tpm12() {
    check(&tis(TpmVersion::V1_2), "DSDT", "DSDT.tis.tpm12");
}

#[test]
fn tcpa_tis() {
    check_tpm(&tis(TpmVersion::V1_2), "TCPA", "TCPA.tis.tpm12");
}

#[test]
fn tpm_table_order_and_log() {
    for (version, sig) in [(TpmVersion::V2_0, "TPM2"), (TpmVersion::V1_2, "TCPA")] {
        let t = build(&tis(version));
        let sigs: Vec<String> = load(&t).into_iter().map(|(s, _)| s).collect();
        assert_eq!(sigs, ["FACS", "DSDT", "FACP", "APIC", "HPET", sig, "MCFG", "WAET", "RSDT"]);
        assert_eq!(t.tcpalog, vec![0; 64 << 10]);
        assert!(t.linker.commands().iter().any(
            |c| matches!(c, Command::Allocate { file, align: 1, fseg: false } if file == "etc/tpm/log")
        ));
    }
}
