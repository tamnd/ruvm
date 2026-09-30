// SPDX-License-Identifier: GPL-2.0-or-later

//! The fw_cfg contents of the boards, compared with what QEMU 11.1 itself exposes.
//!
//! The data under `tests/data/qemu-11.1` was read out of a real `qemu-system-x86_64` 11.1
//! with `scripts/qemu-fw-cfg-dump.py`, which runs QEMU under qtest (no guest code, no KVM)
//! and pulls every fw_cfg file through the DMA interface. Each case directory holds
//! `dir.txt` (the file directory as select, size and name), `keys.txt` (the legacy items)
//! and `files/`, with option ROMs left out and blobs over 1 KiB stripped of their trailing
//! zeros. To refresh a case, run for example
//!
//! ```text
//! scripts/qemu-fw-cfg-dump.py crates/machine-x86/tests/data/qemu-11.1/q35 -- -M q35 -nodefaults
//! ```
//!
//! The case names say which command line produced them.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ruvm_firmware::smbios::{SmbiosOptions, parse_uuid};
use ruvm_machine_x86::q35::{CpuIdent, KVMVAPIC_ROM, SmbiosEntryPointType};
use ruvm_machine_x86::{Microvm, MicrovmConfig, MicrovmProps, Q35, Q35MachineConfig, Q35Props};
use ruvm_mem::{AddressSpace, Endian, MemTxAttrs};

const ATTRS: MemTxAttrs = MemTxAttrs::UNSPECIFIED;

/// The size of the kvmvapic.bin QEMU 11.1 ships, which is what the directory lists.
const KVMVAPIC_SIZE: usize = 9216;

/// The default CPU the goldens were taken with, `qemu64` under qtest (so TCG rules): an
/// AuthenticAMD CPU, family 15 model 107 stepping 1. The SMBIOS type 4 tables carry both
/// words.
const QEMU64_TCG: CpuIdent =
    CpuIdent { amd: true, version: 0x0006_0fb1, features_edx: 0x078b_fbfd };

fn case_dir(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/qemu-11.1").join(case)
}

/// `dir.txt`: `(select, size, name)` in directory order.
fn golden_dir(case: &str) -> Vec<(u16, u32, String)> {
    let text = std::fs::read_to_string(case_dir(case).join("dir.txt")).unwrap();
    text.lines()
        .map(|l| {
            let mut f = l.split_whitespace();
            let select = u16::from_str_radix(f.next().unwrap().trim_start_matches("0x"), 16);
            let size = f.next().unwrap().parse().unwrap();
            (select.unwrap(), size, f.next().unwrap().to_string())
        })
        .collect()
}

/// `keys.txt`: the legacy items by key.
fn golden_keys(case: &str) -> BTreeMap<u16, Vec<u8>> {
    let text = std::fs::read_to_string(case_dir(case).join("keys.txt")).unwrap();
    text.lines()
        .map(|l| {
            let (k, v) = l.split_once(' ').unwrap();
            let key = u16::from_str_radix(k.trim_start_matches("0x"), 16).unwrap();
            let bytes = (0..v.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&v[i..i + 2], 16).unwrap())
                .collect();
            (key, bytes)
        })
        .collect()
}

fn golden_file(case: &str, name: &str) -> Vec<u8> {
    std::fs::read(case_dir(case).join("files").join(name)).unwrap()
}

fn fw_cfg_read(io: &AddressSpace, key: u16, len: usize) -> Vec<u8> {
    assert!(io.store(0x510, 2, key.into(), Endian::Little, ATTRS).is_ok());
    (0..len)
        .map(|_| {
            let (v, r) = io.load(0x511, 1, Endian::Little, ATTRS);
            assert!(r.is_ok());
            v as u8
        })
        .collect()
}

fn fw_cfg_dir(io: &AddressSpace) -> Vec<(u16, u32, String)> {
    let count = u32::from_be_bytes(fw_cfg_read(io, 0x19, 4).try_into().unwrap()) as usize;
    let dir = fw_cfg_read(io, 0x19, 4 + count * 64);
    dir[4..]
        .chunks(64)
        .map(|e| {
            let size = u32::from_be_bytes(e[0..4].try_into().unwrap());
            let select = u16::from_be_bytes(e[4..6].try_into().unwrap());
            let name = e[8..].split(|&b| b == 0).next().unwrap();
            (select, size, String::from_utf8(name.to_vec()).unwrap())
        })
        .collect()
}

fn trim(mut b: Vec<u8>) -> Vec<u8> {
    while b.last() == Some(&0) {
        b.pop();
    }
    b
}

/// Checks the directory, every file QEMU has and the legacy keys `keys` against `case`.
fn check(io: &AddressSpace, case: &str, keys: &[u16]) {
    let got = fw_cfg_dir(io);
    assert_eq!(got, golden_dir(case), "{case}: directory");
    // Mismatching files are written under the test's temporary directory, so they can be
    // compared with the goldens using the usual tools (acpidump, dtc, xxd).
    let mut bad = Vec::new();
    for (select, size, name) in &got {
        if name.starts_with("genroms/") {
            continue;
        }
        let mut data = fw_cfg_read(io, *select, *size as usize);
        if *size > 1024 {
            data = trim(data);
        }
        if data != golden_file(case, name) {
            let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("qemu-11.1").join(case);
            let path = path.join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, &data).unwrap();
            bad.push(format!("{name} (ruvm's copy is in {})", path.display()));
        }
    }
    assert!(bad.is_empty(), "{case}: contents differ: {bad:#?}");
    let golden = golden_keys(case);
    for key in keys {
        let want = &golden[key];
        assert_eq!(&fw_cfg_read(io, *key, want.len()), want, "{case}: key {key:#x}");
    }
}

fn q35(cfg: Q35MachineConfig) -> Q35 {
    let mut cfg = cfg;
    cfg.firmware.get_or_insert_with(|| vec![0; 256 * 1024]);
    // -nodefaults: no serial port.
    cfg.serial_hds = Vec::new();
    cfg.cpu = QEMU64_TCG;
    cfg.rom_files.insert(KVMVAPIC_ROM.into(), vec![0; KVMVAPIC_SIZE]);
    let mut m = Q35::new(cfg).unwrap();
    m.machine_done().unwrap();
    assert!(m.warnings().is_empty(), "{:?}", m.warnings());
    // What SeaBIOS does before it reads the tables, and so what the capture script does:
    // PMBASE and ACPI enable in the LPC bridge, then PCIEXBAR at 0xb0000000 in the MCH.
    let pci_write32 = |devfn: u32, reg: u32, v: u32| {
        let io = m.io_as();
        let addr = 0x8000_0000 | devfn << 8 | reg;
        assert!(io.store(0xcf8, 4, addr.into(), Endian::Little, ATTRS).is_ok());
        assert!(io.store(0xcfc, 4, v.into(), Endian::Little, ATTRS).is_ok());
    };
    pci_write32(0xf8, 0x40, 0x601);
    pci_write32(0xf8, 0x44, 0x80);
    pci_write32(0x00, 0x64, 0);
    pci_write32(0x00, 0x60, 0xb000_0001);
    m
}

fn microvm(opts: &str) -> Microvm {
    let mut props = MicrovmProps::default();
    props.set_all(opts).unwrap();
    let cfg = MicrovmConfig {
        props,
        firmware: Some(vec![0; 64 * 1024]),
        // -nodefaults: no serial port.
        serial_hd: false,
        ..MicrovmConfig::default()
    };
    let mut m = Microvm::new(cfg).unwrap();
    m.machine_done().unwrap();
    m
}

/// The legacy items both boards set: signature, id, uuid, RAM size, nographic, CPU counts,
/// boot menu and IRQ0 override.
const COMMON_KEYS: &[u16] = &[0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x0e, 0x0f];

#[test]
fn q35_nodefaults() {
    let m = q35(Q35MachineConfig::default());
    check(m.io_as(), "q35", &[COMMON_KEYS, &[0x8002]].concat());
}

#[test]
fn q35_4g_smp4() {
    let m = q35(Q35MachineConfig { ram_size: 4 << 30, cpus: 4, ..Q35MachineConfig::default() });
    check(m.io_as(), "q35-4g-smp4", &[0x00, 0x01, 0x03, 0x05, 0x0f, 0x8002]);
}

#[test]
fn q35_smbios() {
    // -M q35,smbios-entry-point-type=64 -m 20G -uuid ... and a -smbios of most types.
    let props =
        Q35Props { smbios_entry_point_type: SmbiosEntryPointType::Ep64, ..Q35Props::default() };
    let mut smbios = SmbiosOptions::new();
    for arg in [
        "type=0,vendor=ACME,version=1.2.3,date=01/02/2003,release=4.5,uefi=on",
        "type=1,manufacturer=Maker,product=Prod,version=V1,serial=SER1,sku=SKU1,family=Fam",
        "type=2,manufacturer=BB,product=BP,version=BV,serial=BS,asset=BA,location=BL",
        "type=3,manufacturer=CM,version=CV,serial=CS,asset=CA,sku=CK",
        "type=4,sock_pfx=SOCK,manufacturer=PM,version=PV,serial=PS,asset=PA,part=PP,\
         max-speed=3000,current-speed=2500",
        "type=11,value=hello,value=world",
        "type=17,loc_pfx=DIMMX,bank=BANKX,manufacturer=MM,serial=MS,asset=MA,part=MP,speed=1600",
    ] {
        smbios.add(arg).unwrap();
    }
    let mut cfg = Q35MachineConfig { ram_size: 20 << 30, props, smbios, ..Default::default() };
    cfg.fw_cfg.uuid = parse_uuid("12345678-9abc-def0-1234-56789abcdef0").unwrap();
    let m = q35(cfg);
    check(m.io_as(), "q35-smbios", &[0x02, 0x03]);
}

#[test]
fn microvm_nodefaults() {
    let m = microvm("");
    check(m.io_as(), "microvm", COMMON_KEYS);
}

#[test]
fn microvm_acpi_off() {
    let m = microvm("acpi=off");
    check(m.io_as(), "microvm-acpi-off", COMMON_KEYS);
}

#[test]
fn q35_props_default_is_pc_q35_11_1() {
    // The golden cases rely on the defaults of the newest machine type.
    let p = Q35Props::default();
    assert!(p.acpi_enabled());
}
