// SPDX-License-Identifier: GPL-2.0-or-later

//! The ACPI side of the virt board: its tables against the blobs of QEMU's bios-tables-test,
//! the fw_cfg files the firmware installs them from, the GED, and the kernel handed to the
//! firmware through fw_cfg.

use std::path::PathBuf;

use ruvm_firmware::acpi::BuildTables;
use ruvm_firmware::acpi::linker::Command;
use ruvm_firmware::acpi::table::{LOADER_FILE, RSDP_FILE, TABLE_FILE, TPMLOG_FILE, checksum};
use ruvm_hw_core::fw_cfg::{
    FW_CFG_CMDLINE_DATA, FW_CFG_CMDLINE_SIZE, FW_CFG_INITRD_DATA, FW_CFG_INITRD_SIZE,
    FW_CFG_KERNEL_DATA, FW_CFG_KERNEL_SIZE,
};
use ruvm_hw_virtio::VirtioPciProps;
use ruvm_hw_virtio::rng::{RandomFile, VirtioRng, VirtioRngConf};
use ruvm_machine_arm::virt::{
    CpuTopology, VIRT_ACPI_GED, VIRT_MEM, VirtConfig, VirtMachine, VirtMsi,
};
use ruvm_mem::MemTxAttrs;
use ruvm_target_arm::cpu::ArmCpuModel;

/// A scratch directory, removed when dropped.
struct TmpDir(PathBuf);

impl TmpDir {
    fn new(tag: &str) -> TmpDir {
        let dir = std::env::temp_dir().join(format!("ruvm-virt-acpi-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        TmpDir(dir)
    }

    fn file(&self, name: &str, contents: &[u8]) -> String {
        let p = self.0.join(name);
        std::fs::write(&p, contents).unwrap();
        p.to_str().unwrap().to_string()
    }
}

impl Drop for TmpDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn a57() -> VirtConfig {
    VirtConfig::new(ArmCpuModel::by_name("cortex-a57").unwrap())
}

/// The board of bios-tables-test: cortex-a57, firmware in the first flash and a PCI function
/// in slot 1 (QEMU's is the virtio-blk-pci test disk; any function gives the same tables).
fn qemu_test_board(tmp: &TmpDir, msi: VirtMsi) -> VirtMachine {
    let mut cfg = a57();
    cfg.firmware = Some(tmp.file("fw.fd", &[0; 0x1000]));
    cfg.msi = msi;
    let mut m = VirtMachine::new(cfg).unwrap();
    let rng = VirtioRng::new(Box::new(RandomFile::default()), VirtioRngConf::default());
    m.attach_virtio_pci(Box::new(rng), Some(1 << 3), &VirtioPciProps::default()).unwrap();
    m.machine_done().unwrap();
    m
}

/// The tables in `etc/acpi/tables` with their checksums filled in as the firmware would, and
/// the FADT pointers zeroed as bios-tables-test does.
fn load(t: &BuildTables) -> Vec<(String, Vec<u8>)> {
    let mut data = t.table_data.clone();
    for cmd in t.linker.commands() {
        if let Command::AddChecksum { file, offset, start, length } = cmd {
            if file != TABLE_FILE {
                continue;
            }
            let (o, s) = (offset as usize, start as usize);
            data[o] = 0;
            data[o] = checksum(&data[s..s + length as usize]);
        }
    }
    let mut tables = Vec::new();
    let mut at = 0;
    while at + 8 <= data.len() && data[at..at + 4] != [0; 4] {
        let len = u32::from_le_bytes(data[at + 4..at + 8].try_into().unwrap()) as usize;
        let sig = String::from_utf8(data[at..at + 4].to_vec()).unwrap();
        let mut table = data[at..at + len].to_vec();
        assert_eq!(checksum(&table), 0, "{sig} checksum");
        if sig == "FACP" {
            table[36..44].fill(0);
            table[132..148].fill(0);
            table[9] = 0;
            table[9] = checksum(&table);
        }
        tables.push((sig, table));
        at += len;
    }
    tables
}

fn expected(name: &str) -> Vec<u8> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../vendor-qemu/acpi-expected/aarch64/virt");
    std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn table<'a>(tables: &'a [(String, Vec<u8>)], sig: &str) -> &'a [u8] {
    &tables.iter().find(|(s, _)| s == sig).unwrap_or_else(|| panic!("no {sig}")).1
}

/// The tables of the board against QEMU's. The base test runs a GICv2, so the MADT is checked
/// against the `its_off` variant, whose GICC still has the PMU interrupt of QEMU's cortex-a57;
/// ruvm has no PMU and puts 0 there. The IORT without an ITS is the base test's, where the
/// GICv2m leaves the root complex unmapped too.
#[test]
fn acpi_tables_match_qemu() {
    let tmp = TmpDir::new("match");
    let m = qemu_test_board(&tmp, VirtMsi::Auto);
    let tables = load(&m.acpi_tables());
    let sigs: Vec<&str> = tables.iter().map(|(s, _)| s.as_str()).collect();
    assert_eq!(
        sigs,
        ["DSDT", "FACP", "APIC", "PPTT", "GTDT", "MCFG", "SPCR", "DBG2", "IORT", "XSDT"]
    );
    for sig in ["DSDT", "FACP", "PPTT", "GTDT", "MCFG", "SPCR", "DBG2"] {
        assert!(table(&tables, sig) == expected(sig), "{sig} differs from QEMU's");
    }

    let m = qemu_test_board(&tmp, VirtMsi::Off);
    let tables = load(&m.acpi_tables());
    assert!(table(&tables, "IORT") == expected("IORT"), "IORT differs from QEMU's");
    assert!(table(&tables, "APIC") == expected("APIC.its_off"), "APIC differs from QEMU's");
}

#[test]
fn acpi_tables_go_to_fw_cfg() {
    let names = |m: &VirtMachine| -> Vec<String> {
        m.fw_cfg().state().files().into_iter().map(|(n, _, _)| n).collect()
    };
    let mut m = VirtMachine::new(a57()).unwrap();
    m.machine_done().unwrap();
    let files = m.fw_cfg().state().files();
    for name in [TABLE_FILE, LOADER_FILE, TPMLOG_FILE, RSDP_FILE] {
        let (_, key, size) = files.iter().find(|(n, _, _)| n == name).unwrap().clone();
        let data = m.fw_cfg().state().entry_data(key).unwrap_or_default();
        assert_eq!(data.len(), size as usize, "{name}");
    }
    let tables = files.iter().find(|(n, _, _)| n == TABLE_FILE).unwrap();
    assert_eq!(tables.2, 0x20000);

    let mut cfg = a57();
    cfg.acpi = false;
    let mut m = VirtMachine::new(cfg).unwrap();
    m.machine_done().unwrap();
    assert!(!names(&m).iter().any(|n| n.starts_with("etc/acpi") || n == TPMLOG_FILE));

    let mut cfg = a57();
    cfg.spcr = false;
    cfg.oem_id = "RUVM".to_string();
    cfg.oem_table_id = "RUVMTBL".to_string();
    let m = VirtMachine::new(cfg).unwrap();
    let tables = load(&m.acpi_tables());
    assert!(!tables.iter().any(|(s, _)| s == "SPCR"));
    for (sig, t) in &tables {
        assert_eq!(&t[10..16], b"RUVM\0\0", "{sig}");
        assert_eq!(&t[16..24], b"RUVMTBL\0", "{sig}");
    }
}

#[test]
fn oem_id_errors() {
    let mut cfg = a57();
    cfg.oem_id = "1234567".to_string();
    assert_eq!(
        VirtMachine::new(cfg).unwrap_err(),
        "User specified oem-id value is bigger than 6 bytes in size"
    );
    let mut cfg = a57();
    cfg.oem_table_id = "123456789".to_string();
    assert_eq!(
        VirtMachine::new(cfg).unwrap_err(),
        "User specified oem-table-id value is bigger than 8 bytes in size"
    );
}

/// The GED is there only when firmware boots with ACPI, and it carries the power button.
#[test]
fn ged_with_firmware_only() {
    let tmp = TmpDir::new("ged");
    let fw = tmp.file("fw.fd", &[0; 0x1000]);
    let mut cfg = a57();
    cfg.firmware = Some(fw.clone());
    let mut m = VirtMachine::new(cfg).unwrap();
    m.machine_done().unwrap();
    assert_eq!(m.ged().unwrap().ged_event_bitmap(), 0x22);
    let dsdt = table(&load(&m.acpi_tables()), "DSDT").to_vec();
    assert!(dsdt.windows(4).any(|w| w == b"GED_"));
    m.system_powerdown();
    let mut b = [0u8; 4];
    assert!(m.memory_as().read(VIRT_ACPI_GED, MemTxAttrs::UNSPECIFIED, &mut b).is_ok());
    assert_eq!(u32::from_le_bytes(b), 2, "power down event");

    let mut cfg = a57();
    cfg.firmware = Some(fw);
    cfg.acpi = false;
    let m = VirtMachine::new(cfg).unwrap();
    assert!(m.ged().is_none());

    // A kernel without firmware: no GED, and the DSDT describes the PL061 power button.
    let m = VirtMachine::new(a57()).unwrap();
    assert!(m.ged().is_none());
    let dsdt = table(&load(&m.acpi_tables()), "DSDT").to_vec();
    assert!(dsdt.windows(4).any(|w| w == b"GPO0"));
    assert!(!dsdt.windows(4).any(|w| w == b"GED_"));
}

/// With firmware, the kernel (inflated), the initrd (as it is) and the command line go
/// through fw_cfg, and the device tree goes to the base of RAM.
#[test]
fn firmware_boot_through_fw_cfg() {
    use std::io::Write;
    let tmp = TmpDir::new("fwboot");
    let image: Vec<u8> = (0..10_000u32).map(|i| (i % 251) as u8).collect();
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    gz.write_all(&image).unwrap();
    let mut cfg = a57();
    cfg.firmware = Some(tmp.file("fw.fd", &[0; 0x1000]));
    cfg.kernel = Some(tmp.file("Image.gz", &gz.finish().unwrap()));
    cfg.initrd = Some(tmp.file("initrd", &[0x1f, 0x8b, 1, 2, 3]));
    cfg.append = Some("console=ttyAMA0".to_string());
    let mut m = VirtMachine::new(cfg).unwrap();
    m.machine_done().unwrap();
    assert!(!m.boot_info().direct);
    assert_eq!(m.boot_info().dtb_start, VIRT_MEM);
    let fwc = m.fw_cfg().state();
    let get = |key| fwc.entry_data(key).unwrap();
    assert_eq!(get(FW_CFG_KERNEL_SIZE), 10_000u32.to_le_bytes());
    assert_eq!(get(FW_CFG_KERNEL_DATA), image);
    assert_eq!(get(FW_CFG_INITRD_SIZE), 5u32.to_le_bytes());
    assert_eq!(get(FW_CFG_INITRD_DATA), [0x1f, 0x8b, 1, 2, 3]);
    assert_eq!(get(FW_CFG_CMDLINE_SIZE), 16u32.to_le_bytes());
    assert_eq!(get(FW_CFG_CMDLINE_DATA), b"console=ttyAMA0\0");

    // No -append is an empty command line.
    let mut cfg = a57();
    cfg.firmware = Some(tmp.file("fw.fd", &[0; 0x1000]));
    cfg.kernel = Some(tmp.file("Image", &image));
    let m = VirtMachine::new(cfg).unwrap();
    let fwc = m.fw_cfg().state();
    assert_eq!(fwc.entry_data(FW_CFG_KERNEL_DATA).unwrap(), image);
    assert_eq!(fwc.entry_data(FW_CFG_CMDLINE_SIZE).unwrap(), 1u32.to_le_bytes());

    let mut cfg = a57();
    cfg.firmware = Some(tmp.file("fw.fd", &[0; 0x1000]));
    cfg.kernel = Some("/nonexistent/Image".to_string());
    assert_eq!(VirtMachine::new(cfg).unwrap_err(), "failed to load \"/nonexistent/Image\"");
}

/// The cpu-map and the PPTT follow the `-smp` topology.
#[test]
fn smp_topology() {
    let mut cfg = a57();
    cfg.smp = 8;
    cfg.topology =
        Some(CpuTopology { sockets: 1, clusters: 2, cores: 2, threads: 2, has_clusters: true });
    let mut m = VirtMachine::new(cfg).unwrap();
    m.machine_done().unwrap();
    let dtb = m.fdt().as_bytes();
    let has = |s: &str| dtb.windows(s.len()).any(|w| w == s.as_bytes());
    for node in ["cluster1", "core1", "thread1"] {
        assert!(has(node), "{node}");
    }
    let tables = load(&m.acpi_tables());
    let pptt = table(&tables, "PPTT");
    // Root, socket, two clusters, four cores and eight threads.
    assert_eq!(pptt.len(), 36 + 16 * 20);

    let mut cfg = a57();
    cfg.smp = 4;
    cfg.topology =
        Some(CpuTopology { sockets: 1, clusters: 1, cores: 2, threads: 1, has_clusters: false });
    assert_eq!(
        VirtMachine::new(cfg).unwrap_err(),
        "Invalid CPU topology: product of the hierarchy must match maxcpus: sockets (1) * \
         clusters (1) * cores (2) * threads (1) != maxcpus (4)"
    );
}
