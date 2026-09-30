// SPDX-License-Identifier: GPL-2.0-or-later

//! The q35 board seen from the guest: fw_cfg and the CMOS through their ports, PCI config
//! space through 0xcf8/0xcfc, the firmware mappings, the legacy devices and the ACPI tables,
//! with the expected values worked out from hw/i386/pc_q35.c and pc.c.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_firmware::acpi::devices::IsaDevice;
use ruvm_firmware::acpi::pci::CrsRange;
use ruvm_firmware::acpi::q35::{self as acpi_q35, PciDevice, PciDeviceAml, Q35Acpi};
use ruvm_firmware::acpi::table::{APPNAME6, APPNAME8, McfgInfo};
use ruvm_firmware::acpi::x86::{MadtConfig, PossibleCpu};
use ruvm_firmware::x86_linux::{
    FW_CFG_CMDLINE_DATA, FW_CFG_CMDLINE_SIZE, FW_CFG_KERNEL_SIZE, FW_CFG_SETUP_SIZE,
    LINUXBOOT_DMA_ROM,
};
use ruvm_hw_acpi::SystemRequest;
use ruvm_hw_storage::{BlockBackend, DriveConfig, VecBackend};
use ruvm_machine_x86::microvm::KernelConfig;
use ruvm_machine_x86::q35::q35_ram_split;
use ruvm_machine_x86::{Q35, Q35MachineConfig, Q35Props};
use ruvm_mem::{Endian, MemTxAttrs};

const ATTRS: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

fn bios() -> Vec<u8> {
    let mut fw = vec![0u8; 256 * 1024];
    fw[256 * 1024 - 16..].copy_from_slice(b"reset vector!!!!");
    fw[128 * 1024] = 0xab;
    fw
}

fn config(opts: &str) -> Q35MachineConfig {
    let mut props = Q35Props::default();
    props.set_all(opts, &mut Vec::new()).unwrap();
    Q35MachineConfig { props, firmware: Some(bios()), ..Q35MachineConfig::default() }
}

fn machine(cfg: Q35MachineConfig) -> Q35 {
    let mut m = Q35::new(cfg).unwrap();
    m.machine_done().unwrap();
    m
}

fn inb(m: &Q35, port: u64) -> u8 {
    let (v, r) = m.io_as().load(port, 1, Endian::Little, ATTRS);
    assert!(r.is_ok(), "read of port {port:#x}");
    v as u8
}

fn outb(m: &Q35, port: u64, v: u8) {
    assert!(m.io_as().store(port, 1, v.into(), Endian::Little, ATTRS).is_ok());
}

fn cfg_addr(devfn: u8, reg: u8) -> u64 {
    0x8000_0000 | u64::from(devfn) << 8 | u64::from(reg & 0xfc)
}

fn pci_read32(m: &Q35, devfn: u8, reg: u8) -> u32 {
    assert!(m.io_as().store(0xcf8, 4, cfg_addr(devfn, reg), Endian::Little, ATTRS).is_ok());
    let (v, r) = m.io_as().load(0xcfc, 4, Endian::Little, ATTRS);
    assert!(r.is_ok());
    v as u32
}

fn pci_write32(m: &Q35, devfn: u8, reg: u8, v: u32) {
    assert!(m.io_as().store(0xcf8, 4, cfg_addr(devfn, reg), Endian::Little, ATTRS).is_ok());
    assert!(m.io_as().store(0xcfc, 4, v.into(), Endian::Little, ATTRS).is_ok());
}

fn cmos(m: &Q35, index: u8) -> u8 {
    outb(m, 0x70, index);
    inb(m, 0x71)
}

fn fw_cfg_read(m: &Q35, key: u16, len: usize) -> Vec<u8> {
    assert!(m.io_as().store(0x510, 2, key.into(), Endian::Little, ATTRS).is_ok());
    (0..len).map(|_| inb(m, 0x511)).collect()
}

fn fw_cfg_files(m: &Q35) -> Vec<(String, u16, u32)> {
    let count = u32::from_be_bytes(fw_cfg_read(m, 0x19, 4).try_into().unwrap()) as usize;
    let dir = fw_cfg_read(m, 0x19, 4 + count * 64);
    dir[4..]
        .chunks(64)
        .map(|e| {
            let size = u32::from_be_bytes(e[0..4].try_into().unwrap());
            let select = u16::from_be_bytes(e[4..6].try_into().unwrap());
            let name = e[8..].split(|&b| b == 0).next().unwrap();
            (String::from_utf8(name.to_vec()).unwrap(), select, size)
        })
        .collect()
}

fn fw_cfg_file(m: &Q35, name: &str) -> Option<Vec<u8>> {
    let (_, select, size) = fw_cfg_files(m).into_iter().find(|f| f.0 == name)?;
    Some(fw_cfg_read(m, select, size as usize))
}

fn le16(b: &[u8]) -> u16 {
    u16::from_le_bytes(b[..2].try_into().unwrap())
}

fn le32(b: &[u8]) -> u32 {
    u32::from_le_bytes(b[..4].try_into().unwrap())
}

fn le64(b: &[u8]) -> u64 {
    u64::from_le_bytes(b[..8].try_into().unwrap())
}

fn e820(m: &Q35) -> Vec<(u64, u64, u32)> {
    fw_cfg_file(m, "etc/e820")
        .unwrap()
        .chunks(20)
        .map(|e| (le64(&e[0..8]), le64(&e[8..16]), le32(&e[16..20])))
        .collect()
}

/// A bzImage with boot protocol 2.15 and four setup sectors.
fn bzimage() -> Vec<u8> {
    let mut k: Vec<u8> = (0..16384).map(|i| (i * 7 % 251) as u8).collect();
    for off in [0x20usize, 0x22, 0x1fa, 0x210, 0x218, 0x21c, 0x224, 0x228, 0x250, 0x254] {
        k[off..off + 4].fill(0);
    }
    k[0x1f1] = 4;
    k[0x202..0x206].copy_from_slice(b"HdrS");
    k[0x206..0x208].copy_from_slice(&0x20fu16.to_le_bytes());
    k[0x211] = 1; // LOADED_HIGH
    k[0x22c..0x230].copy_from_slice(&0x7fff_ffffu32.to_le_bytes());
    k[0x236..0x238].fill(0);
    k
}

#[test]
fn fw_cfg_basics() {
    let mut cfg = config("");
    cfg.cpus = 2;
    cfg.max_cpus = 4;
    let m = machine(cfg);
    assert_eq!(fw_cfg_read(&m, 0, 4), b"QEMU");
    assert_eq!(le16(&fw_cfg_read(&m, 0x05, 2)), 2); // NB_CPUS
    assert_eq!(le16(&fw_cfg_read(&m, 0x0f, 2)), 4); // MAX_CPUS
    assert_eq!(le64(&fw_cfg_read(&m, 0x03, 8)), 128 * MIB); // RAM_SIZE
    assert_eq!(le32(&fw_cfg_read(&m, 0x8002, 4)), 1); // IRQ0_OVERRIDE
    assert_eq!(fw_cfg_read(&m, 0x0d, 40), vec![0; 40]); // NUMA
    // The HPET block, packed: count 1, then the event timer block ID and the address.
    // The HPET block, packed: count 1, then the event timer block ID and the address.
    let hpet = fw_cfg_read(&m, 0x8004, 13);
    assert_eq!(hpet[0], 1);
    assert_eq!(le32(&hpet[1..5]), 0x8086_a201);
    assert_eq!(le64(&hpet[5..13]), 0xfed0_0000);
    assert_eq!(fw_cfg_file(&m, "etc/system-states").unwrap(), [128, 0, 0, 129, 130, 128]);
    assert!(fw_cfg_file(&m, "etc/smi/supported-features").is_some());
    assert!(fw_cfg_file(&m, "etc/smbios/smbios-tables").is_none());
}

#[test]
fn e820_1g() {
    let m = machine(Q35MachineConfig { ram_size: GIB, ..config("") });
    assert_eq!(e820(&m), [(0, GIB, 1)]);
    assert_eq!((m.below_4g_mem_size(), m.above_4g_mem_size()), (GIB, 0));
}

#[test]
fn e820_4g_splits_at_2g() {
    let m = machine(Q35MachineConfig { ram_size: 4 * GIB, ..config("") });
    assert_eq!(e820(&m), [(0, 2 * GIB, 1), (4 * GIB, 2 * GIB, 1)]);
    // Both halves are the same RAM block.
    assert!(m.memory_as().write(0x1000, ATTRS, b"lo").is_ok());
    assert!(m.memory_as().write(4 * GIB, ATTRS, b"hi").is_ok());
    let mut b = [0u8; 2];
    assert!(m.memory_as().read(4 * GIB, ATTRS, &mut b).is_ok());
    assert_eq!(&b, b"hi");
    assert_eq!(m.host().pci_hole64_start(), 6 * GIB);
}

#[test]
fn ram_split_rules() {
    assert_eq!(q35_ram_split(3 * GIB, 0), (2 * GIB, GIB, None));
    assert_eq!(q35_ram_split(0xa000_0000, 0), (0xa000_0000, 0, None));
    assert_eq!(q35_ram_split(0xb000_0000, 0), (2 * GIB, 0x3000_0000, None));
    assert_eq!(q35_ram_split(4 * GIB, GIB), (GIB, 3 * GIB, None));
    let (below, above, w) = q35_ram_split(4 * GIB, 0x3000_0000);
    assert_eq!((below, above), (0x3000_0000, 4 * GIB - 0x3000_0000));
    assert_eq!(
        w.unwrap(),
        "There is possibly poor performance as the ram size  (0x100000000) is more then twice \
         the size of max-ram-below-4g (805306368) and max-ram-below-4g is not a multiple of 1G."
    );
    let m = machine(Q35MachineConfig { ram_size: 2 * GIB, ..config("max-ram-below-4g=1G") });
    assert_eq!(e820(&m), [(0, GIB, 1), (4 * GIB, GIB, 1)]);
}

#[test]
fn kernel_items() {
    let mut cfg = config("");
    cfg.kernel = Some(KernelConfig {
        filename: "bzImage".into(),
        data: bzimage(),
        cmdline: "console=ttyS0".into(),
        ..KernelConfig::default()
    });
    cfg.rom_files.insert(LINUXBOOT_DMA_ROM.into(), vec![0x55, 0xaa, 1, 0xcb]);
    let m = machine(cfg);
    assert_eq!(le32(&fw_cfg_read(&m, FW_CFG_SETUP_SIZE, 4)), 5 * 512);
    assert_eq!(le32(&fw_cfg_read(&m, FW_CFG_KERNEL_SIZE, 4)), 16384 - 5 * 512);
    assert_eq!(le32(&fw_cfg_read(&m, FW_CFG_CMDLINE_SIZE, 4)), 14);
    assert_eq!(fw_cfg_read(&m, FW_CFG_CMDLINE_DATA, 14), b"console=ttyS0\0");
    assert_eq!(fw_cfg_file(&m, "genroms/linuxboot_dma.bin").unwrap(), [0x55, 0xaa, 1, 0xcb]);
    assert_eq!(fw_cfg_file(&m, "bootorder").unwrap(), b"/rom@genroms/linuxboot_dma.bin\0");
    assert!(m.warnings().is_empty());
}

#[test]
fn cmos_memory_and_boot_bytes() {
    let m = machine(config(""));
    // 640 KiB of base memory.
    assert_eq!((cmos(&m, 0x15), cmos(&m, 0x16)), (0x80, 0x02));
    // 127 MiB of extended memory does not fit, so it is capped at 0xffff.
    assert_eq!((cmos(&m, 0x17), cmos(&m, 0x18)), (0xff, 0xff));
    assert_eq!((cmos(&m, 0x30), cmos(&m, 0x31)), (0xff, 0xff));
    // 112 MiB above 16 MiB in 64 KiB units.
    assert_eq!((cmos(&m, 0x34), cmos(&m, 0x35)), (0x00, 0x07));
    assert_eq!((cmos(&m, 0x5b), cmos(&m, 0x5c), cmos(&m, 0x5d)), (0, 0, 0));
    // "cad": hard disk, floppy, then CD-ROM; fd-bootchk on.
    assert_eq!(cmos(&m, 0x3d), 0x12);
    assert_eq!(cmos(&m, 0x38), 0x30);
    // FPU and PS/2 mouse.
    assert_eq!(cmos(&m, 0x14), 0x06);
    assert_eq!(cmos(&m, 0x5f), 0);
    assert_eq!(cmos(&m, 0x10), 0);

    let mut cfg = config("fd-bootchk=off");
    cfg.ram_size = 4 * GIB;
    cfg.boot_order = "nd".into();
    cfg.cpus = 3;
    let m = machine(cfg);
    // 2 GiB above 4 GiB in 64 KiB units.
    assert_eq!((cmos(&m, 0x5b), cmos(&m, 0x5c), cmos(&m, 0x5d)), (0, 0x80, 0));
    assert_eq!((cmos(&m, 0x34), cmos(&m, 0x35)), (0x00, 0x7f));
    assert_eq!(cmos(&m, 0x3d), 0x34);
    assert_eq!(cmos(&m, 0x38), 0x01);
    assert_eq!(cmos(&m, 0x5f), 2);
}

#[test]
fn cmos_hard_disks() {
    let mut m = Q35::new(config("")).unwrap();
    let disk: Arc<dyn BlockBackend> = Arc::new(VecBackend::new(16 * 63 * 100 * 512));
    m.attach_drive(0, DriveConfig::hd(), Some(disk)).unwrap();
    m.attach_drive(1, DriveConfig::cdrom(), None).unwrap();
    m.machine_done().unwrap();
    // Port 0 is the master of IDE bus 0; the CD-ROM does not count.
    assert_eq!(cmos(&m, 0x12), 0xf0);
    assert_eq!(cmos(&m, 0x19), 47);
    assert_eq!((cmos(&m, 0x1b), cmos(&m, 0x1c), cmos(&m, 0x1d)), (100, 0, 16));
    assert_eq!(cmos(&m, 0x23), 63);
    assert_eq!(cmos(&m, 0x39), 0);
    assert!(m.attach_drive(2, DriveConfig::cdrom(), None).is_err());

    let m = Q35::new(config("")).unwrap();
    assert_eq!(
        m.attach_drive(6, DriveConfig::cdrom(), None).unwrap_err(),
        "machine type does not support if=ide,bus=6,unit=0"
    );
    let m = Q35::new(config("sata=off")).unwrap();
    assert!(m.attach_drive(0, DriveConfig::cdrom(), None).is_err());
    assert!(m.ahci().is_none());
}

#[test]
fn pci_functions_are_enumerated() {
    let m = machine(config(""));
    assert_eq!(pci_read32(&m, 0x00, 0), 0x29c0_8086);
    assert_eq!(pci_read32(&m, 0xf8, 0), 0x2918_8086);
    assert_eq!(pci_read32(&m, 0xfa, 0), 0x2922_8086);
    // The class codes: host bridge, ISA bridge and AHCI.
    assert_eq!(pci_read32(&m, 0x00, 8) >> 16, 0x0600);
    assert_eq!(pci_read32(&m, 0xf8, 8) >> 16, 0x0601);
    assert_eq!(pci_read32(&m, 0xfa, 8) >> 8, 0x01_0601);
    // Nothing at 00:01.0 or 00:1f.3 (no VGA, no SMBus).
    assert_eq!(pci_read32(&m, 0x08, 0), 0xffff_ffff);
    assert_eq!(pci_read32(&m, 0xfb, 0), 0xffff_ffff);

    let m = machine(config("sata=off"));
    assert_eq!(pci_read32(&m, 0xfa, 0), 0xffff_ffff);
}

#[test]
fn bios_is_mapped_at_4g_and_below_1m() {
    let m = machine(config(""));
    let mut b = [0u8; 16];
    assert!(m.memory_as().read(0xffff_fff0, ATTRS, &mut b).is_ok());
    assert_eq!(&b, b"reset vector!!!!");
    assert!(m.memory_as().read(0xffff_0000, ATTRS, &mut b[..1]).is_ok());
    let mut want = [0u8; 1];
    want[0] = bios()[0x3_0000];
    assert_eq!(b[..1], want);
    // With the PAM registers at their reset value, 0xe0000 to 0xfffff go to PCI, where the
    // isa-bios alias of the last 128 KiB lives.
    assert!(m.memory_as().read(0xffff0, ATTRS, &mut b).is_ok());
    assert_eq!(&b, b"reset vector!!!!");
    assert!(m.memory_as().read(0xe0000, ATTRS, &mut b[..1]).is_ok());
    assert_eq!(b[0], 0xab);

    // pc.bios is read-only to the guest, and a reset reloads it.
    let mut m = m;
    let _ = m.memory_as().write(0xffff_fff0, ATTRS, b"x");
    assert!(m.memory_as().read(0xffff_fff0, ATTRS, &mut b[..1]).is_ok());
    assert_eq!(b[0], b'r');
    let block = m.memory_system().ram_block(m.bios_region()).unwrap();
    block.write(256 * 1024 - 16, b"x").unwrap();
    assert!(m.memory_as().read(0xffff_fff0, ATTRS, &mut b[..1]).is_ok());
    assert_eq!(b[0], b'x');
    m.system_reset().unwrap();
    assert!(m.memory_as().read(0xffff_fff0, ATTRS, &mut b[..1]).is_ok());
    assert_eq!(b[0], b'r');
}

#[test]
fn bios_errors() {
    let mut cfg = config("");
    cfg.firmware = None;
    assert_eq!(Q35::new(cfg).unwrap_err(), "qemu: could not load PC BIOS 'bios-256k.bin'");
    let mut cfg = config("");
    cfg.firmware = Some(vec![0; 1000]);
    cfg.firmware_name = Some("x.bin".into());
    assert_eq!(Q35::new(cfg).unwrap_err(), "qemu: could not load PC BIOS 'x.bin'");
}

#[test]
fn i8042_self_test_and_reset() {
    let m = machine(config(""));
    let resets = Arc::new(AtomicUsize::new(0));
    let r = Arc::clone(&resets);
    m.set_request_handler(Some(Arc::new(move |req| {
        if req == SystemRequest::Reset {
            r.fetch_add(1, Ordering::SeqCst);
        }
    })));
    outb(&m, 0x64, 0xaa);
    assert_eq!(inb(&m, 0x64) & 1, 1);
    assert_eq!(inb(&m, 0x60), 0x55);
    // Pulse the reset line.
    outb(&m, 0x64, 0xfe);
    assert_eq!(resets.load(Ordering::SeqCst), 1);

    let m = machine(config("i8042=off,vmport=auto"));
    assert!(m.i8042().is_none() && m.port92().is_none());
    assert_eq!(inb(&m, 0x64), 0xff);
}

#[test]
fn port92_drives_a20_and_reset() {
    let mut m = machine(config(""));
    let a20 = Arc::new(Mutex::new(Vec::new()));
    let a = Arc::clone(&a20);
    m.set_a20_handler(Some(Arc::new(move |on| a.lock().unwrap().push(on))));
    let reset = Arc::new(AtomicBool::new(false));
    let r = Arc::clone(&reset);
    m.set_request_handler(Some(Arc::new(move |req| {
        if req == SystemRequest::Reset {
            r.store(true, Ordering::SeqCst);
        }
    })));
    assert!(m.a20_enabled());
    outb(&m, 0x92, 0x02);
    assert_eq!(inb(&m, 0x92), 0x02);
    assert!(m.a20_enabled());
    outb(&m, 0x92, 0x00);
    assert!(!m.a20_enabled());
    // The i8042 output port drives the same line: bit 1 of the output port.
    outb(&m, 0x64, 0xd1);
    outb(&m, 0x60, 0x03);
    assert!(m.a20_enabled());
    assert!(!reset.load(Ordering::SeqCst));
    outb(&m, 0x92, 0x01);
    assert!(reset.load(Ordering::SeqCst));
    assert_eq!(*a20.lock().unwrap(), [true, false, true, false]);
    m.system_reset().unwrap();
    assert!(m.a20_enabled());
}

#[test]
fn hpet_and_ioapic_are_mapped() {
    let m = machine(config(""));
    let mut b = [0u8; 8];
    assert!(m.memory_as().read(0xfed0_0000, ATTRS, &mut b[..4]).is_ok());
    assert_eq!(le32(&b), 0x8086_a201);
    assert!(m.memory_as().read(0xfed0_0004, ATTRS, &mut b[..4]).is_ok());
    assert_eq!(le32(&b), 10_000_000);
    // IOAPIC version register: 24 entries, version 0x20.
    assert!(m.memory_as().write(0xfec0_0000, ATTRS, &1u32.to_le_bytes()).is_ok());
    assert!(m.memory_as().read(0xfec0_0010, ATTRS, &mut b[..4]).is_ok());
    assert_eq!(le32(&b), 0x0017_0020);

    let m = machine(config("hpet=off"));
    assert!(m.hpet().is_none());
    b = [0; 8];
    let _ = m.memory_as().read(0xfed0_0000, ATTRS, &mut b[..4]);
    assert_ne!(le32(&b), 0x8086_a201);
}

#[test]
fn legacy_devices() {
    let m = machine(config(""));
    // Port 0x80 and 0xf0 read as all ones.
    assert_eq!(inb(&m, 0x80), 0xff);
    assert_eq!(inb(&m, 0xf0), 0xff);
    // COM1 scratch register.
    outb(&m, 0x3ff, 0x5a);
    assert_eq!(inb(&m, 0x3ff), 0x5a);
    // The PIC's ELCR registers.
    outb(&m, 0x4d0, 0x20);
    assert_eq!(inb(&m, 0x4d0), 0x20);
    assert!(m.pit().is_some() && m.pcspk().is_some() && m.pic().is_some());
    assert_eq!(m.apic_ids(), [0]);
    assert!(m.smm_as().is_some());
    assert!(!m.ram_ranges().unwrap().is_empty());

    let mut cfg = config("pit=off,pic=off,smm=off");
    cfg.serial_hd = false;
    let m = machine(cfg);
    assert!(m.pit().is_none() && m.pcspk().is_none() && m.pic().is_none());
    assert!(m.serial().is_none() && m.smm_as().is_none());
    assert_eq!(inb(&m, 0x3ff), 0xff);
}

#[test]
fn acpi_tables_match_the_firmware_builder() {
    let m = machine(config(""));
    // What SeaBIOS does before reading the tables: PMBASE and PCIEXBAR.
    pci_write32(&m, 0xf8, 0x40, 0x601);
    pci_write32(&m, 0xf8, 0x44, 0x80);
    pci_write32(&m, 0x00, 0x64, 0);
    pci_write32(&m, 0x00, 0x60, 0xb000_0001);
    let host = m.host();
    let want = Q35Acpi {
        oem_id: APPNAME6.into(),
        oem_table_id: APPNAME8.into(),
        madt: MadtConfig {
            cpus: vec![PossibleCpu { arch_id: 0, present: true }],
            pic: true,
            ioapic2: false,
            apic_xrupt_override: true,
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
        pci_hole: CrsRange {
            base: host.pci_hole_start().into(),
            limit: u64::from(host.pci_hole_end()) - 1,
        },
        pci_hole64: Some(CrsRange { base: 1 << 32, limit: (1 << 32) + (32 << 30) - 1 }),
        // bios-tables-test's defaults without the VGA at 00:01.0, the parallel port and the
        // SMBus controller at 00:1f.3.
        pci_devices: vec![
            PciDevice { devfn: 0, acpi_index: None, aml: PciDeviceAml::Plain },
            PciDevice {
                devfn: 0xf8,
                acpi_index: None,
                aml: PciDeviceAml::Lpc {
                    isa: vec![
                        IsaDevice::I8042 { kbd_irq: 1, mouse_irq: 12 },
                        IsaDevice::Serial { index: 0, iobase: 0x3f8, irq: 4 },
                        IsaDevice::Rtc { io_base: 0x70, irq: 8 },
                    ],
                },
            },
            PciDevice { devfn: 0xfa, acpi_index: None, aml: PciDeviceAml::Plain },
        ],
    };
    assert_eq!(format!("{:?}", m.acpi_input()), format!("{want:?}"));
    let t = acpi_q35::build(&want);
    assert_eq!(fw_cfg_file(&m, "etc/acpi/tables").unwrap(), t.table_data);
    assert_eq!(fw_cfg_file(&m, "etc/acpi/rsdp").unwrap(), t.rsdp);
    assert_eq!(fw_cfg_file(&m, "etc/table-loader").unwrap(), t.linker.cmd_blob());
    assert_eq!(&t.rsdp[..8], b"RSD PTR ");

    let m = machine(config("acpi=off"));
    assert!(fw_cfg_file(&m, "etc/acpi/tables").is_none());
    assert!(fw_cfg_file(&m, "etc/e820").is_some());
}

#[test]
fn property_errors() {
    let mut p = Q35Props::default();
    let mut w = Vec::new();
    assert_eq!(
        p.set("foo", "1", &mut w).unwrap_err(),
        "Property 'pc-q35-11.1-machine.foo' not found"
    );
    assert_eq!(
        p.set("max-ram-below-4g", "5G", &mut w).unwrap_err(),
        "Machine option 'max-ram-below-4g=5368709120' expects size less than or equal to 4G"
    );
    assert_eq!(
        p.set("max-ram-below-4g", "lots", &mut w).unwrap_err(),
        "Parameter 'max-ram-below-4g' expects size"
    );
    p.set("max-ram-below-4g", "512K", &mut w).unwrap();
    assert_eq!(
        w,
        ["Only 524288 bytes of RAM below the 4GiB boundary,BIOS may not work with less than 1MiB"]
    );
    assert_eq!(
        p.set("hpet", "maybe", &mut w).unwrap_err(),
        "Parameter 'hpet' expects 'on' or 'off'"
    );
    assert_eq!(
        p.set("oem-id", "ABCDEFG", &mut w).unwrap_err(),
        "User specified oem-id value is bigger than 6 bytes in size"
    );
    assert_eq!(
        p.set("oem-table-id", "ABCDEFGHI", &mut w).unwrap_err(),
        "User specified oem-table-id value is bigger than 8 bytes in size"
    );
    p.set("smbus", "off", &mut w).unwrap();
    p.set("default-bus-bypass-iommu", "on", &mut w).unwrap();
    p.set("graphics", "off", &mut w).unwrap();

    assert_eq!(
        Q35::new(config("i8042=off,vmport=on")).unwrap_err(),
        "vmport requires the i8042 controller to be enabled"
    );
    let mut cfg = config("smm=on");
    cfg.kvm = true;
    cfg.smm_available = false;
    assert_eq!(
        Q35::new(cfg).unwrap_err(),
        "System Management Mode not supported by this hypervisor."
    );
    let mut cfg = config("");
    cfg.phys_bits = 32;
    assert_eq!(
        Q35::new(cfg).unwrap_err(),
        "Address space limit 0xffffffff < 0x8ffffffff phys-bits too low (32)"
    );
    let mut cfg = config("");
    cfg.max_cpus = 5000;
    assert_eq!(
        Q35::new(cfg).unwrap_err(),
        "Invalid SMP CPUs 5000. The max CPUs supported by machine 'pc-q35-11.1' is 4096"
    );
    let mut cfg = config("");
    cfg.boot_order = "cadn".into();
    assert_eq!(Q35::new(cfg).unwrap().machine_done().unwrap_err(), "Too many boot devices for PC");
    let mut cfg = config("");
    cfg.boot_order = "x".into();
    assert_eq!(
        Q35::new(cfg).unwrap().machine_done().unwrap_err(),
        "Invalid boot device for PC: 'x'"
    );
}
