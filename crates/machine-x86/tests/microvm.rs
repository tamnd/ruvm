// SPDX-License-Identifier: GPL-2.0-or-later

//! The microvm board seen from the guest: fw_cfg through its I/O ports, the ISA devices, the
//! virtio-mmio transports and the ACPI tables, with the expected values worked out from
//! hw/i386/microvm.c.

use std::sync::{Arc, Mutex};

use ruvm_firmware::acpi::devices::{IsaDevice, ged_event};
use ruvm_firmware::acpi::microvm::{self as acpi_microvm, MicrovmAcpi};
use ruvm_firmware::acpi::table::{APPNAME6, APPNAME8};
use ruvm_firmware::acpi::x86::{MadtConfig, PossibleCpu};
use ruvm_firmware::x86_linux::{
    FW_CFG_CMDLINE_DATA, FW_CFG_CMDLINE_SIZE, FW_CFG_KERNEL_SIZE, FW_CFG_SETUP_SIZE,
    LINUXBOOT_DMA_ROM,
};
use ruvm_hw_char::serial::SerialBackend;
use ruvm_hw_virtio::blk::{MemBlockBackend, VirtioBlk, VirtioBlkConf};
use ruvm_hw_virtio::rng::{RandomFile, VirtioRng, VirtioRngConf};
use ruvm_machine_x86::microvm::{
    KernelConfig, OnOffAuto, OptionRom, VIRTIO_MMIO_BASE, default_firmware_name,
};
use ruvm_machine_x86::{Microvm, MicrovmConfig, MicrovmProps};
use ruvm_mem::{Endian, MemTxAttrs};

const ATTRS: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

fn config(opts: &str) -> MicrovmConfig {
    let mut props = MicrovmProps::default();
    props.set_all(opts).unwrap();
    MicrovmConfig { props, firmware: Some(vec![0xf4; 65536]), ..MicrovmConfig::default() }
}

fn machine(cfg: MicrovmConfig) -> Microvm {
    let mut m = Microvm::new(cfg).unwrap();
    m.machine_done().unwrap();
    m
}

fn inb(m: &Microvm, port: u64) -> u8 {
    let (v, r) = m.io_as().load(port, 1, Endian::Little, ATTRS);
    assert!(r.is_ok(), "read of port {port:#x}");
    v as u8
}

fn outb(m: &Microvm, port: u64, v: u8) {
    assert!(m.io_as().store(port, 1, v.into(), Endian::Little, ATTRS).is_ok());
}

/// Selects `key` and reads `len` bytes through the data port, like a BIOS without DMA.
fn fw_cfg_read(m: &Microvm, key: u16, len: usize) -> Vec<u8> {
    assert!(m.io_as().store(0x510, 2, key.into(), Endian::Little, ATTRS).is_ok());
    (0..len).map(|_| inb(m, 0x511)).collect()
}

/// The fw_cfg file directory: `(name, select, size)`.
fn fw_cfg_files(m: &Microvm) -> Vec<(String, u16, u32)> {
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

fn fw_cfg_file(m: &Microvm, name: &str) -> Option<Vec<u8>> {
    let (_, select, size) = fw_cfg_files(m).into_iter().find(|f| f.0 == name)?;
    Some(fw_cfg_read(m, select, size as usize))
}

fn le32(b: &[u8]) -> u32 {
    u32::from_le_bytes(b[..4].try_into().unwrap())
}

fn e820(m: &Microvm) -> Vec<(u64, u64, u32)> {
    fw_cfg_file(m, "etc/e820")
        .unwrap()
        .chunks(20)
        .map(|e| {
            (
                u64::from_le_bytes(e[0..8].try_into().unwrap()),
                u64::from_le_bytes(e[8..16].try_into().unwrap()),
                le32(&e[16..20]),
            )
        })
        .collect()
}

fn rng() -> Box<VirtioRng> {
    Box::new(VirtioRng::new(Box::new(RandomFile::default()), VirtioRngConf::default()))
}

fn blk() -> Box<VirtioBlk> {
    Box::new(VirtioBlk::new(Box::new(MemBlockBackend::new(1 << 20)), VirtioBlkConf::default()))
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

fn kernel_config(opts: &str, cmdline: &str) -> MicrovmConfig {
    let mut cfg = config(opts);
    cfg.kernel = Some(KernelConfig {
        filename: "bzImage".into(),
        data: bzimage(),
        cmdline: cmdline.into(),
        ..KernelConfig::default()
    });
    cfg.rom_files.insert(LINUXBOOT_DMA_ROM.into(), vec![0x55, 0xaa, 1, 0xcb]);
    cfg
}

#[test]
fn fw_cfg_signature_id_and_dma_port() {
    let m = machine(config(""));
    assert_eq!(fw_cfg_read(&m, 0, 4), b"QEMU");
    // FW_CFG_VERSION | FW_CFG_VERSION_DMA
    assert_eq!(le32(&fw_cfg_read(&m, 1, 4)), 3);
    let (sig, r) = m.io_as().load(0x514, 4, Endian::Big, ATTRS);
    assert!(r.is_ok());
    assert_eq!(sig, 0x5145_4d55);
    assert_eq!(fw_cfg_read(&m, 0x05, 2), [1, 0]); // NB_CPUS
    assert_eq!(fw_cfg_read(&m, 0x0f, 2), [1, 0]); // MAX_CPUS
    assert_eq!(fw_cfg_read(&m, 0x03, 8), (128 * MIB).to_le_bytes()); // RAM_SIZE
    assert_eq!(le32(&fw_cfg_read(&m, 0x8002, 4)), 1); // IRQ0_OVERRIDE
    let names: Vec<String> = fw_cfg_files(&m).into_iter().map(|f| f.0).collect();
    for want in ["etc/boot-fail-wait", "etc/e820", "etc/fdt", "bootorder", "bios-geometry"] {
        assert!(names.iter().any(|n| n == want), "{want} missing from {names:?}");
    }
}

#[test]
fn smp_counts() {
    let mut cfg = config("acpi=off");
    cfg.cpus = 2;
    cfg.max_cpus = 4;
    let m = machine(cfg);
    assert_eq!(fw_cfg_read(&m, 0x05, 2), [2, 0]);
    assert_eq!(fw_cfg_read(&m, 0x0f, 2), [4, 0]);
}

#[test]
fn e820_small_ram() {
    let m = machine(config(""));
    assert_eq!(e820(&m), [(0, 128 * MIB, 1)]);
    assert_eq!(m.below_4g_mem_size(), 128 * MIB);
    assert_eq!(m.above_4g_mem_size(), 0);
}

#[test]
fn e820_splits_above_3g() {
    let mut cfg = config("");
    cfg.ram_size = 4 * GIB;
    let m = machine(cfg);
    assert_eq!(e820(&m), [(0, 3 * GIB, 1), (4 * GIB, GIB, 1)]);
    assert_eq!(fw_cfg_read(&m, 0x03, 8), (4 * GIB).to_le_bytes());

    // RAM above 4G is the tail of the same block.
    let block = m.ram_block().unwrap();
    assert!(m.memory_as().write(4 * GIB + 16, ATTRS, b"high").is_ok());
    let mut b = [0u8; 4];
    block.read(3 * GIB + 16, &mut b).unwrap();
    assert_eq!(&b, b"high");

    let ranges = m.ram_ranges().unwrap();
    let covers = |gpa: u64| ranges.iter().any(|r| r.gpa <= gpa && gpa < r.gpa + r.size);
    assert!(covers(0) && covers(3 * GIB - 1) && covers(4 * GIB) && covers(5 * GIB - 1));
    assert!(covers(4 * GIB - 1), "pc.bios");
    assert!(!covers(3 * GIB) && !covers(5 * GIB));
    assert!(ranges.iter().all(|r| !r.readonly));
    let high = ranges.iter().find(|r| r.gpa == 4 * GIB).unwrap();
    assert_eq!((high.size, high.offset), (GIB, 3 * GIB));
    assert!(Arc::ptr_eq(&high.block, &block));
}

#[test]
fn firmware_is_mapped_below_4g_and_at_e0000() {
    let mut cfg = config("");
    let mut fw = vec![0u8; 256 * 1024];
    fw[256 * 1024 - 16..].copy_from_slice(b"reset vector!!!!");
    fw[128 * 1024] = 0xab;
    cfg.firmware = Some(fw);
    let m = machine(cfg);
    let mut b = [0u8; 16];
    assert!(m.memory_as().read(0xffff_fff0, ATTRS, &mut b).is_ok());
    assert_eq!(&b, b"reset vector!!!!");
    // The ISA copy is the last 128 KiB.
    assert!(m.memory_as().read(0xffff0, ATTRS, &mut b).is_ok());
    assert_eq!(&b, b"reset vector!!!!");
    assert!(m.memory_as().read(0xe0000, ATTRS, &mut b[..1]).is_ok());
    assert_eq!(b[0], 0xab);

    // The firmware is RAM: the guest may scribble on it, and a reset reloads it.
    let mut m = m;
    assert!(m.memory_as().write(0xffff_fff0, ATTRS, b"x").is_ok());
    m.system_reset().unwrap();
    assert!(m.memory_as().read(0xffff_fff0, ATTRS, &mut b[..1]).is_ok());
    assert_eq!(b[0], b'r');
}

#[test]
fn firmware_errors() {
    let mut cfg = config("");
    cfg.firmware = Some(vec![0; 1000]);
    assert_eq!(Microvm::new(cfg).unwrap_err(), "qemu: could not load PC BIOS 'bios-microvm.bin'");
    let mut cfg = config("acpi=off");
    cfg.firmware = None;
    assert_eq!(Microvm::new(cfg).unwrap_err(), "qemu: could not load PC BIOS 'qboot.rom'");
    let mut cfg = config("");
    cfg.firmware_name = Some("my.bin".into());
    cfg.firmware = None;
    assert_eq!(Microvm::new(cfg).unwrap_err(), "qemu: could not load PC BIOS 'my.bin'");

    let mut p = MicrovmProps::default();
    assert_eq!(default_firmware_name(&p), "bios-microvm.bin");
    p.acpi = OnOffAuto::Off;
    assert_eq!(default_firmware_name(&p), "qboot.rom");
}

#[test]
fn kernel_items_and_boot_rom() {
    let m = machine(kernel_config("", "console=ttyS0"));
    assert_eq!(le32(&fw_cfg_read(&m, FW_CFG_SETUP_SIZE, 4)), 5 * 512);
    assert_eq!(le32(&fw_cfg_read(&m, FW_CFG_KERNEL_SIZE, 4)), 16384 - 5 * 512);
    assert_eq!(le32(&fw_cfg_read(&m, FW_CFG_CMDLINE_SIZE, 4)), 14);
    assert_eq!(fw_cfg_read(&m, FW_CFG_CMDLINE_DATA, 14), b"console=ttyS0\0");
    assert_eq!(fw_cfg_file(&m, "etc/boot/kernel").unwrap().len(), 16384);
    assert_eq!(fw_cfg_file(&m, "genroms/linuxboot_dma.bin").unwrap(), [0x55, 0xaa, 1, 0xcb]);
    assert_eq!(fw_cfg_file(&m, "bootorder").unwrap(), b"/rom@genroms/linuxboot_dma.bin\0");
    assert!(m.warnings().is_empty());
}

#[test]
fn option_roms_follow_bootindex_and_x_option_roms() {
    let mut cfg = kernel_config("", "");
    cfg.option_roms = vec![
        OptionRom { name: "/roms/a.rom".into(), bootindex: 3 },
        OptionRom { name: "missing.rom".into(), bootindex: -1 },
    ];
    cfg.rom_files.insert("/roms/a.rom".into(), vec![1, 2, 3]);
    let m = machine(cfg);
    assert_eq!(fw_cfg_file(&m, "genroms/a.rom").unwrap(), [1, 2, 3]);
    assert_eq!(
        fw_cfg_file(&m, "bootorder").unwrap(),
        b"/rom@genroms/linuxboot_dma.bin\n/rom@genroms/a.rom\0"
    );
    assert_eq!(
        m.warnings(),
        ["rom: file missing.rom         : error Failed to open file \u{201c}missing.rom\u{201d}: \
          No such file or directory"]
    );

    let m = machine(kernel_config("x-option-roms=off", ""));
    assert!(fw_cfg_file(&m, "genroms/linuxboot_dma.bin").is_none());
    assert_eq!(fw_cfg_file(&m, "bootorder").unwrap(), b"");
}

#[test]
fn cmdline_gets_virtio_devices_without_acpi() {
    let mut m = Microvm::new(kernel_config("acpi=off", "console=ttyS0")).unwrap();
    assert_eq!(m.virtio_irq_base(), 5);
    assert_eq!(m.virtio_transport_count(), 8);
    assert_eq!(m.attach_virtio(rng()).unwrap(), 7);
    assert_eq!(m.attach_virtio(blk()).unwrap(), 6);
    m.machine_done().unwrap();
    let want = "console=ttyS0 virtio_mmio.device=512@0xfeb00e00:12 \
                virtio_mmio.device=512@0xfeb00c00:11";
    let len = want.len() + 1;
    assert_eq!(le32(&fw_cfg_read(&m, FW_CFG_CMDLINE_SIZE, 4)) as usize, len);
    let mut data = want.as_bytes().to_vec();
    data.push(0);
    assert_eq!(fw_cfg_read(&m, FW_CFG_CMDLINE_DATA, len), data);

    // Only once, not again on the next reset.
    m.system_reset().unwrap();
    assert_eq!(le32(&fw_cfg_read(&m, FW_CFG_CMDLINE_SIZE, 4)) as usize, len);
}

#[test]
fn cmdline_untouched_with_acpi_or_auto_kernel_cmdline_off() {
    for opts in ["", "acpi=off,auto-kernel-cmdline=off"] {
        let mut m = Microvm::new(kernel_config(opts, "quiet")).unwrap();
        m.attach_virtio(rng()).unwrap();
        m.machine_done().unwrap();
        assert_eq!(fw_cfg_read(&m, FW_CFG_CMDLINE_DATA, 6), b"quiet\0", "{opts}");
    }
}

#[test]
fn transports_and_irq_bases() {
    let m = machine(config(""));
    assert_eq!((m.virtio_irq_base(), m.virtio_transport_count()), (24, 24));
    assert!(m.ioapic2().is_some());
    assert_eq!(m.gsi().len(), 48);
    let m = machine(config("ioapic2=off"));
    assert_eq!((m.virtio_irq_base(), m.virtio_transport_count()), (16, 8));
    assert!(m.ioapic2().is_none());
    let m = machine(config("acpi=off,ioapic2=on"));
    assert_eq!((m.virtio_irq_base(), m.virtio_transport_count()), (5, 8));
    assert_eq!(m.gsi().len(), 24);
    assert!(m.ged().is_none());
}

#[test]
fn virtio_mmio_magic_and_device_id() {
    let mut m = Microvm::new(config("acpi=off")).unwrap();
    m.attach_virtio(rng()).unwrap();
    m.attach_virtio_at(0, blk(), false).unwrap();
    m.machine_done().unwrap();
    let rd = |off: u64| {
        let (v, r) = m.memory_as().read_u32(off, ATTRS);
        assert!(r.is_ok());
        v
    };
    for i in 0..8 {
        assert_eq!(rd(VIRTIO_MMIO_BASE + i * 0x200), 0x7472_6976, "magic of {i}");
    }
    assert_eq!(rd(VIRTIO_MMIO_BASE + 7 * 0x200 + 8), 4, "virtio-rng");
    assert_eq!(rd(VIRTIO_MMIO_BASE + 7 * 0x200 + 4), 1, "legacy by default");
    assert_eq!(rd(VIRTIO_MMIO_BASE + 8), 2, "virtio-blk");
    assert_eq!(rd(VIRTIO_MMIO_BASE + 4), 2, "force-legacy=off");
    assert_eq!(rd(VIRTIO_MMIO_BASE + 3 * 0x200 + 8), 0, "empty transport");
    assert!(m.virtio_plugged(7) && m.virtio_plugged(0) && !m.virtio_plugged(1));
    assert!(m.attach_virtio(rng()).is_err(), "no plugging after machine_done");
}

#[test]
fn virtio_attach_errors() {
    let m = Microvm::new(config("acpi=off")).unwrap();
    m.attach_virtio_at(2, rng(), true).unwrap();
    assert_eq!(
        m.attach_virtio_at(2, rng(), true).unwrap_err(),
        "Bus 'virtio-mmio-bus.2' does not support hotplugging"
    );
    assert_eq!(
        m.attach_virtio_at(8, rng(), true).unwrap_err(),
        "Bus 'virtio-mmio-bus.8' not found"
    );
    for _ in 0..7 {
        m.attach_virtio(rng()).unwrap();
    }
    assert_eq!(m.attach_virtio(rng()).unwrap_err(), "No 'virtio-bus' bus found for device");
}

#[derive(Default)]
struct Capture(Mutex<Vec<u8>>);

impl SerialBackend for Capture {
    fn write(&self, bytes: &[u8]) -> usize {
        self.0.lock().unwrap().extend_from_slice(bytes);
        bytes.len()
    }
}

#[test]
fn serial_port() {
    let m = machine(config(""));
    assert_eq!(inb(&m, 0x3fd), 0x60, "LSR: THR and TSR empty");
    let out = Arc::new(Capture::default());
    assert!(m.set_serial_backend(Some(out.clone())));
    outb(&m, 0x3f8, b'h');
    outb(&m, 0x3f8, b'i');
    assert_eq!(*out.0.lock().unwrap(), b"hi");

    let m = machine(config("isa-serial=off"));
    assert!(m.serial().is_none());
    assert_eq!(inb(&m, 0x3fd), 0xff, "nothing at 0x3fd");
    let mut cfg = config("");
    cfg.serial_hd = false;
    assert!(machine(cfg).serial().is_none());
}

#[test]
fn rtc_and_cmos_memory_sizes() {
    let mut cfg = config("");
    cfg.ram_size = 4 * GIB;
    let m = machine(cfg);
    let cmos = |i: u8| {
        outb(&m, 0x70, i);
        inb(&m, 0x71)
    };
    assert_eq!(cmos(0x0a), 0x26);
    assert_eq!(cmos(0x0b), 0x02, "24 hour mode");
    assert_eq!((cmos(0x15), cmos(0x16)), (0x80, 0x02));
    assert_eq!((cmos(0x17), cmos(0x18)), (0xff, 0xff));
    assert_eq!((cmos(0x30), cmos(0x31)), (0xff, 0xff));
    // (3 GiB - 16 MiB) / 64 KiB = 0xbf00
    assert_eq!((cmos(0x34), cmos(0x35)), (0x00, 0xbf));
    // 1 GiB / 64 KiB = 0x4000
    assert_eq!((cmos(0x5b), cmos(0x5c), cmos(0x5d)), (0x00, 0x40, 0x00));

    let m = machine(config("rtc=off"));
    assert!(m.rtc().is_none());
    let mut cfg = config("");
    cfg.kvm = true;
    assert!(machine(cfg).rtc().is_none(), "rtc=auto with KVM");
    let mut cfg = config("rtc=on");
    cfg.kvm = true;
    assert!(machine(cfg).rtc().is_some());
}

#[test]
fn pit_and_pic() {
    let m = machine(config(""));
    outb(&m, 0x43, 0x34); // counter 0, lobyte/hibyte, mode 2
    outb(&m, 0x40, 0x00);
    outb(&m, 0x40, 0x10);
    outb(&m, 0x43, 0xe2); // read back the status of counter 0
    assert_eq!(inb(&m, 0x40) & 0x3f, 0x34);

    // The 8259 IMR is plain storage.
    outb(&m, 0x21, 0xa5);
    assert_eq!(inb(&m, 0x21), 0xa5);

    let m = machine(config("pit=off,pic=off"));
    assert!(m.pit().is_none() && m.pic().is_none());
    assert_eq!(inb(&m, 0x21), 0xff);
}

#[test]
fn isa_irq_reaches_the_pic_output() {
    let m = machine(config("acpi=off"));
    // ICW1..ICW4 for the master, then unmask IRQ 4.
    for (port, v) in [(0x20, 0x11), (0x21, 0x08), (0x21, 0x04), (0x21, 0x01), (0x21, 0xef)] {
        outb(&m, port, v);
    }
    let level = Arc::new(Mutex::new(0));
    let l = level.clone();
    m.pic_output().connect(ruvm_hw_core::IrqLine::from_fn(move |v| *l.lock().unwrap() = v));
    m.gsi()[4].set(1);
    assert_eq!(*level.lock().unwrap(), 1);
}

/// `-machine microvm,acpi=on,ioapic2=off,rtc=off` with one virtio-blk-device, the case
/// bios-tables-test uses.
#[test]
fn acpi_tables_match_the_firmware_builder() {
    let mut m = Microvm::new(config("ioapic2=off,rtc=off")).unwrap();
    m.attach_virtio(blk()).unwrap();
    m.machine_done().unwrap();
    let want = MicrovmAcpi {
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
    };
    let t = acpi_microvm::build(&want);
    assert_eq!(fw_cfg_file(&m, "etc/acpi/tables").unwrap(), t.table_data);
    assert_eq!(fw_cfg_file(&m, "etc/acpi/rsdp").unwrap(), t.rsdp);
    assert_eq!(fw_cfg_file(&m, "etc/table-loader").unwrap(), t.linker.cmd_blob());
    assert_eq!(&t.rsdp[..8], b"RSD PTR ");
}

#[test]
fn acpi_input_with_rtc_and_ioapic2() {
    let mut cfg = config("oem-id=RUVM");
    cfg.cpus = 2;
    cfg.max_cpus = 3;
    let m = Microvm::new(cfg).unwrap();
    m.attach_virtio(rng()).unwrap();
    m.attach_virtio(rng()).unwrap();
    let a = m.acpi_input();
    assert_eq!(a.oem_id, "RUVM");
    assert_eq!(a.virtio_irq_base, 24);
    assert_eq!(a.virtio_transports, [23, 22]);
    assert!(a.madt.ioapic2);
    let present: Vec<bool> = a.madt.cpus.iter().map(|c| c.present).collect();
    assert_eq!(present, [true, true, false]);
    assert_eq!(
        a.isa,
        [
            IsaDevice::Serial { index: 0, iobase: 0x3f8, irq: 4 },
            IsaDevice::Rtc { io_base: 0x70, irq: 8 }
        ]
    );
}

#[test]
fn no_acpi_files_with_acpi_off() {
    let m = machine(config("acpi=off"));
    assert!(fw_cfg_file(&m, "etc/acpi/tables").is_none());
    assert!(fw_cfg_file(&m, "etc/acpi/rsdp").is_none());
    assert!(m.ged().is_none());
}

#[test]
fn ged_registers_are_mapped() {
    let m = machine(config(""));
    assert!(m.ged().is_some());
    let mut b = [0u8; 4];
    assert!(m.memory_as().read(0xfea0_0000, ATTRS, &mut b).is_ok());
    assert_eq!(b, [0; 4], "no event pending");
}

#[test]
fn device_tree_blob() {
    let mut m = Microvm::new(config("acpi=off")).unwrap();
    m.attach_virtio(rng()).unwrap();
    m.machine_done().unwrap();
    let fdt = fw_cfg_file(&m, "etc/fdt").unwrap();
    assert_eq!(fdt.len(), 1 << 20);
    assert_eq!(&fdt[..4], &0xd00d_feedu32.to_be_bytes());
    let has = |s: &[u8]| fdt.windows(s.len()).any(|w| w == s);
    assert!(has(b"virtio_mmio@feb00e00\0"));
    assert!(has(b"ioapic1@fec00000\0"));
    assert!(!has(b"ioapic2@"));
    assert!(has(b"serial@3f8\0") && has(b"rtc@70\0"));
    assert!(has(b"linux,microvm\0"));
}

#[test]
fn property_errors() {
    let mut p = MicrovmProps::default();
    assert_eq!(p.set("rtc", "maybe").unwrap_err(), "Parameter 'rtc' does not accept value 'maybe'");
    assert_eq!(
        p.set("isa-serial", "2").unwrap_err(),
        "Parameter 'isa-serial' expects 'on' or 'off'"
    );
    assert_eq!(p.set("bogus", "on").unwrap_err(), "Property 'microvm-machine.bogus' not found");
    assert_eq!(
        p.set("oem-id", "1234567").unwrap_err(),
        "User specified oem-id value is bigger than 6 bytes in size"
    );
    assert_eq!(
        p.set("oem-table-id", "123456789").unwrap_err(),
        "User specified oem-table-id value is bigger than 8 bytes in size"
    );
    p.set_all("pit=off,pic=on,x-option-roms=no,usb=yes").unwrap();
    assert_eq!((p.pit, p.pic, p.option_roms, p.usb), (OnOffAuto::Off, OnOffAuto::On, false, true));
    assert_eq!(MicrovmProps::default().oem_table_id, "BXPC    ");
}

#[test]
fn machine_errors() {
    let mut cfg = config("");
    cfg.cpus = 289;
    assert_eq!(
        Microvm::new(cfg).unwrap_err(),
        "Invalid SMP CPUs 289. The max CPUs supported by machine 'microvm' is 288"
    );
    let mut cfg = config("");
    cfg.cpus = 4;
    cfg.max_cpus = 2;
    assert_eq!(
        Microvm::new(cfg).unwrap_err(),
        "Invalid CPU topology: maxcpus must be equal to or greater than smp: sockets (2) * \
         cores (1) * threads (1) == maxcpus (2) < smp_cpus (4)"
    );
    assert!(Microvm::new(config("usb=on")).is_err());
    assert!(Microvm::new(config("pcie=on")).is_err());
    // Without ACPI QEMU ignores both.
    assert!(Microvm::new(config("acpi=off,usb=on,pcie=on")).is_ok());
}
