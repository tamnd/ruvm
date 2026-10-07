// SPDX-License-Identifier: GPL-2.0-or-later

//! Tests of the ICH9 AHCI controller, following QEMU's `tests/qtest/ahci-test.c` and the libqos
//! AHCI driver: PCI config cycles through 0xcf8/0xcfc, register accesses to the ABAR, and
//! command lists, tables and buffers in guest RAM.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_hw_pci::regs::*;
use ruvm_hw_pci::*;
use ruvm_hw_storage::{
    BlockBackend, DmaMemory, DriveConfig, Ich9Ahci, Ich9AhciVmState, PCI_CLASS_STORAGE_SATA,
    PCI_DEVICE_ID_INTEL_82801IR, PCI_VENDOR_ID_INTEL, VecBackend,
};
use ruvm_mem::{AddressSpace, Endian, MemTxAttrs, MemorySystem};

const ATTRS: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const DEVFN: u8 = 0x10;
const ABAR: u64 = 0xfebf_0000;
const IDP: u64 = 0xc000;
const RAM_SIZE: u64 = 16 << 20;
const DISK_SECTORS: u64 = 2048;

// Host registers.
const CAP: u64 = 0x00;
const GHC: u64 = 0x04;
const IS: u64 = 0x08;
const PI: u64 = 0x0c;
const VS: u64 = 0x10;

// Port registers.
const PX_CLB: u64 = 0x00;
const PX_CLBU: u64 = 0x04;
const PX_FB: u64 = 0x08;
const PX_FBU: u64 = 0x0c;
const PX_IS: u64 = 0x10;
const PX_IE: u64 = 0x14;
const PX_CMD: u64 = 0x18;
const PX_TFD: u64 = 0x20;
const PX_SIG: u64 = 0x24;
const PX_SSTS: u64 = 0x28;
const PX_SCTL: u64 = 0x2c;
const PX_SERR: u64 = 0x30;
const PX_SACT: u64 = 0x34;
const PX_CI: u64 = 0x38;

const GHC_HR: u32 = 1 << 0;
const GHC_IE: u32 = 1 << 1;
const GHC_AE: u32 = 1 << 31;

const CMD_ST: u32 = 1 << 0;
const CMD_SUD: u32 = 1 << 1;
const CMD_POD: u32 = 1 << 2;
const CMD_FRE: u32 = 1 << 4;
const CMD_FR: u32 = 1 << 14;
const CMD_CR: u32 = 1 << 15;

const IS_DHRS: u32 = 1 << 0;
const IS_PSS: u32 = 1 << 1;
const IS_SDBS: u32 = 1 << 3;
const IS_TFES: u32 = 1 << 30;

const OPT_ATAPI: u16 = 1 << 5;
const OPT_WRITE: u16 = 1 << 6;

const CMD_READ_DMA: u8 = 0xc8;
const CMD_READ_DMA_EXT: u8 = 0x25;
const CMD_WRITE_DMA: u8 = 0xca;
const CMD_WRITE_DMA_EXT: u8 = 0x35;
const CMD_READ_PIO: u8 = 0x20;
const CMD_WRITE_PIO: u8 = 0x30;
const CMD_IDENTIFY: u8 = 0xec;
const CMD_FLUSH: u8 = 0xe7;
const CMD_FLUSH_EXT: u8 = 0xea;
const CMD_SET_FEATURES: u8 = 0xef;
const CMD_PACKET: u8 = 0xa0;
const CMD_READ_FPDMA: u8 = 0x60;
const CMD_WRITE_FPDMA: u8 = 0x61;

struct Env {
    mem_as: Arc<AddressSpace>,
    io_as: Arc<AddressSpace>,
    ahci: Arc<Ich9Ahci>,
    levels: Arc<[AtomicI32; 4]>,
    msgs: Arc<Mutex<Vec<(u64, u32)>>>,
    next: Mutex<u64>,
    /// Command list and FIS area of each started port.
    ports: Mutex<[(u64, u64); 6]>,
}

impl Env {
    /// A machine with an ICH9 AHCI controller in slot 2 and a hard disk on port 0.
    fn new() -> (Env, VecBackend) {
        let disk = VecBackend::new((DISK_SECTORS * 512) as usize);
        let env = Env::bare();
        env.ahci
            .attach_drive(
                0,
                DriveConfig {
                    serial: Some("testdisk".into()),
                    version: Some("version".into()),
                    ..DriveConfig::hd()
                },
                Some(Arc::new(disk.clone())),
            )
            .unwrap();
        env.boot();
        (env, disk)
    }

    fn bare() -> Env {
        let mem = Arc::new(MemorySystem::new());
        let sysmem = mem.new_container("system", 1 << 64).unwrap();
        let io = mem.new_container("io", 1 << 16).unwrap();
        let ram = mem.new_ram("ram", RAM_SIZE).unwrap();
        mem.add_subregion(sysmem, 0, ram).unwrap();
        let mem_as = mem.address_space_init(sysmem, "memory").unwrap();
        let io_as = mem.address_space_init(io, "I/O").unwrap();
        let bus = PciBus::new_root("pci.0", Arc::clone(&mem), sysmem, io, 0);
        let host = PciHostState::new(Arc::clone(&bus));
        host.map_ioports(&mem, io).unwrap();

        let levels: Arc<[AtomicI32; 4]> = Arc::new(Default::default());
        let l = Arc::clone(&levels);
        bus.set_irqs(Arc::new(move |irq, level| l[irq as usize].store(level, Ordering::SeqCst)), 4);
        bus.set_map_irq(Arc::new(pci_swizzle_map_irq_fn));
        let msgs: Arc<Mutex<Vec<(u64, u32)>>> = Arc::default();
        let m = Arc::clone(&msgs);
        bus.set_msi_handler(Some(Arc::new(move |a, d| m.lock().unwrap().push((a, d)))));

        let dma: Arc<dyn DmaMemory> = mem_as.clone();
        let ahci = Ich9Ahci::new(&bus, dma, Some(DEVFN)).unwrap();
        // The host bridge would own these in a real machine; keep them alive.
        std::mem::forget(host);
        Env {
            mem_as,
            io_as,
            ahci,
            levels,
            msgs,
            next: Mutex::new(0x10_0000),
            ports: Mutex::new([(0, 0); 6]),
        }
    }

    /// Machine reset, then what the firmware does: map the BARs and enable the function.
    fn boot(&self) {
        self.ahci.pci_device().reset();
        self.cfg_writel(PCI_BASE_ADDRESS_0 as u8 + 5 * 4, ABAR as u32);
        self.cfg_writel(PCI_BASE_ADDRESS_0 as u8 + 4 * 4, IDP as u32);
        self.cfg_writew(PCI_COMMAND as u8, 0x7);
    }

    // PCI config space.

    fn cfg_addr(off: u8) -> u32 {
        (1 << 31) | (u32::from(DEVFN) << 8) | u32::from(off & !3)
    }

    fn cfg_readl(&self, off: u8) -> u32 {
        self.outl(0xcf8, Self::cfg_addr(off));
        self.io_as.load(0xcfc, 4, Endian::Little, ATTRS).0 as u32
    }

    fn cfg_readw(&self, off: u8) -> u16 {
        self.outl(0xcf8, Self::cfg_addr(off));
        self.io_as.load(0xcfc + u64::from(off & 2), 2, Endian::Little, ATTRS).0 as u16
    }

    fn cfg_readb(&self, off: u8) -> u8 {
        self.outl(0xcf8, Self::cfg_addr(off));
        self.io_as.load(0xcfc + u64::from(off & 3), 1, Endian::Little, ATTRS).0 as u8
    }

    fn cfg_writel(&self, off: u8, v: u32) {
        self.outl(0xcf8, Self::cfg_addr(off));
        self.outl(0xcfc, v);
    }

    fn cfg_writew(&self, off: u8, v: u16) {
        self.outl(0xcf8, Self::cfg_addr(off));
        let r = self.io_as.store(0xcfc + u64::from(off & 2), 2, v.into(), Endian::Little, ATTRS);
        assert!(r.is_ok());
    }

    fn outl(&self, port: u64, v: u32) {
        assert!(self.io_as.store(port, 4, v.into(), Endian::Little, ATTRS).is_ok());
    }

    fn inl(&self, port: u64) -> u32 {
        self.io_as.load(port, 4, Endian::Little, ATTRS).0 as u32
    }

    // MMIO and RAM.

    fn readl(&self, addr: u64) -> u32 {
        let (v, r) = self.mem_as.load(addr, 4, Endian::Little, ATTRS);
        assert!(r.is_ok());
        v as u32
    }

    fn writel(&self, addr: u64, v: u32) {
        assert!(self.mem_as.store(addr, 4, v.into(), Endian::Little, ATTRS).is_ok());
    }

    fn hba(&self, reg: u64) -> u32 {
        self.readl(ABAR + reg)
    }

    fn set_hba(&self, reg: u64, v: u32) {
        self.writel(ABAR + reg, v);
    }

    fn px(&self, port: u64, reg: u64) -> u32 {
        self.readl(ABAR + 0x100 + port * 0x80 + reg)
    }

    fn set_px(&self, port: u64, reg: u64, v: u32) {
        self.writel(ABAR + 0x100 + port * 0x80 + reg, v);
    }

    fn memread(&self, addr: u64, len: usize) -> Vec<u8> {
        let mut b = vec![0; len];
        assert!(self.mem_as.read(addr, ATTRS, &mut b).is_ok());
        b
    }

    fn memwrite(&self, addr: u64, data: &[u8]) {
        assert!(self.mem_as.write(addr, ATTRS, data).is_ok());
    }

    /// A zeroed guest buffer.
    fn alloc(&self, len: u64, align: u64) -> u64 {
        let mut n = self.next.lock().unwrap();
        let addr = (*n + align - 1) & !(align - 1);
        *n = addr + len;
        assert!(*n < RAM_SIZE);
        self.memwrite(addr, &vec![0; len as usize]);
        addr
    }

    // The libqos AHCI driver.

    /// `ahci_hba_enable()` for one port: set AE, give the port a command list and FIS area,
    /// start FIS receive and the command engine, then enable interrupts.
    fn start_port(&self, port: u64) {
        self.set_hba(GHC, self.hba(GHC) | GHC_AE);
        let cmd = self.px(port, PX_CMD);
        assert_eq!(cmd & (CMD_ST | CMD_CR | CMD_FRE | CMD_FR), 0);

        let clb = self.alloc(1024, 1024);
        let fb = self.alloc(256, 256);
        self.set_px(port, PX_CLB, clb as u32);
        self.set_px(port, PX_CLBU, (clb >> 32) as u32);
        self.set_px(port, PX_FB, fb as u32);
        self.set_px(port, PX_FBU, (fb >> 32) as u32);
        assert_eq!(self.px(port, PX_CLB), clb as u32);
        assert_eq!(self.px(port, PX_FB), fb as u32);
        self.ports.lock().unwrap()[port as usize] = (clb, fb);

        self.set_px(port, PX_SERR, 0xffff_ffff);
        self.set_px(port, PX_IS, 0xffff_ffff);
        assert_eq!(self.px(port, PX_IS), 0);
        self.set_px(port, PX_IE, 0xffff_ffff);
        assert_eq!(self.px(port, PX_IE), 0xfdc0_00ff);

        self.set_px(port, PX_CMD, self.px(port, PX_CMD) | CMD_FRE);
        assert_ne!(self.px(port, PX_CMD) & CMD_FR, 0);
        // The device sends its signature in a D2H FIS as soon as FIS receive is on.
        assert_eq!(self.px(port, PX_IS), IS_DHRS);
        self.set_px(port, PX_IS, IS_DHRS);

        self.set_px(port, PX_CMD, self.px(port, PX_CMD) | CMD_ST);
        assert_ne!(self.px(port, PX_CMD) & CMD_CR, 0);
        self.set_hba(GHC, self.hba(GHC) | GHC_IE);
    }

    fn fb(&self, port: u64) -> u64 {
        self.ports.lock().unwrap()[port as usize].1
    }

    fn clb(&self, port: u64) -> u64 {
        self.ports.lock().unwrap()[port as usize].0
    }

    /// Builds the command in `slot` and issues it, then returns the PRD byte count.
    fn issue(&self, port: u64, slot: u32, c: &Cmd, buf: u64, len: u64) -> u32 {
        self.prepare(port, slot, c, buf, len);
        if c.ncq {
            self.set_px(port, PX_SACT, 1 << slot);
        }
        self.set_px(port, PX_CI, 1 << slot);
        self.prdbc(port, slot)
    }

    fn prepare(&self, port: u64, slot: u32, c: &Cmd, buf: u64, len: u64) {
        // PRDs of at most 4 KiB, so that transfers use several entries.
        let mut prdt = Vec::new();
        let mut off = 0;
        while off < len {
            let n = (len - off).min(4096);
            prdt.push((buf + off, n));
            off += n;
        }
        let tbl = self.alloc(0x80 + 16 * prdt.len() as u64, 0x80);
        self.memwrite(tbl, &c.fis);
        if let Some(acmd) = &c.acmd {
            self.memwrite(tbl + 0x40, acmd);
        }
        for (i, &(addr, n)) in prdt.iter().enumerate() {
            let mut e = [0u8; 16];
            e[0..8].copy_from_slice(&addr.to_le_bytes());
            e[12..16].copy_from_slice(&((n - 1) as u32).to_le_bytes());
            self.memwrite(tbl + 0x80 + 16 * i as u64, &e);
        }

        let mut opts: u16 = 5;
        if c.write {
            opts |= OPT_WRITE;
        }
        if c.acmd.is_some() {
            opts |= OPT_ATAPI;
        }
        let mut hdr = [0u8; 32];
        hdr[0..2].copy_from_slice(&opts.to_le_bytes());
        hdr[2..4].copy_from_slice(&(prdt.len() as u16).to_le_bytes());
        // A stale byte count, which the controller must reset.
        hdr[4..8].copy_from_slice(&0xdead_beefu32.to_le_bytes());
        hdr[8..16].copy_from_slice(&tbl.to_le_bytes());
        self.memwrite(self.clb(port) + u64::from(slot) * 32, &hdr);
    }

    fn prdbc(&self, port: u64, slot: u32) -> u32 {
        let b = self.memread(self.clb(port) + u64::from(slot) * 32 + 4, 4);
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    /// Checks that a command finished cleanly with exactly `is` raised, and clears it.
    fn expect_done(&self, port: u64, is: u32) {
        assert_eq!(self.px(port, PX_CI), 0);
        assert_eq!(self.px(port, PX_IS), is);
        assert_eq!(self.px(port, PX_TFD), 0x50);
        self.set_px(port, PX_IS, is);
        assert_eq!(self.hba(IS), 0);
    }

    fn disk_write(&self, port: u64, cmd: u8, lba: u64, data: &[u8]) {
        let buf = self.alloc(data.len() as u64, 2);
        self.memwrite(buf, data);
        let c = Cmd::ata(cmd, lba, (data.len() / 512) as u16).write();
        assert_eq!(self.issue(port, 0, &c, buf, data.len() as u64), data.len() as u32);
    }

    fn disk_read(&self, port: u64, cmd: u8, lba: u64, sectors: u16) -> Vec<u8> {
        let len = u64::from(sectors) * 512;
        let buf = self.alloc(len, 2);
        let c = Cmd::ata(cmd, lba, sectors);
        assert_eq!(self.issue(port, 0, &c, buf, len), len as u32);
        self.memread(buf, len as usize)
    }
}

/// A command table: the H2D register FIS and, for ATAPI, the packet.
struct Cmd {
    fis: [u8; 20],
    acmd: Option<[u8; 16]>,
    write: bool,
    ncq: bool,
}

impl Cmd {
    fn ata(cmd: u8, lba: u64, count: u16) -> Cmd {
        let mut fis = [0u8; 20];
        fis[0] = 0x27;
        fis[1] = 0x80;
        fis[2] = cmd;
        fis[4] = lba as u8;
        fis[5] = (lba >> 8) as u8;
        fis[6] = (lba >> 16) as u8;
        // LBA mode; the 28-bit commands take LBA bits 24-27 from here.
        fis[7] = 0x40 | ((lba >> 24) as u8 & 0xf);
        fis[8] = (lba >> 24) as u8;
        fis[9] = (lba >> 32) as u8;
        fis[10] = (lba >> 40) as u8;
        fis[12] = count as u8;
        fis[13] = (count >> 8) as u8;
        Cmd { fis, acmd: None, write: false, ncq: false }
    }

    fn ncq(cmd: u8, tag: u8, lba: u64, count: u16) -> Cmd {
        let mut c = Cmd::ata(cmd, lba, 0);
        c.fis[7] = 0x40;
        // The count is in the feature registers and the tag in the count register.
        c.fis[3] = count as u8;
        c.fis[11] = (count >> 8) as u8;
        c.fis[12] = tag << 3;
        c.ncq = true;
        c.write = cmd == CMD_WRITE_FPDMA;
        c
    }

    fn atapi(packet: &[u8], dma: bool, bcl: u16) -> Cmd {
        let mut c = Cmd::ata(CMD_PACKET, 0, 0);
        c.fis[3] = u8::from(dma);
        c.fis[5] = bcl as u8;
        c.fis[6] = (bcl >> 8) as u8;
        c.fis[7] = 0;
        let mut p = [0u8; 16];
        p[..packet.len()].copy_from_slice(packet);
        c.acmd = Some(p);
        c
    }

    fn write(mut self) -> Cmd {
        self.write = true;
        self
    }
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len).map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed ^ (i >> 9) as u8)).collect()
}

/// An ATA string: two characters per word, the first in the high byte.
fn ata_string(id: &[u8], word: usize, words: usize) -> String {
    let mut s = String::new();
    for w in word..word + words {
        s.push(id[2 * w + 1] as char);
        s.push(id[2 * w] as char);
    }
    s.trim_end().to_string()
}

fn word(id: &[u8], w: usize) -> u16 {
    u16::from_le_bytes([id[2 * w], id[2 * w + 1]])
}

#[test]
fn pci_spec() {
    let (env, _) = Env::new();
    assert_eq!(env.cfg_readw(PCI_VENDOR_ID as u8), PCI_VENDOR_ID_INTEL);
    assert_eq!(env.cfg_readw(PCI_DEVICE_ID as u8), PCI_DEVICE_ID_INTEL_82801IR);
    assert_eq!(env.cfg_readb(PCI_REVISION_ID as u8), 2);
    assert_eq!(env.cfg_readb(PCI_CLASS_PROG as u8), 1);
    assert_eq!(env.cfg_readw(PCI_CLASS_DEVICE as u8), PCI_CLASS_STORAGE_SATA);
    assert_eq!(env.cfg_readb(PCI_HEADER_TYPE as u8) & 0x7f, 0);
    // Realize sets a cache line size of 32 bytes, which the PCI reset then clears, as in QEMU.
    assert_eq!(env.cfg_readb(0x0c), 0);
    assert_eq!(Env::bare().ahci.pci_device().config_read(0x0c, 1), 0x08);
    assert_eq!(env.cfg_readb(PCI_INTERRUPT_PIN as u8), 1);
    assert_eq!(env.cfg_readb(0x90), 0x40);
    assert_ne!(env.cfg_readw(PCI_STATUS as u8) & PCI_STATUS_CAP_LIST, 0);

    // The capability list: MSI, then the SATA capability.
    let mut caps = Vec::new();
    let mut p = env.cfg_readb(PCI_CAPABILITY_LIST as u8);
    while p != 0 {
        caps.push((p, env.cfg_readb(p)));
        p = env.cfg_readb(p + 1);
    }
    assert_eq!(caps, [(0x80, PCI_CAP_ID_MSI), (0xa8, PCI_CAP_ID_SATA)]);
    assert_eq!(env.cfg_readw(0xa8 + 2), 0x10);
    assert_eq!(env.cfg_readl(0xa8 + 4), 0x48);
    assert_ne!(env.cfg_readw(0x80 + PCI_MSI_FLAGS as u8) & PCI_MSI_FLAGS_64BIT, 0);

    // BAR5 is 4 KiB of memory, BAR4 32 bytes of I/O; the others are absent.
    let bar = |n: u8| PCI_BASE_ADDRESS_0 as u8 + n * 4;
    env.cfg_writel(bar(5), 0xffff_ffff);
    assert_eq!(env.cfg_readl(bar(5)), 0xffff_f000);
    env.cfg_writel(bar(4), 0xffff_ffff);
    assert_eq!(env.cfg_readl(bar(4)) & 0xffff, 0xffe1);
    for n in 0..4 {
        env.cfg_writel(bar(n), 0xffff_ffff);
        assert_eq!(env.cfg_readl(bar(n)), 0);
    }
}

#[test]
fn hba_spec() {
    let (env, _) = Env::new();
    let cap = env.hba(CAP);
    assert_eq!(cap & 0x1f, 5, "six ports");
    assert_eq!((cap >> 8) & 0x1f, 31, "32 command slots");
    assert_eq!((cap >> 20) & 0xf, 1, "gen 1 speed");
    assert_ne!(cap & (1 << 18), 0, "AHCI only");
    assert_ne!(cap & (1 << 30), 0, "NCQ");
    assert_ne!(cap & (1 << 31), 0, "64-bit addressing");
    assert_eq!(env.hba(GHC), GHC_AE);
    assert_eq!(env.hba(IS), 0);
    assert_eq!(env.hba(PI), 0x3f);
    assert_eq!(env.hba(VS), 0x0001_0000);
    // Registers past VS read as zero; CAP, PI and VS are read-only.
    assert_eq!(env.hba(0x2c), 0);
    env.set_hba(CAP, 0);
    env.set_hba(PI, 0);
    env.set_hba(VS, 0);
    assert_eq!(env.hba(CAP), cap);
    assert_eq!(env.hba(PI), 0x3f);
    assert_eq!(env.hba(VS), 0x0001_0000);

    // Narrow and unaligned reads come out of the aligned registers.
    assert_eq!(env.mem_as.load(ABAR + VS + 2, 2, Endian::Little, ATTRS).0, 1);
    assert_eq!(env.mem_as.load(ABAR + CAP, 1, Endian::Little, ATTRS).0, u64::from(cap & 0xff));

    for port in 0..6 {
        assert_eq!(env.px(port, PX_CMD), CMD_SUD | CMD_POD);
        assert_eq!(env.px(port, PX_IS), 0);
        assert_eq!(env.px(port, PX_IE), 0);
        assert_eq!(env.px(port, PX_TFD), 0x7f);
        assert_eq!(env.px(port, PX_SIG), 0xffff_ffff);
        assert_eq!(env.px(port, PX_SCTL), 0);
        assert_eq!(env.px(port, PX_SACT), 0);
        assert_eq!(env.px(port, PX_CI), 0);
        let ssts = if port == 0 { 0x113 } else { 0 };
        assert_eq!(env.px(port, PX_SSTS), ssts);
    }
}

#[test]
fn signature_and_initial_fis() {
    let (env, _) = Env::new();
    env.start_port(0);
    assert_eq!(env.px(0, PX_SIG), 0x101);
    // After a port reset a disk reports DSC and DF but not DRDY, and error 1 (no error).
    assert_eq!(env.px(0, PX_TFD), 0x130);
    let fis = env.memread(env.fb(0) + 0x40, 20);
    assert_eq!(&fis[..8], &[0x34, 0x40, 0x30, 0x01, 0x01, 0x00, 0x00, 0xa0]);
    assert_eq!(fis[12], 1);
}

#[test]
fn identify() {
    let (env, _) = Env::new();
    env.start_port(0);
    let buf = env.alloc(512, 2);
    let c = Cmd::ata(CMD_IDENTIFY, 0, 0);
    assert_eq!(env.issue(0, 0, &c, buf, 512), 512);
    // PIO data in: a PIO Setup FIS with the I bit, then the D2H FIS.
    env.expect_done(0, IS_DHRS | IS_PSS);
    let pio = env.memread(env.fb(0) + 0x20, 20);
    assert_eq!(pio[0], 0x5f);
    assert_eq!(pio[1], 0x40);
    assert_eq!(u16::from_le_bytes([pio[16], pio[17]]), 512);

    let id = env.memread(buf, 512);
    assert_eq!(ata_string(&id, 10, 10), "testdisk");
    assert_eq!(ata_string(&id, 23, 4), "version");
    assert_eq!(ata_string(&id, 27, 20), "QEMU HARDDISK");
    assert_eq!(u32::from(word(&id, 60)) | u32::from(word(&id, 61)) << 16, DISK_SECTORS as u32);
    assert_eq!(u64::from(word(&id, 100)) | u64::from(word(&id, 101)) << 16, DISK_SECTORS);
    // NCQ with 32 tags.
    assert_ne!(word(&id, 76) & (1 << 8), 0);
    assert_eq!(word(&id, 75), 31);
}

#[test]
fn default_serial_numbers() {
    let env = Env::bare();
    let d0 = VecBackend::new(512 * 64);
    let d1 = VecBackend::new(512 * 64);
    env.ahci.attach_drive(0, DriveConfig::hd(), Some(Arc::new(d0))).unwrap();
    env.ahci.attach_drive(1, DriveConfig::hd(), Some(Arc::new(d1))).unwrap();
    assert!(env.ahci.attach_drive(1, DriveConfig::hd(), None).is_err());
    assert!(env.ahci.attach_drive(6, DriveConfig::cdrom(), None).is_err());
    env.boot();

    let mut serials = Vec::new();
    for port in 0..2 {
        env.start_port(port);
        let buf = env.alloc(512, 2);
        env.issue(port, 0, &Cmd::ata(CMD_IDENTIFY, 0, 0), buf, 512);
        let s = ata_string(&env.memread(buf, 512), 10, 10);
        assert!(s.starts_with("QM") && s.len() == 7, "{s}");
        serials.push(s[2..].parse::<u32>().unwrap());
    }
    // Two numbers per port, as each AHCI port is an IDE bus with two units.
    assert_eq!(serials[1], serials[0] + 2);
}

#[test]
fn dma_round_trip() {
    let (env, disk) = Env::new();
    env.start_port(0);
    let data = pattern(16 * 512, 1);
    env.disk_write(0, CMD_WRITE_DMA, 7, &data);
    env.expect_done(0, IS_DHRS);
    assert_eq!(&disk.contents()[7 * 512..23 * 512], &data[..]);

    let back = env.disk_read(0, CMD_READ_DMA, 7, 16);
    env.expect_done(0, IS_DHRS);
    assert_eq!(back, data);

    let d2h = env.memread(env.fb(0) + 0x40, 4);
    assert_eq!(d2h, [0x34, 0x40, 0x50, 0x00]);
}

#[test]
fn dma_ext_round_trip() {
    let (env, disk) = Env::new();
    env.start_port(0);
    let data = pattern(40 * 512, 2);
    env.disk_write(0, CMD_WRITE_DMA_EXT, 100, &data);
    env.expect_done(0, IS_DHRS);
    assert_eq!(&disk.contents()[100 * 512..140 * 512], &data[..]);

    disk.fill(200 * 512, &pattern(512, 9));
    let back = env.disk_read(0, CMD_READ_DMA_EXT, 200, 1);
    env.expect_done(0, IS_DHRS);
    assert_eq!(back, pattern(512, 9));
}

#[test]
fn pio_round_trip() {
    let (env, disk) = Env::new();
    env.start_port(0);
    let data = pattern(4 * 512, 3);
    env.disk_write(0, CMD_WRITE_PIO, 3, &data);
    // PIO data out: the first PIO Setup FIS has no I bit, the later ones do.
    env.expect_done(0, IS_DHRS | IS_PSS);
    assert_eq!(&disk.contents()[3 * 512..7 * 512], &data[..]);

    let back = env.disk_read(0, CMD_READ_PIO, 3, 4);
    env.expect_done(0, IS_DHRS | IS_PSS);
    assert_eq!(back, data);
}

#[test]
fn ncq_simple() {
    let (env, disk) = Env::new();
    env.start_port(0);
    // NCQ leaves the error register alone, so clear the reset value of 1 first, as the
    // IDENTIFY a driver sends would.
    env.issue(0, 0, &Cmd::ata(CMD_FLUSH, 0, 0), 0, 0);
    env.expect_done(0, IS_DHRS);
    let data = pattern(8 * 512, 4);
    let buf = env.alloc(data.len() as u64, 2);
    env.memwrite(buf, &data);
    let c = Cmd::ncq(CMD_WRITE_FPDMA, 5, 50, 8);
    assert_eq!(env.issue(0, 5, &c, buf, data.len() as u64), 0xdead_beef);
    assert_eq!(env.px(0, PX_SACT), 0);
    env.expect_done(0, IS_SDBS);
    assert_eq!(&disk.contents()[50 * 512..58 * 512], &data[..]);
    let sdb = env.memread(env.fb(0) + 0x58, 8);
    assert_eq!(sdb, [0xa1, 0x40, 0x50, 0x00, 0x20, 0x00, 0x00, 0x00]);

    let out = env.alloc(data.len() as u64, 2);
    let c = Cmd::ncq(CMD_READ_FPDMA, 3, 50, 8);
    env.issue(0, 3, &c, out, data.len() as u64);
    assert_eq!(env.px(0, PX_SACT), 0);
    env.expect_done(0, IS_SDBS);
    assert_eq!(env.memread(out, data.len()), data);
}

#[test]
fn ncq_short_prdt_overflows() {
    let (env, _) = Env::new();
    env.start_port(0);
    let buf = env.alloc(512, 2);
    let c = Cmd::ncq(CMD_READ_FPDMA, 0, 0, 4);
    env.issue(0, 0, &c, buf, 512);
    assert_ne!(env.px(0, PX_IS) & (1 << 24), 0, "OFS");
    assert_eq!(env.px(0, PX_SACT), 1);
}

#[test]
fn flush() {
    let (env, disk) = Env::new();
    env.start_port(0);
    for (i, cmd) in [CMD_FLUSH, CMD_FLUSH_EXT].into_iter().enumerate() {
        env.issue(0, 0, &Cmd::ata(cmd, 0, 0), 0, 0);
        env.expect_done(0, IS_DHRS);
        assert_eq!(disk.flush_count(), i as u64 + 1);
    }
}

#[test]
fn set_features_write_cache() {
    let (env, _) = Env::new();
    env.start_port(0);
    // Disable the write cache, then check IDENTIFY word 85.
    let mut c = Cmd::ata(CMD_SET_FEATURES, 0, 0);
    c.fis[3] = 0x82;
    env.issue(0, 0, &c, 0, 0);
    env.expect_done(0, IS_DHRS);
    let buf = env.alloc(512, 2);
    env.issue(0, 0, &Cmd::ata(CMD_IDENTIFY, 0, 0), buf, 512);
    env.expect_done(0, IS_DHRS | IS_PSS);
    assert_eq!(word(&env.memread(buf, 512), 85) & (1 << 5), 0);
}

#[test]
fn max_and_min_lba() {
    let (env, disk) = Env::new();
    env.start_port(0);
    for (cmd_w, cmd_r) in [(CMD_WRITE_DMA, CMD_READ_DMA), (CMD_WRITE_DMA_EXT, CMD_READ_DMA_EXT)] {
        for lba in [0, DISK_SECTORS - 1] {
            let data = pattern(512, lba as u8 ^ cmd_w);
            env.disk_write(0, cmd_w, lba, &data);
            env.expect_done(0, IS_DHRS);
            assert_eq!(&disk.contents()[lba as usize * 512..][..512], &data[..]);
            assert_eq!(env.disk_read(0, cmd_r, lba, 1), data);
            env.expect_done(0, IS_DHRS);
        }
    }
}

#[test]
fn out_of_range_fails() {
    let (env, _) = Env::new();
    env.start_port(0);
    let buf = env.alloc(1024, 2);
    env.issue(0, 0, &Cmd::ata(CMD_READ_DMA_EXT, DISK_SECTORS - 1, 2), buf, 1024);
    assert_ne!(env.px(0, PX_IS) & IS_TFES, 0);
    assert_ne!(env.px(0, PX_TFD) & 1, 0, "ERR");
    // A failed command stays issued until the port is reset.
    assert_eq!(env.px(0, PX_CI), 1);
}

#[test]
fn backend_error_is_reported() {
    let (env, disk) = Env::new();
    env.start_port(0);
    disk.set_failing(true);
    let buf = env.alloc(512, 2);
    env.issue(0, 0, &Cmd::ata(CMD_READ_DMA, 0, 1), buf, 512);
    assert_ne!(env.px(0, PX_IS) & IS_TFES, 0);
    assert_eq!(env.px(0, PX_TFD), 0x441);
    assert_eq!(env.px(0, PX_CI), 1);

    // NCQ errors come in a Set Device Bits FIS.
    let (env, disk) = Env::new();
    env.start_port(0);
    disk.set_failing(true);
    env.issue(0, 2, &Cmd::ncq(CMD_READ_FPDMA, 2, 0, 1), buf, 512);
    assert_ne!(env.px(0, PX_IS) & IS_TFES, 0);
    assert_eq!(env.px(0, PX_SACT), 1 << 2);
}

#[test]
fn port_reset() {
    let (env, _) = Env::new();
    env.start_port(0);
    // A stuck command.
    env.issue(0, 0, &Cmd::ata(CMD_READ_DMA, DISK_SECTORS, 1), env.alloc(512, 2), 512);
    assert_eq!(env.px(0, PX_CI), 1);
    env.set_px(0, PX_IS, 0xffff_ffff);

    // COMRESET: SCTL.DET 1 then 0 resets the port, and the device sends its signature again.
    env.set_px(0, PX_CMD, env.px(0, PX_CMD) & !CMD_ST);
    env.set_px(0, PX_SCTL, 1);
    env.set_px(0, PX_SCTL, 0);
    assert_eq!(env.px(0, PX_CI), 0);
    assert_eq!(env.px(0, PX_TFD), 0x130);
    assert_eq!(env.px(0, PX_SIG), 0x101);
    assert_eq!(env.px(0, PX_IS), IS_DHRS);
    env.set_px(0, PX_IS, IS_DHRS);

    env.set_px(0, PX_CMD, env.px(0, PX_CMD) | CMD_ST);
    let data = pattern(512, 5);
    env.disk_write(0, CMD_WRITE_DMA, 1, &data);
    env.expect_done(0, IS_DHRS);
}

#[test]
fn hba_reset() {
    let (env, _) = Env::new();
    env.start_port(0);
    env.set_hba(GHC, GHC_AE | GHC_IE | GHC_HR);
    assert_eq!(env.hba(GHC), GHC_AE);
    assert_eq!(env.hba(IS), 0);
    for port in 0..6 {
        assert_eq!(env.px(port, PX_CMD) & (CMD_ST | CMD_FRE), 0);
        assert_eq!(env.px(port, PX_IS), 0);
        assert_eq!(env.px(port, PX_IE), 0);
        assert_eq!(env.px(port, PX_SIG), 0xffff_ffff);
        assert_eq!(env.px(port, PX_TFD), 0x7f);
    }
    assert_eq!(env.px(0, PX_CMD) & (CMD_CR | CMD_FR), 0);
    env.start_port(0);
    assert_eq!(env.px(0, PX_SIG), 0x101);
    let data = pattern(512, 6);
    env.disk_write(0, CMD_WRITE_DMA, 1, &data);
    env.expect_done(0, IS_DHRS);
}

#[test]
fn soft_reset_sequence() {
    let (env, _) = Env::new();
    env.start_port(0);
    // A control FIS with SRST set and "clear busy upon R_OK", then one with SRST clear.
    let mut c = Cmd::ata(0, 0, 0);
    c.fis[1] = 0;
    c.fis[15] = 1 << 2;
    env.prepare(0, 0, &c, 0, 0);
    let hdr = env.clb(0);
    env.memwrite(hdr, &(5u16 | 1 << 10).to_le_bytes());
    env.set_px(0, PX_CI, 1);
    assert_eq!(env.px(0, PX_CI), 0);

    c.fis[15] = 0;
    env.issue(0, 1, &c, 0, 0);
    // The port reset clears PxCI and sends the signature again.
    assert_eq!(env.px(0, PX_CI), 0);
    assert_eq!(env.px(0, PX_SIG), 0x101);
    assert_eq!(env.px(0, PX_IS), IS_DHRS);
}

#[test]
fn idp_window() {
    let (env, _) = Env::new();
    let cap = env.hba(CAP);
    env.outl(IDP + 0x10, CAP as u32);
    assert_eq!(env.inl(IDP + 0x10), 0);
    assert_eq!(env.inl(IDP + 0x14), cap);
    env.outl(IDP + 0x10, VS as u32 | 0x3003);
    assert_eq!(env.inl(IDP + 0x10), 0x010);
    assert_eq!(env.inl(IDP + 0x14), 0x0001_0000);
    // Writes through the window reach the registers.
    env.outl(IDP + 0x10, 0x100 + PX_IE as u32);
    env.outl(IDP + 0x14, 0x1);
    assert_eq!(env.px(0, PX_IE), 1);
    assert_eq!(env.inl(IDP), 0);
}

#[test]
fn intx() {
    let (env, _) = Env::new();
    env.start_port(0);
    // Slot 2, pin A: (slot + pin) % 4.
    let pirq = 2;
    assert_eq!(env.levels[pirq].load(Ordering::SeqCst), 0);
    env.disk_write(0, CMD_WRITE_DMA, 0, &pattern(512, 7));
    assert_eq!(env.levels[pirq].load(Ordering::SeqCst), 1);
    assert_eq!(env.hba(IS), 1);
    env.set_px(0, PX_IS, IS_DHRS);
    assert_eq!(env.levels[pirq].load(Ordering::SeqCst), 0);

    // GHC.IE gates the line.
    env.set_hba(GHC, GHC_AE);
    env.disk_write(0, CMD_WRITE_DMA, 0, &pattern(512, 7));
    assert_eq!(env.levels[pirq].load(Ordering::SeqCst), 0);
    assert_eq!(env.hba(IS), 1);
    env.set_hba(GHC, GHC_AE | GHC_IE);
    assert_eq!(env.levels[pirq].load(Ordering::SeqCst), 1);
    assert!(env.msgs.lock().unwrap().is_empty());
}

#[test]
fn msi() {
    let (env, _) = Env::new();
    let cap = 0x80u8;
    env.cfg_writel(cap + PCI_MSI_ADDRESS_LO as u8, 0xfee0_0000);
    env.cfg_writel(cap + PCI_MSI_ADDRESS_HI as u8, 0);
    env.cfg_writew(cap + PCI_MSI_DATA_64 as u8, 0x4041);
    let flags = env.cfg_readw(cap + PCI_MSI_FLAGS as u8);
    env.cfg_writew(cap + PCI_MSI_FLAGS as u8, flags | PCI_MSI_FLAGS_ENABLE);

    env.start_port(0);
    env.disk_write(0, CMD_WRITE_DMA, 0, &pattern(512, 8));
    assert_eq!(*env.msgs.lock().unwrap(), [(0xfee0_0000, 0x4041)]);
    assert!(env.levels.iter().all(|l| l.load(Ordering::SeqCst) == 0));
}

#[test]
fn no_dma_without_bus_master() {
    let (env, disk) = Env::new();
    env.start_port(0);
    let buf = env.alloc(512, 2);
    let data = pattern(512, 10);
    env.memwrite(buf, &data);
    env.prepare(0, 0, &Cmd::ata(CMD_WRITE_DMA, 0, 1).write(), buf, 512);
    env.cfg_writew(PCI_COMMAND as u8, 0x3);
    env.set_px(0, PX_CI, 1);
    // The command list cannot be read, so nothing happens.
    assert_eq!(env.px(0, PX_CI), 1);
    assert_eq!(&disk.contents()[..512], &[0u8; 512][..]);

    env.cfg_writew(PCI_COMMAND as u8, 0x7);
    env.set_px(0, PX_CI, 1);
    env.expect_done(0, IS_DHRS);
    assert_eq!(&disk.contents()[..512], &data[..]);
}

fn with_cdrom() -> (Env, VecBackend) {
    let env = Env::bare();
    let disc: Vec<u8> = (0..16 * 2048).map(|i| (i / 2048) as u8 ^ (i as u8)).collect();
    let backend = VecBackend::from_vec(disc);
    let blk: Arc<dyn BlockBackend> = Arc::new(backend.clone());
    env.ahci.attach_drive(1, DriveConfig::cdrom(), Some(blk)).unwrap();
    env.boot();
    env.start_port(1);
    (env, backend)
}

#[test]
fn atapi_signature_and_identify() {
    let (env, _) = with_cdrom();
    assert_eq!(env.px(1, PX_SIG), 0xeb14_0101);
    let buf = env.alloc(512, 2);
    let mut c = Cmd::ata(0xa1, 0, 0);
    c.fis[7] = 0;
    env.issue(1, 0, &c, buf, 512);
    env.expect_done(1, IS_DHRS | IS_PSS);
    let id = env.memread(buf, 512);
    assert_eq!(word(&id, 0) >> 14, 2, "ATAPI device");
    assert_eq!(ata_string(&id, 27, 20), "QEMU DVD-ROM");
    // A hard disk command aborts.
    env.issue(1, 0, &Cmd::ata(CMD_IDENTIFY, 0, 0), buf, 512);
    assert_ne!(env.px(1, PX_IS) & IS_TFES, 0);
}

/// Runs a PIO data-in packet command and returns the data.
fn atapi_pio(env: &Env, packet: &[u8], len: u64) -> Vec<u8> {
    let buf = env.alloc(len.max(2), 2);
    let c = Cmd::atapi(packet, false, len as u16);
    assert_eq!(env.issue(1, 0, &c, buf, len), len as u32);
    env.expect_done(1, IS_DHRS | IS_PSS);
    env.memread(buf, len as usize)
}

#[test]
fn atapi_inquiry_capacity_sense() {
    let (env, _) = with_cdrom();
    let inq = atapi_pio(&env, &[0x12, 0, 0, 0, 36, 0], 36);
    assert_eq!(inq[0], 0x05, "CD/DVD device");
    assert_eq!(inq[1], 0x80, "removable");
    assert_eq!(&inq[8..16], b"QEMU    ");
    assert_eq!(&inq[16..32], b"QEMU DVD-ROM    ");
    assert_eq!(&inq[32..36], b"2.5+");

    // TEST UNIT READY: no data, just the D2H FIS.
    let c = Cmd::atapi(&[0x00], false, 0);
    env.issue(1, 0, &c, 0, 0);
    env.expect_done(1, IS_DHRS);

    let cap = atapi_pio(&env, &[0x25, 0, 0, 0, 0, 0, 0, 0, 0, 0], 8);
    assert_eq!(cap, [0, 0, 0, 15, 0, 0, 0x08, 0]);

    let sense = atapi_pio(&env, &[0x03, 0, 0, 0, 18, 0], 18);
    assert_eq!(sense[0], 0xf0);
    assert_eq!(sense[2], 0);

    let mode = atapi_pio(&env, &[0x5a, 0, 0x2a, 0, 0, 0, 0, 0, 28, 0], 28);
    assert_eq!(mode[8], 0x2a);

    let conf = atapi_pio(&env, &[0x46, 0, 0, 0, 0, 0, 0, 0, 20, 0], 20);
    // Two profiles, DVD-ROM and CD-ROM, and nothing else.
    assert_eq!(u32::from_be_bytes([conf[0], conf[1], conf[2], conf[3]]), 16);
    // The current profile: CD-ROM, as the medium is no bigger than a CD.
    assert_eq!(u16::from_be_bytes([conf[6], conf[7]]), 0x0008);
}

#[test]
fn atapi_read() {
    let (env, disc) = with_cdrom();
    let want = disc.contents()[3 * 2048..5 * 2048].to_vec();
    // READ(10) by PIO, then by DMA.
    let packet = [0x28, 0, 0, 0, 0, 3, 0, 0, 2, 0];
    assert_eq!(atapi_pio(&env, &packet, 4096), want);

    let buf = env.alloc(4096, 2);
    let c = Cmd::atapi(&packet, true, 0);
    assert_eq!(env.issue(1, 0, &c, buf, 4096), 4096);
    env.expect_done(1, IS_DHRS);
    assert_eq!(env.memread(buf, 4096), want);

    // Past the end of the disc: ILLEGAL REQUEST, LOGICAL BLOCK ADDRESS OUT OF RANGE.
    let c = Cmd::atapi(&[0x28, 0, 0, 0, 0, 15, 0, 0, 2, 0], true, 0);
    env.issue(1, 0, &c, buf, 4096);
    assert_ne!(env.px(1, PX_IS) & IS_TFES, 0);
    assert_eq!(env.px(1, PX_TFD) >> 12, 5);
}

#[test]
fn atapi_unknown_opcode() {
    let (env, _) = with_cdrom();
    env.issue(1, 0, &Cmd::atapi(&[0xff], false, 512), 0, 0);
    assert_ne!(env.px(1, PX_IS) & IS_TFES, 0);
    assert_eq!(env.px(1, PX_TFD) & 0xff, 0x41);
}

#[test]
fn empty_port_sends_no_signature() {
    let (env, _) = Env::new();
    // An empty port still answers FIS receive with a D2H FIS, as QEMU's does.
    env.start_port(3);
    assert_eq!(env.px(3, PX_SSTS), 0);
    env.set_px(3, PX_CI, 1);
    assert_eq!(env.px(3, PX_CI), 1);
    assert_eq!(env.px(3, PX_IS), 0);
}

#[test]
fn vmstate_round_trip() {
    let (src, disk) = Env::new();
    src.start_port(0);
    let buf = src.alloc(512, 2);
    src.issue(0, 0, &Cmd::ata(CMD_IDENTIFY, 0, 0), buf, 512);
    src.expect_done(0, IS_DHRS | IS_PSS);
    let data = pattern(512, 9);
    src.disk_write(0, CMD_WRITE_DMA, 3, &data);
    src.expect_done(0, IS_DHRS);

    let v = src.ahci.vmstate_save();
    assert_eq!(v.parent_obj.config.len(), 256);
    assert_eq!((v.ahci.ports, v.ahci.dev.len()), (6, 6));
    let p = &v.ahci.dev[0];
    assert_eq!(p.busy_slot, -1);
    assert_eq!(p.ifs0.identify_set, 1);
    assert_eq!(p.ifs0.identify_data.len(), 512);
    assert_eq!(p.cmd & (CMD_ST | CMD_CR | CMD_FRE | CMD_FR), CMD_ST | CMD_CR | CMD_FRE | CMD_FR);
    assert_eq!(p.scr_stat, 0);
    // An empty port still has its drive's task file.
    assert_eq!(v.ahci.dev[1].ifs0.status, 0x50);

    // The destination: the same image and a copy of guest RAM, then the device.
    let dst = Env::bare();
    let config = DriveConfig { serial: Some("other".into()), ..DriveConfig::hd() };
    dst.ahci.attach_drive(0, config, Some(Arc::new(disk.clone()))).unwrap();
    dst.boot();
    dst.memwrite(0, &src.memread(0, RAM_SIZE as usize));
    dst.ahci.vmstate_load(&v).unwrap();
    assert_eq!(dst.ahci.vmstate_save(), v);
    assert_eq!(dst.cfg_readl(PCI_BASE_ADDRESS_0 as u8 + 5 * 4), ABAR as u32);

    // The engines came back on the source's buffers, and the identify data with them.
    *dst.ports.lock().unwrap() = *src.ports.lock().unwrap();
    *dst.next.lock().unwrap() = *src.next.lock().unwrap();
    assert_eq!(dst.px(0, PX_CMD) & (CMD_CR | CMD_FR), CMD_CR | CMD_FR);
    assert_eq!(dst.disk_read(0, CMD_READ_DMA, 3, 1), data);
    dst.expect_done(0, IS_DHRS);
    let buf = dst.alloc(512, 2);
    dst.issue(0, 0, &Cmd::ata(CMD_IDENTIFY, 0, 0), buf, 512);
    dst.expect_done(0, IS_DHRS | IS_PSS);
    assert_eq!(ata_string(&dst.memread(buf, 512), 10, 10), "testdisk");

    // Only an idle controller loads.
    refused(&dst, &v, |s| s.ahci.dev[1].busy_slot = 0);
    refused(&dst, &v, |s| s.ahci.dev[0].ncq_tfs[4].used = true);
    refused(&dst, &v, |s| {
        s.ahci.dev[0].ncq_tfs[4].used = true;
        s.ahci.dev[0].ncq_tfs[4].halt = true;
    });
    refused(&dst, &v, |s| s.ahci.dev[0].ifs0.status |= 0x08);
    refused(&dst, &v, |s| s.ahci.dev[2].port.error_status = 1);
    refused(&dst, &v, |s| s.ahci.dev[0].cmd &= !CMD_ST);
    refused(&dst, &v, |s| s.ahci.ports = 4);
}

fn refused(env: &Env, v: &Ich9AhciVmState, f: impl FnOnce(&mut Ich9AhciVmState)) {
    let mut bad = v.clone();
    f(&mut bad);
    assert!(env.ahci.vmstate_load(&bad).is_err());
}
