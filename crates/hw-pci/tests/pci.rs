// SPDX-License-Identifier: GPL-2.0-or-later

//! Tests of the PCI core, driven the way QEMU's pci-test and the libqos PCI helpers drive a
//! machine: config cycles through 0xcf8/0xcfc, BAR sizing by writing all ones, and MMIO or port
//! accesses to the mapped BARs.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_hw_pci::regs::*;
use ruvm_hw_pci::*;
use ruvm_mem::{
    AccessCtx, AccessSize, AddressSpace, Endian, MemResult, MemTxAttrs, MemorySystem, MmioOps,
    RegionId,
};

const ATTRS: MemTxAttrs = MemTxAttrs::UNSPECIFIED;

/// A BAR whose reads return `tag | offset` and whose writes are recorded.
#[derive(Debug)]
struct Marker {
    tag: u64,
    writes: Mutex<Vec<(u64, u64)>>,
}

impl MmioOps for Marker {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(self.tag | offset)
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.writes.lock().unwrap().push((offset, value));
        Ok(())
    }
}

type Messages = Arc<Mutex<Vec<(u64, u32)>>>;

struct Env {
    mem: Arc<MemorySystem>,
    sysmem: RegionId,
    mem_as: Arc<AddressSpace>,
    io_as: Arc<AddressSpace>,
    bus: Arc<PciBus>,
    host: Arc<PciHostState>,
    /// The level last driven on each of the four PIRQs.
    levels: Arc<[AtomicI32; 4]>,
    msgs: Messages,
}

impl Env {
    fn new() -> Env {
        let mem = Arc::new(MemorySystem::new());
        let sysmem = mem.new_container("system", 1 << 64).unwrap();
        let io = mem.new_container("io", 1 << 16).unwrap();
        let mem_as = mem.address_space_init(sysmem, "memory").unwrap();
        let io_as = mem.address_space_init(io, "I/O").unwrap();
        let bus = PciBus::new_root("pci.0", Arc::clone(&mem), sysmem, io, 0);
        let host = PciHostState::new(Arc::clone(&bus));
        host.map_ioports(&mem, io).unwrap();

        let levels: Arc<[AtomicI32; 4]> = Arc::new(Default::default());
        let l = Arc::clone(&levels);
        bus.set_irqs(Arc::new(move |irq, level| l[irq as usize].store(level, Ordering::SeqCst)), 4);
        // The PIIX style routing: slot + pin, modulo 4.
        bus.set_map_irq(Arc::new(pci_swizzle_map_irq_fn));

        let msgs: Messages = Arc::default();
        let m = Arc::clone(&msgs);
        bus.set_msi_handler(Some(Arc::new(move |a, d| m.lock().unwrap().push((a, d)))));
        Env { mem, sysmem, mem_as, io_as, bus, host, levels, msgs }
    }

    fn outl(&self, port: u64, v: u32) {
        assert!(self.io_as.store(port, 4, v.into(), Endian::Little, ATTRS).is_ok());
    }

    fn outw(&self, port: u64, v: u16) {
        assert!(self.io_as.store(port, 2, v.into(), Endian::Little, ATTRS).is_ok());
    }

    fn outb(&self, port: u64, v: u8) {
        assert!(self.io_as.store(port, 1, v.into(), Endian::Little, ATTRS).is_ok());
    }

    fn inl(&self, port: u64) -> u32 {
        self.io_as.load(port, 4, Endian::Little, ATTRS).0 as u32
    }

    fn inw(&self, port: u64) -> u16 {
        self.io_as.load(port, 2, Endian::Little, ATTRS).0 as u16
    }

    fn inb(&self, port: u64) -> u8 {
        self.io_as.load(port, 1, Endian::Little, ATTRS).0 as u8
    }

    fn cfg_addr(bus: u8, devfn: u8, off: u8) -> u32 {
        (1 << 31) | (u32::from(bus) << 16) | (u32::from(devfn) << 8) | u32::from(off)
    }

    /// `qpci_config_readl()` as the PC libqos backend does it.
    fn readl(&self, devfn: u8, off: u8) -> u32 {
        self.readl_bus(0, devfn, off)
    }

    fn readl_bus(&self, bus: u8, devfn: u8, off: u8) -> u32 {
        self.outl(0xcf8, Self::cfg_addr(bus, devfn, off));
        self.inl(0xcfc)
    }

    fn readw(&self, devfn: u8, off: u8) -> u16 {
        self.outl(0xcf8, Self::cfg_addr(0, devfn, off));
        self.inw(0xcfc)
    }

    fn readb(&self, devfn: u8, off: u8) -> u8 {
        self.outl(0xcf8, Self::cfg_addr(0, devfn, off));
        self.inb(0xcfc)
    }

    fn writel(&self, devfn: u8, off: u8, v: u32) {
        self.writel_bus(0, devfn, off, v);
    }

    fn writel_bus(&self, bus: u8, devfn: u8, off: u8, v: u32) {
        self.outl(0xcf8, Self::cfg_addr(bus, devfn, off));
        self.outl(0xcfc, v);
    }

    fn writew(&self, devfn: u8, off: u8, v: u16) {
        self.outl(0xcf8, Self::cfg_addr(0, devfn, off));
        self.outw(0xcfc, v);
    }

    fn writeb(&self, devfn: u8, off: u8, v: u8) {
        self.outl(0xcf8, Self::cfg_addr(0, devfn, off));
        self.outb(0xcfc, v);
    }

    fn mem_readl(&self, addr: u64) -> Option<u32> {
        let (v, r) = self.mem_as.load(addr, 4, Endian::Little, ATTRS);
        r.is_ok().then_some(v as u32)
    }

    fn mem_writel(&self, addr: u64, v: u32) {
        assert!(self.mem_as.store(addr, 4, v.into(), Endian::Little, ATTRS).is_ok());
    }

    fn bar(&self, name: &str, size: u128, tag: u64) -> (RegionId, Arc<Marker>) {
        let m = Arc::new(Marker { tag, writes: Mutex::default() });
        let ops: Arc<dyn MmioOps> = m.clone();
        (self.mem.new_io(name, size, ops).unwrap(), m)
    }

    fn device(&self, name: &str, devfn: Option<u8>) -> Arc<PciDevice> {
        self.bus.register_device(&info(name), devfn).unwrap()
    }

    /// `qpci_iomap()`: sizes BAR `n` by writing all ones and maps it at `addr`.
    fn iomap(&self, devfn: u8, n: u8, addr: u64) -> u64 {
        let off = PCI_BASE_ADDRESS_0 as u8 + n * 4;
        self.writel(devfn, off, 0xffff_ffff);
        let raw = self.readl(devfn, off);
        assert_ne!(raw, 0, "BAR {n} is not implemented");
        let io = raw & u32::from(PCI_BASE_ADDRESS_SPACE_IO) != 0;
        let size = if io {
            1u64 << (raw & !3).trailing_zeros()
        } else {
            let is64 = raw & u32::from(PCI_BASE_ADDRESS_MEM_TYPE_64) != 0;
            if is64 {
                self.writel(devfn, off + 4, 0xffff_ffff);
                let hi = self.readl(devfn, off + 4);
                let full = (u64::from(hi) << 32) | u64::from(raw & !0xf);
                1u64 << full.trailing_zeros()
            } else {
                1u64 << (raw & !0xf).trailing_zeros()
            }
        };
        self.writel(devfn, off, addr as u32);
        if !io && raw & u32::from(PCI_BASE_ADDRESS_MEM_TYPE_64) != 0 {
            self.writel(devfn, off + 4, (addr >> 32) as u32);
        }
        size
    }
}

fn info(name: &str) -> PciDeviceInfo {
    PciDeviceInfo {
        name: name.to_string(),
        vendor_id: 0x8086,
        device_id: 0x1234,
        revision: 3,
        class_id: 0x0200,
        ..Default::default()
    }
}

#[test]
fn header_through_cf8_cfc() {
    let env = Env::new();
    env.device("nic", Some(pci_devfn(3, 0)));
    let devfn = pci_devfn(3, 0);
    assert_eq!(env.readl(devfn, 0), 0x1234_8086);
    assert_eq!(env.readw(devfn, PCI_DEVICE_ID as u8), 0x1234);
    assert_eq!(env.readb(devfn, PCI_REVISION_ID as u8), 3);
    assert_eq!(env.readw(devfn, PCI_CLASS_DEVICE as u8), 0x0200);
    // QEMU's default subsystem IDs.
    assert_eq!(env.readw(devfn, PCI_SUBSYSTEM_VENDOR_ID as u8), 0x1af4);
    assert_eq!(env.readw(devfn, PCI_SUBSYSTEM_ID as u8), 0x1100);

    // Nothing at slot 4, or on bus 1.
    assert_eq!(env.readl(pci_devfn(4, 0), 0), 0xffff_ffff);
    assert_eq!(env.readw(pci_devfn(4, 0), 0), 0xffff);
    assert_eq!(env.readl_bus(1, devfn, 0), 0xffff_ffff);

    // CONFIG_ADDRESS reads back, and the enable bit gates CONFIG_DATA.
    env.outl(0xcf8, Env::cfg_addr(0, devfn, 0));
    assert_eq!(env.inl(0xcf8), 0x8000_1800);
    env.outl(0xcf8, 0x0000_1800);
    assert_eq!(env.inl(0xcfc), 0xffff_ffff);
    env.outl(0xcfc, 0);
    assert_eq!(env.host.config_reg(), 0x1800);

    // Only 32 bit writes at 0xcf8 update CONFIG_ADDRESS. Port 0xcf9 is the PIIX reset control
    // register on a real PC, so byte writes must not clobber it.
    env.outb(0xcf8, 0x55);
    env.outw(0xcfa, 0x55);
    assert_eq!(env.host.config_reg(), 0x1800);
}

#[test]
fn sub_dword_data_accesses() {
    let env = Env::new();
    let dev = env.device("dev", Some(pci_devfn(1, 0)));
    let devfn = dev.devfn();

    // CONFIG_ADDRESS holds a dword offset; the data port offset picks the byte lanes.
    env.outl(0xcf8, Env::cfg_addr(0, devfn, 0));
    assert_eq!(env.inb(0xcfc), 0x86);
    assert_eq!(env.inb(0xcfd), 0x80);
    assert_eq!(env.inb(0xcfe), 0x34);
    assert_eq!(env.inb(0xcff), 0x12);
    assert_eq!(env.inw(0xcfc), 0x8086);
    assert_eq!(env.inw(0xcfe), 0x1234);

    // Offset low bits in CONFIG_ADDRESS work too, which libqos relies on.
    env.outl(0xcf8, Env::cfg_addr(0, devfn, 2));
    assert_eq!(env.inw(0xcfc), 0x1234);

    // Byte and word writes land on the right bytes.
    env.outl(0xcf8, Env::cfg_addr(0, devfn, PCI_CACHE_LINE_SIZE as u8));
    env.outb(0xcfc, 0x10);
    assert_eq!(dev.config_bytes()[PCI_CACHE_LINE_SIZE], 0x10);
    env.outl(0xcf8, Env::cfg_addr(0, devfn, 0x40));
    env.outw(0xcfe, 0xbeef);
    env.outb(0xcfd, 0x7a);
    assert_eq!(env.readl(devfn, 0x40), 0xbeef_7a00);

    // A read that would run off the end of config space is clipped.
    assert_eq!(pci_host_config_read_common(&dev, 0xfe, 0x100, 4), 0);
    assert_eq!(pci_host_config_read_common(&dev, 0x100, 0x100, 4), 0xffff_ffff);
    assert_eq!(pci_host_config_read_common(&dev, 0x40, 0x100, 8), 0xffff_ffff);
}

#[test]
fn config_space_masks() {
    let env = Env::new();
    let dev = env.device("dev", Some(pci_devfn(2, 0)));
    let devfn = dev.devfn();

    // Read-only IDs.
    env.writel(devfn, 0, 0);
    assert_eq!(env.readl(devfn, 0), 0x1234_8086);
    env.writel(devfn, PCI_CLASS_REVISION as u8, 0);
    assert_eq!(env.readb(devfn, PCI_REVISION_ID as u8), 3);

    // Only IO, MEMORY, MASTER, SERR and INTX_DISABLE are writable in the command register.
    env.writew(devfn, PCI_COMMAND as u8, 0xffff);
    let want = PCI_COMMAND_IO
        | PCI_COMMAND_MEMORY
        | PCI_COMMAND_MASTER
        | PCI_COMMAND_SERR
        | PCI_COMMAND_INTX_DISABLE;
    assert_eq!(env.readw(devfn, PCI_COMMAND as u8), want);
    assert_eq!(pci_get_word(&dev.wmask_bytes(), PCI_COMMAND), want);

    // Status error bits are write one to clear, the rest read-only.
    dev.with_config(|c| {
        pci_set_word(c.config, PCI_STATUS, PCI_STATUS_REC_MASTER_ABORT | PCI_STATUS_PARITY | 0x10);
    });
    env.writew(devfn, PCI_STATUS as u8, PCI_STATUS_PARITY);
    assert_eq!(env.readw(devfn, PCI_STATUS as u8), PCI_STATUS_REC_MASTER_ABORT | 0x10);
    env.writew(devfn, PCI_STATUS as u8, 0xffff);
    assert_eq!(env.readw(devfn, PCI_STATUS as u8), 0x10);

    // Cache line size and interrupt line are writable; the pin is not.
    env.writeb(devfn, PCI_INTERRUPT_LINE as u8, 11);
    env.writeb(devfn, PCI_INTERRUPT_PIN as u8, 1);
    assert_eq!(env.readb(devfn, PCI_INTERRUPT_LINE as u8), 11);
    assert_eq!(env.readb(devfn, PCI_INTERRUPT_PIN as u8), 0);

    // Device specific space is writable until a capability claims it.
    env.writel(devfn, 0x80, 0xdead_beef);
    assert_eq!(env.readl(devfn, 0x80), 0xdead_beef);

    // Unimplemented BARs read as zero after sizing.
    env.writel(devfn, PCI_BASE_ADDRESS_0 as u8, 0xffff_ffff);
    assert_eq!(env.readl(devfn, PCI_BASE_ADDRESS_0 as u8), 0);

    // cmask covers the IDs and the header type.
    let cmask = dev.cmask_bytes();
    assert_eq!(pci_get_word(&cmask, PCI_VENDOR_ID), 0xffff);
    assert_eq!(cmask[PCI_HEADER_TYPE], 0xff);
    assert_eq!(cmask[PCI_STATUS], PCI_STATUS_CAP_LIST as u8);
    assert_eq!(cmask[PCI_COMMAND], 0);

    // A model's wmask edits stick.
    dev.with_config(|c| c.wmask[0x80] = 0x0f);
    env.writeb(devfn, 0x80, 0xff);
    assert_eq!(env.readb(devfn, 0x80), 0xef);
    let w1c = dev.w1cmask_bytes();
    assert_eq!(pci_get_word(&w1c, PCI_COMMAND), 0);
}

#[test]
fn bar_sizing_and_mapping() {
    let env = Env::new();
    let dev = env.device("dev", Some(pci_devfn(4, 0)));
    let devfn = dev.devfn();
    let (mmio, mmio_m) = env.bar("bar0", 0x1000, 0xa000_0000);
    let (io, _) = env.bar("bar1", 0x20, 0xb000);
    let (pref, _) = env.bar("bar2", 0x10_0000, 0xc000_0000);
    dev.register_bar(0, PCI_BASE_ADDRESS_SPACE_MEMORY, mmio);
    dev.register_bar(1, PCI_BASE_ADDRESS_SPACE_IO, io);
    dev.register_bar(2, PCI_BASE_ADDRESS_SPACE_MEMORY | PCI_BASE_ADDRESS_MEM_PREFETCH, pref);

    // Type bits read back before sizing.
    assert_eq!(env.readl(devfn, PCI_BASE_ADDRESS_1 as u8), 1);
    assert_eq!(env.readl(devfn, PCI_BASE_ADDRESS_2 as u8), 8);

    assert_eq!(env.iomap(devfn, 0, 0xfebf_0000), 0x1000);
    assert_eq!(env.iomap(devfn, 1, 0xc040), 0x20);
    assert_eq!(env.iomap(devfn, 2, 0xfe00_0000), 0x10_0000);
    assert_eq!(env.readl(devfn, PCI_BASE_ADDRESS_0 as u8), 0xfebf_0000);
    assert_eq!(env.readl(devfn, PCI_BASE_ADDRESS_1 as u8), 0xc041);
    assert_eq!(env.readl(devfn, PCI_BASE_ADDRESS_2 as u8), 0xfe00_0008);

    // Nothing is decoded until the command register allows it.
    assert_eq!(dev.bar_addr(0), PCI_BAR_UNMAPPED);
    assert_ne!(env.mem_readl(0xfebf_0010), Some(0xa000_0010));
    env.writew(devfn, PCI_COMMAND as u8, PCI_COMMAND_MEMORY);
    assert_eq!(dev.bar_addr(0), 0xfebf_0000);
    assert_eq!(dev.bar_addr(1), PCI_BAR_UNMAPPED);
    assert_eq!(env.mem_readl(0xfebf_0010), Some(0xa000_0010));
    assert_eq!(env.mem_readl(0xfe00_0100), Some(0xc000_0100));
    env.mem_writel(0xfebf_0004, 0x55);
    assert_eq!(*mmio_m.writes.lock().unwrap(), vec![(4, 0x55)]);

    env.writew(devfn, PCI_COMMAND as u8, PCI_COMMAND_MEMORY | PCI_COMMAND_IO);
    assert_eq!(dev.bar_addr(1), 0xc040);
    assert_eq!(env.inl(0xc044), 0xb004);

    // Moving a BAR moves the mapping.
    env.writel(devfn, PCI_BASE_ADDRESS_0 as u8, 0xfeb0_0000);
    assert_eq!(dev.bar_addr(0), 0xfeb0_0000);
    assert_eq!(env.mem_readl(0xfeb0_0000), Some(0xa000_0000));
    assert_ne!(env.mem_readl(0xfebf_0000), Some(0xa000_0000));

    // Clearing MEMORY unmaps the memory BARs but not the I/O BAR.
    env.writew(devfn, PCI_COMMAND as u8, PCI_COMMAND_IO);
    assert_eq!(dev.bar_addr(0), PCI_BAR_UNMAPPED);
    assert_eq!(dev.bar_addr(2), PCI_BAR_UNMAPPED);
    assert_ne!(env.mem_readl(0xfeb0_0000), Some(0xa000_0000));
    assert_eq!(env.inl(0xc044), 0xb004);

    let info = dev.bar_info(1).unwrap();
    assert_eq!((info.size, info.type_, info.memory), (0x20, 1, io));
    assert!(dev.bar_info(3).is_none());

    // Reset clears the command register and the BAR addresses but keeps the type bits.
    dev.reset();
    assert_eq!(env.readw(devfn, PCI_COMMAND as u8), 0);
    assert_eq!(env.readl(devfn, PCI_BASE_ADDRESS_1 as u8), 1);
    assert_eq!(env.readl(devfn, PCI_BASE_ADDRESS_2 as u8), 8);
    assert_eq!(dev.bar_addr(1), PCI_BAR_UNMAPPED);
}

#[test]
fn bar_address_rules() {
    let env = Env::new();
    let dev = env.device("dev", Some(pci_devfn(5, 0)));
    let devfn = dev.devfn();
    let (m32, _) = env.bar("m32", 0x1000, 0xa000_0000);
    let (io, _) = env.bar("io", 0x100, 0);
    let (rom, _) = env.bar("rom", 0x1_0000, 0xd000_0000);
    dev.register_bar(0, PCI_BASE_ADDRESS_SPACE_MEMORY, m32);
    dev.register_bar(1, PCI_BASE_ADDRESS_SPACE_IO, io);
    dev.register_bar(PCI_ROM_SLOT, PCI_BASE_ADDRESS_SPACE_MEMORY, rom);
    env.writew(devfn, PCI_COMMAND as u8, PCI_COMMAND_MEMORY | PCI_COMMAND_IO);

    // Address 0 means unmapped unless the machine allows it.
    assert_eq!(dev.bar_addr(0), PCI_BAR_UNMAPPED);
    assert_eq!(dev.bar_address(0), PCI_BAR_UNMAPPED);
    // A 32 bit BAR may not touch the last page below 4 GiB, where it would wrap.
    env.writel(devfn, PCI_BASE_ADDRESS_0 as u8, 0xffff_f000);
    assert_eq!(dev.bar_addr(0), PCI_BAR_UNMAPPED);
    env.writel(devfn, PCI_BASE_ADDRESS_0 as u8, 0xffff_e000);
    assert_eq!(dev.bar_addr(0), 0xffff_e000);

    // I/O BARs the same.
    env.writel(devfn, PCI_BASE_ADDRESS_1 as u8, 0xffff_ff00);
    assert_eq!(dev.bar_addr(1), PCI_BAR_UNMAPPED);
    env.writel(devfn, PCI_BASE_ADDRESS_1 as u8, 0x1000);
    assert_eq!(dev.bar_addr(1), 0x1000);

    // The ROM BAR needs its own enable bit.
    let rom_off = PCI_ROM_ADDRESS as u8;
    env.writel(devfn, rom_off, 0xffff_ffff);
    assert_eq!(env.readl(devfn, rom_off), 0xffff_0001);
    assert_eq!(dev.bar_addr(PCI_ROM_SLOT), PCI_BAR_UNMAPPED);
    env.writel(devfn, rom_off, 0xfeb0_0000);
    assert_eq!(dev.bar_addr(PCI_ROM_SLOT), PCI_BAR_UNMAPPED);
    env.writel(devfn, rom_off, 0xfeb0_0001);
    assert_eq!(dev.bar_addr(PCI_ROM_SLOT), 0xfeb0_0000);
    assert_eq!(env.mem_readl(0xfeb0_0010), Some(0xd000_0010));

    // A disabled function decodes nothing and ignores config accesses.
    dev.set_enabled(false);
    assert_eq!(dev.bar_addr(0), PCI_BAR_UNMAPPED);
    assert_eq!(env.readl(devfn, 0), 0xffff_ffff);
    dev.set_enabled(true);
    assert_eq!(env.readl(devfn, 0), 0x1234_8086);

    // With allow_0_address a device registered afterwards may sit at 0.
    env.bus.set_allow_0_address(true);
    let d2 = env.device("d2", Some(pci_devfn(6, 0)));
    let (b, _) = env.bar("b", 0x1000, 0);
    d2.register_bar(0, PCI_BASE_ADDRESS_SPACE_MEMORY, b);
    env.writew(d2.devfn(), PCI_COMMAND as u8, PCI_COMMAND_MEMORY);
    assert_eq!(d2.bar_addr(0), 0);
}

#[test]
fn bar_64bit() {
    let env = Env::new();
    let dev = env.device("dev", Some(pci_devfn(6, 0)));
    let devfn = dev.devfn();
    let (r, _) = env.bar("bar64", 0x4000_0000, 0x1000_0000_0000);
    let ty = PCI_BASE_ADDRESS_SPACE_MEMORY
        | PCI_BASE_ADDRESS_MEM_TYPE_64
        | PCI_BASE_ADDRESS_MEM_PREFETCH;
    dev.register_bar(2, ty, r);
    assert_eq!(env.readl(devfn, PCI_BASE_ADDRESS_2 as u8), 0xc);
    assert_eq!(pci_get_quad(&dev.wmask_bytes(), PCI_BASE_ADDRESS_2), !0x3fff_ffff);
    assert_eq!(pci_get_quad(&dev.cmask_bytes(), PCI_BASE_ADDRESS_2), !0);

    assert_eq!(env.iomap(devfn, 2, 0x80_0000_0000), 0x4000_0000);
    assert_eq!(env.readl(devfn, PCI_BASE_ADDRESS_2 as u8), 0xc);
    assert_eq!(env.readl(devfn, PCI_BASE_ADDRESS_3 as u8), 0x80);
    env.writew(devfn, PCI_COMMAND as u8, PCI_COMMAND_MEMORY);
    assert_eq!(dev.bar_addr(2), 0x80_0000_0000);
    assert_eq!(env.mem_readl(0x80_0000_1000), Some(0x1000));

    // A 64 bit BAR may sit across 4 GiB.
    env.writel(devfn, PCI_BASE_ADDRESS_3 as u8, 0);
    env.writel(devfn, PCI_BASE_ADDRESS_2 as u8, 0xc000_0000);
    env.writel(devfn, PCI_BASE_ADDRESS_3 as u8, 0);
    assert_eq!(dev.bar_addr(2), 0xc000_0000);
    // But not wrap past the top.
    env.writel(devfn, PCI_BASE_ADDRESS_3 as u8, 0xffff_ffff);
    env.writel(devfn, PCI_BASE_ADDRESS_2 as u8, 0xc000_0000);
    assert_eq!(dev.bar_addr(2), PCI_BAR_UNMAPPED);
}

#[test]
fn capability_list() {
    let env = Env::new();
    let dev = env.device("dev", Some(pci_devfn(7, 0)));
    let devfn = dev.devfn();
    assert_eq!(env.readw(devfn, PCI_STATUS as u8) & PCI_STATUS_CAP_LIST, 0);
    assert_eq!(dev.find_capability(PCI_CAP_ID_VNDR), 0);

    let a = dev.add_capability(PCI_CAP_ID_VNDR, 0x50, 8).unwrap();
    let b = dev.add_capability(PCI_CAP_ID_MSI, 0x60, 10).unwrap();
    // Automatic placement takes the first free dword aligned hole.
    let c = dev.add_capability(PCI_CAP_ID_PM, 0, 8).unwrap();
    assert_eq!((a, b, c), (0x50, 0x60, 0x40));
    let d = dev.add_capability(PCI_CAP_ID_SSVID, 0, 8).unwrap();
    assert_eq!(d, 0x48);
    let e = dev.add_capability(PCI_CAP_ID_EXP, 0, 4).unwrap();
    assert_eq!(e, 0x58);
    assert_eq!(dev.add_capability(PCI_CAP_ID_SATA, 0, 8).unwrap(), 0x6c);

    // The list as the guest walks it, newest first.
    assert_ne!(env.readw(devfn, PCI_STATUS as u8) & PCI_STATUS_CAP_LIST, 0);
    let mut walk = Vec::new();
    let mut p = env.readb(devfn, PCI_CAPABILITY_LIST as u8);
    while p != 0 {
        walk.push((p, env.readb(devfn, p)));
        p = env.readb(devfn, p + 1);
    }
    assert_eq!(
        walk,
        vec![
            (0x6c, PCI_CAP_ID_SATA),
            (0x58, PCI_CAP_ID_EXP),
            (0x48, PCI_CAP_ID_SSVID),
            (0x40, PCI_CAP_ID_PM),
            (0x60, PCI_CAP_ID_MSI),
            (0x50, PCI_CAP_ID_VNDR),
        ]
    );
    assert_eq!(dev.find_capability(PCI_CAP_ID_MSI), 0x60);

    // Capabilities are read-only and checked on migration.
    env.writel(devfn, 0x50, 0xffff_ffff);
    assert_eq!(env.readb(devfn, 0x50), PCI_CAP_ID_VNDR);
    assert_eq!(dev.cmask_bytes()[0x52], 0xff);

    // Overlaps are refused with QEMU's message.
    let err = dev.add_capability(PCI_CAP_ID_VNDR, 0x64, 4).unwrap_err();
    assert_eq!(
        err.to_string(),
        "0000:00:00:07.0 Attempt to add PCI capability 9 at offset 64 overlaps existing \
         capability 60 at offset 64"
    );
    let err = dev.add_capability(PCI_CAP_ID_VNDR, 0x4c, 8).unwrap_err();
    assert!(err.to_string().ends_with("overlaps existing capability 48 at offset 4c"), "{err}");

    // Deleting unlinks and makes the space writable again.
    dev.del_capability(PCI_CAP_ID_MSI, 10);
    assert_eq!(dev.find_capability(PCI_CAP_ID_MSI), 0);
    assert_eq!(env.readb(devfn, 0x51), 0);
    env.writeb(devfn, 0x60, 0x77);
    assert_eq!(env.readb(devfn, 0x60), 0x77);
    assert_eq!(dev.add_capability(PCI_CAP_ID_MSI, 0x60, 10).unwrap(), 0x60);

    for (id, size) in [
        (PCI_CAP_ID_MSI, 10),
        (PCI_CAP_ID_SATA, 8),
        (PCI_CAP_ID_EXP, 4),
        (PCI_CAP_ID_SSVID, 8),
        (PCI_CAP_ID_PM, 8),
        (PCI_CAP_ID_VNDR, 8),
    ] {
        dev.del_capability(id, size);
    }
    assert_eq!(env.readb(devfn, PCI_CAPABILITY_LIST as u8), 0);
    assert_eq!(env.readw(devfn, PCI_STATUS as u8) & PCI_STATUS_CAP_LIST, 0);
}

#[test]
fn devfn_allocation_and_conflicts() {
    let env = Env::new();
    let a = env.device("a", None);
    let b = env.device("b", None);
    assert_eq!((a.devfn(), b.devfn()), (0, 8));

    let err = env.bus.register_device(&info("c"), Some(8)).unwrap_err();
    assert_eq!(
        err.to_string(),
        "PCI: slot 1 function 0 not available for c, in use by b,id=(null)"
    );
    let named = PciDeviceInfo { id: Some("net0".into()), ..info("n") };
    env.bus.register_device(&named, Some(pci_devfn(9, 0))).unwrap();
    let err = env.bus.register_device(&info("c"), Some(pci_devfn(9, 0))).unwrap_err();
    assert_eq!(err.to_string(), "PCI: slot 9 function 0 not available for c, in use by n,id=net0");

    // Reserved slots are skipped by automatic placement and refused when asked for.
    env.bus.set_slot_reserved_mask(1 << 2);
    assert_eq!(env.bus.slot_reserved_mask(), 4);
    let err = env.bus.register_device(&info("c"), Some(pci_devfn(2, 3))).unwrap_err();
    assert_eq!(err.to_string(), "PCI: slot 2 function 3 not available for c, reserved");
    assert_eq!(env.device("c", None).devfn(), pci_devfn(3, 0));
    env.bus.clear_slot_reserved_mask(1 << 2);
    assert_eq!(env.device("d", None).devfn(), pci_devfn(2, 0));

    // Multifunction rules.
    let err = env.bus.register_device(&info("f1"), Some(pci_devfn(0, 1))).unwrap_err();
    assert_eq!(err.to_string(), "PCI: single function device can't be populated in function 0.1");
    let mf = PciDeviceInfo { multifunction: true, ..info("mf") };
    let f1 = env.bus.register_device(&info("f1"), Some(pci_devfn(10, 1))).unwrap();
    assert_eq!(f1.func(), 1);
    let err = env.bus.register_device(&info("f0"), Some(pci_devfn(10, 0))).unwrap_err();
    assert_eq!(
        err.to_string(),
        "PCI: a.0 indicates single function, but a.1 is already populated."
    );
    let f0 = env.bus.register_device(&mf, Some(pci_devfn(10, 0))).unwrap();
    assert_eq!(env.readb(f0.devfn(), PCI_HEADER_TYPE as u8), 0x80);
    env.bus.register_device(&info("f2"), Some(pci_devfn(10, 2))).unwrap();

    // Fill every slot, then run out.
    let env = Env::new();
    for slot in 0..32 {
        assert_eq!(env.device("x", None).slot(), slot);
    }
    let err = env.bus.register_device(&info("y"), None).unwrap_err();
    assert_eq!(err.to_string(), "PCI: no slot/function available for y, all in use or reserved");

    // Unregistering frees the slot.
    let dev = env.bus.device(pci_devfn(5, 0)).unwrap();
    env.bus.unregister_device(&dev);
    assert!(env.bus.device(pci_devfn(5, 0)).is_none());
    assert_eq!(env.device("z", None).slot(), 5);
    assert_eq!(env.bus.find_device(0, pci_devfn(5, 0)).unwrap().name(), "z");
}

#[test]
fn intx_swizzle_and_level_counting() {
    let env = Env::new();
    let lvl = |i: usize| env.levels[i].load(Ordering::SeqCst);
    let a = env.device("a", Some(pci_devfn(1, 0)));
    let b = env.device("b", Some(pci_devfn(4, 0)));
    let c = env.device("c", Some(pci_devfn(2, 0)));
    a.with_config(|c| c.config[PCI_INTERRUPT_PIN] = 1);
    b.with_config(|c| c.config[PCI_INTERRUPT_PIN] = 2);
    c.with_config(|c| c.config[PCI_INTERRUPT_PIN] = 4);

    // Slot 1 INTA goes to PIRQ 1; slot 4 INTB to PIRQ 1 as well; slot 2 INTD to PIRQ 1 too.
    a.set_irq(1);
    assert_eq!((lvl(1), env.bus.irq_count(1)), (1, 1));
    b.set_irq(1);
    assert_eq!((lvl(1), env.bus.irq_count(1)), (1, 2));
    a.set_irq(1);
    assert_eq!(env.bus.irq_count(1), 2, "re-asserting does not count twice");
    a.set_irq(0);
    assert_eq!((lvl(1), env.bus.irq_count(1)), (1, 1));
    b.set_irq(0);
    assert_eq!((lvl(1), env.bus.irq_count(1)), (0, 0));
    assert!(!env.bus.irq_level(1));

    // An allocated line drives the interrupt pin, and the status bit follows.
    let line = c.allocate_irq();
    line.raise();
    assert_eq!(lvl(1), 1);
    assert_ne!(env.readw(c.devfn(), PCI_STATUS as u8) & PCI_STATUS_INTERRUPT, 0);

    // INTX_DISABLE hides the pin without forgetting it.
    env.writew(c.devfn(), PCI_COMMAND as u8, PCI_COMMAND_INTX_DISABLE);
    assert_eq!((lvl(1), env.bus.irq_count(1)), (0, 0));
    assert!(c.irq_disabled());
    assert_ne!(env.readw(c.devfn(), PCI_STATUS as u8) & PCI_STATUS_INTERRUPT, 0);
    line.lower();
    line.raise();
    assert_eq!(env.bus.irq_count(1), 0);
    env.writew(c.devfn(), PCI_COMMAND as u8, 0);
    assert_eq!((lvl(1), env.bus.irq_count(1)), (1, 1));

    // Reset drops the pins.
    c.reset();
    assert_eq!((lvl(1), env.bus.irq_count(1)), (0, 0));
    assert_eq!(c.irq_state(), 0);

    // Raw pins on other slots.
    a.irq_handler(3, 1);
    assert_eq!(lvl(0), 1);
    a.irq_handler(3, 0);
    assert_eq!(lvl(0), 0);
    assert_eq!(pci_swizzle_map_irq_fn(pci_devfn(3, 0), 2), 1);
}

#[test]
fn msi_delivery_and_masking() {
    let env = Env::new();
    let dev = env.device("msi", Some(pci_devfn(3, 0)));
    let devfn = dev.devfn();

    let other = Env::new();
    other.bus.set_msi_handler(None);
    let d = other.device("x", None);
    assert_eq!(
        d.msi_init(0, 1, false, false).unwrap_err().to_string(),
        "MSI is not supported by interrupt controller"
    );

    let cap = dev.msi_init(0x50, 4, true, true).unwrap();
    assert_eq!(cap, 0x50);
    assert!(dev.msi_present());
    let c = cap;
    // 64 bit, per vector mask, 4 vectors offered: flags 0x0184.
    assert_eq!(env.readw(devfn, c + 2), 0x0184);
    // The address low bits are reserved, the mask only covers the offered vectors.
    env.writel(devfn, c + 4, 0xfee0_0003);
    env.writel(devfn, c + 8, 0x1);
    env.writew(devfn, c + 0xc, 0x4041);
    env.writel(devfn, c + 0x10, 0xffff_ffff);
    assert_eq!(env.readl(devfn, c + 4), 0xfee0_0000);
    assert_eq!(env.readl(devfn, c + 0x10), 0xf);
    env.writel(devfn, c + 0x10, 0);

    // Enable with 4 vectors (QSIZE 2) and bus mastering.
    let line = dev.allocate_irq_on_pin_a();
    line.raise();
    assert_eq!(env.levels[3].load(Ordering::SeqCst), 1);
    env.writew(devfn, PCI_COMMAND as u8, PCI_COMMAND_MASTER);
    env.writew(devfn, c + 2, PCI_MSI_FLAGS_ENABLE | (2 << 4));
    assert!(dev.msi_enabled());
    assert_eq!(dev.msi_nr_vectors_allocated(), 4);
    assert_eq!(env.levels[3].load(Ordering::SeqCst), 0, "enabling MSI drops INTx");

    dev.msi_notify(2);
    // The low bits of data carry the vector.
    assert_eq!(*env.msgs.lock().unwrap(), vec![(0x1_fee0_0000, 0x4042)]);
    assert_eq!(dev.msi_get_message(1), MsiMessage { address: 0x1_fee0_0000, data: 0x4041 });

    // A masked vector goes pending and fires on unmask.
    env.msgs.lock().unwrap().clear();
    dev.msi_set_mask(1, true).unwrap();
    assert!(dev.msi_is_masked(1));
    dev.msi_notify(1);
    assert!(env.msgs.lock().unwrap().is_empty());
    assert_eq!(env.readl(devfn, c + 0x14), 2);
    // The guest unmasks through config space; the pending bit is delivered on the write that
    // touches the capability.
    env.writel(devfn, c + 0x10, 0);
    env.writew(devfn, c + 2, PCI_MSI_FLAGS_ENABLE | (2 << 4));
    assert_eq!(*env.msgs.lock().unwrap(), vec![(0x1_fee0_0000, 0x4041)]);
    assert_eq!(env.readl(devfn, c + 0x14), 0);

    env.msgs.lock().unwrap().clear();
    dev.msi_set_mask(3, true).unwrap();
    dev.msi_notify(3);
    dev.msi_set_mask(3, false).unwrap();
    assert_eq!(*env.msgs.lock().unwrap(), vec![(0x1_fee0_0000, 0x4043)]);
    assert!(dev.msi_set_mask(32, true).is_err());

    // Without bus mastering the write never reaches the interrupt controller.
    env.msgs.lock().unwrap().clear();
    env.writew(devfn, PCI_COMMAND as u8, 0);
    dev.msi_notify(0);
    assert!(env.msgs.lock().unwrap().is_empty());
    env.writew(devfn, PCI_COMMAND as u8, PCI_COMMAND_MASTER);

    // Asking for more vectors than offered is clamped.
    env.writew(devfn, c + 2, PCI_MSI_FLAGS_ENABLE | (5 << 4));
    assert_eq!(env.readw(devfn, c + 2) & PCI_MSI_FLAGS_QSIZE, 2 << 4);

    // A per device trigger overrides the bus handler.
    let own: Messages = Arc::default();
    let o = Arc::clone(&own);
    dev.set_msi_trigger(Some(Arc::new(move |a, d| o.lock().unwrap().push((a, d)))));
    dev.msi_notify(0);
    assert_eq!(*own.lock().unwrap(), vec![(0x1_fee0_0000, 0x4040)]);

    // Reset turns MSI off and clears the message.
    dev.reset();
    assert!(!dev.msi_enabled());
    assert_eq!(env.readl(devfn, c + 4), 0);
    assert_eq!(env.readw(devfn, c + 2), 0x0184);
    dev.msi_uninit();
    assert!(!dev.msi_present());
    assert_eq!(dev.find_capability(PCI_CAP_ID_MSI), 0);
}

trait PinA {
    fn allocate_irq_on_pin_a(&self) -> ruvm_hw_core::IrqLine;
}

impl PinA for PciDevice {
    fn allocate_irq_on_pin_a(&self) -> ruvm_hw_core::IrqLine {
        self.with_config(|c| c.config[PCI_INTERRUPT_PIN] = 1);
        self.allocate_irq()
    }
}

#[test]
fn msix_table_pba_and_masking() {
    let env = Env::new();
    let dev = env.device("msix", Some(pci_devfn(5, 0)));
    let devfn = dev.devfn();
    assert_eq!(
        dev.msix_init_exclusive_bar(0, 1).unwrap_err().to_string(),
        "The number of MSI-X vectors is invalid"
    );
    let cap = dev.msix_init_exclusive_bar(4, 1).unwrap();
    assert_eq!(dev.msix_exclusive_bar(), Some(1));
    assert_eq!(env.readw(devfn, cap + 2), 3);
    assert_eq!(env.readl(devfn, cap + 4), 1, "table at offset 0 of BAR 1");
    assert_eq!(env.readl(devfn, cap + 8), 0x801, "PBA in the upper half of BAR 1");

    assert_eq!(env.iomap(devfn, 1, 0xfe00_0000), 0x1000);
    env.writew(devfn, PCI_COMMAND as u8, PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER);
    let table = 0xfe00_0000u64;
    let pba = 0xfe00_0800u64;

    // Every vector starts masked.
    for v in 0..4 {
        assert_eq!(env.mem_readl(table + v * 16 + 12), Some(1));
        dev.msix_vector_use(v as u32);
    }
    for v in 0..4u32 {
        let e = table + u64::from(v) * 16;
        env.mem_writel(e, 0xfee0_0000 + v * 0x1000);
        env.mem_writel(e + 4, 0);
        env.mem_writel(e + 8, 0x30 + v);
    }
    // 64 bit accesses are split into two 32 bit ones.
    let (lo, r) = env.mem_as.load(table + 16, 8, Endian::Little, ATTRS);
    assert!(r.is_ok());
    assert_eq!(lo, 0xfee0_1000);

    // Disabled MSI-X means everything is function masked: notifications go pending.
    dev.msix_notify(0);
    assert!(env.msgs.lock().unwrap().is_empty());
    assert!(dev.msix_is_pending(0));
    assert_eq!(env.mem_readl(pba), Some(1));
    // The PBA is read-only.
    env.mem_writel(pba, 0);
    assert_eq!(env.mem_readl(pba), Some(1));

    // Enable with the function mask set, then unmask vector 0 in the table: still masked.
    let ctrl = cap + 3;
    env.writeb(devfn, ctrl, 0xc0);
    assert!(dev.msix_enabled());
    env.mem_writel(table + 12, 0);
    assert!(env.msgs.lock().unwrap().is_empty());
    // Clearing the function mask delivers the pending vector.
    env.writeb(devfn, ctrl, 0x80);
    assert_eq!(*env.msgs.lock().unwrap(), vec![(0xfee0_0000, 0x30)]);
    assert!(!dev.msix_is_pending(0));

    env.msgs.lock().unwrap().clear();
    dev.msix_notify(0);
    dev.msix_notify(2);
    assert_eq!(*env.msgs.lock().unwrap(), vec![(0xfee0_0000, 0x30)]);
    assert!(dev.msix_is_pending(2));
    // Unmasking an entry through the table delivers it.
    env.mem_writel(table + 2 * 16 + 12, 0);
    assert_eq!(env.msgs.lock().unwrap()[1], (0xfee0_2000, 0x32));

    // Unused vectors never fire.
    env.msgs.lock().unwrap().clear();
    dev.msix_vector_unuse(2);
    dev.msix_notify(2);
    assert!(env.msgs.lock().unwrap().is_empty());

    // The model's own mask and message helpers.
    dev.msix_set_mask(0, true);
    assert!(dev.msix_is_masked(0));
    dev.msix_notify(0);
    assert!(dev.msix_is_pending(0));
    dev.msix_set_mask(0, false);
    assert_eq!(*env.msgs.lock().unwrap(), vec![(0xfee0_0000, 0x30)]);
    dev.msix_set_message(3, MsiMessage { address: 0xfee0_f000, data: 0x99 });
    assert_eq!(env.mem_readl(table + 3 * 16 + 8), Some(0x99));
    assert_eq!(env.mem_readl(table + 3 * 16 + 12), Some(0));
    assert_eq!(dev.msix_get_message(3).address, 0xfee0_f000);

    // Reset disables MSI-X and masks every vector.
    dev.reset();
    assert!(!dev.msix_enabled());
    assert_eq!(env.readb(devfn, ctrl), 0);
    assert!(dev.msix_is_masked(1));
    dev.msix_uninit();
    assert!(!dev.msix_present());
    assert_eq!(dev.msix_cap(), 0);
}

#[test]
fn msix_layout_checks() {
    let env = Env::new();
    let dev = env.device("msix", Some(pci_devfn(5, 0)));
    let bar = env.mem.new_container("bar", 0x1000).unwrap();
    let layout = MsixLayout {
        table_bar: bar,
        table_bar_nr: 0,
        table_offset: 0,
        pba_bar: bar,
        pba_bar_nr: 0,
        pba_offset: 0x40,
    };
    let want = "table & pba overlap, or they don't fit in BARs, or don't align";
    assert_eq!(dev.msix_init(8, layout, 0).unwrap_err().to_string(), want);
    let misaligned = MsixLayout { pba_offset: 0x804, ..layout };
    assert_eq!(dev.msix_init(8, misaligned, 0).unwrap_err().to_string(), want);
    let too_big = MsixLayout { pba_offset: 0xffc, ..layout };
    assert_eq!(dev.msix_init(8, too_big, 0).unwrap_err().to_string(), want);
    let ok = MsixLayout { pba_offset: 0x800, ..layout };
    assert_eq!(dev.msix_init(8, ok, 0x70).unwrap(), 0x70);
    assert_eq!(dev.msix_nr_vectors_allocated(), 8);
}

#[test]
fn bridge_windows_bus_numbers_and_swizzle() {
    let env = Env::new();
    let br_info = PciDeviceInfo {
        name: "pci-bridge".into(),
        vendor_id: 0x1b36,
        device_id: 0x0001,
        ..Default::default()
    };
    let br = PciBridge::new(&env.bus, &br_info, Some(pci_devfn(3, 0)), "pci.1").unwrap();
    let bdev = Arc::clone(br.device());
    let bdf = bdev.devfn();
    assert_eq!(env.readb(bdf, PCI_HEADER_TYPE as u8), PCI_HEADER_TYPE_BRIDGE);
    assert_eq!(env.readw(bdf, PCI_CLASS_DEVICE as u8), PCI_CLASS_BRIDGE_PCI);
    assert_eq!(env.readw(bdf, PCI_SEC_STATUS as u8), PCI_STATUS_66MHZ | PCI_STATUS_FAST_BACK);
    assert_eq!(env.readl(bdf, PCI_SUBSYSTEM_VENDOR_ID as u8), 0, "type 1 has no subsystem IDs");

    let sec = Arc::clone(br.sec_bus());
    assert!(!sec.is_root());
    let child = sec.register_device(&info("child"), Some(pci_devfn(1, 0))).unwrap();
    assert_eq!(child.bus_num(), 0, "secondary bus number not programmed yet");

    // Program bus numbers; the child becomes reachable as bus 1.
    env.writel(bdf, PCI_PRIMARY_BUS as u8, 0x0001_0100);
    assert_eq!(sec.bus_num(), 1);
    assert_eq!(env.readl_bus(1, child.devfn(), 0), 0x1234_8086);
    assert_eq!(env.readl_bus(2, child.devfn(), 0), 0xffff_ffff);
    assert_eq!(child.requester_id(), 0x0108);

    // A BAR behind the bridge is only visible through an open window.
    let (r, _) = env.bar("child-bar", 0x1000, 0x7700_0000);
    child.register_bar(0, PCI_BASE_ADDRESS_SPACE_MEMORY, r);
    env.writel_bus(1, child.devfn(), PCI_BASE_ADDRESS_0 as u8, 0xfe10_0000);
    env.writel_bus(1, child.devfn(), PCI_COMMAND as u8, u32::from(PCI_COMMAND_MEMORY));
    assert_eq!(child.bar_addr(0), 0xfe10_0000);
    assert_ne!(env.mem_readl(0xfe10_0000), Some(0x7700_0000));

    // Memory window 0xfe00_0000 to 0xfeff_ffff.
    env.writel(bdf, PCI_MEMORY_BASE as u8, 0xfef0_fe00);
    assert_eq!(br.mem_window(), PciBridgeWindow { base: 0xfe00_0000, size: 0 });
    env.writew(bdf, PCI_COMMAND as u8, PCI_COMMAND_MEMORY);
    assert_eq!(br.mem_window(), PciBridgeWindow { base: 0xfe00_0000, size: 0x100_0000 });
    assert_eq!(env.mem_readl(0xfe10_0008), Some(0x7700_0008));
    // Limit below base closes it.
    env.writel(bdf, PCI_MEMORY_BASE as u8, 0xfd00_fe00);
    assert_eq!(br.mem_window().size, 0);
    assert_ne!(env.mem_readl(0xfe10_0008), Some(0x7700_0008));

    // The I/O window has 4 KiB granularity and a 16 bit type.
    env.writew(bdf, PCI_IO_BASE as u8, 0x2010);
    env.writew(bdf, PCI_COMMAND as u8, PCI_COMMAND_IO);
    assert_eq!(br.io_window(), PciBridgeWindow { base: 0x1000, size: 0x2000 });
    // The prefetchable window is 64 bit capable.
    assert_eq!(env.readw(bdf, PCI_PREF_MEMORY_BASE as u8), 1);
    env.writel(bdf, PCI_PREF_MEMORY_BASE as u8, 0x0010_0001);
    env.writel(bdf, PCI_PREF_BASE_UPPER32 as u8, 0x1);
    env.writel(bdf, PCI_PREF_LIMIT_UPPER32 as u8, 0x1);
    env.writew(bdf, PCI_COMMAND as u8, PCI_COMMAND_MEMORY);
    assert_eq!(br.pref_mem_window(), PciBridgeWindow { base: 0x1_0000_0000, size: 0x20_0000 });
    assert_eq!(
        pci_bridge_get_limit(&bdev.config_bytes(), PCI_BASE_ADDRESS_MEM_PREFETCH),
        0x1_001f_ffff
    );

    // INTx from behind the bridge is swizzled at each level: slot 1 INTA becomes the bridge's
    // INTB, and the bridge in slot 3 sends that to PIRQ (3 + 1) % 4 = 0.
    let line = child.allocate_irq_on_pin_a();
    line.raise();
    assert_eq!(env.levels[0].load(Ordering::SeqCst), 1);
    assert_eq!(env.bus.irq_count(0), 1);
    line.lower();
    assert_eq!(env.levels[0].load(Ordering::SeqCst), 0);

    // Secondary bus reset resets the child.
    env.writel_bus(1, child.devfn(), PCI_COMMAND as u8, u32::from(PCI_COMMAND_MEMORY));
    line.raise();
    env.writew(bdf, PCI_BRIDGE_CONTROL as u8, PCI_BRIDGE_CTL_BUS_RESET);
    assert_eq!(child.config_bytes()[PCI_COMMAND], 0);
    assert_eq!(env.bus.irq_count(0), 0);
    env.writew(bdf, PCI_BRIDGE_CONTROL as u8, 0);
    assert_eq!(env.readl_bus(1, child.devfn(), 0), 0x1234_8086);

    // Bridge reset clears bus numbers and windows.
    bdev.reset();
    assert_eq!(sec.bus_num(), 0);
    assert_eq!(br.mem_window().size, 0);
    assert_eq!(env.readw(bdf, PCI_PREF_MEMORY_BASE as u8), 1);
    let _ = env.sysmem;
}

#[test]
fn bus_reset_and_lookup() {
    let env = Env::new();
    let a = env.device("a", Some(pci_devfn(1, 0)));
    env.writew(a.devfn(), PCI_COMMAND as u8, PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER);
    assert!(a.is_bus_master());
    env.bus.reset();
    assert_eq!(env.readw(a.devfn(), PCI_COMMAND as u8), 0);
    assert_eq!(env.bus.devices().len(), 1);
    assert!(Arc::ptr_eq(&env.bus.find_bus_nr(0).unwrap(), &env.bus));
    assert!(env.bus.find_bus_nr(3).is_none());
    assert_eq!(env.bus.root_bus_path(), "0000:00");
}

/// A model with its own config hooks, the way a device overrides `config_write`.
#[derive(Debug, Default)]
struct Hooked {
    seen: Mutex<Vec<(u32, u32, u32)>>,
}

impl PciDeviceOps for Hooked {
    fn config_read(&self, dev: &PciDevice, addr: u32, len: u32) -> u32 {
        if addr == 0x90 { 0x1234_5678 } else { dev.default_read_config(addr, len) }
    }

    fn config_write(&self, dev: &PciDevice, addr: u32, val: u32, len: u32) {
        self.seen.lock().unwrap().push((addr, val, len));
        dev.default_write_config(addr, val, len);
    }
}

#[test]
fn model_config_hooks() {
    let env = Env::new();
    let dev = env.device("hooked", Some(pci_devfn(2, 0)));
    let ops = Arc::new(Hooked::default());
    dev.set_ops(ops.clone());
    assert_eq!(env.readl(dev.devfn(), 0x90), 0x1234_5678);
    env.writeb(dev.devfn(), 0x41, 0x5a);
    assert_eq!(*ops.seen.lock().unwrap(), vec![(0x41, 0x5a, 1)]);
    assert_eq!(env.readb(dev.devfn(), 0x41), 0x5a);

    // Data port offsets reach the hook as config offsets.
    env.outl(0xcf8, Env::cfg_addr(0, dev.devfn(), 0x44));
    env.outw(0xcfe, 0x1111);
    assert_eq!(ops.seen.lock().unwrap()[1], (0x46, 0x1111, 2));
    assert_eq!(pci_data_read(&env.bus, 0x8000_1046, 2), 0x1111);
    pci_data_write(&env.bus, 0x1046, 0x2222, 2);
    assert_eq!(pci_data_read(&env.bus, 0x1046, 2), 0x2222);
}

#[test]
fn power_management_gates_bars() {
    let env = Env::new();
    let dev = env.device("pm", Some(pci_devfn(2, 0)));
    let devfn = dev.devfn();
    let pm = dev.pm_init(0x40).unwrap();
    // Allow D3 control by the guest: the state bits are the model's to make writable.
    dev.with_config(|c| {
        pci_set_word(c.wmask, usize::from(pm) + PCI_PM_CTRL, PCI_PM_CTRL_STATE_MASK);
    });
    let (r, _) = env.bar("bar", 0x1000, 0x1000_0000);
    dev.register_bar(0, PCI_BASE_ADDRESS_SPACE_MEMORY, r);
    env.writel(devfn, PCI_BASE_ADDRESS_0 as u8, 0xfe00_0000);
    env.writew(devfn, PCI_COMMAND as u8, PCI_COMMAND_MEMORY);
    assert_eq!(dev.bar_addr(0), 0xfe00_0000);

    // D1 and D2 are refused when not advertised; D3 unmaps the BARs.
    env.writew(devfn, pm + PCI_PM_CTRL as u8, 1);
    assert_eq!(env.readw(devfn, pm + PCI_PM_CTRL as u8), 0);
    env.writew(devfn, pm + PCI_PM_CTRL as u8, 3);
    assert_eq!(env.readw(devfn, pm + PCI_PM_CTRL as u8), 3);
    assert_eq!(dev.bar_addr(0), PCI_BAR_UNMAPPED);
    env.writew(devfn, pm + PCI_PM_CTRL as u8, 0);
    assert_eq!(dev.bar_addr(0), 0xfe00_0000);
}

#[test]
fn vmstate_round_trip() {
    let src = Env::new();
    let dev = src.device("dev", Some(pci_devfn(4, 0)));
    dev.with_config(|c| c.config[PCI_INTERRUPT_PIN] = 1);
    let (mmio, _) = src.bar("bar0", 0x1000, 0xa000_0000);
    dev.register_bar(0, PCI_BASE_ADDRESS_SPACE_MEMORY, mmio);
    src.writel(dev.devfn(), PCI_BASE_ADDRESS_0 as u8, 0xfebf_0000);
    src.writew(dev.devfn(), PCI_COMMAND as u8, PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER);
    dev.set_irq(1);
    let saved = dev.vmstate_save();
    assert_eq!(saved.version_id, 2);
    assert_eq!(saved.config.len(), PCI_CONFIG_SPACE_SIZE);
    assert_eq!(saved.irq_state, [1, 0, 0, 0]);
    let counts = src.bus.irq_counts();
    assert_eq!(counts, vec![1, 0, 0, 0]);

    let dst = Env::new();
    let d = dst.device("dev", Some(pci_devfn(4, 0)));
    d.with_config(|c| c.config[PCI_INTERRUPT_PIN] = 1);
    let (mmio, _) = dst.bar("bar0", 0x1000, 0xa000_0000);
    d.register_bar(0, PCI_BASE_ADDRESS_SPACE_MEMORY, mmio);
    d.vmstate_load(&saved).unwrap();
    dst.bus.set_irq_counts(&counts).unwrap();
    assert_eq!(d.vmstate_save(), saved);
    assert_eq!(d.bar_addr(0), 0xfebf_0000);
    assert_eq!(dst.mem_readl(0xfebf_0010), Some(0xa000_0010));
    assert!(d.is_bus_master());
    assert_eq!(d.irq_state(), 1);
    // Loading does not drive the line; lowering the pin afterwards balances the count.
    assert_eq!(dst.levels[0].load(Ordering::SeqCst), 0);
    d.set_irq(0);
    assert_eq!(dst.bus.irq_count(0), 0);
    assert!(dst.bus.set_irq_counts(&[0; 3]).is_err());

    // A read-only, checked bit that differs is refused, as is a bad pin level.
    let mut bad = saved.clone();
    bad.config[PCI_DEVICE_ID] ^= 1;
    assert!(d.vmstate_load(&bad).is_err());
    let mut bad = saved.clone();
    bad.irq_state[2] = 2;
    assert!(d.vmstate_load(&bad).is_err());
    let mut bad = saved;
    bad.config.truncate(64);
    assert!(d.vmstate_load(&bad).is_err());
}
