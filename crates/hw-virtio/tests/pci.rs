// SPDX-License-Identifier: GPL-2.0-or-later

//! The virtio-pci transport, driven the way libqos does it (`tests/qtest/libqos/virtio-pci.c`
//! and `virtio-pci-modern.c`): config space through the 0xcf8/0xcfc host bridge, BARs mapped by
//! config writes, and the virtio structures reached through MMIO and port I/O in a real
//! `ruvm-mem` address space with RAM at 0.

#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_hw_pci::regs::*;
use ruvm_hw_pci::{PciBridge, PciBus, PciDeviceInfo, PciHostState, pci_swizzle_map_irq_fn};
use ruvm_hw_virtio::blk::*;
use ruvm_hw_virtio::pci::*;
use ruvm_hw_virtio::virtio::*;
use ruvm_hw_virtio::{
    AddressSpaceMemory, EntropySource, MemBlockBackend, VirtioBackend, VirtioBlk, VirtioBlkConf,
    VirtioConsole, VirtioDeviceClass, VirtioPci, VirtioPciProps, VirtioPciVariant, VirtioRng,
    VirtioRngConf,
};
use ruvm_mem::{AddressSpace, Endian, MemTxAttrs, MemorySystem};

const ATTRS: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const RAM_SIZE: u64 = 4 << 20;
const DEVFN: u8 = 3 << 3;
const BAR4_ADDR: u64 = 0xe000_0000;
const BAR1_ADDR: u64 = 0xe100_0000;
const BAR0_PORT: u64 = 0xc000;
const BAR2_PORT: u64 = 0xc100;
const MSI_ADDR: u64 = 0xfee0_0000;
const DISK_SIZE: usize = 1 << 20;

type Messages = Arc<Mutex<Vec<(u64, u32)>>>;
type MakeDevice = Box<dyn Fn() -> Box<dyn VirtioDeviceClass>>;

/// Always gives 0xab.
#[derive(Debug)]
struct Fixed;

impl EntropySource for Fixed {
    fn fill(&mut self, buf: &mut [u8]) -> usize {
        buf.fill(0xab);
        buf.len()
    }
}

fn rng() -> Box<dyn VirtioDeviceClass> {
    Box::new(VirtioRng::new(Box::new(Fixed), VirtioRngConf::default()))
}

fn blk(disk: &MemBlockBackend) -> Box<dyn VirtioDeviceClass> {
    Box::new(VirtioBlk::new(Box::new(disk.clone()), VirtioBlkConf::default()))
}

fn console() -> Box<dyn VirtioDeviceClass> {
    Box::new(VirtioConsole::new(None))
}

/// A PC-like machine: RAM at 0, a root PCI bus with the host bridge at 0xcf8.
struct Machine {
    mem_as: Arc<AddressSpace>,
    io_as: Arc<AddressSpace>,
    bus: Arc<PciBus>,
    _host: Arc<PciHostState>,
    levels: Arc<[AtomicI32; 4]>,
    msgs: Messages,
    next_alloc: u64,
}

impl Machine {
    fn new(msi: bool) -> Machine {
        let mem = Arc::new(MemorySystem::new());
        let sysmem = mem.new_container("system", 1 << 64).unwrap();
        let ram = mem.new_ram("ram", RAM_SIZE).unwrap();
        mem.add_subregion(sysmem, 0, ram).unwrap();
        let io = mem.new_container("io", 1 << 16).unwrap();
        let mem_as = mem.address_space_init(sysmem, "memory").unwrap();
        let io_as = mem.address_space_init(io, "I/O").unwrap();
        let bus = PciBus::new_root("pci.0", Arc::clone(&mem), sysmem, io, 0);
        let host = PciHostState::new(Arc::clone(&bus));
        host.map_ioports(&mem, io).unwrap();

        let levels: Arc<[AtomicI32; 4]> = Arc::new(Default::default());
        let l = Arc::clone(&levels);
        bus.set_irqs(Arc::new(move |irq, level| l[irq as usize].store(level, Ordering::SeqCst)), 4);
        bus.set_map_irq(Arc::new(pci_swizzle_map_irq_fn));
        let msgs: Messages = Arc::default();
        if msi {
            let m = Arc::clone(&msgs);
            bus.set_msi_handler(Some(Arc::new(move |a, d| m.lock().unwrap().push((a, d)))));
        }
        Machine { mem_as, io_as, bus, _host: host, levels, msgs, next_alloc: 0x10000 }
    }

    fn backend(&self, class: Box<dyn VirtioDeviceClass>) -> VirtioBackend {
        let mem: SharedGuestMemory = Arc::new(AddressSpaceMemory::new(Arc::clone(&self.mem_as)));
        VirtioBackend::new(class, mem).unwrap()
    }

    fn plug(&self, class: Box<dyn VirtioDeviceClass>, props: &VirtioPciProps) -> VirtioPci {
        VirtioPci::new(&self.bus, Some(DEVFN), self.backend(class), props).unwrap()
    }

    // Config space, the way the PC libqos backend does it.

    fn cfg_read(&self, off: u8, size: u32) -> u32 {
        let addr = (1u32 << 31) | (u32::from(DEVFN) << 8) | u32::from(off & !3);
        self.out(0xcf8, 4, addr);
        self.inp(0xcfc + u64::from(off & 3), size)
    }

    fn cfg_write(&self, off: u8, size: u32, val: u32) {
        let addr = (1u32 << 31) | (u32::from(DEVFN) << 8) | u32::from(off & !3);
        self.out(0xcf8, 4, addr);
        self.out(0xcfc + u64::from(off & 3), size, val);
    }

    /// The size of a BAR, found by writing all ones.
    fn bar_size(&self, reg: u8) -> u64 {
        let old = self.cfg_read(reg, 4);
        self.cfg_write(reg, 4, u32::MAX);
        let v = self.cfg_read(reg, 4);
        self.cfg_write(reg, 4, old);
        if v == 0 {
            return 0;
        }
        let mask = if v & 1 != 0 { !3u32 } else { !0xfu32 };
        let low = v & mask;
        if v & 1 == 0 && v & 4 != 0 {
            let old_hi = self.cfg_read(reg + 4, 4);
            self.cfg_write(reg + 4, 4, u32::MAX);
            let hi = self.cfg_read(reg + 4, 4);
            self.cfg_write(reg + 4, 4, old_hi);
            let full = u64::from(hi) << 32 | u64::from(low);
            return (!full).wrapping_add(1);
        }
        u64::from((!low).wrapping_add(1))
    }

    /// Places the BARs and turns on I/O, memory and bus mastering, `qpci_device_enable()`.
    fn enable(&self) {
        self.cfg_write(0x10, 4, BAR0_PORT as u32);
        self.cfg_write(0x14, 4, BAR1_ADDR as u32);
        self.cfg_write(0x18, 4, BAR2_PORT as u32);
        self.cfg_write(0x20, 4, BAR4_ADDR as u32);
        self.cfg_write(0x24, 4, 0);
        let cmd = PCI_COMMAND_IO | PCI_COMMAND_MEMORY | PCI_COMMAND_MASTER;
        self.cfg_write(PCI_COMMAND as u8, 2, u32::from(cmd));
    }

    fn caps(&self) -> Vec<(u8, u8)> {
        let mut v = Vec::new();
        let mut pos = self.cfg_read(PCI_CAPABILITY_LIST as u8, 1) as u8;
        while pos != 0 {
            v.push((pos, self.cfg_read(pos, 1) as u8));
            pos = self.cfg_read(pos + 1, 1) as u8;
        }
        v
    }

    /// Finds the vendor capability of `cfg_type`, `qpci_find_capability()` plus the type
    /// check of `find_structure()`. The port I/O notify capability of `modern-pio-notify` lives in
    /// BAR2 and is newer in the list, so it is skipped here.
    fn virtio_cap(&self, cfg_type: u8) -> Option<VirtioCap> {
        self.caps()
            .into_iter()
            .filter(|&(_, id)| id == PCI_CAP_ID_VNDR)
            .map(|(pos, _)| VirtioCap {
                pos,
                len: self.cfg_read(pos + 2, 1) as u8,
                cfg_type: self.cfg_read(pos + 3, 1) as u8,
                bar: self.cfg_read(pos + 4, 1) as u8,
                offset: self.cfg_read(pos + 8, 4),
                length: self.cfg_read(pos + 12, 4),
                mult: self.cfg_read(pos + 16, 4),
            })
            .find(|c| c.cfg_type == cfg_type && c.bar != 2)
    }

    fn cap_pos(&self, id: u8) -> Option<u8> {
        self.caps().into_iter().find(|&(_, i)| i == id).map(|(p, _)| p)
    }

    // Bus accesses.

    fn out(&self, port: u64, size: u32, v: u32) {
        assert!(self.io_as.store(port, size, v.into(), Endian::Little, ATTRS).is_ok());
    }

    fn inp(&self, port: u64, size: u32) -> u32 {
        self.io_as.load(port, size, Endian::Little, ATTRS).0 as u32
    }

    fn mmio_write(&self, addr: u64, size: u32, v: u64) {
        assert!(self.mem_as.store(addr, size, v, Endian::Little, ATTRS).is_ok());
    }

    fn mmio_read(&self, addr: u64, size: u32) -> u64 {
        self.mem_as.load(addr, size, Endian::Little, ATTRS).0
    }

    fn intx(&self) -> bool {
        self.levels.iter().any(|l| l.load(Ordering::SeqCst) != 0)
    }

    fn take_msgs(&self) -> Vec<(u64, u32)> {
        std::mem::take(&mut *self.msgs.lock().unwrap())
    }

    // Guest memory.

    fn alloc(&mut self, size: u64, align: u64) -> u64 {
        let addr = self.next_alloc.div_ceil(align) * align;
        self.next_alloc = addr + size;
        assert!(self.next_alloc <= RAM_SIZE);
        let zeros = vec![0; size as usize];
        self.write_mem(addr, &zeros);
        addr
    }

    fn write_mem(&self, addr: u64, data: &[u8]) {
        assert!(self.mem_as.write(addr, ATTRS, data).is_ok());
    }

    fn read_mem(&self, addr: u64, len: usize) -> Vec<u8> {
        let mut v = vec![0; len];
        assert!(self.mem_as.read(addr, ATTRS, &mut v).is_ok());
        v
    }

    fn read_u16(&self, addr: u64) -> u16 {
        u16::from_le_bytes(self.read_mem(addr, 2).try_into().unwrap())
    }

    fn read_u32(&self, addr: u64) -> u32 {
        u32::from_le_bytes(self.read_mem(addr, 4).try_into().unwrap())
    }

    /// Enables MSI-X and points `vectors` table entries at `MSI_ADDR` with data 0x40 + n.
    fn enable_msix(&self, vectors: u32) {
        let cap = self.cap_pos(PCI_CAP_ID_MSIX).expect("MSI-X capability");
        for n in 0..vectors {
            let e = BAR1_ADDR + 16 * u64::from(n);
            self.mmio_write(e, 4, MSI_ADDR);
            self.mmio_write(e + 4, 4, 0);
            self.mmio_write(e + 8, 4, 0x40 + u64::from(n));
            self.mmio_write(e + 12, 4, 0);
        }
        let ctrl = self.cfg_read(cap + PCI_MSIX_FLAGS as u8, 2);
        self.cfg_write(cap + PCI_MSIX_FLAGS as u8, 2, ctrl | u32::from(PCI_MSIX_FLAGS_ENABLE));
    }
}

#[derive(Clone, Copy, Debug)]
struct VirtioCap {
    pos: u8,
    len: u8,
    cfg_type: u8,
    bar: u8,
    offset: u32,
    length: u32,
    mult: u32,
}

/// The driver side of a split ring.
#[derive(Debug)]
struct Ring {
    index: u16,
    size: u16,
    desc: u64,
    avail: u64,
    used: u64,
    avail_idx: u16,
    last_used: u16,
}

impl Ring {
    /// Puts `bufs` (address, length, device writable) in the table from slot 0 and makes the
    /// chain available.
    fn submit(&mut self, m: &Machine, bufs: &[(u64, u32, bool)]) -> u16 {
        let head = (self.avail_idx.wrapping_mul(4)) % self.size;
        for (i, &(addr, len, write)) in bufs.iter().enumerate() {
            let slot = (head + i as u16) % self.size;
            let mut flags = if write { 2u16 } else { 0 };
            if i + 1 < bufs.len() {
                flags |= 1;
            }
            let next = (slot + 1) % self.size;
            let mut d = Vec::with_capacity(16);
            d.extend_from_slice(&addr.to_le_bytes());
            d.extend_from_slice(&len.to_le_bytes());
            d.extend_from_slice(&flags.to_le_bytes());
            d.extend_from_slice(&next.to_le_bytes());
            m.write_mem(self.desc + 16 * u64::from(slot), &d);
        }
        let slot = self.avail_idx % self.size;
        m.write_mem(self.avail + 4 + 2 * u64::from(slot), &head.to_le_bytes());
        self.avail_idx = self.avail_idx.wrapping_add(1);
        m.write_mem(self.avail + 2, &self.avail_idx.to_le_bytes());
        head
    }

    fn get_used(&mut self, m: &Machine) -> Option<(u32, u32)> {
        if m.read_u16(self.used + 2) == self.last_used {
            return None;
        }
        let e = self.used + 4 + 8 * u64::from(self.last_used % self.size);
        self.last_used = self.last_used.wrapping_add(1);
        Some((m.read_u32(e), m.read_u32(e + 4)))
    }
}

/// A modern driver, `QVirtioPCIDevice` with the modern ops.
struct Modern {
    m: Machine,
    vp: VirtioPci,
    common: u64,
    isr: u64,
    device: u64,
    notify: u64,
    mult: u64,
}

impl Modern {
    fn new(m: Machine, vp: VirtioPci) -> Modern {
        m.enable();
        let addr = |t| {
            let c = m.virtio_cap(t).expect("virtio capability");
            assert_eq!(c.bar, 4);
            BAR4_ADDR + u64::from(c.offset)
        };
        let (common, isr, device, notify) = (
            addr(VIRTIO_PCI_CAP_COMMON_CFG),
            addr(VIRTIO_PCI_CAP_ISR_CFG),
            addr(VIRTIO_PCI_CAP_DEVICE_CFG),
            addr(VIRTIO_PCI_CAP_NOTIFY_CFG),
        );
        let mult = u64::from(m.virtio_cap(VIRTIO_PCI_CAP_NOTIFY_CFG).unwrap().mult);
        Modern { m, vp, common, isr, device, notify, mult }
    }

    fn cr(&self, off: u64, size: u32) -> u64 {
        self.m.mmio_read(self.common + off, size)
    }

    fn cw(&self, off: u64, size: u32, v: u64) {
        self.m.mmio_write(self.common + off, size, v);
    }

    fn status(&self) -> u8 {
        self.cr(VIRTIO_PCI_COMMON_STATUS, 1) as u8
    }

    fn set_status(&self, s: u8) {
        self.cw(VIRTIO_PCI_COMMON_STATUS, 1, u64::from(s));
    }

    fn device_features(&self) -> u64 {
        self.cw(VIRTIO_PCI_COMMON_DFSELECT, 4, 0);
        let lo = self.cr(VIRTIO_PCI_COMMON_DF, 4);
        self.cw(VIRTIO_PCI_COMMON_DFSELECT, 4, 1);
        let hi = self.cr(VIRTIO_PCI_COMMON_DF, 4);
        hi << 32 | lo
    }

    fn set_driver_features(&self, f: u64) {
        self.cw(VIRTIO_PCI_COMMON_GFSELECT, 4, 0);
        self.cw(VIRTIO_PCI_COMMON_GF, 4, f & 0xffff_ffff);
        self.cw(VIRTIO_PCI_COMMON_GFSELECT, 4, 1);
        self.cw(VIRTIO_PCI_COMMON_GF, 4, f >> 32);
    }

    fn negotiate(&self, wanted: u64) -> u64 {
        self.set_status(0);
        assert_eq!(self.status(), 0);
        self.set_status(VIRTIO_CONFIG_S_ACKNOWLEDGE);
        self.set_status(VIRTIO_CONFIG_S_ACKNOWLEDGE | VIRTIO_CONFIG_S_DRIVER);
        let f = self.device_features() & wanted;
        self.set_driver_features(f);
        self.set_status(self.status() | VIRTIO_CONFIG_S_FEATURES_OK);
        assert_ne!(self.status() & VIRTIO_CONFIG_S_FEATURES_OK, 0);
        f
    }

    fn driver_ok(&self) {
        self.set_status(self.status() | VIRTIO_CONFIG_S_DRIVER_OK);
    }

    /// `qvirtio_pci_virtqueue_setup_modern()`.
    fn setup_queue(&mut self, index: u16) -> Ring {
        self.cw(VIRTIO_PCI_COMMON_Q_SELECT, 2, u64::from(index));
        let size = self.cr(VIRTIO_PCI_COMMON_Q_SIZE, 2) as u16;
        assert_ne!(size, 0, "queue {index} does not exist");
        let n = u64::from(size);
        let desc = self.m.alloc(16 * n, 16);
        let avail = self.m.alloc(6 + 2 * n, 2);
        let used = self.m.alloc(6 + 8 * n, 4);
        for (off, v) in [
            (VIRTIO_PCI_COMMON_Q_DESCLO, desc),
            (VIRTIO_PCI_COMMON_Q_AVAILLO, avail),
            (VIRTIO_PCI_COMMON_Q_USEDLO, used),
        ] {
            self.cw(off, 4, v & 0xffff_ffff);
            self.cw(off + 4, 4, v >> 32);
        }
        self.cw(VIRTIO_PCI_COMMON_Q_ENABLE, 2, 1);
        assert_eq!(self.cr(VIRTIO_PCI_COMMON_Q_ENABLE, 2), 1);
        Ring { index, size, desc, avail, used, avail_idx: 0, last_used: 0 }
    }

    fn kick(&self, ring: &Ring) {
        self.cw(VIRTIO_PCI_COMMON_Q_SELECT, 2, u64::from(ring.index));
        let off = self.cr(VIRTIO_PCI_COMMON_Q_NOFF, 2);
        self.m.mmio_write(self.notify + off * self.mult, 2, u64::from(ring.index));
    }

    fn isr(&self) -> u8 {
        self.m.mmio_read(self.isr, 1) as u8
    }
}

fn blk_header(ty: u32, sector: u64) -> Vec<u8> {
    let mut h = Vec::with_capacity(16);
    h.extend_from_slice(&ty.to_le_bytes());
    h.extend_from_slice(&0u32.to_le_bytes());
    h.extend_from_slice(&sector.to_le_bytes());
    h
}

/// One virtio-blk request with header, optional data out, optional data in and status.
/// `kick` notifies the device. Returns (status, used length, data read).
fn blk_request(
    m: &mut Machine,
    q: &mut Ring,
    kick: &dyn Fn(&Machine, &Ring),
    ty: u32,
    sector: u64,
    out: &[u8],
    in_len: u32,
) -> (u8, u32, Vec<u8>) {
    let hdr = m.alloc(16, 16);
    m.write_mem(hdr, &blk_header(ty, sector));
    let mut bufs = vec![(hdr, 16, false)];
    if !out.is_empty() {
        let d = m.alloc(out.len() as u64, 16);
        m.write_mem(d, out);
        bufs.push((d, out.len() as u32, false));
    }
    let data = m.alloc(u64::from(in_len.max(1)), 16);
    if in_len > 0 {
        bufs.push((data, in_len, true));
    }
    let status = m.alloc(1, 1);
    m.write_mem(status, &[0xff]);
    bufs.push((status, 1, true));
    let head = q.submit(m, &bufs);
    kick(m, q);
    let (id, len) = q.get_used(m).expect("request completed");
    assert_eq!(id, u32::from(head));
    (m.read_mem(status, 1)[0], len, m.read_mem(data, in_len as usize))
}

fn sector(fill: &[u8]) -> Vec<u8> {
    let mut s = vec![0; 512];
    s[..fill.len()].copy_from_slice(fill);
    s
}

fn wanted() -> u64 {
    !feature(VIRTIO_RING_F_EVENT_IDX)
}

#[test]
fn ids_and_classes() {
    let cases: [(MakeDevice, u16, u16, u16); 3] = [
        (Box::new(rng), 4, PCI_DEVICE_ID_VIRTIO_RNG, PCI_CLASS_OTHERS),
        (
            Box::new(|| blk(&MemBlockBackend::new(DISK_SIZE))),
            2,
            PCI_DEVICE_ID_VIRTIO_BLOCK,
            PCI_CLASS_STORAGE_SCSI,
        ),
        (Box::new(console), 3, PCI_DEVICE_ID_VIRTIO_CONSOLE, PCI_CLASS_COMMUNICATION_OTHER),
    ];
    for (make, id, trans, class) in &cases {
        // The generic type on the root bus is transitional.
        let m = Machine::new(true);
        let vp = m.plug(make(), &VirtioPciProps::default());
        assert!(vp.is_legacy() && vp.is_modern());
        assert_eq!(m.cfg_read(0, 2), u32::from(PCI_VENDOR_ID_REDHAT_QUMRANET));
        assert_eq!(m.cfg_read(PCI_DEVICE_ID as u8, 2), u32::from(*trans));
        assert_eq!(m.cfg_read(PCI_REVISION_ID as u8, 1), 0);
        assert_eq!(m.cfg_read(PCI_CLASS_DEVICE as u8, 2), u32::from(*class));
        assert_eq!(m.cfg_read(PCI_SUBSYSTEM_VENDOR_ID as u8, 2), 0x1af4);
        assert_eq!(m.cfg_read(PCI_SUBSYSTEM_ID as u8, 2), u32::from(*id));
        assert_eq!(m.cfg_read(PCI_INTERRUPT_PIN as u8, 1), 1);
        assert_eq!(m.cfg_read(0x10, 4) & 1, 1, "BAR0 is I/O");
        assert_eq!(m.cfg_read(0x20, 4) & 0xf, 0xc, "BAR4 is 64-bit prefetchable memory");
        assert_eq!(m.bar_size(0x20), 0x4000);
        assert_eq!(m.bar_size(0x14), 0x1000);
        // blk has num_queues + 1 vectors, which with one queue matches the default of 2.
        assert_eq!(vp.nvectors(), 2);

        // Non-transitional: modern ID, revision 1, no legacy BAR, default subsystem.
        let m = Machine::new(true);
        let props = VirtioPciProps::for_variant(VirtioPciVariant::NonTransitional);
        let vp = m.plug(make(), &props);
        assert!(!vp.is_legacy() && vp.is_modern());
        assert_eq!(m.cfg_read(PCI_DEVICE_ID as u8, 2), u32::from(0x1040 + id));
        assert_eq!(m.cfg_read(PCI_REVISION_ID as u8, 1), 1);
        assert_eq!(m.cfg_read(PCI_CLASS_DEVICE as u8, 2), u32::from(*class));
        assert_eq!(m.cfg_read(PCI_SUBSYSTEM_ID as u8, 2), 0x1100);
        assert_eq!(m.bar_size(0x10), 0);

        // Legacy only: no vendor capabilities and no BAR4.
        let m = Machine::new(true);
        let props = VirtioPciProps { disable_modern: true, ..VirtioPciProps::default() };
        let vp = m.plug(make(), &props);
        assert!(vp.is_legacy() && !vp.is_modern());
        assert!(m.virtio_cap(VIRTIO_PCI_CAP_COMMON_CFG).is_none());
        assert_eq!(m.bar_size(0x20), 0);
        assert_eq!(m.cfg_read(PCI_DEVICE_ID as u8, 2), u32::from(*trans));
    }

    assert_eq!(
        VirtioPciVariant::Transitional.type_name(TYPE_VIRTIO_BLK_PCI),
        "virtio-blk-pci-transitional"
    );
    assert_eq!(
        VirtioPciVariant::NonTransitional.type_name(TYPE_VIRTIO_RNG_PCI),
        "virtio-rng-pci-non-transitional"
    );

    // The class property, and virtio-serial's whitelist.
    let m = Machine::new(true);
    let props = VirtioPciProps { class_code: PCI_CLASS_STORAGE_OTHER, ..VirtioPciProps::default() };
    m.plug(blk(&MemBlockBackend::new(DISK_SIZE)), &props);
    assert_eq!(m.cfg_read(PCI_CLASS_DEVICE as u8, 2), u32::from(PCI_CLASS_STORAGE_OTHER));
    for (asked, got) in [
        (PCI_CLASS_STORAGE_SCSI, PCI_CLASS_COMMUNICATION_OTHER),
        (PCI_CLASS_DISPLAY_OTHER, PCI_CLASS_DISPLAY_OTHER),
    ] {
        let m = Machine::new(true);
        let props = VirtioPciProps { class_code: asked, ..VirtioPciProps::default() };
        m.plug(console(), &props);
        assert_eq!(m.cfg_read(PCI_CLASS_DEVICE as u8, 2), u32::from(got));
    }

    // Neither mode is an error.
    let m = Machine::new(true);
    let props = VirtioPciProps { disable_modern: true, ..VirtioPciProps::non_transitional() };
    assert!(VirtioPci::new(&m.bus, None, m.backend(rng()), &props).is_err());

    assert_eq!(virtio_pci_optimal_num_queues(1, 4), 4);
    assert_eq!(virtio_pci_optimal_num_queues(1, 4096), 1023);
}

#[test]
fn capabilities() {
    let m = Machine::new(true);
    let vp = m.plug(rng(), &VirtioPciProps::default());
    let expect = [
        (VIRTIO_PCI_CAP_COMMON_CFG, 0x40, 16, 0x0, 0x1000),
        (VIRTIO_PCI_CAP_ISR_CFG, 0x50, 16, 0x1000, 0x1000),
        (VIRTIO_PCI_CAP_DEVICE_CFG, 0x60, 16, 0x2000, 0x1000),
        (VIRTIO_PCI_CAP_NOTIFY_CFG, 0x70, 20, 0x3000, 0x1000),
    ];
    for (t, pos, len, off, size) in expect {
        let c = m.virtio_cap(t).unwrap();
        assert_eq!((c.pos, c.len, c.bar, c.offset, c.length), (pos, len, 4, off, size), "{t}");
    }
    assert_eq!(m.virtio_cap(VIRTIO_PCI_CAP_NOTIFY_CFG).unwrap().mult, 4);
    let cfg = m.virtio_cap(VIRTIO_PCI_CAP_PCI_CFG).unwrap();
    assert_eq!((cfg.pos, cfg.len), (0x84, 20));
    assert_eq!(m.cap_pos(PCI_CAP_ID_MSIX), Some(0x98));
    assert_eq!(vp.pci_dev().msix_nr_vectors_allocated(), 2);

    // page-per-vq spaces the notify addresses a page apart and grows BAR4.
    let m = Machine::new(true);
    let props = VirtioPciProps { page_per_vq: true, ..VirtioPciProps::default() };
    m.plug(rng(), &props);
    let c = m.virtio_cap(VIRTIO_PCI_CAP_NOTIFY_CFG).unwrap();
    assert_eq!((c.mult, c.length), (0x1000, 0x40_0000));
    assert_eq!(m.bar_size(0x20), 0x80_0000);

    // modern-pio-notify adds a second notify capability for BAR2.
    let m = Machine::new(true);
    let props = VirtioPciProps { modern_pio_notify: true, ..VirtioPciProps::default() };
    m.plug(rng(), &props);
    let notify: Vec<_> = m
        .caps()
        .into_iter()
        .filter(|&(p, id)| id == PCI_CAP_ID_VNDR && m.cfg_read(p + 3, 1) == 2)
        .map(|(p, _)| (m.cfg_read(p + 4, 1), m.cfg_read(p + 12, 4), m.cfg_read(p + 16, 4)))
        .collect();
    assert!(notify.contains(&(2, 4, 0)));
    assert_eq!(m.bar_size(0x18), 4);
}

#[test]
fn feature_negotiation() {
    let m = Machine::new(true);
    let vp = m.plug(blk(&MemBlockBackend::new(DISK_SIZE)), &VirtioPciProps::default());
    let d = Modern::new(m, vp);
    let host = d.device_features();
    assert_ne!(host & feature(VIRTIO_F_VERSION_1), 0);
    assert_eq!(host & feature(VIRTIO_F_BAD_FEATURE), 0, "legacy features are hidden");
    d.cw(VIRTIO_PCI_COMMON_DFSELECT, 4, 2);
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_DF, 4), 0);

    let f = d.negotiate(wanted());
    d.cw(VIRTIO_PCI_COMMON_GFSELECT, 4, 1);
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_GF, 4), f >> 32);
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_GFSELECT, 4), 1);
    assert_eq!(d.vp.with_backend(|b| b.vdev().guest_features()), Some(f));

    // Features cannot change once FEATURES_OK is set.
    d.set_driver_features(0);
    assert_eq!(d.vp.with_backend(|b| b.vdev().guest_features()), Some(f));

    d.driver_ok();
    let all = VIRTIO_CONFIG_S_ACKNOWLEDGE
        | VIRTIO_CONFIG_S_DRIVER
        | VIRTIO_CONFIG_S_FEATURES_OK
        | VIRTIO_CONFIG_S_DRIVER_OK;
    assert_eq!(d.status(), all);
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_CFGGENERATION, 1), 0);

    // Device config: the blk capacity in sectors.
    let cap = d.m.mmio_read(d.device, 4) | d.m.mmio_read(d.device + 4, 4) << 32;
    assert_eq!(cap, (DISK_SIZE / 512) as u64);
}

#[test]
fn queue_setup_and_notify() {
    let m = Machine::new(true);
    let vp = m.plug(rng(), &VirtioPciProps::default());
    let mut d = Modern::new(m, vp);
    d.negotiate(wanted());
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_NUMQ, 2), 1);
    d.cw(VIRTIO_PCI_COMMON_Q_SELECT, 2, 0);
    let max = d.cr(VIRTIO_PCI_COMMON_Q_SIZE, 2);
    assert_ne!(max, 0);
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_Q_NOFF, 2), 0);
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_Q_ENABLE, 2), 0);
    // A smaller queue is fine.
    d.cw(VIRTIO_PCI_COMMON_Q_SIZE, 2, 4);
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_Q_SIZE, 2), 4);
    let mut q = d.setup_queue(0);
    assert_eq!(q.size, 4);
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_Q_DESCLO, 4), q.desc);
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_Q_AVAILLO, 4), q.avail);
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_Q_USEDLO, 4), q.used);
    d.driver_ok();

    let buf = d.m.alloc(16, 16);
    let head = q.submit(&d.m, &[(buf, 16, true)]);
    assert!(q.get_used(&d.m).is_none(), "nothing happens before the kick");
    d.kick(&q);
    assert_eq!(q.get_used(&d.m), Some((u32::from(head), 16)));
    assert_eq!(d.m.read_mem(buf, 16), vec![0xab; 16]);

    // Notify reads return 0 and kicking a queue that does not exist is harmless.
    assert_eq!(d.m.mmio_read(d.notify, 2), 0);
    d.m.mmio_write(d.notify + 5 * d.mult, 2, 5);

    // The I/O port notify window of modern-pio-notify.
    let m = Machine::new(true);
    let props = VirtioPciProps { modern_pio_notify: true, ..VirtioPciProps::default() };
    let vp = m.plug(rng(), &props);
    let mut d = Modern::new(m, vp);
    d.negotiate(wanted());
    let mut q = d.setup_queue(0);
    d.driver_ok();
    let buf = d.m.alloc(8, 16);
    q.submit(&d.m, &[(buf, 8, true)]);
    d.m.out(BAR2_PORT, 2, 0);
    assert_eq!(q.get_used(&d.m).map(|u| u.1), Some(8));

    // A bad Q_ENABLE value breaks the device.
    d.cw(VIRTIO_PCI_COMMON_Q_ENABLE, 2, 2);
    assert_ne!(d.status() & VIRTIO_CONFIG_S_NEEDS_RESET, 0);
}

#[test]
fn msix_vectors() {
    let m = Machine::new(true);
    let vp = m.plug(rng(), &VirtioPciProps::default());
    assert_eq!(vp.nvectors(), 2);
    let mut d = Modern::new(m, vp);
    d.m.enable_msix(2);
    d.negotiate(wanted());

    assert_eq!(d.cr(VIRTIO_PCI_COMMON_MSIX, 2), u64::from(VIRTIO_NO_VECTOR));
    d.cw(VIRTIO_PCI_COMMON_MSIX, 2, 0);
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_MSIX, 2), 0);
    let mut q = d.setup_queue(0);
    d.cw(VIRTIO_PCI_COMMON_Q_MSIX, 2, 7);
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_Q_MSIX, 2), u64::from(VIRTIO_NO_VECTOR), "out of range");
    d.cw(VIRTIO_PCI_COMMON_Q_MSIX, 2, 1);
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_Q_MSIX, 2), 1);
    d.driver_ok();

    let buf = d.m.alloc(8, 16);
    q.submit(&d.m, &[(buf, 8, true)]);
    d.kick(&q);
    assert!(q.get_used(&d.m).is_some());
    assert_eq!(d.m.take_msgs(), vec![(MSI_ADDR, 0x41)]);
    assert!(!d.m.intx(), "no INTx with MSI-X on");

    // A masked vector stays pending until it is unmasked.
    d.m.mmio_write(BAR1_ADDR + 16 + 12, 4, 1);
    q.submit(&d.m, &[(buf, 8, true)]);
    d.kick(&q);
    assert!(d.m.take_msgs().is_empty());
    d.m.mmio_write(BAR1_ADDR + 16 + 12, 4, 0);
    assert_eq!(d.m.take_msgs(), vec![(MSI_ADDR, 0x41)]);

    // Without an MSI capable platform the device falls back to INTx.
    let m = Machine::new(false);
    let vp = m.plug(rng(), &VirtioPciProps::default());
    assert_eq!(vp.nvectors(), 0);
    assert_eq!(m.cap_pos(PCI_CAP_ID_MSIX), None);
    assert_eq!(m.bar_size(0x10), 32, "legacy BAR without the MSI-X registers");

    // vectors=0 does the same.
    let m = Machine::new(true);
    let props = VirtioPciProps { vectors: Some(0), ..VirtioPciProps::default() };
    let vp = m.plug(blk(&MemBlockBackend::new(DISK_SIZE)), &props);
    assert_eq!(vp.nvectors(), 0);
    assert_eq!(m.cap_pos(PCI_CAP_ID_MSIX), None);
}

#[test]
fn intx_and_isr() {
    let m = Machine::new(true);
    let vp = m.plug(rng(), &VirtioPciProps::default());
    let mut d = Modern::new(m, vp);
    d.negotiate(wanted());
    let mut q = d.setup_queue(0);
    d.driver_ok();
    assert!(!d.m.intx());

    let buf = d.m.alloc(8, 16);
    q.submit(&d.m, &[(buf, 8, true)]);
    d.kick(&q);
    assert!(q.get_used(&d.m).is_some());
    assert!(d.m.intx(), "MSI-X is off, so INTx");
    assert!(d.m.take_msgs().is_empty());
    assert_eq!(d.isr(), 1);
    assert!(!d.m.intx(), "reading the ISR lowers the line");
    assert_eq!(d.isr(), 0);

    // With bus mastering off the device stays quiet and DRIVER_OK is dropped.
    let cmd = u32::from(PCI_COMMAND_IO | PCI_COMMAND_MEMORY);
    d.m.cfg_write(PCI_COMMAND as u8, 2, cmd);
    assert_eq!(d.status() & VIRTIO_CONFIG_S_DRIVER_OK, 0);
    assert_eq!(d.vp.with_backend(|b| b.vdev().is_disabled()), Some(true));
    d.m.cfg_write(PCI_COMMAND as u8, 2, cmd | u32::from(PCI_COMMAND_MASTER));
    assert_eq!(d.vp.with_backend(|b| b.vdev().is_disabled()), Some(false));
}

#[test]
fn pci_cfg_access_window() {
    let m = Machine::new(true);
    let vp = m.plug(blk(&MemBlockBackend::new(DISK_SIZE)), &VirtioPciProps::default());
    let d = Modern::new(m, vp);
    let cap = d.m.virtio_cap(VIRTIO_PCI_CAP_PCI_CFG).unwrap().pos;
    let access = |off: u32, len: u32| {
        d.m.cfg_write(cap + 4, 1, 4);
        d.m.cfg_write(cap + 8, 4, off);
        d.m.cfg_write(cap + 12, 4, len);
    };

    // Status through the window.
    access(VIRTIO_PCI_COMMON_STATUS as u32, 1);
    d.m.cfg_write(cap + 16, 1, u32::from(VIRTIO_CONFIG_S_ACKNOWLEDGE));
    assert_eq!(d.status(), VIRTIO_CONFIG_S_ACKNOWLEDGE);
    assert_eq!(d.m.cfg_read(cap + 16, 1), u32::from(VIRTIO_CONFIG_S_ACKNOWLEDGE));

    // NUMQ, and the device config at 0x2000.
    access(VIRTIO_PCI_COMMON_NUMQ as u32, 2);
    assert_eq!(d.m.cfg_read(cap + 16, 2), 1);
    access(0x2000, 4);
    assert_eq!(d.m.cfg_read(cap + 16, 4), (DISK_SIZE / 512) as u32);

    // The window register fields read back, and a bad length does nothing.
    assert_eq!(d.m.cfg_read(cap + 4, 1), 4);
    assert_eq!(d.m.cfg_read(cap + 8, 4), 0x2000);
    access(VIRTIO_PCI_COMMON_STATUS as u32, 3);
    d.m.cfg_write(cap + 16, 4, 0);
    assert_eq!(d.status(), VIRTIO_CONFIG_S_ACKNOWLEDGE);
}

/// The legacy driver of `virtio-pci.c` in libqos.
struct Legacy {
    m: Machine,
}

impl Legacy {
    fn inb(&self, off: u64) -> u8 {
        self.m.inp(BAR0_PORT + off, 1) as u8
    }
    fn inw(&self, off: u64) -> u16 {
        self.m.inp(BAR0_PORT + off, 2) as u16
    }
    fn inl(&self, off: u64) -> u32 {
        self.m.inp(BAR0_PORT + off, 4)
    }
    fn outb(&self, off: u64, v: u8) {
        self.m.out(BAR0_PORT + off, 1, v.into());
    }
    fn outw(&self, off: u64, v: u16) {
        self.m.out(BAR0_PORT + off, 2, v.into());
    }
    fn outl(&self, off: u64, v: u32) {
        self.m.out(BAR0_PORT + off, 4, v);
    }

    fn setup_queue(&mut self, index: u16) -> Ring {
        self.outw(VIRTIO_PCI_QUEUE_SEL, index);
        let size = self.inw(VIRTIO_PCI_QUEUE_NUM);
        assert_ne!(size, 0);
        let n = u64::from(size);
        let used_off = (16 * n + 6 + 2 * n).div_ceil(4096) * 4096;
        let desc = self.m.alloc(used_off + 6 + 8 * n, 4096);
        self.outl(VIRTIO_PCI_QUEUE_PFN, (desc >> 12) as u32);
        assert_eq!(u64::from(self.inl(VIRTIO_PCI_QUEUE_PFN)), desc >> 12);
        Ring {
            index,
            size,
            desc,
            avail: desc + 16 * n,
            used: desc + used_off,
            avail_idx: 0,
            last_used: 0,
        }
    }
}

#[test]
fn legacy_bar0() {
    let disk = MemBlockBackend::new(DISK_SIZE);
    let m = Machine::new(true);
    let vp = m.plug(blk(&disk), &VirtioPciProps::transitional());
    m.enable();
    // Like an old Linux, leave bus mastering off; ACK|DRIVER turns it on.
    m.cfg_write(PCI_COMMAND as u8, 2, u32::from(PCI_COMMAND_IO | PCI_COMMAND_MEMORY));
    let mut g = Legacy { m };

    let host = g.inl(VIRTIO_PCI_HOST_FEATURES);
    assert_ne!(host & (1 << VIRTIO_F_BAD_FEATURE), 0, "legacy sees BAD_FEATURE");
    g.outb(VIRTIO_PCI_STATUS, 0);
    g.outb(VIRTIO_PCI_STATUS, VIRTIO_CONFIG_S_ACKNOWLEDGE);
    g.outb(VIRTIO_PCI_STATUS, VIRTIO_CONFIG_S_ACKNOWLEDGE | VIRTIO_CONFIG_S_DRIVER);
    assert_ne!(g.m.cfg_read(PCI_COMMAND as u8, 2) & u32::from(PCI_COMMAND_MASTER), 0);
    let f = host & wanted() as u32 & !(1 << VIRTIO_F_BAD_FEATURE);
    g.outl(VIRTIO_PCI_GUEST_FEATURES, f);
    assert_eq!(g.inl(VIRTIO_PCI_GUEST_FEATURES), f);
    // A driver that sets BAD_FEATURE gets nothing.
    g.outl(VIRTIO_PCI_GUEST_FEATURES, 1 << VIRTIO_F_BAD_FEATURE);
    assert_eq!(g.inl(VIRTIO_PCI_GUEST_FEATURES), 0);
    g.outl(VIRTIO_PCI_GUEST_FEATURES, f);

    // Device config starts at 20 while MSI-X is off.
    assert_eq!(g.inl(20), (DISK_SIZE / 512) as u32);
    let mut q = g.setup_queue(0);
    g.outb(VIRTIO_PCI_STATUS, g.inb(VIRTIO_PCI_STATUS) | VIRTIO_CONFIG_S_DRIVER_OK);

    let kick =
        |m: &Machine, q: &Ring| m.out(BAR0_PORT + VIRTIO_PCI_QUEUE_NOTIFY, 2, q.index.into());
    let (st, len, _) =
        blk_request(&mut g.m, &mut q, &kick, VIRTIO_BLK_T_OUT, 0, &sector(b"LEGACY"), 0);
    assert_eq!((st, len), (VIRTIO_BLK_S_OK, 1));
    assert_eq!(&disk.data().lock().unwrap()[..6], b"LEGACY");
    assert!(g.m.intx());
    assert_eq!(g.inb(VIRTIO_PCI_ISR), 1);
    assert!(!g.m.intx());
    assert_eq!(g.inb(VIRTIO_PCI_ISR), 0);
    let (st, len, data) = blk_request(&mut g.m, &mut q, &kick, VIRTIO_BLK_T_IN, 0, &[], 512);
    assert_eq!((st, len), (VIRTIO_BLK_S_OK, 513));
    assert_eq!(data, sector(b"LEGACY"));
    g.inb(VIRTIO_PCI_ISR);

    // With MSI-X on, the vector registers appear and the config moves to 24.
    g.m.enable_msix(2);
    assert_eq!(g.inl(24), (DISK_SIZE / 512) as u32);
    assert_eq!(g.inw(VIRTIO_MSI_CONFIG_VECTOR), VIRTIO_NO_VECTOR);
    g.outw(VIRTIO_MSI_CONFIG_VECTOR, 0);
    assert_eq!(g.inw(VIRTIO_MSI_CONFIG_VECTOR), 0);
    g.outw(VIRTIO_MSI_QUEUE_VECTOR, 1);
    assert_eq!(g.inw(VIRTIO_MSI_QUEUE_VECTOR), 1);
    g.outw(VIRTIO_MSI_QUEUE_VECTOR, 9);
    assert_eq!(g.inw(VIRTIO_MSI_QUEUE_VECTOR), VIRTIO_NO_VECTOR);
    g.outw(VIRTIO_MSI_QUEUE_VECTOR, 1);
    let (st, _, _) = blk_request(&mut g.m, &mut q, &kick, VIRTIO_BLK_T_IN, 0, &[], 512);
    assert_eq!(st, VIRTIO_BLK_S_OK);
    assert_eq!(g.m.take_msgs(), vec![(MSI_ADDR, 0x41)]);

    // Writing 0 to the PFN resets the device.
    g.outl(VIRTIO_PCI_QUEUE_PFN, 0);
    assert_eq!(g.inb(VIRTIO_PCI_STATUS), 0);
    assert_eq!(g.inl(VIRTIO_PCI_QUEUE_PFN), 0);
    assert_eq!(g.inw(VIRTIO_MSI_CONFIG_VECTOR), VIRTIO_NO_VECTOR);
    assert_eq!(vp.with_backend(|b| b.vdev().guest_features()), Some(0));
}

#[test]
fn blk_over_modern() {
    let disk = MemBlockBackend::new(DISK_SIZE);
    let m = Machine::new(true);
    let vp = m.plug(blk(&disk), &VirtioPciProps::non_transitional());
    let mut d = Modern::new(m, vp);
    d.m.enable_msix(2);
    d.negotiate(wanted());
    let mut q = d.setup_queue(0);
    d.cw(VIRTIO_PCI_COMMON_Q_MSIX, 2, 1);
    d.driver_ok();

    let (notify, mult) = (d.notify, d.mult);
    let kick = move |m: &Machine, q: &Ring| {
        m.mmio_write(notify + mult * u64::from(q.index), 2, u64::from(q.index));
    };
    let (st, len, _) =
        blk_request(&mut d.m, &mut q, &kick, VIRTIO_BLK_T_OUT, 3, &sector(b"TEST"), 0);
    assert_eq!((st, len), (VIRTIO_BLK_S_OK, 1));
    assert_eq!(&disk.data().lock().unwrap()[3 * 512..3 * 512 + 4], b"TEST");
    let (st, len, data) = blk_request(&mut d.m, &mut q, &kick, VIRTIO_BLK_T_IN, 3, &[], 512);
    assert_eq!((st, len), (VIRTIO_BLK_S_OK, 513));
    assert_eq!(data, sector(b"TEST"));
    assert_eq!(d.m.take_msgs(), vec![(MSI_ADDR, 0x41); 2]);
}

#[test]
fn reset() {
    let m = Machine::new(true);
    let vp = m.plug(rng(), &VirtioPciProps::default());
    let mut d = Modern::new(m, vp);
    d.m.enable_msix(2);
    d.negotiate(wanted());
    d.cw(VIRTIO_PCI_COMMON_MSIX, 2, 0);
    let max = d.cr(VIRTIO_PCI_COMMON_Q_SIZE, 2);
    d.cw(VIRTIO_PCI_COMMON_Q_SIZE, 2, 4);
    let mut q = d.setup_queue(0);
    d.cw(VIRTIO_PCI_COMMON_Q_MSIX, 2, 1);
    d.driver_ok();
    let buf = d.m.alloc(8, 16);
    q.submit(&d.m, &[(buf, 8, true)]);
    d.kick(&q);
    assert!(q.get_used(&d.m).is_some());

    // Status 0 resets the device and the transport's queue state.
    d.set_status(0);
    assert_eq!(d.status(), 0);
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_MSIX, 2), u64::from(VIRTIO_NO_VECTOR));
    d.cw(VIRTIO_PCI_COMMON_Q_SELECT, 2, 0);
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_Q_ENABLE, 2), 0);
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_Q_MSIX, 2), u64::from(VIRTIO_NO_VECTOR));
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_Q_SIZE, 2), max);
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_Q_DESCLO, 4), 0);
    d.cw(VIRTIO_PCI_COMMON_GFSELECT, 4, 0);
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_GF, 4), 0);
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_NUMQ, 2), 1);

    // The device works again after a fresh setup.
    d.negotiate(wanted());
    let mut q = d.setup_queue(0);
    d.driver_ok();
    let buf = d.m.alloc(8, 16);
    q.submit(&d.m, &[(buf, 8, true)]);
    d.kick(&q);
    assert!(q.get_used(&d.m).is_some());
    d.isr();

    // A queue reset through Q_RESET.
    d.cw(VIRTIO_PCI_COMMON_Q_RESET, 2, 1);
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_Q_ENABLE, 2), 0);
    assert_eq!(d.cr(VIRTIO_PCI_COMMON_Q_RESET, 2), 0);

    // A device reset from the board also clears the PCI command register.
    d.vp.reset();
    assert_eq!(d.vp.with_backend(|b| b.vdev().status()), Some(0));
    assert_eq!(d.m.cfg_read(PCI_COMMAND as u8, 2), 0);
}

#[test]
fn pcie_port_is_modern_only() {
    let m = Machine::new(true);
    let info = PciDeviceInfo {
        name: "pcie-root-port".into(),
        vendor_id: 0x1b36,
        device_id: 0x000c,
        class_id: PCI_CLASS_BRIDGE_PCI,
        express: true,
        ..PciDeviceInfo::default()
    };
    let port = PciBridge::new(&m.bus, &info, Some(1 << 3), "pcie.1").unwrap();
    let sec = Arc::clone(port.sec_bus());
    let vp = VirtioPci::new(&sec, Some(0), m.backend(rng()), &VirtioPciProps::default()).unwrap();
    assert!(vp.is_modern() && !vp.is_legacy());
    let dev = vp.pci_dev();
    assert!(dev.is_express());
    assert_eq!(dev.config_size(), 4096);
    assert_eq!(dev.config_read(PCI_DEVICE_ID as u32, 2), 0x1044);
    let pm = dev.find_capability(PCI_CAP_ID_PM);
    assert_ne!(pm, 0);
    assert_eq!(dev.config_read(u32::from(pm) + PCI_PM_PMC as u32, 2), 3);

    // Transitional can still be asked for; it then has a legacy BAR.
    let vp = VirtioPci::new(&sec, Some(1 << 3), m.backend(rng()), &VirtioPciProps::transitional())
        .unwrap();
    assert!(vp.is_legacy() && vp.is_modern());
    assert!(vp.pci_dev().bar_info(0).is_some_and(|b| b.size == 32));
}
