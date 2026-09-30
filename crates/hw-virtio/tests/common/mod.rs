// SPDX-License-Identifier: GPL-2.0-or-later

//! A tiny guest side virtio driver for the integration tests, modelled on libqos
//! (`tests/qtest/libqos/virtio-mmio.c` and `virtio.c`).
//!
//! The machine is guest RAM at 0 and one virtio-mmio window at [`MMIO_BASE`], both in a real
//! `ruvm-mem` address space. The driver pokes the registers through that address space and
//! builds split rings in RAM.

#![allow(dead_code, unreachable_pub)]

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

use ruvm_hw_core::IrqLine;
use ruvm_hw_virtio::mmio::*;
use ruvm_hw_virtio::virtio::*;
use ruvm_hw_virtio::{AddressSpaceMemory, VirtioBackend, VirtioDeviceClass, VirtioMmio};
use ruvm_mem::{AddressSpace, Endian, MemTxAttrs, MemorySystem};

pub const RAM_SIZE: u64 = 4 << 20;
pub const MMIO_BASE: u64 = 0x1000_0000;
pub const PAGE_SIZE: u64 = 4096;

pub const VRING_DESC_F_NEXT: u16 = 1;
pub const VRING_DESC_F_WRITE: u16 = 2;
pub const VRING_DESC_F_INDIRECT: u16 = 4;
pub const VRING_AVAIL_F_NO_INTERRUPT: u16 = 1;

/// One buffer of a chain: guest address, length, device writable.
#[derive(Clone, Copy, Debug)]
pub struct Buf {
    pub addr: u64,
    pub len: u32,
    pub write: bool,
}

impl Buf {
    pub fn out(addr: u64, len: u32) -> Self {
        Buf { addr, len, write: false }
    }
    pub fn inp(addr: u64, len: u32) -> Self {
        Buf { addr, len, write: true }
    }
}

pub struct Guest {
    pub space: Arc<AddressSpace>,
    pub mmio: Arc<VirtioMmio>,
    pub level: Arc<AtomicI32>,
    pub legacy: bool,
    next_alloc: u64,
}

impl Guest {
    /// Builds the machine. `make` gets the guest memory and returns the device model to plug,
    /// or `None` for an empty transport.
    pub fn new(legacy: bool, make: Option<Box<dyn VirtioDeviceClass>>) -> Self {
        Self::try_new(legacy, make).expect("device realizes")
    }

    pub fn try_new(
        legacy: bool,
        make: Option<Box<dyn VirtioDeviceClass>>,
    ) -> ruvm_base::Result<Self> {
        Self::with_backend(legacy, make, |_| {})
    }

    /// Like [`try_new`], with a hook to change the backend (say, host features) before it is
    /// plugged.
    pub fn with_backend(
        legacy: bool,
        make: Option<Box<dyn VirtioDeviceClass>>,
        tweak: impl FnOnce(&mut VirtioBackend),
    ) -> ruvm_base::Result<Self> {
        let sys = MemorySystem::new();
        let root = sys.new_container("system", u128::from(u64::MAX)).unwrap();
        let ram = sys.new_ram("ram", RAM_SIZE).unwrap();
        sys.add_subregion(root, 0, ram).unwrap();
        let space = sys.address_space_init(root, "memory").unwrap();
        let mem: SharedGuestMemory = Arc::new(AddressSpaceMemory::new(Arc::clone(&space)));
        let backend = match make {
            Some(class) => {
                let mut b = VirtioBackend::new(class, mem)?;
                tweak(&mut b);
                Some(b)
            }
            None => None,
        };
        let mmio = Arc::new(VirtioMmio::new(backend, legacy)?);
        let level = Arc::new(AtomicI32::new(0));
        let l = Arc::clone(&level);
        mmio.irq().connect(IrqLine::from_fn(move |v| l.store(v, Ordering::SeqCst)));
        let region =
            sys.new_io("virtio-mmio", u128::from(VIRTIO_MMIO_REGION_SIZE), mmio.clone()).unwrap();
        sys.add_subregion(root, MMIO_BASE, region).unwrap();
        Ok(Guest { space, mmio, level, legacy, next_alloc: 0x10000 })
    }

    pub fn irq(&self) -> bool {
        self.level.load(Ordering::SeqCst) != 0
    }

    // Register access.

    pub fn load(&self, off: u64, size: u32) -> u64 {
        let (v, r) =
            self.space.load(MMIO_BASE + off, size, Endian::Little, MemTxAttrs::UNSPECIFIED);
        assert!(r.is_ok());
        v
    }

    pub fn store(&self, off: u64, size: u32, value: u64) {
        let r =
            self.space.store(MMIO_BASE + off, size, value, Endian::Little, MemTxAttrs::UNSPECIFIED);
        assert!(r.is_ok());
    }

    pub fn readl(&self, off: u64) -> u32 {
        self.load(off, 4) as u32
    }

    pub fn writel(&self, off: u64, value: u32) {
        self.store(off, 4, u64::from(value));
    }

    pub fn config_readb(&self, off: u64) -> u8 {
        self.load(VIRTIO_MMIO_CONFIG + off, 1) as u8
    }

    pub fn config_readw(&self, off: u64) -> u16 {
        self.load(VIRTIO_MMIO_CONFIG + off, 2) as u16
    }

    pub fn config_readl(&self, off: u64) -> u32 {
        self.load(VIRTIO_MMIO_CONFIG + off, 4) as u32
    }

    /// Two 32 bit reads, the way libqos reads a 64 bit config field.
    pub fn config_readq(&self, off: u64) -> u64 {
        u64::from(self.config_readl(off)) | u64::from(self.config_readl(off + 4)) << 32
    }

    pub fn status(&self) -> u8 {
        self.readl(VIRTIO_MMIO_STATUS) as u8
    }

    pub fn set_status(&self, s: u8) {
        self.writel(VIRTIO_MMIO_STATUS, u32::from(s));
    }

    pub fn device_features(&self) -> u64 {
        self.writel(VIRTIO_MMIO_DEVICE_FEATURES_SEL, 0);
        let lo = self.readl(VIRTIO_MMIO_DEVICE_FEATURES);
        self.writel(VIRTIO_MMIO_DEVICE_FEATURES_SEL, 1);
        let hi = self.readl(VIRTIO_MMIO_DEVICE_FEATURES);
        u64::from(hi) << 32 | u64::from(lo)
    }

    pub fn set_driver_features(&self, f: u64) {
        if self.legacy {
            self.writel(VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
            self.writel(VIRTIO_MMIO_DRIVER_FEATURES, f as u32);
        } else {
            self.writel(VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
            self.writel(VIRTIO_MMIO_DRIVER_FEATURES, f as u32);
            self.writel(VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
            self.writel(VIRTIO_MMIO_DRIVER_FEATURES, (f >> 32) as u32);
        }
    }

    /// Reset, ACKNOWLEDGE, DRIVER, then accept `wanted & device features`, then FEATURES_OK on
    /// a modern device. Returns the accepted features.
    pub fn negotiate(&self, wanted: u64) -> u64 {
        self.set_status(0);
        assert_eq!(self.status(), 0);
        self.set_status(VIRTIO_CONFIG_S_ACKNOWLEDGE);
        self.set_status(VIRTIO_CONFIG_S_ACKNOWLEDGE | VIRTIO_CONFIG_S_DRIVER);
        let f = self.device_features() & wanted;
        self.set_driver_features(f);
        if !self.legacy {
            let s = self.status() | VIRTIO_CONFIG_S_FEATURES_OK;
            self.set_status(s);
            assert_ne!(self.status() & VIRTIO_CONFIG_S_FEATURES_OK, 0);
        }
        f
    }

    pub fn driver_ok(&self) {
        let s = self.status() | VIRTIO_CONFIG_S_DRIVER_OK;
        self.set_status(s);
    }

    pub fn isr(&self) -> u32 {
        self.readl(VIRTIO_MMIO_INTERRUPT_STATUS)
    }

    pub fn ack(&self) {
        let isr = self.isr();
        self.writel(VIRTIO_MMIO_INTERRUPT_ACK, isr);
    }

    // Guest memory.

    pub fn alloc(&mut self, size: u64, align: u64) -> u64 {
        let addr = self.next_alloc.div_ceil(align) * align;
        self.next_alloc = addr + size;
        assert!(self.next_alloc <= RAM_SIZE);
        addr
    }

    pub fn write_mem(&self, addr: u64, data: &[u8]) {
        assert!(self.space.write(addr, MemTxAttrs::UNSPECIFIED, data).is_ok());
    }

    pub fn read_mem(&self, addr: u64, len: usize) -> Vec<u8> {
        let mut v = vec![0; len];
        assert!(self.space.read(addr, MemTxAttrs::UNSPECIFIED, &mut v).is_ok());
        v
    }

    pub fn write_u16(&self, addr: u64, v: u16) {
        self.write_mem(addr, &v.to_le_bytes());
    }

    pub fn read_u16(&self, addr: u64) -> u16 {
        u16::from_le_bytes(self.read_mem(addr, 2).try_into().unwrap())
    }

    pub fn read_u32(&self, addr: u64) -> u32 {
        u32::from_le_bytes(self.read_mem(addr, 4).try_into().unwrap())
    }

    /// Sets up queue `index` with `size` entries (or the device maximum when 0) the way
    /// `qvirtio_mmio_virtqueue_setup()` does.
    pub fn setup_queue(&mut self, index: u16, size: u16) -> SplitRing {
        self.writel(VIRTIO_MMIO_QUEUE_SEL, u32::from(index));
        let max = self.readl(VIRTIO_MMIO_QUEUE_NUM_MAX) as u16;
        assert_ne!(max, 0, "queue {index} does not exist");
        let size = if size == 0 { max } else { size };
        let n = u64::from(size);
        if self.legacy {
            self.writel(VIRTIO_MMIO_GUEST_PAGE_SIZE, PAGE_SIZE as u32);
            assert_eq!(self.readl(VIRTIO_MMIO_QUEUE_PFN), 0);
            self.writel(VIRTIO_MMIO_QUEUE_NUM, u32::from(size));
            self.writel(VIRTIO_MMIO_QUEUE_ALIGN, PAGE_SIZE as u32);
            // vring_size() with the legacy alignment.
            let avail_end = 16 * n + 2 * (3 + n);
            let used_off = avail_end.div_ceil(PAGE_SIZE) * PAGE_SIZE;
            let total = used_off + 6 + 8 * n;
            let desc = self.alloc(total, PAGE_SIZE);
            self.write_mem(desc, &vec![0; total as usize]);
            self.writel(VIRTIO_MMIO_QUEUE_PFN, (desc / PAGE_SIZE) as u32);
            assert_eq!(u64::from(self.readl(VIRTIO_MMIO_QUEUE_PFN)), desc / PAGE_SIZE);
            SplitRing::new(index, size, desc, desc + 16 * n, desc + used_off)
        } else {
            let desc = self.alloc(16 * n, 16);
            let avail = self.alloc(6 + 2 * n, 2);
            let used = self.alloc(6 + 8 * n, 4);
            self.write_mem(desc, &vec![0; (16 * n) as usize]);
            self.write_mem(avail, &vec![0; (6 + 2 * n) as usize]);
            self.write_mem(used, &vec![0; (6 + 8 * n) as usize]);
            self.writel(VIRTIO_MMIO_QUEUE_NUM, u32::from(size));
            self.writel(VIRTIO_MMIO_QUEUE_DESC_LOW, desc as u32);
            self.writel(VIRTIO_MMIO_QUEUE_DESC_HIGH, (desc >> 32) as u32);
            self.writel(VIRTIO_MMIO_QUEUE_AVAIL_LOW, avail as u32);
            self.writel(VIRTIO_MMIO_QUEUE_AVAIL_HIGH, (avail >> 32) as u32);
            self.writel(VIRTIO_MMIO_QUEUE_USED_LOW, used as u32);
            self.writel(VIRTIO_MMIO_QUEUE_USED_HIGH, (used >> 32) as u32);
            self.writel(VIRTIO_MMIO_QUEUE_READY, 1);
            assert_eq!(self.readl(VIRTIO_MMIO_QUEUE_READY), 1);
            SplitRing::new(index, size, desc, avail, used)
        }
    }

    pub fn kick(&self, ring: &SplitRing) {
        self.writel(VIRTIO_MMIO_QUEUE_NOTIFY, u32::from(ring.index));
    }
}

/// The driver side of a split ring, `QVirtQueue`.
#[derive(Debug)]
pub struct SplitRing {
    pub index: u16,
    pub size: u16,
    pub desc: u64,
    pub avail: u64,
    pub used: u64,
    free_head: u16,
    avail_idx: u16,
    last_used: u16,
}

impl SplitRing {
    fn new(index: u16, size: u16, desc: u64, avail: u64, used: u64) -> Self {
        SplitRing { index, size, desc, avail, used, free_head: 0, avail_idx: 0, last_used: 0 }
    }

    fn write_desc(&self, g: &Guest, slot: u16, addr: u64, len: u32, flags: u16, next: u16) {
        let mut d = Vec::with_capacity(16);
        d.extend_from_slice(&addr.to_le_bytes());
        d.extend_from_slice(&len.to_le_bytes());
        d.extend_from_slice(&flags.to_le_bytes());
        d.extend_from_slice(&next.to_le_bytes());
        g.write_mem(self.desc + 16 * u64::from(slot), &d);
    }

    /// Puts a chain in the descriptor table without making it available. Returns the head.
    pub fn add_chain(&mut self, g: &Guest, bufs: &[Buf]) -> u16 {
        let head = self.free_head;
        for (i, b) in bufs.iter().enumerate() {
            let slot = (self.free_head + i as u16) % self.size;
            let mut flags = if b.write { VRING_DESC_F_WRITE } else { 0 };
            let next = (slot + 1) % self.size;
            if i + 1 < bufs.len() {
                flags |= VRING_DESC_F_NEXT;
            }
            self.write_desc(g, slot, b.addr, b.len, flags, next);
        }
        self.free_head = (self.free_head + bufs.len() as u16) % self.size;
        head
    }

    /// Puts one descriptor pointing at an indirect table holding `bufs`, `qvring_indirect_desc_add`.
    pub fn add_indirect(&mut self, g: &mut Guest, bufs: &[Buf]) -> u16 {
        let table = g.alloc(16 * bufs.len() as u64, 16);
        let mut raw = Vec::new();
        for (i, b) in bufs.iter().enumerate() {
            let mut flags = if b.write { VRING_DESC_F_WRITE } else { 0 };
            if i + 1 < bufs.len() {
                flags |= VRING_DESC_F_NEXT;
            }
            raw.extend_from_slice(&b.addr.to_le_bytes());
            raw.extend_from_slice(&b.len.to_le_bytes());
            raw.extend_from_slice(&flags.to_le_bytes());
            raw.extend_from_slice(&(i as u16 + 1).to_le_bytes());
        }
        g.write_mem(table, &raw);
        let head = self.free_head;
        self.write_desc(g, head, table, raw.len() as u32, VRING_DESC_F_INDIRECT, 0);
        self.free_head = (self.free_head + 1) % self.size;
        head
    }

    /// Makes `head` available, `qvirtqueue_kick()` minus the notify.
    pub fn make_available(&mut self, g: &Guest, head: u16) {
        let slot = self.avail_idx % self.size;
        g.write_u16(self.avail + 4 + 2 * u64::from(slot), head);
        self.avail_idx = self.avail_idx.wrapping_add(1);
        g.write_u16(self.avail + 2, self.avail_idx);
    }

    /// Adds a chain and makes it available in one go.
    pub fn submit(&mut self, g: &Guest, bufs: &[Buf]) -> u16 {
        let head = self.add_chain(g, bufs);
        self.make_available(g, head);
        head
    }

    pub fn set_avail_flags(&self, g: &Guest, flags: u16) {
        g.write_u16(self.avail, flags);
    }

    /// Writes `used_event`, the slot after the available ring.
    pub fn set_used_event(&self, g: &Guest, idx: u16) {
        g.write_u16(self.avail + 4 + 2 * u64::from(self.size), idx);
    }

    pub fn used_idx(&self, g: &Guest) -> u16 {
        g.read_u16(self.used + 2)
    }

    pub fn used_flags(&self, g: &Guest) -> u16 {
        g.read_u16(self.used)
    }

    /// Takes the next used element, `qvirtqueue_get_buf()`.
    pub fn get_used(&mut self, g: &Guest) -> Option<(u32, u32)> {
        if self.used_idx(g) == self.last_used {
            return None;
        }
        let slot = self.last_used % self.size;
        let e = self.used + 4 + 8 * u64::from(slot);
        self.last_used = self.last_used.wrapping_add(1);
        Some((g.read_u32(e), g.read_u32(e + 4)))
    }
}
