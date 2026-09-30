// SPDX-License-Identifier: MIT OR Apache-2.0

//! Minimal drivers for both ring formats. They write the rings straight into a `VecMemory` the way
//! a guest driver would, so the tests can check what the device side makes of them.

#![allow(dead_code)]

use std::collections::HashMap;

use ruvm_virtio_queue::{
    GuestMemory, RING_EVENT_FLAGS_ENABLE, RingAddresses, VIRTQ_DESC_F_AVAIL, VIRTQ_DESC_F_INDIRECT,
    VIRTQ_DESC_F_NEXT, VIRTQ_DESC_F_USED, VIRTQ_DESC_F_WRITE, VecMemory,
};

/// Where test data buffers live. The rings sit below this.
pub(crate) const DATA: u64 = 0x10_0000;
/// Total memory for a test.
pub(crate) const MEM_SIZE: usize = 0x20_0000;

/// A buffer as a driver describes it: address, length, device writable.
pub(crate) type Buf = (u64, u32, bool);

pub(crate) struct SplitDriver {
    pub(crate) mem: VecMemory,
    pub(crate) addrs: RingAddresses,
    pub(crate) size: u16,
    pub(crate) avail_idx: u16,
}

impl SplitDriver {
    pub(crate) fn new(size: u16) -> Self {
        Self {
            mem: VecMemory::new(MEM_SIZE),
            addrs: RingAddresses { desc_table: 0, driver_area: 0x8_0000, device_area: 0x9_0000 },
            size,
            avail_idx: 0,
        }
    }

    pub(crate) fn write_desc(&self, index: u16, addr: u64, len: u32, flags: u16, next: u16) {
        let at = self.addrs.desc_table + 16 * u64::from(index);
        self.write_raw_desc(at, addr, len, flags, next);
    }

    pub(crate) fn write_raw_desc(&self, at: u64, addr: u64, len: u32, flags: u16, next: u16) {
        self.mem.write_u64(at, addr).unwrap();
        self.mem.write_u32(at + 8, len).unwrap();
        self.mem.write_u16(at + 12, flags).unwrap();
        self.mem.write_u16(at + 14, next).unwrap();
    }

    /// Writes `bufs` as a chain in consecutive table entries starting at `first`.
    pub(crate) fn write_chain(&self, first: u16, bufs: &[Buf]) {
        for (i, &(addr, len, write)) in bufs.iter().enumerate() {
            let index = first + i as u16;
            let mut flags = if write { VIRTQ_DESC_F_WRITE } else { 0 };
            if i + 1 < bufs.len() {
                flags |= VIRTQ_DESC_F_NEXT;
            }
            self.write_desc(index, addr, len, flags, (index + 1) % self.size);
        }
    }

    /// Puts `head` in the available ring and bumps the index.
    pub(crate) fn publish(&mut self, head: u16) {
        let slot = self.avail_idx % self.size;
        self.mem.write_u16(self.addrs.driver_area + 4 + 2 * u64::from(slot), head).unwrap();
        self.avail_idx = self.avail_idx.wrapping_add(1);
        self.mem.write_u16(self.addrs.driver_area + 2, self.avail_idx).unwrap();
    }

    pub(crate) fn add_chain(&mut self, first: u16, bufs: &[Buf]) {
        self.write_chain(first, bufs);
        self.publish(first);
    }

    /// Makes the driver's view start at `idx`, as after a lot of earlier traffic.
    pub(crate) fn set_avail_idx(&mut self, idx: u16) {
        self.avail_idx = idx;
        self.mem.write_u16(self.addrs.driver_area + 2, idx).unwrap();
    }

    pub(crate) fn set_avail_flags(&self, flags: u16) {
        self.mem.write_u16(self.addrs.driver_area, flags).unwrap();
    }

    pub(crate) fn set_used_event(&self, idx: u16) {
        let at = self.addrs.driver_area + 4 + 2 * u64::from(self.size);
        self.mem.write_u16(at, idx).unwrap();
    }

    pub(crate) fn used_flags(&self) -> u16 {
        self.mem.read_u16(self.addrs.device_area).unwrap()
    }

    pub(crate) fn used_idx(&self) -> u16 {
        self.mem.read_u16(self.addrs.device_area + 2).unwrap()
    }

    /// The used element at free running index `idx`.
    pub(crate) fn used_elem(&self, idx: u16) -> (u32, u32) {
        let at = self.addrs.device_area + 4 + 8 * u64::from(idx % self.size);
        (self.mem.read_u32(at).unwrap(), self.mem.read_u32(at + 4).unwrap())
    }

    pub(crate) fn avail_event(&self) -> u16 {
        let at = self.addrs.device_area + 4 + 8 * u64::from(self.size);
        self.mem.read_u16(at).unwrap()
    }
}

pub(crate) struct PackedDriver {
    pub(crate) mem: VecMemory,
    pub(crate) addrs: RingAddresses,
    pub(crate) size: u16,
    pub(crate) next: u16,
    pub(crate) wrap: bool,
    used_next: u16,
    used_wrap: bool,
    slots_by_id: HashMap<u16, u16>,
}

impl PackedDriver {
    pub(crate) fn new(size: u16) -> Self {
        Self {
            mem: VecMemory::new(MEM_SIZE),
            addrs: RingAddresses { desc_table: 0, driver_area: 0x8_0000, device_area: 0x9_0000 },
            size,
            next: 0,
            wrap: true,
            used_next: 0,
            used_wrap: true,
            slots_by_id: HashMap::new(),
        }
    }

    fn avail_bits(wrap: bool) -> u16 {
        if wrap { VIRTQ_DESC_F_AVAIL } else { VIRTQ_DESC_F_USED }
    }

    pub(crate) fn write_desc(&self, offset: u16, addr: u64, len: u32, id: u16, flags: u16) {
        let at = self.addrs.desc_table + 16 * u64::from(offset);
        self.mem.write_u64(at, addr).unwrap();
        self.mem.write_u32(at + 8, len).unwrap();
        self.mem.write_u16(at + 12, id).unwrap();
        self.mem.write_u16(at + 14, flags).unwrap();
    }

    /// Makes ring descriptors available. Each entry is (addr, len, id, extra flags); the avail
    /// and used bits are added here to match the wrap counter at each slot, and the first
    /// descriptor's flags are written last as the specification asks.
    pub(crate) fn add_raw(&mut self, descs: &[(u64, u32, u16, u16)]) {
        let (start, start_wrap) = (self.next, self.wrap);
        let mut head_flags = 0;
        for (i, &(addr, len, id, flags)) in descs.iter().enumerate() {
            let flags = flags | Self::avail_bits(self.wrap);
            if i == 0 {
                head_flags = flags;
                // Flags that do not mark it available yet, so the head is not seen early.
                self.write_desc(self.next, addr, len, id, Self::avail_bits(!start_wrap));
            } else {
                self.write_desc(self.next, addr, len, id, flags);
            }
            self.next += 1;
            if self.next == self.size {
                self.next = 0;
                self.wrap = !self.wrap;
            }
        }
        let at = self.addrs.desc_table + 16 * u64::from(start) + 14;
        self.mem.write_u16(at, head_flags).unwrap();
    }

    /// Makes `bufs` available as one chain with buffer ID `id`.
    pub(crate) fn add_chain(&mut self, id: u16, bufs: &[Buf]) {
        let descs: Vec<_> = bufs
            .iter()
            .enumerate()
            .map(|(i, &(addr, len, write))| {
                let mut flags = if write { VIRTQ_DESC_F_WRITE } else { 0 };
                if i + 1 < bufs.len() {
                    flags |= VIRTQ_DESC_F_NEXT;
                }
                (addr, len, id, flags)
            })
            .collect();
        self.slots_by_id.insert(id, bufs.len() as u16);
        self.add_raw(&descs);
    }

    /// Writes `bufs` as an indirect table at `table` and makes one ring descriptor point to it.
    pub(crate) fn add_indirect(&mut self, id: u16, table: u64, bufs: &[Buf]) {
        for (i, &(addr, len, write)) in bufs.iter().enumerate() {
            let at = table + 16 * i as u64;
            self.mem.write_u64(at, addr).unwrap();
            self.mem.write_u32(at + 8, len).unwrap();
            self.mem.write_u16(at + 12, 0).unwrap();
            self.mem.write_u16(at + 14, if write { VIRTQ_DESC_F_WRITE } else { 0 }).unwrap();
        }
        self.slots_by_id.insert(id, 1);
        self.add_raw(&[(table, 16 * bufs.len() as u32, id, VIRTQ_DESC_F_INDIRECT)]);
    }

    /// Takes the next used descriptor, if the device wrote one, as (id, len, flags).
    pub(crate) fn poll_used(&mut self) -> Option<(u16, u32, u16)> {
        let at = self.addrs.desc_table + 16 * u64::from(self.used_next);
        let flags = self.mem.read_u16(at + 14).unwrap();
        let avail = flags & VIRTQ_DESC_F_AVAIL != 0;
        let used = flags & VIRTQ_DESC_F_USED != 0;
        if avail != self.used_wrap || used != self.used_wrap {
            return None;
        }
        let id = self.mem.read_u16(at + 12).unwrap();
        let len = self.mem.read_u32(at + 8).unwrap();
        let slots = self.slots_by_id.remove(&id).expect("device returned an unknown id");
        self.used_next += slots;
        if self.used_next >= self.size {
            self.used_next -= self.size;
            self.used_wrap = !self.used_wrap;
        }
        Some((id, len, flags))
    }

    pub(crate) fn set_driver_event(&self, off_wrap: u16, flags: u16) {
        self.mem.write_u16(self.addrs.driver_area, off_wrap).unwrap();
        self.mem.write_u16(self.addrs.driver_area + 2, flags).unwrap();
    }

    pub(crate) fn enable_driver_event(&self) {
        self.set_driver_event(0, RING_EVENT_FLAGS_ENABLE);
    }

    /// The device event suppression structure as (off_wrap, flags).
    pub(crate) fn device_event(&self) -> (u16, u16) {
        (
            self.mem.read_u16(self.addrs.device_area).unwrap(),
            self.mem.read_u16(self.addrs.device_area + 2).unwrap(),
        )
    }
}
