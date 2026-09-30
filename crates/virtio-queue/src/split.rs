// SPDX-License-Identifier: MIT OR Apache-2.0

//! The split virtqueue from section 2.7 of the specification.
//!
//! Guest memory holds three areas:
//!
//! - the descriptor table, `size` entries of `{ addr: u64, len: u32, flags: u16, next: u16 }`;
//! - the available ring, `{ flags: u16, idx: u16, ring: [u16; size], used_event: u16 }`;
//! - the used ring, `{ flags: u16, idx: u16, ring: [{ id: u32, len: u32 }; size],
//!   avail_event: u16 }`.
//!
//! The driver writes the first two and the device writes the third.

use std::sync::atomic::{Ordering, fence};

use crate::chain::{ChainBuilder, DescriptorChain};
use crate::consts::{
    DESCRIPTOR_SIZE, VIRTQ_AVAIL_F_NO_INTERRUPT, VIRTQ_DESC_F_INDIRECT, VIRTQ_DESC_F_NEXT,
    VIRTQ_DESC_F_WRITE, VIRTQ_USED_F_NO_NOTIFY,
};
use crate::error::QueueError;
use crate::event::need_event;
use crate::memory::GuestMemory;
use crate::{DEFAULT_MAX_INDIRECT_LEN, MAX_QUEUE_SIZE, RingAddresses};

/// One descriptor as it sits in a split descriptor table.
#[derive(Clone, Copy, Debug)]
struct RawDesc {
    addr: u64,
    len: u32,
    flags: u16,
    next: u16,
}

impl RawDesc {
    fn parse(b: &[u8]) -> Self {
        let u64_at = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap_or_default());
        let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap_or_default());
        let u16_at = |i: usize| u16::from_le_bytes(b[i..i + 2].try_into().unwrap_or_default());
        Self { addr: u64_at(0), len: u32_at(8), flags: u16_at(12), next: u16_at(14) }
    }

    fn read<M: GuestMemory + ?Sized>(mem: &M, addr: u64) -> Result<Self, QueueError> {
        let mut b = [0; DESCRIPTOR_SIZE as usize];
        mem.read(addr, &mut b)?;
        Ok(Self::parse(&b))
    }

    fn has(&self, flag: u16) -> bool {
        self.flags & flag != 0
    }
}

/// Device side state of one split virtqueue.
///
/// The queue does not hold on to guest memory. Every method that touches the rings takes it as an
/// argument, which keeps the queue free of lifetimes and lets the caller decide how memory is
/// shared.
///
/// Errors leave the queue as it was, so a device that gets one can report `DEVICE_NEEDS_RESET`
/// without having consumed half a chain.
#[derive(Clone, Debug)]
pub struct SplitQueue {
    size: u16,
    addrs: RingAddresses,
    next_avail: u16,
    next_used: u16,
    event_idx: bool,
    indirect: bool,
    max_indirect_len: u32,
    notification_enabled: bool,
    last_checked_used: Option<u16>,
}

impl SplitQueue {
    /// Bytes needed for the descriptor table of a queue with `size` entries.
    #[must_use]
    pub fn desc_table_size(size: u16) -> u64 {
        DESCRIPTOR_SIZE * u64::from(size)
    }

    /// Bytes needed for the available ring, including `used_event`.
    #[must_use]
    pub fn avail_ring_size(size: u16) -> u64 {
        6 + 2 * u64::from(size)
    }

    /// Bytes needed for the used ring, including `avail_event`.
    #[must_use]
    pub fn used_ring_size(size: u16) -> u64 {
        6 + 8 * u64::from(size)
    }

    /// A queue of `size` entries at the given addresses.
    ///
    /// The size must be a power of two no larger than 32768. The descriptor table must be 16 byte
    /// aligned, the available ring 2 byte aligned and the used ring 4 byte aligned (section 2.7).
    /// Event index and indirect descriptors start off and are turned on to match the negotiated
    /// features.
    pub fn new(size: u16, addrs: RingAddresses) -> Result<Self, QueueError> {
        if size == 0 || size > MAX_QUEUE_SIZE || !size.is_power_of_two() {
            return Err(QueueError::InvalidSize(size));
        }
        let checks = [
            ("descriptor table", addrs.desc_table, 16),
            ("driver area", addrs.driver_area, 2),
            ("device area", addrs.device_area, 4),
        ];
        for (area, addr, align) in checks {
            if addr % align != 0 {
                return Err(QueueError::Misaligned { area, addr, align });
            }
        }
        Ok(Self {
            size,
            addrs,
            next_avail: 0,
            next_used: 0,
            event_idx: false,
            indirect: false,
            max_indirect_len: DEFAULT_MAX_INDIRECT_LEN,
            notification_enabled: true,
            last_checked_used: None,
        })
    }

    /// Turns `VIRTIO_F_EVENT_IDX` handling on or off.
    pub fn set_event_idx(&mut self, on: bool) {
        self.event_idx = on;
    }

    /// Allows or forbids indirect descriptors, following `VIRTIO_F_INDIRECT_DESC`.
    pub fn set_indirect_desc(&mut self, on: bool) {
        self.indirect = on;
    }

    /// Sets how many entries an indirect table may have. The default is
    /// [`DEFAULT_MAX_INDIRECT_LEN`](crate::DEFAULT_MAX_INDIRECT_LEN).
    pub fn set_max_indirect_len(&mut self, len: u32) {
        self.max_indirect_len = len;
    }

    /// The queue size.
    #[must_use]
    pub fn size(&self) -> u16 {
        self.size
    }

    /// The ring addresses the queue was created with.
    #[must_use]
    pub fn addresses(&self) -> RingAddresses {
        self.addrs
    }

    /// The free running index of the next available ring entry the device will read.
    #[must_use]
    pub fn next_avail(&self) -> u16 {
        self.next_avail
    }

    /// The free running index the device will publish in the used ring's `idx` next.
    #[must_use]
    pub fn next_used(&self) -> u16 {
        self.next_used
    }

    /// Restores the available position, for migration or a vhost handover.
    pub fn set_next_avail(&mut self, idx: u16) {
        self.next_avail = idx;
    }

    /// Restores the used position, for migration or a vhost handover.
    ///
    /// The notification history is forgotten, so the next
    /// [`needs_notification`](Self::needs_notification) errs on the side of notifying.
    pub fn set_next_used(&mut self, idx: u16) {
        self.next_used = idx;
        self.last_checked_used = None;
    }

    fn used_event_addr(&self) -> u64 {
        self.addrs.driver_area + 4 + 2 * u64::from(self.size)
    }

    fn avail_event_addr(&self) -> u64 {
        self.addrs.device_area + 4 + 8 * u64::from(self.size)
    }

    /// Reads the driver's available index.
    pub fn avail_idx<M: GuestMemory + ?Sized>(&self, mem: &M) -> Result<u16, QueueError> {
        Ok(mem.read_u16(self.addrs.driver_area + 2)?)
    }

    /// Whether the driver has made buffers available that the device has not taken yet.
    pub fn has_available<M: GuestMemory + ?Sized>(&self, mem: &M) -> Result<bool, QueueError> {
        Ok(self.avail_idx(mem)? != self.next_avail)
    }

    /// Takes the next available buffer, or returns `None` if there is none.
    ///
    /// The chain is walked and checked completely before anything changes: indices must stay in
    /// the table, the chain may not be longer than the queue (which also catches loops), indirect
    /// tables must be allowed, a whole number of descriptors, not nested and last in the chain, and
    /// device readable buffers must come before device writable ones.
    ///
    /// With `VIRTIO_F_EVENT_IDX` and notifications enabled, `avail_event` is moved along so the
    /// driver keeps notifying for new buffers.
    pub fn pop<M: GuestMemory + ?Sized>(
        &mut self,
        mem: &M,
    ) -> Result<Option<DescriptorChain>, QueueError> {
        let avail_idx = self.avail_idx(mem)?;
        let pending = avail_idx.wrapping_sub(self.next_avail);
        if pending > self.size {
            return Err(QueueError::AvailIndexTooFar { next_avail: self.next_avail, avail_idx });
        }
        if pending == 0 {
            return Ok(None);
        }
        // The ring entries must not be read before the index that covers them.
        fence(Ordering::Acquire);
        let slot = self.next_avail % self.size;
        let head = mem.read_u16(self.addrs.driver_area + 4 + 2 * u64::from(slot))?;
        let chain = self.walk(mem, head)?;
        self.next_avail = self.next_avail.wrapping_add(1);
        if self.event_idx && self.notification_enabled {
            mem.write_u16(self.avail_event_addr(), self.next_avail)?;
        }
        Ok(Some(chain))
    }

    /// Gives back the last chain taken by [`pop`](Self::pop), so the next `pop` returns it again.
    ///
    /// Useful for a device that took a buffer and then found it cannot use it yet.
    pub fn undo_pop(&mut self) {
        self.next_avail = self.next_avail.wrapping_sub(1);
    }

    fn walk<M: GuestMemory + ?Sized>(
        &self,
        mem: &M,
        head: u16,
    ) -> Result<DescriptorChain, QueueError> {
        let table_len = u32::from(self.size);
        let mut builder = ChainBuilder::default();
        let mut index = head;
        let mut count: u16 = 0;
        loop {
            if u32::from(index) >= table_len {
                return Err(QueueError::DescriptorIndexOutOfRange { index, table_len });
            }
            if count == self.size {
                return Err(QueueError::ChainTooLong { limit: table_len });
            }
            count += 1;
            let desc =
                RawDesc::read(mem, self.addrs.desc_table + DESCRIPTOR_SIZE * u64::from(index))?;
            if desc.has(VIRTQ_DESC_F_INDIRECT) {
                if !self.indirect {
                    return Err(QueueError::IndirectNotNegotiated);
                }
                if desc.has(VIRTQ_DESC_F_NEXT) {
                    return Err(QueueError::IndirectWithNext);
                }
                self.walk_indirect(mem, desc, &mut builder)?;
                break;
            }
            builder.push(desc.addr, desc.len, desc.has(VIRTQ_DESC_F_WRITE))?;
            if !desc.has(VIRTQ_DESC_F_NEXT) {
                break;
            }
            index = desc.next;
        }
        Ok(DescriptorChain::new(head, count, builder))
    }

    /// Follows an indirect table. The write flag of the descriptor that points at the table is
    /// ignored, as the specification requires.
    fn walk_indirect<M: GuestMemory + ?Sized>(
        &self,
        mem: &M,
        desc: RawDesc,
        builder: &mut ChainBuilder,
    ) -> Result<(), QueueError> {
        let table_len = desc.len / DESCRIPTOR_SIZE as u32;
        if desc.len == 0
            || u64::from(desc.len) % DESCRIPTOR_SIZE != 0
            || table_len > self.max_indirect_len
        {
            return Err(QueueError::InvalidIndirectLength(desc.len));
        }
        // One bulk read for the whole table, since the driver may not change it while the buffer
        // is available.
        let mut table = vec![0; desc.len as usize];
        mem.read(desc.addr, &mut table)?;
        let mut index: u16 = 0;
        let mut count: u32 = 0;
        loop {
            if u32::from(index) >= table_len {
                return Err(QueueError::DescriptorIndexOutOfRange { index, table_len });
            }
            if count == table_len {
                return Err(QueueError::ChainTooLong { limit: table_len });
            }
            count += 1;
            let start = usize::from(index) * DESCRIPTOR_SIZE as usize;
            let d = RawDesc::parse(&table[start..start + DESCRIPTOR_SIZE as usize]);
            if d.has(VIRTQ_DESC_F_INDIRECT) {
                return Err(QueueError::NestedIndirect);
            }
            builder.push(d.addr, d.len, d.has(VIRTQ_DESC_F_WRITE))?;
            if !d.has(VIRTQ_DESC_F_NEXT) {
                return Ok(());
            }
            index = d.next;
        }
    }

    /// Returns the chain whose head is `head` to the driver, telling it `len` bytes were written.
    ///
    /// The used element is written first and the used index after it, with a release fence in
    /// between, so the driver never sees an index covering an element that is not there yet.
    pub fn add_used<M: GuestMemory + ?Sized>(
        &mut self,
        mem: &M,
        head: u16,
        len: u32,
    ) -> Result<(), QueueError> {
        if head >= self.size {
            return Err(QueueError::InvalidUsedHead { head, size: self.size });
        }
        let slot = self.next_used % self.size;
        let elem = self.addrs.device_area + 4 + 8 * u64::from(slot);
        mem.write_u32(elem, u32::from(head))?;
        mem.write_u32(elem + 4, len)?;
        let next = self.next_used.wrapping_add(1);
        fence(Ordering::Release);
        mem.write_u16(self.addrs.device_area + 2, next)?;
        self.next_used = next;
        Ok(())
    }

    /// Decides whether the driver should get a used buffer notification now.
    ///
    /// Call it after one or more [`add_used`](Self::add_used) calls. Without
    /// `VIRTIO_F_EVENT_IDX` the answer is whether the driver left `VIRTQ_AVAIL_F_NO_INTERRUPT`
    /// clear. With it, the answer is whether the used index moved past `used_event` since the
    /// previous call. The first call after creation or
    /// [`set_next_used`](Self::set_next_used) has no previous position to compare with, so it
    /// always says yes.
    pub fn needs_notification<M: GuestMemory + ?Sized>(
        &mut self,
        mem: &M,
    ) -> Result<bool, QueueError> {
        // The used index store has to be visible before we look at what the driver asked for,
        // otherwise both sides can decide the other one is not waiting.
        fence(Ordering::SeqCst);
        let new = self.next_used;
        let notify = if self.event_idx {
            let used_event = mem.read_u16(self.used_event_addr())?;
            match self.last_checked_used {
                Some(old) => need_event(used_event, new, old),
                None => true,
            }
        } else {
            mem.read_u16(self.addrs.driver_area)? & VIRTQ_AVAIL_F_NO_INTERRUPT == 0
        };
        self.last_checked_used = Some(new);
        Ok(notify)
    }

    /// Asks the driver to notify the device about new buffers again.
    ///
    /// With `VIRTIO_F_EVENT_IDX` this publishes the current position in `avail_event`, otherwise
    /// it clears `VIRTQ_USED_F_NO_NOTIFY`. It then issues a full fence and checks the available
    /// ring once more. A `true` return means the driver added buffers before it could have seen
    /// the change, so it may not notify for them, and the device has to go and process them. A
    /// device that ignores the return value can sleep with work pending.
    #[must_use = "a true result means buffers arrived that no notification will announce"]
    pub fn enable_notification<M: GuestMemory + ?Sized>(
        &mut self,
        mem: &M,
    ) -> Result<bool, QueueError> {
        if self.event_idx {
            mem.write_u16(self.avail_event_addr(), self.next_avail)?;
        } else {
            mem.write_u16(self.addrs.device_area, 0)?;
        }
        self.notification_enabled = true;
        fence(Ordering::SeqCst);
        self.has_available(mem)
    }

    /// Tells the driver it does not need to notify the device, for example while the device is
    /// busy draining the queue anyway.
    ///
    /// Without `VIRTIO_F_EVENT_IDX` this sets `VIRTQ_USED_F_NO_NOTIFY`. With it, `avail_event` is
    /// simply left where it is, so the driver stops notifying once it has moved past it. Either
    /// way this is only a hint and the driver may still notify.
    pub fn disable_notification<M: GuestMemory + ?Sized>(
        &mut self,
        mem: &M,
    ) -> Result<(), QueueError> {
        if !self.event_idx {
            mem.write_u16(self.addrs.device_area, VIRTQ_USED_F_NO_NOTIFY)?;
        }
        self.notification_enabled = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::VecMemory;

    const ADDRS: RingAddresses =
        RingAddresses { desc_table: 0x1000, driver_area: 0x2000, device_area: 0x3000 };

    #[test]
    fn sizes_and_alignment_are_checked() {
        assert!(SplitQueue::new(0, ADDRS).is_err());
        assert_eq!(SplitQueue::new(3, ADDRS).unwrap_err(), QueueError::InvalidSize(3));
        assert!(SplitQueue::new(32768, ADDRS).is_ok());
        let bad = RingAddresses { desc_table: 0x1008, ..ADDRS };
        assert!(matches!(
            SplitQueue::new(8, bad),
            Err(QueueError::Misaligned { area: "descriptor table", .. })
        ));
        let bad = RingAddresses { driver_area: 0x2001, ..ADDRS };
        assert!(matches!(SplitQueue::new(8, bad), Err(QueueError::Misaligned { align: 2, .. })));
        let bad = RingAddresses { device_area: 0x3002, ..ADDRS };
        assert!(matches!(SplitQueue::new(8, bad), Err(QueueError::Misaligned { align: 4, .. })));
    }

    #[test]
    fn layout_sizes() {
        assert_eq!(SplitQueue::desc_table_size(256), 4096);
        assert_eq!(SplitQueue::avail_ring_size(256), 518);
        assert_eq!(SplitQueue::used_ring_size(256), 2054);
    }

    #[test]
    fn empty_queue_pops_nothing() {
        let mem = VecMemory::new(0x4000);
        let mut q = SplitQueue::new(8, ADDRS).unwrap();
        assert_eq!(q.pop(&mem).unwrap(), None);
        assert!(!q.has_available(&mem).unwrap());
    }
}
