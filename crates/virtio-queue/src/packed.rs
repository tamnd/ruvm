// SPDX-License-Identifier: MIT OR Apache-2.0

//! The packed virtqueue from section 2.8 of the specification.
//!
//! Guest memory holds one descriptor ring of `size` entries laid out as
//! `{ addr: u64, len: u32, id: u16, flags: u16 }`, plus two event suppression structures of the
//! form `{ off_wrap: u16, flags: u16 }`. The driver owns the driver area and reads the device area;
//! the device owns the device area and reads the driver area.
//!
//! Both sides keep a position in the ring and a wrap counter that starts at 1 and flips every time
//! the position goes past the end. A descriptor is available when its AVAIL bit equals the
//! driver's wrap counter and its USED bit does not. The device marks a descriptor used by setting
//! both bits to its own wrap counter.

use std::sync::atomic::{Ordering, fence};

use crate::chain::{ChainBuilder, DescriptorChain};
use crate::consts::{
    DESCRIPTOR_SIZE, RING_EVENT_FLAGS_DESC, RING_EVENT_FLAGS_DISABLE, RING_EVENT_FLAGS_ENABLE,
    VIRTQ_DESC_F_AVAIL, VIRTQ_DESC_F_INDIRECT, VIRTQ_DESC_F_NEXT, VIRTQ_DESC_F_USED,
    VIRTQ_DESC_F_WRITE,
};
use crate::error::QueueError;
use crate::memory::GuestMemory;
use crate::{DEFAULT_MAX_INDIRECT_LEN, MAX_QUEUE_SIZE, RingAddresses};

/// Bit 15 of an `off_wrap` value holds the wrap counter, the rest is the ring offset.
const WRAP_BIT: u16 = 1 << 15;

/// One descriptor as it sits in the packed ring or a packed indirect table.
#[derive(Clone, Copy, Debug)]
struct RawDesc {
    addr: u64,
    len: u32,
    id: u16,
    flags: u16,
}

impl RawDesc {
    fn parse(b: &[u8]) -> Self {
        let u64_at = |i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap_or_default());
        let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap_or_default());
        let u16_at = |i: usize| u16::from_le_bytes(b[i..i + 2].try_into().unwrap_or_default());
        Self { addr: u64_at(0), len: u32_at(8), id: u16_at(12), flags: u16_at(14) }
    }

    fn has(&self, flag: u16) -> bool {
        self.flags & flag != 0
    }
}

/// Whether descriptor flags mark it available for the given driver wrap counter.
fn is_available(flags: u16, wrap: bool) -> bool {
    let avail = flags & VIRTQ_DESC_F_AVAIL != 0;
    let used = flags & VIRTQ_DESC_F_USED != 0;
    avail == wrap && used != wrap
}

/// Device side state of one packed virtqueue.
///
/// Like [`SplitQueue`](crate::SplitQueue) it holds no reference to guest memory, and errors leave
/// its state untouched.
///
/// Chains are identified by the buffer ID the driver wrote into their last descriptor, not by
/// their position, and the device may return them in any order. To return one the device needs
/// its ID and the number of ring slots it took, both available on the
/// [`DescriptorChain`](crate::DescriptorChain).
#[derive(Clone, Debug)]
pub struct PackedQueue {
    size: u16,
    addrs: RingAddresses,
    next_avail: u16,
    avail_wrap: bool,
    next_used: u16,
    used_wrap: bool,
    event_idx: bool,
    indirect: bool,
    max_indirect_len: u32,
    notification_enabled: bool,
    last_checked_used: Option<u32>,
}

impl PackedQueue {
    /// Bytes needed for the descriptor ring of a queue with `size` entries.
    #[must_use]
    pub fn desc_ring_size(size: u16) -> u64 {
        DESCRIPTOR_SIZE * u64::from(size)
    }

    /// Bytes needed for one event suppression structure.
    pub const EVENT_SUPPRESSION_SIZE: u64 = 4;

    /// A queue of `size` entries at the given addresses.
    ///
    /// Any size from 1 to 32768 is allowed, the packed format does not need a power of two. The
    /// descriptor ring must be 16 byte aligned and both event suppression structures 4 byte
    /// aligned. Both wrap counters start at 1.
    pub fn new(size: u16, addrs: RingAddresses) -> Result<Self, QueueError> {
        if size == 0 || size > MAX_QUEUE_SIZE {
            return Err(QueueError::InvalidSize(size));
        }
        let checks = [
            ("descriptor table", addrs.desc_table, 16),
            ("driver area", addrs.driver_area, 4),
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
            avail_wrap: true,
            next_used: 0,
            used_wrap: true,
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

    /// Sets how many entries an indirect table may have.
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

    /// The ring offset the device will look at next for an available descriptor.
    #[must_use]
    pub fn next_avail(&self) -> u16 {
        self.next_avail
    }

    /// The wrap counter that goes with [`next_avail`](Self::next_avail).
    #[must_use]
    pub fn avail_wrap_counter(&self) -> bool {
        self.avail_wrap
    }

    /// The ring offset where the device will write its next used descriptor.
    #[must_use]
    pub fn next_used(&self) -> u16 {
        self.next_used
    }

    /// The wrap counter that goes with [`next_used`](Self::next_used).
    #[must_use]
    pub fn used_wrap_counter(&self) -> bool {
        self.used_wrap
    }

    /// Restores the available position and wrap counter, for migration or a vhost handover.
    pub fn set_avail_state(&mut self, offset: u16, wrap: bool) -> Result<(), QueueError> {
        if offset >= self.size {
            return Err(QueueError::InvalidRingOffset(offset));
        }
        self.next_avail = offset;
        self.avail_wrap = wrap;
        Ok(())
    }

    /// Restores the used position and wrap counter, for migration or a vhost handover.
    ///
    /// The notification history is forgotten, so the next
    /// [`needs_notification`](Self::needs_notification) errs on the side of notifying.
    pub fn set_used_state(&mut self, offset: u16, wrap: bool) -> Result<(), QueueError> {
        if offset >= self.size {
            return Err(QueueError::InvalidRingOffset(offset));
        }
        self.next_used = offset;
        self.used_wrap = wrap;
        self.last_checked_used = None;
        Ok(())
    }

    fn slot_addr(&self, offset: u16) -> u64 {
        self.addrs.desc_table + DESCRIPTOR_SIZE * u64::from(offset)
    }

    /// Moves a ring position forward by `n` slots, flipping the wrap counter when it passes the
    /// end. `n` is never more than the ring size.
    fn advance(&self, offset: u16, wrap: bool, n: u16) -> (u16, bool) {
        let next = u32::from(offset) + u32::from(n);
        let size = u32::from(self.size);
        if next >= size { ((next - size) as u16, !wrap) } else { (next as u16, wrap) }
    }

    /// A position and wrap counter folded into one number that counts modulo twice the ring size,
    /// so that positions on consecutive laps compare the way free running split indices do.
    fn linear(&self, offset: u16, wrap: bool) -> u32 {
        // The wrap counter starts at 1, so the first lap is the one with the bit set.
        u32::from(offset) + if wrap { 0 } else { u32::from(self.size) }
    }

    /// Whether the descriptor at the device's position is available.
    pub fn has_available<M: GuestMemory + ?Sized>(&self, mem: &M) -> Result<bool, QueueError> {
        let flags = mem.read_u16(self.slot_addr(self.next_avail) + 14)?;
        Ok(is_available(flags, self.avail_wrap))
    }

    /// Takes the next available buffer, or returns `None` if there is none.
    ///
    /// The chain is followed through consecutive ring slots while the next flag is set, wrapping
    /// at the end of the ring. It may not take more slots than the ring has, an indirect
    /// descriptor must be allowed, not carry the next flag and hold a whole number of descriptors,
    /// and device readable buffers must come before writable ones. Inside an indirect table only
    /// the write flag means anything, as the specification says.
    pub fn pop<M: GuestMemory + ?Sized>(
        &mut self,
        mem: &M,
    ) -> Result<Option<DescriptorChain>, QueueError> {
        if !self.has_available(mem)? {
            return Ok(None);
        }
        // The rest of the descriptor must not be read before the flags that made it available.
        fence(Ordering::Acquire);
        let mut builder = ChainBuilder::default();
        let (mut offset, mut wrap) = (self.next_avail, self.avail_wrap);
        let mut slots: u16 = 0;
        let id = loop {
            if slots == self.size {
                return Err(QueueError::ChainTooLong { limit: u32::from(self.size) });
            }
            let mut b = [0; DESCRIPTOR_SIZE as usize];
            mem.read(self.slot_addr(offset), &mut b)?;
            let desc = RawDesc::parse(&b);
            slots += 1;
            (offset, wrap) = self.advance(offset, wrap, 1);
            if desc.has(VIRTQ_DESC_F_INDIRECT) {
                if !self.indirect {
                    return Err(QueueError::IndirectNotNegotiated);
                }
                if desc.has(VIRTQ_DESC_F_NEXT) {
                    return Err(QueueError::IndirectWithNext);
                }
                self.walk_indirect(mem, desc, &mut builder)?;
                break desc.id;
            }
            builder.push(desc.addr, desc.len, desc.has(VIRTQ_DESC_F_WRITE))?;
            if !desc.has(VIRTQ_DESC_F_NEXT) {
                // The buffer ID lives in the last descriptor of the chain.
                break desc.id;
            }
        };
        self.next_avail = offset;
        self.avail_wrap = wrap;
        if self.event_idx && self.notification_enabled {
            self.write_device_event(mem, RING_EVENT_FLAGS_DESC)?;
        }
        Ok(Some(DescriptorChain::new(id, slots, builder)))
    }

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
        let mut table = vec![0; desc.len as usize];
        mem.read(desc.addr, &mut table)?;
        for entry in table.chunks_exact(DESCRIPTOR_SIZE as usize) {
            let d = RawDesc::parse(entry);
            builder.push(d.addr, d.len, d.has(VIRTQ_DESC_F_WRITE))?;
        }
        Ok(())
    }

    /// Marks the chain with buffer ID `id` used, with `len` bytes written into it.
    ///
    /// `ring_slots` is [`DescriptorChain::ring_slots`] of that chain: the used descriptor is
    /// written at the current used position and the position then skips over as many slots as the
    /// chain took. The ID and length are written before the flags, with a release fence between,
    /// because the flags are what hands the descriptor to the driver.
    pub fn add_used<M: GuestMemory + ?Sized>(
        &mut self,
        mem: &M,
        id: u16,
        len: u32,
        ring_slots: u16,
    ) -> Result<(), QueueError> {
        if ring_slots == 0 || ring_slots > self.size {
            return Err(QueueError::InvalidRingSlots(ring_slots));
        }
        let addr = self.slot_addr(self.next_used);
        mem.write_u32(addr + 8, len)?;
        mem.write_u16(addr + 12, id)?;
        let mut flags = if self.used_wrap { VIRTQ_DESC_F_AVAIL | VIRTQ_DESC_F_USED } else { 0 };
        // The length of a used descriptor only counts when the write flag is set.
        if len != 0 {
            flags |= VIRTQ_DESC_F_WRITE;
        }
        fence(Ordering::Release);
        mem.write_u16(addr + 14, flags)?;
        (self.next_used, self.used_wrap) = self.advance(self.next_used, self.used_wrap, ring_slots);
        Ok(())
    }

    /// Decides whether the driver should get a used buffer notification now.
    ///
    /// Reads the driver event suppression structure. `RING_EVENT_FLAGS_ENABLE` means yes and
    /// `RING_EVENT_FLAGS_DISABLE` means no. `RING_EVENT_FLAGS_DESC` with `VIRTIO_F_EVENT_IDX`
    /// means yes only if the used position passed the given offset and wrap counter since the
    /// previous call; without the feature the value is not valid and is treated as enable, as is
    /// the reserved value 3. The first call has nothing to compare with and says yes.
    pub fn needs_notification<M: GuestMemory + ?Sized>(
        &mut self,
        mem: &M,
    ) -> Result<bool, QueueError> {
        fence(Ordering::SeqCst);
        let off_wrap = mem.read_u16(self.addrs.driver_area)?;
        let flags = mem.read_u16(self.addrs.driver_area + 2)? & 3;
        let new = self.linear(self.next_used, self.used_wrap);
        let notify = match flags {
            RING_EVENT_FLAGS_DISABLE => false,
            RING_EVENT_FLAGS_DESC if self.event_idx => match self.last_checked_used {
                Some(old) => {
                    let modulus = 2 * u32::from(self.size);
                    let event = self.linear(off_wrap & !WRAP_BIT, off_wrap & WRAP_BIT != 0);
                    // The event is due if it lies in [old, new) counting modulo two laps.
                    let since_old = (event + modulus - old) % modulus;
                    let moved = (new + modulus - old) % modulus;
                    since_old < moved
                }
                None => true,
            },
            _ => true,
        };
        self.last_checked_used = Some(new);
        Ok(notify)
    }

    fn write_device_event<M: GuestMemory + ?Sized>(
        &self,
        mem: &M,
        flags: u16,
    ) -> Result<(), QueueError> {
        if flags == RING_EVENT_FLAGS_DESC {
            let off_wrap = self.next_avail | if self.avail_wrap { WRAP_BIT } else { 0 };
            mem.write_u16(self.addrs.device_area, off_wrap)?;
        }
        mem.write_u16(self.addrs.device_area + 2, flags)?;
        Ok(())
    }

    /// Asks the driver to notify the device about new buffers again.
    ///
    /// With `VIRTIO_F_EVENT_IDX` the device event suppression structure is set to
    /// `RING_EVENT_FLAGS_DESC` at the current position, otherwise to `RING_EVENT_FLAGS_ENABLE`.
    /// After a full fence the ring is checked once more. A `true` return means a buffer is already
    /// waiting that the driver may not notify for, and the device has to process it.
    #[must_use = "a true result means buffers arrived that no notification will announce"]
    pub fn enable_notification<M: GuestMemory + ?Sized>(
        &mut self,
        mem: &M,
    ) -> Result<bool, QueueError> {
        let flags = if self.event_idx { RING_EVENT_FLAGS_DESC } else { RING_EVENT_FLAGS_ENABLE };
        self.write_device_event(mem, flags)?;
        self.notification_enabled = true;
        fence(Ordering::SeqCst);
        self.has_available(mem)
    }

    /// Tells the driver it does not need to notify the device, by setting the device event
    /// suppression structure to `RING_EVENT_FLAGS_DISABLE`. It is a hint, the driver may still
    /// notify.
    pub fn disable_notification<M: GuestMemory + ?Sized>(
        &mut self,
        mem: &M,
    ) -> Result<(), QueueError> {
        self.write_device_event(mem, RING_EVENT_FLAGS_DISABLE)?;
        self.notification_enabled = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ADDRS: RingAddresses =
        RingAddresses { desc_table: 0x1000, driver_area: 0x2000, device_area: 0x3000 };

    #[test]
    fn availability_follows_the_wrap_counter() {
        let a = VIRTQ_DESC_F_AVAIL;
        let u = VIRTQ_DESC_F_USED;
        assert!(is_available(a, true));
        assert!(!is_available(a | u, true));
        assert!(!is_available(0, true));
        assert!(is_available(u, false));
        assert!(!is_available(0, false));
        assert!(!is_available(a | u, false));
    }

    #[test]
    fn advance_flips_at_the_end() {
        let q = PackedQueue::new(5, ADDRS).unwrap();
        assert_eq!(q.advance(3, true, 1), (4, true));
        assert_eq!(q.advance(4, true, 1), (0, false));
        assert_eq!(q.advance(3, false, 4), (2, true));
        assert_eq!(q.advance(0, true, 5), (0, false));
    }

    #[test]
    fn sizes_need_not_be_powers_of_two() {
        assert!(PackedQueue::new(3, ADDRS).is_ok());
        assert!(PackedQueue::new(0, ADDRS).is_err());
        assert!(PackedQueue::new(32769, ADDRS).is_err());
        let bad = RingAddresses { driver_area: 0x2002, ..ADDRS };
        assert!(matches!(PackedQueue::new(4, bad), Err(QueueError::Misaligned { .. })));
        assert_eq!(PackedQueue::desc_ring_size(3), 48);
    }

    #[test]
    fn state_setters_check_the_offset() {
        let mut q = PackedQueue::new(4, ADDRS).unwrap();
        assert_eq!(q.set_avail_state(4, true), Err(QueueError::InvalidRingOffset(4)));
        q.set_avail_state(3, false).unwrap();
        assert_eq!((q.next_avail(), q.avail_wrap_counter()), (3, false));
        q.set_used_state(2, false).unwrap();
        assert_eq!((q.next_used(), q.used_wrap_counter()), (2, false));
    }
}
