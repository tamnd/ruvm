// SPDX-License-Identifier: MIT OR Apache-2.0

//! Packed virtqueue tests, with a test driver on one side and `PackedQueue` on the other.

mod common;

use common::{DATA, PackedDriver};
use ruvm_virtio_queue::{
    Descriptor, GuestMemory, PackedQueue, QueueError, RING_EVENT_FLAGS_DESC,
    RING_EVENT_FLAGS_DISABLE, RING_EVENT_FLAGS_ENABLE, VIRTQ_DESC_F_AVAIL, VIRTQ_DESC_F_INDIRECT,
    VIRTQ_DESC_F_NEXT, VIRTQ_DESC_F_USED, VIRTQ_DESC_F_WRITE,
};

const WRAP: u16 = 1 << 15;

fn queue(driver: &PackedDriver) -> PackedQueue {
    PackedQueue::new(driver.size, driver.addrs).unwrap()
}

/// Pops one chain and returns it as used with `len` bytes written.
fn complete(q: &mut PackedQueue, d: &PackedDriver, len: u32) -> u16 {
    let c = q.pop(&d.mem).unwrap().unwrap();
    q.add_used(&d.mem, c.head(), len, c.ring_slots()).unwrap();
    c.head()
}

#[test]
fn request_and_response_round_trip() {
    let mut d = PackedDriver::new(8);
    let mut q = queue(&d);
    d.mem.write(DATA, b"ping").unwrap();
    d.add_chain(42, &[(DATA, 4, false), (DATA + 0x100, 2, true), (DATA + 0x200, 2, true)]);

    let c = q.pop(&d.mem).unwrap().unwrap();
    assert_eq!(c.head(), 42);
    assert_eq!(c.ring_slots(), 3);
    assert_eq!(c.readable(), &[Descriptor::new(DATA, 4, false)]);
    assert_eq!(c.writable_len(), 4);
    assert_eq!(c.reader(&d.mem).read_to_vec().unwrap(), b"ping");
    let mut w = c.writer(&d.mem);
    w.write_all(b"pong").unwrap();
    assert_eq!(d.poll_used(), None);
    q.add_used(&d.mem, c.head(), 4, c.ring_slots()).unwrap();

    let (id, len, flags) = d.poll_used().unwrap();
    assert_eq!((id, len), (42, 4));
    assert_eq!(flags, VIRTQ_DESC_F_AVAIL | VIRTQ_DESC_F_USED | VIRTQ_DESC_F_WRITE);
    assert_eq!(q.next_used(), 3);
    assert_eq!(q.next_avail(), 3);
    assert_eq!(q.pop(&d.mem).unwrap(), None);
    let mut out = [0; 4];
    d.mem.read(DATA + 0x100, &mut out[..2]).unwrap();
    d.mem.read(DATA + 0x200, &mut out[2..]).unwrap();
    assert_eq!(&out, b"pong");
}

#[test]
fn a_zero_length_used_descriptor_has_no_write_flag() {
    let mut d = PackedDriver::new(4);
    let mut q = queue(&d);
    d.add_chain(1, &[(DATA, 4, false)]);
    complete(&mut q, &d, 0);
    let (_, _, flags) = d.poll_used().unwrap();
    assert_eq!(flags, VIRTQ_DESC_F_AVAIL | VIRTQ_DESC_F_USED);
}

#[test]
fn wrap_counters_flip_at_the_end_of_the_ring() {
    // Five slots and two slot chains, so chains straddle the end of the ring on odd laps.
    let mut d = PackedDriver::new(5);
    let mut q = queue(&d);
    for round in 0..40u16 {
        d.add_chain(round, &[(DATA, 8, false), (DATA + 0x100, 8, true)]);
        let c = q.pop(&d.mem).unwrap().unwrap();
        assert_eq!(c.head(), round);
        assert_eq!(c.len(), 2);
        q.add_used(&d.mem, c.head(), 8, c.ring_slots()).unwrap();
        assert_eq!(d.poll_used().map(|u| u.0), Some(round));

        let pos = 2 * (u32::from(round) + 1);
        let expect = ((pos % 5) as u16, (pos / 5) % 2 == 0);
        assert_eq!((q.next_avail(), q.avail_wrap_counter()), expect, "round {round}");
        assert_eq!((q.next_used(), q.used_wrap_counter()), expect, "round {round}");
        assert_eq!(q.pop(&d.mem).unwrap(), None);
    }
}

#[test]
fn stale_descriptors_from_the_last_lap_are_not_available() {
    let mut d = PackedDriver::new(2);
    let mut q = queue(&d);
    d.add_chain(0, &[(DATA, 1, false)]);
    d.add_chain(1, &[(DATA, 1, false)]);
    complete(&mut q, &d, 0);
    complete(&mut q, &d, 0);
    assert!(!q.avail_wrap_counter());
    // Slot 0 still holds the used descriptor from the first lap.
    assert_eq!(q.pop(&d.mem).unwrap(), None);
    // A descriptor marked available for the first lap does not count on the second either.
    d.write_desc(0, DATA, 1, 7, VIRTQ_DESC_F_AVAIL);
    assert_eq!(q.pop(&d.mem).unwrap(), None);
    // Marked for the second lap it does.
    d.write_desc(0, DATA, 1, 7, VIRTQ_DESC_F_USED);
    assert_eq!(q.pop(&d.mem).unwrap().unwrap().head(), 7);
}

#[test]
fn buffers_can_complete_out_of_order() {
    let mut d = PackedDriver::new(8);
    let mut q = queue(&d);
    d.add_chain(10, &[(DATA, 1, true), (DATA, 1, true)]);
    d.add_chain(11, &[(DATA, 1, true)]);
    d.add_chain(12, &[(DATA, 1, true), (DATA, 1, true), (DATA, 1, true)]);
    let chains: Vec<_> = (0..3).map(|_| q.pop(&d.mem).unwrap().unwrap()).collect();
    for c in chains.iter().rev() {
        q.add_used(&d.mem, c.head(), 1, c.ring_slots()).unwrap();
    }
    assert_eq!(q.next_used(), 6);
    let ids: Vec<_> = std::iter::from_fn(|| d.poll_used()).map(|u| u.0).collect();
    assert_eq!(ids, [12, 11, 10]);
}

#[test]
fn indirect_tables() {
    let mut d = PackedDriver::new(4);
    let mut q = queue(&d);
    q.set_indirect_desc(true);
    d.mem.write(DATA, b"xyz").unwrap();
    d.add_indirect(5, DATA + 0x1000, &[(DATA, 3, false), (DATA + 0x10, 4, true)]);
    let c = q.pop(&d.mem).unwrap().unwrap();
    assert_eq!((c.head(), c.ring_slots(), c.len()), (5, 1, 2));
    assert_eq!(c.reader(&d.mem).read_to_vec().unwrap(), b"xyz");
    assert_eq!(c.writable(), &[Descriptor::new(DATA + 0x10, 4, true)]);
    q.add_used(&d.mem, 5, 4, c.ring_slots()).unwrap();
    assert_eq!(d.poll_used().map(|u| u.0), Some(5));
    assert_eq!(q.next_used(), 1);
}

#[test]
fn malformed_chains() {
    let table = DATA + 0x1000;

    let mut d = PackedDriver::new(4);
    let mut q = queue(&d);
    d.add_indirect(1, table, &[(DATA, 1, false)]);
    assert_eq!(q.pop(&d.mem), Err(QueueError::IndirectNotNegotiated));
    assert_eq!((q.next_avail(), q.avail_wrap_counter()), (0, true), "nothing consumed");

    let mut d = PackedDriver::new(4);
    let mut q = queue(&d);
    q.set_indirect_desc(true);
    d.add_raw(&[(table, 16, 1, VIRTQ_DESC_F_INDIRECT | VIRTQ_DESC_F_NEXT), (DATA, 1, 1, 0)]);
    assert_eq!(q.pop(&d.mem), Err(QueueError::IndirectWithNext));

    for len in [0, 24] {
        let mut d = PackedDriver::new(4);
        let mut q = queue(&d);
        q.set_indirect_desc(true);
        d.add_raw(&[(table, len, 1, VIRTQ_DESC_F_INDIRECT)]);
        assert_eq!(q.pop(&d.mem), Err(QueueError::InvalidIndirectLength(len)));
    }

    let mut d = PackedDriver::new(4);
    let mut q = queue(&d);
    q.set_indirect_desc(true);
    q.set_max_indirect_len(1);
    d.add_indirect(1, table, &[(DATA, 1, false), (DATA, 1, false)]);
    assert_eq!(q.pop(&d.mem), Err(QueueError::InvalidIndirectLength(32)));

    let mut d = PackedDriver::new(4);
    let mut q = queue(&d);
    q.set_indirect_desc(true);
    d.add_indirect(1, table, &[(DATA, 1, true), (DATA, 1, false)]);
    assert_eq!(q.pop(&d.mem), Err(QueueError::ReadableAfterWritable));

    let mut d = PackedDriver::new(4);
    let mut q = queue(&d);
    d.add_chain(1, &[(DATA, 1, true), (DATA, 1, false)]);
    assert_eq!(q.pop(&d.mem), Err(QueueError::ReadableAfterWritable));

    // Every slot says "next", so the chain never ends.
    let mut d = PackedDriver::new(4);
    let mut q = queue(&d);
    d.add_raw(&[(DATA, 1, 1, VIRTQ_DESC_F_NEXT); 4]);
    assert_eq!(q.pop(&d.mem), Err(QueueError::ChainTooLong { limit: 4 }));

    let mut d = PackedDriver::new(4);
    let mut q = queue(&d);
    d.add_chain(1, &[(u64::MAX, 2, false)]);
    assert!(matches!(q.pop(&d.mem), Err(QueueError::BufferOverflow { .. })));
}

#[test]
fn entries_in_an_indirect_table_only_honour_the_write_flag() {
    let mut d = PackedDriver::new(4);
    let mut q = queue(&d);
    q.set_indirect_desc(true);
    let table = DATA + 0x1000;
    d.add_indirect(3, table, &[(DATA, 1, false), (DATA, 2, true)]);
    // Stray flags in the table are reserved and ignored.
    d.mem.write_u16(table + 14, VIRTQ_DESC_F_INDIRECT | VIRTQ_DESC_F_NEXT).unwrap();
    let c = q.pop(&d.mem).unwrap().unwrap();
    assert_eq!(c.descriptors(), &[Descriptor::new(DATA, 1, false), Descriptor::new(DATA, 2, true)]);
}

#[test]
fn add_used_checks_the_slot_count() {
    let d = PackedDriver::new(4);
    let mut q = queue(&d);
    assert_eq!(q.add_used(&d.mem, 0, 0, 0), Err(QueueError::InvalidRingSlots(0)));
    assert_eq!(q.add_used(&d.mem, 0, 0, 5), Err(QueueError::InvalidRingSlots(5)));
    assert_eq!(q.next_used(), 0);
}

#[test]
fn driver_event_flags() {
    let mut d = PackedDriver::new(4);
    let mut q = queue(&d);
    for id in 0..3 {
        d.add_chain(id, &[(DATA, 1, true)]);
    }

    d.enable_driver_event();
    complete(&mut q, &d, 1);
    assert!(q.needs_notification(&d.mem).unwrap());

    d.set_driver_event(0, RING_EVENT_FLAGS_DISABLE);
    complete(&mut q, &d, 1);
    assert!(!q.needs_notification(&d.mem).unwrap());

    // Without the event index feature a descriptor event is treated like enable.
    d.set_driver_event(3 | WRAP, RING_EVENT_FLAGS_DESC);
    complete(&mut q, &d, 1);
    assert!(q.needs_notification(&d.mem).unwrap());
}

#[test]
fn driver_event_offset_and_wrap() {
    let mut d = PackedDriver::new(4);
    let mut q = queue(&d);
    q.set_event_idx(true);
    for id in 0..5 {
        d.add_chain(id, &[(DATA, 1, true)]);
        complete(&mut q, &d, 1);
        d.poll_used().unwrap();
        if id == 0 {
            // The first check has nothing to compare with.
            assert!(q.needs_notification(&d.mem).unwrap());
            // Ask to hear about slot 2 on the first lap.
            d.set_driver_event(2 | WRAP, RING_EVENT_FLAGS_DESC);
        }
        let expect = match id {
            0 => continue,
            // Used position moved 1 -> 2, slot 2 not written yet.
            1 => false,
            // 2 -> 3 writes slot 2.
            2 => true,
            // 3 -> 0 on the next lap.
            3 => false,
            // 0 -> 1 on the second lap: slot 2 of the first lap is long gone.
            _ => false,
        };
        assert_eq!(q.needs_notification(&d.mem).unwrap(), expect, "chain {id}");
    }
    assert_eq!((q.next_used(), q.used_wrap_counter()), (1, false));

    // Ask for slot 3 on the second lap, then return a three slot chain from 1 that jumps over it
    // and lands at slot 0 of the third lap.
    d.set_driver_event(3, RING_EVENT_FLAGS_DESC);
    d.add_chain(9, &[(DATA, 1, true), (DATA, 1, true), (DATA, 1, true)]);
    complete(&mut q, &d, 1);
    assert_eq!((q.next_used(), q.used_wrap_counter()), (0, true));
    assert!(q.needs_notification(&d.mem).unwrap());

    // An event that belongs to the previous lap does not fire again.
    d.set_driver_event(3, RING_EVENT_FLAGS_DESC);
    d.add_chain(10, &[(DATA, 1, true)]);
    complete(&mut q, &d, 1);
    assert!(!q.needs_notification(&d.mem).unwrap());
}

#[test]
fn device_event_without_event_idx() {
    let mut d = PackedDriver::new(4);
    let mut q = queue(&d);
    q.disable_notification(&d.mem).unwrap();
    assert_eq!(d.device_event().1, RING_EVENT_FLAGS_DISABLE);
    assert!(!q.enable_notification(&d.mem).unwrap());
    assert_eq!(d.device_event().1, RING_EVENT_FLAGS_ENABLE);

    q.disable_notification(&d.mem).unwrap();
    d.add_chain(0, &[(DATA, 1, false)]);
    assert!(q.enable_notification(&d.mem).unwrap(), "the re-check sees the new buffer");
}

#[test]
fn device_event_with_event_idx() {
    let mut d = PackedDriver::new(2);
    let mut q = queue(&d);
    q.set_event_idx(true);
    assert!(!q.enable_notification(&d.mem).unwrap());
    assert_eq!(d.device_event(), (WRAP, RING_EVENT_FLAGS_DESC));

    d.add_chain(0, &[(DATA, 1, false)]);
    d.add_chain(1, &[(DATA, 1, false)]);
    complete(&mut q, &d, 0);
    assert_eq!(d.device_event(), (1 | WRAP, RING_EVENT_FLAGS_DESC));
    complete(&mut q, &d, 0);
    // The device's position wrapped, and so did the wrap bit it publishes.
    assert_eq!(d.device_event(), (0, RING_EVENT_FLAGS_DESC));

    q.disable_notification(&d.mem).unwrap();
    assert_eq!(d.device_event().1, RING_EVENT_FLAGS_DISABLE);
    d.poll_used().unwrap();
    d.add_chain(2, &[(DATA, 1, false)]);
    complete(&mut q, &d, 0);
    // Disabled, so popping leaves the structure alone.
    assert_eq!(d.device_event(), (0, RING_EVENT_FLAGS_DISABLE));
    assert!(!q.enable_notification(&d.mem).unwrap());
    assert_eq!(d.device_event(), (1, RING_EVENT_FLAGS_DESC));
}

#[test]
fn restored_state_continues_where_it_left_off() {
    let mut d = PackedDriver::new(4);
    let mut q = queue(&d);
    for id in 0..3 {
        d.add_chain(id, &[(DATA, 1, false)]);
    }
    complete(&mut q, &d, 0);
    let mut restored = queue(&d);
    restored.set_avail_state(q.next_avail(), q.avail_wrap_counter()).unwrap();
    restored.set_used_state(q.next_used(), q.used_wrap_counter()).unwrap();
    assert_eq!(complete(&mut restored, &d, 0), 1);
    assert_eq!(complete(&mut restored, &d, 0), 2);
    let ids: Vec<_> = std::iter::from_fn(|| d.poll_used()).map(|u| u.0).collect();
    assert_eq!(ids, [0, 1, 2]);
}
