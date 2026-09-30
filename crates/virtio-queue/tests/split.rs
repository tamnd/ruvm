// SPDX-License-Identifier: MIT OR Apache-2.0

//! Split virtqueue tests, with a test driver on one side and `SplitQueue` on the other.

mod common;

use common::{DATA, SplitDriver};
use ruvm_virtio_queue::{
    Descriptor, GuestMemory, QueueError, SplitQueue, VIRTQ_AVAIL_F_NO_INTERRUPT,
    VIRTQ_DESC_F_INDIRECT, VIRTQ_DESC_F_NEXT, VIRTQ_DESC_F_WRITE, VIRTQ_USED_F_NO_NOTIFY,
};

fn queue(driver: &SplitDriver) -> SplitQueue {
    SplitQueue::new(driver.size, driver.addrs).unwrap()
}

#[test]
fn request_and_response_round_trip() {
    let mut d = SplitDriver::new(8);
    let mut q = queue(&d);
    d.mem.write(DATA, b"hello ").unwrap();
    d.mem.write(DATA + 0x100, b"world").unwrap();
    d.add_chain(
        2,
        &[
            (DATA, 6, false),
            (DATA + 0x100, 5, false),
            (DATA + 0x200, 4, true),
            (DATA + 0x300, 8, true),
        ],
    );

    let chain = q.pop(&d.mem).unwrap().unwrap();
    assert_eq!(chain.head(), 2);
    assert_eq!(chain.len(), 4);
    assert_eq!(chain.ring_slots(), 4);
    assert_eq!(chain.readable().len(), 2);
    assert_eq!(chain.writable()[1], Descriptor::new(DATA + 0x300, 8, true));
    assert_eq!(chain.readable_len(), 11);
    assert_eq!(chain.writable_len(), 12);
    assert_eq!(chain.reader(&d.mem).read_to_vec().unwrap(), b"hello world");

    let mut w = chain.writer(&d.mem);
    w.write_all(b"response").unwrap();
    q.add_used(&d.mem, chain.head(), w.bytes_written() as u32).unwrap();

    let mut out = [0; 8];
    d.mem.read(DATA + 0x200, &mut out[..4]).unwrap();
    d.mem.read(DATA + 0x300, &mut out[4..]).unwrap();
    assert_eq!(&out, b"response");
    assert_eq!(d.used_idx(), 1);
    assert_eq!(d.used_elem(0), (2, 8));
    assert_eq!(q.pop(&d.mem).unwrap(), None);
}

#[test]
fn chains_can_be_used_out_of_order() {
    let mut d = SplitDriver::new(4);
    let mut q = queue(&d);
    for head in 0..4 {
        d.add_chain(head, &[(DATA + 0x10 * u64::from(head), 16, true)]);
    }
    let chains: Vec<_> = (0..4).map(|_| q.pop(&d.mem).unwrap().unwrap()).collect();
    assert_eq!(chains.iter().map(|c| c.head()).collect::<Vec<_>>(), [0, 1, 2, 3]);
    for c in chains.iter().rev() {
        q.add_used(&d.mem, c.head(), u32::from(c.head()) * 10).unwrap();
    }
    assert_eq!(d.used_idx(), 4);
    assert_eq!(
        (0..4).map(|i| d.used_elem(i)).collect::<Vec<_>>(),
        [(3, 30), (2, 20), (1, 10), (0, 0)]
    );
}

#[test]
fn indices_wrap_at_16_bits() {
    let mut d = SplitDriver::new(4);
    let mut q = queue(&d);
    d.set_avail_idx(0xfffe);
    q.set_next_avail(0xfffe);
    q.set_next_used(0xfffd);
    for i in 0..4u16 {
        d.add_chain(i, &[(DATA, 1, false)]);
    }
    assert_eq!(d.avail_idx, 2);
    for i in 0..4u16 {
        let c = q.pop(&d.mem).unwrap().unwrap();
        assert_eq!(c.head(), i);
        q.add_used(&d.mem, c.head(), 0).unwrap();
    }
    assert_eq!(q.next_avail(), 2);
    assert_eq!(q.next_used(), 1);
    assert_eq!(d.used_idx(), 1);
    assert_eq!(d.used_elem(0xffff), (2, 0));
    assert_eq!(q.pop(&d.mem).unwrap(), None);
}

#[test]
fn a_long_run_wraps_the_indices_naturally() {
    let mut d = SplitDriver::new(2);
    let mut q = queue(&d);
    for round in 0..70_000u32 {
        let head = (round % 2) as u16;
        d.add_chain(head, &[(DATA, 4, true)]);
        let c = q.pop(&d.mem).unwrap().unwrap();
        assert_eq!(c.head(), head);
        q.add_used(&d.mem, head, 4).unwrap();
    }
    assert_eq!(q.next_avail(), (70_000u32 % 65536) as u16);
    assert_eq!(d.used_idx(), q.next_used());
}

#[test]
fn avail_index_may_not_run_more_than_a_queue_ahead() {
    let mut d = SplitDriver::new(4);
    let mut q = queue(&d);
    d.set_avail_idx(4);
    d.write_chain(0, &[(DATA, 1, false)]);
    assert!(q.pop(&d.mem).unwrap().is_some());
    d.set_avail_idx(9);
    assert_eq!(q.pop(&d.mem), Err(QueueError::AvailIndexTooFar { next_avail: 1, avail_idx: 9 }));
    // Going backwards looks like almost 65536 new entries and is refused the same way.
    d.set_avail_idx(0);
    assert!(matches!(q.pop(&d.mem), Err(QueueError::AvailIndexTooFar { .. })));
    assert_eq!(q.next_avail(), 1);
}

#[test]
fn head_and_next_must_stay_in_the_table() {
    let mut d = SplitDriver::new(4);
    let mut q = queue(&d);
    d.publish(4);
    assert_eq!(
        q.pop(&d.mem),
        Err(QueueError::DescriptorIndexOutOfRange { index: 4, table_len: 4 })
    );
    assert_eq!(q.next_avail(), 0, "a failed pop consumes nothing");

    let mut d = SplitDriver::new(4);
    let mut q = queue(&d);
    d.write_desc(0, DATA, 1, VIRTQ_DESC_F_NEXT, 7);
    d.publish(0);
    assert_eq!(
        q.pop(&d.mem),
        Err(QueueError::DescriptorIndexOutOfRange { index: 7, table_len: 4 })
    );
}

#[test]
fn loops_are_caught() {
    let mut d = SplitDriver::new(4);
    let mut q = queue(&d);
    d.write_desc(0, DATA, 1, VIRTQ_DESC_F_NEXT, 1);
    d.write_desc(1, DATA, 1, VIRTQ_DESC_F_NEXT, 0);
    d.publish(0);
    assert_eq!(q.pop(&d.mem), Err(QueueError::ChainTooLong { limit: 4 }));

    // A descriptor pointing at itself.
    let mut d = SplitDriver::new(4);
    let mut q = queue(&d);
    d.write_desc(3, DATA, 1, VIRTQ_DESC_F_NEXT, 3);
    d.publish(3);
    assert_eq!(q.pop(&d.mem), Err(QueueError::ChainTooLong { limit: 4 }));

    // A chain using every entry exactly once is fine.
    let mut d = SplitDriver::new(4);
    let mut q = queue(&d);
    d.add_chain(0, &[(DATA, 1, false); 4]);
    assert_eq!(q.pop(&d.mem).unwrap().unwrap().len(), 4);
}

#[test]
fn readable_after_writable_is_refused() {
    let mut d = SplitDriver::new(4);
    let mut q = queue(&d);
    d.add_chain(0, &[(DATA, 1, false), (DATA, 1, true), (DATA, 1, false)]);
    assert_eq!(q.pop(&d.mem), Err(QueueError::ReadableAfterWritable));
}

#[test]
fn buffers_may_not_wrap_the_address_space() {
    let mut d = SplitDriver::new(4);
    let mut q = queue(&d);
    d.add_chain(0, &[(u64::MAX - 1, 8, false)]);
    assert_eq!(q.pop(&d.mem), Err(QueueError::BufferOverflow { addr: u64::MAX - 1, len: 8 }));
}

#[test]
fn memory_faults_are_reported() {
    let d = SplitDriver::new(4);
    let addrs = ruvm_virtio_queue::RingAddresses { driver_area: 0x100_0000, ..d.addrs };
    let mut q = SplitQueue::new(4, addrs).unwrap();
    assert!(matches!(q.pop(&d.mem), Err(QueueError::Memory(_))));

    // A buffer that is outside memory only fails when it is read.
    let mut d = SplitDriver::new(4);
    let mut q = queue(&d);
    d.add_chain(0, &[(0x1000_0000, 4, false)]);
    let c = q.pop(&d.mem).unwrap().unwrap();
    let mut buf = [0; 4];
    assert!(matches!(c.reader(&d.mem).read(&mut buf), Err(QueueError::Memory(_))));
}

#[test]
fn indirect_tables() {
    let mut d = SplitDriver::new(4);
    let mut q = queue(&d);
    q.set_indirect_desc(true);
    let table = DATA + 0x1000;
    // Walking starts at entry 0, which links to entry 2. Entry 1 is not part of the chain.
    d.write_raw_desc(table, DATA, 3, VIRTQ_DESC_F_NEXT, 2);
    d.write_raw_desc(table + 16, DATA + 0x40, 9, 0, 0);
    d.write_raw_desc(table + 32, DATA + 0x10, 5, VIRTQ_DESC_F_WRITE, 0);
    // A direct descriptor first, then the indirect one with a stray write flag that is ignored.
    d.write_desc(0, DATA + 0x20, 2, VIRTQ_DESC_F_NEXT, 1);
    d.write_desc(1, table, 48, VIRTQ_DESC_F_INDIRECT | VIRTQ_DESC_F_WRITE, 0);
    d.publish(0);
    d.mem.write(DATA + 0x20, b"ab").unwrap();
    d.mem.write(DATA, b"cde").unwrap();

    let c = q.pop(&d.mem).unwrap().unwrap();
    assert_eq!(c.head(), 0);
    assert_eq!(c.ring_slots(), 2);
    assert_eq!(
        c.descriptors(),
        &[
            Descriptor::new(DATA + 0x20, 2, false),
            Descriptor::new(DATA, 3, false),
            Descriptor::new(DATA + 0x10, 5, true),
        ]
    );
    assert_eq!(c.reader(&d.mem).read_to_vec().unwrap(), b"abcde");
}

/// Publishes a single indirect descriptor pointing at `table` with the given length.
fn indirect_queue(len: u32) -> (SplitDriver, SplitQueue) {
    let mut d = SplitDriver::new(4);
    let mut q = queue(&d);
    q.set_indirect_desc(true);
    d.write_desc(0, DATA + 0x1000, len, VIRTQ_DESC_F_INDIRECT, 0);
    d.publish(0);
    (d, q)
}

#[test]
fn malformed_indirect_tables() {
    let (d, mut q) = indirect_queue(32);
    q.set_indirect_desc(false);
    assert_eq!(q.pop(&d.mem), Err(QueueError::IndirectNotNegotiated));

    for len in [0, 8, 17, 40] {
        let (d, mut q) = indirect_queue(len);
        assert_eq!(q.pop(&d.mem), Err(QueueError::InvalidIndirectLength(len)), "len {len}");
    }

    let (d, mut q) = indirect_queue(16 * 9);
    q.set_max_indirect_len(8);
    assert_eq!(q.pop(&d.mem), Err(QueueError::InvalidIndirectLength(144)));

    let (d, mut q) = indirect_queue(16);
    d.write_desc(0, DATA + 0x1000, 16, VIRTQ_DESC_F_INDIRECT | VIRTQ_DESC_F_NEXT, 1);
    assert_eq!(q.pop(&d.mem), Err(QueueError::IndirectWithNext));

    let (d, mut q) = indirect_queue(16);
    d.write_raw_desc(DATA + 0x1000, DATA, 16, VIRTQ_DESC_F_INDIRECT, 0);
    assert_eq!(q.pop(&d.mem), Err(QueueError::NestedIndirect));

    let (d, mut q) = indirect_queue(32);
    d.write_raw_desc(DATA + 0x1000, DATA, 1, VIRTQ_DESC_F_NEXT, 1);
    d.write_raw_desc(DATA + 0x1010, DATA, 1, VIRTQ_DESC_F_NEXT, 0);
    assert_eq!(q.pop(&d.mem), Err(QueueError::ChainTooLong { limit: 2 }));

    let (d, mut q) = indirect_queue(32);
    d.write_raw_desc(DATA + 0x1000, DATA, 1, VIRTQ_DESC_F_NEXT, 2);
    assert_eq!(
        q.pop(&d.mem),
        Err(QueueError::DescriptorIndexOutOfRange { index: 2, table_len: 2 })
    );

    let (d, mut q) = indirect_queue(32);
    d.write_raw_desc(DATA + 0x1000, DATA, 1, VIRTQ_DESC_F_WRITE | VIRTQ_DESC_F_NEXT, 1);
    d.write_raw_desc(DATA + 0x1010, DATA, 1, 0, 0);
    assert_eq!(q.pop(&d.mem), Err(QueueError::ReadableAfterWritable));

    // A well formed single entry table goes through.
    let (d, mut q) = indirect_queue(16);
    assert_eq!(q.next_avail(), 0);
    d.write_raw_desc(DATA + 0x1000, DATA, 1, 0, 0);
    assert!(q.pop(&d.mem).unwrap().is_some());
}

#[test]
fn used_head_must_be_in_the_table() {
    let d = SplitDriver::new(4);
    let mut q = queue(&d);
    assert_eq!(q.add_used(&d.mem, 4, 0), Err(QueueError::InvalidUsedHead { head: 4, size: 4 }));
    assert_eq!(d.used_idx(), 0);
}

#[test]
fn undo_pop_returns_the_same_chain() {
    let mut d = SplitDriver::new(4);
    let mut q = queue(&d);
    d.add_chain(1, &[(DATA, 1, true)]);
    let a = q.pop(&d.mem).unwrap().unwrap();
    q.undo_pop();
    let b = q.pop(&d.mem).unwrap().unwrap();
    assert_eq!(a, b);
}

#[test]
fn flags_based_suppression() {
    let mut d = SplitDriver::new(4);
    let mut q = queue(&d);

    q.disable_notification(&d.mem).unwrap();
    assert_eq!(d.used_flags(), VIRTQ_USED_F_NO_NOTIFY);
    assert!(!q.enable_notification(&d.mem).unwrap());
    assert_eq!(d.used_flags(), 0);

    // A buffer that arrived while notifications were off is reported by the re-check.
    q.disable_notification(&d.mem).unwrap();
    d.add_chain(0, &[(DATA, 1, true)]);
    assert!(q.enable_notification(&d.mem).unwrap());
    let c = q.pop(&d.mem).unwrap().unwrap();

    q.add_used(&d.mem, c.head(), 1).unwrap();
    assert!(q.needs_notification(&d.mem).unwrap());
    d.set_avail_flags(VIRTQ_AVAIL_F_NO_INTERRUPT);
    assert!(!q.needs_notification(&d.mem).unwrap());
}

#[test]
fn used_event_decides_interrupts() {
    let mut d = SplitDriver::new(8);
    let mut q = queue(&d);
    q.set_event_idx(true);
    // The flag is ignored once event index is in use.
    d.set_avail_flags(VIRTQ_AVAIL_F_NO_INTERRUPT);
    for head in 0..8 {
        d.add_chain(head, &[(DATA, 1, true)]);
    }
    let used = |q: &mut SplitQueue, n: u16| {
        for _ in 0..n {
            let c = q.pop(&d.mem).unwrap().unwrap();
            q.add_used(&d.mem, c.head(), 0).unwrap();
        }
    };

    used(&mut q, 1);
    // No history yet, so the first check notifies.
    assert!(q.needs_notification(&d.mem).unwrap());

    // The driver wants to hear when the element at index 2 is used, that is when idx goes to 3.
    d.set_used_event(2);
    used(&mut q, 1);
    assert!(!q.needs_notification(&d.mem).unwrap(), "idx 1 -> 2");
    used(&mut q, 1);
    assert!(q.needs_notification(&d.mem).unwrap(), "idx 2 -> 3");
    used(&mut q, 1);
    assert!(!q.needs_notification(&d.mem).unwrap(), "idx 3 -> 4");
    // Nothing new since the last check.
    assert!(!q.needs_notification(&d.mem).unwrap());

    // A batch that jumps over the event still notifies.
    d.set_used_event(5);
    used(&mut q, 3);
    assert!(q.needs_notification(&d.mem).unwrap(), "idx 4 -> 7");
}

#[test]
fn used_event_across_the_16_bit_wrap() {
    let mut d = SplitDriver::new(4);
    let mut q = queue(&d);
    q.set_event_idx(true);
    q.set_next_used(0xfffe);
    d.add_chain(0, &[(DATA, 1, true)]);
    d.add_chain(1, &[(DATA, 1, true)]);
    d.add_chain(2, &[(DATA, 1, true)]);
    assert!(q.needs_notification(&d.mem).unwrap());
    d.set_used_event(0xffff);
    for _ in 0..2 {
        let c = q.pop(&d.mem).unwrap().unwrap();
        q.add_used(&d.mem, c.head(), 0).unwrap();
    }
    assert_eq!(q.next_used(), 0);
    assert!(q.needs_notification(&d.mem).unwrap(), "0xfffe -> 0 passes 0xffff");
    d.set_used_event(0xffff);
    let c = q.pop(&d.mem).unwrap().unwrap();
    q.add_used(&d.mem, c.head(), 0).unwrap();
    assert!(!q.needs_notification(&d.mem).unwrap(), "0 -> 1 does not");
}

#[test]
fn avail_event_follows_the_device() {
    let mut d = SplitDriver::new(4);
    let mut q = queue(&d);
    q.set_event_idx(true);
    d.add_chain(0, &[(DATA, 1, true)]);
    d.add_chain(1, &[(DATA, 1, true)]);
    d.add_chain(2, &[(DATA, 1, true)]);

    // With notifications enabled each pop asks for a notification at the next index.
    q.pop(&d.mem).unwrap().unwrap();
    assert_eq!(d.avail_event(), 1);

    // Disabling leaves avail_event behind, and the used flags alone.
    q.disable_notification(&d.mem).unwrap();
    assert_eq!(d.used_flags(), 0);
    q.pop(&d.mem).unwrap().unwrap();
    assert_eq!(d.avail_event(), 1);

    // Enabling catches up and reports the entry still waiting.
    assert!(q.enable_notification(&d.mem).unwrap());
    assert_eq!(d.avail_event(), 2);
    q.pop(&d.mem).unwrap().unwrap();
    assert_eq!(d.avail_event(), 3);
    assert!(!q.enable_notification(&d.mem).unwrap());
}
