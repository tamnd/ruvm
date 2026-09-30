// SPDX-License-Identifier: GPL-2.0-or-later

//! tests/unit/test-hbitmap.c: every operation is checked against a plain shadow bitmap.
//! `BITS_PER_LONG` is 64, as on the hosts QEMU tests on.

use super::hbitmap::{HBitmap, HBitmapIter};

const L1: u64 = 64;
const L2: u64 = 64 * L1;
const L3: u64 = 64 * L2;
const MAX: u64 = i64::MAX as u64;

struct Data {
    hb: HBitmap,
    bits: Vec<u64>,
    size: u64,
    old_size: u64,
    granularity: u32,
}

fn array_size(bits: u64) -> usize {
    bits.div_ceil(64).max(1) as usize
}

impl Data {
    /// `hbitmap_test_init()`.
    fn new(size: u64, granularity: u32) -> Self {
        let d = Data {
            hb: HBitmap::new(size, granularity),
            bits: vec![0; array_size(size)],
            size,
            old_size: size,
            granularity,
        };
        if size > 0 {
            d.check(0);
        }
        d
    }

    fn shadow(&self, i: u64) -> bool {
        self.bits[(i / 64) as usize] & (1u64 << (i % 64)) != 0
    }

    /// `hbitmap_test_check()`: the iterator and the shadow bitmap agree from `first` on.
    fn check(&self, first: u64) {
        let mut count = 0u64;
        let mut it = HBitmapIter::new(&self.hb, first);
        let mut i = first;
        loop {
            let next = it.next(&self.hb).unwrap_or(self.size);
            while i < next {
                assert!(!self.shadow(i), "bit {i} set in the shadow only");
                i += 1;
            }
            if next == self.size {
                break;
            }
            assert!(self.shadow(i), "bit {i} set in the hbitmap only");
            i += 1;
            count += 1;
        }
        if first == 0 {
            assert_eq!(count << self.granularity, self.hb.count());
        }
    }

    /// `hbitmap_test_truncate_impl()`.
    fn truncate(&mut self, size: u64) {
        self.old_size = self.size;
        self.size = size;
        if self.size == self.old_size {
            return;
        }
        let n = array_size(size);
        self.bits.resize(n, 0);
        if self.size < self.old_size {
            let m = size % 64;
            if m != 0 {
                self.bits[n - 1] &= (1u64 << m) - 1;
            }
        }
        self.hb.truncate(size);
    }

    /// `hbitmap_test_set()`.
    fn set(&mut self, first: u64, count: u64) {
        self.hb.set(first, count);
        for i in first..first + count {
            self.bits[(i / 64) as usize] |= 1u64 << (i % 64);
        }
        if self.granularity == 0 {
            self.check(0);
        }
    }

    /// `hbitmap_test_reset()`.
    fn reset(&mut self, first: u64, count: u64) {
        self.hb.reset(first, count);
        for i in first..first + count {
            self.bits[(i / 64) as usize] &= !(1u64 << (i % 64));
        }
        if self.granularity == 0 {
            self.check(0);
        }
    }

    /// `hbitmap_test_reset_all()`.
    fn reset_all(&mut self) {
        self.hb.reset_all();
        self.bits.fill(0);
        if self.granularity == 0 {
            self.check(0);
        }
    }

    /// `hbitmap_test_check_get()`.
    fn check_get(&self) {
        let mut count = 0;
        for i in 0..self.size {
            let v = self.hb.get(i);
            count += u64::from(v);
            assert_eq!(v, self.shadow(i), "bit {i}");
        }
        assert_eq!(count, self.hb.count());
    }
}

#[test]
fn size_0() {
    let _ = Data::new(0, 0);
}

#[test]
fn size_unaligned() {
    let mut d = Data::new(L3 + 23, 0);
    d.set(0, 1);
    d.set(L3 + 22, 1);
}

#[test]
fn iter_empty() {
    let _ = Data::new(L1, 0);
}

#[test]
fn iter_partial() {
    let mut d = Data::new(L3, 0);
    d.set(0, L3);
    for first in [
        1,
        L1 - 1,
        L1,
        L1 * 2 - 1,
        L2 - 1,
        L2,
        L2 + 1,
        L2 + L1,
        L2 + L1 * 2 - 1,
        L2 * 2 - 1,
        L2 * 2,
        L2 * 2 + 1,
        L2 * 2 + L1,
        L2 * 2 + L1 * 2 - 1,
        L3 / 2,
    ] {
        d.check(first);
    }
}

#[test]
fn iter_granularity() {
    let mut d = Data::new(131072 << 7, 7);
    let mut it = HBitmapIter::new(&d.hb, 0);
    assert_eq!(it.next(&d.hb), None);

    d.set(((L2 + L1 + 1) << 7) + 8, 8);
    let mut it = HBitmapIter::new(&d.hb, 0);
    assert_eq!(it.next(&d.hb), Some((L2 + L1 + 1) << 7));
    assert_eq!(it.next(&d.hb), None);

    let mut it = HBitmapIter::new(&d.hb, (L2 + L1 + 2) << 7);
    assert_eq!(it.next(&d.hb), None);

    d.set((131072 << 7) - 8, 8);
    let mut it = HBitmapIter::new(&d.hb, 0);
    assert_eq!(it.next(&d.hb), Some((L2 + L1 + 1) << 7));
    assert_eq!(it.next(&d.hb), Some(131071 << 7));
    assert_eq!(it.next(&d.hb), None);

    let mut it = HBitmapIter::new(&d.hb, (L2 + L1 + 2) << 7);
    assert_eq!(it.next(&d.hb), Some(131071 << 7));
    assert_eq!(it.next(&d.hb), None);
}

#[test]
fn get_all() {
    let mut d = Data::new(L3, 0);
    d.set(0, L3);
    d.check_get();
}

#[test]
fn get_some() {
    let mut d = Data::new(2 * L2, 0);
    for pos in [10, L1 - 1, L1, L2 - 1, L2] {
        d.set(pos, 1);
        d.check_get();
    }
}

#[test]
fn set_all() {
    let mut d = Data::new(L3, 0);
    d.set(0, L3);
}

#[test]
fn set_one() {
    let mut d = Data::new(2 * L2, 0);
    for pos in [10, L1 - 1, L1, L2 - 1, L2] {
        d.set(pos, 1);
    }
}

#[test]
fn set_two_elem() {
    let mut d = Data::new(2 * L2, 0);
    d.set(L1 - 1, 2);
    d.set(L1 * 2 - 1, 4);
    d.set(L1 * 4, L1 + 1);
    d.set(L1 * 8 - 1, L1 + 1);
    d.set(L2 - 1, 2);
    d.set(L2 + L1 - 1, 8);
    d.set(L2 + L1 * 4, L1 + 1);
    d.set(L2 + L1 * 8 - 1, L1 + 1);
}

#[test]
fn set_general() {
    let mut d = Data::new(L3 * 2, 0);
    d.set(L1 - 1, L1 + 2);
    d.set(L1 * 3 - 1, L1 + 2);
    d.set(L1 * 5, L1 * 2 + 1);
    d.set(L1 * 8 - 1, L1 * 2 + 1);
    d.set(L2 - 1, L1 + 2);
    d.set(L2 + L1 * 2 - 1, L1 + 2);
    d.set(L2 + L1 * 4, L1 * 2 + 1);
    d.set(L2 + L1 * 7 - 1, L1 * 2 + 1);
    d.set(L2 * 2 - 1, L3 * 2 - L2 * 2);
}

#[test]
fn set_twice() {
    let mut d = Data::new(L1 * 3, 0);
    d.set(0, L1 * 3);
    d.set(L1, 1);
}

#[test]
fn set_overlap() {
    let mut d = Data::new(L3 * 2, 0);
    d.set(L1 - 1, L1 + 2);
    d.set(L1 * 2 - 1, L1 * 2 + 2);
    d.set(0, L1 * 3);
    d.set(L1 * 8 - 1, L2);
    d.set(L2, L1);
    d.set(L2 - L1 - 1, L1 * 8 + 2);
    d.set(L2, L3 - L2 + 1);
    d.set(L3 - L1, L1 * 3);
    d.set(L3 - 1, 3);
    d.set(L3 - 1, L2);
}

#[test]
fn reset_empty() {
    let mut d = Data::new(L3, 0);
    d.reset(0, L3);
}

#[test]
fn reset_general() {
    let mut d = Data::new(L3 * 2, 0);
    d.set(L1 - 1, L1 + 2);
    d.reset(L1 * 2 - 1, L1 * 2 + 2);
    d.set(0, L1 * 3);
    d.reset(L1 * 8 - 1, L2);
    d.set(L2, L1);
    d.reset(L2 - L1 - 1, L1 * 8 + 2);
    d.set(L2, L3 - L2 + 1);
    d.reset(L3 - L1, L1 * 3);
    d.set(L3 - 1, 3);
    d.reset(L3 - 1, L2);
    d.set(0, L3 * 2);
    d.reset(0, L1);
    d.reset(0, L2);
    d.reset(L3, L3);
    d.set(L3 / 2, L3);
}

#[test]
fn reset_all() {
    let mut d = Data::new(L3 * 2, 0);
    d.set(L1 - 1, L1 + 2);
    d.reset_all();
    d.set(0, L1 * 3);
    d.reset_all();
    d.set(L2, L1);
    d.reset_all();
    d.set(L2, L3 - L2 + 1);
    d.reset_all();
    d.set(L3 - 1, 3);
    d.reset_all();
    d.set(0, L3 * 2);
    d.reset_all();
    d.set(L3 / 2, L3);
    d.reset_all();
}

#[test]
fn granularity() {
    let mut d = Data::new(L1, 1);
    d.set(0, 1);
    assert_eq!(d.hb.count(), 2);
    d.check(0);
    d.set(2, 1);
    assert_eq!(d.hb.count(), 4);
    d.check(0);
    d.set(0, 3);
    assert_eq!(d.hb.count(), 4);
    d.reset(0, 2);
    assert_eq!(d.hb.count(), 2);
}

/// `hbitmap_test_set_boundary_bits()`.
fn set_boundary_bits(d: &mut Data, diff: i64) {
    let size = d.size;
    d.set(0, 1);
    if diff < 0 {
        d.set((size as i64 + diff - 1) as u64, 1);
        d.set((size as i64 + diff) as u64, 1);
    }
    d.set(size - 1, 1);
    if d.granularity == 0 {
        d.check_get();
    }
}

/// `hbitmap_test_check_boundary_bits()`.
fn check_boundary_bits(d: &Data) {
    let size = d.size.min(d.old_size);
    if d.granularity == 0 {
        d.check_get();
        d.check(0);
    } else {
        assert!(d.hb.get(0));
        assert!(d.hb.get(size - 1));
        assert_eq!(2u64 << d.granularity, d.hb.count());
    }
}

/// `hbitmap_test_truncate()`.
fn truncate_test(size: u64, diff: i64, granularity: u32) {
    let mut d = Data::new(size, granularity);
    set_boundary_bits(&mut d, diff);
    d.truncate((size as i64 + diff) as u64);
    check_boundary_bits(&d);
}

#[test]
fn truncate_nop() {
    truncate_test(L2, 0, 0);
}

#[test]
fn truncate_grow_negligible() {
    truncate_test(L2 - 1, 1, 1);
}

#[test]
fn truncate_shrink_negligible() {
    truncate_test(L2, -1, 1);
}

#[test]
fn truncate_grow_tiny() {
    truncate_test(L2 - 2, 1, 1);
}

#[test]
fn truncate_shrink_tiny() {
    truncate_test(L2 - 1, -1, 1);
}

#[test]
fn truncate_grow_small() {
    truncate_test(L2 + 1, 4, 0);
}

#[test]
fn truncate_shrink_small() {
    truncate_test(L2, -4, 0);
}

#[test]
fn truncate_grow_medium() {
    truncate_test(L2 - 1, 4, 0);
}

#[test]
fn truncate_shrink_medium() {
    truncate_test(L2 + 1, -4, 0);
}

#[test]
fn truncate_grow_large() {
    truncate_test(L2, 64, 0);
}

#[test]
fn truncate_shrink_large() {
    truncate_test(L2, -64, 0);
}

#[test]
fn serialize_align() {
    let d = Data::new(L3 * 2, 3);
    assert!(d.hb.is_serializable());
    assert_eq!(d.hb.serialization_align(), 64 << 3);
}

fn buf_bit(buf: &[u8], i: u64) -> bool {
    buf[(i / 8) as usize] & (1 << (i % 8)) != 0
}

/// `hbitmap_test_serialize_range()`. The buffer is little-endian 64-bit words, so bit `i`
/// is bit `i % 8` of byte `i / 8`.
fn serialize_range(d: &mut Data, buf: &mut [u8], pos: u64, count: u64) {
    assert_eq!(d.hb.granularity(), 0);
    d.hb.reset_all();
    buf.fill(0);
    if count > 0 {
        d.hb.set(pos, count);
    }
    d.hb.serialize_part(buf, 0, d.size);
    for i in 0..d.size {
        assert_eq!(buf_bit(buf, i), i >= pos && i < pos + count);
    }

    buf.fill(0);
    d.hb.serialize_part(buf, 0, d.size);
    d.hb.reset_all();
    d.hb.deserialize_part(buf, 0, d.size, true);
    for i in 0..d.size {
        assert_eq!(d.hb.get(i), i >= pos && i < pos + count);
    }
}

#[test]
fn serialize_basic() {
    let positions = [0, 1, L1 - 1, L1, L2 - 1, L2, L2 + 1, L3 - 1];
    let mut d = Data::new(L3, 0);
    assert!(d.hb.is_serializable());
    let size = d.hb.serialization_size(0, d.size);
    let mut buf = vec![0u8; size as usize];
    for &p in &positions {
        for &q in &positions {
            serialize_range(&mut d, &mut buf, p, q.min(L3 - p));
        }
    }
}

#[test]
fn serialize_part() {
    let positions = [0, 1, L1 - 1, L1, L2 - 1, L2, L2 + 1, L3 - 1];
    let mut d = Data::new(L3, 0);
    let buf_size = L2;
    let mut buf = vec![0u8; buf_size as usize];
    for &p in &positions {
        d.hb.set(p, 1);
    }
    assert!(d.hb.is_serializable());
    let mut i = 0;
    while i < d.size {
        d.hb.serialize_part(&mut buf, i, buf_size);
        for j in 0..buf_size {
            assert_eq!(positions.contains(&(j + i)), buf_bit(&buf, j));
        }
        i += buf_size;
    }
}

#[test]
fn serialize_zeroes() {
    let min_l1 = L1.max(64);
    let positions = [0, min_l1, L2, L3 - min_l1];
    let mut d = Data::new(L3, 0);
    for &p in &positions {
        d.hb.set(p, L1);
    }
    for (i, &p) in positions.iter().enumerate() {
        d.hb.deserialize_zeroes(p, min_l1, true);
        let mut it = HBitmapIter::new(&d.hb, 0);
        assert_eq!(it.next(&d.hb), positions.get(i + 1).copied());
    }
}

#[test]
fn iter_and_reset() {
    let mut d = Data::new(L1 * 2, 0);
    d.hb.set(0, d.size);
    let mut it = HBitmapIter::new(&d.hb, 63);
    it.next(&d.hb);
    d.hb.reset_all();
    it.next(&d.hb);
}

/// `test_hbitmap_next_x_check_range()`.
fn next_x_check_range(d: &Data, start: u64, count: u64) {
    let next_zero = d.hb.next_zero(start, count);
    let next_dirty = d.hb.next_dirty(start, count);
    let end = if start >= d.size || d.size - start < count { d.size } else { start + count };
    let first_bit = d.hb.get(start);
    let mut next = start;
    while next < end && d.hb.get(next) == first_bit {
        next += 1;
    }
    let next = (next != end).then_some(next);
    assert_eq!(next_dirty, if first_bit { Some(start) } else { next }, "next_dirty({start})");
    assert_eq!(next_zero, if first_bit { next } else { Some(start) }, "next_zero({start})");
}

fn next_x_check(d: &Data, start: u64) {
    next_x_check_range(d, start, MAX);
}

fn next_x_do(granularity: u32) {
    let mut d = Data::new(L3, granularity);
    next_x_check(&d, 0);
    next_x_check(&d, L3 - 1);
    next_x_check_range(&d, 0, 1);
    next_x_check_range(&d, L3 - 1, 1);

    d.hb.set(L2, 1);
    for s in [0, L2 - 1, L2, L2 + 1] {
        next_x_check(&d, s);
    }
    for (s, c) in [(0, 1), (0, L2), (L2 - 1, 1), (L2 - 1, 2), (L2, 1), (L2 + 1, 1)] {
        next_x_check_range(&d, s, c);
    }

    d.hb.set(L2 + 5, L1);
    for s in [0, L2 - L1, L2 + 1, L2 + 2, L2 + 5, L2 + L1 - 1, L2 + L1, L2 + L1 + 1] {
        next_x_check(&d, s);
    }
    for (s, c) in [
        (L2 - 2, L1),
        (L2, 4),
        (L2, 6),
        (L2 + 1, 3),
        (L2 + 4, L1),
        (L2 + 5, L1),
        (L2 + 5 + L1 - 1, 1),
        (L2 + 5 + L1, 1),
        (L2 + 5 + L1 + 1, 1),
    ] {
        next_x_check_range(&d, s, c);
    }

    d.hb.set(L2 * 2, L3 - L2 * 2);
    for s in [L2 * 2 - L1, L2 * 2 - 2, L2 * 2 - 1, L2 * 2, L2 * 2 + 1, L2 * 2 + L1, L3 - 1] {
        next_x_check(&d, s);
    }
    next_x_check_range(&d, L2 * 2 - L1, L1 + 1);
    next_x_check_range(&d, L2 * 2, L2);

    d.hb.set(0, L3);
    next_x_check(&d, 0);
}

#[test]
fn next_x_0() {
    next_x_do(0);
}

#[test]
fn next_x_4() {
    next_x_do(4);
}

#[test]
fn next_x_after_truncate() {
    let mut d = Data::new(L1, 0);
    d.truncate(L1 * 2);
    d.hb.set(0, L1);
    next_x_check(&d, 0);
}

/// `test_hbitmap_next_dirty_area_check_limited()`.
fn dirty_area_check_limited(d: &Data, offset: u64, count: u64, max_dirty: u64) {
    let r1 =
        d.hb.next_dirty_area(offset, if count == MAX { MAX } else { offset + count }, max_dirty);
    let end = if offset > d.size || d.size - offset < count { d.size } else { offset + count };
    let mut off2 = offset;
    while off2 < end && !d.hb.get(off2) {
        off2 += 1;
    }
    let mut len2 = 1;
    while off2 + len2 < end && len2 < max_dirty && d.hb.get(off2 + len2) {
        len2 += 1;
    }
    let r2 = (off2 < end).then_some((off2, len2));
    assert_eq!(r1, r2, "next_dirty_area({offset}, {count}, {max_dirty})");
}

fn dirty_area_check(d: &Data, offset: u64, count: u64) {
    dirty_area_check_limited(d, offset, count, MAX);
}

fn dirty_area_do(granularity: u32) {
    let mut d = Data::new(L3, granularity);
    dirty_area_check(&d, 0, MAX);
    dirty_area_check(&d, 0, 1);
    dirty_area_check(&d, L3 - 1, 1);
    dirty_area_check_limited(&d, 0, MAX, 1);

    d.hb.set(L2, 1);
    for (o, c) in [
        (0, 1),
        (0, L2),
        (0, MAX),
        (L2 - 1, MAX),
        (L2 - 1, 1),
        (L2 - 1, 2),
        (L2 - 1, 3),
        (L2, MAX),
        (L2, 1),
        (L2 + 1, 1),
    ] {
        dirty_area_check(&d, o, c);
    }
    dirty_area_check_limited(&d, 0, MAX, 1);
    dirty_area_check_limited(&d, L2 - 1, 2, 1);

    d.hb.set(L2 + 5, L1);
    for (o, c) in [
        (0, MAX),
        (L2 - 2, 8),
        (L2 + 1, 5),
        (L2 + 1, 3),
        (L2 + 4, L1),
        (L2 + 5, L1),
        (L2 + 7, L1),
        (L2 + L1, L1),
        (L2, 0),
        (L2 + 1, 0),
    ] {
        dirty_area_check(&d, o, c);
    }
    dirty_area_check_limited(&d, L2 + 3, MAX, 3);
    dirty_area_check_limited(&d, L2 + 3, 7, 10);

    d.hb.set(L2 * 2, L3 - L2 * 2);
    for (o, c) in [
        (0, MAX),
        (L2, MAX),
        (L2 + 1, MAX),
        (L2 + 5 + L1 - 1, MAX),
        (L2 + 5 + L1, 5),
        (L2 * 2 - L1, L1 + 1),
        (L2 * 2, L2),
    ] {
        dirty_area_check(&d, o, c);
    }
    dirty_area_check_limited(&d, L2 * 2 + 1, MAX, 5);
    dirty_area_check_limited(&d, L2 * 2 + 1, 10, 5);
    dirty_area_check_limited(&d, L2 * 2 + 1, 2, 5);

    d.hb.set(0, L3);
    dirty_area_check(&d, 0, MAX);
}

#[test]
fn next_dirty_area_0() {
    dirty_area_do(0);
}

#[test]
fn next_dirty_area_1() {
    dirty_area_do(1);
}

#[test]
fn next_dirty_area_4() {
    dirty_area_do(4);
}

#[test]
fn next_dirty_area_after_truncate() {
    let mut d = Data::new(L1, 0);
    d.truncate(L1 * 2);
    d.hb.set(L1 + 1, 1);
    dirty_area_check(&d, 0, MAX);
}

#[test]
fn merge_same_and_different_granularity() {
    let mut a = HBitmap::new(L2, 0);
    let mut b = HBitmap::new(L2, 0);
    a.set(1, 3);
    b.set(L1, 2);
    let mut r = HBitmap::new(L2, 0);
    assert!(HBitmap::merge3(&a, &b, &mut r));
    assert_eq!(r.count(), 5);
    assert!(r.get(1) && r.get(3) && r.get(L1 + 1) && !r.get(4));

    let mut coarse = HBitmap::new(L2, 4);
    coarse.merge(&r);
    assert_eq!(coarse.count(), 32);
    assert!(coarse.get(15) && coarse.get(L1 + 15) && !coarse.get(16));
    assert!(!HBitmap::merge3(&a, &HBitmap::new(L1, 0), &mut r));
}

#[test]
fn status_and_sha256() {
    let mut hb = HBitmap::new(L2, 0);
    hb.set(10, 5);
    assert_eq!(hb.status(0, 100), (false, 10));
    assert_eq!(hb.status(10, 100), (true, 5));
    assert_eq!(hb.status(20, 100), (false, 100));
    // The digest of 64 zero words, what QEMU gives for an empty 4096-bit bitmap.
    let empty = HBitmap::new(L2, 0).sha256().unwrap();
    assert_eq!(empty, "076a27c79e5ace2a3d47f9dd2e83e4ff6ea8872b3c2218f66c92b89b55f36560");
    assert_ne!(hb.sha256().unwrap(), empty);
}
