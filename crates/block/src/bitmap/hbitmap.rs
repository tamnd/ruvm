// SPDX-License-Identifier: GPL-2.0-or-later

//! `HBitmap` from util/hbitmap.c: a bitmap of `size` items where one bit stands for
//! `1 << granularity` consecutive items, with fast searches for set and clear bits.
//!
//! QEMU keeps seven levels of words so that finding the next set bit touches few words. Here
//! there are two: the bits themselves and a summary with one bit per word that is set when the
//! word is not zero. That is enough for the sizes block devices have (one summary word covers
//! 4096 bits) and keeps the code small; the results of every operation are the same.
//!
//! Differences from QEMU:
//!
//! - There are no meta bitmaps (`hbitmap_create_meta()`), which only migration uses.
//! - [`HBitmapIter`] is a cursor that finds the next set bit at or after its position each
//!   time. QEMU's iterator caches the rest of the current word, so a bit set behind the
//!   cursor in that word is not seen there either; bits set ahead of the cursor after the
//!   iterator was made may or may not be seen in QEMU and always are here.
//! - Serialisation always uses 64-bit words, as QEMU does on 64-bit hosts.

/// Bits per word.
const BITS: u64 = 64;

/// `HBITMAP_LOG_MAX_SIZE` on a 64-bit host.
const LOG_MAX_SIZE: u32 = 64 * 2 - 7;

/// A hierarchical bitmap, `HBitmap`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HBitmap {
    /// `orig_size`: the number of items.
    orig_size: u64,
    /// `size`: the number of bits.
    size: u64,
    granularity: u32,
    /// The bits, the last level in QEMU.
    words: Vec<u64>,
    /// One bit per word of `words`, set when that word is not zero.
    summary: Vec<u64>,
    /// `count`: how many bits are set.
    count: u64,
}

fn words_for(bits: u64) -> usize {
    bits.div_ceil(BITS).max(1) as usize
}

/// The mask of bits `lo..=hi` within one word.
fn mask(lo: u64, hi: u64) -> u64 {
    let upper = if hi >= 63 { u64::MAX } else { (1u64 << (hi + 1)) - 1 };
    upper & !((1u64 << lo) - 1)
}

impl HBitmap {
    /// `hbitmap_alloc()`: a bitmap of `size` items, one bit for `1 << granularity` of them.
    pub(crate) fn new(size: u64, granularity: u32) -> Self {
        assert!(size <= i64::MAX as u64);
        assert!(granularity < 64);
        let bits = size.div_ceil(1u64 << granularity);
        assert!(bits <= 1u64 << LOG_MAX_SIZE.min(63));
        let n = words_for(bits);
        HBitmap {
            orig_size: size,
            size: bits,
            granularity,
            words: vec![0; n],
            summary: vec![0; words_for(n as u64)],
            count: 0,
        }
    }

    /// The number of items, `orig_size`.
    #[allow(clippy::misnamed_getters, reason = "`size` is the scaled size, as in QEMU")]
    pub(crate) fn size(&self) -> u64 {
        self.orig_size
    }

    /// `hbitmap_granularity()`.
    pub(crate) fn granularity(&self) -> u32 {
        self.granularity
    }

    /// `hbitmap_empty()`.
    pub(crate) fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// `hbitmap_count()`: the number of items covered by set bits.
    pub(crate) fn count(&self) -> u64 {
        self.count << self.granularity
    }

    fn update_summary(&mut self, w: usize) {
        let bit = 1u64 << (w as u64 % BITS);
        let s = &mut self.summary[w / BITS as usize];
        if self.words[w] != 0 {
            *s |= bit;
        } else {
            *s &= !bit;
        }
    }

    /// `hb_count_between()`: the set bits among bits `first..=last`.
    fn count_between(&self, first: u64, last: u64) -> u64 {
        let mut n = 0;
        let mut b = first;
        while b <= last {
            let w = (b / BITS) as usize;
            let hi = (last - (b - b % BITS)).min(BITS - 1);
            n += u64::from((self.words[w] & mask(b % BITS, hi)).count_ones());
            b = (w as u64 + 1) * BITS;
        }
        n
    }

    /// Sets or clears bits `first..=last`, returning whether anything changed.
    fn change_bits(&mut self, first: u64, last: u64, set: bool) -> bool {
        let mut changed = false;
        let mut b = first;
        while b <= last {
            let w = (b / BITS) as usize;
            let hi = (last - (b - b % BITS)).min(BITS - 1);
            let m = mask(b % BITS, hi);
            let old = self.words[w];
            let new = if set { old | m } else { old & !m };
            if new != old {
                self.words[w] = new;
                changed = true;
                self.update_summary(w);
            }
            b = (w as u64 + 1) * BITS;
        }
        changed
    }

    /// `hbitmap_set()`: marks the items `start..start + count`.
    pub(crate) fn set(&mut self, start: u64, count: u64) {
        if count == 0 {
            return;
        }
        let first = start >> self.granularity;
        let last = (start + count - 1) >> self.granularity;
        assert!(last < self.size, "hbitmap_set() past the end");
        let n = last - first + 1;
        self.count += n - self.count_between(first, last);
        self.change_bits(first, last, true);
    }

    /// `hbitmap_reset()`: clears the items `start..start + count`. Both must be aligned to
    /// the granularity, except that the range may end at the end of the bitmap.
    pub(crate) fn reset(&mut self, start: u64, count: u64) {
        if count == 0 {
            return;
        }
        let gran = 1u64 << self.granularity;
        assert!(start % gran == 0, "hbitmap_reset() of an unaligned start");
        assert!(
            count % gran == 0 || start + count == self.orig_size,
            "hbitmap_reset() of an unaligned length"
        );
        let first = start >> self.granularity;
        let last = (start + count - 1) >> self.granularity;
        assert!(last < self.size, "hbitmap_reset() past the end");
        self.count -= self.count_between(first, last);
        self.change_bits(first, last, false);
    }

    /// `hbitmap_reset_all()`.
    pub(crate) fn reset_all(&mut self) {
        self.words.fill(0);
        self.summary.fill(0);
        self.count = 0;
    }

    /// `hbitmap_get()`: whether the bit of `item` is set.
    pub(crate) fn get(&self, item: u64) -> bool {
        let pos = item >> self.granularity;
        assert!(pos < self.size, "hbitmap_get() past the end");
        self.words[(pos / BITS) as usize] & (1u64 << (pos % BITS)) != 0
    }

    /// The first set bit at or after bit `from`.
    fn next_set_bit(&self, from: u64) -> Option<u64> {
        if from >= self.size {
            return None;
        }
        let w = (from / BITS) as usize;
        let cur = self.words[w] & !((1u64 << (from % BITS)) - 1);
        if cur != 0 {
            return Some(w as u64 * BITS + u64::from(cur.trailing_zeros()));
        }
        // Find the next word that is not zero through the summary.
        let next = w + 1;
        if next >= self.words.len() {
            return None;
        }
        let mut s = next / BITS as usize;
        let mut sw = self.summary[s] & !((1u64 << (next as u64 % BITS)) - 1);
        loop {
            if sw != 0 {
                let word = s * BITS as usize + sw.trailing_zeros() as usize;
                let bit = word as u64 * BITS + u64::from(self.words[word].trailing_zeros());
                return Some(bit);
            }
            s += 1;
            if s >= self.summary.len() {
                return None;
            }
            sw = self.summary[s];
        }
    }

    /// The first clear bit at or after bit `from` and before `end`.
    fn next_zero_bit(&self, from: u64, end: u64) -> Option<u64> {
        let mut b = from;
        while b < end {
            let w = (b / BITS) as usize;
            let cur = self.words[w] | ((1u64 << (b % BITS)) - 1);
            if cur != u64::MAX {
                let bit = w as u64 * BITS + u64::from((!cur).trailing_zeros());
                return (bit < end).then_some(bit);
            }
            b = (w as u64 + 1) * BITS;
        }
        None
    }

    /// `hbitmap_next_dirty()`: the first dirty item in `start..start + count`, `None` if
    /// there is none. `count` may reach past the end.
    pub(crate) fn next_dirty(&self, start: u64, count: u64) -> Option<u64> {
        if start >= self.orig_size || count == 0 {
            return None;
        }
        let end = if count > self.orig_size - start { self.orig_size } else { start + count };
        let off = self.next_set_bit(start >> self.granularity)? << self.granularity;
        if off >= end {
            return None;
        }
        Some(off.max(start))
    }

    /// `hbitmap_next_zero()`: the first clean item in `start..start + count`, `None` if
    /// there is none.
    pub(crate) fn next_zero(&self, start: u64, count: u64) -> Option<u64> {
        if start >= self.orig_size || count == 0 {
            return None;
        }
        let end_bit = if count > self.orig_size - start {
            self.size
        } else {
            ((start + count - 1) >> self.granularity) + 1
        };
        let bit = self.next_zero_bit(start >> self.granularity, end_bit)?;
        Some((bit << self.granularity).max(start))
    }

    /// `hbitmap_next_dirty_area()`: the first dirty area in `start..end`, at most
    /// `max_dirty_count` long, as (start, length).
    pub(crate) fn next_dirty_area(
        &self,
        start: u64,
        end: u64,
        max_dirty_count: u64,
    ) -> Option<(u64, u64)> {
        assert!(max_dirty_count > 0);
        let end = end.min(self.orig_size);
        if start >= end {
            return None;
        }
        let start = self.next_dirty(start, end - start)?;
        let mut end = start + (end - start).min(max_dirty_count);
        if let Some(z) = self.next_zero(start, end - start) {
            end = z;
        }
        Some((start, end - start))
    }

    /// `hbitmap_status()`: whether the start of `start..start + count` is dirty, and for how
    /// many items that holds.
    pub(crate) fn status(&self, start: u64, count: u64) -> (bool, u64) {
        assert!(count > 0);
        assert!(start + count <= self.orig_size);
        match self.next_dirty(start, count) {
            None => (false, count),
            Some(d) if d > start => (false, d - start),
            Some(_) => match self.next_zero(start, count) {
                None => (true, count),
                Some(z) => (true, z - start),
            },
        }
    }

    /// `hbitmap_truncate()`: changes the number of items. Growing adds clean bits; shrinking
    /// clears the bits past the new end first.
    pub(crate) fn truncate(&mut self, size: u64) {
        assert!(size <= i64::MAX as u64);
        let num_elements = size;
        self.orig_size = size;
        let bits = size.div_ceil(1u64 << self.granularity);
        if bits == self.size {
            return;
        }
        let shrink = bits < self.size;
        if shrink {
            // Don't clear partial granularity groups, start at the first full one.
            let g = 1u64 << self.granularity;
            let start = num_elements.div_ceil(g) * g;
            let first = start >> self.granularity;
            let last = self.size - 1;
            self.count -= self.count_between(first, last);
            self.change_bits(first, last, false);
        }
        self.size = bits;
        let n = words_for(bits);
        self.words.resize(n, 0);
        // Dropped words were cleared above, so the summary has no bits for them.
        self.summary.resize(words_for(n as u64), 0);
    }

    /// `hbitmap_merge(result, src, result)`: `self |= src`. Both have the same number of
    /// items; the granularities may differ.
    pub(crate) fn merge(&mut self, src: &HBitmap) {
        assert_eq!(self.orig_size, src.orig_size);
        if src.count == 0 {
            return;
        }
        if self.granularity != src.granularity {
            // hbitmap_sparse_merge().
            let mut offset = 0;
            while let Some((o, c)) = src.next_dirty_area(offset, src.orig_size, i64::MAX as u64) {
                self.set(o, c);
                offset = o + c;
            }
            return;
        }
        for (d, s) in self.words.iter_mut().zip(&src.words) {
            *d |= *s;
        }
        for (d, s) in self.summary.iter_mut().zip(&src.summary) {
            *d |= *s;
        }
        self.count = self.words.iter().map(|w| u64::from(w.count_ones())).sum();
    }

    /// `hbitmap_merge(a, b, result)` for a result that is neither input.
    #[cfg(test)]
    pub(crate) fn merge3(a: &HBitmap, b: &HBitmap, result: &mut HBitmap) -> bool {
        if a.orig_size != result.orig_size || b.orig_size != result.orig_size {
            return false;
        }
        result.reset_all();
        result.merge(a);
        result.merge(b);
        true
    }

    /// `hbitmap_is_serializable()`.
    pub(crate) fn is_serializable(&self) -> bool {
        self.granularity < 58
    }

    /// `hbitmap_serialization_align()`: serialised chunks start at multiples of this many
    /// items.
    pub(crate) fn serialization_align(&self) -> u64 {
        assert!(self.is_serializable());
        64u64 << self.granularity
    }

    /// `serialization_chunk()`: the words that hold items `start..start + count`.
    fn serialization_chunk(&self, start: u64, count: u64) -> (usize, usize) {
        let last = start + count - 1;
        let gran = self.serialization_align();
        assert!(start & (gran - 1) == 0);
        assert!((last >> self.granularity) < self.size);
        if (last >> self.granularity) != self.size - 1 {
            assert!(count & (gran - 1) == 0);
        }
        let first = ((start >> self.granularity) / BITS) as usize;
        let last = ((last >> self.granularity) / BITS) as usize;
        (first, last - first + 1)
    }

    /// `hbitmap_serialization_size()`: the bytes items `start..start + count` take.
    pub(crate) fn serialization_size(&self, start: u64, count: u64) -> u64 {
        if count == 0 {
            return 0;
        }
        self.serialization_chunk(start, count).1 as u64 * 8
    }

    /// `hbitmap_serialize_part()`: the words for items `start..start + count` as
    /// little-endian bytes into `buf`.
    pub(crate) fn serialize_part(&self, buf: &mut [u8], start: u64, count: u64) {
        if count == 0 {
            return;
        }
        let (first, n) = self.serialization_chunk(start, count);
        for (i, w) in self.words[first..first + n].iter().enumerate() {
            buf[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
        }
    }

    /// `hbitmap_deserialize_part()`.
    pub(crate) fn deserialize_part(&mut self, buf: &[u8], start: u64, count: u64, finish: bool) {
        if count == 0 {
            return;
        }
        let (first, n) = self.serialization_chunk(start, count);
        for i in 0..n {
            let mut b = [0u8; 8];
            b.copy_from_slice(&buf[i * 8..i * 8 + 8]);
            self.words[first + i] = u64::from_le_bytes(b);
        }
        if finish {
            self.deserialize_finish();
        }
    }

    /// `hbitmap_deserialize_zeroes()`.
    pub(crate) fn deserialize_zeroes(&mut self, start: u64, count: u64, finish: bool) {
        if count == 0 {
            return;
        }
        let (first, n) = self.serialization_chunk(start, count);
        self.words[first..first + n].fill(0);
        if finish {
            self.deserialize_finish();
        }
    }

    /// `hbitmap_deserialize_ones()`.
    pub(crate) fn deserialize_ones(&mut self, start: u64, count: u64, finish: bool) {
        if count == 0 {
            return;
        }
        let (first, n) = self.serialization_chunk(start, count);
        self.words[first..first + n].fill(u64::MAX);
        if finish {
            self.deserialize_finish();
        }
    }

    /// `hbitmap_deserialize_finish()`: rebuilds the summary and the count from the words.
    pub(crate) fn deserialize_finish(&mut self) {
        // Bits past the end may have come in with the last word.
        let tail = self.size % BITS;
        if tail != 0 {
            let last = self.words.len() - 1;
            self.words[last] &= (1u64 << tail) - 1;
        }
        if self.size == 0 {
            self.words[0] = 0;
        }
        self.summary.fill(0);
        for w in 0..self.words.len() {
            self.update_summary(w);
        }
        self.count = self.words.iter().map(|w| u64::from(w.count_ones())).sum();
    }

    /// `hbitmap_sha256()`: the SHA-256 of the words as they sit in memory, little-endian.
    pub(crate) fn sha256(&self) -> ruvm_base::Result<String> {
        let mut data = Vec::with_capacity(self.words.len() * 8);
        for w in &self.words {
            data.extend_from_slice(&w.to_le_bytes());
        }
        ruvm_crypto::hash::hash_digest(ruvm_qapi::types::QCryptoHashAlgo::Sha256, &data)
    }
}

/// `HBitmapIter`: walks the set bits of a bitmap in order.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HBitmapIter {
    /// The next bit to look at.
    pos: u64,
}

impl HBitmapIter {
    /// `hbitmap_iter_init()`: starts at item `first`.
    pub(crate) fn new(hb: &HBitmap, first: u64) -> Self {
        let pos = first >> hb.granularity;
        assert!(pos < hb.size.max(1), "hbitmap_iter_init() past the end");
        HBitmapIter { pos }
    }

    /// `hbitmap_iter_next()`: the first item of the next set bit, `None` at the end.
    pub(crate) fn next(&mut self, hb: &HBitmap) -> Option<u64> {
        let bit = hb.next_set_bit(self.pos)?;
        self.pos = bit + 1;
        Some(bit << hb.granularity)
    }
}
