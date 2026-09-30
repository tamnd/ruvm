// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dirty page tracking, include/system/ram_addr.h and the dirty parts of system/physmem.c.
//!
//! QEMU keeps one bitmap per client over the whole `ram_addr_t` space. Here each [`RamBlock`]
//! carries its own, one per client, and a client's bitmap exists only while logging for it is on.
//! Producers set bits with a relaxed `fetch_or`; consumers take whole words with `swap(0)`.
//!
//! [`RamBlock`]: crate::RamBlock

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

/// A dirty memory client, `DIRTY_MEMORY_VGA`, `DIRTY_MEMORY_CODE` and `DIRTY_MEMORY_MIGRATION`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DirtyClient {
    /// Display devices, which redraw what changed.
    Vga = 0,
    /// The JIT, which throws away translations of pages that were written.
    Code = 1,
    /// Live migration, which resends pages that were written.
    Migration = 2,
}

impl DirtyClient {
    /// All clients in QEMU's order, `DIRTY_MEMORY_NUM` of them.
    pub const ALL: [DirtyClient; 3] = [DirtyClient::Vga, DirtyClient::Code, DirtyClient::Migration];

    /// The client's bit in a [`DirtyMask`].
    pub const fn mask(self) -> DirtyMask {
        DirtyMask(1 << self as u8)
    }
}

/// A set of dirty clients, one bit per [`DirtyClient`], like a region's `dirty_log_mask`.
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash)]
pub struct DirtyMask(u8);

impl DirtyMask {
    /// No client.
    pub const NONE: DirtyMask = DirtyMask(0);
    /// Every client.
    pub const ALL: DirtyMask = DirtyMask(0b111);

    /// A mask from raw bits. Bits above the three clients are dropped.
    pub const fn from_bits(bits: u8) -> Self {
        DirtyMask(bits & 0b111)
    }

    /// The raw bits.
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Whether `client` is in the set.
    pub const fn contains(self, client: DirtyClient) -> bool {
        self.0 & (1 << client as u8) != 0
    }

    /// Whether the set is empty.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The set with `client` added.
    pub const fn with(self, client: DirtyClient) -> Self {
        DirtyMask(self.0 | 1 << client as u8)
    }

    /// The set with `client` removed.
    pub const fn without(self, client: DirtyClient) -> Self {
        DirtyMask(self.0 & !(1 << client as u8))
    }

    /// The union.
    pub const fn union(self, other: DirtyMask) -> Self {
        DirtyMask(self.0 | other.0)
    }

    /// The clients in `self` that are not in `other`.
    pub const fn difference(self, other: DirtyMask) -> Self {
        DirtyMask(self.0 & !other.0)
    }
}

impl fmt::Debug for DirtyMask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let set: Vec<DirtyClient> =
            DirtyClient::ALL.into_iter().filter(|c| self.contains(*c)).collect();
        f.debug_tuple("DirtyMask").field(&set).finish()
    }
}

/// One client's bitmap over the pages of one block.
pub(crate) struct DirtyBitmap {
    words: Box<[AtomicU64]>,
    pages: u64,
}

impl DirtyBitmap {
    pub(crate) fn new(pages: u64) -> Self {
        let words = pages.div_ceil(64) as usize;
        DirtyBitmap { words: (0..words).map(|_| AtomicU64::new(0)).collect(), pages }
    }

    /// Calls `f(word, mask)` for each word the page range `[first, last]` touches.
    fn for_words(&self, first: u64, last: u64, mut f: impl FnMut(&AtomicU64, u64)) {
        let last = last.min(self.pages.saturating_sub(1));
        if self.pages == 0 || first > last {
            return;
        }
        let (fw, lw) = (first / 64, last / 64);
        for w in fw..=lw {
            let lo = if w == fw { first % 64 } else { 0 };
            let hi = if w == lw { last % 64 } else { 63 };
            let mask = (u64::MAX >> (63 - hi)) & (u64::MAX << lo);
            f(&self.words[w as usize], mask);
        }
    }

    pub(crate) fn set(&self, first: u64, last: u64) {
        self.for_words(first, last, |w, m| {
            // Skip the read-modify-write when the bits are already there. The common case under
            // a guest that keeps writing the same pages is a plain load.
            if w.load(Ordering::Relaxed) & m != m {
                w.fetch_or(m, Ordering::Relaxed);
            }
        });
    }

    pub(crate) fn any(&self, first: u64, last: u64) -> bool {
        let mut any = false;
        self.for_words(first, last, |w, m| any |= w.load(Ordering::Acquire) & m != 0);
        any
    }

    pub(crate) fn test_and_clear(&self, first: u64, last: u64) -> bool {
        let mut any = false;
        self.for_words(first, last, |w, m| {
            any |= if m == u64::MAX {
                w.swap(0, Ordering::AcqRel) != 0
            } else {
                w.fetch_and(!m, Ordering::AcqRel) & m != 0
            };
        });
        any
    }

    pub(crate) fn take_words(&self, first_word: usize, out: &mut [u64]) -> usize {
        let words = self.words.get(first_word..).unwrap_or(&[]);
        let n = words.len().min(out.len());
        for (o, w) in out.iter_mut().zip(&words[..n]) {
            *o = w.swap(0, Ordering::AcqRel);
        }
        n
    }

    pub(crate) fn count(&self) -> u64 {
        self.words.iter().map(|w| u64::from(w.load(Ordering::Acquire).count_ones())).sum()
    }

    pub(crate) fn snapshot_words(&self, first_word: u64, last_word: u64) -> Vec<u64> {
        (first_word..=last_word)
            .map(|w| self.words.get(w as usize).map_or(0, |w| w.swap(0, Ordering::AcqRel)))
            .collect()
    }
}

/// The bits of one client over a range of a block, taken and cleared in one step, like
/// `DirtyBitmapSnapshot`.
///
/// The snapshot covers whole 64 page words, so it may be larger than what was asked for, and
/// the extra pages were cleared too. That is what QEMU does and display devices are written for
/// it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirtySnapshot {
    start: u64,
    end: u64,
    page_bits: u32,
    words: Vec<u64>,
}

impl DirtySnapshot {
    pub(crate) fn new(start: u64, end: u64, page_bits: u32, words: Vec<u64>) -> Self {
        DirtySnapshot { start, end, page_bits, words }
    }

    /// The offset in the block of the first byte covered.
    pub fn start(&self) -> u64 {
        self.start
    }

    /// The offset in the block just past the last byte covered.
    pub fn end(&self) -> u64 {
        self.end
    }

    /// `memory_region_snapshot_get_dirty()`: whether any page in `[offset, offset + len)` was
    /// dirty. `offset` is relative to the block and must lie inside the snapshot.
    pub fn get_dirty(&self, offset: u64, len: u64) -> bool {
        assert!(
            offset >= self.start && offset.saturating_add(len) <= self.end,
            "outside the snapshot"
        );
        if len == 0 {
            return false;
        }
        let first = (offset - self.start) >> self.page_bits;
        let last = (offset + len - 1 - self.start) >> self.page_bits;
        (first..=last).any(|p| self.words[(p / 64) as usize] & (1 << (p % 64)) != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_cross_words() {
        let b = DirtyBitmap::new(200);
        b.set(60, 130);
        assert!(b.any(60, 60) && b.any(130, 130) && b.any(100, 100));
        assert!(!b.any(0, 59) && !b.any(131, 199));
        assert_eq!(b.count(), 71);
        assert!(b.test_and_clear(64, 127));
        assert_eq!(b.count(), 7);
        assert!(!b.test_and_clear(64, 127));
        let mut out = [0u64; 4];
        assert_eq!(b.take_words(0, &mut out), 4);
        assert_eq!(out[0], 0xf << 60);
        assert_eq!(out[2], 0b111);
        assert_eq!(b.count(), 0);
    }

    #[test]
    fn masks() {
        let m = DirtyMask::NONE.with(DirtyClient::Vga).with(DirtyClient::Migration);
        assert_eq!(m.bits(), 0b101);
        assert!(m.contains(DirtyClient::Vga) && !m.contains(DirtyClient::Code));
        assert_eq!(m.difference(DirtyClient::Vga.mask()), DirtyClient::Migration.mask());
        assert_eq!(format!("{m:?}"), "DirtyMask([Vga, Migration])");
    }
}
