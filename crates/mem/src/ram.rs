// SPDX-License-Identifier: MIT OR Apache-2.0

//! RAM blocks, the backing of RAM, ROM and ROM device regions.
//!
//! For now a block's bytes are an array of `AtomicU8` on the Rust heap and every access copies
//! through relaxed atomic loads and stores. That keeps the crate free of unsafe code while still
//! being sound when vCPU threads and device threads touch the same guest memory at once, which a
//! plain `Vec<u8>` behind a shared reference would not be. Host mappings (mmap, memfd, shared
//! files) and the opaque copy routines from spec/05 replace the storage later without changing
//! this interface.

use std::fmt;
use std::sync::atomic::{AtomicU8, Ordering};

use arc_swap::ArcSwapOption;
use std::sync::Arc;

use crate::dirty::{DirtyBitmap, DirtyClient, DirtyMask, DirtySnapshot};
use crate::error::MemError;

/// One contiguous piece of guest RAM, `RAMBlock`.
pub struct RamBlock {
    name: String,
    bytes: Box<[AtomicU8]>,
    page_bits: u32,
    dirty: [ArcSwapOption<DirtyBitmap>; 3],
}

impl RamBlock {
    /// A zero filled block of `size` bytes named `name` (the `idstr` migration uses), with dirty
    /// tracking at `1 << page_bits` byte granularity.
    pub fn new(name: &str, size: u64, page_bits: u32) -> Result<Self, MemError> {
        let len = usize::try_from(size).map_err(|_| MemError::TooLarge(u128::from(size)))?;
        Ok(RamBlock {
            name: name.to_string(),
            bytes: (0..len).map(|_| AtomicU8::new(0)).collect(),
            page_bits,
            dirty: Default::default(),
        })
    }

    /// The block's name, `idstr`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The size in bytes, `used_length`.
    pub fn len(&self) -> u64 {
        self.bytes.len() as u64
    }

    /// Whether the block has no bytes.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Dirty tracking granularity as a shift, `TARGET_PAGE_BITS`.
    pub fn page_bits(&self) -> u32 {
        self.page_bits
    }

    fn range(&self, offset: u64, len: usize) -> Result<&[AtomicU8], MemError> {
        let start = usize::try_from(offset).map_err(|_| MemError::OutOfRange)?;
        let end = start.checked_add(len).ok_or(MemError::OutOfRange)?;
        self.bytes.get(start..end).ok_or(MemError::OutOfRange)
    }

    /// Copies bytes at `offset` into `buf`.
    pub fn read(&self, offset: u64, buf: &mut [u8]) -> Result<(), MemError> {
        let src = self.range(offset, buf.len())?;
        for (d, s) in buf.iter_mut().zip(src) {
            *d = s.load(Ordering::Relaxed);
        }
        Ok(())
    }

    /// Copies `buf` to `offset`. This does not mark anything dirty; see
    /// [`RamBlock::set_dirty`].
    pub fn write(&self, offset: u64, buf: &[u8]) -> Result<(), MemError> {
        let dst = self.range(offset, buf.len())?;
        for (d, s) in dst.iter().zip(buf) {
            d.store(*s, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Sets `len` bytes at `offset` to `byte`.
    pub fn fill(&self, offset: u64, len: u64, byte: u8) -> Result<(), MemError> {
        let len = usize::try_from(len).map_err(|_| MemError::OutOfRange)?;
        for d in self.range(offset, len)? {
            d.store(byte, Ordering::Relaxed);
        }
        Ok(())
    }

    fn pages(&self, offset: u64, len: u64) -> Option<(u64, u64)> {
        if len == 0 {
            return None;
        }
        let last = offset.saturating_add(len - 1);
        Some((offset >> self.page_bits, last >> self.page_bits))
    }

    /// Starts keeping a bitmap for `client`. The bitmap starts clean. Starting twice keeps the
    /// bits already collected.
    pub fn start_dirty_log(&self, client: DirtyClient) {
        let slot = &self.dirty[client as usize];
        if slot.load().is_none() {
            let pages = self.len().div_ceil(1 << self.page_bits);
            slot.compare_and_swap(
                &None::<Arc<DirtyBitmap>>,
                Some(Arc::new(DirtyBitmap::new(pages))),
            );
        }
    }

    /// Drops the bitmap for `client`.
    pub fn stop_dirty_log(&self, client: DirtyClient) {
        self.dirty[client as usize].store(None);
    }

    /// Whether a bitmap is kept for `client`.
    pub fn is_dirty_logging(&self, client: DirtyClient) -> bool {
        self.dirty[client as usize].load().is_some()
    }

    /// The clients a bitmap is kept for.
    pub fn dirty_log_mask(&self) -> DirtyMask {
        DirtyClient::ALL
            .into_iter()
            .filter(|c| self.is_dirty_logging(*c))
            .fold(DirtyMask::NONE, DirtyMask::with)
    }

    /// `physical_memory_set_dirty_range()`: marks the pages of `[offset, offset + len)` dirty for
    /// each client in `mask` that has a bitmap.
    pub fn set_dirty(&self, offset: u64, len: u64, mask: DirtyMask) {
        let Some((first, last)) = self.pages(offset, len) else { return };
        for client in DirtyClient::ALL {
            if mask.contains(client) {
                if let Some(b) = &*self.dirty[client as usize].load() {
                    b.set(first, last);
                }
            }
        }
    }

    /// `physical_memory_get_dirty()`: whether any page of `[offset, offset + len)` is dirty for
    /// `client`. Always false for a client without a bitmap.
    pub fn get_dirty(&self, offset: u64, len: u64, client: DirtyClient) -> bool {
        let Some((first, last)) = self.pages(offset, len) else { return false };
        self.dirty[client as usize].load().as_ref().is_some_and(|b| b.any(first, last))
    }

    /// `physical_memory_test_and_clear_dirty()`: clears the pages of `[offset, offset + len)` for
    /// `client` and says whether any was dirty.
    pub fn test_and_clear_dirty(&self, offset: u64, len: u64, client: DirtyClient) -> bool {
        let Some((first, last)) = self.pages(offset, len) else { return false };
        self.dirty[client as usize].load().as_ref().is_some_and(|b| b.test_and_clear(first, last))
    }

    /// The number of dirty pages for `client`.
    pub fn dirty_pages(&self, client: DirtyClient) -> u64 {
        self.dirty[client as usize].load().as_ref().map_or(0, |b| b.count())
    }

    /// Takes whole bitmap words for `client`, starting at word `first_word`, clearing them. Bit
    /// `i` of `out[j]` is page `(first_word + j) * 64 + i`. Returns how many words were written,
    /// zero if there is no bitmap.
    pub fn take_dirty_words(
        &self,
        client: DirtyClient,
        first_word: usize,
        out: &mut [u64],
    ) -> usize {
        self.dirty[client as usize].load().as_ref().map_or(0, |b| b.take_words(first_word, out))
    }

    /// `physical_memory_snapshot_and_clear_dirty()` for this block: takes and clears the bits
    /// covering `[offset, offset + len)`, widened to whole 64 page words.
    pub fn snapshot_and_clear_dirty(
        &self,
        offset: u64,
        len: u64,
        client: DirtyClient,
    ) -> DirtySnapshot {
        let align = 64u64 << self.page_bits;
        let start = offset / align * align;
        let end = offset.saturating_add(len).div_ceil(align).saturating_mul(align);
        let words = if end > start {
            let (fw, lw) = (start / align, (end - 1) / align);
            match &*self.dirty[client as usize].load() {
                Some(b) => b.snapshot_words(fw, lw),
                None => vec![0; (lw - fw + 1) as usize],
            }
        } else {
            Vec::new()
        };
        DirtySnapshot::new(start, end, self.page_bits, words)
    }
}

impl fmt::Debug for RamBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RamBlock")
            .field("name", &self.name)
            .field("len", &self.len())
            .field("dirty_log_mask", &self.dirty_log_mask())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_round_trip_and_bounds_hold() {
        let b = RamBlock::new("ram", 0x3000, 12).unwrap();
        b.write(0xffe, &[1, 2, 3, 4]).unwrap();
        let mut out = [0; 4];
        b.read(0xffe, &mut out).unwrap();
        assert_eq!(out, [1, 2, 3, 4]);
        assert_eq!(b.write(0x2fff, &[0, 0]), Err(MemError::OutOfRange));
        assert_eq!(b.read(u64::MAX, &mut out), Err(MemError::OutOfRange));
        b.fill(0, 2, 0xaa).unwrap();
        b.read(0, &mut out[..2]).unwrap();
        assert_eq!(out[..2], [0xaa, 0xaa]);
    }

    #[test]
    fn dirty_bits_only_for_logging_clients() {
        let b = RamBlock::new("ram", 0x10000, 12).unwrap();
        b.set_dirty(0, 0x1000, DirtyMask::ALL);
        assert!(!b.get_dirty(0, 0x1000, DirtyClient::Vga));
        b.start_dirty_log(DirtyClient::Vga);
        assert_eq!(b.dirty_log_mask(), DirtyClient::Vga.mask());
        b.set_dirty(0x1fff, 2, DirtyMask::ALL);
        assert!(b.get_dirty(0x1000, 1, DirtyClient::Vga));
        assert!(b.get_dirty(0x2000, 0x1000, DirtyClient::Vga));
        assert!(!b.get_dirty(0x3000, 0x1000, DirtyClient::Vga));
        assert_eq!(b.dirty_pages(DirtyClient::Vga), 2);
        let snap = b.snapshot_and_clear_dirty(0x1000, 0x1000, DirtyClient::Vga);
        assert_eq!((snap.start(), snap.end()), (0, 0x40000));
        assert!(
            snap.get_dirty(0x1000, 1) && snap.get_dirty(0x2000, 1) && !snap.get_dirty(0x3000, 1)
        );
        assert_eq!(b.dirty_pages(DirtyClient::Vga), 0);
        b.set_dirty(0x5000, 1, DirtyClient::Vga.mask());
        assert!(b.test_and_clear_dirty(0x4000, 0x2000, DirtyClient::Vga));
        assert!(!b.test_and_clear_dirty(0x4000, 0x2000, DirtyClient::Vga));
        b.stop_dirty_log(DirtyClient::Vga);
        assert!(!b.is_dirty_logging(DirtyClient::Vga));
    }
}
