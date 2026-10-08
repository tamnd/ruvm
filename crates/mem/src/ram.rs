// SPDX-License-Identifier: MIT OR Apache-2.0

//! RAM blocks, the backing of RAM, ROM and ROM device regions.
//!
//! A block's bytes are an anonymous host mapping from ruvm-sys, seen as an array of `AtomicU8`,
//! and every access copies through relaxed atomic loads and stores. That keeps this crate free of
//! unsafe code while still being sound when vCPU threads, the kernel and device threads touch the
//! same guest memory at once. The mapping is page aligned and has a stable address, so
//! accelerators can hand it to the hypervisor. Memfd and file backed blocks and the opaque copy
//! routines from spec/05 replace the storage later without changing this interface.

use std::fmt;
use std::sync::atomic::{AtomicU8, Ordering};

use arc_swap::ArcSwapOption;
use std::sync::Arc;

use crate::dirty::{DirtyBitmap, DirtyClient, DirtyMask, DirtySnapshot};
use crate::error::MemError;
use ruvm_sys::HostMemory;

/// One contiguous piece of guest RAM, `RAMBlock`.
pub struct RamBlock {
    name: String,
    mem: HostMemory,
    page_bits: u32,
    dirty: [ArcSwapOption<DirtyBitmap>; 3],
}

impl RamBlock {
    /// A zero filled block of `size` bytes named `name` (the `idstr` migration uses), with dirty
    /// tracking at `1 << page_bits` byte granularity.
    pub fn new(name: &str, size: u64, page_bits: u32) -> Result<Self, MemError> {
        let len = usize::try_from(size).map_err(|_| MemError::TooLarge(u128::from(size)))?;
        let mem = HostMemory::new(len)
            .map_err(|e| MemError::Alloc(format!("cannot set up guest memory '{name}': {e}")))?;
        Ok(Self::with_memory(name, mem, page_bits))
    }

    /// A block of shared memory, the `RAM_SHARED` branch of `qemu_ram_alloc_internal()`: the
    /// memfd CPR saved under `name` when the process before passed one on, or else a new one
    /// that is saved there for the process after. When no memfd can be had the block quietly
    /// falls back to private memory, as in QEMU.
    #[cfg(unix)]
    pub fn new_shared(name: &str, size: u64, page_bits: u32) -> Result<Self, MemError> {
        let len = usize::try_from(size).map_err(|_| MemError::TooLarge(u128::from(size)))?;
        let mem = match crate::cpr::find_fd(name, 0) {
            // Grown when the block is larger here than it was before, like `reused` does.
            Some(fd) => HostMemory::from_fd(fd, len).map_err(|e| {
                MemError::Alloc(format!("cannot map the shared memory of '{name}': {e}"))
            })?,
            None => match HostMemory::shared(name, len) {
                Ok(mem) => {
                    if let Some(fd) = mem.fd().and_then(|fd| fd.try_clone_to_owned().ok()) {
                        crate::cpr::save_fd(name, 0, fd);
                    }
                    mem
                }
                Err(_) => return Self::new(name, size, page_bits),
            },
        };
        Ok(Self::with_memory(name, mem, page_bits))
    }

    /// A block of shared memory. Without Unix there is none, so this is private memory.
    #[cfg(not(unix))]
    pub fn new_shared(name: &str, size: u64, page_bits: u32) -> Result<Self, MemError> {
        Self::new(name, size, page_bits)
    }

    fn with_memory(name: &str, mem: HostMemory, page_bits: u32) -> Self {
        RamBlock { name: name.to_string(), mem, page_bits, dirty: Default::default() }
    }

    /// Whether the block is shared memory that another process can map,
    /// `qemu_ram_is_shared()` for a block with a descriptor.
    pub fn is_shared(&self) -> bool {
        self.mem.is_shared()
    }

    fn bytes(&self) -> &[AtomicU8] {
        self.mem.as_slice()
    }

    /// The bytes, for host atomic operations on guest RAM (see `ruvm_sys::hostatomic`), the
    /// way TCG's atomic helpers work on `ramblock_ptr()`. Writes through this do not mark pages
    /// dirty.
    pub fn atomic_bytes(&self) -> &[AtomicU8] {
        self.bytes()
    }

    /// The host address of the first byte, `ramblock_ptr(block, 0)`, for accelerators that map
    /// guest RAM into the hypervisor.
    pub fn host_addr(&self) -> usize {
        self.mem.host_addr()
    }

    /// The host memory behind the block, for postcopy migration, which registers it with
    /// userfaultfd.
    pub fn host_memory(&self) -> &HostMemory {
        &self.mem
    }

    /// `ram_block_discard_range()`: drops `len` bytes of pages at `offset` so that they read
    /// as zero, and on Linux are unmapped again.
    pub fn discard_range(&self, offset: u64, len: u64) -> Result<(), MemError> {
        let o = usize::try_from(offset).map_err(|_| MemError::OutOfRange)?;
        let l = usize::try_from(len).map_err(|_| MemError::OutOfRange)?;
        self.mem.discard(o, l).map_err(|_| MemError::OutOfRange)
    }

    /// The block's name, `idstr`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The size in bytes, `used_length`.
    pub fn len(&self) -> u64 {
        self.bytes().len() as u64
    }

    /// Whether the block has no bytes.
    pub fn is_empty(&self) -> bool {
        self.bytes().is_empty()
    }

    /// Dirty tracking granularity as a shift, `TARGET_PAGE_BITS`.
    pub fn page_bits(&self) -> u32 {
        self.page_bits
    }

    fn range(&self, offset: u64, len: usize) -> Result<&[AtomicU8], MemError> {
        let start = usize::try_from(offset).map_err(|_| MemError::OutOfRange)?;
        let end = start.checked_add(len).ok_or(MemError::OutOfRange)?;
        self.bytes().get(start..end).ok_or(MemError::OutOfRange)
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

/// `qemu_ram_free()` forgets the descriptor of shared memory, so that a block made again under
/// the same name gets new memory.
#[cfg(unix)]
impl Drop for RamBlock {
    fn drop(&mut self) {
        if self.mem.is_shared() {
            crate::cpr::delete_fd(&self.name, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn shared_blocks_find_their_memory_again() {
        let a = RamBlock::new_shared("ram-shared-test", 0x2000, 12).unwrap();
        assert!(a.is_shared());
        a.write(0x1000, &[5]).unwrap();
        // What the next process does with the descriptor CPR gave it.
        let b = RamBlock::new_shared("ram-shared-test", 0x3000, 12).unwrap();
        let mut out = [0];
        b.read(0x1000, &mut out).unwrap();
        assert_eq!(out, [5]);
        drop(a);
        assert!(crate::cpr::find_fd("ram-shared-test", 0).is_none());
        assert!(!RamBlock::new("ram", 0x1000, 12).unwrap().is_shared());
    }

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
