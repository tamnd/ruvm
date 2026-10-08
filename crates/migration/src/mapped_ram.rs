// SPDX-License-Identifier: GPL-2.0-or-later

//! The mapped-ram file layout of migration/ram.c.
//!
//! With the `mapped-ram` capability a migration to a `file:` channel gives every page of a RAM
//! block a fixed place in the file instead of sending it in the stream. After the name and
//! length of each block in the block list comes a `MappedRamHeader`, which says where the bitmap
//! of the block is and where its pages start, and the stream itself goes on after the space for
//! the pages. A page written again in a later pass lands on its old copy, so the file never grows
//! past the size of RAM. The bitmap, one bit per page that is in the file, is written at the
//! end; the pages it leaves out are zero.
//!
//! The bitmap is an array of host `unsigned long` words, as QEMU writes it, so its size and byte
//! order follow the host.

use std::ffi::c_ulong;
use std::sync::atomic::{AtomicU64, Ordering};

/// `MAPPED_RAM_HDR_VERSION`.
pub const HDR_VERSION: u32 = 1;
/// `sizeof(MappedRamHeader)`, which is packed.
pub const HDR_LEN: usize = 28;
/// `MAPPED_RAM_FILE_OFFSET_ALIGNMENT`: where the pages of a block may start.
pub const FILE_OFFSET_ALIGNMENT: u64 = 0x100000;
/// `MAPPED_RAM_LOAD_BUF_SIZE`: the most a load reads at once.
pub const LOAD_BUF_SIZE: usize = 0x100000;
/// What `O_DIRECT` buffers are aligned to.
pub const DIRECT_IO_ALIGN: usize = 4096;

const LONG: usize = size_of::<c_ulong>();

/// `MappedRamHeader`, the fields in host order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// `version`, [`HDR_VERSION`].
    pub version: u32,
    /// `page_size`, `TARGET_PAGE_SIZE`.
    pub page_size: u64,
    /// `bitmap_offset`: where the bitmap of the block is in the file.
    pub bitmap_offset: u64,
    /// `pages_offset`: where the first page of the block is in the file.
    pub pages_offset: u64,
}

impl Header {
    /// The header as it is in the file, big-endian.
    pub fn to_bytes(&self) -> [u8; HDR_LEN] {
        let mut b = [0; HDR_LEN];
        b[0..4].copy_from_slice(&self.version.to_be_bytes());
        b[4..12].copy_from_slice(&self.page_size.to_be_bytes());
        b[12..20].copy_from_slice(&self.bitmap_offset.to_be_bytes());
        b[20..28].copy_from_slice(&self.pages_offset.to_be_bytes());
        b
    }

    /// The header from the file.
    pub fn parse(b: &[u8; HDR_LEN]) -> Self {
        let be64 = |at: usize| {
            let mut w = [0; 8];
            w.copy_from_slice(&b[at..at + 8]);
            u64::from_be_bytes(w)
        };
        Header {
            version: u32::from_be_bytes([b[0], b[1], b[2], b[3]]),
            page_size: be64(4),
            bitmap_offset: be64(12),
            pages_offset: be64(20),
        }
    }
}

/// `BITS_TO_LONGS(pages) * sizeof(unsigned long)`: the size of the bitmap of a block.
pub fn bitmap_size(pages: u64) -> u64 {
    let long = LONG as u64;
    pages.div_ceil(long * 8) * long
}

/// The place of one RAM block in the file, and `file_bmap`, the pages of it that were written.
/// The migration thread and the multifd threads share it.
#[derive(Debug)]
pub struct FileBlock {
    /// Where the bitmap goes.
    pub bitmap_offset: u64,
    /// Where page 0 goes.
    pub pages_offset: u64,
    pages: u64,
    bmap: Vec<AtomicU64>,
}

impl FileBlock {
    /// `mapped_ram_setup_ramblock()` for a block of `pages` pages whose bitmap starts at
    /// `bitmap_offset`: the pages start at the next aligned offset after it.
    pub fn new(bitmap_offset: u64, pages: u64) -> Self {
        let pages_offset =
            (bitmap_offset + bitmap_size(pages)).next_multiple_of(FILE_OFFSET_ALIGNMENT);
        FileBlock {
            bitmap_offset,
            pages_offset,
            pages,
            bmap: (0..pages.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// The header that describes the block.
    pub fn header(&self, page_size: u64) -> Header {
        Header {
            version: HDR_VERSION,
            page_size,
            bitmap_offset: self.bitmap_offset,
            pages_offset: self.pages_offset,
        }
    }

    /// `ramblock_set_file_bmap_atomic()`: marks `page` as in the file or not.
    pub fn set(&self, page: u64, on: bool) {
        let Some(w) = self.bmap.get((page / 64) as usize) else { return };
        let bit = 1u64 << (page % 64);
        if on {
            w.fetch_or(bit, Ordering::Relaxed);
        } else {
            w.fetch_and(!bit, Ordering::Relaxed);
        }
    }

    /// The bitmap as the file holds it.
    pub fn bitmap_bytes(&self) -> Vec<u8> {
        let words: Vec<u64> = self.bmap.iter().map(|w| w.load(Ordering::Relaxed)).collect();
        bitmap_to_bytes(&words, self.pages)
    }
}

/// `bitmap` in the host layout of an `unsigned long` array, [`bitmap_size`] bytes.
pub fn bitmap_to_bytes(bitmap: &[u64], pages: u64) -> Vec<u8> {
    let size = bitmap_size(pages) as usize;
    let mut out = Vec::with_capacity(size + 8);
    for &w in bitmap {
        if LONG == 8 {
            out.extend_from_slice(&w.to_ne_bytes());
        } else {
            out.extend_from_slice(&(w as u32).to_ne_bytes());
            out.extend_from_slice(&((w >> 32) as u32).to_ne_bytes());
        }
    }
    out.resize(size, 0);
    out
}

/// The bitmap of `pages` pages from the bytes of the file, with the bits past the last page
/// cleared.
pub fn bitmap_from_bytes(bytes: &[u8], pages: u64) -> Vec<u64> {
    let mut words = vec![0u64; pages.div_ceil(64) as usize];
    for (i, chunk) in bytes.chunks(LONG).enumerate() {
        let mut b = [0u8; 8];
        b[..chunk.len()].copy_from_slice(chunk);
        let v = if LONG == 8 {
            u64::from_ne_bytes(b)
        } else {
            u64::from(u32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        };
        let bit = i * LONG * 8;
        if let Some(w) = words.get_mut(bit / 64) {
            *w |= v << (bit % 64);
        }
    }
    if pages % 64 != 0 {
        if let Some(last) = words.last_mut() {
            *last &= (1u64 << (pages % 64)) - 1;
        }
    }
    words
}

/// The runs of set bits in `bitmap`, as `(first, end)` page ranges.
pub fn runs(bitmap: &[u64], pages: u64) -> Vec<(u64, u64)> {
    let test = |p: u64| bitmap[(p / 64) as usize] & (1 << (p % 64)) != 0;
    let mut out = Vec::new();
    let mut p = 0;
    while p < pages {
        // Skip whole clear words quickly.
        if p % 64 == 0 && bitmap[(p / 64) as usize] == 0 {
            p += 64;
            continue;
        }
        if !test(p) {
            p += 1;
            continue;
        }
        let start = p;
        while p < pages && test(p) {
            p += 1;
        }
        out.push((start, p));
    }
    out
}

/// A buffer whose start is aligned for `O_DIRECT`.
#[derive(Debug, Default)]
pub struct AlignedBuf {
    buf: Vec<u8>,
}

impl AlignedBuf {
    /// `len` bytes starting at an aligned address. What was in them before is undefined.
    pub fn get(&mut self, len: usize) -> &mut [u8] {
        if self.buf.len() < len + DIRECT_IO_ALIGN {
            self.buf.resize(len + DIRECT_IO_ALIGN, 0);
        }
        let off = self.buf.as_ptr().align_offset(DIRECT_IO_ALIGN);
        // align_offset() may give up, in theory; an unaligned buffer only fails with O_DIRECT.
        let off = if off < DIRECT_IO_ALIGN { off } else { 0 };
        &mut self.buf[off..off + len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_and_offsets() {
        let fb = FileBlock::new(0x1234, 70);
        assert_eq!(fb.pages_offset, 0x100000);
        let h = fb.header(4096);
        let b = h.to_bytes();
        assert_eq!(&b[..4], &[0, 0, 0, 1]);
        assert_eq!(&b[4..12], &4096u64.to_be_bytes());
        assert_eq!(Header::parse(&b), h);
        // A bitmap that runs over the alignment pushes the pages one step further.
        let fb = FileBlock::new(FILE_OFFSET_ALIGNMENT - 8, 1024);
        assert_eq!(fb.pages_offset, 2 * FILE_OFFSET_ALIGNMENT);
    }

    #[test]
    fn bitmap_layout() {
        assert_eq!(bitmap_size(1), LONG as u64);
        assert_eq!(bitmap_size(65), if LONG == 8 { 16 } else { 12 });
        let fb = FileBlock::new(0, 70);
        for p in [0, 1, 63, 64, 69] {
            fb.set(p, true);
        }
        fb.set(1, false);
        let bytes = fb.bitmap_bytes();
        assert_eq!(bytes.len() as u64, bitmap_size(70));
        if cfg!(target_endian = "little") {
            assert_eq!(bytes[0], 1);
            assert_eq!(bytes[7], 0x80);
            assert_eq!(bytes[8], 0x21);
        }
        let words = bitmap_from_bytes(&bytes, 70);
        assert_eq!(runs(&words, 70), [(0, 1), (63, 65), (69, 70)]);
        assert_eq!(runs(&[!0, !0], 100), [(0, 100)]);
        assert!(runs(&[0, 0], 128).is_empty());
    }

    #[test]
    fn aligned() {
        let mut b = AlignedBuf::default();
        for len in [4096, 1 << 20, 8192] {
            let s = b.get(len);
            assert_eq!(s.len(), len);
            assert_eq!(s.as_ptr() as usize % DIRECT_IO_ALIGN, 0);
        }
    }
}
