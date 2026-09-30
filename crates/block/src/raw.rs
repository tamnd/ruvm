// SPDX-License-Identifier: GPL-2.0-or-later

//! The `raw` format driver from block/raw-format.c: its `file` child's bytes as they are, or a
//! window of them given by `offset` and `size`.

use std::io;
use std::sync::atomic::{AtomicU64, Ordering};

use ruvm_base::{Error, Result};

use crate::node::{BDRV_SECTOR_SIZE, Driver, Node, errno};
use crate::perm::{BLK_PERM_RESIZE, BLK_PERM_WRITE};

/// `BLOCK_PROBE_BUF_SIZE`.
pub(crate) const BLOCK_PROBE_BUF_SIZE: usize = 512;

/// `BDRVRawState`.
#[derive(Debug)]
pub(crate) struct RawDriver {
    offset: u64,
    has_size: bool,
    /// The size of the window. Without `size` this follows the child, see `getlength`.
    size: AtomicU64,
    /// `bs->probed`: the format was guessed, so block 0 is guarded.
    probed: bool,
}

/// `raw_read_options()` and `raw_apply_options()`. `file_len` is the child's
/// `bdrv_getlength()`.
pub(crate) fn raw_open(
    offset: Option<i64>,
    size: Option<i64>,
    file_len: u64,
    probed: bool,
) -> Result<RawDriver> {
    // The QAPI type is `int`, QEMU reads the value back as a `QEMU_OPT_SIZE`.
    let non_negative = |v: i64, name: &str| {
        u64::try_from(v).map_err(|_| {
            Error::generic(format!("Parameter '{name}' expects a non-negative number below 2^64"))
        })
    };
    let offset = offset.map_or(Ok(0), |v| non_negative(v, "offset"))?;
    let has_size = size.is_some();
    let size = size.map_or(Ok(0), |v| non_negative(v, "size"))?;

    if offset > file_len {
        return Err(Error::generic(format!(
            "Offset ({offset}) cannot be greater than size of the containing file ({file_len})"
        )));
    }
    if has_size && file_len - offset < size {
        return Err(Error::generic(format!(
            "The sum of offset ({offset}) and size ({size}) has to be smaller or equal to the  \
             actual size of the containing file ({file_len})"
        )));
    }
    if has_size && size % BDRV_SECTOR_SIZE != 0 {
        return Err(Error::generic(format!(
            "Specified size is not multiple of {BDRV_SECTOR_SIZE}"
        )));
    }
    let size = if has_size { size } else { file_len - offset };
    Ok(RawDriver { offset, has_size, size: AtomicU64::new(size), probed })
}

impl RawDriver {
    /// Whether the node is a filter over its child, which decides the child's role.
    fn is_filter(&self) -> bool {
        self.offset == 0 && !self.has_size
    }

    /// `raw_adjust_offset()`.
    fn adjust_offset(&self, offset: u64, bytes: u64, is_write: bool) -> io::Result<u64> {
        let size = self.size.load(Ordering::Relaxed);
        if self.has_size && (offset > size || bytes > size - offset) {
            // Do not touch anything outside the window the options gave.
            return Err(errno(if is_write { libc::ENOSPC } else { libc::EINVAL }));
        }
        offset.checked_add(self.offset).filter(|&o| o <= i64::MAX as u64).ok_or(errno(libc::EINVAL))
    }
}

impl Driver for RawDriver {
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let off = self.adjust_offset(offset, buf.len() as u64, false)?;
        bs.file().pread(off, buf)
    }

    /// `raw_co_pwritev()`. A probed image refuses a write that would make block 0 look like
    /// another format, so a guest cannot turn its disk into, say, a qcow2 image with a backing
    /// file pointing at a host file. QEMU makes such images use 512 byte requests. Here a
    /// partial write to block 0 is merged with what is on disk and the result is checked.
    fn pwrite(&self, bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        if self.probed && offset < BLOCK_PROBE_BUF_SIZE as u64 {
            let mut block0 = [0u8; BLOCK_PROBE_BUF_SIZE];
            let end = (offset as usize + buf.len()).min(BLOCK_PROBE_BUF_SIZE);
            if offset != 0 || end != BLOCK_PROBE_BUF_SIZE {
                self.pread(bs, 0, &mut block0)?;
            }
            block0[offset as usize..end].copy_from_slice(&buf[..end - offset as usize]);
            if probe_format(&block0) != "raw" {
                return Err(errno(libc::EPERM));
            }
        }
        let off = self.adjust_offset(offset, buf.len() as u64, true)?;
        bs.file().pwrite(off, buf)
    }

    fn pwrite_zeroes(&self, bs: &Node, offset: u64, bytes: u64, may_unmap: bool) -> io::Result<()> {
        let off = self.adjust_offset(offset, bytes, true)?;
        bs.file().pwrite_zeroes(off, bytes, may_unmap)
    }

    fn pdiscard(&self, bs: &Node, offset: u64, bytes: u64) -> io::Result<()> {
        let off = self.adjust_offset(offset, bytes, true)?;
        bs.file().pdiscard(off, bytes)
    }

    /// `raw_co_getlength()`: what is left of the child after `offset`, capped at `size`.
    fn getlength(&self, bs: &Node) -> io::Result<u64> {
        let len = bs.file().getlength()?;
        let size = if len < self.offset {
            0
        } else if self.has_size {
            self.size.load(Ordering::Relaxed).min(len - self.offset)
        } else {
            len - self.offset
        };
        self.size.store(size, Ordering::Relaxed);
        Ok(size)
    }

    /// `raw_co_truncate()`.
    fn truncate(&self, bs: &Node, len: u64) -> Result<()> {
        if self.has_size {
            return Err(Error::generic("Cannot resize fixed-size raw disks"));
        }
        if i64::MAX as u64 - len < self.offset {
            return Err(Error::generic("Disk size too large for the chosen offset"));
        }
        self.size.store(len, Ordering::Relaxed);
        bs.file().truncate(len + self.offset)
    }

    /// `raw_child_perm()`: a filter without a window, a data child with one. A data child does
    /// not let others resize the file under the window, and asks for write and resize only if
    /// the parents do.
    fn child_perm(&self, index: usize, perm: u64, shared: u64) -> (u64, u64) {
        let (p, s) = default_child_perm(index, perm, shared);
        if self.is_filter() {
            return (p, s);
        }
        let mask = BLK_PERM_WRITE | BLK_PERM_RESIZE;
        ((p & !mask) | (perm & mask), s & !BLK_PERM_RESIZE)
    }
}

/// `bdrv_filter_default_perms()`, which `bdrv_default_perms()` starts from for both roles raw
/// uses. The data role then only takes away `resize` from what is shared, and its extra `write`
/// for `write unchanged` is dropped again by `raw_child_perm()`.
fn default_child_perm(_index: usize, perm: u64, shared: u64) -> (u64, u64) {
    use crate::perm::{DEFAULT_PERM_PASSTHROUGH, DEFAULT_PERM_UNCHANGED};
    (perm & DEFAULT_PERM_PASSTHROUGH, (shared & DEFAULT_PERM_PASSTHROUGH) | DEFAULT_PERM_UNCHANGED)
}

/// `bdrv_probe_all()` over the formats QEMU builds by default, by their magic numbers only.
/// Anything nobody claims is `raw`, whose probe always scores 1.
pub(crate) fn probe_format(buf: &[u8]) -> &'static str {
    let be32 =
        |off: usize| buf.get(off..off + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]));
    let le32 =
        |off: usize| buf.get(off..off + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    let starts = |magic: &[u8]| buf.starts_with(magic);
    if starts(b"QFI\xfb") {
        return match be32(4) {
            Some(1) => "qcow",
            Some(v) if v >= 2 => "qcow2",
            _ => "raw",
        };
    }
    if starts(b"QED\0") {
        return "qed";
    }
    if le32(0x40) == Some(0xbeda_107f) {
        return "vdi";
    }
    if starts(b"KDMV") || starts(b"COWD") || starts(b"# Disk DescriptorFile") {
        return "vmdk";
    }
    if starts(b"vhdxfile") {
        return "vhdx";
    }
    if starts(b"conectix") {
        return "vpc";
    }
    if starts(b"LUKS\xba\xbe") {
        return "luks";
    }
    if starts(b"Bochs Virtual HD Image") {
        return "bochs";
    }
    if starts(b"WithoutFreeSpace") || starts(b"WithouFreSpacExt") {
        return "parallels";
    }
    if starts(b"#!/bin/sh\n#V2.0 Format\n") {
        return "cloop";
    }
    "raw"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options() {
        let e = raw_open(Some(4096), None, 1024, false).unwrap_err();
        assert_eq!(
            e.message(),
            "Offset (4096) cannot be greater than size of the containing file (1024)"
        );
        let e = raw_open(Some(512), Some(1024), 1024, false).unwrap_err();
        assert_eq!(
            e.message(),
            "The sum of offset (512) and size (1024) has to be smaller or equal to the  actual \
             size of the containing file (1024)"
        );
        let e = raw_open(None, Some(100), 1024, false).unwrap_err();
        assert_eq!(e.message(), "Specified size is not multiple of 512");
        let e = raw_open(Some(-1), None, 1024, false).unwrap_err();
        assert_eq!(e.message(), "Parameter 'offset' expects a non-negative number below 2^64");
        let r = raw_open(Some(512), Some(512), 1024, false).unwrap();
        assert_eq!(r.adjust_offset(0, 512, false).unwrap(), 512);
        assert_eq!(r.adjust_offset(1, 512, true).unwrap_err().raw_os_error(), Some(libc::ENOSPC));
        assert_eq!(r.adjust_offset(1, 512, false).unwrap_err().raw_os_error(), Some(libc::EINVAL));
    }

    #[test]
    fn probing() {
        let mut b = [0u8; 512];
        assert_eq!(probe_format(&b), "raw");
        b[..8].copy_from_slice(b"QFI\xfb\0\0\0\x03");
        assert_eq!(probe_format(&b), "qcow2");
        b[..8].copy_from_slice(b"conectix");
        assert_eq!(probe_format(&b), "vpc");
    }
}
