// SPDX-License-Identifier: GPL-2.0-or-later

//! The `raw` format driver from block/raw-format.c: its `file` child's bytes as they are, or a
//! window of them given by `offset` and `size`.

use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use ruvm_base::{Error, Result};
use ruvm_qapi::types::BlockdevOptionsU;

use crate::drivers::{DriverDef, OpenArgs};
use crate::node::{
    BDRV_BLOCK_OFFSET_VALID, BDRV_BLOCK_RAW, BDRV_CHILD_DATA, BDRV_CHILD_FILTERED,
    BDRV_CHILD_PRIMARY, BDRV_REQ_FUA, BDRV_REQ_MAY_UNMAP, BDRV_REQ_NO_FALLBACK,
    BDRV_REQ_WRITE_UNCHANGED, BDRV_SECTOR_SIZE, BlockLimits, BlockStatus, Driver, Node,
    ReopenState, errno,
};
use crate::perm::{BLK_PERM_RESIZE, BLK_PERM_WRITE};
use crate::probe::{BLOCK_PROBE_BUF_SIZE, probe_format, raw_probe};

/// The `raw` format driver, `bdrv_raw`.
pub(crate) static RAW: DriverDef = DriverDef::format("raw", raw_open_node)
    .with_probe(raw_probe)
    .with_create_opts(crate::tools::raw_co_create_opts)
    .with_create_opts_list(&crate::tools::RAW_CREATE_OPTS)
    .with_measure(crate::tools::raw_measure)
    .with_mutable_opts(&["offset", "size"])
    .with_strong_opts(&["offset", "size"]);

/// Takes a size option out of the reopen options, `qemu_opt_get_size_del()` after
/// `qemu_opts_absorb_qdict()`.
fn take_size_opt(options: &mut ruvm_qapi::QDict, name: &str) -> Result<Option<i64>> {
    use ruvm_qapi::QValue;
    match options.remove(name) {
        None => Ok(None),
        Some(QValue::Int(i)) => Ok(Some(i)),
        Some(QValue::Uint(u)) => Ok(Some(u as i64)),
        Some(QValue::Str(s)) => {
            ruvm_qapi::visit::parse_option_size(name, &s).map(|v| Some(v as i64))
        }
        Some(_) => {
            Err(Error::generic(format!("Invalid parameter type for '{name}', expected: size")))
        }
    }
}

/// `raw_open()`: the child is a filtered child without a window, a data child with one.
fn raw_open_node(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::Raw(o) = opts else { unreachable!("raw driver with other options") };
    let role = if o.offset.is_some() || o.size.is_some() {
        BDRV_CHILD_DATA | BDRV_CHILD_PRIMARY
    } else {
        BDRV_CHILD_FILTERED | BDRV_CHILD_PRIMARY
    };
    let child = args.open_child(*o.file, "file", role)?;
    let probed = args.is_probed();
    if probed && !args.flags.read_only {
        child.refresh_filename();
        let name = child.meta.lock().unwrap().filename.clone();
        args.warn(format!(
            "WARNING: Image format was not specified for '{name}' and probing guessed raw.\n         \
             Automatically detecting the format is dangerous for raw images, write operations \
             on block 0 will be restricted.\n         Specify the 'raw' format explicitly to \
             remove the restrictions."
        ));
    }
    let len = child.getlength().map_err(|e| Error::from_io("Could not get image size", e))?;
    Ok(Box::new(raw_open(o.offset, o.size, len, probed)?))
}

/// `BDRVRawState`.
#[derive(Debug)]
pub(crate) struct RawDriver {
    /// Reopen can change the window, so it sits in atomics.
    offset: AtomicU64,
    has_size: AtomicBool,
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
    Ok(RawDriver {
        offset: AtomicU64::new(offset),
        has_size: AtomicBool::new(has_size),
        size: AtomicU64::new(size),
        probed,
    })
}

impl RawDriver {
    fn off(&self) -> u64 {
        self.offset.load(Ordering::Relaxed)
    }

    fn has_size(&self) -> bool {
        self.has_size.load(Ordering::Relaxed)
    }

    /// The block 0 check of `raw_co_pwritev()` for probed images: the write may not make
    /// the image look like another format. A write that covers only part of block 0 is
    /// merged with what is there before the check.
    fn check_block0(&self, bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        if !self.probed || offset >= BLOCK_PROBE_BUF_SIZE as u64 || buf.is_empty() {
            return Ok(());
        }
        let mut block0 = [0u8; BLOCK_PROBE_BUF_SIZE];
        let end = (offset as usize + buf.len()).min(BLOCK_PROBE_BUF_SIZE);
        if offset != 0 || end != BLOCK_PROBE_BUF_SIZE {
            Driver::pread(self, bs, 0, &mut block0)?;
        }
        block0[offset as usize..end].copy_from_slice(&buf[..end - offset as usize]);
        if probe_format(&block0) != "raw" {
            return Err(errno(libc::EPERM));
        }
        Ok(())
    }

    /// Whether the node is a filter over its child, which decides the child's role.
    fn is_filter(&self) -> bool {
        self.off() == 0 && !self.has_size()
    }

    /// `raw_adjust_offset()`.
    fn adjust_offset(&self, offset: u64, bytes: u64, is_write: bool) -> io::Result<u64> {
        let size = self.size.load(Ordering::Relaxed);
        if self.has_size() && (offset > size || bytes > size - offset) {
            // Do not touch anything outside the window the options gave.
            return Err(errno(if is_write { libc::ENOSPC } else { libc::EINVAL }));
        }
        offset.checked_add(self.off()).filter(|&o| o <= i64::MAX as u64).ok_or(errno(libc::EINVAL))
    }
}

impl Driver for RawDriver {
    /// `raw_co_get_info()`: whatever the file says.
    fn get_info(&self, bs: &Node) -> Option<io::Result<crate::node::BlockDriverInfo>> {
        Some(bs.file().get_info())
    }

    /// `raw_has_zero_init()`: whatever the file says.
    fn has_zero_init(&self, bs: &Node) -> Option<bool> {
        Some(bs.file().has_zero_init())
    }

    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let off = self.adjust_offset(offset, buf.len() as u64, false)?;
        bs.file().pread(off, buf)
    }

    /// `raw_co_pwritev()`. A probed image refuses a write that would make block 0 look like
    /// another format, so a guest cannot turn its disk into, say, a qcow2 image with a backing
    /// file pointing at a host file. QEMU makes such images use 512 byte requests. Here a
    /// partial write to block 0 is merged with what is on disk and the result is checked.
    fn pwrite(&self, bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        self.check_block0(bs, offset, buf)?;
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
        let size = if len < self.off() {
            0
        } else if self.has_size() {
            self.size.load(Ordering::Relaxed).min(len - self.off())
        } else {
            len - self.off()
        };
        self.size.store(size, Ordering::Relaxed);
        Ok(size)
    }

    /// `raw_co_truncate()`.
    fn truncate(&self, bs: &Node, len: u64) -> Result<()> {
        if self.has_size() {
            return Err(Error::generic("Cannot resize fixed-size raw disks"));
        }
        if i64::MAX as u64 - len < self.off() {
            return Err(Error::generic("Disk size too large for the chosen offset"));
        }
        self.size.store(len, Ordering::Relaxed);
        bs.file().truncate(len + self.off())
    }

    /// `raw_co_truncate()` with the preallocation mode, which goes on to the child.
    fn truncate_full(
        &self,
        bs: &Node,
        len: u64,
        exact: bool,
        prealloc: ruvm_qapi::types::PreallocMode,
        flags: u32,
    ) -> Result<()> {
        if self.has_size() {
            return Err(Error::generic("Cannot resize fixed-size raw disks"));
        }
        if i64::MAX as u64 - len < self.off() {
            return Err(Error::generic("Disk size too large for the chosen offset"));
        }
        self.size.store(len, Ordering::Relaxed);
        bs.file().truncate_full((len + self.off()) as i64, exact, prealloc, flags)
    }

    /// `raw_co_block_status()`: everything is where the child has it.
    fn block_status(
        &self,
        bs: &Node,
        _want: u32,
        offset: u64,
        bytes: u64,
    ) -> Option<io::Result<BlockStatus>> {
        Some(Ok(BlockStatus {
            ret: BDRV_BLOCK_RAW | BDRV_BLOCK_OFFSET_VALID,
            pnum: bytes,
            map: offset + self.off(),
            file: Some(bs.file()),
        }))
    }

    /// `raw_refresh_limits()`: a probed image is written in whole sectors, so a write to
    /// block 0 always shows the whole of it.
    fn refresh_limits(&self, _bs: &Node, bl: &mut BlockLimits) -> Result<()> {
        if self.probed {
            bl.request_alignment = bl.request_alignment.max(BDRV_SECTOR_SIZE as u32);
        }
        Ok(())
    }

    fn pwrite_flags(&self, bs: &Node, offset: u64, buf: &[u8], flags: u32) -> io::Result<()> {
        self.check_block0(bs, offset, buf)?;
        let off = self.adjust_offset(offset, buf.len() as u64, true)?;
        bs.file().pwrite_flags(off, buf, flags)
    }

    fn supported_write_flags(&self) -> u32 {
        BDRV_REQ_WRITE_UNCHANGED | BDRV_REQ_FUA
    }

    fn pwrite_zeroes_flags(
        &self,
        bs: &Node,
        offset: u64,
        bytes: u64,
        flags: u32,
    ) -> io::Result<()> {
        let off = self.adjust_offset(offset, bytes, true)?;
        bs.file().pwrite_zeroes_flags(off, bytes, flags)
    }

    fn supported_zero_flags(&self) -> u32 {
        BDRV_REQ_WRITE_UNCHANGED | BDRV_REQ_FUA | BDRV_REQ_MAY_UNMAP | BDRV_REQ_NO_FALLBACK
    }

    /// `raw_reopen_prepare()`: the window options may not change here, which the generic
    /// reopen code already refuses for options a driver does not take on reopen.
    /// `raw_reopen_prepare()`: reads `offset` and `size` again and checks them against
    /// the file; `raw_reopen_commit()` puts them in place.
    fn reopen_prepare(&self, bs: &Node, state: &mut ReopenState) -> Option<Result<()>> {
        Some((|| {
            let offset = take_size_opt(&mut state.options, "offset")?;
            let size = take_size_opt(&mut state.options, "size")?;
            let len =
                bs.file().getlength().map_err(|e| Error::from_io("Could not get image size", e))?;
            let new = raw_open(offset, size, len, self.probed)?;
            state.opaque = Some(Box::new(new));
            Ok(())
        })())
    }

    fn reopen_commit(&self, _bs: &Node, state: &mut ReopenState) {
        if let Some(new) = state.opaque.take().and_then(|o| o.downcast::<RawDriver>().ok()) {
            self.offset.store(new.off(), Ordering::Relaxed);
            self.has_size.store(new.has_size(), Ordering::Relaxed);
            self.size.store(new.size.load(Ordering::Relaxed), Ordering::Relaxed);
        }
    }

    fn has_truncate(&self) -> bool {
        true
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
}
