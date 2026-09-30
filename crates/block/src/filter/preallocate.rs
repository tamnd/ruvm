// SPDX-License-Identifier: GPL-2.0-or-later

//! `preallocate` from block/preallocate.c: a filter that, when a write goes past the end of
//! its child, grows the child by more than the write needs with a write of zeroes, so that
//! the following writes find the space already allocated. The preallocation is cut off again
//! on close.
//!
//! The filter keeps three offsets: `data_end`, the end of the data written so far, which is
//! the length it reports; `zero_start`, where the preallocated zeroes past the data begin;
//! and `file_end`, the real length of the child. They are only valid while the filter holds
//! exclusive write and resize permissions on the child. A negative value means unknown.
//!
//! Differences from QEMU:
//!
//! - QEMU drops the preallocation in a bottom half as soon as the parents give up write
//!   access. There are no bottom halves here: the preallocation is dropped on close and on a
//!   reopen to read-only, and until then the filter keeps its exclusive write and resize
//!   permissions on the child.
//! - "Failed to drop preallocation" is followed by the message of the failed truncation
//!   rather than by `strerror()` of its errno, which this code does not have.
//! - `prealloc-align` and `prealloc-size` are integers in the QAPI schema, so a negative
//!   value gets `parse_option_size()`'s error.

use std::io;
use std::sync::{Mutex, Weak};

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{BlockdevOptionsU, PreallocMode};
use ruvm_qapi::visit::parse_option_size;
use ruvm_qapi::{QDict, QValue};

use crate::drivers::{DriverDef, OpenArgs};
use crate::node::{
    BDRV_CHILD_FILTERED, BDRV_CHILD_PRIMARY, BDRV_REQ_FUA, BDRV_REQ_MAY_UNMAP,
    BDRV_REQ_NO_FALLBACK, BDRV_REQ_NO_WAIT, BDRV_REQ_SERIALISING, BDRV_REQ_WRITE_UNCHANGED,
    BDRV_REQ_ZERO_WRITE, BDRV_SECTOR_SIZE, Driver, Node, ReopenState,
};
use crate::perm::{BLK_PERM_RESIZE, BLK_PERM_WRITE, PermCtx, default_perms};

/// `bdrv_preallocate_filter`.
pub(crate) static PREALLOCATE: DriverDef = DriverDef::filter("preallocate", preallocate_open);

const MIB: u64 = 1 << 20;
const EINVAL: i64 = -(libc::EINVAL as i64);

/// `PreallocateOpts`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Opts {
    prealloc_size: u64,
    prealloc_align: u64,
}

/// The offsets of `BDRVPreallocateState`.
#[derive(Debug)]
struct State {
    opts: Opts,
    data_end: i64,
    zero_start: i64,
    file_end: i64,
}

struct PreallocateDriver {
    st: Mutex<State>,
    /// Serialises the requests that change the offsets, as QEMU's coroutines are.
    busy: Mutex<()>,
    /// The child, for `set_perm`, which is not told the node.
    file: Mutex<Weak<Node>>,
    /// `bs->supported_write_flags`.
    write_flags: u32,
    /// `bs->supported_zero_flags`.
    zero_flags: u32,
}

/// `can_write_resize()`.
fn can_write_resize(perm: u64) -> bool {
    perm & BLK_PERM_WRITE != 0 && perm & BLK_PERM_RESIZE != 0
}

fn neg_errno(e: &io::Error) -> i64 {
    -i64::from(e.raw_os_error().unwrap_or(libc::EIO))
}

/// A size option as `qemu_opt_get_size()` reads it.
fn size_value(name: &str, v: i64) -> Result<u64> {
    if v < 0 {
        return parse_option_size(name, &v.to_string());
    }
    Ok(v as u64)
}

/// `preallocate_absorb_opts()` with the values already parsed.
fn check_opts(
    prealloc_align: Option<u64>,
    prealloc_size: Option<u64>,
    child: &Node,
) -> Result<Opts> {
    let o = Opts {
        prealloc_align: prealloc_align.unwrap_or(MIB),
        prealloc_size: prealloc_size.unwrap_or(128 * MIB),
    };
    if o.prealloc_align % BDRV_SECTOR_SIZE != 0 {
        return Err(Error::generic(format!(
            "prealloc-align parameter of preallocate filter is not aligned to {BDRV_SECTOR_SIZE}"
        )));
    }
    let align = child.request_alignment();
    if o.prealloc_align % align != 0 {
        return Err(Error::generic(format!(
            "prealloc-align parameter of preallocate filter is not aligned to underlying node \
             request alignment ({align})"
        )));
    }
    Ok(o)
}

/// `preallocate_absorb_opts()` on reopen options: takes the options out of `options`.
fn absorb_opts(options: &mut QDict, child: &Node) -> Result<Opts> {
    let mut take = |name: &str| -> Result<Option<u64>> {
        match options.remove(name) {
            None => Ok(None),
            Some(QValue::Str(s)) => parse_option_size(name, &s).map(Some),
            Some(v) => match v.as_i64() {
                Some(i) => size_value(name, i).map(Some),
                None => Err(Error::generic(format!(
                    "Invalid parameter type for '{name}', expected: size"
                ))),
            },
        }
    };
    let align = take("prealloc-align")?;
    let size = take("prealloc-size")?;
    check_opts(align, size, child)
}

/// `preallocate_open()`.
fn preallocate_open(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::Preallocate(o) = opts else {
        unreachable!("the preallocate driver gets preallocate options");
    };
    let file = args.open_child(*o.file, "file", BDRV_CHILD_FILTERED | BDRV_CHILD_PRIMARY)?;
    let align = o.prealloc_align.map(|v| size_value("prealloc-align", v)).transpose()?;
    let size = o.prealloc_size.map(|v| size_value("prealloc-size", v)).transpose()?;
    let opts = check_opts(align, size, &file)?;
    Ok(Box::new(PreallocateDriver::new(opts, &file)))
}

impl PreallocateDriver {
    fn new(opts: Opts, file: &Node) -> Self {
        // The offsets are set up when the permissions are.
        PreallocateDriver {
            st: Mutex::new(State { opts, data_end: EINVAL, zero_start: EINVAL, file_end: EINVAL }),
            busy: Mutex::new(()),
            file: Mutex::new(file.weak()),
            write_flags: BDRV_REQ_WRITE_UNCHANGED
                | (BDRV_REQ_FUA & file.driver.supported_write_flags()),
            zero_flags: BDRV_REQ_WRITE_UNCHANGED
                | ((BDRV_REQ_FUA | BDRV_REQ_MAY_UNMAP | BDRV_REQ_NO_FALLBACK)
                    & file.driver.supported_zero_flags()),
        }
    }

    /// `has_prealloc_perms()`.
    fn has_prealloc_perms(bs: &Node) -> bool {
        bs.filter_child().is_some_and(|c| can_write_resize(c.perm().0))
    }

    /// `handle_write()`: grows the child ahead of a write past its end. Returns true if
    /// `want_merge_zero` and the range is zeroes now, from this call or from earlier
    /// preallocation.
    fn handle_write(&self, bs: &Node, offset: u64, bytes: u64, want_merge_zero: bool) -> bool {
        let file = bs.file();
        let offset = offset as i64;
        let end = offset + bytes as i64;
        let file_align = file.request_alignment() as i64;
        let _busy = self.busy.lock().unwrap();

        if !Self::has_prealloc_perms(bs) {
            // We have no state and should not try to recover it.
            return false;
        }

        let mut st = self.st.lock().unwrap();
        let prealloc_align = (st.opts.prealloc_align as i64).max(file_align);
        debug_assert!(prealloc_align % file_align == 0);
        if st.data_end < 0 {
            drop(st);
            let len = file.getlength();
            st = self.st.lock().unwrap();
            match len {
                Ok(l) => st.data_end = l as i64,
                Err(e) => {
                    st.data_end = neg_errno(&e);
                    return false;
                }
            }
            if st.file_end < 0 {
                st.file_end = st.data_end;
            }
        }

        if end <= st.data_end {
            return false;
        }

        // The request writes past a valid data_end.
        st.data_end = end;
        if st.zero_start < 0 || !want_merge_zero {
            st.zero_start = end;
        }

        if st.file_end < 0 {
            drop(st);
            let len = file.getlength();
            st = self.st.lock().unwrap();
            match len {
                Ok(l) => st.file_end = l as i64,
                Err(e) => {
                    st.file_end = neg_errno(&e);
                    return false;
                }
            }
        }

        // data_end, zero_start and file_end are all valid now.
        if end <= st.file_end {
            // No preallocation needed.
            return want_merge_zero && offset >= st.zero_start;
        }

        // The request writes past file_end, so preallocate.
        let start = if want_merge_zero { offset.min(st.file_end) } else { st.file_end };
        let prealloc_start = align_up(start, file_align);
        let prealloc_end =
            align_up(prealloc_start.max(end) + st.opts.prealloc_size as i64, prealloc_align);
        let want_merge_zero = want_merge_zero && prealloc_start <= offset;
        drop(st);

        let r = file.pwrite_zeroes_flags(
            prealloc_start as u64,
            (prealloc_end - prealloc_start) as u64,
            BDRV_REQ_NO_FALLBACK | BDRV_REQ_SERIALISING | BDRV_REQ_NO_WAIT,
        );
        let mut st = self.st.lock().unwrap();
        if let Err(e) = r {
            st.file_end = neg_errno(&e);
            return false;
        }
        st.file_end = prealloc_end;
        want_merge_zero
    }

    /// `preallocate_truncate_to_real_size()`.
    fn truncate_to_real_size(&self, bs: &Node) -> Result<()> {
        let file = bs.file();
        let mut st = self.st.lock().unwrap();
        if st.file_end < 0 {
            match file.getlength() {
                Ok(l) => st.file_end = l as i64,
                Err(e) => {
                    st.file_end = neg_errno(&e);
                    return Err(Error::from_io("Failed to get file length", e));
                }
            }
        }
        if st.data_end < st.file_end {
            let data_end = st.data_end;
            drop(st);
            let r = file.truncate_full(data_end, true, PreallocMode::Off, 0);
            let mut st = self.st.lock().unwrap();
            if let Err(e) = r {
                st.file_end = -i64::from(libc::EIO);
                return Err(e.prepend("Failed to drop preallocation: "));
            }
            st.file_end = st.data_end;
        }
        Ok(())
    }

    /// `preallocate_drop_resize()`: cuts the preallocation off and forgets the offsets, so
    /// that the permissions on the child can be given up.
    fn drop_resize(&self, bs: &Node) -> Result<()> {
        if self.st.lock().unwrap().data_end < 0 {
            return Ok(());
        }
        // Before the child becomes read-only, give it its real size.
        self.truncate_to_real_size(bs)?;
        // Anyone may change the child once the permissions are gone, so the offsets are
        // unknown until a parent wants write access again.
        {
            let mut st = self.st.lock().unwrap();
            st.data_end = EINVAL;
            st.file_end = EINVAL;
            st.zero_start = EINVAL;
        }
        if let Some(c) = bs.filter_child() {
            // As in QEMU, a failure here keeps the permissions.
            let _ = bs.refresh_child_perms(&c);
        }
        Ok(())
    }
}

fn align_up(v: i64, a: i64) -> i64 {
    (v + a - 1) / a * a
}

impl Driver for PreallocateDriver {
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        bs.file().pread(offset, buf)
    }

    fn pwrite(&self, bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        self.pwrite_flags(bs, offset, buf, 0)
    }

    fn pwrite_flags(&self, bs: &Node, offset: u64, buf: &[u8], flags: u32) -> io::Result<()> {
        self.handle_write(bs, offset, buf.len() as u64, false);
        bs.file().pwrite_flags(offset, buf, flags)
    }

    fn supported_write_flags(&self) -> u32 {
        self.write_flags
    }

    fn pwrite_zeroes(&self, bs: &Node, offset: u64, bytes: u64, may_unmap: bool) -> io::Result<()> {
        self.pwrite_zeroes_flags(bs, offset, bytes, if may_unmap { BDRV_REQ_MAY_UNMAP } else { 0 })
    }

    fn pwrite_zeroes_flags(
        &self,
        bs: &Node,
        offset: u64,
        bytes: u64,
        flags: u32,
    ) -> io::Result<()> {
        let want_merge_zero = flags & !(BDRV_REQ_ZERO_WRITE | BDRV_REQ_NO_FALLBACK) == 0;
        if self.handle_write(bs, offset, bytes, want_merge_zero) {
            return Ok(());
        }
        bs.file().pwrite_zeroes_flags(offset, bytes, flags)
    }

    fn supported_zero_flags(&self) -> u32 {
        self.zero_flags
    }

    fn pdiscard(&self, bs: &Node, offset: u64, bytes: u64) -> io::Result<()> {
        bs.file().pdiscard(offset, bytes)
    }

    fn flush_to_disk(&self, bs: &Node) -> io::Result<()> {
        bs.file().flush()
    }

    /// `preallocate_co_getlength()`.
    fn getlength(&self, bs: &Node) -> io::Result<u64> {
        let data_end = self.st.lock().unwrap().data_end;
        if data_end >= 0 {
            return Ok(data_end as u64);
        }
        let r = bs.file().getlength();
        if Self::has_prealloc_perms(bs) {
            let v = match &r {
                Ok(l) => *l as i64,
                Err(e) => neg_errno(e),
            };
            let mut st = self.st.lock().unwrap();
            st.file_end = v;
            st.zero_start = v;
            st.data_end = v;
        }
        r
    }

    /// `preallocate_co_truncate()`.
    fn truncate_full(
        &self,
        bs: &Node,
        offset: u64,
        exact: bool,
        prealloc: PreallocMode,
        flags: u32,
    ) -> Result<()> {
        let file = bs.file();
        let offset = offset as i64;
        let _busy = self.busy.lock().unwrap();
        let mut st = self.st.lock().unwrap();
        if st.data_end >= 0 && offset > st.data_end {
            if st.file_end < 0 {
                match file.getlength() {
                    Ok(l) => st.file_end = l as i64,
                    Err(e) => {
                        st.file_end = neg_errno(&e);
                        return Err(Error::generic("failed to get file length"));
                    }
                }
            }
            if prealloc == PreallocMode::Falloc {
                // If the preallocation already covers the new size, it just moves from
                // the filter's preallocation to the one the user asked for.
                if offset <= st.file_end {
                    st.data_end = offset;
                    return Ok(());
                }
            } else if st.file_end > st.data_end {
                // Drop the preallocation, so that a shrinking truncation does not fail,
                // PREALLOC_MODE_OFF keeps the disk usage small and PREALLOC_MODE_FULL
                // really writes the whole region.
                let data_end = st.data_end;
                drop(st);
                let r = file.truncate_full(data_end, true, PreallocMode::Off, 0);
                st = self.st.lock().unwrap();
                if let Err(e) = r {
                    st.file_end = -i64::from(libc::EIO);
                    return Err(
                        e.prepend("preallocate-filter: failed to drop write-zero preallocation: ")
                    );
                }
                st.file_end = st.data_end;
            }
            st.data_end = offset;
        }
        drop(st);

        let r = file.truncate_full(offset, exact, prealloc, flags);
        let mut st = self.st.lock().unwrap();
        if let Err(e) = r {
            st.file_end = -i64::from(libc::EIO);
            st.zero_start = st.file_end;
            st.data_end = st.file_end;
            return Err(e);
        }
        if Self::has_prealloc_perms(bs) {
            st.file_end = offset;
            st.zero_start = offset;
            st.data_end = offset;
        }
        Ok(())
    }

    /// `preallocate_close()`.
    fn close(&self, bs: &Node) {
        if self.st.lock().unwrap().data_end >= 0 {
            let _ = self.truncate_to_real_size(bs);
        }
    }

    /// `preallocate_reopen_prepare()`.
    fn reopen_prepare(&self, bs: &Node, state: &mut ReopenState) -> Option<Result<()>> {
        let r = (|| {
            let opts = absorb_opts(&mut state.options, &bs.file())?;
            // Drop the preallocation already here if reopening read-only. The child may be
            // reopened read-only too, and afterwards would be too late.
            if state.flags.read_only {
                self.drop_resize(bs)?;
            }
            state.opaque = Some(Box::new(opts));
            Ok(())
        })();
        Some(r)
    }

    fn reopen_commit(&self, _bs: &Node, state: &mut ReopenState) {
        if let Some(o) = state.opaque.take().and_then(|o| o.downcast::<Opts>().ok()) {
            self.st.lock().unwrap().opts = *o;
        }
    }

    fn reopen_abort(&self, _bs: &Node, state: &mut ReopenState) {
        state.opaque = None;
    }

    /// `preallocate_set_perm()`.
    fn set_perm(&self, perm: u64, _shared: u64) -> Result<()> {
        if can_write_resize(perm) {
            let mut st = self.st.lock().unwrap();
            if st.data_end < 0 {
                if let Some(file) = self.file.lock().unwrap().upgrade() {
                    let len = file.total_sectors() * BDRV_SECTOR_SIZE as i64;
                    st.data_end = len;
                    st.file_end = len;
                    st.zero_start = len;
                }
            }
        }
        // QEMU schedules the drop of the preallocation here when the write permissions
        // go. That happens on close or on a reopen to read-only instead, see the module
        // documentation.
        Ok(())
    }

    /// `preallocate_child_perm()`.
    fn child_perm_for(&self, ctx: &PermCtx<'_>, perm: u64, shared: u64) -> (u64, u64) {
        *self.file.lock().unwrap() = std::sync::Arc::downgrade(ctx.child);
        let (mut p, mut s) = default_perms(ctx, perm, shared);
        // Exclusive write and resize are needed not only while a parent may write, but
        // also after the parents gave write access up until the preallocation is dropped.
        if can_write_resize(perm) || self.st.lock().unwrap().data_end != EINVAL {
            p |= BLK_PERM_WRITE | BLK_PERM_RESIZE;
            // Not shared, to keep the offsets valid.
            s &= !(BLK_PERM_WRITE | BLK_PERM_RESIZE);
        }
        (p, s)
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::backend::BlockBackend;
    use crate::filter::copy_on_read::tests::{Mem, mem, mem_node};
    use crate::node::{NodeFlags, NodeMeta, NodeSpec};
    use crate::perm::BLK_PERM_ALL;

    fn prealloc_node(file: &Arc<Node>, opts: Opts) -> Arc<Node> {
        Node::build(NodeSpec {
            name: "p".into(),
            driver_name: "preallocate",
            driver: Box::new(PreallocateDriver::new(opts, file)),
            def: Some(&PREALLOCATE),
            flags: NodeFlags::default(),
            meta: NodeMeta::default(),
            children: vec![("file".into(), file.clone(), BDRV_CHILD_FILTERED | BDRV_CHILD_PRIMARY)],
        })
        .unwrap()
    }

    fn file_len(n: &Node) -> usize {
        mem(n).data.lock().unwrap().len()
    }

    #[test]
    fn grows_ahead_and_trims_on_close() {
        let file = mem_node("file", Mem::with_data(vec![0; 4096], true), None);
        let opts = Opts { prealloc_align: 64 * 1024, prealloc_size: 100 * 1024 };
        let bs = prealloc_node(&file, opts);
        let _blk = BlockBackend::with_node(
            None,
            bs.clone(),
            BLK_PERM_WRITE | BLK_PERM_RESIZE,
            BLK_PERM_ALL,
        )
        .unwrap();

        // Inside the file: nothing to preallocate.
        bs.pwrite(0, &[1u8; 512]).unwrap();
        assert_eq!(file_len(&file), 4096);

        // Past the end: the file grows to align_up(4608 + 100 KiB, 64 KiB) = 128 KiB with
        // a zero write from the old end.
        bs.pwrite(4096, &[2u8; 512]).unwrap();
        assert_eq!(file_len(&file), 128 * 1024);
        let z = mem(&file).zeroes.lock().unwrap().clone();
        assert_eq!(z.len(), 1);
        assert_eq!((z[0].0, z[0].1), (4096, 128 * 1024 - 4096));
        assert_eq!(z[0].2 & BDRV_REQ_NO_FALLBACK, BDRV_REQ_NO_FALLBACK);
        // The filter reports the end of the data.
        assert_eq!(bs.getlength().unwrap(), 4608);

        // A write inside the preallocated area needs no more zeroes.
        bs.pwrite(8192, &[3u8; 512]).unwrap();
        assert_eq!(mem(&file).zeroes.lock().unwrap().len(), 1);
        assert_eq!(bs.getlength().unwrap(), 8704);

        // A zero write past the data but inside the preallocation is already done.
        bs.pwrite_zeroes(16384, 512, false).unwrap();
        assert_eq!(mem(&file).zeroes.lock().unwrap().len(), 1);
        assert_eq!(bs.getlength().unwrap(), 16896);

        drop(_blk);
        bs.driver.close(&bs);
        assert_eq!(file_len(&file), 16896);
    }

    #[test]
    fn truncate_drops_preallocation() {
        let file = mem_node("file", Mem::with_data(vec![0; 4096], true), None);
        let opts = Opts { prealloc_align: MIB, prealloc_size: MIB };
        let bs = prealloc_node(&file, opts);
        let _blk = BlockBackend::with_node(
            None,
            bs.clone(),
            BLK_PERM_WRITE | BLK_PERM_RESIZE,
            BLK_PERM_ALL,
        )
        .unwrap();
        bs.pwrite(4096, &[1u8; 512]).unwrap();
        assert_eq!(file_len(&file), 2 * MIB as usize);
        bs.truncate(8192).unwrap();
        assert_eq!(file_len(&file), 8192);
        assert_eq!(bs.getlength().unwrap(), 8192);

        // A falloc truncation inside the preallocation only moves data_end.
        bs.pwrite(8192, &[1u8; 512]).unwrap();
        let grown = file_len(&file);
        bs.truncate_full(65536, false, PreallocMode::Falloc, 0).unwrap();
        assert_eq!(file_len(&file), grown);
        assert_eq!(bs.getlength().unwrap(), 65536);
    }

    #[test]
    fn option_errors() {
        let file = mem_node("file", Mem::with_data(vec![0; 4096], true), None);
        let e = check_opts(Some(1000), None, &file).unwrap_err();
        assert_eq!(
            e.message(),
            "prealloc-align parameter of preallocate filter is not aligned to 512"
        );
        let o = check_opts(None, None, &file).unwrap();
        assert_eq!(o, Opts { prealloc_align: MIB, prealloc_size: 128 * MIB });

        let mut d = QDict::new();
        d.put("prealloc-align", "2M");
        d.put("prealloc-size", "4096");
        d.put("other", "x");
        let o = absorb_opts(&mut d, &file).unwrap();
        assert_eq!(o, Opts { prealloc_align: 2 * MIB, prealloc_size: 4096 });
        assert_eq!(d.len(), 1);

        let g = crate::graph::BlockGraph::new();
        let mut o = QDict::new();
        o.put("driver", "preallocate");
        o.put("file.driver", "blkdebug");
        o.put("file.align", "4096");
        o.put("file.image.driver", "null-co");
        o.put("prealloc-align", "512");
        let e = g.open_image(None, o).unwrap_err();
        assert_eq!(
            e.message(),
            "prealloc-align parameter of preallocate filter is not aligned to underlying node \
             request alignment (4096)"
        );
    }
}
