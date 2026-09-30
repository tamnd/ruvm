// SPDX-License-Identifier: GPL-2.0-or-later

//! `copy-before-write` from block/copy-before-write.c: a filter that, before a write
//! changes a range of its `file` child, copies the old data of the range to its `target`
//! child. The target then holds a snapshot of the child as it was when the filter was
//! put in place. The backup job uses it, and so do image fleecing setups that read the
//! snapshot through a `snapshot-access` node.
//!
//! Differences from QEMU:
//!
//! - The copying goes through [`crate::block_copy`], see its differences.
//! - `BDRV_O_CBW_DISCARD_SOURCE` is not an open flag: [`cbw_append`] turns
//!   `discard-source` on after the node is open, so a node made by `blockdev-add` never
//!   discards its source, as in QEMU where the flag cannot be given by the user either.
//! - The done and access bitmaps belong to no node, so `query-block` does not list them.

use std::io;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use ruvm_base::{Error, Result};
use ruvm_qapi::QDict;
use ruvm_qapi::types::{BlockdevOptionsU, OnCbwError};

use crate::bitmap::DirtyBitmap;
use crate::block_copy::{BlockCopyState, block_copy};
use crate::drivers::{DriverDef, OpenArgs};
use crate::graph::BlockGraph;
use crate::job::chain::{drop_filter, insert_node};
use crate::node::{
    BDRV_CHILD_DATA, BDRV_CHILD_FILTERED, BDRV_CHILD_PRIMARY, BDRV_REQ_FUA, BDRV_REQ_MAY_UNMAP,
    BDRV_REQ_NO_FALLBACK, BDRV_REQ_WRITE_UNCHANGED, BlockStatus, Driver, Node,
};
use crate::perm::{
    BLK_PERM_ALL, BLK_PERM_CONSISTENT_READ, BLK_PERM_RESIZE, BLK_PERM_WRITE, PermCtx, default_perms,
};

/// `bdrv_cbw_filter`.
pub(crate) static COPY_BEFORE_WRITE: DriverDef = DriverDef::filter("copy-before-write", cbw_open);

/// The part of `BDRVCopyBeforeWriteState` under `lock`.
struct Locked {
    /// `frozen_read_reqs`: ranges of the source a snapshot reader is reading, which guest
    /// writes must not change until the read ends.
    frozen_read_reqs: Vec<(u64, u64, u64)>,
    next_id: u64,
}

/// `BDRVCopyBeforeWriteState`.
pub(crate) struct CbwDriver {
    bcs: Arc<BlockCopyState>,
    target: Arc<Node>,
    on_cbw_error: OnCbwError,
    cbw_timeout_ns: u64,
    discard_source: AtomicBool,
    /// `access_bitmap`: the areas a snapshot reader may read.
    access_bitmap: Arc<DirtyBitmap>,
    /// `done_bitmap`: the areas copy-before-write copied to the target.
    done_bitmap: Arc<DirtyBitmap>,
    lock: Mutex<Locked>,
    /// Signalled when a frozen read ends.
    reqs_cond: Condvar,
    /// `snapshot_error`: 0, or the negative errno of the first failed copy with
    /// `on-cbw-error=break-snapshot`.
    snapshot_error: AtomicI32,
    write_flags: u32,
    zero_flags: u32,
}

/// `cbw_open()`.
fn cbw_open(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::CopyBeforeWrite(o) = opts else {
        unreachable!("the copy-before-write driver gets copy-before-write options");
    };
    let file = args.open_child(*o.file, "file", BDRV_CHILD_FILTERED | BDRV_CHILD_PRIMARY)?;
    let target = args.open_child(*o.target, "target", BDRV_CHILD_DATA)?;

    let mut bitmap = None;
    if let Some(b) = &o.bitmap {
        bitmap = Some(args.graph.dirty_bitmap_lookup(&b.node, &b.name)?.1);
    }
    let on_cbw_error = o.on_cbw_error.unwrap_or(OnCbwError::BreakGuestWrite);
    let cbw_timeout_ns = o.cbw_timeout.map_or(0, |t| u64::from(t) * 1_000_000_000);

    let write_flags =
        BDRV_REQ_WRITE_UNCHANGED | (BDRV_REQ_FUA & file.driver.supported_write_flags());
    let zero_flags = BDRV_REQ_WRITE_UNCHANGED
        | ((BDRV_REQ_FUA | BDRV_REQ_MAY_UNMAP | BDRV_REQ_NO_FALLBACK)
            & file.driver.supported_zero_flags());

    let bcs = BlockCopyState::new(
        &file,
        &target,
        bitmap.as_ref(),
        false,
        o.min_cluster_size.unwrap_or(0),
    )
    .map_err(|e| e.prepend("Cannot create block-copy-state: "))?;

    let cluster_size = bcs.cluster_size() as u32;
    let size = bcs.dirty_bitmap().size();
    let done_bitmap = DirtyBitmap::detached(size, cluster_size);
    // s->access_bitmap starts equal to bcs bitmap
    let access_bitmap = DirtyBitmap::detached(size, cluster_size);
    access_bitmap.merge_internal(bcs.dirty_bitmap(), true);

    Ok(Box::new(CbwDriver {
        bcs,
        target,
        on_cbw_error,
        cbw_timeout_ns,
        discard_source: AtomicBool::new(false),
        access_bitmap,
        done_bitmap,
        lock: Mutex::new(Locked { frozen_read_reqs: Vec::new(), next_id: 1 }),
        reqs_cond: Condvar::new(),
        snapshot_error: AtomicI32::new(0),
        write_flags,
        zero_flags,
    }))
}

fn overlaps(r: &(u64, u64, u64), offset: u64, bytes: u64) -> bool {
    offset < r.1 + r.2 && r.1 < offset + bytes
}

/// Where [`CbwDriver::snapshot_read_lock`] sends a snapshot read.
enum ReadFrom {
    /// The target, which needs no lock.
    Target,
    /// The source, with the id of the frozen read.
    Source(u64),
}

impl CbwDriver {
    /// `cbw_do_copy_before_write()`: on success, also waits for the snapshot reads of the
    /// range, and no new ones start there.
    fn copy_before_write(&self, bs: &Node, offset: u64, bytes: u64, flags: u32) -> io::Result<()> {
        if flags & BDRV_REQ_WRITE_UNCHANGED != 0 {
            return Ok(());
        }
        if self.snapshot_error.load(Ordering::SeqCst) != 0 {
            return Ok(());
        }
        let cluster_size = self.bcs.cluster_size();
        let off = offset / cluster_size * cluster_size;
        let end = (offset + bytes).div_ceil(cluster_size) * cluster_size;

        // Increase in_flight, so that in case of timed-out block-copy, the remaining
        // background block_copy() request (which can't be immediately cancelled by timeout)
        // is presented in bs->in_flight. This way we are sure that on bs close() we'll
        // previously wait for all timed-out but yet running block_copy calls.
        bs.inc_in_flight();
        let me = bs.weak();
        let cb: Box<dyn FnOnce() + Send> = Box::new(move || {
            if let Some(n) = me.upgrade() {
                n.dec_in_flight();
            }
        });
        let ret = block_copy(&self.bcs, off, end - off, true, self.cbw_timeout_ns, Some(cb));
        if let Err(e) = ret {
            if self.on_cbw_error == OnCbwError::BreakGuestWrite {
                return Err(io::Error::from_raw_os_error(e));
            }
        }

        let mut l = self.lock.lock().unwrap();
        match ret {
            Err(e) => {
                let _ =
                    self.snapshot_error.compare_exchange(0, -e, Ordering::SeqCst, Ordering::SeqCst);
            }
            Ok(()) => self.done_bitmap.set_range(off, end - off),
        }
        while l.frozen_read_reqs.iter().any(|r| overlaps(r, off, end - off)) {
            l = self.reqs_cond.wait(l).unwrap();
        }
        Ok(())
    }

    /// `cbw_snapshot_read_lock()`: `None` if the range is not readable, else where to read
    /// how many bytes from.
    fn snapshot_read_lock(&self, offset: u64, bytes: u64) -> Option<(ReadFrom, u64)> {
        let mut l = self.lock.lock().unwrap();
        if self.snapshot_error.load(Ordering::SeqCst) != 0 {
            return None;
        }
        if self.access_bitmap.next_zero(offset, bytes).is_some() {
            return None;
        }
        let (done, pnum) = self.done_bitmap.status(offset, bytes);
        if done {
            // We don't need to lock something to read from s->target.
            Some((ReadFrom::Target, pnum))
        } else {
            let id = l.next_id;
            l.next_id += 1;
            l.frozen_read_reqs.push((id, offset, pnum));
            Some((ReadFrom::Source(id), pnum))
        }
    }

    /// `cbw_snapshot_read_unlock()`.
    fn snapshot_read_unlock(&self, from: &ReadFrom) {
        if let ReadFrom::Source(id) = from {
            let mut l = self.lock.lock().unwrap();
            l.frozen_read_reqs.retain(|r| r.0 != *id);
            self.reqs_cond.notify_all();
        }
    }

    fn node_for<'a>(&'a self, bs: &Node, from: &ReadFrom) -> std::borrow::Cow<'a, Arc<Node>> {
        match from {
            ReadFrom::Target => std::borrow::Cow::Borrowed(&self.target),
            ReadFrom::Source(_) => std::borrow::Cow::Owned(bs.file()),
        }
    }

    /// `cbw_co_preadv_snapshot()`.
    fn preadv_snapshot(&self, bs: &Node, mut offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let mut done = 0usize;
        while done < buf.len() {
            let bytes = (buf.len() - done) as u64;
            let Some((from, cur)) = self.snapshot_read_lock(offset, bytes) else {
                return Err(io::Error::from_raw_os_error(libc::EACCES));
            };
            let cur = cur.min(bytes);
            let node = self.node_for(bs, &from);
            let r = node.preadv_flags(offset, &mut buf[done..done + cur as usize], 0);
            self.snapshot_read_unlock(&from);
            r?;
            offset += cur;
            done += cur as usize;
        }
        Ok(())
    }

    /// `cbw_co_snapshot_block_status()`.
    fn snapshot_block_status(&self, bs: &Node, offset: u64, bytes: u64) -> io::Result<BlockStatus> {
        let Some((from, cur)) = self.snapshot_read_lock(offset, bytes) else {
            return Err(io::Error::from_raw_os_error(libc::EACCES));
        };
        let node = self.node_for(bs, &from);
        let r = node.block_status(offset, cur.min(bytes));
        self.snapshot_read_unlock(&from);
        r
    }

    /// `cbw_co_pdiscard_snapshot()`.
    fn pdiscard_snapshot(&self, offset: u64, bytes: u64) -> io::Result<()> {
        let cluster_size = self.bcs.cluster_size();
        let aligned_offset = offset.div_ceil(cluster_size) * cluster_size;
        let aligned_end = (offset + bytes) / cluster_size * cluster_size;
        if aligned_end <= aligned_offset {
            return Ok(());
        }
        let aligned_bytes = aligned_end - aligned_offset;
        {
            let _l = self.lock.lock().unwrap();
            self.access_bitmap.reset_range(aligned_offset, aligned_bytes);
        }
        self.bcs.reset(aligned_offset, aligned_bytes);
        self.target.pdiscard(aligned_offset, aligned_bytes)
    }
}

impl Driver for CbwDriver {
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        bs.file().preadv_flags(offset, buf, 0)
    }

    fn pwrite(&self, bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        self.pwrite_flags(bs, offset, buf, 0)
    }

    fn pwrite_flags(&self, bs: &Node, offset: u64, buf: &[u8], flags: u32) -> io::Result<()> {
        self.copy_before_write(bs, offset, buf.len() as u64, flags)?;
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
        self.copy_before_write(bs, offset, bytes, flags)?;
        bs.file().pwrite_zeroes_flags(offset, bytes, flags)
    }

    fn supported_zero_flags(&self) -> u32 {
        self.zero_flags
    }

    fn pdiscard(&self, bs: &Node, offset: u64, bytes: u64) -> io::Result<()> {
        self.copy_before_write(bs, offset, bytes, 0)?;
        bs.file().pdiscard(offset, bytes)
    }

    fn flush_to_disk(&self, bs: &Node) -> io::Result<()> {
        bs.file().flush()
    }

    fn getlength(&self, bs: &Node) -> io::Result<u64> {
        bs.file().getlength()
    }

    fn has_truncate(&self) -> bool {
        false
    }

    fn exact_filename(&self, bs: &Node) -> Option<String> {
        bs.file().filename()
    }

    fn child_perm_for(&self, ctx: &PermCtx<'_>, perm: u64, shared: u64) -> (u64, u64) {
        if ctx.role & BDRV_CHILD_FILTERED == 0 {
            // Target child
            //
            // Share write to target (child_file), to not interfere with guest writes to its
            // disk which may be in target backing chain. Can't resize during a backup block
            // job because we check the size only upfront.
            return (BLK_PERM_WRITE, BLK_PERM_ALL & !BLK_PERM_RESIZE);
        }
        // Source child
        let (mut p, mut s) = default_perms(ctx, perm, shared);
        if ctx.parent.parent_count() > 0 {
            // Note, that source child may be shared with backup job. Backup job does create
            // own blk parent on copy-before-write node, so this works even if source node
            // does not have any parents before backup start
            p |= BLK_PERM_CONSISTENT_READ;
            if self.discard_source.load(Ordering::SeqCst) {
                p |= BLK_PERM_WRITE;
            }
            s &= !(BLK_PERM_WRITE | BLK_PERM_RESIZE);
        }
        (p, s)
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

fn cbw_of(bs: &Node) -> Option<&CbwDriver> {
    bs.driver.as_any().and_then(|a| a.downcast_ref::<CbwDriver>())
}

fn enotsup() -> io::Error {
    io::Error::from_raw_os_error(libc::ENOTSUP)
}

/// `bdrv_co_preadv_snapshot()`: reads the snapshot a copy-before-write node keeps.
pub(crate) fn preadv_snapshot(bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
    let Some(d) = cbw_of(bs) else {
        return Err(enotsup());
    };
    bs.inc_in_flight();
    let r = d.preadv_snapshot(bs, offset, buf);
    bs.dec_in_flight();
    r
}

/// `bdrv_co_snapshot_block_status()`.
pub(crate) fn snapshot_block_status(bs: &Node, offset: u64, bytes: u64) -> io::Result<BlockStatus> {
    let Some(d) = cbw_of(bs) else {
        return Err(enotsup());
    };
    bs.inc_in_flight();
    let r = d.snapshot_block_status(bs, offset, bytes);
    bs.dec_in_flight();
    r
}

/// `bdrv_co_pdiscard_snapshot()`.
pub(crate) fn pdiscard_snapshot(bs: &Node, offset: u64, bytes: u64) -> io::Result<()> {
    let Some(d) = cbw_of(bs) else {
        return Err(enotsup());
    };
    bs.inc_in_flight();
    let r = d.pdiscard_snapshot(offset, bytes);
    bs.dec_in_flight();
    r
}

/// `bdrv_cbw_append()`: puts a copy-before-write filter over `source` that copies to
/// `target`, and returns it with its block-copy state.
pub(crate) fn cbw_append(
    graph: &BlockGraph,
    source: &Arc<Node>,
    target: &Arc<Node>,
    filter_node_name: Option<&str>,
    discard_source: bool,
    min_cluster_size: u64,
    on_cbw_error: OnCbwError,
) -> Result<(Arc<Node>, Arc<BlockCopyState>)> {
    let mut opts = QDict::new();
    opts.put("driver", "copy-before-write");
    if let Some(n) = filter_node_name {
        opts.put("node-name", n);
    }
    opts.put("file", source.name.as_str());
    opts.put("target", target.name.as_str());
    opts.put("on-cbw-error", on_cbw_error.as_str());

    if min_cluster_size > i64::MAX as u64 {
        return Err(Error::generic(format!(
            "min-cluster-size too large: {min_cluster_size} > {}",
            i64::MAX
        )));
    }
    opts.put("min-cluster-size", min_cluster_size as i64);

    let top = insert_node(graph, source, opts)?;
    let d = cbw_of(&top).expect("the node has the copy-before-write driver");
    if discard_source {
        d.discard_source.store(true, Ordering::SeqCst);
        d.bcs.set_discard_source(true);
        if let Err(e) = top.file().refresh_perms() {
            let _ = drop_filter(&top);
            return Err(e);
        }
    }
    let bcs = d.bcs.clone();
    Ok((top, bcs))
}

/// `bdrv_cbw_drop()`.
pub(crate) fn cbw_drop(bs: &Arc<Node>) {
    drop_filter(bs).expect("dropping a copy-before-write filter cannot fail");
}
