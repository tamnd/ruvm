// SPDX-License-Identifier: GPL-2.0-or-later

//! `drive-mirror`, `blockdev-mirror` and active `block-commit` from block/mirror.c and
//! blockdev.c: a `mirror_top` filter goes on top of the source and records what the guest
//! writes in a dirty bitmap (or, in `write-blocking` copy mode, writes it to the target
//! too), while the job copies what is dirty to the target. Once source and target are in
//! sync the job is ready, and `block-job-complete` makes the target take the place of the
//! source (or of the `replaces` node).
//!
//! # Differences from QEMU
//!
//! - The job does one operation at a time on its thread, so there are no parallel in-flight
//!   operations and no buffer pool; `buf-size` only bounds how much one iteration copies.
//!   The ranges the job and the active writes work on are kept in a list that stands for
//!   `in_flight_bitmap` and `ops_in_flight`. An active write registers its range once the
//!   conflicting operations are gone, rather than before it waits for them.
//! - `drive-mirror` does not print the `Formatting ...` line when it creates the target,
//!   and with `mode=existing` it opens the target with its backing chain right away instead
//!   of opening the backing chain when the job completes.
//! - The `replaces` node is resolved when the job starts; `block-job-complete` fails with
//!   `Node name '...' not found` if that node went away in between.
//! - There is no `bdrv_cancel_in_flight()`: requests are synchronous, so a cancelled job
//!   stops after the operation it is doing.
//! - There are no implicit filter nodes, so a mirror never replaces one by default.

use std::any::Any;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::time::Instant;

use ruvm_base::report::report_error;
use ruvm_base::{Error, Result};
use ruvm_qapi::types::{
    BlockDeviceIoStatus, BlockErrorAction, BlockJobChangeOptions, BlockJobChangeOptionsU,
    BlockJobInfo, BlockJobInfoMirror, BlockJobInfoU, BlockdevDetectZeroesOptions,
    BlockdevMirrorArg, BlockdevOnError, DriveMirror, JobType, MirrorCopyMode, MirrorSyncMode,
    NewImageMode,
};
use ruvm_qapi::{QDict, QValue};

use super::block_job::{
    BLOCK_JOB_SLICE_TIME, BlockJobParams, block_job_add_bdrv, block_job_create, device_or_node_name,
};
use super::blocker::{
    BlockOpType, Reason, freeze_chain, op_block_all, op_is_blocked, op_unblock_all, unfreeze_chain,
};
use super::chain::{
    append_filter, chain_contains, check_to_replace_node, find_overlay, internal_open,
    recurse_can_replace,
};
use super::core::{Job, JobDriver, JobErr, job_lock};
use super::main_loop::bql_lock;
use super::stream::job_flags;
use crate::backend::BlockBackend;
use crate::bitmap::DirtyBitmap;
use crate::drain::DrainedSection;
use crate::drivers::DriverDef;
use crate::graph::{BlockGraph, OpenCtx};
use crate::node::{
    BDRV_BLOCK_DATA, BDRV_BLOCK_ZERO, BDRV_REQ_MAY_UNMAP, BDRV_REQ_NO_FALLBACK,
    BDRV_REQ_WRITE_UNCHANGED, Driver, Node, errno,
};
use crate::perm::{
    BLK_PERM_ALL, BLK_PERM_CONSISTENT_READ, BLK_PERM_RESIZE, BLK_PERM_WRITE,
    BLK_PERM_WRITE_UNCHANGED, PermCtx, default_perms,
};

const MAX_IN_FLIGHT: u64 = 16;
const MAX_IO_BYTES: u64 = 1 << 20; // 1 Mb
const DEFAULT_MIRROR_BUF_SIZE: u64 = MAX_IN_FLIGHT * MAX_IO_BYTES;
const BDRV_SECTOR_SIZE: u64 = 512;

/// `bdrv_mirror_top`.
pub(crate) static MIRROR_TOP: DriverDef =
    DriverDef::filter("mirror_top", internal_open).with_filtered_backing();

/// `MirrorMethod`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Method {
    Copy,
    Zero,
    Discard,
}

/// `BlockMirrorBackingMode`.
#[allow(clippy::enum_variant_names, reason = "the names of MIRROR_*_BACKING_CHAIN")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BackingMode {
    /// Reuse the existing backing chain from the source for the target: the target's
    /// backing node becomes the source's backing node (or the source for `sync=none`).
    SourceBackingChain,
    /// Open the target's backing chain completely anew.
    OpenBackingChain,
    /// Do not change the target's backing chain.
    LeaveBackingChain,
}

const COPY_MODE_BACKGROUND: u8 = 0;
const COPY_MODE_WRITE_BLOCKING: u8 = 1;

fn copy_mode_to_u8(m: MirrorCopyMode) -> u8 {
    match m {
        MirrorCopyMode::Background => COPY_MODE_BACKGROUND,
        MirrorCopyMode::WriteBlocking => COPY_MODE_WRITE_BLOCKING,
    }
}

fn copy_mode_from_u8(m: u8) -> MirrorCopyMode {
    if m == COPY_MODE_WRITE_BLOCKING {
        MirrorCopyMode::WriteBlocking
    } else {
        MirrorCopyMode::Background
    }
}

/// `bdrv_can_write_zeroes_with_unmap()`.
fn can_write_zeroes_with_unmap(bs: &Node) -> bool {
    bs.flags().unmap && bs.driver.supported_zero_flags() & BDRV_REQ_MAY_UNMAP != 0
}

/// `MirrorBDSOpaque`: what the `mirror_top` node knows about its job.
pub(crate) struct MirrorTopState {
    job: Mutex<Option<Arc<MirrorCore>>>,
    dirty_bitmap: Mutex<Option<Arc<DirtyBitmap>>>,
    stop: AtomicBool,
    is_commit: bool,
}

/// The `mirror_top` driver.
pub(crate) struct MirrorTop(Arc<MirrorTopState>);

impl MirrorTop {
    fn source(bs: &Node) -> io::Result<Arc<Node>> {
        bs.filter_child().map(|c| c.node).ok_or_else(|| errno(crate::node::ENOMEDIUM))
    }

    /// `bdrv_mirror_top_do_write()`.
    fn do_write(
        &self,
        bs: &Node,
        method: Method,
        offset: u64,
        bytes: u64,
        buf: Option<&[u8]>,
        flags: u32,
    ) -> io::Result<()> {
        let source = Self::source(bs)?;
        let core = self.0.job.lock().unwrap().clone();
        let copy_to_target = core.as_ref().filter(|c| c.should_copy_to_target());
        let op = copy_to_target.map(|c| c.active_write_prepare(offset, bytes));

        let ret = match method {
            Method::Copy => source.pwrite_flags(offset, buf.expect("a write has data"), flags),
            Method::Zero => source.pwrite_zeroes_flags(offset, bytes, flags),
            Method::Discard => source.pdiscard(offset, bytes),
        };

        if copy_to_target.is_none() {
            if let Some(c) = &core {
                c.actively_synced.store(false, Ordering::SeqCst);
            }
            if let Some(b) = self.0.dirty_bitmap.lock().unwrap().as_ref() {
                b.set_range(offset, bytes);
            }
        }

        if let Some(c) = copy_to_target {
            if ret.is_ok() {
                c.do_sync_target_write(method, offset, bytes, buf, flags);
            }
            c.active_write_settle(op.expect("registered above"));
        }
        ret
    }
}

impl Driver for MirrorTop {
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        Self::source(bs)?.pread(offset, buf)
    }

    fn pwrite(&self, bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        self.pwrite_flags(bs, offset, buf, 0)
    }

    fn pwrite_flags(&self, bs: &Node, offset: u64, buf: &[u8], flags: u32) -> io::Result<()> {
        // The data is ours for the whole request, so there is no need for a bounce buffer.
        self.do_write(bs, Method::Copy, offset, buf.len() as u64, Some(buf), flags)
    }

    fn supported_write_flags(&self) -> u32 {
        BDRV_REQ_WRITE_UNCHANGED
    }

    fn pwrite_zeroes(&self, bs: &Node, offset: u64, bytes: u64, may_unmap: bool) -> io::Result<()> {
        let flags = if may_unmap { BDRV_REQ_MAY_UNMAP } else { 0 };
        self.pwrite_zeroes_flags(bs, offset, bytes, flags)
    }

    fn pwrite_zeroes_flags(
        &self,
        bs: &Node,
        offset: u64,
        bytes: u64,
        flags: u32,
    ) -> io::Result<()> {
        self.do_write(bs, Method::Zero, offset, bytes, None, flags)
    }

    fn supported_zero_flags(&self) -> u32 {
        BDRV_REQ_WRITE_UNCHANGED | BDRV_REQ_NO_FALLBACK
    }

    fn pdiscard(&self, bs: &Node, offset: u64, bytes: u64) -> io::Result<()> {
        self.do_write(bs, Method::Discard, offset, bytes, None, 0)
    }

    fn getlength(&self, bs: &Node) -> io::Result<u64> {
        match bs.filter_child() {
            Some(c) => c.node.getlength(),
            None => Ok(bs.total_sectors().max(0) as u64 * BDRV_SECTOR_SIZE),
        }
    }

    fn has_truncate(&self) -> bool {
        false
    }

    fn exact_filename(&self, bs: &Node) -> Option<String> {
        // We can be here after failed bdrv_attach_child in bdrv_set_backing_hd
        bs.filter_child().map(|c| c.node.meta.lock().unwrap().filename.clone())
    }

    /// `bdrv_mirror_top_child_perm()`.
    fn child_perm_for(&self, ctx: &PermCtx<'_>, perm: u64, shared: u64) -> (u64, u64) {
        if self.0.stop.load(Ordering::SeqCst) {
            // If the job is to be stopped, we do not need to forward anything to the real
            // image.
            return (0, BLK_PERM_ALL);
        }
        let (mut p, mut s) = default_perms(ctx, perm, shared);
        if self.0.is_commit {
            // For commit jobs, we cannot take CONSISTENT_READ, because that permission is
            // unshared for everything above the base node (except for filters on the base
            // node). We also have to force-share the WRITE permission, or otherwise we would
            // block ourselves at the base node (if writes are blocked for a node, they are
            // also blocked for its backing file).
            p &= !BLK_PERM_CONSISTENT_READ;
            s |= BLK_PERM_WRITE;
        }
        (p, s)
    }
}

/// An operation on a range of chunks, which conflicting operations wait for.
struct Op {
    id: u64,
    start: u64,
    end: u64,
}

#[derive(Default)]
struct Ops {
    list: Vec<Op>,
    next_id: u64,
}

impl Ops {
    fn conflicts(&self, start: u64, end: u64) -> bool {
        self.list.iter().any(|o| o.start < end && start < o.end)
    }

    fn add(&mut self, start: u64, end: u64) -> u64 {
        self.next_id += 1;
        let id = self.next_id;
        self.list.push(Op { id, start, end });
        id
    }
}

/// The part of `MirrorBlockJob` that the `mirror_top` node uses too.
struct MirrorCore {
    job: Weak<Job>,
    target: Mutex<Option<Arc<BlockBackend>>>,
    dirty_bitmap: Arc<DirtyBitmap>,
    granularity: u64,
    bdev_length: AtomicU64,
    copy_mode: AtomicU8,
    /// Set when the target is synced (dirty bitmap is clean, nothing in flight) and the job
    /// is running in active mode.
    actively_synced: AtomicBool,
    ret: AtomicI32,
    on_source_error: BlockdevOnError,
    on_target_error: BlockdevOnError,
    unmap: bool,
    initial_zeroing_ongoing: AtomicBool,
    active_write_bytes_in_flight: AtomicU64,
    /// The ranges in flight, in chunks.
    ops: Mutex<Ops>,
    ops_done: Condvar,
    zero_bitmap: Mutex<Option<Vec<bool>>>,
}

impl MirrorCore {
    fn job(&self) -> Option<Arc<Job>> {
        self.job.upgrade()
    }

    fn target(&self) -> io::Result<Arc<BlockBackend>> {
        self.target.lock().unwrap().clone().ok_or_else(|| errno(crate::node::ENOMEDIUM))
    }

    fn chunks(&self, offset: u64, bytes: u64) -> (u64, u64) {
        (offset / self.granularity, (offset + bytes).div_ceil(self.granularity))
    }

    /// `mirror_error_action()`.
    fn error_action(&self, read: bool, error: i32) -> BlockErrorAction {
        self.actively_synced.store(false, Ordering::SeqCst);
        let Some(job) = self.job() else {
            return BlockErrorAction::Report;
        };
        if read {
            job.error_action(self.on_source_error, true, error)
        } else {
            job.error_action(self.on_target_error, false, error)
        }
    }

    /// `if (s->ret >= 0) s->ret = ret`.
    fn set_ret(&self, ret: i32) {
        let _ =
            self.ret.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |r| (r >= 0).then_some(ret));
    }

    fn ops(&self) -> MutexGuard<'_, Ops> {
        self.ops.lock().unwrap()
    }

    /// `mirror_wait_on_conflicts()`: waits until nothing is in flight in the range, or the
    /// job failed. Returns the lock, so that the caller can claim the range.
    fn wait_on_conflicts(&self, offset: u64, bytes: u64) -> MutexGuard<'_, Ops> {
        let (start, end) = self.chunks(offset, bytes);
        let mut ops = self.ops();
        while ops.conflicts(start, end) && self.ret.load(Ordering::SeqCst) >= 0 {
            ops = self.ops_done.wait(ops).unwrap();
        }
        ops
    }

    fn remove_op(&self, id: u64) {
        self.ops().list.retain(|o| o.id != id);
        self.ops_done.notify_all();
    }

    fn zero_bitmap_clear(&self, start: u64, end: u64) {
        if let Some(z) = self.zero_bitmap.lock().unwrap().as_mut() {
            let end = (end as usize).min(z.len());
            z[(start as usize).min(end)..end].fill(false);
        }
    }

    fn zero_bitmap_set(&self, start: u64, end: u64) {
        if let Some(z) = self.zero_bitmap.lock().unwrap().as_mut() {
            let end = (end as usize).min(z.len());
            z[(start as usize).min(end)..end].fill(true);
        }
    }

    /// Whether the zero bitmap exists and all of `start..end` is set in it.
    fn zero_bitmap_all(&self, start: u64, end: u64) -> bool {
        match self.zero_bitmap.lock().unwrap().as_ref() {
            Some(z) => {
                let end = (end as usize).min(z.len());
                z[(start as usize).min(end)..end].iter().all(|b| *b)
            }
            None => false,
        }
    }

    fn has_zero_bitmap(&self) -> bool {
        self.zero_bitmap.lock().unwrap().is_some()
    }

    /// `should_copy_to_target()`.
    fn should_copy_to_target(&self) -> bool {
        self.ret.load(Ordering::SeqCst) >= 0
            && self.job().is_some_and(|j| !j.is_cancelled())
            && self.copy_mode.load(Ordering::SeqCst) == COPY_MODE_WRITE_BLOCKING
    }

    /// `active_write_prepare()`.
    fn active_write_prepare(&self, offset: u64, bytes: u64) -> u64 {
        // Wait for concurrent requests affecting the area. If there are already running
        // requests that are copying off now-to-be stale data in the area, we must wait for
        // them to finish before we begin writing fresh data to the target so that the write
        // operations appear in the correct order.
        let (start, end) = self.chunks(offset, bytes);
        let mut ops = self.wait_on_conflicts(offset, bytes);
        ops.add(start, end)
    }

    /// `active_write_settle()`.
    fn active_write_settle(&self, id: u64) {
        self.remove_op(id);
    }

    /// `do_sync_target_write()`.
    fn do_sync_target_write(
        &self,
        method: Method,
        mut offset: u64,
        mut bytes: u64,
        buf: Option<&[u8]>,
        flags: u32,
    ) {
        let gran = self.granularity;
        let bm = &self.dirty_bitmap;
        let mut qiov_offset = 0u64;
        if offset % gran != 0 && bm.get(offset) {
            // Dirty unaligned padding: ignore it. If we copy it, we can't reset the
            // corresponding bit in dirty_bitmap as there may be some "dirty" bytes still not
            // copied, and it's already dirty, so skipping it we don't diverge mirror
            // progress.
            qiov_offset = offset.next_multiple_of(gran) - offset;
            if bytes <= qiov_offset {
                // nothing to do after shrink
                return;
            }
            offset += qiov_offset;
            bytes -= qiov_offset;
        }
        if (offset + bytes) % gran != 0 && bm.get(offset + bytes - 1) {
            let tail = (offset + bytes) % gran;
            if bytes <= tail {
                // nothing to do after shrink
                return;
            }
            bytes -= tail;
        }

        // Tails are either clean or shrunk, so for dirty bitmap resetting we safely align
        // the range narrower. But for zero bitmap, round range wider for checking or
        // clearing, and narrower for setting.
        let dirty_offset = offset.next_multiple_of(gran);
        let dirty_end = (offset + bytes) / gran * gran;
        if dirty_offset < dirty_end {
            bm.reset_range(dirty_offset, dirty_end - dirty_offset);
        }
        let (zero_start, zero_end) = self.chunks(offset, bytes);

        let job = self.job();
        if let Some(j) = &job {
            j.progress_increase_remaining(bytes);
        }
        self.active_write_bytes_in_flight.fetch_add(bytes, Ordering::SeqCst);

        let ret = self.target().and_then(|target| match method {
            Method::Copy => {
                self.zero_bitmap_clear(zero_start, zero_end);
                let buf = buf.expect("a write has data");
                let q = qiov_offset as usize;
                target.pwrite(offset, &buf[q..q + bytes as usize])
            }
            Method::Zero => {
                if self.zero_bitmap_all(zero_start, zero_end) {
                    return Ok(());
                }
                let r = target.pwrite_zeroes(offset, bytes, flags & BDRV_REQ_MAY_UNMAP != 0);
                if r.is_ok() && dirty_offset < dirty_end {
                    self.zero_bitmap_set(dirty_offset / gran, dirty_end / gran);
                }
                r
            }
            Method::Discard => {
                self.zero_bitmap_clear(zero_start, zero_end);
                target.pdiscard(offset, bytes)
            }
        });

        self.active_write_bytes_in_flight.fetch_sub(bytes, Ordering::SeqCst);
        match ret {
            Ok(()) => {
                if let Some(j) = &job {
                    j.progress_update(bytes);
                }
            }
            Err(e) => {
                // We failed, so we should mark dirty the whole area, aligned up. Note that we
                // don't care about shrunk tails if any: they were dirty at function start,
                // and they must be still dirty, as we've locked the region for in-flight op.
                let start = offset / gran * gran;
                let end = (offset + bytes).next_multiple_of(gran);
                bm.set_range(start, end.min(bm.size()) - start);
                self.actively_synced.store(false, Ordering::SeqCst);
                let e = e.raw_os_error().unwrap_or(libc::EIO);
                if self.error_action(false, e) == BlockErrorAction::Report {
                    let _ = self.ret.compare_exchange(0, -e, Ordering::SeqCst, Ordering::SeqCst);
                }
            }
        }
    }
}

/// What the job thread keeps while it runs.
struct RunState {
    source: Arc<Node>,
    target: Arc<BlockBackend>,
    target_bs: Arc<Node>,
    buf_size: u64,
    max_iov: u64,
    target_cluster_size: u64,
    cow_bitmap: Option<Vec<bool>>,
    last_pause: Instant,
    /// Where the dirty bitmap iterator is.
    dbi: u64,
}

/// `MirrorBlockJob`.
struct MirrorJob {
    core: Arc<MirrorCore>,
    top: Arc<MirrorTopState>,
    mirror_top_bs: Mutex<Option<Arc<Node>>>,
    target_bs: Arc<Node>,
    base: Option<Arc<Node>>,
    base_overlay: Option<Arc<Node>>,
    /// The name of the graph node to replace
    replaces: Option<String>,
    replaces_node: Option<Weak<Node>>,
    /// The BDS to replace
    to_replace: Mutex<Option<Arc<Node>>>,
    /// Used to block operations on the drive-mirror-replace target
    replace_blocker: Arc<Reason>,
    sync_mode: MirrorSyncMode,
    backing_mode: BackingMode,
    /// Whether the target should be assumed to be already zero initialized
    target_is_zero: bool,
    should_complete: AtomicBool,
    buf_size: u64,
    prepared: AtomicBool,
    in_drain: AtomicBool,
    drain: Mutex<Option<DrainedSection>>,
    base_ro: bool,
    /// `mirror_job_driver` rather than `commit_active_job_driver`.
    is_mirror: bool,
}

impl MirrorJob {
    fn mirror_top(&self) -> Arc<Node> {
        self.mirror_top_bs.lock().unwrap().clone().expect("the job has its filter")
    }

    /// Clip bytes relative to offset to not exceed end-of-file
    fn clip_bytes(&self, offset: u64, bytes: u64) -> u64 {
        bytes.min(self.core.bdev_length.load(Ordering::SeqCst).saturating_sub(offset))
    }

    /// `mirror_cow_align()`: rounds offset and/or bytes to target cluster if COW is needed,
    /// and returns the offset of the adjusted tail against original.
    fn cow_align(&self, run: &RunState, offset: &mut u64, bytes: &mut u64) -> u64 {
        let gran = self.core.granularity;
        let cow = run.cow_bitmap.as_ref().expect("only with a cow bitmap");
        let max_bytes = gran * run.max_iov;
        let mut need_cow = !cow[(*offset / gran) as usize];
        need_cow |= !cow[((*offset + *bytes - 1) / gran) as usize];
        let (align_offset, mut align_bytes) = if need_cow {
            run.target_bs.round_to_subclusters(*offset, *bytes)
        } else {
            (*offset, *bytes)
        };
        if align_bytes > max_bytes {
            align_bytes = max_bytes;
            if need_cow {
                align_bytes = align_bytes / run.target_cluster_size * run.target_cluster_size;
            }
        }
        // Clipping may result in align_bytes unaligned to chunk boundary, but that doesn't
        // matter because it's already the end of source image.
        align_bytes = self.clip_bytes(align_offset, align_bytes);
        let ret = (align_offset + align_bytes) as i64 - (*offset + *bytes) as i64;
        *offset = align_offset;
        *bytes = align_bytes;
        assert!(ret >= 0);
        ret as u64
    }

    /// `mirror_iteration_done()` after `mirror_write_complete()`/`mirror_read_complete()`.
    fn op_done(&self, job: &Job, run: &mut RunState, id: u64, offset: u64, bytes: u64, ok: bool) {
        let (start, end) = self.core.chunks(offset, bytes);
        self.core.remove_op(id);
        if ok {
            if let Some(cow) = run.cow_bitmap.as_mut() {
                let end = (end as usize).min(cow.len());
                cow[(start as usize).min(end)..end].fill(true);
            }
            if !self.core.initial_zeroing_ongoing.load(Ordering::SeqCst) {
                job.progress_update(bytes);
            }
        }
    }

    /// The error half of `mirror_read_complete()` and `mirror_write_complete()`.
    fn op_failed(&self, offset: u64, bytes: u64, read: bool, e: &io::Error) {
        let e = e.raw_os_error().unwrap_or(libc::EIO);
        let bm = &self.core.dirty_bitmap;
        bm.set_range(offset, bytes.min(bm.size().saturating_sub(offset)));
        if self.core.error_action(read, e) == BlockErrorAction::Report {
            self.core.set_ret(-e);
        }
    }

    /// `mirror_perform()` with `mirror_co_read()`, `mirror_co_zero()` and
    /// `mirror_co_discard()`: returns the bytes handled, and whether the I/O was skipped.
    fn perform(
        &self,
        job: &Job,
        run: &mut RunState,
        offset: u64,
        bytes: u64,
        method: Method,
    ) -> (u64, bool) {
        let core = &self.core;
        let gran = core.granularity;
        assert!(offset % gran == 0);
        let (zs, ze) = core.chunks(offset, bytes);
        match method {
            Method::Copy => {
                core.zero_bitmap_clear(zs, ze);
                let max_bytes = gran * run.max_iov;
                // We can only handle as much as buf_size at a time.
                let mut op_offset = offset;
                let mut op_bytes = run.buf_size.min(max_bytes).min(bytes);
                assert!(op_bytes > 0);
                let mut handled = op_bytes;
                if run.cow_bitmap.is_some() {
                    handled += self.cow_align(run, &mut op_offset, &mut op_bytes);
                }
                let (s, e) = core.chunks(op_offset, op_bytes);
                let id = core.ops().add(s, e);
                let mut buf = vec![0u8; op_bytes as usize];
                let ok = match run.source.pread(op_offset, &mut buf) {
                    Err(e) => {
                        self.op_failed(op_offset, op_bytes, true, &e);
                        false
                    }
                    Ok(()) => match run.target.pwrite(op_offset, &buf) {
                        Err(e) => {
                            self.op_failed(op_offset, op_bytes, false, &e);
                            false
                        }
                        Ok(()) => true,
                    },
                };
                self.op_done(job, run, id, op_offset, op_bytes, ok);
                (handled, false)
            }
            Method::Zero => {
                let mut skipped = false;
                let (s, e) = core.chunks(offset, bytes);
                let id = core.ops().add(s, e);
                let mut ret = Ok(());
                if core.zero_bitmap_all(zs, ze) {
                    skipped = true;
                } else {
                    ret = run.target.pwrite_zeroes(offset, bytes, core.unmap);
                }
                if ret.is_ok() && core.has_zero_bitmap() {
                    core.zero_bitmap_set(zs, ze);
                }
                if let Err(e) = &ret {
                    self.op_failed(offset, bytes, false, e);
                }
                self.op_done(job, run, id, offset, bytes, ret.is_ok());
                (bytes, skipped)
            }
            Method::Discard => {
                core.zero_bitmap_clear(zs, ze);
                let (s, e) = core.chunks(offset, bytes);
                let id = core.ops().add(s, e);
                let ret = run.target.pdiscard(offset, bytes);
                if let Err(e) = &ret {
                    self.op_failed(offset, bytes, false, e);
                }
                self.op_done(job, run, id, offset, bytes, ret.is_ok());
                (bytes, false)
            }
        }
    }

    /// `mirror_iteration()`.
    fn iteration(&self, job: &Job, run: &mut RunState) {
        let core = &self.core;
        let gran = core.granularity;
        let len = core.bdev_length.load(Ordering::SeqCst);
        let bm = &core.dirty_bitmap;
        let write_zeroes_ok = can_write_zeroes_with_unmap(&run.target_bs);
        let max_io_bytes = (run.buf_size / MAX_IN_FLIGHT).max(MAX_IO_BYTES);

        let mut offset = match bm.next_dirty(run.dbi, len) {
            Some(o) => o,
            None => match bm.next_dirty(0, len) {
                Some(o) => o,
                // An active write cleaned the bitmap meanwhile.
                None => return,
            },
        };

        // Wait for concurrent requests to @offset. The next loop will limit the copied area
        // based on the requests in flight so we only copy an area that does not overlap with
        // concurrent in-flight requests. Still, we would like to copy something, so wait
        // until there are at least no more requests to the very beginning of the area.
        drop(core.wait_on_conflicts(offset, 1));

        job.pause_point();

        // Find the number of consecutive dirty chunks following the first dirty one, and
        // wait for in flight requests in them.
        let mut ops = core.wait_on_conflicts(offset, 1);
        // At least the first dirty chunk is mirrored in one iteration.
        let mut nb_chunks = 1u64;
        while nb_chunks * gran < run.buf_size {
            let next_offset = offset + nb_chunks * gran;
            let next_chunk = next_offset / gran;
            if next_offset >= len || !bm.get(next_offset) {
                break;
            }
            if ops.conflicts(next_chunk, next_chunk + 1) {
                break;
            }
            nb_chunks += 1;
        }
        run.dbi = offset + nb_chunks * gran;

        // Clear dirty bits before querying the block status, because if some blocks are
        // marked dirty in this window, we need to know.
        bm.reset_range(offset, (nb_chunks * gran).min(len - offset));

        // Before claiming an area, we have to create an operation for it so that conflicting
        // requests can wait for it: a pseudo operation for the whole area, removed once all
        // real operations are done.
        let first_chunk = offset / gran;
        let pseudo_op = ops.add(first_chunk, first_chunk + nb_chunks);
        drop(ops);

        while nb_chunks > 0 && offset < len {
            let mut method = Method::Copy;
            assert!(offset % gran == 0);
            let st = run.source.block_status_above(None, offset, nb_chunks * gran);
            let mut io_bytes = match &st {
                Err(_) => (nb_chunks * gran).min(max_io_bytes),
                Ok(s) if s.ret & BDRV_BLOCK_DATA != 0 => s.pnum.min(max_io_bytes),
                Ok(s) => s.pnum,
            };
            io_bytes -= io_bytes % gran;
            if io_bytes < gran {
                io_bytes = gran;
            } else if let Ok(s) = &st {
                if s.ret & BDRV_BLOCK_DATA == 0 {
                    let (t_off, t_bytes) = run.target_bs.round_to_subclusters(offset, io_bytes);
                    if t_off == offset && t_bytes == io_bytes {
                        method = if s.ret & BDRV_BLOCK_ZERO != 0 {
                            Method::Zero
                        } else {
                            Method::Discard
                        };
                    }
                }
            }

            if core.ret.load(Ordering::SeqCst) < 0 {
                break;
            }

            io_bytes = self.clip_bytes(offset, io_bytes);
            let (handled, io_skipped) = self.perform(job, run, offset, io_bytes, method);
            let acct =
                if io_skipped || (method != Method::Copy && write_zeroes_ok) { 0 } else { handled };
            assert!(handled > 0);
            offset += handled;
            nb_chunks = nb_chunks.saturating_sub(handled.div_ceil(gran));
            job.ratelimit_processed_bytes(acct);
        }

        core.remove_op(pseudo_op);
    }

    /// `mirror_throttle()`.
    fn throttle(&self, job: &Job, run: &mut RunState) {
        let now = Instant::now();
        if now.duration_since(run.last_pause).as_nanos() > u128::from(BLOCK_JOB_SLICE_TIME) {
            run.last_pause = now;
            job.sleep_ns(0);
        } else {
            job.pause_point();
        }
    }

    /// `mirror_dirty_init()`.
    fn dirty_init(&self, job: &Job, run: &mut RunState) -> Result<(), JobErr> {
        let core = &self.core;
        let gran = core.granularity;
        let len = core.bdev_length.load(Ordering::SeqCst);
        let target_bs = run.target_bs.clone();
        let punch_holes = target_bs.flags().detect_zeroes == BlockdevDetectZeroesOptions::Unmap
            && can_write_zeroes_with_unmap(&target_bs);
        let bitmap_length = len.div_ceil(gran);
        let chunk = (i32::MAX as u64) / gran * gran;

        // Determine if the image is already zero, regardless of sync mode.
        *core.zero_bitmap.lock().unwrap() = Some(vec![false; bitmap_length as usize]);
        let zero = if self.target_is_zero { true } else { target_bs.is_all_zeroes()? };

        // Determine if a pre-zeroing pass is necessary.
        if self.sync_mode == MirrorSyncMode::Top {
            // In TOP mode, there is no benefit to a pre-zeroing pass, but the zero bitmap can
            // be set if the destination already reads as zero and we are not punching holes.
            if zero && !punch_holes {
                core.zero_bitmap_set(0, bitmap_length);
            }
        } else if !zero || punch_holes {
            // Here, we are in FULL mode; our goal is to avoid writing zeroes if the
            // destination already reads as zero, except when we are trying to punch holes.
            // If pre-zeroing is not fast, or we need to visit the entire image in order to
            // punch holes even in the non-allocated regions of the source, then just mark
            // the entire image dirty and leave the zero bitmap clear at this point in time.
            // Otherwise, it can be faster to pre-zero the image now, even if we re-write the
            // allocated portions of the disk later, and the pre-zero pass will populate the
            // zero bitmap.
            if !can_write_zeroes_with_unmap(&target_bs) || punch_holes {
                core.dirty_bitmap.set_range(0, len);
                return Ok(());
            }
            core.initial_zeroing_ongoing.store(true, Ordering::SeqCst);
            let mut offset = 0;
            while offset < len {
                let bytes = (len - offset).min(chunk);
                self.throttle(job, run);
                if job.is_cancelled() {
                    core.initial_zeroing_ongoing.store(false, Ordering::SeqCst);
                    return Ok(());
                }
                self.perform(job, run, offset, bytes, Method::Zero);
                offset += bytes;
            }
            core.initial_zeroing_ongoing.store(false, Ordering::SeqCst);
        } else {
            // In FULL mode, and image already reads as zero.
            core.zero_bitmap_set(0, bitmap_length);
        }

        // First part, loop on the sectors and initialize the dirty bitmap.
        let mut offset = 0;
        while offset < len {
            // Just to make sure we are not exceeding int limit.
            let bytes = (len - offset).min(chunk);
            self.throttle(job, run);
            if job.is_cancelled() {
                return Ok(());
            }
            let (depth, count) =
                run.source.is_allocated_above(self.base_overlay.as_ref(), true, offset, bytes)?;
            assert!(count > 0);
            if depth > 0 {
                core.dirty_bitmap.set_range(offset, count);
            }
            offset += count;
        }
        Ok(())
    }

    /// `mirror_flush()`: called when going out of the streaming phase to flush the bulk of
    /// the data to the medium, or just before completing.
    fn flush(&self, run: &RunState) -> io::Result<()> {
        run.target.flush().inspect_err(|e| {
            let e = e.raw_os_error().unwrap_or(libc::EIO);
            if self.core.error_action(false, e) == BlockErrorAction::Report {
                self.core.ret.store(-e, Ordering::SeqCst);
            }
        })
    }

    /// `mirror_run()` up to `immediate_exit`.
    fn run_body(
        &self,
        job: &Arc<Job>,
        bs: &Arc<Node>,
        need_drain: &mut bool,
    ) -> Result<(), JobErr> {
        let core = &self.core;
        let target = core.target()?;
        let target_bs = self.target_bs.clone();

        if job.is_cancelled() {
            return Ok(());
        }
        let bdev_length = bs.getlength()?;
        core.bdev_length.store(bdev_length, Ordering::SeqCst);
        let target_length = target.getlength()?;

        // Active commit must resize the base image if its size differs from the active
        // layer.
        if self.base.as_ref().is_some_and(|b| Arc::ptr_eq(b, &target_bs)) {
            if bdev_length > target_length {
                target.truncate(bdev_length).map_err(|e| JobErr::with(libc::EIO, e))?;
            }
        } else if bdev_length != target_length {
            return Err(JobErr::with(
                libc::EINVAL,
                Error::generic("Source and target image have different sizes"),
            ));
        }

        if bdev_length == 0 {
            // Transition to the READY state and wait for complete.
            job.transition_to_ready();
            core.actively_synced.store(true, Ordering::SeqCst);
            while !job.cancel_requested() && !self.should_complete.load(Ordering::SeqCst) {
                job.yield_();
            }
            return Ok(());
        }

        let length = bdev_length.div_ceil(core.granularity);
        let mut run = RunState {
            source: bs.clone(),
            target,
            target_bs: target_bs.clone(),
            buf_size: self.buf_size,
            max_iov: 1024,
            target_cluster_size: BDRV_SECTOR_SIZE,
            cow_bitmap: None,
            last_pause: Instant::now(),
            dbi: 0,
        };

        // If we have no backing file yet in the destination, we cannot let the destination
        // do COW. Instead, we copy sectors around the dirty data if needed. We need a bitmap
        // to do that.
        let has_backing_filename = !target_bs.meta.lock().unwrap().backing_file.is_empty();
        if let Ok(bdi) = target_bs.get_info() {
            if bdi.cluster_size != 0 {
                run.target_cluster_size = bdi.cluster_size;
            }
        }
        if has_backing_filename
            && target_bs.backing_chain_next().is_none()
            && core.granularity < run.target_cluster_size
        {
            run.buf_size = run.buf_size.max(run.target_cluster_size);
            run.cow_bitmap = Some(vec![false; length as usize]);
        }
        let iov = |n: u32| if n == 0 { 1024 } else { u64::from(n) };
        run.max_iov = iov(bs.limits().max_iov).min(iov(target_bs.limits().max_iov));

        run.last_pause = Instant::now();
        if self.sync_mode != MirrorSyncMode::None {
            self.dirty_init(job, &mut run)?;
            if job.is_cancelled() {
                return Ok(());
            }
        }

        // Only now the job is fully initialised and mirror_top_bs should start accessing it.
        *self.top.job.lock().unwrap() = Some(core.clone());

        loop {
            let ret = core.ret.load(Ordering::SeqCst);
            if ret < 0 {
                return Err(JobErr::errno(-ret));
            }

            job.pause_point();

            if job.is_cancelled() {
                return Ok(());
            }

            let mut cnt = core.dirty_bitmap.count();
            // cnt is the number of dirty bytes remaining; together with the active writes
            // those are the current remaining operation length
            job.progress_set_remaining(
                cnt + core.active_write_bytes_in_flight.load(Ordering::SeqCst),
            );

            // Note that even when no rate limit is applied we need to yield periodically with
            // no pending I/O so that bdrv_drain_all() returns. We do so every
            // BLOCK_JOB_SLICE_TIME nanoseconds, or when there is an error, or when the source
            // is clean, whichever comes first.
            let delta = run.last_pause.elapsed().as_nanos();
            let iostatus = job.s().iostatus;
            if delta < u128::from(BLOCK_JOB_SLICE_TIME)
                && iostatus == BlockDeviceIoStatus::Ok
                && cnt != 0
            {
                self.iteration(job, &mut run);
            }

            let mut should_complete = false;
            if cnt == 0 {
                if !job.is_ready() {
                    if self.flush(&run).is_err() {
                        // Go check s->ret.
                        continue;
                    }
                    // We're out of the streaming phase. From now on, if the job is cancelled
                    // we will actually complete all pending I/O and report completion. This
                    // way, block-job-cancel will leave the target in a consistent state.
                    job.transition_to_ready();
                }
                if core.copy_mode.load(Ordering::SeqCst) != COPY_MODE_BACKGROUND {
                    core.actively_synced.store(true, Ordering::SeqCst);
                }
                should_complete =
                    self.should_complete.load(Ordering::SeqCst) || job.cancel_requested();
                cnt = core.dirty_bitmap.count();
            }

            if cnt == 0 && should_complete {
                // Note that I/O can be submitted by the guest while mirror_populate runs, so
                // pause it now. Before deciding whether to switch to target check one last
                // time if I/O has come in the meanwhile, and if not flush the data to disk.
                self.in_drain.store(true, Ordering::SeqCst);
                let d = bs.drained();
                cnt = core.dirty_bitmap.count();
                if cnt > 0 || self.flush(&run).is_err() {
                    drop(d);
                    self.in_drain.store(false, Ordering::SeqCst);
                    continue;
                }
                // The two disks are in sync. Exit and report successful completion.
                *self.drain.lock().unwrap() = Some(d);
                *need_drain = false;
                return Ok(());
            }

            if job.is_ready() && !should_complete {
                if cnt == 0 {
                    job.sleep_ns(BLOCK_JOB_SLICE_TIME as i64);
                }
            } else {
                job.ratelimit_sleep();
            }
            run.last_pause = Instant::now();
        }
    }

    /// `mirror_exit_common()`.
    fn exit_common(&self, job: &Arc<Job>) -> i32 {
        if self.prepared.swap(true, Ordering::SeqCst) {
            return 0;
        }
        let abort = job.s().ret < 0;
        let mut ret = 0;
        let mirror_top = self.mirror_top();
        let src = mirror_top.filter_child().map(|c| c.node).expect("mirror_top has its source");
        let target_bs = self.target_bs.clone();

        if chain_contains(&src, &target_bs) {
            unfreeze_chain(&mirror_top, Some(&target_bs));
        }
        mirror_top.release_dirty_bitmap(&self.core.dirty_bitmap);

        // Remove target parent that still uses BLK_PERM_WRITE/RESIZE before inserting
        // target_bs at s->to_replace, where we might not be able to get these permissions.
        self.core.target.lock().unwrap().take();

        // We don't access the source any more. Dropping any WRITE/RESIZE is required before
        // it could become a backing file of target_bs. Not having these permissions any more
        // means that we can't allow any new requests on mirror_top_bs from now on, so keep
        // it drained.
        mirror_top.drained_begin();
        target_bs.drained_begin();
        self.top.stop.store(true, Ordering::SeqCst);
        if let Some(c) = mirror_top.filter_child() {
            mirror_top.refresh_child_perms(&c).expect("dropping permissions cannot fail");
        }

        if !abort && self.backing_mode == BackingMode::SourceBackingChain {
            let unfiltered_target = target_bs.skip_filters();
            let backing = if self.sync_mode == MirrorSyncMode::None {
                Some(src.clone())
            } else {
                self.base.clone()
            };
            let cur = unfiltered_target.cow_child().map(|c| c.node);
            let same = match (&cur, &backing) {
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                (None, None) => true,
                _ => false,
            };
            if !same {
                if let Err(e) = unfiltered_target.set_backing_hd(backing) {
                    report_error(&e);
                    ret = -libc::EPERM;
                }
            }
        }

        if self.should_complete.load(Ordering::SeqCst) && !abort {
            let to_replace = self.to_replace.lock().unwrap().clone().unwrap_or_else(|| src.clone());
            let ro = to_replace.read_only();
            if ro != target_bs.read_only() {
                let _ = target_bs.reopen_set_read_only(ro);
            }
            // The mirror job has no requests in flight any more, but we need to drain
            // potential other users of the BDS before changing the graph.
            assert!(self.in_drain.load(Ordering::SeqCst));
            to_replace.drained_begin();
            // Cannot use check_to_replace_node() here, because that would check for an op
            // blocker on @to_replace, and we have our own there.
            let r = if recurse_can_replace(&src, &to_replace) {
                Node::replace_node(&to_replace, &target_bs)
            } else {
                Err(Error::generic(format!(
                    "Can no longer replace '{}' by '{}', because it can no longer be \
                     guaranteed that doing so would not lead to an abrupt change of visible \
                     data",
                    to_replace.name, target_bs.name
                )))
            };
            to_replace.drained_end();
            if let Err(e) = r {
                report_error(&e);
                ret = -libc::EPERM;
            }
        }
        if let Some(tr) = self.to_replace.lock().unwrap().take() {
            op_unblock_all(&tr, &self.replace_blocker);
        }

        // Remove the mirror filter driver from the graph. Before this, get rid of the
        // blockers on the intermediate nodes so that the resulting state is valid.
        job.bj().remove_all_bdrv();
        if let Some(backing) = mirror_top.filter_child().map(|c| c.node) {
            Node::replace_node(&mirror_top, &backing).expect("dropping mirror_top");
        }

        if abort && self.base_ro && !target_bs.read_only() {
            let _ = target_bs.reopen_set_read_only(true);
        }

        target_bs.drained_end();
        *self.top.job.lock().unwrap() = None;
        self.drain.lock().unwrap().take();
        mirror_top.drained_end();
        self.in_drain.store(false, Ordering::SeqCst);
        ret
    }
}

impl JobDriver for MirrorJob {
    /// `mirror_run()`.
    fn run(&self, job: &Arc<Job>) -> Result<(), JobErr> {
        let bs =
            self.mirror_top().filter_child().map(|c| c.node).expect("mirror_top has its source");
        let mut need_drain = true;
        let r = self.run_body(job, &bs, &mut need_drain);
        *self.core.zero_bitmap.lock().unwrap() = None;
        if need_drain {
            self.in_drain.store(true, Ordering::SeqCst);
            *self.drain.lock().unwrap() = Some(bs.drained());
        }
        r
    }

    /// `mirror_prepare()`.
    fn prepare(&self, job: &Arc<Job>) -> Option<Result<(), JobErr>> {
        let r = self.exit_common(job);
        Some(if r < 0 { Err(JobErr::errno(-r)) } else { Ok(()) })
    }

    /// `mirror_abort()`.
    fn abort(&self, job: &Arc<Job>) {
        let r = self.exit_common(job);
        debug_assert_eq!(r, 0);
    }

    fn clean(&self, _job: &Arc<Job>) {
        self.mirror_top_bs.lock().unwrap().take();
    }

    fn has_complete(&self) -> bool {
        true
    }

    /// `mirror_complete()`.
    fn complete(&self, job: &Arc<Job>) -> Result<()> {
        if !job.is_ready() {
            return Err(Error::generic(format!(
                "The active block job '{}' cannot be completed",
                job.id_str()
            )));
        }
        if !self.should_complete.load(Ordering::SeqCst) {
            // block all operations on to_replace bs
            if let Some(replaces) = &self.replaces {
                let Some(n) = self.replaces_node.as_ref().and_then(Weak::upgrade) else {
                    return Err(Error::generic(format!("Node name '{replaces}' not found")));
                };
                op_block_all(&n, &self.replace_blocker);
                *self.to_replace.lock().unwrap() = Some(n);
            }
            self.should_complete.store(true, Ordering::SeqCst);
        }
        // If the job is paused, it will be re-entered when it is resumed
        let mut jl = job_lock();
        let paused = job.s().paused;
        if !paused {
            job.enter_cond_locked(&mut jl, None);
        }
        Ok(())
    }

    /// `mirror_cancel()` and `commit_active_cancel()`: before the job is READY, any
    /// cancellation is a force-cancellation.
    fn cancel(&self, job: &Arc<Job>, force: bool) -> Option<bool> {
        Some(force || !job.is_ready())
    }

    /// `mirror_drained_poll()`.
    fn drained_poll(&self, job: &Arc<Job>) -> bool {
        // If the job isn't paused nor cancelled, we can't be sure that it won't issue more
        // requests. We make an exception if we've reached this point from one of our own
        // drain sections, to avoid a deadlock waiting for ourselves.
        let mut jl = job_lock();
        let paused = job.s().paused;
        !paused && !job.is_cancelled_locked(&mut jl) && !self.in_drain.load(Ordering::SeqCst)
    }

    /// `mirror_change()`.
    fn change(&self, _job: &Arc<Job>, opts: &BlockJobChangeOptions) -> Option<Result<()>> {
        if !self.is_mirror {
            return None;
        }
        let BlockJobChangeOptionsU::Mirror(m) = &opts.u else {
            return Some(Ok(()));
        };
        let core = &self.core;
        if core.copy_mode.load(Ordering::SeqCst) == copy_mode_to_u8(m.copy_mode) {
            return Some(Ok(()));
        }
        if m.copy_mode != MirrorCopyMode::WriteBlocking {
            return Some(Err(Error::generic(format!(
                "Change to copy mode '{}' is not implemented",
                m.copy_mode.as_str()
            ))));
        }
        if let Err(current) = core.copy_mode.compare_exchange(
            COPY_MODE_BACKGROUND,
            COPY_MODE_WRITE_BLOCKING,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            return Some(Err(Error::generic(format!(
                "Expected current copy mode '{}', got '{}'",
                MirrorCopyMode::Background.as_str(),
                copy_mode_from_u8(current).as_str()
            ))));
        }
        Some(Ok(()))
    }

    /// `mirror_query()`.
    fn query(&self, _job: &Arc<Job>, info: &mut BlockJobInfo) {
        if self.is_mirror {
            info.u = BlockJobInfoU::Mirror(BlockJobInfoMirror {
                actively_synced: self.core.actively_synced.load(Ordering::SeqCst),
            });
        }
    }

    fn as_any(&self) -> Option<&dyn Any> {
        Some(self)
    }
}

/// What `mirror_start_job()` gets.
struct StartParams<'a> {
    job_id: Option<&'a str>,
    job_type: JobType,
    bs: Arc<Node>,
    creation_flags: u32,
    target: Arc<Node>,
    replaces: Option<(String, Arc<Node>)>,
    speed: i64,
    granularity: u32,
    buf_size: i64,
    sync_mode: MirrorSyncMode,
    backing_mode: BackingMode,
    target_is_zero: bool,
    on_source_error: BlockdevOnError,
    on_target_error: BlockdevOnError,
    unmap: bool,
    base: Option<Arc<Node>>,
    auto_complete: bool,
    filter_node_name: Option<&'a str>,
    is_mirror: bool,
    copy_mode: MirrorCopyMode,
    base_ro: bool,
}

/// `mirror_start_job()`.
fn mirror_start_job(graph: &BlockGraph, p: StartParams<'_>) -> Result<Arc<Job>> {
    let (bs, target) = (&p.bs, &p.target);
    let granularity =
        if p.granularity == 0 { target.default_bitmap_granularity() } else { p.granularity };
    assert!(granularity.is_power_of_two());
    if p.buf_size < 0 {
        return Err(Error::generic("Invalid parameter 'buf-size'"));
    }
    let buf_size = if p.buf_size == 0 { DEFAULT_MIRROR_BUF_SIZE } else { p.buf_size as u64 };
    if Arc::ptr_eq(&bs.skip_filters(), &target.skip_filters()) {
        return Err(Error::generic("Can't mirror node into itself"));
    }
    let target_is_backing = chain_contains(bs, target);

    // In the case of active commit, add dummy driver to provide consistent reads on the top,
    // while disabling it in the intermediate nodes, and make the backing chain writable.
    let top = Arc::new(MirrorTopState {
        job: Mutex::new(None),
        dirty_bitmap: Mutex::new(None),
        stop: AtomicBool::new(false),
        is_commit: target_is_backing,
    });
    let mirror_top = {
        let _d = bs.drained();
        append_filter(graph, bs, p.filter_node_name, &MIRROR_TOP, Box::new(MirrorTop(top.clone())))?
    };
    let bitmap = match mirror_top.create_dirty_bitmap(granularity, None) {
        Ok(b) => b,
        Err(e) => {
            top.stop.store(true, Ordering::SeqCst);
            let _d = bs.drained();
            Node::replace_node(&mirror_top, bs).expect("dropping mirror_top");
            return Err(e);
        }
    };
    // The mirror job doesn't use the block layer's dirty tracking because it needs to be
    // able to switch seemlessly between background copy mode (which does need dirty
    // tracking) and write blocking mode (which doesn't). Instead, mirror_top_bs takes care
    // of updating the dirty bitmap as appropriate. Note that write blocking mode only
    // becomes effective after mirror_run() sets the job of the filter. Until then, we're
    // still in background copy mode irrespective of @copy_mode.
    bitmap.disable();
    *top.dirty_bitmap.lock().unwrap() = Some(bitmap.clone());

    // Make sure that the source is not resized while the job is running
    let job = block_job_create(
        graph,
        &mirror_top,
        BlockJobParams {
            job_id: p.job_id,
            job_type: p.job_type,
            txn: None,
            perm: BLK_PERM_CONSISTENT_READ,
            shared: BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE_UNCHANGED | BLK_PERM_WRITE,
            speed: p.speed,
            flags: p.creation_flags,
            cb: None,
        },
    );

    let setup = |job: &Arc<Job>| -> Result<Arc<BlockBackend>> {
        // No resize for the target either; while the mirror is still running, a consistent
        // read isn't necessarily possible. In the case of active commit, things look a bit
        // different, though, because the target is an already populated backing file in
        // active use. We can allow anything except resize there.
        let mut target_perms = BLK_PERM_WRITE;
        let mut target_shared_perms = BLK_PERM_WRITE_UNCHANGED;
        if target_is_backing {
            let bs_size = bs
                .getlength()
                .map_err(|e| Error::from_io("Could not inquire top image size", e))?;
            let target_size = target
                .getlength()
                .map_err(|e| Error::from_io("Could not inquire base image size", e))?;
            if target_size < bs_size {
                target_perms |= BLK_PERM_RESIZE;
            }
            target_shared_perms |= BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE;
        } else if chain_contains(bs, &target.skip_filters()) {
            // We may want to allow this in the future, but it would require taking some
            // extra care.
            return Err(Error::generic(
                "Cannot mirror to a filter on top of a node in the source's backing chain",
            ));
        }
        let blk = BlockBackend::with_node(None, target.clone(), target_perms, target_shared_perms)?;
        blk.set_disable_request_queuing(true);
        let blk = Arc::new(blk);

        block_job_add_bdrv(
            job,
            "source",
            bs,
            0,
            BLK_PERM_WRITE_UNCHANGED | BLK_PERM_WRITE | BLK_PERM_CONSISTENT_READ,
        )?;
        // Required permissions are already taken with blk_new()
        block_job_add_bdrv(job, "target", target, 0, BLK_PERM_ALL)?;

        // In commit_active_start() all intermediate nodes disappear, so any jobs in them
        // must be blocked
        if target_is_backing {
            // The topmost node with
            // bdrv_skip_filters(filtered_target) == bdrv_skip_filters(target)
            let filtered_target =
                find_overlay(bs, Some(target)).and_then(|o| o.cow_child()).map(|c| c.node);
            // XXX BLK_PERM_WRITE needs to be allowed so we don't block ourselves at s->base
            // (if writes are blocked for a node, they are also blocked for its backing file).
            let mut iter_shared_perms = BLK_PERM_WRITE_UNCHANGED | BLK_PERM_WRITE;
            let mut iter = bs.filter_or_cow_bs();
            while let Some(it) = iter {
                if Arc::ptr_eq(&it, target) {
                    break;
                }
                if filtered_target.as_ref().is_some_and(|f| Arc::ptr_eq(f, &it)) {
                    // From here on, all nodes are filters on the base. This allows us to
                    // share BLK_PERM_CONSISTENT_READ.
                    iter_shared_perms |= BLK_PERM_CONSISTENT_READ;
                }
                block_job_add_bdrv(job, "intermediate node", &it, 0, iter_shared_perms)?;
                iter = it.filter_or_cow_bs();
            }
            freeze_chain(&mirror_top, Some(target))?;
        }
        Ok(blk)
    };
    let (job, r) = match job {
        Ok(job) => {
            let r = setup(&job);
            (Some(job), r)
        }
        Err(e) => (None, Err(e)),
    };

    match r {
        Ok(blk) => {
            let job = job.expect("checked above");
            let core = Arc::new(MirrorCore {
                job: Arc::downgrade(&job),
                target: Mutex::new(Some(blk)),
                dirty_bitmap: bitmap,
                granularity: u64::from(granularity),
                bdev_length: AtomicU64::new(0),
                copy_mode: AtomicU8::new(copy_mode_to_u8(p.copy_mode)),
                actively_synced: AtomicBool::new(false),
                ret: AtomicI32::new(0),
                on_source_error: p.on_source_error,
                on_target_error: p.on_target_error,
                unmap: p.unmap,
                initial_zeroing_ongoing: AtomicBool::new(false),
                active_write_bytes_in_flight: AtomicU64::new(0),
                ops: Mutex::new(Ops::default()),
                ops_done: Condvar::new(),
                zero_bitmap: Mutex::new(None),
            });
            let (replaces, replaces_node) = match p.replaces {
                Some((name, node)) => (Some(name), Some(Arc::downgrade(&node))),
                None => (None, None),
            };
            job.set_driver(Box::new(MirrorJob {
                core,
                top,
                mirror_top_bs: Mutex::new(Some(mirror_top)),
                target_bs: target.clone(),
                base_overlay: find_overlay(bs, p.base.as_ref()),
                base: p.base,
                replaces,
                replaces_node,
                to_replace: Mutex::new(None),
                replace_blocker: Arc::new(Reason(
                    "block device is in use by block-job-complete".into(),
                )),
                sync_mode: p.sync_mode,
                backing_mode: p.backing_mode,
                target_is_zero: p.target_is_zero,
                should_complete: AtomicBool::new(p.auto_complete),
                buf_size: buf_size.next_multiple_of(u64::from(granularity)),
                prepared: AtomicBool::new(false),
                in_drain: AtomicBool::new(false),
                drain: Mutex::new(None),
                base_ro: p.base_ro,
                is_mirror: p.is_mirror,
            }));
            job.start();
            Ok(job)
        }
        Err(e) => {
            if let Some(job) = job {
                job.early_fail();
            }
            top.stop.store(true, Ordering::SeqCst);
            {
                let _d = bs.drained();
                if let Some(c) = mirror_top.filter_child() {
                    mirror_top.refresh_child_perms(&c).expect("dropping permissions cannot fail");
                }
                Node::replace_node(&mirror_top, bs).expect("dropping mirror_top");
            }
            mirror_top.release_dirty_bitmap(&bitmap);
            Err(e)
        }
    }
}

/// What `mirror_start()` gets.
struct MirrorStart<'a> {
    job_id: Option<&'a str>,
    bs: Arc<Node>,
    target: Arc<Node>,
    replaces: Option<(String, Arc<Node>)>,
    creation_flags: u32,
    speed: i64,
    granularity: u32,
    buf_size: i64,
    mode: MirrorSyncMode,
    backing_mode: BackingMode,
    target_is_zero: bool,
    on_source_error: BlockdevOnError,
    on_target_error: BlockdevOnError,
    unmap: bool,
    filter_node_name: Option<&'a str>,
    copy_mode: MirrorCopyMode,
}

/// `mirror_start()`.
fn mirror_start(graph: &BlockGraph, m: MirrorStart<'_>) -> Result<Arc<Job>> {
    if matches!(m.mode, MirrorSyncMode::Incremental | MirrorSyncMode::Bitmap) {
        return Err(Error::generic(format!("Sync mode '{}' not supported", m.mode.as_str())));
    }
    let base = if m.mode == MirrorSyncMode::Top { m.bs.backing_chain_next() } else { None };
    mirror_start_job(
        graph,
        StartParams {
            job_id: m.job_id,
            job_type: JobType::Mirror,
            bs: m.bs,
            creation_flags: m.creation_flags,
            target: m.target,
            replaces: m.replaces,
            speed: m.speed,
            granularity: m.granularity,
            buf_size: m.buf_size,
            sync_mode: m.mode,
            backing_mode: m.backing_mode,
            target_is_zero: m.target_is_zero,
            on_source_error: m.on_source_error,
            on_target_error: m.on_target_error,
            unmap: m.unmap,
            base,
            auto_complete: false,
            filter_node_name: m.filter_node_name,
            is_mirror: true,
            copy_mode: m.copy_mode,
            base_ro: false,
        },
    )
}

/// `commit_active_start()`.
#[allow(clippy::too_many_arguments, reason = "it is commit_active_start()")]
pub(crate) fn commit_active_start(
    graph: &BlockGraph,
    job_id: Option<&str>,
    bs: &Arc<Node>,
    base: &Arc<Node>,
    creation_flags: u32,
    speed: i64,
    on_error: BlockdevOnError,
    filter_node_name: Option<&str>,
) -> Result<Arc<Job>> {
    let base_read_only = base.read_only();
    if base_read_only {
        base.reopen_set_read_only(false)?;
    }
    let r = mirror_start_job(
        graph,
        StartParams {
            job_id,
            job_type: JobType::Commit,
            bs: bs.clone(),
            creation_flags,
            target: base.clone(),
            replaces: None,
            speed,
            granularity: 0,
            buf_size: 0,
            sync_mode: MirrorSyncMode::Top,
            backing_mode: BackingMode::LeaveBackingChain,
            target_is_zero: false,
            on_source_error: on_error,
            on_target_error: on_error,
            unmap: true,
            base: Some(base.clone()),
            auto_complete: false,
            filter_node_name,
            is_mirror: false,
            copy_mode: MirrorCopyMode::Background,
            base_ro: base_read_only,
        },
    );
    if r.is_err() && base_read_only {
        // ignore error for bdrv_reopen, because we want to propagate the original error
        let _ = base.reopen_set_read_only(true);
    }
    r
}

/// The arguments of `blockdev_mirror_common()`.
struct CommonArgs<'a> {
    job_id: Option<&'a str>,
    bs: Arc<Node>,
    target: Arc<Node>,
    replaces: Option<&'a str>,
    sync: MirrorSyncMode,
    backing_mode: BackingMode,
    target_is_zero: bool,
    speed: Option<i64>,
    granularity: Option<u32>,
    buf_size: Option<i64>,
    on_source_error: Option<BlockdevOnError>,
    on_target_error: Option<BlockdevOnError>,
    unmap: Option<bool>,
    filter_node_name: Option<&'a str>,
    copy_mode: Option<MirrorCopyMode>,
    auto_finalize: Option<bool>,
    auto_dismiss: Option<bool>,
}

/// `blockdev_mirror_common()`.
fn blockdev_mirror_common(graph: &BlockGraph, a: CommonArgs<'_>) -> Result<()> {
    let granularity = a.granularity.unwrap_or(0);
    if granularity != 0 && !(512..=1048576 * 64).contains(&granularity) {
        return Err(Error::generic(
            "Parameter 'granularity' expects a value in range [512B, 64MB]",
        ));
    }
    if granularity & granularity.wrapping_sub(1) != 0 {
        return Err(Error::generic("Parameter 'granularity' expects a power of 2"));
    }
    op_is_blocked(&a.bs, BlockOpType::MirrorSource, &device_or_node_name(graph, &a.bs))?;
    op_is_blocked(&a.target, BlockOpType::MirrorTarget, &device_or_node_name(graph, &a.target))?;

    let mut replaces = None;
    if let Some(r) = a.replaces {
        let bs_size =
            a.bs.getlength().map_err(|e| Error::from_io("Failed to query device's size", e))?;
        let to_replace_bs = check_to_replace_node(graph, &a.bs, r)?;
        let replace_size = to_replace_bs
            .getlength()
            .map_err(|e| Error::from_io("Failed to query the replacement node's size", e))?;
        if bs_size != replace_size {
            return Err(Error::generic(
                "cannot replace image with a mirror image of different size",
            ));
        }
        replaces = Some((r.to_string(), to_replace_bs));
    }

    // pass the node name to replace to mirror start since it's loose coupling and will allow
    // to check whether the node still exist at mirror completion
    mirror_start(
        graph,
        MirrorStart {
            job_id: a.job_id,
            bs: a.bs,
            target: a.target,
            replaces,
            creation_flags: job_flags(a.auto_finalize, a.auto_dismiss),
            speed: a.speed.unwrap_or(0),
            granularity,
            buf_size: a.buf_size.unwrap_or(0),
            mode: a.sync,
            backing_mode: a.backing_mode,
            target_is_zero: a.target_is_zero,
            on_source_error: a.on_source_error.unwrap_or(BlockdevOnError::Report),
            on_target_error: a.on_target_error.unwrap_or(BlockdevOnError::Report),
            unmap: a.unmap.unwrap_or(true),
            filter_node_name: a.filter_node_name,
            copy_mode: a.copy_mode.unwrap_or(MirrorCopyMode::Background),
        },
    )
    .map(drop)
}

/// `bdrv_img_create()` as the QMP commands that make their target use it: quietly, with
/// the size and the backing file of the new image and no other options.
pub(crate) fn img_create(
    graph: &BlockGraph,
    filename: &str,
    fmt: &str,
    base_filename: Option<&str>,
    base_fmt: Option<&str>,
    size: u64,
) -> Result<()> {
    let mut o = QDict::new();
    o.put("size", size.to_string());
    if let Some(b) = base_filename {
        if b == filename {
            return Err(Error::generic(
                "Error: Trying to create an image with the same filename as the backing file",
            ));
        }
        o.put("backing_file", b);
    }
    if let Some(f) = base_fmt {
        o.put("backing_fmt", f);
    }
    graph.create_image(fmt, filename, &mut o)
}

impl BlockGraph {
    /// `qmp_drive_mirror()`.
    pub fn drive_mirror(&self, a: &DriveMirror) -> Result<()> {
        let _bql = bql_lock();
        let bs = self.root_bs(&a.device)?;
        // Early check to avoid creating target
        op_is_blocked(&bs, BlockOpType::MirrorSource, &device_or_node_name(self, &bs))?;

        let mode = a.mode.unwrap_or(NewImageMode::AbsolutePaths);
        let format: Option<String> = match &a.format {
            Some(f) => Some(f.clone()),
            None if mode == NewImageMode::Existing => None,
            None => Some(bs.driver_name.to_string()),
        };
        let mut sync = a.sync;
        let mut target_backing_bs = bs.skip_filters().cow_child().map(|c| c.node);
        if target_backing_bs.is_none() && sync == MirrorSyncMode::Top {
            sync = MirrorSyncMode::Full;
        }
        if sync == MirrorSyncMode::None {
            target_backing_bs = Some(bs.clone());
        }
        let size = bs.getlength().map_err(|e| Error::from_io("bdrv_getlength failed", e))?;

        if a.replaces.is_some() && a.node_name.is_none() {
            return Err(Error::generic(
                "a node-name must be provided when replacing a named node of the graph",
            ));
        }
        let backing_mode = if mode == NewImageMode::AbsolutePaths {
            BackingMode::SourceBackingChain
        } else {
            BackingMode::OpenBackingChain
        };

        if (sync == MirrorSyncMode::Full || target_backing_bs.is_none())
            && mode != NewImageMode::Existing
        {
            // create new image w/o backing file
            let fmt = format.as_deref().expect("a format for a new image");
            img_create(self, &a.target, fmt, None, None, size)?;
        } else if mode == NewImageMode::AbsolutePaths {
            // Create new image with backing file.
            let explicit_backing = target_backing_bs.expect("checked above");
            explicit_backing.refresh_filename();
            let backing_name = explicit_backing.meta.lock().unwrap().filename.clone();
            let fmt = format.as_deref().expect("a format for a new image");
            img_create(
                self,
                &a.target,
                fmt,
                Some(&backing_name),
                Some(explicit_backing.driver_name),
                size,
            )?;
        }

        let mut options = QDict::new();
        if let Some(n) = &a.node_name {
            options.put("node-name", n.as_str());
        }
        if let Some(f) = &format {
            options.put("driver", f.as_str());
        }
        // Mirroring takes care of copy-on-write using the source's backing file.
        if mode != NewImageMode::Existing {
            options.put("backing", QValue::Null);
        }
        let (target_bs, _, _) =
            self.open_nodes_qdict(Some(&a.target), options, OpenCtx::default())?;
        let target_is_zero = mode != NewImageMode::Existing && target_bs.has_zero_init();

        blockdev_mirror_common(
            self,
            CommonArgs {
                job_id: a.job_id.as_deref(),
                bs,
                target: target_bs,
                replaces: a.replaces.as_deref(),
                sync,
                backing_mode,
                target_is_zero,
                speed: a.speed,
                granularity: a.granularity,
                buf_size: a.buf_size,
                on_source_error: a.on_source_error,
                on_target_error: a.on_target_error,
                unmap: a.unmap,
                filter_node_name: None,
                copy_mode: a.copy_mode,
                auto_finalize: a.auto_finalize,
                auto_dismiss: a.auto_dismiss,
            },
        )
    }

    /// `qmp_blockdev_mirror()`.
    pub fn blockdev_mirror(&self, a: &BlockdevMirrorArg) -> Result<()> {
        let _bql = bql_lock();
        let bs = self.root_bs(&a.device)?;
        let target_bs = self.lookup_bs(&a.target)?;
        blockdev_mirror_common(
            self,
            CommonArgs {
                job_id: a.job_id.as_deref(),
                bs,
                target: target_bs,
                replaces: a.replaces.as_deref(),
                sync: a.sync,
                backing_mode: BackingMode::LeaveBackingChain,
                target_is_zero: a.target_is_zero.unwrap_or(false),
                speed: a.speed,
                granularity: a.granularity,
                buf_size: a.buf_size,
                on_source_error: a.on_source_error,
                on_target_error: a.on_target_error,
                unmap: Some(true),
                filter_node_name: a.filter_node_name.as_deref(),
                copy_mode: a.copy_mode,
                auto_finalize: a.auto_finalize,
                auto_dismiss: a.auto_dismiss,
            },
        )
    }
}
