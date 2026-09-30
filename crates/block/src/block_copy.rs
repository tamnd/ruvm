// SPDX-License-Identifier: GPL-2.0-or-later

//! The block-copy API from block/block-copy.c: copies the dirty clusters of a bitmap from a
//! source node to a target node, cluster by cluster, so that a copy-before-write filter and
//! a backup job can share the work. A cluster is copied once; a caller that wants a cluster
//! somebody else is copying waits for that copy.
//!
//! Differences from QEMU:
//!
//! - There is no `copy_range` offload: `COPY_RANGE_SMALL` and `COPY_RANGE_FULL` do not exist
//!   and `use_copy_range` is ignored, every copy is a read followed by a write.
//! - The tasks of one call run one after the other on the caller's thread instead of in an
//!   `AioTaskPool` of `max_workers` coroutines, and there is no `SharedResource` memory
//!   budget; `max_workers` is only checked by the backup job.
//! - `block_copy_async()` runs its call on a thread of its own, and `block_copy()` with a
//!   timeout runs the call on a thread and waits for it for at most the timeout.
//! - The state keeps the source and target nodes it was made with, not the children of the
//!   filter, so replacing either node under a running filter is not followed.
//! - The copy bitmap belongs to no node (see [`DirtyBitmap::detached`]); QEMU creates it on
//!   the copy-before-write node, where `query-block` lists it.

use std::io;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

use ruvm_base::report::warn_report;
use ruvm_base::{Error, Result};

use crate::bitmap::DirtyBitmap;
use crate::job::chain::chain_contains;
use crate::job::core::Job;
use crate::job::ratelimit::RateLimit;
use crate::node::{
    BDRV_BLOCK_ALLOCATED, BDRV_BLOCK_DATA, BDRV_BLOCK_ZERO, BDRV_REQ_SERIALISING,
    BDRV_REQ_WRITE_COMPRESSED, Node,
};

/// `BLOCK_COPY_MAX_BUFFER`.
const BLOCK_COPY_MAX_BUFFER: u64 = 1024 * 1024;
/// `BLOCK_COPY_MAX_WORKERS`.
pub(crate) const BLOCK_COPY_MAX_WORKERS: i64 = 64;
/// `BLOCK_COPY_SLICE_TIME`, in nanoseconds.
const BLOCK_COPY_SLICE_TIME: u64 = 100_000_000;
/// `BLOCK_COPY_CLUSTER_SIZE_DEFAULT`.
const BLOCK_COPY_CLUSTER_SIZE_DEFAULT: u64 = 1 << 16;

/// `BlockCopyMethod`, without the copy-range methods.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Method {
    ReadWriteCluster,
    ReadWrite,
    WriteZeroes,
}

/// A `BlockReq` of the `reqs` list: a range some call is copying.
#[derive(Debug)]
struct Req {
    id: u64,
    offset: u64,
    bytes: u64,
}

/// The part of `BlockCopyState` under `lock`.
#[derive(Debug)]
struct Inner {
    in_flight_bytes: u64,
    method: Method,
    reqs: Vec<Req>,
    next_id: u64,
}

/// `BlockCopyState`.
pub(crate) struct BlockCopyState {
    source: Arc<Node>,
    target: Arc<Node>,
    cluster_size: u64,
    max_transfer: u64,
    len: u64,
    write_flags: AtomicU32,
    inner: Mutex<Inner>,
    /// Signalled when a request of `reqs` shrinks or ends.
    reqs_cond: Condvar,
    discard_source: AtomicBool,
    /// See the comment on `skip_unallocated` in block-copy.c: set while a `sync=top` backup
    /// clears the unallocated clusters out of the bitmap.
    skip_unallocated: AtomicBool,
    copy_bitmap: Arc<DirtyBitmap>,
    progress: Mutex<Option<Weak<Job>>>,
    rate_limit: RateLimit,
}

/// A task: a range of `reqs` a call copies.
struct Task {
    id: u64,
    offset: u64,
    bytes: u64,
    method: Method,
}

impl Task {
    /// `task_end()`.
    fn end(&self) -> u64 {
        self.offset + self.bytes
    }
}

/// `block_copy_max_transfer()`.
fn max_transfer_of(source: &Node, target: &Node) -> u64 {
    let min_non_zero = |a: u64, b: u64| {
        if a == 0 {
            b
        } else if b == 0 {
            a
        } else {
            a.min(b)
        }
    };
    let t = min_non_zero(
        u64::from(source.limits().max_transfer),
        u64::from(target.limits().max_transfer),
    );
    min_non_zero(i32::MAX as u64, t)
}

/// `block_copy_calculate_cluster_size()`.
fn calculate_cluster_size(target: &Arc<Node>, min_cluster_size: u64) -> Result<u64> {
    let min_cluster_size = min_cluster_size.max(BLOCK_COPY_CLUSTER_SIZE_DEFAULT);
    let target_does_cow = target.backing_chain_next().is_some();

    // If there is no backing file on the target, we cannot rely on COW if our backup
    // cluster size is smaller than the target cluster size. Even for targets with a backing
    // file, try to avoid COW if possible.
    match target.get_info() {
        Err(e) if e.raw_os_error() == Some(libc::ENOTSUP) && !target_does_cow => {
            // Cluster size is not defined
            warn_report(&format!(
                "The target block device doesn't provide information about the block size \
                 and it doesn't have a backing file. The (default) block size of \
                 {min_cluster_size} bytes is used. If the actual block size of the target \
                 exceeds this value, the backup may be unusable"
            ));
            Ok(min_cluster_size)
        }
        Err(e) if !target_does_cow => Err(Error::from_io(
            "Couldn't determine the cluster size of the target image, which has no backing file",
            e,
        )
        .hint("Aborting, since this may create an unusable destination image\n")),
        // Not fatal; just trudge on ahead.
        Err(_) => Ok(min_cluster_size),
        Ok(bdi) => Ok(min_cluster_size.max(bdi.cluster_size)),
    }
}

impl BlockCopyState {
    /// `block_copy_state_new()`. `bitmap` gives the clusters to copy; without it every
    /// cluster is copied.
    pub(crate) fn new(
        source: &Arc<Node>,
        target: &Arc<Node>,
        bitmap: Option<&Arc<DirtyBitmap>>,
        discard_source: bool,
        min_cluster_size: u64,
    ) -> Result<Arc<BlockCopyState>> {
        if min_cluster_size > i64::MAX as u64 {
            return Err(Error::generic(format!(
                "min-cluster-size too large: {min_cluster_size} > {}",
                i64::MAX
            )));
        } else if min_cluster_size != 0 && !min_cluster_size.is_power_of_two() {
            return Err(Error::generic("min-cluster-size needs to be a power of 2"));
        }

        let cluster_size = calculate_cluster_size(target, min_cluster_size)?;
        let size =
            source.getlength().map_err(|e| Error::from_io("could not get length of device", e))?;
        let copy_bitmap = DirtyBitmap::detached(size, cluster_size as u32);
        match bitmap {
            Some(b) => {
                copy_bitmap.merge(b, false).map_err(|e| {
                    e.prepend(format!(
                        "Failed to merge bitmap '{}' to internal copy-bitmap: ",
                        b.name().unwrap_or_else(|| "(null)".into())
                    ))
                })?;
            }
            None => copy_bitmap.set_range(0, copy_bitmap.size()),
        }

        // If source is in backing chain of target assume that target is going to be used for
        // "image fleecing", i.e. it should represent a kind of snapshot of source at
        // backup-start point in time. And target is going to be read by somebody (for
        // example, used as NBD export) during backup job.
        //
        // In this case, we need to add BDRV_REQ_SERIALISING write flag to avoid intersection
        // of backup writes and third party reads from target, otherwise reading from target
        // we may occasionally read already updated by guest data.
        let is_fleecing = chain_contains(target, source);

        let max_transfer = max_transfer_of(source, target) / cluster_size * cluster_size;
        let s = BlockCopyState {
            source: source.clone(),
            target: target.clone(),
            cluster_size,
            max_transfer,
            len: copy_bitmap.size(),
            write_flags: AtomicU32::new(if is_fleecing { BDRV_REQ_SERIALISING } else { 0 }),
            inner: Mutex::new(Inner {
                in_flight_bytes: 0,
                method: Method::ReadWrite,
                reqs: Vec::new(),
                next_id: 1,
            }),
            reqs_cond: Condvar::new(),
            discard_source: AtomicBool::new(discard_source),
            skip_unallocated: AtomicBool::new(false),
            copy_bitmap,
            progress: Mutex::new(None),
            rate_limit: RateLimit::default(),
        };
        s.set_copy_opts(false, false);
        Ok(Arc::new(s))
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap()
    }

    /// `block_copy_set_copy_opts()`. There is no copy offload, so `use_copy_range` changes
    /// nothing.
    pub(crate) fn set_copy_opts(&self, use_copy_range: bool, compress: bool) {
        let _ = use_copy_range;
        // Keep BDRV_REQ_SERIALISING set (or not set) in block_copy_state_new()
        let serialising = self.write_flags.load(Ordering::SeqCst) & BDRV_REQ_SERIALISING;
        self.write_flags.store(
            serialising | if compress { BDRV_REQ_WRITE_COMPRESSED } else { 0 },
            Ordering::SeqCst,
        );
        // Requests smaller than the cluster size are not worth it, and compression supports
        // only cluster-size writes.
        self.lock().method = if self.max_transfer < self.cluster_size || compress {
            Method::ReadWriteCluster
        } else {
            Method::ReadWrite
        };
    }

    /// `block_copy_set_progress_meter()`: the job whose progress the copying counts.
    pub(crate) fn set_progress_meter(&self, job: &Arc<Job>) {
        *self.progress.lock().unwrap() = Some(Arc::downgrade(job));
    }

    fn progress_job(&self) -> Option<Arc<Job>> {
        self.progress.lock().unwrap().as_ref().and_then(Weak::upgrade)
    }

    /// `progress_set_remaining()` with what is left in the bitmap and in flight.
    fn update_remaining(&self, inner: &Inner) {
        if let Some(j) = self.progress_job() {
            j.progress_set_remaining(self.copy_bitmap.count() + inner.in_flight_bytes);
        }
    }

    /// `block_copy_set_speed()`.
    pub(crate) fn set_speed(&self, speed: u64) {
        self.rate_limit.set_speed(speed, BLOCK_COPY_SLICE_TIME);
    }

    /// `block_copy_set_skip_unallocated()`.
    pub(crate) fn set_skip_unallocated(&self, skip: bool) {
        self.skip_unallocated.store(skip, Ordering::SeqCst);
    }

    /// Sets `discard_source` after the state was made.
    pub(crate) fn set_discard_source(&self, discard: bool) {
        self.discard_source.store(discard, Ordering::SeqCst);
    }

    /// `block_copy_dirty_bitmap()`.
    pub(crate) fn dirty_bitmap(&self) -> &Arc<DirtyBitmap> {
        &self.copy_bitmap
    }

    /// `block_copy_cluster_size()`.
    pub(crate) fn cluster_size(&self) -> u64 {
        self.cluster_size
    }

    /// `block_copy_chunk_size()`.
    fn chunk_size(&self, inner: &Inner) -> u64 {
        match inner.method {
            Method::ReadWriteCluster => self.cluster_size,
            Method::ReadWrite => {
                self.cluster_size.max(BLOCK_COPY_MAX_BUFFER).min(self.max_transfer)
            }
            // Cannot have COPY_WRITE_ZEROES here.
            Method::WriteZeroes => unreachable!("write-zeroes is never the state's method"),
        }
    }

    /// `block_copy_task_create()`: the first dirty area in the range becomes a task.
    fn task_create(&self, call: &CallState, offset: u64, bytes: u64) -> Option<Task> {
        let mut inner = self.lock();
        let chunk = self.chunk_size(&inner);
        let max_chunk = if call.max_chunk == 0 { chunk } else { chunk.min(call.max_chunk) };
        let (offset, bytes) =
            self.copy_bitmap.next_dirty_area(offset, offset + bytes, max_chunk)?;
        assert!(offset % self.cluster_size == 0);
        let bytes = bytes.div_ceil(self.cluster_size) * self.cluster_size;

        // region is dirty, so no existent tasks possible in it
        assert!(find_conflict(&inner.reqs, offset, bytes).is_none());

        self.copy_bitmap.reset_range(offset, bytes);
        inner.in_flight_bytes += bytes;
        let id = inner.next_id;
        inner.next_id += 1;
        inner.reqs.push(Req { id, offset, bytes });
        Some(Task { id, offset, bytes, method: inner.method })
    }

    /// `block_copy_task_shrink()`: the tail of the task is left for later.
    fn task_shrink(&self, task: &mut Task, new_bytes: u64) {
        let mut inner = self.lock();
        if new_bytes == task.bytes {
            return;
        }
        assert!(new_bytes > 0 && new_bytes < task.bytes);
        inner.in_flight_bytes -= task.bytes - new_bytes;
        self.copy_bitmap.set_range(task.offset + new_bytes, task.bytes - new_bytes);
        if let Some(r) = inner.reqs.iter_mut().find(|r| r.id == task.id) {
            r.bytes = new_bytes;
        }
        task.bytes = new_bytes;
        self.reqs_cond.notify_all();
    }

    /// `block_copy_task_end()`.
    fn task_end(&self, task: &Task, failed: bool) {
        let mut inner = self.lock();
        inner.in_flight_bytes -= task.bytes;
        if failed {
            self.copy_bitmap.set_range(task.offset, task.bytes);
        }
        self.update_remaining(&inner);
        inner.reqs.retain(|r| r.id != task.id);
        self.reqs_cond.notify_all();
    }

    /// `block_copy_do_copy()`: copies a cluster aligned chunk. `method` may change for the
    /// tasks that follow. On failure returns the errno and whether the read failed.
    fn do_copy(&self, offset: u64, bytes: u64, method: &mut Method) -> Result<(), (i32, bool)> {
        let nbytes = (offset + bytes).min(self.len) - offset;
        assert!(offset % self.cluster_size == 0 && bytes % self.cluster_size == 0);
        assert!(offset < self.len);
        let write_flags = self.write_flags.load(Ordering::SeqCst);
        let code = |e: io::Error| e.raw_os_error().unwrap_or(libc::EIO);
        match *method {
            Method::WriteZeroes => self
                .target
                .pwrite_zeroes_flags(offset, nbytes, write_flags & !BDRV_REQ_WRITE_COMPRESSED)
                .map_err(|e| (code(e), false)),
            Method::ReadWriteCluster | Method::ReadWrite => {
                let mut buf = vec![0u8; nbytes as usize];
                self.source.preadv_flags(offset, &mut buf, 0).map_err(|e| (code(e), true))?;
                self.target.pwrite_flags(offset, &buf, write_flags).map_err(|e| (code(e), false))
            }
        }
    }

    /// `block_copy_task_entry()`.
    fn task_entry(&self, call: &CallState, task: Task) -> i32 {
        let mut method = task.method;
        let r = self.do_copy(task.offset, task.bytes, &mut method);
        {
            let mut inner = self.lock();
            if inner.method == task.method && task.method != Method::WriteZeroes {
                inner.method = method;
            }
            match r {
                Err((e, is_read)) => {
                    let mut res = call.result.lock().unwrap();
                    if res.0 == 0 {
                        *res = (e, is_read);
                    }
                }
                Ok(()) => {
                    if let Some(j) = self.progress_job() {
                        j.progress_update(task.bytes);
                    }
                }
            }
        }
        self.task_end(&task, r.is_err());
        if self.discard_source.load(Ordering::SeqCst) && r.is_ok() {
            let nbytes = (task.offset + task.bytes).min(self.len) - task.offset;
            let _ = self.source.pdiscard(task.offset, nbytes);
        }
        match r {
            Ok(()) => 0,
            Err((e, _)) => e,
        }
    }

    /// `block_copy_block_status()`: the status of the source at `offset`, with the length
    /// rounded to clusters.
    fn block_status(&self, offset: u64, bytes: u64) -> (u32, u64) {
        let base = if self.skip_unallocated.load(Ordering::SeqCst) {
            self.source.backing_chain_next()
        } else {
            None
        };
        match self.source.block_status_above(base.as_ref(), offset, bytes) {
            Ok(st) if st.pnum >= self.cluster_size => {
                let num = if offset + st.pnum == self.len {
                    st.pnum.div_ceil(self.cluster_size) * self.cluster_size
                } else {
                    st.pnum / self.cluster_size * self.cluster_size
                };
                (st.ret, num)
            }
            // On error or if failed to obtain large enough chunk just fallback to copy one
            // cluster.
            _ => (BDRV_BLOCK_ALLOCATED | BDRV_BLOCK_DATA, self.cluster_size),
        }
    }

    /// `block_copy_is_cluster_allocated()`: whether the cluster at `offset` is allocated,
    /// and the number of clusters that share this.
    fn is_cluster_allocated(&self, mut offset: u64) -> io::Result<(bool, u64)> {
        assert!(offset % self.cluster_size == 0);
        let mut bytes = self.len - offset;
        let mut total_count = 0;
        loop {
            let (allocated, count) = self.source.is_allocated(offset, bytes)?;
            total_count += count;
            if allocated || count == 0 {
                // allocated: partial segment(s) are considered allocated.
                // otherwise: unallocated tail is treated as an entire segment.
                return Ok((allocated, total_count.div_ceil(self.cluster_size)));
            }
            // Unallocated segment(s) with uncertain following segment(s)
            if total_count >= self.cluster_size {
                return Ok((false, total_count / self.cluster_size));
            }
            offset += count;
            bytes -= count;
        }
    }

    /// `block_copy_reset()`: the range needs no copy.
    pub(crate) fn reset(&self, offset: u64, bytes: u64) {
        let inner = self.lock();
        self.copy_bitmap.reset_range(offset, bytes);
        self.update_remaining(&inner);
    }

    /// `block_copy_reset_unallocated()`: clears the clusters at `offset` from the bitmap if
    /// they are unallocated in the source. Returns whether they are allocated and how many
    /// bytes that holds for.
    pub(crate) fn reset_unallocated(&self, offset: u64) -> io::Result<(bool, u64)> {
        let (allocated, clusters) = self.is_cluster_allocated(offset)?;
        let bytes = clusters * self.cluster_size;
        if !allocated {
            self.reset(offset, bytes);
        }
        Ok((allocated, bytes))
    }

    /// `block_copy_dirty_clusters()`: copies the dirty clusters of the range of `call`.
    /// Returns whether it found any, or the errno of the first failure.
    fn dirty_clusters(&self, call: &CallState) -> Result<bool, i32> {
        let mut offset = call.offset;
        let mut bytes = call.bytes;
        let end = offset + bytes;
        let mut found_dirty = false;
        assert!(offset % self.cluster_size == 0 && bytes % self.cluster_size == 0);

        while bytes > 0 && !call.cancelled.load(Ordering::SeqCst) {
            let Some(mut task) = self.task_create(call, offset, bytes) else {
                // No more dirty bits in the bitmap
                break;
            };
            found_dirty = true;

            let (ret, status_bytes) = self.block_status(task.offset, task.bytes);
            if status_bytes < task.bytes {
                self.task_shrink(&mut task, status_bytes);
            }
            if self.skip_unallocated.load(Ordering::SeqCst) && ret & BDRV_BLOCK_ALLOCATED == 0 {
                self.task_end(&task, false);
                offset = task.end();
                bytes = end - offset;
                continue;
            }
            if ret & BDRV_BLOCK_ZERO != 0 {
                task.method = Method::WriteZeroes;
            }

            if !call.ignore_ratelimit {
                let ns = self.rate_limit.calculate_delay(0);
                if ns > 0 {
                    self.task_end(&task, true);
                    call.sleep_ns(ns);
                    continue;
                }
            }
            self.rate_limit.calculate_delay(task.bytes);

            offset = task.end();
            bytes = end - offset;
            let r = self.task_entry(call, task);
            if r != 0 {
                return Err(r);
            }
        }
        Ok(found_dirty)
    }

    /// `block_copy_common()`: copies the range of `call`, together with the other calls.
    fn common(&self, call: &CallState) {
        loop {
            let mut ret = match self.dirty_clusters(call) {
                Ok(found) => i32::from(found),
                Err(e) => -e,
            };
            if ret == 0 && !call.cancelled.load(Ordering::SeqCst) {
                let mut inner = self.lock();
                // Check that there is no task we still need to wait to complete
                if let Some(id) = find_conflict(&inner.reqs, call.offset, call.bytes) {
                    while inner.reqs.iter().any(|r| r.id == id) {
                        inner = self.reqs_cond.wait(inner).unwrap();
                    }
                    ret = 1;
                } else {
                    // No pending tasks, but check again the bitmap in this same critical
                    // section, since a task might have failed in between.
                    ret = i32::from(self.copy_bitmap.next_dirty(call.offset, call.bytes).is_some());
                }
            }
            // We retry when something was copied, as new dirty bits may have appeared from
            // failed parallel requests, or when we waited for an intersecting request, which
            // may have failed.
            if !(ret > 0 && !call.cancelled.load(Ordering::SeqCst)) {
                break;
            }
        }
        call.finish();
    }
}

/// `reqlist_find_conflict()`.
fn find_conflict(reqs: &[Req], offset: u64, bytes: u64) -> Option<u64> {
    reqs.iter().find(|r| offset < r.offset + r.bytes && r.offset < offset + bytes).map(|r| r.id)
}

/// `BlockCopyCallState`.
pub(crate) struct CallState {
    s: Arc<BlockCopyState>,
    offset: u64,
    bytes: u64,
    max_chunk: u64,
    ignore_ratelimit: bool,
    cb: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    finished: AtomicBool,
    cancelled: AtomicBool,
    /// `ret` as a positive errno (0 for success) and `error_is_read`.
    result: Mutex<(i32, bool)>,
    /// Whether a sleep was cut short by [`CallState::kick`], and whether the call finished.
    wake: Mutex<(bool, bool)>,
    wake_cond: Condvar,
}

impl CallState {
    fn new(
        s: &Arc<BlockCopyState>,
        offset: u64,
        bytes: u64,
        max_chunk: u64,
        ignore_ratelimit: bool,
        cb: Option<Box<dyn FnOnce() + Send>>,
    ) -> Arc<CallState> {
        Arc::new(CallState {
            s: s.clone(),
            offset,
            bytes,
            max_chunk,
            ignore_ratelimit,
            cb: Mutex::new(cb),
            finished: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
            result: Mutex::new((0, false)),
            wake: Mutex::new((false, false)),
            wake_cond: Condvar::new(),
        })
    }

    /// `qemu_co_sleep_ns_wakeable()`.
    fn sleep_ns(&self, ns: i64) {
        let deadline = Instant::now() + Duration::from_nanos(ns.max(0) as u64);
        let mut w = self.wake.lock().unwrap();
        while !w.0 {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            w = self.wake_cond.wait_timeout(w, deadline - now).unwrap().0;
        }
        w.0 = false;
    }

    fn finish(&self) {
        self.finished.store(true, Ordering::SeqCst);
        {
            let mut w = self.wake.lock().unwrap();
            w.1 = true;
            self.wake_cond.notify_all();
        }
        if let Some(cb) = self.cb.lock().unwrap().take() {
            cb();
        }
    }

    /// Waits until the call finished, for at most `timeout`. Returns whether it finished.
    pub(crate) fn wait(&self, timeout: Option<Duration>) -> bool {
        let deadline = timeout.map(|t| Instant::now() + t);
        let mut w = self.wake.lock().unwrap();
        while !w.1 {
            match deadline {
                None => w = self.wake_cond.wait(w).unwrap(),
                Some(d) => {
                    let now = Instant::now();
                    if now >= d {
                        return false;
                    }
                    w = self.wake_cond.wait_timeout(w, d - now).unwrap().0;
                }
            }
        }
        true
    }

    /// `block_copy_kick()`: cuts a rate limit sleep short.
    pub(crate) fn kick(&self) {
        let mut w = self.wake.lock().unwrap();
        w.0 = true;
        self.wake_cond.notify_all();
    }

    /// `block_copy_call_finished()`.
    pub(crate) fn finished(&self) -> bool {
        self.finished.load(Ordering::SeqCst)
    }

    /// `block_copy_call_succeeded()`.
    pub(crate) fn succeeded(&self) -> bool {
        self.finished() && !self.cancelled() && self.result.lock().unwrap().0 == 0
    }

    /// `block_copy_call_failed()`.
    pub(crate) fn failed(&self) -> bool {
        self.finished() && !self.cancelled() && self.result.lock().unwrap().0 != 0
    }

    /// `block_copy_call_cancelled()`.
    pub(crate) fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// `block_copy_call_status()`: the errno (0 for success) and whether a read failed.
    pub(crate) fn status(&self) -> (i32, bool) {
        assert!(self.finished());
        *self.result.lock().unwrap()
    }

    /// `block_copy_call_cancel()`. Cancelling and finishing race; a finished call can be
    /// cancelled too.
    pub(crate) fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
        self.kick();
    }
}

/// `block_copy()`: copies the dirty clusters of the range and returns the errno of the
/// first failure. With a `timeout_ns`, gives up with `ETIMEDOUT` after that long, while the
/// copy goes on in the background until `cb` runs.
pub(crate) fn block_copy(
    s: &Arc<BlockCopyState>,
    start: u64,
    bytes: u64,
    ignore_ratelimit: bool,
    timeout_ns: u64,
    cb: Option<Box<dyn FnOnce() + Send>>,
) -> Result<(), i32> {
    let call = CallState::new(s, start, bytes, 0, ignore_ratelimit, cb);
    if timeout_ns == 0 {
        s.common(&call);
    } else {
        let c = call.clone();
        std::thread::Builder::new()
            .name("block-copy".into())
            .spawn(move || c.s.common(&c))
            .map_err(|e| e.raw_os_error().unwrap_or(libc::EAGAIN))?;
        if !call.wait(Some(Duration::from_nanos(timeout_ns))) {
            call.cancel();
            return Err(libc::ETIMEDOUT);
        }
    }
    match call.result.lock().unwrap().0 {
        0 => Ok(()),
        e => Err(e),
    }
}

/// `block_copy_async()`: copies the range on a thread of its own. `cb` runs when the call
/// finished.
pub(crate) fn block_copy_async(
    s: &Arc<BlockCopyState>,
    offset: u64,
    bytes: u64,
    max_chunk: u64,
    cb: Box<dyn FnOnce() + Send>,
) -> Arc<CallState> {
    let call = CallState::new(s, offset, bytes, max_chunk, false, Some(cb));
    let c = call.clone();
    let spawned =
        std::thread::Builder::new().name("block-copy".into()).spawn(move || c.s.common(&c));
    if let Err(e) = spawned {
        *call.result.lock().unwrap() = (e.raw_os_error().unwrap_or(libc::EAGAIN), false);
        call.finish();
    }
    call
}
