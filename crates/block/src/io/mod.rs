// SPDX-License-Identifier: GPL-2.0-or-later

//! The generic request layer from block/io.c: request checks, alignment padding with
//! read-modify-write, tracked and serialising requests, splitting at `max_transfer`,
//! copy-on-read, write zeroes with its fallback, discard, flush and truncate. Block status is
//! in [`status`].
//!
//! Every request is a method on [`Node`] and runs synchronously on the caller's thread. That
//! matches what a QEMU coroutine sees: the call returns when the request is done. The state
//! a request shares with others (the tracked request list, the in-flight counter, the flush
//! generation) is behind locks and atomics, so requests may run on several threads at once,
//! which is how a ruvm-aio executor drives them.
//!
//! Differences from QEMU:
//!
//! - Buffers are contiguous slices rather than I/O vectors. An unaligned request is padded
//!   into one bounce buffer that covers the aligned range, so the driver sees a single
//!   aligned request exactly as with QEMU's padded I/O vector, at the cost of one copy.
//! - A request conflicting with a request of the same thread does not wait for it. In QEMU
//!   that case is an assertion failure, because it would deadlock.
//! - The block status cache of protocol nodes (`bdrv_bsc_*`) is not implemented; every
//!   query goes to the driver.

pub(crate) mod status;

use std::io;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};
use std::thread::ThreadId;

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{BlkdebugEvent, BlockdevDetectZeroesOptions, PreallocMode};

use crate::drain::aio_wait_kick;
use crate::node::{
    BDRV_BLOCK_ZERO, BDRV_MAX_LENGTH, BDRV_REQ_COPY_ON_READ, BDRV_REQ_FUA, BDRV_REQ_MAY_UNMAP,
    BDRV_REQ_NO_FALLBACK, BDRV_REQ_NO_WAIT, BDRV_REQ_PREFETCH, BDRV_REQ_REGISTERED_BUF,
    BDRV_REQ_SERIALISING, BDRV_REQ_WRITE_COMPRESSED, BDRV_REQ_WRITE_UNCHANGED, BDRV_REQ_ZERO_WRITE,
    BDRV_REQUEST_MAX_BYTES, BDRV_SECTOR_SIZE, BlockDriverInfo, ENOMEDIUM, MAX_BOUNCE_BUFFER, Node,
    errno, is_enotsup, is_errno,
};
use crate::perm::{BLK_PERM_WRITE, BLK_PERM_WRITE_UNCHANGED};

/// `BdrvTrackedRequestType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReqType {
    Read,
    Write,
    Discard,
    Truncate,
}

/// `BdrvTrackedRequest`.
#[derive(Debug)]
struct Tracked {
    id: u64,
    offset: u64,
    bytes: u64,
    serialising: bool,
    overlap_offset: u64,
    overlap_bytes: u64,
    waiting_for: Option<u64>,
    thread: ThreadId,
}

impl Tracked {
    /// `tracked_request_overlaps()`.
    fn overlaps(&self, offset: u64, bytes: u64) -> bool {
        !(offset >= self.overlap_offset + self.overlap_bytes
            || self.overlap_offset >= offset + bytes)
    }
}

#[derive(Debug, Default)]
struct FlushState {
    active: bool,
    flushed_gen: u64,
}

/// The request state of a node: `tracked_requests`, `in_flight`, `serialising_in_flight`,
/// `write_gen`, the flush queue, `wr_highest_offset` and `write_threshold_offset`.
#[derive(Debug, Default)]
pub(crate) struct IoState {
    reqs: Mutex<Vec<Tracked>>,
    reqs_cond: Condvar,
    next_id: AtomicU64,
    in_flight: AtomicU32,
    serialising_in_flight: AtomicU32,
    write_gen: AtomicU64,
    flush: Mutex<FlushState>,
    flush_cond: Condvar,
    wr_highest_offset: AtomicU64,
    write_threshold: AtomicU64,
    /// `dirty_bitmaps`.
    pub(crate) dirty_bitmaps: crate::bitmap::DirtyBitmapList,
}

impl IoState {
    /// `bs->in_flight`.
    pub(crate) fn in_flight(&self) -> u32 {
        self.in_flight.load(Ordering::SeqCst)
    }

    /// `bs->wr_highest_offset`.
    pub(crate) fn wr_highest_offset(&self) -> u64 {
        self.wr_highest_offset.load(Ordering::SeqCst)
    }

    /// The number of tracked requests, for tests.
    #[cfg(test)]
    pub(crate) fn tracked_count(&self) -> usize {
        self.reqs.lock().unwrap().len()
    }
}

/// A tracked request, removed from the list when dropped (`tracked_request_end()`).
pub(crate) struct Req<'a> {
    bs: &'a Node,
    id: u64,
    pub(crate) offset: u64,
    pub(crate) bytes: u64,
    ty: ReqType,
}

impl Drop for Req<'_> {
    fn drop(&mut self) {
        let io = &self.bs.io;
        let mut reqs = io.reqs.lock().unwrap();
        if let Some(i) = reqs.iter().position(|r| r.id == self.id) {
            let r = reqs.remove(i);
            if r.serialising {
                io.serialising_in_flight.fetch_sub(1, Ordering::SeqCst);
            }
        }
        drop(reqs);
        io.reqs_cond.notify_all();
    }
}

/// `bdrv_check_qiov_request()` without the I/O vector: the generic range checks.
pub(crate) fn check_request(offset: i64, bytes: i64) -> Result<()> {
    let max = BDRV_MAX_LENGTH as i64;
    let msg = if offset < 0 {
        format!("offset is negative: {offset}")
    } else if bytes < 0 {
        format!("bytes is negative: {bytes}")
    } else if bytes > max {
        format!("bytes({bytes}) exceeds maximum({max})")
    } else if offset > max {
        format!("offset({offset}) exceeds maximum({max})")
    } else if offset > max - bytes {
        format!("sum of offset({offset}) and bytes({bytes}) exceeds maximum({max})")
    } else {
        return Ok(());
    };
    Err(Error::with_cause(msg, errno(libc::EIO)))
}

/// `bdrv_check_request32()`.
fn check_request32(offset: u64, bytes: u64) -> io::Result<()> {
    let (Ok(o), Ok(b)) = (i64::try_from(offset), i64::try_from(bytes)) else {
        return Err(errno(libc::EIO));
    };
    if check_request(o, b).is_err() || bytes > BDRV_REQUEST_MAX_BYTES {
        return Err(errno(libc::EIO));
    }
    Ok(())
}

fn check_request_io(offset: u64, bytes: u64) -> io::Result<()> {
    let (Ok(o), Ok(b)) = (i64::try_from(offset), i64::try_from(bytes)) else {
        return Err(errno(libc::EIO));
    };
    check_request(o, b).map_err(|_| errno(libc::EIO))
}

fn align_down(x: u64, a: u64) -> u64 {
    x / a * a
}

fn align_up(x: u64, a: u64) -> u64 {
    x.div_ceil(a) * a
}

fn min_non_zero(a: u64, b: u64) -> u64 {
    if a == 0 {
        b
    } else if b == 0 {
        a
    } else {
        a.min(b)
    }
}

/// `buffer_is_zero()`.
pub(crate) fn buffer_is_zero(buf: &[u8]) -> bool {
    buf.iter().all(|&b| b == 0)
}

/// `BdrvRequestPadding`, without the buffer.
#[derive(Clone, Copy, Debug)]
struct Pad {
    head: u64,
    tail: u64,
    buf_len: u64,
    merge_reads: bool,
}

/// `bdrv_init_padding()`.
fn init_padding(align: u64, offset: u64, bytes: u64) -> Option<Pad> {
    let head = offset & (align - 1);
    let mut tail = (offset + bytes) & (align - 1);
    if tail != 0 {
        tail = align - tail;
    }
    if head == 0 && tail == 0 {
        return None;
    }
    let sum = head + bytes + tail;
    let buf_len = if sum > align && head != 0 && tail != 0 { 2 * align } else { align };
    Some(Pad { head, tail, buf_len, merge_reads: sum == buf_len })
}

impl Node {
    /// `bdrv_inc_in_flight()`.
    pub(crate) fn inc_in_flight(&self) {
        self.io.in_flight.fetch_add(1, Ordering::SeqCst);
    }

    /// `bdrv_dec_in_flight()`.
    pub(crate) fn dec_in_flight(&self) {
        self.io.in_flight.fetch_sub(1, Ordering::SeqCst);
        aio_wait_kick();
    }

    /// `bdrv_enable_copy_on_read()`.
    pub(crate) fn enable_copy_on_read(&self) {
        self.copy_on_read.fetch_add(1, Ordering::SeqCst);
    }

    /// `bdrv_disable_copy_on_read()`. Nothing turns copy-on-read off again yet, the block
    /// jobs that do (`block-stream`) do not exist.
    #[cfg(test)]
    pub(crate) fn disable_copy_on_read(&self) {
        let old = self.copy_on_read.fetch_sub(1, Ordering::SeqCst);
        assert!(old >= 1);
    }

    /// `bdrv_write_threshold_get()`.
    pub(crate) fn write_threshold(&self) -> u64 {
        self.io.write_threshold.load(Ordering::SeqCst)
    }

    /// `bdrv_write_threshold_set()`.
    pub(crate) fn set_write_threshold(&self, threshold: u64) {
        self.io.write_threshold.store(threshold, Ordering::SeqCst);
    }

    /// `bdrv_write_threshold_check_write()`.
    fn write_threshold_check_write(&self, offset: u64, bytes: u64) {
        let end = offset + bytes;
        let wtr = self.write_threshold();
        if wtr > 0 && end > wtr {
            crate::event::emit(crate::event::BlockEvent::WriteThreshold {
                node_name: self.name.clone(),
                amount_exceeded: end - wtr,
                write_threshold: wtr,
            });
            // Disable it so the monitor does not get flooded.
            self.set_write_threshold(0);
        }
    }

    /// `bs->write_gen`.
    pub(crate) fn write_gen(&self) -> u64 {
        self.io.write_gen.load(Ordering::SeqCst)
    }

    /// `tracked_request_begin()`.
    pub(crate) fn track(&self, offset: u64, bytes: u64, ty: ReqType) -> Req<'_> {
        let id = self.io.next_id.fetch_add(1, Ordering::Relaxed);
        let t = Tracked {
            id,
            offset,
            bytes,
            serialising: false,
            overlap_offset: offset,
            overlap_bytes: bytes,
            waiting_for: None,
            thread: std::thread::current().id(),
        };
        self.io.reqs.lock().unwrap().insert(0, t);
        Req { bs: self, id, offset, bytes, ty }
    }

    /// `tracked_request_set_serialising()`, with the lock held.
    fn set_serialising_locked(&self, reqs: &mut [Tracked], id: u64, align: u64) {
        let Some(r) = reqs.iter_mut().find(|r| r.id == id) else {
            return;
        };
        let overlap_offset = r.offset & !(align - 1);
        let overlap_bytes = align_up(r.offset + r.bytes, align) - overlap_offset;
        if !r.serialising {
            self.io.serialising_in_flight.fetch_add(1, Ordering::SeqCst);
            r.serialising = true;
        }
        r.overlap_offset = r.overlap_offset.min(overlap_offset);
        r.overlap_bytes = r.overlap_bytes.max(overlap_bytes);
    }

    /// `bdrv_find_conflicting_request()`.
    fn find_conflicting(reqs: &[Tracked], id: u64) -> Option<u64> {
        let me = reqs.iter().find(|r| r.id == id)?;
        for r in reqs {
            if r.id == id || (!r.serialising && !me.serialising) {
                continue;
            }
            if r.overlaps(me.overlap_offset, me.overlap_bytes) {
                // QEMU asserts here; a nested request of the same thread would deadlock.
                if r.thread == me.thread {
                    continue;
                }
                // Already (indirectly) waiting for us, or will as soon as it wakes up.
                if r.waiting_for.is_none() {
                    return Some(r.id);
                }
            }
        }
        None
    }

    /// `bdrv_wait_serialising_requests_locked()`.
    fn wait_serialising_locked<'a>(
        &'a self,
        mut reqs: std::sync::MutexGuard<'a, Vec<Tracked>>,
        id: u64,
    ) -> std::sync::MutexGuard<'a, Vec<Tracked>> {
        while let Some(other) = Self::find_conflicting(&reqs, id) {
            if let Some(r) = reqs.iter_mut().find(|r| r.id == id) {
                r.waiting_for = Some(other);
            }
            reqs = self.io.reqs_cond.wait(reqs).unwrap();
            if let Some(r) = reqs.iter_mut().find(|r| r.id == id) {
                r.waiting_for = None;
            }
        }
        reqs
    }

    /// `bdrv_wait_serialising_requests()`.
    fn wait_serialising(&self, req: &Req<'_>) {
        if self.io.serialising_in_flight.load(Ordering::SeqCst) == 0 {
            return;
        }
        let reqs = self.io.reqs.lock().unwrap();
        drop(self.wait_serialising_locked(reqs, req.id));
    }

    /// `bdrv_make_request_serialising()`.
    pub(crate) fn make_request_serialising(&self, req: &Req<'_>, align: u64) {
        let mut reqs = self.io.reqs.lock().unwrap();
        self.set_serialising_locked(&mut reqs, req.id, align);
        drop(self.wait_serialising_locked(reqs, req.id));
    }

    /// `bdrv_co_get_info()`.
    pub(crate) fn get_info(&self) -> io::Result<BlockDriverInfo> {
        match self.driver.get_info(self) {
            None => match self.filter_child() {
                Some(c) => c.node.get_info(),
                None => Err(errno(libc::ENOTSUP)),
            },
            Some(r) => {
                let mut bdi = r?;
                if bdi.subcluster_size == 0 {
                    bdi.subcluster_size = bdi.cluster_size;
                }
                if bdi.cluster_size > crate::node::BDRV_MAX_ALIGNMENT {
                    return Err(errno(libc::EINVAL));
                }
                Ok(bdi)
            }
        }
    }

    /// `bdrv_round_to_subclusters()`.
    pub(crate) fn round_to_subclusters(&self, offset: u64, bytes: u64) -> (u64, u64) {
        match self.get_info() {
            Ok(bdi) if bdi.subcluster_size != 0 => {
                let c = bdi.subcluster_size;
                let o = align_down(offset, c);
                (o, align_up(offset - o + bytes, c))
            }
            _ => (offset, bytes),
        }
    }

    /// `bdrv_get_cluster_size()`.
    fn cluster_size(&self) -> u64 {
        match self.get_info() {
            Ok(bdi) if bdi.cluster_size != 0 => bdi.cluster_size,
            _ => self.request_alignment(),
        }
    }

    /// `bdrv_co_is_inserted()`: the node and all its children have their medium.
    pub(crate) fn is_inserted(&self) -> bool {
        if !self.driver.is_inserted(self) {
            return false;
        }
        self.children().iter().all(|c| c.node.is_inserted())
    }

    /// `bdrv_driver_preadv()`.
    fn driver_preadv(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.driver.pread(self, offset, buf)
    }

    /// `bdrv_driver_pwritev()`.
    fn driver_pwritev(&self, offset: u64, buf: &[u8], mut flags: u32) -> io::Result<()> {
        if self.flags().no_flush {
            flags &= !BDRV_REQ_FUA;
        }
        let supported = self.driver.supported_write_flags();
        let mut emulate_fua = false;
        if flags & BDRV_REQ_FUA != 0 && !supported & BDRV_REQ_FUA != 0 {
            flags &= !BDRV_REQ_FUA;
            emulate_fua = true;
        }
        flags &= supported;
        self.driver.pwrite_flags(self, offset, buf, flags)?;
        if emulate_fua {
            self.flush()?;
        }
        Ok(())
    }

    /// `bdrv_driver_pwritev_compressed()`.
    fn driver_pwritev_compressed(&self, offset: u64, buf: &[u8]) -> io::Result<()> {
        self.driver.pwrite_compressed(self, offset, buf).unwrap_or(Err(errno(libc::ENOTSUP)))
    }

    /// `bdrv_co_do_copy_on_readv()`. `buf` is `None` for `BDRV_REQ_PREFETCH`.
    fn do_copy_on_readv(
        &self,
        offset: u64,
        bytes: u64,
        mut buf: Option<&mut [u8]>,
        flags: u32,
    ) -> io::Result<()> {
        let max_transfer =
            min_non_zero(u64::from(self.limits().max_transfer), BDRV_REQUEST_MAX_BYTES);
        let mut progress = 0u64;
        // Nothing may be written to an inactive node, and it would not help.
        let skip_write = self.is_inactive();
        // Cover whole clusters so that allocating them needs no more backing file I/O.
        let (mut align_offset, mut align_bytes) = self.round_to_subclusters(offset, bytes);
        let mut skip_bytes = offset - align_offset;
        let mut bounce: Vec<u8> = Vec::new();

        while align_bytes > 0 {
            let (allocated, mut pnum);
            if skip_write {
                allocated = true;
                pnum = align_bytes.min(max_transfer);
            } else {
                match self.is_allocated(align_offset, align_bytes.min(max_transfer)) {
                    Ok((a, n)) => {
                        allocated = a;
                        pnum = n;
                        // The image ends in the middle of the cluster.
                        if !a && n == 0 {
                            break;
                        }
                    }
                    Err(_) => {
                        // Treat errors as unallocated; the read will fail with a better
                        // error soon enough.
                        allocated = false;
                        pnum = align_bytes.min(max_transfer);
                    }
                }
            }

            if !allocated {
                pnum = pnum.min(MAX_BOUNCE_BUFFER);
                if bounce.len() < pnum as usize {
                    bounce.resize(pnum as usize, 0);
                }
                let b = &mut bounce[..pnum as usize];
                self.driver_preadv(align_offset, b)?;
                self.debug_event(BlkdebugEvent::CorWrite);
                if self.driver.has_pwrite_zeroes() && buffer_is_zero(b) {
                    self.do_pwrite_zeroes(align_offset, pnum, BDRV_REQ_WRITE_UNCHANGED)?;
                } else {
                    // This does not change the data, so no flush even with writethrough.
                    self.driver_pwritev(align_offset, b, BDRV_REQ_WRITE_UNCHANGED)?;
                }
                if flags & BDRV_REQ_PREFETCH == 0 {
                    if let Some(out) = buf.as_deref_mut() {
                        let n = (pnum - skip_bytes).min(bytes - progress) as usize;
                        let s = skip_bytes as usize;
                        let p = progress as usize;
                        out[p..p + n].copy_from_slice(&bounce[s..s + n]);
                    }
                }
            } else if flags & BDRV_REQ_PREFETCH == 0 {
                if let Some(out) = buf.as_deref_mut() {
                    let n = (pnum - skip_bytes).min(bytes - progress) as usize;
                    let p = progress as usize;
                    self.driver_preadv(offset + progress, &mut out[p..p + n])?;
                }
            }

            align_offset += pnum;
            align_bytes -= pnum;
            progress += pnum - skip_bytes;
            skip_bytes = 0;
        }
        Ok(())
    }

    /// `bdrv_aligned_preadv()`. `buf` is `None` only with `BDRV_REQ_PREFETCH`.
    fn aligned_preadv(
        &self,
        req: &Req<'_>,
        offset: u64,
        bytes: u64,
        align: u64,
        buf: Option<&mut [u8]>,
        mut flags: u32,
    ) -> io::Result<()> {
        let max_transfer =
            align_down(min_non_zero(u64::from(self.limits().max_transfer), i32::MAX as u64), align);

        if flags & BDRV_REQ_COPY_ON_READ != 0 {
            // Touching the same cluster counts as an overlap, so that the copy-on-read read and
            // write are atomic against guest writes.
            self.make_request_serialising(req, self.cluster_size());
        } else {
            self.wait_serialising(req);
        }

        if flags & BDRV_REQ_COPY_ON_READ != 0 {
            flags &= !BDRV_REQ_COPY_ON_READ;
            let (allocated, pnum) = self.is_allocated(offset, bytes)?;
            if !allocated || pnum != bytes {
                return self.do_copy_on_readv(offset, bytes, buf, flags);
            } else if flags & BDRV_REQ_PREFETCH != 0 {
                return Ok(());
            }
        }

        let total_bytes = self.getlength()?;
        let Some(buf) = buf else {
            return Ok(());
        };
        let mut max_bytes = align_up(total_bytes.saturating_sub(offset), align);
        if bytes <= max_bytes && bytes <= max_transfer {
            return self.driver_preadv(offset, &mut buf[..bytes as usize]);
        }

        let mut done = 0u64;
        while done < bytes {
            let remaining = bytes - done;
            let d = done as usize;
            let num;
            if max_bytes > 0 {
                num = remaining.min(max_bytes.min(max_transfer));
                self.driver_preadv(offset + done, &mut buf[d..d + num as usize])?;
                max_bytes -= num;
            } else {
                num = remaining;
                buf[d..d + num as usize].fill(0);
            }
            done += num;
        }
        Ok(())
    }

    /// `bdrv_co_preadv_part()` with request flags.
    pub(crate) fn preadv_flags(&self, offset: u64, buf: &mut [u8], flags: u32) -> io::Result<()> {
        self.preadv_opt(offset, buf.len() as u64, Some(buf), flags)
    }

    /// `bdrv_co_preadv_part()`. `buf` is `None` for `BDRV_REQ_PREFETCH`, which then covers
    /// `bytes`.
    pub(crate) fn preadv_opt(
        &self,
        offset: u64,
        bytes: u64,
        buf: Option<&mut [u8]>,
        mut flags: u32,
    ) -> io::Result<()> {
        if !self.is_inserted() {
            return Err(errno(ENOMEDIUM));
        }
        check_request32(offset, bytes)?;
        let align = self.request_alignment();
        if bytes == 0 && offset % align != 0 {
            return Ok(());
        }

        self.inc_in_flight();
        // Don't do copy-on-read if we read data before write operation.
        if self.copy_on_read.load(Ordering::SeqCst) > 0 {
            flags |= BDRV_REQ_COPY_ON_READ;
        }
        let r = match init_padding(align, offset, bytes) {
            None => {
                let req = self.track(offset, bytes, ReqType::Read);
                self.aligned_preadv(&req, offset, bytes, align, buf, flags)
            }
            Some(pad) => {
                flags &= !BDRV_REQ_REGISTERED_BUF;
                let start = offset - pad.head;
                let len = pad.head + bytes + pad.tail;
                let req = self.track(start, len, ReqType::Read);
                match buf {
                    Some(out) => {
                        let mut bounce = vec![0u8; len as usize];
                        let r =
                            self.aligned_preadv(&req, start, len, align, Some(&mut bounce), flags);
                        if r.is_ok() {
                            let h = pad.head as usize;
                            out.copy_from_slice(&bounce[h..h + out.len()]);
                        }
                        r
                    }
                    None => self.aligned_preadv(&req, start, len, align, None, flags),
                }
            }
        };
        self.dec_in_flight();
        r
    }

    /// `bdrv_pread()`: reads `buf.len()` bytes at `offset`.
    pub(crate) fn pread(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.preadv_flags(offset, buf, 0)
    }

    /// `bdrv_co_do_pwrite_zeroes()`.
    fn do_pwrite_zeroes(&self, mut offset: u64, mut bytes: u64, mut flags: u32) -> io::Result<()> {
        let bl = self.limits();
        let supported_zero =
            if self.driver.has_pwrite_zeroes() { self.driver.supported_zero_flags() } else { 0 };
        let supported_write = self.driver.supported_write_flags();
        let mut max_write_zeroes = min_non_zero(bl.max_pwrite_zeroes, i64::MAX as u64);
        let alignment = u64::from(bl.pwrite_zeroes_alignment).max(self.request_alignment());
        let max_transfer = min_non_zero(u64::from(bl.max_transfer), MAX_BOUNCE_BUFFER);
        let mut need_flush = false;

        if (flags & !supported_zero) & BDRV_REQ_NO_FALLBACK != 0 {
            return Err(errno(libc::ENOTSUP));
        }
        // There is no user buffer, so this flag makes no sense.
        if flags & BDRV_REQ_REGISTERED_BUF != 0 {
            return Err(errno(libc::EINVAL));
        }
        // With discard=ignore nothing is ever unmapped.
        if !self.flags().unmap {
            flags &= !BDRV_REQ_MAY_UNMAP;
        }

        let mut head = offset % alignment;
        let tail = (offset + bytes) % alignment;
        max_write_zeroes = align_down(max_write_zeroes, alignment);

        let mut zero_buf: Vec<u8> = Vec::new();
        let mut ret: io::Result<()> = Ok(());
        while bytes > 0 && ret.is_ok() {
            let mut num = bytes;
            // Align the request: drivers may expect the bulk of it to be aligned, and the
            // unaligned parts not to cross cluster boundaries.
            if head != 0 {
                // A small request up to the first aligned sector, limited to max_transfer
                // even if writing zeroes needs no fallback.
                num = bytes.min(max_transfer).min(alignment - head);
                head = (head + num) % alignment;
            } else if tail != 0 && num > alignment {
                // Shorten the request to the last aligned sector.
                num -= tail;
            }
            if num > max_write_zeroes {
                num = max_write_zeroes;
            }

            ret = Err(errno(libc::ENOTSUP));
            if self.driver.has_pwrite_zeroes() {
                ret = self.driver.pwrite_zeroes_flags(self, offset, num, flags & supported_zero);
                let unsupported = matches!(&ret, Err(e) if is_enotsup(e));
                if !unsupported && flags & BDRV_REQ_FUA != 0 && supported_zero & BDRV_REQ_FUA == 0 {
                    need_flush = true;
                }
            }

            // Some drivers cannot zero less than their alignment, even with the flags that
            // should make them fall back. Treat that like ENOTSUP.
            let fallback = match &ret {
                Err(e) => is_enotsup(e) || (is_errno(e, libc::EINVAL) && num < alignment),
                Ok(()) => false,
            };
            if fallback && flags & BDRV_REQ_NO_FALLBACK == 0 {
                // Write a bounce buffer of zeroes instead.
                let mut write_flags = flags & !BDRV_REQ_ZERO_WRITE;
                if flags & BDRV_REQ_FUA != 0 && supported_write & BDRV_REQ_FUA == 0 {
                    // One flush at the end rather than one per chunk.
                    write_flags &= !BDRV_REQ_FUA;
                    need_flush = true;
                }
                num = num.min(max_transfer);
                if zero_buf.len() < num as usize {
                    zero_buf = vec![0u8; num as usize];
                }
                ret = self.driver_pwritev(offset, &zero_buf[..num as usize], write_flags);
            }

            offset += num;
            bytes -= num;
        }

        if ret.is_ok() && need_flush {
            ret = self.flush();
        }
        ret
    }

    /// `bdrv_co_write_req_prepare()`.
    fn write_req_prepare(
        &self,
        offset: u64,
        bytes: u64,
        req: &Req<'_>,
        flags: u32,
    ) -> io::Result<()> {
        if self.read_only() {
            return Err(errno(libc::EPERM));
        }
        if flags & BDRV_REQ_SERIALISING != 0 {
            let cluster = self.cluster_size();
            let mut reqs = self.io.reqs.lock().unwrap();
            self.set_serialising_locked(&mut reqs, req.id, cluster);
            if flags & BDRV_REQ_NO_WAIT != 0 && Self::find_conflicting(&reqs, req.id).is_some() {
                return Err(errno(libc::EBUSY));
            }
            drop(self.wait_serialising_locked(reqs, req.id));
        } else {
            self.wait_serialising(req);
        }
        match req.ty {
            ReqType::Write | ReqType::Discard => self.write_threshold_check_write(offset, bytes),
            ReqType::Truncate | ReqType::Read => {}
        }
        Ok(())
    }

    /// `bdrv_co_write_req_finish()`.
    fn write_req_finish(&self, offset: u64, bytes: u64, req: &Req<'_>, ok: bool) {
        let end_sector = (offset + bytes).div_ceil(BDRV_SECTOR_SIZE) as i64;
        self.io.write_gen.fetch_add(1, Ordering::SeqCst);
        // A discard beyond the end cannot grow the node, even when error handling discards
        // past it.
        if ok
            && (req.ty == ReqType::Truncate || end_sector > self.total_sectors())
            && req.ty != ReqType::Discard
        {
            self.set_total_sectors(end_sector);
            self.parent_cb_resize();
            self.dirty_bitmap_truncate(end_sector as u64 * BDRV_SECTOR_SIZE);
        }
        if req.bytes > 0 && req.ty == ReqType::Write {
            self.io.wr_highest_offset.fetch_max(offset + bytes, Ordering::SeqCst);
        }
        if req.bytes > 0 && (req.ty == ReqType::Write || req.ty == ReqType::Discard) {
            self.set_dirty(offset, bytes);
        }
    }

    /// `bdrv_aligned_pwritev()`. `buf` is `None` for a zero write.
    fn aligned_pwritev(
        &self,
        req: &Req<'_>,
        offset: u64,
        bytes: u64,
        align: u64,
        buf: Option<&[u8]>,
        mut flags: u32,
    ) -> io::Result<()> {
        let max_transfer =
            align_down(min_non_zero(u64::from(self.limits().max_transfer), i32::MAX as u64), align);
        let prep = self.write_req_prepare(offset, bytes, req, flags);

        let nflags = self.flags();
        if prep.is_ok()
            && nflags.detect_zeroes != BlockdevDetectZeroesOptions::Off
            && flags & BDRV_REQ_ZERO_WRITE == 0
            && self.driver.has_pwrite_zeroes()
            && buf.is_some_and(buffer_is_zero)
        {
            flags |= BDRV_REQ_ZERO_WRITE;
            if nflags.detect_zeroes == BlockdevDetectZeroesOptions::Unmap {
                flags |= BDRV_REQ_MAY_UNMAP;
            }
            flags &= !BDRV_REQ_REGISTERED_BUF;
        }

        let ret = if let Err(e) = prep {
            Err(e)
        } else if flags & BDRV_REQ_ZERO_WRITE != 0 {
            self.debug_event(BlkdebugEvent::PwritevZero);
            self.do_pwrite_zeroes(offset, bytes, flags)
        } else if flags & BDRV_REQ_WRITE_COMPRESSED != 0 {
            self.driver_pwritev_compressed(offset, buf.unwrap_or(&[]))
        } else {
            let buf = buf.unwrap_or(&[]);
            self.debug_event(BlkdebugEvent::Pwritev);
            if bytes <= max_transfer {
                self.driver_pwritev(offset, &buf[..bytes as usize], flags)
            } else {
                let mut done = 0u64;
                let mut r = Ok(());
                while done < bytes {
                    let remaining = bytes - done;
                    let num = remaining.min(max_transfer);
                    let mut local_flags = flags;
                    // With FUA emulated by a flush, only the last chunk needs it.
                    if num < remaining
                        && flags & BDRV_REQ_FUA != 0
                        && self.driver.supported_write_flags() & BDRV_REQ_FUA == 0
                    {
                        local_flags &= !BDRV_REQ_FUA;
                    }
                    let d = done as usize;
                    r = self.driver_pwritev(offset + done, &buf[d..d + num as usize], local_flags);
                    if r.is_err() {
                        break;
                    }
                    done += num;
                }
                r
            }
        };
        self.debug_event(BlkdebugEvent::PwritevDone);
        self.write_req_finish(offset, bytes, req, ret.is_ok());
        ret
    }

    /// `bdrv_padding_rmw_read()`: reads the head and tail of a padded write into `buf`. The
    /// first byte of `buf` is at `head_off`, the last `align` bytes are at `tail_off`.
    fn padding_rmw_read(
        &self,
        req: &Req<'_>,
        pad: &Pad,
        head_off: u64,
        tail_off: u64,
        buf: &mut [u8],
        zero_middle: bool,
    ) -> io::Result<()> {
        let align = self.request_alignment();
        if pad.head != 0 || pad.merge_reads {
            let bytes = if pad.merge_reads { pad.buf_len } else { align };
            if pad.head != 0 {
                self.debug_event(BlkdebugEvent::PwritevRmwHead);
            }
            if pad.merge_reads && pad.tail != 0 {
                self.debug_event(BlkdebugEvent::PwritevRmwTail);
            }
            self.aligned_preadv(req, head_off, bytes, align, Some(&mut buf[..bytes as usize]), 0)?;
            if pad.head != 0 {
                self.debug_event(BlkdebugEvent::PwritevRmwAfterHead);
            }
            if pad.merge_reads && pad.tail != 0 {
                self.debug_event(BlkdebugEvent::PwritevRmwAfterTail);
            }
        }
        if pad.tail != 0 && !pad.merge_reads {
            let n = buf.len();
            self.debug_event(BlkdebugEvent::PwritevRmwTail);
            self.aligned_preadv(
                req,
                tail_off,
                align,
                align,
                Some(&mut buf[n - align as usize..]),
                0,
            )?;
            self.debug_event(BlkdebugEvent::PwritevRmwAfterTail);
        }
        if zero_middle {
            let n = buf.len();
            buf[pad.head as usize..n - pad.tail as usize].fill(0);
        }
        Ok(())
    }

    /// `bdrv_co_do_zero_pwritev()`.
    fn do_zero_pwritev(
        &self,
        mut offset: u64,
        mut bytes: u64,
        mut flags: u32,
        req: &Req<'_>,
    ) -> io::Result<()> {
        let align = self.request_alignment();
        flags &= !BDRV_REQ_REGISTERED_BUF;
        let pad = init_padding(align, offset, bytes);
        let mut pad_buf = Vec::new();
        if let Some(p) = &pad {
            self.make_request_serialising(req, align);
            pad_buf = vec![0u8; p.buf_len as usize];
            let head_off = offset & !(align - 1);
            let tail_off = align_up(offset + bytes, align) - align;
            // QEMU ignores the result of this read; a failure shows up as a failed write.
            let _ = self.padding_rmw_read(req, p, head_off, tail_off, &mut pad_buf, true);
            if p.head != 0 || p.merge_reads {
                let write_bytes = if p.merge_reads { p.buf_len } else { align };
                self.aligned_pwritev(
                    req,
                    head_off,
                    write_bytes,
                    align,
                    Some(&pad_buf[..write_bytes as usize]),
                    flags & !BDRV_REQ_ZERO_WRITE,
                )?;
                if p.merge_reads {
                    return Ok(());
                }
                offset += write_bytes - p.head;
                bytes -= write_bytes - p.head;
            }
        }
        if bytes >= align {
            // The aligned part in the middle.
            let aligned_bytes = bytes & !(align - 1);
            self.aligned_pwritev(req, offset, aligned_bytes, align, None, flags)?;
            bytes -= aligned_bytes;
            offset += aligned_bytes;
        }
        if bytes > 0 {
            let n = pad_buf.len();
            self.aligned_pwritev(
                req,
                offset,
                align,
                align,
                Some(&pad_buf[n - align as usize..]),
                flags & !BDRV_REQ_ZERO_WRITE,
            )?;
        }
        Ok(())
    }

    /// `bdrv_co_pwritev_part()`. `buf` is `None` for zero writes, which cover `bytes`.
    pub(crate) fn pwritev_opt(
        &self,
        offset: u64,
        bytes: u64,
        buf: Option<&[u8]>,
        mut flags: u32,
    ) -> io::Result<()> {
        if !self.is_inserted() {
            return Err(errno(ENOMEDIUM));
        }
        if flags & BDRV_REQ_ZERO_WRITE != 0 {
            check_request_io(offset, bytes)?;
        } else {
            check_request32(offset, bytes)?;
        }
        let align = self.request_alignment();
        // A misaligned request cannot be made efficient.
        if flags & BDRV_REQ_NO_FALLBACK != 0 && (offset | bytes) % align != 0 {
            return Err(errno(libc::ENOTSUP));
        }
        if bytes == 0 && offset % align != 0 {
            return Ok(());
        }

        let pad = if flags & BDRV_REQ_ZERO_WRITE == 0 {
            init_padding(align, offset, bytes)
        } else {
            None
        };
        if pad.is_some() {
            flags &= !BDRV_REQ_REGISTERED_BUF;
        }
        let (start, len) = match &pad {
            Some(p) => (offset - p.head, p.head + bytes + p.tail),
            None => (offset, bytes),
        };

        self.inc_in_flight();
        let req = self.track(start, len, ReqType::Write);
        let r = if flags & BDRV_REQ_ZERO_WRITE != 0 {
            self.do_zero_pwritev(offset, bytes, flags, &req)
        } else if let Some(p) = pad {
            // The request was widened for read-modify-write, so serialise it against
            // anything touching the widened range.
            self.make_request_serialising(&req, align);
            let mut bounce = vec![0u8; len as usize];
            // QEMU ignores the result of this read too.
            let _ = self.padding_rmw_read(&req, &p, start, start + len - align, &mut bounce, false);
            let h = p.head as usize;
            bounce[h..h + bytes as usize].copy_from_slice(&buf.unwrap_or(&[])[..bytes as usize]);
            self.aligned_pwritev(&req, start, len, align, Some(&bounce), flags)
        } else {
            self.aligned_pwritev(&req, offset, bytes, align, buf, flags)
        };
        drop(req);
        self.dec_in_flight();
        r
    }

    /// `bdrv_co_pwritev()` with request flags.
    pub(crate) fn pwrite_flags(&self, offset: u64, buf: &[u8], flags: u32) -> io::Result<()> {
        self.pwritev_opt(offset, buf.len() as u64, Some(buf), flags)
    }

    /// `bdrv_pwrite()`: writes `buf` at `offset`.
    pub(crate) fn pwrite(&self, offset: u64, buf: &[u8]) -> io::Result<()> {
        self.pwrite_flags(offset, buf, 0)
    }

    /// `bdrv_co_pwrite_zeroes()` with request flags.
    pub(crate) fn pwrite_zeroes_flags(
        &self,
        offset: u64,
        bytes: u64,
        flags: u32,
    ) -> io::Result<()> {
        self.pwritev_opt(offset, bytes, None, BDRV_REQ_ZERO_WRITE | flags)
    }

    /// `bdrv_pwrite_zeroes()`, with `BDRV_REQ_MAY_UNMAP` when `may_unmap`.
    pub(crate) fn pwrite_zeroes(&self, offset: u64, bytes: u64, may_unmap: bool) -> io::Result<()> {
        self.pwrite_zeroes_flags(offset, bytes, if may_unmap { BDRV_REQ_MAY_UNMAP } else { 0 })
    }

    /// `bdrv_co_pwritev(..., BDRV_REQ_WRITE_COMPRESSED)`.
    pub(crate) fn pwrite_compressed(&self, offset: u64, buf: &[u8]) -> io::Result<()> {
        self.pwrite_flags(offset, buf, BDRV_REQ_WRITE_COMPRESSED)
    }

    /// `bdrv_make_zero()`: zeroes the whole node, skipping what already reads as zero.
    pub(crate) fn make_zero(&self, flags: u32) -> io::Result<()> {
        let target_size = self.getlength()?;
        let mut offset = 0u64;
        loop {
            let bytes = target_size.saturating_sub(offset).min(BDRV_REQUEST_MAX_BYTES);
            if bytes == 0 {
                return Ok(());
            }
            let st = self.block_status(offset, bytes)?;
            if st.ret & BDRV_BLOCK_ZERO != 0 {
                offset += st.pnum;
                continue;
            }
            self.pwrite_zeroes_flags(offset, st.pnum, flags)?;
            offset += st.pnum;
        }
    }

    /// `bdrv_co_flush()`.
    pub(crate) fn flush(&self) -> io::Result<()> {
        self.inc_in_flight();
        let r = self.flush_inner();
        self.dec_in_flight();
        r
    }

    fn flush_inner(&self) -> io::Result<()> {
        if !self.is_inserted() || self.read_only() {
            return Ok(());
        }
        let current_gen;
        {
            let mut st = self.io.flush.lock().unwrap();
            current_gen = self.write_gen();
            // Wait until earlier flushes are done.
            while st.active {
                st = self.io.flush_cond.wait(st).unwrap();
            }
            st.active = true;
        }
        let r = self.flush_layers(current_gen);
        let mut st = self.io.flush.lock().unwrap();
        if r.is_ok() {
            st.flushed_gen = current_gen;
        }
        st.active = false;
        drop(st);
        self.io.flush_cond.notify_one();
        r
    }

    fn flush_layers(&self, current_gen: u64) -> io::Result<()> {
        // Write cached data back to the OS, even with cache=unsafe.
        self.primary_debug_event(BlkdebugEvent::FlushToOs);
        self.driver.flush_to_os(self)?;
        // But don't force it to the disk with cache=unsafe, or when nothing changed.
        let flushed_gen = self.io.flush.lock().unwrap().flushed_gen;
        if !self.flags().no_flush && flushed_gen != current_gen {
            self.primary_debug_event(BlkdebugEvent::FlushToDisk);
            self.driver.flush_to_disk(self)?;
        }
        // Now the children, which are cache=unsafe too in that case.
        let mut ret = Ok(());
        for c in self.children() {
            let (perm, _) = c.perm();
            if perm & (BLK_PERM_WRITE | BLK_PERM_WRITE_UNCHANGED) != 0 {
                let r = c.node.flush();
                if ret.is_ok() {
                    ret = r;
                }
            }
        }
        ret
    }

    /// `BLKDBG_CO_EVENT(bdrv_primary_child(bs), ...)`.
    fn primary_debug_event(&self, event: BlkdebugEvent) {
        if let Some(c) = self.primary_bs() {
            c.debug_event(event);
        }
    }

    /// `bdrv_co_pdiscard()`.
    pub(crate) fn pdiscard(&self, mut offset: u64, mut bytes: u64) -> io::Result<()> {
        if !self.is_inserted() {
            return Err(errno(ENOMEDIUM));
        }
        check_request_io(offset, bytes)?;
        // Do nothing if disabled.
        if !self.flags().unmap {
            return Ok(());
        }
        let bl = self.limits();
        let req_align = self.request_alignment();
        // Discard is advisory, but some devices coalesce unaligned requests, so pass
        // everything down. Most devices reject unaligned requests though, so split them off.
        let align = u64::from(bl.pdiscard_alignment).max(req_align);
        let mut head = offset % align;
        let mut tail = (offset + bytes) % align;

        self.inc_in_flight();
        let req = self.track(offset, bytes, ReqType::Discard);
        let r = (|| {
            self.write_req_prepare(offset, bytes, &req, 0)?;
            let max_pdiscard = align_down(min_non_zero(bl.max_pdiscard, i64::MAX as u64), align);
            while bytes > 0 {
                let mut num = bytes;
                if head != 0 {
                    // Small requests up to the alignment boundary.
                    num = bytes.min(align - head);
                    if num % req_align != 0 {
                        num %= req_align;
                    }
                    head = (head + num) % align;
                } else if tail != 0 {
                    if num > align {
                        // Shorten the request to the last aligned cluster.
                        num -= tail;
                    } else if tail % req_align != 0 && tail > req_align {
                        tail %= req_align;
                        num -= tail;
                    }
                }
                if num > max_pdiscard {
                    num = max_pdiscard;
                }
                if let Err(e) = self.driver.pdiscard(self, offset, num) {
                    if is_enotsup(&e) {
                        // Discard is only advice.
                    } else if is_errno(&e, libc::EINVAL)
                        && (offset % align != 0 || num % align != 0)
                    {
                        // Silently skip rejected unaligned head and tail requests.
                    } else {
                        return Err(e);
                    }
                }
                offset += num;
                bytes -= num;
            }
            Ok(())
        })();
        self.write_req_finish(req.offset, req.bytes, &req, r.is_ok());
        drop(req);
        self.dec_in_flight();
        r
    }

    /// `bdrv_co_truncate()` with `exact = false`, `PREALLOC_MODE_OFF` and no flags, which is
    /// what `bdrv_truncate(child, len, ...)` callers mostly want.
    pub(crate) fn truncate(&self, len: u64) -> Result<()> {
        self.truncate_full(len as i64, false, PreallocMode::Off, 0)
    }

    /// `bdrv_co_truncate()`.
    pub(crate) fn truncate_full(
        &self,
        offset: i64,
        exact: bool,
        prealloc: PreallocMode,
        mut flags: u32,
    ) -> Result<()> {
        if !self.is_inserted() {
            return Err(Error::generic("No medium inserted"));
        }
        if offset < 0 {
            return Err(Error::generic("Image size cannot be negative"));
        }
        check_request(offset, 0)?;
        let offset = offset as u64;
        let old_size =
            self.getlength().map_err(|e| Error::from_io("Failed to get old image size", e))?;
        if self.read_only() {
            return Err(Error::generic("Image is read-only"));
        }
        let new_bytes = offset.saturating_sub(old_size);

        self.inc_in_flight();
        let req = self.track(offset - new_bytes, new_bytes, ReqType::Truncate);
        let r = (|| {
            // When growing, keep writes away from the new area, or preallocation could
            // overwrite them.
            if new_bytes > 0 {
                self.make_request_serialising(&req, 1);
            }
            self.write_req_prepare(offset - new_bytes, new_bytes, &req, 0)
                .map_err(|e| Error::from_io("Failed to prepare request for truncation", e))?;

            // A backing file long enough to show through the new area must not show through.
            if new_bytes > 0 {
                if let Some(b) = self.cow_child() {
                    let backing_len = b
                        .node
                        .getlength()
                        .map_err(|e| Error::from_io("Could not get backing file size", e))?;
                    if backing_len > old_size {
                        flags |= BDRV_REQ_ZERO_WRITE;
                    }
                }
            }

            if self.driver.has_truncate() {
                if flags & !self.driver.supported_truncate_flags() != 0 {
                    return Err(Error::generic("Block driver does not support requested flags"));
                }
                self.driver.truncate_full(self, offset, exact, prealloc, flags)?;
            } else if let Some(f) = self.filter_child() {
                f.node.truncate_full(offset as i64, exact, prealloc, flags)?;
            } else {
                return Err(Error::generic("Image format driver does not support resize"));
            }

            let mut end = offset;
            let refresh = self.refresh_total_sectors(Some((offset / BDRV_SECTOR_SIZE) as i64));
            if refresh.is_ok() {
                end = self.total_sectors().max(0) as u64 * BDRV_SECTOR_SIZE;
            }
            self.write_req_finish(end - new_bytes.min(end), new_bytes, &req, true);
            refresh.map_err(|e| Error::from_io("Could not refresh total sector count", e))
        })();
        drop(req);
        self.dec_in_flight();
        r
    }
}

#[cfg(test)]
mod tests;
