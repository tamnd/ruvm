// SPDX-License-Identifier: GPL-2.0-or-later

//! A live node: its driver, its children, the permissions its parents hold, and the hooks the
//! generic request layer (src/io/) calls into.
//!
//! This is `BlockDriverState` and `BlockDriver` from include/block/block_int-common.h. The
//! [`Driver`] trait is the synchronous subset of `BlockDriver` the drivers need, and every
//! method that is optional in QEMU has a default here that behaves like a NULL callback.
//!
//! Requests are synchronous: they run on the caller's thread and block it until the host call
//! returns, which is what QEMU's `aio=threads` does from the point of view of the worker
//! thread. Nothing in a node assumes a particular thread, so an executor on ruvm-aio can run
//! requests on its own worker threads and complete them from there; the in-flight counters,
//! tracked requests and drain below are all thread safe for that reason.
//!
//! Differences from QEMU in `bdrv_refresh_filename()`: there are no implicit nodes to copy
//! the names of and no driver has `.bdrv_gather_child_options`, so every child's options go
//! into `full_open_options` under the edge name. [`Driver::exact_filename`] returning `None`
//! stands both for a driver without `.bdrv_refresh_filename` and for one that leaves
//! `exact_filename` empty; the two only differ for drivers with a primary child, and those
//! here are filters, which never inherit the child's name either way.

use std::io;
use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{BlkdebugEvent, BlockdevDetectZeroesOptions, ImageInfoSpecific};
use ruvm_qapi::{QDict, QValue};

use crate::drivers::DriverDef;
use crate::io::IoState;
use crate::perm::{
    BLK_PERM_ALL, DEFAULT_PERM_PASSTHROUGH, DEFAULT_PERM_UNCHANGED, PermCtx, PermTran,
    default_perms, refresh_perms,
};

/// `BDRV_SECTOR_SIZE`. Lengths are reported in whole sectors, as `bdrv_getlength()` does.
pub(crate) const BDRV_SECTOR_SIZE: u64 = 512;

/// `BDRV_REQUEST_MAX_BYTES`: the largest single request, `MIN(SIZE_MAX, INT_MAX)` aligned down
/// to whole sectors.
pub(crate) const BDRV_REQUEST_MAX_BYTES: u64 = (i32::MAX as u64) & !(BDRV_SECTOR_SIZE - 1);

/// `BDRV_MAX_ALIGNMENT`.
pub(crate) const BDRV_MAX_ALIGNMENT: u64 = 1 << 30;

/// `BDRV_MAX_LENGTH`: the largest image, `INT64_MAX` aligned down to `BDRV_MAX_ALIGNMENT`.
pub(crate) const BDRV_MAX_LENGTH: u64 = (i64::MAX as u64) & !(BDRV_MAX_ALIGNMENT - 1);

/// `MAX_BOUNCE_BUFFER` from block/io.c.
pub(crate) const MAX_BOUNCE_BUFFER: u64 = 32768 << 9;

// `BdrvRequestFlags`.
/// `BDRV_REQ_COPY_ON_READ`.
pub(crate) const BDRV_REQ_COPY_ON_READ: u32 = 0x1;
/// `BDRV_REQ_ZERO_WRITE`.
pub(crate) const BDRV_REQ_ZERO_WRITE: u32 = 0x2;
/// `BDRV_REQ_MAY_UNMAP`: a zero write may deallocate.
pub(crate) const BDRV_REQ_MAY_UNMAP: u32 = 0x4;
/// `BDRV_REQ_REGISTERED_BUF`. Kept for completeness, buffers are never registered here.
pub(crate) const BDRV_REQ_REGISTERED_BUF: u32 = 0x8;
/// `BDRV_REQ_FUA`: the write is on stable storage when it completes.
pub(crate) const BDRV_REQ_FUA: u32 = 0x10;
/// `BDRV_REQ_WRITE_COMPRESSED`.
pub(crate) const BDRV_REQ_WRITE_COMPRESSED: u32 = 0x20;
/// `BDRV_REQ_WRITE_UNCHANGED`: the write does not change what a reader sees.
pub(crate) const BDRV_REQ_WRITE_UNCHANGED: u32 = 0x40;
/// `BDRV_REQ_SERIALISING`.
pub(crate) const BDRV_REQ_SERIALISING: u32 = 0x80;
/// `BDRV_REQ_NO_FALLBACK`: fail with `ENOTSUP` rather than write a buffer of zeroes.
pub(crate) const BDRV_REQ_NO_FALLBACK: u32 = 0x100;
/// `BDRV_REQ_PREFETCH`: copy-on-read without returning the data.
pub(crate) const BDRV_REQ_PREFETCH: u32 = 0x200;
/// `BDRV_REQ_NO_WAIT`: fail with `EBUSY` rather than wait for a conflicting request.
pub(crate) const BDRV_REQ_NO_WAIT: u32 = 0x400;

// The `BDRV_BLOCK_*` block status bits.
/// `BDRV_BLOCK_DATA`: reads return data from this layer.
pub(crate) const BDRV_BLOCK_DATA: u32 = 0x01;
/// `BDRV_BLOCK_ZERO`: reads return zeroes.
pub(crate) const BDRV_BLOCK_ZERO: u32 = 0x02;
/// `BDRV_BLOCK_OFFSET_VALID`: `map` and `file` say where the data is.
pub(crate) const BDRV_BLOCK_OFFSET_VALID: u32 = 0x04;
/// `BDRV_BLOCK_RAW`: ask `file` at `map` instead.
pub(crate) const BDRV_BLOCK_RAW: u32 = 0x08;
/// `BDRV_BLOCK_ALLOCATED`: this layer decides the content.
pub(crate) const BDRV_BLOCK_ALLOCATED: u32 = 0x10;
/// `BDRV_BLOCK_EOF`: the range reaches the end of the node.
pub(crate) const BDRV_BLOCK_EOF: u32 = 0x20;
/// `BDRV_BLOCK_RECURSE`: look at `file` too to see whether the data reads as zero.
pub(crate) const BDRV_BLOCK_RECURSE: u32 = 0x40;
/// `BDRV_BLOCK_COMPRESSED`.
pub(crate) const BDRV_BLOCK_COMPRESSED: u32 = 0x80;

/// `BDRV_WANT_ZERO`.
pub(crate) const BDRV_WANT_ZERO: u32 = BDRV_BLOCK_ZERO;
/// `BDRV_WANT_ALLOCATED`.
pub(crate) const BDRV_WANT_ALLOCATED: u32 = BDRV_BLOCK_ALLOCATED;
/// `BDRV_WANT_PRECISE`.
pub(crate) const BDRV_WANT_PRECISE: u32 = BDRV_BLOCK_ZERO | BDRV_BLOCK_OFFSET_VALID;

// `BdrvChildRole`.
/// `BDRV_CHILD_DATA`: the child holds guest data.
pub(crate) const BDRV_CHILD_DATA: u32 = 1 << 0;
/// `BDRV_CHILD_METADATA`: the child holds metadata of the parent's format.
pub(crate) const BDRV_CHILD_METADATA: u32 = 1 << 1;
/// `BDRV_CHILD_FILTERED`: the parent is a filter on top of this child.
pub(crate) const BDRV_CHILD_FILTERED: u32 = 1 << 2;
/// `BDRV_CHILD_COW`: the child is the backing file.
pub(crate) const BDRV_CHILD_COW: u32 = 1 << 3;
/// `BDRV_CHILD_PRIMARY`: `bs->file` or the filtered child.
pub(crate) const BDRV_CHILD_PRIMARY: u32 = 1 << 4;
/// `BDRV_CHILD_IMAGE`.
pub(crate) const BDRV_CHILD_IMAGE: u32 = BDRV_CHILD_DATA | BDRV_CHILD_METADATA;

/// `ENOMEDIUM`, which QEMU defines as `ENODEV` on hosts that lack it.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) const ENOMEDIUM: i32 = libc::ENOMEDIUM;
/// `ENOMEDIUM`, which QEMU defines as `ENODEV` on hosts that lack it.
#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub(crate) const ENOMEDIUM: i32 = libc::ENODEV;

/// A host error number as an `io::Error`, the way QEMU returns `-errno`.
pub(crate) fn errno(e: i32) -> io::Error {
    io::Error::from_raw_os_error(e)
}

pub(crate) fn is_enotsup(e: &io::Error) -> bool {
    matches!(e.raw_os_error(), Some(n) if n == libc::ENOTSUP || n == libc::EOPNOTSUPP)
}

pub(crate) fn is_errno(e: &io::Error, n: i32) -> bool {
    e.raw_os_error() == Some(n)
}

/// The generic open flags of a node, the parts of `bs->open_flags` and friends that the
/// request layer looks at. Reopen can change them, so a node hands out copies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NodeFlags {
    /// Not `BDRV_O_RDWR`.
    pub read_only: bool,
    /// `BDRV_O_NOCACHE`, `cache.direct`.
    pub direct: bool,
    /// `BDRV_O_NO_FLUSH`, `cache.no-flush`.
    pub no_flush: bool,
    /// `BDRV_O_UNMAP`, `discard=unmap`.
    pub unmap: bool,
    /// `force-share`.
    pub force_share: bool,
    /// `detect-zeroes`.
    pub detect_zeroes: BlockdevDetectZeroesOptions,
    /// `BDRV_O_AUTO_RDONLY`, `auto-read-only`.
    pub auto_read_only: bool,
    /// `BDRV_O_INACTIVE`, `active=off`.
    pub inactive: bool,
    /// `BDRV_O_NO_IO`: opened only to look at metadata.
    pub no_io: bool,
    /// `BDRV_O_CHECK`: opened for `qemu-img check`, so a format must not repair or give up on
    /// what the check is about to look at.
    pub check: bool,
    /// `BDRV_O_ALLOW_RDWR`: reopen may make the node writable.
    pub allow_rdwr: bool,
}

/// `BlockLimits`: the alignment and size limits of a node. Zero means no limit, as in QEMU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockLimits {
    /// `request_alignment`: every request is padded to this.
    pub request_alignment: u32,
    /// `max_pdiscard`.
    pub max_pdiscard: u64,
    /// `pdiscard_alignment`.
    pub pdiscard_alignment: u32,
    /// `max_pwrite_zeroes`.
    pub max_pwrite_zeroes: u64,
    /// `pwrite_zeroes_alignment`.
    pub pwrite_zeroes_alignment: u32,
    /// `opt_transfer`.
    pub opt_transfer: u32,
    /// `max_transfer`: longer requests are split.
    pub max_transfer: u32,
    /// `max_hw_transfer`.
    pub max_hw_transfer: u64,
    /// `max_hw_iov`.
    pub max_hw_iov: u32,
    /// `min_mem_alignment`.
    pub min_mem_alignment: usize,
    /// `opt_mem_alignment`.
    pub opt_mem_alignment: usize,
    /// `max_iov`.
    pub max_iov: u32,
    /// `has_variable_length`: the length is asked from the driver every time.
    pub has_variable_length: bool,
}

/// What `.bdrv_co_block_status` reports for the start of a range.
#[derive(Clone, Debug)]
pub(crate) struct BlockStatus {
    /// The `BDRV_BLOCK_*` bits.
    pub ret: u32,
    /// How many bytes from the start of the range have that status, at least one.
    pub pnum: u64,
    /// With `BDRV_BLOCK_OFFSET_VALID`, where the range is in `file`.
    pub map: u64,
    /// With `BDRV_BLOCK_OFFSET_VALID`, the node the data is in.
    pub file: Option<Arc<Node>>,
}

/// `BlockDriverInfo`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockDriverInfo {
    /// In bytes, 0 if irrelevant.
    pub cluster_size: u64,
    /// In bytes, 0 if irrelevant.
    pub subcluster_size: u64,
    /// Offset at which the VM state can be saved, 0 if not possible.
    pub vm_state_offset: i64,
    pub is_dirty: bool,
    /// Every write must be a compressed write (streamOptimized VMDK).
    pub needs_compressed_writes: bool,
}

/// `BlockFragInfo`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlockFragInfo {
    pub allocated_clusters: u64,
    pub total_clusters: u64,
    pub fragmented_clusters: u64,
    pub compressed_clusters: u64,
}

/// `BdrvCheckResult`, what `qemu-img check` reports.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CheckResult {
    pub corruptions: i64,
    pub leaks: i64,
    pub check_errors: i64,
    pub corruptions_fixed: i64,
    pub leaks_fixed: i64,
    pub image_end_offset: i64,
    pub bfi: BlockFragInfo,
}

/// `BDRV_FIX_LEAKS` for [`Driver::check`].
pub const BDRV_FIX_LEAKS: u32 = 1;
/// `BDRV_FIX_ERRORS` for [`Driver::check`].
pub const BDRV_FIX_ERRORS: u32 = 2;

/// `QEMUSnapshotInfo`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SnapshotEntry {
    pub id_str: String,
    pub name: String,
    pub vm_state_size: u64,
    pub date_sec: u32,
    pub date_nsec: u32,
    pub vm_clock_nsec: u64,
    pub icount: Option<u64>,
}

/// The state of one node in a reopen transaction, `BDRVReopenState`.
#[derive(Debug)]
pub(crate) struct ReopenState {
    /// The flags the node will have.
    pub flags: NodeFlags,
    /// The options for the node, not yet parsed: everything `blockdev-reopen` or the caller
    /// passed, with the unchanged options filled in from the old ones. A driver takes what it
    /// knows out and leaves the rest; anything left over is an error.
    pub options: QDict,
    /// Per-driver data a driver may stash in `reopen_prepare` for `reopen_commit`.
    pub opaque: Option<Box<dyn std::any::Any + Send + Sync>>,
}

/// The driver callbacks, the synchronous subset of `BlockDriver`.
///
/// A method a driver does not implement behaves like a NULL callback in QEMU. New methods are
/// only ever added with a default, so drivers written against an older version keep building.
pub(crate) trait Driver: Send + Sync {
    /// `.bdrv_co_preadv`. The range is inside the node and aligned to its
    /// `request_alignment`, the generic layer made sure of that.
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()>;

    /// `.bdrv_co_pwritev`.
    fn pwrite(&self, bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()>;

    /// `.bdrv_co_pwrite_zeroes`. `ENOTSUP` makes the generic layer write a buffer of zeroes.
    fn pwrite_zeroes(&self, bs: &Node, offset: u64, bytes: u64, may_unmap: bool) -> io::Result<()> {
        let _ = (bs, offset, bytes, may_unmap);
        Err(errno(libc::ENOTSUP))
    }

    /// `.bdrv_co_pdiscard`. `ENOTSUP` is not an error, discard is only advice.
    fn pdiscard(&self, bs: &Node, offset: u64, bytes: u64) -> io::Result<()> {
        let _ = (bs, offset, bytes);
        Err(errno(libc::ENOTSUP))
    }

    /// `.bdrv_co_flush_to_disk`.
    fn flush_to_disk(&self, bs: &Node) -> io::Result<()> {
        let _ = bs;
        Ok(())
    }

    /// The host file behind a protocol driver, for `bs->filename`.
    fn filename(&self) -> Option<String> {
        None
    }

    /// `.bdrv_co_getlength`, in bytes and not rounded.
    fn getlength(&self, bs: &Node) -> io::Result<u64>;

    /// `.bdrv_co_truncate`.
    fn truncate(&self, bs: &Node, len: u64) -> Result<()> {
        let _ = (bs, len);
        Err(Error::generic("Image format driver does not support resize"))
    }

    /// `.bdrv_child_perm` for child number `index`, the old form without the child's role.
    /// The default returns [`CHILD_PERM_DEFAULT`], which makes the graph use
    /// `bdrv_default_perms()` for the role the child was attached with. A driver may
    /// override this or [`Driver::child_perm_for`].
    fn child_perm(&self, index: usize, perm: u64, shared: u64) -> (u64, u64) {
        let _ = (index, perm, shared);
        CHILD_PERM_DEFAULT
    }

    /// `.bdrv_check_perm` followed by `.bdrv_set_perm`: take the cumulative permissions of the
    /// parents, or fail and leave the old ones in place. When a permission transaction fails
    /// later on, this is called again with the old permissions.
    fn set_perm(&self, perm: u64, shared: u64) -> Result<()> {
        let _ = (perm, shared);
        Ok(())
    }

    // Everything below was added for the graph core. All of it has defaults.

    /// `.bdrv_co_pwritev` with the request flags. Only the flags in
    /// [`Driver::supported_write_flags`] are ever passed.
    fn pwrite_flags(&self, bs: &Node, offset: u64, buf: &[u8], flags: u32) -> io::Result<()> {
        let _ = flags;
        self.pwrite(bs, offset, buf)
    }

    /// `bs->supported_write_flags`. `BDRV_REQ_FUA` not in here is emulated with a flush.
    fn supported_write_flags(&self) -> u32 {
        0
    }

    /// `.bdrv_co_pwrite_zeroes` with the request flags, masked with
    /// [`Driver::supported_zero_flags`].
    fn pwrite_zeroes_flags(
        &self,
        bs: &Node,
        offset: u64,
        bytes: u64,
        flags: u32,
    ) -> io::Result<()> {
        self.pwrite_zeroes(bs, offset, bytes, flags & BDRV_REQ_MAY_UNMAP != 0)
    }

    /// `bs->supported_zero_flags`. The default passes `BDRV_REQ_MAY_UNMAP` on, as the plain
    /// [`Driver::pwrite_zeroes`] takes it.
    fn supported_zero_flags(&self) -> u32 {
        BDRV_REQ_MAY_UNMAP
    }

    /// Whether the driver has a `.bdrv_co_pwrite_zeroes` at all. `detect-zeroes` and
    /// copy-on-read only turn writes into zero writes for drivers that do.
    fn has_pwrite_zeroes(&self) -> bool {
        true
    }

    /// `.bdrv_co_pwritev_compressed`. `None` means the driver cannot compress.
    fn pwrite_compressed(&self, bs: &Node, offset: u64, buf: &[u8]) -> Option<io::Result<()>> {
        let _ = (bs, offset, buf);
        None
    }

    /// `block_driver_can_compress()`: whether the driver has a `.bdrv_co_pwritev_compressed`.
    /// A driver that implements [`Driver::pwrite_compressed`] should say so here, or the
    /// `compress` filter refuses to sit on top of it.
    fn can_compress(&self) -> bool {
        false
    }

    /// `.bdrv_co_block_status` for `bytes` at `offset`, both aligned to the request
    /// alignment. `None` means there is no such callback: the generic layer then reports
    /// everything as data, or passes the question on to the filtered child.
    fn block_status(
        &self,
        bs: &Node,
        want: u32,
        offset: u64,
        bytes: u64,
    ) -> Option<io::Result<BlockStatus>> {
        let _ = (bs, want, offset, bytes);
        None
    }

    /// `.bdrv_co_flush_to_os`.
    fn flush_to_os(&self, bs: &Node) -> io::Result<()> {
        let _ = bs;
        Ok(())
    }

    /// `.bdrv_refresh_limits`: adjust the limits the generic layer took from the children.
    fn refresh_limits(&self, bs: &Node, bl: &mut BlockLimits) -> Result<()> {
        let _ = (bs, bl);
        Ok(())
    }

    /// `.bdrv_child_perm` with everything QEMU passes. The default calls
    /// [`Driver::child_perm`] and, when that has no answer, `bdrv_default_perms()`.
    fn child_perm_for(&self, ctx: &PermCtx<'_>, perm: u64, shared: u64) -> (u64, u64) {
        let r = self.child_perm(ctx.index, perm, shared);
        if r == CHILD_PERM_DEFAULT { default_perms(ctx, perm, shared) } else { r }
    }

    /// `.bdrv_close`, before the children go.
    fn close(&self, bs: &Node) {
        let _ = bs;
    }

    /// `.bdrv_reopen_prepare`. `None` means the driver cannot reopen, which fails the reopen
    /// with QEMU's message.
    fn reopen_prepare(&self, bs: &Node, state: &mut ReopenState) -> Option<Result<()>> {
        let _ = (bs, state);
        None
    }

    /// `.bdrv_reopen_commit`.
    fn reopen_commit(&self, bs: &Node, state: &mut ReopenState) {
        let _ = (bs, state);
    }

    /// `.bdrv_reopen_commit_post`, after the new options are in place.
    fn reopen_commit_post(&self, bs: &Node) {
        let _ = bs;
    }

    /// `.bdrv_reopen_abort`.
    fn reopen_abort(&self, bs: &Node, state: &mut ReopenState) {
        let _ = (bs, state);
    }

    /// `.bdrv_debug_event`, what `BLKDBG_EVENT()` reaches.
    fn debug_event(&self, bs: &Node, event: BlkdebugEvent) {
        let _ = (bs, event);
    }

    /// `.bdrv_co_get_info`. `None` is `-ENOTSUP`.
    fn get_info(&self, bs: &Node) -> Option<io::Result<BlockDriverInfo>> {
        let _ = bs;
        None
    }

    /// The persistent dirty bitmap hooks (`.bdrv_supports_persistent_dirty_bitmap`,
    /// `.bdrv_co_can_store_new_dirty_bitmap` and `.bdrv_co_remove_persistent_dirty_bitmap`),
    /// for a format that stores bitmaps in its image.
    fn persistent_bitmaps(&self) -> Option<&dyn crate::bitmap::PersistentBitmaps> {
        None
    }

    /// `.bdrv_get_specific_info`.
    fn get_specific_info(&self, bs: &Node) -> Result<Option<ImageInfoSpecific>> {
        let _ = bs;
        Ok(None)
    }

    /// `.bdrv_co_get_allocated_file_size`. `None` lets the generic layer work it out.
    fn get_allocated_file_size(&self, bs: &Node) -> Option<io::Result<u64>> {
        let _ = bs;
        None
    }

    /// `.bdrv_has_zero_init`. `None` is a missing callback.
    fn has_zero_init(&self, bs: &Node) -> Option<bool> {
        let _ = bs;
        None
    }

    /// `.bdrv_co_check`. `fix` is a mask of `BDRV_FIX_LEAKS` and `BDRV_FIX_ERRORS`. `None` is
    /// `-ENOTSUP`.
    fn check(&self, bs: &Node, fix: u32) -> Option<Result<CheckResult>> {
        let _ = (bs, fix);
        None
    }

    /// `.bdrv_change_backing_file`: record a new backing file name in the image. `None` is
    /// `-ENOTSUP`.
    fn change_backing_file(
        &self,
        bs: &Node,
        file: Option<&str>,
        format: Option<&str>,
    ) -> Option<io::Result<()>> {
        let _ = (bs, file, format);
        None
    }

    /// `.bdrv_make_empty`. `None` is `-ENOTSUP`.
    fn make_empty(&self, bs: &Node) -> Option<io::Result<()>> {
        let _ = bs;
        None
    }

    /// `.bdrv_snapshot_create`. `None` is `-ENOTSUP`.
    fn snapshot_create(&self, bs: &Node, sn: &SnapshotEntry) -> Option<Result<()>> {
        let _ = (bs, sn);
        None
    }

    /// `.bdrv_snapshot_goto`. `None` is `-ENOTSUP`.
    fn snapshot_goto(&self, bs: &Node, id: &str) -> Option<Result<()>> {
        let _ = (bs, id);
        None
    }

    /// `.bdrv_snapshot_delete`. `None` is `-ENOTSUP`.
    fn snapshot_delete(
        &self,
        bs: &Node,
        id: Option<&str>,
        name: Option<&str>,
    ) -> Option<Result<()>> {
        let _ = (bs, id, name);
        None
    }

    /// `.bdrv_snapshot_list`. `None` is `-ENOTSUP`.
    fn snapshot_list(&self, bs: &Node) -> Option<Result<Vec<SnapshotEntry>>> {
        let _ = bs;
        None
    }

    /// `.bdrv_snapshot_load_tmp`: makes the read-only node read the snapshot with `id`
    /// and/or `name` until it is closed. `None` is `-ENOTSUP`.
    fn snapshot_load_tmp(
        &self,
        bs: &Node,
        id: Option<&str>,
        name: Option<&str>,
    ) -> Option<Result<()>> {
        let _ = (bs, id, name);
        None
    }

    /// `.bdrv_load_vmstate`. `None` passes the request on to the primary child.
    fn load_vmstate(&self, bs: &Node, pos: u64, buf: &mut [u8]) -> Option<io::Result<()>> {
        let _ = (bs, pos, buf);
        None
    }

    /// `.bdrv_save_vmstate`. `None` passes the request on to the primary child.
    fn save_vmstate(&self, bs: &Node, pos: u64, buf: &[u8]) -> Option<io::Result<()>> {
        let _ = (bs, pos, buf);
        None
    }

    /// `.bdrv_probe_blocksizes`: (logical, physical) block size of a host device.
    fn probe_blocksizes(&self, bs: &Node) -> Option<io::Result<(u32, u32)>> {
        let _ = bs;
        None
    }

    /// `.bdrv_co_is_inserted`: whether a removable medium is present.
    fn is_inserted(&self, bs: &Node) -> bool {
        let _ = bs;
        true
    }

    /// `.bdrv_co_eject`.
    fn eject(&self, bs: &Node, eject_flag: bool) {
        let _ = (bs, eject_flag);
    }

    /// `.bdrv_co_lock_medium`.
    fn lock_medium(&self, bs: &Node, locked: bool) {
        let _ = (bs, locked);
    }

    /// `.bdrv_drain_begin`: stop issuing requests of your own until `drain_end`.
    fn drain_begin(&self, bs: &Node) {
        let _ = bs;
    }

    /// `.bdrv_drain_end`.
    fn drain_end(&self, bs: &Node) {
        let _ = bs;
    }

    /// `.bdrv_co_invalidate_cache`, when an inactive node becomes active again.
    fn invalidate_cache(&self, bs: &Node) -> Result<()> {
        let _ = bs;
        Ok(())
    }

    /// `.bdrv_inactivate`.
    fn inactivate(&self, bs: &Node) -> Result<()> {
        let _ = bs;
        Ok(())
    }

    /// `.bdrv_refresh_filename`'s answer for `bs->exact_filename`, when the driver can say
    /// what file name would open the same node. `None` lets the generic code decide.
    fn exact_filename(&self, bs: &Node) -> Option<String> {
        let _ = bs;
        None
    }

    /// Whether the driver has a `.bdrv_co_truncate`. A filter without one passes truncation
    /// on to its filtered child.
    fn has_truncate(&self) -> bool {
        true
    }

    /// `bs->supported_truncate_flags`.
    fn supported_truncate_flags(&self) -> u32 {
        0
    }

    /// `.bdrv_co_truncate` with everything QEMU passes. The default calls
    /// [`Driver::truncate`].
    fn truncate_full(
        &self,
        bs: &Node,
        offset: u64,
        exact: bool,
        prealloc: ruvm_qapi::types::PreallocMode,
        flags: u32,
    ) -> Result<()> {
        let _ = (exact, prealloc, flags);
        self.truncate(bs, offset)
    }

    /// `.bdrv_debug_breakpoint`: suspend requests that reach `event` under `tag`. `None` means
    /// the driver has no such callback and the request goes on to the primary child.
    fn debug_breakpoint(&self, bs: &Node, event: &str, tag: &str) -> Option<io::Result<()>> {
        let _ = (bs, event, tag);
        None
    }

    /// `.bdrv_debug_remove_breakpoint`.
    fn debug_remove_breakpoint(&self, bs: &Node, tag: &str) -> Option<io::Result<()>> {
        let _ = (bs, tag);
        None
    }

    /// `.bdrv_debug_resume`.
    fn debug_resume(&self, bs: &Node, tag: &str) -> Option<io::Result<()>> {
        let _ = (bs, tag);
        None
    }

    /// `.bdrv_debug_is_suspended`.
    fn debug_is_suspended(&self, bs: &Node, tag: &str) -> Option<bool> {
        let _ = (bs, tag);
        None
    }

    /// The driver state as `Any`, for code that needs to reach a particular driver (tests,
    /// `query-named-block-nodes` details, throttle group changes).
    #[allow(dead_code, reason = "only tests reach a particular driver so far")]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }

    /// `.bdrv_co_amend`: `x-blockdev-amend` for the driver. `None` means the driver cannot
    /// amend. The caller refreshes the child permissions around the call, see
    /// [`Node::refresh_child_perms`].
    fn amend(
        &self,
        bs: &Node,
        opts: &ruvm_qapi::types::BlockdevAmendOptions,
        force: bool,
    ) -> Option<Result<()>> {
        let _ = (bs, opts, force);
        None
    }
}

/// What [`Driver::child_perm`] returns by default: no opinion.
pub(crate) const CHILD_PERM_DEFAULT: (u64, u64) = (u64::MAX, u64::MAX);

/// `bdrv_filter_default_perms()`.
pub(crate) fn filter_default_perms(perm: u64, shared: u64) -> (u64, u64) {
    (perm & DEFAULT_PERM_PASSTHROUGH, (shared & DEFAULT_PERM_PASSTHROUGH) | DEFAULT_PERM_UNCHANGED)
}

/// What a parent of a node does when the node is drained or resized, the parent side of
/// `BdrvChildClass`.
pub(crate) trait ParentOps: Send + Sync {
    /// `.drained_begin`: stop sending requests.
    fn drained_begin(&self);
    /// `.drained_end`.
    fn drained_end(&self);
    /// `.drained_poll`: whether the parent still has requests in flight.
    fn drained_poll(&self) -> bool;
    /// `.resize`.
    fn resize(&self) {}
    /// The node this parent is, for node parents.
    fn node(&self) -> Option<Arc<Node>> {
        None
    }
    /// The edge now points at `node`, `bdrv_replace_child_noperm()` for parents that are not
    /// nodes. Node parents keep their children themselves and never see this.
    fn set_node(&self, _node: Arc<Node>) {}
    /// `BdrvChildClass.stay_at_node`: `bdrv_replace_node()` leaves this edge alone.
    fn stay_at_node(&self) -> bool {
        false
    }
}

/// One parent of a node and what it holds, a `BdrvChild` seen from the child.
#[derive(Clone)]
pub(crate) struct Parent {
    pub id: u64,
    /// `bdrv_child_user_desc()`: `node 'x'`, `block device 'x'` or `an unnamed block device`.
    pub desc: String,
    /// The name of the edge: `file`, `backing`, `image` or `root`.
    pub child_name: String,
    pub perm: u64,
    pub shared: u64,
    /// The parent itself, for drain and resize. `None` for parents that do not care.
    pub ops: Option<Arc<dyn ParentOps>>,
    /// `BdrvChild.quiesced_parent`: this node drained the parent through this edge.
    pub quiesced: bool,
}

impl std::fmt::Debug for Parent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Parent")
            .field("id", &self.id)
            .field("desc", &self.desc)
            .field("child_name", &self.child_name)
            .field("perm", &self.perm)
            .field("shared", &self.shared)
            .finish_non_exhaustive()
    }
}

static NEXT_EDGE: AtomicU64 = AtomicU64::new(1);

/// A fresh identifier for a parent edge.
pub(crate) fn new_edge_id() -> u64 {
    NEXT_EDGE.fetch_add(1, Ordering::Relaxed)
}

/// An edge from a node to one of its children.
#[derive(Clone)]
pub(crate) struct Child {
    pub name: String,
    pub node: Arc<Node>,
    pub role: u32,
    pub(crate) edge: u64,
}

impl Child {
    /// The permissions this edge holds on the child, `c->perm` and `c->shared_perm`.
    pub(crate) fn perm(&self) -> (u64, u64) {
        self.node.parent_perm(self.edge).unwrap_or((0, BLK_PERM_ALL))
    }
}

/// The parts of a node that are not about I/O: names, the image's own idea of its backing
/// file, and the options it was opened with.
#[derive(Clone, Debug, Default)]
pub(crate) struct NodeMeta {
    /// `bs->filename`.
    pub filename: String,
    /// `bs->exact_filename`.
    pub exact_filename: String,
    /// `bs->backing_file`: the backing file name the image header records.
    pub backing_file: String,
    /// `bs->backing_format`.
    pub backing_format: String,
    /// `bs->auto_backing_file`.
    pub auto_backing_file: String,
    /// `bs->encrypted`.
    pub encrypted: bool,
    /// `bs->options`: every option the node was opened with, children as node names.
    pub options: QDict,
    /// `bs->explicit_options`: the options the user gave.
    pub explicit_options: QDict,
    /// `bs->full_open_options`: the options that open the same subtree again, from
    /// [`Node::refresh_filename`].
    pub full_open_options: QDict,
    /// `bs->inherits_from`: the parent that opened this node from options of its own and
    /// whose options it inherits. Empty for nodes opened on their own or by reference.
    pub inherits_from: Weak<Node>,
}

/// `PATH_MAX`, the size of `bs->filename` in QEMU.
const PATH_MAX: usize = 4096;

/// A `BlockDriverState`.
pub(crate) struct Node {
    pub name: String,
    pub driver_name: &'static str,
    pub driver: Box<dyn Driver>,
    /// The registered driver, `None` for nodes made directly with [`Node::new`].
    pub(crate) def: Option<&'static DriverDef>,
    me: Weak<Node>,
    flags: RwLock<NodeFlags>,
    children: RwLock<Vec<Child>>,
    /// Newest first, like `bs->parents`, so error messages name parents in QEMU's order.
    parents: Mutex<Vec<Parent>>,
    pub(crate) meta: Mutex<NodeMeta>,
    limits: RwLock<BlockLimits>,
    /// `bs->total_sectors`, -1 until known.
    total_sectors: AtomicI64,
    /// The cumulative permissions the driver last accepted.
    drv_perm: Mutex<(u64, u64)>,
    /// `bs->quiesce_counter`.
    pub(crate) quiesce_counter: AtomicU32,
    /// `bs->copy_on_read`, a count of users.
    pub(crate) copy_on_read: AtomicU32,
    /// Tracked requests, in-flight count, flush serialisation and write threshold.
    pub(crate) io: IoState,
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("name", &self.name)
            .field("driver", &self.driver_name)
            .field("flags", &self.flags())
            .finish_non_exhaustive()
    }
}

/// What makes a node, for [`Node::build`].
pub(crate) struct NodeSpec {
    pub name: String,
    pub driver_name: &'static str,
    pub driver: Box<dyn Driver>,
    pub def: Option<&'static DriverDef>,
    pub flags: NodeFlags,
    pub meta: NodeMeta,
    /// (edge name, node, role).
    pub children: Vec<(String, Arc<Node>, u32)>,
}

/// The parent side of an edge from a node to its child.
struct NodeParent(Weak<Node>);

impl ParentOps for NodeParent {
    fn drained_begin(&self) {
        if let Some(n) = self.0.upgrade() {
            n.do_drained_begin(None, false);
        }
    }

    fn drained_end(&self) {
        if let Some(n) = self.0.upgrade() {
            n.do_drained_end(None);
        }
    }

    fn drained_poll(&self) -> bool {
        self.0.upgrade().is_some_and(|n| n.drain_poll(None))
    }

    fn resize(&self) {
        // bdrv_child_cb_resize(): a filter's length follows its child's.
        if let Some(n) = self.0.upgrade() {
            if n.is_filter() {
                let _ = n.refresh_total_sectors(None);
                n.parent_cb_resize();
            }
        }
    }

    fn node(&self) -> Option<Arc<Node>> {
        self.0.upgrade()
    }
}

impl Node {
    /// Makes a node and attaches it to its children with the permissions it needs while it has
    /// no parents of its own. The first child is the primary one; its role is
    /// `BDRV_CHILD_FILTERED | BDRV_CHILD_PRIMARY`.
    #[cfg(test)]
    pub(crate) fn new(
        name: String,
        driver_name: &'static str,
        driver: Box<dyn Driver>,
        flags: NodeFlags,
        children: Vec<(&'static str, Arc<Node>)>,
    ) -> Result<Arc<Node>> {
        let children = children
            .into_iter()
            .enumerate()
            .map(|(i, (n, node))| {
                let role = match (n, i) {
                    ("backing", _) => BDRV_CHILD_COW,
                    (_, 0) => BDRV_CHILD_FILTERED | BDRV_CHILD_PRIMARY,
                    _ => BDRV_CHILD_DATA,
                };
                (n.to_string(), node, role)
            })
            .collect();
        Self::build(NodeSpec {
            name,
            driver_name,
            driver,
            def: None,
            flags,
            meta: NodeMeta::default(),
            children,
        })
    }

    /// Makes a node from `spec`, attaches its children and works out its limits and length,
    /// the tail of `bdrv_open_driver()`.
    pub(crate) fn build(spec: NodeSpec) -> Result<Arc<Node>> {
        let children = spec
            .children
            .into_iter()
            .map(|(name, node, role)| Child { name, node, role, edge: new_edge_id() })
            .collect();
        let node = Arc::new_cyclic(|me| Node {
            name: spec.name,
            driver_name: spec.driver_name,
            driver: spec.driver,
            def: spec.def,
            me: me.clone(),
            flags: RwLock::new(spec.flags),
            children: RwLock::new(children),
            parents: Mutex::new(Vec::new()),
            meta: Mutex::new(spec.meta),
            limits: RwLock::new(BlockLimits::default()),
            total_sectors: AtomicI64::new(-1),
            drv_perm: Mutex::new((0, BLK_PERM_ALL)),
            quiesce_counter: AtomicU32::new(0),
            copy_on_read: AtomicU32::new(0),
            io: IoState::default(),
        });
        crate::drain::register_node(&node);
        // Attach the children, bdrv_attach_child() for each. On failure dropping `node` takes
        // back whatever edges were attached.
        let kids = node.children();
        for c in &kids {
            let mut tran = PermTran::default();
            let (perm, shared) = node.child_perm_of(c, 0, BLK_PERM_ALL, None);
            c.node.add_parent(node.parent_entry(c, perm, shared));
            let r = refresh_perms(std::slice::from_ref(&c.node), None, &mut tran);
            if let Err(e) = r {
                tran.abort();
                c.node.remove_parent(c.edge);
                node.children.write().unwrap().retain(|k| k.edge != c.edge);
                return Err(e);
            }
            tran.commit();
        }
        node.refresh_limits()?;
        node.refresh_total_sectors(None)
            .map_err(|e| Error::from_io("Could not refresh total sector count", e))?;
        node.refresh_filename();
        Ok(node)
    }

    /// This node as an `Arc`.
    pub(crate) fn arc(&self) -> Arc<Node> {
        self.me.upgrade().expect("node is alive while borrowed")
    }

    /// A weak reference to this node.
    pub(crate) fn weak(&self) -> Weak<Node> {
        self.me.clone()
    }

    /// The flags as they are now.
    pub(crate) fn flags(&self) -> NodeFlags {
        *self.flags.read().unwrap()
    }

    /// Replaces the flags, for reopen.
    pub(crate) fn set_flags(&self, flags: NodeFlags) {
        *self.flags.write().unwrap() = flags;
    }

    /// `bdrv_is_read_only()`.
    pub(crate) fn read_only(&self) -> bool {
        self.flags.read().unwrap().read_only
    }

    /// The limits as they are now.
    pub(crate) fn limits(&self) -> BlockLimits {
        *self.limits.read().unwrap()
    }

    /// `bs->bl.request_alignment`.
    pub(crate) fn request_alignment(&self) -> u64 {
        u64::from(self.limits.read().unwrap().request_alignment.max(1))
    }

    /// A snapshot of the children, in the order they were attached.
    pub(crate) fn children(&self) -> Vec<Child> {
        self.children.read().unwrap().clone()
    }

    /// The child edge called `name`.
    pub(crate) fn child(&self, name: &str) -> Option<Child> {
        self.children.read().unwrap().iter().find(|c| c.name == name).cloned()
    }

    /// Whether the driver is a filter, `drv->is_filter`.
    pub(crate) fn is_filter(&self) -> bool {
        self.def.is_some_and(|d| d.is_filter)
    }

    /// Whether the driver is a format driver, `drv->is_format`.
    pub(crate) fn is_format(&self) -> bool {
        self.def.is_some_and(|d| d.is_format)
    }

    /// Whether the driver is a protocol driver, `drv->protocol_name != NULL`.
    pub(crate) fn is_protocol(&self) -> bool {
        match self.def {
            Some(d) => d.protocol_name.is_some(),
            None => self.children.read().unwrap().is_empty(),
        }
    }

    /// `bdrv_primary_child()`: the child with `BDRV_CHILD_PRIMARY`.
    pub(crate) fn primary_child(&self) -> Option<Child> {
        self.children.read().unwrap().iter().find(|c| c.role & BDRV_CHILD_PRIMARY != 0).cloned()
    }

    /// `bdrv_primary_bs()`.
    pub(crate) fn primary_bs(&self) -> Option<Arc<Node>> {
        self.primary_child().map(|c| c.node)
    }

    /// The primary child, `bs->file` for most drivers. Panics when there is none, like the
    /// NULL dereference it would be in QEMU.
    pub(crate) fn file(&self) -> Arc<Node> {
        match self.primary_child() {
            Some(c) => c.node,
            None => self.children.read().unwrap()[0].node.clone(),
        }
    }

    /// `bs->backing`.
    pub(crate) fn backing(&self) -> Option<Child> {
        self.children.read().unwrap().iter().find(|c| c.role & BDRV_CHILD_COW != 0).cloned()
    }

    /// `bdrv_cow_child()`: the backing child of a node that supports backing files.
    pub(crate) fn cow_child(&self) -> Option<Child> {
        if self.is_filter() {
            return None;
        }
        self.backing()
    }

    /// `bdrv_filter_child()`: the filtered child of a filter.
    pub(crate) fn filter_child(&self) -> Option<Child> {
        if !self.is_filter() {
            return None;
        }
        self.children.read().unwrap().iter().find(|c| c.role & BDRV_CHILD_FILTERED != 0).cloned()
    }

    /// `bdrv_filter_or_cow_child()`.
    pub(crate) fn filter_or_cow_child(&self) -> Option<Child> {
        self.cow_child().or_else(|| self.filter_child())
    }

    /// `bdrv_filter_or_cow_bs()`.
    pub(crate) fn filter_or_cow_bs(&self) -> Option<Arc<Node>> {
        self.filter_or_cow_child().map(|c| c.node)
    }

    /// `bdrv_skip_filters()`: the first node down the filtered chain that is not a filter.
    pub(crate) fn skip_filters(self: &Arc<Self>) -> Arc<Node> {
        let mut bs = self.clone();
        while let Some(c) = bs.filter_child() {
            bs = c.node;
        }
        bs
    }

    /// `bdrv_backing_chain_next()`: the next non-filter node down the backing chain.
    pub(crate) fn backing_chain_next(self: &Arc<Self>) -> Option<Arc<Node>> {
        let bs = self.skip_filters();
        bs.cow_child().map(|c| c.node.skip_filters())
    }

    /// `bs->filename`: the host file of a protocol node, or of the primary child of a format
    /// or filter node.
    pub(crate) fn filename(&self) -> Option<String> {
        let f = self.meta.lock().unwrap().filename.clone();
        if !f.is_empty() {
            return Some(f);
        }
        self.driver.filename().or_else(|| self.primary_bs().and_then(|c| c.filename()))
    }

    /// How many parents hold this node, nodes and block backends alike.
    pub(crate) fn parent_count(&self) -> usize {
        self.parents.lock().unwrap().len()
    }

    /// A snapshot of the parents, newest first.
    pub(crate) fn parents(&self) -> Vec<Parent> {
        self.parents.lock().unwrap().clone()
    }

    /// The nodes among the parents.
    pub(crate) fn parent_nodes(&self) -> Vec<Arc<Node>> {
        self.parents().iter().filter_map(|p| p.ops.as_ref().and_then(|o| o.node())).collect()
    }

    /// The permissions the parent edge `id` holds.
    pub(crate) fn parent_perm(&self, id: u64) -> Option<(u64, u64)> {
        self.parents.lock().unwrap().iter().find(|p| p.id == id).map(|p| (p.perm, p.shared))
    }

    /// `bdrv_get_cumulative_perm()`.
    pub(crate) fn cumulative_perm(&self) -> (u64, u64) {
        self.parents
            .lock()
            .unwrap()
            .iter()
            .fold((0, BLK_PERM_ALL), |(p, s), q| (p | q.perm, s & q.shared))
    }

    /// The parent entry this node puts on its child `c`.
    pub(crate) fn parent_entry(&self, c: &Child, perm: u64, shared: u64) -> Parent {
        Parent {
            id: c.edge,
            desc: format!("node '{}'", self.name),
            child_name: c.name.clone(),
            perm,
            shared,
            ops: Some(Arc::new(NodeParent(self.me.clone()))),
            quiesced: false,
        }
    }

    /// `bdrv_refresh_perms()`: works out the permissions of this node and everything below it
    /// again, for a driver whose needs changed (after an amend, say). On failure nothing
    /// changes.
    pub(crate) fn refresh_perms(&self) -> Result<()> {
        let mut tran = PermTran::default();
        match refresh_perms(&[self.arc()], None, &mut tran) {
            Ok(()) => {
                tran.commit();
                Ok(())
            }
            Err(e) => {
                tran.abort();
                Err(e)
            }
        }
    }

    /// `bdrv_child_refresh_perms()`: the same for the child behind the edge `c`.
    pub(crate) fn refresh_child_perms(&self, c: &Child) -> Result<()> {
        c.node.refresh_perms()
    }

    /// `bdrv_child_perm()`: what this node needs on child `c` when its parents hold `perm` and
    /// share `shared`.
    pub(crate) fn child_perm_of(
        &self,
        c: &Child,
        perm: u64,
        shared: u64,
        q: Option<&crate::perm::ReopenFlags>,
    ) -> (u64, u64) {
        let index = self.children.read().unwrap().iter().position(|k| k.edge == c.edge);
        let flags = crate::perm::flags_after_reopen(self, q);
        let ctx = PermCtx {
            parent: self,
            child: &c.node,
            role: c.role,
            index: index.unwrap_or(0),
            writable: !flags.read_only && !flags.inactive,
            no_io: flags.no_io,
            inactive: flags.inactive,
        };
        let (p, mut s) = self.driver.child_perm_for(&ctx, perm, shared);
        if c.node.flags().force_share {
            s = BLK_PERM_ALL;
        }
        (p, s)
    }

    /// Adds a parent edge without any checks, draining the parent if this node is drained.
    pub(crate) fn add_parent(&self, mut p: Parent) {
        // bdrv_replace_child_noperm(): a new parent of a drained node is drained too.
        if self.quiesce_counter.load(Ordering::SeqCst) > 0 {
            if let Some(ops) = &p.ops {
                ops.drained_begin();
                p.quiesced = true;
            }
        }
        self.parents.lock().unwrap().insert(0, p);
    }

    /// Removes a parent edge without any checks, undoing the drain it got from this node.
    pub(crate) fn remove_parent(&self, id: u64) -> Option<Parent> {
        let p = {
            let mut parents = self.parents.lock().unwrap();
            let i = parents.iter().position(|q| q.id == id)?;
            parents.remove(i)
        };
        if p.quiesced {
            if let Some(ops) = &p.ops {
                ops.drained_end();
            }
        }
        Some(p)
    }

    /// Sets or clears `quiesced_parent` on every parent edge but `ignore` that has callbacks
    /// and returns those callbacks, for `bdrv_parent_drained_begin()` and `_end()`. The
    /// callbacks run after the lock is dropped.
    pub(crate) fn mark_parents_quiesced(
        &self,
        ignore: Option<u64>,
        quiesced: bool,
    ) -> Vec<Arc<dyn ParentOps>> {
        let mut parents = self.parents.lock().unwrap();
        let mut out = Vec::new();
        for p in parents.iter_mut() {
            if Some(p.id) == ignore {
                continue;
            }
            if let Some(ops) = &p.ops {
                // Only end what this node began.
                if !quiesced && !p.quiesced {
                    continue;
                }
                p.quiesced = quiesced;
                out.push(ops.clone());
            }
        }
        out
    }

    /// Sets the permissions of the parent edge `id`, returning the old ones.
    pub(crate) fn set_parent_perm(&self, id: u64, perm: u64, shared: u64) -> Option<(u64, u64)> {
        let mut parents = self.parents.lock().unwrap();
        let p = parents.iter_mut().find(|q| q.id == id)?;
        let old = (p.perm, p.shared);
        p.perm = perm;
        p.shared = shared;
        Some(old)
    }

    /// Changes the description of the parent edge `id`.
    pub(crate) fn set_parent_desc(&self, id: u64, desc: String) {
        if let Some(p) = self.parents.lock().unwrap().iter_mut().find(|q| q.id == id) {
            p.desc = desc;
        }
    }

    /// Adds, replaces or (with `None`) removes the parent edge `id`, and updates the permissions
    /// down the graph. On failure everything is as it was, as with QEMU's permission
    /// transaction.
    pub(crate) fn update_parent(&self, id: u64, new: Option<Parent>) -> Result<()> {
        let me = self.arc();
        let mut tran = PermTran::default();
        match new {
            Some(p) => {
                let exists = self.parent_perm(id).is_some();
                if exists {
                    let old = self.set_parent_perm(id, p.perm, p.shared).expect("edge exists");
                    self.set_parent_desc(id, p.desc.clone());
                    tran.edge_perm(me.clone(), id, old);
                } else {
                    self.add_parent(p);
                    tran.edge_added(me.clone(), id);
                }
            }
            None => {
                if let Some(old) = self.remove_parent(id) {
                    tran.edge_removed(me.clone(), old);
                }
            }
        }
        match refresh_perms(&[me], None, &mut tran) {
            Ok(()) => {
                tran.commit();
                Ok(())
            }
            Err(e) => {
                tran.abort();
                Err(e)
            }
        }
    }

    /// `bdrv_child_try_set_perm()`: like [`Node::update_parent`] for an existing edge, but an
    /// error is hidden when the change only gives up permissions.
    pub(crate) fn try_set_parent_perm(&self, id: u64, perm: u64, shared: u64) -> Result<()> {
        let Some((old_perm, old_shared)) = self.parent_perm(id) else {
            return Ok(());
        };
        let me = self.arc();
        let mut tran = PermTran::default();
        self.set_parent_perm(id, perm, shared);
        tran.edge_perm(me.clone(), id, (old_perm, old_shared));
        match refresh_perms(&[me], None, &mut tran) {
            Ok(()) => {
                tran.commit();
                Ok(())
            }
            Err(e) => {
                tran.abort();
                if perm & !old_perm != 0 || old_shared & !shared != 0 { Err(e) } else { Ok(()) }
            }
        }
    }

    /// The permissions the driver last accepted.
    pub(crate) fn drv_perm(&self) -> (u64, u64) {
        *self.drv_perm.lock().unwrap()
    }

    pub(crate) fn set_drv_perm(&self, perm: (u64, u64)) {
        *self.drv_perm.lock().unwrap() = perm;
    }

    /// Adds `child` as a new child edge called `name`, `bdrv_attach_child()`. The node must be
    /// drained if requests may be running.
    #[cfg(test)]
    pub(crate) fn attach_child(&self, name: &str, child: Arc<Node>, role: u32) -> Result<Child> {
        let _wr = crate::graph_lock::wrlock();
        self.attach_child_locked(name, child, role)
    }

    fn attach_child_locked(&self, name: &str, child: Arc<Node>, role: u32) -> Result<Child> {
        let c = Child { name: name.to_string(), node: child.clone(), role, edge: new_edge_id() };
        let (cum_perm, cum_shared) = self.cumulative_perm();
        self.children.write().unwrap().push(c.clone());
        let (perm, shared) = self.child_perm_of(&c, cum_perm, cum_shared, None);
        child.add_parent(self.parent_entry(&c, perm, shared));
        let mut tran = PermTran::default();
        let r = refresh_perms(&[self.arc()], None, &mut tran).and_then(|()| self.refresh_limits());
        match r {
            Ok(()) => {
                tran.commit();
                Ok(c)
            }
            Err(e) => {
                tran.abort();
                child.remove_parent(c.edge);
                self.children.write().unwrap().retain(|k| k.edge != c.edge);
                let _ = self.refresh_limits();
                Err(e)
            }
        }
    }

    /// Removes the child edge `edge`, `bdrv_unref_child()`. Losing permissions cannot fail.
    fn detach_child_locked(&self, edge: u64) {
        let c = {
            let mut children = self.children.write().unwrap();
            let Some(i) = children.iter().position(|k| k.edge == edge) else {
                return;
            };
            children.remove(i)
        };
        let _ = c.node.update_parent(c.edge, None);
        let _ = self.refresh_limits();
    }

    /// `bdrv_set_backing_hd()`: replaces the backing child with `backing`, or removes it.
    pub(crate) fn set_backing_hd(&self, backing: Option<Arc<Node>>) -> Result<()> {
        if !self.def.is_some_and(|d| d.supports_backing || d.filtered_child_is_backing) {
            return Err(Error::generic(format!(
                "Driver '{}' of node '{}' does not support backing files",
                self.driver_name, self.name
            )));
        }
        let _wr = crate::graph_lock::wrlock();
        let role = if self.is_filter() {
            BDRV_CHILD_FILTERED | BDRV_CHILD_PRIMARY
        } else {
            BDRV_CHILD_COW
        };
        let old = self.children.read().unwrap().iter().find(|c| c.name == "backing").cloned();
        if let Some(b) = &backing {
            if self.would_create_loop(b) {
                return Err(Error::generic(format!(
                    "Making '{}' a backing child of '{}' would create a cycle",
                    b.name, self.name
                )));
            }
        }
        if let Some(o) = &old {
            self.detach_child_locked(o.edge);
        }
        if let Some(b) = backing {
            if let Err(e) = self.attach_child_locked("backing", b, role) {
                // Put the old one back, which worked before.
                if let Some(o) = old {
                    let _ = self.attach_child_locked("backing", o.node, o.role);
                }
                return Err(e);
            }
            let mut meta = self.meta.lock().unwrap();
            let b = self.backing().expect("just attached").node;
            let bf = b.filename().unwrap_or_default();
            meta.auto_backing_file = bf.clone();
            if meta.backing_file.is_empty() {
                meta.backing_file = bf;
                meta.backing_format = b.driver_name.to_string();
            }
        }
        Ok(())
    }

    /// Whether making `child` a child of this node would close a loop.
    pub(crate) fn would_create_loop(&self, child: &Arc<Node>) -> bool {
        if std::ptr::eq(Arc::as_ptr(child), self) {
            return true;
        }
        child.children().iter().any(|c| self.would_create_loop(&c.node))
    }

    /// Replaces the node of the child edge `edge` with `to`, `bdrv_replace_child_noperm()`
    /// followed by a permission refresh of both nodes. Used by `bdrv_replace_node()`.
    pub(crate) fn replace_child_node(&self, edge: u64, to: &Arc<Node>, tran: &mut PermTran) {
        let (old, name) = {
            let mut children = self.children.write().unwrap();
            let Some(c) = children.iter_mut().find(|k| k.edge == edge) else {
                return;
            };
            let old = std::mem::replace(&mut c.node, to.clone());
            (old, c.name.clone())
        };
        let p = old.remove_parent(edge).unwrap_or_else(|| Parent {
            id: edge,
            desc: format!("node '{}'", self.name),
            child_name: name,
            perm: 0,
            shared: BLK_PERM_ALL,
            ops: Some(Arc::new(NodeParent(self.me.clone()))),
            quiesced: false,
        });
        let mut p2 = p.clone();
        p2.quiesced = false;
        to.add_parent(p2);
        tran.child_replaced(self.arc(), edge, old, p);
    }

    /// `bdrv_attach_child_noperm()`: adds a child edge without any permission update; the
    /// caller refreshes the permissions and aborts `tran` on failure.
    pub(crate) fn attach_child_noperm(
        &self,
        name: &str,
        child: Arc<Node>,
        role: u32,
        tran: &mut PermTran,
    ) -> Child {
        let c = Child { name: name.to_string(), node: child.clone(), role, edge: new_edge_id() };
        self.children.write().unwrap().push(c.clone());
        child.add_parent(self.parent_entry(&c, 0, BLK_PERM_ALL));
        tran.child_added(self.arc(), c.edge);
        c
    }

    /// `bdrv_remove_child()`: takes the child edge `edge` out without any permission update.
    pub(crate) fn remove_child_noperm(&self, edge: u64, tran: &mut PermTran) {
        let (idx, c) = {
            let mut children = self.children.write().unwrap();
            let Some(i) = children.iter().position(|k| k.edge == edge) else {
                return;
            };
            (i, children.remove(i))
        };
        let Some(p) = c.node.remove_parent(edge) else {
            self.children.write().unwrap().insert(idx, c);
            return;
        };
        tran.child_removed(self.arc(), idx, c, p);
    }

    /// Undoes [`Node::attach_child_noperm`].
    pub(crate) fn unattach_child(&self, edge: u64) {
        let c = {
            let mut children = self.children.write().unwrap();
            let Some(i) = children.iter().position(|k| k.edge == edge) else {
                return;
            };
            children.remove(i)
        };
        c.node.remove_parent(edge);
    }

    /// Undoes [`Node::remove_child_noperm`].
    pub(crate) fn unremove_child(&self, idx: usize, c: Child, mut p: Parent) {
        p.quiesced = false;
        c.node.add_parent(p);
        let mut children = self.children.write().unwrap();
        let idx = idx.min(children.len());
        children.insert(idx, c);
    }

    /// Undoes [`Node::replace_child_node`].
    pub(crate) fn unreplace_child_node(&self, edge: u64, old: Arc<Node>, p: Parent) {
        let cur = {
            let mut children = self.children.write().unwrap();
            let Some(c) = children.iter_mut().find(|k| k.edge == edge) else {
                return;
            };
            std::mem::replace(&mut c.node, old.clone())
        };
        cur.remove_parent(edge);
        let mut p = p;
        p.quiesced = false;
        old.add_parent(p);
    }

    /// `bdrv_replace_child_tran()` for the parent edge `p` of this node: node parents move
    /// their child edge, other parents (backends) are moved here.
    fn replace_parent_edge(self: &Arc<Self>, p: &Parent, to: &Arc<Node>, tran: &mut PermTran) {
        if let Some(parent) = p.ops.as_ref().and_then(|o| o.node()) {
            parent.replace_child_node(p.id, to, tran);
            return;
        }
        let Some(old) = self.remove_parent(p.id) else {
            return;
        };
        let mut moved = old.clone();
        moved.quiesced = false;
        to.add_parent(moved);
        if let Some(ops) = &old.ops {
            ops.set_node(to.clone());
        }
        tran.parent_moved(self.clone(), to.clone(), old);
    }

    /// Undoes [`Node::replace_parent_edge`] for a parent that is not a node.
    pub(crate) fn unmove_parent(self: &Arc<Self>, to: &Arc<Node>, mut p: Parent) {
        to.remove_parent(p.id);
        p.quiesced = false;
        if let Some(ops) = &p.ops {
            ops.set_node(self.clone());
        }
        self.add_parent(p);
    }

    /// `should_update_child()`: the parent edge `edge` of this node may move to `to` unless
    /// `to` reaches that edge through its own subtree, which would close a loop.
    fn should_update_child(edge: u64, to: &Arc<Node>) -> bool {
        let mut found = vec![Arc::as_ptr(to) as usize];
        let mut queue = std::collections::VecDeque::from([to.clone()]);
        while let Some(v) = queue.pop_front() {
            for c in v.children() {
                if c.edge == edge {
                    return false;
                }
                let key = Arc::as_ptr(&c.node) as usize;
                if !found.contains(&key) {
                    found.push(key);
                    queue.push_back(c.node.clone());
                }
            }
        }
        true
    }

    /// `bdrv_replace_node_noperm()`: moves every parent of `from` over to `to`, skipping those
    /// that would close a loop when `auto_skip`, failing on them otherwise.
    fn replace_node_noperm(
        from: &Arc<Node>,
        to: &Arc<Node>,
        auto_skip: bool,
        tran: &mut PermTran,
    ) -> Result<()> {
        for p in from.parents() {
            let stays = p.ops.as_ref().is_some_and(|o| o.stay_at_node());
            if stays || !Self::should_update_child(p.id, to) {
                if auto_skip {
                    continue;
                }
                return Err(Error::generic(format!(
                    "Should not change '{}' link to '{}'",
                    p.child_name, from.name
                )));
            }
            if crate::job::blocker::link_is_frozen(p.id) {
                return Err(Error::generic(format!(
                    "Cannot change '{}' link to '{}'",
                    p.child_name, from.name
                )));
            }
            from.replace_parent_edge(&p, to, tran);
        }
        Ok(())
    }

    /// `bdrv_replace_node()`: every parent of `from` that would not close a loop uses `to`
    /// instead, with the permissions worked out on the new graph. On failure nothing changes.
    /// Both nodes are drained for the length of the change.
    pub(crate) fn replace_node(from: &Arc<Node>, to: &Arc<Node>) -> Result<()> {
        let _d1 = from.drained();
        let _d2 = to.drained();
        let _wr = crate::graph_lock::wrlock();
        let mut tran = PermTran::default();
        let r = Self::replace_node_noperm(from, to, true, &mut tran)
            .and_then(|()| refresh_perms(&[from.clone(), to.clone()], None, &mut tran));
        match r {
            Ok(()) => {
                tran.commit();
                Ok(())
            }
            Err(e) => {
                tran.abort();
                Err(e)
            }
        }
    }

    /// `bdrv_replace_node_common()`: [`Node::replace_node`] that fails on a parent that
    /// would close a loop unless `auto_skip`, as `bdrv_drop_intermediate()` wants.
    pub(crate) fn replace_node_common(
        from: &Arc<Node>,
        to: &Arc<Node>,
        auto_skip: bool,
    ) -> Result<()> {
        let _d1 = from.drained();
        let _d2 = to.drained();
        let _wr = crate::graph_lock::wrlock();
        let mut tran = PermTran::default();
        let r = Self::replace_node_noperm(from, to, auto_skip, &mut tran)
            .and_then(|()| refresh_perms(&[from.clone(), to.clone()], None, &mut tran));
        match r {
            Ok(()) => {
                tran.commit();
                Ok(())
            }
            Err(e) => {
                tran.abort();
                Err(e)
            }
        }
    }

    /// `bdrv_append()`: puts `bs_new` on top of `bs_top`, which becomes its backing child, and
    /// moves the parents of `bs_top` to `bs_new`. `bs_new` must not have a backing child yet.
    /// On failure nothing changes.
    pub(crate) fn append(bs_new: &Arc<Node>, bs_top: &Arc<Node>) -> Result<()> {
        assert!(bs_new.backing().is_none(), "bdrv_append() on a node with a backing child");
        let _d1 = bs_top.drained();
        let _d2 = bs_new.drained();
        let _wr = crate::graph_lock::wrlock();
        let mut tran = PermTran::default();
        // bdrv_backing_role().
        let role = if bs_new.is_filter() {
            BDRV_CHILD_FILTERED | BDRV_CHILD_PRIMARY
        } else {
            BDRV_CHILD_COW
        };
        bs_new.attach_child_noperm("backing", bs_top.clone(), role, &mut tran);
        let r = Self::replace_node_noperm(bs_top, bs_new, true, &mut tran)
            .and_then(|()| refresh_perms(std::slice::from_ref(bs_new), None, &mut tran));
        let r = match r {
            Ok(()) => {
                tran.commit();
                bs_new.refresh_limits()
            }
            Err(e) => {
                tran.abort();
                Err(e)
            }
        };
        let _ = bs_top.refresh_limits();
        r
    }

    /// `bdrv_refresh_limits()`: the limits from the children, then the driver's.
    pub(crate) fn refresh_limits(&self) -> Result<()> {
        let mut bl = BlockLimits {
            // Every driver here has a byte interface.
            request_alignment: 1,
            ..BlockLimits::default()
        };
        let mut have_limits = false;
        for c in self.children.read().unwrap().iter() {
            if c.role & (BDRV_CHILD_DATA | BDRV_CHILD_FILTERED | BDRV_CHILD_COW) != 0 {
                merge_limits(&mut bl, &c.node.limits());
                have_limits = true;
            }
            if c.role & BDRV_CHILD_FILTERED != 0 {
                bl.has_variable_length |= c.node.limits().has_variable_length;
            }
        }
        if !have_limits {
            bl.min_mem_alignment = 512;
            bl.opt_mem_alignment = 4096;
            bl.max_iov = 1024;
        }
        self.driver.refresh_limits(self, &mut bl)?;
        let too_large = u64::from(bl.request_alignment) > BDRV_MAX_ALIGNMENT;
        *self.limits.write().unwrap() = bl;
        if too_large {
            return Err(Error::generic("Driver requires too large request alignment"));
        }
        Ok(())
    }

    /// `bdrv_refresh_total_sectors()`: asks the driver for the length, or takes `hint` (in
    /// sectors) when there is no answer.
    pub(crate) fn refresh_total_sectors(&self, hint: Option<i64>) -> io::Result<()> {
        match self.driver.getlength(self) {
            Ok(len) => {
                if len > BDRV_MAX_LENGTH {
                    return Err(errno(libc::EFBIG));
                }
                let sectors = len.div_ceil(BDRV_SECTOR_SIZE);
                self.total_sectors.store(sectors as i64, Ordering::SeqCst);
                Ok(())
            }
            Err(e) if is_enotsup(&e) && hint.is_some() => {
                self.total_sectors.store(hint.unwrap_or(0), Ordering::SeqCst);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// `bs->total_sectors`.
    pub(crate) fn total_sectors(&self) -> i64 {
        self.total_sectors.load(Ordering::SeqCst)
    }

    /// Sets `bs->total_sectors`, when a write past the end grew the node.
    pub(crate) fn set_total_sectors(&self, sectors: i64) {
        self.total_sectors.store(sectors, Ordering::SeqCst);
    }

    /// `bdrv_co_nb_sectors()`.
    pub(crate) fn nb_sectors(&self) -> io::Result<u64> {
        if self.limits().has_variable_length || self.total_sectors() < 0 {
            let hint = self.total_sectors();
            self.refresh_total_sectors((hint >= 0).then_some(hint))?;
        }
        Ok(self.total_sectors().max(0) as u64)
    }

    /// `bdrv_co_getlength()`: the length in bytes, a multiple of 512.
    pub(crate) fn getlength(&self) -> io::Result<u64> {
        let s = self.nb_sectors()?;
        s.checked_mul(BDRV_SECTOR_SIZE).ok_or_else(|| errno(libc::EFBIG))
    }

    /// `bdrv_co_parent_cb_resize()`: tell the parents the length changed.
    pub(crate) fn parent_cb_resize(&self) {
        for p in self.parents() {
            if let Some(ops) = &p.ops {
                ops.resize();
            }
        }
    }

    /// `bdrv_debug_event()`.
    pub(crate) fn debug_event(&self, event: BlkdebugEvent) {
        self.driver.debug_event(self, event);
    }

    /// The first node from this one down the primary children whose driver answers `f`, the
    /// search `bdrv_debug_breakpoint()` and friends do.
    fn debug_find<T>(&self, f: &dyn Fn(&Node) -> Option<T>) -> Option<T> {
        if let Some(r) = f(self) {
            return Some(r);
        }
        let mut bs = self.primary_bs();
        while let Some(n) = bs {
            if let Some(r) = f(&n) {
                return Some(r);
            }
            bs = n.primary_bs();
        }
        None
    }

    /// `bdrv_debug_breakpoint()`.
    pub(crate) fn debug_breakpoint(&self, event: &str, tag: &str) -> io::Result<()> {
        self.debug_find(&|n| n.driver.debug_breakpoint(n, event, tag))
            .unwrap_or_else(|| Err(errno(libc::ENOTSUP)))
    }

    /// `bdrv_debug_remove_breakpoint()`.
    pub(crate) fn debug_remove_breakpoint(&self, tag: &str) -> io::Result<()> {
        self.debug_find(&|n| n.driver.debug_remove_breakpoint(n, tag))
            .unwrap_or_else(|| Err(errno(libc::ENOTSUP)))
    }

    /// `bdrv_debug_resume()`.
    pub(crate) fn debug_resume(&self, tag: &str) -> io::Result<()> {
        self.debug_find(&|n| n.driver.debug_resume(n, tag))
            .unwrap_or_else(|| Err(errno(libc::ENOTSUP)))
    }

    /// `bdrv_debug_is_suspended()`.
    pub(crate) fn debug_is_suspended(&self, tag: &str) -> bool {
        self.debug_find(&|n| n.driver.debug_is_suspended(n, tag)).unwrap_or(false)
    }

    /// `bdrv_refresh_filename()`: brings `bs->exact_filename`, `bs->full_open_options` and
    /// `bs->filename` up to date, children first. A node that no plain file name opens again
    /// gets a `json:` pseudo-filename of its options.
    pub(crate) fn refresh_filename(&self) {
        let children = self.children();
        for c in &children {
            c.node.refresh_filename();
        }
        let backing = children.iter().find(|c| c.name == "backing");
        // Without I/O the backing file does not change anything, so qemu-img can pretend
        // it was not overridden.
        let backing_overridden = !self.flags().no_io && self.backing_overridden(backing);

        let mut opts = QDict::new();
        let generate_json = self.append_strong_runtime_options(&mut opts) || backing_overridden;
        for c in &children {
            if c.name == "backing" && !backing_overridden {
                // The image header names the same backing file.
                continue;
            }
            let child_opts = c.node.meta.lock().unwrap().full_open_options.clone();
            opts.put(c.name.clone(), QValue::Dict(child_opts));
        }
        if backing_overridden && backing.is_none() {
            // Force no backing file.
            opts.put("backing", QValue::Null);
        }

        // The driver's .bdrv_refresh_filename, or a protocol node's own file.
        let mut exact =
            self.driver.exact_filename(self).or_else(|| self.driver.filename()).unwrap_or_default();
        if exact.is_empty() {
            // A format node without strong options on a protocol node opens again from the
            // file name of the protocol node. Filters cannot be probed, so they never do.
            if let Some(p) = self.primary_bs() {
                let pf = p.meta.lock().unwrap().exact_filename.clone();
                if !pf.is_empty() && p.is_protocol() && !self.is_filter() && !generate_json {
                    exact = pf;
                }
            }
        }
        let filename = if exact.is_empty() {
            let mut f = format!("json:{}", QValue::Dict(opts.clone()).to_json());
            if f.len() >= PATH_MAX {
                // Give the user a hint the name was cut, as QEMU does.
                let mut cut = PATH_MAX - 4;
                while !f.is_char_boundary(cut) {
                    cut -= 1;
                }
                f.truncate(cut);
                f.push_str("...");
            }
            f
        } else {
            exact.clone()
        };
        let mut meta = self.meta.lock().unwrap();
        meta.full_open_options = opts;
        meta.exact_filename = exact;
        meta.filename = filename;
    }

    /// `bdrv_backing_overridden()`: whether the backing node is not the one the image header
    /// names. May say yes when opening the header's backing file would give the same node.
    pub(crate) fn backing_overridden(&self, backing: Option<&Child>) -> bool {
        let auto = self.meta.lock().unwrap().auto_backing_file.clone();
        match backing {
            Some(b) => auto != b.node.meta.lock().unwrap().filename,
            None => !auto.is_empty(),
        }
    }

    /// `append_strong_runtime_options()`: copies the options that change the data of the
    /// node into `d`. True when there was one other than `driver` and `filename`.
    fn append_strong_runtime_options(&self, d: &mut QDict) -> bool {
        let meta = self.meta.lock().unwrap();
        let strong = self.def.map_or(&[][..], |def| def.strong_runtime_opts);
        let mut found_any = false;
        for name in ["driver", "filename"].iter().chain(strong) {
            let mut given = false;
            if let Some(prefix) = name.strip_suffix('.') {
                let prefix = format!("{prefix}.");
                for (k, v) in meta.options.iter() {
                    if k.starts_with(&prefix) {
                        d.put(k, v.clone());
                        given = true;
                    }
                }
            } else if let Some(v) = meta.options.get(name) {
                d.put(*name, v.clone());
                given = true;
            }
            if given && *name != "driver" && *name != "filename" {
                found_any = true;
            }
        }
        if !d.contains_key("driver") {
            // Nodes made without a driver option get one here.
            d.put("driver", self.driver_name);
        }
        found_any
    }

    /// Whether the node is `BDRV_O_INACTIVE`.
    pub(crate) fn is_inactive(&self) -> bool {
        self.flags().inactive
    }
}

impl Drop for Node {
    fn drop(&mut self) {
        // bdrv_close(): the driver first, then the children.
        self.driver.close(self);
        let children = std::mem::take(&mut *self.children.write().unwrap());
        for c in children.iter().rev() {
            let _ = c.node.update_parent(c.edge, None);
        }
    }
}

/// `bdrv_merge_limits()`.
pub(crate) fn merge_limits(dst: &mut BlockLimits, src: &BlockLimits) {
    fn min_non_zero<T: Ord + Default + Copy>(a: T, b: T) -> T {
        if a == T::default() {
            b
        } else if b == T::default() {
            a
        } else {
            a.min(b)
        }
    }
    dst.pdiscard_alignment = dst.pdiscard_alignment.max(src.pdiscard_alignment);
    dst.opt_transfer = dst.opt_transfer.max(src.opt_transfer);
    dst.max_transfer = min_non_zero(dst.max_transfer, src.max_transfer);
    dst.max_hw_transfer = min_non_zero(dst.max_hw_transfer, src.max_hw_transfer);
    dst.opt_mem_alignment = dst.opt_mem_alignment.max(src.opt_mem_alignment);
    dst.min_mem_alignment = dst.min_mem_alignment.max(src.min_mem_alignment);
    dst.max_iov = min_non_zero(dst.max_iov, src.max_iov);
    dst.max_hw_iov = min_non_zero(dst.max_hw_iov, src.max_hw_iov);
}
