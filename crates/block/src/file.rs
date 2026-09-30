// SPDX-License-Identifier: GPL-2.0-or-later

//! The `file` protocol driver from block/file-posix.c, and the part of `host_device` and
//! `host_cdrom` that is the same code in QEMU: `raw_open_common()`, the request handlers and the
//! limits. The device specific parts of those two drivers are in `protocol/host.rs`.
//!
//! Reads and writes go through the engine in `protocol/aio.rs`: positioned reads and writes on
//! the caller's thread for `aio=threads`, io_uring or linux-aio for `aio=native`.
//!
//! Differences from QEMU:
//!
//! - `XFS_IOC_DIOINFO` is not asked when probing the `O_DIRECT` alignment. The trial reads that
//!   follow in `raw_probe_alignment()` find the same alignment on XFS.
//! - `BDRV_REQ_FUA` is not passed down, so the generic layer emulates it with a flush where
//!   QEMU would use `RWF_DSYNC` or the FUA support of linux-aio and io_uring.
//! - The XFS workaround of `raw_do_pwrite_zeroes()`, which serialises zero writes past the end
//!   of the file, is not needed: requests are synchronous.
//! - `drop-cache` and `x-check-cache-dropped` are accepted and have no effect, as there is no
//!   incoming migration to drop the page cache for.
//! - Zoned block devices, SCSI generic passthrough (`bs->sg`), persistent reservation managers
//!   and `max_hw_iov` for SCSI generic devices are not supported.

use std::fs::{File, OpenOptions};
use std::io::{self, Seek, SeekFrom};
use std::os::fd::{AsFd, BorrowedFd};
use std::os::unix::fs::{FileExt, FileTypeExt, MetadataExt, OpenOptionsExt};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, RwLock};

use ruvm_base::{Error, Result, report};
use ruvm_qapi::QDict;
use ruvm_qapi::types::{
    BlockdevAioOptions, BlockdevCreateOptionsFile, BlockdevCreateOptionsU, BlockdevOptionsFile,
    BlockdevOptionsU, OnOffAuto, PreallocMode,
};
use ruvm_qapi::visit::{parse_option_size, qapi_bool_parse};

use crate::drivers::{DriverDef, OpenArgs};

use crate::graph::BlockGraph;
use crate::node::{
    BDRV_REQ_MAY_UNMAP, BDRV_REQ_NO_FALLBACK, BDRV_REQ_ZERO_WRITE, BDRV_SECTOR_SIZE, BlockLimits,
    Driver, Node, NodeFlags, ReopenState, errno, is_enotsup, is_errno,
};
use crate::perm::{BLK_PERM_ALL, BLK_PERM_RESIZE, BLK_PERM_WRITE, perm_names};
use crate::protocol::aio::AioEngine;
use crate::sys::{self, LockOps};

/// `RAW_LOCK_PERM_BASE`. libvirt uses byte 0, QEMU leaves room for it to grow.
pub(crate) const RAW_LOCK_PERM_BASE: u64 = 100;
/// `RAW_LOCK_SHARED_BASE`.
pub(crate) const RAW_LOCK_SHARED_BASE: u64 = 200;

/// `MAX_BLOCKSIZE`, the largest alignment `raw_probe_alignment()` tries.
const MAX_BLOCKSIZE: usize = 4096;

/// Which of file-posix's drivers a node belongs to: `s->type` and the `device` argument of
/// `raw_open_common()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub(crate) enum FileKind {
    /// `file`, a regular file.
    File,
    /// `host_device`, a block or character device.
    HostDevice,
    /// `host_cdrom`, a CD-ROM drive, opened with `O_NONBLOCK` so that an empty drive opens.
    HostCdrom,
}

impl FileKind {
    /// The driver's `format_name`.
    pub(crate) fn format_name(self) -> &'static str {
        match self {
            FileKind::File => "file",
            FileKind::HostDevice => "host_device",
            FileKind::HostCdrom => "host_cdrom",
        }
    }
}

/// The lock bytes this file holds and the permissions they stand for, the lock part of
/// `BDRVRawState`.
#[derive(Debug)]
struct LockState {
    perm: u64,
    shared_perm: u64,
    locked_perm: u64,
    locked_shared_perm: u64,
}

/// How the descriptor is open, the part of `s->open_flags` a reopen may change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OpenMode {
    write: bool,
    direct: bool,
}

/// The alignments `raw_probe_alignment()` finds.
#[derive(Clone, Copy, Debug)]
struct Alignment {
    /// `s->needs_alignment`.
    needs: bool,
    /// `bs->bl.request_alignment`.
    request: u32,
    /// `s->buf_align`.
    buf: usize,
}

/// `BDRVRawState`.
pub(crate) struct FileDriver {
    kind: FileKind,
    /// `s->fd`. A reopen may swap it for one opened with other flags.
    file: RwLock<File>,
    filename: String,
    mode: Mutex<OpenMode>,
    /// `None` when `use_lock` is false.
    lock_ops: Option<LockOps>,
    locks: Mutex<LockState>,
    /// `bs->bl.request_alignment`.
    request_alignment: AtomicU32,
    /// `s->buf_align`.
    buf_align: AtomicUsize,
    /// `s->needs_alignment`.
    needs_alignment: AtomicBool,
    /// The alignment of the offsets, lengths and buffers of the requests this driver issues,
    /// the larger of the two above. Requests from the generic layer are already aligned in
    /// offset and length, the buffers are bounced when this is not 1.
    align: AtomicU64,
    /// Serialises read-modify-write cycles for unaligned writes.
    rmw: Mutex<()>,
    has_discard: AtomicBool,
    has_write_zeroes: AtomicBool,
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    has_fallocate: AtomicBool,
    engine: AioEngine,
}

/// What a reopen prepared: a new descriptor when the flags change.
struct ReopenFd {
    file: File,
    mode: OpenMode,
    alignment: Alignment,
}

/// Opens a `file` node, `raw_open()` and `raw_open_common()`. `flags.read_only` is cleared
/// or set here: with `auto_read_only` a file that cannot be opened for writing is opened
/// read-only instead, as `bdrv_apply_auto_read_only()` allows.
pub(crate) fn file_open(
    kind: FileKind,
    o: &BlockdevOptionsFile,
    flags: &mut NodeFlags,
    auto_read_only: bool,
) -> Result<FileDriver> {
    let filename = o.filename.clone();
    let aio = o.aio;
    let use_lock = match o.locking.unwrap_or_default() {
        OnOffAuto::On => {
            if !sys::has_ofd_lock() {
                report::warn_report(
                    "File lock requested but OFD locking syscall is unavailable, falling back \
                     to POSIX file locks",
                );
                eprintln!("Due to the implementation, locks can be lost unexpectedly.");
            }
            true
        }
        OnOffAuto::Off => false,
        OnOffAuto::Auto => sys::has_ofd_lock(),
    };
    let lock_ops = if !use_lock || !sys::has_locks() {
        None
    } else if sys::has_ofd_lock() {
        Some(LockOps::Ofd)
    } else {
        Some(LockOps::Posix)
    };
    if let Some(id) = &o.pr_manager {
        return Err(Error::generic(format!("No persistent reservation manager with id '{id}'")));
    }

    let nonblock = kind == FileKind::HostCdrom;
    let file = match open_file(&filename, !flags.read_only, flags.direct, nonblock) {
        Ok(f) => f,
        Err(e) if auto_read_only && !flags.read_only && is_read_only_error(&e) => {
            flags.read_only = true;
            open_file(&filename, false, flags.direct, nonblock)
                .map_err(|e| e.into_error(&filename))?
        }
        Err(e) => return Err(e.into_error(&filename)),
    };

    if !flags.read_only {
        check_hdev_writable(&file).map_err(|e| Error::from_io("The device is not writable", e))?;
    }

    if aio == Some(BlockdevAioOptions::Native) {
        if cfg!(any(target_os = "linux", target_os = "android")) {
            if !flags.direct {
                return Err(Error::generic(
                    "aio=native was specified, but it requires cache.direct=on, which was not \
                     specified.",
                ));
            }
        } else {
            return Err(Error::generic(
                "aio=native was specified, but is not supported in this build.",
            ));
        }
    }

    let md = file.metadata().map_err(|e| Error::from_io("Could not stat file", e))?;
    let ft = md.file_type();
    if kind == FileKind::File {
        if !ft.is_file() {
            return Err(Error::generic(format!(
                "'{}' driver requires '{filename}' to be a regular file",
                kind.format_name()
            )));
        }
    } else if !(ft.is_char_device() || ft.is_block_device()) {
        return Err(Error::generic(format!(
            "'{}' driver requires '{filename}' to be either a character or block device",
            kind.format_name()
        )));
    }

    let alignment = probe_alignment(&file, flags.direct)?;
    let engine = match aio {
        Some(BlockdevAioOptions::Native) => AioEngine::native(),
        Some(BlockdevAioOptions::Threads) => AioEngine::Threads,
        None => AioEngine::default_for(flags.direct),
    };

    let d = FileDriver {
        kind,
        file: RwLock::new(file),
        filename,
        mode: Mutex::new(OpenMode { write: !flags.read_only, direct: flags.direct }),
        lock_ops,
        locks: Mutex::new(LockState {
            perm: 0,
            shared_perm: BLK_PERM_ALL,
            locked_perm: 0,
            locked_shared_perm: 0,
        }),
        request_alignment: AtomicU32::new(0),
        buf_align: AtomicUsize::new(0),
        needs_alignment: AtomicBool::new(false),
        align: AtomicU64::new(1),
        rmw: Mutex::new(()),
        has_discard: AtomicBool::new(true),
        has_write_zeroes: AtomicBool::new(true),
        has_fallocate: AtomicBool::new(ft.is_file()),
        engine,
    };
    d.set_alignment(alignment);
    Ok(d)
}

/// Why opening failed, kept apart so the auto-read-only retry can look at the errno.
enum OpenError {
    Io(io::Error),
    NoDirect(io::Error),
}

impl OpenError {
    fn into_error(self, filename: &str) -> Error {
        match self {
            OpenError::Io(e) => Error::from_io(format_args!("Could not open '{filename}'"), e),
            OpenError::NoDirect(e) => Error::with_cause(
                format!("Could not open '{filename}': filesystem does not support O_DIRECT"),
                e,
            ),
        }
    }
}

fn is_read_only_error(e: &OpenError) -> bool {
    let OpenError::Io(e) = e else { return false };
    matches!(e.raw_os_error(), Some(n) if n == libc::EACCES || n == libc::EROFS || n == libc::EPERM)
}

/// `qemu_open()` with the flags `raw_parse_flags()` picks. Without `O_DIRECT` on the host,
/// file-posix uses `O_DSYNC` for `cache.direct=on`, and so does this. `nonblock` is the
/// `O_NONBLOCK` of `host_cdrom`.
fn open_file(
    filename: &str,
    write: bool,
    direct: bool,
    nonblock: bool,
) -> std::result::Result<File, OpenError> {
    let mut oo = OpenOptions::new();
    oo.read(true).write(write);
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let direct_flag = libc::O_DIRECT;
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    let direct_flag = libc::O_DSYNC;
    let mut custom = if nonblock { libc::O_NONBLOCK } else { 0 };
    if direct {
        custom |= direct_flag;
    }
    oo.custom_flags(custom);
    match oo.open(filename) {
        Ok(f) => Ok(f),
        Err(e) if direct && e.raw_os_error() == Some(libc::EINVAL) => {
            // "Give more helpful error message for O_DIRECT".
            let mut plain = OpenOptions::new();
            plain.read(true).write(write);
            if plain.open(filename).is_ok() {
                Err(OpenError::NoDirect(e))
            } else {
                Err(OpenError::Io(e))
            }
        }
        Err(e) => Err(OpenError::Io(e)),
    }
}

/// `check_hdev_writable()`: a Linux block device set read-only with blockdev(8) opens for
/// writing and then fails every write, so ask it.
fn check_hdev_writable(file: &File) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        if !file.metadata()?.file_type().is_block_device() {
            return Ok(());
        }
        if sys::dev_ioctl(file.as_fd(), sys::DevIoctl::BlkRoGet)? != 0 {
            return Err(errno(libc::EACCES));
        }
    }
    let _ = file;
    Ok(())
}

/// `probe_logical_blocksize()`: `BLKSSZGET` on Linux, `DKIOCGETBLOCKSIZE` on macOS. A regular
/// file fails with the ioctl's error, `ENOTTY` usually.
pub(crate) fn probe_logical_blocksize(file: &File) -> io::Result<u32> {
    #[cfg(target_os = "linux")]
    return rustix::fs::ioctl_blksszget(file).map_err(io::Error::from);
    #[cfg(target_os = "macos")]
    return sys::dev_ioctl(file.as_fd(), sys::DevIoctl::GetBlockSize).map(|v| v as u32);
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = file;
        Err(errno(libc::ENOTSUP))
    }
}

/// `probe_physical_blocksize()`: `BLKPBSZGET`, which only Linux has. Only DASD devices would
/// use it, see `hdev_probe_blocksizes()`.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn probe_physical_blocksize(file: &File) -> io::Result<u32> {
    #[cfg(target_os = "linux")]
    return rustix::fs::ioctl_blkpbszget(file).map_err(io::Error::from);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = file;
        Err(errno(libc::ENOTSUP))
    }
}

/// `dio_byte_aligned()`: NFS takes `O_DIRECT` requests of any alignment.
fn dio_byte_aligned(file: &File) -> bool {
    #[cfg(target_os = "linux")]
    {
        const NFS_SUPER_MAGIC: i64 = 0x6969;
        if let Ok(s) = rustix::fs::fstatfs(file) {
            // `f_type` is not an i64 on every architecture.
            #[allow(clippy::unnecessary_cast)]
            return s.f_type as i64 == NFS_SUPER_MAGIC;
        }
    }
    let _ = file;
    false
}

/// `raw_is_io_aligned()`: whether a read of `buf` at offset 0 is allowed.
fn is_io_aligned(file: &File, buf: &mut [u8]) -> bool {
    match file.read_at(buf, 0) {
        Ok(_) => true,
        // Linux says EINVAL for misaligned O_DIRECT reads. Other errors, a failing drive for
        // one, do not tell anything about the alignment.
        Err(e) => cfg!(target_os = "linux") && e.raw_os_error() != Some(libc::EINVAL),
    }
}

/// A buffer of `len` bytes whose start is aligned to `align`, as a vector and the offset of
/// the aligned part in it: `qemu_memalign()`.
fn aligned_vec(len: usize, align: usize) -> (Vec<u8>, usize) {
    let v = vec![0u8; len + align];
    let skip = v.as_ptr().align_offset(align);
    (v, skip)
}

/// `raw_probe_alignment()` after `raw_needs_alignment()`.
fn probe_alignment(file: &File, direct: bool) -> Result<Alignment> {
    let needs = direct && !dio_byte_aligned(file);
    if !needs {
        return Ok(Alignment { needs, request: 1, buf: 1 });
    }
    let max_align = MAX_BLOCKSIZE.max(rustix::param::page_size());
    let alignments = [1usize, 512, 1024, 2048, 4096];
    // Try the logical block size first.
    let mut request = probe_logical_blocksize(file).map_or(0, |v| v as usize);
    if request == 0 {
        let (mut v, skip) = aligned_vec(max_align, max_align);
        for align in alignments {
            if is_io_aligned(file, &mut v[skip..skip + align]) {
                // A byte aligned read working means probing failed: take the safe value.
                request = if align != 1 { align } else { max_align };
                break;
            }
        }
    }
    let mut buf = 0;
    let (mut v, skip) = aligned_vec(2 * max_align, max_align);
    for align in alignments {
        let start = skip + align;
        if is_io_aligned(file, &mut v[start..start + max_align]) {
            buf = if align != 1 { align } else { request };
            break;
        }
    }
    if buf == 0 || request == 0 {
        return Err(Error::generic("Could not find working O_DIRECT alignment")
            .hint("Try cache.direct=off\n"));
    }
    Ok(Alignment { needs, request: request as u32, buf })
}

/// `translate_err()`: the errors that mean the operation is not there.
#[cfg(target_os = "linux")]
fn translate_err(e: io::Error) -> io::Error {
    match e.raw_os_error() {
        Some(n)
            if n == libc::ENODEV
                || n == libc::ENOSYS
                || n == libc::EOPNOTSUPP
                || n == libc::ENOTTY =>
        {
            errno(libc::ENOTSUP)
        }
        _ => e,
    }
}

/// `do_fallocate()`: `fallocate()` retried on `EINTR`, with [`translate_err`].
#[cfg(target_os = "linux")]
fn do_fallocate(
    file: &File,
    mode: rustix::fs::FallocateFlags,
    offset: u64,
    len: u64,
) -> io::Result<()> {
    loop {
        match rustix::fs::fallocate(file, mode, offset, len) {
            Ok(()) => return Ok(()),
            Err(rustix::io::Errno::INTR) => {}
            Err(e) => return Err(translate_err(e.into())),
        }
    }
}

/// `raw_lock_error_setg_errno()`: lock contention gets no strerror text.
fn lock_error(e: io::Error, msg: String) -> Error {
    match e.raw_os_error() {
        Some(n) if n == libc::EAGAIN || n == libc::EACCES => Error::with_cause(msg, e),
        _ => Error::from_io(msg, e),
    }
}

/// `raw_apply_lock_bytes()`: lock the bytes for `perm_lock_bits` and
/// `shared_perm_lock_bits` (the permissions that are not shared), and with `unlock` let go
/// of the bytes that are no longer needed.
fn apply_lock_bytes(
    fd: BorrowedFd<'_>,
    ops: LockOps,
    s: &mut LockState,
    perm_lock_bits: u64,
    shared_perm_lock_bits: u64,
    unlock: bool,
) -> Result<()> {
    let sets = [
        (RAW_LOCK_PERM_BASE, perm_lock_bits, &mut s.locked_perm),
        (RAW_LOCK_SHARED_BASE, shared_perm_lock_bits, &mut s.locked_shared_perm),
    ];
    for (base, want, locked) in sets {
        for i in 0..4 {
            let off = base + i;
            let bit = 1u64 << i;
            if want & bit != 0 && *locked & bit == 0 {
                sys::lock_fd(fd, ops, off, false)
                    .map_err(|e| lock_error(e, format!("Failed to lock byte {off}")))?;
                *locked |= bit;
            } else if unlock && *locked & bit != 0 && want & bit == 0 {
                sys::unlock_fd(fd, ops, off)
                    .map_err(|e| Error::from_io(format!("Failed to unlock byte {off}"), e))?;
                *locked &= !bit;
            }
        }
    }
    Ok(())
}

/// `raw_check_lock_bytes()`: nobody else may have unshared what we want, or hold what we do
/// not share.
fn check_lock_bytes(fd: BorrowedFd<'_>, ops: LockOps, perm: u64, shared_perm: u64) -> Result<()> {
    for i in 0..4 {
        let p = 1u64 << i;
        if perm & p != 0 {
            sys::lock_fd_test(fd, ops, RAW_LOCK_SHARED_BASE + i, true)
                .map_err(|e| lock_error(e, format!("Failed to get \"{}\" lock", perm_names(p))))?;
        }
    }
    for i in 0..4 {
        let p = 1u64 << i;
        if shared_perm & p == 0 {
            sys::lock_fd_test(fd, ops, RAW_LOCK_PERM_BASE + i, true).map_err(|e| {
                lock_error(e, format!("Failed to get shared \"{}\" lock", perm_names(p)))
            })?;
        }
    }
    Ok(())
}

/// Warns once about a file system whose `FALLOC_FL_PUNCH_HOLE` says `EINVAL`.
#[cfg(target_os = "linux")]
static PUNCH_HOLE_EINVAL_WARNED: AtomicBool = AtomicBool::new(false);

impl FileDriver {
    /// Which driver this node belongs to.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn kind(&self) -> FileKind {
        self.kind
    }

    /// The name of the engine reads and writes go through.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn engine_name(&self) -> &'static str {
        self.engine.name()
    }

    /// `s->fd`, for the device ioctls.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub(crate) fn with_fd<R>(&self, f: impl FnOnce(&File) -> R) -> R {
        f(&self.file.read().unwrap())
    }

    fn set_alignment(&self, a: Alignment) {
        self.needs_alignment.store(a.needs, Ordering::Relaxed);
        self.request_alignment.store(a.request, Ordering::Relaxed);
        self.buf_align.store(a.buf, Ordering::Relaxed);
        self.align.store(u64::from(a.request).max(a.buf as u64), Ordering::Relaxed);
    }

    fn align(&self) -> u64 {
        self.align.load(Ordering::Relaxed)
    }

    fn apply_lock_bytes(
        &self,
        ops: LockOps,
        s: &mut LockState,
        perm_lock_bits: u64,
        shared_perm_lock_bits: u64,
        unlock: bool,
    ) -> Result<()> {
        let f = self.file.read().unwrap();
        apply_lock_bytes(f.as_fd(), ops, s, perm_lock_bits, shared_perm_lock_bits, unlock)
    }

    fn check_lock_bytes(&self, ops: LockOps, perm: u64, shared_perm: u64) -> Result<()> {
        check_lock_bytes(self.file.read().unwrap().as_fd(), ops, perm, shared_perm)
    }

    /// `raw_getlength()`: `lseek(SEEK_END)`, and on macOS the block count of a device first.
    fn raw_getlength(&self, file: &File) -> io::Result<u64> {
        #[cfg(target_os = "macos")]
        {
            // QEMU tests the S_IFCHR bit, which block devices have too.
            if file.metadata()?.mode() & u32::from(libc::S_IFCHR) != 0 {
                let fd = file.as_fd();
                let count = sys::dev_ioctl(fd, sys::DevIoctl::GetBlockCount);
                let size = sys::dev_ioctl(fd, sys::DevIoctl::GetBlockSize);
                if let (Ok(count), Ok(size)) = (count, size) {
                    if count * size != 0 {
                        return Ok(count * size);
                    }
                }
            }
        }
        let mut f = file;
        f.seek(SeekFrom::End(0))
    }

    /// Reads at `offset` until `buf` is full, padding with zeroes past the end of the file as
    /// `handle_aiocb_rw()` does.
    fn read_full(&self, file: &File, mut offset: u64, mut buf: &mut [u8]) -> io::Result<()> {
        let mut progress = false;
        while !buf.is_empty() {
            match self.engine.read_at(file, buf, offset) {
                Ok(0) => {
                    buf.fill(0);
                    return Ok(());
                }
                Ok(n) => {
                    buf = &mut buf[n..];
                    offset += n as u64;
                    progress = true;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                // After a short read with O_DIRECT the next offset is unaligned and the kernel
                // says EINVAL. Like handle_aiocb_rw_linear(), take that as the end of the file.
                Err(e) if progress && self.align() > 1 && is_errno(&e, libc::EINVAL) => {
                    buf.fill(0);
                    return Ok(());
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Writes all of `buf` at `offset`. A write that makes no progress is `EINVAL`, as in
    /// `handle_aiocb_rw()`.
    fn write_full(&self, file: &File, mut offset: u64, mut buf: &[u8]) -> io::Result<()> {
        while !buf.is_empty() {
            match self.engine.write_at(file, buf, offset) {
                Ok(0) => return Err(errno(libc::EINVAL)),
                Ok(n) => {
                    buf = &buf[n..];
                    offset += n as u64;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// The aligned span covering `offset..offset + len`.
    fn aligned_span(&self, offset: u64, len: u64) -> (u64, usize) {
        let align = self.align();
        let start = offset - offset % align;
        let end = (offset + len).next_multiple_of(align);
        (start, (end - start) as usize)
    }

    /// `handle_aiocb_write_zeroes_block()`: `BLKZEROOUT`, unless the caller cannot take the
    /// slow fallback the kernel may use for it.
    fn write_zeroes_block(
        &self,
        file: &File,
        offset: u64,
        bytes: u64,
        flags: u32,
    ) -> io::Result<()> {
        if !self.has_write_zeroes.load(Ordering::Relaxed) {
            return Err(errno(libc::ENOTSUP));
        }
        #[cfg(target_os = "linux")]
        if flags & BDRV_REQ_NO_FALLBACK == 0 {
            let r = sys::dev_ioctl(file.as_fd(), sys::DevIoctl::BlkZeroOut(offset, bytes));
            return match r {
                Ok(_) => Ok(()),
                Err(e) => {
                    let e = translate_err(e);
                    if is_enotsup(&e) {
                        self.has_write_zeroes.store(false, Ordering::Relaxed);
                    }
                    Err(e)
                }
            };
        }
        let _ = (file, offset, bytes, flags);
        Err(errno(libc::ENOTSUP))
    }

    /// `handle_aiocb_write_zeroes()` for a regular file: `FALLOC_FL_ZERO_RANGE`, then punching
    /// a hole and allocating it again, then allocating past the end of the file.
    fn write_zeroes_file(&self, file: &File, offset: u64, bytes: u64) -> io::Result<()> {
        #[cfg(target_os = "linux")]
        {
            use rustix::fs::FallocateFlags;
            if self.has_write_zeroes.load(Ordering::Relaxed) {
                match do_fallocate(file, FallocateFlags::ZERO_RANGE, offset, bytes) {
                    Err(e) if is_enotsup(&e) => {
                        self.has_write_zeroes.store(false, Ordering::Relaxed)
                    }
                    // Some file systems say EINVAL for unaligned ranges, try the other ways.
                    Err(e) if is_errno(&e, libc::EINVAL) => {}
                    r => return r,
                }
            }
            if self.has_discard.load(Ordering::Relaxed)
                && self.has_fallocate.load(Ordering::Relaxed)
            {
                let punch = FallocateFlags::PUNCH_HOLE | FallocateFlags::KEEP_SIZE;
                match do_fallocate(file, punch, offset, bytes) {
                    Ok(()) => match do_fallocate(file, FallocateFlags::empty(), offset, bytes) {
                        Err(e) if is_enotsup(&e) => {
                            self.has_fallocate.store(false, Ordering::Relaxed);
                        }
                        r => return r,
                    },
                    Err(e) if is_errno(&e, libc::EINVAL) => {
                        if !PUNCH_HOLE_EINVAL_WARNED.swap(true, Ordering::Relaxed) {
                            report::warn_report(
                                "Your file system is misbehaving: fallocate(FALLOC_FL_PUNCH_HOLE) \
                                 returned EINVAL. Please report this bug to your file system vendor.",
                            );
                        }
                    }
                    Err(e) if is_enotsup(&e) => self.has_discard.store(false, Ordering::Relaxed),
                    Err(e) => return Err(e),
                }
            }
            // Last resort: extend the file with zeroes.
            let len = self.raw_getlength(file);
            if self.has_fallocate.load(Ordering::Relaxed) && len.is_ok_and(|len| offset >= len) {
                match do_fallocate(file, FallocateFlags::empty(), offset, bytes) {
                    Err(e) if is_enotsup(&e) => self.has_fallocate.store(false, Ordering::Relaxed),
                    r => return r,
                }
            }
        }
        let _ = (file, offset, bytes);
        Err(errno(libc::ENOTSUP))
    }

    /// `raw_co_truncate()`.
    fn truncate_impl(&self, offset: u64, exact: bool, prealloc: PreallocMode) -> Result<()> {
        let file = self.file.read().unwrap();
        let md = file.metadata().map_err(|e| Error::from_io("Failed to fstat() the file", e))?;
        let ft = md.file_type();
        if ft.is_file() {
            // Always resizes to the exact `offset`.
            return regular_truncate(&file, offset, prealloc);
        }
        if prealloc != PreallocMode::Off {
            return Err(Error::with_cause(
                format!(
                    "Preallocation mode '{}' unsupported for this non-regular file",
                    prealloc.as_str()
                ),
                errno(libc::ENOTSUP),
            ));
        }
        if ft.is_char_device() || ft.is_block_device() {
            let cur = self.raw_getlength(&file);
            let cur = cur.map_or(-1, |l| l as i64);
            if offset as i64 != cur && exact {
                return Err(Error::with_cause("Cannot resize device files", errno(libc::ENOTSUP)));
            } else if offset as i64 > cur {
                return Err(Error::with_cause("Cannot grow device files", errno(libc::EINVAL)));
            }
            Ok(())
        } else {
            Err(Error::with_cause("Resizing this file is not supported", errno(libc::ENOTSUP)))
        }
    }

    /// `raw_refresh_limits()` for the parts that need a block device: the discard granularity
    /// and segment count from sysfs, `BLKSECTGET`, and the write zeroes alignment.
    #[cfg(target_os = "linux")]
    fn refresh_blk_limits(
        &self,
        file: &File,
        md: &std::fs::Metadata,
        bl: &mut BlockLimits,
    ) -> Result<()> {
        if let Ok(max) = sys::dev_ioctl(file.as_fd(), sys::DevIoctl::BlkSectGet) {
            let max = max * 512;
            if max > 0 && max <= crate::node::BDRV_REQUEST_MAX_BYTES {
                bl.max_hw_transfer = max;
            }
        }
        let sysfs = |attr: &str| -> Option<String> {
            let rdev = md.rdev();
            let path = format!(
                "/sys/dev/block/{}:{}/queue/{attr}",
                rustix::fs::major(rdev),
                rustix::fs::minor(rdev)
            );
            let s = std::fs::read_to_string(path).ok()?;
            Some(s.strip_suffix('\n').unwrap_or(&s).to_owned())
        };
        if let Some(n) = sysfs("max_segments").and_then(|s| s.parse::<u32>().ok()) {
            if n > 0 {
                bl.max_hw_iov = n;
            }
        }
        // Linux's "discard_granularity" is QEMU's "discard_alignment".
        if let Some(mut dalign) = sysfs("discard_granularity").and_then(|s| s.parse::<u32>().ok()) {
            if dalign != 0 {
                let ralign = bl.request_alignment;
                if dalign < ralign && ralign % dalign == 0 {
                    dalign = ralign;
                }
                if dalign % ralign != 0 {
                    return Err(Error::generic(format!(
                        "Invalid pdiscard_alignment limit {dalign} is not a multiple of \
                         request_alignment {ralign}"
                    )));
                }
                bl.pdiscard_alignment = dalign;
            }
        }
        // Linux wants write zeroes aligned to the logical block size even when reads and
        // writes need no alignment.
        if !self.needs_alignment.load(Ordering::Relaxed) {
            bl.pwrite_zeroes_alignment = probe_logical_blocksize(file)
                .map_err(|e| Error::from_io("Failed to probe logical block size", e))?;
        }
        Ok(())
    }

    /// `cdrom_co_is_inserted()`.
    #[cfg(target_os = "linux")]
    fn cdrom_is_inserted(&self) -> bool {
        let r = self.with_fd(|f| sys::dev_ioctl(f.as_fd(), sys::DevIoctl::CdromDriveStatus));
        r.is_ok_and(|s| s == sys::CDS_DISC_OK)
    }
}

impl Driver for FileDriver {
    fn pread(&self, _bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let file = self.file.read().unwrap();
        let align = self.align();
        if align == 1 {
            return self.read_full(&file, offset, buf);
        }
        let (start, len) = self.aligned_span(offset, buf.len() as u64);
        let (mut v, skip) = aligned_vec(len, align as usize);
        let aligned = &mut v[skip..skip + len];
        self.read_full(&file, start, aligned)?;
        let head = (offset - start) as usize;
        buf.copy_from_slice(&aligned[head..head + buf.len()]);
        Ok(())
    }

    fn pwrite(&self, _bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        let file = self.file.read().unwrap();
        let align = self.align();
        if align == 1 {
            return self.write_full(&file, offset, buf);
        }
        let (start, len) = self.aligned_span(offset, buf.len() as u64);
        let (mut v, skip) = aligned_vec(len, align as usize);
        let aligned = &mut v[skip..skip + len];
        let _guard = self.rmw.lock().unwrap();
        let head = (offset - start) as usize;
        if head != 0 || len != buf.len() {
            self.read_full(&file, start, aligned)?;
        }
        aligned[head..head + buf.len()].copy_from_slice(buf);
        if self.kind != FileKind::File {
            return self.write_full(&file, start, aligned);
        }
        // Do not grow the file past what was asked for: a write that ends inside the last
        // aligned block and past the end of the file goes out whole and is cut back.
        let old_len = file.metadata()?.len();
        self.write_full(&file, start, aligned)?;
        let want = old_len.max(offset + buf.len() as u64);
        if start + len as u64 > want {
            file.set_len(want)?;
        }
        Ok(())
    }

    fn pwrite_zeroes(&self, bs: &Node, offset: u64, bytes: u64, may_unmap: bool) -> io::Result<()> {
        let flags = if may_unmap { BDRV_REQ_MAY_UNMAP } else { 0 };
        self.pwrite_zeroes_flags(bs, offset, bytes, flags)
    }

    /// `raw_do_pwrite_zeroes()`: `handle_aiocb_write_zeroes_unmap()` punches a hole first when
    /// unmapping is allowed, then `handle_aiocb_write_zeroes()`.
    fn pwrite_zeroes_flags(
        &self,
        _bs: &Node,
        offset: u64,
        bytes: u64,
        flags: u32,
    ) -> io::Result<()> {
        if self.kind == FileKind::HostCdrom {
            // host_cdrom has no .bdrv_co_pwrite_zeroes.
            return Err(errno(libc::ENOTSUP));
        }
        let file = self.file.read().unwrap();
        #[cfg(target_os = "linux")]
        if flags & BDRV_REQ_MAY_UNMAP != 0 {
            use rustix::fs::FallocateFlags;
            let punch = FallocateFlags::PUNCH_HOLE | FallocateFlags::KEEP_SIZE;
            match do_fallocate(&file, punch, offset, bytes) {
                Err(e)
                    if is_enotsup(&e)
                        || is_errno(&e, libc::EINVAL)
                        || is_errno(&e, libc::EBUSY) => {}
                r => return r,
            }
        }
        if self.kind == FileKind::HostDevice {
            self.write_zeroes_block(&file, offset, bytes, flags)
        } else {
            self.write_zeroes_file(&file, offset, bytes)
        }
    }

    fn supported_zero_flags(&self) -> u32 {
        BDRV_REQ_MAY_UNMAP | BDRV_REQ_NO_FALLBACK
    }

    fn has_pwrite_zeroes(&self) -> bool {
        self.kind != FileKind::HostCdrom
    }

    /// `handle_aiocb_discard()`: `BLKDISCARD` for a block device, `FALLOC_FL_PUNCH_HOLE` on
    /// Linux and `F_PUNCHHOLE` on macOS for a file. `F_PUNCHHOLE` wants whole file system
    /// blocks, so the range shrinks to those as the generic layer's `pdiscard_alignment`
    /// would make it.
    fn pdiscard(&self, _bs: &Node, offset: u64, bytes: u64) -> io::Result<()> {
        if self.kind == FileKind::HostCdrom || !self.has_discard.load(Ordering::Relaxed) {
            return Err(errno(libc::ENOTSUP));
        }
        let file = self.file.read().unwrap();
        let r = if self.kind == FileKind::HostDevice {
            #[cfg(target_os = "linux")]
            let r = sys::dev_ioctl(file.as_fd(), sys::DevIoctl::BlkDiscard(offset, bytes))
                .map(drop)
                .map_err(translate_err);
            #[cfg(not(target_os = "linux"))]
            let r = Err(errno(libc::ENOTSUP));
            r
        } else {
            #[cfg(target_os = "linux")]
            let r = {
                use rustix::fs::FallocateFlags;
                let punch = FallocateFlags::PUNCH_HOLE | FallocateFlags::KEEP_SIZE;
                do_fallocate(&file, punch, offset, bytes)
            };
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            let r = {
                let blk = file.metadata()?.blksize().max(1);
                let start = offset.next_multiple_of(blk);
                let end = (offset + bytes) - (offset + bytes) % blk;
                if end <= start {
                    return Ok(());
                }
                sys::punch_hole(file.as_fd(), start, end - start)
            };
            #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "ios")))]
            let r: io::Result<()> = {
                let _ = (offset, bytes);
                Err(errno(libc::ENOTSUP))
            };
            r
        };
        match r {
            Err(e) if is_enotsup(&e) => {
                self.has_discard.store(false, Ordering::Relaxed);
                Err(errno(libc::ENOTSUP))
            }
            r => r,
        }
    }

    /// `handle_aiocb_flush()`: `qemu_fdatasync()`.
    fn flush_to_disk(&self, _bs: &Node) -> io::Result<()> {
        self.file.read().unwrap().sync_data()
    }

    fn filename(&self) -> Option<String> {
        Some(self.filename.clone())
    }

    /// `raw_co_getlength()`.
    fn getlength(&self, _bs: &Node) -> io::Result<u64> {
        self.raw_getlength(&self.file.read().unwrap())
    }

    fn truncate(&self, _bs: &Node, len: u64) -> Result<()> {
        self.truncate_impl(len, false, PreallocMode::Off)
    }

    fn truncate_full(
        &self,
        _bs: &Node,
        offset: u64,
        exact: bool,
        prealloc: PreallocMode,
        _flags: u32,
    ) -> Result<()> {
        self.truncate_impl(offset, exact, prealloc)
    }

    /// When extending a regular file the host gives zeroes.
    fn supported_truncate_flags(&self) -> u32 {
        let regular = self.file.read().unwrap().metadata().is_ok_and(|m| m.file_type().is_file());
        if regular { BDRV_REQ_ZERO_WRITE } else { 0 }
    }

    /// `raw_co_get_allocated_file_size()`.
    fn get_allocated_file_size(&self, _bs: &Node) -> Option<io::Result<u64>> {
        Some(self.file.read().unwrap().metadata().map(|m| m.blocks() * 512))
    }

    /// `raw_refresh_limits()`, and `cdrom_refresh_limits()` for `host_cdrom`.
    fn refresh_limits(&self, _bs: &Node, bl: &mut BlockLimits) -> Result<()> {
        if self.kind == FileKind::HostCdrom {
            bl.has_variable_length = true;
        }
        let buf_align = self.buf_align.load(Ordering::Relaxed);
        bl.request_alignment = self.request_alignment.load(Ordering::Relaxed);
        bl.min_mem_alignment = buf_align;
        bl.opt_mem_alignment = buf_align.max(rustix::param::page_size());
        let file = self.file.read().unwrap();
        // The maximum transfers are best effort, so a failing fstat() is not an error.
        let Ok(md) = file.metadata() else { return Ok(()) };
        #[cfg(target_os = "macos")]
        if let Ok(s) = rustix::fs::fstatfs(&*file) {
            bl.opt_transfer = s.f_iosize as u32;
            bl.pdiscard_alignment = s.f_bsize;
        }
        #[cfg(target_os = "linux")]
        if md.file_type().is_block_device() {
            self.refresh_blk_limits(&file, &md, bl)?;
        }
        let _ = md;
        Ok(())
    }

    /// `hdev_probe_blocksizes()`: only DASD and zoned devices have block sizes worth passing
    /// to the guest, and this build knows neither, so it is always `ENOTSUP` as on QEMU hosts
    /// other than s390x.
    fn probe_blocksizes(&self, _bs: &Node) -> Option<io::Result<(u32, u32)>> {
        if self.kind != FileKind::HostDevice {
            return None;
        }
        Some(Err(errno(libc::ENOTSUP)))
    }

    fn is_inserted(&self, _bs: &Node) -> bool {
        #[cfg(target_os = "linux")]
        if self.kind == FileKind::HostCdrom {
            return self.cdrom_is_inserted();
        }
        true
    }

    /// `cdrom_co_eject()`.
    fn eject(&self, _bs: &Node, eject_flag: bool) {
        #[cfg(target_os = "linux")]
        if self.kind == FileKind::HostCdrom {
            let req =
                if eject_flag { sys::DevIoctl::CdromEject } else { sys::DevIoctl::CdromCloseTray };
            if let Err(e) = self.with_fd(|f| sys::dev_ioctl(f.as_fd(), req)) {
                // QEMU says CDROMEJECT for both.
                eprintln!("{}", Error::from_io("CDROMEJECT", e).message());
            }
        }
        let _ = eject_flag;
    }

    /// `cdrom_co_lock_medium()`. An error is expected when the distribution mounts the disc
    /// by itself and is ignored.
    fn lock_medium(&self, _bs: &Node, locked: bool) {
        #[cfg(target_os = "linux")]
        if self.kind == FileKind::HostCdrom {
            let _ =
                self.with_fd(|f| sys::dev_ioctl(f.as_fd(), sys::DevIoctl::CdromLockDoor(locked)));
        }
        let _ = locked;
    }

    /// `raw_check_perm()`, `raw_set_perm()` and `raw_abort_perm()` in one go, through
    /// `raw_handle_perm_lock()`.
    fn set_perm(&self, new_perm: u64, new_shared: u64) -> Result<()> {
        let Some(ops) = self.lock_ops else { return Ok(()) };
        let mut s = self.locks.lock().unwrap();
        let grows = (s.perm | new_perm) != s.perm || (s.shared_perm & new_shared) != s.shared_perm;
        if grows {
            // RAW_PL_PREPARE: lock the union of old and new, then look for others.
            let (perm, shared) = (s.perm, s.shared_perm);
            let r = self
                .apply_lock_bytes(ops, &mut s, perm | new_perm, !shared | !new_shared, false)
                .and_then(|()| {
                    self.check_lock_bytes(ops, new_perm, new_shared).map_err(|e| {
                        e.hint(format!("Is another process using the image [{}]?\n", self.filename))
                    })
                });
            if let Err(e) = r {
                // RAW_PL_ABORT.
                if let Err(e2) = self.apply_lock_bytes(ops, &mut s, perm, !shared, true) {
                    report::warn_report(e2.message());
                }
                return Err(e);
            }
        }
        // RAW_PL_COMMIT.
        if let Err(e) = self.apply_lock_bytes(ops, &mut s, new_perm, !new_shared, true) {
            report::warn_report(e.message());
        }
        s.perm = new_perm;
        s.shared_perm = new_shared;
        Ok(())
    }

    /// `raw_reopen_prepare()` with the descriptor part of `raw_check_perm()`: the options a
    /// reopen may change are taken out, the rest stay for the generic code to compare, and a
    /// new descriptor is opened when the read-only flag or `cache.direct` changes.
    fn reopen_prepare(&self, _bs: &Node, state: &mut ReopenState) -> Option<Result<()>> {
        Some(self.reopen_prepare_impl(state))
    }

    /// `raw_reopen_commit()`: switches to the new descriptor and moves the lock bytes to it.
    fn reopen_commit(&self, _bs: &Node, state: &mut ReopenState) {
        let Some(opaque) = state.opaque.take() else { return };
        let Ok(new) = opaque.downcast::<ReopenFd>() else { return };
        let ReopenFd { file, mode, alignment } = *new;
        let old = std::mem::replace(&mut *self.file.write().unwrap(), file);
        // Close the old descriptor before locking the new one: POSIX locks belong to the
        // process and closing any descriptor of the file would drop them again.
        drop(old);
        *self.mode.lock().unwrap() = mode;
        self.set_alignment(alignment);
        if let Some(ops) = self.lock_ops {
            let mut s = self.locks.lock().unwrap();
            let (perm_bits, shared_bits) = (s.locked_perm, s.locked_shared_perm);
            s.locked_perm = 0;
            s.locked_shared_perm = 0;
            if let Err(e) = self.apply_lock_bytes(ops, &mut s, perm_bits, shared_bits, false) {
                report::warn_report(e.message());
            }
        }
    }

    /// `raw_reopen_abort()`: the new descriptor, if any, is closed.
    fn reopen_abort(&self, _bs: &Node, state: &mut ReopenState) {
        state.opaque = None;
    }
}

impl FileDriver {
    fn reopen_prepare_impl(&self, state: &mut ReopenState) -> Result<()> {
        for name in ["drop-cache", "x-check-cache-dropped"] {
            if let Some(v) = state.options.remove(name) {
                if let Some(s) = v.as_str() {
                    qapi_bool_parse(name, s)?;
                } else if v.as_bool().is_none() {
                    return Err(Error::generic(format!(
                        "Parameter '{name}' expects 'on' or 'off'"
                    )));
                }
            }
        }
        if let Some(v) = state.options.remove("aio-max-batch") {
            if let Some(s) = v.as_str() {
                if s.parse::<u64>().is_err() {
                    return Err(Error::generic("Parameter 'aio-max-batch' expects a number"));
                }
            }
        }
        state.opaque = None;
        let mode = OpenMode { write: !state.flags.read_only, direct: state.flags.direct };
        if mode == *self.mode.lock().unwrap() {
            // The existing descriptor is fine.
            return Ok(());
        }
        let nonblock = self.kind == FileKind::HostCdrom;
        let file = open_file(&self.filename, mode.write, mode.direct, nonblock)
            .map_err(|e| e.into_error(&self.filename))?;
        if mode.write {
            check_hdev_writable(&file)
                .map_err(|e| Error::from_io("The device is not writable", e))?;
        }
        let alignment = probe_alignment(&file, mode.direct)?;
        state.opaque = Some(Box::new(ReopenFd { file, mode, alignment }));
        Ok(())
    }
}

/// The `file` protocol driver, `bdrv_file`.
pub(crate) static FILE: DriverDef = DriverDef::protocol("file", "file", file_open_node)
    .with_parse_filename(file_parse_filename)
    .with_needs_filename()
    .with_create(file_co_create)
    .with_create_opts(file_co_create_opts)
    .with_mutable_opts(MUTABLE_OPTS);

/// `mutable_opts` of the file-posix drivers.
pub(crate) const MUTABLE_OPTS: &[&str] = &["aio-max-batch", "drop-cache", "x-check-cache-dropped"];

/// `raw_parse_filename()`: the `file:` prefix is optional.
fn file_parse_filename(filename: &str, options: &mut QDict) -> Result<()> {
    parse_filename_strip_prefix(filename, "file:", options);
    Ok(())
}

/// `bdrv_parse_filename_strip_prefix()`: takes `prefix` off `filename` and stores the rest as
/// the `filename` option. When what is left would look like a protocol prefix of its own,
/// `./` goes in front.
pub(crate) fn parse_filename_strip_prefix(filename: &str, prefix: &str, options: &mut QDict) {
    if let Some(rest) = filename.strip_prefix(prefix) {
        if crate::drivers::protocol_prefix(rest).is_some() {
            options.put("filename", format!("./{rest}"));
        } else {
            options.put("filename", rest);
        }
    }
}

fn file_open_node(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::File(mut o) = opts else {
        unreachable!("file driver with other options")
    };
    // `-drive aio=native` reaches the file node as BDRV_O_NATIVE_AIO, which raw_open_common()
    // takes as the default for its own `aio` option.
    if o.aio.is_none() && args.ctx.inherit.native_aio {
        o.aio = Some(BlockdevAioOptions::Native);
    }
    let auto_read_only = args.flags.auto_read_only;
    let d = file_open(FileKind::File, &o, &mut args.flags, auto_read_only)?;
    args.meta.filename = o.filename.clone();
    Ok(Box::new(d))
}

/// `raw_co_create_opts()`: `qemu-img create` for a plain file.
fn file_co_create_opts(filename: &str, options: &mut QDict) -> Result<()> {
    let filename = filename.strip_prefix("file:").unwrap_or(filename);
    let take = |o: &mut QDict, k: &str| o.remove(k).and_then(|v| v.as_str().map(str::to_owned));
    let size = match take(options, "size") {
        Some(v) => parse_option_size("size", &v)?,
        None => 0,
    };
    let size = size.div_ceil(BDRV_SECTOR_SIZE) * BDRV_SECTOR_SIZE;
    let extent_size_hint = match take(options, "extent_size_hint") {
        Some(v) => Some(parse_option_size("extent_size_hint", &v)?),
        None => None,
    };
    let nocow = match take(options, "nocow") {
        Some(v) => qapi_bool_parse("nocow", &v)?,
        None => false,
    };
    let preallocation = match take(options, "preallocation") {
        Some(v) => PreallocMode::from_name(&v)
            .ok_or_else(|| Error::generic(format!("invalid parameter value: {v}")))?,
        None => PreallocMode::Off,
    };
    raw_co_create(BlockdevCreateOptionsFile {
        filename: filename.to_owned(),
        size,
        preallocation: Some(preallocation),
        nocow: Some(nocow),
        extent_size_hint,
    })
}

/// `raw_co_create()` as `.bdrv_co_create`: `blockdev-create` with `driver: file`.
fn file_co_create(_graph: &BlockGraph, options: BlockdevCreateOptionsU) -> Result<()> {
    let BlockdevCreateOptionsU::File(o) = options else {
        unreachable!("file driver with other create options")
    };
    raw_co_create(o)
}

/// `raw_co_create()`: creates or empties the file, takes the write and resize locks, and
/// grows it to its size with the preallocation asked for.
///
/// Differences from QEMU: `nocow` (`FS_IOC_SETFLAGS`) and the extent size hint
/// (`FS_IOC_FSSETXATTR`) are best-effort optimisations in QEMU whose failure is ignored; they are
/// not applied here at all.
fn raw_co_create(o: BlockdevCreateOptionsFile) -> Result<()> {
    let extent_size_hint = o.extent_size_hint.unwrap_or(1 << 20);
    if extent_size_hint > u64::from(u32::MAX) {
        return Err(Error::generic("Extent size hint is too large"));
    }
    let _ = o.nocow;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        // Emptied only once the locks are taken.
        .truncate(false)
        .mode(0o644)
        .open(&o.filename)
        .map_err(|e| Error::from_io(format!("Could not create '{}'", o.filename), e))?;

    // Take the write and resize permissions and share everything but resizing: after this
    // function the file must have the size asked for.
    let perm = BLK_PERM_WRITE | BLK_PERM_RESIZE;
    let shared = BLK_PERM_ALL & !BLK_PERM_RESIZE;
    let ops = if !sys::has_locks() {
        None
    } else if sys::has_ofd_lock() {
        Some(LockOps::Ofd)
    } else {
        Some(LockOps::Posix)
    };
    let mut locks =
        LockState { perm: 0, shared_perm: BLK_PERM_ALL, locked_perm: 0, locked_shared_perm: 0 };
    let r = (|| {
        if let Some(ops) = ops {
            apply_lock_bytes(file.as_fd(), ops, &mut locks, perm, !shared & BLK_PERM_ALL, false)?;
            check_lock_bytes(file.as_fd(), ops, perm, shared).map_err(|e| {
                e.hint(format!("Is another process using the image [{}]?\n", o.filename))
            })?;
        }
        regular_truncate(&file, 0, PreallocMode::Off)?;
        regular_truncate(&file, o.size, o.preallocation.unwrap_or_default())
    })();
    if let Some(ops) = ops {
        if let Err(e) = apply_lock_bytes(file.as_fd(), ops, &mut locks, 0, 0, true) {
            report::warn_report(e.message());
        }
    }
    r
}

/// `handle_aiocb_truncate()`: resizes `file` to `offset` with `prealloc`.
fn regular_truncate(file: &File, offset: u64, prealloc: PreallocMode) -> Result<()> {
    let current = file.metadata().map_err(|e| Error::from_io("Could not stat file", e))?.len();
    if current > offset && prealloc != PreallocMode::Off {
        return Err(Error::generic("Cannot use preallocation for shrinking files"));
    }
    let r = match prealloc {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        PreallocMode::Falloc => {
            if offset == current {
                return Ok(());
            }
            rustix::fs::fallocate(
                file,
                rustix::fs::FallocateFlags::empty(),
                current,
                offset - current,
            )
            .map_err(|e| Error::from_io("Could not preallocate new data", e.into()))
        }
        PreallocMode::Full => (|| {
            file.set_len(offset).map_err(|e| Error::from_io("Could not resize file", e))?;
            let zeroes = vec![0u8; 65536];
            let mut pos = current;
            while pos < offset {
                let n = (offset - pos).min(zeroes.len() as u64) as usize;
                file.write_all_at(&zeroes[..n], pos)
                    .map_err(|e| Error::from_io("Could not write zeros for preallocation", e))?;
                pos += n as u64;
            }
            file.sync_all().map_err(|e| Error::from_io("Could not flush file to disk", e))
        })(),
        PreallocMode::Off => {
            return file.set_len(offset).map_err(|e| Error::from_io("Could not resize file", e));
        }
        other => {
            return Err(Error::generic(format!(
                "Unsupported preallocation mode: {}",
                other.as_str()
            )));
        }
    };
    if r.is_err() {
        if let Err(e) = file.set_len(current) {
            report::error_report(&format!("Failed to restore old file length: {e}"));
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::OpenCtx;
    use std::sync::Arc;

    fn open(path: &str, read_only: bool) -> (BlockGraph, Arc<Node>) {
        let g = BlockGraph::new();
        let mut o = QDict::new();
        o.put("driver", "file");
        o.put("filename", path);
        o.put("read-only", read_only);
        let ctx = OpenCtx { protocol: true, ..OpenCtx::default() };
        let (bs, _, _) = g.open_nodes_qdict(None, o, ctx).unwrap();
        (g, bs)
    }

    fn temp_file(name: &str, len: u64) -> String {
        let path = std::env::temp_dir().join(format!("ruvm-file-{}-{name}", std::process::id()));
        let f = File::create(&path).unwrap();
        f.set_len(len).unwrap();
        path.to_str().unwrap().to_owned()
    }

    #[test]
    fn reopen_read_write_swaps_fd() {
        let path = temp_file("reopen", 65536);
        let (_g, bs) = open(&path, true);
        assert!(bs.read_only());
        assert!(bs.pwrite(0, b"x").is_err());
        bs.reopen_set_read_only(false).unwrap();
        assert!(!bs.read_only());
        bs.pwrite(0, b"hello").unwrap();
        bs.reopen_set_read_only(true).unwrap();
        let mut buf = [0u8; 5];
        bs.pread(0, &mut buf).unwrap();
        assert_eq!(&buf, b"hello");
        drop(bs);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn reopen_takes_mutable_opts() {
        let path = temp_file("mutable", 4096);
        let (_g, bs) = open(&path, false);
        let mut o = QDict::new();
        o.put("drop-cache", "off");
        o.put("x-check-cache-dropped", "on");
        o.put("aio-max-batch", "8");
        bs.reopen(o, true).unwrap();
        let mut o = QDict::new();
        o.put("drop-cache", "maybe");
        assert_eq!(
            bs.reopen(o, true).unwrap_err().message(),
            "Parameter 'drop-cache' expects 'on' or 'off'"
        );
        drop(bs);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn allocated_size_zeroes_and_discard() {
        let path = temp_file("alloc", 0);
        let (_g, bs) = open(&path, false);
        let data = vec![0xa5u8; 1 << 20];
        bs.pwrite(0, &data).unwrap();
        bs.flush().unwrap();
        let alloc = bs.allocated_file_size().unwrap();
        assert!(alloc >= 1 << 20, "allocated {alloc}");
        assert_eq!(alloc % 512, 0);

        bs.pwrite_zeroes_flags(4096, 8192, 0).unwrap();
        let mut buf = vec![1u8; 8192];
        bs.pread(4096, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0));

        // Discard may be a no-op, but must not fail or grow the file.
        let _ = bs.pdiscard(65536, 65536);
        assert_eq!(bs.getlength().unwrap(), 1 << 20);
        drop(bs);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn truncate_regular_file() {
        let path = temp_file("trunc", 4096);
        let (_g, bs) = open(&path, false);
        bs.truncate_full(8192, true, PreallocMode::Off, 0).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 8192);
        drop(bs);
        std::fs::remove_file(&path).unwrap();
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn device_cannot_grow() {
        let g = BlockGraph::new();
        let mut o = QDict::new();
        o.put("driver", "host_device");
        o.put("filename", "/dev/null");
        o.put("read-only", false);
        let ctx = OpenCtx { protocol: true, ..OpenCtx::default() };
        let (bs, _, _) = g.open_nodes_qdict(None, o, ctx).unwrap();
        let e = bs.truncate_full(4096, true, PreallocMode::Off, 0).unwrap_err();
        assert_eq!(e.message(), "Cannot resize device files");
        let e = bs.truncate_full(4096, false, PreallocMode::Off, 0).unwrap_err();
        assert_eq!(e.message(), "Cannot grow device files");
        let e = bs.truncate_full(0, false, PreallocMode::Full, 0).unwrap_err();
        assert_eq!(e.message(), "Preallocation mode 'full' unsupported for this non-regular file");
    }
}
