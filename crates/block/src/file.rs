// SPDX-License-Identifier: GPL-2.0-or-later

//! The `file` protocol driver from block/file-posix.c, for regular files.
//!
//! Only `aio=threads` is implemented, as plain positioned reads and writes on the caller's thread.
//! `aio=native` is accepted where QEMU accepts it (Linux, with `cache.direct=on`) and then runs
//! the same way, which is what QEMU does too when it cannot set up linux-aio. `aio=io_uring` is
//! not in this build's QAPI schema, like a QEMU built without liburing, so the option parser
//! already refuses it.
//!
//! `host_device` and `host_cdrom`, the block and character device variants in the same C file,
//! share the open path in QEMU with a different file type check. They are not wired up yet.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsFd;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use ruvm_base::{Error, Result, report};
use ruvm_qapi::types::{BlockdevAioOptions, BlockdevOptionsFile, OnOffAuto};

use crate::node::{Driver, Node, NodeFlags, errno, is_enotsup};
use crate::perm::{BLK_PERM_ALL, perm_names};
use crate::sys::{self, LockOps};

/// `RAW_LOCK_PERM_BASE`. libvirt uses byte 0, QEMU leaves room for it to grow.
pub(crate) const RAW_LOCK_PERM_BASE: u64 = 100;
/// `RAW_LOCK_SHARED_BASE`.
pub(crate) const RAW_LOCK_SHARED_BASE: u64 = 200;

/// The request alignment used with `O_DIRECT`. QEMU probes the smallest that works, 4096 always
/// works on the file systems that support `O_DIRECT` at all.
#[cfg(any(target_os = "linux", target_os = "android"))]
const DIRECT_ALIGN: u64 = 4096;

/// The lock bytes this file holds and the permissions they stand for, the lock part of
/// `BDRVRawState`.
#[derive(Debug)]
struct LockState {
    perm: u64,
    shared_perm: u64,
    locked_perm: u64,
    locked_shared_perm: u64,
}

/// `BDRVRawState` for `FTYPE_FILE`.
pub(crate) struct FileDriver {
    file: File,
    filename: String,
    /// `None` when `use_lock` is false.
    lock_ops: Option<LockOps>,
    locks: Mutex<LockState>,
    /// The alignment requests must have, 1 unless the file is open with `O_DIRECT`.
    align: u64,
    /// Serialises read-modify-write cycles for unaligned `O_DIRECT` writes.
    rmw: Mutex<()>,
    has_discard: AtomicBool,
    has_write_zeroes: AtomicBool,
}

/// Opens a `file` node, `raw_open()` and `raw_open_common()`. `flags.read_only` is cleared
/// or set here: with `auto_read_only` a file that cannot be opened for writing is opened
/// read-only instead, as `bdrv_apply_auto_read_only()` allows.
pub(crate) fn file_open(
    o: &BlockdevOptionsFile,
    flags: &mut NodeFlags,
    auto_read_only: bool,
) -> Result<FileDriver> {
    let filename = o.filename.clone();
    let aio = o.aio.unwrap_or_default();
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

    let file = match open_file(&filename, !flags.read_only, flags.direct) {
        Ok(f) => f,
        Err(e) if auto_read_only && !flags.read_only && is_read_only_error(&e) => {
            flags.read_only = true;
            open_file(&filename, false, flags.direct).map_err(|e| e.into_error(&filename))?
        }
        Err(e) => return Err(e.into_error(&filename)),
    };

    if aio == BlockdevAioOptions::Native {
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
    if !md.file_type().is_file() {
        return Err(Error::generic(format!(
            "'file' driver requires '{filename}' to be a regular file"
        )));
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    let align = if flags.direct { DIRECT_ALIGN } else { 1 };
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    let align = 1;

    Ok(FileDriver {
        file,
        filename,
        lock_ops,
        locks: Mutex::new(LockState {
            perm: 0,
            shared_perm: BLK_PERM_ALL,
            locked_perm: 0,
            locked_shared_perm: 0,
        }),
        align,
        rmw: Mutex::new(()),
        has_discard: AtomicBool::new(true),
        has_write_zeroes: AtomicBool::new(true),
    })
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
/// file-posix uses `O_DSYNC` for `cache.direct=on`, and so does this.
fn open_file(filename: &str, write: bool, direct: bool) -> std::result::Result<File, OpenError> {
    let mut oo = OpenOptions::new();
    oo.read(true).write(write);
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let direct_flag = libc::O_DIRECT;
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    let direct_flag = libc::O_DSYNC;
    if direct {
        oo.custom_flags(direct_flag);
    }
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

/// `raw_lock_error_setg_errno()`: lock contention gets no strerror text.
fn lock_error(e: io::Error, msg: String) -> Error {
    match e.raw_os_error() {
        Some(n) if n == libc::EAGAIN || n == libc::EACCES => Error::with_cause(msg, e),
        _ => Error::from_io(msg, e),
    }
}

impl FileDriver {
    /// `raw_apply_lock_bytes()` with the driver state: lock the bytes for `perm_lock_bits` and
    /// `shared_perm_lock_bits` (the permissions that are not shared), and with `unlock` let go
    /// of the bytes that are no longer needed.
    fn apply_lock_bytes(
        &self,
        ops: LockOps,
        s: &mut LockState,
        perm_lock_bits: u64,
        shared_perm_lock_bits: u64,
        unlock: bool,
    ) -> Result<()> {
        let fd = self.file.as_fd();
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
    fn check_lock_bytes(&self, ops: LockOps, perm: u64, shared_perm: u64) -> Result<()> {
        let fd = self.file.as_fd();
        for i in 0..4 {
            let p = 1u64 << i;
            if perm & p != 0 {
                sys::lock_fd_test(fd, ops, RAW_LOCK_SHARED_BASE + i, true).map_err(|e| {
                    lock_error(e, format!("Failed to get \"{}\" lock", perm_names(p)))
                })?;
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

    fn fd_len(&self) -> io::Result<u64> {
        Ok(self.file.metadata()?.len())
    }

    /// Reads at `offset` until `buf` is full, padding with zeroes past the end of the file as
    /// `handle_aiocb_rw()` does.
    fn read_full(&self, mut offset: u64, mut buf: &mut [u8]) -> io::Result<()> {
        let mut progress = false;
        while !buf.is_empty() {
            match self.file.read_at(buf, offset) {
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
                Err(e) if progress && self.align > 1 && e.raw_os_error() == Some(libc::EINVAL) => {
                    buf.fill(0);
                    return Ok(());
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// The aligned span covering `offset..offset + len`.
    fn aligned_span(&self, offset: u64, len: u64) -> (u64, usize) {
        let start = offset - offset % self.align;
        let end = (offset + len).next_multiple_of(self.align);
        (start, (end - start) as usize)
    }

    /// A buffer of `len` bytes whose start is aligned for `O_DIRECT`, as a vector and the
    /// offset of the aligned part in it.
    fn bounce(&self, len: usize) -> (Vec<u8>, usize) {
        let align = self.align as usize;
        let v = vec![0u8; len + align];
        let skip = v.as_ptr().align_offset(align);
        (v, skip)
    }
}

impl Driver for FileDriver {
    fn pread(&self, _bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        if self.align == 1 {
            return self.read_full(offset, buf);
        }
        let (start, len) = self.aligned_span(offset, buf.len() as u64);
        let (mut v, skip) = self.bounce(len);
        let aligned = &mut v[skip..skip + len];
        self.read_full(start, aligned)?;
        let head = (offset - start) as usize;
        buf.copy_from_slice(&aligned[head..head + buf.len()]);
        Ok(())
    }

    fn pwrite(&self, _bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        if self.align == 1 {
            return self.file.write_all_at(buf, offset);
        }
        let (start, len) = self.aligned_span(offset, buf.len() as u64);
        let (mut v, skip) = self.bounce(len);
        let aligned = &mut v[skip..skip + len];
        let _guard = self.rmw.lock().unwrap();
        let head = (offset - start) as usize;
        if head != 0 || len != buf.len() {
            self.read_full(start, aligned)?;
        }
        aligned[head..head + buf.len()].copy_from_slice(buf);
        // Do not grow the file past what was asked for: a write that ends inside the last
        // aligned block and past the end of the file goes out whole and is cut back.
        let old_len = self.fd_len()?;
        self.file.write_all_at(aligned, start)?;
        let want = old_len.max(offset + buf.len() as u64);
        if start + len as u64 > want {
            self.file.set_len(want)?;
        }
        Ok(())
    }

    /// `handle_aiocb_write_zeroes()` and `handle_aiocb_write_zeroes_unmap()`: punch a hole when
    /// unmapping is allowed, otherwise `FALLOC_FL_ZERO_RANGE`. Elsewhere the generic layer
    /// writes zeroes.
    fn pwrite_zeroes(&self, bs: &Node, offset: u64, bytes: u64, may_unmap: bool) -> io::Result<()> {
        if !self.has_write_zeroes.load(Ordering::Relaxed) {
            return Err(errno(libc::ENOTSUP));
        }
        let _ = (bs, may_unmap);
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            use rustix::fs::{FallocateFlags, fallocate};
            if may_unmap {
                let r = fallocate(
                    &self.file,
                    FallocateFlags::PUNCH_HOLE | FallocateFlags::KEEP_SIZE,
                    offset,
                    bytes,
                );
                if r.is_ok() {
                    return Ok(());
                }
            }
            let r = fallocate(&self.file, FallocateFlags::ZERO_RANGE, offset, bytes)
                .map_err(io::Error::from);
            match r {
                Err(e) if is_enotsup(&e) || e.raw_os_error() == Some(libc::ENOSYS) => {
                    self.has_write_zeroes.store(false, Ordering::Relaxed);
                    Err(errno(libc::ENOTSUP))
                }
                r => r,
            }
        }
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        {
            let _ = (offset, bytes);
            self.has_write_zeroes.store(false, Ordering::Relaxed);
            Err(errno(libc::ENOTSUP))
        }
    }

    /// `handle_aiocb_discard()`: `FALLOC_FL_PUNCH_HOLE` on Linux, `F_PUNCHHOLE` on macOS, which
    /// wants whole file system blocks, so the range shrinks to those as the generic layer's
    /// `pdiscard_alignment` would make it.
    fn pdiscard(&self, _bs: &Node, offset: u64, bytes: u64) -> io::Result<()> {
        if !self.has_discard.load(Ordering::Relaxed) {
            return Err(errno(libc::ENOTSUP));
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        let r = {
            use rustix::fs::{FallocateFlags, fallocate};
            fallocate(
                &self.file,
                FallocateFlags::PUNCH_HOLE | FallocateFlags::KEEP_SIZE,
                offset,
                bytes,
            )
            .map_err(io::Error::from)
        };
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let r = {
            let blk = std::os::unix::fs::MetadataExt::blksize(&self.file.metadata()?).max(1);
            let start = offset.next_multiple_of(blk);
            let end = (offset + bytes) - (offset + bytes) % blk;
            if end <= start {
                return Ok(());
            }
            sys::punch_hole(self.file.as_fd(), start, end - start)
        };
        #[cfg(not(any(
            target_os = "linux",
            target_os = "android",
            target_os = "macos",
            target_os = "ios"
        )))]
        let r: io::Result<()> = {
            let _ = (offset, bytes);
            Err(errno(libc::ENOTSUP))
        };
        match r {
            Err(e) if is_enotsup(&e) || e.raw_os_error() == Some(libc::ENOSYS) => {
                self.has_discard.store(false, Ordering::Relaxed);
                Err(errno(libc::ENOTSUP))
            }
            r => r,
        }
    }

    /// `handle_aiocb_flush()`: `qemu_fdatasync()`.
    fn flush_to_disk(&self, _bs: &Node) -> io::Result<()> {
        self.file.sync_data()
    }

    fn filename(&self) -> Option<String> {
        Some(self.filename.clone())
    }

    /// `raw_co_getlength()`: the size of the file.
    fn getlength(&self, _bs: &Node) -> io::Result<u64> {
        self.fd_len()
    }

    /// `raw_co_truncate()` for a regular file: `ftruncate()`.
    fn truncate(&self, _bs: &Node, len: u64) -> Result<()> {
        self.file.set_len(len).map_err(|e| Error::from_io("Failed to resize file", e))
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
}
