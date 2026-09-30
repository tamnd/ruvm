// SPDX-License-Identifier: GPL-2.0-or-later

//! The FUSE export driver, `blk_exp_fuse` of block/export/fuse.c: the node appears as a single
//! regular file mounted over `mountpoint`.
//!
//! The export speaks the FUSE kernel protocol on `/dev/fuse` itself, as QEMU does since it stopped
//! using libfuse's request loop. What QEMU still takes from libfuse, mounting and unmounting, is
//! done here the way libfuse's lib/mount.c does it: a direct `mount(2)` of type `fuse` with the
//! `fd=`, `rootmode=`, `user_id=` and `group_id=` options when that is allowed, and otherwise
//! `fusermount3`, which passes the `/dev/fuse` descriptor back over the socket named by
//! `_FUSE_COMMFD`. Unmounting is `umount2(MNT_DETACH)` or `fusermount3 -u -q -z`.
//!
//! Differences from QEMU:
//!
//! - Requests are served by a thread of the export's own instead of an fd handler in the
//!   export's AioContext, and there is always one queue: the multithreaded queues QEMU sets up
//!   with `FUSE_DEV_IOC_CLONE` are not implemented.
//! - Draining the backend does not pause the export; the thread keeps serving requests.
//! - `FUSE_LSEEK` gets `ENOSYS`, as in a QEMU built without `CONFIG_FUSE_LSEEK`, because the
//!   block status of the backend's root node is not reachable from here. The kernel then treats
//!   the whole file as data.
//! - `st_blocks` is always the length in 512 byte units, as QEMU reports when the allocated file
//!   size is unknown, and `st_blksize` and the `statfs` block size are 512: the request alignment
//!   and optimal transfer size of the root node are not reachable from here.
//! - Truncating goes through `blk_truncate()` without `BDRV_REQ_ZERO_WRITE`, and `fallocate()`
//!   with mode 0 grows the image without `PREALLOC_MODE_FALLOC`. Punching a hole writes zeroes
//!   with `BDRV_REQ_MAY_UNMAP` only, without `BDRV_REQ_NO_FALLBACK`.
//! - Block layer errors that carry no errno are passed to the kernel as `EIO`.
//! - Failing to create the stop pipe or the thread gives "Failed to create pipe" or "Failed to
//!   create thread" with the strerror text, which QEMU has no equivalent of.
//! - The messages libfuse prints when mounting fails (`fuse: ...`) are reproduced for the cases
//!   handled here; the QMP error stays QEMU's "Failed to mount FUSE session to export".

use std::io;
use std::mem::MaybeUninit;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

use rustix::event::{PollFd, PollFlags, poll};
use rustix::fs::{Mode, OFlags};
use rustix::io::{Errno, FdFlags, IoSlice, IoSliceMut};
use rustix::mount::{MountFlags, UnmountFlags};
use rustix::net::{
    AddressFamily, RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, SocketFlags, SocketType,
};
use ruvm_base::error::strerror;
use ruvm_base::{Error, Result, error_report, warn_report};
use ruvm_block::{BLK_PERM_RESIZE, BlockBackend};
use ruvm_qapi::types::FuseExportAllowOther;

use super::{ExportArgs, ExportDriver};

/// `FUSE_MAX_READ_BYTES`: `MIN(BDRV_REQUEST_MAX_BYTES, 1 MiB)`.
const FUSE_MAX_READ_BYTES: u32 = 1024 * 1024;
/// `FUSE_MAX_WRITE_BYTES`.
const FUSE_MAX_WRITE_BYTES: u32 = 64 * 1024;
/// `BDRV_REQUEST_MAX_BYTES`.
const BDRV_REQUEST_MAX_BYTES: u64 = (i32::MAX as u64) & !511;

/// `FUSE_KERNEL_VERSION`.
const FUSE_KERNEL_VERSION: u32 = 7;
/// `FUSE_KERNEL_MINOR_VERSION` of QEMU 11.1's copy of linux/fuse.h.
const FUSE_KERNEL_MINOR_VERSION: u32 = 45;

const FUSE_LOOKUP: u32 = 1;
const FUSE_FORGET: u32 = 2;
const FUSE_GETATTR: u32 = 3;
const FUSE_SETATTR: u32 = 4;
const FUSE_OPEN: u32 = 14;
const FUSE_READ: u32 = 15;
const FUSE_WRITE: u32 = 16;
const FUSE_STATFS: u32 = 17;
const FUSE_RELEASE: u32 = 18;
const FUSE_FSYNC: u32 = 20;
const FUSE_FLUSH: u32 = 25;
const FUSE_INIT: u32 = 26;
const FUSE_DESTROY: u32 = 38;
const FUSE_BATCH_FORGET: u32 = 42;
const FUSE_FALLOCATE: u32 = 43;

/// `sizeof(struct fuse_in_header)`.
const IN_HEADER_LEN: usize = 40;
/// `sizeof(struct fuse_out_header)`.
const OUT_HEADER_LEN: usize = 16;
/// The `fuse_init_in` of protocol versions before 7.36.
const OLD_INIT_IN_LEN: usize = 16;
const INIT_IN_LEN: usize = 64;
const OPEN_IN_LEN: usize = 8;
const SETATTR_IN_LEN: usize = 88;
const READ_IN_LEN: usize = 40;
const WRITE_IN_LEN: usize = 40;
const FALLOCATE_IN_LEN: usize = 32;
/// `sizeof(struct fuse_init_out)`.
const INIT_OUT_LEN: usize = 64;
/// `FUSE_COMPAT_22_INIT_OUT_SIZE`.
const FUSE_COMPAT_22_INIT_OUT_SIZE: usize = 24;
/// `sizeof(struct fuse_attr_out)`.
const ATTR_OUT_LEN: usize = 104;
/// `sizeof(struct fuse_statfs_out)`.
const STATFS_OUT_LEN: usize = 80;
/// `sizeof(struct fuse_open_out)`.
const OPEN_OUT_LEN: usize = 16;
/// `sizeof(struct fuse_write_out)`.
const WRITE_OUT_LEN: usize = 8;

/// `FuseRequestInHeaderBuf.head` plus the WRITE data buffer: what one read of the device takes.
const REQ_BUF_LEN: usize = IN_HEADER_LEN + WRITE_IN_LEN + FUSE_MAX_WRITE_BYTES as usize;

const FUSE_ASYNC_READ: u32 = 1 << 0;
const FUSE_ATOMIC_O_TRUNC: u32 = 1 << 3;
const FUSE_ASYNC_DIO: u32 = 1 << 15;
const FUSE_INIT_EXT: u32 = 1 << 30;
/// `FUSE_DIRECT_IO_ALLOW_MMAP >> 32`.
const FUSE_DIRECT_IO_ALLOW_MMAP_FLAGS2: u32 = 1 << 4;

const FOPEN_DIRECT_IO: u32 = 1 << 0;
const FOPEN_PARALLEL_DIRECT_WRITES: u32 = 1 << 6;

const FATTR_MODE: u32 = 1 << 0;
const FATTR_UID: u32 = 1 << 1;
const FATTR_GID: u32 = 1 << 2;
const FATTR_SIZE: u32 = 1 << 3;
const FATTR_FH: u32 = 1 << 6;
const FATTR_LOCKOWNER: u32 = 1 << 9;
const FATTR_KILL_SUIDGID: u32 = 1 << 11;

const FALLOC_FL_KEEP_SIZE: u32 = 0x01;
const FALLOC_FL_PUNCH_HOLE: u32 = 0x02;
const FALLOC_FL_ZERO_RANGE: u32 = 0x10;

const S_IFREG: u32 = 0o100000;
const S_IRUSR: u32 = 0o400;
const S_IWUSR: u32 = 0o200;
const S_IWGRP: u32 = 0o020;
const S_IWOTH: u32 = 0o002;
const S_IRWXG: u32 = 0o070;
const S_IRWXO: u32 = 0o007;

/// The `fusermount3` program that mounts and unmounts for unprivileged users.
const FUSERMOUNT: &str = "fusermount3";

/// `exports`: the mount points in use, to refuse the same path string twice.
static EXPORTS: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn rd_u32(b: &[u8], off: usize) -> u32 {
    let mut a = [0u8; 4];
    a.copy_from_slice(&b[off..off + 4]);
    u32::from_ne_bytes(a)
}

fn rd_u64(b: &[u8], off: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[off..off + 8]);
    u64::from_ne_bytes(a)
}

fn wr_u16(b: &mut [u8], off: usize, v: u16) {
    b[off..off + 2].copy_from_slice(&v.to_ne_bytes());
}

fn wr_u32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_ne_bytes());
}

fn wr_u64(b: &mut [u8], off: usize, v: u64) {
    b[off..off + 8].copy_from_slice(&v.to_ne_bytes());
}

/// The fields of `struct fuse_in_header` the export looks at.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct InHeader {
    len: u32,
    opcode: u32,
    unique: u64,
}

impl InHeader {
    /// Decodes the header at the start of `b`, which must hold [`IN_HEADER_LEN`] bytes.
    fn parse(b: &[u8]) -> Self {
        InHeader { len: rd_u32(b, 0), opcode: rd_u32(b, 4), unique: rd_u64(b, 8) }
    }

    /// The header as the kernel sends it, for node 1 and a made up caller.
    #[cfg(test)]
    fn encode(&self) -> [u8; IN_HEADER_LEN] {
        let mut b = [0u8; IN_HEADER_LEN];
        wr_u32(&mut b, 0, self.len);
        wr_u32(&mut b, 4, self.opcode);
        wr_u64(&mut b, 8, self.unique);
        wr_u64(&mut b, 16, 1);
        wr_u32(&mut b, 24, 1000);
        wr_u32(&mut b, 28, 1000);
        wr_u32(&mut b, 32, 42);
        b
    }
}

/// `struct fuse_out_header`, with `error` a negative errno or 0.
fn out_header(len: usize, error: i32, unique: u64) -> [u8; OUT_HEADER_LEN] {
    let mut b = [0u8; OUT_HEADER_LEN];
    wr_u32(&mut b, 0, len as u32);
    wr_u32(&mut b, 4, error as u32);
    wr_u64(&mut b, 8, unique);
    b
}

/// `struct fuse_init_in`, with `flags2` zero for the old, shorter structure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct InitIn {
    major: u32,
    minor: u32,
    max_readahead: u32,
    flags: u32,
    flags2: u32,
}

impl InitIn {
    /// Decodes the body of a `FUSE_INIT` request that holds at least [`OLD_INIT_IN_LEN`] bytes,
    /// and [`INIT_IN_LEN`] bytes when the version says the structure is the new one.
    fn parse(b: &[u8]) -> Self {
        let mut i = InitIn {
            major: rd_u32(b, 0),
            minor: rd_u32(b, 4),
            max_readahead: rd_u32(b, 8),
            flags: rd_u32(b, 12),
            flags2: 0,
        };
        if !using_old_fuse_init_in(i.major, i.minor) {
            i.flags2 = rd_u32(b, 16);
        }
        i
    }
}

/// `using_old_fuse_init_in()`: whether the kernel uses the `fuse_init_in` from before 7.36.
fn using_old_fuse_init_in(major: u32, minor: u32) -> bool {
    major < 7 || (major == 7 && minor < 36)
}

/// `struct fuse_init_out`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct InitOut {
    major: u32,
    minor: u32,
    max_readahead: u32,
    flags: u32,
    max_background: u16,
    congestion_threshold: u16,
    max_write: u32,
    time_gran: u32,
    max_pages: u16,
    map_alignment: u16,
    flags2: u32,
}

impl InitOut {
    /// The structure as the kernel of the negotiated minor version expects it: before 7.23 it
    /// ends after `max_write`.
    fn encode(&self) -> Vec<u8> {
        let mut b = vec![0u8; INIT_OUT_LEN];
        wr_u32(&mut b, 0, self.major);
        wr_u32(&mut b, 4, self.minor);
        wr_u32(&mut b, 8, self.max_readahead);
        wr_u32(&mut b, 12, self.flags);
        wr_u16(&mut b, 16, self.max_background);
        wr_u16(&mut b, 18, self.congestion_threshold);
        wr_u32(&mut b, 20, self.max_write);
        wr_u32(&mut b, 24, self.time_gran);
        wr_u16(&mut b, 28, self.max_pages);
        wr_u16(&mut b, 30, self.map_alignment);
        wr_u32(&mut b, 32, self.flags2);
        if self.minor < 23 {
            b.truncate(FUSE_COMPAT_22_INIT_OUT_SIZE);
        }
        b
    }
}

/// The fields of `struct fuse_setattr_in` the export looks at.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SetattrIn {
    valid: u32,
    size: u64,
    mode: u32,
    uid: u32,
    gid: u32,
}

impl SetattrIn {
    fn parse(b: &[u8]) -> Self {
        SetattrIn {
            valid: rd_u32(b, 0),
            size: rd_u64(b, 16),
            mode: rd_u32(b, 68),
            uid: rd_u32(b, 76),
            gid: rd_u32(b, 80),
        }
    }
}

/// `struct fuse_attr_out`, with `attr_valid` 1 second as QEMU sets it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Attr {
    ino: u64,
    size: u64,
    blocks: u64,
    time: u64,
    mode: u32,
    nlink: u32,
    uid: u32,
    gid: u32,
    blksize: u32,
}

impl Attr {
    fn encode(&self) -> Vec<u8> {
        let mut b = vec![0u8; ATTR_OUT_LEN];
        wr_u64(&mut b, 0, 1);
        let a = 16;
        wr_u64(&mut b, a, self.ino);
        wr_u64(&mut b, a + 8, self.size);
        wr_u64(&mut b, a + 16, self.blocks);
        wr_u64(&mut b, a + 24, self.time);
        wr_u64(&mut b, a + 32, self.time);
        wr_u64(&mut b, a + 40, self.time);
        wr_u32(&mut b, a + 60, self.mode);
        wr_u32(&mut b, a + 64, self.nlink);
        wr_u32(&mut b, a + 68, self.uid);
        wr_u32(&mut b, a + 72, self.gid);
        wr_u32(&mut b, a + 80, self.blksize);
        b
    }
}

/// `req_op_hdr_len()`: the length of the operation's own header, `None` for `-ENOSYS`.
fn req_op_hdr_len(opcode: u32) -> Option<usize> {
    match opcode {
        FUSE_INIT => Some(INIT_IN_LEN),
        FUSE_OPEN => Some(OPEN_IN_LEN),
        FUSE_SETATTR => Some(SETATTR_IN_LEN),
        FUSE_READ => Some(READ_IN_LEN),
        FUSE_WRITE => Some(WRITE_IN_LEN),
        FUSE_FALLOCATE => Some(FALLOCATE_IN_LEN),
        FUSE_DESTROY | FUSE_STATFS | FUSE_RELEASE | FUSE_LOOKUP | FUSE_FORGET
        | FUSE_BATCH_FORGET | FUSE_GETATTR | FUSE_FSYNC | FUSE_FLUSH => Some(0),
        _ => None,
    }
}

/// The errno for a failed block layer request.
fn io_errno(e: &io::Error) -> i32 {
    if let Some(n) = e.raw_os_error() {
        return n;
    }
    match e.kind() {
        io::ErrorKind::Unsupported => libc::ENOTSUP,
        io::ErrorKind::InvalidInput => libc::EINVAL,
        io::ErrorKind::PermissionDenied => libc::EACCES,
        _ => libc::EIO,
    }
}

/// The errno for a failed operation that reports an [`Error`]: the one of the host error it
/// wraps, if any.
fn error_errno(e: &Error) -> i32 {
    let mut src = std::error::Error::source(e);
    while let Some(s) = src {
        if let Some(io) = s.downcast_ref::<io::Error>() {
            return io_errno(io);
        }
        src = s.source();
    }
    libc::EIO
}

/// What to do after handling one request.
#[derive(Debug, PartialEq, Eq)]
enum Action {
    /// Write the header and the body to the device.
    Reply([u8; OUT_HEADER_LEN], Vec<u8>),
    /// `FUSE_FORGET` and `FUSE_BATCH_FORGET` get no reply.
    NoReply,
    /// `fuse_export_halt()`: the device can no longer be trusted, stop reading from it.
    Halt,
}

impl Action {
    /// The reply to the request `unique`: the body, or a positive errno.
    fn reply(unique: u64, ret: std::result::Result<Vec<u8>, i32>) -> Self {
        match ret {
            Ok(body) => Action::Reply(out_header(OUT_HEADER_LEN + body.len(), 0, unique), body),
            Err(errno) => Action::Reply(out_header(OUT_HEADER_LEN, -errno, unique), Vec::new()),
        }
    }
}

type OpResult = std::result::Result<Vec<u8>, i32>;

/// The part of `FuseExport` the request handlers use.
struct Handler {
    blk: Arc<BlockBackend>,
    writable: bool,
    growable: bool,
    /// Whether `allow_other` was used as a mount option.
    allow_other: bool,
    st_mode: AtomicU32,
    st_uid: AtomicU32,
    st_gid: AtomicU32,
}

impl Handler {
    fn new(blk: Arc<BlockBackend>, writable: bool, growable: bool, allow_other: bool) -> Self {
        let mut st_mode = S_IFREG | S_IRUSR;
        if writable {
            st_mode |= S_IWUSR;
        }
        Handler {
            blk,
            writable,
            growable,
            allow_other,
            st_mode: AtomicU32::new(st_mode),
            st_uid: AtomicU32::new(rustix::process::getuid().as_raw()),
            st_gid: AtomicU32::new(rustix::process::getgid().as_raw()),
        }
    }

    /// The part of `co_read_from_fuse_fd()` after the read, and `fuse_co_process_request()`:
    /// check the request of `req.len()` bytes and handle it.
    fn handle(&self, req: &[u8]) -> Action {
        let n = req.len();
        if n < IN_HEADER_LEN {
            error_report(&format!(
                "Incomplete read from FUSE device, expected at least {IN_HEADER_LEN} bytes, read \
                 {n} bytes; cannot trust subsequent requests, halting the export"
            ));
            return Action::Halt;
        }
        let hdr = InHeader::parse(req);
        if hdr.len as usize != n {
            error_report(&format!(
                "Number of bytes read from FUSE device does not match request size, expected {} \
                 bytes, read {n} bytes; cannot trust subsequent requests, halting the export",
                hdr.len
            ));
            return Action::Halt;
        }
        let Some(mut op_hdr_len) = req_op_hdr_len(hdr.opcode) else {
            return Action::reply(hdr.unique, Err(libc::ENOSYS));
        };
        if hdr.opcode == FUSE_INIT {
            if n < IN_HEADER_LEN + OLD_INIT_IN_LEN {
                error_report(&format!("FUSE_INIT request truncated, read only {n} bytes"));
                return Action::reply(hdr.unique, Err(libc::EINVAL));
            }
            if using_old_fuse_init_in(rd_u32(req, IN_HEADER_LEN), rd_u32(req, IN_HEADER_LEN + 4)) {
                op_hdr_len = OLD_INIT_IN_LEN;
            }
        }
        if n < IN_HEADER_LEN + op_hdr_len {
            error_report(&format!(
                "FUSE request truncated, expected {} bytes, read {n} bytes",
                IN_HEADER_LEN + op_hdr_len
            ));
            return Action::reply(hdr.unique, Err(libc::EINVAL));
        }
        let body = &req[IN_HEADER_LEN..];
        let ret = match hdr.opcode {
            FUSE_INIT => self.init(&InitIn::parse(body)),
            FUSE_DESTROY | FUSE_RELEASE => Ok(Vec::new()),
            FUSE_STATFS => Ok(self.statfs()),
            FUSE_OPEN => Ok(self.open()),
            // There is no node but the root node.
            FUSE_LOOKUP => Err(libc::ENOENT),
            // These have no response, and there is nothing we need to do.
            FUSE_FORGET | FUSE_BATCH_FORGET => return Action::NoReply,
            FUSE_GETATTR => self.getattr(),
            FUSE_SETATTR => self.setattr(&SetattrIn::parse(body)),
            FUSE_READ => self.read(rd_u64(body, 8), rd_u32(body, 16)),
            FUSE_WRITE => {
                let size = rd_u32(body, 16);
                let received = n - IN_HEADER_LEN - WRITE_IN_LEN;
                if received < size as usize {
                    warn_report(&format!(
                        "FUSE WRITE truncated; received {received} bytes of {size}"
                    ));
                    Err(libc::EINVAL)
                } else {
                    let data = &body[WRITE_IN_LEN..WRITE_IN_LEN + size as usize];
                    self.write(rd_u64(body, 8), data)
                }
            }
            FUSE_FALLOCATE => self
                .fallocate(rd_u64(body, 8), rd_u64(body, 16), rd_u32(body, 24))
                .map(|()| Vec::new()),
            FUSE_FSYNC | FUSE_FLUSH => self.flush().map(|()| Vec::new()),
            _ => Err(libc::ENOSYS),
        };
        Action::reply(hdr.unique, ret)
    }

    /// `fuse_co_init()`.
    fn init(&self, i: &InitIn) -> OpResult {
        let mut supported_flags = FUSE_ASYNC_READ | FUSE_ASYNC_DIO;
        let mut flags2 = 0;
        if !self.growable {
            // Back when libfuse was used, it would always set this flag and thus the kernel did
            // not execute a truncate itself and passed along O_TRUNC to user space. Keep setting
            // it when the export is not growable.
            supported_flags = FUSE_ATOMIC_O_TRUNC;
        }
        if i.major != FUSE_KERNEL_VERSION {
            error_report(&format!(
                "FUSE major version mismatch: We have 7, but kernel has {}",
                i.major
            ));
            return Err(libc::EINVAL);
        }
        // 2007's 7.9 added fuse_attr.blksize; working around that would be hard.
        if i.minor < 9 {
            error_report(&format!(
                "FUSE minor version too old: 9 required, but kernel has {}",
                i.minor
            ));
            return Err(libc::EINVAL);
        }
        if !using_old_fuse_init_in(i.major, i.minor) {
            // flags2 is only considered if FUSE_INIT_EXT is set.
            supported_flags |= FUSE_INIT_EXT;
            flags2 = i.flags2 & FUSE_DIRECT_IO_ALLOW_MMAP_FLAGS2;
        }
        let page_size = rustix::param::page_size() as u32;
        Ok(InitOut {
            major: FUSE_KERNEL_VERSION,
            minor: FUSE_KERNEL_MINOR_VERSION.min(i.minor),
            max_readahead: i.max_readahead,
            max_write: FUSE_MAX_WRITE_BYTES,
            flags: i.flags & supported_flags,
            flags2,
            // libfuse maximum: 2^16 - 1
            max_background: u16::MAX,
            // libfuse default: max_background * 3 / 4
            congestion_threshold: (u32::from(u16::MAX) * 3 / 4) as u16,
            // libfuse default: 1
            time_gran: 1,
            max_pages: FUSE_MAX_WRITE_BYTES.div_ceil(page_size) as u16,
            map_alignment: 0,
        }
        .encode())
    }

    /// `fuse_co_statfs()`: just enough to not break `df`.
    fn statfs(&self) -> Vec<u8> {
        let mut b = vec![0u8; STATFS_OUT_LEN];
        wr_u32(&mut b, 40, 512);
        wr_u32(&mut b, 44, 255);
        b
    }

    /// `fuse_co_open()`: there is only one inode, so just acknowledge the request.
    fn open(&self) -> Vec<u8> {
        let mut b = vec![0u8; OPEN_OUT_LEN];
        wr_u32(&mut b, 8, FOPEN_DIRECT_IO | FOPEN_PARALLEL_DIRECT_WRITES);
        b
    }

    fn getlength(&self) -> std::result::Result<u64, i32> {
        self.blk.getlength().map_err(|e| io_errno(&e))
    }

    /// `fuse_co_getattr()`.
    fn getattr(&self) -> OpResult {
        let length = self.getlength()?;
        let now = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
        Ok(Attr {
            ino: 1,
            size: length,
            blocks: length.div_ceil(512),
            time: now,
            mode: self.st_mode.load(Ordering::Relaxed),
            nlink: 1,
            uid: self.st_uid.load(Ordering::Relaxed),
            gid: self.st_gid.load(Ordering::Relaxed),
            blksize: 512,
        }
        .encode())
    }

    /// `fuse_co_do_truncate()`.
    fn truncate(&self, size: u64) -> std::result::Result<(), i32> {
        self.blk.truncate(size).map_err(|e| error_errno(&e))
    }

    /// `fuse_co_setattr()`: only resizing and changing the permissions, as far as that actually
    /// permits access.
    fn setattr(&self, s: &SetattrIn) -> OpResult {
        // SIZE and MODE are actually supported, the others can be safely ignored.
        let mut supported_attrs =
            FATTR_SIZE | FATTR_MODE | FATTR_FH | FATTR_LOCKOWNER | FATTR_KILL_SUIDGID;
        if self.allow_other {
            supported_attrs |= FATTR_UID | FATTR_GID;
        }
        if s.valid & !supported_attrs != 0 {
            return Err(libc::ENOTSUP);
        }
        if s.valid & FATTR_MODE != 0 {
            // Without allow_other, non-owners can never access the export.
            if !self.allow_other && s.mode & (S_IRWXG | S_IRWXO) != 0 {
                return Err(libc::EPERM);
            }
            // +w for read-only exports makes no sense.
            if !self.writable && s.mode & (S_IWUSR | S_IWGRP | S_IWOTH) != 0 {
                return Err(libc::EROFS);
            }
        }
        if s.valid & FATTR_SIZE != 0 {
            if !self.writable {
                return Err(libc::EACCES);
            }
            self.truncate(s.size)?;
        }
        if s.valid & FATTR_MODE != 0 {
            // Ignore the file type FUSE supplies, only change the mode.
            self.st_mode.store((s.mode & 0o7777) | S_IFREG, Ordering::Relaxed);
        }
        if s.valid & FATTR_UID != 0 {
            self.st_uid.store(s.uid, Ordering::Relaxed);
        }
        if s.valid & FATTR_GID != 0 {
            self.st_gid.store(s.gid, Ordering::Relaxed);
        }
        self.getattr()
    }

    /// `fuse_co_read()`: short reads at the end of the image.
    fn read(&self, offset: u64, size: u32) -> OpResult {
        // Limited by max_read, should not happen.
        if size > FUSE_MAX_READ_BYTES {
            return Err(libc::EINVAL);
        }
        let blk_len = self.getlength()?;
        if offset >= blk_len {
            return Ok(Vec::new());
        }
        let size = u64::from(size).min(blk_len - offset);
        let mut buf = vec![0u8; size as usize];
        self.blk.pread(offset, &mut buf).map_err(|e| io_errno(&e))?;
        Ok(buf)
    }

    /// `fuse_co_write()`: short writes at the end of the image unless it is growable.
    fn write(&self, offset: u64, data: &[u8]) -> OpResult {
        // Limited by max_write, should not happen.
        if data.len() > FUSE_MAX_WRITE_BYTES as usize {
            return Err(libc::EINVAL);
        }
        if !self.writable {
            return Err(libc::EACCES);
        }
        let blk_len = self.getlength()?;
        let mut size = data.len() as u64;
        if offset >= blk_len && !self.growable {
            return Ok(vec![0u8; WRITE_OUT_LEN]);
        }
        let Some(end) = offset.checked_add(size) else {
            return Err(libc::EINVAL);
        };
        if end > blk_len {
            if self.growable {
                self.truncate(end)?;
            } else {
                size = blk_len - offset;
            }
        }
        self.blk.pwrite(offset, &data[..size as usize]).map_err(|e| io_errno(&e))?;
        let mut b = vec![0u8; WRITE_OUT_LEN];
        wr_u32(&mut b, 0, size as u32);
        Ok(b)
    }

    /// Writes zeroes over `length` bytes at `offset` in requests the block layer takes.
    fn zero_range(
        &self,
        mut offset: u64,
        mut length: u64,
        may_unmap: bool,
    ) -> std::result::Result<(), i32> {
        loop {
            let size = length.min(BDRV_REQUEST_MAX_BYTES);
            self.blk.pwrite_zeroes(offset, size, may_unmap).map_err(|e| io_errno(&e))?;
            offset += size;
            length -= size;
            if length == 0 {
                return Ok(());
            }
        }
    }

    /// `fuse_co_fallocate()`.
    fn fallocate(&self, offset: u64, mut length: u64, mode: u32) -> std::result::Result<(), i32> {
        if !self.writable {
            return Err(libc::EACCES);
        }
        let blk_len = self.getlength()?;
        if mode & FALLOC_FL_KEEP_SIZE != 0 {
            // QEMU compares as unsigned, so past the end the length stays as it is.
            length = length.min(blk_len.wrapping_sub(offset));
        }
        if mode == 0 {
            // We can only fallocate at the EOF with a truncate.
            if offset < blk_len {
                return Err(libc::EOPNOTSUPP);
            }
            if offset > blk_len {
                self.truncate(offset)?;
            }
            self.truncate(offset.wrapping_add(length))
        } else if mode & FALLOC_FL_PUNCH_HOLE != 0 {
            if mode & FALLOC_FL_KEEP_SIZE == 0 {
                return Err(libc::EINVAL);
            }
            // fallocate() returns EOPNOTSUPP for unsupported operations.
            self.zero_range(offset, length, true)
                .map_err(|e| if e == libc::ENOTSUP { libc::EOPNOTSUPP } else { e })
        } else if mode & FALLOC_FL_ZERO_RANGE != 0 {
            if mode & FALLOC_FL_KEEP_SIZE == 0 && offset.wrapping_add(length) > blk_len {
                // No need for zeroes, we are going to write them ourselves.
                self.truncate(offset.wrapping_add(length))?;
            }
            self.zero_range(offset, length, false)
        } else {
            Err(libc::EOPNOTSUPP)
        }
    }

    /// `fuse_co_fsync()` and `fuse_co_flush()`.
    fn flush(&self) -> std::result::Result<(), i32> {
        self.blk.flush().map_err(|e| io_errno(&e))
    }
}

/// `fuse_write_response()` and `fuse_write_buf_response()`.
fn write_response(fd: BorrowedFd<'_>, hdr: &[u8; OUT_HEADER_LEN], body: &[u8]) {
    let to_write = hdr.len() + body.len();
    let ret = loop {
        match rustix::io::writev(fd, &[IoSlice::new(hdr), IoSlice::new(body)]) {
            Err(Errno::INTR) => continue,
            r => break r,
        }
    };
    match ret {
        Err(e) => error_report(&format!("Failed to write to FUSE device: {}", strerror(&e.into()))),
        // Short writes are unexpected, treat them as errors.
        Ok(n) if n != to_write => {
            error_report(&format!("Short write to FUSE device, wrote {n} of {to_write} bytes"))
        }
        Ok(_) => {}
    }
}

/// The request loop: `read_from_fuse_fd()` for as long as the export is not shut down.
fn serve(handler: &Handler, fd: BorrowedFd<'_>, stop: &OwnedFd, halted: &AtomicBool) {
    let mut buf = vec![0u8; REQ_BUF_LEN];
    loop {
        let mut fds = [PollFd::new(&fd, PollFlags::IN), PollFd::new(stop, PollFlags::IN)];
        match poll(&mut fds, None) {
            Ok(_) | Err(Errno::INTR) => {}
            Err(e) => {
                error_report(&format!("Failed to read from FUSE device: {}", strerror(&e.into())));
                return;
            }
        }
        if !fds[1].revents().is_empty() {
            return;
        }
        if fds[0].revents().is_empty() {
            continue;
        }
        let n = match rustix::io::read(fd, &mut buf[..]) {
            Ok(n) => n,
            Err(Errno::AGAIN | Errno::INTR) => continue,
            // The file system was unmounted.
            Err(Errno::NODEV) => return,
            Err(e) => {
                error_report(&format!("Failed to read from FUSE device: {}", strerror(&e.into())));
                continue;
            }
        };
        match handler.handle(&buf[..n]) {
            Action::Reply(hdr, body) => write_response(fd, &hdr, &body),
            Action::NoReply => {}
            Action::Halt => {
                halted.store(true, Ordering::Relaxed);
                return;
            }
        }
    }
}

/// How the export was mounted, which decides how it is unmounted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mounter {
    Direct,
    Fusermount,
}

/// The mount options QEMU passes to libfuse.
fn mount_opts(writable: bool, allow_other: bool) -> String {
    format!(
        "{},nosuid,nodev,noatime,max_read={FUSE_MAX_READ_BYTES},default_permissions{}",
        if writable { "rw" } else { "ro" },
        if allow_other { ",allow_other" } else { "" }
    )
}

/// The data of a direct `mount(2)`: what libfuse turns [`mount_opts`] into for the kernel,
/// without the options that become `MS_*` flags.
fn kernel_opts(fd: i32, allow_other: bool, uid: u32, gid: u32) -> String {
    format!(
        "max_read={FUSE_MAX_READ_BYTES},default_permissions{},fd={fd},rootmode={S_IFREG:o},\
         user_id={uid},group_id={gid}",
        if allow_other { ",allow_other" } else { "" }
    )
}

/// `fuse_mount_sys()` of libfuse: `Ok(None)` when an unprivileged user should fall back to
/// `fusermount3`.
fn mount_direct(
    mountpoint: &str,
    writable: bool,
    allow_other: bool,
) -> std::result::Result<Option<OwnedFd>, ()> {
    let fd = match rustix::fs::open("/dev/fuse", OFlags::RDWR | OFlags::CLOEXEC, Mode::empty()) {
        Ok(fd) => fd,
        Err(e) if e == Errno::NODEV || e == Errno::NOENT => {
            eprintln!("fuse: device not found, try 'modprobe fuse' first");
            return Err(());
        }
        Err(e) => {
            eprintln!("fuse: failed to open /dev/fuse: {}", strerror(&e.into()));
            return Err(());
        }
    };
    let mut flags = MountFlags::NOSUID | MountFlags::NODEV | MountFlags::NOATIME;
    if !writable {
        flags |= MountFlags::RDONLY;
    }
    let data = kernel_opts(
        fd.as_raw_fd(),
        allow_other,
        rustix::process::getuid().as_raw(),
        rustix::process::getgid().as_raw(),
    );
    let Ok(data) = std::ffi::CString::new(data) else {
        return Err(());
    };
    match rustix::mount::mount("/dev/fuse", mountpoint, "fuse", flags, data.as_c_str()) {
        Ok(()) => Ok(Some(fd)),
        Err(Errno::PERM) => Ok(None),
        Err(e) => {
            eprintln!("fuse: mount failed: {}", strerror(&e.into()));
            Err(())
        }
    }
}

/// `fuse_mount_fusermount()` of libfuse: run `fusermount3` and take the `/dev/fuse` descriptor
/// it sends back.
fn mount_fusermount(mountpoint: &str, opts: &str) -> Option<OwnedFd> {
    let (ours, theirs) = match rustix::net::socketpair(
        AddressFamily::UNIX,
        SocketType::STREAM,
        SocketFlags::CLOEXEC,
        None,
    ) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("fuse: socketpair() failed: {}", strerror(&e.into()));
            return None;
        }
    };
    // fusermount3 inherits its end of the socket.
    rustix::io::fcntl_setfd(&theirs, FdFlags::empty()).ok()?;
    let child = Command::new(FUSERMOUNT)
        .arg("-o")
        .arg(opts)
        .arg("--")
        .arg(mountpoint)
        .env("_FUSE_COMMFD", theirs.as_raw_fd().to_string())
        .spawn();
    drop(theirs);
    let mut child = match child {
        Ok(c) => c,
        Err(e) => {
            eprintln!("fuse: failed to exec {FUSERMOUNT}: {}", strerror(&e));
            return None;
        }
    };
    let mut byte = [0u8; 1];
    let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1))];
    let mut cmsg = RecvAncillaryBuffer::new(&mut space);
    let ret = loop {
        let mut iov = [IoSliceMut::new(&mut byte)];
        match rustix::net::recvmsg(&ours, &mut iov, &mut cmsg, RecvFlags::CMSG_CLOEXEC) {
            Err(Errno::INTR) => continue,
            r => break r,
        }
    };
    let mut fd = None;
    if ret.is_ok() {
        for msg in cmsg.drain() {
            if let RecvAncillaryMessage::ScmRights(fds) = msg {
                for f in fds {
                    fd.get_or_insert(f);
                }
            }
        }
    }
    let _ = child.wait();
    fd
}

/// `mount_fuse_export()` with libfuse's `fuse_kern_mount()`.
fn mount_fuse_export(
    mountpoint: &str,
    writable: bool,
    allow_other: bool,
) -> Result<(OwnedFd, Mounter)> {
    let failed = || Error::generic("Failed to mount FUSE session to export");
    match mount_direct(mountpoint, writable, allow_other) {
        Ok(Some(fd)) => Ok((fd, Mounter::Direct)),
        Ok(None) => mount_fusermount(mountpoint, &mount_opts(writable, allow_other))
            .map(|fd| (fd, Mounter::Fusermount))
            .ok_or_else(failed),
        Err(()) => Err(failed()),
    }
}

/// `fuse_session_unmount()`.
fn unmount(mountpoint: &str, mounter: Mounter) {
    if mounter == Mounter::Direct
        && rustix::mount::unmount(mountpoint, UnmountFlags::DETACH).is_ok()
    {
        return;
    }
    let _ = Command::new(FUSERMOUNT).args(["-u", "-q", "-z", "--", mountpoint]).status();
}

/// `is_regular_file()`.
fn is_regular_file(path: &str) -> Result<()> {
    let st = std::fs::metadata(path)
        .map_err(|e| Error::from_io(format!("Failed to stat '{path}'"), e))?;
    if !st.is_file() {
        return Err(Error::generic(format!("'{path}' is not a regular file")));
    }
    Ok(())
}

/// The mounted file system and the thread that serves it.
struct Session {
    handler: Arc<Handler>,
    fd: Arc<OwnedFd>,
    mounter: Mounter,
    /// Dropping the write end of the pipe stops the thread.
    stop: Option<OwnedFd>,
    thread: Option<JoinHandle<()>>,
}

/// A FUSE export, `FuseExport`.
pub struct FuseExport {
    mountpoint: String,
    session: Mutex<Option<Session>>,
    halted: Arc<AtomicBool>,
}

impl std::fmt::Debug for FuseExport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FuseExport").field("mountpoint", &self.mountpoint).finish_non_exhaustive()
    }
}

impl FuseExport {
    /// Whether the export stopped reading requests after a request it could not trust.
    pub fn halted(&self) -> bool {
        self.halted.load(Ordering::Relaxed)
    }

    /// `fuse_export_shutdown()` and `fuse_export_delete()`. The file system is unmounted while
    /// the thread still serves requests, so that nothing waits for a reply that never comes, and
    /// the device is closed last.
    fn close(&self) {
        let Some(mut s) = self.session.lock().unwrap().take() else {
            return;
        };
        unmount(&self.mountpoint, s.mounter);
        drop(s.stop.take());
        if let Some(t) = s.thread.take() {
            let _ = t.join();
        }
        let mut exports = EXPORTS.lock().unwrap();
        if let Some(i) = exports.iter().position(|m| *m == self.mountpoint) {
            exports.remove(i);
        }
        drop(exports);
        drop(s.handler);
        drop(s.fd);
    }
}

impl ExportDriver for FuseExport {
    fn in_use(&self) -> bool {
        false
    }

    fn shutdown(&self) {
        self.close();
    }
}

impl Drop for FuseExport {
    fn drop(&mut self) {
        self.close();
    }
}

/// `fuse_export_create()`.
pub fn create(
    args: &ExportArgs<'_>,
    opts: &super::BlockExportOptionsFuse,
) -> Result<Arc<dyn ExportDriver>> {
    Ok(create_export(args, opts)?)
}

/// [`create`] with the concrete type.
pub fn create_export(
    args: &ExportArgs<'_>,
    opts: &super::BlockExportOptionsFuse,
) -> Result<Arc<FuseExport>> {
    let growable = opts.growable.unwrap_or(false);
    let mountpoint = opts.mountpoint.as_str();

    // For growable and writable exports, take the RESIZE permission.
    if growable || args.writable {
        let (perm, shared) = args.blk.perm();
        args.blk.set_perm(perm | BLK_PERM_RESIZE, shared)?;
    }
    args.blk.set_disable_request_queuing(true);

    // This check comes before is_regular_file(): its stat() would hang on our own mount.
    if EXPORTS.lock().unwrap().iter().any(|m| m == mountpoint) {
        return Err(Error::generic(format!("There already is a FUSE export on '{mountpoint}'")));
    }
    is_regular_file(mountpoint)?;

    let allow_other_opt = opts.allow_other.unwrap_or(FuseExportAllowOther::Auto);
    let (fd, mounter, allow_other) = if allow_other_opt == FuseExportAllowOther::Auto {
        // Try allow_other first, ignore errors.
        match mount_fuse_export(mountpoint, args.writable, true) {
            Ok((fd, m)) => (fd, m, true),
            Err(_) => {
                let (fd, m) = mount_fuse_export(mountpoint, args.writable, false)?;
                (fd, m, false)
            }
        }
    } else {
        let allow_other = allow_other_opt == FuseExportAllowOther::On;
        let (fd, m) = mount_fuse_export(mountpoint, args.writable, allow_other)?;
        (fd, m, allow_other)
    };
    EXPORTS.lock().unwrap().push(mountpoint.to_string());

    let handler = Arc::new(Handler::new(args.blk.clone(), args.writable, growable, allow_other));
    let exp = Arc::new(FuseExport {
        mountpoint: mountpoint.to_string(),
        session: Mutex::new(Some(Session {
            handler: handler.clone(),
            fd: Arc::new(fd),
            mounter,
            stop: None,
            thread: None,
        })),
        halted: Arc::new(AtomicBool::new(false)),
    });
    // On failure, dropping `exp` unmounts and forgets the mount point again.
    start(&exp, handler)?;
    Ok(exp)
}

/// Makes the device non-blocking and starts the thread that serves it.
fn start(exp: &FuseExport, handler: Arc<Handler>) -> Result<()> {
    let mut guard = exp.session.lock().unwrap();
    let s = guard.as_mut().expect("session is set up");
    let nonblock = rustix::fs::fcntl_getfl(&*s.fd)
        .and_then(|fl| rustix::fs::fcntl_setfl(&*s.fd, fl | OFlags::NONBLOCK));
    if let Err(e) = nonblock {
        return Err(Error::from_io("Failed to make FUSE FD non-blocking", e.into()));
    }
    let (stop_r, stop_w) = rustix::pipe::pipe_with(rustix::pipe::PipeFlags::CLOEXEC)
        .map_err(|e| Error::from_io("Failed to create pipe", e.into()))?;
    let fd = s.fd.clone();
    let halted = exp.halted.clone();
    let thread = std::thread::Builder::new()
        .name("fuse-export".into())
        .spawn(move || serve(&handler, fd.as_fd(), &stop_r, &halted))
        .map_err(|e| Error::from_io("Failed to create thread", e))?;
    s.stop = Some(stop_w);
    s.thread = Some(thread);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvm_block::{BLK_PERM_ALL, BLK_PERM_CONSISTENT_READ, BLK_PERM_WRITE, BlockGraph};
    use ruvm_qapi::types::BlockdevOptions;
    use ruvm_qapi::visit::{QObjectInputVisitor, Visit};

    fn request(opcode: u32, unique: u64, body: &[u8]) -> Vec<u8> {
        let hdr = InHeader { len: (IN_HEADER_LEN + body.len()) as u32, opcode, unique };
        let mut v = hdr.encode().to_vec();
        v.extend_from_slice(body);
        v
    }

    fn init_in(major: u32, minor: u32, flags: u32, flags2: u32) -> Vec<u8> {
        let old = using_old_fuse_init_in(major, minor);
        let mut b = vec![0u8; if old { OLD_INIT_IN_LEN } else { INIT_IN_LEN }];
        wr_u32(&mut b, 0, major);
        wr_u32(&mut b, 4, minor);
        wr_u32(&mut b, 8, 131072);
        wr_u32(&mut b, 12, flags);
        if !old {
            wr_u32(&mut b, 16, flags2);
        }
        b
    }

    fn read_in(offset: u64, size: u32) -> Vec<u8> {
        let mut b = vec![0u8; READ_IN_LEN];
        wr_u64(&mut b, 8, offset);
        wr_u32(&mut b, 16, size);
        b
    }

    fn write_in(offset: u64, data: &[u8]) -> Vec<u8> {
        let mut b = vec![0u8; WRITE_IN_LEN];
        wr_u64(&mut b, 8, offset);
        wr_u32(&mut b, 16, data.len() as u32);
        b.extend_from_slice(data);
        b
    }

    fn setattr_in(valid: u32, size: u64, mode: u32) -> Vec<u8> {
        let mut b = vec![0u8; SETATTR_IN_LEN];
        wr_u32(&mut b, 0, valid);
        wr_u64(&mut b, 16, size);
        wr_u32(&mut b, 68, mode);
        b
    }

    /// The error and body of a reply to `unique`.
    fn reply(a: Action, unique: u64) -> (i32, Vec<u8>) {
        let Action::Reply(hdr, body) = a else { panic!("no reply: {a:?}") };
        assert_eq!(rd_u32(&hdr, 0) as usize, OUT_HEADER_LEN + body.len());
        assert_eq!(rd_u64(&hdr, 8), unique);
        (rd_u32(&hdr, 4) as i32, body)
    }

    fn handler(test: &str, len: usize, writable: bool, growable: bool) -> Handler {
        let dir = std::env::temp_dir().join("ruvm-fuse-unit").join(test);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("img");
        let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, data).unwrap();
        let g = BlockGraph::new();
        let json = format!(
            r#"{{"driver": "file", "node-name": "f", "filename": "{}"}}"#,
            path.to_str().unwrap()
        );
        let mut v = QObjectInputVisitor::new(ruvm_qapi::json::from_str(&json).unwrap());
        let mut o = BlockdevOptions::default();
        BlockdevOptions::visit(&mut v, None, &mut o).unwrap();
        g.blockdev_add(o).unwrap();
        let mut perm = BLK_PERM_CONSISTENT_READ;
        if writable {
            perm |= BLK_PERM_WRITE | BLK_PERM_RESIZE;
        }
        let blk = BlockBackend::new(&g, "f", perm, BLK_PERM_ALL).unwrap();
        Handler::new(blk, writable, growable, false)
    }

    #[test]
    fn in_header_round_trip() {
        let h = InHeader { len: 0x1234_5678, opcode: FUSE_READ, unique: 0x0102_0304_0506_0708 };
        let b = h.encode();
        assert_eq!(&b[0..4], &0x1234_5678u32.to_ne_bytes());
        assert_eq!(&b[8..16], &0x0102_0304_0506_0708u64.to_ne_bytes());
        assert_eq!(InHeader::parse(&b), h);
    }

    #[test]
    fn out_header_encoding() {
        let b = out_header(16, -libc::ENOENT, 99);
        assert_eq!(rd_u32(&b, 0), 16);
        assert_eq!(rd_u32(&b, 4) as i32, -libc::ENOENT);
        assert_eq!(rd_u64(&b, 8), 99);
    }

    #[test]
    fn op_header_lengths() {
        assert_eq!(req_op_hdr_len(FUSE_INIT), Some(64));
        assert_eq!(req_op_hdr_len(FUSE_SETATTR), Some(88));
        assert_eq!(req_op_hdr_len(FUSE_WRITE), Some(40));
        assert_eq!(req_op_hdr_len(FUSE_FALLOCATE), Some(32));
        assert_eq!(req_op_hdr_len(FUSE_GETATTR), Some(0));
        // FUSE_LSEEK and FUSE_OPENDIR
        assert_eq!(req_op_hdr_len(46), None);
        assert_eq!(req_op_hdr_len(27), None);
    }

    #[test]
    fn init_in_parsing() {
        let old = InitIn::parse(&init_in(7, 31, 5, 0));
        assert_eq!(old, InitIn { major: 7, minor: 31, max_readahead: 131072, flags: 5, flags2: 0 });
        let new = InitIn::parse(&init_in(7, 40, 5, 0x30));
        assert_eq!(new.flags2, 0x30);
    }

    #[test]
    fn init_out_lengths() {
        let mut o = InitOut { major: 7, minor: 22, max_write: 65536, ..Default::default() };
        assert_eq!(o.encode().len(), FUSE_COMPAT_22_INIT_OUT_SIZE);
        o.minor = 23;
        let b = o.encode();
        assert_eq!(b.len(), INIT_OUT_LEN);
        assert_eq!(rd_u32(&b, 20), 65536);
    }

    #[test]
    fn attr_encoding() {
        let b = Attr { ino: 1, size: 4096, blocks: 8, mode: S_IFREG | 0o600, ..Default::default() }
            .encode();
        assert_eq!(b.len(), ATTR_OUT_LEN);
        assert_eq!(rd_u64(&b, 0), 1);
        assert_eq!(rd_u64(&b, 16), 1);
        assert_eq!(rd_u64(&b, 24), 4096);
        assert_eq!(rd_u64(&b, 32), 8);
        assert_eq!(rd_u32(&b, 16 + 60), S_IFREG | 0o600);
    }

    #[test]
    fn setattr_parsing() {
        let s = SetattrIn::parse(&setattr_in(FATTR_SIZE | FATTR_MODE, 1 << 20, 0o640));
        assert_eq!(s.valid, FATTR_SIZE | FATTR_MODE);
        assert_eq!(s.size, 1 << 20);
        assert_eq!(s.mode, 0o640);
    }

    #[test]
    fn options() {
        assert_eq!(
            mount_opts(true, false),
            "rw,nosuid,nodev,noatime,max_read=1048576,default_permissions"
        );
        assert_eq!(
            mount_opts(false, true),
            "ro,nosuid,nodev,noatime,max_read=1048576,default_permissions,allow_other"
        );
        assert_eq!(
            kernel_opts(5, false, 1000, 100),
            "max_read=1048576,default_permissions,fd=5,rootmode=100000,user_id=1000,group_id=100"
        );
    }

    #[test]
    fn init_request() {
        let h = handler("init", 4096, false, false);
        let (err, body) =
            reply(h.handle(&request(FUSE_INIT, 1, &init_in(7, 40, 0xffff_ffff, !0))), 1);
        assert_eq!(err, 0);
        let o = body;
        assert_eq!(o.len(), INIT_OUT_LEN);
        assert_eq!(rd_u32(&o, 4), 40);
        // Not growable: only FUSE_ATOMIC_O_TRUNC, plus FUSE_INIT_EXT for the new structure.
        assert_eq!(rd_u32(&o, 12), FUSE_ATOMIC_O_TRUNC | FUSE_INIT_EXT);
        assert_eq!(rd_u32(&o, 32), FUSE_DIRECT_IO_ALLOW_MMAP_FLAGS2);
        assert_eq!(rd_u32(&o, 20), FUSE_MAX_WRITE_BYTES);

        let (err, _) = reply(h.handle(&request(FUSE_INIT, 2, &init_in(8, 0, 0, 0))), 2);
        assert_eq!(err, -libc::EINVAL);
        let (err, _) = reply(h.handle(&request(FUSE_INIT, 3, &init_in(7, 8, 0, 0))), 3);
        assert_eq!(err, -libc::EINVAL);
        // Truncated: the new structure announced, only the old one sent.
        let mut short = init_in(7, 40, 0, 0);
        short.truncate(OLD_INIT_IN_LEN);
        let (err, _) = reply(h.handle(&request(FUSE_INIT, 4, &short)), 4);
        assert_eq!(err, -libc::EINVAL);
        let (err, body) = reply(h.handle(&request(FUSE_INIT, 5, &init_in(7, 19, 0, 0))), 5);
        assert_eq!(err, 0);
        assert_eq!(body.len(), FUSE_COMPAT_22_INIT_OUT_SIZE);
    }

    #[test]
    fn broken_requests() {
        let h = handler("broken", 4096, false, false);
        assert_eq!(h.handle(&[0u8; 10]), Action::Halt);
        let mut r = request(FUSE_GETATTR, 1, &[]);
        r.push(0);
        assert_eq!(h.handle(&r), Action::Halt);
        let (err, _) = reply(h.handle(&request(46, 2, &[0u8; 24])), 2);
        assert_eq!(err, -libc::ENOSYS);
        let (err, _) = reply(h.handle(&request(FUSE_READ, 3, &[0u8; 8])), 3);
        assert_eq!(err, -libc::EINVAL);
        assert_eq!(h.handle(&request(FUSE_FORGET, 4, &[0u8; 8])), Action::NoReply);
        let (err, _) = reply(h.handle(&request(FUSE_LOOKUP, 5, b"x\0")), 5);
        assert_eq!(err, -libc::ENOENT);
    }

    #[test]
    fn read_write_requests() {
        let h = handler("rw", 4096, true, false);
        let (err, attr) = reply(h.handle(&request(FUSE_GETATTR, 1, &[])), 1);
        assert_eq!(err, 0);
        assert_eq!(rd_u64(&attr, 24), 4096);
        assert_eq!(rd_u32(&attr, 16 + 60), S_IFREG | S_IRUSR | S_IWUSR);

        let (_, data) = reply(h.handle(&request(FUSE_READ, 2, &read_in(4000, 200))), 2);
        assert_eq!(data.len(), 96);
        assert_eq!(data[0], (4000 % 251) as u8);
        let (_, data) = reply(h.handle(&request(FUSE_READ, 3, &read_in(8192, 200))), 3);
        assert!(data.is_empty());

        let (err, out) = reply(h.handle(&request(FUSE_WRITE, 4, &write_in(4090, &[0xaa; 10]))), 4);
        assert_eq!(err, 0);
        assert_eq!(rd_u32(&out, 0), 6);
        let (_, out) = reply(h.handle(&request(FUSE_WRITE, 5, &write_in(4096, &[1; 4]))), 5);
        assert_eq!(rd_u32(&out, 0), 0);
        let (_, data) = reply(h.handle(&request(FUSE_READ, 6, &read_in(4088, 8))), 6);
        assert_eq!(
            data,
            [(4088 % 251) as u8, (4089 % 251) as u8, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa, 0xaa]
        );

        // A WRITE whose header claims more data than the request carries.
        let mut w = write_in(0, &[1; 8]);
        wr_u32(&mut w, 16, 16);
        let (err, _) = reply(h.handle(&request(FUSE_WRITE, 7, &w)), 7);
        assert_eq!(err, -libc::EINVAL);

        let (err, _) = reply(h.handle(&request(FUSE_FSYNC, 8, &[0u8; 16])), 8);
        assert_eq!(err, 0);
    }

    #[test]
    fn growable_write() {
        let h = handler("grow", 4096, true, true);
        let (_, out) = reply(h.handle(&request(FUSE_WRITE, 1, &write_in(8192, &[7; 512]))), 1);
        assert_eq!(rd_u32(&out, 0), 512);
        assert_eq!(h.blk.getlength().unwrap(), 8704);
    }

    #[test]
    fn read_only_requests() {
        let h = handler("ro", 4096, false, false);
        let (err, _) = reply(h.handle(&request(FUSE_WRITE, 1, &write_in(0, &[1; 4]))), 1);
        assert_eq!(err, -libc::EACCES);
        let (err, _) = reply(h.handle(&request(FUSE_SETATTR, 2, &setattr_in(FATTR_SIZE, 0, 0))), 2);
        assert_eq!(err, -libc::EACCES);
        let (err, _) =
            reply(h.handle(&request(FUSE_SETATTR, 3, &setattr_in(FATTR_MODE, 0, 0o600))), 3);
        assert_eq!(err, -libc::EROFS);
        let (err, _) =
            reply(h.handle(&request(FUSE_SETATTR, 4, &setattr_in(FATTR_MODE, 0, 0o440))), 4);
        assert_eq!(err, -libc::EPERM);
        let (err, attr) =
            reply(h.handle(&request(FUSE_SETATTR, 5, &setattr_in(FATTR_MODE, 0, 0o400))), 5);
        assert_eq!(err, 0);
        assert_eq!(rd_u32(&attr, 16 + 60), S_IFREG | 0o400);
        let (err, _) = reply(h.handle(&request(FUSE_SETATTR, 6, &setattr_in(FATTR_UID, 0, 0))), 6);
        assert_eq!(err, -libc::ENOTSUP);
    }

    #[test]
    fn resize_and_fallocate() {
        let h = handler("resize", 4096, true, false);
        let (err, attr) =
            reply(h.handle(&request(FUSE_SETATTR, 1, &setattr_in(FATTR_SIZE, 8192, 0))), 1);
        assert_eq!(err, 0);
        assert_eq!(rd_u64(&attr, 24), 8192);
        assert_eq!(h.fallocate(0, 512, 0), Err(libc::EOPNOTSUPP));
        assert_eq!(h.fallocate(8192, 512, 0), Ok(()));
        assert_eq!(h.blk.getlength().unwrap(), 8704);
        assert_eq!(h.fallocate(0, 512, FALLOC_FL_PUNCH_HOLE), Err(libc::EINVAL));
        assert_eq!(h.fallocate(0, 512, FALLOC_FL_ZERO_RANGE | FALLOC_FL_KEEP_SIZE), Ok(()));
        let (_, data) = reply(h.handle(&request(FUSE_READ, 2, &read_in(0, 512))), 2);
        assert!(data.iter().all(|&b| b == 0));
        assert_eq!(h.fallocate(0, 512, FALLOC_FL_KEEP_SIZE), Err(libc::EOPNOTSUPP));
    }
}
