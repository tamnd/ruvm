// SPDX-License-Identifier: GPL-2.0-or-later

//! The host calls the file driver needs that rustix does not wrap: byte range locks through
//! `fcntl()` (`qemu_lock_fd()`, `qemu_unlock_fd()` and `qemu_lock_fd_test()` from util/osdep.c)
//! and `F_PUNCHHOLE` on macOS, and the block device and CD-ROM ioctls of `host_device` and
//! `host_cdrom`.

#![allow(unsafe_code)]

use std::io;
use std::os::fd::{AsRawFd, BorrowedFd};

/// Which flavour of byte range lock the host has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LockOps {
    /// Open file description locks. They belong to the open file, so two opens of one image in
    /// the same process see each other, which is what QEMU wants.
    Ofd,
    /// Classic POSIX record locks, owned by the process. Closing any descriptor of the file
    /// drops them, and a process never conflicts with itself.
    Posix,
}

/// `qemu_has_ofd_lock()`. QEMU probes `F_OFD_GETLK` on /dev/null at run time. Every Linux
/// kernel ruvm supports has had OFD locks since 3.15, so this is decided at build time.
pub(crate) fn has_ofd_lock() -> bool {
    cfg!(any(target_os = "linux", target_os = "android"))
}

/// Whether this host can take byte range locks at all through [`lock_fd`].
pub(crate) fn has_locks() -> bool {
    cfg!(any(target_os = "linux", target_os = "android", target_os = "macos", target_os = "ios"))
}

// The `l_type` values. They are `c_short` on the BSDs and `c_int` on Linux, this file uses
// `i32` for both.
#[cfg(any(target_os = "linux", target_os = "android"))]
use libc::{F_RDLCK as RDLCK, F_UNLCK as UNLCK, F_WRLCK as WRLCK};
#[cfg(not(any(target_os = "linux", target_os = "android")))]
const RDLCK: i32 = libc::F_RDLCK as i32;
#[cfg(not(any(target_os = "linux", target_os = "android")))]
const WRLCK: i32 = libc::F_WRLCK as i32;
#[cfg(not(any(target_os = "linux", target_os = "android")))]
const UNLCK: i32 = libc::F_UNLCK as i32;

#[cfg(any(target_os = "linux", target_os = "android"))]
mod consts {
    use std::ffi::c_int;
    // From asm-generic/fcntl.h, the same on every architecture.
    pub(super) const F_OFD_GETLK: c_int = 36;
    pub(super) const F_OFD_SETLK: c_int = 37;
}

#[cfg(any(target_os = "linux", target_os = "android", target_os = "macos", target_os = "ios"))]
fn fcntl_lock(
    fd: BorrowedFd<'_>,
    ops: LockOps,
    test: bool,
    start: u64,
    ty: i32,
) -> io::Result<i32> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let (setlk, getlk) = match ops {
        LockOps::Ofd => (consts::F_OFD_SETLK, consts::F_OFD_GETLK),
        LockOps::Posix => (libc::F_SETLK, libc::F_GETLK),
    };
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    let (setlk, getlk) = {
        let _ = ops;
        (libc::F_SETLK, libc::F_GETLK)
    };
    let mut fl = libc::flock {
        l_type: ty as libc::c_short,
        l_whence: libc::SEEK_SET as libc::c_short,
        l_start: start as libc::off_t,
        l_len: 1,
        l_pid: 0,
    };
    let cmd = if test { getlk } else { setlk };
    loop {
        // SAFETY: `fd` is a live descriptor for the duration of the borrow, `cmd` is one of the
        // lock commands, which take a pointer to a `struct flock`, and `fl` is a valid,
        // initialised `struct flock` the kernel may write back into.
        let ret = unsafe { libc::fcntl(fd.as_raw_fd(), cmd, &mut fl as *mut libc::flock) };
        if ret != -1 {
            return Ok(i32::from(fl.l_type));
        }
        let e = io::Error::last_os_error();
        // RETRY_ON_EINTR() for the setter, the tester is not retried in QEMU but EINTR cannot
        // happen for F_GETLK anyway.
        if e.kind() != io::ErrorKind::Interrupted {
            return Err(e);
        }
    }
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios"
)))]
fn fcntl_lock(_: BorrowedFd<'_>, _: LockOps, _: bool, _: u64, _: i32) -> io::Result<i32> {
    Err(io::Error::from_raw_os_error(libc::ENOTSUP))
}

/// `qemu_lock_fd()` for one byte: a shared (read) lock unless `exclusive`.
pub(crate) fn lock_fd(
    fd: BorrowedFd<'_>,
    ops: LockOps,
    byte: u64,
    exclusive: bool,
) -> io::Result<()> {
    let ty = if exclusive { WRLCK } else { RDLCK };
    fcntl_lock(fd, ops, false, byte, ty).map(drop)
}

/// `qemu_unlock_fd()` for one byte.
pub(crate) fn unlock_fd(fd: BorrowedFd<'_>, ops: LockOps, byte: u64) -> io::Result<()> {
    fcntl_lock(fd, ops, false, byte, UNLCK).map(drop)
}

/// `qemu_lock_fd_test()`: fails with `EAGAIN` when somebody else holds a lock that would
/// conflict with taking one of the given kind.
pub(crate) fn lock_fd_test(
    fd: BorrowedFd<'_>,
    ops: LockOps,
    byte: u64,
    exclusive: bool,
) -> io::Result<()> {
    let ty = if exclusive { WRLCK } else { RDLCK };
    let got = fcntl_lock(fd, ops, true, byte, ty)?;
    if got == UNLCK { Ok(()) } else { Err(io::Error::from_raw_os_error(libc::EAGAIN)) }
}

/// `fcntl(F_PUNCHHOLE)`, what file-posix uses for discard on macOS. `ENODEV` means the file
/// system cannot do it and becomes `ENOTSUP`, as in `handle_aiocb_discard()`.
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(crate) fn punch_hole(fd: BorrowedFd<'_>, offset: u64, len: u64) -> io::Result<()> {
    let arg = libc::fpunchhole_t {
        fp_flags: 0,
        reserved: 0,
        fp_offset: offset as libc::off_t,
        fp_length: len as libc::off_t,
    };
    // SAFETY: `fd` is live for the borrow and F_PUNCHHOLE reads a `fpunchhole_t`, which `arg`
    // is, fully initialised and only read by the kernel.
    let ret = unsafe {
        libc::fcntl(fd.as_raw_fd(), libc::F_PUNCHHOLE, &arg as *const libc::fpunchhole_t)
    };
    if ret == -1 {
        let e = io::Error::last_os_error();
        if e.raw_os_error() == Some(libc::ENODEV) {
            return Err(io::Error::from_raw_os_error(libc::ENOTSUP));
        }
        return Err(e);
    }
    Ok(())
}

/// The device ioctls file-posix issues that rustix does not wrap, each with its argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DevIoctl {
    /// `BLKROGET`: whether the block device is read-only. Returns 0 or 1.
    #[cfg(target_os = "linux")]
    BlkRoGet,
    /// `BLKSECTGET`: the most sectors one request may carry. Returns it.
    #[cfg(target_os = "linux")]
    BlkSectGet,
    /// `BLKDISCARD` of `(offset, length)`.
    #[cfg(target_os = "linux")]
    BlkDiscard(u64, u64),
    /// `BLKZEROOUT` of `(offset, length)`.
    #[cfg(target_os = "linux")]
    BlkZeroOut(u64, u64),
    /// `CDROM_DRIVE_STATUS` with `CDSL_CURRENT`. Returns the `CDS_*` status.
    #[cfg(target_os = "linux")]
    CdromDriveStatus,
    /// `CDROMEJECT`.
    #[cfg(target_os = "linux")]
    #[allow(dead_code, reason = "used by the Driver::eject and lock_medium implementations")]
    CdromEject,
    /// `CDROMCLOSETRAY`.
    #[cfg(target_os = "linux")]
    #[allow(dead_code, reason = "used by the Driver::eject and lock_medium implementations")]
    CdromCloseTray,
    /// `CDROM_LOCKDOOR` with the door state.
    #[cfg(target_os = "linux")]
    #[allow(dead_code, reason = "used by the Driver::eject and lock_medium implementations")]
    CdromLockDoor(bool),
    /// `DKIOCGETBLOCKSIZE`: the logical block size. Returns it.
    #[cfg(target_os = "macos")]
    GetBlockSize,
    /// `DKIOCGETBLOCKCOUNT`: the number of logical blocks. Returns it.
    #[cfg(target_os = "macos")]
    GetBlockCount,
}

/// `CDS_DISC_OK` from linux/cdrom.h.
#[cfg(target_os = "linux")]
pub(crate) const CDS_DISC_OK: u64 = 4;

#[cfg(target_os = "linux")]
mod dev {
    use rustix::ioctl::opcode;
    // linux/fs.h. `_IO()` differs between architectures, rustix knows how.
    pub(super) const BLKROGET: u32 = opcode::none(0x12, 94) as u32;
    pub(super) const BLKSECTGET: u32 = opcode::none(0x12, 103) as u32;
    pub(super) const BLKDISCARD: u32 = opcode::none(0x12, 119) as u32;
    pub(super) const BLKZEROOUT: u32 = opcode::none(0x12, 127) as u32;
    // linux/cdrom.h. These are plain numbers, the same everywhere.
    pub(super) const CDROMEJECT: u32 = 0x5309;
    pub(super) const CDROMCLOSETRAY: u32 = 0x5319;
    pub(super) const CDROM_DRIVE_STATUS: u32 = 0x5326;
    pub(super) const CDROM_LOCKDOOR: u32 = 0x5329;
    pub(super) const CDSL_CURRENT: usize = i32::MAX as usize;
}

#[cfg(target_os = "macos")]
mod dev {
    // sys/disk.h: `_IOR('d', 24, uint32_t)` and `_IOR('d', 25, uint64_t)`.
    pub(super) const DKIOCGETBLOCKSIZE: u32 = 0x4004_6418;
    pub(super) const DKIOCGETBLOCKCOUNT: u32 = 0x4008_6419;
}

/// Issues one of the device ioctls in [`DevIoctl`] on `fd`, retrying on `EINTR`. The result
/// is what the ioctl returns or, for the getters, the value it wrote.
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(crate) fn dev_ioctl(fd: BorrowedFd<'_>, req: DevIoctl) -> io::Result<u64> {
    use std::ffi::c_void;
    #[cfg(target_os = "linux")]
    use std::ptr;

    /// Where the result is: the return value, or one of the locals the kernel wrote.
    /// Not every kind is used on every host.
    #[allow(dead_code)]
    #[derive(Clone, Copy)]
    enum Out {
        Ret,
        Int,
        U16,
        U32,
        U64,
    }
    #[cfg_attr(not(target_os = "linux"), allow(unused_mut))]
    let mut int_out: libc::c_int = 0;
    #[cfg_attr(not(target_os = "linux"), allow(unused_mut))]
    let mut u16_out: u16 = 0;
    #[cfg_attr(not(target_os = "macos"), allow(unused_mut))]
    let mut u32_out: u32 = 0;
    #[cfg_attr(not(target_os = "macos"), allow(unused_mut))]
    let mut u64_out: u64 = 0;
    #[cfg(target_os = "linux")]
    let mut range = match req {
        DevIoctl::BlkDiscard(off, len) | DevIoctl::BlkZeroOut(off, len) => [off, len],
        _ => [0u64; 2],
    };
    let (code, arg, out): (u32, *mut c_void, Out) = match req {
        #[cfg(target_os = "linux")]
        DevIoctl::BlkRoGet => (dev::BLKROGET, (&raw mut int_out).cast(), Out::Int),
        #[cfg(target_os = "linux")]
        DevIoctl::BlkSectGet => (dev::BLKSECTGET, (&raw mut u16_out).cast(), Out::U16),
        #[cfg(target_os = "linux")]
        DevIoctl::BlkDiscard(..) => (dev::BLKDISCARD, range.as_mut_ptr().cast(), Out::Ret),
        #[cfg(target_os = "linux")]
        DevIoctl::BlkZeroOut(..) => (dev::BLKZEROOUT, range.as_mut_ptr().cast(), Out::Ret),
        #[cfg(target_os = "linux")]
        DevIoctl::CdromDriveStatus => {
            (dev::CDROM_DRIVE_STATUS, ptr::without_provenance_mut(dev::CDSL_CURRENT), Out::Ret)
        }
        #[cfg(target_os = "linux")]
        DevIoctl::CdromEject => (dev::CDROMEJECT, ptr::null_mut(), Out::Ret),
        #[cfg(target_os = "linux")]
        DevIoctl::CdromCloseTray => (dev::CDROMCLOSETRAY, ptr::null_mut(), Out::Ret),
        #[cfg(target_os = "linux")]
        DevIoctl::CdromLockDoor(locked) => {
            (dev::CDROM_LOCKDOOR, ptr::without_provenance_mut(usize::from(locked)), Out::Ret)
        }
        #[cfg(target_os = "macos")]
        DevIoctl::GetBlockSize => (dev::DKIOCGETBLOCKSIZE, (&raw mut u32_out).cast(), Out::U32),
        #[cfg(target_os = "macos")]
        DevIoctl::GetBlockCount => (dev::DKIOCGETBLOCKCOUNT, (&raw mut u64_out).cast(), Out::U64),
    };
    let ret = loop {
        // SAFETY: `fd` is live for the borrow. Every request above is one whose argument is
        // either a plain integer (passed in the pointer's place, as the C headers do), no
        // argument at all, or a pointer to a local of exactly the type the kernel reads or
        // writes for that request: `int` for BLKROGET, `unsigned short` for BLKSECTGET, `uint64_t[2]` for BLKDISCARD and
        // BLKZEROOUT, `uint32_t` for DKIOCGETBLOCKSIZE and `uint64_t` for DKIOCGETBLOCKCOUNT.
        // Those locals outlive the call.
        let ret = unsafe { libc::ioctl(fd.as_raw_fd(), code as _, arg) };
        if ret != -1 {
            break ret;
        }
        let e = io::Error::last_os_error();
        if e.kind() != io::ErrorKind::Interrupted {
            return Err(e);
        }
    };
    Ok(match out {
        Out::Ret => ret as u64,
        Out::Int => int_out as u64,
        Out::U16 => u64::from(u16_out),
        Out::U32 => u64::from(u32_out),
        Out::U64 => u64_out,
    })
}
