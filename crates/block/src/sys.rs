// SPDX-License-Identifier: GPL-2.0-or-later

//! The host calls the file driver needs that rustix does not wrap: byte range locks through
//! `fcntl()` (`qemu_lock_fd()`, `qemu_unlock_fd()` and `qemu_lock_fd_test()` from util/osdep.c)
//! and `F_PUNCHHOLE` on macOS.

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
