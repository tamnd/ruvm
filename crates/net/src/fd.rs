// SPDX-License-Identifier: GPL-2.0-or-later

//! Taking over descriptors passed in by number, as `fd=` and `fds=` do.

#![allow(unsafe_code)]

use std::io;
use std::os::fd::{FromRawFd, OwnedFd, RawFd};

/// Takes ownership of descriptor `fd`, which the user handed over on the command line or
/// through the monitor. Fails with `EBADF` if nothing is open under that number.
///
/// QEMU trusts the number the same way and closes the descriptor when the backend goes away.
pub(crate) fn adopt(fd: RawFd) -> io::Result<OwnedFd> {
    if fd < 0 {
        return Err(io::Error::from_raw_os_error(libc::EBADF));
    }
    // SAFETY: F_GETFD only reads the descriptor flags and touches no memory.
    let r = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    if r < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the descriptor is open, and the user gave it to this process for the backend to
    // own, which is what fd= means in QEMU as well.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}
