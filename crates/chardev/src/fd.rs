// SPDX-License-Identifier: GPL-2.0-or-later

//! `socket_get_fd()` from util/qemu-sockets.c: taking over a socket that was passed in by
//! number, as `-chardev socket,fd=N` does.

#![allow(unsafe_code)]

use std::num::IntErrorKind;
use std::os::fd::{BorrowedFd, FromRawFd, OwnedFd};

use rustix::io::Errno;
use ruvm_base::{Error, Result};

/// `qemu_strtoi()` in base 10 with no end pointer: the whole string has to be the number.
fn parse_fd(s: &str) -> std::result::Result<i32, Errno> {
    let t = s.trim_start_matches(|c: char| c.is_ascii_whitespace());
    let digits = t.strip_prefix(['+', '-']).unwrap_or(t);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(Errno::INVAL);
    }
    t.parse().map_err(|e: std::num::ParseIntError| match e.kind() {
        IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => Errno::RANGE,
        _ => Errno::INVAL,
    })
}

/// `socket_get_fd()` with no monitor command running, which is how the command line reaches
/// it: `fdstr` is a descriptor number and the descriptor has to be a socket. The chardev owns
/// the descriptor from then on and closes it when it goes away, as QEMU does. A descriptor
/// that is open but not a socket is closed here, the way QEMU closes it too.
pub(crate) fn socket_get_fd(fdstr: &str) -> Result<OwnedFd> {
    let fd = parse_fd(fdstr)
        .map_err(|e| Error::from_io(format!("Unable to parse FD number {fdstr}"), e.into()))?;
    let not_socket = || Error::generic(format!("File descriptor '{fdstr}' is not a socket"));
    if fd < 0 {
        return Err(not_socket());
    }
    // SAFETY: the borrow only lives for the F_GETFD call, which touches no memory and fails
    // with EBADF when nothing is open under the number.
    if rustix::io::fcntl_getfd(unsafe { BorrowedFd::borrow_raw(fd) }).is_err() {
        return Err(not_socket());
    }
    // SAFETY: the descriptor is open, and the user handed it to this process for the chardev
    // to own, which is what fd= means in QEMU as well.
    let fd = unsafe { OwnedFd::from_raw_fd(fd) };
    if rustix::net::sockopt::socket_type(&fd).is_err() {
        return Err(not_socket());
    }
    Ok(fd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_parse_like_qemu_strtoi() {
        assert_eq!(parse_fd("5"), Ok(5));
        assert_eq!(parse_fd(" +7"), Ok(7));
        assert_eq!(parse_fd("-1"), Ok(-1));
        assert_eq!(parse_fd(""), Err(Errno::INVAL));
        assert_eq!(parse_fd("mon"), Err(Errno::INVAL));
        assert_eq!(parse_fd("5 "), Err(Errno::INVAL));
        assert_eq!(parse_fd("99999999999"), Err(Errno::RANGE));
    }

    #[test]
    fn only_sockets_are_taken() {
        let e = socket_get_fd("x").unwrap_err();
        assert_eq!(e.message(), "Unable to parse FD number x: Invalid argument");
        let e = socket_get_fd("-3").unwrap_err();
        assert_eq!(e.message(), "File descriptor '-3' is not a socket");
        let file = std::fs::File::open("/dev/null").unwrap();
        let n = std::os::fd::IntoRawFd::into_raw_fd(file);
        let e = socket_get_fd(&n.to_string()).unwrap_err();
        assert_eq!(e.message(), format!("File descriptor '{n}' is not a socket"));
        let (a, _b) = std::os::unix::net::UnixStream::pair().unwrap();
        let n = std::os::fd::IntoRawFd::into_raw_fd(a);
        assert!(socket_get_fd(&n.to_string()).is_ok());
    }
}
