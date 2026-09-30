// SPDX-License-Identifier: GPL-2.0-or-later

//! The Linux kernel NBD client: `nbd_init()`, `nbd_client()` and `nbd_disconnect()` from the
//! `__linux__` part of nbd/client.c. They hand a connected socket to `/dev/nbdN` with the
//! `NBD_*` ioctls of `<linux/nbd.h>`.

#![allow(unsafe_code)]

use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, RawFd};

use ruvm_base::Error;
use ruvm_block::nbd::{NBD_FLAG_READ_ONLY, NbdExportInfo};

const NBD_SET_SOCK: u32 = 0xab00;
const NBD_SET_BLKSIZE: u32 = 0xab01;
const NBD_DO_IT: u32 = 0xab03;
const NBD_CLEAR_SOCK: u32 = 0xab04;
const NBD_CLEAR_QUE: u32 = 0xab05;
const NBD_SET_SIZE_BLOCKS: u32 = 0xab07;
const NBD_DISCONNECT: u32 = 0xab08;
const NBD_SET_FLAGS: u32 = 0xab0a;
/// `BLKROSET`, `_IO(0x12, 93)`.
const BLKROSET: u32 = 0x125d;

const BDRV_SECTOR_SIZE: u64 = 512;

/// The argument of one ioctl.
enum Arg<'a> {
    Int(libc::c_ulong),
    IntPtr(&'a libc::c_int),
}

fn ioctl(fd: RawFd, request: u32, arg: Arg<'_>) -> io::Result<libc::c_int> {
    let arg = match arg {
        Arg::Int(v) => v,
        Arg::IntPtr(p) => p as *const libc::c_int as libc::c_ulong,
    };
    // SAFETY: `fd` is an open descriptor owned by the caller for the duration of the call. The
    // NBD ioctls used here take their argument by value, except BLKROSET, which reads one
    // `int` through the pointer, and `Arg::IntPtr` holds a reference that outlives the call.
    let r = unsafe { libc::ioctl(fd, request as libc::Ioctl, arg) };
    if r < 0 { Err(io::Error::last_os_error()) } else { Ok(r) }
}

/// `nbd_init()`: give the negotiated connection `sock` to the NBD device `dev`.
pub(crate) fn nbd_init(dev: &File, sock: RawFd, info: &NbdExportInfo) -> Result<(), Error> {
    let fd = dev.as_raw_fd();
    let sector_size = BDRV_SECTOR_SIZE.max(u64::from(info.min_block));
    let sectors = info.size / sector_size;
    if libc::c_ulong::try_from(sectors).is_err() {
        return Err(Error::generic(format!(
            "Export size {} too large for 32-bit kernel",
            info.size
        )));
    }
    fn fail(msg: &'static str) -> impl Fn(io::Error) -> Error {
        move |_| Error::generic(msg)
    }
    ioctl(fd, NBD_SET_SOCK, Arg::Int(sock as libc::c_ulong))
        .map_err(fail("Failed to set NBD socket"))?;
    ioctl(fd, NBD_SET_BLKSIZE, Arg::Int(sector_size as libc::c_ulong))
        .map_err(fail("Failed setting NBD block size"))?;
    ioctl(fd, NBD_SET_SIZE_BLOCKS, Arg::Int(sectors as libc::c_ulong))
        .map_err(fail("Failed setting size (in blocks)"))?;
    if let Err(e) = ioctl(fd, NBD_SET_FLAGS, Arg::Int(libc::c_ulong::from(info.flags))) {
        if e.raw_os_error() != Some(libc::ENOTTY) {
            return Err(Error::generic("Failed setting flags"));
        }
        let read_only: libc::c_int = libc::c_int::from(info.flags & NBD_FLAG_READ_ONLY != 0);
        ioctl(fd, BLKROSET, Arg::IntPtr(&read_only))
            .map_err(fail("Failed setting read-only attribute"))?;
    }
    Ok(())
}

/// `nbd_client()`: serve the device until it is disconnected. A disconnect through
/// `NBD_DISCONNECT` counts as success.
pub(crate) fn nbd_client(dev: &File) -> io::Result<()> {
    let fd = dev.as_raw_fd();
    let r = match ioctl(fd, NBD_DO_IT, Arg::Int(0)) {
        Err(e) if e.raw_os_error() == Some(libc::EPIPE) => Ok(()),
        Err(e) => Err(e),
        Ok(_) => Ok(()),
    };
    let _ = ioctl(fd, NBD_CLEAR_QUE, Arg::Int(0));
    let _ = ioctl(fd, NBD_CLEAR_SOCK, Arg::Int(0));
    r
}

/// `nbd_disconnect()`.
pub(crate) fn nbd_disconnect(dev: &File) {
    let fd = dev.as_raw_fd();
    let _ = ioctl(fd, NBD_CLEAR_QUE, Arg::Int(0));
    let _ = ioctl(fd, NBD_DISCONNECT, Arg::Int(0));
    let _ = ioctl(fd, NBD_CLEAR_SOCK, Arg::Int(0));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ioctls_on_a_regular_file_fail() {
        let f = File::open("/proc/self/stat").unwrap();
        let info = NbdExportInfo { size: 1 << 20, ..Default::default() };
        let e = nbd_init(&f, -1, &info).unwrap_err();
        assert_eq!(e.message(), "Failed to set NBD socket");
        nbd_disconnect(&f);
        assert!(nbd_client(&f).is_err());
    }
}
