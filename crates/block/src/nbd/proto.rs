// SPDX-License-Identifier: GPL-2.0-or-later

//! The wire protocol: constants from include/block/nbd.h and nbd/nbd-internal.h, the lookup
//! tables and errno maps from nbd/common.c and nbd/server.c, the request and reply headers, and
//! the socket I/O helpers with the error text QEMU's channel layer produces.

use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpStream};
#[cfg(unix)]
use std::os::unix::net::UnixStream;

use ruvm_base::{Error, Result};

pub(crate) const NBD_INIT_MAGIC: u64 = 0x4e42_444d_4147_4943; // "NBDMAGIC"
pub(crate) const NBD_OPTS_MAGIC: u64 = 0x4948_4156_454f_5054; // "IHAVEOPT"
pub(crate) const NBD_CLIENT_MAGIC: u64 = 0x0000_4202_8186_1253;
pub(crate) const NBD_REP_MAGIC: u64 = 0x0003_e889_0455_65a9;

pub(crate) const NBD_REQUEST_MAGIC: u32 = 0x2560_9513;
pub(crate) const NBD_EXTENDED_REQUEST_MAGIC: u32 = 0x21e4_1c71;
pub(crate) const NBD_SIMPLE_REPLY_MAGIC: u32 = 0x6744_6698;
pub(crate) const NBD_STRUCTURED_REPLY_MAGIC: u32 = 0x668e_33ef;
pub(crate) const NBD_EXTENDED_REPLY_MAGIC: u32 = 0x6e8a_278c;

pub(crate) const NBD_REQUEST_SIZE: usize = 4 + 2 + 2 + 8 + 8 + 4;
pub(crate) const NBD_EXTENDED_REQUEST_SIZE: usize = 4 + 2 + 2 + 8 + 8 + 8;

/// Transmission flags, sent by the server for each export.
pub const NBD_FLAG_HAS_FLAGS: u16 = 1 << 0;
pub const NBD_FLAG_READ_ONLY: u16 = 1 << 1;
pub const NBD_FLAG_SEND_FLUSH: u16 = 1 << 2;
pub const NBD_FLAG_SEND_FUA: u16 = 1 << 3;
pub const NBD_FLAG_ROTATIONAL: u16 = 1 << 4;
pub const NBD_FLAG_SEND_TRIM: u16 = 1 << 5;
pub const NBD_FLAG_SEND_WRITE_ZEROES: u16 = 1 << 6;
pub const NBD_FLAG_SEND_DF: u16 = 1 << 7;
pub const NBD_FLAG_CAN_MULTI_CONN: u16 = 1 << 8;
pub const NBD_FLAG_SEND_RESIZE: u16 = 1 << 9;
pub const NBD_FLAG_SEND_CACHE: u16 = 1 << 10;
pub const NBD_FLAG_SEND_FAST_ZERO: u16 = 1 << 11;
pub const NBD_FLAG_BLOCK_STAT_PAYLOAD: u16 = 1 << 12;

/// Handshake flags, from the server and from the client.
pub(crate) const NBD_FLAG_FIXED_NEWSTYLE: u16 = 1 << 0;
pub(crate) const NBD_FLAG_NO_ZEROES: u16 = 1 << 1;
pub(crate) const NBD_FLAG_C_FIXED_NEWSTYLE: u32 = 1 << 0;
pub(crate) const NBD_FLAG_C_NO_ZEROES: u32 = 1 << 1;

pub(crate) const NBD_OPT_EXPORT_NAME: u32 = 1;
pub(crate) const NBD_OPT_ABORT: u32 = 2;
pub(crate) const NBD_OPT_LIST: u32 = 3;
pub(crate) const NBD_OPT_STARTTLS: u32 = 5;
pub(crate) const NBD_OPT_INFO: u32 = 6;
pub(crate) const NBD_OPT_GO: u32 = 7;
pub(crate) const NBD_OPT_STRUCTURED_REPLY: u32 = 8;
pub(crate) const NBD_OPT_LIST_META_CONTEXT: u32 = 9;
pub(crate) const NBD_OPT_SET_META_CONTEXT: u32 = 10;
pub(crate) const NBD_OPT_EXTENDED_HEADERS: u32 = 11;

const fn rep_err(v: u32) -> u32 {
    (1 << 31) | v
}
pub(crate) const NBD_REP_ACK: u32 = 1;
pub(crate) const NBD_REP_SERVER: u32 = 2;
pub(crate) const NBD_REP_INFO: u32 = 3;
pub(crate) const NBD_REP_META_CONTEXT: u32 = 4;
pub(crate) const NBD_REP_ERR_UNSUP: u32 = rep_err(1);
pub(crate) const NBD_REP_ERR_POLICY: u32 = rep_err(2);
pub(crate) const NBD_REP_ERR_INVALID: u32 = rep_err(3);
pub(crate) const NBD_REP_ERR_PLATFORM: u32 = rep_err(4);
pub(crate) const NBD_REP_ERR_TLS_REQD: u32 = rep_err(5);
pub(crate) const NBD_REP_ERR_UNKNOWN: u32 = rep_err(6);
pub(crate) const NBD_REP_ERR_SHUTDOWN: u32 = rep_err(7);
pub(crate) const NBD_REP_ERR_BLOCK_SIZE_REQD: u32 = rep_err(8);
pub(crate) const NBD_REP_ERR_TOO_BIG: u32 = rep_err(9);
pub(crate) const NBD_REP_ERR_EXT_HEADER_REQD: u32 = rep_err(10);

pub(crate) const NBD_INFO_EXPORT: u16 = 0;
pub(crate) const NBD_INFO_NAME: u16 = 1;
pub(crate) const NBD_INFO_DESCRIPTION: u16 = 2;
pub(crate) const NBD_INFO_BLOCK_SIZE: u16 = 3;

/// Request flags.
pub const NBD_CMD_FLAG_FUA: u16 = 1 << 0;
pub const NBD_CMD_FLAG_NO_HOLE: u16 = 1 << 1;
pub const NBD_CMD_FLAG_DF: u16 = 1 << 2;
pub const NBD_CMD_FLAG_REQ_ONE: u16 = 1 << 3;
pub const NBD_CMD_FLAG_FAST_ZERO: u16 = 1 << 4;
pub const NBD_CMD_FLAG_PAYLOAD_LEN: u16 = 1 << 5;

/// Commands.
pub const NBD_CMD_READ: u16 = 0;
pub const NBD_CMD_WRITE: u16 = 1;
pub const NBD_CMD_DISC: u16 = 2;
pub const NBD_CMD_FLUSH: u16 = 3;
pub const NBD_CMD_TRIM: u16 = 4;
pub const NBD_CMD_CACHE: u16 = 5;
pub const NBD_CMD_WRITE_ZEROES: u16 = 6;
pub const NBD_CMD_BLOCK_STATUS: u16 = 7;

/// `NBD_DEFAULT_PORT`.
pub const NBD_DEFAULT_PORT: u16 = 10809;
/// `NBD_MAX_BUFFER_SIZE`, the largest payload either side sends or accepts.
pub const NBD_MAX_BUFFER_SIZE: u32 = 32 * 1024 * 1024;
/// `NBD_MAX_STRING_SIZE`, the longest name or query.
pub const NBD_MAX_STRING_SIZE: usize = 4096;
/// `NBD_DEFAULT_HANDSHAKE_MAX_SECS`.
pub const NBD_DEFAULT_HANDSHAKE_MAX_SECS: u32 = 10;
/// `NBD_DEFAULT_MAX_CONNECTIONS`.
pub const NBD_DEFAULT_MAX_CONNECTIONS: u32 = 100;

pub(crate) const NBD_REPLY_FLAG_DONE: u16 = 1 << 0;
const fn reply_err(v: u16) -> u16 {
    (1 << 15) | v
}
pub(crate) const NBD_REPLY_TYPE_NONE: u16 = 0;
pub(crate) const NBD_REPLY_TYPE_OFFSET_DATA: u16 = 1;
pub(crate) const NBD_REPLY_TYPE_OFFSET_HOLE: u16 = 2;
pub(crate) const NBD_REPLY_TYPE_BLOCK_STATUS: u16 = 5;
pub(crate) const NBD_REPLY_TYPE_BLOCK_STATUS_EXT: u16 = 6;
pub(crate) const NBD_REPLY_TYPE_ERROR: u16 = reply_err(1);
pub(crate) const NBD_REPLY_TYPE_ERROR_OFFSET: u16 = reply_err(2);

/// Block status flags of `base:allocation`.
pub const NBD_STATE_HOLE: u64 = 1 << 0;
pub const NBD_STATE_ZERO: u64 = 1 << 1;
/// The flag of `qemu:dirty-bitmap:` contexts.
pub const NBD_STATE_DIRTY: u64 = 1 << 0;

pub(crate) const NBD_SUCCESS: u32 = 0;
pub(crate) const NBD_EPERM: u32 = 1;
pub(crate) const NBD_EIO: u32 = 5;
pub(crate) const NBD_ENOMEM: u32 = 12;
pub(crate) const NBD_EINVAL: u32 = 22;
pub(crate) const NBD_ENOSPC: u32 = 28;
pub(crate) const NBD_EOVERFLOW: u32 = 75;
pub(crate) const NBD_ENOTSUP: u32 = 95;
pub(crate) const NBD_ESHUTDOWN: u32 = 108;

/// `ESHUTDOWN`, which the Windows CRT does not have. Winsock's `WSAESHUTDOWN` stands in.
#[cfg(unix)]
pub(crate) const ESHUTDOWN: i32 = libc::ESHUTDOWN;
#[cfg(not(unix))]
pub(crate) const ESHUTDOWN: i32 = 10058;

/// `NBDMode`: how far negotiation got, in increasing order of features.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum NbdMode {
    /// The server lacks newstyle negotiation.
    Oldstyle,
    /// Newstyle, but only `NBD_OPT_EXPORT_NAME` is safe.
    ExportName,
    /// Newstyle with simple replies only.
    Simple,
    /// Structured replies are enabled.
    Structured,
    /// Extended headers are enabled.
    #[default]
    Extended,
}

impl NbdMode {
    /// `nbd_mode_lookup()`.
    pub fn as_str(self) -> &'static str {
        match self {
            NbdMode::Oldstyle => "oldstyle",
            NbdMode::ExportName => "export name only",
            NbdMode::Simple => "simple headers",
            NbdMode::Structured => "structured replies",
            NbdMode::Extended => "extended headers",
        }
    }
}

/// `nbd_opt_lookup()`.
pub fn nbd_opt_lookup(opt: u32) -> &'static str {
    match opt {
        NBD_OPT_EXPORT_NAME => "export name",
        NBD_OPT_ABORT => "abort",
        NBD_OPT_LIST => "list",
        NBD_OPT_STARTTLS => "starttls",
        NBD_OPT_INFO => "info",
        NBD_OPT_GO => "go",
        NBD_OPT_STRUCTURED_REPLY => "structured reply",
        NBD_OPT_LIST_META_CONTEXT => "list meta context",
        NBD_OPT_SET_META_CONTEXT => "set meta context",
        NBD_OPT_EXTENDED_HEADERS => "extended headers",
        _ => "<unknown>",
    }
}

/// `nbd_rep_lookup()`.
pub fn nbd_rep_lookup(rep: u32) -> &'static str {
    match rep {
        NBD_REP_ACK => "ack",
        NBD_REP_SERVER => "server",
        NBD_REP_INFO => "info",
        NBD_REP_META_CONTEXT => "meta context",
        NBD_REP_ERR_UNSUP => "unsupported",
        NBD_REP_ERR_POLICY => "denied by policy",
        NBD_REP_ERR_INVALID => "invalid",
        NBD_REP_ERR_PLATFORM => "platform lacks support",
        NBD_REP_ERR_TLS_REQD => "TLS required",
        NBD_REP_ERR_UNKNOWN => "export unknown",
        NBD_REP_ERR_SHUTDOWN => "server shutting down",
        NBD_REP_ERR_BLOCK_SIZE_REQD => "block size required",
        NBD_REP_ERR_TOO_BIG => "option payload too big",
        NBD_REP_ERR_EXT_HEADER_REQD => "extended headers required",
        _ => "<unknown>",
    }
}

/// `nbd_info_lookup()`.
pub fn nbd_info_lookup(info: u16) -> &'static str {
    match info {
        NBD_INFO_EXPORT => "export",
        NBD_INFO_NAME => "name",
        NBD_INFO_DESCRIPTION => "description",
        NBD_INFO_BLOCK_SIZE => "block size",
        _ => "<unknown>",
    }
}

/// `nbd_cmd_lookup()`.
pub fn nbd_cmd_lookup(cmd: u16) -> &'static str {
    match cmd {
        NBD_CMD_READ => "read",
        NBD_CMD_WRITE => "write",
        NBD_CMD_DISC => "disconnect",
        NBD_CMD_FLUSH => "flush",
        NBD_CMD_TRIM => "trim",
        NBD_CMD_CACHE => "cache",
        NBD_CMD_WRITE_ZEROES => "write zeroes",
        NBD_CMD_BLOCK_STATUS => "block status",
        _ => "<unknown>",
    }
}

/// `nbd_reply_type_lookup()`.
pub fn nbd_reply_type_lookup(t: u16) -> &'static str {
    match t {
        NBD_REPLY_TYPE_NONE => "none",
        NBD_REPLY_TYPE_OFFSET_DATA => "data",
        NBD_REPLY_TYPE_OFFSET_HOLE => "hole",
        NBD_REPLY_TYPE_BLOCK_STATUS => "block status (32-bit)",
        NBD_REPLY_TYPE_BLOCK_STATUS_EXT => "block status (64-bit)",
        NBD_REPLY_TYPE_ERROR => "generic error",
        NBD_REPLY_TYPE_ERROR_OFFSET => "error at offset",
        _ if t & (1 << 15) != 0 => "<unknown error>",
        _ => "<unknown>",
    }
}

/// `nbd_err_lookup()`.
pub fn nbd_err_lookup(err: u32) -> &'static str {
    match err {
        NBD_SUCCESS => "success",
        NBD_EPERM => "EPERM",
        NBD_EIO => "EIO",
        NBD_ENOMEM => "ENOMEM",
        NBD_EINVAL => "EINVAL",
        NBD_ENOSPC => "ENOSPC",
        NBD_EOVERFLOW => "EOVERFLOW",
        NBD_ENOTSUP => "ENOTSUP",
        NBD_ESHUTDOWN => "ESHUTDOWN",
        _ => "<unknown>",
    }
}

/// `nbd_reply_type_is_error()`.
pub(crate) fn nbd_reply_type_is_error(t: u16) -> bool {
    t & (1 << 15) != 0
}

/// `nbd_errno_to_system_errno()`: unknown values become `EINVAL`.
pub fn nbd_errno_to_system_errno(err: u32) -> i32 {
    match err {
        NBD_SUCCESS => 0,
        NBD_EPERM => libc::EPERM,
        NBD_EIO => libc::EIO,
        NBD_ENOMEM => libc::ENOMEM,
        NBD_ENOSPC => libc::ENOSPC,
        NBD_EOVERFLOW => libc::EOVERFLOW,
        NBD_ENOTSUP => libc::ENOTSUP,
        NBD_ESHUTDOWN => ESHUTDOWN,
        _ => libc::EINVAL,
    }
}

/// `system_errno_to_nbd_errno()` from nbd/server.c.
pub fn system_errno_to_nbd_errno(err: i32) -> u32 {
    #[cfg(unix)]
    if err == libc::EDQUOT {
        return NBD_ENOSPC;
    }
    match err {
        0 => NBD_SUCCESS,
        _ if err == libc::EPERM || err == libc::EROFS => NBD_EPERM,
        _ if err == libc::EIO => NBD_EIO,
        _ if err == libc::ENOMEM => NBD_ENOMEM,
        _ if err == libc::EFBIG || err == libc::ENOSPC => NBD_ENOSPC,
        _ if err == libc::EOVERFLOW => NBD_EOVERFLOW,
        _ if err == libc::ENOTSUP || err == libc::EOPNOTSUPP => NBD_ENOTSUP,
        _ if err == ESHUTDOWN => NBD_ESHUTDOWN,
        _ => NBD_EINVAL,
    }
}

/// The errno of an I/O error, `EIO` when it has none.
pub(crate) fn io_errno(e: &io::Error) -> i32 {
    e.raw_os_error().unwrap_or(libc::EIO)
}

/// A connected stream socket, TCP or Unix domain.
#[derive(Debug)]
pub enum NbdStream {
    Tcp(TcpStream),
    #[cfg(unix)]
    Unix(UnixStream),
}

impl NbdStream {
    pub(crate) fn try_clone(&self) -> io::Result<NbdStream> {
        match self {
            NbdStream::Tcp(s) => s.try_clone().map(NbdStream::Tcp),
            #[cfg(unix)]
            NbdStream::Unix(s) => s.try_clone().map(NbdStream::Unix),
        }
    }

    /// `qio_channel_shutdown(QIO_CHANNEL_SHUTDOWN_BOTH)`. Errors are ignored as in QEMU.
    pub(crate) fn shutdown(&self) {
        let _ = match self {
            NbdStream::Tcp(s) => s.shutdown(Shutdown::Both),
            #[cfg(unix)]
            NbdStream::Unix(s) => s.shutdown(Shutdown::Both),
        };
    }

    pub(crate) fn set_nonblocking(&self, nb: bool) -> io::Result<()> {
        match self {
            NbdStream::Tcp(s) => s.set_nonblocking(nb),
            #[cfg(unix)]
            NbdStream::Unix(s) => s.set_nonblocking(nb),
        }
    }
}

impl Read for NbdStream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            NbdStream::Tcp(s) => s.read(buf),
            #[cfg(unix)]
            NbdStream::Unix(s) => s.read(buf),
        }
    }
}

impl Write for NbdStream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            NbdStream::Tcp(s) => s.write(buf),
            #[cfg(unix)]
            NbdStream::Unix(s) => s.write(buf),
        }
    }

    fn write_vectored(&mut self, bufs: &[io::IoSlice<'_>]) -> io::Result<usize> {
        match self {
            NbdStream::Tcp(s) => s.write_vectored(bufs),
            #[cfg(unix)]
            NbdStream::Unix(s) => s.write_vectored(bufs),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// `qio_channel_read_all()`.
pub(crate) fn read_all<R: Read + ?Sized>(r: &mut R, mut buf: &mut [u8]) -> Result<()> {
    while !buf.is_empty() {
        match r.read(buf) {
            Ok(0) => {
                return Err(Error::generic("Unexpected end-of-file before all data were read"));
            }
            Ok(n) => buf = &mut buf[n..],
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(Error::from_io("Unable to read from socket", e)),
        }
    }
    Ok(())
}

/// `nbd_read()`: with a description the error says what was being read.
pub(crate) fn nbd_read<R: Read + ?Sized>(
    r: &mut R,
    buf: &mut [u8],
    desc: Option<&str>,
) -> Result<()> {
    read_all(r, buf).map_err(|e| match desc {
        Some(d) => e.prepend(format!("Failed to read {d}: ")),
        None => e,
    })
}

pub(crate) fn nbd_read16<R: Read + ?Sized>(r: &mut R, desc: &str) -> Result<u16> {
    let mut b = [0; 2];
    nbd_read(r, &mut b, Some(desc))?;
    Ok(u16::from_be_bytes(b))
}

pub(crate) fn nbd_read32<R: Read + ?Sized>(r: &mut R, desc: &str) -> Result<u32> {
    let mut b = [0; 4];
    nbd_read(r, &mut b, Some(desc))?;
    Ok(u32::from_be_bytes(b))
}

pub(crate) fn nbd_read64<R: Read + ?Sized>(r: &mut R, desc: &str) -> Result<u64> {
    let mut b = [0; 8];
    nbd_read(r, &mut b, Some(desc))?;
    Ok(u64::from_be_bytes(b))
}

/// `nbd_read_eof()`: `Ok(false)` on end-of-file before any byte was read.
pub(crate) fn nbd_read_eof<R: Read + ?Sized>(r: &mut R, mut buf: &mut [u8]) -> Result<bool> {
    let mut partial = false;
    while !buf.is_empty() {
        match r.read(buf) {
            Ok(0) if partial => {
                return Err(Error::generic("Unexpected end-of-file before all bytes were read"));
            }
            Ok(0) => return Ok(false),
            Ok(n) => {
                partial = true;
                buf = &mut buf[n..];
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(Error::from_io("Unable to read from socket", e)),
        }
    }
    Ok(true)
}

/// `nbd_write()`, `qio_channel_write_all()`.
pub(crate) fn nbd_write<W: Write + ?Sized>(w: &mut W, buf: &[u8]) -> Result<()> {
    w.write_all(buf).map_err(|e| Error::from_io("Unable to write to socket", e))
}

/// `nbd_drop()`: reads and throws away `size` bytes, 64 KiB at a time.
pub(crate) fn nbd_drop<R: Read + ?Sized>(r: &mut R, mut size: u64) -> Result<()> {
    let mut buf = vec![0u8; size.min(65536) as usize];
    while size > 0 {
        let n = size.min(65536) as usize;
        nbd_read(r, &mut buf[..n], None)?;
        size -= n as u64;
    }
    Ok(())
}

/// `NBDRequest`, in the form both header widths share.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NbdRequest {
    pub cookie: u64,
    pub from: u64,
    pub len: u64,
    pub flags: u16,
    pub typ: u16,
    pub mode: NbdMode,
}

impl NbdRequest {
    /// The header `nbd_send_request()` writes: compact below extended mode, wide from there.
    pub fn encode(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(NBD_EXTENDED_REQUEST_SIZE);
        let ext = self.mode >= NbdMode::Extended;
        let magic = if ext { NBD_EXTENDED_REQUEST_MAGIC } else { NBD_REQUEST_MAGIC };
        b.extend_from_slice(&magic.to_be_bytes());
        b.extend_from_slice(&self.flags.to_be_bytes());
        b.extend_from_slice(&self.typ.to_be_bytes());
        b.extend_from_slice(&self.cookie.to_be_bytes());
        b.extend_from_slice(&self.from.to_be_bytes());
        if ext {
            b.extend_from_slice(&self.len.to_be_bytes());
        } else {
            debug_assert!(self.len <= u64::from(u32::MAX));
            b.extend_from_slice(&(self.len as u32).to_be_bytes());
        }
        b
    }
}

/// A reply header after `nbd_receive_reply()` normalized it: simple, or a chunk of either width.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NbdReply {
    /// `NBD_SIMPLE_REPLY_MAGIC` or `NBD_STRUCTURED_REPLY_MAGIC` (also for extended chunks).
    pub magic: u32,
    pub cookie: u64,
    /// The error of a simple reply.
    pub error: u32,
    pub flags: u16,
    pub typ: u16,
    pub length: u64,
}

impl NbdReply {
    pub(crate) fn is_simple(&self) -> bool {
        self.magic == NBD_SIMPLE_REPLY_MAGIC
    }
}

/// `nbd_receive_reply()`: `Ok(None)` on a clean end-of-file.
pub(crate) fn nbd_receive_reply<R: Read + ?Sized>(
    r: &mut R,
    mode: NbdMode,
) -> Result<Option<NbdReply>> {
    let _ = mode;
    let mut m = [0u8; 4];
    if !nbd_read_eof(r, &mut m)? {
        return Ok(None);
    }
    let magic = u32::from_be_bytes(m);
    let mut reply = NbdReply { magic, ..NbdReply::default() };
    match magic {
        NBD_SIMPLE_REPLY_MAGIC => {
            let mut b = [0u8; 12];
            nbd_read(r, &mut b, Some("reply"))?;
            reply.error = be32(&b[0..]);
            reply.cookie = be64(&b[4..]);
        }
        NBD_STRUCTURED_REPLY_MAGIC | NBD_EXTENDED_REPLY_MAGIC => {
            let ext = magic == NBD_EXTENDED_REPLY_MAGIC;
            let mut b = [0u8; 28];
            let n = if ext { 28 } else { 16 };
            nbd_read(r, &mut b[..n], Some("structured chunk"))?;
            reply.flags = be16(&b[0..]);
            reply.typ = be16(&b[2..]);
            reply.cookie = be64(&b[4..]);
            // The extended header's offset is ignored, as in QEMU.
            reply.length = if ext { be64(&b[20..]) } else { u64::from(be32(&b[12..])) };
            reply.magic = NBD_STRUCTURED_REPLY_MAGIC;
            if reply.length > u64::from(NBD_MAX_BUFFER_SIZE) + 8 {
                return Err(Error::generic(format!(
                    "server chunk {} ({}) payload is too long",
                    reply.typ,
                    nbd_rep_lookup(u32::from(reply.typ))
                )));
            }
        }
        _ => return Err(Error::generic(format!("invalid magic (got 0x{magic:x})"))),
    }
    Ok(Some(reply))
}

pub(crate) fn be16(b: &[u8]) -> u16 {
    u16::from_be_bytes([b[0], b[1]])
}

pub(crate) fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

pub(crate) fn be64(b: &[u8]) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[..8]);
    u64::from_be_bytes(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_encoding() {
        let r = NbdRequest {
            cookie: 1,
            from: 0x1000,
            len: 512,
            flags: NBD_CMD_FLAG_FUA,
            typ: NBD_CMD_WRITE,
            mode: NbdMode::Structured,
        };
        let b = r.encode();
        assert_eq!(b.len(), NBD_REQUEST_SIZE);
        assert_eq!(&b[..4], &[0x25, 0x60, 0x95, 0x13]);
        assert_eq!(&b[4..8], &[0, 1, 0, 1]);
        assert_eq!(&b[24..], &[0, 0, 2, 0]);
        let b = NbdRequest { mode: NbdMode::Extended, ..r }.encode();
        assert_eq!(b.len(), NBD_EXTENDED_REQUEST_SIZE);
        assert_eq!(&b[..4], &[0x21, 0xe4, 0x1c, 0x71]);
        assert_eq!(&b[24..], &[0, 0, 0, 0, 0, 0, 2, 0]);
    }

    #[test]
    fn errno_maps() {
        assert_eq!(nbd_errno_to_system_errno(NBD_ENOSPC), libc::ENOSPC);
        assert_eq!(nbd_errno_to_system_errno(12345), libc::EINVAL);
        assert_eq!(system_errno_to_nbd_errno(libc::EROFS), NBD_EPERM);
        assert_eq!(system_errno_to_nbd_errno(libc::EFBIG), NBD_ENOSPC);
        assert_eq!(system_errno_to_nbd_errno(libc::EOPNOTSUPP), NBD_ENOTSUP);
        assert_eq!(system_errno_to_nbd_errno(libc::EBADF), NBD_EINVAL);
    }

    #[test]
    fn reply_parsing() {
        let mut b = Vec::new();
        b.extend_from_slice(&NBD_EXTENDED_REPLY_MAGIC.to_be_bytes());
        b.extend_from_slice(&1u16.to_be_bytes());
        b.extend_from_slice(&NBD_REPLY_TYPE_NONE.to_be_bytes());
        b.extend_from_slice(&7u64.to_be_bytes());
        b.extend_from_slice(&0u64.to_be_bytes());
        b.extend_from_slice(&0u64.to_be_bytes());
        let r = nbd_receive_reply(&mut &b[..], NbdMode::Extended).unwrap().unwrap();
        assert_eq!(r.magic, NBD_STRUCTURED_REPLY_MAGIC);
        assert_eq!(r.cookie, 7);
        assert_eq!(r.flags, NBD_REPLY_FLAG_DONE);
        assert!(nbd_receive_reply(&mut &[][..], NbdMode::Extended).unwrap().is_none());
        let e = nbd_receive_reply(&mut &[1u8, 2, 3, 4][..], NbdMode::Simple).unwrap_err();
        assert_eq!(e.message(), "invalid magic (got 0x1020304)");
        let e = nbd_receive_reply(&mut &[0x67u8, 0x44][..], NbdMode::Simple).unwrap_err();
        assert_eq!(e.message(), "Unexpected end-of-file before all bytes were read");
    }
}
