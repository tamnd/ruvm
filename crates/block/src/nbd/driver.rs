// SPDX-License-Identifier: GPL-2.0-or-later

//! The `nbd` block driver from block/nbd.c.
//!
//! [`NbdClient`] is `BDRVNBDState` with the request functions: it sends one request at a time
//! and reads the reply chunks for it, handles the reconnect states, and maps what the server
//! says to errno values and block status. [`NbdDriver`] puts it behind the block layer's driver
//! interface.

use std::io;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{BlockdevOptionsNbd, BlockdevOptionsU, SocketAddress, SocketAddressU};

use super::client::NbdExportInfo;
use super::connection::NbdClientConnection;
use super::proto::*;
use super::server::NbdBlockStatus;
use super::uri::nbd_parse_filename;
use crate::drivers::{DriverDef, OpenArgs};
use crate::node::{
    BDRV_BLOCK_DATA, BDRV_BLOCK_OFFSET_VALID, BDRV_BLOCK_ZERO, BDRV_REQ_FUA, BDRV_REQ_MAY_UNMAP,
    BDRV_REQ_NO_FALLBACK, BlockLimits, BlockStatus, Driver, Node, NodeFlags, ReopenState,
};

const BDRV_SECTOR_SIZE: u64 = 512;

/// `PATH_MAX`, the size of `bs->exact_filename`.
const PATH_MAX: usize = 4096;

/// The only cookie in use: requests are sent one at a time, and QEMU's first slot is cookie 1.
const COOKIE: u64 = 1;

/// `NBDClientState`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ClientState {
    ConnectingWait,
    ConnectingNowait,
    Connected,
    Quit,
}

#[derive(Debug)]
struct Inner {
    ioc: Option<NbdStream>,
    state: ClientState,
    info: NbdExportInfo,
    /// Whether the node is read-only, which a read-only export requires.
    read_only: bool,
    /// The message of the last failed request, which QEMU only traces.
    last_error: Option<String>,
}

/// An error on the channel: the negative errno and what went wrong.
type ChanError = (i32, Error);

/// `NBDReplyChunkIter`.
#[derive(Debug)]
struct Iter {
    ret: i32,
    request_ret: i32,
    err: Option<Error>,
    done: bool,
    only_structured: bool,
}

impl Iter {
    fn new(only_structured: bool) -> Iter {
        Iter { ret: 0, request_ret: 0, err: None, done: false, only_structured }
    }

    /// `nbd_iter_channel_error()`: the first channel error wins.
    fn channel_error(&mut self, ret: i32, e: Error) {
        assert!(ret < 0);
        if self.ret == 0 {
            self.ret = ret;
            self.err = Some(e);
        }
    }

    /// `nbd_iter_request_error()`.
    fn request_error(&mut self, ret: i32) {
        assert!(ret < 0);
        if self.request_ret == 0 {
            self.request_ret = ret;
        }
    }
}

/// Where a read reply goes.
#[derive(Debug)]
struct ReadTarget<'a> {
    buf: &'a mut [u8],
    offset: u64,
}

/// A chunk the iterator hands to the caller: its header and, if any, its small payload.
#[derive(Debug)]
struct Chunk {
    reply: NbdReply,
    payload: Vec<u8>,
}

/// One extent of `NBD_CMD_BLOCK_STATUS`, `NBDExtent64`.
#[derive(Clone, Copy, Debug, Default)]
struct Extent {
    length: u64,
    flags: u64,
}

/// `NBD_MAX_MALLOC_PAYLOAD`.
const NBD_MAX_MALLOC_PAYLOAD: u64 = 1000;

/// The client side of an NBD export, `BDRVNBDState`.
#[derive(Debug)]
pub struct NbdClient {
    inner: Mutex<Inner>,
    conn: NbdClientConnection,
    saddr: SocketAddress,
    export: Option<String>,
    x_dirty_bitmap: Option<String>,
    alloc_depth: bool,
    reconnect_delay: u32,
}

fn errno(e: i32) -> io::Error {
    io::Error::from_raw_os_error(e)
}

fn neg_to_io(ret: i32) -> io::Result<()> {
    if ret < 0 { Err(errno(-ret)) } else { Ok(()) }
}

/// The errno `nbd_receive_reply()` fails with: protocol violations are `EINVAL`, everything
/// else happened on the socket.
fn reply_errno(e: &Error) -> i32 {
    let m = e.message();
    if m.starts_with("invalid magic") || m.starts_with("server chunk") {
        -libc::EINVAL
    } else {
        -libc::EIO
    }
}

impl NbdClient {
    /// `nbd_open()`: check the options, connect and negotiate.
    ///
    /// `read_only` is the node's flag, and is set when the export is read-only and
    /// `auto_read_only` allows that. `filename` is the file name the node was opened with, for
    /// the error QEMU's block layer reports when the driver gives no message.
    pub fn open(
        opts: &BlockdevOptionsNbd,
        filename: Option<&str>,
        read_only: &mut bool,
        auto_read_only: bool,
    ) -> Result<NbdClient> {
        // nbd_process_options().
        if let SocketAddressU::Fd(f) = &opts.server.u {
            // socket_address_parse_named_fd() without a monitor to look names up in.
            if !f.str.starts_with(|c: char| c.is_ascii_digit()) || f.str.parse::<i32>().is_err() {
                return Err(Error::generic(format!("Invalid file descriptor number '{}'", f.str)));
            }
        }
        if opts.export.as_ref().is_some_and(|e| e.len() > NBD_MAX_STRING_SIZE) {
            return Err(Error::generic("export name too long to send to server"));
        }
        if let Some(id) = &opts.tls_creds {
            // There are no QOM objects, so no TLS credentials either.
            return Err(Error::generic(format!("No TLS credentials with id '{id}'")));
        }
        if opts.x_dirty_bitmap.as_ref().is_some_and(|e| e.len() > NBD_MAX_STRING_SIZE) {
            return Err(Error::generic("x-dirty-bitmap query too long to send to server"));
        }
        let reconnect_delay = opts.reconnect_delay.unwrap_or(0);
        let open_timeout = opts.open_timeout.unwrap_or(0);

        let conn = NbdClientConnection::new(
            &opts.server,
            true,
            opts.export.as_deref(),
            opts.x_dirty_bitmap.as_deref(),
        );
        let mut deadline = None;
        if open_timeout > 0 {
            conn.enable_retry();
            deadline = Some(Instant::now() + Duration::from_secs(u64::from(open_timeout)));
        }
        let mut c = NbdClient {
            inner: Mutex::new(Inner {
                ioc: None,
                state: ClientState::ConnectingWait,
                info: NbdExportInfo::default(),
                read_only: *read_only,
                last_error: None,
            }),
            conn,
            saddr: opts.server.clone(),
            export: opts.export.clone(),
            x_dirty_bitmap: opts.x_dirty_bitmap.clone(),
            alloc_depth: opts.x_dirty_bitmap.as_deref() == Some("qemu:allocation-depth"),
            reconnect_delay,
        };
        {
            let inner = c.inner.get_mut().unwrap_or_else(|e| e.into_inner());
            let r =
                Self::establish(&c.conn, &c.x_dirty_bitmap, inner, true, deadline, auto_read_only);
            match r {
                Ok(()) => {}
                Err((_, Some(e))) => return Err(e),
                Err((ret, None)) => {
                    let io = errno(-ret);
                    return Err(match filename {
                        Some(f) if !f.is_empty() => {
                            Error::from_io(format!("Could not open '{f}'"), io)
                        }
                        _ => Error::from_io("Could not open image", io),
                    });
                }
            }
            *read_only = inner.read_only;
        }
        c.conn.enable_retry();
        Ok(c)
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// What negotiation found out about the export.
    pub fn info(&self) -> NbdExportInfo {
        self.lock().info.clone()
    }

    /// `nbd_co_getlength()`.
    pub fn size(&self) -> u64 {
        self.lock().info.size
    }

    /// Whether the node is read-only, possibly because the export is.
    pub fn is_read_only(&self) -> bool {
        self.lock().read_only
    }

    /// The message of the last request that failed on the connection, for debugging. QEMU
    /// only traces these.
    pub fn last_error(&self) -> Option<String> {
        self.lock().last_error.clone()
    }

    /// `bs->bl.request_alignment` as `nbd_refresh_limits()` sets it.
    pub fn request_alignment(&self) -> u32 {
        Self::alignment_of(&self.lock().info)
    }

    fn alignment_of(info: &NbdExportInfo) -> u32 {
        if info.min_block != 0 {
            return info.min_block;
        }
        if info.size % BDRV_SECTOR_SIZE != 0 || info.base_allocation {
            1
        } else {
            BDRV_SECTOR_SIZE as u32
        }
    }

    /// `bs->bl.max_transfer`.
    pub fn max_transfer(&self) -> u32 {
        let info = &self.lock().info;
        if info.max_block == 0 {
            NBD_MAX_BUFFER_SIZE
        } else {
            NBD_MAX_BUFFER_SIZE.min(info.max_block)
        }
    }

    /// The `(max_pdiscard, max_pwrite_zeroes, opt_transfer)` limits, with 0 for no limit.
    fn limits(&self) -> (u64, u64, u32) {
        let info = self.lock().info.clone();
        let min = u64::from(Self::alignment_of(&info));
        let max = u64::from(if info.max_block == 0 {
            NBD_MAX_BUFFER_SIZE
        } else {
            NBD_MAX_BUFFER_SIZE.min(info.max_block)
        });
        let (discard, zero) = if info.mode >= NbdMode::Extended {
            (0, 0)
        } else {
            ((i32::MAX as u64) / min * min, max)
        };
        (discard, zero, info.opt_block)
    }

    /// `nbd_handle_updated_info()`. Fails without a message, as QEMU passes no error pointer.
    fn handle_updated_info(
        x_dirty_bitmap: &Option<String>,
        inner: &mut Inner,
        auto_read_only: bool,
    ) -> std::result::Result<(), i32> {
        if x_dirty_bitmap.is_some() && !inner.info.base_allocation {
            // "requested x-dirty-bitmap %s not found"
            return Err(-libc::EINVAL);
        }
        if inner.info.flags & NBD_FLAG_READ_ONLY != 0 && !inner.read_only {
            // bdrv_apply_auto_read_only(): "NBD export is read-only"
            if !auto_read_only {
                return Err(-libc::EACCES);
            }
            inner.read_only = true;
        }
        Ok(())
    }

    /// `nbd_co_do_establish_connection()`. The error has a message unless it came from
    /// `nbd_handle_updated_info()`.
    fn establish(
        conn: &NbdClientConnection,
        x_dirty_bitmap: &Option<String>,
        inner: &mut Inner,
        blocking: bool,
        deadline: Option<Instant>,
        auto_read_only: bool,
    ) -> std::result::Result<(), (i32, Option<Error>)> {
        assert!(inner.ioc.is_none());
        let (mut ioc, mut info) = match conn.establish(blocking, deadline) {
            Ok(r) => r,
            Err(e) => return Err((-libc::ECONNREFUSED, Some(e))),
        };
        info.x_dirty_bitmap = x_dirty_bitmap.clone();
        inner.info = info;
        if let Err(ret) = Self::handle_updated_info(x_dirty_bitmap, inner, auto_read_only) {
            // Say goodbye before hanging up.
            let req = NbdRequest { typ: NBD_CMD_DISC, mode: inner.info.mode, ..Default::default() };
            let _ = nbd_write(&mut ioc, &req.encode());
            ioc.shutdown();
            return Err((ret, None));
        }
        inner.ioc = Some(ioc);
        inner.state = ClientState::Connected;
        Ok(())
    }

    /// `nbd_channel_error_locked()`.
    fn channel_error(&self, inner: &mut Inner, ret: i32) {
        if inner.state == ClientState::Connected {
            if let Some(s) = &inner.ioc {
                s.shutdown();
            }
        }
        if ret == -libc::EIO {
            if inner.state == ClientState::Connected {
                inner.state = if self.reconnect_delay > 0 {
                    ClientState::ConnectingWait
                } else {
                    ClientState::ConnectingNowait
                };
            }
        } else {
            inner.state = ClientState::Quit;
        }
    }

    /// `nbd_reconnect_attempt()`.
    fn reconnect_attempt(&self, inner: &mut Inner) {
        let blocking = inner.state == ClientState::ConnectingWait;
        let deadline =
            blocking.then(|| Instant::now() + Duration::from_secs(u64::from(self.reconnect_delay)));
        if let Some(s) = inner.ioc.take() {
            s.shutdown();
        }
        let r = Self::establish(&self.conn, &self.x_dirty_bitmap, inner, blocking, deadline, false);
        if r.is_err() {
            if let Some(d) = deadline {
                // reconnect_delay_timer_cb(): the delay is over, fail requests from now on.
                if Instant::now() >= d && inner.state == ClientState::ConnectingWait {
                    inner.state = ClientState::ConnectingNowait;
                }
            }
        }
    }

    /// `nbd_co_send_request()`.
    fn send_request(&self, inner: &mut Inner, req: &mut NbdRequest, data: Option<&[u8]>) -> i32 {
        if inner.state != ClientState::Connected {
            if matches!(inner.state, ClientState::ConnectingWait | ClientState::ConnectingNowait) {
                self.reconnect_attempt(inner);
            }
            if inner.state != ClientState::Connected {
                self.channel_error(inner, -libc::EIO);
                return -libc::EIO;
            }
        }
        req.cookie = COOKIE;
        req.mode = inner.info.mode;
        let ioc = inner.ioc.as_mut().expect("connected");
        let mut buf = req.encode();
        let r = match data {
            Some(d) if d.len() <= 65536 => {
                buf.extend_from_slice(d);
                nbd_write(ioc, &buf)
            }
            Some(d) => nbd_write(ioc, &buf).and_then(|()| nbd_write(ioc, d)),
            None => nbd_write(ioc, &buf),
        };
        if r.is_err() {
            self.channel_error(inner, -libc::EIO);
            return -libc::EIO;
        }
        0
    }

    /// `nbd_receive_replies()` for the one request in flight.
    fn receive_replies(&self, inner: &mut Inner) -> std::result::Result<NbdReply, ChanError> {
        let mode = inner.info.mode;
        let ioc = inner.ioc.as_mut().expect("connected");
        let r = match nbd_receive_reply(ioc, mode) {
            Ok(Some(r)) => r,
            Ok(None) => {
                self.channel_error(inner, -libc::EIO);
                return Err((-libc::EIO, Error::generic("server dropped connection")));
            }
            Err(e) => {
                let ret = reply_errno(&e);
                self.channel_error(inner, ret);
                return Err((ret, e));
            }
        };
        if !r.is_simple() && mode < NbdMode::Structured {
            self.channel_error(inner, -libc::EINVAL);
            return Err((-libc::EINVAL, Error::generic("unexpected structured reply")));
        }
        if r.cookie != COOKIE {
            self.channel_error(inner, -libc::EINVAL);
            return Err((-libc::EINVAL, Error::generic("unexpected cookie value")));
        }
        Ok(r)
    }

    /// `nbd_co_do_receive_one_chunk()`.
    fn do_receive_one_chunk(
        &self,
        inner: &mut Inner,
        only_structured: bool,
        request_ret: &mut i32,
        target: Option<&mut ReadTarget<'_>>,
        want_payload: bool,
    ) -> std::result::Result<Chunk, ChanError> {
        *request_ret = 0;
        let reply = self
            .receive_replies(inner)
            .map_err(|(_, e)| (-libc::EIO, e.prepend("Connection closed: ")))?;
        let ioc = inner.ioc.as_mut().expect("connected");
        let chunk = |payload| Chunk { reply, payload };
        if reply.is_simple() {
            if only_structured {
                return Err((
                    -libc::EINVAL,
                    Error::generic(
                        "Protocol error: simple reply when structured reply chunk was expected",
                    ),
                ));
            }
            *request_ret = -nbd_errno_to_system_errno(reply.error);
            let Some(t) = target else {
                return Ok(chunk(Vec::new()));
            };
            if *request_ret < 0 {
                return Ok(chunk(Vec::new()));
            }
            read_all(ioc, t.buf).map_err(|e| (-libc::EIO, e))?;
            return Ok(chunk(Vec::new()));
        }

        assert!(inner.info.mode >= NbdMode::Structured);
        if reply.typ == NBD_REPLY_TYPE_NONE {
            if reply.flags & NBD_REPLY_FLAG_DONE == 0 {
                return Err((
                    -libc::EINVAL,
                    Error::generic(
                        "Protocol error: NBD_REPLY_TYPE_NONE chunk without NBD_REPLY_FLAG_DONE \
                         flag set",
                    ),
                ));
            }
            if reply.length != 0 {
                return Err((
                    -libc::EINVAL,
                    Error::generic("Protocol error: NBD_REPLY_TYPE_NONE chunk with nonzero length"),
                ));
            }
            return Ok(chunk(Vec::new()));
        }
        if reply.typ == NBD_REPLY_TYPE_OFFSET_DATA {
            let Some(t) = target else {
                return Err((
                    -libc::EINVAL,
                    Error::generic("Unexpected NBD_REPLY_TYPE_OFFSET_DATA chunk"),
                ));
            };
            Self::receive_offset_data_payload(ioc, &reply, t)?;
            return Ok(chunk(Vec::new()));
        }
        let is_error = nbd_reply_type_is_error(reply.typ);
        // nbd_co_receive_structured_payload().
        let mut payload = Vec::new();
        if reply.length > 0 {
            if !want_payload && !is_error {
                return Err((-libc::EINVAL, Error::generic("Unexpected structured payload")));
            }
            if reply.length > NBD_MAX_MALLOC_PAYLOAD {
                return Err((-libc::EINVAL, Error::generic("Payload too large")));
            }
            payload = vec![0u8; reply.length as usize];
            nbd_read(ioc, &mut payload, Some("structured payload")).map_err(|e| (-libc::EIO, e))?;
        }
        if is_error {
            Self::parse_error_payload(&reply, &payload, request_ret)?;
            return Ok(chunk(Vec::new()));
        }
        Ok(chunk(payload))
    }

    /// `nbd_parse_error_payload()`.
    fn parse_error_payload(
        reply: &NbdReply,
        payload: &[u8],
        request_ret: &mut i32,
    ) -> std::result::Result<(), ChanError> {
        let inval = |m: &str| Err((-libc::EINVAL, Error::generic(m)));
        if reply.length < 6 {
            return inval("Protocol error: invalid payload for structured error");
        }
        let error = nbd_errno_to_system_errno(be32(payload));
        if error == 0 {
            return inval("Protocol error: server sent structured error chunk with error = 0");
        }
        *request_ret = -error;
        let message_size = u64::from(be16(&payload[4..]));
        if message_size > reply.length - 6 {
            return inval(
                "Protocol error: server sent structured error chunk with incorrect message size",
            );
        }
        Ok(())
    }

    /// `nbd_co_receive_offset_data_payload()`.
    fn receive_offset_data_payload(
        ioc: &mut NbdStream,
        reply: &NbdReply,
        t: &mut ReadTarget<'_>,
    ) -> std::result::Result<(), ChanError> {
        if reply.length <= 8 {
            return Err((
                -libc::EINVAL,
                Error::generic("Protocol error: invalid payload for NBD_REPLY_TYPE_OFFSET_DATA"),
            ));
        }
        let offset = nbd_read64(ioc, "OFFSET_DATA offset").map_err(|e| (-libc::EIO, e))?;
        let data_size = reply.length - 8;
        let size = t.buf.len() as u64;
        if offset < t.offset || data_size > size || offset > t.offset + size - data_size {
            return Err((
                -libc::EINVAL,
                Error::generic("Protocol error: server sent chunk exceeding requested region"),
            ));
        }
        let start = (offset - t.offset) as usize;
        read_all(ioc, &mut t.buf[start..start + data_size as usize]).map_err(|e| (-libc::EIO, e))
    }

    /// `nbd_reply_chunk_iter_receive()`: the next chunk the caller has to look at, or `None`
    /// when the reply is complete or failed.
    fn iter_receive(
        &self,
        inner: &mut Inner,
        iter: &mut Iter,
        target: Option<&mut ReadTarget<'_>>,
        want_payload: bool,
    ) -> Option<Chunk> {
        if iter.done {
            return None;
        }
        let mut request_ret = 0;
        // nbd_co_receive_one_chunk().
        let r = self.do_receive_one_chunk(
            inner,
            iter.only_structured,
            &mut request_ret,
            target,
            want_payload,
        );
        let chunk = match r {
            Ok(c) => {
                if request_ret < 0 {
                    iter.request_error(request_ret);
                }
                Some(c)
            }
            Err((ret, e)) => {
                self.channel_error(inner, ret);
                iter.channel_error(ret, e);
                None
            }
        };
        let chunk = chunk?;
        if chunk.reply.is_simple() || iter.ret < 0 {
            return None;
        }
        iter.only_structured = true;
        if chunk.reply.typ == NBD_REPLY_TYPE_NONE {
            return None;
        }
        if chunk.reply.flags & NBD_REPLY_FLAG_DONE != 0 {
            iter.done = true;
        }
        Some(chunk)
    }

    fn finish(inner: &mut Inner, iter: Iter) -> (i32, i32) {
        if let Some(e) = iter.err {
            inner.last_error = Some(e.message().to_string());
        }
        (iter.ret, iter.request_ret)
    }

    /// `nbd_co_receive_return_code()`.
    fn receive_return_code(&self, inner: &mut Inner) -> (i32, i32) {
        let mut iter = Iter::new(false);
        while self.iter_receive(inner, &mut iter, None, false).is_some() {}
        Self::finish(inner, iter)
    }

    /// `nbd_co_receive_cmdread_reply()`.
    fn receive_cmdread_reply(&self, inner: &mut Inner, t: &mut ReadTarget<'_>) -> (i32, i32) {
        let mut iter = Iter::new(inner.info.mode >= NbdMode::Structured);
        while let Some(c) = self.iter_receive(inner, &mut iter, Some(t), true) {
            match c.reply.typ {
                NBD_REPLY_TYPE_OFFSET_DATA => {}
                NBD_REPLY_TYPE_OFFSET_HOLE => {
                    if let Err((ret, e)) = Self::parse_offset_hole_payload(&c, t) {
                        self.channel_error(inner, ret);
                        iter.channel_error(ret, e);
                    }
                }
                typ => {
                    if !nbd_reply_type_is_error(typ) {
                        self.channel_error(inner, -libc::EINVAL);
                        let e = Error::generic(format!(
                            "Unexpected reply type: {typ} ({}) for CMD_READ",
                            nbd_reply_type_lookup(typ)
                        ));
                        iter.channel_error(-libc::EINVAL, e);
                    }
                }
            }
        }
        Self::finish(inner, iter)
    }

    /// `nbd_parse_offset_hole_payload()`.
    fn parse_offset_hole_payload(
        c: &Chunk,
        t: &mut ReadTarget<'_>,
    ) -> std::result::Result<(), ChanError> {
        if c.reply.length != 12 {
            return Err((
                -libc::EINVAL,
                Error::generic("Protocol error: invalid payload for NBD_REPLY_TYPE_OFFSET_HOLE"),
            ));
        }
        let offset = be64(&c.payload);
        let hole_size = u64::from(be32(&c.payload[8..]));
        let size = t.buf.len() as u64;
        if hole_size == 0
            || offset < t.offset
            || hole_size > size
            || offset > t.offset + size - hole_size
        {
            return Err((
                -libc::EINVAL,
                Error::generic("Protocol error: server sent chunk exceeding requested region"),
            ));
        }
        let start = (offset - t.offset) as usize;
        t.buf[start..start + hole_size as usize].fill(0);
        Ok(())
    }

    /// `nbd_parse_blockstatus_payload()`.
    fn parse_blockstatus_payload(
        &self,
        info: &NbdExportInfo,
        c: &Chunk,
        wide: bool,
        orig_length: u64,
        extent: &mut Extent,
    ) -> std::result::Result<(), ChanError> {
        let ext_len = if wide { 16 } else { 8 };
        let pay_len = 4 + if wide { 4 } else { 0 } + ext_len;
        if c.reply.length < pay_len {
            return Err((
                -libc::EINVAL,
                Error::generic("Protocol error: invalid payload for NBD_REPLY_TYPE_BLOCK_STATUS"),
            ));
        }
        let p = &c.payload;
        let context_id = be32(p);
        if info.context_id != context_id {
            return Err((
                -libc::EINVAL,
                Error::generic(format!(
                    "Protocol error: unexpected context id {} for NBD_REPLY_TYPE_BLOCK_STATUS, \
                     when negotiated context id is {}",
                    context_id as i32, info.context_id as i32
                )),
            ));
        }
        if wide {
            extent.length = be64(&p[8..]);
            extent.flags = be64(&p[16..]);
        } else {
            extent.length = u64::from(be32(&p[4..]));
            extent.flags = u64::from(be32(&p[8..]));
        }
        if extent.length == 0 {
            return Err((
                -libc::EINVAL,
                Error::generic("Protocol error: server sent status chunk with zero length"),
            ));
        }
        // Work around servers that report unaligned status at the end of an unaligned file.
        let min = u64::from(info.min_block);
        if min != 0 && extent.length % min != 0 {
            if extent.length > min {
                extent.length = extent.length / min * min;
            } else {
                extent.length = min;
                extent.flags = 0;
            }
        }
        // Extra extents and status past the request are ignored.
        if extent.length > orig_length {
            extent.length = orig_length;
        }
        if self.alloc_depth && extent.flags > 2 {
            extent.flags = 2;
        }
        Ok(())
    }

    /// `nbd_co_receive_blockstatus_reply()`.
    fn receive_blockstatus_reply(
        &self,
        inner: &mut Inner,
        length: u64,
        extent: &mut Extent,
    ) -> (i32, i32) {
        let mut iter = Iter::new(false);
        let mut received = false;
        let info = inner.info.clone();
        while let Some(c) = self.iter_receive(inner, &mut iter, None, true) {
            match c.reply.typ {
                NBD_REPLY_TYPE_BLOCK_STATUS | NBD_REPLY_TYPE_BLOCK_STATUS_EXT => {
                    let wide = c.reply.typ == NBD_REPLY_TYPE_BLOCK_STATUS_EXT;
                    if received {
                        self.channel_error(inner, -libc::EINVAL);
                        iter.channel_error(
                            -libc::EINVAL,
                            Error::generic("Several BLOCK_STATUS chunks in reply"),
                        );
                    }
                    received = true;
                    if let Err((ret, e)) =
                        self.parse_blockstatus_payload(&info, &c, wide, length, extent)
                    {
                        self.channel_error(inner, ret);
                        iter.channel_error(ret, e);
                    }
                }
                typ => {
                    if !nbd_reply_type_is_error(typ) {
                        self.channel_error(inner, -libc::EINVAL);
                        let e = Error::generic(format!(
                            "Unexpected reply type: {typ} ({}) for CMD_BLOCK_STATUS",
                            nbd_reply_type_lookup(typ)
                        ));
                        iter.channel_error(-libc::EINVAL, e);
                    }
                }
            }
        }
        if extent.length == 0 && iter.request_ret == 0 {
            iter.channel_error(
                -libc::EIO,
                Error::generic("Server did not reply with any status extents"),
            );
        }
        Self::finish(inner, iter)
    }

    fn will_reconnect(inner: &Inner) -> bool {
        inner.state == ClientState::ConnectingWait
    }

    /// `nbd_co_request()`.
    fn co_request(&self, mut req: NbdRequest, data: Option<&[u8]>) -> io::Result<()> {
        let mut inner = self.lock();
        let mut request_ret = 0;
        loop {
            let mut ret = self.send_request(&mut inner, &mut req, data);
            if ret >= 0 {
                (ret, request_ret) = self.receive_return_code(&mut inner);
            }
            if !(ret < 0 && Self::will_reconnect(&inner)) {
                return neg_to_io(if ret != 0 { ret } else { request_ret });
            }
        }
    }

    /// `nbd_client_co_preadv()` for one request of at most `max_transfer` bytes.
    fn preadv(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let bytes = buf.len() as u64;
        if bytes == 0 {
            return Ok(());
        }
        let mut inner = self.lock();
        let size = inner.info.size;
        // The block layer rounds the size up to whole sectors, so reads may go past the end
        // of the export by a little: pad them with zeroes.
        if offset >= size {
            buf.fill(0);
            return Ok(());
        }
        let mut len = bytes;
        if offset + bytes > size {
            let slop = offset + bytes - size;
            buf[(bytes - slop) as usize..].fill(0);
            len -= slop;
        }
        let buf = &mut buf[..len as usize];
        let mut req = NbdRequest { typ: NBD_CMD_READ, from: offset, len, ..Default::default() };
        let mut request_ret = 0;
        loop {
            let mut ret = self.send_request(&mut inner, &mut req, None);
            if ret >= 0 {
                let mut t = ReadTarget { buf: &mut *buf, offset };
                (ret, request_ret) = self.receive_cmdread_reply(&mut inner, &mut t);
            }
            if !(ret < 0 && Self::will_reconnect(&inner)) {
                return neg_to_io(if ret != 0 { ret } else { request_ret });
            }
        }
    }

    /// Reads `buf.len()` bytes at `offset`, in requests of at most
    /// [`NbdClient::max_transfer`] bytes.
    pub fn pread(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let max = self.max_transfer() as usize;
        let mut done = 0;
        for piece in buf.chunks_mut(max) {
            self.preadv(offset + done, piece)?;
            done += piece.len() as u64;
        }
        Ok(())
    }

    /// `nbd_client_co_pwritev()`, split into requests of at most
    /// [`NbdClient::max_transfer`] bytes. `fua` asks for the data to be on stable storage
    /// when the call returns.
    pub fn pwrite(&self, offset: u64, buf: &[u8], fua: bool) -> io::Result<()> {
        let info = self.info();
        if info.flags & NBD_FLAG_READ_ONLY != 0 {
            return Err(errno(libc::EACCES));
        }
        let mut flags = 0;
        if fua && info.flags & NBD_FLAG_SEND_FUA != 0 {
            flags |= NBD_CMD_FLAG_FUA;
        }
        let max = self.max_transfer() as usize;
        let mut done = 0u64;
        for piece in buf.chunks(max) {
            let req = NbdRequest {
                typ: NBD_CMD_WRITE,
                from: offset + done,
                len: piece.len() as u64,
                flags,
                ..Default::default()
            };
            self.co_request(req, Some(piece))?;
            done += piece.len() as u64;
        }
        if fua && flags == 0 {
            self.flush()?;
        }
        Ok(())
    }

    /// `nbd_client_co_pwrite_zeroes()`: `NBD_CMD_WRITE_ZEROES`, with `NBD_CMD_FLAG_NO_HOLE`
    /// unless `may_unmap` and `NBD_CMD_FLAG_FAST_ZERO` for `fast`. Fails with `ENOTSUP` when
    /// the server cannot do it, like the driver, so the caller can write zeroes instead.
    pub fn pwrite_zeroes(
        &self,
        offset: u64,
        bytes: u64,
        may_unmap: bool,
        fua: bool,
        fast: bool,
    ) -> io::Result<()> {
        let info = self.info();
        if info.flags & NBD_FLAG_READ_ONLY != 0 {
            return Err(errno(libc::EACCES));
        }
        if info.flags & NBD_FLAG_SEND_WRITE_ZEROES == 0 {
            return Err(errno(libc::ENOTSUP));
        }
        let mut flags = 0;
        if fua && info.flags & NBD_FLAG_SEND_FUA != 0 {
            flags |= NBD_CMD_FLAG_FUA;
        }
        if !may_unmap {
            flags |= NBD_CMD_FLAG_NO_HOLE;
        }
        if fast {
            if info.flags & NBD_FLAG_SEND_FAST_ZERO == 0 {
                return Err(errno(libc::ENOTSUP));
            }
            flags |= NBD_CMD_FLAG_FAST_ZERO;
        }
        if bytes == 0 {
            return Ok(());
        }
        let (_, max_zero, _) = self.limits();
        let max = if max_zero == 0 { u64::MAX } else { max_zero };
        let mut done = 0;
        while done < bytes {
            let n = (bytes - done).min(max);
            let req = NbdRequest {
                typ: NBD_CMD_WRITE_ZEROES,
                from: offset + done,
                len: n,
                flags,
                ..Default::default()
            };
            self.co_request(req, None)?;
            done += n;
        }
        if fua && flags & NBD_CMD_FLAG_FUA == 0 {
            self.flush()?;
        }
        Ok(())
    }

    /// `nbd_client_co_flush()`.
    pub fn flush(&self) -> io::Result<()> {
        if self.info().flags & NBD_FLAG_SEND_FLUSH == 0 {
            return Ok(());
        }
        self.co_request(NbdRequest { typ: NBD_CMD_FLUSH, ..Default::default() }, None)
    }

    /// `nbd_client_co_pdiscard()`.
    pub fn pdiscard(&self, offset: u64, bytes: u64) -> io::Result<()> {
        let info = self.info();
        if info.flags & NBD_FLAG_READ_ONLY != 0 {
            return Err(errno(libc::EACCES));
        }
        if info.flags & NBD_FLAG_SEND_TRIM == 0 || bytes == 0 {
            return Ok(());
        }
        let (max_discard, _, _) = self.limits();
        let max = if max_discard == 0 { u64::MAX } else { max_discard };
        let mut done = 0;
        while done < bytes {
            let n = (bytes - done).min(max);
            let req =
                NbdRequest { typ: NBD_CMD_TRIM, from: offset + done, len: n, ..Default::default() };
            self.co_request(req, None)?;
            done += n;
        }
        Ok(())
    }

    /// `NBD_CMD_CACHE`: ask the server to prefetch a range. QEMU's driver never sends it; this
    /// is here for tools. Does nothing when the server does not offer it.
    pub fn cache(&self, offset: u64, bytes: u64) -> io::Result<()> {
        if self.info().flags & NBD_FLAG_SEND_CACHE == 0 || bytes == 0 {
            return Ok(());
        }
        let max = u64::from(self.max_transfer());
        let mut done = 0;
        while done < bytes {
            let n = (bytes - done).min(max);
            let req = NbdRequest {
                typ: NBD_CMD_CACHE,
                from: offset + done,
                len: n,
                ..Default::default()
            };
            self.co_request(req, None)?;
            done += n;
        }
        Ok(())
    }

    /// `nbd_client_co_block_status()`: the status of the start of the range. Without a
    /// negotiated context everything is data.
    pub fn block_status(&self, offset: u64, bytes: u64) -> io::Result<NbdBlockStatus> {
        let mut inner = self.lock();
        let info = inner.info.clone();
        if !info.base_allocation {
            return Ok(NbdBlockStatus { data: true, zero: false, bytes });
        }
        // Past the end, in the part the block layer rounded the size up by.
        if offset >= info.size {
            return Ok(NbdBlockStatus { data: false, zero: true, bytes });
        }
        let mut len = bytes.min(info.size - offset);
        if info.mode < NbdMode::Extended {
            let align = u64::from(Self::alignment_of(&info));
            len = len.min((i32::MAX as u64) / align * align);
        }
        let mut req = NbdRequest {
            typ: NBD_CMD_BLOCK_STATUS,
            from: offset,
            len,
            flags: NBD_CMD_FLAG_REQ_ONE,
            ..Default::default()
        };
        let mut extent = Extent::default();
        let mut request_ret = 0;
        let ret = loop {
            let mut ret = self.send_request(&mut inner, &mut req, None);
            if ret >= 0 {
                extent = Extent::default();
                (ret, request_ret) = self.receive_blockstatus_reply(&mut inner, bytes, &mut extent);
            }
            if !(ret < 0 && Self::will_reconnect(&inner)) {
                break ret;
            }
        };
        if ret < 0 || request_ret < 0 {
            return Err(errno(-(if ret != 0 { ret } else { request_ret })));
        }
        assert!(extent.length != 0);
        Ok(NbdBlockStatus {
            data: extent.flags & NBD_STATE_HOLE == 0,
            zero: extent.flags & NBD_STATE_ZERO != 0,
            bytes: extent.length,
        })
    }

    /// `nbd_co_truncate()`: NBD cannot resize, but asking for the current size is fine.
    pub fn truncate(&self, offset: u64, exact: bool) -> Result<()> {
        let size = self.size();
        if offset != size && exact {
            return Err(Error::generic("Cannot resize NBD nodes"));
        }
        if offset > size {
            return Err(Error::generic("Cannot grow NBD nodes"));
        }
        Ok(())
    }

    /// `nbd_refresh_filename()`: the URI for the connection, if there is one.
    pub fn exact_filename(&self) -> Option<String> {
        let (host, port, path) = match &self.saddr.u {
            SocketAddressU::Inet(i) if i.ipv4.is_none() && i.ipv6.is_none() && i.to.is_none() => {
                (Some(i.host.as_str()), Some(i.port.as_str()), None)
            }
            SocketAddressU::Unix(u) => (None, None, Some(u.path.as_str())),
            _ => (None, None, None),
        };
        let s = match (path, host, &self.export) {
            (Some(p), _, Some(e)) => format!("nbd+unix:///{e}?socket={p}"),
            (Some(p), _, None) => format!("nbd+unix://?socket={p}"),
            (None, Some(h), Some(e)) => format!("nbd://{h}:{}/{e}", port.unwrap_or("")),
            (None, Some(h), None) => format!("nbd://{h}:{}", port.unwrap_or("")),
            _ => return None,
        };
        (s.len() < PATH_MAX).then_some(s)
    }

    /// `nbd_client_reopen_prepare()`.
    pub fn reopen_check(&self, read_only: bool) -> Result<()> {
        if !read_only && self.info().flags & NBD_FLAG_READ_ONLY != 0 {
            return Err(Error::generic("Can't reopen read-only NBD mount as read/write"));
        }
        Ok(())
    }

    /// `nbd_client_close()`: send `NBD_CMD_DISC` and hang up.
    pub fn close(&self) {
        let mut inner = self.lock();
        let mode = inner.info.mode;
        if let Some(mut s) = inner.ioc.take() {
            let req = NbdRequest { typ: NBD_CMD_DISC, mode, ..Default::default() };
            let _ = nbd_write(&mut s, &req.encode());
            s.shutdown();
        }
        inner.state = ClientState::Quit;
    }
}

impl Drop for NbdClient {
    fn drop(&mut self) {
        self.close();
    }
}

/// The `nbd` driver: an [`NbdClient`] behind the block layer's driver callbacks.
#[derive(Debug)]
pub(crate) struct NbdDriver {
    client: NbdClient,
}

/// `bdrv_nbd`, `bdrv_nbd_tcp` and `bdrv_nbd_unix`: one driver under three protocol names.
pub(crate) static NBD: DriverDef = DriverDef::protocol("nbd", "nbd", open_nbd)
    .with_parse_filename(nbd_parse_filename)
    .with_create_opts(nbd_co_create_opts);
/// `bdrv_nbd_tcp`.
pub(crate) static NBD_TCP: DriverDef = DriverDef::protocol("nbd", "nbd+tcp", open_nbd)
    .with_parse_filename(nbd_parse_filename)
    .with_create_opts(nbd_tcp_co_create_opts);
/// `bdrv_nbd_unix`.
pub(crate) static NBD_UNIX: DriverDef = DriverDef::protocol("nbd", "nbd+unix", open_nbd)
    .with_parse_filename(nbd_parse_filename)
    .with_create_opts(nbd_unix_co_create_opts);

/// `bdrv_co_create_opts_simple()` is what the three NBD drivers have for creating an image:
/// the export must exist, be large enough, and gets its first sector zeroed.
fn nbd_co_create_opts(filename: &str, options: &mut ruvm_qapi::QDict) -> Result<()> {
    crate::create::create_opts_simple(&NBD, filename, options)
}

fn nbd_tcp_co_create_opts(filename: &str, options: &mut ruvm_qapi::QDict) -> Result<()> {
    crate::create::create_opts_simple(&NBD_TCP, filename, options)
}

fn nbd_unix_co_create_opts(filename: &str, options: &mut ruvm_qapi::QDict) -> Result<()> {
    crate::create::create_opts_simple(&NBD_UNIX, filename, options)
}

fn open_nbd(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::Nbd(o) = opts else {
        unreachable!("the nbd driver gets nbd options");
    };
    let filename = args.meta.filename.clone();
    let d = nbd_open(&o, &mut args.flags, Some(&filename))?;
    Ok(Box::new(d))
}

/// `nbd_open()` for a node: connects, and makes the node read-only when the export is and
/// `auto-read-only` allows it.
pub(crate) fn nbd_open(
    opts: &BlockdevOptionsNbd,
    flags: &mut NodeFlags,
    filename: Option<&str>,
) -> Result<NbdDriver> {
    let mut read_only = flags.read_only;
    let client = NbdClient::open(opts, filename, &mut read_only, flags.auto_read_only)?;
    flags.read_only = read_only;
    Ok(NbdDriver { client })
}

impl Driver for NbdDriver {
    fn pread(&self, _bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.client.pread(offset, buf)
    }

    fn pwrite(&self, _bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        self.client.pwrite(offset, buf, false)
    }

    fn pwrite_flags(&self, _bs: &Node, offset: u64, buf: &[u8], flags: u32) -> io::Result<()> {
        self.client.pwrite(offset, buf, flags & BDRV_REQ_FUA != 0)
    }

    fn supported_write_flags(&self) -> u32 {
        if self.client.info().flags & NBD_FLAG_SEND_FUA != 0 { BDRV_REQ_FUA } else { 0 }
    }

    fn pwrite_zeroes(&self, bs: &Node, offset: u64, bytes: u64, may_unmap: bool) -> io::Result<()> {
        let flags = if may_unmap { BDRV_REQ_MAY_UNMAP } else { 0 };
        self.pwrite_zeroes_flags(bs, offset, bytes, flags)
    }

    fn pwrite_zeroes_flags(
        &self,
        _bs: &Node,
        offset: u64,
        bytes: u64,
        flags: u32,
    ) -> io::Result<()> {
        self.client.pwrite_zeroes(
            offset,
            bytes,
            flags & BDRV_REQ_MAY_UNMAP != 0,
            flags & BDRV_REQ_FUA != 0,
            flags & BDRV_REQ_NO_FALLBACK != 0,
        )
    }

    fn supported_zero_flags(&self) -> u32 {
        let f = self.client.info().flags;
        let mut z = 0;
        if f & NBD_FLAG_SEND_FUA != 0 {
            z |= BDRV_REQ_FUA;
        }
        if f & NBD_FLAG_SEND_WRITE_ZEROES != 0 {
            z |= BDRV_REQ_MAY_UNMAP;
            if f & NBD_FLAG_SEND_FAST_ZERO != 0 {
                z |= BDRV_REQ_NO_FALLBACK;
            }
        }
        z
    }

    fn pdiscard(&self, _bs: &Node, offset: u64, bytes: u64) -> io::Result<()> {
        self.client.pdiscard(offset, bytes)
    }

    fn flush_to_disk(&self, _bs: &Node) -> io::Result<()> {
        self.client.flush()
    }

    fn getlength(&self, _bs: &Node) -> io::Result<u64> {
        Ok(self.client.size())
    }

    fn truncate(&self, _bs: &Node, len: u64) -> Result<()> {
        self.client.truncate(len, false)
    }

    fn truncate_full(
        &self,
        _bs: &Node,
        offset: u64,
        exact: bool,
        _prealloc: ruvm_qapi::types::PreallocMode,
        _flags: u32,
    ) -> Result<()> {
        self.client.truncate(offset, exact)
    }

    fn block_status(
        &self,
        bs: &Node,
        _want: u32,
        offset: u64,
        bytes: u64,
    ) -> Option<io::Result<BlockStatus>> {
        Some(self.client.block_status(offset, bytes).map(|s| BlockStatus {
            ret: (if s.data { BDRV_BLOCK_DATA } else { 0 })
                | (if s.zero { BDRV_BLOCK_ZERO } else { 0 })
                | BDRV_BLOCK_OFFSET_VALID,
            pnum: s.bytes,
            map: offset,
            file: Some(bs.arc()),
        }))
    }

    fn refresh_limits(&self, _bs: &Node, bl: &mut BlockLimits) -> Result<()> {
        let (max_pdiscard, max_pwrite_zeroes, opt_block) = self.client.limits();
        bl.request_alignment = self.client.request_alignment();
        bl.max_pdiscard = max_pdiscard;
        bl.max_pwrite_zeroes = max_pwrite_zeroes;
        bl.max_transfer = self.client.max_transfer();
        bl.opt_transfer = bl.opt_transfer.max(opt_block);
        Ok(())
    }

    fn reopen_prepare(&self, _bs: &Node, state: &mut ReopenState) -> Option<Result<()>> {
        Some(self.client.reopen_check(state.flags.read_only))
    }

    fn exact_filename(&self, _bs: &Node) -> Option<String> {
        self.client.exact_filename()
    }

    fn close(&self, _bs: &Node) {
        self.client.close();
    }
}
