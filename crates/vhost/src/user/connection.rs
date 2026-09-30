// SPDX-License-Identifier: MIT OR Apache-2.0

//! Framing vhost-user messages on a Unix stream socket, with file descriptors in `SCM_RIGHTS`
//! ancillary data.
//!
//! The protocol says file descriptors travel with the first byte of a message header, so they
//! go out with the first `sendmsg` of a message and are collected from every `recvmsg` that
//! reads part of a header. The payload is read with plain reads.

use std::io::{self, IoSlice, IoSliceMut, Read, Write};
use std::mem::MaybeUninit;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::os::unix::net::UnixStream;

use rustix::io::Errno;
use rustix::net::{
    RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, ReturnFlags, SendAncillaryBuffer,
    SendAncillaryMessage, SendFlags, recvmsg, sendmsg,
};

use super::message::{HEADER_SIZE, Header, MAX_FDS, MAX_PAYLOAD, VERSION};
use crate::{Error, Result};

/// One received message.
#[derive(Debug)]
pub struct Message {
    /// The header.
    pub header: Header,
    /// The payload, `header.size` bytes.
    pub payload: Vec<u8>,
    /// File descriptors that came with the header.
    pub fds: Vec<OwnedFd>,
}

/// One end of a vhost-user socket, either the main one or the backend request channel.
#[derive(Debug)]
pub struct Connection {
    stream: UnixStream,
}

impl Connection {
    /// Wrap a connected stream.
    pub fn new(stream: UnixStream) -> Self {
        Connection { stream }
    }

    /// The socket.
    pub fn stream(&self) -> &UnixStream {
        &self.stream
    }

    /// Send one message with up to [`MAX_FDS`] file descriptors.
    pub fn send(&self, header: Header, payload: &[u8], fds: &[BorrowedFd<'_>]) -> Result<()> {
        if fds.len() > MAX_FDS {
            return Err(Error::TooManyFds { count: fds.len(), max: MAX_FDS });
        }
        if payload.len() > MAX_PAYLOAD || header.size as usize != payload.len() {
            return Err(Error::PayloadTooLarge(payload.len()));
        }
        let mut bytes = Vec::with_capacity(HEADER_SIZE + payload.len());
        bytes.extend_from_slice(&header.to_bytes());
        bytes.extend_from_slice(payload);

        let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(MAX_FDS))];
        let mut control = SendAncillaryBuffer::new(&mut space);
        if !fds.is_empty() && !control.push(SendAncillaryMessage::ScmRights(fds)) {
            return Err(Error::TooManyFds { count: fds.len(), max: MAX_FDS });
        }
        let sent = loop {
            match sendmsg(&self.stream, &[IoSlice::new(&bytes)], &mut control, SendFlags::empty()) {
                Err(Errno::INTR) => continue,
                other => break other.map_err(io::Error::from)?,
            }
        };
        // A stream socket may take less than the whole message. The descriptors went with the
        // first byte, so the rest is plain data.
        (&self.stream).write_all(&bytes[sent..])?;
        Ok(())
    }

    /// Receive one message. Returns [`Error::Disconnected`] if the peer closed the socket
    /// cleanly between messages.
    pub fn recv(&self) -> Result<Message> {
        let mut raw = [0u8; HEADER_SIZE];
        let mut got = 0;
        let mut fds = Vec::new();
        while got < HEADER_SIZE {
            let n = self.recv_with_fds(&mut raw[got..], &mut fds)?;
            if n == 0 {
                return Err(if got == 0 {
                    Error::Disconnected
                } else {
                    Error::ShortMessage { expected: HEADER_SIZE, got }
                });
            }
            got += n;
        }
        let header = Header::from_bytes(&raw);
        if header.version() != VERSION {
            return Err(Error::BadVersion { flags: header.flags });
        }
        let size = header.size as usize;
        if size > MAX_PAYLOAD {
            return Err(Error::PayloadTooLarge(size));
        }
        let mut payload = vec![0u8; size];
        let mut read = 0;
        while read < size {
            match (&self.stream).read(&mut payload[read..]) {
                Ok(0) => {
                    return Err(Error::ShortMessage {
                        expected: HEADER_SIZE + size,
                        got: HEADER_SIZE + read,
                    });
                }
                Ok(n) => read += n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e.into()),
            }
        }
        Ok(Message { header, payload, fds })
    }

    fn recv_with_fds(&self, buf: &mut [u8], fds: &mut Vec<OwnedFd>) -> Result<usize> {
        let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(MAX_FDS))];
        let mut control = RecvAncillaryBuffer::new(&mut space);
        #[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
        let flags = RecvFlags::CMSG_CLOEXEC;
        #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "freebsd")))]
        let flags = RecvFlags::empty();
        let result = loop {
            let mut iov = [IoSliceMut::new(buf)];
            match recvmsg(&self.stream, &mut iov, &mut control, flags) {
                Err(Errno::INTR) => continue,
                other => break other.map_err(io::Error::from)?,
            }
        };
        for message in control.drain() {
            if let RecvAncillaryMessage::ScmRights(received) = message {
                fds.extend(received);
            }
        }
        if result.flags.contains(ReturnFlags::CTRUNC) {
            return Err(Error::TooManyFds { count: fds.len() + 1, max: MAX_FDS });
        }
        Ok(result.bytes)
    }
}

impl AsFd for Connection {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.stream.as_fd()
    }
}
