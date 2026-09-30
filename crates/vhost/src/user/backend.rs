// SPDX-License-Identifier: MIT OR Apache-2.0

//! The backend request channel: a second socket, handed to the backend with
//! `SET_BACKEND_REQ_FD`, on which the backend sends requests to the VMM.

use std::fmt;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::os::unix::net::UnixStream;

use super::connection::Connection;
use super::message::{Header, Payload, REPLY_FLAG, VERSION, backend_request, u32_at};
use crate::{Error, Result};

/// A request a backend sent on the backend channel.
#[derive(Debug)]
#[non_exhaustive]
pub enum BackendRequest<'a> {
    /// `CONFIG_CHANGE_MSG`: the device config space changed and the VMM should fetch it again
    /// with `GET_CONFIG` and tell the guest.
    ConfigChange,
    /// `IOTLB_MSG`, with the raw `struct vhost_iotlb_msg`.
    IotlbMsg(&'a [u8]),
    /// `VRING_CALL`: the in-band version of signalling a ring's call eventfd.
    VringCall {
        /// The ring.
        index: u32,
    },
    /// `VRING_ERR`: the in-band version of signalling a ring's error eventfd.
    VringErr {
        /// The ring.
        index: u32,
    },
    /// Anything else, left to the handler.
    Other {
        /// The request code.
        request: u32,
        /// The payload.
        payload: &'a [u8],
        /// File descriptors that came with it. They are closed when the handler returns unless
        /// it takes them.
        fds: &'a mut Vec<OwnedFd>,
    },
}

/// The function that handles backend requests. It returns the status to send back when the
/// backend asked for a reply: zero for success, anything else for failure.
pub type BackendHandler = Box<dyn FnMut(BackendRequest<'_>) -> u64 + Send>;

/// The VMM's end of the backend request channel.
///
/// The channel does not run on its own. Poll it through [`AsFd`] and call
/// [`BackendChannel::handle_one`] when it is readable, or give it a thread and call
/// [`BackendChannel::run`].
pub struct BackendChannel {
    conn: Connection,
    handler: BackendHandler,
}

impl fmt::Debug for BackendChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BackendChannel").field("conn", &self.conn).finish_non_exhaustive()
    }
}

impl BackendChannel {
    /// Wrap the VMM's end of the channel.
    pub fn new(stream: UnixStream, handler: BackendHandler) -> Self {
        BackendChannel { conn: Connection::new(stream), handler }
    }

    /// Handle one request, blocking until it arrives. Returns `false` when the backend closed
    /// the channel.
    pub fn handle_one(&mut self) -> Result<bool> {
        let mut message = match self.conn.recv() {
            Ok(message) => message,
            Err(Error::Disconnected) => return Ok(false),
            Err(e) => return Err(e),
        };
        let header = message.header;
        if header.is_reply() {
            return Err(Error::BadFlags { request: header.request, flags: header.flags });
        }
        let payload = &message.payload;
        let short = |len: usize| -> Result<()> {
            if payload.len() < len {
                Err(Error::BadPayloadSize {
                    request: header.request,
                    expected: len,
                    got: payload.len(),
                })
            } else {
                Ok(())
            }
        };
        let request = match header.request {
            backend_request::CONFIG_CHANGE_MSG => BackendRequest::ConfigChange,
            backend_request::IOTLB_MSG => BackendRequest::IotlbMsg(payload),
            backend_request::VRING_CALL => {
                short(8)?;
                BackendRequest::VringCall { index: u32_at(payload, 0) }
            }
            backend_request::VRING_ERR => {
                short(8)?;
                BackendRequest::VringErr { index: u32_at(payload, 0) }
            }
            other => BackendRequest::Other { request: other, payload, fds: &mut message.fds },
        };
        let status = (self.handler)(request);
        if header.needs_reply() {
            let reply = Header { request: header.request, flags: VERSION | REPLY_FLAG, size: 8 };
            self.conn.send(reply, &Payload::default().u64(status).0, &[])?;
        }
        Ok(true)
    }

    /// Handle requests until the backend closes the channel.
    pub fn run(&mut self) -> Result<()> {
        while self.handle_one()? {}
        Ok(())
    }
}

impl AsFd for BackendChannel {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.conn.as_fd()
    }
}
