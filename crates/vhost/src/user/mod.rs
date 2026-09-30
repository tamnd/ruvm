// SPDX-License-Identifier: MIT OR Apache-2.0

//! The vhost-user frontend: the VMM side of the protocol.
//!
//! [`Frontend`] owns the socket to one backend and sends it requests one at a time, waiting for
//! each reply before the next request goes out, which is the only ordering the protocol allows.
//! It keeps track of what the two sides negotiated so a request that needs a feature the backend
//! does not have fails here rather than confusing the backend.
//!
//! When `REPLY_ACK` is negotiated every request that has no reply of its own goes out with
//! `need_reply` set, and the frontend waits for the backend's status. That turns a failure in the
//! backend into an error on the request that caused it rather than a silent divergence.

#[cfg(unix)]
mod backend;
#[cfg(unix)]
mod connection;
#[cfg(unix)]
mod frontend;
pub mod message;

#[cfg(unix)]
pub use backend::{BackendChannel, BackendHandler, BackendRequest};
#[cfg(unix)]
pub use connection::{Connection, Message};
#[cfg(unix)]
pub use frontend::Frontend;
