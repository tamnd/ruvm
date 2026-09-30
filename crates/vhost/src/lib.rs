// SPDX-License-Identifier: MIT OR Apache-2.0

//! The vhost kernel interface and the vhost-user frontend protocol.
//!
//! vhost moves the data path of a virtio device out of the VMM. The VMM keeps the control path:
//! it tells the backend where guest memory is, where each ring lives and which eventfds to use
//! for kicks and interrupts, and then gets out of the way. There are two ways to talk to a
//! backend and this crate covers both.
//!
//! - `user::Frontend` speaks the vhost-user protocol over a Unix socket to a backend running in
//!   another process. It is written from the protocol description (docs/interop/vhost-user.rst in
//!   the QEMU tree), not from QEMU's implementation. It needs Unix sockets, so it only exists on
//!   Unix hosts; the message encoding in [`user::message`] builds everywhere.
//! - `kernel::VhostKernel` drives the in-kernel backends in `/dev/vhost-net` and
//!   `/dev/vhost-vsock` through the ioctls in the Linux UAPI header `linux/vhost.h`. It only
//!   exists on Linux.
//!
//! Both implement [`VhostBackend`], the handful of operations every vhost device needs, so the
//! virtio device models can drive either without caring which one they have.
//!
//! vhost-vdpa and VDUSE are not here yet.

#![deny(unsafe_code)]

use std::fmt;
use std::io;

/// A borrowed file: a file descriptor on Unix, a handle on Windows. Memory regions, logs and
/// eventfds are passed to backends as these.
#[cfg(unix)]
pub type BorrowedFile<'a> = std::os::fd::BorrowedFd<'a>;
/// A borrowed file: a file descriptor on Unix, a handle on Windows. Memory regions, logs and
/// eventfds are passed to backends as these.
#[cfg(windows)]
pub type BorrowedFile<'a> = std::os::windows::io::BorrowedHandle<'a>;

#[cfg(target_os = "linux")]
pub mod kernel;
pub mod user;

/// `VHOST_F_LOG_ALL`: the backend can log every write it makes to guest memory, which live
/// migration needs.
pub const VHOST_F_LOG_ALL: u64 = 1 << 26;

/// `VHOST_USER_F_PROTOCOL_FEATURES`: the backend understands `GET_PROTOCOL_FEATURES` and
/// `SET_PROTOCOL_FEATURES`. This reuses virtio feature bit 30, which no virtio device offers.
pub const VHOST_USER_F_PROTOCOL_FEATURES: u64 = 1 << 30;

/// `VHOST_VRING_F_LOG`: in [`VringAddr::flags`], log writes to the used ring at
/// [`VringAddr::log_guest_addr`].
pub const VHOST_VRING_F_LOG: u32 = 1 << 0;

/// Everything that can go wrong talking to a vhost backend.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The socket or device file returned an error.
    Io(io::Error),
    /// The peer closed the connection between two messages.
    Disconnected,
    /// The peer closed the connection partway through a message.
    ShortMessage {
        /// Bytes the message should have had.
        expected: usize,
        /// Bytes that arrived before the connection closed.
        got: usize,
    },
    /// A message header carried a protocol version other than 1.
    BadVersion {
        /// The whole flags word from the header.
        flags: u32,
    },
    /// A reply came back for a different request than the one sent.
    UnexpectedReply {
        /// The request that was sent.
        expected: u32,
        /// The request the reply was for.
        got: u32,
    },
    /// A reply did not have the reply flag set, or a request from the backend did.
    BadFlags {
        /// The request the message was for.
        request: u32,
        /// The whole flags word from the header.
        flags: u32,
    },
    /// A reply payload had the wrong size for its request.
    BadPayloadSize {
        /// The request the reply was for.
        request: u32,
        /// The size the request calls for.
        expected: usize,
        /// The size that arrived.
        got: usize,
    },
    /// A message header announced a payload larger than any the protocol defines.
    PayloadTooLarge(usize),
    /// The backend answered a request with a non-zero status, either in a `REPLY_ACK` or by
    /// returning an empty config space.
    BackendFailed {
        /// The request that failed.
        request: u32,
        /// The status the backend returned.
        status: u64,
    },
    /// The request needs a feature or protocol feature that was not negotiated.
    NotNegotiated(&'static str),
    /// More file descriptors than one message may carry.
    TooManyFds {
        /// Descriptors the caller passed.
        count: usize,
        /// The most one message can carry.
        max: usize,
    },
    /// More memory regions than `SET_MEM_TABLE` can describe.
    TooManyRegions {
        /// Regions the caller passed.
        count: usize,
        /// The most the table holds.
        max: usize,
    },
    /// The caller passed something the protocol cannot express.
    InvalidArgument(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "vhost I/O error: {e}"),
            Error::Disconnected => f.write_str("vhost backend closed the connection"),
            Error::ShortMessage { expected, got } => {
                write!(f, "vhost backend closed the connection after {got} of {expected} bytes")
            }
            Error::BadVersion { flags } => {
                write!(f, "vhost-user message has flags {flags:#x}, expected version 1")
            }
            Error::UnexpectedReply { expected, got } => {
                write!(f, "vhost-user reply is for request {got}, expected {expected}")
            }
            Error::BadFlags { request, flags } => {
                write!(f, "vhost-user message for request {request} has bad flags {flags:#x}")
            }
            Error::BadPayloadSize { request, expected, got } => write!(
                f,
                "vhost-user reply to request {request} has a {got} byte payload, expected {expected}"
            ),
            Error::PayloadTooLarge(size) => {
                write!(f, "vhost-user message announces a {size} byte payload")
            }
            Error::BackendFailed { request, status } => {
                write!(f, "vhost-user backend failed request {request} with status {status}")
            }
            Error::NotNegotiated(what) => write!(f, "vhost backend did not negotiate {what}"),
            Error::TooManyFds { count, max } => {
                write!(f, "{count} file descriptors in one vhost-user message, at most {max}")
            }
            Error::TooManyRegions { count, max } => {
                write!(f, "{count} memory regions, the vhost memory table holds at most {max}")
            }
            Error::InvalidArgument(what) => write!(f, "invalid vhost argument: {what}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

/// The result of a vhost operation.
pub type Result<T> = std::result::Result<T, Error>;

/// One region of guest memory as a vhost backend sees it.
///
/// vhost-user backends map `fd` from `mmap_offset` for `memory_size` bytes, so every region
/// sent over vhost-user needs one. The kernel backends share the VMM's address space and use
/// `userspace_addr` directly, so they ignore `fd` and `mmap_offset`.
#[derive(Debug, Clone, Copy)]
pub struct MemoryRegion<'a> {
    /// Guest physical address of the start of the region.
    pub guest_phys_addr: u64,
    /// Size in bytes.
    pub memory_size: u64,
    /// Where the region is mapped in the VMM.
    pub userspace_addr: u64,
    /// Offset of the region in `fd`.
    pub mmap_offset: u64,
    /// The file backing the region.
    pub fd: Option<BorrowedFile<'a>>,
}

impl MemoryRegion<'_> {
    fn overlaps(&self, other: &MemoryRegion<'_>) -> bool {
        fn overlap(a: u64, a_len: u64, b: u64, b_len: u64) -> bool {
            let a_end = a.saturating_add(a_len);
            let b_end = b.saturating_add(b_len);
            a < b_end && b < a_end
        }
        overlap(self.guest_phys_addr, self.memory_size, other.guest_phys_addr, other.memory_size)
            || overlap(
                self.userspace_addr,
                self.memory_size,
                other.userspace_addr,
                other.memory_size,
            )
    }
}

/// Check a memory table for the two rules every vhost backend relies on: no empty regions, and
/// no two regions that overlap in guest physical or in VMM address space.
pub fn check_regions(regions: &[MemoryRegion<'_>]) -> Result<()> {
    for (i, a) in regions.iter().enumerate() {
        if a.memory_size == 0 {
            return Err(Error::InvalidArgument("empty memory region"));
        }
        if regions[i + 1..].iter().any(|b| a.overlaps(b)) {
            return Err(Error::InvalidArgument("overlapping memory regions"));
        }
    }
    Ok(())
}

/// Where a ring lives, as `struct vhost_vring_addr` has it.
///
/// The ring addresses are VMM virtual addresses, unless the device negotiated
/// `VIRTIO_F_IOMMU_PLATFORM`, in which case they are I/O virtual addresses. `log_guest_addr` is a
/// guest physical address and only matters when `flags` has [`VHOST_VRING_F_LOG`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VringAddr {
    /// The ring's index.
    pub index: u32,
    /// [`VHOST_VRING_F_LOG`] or zero.
    pub flags: u32,
    /// Address of the descriptor table.
    pub desc_user_addr: u64,
    /// Address of the used ring.
    pub used_user_addr: u64,
    /// Address of the available ring.
    pub avail_user_addr: u64,
    /// Guest physical address of the used ring, for dirty logging.
    pub log_guest_addr: u64,
}

/// A shared memory dirty log, for backends that map the log themselves.
///
/// vhost-user backends that offer `LOG_SHMFD` take the log as a file descriptor plus the size
/// and offset of the log in it. The kernel backends take a VMM address instead and ignore this.
#[derive(Debug, Clone, Copy)]
pub struct LogRegion<'a> {
    /// The file holding the log.
    pub fd: BorrowedFile<'a>,
    /// Size of the log in bytes.
    pub size: u64,
    /// Offset of the log in `fd`.
    pub offset: u64,
}

/// The operations every vhost device needs, whichever kind of backend is behind it.
///
/// The order a device uses them in is the same for both: `set_owner`, feature negotiation,
/// `set_mem_table`, then for each ring `set_vring_num`, `set_vring_addr`, `set_vring_base`,
/// `set_vring_call` and `set_vring_kick`. `get_vring_base` stops a ring and returns where it
/// stopped. Eventfd arguments are `None` to unbind the ring from any eventfd.
pub trait VhostBackend {
    /// Claim the backend for this VMM.
    fn set_owner(&mut self) -> Result<()>;
    /// Give up ownership, dropping the backend's device state.
    fn reset_owner(&mut self) -> Result<()>;
    /// The virtio and vhost features the backend offers.
    fn get_features(&mut self) -> Result<u64>;
    /// Acknowledge the features the device will use.
    fn set_features(&mut self, features: u64) -> Result<()>;
    /// Tell the backend where guest memory is.
    fn set_mem_table(&mut self, regions: &[MemoryRegion<'_>]) -> Result<()>;
    /// Set the dirty log. `base` is the log's address for backends that share the VMM's address
    /// space, `region` the shared memory log for those that map it themselves.
    fn set_log_base(&mut self, base: u64, region: Option<LogRegion<'_>>) -> Result<()>;
    /// Set the eventfd the backend signals after writing to the log.
    fn set_log_fd(&mut self, fd: BorrowedFile<'_>) -> Result<()>;
    /// Set the size of ring `index`.
    fn set_vring_num(&mut self, index: u32, num: u32) -> Result<()>;
    /// Set where ring `addr.index` lives.
    fn set_vring_addr(&mut self, addr: &VringAddr) -> Result<()>;
    /// Set the next available ring index the backend will process.
    fn set_vring_base(&mut self, index: u32, base: u32) -> Result<()>;
    /// Stop ring `index` and return the next available ring index it would have processed.
    fn get_vring_base(&mut self, index: u32) -> Result<u32>;
    /// Set the eventfd the guest kicks when it adds buffers to ring `index`.
    fn set_vring_kick(&mut self, index: u32, fd: Option<BorrowedFile<'_>>) -> Result<()>;
    /// Set the eventfd the backend signals when it has used buffers from ring `index`.
    fn set_vring_call(&mut self, index: u32, fd: Option<BorrowedFile<'_>>) -> Result<()>;
    /// Set the eventfd the backend signals when ring `index` hits an error.
    fn set_vring_err(&mut self, index: u32, fd: Option<BorrowedFile<'_>>) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::{MemoryRegion, check_regions};

    fn region(gpa: u64, size: u64, uva: u64) -> MemoryRegion<'static> {
        MemoryRegion {
            guest_phys_addr: gpa,
            memory_size: size,
            userspace_addr: uva,
            mmap_offset: 0,
            fd: None,
        }
    }

    #[test]
    fn disjoint_regions_pass() {
        let regions = [region(0, 0x1000, 0x10_0000), region(0x1000, 0x1000, 0x20_0000)];
        assert!(check_regions(&regions).is_ok());
    }

    #[test]
    fn overlap_in_either_space_fails() {
        let gpa = [region(0, 0x2000, 0x10_0000), region(0x1000, 0x1000, 0x20_0000)];
        assert!(check_regions(&gpa).is_err());
        let uva = [region(0, 0x1000, 0x10_0000), region(0x1000, 0x1000, 0x10_0800)];
        assert!(check_regions(&uva).is_err());
        assert!(check_regions(&[region(0, 0, 0)]).is_err());
    }
}
