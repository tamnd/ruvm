// SPDX-License-Identifier: MIT OR Apache-2.0

//! Split and packed virtqueue processing, descriptor chains and notification suppression, independent of device and transport.
//!
//! This crate is the device side of a virtqueue as the VIRTIO 1.2 specification describes it in
//! sections 2.7 (split virtqueues) and 2.8 (packed virtqueues). It knows how the rings are laid out
//! in guest memory, how to take a buffer the driver made available, how to hand it back as used,
//! and when the driver wants to be told about it. It does not know which device it serves or which
//! transport carried the queue addresses.
//!
//! Guest memory is reached through the small [`GuestMemory`] trait, so the same code runs against
//! ruvm's address spaces, against another VMM's memory model, or against [`VecMemory`] in tests.
//!
//! A typical device loop looks like this:
//!
//! ```
//! use ruvm_virtio_queue::{GuestMemory, RingAddresses, SplitQueue, VecMemory};
//!
//! # fn main() -> Result<(), ruvm_virtio_queue::QueueError> {
//! let mem = VecMemory::new(0x10000);
//! let addrs = RingAddresses { desc_table: 0x1000, driver_area: 0x2000, device_area: 0x3000 };
//! let mut queue = SplitQueue::new(16, addrs)?;
//!
//! loop {
//!     queue.disable_notification(&mem)?;
//!     while let Some(chain) = queue.pop(&mem)? {
//!         let mut writer = chain.writer(&mem);
//!         writer.write_all(b"done")?;
//!         queue.add_used(&mem, chain.head(), writer.bytes_written() as u32)?;
//!     }
//!     if queue.needs_notification(&mem)? {
//!         // raise the queue interrupt here
//!     }
//!     // Re-enabling returns true when the driver slipped a buffer in meanwhile.
//!     if !queue.enable_notification(&mem)? {
//!         break;
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! The crate is written from the OASIS text and contains no code taken from QEMU. It is plain
//! safe Rust with no dependencies.

#![forbid(unsafe_code)]

mod chain;
mod consts;
mod error;
mod event;
mod memory;
mod packed;
mod split;

pub use chain::{Descriptor, DescriptorChain, Reader, Writer};
pub use consts::*;
pub use error::{MemoryError, QueueError};
pub use event::need_event;
pub use memory::{GuestMemory, VecMemory};
pub use packed::PackedQueue;
pub use split::SplitQueue;

/// The largest queue size either ring format allows.
///
/// The packed format keeps the wrap counter in bit 15 of the ring offsets, and the split format
/// needs a power of two that fits in 16 bits, so both stop at 32768.
pub const MAX_QUEUE_SIZE: u16 = 32768;

/// How many entries an indirect table may hold unless the device says otherwise.
///
/// The specification does not put a number on it. 1024 is what drivers in the wild stay under and
/// keeps a hostile table from making the device allocate without bound.
pub const DEFAULT_MAX_INDIRECT_LEN: u32 = 1024;

/// The three guest physical addresses a transport hands over for one queue.
///
/// The names follow the transport registers in the specification. For a split queue the driver
/// area is the available ring and the device area is the used ring. For a packed queue the
/// descriptor table is the descriptor ring, the driver area is the driver event suppression
/// structure and the device area is the device event suppression structure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RingAddresses {
    /// Where the descriptor table (split) or descriptor ring (packed) starts.
    pub desc_table: u64,
    /// The available ring (split) or the driver event suppression structure (packed).
    pub driver_area: u64,
    /// The used ring (split) or the device event suppression structure (packed).
    pub device_area: u64,
}
