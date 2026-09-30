// SPDX-License-Identifier: MIT OR Apache-2.0

//! Error types for guest memory access and for ring processing.

use std::fmt;

/// A guest memory access that could not be done.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MemoryError {
    /// Some or all of the range is not backed by guest memory.
    OutOfRange {
        /// Start of the access.
        addr: u64,
        /// Length of the access in bytes.
        len: u64,
    },
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfRange { addr, len } => {
                write!(f, "guest memory access of {len} bytes at {addr:#x} is out of range")
            }
        }
    }
}

impl std::error::Error for MemoryError {}

/// Something the driver put in the rings, or the configuration of the queue, is not valid.
///
/// Apart from [`QueueError::ChainExhausted`], which only means a request was shorter than the
/// device expected, every one of these is a driver bug or a hostile driver. The specification's
/// answer is for the device to set `DEVICE_NEEDS_RESET` and stop using the queue. The queue state
/// is left as it was before the failing call, so nothing is half consumed.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum QueueError {
    /// Reading or writing the rings or a buffer failed.
    Memory(MemoryError),
    /// The queue size is zero, too large, or (for a split queue) not a power of two.
    InvalidSize(u16),
    /// One of the ring areas is not aligned the way the ring format requires.
    Misaligned {
        /// Which area: "descriptor table", "driver area" or "device area".
        area: &'static str,
        /// The address that was given.
        addr: u64,
        /// The alignment the format needs.
        align: u64,
    },
    /// The driver moved the available index further ahead than the queue has entries.
    AvailIndexTooFar {
        /// The next entry the device expected to read.
        next_avail: u16,
        /// The index the driver published.
        avail_idx: u16,
    },
    /// A head or `next` field points outside the descriptor table.
    DescriptorIndexOutOfRange {
        /// The index that was read.
        index: u16,
        /// How many entries the table has.
        table_len: u32,
    },
    /// The chain has more descriptors than the table can hold, which also catches loops.
    ChainTooLong {
        /// The number of descriptors allowed.
        limit: u32,
    },
    /// A descriptor has the indirect flag but `VIRTIO_F_INDIRECT_DESC` was not negotiated.
    IndirectNotNegotiated,
    /// A descriptor has both the indirect and the next flag.
    IndirectWithNext,
    /// A descriptor inside an indirect table points at another indirect table.
    NestedIndirect,
    /// An indirect table is empty, not a whole number of descriptors, or too large.
    InvalidIndirectLength(u32),
    /// A device readable descriptor follows a device writable one.
    ReadableAfterWritable,
    /// A buffer's address plus its length does not fit in 64 bits.
    BufferOverflow {
        /// Start of the buffer.
        addr: u64,
        /// Its length.
        len: u32,
    },
    /// `add_used` was given a split queue head outside the descriptor table.
    InvalidUsedHead {
        /// The head that was passed.
        head: u16,
        /// The queue size.
        size: u16,
    },
    /// `add_used` on a packed queue was given zero ring slots or more than the ring holds.
    InvalidRingSlots(u16),
    /// A ring offset or wrap state given to a setter is outside the ring.
    InvalidRingOffset(u16),
    /// A reader or writer ran out of buffer space before the whole request was copied.
    ChainExhausted {
        /// How many bytes the caller wanted to copy.
        wanted: usize,
        /// How many bytes were left in the chain.
        available: u64,
    },
}

impl fmt::Display for QueueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Memory(e) => write!(f, "{e}"),
            Self::InvalidSize(size) => write!(f, "invalid queue size {size}"),
            Self::Misaligned { area, addr, align } => {
                write!(f, "{area} address {addr:#x} is not aligned to {align} bytes")
            }
            Self::AvailIndexTooFar { next_avail, avail_idx } => {
                write!(f, "available index {avail_idx} is more than a queue ahead of {next_avail}")
            }
            Self::DescriptorIndexOutOfRange { index, table_len } => {
                write!(f, "descriptor index {index} is outside a table of {table_len}")
            }
            Self::ChainTooLong { limit } => {
                write!(f, "descriptor chain is longer than {limit} or loops")
            }
            Self::IndirectNotNegotiated => {
                write!(f, "indirect descriptor used without VIRTIO_F_INDIRECT_DESC")
            }
            Self::IndirectWithNext => {
                write!(f, "descriptor has both the indirect and the next flag")
            }
            Self::NestedIndirect => write!(f, "indirect table inside an indirect table"),
            Self::InvalidIndirectLength(len) => {
                write!(f, "indirect table length {len} is not valid")
            }
            Self::ReadableAfterWritable => {
                write!(f, "device readable descriptor after a device writable one")
            }
            Self::BufferOverflow { addr, len } => {
                write!(f, "buffer of {len} bytes at {addr:#x} wraps the address space")
            }
            Self::InvalidUsedHead { head, size } => {
                write!(f, "used head {head} is outside a queue of {size}")
            }
            Self::InvalidRingSlots(n) => write!(f, "{n} is not a valid number of ring slots"),
            Self::InvalidRingOffset(n) => write!(f, "ring offset {n} is outside the ring"),
            Self::ChainExhausted { wanted, available } => write!(
                f,
                "wanted {wanted} bytes but the descriptor chain has only {available} left"
            ),
        }
    }
}

impl std::error::Error for QueueError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Memory(e) => Some(e),
            _ => None,
        }
    }
}

impl From<MemoryError> for QueueError {
    fn from(e: MemoryError) -> Self {
        Self::Memory(e)
    }
}
