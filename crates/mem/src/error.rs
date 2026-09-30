// SPDX-License-Identifier: MIT OR Apache-2.0

//! Errors from the memory API.
//!
//! Most of these are assertions in QEMU. A device model that trips one has a bug, but returning
//! it lets the caller report which device instead of taking the process down.

use std::fmt;

/// What went wrong in a call to the memory API.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemError {
    /// The region id is stale or was never valid.
    NoSuchRegion,
    /// The region is already a subregion of some container.
    AlreadyMapped(String),
    /// The region is not a subregion of the container it was removed from.
    NotASubregion(String),
    /// Adding the subregion would make a region contain itself, directly or through aliases.
    Cycle(String),
    /// The region is still used by an alias, as the root of an address space, or by a container.
    InUse(String),
    /// The operation needs a region of another kind, for example an alias offset on RAM.
    WrongKind(String),
    /// `memory_region_set_log()` only takes the VGA client.
    InvalidClient,
    /// A size that does not fit in guest RAM on this host, or above 2^64.
    TooLarge(u128),
    /// An offset or length outside a RAM block.
    OutOfRange,
    /// The listener id is not registered.
    NoSuchListener,
    /// The address space was destroyed.
    NoSuchAddressSpace,
    /// A listener refused `log_global_start`.
    Listener(String),
}

impl fmt::Display for MemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemError::NoSuchRegion => f.write_str("no such memory region"),
            MemError::AlreadyMapped(n) => write!(f, "memory region '{n}' is already mapped"),
            MemError::NotASubregion(n) => {
                write!(f, "memory region '{n}' is not a subregion of that container")
            }
            MemError::Cycle(n) => write!(f, "memory region '{n}' would contain itself"),
            MemError::InUse(n) => write!(f, "memory region '{n}' is still in use"),
            MemError::WrongKind(n) => {
                write!(f, "memory region '{n}' is not of the kind this needs")
            }
            MemError::InvalidClient => f.write_str("only the VGA client can be logged per region"),
            MemError::TooLarge(s) => write!(f, "size 0x{s:x} is too large"),
            MemError::OutOfRange => f.write_str("access outside the RAM block"),
            MemError::NoSuchListener => f.write_str("no such memory listener"),
            MemError::NoSuchAddressSpace => f.write_str("no such address space"),
            MemError::Listener(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for MemError {}
