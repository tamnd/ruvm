// SPDX-License-Identifier: GPL-2.0-or-later

//! Guest memory for the rings: an adapter that lets `ruvm-virtio-queue` read and write through a
//! `ruvm-mem` address space.
//!
//! The adapter is a separate type because neither [`GuestMemory`] nor [`AddressSpace`] lives in
//! this crate, so the trait cannot be implemented on the address space directly.

use std::sync::Arc;

use ruvm_mem::{AddressSpace, MemTxAttrs};
use ruvm_virtio_queue::{GuestMemory, MemoryError};

/// [`GuestMemory`] on top of an [`AddressSpace`].
///
/// Every access is a plain `address_space_read()` or `address_space_write()` with the attributes
/// given at construction. Any transaction result other than OK is reported as
/// [`MemoryError::OutOfRange`], which the queue code turns into a broken device, the same way
/// QEMU's failed descriptor mapping ends in `virtio_error()`.
#[derive(Clone, Debug)]
pub struct AddressSpaceMemory {
    space: Arc<AddressSpace>,
    attrs: MemTxAttrs,
}

impl AddressSpaceMemory {
    /// Accesses `space` with `MEMTXATTRS_UNSPECIFIED`, what virtio uses for its DMA.
    pub fn new(space: Arc<AddressSpace>) -> Self {
        Self::with_attrs(space, MemTxAttrs::UNSPECIFIED)
    }

    /// Accesses `space` with the given transaction attributes.
    pub fn with_attrs(space: Arc<AddressSpace>, attrs: MemTxAttrs) -> Self {
        AddressSpaceMemory { space, attrs }
    }

    /// The address space behind the adapter.
    pub fn space(&self) -> &Arc<AddressSpace> {
        &self.space
    }
}

impl GuestMemory for AddressSpaceMemory {
    fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemoryError> {
        if self.space.read(addr, self.attrs, buf).is_ok() {
            Ok(())
        } else {
            Err(MemoryError::OutOfRange { addr, len: buf.len() as u64 })
        }
    }

    fn write(&self, addr: u64, buf: &[u8]) -> Result<(), MemoryError> {
        if self.space.write(addr, self.attrs, buf).is_ok() {
            Ok(())
        } else {
            Err(MemoryError::OutOfRange { addr, len: buf.len() as u64 })
        }
    }
}
