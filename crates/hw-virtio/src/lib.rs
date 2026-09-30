// SPDX-License-Identifier: GPL-2.0-or-later

//! Virtio devices and their transports.
//!
//! - [`virtio`]: the device core from `hw/virtio/virtio.c`: status, feature negotiation, config
//!   space, queues and interrupts. Rings are handled by `ruvm-virtio-queue`.
//! - [`mmio`]: the virtio-mmio transport, legacy and virtio 1.0 register layouts.
//! - [`pci`]: the virtio-pci transport, modern, transitional and legacy, with MSI-X.
//! - [`rng`], [`console`] and [`blk`]: virtio-rng, a single port virtio-console and virtio-blk.
//! - [`memory`]: lets the rings live in a `ruvm-mem` address space.
//!
//! A device is put together in three steps: build the device model, realize it on guest memory
//! with [`VirtioBackend::new`], and hand the result to a transport such as [`VirtioMmio::new`].
//! The MMIO transport is then mapped into the guest physical address space and its interrupt
//! pin connected. [`VirtioPci::new`] instead registers a PCI function on a bus, and the guest
//! finds and programs it through config space like any other PCI device.
//!
//! The CCW transport, virtio-net and the other device types are not here yet.

#![forbid(unsafe_code)]

pub mod blk;
pub mod console;
pub mod memory;
pub mod mmio;
pub mod pci;
pub mod rng;
pub mod virtio;

pub use blk::{BlockBackend, MemBlockBackend, VirtioBlk, VirtioBlkConf};
pub use console::{ConsoleBackend, VirtioConsole};
pub use memory::AddressSpaceMemory;
pub use mmio::VirtioMmio;
pub use pci::{VirtioPci, VirtioPciProps, VirtioPciVariant};
pub use rng::{EntropySource, RandomFile, VirtioRng, VirtioRngConf};
pub use virtio::{
    SharedGuestMemory, VirtIODevice, VirtQueue, VirtioBackend, VirtioDeviceClass, VirtioTransport,
};
