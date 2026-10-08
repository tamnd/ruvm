// SPDX-License-Identifier: MIT OR Apache-2.0

//! Memory regions, address spaces, flat views and dirty tracking, the memory API of spec/05.
//!
//! A [`MemorySystem`] owns a tree of regions: RAM, ROM, ROM devices, MMIO, aliases, containers,
//! reservations and IOMMUs. Subregions are placed in their container at an offset and with a
//! priority, and where siblings overlap the higher priority wins, then the one added last. That
//! is the order `memory_region_add_subregion_overlap()` gives, and the rendering that turns the
//! tree into what the guest sees follows `render_memory_region()` and `flatview_simplify()`
//! closely, because any difference there is guest visible.
//!
//! Each [`AddressSpace`] publishes the rendered [`FlatView`] of its root. A view is a sorted list
//! of [`FlatRange`]s plus a sorted array of every range and hole start, which a lookup binary
//! searches (in Eytzinger order once the array is large). Views never change once published: a
//! topology change renders new ones at the end of the outermost transaction and swaps them in,
//! so readers take no lock and always see a whole map. The [`MemoryListener`]s registered on a
//! space are told what changed in the same order QEMU tells its listeners.
//!
//! Accesses through an address space follow system/physmem.c: RAM is copied directly and marked
//! dirty, device accesses are cut by [`memory_access_size`], checked by [`access_valid`] and then
//! split or widened to what the device implements, and holes read as zero with
//! `MEMTX_DECODE_ERROR`.
//!
//! Dirty memory is tracked per RAM block with one bitmap per [`DirtyClient`]. Writes set the
//! bits of the clients in the range's dirty log mask: VGA when a display asked with
//! [`MemorySystem::set_log`], MIGRATION while global dirty tracking is on, and CODE when the
//! system was configured for it.
//!
//! # Testing
//!
//! `tests/flatview_props.rs` builds random region trees, applies random edits and checks every
//! rendered view against a plain tree walk, along with the view's own invariants and what a
//! recording listener saw. It runs 256 cases by default. The case count comes from proptest's
//! `PROPTEST_CASES` variable, so the full run of the M1 exit criterion is:
//!
//! ```text
//! PROPTEST_CASES=1000000 cargo test -p ruvm-mem --release --test flatview_props
//! ```
//!
//! # Not done yet
//!
//! Incremental rendering, the per thread lookup cache, ioeventfd and coalesced MMIO, RAM device
//! regions, host mapped RAM with the guest pointer types of spec/05, bounce buffers and the
//! reentrancy guard for devices all come later. IOMMU regions translate but have no notifiers.

#![forbid(unsafe_code)]

mod access;
mod address_space;
mod attrs;
#[cfg(unix)]
pub mod cpr;
mod dirty;
mod error;
mod flatview;
mod iommu;
mod listener;
mod ram;
mod region;
mod system;

pub use access::{
    AccessConstraints, AccessSize, Endian, GuestErrorSink, MmioOps, access_valid, dispatch_read,
    dispatch_write, memory_access_size, set_guest_error_log,
};
pub use address_space::AddressSpace;
pub use attrs::{
    AccessCtx, MEMTX_ACCESS_ERROR, MEMTX_DECODE_ERROR, MEMTX_ERROR, MEMTX_OK, MemResult,
    MemTxAttrs, MemTxResult,
};
pub use dirty::{DirtyClient, DirtyMask, DirtySnapshot};
pub use error::MemError;
pub use flatview::{EYTZINGER_THRESHOLD, FlatRange, FlatView};
pub use iommu::{IommuAccessFlags, IommuOps, IommuTlbEntry};
pub use listener::{ListenerId, MemoryListener};
pub use ram::RamBlock;
pub use region::{RegionId, RegionInfo, RegionType};
pub use system::{
    GLOBAL_DIRTY_DIRTY_RATE, GLOBAL_DIRTY_LIMIT, GLOBAL_DIRTY_MIGRATION, MemoryConfig,
    MemorySystem, Transaction,
};
