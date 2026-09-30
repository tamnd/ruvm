// SPDX-License-Identifier: MIT OR Apache-2.0

//! Memory listeners, `MemoryListener`: code outside the core that follows the memory map, such
//! as an accelerator mirroring RAM into memory slots or migration following dirty memory.

use crate::address_space::AddressSpace;
use crate::dirty::DirtyMask;
use crate::error::MemError;
use crate::flatview::FlatRange;

/// A registered listener, returned by
/// [`MemorySystem::register_listener`](crate::MemorySystem::register_listener).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ListenerId(pub(crate) u64);

/// Callbacks for changes to the memory map.
///
/// Every method has an empty default. A section is passed as the [`FlatRange`] it came from,
/// which carries what `MemoryRegionSection` does: the region, the offset in it, the address in the
/// space and the size.
///
/// Listeners are sorted by [`MemoryListener::priority`], lowest first, and a new one goes after
/// the others of the same priority. Callbacks that add things run in that order and callbacks
/// that take things away run in reverse, as in QEMU. The callbacks run with the memory system
/// locked, so they must not call back into [`MemorySystem`](crate::MemorySystem); doing so
/// panics. Reading and writing guest memory through an [`AddressSpace`] is fine.
pub trait MemoryListener: Send + Sync {
    /// A name for debugging, `name`.
    fn name(&self) -> &str {
        ""
    }

    /// The position among listeners, `priority`.
    fn priority(&self) -> i32 {
        0
    }

    /// A topology update starts, `begin`. Called for every listener, whatever its space.
    fn begin(&self) {}

    /// A topology update is done and the new views are published, `commit`.
    fn commit(&self) {}

    /// A range appeared, or an existing one changed its attributes, `region_add`.
    fn region_add(&self, space: &AddressSpace, section: &FlatRange) {
        let _ = (space, section);
    }

    /// A range went away, or is about to be added back with other attributes, `region_del`.
    fn region_del(&self, space: &AddressSpace, section: &FlatRange) {
        let _ = (space, section);
    }

    /// A range is still there unchanged, `region_nop`.
    fn region_nop(&self, space: &AddressSpace, section: &FlatRange) {
        let _ = (space, section);
    }

    /// Dirty logging of a range gained clients, `log_start`.
    fn log_start(&self, space: &AddressSpace, section: &FlatRange, old: DirtyMask, new: DirtyMask) {
        let _ = (space, section, old, new);
    }

    /// Dirty logging of a range lost clients, `log_stop`.
    fn log_stop(&self, space: &AddressSpace, section: &FlatRange, old: DirtyMask, new: DirtyMask) {
        let _ = (space, section, old, new);
    }

    /// Whether the listener syncs all its dirty state at once through
    /// [`MemoryListener::log_sync_global`] instead of one range at a time through
    /// [`MemoryListener::log_sync`]. In QEMU this is which of the two callbacks is set.
    fn log_sync_is_global(&self) -> bool {
        false
    }

    /// Copy the dirty bits the listener collected for a logged range into the RAM block's
    /// bitmaps, `log_sync`.
    fn log_sync(&self, space: &AddressSpace, section: &FlatRange) {
        let _ = (space, section);
    }

    /// Copy every dirty bit the listener collected, `log_sync_global`.
    fn log_sync_global(&self, last_stage: bool) {
        let _ = last_stage;
    }

    /// Global dirty tracking starts, `log_global_start`. An error stops the start and the
    /// listeners that already started are stopped again.
    fn log_global_start(&self) -> Result<(), MemError> {
        Ok(())
    }

    /// Global dirty tracking stops, `log_global_stop`.
    fn log_global_stop(&self) {}

    /// A global dirty sync finished, `log_global_after_sync`.
    fn log_global_after_sync(&self) {}
}
