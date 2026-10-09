// SPDX-License-Identifier: GPL-2.0-or-later

//! KVM memory slots, `kvm_region_add()` and `kvm_region_del()` with `kvm_set_phys_mem()`, and
//! dirty logging on them, `kvm_log_start()`, `kvm_log_stop()`, `kvm_log_sync()` and
//! `kvm_log_sync_global()`.
//!
//! Each RAM backed flat range becomes one slot. Ranges are trimmed to host pages first, as
//! `kvm_align_section()` does, and anything that is not RAM stays out so the guest's accesses
//! to it exit to userspace.
//!
//! A slot whose range has any dirty client gets `KVM_MEM_LOG_DIRTY_PAGES`. Without a dirty ring
//! a sync reads the slot's bitmap with `KVM_GET_DIRTY_LOG`, which also clears it and write
//! protects the pages again. With a ring the sync is global: it flushes the rings into the slot
//! bitmaps kept in [`DirtyRings`] and copies those. Either way the bits go into the RAM block
//! for every client of the range but the TCG code one, as `kvm_slot_sync_dirty_pages()` does.

use std::sync::{Arc, Mutex, MutexGuard};

use kvm_bindings::{KVM_MEM_LOG_DIRTY_PAGES, KVM_MEM_READONLY, kvm_userspace_memory_region};
use kvm_ioctls::VmFd;
use ruvm_base::error_report;
use ruvm_mem::{
    AddressSpace, DirtyClient, DirtyMask, FlatRange, MemoryListener, RamBlock, RegionType,
};

use super::dirty::{DirtyRings, HOST_PAGE, set_dirty_lebitmap};
use super::os_error;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Slot {
    gpa: u64,
    size: u64,
    host: u64,
    flags: u32,
}

impl Slot {
    /// Whether `other` maps the same memory, whatever the flags. This is how QEMU finds a slot
    /// again with `kvm_lookup_matching_slot()`.
    fn same_place(&self, other: &Slot) -> bool {
        self.gpa == other.gpa && self.size == other.size && self.host == other.host
    }

    fn pages(&self) -> u64 {
        self.size / HOST_PAGE
    }

    fn logging(&self) -> bool {
        self.flags & KVM_MEM_LOG_DIRTY_PAGES != 0
    }
}

/// What a slot maps: the RAM block, kept alive while the slot exists, where in it the slot
/// starts, and the dirty clients of the range.
#[derive(Clone, Debug)]
struct Entry {
    slot: Slot,
    block: Arc<RamBlock>,
    offset: u64,
    mask: DirtyMask,
}

impl Entry {
    /// `kvm_slot_sync_dirty_pages()`: copies a slot bitmap into the block.
    fn sync(&self, bits: &[u64]) {
        set_dirty_lebitmap(
            &self.block,
            self.offset,
            bits,
            self.slot.pages(),
            self.mask.without(DirtyClient::Code),
        );
    }
}

/// Slot number to what it maps.
type Table = Vec<Option<Entry>>;

/// `kvm_mem_flags()`.
fn slot_flags(readonly: bool, mask: DirtyMask) -> u32 {
    let mut flags = 0;
    if !mask.is_empty() {
        flags |= KVM_MEM_LOG_DIRTY_PAGES;
    }
    if readonly {
        flags |= KVM_MEM_READONLY;
    }
    flags
}

/// Keeps KVM's memory slots in step with an address space, `KVMMemoryListener`.
#[derive(Debug)]
pub struct SlotListener {
    vm: Arc<VmFd>,
    readonly_mem: bool,
    rings: Arc<DirtyRings>,
    slots: Mutex<Table>,
}

impl SlotListener {
    pub(crate) fn new(
        vm: Arc<VmFd>,
        readonly_mem: bool,
        nr_slots: usize,
        rings: Arc<DirtyRings>,
    ) -> Self {
        SlotListener { vm, readonly_mem, rings, slots: Mutex::new(vec![None; nr_slots]) }
    }

    fn lock(&self) -> MutexGuard<'_, Table> {
        self.slots.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The slots in use as `(guest address, size, read only)`, for tests and `info kvm`.
    pub fn slots(&self) -> Vec<(u64, u64, bool)> {
        self.lock()
            .iter()
            .flatten()
            .map(|e| (e.slot.gpa, e.slot.size, e.slot.flags & KVM_MEM_READONLY != 0))
            .collect()
    }

    /// What a range maps to, or `None` when it must trap. This is the filter at the top of
    /// `kvm_set_phys_mem()`: RAM always gets a slot, read only when the region is and the host
    /// can do it, and a ROM device gets a read only slot only in romd mode on such a host.
    fn entry_for(&self, fr: &FlatRange) -> Option<Entry> {
        let block = fr.ram_block()?;
        let readonly = match fr.region_type() {
            RegionType::Ram => fr.readonly() && self.readonly_mem,
            RegionType::RomDevice if self.readonly_mem && fr.romd_mode() => true,
            _ => return None,
        };
        let size = u64::try_from(fr.size()).ok()?;
        let start = fr.addr().next_multiple_of(HOST_PAGE);
        let delta = start - fr.addr();
        if delta >= size {
            return None;
        }
        let size = (size - delta) & !(HOST_PAGE - 1);
        if size == 0 {
            return None;
        }
        let offset = fr.offset_in_region() + delta;
        let host = block.host_addr() as u64 + offset;
        let mask = fr.dirty_log_mask();
        let flags = slot_flags(readonly, mask);
        Some(Entry {
            slot: Slot { gpa: start, size, host, flags },
            block: Arc::clone(block),
            offset,
            mask,
        })
    }

    /// The slot that maps `fr`, with its number.
    fn find<'a>(slots: &'a mut Table, want: &Entry) -> Option<(usize, &'a mut Entry)> {
        slots
            .iter_mut()
            .enumerate()
            .find_map(|(i, e)| e.as_mut().filter(|e| e.slot.same_place(&want.slot)).map(|e| (i, e)))
    }

    fn set(&self, index: usize, slot: Slot, size: u64) {
        let region = kvm_userspace_memory_region {
            slot: index as u32,
            flags: slot.flags,
            guest_phys_addr: slot.gpa,
            memory_size: size,
            userspace_addr: slot.host,
        };
        // SAFETY: `host` points `size` bytes into a RamBlock mapping, and the slot table holds an
        // Arc to that block until the slot is removed again, so the mapping outlives the slot.
        // Removing a slot passes size 0, which the kernel never dereferences.
        if let Err(e) = unsafe { self.vm.set_user_memory_region(region) } {
            // QEMU aborts here too. A guest map the kernel refuses cannot be run.
            panic!("kvm_set_phys_mem: error registering slot: {}", crate::strerror(&os_error(e)));
        }
    }

    /// `kvm_slot_get_dirty_log()` followed by `kvm_slot_sync_dirty_pages()`. A slot the kernel
    /// no longer knows is not an error.
    fn get_dirty_log(&self, index: usize, entry: &Entry) {
        match self.vm.get_dirty_log(index as u32, entry.slot.size as usize) {
            Ok(bits) => entry.sync(&bits),
            Err(e) if e.errno() == libc::ENOENT => {}
            Err(e) => error_report(&format!(
                "kvm_slot_get_dirty_log: KVM_GET_DIRTY_LOG failed: {}",
                crate::strerror(&os_error(e))
            )),
        }
    }

    /// `kvm_slot_update_flags()` after the range's dirty clients changed to `mask`.
    fn update_flags(&self, fr: &FlatRange, mask: DirtyMask) {
        let Some(want) = self.entry_for(fr) else { return };
        let mut slots = self.lock();
        let Some((index, entry)) = Self::find(&mut slots, &want) else { return };
        entry.mask = mask;
        let readonly = entry.slot.flags & KVM_MEM_READONLY != 0;
        let flags = slot_flags(readonly, mask);
        if flags == entry.slot.flags {
            return;
        }
        entry.slot.flags = flags;
        let slot = entry.slot;
        self.set(index, slot, slot.size);
        if self.rings.enabled() {
            if slot.logging() {
                self.rings.init_slot(index, slot.pages());
            } else {
                self.rings.drop_slot(index);
            }
        }
    }
}

impl MemoryListener for SlotListener {
    fn name(&self) -> &str {
        "kvm-memory"
    }

    fn priority(&self) -> i32 {
        10
    }

    fn region_add(&self, _space: &AddressSpace, fr: &FlatRange) {
        let Some(entry) = self.entry_for(fr) else { return };
        let mut slots = self.lock();
        let Some(index) = slots.iter().position(Option::is_none) else {
            panic!("kvm_alloc_slot: no free slot anymore");
        };
        self.set(index, entry.slot, entry.slot.size);
        if entry.slot.logging() && self.rings.enabled() {
            self.rings.init_slot(index, entry.slot.pages());
        }
        slots[index] = Some(entry);
    }

    /// `kvm_set_phys_mem()` on removal: the dirty pages of a logged slot are synced first, so
    /// that nothing the guest wrote there is lost.
    fn region_del(&self, _space: &AddressSpace, fr: &FlatRange) {
        let Some(want) = self.entry_for(fr) else { return };
        let mut slots = self.lock();
        let Some((index, entry)) = Self::find(&mut slots, &want) else { return };
        let entry = entry.clone();
        if entry.slot.logging() {
            if self.rings.enabled() {
                if let Some(bits) = self.rings.retire_slot(index) {
                    entry.sync(&bits);
                }
                if self.rings.with_bitmap() {
                    self.get_dirty_log(index, &entry);
                }
            } else {
                self.get_dirty_log(index, &entry);
            }
        }
        self.set(index, entry.slot, 0);
        slots[index] = None;
    }

    fn log_start(&self, _space: &AddressSpace, fr: &FlatRange, _old: DirtyMask, new: DirtyMask) {
        self.update_flags(fr, new);
    }

    fn log_stop(&self, _space: &AddressSpace, fr: &FlatRange, _old: DirtyMask, new: DirtyMask) {
        self.update_flags(fr, new);
    }

    /// With a dirty ring the listener syncs everything at once, as QEMU picks `log_sync_global`
    /// over `log_sync` in `kvm_memory_listener_register()`.
    fn log_sync_is_global(&self) -> bool {
        self.rings.enabled()
    }

    /// `kvm_log_sync()` and `kvm_physical_sync_dirty_bitmap()`.
    fn log_sync(&self, _space: &AddressSpace, fr: &FlatRange) {
        let Some(want) = self.entry_for(fr) else { return };
        let mut slots = self.lock();
        let Some((index, entry)) = Self::find(&mut slots, &want) else { return };
        if entry.slot.logging() {
            let entry = entry.clone();
            self.get_dirty_log(index, &entry);
        }
    }

    /// `kvm_log_sync_global()`: flushes the rings, then copies every logged slot's bitmap into
    /// its block and clears it. On the last pass of a migration the backup bitmap, when there
    /// is one, is read too, for pages the kernel dirtied without a vCPU.
    fn log_sync_global(&self, last_stage: bool) {
        self.rings.flush();
        let slots = self.lock();
        for (index, entry) in slots.iter().enumerate() {
            let Some(entry) = entry.as_ref().filter(|e| e.slot.logging()) else { continue };
            if let Some(bits) = self.rings.take_slot(index) {
                entry.sync(&bits);
            }
            if self.rings.with_bitmap() && last_stage {
                self.get_dirty_log(index, entry);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logged_ranges_get_the_dirty_log_flag() {
        assert_eq!(slot_flags(false, DirtyMask::NONE), 0);
        assert_eq!(slot_flags(true, DirtyMask::NONE), KVM_MEM_READONLY);
        assert_eq!(slot_flags(false, DirtyClient::Migration.mask()), KVM_MEM_LOG_DIRTY_PAGES);
        assert_eq!(
            slot_flags(true, DirtyClient::Vga.mask()),
            KVM_MEM_LOG_DIRTY_PAGES | KVM_MEM_READONLY
        );
    }

    #[test]
    fn slots_are_found_again_whatever_their_flags() {
        let a = Slot { gpa: 0x1000, size: 0x4000, host: 0x7000_0000, flags: 0 };
        let b = Slot { flags: KVM_MEM_LOG_DIRTY_PAGES, ..a };
        assert!(a.same_place(&b));
        assert!(b.logging() && !a.logging());
        assert_eq!(a.pages(), 4);
        assert!(!a.same_place(&Slot { size: 0x2000, ..a }));
    }
}
