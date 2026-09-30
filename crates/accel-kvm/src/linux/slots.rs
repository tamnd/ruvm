// SPDX-License-Identifier: GPL-2.0-or-later

//! KVM memory slots, `kvm_region_add()` and `kvm_region_del()` with `kvm_set_phys_mem()`.
//!
//! Each RAM backed flat range becomes one slot. Ranges are trimmed to host pages first, as
//! `kvm_align_section()` does, and anything that is not RAM stays out so the guest's accesses
//! to it exit to userspace.

use std::sync::{Arc, Mutex, MutexGuard};

use kvm_bindings::{KVM_MEM_READONLY, kvm_userspace_memory_region};
use kvm_ioctls::VmFd;
use ruvm_mem::{AddressSpace, FlatRange, MemoryListener, RamBlock, RegionType};

use super::os_error;

/// x86 hosts use 4 KiB pages for slot boundaries.
const HOST_PAGE: u64 = 4096;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Slot {
    gpa: u64,
    size: u64,
    host: u64,
    flags: u32,
}

/// Slot number to what it maps, with the RAM block kept alive while the slot exists.
type Table = Vec<Option<(Slot, Arc<RamBlock>)>>;

/// Keeps KVM's memory slots in step with an address space, `KVMMemoryListener`.
#[derive(Debug)]
pub struct SlotListener {
    vm: Arc<VmFd>,
    readonly_mem: bool,
    slots: Mutex<Table>,
}

impl SlotListener {
    pub(crate) fn new(vm: Arc<VmFd>, readonly_mem: bool, nr_slots: usize) -> Self {
        SlotListener { vm, readonly_mem, slots: Mutex::new(vec![None; nr_slots]) }
    }

    fn lock(&self) -> MutexGuard<'_, Table> {
        self.slots.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The slots in use as `(guest address, size, read only)`, for tests and `info kvm`.
    pub fn slots(&self) -> Vec<(u64, u64, bool)> {
        self.lock()
            .iter()
            .flatten()
            .map(|(s, _)| (s.gpa, s.size, s.flags & KVM_MEM_READONLY != 0))
            .collect()
    }

    /// What a range maps to, or `None` when it must trap. This is the filter at the top of
    /// `kvm_set_phys_mem()`: RAM always gets a slot, read only when the region is and the host
    /// can do it, and a ROM device gets a read only slot only in romd mode on such a host.
    fn slot_for(&self, fr: &FlatRange) -> Option<(Slot, Arc<RamBlock>)> {
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
        let host = block.host_addr() as u64 + fr.offset_in_region() + delta;
        let flags = if readonly { KVM_MEM_READONLY } else { 0 };
        Some((Slot { gpa: start, size, host, flags }, Arc::clone(block)))
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
}

impl MemoryListener for SlotListener {
    fn name(&self) -> &str {
        "kvm-memory"
    }

    fn priority(&self) -> i32 {
        10
    }

    fn region_add(&self, _space: &AddressSpace, fr: &FlatRange) {
        let Some((slot, block)) = self.slot_for(fr) else { return };
        let mut slots = self.lock();
        let Some(index) = slots.iter().position(Option::is_none) else {
            panic!("kvm_alloc_slot: no free slot anymore");
        };
        self.set(index, slot, slot.size);
        slots[index] = Some((slot, block));
    }

    fn region_del(&self, _space: &AddressSpace, fr: &FlatRange) {
        let Some((slot, _)) = self.slot_for(fr) else { return };
        let mut slots = self.lock();
        let Some(index) = slots.iter().position(|s| s.as_ref().is_some_and(|(s, _)| *s == slot))
        else {
            return;
        };
        self.set(index, slot, 0);
        slots[index] = None;
    }
}
