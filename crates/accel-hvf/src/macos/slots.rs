// SPDX-License-Identifier: GPL-2.0-or-later

//! The guest physical map, `hvf_set_phys_mem()` and the dirty logging callbacks of
//! accel/hvf/hvf-all.c.
//!
//! RAM, and ROM devices in romd mode, are mapped with `hv_vm_map()`; everything else stays
//! unmapped so the guest's accesses exit as data aborts. Dirty logging write protects a range
//! with `hv_vm_protect()`: the first write to a page faults, the vCPU marks the page dirty,
//! gives write access back and retries, as `hvf_handle_exception()` does.
//!
//! Two differences from QEMU. The listener keeps a table of what it mapped, so it only unmaps
//! what is there and holds the RAM blocks alive while they are mapped, where QEMU unmaps
//! blindly and asserts. And ruvm has no `log_clear` callback, so the write protection comes
//! back on `log_sync` instead, which is when the consumer is about to read the bits.

use std::ffi::c_void;
use std::sync::{Arc, Mutex, MutexGuard};

use ruvm_mem::{
    AddressSpace, DirtyClient, DirtyMask, FlatRange, MemoryListener, RamBlock, RegionType,
};

use super::ffi::{self, HV_MEMORY_EXEC, HV_MEMORY_READ, HV_MEMORY_WRITE, HvMemoryFlags};
use crate::HvfError;

/// The host page size on Apple Silicon. `hv_vm_map()` works in these.
pub(crate) const HOST_PAGE: u64 = 0x4000;

/// One `hv_vm_map()` call that is in place.
#[derive(Clone, Debug)]
struct Mapping {
    gpa: u64,
    size: u64,
    writable: bool,
    block: Arc<RamBlock>,
    offset: u64,
    mask: DirtyMask,
}

impl Mapping {
    fn contains(&self, gpa: u64) -> bool {
        gpa >= self.gpa && gpa - self.gpa < self.size
    }

    fn same_place(&self, other: &Mapping) -> bool {
        self.gpa == other.gpa && self.size == other.size
    }
}

/// What a write fault on a mapped page turned out to be.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum WriteFault {
    /// Writable RAM: the page was marked dirty and made writable, so retry the access.
    Retry,
    /// Not writable RAM: emulate the access as MMIO.
    Mmio,
}

/// Keeps the framework's guest physical map in step with an address space.
#[derive(Debug, Default)]
pub struct SlotListener {
    maps: Mutex<Vec<Mapping>>,
}

fn protect(gpa: u64, size: u64, flags: HvMemoryFlags) {
    // SAFETY: the range was mapped by this listener, and changing the guest's permissions on
    // it does not touch host memory.
    let r = unsafe { ffi::hv_vm_protect(gpa, size as usize, flags) };
    if let Err(e) = HvfError::check("hv_vm_protect", r) {
        // QEMU aborts here too: the dirty log would be wrong from now on.
        panic!("{e}");
    }
}

impl SlotListener {
    pub(crate) fn new() -> SlotListener {
        SlotListener::default()
    }

    fn lock(&self) -> MutexGuard<'_, Vec<Mapping>> {
        self.maps.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The mappings in place as `(guest address, size, writable)`.
    pub fn mappings(&self) -> Vec<(u64, u64, bool)> {
        self.lock().iter().map(|m| (m.gpa, m.size, m.writable)).collect()
    }

    /// What `hvf_set_phys_mem()` would map for a range, or `None` when it must trap.
    fn mapping_for(fr: &FlatRange) -> Option<Mapping> {
        let block = fr.ram_block()?;
        let writable = match fr.region_type() {
            RegionType::Ram => !fr.readonly(),
            RegionType::RomDevice if fr.romd_mode() => false,
            _ => return None,
        };
        let size = u64::try_from(fr.size()).ok()?;
        let gpa = fr.addr();
        let offset = fr.offset_in_region();
        let host = block.host_addr() as u64 + offset;
        // Not page aligned, so it cannot be mapped as RAM.
        if size == 0 || (size | gpa | host) & (HOST_PAGE - 1) != 0 {
            return None;
        }
        Some(Mapping {
            gpa,
            size,
            writable,
            block: Arc::clone(block),
            offset,
            mask: fr.dirty_log_mask(),
        })
    }

    /// The flags a mapping gets: no write while a dirty client watches it.
    fn flags(m: &Mapping) -> HvMemoryFlags {
        let mut flags = HV_MEMORY_READ | HV_MEMORY_EXEC;
        if m.writable && m.mask.is_empty() {
            flags |= HV_MEMORY_WRITE;
        }
        flags
    }

    /// The write fault side of `hvf_handle_exception()`: a write to `gpa` hit a page the
    /// guest cannot write.
    pub(crate) fn write_fault(&self, gpa: u64) -> WriteFault {
        let maps = self.lock();
        let Some(m) = maps.iter().find(|m| m.contains(gpa)) else { return WriteFault::Mmio };
        if !m.writable {
            return WriteFault::Mmio;
        }
        let page = gpa & !(HOST_PAGE - 1);
        if !m.mask.is_empty() {
            m.block.set_dirty(
                m.offset + (page - m.gpa),
                HOST_PAGE,
                m.mask.without(DirtyClient::Code),
            );
        }
        protect(page, HOST_PAGE, HV_MEMORY_READ | HV_MEMORY_WRITE | HV_MEMORY_EXEC);
        WriteFault::Retry
    }

    fn update_mask(&self, fr: &FlatRange, old: DirtyMask, new: DirtyMask) {
        let Some(want) = Self::mapping_for(fr) else { return };
        let mut maps = self.lock();
        let Some(m) = maps.iter_mut().find(|m| m.same_place(&want)) else { return };
        m.mask = new;
        if m.writable && old.is_empty() != new.is_empty() {
            protect(m.gpa, m.size, Self::flags(m));
        }
    }
}

impl MemoryListener for SlotListener {
    fn name(&self) -> &str {
        "hvf"
    }

    fn priority(&self) -> i32 {
        10
    }

    fn region_add(&self, _space: &AddressSpace, fr: &FlatRange) {
        let Some(m) = Self::mapping_for(fr) else { return };
        let host = (m.block.host_addr() as u64 + m.offset) as *mut c_void;
        // SAFETY: `host` is `size` bytes inside the RAM block's mapping, and the table keeps
        // an Arc to the block until the range is unmapped again, so the memory outlives the
        // guest mapping.
        let r = unsafe { ffi::hv_vm_map(host, m.gpa, m.size as usize, Self::flags(&m)) };
        if let Err(e) = HvfError::check("hv_vm_map", r) {
            panic!("{e}");
        }
        self.lock().push(m);
    }

    fn region_del(&self, _space: &AddressSpace, fr: &FlatRange) {
        let Some(want) = Self::mapping_for(fr) else { return };
        let mut maps = self.lock();
        let Some(i) = maps.iter().position(|m| m.same_place(&want)) else { return };
        let m = maps.remove(i);
        // SAFETY: the range was mapped by this listener. After this the guest cannot reach
        // the block, so dropping the Arc is fine.
        let r = unsafe { ffi::hv_vm_unmap(m.gpa, m.size as usize) };
        if let Err(e) = HvfError::check("hv_vm_unmap", r) {
            panic!("{e}");
        }
    }

    /// `hvf_log_start()`: the first client write protects the range.
    fn log_start(&self, _space: &AddressSpace, fr: &FlatRange, old: DirtyMask, new: DirtyMask) {
        self.update_mask(fr, old, new);
    }

    /// `hvf_log_stop()`: when the last client goes the range is writable again.
    fn log_stop(&self, _space: &AddressSpace, fr: &FlatRange, old: DirtyMask, new: DirtyMask) {
        self.update_mask(fr, old, new);
    }

    /// The bits are already in the block, set at fault time. What is left is QEMU's
    /// `hvf_log_clear()`: protect the range again so the next write is seen.
    fn log_sync(&self, _space: &AddressSpace, fr: &FlatRange) {
        let Some(want) = Self::mapping_for(fr) else { return };
        let maps = self.lock();
        let Some(m) = maps.iter().find(|m| m.same_place(&want)) else { return };
        if m.writable && !m.mask.is_empty() {
            protect(m.gpa, m.size, HV_MEMORY_READ | HV_MEMORY_EXEC);
        }
    }
}
