// SPDX-License-Identifier: GPL-2.0-or-later

//! The guest physical map, `whpx_set_phys_mem()` and `whpx_log_sync()` of
//! accel/whpx/whpx-common.c.
//!
//! RAM is mapped read, write and execute, and ROM devices in romd mode read and execute, so a
//! write to them exits as MMIO. Everything else stays unmapped. WHPX has no dirty page
//! tracking that QEMU uses, so a sync marks every mapped page dirty for the clients that
//! watch it.
//!
//! As in the HVF listener, a table of what is mapped means only mapped ranges are unmapped and
//! the RAM blocks stay alive while the guest can reach them.

use std::sync::{Arc, Mutex, MutexGuard};

use ruvm_mem::{AddressSpace, FlatRange, MemoryListener, RamBlock, RegionType};
use windows_sys::Win32::System::Hypervisor::{
    WHvMapGpaRangeFlagExecute, WHvMapGpaRangeFlagRead, WHvMapGpaRangeFlagWrite,
};

use super::Partition;
use crate::WhpxError;

/// The page size `WHvMapGpaRange()` works in.
const PAGE: u64 = 0x1000;

#[derive(Clone, Debug)]
struct Mapping {
    gpa: u64,
    size: u64,
    writable: bool,
    block: Arc<RamBlock>,
    offset: u64,
}

/// Keeps the partition's guest physical map in step with an address space.
#[derive(Debug)]
pub struct SlotListener {
    part: Arc<Partition>,
    maps: Mutex<Vec<Mapping>>,
}

impl SlotListener {
    pub(crate) fn new(part: Arc<Partition>) -> SlotListener {
        SlotListener { part, maps: Mutex::new(Vec::new()) }
    }

    fn lock(&self) -> MutexGuard<'_, Vec<Mapping>> {
        self.maps.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The mappings in place as `(guest address, size, writable)`.
    pub fn mappings(&self) -> Vec<(u64, u64, bool)> {
        self.lock().iter().map(|m| (m.gpa, m.size, m.writable)).collect()
    }

    /// What `whpx_set_phys_mem()` would map for a range, or `None` when it must trap.
    fn mapping_for(fr: &FlatRange) -> Option<Mapping> {
        let block = fr.ram_block()?;
        let writable = match fr.region_type() {
            RegionType::Ram => !fr.readonly(),
            RegionType::RomDevice if fr.romd_mode() => false,
            _ => return None,
        };
        let size = u64::try_from(fr.size()).ok()?;
        let gpa = fr.addr();
        // Not page aligned, so it cannot be mapped as RAM.
        if size == 0 || (size | gpa) & (PAGE - 1) != 0 {
            return None;
        }
        Some(Mapping {
            gpa,
            size,
            writable,
            block: Arc::clone(block),
            offset: fr.offset_in_region(),
        })
    }
}

impl MemoryListener for SlotListener {
    fn name(&self) -> &str {
        "whpx"
    }

    fn priority(&self) -> i32 {
        10
    }

    fn region_add(&self, _space: &AddressSpace, fr: &FlatRange) {
        let Some(m) = Self::mapping_for(fr) else { return };
        let mut flags = WHvMapGpaRangeFlagRead | WHvMapGpaRangeFlagExecute;
        if m.writable {
            flags |= WHvMapGpaRangeFlagWrite;
        }
        let host = (m.block.host_addr() as u64 + m.offset) as *const std::ffi::c_void;
        // SAFETY: `host` is `size` bytes inside the RAM block's mapping, and the table keeps an
        // Arc to the block until the range is unmapped again.
        let hr = unsafe {
            (self.part.d().map_gpa_range)(self.part.handle(), host, m.gpa, m.size, flags)
        };
        if let Err(e) = WhpxError::check("failed to map GPA range", hr) {
            // QEMU aborts here too.
            panic!("{e}");
        }
        self.lock().push(m);
    }

    fn region_del(&self, _space: &AddressSpace, fr: &FlatRange) {
        let Some(want) = Self::mapping_for(fr) else { return };
        let mut maps = self.lock();
        let Some(i) = maps.iter().position(|m| m.gpa == want.gpa && m.size == want.size) else {
            return;
        };
        let m = maps.remove(i);
        // SAFETY: the range was mapped by this listener. After this the guest cannot reach the
        // block, so dropping the Arc is fine.
        let hr = unsafe { (self.part.d().unmap_gpa_range)(self.part.handle(), m.gpa, m.size) };
        if let Err(e) = WhpxError::check("failed to unmap GPA range", hr) {
            panic!("{e}");
        }
    }

    /// `whpx_log_sync()`: without dirty tracking every page counts as written.
    fn log_sync(&self, _space: &AddressSpace, fr: &FlatRange) {
        let mask = fr.dirty_log_mask();
        if fr.region_type() != RegionType::Ram || mask.is_empty() {
            return;
        }
        let (Some(block), Ok(size)) = (fr.ram_block(), u64::try_from(fr.size())) else { return };
        block.set_dirty(fr.offset_in_region(), size, mask);
    }
}
