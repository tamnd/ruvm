// SPDX-License-Identifier: GPL-2.0-or-later

//! Programmable Attribute Map (PAM) regions, from hw/pci-host/pam.c.
//!
//! The i440FX and the Q35 MCH decide per segment of the 0xc0000 to 0xfffff range whether reads
//! and writes go to DRAM or to the PCI bus (where the BIOS ROM lives). QEMU models each segment
//! with four aliases mapped over system memory at priority 1, one per two bit PAM value, and
//! enables exactly one of them:
//!
//! - 0: both reads and writes go to PCI ("pam-pci" into the PCI address space).
//! - 1: read only DRAM ("pam-rom", a read-only alias of RAM). Writes are dropped, where real
//!   hardware would forward them to PCI.
//! - 2: write only DRAM. QEMU does not split reads and writes here and simply maps RAM read and
//!   write, and so does this port.
//! - 3: DRAM for both ("pam-ram").
//!
//! Also the SMRAM and PAM register constants shared by the i440FX and the Q35 MCH.
//!
//! Not ported: VMState (the current PAM value is derived from config space again after load).

use std::fmt;
use std::sync::{Arc, Mutex};

use ruvm_mem::{MemError, MemorySystem, RegionId};

pub const SMRAM_C_BASE: u64 = 0xa0000;
pub const SMRAM_C_END: u64 = 0xc0000;
pub const SMRAM_C_SIZE: u64 = 0x20000;

pub const PAM_EXPAN_BASE: u64 = 0xc0000;
pub const PAM_EXPAN_SIZE: u64 = 0x04000;

pub const PAM_EXBIOS_BASE: u64 = 0xe0000;
pub const PAM_EXBIOS_SIZE: u64 = 0x04000;

pub const PAM_BIOS_BASE: u64 = 0xf0000;
pub const PAM_BIOS_END: u64 = 0xfffff;
pub const PAM_BIOS_SIZE: u64 = 0x10000;

pub const PAM_ATTR_WE: u8 = 2;
pub const PAM_ATTR_RE: u8 = 1;
pub const PAM_ATTR_MASK: u8 = 3;

pub const SMRAM_D_OPEN: u8 = 1 << 6;
pub const SMRAM_D_CLS: u8 = 1 << 5;
pub const SMRAM_D_LCK: u8 = 1 << 4;
pub const SMRAM_G_SMRAME: u8 = 1 << 3;
pub const SMRAM_C_BASE_SEG_MASK: u8 = 0x7;
/// Hardwired to 0b010.
pub const SMRAM_C_BASE_SEG: u8 = 0x2;

/// The number of PAM segments: the 64 KiB BIOS area plus twelve 16 KiB segments.
pub const PAM_REGIONS_COUNT: usize = 13;

/// One PAM segment, `PAMMemoryRegion`: four aliases indexed by PAM value, one enabled.
pub struct PamMemoryRegion {
    memory: Arc<MemorySystem>,
    alias: [RegionId; 4],
    current: Mutex<u8>,
}

impl fmt::Debug for PamMemoryRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PamMemoryRegion")
            .field("alias", &self.alias)
            .field("current", &self.current())
            .finish_non_exhaustive()
    }
}

impl PamMemoryRegion {
    /// `init_pam()`: creates the four aliases for `size` bytes at `start`, maps them all over
    /// `system_memory` at priority 1 and leaves them all disabled, with the current value 0.
    pub fn new(
        memory: &Arc<MemorySystem>,
        ram_memory: RegionId,
        system_memory: RegionId,
        pci_address_space: RegionId,
        start: u64,
        size: u64,
    ) -> Result<PamMemoryRegion, MemError> {
        let size128 = u128::from(size);
        let ram = memory.new_alias("pam-ram", ram_memory, start, size128)?;
        // ROM (not quite correct, as QEMU says).
        let rom = memory.new_alias("pam-rom", ram_memory, start, size128)?;
        memory.set_readonly(rom, true)?;
        // QEMU does not distinguish the read and write cases here.
        let pci = memory.new_alias("pam-pci", pci_address_space, start, size128)?;
        let wo = memory.new_alias("pam-pci", ram_memory, start, size128)?;
        let alias = [pci, rom, wo, ram];

        let _t = memory.transaction();
        for a in alias {
            memory.set_enabled(a, false)?;
            memory.add_subregion_overlap(system_memory, start, a, 1)?;
        }
        Ok(PamMemoryRegion { memory: Arc::clone(memory), alias, current: Mutex::new(0) })
    }

    /// `pam_update()`: segment `idx` takes its value from `val`, the high nibble for even
    /// segments and the low nibble for odd ones.
    pub fn update(&self, idx: usize, val: u8) {
        assert!(idx < PAM_REGIONS_COUNT, "PAM segment {idx} out of range");
        let mut cur = self.current.lock().unwrap_or_else(|p| p.into_inner());
        let shift = if idx & 1 == 0 { 4 } else { 0 };
        let next = (val >> shift) & PAM_ATTR_MASK;
        let _t = self.memory.transaction();
        self.memory.set_enabled(self.alias[usize::from(*cur)], false).expect("PAM alias exists");
        *cur = next;
        self.memory.set_enabled(self.alias[usize::from(next)], true).expect("PAM alias exists");
    }

    /// The PAM value currently in effect, the index of the enabled alias.
    pub fn current(&self) -> u8 {
        *self.current.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The alias used for PAM value `value` (0 to 3).
    pub fn alias(&self, value: u8) -> RegionId {
        self.alias[usize::from(value & PAM_ATTR_MASK)]
    }
}
