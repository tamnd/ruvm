// SPDX-License-Identifier: GPL-2.0-or-later

//! The Physical Presence Interface memory, hw/tpm/tpm_ppi.c.
//!
//! The firmware and the PPI methods in the DSDT share 0x400 bytes at 0xFED45000. The layout is
//! in docs/specs/tpm.rst; the device only looks at the memory overwrite request at 0x15a.

use std::sync::Mutex;

use ruvm_mem::{
    AccessConstraints, AccessCtx, AccessSize, AddressSpace, MemResult, MmioOps, RegionType,
};

/// `TPM_PPI_ADDR_BASE`.
pub const TPM_PPI_ADDR_BASE: u64 = 0xfed4_5000;
/// `TPM_PPI_ADDR_SIZE`.
pub const TPM_PPI_ADDR_SIZE: usize = 0x400;

/// The `MOVV` byte; bit 0 asks for guest memory to be cleared on the next reset.
const PPI_MOVV: usize = 0x15a;

/// `TPMPPI`: the shared memory, a `ram_device` region in QEMU.
#[derive(Debug)]
pub struct TpmPpi {
    buf: Mutex<[u8; TPM_PPI_ADDR_SIZE]>,
}

impl Default for TpmPpi {
    fn default() -> Self {
        TpmPpi { buf: Mutex::new([0; TPM_PPI_ADDR_SIZE]) }
    }
}

impl TpmPpi {
    /// Whether the guest asked for its memory to be cleared on the next reset.
    pub fn memory_clear_requested(&self) -> bool {
        self.buf.lock().unwrap_or_else(|p| p.into_inner())[PPI_MOVV] & 1 != 0
    }

    /// `tpm_ppi_reset()`: when the guest asked for it, zeroes the RAM `space` maps, which is
    /// `address_space_memory`. As with `guest_phys_blocks_append()`, ROM is RAM too, and
    /// nonvolatile memory and devices are left alone. The request itself stays set.
    pub fn reset(&self, space: &AddressSpace) {
        if !self.memory_clear_requested() {
            return;
        }
        for fr in space.flatview().ranges() {
            if fr.region_type() != RegionType::Ram || fr.nonvolatile() {
                continue;
            }
            let (Some(block), Ok(len)) = (fr.ram_block(), u64::try_from(fr.size())) else {
                continue;
            };
            if block.fill(fr.offset_in_region(), len, 0).is_ok() {
                block.set_dirty(fr.offset_in_region(), len, fr.dirty_log_mask());
            }
        }
    }
}

impl MmioOps for TpmPpi {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        let buf = self.buf.lock().unwrap_or_else(|p| p.into_inner());
        let mut v = [0; 8];
        let (o, n) = (offset as usize, size.bytes() as usize);
        let n = n.min(TPM_PPI_ADDR_SIZE.saturating_sub(o));
        v[..n].copy_from_slice(&buf[o..o + n]);
        Ok(u64::from_le_bytes(v))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        let mut buf = self.buf.lock().unwrap_or_else(|p| p.into_inner());
        let (o, n) = (offset as usize, size.bytes() as usize);
        let n = n.min(TPM_PPI_ADDR_SIZE.saturating_sub(o));
        buf[o..o + n].copy_from_slice(&value.to_le_bytes()[..n]);
        Ok(())
    }

    /// `ram_device_mem_ops`.
    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 8).allow_unaligned()
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 8).allow_unaligned()
    }
}
