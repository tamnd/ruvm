// SPDX-License-Identifier: GPL-2.0-or-later

//! The Physical Presence Interface memory, hw/tpm/tpm_ppi.c.
//!
//! The firmware and the PPI methods in the DSDT share 0x400 bytes at 0xFED45000. The layout is
//! in docs/specs/tpm.rst; the device only looks at the memory overwrite request at 0x15a.

use std::sync::Mutex;

use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MemResult, MmioOps};

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
    /// The memory clear `tpm_ppi_reset()` does when the guest asked for it: the machine
    /// zeroes all of guest RAM when this is set at reset time.
    pub fn memory_clear_requested(&self) -> bool {
        self.buf.lock().unwrap_or_else(|p| p.into_inner())[PPI_MOVV] & 1 != 0
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
