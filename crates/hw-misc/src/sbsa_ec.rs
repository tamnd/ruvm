// SPDX-License-Identifier: GPL-2.0-or-later

//! The `sbsa-ref` secure embedded controller, hw/misc/sbsa_ec.c.
//!
//! [`SbsaEc`] is `SECUREECState`, a 0x1000 byte block in the secure part of the `sbsa-ref`
//! memory map. Trusted firmware's PSCI code writes [`SBSA_EC_CMD_POWEROFF`] or
//! [`SBSA_EC_CMD_REBOOT`] to offset 0 to power the machine off or reboot it. Nothing is
//! readable and reads return 0. Only 4 byte accesses are valid.
//!
//! Differences from QEMU:
//!
//! - Not ported: QOM registration.
//! - The requests go to a handler the board gives [`SbsaEc::new`] instead of
//!   `qemu_system_shutdown_request()` and `qemu_system_reset_request()`.
//! - The `qemu_log_mask()` guest error messages are not printed, since the workspace has no
//!   `-d` log yet.
//! - `sbsa_ec_ops` is `DEVICE_NATIVE_ENDIAN`, which is little endian on every Arm target.

use std::fmt;
use std::sync::Arc;

use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MemResult, MmioOps};

/// `TYPE_SBSA_SECURE_EC`.
pub const TYPE_SBSA_SECURE_EC: &str = "sbsa-ec";

/// Size of the `sbsa-ec` MMIO region.
pub const SBSA_EC_MMIO_SIZE: u64 = 0x1000;

/// `SBSA_EC_CMD_POWEROFF`.
pub const SBSA_EC_CMD_POWEROFF: u64 = 0x01;
/// `SBSA_EC_CMD_REBOOT`.
pub const SBSA_EC_CMD_REBOOT: u64 = 0x02;

/// What a write to the power command register asks for.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SbsaEcRequest {
    /// `qemu_system_shutdown_request(SHUTDOWN_CAUSE_GUEST_SHUTDOWN)`.
    Poweroff,
    /// `qemu_system_reset_request(SHUTDOWN_CAUSE_GUEST_RESET)`.
    Reboot,
}

/// Called on every power command, from the vCPU thread that wrote it.
pub type SbsaEcHandler = Arc<dyn Fn(SbsaEcRequest) + Send + Sync>;

/// `SECUREECState`, the `sbsa-ec` device.
pub struct SbsaEc {
    handler: SbsaEcHandler,
}

impl fmt::Debug for SbsaEc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SbsaEc").finish_non_exhaustive()
    }
}

impl SbsaEc {
    /// `sbsa_ec_init()`, with the handler that stands in for the shutdown and reset requests.
    pub fn new(handler: impl Fn(SbsaEcRequest) + Send + Sync + 'static) -> Arc<SbsaEc> {
        Arc::new(SbsaEc { handler: Arc::new(handler) })
    }

    /// `sbsa_ec_write()`.
    pub fn reg_write(&self, offset: u64, value: u64) {
        // QEMU logs "sbsa-ec: unknown EC register" for other offsets and "sbsa-ec: unknown
        // power command" for other values.
        if offset != 0 {
            return;
        }
        match value {
            SBSA_EC_CMD_POWEROFF => (self.handler)(SbsaEcRequest::Poweroff),
            SBSA_EC_CMD_REBOOT => (self.handler)(SbsaEcRequest::Reboot),
            _ => {}
        }
    }
}

/// `sbsa_ec_ops`.
impl MmioOps for SbsaEc {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        // QEMU logs "sbsa-ec: no readable registers".
        Ok(0)
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.reg_write(offset, value);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::exact(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn power_commands() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        let ec = SbsaEc::new(move |r| s.lock().unwrap().push(r));
        ec.reg_write(0, SBSA_EC_CMD_POWEROFF);
        ec.reg_write(0, SBSA_EC_CMD_REBOOT);
        ec.reg_write(0, 3);
        ec.reg_write(4, SBSA_EC_CMD_POWEROFF);
        assert_eq!(*seen.lock().unwrap(), [SbsaEcRequest::Poweroff, SbsaEcRequest::Reboot]);
        assert_eq!(ec.valid(), AccessConstraints::exact(4));
    }
}
