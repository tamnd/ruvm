// SPDX-License-Identifier: GPL-2.0-or-later

//! The SiFive test finisher, hw/misc/sifive_test.c and include/hw/misc/sifive_test.h.
//!
//! A guest ends a run by writing a status to offset 0: [`FINISHER_PASS`] or [`FINISHER_FAIL`]
//! in the low 16 bits with an exit code in the high 16, or [`FINISHER_RESET`] to reset the
//! machine. Reads return 0. The region is 0x1000 bytes and takes aligned 2 and 4 byte accesses
//! (`sifive_test_ops.valid`), in the target's byte order, which is little endian on RISC-V.
//!
//! What the board does with a request, to match QEMU (QEMU does not call `exit()` here, it asks
//! the main loop to shut down or reset):
//!
//! - A [`SiFiveTestRequest::Fail`] is `qemu_system_shutdown_request_with_code()` with
//!   `SHUTDOWN_CAUSE_GUEST_PANIC` and the code. The process exits with the code if it is not 0.
//!   With a code of 0 it exits with `EXIT_FAILURE` only under `-action panic=exit-failure`, and
//!   with 0 otherwise. `-action shutdown=pause` (`-no-shutdown`) pauses instead of exiting.
//! - A [`SiFiveTestRequest::Pass`] is the same with `SHUTDOWN_CAUSE_GUEST_SHUTDOWN`, so the process
//!   exits with the code (0 for a plain pass) unless shutdown is set to pause.
//! - A [`SiFiveTestRequest::Reset`] is `qemu_system_reset_request(SHUTDOWN_CAUSE_GUEST_RESET)`.
//!
//! Differences from QEMU:
//!
//! - Not ported: QOM registration.
//! - The requests go to a handler the board gives [`SiFiveTest::new`] instead of the global run
//!   state functions.
//! - The `qemu_log_mask()` guest error message for other writes is not printed, since the
//!   workspace has no `-d` log yet.

use std::fmt;
use std::sync::Arc;

use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, Endian, MemResult, MmioOps};

/// `TYPE_SIFIVE_TEST`.
pub const TYPE_SIFIVE_TEST: &str = "riscv.sifive.test";

/// Size of the `riscv.sifive.test` MMIO region.
pub const SIFIVE_TEST_MMIO_SIZE: u64 = 0x1000;

/// `FINISHER_FAIL`.
pub const FINISHER_FAIL: u16 = 0x3333;
/// `FINISHER_PASS`.
pub const FINISHER_PASS: u16 = 0x5555;
/// `FINISHER_RESET`.
pub const FINISHER_RESET: u16 = 0x7777;

/// What a guest write asks for.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SiFiveTestRequest {
    /// `FINISHER_PASS`: a guest shutdown with this exit code.
    Pass(u16),
    /// `FINISHER_FAIL`: a guest panic shutdown with this exit code.
    Fail(u16),
    /// `FINISHER_RESET`: a guest reset.
    Reset,
}

/// Called on every finisher write, from the vCPU thread that made it.
pub type SiFiveTestHandler = Arc<dyn Fn(SiFiveTestRequest) + Send + Sync>;

/// The request a write of `value` at `addr` makes, if any.
pub fn sifive_test_request(addr: u64, value: u64) -> Option<SiFiveTestRequest> {
    if addr != 0 {
        return None;
    }
    let status = (value & 0xffff) as u16;
    let code = ((value >> 16) & 0xffff) as u16;
    match status {
        FINISHER_FAIL => Some(SiFiveTestRequest::Fail(code)),
        FINISHER_PASS => Some(SiFiveTestRequest::Pass(code)),
        FINISHER_RESET => Some(SiFiveTestRequest::Reset),
        _ => None,
    }
}

/// `SiFiveTestState`, the `riscv.sifive.test` device.
pub struct SiFiveTest {
    handler: SiFiveTestHandler,
}

impl fmt::Debug for SiFiveTest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SiFiveTest").finish_non_exhaustive()
    }
}

impl SiFiveTest {
    /// `sifive_test_init()`, with the handler that stands in for the shutdown and reset
    /// requests.
    pub fn new(handler: impl Fn(SiFiveTestRequest) + Send + Sync + 'static) -> Arc<SiFiveTest> {
        Arc::new(SiFiveTest { handler: Arc::new(handler) })
    }

    /// `sifive_test_write()`.
    pub fn reg_write(&self, addr: u64, value: u64) {
        // Anything else QEMU logs as "sifive_test_write: write: addr=0x%x val=0x%016" PRIx64.
        if let Some(request) = sifive_test_request(addr, value) {
            (self.handler)(request);
        }
    }
}

/// `sifive_test_ops`.
impl MmioOps for SiFiveTest {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(0)
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.reg_write(offset, value);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(2, 4)
    }

    /// `DEVICE_NATIVE_ENDIAN`, and every RISC-V target is little endian.
    fn endianness(&self) -> Endian {
        Endian::Little
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn finisher_codes() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        let dev = SiFiveTest::new(move |r| s.lock().unwrap().push(r));
        dev.reg_write(0, 0x5555);
        dev.reg_write(0, (3 << 16) | 0x3333);
        dev.reg_write(0, 0x7777);
        dev.reg_write(0, 0xabcd_5555);
        // Not a request: another status, another offset.
        dev.reg_write(0, 0x1234);
        dev.reg_write(4, 0x5555);
        assert_eq!(
            *seen.lock().unwrap(),
            [
                SiFiveTestRequest::Pass(0),
                SiFiveTestRequest::Fail(3),
                SiFiveTestRequest::Reset,
                SiFiveTestRequest::Pass(0xabcd),
            ]
        );
        assert_eq!(dev.valid(), AccessConstraints::any_size(2, 4));
    }
}
