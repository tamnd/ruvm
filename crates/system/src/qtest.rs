// SPDX-License-Identifier: GPL-2.0-or-later

//! `-qtest` and `-qtest-log`: `qtest_server_init()` from system/qtest.c, with the protocol
//! server from ruvm-accel-qtest put on a chardev, and the machine side it talks to.

use std::io::{self, Read};
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use ruvm_accel_qtest::{IrqHandler, Qtest, QtestBackend, open_log};
use ruvm_base::report::{error_report, report_error};
use ruvm_base::{Error, Result};
use ruvm_chardev::opts::{chardev_opts, parse_compat};
use ruvm_chardev::{Attachment, Chardevs, Connection, Frontend};

/// `QEMU_CLOCK_VIRTUAL` while qtest drives it. Nothing runs on it yet, so it only moves when
/// the test steps it.
#[derive(Debug, Default)]
pub struct VirtualClock(AtomicI64);

impl VirtualClock {
    pub fn get_ns(&self) -> i64 {
        self.0.load(Ordering::Acquire)
    }

    /// `qemu_clock_advance_virtual_time()`: the clock never goes back.
    pub fn advance_to(&self, dest: i64) -> i64 {
        self.0.fetch_max(dest, Ordering::AcqRel).max(dest)
    }
}

/// `target_big_endian()` for a `qemu-system-<target>` name.
pub fn target_big_endian(target: &str) -> bool {
    matches!(
        target,
        "hppa"
            | "m68k"
            | "microblaze"
            | "mips"
            | "mips64"
            | "or1k"
            | "ppc"
            | "ppc64"
            | "s390x"
            | "sh4eb"
            | "sparc"
            | "sparc64"
            | "xtensaeb"
    )
}

/// Machine `none`: no RAM, no devices and nothing on the I/O ports. Reads from unassigned
/// memory give zeroes and reads from unassigned ports give all ones, as they do in QEMU.
#[derive(Debug)]
struct NoneMachine {
    big_endian: bool,
    clock: Arc<VirtualClock>,
}

impl QtestBackend for NoneMachine {
    type Device = ();

    fn big_endian(&self) -> bool {
        self.big_endian
    }

    fn memory_read(&mut self, _addr: u64, buf: &mut [u8]) {
        buf.fill(0);
    }

    fn memory_write(&mut self, _addr: u64, _buf: &[u8]) {}

    fn port_read(&mut self, _addr: u16, _size: usize) -> u32 {
        u32::MAX
    }

    fn port_write(&mut self, _addr: u16, _size: usize, _value: u32) {}

    fn clock_get_ns(&mut self) -> i64 {
        self.clock.get_ns()
    }

    fn clock_deadline_ns_all(&mut self) -> i64 {
        -1
    }

    fn clock_advance_virtual_time(&mut self, dest: i64) -> i64 {
        self.clock.advance_to(dest)
    }

    fn resolve_device(&mut self, _path: &str) -> Option<()> {
        None
    }

    fn irq_intercept_in(&mut self, _dev: &(), _handler: &IrqHandler) -> bool {
        false
    }

    fn irq_intercept_out(&mut self, _dev: &(), _name: Option<&str>, _h: &IrqHandler) -> bool {
        false
    }

    fn set_irq_in(&mut self, _dev: &(), _name: Option<&str>, _num: i32, _level: i32) {}
}

/// Reads what a chardev client sends.
struct ConnReader<'a>(&'a mut Connection);

impl Read for ConnReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.recv(buf)
    }
}

struct QtestFrontend(Qtest<NoneMachine>);

impl Frontend for QtestFrontend {
    fn serve(&self, conn: &mut Connection) -> io::Result<()> {
        let writer = conn.writer()?;
        match self.0.serve(ConnReader(conn), writer) {
            // A failed assertion on what the client sent. QEMU aborts there.
            Err(e) if e.kind() == io::ErrorKind::Other => {
                error_report(&e.to_string());
                std::process::abort();
            }
            r => r,
        }
    }
}

/// `qtest_server_init()`: a chardev called `qtest` from the old style `-qtest` string, and the
/// protocol server on it.
pub fn server_init(
    chardevs: &Chardevs,
    chrdev: &str,
    log: Option<&str>,
    target: &str,
    clock: Arc<VirtualClock>,
) -> Result<Attachment> {
    let failed = || Error::generic(format!("Failed to initialize device for qtest: \"{chrdev}\""));
    let mut list = chardev_opts();
    // qemu_chr_new() reports what went wrong itself and only returns NULL to us.
    let reported = |e: Option<Error>| {
        if let Some(e) = e {
            report_error(&e);
        }
        failed()
    };
    let handle = parse_compat(&mut list, "qtest", chrdev, false).map_err(reported)?;
    let opts = list.get(handle).expect("just parsed");
    let chr = chardevs.new_from_opts(opts).map_err(|e| reported(Some(e)))?.ok_or_else(failed)?;
    let machine = NoneMachine { big_endian: target_big_endian(target), clock };
    let qtest = Qtest::new(machine);
    qtest.set_log(open_log(log));
    chr.attach(Arc::new(QtestFrontend(qtest)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_only_moves_forward() {
        let c = VirtualClock::default();
        assert_eq!(c.advance_to(100), 100);
        assert_eq!(c.advance_to(50), 100);
        assert_eq!(c.get_ns(), 100);
    }

    #[test]
    fn endianness() {
        assert!(target_big_endian("s390x"));
        assert!(!target_big_endian("x86_64"));
        assert!(!target_big_endian("mipsel"));
    }
}
