// SPDX-License-Identifier: GPL-2.0-or-later

//! `-qtest` and `-qtest-log`: `qtest_server_init()` from system/qtest.c, with the protocol
//! server from ruvm-accel-qtest put on a chardev, and the machine side it talks to.

use std::io::{self, Read};
use std::sync::Arc;

use ruvm_accel_qtest::{IrqHandler, Qtest, QtestBackend, open_log};
use ruvm_base::report::{error_report, report_error};
use ruvm_base::{Error, Result};
use ruvm_chardev::opts::{chardev_opts, parse_compat};
use ruvm_chardev::{Attachment, Chardevs, Connection, Frontend};
use ruvm_hw_core::{Clock, Machine};
use ruvm_mem::{AddressSpace, MemTxAttrs};
use ruvm_qom::{
    Object, Registry, StrGetter, StrSetter, TYPE_OBJECT, TYPE_USER_CREATABLE, TypeInfo,
    UserCreatableClass,
};

/// `TYPE_QTEST`.
pub const TYPE_QTEST: &str = "qtest";

/// `QTest`: the object `-qtest` puts at `/machine/qtest`.
#[derive(Debug, Default)]
struct QtestObject {
    chardev: std::sync::Mutex<Option<String>>,
    log: std::sync::Mutex<Option<String>>,
}

fn qtest_state(o: &Object) -> Arc<QtestObject> {
    o.state::<QtestObject>().expect("a qtest object")
}

/// Registers `qtest`. Only `-qtest` makes one for now, `-object qtest` is refused when it
/// completes.
pub fn register_types(registry: &Registry) {
    let ty = TypeInfo::new(TYPE_QTEST)
        .parent(TYPE_OBJECT)
        .interface(TYPE_USER_CREATABLE)
        .instance_state(QtestObject::default)
        .class_init(|k| {
            for (name, log) in [("chardev", false), ("log", true)] {
                let get: StrGetter = Arc::new(move |o: &Object| {
                    let q = qtest_state(o);
                    let v = if log { &q.log } else { &q.chardev };
                    Ok(v.lock().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_default())
                });
                let set: StrSetter = Arc::new(move |o: &Object, value: &str| {
                    let q = qtest_state(o);
                    let v = if log { &q.log } else { &q.chardev };
                    *v.lock().unwrap_or_else(|e| e.into_inner()) = Some(value.to_string());
                    Ok(())
                });
                k.property_add_str(name, Some(get), Some(set));
            }
            let uc = k.interface(TYPE_USER_CREATABLE).expect("qtest is user creatable");
            uc.set_ext(UserCreatableClass {
                complete: Some(Arc::new(|_: &Object| {
                    Err(Error::generic("-object qtest is not supported by ruvm yet, use -qtest"))
                })),
                prepare_delete: None,
            });
        });
    registry.register(ty);
}

/// The `/machine/qtest` object `qtest_server_init()` adds.
pub fn add_object(machine: &Object, log: Option<&str>) -> Result<()> {
    let obj = machine.registry().object_new(TYPE_QTEST)?;
    obj.property_set_str("chardev", "qtest")?;
    if let Some(log) = log {
        obj.property_set_str("log", log)?;
    }
    machine.property_try_add_child("qtest", &obj)?;
    Ok(())
}

/// `QEMU_CLOCK_VIRTUAL` while qtest drives it: it only moves when the test steps it, and each
/// step runs the device timers that come due on the way.
pub type VirtualClock = Clock;

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

/// The machine side of the protocol: memory and port accesses go to the system memory and
/// I/O address spaces, as they do in QEMU when there is no CPU.
#[derive(Debug)]
struct NoneMachine {
    big_endian: bool,
    /// The virtual clock the qtest accelerator drives, or `None` when another accelerator runs
    /// the machine and the test only drives it, where `qtest_enabled()` is false and the
    /// clock commands do not exist.
    clock: Option<Arc<VirtualClock>>,
    memory: Arc<AddressSpace>,
    io: Arc<AddressSpace>,
}

impl QtestBackend for NoneMachine {
    type Device = ();

    fn qtest_enabled(&self) -> bool {
        self.clock.is_some()
    }

    fn big_endian(&self) -> bool {
        self.big_endian
    }

    fn memory_read(&mut self, addr: u64, buf: &mut [u8]) {
        buf.fill(0);
        let _ = self.memory.read(addr, MemTxAttrs::UNSPECIFIED, buf);
    }

    fn memory_write(&mut self, addr: u64, buf: &[u8]) {
        let _ = self.memory.write(addr, MemTxAttrs::UNSPECIFIED, buf);
    }

    /// `cpu_inb()` and friends: the bytes come back in target order.
    fn port_read(&mut self, addr: u16, size: usize) -> u32 {
        let mut buf = [0u8; 4];
        let buf = &mut buf[..size.min(4)];
        let _ = self.io.read(u64::from(addr), MemTxAttrs::UNSPECIFIED, buf);
        let v = buf.iter().fold(0u32, |v, b| (v << 8) | u32::from(*b));
        if self.big_endian { v } else { v.swap_bytes() >> (32 - 8 * buf.len() as u32) }
    }

    /// `cpu_outb()` and friends.
    fn port_write(&mut self, addr: u16, size: usize, value: u32) {
        let size = size.min(4);
        let bytes = if self.big_endian {
            value.to_be_bytes()[4 - size..].to_vec()
        } else {
            value.to_le_bytes()[..size].to_vec()
        };
        let _ = self.io.write(u64::from(addr), MemTxAttrs::UNSPECIFIED, &bytes);
    }

    fn clock_get_ns(&mut self) -> i64 {
        self.clock.as_ref().map_or(0, |c| c.get_ns())
    }

    fn clock_deadline_ns_all(&mut self) -> i64 {
        self.clock.as_ref().map_or(-1, |c| c.deadline_ns())
    }

    fn clock_advance_virtual_time(&mut self, dest: i64) -> i64 {
        self.clock.as_ref().map_or(0, |c| c.advance_to(dest))
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
/// protocol server on it, for the `none` machine on the qtest accelerator.
pub fn server_init(
    chardevs: &Chardevs,
    chrdev: &str,
    log: Option<&str>,
    target: &str,
    clock: Arc<VirtualClock>,
    machine: &Machine,
) -> Result<Attachment> {
    let machine = NoneMachine {
        big_endian: target_big_endian(target),
        clock: Some(clock),
        memory: machine.address_space_memory.clone(),
        io: machine.address_space_io.clone(),
    };
    serve(chardevs, chrdev, log, machine)
}

/// `qtest_server_init()` for a board, with its system memory and I/O address spaces. The test
/// reads and writes guest memory and ports through the protocol. `clock` is the virtual clock
/// of the qtest accelerator. On a real accelerator it is `None` and the clock commands are not
/// there, as in QEMU when `qtest_enabled()` is false.
pub fn server_init_board(
    chardevs: &Chardevs,
    chrdev: &str,
    log: Option<&str>,
    target: &str,
    memory: Arc<AddressSpace>,
    io: Arc<AddressSpace>,
    clock: Option<Arc<VirtualClock>>,
) -> Result<Attachment> {
    let machine = NoneMachine { big_endian: target_big_endian(target), clock, memory, io };
    serve(chardevs, chrdev, log, machine)
}

fn serve(
    chardevs: &Chardevs,
    chrdev: &str,
    log: Option<&str>,
    machine: NoneMachine,
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
    let qtest = Qtest::new(machine);
    qtest.set_log(open_log(log));
    chr.attach(Arc::new(QtestFrontend(qtest)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvm_base::ClockType;

    #[test]
    fn clock_only_moves_forward() {
        let c = Clock::manual(ClockType::Virtual);
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
