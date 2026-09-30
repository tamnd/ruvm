// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::HashMap;
use std::io::{self, Cursor, Write};
use std::sync::{Arc, Mutex};

use ruvm_accel_qtest::{IrqHandler, Qtest, QtestBackend, open_log};
use ruvm_base::{Error, Result};

/// A write sink tests can read back.
#[derive(Clone, Default)]
struct Buf(Arc<Mutex<Vec<u8>>>);

impl Buf {
    fn take(&self) -> String {
        let mut v = self.0.lock().unwrap();
        String::from_utf8(std::mem::take(&mut *v)).unwrap()
    }
}

impl Write for Buf {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A device with an unnamed input list, an unnamed output list and a named output list.
struct Dev {
    num_in: usize,
    outs: Vec<(Option<&'static str>, usize)>,
}

#[derive(Default)]
struct TestBackend {
    big_endian: bool,
    mem: HashMap<u64, u8>,
    ports: HashMap<u16, u32>,
    port_writes: Vec<(u16, usize, u32)>,
    clock: i64,
    deadline: i64,
    devices: HashMap<&'static str, Dev>,
    in_handler: Option<(IrqHandler, usize)>,
    out_handlers: Vec<(IrqHandler, usize)>,
    irq_sets: Vec<(Option<String>, i32, i32)>,
    not_enabled: bool,
    rtas: bool,
    csr: bool,
}

impl TestBackend {
    fn new() -> Self {
        let mut devices = HashMap::new();
        devices
            .insert("ioapic", Dev { num_in: 24, outs: vec![(None, 4), (Some("sysbus-irq"), 2)] });
        devices.insert("other", Dev { num_in: 2, outs: vec![] });
        devices.insert("nogpio", Dev { num_in: 0, outs: vec![] });
        TestBackend { deadline: -1, devices, ..Default::default() }
    }
}

impl QtestBackend for TestBackend {
    type Device = &'static str;

    fn qtest_enabled(&self) -> bool {
        !self.not_enabled
    }

    fn big_endian(&self) -> bool {
        self.big_endian
    }

    fn memory_read(&mut self, addr: u64, buf: &mut [u8]) {
        for (i, b) in buf.iter_mut().enumerate() {
            *b = self.mem.get(&(addr + i as u64)).copied().unwrap_or(0);
        }
    }

    fn memory_write(&mut self, addr: u64, buf: &[u8]) {
        for (i, b) in buf.iter().enumerate() {
            self.mem.insert(addr + i as u64, *b);
        }
    }

    fn port_read(&mut self, addr: u16, _size: usize) -> u32 {
        self.ports.get(&addr).copied().unwrap_or(0xffff_ffff)
    }

    fn port_write(&mut self, addr: u16, size: usize, value: u32) {
        self.port_writes.push((addr, size, value));
    }

    fn clock_get_ns(&mut self) -> i64 {
        self.clock
    }

    fn clock_deadline_ns_all(&mut self) -> i64 {
        self.deadline
    }

    fn clock_advance_virtual_time(&mut self, dest: i64) -> i64 {
        if dest > self.clock {
            self.clock = dest;
        }
        self.clock
    }

    fn resolve_device(&mut self, path: &str) -> Option<&'static str> {
        self.devices.keys().find(|k| **k == path).copied()
    }

    fn irq_intercept_in(&mut self, dev: &&'static str, handler: &IrqHandler) -> bool {
        let d = &self.devices[dev];
        if d.num_in == 0 && d.outs.is_empty() {
            return false;
        }
        self.in_handler = Some((handler.clone(), d.num_in));
        true
    }

    fn irq_intercept_out(
        &mut self,
        dev: &&'static str,
        name: Option<&str>,
        handler: &IrqHandler,
    ) -> bool {
        let d = &self.devices[dev];
        let Some(&(_, n)) = d.outs.iter().find(|(nm, _)| *nm == name) else {
            return false;
        };
        for _ in 0..n {
            self.out_handlers.push((handler.clone(), n));
        }
        true
    }

    fn set_irq_in(&mut self, _dev: &&'static str, name: Option<&str>, num: i32, level: i32) {
        self.irq_sets.push((name.map(str::to_string), num, level));
        // Loop the line back like a device whose input is intercepted.
        if let Some((h, _)) = &self.in_handler {
            h.set(num, level);
        }
    }

    fn module_load(&mut self, prefix: &str, name: &str) -> Result<bool> {
        match (prefix, name) {
            ("block-", "curl") => Ok(true),
            ("block-", "broken") => Err(Error::generic("could not load")),
            _ => Ok(false),
        }
    }

    fn has_rtas(&self) -> bool {
        self.rtas
    }

    fn rtas_call(&mut self, name: &str, _nargs: u32, _args: u64, _nret: u32, _rets: u64) -> bool {
        name == "get-time-of-day"
    }

    fn has_csr(&self) -> bool {
        self.csr
    }

    fn csr_call(&mut self, cmd: &str, _cpu: u64, csrno: i32, val: u64) -> Option<u64> {
        match (cmd, csrno) {
            (_, 0xfff) => None,
            ("get_csr", _) => Some(0x1234),
            _ => Some(val),
        }
    }
}

struct Harness {
    q: Qtest<TestBackend>,
    out: Buf,
    log: Buf,
}

fn harness_with(backend: TestBackend) -> Harness {
    let q = Qtest::new(backend);
    let out = Buf::default();
    let log = Buf::default();
    q.set_log(Some(Box::new(log.clone())));
    q.open(Box::new(out.clone()));
    Harness { q, out, log }
}

fn harness() -> Harness {
    harness_with(TestBackend::new())
}

impl Harness {
    /// Sends one command line and returns everything written back.
    fn cmd(&self, line: &str) -> String {
        self.q.receive(format!("{line}\n").as_bytes()).unwrap();
        self.out.take()
    }

    /// Sends one command line that must trip an assertion, and returns its text.
    fn assert_fails(&self, line: &str) -> String {
        let e = self.q.receive(format!("{line}\n").as_bytes()).unwrap_err();
        assert_eq!(self.out.take(), "");
        e.message().to_string()
    }
}

/// Replaces every log timestamp with `T` after checking it has the `%.06f` shape.
fn normalize(log: &str) -> String {
    let mut out = String::new();
    for line in log.lines() {
        let (head, rest) = line.split_once(']').unwrap();
        let (tag, stamp) = head.split_at(3);
        let (plus, stamp) = match stamp.strip_prefix('+') {
            Some(s) => ("+", s),
            None => ("", stamp),
        };
        let (int, frac) = stamp.split_once('.').unwrap();
        assert!(!int.is_empty() && int.bytes().all(|b| b.is_ascii_digit()), "{line}");
        assert!(frac.len() == 6 && frac.bytes().all(|b| b.is_ascii_digit()), "{line}");
        out.push_str(&format!("{tag}{plus}T]{rest}\n"));
    }
    out
}

#[test]
fn port_io() {
    let h = harness();
    assert_eq!(h.cmd("outb 0x80 0x1ff"), "OK\n");
    assert_eq!(h.cmd("outw 0x70 70000"), "OK\n");
    assert_eq!(h.cmd("outl 0xcf8 0x80000000"), "OK\n");
    h.q.with_backend(|b| {
        assert_eq!(
            b.port_writes,
            vec![(0x80, 1, 0xff), (0x70, 2, 70000 & 0xffff), (0xcf8, 4, 0x8000_0000)]
        );
        b.ports.insert(0x71, 0x12345678);
    });
    assert_eq!(h.cmd("inb 0x71"), "OK 0x0078\n");
    assert_eq!(h.cmd("inw 0x71"), "OK 0x5678\n");
    assert_eq!(h.cmd("inl 0x71"), "OK 0x12345678\n");
    assert_eq!(h.cmd("inb 0x72"), "OK 0x00ff\n");
    assert_eq!(h.cmd("inl 0x72"), "OK 0xffffffff\n");
    // Octal and decimal, as strtoul() with base 0.
    assert_eq!(h.cmd("inl 0161"), "OK 0x12345678\n");
    assert_eq!(h.cmd("inl 113"), "OK 0x12345678\n");
}

#[test]
fn port_io_assertions() {
    let h = harness();
    assert_eq!(
        h.assert_fails("outb 0x80"),
        "qtest_process_command: assertion failed: (words[1] && words[2])"
    );
    assert_eq!(
        h.assert_fails("outb 0x10000 1"),
        "qtest_process_command: assertion failed: (addr <= 0xffff)"
    );
    assert_eq!(h.assert_fails("outb zz 1"), "qtest_process_command: assertion failed: (ret == 0)");
    assert_eq!(h.assert_fails("inb"), "qtest_process_command: assertion failed: (words[1])");
    assert_eq!(h.assert_fails("inw 0x80x"), "qtest_process_command: assertion failed: (ret == 0)");
}

#[test]
fn memory_values_little_endian() {
    let h = harness();
    assert_eq!(h.cmd("writeb 0x1000 0x1234"), "OK\n");
    assert_eq!(h.cmd("readb 0x1000"), "OK 0x0000000000000034\n");
    assert_eq!(h.cmd("writew 0x1000 0xabcd"), "OK\n");
    assert_eq!(h.cmd("read 0x1000 2"), "OK 0xcdab\n");
    assert_eq!(h.cmd("readw 0x1000"), "OK 0x000000000000abcd\n");
    assert_eq!(h.cmd("writel 0x1000 0x11223344"), "OK\n");
    assert_eq!(h.cmd("readl 0x1000"), "OK 0x0000000011223344\n");
    assert_eq!(h.cmd("writeq 0x1000 0x0102030405060708"), "OK\n");
    assert_eq!(h.cmd("readq 0x1000"), "OK 0x0102030405060708\n");
    assert_eq!(h.cmd("read 0x1000 8"), "OK 0x0807060504030201\n");
    // A negative value wraps, as strtoull() does.
    assert_eq!(h.cmd("writeq 0x2000 -1"), "OK\n");
    assert_eq!(h.cmd("readq 0x2000"), "OK 0xffffffffffffffff\n");
    assert_eq!(h.cmd("endianness"), "OK little\n");
}

#[test]
fn memory_values_big_endian() {
    let h = harness_with(TestBackend { big_endian: true, ..TestBackend::new() });
    assert_eq!(h.cmd("endianness"), "OK big\n");
    assert_eq!(h.cmd("writew 0x1000 0xabcd"), "OK\n");
    assert_eq!(h.cmd("read 0x1000 2"), "OK 0xabcd\n");
    assert_eq!(h.cmd("readw 0x1000"), "OK 0x000000000000abcd\n");
    assert_eq!(h.cmd("writel 0x1000 0x11223344"), "OK\n");
    assert_eq!(h.cmd("read 0x1000 4"), "OK 0x11223344\n");
    assert_eq!(h.cmd("readl 0x1000"), "OK 0x0000000011223344\n");
    assert_eq!(h.cmd("writeq 0x1000 0x0102030405060708"), "OK\n");
    assert_eq!(h.cmd("read 0x1000 8"), "OK 0x0102030405060708\n");
    assert_eq!(h.cmd("readq 0x1000"), "OK 0x0102030405060708\n");
    assert_eq!(h.cmd("readb 0x1007"), "OK 0x0000000000000008\n");
}

#[test]
fn memory_value_assertions() {
    let h = harness();
    assert_eq!(
        h.assert_fails("writel 0x1000"),
        "qtest_process_command: assertion failed: (words[1] && words[2])"
    );
    assert_eq!(
        h.assert_fails("writel 0x1000 x"),
        "qtest_process_command: assertion failed: (ret == 0)"
    );
    assert_eq!(h.assert_fails("readq"), "qtest_process_command: assertion failed: (words[1])");
    assert_eq!(
        h.assert_fails("writeq 0x1000 0x10000000000000000"),
        "qtest_process_command: assertion failed: (ret == 0)"
    );
}

#[test]
fn read_write_hex() {
    let h = harness();
    assert_eq!(h.cmd("write 0x100 4 0xdeadBEEF"), "OK\n");
    assert_eq!(h.cmd("read 0x100 4"), "OK 0xdeadbeef\n");
    // Short data is zero filled at the end.
    assert_eq!(h.cmd("write 0x100 4 0x11"), "OK\n");
    assert_eq!(h.cmd("read 0x100 4"), "OK 0x11000000\n");
    // An odd trailing digit is dropped, and the prefix is not checked.
    assert_eq!(h.cmd("write 0x100 2 zz123"), "OK\n");
    assert_eq!(h.cmd("read 0x100 2"), "OK 0x1200\n");
    // An invalid digit sets every bit it touches, as hex2nib() returning -1 does.
    assert_eq!(h.cmd("write 0x100 3 0xg1a-zz"), "OK\n");
    assert_eq!(h.cmd("read 0x100 3"), "OK 0xf1ffff\n");
    assert_eq!(h.cmd("write 0x100 4 0x"), "ERR invalid argument size\n");
    assert_eq!(
        h.assert_fails("write 0x100 4"),
        "qtest_process_command: assertion failed: (words[1] && words[2] && words[3])"
    );
    assert_eq!(
        h.assert_fails("read 0x100"),
        "qtest_process_command: assertion failed: (words[1] && words[2])"
    );
    assert_eq!(h.assert_fails("read 0x100 0"), "qtest_process_command: assertion failed: (len)");
    assert_eq!(
        h.assert_fails("read 0x100 1k"),
        "qtest_process_command: assertion failed: (ret == 0)"
    );
}

#[test]
fn memset() {
    let h = harness();
    assert_eq!(h.cmd("memset 0x200 3 0x1a5"), "OK\n");
    assert_eq!(h.cmd("read 0x1ff 5"), "OK 0x00a5a5a500\n");
    assert_eq!(h.cmd("memset 0x200 0 0xff"), "OK\n");
    assert_eq!(h.cmd("read 0x200 1"), "OK 0xa5\n");
    assert_eq!(
        h.assert_fails("memset 0x200 3"),
        "qtest_process_command: assertion failed: (words[1] && words[2] && words[3])"
    );
    assert_eq!(
        h.assert_fails("memset 0x200 3 q"),
        "qtest_process_command: assertion failed: (ret == 0)"
    );
}

#[test]
fn base64() {
    let h = harness();
    assert_eq!(h.cmd("b64write 0x300 5 aGVsbG8="), "OK\n");
    assert_eq!(h.cmd("read 0x300 5"), "OK 0x68656c6c6f\n");
    assert_eq!(h.cmd("b64read 0x300 5"), "OK aGVsbG8=\n");
    assert_eq!(h.cmd("b64read 0x300 4"), "OK aGVsbA==\n");
    assert_eq!(h.cmd("b64read 0x300 3"), "OK aGVs\n");
    assert_eq!(h.cmd("b64read 0x300 0"), "OK \n");
    // Characters outside the alphabet are skipped.
    assert_eq!(h.cmd("b64write 0x300 3 A.A.A.A"), "OK\n");
    assert_eq!(h.cmd("read 0x300 3"), "OK 0x000000\n");
    // Too much data is cut to the told size, too little writes what there is.
    assert_eq!(h.cmd("b64write 0x300 2 /////w=="), "OK\n");
    assert_eq!(h.cmd("read 0x300 5"), "OK 0xffff006c6f\n");
    assert_eq!(h.cmd("b64write 0x300 5 AAAA"), "OK\n");
    assert_eq!(h.cmd("read 0x300 5"), "OK 0x0000006c6f\n");
    assert_eq!(h.cmd("b64write 0x300 1 QQ"), "ERR invalid argument size\n");
    assert_eq!(
        h.assert_fails("b64read 0x300"),
        "qtest_process_command: assertion failed: (words[1] && words[2])"
    );
    assert_eq!(
        h.assert_fails("b64write 0x300 1"),
        "qtest_process_command: assertion failed: (words[1] && words[2] && words[3])"
    );
}

#[test]
fn b64write_mismatch_is_logged() {
    let h = harness();
    h.log.take();
    assert_eq!(h.cmd("b64write 0x300 2 aGVsbG8="), "OK\n");
    assert_eq!(
        normalize(&h.log.take()),
        "[R +T] b64write 0x300 2 aGVsbG8=\n\
         [S +T] b64write: data length mismatch (told 2, found 5)\n\
         [S +T] OK\n"
    );
}

#[test]
fn clock() {
    let h = harness();
    assert_eq!(
        h.cmd("clock_step"),
        "FAIL cannot advance clock to the next deadline because there is no pending deadline\n"
    );
    h.q.with_backend(|b| b.deadline = 500);
    assert_eq!(h.cmd("clock_step"), "OK 500\n");
    assert_eq!(h.cmd("clock_step 100"), "OK 600\n");
    assert_eq!(h.cmd("clock_step 0"), "FAIL could not advance time\n");
    assert_eq!(h.cmd("clock_step -5"), "FAIL could not advance time\n");
    assert_eq!(h.cmd("clock_set 1000"), "OK 1000\n");
    assert_eq!(h.cmd("clock_set 1000"), "OK 1000\n");
    assert_eq!(h.cmd("clock_set 10"), "FAIL 1000\n");
    assert_eq!(
        h.assert_fails("clock_step 1.5"),
        "qtest_process_command: assertion failed: (ret == 0)"
    );
    assert_eq!(h.assert_fails("clock_set"), "qtest_process_command: assertion failed: (words[1])");
    assert_eq!(
        h.assert_fails("clock_set x"),
        "qtest_process_command: assertion failed: (ret == 0)"
    );
}

#[test]
fn clock_needs_qtest_accel() {
    let h = harness_with(TestBackend { not_enabled: true, ..TestBackend::new() });
    assert_eq!(h.cmd("clock_step 10"), "FAIL Unknown command 'clock_step'\n");
    assert_eq!(h.cmd("clock_set 10"), "FAIL Unknown command 'clock_set'\n");
}

#[test]
fn irq_intercept_in() {
    let h = harness();
    assert_eq!(h.cmd("irq_intercept_in nosuch"), "FAIL Unknown device\n");
    assert_eq!(
        h.cmd("irq_intercept_in ioapic name"),
        "FAIL Interception of named in-GPIOs not yet supported\n"
    );
    assert_eq!(h.cmd("irq_intercept_in nogpio"), "FAIL No intercepts installed\n");
    assert_eq!(h.cmd("irq_intercept_in ioapic"), "OK\n");
    assert_eq!(h.cmd("irq_intercept_in ioapic"), "OK\n");
    assert_eq!(h.cmd("irq_intercept_in other"), "FAIL IRQ intercept already enabled\n");
    assert_eq!(h.cmd("irq_intercept_out other"), "FAIL IRQ intercept already enabled\n");

    // The async line comes before the reply of the command that caused it.
    assert_eq!(h.cmd("set_irq_in ioapic unnamed-gpio-in 4 1"), "IRQ raise 4\nOK\n");
    assert_eq!(h.cmd("set_irq_in ioapic unnamed-gpio-in 4 1"), "OK\n");
    assert_eq!(h.cmd("set_irq_in ioapic unnamed-gpio-in 4 0"), "IRQ lower 4\nOK\n");
    assert_eq!(h.cmd("set_irq_in ioapic my-gpio 3 -1"), "IRQ raise 3\nOK\n");
    h.q.with_backend(|b| {
        assert_eq!(
            b.irq_sets,
            vec![(None, 4, 1), (None, 4, 1), (None, 4, 0), (Some("my-gpio".to_string()), 3, -1)]
        );
    });

    // An interrupt raised outside any command goes out on its own.
    let handler = h.q.with_backend(|b| b.in_handler.clone().unwrap().0);
    handler.set(9, 1);
    assert_eq!(h.out.take(), "IRQ raise 9\n");
    handler.set(9, 1);
    assert_eq!(h.out.take(), "");

    // A new connection starts from all lines low, but keeps the intercept.
    h.q.close();
    let out = Buf::default();
    h.q.open(Box::new(out.clone()));
    handler.set(9, 1);
    assert_eq!(out.take(), "IRQ raise 9\n");
    h.q.receive(b"irq_intercept_in ioapic\n").unwrap();
    assert_eq!(out.take(), "OK\n");
}

#[test]
fn irq_intercept_out() {
    let h = harness();
    assert_eq!(h.cmd("irq_intercept_out ioapic nosuch"), "FAIL No intercepts installed\n");
    assert_eq!(h.cmd("irq_intercept_out ioapic sysbus-irq"), "OK\n");
    let handlers = h.q.with_backend(|b| b.out_handlers.clone());
    assert_eq!(handlers.len(), 2);
    handlers[1].0.set(1, 1);
    handlers[1].0.set(1, 5);
    handlers[0].0.set(0, 0);
    handlers[1].0.set(1, 0);
    assert_eq!(h.out.take(), "IRQ raise 1\nIRQ raise 1\nIRQ lower 1\n");

    let h = harness();
    assert_eq!(h.cmd("irq_intercept_out ioapic"), "OK\n");
    assert_eq!(h.q.with_backend(|b| b.out_handlers.len()), 4);
    assert_eq!(
        h.assert_fails("irq_intercept_out"),
        "qtest_process_command: assertion failed: (words[1])"
    );
}

#[test]
fn set_irq_in_errors() {
    let h = harness();
    assert_eq!(h.cmd("set_irq_in nosuch x 1 1"), "FAIL Unknown device\n");
    assert_eq!(
        h.assert_fails("set_irq_in ioapic x 1"),
        "qtest_process_command: assertion failed: (words[1] && words[2] && words[3] && words[4])"
    );
    assert_eq!(
        h.assert_fails("set_irq_in ioapic x 0x100000000 1"),
        "qtest_process_command: assertion failed: (!ret)"
    );
    assert_eq!(
        h.assert_fails("set_irq_in ioapic x 1 y"),
        "qtest_process_command: assertion failed: (!ret)"
    );
}

#[test]
fn module_load_and_qom_tests() {
    let h = harness();
    assert_eq!(h.cmd("module_load block- curl"), "OK\n");
    assert_eq!(h.cmd("module_load block- nosuch"), "FAIL\n");
    assert_eq!(h.cmd("module_load block- broken"), "FAIL\n");
    assert_eq!(
        h.assert_fails("module_load block-"),
        "qtest_process_command: assertion failed: (words[1] && words[2])"
    );
    assert_eq!(h.cmd("qom-tests"), "OK\n");
}

#[test]
fn rtas_and_csr_callbacks() {
    let h = harness();
    assert_eq!(h.cmd("rtas get-time-of-day 0 0x0 8 0x100"), "FAIL Unknown command 'rtas'\n");
    assert_eq!(h.cmd("csr get_csr 0 0x300 0"), "FAIL Unknown command 'csr'\n");

    let h = harness_with(TestBackend { rtas: true, csr: true, ..TestBackend::new() });
    assert_eq!(h.cmd("rtas get-time-of-day 0 0x0 8 0x100"), "OK 0\n");
    assert_eq!(h.cmd("rtas nosuch 0 0x0 8 0x100"), "OK 18446744073709551612\n");
    assert_eq!(
        h.assert_fails("rtas nosuch 0 0x0 8"),
        "qtest_process_command: assertion failed: (rc == 0)"
    );
    assert_eq!(h.cmd("csr get_csr 0 0x300 0"), "OK 0 1234\n");
    assert_eq!(h.cmd("csr set_csr 0 0x300 0xABC"), "OK 0 abc\n");
    assert_eq!(
        h.assert_fails("csr get_csr 0 0xfff 0"),
        "qtest_process_command: assertion failed: (ret == RISCV_EXCP_NONE)"
    );
    assert_eq!(
        h.assert_fails("csr get_csr 0 0x300"),
        "qtest_process_command: assertion failed: (rc == 0)"
    );
}

#[test]
fn line_handling() {
    let h = harness();
    assert_eq!(h.cmd(""), "");
    assert_eq!(h.cmd("# a comment"), "");
    assert_eq!(h.cmd("#readb 0"), "");
    // Words are split on every single space, as g_strsplit() does.
    assert_eq!(h.cmd(" "), "FAIL Unknown command ''\n");
    assert_eq!(h.cmd("nosuch 1 2"), "FAIL Unknown command 'nosuch'\n");
    assert_eq!(h.cmd("endianness\r"), "FAIL Unknown command 'endianness\r'\n");

    // Partial lines wait for the rest, and one read may hold many lines.
    h.q.receive(b"endian").unwrap();
    assert_eq!(h.out.take(), "");
    h.q.receive(b"ness\nendianness\nendi").unwrap();
    assert_eq!(h.out.take(), "OK little\nOK little\n");
    h.q.receive(b"anness\n").unwrap();
    assert_eq!(h.out.take(), "OK little\n");

    // Anything after a NUL byte on a line is ignored.
    h.q.receive(b"endianness\0junk\n").unwrap();
    assert_eq!(h.out.take(), "OK little\n");
}

#[test]
fn readb_with_two_spaces_is_an_assertion() {
    let h = harness();
    // "readb  0" splits into ["readb", "", "0"], and an empty string does not parse.
    let e = h.q.receive(b"readb  0\n").unwrap_err();
    assert_eq!(e.message(), "qtest_process_command: assertion failed: (ret == 0)");
}

#[test]
fn log_format() {
    let q = Qtest::new(TestBackend::new());
    let log = Buf::default();
    let out = Buf::default();
    q.set_log(Some(Box::new(log.clone())));

    // Before a client connects, received commands are logged and replies are not.
    q.receive(b"endianness\n").unwrap();
    assert_eq!(log.take(), "[R +0.000000] endianness\n");

    q.open(Box::new(out.clone()));
    q.receive(b"writeb 0x10 0x5\nreadb 0x10\n\n# note\nbogus\n").unwrap();
    q.close();
    q.close();
    let text = log.take();
    assert!(text.starts_with("[I 0.0000"), "{text}");
    assert_eq!(
        normalize(&text),
        "[I T] OPENED\n\
         [R +T] writeb 0x10 0x5\n\
         [S +T] OK\n\
         [R +T] readb 0x10\n\
         [S +T] OK 0x0000000000000005\n\
         [R +T]\n\
         [R +T] # note\n\
         [R +T] bogus\n\
         [S +T] FAIL Unknown command 'bogus'\n\
         [I +T] CLOSED\n"
    );
    assert_eq!(out.take(), "OK\nOK 0x0000000000000005\nFAIL Unknown command 'bogus'\n");
}

#[test]
fn irq_lines_are_logged() {
    let h = harness();
    h.cmd("irq_intercept_in ioapic");
    h.log.take();
    h.q.irq_handler().set(2, 1);
    assert_eq!(normalize(&h.log.take()), "[S +T] IRQ raise 2\n");
}

#[test]
fn serve_over_a_stream() {
    let q = Qtest::new(TestBackend::new());
    let log = Buf::default();
    let out = Buf::default();
    q.set_log(Some(Box::new(log.clone())));
    let input = Cursor::new(b"writel 0x40 0xcafe\nreadl 0x40\nendianness".to_vec());
    q.serve(input, out.clone()).unwrap();
    // The last line has no newline, so it waits.
    assert_eq!(out.take(), "OK\nOK 0x000000000000cafe\n");
    assert!(!q.is_opened());
    assert_eq!(
        normalize(&log.take()),
        "[I T] OPENED\n\
         [R +T] writel 0x40 0xcafe\n\
         [S +T] OK\n\
         [R +T] readl 0x40\n\
         [S +T] OK 0x000000000000cafe\n\
         [I +T] CLOSED\n"
    );

    // The partial line survives into the next connection, as the input buffer in qtest.c is
    // never reset. A failed assertion ends serving with an error and no CLOSED line, as QEMU
    // would abort.
    let e = q.serve(Cursor::new(b"\nreadl\n".to_vec()), out.clone()).unwrap_err();
    assert_eq!(e.to_string(), "qtest_process_command: assertion failed: (words[1])");
    assert_eq!(out.take(), "OK little\n");
    assert!(q.is_opened());
    assert!(!normalize(&log.take()).contains("CLOSED"));
}

#[test]
fn no_log() {
    assert!(open_log(Some("none")).is_none());
    assert!(open_log(None).is_some());
    let q = Qtest::new(TestBackend::new());
    let out = Buf::default();
    q.open(Box::new(out.clone()));
    q.receive(b"endianness\n").unwrap();
    assert_eq!(out.take(), "OK little\n");
}

#[test]
fn log_file() {
    let dir = std::env::temp_dir().join(format!("ruvm-qtest-log-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("qtest.log");
    std::fs::write(&path, "stale\n").unwrap();
    let q = Qtest::new(TestBackend::new());
    q.set_log(open_log(Some(path.to_str().unwrap())));
    q.serve(Cursor::new(b"endianness\n".to_vec()), Buf::default()).unwrap();
    q.set_log(None);
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(
        normalize(&text),
        "[I T] OPENED\n[R +T] endianness\n[S +T] OK little\n[I +T] CLOSED\n"
    );
}
