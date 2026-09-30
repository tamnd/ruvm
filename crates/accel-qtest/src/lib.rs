// SPDX-License-Identifier: GPL-2.0-or-later

//! The qtest accelerator and its protocol server, a port of system/qtest.c.
//!
//! qtest is a line based text protocol a test program speaks over a chardev to poke at the
//! machine: memory and port io, the virtual clock and interrupt lines. [`Qtest`] is the protocol
//! engine. It parses command lines, runs them against a [`QtestBackend`], writes the replies and
//! keeps the `-qtest-log` log. The backend is where the memory core, the timers and qdev plug in,
//! so the engine itself knows nothing about them.
//!
//! [`Qtest::serve`] runs one client connection over any reader and writer pair, with the chardev
//! open and close events around it. The chardev crate sits above this one, so the system crate
//! wraps a shared [`Qtest`] in its own chardev frontend and calls [`Qtest::serve`] from it.
//!
//! Where qtest.c has a `g_assert()` on client input, a missing argument or a number that does
//! not parse, QEMU aborts. The engine returns an error with the assertion text instead, and the
//! caller is expected to abort.

#![forbid(unsafe_code)]

mod backend;
mod base64;

use std::fs::File;
use std::io::{self, LineWriter, Read, Write};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use ruvm_base::{Error, Result, error_report};
use ruvm_qapi::cutils;

pub use backend::QtestBackend;

/// `MAX_IRQ`: how many IRQ levels the engine remembers to filter repeated edges.
pub const MAX_IRQ: usize = 256;

/// The name of the QOM type, `TYPE_QTEST`.
pub const TYPE_QTEST: &str = "qtest";

/// A sink the engine writes replies or log lines to.
pub type Sink = Box<dyn Write + Send>;

/// The state the reply path and the IRQ handler share: the client connection, the log file,
/// whether a client is connected, the timer the log timestamps count from and the last level of
/// each intercepted IRQ. These are the file scope statics of qtest.c.
struct Output {
    chr: Option<Sink>,
    log: Option<Sink>,
    opened: bool,
    timer: Option<Instant>,
    irq_levels: [i32; MAX_IRQ],
}

impl Output {
    /// `g_timer_elapsed(timer, NULL)`, which is 0 when there is no timer.
    fn elapsed(&self) -> f64 {
        self.timer.map_or(0.0, |t| t.elapsed().as_secs_f64())
    }

    fn write_log(&mut self, text: &str) {
        if let Some(log) = self.log.as_mut() {
            let _ = log.write_all(text.as_bytes());
            let _ = log.flush();
        }
    }

    /// `qtest_log_timestamp()`.
    fn log_timestamp(&mut self) {
        if self.log.is_none() || !self.opened {
            return;
        }
        let stamp = format!("[S +{:.6}] ", self.elapsed());
        self.write_log(&stamp);
    }

    /// `qtest_log_send()`.
    fn log_send(&mut self, text: &str) {
        if self.log.is_none() || !self.opened {
            return;
        }
        self.log_timestamp();
        self.write_log(text);
    }

    /// `qtest_send()` with `qtest_server_char_be_send()` as the send handler. Write errors are
    /// dropped, as `qemu_chr_fe_write_all()` failures are.
    fn send(&mut self, text: &str) {
        self.log_timestamp();
        if let Some(chr) = self.chr.as_mut() {
            let _ = chr.write_all(text.as_bytes());
            let _ = chr.flush();
        }
        if self.opened {
            self.write_log(text);
        }
    }
}

type Shared = Arc<Mutex<Output>>;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The IRQ observer the engine hands to [`QtestBackend::irq_intercept_in`] and
/// [`QtestBackend::irq_intercept_out`]. Cloning it is cheap, and it may be called from any
/// thread.
#[derive(Clone)]
pub struct IrqHandler(Shared);

impl std::fmt::Debug for IrqHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("IrqHandler")
    }
}

impl IrqHandler {
    /// `qtest_irq_handler()`: sends `IRQ raise N` or `IRQ lower N` when line `n` changes level.
    /// qtest.c indexes a fixed array with `n`, so a line number at or past [`MAX_IRQ`] is a
    /// QEMU bug. Here such a line is reported on every call instead.
    pub fn set(&self, n: i32, level: i32) {
        let mut out = lock(&self.0);
        let slot = usize::try_from(n).ok().filter(|&i| i < MAX_IRQ);
        if let Some(i) = slot {
            if out.irq_levels[i] == level {
                return;
            }
            out.irq_levels[i] = level;
        }
        let dir = if level != 0 { "raise" } else { "lower" };
        out.send(&format!("IRQ {dir} {n}\n"));
    }
}

/// The part of the engine that runs commands: the backend, the partial input line and the
/// device whose IRQs are intercepted.
struct Engine<B: QtestBackend> {
    backend: B,
    inbuf: Vec<u8>,
    irq_intercept_dev: Option<B::Device>,
}

/// The qtest protocol server, the `QTest` object and the statics of system/qtest.c.
///
/// All methods take `&self`, so one `Arc<Qtest<B>>` can be shared between the chardev frontend
/// and the code that owns the machine.
pub struct Qtest<B: QtestBackend> {
    out: Shared,
    engine: Mutex<Engine<B>>,
}

impl<B: QtestBackend> std::fmt::Debug for Qtest<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let out = lock(&self.out);
        f.debug_struct("Qtest")
            .field("opened", &out.opened)
            .field("log", &out.log.is_some())
            .finish_non_exhaustive()
    }
}

/// `g_assert()`: the error returned when client input trips an assertion in qtest.c.
fn check(cond: bool, expr: &str) -> Result<()> {
    if cond {
        Ok(())
    } else {
        Err(Error::generic(format!("qtest_process_command: assertion failed: ({expr})")))
    }
}

/// `qemu_strtou64(word, NULL, 0, &value)`, `None` where it returns an error.
fn strtou64(word: Option<&str>) -> Option<u64> {
    cutils::strtou64(word?, 0, true).ok().map(|(v, _)| v)
}

/// `qemu_strtoul()`. `unsigned long` is 64 bits on the hosts ruvm supports.
fn strtoul(word: Option<&str>) -> Option<u64> {
    strtou64(word)
}

/// `qemu_strtoi64(word, NULL, 0, &value)`.
fn strtoi64(word: Option<&str>) -> Option<i64> {
    cutils::strtoi64(word?, 0, true).ok().map(|(v, _)| v)
}

/// `qemu_strtoi(word, NULL, 0, &value)`: `strtoll()` with a range check against `int`.
fn strtoi(word: Option<&str>) -> Option<i32> {
    strtoi64(word).and_then(|v| i32::try_from(v).ok())
}

/// `hex2nib()`.
fn hex2nib(ch: u8) -> i32 {
    match ch {
        b'0'..=b'9' => i32::from(ch - b'0'),
        b'a'..=b'f' => 10 + i32::from(ch - b'a'),
        b'A'..=b'F' => 10 + i32::from(ch - b'A'),
        _ => -1,
    }
}

/// `qemu_hexdump_line(NULL, data, len, 0, 0)`: lower case hex digits with no separators.
fn hexdump(data: &[u8]) -> String {
    let mut s = String::with_capacity(data.len() * 2);
    for b in data {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// `H_SUCCESS` and `H_PARAMETER` from include/hw/ppc/spapr.h, the results `rtas` replies with.
const H_SUCCESS: u64 = 0;
const H_PARAMETER: u64 = -4i64 as u64;

/// Opens the `-qtest-log` destination the way `qtest_server_start()` does: no option means
/// stderr, `none` means no log, and anything else is a file that is created or truncated. A file
/// that cannot be opened silently means no log, as the `fopen()` result is not checked in
/// qtest.c.
pub fn open_log(path: Option<&str>) -> Option<Sink> {
    match path {
        None => Some(Box::new(io::stderr())),
        Some("none") => None,
        Some(p) => File::create(p).ok().map(|f| Box::new(LineWriter::new(f)) as Sink),
    }
}

impl<B: QtestBackend> Qtest<B> {
    /// A server with no client and no log. `qtest_server_start()` also opens the log, which is
    /// [`Qtest::set_log`] here.
    pub fn new(backend: B) -> Self {
        Qtest {
            out: Arc::new(Mutex::new(Output {
                chr: None,
                log: None,
                opened: false,
                timer: None,
                irq_levels: [0; MAX_IRQ],
            })),
            engine: Mutex::new(Engine { backend, inbuf: Vec::new(), irq_intercept_dev: None }),
        }
    }

    /// Sets where the log goes, `qtest_log_fp`. `None` turns logging off. See [`open_log`] for
    /// the `-qtest-log` rules.
    pub fn set_log(&self, log: Option<Sink>) {
        lock(&self.out).log = log;
    }

    /// The IRQ observer, for wiring interrupts that do not go through the intercept commands.
    pub fn irq_handler(&self) -> IrqHandler {
        IrqHandler(self.out.clone())
    }

    /// Runs `f` with the backend locked.
    pub fn with_backend<R>(&self, f: impl FnOnce(&mut B) -> R) -> R {
        f(&mut lock(&self.engine).backend)
    }

    /// Whether a client is connected, `qtest_opened`.
    pub fn is_opened(&self) -> bool {
        lock(&self.out).opened
    }

    /// Attaches the client connection replies are written to, and runs the `CHR_EVENT_OPENED`
    /// half of `qtest_event()`: IRQ levels go back to 0, the log timer restarts and the log gets
    /// `[I 0.000000] OPENED`.
    pub fn open(&self, chr: Sink) {
        let mut out = lock(&self.out);
        out.chr = Some(chr);
        out.irq_levels = [0; MAX_IRQ];
        out.timer = Some(Instant::now());
        out.opened = true;
        let line = format!("[I {:.6}] OPENED\n", out.elapsed());
        out.write_log(&line);
    }

    /// The `CHR_EVENT_CLOSED` half of `qtest_event()`: the log gets `[I +t] CLOSED` and the
    /// client connection is dropped. Does nothing when no client is connected.
    pub fn close(&self) {
        let mut out = lock(&self.out);
        if !out.opened {
            return;
        }
        out.opened = false;
        let line = format!("[I +{:.6}] CLOSED\n", out.elapsed());
        out.write_log(&line);
        out.timer = None;
        out.chr = None;
    }

    /// `qtest_read()` and `qtest_process_inbuf()`: appends `buf` to the input and runs every
    /// complete line. A partial line waits for the rest. An error is a failed assertion, and
    /// the lines after the failing one stay unprocessed.
    pub fn receive(&self, buf: &[u8]) -> Result<()> {
        let mut engine = lock(&self.engine);
        engine.inbuf.extend_from_slice(buf);
        while let Some(end) = engine.inbuf.iter().position(|&b| b == b'\n') {
            let mut cmd: Vec<u8> = engine.inbuf.drain(..=end).collect();
            cmd.pop();
            // g_strndup() stops at a NUL byte.
            if let Some(nul) = cmd.iter().position(|&b| b == 0) {
                cmd.truncate(nul);
            }
            let cmd = String::from_utf8_lossy(&cmd);
            // g_strsplit(cmd, " ", 0), which turns an empty line into no words at all.
            let words: Vec<&str> =
                if cmd.is_empty() { Vec::new() } else { cmd.split(' ').collect() };
            self.process(&mut engine, &words)?;
        }
        Ok(())
    }

    /// Runs one command that is already split into words, as `qtest_process_command()`.
    pub fn process_command(&self, words: &[&str]) -> Result<()> {
        let mut engine = lock(&self.engine);
        self.process(&mut engine, words)
    }

    /// Serves one client: [`Qtest::open`] with `writer`, then [`Qtest::receive`] on everything
    /// read from `reader` until it reports end of file, then [`Qtest::close`]. A read error other
    /// than `Interrupted` also ends the connection and is returned after the close. A failed
    /// assertion is returned as an error without a close, since QEMU would have aborted.
    pub fn serve<R: Read, W: Write + Send + 'static>(
        &self,
        mut reader: R,
        writer: W,
    ) -> io::Result<()> {
        self.open(Box::new(writer));
        // qtest_can_read() asks for at most 1024 bytes at a time.
        let mut buf = [0u8; 1024];
        let result = loop {
            match reader.read(&mut buf) {
                Ok(0) => break Ok(()),
                Ok(n) => {
                    if let Err(e) = self.receive(&buf[..n]) {
                        return Err(io::Error::other(e.message().to_string()));
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => break Err(e),
            }
        };
        self.close();
        result
    }

    fn send(&self, text: &str) {
        lock(&self.out).send(text);
    }

    fn log_send(&self, text: &str) {
        lock(&self.out).log_send(text);
    }

    /// `qtest_process_command()`.
    fn process(&self, engine: &mut Engine<B>, words: &[&str]) -> Result<()> {
        {
            let mut out = lock(&self.out);
            if out.log.is_some() {
                let mut line = format!("[R +{:.6}]", out.elapsed());
                for w in words {
                    line.push(' ');
                    line.push_str(w);
                }
                line.push('\n');
                out.write_log(&line);
            }
        }

        let Some(&command) = words.first() else {
            return Ok(());
        };
        if command.starts_with('#') {
            return Ok(());
        }
        let word = |i: usize| words.get(i).copied();
        let be = engine.backend.big_endian();
        let b = &mut engine.backend;

        match command {
            "irq_intercept_out" | "irq_intercept_in" => {
                check(word(1).is_some(), "words[1]")?;
                let is_named = word(2).is_some();
                let is_outbound = command == "irq_intercept_out";
                let Some(dev) = b.resolve_device(words[1]) else {
                    self.send("FAIL Unknown device\n");
                    return Ok(());
                };
                if is_named && !is_outbound {
                    self.send("FAIL Interception of named in-GPIOs not yet supported\n");
                    return Ok(());
                }
                if let Some(cur) = &engine.irq_intercept_dev {
                    if *cur != dev {
                        self.send("FAIL IRQ intercept already enabled\n");
                    } else {
                        self.send("OK\n");
                    }
                    return Ok(());
                }
                let handler = self.irq_handler();
                let ok = if is_outbound {
                    b.irq_intercept_out(&dev, word(2), &handler)
                } else {
                    b.irq_intercept_in(&dev, &handler)
                };
                if ok {
                    engine.irq_intercept_dev = Some(dev);
                    self.send("OK\n");
                } else {
                    self.send("FAIL No intercepts installed\n");
                }
            }
            "set_irq_in" => {
                check(
                    word(1).is_some()
                        && word(2).is_some()
                        && word(3).is_some()
                        && word(4).is_some(),
                    "words[1] && words[2] && words[3] && words[4]",
                )?;
                let Some(dev) = b.resolve_device(words[1]) else {
                    self.send("FAIL Unknown device\n");
                    return Ok(());
                };
                let name = if words[2] == "unnamed-gpio-in" { None } else { Some(words[2]) };
                let num = strtoi(word(3));
                check(num.is_some(), "!ret")?;
                let level = strtoi(word(4));
                check(level.is_some(), "!ret")?;
                b.set_irq_in(&dev, name, num.unwrap_or(0), level.unwrap_or(0));
                self.send("OK\n");
            }
            "outb" | "outw" | "outl" => {
                check(word(1).is_some() && word(2).is_some(), "words[1] && words[2]")?;
                let addr = strtoul(word(1));
                check(addr.is_some(), "ret == 0")?;
                let value = strtoul(word(2));
                check(value.is_some(), "ret == 0")?;
                let addr = addr.unwrap_or(0);
                let value = value.unwrap_or(0);
                check(addr <= 0xffff, "addr <= 0xffff")?;
                let (size, value) = match command.as_bytes()[3] {
                    b'b' => (1, value as u8 as u32),
                    b'w' => (2, value as u16 as u32),
                    _ => (4, value as u32),
                };
                b.port_write(addr as u16, size, value);
                self.send("OK\n");
            }
            "inb" | "inw" | "inl" => {
                check(word(1).is_some(), "words[1]")?;
                let addr = strtoul(word(1));
                check(addr.is_some(), "ret == 0")?;
                let addr = addr.unwrap_or(0);
                check(addr <= 0xffff, "addr <= 0xffff")?;
                let value = match command.as_bytes()[2] {
                    b'b' => b.port_read(addr as u16, 1) & 0xff,
                    b'w' => b.port_read(addr as u16, 2) & 0xffff,
                    _ => b.port_read(addr as u16, 4),
                };
                self.send(&format!("OK 0x{value:04x}\n"));
            }
            "writeb" | "writew" | "writel" | "writeq" => {
                check(word(1).is_some() && word(2).is_some(), "words[1] && words[2]")?;
                let addr = strtou64(word(1));
                check(addr.is_some(), "ret == 0")?;
                let value = strtou64(word(2));
                check(value.is_some(), "ret == 0")?;
                let addr = addr.unwrap_or(0);
                let value = value.unwrap_or(0);
                let bytes = if be { value.to_be_bytes() } else { value.to_le_bytes() };
                let size = match command.as_bytes()[5] {
                    b'b' => 1,
                    b'w' => 2,
                    b'l' => 4,
                    _ => 8,
                };
                // The value truncated to `size` bytes, in target order.
                let data = if be { &bytes[8 - size..] } else { &bytes[..size] };
                b.memory_write(addr, data);
                self.send("OK\n");
            }
            "readb" | "readw" | "readl" | "readq" => {
                check(word(1).is_some(), "words[1]")?;
                let addr = strtou64(word(1));
                check(addr.is_some(), "ret == 0")?;
                let size = match command.as_bytes()[4] {
                    b'b' => 1,
                    b'w' => 2,
                    b'l' => 4,
                    _ => 8,
                };
                let mut data = [0u8; 8];
                if be {
                    b.memory_read(addr.unwrap_or(0), &mut data[8 - size..]);
                } else {
                    b.memory_read(addr.unwrap_or(0), &mut data[..size]);
                }
                let value = if be { u64::from_be_bytes(data) } else { u64::from_le_bytes(data) };
                self.send(&format!("OK 0x{value:016x}\n"));
            }
            "read" => {
                check(word(1).is_some() && word(2).is_some(), "words[1] && words[2]")?;
                let addr = strtou64(word(1));
                check(addr.is_some(), "ret == 0")?;
                let len = strtou64(word(2));
                check(len.is_some(), "ret == 0")?;
                let len = len.unwrap_or(0);
                // We'd send garbage to libqtest if len is 0.
                check(len != 0, "len")?;
                let mut data = vec![0u8; len as usize];
                b.memory_read(addr.unwrap_or(0), &mut data);
                self.send(&format!("OK 0x{}\n", hexdump(&data)));
            }
            "b64read" => {
                check(word(1).is_some() && word(2).is_some(), "words[1] && words[2]")?;
                let addr = strtou64(word(1));
                check(addr.is_some(), "ret == 0")?;
                let len = strtou64(word(2));
                check(len.is_some(), "ret == 0")?;
                let mut data = vec![0u8; len.unwrap_or(0) as usize];
                b.memory_read(addr.unwrap_or(0), &mut data);
                self.send(&format!("OK {}\n", base64::encode(&data)));
            }
            "write" => {
                check(
                    word(1).is_some() && word(2).is_some() && word(3).is_some(),
                    "words[1] && words[2] && words[3]",
                )?;
                let addr = strtou64(word(1));
                check(addr.is_some(), "ret == 0")?;
                let len = strtou64(word(2));
                check(len.is_some(), "ret == 0")?;
                let hex = words[3].as_bytes();
                if hex.len() < 3 {
                    self.send("ERR invalid argument size\n");
                    return Ok(());
                }
                let len = len.unwrap_or(0) as usize;
                let mut data = vec![0u8; len];
                for (i, byte) in data.iter_mut().enumerate() {
                    if i * 2 + 4 <= hex.len() {
                        // An invalid digit is -1 in C, which sets every bit it touches.
                        let hi = (hex2nib(hex[i * 2 + 2]) << 4) as u8;
                        *byte = (i32::from(hi) | hex2nib(hex[i * 2 + 3])) as u8;
                    }
                }
                b.memory_write(addr.unwrap_or(0), &data);
                self.send("OK\n");
            }
            "memset" => {
                check(
                    word(1).is_some() && word(2).is_some() && word(3).is_some(),
                    "words[1] && words[2] && words[3]",
                )?;
                let addr = strtou64(word(1));
                check(addr.is_some(), "ret == 0")?;
                let len = strtou64(word(2));
                check(len.is_some(), "ret == 0")?;
                let pattern = strtoul(word(3));
                check(pattern.is_some(), "ret == 0")?;
                let len = len.unwrap_or(0);
                if len != 0 {
                    let data = vec![pattern.unwrap_or(0) as u8; len as usize];
                    b.memory_write(addr.unwrap_or(0), &data);
                }
                self.send("OK\n");
            }
            "b64write" => {
                check(
                    word(1).is_some() && word(2).is_some() && word(3).is_some(),
                    "words[1] && words[2] && words[3]",
                )?;
                let addr = strtou64(word(1));
                check(addr.is_some(), "ret == 0")?;
                let len = strtou64(word(2));
                check(len.is_some(), "ret == 0")?;
                let text = words[3].as_bytes();
                if text.len() < 3 {
                    self.send("ERR invalid argument size\n");
                    return Ok(());
                }
                let len = len.unwrap_or(0);
                let mut data = base64::decode(text);
                let out_len = data.len() as u64;
                if out_len != len {
                    self.log_send(&format!(
                        "b64write: data length mismatch (told {len}, found {out_len})\n"
                    ));
                    data.truncate(out_len.min(len) as usize);
                }
                b.memory_write(addr.unwrap_or(0), &data);
                self.send("OK\n");
            }
            "endianness" => {
                self.send(if be { "OK big\n" } else { "OK little\n" });
            }
            "clock_step" if b.qtest_enabled() => {
                let old_ns = b.clock_get_ns();
                let ns = if word(1).is_some() {
                    let ns = strtoi64(word(1));
                    check(ns.is_some(), "ret == 0")?;
                    ns.unwrap_or(0)
                } else {
                    let ns = b.clock_deadline_ns_all();
                    if ns < 0 {
                        self.send(
                            "FAIL cannot advance clock to the next deadline because there is no pending deadline\n",
                        );
                        return Ok(());
                    }
                    ns
                };
                let new_ns = b.clock_advance_virtual_time(old_ns.wrapping_add(ns));
                if new_ns > old_ns {
                    self.send(&format!("OK {new_ns}\n"));
                } else {
                    self.send("FAIL could not advance time\n");
                }
            }
            "module_load" => {
                check(word(1).is_some() && word(2).is_some(), "words[1] && words[2]")?;
                match b.module_load(words[1], words[2]) {
                    Ok(true) => self.send("OK\n"),
                    Ok(false) => self.send("FAIL\n"),
                    Err(e) => {
                        error_report(e.message());
                        self.send("FAIL\n");
                    }
                }
            }
            "clock_set" if b.qtest_enabled() => {
                check(word(1).is_some(), "words[1]")?;
                let ns = strtoi64(word(1));
                check(ns.is_some(), "ret == 0")?;
                let ns = ns.unwrap_or(0);
                let new_ns = b.clock_advance_virtual_time(ns);
                let status = if new_ns == ns { "OK" } else { "FAIL" };
                self.send(&format!("{status} {new_ns}\n"));
            }
            "qom-tests" => {
                b.qom_tests();
                self.send("OK\n");
            }
            // spapr_qtest_callback() in hw/ppc/spapr_rtas.c.
            "rtas" if b.has_rtas() => {
                let nargs = strtoul(word(2));
                check(nargs.is_some(), "rc == 0")?;
                let args = strtou64(word(3));
                check(args.is_some(), "rc == 0")?;
                let nret = strtoul(word(4));
                check(nret.is_some(), "rc == 0")?;
                let rets = strtou64(word(5));
                check(rets.is_some(), "rc == 0")?;
                let found = b.rtas_call(
                    word(1).unwrap_or(""),
                    nargs.unwrap_or(0) as u32,
                    args.unwrap_or(0),
                    nret.unwrap_or(0) as u32,
                    rets.unwrap_or(0),
                );
                let res = if found { H_SUCCESS } else { H_PARAMETER };
                self.send(&format!("OK {res}\n"));
            }
            // csr_qtest_callback() in hw/riscv/riscv_hart.c.
            "csr" if b.has_csr() => {
                let cpu = strtou64(word(2));
                check(cpu.is_some(), "rc == 0")?;
                let csr = strtoi(word(3));
                check(csr.is_some(), "rc == 0")?;
                let val = strtou64(word(4));
                check(val.is_some(), "rc == 0")?;
                let val = b.csr_call(
                    word(1).unwrap_or(""),
                    cpu.unwrap_or(0),
                    csr.unwrap_or(0),
                    val.unwrap_or(0),
                );
                check(val.is_some(), "ret == RISCV_EXCP_NONE")?;
                self.send(&format!("OK 0 {:x}\n", val.unwrap_or(0)));
            }
            _ => {
                self.send(&format!("FAIL Unknown command '{command}'\n"));
            }
        }
        Ok(())
    }
}
