// SPDX-License-Identifier: GPL-2.0-or-later

//! `-g` of the user mode emulator: `gdbstub/user.c`, the user mode handlers of
//! `gdbstub/user-target.c` and the protocol of `gdbstub/gdbstub.c`.
//!
//! As in QEMU, the stub talks to the debugger on the thread that stopped, and the other threads
//! run on. A lock stands in for QEMU's single state, so threads that stop at once take turns.

use std::cell::RefCell;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{ErrorKind, Read, Write};
use std::net::TcpListener;
use std::os::fd::OwnedFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::{Mutex, MutexGuard, OnceLock};

use ruvm_linux_user::gdb::{self, Cx, Regs, Stub};

use crate::packet::{
    MAX_PACKET_LENGTH, Param, ThreadId, fromhex, hextomem, host_to_gdb_errno, memtohex, memtox,
    parse_params, strtoul,
};
use crate::signal::{GDB_SIGNAL_TRAP, to_gdb, to_target};

/// What the stub needs of a target: its registers as GDB numbers them, and their description.
pub(crate) struct Target {
    /// `target_name()`.
    pub(crate) name: &'static str,
    /// `gdb_arch_name()`, for `<architecture>`.
    pub(crate) arch: Option<&'static str>,
    /// The XML of each register feature, the core one first.
    pub(crate) features: &'static [(&'static str, &'static str)],
    /// `gdb_num_g_regs`: how many registers `g` and `G` carry.
    pub(crate) g_regs: usize,
    /// The hex digits of `TARGET_ABI_FMT_lx`: 8 for a 32-bit ABI, 16 for a 64-bit one.
    pub(crate) abi_digits: usize,
    /// `gdb_read_register()`: appends register `n` of a thread, returning its size, 0 for no
    /// such register.
    pub(crate) read: fn(&mut Cx<'_, '_>, u32, usize, &mut Vec<u8>) -> usize,
    /// `gdb_write_register()`: sets register `n` from at least 16 bytes, returning its size.
    pub(crate) write: fn(&mut Cx<'_, '_>, u32, usize, &[u8]) -> usize,
    /// `set_pc()`.
    pub(crate) set_pc: fn(&mut [u8], u64),
}

/// The target for `uname -m` of the guest.
fn target_of(machine: &str) -> Option<&'static Target> {
    match machine {
        "x86_64" => Some(&crate::x86_64::TARGET),
        _ => None,
    }
}

/// `TARGET_SIGTRAP`, the same on every target.
const TARGET_SIGTRAP: i32 = 5;
/// `SSTEP_ENABLE`, `SSTEP_NOIRQ` and `SSTEP_NOTIMER`, all of which TCG has.
const SSTEP_FLAGS: u64 = 7;
/// `GDB_NR_SYSCALLS`.
const NR_SYSCALLS: usize = 1024;
/// `BUFSIZ`, the most `vFile:pread` and `vFile:readlink` return.
const BUFSIZ: usize = 8192;
/// What `-g` takes.
const USAGE: &str = "\nUsage: -g {port|path}[,suspend={y|n}]";
/// The version `qGDBServerVersion` reports, the QEMU this stub follows.
const QEMU_VERSION: &str = "11.1.0";

/// `RSState`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rs {
    Idle,
    GetLine,
    GetLineEsc,
    GetLineRle,
    Chksum1,
    Chksum2,
}

/// `enum GDBForkState`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fork {
    Disabled,
    Enabled,
    Active,
    Inactive,
    Enabling,
    Disabling,
    Deactivating,
}

/// `GDB_FORK_ACTIVATE`, `GDB_FORK_ENABLE` and `GDB_FORK_DISABLE`, the messages between the
/// parent and the child after a fork.
const FORK_ACTIVATE: u8 = b'a';
const FORK_ENABLE: u8 = b'e';
const FORK_DISABLE: u8 = b'd';

/// `GDBState` and `GDBUserState`, from `gdb_accept_init()` on.
struct State {
    target: &'static Target,
    /// The connection, `None` once it closed.
    fd: Option<File>,
    socket_path: Option<String>,
    rs: Rs,
    line: Vec<u8>,
    line_sum: u8,
    line_csum: u8,
    /// The process ID of the only process, and whether the debugger is attached to it.
    pid: u32,
    attached: bool,
    /// The threads that run and that registers and memory come from, by thread ID.
    c_cpu: Option<u32>,
    g_cpu: Option<u32>,
    query_cpu: Option<u32>,
    multiprocess: bool,
    allow_stop_reply: bool,
    running: bool,
    signal: i32,
    sstep_flags: u64,
    catch_all_syscalls: bool,
    catch_syscalls_mask: [u64; NR_SYSCALLS / 64],
    fork_events: bool,
    fork_state: Fork,
    fork_sockets: [Option<File>; 2],
    fork_peer_pid: u32,
    fork_peer_tid: u32,
    siginfo: Vec<u8>,
    target_xml: Option<String>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// The target, from when `-g` started the stub.
static TARGET: OnceLock<&'static Target> = OnceLock::new();

thread_local! {
    /// The state, held from `gdbserver_fork_start()` to `gdbserver_fork_end()` so that the
    /// child has it to itself.
    static FORK_GUARD: RefCell<Option<MutexGuard<'static, Option<State>>>> =
        const { RefCell::new(None) };
}

fn state() -> MutexGuard<'static, Option<State>> {
    STATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// `strerror()` of an error, as `perror()` and `error_setg_errno()` print it.
fn strerror(e: &std::io::Error) -> String {
    gdb::strerror(e)
}

/// The stub `-g` starts, for [`gdb::set_stub`].
pub static STUB: User = User;

/// `gdbstub/user.c`.
#[derive(Debug)]
pub struct User;

impl Stub for User {
    fn start(&self, cx: &mut Cx<'_, '_>, args: &str) -> Result<(), String> {
        gdbserver_start(cx, args)
    }

    fn handlesig(
        &self,
        cx: &mut Cx<'_, '_>,
        sig: i32,
        reason: Option<&str>,
        siginfo: Option<&[u8]>,
    ) -> i32 {
        let mut g = state();
        match g.as_mut() {
            Some(s) => s.handlesig(cx, sig, reason, siginfo),
            None => sig,
        }
    }

    fn exit(&self, code: i32) {
        let mut g = state();
        let Some(s) = g.as_mut() else { return };
        if let Some(p) = &s.socket_path {
            let _ = std::fs::remove_file(p);
        }
        if s.fd.is_none() {
            return;
        }
        if s.allow_stop_reply {
            s.put_packet(format!("W{:02x}", code as u8).as_bytes());
            s.allow_stop_reply = false;
        }
    }

    fn signalled(&self, sig: i32) {
        let mut g = state();
        let Some(s) = g.as_mut() else { return };
        if s.fd.is_none() || !s.allow_stop_reply {
            return;
        }
        s.put_packet(format!("X{:02x}", to_gdb(sig)).as_bytes());
        s.allow_stop_reply = false;
    }

    fn syscall_entry(&self, cx: &mut Cx<'_, '_>, num: u64) {
        self.catch_syscall(cx, num, "syscall_entry");
    }

    fn syscall_return(&self, cx: &mut Cx<'_, '_>, num: u64) {
        self.catch_syscall(cx, num, "syscall_return");
    }

    fn fork_start(&self) {
        let mut g = state();
        if let Some(s) = g.as_mut() {
            s.fork_start();
        }
        FORK_GUARD.with(|f| *f.borrow_mut() = Some(g));
    }

    fn fork_end(&self, cx: &mut Cx<'_, '_>, pid: i64) {
        let mut g = FORK_GUARD.with(|f| f.borrow_mut().take()).unwrap_or_else(state);
        if let Some(s) = g.as_mut() {
            s.fork_end(cx, pid);
        }
    }
}

impl User {
    /// `gdb_syscall_entry()` and `gdb_syscall_return()`.
    fn catch_syscall(&self, cx: &mut Cx<'_, '_>, num: u64, what: &str) {
        let mut g = state();
        let Some(s) = g.as_mut() else { return };
        let num = num as i32;
        let catch = s.catch_all_syscalls
            || (0..NR_SYSCALLS as i32).contains(&num)
                && s.catch_syscalls_mask[num as usize / 64] & 1 << (num % 64) != 0;
        if catch {
            let reason = format!("{what}:{:x};", num as u32);
            s.handlesig(cx, TARGET_SIGTRAP, Some(&reason), None);
        }
    }
}

/// `gdbserver_start()`.
fn gdbserver_start(cx: &mut Cx<'_, '_>, args: &str) -> Result<(), String> {
    let mut port_or_path = None;
    let mut suspend = true;
    for arg in args.split(',') {
        let mut tokens = arg.splitn(2, '=');
        if tokens.next() == Some("suspend") {
            match tokens.next() {
                None => return Err(format!("gdbstub: missing \"suspend\" option value{USAGE}")),
                Some("on" | "yes" | "true" | "y") => suspend = true,
                Some("off" | "no" | "false" | "n") => suspend = false,
                Some(_) => return Err("Parameter 'suspend' expects 'on' or 'off'".to_string()),
            }
        } else {
            if port_or_path.is_some() {
                return Err(format!("gdbstub: unknown option \"{arg}\"{USAGE}"));
            }
            port_or_path = Some(arg);
        }
    }
    let Some(port_or_path) = port_or_path.filter(|_| !args.is_empty()) else {
        return Err(format!("gdbstub: port or path not specified{USAGE}"));
    };
    let Some(target) = target_of(cx.machine()) else {
        return Err(format!("gdbstub: -g is not supported for {} by ruvm yet", cx.machine()));
    };
    let _ = TARGET.set(target);
    // g_ascii_strtoull() then a cast to int.
    let port = strtoul(port_or_path.as_bytes(), 10).0.unwrap_or(u64::MAX) as i32;
    let listener = if port > 0 {
        match TcpListener::bind(("0.0.0.0", port as u16)) {
            Ok(l) => Listener::Tcp(l),
            Err(e) => return Err(format!("Failed to bind socket: {}", strerror(&e))),
        }
    } else {
        match open_socket(cx, port_or_path) {
            Some(l) => Listener::Unix(l),
            // QEMU goes on without the stub.
            None => return Ok(()),
        }
    };
    let path = port_or_path.to_string();
    if suspend {
        if !accept(&listener, &path) {
            return Err("gdbstub: failed to accept connection".to_string());
        }
        STUB.handlesig(cx, 0, None, None);
    } else {
        let spawned = std::thread::Builder::new().name("gdb-accept".into()).spawn(move || {
            if accept(&listener, &path) {
                gdb::on_first_cpu(|cx| {
                    let sig = STUB.handlesig(cx, 0, None, None);
                    let tid = cx.tid();
                    cx.kill_thread(tid, sig);
                });
            }
        });
        if let Err(e) = spawned {
            return Err(format!("gdbstub: {}", strerror(&e)));
        }
    }
    Ok(())
}

enum Listener {
    Tcp(TcpListener),
    Unix(UnixListener),
}

/// `gdbserver_open_socket()`: `%d` in `path` is the thread ID. Failures go to standard error.
fn open_socket(cx: &Cx<'_, '_>, path: &str) -> Option<UnixListener> {
    let path = match path.find("%d") {
        Some(i) => format!("{}{}{}", &path[..i], cx.tid(), &path[i + 2..]),
        None => path.to_string(),
    };
    // pstrcpy() into sun_path, less one.
    let mut end = path.len().min(106);
    while !path.is_char_boundary(end) {
        end -= 1;
    }
    let path = &path[..end];
    let _ = std::fs::remove_file(path);
    match UnixListener::bind(path) {
        Ok(l) => Some(l),
        Err(e) => {
            eprintln!("bind socket: {}", strerror(&e));
            None
        }
    }
}

/// `gdbserver_accept()`: waits for the debugger, then `gdb_accept_init()`.
fn accept(listener: &Listener, path: &str) -> bool {
    let fd: OwnedFd = match listener {
        Listener::Tcp(l) => {
            let stream = loop {
                match l.accept() {
                    Ok((s, _)) => break s,
                    Err(e) if e.kind() == ErrorKind::Interrupted => {}
                    Err(e) => {
                        eprintln!("accept: {}", strerror(&e));
                        return false;
                    }
                }
            };
            if let Err(e) = stream.set_nodelay(true) {
                eprintln!("setsockopt: {}", strerror(&e));
                return false;
            }
            stream.into()
        }
        Listener::Unix(l) => {
            let stream: UnixStream = loop {
                match l.accept() {
                    Ok((s, _)) => break s,
                    Err(e) if e.kind() == ErrorKind::Interrupted => {}
                    Err(e) => {
                        eprintln!("accept socket: {}", strerror(&e));
                        return false;
                    }
                }
            };
            stream.into()
        }
    };
    let target = TARGET.get().copied().expect("-g picked the target");
    let mut s = State::new(target, File::from(fd), gdb::first_thread());
    if matches!(listener, Listener::Unix(_)) {
        s.socket_path = Some(path.to_string());
    }
    *state() = Some(s);
    true
}

impl State {
    /// `gdb_accept_init()`.
    fn new(target: &'static Target, fd: File, first: Option<u32>) -> State {
        State {
            target,
            fd: Some(fd),
            socket_path: None,
            rs: Rs::Idle,
            line: Vec::new(),
            line_sum: 0,
            line_csum: 0,
            pid: std::process::id(),
            attached: true,
            c_cpu: first,
            g_cpu: first,
            query_cpu: None,
            multiprocess: false,
            allow_stop_reply: false,
            running: false,
            signal: 0,
            sstep_flags: SSTEP_FLAGS,
            catch_all_syscalls: false,
            catch_syscalls_mask: [0; NR_SYSCALLS / 64],
            fork_events: false,
            fork_state: Fork::Disabled,
            fork_sockets: [None, None],
            fork_peer_pid: 0,
            fork_peer_tid: 0,
            siginfo: Vec::new(),
            target_xml: None,
        }
    }

    // Input and output.

    /// `gdb_get_char()`: `None` when the connection failed.
    fn get_char(&mut self) -> Option<u8> {
        let f = self.fd.as_mut()?;
        let mut b = [0u8];
        loop {
            match f.read(&mut b) {
                Ok(0) => {
                    self.fd = None;
                    return None;
                }
                Ok(_) => return Some(b[0]),
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => {
                    if e.kind() == ErrorKind::ConnectionReset {
                        self.fd = None;
                    }
                    return None;
                }
            }
        }
    }

    /// `gdb_put_buffer()`.
    fn put_buffer(&mut self, buf: &[u8]) {
        if let Some(f) = self.fd.as_mut() {
            let _ = f.write_all(buf);
        }
    }

    /// `gdb_put_packet_binary()`: sends `buf` until the debugger takes it.
    fn put_packet(&mut self, buf: &[u8]) {
        let csum = buf.iter().fold(0u8, |a, &c| a.wrapping_add(c));
        let mut p = Vec::with_capacity(buf.len() + 4);
        p.push(b'$');
        p.extend_from_slice(buf);
        p.push(b'#');
        memtohex(&mut p, &[csum]);
        loop {
            self.put_buffer(&p);
            // gdb_got_immediate_ack(): a failed read counts as one.
            match self.get_char() {
                None | Some(b'+') => break,
                Some(_) => {}
            }
        }
    }

    // Threads.

    /// `gdb_get_cpu_process()` is the only process for every thread, so a thread is attached
    /// when the process is.
    fn attached_threads(&self, cx: &Cx<'_, '_>) -> Vec<u32> {
        if self.attached { cx.threads().into_iter().map(|(t, _)| t).collect() } else { Vec::new() }
    }

    /// `gdb_first_attached_cpu()`.
    fn first_attached(&self, cx: &Cx<'_, '_>) -> Option<u32> {
        self.attached_threads(cx).first().copied()
    }

    /// `gdb_next_attached_cpu()`.
    fn next_attached(&self, cx: &Cx<'_, '_>, tid: u32) -> Option<u32> {
        let all = self.attached_threads(cx);
        let i = all.iter().position(|&t| t == tid)?;
        all.get(i + 1).copied()
    }

    /// `gdb_get_process()`: whether `pid` names the process, 0 meaning it.
    fn is_process(&self, pid: u32) -> bool {
        pid == 0 || pid == self.pid
    }

    /// `gdb_get_cpu()`.
    fn get_cpu(&self, cx: &Cx<'_, '_>, pid: u32, tid: u32) -> Option<u32> {
        if pid == 0 && tid == 0 {
            return self.first_attached(cx);
        }
        if tid == 0 {
            if !self.is_process(pid) || !self.attached {
                return None;
            }
            return cx.threads().first().map(|&(t, _)| t);
        }
        cx.threads().iter().find(|&&(t, _)| t == tid)?;
        if (pid != 0 && pid != self.pid) || !self.attached {
            return None;
        }
        Some(tid)
    }

    /// `gdb_append_thread_id()`.
    fn thread_id(&self, out: &mut String, tid: u32) {
        if self.multiprocess {
            let _ = write!(out, "p{:02x}.{tid:02x}", self.pid);
        } else {
            let _ = write!(out, "{tid:02x}");
        }
    }

    /// The thread registers and memory come from, the stopped one when there is none.
    fn g_tid(&self, cx: &Cx<'_, '_>) -> u32 {
        self.g_cpu.unwrap_or_else(|| cx.tid())
    }

    /// `gdb_handlesig()`.
    fn handlesig(
        &mut self,
        cx: &mut Cx<'_, '_>,
        sig: i32,
        reason: Option<&str>,
        siginfo: Option<&[u8]>,
    ) -> i32 {
        if self.fd.is_none() {
            return sig;
        }
        if let Some(info) = siginfo {
            self.siginfo = info.to_vec();
        }
        let me = cx.tid();
        cx.single_step(me, false);
        if sig != 0 {
            // gdb_set_stop_cpu()
            if self.attached {
                self.c_cpu = Some(me);
                self.g_cpu = Some(me);
            }
            if self.allow_stop_reply {
                let mut b = format!("T{:02x}thread:", to_gdb(sig));
                self.thread_id(&mut b, me);
                b.push(';');
                if let Some(r) = reason {
                    b.push_str(r);
                }
                self.put_packet(b.as_bytes());
                self.allow_stop_reply = false;
            }
        }
        if self.fd.is_none() {
            return sig;
        }
        self.rs = Rs::Idle;
        self.running = false;
        while !self.running {
            let mut buf = [0u8; 256];
            let Some(f) = self.fd.as_mut() else { return 0 };
            let n = loop {
                match f.read(&mut buf) {
                    Err(e) if e.kind() == ErrorKind::Interrupted => {}
                    r => break r.unwrap_or(0),
                }
            };
            if n == 0 {
                // The connection closed.
                self.fd = None;
                return 0;
            }
            for &c in &buf[..n] {
                self.read_byte(cx, c);
            }
        }
        std::mem::take(&mut self.signal)
    }

    /// `gdb_read_byte()`.
    fn read_byte(&mut self, cx: &mut Cx<'_, '_>, ch: u8) {
        self.allow_stop_reply = false;
        // The line buffer holds MAX_PACKET_LENGTH bytes and a NUL.
        let full = MAX_PACKET_LENGTH - 1;
        match self.rs {
            Rs::Idle => {
                if ch == b'$' {
                    self.line.clear();
                    self.line_sum = 0;
                    self.rs = Rs::GetLine;
                }
            }
            Rs::GetLine => {
                if ch == b'}' {
                    self.rs = Rs::GetLineEsc;
                    self.line_sum = self.line_sum.wrapping_add(ch);
                } else if ch == b'*' {
                    self.rs = Rs::GetLineRle;
                    self.line_sum = self.line_sum.wrapping_add(ch);
                } else if ch == b'#' {
                    self.rs = Rs::Chksum1;
                } else if self.line.len() >= full {
                    self.rs = Rs::Idle;
                } else {
                    self.line.push(ch);
                    self.line_sum = self.line_sum.wrapping_add(ch);
                }
            }
            Rs::GetLineEsc => {
                if ch == b'#' {
                    self.rs = Rs::Chksum1;
                } else if self.line.len() >= full {
                    self.rs = Rs::Idle;
                } else {
                    self.line.push(ch ^ 0x20);
                    self.line_sum = self.line_sum.wrapping_add(ch);
                    self.rs = Rs::GetLine;
                }
            }
            Rs::GetLineRle => {
                if !(b' '..=126).contains(&ch) || ch == b'#' || ch == b'$' {
                    self.rs = Rs::GetLine;
                } else {
                    let repeat = usize::from(ch - b' ') + 3;
                    if self.line.len() + repeat >= full {
                        self.rs = Rs::Idle;
                    } else if let Some(&last) = self.line.last() {
                        self.line.resize(self.line.len() + repeat, last);
                        self.line_sum = self.line_sum.wrapping_add(ch);
                        self.rs = Rs::GetLine;
                    } else {
                        self.rs = Rs::GetLine;
                    }
                }
            }
            Rs::Chksum1 => {
                if !ch.is_ascii_hexdigit() {
                    self.rs = Rs::GetLine;
                    return;
                }
                self.line_csum = fromhex(ch) << 4;
                self.rs = Rs::Chksum2;
            }
            Rs::Chksum2 => {
                if !ch.is_ascii_hexdigit() {
                    self.rs = Rs::GetLine;
                    return;
                }
                self.line_csum |= fromhex(ch);
                if self.line_csum != self.line_sum {
                    self.put_buffer(b"-");
                    self.rs = Rs::Idle;
                } else {
                    self.put_buffer(b"+");
                    let mut line = std::mem::take(&mut self.line);
                    // The line is a C string.
                    if let Some(i) = line.iter().position(|&c| c == 0) {
                        line.truncate(i);
                    }
                    self.handle_packet(cx, &line);
                    self.line = line;
                    self.rs = Rs::Idle;
                }
            }
        }
    }

    /// `gdb_handle_packet()`.
    fn handle_packet(&mut self, cx: &mut Cx<'_, '_>, line: &[u8]) {
        // The command, its schema and whether it may answer with a stop reply.
        let (schema, allow): (&str, bool) = match line.first() {
            Some(b'!') => return self.put_packet(b"OK"),
            Some(b'?') => ("", true),
            Some(b'c' | b's') => ("L0", true),
            Some(b'C') => ("l0", true),
            Some(b'b') => ("o0", true),
            Some(b'v' | b'G' | b'q' | b'Q') => ("s0", false),
            Some(b'k') => {
                eprintln!("{}: QEMU: Terminated via GDBstub", gdb::prog());
                self.exit_now(0);
            }
            Some(b'D') => ("?.l0", false),
            Some(b'F') => ("L,L,o0", false),
            Some(b'g') => ("", false),
            Some(b'm') => ("L,L0", false),
            Some(b'M') => ("L,L:s0", false),
            Some(b'p') => ("L0", false),
            Some(b'P') => ("L?s0", false),
            Some(b'Z' | b'z') => ("l?L?L0", false),
            Some(b'H') => ("o.t0", false),
            Some(b'T') => ("t0", false),
            _ => return self.put_packet(b""),
        };
        let Some(params) = parse_params(&line[1..], schema) else {
            return self.put_packet(b"");
        };
        self.allow_stop_reply = allow;
        match line[0] {
            b'?' => self.handle_target_halt(cx),
            b'c' => self.handle_continue(cx, &params),
            b'C' => self.handle_cont_with_sig(&params),
            b'v' => self.handle_v_commands(cx, &params),
            b'D' => self.handle_detach(cx, &params),
            b's' => self.handle_step(cx, &params),
            // There is no reverse execution.
            b'b' => self.put_packet(b"E22"),
            b'F' => self.handle_file_io(&params),
            b'g' => self.handle_read_all_regs(cx),
            b'G' => self.handle_write_all_regs(cx, &params),
            b'm' => self.handle_read_mem(cx, &params),
            b'M' => self.handle_write_mem(cx, &params),
            b'p' => self.handle_get_reg(cx, &params),
            b'P' => self.handle_set_reg(cx, &params),
            b'Z' => self.handle_insert_bp(cx, &params),
            b'z' => self.handle_remove_bp(cx, &params),
            b'H' => self.handle_set_thread(cx, &params),
            b'T' => self.handle_thread_alive(cx, &params),
            b'q' => self.handle_gen_query(cx, &params),
            _ => self.handle_gen_set(&params),
        }
    }

    /// `gdb_exit()` then `exit()`, for `k` and `vKill`.
    fn exit_now(&mut self, code: i32) -> ! {
        if let Some(p) = &self.socket_path {
            let _ = std::fs::remove_file(p);
        }
        if self.fd.is_some() && self.allow_stop_reply {
            self.put_packet(format!("W{:02x}", code as u8).as_bytes());
        }
        std::process::exit(code)
    }

    /// `?`: the stop reply, and no breakpoints, as the debugger is connecting.
    fn handle_target_halt(&mut self, cx: &mut Cx<'_, '_>) {
        let c = self.c_cpu.unwrap_or_else(|| cx.tid());
        if self.allow_stop_reply {
            let mut b = format!("T{GDB_SIGNAL_TRAP:02x}thread:");
            self.thread_id(&mut b, c);
            b.push(';');
            self.allow_stop_reply = false;
            self.put_packet(b.as_bytes());
        }
        cx.breakpoint_remove_all(c);
    }

    /// `gdb_set_cpu_pc()` of the thread that runs.
    fn set_cpu_pc(&mut self, cx: &mut Cx<'_, '_>, pc: u64) {
        let c = self.c_cpu.unwrap_or_else(|| cx.tid());
        if let Some(Regs::Live(cpu)) = cx.regs(c) {
            (self.target.set_pc)(cpu.env, pc);
        }
    }

    fn handle_continue(&mut self, cx: &mut Cx<'_, '_>, params: &[Param<'_>]) {
        if let Some(p) = params.first() {
            self.set_cpu_pc(cx, p.num());
        }
        self.signal = 0;
        self.running = true;
    }

    fn handle_cont_with_sig(&mut self, params: &[Param<'_>]) {
        let mut signal = 0;
        if let Some(p) = params.first() {
            signal = to_target(p.num());
            if signal == -1 {
                signal = 0;
            }
        }
        self.signal = signal;
        self.running = true;
    }

    fn handle_step(&mut self, cx: &mut Cx<'_, '_>, params: &[Param<'_>]) {
        if let Some(p) = params.first() {
            self.set_cpu_pc(cx, p.num());
        }
        let c = self.c_cpu.unwrap_or_else(|| cx.tid());
        cx.single_step(c, self.sstep_flags != 0);
        self.running = true;
    }

    fn handle_file_io(&mut self, params: &[Param<'_>]) {
        if params.len() >= 3 && params[2].op() == b'C' {
            self.put_packet(b"T02");
            return;
        }
        self.running = true;
    }

    fn handle_detach(&mut self, cx: &mut Cx<'_, '_>, params: &[Param<'_>]) {
        let mut pid = 1;
        if self.multiprocess {
            let Some(p) = params.first() else { return self.put_packet(b"E22") };
            pid = p.num() as u32;
        }
        // gdb_handle_detach_user()
        if self.fork_state == Fork::Active {
            let enable = pid == self.fork_peer_pid;
            if enable || pid == self.pid {
                self.fork_state = if enable { Fork::Enabling } else { Fork::Disabling };
                return self.put_packet(b"OK");
            }
        }
        if self.is_process(pid) {
            for t in cx.threads() {
                cx.breakpoint_remove_all(t.0);
            }
            self.attached = false;
            if !self.attached {
                self.c_cpu = self.first_attached(cx);
                self.g_cpu = self.first_attached(cx);
            }
        }
        if self.c_cpu.is_none() {
            // gdb_disable_syscalls(), then on with no debugger.
            self.running = true;
        }
        self.put_packet(b"OK");
    }

    fn handle_thread_alive(&mut self, cx: &mut Cx<'_, '_>, params: &[Param<'_>]) {
        let Some((pid, tid)) = params.first().and_then(|p| ids(p.thread())) else {
            return self.put_packet(b"E22");
        };
        if self.get_cpu(cx, pid, tid).is_none() {
            return self.put_packet(b"E22");
        }
        self.put_packet(b"OK");
    }

    fn handle_set_thread(&mut self, cx: &mut Cx<'_, '_>, params: &[Param<'_>]) {
        if params.len() != 2 {
            return self.put_packet(b"E22");
        }
        let (pid, tid) = match params[1].thread() {
            ThreadId::Err => return self.put_packet(b"E22"),
            ThreadId::One { pid, tid } => (pid, tid),
            _ => return self.put_packet(b"OK"),
        };
        // gdb_handle_set_thread_user()
        if self.fork_state == Fork::Active && pid == self.fork_peer_pid && tid == self.fork_peer_tid
        {
            self.fork_state = Fork::Deactivating;
            return self.put_packet(b"OK");
        }
        let Some(cpu) = self.get_cpu(cx, pid, tid) else { return self.put_packet(b"E22") };
        match params[0].op() {
            b'c' => self.c_cpu = Some(cpu),
            b'g' => self.g_cpu = Some(cpu),
            _ => return self.put_packet(b"E22"),
        }
        self.put_packet(b"OK");
    }

    fn handle_insert_bp(&mut self, cx: &mut Cx<'_, '_>, params: &[Param<'_>]) {
        if params.len() != 3 {
            return self.put_packet(b"E22");
        }
        match params[0].num() {
            // GDB_BREAKPOINT_SW and GDB_BREAKPOINT_HW; user mode has no watchpoints.
            0 | 1 => {
                cx.breakpoint_insert(params[1].num());
                self.put_packet(b"OK");
            }
            _ => self.put_packet(b""),
        }
    }

    fn handle_remove_bp(&mut self, cx: &mut Cx<'_, '_>, params: &[Param<'_>]) {
        if params.len() != 3 {
            return self.put_packet(b"E22");
        }
        match params[0].num() {
            0 | 1 => {
                let found = cx.breakpoint_remove(params[1].num());
                self.put_packet(if found { b"OK" } else { b"E22" });
            }
            _ => self.put_packet(b""),
        }
    }

    /// `gdb_write_register()` from `mem`, padded so that a short value reads as zeros.
    fn write_register(&mut self, cx: &mut Cx<'_, '_>, mem: &[u8], n: usize) -> usize {
        let mut m = mem.to_vec();
        m.resize(mem.len().max(16), 0);
        let tid = self.g_tid(cx);
        (self.target.write)(cx, tid, n, &m)
    }

    fn handle_set_reg(&mut self, cx: &mut Cx<'_, '_>, params: &[Param<'_>]) {
        if params.len() != 2 {
            return self.put_packet(b"E22");
        }
        let data = params[1].str();
        let mut mem = Vec::new();
        hextomem(&mut mem, data, data.len() / 2);
        self.write_register(cx, &mem, params[0].num() as usize);
        self.put_packet(b"OK");
    }

    fn handle_get_reg(&mut self, cx: &mut Cx<'_, '_>, params: &[Param<'_>]) {
        let Some(p) = params.first() else { return self.put_packet(b"E14") };
        let tid = self.g_tid(cx);
        let mut mem = Vec::new();
        let n = usize::try_from(p.num()).unwrap_or(usize::MAX);
        if (self.target.read)(cx, tid, n, &mut mem) == 0 {
            return self.put_packet(b"E14");
        }
        let mut out = Vec::new();
        memtohex(&mut out, &mem);
        self.put_packet(&out);
    }

    fn handle_write_mem(&mut self, cx: &mut Cx<'_, '_>, params: &[Param<'_>]) {
        if params.len() != 3 {
            return self.put_packet(b"E22");
        }
        let data = params[2].str();
        let len = params[1].num();
        if len > (data.len() / 2) as u64 {
            return self.put_packet(b"E22");
        }
        let mut mem = Vec::new();
        hextomem(&mut mem, data, len as usize);
        if !cx.write_memory(params[0].num(), &mem) {
            return self.put_packet(b"E14");
        }
        self.put_packet(b"OK");
    }

    fn handle_read_mem(&mut self, cx: &mut Cx<'_, '_>, params: &[Param<'_>]) {
        if params.len() != 2 {
            return self.put_packet(b"E22");
        }
        let len = params[1].num();
        if len > (MAX_PACKET_LENGTH / 2) as u64 {
            return self.put_packet(b"E22");
        }
        let mut mem = vec![0u8; len as usize];
        if !cx.read_memory(params[0].num(), &mut mem) {
            return self.put_packet(b"E14");
        }
        let mut out = Vec::new();
        memtohex(&mut out, &mem);
        self.put_packet(&out);
    }

    fn handle_write_all_regs(&mut self, cx: &mut Cx<'_, '_>, params: &[Param<'_>]) {
        let Some(p) = params.first() else { return };
        let data = p.str();
        let mut len = data.len() / 2;
        let mut mem = Vec::new();
        hextomem(&mut mem, data, len);
        let mut off = 0;
        let mut reg = 0;
        // len is a size_t in QEMU: a last register larger than what is left wraps it.
        while reg < self.target.g_regs && len > 0 {
            let size = self.write_register(cx, mem.get(off..).unwrap_or(&[]), reg);
            len = len.wrapping_sub(size);
            off += size;
            reg += 1;
        }
        self.put_packet(b"OK");
    }

    fn handle_read_all_regs(&mut self, cx: &mut Cx<'_, '_>) {
        let tid = self.g_tid(cx);
        let mut mem = Vec::new();
        for reg in 0..self.target.g_regs {
            (self.target.read)(cx, tid, reg, &mut mem);
        }
        let mut out = Vec::new();
        memtohex(&mut out, &mem);
        self.put_packet(&out);
    }

    /// `process_string_cmd()` over one table: the handler of the first entry `data` matches,
    /// with its parameters. `None` when none matches or the parameters do not parse.
    fn find_cmd<'d>(
        &mut self,
        data: &'d [u8],
        table: &[(&'static str, bool, &'static str, bool)],
    ) -> Option<(&'static str, Vec<Param<'d>>)> {
        for &(cmd, startswith, schema, allow) in table {
            let hit =
                if startswith { data.starts_with(cmd.as_bytes()) } else { data == cmd.as_bytes() };
            if !hit {
                continue;
            }
            let params = if schema.is_empty() {
                Vec::new()
            } else {
                parse_params(&data[cmd.len()..], schema)?
            };
            self.allow_stop_reply = allow;
            return Some((cmd, params));
        }
        None
    }

    fn handle_v_commands(&mut self, cx: &mut Cx<'_, '_>, params: &[Param<'_>]) {
        let Some(p) = params.first() else { return };
        const TABLE: &[(&str, bool, &str, bool)] = &[
            ("Cont?", true, "", false),
            ("Cont", true, "s0", true),
            ("Attach;", true, "l0", true),
            ("Kill;", true, "", false),
            ("File:open:", true, "s,L,L0", false),
            ("File:close:", true, "l0", false),
            ("File:pread:", true, "l,L,L0", false),
            ("File:readlink:", true, "s0", false),
        ];
        let Some((cmd, params)) = self.find_cmd(p.str(), TABLE) else {
            return self.put_packet(b"");
        };
        let num = |i: usize| params.get(i).map_or(0, |p| p.num());
        let s = |i: usize| params.get(i).map_or(&[][..], |p| p.str());
        match cmd {
            "Cont?" => self.put_packet(b"vCont;c;C;s;S"),
            "Cont" => {
                if params.is_empty() {
                    return;
                }
                match self.handle_vcont(cx, s(0)) {
                    Ok(()) => {}
                    Err(VcontError::NotSup) => self.put_packet(b""),
                    Err(VcontError::Inval) => self.put_packet(b"E22"),
                }
            }
            "Attach;" => self.handle_v_attach(cx, num(0)),
            "Kill;" => {
                self.put_packet(b"OK");
                eprintln!("{}: QEMU: Terminated via GDBstub", gdb::prog());
                self.exit_now(0);
            }
            "File:open:" => {
                let name = filename(s(0));
                let fd = cx.open(&name, num(1), num(2));
                self.reply_fd(fd, |fd| format!("F{:x}", fd as i32));
            }
            "File:close:" => {
                let r = cx.close(num(0) as i32);
                self.reply_fd(r, |_| "F00".to_string());
            }
            "File:pread:" => {
                let count = (num(1) as usize).min(BUFSIZ);
                match cx.pread(num(0) as i32, count, num(2)) {
                    Ok(b) => self.reply_with_data(&b),
                    Err(e) => self.reply_fd(e, |_| String::new()),
                }
            }
            _ => {
                let name = filename(s(0));
                match cx.readlink(&name, BUFSIZ) {
                    Ok(b) => self.reply_with_data(&b),
                    Err(e) => self.reply_fd(e, |_| String::new()),
                }
            }
        }
    }

    /// The reply of a host I/O call: `ok` of its result, or `F-1` and the error.
    fn reply_fd(&mut self, r: i64, ok: impl FnOnce(i64) -> String) {
        let b = if r < 0 { format!("F-1,{:x}", host_to_gdb_errno(-r as i32)) } else { ok(r) };
        self.put_packet(b.as_bytes());
    }

    /// `hostio_reply_with_data()`.
    fn reply_with_data(&mut self, buf: &[u8]) {
        let mut out = format!("F{:x};", buf.len()).into_bytes();
        memtox(&mut out, buf);
        self.put_packet(&out);
    }

    /// `gdb_handle_vcont()`.
    fn handle_vcont(&mut self, cx: &mut Cx<'_, '_>, p: &[u8]) -> Result<(), VcontError> {
        let threads = cx.threads();
        // The action of each thread, by its place in the list; 1 for none yet.
        let mut newstates = vec![1u8; threads.len()];
        let index = |tid: u32| threads.iter().position(|&(t, _)| t == tid);
        let mut signal = 0;
        let mut target_count = 0;
        let mut last_target = None;
        let mut i = 0;
        while i < p.len() {
            if p[i] != b';' {
                return Err(VcontError::NotSup);
            }
            i += 1;
            let mut action = p.get(i).copied().unwrap_or(0);
            i += 1;
            if action == b'C' || action == b'S' {
                action = action.to_ascii_lowercase();
                let (r, n) = strtoul(p.get(i..).unwrap_or(&[]), 16);
                signal = to_target(r.map_err(|_| VcontError::Inval)?);
                i += n;
            } else if action != b'c' && action != b's' {
                return Err(VcontError::NotSup);
            }
            let kind = match p.get(i) {
                None | Some(b';') => ThreadId::AllProcesses,
                Some(b':') => {
                    i += 1;
                    let (t, n) = crate::packet::read_thread_id(&p[i..]);
                    i += n;
                    t
                }
                Some(_) => return Err(VcontError::NotSup),
            };
            let mut mark = |tid: u32| {
                if let Some(k) = index(tid) {
                    if newstates[k] == 1 {
                        newstates[k] = action;
                        target_count += 1;
                        last_target = Some(tid);
                    }
                }
            };
            match kind {
                ThreadId::Err => return Err(VcontError::Inval),
                ThreadId::AllProcesses => {
                    for t in self.attached_threads(cx) {
                        mark(t);
                    }
                }
                ThreadId::AllThreads { pid } => {
                    if !self.is_process(pid) || !self.attached {
                        return Err(VcontError::Inval);
                    }
                    for &(t, _) in &threads {
                        mark(t);
                    }
                }
                ThreadId::One { pid, tid } => {
                    let Some(cpu) = self.get_cpu(cx, pid, tid) else {
                        return Err(VcontError::Inval);
                    };
                    mark(cpu);
                }
            }
        }
        if target_count > 0 {
            self.c_cpu = last_target;
        }
        self.signal = signal;
        // gdb_continue_partial()
        for (k, &(t, _)) in threads.iter().enumerate() {
            if newstates[k] == b's' {
                cx.single_step(t, self.sstep_flags != 0);
            }
        }
        self.running = true;
        Ok(())
    }

    fn handle_v_attach(&mut self, cx: &mut Cx<'_, '_>, pid: u64) {
        let pid = pid as u32;
        let cpu = if self.is_process(pid) { cx.threads().first().map(|&(t, _)| t) } else { None };
        let Some(cpu) = cpu else { return self.put_packet(b"E22") };
        self.attached = true;
        self.g_cpu = Some(cpu);
        self.c_cpu = Some(cpu);
        if self.allow_stop_reply {
            let mut b = format!("T{GDB_SIGNAL_TRAP:02x}thread:");
            self.thread_id(&mut b, cpu);
            b.push(';');
            self.put_packet(b.as_bytes());
            self.allow_stop_reply = false;
        }
    }

    /// The tables both `q` and `Q` look in first.
    const COMMON: &'static [(&'static str, bool, &'static str, bool)] = &[
        ("qemu.sstepbits", false, "", false),
        ("qemu.sstep", false, "", false),
        ("qemu.sstep=", true, "l0", false),
    ];

    /// The common table: whether it took the command.
    fn handle_common(&mut self, data: &[u8]) -> bool {
        let Some((cmd, params)) = self.find_cmd(data, Self::COMMON) else { return false };
        match cmd {
            "qemu.sstepbits" => self.put_packet(b"ENABLE=1,NOIRQ=2,NOTIMER=4"),
            "qemu.sstep" => self.put_packet(format!("0x{:x}", self.sstep_flags).as_bytes()),
            _ => self.set_sstep(&params),
        }
        true
    }

    /// `handle_set_qemu_sstep()`.
    fn set_sstep(&mut self, params: &[Param<'_>]) {
        let Some(p) = params.first() else { return };
        let flags = p.num() as u32 as i32 as u64;
        if flags & !SSTEP_FLAGS != 0 {
            return self.put_packet(b"E22");
        }
        self.sstep_flags = flags;
        self.put_packet(b"OK");
    }

    fn handle_gen_query(&mut self, cx: &mut Cx<'_, '_>, params: &[Param<'_>]) {
        let Some(p) = params.first() else { return };
        let data = p.str();
        if self.handle_common(data) {
            return;
        }
        const TABLE: &[(&str, bool, &str, bool)] = &[
            ("C", false, "", false),
            ("sThreadInfo", false, "", false),
            ("GDBServerVersion", false, "", false),
            ("fThreadInfo", false, "", false),
            ("ThreadExtraInfo,", true, "t0", false),
            ("Offsets", false, "", false),
            ("Supported:", true, "s0", false),
            ("Supported", false, "s0", false),
            ("Xfer:features:read:", true, "s:l,l0", false),
            ("Xfer:auxv:read::", true, "l,l0", false),
            ("Xfer:siginfo:read::", true, "l,l0", false),
            ("Xfer:exec-file:read:", true, "l:l,l0", false),
            ("Attached:", true, "", false),
            ("Attached", false, "", false),
            ("qemu.Supported", false, "", false),
        ];
        let Some((cmd, params)) = self.find_cmd(data, TABLE) else {
            return self.put_packet(b"");
        };
        let num = |i: usize| params.get(i).map_or(0, |p| p.num());
        match cmd {
            "C" => {
                let mut b = "QC".to_string();
                if let Some(&(t, _)) = cx.threads().first() {
                    self.thread_id(&mut b, t);
                }
                self.put_packet(b.as_bytes());
            }
            "sThreadInfo" => self.query_threads(cx),
            "GDBServerVersion" => {
                let b = format!("name:qemu-{};version:{QEMU_VERSION};", self.target.name);
                self.put_packet(b.as_bytes());
            }
            "fThreadInfo" => {
                self.query_cpu = self.first_attached(cx);
                self.query_threads(cx);
            }
            "ThreadExtraInfo," => self.query_thread_extra(cx, &params),
            "Offsets" => {
                let o = cx.load_bias();
                self.put_packet(
                    format!("Text={o:0w$x};Data={o:0w$x};Bss={o:0w$x}", w = self.target.abi_digits)
                        .as_bytes(),
                );
            }
            "Supported:" | "Supported" => self.query_supported(&params),
            "Xfer:features:read:" => self.query_xfer_features(&params),
            "Xfer:auxv:read::" => self.query_xfer_auxv(cx, &params),
            "Xfer:siginfo:read::" => {
                let (off, len) = (num(0), num(1));
                let Some(end) = off.checked_add(len).filter(|&e| e <= self.siginfo.len() as u64)
                else {
                    return self.put_packet(b"E01");
                };
                let mut out = b"l".to_vec();
                memtox(&mut out, &self.siginfo[off as usize..end as usize]);
                self.put_packet(&out);
            }
            "Xfer:exec-file:read:" => {
                let pid = num(0) as u32;
                let name = cx.exec_file().as_bytes().to_vec();
                if !self.is_process(pid) || cx.threads().is_empty() || name.is_empty() {
                    return self.put_packet(b"E00");
                }
                let off = num(1) as u32 as usize;
                let mut len = num(2) as u32 as usize;
                if off > name.len() {
                    return self.put_packet(b"E00");
                }
                if (off as u32).wrapping_add(len as u32) as usize > name.len() {
                    len = name.len() - off;
                }
                let mut out = b"l".to_vec();
                out.extend_from_slice(&name[off..(off + len).min(name.len())]);
                self.put_packet(&out);
            }
            "Attached:" | "Attached" => self.put_packet(b"0"),
            _ => self.put_packet(b"sstepbits;sstep"),
        }
    }

    /// `handle_query_threads()`.
    fn query_threads(&mut self, cx: &Cx<'_, '_>) {
        let Some(q) = self.query_cpu else { return self.put_packet(b"l") };
        let mut b = "m".to_string();
        self.thread_id(&mut b, q);
        self.put_packet(b.as_bytes());
        self.query_cpu = self.next_attached(cx, q);
    }

    /// `handle_query_thread_extra()`.
    fn query_thread_extra(&mut self, cx: &Cx<'_, '_>, params: &[Param<'_>]) {
        let Some((pid, tid)) = params.first().and_then(|p| ids(p.thread())) else {
            return self.put_packet(b"E22");
        };
        let Some(cpu) = self.get_cpu(cx, pid, tid) else { return };
        let index = cx.threads().iter().find(|&&(t, _)| t == cpu).map_or(0, |&(_, i)| i);
        let mut out = Vec::new();
        memtohex(&mut out, format!("CPU#{index} [running]").as_bytes());
        self.put_packet(&out);
    }

    /// `handle_query_supported()` and `gdb_handle_query_supported_user()`.
    fn query_supported(&mut self, params: &[Param<'_>]) {
        let mut b = format!("PacketSize={MAX_PACKET_LENGTH:x}");
        b.push_str(";qXfer:features:read+");
        b.push_str(";qXfer:auxv:read+;QCatchSyscalls+;qXfer:siginfo:read+");
        b.push_str(";qXfer:exec-file:read+");
        if let Some(p) = params.first() {
            let s = p.str();
            let has = |w: &[u8]| s.windows(w.len()).any(|x| x == w);
            if has(b"multiprocess+") {
                self.multiprocess = true;
            }
            if has(b"fork-events+") {
                self.fork_events = true;
            }
            b.push_str(";fork-events+");
        }
        b.push_str(";vContSupported+;multiprocess+");
        self.put_packet(b.as_bytes());
    }

    /// `get_feature_xml()`: the annex runs to the first `:` and names a file by its prefix.
    fn feature_xml(&mut self, p: &[u8]) -> Option<String> {
        let len = p.iter().position(|&c| c == b':')?;
        let annex = &p[..len];
        if b"target.xml".starts_with(annex) {
            let t = self.target;
            let xml = self.target_xml.get_or_insert_with(|| {
                let mut x = String::from(
                    "<?xml version=\"1.0\"?><!DOCTYPE target SYSTEM \"gdb-target.dtd\"><target>",
                );
                if let Some(a) = t.arch {
                    let _ = write!(x, "<architecture>{a}</architecture>");
                }
                for (name, _) in t.features {
                    let _ = write!(x, "<xi:include href=\"{name}\"/>");
                }
                x.push_str("</target>");
                x
            });
            return Some(xml.clone());
        }
        let found =
            self.target.features.iter().find(|(name, _)| name.as_bytes().starts_with(annex));
        found.map(|(_, xml)| (*xml).to_string())
    }

    fn query_xfer_features(&mut self, params: &[Param<'_>]) {
        if params.len() < 3 {
            return self.put_packet(b"E22");
        }
        let Some(xml) = self.feature_xml(params[0].str()) else {
            return self.put_packet(b"E00");
        };
        let (addr, len) = (params[1].num(), params[2].num());
        let total = xml.len() as u64;
        if addr > total {
            return self.put_packet(b"E00");
        }
        let len = len.min(((MAX_PACKET_LENGTH - 5) / 2) as u64);
        let xml = xml.as_bytes();
        let mut out = Vec::new();
        if len < total - addr {
            out.push(b'm');
            memtox(&mut out, &xml[addr as usize..(addr + len) as usize]);
        } else {
            out.push(b'l');
            memtox(&mut out, &xml[addr as usize..]);
        }
        self.put_packet(&out);
    }

    fn query_xfer_auxv(&mut self, cx: &mut Cx<'_, '_>, params: &[Param<'_>]) {
        if params.len() < 2 {
            return self.put_packet(b"E22");
        }
        let (offset, mut len) = (params[0].num(), params[1].num());
        let (saved_auxv, auxv_len) = cx.auxv();
        if offset >= auxv_len {
            return self.put_packet(b"E00");
        }
        len = len.min(((MAX_PACKET_LENGTH - 5) / 2) as u64);
        let mut out = Vec::new();
        if len < auxv_len - offset {
            out.push(b'm');
        } else {
            out.push(b'l');
            len = auxv_len - offset;
        }
        let mut mem = vec![0u8; len as usize];
        if !cx.read_memory(saved_auxv.wrapping_add(offset), &mut mem) {
            return self.put_packet(b"E14");
        }
        memtox(&mut out, &mem);
        self.put_packet(&out);
    }

    fn handle_gen_set(&mut self, params: &[Param<'_>]) {
        let Some(p) = params.first() else { return };
        let data = p.str();
        if self.handle_common(data) {
            return;
        }
        const TABLE: &[(&str, bool, &str, bool)] =
            &[("qemu.sstep:", true, "l0", false), ("CatchSyscalls:", true, "s0", false)];
        let Some((cmd, params)) = self.find_cmd(data, TABLE) else {
            return self.put_packet(b"");
        };
        match cmd {
            "qemu.sstep:" => self.set_sstep(&params),
            _ => {
                let s = params.first().map_or(&[][..], |p| p.str());
                self.set_catch_syscalls(s);
            }
        }
    }

    /// `gdb_handle_set_catch_syscalls()`.
    fn set_catch_syscalls(&mut self, param: &[u8]) {
        if param == b"0" {
            self.catch_all_syscalls = false;
            self.catch_syscalls_mask = [0; NR_SYSCALLS / 64];
            return self.put_packet(b"OK");
        }
        if param == b"1" {
            self.catch_all_syscalls = true;
            return self.put_packet(b"OK");
        }
        if param.starts_with(b"1;") {
            let mut all = false;
            let mut mask = [0u64; NR_SYSCALLS / 64];
            let mut i = 2;
            loop {
                // qemu_strtoui(), then a ';' or the end.
                let (r, n) = strtoul(&param[i..], 16);
                let num = match r {
                    Ok(v) if v <= u64::from(u32::MAX) || v >= u64::MAX - u64::from(u32::MAX) => {
                        v as u32
                    }
                    _ => return self.put_packet(b"E00"),
                };
                i += n;
                if i < param.len() && param[i] != b';' {
                    return self.put_packet(b"E00");
                }
                if num as usize >= NR_SYSCALLS {
                    all = true;
                } else {
                    mask[num as usize / 64] |= 1 << (num % 64);
                }
                if i >= param.len() {
                    break;
                }
                i += 1;
            }
            self.catch_all_syscalls = all;
            if !all {
                self.catch_syscalls_mask = mask;
            }
            return self.put_packet(b"OK");
        }
        self.put_packet(b"E00");
    }

    // fork()

    /// `gdbserver_fork_start()`.
    fn fork_start(&mut self) {
        if self.fd.is_none() {
            return;
        }
        let pair = if self.fork_events { UnixStream::pair().ok() } else { None };
        let Some((a, b)) = pair else {
            self.fork_state = Fork::Disabled;
            return;
        };
        self.fork_sockets =
            [Some(File::from(OwnedFd::from(a))), Some(File::from(OwnedFd::from(b)))];
        self.fork_state = Fork::Inactive;
        self.fork_peer_pid = std::process::id();
        self.fork_peer_tid = gdb::gettid();
    }

    /// `disable_gdbstub()`.
    fn disable(&mut self, cx: &mut Cx<'_, '_>) {
        self.fd = None;
        for (t, _) in cx.threads() {
            cx.breakpoint_remove_all(t);
            cx.single_step(t, false);
        }
    }

    /// `gdbserver_fork_end()`.
    fn fork_end(&mut self, cx: &mut Cx<'_, '_>, pid: i64) {
        if self.fd.is_none() {
            return;
        }
        if pid == -1 {
            self.fork_sockets = [None, None];
            return;
        }
        if self.fork_state == Fork::Disabled {
            if pid == 0 {
                self.disable(cx);
            }
            return;
        }
        let [s0, s1] = std::mem::take(&mut self.fork_sockets);
        let mut fd = if pid == 0 {
            self.pid = std::process::id();
            // The thread that forked is the child's only one, under a new ID.
            let me = cx.tid();
            for c in [&mut self.c_cpu, &mut self.g_cpu, &mut self.query_cpu] {
                if c.is_some() {
                    *c = Some(me);
                }
            }
            s1
        } else {
            self.fork_state = Fork::Active;
            self.fork_peer_pid = pid as u32;
            self.fork_peer_tid = pid as u32;
            if !self.allow_stop_reply {
                return;
            }
            let b = format!(
                "T{GDB_SIGNAL_TRAP:02x}fork:p{pid:02x}.{pid:02x};thread:p{:02x}.{:02x};",
                std::process::id(),
                gdb::gettid()
            );
            self.put_packet(b.as_bytes());
            s0
        };
        let Some(sock) = fd.as_mut() else { return };
        self.rs = Rs::Idle;
        self.allow_stop_reply = false;
        self.running = false;
        let failed = loop {
            let mut b = [0u8];
            match self.fork_state {
                Fork::Enabled if self.running => return,
                Fork::Enabled | Fork::Active => {
                    let Some(f) = self.fd.as_mut() else { break true };
                    if f.read(&mut b).ok() != Some(1) {
                        break true;
                    }
                    self.read_byte(cx, b[0]);
                }
                Fork::Deactivating => {
                    if sock.write(&[FORK_ACTIVATE]).ok() != Some(1) {
                        break true;
                    }
                    self.fork_state = Fork::Inactive;
                }
                Fork::Inactive => {
                    if sock.read(&mut b).ok() != Some(1) {
                        break true;
                    }
                    self.fork_state = match b[0] {
                        FORK_ACTIVATE => Fork::Active,
                        FORK_ENABLE => Fork::Enabled,
                        _ => Fork::Disabled,
                    };
                }
                Fork::Enabling => {
                    if sock.write(&[FORK_DISABLE]).ok() != Some(1) {
                        break true;
                    }
                    self.fork_state = Fork::Enabled;
                }
                Fork::Disabling => {
                    if sock.write(&[FORK_ENABLE]).ok() != Some(1) {
                        break true;
                    }
                    self.fork_state = Fork::Disabled;
                }
                Fork::Disabled => break false,
            }
        };
        drop(fd);
        if !failed || pid == 0 {
            self.disable(cx);
        }
    }
}

/// What can go wrong with `vCont`: `-ENOTSUP` answers nothing, the rest `E22`.
enum VcontError {
    NotSup,
    Inval,
}

/// The process and thread a thread ID names, -1 standing for all, which no thread matches.
fn ids(t: ThreadId) -> Option<(u32, u32)> {
    match t {
        ThreadId::Err => None,
        ThreadId::One { pid, tid } => Some((pid, tid)),
        ThreadId::AllThreads { pid } => Some((pid, u32::MAX)),
        ThreadId::AllProcesses => Some((u32::MAX, u32::MAX)),
    }
}

/// `get_filename_param()`: the hex of the whole parameter, cut at the first NUL.
fn filename(hex: &[u8]) -> Vec<u8> {
    let mut name = Vec::new();
    hextomem(&mut name, hex, hex.len() / 2);
    if let Some(i) = name.iter().position(|&c| c == 0) {
        name.truncate(i);
    }
    name
}
