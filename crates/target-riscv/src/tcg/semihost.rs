// SPDX-License-Identifier: GPL-2.0-or-later

//! RISC-V semihosting: the port of QEMU's `semihosting/arm-compat-semi.c` with the parts of
//! `semihosting/syscalls.c`, `guestfd.c` and `console.c` it needs, and
//! `target/riscv/common-semi-target.c`, copied from the arm front end.
//!
//! When the board turns it on with [`Riscv::with_semihosting`](super::Riscv::with_semihosting)
//! (QEMU's `-semihosting`), an `ebreak` between `slli zero, zero, 0x1f` and
//! `srai zero, zero, 7` raises `EXCP_SEMIHOST`, and taking that exception runs the call in
//! a0 with the parameter block at a1, puts the result in a0 and steps over the `ebreak`, as
//! `tcg_handle_semihosting()` does. In U mode the call is only made when the board also
//! allows it from userspace (`-semihosting-config userspace=on`); otherwise the `ebreak`
//! is a breakpoint.
//!
//! Every call of the semihosting specification 2.0 that QEMU implements is here, with the
//! same set as the arm port. Guest memory is accessed through the current translation
//! without permission checks or A/D updates, as `cpu_memory_rw_debug()` does.
//!
//! Differences from QEMU:
//!
//! - SYS_EXIT does not end the process: it calls [`SemihostingHost::exit`] and halts the
//!   calling CPU, so the board decides how to stop. QEMU calls `gdb_exit()` and `exit()`.
//! - SYS_READC blocks the vCPU thread in [`SemihostingHost::console_read`] where QEMU
//!   halts the vCPU until the console FIFO has data.
//! - SYS_CLOCK counts the centiseconds since semihosting was set up rather than the
//!   process CPU time `clock()` reports, and SYS_ELAPSED counts nanoseconds from the same
//!   point rather than from QEMU's start.
//! - SYS_SYSTEM runs nothing unless the host implements [`SemihostingHost::system`]; the
//!   default fails it with ENOSYS. QEMU passes the string to the host's `system()`.
//! - There is no gdbstub, so the calls are never forwarded to a debugger.
//! - Errors carry the host errno on Unix hosts, as in QEMU; on Windows they are mapped from
//!   the error kind to the Linux numbers. An unsupported call prints QEMU's message and
//!   calls [`SemihostingHost::unsupported`] instead of dumping the CPU state and aborting.
//! - Only RV64 callers exist, so the 32-bit parameter block forms are not needed.

use std::fs::{File, OpenOptions};
use std::io::{IsTerminal, Read, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use ruvm_jit::{Cpu, interrupt};
use ruvm_mem::{MemTxAttrs, MemTxResult};

use super::{Riscv, ptw};
use crate::cpu::CpuRiscvState;

const SYS_OPEN: u32 = 0x01;
const SYS_CLOSE: u32 = 0x02;
const SYS_WRITEC: u32 = 0x03;
const SYS_WRITE0: u32 = 0x04;
const SYS_WRITE: u32 = 0x05;
const SYS_READ: u32 = 0x06;
const SYS_READC: u32 = 0x07;
const SYS_ISERROR: u32 = 0x08;
const SYS_ISTTY: u32 = 0x09;
const SYS_SEEK: u32 = 0x0a;
const SYS_FLEN: u32 = 0x0c;
const SYS_TMPNAM: u32 = 0x0d;
const SYS_REMOVE: u32 = 0x0e;
const SYS_RENAME: u32 = 0x0f;
const SYS_CLOCK: u32 = 0x10;
const SYS_TIME: u32 = 0x11;
const SYS_SYSTEM: u32 = 0x12;
const SYS_ERRNO: u32 = 0x13;
const SYS_GET_CMDLINE: u32 = 0x15;
const SYS_HEAPINFO: u32 = 0x16;
const SYS_EXIT: u32 = 0x18;
const SYS_SYNCCACHE: u32 = 0x19;
const SYS_EXIT_EXTENDED: u32 = 0x20;
const SYS_ELAPSED: u32 = 0x30;
const SYS_TICKFREQ: u32 = 0x31;

/// `ADP_Stopped_ApplicationExit`, the reason code of a normal exit.
pub const ADP_STOPPED_APPLICATION_EXIT: u64 = 0x20026;

/// The errno values the calls themselves report (the same on Linux and macOS).
const EBADF: i32 = 9;
const EACCES: i32 = 13;
const EFAULT: i32 = 14;
const EINVAL: i32 = 22;
const ENOTTY: i32 = 25;
const ESPIPE: i32 = 29;
const E2BIG: i32 = 7;
const ENOSYS: i32 = 38;
const ENAMETOOLONG: i32 = 36;
const EIO: i32 = 5;

/// The contents of `:semihosting-features`: the magic and feature byte 0 with
/// SH_EXT_EXIT_EXTENDED and SH_EXT_STDOUT_STDERR.
const FEATURE_FILE: [u8; 5] = [0x53, 0x48, 0x46, 0x42, 0x03];

/// The open modes of SYS_OPEN, `gdb_open_modeflags[]`: read, write, create, truncate,
/// append.
const OPEN_MODES: [(bool, bool, bool, bool, bool); 12] = [
    (true, false, false, false, false),
    (true, false, false, false, false),
    (true, true, false, false, false),
    (true, true, false, false, false),
    (false, true, true, true, false),
    (false, true, true, true, false),
    (true, true, true, true, false),
    (true, true, true, true, false),
    (false, true, true, false, true),
    (false, true, true, false, true),
    (true, true, true, false, true),
    (true, true, true, false, true),
];

/// What a board provides for semihosting: the console, the command line, the memory
/// layout and what to do on exit.
pub trait SemihostingHost: Send + Sync {
    /// Write `buf` to the semihosting console, `qemu_semihosting_console_write()`, and
    /// return how many bytes went out (0 is an error). QEMU writes to the `chardev` of
    /// `-semihosting-config`, or to stderr without one.
    fn console_write(&self, buf: &[u8]) -> usize;

    /// Read one byte from the semihosting console, waiting for it.
    fn console_read(&self) -> u8;

    /// The guest asked to stop with exit status `code` (SYS_EXIT and SYS_EXIT_EXTENDED).
    fn exit(&self, code: u32);

    /// The command line SYS_GET_CMDLINE returns, `semihosting_get_cmdline()`: the
    /// `-semihosting-config arg=` values, or the kernel file name and `-append`.
    fn cmdline(&self) -> Option<String> {
        None
    }

    /// The heap base and limit SYS_HEAPINFO reports, `common_semi_find_bases()`: the
    /// largest gap between loaded images in the largest RAM region. The stack base and
    /// limit are the heap limit and base.
    fn heap_info(&self) -> (u64, u64);

    /// Write to host stream `fd` (1 for stdout, 2 for stderr), what a file opened as `:tt`
    /// writes to. The default writes to this process's stdout or stderr, as QEMU does.
    fn stdio_write(&self, fd: u32, buf: &[u8]) -> std::io::Result<usize> {
        if fd == 2 {
            std::io::stderr().write(buf)
        } else {
            let mut out = std::io::stdout();
            let n = out.write(buf)?;
            out.flush()?;
            Ok(n)
        }
    }

    /// Read from the host's stdin, what a file opened as `:tt` for reading reads.
    fn stdio_read(&self, buf: &mut [u8]) -> std::io::Result<usize> {
        std::io::stdin().read(buf)
    }

    /// Run `cmd` for SYS_SYSTEM and return its status, or `None` to fail with ENOSYS.
    fn system(&self, cmd: &str) -> Option<i64> {
        let _ = cmd;
        None
    }

    /// An unsupported call `nr` was made. QEMU aborts; the default does too.
    fn unsupported(&self, nr: u32) {
        let _ = nr;
        std::process::abort();
    }
}

/// An open semihosting file, `GuestFD`.
enum GuestFd {
    /// A host standard stream, `GuestFDHost` with fd 0, 1 or 2.
    Stdio(u32),
    /// A host file.
    Host(File),
    /// A file whose contents are built in, `GuestFDStatic`, with the read offset.
    Static(&'static [u8], usize),
}

/// The semihosting state of a machine: the host, the file table and `syscall_err`.
pub(crate) struct Semihosting {
    host: Arc<dyn SemihostingHost>,
    /// Whether U mode may make calls (`userspace_enabled`).
    pub(crate) userspace: bool,
    start: Instant,
    state: Mutex<SemiState>,
}

struct SemiState {
    /// `syscall_err`, the last error, which SYS_ERRNO returns.
    errno: i32,
    /// `guestfd_array`; slot 0 is never used.
    fds: Vec<Option<GuestFd>>,
}

/// A failed guest memory access: the call fails with EFAULT.
struct Fault;

/// The result of a host call, `(ret, err)` as passed to the completion callbacks.
type HostRet = (i64, i32);

fn errno_of(e: &std::io::Error) -> i32 {
    #[cfg(unix)]
    if let Some(n) = e.raw_os_error() {
        return n;
    }
    use std::io::ErrorKind as K;
    match e.kind() {
        K::NotFound => 2,
        K::PermissionDenied => EACCES,
        K::AlreadyExists => 17,
        K::InvalidInput => EINVAL,
        K::Interrupted => 4,
        K::WouldBlock => 11,
        _ => EIO,
    }
}

impl Semihosting {
    pub(crate) fn new(host: Arc<dyn SemihostingHost>, userspace: bool) -> Semihosting {
        Semihosting {
            host,
            userspace,
            start: Instant::now(),
            state: Mutex::new(SemiState { errno: 0, fds: vec![None] }),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, SemiState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl SemiState {
    /// `alloc_guestfd()`: the first free slot from 1.
    fn alloc(&mut self, fd: GuestFd) -> i64 {
        let slot = (1..self.fds.len()).find(|&i| self.fds[i].is_none());
        let i = match slot {
            Some(i) => i,
            None => {
                self.fds.push(None);
                self.fds.len() - 1
            }
        };
        self.fds[i] = Some(fd);
        i as i64
    }

    fn get(&mut self, fd: u64) -> Option<&mut GuestFd> {
        let i = usize::try_from(fd as i32).ok()?;
        self.fds.get_mut(i).and_then(|f| f.as_mut())
    }
}

/// Guest memory accesses through the current regime, `cpu_memory_rw_debug()`.
struct Guest<'a, 'c> {
    cpu: &'a mut Cpu<'c>,
    mmu_idx: usize,
}

impl Guest<'_, '_> {
    fn translate(&mut self, va: u64) -> Result<u64, Fault> {
        let st = CpuRiscvState::load(self.cpu.env);
        let as_ = self.cpu.core.address_space().clone();
        ptw::translate_debug(&st, &as_, va, self.mmu_idx).ok_or(Fault)
    }

    fn read(&mut self, va: u64, len: usize) -> Result<Vec<u8>, Fault> {
        let mut out = vec![0; len];
        let mut done = 0;
        while done < len {
            let addr = va.wrapping_add(done as u64);
            let chunk = (0x1000 - (addr & 0xfff) as usize).min(len - done);
            let pa = self.translate(addr)?;
            let as_ = self.cpu.core.address_space().clone();
            if as_.read(pa, MemTxAttrs::default(), &mut out[done..done + chunk]) != MemTxResult::OK
            {
                return Err(Fault);
            }
            done += chunk;
        }
        Ok(out)
    }

    fn write(&mut self, va: u64, data: &[u8]) -> Result<(), Fault> {
        let mut done = 0;
        while done < data.len() {
            let addr = va.wrapping_add(done as u64);
            let chunk = (0x1000 - (addr & 0xfff) as usize).min(data.len() - done);
            let pa = self.translate(addr)?;
            let as_ = self.cpu.core.address_space().clone();
            if as_.write(pa, MemTxAttrs::default(), &data[done..done + chunk]) != MemTxResult::OK {
                return Err(Fault);
            }
            done += chunk;
        }
        Ok(())
    }

    fn get_u64(&mut self, va: u64) -> Result<u64, Fault> {
        let b = self.read(va, 8)?;
        Ok(u64::from_le_bytes(b.try_into().expect("8 bytes")))
    }

    fn put_u64(&mut self, va: u64, v: u64) -> Result<(), Fault> {
        self.write(va, &v.to_le_bytes())
    }

    /// `target_strlen()`.
    fn strlen(&mut self, va: u64) -> Result<usize, Fault> {
        let mut len = 0usize;
        loop {
            let addr = va.wrapping_add(len as u64);
            let chunk = 0x1000 - (addr & 0xfff) as usize;
            let bytes = self.read(addr, chunk)?;
            if let Some(i) = bytes.iter().position(|&b| b == 0) {
                return Ok(len + i);
            }
            len += chunk;
            if len >= i32::MAX as usize {
                return Ok(len);
            }
        }
    }

    /// `validate_lock_user_string()`: the string at `va` whose length with the NUL is
    /// `tlen` (0 for unknown), or an errno.
    fn string(&mut self, va: u64, tlen: u64) -> Result<String, i32> {
        let len = if tlen == 0 {
            let l = self.strlen(va).map_err(|_| EFAULT)?;
            if l >= i32::MAX as usize {
                return Err(ENAMETOOLONG);
            }
            l + 1
        } else {
            if tlen > i32::MAX as u64 {
                return Err(ENAMETOOLONG);
            }
            let last = self.read(va + tlen - 1, 1).map_err(|_| EFAULT)?;
            if last[0] != 0 {
                return Err(EINVAL);
            }
            tlen as usize
        };
        let bytes = self.read(va, len).map_err(|_| EFAULT)?;
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        Ok(String::from_utf8_lossy(&bytes[..end]).into_owned())
    }
}

/// Run the semihosting call of the vCPU `cpu`, `do_common_semihosting()`. The caller
/// steps over the `ebreak`.
pub(crate) fn handle(_rv: &Riscv, sh: &Semihosting, cpu: &mut Cpu<'_>) {
    let st = CpuRiscvState::load(cpu.env);
    let mmu_idx = super::mmu_index_st(&st, false);
    let nr = st.gpr[10] as u32;
    let args = st.gpr[11];
    let sp = st.gpr[2];
    let mut g = Guest { cpu, mmu_idx };
    let ret = do_call(sh, &mut g, nr, args, sp);
    if let Some(ret) = ret {
        let mut st = CpuRiscvState::load(cpu.env);
        st.gpr[10] = ret;
        st.store(cpu.env);
    }
    if matches!(nr, SYS_EXIT | SYS_EXIT_EXTENDED) || (ret.is_none() && !is_known(nr)) {
        cpu.core.shared().set_interrupt(interrupt::HALT);
    }
}

fn is_known(nr: u32) -> bool {
    matches!(nr, SYS_OPEN..=SYS_SEEK | SYS_FLEN..=SYS_ERRNO | SYS_GET_CMDLINE | SYS_HEAPINFO)
        || matches!(nr, SYS_EXIT | SYS_SYNCCACHE | SYS_EXIT_EXTENDED | SYS_ELAPSED | SYS_TICKFREQ)
}

/// `common_semi_cb()`: record the error and return the value.
fn cb(sh: &Semihosting, (ret, err): HostRet) -> Option<u64> {
    if err != 0 {
        sh.state().errno = err;
    }
    Some(ret as u64)
}

/// The value of a call that has no defined return value, `common_semi_dead_cb()`.
const DEAD: Option<u64> = Some(0xdead_beef);

/// Run call `nr`; `None` leaves X0 alone (only after an exit or an unsupported call).
fn do_call(sh: &Semihosting, g: &mut Guest<'_, '_>, nr: u32, args: u64, sp: u64) -> Option<u64> {
    let arg = |g: &mut Guest<'_, '_>, n: u64| g.get_u64(args + n * 8);
    macro_rules! arg {
        ($n:expr) => {
            match arg(g, $n) {
                Ok(v) => v,
                Err(Fault) => return cb(sh, (-1, EFAULT)),
            }
        };
    }
    match nr {
        SYS_OPEN => {
            let (a0, a1, a2) = (arg!(0), arg!(1), arg!(2));
            let name = match g.string(a0, 0) {
                Ok(s) => s,
                Err(_) => return cb(sh, (-1, EFAULT)),
            };
            if a1 >= 12 {
                return cb(sh, (-1, EINVAL));
            }
            if name == ":tt" {
                // We implement SH_EXT_STDOUT_STDERR, so open for read is stdin, open for
                // write is stdout and open for append is stderr.
                let fd = if a1 < 4 {
                    0
                } else if a1 < 8 {
                    1
                } else {
                    2
                };
                let r = sh.state().alloc(GuestFd::Stdio(fd));
                return cb(sh, (r, 0));
            }
            if name == ":semihosting-features" {
                // We must fail opens for modes other than 0 ('r') or 1 ('rb').
                if a1 != 0 && a1 != 1 {
                    return cb(sh, (-1, EACCES));
                }
                let r = sh.state().alloc(GuestFd::Static(&FEATURE_FILE, 0));
                return cb(sh, (r, 0));
            }
            let path = match g.string(a0, a2 + 1) {
                Ok(p) => p,
                Err(e) => return cb(sh, (-1, e)),
            };
            let (read, write, create, truncate, append) = OPEN_MODES[a1 as usize];
            let mut o = OpenOptions::new();
            o.read(read).write(write && !append).append(append).create(create);
            o.truncate(truncate);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                o.mode(0o644);
            }
            match o.open(&path) {
                Ok(f) => {
                    let r = sh.state().alloc(GuestFd::Host(f));
                    cb(sh, (r, 0))
                }
                Err(e) => cb(sh, (-1, errno_of(&e))),
            }
        }
        SYS_CLOSE => {
            let a0 = arg!(0);
            let mut s = sh.state();
            if s.get(a0).is_none() {
                drop(s);
                return cb(sh, (-1, EBADF));
            }
            // Dropping a host file closes it; the standard streams stay open.
            s.fds[a0 as usize] = None;
            drop(s);
            cb(sh, (0, 0))
        }
        SYS_WRITEC => {
            if let Ok(b) = g.read(args, 1) {
                sh.host.console_write(&b);
            }
            DEAD
        }
        SYS_WRITE0 => {
            if let Ok(len) = g.strlen(args) {
                if let Ok(b) = g.read(args, len) {
                    sh.host.console_write(&b);
                }
            }
            DEAD
        }
        SYS_WRITE => {
            let (a0, a1, a2) = (arg!(0), arg!(1), arg!(2));
            let (ret, err) = sys_write(sh, g, a0, a1, a2);
            if err != 0 {
                sh.state().errno = err;
            }
            // SYS_READ and SYS_WRITE return the number of bytes not transferred.
            let done = if err != 0 { 0 } else { ret as u64 };
            Some(a2.wrapping_sub(done))
        }
        SYS_READ => {
            let (a0, a1, a2) = (arg!(0), arg!(1), arg!(2));
            let (ret, err) = sys_read(sh, g, a0, a1, a2);
            if err != 0 {
                sh.state().errno = err;
            }
            let done = if err != 0 { 0 } else { ret as u64 };
            Some(a2.wrapping_sub(done))
        }
        SYS_READC => {
            // The console read goes through the byte below the stack pointer.
            let c = sh.host.console_read();
            let addr = sp.wrapping_sub(1);
            if g.write(addr, &[c]).is_err() {
                return cb(sh, (-1, EFAULT));
            }
            match g.read(addr, 1) {
                Ok(b) => cb(sh, (i64::from(b[0]), 0)),
                Err(Fault) => cb(sh, (-1, EFAULT)),
            }
        }
        SYS_ISERROR => {
            let a0 = arg!(0);
            Some(u64::from((a0 as i64) < 0))
        }
        SYS_ISTTY => {
            let a0 = arg!(0);
            let r = match sh.state().get(a0) {
                None => (0, EBADF),
                Some(GuestFd::Stdio(fd)) => {
                    let tty = match fd {
                        0 => std::io::stdin().is_terminal(),
                        1 => std::io::stdout().is_terminal(),
                        _ => std::io::stderr().is_terminal(),
                    };
                    if tty { (1, 0) } else { (0, ENOTTY) }
                }
                Some(_) => (0, ENOTTY),
            };
            // common_semi_istty_cb(): ENOTTY is a plain "no", other errors are -1.
            let ret = if r.1 != 0 { if r.1 == ENOTTY { 0 } else { -1 } } else { r.0 };
            cb(sh, (ret, r.1))
        }
        SYS_SEEK => {
            let (a0, a1) = (arg!(0), arg!(1));
            let r = match sh.state().get(a0) {
                None => (-1, EBADF),
                Some(GuestFd::Stdio(_)) => (-1, ESPIPE),
                Some(GuestFd::Host(f)) => {
                    if (a1 as i64) < 0 {
                        (-1, EINVAL)
                    } else {
                        match f.seek(SeekFrom::Start(a1)) {
                            Ok(p) => (p as i64, 0),
                            Err(e) => (-1, errno_of(&e)),
                        }
                    }
                }
                Some(GuestFd::Static(data, off)) => {
                    let pos = a1 as i64;
                    if pos >= 0 && pos as usize <= data.len() {
                        *off = pos as usize;
                        (pos, 0)
                    } else {
                        (-1, EINVAL)
                    }
                }
            };
            // SYS_SEEK returns 0 on success, not the resulting offset.
            cb(sh, if r.1 == 0 { (0, 0) } else { r })
        }
        SYS_FLEN => {
            let a0 = arg!(0);
            let r = match sh.state().get(a0) {
                None => (-1, EBADF),
                Some(GuestFd::Host(f)) => match f.metadata() {
                    Ok(m) => (m.len() as i64, 0),
                    Err(e) => (-1, errno_of(&e)),
                },
                Some(GuestFd::Static(data, _)) => (data.len() as i64, 0),
                Some(GuestFd::Stdio(_)) => (0, 0),
            };
            cb(sh, r)
        }
        SYS_TMPNAM => {
            let (a0, a1, a2) = (arg!(0), arg!(1), arg!(2));
            let dir = std::env::temp_dir();
            let dir = dir.to_string_lossy();
            let s = format!(
                "{}/qemu-{:x}{:02x}",
                dir.trim_end_matches('/'),
                std::process::id(),
                a1 & 0xff
            );
            // Allow for trailing NUL.
            let mut bytes = s.into_bytes();
            bytes.push(0);
            if bytes.len() as u64 > a2 {
                return Some(u64::MAX);
            }
            if g.write(a0, &bytes).is_err() {
                return cb(sh, (-1, EFAULT));
            }
            Some(0)
        }
        SYS_REMOVE => {
            let (a0, a1) = (arg!(0), arg!(1));
            match g.string(a0, a1 + 1) {
                Ok(p) => {
                    let is_dir = std::fs::metadata(&p).map(|m| m.is_dir()).unwrap_or(false);
                    let r = if is_dir { std::fs::remove_dir(&p) } else { std::fs::remove_file(&p) };
                    match r {
                        Ok(()) => cb(sh, (0, 0)),
                        Err(e) => cb(sh, (-1, errno_of(&e))),
                    }
                }
                Err(e) => cb(sh, (-1, e)),
            }
        }
        SYS_RENAME => {
            let (a0, a1, a2, a3) = (arg!(0), arg!(1), arg!(2), arg!(3));
            let from = match g.string(a0, a1 + 1) {
                Ok(p) => p,
                Err(e) => return cb(sh, (-1, e)),
            };
            let to = match g.string(a2, a3 + 1) {
                Ok(p) => p,
                Err(e) => return cb(sh, (-1, e)),
            };
            match std::fs::rename(from, to) {
                Ok(()) => cb(sh, (0, 0)),
                Err(e) => cb(sh, (-1, errno_of(&e))),
            }
        }
        SYS_CLOCK => Some((sh.start.elapsed().as_millis() / 10) as u64),
        SYS_TIME => {
            let t = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs());
            cb(sh, (t.unwrap_or(0) as i64, 0))
        }
        SYS_SYSTEM => {
            let (a0, a1) = (arg!(0), arg!(1));
            match g.string(a0, a1 + 1) {
                Ok(cmd) => match sh.host.system(&cmd) {
                    Some(r) => cb(sh, (r, 0)),
                    None => cb(sh, (-1, ENOSYS)),
                },
                Err(e) => cb(sh, (-1, e)),
            }
        }
        SYS_ERRNO => Some(sh.state().errno as u64),
        SYS_GET_CMDLINE => {
            let (a0, a1) = (arg!(0), arg!(1));
            let cmdline = sh.host.cmdline().unwrap_or_default();
            let mut out = cmdline.into_bytes();
            // Count terminating 0.
            out.push(0);
            if out.len() as u64 > a1 {
                // Not enough space to store command-line arguments.
                return cb(sh, (-1, E2BIG));
            }
            // Adjust the command-line length.
            if g.put_u64(args + 8, out.len() as u64 - 1).is_err() {
                return cb(sh, (-1, EFAULT));
            }
            if g.write(a0, &out).is_err() {
                return cb(sh, (-1, EFAULT));
            }
            cb(sh, (0, 0))
        }
        SYS_HEAPINFO => {
            let a0 = arg!(0);
            let (base, limit) = sh.host.heap_info();
            // Heap base, heap limit, stack base, stack limit.
            for (i, v) in [base, limit, limit, base].into_iter().enumerate() {
                if g.put_u64(a0 + i as u64 * 8, v).is_err() {
                    // Couldn't write back to argument block.
                    return cb(sh, (-1, EFAULT));
                }
            }
            Some(0)
        }
        SYS_EXIT | SYS_EXIT_EXTENDED => {
            // The RV64 version of SYS_EXIT takes a parameter block, so the application-exit
            // type can return a subcode which is the exit status code from the
            // application.
            let (a0, a1) = (arg!(0), arg!(1));
            let ret = if a0 == ADP_STOPPED_APPLICATION_EXIT { a1 as u32 } else { 1 };
            sh.host.exit(ret);
            None
        }
        SYS_ELAPSED => {
            let elapsed = sh.start.elapsed().as_nanos() as u64;
            if g.put_u64(args, elapsed).is_err() {
                return cb(sh, (-1, EFAULT));
            }
            Some(0)
        }
        // QEMU always uses nsec.
        SYS_TICKFREQ => Some(1_000_000_000),
        // Clean the D-cache and invalidate the I-cache for the specified virtual address
        // range. This is a nop for us since we don't implement caches.
        SYS_SYNCCACHE => Some(0),
        _ => {
            eprintln!("qemu: Unsupported SemiHosting SWI 0x{nr:02x}");
            sh.host.unsupported(nr);
            None
        }
    }
}

/// `semihost_sys_write()`.
fn sys_write(sh: &Semihosting, g: &mut Guest<'_, '_>, fd: u64, buf: u64, len: u64) -> HostRet {
    let mut s = sh.state();
    let Some(f) = s.get(fd) else {
        return (-1, EBADF);
    };
    if let GuestFd::Static(..) = f {
        // Static files are never open for writing: EBADF.
        return (-1, EBADF);
    }
    let Ok(data) = usize::try_from(len).map_err(|_| Fault).and_then(|l| g.read(buf, l)) else {
        return (-1, EFAULT);
    };
    let r = match f {
        GuestFd::Stdio(n) => sh.host.stdio_write(*n, &data),
        GuestFd::Host(file) => file.write(&data),
        GuestFd::Static(..) => unreachable!("handled above"),
    };
    match r {
        Ok(n) => (n as i64, 0),
        Err(e) => (-1, errno_of(&e)),
    }
}

/// `semihost_sys_read()`.
fn sys_read(sh: &Semihosting, g: &mut Guest<'_, '_>, fd: u64, buf: u64, len: u64) -> HostRet {
    let mut s = sh.state();
    let Some(f) = s.get(fd) else {
        return (-1, EBADF);
    };
    let Ok(len) = usize::try_from(len) else {
        return (-1, EFAULT);
    };
    let mut data = vec![0; len];
    let r = match f {
        GuestFd::Stdio(_) => sh.host.stdio_read(&mut data),
        GuestFd::Host(file) => file.read(&mut data),
        GuestFd::Static(bytes, off) => {
            let n = len.min(bytes.len() - *off);
            data[..n].copy_from_slice(&bytes[*off..*off + n]);
            *off += n;
            Ok(n)
        }
    };
    drop(s);
    match r {
        Ok(n) => {
            if g.write(buf, &data[..n]).is_err() {
                return (-1, EFAULT);
            }
            (n as i64, 0)
        }
        Err(e) => (-1, errno_of(&e)),
    }
}
