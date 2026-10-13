// SPDX-License-Identifier: GPL-2.0-or-later

//! `-g`: the calls `linux-user/` makes into `gdbstub/user.c`, and what the stub reads and
//! changes of the program through a [`Cx`].
//!
//! The protocol is `ruvm-gdbstub`, which sits above this crate and registers itself with
//! [`set_stub`] before [`crate::run`]. Every hook does nothing unless `-g` started the stub.
//!
//! A vCPU belongs to its host thread, so the registers of another thread are the copy that
//! thread last left: when it started, entered a system call or went on after a stop. A thread
//! blocked in a system call, which is where the other threads of a stopped program usually
//! are, reads as it is. One running guest code reads as it was then, and writes to it are
//! dropped.

use std::cell::RefCell;
use std::fmt;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use ruvm_jit::{Cpu, CpuShared, bp};
use ruvm_user_common::space::PAGE_SIZE;

use crate::host::sys;
use crate::procfs;
use crate::signal::{self, Task};
use crate::syscall::{Proc, lock};

/// What `linux-user/` calls of the stub, `gdbstub/user.c`.
pub trait Stub: Sync {
    /// `gdbserver_start()`, with what `-g` was given. With `suspend=y`, the default, it returns
    /// once the debugger lets the program run.
    fn start(&self, cx: &mut Cx<'_, '_>, args: &str) -> Result<(), String>;
    /// `gdb_handlesig()`: tells the debugger the thread stopped for target signal `sig`, unless
    /// it is 0, and serves it until it lets the thread go on. Returns the signal the thread
    /// takes instead, 0 for none.
    fn handlesig(
        &self,
        cx: &mut Cx<'_, '_>,
        sig: i32,
        reason: Option<&str>,
        siginfo: Option<&[u8]>,
    ) -> i32;
    /// `gdb_exit()`: the process exits with `code`.
    fn exit(&self, code: i32);
    /// `gdb_signalled()`: the process dies of target signal `sig`.
    fn signalled(&self, sig: i32);
    /// `gdb_syscall_entry()`, with the target's number of the call.
    fn syscall_entry(&self, cx: &mut Cx<'_, '_>, num: u64);
    /// `gdb_syscall_return()`.
    fn syscall_return(&self, cx: &mut Cx<'_, '_>, num: u64);
    /// `gdbserver_fork_start()`, before the host `fork()`.
    fn fork_start(&self);
    /// `gdbserver_fork_end()`: `pid` is what `fork()` returned, -1 when it failed.
    fn fork_end(&self, cx: &mut Cx<'_, '_>, pid: i64);
}

static STUB: OnceLock<&'static dyn Stub> = OnceLock::new();

/// Whether `-g` started the stub.
static ON: AtomicBool = AtomicBool::new(false);

/// The process, for the stub's calls that do not come from a system call.
static PROC: OnceLock<Arc<Proc>> = OnceLock::new();

/// Makes `stub` what `-g` starts.
pub fn set_stub(stub: &'static dyn Stub) {
    let _ = STUB.set(stub);
}

fn stub() -> Option<&'static dyn Stub> {
    if ON.load(Ordering::Relaxed) { STUB.get().copied() } else { None }
}

/// A guest thread as the debugger sees it, `CPU_FOREACH()` order being the order of the list.
struct Thread {
    tid: AtomicU32,
    shared: Arc<CpuShared>,
    /// `TaskState.orig_ax`.
    orig_ax: AtomicU64,
    /// The registers as the thread last left them.
    saved: Mutex<Vec<u8>>,
}

static THREADS: Mutex<Vec<Arc<Thread>>> = Mutex::new(Vec::new());

thread_local! {
    static ME: RefCell<Option<Arc<Thread>>> = const { RefCell::new(None) };
}

fn me() -> Option<Arc<Thread>> {
    ME.with(|m| m.borrow().clone())
}

/// The calling thread's ID.
pub fn gettid() -> u32 {
    sys(libc::SYS_gettid, &[]) as u32
}

fn find(tid: u32) -> Option<Arc<Thread>> {
    lock(&THREADS).iter().find(|t| t.tid.load(Ordering::Relaxed) == tid).cloned()
}

/// The emulator's name, for its messages.
static PROG: OnceLock<String> = OnceLock::new();

/// `bprm->filename`, the program as the command line names it.
static EXEC: OnceLock<String> = OnceLock::new();

/// The emulator's name as messages start with it.
pub fn prog() -> &'static str {
    PROG.get().map_or("qemu", String::as_str)
}

/// `strerror()` of an error, without Rust's `(os error N)`.
pub fn strerror(e: &std::io::Error) -> String {
    crate::strerror_of(e)
}

/// The first guest thread, `first_cpu`.
pub fn first_thread() -> Option<u32> {
    lock(&THREADS).first().map(|t| t.tid.load(Ordering::Relaxed))
}

/// `gdbserver_start()` for the first thread, on `cpu`, when `-g` gave `args`. `prog` is the
/// emulator's name.
pub(crate) fn start(
    p: &Arc<Proc>,
    t: &Task,
    cpu: &mut Cpu<'_>,
    prog: &str,
    exec: &str,
    args: &str,
) -> Result<(), String> {
    let Some(s) = STUB.get().copied() else {
        return Err("qemu: option '-g' is not supported by ruvm yet".to_string());
    };
    let _ = PROG.set(prog.to_string());
    let _ = EXEC.set(exec.to_string());
    let _ = PROC.set(Arc::clone(p));
    ON.store(true, Ordering::Relaxed);
    register(cpu);
    let mut cx = Cx::new(cpu, Some(t));
    s.start(&mut cx, args)
}

/// Adds the calling thread, which runs `cpu`, to the threads the debugger sees.
pub(crate) fn register(cpu: &Cpu<'_>) {
    if !ON.load(Ordering::Relaxed) {
        return;
    }
    let t = Arc::new(Thread {
        tid: AtomicU32::new(gettid()),
        shared: cpu.shared(),
        orig_ax: AtomicU64::new(0),
        saved: Mutex::new(cpu.env.to_vec()),
    });
    lock(&THREADS).push(Arc::clone(&t));
    ME.with(|m| *m.borrow_mut() = Some(t));
}

/// Takes the calling thread, which is exiting, off the list.
pub(crate) fn unregister() {
    if let Some(t) = ME.with(|m| m.borrow_mut().take()) {
        lock(&THREADS).retain(|x| !Arc::ptr_eq(x, &t));
    }
}

/// Keeps the registers of the calling thread for the debugger to read while it runs.
fn save(cpu: &Cpu<'_>) {
    if let Some(t) = me() {
        lock(&t.saved).copy_from_slice(cpu.env);
    }
}

/// Sets `orig_ax` of the calling thread, the system call it is making.
pub(crate) fn set_orig_ax(v: u64) {
    if ON.load(Ordering::Relaxed) {
        if let Some(t) = me() {
            t.orig_ax.store(v, Ordering::Relaxed);
        }
    }
}

/// `gdb_handlesig()` for a signal the thread takes, with its `siginfo`.
pub(crate) fn handlesig(t: &Task, cpu: &mut Cpu<'_>, sig: i32, info: &[u8]) -> i32 {
    let Some(s) = stub() else { return sig };
    let sig = s.handlesig(&mut Cx::new(cpu, Some(t)), sig, None, Some(info));
    save(cpu);
    sig
}

/// `gdb_syscall_entry()`.
pub(crate) fn syscall_entry(t: &Task, cpu: &mut Cpu<'_>, num: u64) {
    let Some(s) = stub() else { return };
    save(cpu);
    s.syscall_entry(&mut Cx::new(cpu, Some(t)), num);
}

/// `gdb_syscall_return()`.
pub(crate) fn syscall_return(t: &Task, cpu: &mut Cpu<'_>, num: u64) {
    let Some(s) = stub() else { return };
    s.syscall_return(&mut Cx::new(cpu, Some(t)), num);
    save(cpu);
}

/// `gdb_exit()`.
pub(crate) fn exit(code: i32) {
    if let Some(s) = stub() {
        s.exit(code);
    }
}

/// `gdb_signalled()`.
pub(crate) fn signalled(sig: i32) {
    if let Some(s) = stub() {
        s.signalled(sig);
    }
}

/// `gdbserver_fork_start()`.
pub(crate) fn fork_start() {
    if let Some(s) = stub() {
        s.fork_start();
    }
}

/// The child's thread list and `gdbserver_fork_end()`: `pid` is what the fork returned.
pub(crate) fn fork_end(t: &Task, cpu: &mut Cpu<'_>, pid: i64) {
    let Some(s) = stub() else { return };
    if pid == 0 {
        // The other threads did not come along.
        if let Some(me) = me() {
            me.tid.store(gettid(), Ordering::Relaxed);
            lock(&THREADS).retain(|x| Arc::ptr_eq(x, &me));
        }
    }
    s.fork_end(&mut Cx::new(cpu, Some(t)), pid.max(-1));
}

/// Runs `f` on the vCPU of the first thread, outside any system call, with every other vCPU
/// stopped: `async_safe_run_on_cpu(first_cpu)`.
pub fn on_first_cpu(f: impl FnOnce(&mut Cx<'_, '_>) + Send + 'static) {
    let first = lock(&THREADS).first().map(|t| Arc::clone(&t.shared));
    if let Some(c) = first {
        c.async_safe_run_on_cpu(move |cpu| f(&mut Cx::new(cpu, None)));
    }
}

/// The registers of a thread: its vCPU when it is the one stopped, else a copy of what it
/// left, which takes no writes.
pub enum Regs<'x, 'y> {
    /// The calling thread's vCPU.
    Live(&'x mut Cpu<'y>),
    /// A copy of another thread's.
    Saved(Vec<u8>),
}

impl fmt::Debug for Regs<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Live(cpu) => f.debug_tuple("Live").field(&cpu.core.shared().cpu_index).finish(),
            Self::Saved(v) => f.debug_tuple("Saved").field(&v.len()).finish(),
        }
    }
}

/// What the stub sees of the program from the thread it runs on.
pub struct Cx<'a, 'b> {
    cpu: &'a mut Cpu<'b>,
    task: Option<&'a Task>,
    p: &'static Proc,
}

impl fmt::Debug for Cx<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cx").field("cpu", &self.cpu.core.shared().cpu_index).finish_non_exhaustive()
    }
}

impl<'a, 'b> Cx<'a, 'b> {
    fn new(cpu: &'a mut Cpu<'b>, task: Option<&'a Task>) -> Self {
        let p = PROC.get().expect("-g started");
        Cx { cpu, task, p }
    }

    /// `UNAME_MACHINE` of the target, which picks its registers.
    pub fn machine(&self) -> &'static str {
        crate::guest::guest().machine
    }

    /// The process ID.
    pub fn pid(&self) -> u32 {
        sys(libc::SYS_getpid, &[]) as u32
    }

    /// The thread ID of the calling thread.
    pub fn tid(&self) -> u32 {
        me().map_or_else(gettid, |t| t.tid.load(Ordering::Relaxed))
    }

    /// The threads, in the order they started: thread ID and `cpu_index`.
    pub fn threads(&self) -> Vec<(u32, usize)> {
        lock(&THREADS).iter().map(|t| (t.tid.load(Ordering::Relaxed), t.shared.cpu_index)).collect()
    }

    /// The registers of thread `tid`, `None` when there is no such thread.
    pub fn regs(&mut self, tid: u32) -> Option<Regs<'_, 'b>> {
        if tid == self.tid() {
            return Some(Regs::Live(self.cpu));
        }
        find(tid).map(|t| Regs::Saved(lock(&t.saved).clone()))
    }

    /// `orig_ax` of thread `tid`.
    pub fn orig_ax(&self, tid: u32) -> u64 {
        find(tid).map_or(0, |t| t.orig_ax.load(Ordering::Relaxed))
    }

    /// Sets `orig_ax` of thread `tid`.
    pub fn set_orig_ax(&self, tid: u32, v: u64) {
        if let Some(t) = find(tid) {
            t.orig_ax.store(v, Ordering::Relaxed);
        }
    }

    /// `cpu_memory_rw_debug()` reading: every page of the range must be mapped, but need not
    /// be readable.
    pub fn read_memory(&self, addr: u64, buf: &mut [u8]) -> bool {
        let s = self.p.space();
        if !s.check(addr, buf.len() as u64, 0) {
            return false;
        }
        if s.read_raw(addr, buf) {
            return true;
        }
        // A page the host may not read either is read past its protection, as QEMU does.
        File::open("/proc/self/mem").and_then(|f| f.read_exact_at(buf, s.g2h(addr) as u64)).is_ok()
    }

    /// `cpu_memory_rw_debug()` writing: every page of the range must be mapped, but need not
    /// be writable. Translated code of the range is dropped.
    pub fn write_memory(&self, addr: u64, buf: &[u8]) -> bool {
        let s = self.p.space();
        s.check(addr, buf.len() as u64, 0) && s.write_raw(addr, buf)
    }

    /// `cpu_breakpoint_insert()` with `BP_GDB` on every thread. The code of the page is
    /// translated again, so that code chained to it sees the breakpoint.
    pub fn breakpoint_insert(&mut self, pc: u64) {
        self.cpu.breakpoint_insert(pc, bp::GDB);
        self.others(move |c| c.breakpoint_insert(pc, bp::GDB));
        self.p.space().notify_write(pc & !(PAGE_SIZE - 1), PAGE_SIZE);
    }

    /// `cpu_breakpoint_remove()` with `BP_GDB` on every thread: `false` when there was none.
    pub fn breakpoint_remove(&mut self, pc: u64) -> bool {
        let found = self.cpu.breakpoint_remove(pc, bp::GDB);
        self.others(move |c| {
            c.breakpoint_remove(pc, bp::GDB);
        });
        found
    }

    /// `cpu_breakpoint_remove_all()` with `BP_GDB` on thread `tid`.
    pub fn breakpoint_remove_all(&mut self, tid: u32) {
        let f = |c: &mut Cpu<'_>| c.core.breakpoints.retain(|b| b.flags & bp::GDB == 0);
        self.on_thread(tid, f);
    }

    /// `cpu_single_step()` of thread `tid`.
    pub fn single_step(&mut self, tid: u32, on: bool) {
        self.on_thread(tid, move |c| c.core.singlestep_enabled = on);
    }

    /// Runs `f` on the vCPU of thread `tid`: now for the calling thread, else before the
    /// thread next runs guest code.
    fn on_thread(&mut self, tid: u32, f: impl FnOnce(&mut Cpu<'_>) + Send + 'static) {
        if tid == self.tid() {
            f(self.cpu);
        } else if let Some(t) = find(tid) {
            t.shared.async_run_on_cpu(f);
        }
    }

    /// Runs `f` on the vCPU of every other thread before it next runs guest code.
    fn others(&self, f: impl Fn(&mut Cpu<'_>) + Send + Sync + Clone + 'static) {
        let me = self.tid();
        let others: Vec<_> = lock(&THREADS)
            .iter()
            .filter(|t| t.tid.load(Ordering::Relaxed) != me)
            .map(|t| Arc::clone(&t.shared))
            .collect();
        for c in others {
            c.async_run_on_cpu(f.clone());
        }
    }

    /// `info->saved_auxv` and `info->auxv_len`.
    pub fn auxv(&self) -> (u64, u64) {
        self.p.image().auxv
    }

    /// `info->code_offset` and `info->data_offset`, both the program's load bias.
    pub fn load_bias(&self) -> u64 {
        self.p.image().load_bias
    }

    /// `bprm->filename`, the program as the command line names it.
    pub fn exec_file(&self) -> &str {
        EXEC.get().map_or("", String::as_str)
    }

    /// `do_guest_openat()` at `AT_FDCWD`'s stand-in 0, as QEMU calls it, with `flags` and
    /// `mode` as given: a file descriptor or `-errno`.
    pub fn open(&self, name: &[u8], flags: u64, mode: u64) -> i64 {
        let path = if procfs::is_proc_myself(name, "exe") {
            self.p.exec_path().as_bytes().to_vec()
        } else {
            if let Some(fd) = self.task.and_then(|t| procfs::fake_open(self.p, t, name)) {
                return fd;
            }
            match std::str::from_utf8(name) {
                Ok(n) => crate::elf::path(self.p.ld_prefix(), n).into_bytes(),
                Err(_) => name.to_vec(),
            }
        };
        let mut c = path;
        c.push(0);
        sys(libc::SYS_openat, &[0, c.as_ptr() as u64, flags, mode])
    }

    /// `pread()` of at most `count` bytes: the bytes or `-errno`.
    pub fn pread(&self, fd: i32, count: usize, off: u64) -> Result<Vec<u8>, i64> {
        let mut buf = vec![0u8; count];
        let r = sys(libc::SYS_pread64, &[fd as u64, buf.as_mut_ptr() as u64, count as u64, off]);
        if r < 0 {
            return Err(r);
        }
        buf.truncate(r as usize);
        Ok(buf)
    }

    /// `close()`: 0 or `-errno`.
    pub fn close(&self, fd: i32) -> i64 {
        sys(libc::SYS_close, &[fd as u64])
    }

    /// `do_guest_readlink()` into `max` bytes: the link or `-errno`.
    pub fn readlink(&self, name: &[u8], max: usize) -> Result<Vec<u8>, i64> {
        if procfs::is_proc_myself(name, "exe") {
            let e = self.p.exec_path().as_bytes();
            return Ok(e[..e.len().min(max)].to_vec());
        }
        let mut c = match std::str::from_utf8(name) {
            Ok(n) => crate::elf::path(self.p.ld_prefix(), n).into_bytes(),
            Err(_) => name.to_vec(),
        };
        c.push(0);
        let mut buf = vec![0u8; max];
        let r = sys(
            libc::SYS_readlinkat,
            &[libc::AT_FDCWD as u64, c.as_ptr() as u64, buf.as_mut_ptr() as u64, max as u64],
        );
        if r < 0 {
            return Err(r);
        }
        buf.truncate(r as usize);
        Ok(buf)
    }

    /// `qemu_kill_thread()` of thread `tid` with target signal `sig`.
    pub fn kill_thread(&self, tid: u32, sig: i32) {
        let h = signal::t2h(sig);
        if (1..65).contains(&h) {
            sys(libc::SYS_tgkill, &[u64::from(self.pid()), u64::from(tid), h as u64]);
        }
    }
}
