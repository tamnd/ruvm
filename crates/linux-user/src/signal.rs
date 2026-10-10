// SPDX-License-Identifier: GPL-2.0-or-later

//! `linux-user/signal.c`: guest signals on top of host ones.
//!
//! The host kernel delivers every signal the guest can receive to [`host_signal`], which runs
//! on the thread of the vCPU it interrupts. That only records the signal in the thread's
//! [`ThreadSignals`], blocks further host signals until the guest handler has been entered and
//! kicks the vCPU out of translated code. The cpu loop then calls
//! [`process_pending_signals`], which runs the guest's disposition: the default action, or a
//! signal frame on the guest stack and a jump to its handler.
//!
//! Signal numbers are the target's except where the names say host. The host mask of a thread
//! is the guest's own mask, translated, whenever guest code runs or a guest system call sleeps,
//! so the host kernel decides when a signal is delivered as it would for the guest. The guest
//! RT signals start at host `SIGRTMIN + 2`; `SIGRTMIN` itself interrupts vCPUs and the next one
//! carries the guest's `SIGABRT`, to tell it from an abort of the emulator.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use ruvm_jit::{Cpu, CpuShared};
use ruvm_user_common::GuestSpace;

use crate::host;

/// `TARGET_NSIG`, and the host's `_NSIG - 1`: signals are 1 to 64.
pub(crate) const NSIG: i32 = 64;
/// What an unmapped signal translates to, `_NSIG`.
const UNMAPPED: i32 = NSIG + 1;

pub(crate) const SIGILL: i32 = 4;
pub(crate) const SIGTRAP: i32 = 5;
pub(crate) const SIGABRT: i32 = 6;
pub(crate) const SIGBUS: i32 = 7;
pub(crate) const SIGFPE: i32 = 8;
pub(crate) const SIGKILL: i32 = 9;
pub(crate) const SIGSEGV: i32 = 11;
const SIGQUIT: i32 = 3;
const SIGSTOP: i32 = 19;
const SIGCHLD: i32 = 17;
const SIGCONT: i32 = 18;
const SIGTSTP: i32 = 20;
const SIGTTIN: i32 = 21;
const SIGTTOU: i32 = 22;
const SIGURG: i32 = 23;
const SIGWINCH: i32 = 28;
const SIGIO: i32 = 29;

/// `TARGET_SIG_DFL`, `TARGET_SIG_IGN` and `TARGET_SIG_ERR`.
pub(crate) const SIG_DFL: u64 = 0;
pub(crate) const SIG_IGN: u64 = 1;
const SIG_ERR: u64 = u64::MAX;

pub(crate) const SA_SIGINFO: u64 = 4;
pub(crate) const SA_ONSTACK: u64 = 0x0800_0000;
pub(crate) const SA_RESTART: u64 = 0x1000_0000;
pub(crate) const SA_NODEFER: u64 = 0x4000_0000;
pub(crate) const SA_RESETHAND: u64 = 0x8000_0000;
pub(crate) const SA_RESTORER: u64 = 0x0400_0000;

/// `si_code` values.
const SI_USER: i32 = 0;
const SI_KERNEL: i32 = 0x80;
const SI_TKILL: i32 = -6;
const CLD_EXITED: i32 = 1;
pub(crate) const SEGV_MAPERR: i32 = 1;
pub(crate) const SEGV_ACCERR: i32 = 2;
pub(crate) const FPE_INTDIV: i32 = 1;
pub(crate) const TRAP_BRKPT: i32 = 1;
pub(crate) const ILL_ILLOPN: i32 = 2;
pub(crate) const ILL_ILLOPC: i32 = 1;
pub(crate) const BUS_ADRALN: i32 = 1;
pub(crate) const SEGV_MTEAERR: i32 = 8;
pub(crate) const SEGV_MTESERR: i32 = 9;

/// `SS_ONSTACK` and `SS_DISABLE`.
const SS_ONSTACK: i32 = 1;
const SS_DISABLE: i32 = 2;

/// `QEMU_ERESTARTSYS`: the system call is restarted once the signal is delivered.
pub(crate) const ERESTARTSYS: i64 = -512;
/// `QEMU_ESIGRETURN`: the registers are already what the guest returns to.
pub(crate) const ESIGRETURN: i64 = -513;

const EINVAL: i64 = libc::EINVAL as i64;
const EFAULT: i64 = libc::EFAULT as i64;

/// A `siginfo_t`, as the guest sees it.
pub(crate) type Info = [u8; 128];

pub(crate) fn get32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

pub(crate) fn put32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

pub(crate) fn get64(b: &[u8], off: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(a)
}

pub(crate) fn put64(b: &mut [u8], off: usize, v: u64) {
    b[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

/// `host_to_target_signal_table` and `target_to_host_signal_table`.
struct Tables {
    h2t: [u8; UNMAPPED as usize],
    t2h: [u8; UNMAPPED as usize],
    /// `host_interrupt_signal`.
    interrupt: i32,
}

static TABLES: OnceLock<Tables> = OnceLock::new();

/// `signal_table_init()` with the default RT signal mapping.
fn table_init() -> Tables {
    let rtmin = libc::SIGRTMIN();
    let rtmax = libc::SIGRTMAX();
    let mut h2t = [0u8; UNMAPPED as usize];
    let mut t2h = [0u8; UNMAPPED as usize];
    for s in 1..32u8 {
        h2t[s as usize] = s;
    }
    let (mut hsig, mut tsig) = (rtmin + 2, 32);
    while hsig <= rtmax && tsig <= NSIG {
        h2t[hsig as usize] = tsig as u8;
        hsig += 1;
        tsig += 1;
    }
    // The target SIGABRT moves to a host RT signal, so that an abort of the emulator is not
    // taken for the guest's.
    h2t[SIGABRT as usize] = 0;
    let mut interrupt = 0;
    let mut abrt = false;
    for hsig in rtmin..=rtmax {
        if h2t[hsig as usize] == 0 {
            if interrupt != 0 {
                h2t[hsig as usize] = SIGABRT as u8;
                abrt = true;
                break;
            }
            interrupt = hsig;
        }
    }
    if !abrt {
        eprintln!("No rt signals left for interrupt and SIGABRT mapping");
        std::process::exit(1);
    }
    for (hsig, &tsig) in h2t.iter().enumerate().skip(1) {
        if tsig != 0 {
            t2h[tsig as usize] = hsig as u8;
        }
    }
    h2t[SIGABRT as usize] = SIGABRT as u8;
    for v in h2t.iter_mut().skip(1).chain(t2h.iter_mut().skip(1)) {
        if *v == 0 {
            *v = UNMAPPED as u8;
        }
    }
    Tables { h2t, t2h, interrupt }
}

fn tables() -> &'static Tables {
    TABLES.get_or_init(table_init)
}

/// `host_to_target_signal()`.
pub(crate) fn h2t(sig: i32) -> i32 {
    if !(0..UNMAPPED).contains(&sig) {
        return sig;
    }
    i32::from(tables().h2t[sig as usize])
}

/// `target_to_host_signal()`.
pub(crate) fn t2h(sig: i32) -> i32 {
    if !(0..UNMAPPED).contains(&sig) {
        return sig;
    }
    i32::from(tables().t2h[sig as usize])
}

/// The bit of host signal `sig` in a host mask, none for one out of range.
fn bit(sig: i32) -> u64 {
    if (1..=NSIG).contains(&sig) { 1 << (sig - 1) } else { 0 }
}

/// `sigismember()` on a host mask: a signal out of range counts as blocked, as its -1 does.
fn is_member(set: u64, sig: i32) -> bool {
    !(1..=NSIG).contains(&sig) || set & bit(sig) != 0
}

/// `target_to_host_sigset()`.
pub(crate) fn t2h_set(set: u64) -> u64 {
    (1..=NSIG).filter(|&s| set & bit(s) != 0).fold(0, |m, s| m | bit(t2h(s)))
}

/// `host_to_target_sigset()`.
pub(crate) fn h2t_set(set: u64) -> u64 {
    (1..=NSIG).filter(|&s| set & bit(s) != 0).fold(0, |m, s| m | bit(h2t(s)))
}

/// `core_dump_signal()`.
fn core_dump_signal(sig: i32) -> bool {
    matches!(sig, SIGABRT | SIGFPE | SIGILL | SIGQUIT | SIGSEGV | SIGTRAP | SIGBUS)
}

/// `host_to_target_siginfo()` of a host `siginfo_t` as the kernel wrote it.
pub(crate) fn host_to_target_siginfo(h: &Info) -> Info {
    let sig = h2t(get32(h, 0) as i32);
    // tswap_siginfo() keeps the low 16 bits of si_code, sign extended.
    let code = i32::from(get32(h, 8) as i16);
    let mut t = [0u8; 128];
    put32(&mut t, 0, sig as u32);
    put32(&mut t, 8, code as u32);
    let copy = |t: &mut Info, from: usize, to: usize| t[from..to].copy_from_slice(&h[from..to]);
    match code {
        // kill(), tkill() and tgkill(), or the kernel: pid and uid.
        SI_USER | SI_TKILL | SI_KERNEL => copy(&mut t, 16, 24),
        _ => match sig {
            SIGCHLD => {
                // pid, uid, status, utime, stime.
                copy(&mut t, 16, 48);
                if code != CLD_EXITED {
                    let st = get32(h, 24) as i32;
                    put32(&mut t, 24, (h2t(st & 0x7f) | (st & !0x7f)) as u32);
                }
            }
            // band and fd.
            SIGIO => copy(&mut t, 16, 28),
            // pid, uid and the value of sigqueue() and the like.
            _ => copy(&mut t, 16, 32),
        },
    }
    t
}

/// `target_to_host_siginfo()`, for `rt_sigqueueinfo()`: the `_rt` fields.
fn target_to_host_siginfo(t: &Info) -> Info {
    let mut h = [0u8; 128];
    h[..12].copy_from_slice(&t[..12]);
    h[16..32].copy_from_slice(&t[16..32]);
    h
}

/// One entry of `sigtab`: a pending signal and its information.
struct Slot {
    pending: AtomicBool,
    info: [AtomicU64; 16],
}

impl Slot {
    const fn new() -> Self {
        Slot { pending: AtomicBool::new(false), info: [const { AtomicU64::new(0) }; 16] }
    }
}

/// The half of a thread's signal state that the host signal handler writes: only atomics, so
/// that the handler can not observe it half written.
pub(crate) struct ThreadSignals {
    /// `signal_pending`, what `safe_syscall` checks.
    pub(crate) pending: AtomicU32,
    /// `sigtab`, by target signal.
    sigtab: [Slot; NSIG as usize],
    /// The vCPU to kick out of translated code. Only its own thread sets it, with every host
    /// signal blocked, so the handler's `try_lock` always gets it.
    cpu: Mutex<Option<Arc<CpuShared>>>,
}

impl ThreadSignals {
    fn new() -> Self {
        ThreadSignals {
            pending: AtomicU32::new(0),
            sigtab: [const { Slot::new() }; NSIG as usize],
            cpu: Mutex::new(None),
        }
    }

    fn cpu_exit(&self) {
        if let Ok(c) = self.cpu.try_lock() {
            if let Some(c) = c.as_ref() {
                c.cpu_exit();
            }
        }
    }
}

/// The signal state of threads that have exited, for the next ones.
static FREE: Mutex<Vec<&'static ThreadSignals>> = Mutex::new(Vec::new());

thread_local! {
    /// `thread_cpu`, as far as signals are concerned.
    static CURRENT: Cell<Option<&'static ThreadSignals>> = const { Cell::new(None) };
}

/// A guest system call: `safe_syscall()` on the calling thread's pending flag, so that it
/// returns [`ERESTARTSYS`] rather than sleeping through a signal that arrived just before.
pub(crate) fn guest_syscall(nr: i64, a: [u64; 6]) -> i64 {
    match CURRENT.get() {
        Some(ts) => host::safe_syscall(&ts.pending, nr, a),
        None => host::syscall(nr, a),
    }
}

/// [`guest_syscall`] with fewer arguments.
pub(crate) fn guest_sys(nr: i64, args: &[u64]) -> i64 {
    let mut a = [0u64; 6];
    a[..args.len()].copy_from_slice(args);
    guest_syscall(nr, a)
}

/// A `struct target_sigaction`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Sigaction {
    pub(crate) handler: u64,
    pub(crate) flags: u64,
    pub(crate) restorer: u64,
    pub(crate) mask: u64,
}

/// `sigact_table`, shared by all threads.
static SIGACT: Mutex<[Sigaction; NSIG as usize]> =
    Mutex::new([Sigaction { handler: 0, flags: 0, restorer: 0, mask: 0 }; NSIG as usize]);

fn sigact() -> MutexGuard<'static, [Sigaction; NSIG as usize]> {
    SIGACT.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The name in "QEMU internal" messages.
static PROG: OnceLock<String> = OnceLock::new();

/// The rest of a thread's signal state, owned by its vCPU thread: the parts of `TaskState`
/// the signal handler does not write.
pub(crate) struct Task {
    pub(crate) ts: &'static ThreadSignals,
    /// `signal_mask`, a host mask.
    pub(crate) signal_mask: u64,
    /// `in_sigsuspend` and `sigsuspend_mask`.
    in_sigsuspend: bool,
    sigsuspend_mask: u64,
    /// `sigaltstack_used`: sp and size.
    altstack: (u64, u64),
    /// `sync_signal`: a signal the vCPU raised itself.
    sync: Option<Info>,
    /// `child_tidptr`: cleared and woken when the thread exits.
    pub(crate) child_tidptr: u64,
    /// `start_boottime`, in clock ticks since boot.
    pub(crate) start_boottime: u64,
}

impl Task {
    /// The signal state of the calling thread, with `mask` as its signal mask, and the vCPU
    /// that a signal kicks out of translated code.
    pub(crate) fn new(mask: u64, cpu: Arc<CpuShared>) -> Self {
        let reuse = FREE.lock().unwrap_or_else(std::sync::PoisonError::into_inner).pop();
        let ts: &'static ThreadSignals =
            reuse.unwrap_or_else(|| Box::leak(Box::new(ThreadSignals::new())));
        *ts.cpu.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(cpu);
        CURRENT.set(Some(ts));
        Task {
            ts,
            signal_mask: mask,
            in_sigsuspend: false,
            sigsuspend_mask: 0,
            altstack: (0, 0),
            sync: None,
            child_tidptr: 0,
            start_boottime: boottime_ticks(),
        }
    }

    /// The end of a thread: with every host signal blocked, its signal state goes back for the
    /// next thread.
    pub(crate) fn exit_thread(self) {
        host::set_mask(!0);
        CURRENT.set(None);
        let ts = self.ts;
        *ts.cpu.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        ts.pending.store(0, Ordering::SeqCst);
        for k in &ts.sigtab {
            k.pending.store(false, Ordering::Relaxed);
        }
        FREE.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(ts);
    }

    /// The host mask while guest code runs: the guest's, with the signals of the emulator's
    /// own faults always let through.
    pub(crate) fn run_mask(&self) -> u64 {
        self.signal_mask & !bit(libc::SIGSEGV) & !bit(libc::SIGBUS)
    }
}

/// `CLOCK_BOOTTIME` in clock ticks, of which Linux has 100 a second.
fn boottime_ticks() -> u64 {
    let mut ts = [0u64; 2];
    if host::sys(libc::SYS_clock_gettime, &[libc::CLOCK_BOOTTIME as u64, ts.as_mut_ptr() as u64])
        != 0
    {
        return 0;
    }
    ts[0] * 100 + ts[1] * 100 / 1_000_000_000
}

/// `signal_init()`: the conversion tables, the guest's initial dispositions from the host's,
/// and the host handler on every signal whose default action dumps core. `prog` names the
/// emulator in its own fault messages.
pub(crate) fn signal_init(prog: &str) {
    let _ = PROG.set(prog.to_string());
    let tab = tables();
    let mut sa = sigact();
    for tsig in 1..=NSIG {
        let hsig = t2h(tsig);
        if hsig >= UNMAPPED {
            continue;
        }
        let old = if tsig == SIGABRT {
            let old = host::disposition(libc::SIGABRT);
            host::install_handler(hsig, false);
            old
        } else {
            let old = host::disposition(hsig);
            if core_dump_signal(tsig) {
                host::install_handler(hsig, false);
            }
            old
        };
        sa[tsig as usize - 1].handler = if old == SIG_IGN { SIG_IGN } else { SIG_DFL };
    }
    host::install_handler(tab.interrupt, false);
}

/// `host_signal_handler()` past the kernel's structures: the signal `sig` with the host
/// siginfo `raw`, interrupting the thread at `pc`, whose mask after the handler is `mask`.
/// Only atomics and the stack are used here.
pub(crate) fn host_signal(sig: i32, raw: &[u64; 16], pc: &mut i64, mask: &mut u64) {
    let Some(tab) = TABLES.get() else { return };
    let Some(ts) = CURRENT.get() else { return };
    if sig == tab.interrupt {
        ts.pending.store(1, Ordering::SeqCst);
        ts.cpu_exit();
        return;
    }
    let mut info = [0u8; 128];
    for (i, w) in raw.iter().enumerate() {
        put64(&mut info, 8 * i, *w);
    }
    // Faults the kernel raised are the emulator's own: guest memory is always mapped on the
    // host, and guest faults are found by the softmmu.
    if (get32(&info, 8) as i32) > 0
        && matches!(sig, libc::SIGSEGV | libc::SIGBUS | libc::SIGILL | libc::SIGFPE | libc::SIGTRAP)
    {
        die_from_signal(&info);
    }
    let guest = h2t(sig);
    if !(1..=NSIG).contains(&guest) {
        return;
    }
    let t = host_to_target_siginfo(&info);
    let k = &ts.sigtab[guest as usize - 1];
    for (i, w) in k.info.iter().enumerate() {
        w.store(get64(&t, 8 * i), Ordering::Relaxed);
    }
    k.pending.store(true, Ordering::Release);
    ts.pending.store(1, Ordering::SeqCst);
    if let Some(start) = host::rewind_pc(*pc as u64) {
        *pc = start as i64;
    }
    // Block host signals until the guest handler is entered, but not the faults.
    *mask = !(bit(libc::SIGSEGV) | bit(libc::SIGBUS));
    ts.cpu_exit();
}

/// A `fmt::Write` into a fixed buffer, for messages from the signal handler.
struct StackBuf {
    b: [u8; 160],
    n: usize,
}

impl std::fmt::Write for StackBuf {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let s = s.as_bytes();
        let l = s.len().min(self.b.len() - self.n);
        self.b[self.n..self.n + l].copy_from_slice(&s[..l]);
        self.n += l;
        Ok(())
    }
}

/// `die_from_signal()`: the emulator itself took a fault.
fn die_from_signal(info: &Info) -> ! {
    use std::fmt::Write;
    let sig = get32(info, 0) as i32;
    let code = get32(info, 8) as i32;
    let addr = get64(info, 16);
    let (name, codes): (Option<&str>, &[(i32, &str)]) = match sig {
        libc::SIGSEGV => (Some("SEGV"), &[(1, "MAPERR"), (2, "ACCERR")]),
        libc::SIGBUS => (Some("BUS"), &[(1, "ADRALN"), (2, "ADRERR")]),
        libc::SIGILL => (
            Some("ILL"),
            &[
                (1, "ILLOPC"),
                (2, "ILLOPN"),
                (3, "ILLADR"),
                (5, "PRVOPC"),
                (6, "PRVREG"),
                (7, "COPROC"),
            ],
        ),
        libc::SIGFPE => (Some("FPE"), &[(1, "INTDIV"), (2, "INTOVF")]),
        libc::SIGTRAP => (Some("TRAP"), &[]),
        _ => (None, &[]),
    };
    let mut m = StackBuf { b: [0; 160], n: 0 };
    let _ = write!(m, "{}: QEMU internal SIG", PROG.get().map_or("qemu", String::as_str));
    let _ = match name {
        Some(n) => write!(m, "{n}"),
        None => write!(m, "{sig}"),
    };
    let _ = match codes.iter().find(|c| c.0 == code) {
        Some(c) => write!(m, " {{code={}", c.1),
        None => write!(m, " {{code={code}"),
    };
    let _ =
        if addr == 0 { writeln!(m, ", addr=(nil)}}") } else { writeln!(m, ", addr={addr:#x}}}") };
    host::sys(libc::SYS_write, &[2, m.b.as_ptr() as u64, m.n as u64]);
    host::die_with_signal(sig)
}

/// `block_signals()`: blocks every host signal, and tells whether one is already pending, in
/// which case the caller restarts its system call once it is delivered.
pub(crate) fn block_signals(t: &Task) -> bool {
    host::set_mask(!0);
    t.ts.pending.swap(1, Ordering::SeqCst) != 0
}

/// `do_sigprocmask()` on host masks: `how` is `SIG_BLOCK`, `SIG_UNBLOCK` or `SIG_SETMASK`.
fn do_sigprocmask(t: &mut Task, how: i32, set: Option<u64>) -> i64 {
    let Some(set) = set else { return 0 };
    if block_signals(t) {
        return ERESTARTSYS;
    }
    t.signal_mask = match how {
        libc::SIG_BLOCK => t.signal_mask | set,
        libc::SIG_UNBLOCK => t.signal_mask & !set,
        _ => set,
    };
    t.signal_mask &= !bit(libc::SIGKILL) & !bit(libc::SIGSTOP);
    0
}

/// `set_sigmask()`.
pub(crate) fn set_sigmask(t: &mut Task, set: u64) {
    t.signal_mask = set;
}

/// `force_sig()`.
pub(crate) fn force_sig(t: &mut Task, sig: i32) {
    let mut info = [0u8; 128];
    put32(&mut info, 0, sig as u32);
    put32(&mut info, 8, SI_KERNEL as u32);
    queue_sync(t, info);
}

/// `force_sig_fault()`.
pub(crate) fn force_sig_fault(t: &mut Task, sig: i32, code: i32, addr: u64) {
    let mut info = [0u8; 128];
    put32(&mut info, 0, sig as u32);
    put32(&mut info, 8, code as u32);
    put64(&mut info, 16, addr);
    queue_sync(t, info);
}

/// `force_sigsegv()`: the frame for `oldsig` could not be built.
pub(crate) fn force_sigsegv(t: &mut Task, oldsig: i32) {
    if oldsig == SIGSEGV {
        sigact()[SIGSEGV as usize - 1].handler = SIG_DFL;
    }
    force_sig(t, SIGSEGV);
}

/// `queue_signal()` of a synchronous signal.
fn queue_sync(t: &mut Task, info: Info) {
    t.sync = Some(info);
    t.ts.pending.store(1, Ordering::SeqCst);
}

/// `strsignal()` of a host signal, for the core dump message.
pub(crate) fn strsignal(sig: i32) -> &'static str {
    match sig {
        libc::SIGQUIT => "Quit",
        libc::SIGILL => "Illegal instruction",
        libc::SIGTRAP => "Trace/breakpoint trap",
        libc::SIGABRT => "Aborted",
        libc::SIGBUS => "Bus error",
        libc::SIGFPE => "Floating point exception",
        libc::SIGSEGV => "Segmentation fault",
        _ => "Unknown signal",
    }
}

/// `dump_core_and_abort()`: the guest dies of `sig`. QEMU writes no core file when the core
/// limit is 0, and neither does this, but it prints the same line.
fn dump_core_and_abort(sig: i32) -> ! {
    let host_sig = if sig == SIGABRT { libc::SIGABRT } else { t2h(sig) };
    if core_dump_signal(sig) {
        eprintln!("qemu: uncaught target signal {sig} ({}) - core dumped", strsignal(host_sig));
    }
    host::die_with_signal(host_sig)
}

/// `handle_pending_signal()`.
fn handle_pending_signal(
    space: &GuestSpace,
    t: &mut Task,
    cpu: &mut Cpu<'_>,
    sig: i32,
    info: &Info,
) {
    let sa = sigact()[sig as usize - 1];
    match sa.handler {
        SIG_DFL => {
            if matches!(sig, SIGTSTP | SIGTTIN | SIGTTOU) {
                let pid = host::sys(libc::SYS_getpid, &[]);
                host::sys(libc::SYS_kill, &[pid as u64, libc::SIGSTOP as u64]);
            } else if !matches!(sig, SIGCHLD | SIGURG | SIGWINCH | SIGCONT) {
                dump_core_and_abort(sig);
            }
        }
        SIG_IGN => {}
        SIG_ERR => dump_core_and_abort(sig),
        _ => {
            let mut set = t2h_set(sa.mask);
            if sa.flags & SA_NODEFER == 0 {
                set |= bit(t2h(sig));
            }
            let old = h2t_set(t.signal_mask);
            let blocked = if t.in_sigsuspend { t.sigsuspend_mask } else { t.signal_mask };
            t.signal_mask = blocked | set;
            t.in_sigsuspend = false;
            (crate::guest::guest().setup_rt_frame)(space, t, cpu, sig, &sa, info, old);
            if sa.flags & SA_RESETHAND != 0 {
                sigact()[sig as usize - 1].handler = SIG_DFL;
            }
        }
    }
}

/// `process_pending_signals()`: delivers what is pending and not blocked, then lets host
/// signals in again.
pub(crate) fn process_pending_signals(space: &GuestSpace, t: &mut Task, cpu: &mut Cpu<'_>) {
    while t.ts.pending.load(Ordering::SeqCst) != 0 {
        host::set_mask(!0);
        loop {
            if let Some(info) = t.sync.take() {
                // Synchronous signals are forced: neither blocked nor ignored.
                let sig = get32(&info, 0) as i32;
                let h = t2h(sig);
                {
                    let mut sa = sigact();
                    if is_member(t.signal_mask, h) || sa[sig as usize - 1].handler == SIG_IGN {
                        t.signal_mask &= !bit(h);
                        sa[sig as usize - 1].handler = SIG_DFL;
                    }
                }
                handle_pending_signal(space, t, cpu, sig, &info);
                continue;
            }
            let blocked = if t.in_sigsuspend { t.sigsuspend_mask } else { t.signal_mask };
            let next = (1..=NSIG).find(|&s| {
                t.ts.sigtab[s as usize - 1].pending.load(Ordering::Acquire)
                    && !is_member(blocked, t2h(s))
            });
            let Some(sig) = next else { break };
            let k = &t.ts.sigtab[sig as usize - 1];
            k.pending.store(false, Ordering::Relaxed);
            let mut info = [0u8; 128];
            for (i, w) in k.info.iter().enumerate() {
                put64(&mut info, 8 * i, w.load(Ordering::Relaxed));
            }
            handle_pending_signal(space, t, cpu, sig, &info);
        }
        // Nothing left: unblock, which may take another host signal and set pending again.
        t.ts.pending.store(0, Ordering::SeqCst);
        t.in_sigsuspend = false;
        host::set_mask(t.run_mask());
    }
    t.in_sigsuspend = false;
}

/// `on_sig_stack()`.
pub(crate) fn on_sig_stack(t: &Task, sp: u64) -> bool {
    sp.wrapping_sub(t.altstack.0) < t.altstack.1
}

/// `sas_ss_flags()`.
fn sas_ss_flags(t: &Task, sp: u64) -> i32 {
    if t.altstack.1 == 0 {
        SS_DISABLE
    } else if on_sig_stack(t, sp) {
        SS_ONSTACK
    } else {
        0
    }
}

/// `target_sigsp()`: where a frame for a handler of `sa` goes, from the stack pointer `sp`.
pub(crate) fn target_sigsp(t: &Task, sp: u64, sa: &Sigaction) -> u64 {
    if sa.flags & SA_ONSTACK != 0 && sas_ss_flags(t, sp) == 0 {
        return t.altstack.0.wrapping_add(t.altstack.1);
    }
    sp
}

/// `target_save_altstack()`: the guest `stack_t` at `sp`.
pub(crate) fn save_altstack(t: &Task, sp: u64) -> [u8; 24] {
    let mut b = [0u8; 24];
    put64(&mut b, 0, t.altstack.0);
    put32(&mut b, 8, sas_ss_flags(t, sp) as u32);
    put64(&mut b, 16, t.altstack.1);
    b
}

/// `target_restore_altstack()` from the guest `stack_t` `uss`.
pub(crate) fn restore_altstack(t: &mut Task, uss: &[u8], sp: u64) -> i64 {
    let (ss_sp, flags, mut size) = (get64(uss, 0), get32(uss, 8) as i32, get64(uss, 16));
    if on_sig_stack(t, sp) {
        return -i64::from(libc::EPERM);
    }
    let ss_sp = match flags {
        SS_DISABLE => {
            size = 0;
            0
        }
        SS_ONSTACK | 0 => {
            if size < crate::guest::guest().minsigstksz {
                return -i64::from(libc::ENOMEM);
            }
            ss_sp
        }
        _ => return -EINVAL,
    };
    t.altstack = (ss_sp, size);
    0
}

/// `do_sigaltstack()`.
fn do_sigaltstack(space: &GuestSpace, t: &mut Task, uss: u64, uoss: u64, sp: u64) -> i64 {
    if uoss != 0 && !space.check(uoss, 24, ruvm_user_common::page::WRITE) {
        return -EFAULT;
    }
    let old = save_altstack(t, sp);
    if uss != 0 {
        let mut b = [0u8; 24];
        if !space.read(uss, &mut b) {
            return -EFAULT;
        }
        let r = restore_altstack(t, &b, sp);
        if r != 0 {
            return r;
        }
    }
    if uoss != 0 && !space.write(uoss, &old) {
        return -EFAULT;
    }
    0
}

/// `do_sigaction()` for `rt_sigaction(sig, act, oact, sigsetsize)`.
fn do_sigaction(space: &GuestSpace, t: &Task, a: [u64; 6]) -> i64 {
    let sig = a[0] as i32;
    if a[3] != 8 {
        return -EINVAL;
    }
    // handler, flags, restorer when the target has it, and mask.
    let restorer = crate::guest::guest().sa_restorer;
    let size = if restorer { 32 } else { 24 };
    let mut act = [0u8; 32];
    if a[1] != 0 && !space.read(a[1], &mut act[..size]) {
        return -EFAULT;
    }
    if a[2] != 0 && !space.check(a[2], size as u64, ruvm_user_common::page::WRITE) {
        return -EFAULT;
    }
    if !(1..=NSIG).contains(&sig) {
        return -EINVAL;
    }
    if a[1] != 0 && (sig == SIGKILL || sig == SIGSTOP) {
        return -EINVAL;
    }
    if block_signals(t) {
        return ERESTARTSYS;
    }
    let mut tab = sigact();
    let k = &mut tab[sig as usize - 1];
    if a[2] != 0 {
        let mut b = [0u8; 32];
        put64(&mut b, 0, k.handler);
        put64(&mut b, 8, k.flags);
        if restorer {
            put64(&mut b, 16, k.restorer);
        }
        put64(&mut b, size - 8, k.mask);
        if !space.write(a[2], &b[..size]) {
            return -EFAULT;
        }
    }
    if a[1] == 0 {
        return 0;
    }
    *k = Sigaction {
        handler: get64(&act, 0),
        flags: get64(&act, 8),
        restorer: if restorer { get64(&act, 16) } else { 0 },
        mask: get64(&act, size - 8),
    };
    let host_sig = t2h(sig);
    if host_sig > NSIG {
        // Not enough host signals for every target one; programs that register handlers for
        // all of them must not fail.
        return 0;
    }
    if host_sig == libc::SIGSEGV || host_sig == libc::SIGBUS {
        return 0;
    }
    match k.handler {
        SIG_IGN => host::set_disposition(host_sig, SIG_IGN),
        SIG_DFL if !core_dump_signal(sig) => host::set_disposition(host_sig, SIG_DFL),
        SIG_DFL => host::install_handler(host_sig, false),
        _ => host::install_handler(host_sig, k.flags & SA_RESTART != 0),
    }
}

/// `process_sigsuspend_mask()`: the host form of the guest mask at `set`, `size` bytes.
fn sigsuspend_mask(space: &GuestSpace, t: &mut Task, set: u64, size: u64) -> Result<(), i64> {
    if size != 8 {
        return Err(-EINVAL);
    }
    let mut b = [0u8; 8];
    if !space.read(set, &mut b) {
        return Err(-EFAULT);
    }
    t.sigsuspend_mask = t2h_set(u64::from_le_bytes(b));
    Ok(())
}

/// `finish_sigsuspend_mask()`.
fn finish_sigsuspend(t: &mut Task, ret: i64) {
    if ret != ERESTARTSYS {
        t.in_sigsuspend = true;
    }
}

/// A guest call that sleeps with a temporary signal mask, `ppoll()`, `pselect6()` and
/// `epoll_pwait()`: `call` gets the host address of the host mask, or 0 when `set` is 0.
pub(crate) fn with_sigmask(
    space: &GuestSpace,
    t: &mut Task,
    set: u64,
    size: u64,
    call: impl FnOnce(u64) -> i64,
) -> i64 {
    if set == 0 {
        return call(0);
    }
    if let Err(e) = sigsuspend_mask(space, t, set, size) {
        return e;
    }
    let mask = t.sigsuspend_mask;
    let r = call((&raw const mask) as u64);
    finish_sigsuspend(t, r);
    r
}

fn read_u64(space: &GuestSpace, addr: u64) -> Result<u64, i64> {
    let mut b = [0u8; 8];
    if space.read(addr, &mut b) { Ok(u64::from_le_bytes(b)) } else { Err(-EFAULT) }
}

/// `host_to_target_waitstatus()`.
pub(crate) fn host_to_target_waitstatus(st: i32) -> i32 {
    if st & 0x7f != 0 && st & 0x7f != 0x7f {
        return h2t(st & 0x7f) | (st & !0x7f);
    }
    if st & 0xff == 0x7f {
        return (h2t((st >> 8) & 0xff) << 8) | (st & 0xff);
    }
    st
}

/// The signal system calls, `None` for any other.
pub(crate) fn do_signal_syscall(
    space: &GuestSpace,
    t: &mut Task,
    sp: u64,
    n: i64,
    a: [u64; 6],
) -> Option<i64> {
    let r = match n {
        libc::SYS_rt_sigaction => do_sigaction(space, t, a),
        libc::SYS_rt_sigprocmask => {
            if a[3] != 8 {
                return Some(-EINVAL);
            }
            let (how, set) = if a[1] != 0 {
                let set = match read_u64(space, a[1]) {
                    Ok(s) => t2h_set(s),
                    Err(e) => return Some(e),
                };
                let how = a[0] as i32;
                if !matches!(how, libc::SIG_BLOCK | libc::SIG_UNBLOCK | libc::SIG_SETMASK) {
                    return Some(-EINVAL);
                }
                (how, Some(set))
            } else {
                (0, None)
            };
            let old = t.signal_mask;
            let r = do_sigprocmask(t, how, set);
            if r == 0 && a[2] != 0 && !space.write(a[2], &h2t_set(old).to_le_bytes()) {
                return Some(-EFAULT);
            }
            r
        }
        libc::SYS_rt_sigpending => {
            if a[1] > 8 {
                return Some(-EINVAL);
            }
            let mut set = 0u64;
            let r = host::sys(libc::SYS_rt_sigpending, &[(&raw mut set) as u64, 8]);
            if r < 0 {
                return Some(r);
            }
            if !space.write(a[0], &h2t_set(set).to_le_bytes()) {
                return Some(-EFAULT);
            }
            r
        }
        libc::SYS_rt_sigsuspend => {
            if let Err(e) = sigsuspend_mask(space, t, a[0], a[1]) {
                return Some(e);
            }
            let mask = t.sigsuspend_mask;
            let r = guest_sys(libc::SYS_rt_sigsuspend, &[(&raw const mask) as u64, 8]);
            finish_sigsuspend(t, r);
            r
        }
        libc::SYS_pause => {
            if !block_signals(t) {
                let mask = t.signal_mask;
                host::sys(libc::SYS_rt_sigsuspend, &[(&raw const mask) as u64, 8]);
            }
            -i64::from(libc::EINTR)
        }
        libc::SYS_rt_sigtimedwait => {
            if a[3] != 8 {
                return Some(-EINVAL);
            }
            let set = match read_u64(space, a[0]) {
                Ok(s) => t2h_set(s),
                Err(e) => return Some(e),
            };
            let mut ts = [0u8; 16];
            if a[2] != 0 && !space.read(a[2], &mut ts) {
                return Some(-EFAULT);
            }
            let mut hinfo = [0u64; 16];
            let tsp = if a[2] != 0 { ts.as_ptr() as u64 } else { 0 };
            let r = guest_sys(
                libc::SYS_rt_sigtimedwait,
                &[(&raw const set) as u64, hinfo.as_mut_ptr() as u64, tsp, 8],
            );
            if r < 0 {
                return Some(r);
            }
            if a[1] != 0 {
                let mut b = [0u8; 128];
                for (i, w) in hinfo.iter().enumerate() {
                    put64(&mut b, 8 * i, *w);
                }
                if !space.write(a[1], &host_to_target_siginfo(&b)) {
                    return Some(-EFAULT);
                }
            }
            i64::from(h2t(r as i32))
        }
        libc::SYS_rt_sigqueueinfo | libc::SYS_rt_tgsigqueueinfo => {
            let at = if n == libc::SYS_rt_sigqueueinfo { 2 } else { 3 };
            let mut b = [0u8; 128];
            if !space.read(a[at], &mut b) {
                return Some(-EFAULT);
            }
            let h = target_to_host_siginfo(&b);
            let mut args = a;
            args[at - 1] = t2h(a[at - 1] as i32) as u64;
            args[at] = h.as_ptr() as u64;
            guest_syscall(n, args)
        }
        libc::SYS_kill | libc::SYS_tkill => guest_sys(n, &[a[0], t2h(a[1] as i32) as i64 as u64]),
        libc::SYS_tgkill => guest_sys(n, &[a[0], a[1], t2h(a[2] as i32) as i64 as u64]),
        libc::SYS_signalfd | libc::SYS_signalfd4 => {
            let flags = if n == libc::SYS_signalfd { 0 } else { a[3] };
            if flags & !(libc::O_NONBLOCK as u64 | libc::O_CLOEXEC as u64) != 0 {
                return Some(-EINVAL);
            }
            let set = match read_u64(space, a[1]) {
                Ok(s) => t2h_set(s),
                Err(e) => return Some(e),
            };
            guest_sys(libc::SYS_signalfd4, &[a[0], (&raw const set) as u64, 8, flags])
        }
        libc::SYS_sigaltstack => do_sigaltstack(space, t, a[0], a[1], sp),
        _ => return None,
    };
    Some(r)
}
