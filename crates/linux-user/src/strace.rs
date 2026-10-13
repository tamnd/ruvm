// SPDX-License-Identifier: GPL-2.0-or-later

//! `-strace`: `print_syscall()`, `print_syscall_ret()` and `print_taken_signal()` from
//! `linux-user/strace.c`, with the same printers and formats, so the log reads as QEMU's.
//!
//! The table is `strace.list`, by the host's (x86-64) number of each system call; the targets
//! that number their calls differently look an entry up by the number they translate to.
//! Each part of a line goes out in one `write()`, as QEMU's `fprintf()`s do on an unbuffered
//! `stderr`, so the call shows before it runs and its result after.
//!
//! The printers are those of a 64-bit little-endian target. Where `strace.list` gives a bare
//! format, QEMU hands guest addresses to the host's `printf()`: `%s` reads the string at the
//! guest address, which this does through the guest space and prints `(null)` for 0. Of the
//! ioctls only the terminal and `FIO*` ones have names here; QEMU knows about 700, and prints
//! the others as numbers too.

mod table;

use std::fs::File;
use std::io::Write as _;
use std::os::fd::AsRawFd;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI32, Ordering};

use ruvm_user_common::{GuestSpace, page};

use crate::host;
use crate::signal::{Info, get32, get64};

/// A printer of a call, with its arguments.
type Call = fn(&mut Out<'_>, &Entry, &[u64; 6]);
/// A printer of a result, with the call's arguments.
type Ret = fn(&mut Out<'_>, &Entry, i64, &[u64; 6]);

/// One line of `strace.list`, `struct syscallname`.
pub(crate) struct Entry {
    name: &'static str,
    format: Option<&'static str>,
    call: Option<Call>,
    result: Option<Ret>,
}

/// Where the log goes, -1 when `-strace` is off.
static FD: AtomicI32 = AtomicI32::new(-1);

/// The `-D` file, kept open.
static FILE: OnceLock<File> = OnceLock::new();

/// `qemu_set_log_filename_flags()` for `-strace` and `-D`: checks the file name, which may
/// hold one `%d` for the process ID, and when `on` opens it, or logs to `stderr` without one.
pub(crate) fn start(on: bool, file: Option<&str>) -> Result<(), String> {
    let name = match file {
        Some(t) => match t.find('%') {
            Some(i) if !t[i + 1..].starts_with('d') || t[i + 2..].contains('%') => {
                return Err(format!("Bad logfile template: {t}"));
            }
            Some(i) => Some(format!("{}{}{}", &t[..i], std::process::id(), &t[i + 2..])),
            None => Some(t.to_string()),
        },
        None => None,
    };
    if !on {
        return Ok(());
    }
    let fd = match name {
        Some(name) => match File::create(&name) {
            Ok(f) => FILE.get_or_init(|| f).as_raw_fd(),
            Err(e) => {
                return Err(format!("Error opening logfile {name}: {}", crate::strerror_of(&e)));
            }
        },
        None => 2,
    };
    FD.store(fd, Ordering::Relaxed);
    Ok(())
}

/// `qemu_loglevel_mask(LOG_STRACE)`.
pub(crate) fn on() -> bool {
    FD.load(Ordering::Relaxed) >= 0
}

/// The entry of the host's system call `nr`.
pub(crate) fn entry(nr: i64) -> Option<&'static Entry> {
    let i = table::ENTRIES.binary_search_by_key(&nr, |(n, _)| i64::from(*n)).ok()?;
    Some(&table::ENTRIES[i].1)
}

/// The entry of `riscv_hwprobe`, which the host does not have.
pub(crate) fn riscv_hwprobe() -> &'static Entry {
    &table::RISCV_HWPROBE
}

fn emit(mut s: &[u8]) {
    let fd = FD.load(Ordering::Relaxed);
    while !s.is_empty() {
        let n = host::sys(libc::SYS_write, &[fd as u64, s.as_ptr() as u64, s.len() as u64]);
        if n == -i64::from(libc::EINTR) {
            continue;
        }
        if n <= 0 {
            return;
        }
        s = &s[n as usize..];
    }
}

/// `print_syscall()`: the thread, then the call to the target's system call `nr`, whose
/// entry is `e`. Gives `e` back for [`syscall_ret`].
pub(crate) fn syscall(
    space: &GuestSpace,
    nr: u64,
    e: Option<&'static Entry>,
    args: &[u64; 6],
) -> Option<&'static Entry> {
    let mut o = Out { s: Vec::new(), space };
    let tid = host::sys(libc::SYS_gettid, &[]);
    let _ = write!(o.s, "{tid} ");
    match e {
        Some(e) => match e.call {
            Some(call) => call(&mut o, e, args),
            None => {
                let f = e.format.unwrap_or("%s(%ld,%ld,%ld,%ld,%ld,%ld)");
                o.format(f, e.name, args);
            }
        },
        None => {
            let _ = writeln!(o.s, "Unknown syscall {}", nr as i32);
        }
    }
    emit(&o.s);
    e
}

/// `print_syscall_ret()`: what the call of `e` returned, unless the thread is gone.
pub(crate) fn syscall_ret(
    space: &GuestSpace,
    e: Option<&'static Entry>,
    ret: i64,
    args: &[u64; 6],
) {
    let Some(e) = e else { return };
    if ret == crate::syscall::THREAD_EXIT {
        return;
    }
    let mut o = Out { s: Vec::new(), space };
    match e.result {
        Some(result) => result(&mut o, e, ret, args),
        None => {
            if !o.err(ret) {
                o.ld(ret);
            }
            o.str("\n");
        }
    }
    emit(&o.s);
}

/// `print_taken_signal()`: the guest takes `sig` with `info`; `fault` is QEMU's
/// `QEMU_SI_FAULT`, a signal the CPU raised.
pub(crate) fn taken_signal(space: &GuestSpace, sig: i32, info: &Info, fault: bool) {
    let mut o = Out { s: Vec::new(), space };
    o.str("--- ");
    o.signal(sig as u64, true);
    o.str(" ");
    o.siginfo(info, fault);
    o.str(" ---\n");
    emit(&o.s);
}

/// One `flags` row: printed when `v & mask == value`.
struct Flag(&'static str, u64, u64);

const fn f(name: &'static str, v: u64) -> Flag {
    Flag(name, v, v)
}

const fn fm(name: &'static str, v: u64, mask: u64) -> Flag {
    Flag(name, v, mask)
}

const ACCESS: &[Flag] = &[fm("F_OK", 0, 7), f("R_OK", 4), f("W_OK", 2), f("X_OK", 1)];
const AT_FILE: &[Flag] = &[f("AT_EACCESS", 0x200), f("AT_SYMLINK_NOFOLLOW", 0x100)];
const UNLINKAT: &[Flag] = &[f("AT_REMOVEDIR", 0x200)];
const EXECVEAT: &[Flag] = &[f("AT_EMPTY_PATH", 0x1000), f("AT_SYMLINK_NOFOLLOW", 0x100)];
const MODE: &[Flag] = &[
    f("S_IFSOCK", 0o140000),
    f("S_IFLNK", 0o120000),
    f("S_IFREG", 0o100000),
    f("S_IFBLK", 0o60000),
    f("S_IFDIR", 0o40000),
    f("S_IFCHR", 0o20000),
    f("S_IFIFO", 0o10000),
];
const OPEN_ACCESS: &[Flag] = &[fm("O_RDONLY", 0, 3), fm("O_WRONLY", 1, 3), fm("O_RDWR", 2, 3)];
const OPEN: &[Flag] = &[
    f("O_APPEND", 0o2000),
    f("O_CREAT", 0o100),
    f("O_DIRECTORY", 0o200000),
    f("O_EXCL", 0o200),
    f("O_LARGEFILE", 0o100000),
    f("O_NOCTTY", 0o400),
    f("O_NOFOLLOW", 0o400000),
    f("O_NONBLOCK", 0o4000),
    f("O_DSYNC", 0o10000),
    f("__O_SYNC", 0o4000000),
    f("O_TRUNC", 0o1000),
    f("O_DIRECT", 0o40000),
    f("O_NOATIME", 0o1000000),
    f("O_CLOEXEC", 0o2000000),
    f("O_PATH", 0o10000000),
    f("O_TMPFILE", 0o20000000 | 0o200000),
    f("__O_TMPFILE", 0o20000000),
];
/// aarch64 moves four of them.
const OPEN_AARCH64: &[Flag] = &[
    f("O_APPEND", 0o2000),
    f("O_CREAT", 0o100),
    f("O_DIRECTORY", 0o40000),
    f("O_EXCL", 0o200),
    f("O_LARGEFILE", 0o400000),
    f("O_NOCTTY", 0o400),
    f("O_NOFOLLOW", 0o100000),
    f("O_NONBLOCK", 0o4000),
    f("O_DSYNC", 0o10000),
    f("__O_SYNC", 0o4000000),
    f("O_TRUNC", 0o1000),
    f("O_DIRECT", 0o200000),
    f("O_NOATIME", 0o1000000),
    f("O_CLOEXEC", 0o2000000),
    f("O_PATH", 0o10000000),
    f("O_TMPFILE", 0o20000000 | 0o40000),
    f("__O_TMPFILE", 0o20000000),
];
const OPENAT2_RESOLVE: &[Flag] = &[
    f("RESOLVE_NO_XDEV", 1),
    f("RESOLVE_NO_MAGICLINKS", 2),
    f("RESOLVE_NO_SYMLINKS", 4),
    f("RESOLVE_BENEATH", 8),
    f("RESOLVE_IN_ROOT", 0x10),
    f("RESOLVE_CACHED", 0x20),
];
const MOUNT: &[Flag] = &[
    f("MS_BIND", 4096),
    f("MS_DIRSYNC", 128),
    f("MS_MANDLOCK", 64),
    f("MS_MOVE", 8192),
    f("MS_NOATIME", 1024),
    f("MS_NODEV", 4),
    f("MS_NODIRATIME", 2048),
    f("MS_NOEXEC", 8),
    f("MS_NOSUID", 2),
    f("MS_RDONLY", 1),
    f("MS_RELATIME", 1 << 21),
    f("MS_REMOUNT", 32),
    f("MS_SYNCHRONOUS", 16),
];
const UMOUNT2: &[Flag] = &[f("MNT_FORCE", 1), f("MNT_DETACH", 2), f("MNT_EXPIRE", 4)];
const PROT: &[Flag] = &[
    fm("PROT_NONE", 0, 7),
    f("PROT_EXEC", 4),
    f("PROT_READ", 1),
    f("PROT_WRITE", 2),
    f("PROT_SEM", 8),
    f("PROT_GROWSDOWN", 0x100_0000),
    f("PROT_GROWSUP", 0x200_0000),
];
const MMAP: &[Flag] = &[
    fm("MAP_SHARED", 1, 0xf),
    fm("MAP_PRIVATE", 2, 0xf),
    fm("MAP_SHARED_VALIDATE", 3, 0xf),
    f("MAP_ANONYMOUS", 0x20),
    f("MAP_DENYWRITE", 0x800),
    f("MAP_EXECUTABLE", 0x1000),
    f("MAP_FIXED", 0x10),
    f("MAP_FIXED_NOREPLACE", 0x10_0000),
    f("MAP_GROWSDOWN", 0x100),
    f("MAP_HUGETLB", 0x4_0000),
    f("MAP_LOCKED", 0x2000),
    f("MAP_NONBLOCK", 0x1_0000),
    f("MAP_NORESERVE", 0x4000),
    f("MAP_POPULATE", 0x8000),
    f("MAP_STACK", 0x2_0000),
    f("MAP_SYNC", 0x8_0000),
    f("MAP_UNINITIALIZED", 0x400_0000),
];
const CLONE: &[Flag] = &[
    f("CLONE_VM", 0x100),
    f("CLONE_FS", 0x200),
    f("CLONE_FILES", 0x400),
    f("CLONE_SIGHAND", 0x800),
    f("CLONE_PIDFD", 0x1000),
    f("CLONE_PTRACE", 0x2000),
    f("CLONE_VFORK", 0x4000),
    f("CLONE_PARENT", 0x8000),
    f("CLONE_THREAD", 0x1_0000),
    f("CLONE_NEWNS", 0x2_0000),
    f("CLONE_SYSVSEM", 0x4_0000),
    f("CLONE_SETTLS", 0x8_0000),
    f("CLONE_PARENT_SETTID", 0x10_0000),
    f("CLONE_CHILD_CLEARTID", 0x20_0000),
    f("CLONE_DETACHED", 0x40_0000),
    f("CLONE_UNTRACED", 0x80_0000),
    f("CLONE_CHILD_SETTID", 0x100_0000),
    f("CLONE_NEWUTS", 0x400_0000),
    f("CLONE_NEWIPC", 0x800_0000),
    f("CLONE_NEWUSER", 0x1000_0000),
    f("CLONE_NEWPID", 0x2000_0000),
    f("CLONE_NEWNET", 0x4000_0000),
    f("CLONE_NEWCGROUP", 0x200_0000),
    f("CLONE_NEWTIME", 0x80),
    f("CLONE_IO", 0x8000_0000),
];
const MSG: &[Flag] = &[
    f("MSG_CONFIRM", 0x800),
    f("MSG_DONTROUTE", 4),
    f("MSG_DONTWAIT", 0x40),
    f("MSG_EOR", 0x80),
    f("MSG_MORE", 0x8000),
    f("MSG_NOSIGNAL", 0x4000),
    f("MSG_OOB", 1),
    f("MSG_CMSG_CLOEXEC", 0x4000_0000),
    f("MSG_ERRQUEUE", 0x2000),
    f("MSG_PEEK", 2),
    f("MSG_TRUNC", 0x20),
    f("MSG_WAITALL", 0x100),
    f("MSG_CTRUNC", 8),
];
const STATX_FLAGS: &[Flag] = &[
    f("AT_EMPTY_PATH", 0x1000),
    f("AT_NO_AUTOMOUNT", 0x800),
    f("AT_SYMLINK_NOFOLLOW", 0x100),
    fm("AT_STATX_SYNC_AS_STAT", 0, 0x6000),
    fm("AT_STATX_FORCE_SYNC", 0x2000, 0x6000),
    fm("AT_STATX_DONT_SYNC", 0x4000, 0x6000),
];
const STATX_MASK: &[Flag] = &[
    f("STATX_ALL", 0xfff),
    f("STATX_BASIC_STATS", 0x7ff),
    f("STATX_TYPE", 1),
    f("STATX_MODE", 2),
    f("STATX_NLINK", 4),
    f("STATX_UID", 8),
    f("STATX_GID", 0x10),
    f("STATX_ATIME", 0x20),
    f("STATX_MTIME", 0x40),
    f("STATX_CTIME", 0x80),
    f("STATX_INO", 0x100),
    f("STATX_SIZE", 0x200),
    f("STATX_BLOCKS", 0x400),
    f("STATX_BTIME", 0x800),
];
const FALLOC: &[Flag] = &[
    f("FALLOC_FL_KEEP_SIZE", 1),
    f("FALLOC_FL_PUNCH_HOLE", 2),
    f("FALLOC_FL_NO_HIDE_STALE", 4),
    f("FALLOC_FL_COLLAPSE_RANGE", 8),
    f("FALLOC_FL_ZERO_RANGE", 0x10),
    f("FALLOC_FL_INSERT_RANGE", 0x20),
    f("FALLOC_FL_UNSHARE_RANGE", 0x40),
];
const MLOCKALL: &[Flag] = &[f("MCL_CURRENT", 1), f("MCL_FUTURE", 2), f("MCL_ONFAULT", 4)];
const SHMAT: &[Flag] = &[
    f("SHM_RND", 0o20000),
    f("SHM_REMAP", 0o40000),
    f("SHM_RDONLY", 0o10000),
    f("SHM_EXEC", 0o100000),
];

const TERMIOS_I: &[Flag] = &[
    f("IGNBRK", 0o1),
    f("BRKINT", 0o2),
    f("IGNPAR", 0o4),
    f("PARMRK", 0o10),
    f("INPCK", 0o20),
    f("ISTRIP", 0o40),
    f("INLCR", 0o100),
    f("IGNCR", 0o200),
    f("ICRNL", 0o400),
    f("IUCLC", 0o1000),
    f("IXON", 0o2000),
    f("IXANY", 0o4000),
    f("IXOFF", 0o10000),
    f("IMAXBEL", 0o20000),
    f("IUTF8", 0o40000),
];
const TERMIOS_O: &[Flag] = &[
    f("OPOST", 0o1),
    f("OLCUC", 0o2),
    f("ONLCR", 0o4),
    f("OCRNL", 0o10),
    f("ONOCR", 0o20),
    f("ONLRET", 0o40),
    f("OFILL", 0o100),
    f("OFDEL", 0o200),
];
const TERMIOS_C: &[Flag] = &[
    f("CSTOPB", 0o100),
    f("CREAD", 0o200),
    f("PARENB", 0o400),
    f("PARODD", 0o1000),
    f("HUPCL", 0o2000),
    f("CLOCAL", 0o4000),
    f("CRTSCTS", 0o20000000000),
];
const TERMIOS_L: &[Flag] = &[
    f("ISIG", 0o1),
    f("ICANON", 0o2),
    f("XCASE", 0o4),
    f("ECHO", 0o10),
    f("ECHOE", 0o20),
    f("ECHOK", 0o40),
    f("ECHONL", 0o100),
    f("NOFLSH", 0o200),
    f("TOSTOP", 0o400),
    f("ECHOCTL", 0o1000),
    f("ECHOPRT", 0o2000),
    f("ECHOKE", 0o4000),
    f("FLUSHO", 0o10000),
    f("PENDIN", 0o40000),
    f("IEXTEN", 0o100000),
    f("EXTPROC", 0o200000),
];

/// The delay fields of `c_oflag`, each with its values, in the order QEMU prints them.
const TERMIOS_DLY: &[(u32, &[(&str, u64)])] = &[
    (0o400, &[("NL0", 0), ("NL1", 0o400)]),
    (0o3000, &[("CR0", 0), ("CR1", 0o1000), ("CR2", 0o2000), ("CR3", 0o3000)]),
    (0o14000, &[("TAB0", 0), ("TAB1", 0o4000), ("TAB2", 0o10000), ("TAB3", 0o14000)]),
    (0o20000, &[("BS0", 0), ("BS1", 0o20000)]),
    (0o40000, &[("VT0", 0), ("VT1", 0o40000)]),
    (0o100000, &[("FF0", 0), ("FF1", 0o100000)]),
];
const CBAUD: &[(&str, u64)] = &[
    ("B0", 0),
    ("B50", 1),
    ("B75", 2),
    ("B110", 3),
    ("B134", 4),
    ("B150", 5),
    ("B200", 6),
    ("B300", 7),
    ("B600", 0o10),
    ("B1200", 0o11),
    ("B1800", 0o12),
    ("B2400", 0o13),
    ("B4800", 0o14),
    ("B9600", 0o15),
    ("B19200", 0o16),
    ("B38400", 0o17),
    ("B57600", 0o10001),
    ("B115200", 0o10002),
    ("B230400", 0o10003),
    ("B460800", 0o10004),
];
const CSIZE: &[(&str, u64)] = &[("CS5", 0), ("CS6", 0o20), ("CS7", 0o40), ("CS8", 0o60)];

const CLOCKIDS: &[(&str, u64)] = &[
    ("CLOCK_REALTIME", 0),
    ("CLOCK_MONOTONIC", 1),
    ("CLOCK_PROCESS_CPUTIME_ID", 2),
    ("CLOCK_THREAD_CPUTIME_ID", 3),
    ("CLOCK_MONOTONIC_RAW", 4),
    ("CLOCK_REALTIME_COARSE", 5),
    ("CLOCK_MONOTONIC_COARSE", 6),
    ("CLOCK_BOOTTIME", 7),
    ("CLOCK_REALTIME_ALARM", 8),
    ("CLOCK_BOOTTIME_ALARM", 9),
    ("CLOCK_SGI_CYCLE", 10),
    ("CLOCK_TAI", 11),
];
const ITIMERS: &[(&str, u64)] = &[("ITIMER_REAL", 0), ("ITIMER_VIRTUAL", 1), ("ITIMER_PROF", 2)];
const MADVISE: &[(&str, u64)] = &[
    ("MADV_NORMAL", 0),
    ("MADV_RANDOM", 1),
    ("MADV_SEQUENTIAL", 2),
    ("MADV_WILLNEED", 3),
    ("MADV_DONTNEED", 4),
    ("MADV_FREE", 8),
    ("MADV_REMOVE", 9),
    ("MADV_DONTFORK", 10),
    ("MADV_DOFORK", 11),
    ("MADV_MERGEABLE", 12),
    ("MADV_UNMERGEABLE", 13),
    ("MADV_HUGEPAGE", 14),
    ("MADV_NOHUGEPAGE", 15),
    ("MADV_DONTDUMP", 16),
    ("MADV_DODUMP", 17),
    ("MADV_WIPEONFORK", 18),
    ("MADV_KEEPONFORK", 19),
    ("MADV_COLD", 20),
    ("MADV_PAGEOUT", 21),
    ("MADV_POPULATE_READ", 22),
    ("MADV_POPULATE_WRITE", 23),
    ("MADV_DONTNEED_LOCKED", 24),
];
const SYSLOG: &[&str] = &[
    "SYSLOG_ACTION_CLOSE",
    "SYSLOG_ACTION_OPEN",
    "SYSLOG_ACTION_READ",
    "SYSLOG_ACTION_READ_ALL",
    "SYSLOG_ACTION_READ_CLEAR",
    "SYSLOG_ACTION_CLEAR",
    "SYSLOG_ACTION_CONSOLE_OFF",
    "SYSLOG_ACTION_CONSOLE_ON",
    "SYSLOG_ACTION_CONSOLE_LEVEL",
    "SYSLOG_ACTION_SIZE_UNREAD",
    "SYSLOG_ACTION_SIZE_BUFFER",
];
const RLIMITS: &[&str] = &[
    "RLIMIT_CPU",
    "RLIMIT_FSIZE",
    "RLIMIT_DATA",
    "RLIMIT_STACK",
    "RLIMIT_CORE",
    "RLIMIT_RSS",
    "RLIMIT_NPROC",
    "RLIMIT_NOFILE",
    "RLIMIT_MEMLOCK",
    "RLIMIT_AS",
    "RLIMIT_LOCKS",
    "RLIMIT_SIGPENDING",
    "RLIMIT_MSGQUEUE",
    "RLIMIT_NICE",
    "RLIMIT_RTPRIO",
    "RLIMIT_RTTIME",
];
/// `target_signal_name[]`, where `SIGIOT` comes after `SIGABRT` and so replaces it.
const SIGNALS: [Option<&str>; 32] = [
    None,
    Some("SIGHUP"),
    Some("SIGINT"),
    Some("SIGQUIT"),
    Some("SIGILL"),
    Some("SIGTRAP"),
    Some("SIGIOT"),
    Some("SIGBUS"),
    Some("SIGFPE"),
    Some("SIGKILL"),
    Some("SIGUSR1"),
    Some("SIGSEGV"),
    Some("SIGUSR2"),
    Some("SIGPIPE"),
    Some("SIGALRM"),
    Some("SIGTERM"),
    Some("SIGSTKFLT"),
    Some("SIGCHLD"),
    Some("SIGCONT"),
    Some("SIGSTOP"),
    Some("SIGTSTP"),
    Some("SIGTTIN"),
    Some("SIGTTOU"),
    Some("SIGURG"),
    Some("SIGXCPU"),
    Some("SIGXFSZ"),
    Some("SIGVTALRM"),
    Some("SIGPROF"),
    Some("SIGWINCH"),
    Some("SIGIO"),
    Some("SIGPWR"),
    Some("SIGSYS"),
];
const FUTEX: &[&str] = &[
    "FUTEX_WAIT",
    "FUTEX_WAKE",
    "FUTEX_FD",
    "FUTEX_REQUEUE",
    "FUTEX_CMP_REQUEUE",
    "FUTEX_WAKE_OP",
    "FUTEX_LOCK_PI",
    "FUTEX_UNLOCK_PI",
    "FUTEX_TRYLOCK_PI",
    "FUTEX_WAIT_BITSET",
    "FUTEX_WAKE_BITSET",
    "FUTEX_WAIT_REQUEUE_PI",
    "FUTEX_CMP_REQUEUE_PI",
    "FUTEX_LOCK_PI2",
];

/// What an ioctl's argument is, `arg_type` of an `IOCTLEntry`.
#[derive(Clone, Copy, PartialEq)]
enum Arg {
    Null,
    Int,
    PtrInt,
    Termios,
    Termios2,
    Winsize,
}

/// `IOC_R`, `IOC_W` and `IOC_RW`.
const R: u8 = 1;
const W: u8 = 2;

/// The terminal and `FIO*` lines of `ioctls.h`, in its order.
const IOCTLS: &[(u32, &str, u8, Arg)] = &[
    (0x802c_542a, "TCGETS2", R, Arg::Termios2),
    (0x402c_542b, "TCSETS2", W, Arg::Termios2),
    (0x402c_542c, "TCSETSW2", W, Arg::Termios2),
    (0x402c_542d, "TCSETSF2", W, Arg::Termios2),
    (0x5401, "TCGETS", R, Arg::Termios),
    (0x5402, "TCSETS", W, Arg::Termios),
    (0x5404, "TCSETSF", W, Arg::Termios),
    (0x5403, "TCSETSW", W, Arg::Termios),
    (0x5413, "TIOCGWINSZ", R, Arg::Winsize),
    (0x5414, "TIOCSWINSZ", W, Arg::Winsize),
    (0x541b, "FIONREAD", R, Arg::PtrInt),
    (0x5405, "TCGETA", R, Arg::PtrInt),
    (0x5406, "TCSETA", W, Arg::PtrInt),
    (0x5407, "TCSETAW", W, Arg::PtrInt),
    (0x5408, "TCSETAF", W, Arg::PtrInt),
    (0x5409, "TCSBRK", 0, Arg::Int),
    (0x5425, "TCSBRKP", 0, Arg::Int),
    (0x540a, "TCXONC", 0, Arg::Int),
    (0x540b, "TCFLSH", 0, Arg::Int),
    (0x540c, "TIOCEXCL", 0, Arg::Null),
    (0x540d, "TIOCNXCL", 0, Arg::Null),
    (0x540e, "TIOCSCTTY", 0, Arg::Int),
    (0x540f, "TIOCGPGRP", R, Arg::PtrInt),
    (0x5410, "TIOCSPGRP", W, Arg::PtrInt),
    (0x5429, "TIOCGSID", R, Arg::PtrInt),
    (0x5411, "TIOCOUTQ", R, Arg::PtrInt),
    (0x5412, "TIOCSTI", W, Arg::PtrInt),
    (0x5415, "TIOCMGET", R, Arg::PtrInt),
    (0x5416, "TIOCMBIS", W, Arg::PtrInt),
    (0x5417, "TIOCMBIC", W, Arg::PtrInt),
    (0x5418, "TIOCMSET", W, Arg::PtrInt),
    (0x5419, "TIOCGSOFTCAR", R, Arg::PtrInt),
    (0x541a, "TIOCSSOFTCAR", W, Arg::PtrInt),
    (0x541c, "TIOCLINUX", R | W, Arg::PtrInt),
    (0x541d, "TIOCCONS", 0, Arg::Null),
    (0x541e, "TIOCGSERIAL", R, Arg::PtrInt),
    (0x541f, "TIOCSSERIAL", W, Arg::PtrInt),
    (0x5420, "TIOCPKT", W, Arg::PtrInt),
    (0x5421, "FIONBIO", W, Arg::PtrInt),
    (0x5422, "TIOCNOTTY", 0, Arg::Null),
    (0x5424, "TIOCGETD", R, Arg::PtrInt),
    (0x5423, "TIOCSETD", W, Arg::PtrInt),
    (0x8004_5430, "TIOCGPTN", R, Arg::PtrInt),
    (0x4004_5431, "TIOCSPTLCK", W, Arg::PtrInt),
    (0x5441, "TIOCGPTPEER", 0, Arg::Int),
    (0x5451, "FIOCLEX", 0, Arg::Null),
    (0x5450, "FIONCLEX", 0, Arg::Null),
    (0x5452, "FIOASYNC", W, Arg::PtrInt),
    (0x5456, "TIOCGLCKTRMIOS", R, Arg::Termios),
    (0x5457, "TIOCSLCKTRMIOS", W, Arg::Termios),
    (0x5453, "TIOCSERCONFIG", 0, Arg::Null),
    (0x5459, "TIOCSERGETLSR", R, Arg::PtrInt),
    (0x545c, "TIOCMIWAIT", 0, Arg::Int),
    (0x8903, "FIOGETOWN", R, Arg::PtrInt),
    (0x8901, "FIOSETOWN", W, Arg::PtrInt),
];

/// `print_syscall_err()` and friends, building one part of a line.
pub(crate) struct Out<'a> {
    s: Vec<u8>,
    space: &'a GuestSpace,
}

fn comma(last: bool) -> &'static str {
    if last { "" } else { "," }
}

/// `%#x`.
fn alt_hex(v: u32) -> String {
    if v == 0 { "0".into() } else { format!("{v:#x}") }
}

impl Out<'_> {
    fn str(&mut self, s: &str) {
        self.s.extend_from_slice(s.as_bytes());
    }

    fn ld(&mut self, v: i64) {
        let _ = write!(self.s, "{v}");
    }

    /// `print_raw_param("%d", v, last)`.
    fn d(&mut self, v: u64, last: bool) {
        let _ = write!(self.s, "{}{}", v as i32, comma(last));
    }

    /// `print_raw_param("%u", v, last)`.
    fn u(&mut self, v: u64, last: bool) {
        let _ = write!(self.s, "{}{}", v as u32, comma(last));
    }

    /// `print_raw_param(TARGET_ABI_FMT_ld, v, last)`.
    fn l(&mut self, v: u64, last: bool) {
        let _ = write!(self.s, "{}{}", v as i64, comma(last));
    }

    /// `print_raw_param(TARGET_ABI_FMT_lu, v, last)`.
    fn lu(&mut self, v: u64, last: bool) {
        let _ = write!(self.s, "{v}{}", comma(last));
    }

    /// `print_raw_param("%#x", v, last)`.
    fn x(&mut self, v: u64, last: bool) {
        let _ = write!(self.s, "{}{}", alt_hex(v as u32), comma(last));
    }

    fn read32(&self, addr: u64) -> Option<u32> {
        let mut b = [0u8; 4];
        self.space.read(addr, &mut b).then(|| u32::from_le_bytes(b))
    }

    fn read64(&self, addr: u64) -> Option<u64> {
        let mut b = [0u8; 8];
        self.space.read(addr, &mut b).then(|| u64::from_le_bytes(b))
    }

    /// `print_syscall_err()`: true when `ret` is an error, which it has printed.
    fn err(&mut self, ret: i64) -> bool {
        self.str(" = ");
        if (ret as u64) < (-4096i64) as u64 {
            return false;
        }
        let e = -ret as i32;
        let text = match e {
            512 => "To be restarted".to_string(),
            513 => "Successful exit from sigreturn".to_string(),
            _ => crate::strerror(e),
        };
        let _ = write!(self.s, "-1 errno={e} ({text})");
        true
    }

    /// `print_pointer()`.
    fn pointer(&mut self, p: u64, last: bool) {
        if p == 0 {
            let _ = write!(self.s, "NULL{}", comma(last));
        } else {
            let _ = write!(self.s, "0x{p:x}{}", comma(last));
        }
    }

    /// `print_string()`.
    fn string(&mut self, addr: u64, last: bool) {
        match self.space.read_cstr(addr, usize::MAX) {
            Some(s) => {
                self.s.push(b'"');
                self.s.extend_from_slice(&s);
                self.s.push(b'"');
                self.str(comma(last));
            }
            None => self.pointer(addr, last),
        }
    }

    /// `print_buf()`: at most 40 bytes, the unprintable ones in octal.
    fn buf(&mut self, addr: u64, len: u64, last: bool) {
        let n = len.min(40) as usize;
        let mut b = vec![0u8; n];
        if !self.space.check(addr, len, page::READ) || !self.space.read(addr, &mut b) {
            return self.pointer(addr, last);
        }
        self.s.push(b'"');
        for c in b {
            if (0x20..0x7f).contains(&c) {
                self.s.push(c);
            } else {
                let _ = write!(self.s, "\\{c:o}");
            }
        }
        self.s.push(b'"');
        if n as u64 != len {
            self.str("...");
        }
        self.str(comma(last));
    }

    /// `print_number()`.
    fn number(&mut self, addr: u64, last: bool) {
        if addr == 0 {
            let _ = write!(self.s, "NULL{}", comma(last));
        } else {
            let v = self.read32(addr).unwrap_or(0) as i32;
            let _ = write!(self.s, "[{v}]{}", comma(last));
        }
    }

    /// `print_timeval()` and `print_timespec()`, whose second field is `usec` or `nsec`.
    fn time(&mut self, addr: u64, sub: &str, last: bool) {
        if addr == 0 {
            let _ = write!(self.s, "NULL{}", comma(last));
            return;
        }
        let mut b = [0u8; 16];
        if !self.space.read(addr, &mut b) {
            return self.pointer(addr, last);
        }
        let (sec, frac) = (get64(&b, 0) as i64, get64(&b, 8) as i64);
        let _ = write!(self.s, "{{tv_sec = {sec},tv_{sub} = {frac}}}{}", comma(last));
    }

    fn timeval(&mut self, addr: u64, last: bool) {
        self.time(addr, "usec", last);
    }

    fn timespec(&mut self, addr: u64, last: bool) {
        self.time(addr, "nsec", last);
    }

    /// `print_timezone()`.
    fn timezone(&mut self, addr: u64, last: bool) {
        if addr == 0 {
            let _ = write!(self.s, "NULL{}", comma(last));
            return;
        }
        let mut b = [0u8; 8];
        if !self.space.read(addr, &mut b) {
            return self.pointer(addr, last);
        }
        let (w, d) = (get32(&b, 0) as i32, get32(&b, 4) as i32);
        let _ = write!(self.s, "{{{w},{d}}}{}", comma(last));
    }

    /// `print_itimerval()`.
    fn itimerval(&mut self, addr: u64, last: bool) {
        if addr == 0 {
            let _ = write!(self.s, "NULL{}", comma(last));
            return;
        }
        self.str("{it_interval=");
        self.timeval(addr, false);
        self.str("it_value=");
        self.timeval(addr + 16, false);
        let _ = write!(self.s, "}}{}", comma(last));
    }

    /// `print_rlimit64()`.
    fn rlimit64(&mut self, addr: u64, last: bool) {
        if addr == 0 {
            let _ = write!(self.s, "NULL{}", comma(last));
            return;
        }
        let mut b = [0u8; 16];
        if !self.space.read(addr, &mut b) {
            return self.pointer(addr, last);
        }
        let (cur, max) = (get64(&b, 0) as i64, get64(&b, 8) as i64);
        let _ = write!(self.s, "{{rlim_cur={cur},rlim_max={max}}}{}", comma(last));
    }

    /// `print_flags()`.
    fn flags(&mut self, table: &[Flag], v: u64, last: bool) {
        let mut v = v;
        let mut sep = "";
        let mut n = 0;
        for &Flag(name, value, mask) in table {
            if v & mask == value {
                let _ = write!(self.s, "{sep}{name}");
                v &= !mask;
                sep = "|";
                n += 1;
            }
        }
        if n > 0 {
            if v != 0 {
                let _ = write!(self.s, "{sep}{}{}", alt_hex(v as u32), comma(last));
            } else {
                self.str(comma(last));
            }
        } else {
            let _ = write!(self.s, "{}{}", alt_hex(v as u32), comma(last));
        }
    }

    /// `print_enums()`.
    fn enums(&mut self, table: &[(&str, u64)], v: u64, last: bool) {
        match table.iter().find(|(_, e)| *e == v) {
            Some((name, _)) => self.str(name),
            None => self.str(&alt_hex(v as u32)),
        }
        self.str(comma(last));
    }

    /// `print_at_dirfd()`.
    fn at_dirfd(&mut self, fd: u64, last: bool) {
        if fd as i64 == -100 {
            let _ = write!(self.s, "AT_FDCWD{}", comma(last));
        } else {
            self.d(fd, last);
        }
    }

    /// `print_file_mode()`.
    fn file_mode(&mut self, mode: u64, last: bool) {
        if mode == 0 {
            let _ = write!(self.s, "000{}", comma(last));
            return;
        }
        let mut mode = mode;
        let mut sep = "";
        if let Some(&Flag(name, value, _)) = MODE.iter().find(|m| m.1 & mode == m.1) {
            self.str(name);
            sep = "|";
            mode &= !value;
        }
        mode &= !0o170000;
        if mode != 0 {
            let _ = write!(self.s, "{sep}0{:o}", mode as u32);
        }
        self.str(comma(last));
    }

    /// `print_open_flags()`.
    fn open_flags(&mut self, v: u64, last: bool) {
        self.flags(OPEN_ACCESS, v & 3, true);
        let v = v & !3;
        if v == 0 {
            self.str(comma(last));
            return;
        }
        self.str("|");
        let table = if crate::guest::guest().machine == "aarch64" { OPEN_AARCH64 } else { OPEN };
        self.flags(table, v, last);
    }

    /// `print_signal_1()`.
    fn signal_1(&mut self, sig: u64) {
        match SIGNALS.get(sig as usize) {
            Some(Some(name)) => self.str(name),
            Some(None) => self.str("(null)"),
            None => self.ld(sig as i64),
        }
    }

    /// `print_signal()`.
    fn signal(&mut self, sig: u64, last: bool) {
        self.signal_1(sig);
        self.str(comma(last));
    }

    /// `print_target_sigset_t()`.
    fn sigset(&mut self, addr: u64, size: u64, last: bool) {
        let bits = if addr != 0 && size == 8 { self.read64(addr) } else { None };
        let Some(bits) = bits else { return self.pointer(addr, last) };
        self.str("[");
        let mut first = true;
        for i in 0..64 {
            if bits & (1 << i) != 0 {
                if !first {
                    self.str(" ");
                }
                first = false;
                self.signal_1(i + 1);
            }
        }
        let _ = write!(self.s, "]{}", comma(last));
    }

    /// `print_siginfo()`, with the part of the union `get_target_siginfo()` or the queueing
    /// picks: `fault` for a signal the CPU raised, else by the code and the signal.
    fn siginfo(&mut self, info: &Info, fault: bool) {
        let sig = get32(info, 0);
        let code = i32::from(get32(info, 8) as i16);
        self.str("{si_signo=");
        self.signal(u64::from(sig), true);
        self.str(", si_code=");
        let name = match code {
            0 => "SI_USER",
            0x80 => "SI_KERNEL",
            -1 => "SI_QUEUE",
            -2 => "SI_TIMER",
            -3 => "SI_MESGQ",
            -4 => "SI_ASYNCIO",
            -5 => "SI_SIGIO",
            -6 => "SI_TKILL",
            _ => "",
        };
        if name.is_empty() {
            self.ld(i64::from(code));
        } else {
            self.str(name);
        }
        let (pid, uid) = (get32(info, 16), get32(info, 20));
        if fault {
            self.str(", si_addr=");
            self.pointer(get64(info, 16), true);
        } else if matches!(code, 0 | -6 | 0x80) {
            let _ = write!(self.s, ", si_pid={pid}, si_uid={uid}");
        } else if sig == 17 {
            let _ = write!(
                self.s,
                ", si_pid={pid}, si_uid={uid}, si_status={}, si_utime={}, si_stime={}",
                get32(info, 24) as i32,
                get64(info, 32) as i64,
                get64(info, 40) as i64
            );
        } else if sig == 29 {
            let (band, fd) = (get32(info, 16) as i32, get32(info, 24) as i32);
            let _ = write!(self.s, ", si_band={band}, si_fd={fd}");
        } else {
            let val = get64(info, 24) as i64;
            let _ = write!(self.s, ", si_pid={pid}, si_uid={uid}, si_sigval={val}");
        }
        self.str("}");
    }

    /// A `siginfo_t` the guest passes, or its address.
    fn user_siginfo(&mut self, addr: u64, last: bool) {
        let mut info = [0u8; 128];
        if self.space.read(addr, &mut info) {
            self.siginfo(&info, false);
        } else {
            self.pointer(addr, last);
        }
    }

    /// `print_sockaddr()`.
    fn sockaddr(&mut self, addr: u64, len: u64, last: bool) {
        let mut sa = [0u8; 128];
        let n = (len as usize).min(sa.len());
        if !self.space.check(addr, len, page::READ) || !self.space.read(addr, &mut sa[..n]) {
            self.pointer(addr, false);
        } else {
            let family = u16::from_le_bytes([sa[0], sa[1]]);
            match family {
                1 => {
                    self.str("{sun_family=AF_UNIX,sun_path=\"");
                    let path = &sa[2..];
                    for &c in path.iter().take((len as usize).saturating_sub(2)) {
                        if c == 0 {
                            break;
                        }
                        self.s.push(c);
                    }
                    self.str("\"},");
                }
                2 => {
                    let port = u16::from_be_bytes([sa[2], sa[3]]);
                    let _ = write!(
                        self.s,
                        "{{sin_family=AF_INET,sin_port=htons({port}),sin_addr=inet_addr(\"{}.{}.{}.{}\")}},",
                        sa[4], sa[5], sa[6], sa[7]
                    );
                }
                17 => {
                    let proto = u16::from_be_bytes([sa[2], sa[3]]);
                    let ifindex = get32(&sa, 4) as i32;
                    let _ = write!(
                        self.s,
                        "{{sll_family=AF_PACKET,sll_protocol=htons(0x{proto:04x}),if{ifindex},pkttype="
                    );
                    match sa[10] {
                        0 => self.str("PACKET_HOST"),
                        1 => self.str("PACKET_BROADCAST"),
                        2 => self.str("PACKET_MULTICAST"),
                        3 => self.str("PACKET_OTHERHOST"),
                        4 => self.str("PACKET_OUTGOING"),
                        t => self.ld(i64::from(t)),
                    }
                    let a = &sa[12..20];
                    let _ = write!(
                        self.s,
                        ",sll_addr={:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}}},",
                        a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
                    );
                }
                16 => {
                    let (pid, groups) = (get32(&sa, 4), get32(&sa, 8));
                    let _ =
                        write!(self.s, "{{nl_family=AF_NETLINK,nl_pid={pid},nl_groups={groups}}},");
                }
                _ => {
                    let _ = write!(self.s, "{{sa_family={family}, sa_data={{");
                    // sa_data is char, signed on the host.
                    for &c in &sa[2..15] {
                        let _ = write!(self.s, "{:02x}, ", i32::from(c as i8) as u32);
                    }
                    let _ = write!(self.s, "{:02x}}}}},", i32::from(sa[15] as i8) as u32);
                }
            }
        }
        let _ = write!(self.s, "{}{}", len as i64, comma(last));
    }

    /// `print_termios()` and `print_termios2()`.
    fn termios(&mut self, b: &[u8], two: bool) {
        let (i, o, c, l) = (get32(b, 0), get32(b, 4), get32(b, 8), get32(b, 12));
        self.str("{c_iflag = ");
        self.flags(TERMIOS_I, u64::from(i), false);
        self.str("c_oflag = ");
        let dly = TERMIOS_DLY.iter().fold(0, |m, (f, _)| m | f);
        self.flags(TERMIOS_O, u64::from(o & !dly), false);
        for &(field, values) in TERMIOS_DLY {
            if o & field != 0 {
                self.enums(values, u64::from(o & field), false);
            }
        }
        self.str("c_cflag = ");
        if c & 0o10017 != 0 {
            self.enums(CBAUD, u64::from(c & 0o10017), false);
        }
        if c & 0o60 != 0 {
            self.enums(CSIZE, u64::from(c & 0o60), false);
        }
        self.flags(TERMIOS_C, u64::from(c & !0o10077), false);
        self.str("c_lflag = ");
        self.flags(TERMIOS_L, u64::from(l), false);
        if two {
            let _ = write!(self.s, "c_ispeed = {},c_ospeed = {},", get32(b, 36), get32(b, 40));
        }
        self.str("c_cc = \"");
        let cc = &b[17..36];
        let end = cc.iter().position(|&x| x == 0).unwrap_or(cc.len());
        self.s.extend_from_slice(&cc[..end]);
        self.str("\",c_line = '");
        self.s.push(b[16]);
        self.str("'}");
    }

    /// `thunk_print()` of what an ioctl's argument points to, or the pointer.
    fn ioctl_arg(&mut self, arg: Arg, addr: u64) {
        let size = match arg {
            Arg::Termios => 36,
            Arg::Termios2 => 44,
            Arg::Winsize => 8,
            _ => 4,
        };
        let mut b = [0u8; 44];
        if !self.space.read(addr, &mut b[..size]) {
            return self.pointer(addr, true);
        }
        match arg {
            Arg::Termios | Arg::Termios2 => self.termios(&b, arg == Arg::Termios2),
            Arg::Winsize => {
                let h = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
                let _ = write!(self.s, "{{{},{},{},{}}}", h(0), h(2), h(4), h(6));
            }
            _ => self.ld(i64::from(get32(&b, 0) as i32)),
        }
    }

    /// The `printf()` of a `strace.list` format, with `name` and the arguments.
    fn format(&mut self, f: &str, name: &str, args: &[u64; 6]) {
        let mut next = 0;
        let mut chars = f.chars();
        while let Some(c) = chars.next() {
            if c != '%' {
                let mut b = [0u8; 4];
                self.str(c.encode_utf8(&mut b));
                continue;
            }
            let mut alt = false;
            let mut long = false;
            let mut conv = chars.next().unwrap_or('%');
            if conv == '#' {
                alt = true;
                conv = chars.next().unwrap_or('%');
            }
            if conv == 'l' {
                long = true;
                conv = chars.next().unwrap_or('%');
            }
            if conv == '%' {
                self.str("%");
                continue;
            }
            // The name is the first argument, then the system call's.
            let i = next;
            next += 1;
            if i == 0 {
                self.str(name);
                continue;
            }
            let v = args.get(i - 1).copied().unwrap_or(0);
            match conv {
                'd' if long => self.ld(v as i64),
                'd' => self.ld(i64::from(v as i32)),
                'u' if long => {
                    let _ = write!(self.s, "{v}");
                }
                'u' => {
                    let _ = write!(self.s, "{}", v as u32);
                }
                'x' if long => {
                    let _ = write!(self.s, "{v:x}");
                }
                'x' if alt => self.str(&alt_hex(v as u32)),
                'x' => {
                    let _ = write!(self.s, "{:x}", v as u32);
                }
                'o' if alt && v as u32 != 0 => {
                    let _ = write!(self.s, "0{:o}", v as u32);
                }
                'o' => {
                    let _ = write!(self.s, "{:o}", v as u32);
                }
                'p' if v == 0 => self.str("(nil)"),
                'p' => {
                    let _ = write!(self.s, "0x{v:x}");
                }
                's' if v == 0 => self.str("(null)"),
                's' => match self.space.read_cstr(v, usize::MAX) {
                    Some(s) => self.s.extend_from_slice(&s),
                    None => {
                        let _ = write!(self.s, "0x{v:x}");
                    }
                },
                _ => {}
            }
        }
    }
}

impl Out<'_> {
    /// `print_syscall_prologue()`.
    fn begin(&mut self, e: &Entry) {
        let _ = write!(self.s, "{}(", e.name);
    }

    /// `print_syscall_epilogue()`.
    fn end(&mut self) {
        self.str(")");
    }

    /// `print_execve_argv()`.
    fn argv(&mut self, addr: u64, last: bool) {
        self.str("{");
        let mut p = addr;
        loop {
            let Some(s) = self.read64(p) else { return };
            if s == 0 {
                break;
            }
            if let Some(b) = self.space.read_cstr(s, usize::MAX) {
                self.s.push(b'"');
                self.s.extend_from_slice(&b);
                self.str("\",");
            }
            p += 8;
        }
        let _ = write!(self.s, "NULL}}{}", comma(last));
    }

    /// `print_open_flags()` and, with `O_CREAT`, the mode.
    fn open_mode(&mut self, flags: u64, mode: u64, last: bool) {
        let creat = flags & 0o100 != 0;
        self.open_flags(flags, last && !creat);
        if creat {
            self.file_mode(mode, last);
        }
    }

    /// `print_mknod()` after the path.
    fn mknod(&mut self, mode: u64, dev: u64) {
        let hasdev = mode & 0o60000 != 0;
        self.file_mode(mode, !hasdev);
        if hasdev {
            let major = ((dev >> 8) & 0xfff) | ((dev >> 32) & !0xfff);
            let minor = (dev & 0xff) | ((dev >> 12) & !0xff);
            let _ = write!(self.s, "makedev({},{})", major as i32, minor as i32);
        }
    }
}

fn call_accept(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.d(a[0], false);
    o.pointer(a[1], false);
    o.number(a[2], true);
    o.end();
}

fn call_sendto(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.l(a[0], false);
    o.buf(a[1], a[2], false);
    o.l(a[2], false);
    o.flags(MSG, a[3], false);
    o.sockaddr(a[4], a[5], true);
    o.end();
}

fn call_recvfrom(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.l(a[0], false);
    o.pointer(a[1], false);
    o.l(a[2], false);
    o.flags(MSG, a[3], false);
    o.pointer(a[4], false);
    o.pointer(a[5], true);
    o.end();
}

fn call_bind(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.l(a[0], false);
    o.sockaddr(a[1], a[2], true);
    o.end();
}

fn call_socket(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    let (domain, ty, proto) = (a[0] as i32, a[1] as i32, a[2] as i32);
    o.begin(e);
    match domain {
        1 => o.str("PF_UNIX"),
        2 => o.str("PF_INET"),
        16 => o.str("PF_NETLINK"),
        17 => o.str("PF_PACKET"),
        _ => o.ld(i64::from(domain)),
    }
    o.str(",");
    match ty & 0xf {
        2 => o.str("SOCK_DGRAM"),
        1 => o.str("SOCK_STREAM"),
        3 => o.str("SOCK_RAW"),
        4 => o.str("SOCK_RDM"),
        5 => o.str("SOCK_SEQPACKET"),
        10 => o.str("SOCK_PACKET"),
        _ => {}
    }
    if ty & 0o2000000 != 0 {
        o.str("|SOCK_CLOEXEC");
    }
    if ty & 0o4000 != 0 {
        o.str("|SOCK_NONBLOCK");
    }
    o.str(",");
    const NETLINK: [&str; 23] = [
        "NETLINK_ROUTE",
        "NETLINK_UNUSED",
        "NETLINK_USERSOCK",
        "NETLINK_FIREWALL",
        "NETLINK_SOCK_DIAG",
        "NETLINK_NFLOG",
        "NETLINK_XFRM",
        "NETLINK_SELINUX",
        "NETLINK_ISCSI",
        "NETLINK_AUDIT",
        "NETLINK_FIB_LOOKUP",
        "NETLINK_CONNECTOR",
        "NETLINK_NETFILTER",
        "NETLINK_IP6_FW",
        "NETLINK_DNRTMSG",
        "NETLINK_KOBJECT_UEVENT",
        "NETLINK_GENERIC",
        "",
        "NETLINK_SCSITRANSPORT",
        "NETLINK_ECRYPTFS",
        "NETLINK_RDMA",
        "NETLINK_CRYPTO",
        "NETLINK_SMC",
    ];
    let name = match domain {
        17 if proto == 3 => "ETH_P_ALL",
        16 => NETLINK.get(proto as usize).copied().unwrap_or(""),
        2 | 10 => match proto {
            3 if domain == 2 && ty == 10 => "ETH_P_ALL",
            0 => "IPPROTO_IP",
            6 => "IPPROTO_TCP",
            17 => "IPPROTO_UDP",
            255 => "IPPROTO_RAW",
            _ => "",
        },
        _ => "",
    };
    if name.is_empty() {
        o.ld(i64::from(proto));
    } else {
        o.str(name);
    }
    o.end();
}

fn call_openat2(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.at_dirfd(a[0], false);
    o.string(a[1], false);
    let mut how = [0u8; 24];
    if a[3] >= 24 && o.space.read(a[2], &mut how) {
        let (flags, mode, resolve) = (get64(&how, 0), get64(&how, 8), get64(&how, 16));
        o.str("{");
        o.open_flags(flags, false);
        if flags & 0o100 != 0 {
            o.file_mode(mode, false);
        }
        o.flags(OPENAT2_RESOLVE, resolve, true);
        o.str("},");
    } else {
        o.pointer(a[2], false);
    }
    o.lu(a[3], true);
    o.end();
}

fn call_statx(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.at_dirfd(a[0], false);
    o.string(a[1], false);
    o.flags(STATX_FLAGS, a[2], false);
    o.flags(STATX_MASK, a[3], false);
    o.pointer(a[4], true);
    o.end();
}

fn call_shmat(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.l(a[0], false);
    o.pointer(a[1], false);
    o.flags(SHMAT, a[2], true);
    o.end();
}

fn call_utimensat(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.at_dirfd(a[0], false);
    o.string(a[1], false);
    o.pointer(a[2], false);
    o.flags(AT_FILE, a[3], true);
    o.end();
}

fn call_getxattr(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.string(a[0], false);
    o.string(a[1], false);
    o.pointer(a[2], false);
    o.lu(a[3], true);
    o.end();
}

fn call_fgetxattr(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.d(a[0], false);
    o.string(a[1], false);
    o.pointer(a[2], false);
    o.lu(a[3], true);
    o.end();
}

fn call_listxattr(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.string(a[0], false);
    o.pointer(a[1], false);
    o.lu(a[2], true);
    o.end();
}

fn call_flistxattr(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.d(a[0], false);
    o.pointer(a[1], false);
    o.lu(a[2], true);
    o.end();
}

fn call_removexattr(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.string(a[0], false);
    o.string(a[1], true);
    o.end();
}

fn call_fremovexattr(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.d(a[0], false);
    o.string(a[1], true);
    o.end();
}

fn call_mq_open(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    let creat = a[1] & 0o100 != 0;
    o.begin(e);
    o.string(a[0], false);
    o.open_flags(a[1], !creat);
    if creat {
        o.file_mode(a[2], false);
        o.pointer(a[3], true);
    }
    o.end();
}

fn call_open(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.string(a[0], false);
    o.open_mode(a[1], a[2], true);
    o.end();
}

fn call_openat(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.at_dirfd(a[0], false);
    o.string(a[1], false);
    o.open_mode(a[2], a[3], true);
    o.end();
}

fn call_syslog(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    match SYSLOG.get(a[0] as usize) {
        Some(name) => {
            o.str(name);
            o.str(",");
        }
        None => o.l(a[0], false),
    }
    o.pointer(a[1], false);
    o.d(a[2], true);
    o.end();
}

fn call_madvise(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.pointer(a[0], false);
    o.d(a[1], false);
    o.enums(MADVISE, a[2], true);
    o.end();
}

fn call_fstatat64(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.at_dirfd(a[0], false);
    o.string(a[1], false);
    o.pointer(a[2], false);
    o.flags(AT_FILE, a[3], true);
    o.end();
}

fn call_readlink(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.string(a[0], false);
    o.pointer(a[1], false);
    o.u(a[2], true);
    o.end();
}

fn call_readlinkat(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.at_dirfd(a[0], false);
    o.string(a[1], false);
    o.pointer(a[2], false);
    o.u(a[3], true);
    o.end();
}

fn call_kill(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.d(a[0], false);
    o.signal(a[1], true);
    o.end();
}

fn call_tkill(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    call_kill(o, e, a);
}

fn call_tgkill(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.d(a[0], false);
    o.d(a[1], false);
    o.signal(a[2], true);
    o.end();
}

fn call_pread64(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.d(a[0], false);
    o.pointer(a[1], false);
    o.d(a[2], false);
    o.l(a[3], true);
    o.end();
}

fn call_prlimit64(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.d(a[0], false);
    match RLIMITS.get(a[1] as usize) {
        Some(name) => {
            o.str(name);
            o.str(",");
        }
        None => o.d(a[1], false),
    }
    o.rlimit64(a[2], false);
    o.pointer(a[3], true);
    o.end();
}

fn call_mlockall(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.flags(MLOCKALL, a[0], true);
    o.end();
}

fn call_mount(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.string(a[0], false);
    o.string(a[1], false);
    o.string(a[2], false);
    o.flags(MOUNT, a[3], false);
    o.pointer(a[4], true);
    o.end();
}

fn call_mmap(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.pointer(a[0], false);
    o.d(a[1], false);
    o.flags(PROT, a[2], false);
    o.flags(MMAP, a[3], false);
    o.d(a[4], false);
    o.x(a[5], true);
    o.end();
}

fn call_mprotect(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.pointer(a[0], false);
    o.d(a[1], false);
    o.flags(PROT, a[2], true);
    o.end();
}

fn call_munmap(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.pointer(a[0], false);
    o.d(a[1], true);
    o.end();
}

fn call_brk(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.pointer(a[0], true);
    o.end();
}

fn call_rt_sigaction(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.signal(a[0], false);
    o.pointer(a[1], false);
    o.pointer(a[2], true);
    o.end();
}

fn call_access(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.string(a[0], false);
    o.flags(ACCESS, a[1], true);
    o.end();
}

fn call_chmod(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.string(a[0], false);
    o.file_mode(a[1], true);
    o.end();
}

fn call_mkdir(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    call_chmod(o, e, a);
}

fn call_creat(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    call_chmod(o, e, a);
}

fn call_stat(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.string(a[0], false);
    o.pointer(a[1], true);
    o.end();
}

fn call_fstat(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.d(a[0], false);
    o.pointer(a[1], true);
    o.end();
}

fn call_truncate(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.string(a[0], false);
    o.l(a[1], true);
    o.end();
}

/// The calls that take one path.
fn call_path(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.string(a[0], true);
    o.end();
}

fn call_chdir(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    call_path(o, e, a);
}

fn call_unlink(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    call_path(o, e, a);
}

fn call_acct(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    call_path(o, e, a);
}

fn call_chroot(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    call_path(o, e, a);
}

fn call_mq_unlink(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    call_path(o, e, a);
}

/// QEMU leaves the comma here.
fn call_rmdir(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.string(a[0], false);
    o.end();
}

/// The calls that take two paths.
fn call_rename(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.string(a[0], false);
    o.string(a[1], true);
    o.end();
}

fn call_link(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    call_rename(o, e, a);
}

fn call_symlink(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    call_rename(o, e, a);
}

fn call_chown(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.string(a[0], false);
    o.d(a[1], false);
    o.d(a[2], true);
    o.end();
}

fn call_settimeofday(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.timeval(a[0], false);
    o.timezone(a[1], true);
    o.end();
}

fn call_gettimeofday(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.pointer(a[0], false);
    o.pointer(a[1], true);
    o.end();
}

fn call_utime(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    call_stat(o, e, a);
}

fn call_utimes(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    call_stat(o, e, a);
}

fn call_statfs(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    call_stat(o, e, a);
}

fn call_fchownat(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.at_dirfd(a[0], false);
    o.string(a[1], false);
    o.d(a[2], false);
    o.d(a[3], false);
    o.flags(AT_FILE, a[4], true);
    o.end();
}

fn call_futimesat(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.at_dirfd(a[0], false);
    o.string(a[1], false);
    o.timeval(a[2], false);
    o.timeval(a[2] + 16, true);
    o.end();
}

fn call_unlinkat(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.at_dirfd(a[0], false);
    o.string(a[1], false);
    o.flags(UNLINKAT, a[2], true);
    o.end();
}

fn call_renameat(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.at_dirfd(a[0], false);
    o.string(a[1], false);
    o.at_dirfd(a[2], false);
    o.string(a[3], true);
    o.end();
}

fn call_linkat(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.at_dirfd(a[0], false);
    o.string(a[1], false);
    o.at_dirfd(a[2], false);
    o.string(a[3], false);
    o.flags(AT_FILE, a[4], true);
    o.end();
}

fn call_symlinkat(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.string(a[0], false);
    o.at_dirfd(a[1], false);
    o.string(a[2], true);
    o.end();
}

fn call_fchmodat(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.at_dirfd(a[0], false);
    o.string(a[1], false);
    o.file_mode(a[2], false);
    o.flags(AT_FILE, a[3], true);
    o.end();
}

fn call_faccessat(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.at_dirfd(a[0], false);
    o.string(a[1], false);
    o.flags(ACCESS, a[2], false);
    o.flags(AT_FILE, a[3], true);
    o.end();
}

fn call_mkdirat(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.at_dirfd(a[0], false);
    o.string(a[1], false);
    o.file_mode(a[2], true);
    o.end();
}

fn call_mknod(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.string(a[0], false);
    o.mknod(a[1], a[2]);
    o.end();
}

fn call_mknodat(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.at_dirfd(a[0], false);
    o.string(a[1], false);
    o.mknod(a[2], a[3]);
    o.end();
}

fn call_unshare(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.flags(CLONE, a[0], true);
    o.end();
}

fn call_fallocate(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.d(a[0], false);
    o.flags(FALLOC, a[1], false);
    o.l(a[2], false);
    o.l(a[3], true);
    o.end();
}

fn call_clock_adjtime(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.enums(CLOCKIDS, a[0], false);
    o.pointer(a[1], true);
    o.end();
}

fn call_clock_settime(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.enums(CLOCKIDS, a[0], false);
    o.timespec(a[1], true);
    o.end();
}

fn call_clock_gettime(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    call_clock_adjtime(o, e, a);
}

fn call_clock_nanosleep(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.enums(CLOCKIDS, a[0], false);
    o.d(a[1], false);
    o.timespec(a[2], false);
    o.timespec(a[3], true);
    o.end();
}

fn call_getitimer(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.enums(ITIMERS, a[0], false);
    o.pointer(a[1], true);
    o.end();
}

fn call_setitimer(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.enums(ITIMERS, a[0], false);
    o.itimerval(a[1], false);
    o.pointer(a[2], true);
    o.end();
}

fn call_pidfd_send_signal(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.d(a[0], false);
    o.signal(a[1], false);
    o.user_siginfo(a[2], false);
    o.u(a[3], true);
    o.end();
}

fn call_umount2(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.string(a[0], false);
    o.flags(UMOUNT2, a[1], true);
    o.end();
}

fn call_rt_sigqueueinfo(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.d(a[0], false);
    o.signal(a[1], false);
    o.user_siginfo(a[2], true);
    o.end();
}

fn call_rt_tgsigqueueinfo(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.d(a[0], false);
    o.d(a[1], false);
    o.signal(a[2], false);
    o.user_siginfo(a[3], true);
    o.end();
}

fn call_fsconfig(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    const CMDS: [&str; 9] = [
        "FSCONFIG_SET_FLAG",
        "FSCONFIG_SET_STRING",
        "FSCONFIG_SET_BINARY",
        "FSCONFIG_SET_PATH",
        "FSCONFIG_SET_PATH_EMPTY",
        "FSCONFIG_SET_FD",
        "FSCONFIG_CMD_CREATE",
        "FSCONFIG_CMD_RECONFIGURE",
        "FSCONFIG_CMD_CREATE_EXCL",
    ];
    let cmd = a[1] as i32;
    o.begin(e);
    o.d(a[0], false);
    match CMDS.get(cmd as usize) {
        Some(name) => {
            o.str(name);
            o.str(",");
        }
        None => {
            let _ = write!(o.s, "UNKNOWN_CMD ({cmd}),");
        }
    }
    if (0..=5).contains(&cmd) {
        o.string(a[2], false);
    } else {
        o.pointer(a[2], false);
    }
    if matches!(cmd, 1 | 3 | 4) {
        o.string(a[3], false);
    } else {
        o.pointer(a[3], false);
    }
    if matches!(cmd, 3 | 4) {
        o.at_dirfd(a[4], true);
    } else {
        o.d(a[4], true);
    }
    o.end();
}

fn call_execve(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.string(a[0], false);
    o.argv(a[1], true);
    o.end();
}

fn call_execveat(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.at_dirfd(a[0], false);
    o.string(a[1], false);
    o.argv(a[2], false);
    o.flags(EXECVEAT, a[4], true);
    o.end();
}

fn call_futex(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    let op = a[1] as u32;
    let base = op & !384;
    o.begin(e);
    o.pointer(a[0], false);
    if op & 128 != 0 {
        o.str("FUTEX_PRIVATE_FLAG|");
    }
    if op & 256 != 0 {
        o.str("FUTEX_CLOCK_REALTIME|");
    }
    match FUTEX.get(base as usize) {
        Some(name) => o.str(name),
        None => {
            let _ = write!(o.s, "0x{op:x}");
        }
    }
    let _ = write!(o.s, ",{},", a[2] as i32);
    if matches!(base, 0 | 9 | 6 | 13 | 11) {
        o.timespec(a[3], false);
    } else {
        o.pointer(a[3], false);
    }
    o.pointer(a[4], false);
    if matches!(base, 9 | 10) && a[5] as u32 == u32::MAX {
        o.str("FUTEX_BITSET_MATCH_ANY");
    } else {
        o.x(a[5], true);
    }
    o.end();
}

fn call_lseek(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.d(a[0], false);
    o.l(a[1], false);
    match a[2] {
        0 => o.str("SEEK_SET"),
        1 => o.str("SEEK_CUR"),
        2 => o.str("SEEK_END"),
        3 => o.str("SEEK_DATA"),
        4 => o.str("SEEK_HOLE"),
        w => o.x(w, true),
    }
    o.end();
}

fn call_semctl(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    let _ = write!(o.s, "{}({},{},", e.name, a[0] as i64, a[1] as i64);
    let cmd = (a[2] & 0xff) as i32;
    let name = match cmd {
        0 => "IPC_RMID",
        1 => "IPC_SET",
        2 => "IPC_STAT",
        3 => "IPC_INFO",
        11 => "MSG_STAT",
        12 => "MSG_INFO",
        13 => "SHM_STAT",
        14 => "SHM_INFO",
        15 => "GETZCNT",
        16 => "SETVAL",
        17 => "SETALL",
        18 => "SEM_STAT",
        19 => "SEM_INFO",
        _ => "",
    };
    if name.is_empty() {
        o.ld(i64::from(cmd));
    } else {
        o.str(name);
    }
    let _ = write!(o.s, ",0x{:x})", a[3]);
}

fn call_clone(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    // Only x86-64 lacks CLONE_BACKWARDS, which puts the TLS before the child's TID.
    let (tls, ctid) =
        if crate::guest::guest().machine == "x86_64" { (a[4], a[3]) } else { (a[3], a[4]) };
    o.begin(e);
    o.flags(CLONE, u64::from(a[0] as u32), false);
    let _ = write!(
        o.s,
        "child_stack=0x{:x},parent_tidptr=0x{:x},tls=0x{tls:x},child_tidptr=0x{ctid:x}",
        a[1], a[2]
    );
    o.end();
}

fn call_rt_sigprocmask(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    let how = match a[0] {
        0 => "SIG_BLOCK",
        1 => "SIG_UNBLOCK",
        2 => "SIG_SETMASK",
        _ => "UNKNOWN",
    };
    let _ = write!(o.s, "{how},");
    o.sigset(a[1], a[3], false);
    o.pointer(a[2], false);
    o.u(a[3], true);
    o.end();
}

fn call_fcntl(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.d(a[0], false);
    match a[1] {
        0 | 2 | 1024 | 1026 | 1030 | 1031 => {
            let name = match a[1] {
                0 => "F_DUPFD",
                2 => "F_SETFD",
                1024 => "F_SETLEASE",
                1026 => "F_NOTIFY",
                1030 => "F_DUPFD_CLOEXEC",
                _ => "F_SETPIPE_SZ",
            };
            let _ = write!(o.s, "{name},");
            o.l(a[2], true);
        }
        1 => o.str("F_GETFD"),
        3 => o.str("F_GETFL"),
        9 => o.str("F_GETOWN"),
        11 => o.str("F_GETSIG"),
        1025 => o.str("F_GETLEASE"),
        1032 => o.str("F_GETPIPE_SZ"),
        1034 => o.str("F_GET_SEALS"),
        4 => {
            o.str("F_SETFL,");
            o.open_flags(a[2], true);
        }
        5 | 6 | 7 | 36 | 37 | 38 | 15 | 16 => {
            let name = match a[1] {
                5 => "F_GETLK",
                6 => "F_SETLK",
                7 => "F_SETLKW",
                36 => "F_OFD_GETLK",
                37 => "F_OFD_SETLK",
                38 => "F_OFD_SETLKW",
                15 => "F_SETOWN_EX",
                _ => "F_GETOWN_EX",
            };
            let _ = write!(o.s, "{name},");
            o.pointer(a[2], true);
        }
        8 => {
            o.str("F_SETOWN,");
            o.l(a[2], false);
        }
        10 => {
            o.str("F_SETSIG,");
            o.l(a[2], false);
        }
        1033 => {
            let _ = write!(o.s, "F_ADD_SEALS,0x{:x}", a[2]);
        }
        _ => {
            o.l(a[1], false);
            o.pointer(a[2], true);
        }
    }
    o.end();
}

fn ioctl(cmd: u64) -> Option<&'static (u32, &'static str, u8, Arg)> {
    IOCTLS.iter().find(|i| i.0 == cmd as u32)
}

fn call_ioctl(o: &mut Out<'_>, e: &Entry, a: &[u64; 6]) {
    o.begin(e);
    o.d(a[0], false);
    match ioctl(a[1]) {
        None => {
            o.x(a[1], false);
            o.x(a[2], true);
        }
        Some(&(_, name, access, arg)) => {
            o.str(name);
            if arg != Arg::Null {
                o.str(",");
                match arg {
                    Arg::Int => o.d(a[2], true),
                    _ if access == R => o.pointer(a[2], true),
                    _ => o.ioctl_arg(arg, a[2]),
                }
            }
        }
    }
    o.end();
}

fn ret_addr(o: &mut Out<'_>, _: &Entry, ret: i64, _: &[u64; 6]) {
    if !o.err(ret) {
        let _ = write!(o.s, "0x{:x}", ret as u64);
    }
    o.str("\n");
}

fn ret_adjtimex(o: &mut Out<'_>, _: &Entry, ret: i64, _: &[u64; 6]) {
    if !o.err(ret) {
        o.ld(ret);
        o.str(match ret {
            0 => " TIME_OK (clock synchronized, no leap second)",
            1 => " TIME_INS (insert leap second)",
            2 => " TIME_DEL (delete leap second)",
            3 => " TIME_OOP (leap second in progress)",
            4 => " TIME_WAIT (leap second has occurred)",
            5 => " TIME_ERROR (clock not synchronized)",
            _ => "",
        });
    }
    o.str("\n");
}

fn ret_clock_gettime(o: &mut Out<'_>, _: &Entry, ret: i64, a: &[u64; 6]) {
    if !o.err(ret) {
        o.ld(ret);
        o.str(" (");
        o.timespec(a[1], true);
        o.str(")");
    }
    o.str("\n");
}

fn ret_gettimeofday(o: &mut Out<'_>, _: &Entry, ret: i64, a: &[u64; 6]) {
    if !o.err(ret) {
        o.ld(ret);
        o.str(" (");
        o.timeval(a[0], false);
        o.timezone(a[1], true);
        o.str(")");
    }
    o.str("\n");
}

fn ret_getitimer(o: &mut Out<'_>, _: &Entry, ret: i64, a: &[u64; 6]) {
    if !o.err(ret) {
        o.ld(ret);
        o.str(" (");
        o.itimerval(a[1], true);
        o.str(")");
    }
    o.str("\n");
}

fn ret_setitimer(o: &mut Out<'_>, _: &Entry, ret: i64, a: &[u64; 6]) {
    if !o.err(ret) {
        o.ld(ret);
        o.str(" (old_value = ");
        o.itimerval(a[2], true);
        o.str(")");
    }
    o.str("\n");
}

fn ret_listxattr(o: &mut Out<'_>, _: &Entry, ret: i64, a: &[u64; 6]) {
    if !o.err(ret) {
        o.ld(ret);
        o.str(" (list = ");
        if a[1] != 0 {
            let mut attr = a[1];
            let mut left = ret;
            while left > 0 {
                if attr != a[1] {
                    o.str(",");
                }
                o.string(attr, true);
                let Some(s) = o.space.read_cstr(attr, usize::MAX) else { break };
                left -= s.len() as i64 + 1;
                attr += s.len() as u64 + 1;
            }
        } else {
            o.str("NULL");
        }
        o.str(")");
    }
    o.str("\n");
}

fn ret_prlimit64(o: &mut Out<'_>, _: &Entry, ret: i64, a: &[u64; 6]) {
    if !o.err(ret) {
        o.ld(ret);
        if a[3] != 0 {
            o.str(" (");
            o.rlimit64(a[3], true);
            o.str(")");
        }
    }
    o.str("\n");
}

fn ret_rt_sigprocmask(o: &mut Out<'_>, _: &Entry, ret: i64, a: &[u64; 6]) {
    if !o.err(ret) {
        o.ld(ret);
        if a[2] != 0 {
            o.str(" (oldset=");
            o.sigset(a[2], a[3], true);
            o.str(")");
        }
    }
    o.str("\n");
}

/// `print_ret_wstatus()`, which prints only when the status cannot be read: QEMU tests
/// `get_user_s32()` as if it gave true on success, and then prints the zero it stores.
fn ret_wait4(o: &mut Out<'_>, _: &Entry, ret: i64, a: &[u64; 6]) {
    if !o.err(ret) && a[1] != 0 && o.read32(a[1]).is_none() {
        o.ld(ret);
        o.str(" (wstatus={WIFEXITED(s) && WEXITSTATUS(s) == 0})");
    }
    o.str("\n");
}

fn ret_ioctl(o: &mut Out<'_>, _: &Entry, ret: i64, a: &[u64; 6]) {
    if !o.err(ret) {
        o.ld(ret);
        if let Some(&(_, _, access, arg)) = ioctl(a[1]) {
            if access & R != 0 {
                o.str(" (");
                o.ioctl_arg(arg, a[2]);
                o.str(")");
            }
        }
    }
    o.str("\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvm_user_common::MapKind;

    const ANON: MapKind = MapKind { fixed: false, noreplace: false, shared: false, anon: true };

    /// A guest space with one page at the returned address.
    fn space() -> (GuestSpace, u64) {
        let s = GuestSpace::new(1 << 36, 0x2aaa_aaaa_b000, 0x5555_5555_6000).unwrap();
        let a = s.mmap(0, 4096, page::READ | page::WRITE, ANON, None, 0).unwrap();
        (s, a)
    }

    fn call(s: &GuestSpace, nr: i64, args: [u64; 6]) -> String {
        let e = entry(nr).unwrap();
        let mut o = Out { s: Vec::new(), space: s };
        match e.call {
            Some(call) => call(&mut o, e, &args),
            None => o.format(e.format.unwrap_or("%s(%ld,%ld,%ld,%ld,%ld,%ld)"), e.name, &args),
        }
        String::from_utf8(o.s).unwrap()
    }

    fn ret(s: &GuestSpace, nr: i64, r: i64, args: [u64; 6]) -> String {
        let e = entry(nr).unwrap();
        let mut o = Out { s: Vec::new(), space: s };
        match e.result {
            Some(result) => result(&mut o, e, r, &args),
            None => {
                if !o.err(r) {
                    o.ld(r);
                }
                o.str("\n");
            }
        }
        String::from_utf8(o.s).unwrap()
    }

    #[test]
    fn table_is_sorted() {
        assert!(table::ENTRIES.windows(2).all(|w| w[0].0 < w[1].0));
        assert_eq!(entry(1).unwrap().name, "write");
        assert!(entry(1000).is_none());
    }

    #[test]
    fn formats() {
        let (s, a) = space();
        assert!(s.write(a, b"/etc\0"));
        assert_eq!(call(&s, 1, [1, a, 5, 0, 0, 0]), format!("write(1,0x{a:x},5)"));
        assert_eq!(call(&s, 0, [3, 0, 7, 0, 0, 0]), "read(3,(nil),7)");
        assert_eq!(call(&s, 95, [0o22, 0, 0, 0, 0, 0]), "umask(022)");
        assert_eq!(call(&s, 95, [0, 0, 0, 0, 0, 0]), "umask(0)");
        assert_eq!(call(&s, 60, [u64::MAX, 0, 0, 0, 0, 0]), "exit(-1)\n");
        assert_eq!(call(&s, 80, [a, 0, 0, 0, 0, 0]), "chdir(\"/etc\")");
        assert_eq!(call(&s, 87, [0, 0, 0, 0, 0, 0]), "unlink(NULL)");
        assert_eq!(call(&s, 84, [a, 0, 0, 0, 0, 0]), "rmdir(\"/etc\",)");
    }

    #[test]
    fn results() {
        let (s, _) = space();
        assert_eq!(ret(&s, 1, 5, [0; 6]), " = 5\n");
        assert_eq!(ret(&s, 1, -9, [0; 6]), " = -1 errno=9 (Bad file descriptor)\n");
        assert_eq!(ret(&s, 1, -512, [0; 6]), " = -1 errno=512 (To be restarted)\n");
        assert_eq!(ret(&s, 9, 0x7f00_0000_0000, [0; 6]), " = 0x7f0000000000\n");
        // QEMU prints a wait4() status only when it cannot read it.
        assert_eq!(ret(&s, 61, 42, [0; 6]), " = \n");
        assert_eq!(
            ret(&s, 61, 42, [0, 8, 0, 0, 0, 0]),
            " = 42 (wstatus={WIFEXITED(s) && WEXITSTATUS(s) == 0})\n"
        );
    }

    #[test]
    fn flags_and_modes() {
        let (s, _) = space();
        assert_eq!(
            call(&s, 9, [0, 4096, 3, 0x22, u64::MAX, 0]),
            "mmap(NULL,4096,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0)"
        );
        assert_eq!(call(&s, 10, [0x1000, 8192, 0, 0, 0, 0]), "mprotect(0x1000,8192,PROT_NONE)");
        assert_eq!(
            call(&s, 10, [0x1000, 8192, 0x81, 0, 0, 0]),
            "mprotect(0x1000,8192,PROT_READ|0x80)"
        );
        let mut o = Out { s: Vec::new(), space: &s };
        o.file_mode(0o100644, true);
        o.file_mode(0o755, false);
        o.file_mode(0, true);
        assert_eq!(String::from_utf8(o.s).unwrap(), "S_IFREG|06440755,000");
    }

    #[test]
    fn signals() {
        let (s, a) = space();
        assert!(s.write(a, &((1u64 << 1) | (1 << 15)).to_le_bytes()));
        assert_eq!(
            call(&s, 14, [0, a, 0, 8, 0, 0]),
            "rt_sigprocmask(SIG_BLOCK,[SIGINT SIGSTKFLT],NULL,8)"
        );
        assert_eq!(call(&s, 62, [7, 9, 0, 0, 0, 0]), "kill(7,SIGKILL)");
        assert_eq!(call(&s, 62, [7, 40, 0, 0, 0, 0]), "kill(7,40)");
        let mut info = [0u8; 128];
        info[0] = 11;
        info[8] = 1;
        info[16] = 0x10;
        let mut o = Out { s: Vec::new(), space: &s };
        o.siginfo(&info, true);
        assert_eq!(String::from_utf8(o.s).unwrap(), "{si_signo=SIGSEGV, si_code=1, si_addr=0x10}");
    }

    #[test]
    fn futex_and_sockets() {
        let (s, a) = space();
        assert_eq!(
            call(&s, 202, [0x1000, 0x81, 1, 0, 0, 0]),
            "futex(0x1000,FUTEX_PRIVATE_FLAG|FUTEX_WAKE,1,NULL,NULL,0)"
        );
        assert_eq!(
            call(&s, 41, [2, 0o2000001, 6, 0, 0, 0]),
            "socket(PF_INET,SOCK_STREAM|SOCK_CLOEXEC,IPPROTO_TCP)"
        );
        let sin = [2u8, 0, 0, 80, 127, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(s.write(a, &sin));
        assert_eq!(
            call(&s, 49, [3, a, 16, 0, 0, 0]),
            "bind(3,{sin_family=AF_INET,sin_port=htons(80),sin_addr=inet_addr(\"127.0.0.1\")},16)"
        );
    }

    #[test]
    fn terminals() {
        let (s, a) = space();
        let mut t = [0u8; 36];
        t[0..4].copy_from_slice(&0o2400u32.to_le_bytes());
        t[4..8].copy_from_slice(&0o5u32.to_le_bytes());
        t[8..12].copy_from_slice(&0o277u32.to_le_bytes());
        t[12..16].copy_from_slice(&0o12u32.to_le_bytes());
        t[17] = 3;
        assert!(s.write(a, &t));
        assert_eq!(
            call(&s, 16, [0, 0x5402, a, 0, 0, 0]),
            "ioctl(0,TCSETS,{c_iflag = ICRNL|IXON,c_oflag = OPOST|ONLCR,c_cflag = B38400,CS8,CREAD,\
             c_lflag = ICANON|ECHO,c_cc = \"\x03\",c_line = '\0'})"
        );
        assert_eq!(call(&s, 16, [1, 0x5413, a, 0, 0, 0]), format!("ioctl(1,TIOCGWINSZ,0x{a:x})"));
        assert_eq!(call(&s, 16, [1, 0x1234, 5, 0, 0, 0]), "ioctl(1,0x1234,0x5)");
        assert!(s.write(a, &[24, 0, 80, 0, 0, 0, 0, 0]));
        assert_eq!(ret(&s, 16, 0, [1, 0x5413, a, 0, 0, 0]), " = 0 ({24,80,0,0})\n");
    }
}
