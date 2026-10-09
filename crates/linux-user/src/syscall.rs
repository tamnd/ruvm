// SPDX-License-Identifier: GPL-2.0-or-later

//! `do_syscall()`: the guest's system calls, on a host with the same system call ABI.
//!
//! Most calls go to the host as they are, once their pointer arguments have been checked against
//! the guest's mappings and turned into host addresses; [`spec`] says which argument is what.
//! The rest are emulated here: memory management (`mmap.c`), the break, `clone()`, `execve()`,
//! the signal calls (in [`crate::signal`]) and the handful of calls whose results name the
//! emulator rather than the program.

use std::sync::Arc;

use ruvm_jit::Cpu;
use ruvm_jit::cputlb::tlb_flush;
use ruvm_user_common::{GuestSpace, MapKind, PAGE_SIZE, page, page_align};

use crate::host::sys;
use crate::signal::{self, Task, guest_sys, guest_syscall};
use crate::x86_64;

const EFAULT: i64 = libc::EFAULT as i64;
const EINVAL: i64 = libc::EINVAL as i64;
const ENOSYS: i64 = libc::ENOSYS as i64;
const ENOTTY: i64 = libc::ENOTTY as i64;

/// `PATH_MAX`, the longest string argument.
const PATH_MAX: usize = 4096;
/// `IOV_MAX`.
const IOV_MAX: u64 = 1024;

/// System calls the libc crate may not name on every version.
mod nr {
    pub(super) const GETRANDOM: i64 = 318;
    pub(super) const MEMFD_CREATE: i64 = 319;
    pub(super) const COPY_FILE_RANGE: i64 = 326;
    pub(super) const PREADV2: i64 = 327;
    pub(super) const PWRITEV2: i64 = 328;
    pub(super) const STATX: i64 = 332;
    pub(super) const RSEQ: i64 = 334;
    pub(super) const CLONE3: i64 = 435;
    pub(super) const CLOSE_RANGE: i64 = 436;
    pub(super) const FACCESSAT2: i64 = 439;
    pub(super) const RENAMEAT2: i64 = 316;
}

/// What a system call argument is.
#[derive(Clone, Copy, Debug)]
enum A {
    /// A value, passed as it is.
    V,
    /// A string the kernel reads, NULL allowed.
    S,
    /// A path the kernel reads, looked up under `-L` first.
    P,
    /// `n` bytes the kernel reads.
    R(u64),
    /// `n` bytes the kernel writes.
    W(u64),
    /// `n` bytes the kernel reads and writes.
    M(u64),
    /// Bytes the kernel reads, as many as argument `i` says.
    RL(usize),
    /// Bytes the kernel writes, as many as argument `i` says.
    WL(usize),
    /// Argument `i` elements of `n` bytes the kernel reads.
    RN(usize, u64),
    /// Argument `i` elements of `n` bytes the kernel writes.
    WN(usize, u64),
    /// Argument `i` elements of `n` bytes the kernel reads and writes.
    MN(usize, u64),
    /// Bytes the kernel writes, as many as the `u32` argument `i` points to says.
    WP(usize),
    /// An `fd_set` of as many descriptors as argument `i` says.
    F(usize),
}

use A::*;

/// The arguments of the system calls that go to the host as they are, once their pointers
/// are checked.
fn spec(n: i64) -> Option<&'static [A]> {
    Some(match n {
        libc::SYS_read => &[V, WL(2), V],
        libc::SYS_write => &[V, RL(2), V],
        libc::SYS_close => &[V],
        libc::SYS_stat | libc::SYS_lstat => &[P, W(144)],
        libc::SYS_fstat => &[V, W(144)],
        libc::SYS_poll => &[MN(1, 8), V, V],
        libc::SYS_lseek => &[V, V, V],
        libc::SYS_pread64 => &[V, WL(2), V, V],
        libc::SYS_pwrite64 => &[V, RL(2), V, V],
        libc::SYS_access => &[P, V],
        libc::SYS_pipe => &[W(8)],
        libc::SYS_select => &[V, F(0), F(0), F(0), M(16)],
        libc::SYS_sched_yield
        | libc::SYS_getpid
        | libc::SYS_getppid
        | libc::SYS_getpgrp
        | libc::SYS_setsid
        | libc::SYS_sync
        | libc::SYS_getuid
        | libc::SYS_getgid
        | libc::SYS_geteuid
        | libc::SYS_getegid
        | libc::SYS_gettid
        | libc::SYS_pause
        | libc::SYS_vhangup
        | libc::SYS_munlockall => &[],
        libc::SYS_dup
        | libc::SYS_fsync
        | libc::SYS_fdatasync
        | libc::SYS_fchdir
        | libc::SYS_umask
        | libc::SYS_alarm
        | libc::SYS_setuid
        | libc::SYS_setgid
        | libc::SYS_getpgid
        | libc::SYS_getsid
        | libc::SYS_epoll_create
        | libc::SYS_epoll_create1
        | libc::SYS_syncfs
        | libc::SYS_inotify_init1
        | libc::SYS_personality => &[V],
        libc::SYS_dup2
        | libc::SYS_listen
        | libc::SYS_shutdown
        | libc::SYS_flock
        | libc::SYS_ftruncate
        | libc::SYS_fchmod
        | libc::SYS_setpgid
        | libc::SYS_setreuid
        | libc::SYS_setregid
        | libc::SYS_getpriority
        | libc::SYS_eventfd2
        | libc::SYS_munlock
        | libc::SYS_mlock
        | libc::SYS_timerfd_create
        | libc::SYS_inotify_rm_watch => &[V, V],
        libc::SYS_dup3
        | libc::SYS_fchown
        | libc::SYS_setpriority
        | libc::SYS_setresuid
        | libc::SYS_setresgid
        | libc::SYS_fadvise64
        | libc::SYS_socket => &[V, V, V],
        libc::SYS_nanosleep => &[R(16), W(16)],
        libc::SYS_getitimer => &[V, W(24)],
        libc::SYS_setitimer => &[V, R(24), W(24)],
        libc::SYS_sendfile => &[V, V, M(8), V],
        libc::SYS_connect | libc::SYS_bind => &[V, RL(2), V],
        libc::SYS_accept => &[V, WP(2), M(4)],
        libc::SYS_accept4 => &[V, WP(2), M(4), V],
        libc::SYS_getsockname | libc::SYS_getpeername => &[V, WP(2), M(4)],
        libc::SYS_socketpair => &[V, V, V, W(8)],
        libc::SYS_sendto => &[V, RL(2), V, V, RL(5), V],
        libc::SYS_recvfrom => &[V, WL(2), V, V, WP(5), M(4)],
        libc::SYS_setsockopt => &[V, V, V, RL(4), V],
        libc::SYS_getsockopt => &[V, V, V, WP(4), M(4)],
        libc::SYS_truncate => &[S, V],
        libc::SYS_getdents | libc::SYS_getdents64 => &[V, WL(2), V],
        libc::SYS_getcwd => &[WL(1), V],
        libc::SYS_chdir
        | libc::SYS_rmdir
        | libc::SYS_unlink
        | libc::SYS_chroot
        | libc::SYS_acct => &[S],
        libc::SYS_rename | libc::SYS_link | libc::SYS_symlink => &[S, S],
        libc::SYS_mkdir | libc::SYS_creat | libc::SYS_chmod => &[S, V],
        libc::SYS_mknod => &[S, V, V],
        libc::SYS_chown | libc::SYS_lchown => &[S, V, V],
        libc::SYS_readlink => &[P, WL(2), V],
        libc::SYS_gettimeofday => &[W(16), W(8)],
        libc::SYS_settimeofday => &[R(16), R(8)],
        libc::SYS_getrlimit => &[V, W(16)],
        libc::SYS_getrusage => &[V, W(144)],
        libc::SYS_sysinfo => &[W(112)],
        libc::SYS_times => &[W(32)],
        libc::SYS_getgroups => &[V, WN(0, 4)],
        libc::SYS_setgroups => &[V, RN(0, 4)],
        libc::SYS_getresuid | libc::SYS_getresgid => &[W(4), W(4), W(4)],
        libc::SYS_statfs => &[P, W(120)],
        libc::SYS_fstatfs => &[V, W(120)],
        libc::SYS_sched_getaffinity => &[V, V, WL(1)],
        libc::SYS_sched_setaffinity => &[V, V, RL(1)],
        libc::SYS_sched_getparam => &[V, W(4)],
        libc::SYS_sched_setparam => &[V, R(4)],
        libc::SYS_sched_getscheduler
        | libc::SYS_sched_get_priority_max
        | libc::SYS_sched_get_priority_min => &[V],
        libc::SYS_sched_setscheduler => &[V, V, R(4)],
        libc::SYS_time => &[W(8)],
        libc::SYS_clock_gettime | libc::SYS_clock_getres => &[V, W(16)],
        libc::SYS_clock_nanosleep => &[V, V, R(16), W(16)],
        libc::SYS_utimes => &[S, R(32)],
        libc::SYS_utime => &[S, R(16)],
        libc::SYS_futimesat => &[V, S, R(32)],
        libc::SYS_openat => &[V, P, V, V],
        libc::SYS_open => &[P, V, V],
        libc::SYS_mkdirat => &[V, S, V],
        libc::SYS_mknodat => &[V, S, V, V],
        libc::SYS_fchownat => &[V, S, V, V, V],
        libc::SYS_newfstatat => &[V, P, W(144), V],
        libc::SYS_unlinkat => &[V, S, V],
        libc::SYS_renameat => &[V, S, V, S],
        libc::SYS_linkat => &[V, S, V, S, V],
        libc::SYS_symlinkat => &[S, V, S],
        libc::SYS_readlinkat => &[V, P, WL(3), V],
        libc::SYS_fchmodat => &[V, S, V],
        libc::SYS_faccessat => &[V, P, V],
        libc::SYS_utimensat => &[V, S, R(32), V],
        libc::SYS_fallocate => &[V, V, V, V],
        libc::SYS_pipe2 => &[W(8), V],
        libc::SYS_epoll_ctl => &[V, V, V, R(12)],
        libc::SYS_epoll_wait => &[V, WN(2, 12), V, V],
        libc::SYS_timerfd_settime => &[V, V, R(32), W(32)],
        libc::SYS_timerfd_gettime => &[V, W(32)],
        libc::SYS_inotify_add_watch => &[V, S, V],
        libc::SYS_getxattr | libc::SYS_lgetxattr => &[S, S, WL(3), V],
        libc::SYS_fgetxattr => &[V, S, WL(3), V],
        libc::SYS_listxattr | libc::SYS_llistxattr => &[S, WL(2), V],
        libc::SYS_flistxattr => &[V, WL(2), V],
        libc::SYS_getcpu => &[W(4), W(4), V],
        libc::SYS_syslog => &[V, WL(2), V],
        libc::SYS_splice => &[V, M(8), V, M(8), V, V],
        libc::SYS_tee => &[V, V, V, V],
        nr::FACCESSAT2 => &[V, P, V, V],
        nr::STATX => &[V, P, V, V, W(256)],
        nr::GETRANDOM => &[WL(1), V, V],
        nr::MEMFD_CREATE => &[S, V],
        nr::COPY_FILE_RANGE => &[V, M(8), V, M(8), V, V],
        nr::CLOSE_RANGE => &[V, V, V],
        nr::RENAMEAT2 => &[V, S, V, S, V],
        _ => return None,
    })
}

/// The per-process state of the emulated kernel.
pub(crate) struct Proc {
    space: Arc<GuestSpace>,
    /// `target_brk`.
    brk: u64,
    /// `initial_target_brk`.
    initial_brk: u64,
    /// The program, what `/proc/self/exe` is.
    exec_path: String,
    /// `-r`.
    uname_release: Option<String>,
    /// `-L`.
    ld_prefix: String,
}

impl Proc {
    pub(crate) fn new(
        space: Arc<GuestSpace>,
        brk: u64,
        exec_path: String,
        uname_release: Option<String>,
        ld_prefix: String,
    ) -> Self {
        let brk = page_align(brk).unwrap_or(brk);
        Proc { space, brk, initial_brk: brk, exec_path, uname_release, ld_prefix }
    }

    /// The guest's address space.
    pub(crate) fn space(&self) -> &Arc<GuestSpace> {
        &self.space
    }

    /// The host address of `len` bytes at `addr` with the page bits `need`; NULL stays NULL.
    fn buf(&self, addr: u64, len: u64, need: u32) -> Result<u64, i64> {
        if addr == 0 {
            return Ok(0);
        }
        if !self.space.range_valid(addr, len.max(1)) || !self.space.check(addr, len, need) {
            return Err(-EFAULT);
        }
        Ok(self.space.g2h(addr) as u64)
    }

    fn get_u32(&self, addr: u64) -> Result<u32, i64> {
        let mut b = [0u8; 4];
        if self.space.read(addr, &mut b) { Ok(u32::from_le_bytes(b)) } else { Err(-EFAULT) }
    }

    fn get_u64(&self, addr: u64) -> Result<u64, i64> {
        let mut b = [0u8; 8];
        if self.space.read(addr, &mut b) { Ok(u64::from_le_bytes(b)) } else { Err(-EFAULT) }
    }

    fn put(&self, addr: u64, b: &[u8]) -> Result<(), i64> {
        if self.space.write(addr, b) { Ok(()) } else { Err(-EFAULT) }
    }

    /// A string argument, NUL terminated for the host.
    fn cstr(&self, addr: u64) -> Result<Vec<u8>, i64> {
        let mut s = self.space.read_cstr(addr, PATH_MAX).ok_or(-EFAULT)?;
        s.push(0);
        Ok(s)
    }

    /// `is_proc_myself(name, "exe")`.
    fn is_self_exe(name: &[u8]) -> bool {
        let Some(rest) = name.strip_prefix(b"/proc/") else { return false };
        let Some(rest) = rest.strip_suffix(b"/exe") else { return false };
        rest == b"self" || rest == sys(libc::SYS_getpid, &[]).to_string().as_bytes()
    }

    /// Runs a call of [`spec`].
    fn generic(&self, n: i64, args: [u64; 6], spec: &[A]) -> i64 {
        match self.generic_inner(n, args, spec) {
            Ok(r) | Err(r) => r,
        }
    }

    fn generic_inner(&self, n: i64, args: [u64; 6], spec: &[A]) -> Result<i64, i64> {
        let mut a = args;
        let mut keep: Vec<Vec<u8>> = Vec::new();
        let mut written: Vec<(u64, u64)> = Vec::new();
        for (i, s) in spec.iter().enumerate() {
            let v = args[i];
            let (len, need) = match *s {
                V => continue,
                S | P => {
                    if v == 0 {
                        continue;
                    }
                    let mut s = self.cstr(v)?;
                    if matches!(spec[i], P) && s.first() == Some(&b'/') {
                        let name = String::from_utf8_lossy(&s[..s.len() - 1]).into_owned();
                        let p = crate::elf::path(&self.ld_prefix, &name);
                        if p != name {
                            s = p.into_bytes();
                            s.push(0);
                        }
                    }
                    a[i] = s.as_ptr() as u64;
                    keep.push(s);
                    continue;
                }
                R(n) => (n, page::READ),
                W(n) => (n, page::WRITE),
                M(n) => (n, page::READ | page::WRITE),
                RL(j) => (args[j], page::READ),
                WL(j) => (args[j], page::WRITE),
                RN(j, n) => (args[j].checked_mul(n).ok_or(-EFAULT)?, page::READ),
                WN(j, n) => (args[j].checked_mul(n).ok_or(-EFAULT)?, page::WRITE),
                MN(j, n) => (args[j].checked_mul(n).ok_or(-EFAULT)?, page::READ | page::WRITE),
                WP(j) => {
                    let l =
                        if args[j] == 0 || v == 0 { 0 } else { u64::from(self.get_u32(args[j])?) };
                    (l, page::WRITE)
                }
                F(j) => {
                    let n = args[j] as i32;
                    if n < 0 {
                        return Err(-EINVAL);
                    }
                    ((n as u64).div_ceil(64) * 8, page::READ | page::WRITE)
                }
            };
            a[i] = self.buf(v, len, need)?;
            if need & page::WRITE != 0 && v != 0 {
                written.push((v, len));
            }
        }
        let r = guest_syscall(n, a);
        drop(keep);
        if r >= 0 {
            for (g, l) in written {
                self.space.notify_write(g, l);
            }
        }
        Ok(r)
    }
}

/// `do_brk()`.
fn do_brk(p: &mut Proc, cpu: &mut Cpu<'_>, brk: u64) -> i64 {
    if brk < p.initial_brk {
        return p.brk as i64;
    }
    let Some(new_brk) = page_align(brk) else { return p.brk as i64 };
    let old_brk = page_align(p.brk).unwrap_or(p.brk);
    if new_brk == old_brk {
        p.brk = brk;
        return brk as i64;
    }
    if new_brk < old_brk {
        let _ = p.space.munmap(new_brk, old_brk - new_brk);
        tlb_flush(cpu);
        p.brk = brk;
        return brk as i64;
    }
    let kind = MapKind { noreplace: true, anon: true, ..MapKind::default() };
    let r = p.space.mmap(old_brk, new_brk - old_brk, page::READ | page::WRITE, kind, None, 0);
    tlb_flush(cpu);
    if r == Ok(old_brk) {
        p.brk = brk;
    }
    p.brk as i64
}

fn errno(r: Result<u64, i32>) -> i64 {
    match r {
        Ok(v) => v as i64,
        Err(e) => -i64::from(e),
    }
}

/// `validate_prot_to_pageflags()`.
fn prot_flags(prot: u64) -> Option<u32> {
    if prot & !7 != 0 { None } else { Some(prot as u32) }
}

/// `do_mmap()` and `target_mmap()`.
fn do_mmap(p: &Proc, cpu: &mut Cpu<'_>, a: [u64; 6]) -> i64 {
    const MAP_TYPE: u64 = 0xf;
    const MAP_SHARED: u64 = 1;
    const MAP_PRIVATE: u64 = 2;
    const MAP_SHARED_VALIDATE: u64 = 3;
    const MAP_FIXED: u64 = 0x10;
    const MAP_ANONYMOUS: u64 = 0x20;
    const MAP_FIXED_NOREPLACE: u64 = 0x10_0000;
    let (start, len, prot, flags, fd, off) = (a[0], a[1], a[2], a[3], a[4] as i32, a[5]);
    let Some(prot) = prot_flags(prot) else { return -EINVAL };
    let shared = match flags & MAP_TYPE {
        MAP_SHARED | MAP_SHARED_VALIDATE => true,
        MAP_PRIVATE => false,
        _ => return -EINVAL,
    };
    let kind = MapKind {
        fixed: flags & MAP_FIXED != 0,
        noreplace: flags & MAP_FIXED_NOREPLACE != 0 && flags & MAP_FIXED == 0,
        shared,
        anon: flags & MAP_ANONYMOUS != 0,
    };
    let fd = if kind.anon { None } else { Some(fd) };
    let r = p.space.mmap(start, len, prot, kind, fd, off);
    tlb_flush(cpu);
    errno(r)
}

/// The kernel's `struct iovec` array for `cnt` guest iovecs at `addr`, `lock_iovec()`.
fn iovecs(p: &Proc, addr: u64, cnt: u64, write: bool) -> Result<Vec<[u64; 2]>, i64> {
    if cnt > IOV_MAX {
        return Err(-EINVAL);
    }
    let need = if write { page::WRITE } else { page::READ };
    let mut v = Vec::with_capacity(cnt as usize);
    let mut bad = false;
    for i in 0..cnt {
        let base = p.get_u64(addr + 16 * i)?;
        let len = p.get_u64(addr + 16 * i + 8)?;
        if len as i64 & i64::MIN != 0 {
            return Err(-EINVAL);
        }
        if bad || len == 0 {
            v.push([0, 0]);
            continue;
        }
        match p.buf(base, len, need) {
            Ok(h) if base != 0 => v.push([h, len]),
            _ if i == 0 => return Err(-EFAULT),
            _ => {
                // The kernel would stop at the first bad buffer; so does lock_iovec().
                bad = true;
                v.push([0, 0]);
            }
        }
    }
    Ok(v)
}

/// `readv()`, `writev()` and their positioned forms: `iov` is argument 1, the count argument 2.
fn do_iov(p: &Proc, n: i64, a: [u64; 6], write_mem: bool) -> i64 {
    let v = match iovecs(p, a[1], a[2], write_mem) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut h = a;
    h[1] = v.as_ptr() as u64;
    let r = guest_syscall(n, h);
    if r > 0 && write_mem {
        let mut left = r as u64;
        for i in 0..a[2] {
            if left == 0 {
                break;
            }
            let base = p.get_u64(a[1] + 16 * i).unwrap_or(0);
            let l = v[i as usize][1].min(left);
            p.space.notify_write(base, l);
            left -= l;
        }
    }
    r
}

/// A NULL terminated array of strings, for `execve()`.
fn str_array(p: &Proc, mut addr: u64, keep: &mut Vec<Vec<u8>>) -> Result<Vec<u64>, i64> {
    let mut v = Vec::new();
    if addr == 0 {
        v.push(0);
        return Ok(v);
    }
    loop {
        let s = p.get_u64(addr)?;
        if s == 0 {
            break;
        }
        let c = p.cstr(s)?;
        v.push(c.as_ptr() as u64);
        keep.push(c);
        addr += 8;
    }
    v.push(0);
    Ok(v)
}

fn do_execve(p: &Proc, dirfd: Option<u64>, a: [u64; 6]) -> i64 {
    let (path, argv, envp) = if dirfd.is_some() { (a[1], a[2], a[3]) } else { (a[0], a[1], a[2]) };
    let r = (|| {
        let mut keep = Vec::new();
        let mut name = p.cstr(path)?;
        if Proc::is_self_exe(&name[..name.len() - 1]) {
            name = p.exec_path.clone().into_bytes();
            name.push(0);
        }
        let argv = str_array(p, argv, &mut keep)?;
        let envp = str_array(p, envp, &mut keep)?;
        let r = match dirfd {
            Some(fd) => guest_sys(
                libc::SYS_execveat,
                &[fd, name.as_ptr() as u64, argv.as_ptr() as u64, envp.as_ptr() as u64, a[4]],
            ),
            None => guest_sys(
                libc::SYS_execve,
                &[name.as_ptr() as u64, argv.as_ptr() as u64, envp.as_ptr() as u64],
            ),
        };
        Ok(r)
    })();
    match r {
        Ok(v) | Err(v) => v,
    }
}

/// `open()` and `openat()`: `/proc/self/exe` is the program, not the emulator.
fn do_open(p: &Proc, n: i64, a: [u64; 6], at: bool) -> i64 {
    let path = if at { a[1] } else { a[0] };
    if path != 0 {
        if let Some(name) = p.space.read_cstr(path, PATH_MAX) {
            if Proc::is_self_exe(&name) {
                let mut e = p.exec_path.clone().into_bytes();
                e.push(0);
                let (fl, mode) = if at { (a[2], a[3]) } else { (a[1], a[2]) };
                return sys(
                    libc::SYS_openat,
                    &[libc::AT_FDCWD as u64, e.as_ptr() as u64, fl, mode],
                );
            }
        }
    }
    let s = if at { &[V, P, V, V][..] } else { &[P, V, V][..] };
    p.generic(n, a, s)
}

/// `readlink()` and `readlinkat()` of `/proc/self/exe`.
fn do_readlink(p: &Proc, n: i64, a: [u64; 6], at: bool) -> i64 {
    let (path, buf, len) = if at { (a[1], a[2], a[3]) } else { (a[0], a[1], a[2]) };
    let Some(name) = p.space.read_cstr(path, PATH_MAX) else { return -EFAULT };
    if Proc::is_self_exe(&name) {
        let e = p.exec_path.as_bytes();
        let l = e.len().min(len as usize);
        if !p.space.write(buf, &e[..l]) {
            return -EFAULT;
        }
        return l as i64;
    }
    p.generic(n, a, if at { &[V, P, WL(3), V] } else { &[P, WL(2), V] })
}

/// `do_fork()` without `CLONE_VM`: a host fork.
fn do_fork(
    p: &Proc,
    t: &Task,
    cpu: &mut Cpu<'_>,
    flags: u64,
    [newsp, ptid, ctid, tls]: [u64; 4],
) -> i64 {
    const CSIGNAL: u64 = 0xff;
    const CLONE_VM: u64 = 0x100;
    const CLONE_VFORK: u64 = 0x4000;
    const CLONE_SETTLS: u64 = 0x8_0000;
    const CLONE_PARENT_SETTID: u64 = 0x10_0000;
    const CLONE_CHILD_CLEARTID: u64 = 0x20_0000;
    const CLONE_DETACHED: u64 = 0x40_0000;
    const CLONE_CHILD_SETTID: u64 = 0x100_0000;
    const CLONE_IO: u64 = 0x8000_0000;
    let mut flags = flags & 0xffff_ffff;
    if flags & CLONE_VFORK != 0 {
        flags &= !(CLONE_VFORK | CLONE_VM);
    }
    if flags & CLONE_VM != 0 {
        // Threads come with signal delivery.
        return -EINVAL;
    }
    let ok = CSIGNAL
        | CLONE_SETTLS
        | CLONE_PARENT_SETTID
        | CLONE_CHILD_CLEARTID
        | CLONE_CHILD_SETTID
        | CLONE_DETACHED
        | CLONE_IO;
    if flags & !ok != 0 {
        return -EINVAL;
    }
    // Signals stay blocked across the fork; the cpu loop lets them in again.
    if signal::block_signals(t) {
        return signal::ERESTARTSYS;
    }
    let r = sys(libc::SYS_clone, &[flags & CSIGNAL, 0, 0, 0, 0]);
    if r < 0 {
        return r;
    }
    if r == 0 {
        if newsp != 0 {
            x86_64::set_reg(cpu, x86_64::RSP, newsp);
        }
        if flags & CLONE_SETTLS != 0 {
            x86_64::set_tls(cpu, tls);
        }
        let tid = sys(libc::SYS_gettid, &[]) as u32;
        if flags & CLONE_CHILD_SETTID != 0 {
            let _ = p.put(ctid, &tid.to_le_bytes());
        }
        if flags & CLONE_PARENT_SETTID != 0 {
            let _ = p.put(ptid, &tid.to_le_bytes());
        }
    } else if flags & CLONE_PARENT_SETTID != 0 {
        let _ = p.put(ptid, &(r as u32).to_le_bytes());
    }
    r
}

/// The `ioctl()`s whose argument is understood: tty and file descriptor requests.
fn do_ioctl(p: &Proc, a: [u64; 6]) -> i64 {
    let req = a[1] & 0xffff_ffff;
    let s: &[A] = match req {
        0x5401 => &[V, V, W(36)],
        0x5402..=0x5404 => &[V, V, R(36)],
        0x540F | 0x5429 | 0x541B => &[V, V, W(4)],
        0x5410 | 0x5421 => &[V, V, R(4)],
        0x5413 => &[V, V, W(8)],
        0x5414 => &[V, V, R(8)],
        0x5450 | 0x5451 | 0x540B | 0x540A | 0x5409 | 0x540E | 0x5422 => &[V, V, V],
        0x802C_542A => &[V, V, W(44)],
        0x402C_542B => &[V, V, R(44)],
        _ => return -ENOTTY,
    };
    p.generic(libc::SYS_ioctl, a, s)
}

fn do_fcntl(p: &Proc, a: [u64; 6]) -> i64 {
    let s: &[A] = match a[1] & 0xffff_ffff {
        5 | 36 => &[V, V, M(32)],
        6 | 7 | 37 | 38 => &[V, V, R(32)],
        16 | 1035 | 1037 => &[V, V, W(8)],
        15 | 1036 | 1038 => &[V, V, R(8)],
        0..=4 | 8..=11 | 1024..=1026 | 1030..=1034 => &[V, V, V],
        _ => return -EINVAL,
    };
    p.generic(libc::SYS_fcntl, a, s)
}

fn do_prctl(p: &Proc, a: [u64; 6]) -> i64 {
    let s: &[A] = match a[0] {
        1 | 3 | 4 | 36 | 38 | 39 => &[V, V, V, V, V],
        2 | 37 => &[V, W(4)],
        15 => &[V, R(16)],
        16 => &[V, W(16)],
        _ => return -EINVAL,
    };
    p.generic(libc::SYS_prctl, a, s)
}

/// `uname()`, with `-r`.
fn do_uname(p: &Proc, a: [u64; 6]) -> i64 {
    let r = p.generic(libc::SYS_uname, a, &[W(390)]);
    if r == 0 {
        if let Some(rel) = &p.uname_release {
            let mut b = [0u8; 65];
            let n = rel.len().min(64);
            b[..n].copy_from_slice(&rel.as_bytes()[..n]);
            let _ = p.put(a[0] + 130, &b);
        }
    }
    r
}

/// `setrlimit()` and `prlimit64()`: the limits that would bind the emulator rather than the
/// program are not set.
fn ignored_rlimit(res: u64) -> bool {
    res == libc::RLIMIT_AS as u64
        || res == libc::RLIMIT_DATA as u64
        || res == libc::RLIMIT_STACK as u64
}

fn do_futex(p: &Proc, a: [u64; 6]) -> i64 {
    let s: &[A] = match a[1] & 0x7f {
        0 | 9 | 6 | 13 => &[M(4), V, V, R(16), V, V],
        1 | 10 | 7 | 8 => &[M(4), V, V, V, V, V],
        3 | 4 | 5 | 12 => &[M(4), V, V, V, M(4), V],
        11 => &[M(4), V, V, R(16), M(4), V],
        _ => return -ENOSYS,
    };
    p.generic(libc::SYS_futex, a, s)
}

/// `pselect6()`: the last argument is `{ const sigset_t *, size_t }`, the mask to wait with.
fn do_pselect6(p: &Proc, t: &mut Task, a: [u64; 6]) -> i64 {
    let (mut set, mut size) = (0, 0);
    if a[5] != 0 {
        let (Ok(s), Ok(z)) = (p.get_u64(a[5]), p.get_u64(a[5] + 8)) else { return -EFAULT };
        (set, size) = (s, z);
    }
    let space = Arc::clone(&p.space);
    signal::with_sigmask(&space, t, set, size, |m| {
        let ss = [m, 8u64];
        let mut h = a;
        if a[5] != 0 {
            h[5] = ss.as_ptr() as u64;
        }
        p.generic(libc::SYS_pselect6, h, &[V, F(0), F(0), F(0), M(16), V])
    })
}

/// `do_syscall()`.
pub(crate) fn do_syscall(
    p: &mut Proc,
    t: &mut Task,
    cpu: &mut Cpu<'_>,
    n: u64,
    a: [u64; 6],
) -> i64 {
    let n = n as i64;
    let sp = x86_64::reg(cpu, x86_64::RSP);
    if let Some(r) = signal::do_signal_syscall(&p.space, t, sp, n, a) {
        return r;
    }
    match n {
        libc::SYS_brk => do_brk(p, cpu, a[0]),
        libc::SYS_mmap => do_mmap(p, cpu, a),
        libc::SYS_munmap => {
            let r = p.space.munmap(a[0], a[1]);
            tlb_flush(cpu);
            errno(r.map(|()| 0))
        }
        libc::SYS_mprotect => {
            let Some(prot) = prot_flags(a[2]) else { return -EINVAL };
            let r = p.space.mprotect(a[0], a[1], prot);
            tlb_flush(cpu);
            errno(r.map(|()| 0))
        }
        libc::SYS_mremap => {
            const MREMAP_MAYMOVE: u64 = 1;
            const MREMAP_FIXED: u64 = 2;
            if a[3] & !(MREMAP_MAYMOVE | MREMAP_FIXED) != 0 {
                return -EINVAL;
            }
            let r = p.space.mremap(
                a[0],
                a[1],
                a[2],
                a[3] & MREMAP_MAYMOVE != 0,
                a[3] & MREMAP_FIXED != 0,
                a[4],
            );
            tlb_flush(cpu);
            errno(r)
        }
        libc::SYS_madvise => {
            const MADV_DONTNEED: u64 = 4;
            if a[0] % PAGE_SIZE != 0 {
                return -EINVAL;
            }
            if a[2] == MADV_DONTNEED && a[1] != 0 {
                let Some(len) = page_align(a[1]) else { return -EINVAL };
                if !p.space.range_valid(a[0], len) {
                    return -i64::from(libc::ENOMEM);
                }
                // Only anonymous private pages read back as zeros; leave the others alone.
                if p.space.check(a[0], len, page::ANON)
                    && p.space.page_flags(a[0]) & page::SHARED == 0
                {
                    let r = sys(libc::SYS_madvise, &[p.space.g2h(a[0]) as u64, len, MADV_DONTNEED]);
                    p.space.notify_write(a[0], len);
                    return r;
                }
            }
            0
        }
        libc::SYS_msync => {
            let Some(len) = page_align(a[1]) else { return -i64::from(libc::ENOMEM) };
            if a[0] % PAGE_SIZE != 0 {
                return -EINVAL;
            }
            if !p.space.check(a[0], len, 0) {
                return -i64::from(libc::ENOMEM);
            }
            sys(libc::SYS_msync, &[p.space.g2h(a[0]) as u64, len, a[2]])
        }
        libc::SYS_exit | libc::SYS_exit_group => {
            sys(libc::SYS_exit_group, &[a[0]]);
            std::process::exit(a[0] as i32)
        }
        libc::SYS_arch_prctl => x86_64::arch_prctl(&p.space, cpu, a[0], a[1]),
        libc::SYS_set_tid_address => sys(libc::SYS_gettid, &[]),
        libc::SYS_set_robust_list | libc::SYS_get_robust_list | nr::RSEQ | nr::CLONE3 => -ENOSYS,
        libc::SYS_clone => do_fork(p, t, cpu, a[0], [a[1], a[2], a[3], a[4]]),
        libc::SYS_fork => do_fork(p, t, cpu, libc::SIGCHLD as u64, [0; 4]),
        libc::SYS_vfork => do_fork(p, t, cpu, 0x4100 | libc::SIGCHLD as u64, [0; 4]),
        libc::SYS_rt_sigreturn => {
            if signal::block_signals(t) {
                return signal::ERESTARTSYS;
            }
            let space = Arc::clone(&p.space);
            x86_64::do_rt_sigreturn(&space, t, cpu)
        }
        libc::SYS_execve => do_execve(p, None, a),
        libc::SYS_execveat => do_execve(p, Some(a[0]), a),
        libc::SYS_readv | libc::SYS_preadv | nr::PREADV2 => do_iov(p, n, a, true),
        libc::SYS_writev | libc::SYS_pwritev | nr::PWRITEV2 => do_iov(p, n, a, false),
        libc::SYS_ioctl => do_ioctl(p, a),
        libc::SYS_fcntl => do_fcntl(p, a),
        libc::SYS_prctl => do_prctl(p, a),
        libc::SYS_open => do_open(p, n, a, false),
        libc::SYS_openat => do_open(p, n, a, true),
        libc::SYS_readlink => do_readlink(p, n, a, false),
        libc::SYS_readlinkat => do_readlink(p, n, a, true),
        libc::SYS_uname => do_uname(p, a),
        libc::SYS_setrlimit => {
            if ignored_rlimit(a[0]) {
                return 0;
            }
            p.generic(n, a, &[V, R(16)])
        }
        libc::SYS_prlimit64 => {
            let mut h = a;
            if ignored_rlimit(a[1]) {
                h[2] = 0;
            }
            p.generic(n, h, &[V, V, R(16), W(16)])
        }
        libc::SYS_futex => do_futex(p, a),
        libc::SYS_pselect6 => do_pselect6(p, t, a),
        libc::SYS_ppoll => {
            let space = Arc::clone(&p.space);
            signal::with_sigmask(&space, t, a[3], a[4], |m| {
                let h = [a[0], a[1], a[2], m, 8, 0];
                p.generic(n, h, &[MN(1, 8), V, M(16), V, V])
            })
        }
        libc::SYS_epoll_pwait => {
            let space = Arc::clone(&p.space);
            signal::with_sigmask(&space, t, a[4], a[5], |m| {
                let h = [a[0], a[1], a[2], a[3], m, 8];
                p.generic(n, h, &[V, WN(2, 12), V, V, V, V])
            })
        }
        libc::SYS_wait4 => {
            let r = p.generic(n, a, &[V, W(4), V, W(144)]);
            if r > 0 && a[1] != 0 {
                if let Ok(st) = p.get_u32(a[1]) {
                    let st = signal::host_to_target_waitstatus(st as i32);
                    let _ = p.put(a[1], &st.to_le_bytes());
                }
            }
            r
        }
        libc::SYS_waitid => {
            let r = p.generic(n, a, &[V, V, W(128), V, W(144)]);
            if r == 0 && a[2] != 0 {
                let mut b = [0u8; 128];
                if p.space.read(a[2], &mut b) {
                    let _ = p.put(a[2], &signal::host_to_target_siginfo(&b));
                }
            }
            r
        }
        _ => match spec(n) {
            Some(s) => p.generic(n, a, s),
            None => -ENOSYS,
        },
    }
}
