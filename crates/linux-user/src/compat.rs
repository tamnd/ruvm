// SPDX-License-Identifier: GPL-2.0-or-later

//! The system calls of a 32-bit guest on the 64-bit host: the parts of `do_syscall()` that
//! convert between the target's structures, where longs, pointers and times are 32 bits, and
//! the host's. The calls are found by name, so that each 32-bit target only numbers them.
//!
//! Arguments arrive zero extended from the guest's 32-bit registers. A call whose arguments
//! and structures are the host's goes on to [`syscall::do_syscall`] by its host number, as
//! the 64-bit kernel's compat layer does for the calls it has no compat entry for.

use std::sync::Arc;

use ruvm_jit::Cpu;
use ruvm_user_common::page;

use crate::signal::{self, Sigaction, Task, get32, get64, guest_sys, put32, put64};
use crate::syscall::{self, A, Proc};
use crate::syscall::{M, MN, P, R, S, V};

const EFAULT: i64 = libc::EFAULT as i64;
const EINVAL: i64 = libc::EINVAL as i64;
const ENOSYS: i64 = libc::ENOSYS as i64;

type Res = Result<i64, i64>;

/// `abi_long` as a host `long`.
fn sx(v: u64) -> u64 {
    i64::from(v as u32 as i32) as u64
}

/// A 64-bit value passed as two 32-bit registers, low first.
fn pair(lo: u64, hi: u64) -> u64 {
    (lo & 0xffff_ffff) | (hi << 32)
}

/// The 64-bit argument that would be in registers `i` and `i + 1`, and the index of the
/// argument after it. On the ARM EABI the pair starts at an even register, so an odd `i`
/// moves up by one, `regpairs_aligned()`.
fn arg64(a: &[u64; 6], i: usize, eabi: bool) -> (u64, usize) {
    let i = if eabi { (i + 1) & !1 } else { i };
    (pair(a[i], a[i + 1]), i + 2)
}

fn read<const N: usize>(p: &Proc, addr: u64) -> Result<[u8; N], i64> {
    let mut b = [0u8; N];
    if p.space().read(addr, &mut b) { Ok(b) } else { Err(-EFAULT) }
}

/// `low2highuid()`: the 16-bit ID calls take -1 as "unchanged".
fn low2high(id: u64) -> u64 {
    if id as u16 == u16::MAX { u64::from(u32::MAX) } else { id & 0xffff }
}

/// `high2lowuid()`: an ID 16 bits do not hold reads as the overflow ID.
fn high2low(id: u32) -> u16 {
    u16::try_from(id).unwrap_or(65534)
}

/// `target_to_host_timespec()` of a 32-bit `struct timespec` or `struct timeval`.
fn get_time32(p: &Proc, addr: u64) -> Result<[i64; 2], i64> {
    let b: [u8; 8] = read(p, addr)?;
    Ok([i64::from(get32(&b, 0) as i32), i64::from(get32(&b, 4) as i32)])
}

/// `host_to_target_timespec()` into a 32-bit one.
fn put_time32(p: &Proc, addr: u64, t: &[i64]) -> Result<(), i64> {
    let mut b = vec![0u8; 4 * t.len()];
    for (i, v) in t.iter().enumerate() {
        put32(&mut b, 4 * i, *v as u32);
    }
    p.put(addr, &b)
}

/// `n` 32-bit times at `addr`, sign extended, or `None` for NULL.
fn get_times32(p: &Proc, addr: u64, n: usize) -> Result<Option<Vec<i64>>, i64> {
    if addr == 0 {
        return Ok(None);
    }
    let mut v = Vec::with_capacity(n);
    for i in 0..n as u64 / 2 {
        v.extend_from_slice(&get_time32(p, addr + 8 * i)?);
    }
    Ok(Some(v))
}

fn ptr_or_null<T>(v: &Option<Vec<T>>) -> u64 {
    v.as_ref().map_or(0, |v| v.as_ptr() as u64)
}

/// The host call `n` as the 64-bit guest makes it.
fn call(p: &Arc<Proc>, t: &mut Task, cpu: &mut Cpu<'_>, n: i64, a: [u64; 6]) -> i64 {
    syscall::do_syscall(p, t, cpu, n as u64, a)
}

fn host(n: i64, args: &[u64]) -> i64 {
    guest_sys(n, args)
}

/// The host's `struct stat` of `path`, or of `fd` when `path` is `None`.
fn host_stat(p: &Proc, fd: u64, path: Option<u64>, flags: u64) -> Result<[u8; 144], i64> {
    let mut st = [0u8; 144];
    let h = st.as_mut_ptr() as u64;
    let r = match path {
        Some(path) => p.generic(libc::SYS_newfstatat, [fd, path, h, flags, 0, 0], &[V, P, V, V]),
        None => p.generic(libc::SYS_fstat, [fd, h, 0, 0, 0, 0], &[V, V]),
    };
    if r < 0 { Err(r) } else { Ok(st) }
}

/// `struct stat64` from the host's `struct stat`, laid out as the kernel's `cp_new_stat64()`
/// writes it: i386's, or with `eabi` the ARM EABI one, `struct target_eabi_stat64`, whose
/// 64-bit members from `st_size` on are 8-byte aligned.
fn stat64(h: &[u8; 144], eabi: bool) -> Vec<u8> {
    // st_size, st_blksize, st_blocks, st_atime, st_ino and the size.
    let (size, blksize, blocks, times, ino_at, len) =
        if eabi { (48, 56, 64, 72, 96, 104) } else { (44, 52, 56, 64, 88, 96) };
    let mut t = vec![0u8; len];
    let ino = get64(h, 8);
    put64(&mut t, 0, get64(h, 0));
    put32(&mut t, 12, ino as u32);
    put32(&mut t, 16, get32(h, 24));
    put32(&mut t, 20, get64(h, 16) as u32);
    put32(&mut t, 24, get32(h, 28));
    put32(&mut t, 28, get32(h, 32));
    put64(&mut t, 32, get64(h, 40));
    put64(&mut t, size, get64(h, 48));
    put32(&mut t, blksize, get64(h, 56) as u32);
    put64(&mut t, blocks, get64(h, 64));
    for i in 0..6 {
        put32(&mut t, times + 4 * i, get64(h, 72 + 8 * i) as u32);
    }
    put64(&mut t, ino_at, ino);
    t
}

/// The old `struct stat`, with 16-bit IDs and 32-bit sizes.
fn stat32(h: &[u8; 144]) -> [u8; 64] {
    let mut t = [0u8; 64];
    put32(&mut t, 0, get64(h, 0) as u32);
    put32(&mut t, 4, get64(h, 8) as u32);
    let half =
        |t: &mut [u8; 64], at: usize, v: u16| t[at..at + 2].copy_from_slice(&v.to_le_bytes());
    half(&mut t, 8, get32(h, 24) as u16);
    half(&mut t, 10, get64(h, 16) as u16);
    half(&mut t, 12, high2low(get32(h, 28)));
    half(&mut t, 14, high2low(get32(h, 32)));
    put32(&mut t, 16, get64(h, 40) as u32);
    for (i, at) in [48, 56, 64, 72, 80, 88, 96, 104, 112].iter().enumerate() {
        put32(&mut t, 20 + 4 * i, get64(h, *at) as u32);
    }
    t
}

/// `struct statfs` (`wide` false) or `struct statfs64` from the host's.
fn statfs32(h: &[u8; 120], wide: bool) -> Vec<u8> {
    let mut t = vec![0u8; if wide { 84 } else { 64 }];
    put32(&mut t, 0, get64(h, 0) as u32);
    put32(&mut t, 4, get64(h, 8) as u32);
    // f_blocks, f_bfree, f_bavail, f_files and f_ffree.
    let mut at = 8;
    for i in 0..5 {
        let v = get64(h, 16 + 8 * i);
        if wide {
            put64(&mut t, at, v);
            at += 8;
        } else {
            put32(&mut t, at, v as u32);
            at += 4;
        }
    }
    // f_fsid, then f_namelen, f_frsize and f_flags.
    t[at..at + 8].copy_from_slice(&h[56..64]);
    for i in 0..3 {
        put32(&mut t, at + 8 + 4 * i, get64(h, 64 + 8 * i) as u32);
    }
    t
}

/// `struct rusage` with 32-bit times and longs.
fn rusage32(h: &[u8; 144]) -> [u8; 72] {
    let mut t = [0u8; 72];
    for i in 0..18 {
        put32(&mut t, 4 * i, get64(h, 8 * i) as u32);
    }
    t
}

/// `host_to_target_rlim()`: what does not fit an `abi_long` is infinite.
fn rlim32(v: u64) -> u32 {
    if v == u64::MAX || v > i32::MAX as u64 { u32::MAX } else { v as u32 }
}

/// `target_to_host_rlim()`.
fn rlim64(v: u32) -> u64 {
    if v == u32::MAX { u64::MAX } else { u64::from(v) }
}

/// `wait4()` into 32-bit `struct rusage`.
fn wait4(p: &Proc, pid: u64, status: u64, options: u64, ru: u64) -> Res {
    let mut st = 0i32;
    let mut hru = [0u8; 144];
    let hr = if ru != 0 { hru.as_mut_ptr() as u64 } else { 0 };
    let r = host(libc::SYS_wait4, &[pid, (&raw mut st) as u64, options, hr]);
    if r < 0 {
        return Err(r);
    }
    if status != 0 && r != 0 {
        p.put(status, &signal::host_to_target_waitstatus(st).to_le_bytes())?;
    }
    if ru != 0 {
        p.put(ru, &rusage32(&hru))?;
    }
    Ok(r)
}

/// `waitid()`, with the 32-bit `siginfo_t` and `struct rusage`.
fn waitid(p: &Proc, a: [u64; 6]) -> Res {
    let mut info = [0u8; 128];
    let mut hru = [0u8; 144];
    let hr = if a[4] != 0 { hru.as_mut_ptr() as u64 } else { 0 };
    let r = host(libc::SYS_waitid, &[a[0], a[1], info.as_mut_ptr() as u64, a[3], hr]);
    if r < 0 {
        return Err(r);
    }
    if a[2] != 0 {
        p.put(a[2], &signal::guest_siginfo(&signal::host_to_target_siginfo(&info)))?;
    }
    if a[4] != 0 {
        p.put(a[4], &rusage32(&hru))?;
    }
    Ok(r)
}

/// `sigaction()` with the old `struct sigaction`: handler, a one-word mask, flags and
/// restorer.
fn old_sigaction(p: &Proc, t: &mut Task, a: [u64; 6]) -> Res {
    let act = if a[1] != 0 {
        let b: [u8; 16] = read(p, a[1])?;
        let f = |i: usize| u64::from(get32(&b, 4 * i));
        Some(Sigaction { handler: f(0), mask: f(1), flags: f(2), restorer: f(3) })
    } else {
        None
    };
    let old = signal::do_sigaction(t, a[0] as i32, act)?;
    if a[2] != 0 {
        let mut b = [0u8; 16];
        for (i, v) in [old.handler, old.mask, old.flags, old.restorer].into_iter().enumerate() {
            put32(&mut b, 4 * i, v as u32);
        }
        p.put(a[2], &b)?;
    }
    Ok(0)
}

/// `sigprocmask()` with one-word masks, which are read and written as `target_sigset_t`.
fn old_sigprocmask(p: &Proc, t: &mut Task, a: [u64; 6]) -> Res {
    let (how, set) = if a[1] != 0 {
        let b: [u8; 8] = read(p, a[1])?;
        let how = a[0] as i32;
        if !matches!(how, libc::SIG_BLOCK | libc::SIG_UNBLOCK | libc::SIG_SETMASK) {
            return Err(-EINVAL);
        }
        (how, Some(u64::from(get32(&b, 0))))
    } else {
        (0, None)
    };
    let old = signal::old_sigprocmask(t, how, set)?;
    if a[2] != 0 {
        if !p.space().check(a[2], 8, page::WRITE) {
            return Err(-EFAULT);
        }
        p.put(a[2], &(old as u32).to_le_bytes())?;
    }
    Ok(0)
}

/// The `fd_set` of `n` descriptors at `addr`, in the host's longs, `copy_from_user_fdset()`.
/// Bit `i` is in the same byte either way.
fn get_fdset(p: &Proc, addr: u64, n: u64) -> Result<Option<Vec<u64>>, i64> {
    if addr == 0 {
        return Ok(None);
    }
    let mut b = vec![0u8; (n.div_ceil(32) * 4) as usize];
    if !p.space().read(addr, &mut b) {
        return Err(-EFAULT);
    }
    let mut v = vec![0u64; n.div_ceil(64) as usize];
    for (i, w) in v.iter_mut().enumerate() {
        let mut a = [0u8; 8];
        let end = b.len().min(8 * i + 8);
        a[..end - 8 * i].copy_from_slice(&b[8 * i..end]);
        *w = u64::from_le_bytes(a);
    }
    Ok(Some(v))
}

fn put_fdset(p: &Proc, addr: u64, n: u64, v: &Option<Vec<u64>>) -> Result<(), i64> {
    let Some(v) = v else { return Ok(()) };
    let b: Vec<u8> = v.iter().flat_map(|w| w.to_le_bytes()).collect();
    p.put(addr, &b[..(n.div_ceil(32) * 4) as usize])
}

/// `do_select()` and `do_pselect6()`: the three sets of `n` descriptors, the timeout as a
/// 32-bit `timeval` (`kind` 0), `timespec` (1) or 64-bit `timespec` (2), and for
/// `pselect6()` the guest's `{ sigset, size }` of 32-bit words.
fn do_select(p: &Proc, t: &mut Task, a: [u64; 6], kind: u32, pselect: bool) -> Res {
    let n = a[0] as i32;
    if n < 0 {
        return Err(-EINVAL);
    }
    // The kernel takes no more than its descriptor limit, which is below this.
    let n = (n as u64).min(1 << 20);
    let mut sets = [get_fdset(p, a[1], n)?, get_fdset(p, a[2], n)?, get_fdset(p, a[3], n)?];
    let mut ts = match (a[4], kind) {
        (0, _) => None,
        (at, 2) => {
            let b: [u8; 16] = read(p, at)?;
            Some([get64(&b, 0) as i64, get64(&b, 8) as i64])
        }
        (at, _) => Some(get_time32(p, at)?),
    };
    let tsp = ts.as_mut().map_or(0, |v| v.as_mut_ptr() as u64);
    let hs = |s: &mut Option<Vec<u64>>| s.as_mut().map_or(0, |v| v.as_mut_ptr() as u64);
    let h = [n, hs(&mut sets[0]), hs(&mut sets[1]), hs(&mut sets[2]), tsp];
    let r = if pselect {
        let (mut set, mut size) = (0, 0);
        if a[5] != 0 {
            let b: [u8; 8] = read(p, a[5])?;
            (set, size) = (u64::from(get32(&b, 0)), u64::from(get32(&b, 4)));
        }
        let space = Arc::clone(p.space());
        signal::with_sigmask(&space, t, set, size, |m| {
            let ss = [m, 8u64];
            let sp = if a[5] != 0 { ss.as_ptr() as u64 } else { 0 };
            host(libc::SYS_pselect6, &[h[0], h[1], h[2], h[3], h[4], sp])
        })
    } else {
        host(libc::SYS_select, &h)
    };
    if r < 0 {
        return Err(r);
    }
    for (i, s) in sets.iter().enumerate() {
        put_fdset(p, a[1 + i], n, s)?;
    }
    if let Some(ts) = ts {
        match kind {
            2 => {
                let mut b = [0u8; 16];
                put64(&mut b, 0, ts[0] as u64);
                put64(&mut b, 8, ts[1] as u64);
                p.put(a[4], &b)?;
            }
            _ => put_time32(p, a[4], &ts)?,
        }
    }
    Ok(r)
}

/// The i386 `fcntl()` commands that take a `struct flock` (`Some(false)`) or a
/// `struct flock64` (`Some(true)`), and the host command for each.
fn flock_cmd(cmd: u64) -> Option<(u64, bool)> {
    Some(match cmd {
        5..=7 => (cmd, false),
        12..=14 => (cmd - 7, true),
        36..=38 => (cmd, true),
        _ => return None,
    })
}

/// The offsets of `l_start`, `l_len` and `l_pid` in the target's `struct flock`, or with `wide`
/// its `struct flock64`, and the size. i386 packs `struct flock64`; the ARM EABI aligns its
/// 64-bit members to 8 bytes, `struct target_eabi_flock64`.
fn flock_layout(wide: bool, eabi: bool) -> [usize; 4] {
    match (wide, eabi) {
        (false, _) => [4, 8, 12, 16],
        (true, false) => [4, 12, 20, 24],
        (true, true) => [8, 16, 24, 32],
    }
}

/// `fcntl()` and `fcntl64()`: the locks are converted, the rest go to the host as they are.
fn do_fcntl(p: &Arc<Proc>, t: &mut Task, cpu: &mut Cpu<'_>, a: [u64; 6], eabi: bool) -> i64 {
    let Some((cmd, wide)) = flock_cmd(a[1]) else {
        return syscall::do_syscall(p, t, cpu, libc::SYS_fcntl as u64, a);
    };
    let [start_at, len_at, pid_at, size] = flock_layout(wide, eabi);
    let r = (|| {
        let mut fl = [0u8; 32];
        let mut b = [0u8; 32];
        if !p.space().read(a[2], &mut b[..size]) {
            return Err(-EFAULT);
        }
        fl[..4].copy_from_slice(&b[..4]);
        let (start, len, pid) = if wide {
            (get64(&b, start_at), get64(&b, len_at), get32(&b, pid_at))
        } else {
            let start = sx(u64::from(get32(&b, start_at)));
            (start, sx(u64::from(get32(&b, len_at))), get32(&b, pid_at))
        };
        put64(&mut fl, 8, start);
        put64(&mut fl, 16, len);
        put32(&mut fl, 24, pid);
        let r = host(libc::SYS_fcntl, &[a[0], cmd, fl.as_mut_ptr() as u64]);
        if r < 0 {
            return Err(r);
        }
        if cmd == 5 || cmd == 36 {
            b[..4].copy_from_slice(&fl[..4]);
            if wide {
                put64(&mut b, start_at, get64(&fl, 8));
                put64(&mut b, len_at, get64(&fl, 16));
            } else {
                put32(&mut b, start_at, get64(&fl, 8) as u32);
                put32(&mut b, len_at, get64(&fl, 16) as u32);
            }
            put32(&mut b, pid_at, get32(&fl, 24));
            p.put(a[2], &b[..size])?;
        }
        Ok(r)
    })();
    r.unwrap_or_else(|e| e)
}

/// `getdents()` into the 32-bit `struct linux_dirent`, from the host's `getdents64()`, whose
/// records are never smaller.
fn getdents(p: &Proc, fd: u64, dirp: u64, count: u64) -> Res {
    let count = count.min(1 << 20);
    p.buf(dirp, count, page::WRITE)?;
    let mut h = vec![0u8; count as usize];
    let r = host(libc::SYS_getdents64, &[fd, h.as_mut_ptr() as u64, count]);
    if r < 0 {
        return Err(r);
    }
    let mut out = Vec::with_capacity(r as usize);
    let mut at = 0;
    while at < r as usize {
        let reclen = usize::from(u16::from_le_bytes([h[at + 16], h[at + 17]]));
        let name = &h[at + 19..at + reclen];
        let name = &name[..name.iter().position(|&c| c == 0).unwrap_or(name.len())];
        let treclen = (10 + name.len() + 2).next_multiple_of(4);
        let start = out.len();
        out.resize(start + treclen, 0);
        let rec = &mut out[start..];
        put32(rec, 0, get64(&h, at) as u32);
        put32(rec, 4, get64(&h, at + 8) as u32);
        rec[8..10].copy_from_slice(&(treclen as u16).to_le_bytes());
        rec[10..10 + name.len()].copy_from_slice(name);
        rec[treclen - 1] = h[at + 18];
        at += reclen;
    }
    p.put(dirp, &out)?;
    Ok(out.len() as i64)
}

/// `sched_getaffinity()` and `sched_setaffinity()` with masks of 32-bit longs.
fn affinity(p: &Proc, set: bool, a: [u64; 6]) -> Res {
    let len = a[1];
    if !set && len & 3 != 0 {
        return Err(-EINVAL);
    }
    let size = len.next_multiple_of(8).min(1 << 16);
    let mut m = vec![0u8; size as usize];
    if set {
        let n = len.min(size) as usize;
        if !p.space().read(a[2], &mut m[..n]) {
            return Err(-EFAULT);
        }
        let r = host(libc::SYS_sched_setaffinity, &[a[0], size, m.as_ptr() as u64]);
        return if r < 0 { Err(r) } else { Ok(r) };
    }
    let r = host(libc::SYS_sched_getaffinity, &[a[0], size, m.as_mut_ptr() as u64]);
    if r < 0 {
        return Err(r);
    }
    // Only a mask of an odd number of 32-bit words is shorter than the host's: what the
    // host adds is unused unless there are more CPUs than the guest asked about.
    let mut r = r as u64;
    if r > len {
        if crate::host::configured_cpus() > len * 8 {
            return Err(-EINVAL);
        }
        r = len;
    }
    p.put(a[2], &m[..r as usize])?;
    Ok(r as i64)
}

/// `sysinfo()` into the 32-bit `struct sysinfo`.
fn sysinfo(p: &Proc, addr: u64) -> Res {
    let mut h = [0u8; 112];
    let r = host(libc::SYS_sysinfo, &[h.as_mut_ptr() as u64]);
    if r < 0 {
        return Err(r);
    }
    let mut t = [0u8; 64];
    for i in 0..10 {
        put32(&mut t, 4 * i, get64(&h, 8 * i) as u32);
    }
    t[40..42].copy_from_slice(&h[80..82]);
    put32(&mut t, 44, get64(&h, 88) as u32);
    put32(&mut t, 48, get64(&h, 96) as u32);
    put32(&mut t, 52, get32(&h, 104));
    p.put(addr, &t)?;
    Ok(r)
}

/// A call with one 32-bit time structure of `n` words read at argument `i`, passed to the
/// host's call `nr` in its 64-bit form with the other arguments as `spec` says.
fn time_in(p: &Proc, nr: i64, mut a: [u64; 6], i: usize, n: usize, spec: &[A]) -> Res {
    let ts = get_times32(p, a[i], n)?;
    a[i] = ptr_or_null(&ts);
    let r = p.generic(nr, a, spec);
    if r < 0 { Err(r) } else { Ok(r) }
}

/// A call that writes a time structure of `n` words at argument `i`.
fn time_out(p: &Proc, nr: i64, mut a: [u64; 6], i: usize, n: usize, spec: &[A]) -> Res {
    let at = a[i];
    let mut ts = vec![0i64; n];
    if at != 0 {
        a[i] = ts.as_mut_ptr() as u64;
    }
    let r = p.generic(nr, a, spec);
    if r < 0 {
        return Err(r);
    }
    if at != 0 {
        put_time32(p, at, &ts)?;
    }
    Ok(r)
}

/// `futex()` with a 32-bit `struct timespec`, for the operations whose fourth argument is
/// one.
fn futex(p: &Arc<Proc>, t: &mut Task, cpu: &mut Cpu<'_>, a: [u64; 6]) -> Res {
    let s: &[A] = match a[1] & 0x7f {
        0 | 9 | 6 | 13 => &[M(4), V, V, V, V, V],
        11 => &[M(4), V, V, V, M(4), V],
        _ => return Ok(syscall::do_syscall(p, t, cpu, libc::SYS_futex as u64, a)),
    };
    time_in(p, libc::SYS_futex, a, 3, 2, s)
}

/// `ppoll()` with a 32-bit timeout, which it writes back.
fn ppoll(p: &Proc, t: &mut Task, a: [u64; 6]) -> Res {
    let mut ts = get_times32(p, a[2], 2)?;
    let tsp = ts.as_mut().map_or(0, |v| v.as_mut_ptr() as u64);
    let space = Arc::clone(p.space());
    let r = signal::with_sigmask(&space, t, a[3], a[4], |m| {
        p.generic(libc::SYS_ppoll, [a[0], a[1], tsp, m, 8, 0], &[MN(1, 8), V, V, V, V])
    });
    if r < 0 {
        return Err(r);
    }
    if let Some(ts) = ts {
        put_time32(p, a[2], &ts)?;
    }
    Ok(r)
}

/// The system call `name` of a 32-bit guest, whose host number is `host_nr` when it goes to
/// the host as it is.
pub(crate) fn do_syscall(
    p: &Arc<Proc>,
    t: &mut Task,
    cpu: &mut Cpu<'_>,
    name: &str,
    host_nr: Option<i64>,
    a: [u64; 6],
) -> i64 {
    let eabi = crate::guest::guest().eabi;
    let r = (|| -> Res {
        Ok(match name {
            "waitpid" => return wait4(p, a[0], a[1], a[2], 0),
            "wait4" => return wait4(p, a[0], a[1], a[2], a[3]),
            "time" => {
                let r = host(libc::SYS_time, &[0]);
                if a[0] != 0 {
                    p.put(a[0], &(r as u32).to_le_bytes())?;
                }
                r
            }
            "lseek" => call(p, t, cpu, libc::SYS_lseek, [a[0], sx(a[1]), a[2], 0, 0, 0]),
            "_llseek" => {
                let r = host(libc::SYS_lseek, &[a[0], pair(a[2], a[1]), a[4]]);
                if r < 0 {
                    return Err(r);
                }
                p.put(a[3], &r.to_le_bytes())?;
                0
            }
            "truncate" => call(p, t, cpu, libc::SYS_truncate, [a[0], sx(a[1]), 0, 0, 0, 0]),
            "ftruncate" => call(p, t, cpu, libc::SYS_ftruncate, [a[0], sx(a[1]), 0, 0, 0, 0]),
            "truncate64" | "ftruncate64" => {
                let n = if name == "truncate64" { libc::SYS_truncate } else { libc::SYS_ftruncate };
                let (len, _) = arg64(&a, 1, eabi);
                call(p, t, cpu, n, [a[0], len, 0, 0, 0, 0])
            }
            "pread64" | "pwrite64" => {
                let n = if name == "pread64" { libc::SYS_pread64 } else { libc::SYS_pwrite64 };
                let (pos, _) = arg64(&a, 3, eabi);
                call(p, t, cpu, n, [a[0], a[1], a[2], pos, 0, 0])
            }
            "readahead" => {
                let (off, i) = arg64(&a, 1, eabi);
                call(p, t, cpu, libc::SYS_readahead, [a[0], off, a[i], 0, 0, 0])
            }
            // The ARM calls that take the 32-bit argument before the 64-bit ones, so that
            // these need no padding.
            "arm_fadvise64_64" => {
                let h = [a[0], pair(a[2], a[3]), pair(a[4], a[5]), a[1], 0, 0];
                call(p, t, cpu, libc::SYS_fadvise64, h)
            }
            "arm_sync_file_range" => {
                let h = [a[0], pair(a[2], a[3]), pair(a[4], a[5]), a[1], 0, 0];
                call(p, t, cpu, libc::SYS_sync_file_range, h)
            }
            "sync_file_range" => {
                let h = [a[0], pair(a[1], a[2]), pair(a[3], a[4]), a[5], 0, 0];
                call(p, t, cpu, libc::SYS_sync_file_range, h)
            }
            "fadvise64" => {
                call(p, t, cpu, libc::SYS_fadvise64, [a[0], pair(a[1], a[2]), a[3], a[4], 0, 0])
            }
            "fadvise64_64" => {
                let h = [a[0], pair(a[1], a[2]), pair(a[3], a[4]), a[5], 0, 0];
                call(p, t, cpu, libc::SYS_fadvise64, h)
            }
            "fallocate" => {
                let (off, i) = arg64(&a, 2, eabi);
                let (len, _) = arg64(&a, i, eabi);
                call(p, t, cpu, libc::SYS_fallocate, [a[0], a[1], off, len, 0, 0])
            }
            "send" => call(p, t, cpu, libc::SYS_sendto, [a[0], a[1], a[2], a[3], 0, 0]),
            "recv" => call(p, t, cpu, libc::SYS_recvfrom, [a[0], a[1], a[2], a[3], 0, 0]),
            // The EABI `struct epoll_event` is not the host's, which `epoll_pwait()` converts.
            "epoll_wait" if eabi => {
                call(p, t, cpu, libc::SYS_epoll_pwait, [a[0], a[1], a[2], a[3], 0, 0])
            }
            "readv" | "writev" => {
                let n = if name == "readv" { libc::SYS_readv } else { libc::SYS_writev };
                call(p, t, cpu, n, a)
            }
            "preadv" | "pwritev" | "preadv2" | "pwritev2" => {
                let n = match name {
                    "preadv" => libc::SYS_preadv,
                    "pwritev" => libc::SYS_pwritev,
                    "preadv2" => libc::SYS_preadv2,
                    _ => libc::SYS_pwritev2,
                };
                call(p, t, cpu, n, [a[0], a[1], a[2], pair(a[3], a[4]), 0, a[5]])
            }
            "sendfile" => {
                let mut off = 0i64;
                if a[2] != 0 {
                    off = i64::from(p.get_u32(a[2])? as i32);
                }
                let op = if a[2] != 0 { (&raw mut off) as u64 } else { 0 };
                let r = host(libc::SYS_sendfile, &[a[0], a[1], op, a[3]]);
                if r >= 0 && a[2] != 0 {
                    p.put(a[2], &(off as u32).to_le_bytes())?;
                }
                r
            }
            "mmap" => {
                let b: [u8; 24] = read(p, a[0])?;
                let v: Vec<u64> = (0..6).map(|i| u64::from(get32(&b, 4 * i))).collect();
                call(p, t, cpu, libc::SYS_mmap, [v[0], v[1], v[2], v[3], v[4], v[5]])
            }
            "mmap2" => call(p, t, cpu, libc::SYS_mmap, [a[0], a[1], a[2], a[3], a[4], a[5] << 12]),
            "stat" | "lstat" | "fstat" | "stat64" | "lstat64" | "fstat64" | "fstatat64" => {
                const AT_SYMLINK_NOFOLLOW: u64 = 0x100;
                const AT_FDCWD: u64 = libc::AT_FDCWD as u64;
                let (st, out) = match name {
                    "stat" | "stat64" => (host_stat(p, AT_FDCWD, Some(a[0]), 0)?, a[1]),
                    "lstat" | "lstat64" => {
                        (host_stat(p, AT_FDCWD, Some(a[0]), AT_SYMLINK_NOFOLLOW)?, a[1])
                    }
                    "fstat" | "fstat64" => (host_stat(p, a[0], None, 0)?, a[1]),
                    _ => (host_stat(p, sx(a[0]), Some(a[1]), a[3])?, a[2]),
                };
                if name.ends_with('4') {
                    p.put(out, &stat64(&st, eabi))?;
                } else {
                    p.put(out, &stat32(&st))?;
                }
                0
            }
            "statfs" | "fstatfs" | "statfs64" | "fstatfs64" => {
                let mut h = [0u8; 120];
                let hp = h.as_mut_ptr() as u64;
                let r = if name.starts_with('f') {
                    host(libc::SYS_fstatfs, &[a[0], hp])
                } else {
                    p.generic(libc::SYS_statfs, [a[0], hp, 0, 0, 0, 0], &[P, V])
                };
                if r < 0 {
                    return Err(r);
                }
                let wide = name.ends_with("64");
                p.put(if wide { a[2] } else { a[1] }, &statfs32(&h, wide))?;
                r
            }
            "getdents" => return getdents(p, a[0], a[1], a[2]),
            "fcntl" | "fcntl64" => do_fcntl(p, t, cpu, a, eabi),
            "clone" => call(p, t, cpu, libc::SYS_clone, [a[0], a[1], a[2], a[4], a[3], 0]),
            "execve" => call(p, t, cpu, libc::SYS_execve, a),
            "execveat" => {
                call(p, t, cpu, libc::SYS_execveat, [sx(a[0]), a[1], a[2], a[3], a[4], 0])
            }
            "eventfd" => call(p, t, cpu, libc::SYS_eventfd2, [a[0], 0, 0, 0, 0, 0]),
            "select" => {
                let b: [u8; 20] = read(p, a[0])?;
                let v: Vec<u64> = (0..5).map(|i| u64::from(get32(&b, 4 * i))).collect();
                return do_select(p, t, [v[0], v[1], v[2], v[3], v[4], 0], 0, false);
            }
            "_newselect" => return do_select(p, t, a, 0, false),
            "pselect6" => return do_select(p, t, a, 1, true),
            "pselect6_time64" => return do_select(p, t, a, 2, true),
            "ppoll" => return ppoll(p, t, a),
            "futex" => return futex(p, t, cpu, a),
            "nanosleep" => {
                let req = get_time32(p, a[0])?;
                let mut rem = [0i64; 2];
                let r = host(libc::SYS_nanosleep, &[req.as_ptr() as u64, rem.as_mut_ptr() as u64]);
                if r < 0 && a[1] != 0 {
                    put_time32(p, a[1], &rem)?;
                }
                r
            }
            "clock_nanosleep" => {
                const TIMER_ABSTIME: u64 = 1;
                let mut ts = get_time32(p, a[2])?;
                let tp = ts.as_mut_ptr() as u64;
                let r = host(
                    libc::SYS_clock_nanosleep,
                    &[a[0], a[1], tp, if a[3] != 0 { tp } else { 0 }],
                );
                if r == -i64::from(libc::EINTR) && a[3] != 0 && a[1] != TIMER_ABSTIME {
                    put_time32(p, a[3], &ts)?;
                }
                r
            }
            "clock_gettime" => return time_out(p, libc::SYS_clock_gettime, a, 1, 2, &[V, V]),
            "clock_getres" => return time_out(p, libc::SYS_clock_getres, a, 1, 2, &[V, V]),
            "clock_settime" => return time_in(p, libc::SYS_clock_settime, a, 1, 2, &[V, V]),
            "sched_rr_get_interval" => {
                return time_out(p, libc::SYS_sched_rr_get_interval, a, 1, 2, &[V, V]);
            }
            "gettimeofday" => {
                return time_out(p, libc::SYS_gettimeofday, a, 0, 2, &[V, M(8)]);
            }
            "settimeofday" => return time_in(p, libc::SYS_settimeofday, a, 0, 2, &[V, R(8)]),
            "getitimer" => return time_out(p, libc::SYS_getitimer, a, 1, 4, &[V, V]),
            "setitimer" => {
                let new = get_times32(p, a[1], 4)?;
                let mut old = [0i64; 4];
                let op = if a[2] != 0 { old.as_mut_ptr() as u64 } else { 0 };
                let r = host(libc::SYS_setitimer, &[a[0], ptr_or_null(&new), op]);
                if r >= 0 && a[2] != 0 {
                    put_time32(p, a[2], &old)?;
                }
                r
            }
            "timerfd_gettime" => return time_out(p, libc::SYS_timerfd_gettime, a, 1, 4, &[V, V]),
            "timerfd_settime" => {
                let new = get_times32(p, a[2], 4)?;
                let mut old = [0i64; 4];
                let op = if a[3] != 0 { old.as_mut_ptr() as u64 } else { 0 };
                let r = host(libc::SYS_timerfd_settime, &[a[0], a[1], ptr_or_null(&new), op]);
                if r >= 0 && a[3] != 0 {
                    put_time32(p, a[3], &old)?;
                }
                r
            }
            "utime" => {
                let tb = match a[1] {
                    0 => None,
                    at => {
                        let b: [u8; 8] = read(p, at)?;
                        Some(vec![i64::from(get32(&b, 0) as i32), i64::from(get32(&b, 4) as i32)])
                    }
                };
                p.generic(libc::SYS_utime, [a[0], ptr_or_null(&tb), 0, 0, 0, 0], &[S, V])
            }
            "utimes" => return time_in(p, libc::SYS_utimes, a, 1, 4, &[S, V]),
            "futimesat" => return time_in(p, libc::SYS_futimesat, a, 2, 4, &[V, S, V]),
            "utimensat" => return time_in(p, libc::SYS_utimensat, a, 2, 4, &[V, S, V, V]),
            "times" => {
                let mut tms = [0i64; 4];
                let r = host(libc::SYS_times, &[tms.as_mut_ptr() as u64]);
                if r >= 0 && a[0] != 0 {
                    put_time32(p, a[0], &tms)?;
                }
                r
            }
            "getrusage" => {
                let mut ru = [0u8; 144];
                let r = host(libc::SYS_getrusage, &[a[0], ru.as_mut_ptr() as u64]);
                if r >= 0 {
                    p.put(a[1], &rusage32(&ru))?;
                }
                r
            }
            "sysinfo" => return sysinfo(p, a[0]),
            "getrlimit" | "ugetrlimit" => {
                let mut rl = [0u64; 2];
                let r = host(libc::SYS_getrlimit, &[a[0], rl.as_mut_ptr() as u64]);
                if r >= 0 {
                    let mut b = [0u8; 8];
                    put32(&mut b, 0, rlim32(rl[0]));
                    put32(&mut b, 4, rlim32(rl[1]));
                    p.put(a[1], &b)?;
                }
                r
            }
            "setrlimit" => {
                let b: [u8; 8] = read(p, a[1])?;
                if syscall::ignored_rlimit(a[0]) {
                    return Ok(0);
                }
                let rl = [rlim64(get32(&b, 0)), rlim64(get32(&b, 4))];
                host(libc::SYS_setrlimit, &[a[0], rl.as_ptr() as u64])
            }
            "sched_getaffinity" => return affinity(p, false, a),
            "sched_setaffinity" => return affinity(p, true, a),
            "lchown" | "chown" | "fchown" => {
                let n = match name {
                    "lchown" => libc::SYS_lchown,
                    "chown" => libc::SYS_chown,
                    _ => libc::SYS_fchown,
                };
                call(p, t, cpu, n, [a[0], low2high(a[1]), low2high(a[2]), 0, 0, 0])
            }
            "setuid" | "setgid" => {
                let n = if name == "setuid" { libc::SYS_setuid } else { libc::SYS_setgid };
                call(p, t, cpu, n, [low2high(a[0]), 0, 0, 0, 0, 0])
            }
            "setreuid" | "setregid" => {
                let n = if name == "setreuid" { libc::SYS_setreuid } else { libc::SYS_setregid };
                call(p, t, cpu, n, [low2high(a[0]), low2high(a[1]), 0, 0, 0, 0])
            }
            "setresuid" | "setresgid" => {
                let n = if name == "setresuid" { libc::SYS_setresuid } else { libc::SYS_setresgid };
                call(p, t, cpu, n, [low2high(a[0]), low2high(a[1]), low2high(a[2]), 0, 0, 0])
            }
            "getuid" | "getgid" | "geteuid" | "getegid" => {
                let n = match name {
                    "getuid" => libc::SYS_getuid,
                    "getgid" => libc::SYS_getgid,
                    "geteuid" => libc::SYS_geteuid,
                    _ => libc::SYS_getegid,
                };
                i64::from(high2low(host(n, &[]) as u32))
            }
            "getresuid" | "getresgid" => {
                let n = if name == "getresuid" { libc::SYS_getresuid } else { libc::SYS_getresgid };
                let mut ids = [0u32; 3];
                let r = host(
                    n,
                    &[
                        ids.as_mut_ptr() as u64,
                        ids.as_mut_ptr() as u64 + 4,
                        ids.as_mut_ptr() as u64 + 8,
                    ],
                );
                if r >= 0 {
                    for (i, id) in ids.iter().enumerate() {
                        p.put(a[i], &high2low(*id).to_le_bytes())?;
                    }
                }
                r
            }
            "getgroups" => {
                let n = a[0] as i32;
                if !(0..=65536).contains(&n) {
                    return Err(-EINVAL);
                }
                let mut g = vec![0u32; n as usize];
                let r = host(libc::SYS_getgroups, &[n as u64, g.as_mut_ptr() as u64]);
                if r > 0 && n != 0 {
                    let b: Vec<u8> =
                        g[..r as usize].iter().flat_map(|id| high2low(*id).to_le_bytes()).collect();
                    p.put(a[1], &b)?;
                }
                r
            }
            "setgroups" => {
                let n = a[0] as i32;
                if !(0..=65536).contains(&n) {
                    return Err(-EINVAL);
                }
                let mut b = vec![0u8; 2 * n as usize];
                if !p.space().read(a[1], &mut b) {
                    return Err(-EFAULT);
                }
                let g: Vec<u32> = b
                    .chunks_exact(2)
                    .map(|c| low2high(u64::from(u16::from_le_bytes([c[0], c[1]]))) as u32)
                    .collect();
                host(libc::SYS_setgroups, &[n as u64, g.as_ptr() as u64])
            }
            "waitid" => return waitid(p, a),
            "sigaction" => return old_sigaction(p, t, a),
            "sigprocmask" => return old_sigprocmask(p, t, a),
            "sgetmask" | "ssetmask" => {
                let (how, set) =
                    if name == "sgetmask" { (0, None) } else { (libc::SIG_SETMASK, Some(a[0])) };
                return signal::old_sigprocmask(t, how, set).map(|m| i64::from(m as u32 as i32));
            }
            "sigpending" => {
                let mut set = 0u64;
                let r = host(libc::SYS_rt_sigpending, &[(&raw mut set) as u64, 8]);
                if r < 0 {
                    return Err(r);
                }
                if !p.space().check(a[0], 8, page::WRITE) {
                    return Err(-EFAULT);
                }
                p.put(a[0], &(signal::h2t_set(set) as u32).to_le_bytes())?;
                r
            }
            // QEMU reads a whole target_sigset_t at the first argument, rather than the mask
            // in the third the kernel takes.
            "sigsuspend" => signal::sigsuspend(p.space(), t, a[0], 8),
            "rt_sigtimedwait" | "rt_sigtimedwait_time64" => {
                if a[3] != 8 {
                    return Err(-EINVAL);
                }
                let mut ts = [0u8; 16];
                if a[2] != 0 {
                    if name == "rt_sigtimedwait" {
                        let v = get_time32(p, a[2])?;
                        put64(&mut ts, 0, v[0] as u64);
                        put64(&mut ts, 8, v[1] as u64);
                    } else if !p.space().read(a[2], &mut ts) {
                        return Err(-EFAULT);
                    }
                }
                signal::sigtimedwait(p.space(), a[0], a[1], (a[2] != 0).then_some(&ts), a[3])
            }
            "rt_sigreturn" => call(p, t, cpu, libc::SYS_rt_sigreturn, a),
            "rt_sigaction" => call(p, t, cpu, libc::SYS_rt_sigaction, a),
            "rt_sigqueueinfo" => call(p, t, cpu, libc::SYS_rt_sigqueueinfo, a),
            "rt_tgsigqueueinfo" => call(p, t, cpu, libc::SYS_rt_tgsigqueueinfo, a),
            "sigaltstack" => call(p, t, cpu, libc::SYS_sigaltstack, a),
            _ => match host_nr {
                Some(n) => call(p, t, cpu, n, a),
                None => -ENOSYS,
            },
        })
    })();
    r.unwrap_or_else(|e| e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eabi_register_pairs_start_even() {
        let a = [1, 2, 3, 4, 5, 6];
        // pread64(fd, buf, count, pos): the i386 pair is R3 and R4, the EABI one R4 and R5.
        assert_eq!(arg64(&a, 3, false), (4 | 5 << 32, 5));
        assert_eq!(arg64(&a, 3, true), (5 | 6 << 32, 6));
        // An even index stays where it is.
        assert_eq!(arg64(&a, 2, true), (3 | 4 << 32, 4));
        assert_eq!(pair(0xffff_ffff_8000_0000, 1), 0x1_8000_0000);
    }

    #[test]
    fn stat64_layouts() {
        let mut h = [0u8; 144];
        put64(&mut h, 8, 0x1_0000_0002);
        put64(&mut h, 48, 0x1234_5678_9abc);
        put64(&mut h, 56, 4096);
        put64(&mut h, 64, 77);
        put64(&mut h, 72, 1000);
        put64(&mut h, 104, 3000);
        let i386 = stat64(&h, false);
        assert_eq!(i386.len(), 96);
        assert_eq!(get64(&i386, 44), 0x1234_5678_9abc);
        assert_eq!(get32(&i386, 52), 4096);
        assert_eq!(get64(&i386, 56), 77);
        assert_eq!(get32(&i386, 64), 1000);
        assert_eq!(get64(&i386, 88), 0x1_0000_0002);
        let eabi = stat64(&h, true);
        assert_eq!(eabi.len(), 104);
        assert_eq!(get32(&eabi, 12), 2);
        assert_eq!(get64(&eabi, 48), 0x1234_5678_9abc);
        assert_eq!(get32(&eabi, 56), 4096);
        assert_eq!(get64(&eabi, 64), 77);
        assert_eq!(get32(&eabi, 72), 1000);
        assert_eq!(get32(&eabi, 88), 3000);
        assert_eq!(get64(&eabi, 96), 0x1_0000_0002);
    }

    #[test]
    fn flock64_layouts() {
        assert_eq!(flock_layout(false, true), flock_layout(false, false));
        assert_eq!(flock_layout(true, false), [4, 12, 20, 24]);
        assert_eq!(flock_layout(true, true), [8, 16, 24, 32]);
    }
}
