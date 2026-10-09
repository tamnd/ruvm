// SPDX-License-Identifier: GPL-2.0-or-later

//! The one door to the host kernel: a raw system call.
//!
//! Every guest system call that the emulator does not answer itself ends up here with the
//! guest's arguments, pointers already turned into host addresses. The callers keep one
//! invariant, and it is what makes this sound: a pointer argument is either 0 or the host
//! address of a guest range that [`GuestSpace::check`](ruvm_user_common::GuestSpace::check)
//! accepted for the access the call makes, with the length the call is given, or a live Rust
//! buffer at least as big as the call reads or writes. The guest reservation stays mapped
//! readable and writable for the life of the process, so the kernel only ever reads or writes
//! memory of that mapping, which Rust sees as `AtomicU8`s, or memory Rust handed it.

/// `syscall(nr, a0, ..., a5)`, returning the raw result: a value, or `-errno`.
pub(crate) fn syscall(nr: i64, a: [u64; 6]) -> i64 {
    // SAFETY: see the module documentation. The pointer arguments are 0, point into the
    // guest reservation, which is always mapped and only seen through atomics, or point to a
    // live Rust buffer, for as many bytes as the call touches; the callers check this before
    // they get here. Calls that
    // change the process itself (clone without CLONE_VM, execve, exit) do not touch Rust's
    // memory either: the child of a fork gets a copy of it.
    let r = unsafe { libc::syscall(nr, a[0], a[1], a[2], a[3], a[4], a[5]) };
    if r == -1 {
        -i64::from(std::io::Error::last_os_error().raw_os_error().unwrap_or(libc::ENOSYS))
    } else {
        r
    }
}

/// [`syscall`] with fewer arguments.
pub(crate) fn sys(nr: i64, args: &[u64]) -> i64 {
    let mut a = [0u64; 6];
    a[..args.len()].copy_from_slice(args);
    syscall(nr, a)
}

/// `getrlimit(RLIMIT_STACK)`, when it is finite.
pub(crate) fn stack_rlimit() -> Option<u64> {
    let mut lim = [0u64; 2];
    let r = sys(libc::SYS_prlimit64, &[0, libc::RLIMIT_STACK as u64, 0, lim.as_mut_ptr() as u64]);
    (r == 0 && lim[0] != libc::RLIM_INFINITY).then_some(lim[0])
}

/// Sets the host disposition of `sig` to `SIG_DFL` or `SIG_IGN` (`handler` 0 or 1).
pub(crate) fn set_disposition(sig: i32, handler: u64) -> i64 {
    // struct kernel_sigaction: handler, flags, restorer, mask.
    let act = [handler, 0u64, 0, 0];
    sys(libc::SYS_rt_sigaction, &[sig as u64, act.as_ptr() as u64, 0, 8])
}

/// Undoes Rust's `SIG_IGN` for `SIGPIPE`, which a program would otherwise inherit.
pub(crate) fn reset_sigpipe() {
    set_disposition(libc::SIGPIPE, 0);
}

/// 16 random bytes for `AT_RANDOM`.
pub(crate) fn random16() -> [u8; 16] {
    let mut b = [0u8; 16];
    let _ = sys(libc::SYS_getrandom, &[b.as_mut_ptr() as u64, 16, 0]);
    b
}

/// The end of `dump_core_and_abort()`: no host core dump, then `sig` with its default action.
pub(crate) fn die_with_signal(sig: i32) -> ! {
    let core = libc::RLIMIT_CORE as u64;
    let mut lim = [0u64; 2];
    if sys(libc::SYS_prlimit64, &[0, core, 0, lim.as_mut_ptr() as u64]) == 0 {
        lim[0] = 0;
        sys(libc::SYS_prlimit64, &[0, core, lim.as_ptr() as u64, 0]);
    }
    // A pipe in core_pattern ignores RLIMIT_CORE; not being dumpable does not.
    sys(libc::SYS_prctl, &[libc::PR_SET_DUMPABLE as u64, 0]);
    set_disposition(sig, 0);
    let mask = 1u64 << (sig - 1);
    sys(libc::SYS_rt_sigprocmask, &[libc::SIG_UNBLOCK as u64, (&raw const mask) as u64, 0, 8]);
    let pid = sys(libc::SYS_getpid, &[]);
    sys(libc::SYS_kill, &[pid as u64, sig as u64]);
    std::process::abort()
}
