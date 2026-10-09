// SPDX-License-Identifier: GPL-2.0-or-later

//! The one door to the host kernel: a raw system call, and the host signal handler.
//!
//! Every guest system call that the emulator does not answer itself ends up here with the
//! guest's arguments, pointers already turned into host addresses. The callers keep one
//! invariant, and it is what makes this sound: a pointer argument is either 0 or the host
//! address of a guest range that [`GuestSpace::check`](ruvm_user_common::GuestSpace::check)
//! accepted for the access the call makes, with the length the call is given, or a live Rust
//! buffer at least as big as the call reads or writes. The guest reservation stays mapped
//! readable and writable for the life of the process, so the kernel only ever reads or writes
//! memory of that mapping, which Rust sees as `AtomicU8`s, or memory Rust handed it.
//!
//! The system call instruction lives in `safe_syscall_base`, `safe-syscall.inc.S` of QEMU: it
//! checks the thread's pending signal flag and only then enters the kernel. A host signal
//! that lands between the check and the instruction rewinds the thread to the check, so a
//! guest signal that arrives just before a blocking call can not be lost while the call
//! sleeps; the call returns `-ERESTARTSYS` instead and the cpu loop restarts it once the
//! signal is delivered.

use std::sync::atomic::AtomicU32;

// safe_syscall_base(pending, nr, a0, a1, a2, a3, a4, a5): the System V arguments move to the
// kernel's registers, %rbp keeps the flag's address, and the flag is checked in the window
// [start, end) that the signal handler rewinds.
std::arch::global_asm!(
    ".pushsection .text.ruvm_lu_safe_syscall,\"ax\",@progbits",
    ".globl ruvm_lu_safe_syscall_base",
    ".hidden ruvm_lu_safe_syscall_base",
    ".globl ruvm_lu_safe_syscall_start",
    ".hidden ruvm_lu_safe_syscall_start",
    ".globl ruvm_lu_safe_syscall_end",
    ".hidden ruvm_lu_safe_syscall_end",
    ".globl ruvm_lu_sigreturn",
    ".hidden ruvm_lu_sigreturn",
    ".type ruvm_lu_safe_syscall_base, @function",
    "ruvm_lu_safe_syscall_base:",
    ".cfi_startproc",
    "push %rbp",
    ".cfi_adjust_cfa_offset 8",
    ".cfi_rel_offset %rbp, 0",
    "mov %rsi, %rax",
    "mov %rdi, %rbp",
    "mov %rdx, %rdi",
    "mov %rcx, %rsi",
    "mov %r8, %rdx",
    "mov %r9, %r10",
    "mov 16(%rsp), %r8",
    "mov 24(%rsp), %r9",
    "ruvm_lu_safe_syscall_start:",
    "cmpl $0, (%rbp)",
    "jnz 2f",
    "syscall",
    "ruvm_lu_safe_syscall_end:",
    "pop %rbp",
    ".cfi_remember_state",
    ".cfi_adjust_cfa_offset -8",
    ".cfi_restore %rbp",
    "ret",
    ".cfi_restore_state",
    "2:",
    "mov $-512, %rax",
    "pop %rbp",
    ".cfi_adjust_cfa_offset -8",
    ".cfi_restore %rbp",
    "ret",
    ".cfi_endproc",
    ".size ruvm_lu_safe_syscall_base, .-ruvm_lu_safe_syscall_base",
    // The sa_restorer of the host handler: rt_sigreturn.
    ".type ruvm_lu_sigreturn, @function",
    "ruvm_lu_sigreturn:",
    "mov $15, %eax",
    "syscall",
    ".size ruvm_lu_sigreturn, .-ruvm_lu_sigreturn",
    ".popsection",
    options(att_syntax)
);

// SAFETY: the declarations match the assembly above: a System V function of eight integer
// arguments returning one, and three code labels that are only ever compared or handed to the
// kernel, never read through.
unsafe extern "C" {
    fn ruvm_lu_safe_syscall_base(
        pending: *const AtomicU32,
        nr: i64,
        a0: u64,
        a1: u64,
        a2: u64,
        a3: u64,
        a4: u64,
        a5: u64,
    ) -> i64;
    safe static ruvm_lu_safe_syscall_start: u8;
    safe static ruvm_lu_safe_syscall_end: u8;
    safe static ruvm_lu_sigreturn: u8;
}

/// A flag that is never set, for the emulator's own system calls.
static NEVER: AtomicU32 = AtomicU32::new(0);

/// `safe_syscall(nr, a0, ..., a5)`: the system call, unless `pending` is or becomes set before
/// the thread enters the kernel, which returns `-ERESTARTSYS`. The result is a value or
/// `-errno`.
pub(crate) fn safe_syscall(pending: &AtomicU32, nr: i64, a: [u64; 6]) -> i64 {
    // SAFETY: see the module documentation. The pointer arguments are 0, point into the
    // guest reservation, which is always mapped and only seen through atomics, or point to a
    // live Rust buffer, for as many bytes as the call touches; the callers check this before
    // they get here. Calls that change the process itself (clone without CLONE_VM, execve,
    // exit) do not touch Rust's memory either: the child of a fork gets a copy of it.
    // `pending` is a live atomic the assembly only reads.
    unsafe { ruvm_lu_safe_syscall_base(pending, nr, a[0], a[1], a[2], a[3], a[4], a[5]) }
}

/// `fork()` of the C library, which also makes its allocator usable in the child when other
/// threads were inside it. The child's pid in the parent, 0 in the child, or `-errno`.
pub(crate) fn fork() -> i64 {
    // SAFETY: fork() has no arguments and touches no Rust memory itself. The child goes on
    // with a copy of this thread only; the callers hold the emulator's locks that another
    // thread could otherwise have left taken, and no other vCPU runs generated code.
    let r = unsafe { libc::fork() };
    if r < 0 {
        -i64::from(std::io::Error::last_os_error().raw_os_error().unwrap_or(0))
    } else {
        i64::from(r)
    }
}

/// `syscall(nr, a0, ..., a5)`, returning the raw result: a value, or `-errno`.
pub(crate) fn syscall(nr: i64, a: [u64; 6]) -> i64 {
    safe_syscall(&NEVER, nr, a)
}

/// `rewind_if_in_safe_syscall()`: the start of the window when `pc` is inside it.
pub(crate) fn rewind_pc(pc: u64) -> Option<u64> {
    let start = (&raw const ruvm_lu_safe_syscall_start) as u64;
    let end = (&raw const ruvm_lu_safe_syscall_end) as u64;
    (pc > start && pc < end).then_some(start)
}

/// `host_signal_handler()` as the kernel calls it. Everything but reaching into the kernel's
/// structures is [`crate::signal::host_signal`].
extern "C" fn host_signal_handler(sig: i32, info: *mut libc::siginfo_t, uc: *mut libc::c_void) {
    let uc = uc.cast::<libc::ucontext_t>();
    // SAFETY: the kernel calls a SA_SIGINFO handler with a siginfo_t and a ucontext_t on the
    // signal stack, both valid and used by nobody else until the handler returns. The siginfo
    // is 128 bytes, 8 byte aligned. The kernel's sigset in the ucontext is the first 8 bytes of
    // libc's bigger sigset_t, which is all that is written.
    unsafe {
        let raw = info.cast::<[u64; 16]>().read();
        let pc = &mut (*uc).uc_mcontext.gregs[libc::REG_RIP as usize];
        let mask = &mut *(&raw mut (*uc).uc_sigmask).cast::<u64>();
        crate::signal::host_signal(sig, &raw, pc, mask);
    }
}

/// `SA_RESTORER`.
const SA_RESTORER: u64 = 0x0400_0000;

/// Installs the host signal handler for `sig` with every signal blocked while it runs, and
/// `SA_RESTART` when `restart`.
pub(crate) fn install_handler(sig: i32, restart: bool) -> i64 {
    let flags = libc::SA_SIGINFO as u64 | if restart { libc::SA_RESTART as u64 } else { 0 };
    let handler: extern "C" fn(i32, *mut libc::siginfo_t, *mut libc::c_void) = host_signal_handler;
    let handler = handler as usize as u64;
    let restorer = (&raw const ruvm_lu_sigreturn) as u64;
    let act = [handler, flags | SA_RESTORER, restorer, !0u64];
    sys(libc::SYS_rt_sigaction, &[sig as u64, act.as_ptr() as u64, 0, 8])
}

/// The host handler of `sig`: 0 for `SIG_DFL`, 1 for `SIG_IGN`, or an address.
pub(crate) fn disposition(sig: i32) -> u64 {
    let mut old = [0u64; 4];
    sys(libc::SYS_rt_sigaction, &[sig as u64, 0, old.as_mut_ptr() as u64, 8]);
    old[0]
}

/// `sigprocmask(SIG_SETMASK, mask)` for this thread, returning the old mask.
pub(crate) fn set_mask(mask: u64) -> u64 {
    let mut old = 0u64;
    sys(
        libc::SYS_rt_sigprocmask,
        &[libc::SIG_SETMASK as u64, (&raw const mask) as u64, (&raw mut old) as u64, 8],
    );
    old
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

/// `sysconf(_SC_NPROCESSORS_ONLN)`, the processors `/proc/cpuinfo` lists.
pub(crate) fn online_cpus() -> u64 {
    // SAFETY: sysconf has no preconditions.
    let n = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
    u64::try_from(n).unwrap_or(1).max(1)
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
