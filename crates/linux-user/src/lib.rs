// SPDX-License-Identifier: GPL-2.0-or-later

//! Linux user mode emulation, `qemu-<target>`: runs one Linux program built for the target on
//! the host's kernel, translating its code with TCG and its system calls to the host's.
//!
//! The pieces follow `linux-user/`:
//!
//! - [`opts`] is `parse_args()` and `usage()` from `main.c`.
//! - [`elf`] is the ELF loader from `elfload.c`.
//! - `syscall` is `do_syscall()` with the memory management of `mmap.c`.
//! - `x86_64` is the target: the CPU state a program starts with, `cpu_loop()` and
//!   `arch_prctl()`.
//!
//! The guest's address space is one reservation of host memory ([`ruvm_user_common::GuestSpace`])
//! with guest address `g` at offset `g`, always mapped readable and writable on the host.
//! Guest page protections are enforced in software: the vCPU's `tlb_fill()` looks at the page
//! flags, and every pointer a system call passes to the host kernel is checked against them and
//! against the size of the reservation first, so the kernel only ever sees memory inside it.
//!
//! What this first version runs: x86_64 programs on an x86_64 Linux host, static or dynamic,
//! with the system calls of single threaded programs (files, memory, processes, time). The
//! differences from QEMU are:
//!
//! - There is no vDSO; the C library falls back to system calls.
//! - Signals are delivered as QEMU delivers them: host signals are queued and run on the
//!   guest's handlers with the kernel's `rt_sigframe`, faults the guest takes become its
//!   `SIGSEGV`, `SIGBUS`, `SIGFPE`, `SIGILL` and `SIGTRAP`, and blocking system calls restart
//!   or fail with `EINTR` as they should. Reads from a `signalfd` are not translated, and a
//!   file mapping touched past its end does not raise `SIGBUS`.
//! - `clone()` with `CLONE_VM` (threads) fails with `EINVAL`; `fork()`, `vfork()` and
//!   `posix_spawn()` work, as forks.
//! - Guest memory goes through the softmmu TLB rather than straight to host addresses.
//! - The vsyscall page is not emulated, `-g`, `-strace`, `-t`, `-trace` and `-plugin` are not
//!   supported yet, and `/proc/self` is the host's except for `/proc/self/exe`.
//! - The guest space is a 64 TiB reservation, as QEMU lays it out with `-R`, so addresses
//!   differ from QEMU's default layout.

#[cfg(any(target_os = "linux", target_os = "android"))]
pub mod elf;
pub mod opts;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod host;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod signal;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod syscall;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod x86_64;

use std::process::ExitCode;

/// `strerror()`.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) fn strerror(e: i32) -> String {
    let s = std::io::Error::from_raw_os_error(e).to_string();
    match s.rfind(" (os error ") {
        Some(i) => s[..i].to_string(),
        None => s,
    }
}

/// `strerror()` of an I/O error.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) fn strerror_of(e: &std::io::Error) -> String {
    match e.raw_os_error() {
        Some(n) => strerror(n),
        None => e.to_string(),
    }
}

/// Runs `qemu-<target>` with `args`, the arguments after the program name, when this build can
/// emulate `target` on this host. `None` means it cannot.
pub fn run(target: &str, argv0: &str, args: &[String]) -> Option<ExitCode> {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    if target == "x86_64" {
        return Some(x86_64::main(argv0, args));
    }
    let _ = (target, argv0, args);
    None
}
