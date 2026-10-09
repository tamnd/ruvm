// SPDX-License-Identifier: GPL-2.0-or-later

//! Linux user mode emulation, `qemu-<target>`: runs one Linux program built for the target on
//! the host's kernel, translating its code with TCG and its system calls to the host's.
//!
//! The pieces follow `linux-user/`:
//!
//! - [`opts`] is `parse_args()` and `usage()` from `main.c`.
//! - [`elf`] is the ELF loader from `elfload.c`.
//! - `syscall` is `do_syscall()` with the memory management of `mmap.c`.
//! - `start` is `main()`, the same for every target.
//! - `x86_64` and `aarch64` are the targets: the CPU state a program starts with,
//!   `cpu_loop()`, the signal frames and what else differs between them, behind `guest`.
//!   `generic` numbers the system calls of aarch64 as the host does.
//!
//! The guest's address space is one reservation of host memory ([`ruvm_user_common::GuestSpace`])
//! with guest address `g` at offset `g`, always mapped readable and writable on the host.
//! Guest page protections are enforced in software: the vCPU's `tlb_fill()` looks at the page
//! flags, and every pointer a system call passes to the host kernel is checked against them and
//! against the size of the reservation first, so the kernel only ever sees memory inside it.
//!
//! What this version runs: x86_64 and aarch64 programs on an x86_64 Linux host, static or
//! dynamic, threaded or not, with the system calls of files, memory, processes, time and signals. The
//! differences from QEMU are:
//!
//! - There is no vDSO; the C library falls back to system calls.
//! - Signals are delivered as QEMU delivers them: host signals are queued and run on the
//!   guest's handlers with the kernel's `rt_sigframe`, faults the guest takes become its
//!   `SIGSEGV`, `SIGBUS`, `SIGFPE`, `SIGILL` and `SIGTRAP`, and blocking system calls restart
//!   or fail with `EINTR` as they should. Reads from a `signalfd` are not translated, and a
//!   file mapping touched past its end does not raise `SIGBUS`.
//! - Threads are `clone()` with QEMU's flags, each a vCPU on its own host thread, with
//!   `set_tid_address()`, `CLONE_CHILD_CLEARTID` and `exit` of one thread as QEMU has them.
//!   Robust futex lists are `ENOSYS` there and here. `vfork()` and `posix_spawn()` are forks.
//! - Guest memory goes through the softmmu TLB rather than straight to host addresses.
//! - The vsyscall page is not emulated, `-g`, `-strace`, `-t`, `-trace` and `-plugin` are not
//!   supported yet. `/proc/self` is the host's except for `exe`, `maps`, `smaps`, `stat`,
//!   `auxv` and `cmdline`, which describe the guest as QEMU's do.
//! - On aarch64 there are no MTE tag storage, no SVE or SME `prctl()`s and no ZA, ZT, GCS or
//!   FPMR signal frame records, and `rt_sigreturn()` keeps the program at EL0t whatever the
//!   frame says. Handlers without `SA_RESTORER` return through a page QEMU maps when it has
//!   no vDSO.
//! - The guest space is a 64 TiB reservation, as QEMU lays it out with `-R`, so addresses
//!   differ from QEMU's default layout.

#[cfg(any(target_os = "linux", target_os = "android"))]
pub mod elf;
pub mod opts;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod aarch64;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod generic;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod guest;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod host;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod procfs;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod signal;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod start;
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
    match target {
        "x86_64" => return Some(start::main(argv0, args, &mut x86_64::Target::default())),
        "aarch64" => return Some(start::main(argv0, args, &mut aarch64::Target::default())),
        _ => {}
    }
    let _ = (target, argv0, args);
    None
}
