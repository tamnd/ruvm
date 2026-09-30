// SPDX-License-Identifier: MIT OR Apache-2.0

//! The termination signals, `os_setup_signal_handling()` from os-posix.c.
//!
//! The handler only records which signal came from which process and writes a byte to a socket
//! pair. A thread reads the other end and calls back into safe code, so nothing that is unsafe
//! in a signal handler ever runs in one.

use std::io::{self, Read};
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicI32, Ordering};

/// A termination signal that arrived, what `qemu_system_killed()` records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Killed {
    pub signo: i32,
    /// The sending process, or 0 when the signal came from the terminal or the kernel.
    pub pid: i32,
}

/// The signals QEMU treats as a request to quit.
pub const TERMINATION_SIGNALS: [i32; 3] = [libc::SIGINT, libc::SIGHUP, libc::SIGTERM];

static WAKE_FD: AtomicI32 = AtomicI32::new(-1);
static SIGNO: AtomicI32 = AtomicI32::new(0);
static PID: AtomicI32 = AtomicI32::new(0);

extern "C" fn termsig_handler(
    signo: libc::c_int,
    info: *mut libc::siginfo_t,
    _: *mut libc::c_void,
) {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    // SAFETY: the kernel passes a valid siginfo_t to an SA_SIGINFO handler, and si_pid is set
    // for the signals this handler is installed for.
    let pid = unsafe { (*info).si_pid() };
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    // SAFETY: the kernel passes a valid siginfo_t to an SA_SIGINFO handler.
    let pid = unsafe { (*info).si_pid };
    SIGNO.store(signo, Ordering::Release);
    PID.store(pid, Ordering::Release);
    let fd = WAKE_FD.load(Ordering::Acquire);
    if fd >= 0 {
        let byte = 1u8;
        // SAFETY: write() is async-signal-safe, the buffer is one valid byte, and the fd is the
        // nonblocking end of a socket pair that lives for the rest of the process. A full
        // buffer means the reader has a wakeup pending already, so a failed write is fine.
        unsafe { libc::write(fd, (&raw const byte).cast(), 1) };
    }
}

/// Installs the handler for SIGINT, SIGHUP and SIGTERM and calls `on_signal` from a thread of
/// its own for each one that arrives. Only the first call in a process does anything.
pub fn on_termination(on_signal: impl Fn(Killed) + Send + 'static) -> io::Result<()> {
    let (mut rx, tx) = UnixStream::pair()?;
    tx.set_nonblocking(true)?;
    if WAKE_FD.compare_exchange(-1, tx.as_raw_fd(), Ordering::AcqRel, Ordering::Acquire).is_err() {
        return Ok(());
    }
    // The write end is used by the handler for as long as the process runs.
    std::mem::forget(tx);
    std::thread::Builder::new().name("signals".into()).spawn(move || {
        let mut buf = [0u8; 16];
        while let Ok(n) = rx.read(&mut buf) {
            if n == 0 {
                return;
            }
            let signo = SIGNO.load(Ordering::Acquire);
            let pid = PID.load(Ordering::Acquire);
            on_signal(Killed { signo, pid });
        }
    })?;
    // SAFETY: an all zero sigaction is a valid value to fill in, the handler has the
    // SA_SIGINFO signature, and it only touches atomics and calls write().
    let rc = unsafe {
        let mut act: libc::sigaction = std::mem::zeroed();
        act.sa_sigaction = termsig_handler as *const () as libc::sighandler_t;
        act.sa_flags = libc::SA_SIGINFO;
        libc::sigemptyset(&mut act.sa_mask);
        TERMINATION_SIGNALS
            .iter()
            .map(|&sig| libc::sigaction(sig, &act, std::ptr::null_mut()))
            .fold(0, |a, r| a | r)
    };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// `set_exit_with_parent()` for `-run-with exit-with-parent=on`: calls `on_exit` with a SIGTERM
/// from the parent once the parent process is gone. The macOS version of QEMU does the same
/// from a thread, and this polls the parent pid so it works the same way everywhere.
pub fn on_parent_exit(on_exit: impl FnOnce(Killed) + Send + 'static) -> io::Result<()> {
    let ppid = std::os::unix::process::parent_id();
    std::thread::Builder::new().name("exit-parent".into()).spawn(move || {
        while std::os::unix::process::parent_id() == ppid {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        on_exit(Killed { signo: libc::SIGTERM, pid: ppid as i32 });
    })?;
    Ok(())
}
