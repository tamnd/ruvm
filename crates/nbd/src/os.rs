// SPDX-License-Identifier: GPL-2.0-or-later

//! The process plumbing qemu-nbd and qemu-storage-daemon share: `qemu_write_pidfile()` from
//! util/oslib-posix.c, `qemu_daemon()`, `os_daemonize()` and `os_setup_post()` from
//! os-posix.c, and `check_socket_activation()` from util/systemd.c.
//!
//! Differences from QEMU:
//!
//! - The PID file lock is taken with rustix's `fcntl_lock()`, which is an open file
//!   description lock on Linux and `flock()` elsewhere, instead of a process wide `F_SETLK`
//!   record lock. Two daemons with the same PID file still exclude each other.
//! - [`Daemon`] does not ignore `SIGTSTP`, `SIGTTOU` and `SIGTTIN`. After `setsid()` the
//!   daemon has no controlling terminal, so they are not sent to it.

#![allow(unsafe_code)]

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::fd::{AsFd, FromRawFd, OwnedFd};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

use rustix::fs::FlockOperation;
use rustix::io::FdFlags;
use ruvm_base::Error;

/// `FIRST_SOCKET_ACTIVATION_FD`.
pub const FIRST_SOCKET_ACTIVATION_FD: i32 = 3;

/// A PID file written by [`write_pidfile`]. It stays locked for as long as this lives.
#[derive(Debug)]
pub struct PidFile {
    _file: File,
}

/// `qemu_write_pidfile()`: create and lock `path` and write the process ID to it.
pub fn write_pidfile(path: &str) -> Result<PidFile, Error> {
    let file = loop {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(path)
            .map_err(|e| Error::from_io(format!("Could not create '{path}'"), e))?;
        let b = file.metadata().map_err(|e| Error::from_io("Cannot stat file", e))?;
        rustix::fs::fcntl_lock(&file, FlockOperation::NonBlockingLockExclusive)
            .map_err(|e| Error::from_io("Cannot lock pid file", io::Error::from(e)))?;
        // Make sure the file that is locked is still the one at `path`.
        match std::fs::metadata(path) {
            // Someone else removed it; try again.
            Err(_) => continue,
            Ok(a) if a.ino() == b.ino() && a.dev() == b.dev() => break file,
            Ok(_) => continue,
        }
    };
    let fail = |e: Error| {
        let _ = std::fs::remove_file(path);
        e
    };
    file.set_len(0).map_err(|e| fail(Error::from_io("Failed to truncate pid file", e)))?;
    let pid = format!("{}\n", std::process::id());
    (&file)
        .write_all(pid.as_bytes())
        .map_err(|_| fail(Error::generic("Failed to write pid file")))?;
    Ok(PidFile { _file: file })
}

/// `fork()`. Returns the child's ID in the parent and `None` in the child.
///
/// Only call this while the process has a single thread: the child gets a copy of the
/// calling thread only, and locks other threads held stay locked in it forever.
pub fn fork() -> io::Result<Option<u32>> {
    let _ = io::stdout().flush();
    // SAFETY: fork() has no memory safety preconditions of its own. The child continues with
    // a copy of this thread only, which is sound as long as no other thread held a lock or
    // was in the middle of an allocation, and the callers fork before they start threads.
    let pid = unsafe { libc::fork() };
    match pid {
        -1 => Err(io::Error::last_os_error()),
        0 => Ok(None),
        p => Ok(Some(p as u32)),
    }
}

/// `g_unix_open_pipe(fds, FD_CLOEXEC)`: a pipe whose ends are closed on exec. Returns the
/// read end and the write end.
pub fn cloexec_pipe() -> io::Result<(OwnedFd, OwnedFd)> {
    let (r, w) = rustix::pipe::pipe()?;
    rustix::io::fcntl_setfd(&r, FdFlags::CLOEXEC)?;
    rustix::io::fcntl_setfd(&w, FdFlags::CLOEXEC)?;
    Ok((r, w))
}

fn devnull_stdio(stderr_too: bool) -> io::Result<()> {
    let null = OpenOptions::new().read(true).write(true).open("/dev/null")?;
    rustix::stdio::dup2_stdin(&null)?;
    rustix::stdio::dup2_stdout(&null)?;
    if stderr_too {
        rustix::stdio::dup2_stderr(&null)?;
    }
    Ok(())
}

/// `qemu_daemon(1, 0)`, which is `daemon(1, 0)`: fork, let the parent exit, start a new
/// session and point the standard descriptors at `/dev/null`.
pub fn qemu_daemon() -> io::Result<()> {
    if fork()?.is_some() {
        std::process::exit(0);
    }
    rustix::process::setsid()?;
    devnull_stdio(true)
}

/// `dup(STDERR_FILENO)`.
pub fn dup_stderr() -> io::Result<OwnedFd> {
    Ok(rustix::io::dup(io::stderr().as_fd())?)
}

/// `dup2(fd, STDERR_FILENO)`.
pub fn dup2_stderr(fd: impl AsFd) -> io::Result<()> {
    Ok(rustix::stdio::dup2_stderr(fd)?)
}

/// `dup2(STDOUT_FILENO, STDERR_FILENO)`.
pub fn stdout_to_stderr() -> io::Result<()> {
    Ok(rustix::stdio::dup2_stderr(io::stdout().as_fd())?)
}

/// The daemon side of `os_daemonize()`, kept until `os_setup_post()` tells the waiting
/// parent that startup is over.
#[derive(Debug)]
pub struct Daemon {
    pipe: File,
}

/// `os_daemonize()` with `--daemonize`: fork, and in the parent wait for the child to report
/// that its startup went well, then exit with 0 if it did and 1 if not. Only the daemon
/// returns from here.
pub fn daemonize() -> Daemon {
    let Ok((r, w)) = cloexec_pipe() else {
        std::process::exit(1);
    };
    match fork() {
        Err(_) => std::process::exit(1),
        Ok(Some(_)) => {
            drop(w);
            let mut r = File::from(r);
            let mut status = [0u8; 1];
            let len = loop {
                match r.read(&mut status) {
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(_) => break 0,
                    Ok(n) => break n,
                }
            };
            // Only exit successfully if the child wrote a zero byte after a good startup.
            std::process::exit(if len == 1 && status[0] == 0 { 0 } else { 1 });
        }
        Ok(None) => {}
    }
    drop(r);
    let _ = rustix::process::setsid();
    match fork() {
        Ok(Some(_)) => std::process::exit(0),
        Err(_) => std::process::exit(1),
        Ok(None) => {}
    }
    rustix::process::umask(rustix::fs::Mode::from_raw_mode(0o027));
    Daemon { pipe: File::from(w) }
}

impl Daemon {
    /// `os_setup_post()`: change to `/`, point the standard descriptors at `/dev/null` and
    /// tell the parent that startup is over.
    pub fn setup_post(self) {
        if let Err(e) = std::env::set_current_dir("/") {
            ruvm_base::report::error_report(&format!(
                "not able to chdir to /: {}",
                ruvm_base::error::strerror(&e)
            ));
            std::process::exit(1);
        }
        if let Err(e) = devnull_stdio(true) {
            eprintln!("Failed to open /dev/null: {}", ruvm_base::error::strerror(&e));
            std::process::exit(1);
        }
        let mut pipe = self.pipe;
        if pipe.write_all(&[0]).is_err() {
            std::process::exit(1);
        }
    }
}

/// `check_socket_activation()`: the listening sockets systemd passed in, starting at
/// [`FIRST_SOCKET_ACTIVATION_FD`]. Empty when the process was not socket activated.
pub fn check_socket_activation() -> Result<Vec<OwnedFd>, String> {
    let Some(pid) = std::env::var("LISTEN_PID").ok() else {
        return Ok(Vec::new());
    };
    let Ok((pid, _)) = ruvm_qapi::cutils::strtou64(&pid, 10, true) else {
        return Ok(Vec::new());
    };
    if pid != u64::from(std::process::id()) {
        return Ok(Vec::new());
    }
    let Some(nr) = std::env::var("LISTEN_FDS").ok() else {
        return Ok(Vec::new());
    };
    let Ok((nr, _)) = ruvm_qapi::cutils::strtou64(&nr, 10, true) else {
        return Ok(Vec::new());
    };
    // SAFETY: this runs while the process has a single thread, before the tools start any,
    // so nothing can read the environment at the same time.
    unsafe {
        // So these are not passed to any child processes we might start.
        std::env::remove_var("LISTEN_FDS");
        std::env::remove_var("LISTEN_PID");
        std::env::remove_var("LISTEN_FDNAMES");
    }
    let mut fds = Vec::new();
    for i in 0..nr {
        let fd = FIRST_SOCKET_ACTIVATION_FD + i as i32;
        // SAFETY: fcntl() on a descriptor number that may not be open only fails with EBADF.
        // The descriptor is only adopted once it is known to be open, and systemd hands it
        // to this process to own; nothing else in the program refers to it.
        let owned = unsafe {
            let f = libc::fcntl(fd, libc::F_GETFD);
            if f == -1 || libc::fcntl(fd, libc::F_SETFD, f | libc::FD_CLOEXEC) == -1 {
                None
            } else {
                Some(OwnedFd::from_raw_fd(fd))
            }
        };
        match owned {
            Some(o) => fds.push(o),
            None => {
                let e = io::Error::last_os_error();
                return Err(format!(
                    "Socket activation failed: invalid file descriptor fd = {fd}: {}",
                    ruvm_base::error::strerror(&e)
                ));
            }
        }
    }
    Ok(fds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pidfile_holds_the_pid_and_locks() {
        let dir = std::env::temp_dir().join(format!("ruvm-nbd-pid-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pid");
        let p = path.to_str().unwrap();
        let f = write_pidfile(p).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text, format!("{}\n", std::process::id()));
        drop(f);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pidfile_in_missing_dir() {
        let e = write_pidfile("/nonexistent-dir/x.pid").unwrap_err();
        assert_eq!(
            e.message(),
            "Could not create '/nonexistent-dir/x.pid': No such file or directory"
        );
    }
}
