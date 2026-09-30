// SPDX-License-Identifier: GPL-2.0-or-later

//! The `pty` chardev, chardev/char-pty.c.
//!
//! ruvm keeps the master side and says which slave to open. Nothing is read or written until
//! something opens the slave: while nobody has it open the chardev looks for a new user now and
//! then, and what the guest writes meanwhile is dropped.

use std::fs::File;
use std::io::{self, Read, Write};
use std::os::fd::{AsFd, OwnedFd};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};

use ruvm_base::{Error, Result};
use ruvm_qapi::types::ChardevPty;

use crate::local::{Readiness, poll_in, timed_out};

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// `PtyChardev`.
#[derive(Debug)]
pub(crate) struct Pty {
    master: File,
    name: String,
    /// The symlink made for the `path` option, removed with the chardev.
    link: Mutex<Option<PathBuf>>,
    connected: AtomicBool,
}

/// `qemu_openpty_raw()`: a new pseudo terminal in raw mode, its master and the slave's name.
pub fn openpty_raw() -> io::Result<(OwnedFd, String)> {
    use rustix::fs::{Mode, OFlags, open};
    use rustix::pty::{OpenptFlags, grantpt, openpt, ptsname, unlockpt};
    use rustix::termios::{OptionalActions, tcgetattr, tcsetattr};

    let master = openpt(OpenptFlags::RDWR | OpenptFlags::NOCTTY)?;
    grantpt(&master)?;
    unlockpt(&master)?;
    let name = ptsname(&master, Vec::new())?;
    let slave = open(name.as_c_str(), OFlags::RDWR | OFlags::NOCTTY, Mode::empty())?;
    let mut tty = tcgetattr(&slave)?;
    tty.make_raw();
    tcsetattr(&slave, OptionalActions::Flush, &tty)?;
    rustix::io::fcntl_setfd(&master, rustix::io::FdFlags::CLOEXEC)?;
    Ok((master, name.to_string_lossy().into_owned()))
}

/// Whether the slave of `master` is open: the master hangs up while it is not.
fn slave_open(master: impl AsFd) -> bool {
    use rustix::event::{PollFd, PollFlags, Timespec, poll};

    let mut fds = [PollFd::new(&master, PollFlags::OUT)];
    let zero = Timespec { tv_sec: 0, tv_nsec: 0 };
    match poll(&mut fds, Some(&zero)) {
        Ok(_) => !fds[0].revents().contains(PollFlags::HUP),
        Err(_) => false,
    }
}

impl Pty {
    /// `pty_chr_open()`. The message saying where the slave is goes to standard output when
    /// `announce` is set, as `qemu_printf()` outside a monitor command does.
    pub(crate) fn open(label: &str, opts: &ChardevPty, announce: bool) -> Result<Pty> {
        let (master, name) =
            openpty_raw().map_err(|e| Error::from_io("Failed to create PTY", e))?;
        rustix::fs::fcntl_setfl(&master, rustix::fs::OFlags::NONBLOCK)
            .map_err(|e| Error::from_io("Failed to set FD nonblocking", e.into()))?;
        if announce {
            println!("{}", redirected_message(&name, label));
        }
        let mut link = None;
        if let Some(path) = &opts.path {
            std::os::unix::fs::symlink(&name, path)
                .map_err(|e| Error::from_io("Failed to create PTY symlink", e))?;
            link = Some(PathBuf::from(path));
        }
        Ok(Pty {
            master: File::from(master),
            name,
            link: Mutex::new(link),
            connected: AtomicBool::new(false),
        })
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// Removes the symlink, `char_pty_finalize()`.
    pub(crate) fn close(&self) {
        if let Some(p) = lock(&self.link).take() {
            let _ = std::fs::remove_file(p);
        }
    }

    /// `pty_chr_update_read_handler()`: looks whether a slave is there now.
    pub(crate) fn poll_connected(&self) -> bool {
        let c = slave_open(&self.master);
        self.connected.store(c, Ordering::Release);
        c
    }

    pub(crate) fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Acquire)
    }

    /// `pty_chr_read()`. 0 once the slave is closed.
    pub(crate) fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        if !self.is_connected() {
            return Ok(0);
        }
        let r = match poll_in(&self.master, crate::conn::POLL_INTERVAL)? {
            Readiness::Nothing => return Err(timed_out()),
            Readiness::Hangup => Ok(0),
            Readiness::Readable => (&self.master).read(buf),
        };
        match r {
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => Err(e),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => Err(e),
            Ok(n) if n > 0 => Ok(n),
            // Anything else means the slave went away, `pty_chr_state(chr, 0)`.
            _ => {
                self.connected.store(false, Ordering::Release);
                Ok(0)
            }
        }
    }

    /// `pty_chr_write()`: bytes nobody can read are dropped.
    pub(crate) fn write(&self, buf: &[u8]) -> io::Result<usize> {
        if self.is_connected() || slave_open(&self.master) {
            return (&self.master).write(buf);
        }
        Ok(buf.len())
    }
}

impl Drop for Pty {
    fn drop(&mut self) {
        self.close();
    }
}

/// What QEMU prints when it made a pty.
pub fn redirected_message(name: &str, label: &str) -> String {
    format!("char device redirected to {name} (label {label})")
}
