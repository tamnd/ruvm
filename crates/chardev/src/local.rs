// SPDX-License-Identifier: GPL-2.0-or-later

//! The pieces the backends that are not sockets share: where their input comes from and where
//! their output goes, as char-fd.c does for file descriptors.

use std::collections::VecDeque;
use std::fmt;
use std::io::{self, Read, Write};
use std::sync::{Condvar, Mutex, MutexGuard};

use crate::conn::POLL_INTERVAL;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// What a read gives when nothing came in time. [`crate::Connection::recv`] then looks at its
/// stop flag and tries again.
pub(crate) fn timed_out() -> io::Error {
    io::Error::from(io::ErrorKind::WouldBlock)
}

/// Bytes waiting for a reader, with a way to wait for more. `qemu_chr_be_write()` from outside
/// the backend puts its bytes here.
#[derive(Debug, Default)]
pub(crate) struct ByteQueue {
    bytes: Mutex<VecDeque<u8>>,
    more: Condvar,
}

impl ByteQueue {
    pub(crate) fn push(&self, buf: &[u8]) {
        lock(&self.bytes).extend(buf);
        self.more.notify_all();
    }

    /// Takes what is there, up to `buf.len()` bytes. `None` when there is nothing.
    pub(crate) fn pop(&self, buf: &mut [u8]) -> Option<usize> {
        let mut q = lock(&self.bytes);
        if q.is_empty() || buf.is_empty() {
            return None;
        }
        let n = q.len().min(buf.len());
        for (d, s) in buf.iter_mut().zip(q.drain(..n)) {
            *d = s;
        }
        Some(n)
    }

    /// Takes what is there, waiting up to [`POLL_INTERVAL`] for something to come.
    pub(crate) fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        {
            let q = lock(&self.bytes);
            if q.is_empty() {
                let _ = self.more.wait_timeout(q, POLL_INTERVAL).unwrap_or_else(|e| e.into_inner());
            }
        }
        self.pop(buf).ok_or_else(timed_out)
    }
}

/// Where a backend's input comes from. `read` waits at most about [`POLL_INTERVAL`] and then
/// fails with [`io::ErrorKind::WouldBlock`]. It gives 0 at the end of the input.
pub(crate) trait Source: Send + fmt::Debug {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize>;
}

/// A readable descriptor, waited on with `poll()`.
#[cfg(unix)]
#[derive(Debug)]
pub(crate) struct FdSource(pub(crate) std::fs::File);

#[cfg(unix)]
impl Source for FdSource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match poll_in(&self.0, POLL_INTERVAL)? {
            Readiness::Nothing => Err(timed_out()),
            Readiness::Hangup => Ok(0),
            Readiness::Readable => match self.0.read(buf) {
                // `EIO` is what a terminal gives once its other side is closed.
                Err(e) if e.raw_os_error() == Some(rustix::io::Errno::IO.raw_os_error()) => Ok(0),
                r => r,
            },
        }
    }
}

/// What `poll()` said of a descriptor.
#[cfg(unix)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Readiness {
    Nothing,
    Readable,
    /// Hung up with nothing left to read.
    Hangup,
}

/// Waits up to `timeout` for `fd` to have something to read.
#[cfg(unix)]
pub(crate) fn poll_in(
    fd: impl std::os::fd::AsFd,
    timeout: std::time::Duration,
) -> io::Result<Readiness> {
    use rustix::event::{PollFd, PollFlags, Timespec, poll};

    let ts = Timespec { tv_sec: timeout.as_secs() as _, tv_nsec: timeout.subsec_nanos() as _ };
    let mut fds = [PollFd::new(&fd, PollFlags::IN)];
    match poll(&mut fds, Some(&ts)) {
        Ok(0) => return Ok(Readiness::Nothing),
        Ok(_) => {}
        Err(rustix::io::Errno::INTR) => return Ok(Readiness::Nothing),
        Err(e) => return Err(e.into()),
    }
    let revents = fds[0].revents();
    Ok(if revents.contains(PollFlags::IN) {
        Readiness::Readable
    } else if revents.intersects(PollFlags::HUP | PollFlags::ERR | PollFlags::NVAL) {
        Readiness::Hangup
    } else {
        Readiness::Nothing
    })
}

/// Input read by a thread of its own, for handles that cannot be polled.
#[cfg_attr(unix, allow(dead_code))]
#[derive(Debug)]
pub(crate) struct ThreadSource {
    rx: std::sync::mpsc::Receiver<Vec<u8>>,
    pending: VecDeque<u8>,
    eof: bool,
}

#[cfg_attr(unix, allow(dead_code))]
impl ThreadSource {
    pub(crate) fn spawn(name: String, mut input: impl Read + Send + 'static) -> io::Result<Self> {
        let (tx, rx) = std::sync::mpsc::sync_channel(4);
        std::thread::Builder::new().name(name).spawn(move || {
            let mut buf = [0u8; 4096];
            loop {
                match input.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        if tx.send(buf[..n].to_vec()).is_err() {
                            break;
                        }
                    }
                }
            }
        })?;
        Ok(ThreadSource { rx, pending: VecDeque::new(), eof: false })
    }
}

impl Source for ThreadSource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pending.is_empty() && !self.eof {
            match self.rx.recv_timeout(POLL_INTERVAL) {
                Ok(v) => self.pending.extend(v),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => return Err(timed_out()),
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => self.eof = true,
            }
        }
        let n = self.pending.len().min(buf.len());
        for (d, s) in buf.iter_mut().zip(self.pending.drain(..n)) {
            *d = s;
        }
        Ok(n)
    }
}

/// Which of the descriptor backends a [`Local`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LocalKind {
    File,
    #[cfg_attr(not(unix), allow(dead_code))]
    Pipe,
    Stdio,
}

/// A chardev on a pair of host handles: `file`, `pipe` and `stdio`, `FDChardev`.
pub(crate) struct Local {
    pub(crate) kind: LocalKind,
    /// `None` when the chardev takes no input, as `file` without `in`.
    pub(crate) input: Mutex<Option<Box<dyn Source>>>,
    pub(crate) output: Mutex<Box<dyn Write + Send>>,
}

impl fmt::Debug for Local {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Local").field("kind", &self.kind).finish_non_exhaustive()
    }
}

impl Local {
    pub(crate) fn new(
        kind: LocalKind,
        input: Option<Box<dyn Source>>,
        output: Box<dyn Write + Send>,
    ) -> Local {
        Local { kind, input: Mutex::new(input), output: Mutex::new(output) }
    }

    /// Reads the input, or waits on `queue` when there is none.
    pub(crate) fn read(&self, queue: &ByteQueue, buf: &mut [u8]) -> io::Result<usize> {
        match lock(&self.input).as_mut() {
            Some(src) => src.read(buf),
            None => queue.read(buf),
        }
    }

    pub(crate) fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let mut out = lock(&self.output);
        let n = out.write(buf)?;
        out.flush()?;
        Ok(n)
    }
}
