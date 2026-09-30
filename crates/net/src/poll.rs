// SPDX-License-Identifier: GPL-2.0-or-later

//! The I/O threads backends read on.
//!
//! QEMU watches backend file descriptors from its main loop. This crate gives each backend a
//! thread that polls the backend's descriptor and a wake pipe instead, since nothing runs a
//! reactor yet. The backend tells the thread what it wants to hear about through
//! [`IoHandler::interest`], and pokes it with [`Waker::wake`] whenever that changes.

use std::io;
use std::os::fd::{AsFd, AsRawFd, OwnedFd, RawFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use rustix::event::{PollFd, PollFlags, Timespec, poll};

use ruvm_base::error::strerror;
use ruvm_base::{Error, Result};

use crate::client::lock;

/// Sets `O_NONBLOCK` the way `qemu_set_blocking(fd, false)` does.
pub(crate) fn set_nonblocking(fd: impl AsFd) -> io::Result<()> {
    rustix::io::ioctl_fionbio(fd, true).map_err(io::Error::from)
}

/// QEMU's error for a descriptor that will not go non-blocking. GLib's message and the errno
/// text are the same here, so the text shows up twice just as it does in QEMU.
pub(crate) fn nonblocking_error(fd: RawFd, e: &io::Error) -> Error {
    let s = strerror(e);
    Error::generic(format!("Can't set file descriptor {fd} non-blocking: {s}: {s}"))
}

/// `qemu_set_blocking(fd, false, errp)`.
pub(crate) fn unblock(fd: &OwnedFd) -> Result<()> {
    set_nonblocking(fd).map_err(|e| nonblocking_error(fd.as_raw_fd(), &e))
}

/// The self pipe an I/O thread sleeps on besides its descriptor.
#[derive(Debug)]
pub(crate) struct Waker {
    r: OwnedFd,
    w: OwnedFd,
    stop: AtomicBool,
}

impl Waker {
    pub(crate) fn new() -> io::Result<Arc<Self>> {
        let (r, w) = rustix::pipe::pipe().map_err(io::Error::from)?;
        set_nonblocking(&r)?;
        set_nonblocking(&w)?;
        Ok(Arc::new(Waker { r, w, stop: AtomicBool::new(false) }))
    }

    /// Makes the thread look at [`IoHandler::interest`] again.
    pub(crate) fn wake(&self) {
        // A full pipe means a wakeup is pending anyway.
        let _ = rustix::io::write(&self.w, &[1]);
    }

    fn drain(&self) {
        let mut buf = [0u8; 64];
        while matches!(rustix::io::read(&self.r, &mut buf), Ok(n) if n > 0) {}
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }
}

/// What an I/O thread wants to hear about right now.
#[derive(Debug, Default)]
pub(crate) struct Interest {
    pub(crate) fd: Option<Arc<OwnedFd>>,
    pub(crate) read: bool,
    pub(crate) write: bool,
    /// When to call [`IoHandler::timeout`], counted from now.
    pub(crate) timeout: Option<Duration>,
}

/// The backend side of an I/O thread. The calls all come from the thread.
pub(crate) trait IoHandler: Send + Sync + 'static {
    fn interest(&self) -> Interest;

    /// The descriptor is readable, or hung up or in error, which a read will find out about.
    fn readable(&self, fd: &Arc<OwnedFd>);

    fn writable(&self, fd: &Arc<OwnedFd>) {
        let _ = fd;
    }

    /// The timeout from [`Interest::timeout`] ran out.
    fn timeout(&self) {}
}

/// An I/O thread and the way to stop it.
#[derive(Debug)]
pub(crate) struct IoThread {
    waker: Arc<Waker>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

impl IoThread {
    pub(crate) fn spawn(name: &str, handler: Arc<dyn IoHandler>) -> io::Result<IoThread> {
        let waker = Waker::new()?;
        let w = waker.clone();
        let handle = std::thread::Builder::new()
            .name(format!("net-{name}"))
            .spawn(move || run(&w, handler.as_ref()))?;
        Ok(IoThread { waker, handle: Mutex::new(Some(handle)) })
    }

    pub(crate) fn wake(&self) {
        self.waker.wake();
    }

    /// Stops the thread and waits for it, unless this is the thread itself.
    pub(crate) fn stop(&self) {
        self.waker.stop.store(true, Ordering::SeqCst);
        self.waker.wake();
        let handle = lock(&self.handle).take();
        if let Some(h) = handle {
            if h.thread().id() != std::thread::current().id() {
                let _ = h.join();
            }
        }
    }
}

impl Drop for IoThread {
    fn drop(&mut self) {
        self.stop();
    }
}

fn run(waker: &Waker, handler: &dyn IoHandler) {
    while !waker.stopped() {
        let interest = handler.interest();
        let deadline = interest.timeout.map(|t| Instant::now() + t);
        let mut flags = PollFlags::empty();
        if interest.read {
            flags |= PollFlags::IN;
        }
        if interest.write {
            flags |= PollFlags::OUT;
        }
        let (wake_ready, fd_events) = {
            let mut fds = vec![PollFd::new(&waker.r, PollFlags::IN)];
            if let Some(fd) = &interest.fd {
                if !flags.is_empty() {
                    fds.push(PollFd::new(fd.as_ref(), flags));
                }
            }
            let ts = interest
                .timeout
                .map(|t| Timespec { tv_sec: t.as_secs() as _, tv_nsec: t.subsec_nanos() as _ });
            match poll(&mut fds, ts.as_ref()) {
                Ok(_) => {}
                Err(rustix::io::Errno::INTR) => continue,
                Err(_) => {
                    // Nothing sensible to do but not spin.
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                }
            }
            (!fds[0].revents().is_empty(), fds.get(1).map(|f| f.revents()))
        };
        if wake_ready {
            waker.drain();
        }
        if waker.stopped() {
            break;
        }
        if let (Some(fd), Some(ev)) = (&interest.fd, fd_events) {
            let bad = PollFlags::HUP | PollFlags::ERR | PollFlags::NVAL;
            if interest.read && ev.intersects(PollFlags::IN | bad) {
                handler.readable(fd);
            }
            if interest.write && ev.intersects(PollFlags::OUT | bad) {
                handler.writable(fd);
            }
            if ev.contains(PollFlags::NVAL) {
                // A closed descriptor would spin; wait for somebody to change the interest.
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        if let Some(d) = deadline {
            if Instant::now() >= d {
                handler.timeout();
            }
        }
    }
}
