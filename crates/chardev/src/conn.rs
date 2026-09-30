// SPDX-License-Identifier: GPL-2.0-or-later

//! One client connection of a chardev, what `QIOChannelSocket` is to char-socket.c.

use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpStream};
#[cfg(unix)]
use std::os::fd::OwnedFd;
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// How often a blocked read looks at the stop flag. Detaching a frontend waits at most this long
/// for the reader to notice.
pub(crate) const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// `TCP_MAX_FDS`: how many descriptors one read takes. More than that are dropped by the
/// kernel with MSG_CTRUNC set, as in QEMU.
#[cfg(unix)]
pub const TCP_MAX_FDS: usize = 16;

#[derive(Debug)]
pub(crate) enum Stream {
    #[cfg(unix)]
    Unix(UnixStream),
    Tcp(TcpStream),
}

impl Stream {
    pub(crate) fn try_clone(&self) -> io::Result<Stream> {
        Ok(match self {
            #[cfg(unix)]
            Stream::Unix(s) => Stream::Unix(s.try_clone()?),
            Stream::Tcp(s) => Stream::Tcp(s.try_clone()?),
        })
    }

    pub(crate) fn shutdown(&self) {
        // The peer may already be gone, and then there is nothing left to shut.
        let _ = match self {
            #[cfg(unix)]
            Stream::Unix(s) => s.shutdown(Shutdown::Both),
            Stream::Tcp(s) => s.shutdown(Shutdown::Both),
        };
    }

    fn set_read_timeout(&self, t: Option<Duration>) -> io::Result<()> {
        match self {
            #[cfg(unix)]
            Stream::Unix(s) => s.set_read_timeout(t),
            Stream::Tcp(s) => s.set_read_timeout(t),
        }
    }

    fn writer(&self) -> io::Result<Box<dyn Write + Send>> {
        Ok(match self.try_clone()? {
            #[cfg(unix)]
            Stream::Unix(s) => Box::new(s),
            Stream::Tcp(s) => Box::new(s),
        })
    }
}

/// A connected client as the frontend sees it. Reads return 0 when the peer closes the
/// connection and also when the frontend is being detached, and the chardev tells the two
/// apart itself.
#[derive(Debug)]
pub struct Connection {
    stream: Stream,
    stop: Arc<AtomicBool>,
    fd_pass: bool,
    #[cfg(unix)]
    fds: Vec<OwnedFd>,
}

impl Connection {
    pub(crate) fn new(stream: Stream, stop: Arc<AtomicBool>) -> io::Result<Connection> {
        stream.set_read_timeout(Some(POLL_INTERVAL))?;
        #[cfg(unix)]
        let fd_pass = matches!(stream, Stream::Unix(_));
        #[cfg(not(unix))]
        let fd_pass = false;
        Ok(Connection {
            stream,
            stop,
            fd_pass,
            #[cfg(unix)]
            fds: Vec::new(),
        })
    }

    /// Whether the peer can pass descriptors, `QEMU_CHAR_FEATURE_FD_PASS`.
    pub fn can_pass_fds(&self) -> bool {
        self.fd_pass
    }

    /// A handle to write to the client with. It stays usable after the connection ends, and
    /// writes then fail.
    pub fn writer(&self) -> io::Result<Box<dyn Write + Send>> {
        self.stream.writer()
    }

    /// Reads what the client sent, waiting for it. Descriptors that came with the bytes are
    /// kept for [`Connection::take_fds`].
    pub fn recv(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        loop {
            if self.stop.load(Ordering::Acquire) {
                return Ok(0);
            }
            match self.recv_once(buf) {
                Err(e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::WouldBlock
                            | io::ErrorKind::TimedOut
                            | io::ErrorKind::Interrupted
                    ) => {}
                r => return r,
            }
        }
    }

    fn recv_once(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match &mut self.stream {
            #[cfg(unix)]
            Stream::Unix(s) => {
                let (n, fds) = recv_with_fds(s, buf)?;
                if !fds.is_empty() {
                    self.fds = fds;
                }
                Ok(n)
            }
            Stream::Tcp(s) => s.read(buf),
        }
    }

    /// The descriptors that came with the last read that had any, `read_msgfds` in
    /// char-socket.c. Taking them leaves none.
    #[cfg(unix)]
    pub fn take_fds(&mut self) -> Vec<OwnedFd> {
        std::mem::take(&mut self.fds)
    }
}

/// One `recvmsg()` that keeps the descriptors that came with the bytes, close on exec.
#[cfg(unix)]
pub fn recv_with_fds(stream: &UnixStream, buf: &mut [u8]) -> io::Result<(usize, Vec<OwnedFd>)> {
    use std::mem::MaybeUninit;

    use rustix::net::{RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, recvmsg};

    let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(TCP_MAX_FDS))];
    let mut control = RecvAncillaryBuffer::new(&mut space);
    #[cfg(not(target_vendor = "apple"))]
    let flags = RecvFlags::CMSG_CLOEXEC;
    #[cfg(target_vendor = "apple")]
    let flags = RecvFlags::empty();
    let msg = recvmsg(stream, &mut [io::IoSliceMut::new(buf)], &mut control, flags)?;
    let mut fds = Vec::new();
    for m in control.drain() {
        if let RecvAncillaryMessage::ScmRights(received) = m {
            fds.extend(received);
        }
    }
    // Without MSG_CMSG_CLOEXEC the flag is set right after, as qio_channel_socket_copy_fds()
    // does on such hosts.
    #[cfg(target_vendor = "apple")]
    for fd in &fds {
        rustix::io::fcntl_setfd(fd, rustix::io::FdFlags::CLOEXEC)?;
    }
    Ok((msg.bytes, fds))
}
