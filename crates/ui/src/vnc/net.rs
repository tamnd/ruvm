// SPDX-License-Identifier: GPL-2.0-or-later

//! The sockets of a VNC display, standing in for QEMU's `QIONetListener`, `inet_listen_saddr()`,
//! `unix_listen_saddr()` and the client channel.
//!
//! Each listening socket has a thread accepting clients. Each client has a thread reading its
//! socket into the protocol handlers and a thread writing what they queued, so a client that
//! stops reading only stalls its own writer.

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
#[cfg(unix)]
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::Duration;

use ruvm_base::{Error, Result};
use ruvm_qapi::types::NetworkAddressFamily;

use super::{VncDisplay, lock};

/// A socket address the way `vnc_init_basic_info()` reports it.
#[derive(Clone, Debug)]
pub(crate) struct AddrInfo {
    pub(crate) host: String,
    pub(crate) service: String,
    pub(crate) family: NetworkAddressFamily,
}

impl AddrInfo {
    fn inet(addr: SocketAddr) -> AddrInfo {
        let family = match addr {
            SocketAddr::V4(_) => NetworkAddressFamily::Ipv4,
            SocketAddr::V6(_) => NetworkAddressFamily::Ipv6,
        };
        AddrInfo { host: addr.ip().to_string(), service: addr.port().to_string(), family }
    }

    #[cfg(unix)]
    fn unix(path: &str) -> AddrInfo {
        AddrInfo {
            host: String::new(),
            service: path.to_string(),
            family: NetworkAddressFamily::Unix,
        }
    }
}

/// What `-vnc` asked to listen on, one `SocketAddress` of `vnc_display_get_addresses()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ListenAddr {
    Inet {
        host: String,
        port: u32,
        to: Option<u32>,
        ipv4: Option<bool>,
        ipv6: Option<bool>,
    },
    #[cfg_attr(not(unix), allow(dead_code))]
    Unix(String),
}

/// A listening socket.
pub(crate) enum Listener {
    Tcp(TcpListener),
    #[cfg(unix)]
    Unix(UnixListener, String),
}

impl Listener {
    pub(crate) fn info(&self) -> AddrInfo {
        match self {
            Listener::Tcp(l) => match l.local_addr() {
                Ok(a) => AddrInfo::inet(a),
                Err(_) => AddrInfo {
                    host: String::new(),
                    service: String::new(),
                    family: NetworkAddressFamily::Unknown,
                },
            },
            #[cfg(unix)]
            Listener::Unix(_, path) => AddrInfo::unix(path),
        }
    }
}

/// A connected client socket.
pub(crate) enum Stream {
    Tcp(TcpStream),
    #[cfg(unix)]
    Unix(UnixStream),
}

impl Stream {
    fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Stream::Tcp(s) => (&*s).read(buf),
            #[cfg(unix)]
            Stream::Unix(s) => (&*s).read(buf),
        }
    }

    fn write(&self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Stream::Tcp(s) => (&*s).write(buf),
            #[cfg(unix)]
            Stream::Unix(s) => (&*s).write(buf),
        }
    }

    fn shutdown(&self) {
        let _ = match self {
            Stream::Tcp(s) => s.shutdown(std::net::Shutdown::Both),
            #[cfg(unix)]
            Stream::Unix(s) => s.shutdown(std::net::Shutdown::Both),
        };
    }

    fn set_write_timeout(&self, t: Duration) {
        let _ = match self {
            Stream::Tcp(s) => s.set_write_timeout(Some(t)),
            #[cfg(unix)]
            Stream::Unix(s) => s.set_write_timeout(Some(t)),
        };
    }
}

/// The text of `strerror()` for an I/O error.
pub(crate) fn strerror(e: &io::Error) -> String {
    let s = e.to_string();
    match s.find(" (os error ") {
        Some(at) => s[..at].to_string(),
        None => s,
    }
}

struct Out {
    buf: Vec<u8>,
    inflight: usize,
    force_pending: usize,
    closing: bool,
    closed: bool,
}

/// The output side of a client: what the protocol queued and the thread writing it.
pub(crate) struct ClientIo {
    stream: Stream,
    out: Mutex<Out>,
    cv: Condvar,
}

/// The most a close waits to send, the `vnc_flush()` before QEMU's close.
const CLOSE_DRAIN_LIMIT: usize = 64 * 1024;

impl ClientIo {
    /// Starts the writer thread.
    pub(crate) fn start(stream: Stream) -> Option<Arc<ClientIo>> {
        let io = Arc::new(ClientIo {
            stream,
            out: Mutex::new(Out {
                buf: Vec::new(),
                inflight: 0,
                force_pending: 0,
                closing: false,
                closed: false,
            }),
            cv: Condvar::new(),
        });
        let writer = Arc::clone(&io);
        std::thread::Builder::new().name("vnc-write".into()).spawn(move || writer.run()).ok()?;
        Some(io)
    }

    /// Queues bytes for the client.
    pub(crate) fn push(&self, data: Vec<u8>) {
        let mut o = lock(&self.out);
        if o.closing {
            return;
        }
        if o.buf.is_empty() {
            o.buf = data;
        } else {
            o.buf.extend_from_slice(&data);
        }
        self.cv.notify_all();
    }

    /// `vs->output.offset`: what is queued and not written yet.
    pub(crate) fn pending(&self) -> usize {
        let o = lock(&self.out);
        o.buf.len() + o.inflight
    }

    /// `vs->force_update_offset`.
    pub(crate) fn force_pending(&self) -> usize {
        lock(&self.out).force_pending
    }

    pub(crate) fn set_force_pending(&self, n: usize) {
        lock(&self.out).force_pending = n;
    }

    /// Closes the socket once the queued bytes are out. A client that has a lot queued is
    /// cut off at once, as QEMU's non-blocking flush would leave most of it behind too.
    pub(crate) fn close(&self) {
        let mut o = lock(&self.out);
        if o.closing {
            return;
        }
        o.closing = true;
        if o.buf.len() + o.inflight > CLOSE_DRAIN_LIMIT {
            o.buf.clear();
            o.closed = true;
            self.stream.shutdown();
        } else {
            self.stream.set_write_timeout(Duration::from_secs(1));
        }
        self.cv.notify_all();
    }

    fn run(&self) {
        loop {
            let data = {
                let mut o = lock(&self.out);
                while o.buf.is_empty() && !o.closing {
                    o = self.cv.wait(o).unwrap_or_else(std::sync::PoisonError::into_inner);
                }
                if o.closed || (o.buf.is_empty() && o.closing) {
                    break;
                }
                let data = std::mem::take(&mut o.buf);
                o.inflight = data.len();
                data
            };
            let mut done = 0;
            while done < data.len() {
                let end = data.len().min(done + 64 * 1024);
                match self.stream.write(&data[done..end]) {
                    Ok(0) | Err(_) => {
                        let mut o = lock(&self.out);
                        o.buf.clear();
                        o.inflight = 0;
                        o.closing = true;
                        o.closed = true;
                        break;
                    }
                    Ok(n) => {
                        done += n;
                        let mut o = lock(&self.out);
                        o.inflight -= n;
                        o.force_pending = o.force_pending.saturating_sub(n);
                    }
                }
            }
        }
        // The reader sees the end of the stream and tells the display the client is gone.
        lock(&self.out).closed = true;
        self.stream.shutdown();
    }
}

/// Feeds a client's bytes to its display until either side gives up.
pub(crate) fn spawn_reader(vd: Weak<VncDisplay>, id: u64, io: Arc<ClientIo>) {
    let weak = vd.clone();
    let spawned = std::thread::Builder::new().name("vnc-read".into()).spawn(move || {
        let mut buf = vec![0u8; 4096];
        loop {
            let r = io.stream.read(&mut buf);
            let Some(vd) = vd.upgrade() else { break };
            match r {
                Ok(n) if n > 0 => {
                    if !vd.client_input(id, &buf[..n]) {
                        break;
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                _ => {
                    vd.client_gone(id);
                    break;
                }
            }
        }
        io.close();
    });
    if spawned.is_err() {
        if let Some(vd) = weak.upgrade() {
            vd.client_gone(id);
        }
    }
}

/// Accepts clients for a display until the display goes away.
pub(crate) fn spawn_acceptor(vd: Weak<VncDisplay>, l: Listener) {
    let _ = std::thread::Builder::new().name("vnc-accept".into()).spawn(move || {
        loop {
            let accepted = match &l {
                Listener::Tcp(l) => l.accept().map(|(s, peer)| {
                    let _ = s.set_nodelay(true);
                    (Stream::Tcp(s), AddrInfo::inet(peer))
                }),
                #[cfg(unix)]
                Listener::Unix(l, _) => l.accept().map(|(s, peer)| {
                    let path = peer
                        .as_pathname()
                        .map(|p| p.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    (Stream::Unix(s), AddrInfo::unix(&path))
                }),
            };
            let Some(vd) = vd.upgrade() else { return };
            match accepted {
                Ok((stream, peer)) => vd.connect(stream, peer),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                // Out of descriptors and the like: try again a little later.
                Err(_) => std::thread::sleep(Duration::from_millis(100)),
            }
        }
    });
}

/// `inet_ai_family_from_address()`: which families to keep, IPv4 and IPv6.
fn families(host: &str, ipv4: Option<bool>, ipv6: Option<bool>) -> Result<(bool, bool)> {
    Ok(match (ipv4, ipv6) {
        (Some(false), Some(false)) => {
            return Err(Error::generic("Cannot disable IPv4 and IPv6 at same time"));
        }
        // An empty host is "::" with IPV6_V6ONLY off, which takes both.
        (Some(true), Some(true)) => (!host.is_empty(), true),
        (_, Some(true)) | (Some(false), _) => (false, true),
        (Some(true), _) | (_, Some(false)) => (true, false),
        (None, None) => (true, true),
    })
}

/// `qio_dns_resolver_lookup_sync()` with `AI_PASSIVE`.
fn resolve(
    host: &str,
    port: u32,
    ipv4: Option<bool>,
    ipv6: Option<bool>,
) -> Result<Vec<SocketAddr>> {
    let (want4, want6) = families(host, ipv4, ipv6)?;
    let all: Vec<SocketAddr> = if host.is_empty() {
        // getaddrinfo() with a null node and AI_PASSIVE: the wildcard of each family.
        vec![
            SocketAddr::from(([0, 0, 0, 0], port as u16)),
            SocketAddr::from(([0u16; 8], port as u16)),
        ]
    } else {
        match (host, port as u16).to_socket_addrs() {
            Ok(a) => a.collect(),
            Err(e) => {
                let msg = e.to_string();
                let msg =
                    msg.strip_prefix("failed to lookup address information: ").unwrap_or(&msg);
                return Err(Error::generic(format!(
                    "address resolution failed for {host}:{port}: {msg}"
                )));
            }
        }
    };
    let mut out = Vec::new();
    for a in all {
        let keep = if a.is_ipv4() { want4 } else { want6 };
        if keep && !out.contains(&a) {
            out.push(a);
        }
    }
    if out.is_empty() {
        return Err(Error::generic(format!(
            "address resolution failed for {host}:{port}: Address family for hostname not supported"
        )));
    }
    Ok(out)
}

/// `qemu_socket()`: a stream socket that is closed on exec.
#[cfg(unix)]
fn stream_socket(family: rustix::net::AddressFamily) -> io::Result<rustix::fd::OwnedFd> {
    use rustix::net::SocketType;
    #[cfg(not(target_vendor = "apple"))]
    let fd = rustix::net::socket_with(
        family,
        SocketType::STREAM,
        rustix::net::SocketFlags::CLOEXEC,
        None,
    )?;
    // There is no SOCK_CLOEXEC on macOS, so the flag goes on afterwards, as QEMU does there.
    #[cfg(target_vendor = "apple")]
    let fd = {
        let fd = rustix::net::socket(family, SocketType::STREAM, None)?;
        rustix::io::fcntl_setfd(&fd, rustix::io::FdFlags::CLOEXEC)?;
        fd
    };
    Ok(fd)
}

#[cfg(unix)]
fn socket_for(addr: &SocketAddr) -> io::Result<rustix::fd::OwnedFd> {
    use rustix::net::AddressFamily;
    let family = match addr {
        SocketAddr::V4(_) => AddressFamily::INET,
        SocketAddr::V6(_) => AddressFamily::INET6,
    };
    let fd = stream_socket(family)?;
    rustix::net::sockopt::set_socket_reuseaddr(&fd, true)?;
    Ok(fd)
}

/// `try_bind()`: an IPv6 socket takes IPv4 too unless the options picked one family, and gives
/// that up when IPv4 is taken already.
#[cfg(unix)]
fn try_bind(
    fd: &rustix::fd::OwnedFd,
    addr: &SocketAddr,
    ipv4: Option<bool>,
    ipv6: Option<bool>,
) -> io::Result<()> {
    if addr.is_ipv6() {
        let v6only = !matches!((ipv4, ipv6), (None, None) | (Some(true), Some(true)));
        rustix::net::sockopt::set_ipv6_v6only(fd, v6only)?;
        match rustix::net::bind(fd, addr) {
            Err(e) if e == rustix::io::Errno::ADDRINUSE && !v6only => {
                rustix::net::sockopt::set_ipv6_v6only(fd, true)?;
                rustix::net::bind(fd, addr)?;
            }
            r => r?,
        }
        return Ok(());
    }
    rustix::net::bind(fd, addr)?;
    Ok(())
}

/// `inet_listen_saddr()` for one resolved address: the first port of the range that works.
#[cfg(unix)]
fn listen_one(
    addr: SocketAddr,
    port_max: u32,
    ipv4: Option<bool>,
    ipv6: Option<bool>,
) -> Result<TcpListener> {
    let port_min = u32::from(addr.port());
    let mut created = false;
    let mut last = io::Error::from_raw_os_error(rustix::io::Errno::ADDRINUSE.raw_os_error());
    for p in port_min..=port_max.min(65535) {
        let mut a = addr;
        a.set_port(p as u16);
        let fd = match socket_for(&a) {
            Ok(fd) => fd,
            Err(e) => {
                if p == port_min {
                    last = e;
                    continue;
                }
                return Err(Error::generic(format!(
                    "Failed to recreate failed listening socket: {}",
                    strerror(&e)
                )));
            }
        };
        created = true;
        if let Err(e) = try_bind(&fd, &a, ipv4, ipv6) {
            if e.kind() == io::ErrorKind::AddrInUse {
                last = e;
                continue;
            }
            return Err(Error::generic(format!("Failed to bind socket: {}", strerror(&e))));
        }
        if let Err(e) = rustix::net::listen(&fd, 1) {
            let e = io::Error::from(e);
            if e.kind() == io::ErrorKind::AddrInUse {
                last = e;
                continue;
            }
            return Err(Error::generic(format!("Failed to listen on socket: {}", strerror(&e))));
        }
        return Ok(TcpListener::from(fd));
    }
    let what =
        if created { "Failed to find an available port" } else { "Failed to create a socket" };
    Err(Error::generic(format!("{what}: {}", strerror(&last))))
}

/// Windows has no `IPV6_V6ONLY` to play with through std, so each address is bound as it is.
#[cfg(not(unix))]
fn listen_one(
    addr: SocketAddr,
    port_max: u32,
    _ipv4: Option<bool>,
    _ipv6: Option<bool>,
) -> Result<TcpListener> {
    let port_min = u32::from(addr.port());
    let mut last = io::Error::from(io::ErrorKind::AddrInUse);
    for p in port_min..=port_max.min(65535) {
        let mut a = addr;
        a.set_port(p as u16);
        match TcpListener::bind(a) {
            Ok(l) => return Ok(l),
            Err(e) if e.kind() == io::ErrorKind::AddrInUse => last = e,
            Err(e) => {
                return Err(Error::generic(format!("Failed to bind socket: {}", strerror(&e))));
            }
        }
    }
    Err(Error::generic(format!("Failed to find an available port: {}", strerror(&last))))
}

/// `unix_listen_saddr()`.
#[cfg(unix)]
fn listen_unix(path: &str) -> Result<Listener> {
    const SUN_PATH: usize = if cfg!(target_os = "linux") { 108 } else { 104 };
    if path.len() > SUN_PATH {
        return Err(Error::generic(format!("UNIX socket path '{path}' is too long"))
            .hint(format!("Path must be less than {SUN_PATH} bytes\n")));
    }
    if let Err(e) = std::fs::remove_file(path) {
        if e.kind() != io::ErrorKind::NotFound {
            return Err(Error::generic(format!(
                "Failed to unlink socket {path}: {}",
                strerror(&e)
            )));
        }
    }
    let fd = stream_socket(rustix::net::AddressFamily::UNIX)
        .map_err(|e| Error::generic(format!("Failed to create Unix socket: {}", strerror(&e))))?;
    let addr = rustix::net::SocketAddrUnix::new(path).map_err(|e| {
        Error::generic(format!("Failed to bind socket to {path}: {}", strerror(&e.into())))
    })?;
    rustix::net::bind(&fd, &addr).map_err(|e| {
        Error::generic(format!("Failed to bind socket to {path}: {}", strerror(&e.into())))
    })?;
    rustix::net::listen(&fd, 1).map_err(|e| {
        Error::generic(format!("Failed to listen on socket: {}", strerror(&e.into())))
    })?;
    Ok(Listener::Unix(UnixListener::from(fd), path.to_string()))
}

/// `vnc_display_listen()` for the addresses of one display: every address that resolves and
/// binds becomes a listener, and the first error is reported when none does.
pub(crate) fn listen(addrs: &[ListenAddr]) -> Result<Vec<Listener>> {
    let mut listeners = Vec::new();
    for addr in addrs {
        match addr {
            ListenAddr::Inet { host, port, to, ipv4, ipv6 } => {
                let resolved = resolve(host, *port, *ipv4, *ipv6)?;
                let mut first_err = None;
                let mut any = false;
                for a in resolved {
                    match listen_one(a, to.unwrap_or(*port), *ipv4, *ipv6) {
                        Ok(l) => {
                            any = true;
                            listeners.push(Listener::Tcp(l));
                        }
                        Err(e) => {
                            first_err.get_or_insert(e);
                        }
                    }
                }
                if !any {
                    if let Some(e) = first_err {
                        return Err(e);
                    }
                }
            }
            #[cfg(unix)]
            ListenAddr::Unix(path) => listeners.push(listen_unix(path)?),
            #[cfg(not(unix))]
            ListenAddr::Unix(_) => {
                return Err(Error::generic("UNIX sockets are not supported on this platform"));
            }
        }
    }
    Ok(listeners)
}
