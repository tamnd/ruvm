// SPDX-License-Identifier: GPL-2.0-or-later

//! What the socket, stream and dgram backends share.
//!
//! All three move frames over a socket, either as datagrams or on a byte stream where every
//! frame has a 4 byte big-endian length in front, and all three may sit on a listening socket
//! waiting for the other side. [`Sock`] does that part: the I/O thread, the framing, accepting,
//! and connecting a stream client again after the other side went away. The backend modules
//! set it up and pick the info strings.

use std::io::{self, IoSlice};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, ToSocketAddrs};
use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use rustix::io::{Errno, FdFlags};
use rustix::net::{AddressFamily, RecvFlags, SendFlags, SocketAddrAny, SocketAddrUnix, SocketType};
use ruvm_base::{Error, Result};
use ruvm_qapi::types::{InetSocketAddress, NetClientDriver};

use crate::client::{NET_BUFSIZE, NetClient, NetClientOps, lock};
use crate::net::Net;
use crate::poll::{Interest, IoHandler, IoThread, unblock};
use crate::util::SocketReadState;

/// How frames travel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Framing {
    /// A length in front of every frame on a byte stream.
    Stream,
    /// One frame per datagram.
    Dgram,
}

/// Whose rules apply when a connection comes and goes: net/socket.c or net/stream.c. They
/// differ only in the info strings and in stream clients connecting again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Flavour {
    Socket,
    Stream,
}

/// Makes a connected socket for a stream client. It runs on the I/O thread, so a slow connect
/// holds up nobody else.
pub(crate) type Connector = Box<dyn Fn() -> Result<OwnedFd> + Send + Sync>;

/// How to set a [`Sock`] up.
pub(crate) struct SockConfig {
    pub(crate) framing: Framing,
    pub(crate) flavour: Flavour,
    /// The connected (or connecting, or datagram) socket.
    pub(crate) fd: Option<OwnedFd>,
    /// A listening socket to take one connection at a time from.
    pub(crate) listen_fd: Option<OwnedFd>,
    /// A stream client connects with this from the I/O thread, and again after hang-ups when
    /// `reconnect_ms` is not zero.
    pub(crate) connector: Option<Connector>,
    pub(crate) reconnect_ms: u64,
    /// `fd` is still connecting; it is ready once it turns writable.
    pub(crate) connecting: bool,
    /// Where datagrams go. Without it they go wherever the socket is connected to.
    pub(crate) dest: Option<SocketAddrAny>,
    pub(crate) info: String,
    /// A Unix socket path to remove at cleanup.
    pub(crate) unlink: Option<PathBuf>,
}

impl SockConfig {
    pub(crate) fn new(framing: Framing, flavour: Flavour) -> Self {
        SockConfig {
            framing,
            flavour,
            fd: None,
            listen_fd: None,
            connector: None,
            reconnect_ms: 0,
            connecting: false,
            dest: None,
            info: String::new(),
            unlink: None,
        }
    }
}

/// The state of a socket, stream or dgram client.
pub(crate) struct Sock {
    this: Weak<Sock>,
    nc: Weak<NetClient>,
    framing: Framing,
    flavour: Flavour,
    fd: Mutex<Option<Arc<OwnedFd>>>,
    listen_fd: Mutex<Option<Arc<OwnedFd>>>,
    accepting: AtomicBool,
    connecting: AtomicBool,
    read_poll: AtomicBool,
    write_poll: AtomicBool,
    rs: Mutex<SocketReadState>,
    send_index: Mutex<usize>,
    dest: Option<SocketAddrAny>,
    connector: Option<Connector>,
    reconnect_ms: u64,
    reconnect_at: Mutex<Option<Instant>>,
    unlink: Mutex<Option<PathBuf>>,
    io: OnceLock<IoThread>,
}

impl std::fmt::Debug for Sock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sock")
            .field("framing", &self.framing)
            .field("flavour", &self.flavour)
            .field("fd", &self.fd().map(|f| f.as_raw_fd()))
            .finish_non_exhaustive()
    }
}

/// Makes the client and starts its I/O thread.
pub(crate) fn new_sock(
    net: &mut Net,
    driver: NetClientDriver,
    peer: Option<&Arc<NetClient>>,
    model: &str,
    name: &str,
    cfg: SockConfig,
) -> Result<(Arc<NetClient>, Arc<Sock>)> {
    let SockConfig {
        framing,
        flavour,
        fd,
        listen_fd,
        connector,
        reconnect_ms,
        connecting,
        dest,
        info,
        unlink,
    } = cfg;
    let has_fd = fd.is_some();
    let has_listen = listen_fd.is_some();
    let connect_now = connector.is_some();
    let mut state = None;
    let nc = net.new_client(driver, peer, model, Some(name), |w| {
        let s = Arc::new_cyclic(|this| Sock {
            this: this.clone(),
            nc: w.clone(),
            framing,
            flavour,
            fd: Mutex::new(fd.map(Arc::new)),
            listen_fd: Mutex::new(listen_fd.map(Arc::new)),
            accepting: AtomicBool::new(!has_fd && has_listen),
            connecting: AtomicBool::new(connecting),
            read_poll: AtomicBool::new(has_fd && !connecting),
            write_poll: AtomicBool::new(false),
            rs: Mutex::new(SocketReadState::new(false)),
            send_index: Mutex::new(0),
            dest,
            connector,
            reconnect_ms,
            reconnect_at: Mutex::new(connect_now.then(Instant::now)),
            unlink: Mutex::new(unlink),
            io: OnceLock::new(),
        });
        state = Some(s.clone());
        s
    });
    let s = state.expect("make_ops ran");
    nc.set_info_str(&info);
    if !has_fd {
        nc.set_link_down(true);
    }
    match IoThread::spawn(name, s.clone()) {
        Ok(io) => {
            let _ = s.io.set(io);
        }
        Err(e) => {
            net.del_client(&nc);
            return Err(Error::from_io("could not start the network I/O thread", e));
        }
    }
    Ok((nc, s))
}

impl Sock {
    fn fd(&self) -> Option<Arc<OwnedFd>> {
        lock(&self.fd).clone()
    }

    fn listen_fd(&self) -> Option<Arc<OwnedFd>> {
        lock(&self.listen_fd).clone()
    }

    fn wake(&self) {
        if let Some(io) = self.io.get() {
            io.wake();
        }
    }

    fn read_poll(&self, enable: bool) {
        self.read_poll.store(enable, Ordering::SeqCst);
        self.wake();
    }

    fn write_poll(&self, enable: bool) {
        self.write_poll.store(enable, Ordering::SeqCst);
        self.wake();
    }

    fn set_info(&self, s: &str) {
        if let Some(nc) = self.nc.upgrade() {
            nc.set_info_str(s);
        }
    }

    /// Hands a received frame to the peer and stops reading if the peer is full, the
    /// `rs_finalize` and `send_dgram` callbacks.
    fn deliver(&self, nc: &NetClient, pkt: &[u8]) {
        let me = self.this.clone();
        let sent = nc.send_packet_async(
            pkt,
            Some(Box::new(move |_| {
                if let Some(s) = me.upgrade() {
                    if !s.read_poll.load(Ordering::SeqCst) {
                        s.read_poll(true);
                    }
                }
            })),
        );
        if sent == 0 {
            self.read_poll.store(false, Ordering::SeqCst);
        }
    }

    /// `net_socket_send()` and `net_stream_data_send()`.
    fn stream_send(&self, fd: &OwnedFd) {
        let Some(nc) = self.nc.upgrade() else {
            return;
        };
        let mut buf = vec![0u8; NET_BUFSIZE];
        let size = match rustix::net::recv(fd, &mut buf[..], RecvFlags::empty()) {
            Ok((n, _)) => n,
            Err(Errno::AGAIN | Errno::INTR) => return,
            Err(_) => 0,
        };
        if size == 0 {
            self.end_of_connection();
            return;
        }
        let mut rs = lock(&self.rs);
        let r = rs.fill(&buf[..size], |pkt| self.deliver(&nc, pkt));
        drop(rs);
        if r.is_err() {
            self.end_of_connection();
        }
    }

    /// `net_socket_send_dgram()` and `net_dgram_send()`.
    fn dgram_send(&self, fd: &OwnedFd) {
        let Some(nc) = self.nc.upgrade() else {
            return;
        };
        let mut buf = vec![0u8; NET_BUFSIZE];
        let size = match rustix::net::recv(fd, &mut buf[..], RecvFlags::empty()) {
            Ok((n, _)) => n,
            Err(_) => return,
        };
        if size == 0 {
            // QEMU treats an empty datagram as the end of the connection.
            self.read_poll.store(false, Ordering::SeqCst);
            self.write_poll.store(false, Ordering::SeqCst);
            return;
        }
        self.deliver(&nc, &buf[..size]);
    }

    /// The `eoc:` path: drops the connection and goes back to listening, or to connecting
    /// again for a stream client with `reconnect-ms`.
    fn end_of_connection(&self) {
        self.read_poll.store(false, Ordering::SeqCst);
        self.write_poll.store(false, Ordering::SeqCst);
        lock(&self.fd).take();
        lock(&self.rs).reset();
        *lock(&self.send_index) = 0;
        if let Some(nc) = self.nc.upgrade() {
            nc.set_link_down(true);
        }
        let listening = self.listen_fd().is_some();
        match self.flavour {
            Flavour::Socket => self.set_info(""),
            Flavour::Stream => {
                if listening {
                    self.set_info("listening");
                }
            }
        }
        if listening {
            self.accepting.store(true, Ordering::SeqCst);
        }
        if self.flavour == Flavour::Stream {
            self.arm_reconnect();
        }
    }

    /// `net_stream_arm_reconnect()`.
    fn arm_reconnect(&self) {
        if self.reconnect_ms == 0 {
            return;
        }
        let mut at = lock(&self.reconnect_at);
        if at.is_none() {
            self.set_info("connecting");
            *at = Some(Instant::now() + Duration::from_millis(self.reconnect_ms));
        }
    }

    /// Starts using a new connection.
    fn install(&self, fd: OwnedFd) {
        *lock(&self.fd) = Some(Arc::new(fd));
        lock(&self.rs).reset();
        *lock(&self.send_index) = 0;
        if let Some(nc) = self.nc.upgrade() {
            nc.set_link_down(false);
        }
        self.read_poll.store(true, Ordering::SeqCst);
    }

    /// `net_socket_accept()` and `net_stream_listen()`.
    fn accept(&self, listen: &OwnedFd) {
        let (fd, from) = loop {
            match rustix::net::acceptfrom(listen) {
                Ok(r) => break r,
                Err(Errno::INTR) => continue,
                Err(_) => return,
            }
        };
        let _ = fd_cloexec(&fd);
        if unblock(&fd).is_err() {
            return;
        }
        self.accepting.store(false, Ordering::SeqCst);
        let info = match self.flavour {
            Flavour::Socket => match from.map(SocketAddrV4::try_from) {
                Some(Ok(a)) => format!("socket: connection from {}:{}", a.ip(), a.port()),
                _ => "socket: connection from 0.0.0.0:0".to_string(),
            },
            Flavour::Stream => {
                let local = rustix::net::getsockname(&fd).ok();
                let addr = if local.as_ref().map(SocketAddrAny::address_family)
                    == Some(AddressFamily::UNIX)
                {
                    local
                } else {
                    rustix::net::getpeername(&fd).ok().flatten()
                };
                addr.map(|a| socket_uri(&a)).unwrap_or_default()
            }
        };
        self.install(fd);
        self.set_info(&info);
    }

    /// A stream client connecting, from `net_stream_client_connected()`.
    fn connect(&self) {
        let Some(connector) = &self.connector else {
            return;
        };
        let fd = connector().and_then(|fd| {
            unblock(&fd)?;
            Ok(fd)
        });
        match fd {
            Ok(fd) => {
                let _ = rustix::net::sockopt::set_tcp_nodelay(&fd, true);
                let info = peer_uri(&fd);
                self.install(fd);
                self.set_info(&info);
            }
            Err(e) => {
                self.set_info(&format!("error: {}", e.message()));
                self.arm_reconnect();
            }
        }
    }
}

impl IoHandler for Sock {
    fn interest(&self) -> Interest {
        let timeout = lock(&self.reconnect_at).map(|t| t.saturating_duration_since(Instant::now()));
        if let Some(fd) = self.fd() {
            if self.connecting.load(Ordering::SeqCst) {
                return Interest { fd: Some(fd), read: false, write: true, timeout };
            }
            return Interest {
                fd: Some(fd),
                read: self.read_poll.load(Ordering::SeqCst),
                write: self.write_poll.load(Ordering::SeqCst),
                timeout,
            };
        }
        if self.accepting.load(Ordering::SeqCst) {
            if let Some(l) = self.listen_fd() {
                return Interest { fd: Some(l), read: true, write: false, timeout };
            }
        }
        Interest { fd: None, read: false, write: false, timeout }
    }

    fn readable(&self, fd: &Arc<OwnedFd>) {
        if self.listen_fd().is_some_and(|l| Arc::ptr_eq(&l, fd)) {
            self.accept(fd);
            return;
        }
        if !self.fd().is_some_and(|f| Arc::ptr_eq(&f, fd)) {
            return;
        }
        match self.framing {
            Framing::Stream => self.stream_send(fd),
            Framing::Dgram => self.dgram_send(fd),
        }
    }

    fn writable(&self, fd: &Arc<OwnedFd>) {
        if !self.fd().is_some_and(|f| Arc::ptr_eq(&f, fd)) {
            return;
        }
        if self.connecting.swap(false, Ordering::SeqCst) {
            // net_socket_connect(): whatever happened, reading will find out.
            self.read_poll.store(true, Ordering::SeqCst);
            return;
        }
        self.write_poll.store(false, Ordering::SeqCst);
        if let Some(nc) = self.nc.upgrade() {
            nc.flush_queued_packets();
        }
    }

    fn timeout(&self) {
        {
            let mut at = lock(&self.reconnect_at);
            match *at {
                Some(t) if Instant::now() >= t => *at = None,
                _ => return,
            }
        }
        self.connect();
    }
}

impl NetClientOps for Sock {
    fn receive(&self, _nc: &NetClient, iov: &[&[u8]]) -> isize {
        let Some(fd) = self.fd() else {
            return -(libc::EBADF as isize);
        };
        let buf: Vec<u8> = iov.concat();
        match self.framing {
            Framing::Stream => self.receive_stream(&fd, &buf),
            Framing::Dgram => self.receive_dgram(&fd, &buf),
        }
    }

    fn cleanup(&self, nc: &NetClient) {
        let _ = nc;
        lock(&self.reconnect_at).take();
        self.read_poll.store(false, Ordering::SeqCst);
        self.write_poll.store(false, Ordering::SeqCst);
        self.accepting.store(false, Ordering::SeqCst);
        if let Some(io) = self.io.get() {
            io.stop();
        }
        lock(&self.fd).take();
        lock(&self.listen_fd).take();
        if let Some(path) = lock(&self.unlink).take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

impl Sock {
    /// `net_socket_receive()` and `net_stream_data_receive()`: the length and then the
    /// frame, picking up where a short write left off.
    fn receive_stream(&self, fd: &OwnedFd, buf: &[u8]) -> isize {
        let hdr = (buf.len() as u32).to_be_bytes();
        let mut index = lock(&self.send_index);
        let total = hdr.len() + buf.len();
        let remaining = total - *index;
        let slices: Vec<IoSlice<'_>> = if *index < hdr.len() {
            vec![IoSlice::new(&hdr[*index..]), IoSlice::new(buf)]
        } else {
            vec![IoSlice::new(&buf[*index - hdr.len()..])]
        };
        let sent = loop {
            match rustix::io::writev(fd, &slices) {
                Ok(n) => break n,
                Err(Errno::INTR) => continue,
                Err(Errno::AGAIN) => break 0,
                Err(e) => {
                    *index = 0;
                    return -(e.raw_os_error() as isize);
                }
            }
        };
        if sent < remaining {
            *index += sent;
            drop(index);
            self.write_poll(true);
            return 0;
        }
        *index = 0;
        buf.len() as isize
    }

    /// `net_socket_receive_dgram()` and `net_dgram_receive()`.
    fn receive_dgram(&self, fd: &OwnedFd, buf: &[u8]) -> isize {
        loop {
            let r = match &self.dest {
                Some(a) => rustix::net::sendto(fd, buf, SendFlags::empty(), a),
                None => rustix::net::send(fd, buf, SendFlags::empty()),
            };
            match r {
                Ok(n) => return n as isize,
                Err(Errno::INTR) => continue,
                Err(Errno::AGAIN) => {
                    self.write_poll(true);
                    return 0;
                }
                Err(e) => return -(e.raw_os_error() as isize),
            }
        }
    }
}

// Addresses and sockets.

/// Marks a descriptor close-on-exec, which QEMU's `qemu_socket()` does too.
pub(crate) fn fd_cloexec(fd: impl AsFd) -> io::Result<()> {
    rustix::io::fcntl_setfd(fd, FdFlags::CLOEXEC).map_err(io::Error::from)
}

/// `qemu_socket()`.
pub(crate) fn new_socket(family: AddressFamily, ty: SocketType) -> io::Result<OwnedFd> {
    let fd = rustix::net::socket(family, ty, None)?;
    fd_cloexec(&fd)?;
    Ok(fd)
}

/// `socket_uri()` for an address the kernel handed back.
pub(crate) fn socket_uri(addr: &SocketAddrAny) -> String {
    if let Ok(a) = SocketAddr::try_from(addr.clone()) {
        return format!("tcp:{}:{}", a.ip(), a.port());
    }
    if let Ok(u) = SocketAddrUnix::try_from(addr.clone()) {
        return format!("unix:{}", unix_path(&u));
    }
    "unknown address type".to_string()
}

/// `socket_uri()` of the remote address. An unnamed Unix peer, as from `socketpair()`, has
/// no address at all on some hosts, where QEMU would still print `unix:`.
fn peer_uri(fd: &OwnedFd) -> String {
    match rustix::net::getpeername(fd) {
        Ok(Some(a)) => socket_uri(&a),
        _ => match local_address(fd) {
            Ok(a) if a.address_family() == AddressFamily::UNIX => "unix:".to_string(),
            Ok(a) => socket_uri(&a),
            Err(_) => String::new(),
        },
    }
}

/// The path of a Unix socket address, empty for an unnamed one.
pub(crate) fn unix_path(u: &SocketAddrUnix) -> String {
    match u.path() {
        Some(p) => p.to_string_lossy().into_owned(),
        None => u
            .path_bytes()
            .map(|b| String::from_utf8_lossy(b.strip_prefix(&[0]).unwrap_or(b)).into_owned())
            .unwrap_or_default(),
    }
}

/// `SocketAddressType_str()` of the family of a local address.
pub(crate) fn address_type_str(addr: &SocketAddrAny) -> &'static str {
    match addr.address_family() {
        AddressFamily::INET | AddressFamily::INET6 => "inet",
        AddressFamily::UNIX => "unix",
        _ => "fd",
    }
}

/// `socket_local_address()`.
pub(crate) fn local_address(fd: impl AsFd) -> Result<SocketAddrAny> {
    rustix::net::getsockname(fd)
        .map_err(|e| Error::from_io("Unable to query local socket address", e.into()))
}

/// Makes a Unix socket address, with QEMU's complaint when the path does not fit.
pub(crate) fn unix_addr(path: &str) -> Result<SocketAddrUnix> {
    let max = size_of::<libc::sockaddr_un>() - std::mem::offset_of!(libc::sockaddr_un, sun_path);
    let too_long = || {
        Error::generic(format!("UNIX socket path '{path}' is too long"))
            .hint(format!("Path must be less than {max} bytes\n"))
    };
    if path.len() > max {
        return Err(too_long());
    }
    SocketAddrUnix::new(path).map_err(|_| too_long())
}

/// `socket_get_fd()`: a descriptor given by number or by name, which must be a socket.
pub(crate) fn socket_get_fd(net: &mut Net, fdstr: &str) -> Result<OwnedFd> {
    let raw = net.fd_param(fdstr)?;
    let not_socket = || Error::generic(format!("File descriptor '{fdstr}' is not a socket"));
    let fd = crate::fd::adopt(raw).map_err(|_| not_socket())?;
    if rustix::net::sockopt::socket_type(&fd).is_err() {
        return Err(not_socket());
    }
    Ok(fd)
}

/// `inet_ai_family_from_address()`: which families `ipv4=` and `ipv6=` allow.
fn inet_families(addr: &InetSocketAddress) -> Result<(bool, bool)> {
    match (addr.ipv4, addr.ipv6) {
        (Some(false), Some(false)) => {
            Err(Error::generic("Cannot disable IPv4 and IPv6 at same time"))
        }
        (Some(true), Some(true)) => Ok((true, true)),
        (_, Some(true)) | (Some(false), _) => Ok((false, true)),
        (Some(true), _) | (_, Some(false)) => Ok((true, false)),
        (None, None) => Ok((true, true)),
    }
}

/// Resolves an inet address the way getaddrinfo() does for QEMU, numbers only for ports.
fn inet_resolve(addr: &InetSocketAddress, passive: bool) -> Result<Vec<SocketAddr>> {
    let (v4, v6) = inet_families(addr)?;
    let fail = |why: &str| {
        Error::generic(format!("address resolution failed for {}:{}: {why}", addr.host, addr.port))
    };
    let port: u16 = if addr.port.is_empty() && passive {
        0
    } else {
        addr.port.parse().map_err(|_| fail("Servname not supported for ai_socktype"))?
    };
    let found: Vec<SocketAddr> = if addr.host.is_empty() && passive {
        vec![
            SocketAddr::from((Ipv4Addr::UNSPECIFIED, port)),
            SocketAddr::from((std::net::Ipv6Addr::UNSPECIFIED, port)),
        ]
    } else {
        (addr.host.as_str(), port).to_socket_addrs().map_err(|e| fail(&e.to_string()))?.collect()
    };
    let found: Vec<SocketAddr> =
        found.into_iter().filter(|a| if a.is_ipv4() { v4 } else { v6 }).collect();
    if found.is_empty() {
        return Err(fail("Address family for hostname not supported"));
    }
    Ok(found)
}

fn family_of(a: &SocketAddr) -> AddressFamily {
    if a.is_ipv4() { AddressFamily::INET } else { AddressFamily::INET6 }
}

/// `inet_listen_saddr()`, with the `to=` port range.
pub(crate) fn inet_listen(addr: &InetSocketAddress) -> Result<OwnedFd> {
    if addr.host.is_empty() && addr.port.is_empty() {
        // getaddrinfo() needs one of them.
        return Err(Error::generic(format!(
            "address resolution failed for {}:{}: Name or service not known",
            addr.host, addr.port
        )));
    }
    let mut created = false;
    let mut last = io::Error::from_raw_os_error(libc::EADDRINUSE);
    for a in inet_resolve(addr, true)? {
        let first = a.port();
        let last_port = addr.to.unwrap_or(first).max(first);
        for p in first..=last_port {
            let fd = match new_socket(family_of(&a), SocketType::STREAM) {
                Ok(fd) => fd,
                Err(e) if p == first => {
                    last = e;
                    break;
                }
                Err(e) => {
                    return Err(Error::from_io("Failed to recreate failed listening socket", e));
                }
            };
            created = true;
            let _ = rustix::net::sockopt::set_socket_reuseaddr(&fd, true);
            if a.is_ipv6() {
                let _ = rustix::net::sockopt::set_ipv6_v6only(&fd, addr.ipv4 == Some(false));
            }
            let mut sa = a;
            sa.set_port(p);
            if let Err(e) = rustix::net::bind(&fd, &sa) {
                if e == Errno::ADDRINUSE {
                    last = e.into();
                    continue;
                }
                return Err(Error::from_io("Failed to bind socket", e.into()));
            }
            if let Err(e) = rustix::net::listen(&fd, 1) {
                if e == Errno::ADDRINUSE {
                    last = e.into();
                    continue;
                }
                return Err(Error::from_io("Failed to listen on socket", e.into()));
            }
            return Ok(fd);
        }
    }
    let msg =
        if created { "Failed to find an available port" } else { "Failed to create a socket" };
    Err(Error::from_io(msg, last))
}

/// `inet_connect_saddr()`: tries every address in turn, blocking.
pub(crate) fn inet_connect(addr: &InetSocketAddress) -> Result<OwnedFd> {
    if addr.host.is_empty() || addr.port.is_empty() {
        return Err(Error::generic("host and/or port not specified"));
    }
    let mut last = io::Error::from_raw_os_error(libc::ECONNREFUSED);
    for a in inet_resolve(addr, false)? {
        let fd = match new_socket(family_of(&a), SocketType::STREAM) {
            Ok(fd) => fd,
            Err(e) => {
                last = e;
                continue;
            }
        };
        loop {
            match rustix::net::connect(&fd, &a) {
                Ok(()) => return Ok(fd),
                Err(Errno::INTR) => continue,
                Err(e) => {
                    last = e.into();
                    break;
                }
            }
        }
    }
    Err(Error::from_io(format!("Failed to connect to '{}:{}'", addr.host, addr.port), last))
}

/// `unix_listen_saddr()`.
pub(crate) fn unix_listen(path: &str) -> Result<OwnedFd> {
    let fd = new_socket(AddressFamily::UNIX, SocketType::STREAM)
        .map_err(|e| Error::from_io("Failed to create Unix socket", e))?;
    let sa = unix_addr(path)?;
    if let Err(e) = std::fs::remove_file(path) {
        if e.kind() != io::ErrorKind::NotFound {
            return Err(Error::from_io(format!("Failed to unlink socket {path}"), e));
        }
    }
    rustix::net::bind(&fd, &sa)
        .map_err(|e| Error::from_io(format!("Failed to bind socket to {path}"), e.into()))?;
    rustix::net::listen(&fd, 1)
        .map_err(|e| Error::from_io("Failed to listen on socket", e.into()))?;
    Ok(fd)
}

/// `unix_connect_saddr()`.
pub(crate) fn unix_connect(path: &str) -> Result<OwnedFd> {
    let fd = new_socket(AddressFamily::UNIX, SocketType::STREAM)
        .map_err(|e| Error::from_io("Failed to create socket", e))?;
    let sa = unix_addr(path)?;
    loop {
        match rustix::net::connect(&fd, &sa) {
            Ok(()) => return Ok(fd),
            Err(Errno::INTR) => continue,
            Err(e) => {
                return Err(Error::from_io(format!("Failed to connect to '{path}'"), e.into()));
            }
        }
    }
}

/// `net_socket_mcast_create()` and `net_dgram_mcast_create()`.
pub(crate) fn mcast_create(mcast: SocketAddrV4, local: Option<Ipv4Addr>) -> Result<OwnedFd> {
    if !mcast.ip().is_multicast() {
        return Err(Error::generic(format!(
            "specified mcastaddr {} (0x{:08x}) does not contain a multicast address",
            mcast.ip(),
            u32::from(*mcast.ip())
        )));
    }
    let fd = new_socket(AddressFamily::INET, SocketType::DGRAM)
        .map_err(|e| Error::from_io("can't create datagram socket", e))?;
    rustix::net::sockopt::set_socket_reuseaddr(&fd, true)
        .map_err(|e| Error::from_io("can't set socket option SO_REUSEADDR", e.into()))?;
    rustix::net::bind(&fd, &mcast)
        .map_err(|e| Error::from_io(format!("can't bind ip={} to socket", mcast.ip()), e.into()))?;
    let iface = local.unwrap_or(Ipv4Addr::UNSPECIFIED);
    rustix::net::sockopt::set_ip_add_membership(&fd, mcast.ip(), &iface).map_err(|e| {
        Error::from_io(format!("can't add socket to multicast group {}", mcast.ip()), e.into())
    })?;
    rustix::net::sockopt::set_ip_multicast_loop(&fd, true)
        .map_err(|e| Error::from_io("can't force multicast message to loopback", e.into()))?;
    if let Some(l) = local {
        rustix::net::sockopt::set_ip_multicast_if(&fd, &l).map_err(|e| {
            Error::from_io("can't set the default network send interface", e.into())
        })?;
    }
    unblock(&fd)?;
    Ok(fd)
}

/// A UDP socket bound to `laddr`, as the socket `udp=` and dgram inet paths make.
pub(crate) fn udp_bind(laddr: SocketAddrV4) -> Result<OwnedFd> {
    let fd = new_socket(AddressFamily::INET, SocketType::DGRAM)
        .map_err(|e| Error::from_io("can't create datagram socket", e))?;
    rustix::net::sockopt::set_socket_reuseaddr(&fd, true)
        .map_err(|e| Error::from_io("can't set socket option SO_REUSEADDR", e.into()))?;
    rustix::net::bind(&fd, &laddr)
        .map_err(|e| Error::from_io(format!("can't bind ip={} to socket", laddr.ip()), e.into()))?;
    unblock(&fd)?;
    Ok(fd)
}
