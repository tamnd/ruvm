// SPDX-License-Identifier: GPL-2.0-or-later

//! The socket chardev, chardev/char-socket.c, for Unix and TCP sockets.
//!
//! A client connects when the chardev is opened, and a server with `wait` accepts its first
//! client then. Either way the connection is kept until a frontend is attached, so bytes the
//! peer sends early wait in the socket as they do in QEMU. Only one client is served at a
//! time. A server goes back to accepting when its client leaves, and a client with
//! `reconnect-ms` connects again.

use std::io;
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
#[cfg(unix)]
use std::os::unix::net::{UnixListener, UnixStream};
#[cfg(unix)]
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use ruvm_base::report::info_report;
use ruvm_base::{Error, Result};
#[cfg(unix)]
use ruvm_qapi::types::UnixSocketAddress;
use ruvm_qapi::types::{
    ChardevSocket, InetSocketAddress, SocketAddress, SocketAddressLegacyU, SocketAddressU,
};

use crate::conn::{Connection, POLL_INTERVAL, Stream};

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Where the socket is, after a server knows its port.
#[derive(Clone, Debug, PartialEq)]
enum Addr {
    #[cfg(unix)]
    Unix(PathBuf),
    Inet {
        host: String,
        port: String,
    },
}

#[derive(Debug)]
enum Listener {
    #[cfg(unix)]
    Unix(UnixListener, PathBuf),
    Tcp(TcpListener),
}

impl Listener {
    fn accept(&self) -> io::Result<Stream> {
        match self {
            #[cfg(unix)]
            Listener::Unix(l, _) => {
                let (s, _) = l.accept()?;
                s.set_nonblocking(false)?;
                Ok(Stream::Unix(s))
            }
            Listener::Tcp(l) => {
                let (s, _) = l.accept()?;
                s.set_nonblocking(false)?;
                Ok(Stream::Tcp(s))
            }
        }
    }

    fn set_nonblocking(&self, on: bool) -> io::Result<()> {
        match self {
            #[cfg(unix)]
            Listener::Unix(l, _) => l.set_nonblocking(on),
            Listener::Tcp(l) => l.set_nonblocking(on),
        }
    }
}

impl Drop for Listener {
    fn drop(&mut self) {
        // socket_listen_cleanup() removes the socket file with the listener.
        #[cfg(unix)]
        if let Listener::Unix(_, path) = self {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[derive(Debug, Default)]
struct State {
    conn: Option<Stream>,
    /// `qemu_chr_compute_filename()` of the current connection.
    peer: Option<String>,
}

/// A socket chardev, `SocketChardev`.
#[derive(Debug)]
pub struct SocketChardev {
    addr: Addr,
    listen: bool,
    telnet: bool,
    nodelay: bool,
    reconnect: Option<Duration>,
    listener: Option<Listener>,
    state: Mutex<State>,
}

/// `qmp_chardev_validate_socket()` for the address types ruvm has.
fn validate(sock: &ChardevSocket) -> Result<()> {
    let tls = sock.tls_creds.is_some();
    match &sock.addr.u {
        SocketAddressLegacyU::Unix(_) if tls => {
            return Err(Error::generic(
                "'tls_creds' option is incompatible with 'unix' address type",
            ));
        }
        SocketAddressLegacyU::Vsock(_) if tls => {
            return Err(Error::generic(
                "'tls_creds' option is incompatible with 'vsock' address type",
            ));
        }
        _ => {}
    }
    if sock.tls_authz.is_some() && !tls {
        return Err(Error::generic("'tls_authz' option requires 'tls_creds' option"));
    }
    if sock.server.unwrap_or(true) {
        if sock.reconnect_ms.is_some() {
            return Err(Error::generic(
                "'reconnect-ms' option is incompatible with socket in server listen mode",
            ));
        }
    } else {
        if sock.websocket == Some(true) {
            return Err(Error::generic("Websocket client is not implemented"));
        }
        if sock.wait.is_some() {
            return Err(Error::generic(
                "'wait' option is incompatible with socket in client connect mode",
            ));
        }
    }
    Ok(())
}

fn not_supported(what: &str) -> Error {
    Error::generic(format!("{what} is not supported by ruvm yet"))
}

/// `inet_parse_port()` on the number part: QEMU takes only numeric ports.
fn inet_port(port: &str) -> Result<u16> {
    let n: i64 =
        port.parse().map_err(|_| Error::generic(format!("can't convert to a number: {port}")))?;
    u16::try_from(n).map_err(|_| Error::generic(format!("port {port} out of range")))
}

fn resolve(host: &str, port: &str) -> Result<Vec<SocketAddr>> {
    let p = inet_port(port)?;
    let host = host.trim_start_matches('[').trim_end_matches(']');
    let found = (host, p)
        .to_socket_addrs()
        .map_err(|e| Error::generic(format!("address resolution failed for {host}:{port}: {e}")))?;
    Ok(found.collect())
}

fn numeric(ip: IpAddr) -> String {
    ip.to_string()
}

impl SocketChardev {
    /// `tcp_chr_open()`.
    pub fn open(sock: &ChardevSocket) -> Result<SocketChardev> {
        validate(sock)?;
        if sock.tls_creds.is_some() {
            return Err(not_supported("TLS on a socket chardev"));
        }
        if sock.websocket == Some(true) {
            return Err(not_supported("A websocket chardev"));
        }
        if sock.tn3270 == Some(true) {
            return Err(not_supported("tn3270 on a socket chardev"));
        }
        let listen = sock.server.unwrap_or(true);
        let telnet = sock.telnet.unwrap_or(false);
        if telnet {
            return Err(not_supported("telnet on a socket chardev"));
        }
        let wait = sock.wait.unwrap_or(false);
        let nodelay = sock.nodelay.unwrap_or(false);
        let reconnect =
            sock.reconnect_ms.filter(|ms| *ms > 0).map(|ms| Duration::from_millis(ms as u64));
        let addr = match &sock.addr.u {
            #[cfg(unix)]
            SocketAddressLegacyU::Unix(u) => Addr::Unix(PathBuf::from(&u.data.path)),
            #[cfg(not(unix))]
            SocketAddressLegacyU::Unix(_) => return Err(not_supported("A Unix socket chardev")),
            SocketAddressLegacyU::Inet(i) => {
                Addr::Inet { host: i.data.host.clone(), port: i.data.port.clone() }
            }
            SocketAddressLegacyU::Vsock(_) => {
                return Err(Error::generic("socket family AF_VSOCK unsupported"));
            }
            SocketAddressLegacyU::Fd(_) => return Err(not_supported("A socket chardev on an fd")),
        };
        let mut chr = SocketChardev {
            addr,
            listen,
            telnet,
            nodelay,
            reconnect,
            listener: None,
            state: Mutex::new(State::default()),
        };
        if listen {
            chr.listen()?;
            if wait {
                let filename = chr.filename();
                info_report(&format!("QEMU waiting for connection on: {filename}"));
                let stream = chr
                    .listener
                    .as_ref()
                    .expect("just opened")
                    .accept()
                    .map_err(|e| Error::from_io("Unable to accept connection", e))?;
                chr.connected(stream)?;
            }
        } else {
            match chr.connect() {
                Ok(stream) => chr.connected(stream)?,
                // With reconnect-ms the first attempt may fail too, and the frontend's thread
                // tries again.
                Err(_) if reconnect.is_some() => {}
                Err(e) => return Err(e),
            }
        }
        Ok(chr)
    }

    fn listen(&mut self) -> Result<()> {
        match &self.addr {
            #[cfg(unix)]
            Addr::Unix(path) => {
                match std::fs::remove_file(path) {
                    Ok(()) => {}
                    Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                    Err(e) => {
                        return Err(Error::from_io(
                            format!("Failed to unlink socket {}", path.display()),
                            e,
                        ));
                    }
                }
                let l = UnixListener::bind(path).map_err(|e| {
                    Error::from_io(format!("Failed to bind socket to {}", path.display()), e)
                })?;
                self.listener = Some(Listener::Unix(l, path.clone()));
            }
            Addr::Inet { host, port } => {
                let addrs = if host.is_empty() {
                    let p = inet_port(port)?;
                    vec![SocketAddr::from(([0, 0, 0, 0], p))]
                } else {
                    resolve(host, port)?
                };
                let l = TcpListener::bind(&addrs[..])
                    .map_err(|e| Error::from_io("Failed to bind socket", e))?;
                let local =
                    l.local_addr().map_err(|e| Error::from_io("Failed to bind socket", e))?;
                // qio_net_listener_get_local_address(): the address as the kernel has it.
                self.addr =
                    Addr::Inet { host: numeric(local.ip()), port: local.port().to_string() };
                self.listener = Some(Listener::Tcp(l));
            }
        }
        Ok(())
    }

    fn connect(&self) -> Result<Stream> {
        match &self.addr {
            #[cfg(unix)]
            Addr::Unix(path) => UnixStream::connect(path).map(Stream::Unix).map_err(|e| {
                Error::from_io(format!("Failed to connect to '{}'", path.display()), e)
            }),
            Addr::Inet { host, port } => {
                let addrs = resolve(host, port)?;
                TcpStream::connect(&addrs[..])
                    .map(Stream::Tcp)
                    .map_err(|e| Error::from_io(format!("Failed to connect to '{host}:{port}'"), e))
            }
        }
    }

    /// `tcp_chr_new_client()`: keeps the stream and works out what `query-chardev` says of it.
    fn connected(&self, stream: Stream) -> Result<()> {
        let peer = match &stream {
            #[cfg(unix)]
            Stream::Unix(_) => {
                let Addr::Unix(path) = &self.addr else { unreachable!("a Unix stream") };
                let server = if self.listen { ",server=on" } else { "" };
                format!("unix:{}{server}", path.display())
            }
            Stream::Tcp(s) => {
                if self.nodelay {
                    s.set_nodelay(true).map_err(|e| Error::from_io("Failed to set nodelay", e))?;
                }
                let local = s.local_addr().ok();
                let remote = s.peer_addr().ok();
                let fmt = |a: Option<SocketAddr>| match a {
                    Some(SocketAddr::V4(a)) => format!("{}:{}", a.ip(), a.port()),
                    Some(SocketAddr::V6(a)) => format!("[{}]:{}", a.ip(), a.port()),
                    None => String::from(":"),
                };
                let server = if self.listen { ",server=on" } else { "" };
                format!("{}:{}{server} <-> {}", self.protocol(), fmt(local), fmt(remote))
            }
        };
        let mut st = lock(&self.state);
        st.conn = Some(stream);
        st.peer = Some(peer);
        Ok(())
    }

    fn disconnected(&self) {
        let mut st = lock(&self.state);
        if let Some(s) = st.conn.take() {
            s.shutdown();
        }
        st.peer = None;
    }

    /// `qemu_chr_socket_protocol()`.
    fn protocol(&self) -> &'static str {
        if self.telnet { "telnet" } else { "tcp" }
    }

    /// `tcp_chr_get_filename()`.
    pub fn filename(&self) -> String {
        if let Some(peer) = &lock(&self.state).peer {
            return peer.clone();
        }
        let server = if self.listen { ",server=on" } else { "" };
        match &self.addr {
            #[cfg(unix)]
            Addr::Unix(path) => format!("disconnected:unix:{}{server}", path.display()),
            Addr::Inet { host, port } => {
                format!("disconnected:{}:{host}:{port}{server}", self.protocol())
            }
        }
    }

    /// The `addr` property: where a server listens, or what a client connects to.
    pub fn address(&self) -> SocketAddress {
        let u = match &self.addr {
            #[cfg(unix)]
            // `abstract` and `tight` only exist on Linux.
            #[allow(clippy::needless_update)]
            Addr::Unix(path) => SocketAddressU::Unix(UnixSocketAddress {
                path: path.display().to_string(),
                ..Default::default()
            }),
            Addr::Inet { host, port } => {
                // socket_sockaddr_to_address_inet() says which family the listener is on.
                let ip = host.trim_start_matches('[').trim_end_matches(']').parse::<IpAddr>();
                let local = |v4: bool| {
                    (self.listen && ip.as_ref().is_ok_and(|ip| ip.is_ipv4() == v4)).then_some(true)
                };
                SocketAddressU::Inet(InetSocketAddress {
                    host: host.clone(),
                    port: port.clone(),
                    ipv4: local(true),
                    ipv6: local(false),
                    ..Default::default()
                })
            }
        };
        SocketAddress { u }
    }

    /// Whether a client is connected, the `connected` property.
    pub fn is_connected(&self) -> bool {
        lock(&self.state).conn.is_some()
    }

    /// Waits for a client on a server, looking at `stop` now and then.
    fn accept(&self, stop: &AtomicBool) -> Option<Stream> {
        let l = self.listener.as_ref()?;
        if l.set_nonblocking(true).is_err() {
            return None;
        }
        let res = loop {
            if stop.load(Ordering::Acquire) {
                break None;
            }
            match l.accept() {
                Ok(s) => break Some(s),
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    std::thread::sleep(POLL_INTERVAL / 4)
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(_) => break None,
            }
        };
        let _ = l.set_nonblocking(false);
        res
    }

    /// The next connection to hand to the frontend, or `None` when there will be none: the
    /// frontend is going away, or a client without `reconnect-ms` lost its server.
    fn next_stream(&self, stop: &AtomicBool) -> Option<Stream> {
        loop {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            if let Some(s) = lock(&self.state).conn.as_ref() {
                return s.try_clone().ok();
            }
            let stream = if self.listen {
                self.accept(stop)?
            } else {
                let delay = self.reconnect?;
                match self.connect() {
                    Ok(s) => s,
                    Err(_) => {
                        sleep_unless(stop, delay);
                        continue;
                    }
                }
            };
            if self.connected(stream).is_err() {
                self.disconnected();
            }
        }
    }

    /// Serves the frontend until `stop` is set: each connection is handed to `serve`, and a
    /// connection the peer closed is dropped before looking for the next one.
    pub(crate) fn run(
        &self,
        stop: &Arc<AtomicBool>,
        mut serve: impl FnMut(&mut Connection) -> io::Result<()>,
    ) {
        while let Some(stream) = self.next_stream(stop) {
            let Ok(mut conn) = Connection::new(stream, stop.clone()) else {
                self.disconnected();
                continue;
            };
            // An I/O error ends the connection the same way the peer closing it does.
            let _ = serve(&mut conn);
            if stop.load(Ordering::Acquire) {
                break;
            }
            self.disconnected();
            if !self.listen {
                if let Some(delay) = self.reconnect {
                    sleep_unless(stop, delay);
                }
            }
        }
    }
}

fn sleep_unless(stop: &AtomicBool, total: Duration) {
    let mut left = total;
    while !left.is_zero() && !stop.load(Ordering::Acquire) {
        let step = left.min(POLL_INTERVAL);
        std::thread::sleep(step);
        left -= step;
    }
}
