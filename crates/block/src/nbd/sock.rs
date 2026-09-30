// SPDX-License-Identifier: GPL-2.0-or-later

//! The parts of util/qemu-sockets.c that NBD needs: connecting to and listening on a
//! `SocketAddress`, with QEMU's error text.

use std::io;
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
#[cfg(unix)]
use std::os::unix::net::{UnixListener, UnixStream};

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{InetSocketAddress, SocketAddress, SocketAddressU, UnixSocketAddress};

use super::proto::NbdStream;

/// `sizeof(struct sockaddr_un.sun_path)`.
#[cfg(any(target_os = "linux", target_os = "android"))]
const SUN_PATH_LEN: usize = 108;
/// `sizeof(struct sockaddr_un.sun_path)`.
#[cfg(not(any(target_os = "linux", target_os = "android")))]
const SUN_PATH_LEN: usize = 104;

/// The text `getaddrinfo()` failures carry, without the prefix std adds.
fn gai_text(e: &io::Error) -> String {
    let s = e.to_string();
    match s.strip_prefix("failed to lookup address information: ") {
        Some(rest) => rest.to_string(),
        None => s,
    }
}

/// `inet_ai_family_from_address()`: which families the address allows.
fn families(saddr: &InetSocketAddress) -> Result<(bool, bool)> {
    let v4 = saddr.ipv4.unwrap_or(true);
    let v6 = saddr.ipv6.unwrap_or(true);
    if saddr.ipv4 == Some(false) && saddr.ipv6 == Some(false) {
        return Err(Error::generic("Cannot disable IPv4 and IPv6 at same time"));
    }
    // Asking for only one family turns the other off, as in QEMU.
    match (saddr.ipv4, saddr.ipv6) {
        (Some(true), None) => Ok((true, false)),
        (None, Some(true)) => Ok((false, true)),
        _ => Ok((v4, v6)),
    }
}

fn resolve(host: &str, port: &str, saddr: &InetSocketAddress) -> Result<Vec<SocketAddr>> {
    let (v4, v6) = families(saddr)?;
    let fail = |text: String| {
        Error::generic(format!("address resolution failed for {}:{}: {}", saddr.host, port, text))
    };
    let Ok(p) = port.parse::<u16>() else {
        return Err(fail("nodename nor servname provided, or not known".to_string()));
    };
    let addrs = (host, p).to_socket_addrs().map_err(|e| fail(gai_text(&e)))?;
    Ok(addrs.filter(|a| if a.is_ipv4() { v4 } else { v6 }).collect())
}

/// `inet_connect_saddr()`.
/// Whether the address asks for `IPPROTO_MPTCP`, which only Linux has.
fn want_mptcp(saddr: &InetSocketAddress) -> bool {
    #[cfg(target_os = "linux")]
    return saddr.mptcp == Some(true);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = saddr;
        false
    }
}

/// A stream socket with `IPPROTO_MPTCP`, which std cannot make.
#[cfg(target_os = "linux")]
fn mptcp_socket(a: &SocketAddr) -> io::Result<rustix::fd::OwnedFd> {
    use rustix::net::{AddressFamily, SocketType, ipproto, socket};
    let family = if a.is_ipv4() { AddressFamily::INET } else { AddressFamily::INET6 };
    Ok(socket(family, SocketType::STREAM, Some(ipproto::MPTCP))?)
}

fn tcp_connect(a: SocketAddr, mptcp: bool) -> io::Result<TcpStream> {
    #[cfg(target_os = "linux")]
    if mptcp {
        let fd = mptcp_socket(&a)?;
        rustix::net::connect(&fd, &a)?;
        return Ok(TcpStream::from(fd));
    }
    let _ = mptcp;
    TcpStream::connect(a)
}

fn tcp_bind(a: SocketAddr, mptcp: bool) -> io::Result<TcpListener> {
    #[cfg(target_os = "linux")]
    if mptcp {
        let fd = mptcp_socket(&a)?;
        rustix::net::sockopt::set_socket_reuseaddr(&fd, true)?;
        rustix::net::bind(&fd, &a)?;
        rustix::net::listen(&fd, 128)?;
        return Ok(TcpListener::from(fd));
    }
    let _ = mptcp;
    TcpListener::bind(a)
}

/// `inet_set_sockopts()`.
#[cfg(unix)]
fn inet_set_sockopts<F: std::os::fd::AsFd>(fd: F, saddr: &InetSocketAddress) -> Result<()> {
    use rustix::net::sockopt;
    use std::time::Duration;
    if saddr.keep_alive != Some(true) {
        return Ok(());
    }
    let fail = |msg: &str, e: rustix::io::Errno| Error::from_io(msg, io::Error::from(e));
    sockopt::set_socket_keepalive(&fd, true)
        .map_err(|e| fail("Unable to set keep-alive option on socket", e))?;
    if let Some(n) = saddr.keep_alive_count.filter(|&n| n != 0) {
        sockopt::set_tcp_keepcnt(&fd, n)
            .map_err(|e| fail("Unable to set TCP keep-alive count option on socket", e))?;
    }
    if let Some(n) = saddr.keep_alive_idle.filter(|&n| n != 0) {
        sockopt::set_tcp_keepidle(&fd, Duration::from_secs(n.into()))
            .map_err(|e| fail("Unable to set TCP keep-alive idle option on socket", e))?;
    }
    if let Some(n) = saddr.keep_alive_interval.filter(|&n| n != 0) {
        sockopt::set_tcp_keepintvl(&fd, Duration::from_secs(n.into()))
            .map_err(|e| fail("Unable to set TCP keep-alive interval option on socket", e))?;
    }
    Ok(())
}

/// Windows has no keep-alive tuning in std; the option is left alone there.
#[cfg(not(unix))]
fn inet_set_sockopts<F>(_fd: F, _saddr: &InetSocketAddress) -> Result<()> {
    Ok(())
}

fn inet_connect(saddr: &InetSocketAddress) -> Result<TcpStream> {
    let addrs = resolve(&saddr.host, &saddr.port, saddr)?;
    let mut last = None;
    for a in addrs {
        match tcp_connect(a, want_mptcp(saddr)) {
            Ok(s) => {
                inet_set_sockopts(&s, saddr)?;
                return Ok(s);
            }
            Err(e) => last = Some(e),
        }
    }
    let e = last.unwrap_or_else(|| io::Error::from_raw_os_error(libc::EADDRNOTAVAIL));
    Err(Error::from_io(format!("Failed to connect to '{}:{}'", saddr.host, saddr.port), e))
}

/// `saddr_is_abstract()`.
#[cfg(unix)]
fn is_abstract(saddr: &UnixSocketAddress) -> bool {
    #[cfg(target_os = "linux")]
    return saddr.abstract_ == Some(true);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = saddr;
        false
    }
}

fn check_unix_path(path: &str, abstract_: bool) -> Result<()> {
    let max = if abstract_ { SUN_PATH_LEN - 1 } else { SUN_PATH_LEN };
    if path.len() > max {
        return Err(Error::generic(format!("UNIX socket path '{path}' is too long"))
            .hint(format!("Path must be less than {max} bytes\n")));
    }
    Ok(())
}

/// The address of a Linux abstract socket. A name that is not `tight` fills the whole of
/// `sun_path`, padded with zero bytes, like QEMU's `addrlen = sizeof(un)`.
#[cfg(target_os = "linux")]
fn abstract_addr(saddr: &UnixSocketAddress) -> io::Result<std::os::unix::net::SocketAddr> {
    use std::os::linux::net::SocketAddrExt;
    let mut name = saddr.path.as_bytes().to_vec();
    if saddr.tight == Some(false) {
        name.resize(SUN_PATH_LEN - 1, 0);
    }
    std::os::unix::net::SocketAddr::from_abstract_name(&name)
}

#[cfg(unix)]
fn unix_connect(saddr: &UnixSocketAddress) -> Result<NbdStream> {
    check_unix_path(&saddr.path, is_abstract(saddr))?;
    #[cfg(target_os = "linux")]
    let r = if is_abstract(saddr) {
        abstract_addr(saddr).and_then(|a| UnixStream::connect_addr(&a))
    } else {
        UnixStream::connect(&saddr.path)
    };
    #[cfg(not(target_os = "linux"))]
    let r = UnixStream::connect(&saddr.path);
    r.map(NbdStream::Unix)
        .map_err(|e| Error::from_io(format!("Failed to connect to '{}'", saddr.path), e))
}

#[cfg(not(unix))]
fn unix_connect(saddr: &UnixSocketAddress) -> Result<NbdStream> {
    check_unix_path(&saddr.path, false)?;
    Err(Error::generic("socket family 1 unsupported"))
}

/// `socket_connect()`, then `qio_channel_set_delay(false)` as the NBD client does.
pub(crate) fn socket_connect(addr: &SocketAddress) -> Result<NbdStream> {
    match &addr.u {
        SocketAddressU::Inet(i) => {
            let s = inet_connect(i)?;
            let _ = s.set_nodelay(true);
            Ok(NbdStream::Tcp(s))
        }
        SocketAddressU::Unix(u) => unix_connect(u),
        SocketAddressU::Vsock(_) => Err(Error::generic("socket family AF_VSOCK unsupported")),
        SocketAddressU::Fd(f) => Err(fd_unsupported(&f.str)),
    }
}

/// Descriptors passed by number are not supported: taking ownership of one needs unsafe code.
fn fd_unsupported(fd: &str) -> Error {
    if fd.parse::<i32>().is_err() {
        return Error::from_io(
            format!("Unable to parse FD number {fd}"),
            io::Error::from_raw_os_error(libc::EINVAL),
        );
    }
    Error::generic(format!("File descriptor '{fd}' is not a socket"))
}

/// A listening socket.
#[derive(Debug)]
pub(crate) enum Listener {
    Tcp(TcpListener),
    /// The listener and the file to remove when it goes away (none for abstract sockets).
    #[cfg(unix)]
    Unix(UnixListener, Option<String>),
}

impl Listener {
    /// A non-blocking accept. `Ok(None)` when nobody is waiting.
    pub(crate) fn accept(&self) -> io::Result<Option<NbdStream>> {
        let r = match self {
            Listener::Tcp(l) => l.accept().map(|(s, _)| {
                let _ = s.set_nodelay(true);
                NbdStream::Tcp(s)
            }),
            #[cfg(unix)]
            Listener::Unix(l, _) => l.accept().map(|(s, _)| NbdStream::Unix(s)),
        };
        match r {
            Ok(s) => {
                // Accepted sockets inherit O_NONBLOCK on the BSDs.
                s.set_nonblocking(false)?;
                Ok(Some(s))
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e),
        }
    }
}

#[cfg(unix)]
impl Drop for Listener {
    fn drop(&mut self) {
        // QIONetListener unlinks nothing, but socket_listen() cleans up stale sockets before
        // binding, so a leftover file is harmless. Remove ours anyway to keep /tmp tidy.
        if let Listener::Unix(_, Some(path)) = self {
            let _ = std::fs::remove_file(path.as_str());
        }
    }
}

/// `inet_listen_saddr()` with no port offset.
fn inet_listen(saddr: &InetSocketAddress) -> Result<(TcpListener, SocketAddress)> {
    let host = if saddr.host.is_empty() { "0.0.0.0" } else { saddr.host.as_str() };
    let addrs = resolve(host, &saddr.port, saddr)?;
    let mut last: Option<io::Error> = None;
    for a in addrs {
        let port_min = a.port();
        let port_max = saddr.to.unwrap_or(port_min).max(port_min);
        for p in port_min..=port_max {
            let mut a = a;
            a.set_port(p);
            match tcp_bind(a, want_mptcp(saddr)) {
                Ok(l) => {
                    inet_set_sockopts(&l, saddr)?;
                    let local =
                        l.local_addr().map_err(|e| Error::from_io("Failed to bind socket", e))?;
                    let mut out = saddr.clone();
                    out.port = local.port().to_string();
                    return Ok((l, SocketAddress { u: SocketAddressU::Inet(out) }));
                }
                Err(e) if e.kind() == io::ErrorKind::AddrInUse => last = Some(e),
                Err(e) => return Err(Error::from_io("Failed to bind socket", e)),
            }
        }
    }
    match last {
        Some(e) => Err(Error::from_io("Failed to find an available port", e)),
        None => Err(Error::from_io(
            "Failed to create socket",
            io::Error::from_raw_os_error(libc::EAFNOSUPPORT),
        )),
    }
}

/// `unix_listen_saddr()`. An empty path, and no abstract name, listens on a new socket in the
/// temporary directory, whose path is returned with the listener.
#[cfg(unix)]
fn unix_listen(saddr: &UnixSocketAddress) -> Result<(Listener, UnixSocketAddress)> {
    let abstract_ = is_abstract(saddr);
    let mut bound = saddr.clone();
    if saddr.path.is_empty() && !abstract_ {
        bound.path = temp_socket_path();
    }
    let path = &bound.path;
    check_unix_path(path, abstract_)?;
    if !abstract_ {
        if let Err(e) = std::fs::remove_file(path) {
            if e.kind() != io::ErrorKind::NotFound {
                return Err(Error::from_io(format!("Failed to unlink socket {path}"), e));
            }
        }
    }
    #[cfg(target_os = "linux")]
    let r = if abstract_ {
        abstract_addr(saddr).and_then(|a| UnixListener::bind_addr(&a))
    } else {
        UnixListener::bind(path)
    };
    #[cfg(not(target_os = "linux"))]
    let r = UnixListener::bind(path);
    let l = r.map_err(|e| Error::from_io(format!("Failed to bind socket to {path}"), e))?;
    let owned = if abstract_ { None } else { Some(path.clone()) };
    Ok((Listener::Unix(l, owned), bound))
}

/// `qemu-socket-XXXXXX` in the temporary directory. QEMU gets the name from mkstemp(); this
/// takes one that is not in use.
#[cfg(unix)]
fn temp_socket_path() -> String {
    use std::sync::atomic::{AtomicU32, Ordering};
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let dir = std::env::temp_dir();
    loop {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let name = format!("qemu-socket-{:06x}", ((std::process::id() << 8) ^ n) & 0xff_ffff);
        let p = dir.join(name);
        if !p.exists() {
            return p.to_string_lossy().into_owned();
        }
    }
}

#[cfg(not(unix))]
fn unix_listen(saddr: &UnixSocketAddress) -> Result<(Listener, UnixSocketAddress)> {
    check_unix_path(&saddr.path, false)?;
    Err(Error::generic("socket family 1 unsupported"))
}

/// `socket_listen()`. Returns the listener in non-blocking mode and the address it is bound
/// to, with the port filled in when the caller asked for port 0.
pub(crate) fn socket_listen(addr: &SocketAddress) -> Result<(Listener, SocketAddress)> {
    let (l, bound) = match &addr.u {
        SocketAddressU::Inet(i) => {
            let (l, bound) = inet_listen(i)?;
            (Listener::Tcp(l), bound)
        }
        SocketAddressU::Unix(u) => {
            let (l, bound) = unix_listen(u)?;
            (l, SocketAddress { u: SocketAddressU::Unix(bound) })
        }
        SocketAddressU::Vsock(_) => {
            return Err(Error::generic("socket family AF_VSOCK unsupported"));
        }
        SocketAddressU::Fd(f) => return Err(fd_unsupported(&f.str)),
    };
    let nb = match &l {
        Listener::Tcp(t) => t.set_nonblocking(true),
        #[cfg(unix)]
        Listener::Unix(u, _) => u.set_nonblocking(true),
    };
    nb.map_err(|e| Error::from_io("Failed to listen on socket", e))?;
    Ok((l, bound))
}
