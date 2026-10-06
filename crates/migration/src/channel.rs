// SPDX-License-Identifier: GPL-2.0-or-later

//! Migration channels, migration/channel.c, socket.c, fd.c, exec.c and file.c.
//!
//! A migration goes over one channel: a `tcp`, `unix` or `vsock` socket, a file descriptor the
//! monitor holds (`fd:`), the standard input or output of a command (`exec:`) or a plain file
//! (`file:`). The address comes either as a URI or as the `channels` argument of `migrate` and
//! `migrate-incoming`; both turn into a QAPI `MigrationAddress`.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use ruvm_base::{Error, Result, bail};
use ruvm_qapi::types::{
    FdSocketAddress, FileMigrationArgs, InetSocketAddress, MigrationAddress, MigrationAddressU,
    MigrationChannel, MigrationChannelType, MigrationExecCommand, SocketAddress, SocketAddressU,
    UnixSocketAddress,
};

/// A migration address, QAPI `MigrationAddress`.
pub type MigrationAddr = MigrationAddress;

/// Looks up a file descriptor the monitor holds by name, for `fd:` channels. `monitor_get_fd()`
/// in QEMU also takes a number for a descriptor the process inherited; whether that works is up
/// to the resolver.
#[cfg(unix)]
pub type FdResolver = dyn Fn(&str) -> Result<std::os::fd::OwnedFd> + Send + Sync;

/// Looks up a file descriptor the monitor holds. Windows has no `fd:` channels.
#[cfg(not(unix))]
pub type FdResolver = dyn Fn(&str) -> Result<File> + Send + Sync;

/// `qemu_strtosz()` for the `offset=` option: a number with an optional size suffix.
fn parse_size(s: &str) -> Option<u64> {
    let (num, mult) = match s.char_indices().last() {
        Some((i, c)) if c.is_ascii_alphabetic() && !s.starts_with("0x") => {
            let m: u64 = match c.to_ascii_uppercase() {
                'B' => 1,
                'K' => 1 << 10,
                'M' => 1 << 20,
                'G' => 1 << 30,
                'T' => 1 << 40,
                'P' => 1 << 50,
                'E' => 1 << 60,
                _ => return None,
            };
            (&s[..i], m)
        }
        _ => (s, 1),
    };
    let v = match num.strip_prefix("0x").or_else(|| num.strip_prefix("0X")) {
        Some(h) => u64::from_str_radix(h, 16).ok()?,
        None => num.parse::<u64>().ok()?,
    };
    v.checked_mul(mult)
}

/// `inet_parse()` for the address part of a `tcp:` URI, ignoring the options QEMU accepts after
/// it.
fn inet_parse(s: &str) -> Result<InetSocketAddress> {
    let (addr, opts) = match s.find(',') {
        Some(i) => (&s[..i], &s[i + 1..]),
        None => (s, ""),
    };
    let mut inet = InetSocketAddress::default();
    if let Some(rest) = addr.strip_prefix('[') {
        match rest.find("]:") {
            Some(e) if e >= 1 && rest.len() - e >= 3 => {
                inet.host = rest[..e].to_string();
                inet.port = rest[e + 2..].to_string();
            }
            _ => bail!("error parsing IPv6 address '{}'", addr),
        }
    } else {
        match addr.rfind(':').filter(|i| addr.len() - i >= 2 && !addr[..*i].contains(':')) {
            Some(i) => {
                inet.host = addr[..i].to_string();
                inet.port = addr[i + 1..].to_string();
            }
            None => bail!("error parsing address '{}'", addr),
        }
    }
    for opt in opts.split(',').filter(|o| !o.is_empty()) {
        let (k, v) = opt.split_once('=').unwrap_or((opt, "on"));
        let on = || match v {
            "on" | "yes" | "true" | "y" => Ok(true),
            "off" | "no" | "false" | "n" => Ok(false),
            _ => Err(Error::generic(format!("Parameter '{k}' expects 'on' or 'off'"))),
        };
        match k {
            "numeric" => inet.numeric = Some(on()?),
            "ipv4" => inet.ipv4 = Some(on()?),
            "ipv6" => inet.ipv6 = Some(on()?),
            "keep-alive" => inet.keep_alive = Some(on()?),
            "to" => {
                inet.to =
                    Some(v.parse().map_err(|_| {
                        Error::generic("Parameter 'to' expects a number".to_string())
                    })?)
            }
            _ => bail!("Invalid parameter '{}'", k),
        }
    }
    Ok(inet)
}

/// `migrate_uri_parse()`.
pub fn parse_uri(uri: &str) -> Result<MigrationAddr> {
    let u = if let Some(cmd) = uri.strip_prefix("exec:") {
        let args = if cfg!(windows) {
            vec!["cmd.exe".to_string(), "/c".to_string(), cmd.to_string()]
        } else {
            vec!["/bin/sh".to_string(), "-c".to_string(), cmd.to_string()]
        };
        MigrationAddressU::Exec(MigrationExecCommand { args })
    } else if let Some(rest) = uri.strip_prefix("rdma:") {
        MigrationAddressU::Rdma(inet_parse(rest)?)
    } else if let Some(rest) = uri.strip_prefix("tcp:") {
        MigrationAddressU::Socket(SocketAddress { u: SocketAddressU::Inet(inet_parse(rest)?) })
    } else if let Some(path) = uri.strip_prefix("unix:") {
        MigrationAddressU::Socket(SocketAddress {
            u: SocketAddressU::Unix(UnixSocketAddress {
                path: path.to_string(),
                ..Default::default()
            }),
        })
    } else if uri.starts_with("vsock:") {
        bail!("vsock migration channels are not supported by ruvm yet");
    } else if let Some(name) = uri.strip_prefix("fd:") {
        MigrationAddressU::Socket(SocketAddress {
            u: SocketAddressU::Fd(FdSocketAddress { str: name.to_string() }),
        })
    } else if let Some(spec) = uri.strip_prefix("file:") {
        let (filename, offset) = match spec.find(",offset=") {
            Some(i) => {
                let opt = &spec[i + ",offset=".len()..];
                let off = parse_size(opt).ok_or_else(|| {
                    Error::generic(format!("file URI has bad offset {opt}: Invalid argument"))
                })?;
                (&spec[..i], off)
            }
            None => (spec, 0),
        };
        MigrationAddressU::File(FileMigrationArgs { filename: filename.to_string(), offset })
    } else {
        bail!("unknown migration protocol: {}", uri);
    };
    Ok(MigrationAddress { u })
}

/// `migration_channel_parse_input()` without a `cpr` channel: the main address from either a
/// URI or a channel list.
pub fn parse_input(
    uri: Option<&str>,
    channels: Option<&[MigrationChannel]>,
) -> Result<MigrationAddr> {
    match (uri, channels) {
        (Some(uri), None) => parse_uri(uri),
        (None, Some(channels)) => {
            if channels.len() > 1 {
                bail!("Channel list must have only one entry, for type 'main'");
            }
            match channels.first() {
                Some(c) if c.channel_type == MigrationChannelType::Main => Ok(c.addr.clone()),
                _ => bail!("Channel list has no main entry"),
            }
        }
        _ => bail!("need either 'uri' or 'channels' argument"),
    }
}

/// The URI form of an address, for messages.
pub fn addr_to_string(addr: &MigrationAddr) -> String {
    match &addr.u {
        MigrationAddressU::Socket(s) => match &s.u {
            SocketAddressU::Inet(i) if i.host.contains(':') => {
                format!("tcp:[{}]:{}", i.host, i.port)
            }
            SocketAddressU::Inet(i) => format!("tcp:{}:{}", i.host, i.port),
            SocketAddressU::Unix(u) => format!("unix:{}", u.path),
            SocketAddressU::Vsock(v) => format!("vsock:{}:{}", v.cid, v.port),
            SocketAddressU::Fd(f) => format!("fd:{}", f.str),
        },
        MigrationAddressU::Exec(e) => format!("exec:{}", e.args.join(" ")),
        MigrationAddressU::Rdma(i) => format!("rdma:{}:{}", i.host, i.port),
        MigrationAddressU::File(f) => format!("file:{},offset={}", f.filename, f.offset),
    }
}

fn host_port(inet: &InetSocketAddress) -> String {
    let host = match inet.host.as_str() {
        "" => "0.0.0.0",
        h => h,
    };
    if host.contains(':') {
        format!("[{}]:{}", host, inet.port)
    } else {
        format!("{}:{}", host, inet.port)
    }
}

fn resolve(inet: &InetSocketAddress) -> Result<Vec<std::net::SocketAddr>> {
    let s = host_port(inet);
    let addrs: Vec<_> = s
        .to_socket_addrs()
        .map_err(|e| Error::from_io(format!("address resolution failed for {s}"), e))?
        .filter(|a| {
            !(inet.ipv4 == Some(false) && a.is_ipv4() || inet.ipv6 == Some(false) && a.is_ipv6())
        })
        .collect();
    if addrs.is_empty() {
        bail!("address resolution failed for {}", s);
    }
    Ok(addrs)
}

/// The writing end of an `exec:` channel: the command's standard input. Closing it waits for the
/// command.
struct ExecWriter {
    stdin: Option<ChildStdin>,
    child: Child,
}

impl Write for ExecWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.stdin.as_mut().map_or(Err(io::ErrorKind::BrokenPipe.into()), |s| s.write(buf))
    }
    fn flush(&mut self) -> io::Result<()> {
        self.stdin.as_mut().map_or(Ok(()), |s| s.flush())
    }
}

impl Drop for ExecWriter {
    fn drop(&mut self) {
        drop(self.stdin.take());
        let _ = self.child.wait();
    }
}

struct ExecReader {
    stdout: ChildStdout,
    child: Child,
}

impl Read for ExecReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.stdout.read(buf)
    }
}

impl Drop for ExecReader {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn exec_command(args: &[String]) -> Result<Command> {
    let Some((prog, rest)) = args.split_first() else {
        bail!("exec migration needs a command");
    };
    let mut c = Command::new(prog);
    c.args(rest);
    Ok(c)
}

/// The sending side of a channel.
#[derive(Debug)]
pub struct Channel;

/// What `migrate` writes the stream into.
pub type Outgoing = Box<dyn Write + Send>;

/// What `migrate-incoming` reads the stream from.
pub type Incoming = Box<dyn Read + Send>;

impl Channel {
    /// `migration_connect_outgoing()`: connects, or opens, the channel at `addr`.
    pub fn connect(addr: &MigrationAddr, fds: Option<&FdResolver>) -> Result<Outgoing> {
        match &addr.u {
            MigrationAddressU::Socket(s) => match &s.u {
                SocketAddressU::Inet(inet) => {
                    let mut last = None;
                    for a in resolve(inet)? {
                        match TcpStream::connect(a) {
                            Ok(s) => {
                                let _ = s.set_nodelay(true);
                                return Ok(Box::new(io::BufWriter::with_capacity(1 << 16, s)));
                            }
                            Err(e) => last = Some(e),
                        }
                    }
                    let e = last.unwrap_or_else(|| io::ErrorKind::NotFound.into());
                    Err(Error::from_io(format!("Failed to connect to '{}'", host_port(inet)), e))
                }
                #[cfg(unix)]
                SocketAddressU::Unix(u) => {
                    let s = std::os::unix::net::UnixStream::connect(&u.path).map_err(|e| {
                        Error::from_io(format!("Failed to connect to '{}'", u.path), e)
                    })?;
                    Ok(Box::new(io::BufWriter::with_capacity(1 << 16, s)))
                }
                SocketAddressU::Fd(f) => {
                    let Some(fds) = fds else { bail!("No file descriptor named {} found", f.str) };
                    let file = File::from(fds(&f.str)?);
                    if file.metadata().is_ok_and(|m| m.is_file()) {
                        bail!("fd: migration to a file is not supported. Use file: instead.");
                    }
                    Ok(Box::new(file))
                }
                _ => bail!("uri is not a valid migration protocol"),
            },
            MigrationAddressU::Exec(e) => {
                let mut child = exec_command(&e.args)?
                    .stdin(Stdio::piped())
                    .spawn()
                    .map_err(|err| Error::from_io("Failed to start the migration command", err))?;
                let stdin = child.stdin.take();
                Ok(Box::new(ExecWriter { stdin, child }))
            }
            MigrationAddressU::File(f) => {
                let mut file = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(false)
                    .open(&f.filename)
                    .map_err(|e| Error::from_io(format!("Could not open '{}'", f.filename), e))?;
                file.set_len(f.offset).map_err(|e| {
                    Error::from_io(
                        format!("failed to truncate migration file to offset {:x}", f.offset),
                        e,
                    )
                })?;
                file.seek(SeekFrom::Start(f.offset))
                    .map_err(|e| Error::from_io("Unable to seek the migration file", e))?;
                Ok(Box::new(io::BufWriter::with_capacity(1 << 16, file)))
            }
            MigrationAddressU::Rdma(_) => bail!("RDMA migration is not supported by ruvm"),
        }
    }

    /// `migration_connect_incoming()`: starts listening at `addr`, or opens it. The stream comes
    /// from [`Listener::accept`].
    pub fn listen(addr: &MigrationAddr, fds: Option<&FdResolver>) -> Result<Listener> {
        let kind = match &addr.u {
            MigrationAddressU::Socket(s) => match &s.u {
                SocketAddressU::Inet(inet) => {
                    let addrs = resolve(inet)?;
                    let l = TcpListener::bind(&addrs[..]).map_err(|e| {
                        Error::from_io(
                            format!("Failed to bind socket for '{}'", host_port(inet)),
                            e,
                        )
                    })?;
                    ListenerKind::Tcp(l)
                }
                #[cfg(unix)]
                SocketAddressU::Unix(u) => {
                    let _ = std::fs::remove_file(&u.path);
                    let l = std::os::unix::net::UnixListener::bind(&u.path).map_err(|e| {
                        Error::from_io(format!("Failed to bind socket to {}", u.path), e)
                    })?;
                    ListenerKind::Unix(l, u.path.clone())
                }
                SocketAddressU::Fd(f) => {
                    let Some(fds) = fds else { bail!("No file descriptor named {} found", f.str) };
                    let file = File::from(fds(&f.str)?);
                    ListenerKind::Ready(Box::new(file))
                }
                _ => bail!("unknown migration protocol"),
            },
            MigrationAddressU::Exec(e) => {
                let mut child = exec_command(&e.args)?
                    .stdout(Stdio::piped())
                    .spawn()
                    .map_err(|err| Error::from_io("Failed to start the migration command", err))?;
                let Some(stdout) = child.stdout.take() else {
                    bail!("The migration command has no standard output");
                };
                ListenerKind::Ready(Box::new(ExecReader { stdout, child }))
            }
            MigrationAddressU::File(f) => {
                let mut file = File::open(&f.filename)
                    .map_err(|e| Error::from_io(format!("Could not open '{}'", f.filename), e))?;
                if f.offset != 0 {
                    file.seek(SeekFrom::Start(f.offset))
                        .map_err(|e| Error::from_io("Unable to seek the migration file", e))?;
                }
                ListenerKind::Ready(Box::new(io::BufReader::with_capacity(1 << 16, file)))
            }
            MigrationAddressU::Rdma(_) => bail!("RDMA migration is not supported by ruvm"),
        };
        Ok(Listener { kind })
    }
}

enum ListenerKind {
    Tcp(TcpListener),
    #[cfg(unix)]
    Unix(std::os::unix::net::UnixListener, String),
    Ready(Incoming),
}

/// An incoming channel waiting for its source.
pub struct Listener {
    kind: ListenerKind,
}

impl std::fmt::Debug for Listener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Listener").field("address", &self.local_addr()).finish()
    }
}

impl Listener {
    /// The address a socket listens on, with the port the system picked for port 0, for the
    /// `socket-address` of `query-migrate`.
    pub fn local_addr(&self) -> Option<SocketAddress> {
        match &self.kind {
            ListenerKind::Tcp(l) => l.local_addr().ok().map(|a| SocketAddress {
                u: SocketAddressU::Inet(InetSocketAddress {
                    host: a.ip().to_string(),
                    port: a.port().to_string(),
                    ..Default::default()
                }),
            }),
            #[cfg(unix)]
            ListenerKind::Unix(_, path) => Some(SocketAddress {
                u: SocketAddressU::Unix(UnixSocketAddress {
                    path: path.clone(),
                    ..Default::default()
                }),
            }),
            ListenerKind::Ready(_) => None,
        }
    }

    /// Waits for the source to connect, and returns the stream.
    pub fn accept(self) -> Result<Incoming> {
        match self.kind {
            ListenerKind::Tcp(l) => {
                let (s, _) = l.accept().map_err(|e| Error::from_io("Failed to accept", e))?;
                Ok(Box::new(s))
            }
            #[cfg(unix)]
            ListenerKind::Unix(l, path) => {
                let r = l.accept().map_err(|e| Error::from_io("Failed to accept", e));
                let _ = std::fs::remove_file(path);
                Ok(Box::new(r?.0))
            }
            ListenerKind::Ready(r) => Ok(r),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uris() {
        let a = parse_uri("tcp:0:4444").unwrap();
        assert_eq!(addr_to_string(&a), "tcp:0:4444");
        let a = parse_uri("tcp:[::1]:5555").unwrap();
        assert_eq!(addr_to_string(&a), "tcp:[::1]:5555");
        let a = parse_uri("tcp::4444").unwrap();
        assert_eq!(addr_to_string(&a), "tcp::4444");
        let a = parse_uri("file:/tmp/x,offset=0x1000").unwrap();
        assert_eq!(addr_to_string(&a), "file:/tmp/x,offset=4096");
        let a = parse_uri("file:/tmp/x,offset=1M").unwrap();
        assert!(matches!(a.u, MigrationAddressU::File(FileMigrationArgs { offset: 1048576, .. })));
        let a = parse_uri("exec:cat > /dev/null").unwrap();
        assert!(matches!(&a.u, MigrationAddressU::Exec(e) if e.args.len() == 3));
        assert_eq!(
            parse_uri("foo:bar").unwrap_err().message(),
            "unknown migration protocol: foo:bar"
        );
        assert_eq!(parse_uri("tcp:4444").unwrap_err().message(), "error parsing address '4444'");
        assert_eq!(
            parse_input(None, None).unwrap_err().message(),
            "need either 'uri' or 'channels' argument"
        );
    }

    #[test]
    fn tcp_round_trip() {
        let l = Channel::listen(&parse_uri("tcp:127.0.0.1:0").unwrap(), None).unwrap();
        let Some(SocketAddress { u: SocketAddressU::Inet(inet) }) = l.local_addr() else {
            panic!("no address")
        };
        let t = std::thread::spawn(move || {
            let mut r = l.accept().unwrap();
            let mut v = Vec::new();
            r.read_to_end(&mut v).unwrap();
            v
        });
        let mut w =
            Channel::connect(&parse_uri(&format!("tcp:127.0.0.1:{}", inet.port)).unwrap(), None)
                .unwrap();
        w.write_all(b"QEVM").unwrap();
        w.flush().unwrap();
        drop(w);
        assert_eq!(t.join().unwrap(), b"QEVM");
    }

    #[test]
    fn file_round_trip() {
        let path = std::env::temp_dir().join(format!("ruvm-mig-{}", std::process::id()));
        let uri = format!("file:{},offset=16", path.display());
        let mut w = Channel::connect(&parse_uri(&uri).unwrap(), None).unwrap();
        w.write_all(b"stream").unwrap();
        w.flush().unwrap();
        drop(w);
        let mut r = Channel::listen(&parse_uri(&uri).unwrap(), None).unwrap().accept().unwrap();
        let mut v = Vec::new();
        r.read_to_end(&mut v).unwrap();
        assert_eq!(v, b"stream");
        let _ = std::fs::remove_file(path);
    }
}
