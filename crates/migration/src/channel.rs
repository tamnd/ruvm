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

/// Opens `/dev/fdset/N` with the given flags, `monitor_fdset_dup_fd_add()`: a duplicate of a
/// member of fd set N whose access mode matches. The fd sets are the monitor's.
#[cfg(unix)]
pub type FdsetOpener =
    dyn Fn(i64, rustix::fs::OFlags) -> Result<std::os::fd::OwnedFd> + Send + Sync;

/// The fd sets `/dev/fdset/N` paths open from. QEMU has one `mon_fdsets` for the process, so
/// this is not part of a migration.
#[cfg(unix)]
static FDSET_OPENER: std::sync::RwLock<Option<std::sync::Arc<FdsetOpener>>> =
    std::sync::RwLock::new(None);

/// Sets where `/dev/fdset/N` paths of `file:` channels come from.
#[cfg(unix)]
pub fn set_fdset_opener(open: std::sync::Arc<FdsetOpener>) {
    *FDSET_OPENER.write().unwrap_or_else(|e| e.into_inner()) = Some(open);
}

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

/// `migration_channel_parse_input()` for `migrate`, which also takes a `cpr` channel: the main
/// address and the `cpr` one, if there is one. `cpr_transfer` is whether the migration mode is
/// `cpr-transfer`, which needs the `cpr` channel.
pub fn parse_input_cpr(
    uri: Option<&str>,
    channels: Option<&[MigrationChannel]>,
    cpr_transfer: bool,
) -> Result<(MigrationAddr, Option<MigrationAddr>)> {
    let channels = match (uri, channels) {
        // QEMU asserts on a URI in cpr-transfer mode, which has no way to name the cpr channel.
        (Some(_), None) if cpr_transfer => bail!("missing 'cpr' migration channel"),
        (Some(uri), None) => return Ok((parse_uri(uri)?, None)),
        (None, Some(channels)) => channels,
        _ => bail!("need either 'uri' or 'channels' argument"),
    };
    let (mut main, mut cpr) = (None, None);
    for c in channels {
        let (slot, name) = match c.channel_type {
            MigrationChannelType::Main => (&mut main, "main"),
            MigrationChannelType::Cpr => (&mut cpr, "cpr"),
        };
        if slot.is_some() {
            bail!("Channel list has more than one {} entry", name);
        }
        *slot = Some(c.addr.clone());
    }
    if cpr_transfer && cpr.is_none() {
        bail!("missing 'cpr' migration channel");
    }
    match main {
        Some(main) => Ok((main, cpr)),
        None => bail!("Channel list has no main entry"),
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

/// A socket channel, which can carry the return path back from the destination
/// (`qemu_file_get_return_path()`).
#[derive(Debug)]
pub enum Socket {
    /// A TCP connection.
    Tcp(TcpStream),
    /// A Unix socket, or a socket the monitor passed as `fd:`.
    #[cfg(unix)]
    Unix(std::os::unix::net::UnixStream),
}

impl Socket {
    /// Another handle on the same connection, for the other direction.
    pub fn try_clone(&self) -> io::Result<Socket> {
        Ok(match self {
            Socket::Tcp(s) => Socket::Tcp(s.try_clone()?),
            #[cfg(unix)]
            Socket::Unix(s) => Socket::Unix(s.try_clone()?),
        })
    }

    /// `qemu_file_shutdown()`: shuts both directions down, which wakes a thread blocked on it.
    pub fn shutdown(&self) {
        let _ = match self {
            Socket::Tcp(s) => s.shutdown(std::net::Shutdown::Both),
            #[cfg(unix)]
            Socket::Unix(s) => s.shutdown(std::net::Shutdown::Both),
        };
    }
}

impl Read for Socket {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Socket::Tcp(s) => s.read(buf),
            #[cfg(unix)]
            Socket::Unix(s) => s.read(buf),
        }
    }
}

impl Write for Socket {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Socket::Tcp(s) => s.write(buf),
            #[cfg(unix)]
            Socket::Unix(s) => s.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A file descriptor from the monitor as a socket, if it is one.
#[cfg(unix)]
fn fd_socket(fd: std::os::fd::OwnedFd) -> std::result::Result<Socket, File> {
    use std::os::unix::fs::FileTypeExt;
    let file = File::from(fd);
    if file.metadata().is_ok_and(|m| m.file_type().is_socket()) {
        Ok(Socket::Unix(std::os::unix::net::UnixStream::from(std::os::fd::OwnedFd::from(file))))
    } else {
        Err(file)
    }
}

/// `qio_channel_file_new_path()` for one more handle on a `file:` channel, a multifd channel
/// with mapped-ram. `direct` adds `O_DIRECT`.
pub fn open_file(path: &str, write: bool, direct: bool) -> Result<File> {
    qemu_open(path, write, false, direct)
}

/// `qemu_open()`: opens `path` read-only or write-only, creating it with mode 0600 if `create`
/// is set. A `/dev/fdset/N` path duplicates a descriptor of fd set N instead.
fn qemu_open(path: &str, write: bool, create: bool, direct: bool) -> Result<File> {
    #[cfg(unix)]
    let o_direct = if direct { ruvm_sys::directio::o_direct() } else { None };
    #[cfg(not(unix))]
    let _ = direct;
    #[cfg(unix)]
    if let Some(id) = path.strip_prefix("/dev/fdset/") {
        use rustix::fs::OFlags;
        // qemu_parse_fdset()
        let Ok(id) = id.parse::<i32>() else { bail!("Could not parse fdset {path}") };
        let open = FDSET_OPENER.read().unwrap_or_else(|e| e.into_inner()).clone();
        let Some(open) = open else { bail!("Failed to find fdset {path}") };
        let mut flags = if write { OFlags::WRONLY } else { OFlags::RDONLY };
        if create {
            flags |= OFlags::CREATE;
        }
        if let Some(flag) = o_direct {
            flags |= OFlags::from_bits_retain(flag as _);
        }
        return open(i64::from(id), flags).map(File::from);
    }
    let mut o = OpenOptions::new();
    o.read(!write).write(write).create(create).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        if create {
            o.mode(0o600);
        }
        if let Some(flag) = o_direct {
            o.custom_flags(flag);
        }
    }
    o.open(path).map_err(|e| Error::from_io(format!("Could not open '{path}'"), e))
}

/// The seekable side of a `file:` channel: the file offset of the stream, and reads and writes
/// at a given offset that leave the stream where it is (`qemu_get_offset()`,
/// `qemu_put_buffer_at()` and their kin).
#[derive(Debug)]
pub struct FileChannel {
    path: String,
    // Shares its position with the handle the stream goes through.
    seek: File,
    // For the positioned reads and writes. Windows moves the file position on those, so there
    // it is a handle of its own.
    pio: File,
}

impl FileChannel {
    fn new(path: &str, stream: &File, write: bool) -> Result<Self> {
        let seek = stream.try_clone().map_err(|e| Error::from_io("Could not dup the file", e))?;
        #[cfg(unix)]
        let pio = stream.try_clone().map_err(|e| Error::from_io("Could not dup the file", e))?;
        #[cfg(not(unix))]
        let pio = open_file(path, write, false)?;
        let _ = write;
        Ok(FileChannel { path: path.to_string(), seek, pio })
    }

    /// The file name, for the handles of the multifd channels.
    pub fn path(&self) -> &str {
        &self.path
    }

    /// `qio_channel_io_seek(SEEK_CUR)`: where the stream is in the file, with everything
    /// written so far flushed.
    pub fn offset(&self) -> Result<u64> {
        (&self.seek)
            .stream_position()
            .map_err(|e| Error::from_io("Unable to seek to offset 0 whence 1 in file", e))
    }

    /// `qio_channel_io_seek(SEEK_SET)`.
    pub fn set_offset(&self, off: u64) -> Result<()> {
        (&self.seek).seek(SeekFrom::Start(off)).map(|_| ()).map_err(|e| {
            Error::from_io(format!("Unable to seek to offset {off} whence 0 in file"), e)
        })
    }

    /// `qio_channel_pwrite_all()`.
    pub fn write_at(&self, buf: &[u8], off: u64) -> Result<()> {
        write_all_at(&self.pio, buf, off)
    }

    /// `qio_channel_pread_all()`.
    pub fn read_at(&self, buf: &mut [u8], off: u64) -> Result<()> {
        read_exact_at(&self.pio, buf, off)
    }
}

/// `qio_channel_pwrite_all()` on any file handle.
pub fn write_all_at(file: &File, buf: &[u8], off: u64) -> Result<()> {
    #[cfg(unix)]
    let ret = std::os::unix::fs::FileExt::write_all_at(file, buf, off);
    #[cfg(windows)]
    let ret = {
        let (mut done, mut ret) = (0, Ok(()));
        while done < buf.len() {
            match std::os::windows::fs::FileExt::seek_write(file, &buf[done..], off + done as u64) {
                Ok(0) => {
                    ret = Err(io::ErrorKind::WriteZero.into());
                    break;
                }
                Ok(n) => done += n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => {
                    ret = Err(e);
                    break;
                }
            }
        }
        ret
    };
    ret.map_err(|e| Error::from_io("Unable to write to file", e))
}

/// `qio_channel_pread_all()` on any file handle. Running into the end of the file is an error.
pub fn read_exact_at(file: &File, buf: &mut [u8], off: u64) -> Result<()> {
    #[cfg(unix)]
    let ret = std::os::unix::fs::FileExt::read_exact_at(file, buf, off);
    #[cfg(windows)]
    let ret = {
        let (mut done, mut ret) = (0, Ok(()));
        while done < buf.len() {
            match std::os::windows::fs::FileExt::seek_read(
                file,
                &mut buf[done..],
                off + done as u64,
            ) {
                Ok(0) => {
                    ret = Err(io::ErrorKind::UnexpectedEof.into());
                    break;
                }
                Ok(n) => done += n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => {
                    ret = Err(e);
                    break;
                }
            }
        }
        ret
    };
    ret.map_err(|e| match e.kind() {
        io::ErrorKind::UnexpectedEof => {
            Error::generic("Unexpected end-of-file before all data were read")
        }
        _ => Error::from_io("Unable to read from file", e),
    })
}

/// The sending side of a channel.
#[derive(Debug)]
pub struct Channel;

/// A channel [`Channel::connect_socket`] opened.
pub struct Connection {
    /// Where the stream goes.
    pub out: Outgoing,
    /// For a socket, a second handle on it to read the return path from.
    pub socket: Option<Socket>,
    /// For a file, its seekable side.
    pub file: Option<std::sync::Arc<FileChannel>>,
}

impl std::fmt::Debug for Connection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Connection")
            .field("socket", &self.socket)
            .field("file", &self.file)
            .finish()
    }
}

/// What `migrate` writes the stream into.
pub type Outgoing = Box<dyn Write + Send>;

/// What `migrate-incoming` reads the stream from.
pub type Incoming = Box<dyn Read + Send>;

impl Channel {
    /// `migration_connect_outgoing()`: connects, or opens, the channel at `addr`.
    pub fn connect(addr: &MigrationAddr, fds: Option<&FdResolver>) -> Result<Outgoing> {
        Self::connect_socket(addr, fds).map(|c| c.out)
    }

    /// [`connect`](Self::connect), and for a socket also a second handle on it to read the
    /// return path from, for a file its seekable side.
    pub fn connect_socket(addr: &MigrationAddr, fds: Option<&FdResolver>) -> Result<Connection> {
        let with_socket = |s: Socket| -> Result<Connection> {
            let socket = s.try_clone().ok();
            Ok(Connection {
                out: Box::new(io::BufWriter::with_capacity(1 << 16, s)),
                socket,
                file: None,
            })
        };
        let out: Outgoing = match &addr.u {
            MigrationAddressU::Socket(s) => match &s.u {
                SocketAddressU::Inet(_) => return with_socket(Self::connect_raw(addr)?),
                #[cfg(unix)]
                SocketAddressU::Unix(_) => return with_socket(Self::connect_raw(addr)?),
                SocketAddressU::Fd(f) => {
                    let Some(fds) = fds else { bail!("No file descriptor named {} found", f.str) };
                    #[cfg(unix)]
                    let file = match fd_socket(fds(&f.str)?) {
                        Ok(s) => return with_socket(s),
                        Err(file) => file,
                    };
                    #[cfg(not(unix))]
                    let file = fds(&f.str)?;
                    if file.metadata().is_ok_and(|m| m.is_file()) {
                        bail!("fd: migration to a file is not supported. Use file: instead.");
                    }
                    Box::new(file)
                }
                _ => bail!("uri is not a valid migration protocol"),
            },
            MigrationAddressU::Exec(e) => {
                let mut child = exec_command(&e.args)?
                    .stdin(Stdio::piped())
                    .spawn()
                    .map_err(|err| Error::from_io("Failed to start the migration command", err))?;
                let stdin = child.stdin.take();
                Box::new(ExecWriter { stdin, child })
            }
            MigrationAddressU::File(f) => {
                // file_start_outgoing_migration()
                let mut file = qemu_open(&f.filename, true, true, false)?;
                file.set_len(f.offset).map_err(|e| {
                    Error::from_io(
                        format!("failed to truncate migration file to offset {:x}", f.offset),
                        e,
                    )
                })?;
                file.seek(SeekFrom::Start(f.offset))
                    .map_err(|e| Error::from_io("Unable to seek the migration file", e))?;
                let fc = FileChannel::new(&f.filename, &file, true)?;
                // QemuFile buffers already, and the file position has to follow its flushes.
                return Ok(Connection {
                    out: Box::new(file),
                    socket: None,
                    file: Some(std::sync::Arc::new(fc)),
                });
            }
            MigrationAddressU::Rdma(_) => bail!("RDMA migration is not supported by ruvm"),
        };
        Ok(Connection { out, socket: None, file: None })
    }

    /// `socket_send_channel_create()`: one more connection to a socket address, for a multifd
    /// channel.
    pub fn connect_raw(addr: &MigrationAddr) -> Result<Socket> {
        match &addr.u {
            MigrationAddressU::Socket(s) => match &s.u {
                SocketAddressU::Inet(inet) => {
                    let mut last = None;
                    for a in resolve(inet)? {
                        match TcpStream::connect(a) {
                            Ok(s) => {
                                let _ = s.set_nodelay(true);
                                return Ok(Socket::Tcp(s));
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
                    Ok(Socket::Unix(s))
                }
                _ => bail!("Migration requires multi-channel URIs (e.g. tcp)"),
            },
            _ => bail!("Migration requires multi-channel URIs (e.g. tcp)"),
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
                    #[cfg(unix)]
                    match fd_socket(fds(&f.str)?) {
                        Ok(s) => ListenerKind::Socket(s),
                        Err(file) => ListenerKind::Ready(Box::new(file)),
                    }
                    #[cfg(not(unix))]
                    ListenerKind::Ready(Box::new(fds(&f.str)?))
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
                let mut file = qemu_open(&f.filename, false, false, false)?;
                if f.offset != 0 {
                    file.seek(SeekFrom::Start(f.offset))
                        .map_err(|e| Error::from_io("Unable to seek the migration file", e))?;
                }
                // The stream reader buffers already, and mapped-ram moves the file position
                // under it.
                let fc = FileChannel::new(&f.filename, &file, false)?;
                return Ok(Listener {
                    kind: ListenerKind::Ready(Box::new(file)),
                    file: Some(std::sync::Arc::new(fc)),
                });
            }
            MigrationAddressU::Rdma(_) => bail!("RDMA migration is not supported by ruvm"),
        };
        Ok(Listener { kind, file: None })
    }
}

enum ListenerKind {
    Tcp(TcpListener),
    #[cfg(unix)]
    Unix(std::os::unix::net::UnixListener, String),
    #[cfg(unix)]
    Socket(Socket),
    Ready(Incoming),
    // A one-shot channel that was handed out already.
    Used,
}

/// One connection [`Listener::accept_next`] took.
pub enum Accepted {
    /// A socket, which says itself what it carries.
    Socket(Socket),
    /// A stream that cannot be peeked at, so it is the main channel.
    Stream(Incoming),
}

impl std::fmt::Debug for Accepted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Accepted::Socket(s) => f.debug_tuple("Socket").field(s).finish(),
            Accepted::Stream(_) => f.write_str("Stream"),
        }
    }
}

/// `transport_supports_multi_channels()`: socket addresses can open more than one channel, and
/// a file can with `mapped_ram`, since each channel then writes its pages at their own place.
pub fn supports_multi_channels(addr: &MigrationAddr, mapped_ram: bool) -> bool {
    match &addr.u {
        MigrationAddressU::Socket(s) => {
            matches!(
                s.u,
                SocketAddressU::Inet(_) | SocketAddressU::Unix(_) | SocketAddressU::Vsock(_)
            )
        }
        MigrationAddressU::File(_) => mapped_ram,
        _ => false,
    }
}

/// `transport_supports_seeking()`.
pub fn supports_seeking(addr: &MigrationAddr) -> bool {
    matches!(addr.u, MigrationAddressU::File(_))
}

/// An incoming channel waiting for its source.
pub struct Listener {
    kind: ListenerKind,
    file: Option<std::sync::Arc<FileChannel>>,
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
            #[cfg(unix)]
            ListenerKind::Socket(_) => None,
            ListenerKind::Ready(_) | ListenerKind::Used => None,
        }
    }

    /// The seekable side of a `file:` channel.
    pub fn file(&self) -> Option<std::sync::Arc<FileChannel>> {
        self.file.clone()
    }

    /// Waits for the source to connect, and returns the stream.
    pub fn accept(self) -> Result<Incoming> {
        self.accept_socket().map(|(r, _)| r)
    }

    /// [`accept`](Self::accept), and for a socket also a second handle on it for the return
    /// path.
    pub fn accept_socket(mut self) -> Result<(Incoming, Option<Socket>)> {
        Ok(match self.accept_next()? {
            Accepted::Socket(s) => {
                let back = s.try_clone().ok();
                (Box::new(s), back)
            }
            Accepted::Stream(r) => (r, None),
        })
    }

    /// Waits for the next connection. A listening socket can take any number of them, the other
    /// kinds have just the one.
    pub fn accept_next(&mut self) -> Result<Accepted> {
        match &self.kind {
            ListenerKind::Tcp(l) => {
                let (s, _) = l.accept().map_err(|e| Error::from_io("Failed to accept", e))?;
                let _ = s.set_nodelay(true);
                return Ok(Accepted::Socket(Socket::Tcp(s)));
            }
            #[cfg(unix)]
            ListenerKind::Unix(l, _) => {
                let (s, _) = l.accept().map_err(|e| Error::from_io("Failed to accept", e))?;
                return Ok(Accepted::Socket(Socket::Unix(s)));
            }
            _ => {}
        }
        match std::mem::replace(&mut self.kind, ListenerKind::Used) {
            #[cfg(unix)]
            ListenerKind::Socket(s) => Ok(Accepted::Socket(s)),
            ListenerKind::Ready(r) => Ok(Accepted::Stream(r)),
            _ => bail!("non-peekable channel used without multifd"),
        }
    }
}

impl Drop for Listener {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let ListenerKind::Unix(_, path) = &self.kind {
            let _ = std::fs::remove_file(path);
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
    fn cpr_channels() {
        let ch = |t, uri| MigrationChannel { channel_type: t, addr: parse_uri(uri).unwrap() };
        let main = ch(MigrationChannelType::Main, "unix:/m.sock");
        let cpr = ch(MigrationChannelType::Cpr, "unix:/c.sock");
        let (m, c) = parse_input_cpr(None, Some(&[cpr.clone(), main.clone()]), true).unwrap();
        assert_eq!(addr_to_string(&m), "unix:/m.sock");
        assert_eq!(addr_to_string(&c.unwrap()), "unix:/c.sock");
        let (_, c) = parse_input_cpr(Some("unix:/m.sock"), None, false).unwrap();
        assert!(c.is_none());
        let err = |uri, chans: Option<&[MigrationChannel]>, cpr| {
            parse_input_cpr(uri, chans, cpr).unwrap_err().message().to_string()
        };
        assert_eq!(err(Some("unix:/m.sock"), None, true), "missing 'cpr' migration channel");
        assert_eq!(
            err(None, Some(std::slice::from_ref(&main)), true),
            "missing 'cpr' migration channel"
        );
        assert_eq!(
            err(None, Some(&[main.clone(), main.clone()]), false),
            "Channel list has more than one main entry"
        );
        assert_eq!(
            err(None, Some(std::slice::from_ref(&cpr)), false),
            "Channel list has no main entry"
        );
        // The missing cpr channel is found before the missing main one.
        assert_eq!(err(None, Some(&[]), true), "missing 'cpr' migration channel");
        assert_eq!(err(None, None, false), "need either 'uri' or 'channels' argument");
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

    #[cfg(unix)]
    #[test]
    fn fdset_paths() {
        use rustix::fs::OFlags;
        let path = std::env::temp_dir().join(format!("ruvm-mig-fdset-{}", std::process::id()));
        let p = path.clone();
        set_fdset_opener(std::sync::Arc::new(move |id, flags| {
            if id != 7 {
                bail!("Failed to find fdset /dev/fdset/{id}");
            }
            let write = flags & OFlags::ACCMODE == OFlags::WRONLY;
            let f = OpenOptions::new().read(!write).write(write).create(write).open(&p).unwrap();
            Ok(f.into())
        }));
        let err = open_file("/dev/fdset/x", false, false).unwrap_err();
        assert_eq!(err.message(), "Could not parse fdset /dev/fdset/x");
        let err = open_file("/dev/fdset/2", false, false).unwrap_err();
        assert_eq!(err.message(), "Failed to find fdset /dev/fdset/2");
        let uri = "file:/dev/fdset/7,offset=8";
        let mut w = Channel::connect(&parse_uri(uri).unwrap(), None).unwrap();
        w.write_all(b"fdset").unwrap();
        drop(w);
        let mut r = Channel::listen(&parse_uri(uri).unwrap(), None).unwrap().accept().unwrap();
        let mut v = Vec::new();
        r.read_to_end(&mut v).unwrap();
        assert_eq!(v, b"fdset");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn file_offsets() {
        let path = std::env::temp_dir().join(format!("ruvm-mig-at-{}", std::process::id()));
        let uri = format!("file:{},offset=16", path.display());
        let c = Channel::connect_socket(&parse_uri(&uri).unwrap(), None).unwrap();
        let (mut w, fc) = (c.out, c.file.unwrap());
        assert_eq!(fc.offset().unwrap(), 16);
        w.write_all(b"head").unwrap();
        assert_eq!(fc.offset().unwrap(), 20);
        // A write at an offset leaves the stream where it is.
        fc.write_at(b"far", 100).unwrap();
        assert_eq!(fc.offset().unwrap(), 20);
        fc.set_offset(103).unwrap();
        w.write_all(b"tail").unwrap();
        let extra = open_file(fc.path(), true, false).unwrap();
        write_all_at(&extra, b"mid", 50).unwrap();
        drop((w, fc, extra));

        let l = Channel::listen(&parse_uri(&uri).unwrap(), None).unwrap();
        let fc = l.file().unwrap();
        assert_eq!(fc.offset().unwrap(), 16);
        let mut b = [0u8; 3];
        fc.read_at(&mut b, 50).unwrap();
        assert_eq!(&b, b"mid");
        let mut b = [0u8; 7];
        fc.read_at(&mut b, 100).unwrap();
        assert_eq!(&b, b"fartail");
        let err = fc.read_at(&mut b, 105).unwrap_err();
        assert_eq!(err.message(), "Unexpected end-of-file before all data were read");
        let mut r = l.accept().unwrap();
        let mut v = [0u8; 4];
        r.read_exact(&mut v).unwrap();
        assert_eq!(&v, b"head");
        let _ = std::fs::remove_file(path);
    }
}
