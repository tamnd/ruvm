// SPDX-License-Identifier: GPL-2.0-or-later

use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Duration;

use ruvm_chardev::{Chardevs, Connection, Frontend};
use ruvm_qapi::types::{
    ChardevBackend, ChardevBackendU, ChardevCommonWrapper, ChardevSocket, ChardevSocketWrapper,
    InetSocketAddress, InetSocketAddressWrapper, SocketAddressLegacy, SocketAddressLegacyU,
};

/// Writes back every byte it reads, with a prefix so a test can tell frontends apart.
struct Echo(&'static str);

impl Frontend for Echo {
    fn serve(&self, conn: &mut Connection) -> io::Result<()> {
        let mut out = conn.writer()?;
        let mut buf = [0u8; 256];
        loop {
            let n = conn.recv(&mut buf)?;
            if n == 0 {
                return Ok(());
            }
            out.write_all(self.0.as_bytes())?;
            out.write_all(&buf[..n])?;
        }
    }
}

fn null() -> ChardevBackend {
    ChardevBackend { u: ChardevBackendU::Null(ChardevCommonWrapper::default()) }
}

fn socket(addr: SocketAddressLegacyU, server: bool) -> ChardevBackend {
    let data = ChardevSocket {
        addr: SocketAddressLegacy { u: addr },
        server: Some(server),
        wait: server.then_some(false),
        ..Default::default()
    };
    ChardevBackend { u: ChardevBackendU::Socket(ChardevSocketWrapper { data }) }
}

fn tcp(port: &str) -> SocketAddressLegacyU {
    SocketAddressLegacyU::Inet(InetSocketAddressWrapper {
        data: InetSocketAddress {
            host: "127.0.0.1".into(),
            port: port.into(),
            ..Default::default()
        },
    })
}

fn read_exact(s: &mut impl Read, n: usize) -> String {
    let mut buf = vec![0u8; n];
    s.read_exact(&mut buf).unwrap();
    String::from_utf8(buf).unwrap()
}

#[test]
fn add_query_remove() {
    let chardevs = Chardevs::new();
    chardevs.add("a", &null()).unwrap();
    chardevs.add("b", &null()).unwrap();
    let e = chardevs.add("a", &null()).unwrap_err();
    assert_eq!(e.message(), "Failed to add chardev 'a': Chardev with id 'a' already exists");
    let labels: Vec<String> = chardevs.query().into_iter().map(|i| i.label).collect();
    assert_eq!(labels, ["b", "a"]);
    assert_eq!(chardevs.query()[0].filename, "null");

    let a = chardevs.find("a").unwrap();
    let fe = a.attach(Arc::new(Echo(""))).unwrap();
    let e = a.attach(Arc::new(Echo(""))).unwrap_err();
    assert_eq!(e.message(), "chardev 'a' is already in use");
    assert!(chardevs.query()[1].frontend_open);
    assert_eq!(chardevs.remove("a").unwrap_err().message(), "Chardev 'a' is busy");
    drop(fe);
    chardevs.remove("a").unwrap();
    assert_eq!(chardevs.remove("a").unwrap_err().message(), "Chardev 'a' not found");
}

#[test]
fn socket_options_are_checked() {
    let chardevs = Chardevs::new();
    let mut b = socket(tcp("0"), false);
    if let ChardevBackendU::Socket(s) = &mut b.u {
        s.data.wait = Some(true);
    }
    let e = chardevs.add("s", &b).unwrap_err();
    assert_eq!(
        e.message(),
        "Failed to add chardev 's': 'wait' option is incompatible with socket in client connect mode"
    );
    let e = chardevs.add("s", &socket(tcp("http"), true)).unwrap_err();
    assert_eq!(e.message(), "Failed to add chardev 's': can't convert to a number: http");
}

#[test]
fn tcp_server_serves_one_client_after_another() {
    let chardevs = Chardevs::new();
    let chr = chardevs.add("srv", &socket(tcp("0"), true)).unwrap();
    let name = chr.filename();
    let port = name
        .strip_prefix("disconnected:tcp:127.0.0.1:")
        .and_then(|r| r.strip_suffix(",server=on"))
        .unwrap_or_else(|| panic!("{name}"))
        .to_string();
    let _fe = chr.attach(Arc::new(Echo("> "))).unwrap();

    for round in 0..2 {
        let mut c = TcpStream::connect(("127.0.0.1", port.parse::<u16>().unwrap())).unwrap();
        c.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        c.write_all(b"hi").unwrap();
        assert_eq!(read_exact(&mut c, 4), "> hi", "round {round}");
        let local = c.local_addr().unwrap();
        let name = chr.filename();
        assert_eq!(name, format!("tcp:127.0.0.1:{port},server=on <-> 127.0.0.1:{}", local.port()));
        drop(c);
        // The server notices the client left and goes back to listening.
        let want = format!("disconnected:tcp:127.0.0.1:{port},server=on");
        for _ in 0..100 {
            if chr.filename() == want {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(chr.filename(), want);
    }
}

#[cfg(unix)]
mod unix {
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::path::PathBuf;

    use ruvm_qapi::types::{UnixSocketAddress, UnixSocketAddressWrapper};

    use super::*;

    fn path(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("ruvm-chardev-{}-{name}", std::process::id()));
        let _ = std::fs::remove_file(&p);
        p
    }

    // Linux hosts have the abstract and tight members too.
    #[allow(clippy::needless_update)]
    fn unix(p: &std::path::Path) -> SocketAddressLegacyU {
        SocketAddressLegacyU::Unix(UnixSocketAddressWrapper {
            data: UnixSocketAddress { path: p.to_str().unwrap().into(), ..Default::default() },
        })
    }

    #[test]
    fn client_keeps_its_connection_between_frontends() {
        let p = path("client");
        let listener = UnixListener::bind(&p).unwrap();
        let chardevs = Chardevs::new();
        let chr = chardevs.add("cli", &socket(unix(&p), false)).unwrap();
        assert_eq!(chr.filename(), format!("unix:{}", p.display()));
        let (mut peer, _) = listener.accept().unwrap();
        peer.set_read_timeout(Some(Duration::from_secs(10))).unwrap();

        // Bytes sent before any frontend is attached wait for one.
        peer.write_all(b"early").unwrap();
        let fe = chr.attach(Arc::new(Echo("1:"))).unwrap();
        assert_eq!(read_exact(&mut peer, 7), "1:early");
        fe.join();

        let _fe = chr.attach(Arc::new(Echo("2:"))).unwrap();
        peer.write_all(b"late").unwrap();
        assert_eq!(read_exact(&mut peer, 6), "2:late");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn server_removes_its_socket() {
        let p = path("server");
        let chardevs = Chardevs::new();
        let chr = chardevs.add("srv", &socket(unix(&p), true)).unwrap();
        assert_eq!(chr.filename(), format!("disconnected:unix:{},server=on", p.display()));
        let fe = chr.attach(Arc::new(Echo(""))).unwrap();
        let mut c = UnixStream::connect(&p).unwrap();
        c.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        c.write_all(b"x").unwrap();
        assert_eq!(read_exact(&mut c, 1), "x");
        assert_eq!(chr.filename(), format!("unix:{},server=on", p.display()));
        fe.join();
        drop(chr);
        chardevs.remove("srv").unwrap();
        assert!(!p.exists());
    }
}
