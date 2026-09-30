// SPDX-License-Identifier: GPL-2.0-or-later

//! Tests of ruvm's qemu-storage-daemon: a QMP session over a unix socket that adds a node,
//! exports it over NBD and reads it back, and the command line errors and help text against
//! QEMU's qemu-storage-daemon when it is installed (looked up in `$QEMU_BIN_DIR`, then
//! /opt/homebrew/bin, /usr/local/bin and /usr/bin).

#![cfg(unix)]

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use ruvm_block::nbd::NbdClient;
use ruvm_qapi::types::{BlockdevOptionsNbd, SocketAddress, SocketAddressU, UnixSocketAddress};

const OURS: &str = env!("CARGO_BIN_EXE_ruvm-qemu-storage-daemon");

fn qemu_tool(name: &str) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(d) = std::env::var_os("QEMU_BIN_DIR") {
        dirs.push(d.into());
    }
    dirs.extend(["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"].map(PathBuf::from));
    let found = dirs.into_iter().map(|d| d.join(name)).find(|p| p.exists());
    if found.is_none() {
        eprintln!("skipping the part that needs {name}: it is not installed");
    }
    found
}

/// A scratch directory in /tmp, where unix socket paths stay short, removed at the end.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        static N: AtomicU32 = AtomicU32::new(0);
        let dir = PathBuf::from(format!(
            "/tmp/rqsd-{}-{name}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    fn s(&self, name: &str) -> String {
        self.path(name).to_str().unwrap().to_string()
    }

    /// A 4 MiB raw image whose first 64 KiB are 0x5a and the rest zero.
    fn image(&self) -> String {
        let p = self.path("a.img");
        let mut data = vec![0u8; 4 << 20];
        data[..65536].fill(0x5a);
        std::fs::write(&p, data).unwrap();
        p.to_str().unwrap().to_string()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Daemon(Child);

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn wait_for(path: &Path) {
    let start = Instant::now();
    while !path.exists() {
        assert!(start.elapsed() < Duration::from_secs(20), "{} never appeared", path.display());
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// A QMP client on a unix socket.
struct Qmp {
    r: BufReader<UnixStream>,
    w: UnixStream,
    events: Vec<String>,
}

impl Qmp {
    fn connect(path: &Path) -> Qmp {
        wait_for(path);
        let w = UnixStream::connect(path).unwrap();
        w.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
        Qmp { r: BufReader::new(w.try_clone().unwrap()), w, events: Vec::new() }
    }

    fn line(&mut self) -> String {
        let mut l = String::new();
        self.r.read_line(&mut l).unwrap();
        assert!(!l.is_empty(), "the monitor closed the connection");
        l.trim_end().to_string()
    }

    /// Sends `cmd` and returns the reply, keeping the events that come before it.
    fn cmd(&mut self, cmd: &str) -> String {
        self.w.write_all(cmd.as_bytes()).unwrap();
        self.w.write_all(b"\n").unwrap();
        loop {
            let l = self.line();
            if l.starts_with("{\"timestamp\"") || l.contains("\"event\"") {
                self.events.push(l);
            } else {
                return l;
            }
        }
    }

    fn ok(&mut self, cmd: &str) {
        assert_eq!(self.cmd(cmd), r#"{"return": {}}"#, "{cmd}");
    }
}

fn nbd_client(sock: &Path, export: &str) -> NbdClient {
    let o = BlockdevOptionsNbd {
        server: SocketAddress {
            u: SocketAddressU::Unix(
                // The abstract socket fields exist only on Linux.
                #[allow(clippy::needless_update)]
                UnixSocketAddress {
                    path: sock.to_str().unwrap().to_string(),
                    ..UnixSocketAddress::default()
                },
            ),
        },
        export: Some(export.into()),
        ..BlockdevOptionsNbd::default()
    };
    let mut ro = false;
    NbdClient::open(&o, None, &mut ro, true).unwrap()
}

#[test]
fn qmp_session_exports_over_nbd() {
    let s = Scratch::new("qmp");
    let img = s.image();
    let mon = s.path("qmp");
    let nbd = s.path("nbd");
    let pid = s.path("pid");
    let child = Command::new(OURS)
        .args(["--chardev", &format!("socket,id=m,path={},server=on,wait=off", mon.display())])
        .args(["--monitor", "chardev=m"])
        .args(["--pidfile", pid.to_str().unwrap()])
        .stdin(Stdio::null())
        .spawn()
        .unwrap();
    let mut d = Daemon(child);
    let mut q = Qmp::connect(&mon);

    let greeting = q.line();
    assert!(
        greeting
            .starts_with(r#"{"QMP": {"version": {"qemu": {"micro": 0, "minor": 1, "major": 11}"#),
        "{greeting}"
    );
    // A socket monitor offers out-of-band execution, as in QEMU.
    assert!(greeting.ends_with(r#""capabilities": ["oob"]}}"#), "{greeting}");
    let e = q.cmd(r#"{"execute":"query-block-exports"}"#);
    assert!(e.contains("CommandNotFound"), "{e}");
    q.ok(r#"{"execute":"qmp_capabilities"}"#);
    assert!(pid.exists());

    q.ok(&format!(
        r#"{{"execute":"blockdev-add","arguments":{{"driver":"file","filename":"{img}","node-name":"n"}}}}"#
    ));
    assert_eq!(
        q.cmd(
            r#"{"execute":"block-export-add","arguments":{"type":"nbd","id":"e","node-name":"n"}}"#
        ),
        r#"{"error": {"class": "GenericError", "desc": "NBD server not running"}}"#
    );
    q.ok(&format!(
        r#"{{"execute":"nbd-server-start","arguments":{{"addr":{{"type":"unix","data":{{"path":"{}"}}}}}}}}"#,
        nbd.display()
    ));
    assert_eq!(
        q.cmd(r#"{"execute":"nbd-server-start","arguments":{"addr":{"type":"unix","data":{"path":"/tmp/x"}}}}"#),
        r#"{"error": {"class": "GenericError", "desc": "NBD server already running"}}"#
    );
    assert_eq!(
        q.cmd(r#"{"execute":"block-export-add","arguments":{"type":"nbd","id":"e","node-name":"n","writable":true,"iothread":"io0"}}"#),
        r#"{"error": {"class": "GenericError", "desc": "iothread \"io0\" not found"}}"#
    );
    q.ok(r#"{"execute":"block-export-add","arguments":{"type":"nbd","id":"e","node-name":"n","writable":true}}"#);
    assert_eq!(
        q.cmd(r#"{"execute":"query-block-exports"}"#),
        r#"{"return": [{"node-name": "n", "shutting-down": false, "type": "nbd", "id": "e"}]}"#
    );
    let nodes = q.cmd(r#"{"execute":"query-named-block-nodes","arguments":{"flat":true}}"#);
    assert!(nodes.contains(r#""node-name": "n""#), "{nodes}");

    // Read and write through the export.
    let c = nbd_client(&nbd, "n");
    assert_eq!(c.size(), 4 << 20);
    let mut buf = vec![0u8; 131072];
    c.pread(0, &mut buf).unwrap();
    assert!(buf[..65536].iter().all(|&b| b == 0x5a));
    assert!(buf[65536..].iter().all(|&b| b == 0));
    c.pwrite(1 << 20, &[0xa5; 512], false).unwrap();
    c.flush().unwrap();
    c.close();
    drop(c);
    assert!(std::fs::read(&img).unwrap()[1 << 20..(1 << 20) + 512].iter().all(|&b| b == 0xa5));

    // QEMU's tools read it too.
    if let Some(qio) = qemu_tool("qemu-io") {
        let uri = format!("nbd+unix:///n?socket={}", nbd.display());
        let out = Command::new(qio)
            .args(["-r", "-f", "raw", "-c", "read -P 0x5a 0 64k", &uri])
            .output()
            .unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    }

    // The legacy commands, then removal with its event.
    q.ok(r#"{"execute":"nbd-server-add","arguments":{"device":"n","name":"legacy"}}"#);
    assert_eq!(
        q.cmd(r#"{"execute":"nbd-server-add","arguments":{"device":"nope"}}"#),
        r#"{"error": {"class": "GenericError", "desc": "Cannot find device='nope' nor node-name='nope'"}}"#
    );
    q.ok(r#"{"execute":"nbd-server-remove","arguments":{"name":"legacy"}}"#);
    q.ok(r#"{"execute":"block-export-del","arguments":{"id":"e"}}"#);
    let start = Instant::now();
    while q.events.iter().filter(|e| e.contains("BLOCK_EXPORT_DELETED")).count() < 2 {
        assert!(start.elapsed() < Duration::from_secs(20), "events: {:?}", q.events);
        // Any command lets the events that came meanwhile arrive first.
        q.cmd(r#"{"execute":"query-block-exports"}"#);
    }
    assert!(q.events.iter().any(|e| e.contains(r#""data": {"id": "legacy"}"#)), "{:?}", q.events);
    assert!(q.events.iter().any(|e| e.contains(r#""data": {"id": "e"}"#)), "{:?}", q.events);
    assert_eq!(q.cmd(r#"{"execute":"query-block-exports"}"#), r#"{"return": []}"#);
    q.ok(r#"{"execute":"nbd-server-stop"}"#);
    q.ok(r#"{"execute":"blockdev-del","arguments":{"node-name":"n"}}"#);
    let jobs = q.cmd(r#"{"execute":"query-jobs"}"#);
    assert_eq!(jobs, r#"{"return": []}"#);
    q.ok(r#"{"execute":"quit"}"#);

    let start = Instant::now();
    let st = loop {
        if let Some(st) = d.0.try_wait().unwrap() {
            break st;
        }
        assert!(start.elapsed() < Duration::from_secs(20), "the daemon did not quit");
        std::thread::sleep(Duration::from_millis(20));
    };
    assert!(st.success());
    assert!(!pid.exists(), "the pid file is still there");
}

#[test]
fn command_line_exports() {
    let s = Scratch::new("cli");
    let img = s.image();
    let nbd = s.path("nbd");
    let child = Command::new(OURS)
        .args(["--blockdev", &format!("driver=file,filename={img},node-name=n,read-only=on")])
        .args(["--nbd-server", &format!("addr.type=unix,addr.path={}", nbd.display())])
        .args(["--export", "type=nbd,id=e,node-name=n,name=disk"])
        .stdin(Stdio::null())
        .spawn()
        .unwrap();
    let _d = Daemon(child);
    wait_for(&nbd);
    let c = nbd_client(&nbd, "disk");
    assert!(c.is_read_only());
    let mut buf = [0u8; 512];
    c.pread(0, &mut buf).unwrap();
    assert_eq!(buf, [0x5a; 512]);
    c.close();
}

fn run(bin: &Path, args: &[&str]) -> (String, String, Option<i32>) {
    let out = Command::new(bin).args(args).stdin(Stdio::null()).output().unwrap();
    let b = bin.to_str().unwrap();
    (
        String::from_utf8_lossy(&out.stdout).replace(b, "qemu-storage-daemon"),
        String::from_utf8_lossy(&out.stderr).replace(b, "qemu-storage-daemon"),
        out.status.code(),
    )
}

#[test]
fn errors_and_help_match_qemu() {
    let Some(real) = qemu_tool("qemu-storage-daemon") else {
        return;
    };
    let s = Scratch::new("errors");
    let img = s.image();
    let bd = format!("driver=file,filename={img},node-name=n");
    let pidfile = s.s("nodir/pid");
    let cases: Vec<Vec<&str>> = vec![
        vec!["-h"],
        vec!["x"],
        vec!["--bogus"],
        vec!["--blockdev"],
        vec!["--blockdev", "driver=nosuch,node-name=a"],
        vec!["--blockdev", "{bad json"],
        vec!["--blockdev", "node-name=a"],
        vec!["--chardev", "socket"],
        vec!["--monitor", "nosuch"],
        vec!["--chardev", "null,id=c", "--monitor", "c,mode=readline"],
        vec!["--export", "type=nbd,id=e,node-name=n"],
        vec!["--blockdev", &bd, "--export", "type=nbd,id=e,node-name=n"],
        vec!["--blockdev", &bd, "--export", "type=nbd,node-name=n"],
        vec!["--blockdev", &bd, "--export", "type=fuse,id=e,node-name=n,mountpoint=/x"],
        vec!["--blockdev", &bd, "--export", "type=bogus,id=e,node-name=n"],
        vec!["--nbd-server", "addr.type=unix"],
        vec!["--object", "nosuchtype,id=x"],
        vec!["--pidfile", &pidfile],
    ];
    for args in &cases {
        // The daemon's export types depend on the host QEMU was built for; skip the ones
        // that differ between ruvm's build and QEMU's on this host.
        if cfg!(target_os = "linux") && args.iter().any(|a| a.starts_with("type=fuse")) {
            continue;
        }
        let ours = run(Path::new(OURS), args);
        let theirs = run(&real, args);
        assert_eq!(ours, theirs, "{args:?}");
    }
}

#[test]
fn version() {
    let (out, _, code) = run(Path::new(OURS), &["-V"]);
    assert_eq!(code, Some(0));
    assert!(out.starts_with("qemu-storage-daemon version 11.1.0"), "{out}");
}
