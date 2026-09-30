// SPDX-License-Identifier: GPL-2.0-or-later

//! Tests of ruvm's qemu-nbd: serving an image that ruvm's NBD client and QEMU's tools read,
//! `-L` against QEMU's qemu-nbd on the same server, and error messages against QEMU's. The
//! parts that need QEMU's tools skip themselves when they are not installed. They are looked
//! up in `$QEMU_BIN_DIR`, then /opt/homebrew/bin, /usr/local/bin and /usr/bin.

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use ruvm_block::nbd::NbdClient;
use ruvm_qapi::types::{BlockdevOptionsNbd, SocketAddress, SocketAddressU, UnixSocketAddress};

const OURS: &str = env!("CARGO_BIN_EXE_ruvm-qemu-nbd");

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

/// A scratch directory that is removed at the end of the test. Unix socket paths are short
/// on macOS, so it lives in /tmp.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        static N: AtomicU32 = AtomicU32::new(0);
        let dir = PathBuf::from(format!(
            "/tmp/rnbd-{}-{name}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    /// A 4 MiB raw image whose first 64 KiB are 0x5a and the rest zero.
    fn image(&self) -> PathBuf {
        let p = self.path("a.img");
        let mut data = vec![0u8; 4 << 20];
        data[..65536].fill(0x5a);
        std::fs::write(&p, data).unwrap();
        p
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A server process that is killed when the test ends.
struct Server(Child);

impl Drop for Server {
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

fn serve(sock: &Path, args: &[&str], image: &Path) -> Server {
    let child = Command::new(OURS)
        .arg("-k")
        .arg(sock)
        .args(args)
        .arg(image)
        .stdin(Stdio::null())
        .spawn()
        .unwrap();
    wait_for(sock);
    Server(child)
}

fn unix_addr(path: &Path) -> SocketAddress {
    SocketAddress {
        u: SocketAddressU::Unix(
            // The abstract socket fields exist only on Linux.
            #[allow(clippy::needless_update)]
            UnixSocketAddress {
                path: path.to_str().unwrap().to_string(),
                ..UnixSocketAddress::default()
            },
        ),
    }
}

fn open_client(sock: &Path, export: &str) -> NbdClient {
    let o = BlockdevOptionsNbd {
        server: unix_addr(sock),
        export: Some(export.into()),
        ..BlockdevOptionsNbd::default()
    };
    let mut ro = false;
    NbdClient::open(&o, None, &mut ro, true).unwrap()
}

fn text(o: &[u8]) -> String {
    String::from_utf8_lossy(o).into_owned()
}

/// Runs `bin` with `args` and replaces `bin`'s own path in its messages by `qemu-nbd`.
fn run(bin: &Path, args: &[&str]) -> (String, String, Option<i32>) {
    let out: Output = Command::new(bin).args(args).stdin(Stdio::null()).output().unwrap();
    let b = bin.to_str().unwrap();
    (
        text(&out.stdout).replace(b, "qemu-nbd"),
        text(&out.stderr).replace(b, "qemu-nbd"),
        out.status.code(),
    )
}

#[test]
fn ruvm_client_reads_the_export() {
    let s = Scratch::new("client");
    let img = s.image();
    let sock = s.path("s");
    let _srv = serve(&sock, &["-f", "raw", "-t", "-x", "ex", "-r"], &img);
    let c = open_client(&sock, "ex");
    assert_eq!(c.size(), 4 << 20);
    assert!(c.is_read_only());
    let mut buf = vec![0u8; 131072];
    c.pread(0, &mut buf).unwrap();
    assert!(buf[..65536].iter().all(|&b| b == 0x5a));
    assert!(buf[65536..].iter().all(|&b| b == 0));
    c.close();
}

#[test]
fn ruvm_client_writes_through_the_export() {
    let s = Scratch::new("write");
    let img = s.image();
    let sock = s.path("s");
    let srv = serve(&sock, &["-f", "raw"], &img);
    let c = open_client(&sock, "");
    c.pwrite(1 << 20, &[0xa5; 4096], false).unwrap();
    c.flush().unwrap();
    c.close();
    drop(c);
    // Without -t the server exits after its only client.
    let mut srv = srv;
    let start = Instant::now();
    loop {
        if let Some(st) = srv.0.try_wait().unwrap() {
            assert!(st.success());
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(20), "qemu-nbd did not exit");
        std::thread::sleep(Duration::from_millis(20));
    }
    let data = std::fs::read(&img).unwrap();
    assert!(data[1 << 20..(1 << 20) + 4096].iter().all(|&b| b == 0xa5));
}

#[test]
fn qemu_tools_read_the_export() {
    let (Some(qimg), Some(qio)) = (qemu_tool("qemu-img"), qemu_tool("qemu-io")) else {
        return;
    };
    let s = Scratch::new("qemu");
    let img = s.image();
    let sock = s.path("s");
    let _srv = serve(&sock, &["-f", "raw", "-t", "-x", "ex"], &img);
    let uri = format!("nbd+unix:///ex?socket={}", sock.display());
    let (out, err, code) = run(&qimg, &["info", &uri]);
    assert_eq!(code, Some(0), "{err}");
    assert!(out.contains("virtual size: 4 MiB (4194304 bytes)"), "{out}");
    let (out, err, code) = run(&qio, &["-f", "raw", "-c", "read -P 0x5a 0 64k", &uri]);
    assert_eq!(code, Some(0), "{err}");
    assert!(out.starts_with("read 65536/65536 bytes at offset 0"), "{out}");
    let (out, _, _) = run(&qio, &["-f", "raw", "-c", "read -P 0 64k 64k", &uri]);
    assert!(out.starts_with("read 65536/65536 bytes at offset 65536"), "{out}");
}

#[test]
fn list_matches_qemu() {
    let Some(real) = qemu_tool("qemu-nbd") else {
        return;
    };
    let s = Scratch::new("list");
    let img = s.image();
    let sock = s.path("s");
    let _srv = serve(&sock, &["-f", "raw", "-t", "-x", "ex", "-D", "a description", "-A"], &img);
    let sk = sock.to_str().unwrap();
    let ours = run(Path::new(OURS), &["-L", "-k", sk]);
    let theirs = run(&real, &["-L", "-k", sk]);
    assert_eq!(ours.2, Some(0), "{}", ours.1);
    assert!(ours.0.contains("export: 'ex'"), "{}", ours.0);
    assert!(ours.0.contains("description: a description"), "{}", ours.0);
    assert_eq!(ours, theirs);
}

#[test]
fn list_without_qemu() {
    let s = Scratch::new("list2");
    let img = s.image();
    let sock = s.path("s");
    let _srv = serve(&sock, &["-f", "raw", "-t", "-x", "ex"], &img);
    let (out, err, code) = run(Path::new(OURS), &["-L", "-k", sock.to_str().unwrap()]);
    assert_eq!(code, Some(0), "{err}");
    assert!(out.starts_with("exports available: 1\n export: 'ex'\n"), "{out}");
    assert!(out.contains("  size:  4194304\n"), "{out}");
}

#[test]
fn errors_match_qemu() {
    let Some(real) = qemu_tool("qemu-nbd") else {
        return;
    };
    let s = Scratch::new("errors");
    let img = s.image();
    let i = img.to_str().unwrap();
    let missing = s.path("missing.img");
    let m = missing.to_str().unwrap();
    let cases: &[&[&str]] = &[
        &[],
        &["--bogus"],
        &["-p"],
        &["-p", "x", i],
        &["-p", "70000", i],
        &["-e", "x", i],
        &["--persistent", "--shared=1", i, i],
        &["-f", "nosuchfmt", i],
        &["--cache=bogus", i],
        &["--aio=bogus", i],
        &["--discard=bogus", i],
        &["--detect-zeroes=bogus", i],
        &["--detect-zeroes=unmap", i],
        &["-o", "x", i],
        &["-l", "snap", "-s", i],
        &["-L", i],
        &["-k", "/tmp/x", "-b", "localhost", i],
        &["-x", "e", "-L", "-k", "/nonexistent/s"],
        &["--tls-creds", "tls0", "-k", "/tmp/x", i],
        &["--tls-authz", "a", i],
        &["-B", "b", "-f", "raw", i],
        &["-f", "raw", m],
        &["-d"],
        &["--object", "nosuchtype,id=x", i],
    ];
    for args in cases {
        let ours = run(Path::new(OURS), args);
        let theirs = run(&real, args);
        assert_eq!(ours, theirs, "{args:?}");
    }
}

#[test]
fn help_and_version() {
    let (out, _, code) = run(Path::new(OURS), &["--help"]);
    assert_eq!(code, Some(0));
    assert!(out.starts_with("Usage: qemu-nbd [OPTIONS] FILE\n"), "{out}");
    let (out, _, code) = run(Path::new(OURS), &["-V"]);
    assert_eq!(code, Some(0));
    assert!(out.starts_with("qemu-nbd 11.1.0"), "{out}");
    if let Some(real) = qemu_tool("qemu-nbd") {
        let theirs = run(&real, &["--help"]);
        assert_eq!(out_lines(&run(Path::new(OURS), &["--help"]).0), out_lines(&theirs.0));
    }
}

/// Help text without the lines that name the build: the bug report and home page lines.
fn out_lines(s: &str) -> Vec<String> {
    s.lines().filter(|l| !l.contains("http")).map(str::to_string).collect()
}

#[test]
fn fork_writes_the_pid_file() {
    let s = Scratch::new("fork");
    let img = s.image();
    let sock = s.path("s");
    let pid = s.path("pid");
    let out = Command::new(OURS)
        .args(["-f", "raw", "-t", "--fork", "--pid-file"])
        .arg(&pid)
        .arg("-k")
        .arg(&sock)
        .arg(&img)
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", text(&out.stderr));
    // --fork returns only once the server listens.
    assert!(sock.exists());
    let p: i32 = std::fs::read_to_string(&pid).unwrap().trim().parse().unwrap();
    let c = open_client(&sock, "");
    let mut buf = [0u8; 512];
    c.pread(0, &mut buf).unwrap();
    assert_eq!(buf, [0x5a; 512]);
    c.close();
    let _ = Command::new("kill").arg(p.to_string()).status();
    let start = Instant::now();
    while pid.exists() && start.elapsed() < Duration::from_secs(20) {
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn snapshot_mode_does_not_touch_the_image() {
    if !ruvm_block::tools::format_exists("qcow2") {
        eprintln!("skipping: qcow2 is not registered, and -s needs it");
        return;
    }
    let s = Scratch::new("snap");
    let img = s.image();
    let sock = s.path("s");
    let _srv = serve(&sock, &["-f", "raw", "-t", "-s"], &img);
    let c = open_client(&sock, "");
    c.pwrite(0, &[1; 4096], false).unwrap();
    let mut buf = [0u8; 4096];
    c.pread(0, &mut buf).unwrap();
    assert_eq!(buf, [1; 4096]);
    c.close();
    assert_eq!(std::fs::read(&img).unwrap()[..4096], [0x5a; 4096]);
}
