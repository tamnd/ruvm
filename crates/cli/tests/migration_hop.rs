// SPDX-License-Identifier: GPL-2.0-or-later

//! A live migration from one ruvm to another over TCP, with the guests from scripts/migration.
//!
//! Ignored by default: it needs the guest images, which `scripts/migration/build-guests.sh DIR`
//! assembles, and takes a minute. Run it with
//!
//! ```text
//! RUVM_MIGRATION_GUESTS=DIR cargo test -p ruvm-cli --release --test migration_hop -- --ignored
//! ```
//!
//! Each guest found in `DIR` (`checksum-guest.bin`, `irq-guest.bin`) runs on a source, moves to
//! a destination while it runs, and must carry on there: its report lines, joined across the
//! two outputs, are checked the way scripts/migration/expect.py checks them. Both run on q35,
//! the interrupt-driven one on microvm too.

#![cfg(unix)]

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const K: u32 = 0x0100_0193;

/// The guests and how to read their output.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Guest {
    /// checksum-guest.S: "STEP SUM OK" on debugcon.
    Checksum,
    /// irq-guest.S: "REPORT PIT APIC STEP SUM OK" on COM1, written by interrupt handlers.
    Irq,
}

impl Guest {
    fn image(self) -> &'static str {
        match self {
            Guest::Checksum => "checksum-guest.bin",
            Guest::Irq => "irq-guest.bin",
        }
    }

    fn npages(self) -> u32 {
        match self {
            Guest::Checksum => 2048,
            Guest::Irq => 256,
        }
    }

    fn output_args(self, path: &Path) -> Vec<String> {
        let path = path.display();
        match self {
            Guest::Checksum => vec![
                "-chardev".into(),
                format!("file,id=d,path={path}"),
                "-device".into(),
                "isa-debugcon,iobase=0xe9,chardev=d".into(),
            ],
            Guest::Irq => vec!["-serial".into(), format!("file:{path}")],
        }
    }
}

/// The checksum of the guest's area after `step` steps.
fn want(guest: Guest, step: u32) -> u32 {
    let mut s = 0u32;
    for p in (0..guest.npages()).filter(|p| p % 4 != 0) {
        let b = p.wrapping_mul(0x9E37_79B1);
        s = s.wrapping_add(b.wrapping_mul(1024)).wrapping_add(K.wrapping_mul(1023 * 512));
    }
    let i = u64::from(step);
    let tri = (i * (i + 1) / 2) as u32;
    s.wrapping_add(K.wrapping_mul(tri)).wrapping_add(step)
}

/// The complete lines of `text`, split into fields.
fn lines(text: &str) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    for line in text.split_inclusive('\n').filter(|l| l.ends_with('\n')) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let (ok, nums) = f.split_last().expect("empty line");
        assert_eq!(*ok, "OK", "guest reported {line:?}");
        out.push(nums.iter().map(|n| u32::from_str_radix(n, 16).expect(line)).collect());
    }
    out
}

/// Checks the joined output and returns the number of lines.
fn check(guest: Guest, text: &str) -> usize {
    let lines = lines(text);
    for pair in lines.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        match guest {
            Guest::Checksum => assert_eq!(b[0], a[0].wrapping_add(0x4000), "{a:x?} {b:x?}"),
            Guest::Irq => {
                assert_eq!(b[0], a[0] + 1, "{a:x?} {b:x?}");
                assert!(b[1] >= a[1] && b[2] >= a[2] && b[3] > a[3], "{a:x?} {b:x?}");
            }
        }
    }
    for l in &lines {
        let (step, sum) = (l[l.len() - 2], l[l.len() - 1]);
        assert_eq!(sum, want(guest, step), "{l:x?}");
        if guest == Guest::Irq {
            assert!(l[1] >= 25 * l[0], "PIT ticks behind: {l:x?}");
        }
    }
    if guest == Guest::Irq && lines.len() > 1 {
        let (first, last) = (&lines[0], &lines[lines.len() - 1]);
        assert!(last[1] > first[1] && last[2] > first[2], "timers stopped: {first:x?} {last:x?}");
    }
    lines.len()
}

/// A QMP connection.
struct Qmp(BufReader<UnixStream>);

impl Qmp {
    fn connect(path: &Path) -> Qmp {
        let start = Instant::now();
        let s = loop {
            match UnixStream::connect(path) {
                Ok(s) => break s,
                Err(_) if start.elapsed() < Duration::from_secs(30) => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(e) => panic!("{}: {e}", path.display()),
            }
        };
        let mut q = Qmp(BufReader::new(s));
        q.line();
        q.cmd(r#"{"execute":"qmp_capabilities"}"#);
        q
    }

    fn line(&mut self) -> String {
        let mut l = String::new();
        self.0.read_line(&mut l).unwrap();
        assert!(!l.is_empty(), "QMP connection closed");
        l
    }

    /// Sends `c` and returns its reply without whitespace, skipping events.
    fn cmd(&mut self, c: &str) -> String {
        self.0.get_mut().write_all(format!("{c}\n").as_bytes()).unwrap();
        loop {
            let l: String = self.line().chars().filter(|c| !c.is_whitespace()).collect();
            // Events start with their timestamp: {"timestamp":{...},"event":"RESUME"}.
            if !l.contains(r#""event":"#) || l.contains(r#""return":"#) {
                assert!(!l.contains(r#""error""#), "{c}: {l}");
                return l;
            }
        }
    }
}

struct Vm {
    child: Child,
    out: PathBuf,
    qmp: Qmp,
}

impl Vm {
    fn start(dir: &Path, name: &str, hop: &Hop<'_>, extra: &[String]) -> Vm {
        let Hop { images, guest, machine } = *hop;
        let out = dir.join(format!("{name}.txt"));
        let sock = dir.join(format!("{name}.sock"));
        let _ = std::fs::remove_file(&sock);
        let child = Command::new(env!("CARGO_BIN_EXE_ruvm"))
            .arg("qemu-system-x86_64")
            .args(["-M", machine, "-nodefaults", "-accel", "tcg", "-display", "none", "-m", "128M"])
            .arg("-bios")
            .arg(images.join(guest.image()))
            .args(guest.output_args(&out))
            .arg("-qmp")
            .arg(format!("unix:{},server=on,wait=off", sock.display()))
            .args(extra)
            .stdin(Stdio::null())
            .spawn()
            .unwrap();
        let qmp = Qmp::connect(&sock);
        Vm { child, out, qmp }
    }

    fn output(&self) -> String {
        std::fs::read_to_string(&self.out).unwrap_or_default()
    }

    /// Waits until the output has `n` more complete lines than `base`.
    fn wait_lines(&self, base: usize, n: usize, timeout: Duration) {
        let start = Instant::now();
        while self.output().matches('\n').count() < base + n {
            assert!(start.elapsed() < timeout, "{}: no output", self.out.display());
            std::thread::sleep(Duration::from_millis(200));
        }
    }
}

impl Drop for Vm {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// One migration: the guest images, the guest and the `-M` value.
#[derive(Clone, Copy, Debug)]
struct Hop<'a> {
    images: &'a Path,
    guest: Guest,
    machine: &'a str,
}

fn hop(h: &Hop<'_>) {
    let Hop { guest, machine, .. } = *h;
    let id = std::process::id();
    let dir = std::env::temp_dir().join(format!("ruvm-hop-{id}-{machine}-{guest:?}"));
    std::fs::create_dir_all(&dir).unwrap();
    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();

    let incoming = vec!["-incoming".to_string(), format!("tcp:127.0.0.1:{port}")];
    let mut dst = Vm::start(&dir, "dst", h, &incoming);
    let mut src = Vm::start(&dir, "src", h, &[]);
    src.wait_lines(0, 3, Duration::from_secs(120));

    src.qmp.cmd(
        r#"{"execute":"migrate-set-parameters","arguments":{"max-bandwidth":4294967296,"downtime-limit":300}}"#,
    );
    src.qmp
        .cmd(&format!(r#"{{"execute":"migrate","arguments":{{"uri":"tcp:127.0.0.1:{port}"}}}}"#));
    let start = Instant::now();
    loop {
        let r = src.qmp.cmd(r#"{"execute":"query-migrate"}"#);
        if r.contains(r#""status":"completed""#) {
            break;
        }
        assert!(!r.contains(r#""status":"failed""#), "migration failed: {r}");
        assert!(start.elapsed() < Duration::from_secs(180), "migration did not finish: {r}");
        std::thread::sleep(Duration::from_millis(100));
    }
    let r = dst.qmp.cmd(r#"{"execute":"query-status"}"#);
    assert!(r.contains(r#""status":"running""#), "{r}");
    let r = src.qmp.cmd(r#"{"execute":"query-status"}"#);
    assert!(r.contains(r#""status":"postmigrate""#), "{r}");

    dst.wait_lines(1, 3, Duration::from_secs(120));
    let (src_out, dst_out) = (src.output(), dst.output());
    let n = check(guest, &format!("{src_out}{dst_out}"));
    assert!(n >= 5, "{machine}: {n} lines");
    drop((src, dst));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "needs RUVM_MIGRATION_GUESTS, see the module doc"]
fn ruvm_to_ruvm() {
    let Some(images) = std::env::var_os("RUVM_MIGRATION_GUESTS").map(PathBuf::from) else {
        eprintln!("RUVM_MIGRATION_GUESTS is not set; skipping");
        return;
    };
    let mut ran = 0;
    let runs = [
        (Guest::Checksum, "q35"),
        (Guest::Irq, "q35"),
        (Guest::Irq, "pc-q35-10.2"),
        (Guest::Irq, "microvm"),
    ];
    for (guest, machine) in runs {
        if images.join(guest.image()).exists() {
            hop(&Hop { images: &images, guest, machine });
            ran += 1;
        }
    }
    assert!(ran > 0, "no guest image in {}", images.display());
}

#[test]
fn checker_accepts_the_closed_form() {
    let text = format!(
        "{:08x} {:08x} OK\n{:08x} {:08x} OK\npartial",
        0x4000,
        want(Guest::Checksum, 0x4000),
        0x8000,
        want(Guest::Checksum, 0x8000)
    );
    assert_eq!(check(Guest::Checksum, &text), 2);
}
