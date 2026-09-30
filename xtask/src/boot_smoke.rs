// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo xtask boot-smoke`: boot a real Linux kernel on `-M microvm` and `-M q35` under KVM, time
//! the boot from the serial console and hold microvm to the start latency budget of
//! spec/21-performance.md.
//!
//! The guest is a stock kernel with no initramfs and no disk. It prints its banner, fails to mount
//! a root filesystem and panics, and `panic=-1` together with `-no-reboot` turns that panic into
//! the VMM exiting. So one run covers VMM start, firmware, the kernel entry path, most of
//! `start_kernel` and teardown, without a root filesystem to build or cache.
//!
//! What it measures, all as wall time from just before the VMM is spawned:
//!
//! - first serial byte, the first thing the guest says (the bzImage decompressor, as a rule);
//! - the `Linux version` banner, which `start_kernel` prints early on;
//! - the `Kernel panic` marker, the end of the run;
//! - process exit, and the child's peak resident set (`VmHWM`, polled from `/proc`).
//!
//! # The budget
//!
//! spec/21 area 2 gives two absolute targets for the U1 microvm configuration: T1 minus T0 (first
//! guest instruction) at most 15 ms, and T3 minus T0 (first instruction of init) at most 110 ms.
//! T1 needs host tracepoints to see, which a hosted runner does not give us, so the check here is
//! the 110 ms one, applied to the `Linux version` banner. The banner comes before init, so a
//! banner later than 110 ms is a boot that cannot meet the canon target. The run here is slower
//! than U1 in known ways (a debug build, a compressed kernel, a serial console), and all of them
//! push the number up, never down, so a pass here does not prove U1 is met, but a failure is worth
//! looking at. The spec allows no override of a canon target, so there is no knob to raise it.
//!
//! # Environment
//!
//! - `RUVM_BOOT_KERNEL`: a bzImage to boot instead of the pinned download.
//! - `RUVM_BOOT_KERNEL_SHA256`: if set, the kernel above is checked against it.
//! - `RUVM_BOOT_FIRMWARE_DIR`: passed as `-L`, for when the firmware is not installed where ruvm
//!   looks by default (`/usr/share/qemu`, `/usr/share/seabios`).
//! - `RUVM_BOOT_TIMEOUT`: seconds to wait for each boot, 60 by default.
//! - `RUVM_REQUIRE_KVM=1`: fail rather than skip when `/dev/kvm` cannot be opened.
//!
//! KVM is checked first, so a host without it skips before downloading anything. Downloads and
//! results go to `target/boot-smoke/`, never into the source tree.

use std::ffi::OsString;
use std::fmt::Write as _;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

/// Alpine's `virt` kernel (Linux 6.12.31) from the 3.22.0 netboot set: a bzImage of 11 MiB, built for virtual
/// machines, so it boots fast and needs no modules to reach the root mount. Alpine keeps every
/// release directory, so the URL does not move. Bump the URL, the file name and the hash together.
const KERNEL_URL: &str =
    "https://dl-cdn.alpinelinux.org/alpine/v3.22/releases/x86_64/netboot-3.22.0/vmlinuz-virt";
const KERNEL_FILE: &str = "alpine-3.22.0-vmlinuz-virt";
const KERNEL_SHA256: &str = "85d02ea8180af608c47640e1a34c0c87a3c21baf9614c878a74b3f1b225cc5fb";

/// Kernel command line. There is no initramfs and no disk, so the kernel panics at the root mount,
/// `panic=-1` reboots straight away and `-no-reboot` turns the reboot into an exit.
const APPEND: &str = "console=ttyS0 earlyprintk=serial rdinit=/nonexistent panic=-1 reboot=t";

/// spec/21 area 2, U1: T3 minus T0 at most 110 ms. See the module docs for why it is applied to the
/// banner.
const BUDGET: Duration = Duration::from_millis(110);

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// How long the VMM gets to exit on its own after the panic marker before it is killed.
const EXIT_GRACE: Duration = Duration::from_secs(5);

const FIRST_BYTE: &str = "first serial byte";
const BANNER: &str = "Linux version";
const PANIC: &str = "Kernel panic";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Machine {
    Microvm,
    Q35,
}

impl Machine {
    fn name(self) -> &'static str {
        match self {
            Machine::Microvm => "microvm",
            Machine::Q35 => "q35",
        }
    }
}

/// Something the guest said, in the order it is expected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Marker {
    FirstByte,
    Banner,
    Panic,
}

/// One boot, as measured.
#[derive(Debug, Default)]
struct Run {
    machine: &'static str,
    first_byte: Option<Duration>,
    banner: Option<Duration>,
    panic: Option<Duration>,
    total: Duration,
    max_rss_kib: Option<u64>,
    /// How the run ended: `exited`, `killed after the panic marker`, `timed out` and so on.
    ending: String,
}

pub(crate) fn run(root: &Path) -> Result<(), String> {
    if let Err(why) = kvm_usable() {
        if require_kvm() {
            return Err(format!("RUVM_REQUIRE_KVM is set, but {why}"));
        }
        println!("skipping the boot smoke test: {why}");
        println!("set RUVM_REQUIRE_KVM=1 to make this an error");
        return Ok(());
    }

    let target = target_dir(root);
    let dir = target.join("boot-smoke");
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    let kernel = kernel(&dir)?;
    let firmware = std::env::var_os("RUVM_BOOT_FIRMWARE_DIR").map(PathBuf::from);
    let timeout = match std::env::var("RUVM_BOOT_TIMEOUT") {
        Ok(s) => Duration::from_secs(
            s.parse().map_err(|_| format!("RUVM_BOOT_TIMEOUT is not a number of seconds: {s}"))?,
        ),
        Err(_) => DEFAULT_TIMEOUT,
    };

    crate::cargo(&["build", "-p", "ruvm-cli", "--bin", "ruvm"])?;
    let ruvm = target.join("debug").join(format!("ruvm{}", std::env::consts::EXE_SUFFIX));

    let mut runs = Vec::new();
    for machine in [Machine::Microvm, Machine::Q35] {
        let args = qemu_args(machine, &kernel, firmware.as_deref());
        println!("{} {}", ruvm.display(), shown(&args));
        runs.push(boot(&ruvm, &args, machine, &dir, timeout)?);
    }

    print!("{}", table(&runs));
    let json = summary(&runs, &kernel);
    let out = dir.join("results.json");
    let text = serde_json::to_string_pretty(&json).map_err(|e| e.to_string())? + "\n";
    std::fs::write(&out, text).map_err(|e| format!("could not write {}: {e}", out.display()))?;
    println!("results in {}", out.display());

    let problems: Vec<String> = runs.iter().flat_map(|r| check(r, BUDGET)).collect();
    if problems.is_empty() {
        println!("both machines booted, and microvm reached the banner within {BUDGET:?}");
        Ok(())
    } else {
        for p in &problems {
            eprintln!("  {p}");
        }
        Err(format!("{} boot smoke problems; serial logs are in {}", problems.len(), dir.display()))
    }
}

fn require_kvm() -> bool {
    std::env::var("RUVM_REQUIRE_KVM").is_ok_and(|v| !v.is_empty() && v != "0")
}

/// Whether this host can run the test: Linux on x86-64 with a `/dev/kvm` we can open.
fn kvm_usable() -> Result<(), String> {
    if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        return Err(format!(
            "it needs Linux on x86-64 with KVM, and this is {} on {}",
            std::env::consts::OS,
            std::env::consts::ARCH
        ));
    }
    std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/kvm")
        .map(drop)
        .map_err(|e| format!("/dev/kvm cannot be opened: {e}"))
}

fn target_dir(root: &Path) -> PathBuf {
    match std::env::var_os("CARGO_TARGET_DIR") {
        Some(dir) => root.join(dir),
        None => root.join("target"),
    }
}

/// The kernel to boot: `RUVM_BOOT_KERNEL`, or the pinned download, fetched once into `dir`.
fn kernel(dir: &Path) -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("RUVM_BOOT_KERNEL") {
        let path = PathBuf::from(path);
        if let Ok(want) = std::env::var("RUVM_BOOT_KERNEL_SHA256") {
            verify(&path, &want)?;
        }
        return Ok(path);
    }
    let path = dir.join(KERNEL_FILE);
    if path.exists() {
        match verify(&path, KERNEL_SHA256) {
            Ok(()) => return Ok(path),
            Err(e) => {
                eprintln!("{e}; downloading it again");
                let _ = std::fs::remove_file(&path);
            }
        }
    }
    let part = dir.join(format!("{KERNEL_FILE}.part"));
    println!("downloading {KERNEL_URL}");
    let status = Command::new("curl")
        .args(["-fsSL", "--retry", "3", "-o"])
        .arg(&part)
        .arg(KERNEL_URL)
        .status()
        .map_err(|e| format!("could not run curl: {e}"))?;
    if !status.success() {
        let _ = std::fs::remove_file(&part);
        return Err(format!("curl could not download {KERNEL_URL}"));
    }
    if let Err(e) = verify(&part, KERNEL_SHA256) {
        let _ = std::fs::remove_file(&part);
        return Err(e);
    }
    std::fs::rename(&part, &path)
        .map_err(|e| format!("could not rename {}: {e}", part.display()))?;
    Ok(path)
}

fn verify(path: &Path, want: &str) -> Result<(), String> {
    let mut file =
        std::fs::File::open(path).map_err(|e| format!("could not open {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;
    check_sha256(&hex(&hasher.finalize()), want).map_err(|e| format!("{}: {e}", path.display()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// Compares two hex digests, ignoring case and surrounding space.
fn check_sha256(got: &str, want: &str) -> Result<(), String> {
    if got.trim().eq_ignore_ascii_case(want.trim()) {
        Ok(())
    } else {
        Err(format!("sha256 is {got}, expected {}", want.trim()))
    }
}

/// The command line after the binary name. The ruvm multi-call binary picks its personality from
/// the first argument.
fn qemu_args(machine: Machine, kernel: &Path, firmware: Option<&Path>) -> Vec<OsString> {
    let mut args: Vec<OsString> = [
        "qemu-system-x86_64",
        "-M",
        machine.name(),
        "-accel",
        "kvm",
        "-m",
        "256M",
        "-smp",
        "1",
        "-nodefaults",
        "-no-reboot",
        "-display",
        "none",
        "-serial",
        "stdio",
        "-kernel",
    ]
    .iter()
    .map(OsString::from)
    .collect();
    args.push(kernel.into());
    args.push("-append".into());
    args.push(APPEND.into());
    if let Some(dir) = firmware {
        args.push("-L".into());
        args.push(dir.into());
    }
    args
}

/// The arguments as they would be typed, for the log.
fn shown(args: &[OsString]) -> String {
    let words: Vec<String> = args
        .iter()
        .map(|a| {
            let a = a.to_string_lossy();
            if a.contains(' ') { format!("\"{a}\"") } else { a.into_owned() }
        })
        .collect();
    words.join(" ")
}

/// Finds the markers in serial output as it arrives in chunks, including a marker split across
/// two chunks. Each marker is reported once.
#[derive(Default)]
struct Scanner {
    seen: Vec<u8>,
    found: Vec<Marker>,
}

impl Scanner {
    fn feed(&mut self, chunk: &[u8]) -> Vec<Marker> {
        let mut new = Vec::new();
        if chunk.is_empty() {
            return new;
        }
        let longest = BANNER.len().max(PANIC.len());
        let from = self.seen.len().saturating_sub(longest - 1);
        self.seen.extend_from_slice(chunk);
        let window = &self.seen[from..];
        for (marker, text) in [
            (Marker::FirstByte, None),
            (Marker::Banner, Some(BANNER)),
            (Marker::Panic, Some(PANIC)),
        ] {
            if self.found.contains(&marker) {
                continue;
            }
            let hit = match text {
                None => true,
                Some(t) => window.windows(t.len()).any(|w| w == t.as_bytes()),
            };
            if hit {
                self.found.push(marker);
                new.push(marker);
            }
        }
        new
    }
}

/// What the stdout reader tells the main thread.
enum Seen {
    Marker(Marker, Instant),
    /// End of output, with everything that was read.
    Closed(Vec<u8>),
}

fn boot(
    ruvm: &Path,
    args: &[OsString],
    machine: Machine,
    dir: &Path,
    timeout: Duration,
) -> Result<Run, String> {
    let stderr_path = dir.join(format!("{}.stderr", machine.name()));
    let stderr = std::fs::File::create(&stderr_path)
        .map_err(|e| format!("could not create {}: {e}", stderr_path.display()))?;
    let start = Instant::now();
    let mut child = Command::new(ruvm)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(stderr)
        .spawn()
        .map_err(|e| format!("could not run {}: {e}", ruvm.display()))?;
    let pid = child.id();
    let mut stdout = child.stdout.take().ok_or("no stdout from the child")?;
    let (tx, rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut scanner = Scanner::default();
        let mut buf = [0u8; 4096];
        loop {
            match stdout.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let now = Instant::now();
                    for m in scanner.feed(&buf[..n]) {
                        let _ = tx.send(Seen::Marker(m, now));
                    }
                }
            }
        }
        let _ = tx.send(Seen::Closed(scanner.seen));
    });

    let mut run = Run { machine: machine.name(), ..Run::default() };
    let mut log = Vec::new();
    let mut exited = None;
    let mut deadline = start + timeout;
    loop {
        match rx.recv_timeout(Duration::from_millis(10)) {
            Ok(Seen::Marker(m, at)) => {
                let t = at.duration_since(start);
                match m {
                    Marker::FirstByte => run.first_byte = Some(t),
                    Marker::Banner => run.banner = Some(t),
                    Marker::Panic => {
                        run.panic = Some(t);
                        deadline = deadline.min(Instant::now() + EXIT_GRACE);
                    }
                }
            }
            Ok(Seen::Closed(bytes)) => log = bytes,
            Err(mpsc::RecvTimeoutError::Timeout | mpsc::RecvTimeoutError::Disconnected) => {}
        }
        if let Some(kib) = vm_hwm(pid) {
            run.max_rss_kib = Some(run.max_rss_kib.map_or(kib, |m| m.max(kib)));
        }
        match child.try_wait() {
            Ok(Some(status)) => {
                exited = Some((status, Instant::now()));
                break;
            }
            Ok(None) => {}
            Err(e) => return Err(format!("could not wait for ruvm: {e}")),
        }
        if Instant::now() >= deadline {
            break;
        }
    }

    match exited {
        Some((status, at)) => {
            run.total = at.duration_since(start);
            run.ending = format!("exited, {status}");
        }
        None => {
            let _ = child.kill();
            let _ = child.wait();
            run.total = start.elapsed();
            run.ending = if run.panic.is_some() {
                "killed, it did not exit after the panic".into()
            } else {
                format!("killed after the {}s timeout", timeout.as_secs())
            };
        }
    }
    drop(child);
    let _ = reader.join();
    while let Ok(seen) = rx.try_recv() {
        if let Seen::Closed(bytes) = seen {
            log = bytes;
        }
    }
    let log_path = dir.join(format!("{}.log", machine.name()));
    std::fs::write(&log_path, &log)
        .map_err(|e| format!("could not write {}: {e}", log_path.display()))?;
    Ok(run)
}

/// The child's peak resident set in KiB, from `/proc/<pid>/status`. Nothing on hosts without
/// `/proc`, or once the process is gone.
fn vm_hwm(pid: u32) -> Option<u64> {
    parse_vm_hwm(&std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?)
}

fn parse_vm_hwm(status: &str) -> Option<u64> {
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let mut words = line["VmHWM:".len()..].split_whitespace();
    let n: u64 = words.next()?.parse().ok()?;
    match words.next() {
        Some("kB") | None => Some(n),
        Some(_) => None,
    }
}

/// What is wrong with a run: a marker never seen, or microvm over the budget.
fn check(run: &Run, budget: Duration) -> Vec<String> {
    let mut problems = Vec::new();
    for (what, at) in [(FIRST_BYTE, run.first_byte), (BANNER, run.banner), (PANIC, run.panic)] {
        if at.is_none() {
            problems.push(format!("{}: no {what} ({})", run.machine, run.ending));
        }
    }
    if run.machine == Machine::Microvm.name() {
        if let Some(banner) = run.banner {
            if banner > budget {
                problems.push(format!(
                    "microvm: \"{BANNER}\" after {}, over the {} budget of spec/21 area 2 (U1, T3 minus T0)",
                    ms(Some(banner)),
                    ms(Some(budget))
                ));
            }
        }
    }
    problems
}

fn ms(d: Option<Duration>) -> String {
    match d {
        Some(d) => format!("{:.1} ms", d.as_secs_f64() * 1000.0),
        None => "-".into(),
    }
}

fn millis(d: Option<Duration>) -> serde_json::Value {
    match d {
        Some(d) => serde_json::json!((d.as_secs_f64() * 1e6).round() / 1e3),
        None => serde_json::Value::Null,
    }
}

fn table(runs: &[Run]) -> String {
    let mut s = format!(
        "{:<8} {:>12} {:>12} {:>12} {:>12} {:>12}  {}\n",
        "machine", "first byte", "banner", "panic", "total", "max rss", "ending"
    );
    for r in runs {
        let rss = r.max_rss_kib.map_or_else(|| "-".into(), |k| format!("{} MiB", k / 1024));
        let _ = writeln!(
            s,
            "{:<8} {:>12} {:>12} {:>12} {:>12} {:>12}  {}",
            r.machine,
            ms(r.first_byte),
            ms(r.banner),
            ms(r.panic),
            ms(Some(r.total)),
            rss,
            r.ending
        );
    }
    s
}

fn summary(runs: &[Run], kernel: &Path) -> serde_json::Value {
    let runs: Vec<serde_json::Value> = runs
        .iter()
        .map(|r| {
            serde_json::json!({
                "machine": r.machine,
                "first_serial_byte_ms": millis(r.first_byte),
                "linux_version_ms": millis(r.banner),
                "kernel_panic_ms": millis(r.panic),
                "total_ms": millis(Some(r.total)),
                "max_rss_kib": r.max_rss_kib,
                "ending": r.ending,
            })
        })
        .collect();
    serde_json::json!({
        "kernel": kernel.display().to_string(),
        "append": APPEND,
        "budget": {
            "machine": Machine::Microvm.name(),
            "metric": "linux_version_ms",
            "limit_ms": millis(Some(BUDGET)),
            "source": "spec/21-performance.md, area 2, U1: T3 minus T0 at most 110 ms",
        },
        "runs": runs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed_all(chunks: &[&str]) -> Vec<Marker> {
        let mut s = Scanner::default();
        chunks.iter().flat_map(|c| s.feed(c.as_bytes())).collect()
    }

    #[test]
    fn markers_come_once_and_in_order() {
        let got = feed_all(&[
            "Decompressing Linux... ",
            "Linux version 6.12.1 (builder)\n",
            "Linux version again\n",
            "Kernel panic - not syncing: VFS: Unable to mount root fs\n",
            "Kernel panic again\n",
        ]);
        assert_eq!(got, [Marker::FirstByte, Marker::Banner, Marker::Panic]);
    }

    #[test]
    fn markers_split_across_chunks_are_found() {
        let got = feed_all(&["x", "Linux ver", "sion 6", "\nKer", "n", "el pan", "ic"]);
        assert_eq!(got, [Marker::FirstByte, Marker::Banner, Marker::Panic]);
    }

    #[test]
    fn empty_chunks_are_not_a_first_byte() {
        assert!(feed_all(&["", ""]).is_empty());
        assert_eq!(feed_all(&["", "Kernel panic"]), [Marker::FirstByte, Marker::Panic]);
    }

    #[test]
    fn sha256_of_known_input() {
        assert_eq!(
            hex(&Sha256::digest(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha256_comparison() {
        let h = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert!(check_sha256(h, h).is_ok());
        assert!(check_sha256(h, &format!(" {} \n", h.to_uppercase())).is_ok());
        let e = check_sha256(h, "00").unwrap_err();
        assert!(e.contains("expected 00"), "{e}");
    }

    #[test]
    fn verify_reads_the_file() {
        let path = std::env::temp_dir().join(format!("xtask-boot-smoke-{}", std::process::id()));
        std::fs::write(&path, b"abc").unwrap();
        let good = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        let ok = verify(&path, good);
        let bad = verify(&path, &"0".repeat(64));
        std::fs::remove_file(&path).unwrap();
        assert!(ok.is_ok());
        assert!(bad.is_err());
    }

    #[test]
    fn pinned_hash_is_a_sha256() {
        assert_eq!(KERNEL_SHA256.len(), 64);
        assert!(KERNEL_SHA256.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
    }

    #[test]
    fn command_lines() {
        let k = Path::new("/k/bzImage");
        let args = qemu_args(Machine::Microvm, k, None);
        assert_eq!(
            shown(&args),
            "qemu-system-x86_64 -M microvm -accel kvm -m 256M -smp 1 -nodefaults -no-reboot \
             -display none -serial stdio -kernel /k/bzImage -append \"console=ttyS0 \
             earlyprintk=serial rdinit=/nonexistent panic=-1 reboot=t\""
        );
        let args = qemu_args(Machine::Q35, k, Some(Path::new("/fw")));
        assert_eq!(args[2], "q35");
        assert_eq!(args[args.len() - 2..], [OsString::from("-L"), OsString::from("/fw")]);
    }

    #[test]
    fn vm_hwm_parsing() {
        let status = "Name:\truvm\nVmPeak:\t  300000 kB\nVmHWM:\t   51234 kB\nVmRSS:\t 40000 kB\n";
        assert_eq!(parse_vm_hwm(status), Some(51234));
        assert_eq!(parse_vm_hwm("Name:\truvm\n"), None);
        assert_eq!(parse_vm_hwm("VmHWM:\t12 MB\n"), None);
    }

    fn run(machine: Machine, banner_ms: Option<u64>) -> Run {
        Run {
            machine: machine.name(),
            first_byte: Some(Duration::from_millis(5)),
            banner: banner_ms.map(Duration::from_millis),
            panic: Some(Duration::from_millis(900)),
            total: Duration::from_millis(950),
            max_rss_kib: Some(1024),
            ending: "exited".into(),
        }
    }

    #[test]
    fn budget_applies_to_microvm_only() {
        assert!(check(&run(Machine::Microvm, Some(110)), BUDGET).is_empty());
        let over = check(&run(Machine::Microvm, Some(111)), BUDGET);
        assert_eq!(over.len(), 1);
        assert!(over[0].contains("over the 110.0 ms budget"), "{over:?}");
        assert!(check(&run(Machine::Q35, Some(5000)), BUDGET).is_empty());
    }

    #[test]
    fn a_missing_marker_fails() {
        let problems = check(&run(Machine::Q35, None), BUDGET);
        assert_eq!(problems, ["q35: no Linux version (exited)"]);
    }

    #[test]
    fn table_and_summary_have_every_run() {
        let runs = [run(Machine::Microvm, Some(80)), run(Machine::Q35, None)];
        let t = table(&runs);
        assert_eq!(t.lines().count(), 3);
        assert!(t.contains("80.0 ms"));
        let j = summary(&runs, Path::new("/k"));
        assert_eq!(j["runs"][0]["linux_version_ms"], 80.0);
        assert!(j["runs"][1]["linux_version_ms"].is_null());
        assert_eq!(j["budget"]["limit_ms"], 110.0);
    }
}
