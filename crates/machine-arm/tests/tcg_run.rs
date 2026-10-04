// SPDX-License-Identifier: GPL-2.0-or-later

//! virt on TCG, running the AArch64 system tests of QEMU's tests/tcg (see `data/tcg/SOURCES`)
//! the way `make check-tcg` does: `-M virt -cpu max` with semihosting, the console of which is
//! the test output. The output and the SYS_EXIT status must be QEMU's.
//!
//! The memory test is slow: on the native backend it runs only in a release build (`cargo test
//! --release`), and on the interpreter only with `--include-ignored`. `RUVM_TEST_TIMEOUT_SECS`
//! changes the ten minute limit on each guest.

use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use ruvm_accel::tcg::{TcgOptions, ThreadMode};
use ruvm_base::ClockType;
use ruvm_hw_core::Clock;
use ruvm_hw_core::timer::TimeSource;
use ruvm_jit::native::BackendKind;
use ruvm_machine_arm::tcg_run::{VirtEvent, VirtRunConfig, VirtTcgMachine};
use ruvm_machine_arm::virt::{VirtConfig, VirtMachine};
use ruvm_target_arm::cpu::ArmCpuModel;
use ruvm_target_arm::tcg::SemihostingHost;

/// How long one guest may run, `RUVM_TEST_TIMEOUT_SECS` or ten minutes.
fn timeout() -> Duration {
    let secs = std::env::var("RUVM_TEST_TIMEOUT_SECS").ok().and_then(|s| s.parse().ok());
    Duration::from_secs(secs.unwrap_or(600))
}

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/tcg").join(name)
}

fn gunzip(name: &str) -> Vec<u8> {
    let f = std::fs::File::open(data(name)).unwrap();
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(f).read_to_end(&mut out).unwrap();
    out
}

/// How a run ended.
#[derive(Clone, Debug, PartialEq, Eq)]
enum End {
    /// SYS_EXIT with this status.
    Exit(u32),
    /// PSCI SYSTEM_OFF.
    Shutdown,
    /// PSCI SYSTEM_RESET, with `-no-reboot`.
    Reset,
    Error(String),
    Timeout,
}

#[derive(Default)]
struct Outcome {
    end: Mutex<Option<End>>,
    cv: Condvar,
}

impl Outcome {
    fn set(&self, e: End) {
        let mut g = self.end.lock().unwrap();
        if g.is_none() {
            *g = Some(e);
        }
        self.cv.notify_all();
    }

    fn wait(&self, timeout: Duration) -> End {
        let deadline = Instant::now() + timeout;
        let mut g = self.end.lock().unwrap();
        while g.is_none() {
            let now = Instant::now();
            if now >= deadline {
                return End::Timeout;
            }
            g = self.cv.wait_timeout(g, deadline - now).unwrap().0;
        }
        g.clone().unwrap()
    }
}

/// The semihosting console of the tests, `-semihosting-config chardev=output`.
#[derive(Default)]
struct Host {
    out: Mutex<Vec<u8>>,
    outcome: Outcome,
}

impl SemihostingHost for Host {
    fn console_write(&self, buf: &[u8]) -> usize {
        self.out.lock().unwrap().extend_from_slice(buf);
        buf.len()
    }

    fn console_read(&self) -> u8 {
        0
    }

    fn exit(&self, code: u32) {
        self.outcome.set(End::Exit(code));
    }

    fn heap_info(&self) -> (u64, u64) {
        (0, 0)
    }
}

/// A file under the temp directory, removed on drop.
struct TempFile(PathBuf);

impl TempFile {
    fn new(name: &str, bytes: &[u8]) -> TempFile {
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let p =
            std::env::temp_dir().join(format!("ruvm-arm-tcg-{name}-{}-{n}", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        TempFile(p)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Runs test kernel `name` on `-M virt -cpu max -smp smp` and gives the semihosting console
/// output and how it ended.
fn run(
    name: &str,
    smp: usize,
    backend: Option<BackendKind>,
    thread: Option<ThreadMode>,
) -> (Vec<u8>, End) {
    let kernel = TempFile::new(name, &gunzip(&format!("{name}.gz")));
    let clock = Clock::new(ClockType::Virtual, TimeSource::Monotonic(Instant::now()));
    let rtc_clock = Clock::new(ClockType::Host, TimeSource::Wall);
    let host = Arc::new(Host::default());
    let mut cfg = VirtConfig::new(ArmCpuModel::by_name("max").unwrap());
    cfg.smp = smp;
    cfg.kernel = Some(kernel.0.to_str().unwrap().to_string());
    cfg.semihosting = Some(host.clone());
    cfg.clock = Some(Arc::clone(&clock));
    cfg.rtc_clock = Some(Arc::clone(&rtc_clock));
    let board = VirtMachine::new(cfg).unwrap();

    let h = Arc::clone(&host);
    let handler = Arc::new(move |e: VirtEvent| match e {
        VirtEvent::Shutdown(ruvm_machine_arm::tcg_run::ShutdownReason::GuestShutdown) => {
            h.outcome.set(End::Shutdown);
        }
        VirtEvent::Shutdown(_) | VirtEvent::Reset => h.outcome.set(End::Reset),
        VirtEvent::InternalError(m) => h.outcome.set(End::Error(m)),
    });
    let cfg = VirtRunConfig {
        no_reboot: true,
        tcg: TcgOptions { thread, ..TcgOptions::default() },
        backend,
    };
    let (m, warnings) = VirtTcgMachine::new(board, vec![clock, rtc_clock], &cfg, handler).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    // Round robin unless thread=multi asks for MTTCG.
    assert_eq!(m.mttcg(), thread == Some(ThreadMode::Multi));
    assert_eq!(m.vcpu_count(), smp);
    m.start();
    let end = host.outcome.wait(timeout());
    m.pause();
    m.quit();
    let bytes = host.out.lock().unwrap().clone();
    (bytes, end)
}

fn check(name: &str, want: &[u8], backend: Option<BackendKind>, thread: Option<ThreadMode>) {
    let (got, end) = run(name, 1, backend, thread);
    let text = String::from_utf8_lossy(&got);
    assert_eq!(end, End::Exit(0), "{name}: output so far:\n{text}");
    assert!(got == want, "{name}: output differs from QEMU's:\n{text}");
}

fn out(name: &str) -> Vec<u8> {
    std::fs::read(data(&format!("{name}.out"))).unwrap()
}

#[test]
fn hello_native() {
    check("hello", b"Hello World\n", None, None);
}

#[test]
fn hello_interp_round_robin() {
    check("hello", b"Hello World\n", Some(BackendKind::Interp), Some(ThreadMode::Single));
}

#[test]
fn hello_smp2() {
    // The secondary stays off; the primary exits with the test.
    let (got, end) = run("hello", 2, None, None);
    assert_eq!((String::from_utf8(got).unwrap().as_str(), end), ("Hello World\n", End::Exit(0)));
}

#[test]
fn interrupt_native() {
    check("interrupt", b"", None, None);
}

#[test]
fn interrupt_interp() {
    check("interrupt", b"", Some(BackendKind::Interp), None);
}

#[test]
fn asid2_native() {
    check("asid2", &out("asid2"), None, None);
}

#[test]
fn feat_xs_native() {
    check("feat-xs", b"", None, None);
}

#[test]
fn semiheap_native() {
    check("semiheap", &out("semiheap"), None, None);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "slow in a debug build")]
fn memory_native() {
    check("memory", &gunzip("memory.out.gz"), None, None);
}

#[test]
#[ignore = "slow on the interpreter"]
fn memory_interp() {
    check("memory", &gunzip("memory.out.gz"), Some(BackendKind::Interp), Some(ThreadMode::Single));
}
