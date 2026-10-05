// SPDX-License-Identifier: GPL-2.0-or-later

//! virt on TCG, running the AArch64 system tests of QEMU's tests/tcg (see `data/tcg/SOURCES`)
//! the way `make check-tcg` does: `-M virt -cpu max` with semihosting, the console of which is
//! the test output. The output and the SYS_EXIT status must be QEMU's.
//!
//! The memory test is slow: on the native backend it runs only in a release build (`cargo test
//! --release`), and on the interpreter only with `--include-ignored`. `RUVM_TEST_TIMEOUT_SECS`
//! changes the ten minute limit on each guest.

use std::collections::VecDeque;
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
use ruvm_target_arm::cpu::{ArmCpuModel, PauthAlg};
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
    /// What SYS_READC reads, the chardev input; 0 once it runs out.
    input: Mutex<VecDeque<u8>>,
    /// `-semihosting-config arg=`.
    args: Option<String>,
    outcome: Outcome,
}

impl SemihostingHost for Host {
    fn console_write(&self, buf: &[u8]) -> usize {
        self.out.lock().unwrap().extend_from_slice(buf);
        buf.len()
    }

    fn console_read(&self) -> u8 {
        self.input.lock().unwrap().pop_front().unwrap_or(0)
    }

    fn cmdline(&self) -> Option<String> {
        self.args.clone()
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
    run_with(name, smp, backend, thread, Host::default(), |_| {})
}

/// [`run`] with the semihosting host `host`, and `setup` changing the board configuration
/// (CPU model and machine properties).
fn run_with(
    name: &str,
    smp: usize,
    backend: Option<BackendKind>,
    thread: Option<ThreadMode>,
    host: Host,
    setup: impl FnOnce(&mut VirtConfig),
) -> (Vec<u8>, End) {
    let kernel = TempFile::new(name, &gunzip(&format!("{name}.gz")));
    let clock = Clock::new(ClockType::Virtual, TimeSource::Monotonic(Instant::now()));
    let rtc_clock = Clock::new(ClockType::Host, TimeSource::Wall);
    let host = Arc::new(host);
    let mut cfg = VirtConfig::new(ArmCpuModel::by_name("max").unwrap());
    setup(&mut cfg);
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
    // MTTCG unless thread=single asks for round robin, as AArch64 supports MTTCG.
    assert_eq!(m.mttcg(), thread != Some(ThreadMode::Single));
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

#[test]
fn pauth_3_qarma5() {
    // -cpu max,pauth-qarma5=on
    let (got, end) = run_with("pauth-3", 1, None, None, Host::default(), |c| {
        c.cpu = c.cpu.clone().with_pauth(Some(PauthAlg::Qarma5));
    });
    assert_eq!((String::from_utf8_lossy(&got).as_ref(), end), ("OK\n", End::Exit(0)));
}

#[test]
fn pauth_3_default() {
    // -cpu max uses the IMPDEF algorithm, so the architected PAC the test wants is not there:
    // QEMU fails the same way.
    let (got, end) = run("pauth-3", 1, None, None);
    assert_eq!(
        (String::from_utf8_lossy(&got).as_ref(), end),
        ("FAIL: 3f52a54200000000 != c003b93900000000\n", End::Exit(1))
    );
}

#[test]
fn semiconsole_native() {
    // SYS_READC reads the chardev: what make check-tcg's run-semiconsole types.
    let host =
        Host { input: Mutex::new(b"hello worldX".iter().copied().collect()), ..Host::default() };
    let (got, end) = run_with("semiconsole", 1, None, None, host, |_| {});
    assert_eq!(
        (String::from_utf8_lossy(&got).as_ref(), end),
        ("Semihosting Console Test\nhit X to exit:hello worldX", End::Exit(0))
    );
}

/// `mte_page` in the test's kernel.ld, the exit code of `mte`.
const MTE_PAGE: u32 = 0x4040_0000;

#[test]
fn mte_native() {
    // -M virt,mte=on -cpu max: IRG, STG and a tag checked store that must not fault. main
    // returns the address of the tagged page, which boot.S passes to SYS_EXIT; QEMU exits
    // with it, and the shell sees its low byte, 0.
    let (got, end) = run_with("mte", 1, None, None, Host::default(), |cfg| cfg.mte = true);
    assert_eq!((String::from_utf8_lossy(&got).as_ref(), end), ("", End::Exit(MTE_PAGE)));
}

#[test]
fn mte_interp() {
    let (got, end) =
        run_with("mte", 1, Some(BackendKind::Interp), None, Host::default(), |cfg| cfg.mte = true);
    assert_eq!((String::from_utf8_lossy(&got).as_ref(), end), ("", End::Exit(MTE_PAGE)));
}

#[test]
fn rme_gdi_native() {
    // -cpu max has no FEAT_RME_GDI: QEMU skips the same way.
    let (got, end) = run("rme_gdi", 1, None, None);
    assert_eq!(
        (String::from_utf8_lossy(&got).as_ref(), end),
        ("SKIP: GDI not implemented\n", End::Exit(0))
    );
}

#[test]
#[cfg_attr(debug_assertions, ignore = "slow in a debug build")]
fn memory_sve_native() {
    // memory.c built with -march=armv8.1-a+sve -O3: the same output as the plain build.
    check("memory-sve", &gunzip("memory.out.gz"), None, None);
}

#[test]
fn vtimer_native() {
    // make check-tcg runs it on virt,virtualization=on,gic-version=2 with four cortex-a57s
    // and `-semihosting-config arg=2`; QEMU prints the same with gic-version=3.
    let host = Host { args: Some("2".to_string()), ..Host::default() };
    let (got, end) = run_with("vtimer", 4, None, None, host, |cfg| {
        cfg.cpu = ArmCpuModel::by_name("cortex-a57").unwrap();
        cfg.virtualization = true;
    });
    let text = String::from_utf8_lossy(&got);
    assert_eq!(end, End::Exit(0), "output so far:\n{text}");
    assert_eq!(text, String::from_utf8(out("vtimer")).unwrap());
}

#[test]
fn gpc_test_native() {
    // make check-tcg runs it on virt,virtualization=on,secure=on,gic-version=3 with
    // `-cpu max,x-rme=on` and `-semihosting-config arg=3`, so that it boots at EL3.
    let host = Host { args: Some("3".to_string()), ..Host::default() };
    let (got, end) = run_with("gpc-test", 1, None, None, host, |cfg| {
        cfg.cpu = ArmCpuModel::by_name("max").unwrap().with_rme(true);
        cfg.virtualization = true;
        cfg.secure = true;
    });
    let text = String::from_utf8_lossy(&got);
    assert_eq!(end, End::Exit(0), "output so far:\n{text}");
    assert_eq!(text, String::from_utf8(out("gpc-test")).unwrap());
}
