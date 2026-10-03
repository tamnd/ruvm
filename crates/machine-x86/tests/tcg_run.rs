// SPDX-License-Identifier: GPL-2.0-or-later

//! q35 on TCG, running the x86_64 system tests of QEMU's tests/tcg (`hello`, `memory` and
//! `interrupt`, see `data/tcg/SOURCES`) through SeaBIOS and the PVH option ROM, as QEMU's
//! `make check-tcg` does. The debugcon output and exit status must be QEMU's.
//!
//! The firmware comes from `RUVM_TEST_FIRMWARE_DIR`, taken as a `-L` directory, and then the
//! usual search path (`/usr/share/qemu` and friends, or the `share/qemu` directory of a
//! `qemu-system-x86_64` on `PATH`). Without `bios-256k.bin` and `pvh.bin` there the tests say
//! so and pass without running.
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
use ruvm_hw_misc::debugexit::{IsaDebugExit, IsaDebugExitConfig};
use ruvm_jit::native::BackendKind;
use ruvm_machine_x86::FirmwareSearch;
use ruvm_machine_x86::board::{BoardKind, BoardSpec, KernelFiles, build_board};
use ruvm_machine_x86::debugcon::{DebugconConfig, IsaDebugcon};
use ruvm_machine_x86::run_event::GuestEvent;
use ruvm_machine_x86::tcg_run::{TCG_SMM_AVAILABLE, TcgCpuModel, TcgMachine, TcgRunConfig};

/// How long one guest may run, `RUVM_TEST_TIMEOUT_SECS` or ten minutes. The memory test in a
/// debug build on the interpreter is the slowest, at about four minutes on an idle machine.
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

/// The firmware, or `None` (with a note) when it is not installed.
fn firmware() -> Option<FirmwareSearch> {
    let dirs: Vec<PathBuf> =
        std::env::var_os("RUVM_TEST_FIRMWARE_DIR").map(PathBuf::from).into_iter().collect();
    let fw = FirmwareSearch::new(&dirs);
    for f in ["bios-256k.bin", "pvh.bin"] {
        if fw.find(f).is_none() {
            eprintln!("skipped: {f} is not in the firmware search path");
            return None;
        }
    }
    Some(fw)
}

/// How a run ended.
#[derive(Clone, Debug, PartialEq, Eq)]
enum End {
    /// ACPI power off, exit status 0.
    Shutdown,
    /// `isa-debug-exit`, with the exit status.
    DebugExit(u64),
    /// A reset with `-no-reboot`, such as a triple fault.
    Reset,
    /// A vCPU failed.
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

/// A file under the temp directory, removed on drop.
struct TempFile(PathBuf);

impl TempFile {
    fn new(name: &str, bytes: &[u8]) -> TempFile {
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("ruvm-tcg-{name}-{}-{n}", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        TempFile(p)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Runs test kernel `name` on q35 the way tests/tcg/x86_64/Makefile.softmmu-target does and
/// gives the debugcon output and how it ended.
fn run(
    fw: FirmwareSearch,
    name: &str,
    backend: Option<BackendKind>,
    thread: Option<ThreadMode>,
) -> (Vec<u8>, End) {
    let kernel = TempFile::new(name, &gunzip(&format!("{name}.gz")));
    let cpu = TcgCpuModel::new(None).unwrap();
    let clock = Clock::new(ClockType::Virtual, TimeSource::Monotonic(Instant::now()));
    let rtc_clock = Clock::new(ClockType::Host, TimeSource::Wall);
    let spec = BoardSpec {
        kind: BoardKind::Q35,
        props: Vec::new(),
        ram_size: None,
        cpus: 1,
        max_cpus: 1,
        kvm: false,
        pit_in_kernel: false,
        smm_available: TCG_SMM_AVAILABLE,
        phys_bits: cpu.phys_bits(),
        cpu: cpu.ident(),
        bios: None,
        pflash: [None, None],
        uuid: None,
        smbios: Default::default(),
        topology: None,
        kernel: Some(KernelFiles {
            kernel: kernel.0.to_str().unwrap().to_string(),
            initrd: None,
            append: String::new(),
        }),
        firmware: fw,
        serial_hds: Vec::new(),
        clock: Arc::clone(&clock),
        rtc_clock: Arc::clone(&rtc_clock),
    };
    let (board, _) = build_board(spec).unwrap();

    let out = Arc::new(Mutex::new(Vec::new()));
    let outcome = Arc::new(Outcome::default());
    let io_root = board.io_as().root();
    let o = Arc::clone(&out);
    IsaDebugcon::realize(
        board.memory_system(),
        io_root,
        DebugconConfig::default(),
        Arc::new(move |b: &[u8]| o.lock().unwrap().extend_from_slice(b)),
    )
    .unwrap();
    if std::env::var_os("RUVM_TCG_TRACE").is_some() {
        IsaDebugcon::realize(
            board.memory_system(),
            io_root,
            DebugconConfig { iobase: 0x402, ..DebugconConfig::default() },
            Arc::new(|b: &[u8]| eprint!("{}", String::from_utf8_lossy(b))),
        )
        .unwrap();
    }
    let oc = Arc::clone(&outcome);
    IsaDebugExit::realize(
        board.memory_system(),
        io_root,
        IsaDebugExitConfig { iobase: 0xf4, iosize: 4 },
        Arc::new(move |code| oc.set(End::DebugExit(code))),
    )
    .unwrap();

    let oc = Arc::clone(&outcome);
    let handler = Arc::new(move |e: GuestEvent| match e {
        GuestEvent::Shutdown(ruvm_machine_x86::run_event::ShutdownReason::GuestShutdown) => {
            oc.set(End::Shutdown);
        }
        GuestEvent::Shutdown(_) | GuestEvent::Reset => oc.set(End::Reset),
        GuestEvent::Panicked => oc.set(End::Error("panicked".into())),
        GuestEvent::InternalError(m) => oc.set(End::Error(m)),
    });
    let cfg = TcgRunConfig {
        no_reboot: true,
        tcg: TcgOptions { thread, ..TcgOptions::default() },
        backend,
    };
    let (m, warnings) =
        TcgMachine::new(board, &cpu, vec![clock, rtc_clock], &cfg, handler).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(m.mttcg(), thread != Some(ThreadMode::Single));
    m.start();
    if std::env::var_os("RUVM_TCG_TRACE").is_some() {
        for _ in 0..10 {
            if outcome.end.lock().unwrap().is_some() {
                break;
            }
            std::thread::sleep(Duration::from_secs(2));
            m.vcpus().run_on_each(|cpu| {
                let ops = cpu.ops();
                eprintln!(
                    "cpu {} pc {:#x} halted {} req {:#x}",
                    cpu.shared().cpu_index,
                    ops.get_pc(cpu),
                    cpu.shared().halted.load(Ordering::Relaxed),
                    cpu.shared().interrupt_request()
                );
            });
        }
    }
    let end = outcome.wait(timeout());
    m.pause();
    m.quit();
    let bytes = out.lock().unwrap().clone();
    (bytes, end)
}

fn check(name: &str, want: &[u8], backend: Option<BackendKind>, thread: Option<ThreadMode>) {
    let Some(fw) = firmware() else { return };
    let (got, end) = run(fw, name, backend, thread);
    let text = String::from_utf8_lossy(&got);
    assert_eq!(end, End::Shutdown, "{name}: output so far:\n{text}");
    assert!(got == want, "{name}: output differs from QEMU's:\n{text}");
}

fn memory_out() -> Vec<u8> {
    gunzip("memory.out.gz")
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
fn interrupt_native() {
    check("interrupt", b"", None, None);
}

#[test]
fn interrupt_interp() {
    check("interrupt", b"", Some(BackendKind::Interp), None);
}

/// Half a minute in a debug build on an idle machine, but far longer on a loaded one, so a
/// debug build runs it only with `--ignored` (or `--include-ignored`).
#[test]
#[cfg_attr(debug_assertions, ignore = "slow in a debug build")]
fn memory_native() {
    check("memory", &memory_out(), None, None);
}

/// About four minutes in a debug build on an idle machine and longer than the default limit
/// even in a release build on a loaded one, so only with `--ignored` (or `--include-ignored`).
#[test]
#[ignore = "slow on the interpreter"]
fn memory_interp() {
    check("memory", &memory_out(), Some(BackendKind::Interp), Some(ThreadMode::Single));
}
