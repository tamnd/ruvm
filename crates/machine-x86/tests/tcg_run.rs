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
//!
//! `linux_smp1` and `linux_smp2` boot a Linux kernel and initramfs to a shell on the serial
//! console. They are ignored, since the images are not in the tree: point
//! `RUVM_TEST_LINUX_KERNEL` and `RUVM_TEST_LINUX_INITRD` at them and run with `--ignored`.

use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ruvm_accel::tcg::{TcgOptions, ThreadMode};
use ruvm_base::ClockType;
use ruvm_hw_char::serial::{Serial, SerialBackend};
use ruvm_hw_core::Clock;
use ruvm_hw_core::timer::TimeSource;
use ruvm_hw_misc::debugexit::{IsaDebugExit, IsaDebugExitConfig};
use ruvm_hw_timer::mc146818::gmtime;
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
        machine_type: "pc-q35-11.1",
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
            dump_vcpus(&m);
        }
    }
    let end = outcome.wait(timeout());
    m.pause();
    m.quit();
    let bytes = out.lock().unwrap().clone();
    (bytes, end)
}

/// Prints where each vCPU is, for `RUVM_TCG_TRACE`.
fn dump_vcpus(m: &TcgMachine) {
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

/// With a thread per vCPU, as `thread=multi` asks, which is also the default.
#[test]
fn interrupt_native() {
    check("interrupt", b"", None, Some(ThreadMode::Multi));
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

/// The serial console of a Linux guest: what it printed so far.
#[derive(Default)]
struct Console {
    out: Mutex<Vec<u8>>,
    cv: Condvar,
}

impl SerialBackend for Console {
    fn write(&self, bytes: &[u8]) -> usize {
        if std::env::var_os("RUVM_TCG_TRACE").is_some() {
            eprint!("{}", String::from_utf8_lossy(bytes));
        }
        self.out.lock().unwrap().extend_from_slice(bytes);
        self.cv.notify_all();
        bytes.len()
    }
}

impl Console {
    /// Waits until the output after `from` contains `what` and gives the offset after it.
    fn wait_for(&self, from: usize, what: &str, until: Instant) -> Option<usize> {
        let mut g = self.out.lock().unwrap();
        loop {
            let tail = &g[from.min(g.len())..];
            if let Some(i) = tail.windows(what.len()).position(|w| w == what.as_bytes()) {
                return Some(from + i + what.len());
            }
            let now = Instant::now();
            if now >= until {
                return None;
            }
            g = self.cv.wait_timeout(g, until - now).unwrap().0;
        }
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.out.lock().unwrap()).into_owned()
    }
}

/// Types `line` into the UART as fast as its receive FIFO takes it.
fn type_line(serial: &Serial, line: &str, until: Instant) {
    for b in line.bytes() {
        while serial.can_receive() == 0 {
            assert!(Instant::now() < until, "the guest stopped reading the console");
            std::thread::sleep(Duration::from_millis(5));
        }
        serial.receive(&[b]);
    }
}

/// Boots the kernel `RUVM_TEST_LINUX_KERNEL` with the initramfs `RUVM_TEST_LINUX_INITRD` on
/// q35 with SeaBIOS and `cpus` vCPUs, the way `qemu-system-x86_64 -M q35 -accel tcg -smp
/// cpus -m 256 -nographic -kernel ... -initrd ... -append 'console=ttyS0 rdinit=/bin/sh'`
/// does, waits for the shell, has it compute something, count the CPUs and print the year
/// the CMOS clock gave the kernel, and powers off. The commands run as `busybox` applets, so
/// the initramfs needs only `/bin/sh` and `busybox`. `RUVM_TEST_LINUX_APPEND` replaces the
/// command line. The images are not in the tree; an Alpine `netboot` `vmlinuz-virt` and
/// `initramfs-virt` work.
fn boot_linux(cpus: u32) {
    let (Some(kernel), Some(initrd)) = (
        std::env::var("RUVM_TEST_LINUX_KERNEL").ok(),
        std::env::var("RUVM_TEST_LINUX_INITRD").ok(),
    ) else {
        eprintln!("skipped: RUVM_TEST_LINUX_KERNEL and RUVM_TEST_LINUX_INITRD are not set");
        return;
    };
    let Some(fw) = firmware() else { return };
    let append = std::env::var("RUVM_TEST_LINUX_APPEND")
        .unwrap_or_else(|_| "console=ttyS0 rdinit=/bin/sh".to_string());
    let cpu = TcgCpuModel::new(None).unwrap();
    let clock = Clock::new(ClockType::Virtual, TimeSource::Monotonic(Instant::now()));
    let rtc_clock = Clock::new(ClockType::Host, TimeSource::Wall);
    let spec = BoardSpec {
        kind: BoardKind::Q35,
        machine_type: "pc-q35-11.1",
        props: Vec::new(),
        ram_size: Some(256 << 20),
        cpus,
        max_cpus: cpus,
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
        kernel: Some(KernelFiles { kernel, initrd: Some(initrd), append }),
        firmware: fw,
        serial_hds: vec![true],
        clock: Arc::clone(&clock),
        rtc_clock: Arc::clone(&rtc_clock),
    };
    let (board, _) = build_board(spec).unwrap();
    let console = Arc::new(Console::default());
    assert!(board.set_serial_backend(0, Some(console.clone())));
    let serial = Arc::clone(board.serial(0).unwrap());

    let outcome = Arc::new(Outcome::default());
    let oc = Arc::clone(&outcome);
    let handler = Arc::new(move |e: GuestEvent| match e {
        GuestEvent::Shutdown(ruvm_machine_x86::run_event::ShutdownReason::GuestShutdown) => {
            oc.set(End::Shutdown);
        }
        GuestEvent::Shutdown(_) | GuestEvent::Reset => oc.set(End::Reset),
        GuestEvent::Panicked => oc.set(End::Error("panicked".into())),
        GuestEvent::InternalError(m) => oc.set(End::Error(m)),
    });
    let cfg = TcgRunConfig { no_reboot: true, ..TcgRunConfig::default() };
    let (m, _) = TcgMachine::new(board, &cpu, vec![clock, rtc_clock], &cfg, handler).unwrap();
    m.start();

    let until = Instant::now() + timeout();
    let fail = |what: &str| -> ! {
        let end = outcome.end.lock().unwrap().clone();
        panic!("{what} ({end:?}); console so far:\n{}", console.text());
    };
    // With RUVM_TCG_TRACE, says where the vCPUs are every half minute until the shell runs.
    let trace = std::env::var_os("RUVM_TCG_TRACE").is_some();
    let shell = loop {
        let step = if trace { until.min(Instant::now() + Duration::from_secs(30)) } else { until };
        let at = console.wait_for(0, "/bin/sh: can't access tty", step);
        if at.is_some() || Instant::now() >= until {
            break at;
        }
        dump_vcpus(&m);
    };
    let Some(at) = shell else { fail("no shell") };
    let Some(at) = console.wait_for(at, "# ", until) else { fail("no prompt") };
    type_line(
        &serial,
        "busybox mount -t proc proc /proc; echo ruvm-$((6*7)); \
         busybox grep -c ^processor /proc/cpuinfo; busybox date -u +ruvm-year-%Y\n",
        until,
    );
    let Some(at) = console.wait_for(at, "ruvm-42", until) else { fail("no answer") };
    let want = format!("\r\n{cpus}\r\n");
    let Some(at) = console.wait_for(at, &want, until) else { fail(&format!("not {cpus} CPUs")) };
    // The CMOS clock counts on the host clock, so the guest has today's date.
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
    let year = format!("ruvm-year-{}", 1900 + gmtime(secs).year);
    if console.wait_for(at, &year, until).is_none() {
        fail(&format!("not {year}"));
    }
    type_line(&serial, "busybox poweroff -f\n", until);
    let end = outcome.wait(until.saturating_duration_since(Instant::now()));
    m.pause();
    m.quit();
    assert_eq!(end, End::Shutdown, "console:\n{}", console.text());
}

#[test]
#[ignore = "needs a Linux kernel and initramfs, see boot_linux()"]
fn linux_smp1() {
    boot_linux(1);
}

#[test]
#[ignore = "needs a Linux kernel and initramfs, see boot_linux()"]
fn linux_smp2() {
    boot_linux(2);
}
