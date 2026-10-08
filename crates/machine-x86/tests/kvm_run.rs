// SPDX-License-Identifier: GPL-2.0-or-later

//! The boards on KVM. Every test is skipped, with a note on stderr, when `/dev/kvm` cannot be
//! used, unless `RUVM_REQUIRE_KVM` is set, in which case that is a failure.
//!
//! The boot test also needs a kernel: set `RUVM_TEST_KERNEL` to a bzImage with a serial
//! console. It looks for the firmware in `RUVM_TEST_FIRMWARE_DIR` and then the usual places,
//! and waits `RUVM_TEST_BOOT_TIMEOUT` seconds (default 30) for "Linux version" on the serial
//! port.

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use ruvm_accel_kvm::{KernelIrqchip, KvmAccel, KvmError, KvmOptions};
use ruvm_base::ClockType;
use ruvm_hw_char::serial::SerialBackend;
use ruvm_hw_core::Clock;
use ruvm_hw_core::timer::TimeSource;
use ruvm_machine_x86::board::{BoardKind, BoardSpec, KernelFiles, build_board};
use ruvm_machine_x86::kvm_run::{
    CpuModel, GuestEvent, KvmMachine, KvmRunConfig, init_error_lines, pit_in_kernel,
};
use ruvm_machine_x86::{FirmwareSearch, X86Board};

fn open(kind: BoardKind, irqchip: Option<KernelIrqchip>) -> Option<KvmAccel> {
    let opts = KvmOptions { kernel_irqchip: irqchip, ..KvmOptions::default() };
    match KvmAccel::new(&opts, kind.default_kernel_irqchip_split()) {
        Ok(a) => Some(a),
        Err(e @ (KvmError::Open(_) | KvmError::Unavailable)) => {
            if std::env::var_os("RUVM_REQUIRE_KVM").is_some() {
                panic!("RUVM_REQUIRE_KVM is set but KVM cannot be used: {e}");
            }
            eprintln!("skipping: {e}");
            None
        }
        Err(e) => panic!("{}", init_error_lines(&e).join("\n")),
    }
}

/// What the guest wrote to the serial port.
#[derive(Default)]
struct Capture {
    out: Mutex<Vec<u8>>,
    grew: Condvar,
}

impl SerialBackend for Capture {
    fn write(&self, bytes: &[u8]) -> usize {
        self.out.lock().unwrap().extend_from_slice(bytes);
        self.grew.notify_all();
        bytes.len()
    }
}

impl Capture {
    fn wait_for(&self, needle: &str, timeout: Duration) -> bool {
        let end = Instant::now() + timeout;
        let mut out = self.out.lock().unwrap();
        loop {
            if String::from_utf8_lossy(&out).contains(needle) {
                return true;
            }
            let now = Instant::now();
            if now >= end {
                return false;
            }
            out = self.grew.wait_timeout(out, end - now).unwrap().0;
        }
    }
}

struct Run {
    machine: KvmMachine,
    serial: Arc<Capture>,
    events: Arc<Mutex<Vec<GuestEvent>>>,
}

fn run(
    kind: BoardKind,
    irqchip: Option<KernelIrqchip>,
    firmware: FirmwareSearch,
    bios: Option<String>,
    kernel: Option<KernelFiles>,
) -> Option<Run> {
    let accel = open(kind, irqchip)?;
    let cpu = CpuModel::new(&accel, None).unwrap();
    let clock = Clock::new(ClockType::Virtual, TimeSource::Monotonic(Instant::now()));
    let rtc_clock = Clock::new(ClockType::Host, TimeSource::Wall);
    let spec = BoardSpec {
        kind,
        machine_type: if kind == BoardKind::Q35 { "pc-q35-11.1" } else { "microvm" },
        props: Vec::new(),
        ram_size: Some(256 << 20),
        memdev: None,
        aux_ram_share: false,
        cpus: 1,
        max_cpus: 0,
        kvm: true,
        pit_in_kernel: pit_in_kernel(&accel),
        smm_available: false,
        phys_bits: cpu.phys_bits(),
        cpu: cpu.ident(),
        bios,
        pflash: [None, None],
        uuid: None,
        smbios: Default::default(),
        topology: None,
        kernel,
        firmware,
        serial_hds: vec![true],
        clock: Arc::clone(&clock),
        rtc_clock: Arc::clone(&rtc_clock),
    };
    let (board, _) = build_board(spec).unwrap();
    let serial = Arc::new(Capture::default());
    assert!(board.set_serial_backend(0, Some(serial.clone())));
    let events = Arc::new(Mutex::new(Vec::new()));
    let ev = Arc::clone(&events);
    let machine = KvmMachine::new(
        accel,
        board,
        &cpu,
        vec![clock, rtc_clock],
        &KvmRunConfig::default(),
        Arc::new(move |e| ev.lock().unwrap().push(e)),
    )
    .unwrap();
    Some(Run { machine, serial, events })
}

/// A firmware image of `hlt` instructions, enough to get the vCPUs going.
fn hlt_firmware(name: &str, size: usize) -> (FirmwareSearch, PathBuf) {
    let dir = std::env::temp_dir().join(format!("ruvm-kvm-run-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(name), vec![0xf4; size]).unwrap();
    (FirmwareSearch::from_dirs(vec![dir.clone()]), dir)
}

fn start_pause_quit(kind: BoardKind, irqchip: Option<KernelIrqchip>, name: &str, size: usize) {
    let (fw, dir) = hlt_firmware(name, size);
    let Some(r) = run(kind, irqchip, fw, None, None) else { return };
    assert_eq!(r.machine.vcpu_count(), 1);
    r.machine.start();
    std::thread::sleep(Duration::from_millis(100));
    r.machine.pause();
    r.machine.start();
    r.machine.request_reset();
    std::thread::sleep(Duration::from_millis(100));
    r.machine.quit();
    assert_eq!(*r.events.lock().unwrap(), vec![GuestEvent::Reset]);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn microvm_split_starts_and_stops() {
    start_pause_quit(BoardKind::Microvm, None, "bios-microvm.bin", 128 * 1024);
}

#[test]
fn microvm_on_starts_and_stops() {
    start_pause_quit(BoardKind::Microvm, Some(KernelIrqchip::On), "bios-microvm.bin", 128 * 1024);
}

#[test]
fn q35_on_starts_and_stops() {
    start_pause_quit(BoardKind::Q35, None, "bios-256k.bin", 256 * 1024);
}

#[test]
fn q35_split_starts_and_stops() {
    start_pause_quit(BoardKind::Q35, Some(KernelIrqchip::Split), "bios-256k.bin", 256 * 1024);
}

#[test]
fn irqchip_off_is_refused() {
    let (fw, dir) = hlt_firmware("bios-256k.bin", 256 * 1024);
    let Some(accel) = open(BoardKind::Q35, Some(KernelIrqchip::Off)) else { return };
    let cpu = CpuModel::new(&accel, None).unwrap();
    let spec = BoardSpec {
        kind: BoardKind::Q35,
        machine_type: "pc-q35-11.1",
        props: Vec::new(),
        ram_size: None,
        memdev: None,
        aux_ram_share: false,
        cpus: 1,
        max_cpus: 0,
        kvm: true,
        pit_in_kernel: false,
        smm_available: false,
        phys_bits: cpu.phys_bits(),
        cpu: cpu.ident(),
        bios: None,
        pflash: [None, None],
        uuid: None,
        smbios: Default::default(),
        topology: None,
        kernel: None,
        firmware: fw,
        serial_hds: vec![true],
        clock: Clock::manual(ClockType::Virtual),
        rtc_clock: Clock::manual(ClockType::Host),
    };
    let board: X86Board = build_board(spec).unwrap().0;
    let e = KvmMachine::new(accel, board, &cpu, Vec::new(), &KvmRunConfig::default(), {
        Arc::new(|_| {})
    })
    .unwrap_err();
    assert_eq!(e, "kernel-irqchip=off is not supported yet");
    let _ = std::fs::remove_dir_all(dir);
}

fn boot(kind: BoardKind) {
    let Some(kernel) = std::env::var_os("RUVM_TEST_KERNEL") else {
        eprintln!("skipping: RUVM_TEST_KERNEL is not set");
        return;
    };
    let extra: Vec<PathBuf> =
        std::env::var_os("RUVM_TEST_FIRMWARE_DIR").map(PathBuf::from).into_iter().collect();
    let fw = FirmwareSearch::new(&extra);
    let bios = match kind {
        BoardKind::Microvm => "bios-microvm.bin",
        BoardKind::Q35 => "bios-256k.bin",
    };
    if fw.find(bios).is_none() {
        eprintln!("skipping: {bios} not found in {:?}", fw.dirs());
        return;
    }
    let timeout =
        std::env::var("RUVM_TEST_BOOT_TIMEOUT").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    let files = KernelFiles {
        kernel: kernel.to_string_lossy().into_owned(),
        initrd: None,
        append: "console=ttyS0 panic=-1".to_string(),
    };
    let Some(r) = run(kind, None, fw, None, Some(files)) else { return };
    r.machine.start();
    let ok = r.serial.wait_for("Linux version", Duration::from_secs(timeout));
    r.machine.quit();
    let out = String::from_utf8_lossy(&r.serial.out.lock().unwrap()).into_owned();
    assert!(ok, "no \"Linux version\" on the serial port within {timeout}s; got:\n{out}");
}

#[test]
fn microvm_boots_linux() {
    boot(BoardKind::Microvm);
}

#[test]
fn q35_boots_linux() {
    boot(BoardKind::Q35);
}
