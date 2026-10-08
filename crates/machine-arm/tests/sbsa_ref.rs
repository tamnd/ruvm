// SPDX-License-Identifier: GPL-2.0-or-later

//! The sbsa-ref board: its memory map, its device tree, the errors of `sbsa_ref_init()`, the
//! power key, the `sbsa-ec`, the watchdog and the AHCI ports, and the firmware booting on TCG.

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use ruvm_accel::tcg::TcgOptions;

use ruvm_base::ClockType;
use ruvm_hw_char::serial::SerialBackend;
use ruvm_hw_core::Clock;
use ruvm_hw_core::timer::TimeSource;
use ruvm_hw_storage::{BlockBackend, DriveConfig, VecBackend};
use ruvm_machine_arm::fdt::Fdt;
use ruvm_machine_arm::pflash::PflashBacking;
use ruvm_machine_arm::sbsa_ref::{
    NUM_SATA_PORTS, SBSA_AHCI, SBSA_GIC_DIST, SBSA_GIC_ITS, SBSA_GPIO, SBSA_GWDT_CONTROL,
    SBSA_GWDT_REFRESH, SBSA_MEM, SBSA_PCIE_ECAM, SBSA_RTC, SBSA_SECURE_EC, SBSA_SECURE_MEM,
    SBSA_SECURE_UART, SBSA_SECURE_UART_MM, SBSA_SMMU, SBSA_UART, SbsaRefConfig, SbsaRefMachine,
};
use ruvm_machine_arm::tcg_run::{SbsaRefTcgMachine, VirtEvent, VirtRunConfig};
use ruvm_machine_arm::virt::{CpuTopology, VirtRequest};
use ruvm_mem::MemTxAttrs;
use ruvm_target_arm::cpu::{ArmCpuModel, CpuArmState};

const U: MemTxAttrs = MemTxAttrs::UNSPECIFIED;

fn n1() -> SbsaRefConfig {
    SbsaRefConfig::new(ArmCpuModel::by_name("neoverse-n1").unwrap())
}

fn board(cfg: SbsaRefConfig) -> SbsaRefMachine {
    let mut m = SbsaRefMachine::new(cfg).unwrap();
    m.machine_done().unwrap();
    m
}

fn r32(m: &SbsaRefMachine, addr: u64) -> u32 {
    let mut b = [0; 4];
    assert!(m.memory_as().read(addr, U, &mut b).is_ok(), "read at {addr:#x}");
    u32::from_le_bytes(b)
}

fn w32(m: &SbsaRefMachine, addr: u64, v: u32) {
    assert!(m.memory_as().write(addr, U, &v.to_le_bytes()).is_ok(), "write at {addr:#x}");
}

fn cell(fdt: &Fdt, path: &str, name: &str) -> u32 {
    fdt.getprop_cell(path, name).unwrap()
}

fn cells(fdt: &Fdt, path: &str, name: &str) -> Vec<u64> {
    let v = fdt.getprop(path, name).unwrap();
    v.chunks(8).map(|c| u64::from_be_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn memory_map() {
    let m = board(n1());
    // The PrimeCell IDs of the PL011s, the PL031 and the PL061.
    for base in [SBSA_UART, SBSA_SECURE_UART, SBSA_SECURE_UART_MM] {
        assert_eq!(r32(&m, base + 0xfe0), 0x11, "pl011 at {base:#x}");
    }
    assert_eq!(r32(&m, SBSA_RTC + 0xfe0), 0x31);
    assert_eq!(r32(&m, SBSA_GPIO + 0xfe0), 0x61);
    // GICD_PIDR2 and GITS_PIDR2 give the architecture revision, 3.
    assert_eq!(r32(&m, SBSA_GIC_DIST + 0xffe8) >> 4 & 0xf, 3);
    assert_eq!(r32(&m, SBSA_GIC_ITS + 0xffe8) >> 4 & 0xf, 3);
    // HOST_CAP.NP and HOST_PORTS_IMPL of the six AHCI ports.
    assert_eq!(r32(&m, SBSA_AHCI) & 0x1f, NUM_SATA_PORTS as u32 - 1);
    assert_eq!(r32(&m, SBSA_AHCI + 0x0c), 0x3f);
    // W_IIDR in both watchdog frames.
    assert_eq!(r32(&m, SBSA_GWDT_REFRESH + 0xfcc), 0x1043b);
    assert_eq!(r32(&m, SBSA_GWDT_CONTROL + 0xfcc), 0x1043b);
    // The SMMU's IDR0 is not zero, and the host bridge is at 00:00.0 of the ECAM.
    assert_ne!(r32(&m, SBSA_SMMU), 0);
    assert_eq!(r32(&m, SBSA_PCIE_ECAM), 0x0008_1b36);
    // The secure RAM and the RAM.
    for addr in [SBSA_SECURE_MEM, SBSA_MEM, SBSA_MEM + (1 << 30) - 4] {
        w32(&m, addr, 0x1234_5678);
        assert_eq!(r32(&m, addr), 0x1234_5678);
    }
    // The generic timers count at 1 GHz.
    assert_eq!(m.cpu_model().cntfrq, 1_000_000_000);
    assert!(m.cpu_model().features.el3 && m.cpu_model().features.el2);
}

#[test]
fn device_tree() {
    let mut cfg = n1();
    cfg.smp = 10;
    let m = SbsaRefMachine::new(cfg).unwrap();
    let fdt = m.fdt();
    assert_eq!(fdt.getprop("/", "compatible").unwrap(), b"linux,sbsa-ref\0");
    assert_eq!(cell(fdt, "/", "machine-version-major"), 0);
    assert_eq!(cell(fdt, "/", "machine-version-minor"), 4);
    assert_eq!(cell(fdt, "/cpus", "#address-cells"), 2);
    assert_eq!(cell(fdt, "/cpus", "#size-cells"), 0);
    // Eight CPUs per cluster. The nodes are added from the last CPU down, and each new one
    // goes before its siblings, so they end up in order.
    assert_eq!(cells(fdt, "/cpus/cpu@0", "reg"), [0]);
    assert_eq!(cells(fdt, "/cpus/cpu@7", "reg"), [7]);
    assert_eq!(cells(fdt, "/cpus/cpu@9", "reg"), [0x101]);
    let blob = fdt.as_bytes();
    let at = |s: &[u8]| blob.windows(s.len()).position(|w| w == s).unwrap();
    assert!(at(b"cpu@0\0") < at(b"cpu@9\0"));
    // Without -smp the topology is one of everything.
    for name in ["sockets", "clusters", "cores", "threads"] {
        assert_eq!(cell(fdt, "/cpus/topology", name), 1);
    }
    assert_eq!(cells(fdt, "/intc", "reg"), [0x4006_0000, 0x1_0000, 0x4008_0000, 0x0400_0000]);
    assert_eq!(cells(fdt, "/intc/its", "reg"), [0x4408_1000, 0x2_0000]);
    assert!(!fdt.exists("/psci") && !fdt.exists("/chosen"));

    // machine_done adds the memory node and /chosen, but no PSCI node.
    let mut cfg = n1();
    cfg.smp = 2;
    cfg.topology =
        Some(CpuTopology { sockets: 1, clusters: 1, cores: 2, threads: 1, has_clusters: false });
    let m = board(cfg);
    let fdt = m.fdt();
    assert_eq!(cell(fdt, "/cpus/topology", "cores"), 2);
    assert_eq!(cells(fdt, "/memory@10000000000", "reg"), [SBSA_MEM, 1 << 30]);
    assert!(fdt.exists("/chosen") && !fdt.exists("/psci"));
    // The device tree goes to the base of RAM for the firmware.
    assert_eq!(m.boot_info().dtb_start, SBSA_MEM);
    assert!(m.roms().iter().any(|r| r.name == "dtb" && r.addr == SBSA_MEM));
}

/// A scratch file, removed when dropped.
struct TmpFile(PathBuf);

impl Drop for TmpFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn init_errors() {
    let e = |cfg| SbsaRefMachine::new(cfg).unwrap_err();
    let mut cfg = n1();
    cfg.max_cpus = Some(513);
    assert_eq!(
        e(cfg),
        "Number of SMP CPUs requested (513) exceeds max CPUs supported by machine 'sbsa-ref' (512)"
    );
    let mut cfg = n1();
    cfg.ram_size = (8 << 40) + 1;
    assert_eq!(e(cfg), "sbsa-ref: cannot model more than 8 TiB of RAM");
    let mut cfg = n1();
    cfg.firmware = Some("/nonexistent/SBSA_FLASH0.fd".to_string());
    assert_eq!(e(cfg), "Could not find ROM image '/nonexistent/SBSA_FLASH0.fd'");

    let fw = TmpFile(std::env::temp_dir().join(format!("ruvm-sbsa-{}.fd", std::process::id())));
    std::fs::write(&fw.0, [0u8; 4096]).unwrap();
    let mut cfg = n1();
    cfg.firmware = Some(fw.0.to_str().unwrap().to_string());
    cfg.kernel = Some("Image".to_string());
    assert!(e(cfg).starts_with(
        "This machine type does not support loading both a guest firmware/BIOS image and a \
         guest kernel at the same time."
    ));
    // The firmware alone is fine, and goes into the first flash.
    let mut cfg = n1();
    cfg.firmware = Some(fw.0.to_str().unwrap().to_string());
    let m = board(cfg);
    assert!(!m.boot_info().direct);
}

#[test]
fn power_key_on_the_pl061() {
    let m = board(n1());
    assert_eq!(r32(&m, SBSA_GPIO + 0x20), 0);
    // GPIOIEV and GPIOIE for a rising edge on pin 3, then the press raises pin 3 and the
    // PL061 interrupt.
    w32(&m, SBSA_GPIO + 0x40c, 8);
    w32(&m, SBSA_GPIO + 0x410, 8);
    m.system_powerdown();
    assert_eq!(r32(&m, SBSA_GPIO + 0x20), 8);
    assert_eq!(r32(&m, SBSA_GPIO + 0x418), 8);
}

#[test]
fn secure_ec_requests() {
    let m = board(n1());
    assert_eq!(m.take_request(), None);
    w32(&m, SBSA_SECURE_EC, 2);
    assert_eq!(m.take_request(), Some(VirtRequest::Reset));
    w32(&m, SBSA_SECURE_EC, 1);
    assert_eq!(m.take_request(), Some(VirtRequest::Shutdown));
    w32(&m, SBSA_SECURE_EC, 3);
    assert_eq!(m.take_request(), None);
}

#[test]
fn watchdog_resets_the_machine() {
    let clock = Clock::new(ClockType::Virtual, TimeSource::Manual);
    let mut cfg = n1();
    cfg.clock = Some(clock.clone());
    let m = board(cfg);
    // WOR of 1000 ticks at 1 GHz, then WCS.EN: WS0 after 1 us and the reset after 2 us.
    w32(&m, SBSA_GWDT_CONTROL + 0x8, 1000);
    w32(&m, SBSA_GWDT_CONTROL, 1);
    clock.advance_to(1000);
    assert_eq!(r32(&m, SBSA_GWDT_CONTROL) & 2, 2);
    assert_eq!(m.take_request(), None);
    clock.advance_to(2000);
    assert_eq!(m.take_request(), Some(VirtRequest::Reset));
}

#[test]
fn ahci_drives() {
    let m = SbsaRefMachine::new(n1()).unwrap();
    let blk: Arc<dyn BlockBackend> = Arc::new(VecBackend::new(1 << 20));
    m.attach_drive(0, DriveConfig::hd(), Some(blk)).unwrap();
    m.attach_drive(5, DriveConfig::cdrom(), None).unwrap();
    assert!(m.attach_drive(6, DriveConfig::cdrom(), None).is_err());
}

/// The UART: collects what the guest writes and wakes the waiter.
#[derive(Default)]
struct Console {
    out: Mutex<Vec<u8>>,
    cv: Condvar,
    ended: Mutex<Option<String>>,
}

impl SerialBackend for Console {
    fn write(&self, bytes: &[u8]) -> usize {
        self.out.lock().unwrap().extend_from_slice(bytes);
        self.cv.notify_all();
        bytes.len()
    }
}

/// Prints where each vCPU is, for `RUVM_TCG_TRACE`.
fn dump_vcpus(m: &SbsaRefTcgMachine) {
    m.vcpus().run_on_each(|cpu| {
        let st = CpuArmState::load(cpu.env);
        let el = ((st.pstate >> 2) & 3).max(1) as usize;
        let shared = cpu.shared();
        eprintln!(
            "cpu {} pc {:#x} pstate {:#x} halted {} req {:#x}",
            shared.cpu_index,
            st.pc,
            st.pstate,
            shared.halted.load(Ordering::Relaxed),
            shared.interrupt_request(),
        );
        eprintln!(
            "  elr {:#x} esr {:#x} far {:#x} lr {:#x}",
            st.elr_el[el], st.esr_el[el], st.far_el[el], st.xregs[30],
        );
    });
}

/// The SBSA-REF firmware QEMU's tests use (TF-A and EDK2 in `SBSA_FLASH0.fd`, the variables in
/// `SBSA_FLASH1.fd`), from the directory `RUVM_TEST_SBSA_FW`, up to the UEFI boot manager on
/// the UART. Without it the test passes without running. `RUVM_TEST_TIMEOUT_SECS` changes the
/// 1200 s timeout, and with `RUVM_TCG_TRACE` it prints the output and the vCPUs every ten
/// seconds.
#[test]
fn firmware_boot() {
    let Some(dir) = std::env::var_os("RUVM_TEST_SBSA_FW") else {
        return;
    };
    let dir = PathBuf::from(dir);
    let flash = |name: &str| PflashBacking::Bytes(std::fs::read(dir.join(name)).unwrap());
    let clock = Clock::new(ClockType::Virtual, TimeSource::Monotonic(Instant::now()));
    let rtc_clock = Clock::new(ClockType::Host, TimeSource::Wall);
    let console = Arc::new(Console::default());
    let mut cfg = n1();
    cfg.pflash = [flash("SBSA_FLASH0.fd"), flash("SBSA_FLASH1.fd")];
    cfg.serial = Some(console.clone());
    cfg.clock = Some(Arc::clone(&clock));
    cfg.rtc_clock = Some(Arc::clone(&rtc_clock));
    let board = SbsaRefMachine::new(cfg).unwrap();
    let c = Arc::clone(&console);
    let handler = Arc::new(move |e: VirtEvent| {
        *c.ended.lock().unwrap() = Some(format!("{e:?}"));
        c.cv.notify_all();
    });
    let cfg = VirtRunConfig { no_reboot: false, tcg: TcgOptions::default(), backend: None };
    let (m, _) = SbsaRefTcgMachine::new(board, vec![clock, rtc_clock], &cfg, handler).unwrap();
    m.start();
    let trace = std::env::var_os("RUVM_TCG_TRACE").is_some();
    let secs = std::env::var("RUVM_TEST_TIMEOUT_SECS").ok().and_then(|s| s.parse().ok());
    let deadline = Instant::now() + Duration::from_secs(secs.unwrap_or(1200));
    let want = b"Select Language";
    let mut shown = 0;
    let mut next_tick = Instant::now() + Duration::from_secs(10);
    let ok = loop {
        let out = console.out.lock().unwrap();
        if out.windows(want.len()).any(|w| w == want) {
            break true;
        }
        if console.ended.lock().unwrap().is_some() || Instant::now() >= deadline {
            break false;
        }
        let (out, _) = console.cv.wait_timeout(out, Duration::from_secs(1)).unwrap();
        if trace && Instant::now() >= next_tick {
            eprint!("{}", String::from_utf8_lossy(&out[shown..]));
            shown = out.len();
            drop(out);
            dump_vcpus(&m);
            next_tick = Instant::now() + Duration::from_secs(10);
        }
    };
    m.pause();
    m.quit();
    let text = String::from_utf8_lossy(&console.out.lock().unwrap()).into_owned();
    assert!(ok, "no boot manager (ended: {:?}); output:\n{text}", console.ended.lock().unwrap());
    assert!(text.contains("Booting Trusted Firmware"), "{text}");
}
