// SPDX-License-Identifier: GPL-2.0-or-later

//! virt on TCG booting the UEFI firmware QEMU ships for it (`pc-bios/edk2-aarch64-code.fd`,
//! uncompressed) to the UEFI shell on the PL011, as `qemu-system-aarch64 -M virt -cpu max
//! -m 1G -nographic -bios edk2-aarch64-code.fd` does. The firmware comes from
//! `RUVM_TEST_ARM_EDK2`, a path; without it the tests pass without running. The shell comes
//! up after about 640 s on a loaded server3, where QEMU takes about 7 s, so the default
//! timeout is 1200 s; `RUVM_TEST_TIMEOUT_SECS` changes it. With `RUVM_TCG_TRACE` they print
//! the new output and where the vCPUs are every ten seconds, and with `RUVM_EDK2_DUMP` (a
//! path) they write RAM there at the end.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use ruvm_accel::tcg::TcgOptions;
use ruvm_base::ClockType;
use ruvm_hw_char::serial::SerialBackend;
use ruvm_hw_core::Clock;
use ruvm_hw_core::timer::TimeSource;
use ruvm_machine_arm::tcg_run::{VirtEvent, VirtRunConfig, VirtTcgMachine};
use ruvm_machine_arm::virt::{VirtConfig, VirtMachine};
use ruvm_mem::MemTxAttrs;
use ruvm_target_arm::cpu::{ArmCpuModel, CpuArmState};

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

impl Console {
    /// Waits until the output has `want` or the run ends, up to `timeout`.
    fn wait_for(&self, want: &[u8], timeout: Duration, tick: impl Fn()) -> bool {
        let deadline = Instant::now() + timeout;
        let mut next_tick = Instant::now() + Duration::from_secs(10);
        let mut g = self.out.lock().unwrap();
        loop {
            if g.windows(want.len()).any(|w| w == want) {
                return true;
            }
            if self.ended.lock().unwrap().is_some() {
                return false;
            }
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            if now >= next_tick {
                drop(g);
                tick();
                next_tick = now + Duration::from_secs(10);
                g = self.out.lock().unwrap();
            }
            let wait = (deadline - now).min(Duration::from_secs(1));
            g = self.cv.wait_timeout(g, wait).unwrap().0;
        }
    }
}

fn timeout() -> Duration {
    let secs = std::env::var("RUVM_TEST_TIMEOUT_SECS").ok().and_then(|s| s.parse().ok());
    Duration::from_secs(secs.unwrap_or(1200))
}

/// Prints where each vCPU is, for `RUVM_TCG_TRACE`.
fn dump_vcpus(m: &VirtTcgMachine) {
    m.vcpus().run_on_each(|cpu| {
        let st = CpuArmState::load(cpu.env);
        let el = (st.pstate >> 2) & 3;
        eprintln!(
            "cpu {} pc {:#x} el {el} pstate {:#x} halted {} req {:#x} elr {:#x} esr {:#x} far {:#x} lr {:#x}",
            cpu.shared().cpu_index,
            st.pc,
            st.pstate,
            cpu.shared().halted.load(Ordering::Relaxed),
            cpu.shared().interrupt_request(),
            st.elr_el[el.max(1) as usize],
            st.esr_el[el.max(1) as usize],
            st.far_el[el.max(1) as usize],
            st.xregs[30],
        );
        eprintln!("  x0-x7 {:x?} sp {:#x}", &st.xregs[..8], st.xregs[31]);
    });
}

fn boot(setup: impl FnOnce(&mut VirtConfig)) {
    let Some(fw) = std::env::var_os("RUVM_TEST_ARM_EDK2") else {
        return;
    };
    let clock = Clock::new(ClockType::Virtual, TimeSource::Monotonic(Instant::now()));
    let rtc_clock = Clock::new(ClockType::Host, TimeSource::Wall);
    let console = Arc::new(Console::default());
    let mut cfg = VirtConfig::new(ArmCpuModel::by_name("max").unwrap());
    cfg.ram_size = 1 << 30;
    cfg.firmware = Some(fw.to_str().unwrap().to_string());
    cfg.serial = Some(console.clone());
    cfg.clock = Some(Arc::clone(&clock));
    cfg.rtc_clock = Some(Arc::clone(&rtc_clock));
    setup(&mut cfg);
    let board = VirtMachine::new(cfg).unwrap();
    let c = Arc::clone(&console);
    let handler = Arc::new(move |e: VirtEvent| {
        *c.ended.lock().unwrap() = Some(format!("{e:?}"));
        c.cv.notify_all();
    });
    let cfg = VirtRunConfig { no_reboot: true, tcg: TcgOptions::default(), backend: None };
    let (m, warnings) = VirtTcgMachine::new(board, vec![clock, rtc_clock], &cfg, handler).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    m.start();
    let trace = std::env::var_os("RUVM_TCG_TRACE").is_some();
    let shown = Mutex::new(0);
    let ok = console.wait_for(b"Shell> ", timeout(), || {
        if trace {
            // The output since the last tick, then the vCPUs.
            let out = console.out.lock().unwrap();
            let mut shown = shown.lock().unwrap();
            eprint!("{}", String::from_utf8_lossy(&out[*shown..]));
            *shown = out.len();
            drop(out);
            dump_vcpus(&m);
        }
    });
    m.pause();
    if let Some(path) = std::env::var_os("RUVM_EDK2_DUMP") {
        // All of RAM, to disassemble where the vCPUs were.
        let board = m.board().lock().unwrap();
        let mut ram = vec![0; board.ram_size() as usize];
        assert!(board.memory_as().read(0x4000_0000, MemTxAttrs::UNSPECIFIED, &mut ram).is_ok());
        std::fs::write(path, ram).unwrap();
    }
    m.quit();
    let text = String::from_utf8_lossy(&console.out.lock().unwrap()).into_owned();
    let ended = console.ended.lock().unwrap().clone();
    assert!(ok, "no shell prompt (ended: {ended:?}); output:\n{text}");
}

#[test]
fn edk2_shell() {
    boot(|_| {});
}

#[test]
fn edk2_shell_el2() {
    // -M virt,virtualization=on: the firmware starts at EL2.
    boot(|cfg| cfg.virtualization = true);
}
