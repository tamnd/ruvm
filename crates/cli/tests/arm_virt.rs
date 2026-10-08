// SPDX-License-Identifier: GPL-2.0-or-later

//! `qemu-system-aarch64 -M virt` on TCG: the startup errors, `dumpdtb`, and QEMU's tests/tcg
//! system tests (see crates/machine-arm/tests/data/tcg/SOURCES) run with the options of
//! tests/tcg/aarch64/Makefile.softmmu-target, their semihosting console on a file chardev.
//!
//! A Linux boot to a shell needs a kernel and an initramfs, which are not in the tree; see
//! `linux_boots_to_a_shell`.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use ruvm_machine_arm::virt::{VirtConfig, VirtMachine, VirtMsi};

fn ruvm() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ruvm"))
}

/// Runs `qemu-system-aarch64` with `args`, returning the exit code, stdout and stderr.
fn system(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(ruvm()).arg("qemu-system-aarch64").args(args).output().unwrap();
    let text = |b: Vec<u8>| String::from_utf8_lossy(&b).into_owned();
    (out.status.code().unwrap_or(-1), text(out.stdout), text(out.stderr))
}

fn arm_data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../machine-arm/tests/data").join(name)
}

fn gunzip(path: &Path) -> Vec<u8> {
    let f = std::fs::File::open(path).unwrap();
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(f).read_to_end(&mut out).unwrap();
    out
}

/// A directory under the temp directory, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        let p = std::env::temp_dir().join(format!("ruvm-cli-arm-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }

    fn path(&self, name: &str) -> String {
        self.0.join(name).to_str().unwrap().to_string()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn virt_errors() {
    let p = "qemu-system-aarch64: ";
    for (args, want) in [
        (&["-M", "virt", "-cpu", "foo"][..], format!("{p}unable to find CPU model 'foo'\n")),
        (
            &["-M", "virt", "-cpu", "cortex-a53"],
            format!("{p}CPU model 'cortex-a53' is not supported by ruvm yet\n"),
        ),
        (
            &["-M", "virt", "-cpu", "max,sve-max-vq=17"],
            format!("{p}unsupported SVE vector length\nValid sve-max-vq in range [1-16]\n"),
        ),
        (&["-M", "virt,ras=on"], format!("{p}ras=on is not supported by ruvm yet\n")),
        (
            &["-M", "virt,mte=on", "-cpu", "cortex-a57"],
            format!("{p}MTE requested, but not supported by the guest CPU\n"),
        ),
        (
            &["-M", "virt", "-bios", "/nonexistent/ruvm.fd"],
            format!("{p}Could not find ROM image '/nonexistent/ruvm.fd'\n"),
        ),
        (&["-M", "virt,foo=on"], format!("{p}Property 'virt-11.1-machine.foo' not found\n")),
        (
            &["-M", "virt,highmem-mmio-size=1G"],
            format!(
                "{p}highmem-mmio-size cannot be set to a lower value than the default (512 GiB)\n"
            ),
        ),
        (
            &["-M", "virt,highmem=off", "-m", "4G"],
            format!(
                "{p}Addressing limited to 32 bits, but memory exceeds it by 1073741824 bytes\n"
            ),
        ),
        (
            &["-M", "virt", "-device", "virtio-blk-pci"],
            format!("{p}-device virtio-blk-pci: drive property not set\n"),
        ),
        (
            &["-M", "virt", "-drive", "file=/dev/null,format=raw,if=ide"],
            format!(
                "{p}-drive file=/dev/null,format=raw,if=ide: machine type does not support \
                 if=ide,bus=0,unit=0\n"
            ),
        ),
        (
            &["-M", "virt,highmem-redists=off", "-smp", "124"],
            format!(
                "{p}Number of SMP CPUs requested (124) exceeds max CPUs supported by machine \
                 'mach-virt' (123)\nTry 'highmem-redists=on' for more CPUs\n"
            ),
        ),
        (
            &["-M", "virt,msi=foo"],
            format!("{p}Invalid msi value\nValid values are auto, gicv2m, its, off\n"),
        ),
        (&["-M", "virt,msi=gicv2m"], format!("{p}msi=gicv2m is not supported by ruvm yet\n")),
        (
            &["-M", "virt", "-smp", "dies=2"],
            format!("{p}dies > 1 not supported by this machine's CPU topology\n"),
        ),
        (
            &["-semihosting-config", "target=foo"],
            format!(
                "{p}-semihosting-config target=foo: unsupported semihosting-config target=foo\n"
            ),
        ),
        (
            &["-M", "virt", "-semihosting-config", "chardev=nope"],
            format!("{p}semihosting chardev 'nope' not found\n"),
        ),
        (
            &["-M", "virt", "-accel", "qtest"],
            format!("{p}this machine type is only supported with -accel kvm or tcg by ruvm yet\n"),
        ),
    ] {
        let (code, _, err) = system(args);
        assert_eq!((code, err), (1, want), "{args:?}");
    }
    let (code, out, _) = system(&["-M", "help"]);
    assert_eq!(code, 0);
    assert!(
        out.contains(
            "virt                 QEMU 11.1 ARM Virtual Machine (alias of virt-11.1)\n\
             virt-11.1            QEMU 11.1 ARM Virtual Machine\n"
        ),
        "{out}"
    );
    let (code, out, _) = system(&["-accel", "help"]);
    assert_eq!((code, out.as_str()), (0, "Accelerators supported in QEMU binary:\ntcg\n"));
}

/// `-M virt,dumpdtb=` writes the board's device tree and exits. The tree has to be the one QEMU
/// writes, and the board's own, so this checks that the options (`pmu=off` too) reach it.
#[test]
fn virt_dumpdtb() {
    let dir = TempDir::new("dumpdtb");
    let dtb = dir.path("virt.dtb");
    let m = format!("virt,gic-version=3,its=off,dtb-randomness=off,dumpdtb={dtb}");
    let (code, out, err) =
        system(&["-nodefaults", "-display", "none", "-M", &m, "-cpu", "cortex-a57,pmu=off"]);
    assert_eq!((code, out.as_str(), err.as_str()), (0, "", ""));
    let cpu = ruvm_target_arm::cpu::ArmCpuModel::by_name("cortex-a57").unwrap().without_pmu();
    let mut cfg = VirtConfig::new(cpu.clone());
    cfg.msi = VirtMsi::Off;
    let mut board = VirtMachine::new(cfg).unwrap();
    board.machine_done().unwrap();
    let want = board.fdt().as_bytes();
    let got = std::fs::read(&dtb).unwrap();
    assert!(got == want, "the device tree differs from the board's");
    assert!(got == gunzip(&arm_data("virt-a57.dtb.gz")), "the device tree differs from QEMU's");

    // With the ITS and two redistributor regions, which -smp 130 needs.
    let m = format!("virt,gic-version=3,dtb-randomness=off,dumpdtb={dtb}");
    let args = ["-nodefaults", "-display", "none", "-M", &m, "-cpu", "cortex-a57,pmu=off"];
    let (code, out, err) = system(&[&args[..], &["-smp", "130"]].concat());
    assert_eq!((code, out.as_str(), err.as_str()), (0, "", ""));
    let mut cfg = VirtConfig::new(cpu);
    cfg.smp = 130;
    let mut board = VirtMachine::new(cfg).unwrap();
    board.machine_done().unwrap();
    let got = std::fs::read(&dtb).unwrap();
    assert!(got == board.fdt().as_bytes(), "the device tree differs from the board's");
}

/// Runs test kernel `name` the way `make check-tcg` does, with `extra` options. Gives the exit
/// status and the semihosting console output.
fn run_tcg_test(name: &str, extra: &[&str]) -> (i32, Vec<u8>, String) {
    let dir = TempDir::new(name);
    let kernel = dir.path(name);
    std::fs::write(&kernel, gunzip(&arm_data(&format!("tcg/{name}.gz")))).unwrap();
    let out = dir.path("out");
    let chardev = format!("file,path={out},id=output");
    let mut args = vec![
        "-monitor",
        "none",
        "-display",
        "none",
        "-chardev",
        &chardev,
        "-M",
        "virt",
        "-cpu",
        "max",
        "-display",
        "none",
        "-semihosting-config",
        "enable=on,target=native,chardev=output",
    ];
    args.extend_from_slice(extra);
    args.extend_from_slice(&["-kernel", &kernel]);
    let (code, _, err) = system(&args);
    (code, std::fs::read(&out).unwrap_or_default(), err)
}

#[test]
fn tcg_hello() {
    let (code, out, err) = run_tcg_test("hello", &[]);
    assert_eq!((code, out.as_slice(), err.as_str()), (0, &b"Hello World\n"[..], ""));
}

#[test]
fn tcg_interrupt_smp2_single_thread() {
    let (code, out, err) = run_tcg_test("interrupt", &["-smp", "2", "-accel", "tcg,thread=single"]);
    assert_eq!((code, out.as_slice(), err.as_str()), (0, &b""[..], ""));
}

#[test]
fn tcg_semiheap() {
    let want = std::fs::read(arm_data("tcg/semiheap.out")).unwrap();
    let (code, out, err) = run_tcg_test("semiheap", &[]);
    assert_eq!((code, err.as_str()), (0, ""));
    assert!(out == want, "{}", String::from_utf8_lossy(&out));
}

/// FEAT_TCR2 with FEAT_ASID2, and FEAT_XS, on `-cpu max`.
#[test]
fn tcg_asid2_and_feat_xs() {
    let (code, out, err) = run_tcg_test("asid2", &[]);
    assert_eq!((code, out.as_slice(), err.as_str()), (0, &b"OK\n"[..], ""));
    let (code, out, err) = run_tcg_test("feat-xs", &[]);
    assert_eq!((code, out.as_slice(), err.as_str()), (0, &b""[..], ""));
}

/// Without a chardev the semihosting console is standard error.
#[test]
fn semihosting_console_on_stderr() {
    let dir = TempDir::new("stderr");
    let kernel = dir.path("hello");
    std::fs::write(&kernel, gunzip(&arm_data("tcg/hello.gz"))).unwrap();
    let (code, out, err) = system(&[
        "-M",
        "virt",
        "-cpu",
        "max",
        "-display",
        "none",
        "-semihosting",
        "-kernel",
        &kernel,
    ]);
    assert_eq!((code, out.as_str(), err.as_str()), (0, "", "Hello World\n"));
}

/// Boots Linux on `-M virt -nographic` to a busybox shell on the PL011 and runs a command
/// there. The kernel and initramfs come from `RUVM_TEST_ARM64_KERNEL` and
/// `RUVM_TEST_ARM64_INITRD` (an initramfs whose `/init` starts a shell on the console, as
/// `scripts/arm64-linux-test-image.py` makes); without them the test says so and passes.
/// `RUVM_TEST_ARM64_SMP` sets `-smp` (2 by default) and `RUVM_TEST_TIMEOUT_SECS` how long to
/// wait for the prompt and then for the command.
#[test]
#[ignore = "needs a kernel and an initramfs, and minutes"]
fn linux_boots_to_a_shell() {
    let (Some(kernel), Some(initrd)) = (
        std::env::var("RUVM_TEST_ARM64_KERNEL").ok(),
        std::env::var("RUVM_TEST_ARM64_INITRD").ok(),
    ) else {
        eprintln!("skipped: RUVM_TEST_ARM64_KERNEL and RUVM_TEST_ARM64_INITRD are not set");
        return;
    };
    let smp = std::env::var("RUVM_TEST_ARM64_SMP").unwrap_or_else(|_| "2".to_string());
    let mut child = Command::new(ruvm())
        .arg("qemu-system-aarch64")
        .args(["-M", "virt", "-cpu", "max", "-smp", &smp, "-m", "512", "-nographic"])
        .args(["-kernel", &kernel, "-initrd", &initrd, "-append", "console=ttyAMA0"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = child.stdout.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel::<Vec<u8>>();
    let reader = std::thread::spawn(move || {
        let mut buf = [0u8; 4096];
        loop {
            match stdout.read(&mut buf) {
                Ok(0) | Err(_) => return,
                Ok(n) => {
                    if tx.send(buf[..n].to_vec()).is_err() {
                        return;
                    }
                }
            }
        }
    });
    let mut log = Vec::new();
    let mut wait_for = |needle: &str, limit: Duration| -> bool {
        let deadline = Instant::now() + limit;
        loop {
            if String::from_utf8_lossy(&log).contains(needle) {
                return true;
            }
            let now = Instant::now();
            if now >= deadline {
                return false;
            }
            match rx.recv_timeout(deadline - now) {
                Ok(b) => {
                    std::io::stderr().write_all(&b).unwrap();
                    log.extend_from_slice(&b);
                }
                Err(_) => return false,
            }
        }
    };
    let secs = std::env::var("RUVM_TEST_TIMEOUT_SECS").ok().and_then(|s| s.parse().ok());
    // An hour: the boot took 85 minutes of wall time with a third of a busy host core.
    let limit = Duration::from_secs(secs.unwrap_or(3600));
    let booted = wait_for("/ # ", limit);
    if booted {
        stdin.write_all(b"echo ruvm-$((6*7)); poweroff -f\n").unwrap();
    }
    let ran = booted && wait_for("ruvm-42", limit);
    let _ = child.kill();
    let _ = child.wait();
    drop(reader);
    assert!(booted, "no shell prompt");
    assert!(ran, "the command did not run");
}
