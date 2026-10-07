// SPDX-License-Identifier: GPL-2.0-or-later

//! `qemu-system-riscv64 -M virt` on TCG: the startup errors, `dumpdtb`, and small guests
//! booted the way QEMU's tests/tcg/riscv64 system tests are, `-bios none -semihosting
//! -device loader,file=<elf>`. The guests are a few instructions each, written into an
//! ELF by the test, so no binaries live in the tree.
//!
//! The loader runs without `cpu-num`, so the hart starts at the reset vector in the boot
//! ROM at 0x1000, which jumps to the base of RAM where the ELF sits.
//!
//! A Linux boot to a shell needs a kernel, an initramfs and the OpenSBI firmware, which are
//! not in the tree; see `linux_boots_to_a_shell`.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn ruvm() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ruvm"))
}

/// Runs `qemu-system-riscv64` with `args`, returning the exit code, stdout and stderr.
fn system(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(ruvm()).arg("qemu-system-riscv64").args(args).output().unwrap();
    let text = |b: Vec<u8>| String::from_utf8_lossy(&b).into_owned();
    (out.status.code().unwrap_or(-1), text(out.stdout), text(out.stderr))
}

/// A directory under the temp directory, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        let p = std::env::temp_dir().join(format!("ruvm-cli-riscv-{tag}-{}", std::process::id()));
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

const DRAM_BASE: u64 = 0x8000_0000;

/// A little-endian ELF64 RISC-V executable with one loadable segment holding `code` at
/// the base of RAM, the entry point at its first instruction.
fn riscv_elf(code: &[u32]) -> Vec<u8> {
    const EHDR: usize = 64;
    const PHDR: usize = 56;
    const OFFSET: usize = 0x80;
    let text: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
    let mut e = Vec::new();
    e.extend_from_slice(b"\x7fELF");
    e.extend_from_slice(&[2, 1, 1, 0]); // ELFCLASS64, ELFDATA2LSB, EV_CURRENT, SYSV
    e.extend_from_slice(&[0; 8]);
    e.extend_from_slice(&2u16.to_le_bytes()); // ET_EXEC
    e.extend_from_slice(&243u16.to_le_bytes()); // EM_RISCV
    e.extend_from_slice(&1u32.to_le_bytes());
    e.extend_from_slice(&DRAM_BASE.to_le_bytes()); // e_entry
    e.extend_from_slice(&(EHDR as u64).to_le_bytes()); // e_phoff
    e.extend_from_slice(&0u64.to_le_bytes()); // e_shoff
    e.extend_from_slice(&0u32.to_le_bytes()); // e_flags
    e.extend_from_slice(&(EHDR as u16).to_le_bytes());
    e.extend_from_slice(&(PHDR as u16).to_le_bytes());
    e.extend_from_slice(&1u16.to_le_bytes()); // e_phnum
    e.extend_from_slice(&0u16.to_le_bytes()); // e_shentsize
    e.extend_from_slice(&0u16.to_le_bytes()); // e_shnum
    e.extend_from_slice(&0u16.to_le_bytes()); // e_shstrndx
    assert_eq!(e.len(), EHDR);
    e.extend_from_slice(&1u32.to_le_bytes()); // PT_LOAD
    e.extend_from_slice(&5u32.to_le_bytes()); // PF_R | PF_X
    e.extend_from_slice(&(OFFSET as u64).to_le_bytes());
    e.extend_from_slice(&DRAM_BASE.to_le_bytes()); // p_vaddr
    e.extend_from_slice(&DRAM_BASE.to_le_bytes()); // p_paddr
    e.extend_from_slice(&(text.len() as u64).to_le_bytes()); // p_filesz
    e.extend_from_slice(&(text.len() as u64).to_le_bytes()); // p_memsz
    e.extend_from_slice(&4u64.to_le_bytes()); // p_align
    e.resize(OFFSET, 0);
    e.extend_from_slice(&text);
    e
}

/// Writes `(5 << 16) | FINISHER_FAIL` to the SiFive test device.
const FINISHER_FAIL_5: &[u32] = &[
    0x0010_02b7, // lui    t0, 0x100
    0x0005_3337, // lui    t1, 0x53
    0x3333_031b, // addiw  t1, t1, 0x333
    0x0062_a023, // sw     t1, 0(t0)
    0x0000_006f, // j      .
];

/// SYS_EXIT_EXTENDED with ADP_Stopped_ApplicationExit and exit code 3.
const SEMIHOSTING_EXIT_3: &[u32] = &[
    0x0000_1597, // auipc  a1, 1
    0x0002_02b7, // lui    t0, 0x20
    0x0262_829b, // addiw  t0, t0, 0x26
    0x0055_b023, // sd     t0, 0(a1)
    0x0030_0293, // li     t0, 3
    0x0055_b423, // sd     t0, 8(a1)
    0x0200_0513, // li     a0, 0x20
    0x0000_0013, // nop, so the sequence below is 16-byte aligned
    0x01f0_1013, // slli   zero, zero, 0x1f
    0x0010_0073, // ebreak
    0x4070_5013, // srai   zero, zero, 7
    0x0000_006f, // j      .
];

/// Reads the first word of the boot ROM, runs the reset vector again, and checks the
/// registers it hands over: a2 points at fw_dynamic_info at 0x1028 and a1 at the FDT.
/// Exits through the SiFive test device with code 7 when all holds, 9, 11 or 13 if not.
const RESET_VECTOR_PROBE: &[u32] = &[
    0x0000_63b7, // lui    t2, 0x6
    0xa5a3_839b, // addiw  t2, t2, -1446
    0x0274_0263, // beq    s0, t2, back
    0x0000_12b7, // lui    t0, 0x1
    0x0002_a303, // lw     t1, 0(t0)
    0x2970_0e13, // li     t3, 0x297
    0x0090_0793, // li     a5, 9
    0x03c3_1663, // bne    t1, t3, fail
    0x0003_8413, // mv     s0, t2
    0x0000_12b7, // lui    t0, 0x1
    0x0002_8067, // jr     t0
    0x0000_1e37, // back: lui t3, 0x1
    0x028e_0e1b, // addiw  t3, t3, 40
    0x00b0_0793, // li     a5, 11
    0x01c6_1863, // bne    a2, t3, fail
    0x00d0_0793, // li     a5, 13
    0x0005_8463, // beqz   a1, fail
    0x0070_0793, // li     a5, 7
    0x0010_02b7, // fail: lui t0, 0x100
    0x0107_9793, // slli   a5, a5, 16
    0x0000_3337, // lui    t1, 0x3
    0x3333_031b, // addiw  t1, t1, 0x333
    0x00f3_6333, // or     t1, t1, a5
    0x0062_a023, // sw     t1, 0(t0)
    0x0000_006f, // j      .
];

fn run_guest(tag: &str, code: &[u32], extra: &[&str]) -> (i32, String, String) {
    let dir = TempDir::new(tag);
    let elf = dir.path("guest.elf");
    std::fs::write(&elf, riscv_elf(code)).unwrap();
    let loader = format!("loader,file={elf}");
    let mut args = vec!["-M", "virt", "-bios", "none", "-display", "none", "-semihosting"];
    args.extend_from_slice(extra);
    args.extend_from_slice(&["-device", &loader]);
    system(&args)
}

#[test]
fn machine_help_lists_virt() {
    let (code, out, _) = system(&["-M", "help"]);
    assert_eq!(code, 0);
    assert!(out.contains("virt                 RISC-V VirtIO board"), "{out}");
}

#[test]
fn startup_errors() {
    let (code, _, err) = system(&["-M", "virt", "-display", "none", "-bios", "/nonexistent"]);
    assert_eq!(code, 1);
    assert!(err.contains("Unable to find the RISC-V BIOS \"/nonexistent\""), "{err}");

    let (code, _, err) =
        system(&["-M", "virt", "-display", "none", "-bios", "none", "-append", "x"]);
    assert_eq!(code, 1);
    assert!(err.contains("-append only allowed with -kernel option"), "{err}");

    let (code, _, err) = system(&["-M", "virt,nosuch=on", "-display", "none", "-bios", "none"]);
    assert_eq!(code, 1);
    assert!(err.contains("Property 'virt-machine.nosuch' not found"), "{err}");
}

#[test]
fn dumpdtb_writes_a_flattened_tree() {
    let dir = TempDir::new("dtb");
    let dtb = dir.path("virt.dtb");
    let m = format!("virt,dumpdtb={dtb}");
    let (code, _, err) = system(&["-M", &m, "-display", "none", "-bios", "none", "-smp", "2"]);
    assert_eq!(code, 0, "{err}");
    let blob = std::fs::read(&dtb).unwrap();
    assert_eq!(&blob[..4], &[0xd0, 0x0d, 0xfe, 0xed]);
    let has = |s: &[u8]| blob.windows(s.len()).any(|w| w == s);
    assert!(has(b"riscv-virtio,qemu\0"));
    assert!(has(b"cpu@1\0"));
    assert!(has(b"sifive,test1\0"));
}

#[test]
fn reset_vector_reaches_the_loader_image() {
    let (code, _, err) = run_guest("finisher", FINISHER_FAIL_5, &[]);
    assert_eq!(code, 5, "{err}");
}

#[test]
fn reset_vector_hands_over_the_boot_registers() {
    let (code, _, err) = run_guest("probe", RESET_VECTOR_PROBE, &[]);
    assert_eq!(code, 7, "{err}");
}

#[test]
fn semihosting_exit_code_is_the_exit_status() {
    let (code, _, err) = run_guest("semi", SEMIHOSTING_EXIT_3, &[]);
    assert_eq!(code, 3, "{err}");
    let (code, _, err) = run_guest("semi-smp", SEMIHOSTING_EXIT_3, &["-smp", "2"]);
    assert_eq!(code, 3, "{err}");
}

/// Boots Linux on `-M virt -nographic` through the default OpenSBI firmware to a busybox
/// shell on the 16550 and runs a command there. The kernel (a flat `Image`, such as the
/// `linux` of Debian's riscv64 netboot installer) and the initramfs come from
/// `RUVM_TEST_RISCV64_KERNEL` and `RUVM_TEST_RISCV64_INITRD`; the initramfs holds a static
/// busybox and an `/init` that mounts /proc, /sys and /dev and starts a shell on the console
/// (`exec setsid cttyhack sh`), as `scripts/arm64-linux-test-image.py` makes for arm64.
/// Without them the test says so and passes. `RUVM_TEST_FIRMWARE_DIR` is passed as `-L`, for
/// when `opensbi-riscv64-generic-fw_dynamic.bin` is not in a default data directory.
/// `RUVM_TEST_RISCV64_SMP` sets `-smp` (2 by default) and `RUVM_TEST_TIMEOUT_SECS` how long
/// to wait for the prompt and then for the command.
#[test]
#[ignore = "needs a kernel, an initramfs and OpenSBI"]
fn linux_boots_to_a_shell() {
    let (Some(kernel), Some(initrd)) = (
        std::env::var("RUVM_TEST_RISCV64_KERNEL").ok(),
        std::env::var("RUVM_TEST_RISCV64_INITRD").ok(),
    ) else {
        eprintln!("skipped: RUVM_TEST_RISCV64_KERNEL and RUVM_TEST_RISCV64_INITRD are not set");
        return;
    };
    let smp = std::env::var("RUVM_TEST_RISCV64_SMP").unwrap_or_else(|_| "2".to_string());
    let fw = std::env::var("RUVM_TEST_FIRMWARE_DIR").ok();
    let fw_args: Vec<&str> = fw.iter().flat_map(|d| ["-L", d.as_str()]).collect();
    let mut child = Command::new(ruvm())
        .arg("qemu-system-riscv64")
        .args(fw_args)
        .args(["-M", "virt", "-smp", &smp, "-m", "512", "-nographic"])
        .args(["-kernel", &kernel, "-initrd", &initrd, "-append", "console=ttyS0"])
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
    let mut wait_for = |needles: &[&str], limit: Duration| -> bool {
        let deadline = Instant::now() + limit;
        loop {
            let text = String::from_utf8_lossy(&log);
            if needles.iter().any(|n| text.contains(n)) {
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
    // Ten minutes: the boot takes 10 to 40 seconds of wall time on a busy x86-64 host.
    let limit = Duration::from_secs(secs.unwrap_or(600));
    // The busybox prompt shows the working directory, `/`, as `~` when it is also $HOME.
    let booted = wait_for(&["/ # ", "~ # "], limit);
    if booted {
        stdin.write_all(b"echo ruvm-$((6*7)); poweroff -f\n").unwrap();
    }
    let ran = booted && wait_for(&["ruvm-42"], limit);
    let _ = child.kill();
    let _ = child.wait();
    drop(reader);
    assert!(booted, "no shell prompt");
    assert!(ran, "the command did not run");
}
