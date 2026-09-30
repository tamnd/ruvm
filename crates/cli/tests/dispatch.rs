// SPDX-License-Identifier: GPL-2.0-or-later

//! The dispatcher as a user meets it: the real binary, started under QEMU's names.

use std::path::PathBuf;
use std::process::Command;

fn ruvm() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ruvm"))
}

fn run(args: &[&str]) -> (bool, String) {
    let out = Command::new(ruvm()).args(args).output().unwrap();
    (out.status.success(), String::from_utf8(out.stdout).unwrap())
}

#[test]
fn system_version_through_the_name_argument() {
    let (ok, out) = run(&["qemu-system-x86_64", "--version"]);
    assert!(ok);
    assert!(out.starts_with("QEMU emulator version 11.1.0 (ruvm "), "{out}");
    let (ok, single) = run(&["qemu-system-x86_64", "-version"]);
    assert!(ok);
    assert_eq!(out, single);
}

#[cfg(unix)]
#[test]
fn system_version_through_a_symlink() {
    let dir = std::env::temp_dir().join(format!("ruvm-dispatch-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let link = dir.join("qemu-system-aarch64");
    let _ = std::fs::remove_file(&link);
    std::os::unix::fs::symlink(ruvm(), &link).unwrap();
    let out = Command::new(&link).arg("-version").output().unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8(out.stdout).unwrap().starts_with("QEMU emulator version 11.1.0"));
}

#[test]
fn tools_answer_minus_capital_v() {
    let (ok, out) = run(&["qemu-img", "-V"]);
    assert!(ok);
    assert!(out.starts_with("qemu-img version 11.1.0 (ruvm "), "{out}");
}

#[test]
fn list_names_every_system_target() {
    let (ok, out) = run(&["--list"]);
    assert!(ok);
    let names: Vec<&str> = out.lines().collect();
    assert!(names.contains(&"qemu-system-x86_64"));
    assert!(names.contains(&"qemu-img"));
    assert!(names.contains(&"qemu-riscv64"));
    assert!(!names.contains(&"ruvm"));
}

#[test]
fn unimplemented_programs_fail_rather_than_pretend() {
    let (ok, _) = run(&["qemu-system-x86_64", "-machine", "q35"]);
    assert!(!ok);
    let (ok, _) = run(&["qemu-nonsense"]);
    assert!(!ok);
}
