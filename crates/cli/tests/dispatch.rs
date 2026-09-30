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

/// The target tables in `names.rs` are QEMU's `configs/targets/`, which is vendored. A sync that
/// adds or drops a target fails here until the dispatcher learns about it.
#[test]
fn every_vendored_target_is_a_name() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vendor-qemu/targets");
    let (_, out) = run(&["--list"]);
    let names: Vec<&str> = out.lines().collect();
    let mut want = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let file = entry.unwrap().file_name().into_string().unwrap();
        let Some(stem) = file.strip_suffix(".mak") else { continue };
        if let Some(target) = stem.strip_suffix("-softmmu") {
            want.push(format!("qemu-system-{target}"));
        } else if let Some(target) =
            stem.strip_suffix("-linux-user").or_else(|| stem.strip_suffix("-bsd-user"))
        {
            want.push(format!("qemu-{target}"));
        }
    }
    assert!(want.len() > 60);
    for name in &want {
        assert!(names.contains(&name.as_str()), "{name} is a QEMU target but not a ruvm name");
    }
    let count = |prefix: &str| names.iter().filter(|n| n.starts_with(prefix)).count();
    let system = want.iter().filter(|n| n.starts_with("qemu-system-")).count();
    assert_eq!(count("qemu-system-"), system, "ruvm answers to a system target QEMU does not have");
}

/// The version ruvm reports is the version of the QEMU it vendors.
#[test]
fn the_reported_version_is_the_vendored_tag() {
    let upstream = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../vendor-qemu/UPSTREAM");
    let text = std::fs::read_to_string(upstream).unwrap();
    let tag = text.lines().find_map(|l| l.strip_prefix("tag v")).unwrap();
    let (_, out) = run(&["qemu-system-x86_64", "--version"]);
    assert!(out.starts_with(&format!("QEMU emulator version {tag} ")), "{out}");
}
