// SPDX-License-Identifier: GPL-2.0-or-later

//! Images written by one implementation must pass `qemu-img check` from the other.
//!
//! For every format, QEMU's qemu-img and ours each convert the same source image, then the
//! other program checks the result and compares it with the source. Formats without a check
//! routine must say so in the same way from both sides. The test skips itself when QEMU's
//! tools are not installed.

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const QEMU_IMG: &str = "/opt/homebrew/bin/qemu-img";
const QEMU_IO: &str = "/opt/homebrew/bin/qemu-io";
const OURS: &str = env!("CARGO_BIN_EXE_ruvm-qemu-img");

/// Formats with a check routine in QEMU.
const CHECKED: &[&str] = &["qcow2", "qed", "vdi", "vhdx", "vmdk", "parallels"];
/// Formats without one, where both programs must refuse in the same way.
const UNCHECKED: &[&str] = &["raw", "qcow", "vpc"];

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!("ruvm-img-cross-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run(dir: &Path, bin: &str, args: &[&str]) -> Output {
    Command::new(bin).args(args).current_dir(dir).output().unwrap()
}

fn run_ok(dir: &Path, bin: &str, args: &[&str]) {
    let out = run(dir, bin, args);
    assert!(out.status.success(), "{bin} {args:?}: {}", String::from_utf8_lossy(&out.stderr));
}

fn name(bin: &str) -> &'static str {
    if bin == OURS { "ruvm" } else { "qemu" }
}

/// Writes `out` from `src` with `writer`, then checks and compares it with `checker`.
fn cross(dir: &Path, writer: &str, checker: &str, fmt: &str, extra: &[&str], src: &str) {
    let out = format!("{}-{fmt}{}.img", name(writer), extra.len());
    let mut args = vec!["convert", "-O", fmt];
    args.extend_from_slice(extra);
    args.extend([src, out.as_str()]);
    run_ok(dir, writer, &args);

    let check = run(dir, checker, &["check", "-f", fmt, &out]);
    let text = String::from_utf8_lossy(&check.stdout);
    let what = format!("{} wrote {fmt} {extra:?}, {} checked", name(writer), name(checker));
    if CHECKED.contains(&fmt) {
        assert!(check.status.success(), "{what}: {text}{}", String::from_utf8_lossy(&check.stderr));
        assert!(text.contains("No errors were found on the image."), "{what}: {text}");
    } else {
        let theirs = run(dir, writer, &["check", "-f", fmt, &out]);
        assert_eq!(check.status.code(), theirs.status.code(), "{what}: exit codes differ");
        assert_eq!(check.stderr, theirs.stderr, "{what}: messages differ");
    }
    let cmp = run(dir, checker, &["compare", "-f", "raw", "-F", fmt, src, &out]);
    assert!(cmp.status.success(), "{what}: {}", String::from_utf8_lossy(&cmp.stdout));
}

#[test]
fn images_pass_the_other_check() {
    if !Path::new(QEMU_IMG).exists() || !Path::new(QEMU_IO).exists() {
        eprintln!("skipping: QEMU's qemu-img or qemu-io is missing");
        return;
    }
    let s = Scratch::new();
    let dir = s.0.as_path();
    run_ok(dir, QEMU_IMG, &["create", "-q", "-f", "raw", "src.raw", "8M"]);
    run_ok(
        dir,
        QEMU_IO,
        &[
            "-f",
            "raw",
            "-c",
            "write -P 0x11 0 64k",
            "-c",
            "write -P 0x22 1M 192k",
            "-c",
            "write -P 0x33 7M 4k",
            "src.raw",
        ],
    );

    let mut cases: Vec<(&str, Vec<&str>)> = Vec::new();
    for fmt in CHECKED.iter().chain(UNCHECKED) {
        if ruvm_block::tools::format_exists(fmt) {
            cases.push((fmt, vec![]));
        } else {
            eprintln!("skipping {fmt}: the driver is not registered");
        }
    }
    if ruvm_block::tools::format_exists("qcow2") {
        cases.push(("qcow2", vec!["-c"]));
        cases.push(("qcow2", vec!["-o", "cluster_size=4096,lazy_refcounts=on"]));
        cases.push(("qcow2", vec!["-o", "compat=0.10"]));
        cases.push(("qcow2", vec!["-o", "extended_l2=on"]));
    }
    if ruvm_block::tools::format_exists("vmdk") {
        cases.push(("vmdk", vec!["-o", "subformat=streamOptimized"]));
    }
    if ruvm_block::tools::format_exists("vdi") {
        cases.push(("vdi", vec!["-o", "static=on"]));
    }
    for (fmt, extra) in &cases {
        cross(dir, OURS, QEMU_IMG, fmt, extra, "src.raw");
        cross(dir, QEMU_IMG, OURS, fmt, extra, "src.raw");
    }
}
