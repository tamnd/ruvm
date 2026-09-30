// SPDX-License-Identifier: GPL-2.0-or-later

//! The `dmg` format driver end to end: images made by `hdiutil` in each chunk format (zlib,
//! bzip2, lzfse, raw) read in full and compared with `hdiutil`'s own raw conversion and with
//! `qemu-img convert`, probing by file name, and the open errors QEMU gives. The tests skip
//! themselves when `hdiutil` (macOS only) or `qemu-img` is not there.

#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use ruvm_block::{BLK_PERM_CONSISTENT_READ, BlockBackend, BlockGraph};
use ruvm_qapi::QDict;
use ruvm_qapi::json;
use ruvm_qapi::types::BlockdevOptions;
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};

const ALL: u64 = 0x1f;

fn qemu_img() -> Option<PathBuf> {
    let candidates = ["/opt/homebrew/bin/qemu-img", "/usr/local/bin/qemu-img", "/usr/bin/qemu-img"];
    let found = candidates.iter().map(PathBuf::from).find(|p| p.exists());
    if found.is_none() {
        eprintln!("qemu-img not found, skipping the qemu-img comparisons");
    }
    found
}

fn hdiutil() -> Option<PathBuf> {
    let p = PathBuf::from("/usr/bin/hdiutil");
    if p.exists() {
        Some(p)
    } else {
        eprintln!("hdiutil not found, skipping");
        None
    }
}

fn scratch(test: &str) -> PathBuf {
    let base = option_env!("CARGO_TARGET_TMPDIR").map_or_else(std::env::temp_dir, PathBuf::from);
    let dir = base.join("ruvm-block-dmg").join(test);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn path_str(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn run(cmd: &Path, args: &[&str]) {
    let out = Command::new(cmd).args(args).output().unwrap();
    assert!(
        out.status.success(),
        "{} {args:?} failed: {}",
        cmd.display(),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Runs `cmd`, which must fail, and returns its stderr.
fn run_fail(cmd: &Path, args: &[&str]) -> String {
    let out = Command::new(cmd).args(args).output().unwrap();
    assert!(!out.status.success(), "{} {args:?} succeeded", cmd.display());
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn from_json<T: Visit + Default>(s: &str) -> T {
    let mut v = QObjectInputVisitor::new(json::from_str(s).unwrap());
    let mut o = T::default();
    T::visit(&mut v, None, &mut o).unwrap();
    o
}

/// `blockdev-add` of a read-only dmg node `name` over `path`.
fn add_dmg(g: &BlockGraph, name: &str, path: &Path, read_only: bool) -> ruvm_base::Result<()> {
    g.blockdev_add(from_json::<BlockdevOptions>(&format!(
        r#"{{"driver": "dmg", "node-name": "{name}", "read-only": {read_only},
            "file": {{"driver": "file", "filename": "{}"}}}}"#,
        path_str(path)
    )))
}

fn read_all(blk: &BlockBackend) -> std::io::Result<Vec<u8>> {
    let len = blk.getlength().unwrap() as usize;
    let mut buf = vec![0u8; len];
    blk.pread(0, &mut buf)?;
    Ok(buf)
}

/// A folder with incompressible and compressible files and some zeroes, so that the image
/// has chunks of every kind.
fn source_folder(dir: &Path) -> PathBuf {
    let src = dir.join("src");
    fs::create_dir_all(&src).unwrap();
    let mut x: u32 = 99;
    let random: Vec<u8> = (0..300_000)
        .map(|_| {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (x >> 16) as u8
        })
        .collect();
    fs::write(src.join("random.bin"), random).unwrap();
    let text: String = (0..30_000).map(|i| format!("line {i} of some text\n")).collect();
    fs::write(src.join("text.txt"), text).unwrap();
    fs::write(src.join("zeroes.bin"), vec![0u8; 100_000]).unwrap();
    src
}

#[test]
fn hdiutil_images() {
    let Some(hdiutil) = hdiutil() else { return };
    let qemu_img = qemu_img();
    let dir = scratch("hdiutil");
    let src = source_folder(&dir);

    for format in ["UDZO", "UDBZ", "ULFO", "UDRO"] {
        let img = dir.join(format!("{format}.dmg"));
        // hdiutil writes UDRO images of MS-DOS volumes that it then calls corrupt itself, so
        // that one gets the default file system.
        let fs = if format == "UDRO" { "HFS+" } else { "MS-DOS" };
        run(
            &hdiutil,
            &[
                "create",
                "-quiet",
                "-srcfolder",
                path_str(&src),
                "-fs",
                fs,
                "-volname",
                "T",
                "-format",
                format,
                "-ov",
                path_str(&img),
            ],
        );
        // hdiutil's own raw copy of the disk; it adds the .cdr suffix.
        let reference = dir.join(format!("{format}-ref"));
        run(
            &hdiutil,
            &[
                "convert",
                "-quiet",
                path_str(&img),
                "-format",
                "UDTO",
                "-ov",
                "-o",
                path_str(&reference),
            ],
        );
        let reference = fs::read(dir.join(format!("{format}-ref.cdr"))).unwrap();

        let g = BlockGraph::new();
        add_dmg(&g, "d", &img, true).unwrap();
        assert_eq!(g.node("d").unwrap().driver, "dmg");
        let blk = BlockBackend::new(&g, "d", BLK_PERM_CONSISTENT_READ, ALL).unwrap();
        let ours = read_all(&blk);
        let lzfse_missing = format == "ULFO" && !cfg!(feature = "dmg-lzfse");
        let bzip2_missing = format == "UDBZ" && !cfg!(feature = "dmg-bzip2");
        if lzfse_missing || bzip2_missing {
            // As QEMU without the module: the chunks are left out and reading them fails.
            let e = ours.unwrap_err();
            assert_eq!(e.raw_os_error(), Some(libc::EIO), "{format}");
        } else {
            let ours = ours.unwrap();
            assert_eq!(ours.len(), reference.len(), "{format}");
            assert!(ours == reference, "{format}: data differs from hdiutil");
        }

        // Unaligned reads go through the 512 byte alignment.
        if !(lzfse_missing || bzip2_missing) {
            let mut b = vec![0u8; 3000];
            blk.pread(12_345, &mut b).unwrap();
            assert!(b[..] == reference[12_345..15_345], "{format}");
        }
        drop(blk);
        drop(g);

        let Some(qemu_img) = &qemu_img else { continue };
        let theirs = dir.join(format!("{format}-qemu.raw"));
        let args = ["convert", "-f", "dmg", "-O", "raw", path_str(&img), path_str(&theirs)];
        if format == "ULFO" {
            // Homebrew's QEMU has no dmg-lzfse module; when it has one, it must agree.
            let out = Command::new(qemu_img).args(args).output().unwrap();
            if out.status.success() {
                assert!(fs::read(&theirs).unwrap() == reference, "{format}: qemu-img differs");
            } else {
                let err = String::from_utf8_lossy(&out.stderr);
                assert!(err.contains("dmg-lzfse module is missing"), "{err}");
            }
        } else {
            run(qemu_img, &args);
            assert!(fs::read(&theirs).unwrap() == reference, "{format}: qemu-img differs");
        }
    }
}

#[test]
fn probe_by_name() {
    let Some(hdiutil) = hdiutil() else { return };
    let dir = scratch("probe");
    let src = source_folder(&dir);
    let img = dir.join("disk.dmg");
    run(
        &hdiutil,
        &[
            "create",
            "-quiet",
            "-srcfolder",
            path_str(&src),
            "-format",
            "UDZO",
            "-ov",
            path_str(&img),
        ],
    );

    let mut opts = QDict::new();
    opts.put("read-only", "on");
    let g = BlockGraph::new();
    let name = g.open_image(Some(path_str(&img)), opts.clone()).unwrap();
    assert_eq!(g.node(&name).unwrap().driver, "dmg");

    // The same bytes under another name are not probed as dmg.
    let other = dir.join("disk.img");
    fs::copy(&img, &other).unwrap();
    let g = BlockGraph::new();
    let name = g.open_image(Some(path_str(&other)), opts).unwrap();
    assert_ne!(g.node(&name).unwrap().driver, "dmg");
}

#[test]
fn open_errors() {
    let dir = scratch("errors");
    let qemu_img = qemu_img();

    // Each case: the file and the message, which qemu-img must show too. The length check
    // sees whole sectors, so a 511 byte file passes it and only an empty one fails it.
    let empty = dir.join("empty.dmg");
    fs::write(&empty, []).unwrap();
    let short = dir.join("short.dmg");
    fs::write(&short, [0u8; 511]).unwrap();
    let no_koly = dir.join("nokoly.dmg");
    fs::write(&no_koly, vec![0u8; 4096]).unwrap();

    // A trailer whose data fork starts after it.
    let bad_fork = dir.join("badfork.dmg");
    let mut b = vec![0u8; 1024];
    b[512..516].copy_from_slice(b"koly");
    b[512 + 0x18..512 + 0x20].copy_from_slice(&4096u64.to_be_bytes());
    fs::write(&bad_fork, &b).unwrap();

    // A trailer without resource fork or property list.
    let no_tables = dir.join("notables.dmg");
    let mut b = vec![0u8; 1024];
    b[512..516].copy_from_slice(b"koly");
    fs::write(&no_tables, &b).unwrap();

    let einval = |p: &Path| format!("Could not open '{}': Invalid argument", path_str(p));
    let cases = [
        (&empty, "dmg file must be at least 512 bytes long".to_string()),
        (&short, "Could not locate UDIF trailer in dmg file".to_string()),
        (&no_koly, "Could not locate UDIF trailer in dmg file".to_string()),
        (&bad_fork, einval(&bad_fork)),
        (&no_tables, einval(&no_tables)),
    ];
    for (path, msg) in cases {
        let g = BlockGraph::new();
        let e = add_dmg(&g, "d", path, true).unwrap_err();
        assert_eq!(e.message(), msg);
        if let Some(q) = &qemu_img {
            let err = run_fail(q, &["info", "-f", "dmg", path_str(path)]);
            assert!(err.contains(&msg), "qemu-img said {err}, we said {msg}");
        }
    }

    // Opening read-write without auto-read-only fails.
    let g = BlockGraph::new();
    let e = add_dmg(&g, "d", &short, false).unwrap_err();
    assert_eq!(e.message(), "Image is read-only");
}
