// SPDX-License-Identifier: GPL-2.0-or-later

//! The `luks` format driver end to end: images made by `qemu-img` read and written here, and
//! images made here (with `blockdev-create`) read by `qemu-img`, attached and detached headers,
//! several ciphers. The interop tests skip themselves when `qemu-img` is not installed; the
//! `cryptsetup` check skips itself when `cryptsetup` is not there.

#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Once;

use ruvm_block::{
    BLK_PERM_CONSISTENT_READ, BLK_PERM_RESIZE, BLK_PERM_WRITE, BlockBackend, BlockGraph,
};
use ruvm_qapi::json;
use ruvm_qapi::types::{BlockdevCreateOptions, BlockdevOptions};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};

const RW: u64 = BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE;
const SHARED: u64 = BLK_PERM_CONSISTENT_READ;
const PASSWORD: &str = "123456";

/// The global secrets and a fast PBKDF calibration, once per test binary.
fn setup() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        ruvm_crypto::pbkdf::set_iters_per_second_override(Some(1000));
        ruvm_crypto::secret::secret_object_add_global(&format!("secret,id=sec0,data={PASSWORD}"))
            .unwrap();
        ruvm_crypto::secret::secret_object_add_global("secret,id=sec1,data=wrong").unwrap();
    });
}

/// `qemu-img`, or `None` (and a note) when it is not installed.
fn qemu_img() -> Option<PathBuf> {
    let candidates = ["/opt/homebrew/bin/qemu-img", "/usr/local/bin/qemu-img", "/usr/bin/qemu-img"];
    let found = candidates.iter().map(PathBuf::from).find(|p| p.exists());
    if found.is_none() {
        eprintln!("qemu-img not found, skipping");
    }
    found
}

fn scratch(test: &str) -> PathBuf {
    let base = option_env!("CARGO_TARGET_TMPDIR").map_or_else(std::env::temp_dir, PathBuf::from);
    let dir = base.join("ruvm-block-luks").join(test);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len).map(|i| ((i % 251) as u8).wrapping_add(seed)).collect()
}

fn path_str(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// Runs `qemu-img` with `args` and the `sec0` secret object, and returns stdout.
fn run(qemu_img: &Path, args: &[&str]) -> String {
    let secret = format!("secret,id=sec0,data={PASSWORD}");
    let mut cmd = Command::new(qemu_img);
    // --object goes after the subcommand.
    cmd.arg(args[0]).arg("--object").arg(&secret).args(&args[1..]);
    let out = cmd.output().unwrap();
    assert!(
        out.status.success(),
        "qemu-img {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn from_json<T: Visit + Default>(s: &str) -> T {
    let mut v = QObjectInputVisitor::new(json::from_str(s).unwrap());
    let mut o = T::default();
    T::visit(&mut v, None, &mut o).unwrap();
    o
}

/// `blockdev-add` of a luks node `name` over `path`, with a detached header if `header`.
fn add_luks(
    g: &BlockGraph,
    name: &str,
    path: &Path,
    header: Option<&Path>,
    secret: &str,
) -> ruvm_base::Result<()> {
    let header = header.map_or(String::new(), |h| {
        format!(r#""header": {{"driver": "file", "filename": "{}"}},"#, path_str(h))
    });
    g.blockdev_add(from_json::<BlockdevOptions>(&format!(
        r#"{{"driver": "luks", "node-name": "{name}", "key-secret": "{secret}", {header}
            "file": {{"driver": "file", "node-name": "{name}-file", "filename": "{}"}}}}"#,
        path_str(path)
    )))
}

/// `blockdev-create` of an empty file of `size` bytes.
fn create_file(g: &BlockGraph, path: &Path) {
    g.blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
        r#"{{"driver": "file", "filename": "{}", "size": 0}}"#,
        path_str(path)
    )))
    .unwrap();
}

fn read_all(blk: &BlockBackend) -> Vec<u8> {
    let len = blk.getlength().unwrap() as usize;
    let mut buf = vec![0u8; len];
    blk.pread(0, &mut buf).unwrap();
    buf
}

#[test]
fn qemu_img_image_read_and_written_here() {
    setup();
    let Some(qemu_img) = qemu_img() else { return };
    let dir = scratch("from-qemu");
    let raw = dir.join("data.raw");
    let luks = dir.join("disk.luks");
    let size = 4 << 20;
    let data = pattern(size, 7);
    fs::write(&raw, &data).unwrap();
    run(
        &qemu_img,
        &[
            "convert",
            "-f",
            "raw",
            "-O",
            "luks",
            "-o",
            "key-secret=sec0,iter-time=10",
            path_str(&raw),
            path_str(&luks),
        ],
    );

    {
        let g = BlockGraph::new();
        add_luks(&g, "l", &luks, None, "sec0").unwrap();
        let blk = BlockBackend::new(&g, "l", RW, SHARED).unwrap();
        assert_eq!(blk.getlength().unwrap(), size as u64);
        assert_eq!(read_all(&blk), data);
        // Unaligned write: the generic layer pads it to whole sectors.
        blk.pwrite(1000, b"written by ruvm").unwrap();
        blk.pwrite(3 << 20, &pattern(1 << 20, 99)).unwrap();
    }

    let out = dir.join("out.raw");
    run(
        &qemu_img,
        &[
            "convert",
            "--image-opts",
            &format!("driver=luks,key-secret=sec0,file.filename={}", path_str(&luks)),
            "-O",
            "raw",
            path_str(&out),
        ],
    );
    let mut expect = data;
    expect[1000..1015].copy_from_slice(b"written by ruvm");
    expect[3 << 20..].copy_from_slice(&pattern(1 << 20, 99));
    assert!(fs::read(&out).unwrap() == expect);

    // The wrong password.
    let g = BlockGraph::new();
    let e = add_luks(&g, "l", &luks, None, "sec1").unwrap_err();
    assert_eq!(e.message(), "Invalid password, cannot unlock any keyslot");
}

/// `blockdev-create` here, then `qemu-img` reads and checks the image.
fn created_here(test: &str, crypto: &str, detached: bool) {
    setup();
    let Some(qemu_img) = qemu_img() else { return };
    let dir = scratch(test);
    let luks = dir.join("disk.luks");
    let hdr = dir.join("disk.hdr");
    let size: usize = 3 << 20;
    let data = pattern(size, 3);
    {
        let g = BlockGraph::new();
        create_file(&g, &luks);
        let header = if detached {
            create_file(&g, &hdr);
            format!(r#""header": {{"driver": "file", "filename": "{}"}},"#, path_str(&hdr))
        } else {
            String::new()
        };
        g.blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
            r#"{{"driver": "luks", "key-secret": "sec0", "iter-time": 10, {crypto} {header}
                "file": {{"driver": "file", "filename": "{}"}}, "size": {size}}}"#,
            path_str(&luks)
        )))
        .unwrap();
        if detached {
            assert_eq!(fs::metadata(&luks).unwrap().len(), size as u64);
        }
        add_luks(&g, "l", &luks, detached.then_some(hdr.as_path()), "sec0").unwrap();
        let blk = BlockBackend::new(&g, "l", RW | BLK_PERM_RESIZE, SHARED).unwrap();
        assert_eq!(blk.getlength().unwrap(), size as u64);
        blk.pwrite(0, &data).unwrap();
        assert_eq!(read_all(&blk), data);
    }

    // qemu-img sees the same plaintext.
    let out = dir.join("out.raw");
    let image_opts = if detached {
        format!(
            "driver=luks,key-secret=sec0,file.filename={},header.driver=file,header.filename={}",
            path_str(&luks),
            path_str(&hdr)
        )
    } else {
        format!("driver=luks,key-secret=sec0,file.filename={}", path_str(&luks))
    };
    run(&qemu_img, &["convert", "--image-opts", &image_opts, "-O", "raw", path_str(&out)]);
    assert!(fs::read(&out).unwrap() == data);

    if !detached {
        let info = run(&qemu_img, &["info", "--output=json", path_str(&luks)]);
        let info = json::from_str(&info).unwrap();
        let info = info.as_dict().unwrap();
        assert_eq!(info.get_str("format"), Some("luks"));
        assert_eq!(info.get("encrypted").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(info.get("virtual-size").and_then(|v| v.as_u64()), Some(size as u64));
    }
}

#[test]
fn created_here_default() {
    created_here("default", "", false);
}

#[test]
fn created_here_aes_128_cbc_essiv() {
    created_here(
        "cbc-essiv",
        r#""cipher-alg": "aes-128", "cipher-mode": "cbc", "ivgen-alg": "essiv",
           "ivgen-hash-alg": "sha256", "hash-alg": "sha512","#,
        false,
    );
}

#[test]
fn created_here_aes_256_cbc_plain() {
    // Homebrew's qemu-img uses the GnuTLS cipher backend, which has AES but not Twofish,
    // Serpent or CAST5, so the interop cases stick to AES.
    created_here(
        "cbc-plain",
        r#""cipher-alg": "aes-256", "cipher-mode": "cbc", "ivgen-alg": "plain",
           "hash-alg": "sha1","#,
        false,
    );
}

#[test]
fn created_here_detached_header() {
    created_here("detached", "", true);
}

#[test]
fn qemu_img_detached_header_read_here() {
    setup();
    let Some(qemu_img) = qemu_img() else { return };
    let dir = scratch("qemu-detached");
    let hdr = dir.join("disk.hdr");
    let luks = dir.join("disk.luks");
    run(
        &qemu_img,
        &[
            "create",
            "-f",
            "luks",
            "-o",
            "key-secret=sec0,iter-time=10,detached-header=on",
            path_str(&hdr),
        ],
    );
    let size: usize = 1 << 20;
    fs::write(&luks, vec![0u8; size]).unwrap();
    let data = pattern(size, 11);
    {
        let g = BlockGraph::new();
        add_luks(&g, "l", &luks, Some(&hdr), "sec0").unwrap();
        let blk = BlockBackend::new(&g, "l", RW, SHARED).unwrap();
        assert_eq!(blk.getlength().unwrap(), size as u64);
        blk.pwrite(0, &data).unwrap();
    }
    let out = dir.join("out.raw");
    run(
        &qemu_img,
        &[
            "convert",
            "--image-opts",
            &format!(
                "driver=luks,key-secret=sec0,file.filename={},header.driver=file,header.filename={}",
                path_str(&luks),
                path_str(&hdr)
            ),
            "-O",
            "raw",
            path_str(&out),
        ],
    );
    assert!(fs::read(&out).unwrap() == data);
}

#[test]
fn create_errors() {
    setup();
    let dir = scratch("create-errors");
    let g = BlockGraph::new();
    let e = g
        .blockdev_create(from_json::<BlockdevCreateOptions>(r#"{"driver": "luks", "size": 0}"#))
        .unwrap_err();
    assert_eq!(e.message(), "Either the parameter 'header' or 'file' must be specified");
    let hdr = dir.join("h");
    create_file(&g, &hdr);
    let e = g
        .blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
            r#"{{"driver": "luks", "size": 0, "preallocation": "full",
                "header": {{"driver": "file", "filename": "{}"}}}}"#,
            path_str(&hdr)
        )))
        .unwrap_err();
    assert_eq!(
        e.message(),
        "Parameter 'preallocation' requires 'file' to be specified for formatting LUKS disk"
    );
    let e = g
        .blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
            r#"{{"driver": "luks", "size": 0, "file": {{"driver": "file", "filename": "{}"}}}}"#,
            path_str(&hdr)
        )))
        .unwrap_err();
    assert_eq!(e.message(), "Parameter 'key-secret' is required for cipher");

    // Not a LUKS image.
    fs::write(dir.join("plain"), vec![0u8; 4096]).unwrap();
    let e = add_luks(&g, "p", &dir.join("plain"), None, "sec0").unwrap_err();
    assert_eq!(e.message(), "Volume is not in LUKS format");
}

#[test]
fn cryptsetup_sees_luks() {
    setup();
    let Ok(out) = Command::new("cryptsetup").arg("--version").output() else {
        eprintln!("cryptsetup not found, skipping");
        return;
    };
    if !out.status.success() {
        eprintln!("cryptsetup not usable, skipping");
        return;
    }
    let dir = scratch("cryptsetup");
    let luks = dir.join("disk.luks");
    let g = BlockGraph::new();
    create_file(&g, &luks);
    g.blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
        r#"{{"driver": "luks", "key-secret": "sec0", "iter-time": 10,
            "file": {{"driver": "file", "filename": "{}"}}, "size": 1048576}}"#,
        path_str(&luks)
    )))
    .unwrap();
    let st = Command::new("cryptsetup").arg("isLuks").arg(&luks).status().unwrap();
    assert!(st.success());
    let out = Command::new("cryptsetup").arg("luksDump").arg(&luks).output().unwrap();
    let dump = String::from_utf8_lossy(&out.stdout);
    assert!(dump.contains("aes"), "{dump}");
}
