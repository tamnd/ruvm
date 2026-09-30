// SPDX-License-Identifier: GPL-2.0-or-later

//! The `cloop` format driver end to end: a V2.0 image built by hand (script header, block size,
//! block count, offsets, zlib blocks) read back through `blockdev-add` and probing, compared
//! with `qemu-img convert`, and every open error of QEMU checked, with qemu-img's text when it
//! is there.

#![cfg(unix)]

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use flate2::Compression;
use flate2::write::ZlibEncoder;
use ruvm_block::{BLK_PERM_CONSISTENT_READ, BlockBackend, BlockGraph};
use ruvm_qapi::QDict;
use ruvm_qapi::json;
use ruvm_qapi::types::BlockdevOptions;
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};

const ALL: u64 = 0x1f;

const MAGIC: &[u8] =
    b"#!/bin/sh\n#V2.0 Format\nmodprobe cloop file=$0 && mount -r -t iso9660 /dev/cloop $1\n";

fn qemu_img() -> Option<PathBuf> {
    let candidates = ["/opt/homebrew/bin/qemu-img", "/usr/local/bin/qemu-img", "/usr/bin/qemu-img"];
    let found = candidates.iter().map(PathBuf::from).find(|p| p.exists());
    if found.is_none() {
        eprintln!("qemu-img not found, skipping the qemu-img comparisons");
    }
    found
}

fn scratch(test: &str) -> PathBuf {
    let base = option_env!("CARGO_TARGET_TMPDIR").map_or_else(std::env::temp_dir, PathBuf::from);
    let dir = base.join("ruvm-block-cloop").join(test);
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

fn add_cloop(g: &BlockGraph, path: &Path, read_only: bool) -> ruvm_base::Result<()> {
    g.blockdev_add(from_json::<BlockdevOptions>(&format!(
        r#"{{"driver": "cloop", "node-name": "c", "read-only": {read_only},
            "file": {{"driver": "file", "filename": "{}"}}}}"#,
        path_str(path)
    )))
}

/// The disk contents: `n` blocks of `block_size`, a mix of text, noise and zeroes.
fn contents(block_size: usize, n: usize) -> Vec<u8> {
    let mut x: u32 = 7;
    (0..block_size * n)
        .map(|i| match (i / block_size) % 3 {
            0 => b"cloop test data "[i % 16],
            1 => {
                x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
                (x >> 16) as u8
            }
            _ => 0,
        })
        .collect()
}

/// A cloop image of `data` in blocks of `block_size`.
fn build(data: &[u8], block_size: u32) -> Vec<u8> {
    let n_blocks = data.len() / block_size as usize;
    let mut blocks = Vec::new();
    for b in data.chunks(block_size as usize) {
        let mut z = ZlibEncoder::new(Vec::new(), Compression::best());
        z.write_all(b).unwrap();
        blocks.push(z.finish().unwrap());
    }
    let mut img = vec![0u8; 128];
    img[..MAGIC.len()].copy_from_slice(MAGIC);
    img.extend_from_slice(&block_size.to_be_bytes());
    img.extend_from_slice(&(n_blocks as u32).to_be_bytes());
    let mut offset = (136 + 8 * (n_blocks + 1)) as u64;
    img.extend_from_slice(&offset.to_be_bytes());
    for b in &blocks {
        offset += b.len() as u64;
        img.extend_from_slice(&offset.to_be_bytes());
    }
    for b in &blocks {
        img.extend_from_slice(b);
    }
    img
}

#[test]
fn read_image() {
    let dir = scratch("read");
    let data = contents(4096, 10);
    let img = dir.join("disk.cloop");
    fs::write(&img, build(&data, 4096)).unwrap();

    let g = BlockGraph::new();
    add_cloop(&g, &img, true).unwrap();
    let blk = BlockBackend::new(&g, "c", BLK_PERM_CONSISTENT_READ, ALL).unwrap();
    assert_eq!(blk.getlength().unwrap(), data.len() as u64);
    let mut buf = vec![0u8; data.len()];
    blk.pread(0, &mut buf).unwrap();
    assert!(buf == data);

    // Unaligned, crossing blocks, going back to an earlier block.
    for (off, len) in [(5000usize, 9000usize), (100, 50), (40_000, 960)] {
        let mut b = vec![0u8; len];
        blk.pread(off as u64, &mut b).unwrap();
        assert!(b[..] == data[off..off + len], "{off}+{len}");
    }
    drop(blk);
    drop(g);

    // Probing finds cloop.
    let mut opts = QDict::new();
    opts.put("read-only", "on");
    let g = BlockGraph::new();
    let name = g.open_image(Some(path_str(&img)), opts).unwrap();
    assert_eq!(g.node(&name).unwrap().driver, "cloop");
    drop(g);

    // Opening read-write without auto-read-only fails.
    let g = BlockGraph::new();
    assert_eq!(add_cloop(&g, &img, false).unwrap_err().message(), "Image is read-only");

    if let Some(q) = qemu_img() {
        let raw = dir.join("qemu.raw");
        run(&q, &["convert", "-f", "cloop", "-O", "raw", path_str(&img), path_str(&raw)]);
        assert!(fs::read(&raw).unwrap() == data);
    }
}

#[test]
fn corrupt_block() {
    let dir = scratch("corrupt");
    let data = contents(1024, 4);
    let mut bytes = build(&data, 1024);
    // Flip a byte in the last block; its adler32 no longer matches.
    let n = bytes.len();
    bytes[n - 3] ^= 0xff;
    let img = dir.join("disk.cloop");
    fs::write(&img, bytes).unwrap();

    let g = BlockGraph::new();
    add_cloop(&g, &img, true).unwrap();
    let blk = BlockBackend::new(&g, "c", BLK_PERM_CONSISTENT_READ, ALL).unwrap();
    let mut b = vec![0u8; 1024];
    blk.pread(0, &mut b).unwrap();
    assert!(b[..] == data[..1024]);
    let e = blk.pread(3 * 1024, &mut b).unwrap_err();
    assert_eq!(e.raw_os_error(), Some(libc::EIO));
}

#[test]
fn open_errors() {
    let dir = scratch("errors");
    let qemu_img = qemu_img();
    let good = build(&contents(1024, 4), 1024);

    let with_header = |block_size: u32, n_blocks: u32| {
        let mut b = good.clone();
        b[128..132].copy_from_slice(&block_size.to_be_bytes());
        b[132..136].copy_from_slice(&n_blocks.to_be_bytes());
        b
    };
    let with_offset = |i: usize, v: u64| {
        let mut b = good.clone();
        b[136 + 8 * i..144 + 8 * i].copy_from_slice(&v.to_be_bytes());
        b
    };

    let cases: Vec<(&str, Vec<u8>, String)> = vec![
        ("multiple", with_header(1000, 4), "block_size 1000 must be a multiple of 512".into()),
        ("zero", with_header(0, 4), "block_size cannot be zero".into()),
        ("big", with_header(128 << 20, 4), "block_size 134217728 must be 64 MB or less".into()),
        (
            "nblocks",
            with_header(1024, 0x2000_0000),
            "n_blocks 536870912 must be 536870911 or less".into(),
        ),
        (
            "offsets",
            with_header(1024, 0x0400_0000),
            "image requires too many offsets, try increasing block size".into(),
        ),
        (
            "monotonic",
            with_offset(2, 0),
            "offsets not monotonically increasing at index 2, image file is corrupt".into(),
        ),
        (
            "size",
            with_offset(4, 1 << 40),
            "invalid compressed block size at index 4, image file is corrupt".into(),
        ),
    ];
    for (name, bytes, msg) in cases {
        let img = dir.join(format!("{name}.cloop"));
        fs::write(&img, bytes).unwrap();
        let g = BlockGraph::new();
        let e = add_cloop(&g, &img, true).unwrap_err();
        assert_eq!(e.message(), msg, "{name}");
        if let Some(q) = &qemu_img {
            let err = run_fail(q, &["info", "-f", "cloop", path_str(&img)]);
            assert!(err.contains(&msg), "{name}: qemu-img said {err}");
        }
    }

    // A file cut short after the header: reads past the end see zeroes, so as in QEMU the
    // offsets are all zero, the open succeeds and reading a block fails.
    let img = dir.join("short.cloop");
    fs::write(&img, &good[..140]).unwrap();
    let g = BlockGraph::new();
    add_cloop(&g, &img, true).unwrap();
    let blk = BlockBackend::new(&g, "c", BLK_PERM_CONSISTENT_READ, ALL).unwrap();
    assert_eq!(blk.getlength().unwrap(), 4096);
    let e = blk.pread(0, &mut [0u8; 512]).unwrap_err();
    assert_eq!(e.raw_os_error(), Some(libc::EIO));
    if let Some(q) = &qemu_img {
        let raw = dir.join("short.raw");
        let err =
            run_fail(q, &["convert", "-f", "cloop", "-O", "raw", path_str(&img), path_str(&raw)]);
        assert!(err.contains("Input/output error"), "{err}");
    }
}
