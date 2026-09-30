// SPDX-License-Identifier: GPL-2.0-or-later

//! The `bochs` format driver end to end: growing redolog images built by hand in the current
//! and the v1 header layouts, with unallocated extents, extents stored out of order and
//! sectors whose bitmap bit is clear, read back through `blockdev-add` and probing and compared
//! with `qemu-img convert`. The open errors of QEMU are checked, with qemu-img's text when it
//! is there.

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

fn scratch(test: &str) -> PathBuf {
    let base = option_env!("CARGO_TARGET_TMPDIR").map_or_else(std::env::temp_dir, PathBuf::from);
    let dir = base.join("ruvm-block-bochs").join(test);
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

fn add_bochs(g: &BlockGraph, path: &Path, read_only: bool) -> ruvm_base::Result<()> {
    g.blockdev_add(from_json::<BlockdevOptions>(&format!(
        r#"{{"driver": "bochs", "node-name": "b", "read-only": {read_only},
            "file": {{"driver": "file", "filename": "{}"}}}}"#,
        path_str(path)
    )))
}

/// Header fields of a test image.
#[derive(Clone, Copy)]
struct Params {
    v1: bool,
    catalog: u32,
    bitmap: u32,
    extent: u32,
    disk: u64,
}

impl Default for Params {
    fn default() -> Self {
        Params { v1: false, catalog: 16, bitmap: 1, extent: 4096, disk: 64 * 1024 }
    }
}

fn header(p: Params) -> Vec<u8> {
    let mut h = vec![0u8; 512];
    h[..22].copy_from_slice(b"Bochs Virtual HD Image");
    h[32..39].copy_from_slice(b"Redolog");
    h[48..55].copy_from_slice(b"Growing");
    let version: u32 = if p.v1 { 0x10000 } else { 0x20000 };
    h[64..68].copy_from_slice(&version.to_le_bytes());
    h[68..72].copy_from_slice(&512u32.to_le_bytes());
    h[72..76].copy_from_slice(&p.catalog.to_le_bytes());
    h[76..80].copy_from_slice(&p.bitmap.to_le_bytes());
    h[80..84].copy_from_slice(&p.extent.to_le_bytes());
    let at = if p.v1 { 84 } else { 88 };
    h[at..at + 8].copy_from_slice(&p.disk.to_le_bytes());
    h
}

/// A 64 KiB disk of 16 extents of 8 sectors. Every third extent is unallocated, the others
/// are stored in reverse order, and in each stored extent sector 3 has its bitmap bit clear.
/// Returns the image and the disk contents it should read as.
fn build(v1: bool) -> (Vec<u8>, Vec<u8>) {
    let p = Params { v1, ..Params::default() };
    let n_extents = 16u32;
    let allocated: Vec<u32> = (0..n_extents).filter(|e| e % 3 != 1).collect();
    let n_stored = allocated.len() as u32;

    let mut catalog = vec![0xffff_ffffu32; n_extents as usize];
    for (k, &e) in allocated.iter().enumerate() {
        catalog[e as usize] = n_stored - 1 - k as u32;
    }

    let mut img = header(p);
    for c in &catalog {
        img.extend_from_slice(&c.to_le_bytes());
    }
    let data_offset = img.len();
    // Each stored extent: one bitmap sector, then 8 data sectors.
    img.resize(data_offset + n_stored as usize * 9 * 512, 0);

    let mut disk = vec![0u8; p.disk as usize];
    for &e in &allocated {
        let stored = catalog[e as usize] as usize;
        let base = data_offset + stored * 9 * 512;
        img[base] = !(1u8 << 3);
        for s in 0..8usize {
            let sector: Vec<u8> =
                (0..512).map(|i| (e as usize * 8 + s + i * 7) as u8 | 1).collect();
            img[base + 512 * (1 + s)..][..512].copy_from_slice(&sector);
            if s != 3 {
                disk[(e as usize * 8 + s) * 512..][..512].copy_from_slice(&sector);
            }
        }
    }
    (img, disk)
}

#[test]
fn read_images() {
    let dir = scratch("read");
    let qemu_img = qemu_img();
    for v1 in [false, true] {
        let (bytes, data) = build(v1);
        let img = dir.join(format!("disk-v1-{v1}.bochs"));
        fs::write(&img, bytes).unwrap();

        let g = BlockGraph::new();
        add_bochs(&g, &img, true).unwrap();
        let blk = BlockBackend::new(&g, "b", BLK_PERM_CONSISTENT_READ, ALL).unwrap();
        assert_eq!(blk.getlength().unwrap(), data.len() as u64);
        let mut buf = vec![0u8; data.len()];
        blk.pread(0, &mut buf).unwrap();
        assert!(buf == data, "v1 {v1}");
        let mut b = vec![0u8; 5000];
        blk.pread(1234, &mut b).unwrap();
        assert!(b[..] == data[1234..6234], "v1 {v1}");
        drop(blk);
        drop(g);

        let mut opts = QDict::new();
        opts.put("read-only", "on");
        let g = BlockGraph::new();
        let name = g.open_image(Some(path_str(&img)), opts).unwrap();
        assert_eq!(g.node(&name).unwrap().driver, "bochs");
        drop(g);

        let g = BlockGraph::new();
        assert_eq!(add_bochs(&g, &img, false).unwrap_err().message(), "Image is read-only");

        if let Some(q) = &qemu_img {
            let raw = dir.join(format!("qemu-v1-{v1}.raw"));
            run(q, &["convert", "-f", "bochs", "-O", "raw", path_str(&img), path_str(&raw)]);
            assert!(fs::read(&raw).unwrap() == data, "v1 {v1}");
        }
    }
}

#[test]
fn open_errors() {
    let dir = scratch("errors");
    let qemu_img = qemu_img();
    let d = Params::default();

    let mut bad_magic = header(d);
    bad_magic[0] = b'b';
    let mut bad_subtype = header(d);
    bad_subtype[55] = b'X';
    let mut bad_version = header(d);
    bad_version[64..68].copy_from_slice(&0x30000u32.to_le_bytes());

    let cases: Vec<(&str, Vec<u8>, String)> = vec![
        ("magic", bad_magic, "Image not in Bochs format".into()),
        ("subtype", bad_subtype, "Image not in Bochs format".into()),
        ("version", bad_version, "Image not in Bochs format".into()),
        ("catalog", header(Params { catalog: 0x10_0001, ..d }), "Catalog size is too large".into()),
        ("small", header(Params { extent: 256, ..d }), "Extent size must be at least 512".into()),
        (
            "pow2",
            header(Params { extent: 1536, ..d }),
            "Extent size 1536 is not a power of two".into(),
        ),
        (
            "large",
            header(Params { extent: 1 << 24, ..d }),
            "Extent size 16777216 is too large".into(),
        ),
        (
            "toosmall",
            header(Params { disk: 1 << 20, ..d }),
            "Catalog size is too small for this disk size".into(),
        ),
    ];
    for (name, mut bytes, msg) in cases {
        bytes.resize(4096, 0);
        let img = dir.join(format!("{name}.bochs"));
        fs::write(&img, bytes).unwrap();
        let g = BlockGraph::new();
        let e = add_bochs(&g, &img, true).unwrap_err();
        assert_eq!(e.message(), msg, "{name}");
        if let Some(q) = &qemu_img {
            let err = run_fail(q, &["info", "-f", "bochs", path_str(&img)]);
            assert!(err.contains(&msg), "{name}: qemu-img said {err}");
        }
    }
}
