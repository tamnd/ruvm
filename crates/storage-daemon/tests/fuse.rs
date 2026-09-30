// SPDX-License-Identifier: GPL-2.0-or-later

//! The FUSE export through a real mount: reads and writes through the mount point, the size,
//! growable and read-only exports. The tests that need a mount print a note and pass when there is
//! no `/dev/fuse`, or neither root nor `fusermount3`, or the mount is refused anyway.

#![cfg(target_os = "linux")]

use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ruvm_block::{
    BLK_PERM_ALL, BLK_PERM_CONSISTENT_READ, BLK_PERM_WRITE, BlockBackend, BlockGraph,
};
use ruvm_qapi::types::{BlockdevOptions, FuseExportAllowOther};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};
use ruvm_storage_daemon::export::fuse::{FuseExport, create_export};
use ruvm_storage_daemon::export::{BlockExportOptionsFuse, ExportArgs, ExportDriver};

const LEN: usize = 1 << 20;

fn scratch(test: &str) -> PathBuf {
    let base = option_env!("CARGO_TARGET_TMPDIR").map_or_else(std::env::temp_dir, PathBuf::from);
    let dir = base.join("ruvm-fuse").join(test);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

/// Whether FUSE mounts can be tried here, with a note when they cannot.
fn fuse_available() -> bool {
    if !Path::new("/dev/fuse").exists() {
        eprintln!("/dev/fuse not found, skipping");
        return false;
    }
    if rustix::process::getuid().is_root() {
        return true;
    }
    let found = std::env::var_os("PATH")
        .is_some_and(|p| std::env::split_paths(&p).any(|d| d.join("fusermount3").exists()));
    if !found {
        eprintln!("not root and fusermount3 not found, skipping");
    }
    found
}

/// A graph with a `file` node `f` over a new image of [`LEN`] bytes of [`pattern`].
fn image(dir: &Path) -> (BlockGraph, PathBuf) {
    let img = dir.join("img");
    fs::write(&img, pattern(LEN)).unwrap();
    let g = BlockGraph::new();
    let json =
        format!(r#"{{"driver": "file", "node-name": "f", "filename": "{}"}}"#, img.display());
    let mut v = QObjectInputVisitor::new(ruvm_qapi::json::from_str(&json).unwrap());
    let mut o = BlockdevOptions::default();
    BlockdevOptions::visit(&mut v, None, &mut o).unwrap();
    g.blockdev_add(o).unwrap();
    (g, img)
}

/// The backend `blk_exp_add()` makes.
fn backend(g: &BlockGraph, writable: bool) -> Arc<BlockBackend> {
    let mut perm = BLK_PERM_CONSISTENT_READ;
    if writable {
        perm |= BLK_PERM_WRITE;
    }
    BlockBackend::new(g, "f", perm, BLK_PERM_ALL).unwrap()
}

fn opts(mountpoint: &Path, growable: bool) -> BlockExportOptionsFuse {
    BlockExportOptionsFuse {
        mountpoint: mountpoint.to_str().unwrap().to_string(),
        growable: Some(growable),
        allow_other: Some(FuseExportAllowOther::Off),
    }
}

fn args<'a>(g: &'a BlockGraph, blk: &Arc<BlockBackend>, writable: bool) -> ExportArgs<'a> {
    ExportArgs { graph: g, id: "exp", node_name: "f", blk: blk.clone(), writable }
}

/// Exports `f` on a new file `mnt` in `dir`, or `None` (and a note) when this host cannot mount.
fn export(
    dir: &Path,
    g: &BlockGraph,
    writable: bool,
    growable: bool,
) -> Option<(Arc<FuseExport>, PathBuf, Arc<BlockBackend>)> {
    if !fuse_available() {
        return None;
    }
    let mnt = dir.join("mnt");
    fs::write(&mnt, b"").unwrap();
    let blk = backend(g, writable);
    match create_export(&args(g, &blk, writable), &opts(&mnt, growable)) {
        Ok(e) => Some((e, mnt, blk)),
        Err(e) if e.message() == "Failed to mount FUSE session to export" => {
            eprintln!("{e}, skipping");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

#[test]
fn read_write() {
    let dir = scratch("read_write");
    let (g, img) = image(&dir);
    let Some((exp, mnt, _blk)) = export(&dir, &g, true, false) else { return };

    assert_eq!(fs::metadata(&mnt).unwrap().len(), LEN as u64);
    assert_eq!(fs::read(&mnt).unwrap(), pattern(LEN));

    let f = OpenOptions::new().read(true).write(true).open(&mnt).unwrap();
    f.write_all_at(&[0x5a; 4096], 8192).unwrap();
    f.sync_all().unwrap();
    let mut buf = vec![0u8; 4096];
    f.read_exact_at(&mut buf, 8192).unwrap();
    assert!(buf.iter().all(|&b| b == 0x5a));

    // Not growable: writes stop at the end of the image.
    let mut f2 = OpenOptions::new().write(true).open(&mnt).unwrap();
    f2.seek(SeekFrom::Start(LEN as u64 - 10)).unwrap();
    assert_eq!(f2.write(&[1; 100]).unwrap(), 10);
    assert_eq!(f2.write(&[1; 100]).unwrap(), 0);
    drop((f, f2));
    assert_eq!(fs::metadata(&mnt).unwrap().len(), LEN as u64);

    exp.shutdown();
    assert!(!exp.halted());
    // The mount point is the empty file again and the image has the data.
    assert_eq!(fs::metadata(&mnt).unwrap().len(), 0);
    let data = fs::read(&img).unwrap();
    assert!(data[8192..12288].iter().all(|&b| b == 0x5a));
    assert_eq!(&data[LEN - 10..], &[1; 10]);
}

#[test]
fn size() {
    let dir = scratch("size");
    let (g, _img) = image(&dir);
    let Some((exp, mnt, blk)) = export(&dir, &g, true, false) else { return };

    let f = OpenOptions::new().write(true).open(&mnt).unwrap();
    f.set_len(LEN as u64 / 2).unwrap();
    assert_eq!(fs::metadata(&mnt).unwrap().len(), LEN as u64 / 2);
    assert_eq!(blk.getlength().unwrap(), LEN as u64 / 2);
    f.set_len(LEN as u64 * 2).unwrap();
    assert_eq!(blk.getlength().unwrap(), LEN as u64 * 2);
    drop(f);
    let mut tail = Vec::new();
    let mut r = fs::File::open(&mnt).unwrap();
    r.seek(SeekFrom::Start(LEN as u64 * 2 - 4096)).unwrap();
    r.read_to_end(&mut tail).unwrap();
    assert_eq!(tail.len(), 4096);
    drop(r);
    exp.shutdown();
}

#[test]
fn growable() {
    let dir = scratch("growable");
    let (g, img) = image(&dir);
    let Some((exp, mnt, blk)) = export(&dir, &g, true, true) else { return };

    let f = OpenOptions::new().write(true).open(&mnt).unwrap();
    f.write_all_at(&[9; 1000], LEN as u64 + 100).unwrap();
    drop(f);
    // The file grows to exactly LEN + 1100, but a node's length is counted in whole sectors,
    // so bdrv_getlength() and the size FUSE reports round up to 512, as they do in QEMU.
    let sectors = (LEN as u64 + 1100).next_multiple_of(512);
    assert_eq!(blk.getlength().unwrap(), sectors);
    assert_eq!(fs::metadata(&mnt).unwrap().len(), sectors);
    exp.shutdown();
    let data = fs::read(&img).unwrap();
    assert_eq!(data.len(), LEN + 1100);
    assert_eq!(&data[LEN + 100..], &[9; 1000][..]);
}

#[test]
fn read_only() {
    let dir = scratch("read_only");
    let (g, _img) = image(&dir);
    let Some((exp, mnt, _blk)) = export(&dir, &g, false, false) else { return };

    let md = fs::metadata(&mnt).unwrap();
    assert_eq!(md.len(), LEN as u64);
    assert!(md.permissions().readonly());
    assert_eq!(&fs::read(&mnt).unwrap()[..4096], &pattern(4096)[..]);
    assert!(OpenOptions::new().write(true).open(&mnt).is_err());
    exp.shutdown();
}

#[test]
fn same_mountpoint_twice() {
    let dir = scratch("twice");
    let (g, _img) = image(&dir);
    let Some((exp, mnt, _blk)) = export(&dir, &g, false, false) else { return };
    let blk = backend(&g, false);
    let err = create_export(&args(&g, &blk, false), &opts(&mnt, false)).unwrap_err();
    assert_eq!(err.message(), format!("There already is a FUSE export on '{}'", mnt.display()));
    exp.shutdown();
}

#[test]
fn bad_mountpoint() {
    let dir = scratch("bad_mountpoint");
    let (g, _img) = image(&dir);
    let blk = backend(&g, false);

    let missing = dir.join("missing");
    let err = create_export(&args(&g, &blk, false), &opts(&missing, false)).unwrap_err();
    assert_eq!(
        err.message(),
        format!("Failed to stat '{}': No such file or directory", missing.display())
    );

    let err = create_export(&args(&g, &blk, false), &opts(&dir, false)).unwrap_err();
    assert_eq!(err.message(), format!("'{}' is not a regular file", dir.display()));
}
