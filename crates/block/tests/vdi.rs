// SPDX-License-Identifier: GPL-2.0-or-later

//! The `vdi` format driver against QEMU: images `qemu-img` makes and `qemu-io` writes read the
//! same here, images made here (with `qemu-img create` style options and with
//! `blockdev-create`) and written here pass `qemu-img check` and compare equal to a raw copy,
//! and the virtual sizes, probing, cluster sizes and check results agree. The interop tests
//! skip themselves when `qemu-img` or `qemu-io` is not installed.

#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use ruvm_block::{
    BDRV_FIX_ERRORS, BLK_PERM_CONSISTENT_READ, BLK_PERM_WRITE, BlockBackend, BlockGraph,
};
use ruvm_qapi::types::{BlockdevCreateOptions, BlockdevOptions};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};
use ruvm_qapi::{QDict, json};

const RW: u64 = BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE;
const SHARED: u64 = BLK_PERM_CONSISTENT_READ;
const MIB: usize = 1 << 20;

/// `name` from homebrew or the system, or `None` (and a note) when it is not installed.
fn tool(name: &str) -> Option<PathBuf> {
    let found = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"]
        .iter()
        .map(|d| Path::new(d).join(name))
        .find(|p| p.exists());
    if found.is_none() {
        eprintln!("{name} not found, skipping");
    }
    found
}

/// `qemu-img` and `qemu-io`, when both are installed.
fn tools() -> Option<(PathBuf, PathBuf)> {
    Some((tool("qemu-img")?, tool("qemu-io")?))
}

fn scratch(test: &str) -> PathBuf {
    let base = option_env!("CARGO_TARGET_TMPDIR").map_or_else(std::env::temp_dir, PathBuf::from);
    let dir = base.join("ruvm-block-vdi").join(test);
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

/// Runs `prog` with `args` and returns the exit code and stdout.
fn run_status(prog: &Path, args: &[&str]) -> (i32, String) {
    let out = Command::new(prog).args(args).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    (out.status.code().unwrap_or(-1), text + &String::from_utf8_lossy(&out.stderr))
}

/// Runs `prog` with `args`, which must succeed, and returns stdout.
fn run(prog: &Path, args: &[&str]) -> String {
    let (code, out) = run_status(prog, args);
    assert_eq!(code, 0, "{} {args:?} failed: {out}", prog.display());
    out
}

/// The number after `"key": ` in `qemu-img info --output=json` output, the top level one.
fn info_num(info: &str, key: &str) -> Option<u64> {
    let pat = format!("\"{key}\": ");
    let i = info.rfind(&pat)? + pat.len();
    let digits: String = info[i..].chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

fn from_json<T: Visit + Default>(s: &str) -> T {
    let mut v = QObjectInputVisitor::new(json::from_str(s).unwrap());
    let mut o = T::default();
    T::visit(&mut v, None, &mut o).unwrap();
    o
}

/// `blockdev-add` of a vdi node `v` over `path`.
fn add_vdi(g: &BlockGraph, path: &Path) -> ruvm_base::Result<()> {
    g.blockdev_add(from_json::<BlockdevOptions>(&format!(
        r#"{{"driver": "vdi", "node-name": "v",
            "file": {{"driver": "file", "filename": "{}"}}}}"#,
        path_str(path)
    )))
}

fn read_all(blk: &BlockBackend) -> Vec<u8> {
    let len = blk.getlength().unwrap() as usize;
    let mut buf = vec![0u8; len];
    blk.pread(0, &mut buf).unwrap();
    buf
}

/// Probes `path` without a format and opens it as node `v`: the same format, virtual size and
/// cluster size as `qemu-img info` gives.
fn open_and_compare(g: &BlockGraph, qemu_img: &Path, path: &Path) {
    {
        let p = BlockGraph::new();
        let name = p.open_image(Some(path_str(path)), QDict::new()).unwrap();
        assert_eq!(p.node(&name).unwrap().driver, "vdi");
    }
    let info = run(qemu_img, &["info", "--output=json", path_str(path)]);
    assert!(info.contains("\"format\": \"vdi\""), "{info}");
    add_vdi(g, path).unwrap();
    let nodes = g.query_named_block_nodes(Some(true)).unwrap();
    let n = nodes.iter().find(|n| n.node_name == "v").unwrap();
    assert_eq!(n.image.virtual_size as u64, info_num(&info, "virtual-size").unwrap());
    assert_eq!(n.image.cluster_size.map(|c| c as u64), info_num(&info, "cluster-size"));
}

/// `qemu-img create` and `qemu-io` writes, then everything reads the same here, and writes
/// here pass `qemu-img check` and read back in `qemu-img`.
fn qemu_image(test: &str, create_opts: &str) {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    let dir = scratch(test);
    let img = dir.join("disk.vdi");
    run(&qemu_img, &["create", "-f", "vdi", "-o", create_opts, path_str(&img), "10M"]);
    run(
        &qemu_io,
        &[
            "-f",
            "vdi",
            "-c",
            "write -P 0x55 0 64k",
            "-c",
            "write -P 0xaa 3000k 1500k",
            "-c",
            "write -P 0x11 9M 512",
            path_str(&img),
        ],
    );
    let raw = dir.join("disk.raw");
    run(&qemu_img, &["convert", "-f", "vdi", "-O", "raw", path_str(&img), path_str(&raw)]);
    let mut expect = fs::read(&raw).unwrap();

    let g = BlockGraph::new();
    open_and_compare(&g, &qemu_img, &img);
    let blk = BlockBackend::new(&g, "v", RW, SHARED).unwrap();
    assert!(read_all(&blk) == expect);
    // One write in an allocated block, two over a block boundary into unallocated ones.
    for (off, len, seed) in [(1000usize, 3000usize, 1u8), (6 * MIB - 100, MIB + 200, 2)] {
        let data = pattern(len, seed);
        blk.pwrite(off as u64, &data).unwrap();
        expect[off..off + len].copy_from_slice(&data);
    }
    assert!(read_all(&blk) == expect);
    drop(blk);
    drop(g);
    fs::write(&raw, &expect).unwrap();
    run(&qemu_img, &["check", "-f", "vdi", path_str(&img)]);
    run(&qemu_img, &["compare", "-f", "vdi", "-F", "raw", path_str(&img), path_str(&raw)]);
}

#[test]
fn qemu_dynamic_image() {
    qemu_image("qemu-dynamic", "static=off");
}

#[test]
fn qemu_static_image() {
    qemu_image("qemu-static", "static=on");
}

/// Writes patterns here to node `v` over `img`, then `qemu-img check` and `qemu-img compare`
/// against a raw copy.
fn write_and_compare(g: &BlockGraph, qemu_img: &Path, img: &Path, dir: &Path) {
    open_and_compare(g, qemu_img, img);
    let blk = BlockBackend::new(g, "v", RW, SHARED).unwrap();
    let len = blk.getlength().unwrap() as usize;
    let mut expect = vec![0u8; len];
    for (off, n, seed) in
        [(0usize, 4096usize, 1u8), (MIB, 3 * MIB, 2), (len - 5000, 5000, 3), (777, 10, 4)]
    {
        let data = pattern(n, seed);
        blk.pwrite(off as u64, &data).unwrap();
        expect[off..off + n].copy_from_slice(&data);
    }
    assert!(read_all(&blk) == expect);
    let r = g.check("v", 0).unwrap();
    assert_eq!((r.corruptions, r.leaks, r.check_errors), (0, 0, 0));
    drop(blk);
    let raw = dir.join("expect.raw");
    fs::write(&raw, &expect).unwrap();
    run(qemu_img, &["check", "-f", "vdi", path_str(img)]);
    run(qemu_img, &["compare", "-f", "vdi", "-F", "raw", path_str(img), path_str(&raw)]);
}

#[test]
fn created_with_options() {
    let Some((qemu_img, _)) = tools() else { return };
    for is_static in ["off", "on"] {
        let dir = scratch(&format!("opts-static-{is_static}"));
        let img = dir.join("disk.vdi");
        let g = BlockGraph::new();
        let mut o = QDict::new();
        o.put("size", "10M");
        o.put("static", is_static);
        g.create_image("vdi", path_str(&img), &mut o).unwrap();

        // Byte for byte what qemu-img makes, except for the two random UUIDs.
        let theirs = dir.join("theirs.vdi");
        let opts = format!("static={is_static}");
        run(&qemu_img, &["create", "-f", "vdi", "-o", &opts, path_str(&theirs), "10M"]);
        let (mine, theirs) = (fs::read(&img).unwrap(), fs::read(&theirs).unwrap());
        assert_eq!(mine.len(), theirs.len());
        for (i, (x, y)) in mine.iter().zip(&theirs).enumerate() {
            assert!(x == y || (392..424).contains(&i), "byte {i} differs");
        }
        write_and_compare(&g, &qemu_img, &img, &dir);
    }
}

#[test]
fn created_with_blockdev_create() {
    let Some((qemu_img, _)) = tools() else { return };
    for (prealloc, size) in [("off", 10 * MIB), ("metadata", 10 * MIB), ("off", 5 * MIB + 512)] {
        let dir = scratch(&format!("create-{prealloc}-{size}"));
        let img = dir.join("disk.vdi");
        let g = BlockGraph::new();
        g.blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
            r#"{{"driver": "file", "filename": "{}", "size": 0}}"#,
            path_str(&img)
        )))
        .unwrap();
        g.blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
            r#"{{"driver": "vdi", "file": {{"driver": "file", "filename": "{}"}},
                "size": {size}, "preallocation": "{prealloc}"}}"#,
            path_str(&img)
        )))
        .unwrap();
        let info = run(&qemu_img, &["info", "--output=json", "-f", "vdi", path_str(&img)]);
        assert_eq!(info_num(&info, "virtual-size"), Some(size as u64));
        write_and_compare(&g, &qemu_img, &img, &dir);
    }
}

#[test]
fn create_errors() {
    let dir = scratch("create-errors");
    let img = dir.join("disk.vdi");
    let g = BlockGraph::new();
    let create = |extra: &str| {
        g.blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
            r#"{{"driver": "vdi", "file": {{"driver": "file", "filename": "{}"}}, {extra}}}"#,
            path_str(&img)
        )))
        .unwrap_err()
        .message()
        .to_string()
    };
    assert_eq!(
        create(r#""size": 1048576, "preallocation": "full""#),
        "Preallocation mode not supported for vdi"
    );
    assert_eq!(
        create(r#""size": 562949819203585"#),
        "Unsupported VDI image size (size is 0x1fffff8000001, max supported is \
         0x1fffff8000000)"
    );
}

#[test]
fn virtual_sizes_match_qemu() {
    let Some((qemu_img, _)) = tools() else { return };
    let dir = scratch("sizes");
    for size in ["1", "512", "1M", "1000001", "123456789", "3G"] {
        let mine = dir.join("mine.vdi");
        let theirs = dir.join("theirs.vdi");
        let _ = fs::remove_file(&mine);
        let g = BlockGraph::new();
        let mut o = QDict::new();
        o.put("size", size);
        g.create_image("vdi", path_str(&mine), &mut o).unwrap();
        run(&qemu_img, &["create", "-f", "vdi", path_str(&theirs), size]);
        let a = run(&qemu_img, &["info", "--output=json", path_str(&mine)]);
        let b = run(&qemu_img, &["info", "--output=json", path_str(&theirs)]);
        assert_eq!(info_num(&a, "virtual-size"), info_num(&b, "virtual-size"), "size {size}");
        assert_eq!(fs::metadata(&mine).unwrap().len(), fs::metadata(&theirs).unwrap().len());
        let name = g.open_image(Some(path_str(&mine)), QDict::new()).unwrap();
        assert_eq!(Some(g.node(&name).unwrap().size), info_num(&b, "virtual-size"));
    }
}

/// A corrupted bmap: `qemu-img check` and the check here find the same number of errors.
#[test]
fn check_finds_corruptions() {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    let dir = scratch("check");
    let img = dir.join("disk.vdi");
    run(&qemu_img, &["create", "-f", "vdi", path_str(&img), "4M"]);
    run(&qemu_io, &["-f", "vdi", "-c", "write 0 1M", "-c", "write 2M 1M", path_str(&img)]);
    // Block 1 points at the data of block 0, block 3 past the end.
    let mut bytes = fs::read(&img).unwrap();
    bytes[0x204..0x208].copy_from_slice(&0u32.to_le_bytes());
    bytes[0x20c..0x210].copy_from_slice(&9u32.to_le_bytes());
    fs::write(&img, &bytes).unwrap();

    let (code, out) = run_status(&qemu_img, &["check", "-f", "vdi", path_str(&img)]);
    assert_eq!(code, 2, "{out}");
    assert!(out.contains("3 errors were found on the image."), "{out}");

    let g = BlockGraph::new();
    add_vdi(&g, &img).unwrap();
    let r = g.check("v", 0).unwrap();
    assert_eq!(r.corruptions, 3);
    assert_eq!(
        g.check("v", BDRV_FIX_ERRORS).unwrap_err().message(),
        "This image format does not support checks"
    );
}

#[test]
fn open_errors() {
    let dir = scratch("open-errors");
    let open = |patch: &dyn Fn(&mut [u8])| {
        let mut h = vec![0u8; 1024];
        h[64..68].copy_from_slice(&0xbeda_107fu32.to_le_bytes());
        h[68..72].copy_from_slice(&0x0001_0001u32.to_le_bytes());
        h[340..344].copy_from_slice(&512u32.to_le_bytes());
        h[344..348].copy_from_slice(&1024u32.to_le_bytes());
        h[360..364].copy_from_slice(&512u32.to_le_bytes());
        h[368..376].copy_from_slice(&(1u64 << 20).to_le_bytes());
        h[376..380].copy_from_slice(&(1u32 << 20).to_le_bytes());
        h[384..388].copy_from_slice(&1u32.to_le_bytes());
        h[512..516].copy_from_slice(&u32::MAX.to_le_bytes());
        patch(&mut h);
        let p = dir.join("bad.vdi");
        fs::write(&p, &h).unwrap();
        let g = BlockGraph::new();
        add_vdi(&g, &p).err().map(|e| e.message().to_string())
    };
    assert_eq!(open(&|_| {}), None);
    assert_eq!(open(&|h| h[64] = 0).unwrap(), "Image not in VDI format (bad signature beda1000)");
    assert_eq!(
        open(&|h| h[68..72].copy_from_slice(&0x0001_0000u32.to_le_bytes())).unwrap(),
        "unsupported VDI image (version 1.0)"
    );
    assert_eq!(
        open(&|h| h[340] = 1).unwrap(),
        "unsupported VDI image (unaligned block map offset 0x201)"
    );
    assert_eq!(
        open(&|h| h[369] = 0x20).unwrap(),
        "unsupported VDI image (disk size 1056768, image bitmap has room for 1048576)"
    );
    assert_eq!(open(&|h| h[440] = 1).unwrap(), "unsupported VDI image (non-NULL parent UUID)");
    assert_eq!(
        open(&|h| h[376..380].copy_from_slice(&4096u32.to_le_bytes())).unwrap(),
        "unsupported VDI image (block size 4096 is not 1048576)"
    );
}
