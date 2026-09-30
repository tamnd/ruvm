// SPDX-License-Identifier: GPL-2.0-or-later

//! The `vpc` format driver against QEMU: images `qemu-img` makes and `qemu-io` writes read the
//! same here, images made here (with `qemu-img create` style options and with
//! `blockdev-create`) and written here compare equal to a raw copy in `qemu-img`, and the
//! virtual sizes, probing and cluster sizes agree. The interop tests skip themselves when
//! `qemu-img` or `qemu-io` is not installed.

#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use ruvm_block::{BLK_PERM_CONSISTENT_READ, BLK_PERM_WRITE, BlockBackend, BlockGraph};
use ruvm_qapi::types::{BlockdevCreateOptions, BlockdevOptions};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};
use ruvm_qapi::{QDict, json};

const RW: u64 = BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE;
const SHARED: u64 = BLK_PERM_CONSISTENT_READ;

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
    let dir = base.join("ruvm-block-vpc").join(test);
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

/// Runs `prog` with `args`, which must succeed, and returns stdout.
fn run(prog: &Path, args: &[&str]) -> String {
    let out = Command::new(prog).args(args).output().unwrap();
    assert!(
        out.status.success(),
        "{} {args:?} failed: {}{}",
        prog.display(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

/// The number after `"key": ` in `qemu-img info --output=json` output.
fn info_num(info: &str, key: &str) -> Option<u64> {
    let pat = format!("\"{key}\": ");
    // The top level entry is the last one, after those of the children.
    let i = info.rfind(&pat)? + pat.len();
    let digits: String = info[i..].chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

fn qemu_info(qemu_img: &Path, path: &Path) -> String {
    run(qemu_img, &["info", "--output=json", "-f", "vpc", path_str(path)])
}

fn from_json<T: Visit + Default>(s: &str) -> T {
    let mut v = QObjectInputVisitor::new(json::from_str(s).unwrap());
    let mut o = T::default();
    T::visit(&mut v, None, &mut o).unwrap();
    o
}

/// `blockdev-add` of a vpc node `name` over `path`.
fn add_vpc(g: &BlockGraph, name: &str, path: &Path) {
    g.blockdev_add(from_json::<BlockdevOptions>(&format!(
        r#"{{"driver": "vpc", "node-name": "{name}",
            "file": {{"driver": "file", "filename": "{}"}}}}"#,
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

/// The string after `"key": ` in `qemu-img info --output=json` output.
fn info_str(info: &str, key: &str) -> String {
    let pat = format!("\"{key}\": \"");
    let i = info.rfind(&pat).unwrap() + pat.len();
    info[i..].split('"').next().unwrap().to_string()
}

/// Probes `path` the way `qemu-img` does without `-f`: the same format and virtual size.
/// Fixed images have no header at the start, so both take them for raw.
fn probe_and_compare(qemu_img: &Path, path: &Path) {
    let g = BlockGraph::new();
    let name = g.open_image(Some(path_str(path)), QDict::new()).unwrap();
    let info = run(qemu_img, &["info", "--output=json", path_str(path)]);
    assert_eq!(g.node(&name).unwrap().driver, info_str(&info, "format"));
    assert_eq!(Some(g.node(&name).unwrap().size), info_num(&info, "virtual-size"));
}

/// Opens `path` as vpc and checks the virtual size and the cluster size against
/// `qemu-img info -f vpc`; returns the node name.
fn open_and_compare(g: &BlockGraph, qemu_img: &Path, path: &Path) -> String {
    add_vpc(g, "v", path);
    let info = qemu_info(qemu_img, path);
    let nodes = g.query_named_block_nodes(Some(true)).unwrap();
    let n = nodes.iter().find(|n| n.node_name == "v").unwrap();
    assert_eq!(n.image.virtual_size as u64, info_num(&info, "virtual-size").unwrap());
    assert_eq!(n.image.cluster_size.map(|c| c as u64), info_num(&info, "cluster-size"));
    "v".to_string()
}

/// `qemu-img create` and `qemu-io` writes, then everything reads the same here, and a write
/// here reads back in `qemu-img`.
fn qemu_image(test: &str, create_opts: &str, size: &str) {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    let dir = scratch(test);
    let img = dir.join("disk.vhd");
    let mut args = vec!["create", "-f", "vpc"];
    if !create_opts.is_empty() {
        args.extend(["-o", create_opts]);
    }
    args.extend([path_str(&img), size]);
    run(&qemu_img, &args);
    run(
        &qemu_io,
        &[
            "-f",
            "vpc",
            "-c",
            "write -P 0x55 0 64k",
            "-c",
            "write -P 0xaa 3000k 1500k",
            "-c",
            "write -P 0x11 5M 512",
            path_str(&img),
        ],
    );
    let raw = dir.join("disk.raw");
    run(&qemu_img, &["convert", "-f", "vpc", "-O", "raw", path_str(&img), path_str(&raw)]);
    let expect = fs::read(&raw).unwrap();

    probe_and_compare(&qemu_img, &img);
    let g = BlockGraph::new();
    let name = open_and_compare(&g, &qemu_img, &img);
    let blk = BlockBackend::new(&g, &name, BLK_PERM_CONSISTENT_READ, SHARED).unwrap();
    assert!(read_all(&blk) == expect);
    drop(blk);
    drop(g);

    // Writes here, one of them in a block qemu-io left unallocated.
    let g = BlockGraph::new();
    add_vpc(&g, "v", &img);
    let blk = BlockBackend::new(&g, "v", RW, SHARED).unwrap();
    let mut expect = expect;
    for (off, len, seed) in [(1000usize, 3000usize, 1u8), (7 << 20, 70_000, 2)] {
        let data = pattern(len, seed);
        blk.pwrite(off as u64, &data).unwrap();
        expect[off..off + len].copy_from_slice(&data);
    }
    assert!(read_all(&blk) == expect);
    drop(blk);
    drop(g);
    fs::write(&raw, &expect).unwrap();
    run(&qemu_img, &["compare", "-f", "vpc", "-F", "raw", path_str(&img), path_str(&raw)]);
}

#[test]
fn qemu_dynamic_image() {
    qemu_image("qemu-dynamic", "", "10M");
}

#[test]
fn qemu_fixed_image() {
    qemu_image("qemu-fixed", "subformat=fixed", "9M");
}

#[test]
fn qemu_force_size_image() {
    qemu_image("qemu-force-size", "force_size=on", "10M");
}

/// Writes patterns here to the vpc node `v`, then `qemu-img compare` against a raw copy.
fn write_and_compare(g: &BlockGraph, qemu_img: &Path, img: &Path, dir: &Path) {
    open_and_compare(g, qemu_img, img);
    let blk = BlockBackend::new(g, "v", RW, SHARED).unwrap();
    let len = blk.getlength().unwrap() as usize;
    let mut expect = vec![0u8; len];
    for (off, n, seed) in
        [(0usize, 4096usize, 1u8), (1 << 20, 3 << 20, 2), (len - 5000, 5000, 3), (777, 10, 4)]
    {
        let data = pattern(n, seed);
        blk.pwrite(off as u64, &data).unwrap();
        expect[off..off + n].copy_from_slice(&data);
    }
    assert!(read_all(&blk) == expect);
    drop(blk);
    let raw = dir.join("expect.raw");
    fs::write(&raw, &expect).unwrap();
    run(qemu_img, &["compare", "-f", "vpc", "-F", "raw", path_str(img), path_str(&raw)]);
}

#[test]
fn created_with_options() {
    let Some((qemu_img, _)) = tools() else { return };
    for (sub, force) in [("dynamic", false), ("fixed", false), ("dynamic", true), ("fixed", true)] {
        let dir = scratch(&format!("opts-{sub}-{force}"));
        let img = dir.join("disk.vhd");
        let g = BlockGraph::new();
        let mut o = QDict::new();
        o.put("size", "10M");
        o.put("subformat", sub);
        if force {
            o.put("force_size", "on");
        }
        g.create_image("vpc", path_str(&img), &mut o).unwrap();

        // The same virtual size and file size as qemu-img makes.
        let theirs = dir.join("theirs.vhd");
        let opts = format!("subformat={sub},force_size={}", if force { "on" } else { "off" });
        run(&qemu_img, &["create", "-f", "vpc", "-o", &opts, path_str(&theirs), "10M"]);
        let (a, b) = (qemu_info(&qemu_img, &img), qemu_info(&qemu_img, &theirs));
        assert_eq!(info_num(&a, "virtual-size"), info_num(&b, "virtual-size"));
        assert_eq!(fs::metadata(&img).unwrap().len(), fs::metadata(&theirs).unwrap().len());
        // Byte for byte except for the timestamp, the UUID and the checksum.
        let (mine, theirs) = (fs::read(&img).unwrap(), fs::read(&theirs).unwrap());
        let footer = mine.len() - 512;
        for (i, (x, y)) in mine.iter().zip(&theirs).enumerate() {
            let f = if i >= footer { i - footer } else { i };
            let variable = (24..28).contains(&f) || (64..84).contains(&f);
            let in_footer = i >= footer || (sub == "dynamic" && i < 512);
            assert!(*x == *y || (in_footer && variable), "byte {i} differs");
        }

        probe_and_compare(&qemu_img, &img);
        write_and_compare(&g, &qemu_img, &img, &dir);
    }
}

#[test]
fn created_with_blockdev_create() {
    let Some((qemu_img, _)) = tools() else { return };
    for (sub, size) in [("dynamic", 10_514_432u64), ("fixed", 10_514_432), ("dynamic", 5 << 20)] {
        let dir = scratch(&format!("create-{sub}-{size}"));
        let img = dir.join("disk.vhd");
        let g = BlockGraph::new();
        g.blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
            r#"{{"driver": "file", "filename": "{}", "size": 0}}"#,
            path_str(&img)
        )))
        .unwrap();
        let force = size == 5 << 20;
        g.blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
            r#"{{"driver": "vpc", "file": {{"driver": "file", "filename": "{}"}},
                "size": {size}, "subformat": "{sub}", "force-size": {force}}}"#,
            path_str(&img)
        )))
        .unwrap();
        let info = qemu_info(&qemu_img, &img);
        assert_eq!(info_num(&info, "virtual-size"), Some(size));
        write_and_compare(&g, &qemu_img, &img, &dir);
    }
}

#[test]
fn blockdev_create_errors() {
    let dir = scratch("create-errors");
    let img = dir.join("disk.vhd");
    let g = BlockGraph::new();
    g.blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
        r#"{{"driver": "file", "filename": "{}", "size": 0}}"#,
        path_str(&img)
    )))
    .unwrap();
    let create = |extra: &str| {
        g.blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
            r#"{{"driver": "vpc", "file": {{"driver": "file", "filename": "{}"}}, {extra}}}"#,
            path_str(&img)
        )))
        .unwrap_err()
    };
    let e = create(r#""size": 10485760"#);
    assert_eq!(e.message(), "The requested image size cannot be represented in CHS geometry");
    assert_eq!(
        e.hint_text().unwrap_or_default(),
        "Try size=10514432 or force-size=on (the latter makes the image incompatible with \
         Virtual PC)"
    );
    let e = create(r#""size": 3298534883328, "force-size": true"#);
    assert_eq!(e.message(), "Disk size is too large, max size is 2040 GiB");

    let mut o = QDict::new();
    o.put("size", "1M");
    o.put("subformat", "sparse");
    let e = g.create_image("vpc", path_str(&dir.join("x.vhd")), &mut o).unwrap_err();
    assert_eq!(e.message(), "Parameter 'subformat' does not accept value 'sparse'");
}

#[test]
fn virtual_sizes_match_qemu() {
    let Some((qemu_img, _)) = tools() else { return };
    let dir = scratch("sizes");
    for size in ["1", "512", "1M", "100M", "123456789", "1G", "130G"] {
        let mine = dir.join("mine.vhd");
        let theirs = dir.join("theirs.vhd");
        let _ = fs::remove_file(&mine);
        let g = BlockGraph::new();
        let mut o = QDict::new();
        o.put("size", size);
        g.create_image("vpc", path_str(&mine), &mut o).unwrap();
        run(&qemu_img, &["create", "-f", "vpc", path_str(&theirs), size]);
        let (a, b) = (qemu_info(&qemu_img, &mine), qemu_info(&qemu_img, &theirs));
        assert_eq!(info_num(&a, "virtual-size"), info_num(&b, "virtual-size"), "size {size}");
        let name = g.open_image(Some(path_str(&mine)), QDict::new()).unwrap();
        assert_eq!(Some(g.node(&name).unwrap().size), info_num(&b, "virtual-size"));
    }
}

#[test]
fn open_errors() {
    let dir = scratch("open-errors");
    let open = |bytes: &[u8]| {
        let p = dir.join("bad.vhd");
        fs::write(&p, bytes).unwrap();
        let g = BlockGraph::new();
        let e = g
            .blockdev_add(from_json::<BlockdevOptions>(&format!(
                r#"{{"driver": "vpc", "node-name": "v",
                    "file": {{"driver": "file", "filename": "{}"}}}}"#,
                path_str(&p)
            )))
            .unwrap_err();
        e.message().to_string()
    };
    assert_eq!(open(&[]), "File too small for a VHD header");
    assert_eq!(open(&[0u8; 2048]), "invalid VPC image");
    let mut footer = [0u8; 512];
    footer[..8].copy_from_slice(b"conectix");
    footer[60..64].copy_from_slice(&2u32.to_be_bytes());
    let mut img = vec![0u8; 1024];
    img.extend_from_slice(&footer);
    assert_eq!(open(&img), "Incorrect header checksum");
}
