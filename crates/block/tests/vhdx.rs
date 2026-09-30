// SPDX-License-Identifier: GPL-2.0-or-later

//! The `vhdx` format driver against QEMU: images `qemu-img` makes and `qemu-io` writes read the
//! same here, images made here (with `qemu-img create` style options and with
//! `blockdev-create`) and written here pass `qemu-img check` and compare equal to a raw copy,
//! probing finds vhdx, and a log built by hand is replayed the way QEMU replays it. The
//! interop tests skip themselves when `qemu-img` or `qemu-io` is not installed.

#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use ruvm_block::{
    BDRV_FIX_ERRORS, BDRV_FIX_LEAKS, BLK_PERM_CONSISTENT_READ, BLK_PERM_RESIZE, BLK_PERM_WRITE,
    BlockBackend, BlockGraph,
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
    let dir = base.join("ruvm-block-vhdx").join(test);
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
    run(qemu_img, &["info", "--output=json", "-f", "vhdx", path_str(path)])
}

fn from_json<T: Visit + Default>(s: &str) -> T {
    let mut v = QObjectInputVisitor::new(json::from_str(s).unwrap());
    let mut o = T::default();
    T::visit(&mut v, None, &mut o).unwrap();
    o
}

/// `blockdev-add` of a vhdx node `name` over `path`.
fn add_vhdx(g: &BlockGraph, name: &str, path: &Path, read_only: bool) -> ruvm_base::Result<()> {
    g.blockdev_add(from_json::<BlockdevOptions>(&format!(
        r#"{{"driver": "vhdx", "node-name": "{name}", "read-only": {read_only},
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

/// Probes `path` without a format and checks that it is vhdx with the size `qemu-img` sees.
fn probe_and_compare(qemu_img: &Path, path: &Path) {
    let g = BlockGraph::new();
    let name = g.open_image(Some(path_str(path)), QDict::new()).unwrap();
    assert_eq!(g.node(&name).unwrap().driver, "vhdx");
    let info = run(qemu_img, &["info", "--output=json", path_str(path)]);
    assert!(info.contains("\"format\": \"vhdx\""));
    assert_eq!(Some(g.node(&name).unwrap().size), info_num(&info, "virtual-size"));
}

/// Opens `path` as vhdx and checks the virtual size and the cluster size against
/// `qemu-img info -f vhdx`.
fn open_and_compare(g: &BlockGraph, qemu_img: &Path, path: &Path) {
    add_vhdx(g, "v", path, false).unwrap();
    let info = qemu_info(qemu_img, path);
    let nodes = g.query_named_block_nodes(Some(true)).unwrap();
    let n = nodes.iter().find(|n| n.node_name == "v").unwrap();
    assert_eq!(n.image.virtual_size as u64, info_num(&info, "virtual-size").unwrap());
    assert!(n.image.cluster_size.is_some());
    assert_eq!(n.image.cluster_size.map(|c| c as u64), info_num(&info, "cluster-size"));
}

/// Writes through a backend on `path` here, at places that cover unallocated blocks, partial
/// sectors of blocks and block boundaries, then checks the image with `qemu-img check` and
/// compares it with a raw copy.
fn write_and_compare(qemu_img: &Path, path: &Path, dir: &Path, block_size: usize) {
    let g = BlockGraph::new();
    add_vhdx(&g, "v", path, false).unwrap();
    let blk = BlockBackend::new(&g, "v", RW, SHARED).unwrap();
    let mut expect = read_all(&blk);
    let len = expect.len();
    let writes = [
        (0usize, 4096usize, 1u8),
        (1000, 3000, 2),
        (block_size - 1024, 2048, 3),
        (len / 2 + 512, 70_000, 4),
        (len - 512, 512, 5),
        (3 * block_size / 2, block_size, 6),
    ];
    for (off, n, seed) in writes {
        if off + n > len {
            continue;
        }
        let data = pattern(n, seed);
        blk.pwrite(off as u64, &data).unwrap();
        expect[off..off + n].copy_from_slice(&data);
    }
    assert!(read_all(&blk) == expect);
    drop(blk);
    drop(g);

    // And again after reopening.
    let g = BlockGraph::new();
    add_vhdx(&g, "v", path, false).unwrap();
    let blk = BlockBackend::new(&g, "v", BLK_PERM_CONSISTENT_READ, SHARED).unwrap();
    assert!(read_all(&blk) == expect);
    drop(blk);
    drop(g);

    let raw = dir.join("expect.raw");
    fs::write(&raw, &expect).unwrap();
    run(qemu_img, &["check", "-f", "vhdx", path_str(path)]);
    run(qemu_img, &["compare", "-f", "vhdx", "-F", "raw", path_str(path), path_str(&raw)]);
    fs::remove_file(&raw).unwrap();
}

/// `qemu-img create` and `qemu-io` writes, then everything reads the same here, and writes
/// here read back in `qemu-img`.
fn qemu_image(test: &str, create_opts: &str, size: usize, block_size: usize) {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    let dir = scratch(test);
    let img = dir.join("disk.vhdx");
    let size_s = size.to_string();
    let mut args = vec!["create", "-q", "-f", "vhdx"];
    if !create_opts.is_empty() {
        args.extend(["-o", create_opts]);
    }
    args.extend([path_str(&img), &size_s]);
    run(&qemu_img, &args);
    let w1 = format!("write -P 0xaa {} {}", block_size - 4096, 2 * 4096 + 512);
    let w2 = format!("write -P 0x11 {} 512", size - 512);
    run(
        &qemu_io,
        &["-f", "vhdx", "-c", "write -P 0x55 0 64k", "-c", &w1, "-c", &w2, path_str(&img)],
    );
    let raw = dir.join("disk.raw");
    run(&qemu_img, &["convert", "-f", "vhdx", "-O", "raw", path_str(&img), path_str(&raw)]);
    let expect = fs::read(&raw).unwrap();
    fs::remove_file(&raw).unwrap();

    probe_and_compare(&qemu_img, &img);
    let g = BlockGraph::new();
    open_and_compare(&g, &qemu_img, &img);
    let blk = BlockBackend::new(&g, "v", BLK_PERM_CONSISTENT_READ, SHARED).unwrap();
    assert!(read_all(&blk) == expect);
    drop(blk);
    drop(g);

    write_and_compare(&qemu_img, &img, &dir, block_size);
}

#[test]
fn qemu_dynamic_image() {
    qemu_image("qemu-dynamic", "", 20 * MIB, 8 * MIB);
}

#[test]
fn qemu_dynamic_small_blocks() {
    qemu_image("qemu-dynamic-1m", "block_size=1M,log_size=2M", 5 * MIB + 512, MIB);
}

#[test]
fn qemu_dynamic_no_zero_blocks() {
    qemu_image("qemu-dynamic-nozero", "block_state_zero=off,block_size=2M", 9 * MIB, 2 * MIB);
}

#[test]
fn qemu_fixed_image() {
    qemu_image("qemu-fixed", "subformat=fixed,block_size=4M", 12 * MIB, 4 * MIB);
}

#[test]
fn qemu_large_blocks() {
    qemu_image("qemu-32m", "block_size=32M,log_size=3M", 70 * MIB, 32 * MIB);
}

/// Every `-o` option through `create_image`; the images must pass `qemu-img check`, look the
/// same in `qemu-img info` as QEMU's own, and take writes.
#[test]
fn created_with_create_opts() {
    let Some((qemu_img, _)) = tools() else { return };
    type Case<'a> = (&'a str, &'a [(&'a str, &'a str)], usize);
    let cases: &[Case<'_>] = &[
        ("default", &[], 8 * MIB),
        ("fixed", &[("subformat", "fixed")], 6 * MIB),
        ("dynamic", &[("subformat", "dynamic"), ("block_size", "1M")], 3 * MIB + 4096),
        ("log", &[("log_size", "4M"), ("block_size", "2M")], 10 * MIB),
        ("nozero", &[("block_state_zero", "off"), ("block_size", "4M")], 9 * MIB),
        ("fixed-nozero", &[("subformat", "fixed"), ("block_state_zero", "off")], 2 * MIB),
        // Rounded up to 1 MiB and to 512 bytes.
        ("round", &[("block_size", "1500k"), ("log_size", "1000k")], 5 * MIB + 100),
    ];
    for (name, opts, size) in cases {
        let dir = scratch(&format!("create-opts-{name}"));
        let mine = dir.join("mine.vhdx");
        let theirs = dir.join("theirs.vhdx");
        let g = BlockGraph::new();
        let mut o = QDict::new();
        o.put("size", size.to_string().as_str());
        let mut qopts = Vec::new();
        for (k, v) in *opts {
            o.put(*k, *v);
            qopts.push(format!("{k}={v}"));
        }
        g.create_image("vhdx", path_str(&mine), &mut o).unwrap();
        let size_s = size.to_string();
        let mut args = vec!["create", "-q", "-f", "vhdx"];
        let qopts = qopts.join(",");
        if !qopts.is_empty() {
            args.extend(["-o", &qopts]);
        }
        args.extend([path_str(&theirs), &size_s]);
        run(&qemu_img, &args);

        run(&qemu_img, &["check", "-f", "vhdx", path_str(&mine)]);
        same_layout(&fs::read(&mine).unwrap(), &fs::read(&theirs).unwrap(), name);
        let (a, b) = (qemu_info(&qemu_img, &mine), qemu_info(&qemu_img, &theirs));
        for key in ["virtual-size", "cluster-size", "actual-size"] {
            if key == "actual-size" {
                // The allocated size depends on the host file system; the file size does not.
                assert_eq!(
                    fs::metadata(&mine).unwrap().len(),
                    fs::metadata(&theirs).unwrap().len(),
                    "{name}: file size"
                );
                continue;
            }
            assert_eq!(info_num(&a, key), info_num(&b, key), "{name}: {key}");
        }
        let block_size = info_num(&b, "cluster-size").unwrap() as usize;
        probe_and_compare(&qemu_img, &mine);
        write_and_compare(&qemu_img, &mine, &dir, block_size);
    }
}

/// Checks that two new images are the same byte for byte, except for what is random or
/// names the version: the creator, the sequence numbers, guids and checksums of the headers
/// and the page 83 data.
fn same_layout(mine: &[u8], theirs: &[u8], name: &str) {
    assert_eq!(mine.len(), theirs.len(), "{name}");
    let metadata = metadata_offset(theirs);
    let page83 = metadata + le32(theirs, metadata + 32 + 2 * 32 + 16) as usize;
    let variable = |i: usize| {
        (8..0x10000).contains(&i)
            || (0x10004..0x10030).contains(&i)
            || (0x20004..0x20030).contains(&i)
            || (page83..page83 + 16).contains(&i)
    };
    for (i, (x, y)) in mine.iter().zip(theirs).enumerate() {
        assert!(x == y || variable(i), "{name}: byte {i:#x} differs");
    }
}

#[test]
fn created_with_blockdev_create() {
    let Some((qemu_img, _)) = tools() else { return };
    let cases = [
        (7 * MIB, r#""log-size": 1048576"#, MIB * 8),
        (6 * MIB, r#""block-size": 1048576, "subformat": "fixed""#, MIB),
        (11 * MIB, r#""block-size": 2097152, "block-state-zero": false"#, 2 * MIB),
        (
            3 * MIB,
            r#""log-size": 2097152, "subformat": "dynamic", "block-state-zero": true"#,
            8 * MIB,
        ),
    ];
    for (i, (size, extra, block_size)) in cases.into_iter().enumerate() {
        let dir = scratch(&format!("blockdev-create-{i}"));
        let img = dir.join("disk.vhdx");
        let g = BlockGraph::new();
        g.blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
            r#"{{"driver": "file", "filename": "{}", "size": 0}}"#,
            path_str(&img)
        )))
        .unwrap();
        g.blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
            r#"{{"driver": "vhdx", "file": {{"driver": "file", "filename": "{}"}},
                "size": {size}, {extra}}}"#,
            path_str(&img)
        )))
        .unwrap();
        drop(g);
        run(&qemu_img, &["check", "-f", "vhdx", path_str(&img)]);
        let info = qemu_info(&qemu_img, &img);
        assert_eq!(info_num(&info, "virtual-size"), Some(size as u64));
        assert_eq!(info_num(&info, "cluster-size"), Some(block_size as u64));
        write_and_compare(&qemu_img, &img, &dir, block_size);
    }
}

#[test]
fn create_errors() {
    let dir = scratch("create-errors");
    let g = BlockGraph::new();
    let create = |opts: &[(&str, &str)]| {
        let mut o = QDict::new();
        for (k, v) in opts {
            o.put(*k, *v);
        }
        let p = dir.join("x.vhdx");
        let _ = fs::remove_file(&p);
        g.create_image("vhdx", path_str(&p), &mut o).unwrap_err().message().to_string()
    };
    assert_eq!(create(&[("size", "65T")]), "Image size too large; max of 64TB");
    assert_eq!(create(&[("size", "1M"), ("log_size", "5G")]), "Log size must be smaller than 4 GB");
    assert_eq!(
        create(&[("size", "1M"), ("block_size", "3M")]),
        "Block size must be a power of two"
    );
    assert_eq!(
        create(&[("size", "1M"), ("subformat", "x")]),
        "Parameter 'subformat' does not accept value 'x'"
    );
    assert_eq!(
        create(&[("size", "1M"), ("block_state_zero", "x")]),
        "Parameter 'block_state_zero' expects 'on' or 'off'"
    );
    assert!(!dir.join("x.vhdx").exists());

    let img = dir.join("disk.vhdx");
    g.blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
        r#"{{"driver": "file", "filename": "{}", "size": 0}}"#,
        path_str(&img)
    )))
    .unwrap();
    let bc = |extra: &str| {
        g.blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
            r#"{{"driver": "vhdx", "file": {{"driver": "file", "filename": "{}"}},
                "size": 1048576, {extra}}}"#,
            path_str(&img)
        )))
        .unwrap_err()
        .message()
        .to_string()
    };
    assert_eq!(bc(r#""log-size": 1000"#), "Log size must be a multiple of 1 MB");
    assert_eq!(bc(r#""block-size": 1000"#), "Block size must be a multiple of 1 MB");
    assert_eq!(bc(r#""block-size": 536870912"#), "Block size must not exceed 268435456");
}

#[test]
fn truncate_is_refused() {
    let dir = scratch("truncate");
    let img = dir.join("disk.vhdx");
    let g = BlockGraph::new();
    let mut o = QDict::new();
    o.put("size", "4M");
    g.create_image("vhdx", path_str(&img), &mut o).unwrap();
    add_vhdx(&g, "v", &img, false).unwrap();
    let blk = BlockBackend::new(&g, "v", RW | BLK_PERM_RESIZE, SHARED).unwrap();
    let e = blk.truncate(8 << 20).unwrap_err();
    assert_eq!(e.message(), "Image format driver does not support resize");
}

#[test]
fn open_errors() {
    let dir = scratch("open-errors");
    let p = dir.join("bad.vhdx");
    let open = |bytes: &[u8]| {
        fs::write(&p, bytes).unwrap();
        let g = BlockGraph::new();
        add_vhdx(&g, "v", &p, false).unwrap_err().message().to_string()
    };
    let not_vhdx = open(&[0u8; 4096]);
    assert_eq!(not_vhdx, format!("Could not open '{}': Invalid argument", path_str(&p)));

    let mut img = vec![0u8; 4 * MIB];
    img[..8].copy_from_slice(b"vhdxfile");
    assert_eq!(open(&img), "No valid VHDX header found");

    // A good image with a broken region table.
    let g = BlockGraph::new();
    let mut o = QDict::new();
    o.put("size", "4M");
    g.create_image("vhdx", path_str(&p), &mut o).unwrap();
    drop(g);
    let mut img = fs::read(&p).unwrap();
    let clean = img.clone();
    img[192 * 1024 + 20] ^= 1;
    let einval = format!("Could not open '{}': Invalid argument", path_str(&p));
    assert_eq!(open(&img), einval);

    // Differencing images: has_parent without a parent locator is invalid, and an image with
    // a parent locator is not supported.
    let mut img = clean.clone();
    let md = metadata_offset(&img);
    let params = md + le32(&img, md + 32 + 16) as usize;
    img[params + 4] |= 2;
    assert_eq!(open(&img), einval);
    let locator_guid = [
        0x2d, 0x5f, 0xd3, 0xa8, 0x0b, 0xb3, 0x4d, 0x45, 0xab, 0xf7, 0xd3, 0xd8, 0x48, 0x34, 0xab,
        0x0c,
    ];
    let e = md + 32 + 5 * 32;
    img[md + 10] = 6;
    img[e..e + 16].copy_from_slice(&locator_guid);
    img[e + 16..e + 20].copy_from_slice(&(128u32 * 1024).to_le_bytes());
    img[e + 24..e + 28].copy_from_slice(&4u32.to_le_bytes());
    let enotsup = format!("Could not open '{}': Operation not supported", path_str(&p));
    assert_eq!(open(&img), enotsup);
    img[params + 4] &= !2;
    assert_eq!(open(&img), enotsup);

    // Logical sectors of 4 KiB are not supported either.
    let mut img = clean.clone();
    let logical = md + le32(&img, md + 32 + 3 * 32 + 16) as usize;
    img[logical..logical + 4].copy_from_slice(&4096u32.to_le_bytes());
    assert_eq!(open(&img), enotsup);

    // A payload block past the end of the file.
    let mut img = clean;
    let bat = bat_offset(&img);
    img[bat..bat + 8].copy_from_slice(&((64u64 << 20) | 6).to_le_bytes());
    assert_eq!(open(&img), einval);
}

/// The file offset of the region with `guid` in `img`.
fn region_offset(img: &[u8], guid: &[u8; 16]) -> usize {
    let rt = 192 * 1024;
    let n = le32(img, rt + 8) as usize;
    let e = (0..n).map(|i| rt + 16 + i * 32).find(|&e| img[e..e + 16] == *guid).unwrap();
    le64(img, e + 16) as usize
}

fn metadata_offset(img: &[u8]) -> usize {
    region_offset(
        img,
        &[
            0x06, 0xa2, 0x7c, 0x8b, 0x90, 0x47, 0x9a, 0x4b, 0xb8, 0xfe, 0x57, 0x5f, 0x05, 0x0f,
            0x88, 0x6e,
        ],
    )
}

fn bat_offset(img: &[u8]) -> usize {
    region_offset(
        img,
        &[
            0x66, 0x77, 0xc2, 0x2d, 0x23, 0xf6, 0x00, 0x42, 0x9d, 0x64, 0x11, 0x5e, 0x9b, 0xfd,
            0x4a, 0x08,
        ],
    )
}

/// CRC-32C, for building log entries.
fn crc32c(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0x82F6_3B78 } else { crc >> 1 };
        }
    }
    !crc
}

fn set_crc(buf: &mut [u8]) {
    buf[4..8].fill(0);
    let crc = crc32c(buf);
    buf[4..8].copy_from_slice(&crc.to_le_bytes());
}

fn le32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn le64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

/// Puts a log into the image in `img`: the active header gets a log guid, and the log one
/// entry with a data descriptor for the 4 KiB at `data_off` and a zero descriptor for the
/// 8 KiB at `zero_off`, both file offsets. The previous entry, with a lower sequence number
/// and a different guid, must be ignored.
fn add_log(img: &mut [u8], data_off: u64, data: &[u8; 4096], zero_off: u64) {
    let guid: [u8; 16] = *b"ruvm-vhdx-log-01";
    let (h1, h2) = (64 * 1024, 128 * 1024);
    let active = if le64(img, h1 + 8) > le64(img, h2 + 8) { h1 } else { h2 };
    img[active + 48..active + 64].copy_from_slice(&guid);
    set_crc(&mut img[active..active + 4096]);
    let log_offset = le64(img, active + 72) as usize;
    assert_eq!(log_offset, MIB);

    let seq = 0x1234_5678_9abc_u64;
    let mut entry = vec![0u8; 2 * 4096];
    entry[0..4].copy_from_slice(b"loge");
    entry[8..12].copy_from_slice(&(2u32 * 4096).to_le_bytes());
    entry[16..24].copy_from_slice(&seq.to_le_bytes());
    entry[24..28].copy_from_slice(&2u32.to_le_bytes());
    entry[32..48].copy_from_slice(&guid);
    let file_len = img.len() as u64;
    entry[48..56].copy_from_slice(&file_len.to_le_bytes());
    entry[56..64].copy_from_slice(&file_len.to_le_bytes());
    // The data descriptor.
    let d = 64;
    entry[d..d + 4].copy_from_slice(b"desc");
    entry[d + 4..d + 8].copy_from_slice(&data[4092..]);
    entry[d + 8..d + 16].copy_from_slice(&data[..8]);
    entry[d + 16..d + 24].copy_from_slice(&data_off.to_le_bytes());
    entry[d + 24..d + 32].copy_from_slice(&seq.to_le_bytes());
    // The zero descriptor.
    let z = 96;
    entry[z..z + 4].copy_from_slice(b"zero");
    entry[z + 8..z + 16].copy_from_slice(&8192u64.to_le_bytes());
    entry[z + 16..z + 24].copy_from_slice(&zero_off.to_le_bytes());
    entry[z + 24..z + 32].copy_from_slice(&seq.to_le_bytes());
    // The data sector.
    let s = 4096;
    entry[s..s + 4].copy_from_slice(b"data");
    entry[s + 4..s + 8].copy_from_slice(&((seq >> 32) as u32).to_le_bytes());
    entry[s + 8..s + 4092].copy_from_slice(&data[8..4092]);
    entry[s + 4092..s + 4096].copy_from_slice(&(seq as u32).to_le_bytes());
    set_crc(&mut entry);

    // A stale entry from an older sequence at the start of the log, then the live one.
    let mut stale = entry.clone();
    stale[16..24].copy_from_slice(&(seq - 5).to_le_bytes());
    stale[32] ^= 0xff;
    set_crc(&mut stale);
    img[log_offset..log_offset + stale.len()].copy_from_slice(&stale);
    let at = log_offset + 3 * 4096;
    img[at..at + entry.len()].copy_from_slice(&entry);
}

/// The file offset of the block `idx` in `img`, from the BAT.
fn block_offset(img: &[u8], idx: usize) -> u64 {
    le64(img, bat_offset(img) + idx * 8) & !0xfffff
}

#[test]
fn log_replay() {
    let dir = scratch("log-replay");
    let img = dir.join("disk.vhdx");
    let g = BlockGraph::new();
    let mut o = QDict::new();
    o.put("size", "4M");
    o.put("block_size", "1M");
    g.create_image("vhdx", path_str(&img), &mut o).unwrap();
    add_vhdx(&g, "v", &img, false).unwrap();
    let blk = BlockBackend::new(&g, "v", RW, SHARED).unwrap();
    let mut expect = pattern(4 * MIB, 9);
    blk.pwrite(0, &expect).unwrap();
    drop(blk);
    drop(g);

    let mut bytes = fs::read(&img).unwrap();
    let block1 = block_offset(&bytes, 1);
    let block2 = block_offset(&bytes, 2);
    assert!(block1 >= MIB as u64 && block2 >= MIB as u64);
    let data: [u8; 4096] = pattern(4096, 77).try_into().unwrap();
    add_log(&mut bytes, block1 + 8192, &data, block2 + 4096);
    fs::write(&img, &bytes).unwrap();
    expect[MIB + 8192..MIB + 8192 + 4096].copy_from_slice(&data);
    expect[2 * MIB + 4096..2 * MIB + 3 * 4096].fill(0);
    let dirty = dir.join("dirty.vhdx");
    fs::copy(&img, &dirty).unwrap();

    // Read-only, the log cannot be replayed.
    let g = BlockGraph::new();
    let e = add_vhdx(&g, "v", &img, true).unwrap_err();
    assert_eq!(
        e.message(),
        format!(
            "VHDX image file '{}' opened read-only, but contains a log that needs to be replayed",
            path_str(&img)
        )
    );
    assert_eq!(
        e.hint_text().unwrap(),
        format!("To replay the log, run:\nqemu-img check -r all '{}'\n", path_str(&img))
    );
    drop(g);

    // Read-write, it is.
    let g = BlockGraph::new();
    add_vhdx(&g, "v", &img, false).unwrap();
    let blk = BlockBackend::new(&g, "v", BLK_PERM_CONSISTENT_READ, SHARED).unwrap();
    assert!(read_all(&blk) == expect);
    drop(blk);
    drop(g);
    // And the log is empty now, so a read-only open works.
    let g = BlockGraph::new();
    add_vhdx(&g, "v", &img, true).unwrap();
    drop(g);

    let Some((qemu_img, _)) = tools() else { return };
    // QEMU refuses the dirty image read-only in the same words.
    let out =
        Command::new(&qemu_img).args(["info", "-f", "vhdx", path_str(&dirty)]).output().unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("opened read-only, but contains a log that needs to be replayed"),
        "{stderr}"
    );
    // QEMU replays the log to the same contents, and likes what the replay here left.
    let check = Command::new(&qemu_img)
        .args(["check", "-r", "all", "-f", "vhdx", path_str(&dirty)])
        .output()
        .unwrap();
    assert!(check.status.success(), "{}", String::from_utf8_lossy(&check.stderr));
    run(&qemu_img, &["check", "-f", "vhdx", path_str(&img)]);
    run(&qemu_img, &["compare", "-f", "vhdx", "-F", "vhdx", path_str(&img), path_str(&dirty)]);
    let raw = dir.join("expect.raw");
    fs::write(&raw, &expect).unwrap();
    run(&qemu_img, &["compare", "-f", "vhdx", "-F", "raw", path_str(&img), path_str(&raw)]);
}

/// A log that was replayed on open counts as a fixed corruption in `check`, as in QEMU, and
/// a clean image has nothing to report.
#[test]
fn check_counts_replayed_log() {
    let dir = scratch("check-log");
    let img = dir.join("disk.vhdx");
    let g = BlockGraph::new();
    let mut o = QDict::new();
    o.put("size", "2M");
    o.put("block_size", "1M");
    g.create_image("vhdx", path_str(&img), &mut o).unwrap();
    add_vhdx(&g, "v", &img, false).unwrap();
    let blk = BlockBackend::new(&g, "v", RW, SHARED).unwrap();
    blk.pwrite(0, &pattern(2 * MIB, 3)).unwrap();
    drop(blk);
    let r = g.check("v", 0).unwrap();
    assert_eq!((r.corruptions, r.corruptions_fixed), (0, 0));
    drop(g);

    let mut bytes = fs::read(&img).unwrap();
    let block0 = block_offset(&bytes, 0);
    let block1 = block_offset(&bytes, 1);
    add_log(&mut bytes, block0, &[0x42; 4096], block1);
    fs::write(&img, &bytes).unwrap();
    let dirty = dir.join("dirty.vhdx");
    fs::copy(&img, &dirty).unwrap();

    let g = BlockGraph::new();
    add_vhdx(&g, "v", &img, false).unwrap();
    let r = g.check("v", BDRV_FIX_ERRORS | BDRV_FIX_LEAKS).unwrap();
    assert_eq!((r.corruptions, r.corruptions_fixed), (0, 1));
    drop(g);

    let Some((qemu_img, _)) = tools() else { return };
    let out = Command::new(&qemu_img)
        .args(["check", "-r", "all", "-f", "vhdx", path_str(&dirty)])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("The following inconsistencies were found and repaired"), "{stdout}");
    run(&qemu_img, &["compare", "-f", "vhdx", "-F", "vhdx", path_str(&img), path_str(&dirty)]);
}
