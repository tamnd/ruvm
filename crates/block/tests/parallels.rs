// SPDX-License-Identifier: GPL-2.0-or-later

//! The `parallels` format driver against QEMU: images `qemu-img` makes and `qemu-io` writes
//! read the same here and map to the same places, images made and written here pass
//! `qemu-img check` and compare equal to a raw copy, probing picks parallels, and checking
//! and repairing a hand corrupted BAT finds and fixes what `qemu-img check` does. The interop
//! tests skip themselves when `qemu-img` or `qemu-io` is not installed.

#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

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
    let dir = base.join("ruvm-block-parallels").join(test);
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

fn output(prog: &Path, args: &[&str]) -> Output {
    Command::new(prog).args(args).output().unwrap()
}

/// Runs `prog` with `args`, which must succeed, and returns stdout.
fn run(prog: &Path, args: &[&str]) -> String {
    let out = output(prog, args);
    assert!(
        out.status.success(),
        "{} {args:?} failed: {}{}",
        prog.display(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

/// The number after `"key": ` in JSON output of `qemu-img`, the last one when there are
/// several (the top level entry comes after those of the children).
fn json_num(info: &str, key: &str) -> Option<u64> {
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

/// `blockdev-add` of a parallels node `name` over `path`.
fn add_parallels(g: &BlockGraph, name: &str, path: &Path, read_only: bool) {
    g.blockdev_add(from_json::<BlockdevOptions>(&format!(
        r#"{{"driver": "parallels", "node-name": "{name}", "read-only": {read_only},
            "file": {{"driver": "file", "filename": "{}", "read-only": {read_only}}}}}"#,
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

/// The allocated ranges of `blk` as (guest offset, length, host offset), merged the way
/// `qemu-img map` merges them.
fn our_map(blk: &BlockBackend) -> Vec<(u64, u64, u64)> {
    let len = blk.getlength().unwrap();
    let mut out: Vec<(u64, u64, u64)> = Vec::new();
    let mut off = 0;
    while off < len {
        let e = blk.map_entry(off, len - off).unwrap();
        if e.data {
            let host = e.offset.unwrap();
            match out.last_mut() {
                Some(l) if l.0 + l.1 == off && l.2 + l.1 == host => l.1 += e.length,
                _ => out.push((off, e.length, host)),
            }
        }
        off += e.length;
    }
    out
}

/// The same from `qemu-img map --output=json`.
fn qemu_map(qemu_img: &Path, path: &Path) -> Vec<(u64, u64, u64)> {
    let out = run(qemu_img, &["map", "--output=json", "-f", "parallels", path_str(path)]);
    out.lines()
        .filter(|l| l.contains("\"data\": true"))
        .map(|l| {
            (
                json_num(l, "start").unwrap(),
                json_num(l, "length").unwrap(),
                json_num(l, "offset").unwrap(),
            )
        })
        .collect()
}

/// `qemu-img check` must find nothing wrong.
fn qemu_check_clean(qemu_img: &Path, path: &Path) {
    let out = run(qemu_img, &["check", "-f", "parallels", path_str(path)]);
    assert!(out.contains("No errors were found on the image."), "{out}");
}

/// `qemu-img compare` of `img` against a raw file holding `expect`.
fn qemu_compare(qemu_img: &Path, img: &Path, expect: &[u8]) {
    let raw = img.with_extension("raw");
    fs::write(&raw, expect).unwrap();
    run(qemu_img, &["compare", "-f", "parallels", "-F", "raw", path_str(img), path_str(&raw)]);
}

/// A qemu-img image with qemu-io writes reads and maps the same here, and writes here pass
/// `qemu-img check` and `qemu-img compare`.
#[test]
fn qemu_image() {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    for cluster in ["64k", "1M", "33k"] {
        let dir = scratch(&format!("qemu-{cluster}"));
        let img = dir.join("disk.img");
        let opts = format!("cluster_size={cluster}");
        run(&qemu_img, &["create", "-f", "parallels", "-o", &opts, path_str(&img), "20M"]);
        run(
            &qemu_io,
            &[
                "-f",
                "parallels",
                "-c",
                "write -P 0x55 0 64k",
                "-c",
                "write -P 0xaa 3000k 1500k",
                "-c",
                "write -P 0x11 15M 512",
                path_str(&img),
            ],
        );
        let raw = dir.join("ref.raw");
        run(
            &qemu_img,
            &["convert", "-f", "parallels", "-O", "raw", path_str(&img), path_str(&raw)],
        );
        let mut expect = fs::read(&raw).unwrap();

        let g = BlockGraph::new();
        add_parallels(&g, "p", &img, true);
        let blk = BlockBackend::new(&g, "p", BLK_PERM_CONSISTENT_READ, SHARED).unwrap();
        assert!(read_all(&blk) == expect);
        assert_eq!(our_map(&blk), qemu_map(&qemu_img, &img));
        drop(blk);
        drop(g);

        // Writes here, into allocated and unallocated clusters.
        let g = BlockGraph::new();
        add_parallels(&g, "p", &img, false);
        let blk = BlockBackend::new(&g, "p", RW, SHARED).unwrap();
        for (off, len, seed) in [(1000usize, 3000usize, 1u8), (7 << 20, 70_000, 2), (3 << 20, 5, 3)]
        {
            let data = pattern(len, seed);
            blk.pwrite(off as u64, &data).unwrap();
            expect[off..off + len].copy_from_slice(&data);
        }
        assert!(read_all(&blk) == expect);
        let map = our_map(&blk);
        drop(blk);
        drop(g);
        qemu_check_clean(&qemu_img, &img);
        qemu_compare(&qemu_img, &img, &expect);
        assert_eq!(map, qemu_map(&qemu_img, &img));
    }
}

/// Images created here are byte for byte what `qemu-img create` makes, and pass
/// `qemu-img check` and `qemu-img compare` after writes here.
#[test]
fn created_with_options() {
    let Some((qemu_img, _)) = tools() else { return };
    for (size, cluster) in
        [("20M", None), ("4M", Some("64k")), ("1000", Some("1000")), ("300M", None)]
    {
        let dir = scratch(&format!("opts-{size}-{}", cluster.unwrap_or("default")));
        let img = dir.join("mine.img");
        let theirs = dir.join("theirs.img");
        let g = BlockGraph::new();
        let mut o = QDict::new();
        o.put("size", size);
        let mut args = vec!["create", "-f", "parallels"];
        let opts;
        if let Some(c) = cluster {
            o.put("cluster_size", c);
            opts = format!("cluster_size={c}");
            args.extend(["-o", &opts]);
        }
        g.create_image("parallels", path_str(&img), &mut o).unwrap();
        args.extend([path_str(&theirs), size]);
        run(&qemu_img, &args);
        assert!(fs::read(&img).unwrap() == fs::read(&theirs).unwrap(), "{size} {cluster:?}");
        qemu_check_clean(&qemu_img, &img);

        // Probing picks parallels, with the same virtual size as qemu-img info.
        let name = g.open_image(Some(path_str(&img)), QDict::new()).unwrap();
        let info = run(&qemu_img, &["info", "--output=json", path_str(&img)]);
        assert!(info.contains("\"format\": \"parallels\""));
        assert_eq!(g.node(&name).unwrap().driver, "parallels");
        assert_eq!(Some(g.node(&name).unwrap().size), json_num(&info, "virtual-size"));

        // The info fields qemu-img prints: no cluster size, no dirty flag.
        let nodes = g.query_named_block_nodes(Some(true)).unwrap();
        let n = nodes.iter().find(|n| n.node_name == name).unwrap();
        assert_eq!(n.image.virtual_size as u64, json_num(&info, "virtual-size").unwrap());
        assert_eq!(n.image.cluster_size.map(|c| c as u64), json_num(&info, "cluster-size"));
        assert_eq!(n.image.dirty_flag.unwrap_or(false), info.contains("\"dirty-flag\": true"));
        assert!(
            n.image.format_specific.is_none()
                && !info.contains("\"format-specific\": {\n        \"type\": \"parallels")
        );
        drop(nodes);
        drop(g);

        let g = BlockGraph::new();
        add_parallels(&g, "p", &img, false);
        let blk = BlockBackend::new(&g, "p", RW, SHARED).unwrap();
        let len = blk.getlength().unwrap() as usize;
        let mut expect = vec![0u8; len];
        let writes = [(0usize, 4096usize, 1u8), (len / 2, 3 << 20, 2), (len - 512, 512, 3)];
        for (off, n, seed) in writes {
            let n = n.min(len - off);
            let data = pattern(n, seed);
            blk.pwrite(off as u64, &data).unwrap();
            expect[off..off + n].copy_from_slice(&data);
        }
        assert!(read_all(&blk) == expect);
        drop(blk);
        drop(g);
        qemu_check_clean(&qemu_img, &img);
        qemu_compare(&qemu_img, &img, &expect);
    }
}

#[test]
fn created_with_blockdev_create() {
    let Some((qemu_img, _)) = tools() else { return };
    let dir = scratch("blockdev-create");
    let img = dir.join("disk.img");
    let g = BlockGraph::new();
    g.blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
        r#"{{"driver": "file", "filename": "{}", "size": 0}}"#,
        path_str(&img)
    )))
    .unwrap();
    g.blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
        r#"{{"driver": "parallels", "file": {{"driver": "file", "filename": "{}"}},
            "size": 10485760, "cluster-size": 131072}}"#,
        path_str(&img)
    )))
    .unwrap();
    let theirs = dir.join("theirs.img");
    run(
        &qemu_img,
        &["create", "-f", "parallels", "-o", "cluster_size=128k", path_str(&theirs), "10M"],
    );
    assert!(fs::read(&img).unwrap() == fs::read(&theirs).unwrap());

    let create = |extra: &str| {
        g.blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
            r#"{{"driver": "parallels", "file": {{"driver": "file", "filename": "{}"}}, {extra}}}"#,
            path_str(&img)
        )))
        .unwrap_err()
        .message()
        .to_string()
    };
    assert_eq!(create(r#""size": 1000"#), "Image size must be a multiple of 512 bytes");
    assert_eq!(
        create(r#""size": 1024, "cluster-size": 1000"#),
        "Cluster size must be a multiple of 512 bytes"
    );
    assert_eq!(create(r#""size": 1024, "cluster-size": 2147483648"#), "Cluster size is too large");
    assert_eq!(
        create(r#""size": 2199023255552, "cluster-size": 512"#),
        "Image size is too large for this cluster size"
    );
    assert_eq!(
        create(r#""size": 1024, "cluster-size": 0"#),
        "Image size is too large for this cluster size"
    );
    assert_eq!(create(r#""size": 1099511627776, "cluster-size": 512"#), "Catalog too large");
}

/// An image with the old "WithoutFreeSpace" magic, whose BAT counts sectors and whose
/// `data_off` is zero, reads the same here as in qemu-img.
#[test]
fn old_magic() {
    let Some((qemu_img, _)) = tools() else { return };
    let dir = scratch("old-magic");
    let img = dir.join("old.img");
    // 1 MiB disk, 4 KiB clusters (8 sectors), 256 BAT entries, data after the BAT at 2 KiB.
    let mut b = vec![0u8; 2048];
    b[..16].copy_from_slice(b"WithoutFreeSpace");
    b[16..20].copy_from_slice(&2u32.to_le_bytes());
    b[28..32].copy_from_slice(&8u32.to_le_bytes());
    b[32..36].copy_from_slice(&256u32.to_le_bytes());
    b[36..44].copy_from_slice(&2048u64.to_le_bytes());
    // Guest cluster 3 at sector 4, guest cluster 100 at sector 12.
    b[64 + 4 * 3..64 + 4 * 4].copy_from_slice(&4u32.to_le_bytes());
    b[64 + 4 * 100..64 + 4 * 101].copy_from_slice(&12u32.to_le_bytes());
    b.extend(pattern(8192, 7));
    fs::write(&img, &b).unwrap();

    let raw = dir.join("ref.raw");
    run(&qemu_img, &["convert", "-f", "parallels", "-O", "raw", path_str(&img), path_str(&raw)]);
    let expect = fs::read(&raw).unwrap();
    let g = BlockGraph::new();
    let name = g.open_image(Some(path_str(&img)), QDict::new()).unwrap();
    assert_eq!(g.node(&name).unwrap().driver, "parallels");
    drop(g);
    let g = BlockGraph::new();
    add_parallels(&g, "p", &img, true);
    let blk = BlockBackend::new(&g, "p", BLK_PERM_CONSISTENT_READ, SHARED).unwrap();
    assert!(read_all(&blk) == expect);
    assert_eq!(our_map(&blk), qemu_map(&qemu_img, &img));
    assert_eq!(&expect[3 * 4096..4 * 4096], &b[2048..2048 + 4096]);
}

/// A 4 MiB image with 64 KiB clusters: guest cluster 0 at host cluster 1, guest clusters 16
/// and 17 at host clusters 2 and 3, as qemu-io writes it.
fn base_image(qemu_img: &Path, qemu_io: &Path, img: &Path) {
    run(qemu_img, &["create", "-f", "parallels", "-o", "cluster_size=64k", path_str(img), "4M"]);
    run(
        qemu_io,
        &[
            "-f",
            "parallels",
            "-c",
            "write -P 0x11 0 64k",
            "-c",
            "write -P 0x22 1M 100k",
            path_str(img),
        ],
    );
}

fn set_bat(img: &Path, idx: usize, val: u32) {
    let mut b = fs::read(img).unwrap();
    b[64 + 4 * idx..68 + 4 * idx].copy_from_slice(&val.to_le_bytes());
    fs::write(img, b).unwrap();
}

fn append(img: &Path, n: usize) {
    let mut b = fs::read(img).unwrap();
    b.extend(std::iter::repeat_n(0xaa, n));
    fs::write(img, b).unwrap();
}

/// A read-only check of an image with a duplicate BAT entry and a leak counts what
/// `qemu-img check` counts.
#[test]
fn check_counts() {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    let dir = scratch("check-counts");
    let img = dir.join("dup.img");
    base_image(&qemu_img, &qemu_io, &img);
    set_bat(&img, 5, 1);
    append(&img, 100_000);

    let out = output(&qemu_img, &["check", "--output=json", "-f", "parallels", path_str(&img)]);
    let theirs = String::from_utf8(out.stdout).unwrap();
    let g = BlockGraph::new();
    add_parallels(&g, "p", &img, true);
    let r = g.check("p", 0).unwrap();
    assert_eq!(r.corruptions as u64, json_num(&theirs, "corruptions").unwrap());
    assert_eq!(r.leaks as u64, json_num(&theirs, "leaks").unwrap());
    assert_eq!(r.check_errors as u64, json_num(&theirs, "check-errors").unwrap());
    assert_eq!(r.image_end_offset as u64, json_num(&theirs, "image-end-offset").unwrap());
    assert_eq!(r.bfi.allocated_clusters, json_num(&theirs, "allocated-clusters").unwrap());
    assert_eq!(r.bfi.fragmented_clusters, json_num(&theirs, "fragmented-clusters").unwrap_or(0));
    assert_eq!(r.bfi.total_clusters, json_num(&theirs, "total-clusters").unwrap());
    assert_eq!((r.corruptions, r.leaks), (1, 2));
}

/// Opening an image with a BAT entry outside the image, a duplicate and a leak read-write
/// repairs it to exactly what `qemu-img check -r all` makes of it.
#[test]
fn repair_matches_qemu() {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    let dir = scratch("repair");
    let mine = dir.join("mine.img");
    base_image(&qemu_img, &qemu_io, &mine);
    set_bat(&mine, 2, 100);
    set_bat(&mine, 3, 1);
    append(&mine, 100_000);
    let theirs = dir.join("theirs.img");
    fs::copy(&mine, &theirs).unwrap();

    // Read-only, nothing changes. (qemu-img check aborts on an assertion here.)
    let g = BlockGraph::new();
    add_parallels(&g, "p", &mine, true);
    let r = g.check("p", 0).unwrap();
    assert_eq!((r.corruptions, r.leaks, r.check_errors), (2, 2, 0));
    drop(g);

    let out = run(&qemu_img, &["check", "-r", "all", "-f", "parallels", path_str(&theirs)]);
    assert!(out.contains("2 leaked clusters") && out.contains("2 corruptions"), "{out}");

    let g = BlockGraph::new();
    add_parallels(&g, "p", &mine, false);
    let r = g.check("p", 0).unwrap();
    assert_eq!((r.corruptions, r.leaks, r.check_errors), (0, 0, 0));
    let blk = BlockBackend::new(&g, "p", RW, SHARED).unwrap();
    let data = read_all(&blk);
    // A write after the repair must not reuse a cluster that is in use (QEMU does).
    blk.pwrite(5 << 16, &pattern(65536, 9)).unwrap();
    drop(blk);
    drop(g);

    let raw = dir.join("theirs.raw");
    run(&qemu_img, &["convert", "-f", "parallels", "-O", "raw", path_str(&theirs), path_str(&raw)]);
    assert!(data == fs::read(&raw).unwrap());
    assert_eq!(data[3 << 16..4 << 16], data[..1 << 16]);
    qemu_check_clean(&qemu_img, &mine);
    let mut expect = data;
    expect[5 << 16..6 << 16].copy_from_slice(&pattern(65536, 9));
    qemu_compare(&qemu_img, &mine, &expect);

    // Before that write, the repaired files were the same.
    let mine2 = dir.join("mine2.img");
    base_image(&qemu_img, &qemu_io, &mine2);
    set_bat(&mine2, 2, 100);
    set_bat(&mine2, 3, 1);
    append(&mine2, 100_000);
    let g = BlockGraph::new();
    add_parallels(&g, "p", &mine2, false);
    drop(g);
    assert!(fs::read(&mine2).unwrap() == fs::read(&theirs).unwrap());
}

/// An image left with the `inuse` flag set is reported as not closed correctly, and opening
/// it read-write repairs it.
#[test]
fn unclean_image() {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    let dir = scratch("unclean");
    let img = dir.join("disk.img");
    base_image(&qemu_img, &qemu_io, &img);
    let mut b = fs::read(&img).unwrap();
    b[44..48].copy_from_slice(&0x746F_6E59u32.to_le_bytes());
    fs::write(&img, &b).unwrap();

    let out = output(&qemu_img, &["check", "--output=json", "-f", "parallels", path_str(&img)]);
    let theirs = String::from_utf8(out.stdout).unwrap();
    let g = BlockGraph::new();
    add_parallels(&g, "p", &img, true);
    let r = g.check("p", 0).unwrap();
    assert_eq!(r.corruptions as u64, json_num(&theirs, "corruptions").unwrap());
    assert_eq!(r.corruptions, 1);
    drop(g);

    let g = BlockGraph::new();
    add_parallels(&g, "p", &img, false);
    // While open read-write the flag is set, as in QEMU.
    assert_eq!(fs::read(&img).unwrap()[44..48], 0x746F_6E59u32.to_le_bytes());
    drop(g);
    assert_eq!(fs::read(&img).unwrap()[44..48], [0; 4]);
    qemu_check_clean(&qemu_img, &img);
}

/// Writing zeroes to whole clusters and discarding them frees the clusters, and partial
/// clusters are written as zeroes.
#[test]
fn zeroes_and_discard() {
    let Some((qemu_img, _)) = tools() else { return };
    let dir = scratch("zeroes");
    let img = dir.join("disk.img");
    let g = BlockGraph::new();
    let mut o = QDict::new();
    o.put("size", "4M");
    o.put("cluster_size", "64k");
    g.create_image("parallels", path_str(&img), &mut o).unwrap();
    // Discards only reach the driver with discard=unmap, BDRV_O_UNMAP.
    g.blockdev_add(from_json::<BlockdevOptions>(&format!(
        r#"{{"driver": "parallels", "node-name": "p", "discard": "unmap",
            "file": {{"driver": "file", "filename": "{}"}}}}"#,
        path_str(&img)
    )))
    .unwrap();
    let blk = BlockBackend::new(&g, "p", RW, SHARED).unwrap();
    let mut expect = pattern(1 << 20, 4);
    expect.resize(4 << 20, 0);
    blk.pwrite(0, &expect[..1 << 20]).unwrap();
    blk.pwrite_zeroes(64 << 10, 128 << 10, true).unwrap();
    blk.pdiscard(512 << 10, 64 << 10).unwrap();
    blk.pwrite_zeroes(1000, 3000, false).unwrap();
    expect[64 << 10..192 << 10].fill(0);
    expect[512 << 10..576 << 10].fill(0);
    expect[1000..4000].fill(0);
    let got = read_all(&blk);
    let first = got.iter().zip(&expect).position(|(a, b)| a != b);
    assert!(first.is_none(), "differs at {first:?}");
    let map = our_map(&blk);
    assert_eq!(map.iter().map(|m| m.1).sum::<u64>(), (16 - 3) << 16);
    drop(blk);
    drop(g);
    qemu_check_clean(&qemu_img, &img);
    qemu_compare(&qemu_img, &img, &expect);
    assert_eq!(map, qemu_map(&qemu_img, &img));
}

#[test]
fn open_errors() {
    let dir = scratch("open-errors");
    let open = |bytes: &[u8]| {
        let p = dir.join("bad.img");
        fs::write(&p, bytes).unwrap();
        let g = BlockGraph::new();
        let e = g
            .blockdev_add(from_json::<BlockdevOptions>(&format!(
                r#"{{"driver": "parallels", "node-name": "p", "read-only": true,
                    "file": {{"driver": "file", "filename": "{}"}}}}"#,
                path_str(&p)
            )))
            .unwrap_err();
        e.message().to_string()
    };
    let header = |tracks: u32, bat: u32, sectors: u64| {
        let mut b = vec![0u8; 4096];
        b[..16].copy_from_slice(b"WithouFreSpacExt");
        b[16..20].copy_from_slice(&2u32.to_le_bytes());
        b[28..32].copy_from_slice(&tracks.to_le_bytes());
        b[32..36].copy_from_slice(&bat.to_le_bytes());
        b[36..44].copy_from_slice(&sectors.to_le_bytes());
        b
    };
    assert_eq!(open(&[0u8; 4096]), "Image not in Parallels format");
    let mut b = header(8, 1, 8);
    b[0] = b'X';
    assert_eq!(open(&b), "Image not in Parallels format");
    assert_eq!(open(&header(0, 1, 8)), "Invalid image: Zero sectors per track");
    assert_eq!(open(&header(1 << 22, 1, 8)), "Invalid image: Too big cluster");
    assert_eq!(open(&header(8, 1 << 30, 8)), "Catalog too large");
    assert_eq!(
        open(&header(8, 1, 9)),
        "Invalid image: Catalog size too small for advertised disk size"
    );
    let mut b = header(8, 1, 8);
    b[56..64].copy_from_slice(&(1u64 << 60).to_le_bytes());
    assert_eq!(open(&b), "Invalid image: Too big offset");
}
