// SPDX-License-Identifier: GPL-2.0-or-later

//! The `qed` format driver against QEMU: images `qemu-img` makes and `qemu-io` writes read the
//! same here, writes here (allocating, copy-on-write from a raw backing file, zero writes)
//! pass `qemu-img check` and compare equal in `qemu-img compare`, images made here are byte
//! for byte what `qemu-img create` makes, and check results, the dirty flag, `map`, `info`
//! and the error messages agree. The interop tests skip themselves when `qemu-img` or
//! `qemu-io` is not installed.

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
const KIB: usize = 1 << 10;
const MIB: usize = 1 << 20;
const CLUSTER: usize = 64 * KIB;

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
    let dir = base.join("ruvm-block-qed").join(test);
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

/// Runs `prog` with `args` and returns the exit code and its output.
fn run_status(prog: &Path, args: &[&str]) -> (i32, String) {
    let out = Command::new(prog).args(args).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    (out.status.code().unwrap_or(-1), text + &String::from_utf8_lossy(&out.stderr))
}

/// Runs `prog` with `args`, which must succeed, and returns its output.
fn run(prog: &Path, args: &[&str]) -> String {
    let (code, out) = run_status(prog, args);
    assert_eq!(code, 0, "{} {args:?} failed: {out}", prog.display());
    out
}

/// The number after `"key": ` in JSON output from `qemu-img`, the first one.
fn json_num(out: &str, key: &str) -> Option<u64> {
    let pat = format!("\"{key}\": ");
    let i = out.find(&pat)? + pat.len();
    let digits: String = out[i..].chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
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

/// `blockdev-add` of a qed node `q` over `path`, and whatever backing file it names.
fn add_qed(g: &BlockGraph, path: &Path, read_only: bool) -> ruvm_base::Result<()> {
    g.blockdev_add(from_json::<BlockdevOptions>(&format!(
        r#"{{"driver": "qed", "node-name": "q", "read-only": {read_only},
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

/// The features field of the QED header in `path`.
fn features(path: &Path) -> u64 {
    let b = fs::read(path).unwrap();
    u64::from_le_bytes(b[16..24].try_into().unwrap())
}

/// `qemu-img compare` of `img` against a raw file holding `expected`.
fn compare_raw(qemu_img: &Path, img: &Path, expected: &[u8]) {
    let raw = img.with_extension("expected");
    fs::write(&raw, expected).unwrap();
    let out = run(qemu_img, &["compare", "-f", "qed", "-F", "raw", path_str(img), path_str(&raw)]);
    assert!(out.contains("Images are identical."), "{out}");
}

#[test]
fn qemu_image_read_and_written_here() {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    let dir = scratch("qemu-image");
    let img = dir.join("disk.qed");
    run(&qemu_img, &["create", "-f", "qed", path_str(&img), "8M"]);
    run(
        &qemu_io,
        &[
            "-f",
            "qed",
            "-c",
            "write -P 0x55 0 64k",
            "-c",
            "write -P 0xaa 1049088 4k",
            "-c",
            "write -z 2M 128k",
            path_str(&img),
        ],
    );

    let mut expected = vec![0u8; 8 * MIB];
    expected[..CLUSTER].fill(0x55);
    expected[MIB + 512..MIB + 512 + 4 * KIB].fill(0xaa);

    {
        let g = BlockGraph::new();
        let name = g.open_image(Some(path_str(&img)), QDict::new()).unwrap();
        assert_eq!(g.node(&name).unwrap().driver, "qed");
    }

    let g = BlockGraph::new();
    add_qed(&g, &img, false).unwrap();
    let blk = BlockBackend::new(&g, "q", RW, SHARED).unwrap();
    assert!(read_all(&blk) == expected);

    let e = blk.map_entry(2 * MIB as u64, 128 * KIB as u64).unwrap();
    assert!(e.zero && !e.data, "{e:?}");
    let e = blk.map_entry(0, 8 * MIB as u64).unwrap();
    assert!(e.data && e.offset.is_some(), "{e:?}");

    // Writes across clusters and into both allocated and new clusters.
    let a = pattern(200 * KIB + 100, 3);
    blk.pwrite(3 * MIB as u64 + 700, &a).unwrap();
    expected[3 * MIB + 700..3 * MIB + 700 + a.len()].copy_from_slice(&a);
    let b = pattern(10 * KIB, 9);
    blk.pwrite(30 * KIB as u64, &b).unwrap();
    expected[30 * KIB..40 * KIB].copy_from_slice(&b);
    // The last sector of the image.
    let c = pattern(512, 77);
    blk.pwrite(8 * MIB as u64 - 512, &c).unwrap();
    expected[8 * MIB - 512..].copy_from_slice(&c);
    blk.pwrite_zeroes(0, CLUSTER as u64, false).unwrap();
    expected[..CLUSTER].fill(0);
    blk.pwrite_zeroes(5 * MIB as u64, 2 * CLUSTER as u64, false).unwrap();
    assert!(read_all(&blk) == expected);

    // Allocating writes mark the image dirty until the flush.
    assert_ne!(features(&img) & 2, 0);
    blk.flush().unwrap();
    assert_eq!(features(&img) & 2, 0);
    drop(blk);
    g.blockdev_del("q").unwrap();

    let out = run(&qemu_img, &["check", "-f", "qed", path_str(&img)]);
    assert!(out.contains("No errors were found on the image."), "{out}");
    compare_raw(&qemu_img, &img, &expected);

    // qemu-img map and ours agree on what is data and what reads as zeroes.
    let map = run(&qemu_img, &["map", "-f", "qed", "--output=json", path_str(&img)]);
    let g = BlockGraph::new();
    add_qed(&g, &img, true).unwrap();
    let blk = BlockBackend::new(&g, "q", BLK_PERM_CONSISTENT_READ, SHARED).unwrap();
    for line in map.lines().filter(|l| l.contains("\"start\"")) {
        let start = json_num(line, "start").unwrap();
        let length = json_num(line, "length").unwrap();
        let e = blk.map_entry(start, length).unwrap();
        assert_eq!(e.length, length, "{line}");
        assert_eq!(e.data, line.contains("\"data\": true"), "{line}");
        assert_eq!(e.zero, line.contains("\"zero\": true"), "{line}");
        assert_eq!(e.offset, json_num(line, "offset"), "{line}");
    }
}

#[test]
fn backing_file_copy_on_write() {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    let dir = scratch("backing");
    let base = dir.join("base.raw");
    let base_data = pattern(4 * MIB, 11);
    fs::write(&base, &base_data).unwrap();
    let img = dir.join("overlay.qed");
    run(
        &qemu_img,
        &["create", "-f", "qed", "-b", path_str(&base), "-F", "raw", path_str(&img), "4M"],
    );
    run(&qemu_io, &["-f", "qed", "-c", "write -P 0x33 1M 4k", path_str(&img)]);
    let mut expected = base_data.clone();
    expected[MIB..MIB + 4 * KIB].fill(0x33);

    let g = BlockGraph::new();
    add_qed(&g, &img, false).unwrap();
    let nodes = g.query_named_block_nodes(Some(true)).unwrap();
    let n = nodes.iter().find(|n| n.node_name == "q").unwrap();
    assert_eq!(n.image.backing_filename.as_deref(), Some(path_str(&base)));
    assert_eq!(n.image.backing_filename_format.as_deref(), Some("raw"));
    assert_eq!(n.image.dirty_flag, Some(false));
    assert_eq!(n.image.cluster_size, Some(CLUSTER as i64));

    let blk = BlockBackend::new(&g, "q", RW, SHARED).unwrap();
    assert!(read_all(&blk) == expected);
    // Partial cluster writes copy the rest of the cluster from the backing file.
    let a = pattern(300, 200);
    blk.pwrite(100, &a).unwrap();
    expected[100..400].copy_from_slice(&a);
    let b = pattern(70 * KIB, 5);
    blk.pwrite(2 * MIB as u64 + 1000, &b).unwrap();
    expected[2 * MIB + 1000..2 * MIB + 1000 + b.len()].copy_from_slice(&b);
    blk.pwrite_zeroes(3 * MIB as u64, CLUSTER as u64, false).unwrap();
    expected[3 * MIB..3 * MIB + CLUSTER].fill(0);
    assert!(read_all(&blk) == expected);
    // With a backing file the image is never marked dirty.
    assert_eq!(features(&img) & 2, 0);
    // A qed image with a backing file does not read as zeroes where it has no data.
    assert!(!blk.has_zero_init());
    drop(blk);
    g.blockdev_del("q").unwrap();

    let out = run(&qemu_img, &["check", "-f", "qed", path_str(&img)]);
    assert!(out.contains("No errors were found on the image."), "{out}");
    compare_raw(&qemu_img, &img, &expected);
    // The base is untouched.
    assert!(fs::read(&base).unwrap() == base_data);
}

#[test]
fn created_here_like_qemu_img() {
    let Some((qemu_img, _)) = tools() else { return };
    let dir = scratch("create");
    let base = dir.join("base.raw");
    fs::write(&base, vec![0u8; MIB]).unwrap();

    // Same options, same bytes.
    let cases: [(&[(&str, &str)], &str); 4] = [
        (&[("size", "10M")], "-osize=10M"),
        (
            &[("size", "3M"), ("cluster_size", "4096"), ("table_size", "2")],
            "-osize=3M,cluster_size=4096,table_size=2",
        ),
        // The size is rounded up to whole sectors.
        (&[("size", "1000")], "-osize=1000"),
        (&[("size", "1M"), ("backing_file", path_str(&base)), ("backing_fmt", "raw")], ""),
    ];
    for (i, (opts, qemu_opts)) in cases.iter().enumerate() {
        let mine = dir.join(format!("mine{i}.qed"));
        let theirs = dir.join(format!("theirs{i}.qed"));
        let g = BlockGraph::new();
        let mut o = QDict::new();
        for (k, v) in *opts {
            o.put(*k, *v);
        }
        g.create_image("qed", path_str(&mine), &mut o).unwrap();
        if qemu_opts.is_empty() {
            run(
                &qemu_img,
                &[
                    "create",
                    "-f",
                    "qed",
                    "-b",
                    path_str(&base),
                    "-F",
                    "raw",
                    path_str(&theirs),
                    "1M",
                ],
            );
        } else {
            run(&qemu_img, &["create", "-f", "qed", qemu_opts, path_str(&theirs)]);
        }
        assert!(fs::read(&mine).unwrap() == fs::read(&theirs).unwrap(), "case {i}");
        let out = run(&qemu_img, &["check", "-f", "qed", path_str(&mine)]);
        assert!(out.contains("No errors were found on the image."), "{out}");
    }

    // blockdev-create, then written here and checked by qemu-img.
    let img = dir.join("bc.qed");
    let g = BlockGraph::new();
    g.blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
        r#"{{"driver": "file", "filename": "{}", "size": 0}}"#,
        path_str(&img)
    )))
    .unwrap();
    g.blockdev_add(from_json::<BlockdevOptions>(&format!(
        r#"{{"driver": "file", "node-name": "f", "filename": "{}"}}"#,
        path_str(&img)
    )))
    .unwrap();
    g.blockdev_create(from_json::<BlockdevCreateOptions>(
        r#"{"driver": "qed", "file": "f", "size": 4194304, "cluster-size": 8192,
            "table-size": 2}"#,
    ))
    .unwrap();
    g.blockdev_del("f").unwrap();
    let info = run(&qemu_img, &["info", "--output=json", path_str(&img)]);
    assert!(info.contains("\"format\": \"qed\""), "{info}");
    assert_eq!(info_num(&info, "virtual-size"), Some(4 * MIB as u64));
    assert_eq!(info_num(&info, "cluster-size"), Some(8192));

    add_qed(&g, &img, false).unwrap();
    let blk = BlockBackend::new(&g, "q", RW | BLK_PERM_RESIZE, SHARED).unwrap();
    assert!(blk.has_zero_init());
    let a = pattern(3 * MIB, 1);
    blk.pwrite(512, &a).unwrap();
    blk.truncate(6 * MIB as u64).unwrap();
    let mut expected = vec![0u8; 6 * MIB];
    expected[512..512 + a.len()].copy_from_slice(&a);
    assert!(read_all(&blk) == expected);
    drop(blk);
    g.blockdev_del("q").unwrap();
    let out = run(&qemu_img, &["check", "-f", "qed", path_str(&img)]);
    assert!(out.contains("No errors were found on the image."), "{out}");
    compare_raw(&qemu_img, &img, &expected);
}

#[test]
fn create_errors() {
    let dir = scratch("create-errors");
    let img = dir.join("bad.qed");
    let create = |opts: &[(&str, &str)]| {
        let g = BlockGraph::new();
        let mut o = QDict::new();
        for (k, v) in opts {
            o.put(*k, *v);
        }
        g.create_image("qed", path_str(&img), &mut o).unwrap_err().message().to_string()
    };
    assert_eq!(
        create(&[("size", "1M"), ("cluster_size", "1000")]),
        "QED cluster size must be within range [4096, 67108864] and power of 2"
    );
    assert_eq!(
        create(&[("size", "1M"), ("table_size", "3")]),
        "QED table size must be within range [1, 16] and power of 2"
    );
    assert_eq!(
        create(&[("size", "1T"), ("cluster_size", "4096"), ("table_size", "1")]),
        "QED image size must be a non-zero multiple of cluster size and less than 1073741824 \
         bytes"
    );
}

#[test]
fn truncate_and_change_backing_file() {
    let Some((qemu_img, _)) = tools() else { return };
    let dir = scratch("truncate");
    let img = dir.join("disk.qed");
    run(&qemu_img, &["create", "-f", "qed", path_str(&img), "1M"]);
    let g = BlockGraph::new();
    add_qed(&g, &img, false).unwrap();
    let blk = BlockBackend::new(&g, "q", RW | BLK_PERM_RESIZE, SHARED).unwrap();
    assert_eq!(
        blk.truncate(512 * KIB as u64).unwrap_err().message(),
        "Shrinking images is currently not supported"
    );
    assert_eq!(blk.truncate(MIB as u64 + 1).unwrap_err().message(), "Invalid image size specified");
    blk.truncate(2 * MIB as u64).unwrap();
    assert_eq!(blk.getlength().unwrap(), 2 * MIB as u64);

    blk.change_backing_file(Some("base.raw"), Some("raw"), false).unwrap();
    // No room for a name that does not fit in the header cluster.
    let long = "x".repeat(CLUSTER);
    assert_eq!(
        blk.change_backing_file(Some(&long), None, false).unwrap_err().raw_os_error(),
        Some(libc::ENOSPC)
    );
    drop(blk);
    g.blockdev_del("q").unwrap();
    let info = run(&qemu_img, &["info", "-U", "--output=json", path_str(&img)]);
    assert_eq!(info_num(&info, "virtual-size"), Some(2 * MIB as u64));
    assert!(info.contains("\"backing-filename\": \"base.raw\""), "{info}");
    assert!(info.contains("\"backing-filename-format\": \"raw\""), "{info}");
}

#[test]
fn dirty_flag_and_repair() {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    let dir = scratch("dirty");
    let img = dir.join("disk.qed");
    run(&qemu_img, &["create", "-f", "qed", path_str(&img), "4M"]);
    run(&qemu_io, &["-f", "qed", "-c", "write -P 0x44 0 128k", path_str(&img)]);
    let mut bytes = fs::read(&img).unwrap();
    bytes[16] |= 2;
    // A leaked cluster at the end.
    bytes.extend_from_slice(&[0u8; CLUSTER]);
    fs::write(&img, &bytes).unwrap();

    let (code, out) =
        run_status(&qemu_img, &["check", "-f", "qed", "--output=json", path_str(&img)]);
    assert_eq!(code, 3, "{out}");
    assert_eq!(json_num(&out, "leaks"), Some(1), "{out}");

    // Read-only: dirty, not repaired. QEMU's need-check timer clears the flag in memory when
    // the node is first drained, even read-only, so its `info` says not dirty here.
    {
        let g = BlockGraph::new();
        add_qed(&g, &img, true).unwrap();
        let nodes = g.query_named_block_nodes(Some(true)).unwrap();
        let n = nodes.iter().find(|n| n.node_name == "q").unwrap();
        assert_eq!(n.image.dirty_flag, Some(true));
        let r = g.check("q", 0).unwrap();
        assert_eq!((r.corruptions, r.leaks, r.check_errors), (0, 1, 0));
    }
    assert_ne!(features(&img) & 2, 0);

    // Read-write: checked and marked clean while opening.
    {
        let g = BlockGraph::new();
        add_qed(&g, &img, false).unwrap();
        assert_eq!(features(&img) & 2, 0);
        let blk = BlockBackend::new(&g, "q", RW, SHARED).unwrap();
        let mut buf = vec![0u8; 128 * KIB];
        blk.pread(0, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0x44));
    }
    let info = run(&qemu_img, &["info", "--output=json", path_str(&img)]);
    assert!(info.contains("\"dirty-flag\": false"), "{info}");
}

#[test]
fn check_matches_qemu() {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    let dir = scratch("check");
    let img = dir.join("disk.qed");
    run(&qemu_img, &["create", "-f", "qed", path_str(&img), "4M"]);
    run(&qemu_io, &["-f", "qed", "-c", "write 0 64k", "-c", "write 256k 128k", path_str(&img)]);
    let mut bytes = fs::read(&img).unwrap();
    let l2 = u64::from_le_bytes(bytes[CLUSTER..CLUSTER + 8].try_into().unwrap()) as usize;
    let e0 = u64::from_le_bytes(bytes[l2..l2 + 8].try_into().unwrap());
    // Entry 1 points past the end of the file, entry 2 at the cluster of entry 0.
    bytes[l2 + 8..l2 + 16].copy_from_slice(&(1u64 << 40).to_le_bytes());
    bytes[l2 + 16..l2 + 24].copy_from_slice(&e0.to_le_bytes());
    fs::write(&img, &bytes).unwrap();

    let (code, out) =
        run_status(&qemu_img, &["check", "-f", "qed", "--output=json", path_str(&img)]);
    assert_eq!(code, 2, "{out}");
    let g = BlockGraph::new();
    add_qed(&g, &img, true).unwrap();
    let r = g.check("q", 0).unwrap();
    assert_eq!(Some(r.corruptions as u64), json_num(&out, "corruptions"), "{out}");
    assert_eq!(Some(r.leaks as u64), json_num(&out, "leaks").or(Some(0)), "{out}");
    assert_eq!(
        Some(r.bfi.allocated_clusters),
        json_num(&out, "allocated-clusters"),
        "{out}"
    );
    assert_eq!(Some(r.bfi.total_clusters), json_num(&out, "total-clusters"), "{out}");
    assert_eq!(
        Some(r.bfi.fragmented_clusters),
        json_num(&out, "fragmented-clusters"),
        "{out}"
    );
    assert_eq!(
        Some(r.image_end_offset as u64),
        json_num(&out, "image-end-offset").or(Some(0)),
        "{out}"
    );
    g.blockdev_del("q").unwrap();

    // Repair here: the bad entry goes, and qemu-img agrees on what is left.
    add_qed(&g, &img, false).unwrap();
    let r = g.check("q", BDRV_FIX_ERRORS | BDRV_FIX_LEAKS).unwrap();
    assert_eq!(r.corruptions_fixed, 1);
    g.blockdev_del("q").unwrap();
    let fixed = fs::read(&img).unwrap();
    assert_eq!(u64::from_le_bytes(fixed[l2 + 8..l2 + 16].try_into().unwrap()), 0);
    let (code2, out2) =
        run_status(&qemu_img, &["check", "-f", "qed", "--output=json", path_str(&img)]);
    let g = BlockGraph::new();
    add_qed(&g, &img, true).unwrap();
    let r = g.check("q", 0).unwrap();
    assert_eq!(Some(r.corruptions as u64), json_num(&out2, "corruptions"), "{out2}");
    assert_eq!(code2 == 0, r.corruptions == 0 && r.leaks == 0, "{out2}");
}

#[test]
fn open_errors() {
    let dir = scratch("open-errors");
    let open = |patch: &dyn Fn(&mut Vec<u8>)| {
        let mut h = vec![0u8; 3 * CLUSTER];
        h[0..4].copy_from_slice(b"QED\0");
        h[4..8].copy_from_slice(&(CLUSTER as u32).to_le_bytes());
        h[8..12].copy_from_slice(&2u32.to_le_bytes());
        h[12..16].copy_from_slice(&1u32.to_le_bytes());
        h[40..48].copy_from_slice(&(CLUSTER as u64).to_le_bytes());
        h[48..56].copy_from_slice(&(MIB as u64).to_le_bytes());
        patch(&mut h);
        let p = dir.join("bad.qed");
        fs::write(&p, &h).unwrap();
        let g = BlockGraph::new();
        add_qed(&g, &p, true).err().map(|e| e.message().to_string())
    };
    assert_eq!(open(&|_| {}), None);
    assert_eq!(open(&|h| h[0] = b'X').as_deref(), Some("Image not in QED format"));
    assert_eq!(open(&|h| h[17] = 1).as_deref(), Some("Unsupported QED features: 100"));
    assert_eq!(open(&|h| h[5] = 0x11).as_deref(), Some("QED cluster size is invalid"));
    assert_eq!(open(&|h| h[8] = 3).as_deref(), Some("QED table size is invalid"));
    assert_eq!(open(&|h| h[48] = 1).as_deref(), Some("QED image size is invalid"));
    assert_eq!(open(&|h| h[42] = 9).as_deref(), Some("QED table offset is invalid"));
    // Like QEMU, a table of one cluster never passes the table offset check.
    assert_eq!(open(&|h| h[8] = 1).as_deref(), Some("QED table offset is invalid"));
    assert_eq!(
        open(&|h| {
            h[16] = 1;
            h[56..60].copy_from_slice(&64u32.to_le_bytes());
            h[60..64].copy_from_slice(&(CLUSTER as u32).to_le_bytes());
        })
        .as_deref(),
        Some("QED backing filename offset is invalid")
    );
    // A short file reads as zeroes past its end.
    assert_eq!(open(&|h| h.truncate(2)).as_deref(), Some("Image not in QED format"));
    // Too short for the L1 table the header points at.
    assert_eq!(
        open(&|h| h.truncate(2 * CLUSTER + 100)).as_deref(),
        Some("QED table offset is invalid")
    );
}
