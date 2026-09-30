// SPDX-License-Identifier: GPL-2.0-or-later

//! The `qcow2` format driver against QEMU, in both directions: images `qemu-img` makes and
//! `qemu-io` writes (plain, compressed, zeroed, with a backing file, LUKS encrypted) read the
//! same here, and images written here pass `qemu-img check`, compare equal in `qemu-img
//! compare` and read back the same in `qemu-io`. Images created here are byte for byte what
//! `qemu-img create` makes, `map` agrees, check finds and repairs the same leaks, internal
//! snapshots and persistent dirty bitmaps made on either side work on the other, and amend
//! and resize leave images QEMU accepts. The tests skip themselves when `qemu-img` or
//! `qemu-io` is not installed.

#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Once};

use ruvm_block::{
    BDRV_FIX_ERRORS, BDRV_FIX_LEAKS, BLK_PERM_CONSISTENT_READ, BLK_PERM_RESIZE, BLK_PERM_WRITE,
    BlockBackend, BlockGraph,
};
use ruvm_qapi::types::{BlockDirtyBitmap, BlockDirtyBitmapAdd, BlockdevOptions};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};
use ruvm_qapi::{QDict, json};

const RW: u64 = BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE;
const SHARED: u64 = BLK_PERM_CONSISTENT_READ;
const KIB: usize = 1 << 10;
const MIB: usize = 1 << 20;
const CLUSTER: usize = 64 * KIB;
const PASSWORD: &str = "123456";

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

/// The `sec0` secret and a fast PBKDF, once per test binary.
fn setup_secrets() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        ruvm_crypto::pbkdf::set_iters_per_second_override(Some(1000));
        ruvm_crypto::secret::secret_object_add_global(&format!("secret,id=sec0,data={PASSWORD}"))
            .unwrap();
    });
}

fn scratch(test: &str) -> PathBuf {
    let base = option_env!("CARGO_TARGET_TMPDIR").map_or_else(std::env::temp_dir, PathBuf::from);
    let dir = base.join("ruvm-block-qcow2").join(test);
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

fn from_json<T: Visit + Default>(s: &str) -> T {
    let mut v = QObjectInputVisitor::new(json::from_str(s).unwrap());
    let mut o = T::default();
    T::visit(&mut v, None, &mut o).unwrap();
    o
}

/// `blockdev-add` of a qcow2 node `q` over `path`, with `extra` JSON members, and whatever
/// backing file it names.
fn add_qcow2(g: &BlockGraph, path: &Path, read_only: bool, extra: &str) -> ruvm_base::Result<()> {
    g.blockdev_add(from_json::<BlockdevOptions>(&format!(
        r#"{{"driver": "qcow2", "node-name": "q", "read-only": {read_only}, {extra}
            "file": {{"driver": "file", "filename": "{}"}}}}"#,
        path_str(path)
    )))
}

fn open_rw(g: &BlockGraph, path: &Path) -> Arc<BlockBackend> {
    add_qcow2(g, path, false, r#""discard": "unmap","#).unwrap();
    BlockBackend::new(g, "q", RW | BLK_PERM_RESIZE, SHARED).unwrap()
}

fn read_all(blk: &BlockBackend) -> Vec<u8> {
    let len = blk.getlength().unwrap() as usize;
    let mut buf = vec![0u8; len];
    blk.pread(0, &mut buf).unwrap();
    buf
}

/// `qemu-img check` must find nothing wrong.
fn check_clean(qemu_img: &Path, img: &Path) {
    let out = run(qemu_img, &["check", "-f", "qcow2", path_str(img)]);
    assert!(out.contains("No errors were found on the image."), "{out}");
}

/// `qemu-img compare` of `img` against a raw file holding `expected`.
fn compare_raw(qemu_img: &Path, img: &Path, expected: &[u8]) {
    let raw = img.with_extension("expected");
    fs::write(&raw, expected).unwrap();
    let out =
        run(qemu_img, &["compare", "-f", "qcow2", "-F", "raw", path_str(img), path_str(&raw)]);
    assert!(out.contains("Images are identical."), "{out}");
}

/// The 64 bit big-endian field at `off` of the file.
fn be64(path: &Path, off: usize) -> u64 {
    let b = fs::read(path).unwrap();
    u64::from_be_bytes(b[off..off + 8].try_into().unwrap())
}

/// The ranges of `bitmap` in `img` that are dirty, read through `qemu-nbd` the way the
/// iotests do. `None` when `qemu-nbd` is missing.
fn dirty_ranges(qemu_img: &Path, img: &Path, bitmap: &str) -> Option<Vec<(u64, u64)>> {
    let qemu_nbd = tool("qemu-nbd")?;
    let sock = img.with_extension("sock");
    let _ = fs::remove_file(&sock);
    let mut child = Command::new(&qemu_nbd)
        .args(["-r", "-f", "qcow2", "-B", bitmap, "-k", path_str(&sock), path_str(img)])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    for _ in 0..100 {
        if sock.exists() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let opts = format!(
        "driver=nbd,server.type=unix,server.path={},x-dirty-bitmap=qemu:dirty-bitmap:{bitmap}",
        path_str(&sock)
    );
    let out = run(qemu_img, &["map", "--output=json", "--image-opts", &opts]);
    let _ = child.kill();
    let _ = child.wait();
    // The bitmap reports dirty areas as holes.
    Some(
        out.lines()
            .filter(|l| l.contains("\"data\": false"))
            .map(|l| (json_num(l, "start").unwrap(), json_num(l, "length").unwrap()))
            .collect(),
    )
}

#[test]
fn qemu_image_read_and_written_here() {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    let dir = scratch("qemu-image");
    let img = dir.join("disk.qcow2");
    run(&qemu_img, &["create", "-f", "qcow2", path_str(&img), "8M"]);
    run(
        &qemu_io,
        &[
            "-f",
            "qcow2",
            "-c",
            "write -P 0x55 0 64k",
            "-c",
            "write -P 0xaa 1049088 4k",
            "-c",
            "write -z 2M 128k",
            "-c",
            "write -c -P 0x33 4M 64k",
            path_str(&img),
        ],
    );

    let mut expected = vec![0u8; 8 * MIB];
    expected[..CLUSTER].fill(0x55);
    expected[MIB + 512..MIB + 512 + 4 * KIB].fill(0xaa);
    expected[4 * MIB..4 * MIB + CLUSTER].fill(0x33);

    {
        let g = BlockGraph::new();
        let name = g.open_image(Some(path_str(&img)), QDict::new()).unwrap();
        assert_eq!(g.node(&name).unwrap().driver, "qcow2");
    }

    let g = BlockGraph::new();
    let blk = open_rw(&g, &img);
    assert!(read_all(&blk) == expected);

    let e = blk.map_entry(2 * MIB as u64, 128 * KIB as u64).unwrap();
    assert!(e.zero && !e.data, "{e:?}");
    let e = blk.map_entry(4 * MIB as u64, CLUSTER as u64).unwrap();
    assert!(e.data && e.compressed, "{e:?}");

    // Writes across clusters, into allocated, compressed and new clusters.
    let a = pattern(200 * KIB + 100, 3);
    blk.pwrite(3 * MIB as u64 + 700, &a).unwrap();
    expected[3 * MIB + 700..3 * MIB + 700 + a.len()].copy_from_slice(&a);
    let b = pattern(10 * KIB, 9);
    blk.pwrite(30 * KIB as u64, &b).unwrap();
    expected[30 * KIB..40 * KIB].copy_from_slice(&b);
    let c = pattern(512, 77);
    blk.pwrite(4 * MIB as u64 + 1024, &c).unwrap();
    expected[4 * MIB + 1024..4 * MIB + 1536].copy_from_slice(&c);
    // The last sector of the image.
    blk.pwrite(8 * MIB as u64 - 512, &c).unwrap();
    expected[8 * MIB - 512..].copy_from_slice(&c);
    // A compressed cluster written here.
    let d = vec![0x42u8; CLUSTER];
    blk.pwrite_compressed(6 * MIB as u64, &d).unwrap();
    expected[6 * MIB..6 * MIB + CLUSTER].copy_from_slice(&d);
    blk.pwrite_zeroes(0, CLUSTER as u64, true).unwrap();
    expected[..CLUSTER].fill(0);
    blk.pwrite_zeroes(5 * MIB as u64, 2 * CLUSTER as u64, false).unwrap();
    blk.pdiscard(MIB as u64, CLUSTER as u64).unwrap();
    expected[MIB..MIB + CLUSTER].fill(0);
    assert!(read_all(&blk) == expected);
    blk.flush().unwrap();
    drop(blk);
    g.blockdev_del("q").unwrap();

    check_clean(&qemu_img, &img);
    compare_raw(&qemu_img, &img, &expected);

    // qemu-img map and ours agree.
    let map = run(&qemu_img, &["map", "-f", "qcow2", "--output=json", path_str(&img)]);
    assert!(map.contains("\"compressed\": true"), "{map}");
    let g = BlockGraph::new();
    add_qcow2(&g, &img, true, "").unwrap();
    let blk = BlockBackend::new(&g, "q", BLK_PERM_CONSISTENT_READ, SHARED).unwrap();
    for line in map.lines().filter(|l| l.contains("\"start\"")) {
        let start = json_num(line, "start").unwrap();
        let length = json_num(line, "length").unwrap();
        let e = blk.map_entry(start, length).unwrap();
        assert_eq!(e.length, length, "{line}");
        assert_eq!(e.data, line.contains("\"data\": true"), "{line}");
        assert_eq!(e.zero, line.contains("\"zero\": true"), "{line}");
        assert_eq!(e.compressed, line.contains("\"compressed\": true"), "{line}");
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
    let top = dir.join("top.qcow2");
    run(
        &qemu_img,
        &["create", "-f", "qcow2", "-b", path_str(&base), "-F", "raw", path_str(&top), "4M"],
    );

    let mut expected = base_data.clone();
    let g = BlockGraph::new();
    let blk = open_rw(&g, &top);
    // Partial cluster writes copy the rest of the cluster from the backing file.
    let a = pattern(3000, 200);
    blk.pwrite(100_000, &a).unwrap();
    expected[100_000..103_000].copy_from_slice(&a);
    blk.pwrite_zeroes(2 * MIB as u64, 4096, false).unwrap();
    expected[2 * MIB..2 * MIB + 4096].fill(0);
    assert!(read_all(&blk) == expected);
    drop(blk);
    g.blockdev_del("q").unwrap();

    check_clean(&qemu_img, &top);
    compare_raw(&qemu_img, &top, &expected);
    let out = run(
        &qemu_io,
        &["-f", "qcow2", "-c", "read -P 0 2M 4k", "-c", "read -P 11 0 1", path_str(&top)],
    );
    assert!(!out.contains("Pattern verification failed"), "{out}");
    let map = run(&qemu_img, &["map", "-f", "qcow2", "--output=json", path_str(&top)]);
    assert!(map.contains("\"depth\": 1"), "{map}");
}

#[test]
fn created_here_like_qemu_img() {
    let Some((qemu_img, _)) = tools() else { return };
    let dir = scratch("create");
    let base = dir.join("base.raw");
    fs::write(&base, vec![0u8; MIB]).unwrap();
    let backing = format!("backing_file={},backing_fmt=raw", path_str(&base));

    let cases = [
        "size=10M".to_string(),
        "size=1000".to_string(),
        "size=64M,cluster_size=4096,refcount_bits=64".to_string(),
        "size=16M,compat=0.10".to_string(),
        "size=16M,lazy_refcounts=on".to_string(),
        "size=16M,extended_l2=on".to_string(),
        "size=8M,preallocation=metadata".to_string(),
        "size=8M,cluster_size=2M,refcount_bits=1".to_string(),
        format!("size=1M,{backing}"),
    ];
    for (i, opts) in cases.iter().enumerate() {
        let mine = dir.join(format!("mine{i}.qcow2"));
        let theirs = dir.join(format!("theirs{i}.qcow2"));
        let g = BlockGraph::new();
        let mut o = QDict::new();
        for kv in opts.split(',') {
            let (k, v) = kv.split_once('=').unwrap();
            o.put(k, v);
        }
        g.create_image("qcow2", path_str(&mine), &mut o).unwrap();
        run(&qemu_img, &["create", "-f", "qcow2", "-o", opts, path_str(&theirs)]);
        assert!(fs::read(&mine).unwrap() == fs::read(&theirs).unwrap(), "{opts}");
        check_clean(&qemu_img, &mine);
    }
}

#[test]
fn create_errors() {
    let dir = scratch("create-errors");
    let img = dir.join("x.qcow2");
    let cases = [
        ("size=1M,cluster_size=1000", "Cluster size must be a power of two between 512 and 2048k"),
        (
            "size=1M,compat=0.10,refcount_bits=8",
            "Different refcount widths than 16 bits require compatibility level 1.1 or above (use version=v3 or greater)",
        ),
        (
            "size=1M,compat=0.10,lazy_refcounts=on",
            "Lazy refcounts only supported with compatibility level 1.1 and above (use version=v3 or greater)",
        ),
        (
            "size=1M,refcount_bits=3",
            "Refcount width must be a power of two and may not exceed 64 bits",
        ),
        ("size=1M,keep_data_file=on", "Must not use 'keep_data_file=on' without 'data_file'"),
    ];
    for (opts, msg) in cases {
        let g = BlockGraph::new();
        let mut o = QDict::new();
        for kv in opts.split(',') {
            let (k, v) = kv.split_once('=').unwrap();
            o.put(k, v);
        }
        let e = g.create_image("qcow2", path_str(&img), &mut o).unwrap_err();
        assert_eq!(e.message(), msg, "{opts}");
        // A failed create removes the file it made.
        assert!(!img.exists(), "{opts}");
    }
    if let Some((qemu_img, _)) = tools() {
        let (code, out) = run_status(
            &qemu_img,
            &["create", "-f", "qcow2", "-o", "cluster_size=1000", path_str(&img), "1M"],
        );
        assert_ne!(code, 0);
        assert!(out.contains(cases[0].1), "{out}");
    }
}

#[test]
fn check_finds_and_repairs_leaks() {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    let dir = scratch("check");
    let img = dir.join("leak.qcow2");
    run(&qemu_img, &["create", "-f", "qcow2", path_str(&img), "4M"]);
    run(&qemu_io, &["-f", "qcow2", "-c", "write -P 1 0 128k", path_str(&img)]);

    // Drop the L2 entry of the first cluster: its cluster stays referenced but unused.
    let l1_offset = be64(&img, 40);
    let l2_offset = be64(&img, l1_offset as usize) & 0x00ff_ffff_ffff_fe00;
    let mut bytes = fs::read(&img).unwrap();
    bytes[l2_offset as usize..l2_offset as usize + 8].fill(0);
    fs::write(&img, &bytes).unwrap();

    let theirs =
        run_status(&qemu_img, &["check", "-f", "qcow2", "--output=json", path_str(&img)]).1;
    let g = BlockGraph::new();
    add_qcow2(&g, &img, true, "").unwrap();
    let r = g.check("q", 0).unwrap();
    assert_eq!(r.leaks as u64, json_num(&theirs, "leaks").unwrap(), "{theirs}");
    assert_eq!(r.leaks, 1);
    assert_eq!(r.corruptions, 0);
    assert_eq!(r.image_end_offset as u64, json_num(&theirs, "image-end-offset").unwrap());
    assert_eq!(r.bfi.allocated_clusters, json_num(&theirs, "allocated-clusters").unwrap());
    g.blockdev_del("q").unwrap();

    let g = BlockGraph::new();
    add_qcow2(&g, &img, false, "").unwrap();
    let r = g.check("q", BDRV_FIX_LEAKS | BDRV_FIX_ERRORS).unwrap();
    assert_eq!(r.leaks_fixed, 1);
    g.blockdev_del("q").unwrap();
    check_clean(&qemu_img, &img);
}

#[test]
fn snapshots_both_ways() {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    let dir = scratch("snapshots");
    let img = dir.join("snap.qcow2");
    run(&qemu_img, &["create", "-f", "qcow2", path_str(&img), "4M"]);
    run(&qemu_io, &["-f", "qcow2", "-c", "write -P 1 0 64k", path_str(&img)]);
    run(&qemu_img, &["snapshot", "-c", "theirs", path_str(&img)]);

    let g = BlockGraph::new();
    let blk = open_rw(&g, &img);
    let list = g.snapshot_list("q").unwrap().unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].name, "theirs");
    blk.pwrite(0, &[2u8; 4096]).unwrap();
    g.snapshot_create("q", "mine").unwrap();
    blk.pwrite(0, &[3u8; 4096]).unwrap();
    drop(blk);
    g.blockdev_del("q").unwrap();

    check_clean(&qemu_img, &img);
    let out = run(&qemu_img, &["snapshot", "-l", path_str(&img)]);
    assert!(out.contains("theirs") && out.contains("mine"), "{out}");

    // Apply ours in qemu-img, theirs here.
    run(&qemu_img, &["snapshot", "-a", "mine", path_str(&img)]);
    let out = run(&qemu_io, &["-f", "qcow2", "-c", "read -P 2 0 4k", path_str(&img)]);
    assert!(!out.contains("Pattern verification failed"), "{out}");
    let g = BlockGraph::new();
    add_qcow2(&g, &img, false, "").unwrap();
    g.snapshot_goto("q", "theirs").unwrap();
    g.snapshot_delete("q", None, Some("mine")).unwrap();
    g.blockdev_del("q").unwrap();
    let out = run(&qemu_io, &["-f", "qcow2", "-c", "read -P 1 0 64k", path_str(&img)]);
    assert!(!out.contains("Pattern verification failed"), "{out}");
    let out = run(&qemu_img, &["snapshot", "-l", path_str(&img)]);
    assert!(out.contains("theirs") && !out.contains("mine"), "{out}");
    check_clean(&qemu_img, &img);
}

#[test]
fn persistent_bitmaps_both_ways() {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    let dir = scratch("bitmaps");
    let img = dir.join("bm.qcow2");
    run(&qemu_img, &["create", "-f", "qcow2", path_str(&img), "8M"]);
    run(&qemu_img, &["bitmap", "--add", path_str(&img), "b0"]);
    run(&qemu_io, &["-f", "qcow2", "-c", "write 1M 64k", path_str(&img)]);

    // Loaded here, written here, stored on close; plus one made here.
    let g = BlockGraph::new();
    let blk = open_rw(&g, &img);
    g.block_dirty_bitmap_add(&BlockDirtyBitmapAdd {
        node: "q".into(),
        name: "b1".into(),
        granularity: Some(65536),
        persistent: Some(true),
        disabled: None,
    })
    .unwrap();
    blk.pwrite(3 * MIB as u64, &[7u8; 4096]).unwrap();
    drop(blk);
    g.blockdev_del("q").unwrap();

    check_clean(&qemu_img, &img);
    let info = run(&qemu_img, &["info", "--output=json", path_str(&img)]);
    assert!(info.contains("\"name\": \"b0\"") && info.contains("\"name\": \"b1\""), "{info}");
    if let Some(b0) = dirty_ranges(&qemu_img, &img, "b0") {
        assert_eq!(b0, [(MIB as u64, CLUSTER as u64), (3 * MIB as u64, CLUSTER as u64)]);
        let b1 = dirty_ranges(&qemu_img, &img, "b1").unwrap();
        assert_eq!(b1, [(3 * MIB as u64, CLUSTER as u64)]);
    }

    // Removed here, gone for QEMU.
    let g = BlockGraph::new();
    add_qcow2(&g, &img, false, "").unwrap();
    g.block_dirty_bitmap_remove(&BlockDirtyBitmap { node: "q".into(), name: "b0".into() }).unwrap();
    g.blockdev_del("q").unwrap();
    let info = run(&qemu_img, &["info", "--output=json", path_str(&img)]);
    assert!(!info.contains("\"name\": \"b0\"") && info.contains("\"name\": \"b1\""), "{info}");
    check_clean(&qemu_img, &img);

    // Bitmaps need version 3.
    let v2 = dir.join("v2.qcow2");
    run(&qemu_img, &["create", "-f", "qcow2", "-o", "compat=0.10", path_str(&v2), "1M"]);
    let g = BlockGraph::new();
    let pv2 = format!(
        r#"{{"driver": "qcow2", "node-name": "v", "file": {{"driver": "file", "filename": "{}"}}}}"#,
        path_str(&v2)
    );
    g.blockdev_add(from_json::<BlockdevOptions>(&pv2)).unwrap();
    let e = g
        .block_dirty_bitmap_add(&BlockDirtyBitmapAdd {
            node: "v".into(),
            name: "x".into(),
            granularity: None,
            persistent: Some(true),
            disabled: None,
        })
        .unwrap_err();
    assert_eq!(
        e.message(),
        "Can't make bitmap 'x' persistent in 'v': Cannot store dirty bitmaps in qcow2 v2 files"
    );
}

#[test]
fn luks_encryption_both_ways() {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    setup_secrets();
    let dir = scratch("luks");
    let secret = format!("secret,id=sec0,data={PASSWORD}");
    let theirs = dir.join("theirs.qcow2");
    run(
        &qemu_img,
        &[
            "create",
            "--object",
            &secret,
            "-f",
            "qcow2",
            "-o",
            "encrypt.format=luks,encrypt.key-secret=sec0,encrypt.iter-time=10",
            path_str(&theirs),
            "2M",
        ],
    );
    let image_opts =
        |p: &Path| format!("driver=qcow2,encrypt.key-secret=sec0,file.filename={}", path_str(p));
    run(
        &qemu_io,
        &["--object", &secret, "--image-opts", "-c", "write -P 0x21 64k 8k", &image_opts(&theirs)],
    );

    let luks = r#""encrypt": {"format": "luks", "key-secret": "sec0"},"#;
    let g = BlockGraph::new();
    add_qcow2(&g, &theirs, false, luks).unwrap();
    let blk = BlockBackend::new(&g, "q", RW, SHARED).unwrap();
    let mut buf = vec![0u8; 16 * KIB];
    blk.pread(60 * KIB as u64, &mut buf).unwrap();
    assert!(buf[..4 * KIB].iter().all(|b| *b == 0));
    assert!(buf[4 * KIB..12 * KIB].iter().all(|b| *b == 0x21));
    blk.pwrite(MIB as u64, &[0x5au8; 4096]).unwrap();
    drop(blk);
    g.blockdev_del("q").unwrap();
    let out = run(
        &qemu_io,
        &["--object", &secret, "--image-opts", "-c", "read -P 0x5a 1M 4k", &image_opts(&theirs)],
    );
    assert!(!out.contains("Pattern verification failed"), "{out}");
    let info = run(&qemu_img, &["info", path_str(&theirs)]);
    assert!(info.contains("encrypted: yes") && info.contains("format: luks"), "{info}");

    // Made here, read by QEMU.
    let mine = dir.join("mine.qcow2");
    let g = BlockGraph::new();
    let mut o = QDict::new();
    for (k, v) in [
        ("size", "2M"),
        ("encrypt.format", "luks"),
        ("encrypt.key-secret", "sec0"),
        ("encrypt.iter-time", "10"),
    ] {
        o.put(k, v);
    }
    g.create_image("qcow2", path_str(&mine), &mut o).unwrap();
    add_qcow2(&g, &mine, false, luks).unwrap();
    let blk = BlockBackend::new(&g, "q", RW, SHARED).unwrap();
    blk.pwrite(4096, &[0x77u8; 8192]).unwrap();
    drop(blk);
    g.blockdev_del("q").unwrap();
    let out = run(
        &qemu_io,
        &[
            "--object",
            &secret,
            "--image-opts",
            "-c",
            "read -P 0x77 4k 8k",
            "-c",
            "read -P 0 0 4k",
            &image_opts(&mine),
        ],
    );
    assert!(!out.contains("Pattern verification failed"), "{out}");
    let out = run(&qemu_img, &["check", "--object", &secret, "--image-opts", &image_opts(&mine)]);
    assert!(out.contains("No errors were found on the image."), "{out}");

    // Without the secret the image does not open.
    let g = BlockGraph::new();
    let e = add_qcow2(&g, &mine, false, "").unwrap_err();
    assert_eq!(e.message(), "Parameter 'encrypt.key-secret' is required for cipher");
}

#[test]
fn resize_and_amend() {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    let dir = scratch("resize");
    let img = dir.join("r.qcow2");
    run(&qemu_img, &["create", "-f", "qcow2", path_str(&img), "1M"]);
    run(&qemu_io, &["-f", "qcow2", "-c", "write -P 9 0 64k", path_str(&img)]);

    let g = BlockGraph::new();
    let blk = open_rw(&g, &img);
    blk.truncate(100 * MIB as u64).unwrap();
    blk.pwrite(99 * MIB as u64, &[4u8; 4096]).unwrap();
    assert_eq!(blk.getlength().unwrap(), 100 * MIB as u64);
    drop(blk);

    let mut o = QDict::new();
    o.put("compat", "0.10");
    g.amend_options("q", &mut o, false).unwrap();
    g.blockdev_del("q").unwrap();

    check_clean(&qemu_img, &img);
    let info = run(&qemu_img, &["info", "--output=json", path_str(&img)]);
    let i = info.rfind("\"virtual-size\"").unwrap();
    assert_eq!(json_num(&info[i..], "virtual-size"), Some(100 * MIB as u64), "{info}");
    assert!(info.contains("\"compat\": \"0.10\""), "{info}");
    let out = run(
        &qemu_io,
        &["-f", "qcow2", "-c", "read -P 9 0 64k", "-c", "read -P 4 99M 4k", path_str(&img)],
    );
    assert!(!out.contains("Pattern verification failed"), "{out}");

    // Shrinking needs the data gone first, as in QEMU.
    let g = BlockGraph::new();
    let blk = open_rw(&g, &img);
    blk.truncate(2 * MIB as u64).unwrap();
    assert_eq!(blk.getlength().unwrap(), 2 * MIB as u64);
    drop(blk);
    g.blockdev_del("q").unwrap();
    check_clean(&qemu_img, &img);
}

#[test]
fn open_errors() {
    let dir = scratch("open-errors");
    let img = dir.join("bad.qcow2");
    let mut h = vec![0u8; 512];
    h[..4].copy_from_slice(b"QFI\xfb");
    h[7] = 4;
    fs::write(&img, &h).unwrap();
    let g = BlockGraph::new();
    let e = add_qcow2(&g, &img, true, "").unwrap_err();
    assert_eq!(e.message(), "Unsupported qcow2 version 4");

    h[7] = 3;
    h[23] = 8; // cluster_bits
    fs::write(&img, &h).unwrap();
    let e = add_qcow2(&g, &img, true, "").unwrap_err();
    assert_eq!(e.message(), "Unsupported cluster size: 2^8");
}
