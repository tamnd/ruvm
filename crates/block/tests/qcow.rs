// SPDX-License-Identifier: GPL-2.0-or-later

//! The `qcow` format driver end to end: images made by `qemu-img` and `qemu-io` (plain, with a
//! backing file, compressed, AES encrypted) read and written here, and images made here (with
//! `qemu-img create` style options and with `blockdev-create`, including compressed and
//! encrypted writes) checked with `qemu-img compare` against a raw reference. Probing and the
//! `qemu-img info` fields are compared too. The interop tests skip themselves when `qemu-img`
//! or `qemu-io` is not installed.

#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Once;

use ruvm_block::{
    BLK_PERM_CONSISTENT_READ, BLK_PERM_RESIZE, BLK_PERM_WRITE, BlockBackend, BlockGraph,
};
use ruvm_qapi::QDict;
use ruvm_qapi::json;
use ruvm_qapi::types::{BlockdevCreateOptions, BlockdevOptions};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};

const RW: u64 = BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE;
const SHARED: u64 = BLK_PERM_CONSISTENT_READ;
const PASSWORD: &str = "qcowpass";

fn setup() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        ruvm_crypto::secret::secret_object_add_global(&format!("secret,id=sec0,data={PASSWORD}"))
            .unwrap();
    });
}

fn find_tool(name: &str) -> Option<PathBuf> {
    ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"]
        .iter()
        .map(|d| Path::new(d).join(name))
        .find(|p| p.exists())
}

/// `qemu-img` and `qemu-io`, or `None` (and a note) when either is not installed.
fn tools() -> Option<(PathBuf, PathBuf)> {
    match (find_tool("qemu-img"), find_tool("qemu-io")) {
        (Some(i), Some(o)) => Some((i, o)),
        _ => {
            eprintln!("qemu-img or qemu-io not found, skipping");
            None
        }
    }
}

fn scratch(test: &str) -> PathBuf {
    let base = option_env!("CARGO_TARGET_TMPDIR").map_or_else(std::env::temp_dir, PathBuf::from);
    let dir = base.join("ruvm-block-qcow").join(test);
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

/// Runs `tool` with `args`, the `sec0` secret object after the subcommand for `qemu-img`,
/// and returns stdout.
fn run(tool: &Path, args: &[&str]) -> String {
    run_status(tool, args, true)
}

/// [`run`], checking the exit status only when `check` is set.
fn run_status(tool: &Path, args: &[&str], check: bool) -> String {
    let secret = format!("secret,id=sec0,data={PASSWORD}");
    let mut cmd = Command::new(tool);
    if tool.ends_with("qemu-img") {
        cmd.arg(args[0]).arg("--object").arg(&secret).args(&args[1..]);
    } else {
        cmd.arg("--object").arg(&secret).args(args);
    }
    let out = cmd.output().unwrap();
    assert!(
        !check || out.status.success(),
        "{} {args:?} failed: {}{}",
        tool.display(),
        String::from_utf8_lossy(&out.stdout),
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

/// `blockdev-add` of a qcow node `name` over `path`, with extra members `extra` (ending in a
/// comma).
fn add_qcow(g: &BlockGraph, name: &str, path: &Path, extra: &str) -> ruvm_base::Result<()> {
    g.blockdev_add(from_json::<BlockdevOptions>(&format!(
        r#"{{"driver": "qcow", "node-name": "{name}", {extra}
            "file": {{"driver": "file", "filename": "{}"}}}}"#,
        path_str(path)
    )))
}

const AES: &str = r#""encrypt": {"format": "aes", "key-secret": "sec0"},"#;

fn read_all(blk: &BlockBackend) -> Vec<u8> {
    let len = blk.getlength().unwrap() as usize;
    let mut buf = vec![0u8; len];
    blk.pread(0, &mut buf).unwrap();
    buf
}

/// `qemu-img compare` of the image `img` (opened with `--image-opts` from `opts`) and the raw
/// file `raw`.
fn compare(qemu_img: &Path, opts: &str, raw: &Path) {
    let raw = format!("driver=raw,file.filename={}", path_str(raw));
    run(qemu_img, &["compare", "--image-opts", opts, &raw]);
}

fn qcow_opts(path: &Path, encrypted: bool) -> String {
    let enc = if encrypted { "encrypt.format=aes,encrypt.key-secret=sec0," } else { "" };
    format!("driver=qcow,{enc}file.filename={}", path_str(path))
}

/// The number after `"key": ` in `qemu-img info --output=json` output, at the top level
/// (the last occurrence, children come first).
fn info_num(info: &str, key: &str) -> Option<u64> {
    let pat = format!("\"{key}\": ");
    let i = info.rfind(&pat)? + pat.len();
    info[i..].split(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
}

fn info_str(info: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\": \"");
    let i = info.rfind(&pat)? + pat.len();
    info[i..].split('"').next().map(str::to_string)
}

/// Opens `path` here by probing, as `qemu-img` does without `-f`, and checks the format, the
/// virtual size, the cluster size, the encryption flag and the backing file name against
/// `qemu-img info`.
fn probe_and_compare(qemu_img: &Path, path: &Path, encrypted: bool) {
    let g = BlockGraph::new();
    let mut o = QDict::new();
    if encrypted {
        o.put("encrypt.format", "aes");
        o.put("encrypt.key-secret", "sec0");
    }
    let name = g.open_image(Some(path_str(path)), o).unwrap();
    let info = if encrypted {
        run(qemu_img, &["info", "--output=json", "--image-opts", &qcow_opts(path, true)])
    } else {
        run(qemu_img, &["info", "--output=json", path_str(path)])
    };
    let nodes = g.query_named_block_nodes(Some(true)).unwrap();
    let n = nodes.iter().find(|n| n.node_name == name).unwrap();
    assert_eq!(n.drv, "qcow");
    assert_eq!(info_str(&info, "format").as_deref(), Some("qcow"));
    assert_eq!(Some(n.image.virtual_size as u64), info_num(&info, "virtual-size"));
    assert_eq!(n.image.cluster_size.map(|c| c as u64), info_num(&info, "cluster-size"));
    assert_eq!(n.encrypted, encrypted);
    assert_eq!(n.image.encrypted.unwrap_or(false), info.contains("\"encrypted\": true"));
    assert_eq!(n.image.backing_filename, info_str(&info, "backing-filename"));
}

#[test]
fn qemu_img_plain_image() {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    let dir = scratch("plain");
    let img = dir.join("disk.qcow");
    let size = 5 << 20;
    run(&qemu_img, &["create", "-f", "qcow", path_str(&img), "5M"]);
    run(
        &qemu_io,
        &[
            "-f",
            "qcow",
            "-c",
            "write -P 0x55 0 64k",
            "-c",
            "write -P 0xaa 3000k 1500k",
            "-c",
            "write -P 0x11 4608k 512",
            path_str(&img),
        ],
    );
    let mut expect = vec![0u8; size];
    expect[..64 << 10].fill(0x55);
    expect[3000 << 10..4500 << 10].fill(0xaa);
    expect[4608 << 10..(4608 << 10) + 512].fill(0x11);

    probe_and_compare(&qemu_img, &img, false);
    {
        let g = BlockGraph::new();
        add_qcow(&g, "q", &img, "").unwrap();
        let blk = BlockBackend::new(&g, "q", RW, SHARED).unwrap();
        assert_eq!(blk.getlength().unwrap(), size as u64);
        assert!(read_all(&blk) == expect);

        // Allocated clusters map to the file, holes do not.
        let m = blk.map_entry(0, 4096).unwrap();
        assert!(m.data && m.offset.is_some());
        let m = blk.map_entry(1 << 20, 4096).unwrap();
        assert!(!m.data);

        // Unaligned writes, a write over allocated and unallocated clusters.
        blk.pwrite(1000, b"written by ruvm").unwrap();
        let p = pattern(300_000, 9);
        blk.pwrite(2_000_000, &p).unwrap();
        blk.flush().unwrap();
        expect[1000..1015].copy_from_slice(b"written by ruvm");
        expect[2_000_000..2_300_000].copy_from_slice(&p);
        assert!(read_all(&blk) == expect);
    }
    let raw = dir.join("expect.raw");
    fs::write(&raw, &expect).unwrap();
    compare(&qemu_img, &qcow_opts(&img, false), &raw);
}

#[test]
fn qemu_img_backing_file() {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    let dir = scratch("backing");
    let base = dir.join("base.raw");
    let img = dir.join("overlay.qcow");
    let size = 2 << 20;
    let base_data = pattern(size, 1);
    fs::write(&base, &base_data).unwrap();
    run(&qemu_img, &["create", "-f", "qcow", "-b", path_str(&base), "-F", "raw", path_str(&img)]);
    run(&qemu_io, &["-f", "qcow", "-c", "write -P 0x77 4096 1024", path_str(&img)]);
    let mut expect = base_data.clone();
    expect[4096..5120].fill(0x77);

    probe_and_compare(&qemu_img, &img, false);
    {
        let g = BlockGraph::new();
        let name = g.open_image(Some(path_str(&img)), QDict::new()).unwrap();
        let blk = BlockBackend::new(&g, &name, RW, SHARED).unwrap();
        assert_eq!(blk.getlength().unwrap(), size as u64);
        assert!(read_all(&blk) == expect);
        // Only the written sectors are in the overlay: with a backing file the clusters are
        // 512 bytes, and block status answers one cluster at a time.
        for at in [4096, 4608] {
            let m = blk.map_entry(at, 1024).unwrap();
            assert_eq!((m.depth, m.length), (0, 512));
        }
        let m = blk.map_entry(5120, 4096).unwrap();
        assert_eq!(m.depth, 1);
        let m = blk.map_entry(0, 4096).unwrap();
        assert_eq!(m.depth, 1);

        blk.pwrite(100_000, &pattern(1000, 50)).unwrap();
        expect[100_000..101_000].copy_from_slice(&pattern(1000, 50));
        assert!(read_all(&blk) == expect);

        // make_empty drops what the overlay holds; the backing file shows through again.
        blk.make_empty().unwrap();
        assert!(read_all(&blk) == base_data);
        blk.pwrite(0, &[9u8; 512]).unwrap();
    }
    expect = base_data;
    expect[..512].fill(9);
    let raw = dir.join("expect.raw");
    fs::write(&raw, &expect).unwrap();
    run(&qemu_img, &["compare", "-f", "qcow", "-F", "raw", path_str(&img), path_str(&raw)]);
}

#[test]
fn qemu_img_compressed() {
    let Some((qemu_img, _)) = tools() else { return };
    let dir = scratch("compressed");
    let raw = dir.join("data.raw");
    let img = dir.join("disk.qcow");
    // Compressible and incompressible clusters, a hole, and a size that is not a whole
    // number of clusters.
    let size = (1 << 20) + 1536;
    let mut data = vec![0u8; size];
    data[..256 << 10].copy_from_slice(&pattern(256 << 10, 3));
    let mut x = 0x9e37_79b9u32;
    for b in &mut data[512 << 10..516 << 10] {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *b = x as u8;
    }
    data[size - 1000..].fill(0x42);
    fs::write(&raw, &data).unwrap();
    // qemu-img 11.1 exits with status 1 here, without a message: after copying it signals
    // the end with a zero-length compressed write at offset 0, which qcow_co_pwritev_compressed()
    // refuses with EINVAL. Everything before that has been written, so the image is complete.
    run_status(
        &qemu_img,
        &["convert", "-c", "-f", "raw", "-O", "qcow", path_str(&raw), path_str(&img)],
        false,
    );

    probe_and_compare(&qemu_img, &img, false);
    let mut expect = data.clone();
    {
        let g = BlockGraph::new();
        add_qcow(&g, "q", &img, "").unwrap();
        let blk = BlockBackend::new(&g, "q", RW, SHARED).unwrap();
        assert_eq!(blk.getlength().unwrap(), size as u64);
        assert!(read_all(&blk) == data);
        let m = blk.map_entry(0, 4096).unwrap();
        assert!(m.data && m.compressed);

        // A partial write to a compressed cluster decompresses it into a new cluster.
        blk.pwrite(8192 + 512, &[0xeeu8; 1024]).unwrap();
        expect[8192 + 512..8192 + 1536].fill(0xee);
        assert!(read_all(&blk) == expect);
        let m = blk.map_entry(8192, 4096).unwrap();
        assert!(m.data && !m.compressed && m.offset.is_some());
    }
    fs::write(&raw, &expect).unwrap();
    compare(&qemu_img, &qcow_opts(&img, false), &raw);
}

#[test]
fn qemu_img_aes() {
    setup();
    let Some((qemu_img, _)) = tools() else { return };
    let dir = scratch("aes");
    let raw = dir.join("data.raw");
    let img = dir.join("disk.qcow");
    let size = 1 << 20;
    let mut data = pattern(size, 5);
    data[300_000..400_000].fill(0);
    fs::write(&raw, &data).unwrap();
    run(
        &qemu_img,
        &[
            "convert",
            "-f",
            "raw",
            "-O",
            "qcow",
            "-o",
            "encrypt.format=aes,encrypt.key-secret=sec0",
            path_str(&raw),
            path_str(&img),
        ],
    );

    probe_and_compare(&qemu_img, &img, true);
    let mut expect = data.clone();
    {
        let g = BlockGraph::new();
        add_qcow(&g, "q", &img, AES).unwrap();
        let blk = BlockBackend::new(&g, "q", RW, SHARED).unwrap();
        assert!(read_all(&blk) == data);
        // Encrypted clusters have no offset to show.
        let m = blk.map_entry(0, 4096).unwrap();
        assert!(m.data && m.offset.is_none());
        // Partial writes, one to a new cluster whose rest must read as zeroes.
        blk.pwrite(777, b"secret data").unwrap();
        blk.pwrite(350_000, &[0x31u8; 700]).unwrap();
        expect[777..788].copy_from_slice(b"secret data");
        expect[350_000..350_700].fill(0x31);
        assert!(read_all(&blk) == expect);
    }
    fs::write(&raw, &expect).unwrap();
    compare(&qemu_img, &qcow_opts(&img, true), &raw);

    // Without the key, or with encryption options on a plain image.
    let g = BlockGraph::new();
    let e = add_qcow(&g, "q", &img, r#""encrypt": {"format": "aes"},"#).unwrap_err();
    assert_eq!(e.message(), "Parameter 'encrypt.key-secret' is required for cipher");
    let plain = dir.join("plain.qcow");
    run(&qemu_img, &["create", "-f", "qcow", path_str(&plain), "1M"]);
    let e = add_qcow(&g, "p", &plain, AES).unwrap_err();
    assert_eq!(e.message(), "No encryption in image header, but options specified format 'aes'");
}

/// Writes the test content through `blk`, some of it compressed, and returns what the whole
/// image must read as.
fn write_content(blk: &BlockBackend, size: usize, cluster: usize) -> Vec<u8> {
    let mut expect = vec![0u8; size];
    let p = pattern(100_000, 11);
    blk.pwrite(12_345, &p).unwrap();
    expect[12_345..112_345].copy_from_slice(&p);
    // Compressed clusters, one compressible and one not.
    let c = pattern(cluster, 1);
    let at = 64 * cluster;
    blk.pwrite_compressed(at as u64, &c).unwrap();
    expect[at..at + cluster].copy_from_slice(&c);
    let mut x = 0x1234_5678u32;
    let noise: Vec<u8> = (0..cluster)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x as u8
        })
        .collect();
    let at = 66 * cluster;
    blk.pwrite_compressed(at as u64, &noise).unwrap();
    expect[at..at + cluster].copy_from_slice(&noise);
    blk.pwrite(size as u64 - 512, &[0xfe; 512]).unwrap();
    expect[size - 512..].fill(0xfe);
    blk.flush().unwrap();
    assert!(read_all(blk) == expect);
    expect
}

#[test]
fn created_here_with_create_opts() {
    setup();
    let dir = scratch("create-opts");
    let tools = tools();
    for encrypted in [false, true] {
        let img = dir.join(if encrypted { "aes.qcow" } else { "plain.qcow" });
        let size = 3 << 20;
        let g = BlockGraph::new();
        let mut o = QDict::new();
        o.put("size", "3M");
        if encrypted {
            // The deprecated spelling.
            o.put("encryption", "on");
            o.put("encrypt.key-secret", "sec0");
        }
        g.create_image("qcow", path_str(&img), &mut o).unwrap();
        assert!(o.is_empty());
        // Header, then one sector of L1 table.
        assert_eq!(fs::metadata(&img).unwrap().len(), 48 + 512);
        let expect = {
            let g = BlockGraph::new();
            add_qcow(&g, "q", &img, if encrypted { AES } else { "" }).unwrap();
            let blk = BlockBackend::new(&g, "q", RW, SHARED).unwrap();
            assert_eq!(blk.getlength().unwrap(), size as u64);
            write_content(&blk, size, 4096)
        };
        let Some((qemu_img, _)) = &tools else { continue };
        let raw = dir.join("expect.raw");
        fs::write(&raw, &expect).unwrap();
        compare(qemu_img, &qcow_opts(&img, encrypted), &raw);
        probe_and_compare(qemu_img, &img, encrypted);
    }
}

#[test]
fn created_here_with_blockdev_create() {
    setup();
    let dir = scratch("blockdev-create");
    let base = dir.join("base.raw");
    let img = dir.join("overlay.qcow");
    let size: usize = 1 << 20;
    let base_data = pattern(size, 77);
    fs::write(&base, &base_data).unwrap();
    {
        let g = BlockGraph::new();
        g.blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
            r#"{{"driver": "file", "filename": "{}", "size": 0}}"#,
            path_str(&img)
        )))
        .unwrap();
        g.blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
            r#"{{"driver": "qcow", "file": {{"driver": "file", "filename": "{}"}},
                "size": {size}, "backing-file": "{}"}}"#,
            path_str(&img),
            path_str(&base)
        )))
        .unwrap();
    }
    let mut expect = base_data.clone();
    {
        let g = BlockGraph::new();
        let name = g.open_image(Some(path_str(&img)), QDict::new()).unwrap();
        let blk = BlockBackend::new(&g, &name, RW | BLK_PERM_RESIZE, SHARED).unwrap();
        assert!(read_all(&blk) == base_data);
        blk.pwrite(513, b"over the base").unwrap();
        expect[513..526].copy_from_slice(b"over the base");
        assert!(read_all(&blk) == expect);
        // qcow cannot be resized.
        let e = blk.truncate(2 << 20).unwrap_err();
        assert_eq!(e.message(), "Image format driver does not support resize");
    }
    let Some((qemu_img, _)) = tools() else { return };
    probe_and_compare(&qemu_img, &img, false);
    let raw = dir.join("expect.raw");
    fs::write(&raw, &expect).unwrap();
    run(&qemu_img, &["compare", "-f", "qcow", "-F", "raw", path_str(&img), path_str(&raw)]);
}

#[test]
fn create_errors() {
    setup();
    let dir = scratch("create-errors");
    let img = dir.join("x.qcow");
    let g = BlockGraph::new();
    let mut o = QDict::new();
    o.put("size", "0");
    let e = g.create_image("qcow", path_str(&img), &mut o).unwrap_err();
    assert_eq!(e.message(), "Image size is too small, cannot be zero length");

    let mut o = QDict::new();
    o.put("size", "1M");
    o.put("encryption", "on");
    o.put("encrypt.format", "aes");
    let e = g.create_image("qcow", path_str(&img), &mut o).unwrap_err();
    assert_eq!(
        e.message(),
        "'encrypt.format' and its alias 'encryption' can't be used at the same time"
    );

    let mut o = QDict::new();
    o.put("size", "1M");
    o.put("encrypt.format", "aes");
    let e = g.create_image("qcow", path_str(&img), &mut o).unwrap_err();
    assert_eq!(e.message(), "Parameter 'encrypt.key-secret' is required for cipher");

    let mut o = QDict::new();
    o.put("size", "1M");
    o.put("backing_fmt", "bogus");
    let e = g.create_image("qcow", path_str(&img), &mut o).unwrap_err();
    assert_eq!(e.message(), "unrecognized backing format 'bogus'");

    // The size is rounded up to whole sectors.
    let mut o = QDict::new();
    o.put("size", "1000");
    g.create_image("qcow", path_str(&img), &mut o).unwrap();
    let h = fs::read(&img).unwrap();
    assert_eq!(u64::from_be_bytes(h[24..32].try_into().unwrap()), 1024);
}

/// A header with the fields that matter for the open checks.
fn write_header(path: &Path, version: u32, size: u64, cluster_bits: u8, l2_bits: u8, crypt: u32) {
    let mut h = vec![0u8; 48 + 512];
    h[..4].copy_from_slice(b"QFI\xfb");
    h[4..8].copy_from_slice(&version.to_be_bytes());
    h[24..32].copy_from_slice(&size.to_be_bytes());
    h[32] = cluster_bits;
    h[33] = l2_bits;
    h[36..40].copy_from_slice(&crypt.to_be_bytes());
    h[40..48].copy_from_slice(&48u64.to_be_bytes());
    fs::write(path, h).unwrap();
}

#[test]
fn open_errors() {
    let dir = scratch("open-errors");
    let img = dir.join("bad.qcow");
    let err = |version, size, cb, l2, crypt| {
        write_header(&img, version, size, cb, l2, crypt);
        let g = BlockGraph::new();
        add_qcow(&g, "q", &img, "").unwrap_err()
    };
    let e = err(2, 1 << 20, 12, 9, 0);
    assert_eq!(e.message(), "qcow (v1) does not support qcow version 2");
    assert_eq!(e.hint_text(), Some("Try the 'qcow2' driver instead.\n"));
    let e = err(7, 1 << 20, 12, 9, 0);
    assert_eq!(e.message(), "qcow (v1) does not support qcow version 7");
    assert_eq!(e.hint_text(), None);
    let e = err(1, 1, 12, 9, 0);
    assert_eq!(e.message(), "Image size is too small (must be at least 2 bytes)");
    let e = err(1, 1 << 20, 17, 9, 0);
    assert_eq!(e.message(), "Cluster size must be between 512 and 64k");
    let e = err(1, 1 << 20, 12, 5, 0);
    assert_eq!(e.message(), "L2 table size must be between 512 and 64k");
    let e = err(1, 1 << 20, 12, 9, 2);
    assert_eq!(e.message(), "invalid encryption method in qcow header");
    let e = err(1, u64::MAX - 10, 12, 9, 0);
    assert_eq!(e.message(), "Image too large");
    let e = err(1, 1 << 62, 9, 6, 0);
    assert_eq!(e.message(), "Image too large");

    fs::write(&img, b"not a qcow image at all, just some bytes that are long enough...").unwrap();
    let g = BlockGraph::new();
    let e = add_qcow(&g, "q", &img, "").unwrap_err();
    assert_eq!(e.message(), "Image not in qcow format");
}
