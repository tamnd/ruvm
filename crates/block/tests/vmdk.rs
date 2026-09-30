// SPDX-License-Identifier: GPL-2.0-or-later

//! The `vmdk` format driver against QEMU. Images of every subformat that `qemu-img` makes and
//! `qemu-io` writes read the same here; images made here have the same files, byte for byte
//! but for the random CID, as `qemu-img create` makes, and after writes here they pass
//! `qemu-img check` and compare equal to a raw copy. Backing chains, zeroed grains,
//! stream-optimized images, split extents, `blockdev-create` with extent lists, and hand made
//! seSparse, VMFS and VMFS sparse images are covered too, as are the error messages. The
//! interop tests skip themselves when `qemu-img` or `qemu-io` is not installed.

#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use ruvm_block::{BLK_PERM_CONSISTENT_READ, BLK_PERM_WRITE, BlockBackend, BlockGraph};
use ruvm_qapi::types::{BlockdevCreateOptions, ImageInfoSpecificU, VmdkExtentInfo};
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
    let dir = base.join("ruvm-block-vmdk").join(test);
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

/// Runs `prog` with `args`, which must fail, and returns stdout and stderr.
fn run_fail(prog: &Path, args: &[&str]) -> String {
    let out = Command::new(prog).args(args).output().unwrap();
    assert!(!out.status.success(), "{} {args:?} succeeded", prog.display());
    format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr))
}

/// The number after `"key": ` in `qemu-img info --output=json` output.
fn info_num(info: &str, key: &str) -> Option<u64> {
    let pat = format!("\"{key}\": ");
    // The top level entry is the last one, after those of the children.
    let i = info.rfind(&pat)? + pat.len();
    let digits: String = info[i..].chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// The string after `"key": ` in `qemu-img info --output=json` output.
fn info_str(info: &str, key: &str) -> String {
    let pat = format!("\"{key}\": \"");
    let i = info.rfind(&pat).unwrap() + pat.len();
    info[i..].split('"').next().unwrap().to_string()
}

fn from_json<T: Visit + Default>(s: &str) -> T {
    let mut v = QObjectInputVisitor::new(json::from_str(s).unwrap());
    let mut o = T::default();
    T::visit(&mut v, None, &mut o).unwrap();
    o
}

/// Opens `path` with its format probed, as `qemu-img` does without `-f`.
fn open(g: &BlockGraph, path: &Path, read_only: bool) -> String {
    let mut o = QDict::new();
    o.put("read-only", if read_only { "on" } else { "off" });
    g.open_image(Some(path_str(path)), o).unwrap()
}

fn read_all(blk: &BlockBackend) -> Vec<u8> {
    let len = blk.getlength().unwrap() as usize;
    let mut buf = vec![0u8; len];
    blk.pread(0, &mut buf).unwrap();
    buf
}

/// `qemu-img convert` to raw, read back.
fn qemu_raw(qemu_img: &Path, img: &Path, dir: &Path) -> Vec<u8> {
    let raw = dir.join("qemu.raw");
    run(qemu_img, &["convert", "-f", "vmdk", "-O", "raw", path_str(img), path_str(&raw)]);
    fs::read(&raw).unwrap()
}

/// `qemu-img check` and `qemu-img compare` of `img` against `expect`.
fn qemu_verify(qemu_img: &Path, img: &Path, dir: &Path, expect: &[u8]) {
    run(qemu_img, &["check", "-f", "vmdk", path_str(img)]);
    let raw = dir.join("expect.raw");
    fs::write(&raw, expect).unwrap();
    run(qemu_img, &["compare", "-f", "vmdk", "-F", "raw", path_str(img), path_str(&raw)]);
}

/// The extents of the vmdk node `name`.
fn extents(g: &BlockGraph, name: &str) -> (String, Vec<VmdkExtentInfo>) {
    let Some(info) = g.specific_info(name).unwrap() else { panic!("no specific info") };
    let ImageInfoSpecificU::Vmdk(v) = info.u else { panic!("not vmdk") };
    (v.data.create_type, v.data.extents)
}

/// Checks the format, the virtual size, the create type and the extents against
/// `qemu-img info`.
fn compare_info(qemu_img: &Path, g: &BlockGraph, name: &str, img: &Path) {
    let info = run(qemu_img, &["info", "--output=json", path_str(img)]);
    // The keys of the image itself are indented by four spaces, those of the children more.
    let top = |key: &str| {
        let pat = format!("\n    \"{key}\": ");
        let i = info.find(&pat).unwrap() + pat.len();
        info[i..].split([',', '\n']).next().unwrap().trim_matches('"').to_string()
    };
    let node = g.node(name).unwrap();
    assert_eq!(node.driver, top("format"));
    assert_eq!(node.size.to_string(), top("virtual-size"));
    let (create_type, ext) = extents(g, name);
    assert_eq!(create_type, info_str(&info, "create-type"));
    let spec = &info[info.find("\"create-type\"").unwrap()..];
    assert_eq!(ext.len(), spec.matches("\"format\": \"").count(), "{spec}");
    for e in &ext {
        assert!(spec.contains(&format!("\"filename\": \"{}\"", e.filename)), "{}", e.filename);
        assert!(spec.contains(&format!("\"format\": \"{}\"", e.format)), "{spec}");
        assert!(spec.contains(&format!("\"virtual-size\": {}", e.virtual_size)), "{spec}");
        if let Some(c) = e.cluster_size {
            assert!(spec.contains(&format!("\"cluster-size\": {c}")), "{spec}");
        }
        if e.compressed == Some(true) {
            assert!(spec.contains("\"compressed\": true"));
        }
    }
    assert_eq!(spec.contains("\"cluster-size\""), ext.iter().any(|e| e.cluster_size.is_some()));
    assert_eq!(spec.contains("\"compressed\""), ext.iter().any(|e| e.compressed.is_some()));
}

/// `k=v,k=v` into a `QDict`.
fn opts(s: &str) -> QDict {
    let mut o = QDict::new();
    for kv in s.split(',').filter(|kv| !kv.is_empty()) {
        let (k, v) = kv.split_once('=').unwrap();
        o.put(k, v);
    }
    o
}

/// `qemu-img create` and `qemu-io` writes, then it all reads the same here, `qemu-img info`
/// agrees, and writes here read back in `qemu-img`.
#[test]
fn qemu_made_images() {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    for (sub, o) in [
        ("monolithicSparse", ""),
        ("monolithicSparse", "zeroed_grain=on"),
        ("monolithicFlat", ""),
        ("twoGbMaxExtentSparse", ""),
        ("twoGbMaxExtentFlat", ""),
        ("streamOptimized", ""),
    ] {
        let dir = scratch(&format!("qemu-{sub}-{o}"));
        let img = dir.join("disk.vmdk");
        let create_opts = format!("subformat={sub}{}{o}", if o.is_empty() { "" } else { "," });
        if sub == "streamOptimized" {
            // Only whole grains can be written, so convert.
            let src = dir.join("src.raw");
            let mut data = vec![0u8; 10 << 20];
            data[..65536].copy_from_slice(&pattern(65536, 7));
            data[5 << 20..(5 << 20) + 1000].copy_from_slice(&pattern(1000, 9));
            fs::write(&src, &data).unwrap();
            run(
                &qemu_img,
                &["convert", "-O", "vmdk", "-o", &create_opts, path_str(&src), path_str(&img)],
            );
        } else {
            run(&qemu_img, &["create", "-f", "vmdk", "-o", &create_opts, path_str(&img), "10M"]);
            run(
                &qemu_io,
                &[
                    "-f",
                    "vmdk",
                    "-c",
                    "write -P 0x55 0 64k",
                    "-c",
                    "write -P 0xaa 3000k 1500k",
                    "-c",
                    "write -P 0x11 5M 512",
                    "-c",
                    "write -z 8M 128k",
                    path_str(&img),
                ],
            );
        }
        let expect = qemu_raw(&qemu_img, &img, &dir);

        let g = BlockGraph::new();
        let name = open(&g, &img, true);
        compare_info(&qemu_img, &g, &name, &img);
        let blk = BlockBackend::new(&g, &name, BLK_PERM_CONSISTENT_READ, SHARED).unwrap();
        assert!(read_all(&blk) == expect, "{sub} {o}");
        g.check(&name, 0).unwrap();
        drop(blk);
        drop(g);

        // Writes here.
        let g = BlockGraph::new();
        let name = open(&g, &img, false);
        let blk = BlockBackend::new(&g, &name, RW, SHARED).unwrap();
        let mut expect = expect;
        let writes: &[(usize, usize, u8)] = if sub == "streamOptimized" {
            // Whole unallocated grains only.
            &[(1 << 20, 65536, 1), (7 << 20, 65536, 2)]
        } else {
            &[(1000, 3000, 1), (7 << 20, 70_000, 2), ((10 << 20) - 512, 512, 3)]
        };
        for &(off, len, seed) in writes {
            let data = pattern(len, seed);
            blk.pwrite(off as u64, &data).unwrap();
            expect[off..off + len].copy_from_slice(&data);
        }
        assert!(read_all(&blk) == expect);
        if sub == "streamOptimized" {
            // Allocated grains cannot be written again.
            assert!(blk.pwrite(0, &[1u8; 65536]).is_err());
        }
        drop(blk);
        drop(g);
        qemu_verify(&qemu_img, &img, &dir, &expect);
    }
}

/// The bytes of a file made by `vmdk` create with the value of the random `CID` taken out,
/// and the descriptor of a sparse extent as text.
fn normalize(bytes: &[u8]) -> (Vec<u8>, String) {
    let strip_cid = |text: &str| {
        text.lines()
            .map(|l| if l.starts_with("CID=") { "CID=<random>" } else { l })
            .collect::<Vec<_>>()
            .join("\n")
    };
    if bytes.starts_with(b"KDMV") {
        let desc = &bytes[512..512 + 20 * 512];
        let end = desc.iter().position(|&b| b == 0).unwrap_or(desc.len());
        let text = strip_cid(std::str::from_utf8(&desc[..end]).unwrap());
        let mut rest = bytes[..512].to_vec();
        rest.extend_from_slice(&bytes[512 + 20 * 512..]);
        (rest, text)
    } else {
        (Vec::new(), strip_cid(std::str::from_utf8(bytes).unwrap()))
    }
}

/// Every file in `dir`, sorted by name.
fn files(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut v: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .map(|e| {
            let e = e.unwrap();
            (e.file_name().into_string().unwrap(), fs::read(e.path()).unwrap())
        })
        .collect();
    v.sort();
    v
}

/// Images made here with every option are the files `qemu-img create` makes, pass
/// `qemu-img check`, and hold what is written here.
#[test]
fn created_with_options() {
    let Some((qemu_img, _)) = tools() else { return };
    for o in [
        "",
        "adapter_type=lsilogic,hwversion=7,toolsversion=12345",
        "compat6=on",
        "zeroed_grain=on",
        "adapter_type=buslogic,subformat=monolithicFlat",
        "subformat=twoGbMaxExtentSparse,adapter_type=legacyESX",
        "subformat=twoGbMaxExtentFlat",
        "subformat=streamOptimized",
        "subformat=monolithicSparse,hwversion=undefined",
    ] {
        let dir = scratch(&format!("create-{o}"));
        let (mine, theirs) = (dir.join("mine"), dir.join("theirs"));
        fs::create_dir_all(&mine).unwrap();
        fs::create_dir_all(&theirs).unwrap();
        let img = mine.join("disk.vmdk");
        let g = BlockGraph::new();
        let mut qd = opts(o);
        qd.put("size", "10M");
        g.create_image("vmdk", path_str(&img), &mut qd).unwrap();
        let mut args = vec!["create", "-f", "vmdk"];
        if !o.is_empty() {
            args.extend(["-o", o]);
        }
        let theirs_img = theirs.join("disk.vmdk");
        args.extend([path_str(&theirs_img), "10M"]);
        run(&qemu_img, &args);
        let (a, b) = (files(&mine), files(&theirs));
        assert_eq!(a.len(), b.len(), "{o}");
        for ((na, da), (nb, db)) in a.iter().zip(&b) {
            assert_eq!(na, nb);
            // The CID is printed in hex without padding, so the length of a descriptor file
            // varies with it.
            if !da.starts_with(b"# Disk") {
                assert_eq!(da.len(), db.len(), "{o}: {na}");
            }
            assert!(normalize(da) == normalize(db), "{o}: {na} differs");
        }

        let name = open(&g, &img, false);
        compare_info(&qemu_img, &g, &name, &img);
        let blk = BlockBackend::new(&g, &name, RW, SHARED).unwrap();
        let mut expect = vec![0u8; 10 << 20];
        if o.contains("streamOptimized") {
            for (off, seed) in [(0usize, 1u8), (3 << 20, 2), ((10 << 20) - 65536, 3)] {
                let data = pattern(65536, seed);
                blk.pwrite_compressed(off as u64, &data).unwrap();
                expect[off..off + 65536].copy_from_slice(&data);
            }
            blk.pwrite_compressed(0, &[]).unwrap();
        } else {
            for (off, n, seed) in [(0usize, 4096usize, 1u8), (1 << 20, 3 << 20, 2), (777, 10, 4)] {
                let data = pattern(n, seed);
                blk.pwrite(off as u64, &data).unwrap();
                expect[off..off + n].copy_from_slice(&data);
            }
        }
        assert!(read_all(&blk) == expect);
        drop(blk);
        drop(g);
        qemu_verify(&qemu_img, &img, &dir, &expect);
    }
}

/// Images bigger than 2 GiB in split subformats: the extents, and reads and writes across
/// the boundary between them, agree with QEMU.
#[test]
fn split_extents() {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    for sub in ["twoGbMaxExtentSparse", "twoGbMaxExtentFlat"] {
        let dir = scratch(&format!("split-{sub}"));
        let img = dir.join("big.vmdk");
        let g = BlockGraph::new();
        let mut o = opts(&format!("subformat={sub}"));
        o.put("size", "5G");
        g.create_image("vmdk", path_str(&img), &mut o).unwrap();
        let desc = fs::read_to_string(&img).unwrap();
        let s = if sub.ends_with("Sparse") { 's' } else { 'f' };
        let kind = if s == 's' { "SPARSE" } else { "FLAT" };
        let tail = if s == 's' { "" } else { " 0" };
        for (i, sectors) in [(1, 4_194_304), (2, 4_194_304), (3, 2_097_152)] {
            let line = format!("RW {sectors} {kind} \"big-{s}{i:03}.vmdk\"{tail}\n");
            assert!(desc.contains(&line), "{desc}");
        }

        let name = open(&g, &img, false);
        compare_info(&qemu_img, &g, &name, &img);
        let blk = BlockBackend::new(&g, &name, RW, SHARED).unwrap();
        let data = pattern(200_000, 5);
        let at = (2u64 << 30) - 100_000;
        blk.pwrite(at, &data).unwrap();
        drop(blk);
        drop(g);
        run(
            &qemu_io,
            &[
                "-f",
                "vmdk",
                "-c",
                "write -P 0x33 4G 64k",
                "-c",
                "read -P 0 4294901760 65536",
                path_str(&img),
            ],
        );
        run(&qemu_img, &["check", "-f", "vmdk", path_str(&img)]);

        let g = BlockGraph::new();
        let name = open(&g, &img, true);
        let blk = BlockBackend::new(&g, &name, BLK_PERM_CONSISTENT_READ, SHARED).unwrap();
        let mut buf = vec![0u8; 200_000];
        blk.pread(at, &mut buf).unwrap();
        assert!(buf == data);
        let mut buf = vec![0u8; 65536 * 2];
        blk.pread((4 << 30) - 65536, &mut buf).unwrap();
        assert!(buf[..65536].iter().all(|&b| b == 0));
        assert!(buf[65536..].iter().all(|&b| b == 0x33));
        // What QEMU reads across the boundary.
        let out = dir.join("span.raw");
        let start = (2u64 << 30) - 102_400;
        let src = format!(
            "driver=raw,offset={start},size=307200,file.driver=vmdk,file.file.filename={}",
            path_str(&img)
        );
        run(&qemu_img, &["convert", "--image-opts", &src, "-O", "raw", path_str(&out)]);
        let theirs = fs::read(&out).unwrap();
        let mut mine = vec![0u8; 307_200];
        blk.pread(start, &mut mine).unwrap();
        assert!(mine == theirs);
    }
}

/// Overlays with `parentFileNameHint`: made here or by QEMU, read the parent's data where
/// they have none, copy it on partial grain writes, and check the parent's CID.
#[test]
fn backing_chain() {
    let Some((qemu_img, qemu_io)) = tools() else { return };
    let dir = scratch("backing");
    let base = dir.join("base.vmdk");
    run(&qemu_img, &["create", "-f", "vmdk", path_str(&base), "4M"]);
    run(&qemu_io, &["-f", "vmdk", "-c", "write -P 0x42 0 4M", path_str(&base)]);
    let base_data = qemu_raw(&qemu_img, &base, &dir);

    // Made here, with a relative backing file name.
    let top = dir.join("top.vmdk");
    let g = BlockGraph::new();
    let mut o = opts("backing_file=base.vmdk,backing_fmt=vmdk");
    o.put("size", "4M");
    g.create_image("vmdk", path_str(&top), &mut o).unwrap();
    let info = run(&qemu_img, &["info", "--output=json", path_str(&top)]);
    assert_eq!(info_str(&info, "backing-filename"), "base.vmdk");
    let base_info = run(&qemu_img, &["info", "--output=json", path_str(&base)]);
    assert_eq!(info_num(&info, "parent-cid"), info_num(&base_info, "cid"));

    let name = open(&g, &top, false);
    let blk = BlockBackend::new(&g, &name, RW, SHARED).unwrap();
    assert!(read_all(&blk) == base_data);
    let mut expect = base_data.clone();
    for (off, n, seed) in [(1000usize, 3000usize, 1u8), ((2 << 20) + 65536, 65536, 2)] {
        let data = pattern(n, seed);
        blk.pwrite(off as u64, &data).unwrap();
        expect[off..off + n].copy_from_slice(&data);
    }
    assert!(read_all(&blk) == expect);
    drop(blk);
    drop(g);
    qemu_verify(&qemu_img, &top, &dir, &expect);
    // Only the grains written here are in the overlay.
    let map = run(&qemu_img, &["map", "--output=json", "-f", "vmdk", path_str(&top)]);
    assert!(map.contains("\"depth\": 1"));

    // Made by QEMU, read here.
    let top2 = dir.join("top2.vmdk");
    run(
        &qemu_img,
        &["create", "-f", "vmdk", "-b", "base.vmdk", "-F", "vmdk", path_str(&top2), "4M"],
    );
    run(&qemu_io, &["-f", "vmdk", "-c", "write -P 0x17 64k 100k", path_str(&top2)]);
    let expect2 = qemu_raw(&qemu_img, &top2, &dir);
    let g = BlockGraph::new();
    let name = open(&g, &top2, true);
    let blk = BlockBackend::new(&g, &name, BLK_PERM_CONSISTENT_READ, SHARED).unwrap();
    assert!(read_all(&blk) == expect2);
    drop(blk);
    drop(g);

    // Once the parent changes, its CID does not match any more.
    run(&qemu_io, &["-f", "vmdk", "-c", "write -P 0x43 0 512", path_str(&base)]);
    let g = BlockGraph::new();
    let name = open(&g, &top2, true);
    let blk = BlockBackend::new(&g, &name, BLK_PERM_CONSISTENT_READ, SHARED).unwrap();
    let mut buf = [0u8; 512];
    assert_eq!(blk.pread(0, &mut buf).unwrap_err().raw_os_error(), Some(libc::EINVAL));
    // Data of the overlay itself still reads.
    blk.pread(64 << 10, &mut buf).unwrap();
    assert_eq!(buf, [0x17; 512]);
}

/// Zero writes of whole grains become zeroed grain table entries with `zeroed_grain`, and
/// QEMU sees zeroes there.
#[test]
fn zeroed_grains() {
    let Some((qemu_img, _)) = tools() else { return };
    let dir = scratch("zeroed");
    let img = dir.join("z.vmdk");
    let g = BlockGraph::new();
    let mut o = opts("zeroed_grain=on");
    o.put("size", "2M");
    g.create_image("vmdk", path_str(&img), &mut o).unwrap();
    let name = open(&g, &img, false);
    let blk = BlockBackend::new(&g, &name, RW, SHARED).unwrap();
    blk.pwrite(0, &pattern(256 << 10, 1)).unwrap();
    blk.pwrite_zeroes(64 << 10, 128 << 10, false).unwrap();
    // Not whole grains: zeroes get written as data.
    blk.pwrite_zeroes(1000, 1000, false).unwrap();
    let mut expect = vec![0u8; 2 << 20];
    expect[..256 << 10].copy_from_slice(&pattern(256 << 10, 1));
    expect[64 << 10..192 << 10].fill(0);
    expect[1000..2000].fill(0);
    assert!(read_all(&blk) == expect);
    let st = g.block_status(&name, 64 << 10, 128 << 10).unwrap();
    assert_eq!(st.ret & 0x03, 0x02);
    assert_eq!(st.pnum, 64 << 10);
    let st = g.block_status(&name, 0, 128 << 10).unwrap();
    assert_eq!(st.ret & 0x07, 0x05);
    drop(blk);
    drop(g);
    qemu_verify(&qemu_img, &img, &dir, &expect);
    let map = run(&qemu_img, &["map", "--output=json", "-f", "vmdk", path_str(&img)]);
    assert!(map.contains(
        "\"start\": 65536, \"length\": 131072, \"depth\": 0, \"present\": true, \"zero\": true, \
         \"data\": false"
    ));
}

fn put64(b: &mut [u8], off: usize, v: u64) {
    b[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

fn put32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

/// A hand made seSparse extent of 1 MiB with 4 KiB grains: grain 0 and 5 allocated, grain 2
/// zeroed.
#[test]
fn sesparse_reads() {
    let Some((qemu_img, _)) = tools() else { return };
    let dir = scratch("sesparse");
    let mut f = vec![0u8; 67 * 512];
    put64(&mut f, 0, 0xcafe_babe);
    put64(&mut f, 8, 0x0000_0002_0000_0001);
    put64(&mut f, 16, 2048);
    put64(&mut f, 24, 8);
    put64(&mut f, 32, 64);
    put64(&mut f, 80, 1);
    put64(&mut f, 128, 2);
    put64(&mut f, 136, 1);
    put64(&mut f, 144, 3);
    put64(&mut f, 192, 67);
    put64(&mut f, 512, 0xcafe_cafe);
    // The grain directory, then the grain table.
    put64(&mut f, 1024, 0x1000_0000_0000_0000);
    put64(&mut f, 1536, 0x3000_0000_0000_0000);
    put64(&mut f, 1536 + 2 * 8, 0x2000_0000_0000_0000);
    put64(&mut f, 1536 + 5 * 8, 0x3001_0000_0000_0000);
    f.extend_from_slice(&pattern(4096, 1));
    f.extend_from_slice(&pattern(4096, 2));
    fs::write(dir.join("x-sesparse.vmdk"), &f).unwrap();
    let img = dir.join("x.vmdk");
    fs::write(
        &img,
        "# Disk DescriptorFile\nversion=1\nCID=1\nparentCID=ffffffff\ncreateType=\"seSparse\"\n\
         \n# Extent description\nRW 2048 SESPARSE \"x-sesparse.vmdk\"\n",
    )
    .unwrap();
    let expect = qemu_raw(&qemu_img, &img, &dir);
    assert!(expect[..4096] == pattern(4096, 1)[..]);

    let g = BlockGraph::new();
    let name = open(&g, &img, true);
    compare_info(&qemu_img, &g, &name, &img);
    let blk = BlockBackend::new(&g, &name, BLK_PERM_CONSISTENT_READ, SHARED).unwrap();
    assert!(read_all(&blk) == expect);
    assert_eq!(g.block_status(&name, 8192, 4096).unwrap().ret & 0x03, 0x02);
    g.check(&name, 0).unwrap();
    drop(blk);

    // No writes.
    let mut o = QDict::new();
    o.put("read-only", "off");
    let e = BlockGraph::new().open_image(Some(path_str(&img)), o).unwrap_err();
    assert_eq!(e.message(), "No write support for seSparse images available");
}

/// Hand made VMFS (flat) and VMFS sparse ("COWD") extents read as in QEMU, and the sparse
/// one takes writes.
#[test]
fn vmfs_extents() {
    let Some((qemu_img, _)) = tools() else { return };
    let dir = scratch("vmfs");
    fs::write(dir.join("flat.img"), pattern(1 << 20, 3)).unwrap();
    // COWD: 2048 sectors in grains of 8, the L1 table at sector 1, one grain table at 2.
    let mut c = vec![0u8; 34 * 512];
    c[..4].copy_from_slice(b"COWD");
    put32(&mut c, 4, 1);
    put32(&mut c, 12, 2048);
    put32(&mut c, 16, 8);
    put32(&mut c, 20, 1);
    put32(&mut c, 24, 1);
    put32(&mut c, 512, 2);
    put32(&mut c, 1024, 34);
    put32(&mut c, 1024 + 3 * 4, 42);
    c.extend_from_slice(&pattern(4096, 4));
    c.extend_from_slice(&pattern(4096, 5));
    fs::write(dir.join("s.cowd"), &c).unwrap();
    let img = dir.join("v.vmdk");
    fs::write(
        &img,
        "# Disk DescriptorFile\nversion=1\nCID=1\nparentCID=ffffffff\ncreateType=\"vmfs\"\n\
         RW 2048 VMFS \"flat.img\"\nRW 2048 VMFSSPARSE \"s.cowd\"\n",
    )
    .unwrap();
    let expect = qemu_raw(&qemu_img, &img, &dir);
    assert_eq!(expect.len(), 2 << 20);

    let g = BlockGraph::new();
    let name = open(&g, &img, false);
    compare_info(&qemu_img, &g, &name, &img);
    let blk = BlockBackend::new(&g, &name, RW, SHARED).unwrap();
    assert!(read_all(&blk) == expect);
    let mut expect = expect;
    for (off, n, seed) in [(1000usize, 10usize, 6u8), ((1 << 20) + 5000, 9000, 7)] {
        let data = pattern(n, seed);
        blk.pwrite(off as u64, &data).unwrap();
        expect[off..off + n].copy_from_slice(&data);
    }
    assert!(read_all(&blk) == expect);
    drop(blk);
    drop(g);
    qemu_verify(&qemu_img, &img, &dir, &expect);
}

/// `vmdk_co_check()` finds a grain past the end of the file, as QEMU does.
#[test]
fn check_finds_corruption() {
    let Some((qemu_img, _)) = tools() else { return };
    let dir = scratch("check");
    let img = dir.join("c.vmdk");
    let g = BlockGraph::new();
    let mut o = QDict::new();
    o.put("size", "1M");
    g.create_image("vmdk", path_str(&img), &mut o).unwrap();
    drop(g);
    let mut f = fs::read(&img).unwrap();
    // The grain directory (not the redundant one) points to the grain table after it.
    let gd = u64::from_le_bytes(f[56..64].try_into().unwrap()) as usize;
    let gt = u32::from_le_bytes(f[gd * 512..gd * 512 + 4].try_into().unwrap()) as usize;
    put32(&mut f, gt * 512 + 8, 0x0100_0000);
    fs::write(&img, &f).unwrap();
    let out = run_fail(&qemu_img, &["check", "-f", "vmdk", path_str(&img)]);
    assert!(out.contains("ERROR: cluster offset for sector 256 points after EOF"), "{out}");
    assert!(out.contains("Check failed: Invalid argument"), "{out}");

    let g = BlockGraph::new();
    let name = open(&g, &img, true);
    let e = g.check(&name, 0).unwrap_err();
    assert_eq!(e.to_string(), "Invalid argument");
    let e = g.check(&name, 2).unwrap_err();
    assert_eq!(e.to_string(), "This image format does not support checks");
}

/// `blockdev-create` with the extents given as a list.
#[test]
fn blockdev_create_with_extents() {
    let Some((qemu_img, _)) = tools() else { return };
    let dir = scratch("blockdev-create");
    let g = BlockGraph::new();
    let file = |name: &str| {
        let p = dir.join(name);
        g.blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
            r#"{{"driver": "file", "filename": "{}", "size": 0}}"#,
            path_str(&p)
        )))
        .unwrap();
        format!(r#"{{"driver": "file", "filename": "{}"}}"#, path_str(&p))
    };
    let create = |extra: &str| {
        g.blockdev_create(from_json::<BlockdevCreateOptions>(&format!(
            r#"{{"driver": "vmdk", {extra}}}"#
        )))
    };
    let (d, e1, e2) = (file("d.vmdk"), file("d-f001.vmdk"), file("d-f002.vmdk"));
    create(&format!(
        r#""file": {d}, "extents": [{e1}], "size": 1048576, "subformat": "twoGbMaxExtentFlat",
           "adapter-type": "lsilogic", "hwversion": "8", "toolsversion": "1""#
    ))
    .unwrap();
    let desc = fs::read_to_string(dir.join("d.vmdk")).unwrap();
    assert!(desc.contains("RW 2048 FLAT \"d-f001.vmdk\" 0\n"), "{desc}");
    assert!(desc.contains("ddb.virtualHWVersion = \"8\"\n"));
    assert!(desc.contains("ddb.geometry.heads = \"255\"\n"));
    assert!(desc.contains("ddb.adapterType = \"lsilogic\"\n"));
    assert!(desc.contains("ddb.toolsVersion = \"1\"\n"));
    assert_eq!(fs::metadata(dir.join("d-f001.vmdk")).unwrap().len(), 1 << 20);
    run(&qemu_img, &["check", "-f", "vmdk", path_str(&dir.join("d.vmdk"))]);

    let e = create(&format!(r#""file": {d}, "size": 1048576, "subformat": "twoGbMaxExtentFlat""#))
        .unwrap_err();
    assert_eq!(e.message(), "Extent [0] not specified");
    let e = create(&format!(
        r#""file": {d}, "extents": [{e1}, {e2}], "size": 1048576,
           "subformat": "twoGbMaxExtentSparse""#
    ))
    .unwrap_err();
    assert_eq!(e.message(), "List of extents contains unused extents");
    let e = create(&format!(r#""file": {d}, "size": 1000"#)).unwrap_err();
    assert_eq!(e.message(), "Image size must be a multiple of 512 bytes");
    let e = create(&format!(
        r#""file": {d}, "extents": [{e1}], "size": 1048576, "subformat": "monolithicFlat",
           "zeroed-grain": true"#
    ))
    .unwrap_err();
    assert_eq!(e.message(), "Flat image can't enable zeroed grain");

    // monolithicSparse in the file itself, with zeroed grains.
    create(&format!(r#""file": {d}, "size": 1048576, "zeroed-grain": true"#)).unwrap();
    run(&qemu_img, &["check", "-f", "vmdk", path_str(&dir.join("d.vmdk"))]);
    let info = run(&qemu_img, &["info", "--output=json", path_str(&dir.join("d.vmdk"))]);
    assert_eq!(info_str(&info, "create-type"), "monolithicSparse");
}

/// Error messages as QEMU words them.
#[test]
fn errors() {
    let dir = scratch("errors");
    let g = BlockGraph::new();
    let create = |o: &str| {
        let mut qd = opts(o);
        qd.put("size", "1M");
        g.create_image("vmdk", path_str(&dir.join("e.vmdk")), &mut qd).unwrap_err().to_string()
    };
    assert_eq!(create("backing_fmt=qcow2"), "backing_file must be a vmdk image");
    assert_eq!(create("compat6=on,hwversion=7"), "compat6 cannot be enabled with hwversion set");
    assert_eq!(create("adapter_type=scsi"), "invalid parameter value: scsi");
    assert_eq!(create("subformat=sparse"), "invalid parameter value: sparse");
    assert_eq!(
        create("subformat=monolithicFlat,backing_file=b.vmdk"),
        "Flat image can't have backing file"
    );
    fs::write(dir.join("raw.img"), [0u8; 4096]).unwrap();
    assert_eq!(create("backing_file=raw.img"), "Invalid backing file format: raw. Must be vmdk");

    let open_err = |name: &str, bytes: &[u8]| {
        let p = dir.join(name);
        fs::write(&p, bytes).unwrap();
        let mut o = QDict::new();
        o.put("driver", "vmdk");
        BlockGraph::new().open_image(Some(path_str(&p)), o).unwrap_err().to_string()
    };
    assert_eq!(open_err("a.vmdk", b"KD"), "invalid VMDK image descriptor");
    assert_eq!(open_err("b.vmdk", b"version=1\n"), "invalid VMDK image descriptor");
    assert_eq!(
        open_err("c.vmdk", b"version=1\ncreateType=\"custom\"\n"),
        "Unsupported image type 'custom'"
    );
    assert_eq!(
        open_err("d.vmdk", b"version=1\ncreateType=\"monolithicFlat\"\nRW 12 FLAT \"x\"\n"),
        "Invalid extent line: RW 12 FLAT \"x\""
    );
    let mut h = vec![0u8; 4096];
    h[..4].copy_from_slice(b"KDMV");
    put32(&mut h, 4, 4);
    assert_eq!(open_err("e.vmdk", &h), "Unsupported VMDK version 4");
    put32(&mut h, 4, 1);
    put64(&mut h, 12, 2048);
    put64(&mut h, 20, 8);
    put32(&mut h, 44, 1024);
    assert_eq!(open_err("f.vmdk", &h), "L2 table size too big");
    put32(&mut h, 44, 0);
    assert_eq!(open_err("g.vmdk", &h), "L1 entry size is invalid");
}
