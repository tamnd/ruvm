// SPDX-License-Identifier: GPL-2.0-or-later

//! The `vvfat` protocol driver end to end: the disk made from a host directory compared byte
//! for byte with what `qemu-img convert` of the same `fat:` file name gives (floppy FAT12,
//! FAT16, FAT32 and a volume label), the open errors, and writes through `fat:rw:` landing in
//! the host directory the way they do when `qemu-io` makes the same writes. The comparisons
//! skip themselves when `qemu-img` or `qemu-io` is not installed.
//!
//! Two things are not compared. The unused tail of the last cluster of a file is whatever an
//! earlier read left in the driver's cluster buffer (in QEMU as well), so it depends on the
//! order of the reads; those bytes are masked out using the directory entries. The access
//! dates come from the host and running `qemu-img` reads the files, so they could differ if the
//! date changed in between; they are masked too.

#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use ruvm_block::{BLK_PERM_CONSISTENT_READ, BLK_PERM_WRITE, BlockBackend, BlockGraph};
use ruvm_qapi::QDict;

const RW: u64 = BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE;
const SHARED: u64 = BLK_PERM_CONSISTENT_READ;

/// `qemu-img`, or `None` (and a note) when it is not installed.
fn qemu_img() -> Option<PathBuf> {
    let candidates = ["/opt/homebrew/bin/qemu-img", "/usr/local/bin/qemu-img", "/usr/bin/qemu-img"];
    let found = candidates.iter().map(PathBuf::from).find(|p| p.exists());
    if found.is_none() {
        eprintln!("qemu-img not found, skipping");
    }
    found
}

fn scratch(test: &str) -> PathBuf {
    let base = option_env!("CARGO_TARGET_TMPDIR").map_or_else(std::env::temp_dir, PathBuf::from);
    let dir = base.join("ruvm-block-vvfat").join(test);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// A host tree with short and long names, names that need a `~1` short name, an empty file,
/// a file of several clusters and nested directories.
fn make_tree(root: &Path) {
    let w = |p: &str, data: &[u8]| fs::write(root.join(p), data).unwrap();
    fs::create_dir_all(root.join("sub/deeper")).unwrap();
    fs::create_dir_all(root.join("Another Directory")).unwrap();
    w("README.TXT", b"read me\n");
    w("lower.txt", b"lower case name\n");
    w("empty", b"");
    let big: Vec<u8> = (0..40_000u32).map(|i| (i * 7 + i / 251) as u8).collect();
    w("big.bin", &big);
    w("A long file name.text", b"long\n");
    w("A long file name.other", b"long 2\n");
    w("abc+def.txt", b"plus\n");
    w("abc def.txt", b"space\n");
    w("verylongname1.txt", b"1\n");
    w("verylongname2.txt", b"2\n");
    w("Mixed.Case", b"mixed\n");
    w("sub/deeper/f", b"deep file\n");
    w("sub/Deep Long Name In Sub.dat", &big[..5000]);
    w("Another Directory/x.c", b"int main;\n");
}

/// Opens `filename`, read-only unless `perm` asks for writes as `qemu-img` does.
fn open(
    g: &BlockGraph,
    filename: &str,
    perm: u64,
) -> ruvm_base::Result<std::sync::Arc<BlockBackend>> {
    let mut options = QDict::new();
    options.put("read-only", perm & BLK_PERM_WRITE == 0);
    let name = g.open_image(Some(filename), options)?;
    BlockBackend::new(g, &name, perm, SHARED)
}

fn le16(b: &[u8], o: usize) -> usize {
    usize::from(u16::from_le_bytes([b[o], b[o + 1]]))
}

fn le32(b: &[u8], o: usize) -> usize {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize
}

/// The layout of the file system from its boot sector.
struct Layout {
    bootsector: usize,
    fat: usize,
    root: usize,
    cluster_size: usize,
    fat_type: u32,
}

impl Layout {
    fn new(disk: &[u8], fat_type: u32) -> Self {
        // A floppy has no MBR; otherwise the first partition starts at sector 63.
        let bootsector =
            if disk[0x1fe] == 0x55 && disk[0] == 0xeb { 0 } else { le32(disk, 0x1c6) * 512 };
        let b = &disk[bootsector..];
        let spc = usize::from(b[13]);
        let fat = bootsector + le16(b, 14) * 512;
        let fat_bytes = le16(b, 22) * 512;
        Layout { bootsector, fat, root: fat + 2 * fat_bytes, cluster_size: spc * 512, fat_type }
    }

    fn cluster(&self, c: usize) -> usize {
        self.root + c * self.cluster_size
    }

    fn fat_get(&self, disk: &[u8], c: usize) -> usize {
        let f = &disk[self.fat..];
        match self.fat_type {
            12 => {
                let v = le16(f, c * 3 / 2);
                if c & 1 != 0 { v >> 4 } else { v & 0xfff }
            }
            16 => le16(f, c * 2),
            _ => le32(f, c * 4) & 0x0fff_ffff,
        }
    }

    fn eof(&self, c: usize) -> bool {
        match self.fat_type {
            12 => c >= 0xff8,
            16 => c >= 0xfff8,
            _ => c >= 0x0fff_fff8,
        }
    }

    fn chain(&self, disk: &[u8], first: usize) -> Vec<usize> {
        let mut v = Vec::new();
        let mut c = first;
        while c >= 2 && !self.eof(c) && v.len() < 100_000 {
            v.push(c);
            c = self.fat_get(disk, c);
        }
        v
    }

    /// The entries of the directory whose clusters are `clusters` (the root when empty).
    fn entries(&self, disk: &[u8], clusters: &[usize]) -> Vec<[u8; 32]> {
        let ranges: Vec<usize> = if clusters.is_empty() { vec![0] } else { clusters.to_vec() };
        let mut v = Vec::new();
        for c in ranges {
            let at = self.cluster(c);
            for e in disk[at..at + self.cluster_size].chunks(32) {
                v.push(e.try_into().unwrap());
            }
        }
        v
    }

    /// Walks the tree and calls `f` with each short name entry's offset in `disk`.
    fn walk(
        &self,
        disk: &[u8],
        clusters: &[usize],
        names: &mut Vec<String>,
        f: &mut impl FnMut(usize, &[u8; 32]),
    ) {
        let ranges: Vec<usize> = if clusters.is_empty() { vec![0] } else { clusters.to_vec() };
        for c in ranges {
            let at = self.cluster(c);
            for i in 0..self.cluster_size / 32 {
                let e: [u8; 32] = disk[at + i * 32..at + i * 32 + 32].try_into().unwrap();
                if e[0] == 0 || e[0] == 0xe5 || e[11] == 0x0f || e[11] & 0x08 != 0 || e[0] == b'.' {
                    continue;
                }
                names.push(String::from_utf8_lossy(&e[..11]).into_owned());
                f(at + i * 32, &e);
                let begin = le16(&e, 26) | (le16(&e, 20) << 16);
                if e[11] & 0x10 != 0 {
                    let ch = self.chain(disk, begin);
                    self.walk(disk, &ch, names, f);
                }
            }
        }
    }
}

/// Masks the access dates and the unused tails of the files' last clusters.
fn mask(disk: &mut [u8], fat_type: u32) {
    let l = Layout::new(disk, fat_type);
    let mut tails = Vec::new();
    let mut dates = Vec::new();
    let copy = disk.to_vec();
    l.walk(&copy, &[], &mut Vec::new(), &mut |at, e| {
        dates.push(at + 18);
        if e[11] & 0x10 == 0 {
            let size = le32(e, 28);
            let begin = le16(e, 26) | (le16(e, 20) << 16);
            if let Some(&last) = l.chain(&copy, begin).last() {
                let used = size - (size - 1) / l.cluster_size * l.cluster_size;
                tails.push((l.cluster(last) + used, l.cluster(last) + l.cluster_size));
            }
        }
    });
    for at in dates {
        disk[at..at + 2].fill(0);
    }
    for (a, b) in tails {
        disk[a..b].fill(0);
    }
}

/// How much of the disk to compare: past the FATs, the root directory and the data.
const COMPARED: usize = 8 << 20;

/// Compares what this driver reads from `filename` with `qemu-img convert` of it.
fn compare(test: &str, variant: &str, dir_opts: &str, expect_size: u64, fat_type: u32) {
    let Some(qemu_img) = qemu_img() else { return };
    let dir = scratch(test);
    let tree = dir.join("tree");
    fs::create_dir_all(&tree).unwrap();
    make_tree(&tree);
    let tree = tree.to_str().unwrap();
    let filename = if dir_opts.is_empty() {
        format!("fat:{variant}{tree}")
    } else {
        format!(r#"json:{{"driver": "vvfat", "dir": "{tree}", {dir_opts}}}"#)
    };

    let out = dir.join("out.raw");
    let st = Command::new(&qemu_img)
        .args(["convert", "-f", "vvfat", "-O", "raw", &filename, out.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        st.status.success(),
        "qemu-img convert failed: {}",
        String::from_utf8_lossy(&st.stderr)
    );
    let theirs = fs::read(&out).unwrap();
    // The image is sparse and large; nothing else is needed from it.
    fs::remove_file(&out).unwrap();

    let g = BlockGraph::new();
    let blk = open(&g, &filename, BLK_PERM_CONSISTENT_READ).unwrap();
    let len = blk.getlength().unwrap();
    assert_eq!(len, expect_size);
    assert_eq!(len, theirs.len() as u64);
    let n = COMPARED.min(len as usize);
    let mut ours = vec![0u8; n];
    // Read in pieces the size `qemu-img convert` uses, in order.
    for (i, chunk) in ours.chunks_mut(2 << 20).enumerate() {
        blk.pread((i * (2 << 20)) as u64, chunk).unwrap();
    }
    let mut last = vec![0u8; 1 << 20];
    blk.pread(len - (1 << 20), &mut last).unwrap();
    assert_eq!(last, theirs[len as usize - (1 << 20)..], "{test}: the end of the disk differs");

    let l = Layout::new(&theirs, fat_type);
    // QEMU writes a FAT16 style boot sector for FAT32 too.
    let fs_type: &[u8] = if fat_type == 12 { b"FAT12   " } else { b"FAT16   " };
    assert_eq!(&theirs[l.bootsector + 0x36..l.bootsector + 0x3e], fs_type, "{test}");
    // The MBR and the boot sector exactly.
    assert_eq!(
        ours[..l.bootsector + 512],
        theirs[..l.bootsector + 512],
        "{test}: MBR or boot sector"
    );
    // Both FATs exactly.
    assert_eq!(ours[l.fat..l.root], theirs[l.fat..l.root], "{test}: FAT");

    // The same names in the same places.
    let mut names_theirs = Vec::new();
    let mut names_ours = Vec::new();
    let mut theirs_head = theirs[..n].to_vec();
    Layout::new(&theirs_head, fat_type).walk(&theirs_head, &[], &mut names_theirs, &mut |_, _| {});
    Layout::new(&ours, fat_type).walk(&ours, &[], &mut names_ours, &mut |_, _| {});
    assert_eq!(names_ours, names_theirs, "{test}: directory tree");
    assert!(names_theirs.iter().any(|n| n.contains("~1")), "{names_theirs:?}");
    // The long name entries of the root directory exactly.
    let lfn =
        |d: &[u8]| l.entries(d, &[]).into_iter().filter(|e| e[11] == 0x0f).collect::<Vec<_>>();
    assert_eq!(lfn(&ours), lfn(&theirs_head), "{test}: long names");

    // Everything else, with the nondeterministic bytes masked.
    mask(&mut theirs_head, fat_type);
    mask(&mut ours, fat_type);
    if let Some(i) = (0..n).find(|&i| ours[i] != theirs_head[i]) {
        panic!(
            "{test}: first difference at {i:#x}: ours {:02x?} theirs {:02x?}",
            &ours[i..(i + 32).min(n)],
            &theirs_head[i..(i + 32).min(n)]
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn floppy_fat12_matches_qemu() {
    compare("fat12", "12:floppy:", "", 1_474_560, 12);
}

#[test]
fn fat16_matches_qemu() {
    compare("fat16", "16:", "", 528_482_304, 16);
}

#[test]
fn fat32_matches_qemu() {
    compare("fat32", "32:", "", 528_482_304, 32);
}

#[test]
fn default_floppy_matches_qemu() {
    // FAT12 unless asked otherwise, 36 sectors per track.
    compare("floppy", "floppy:", "", 2_949_120, 12);
}

#[test]
fn label_matches_qemu() {
    compare("label", "", r#""label": "MY DISK""#, 528_482_304, 16);
}

fn open_err(filename: &str, perm: u64) -> String {
    let g = BlockGraph::new();
    open(&g, filename, perm).map(|_| ()).unwrap_err().message().to_string()
}

#[test]
fn open_errors() {
    let dir = scratch("errors");
    let d = dir.to_str().unwrap();
    assert_eq!(
        open_err(
            &format!(r#"json:{{"driver": "vvfat", "dir": "{d}", "label": "TWELVE CHARS"}}"#),
            BLK_PERM_CONSISTENT_READ
        ),
        "vvfat label cannot be longer than 11 bytes"
    );
    assert_eq!(
        open_err(
            &format!(r#"json:{{"driver": "vvfat", "dir": "{d}", "fat-type": 8}}"#),
            BLK_PERM_CONSISTENT_READ
        ),
        "Valid FAT types are only 12, 16 and 32"
    );
    assert_eq!(
        open_err(&format!("fat:{d}/does-not-exist"), BLK_PERM_CONSISTENT_READ),
        format!("Could not read directory {d}/does-not-exist")
    );
    // Without rw: the node is read-only.
    assert_eq!(open_err(&format!("fat:{d}"), RW), "Image is read-only");
}

/// The tree the write tests start from.
fn make_rw_tree(tree: &Path) -> Vec<u8> {
    fs::create_dir_all(tree).unwrap();
    fs::write(tree.join("hello.txt"), b"hello, world\n").unwrap();
    let big: Vec<u8> = (0..20_000u32).map(|i| i as u8).collect();
    fs::write(tree.join("big.bin"), &big).unwrap();
    big
}

/// The names and contents of the files in `tree`, sorted.
fn contents(tree: &Path) -> Vec<(String, Vec<u8>)> {
    let mut v: Vec<_> = fs::read_dir(tree)
        .unwrap()
        .map(|e| {
            let e = e.unwrap();
            (e.file_name().into_string().unwrap(), fs::read(e.path()).unwrap())
        })
        .collect();
    v.sort();
    v
}

#[test]
fn writes_reach_the_host() {
    let dir = scratch("rw");
    let tree = dir.join("tree");
    let big = make_rw_tree(&tree);
    let t = tree.to_str().unwrap();

    let g = BlockGraph::new();
    let blk = match open(&g, &format!("fat:16:rw:{t}"), RW) {
        Ok(b) => b,
        Err(e) if e.message() == "Failed to locate qcow driver" => {
            eprintln!("qcow driver not available, skipping");
            return;
        }
        Err(e) => panic!("{}", e.message()),
    };
    let len = blk.getlength().unwrap() as usize;
    let mut head = vec![0u8; COMPARED.min(len)];
    blk.pread(0, &mut head).unwrap();
    let l = Layout::new(&head, 16);

    let mut found = None;
    let mut big_first = 0;
    let mut big_at = 0;
    l.walk(&head, &[], &mut Vec::new(), &mut |at, e| match &e[..11] {
        b"HELLO   TXT" => found = Some((at, le16(e, 26))),
        b"BIG     BIN" => (big_first, big_at) = (le16(e, 26), at),
        _ => {}
    });
    let (hello_at, cluster) = found.expect("hello.txt in the root directory");

    // Every write, to replay through QEMU.
    let mut writes: Vec<(u64, Vec<u8>)> = Vec::new();
    let mut write = |at: usize, data: &[u8]| {
        blk.pwrite(at as u64, data).unwrap();
        writes.push((at as u64, data.to_vec()));
    };
    let sector_of = |at: usize| at / 512 * 512;
    let read_sector = |at: usize| {
        let mut s = vec![0u8; 512];
        blk.pread(sector_of(at) as u64, &mut s).unwrap();
        s
    };

    // Same length, new contents: only the data changes.
    let at = l.cluster(cluster);
    let mut sector = head[at..at + 512].to_vec();
    sector[..13].copy_from_slice(b"HELLO, WORLD\n");
    write(at, &sector);
    assert_eq!(fs::read(tree.join("hello.txt")).unwrap(), b"HELLO, WORLD\n");

    // A change in the second cluster of a larger file.
    let second = l.fat_get(&head, big_first);
    let at = l.cluster(second);
    let mut sector = head[at..at + 512].to_vec();
    sector[..4].copy_from_slice(b"ruvm");
    write(at, &sector);
    let mut expect = big.clone();
    expect[l.cluster_size..l.cluster_size + 4].copy_from_slice(b"ruvm");
    assert_eq!(fs::read(tree.join("big.bin")).unwrap(), expect);
    // Reads see the change too.
    assert_eq!(read_sector(at), sector);

    // A new short name without the matching long name renames the file.
    let mut dir_sector = read_sector(hello_at);
    dir_sector[hello_at % 512 + 4] = b'X';
    write(sector_of(hello_at), &dir_sector);
    assert!(!tree.join("hello.txt").exists());
    assert_eq!(fs::read(tree.join("hellx.txt")).unwrap(), b"HELLO, WORLD\n");

    // Freeing the entry (and its long name) and then the clusters. The directory alone is
    // not consistent, as the FAT still has the chain. As in QEMU, handle_deletes() leaves
    // the host file of a freed entry alone, so the file stays.
    let mut dir_sector = read_sector(big_at);
    let mut i = big_at % 512;
    dir_sector[i] = 0xe5;
    while i >= 32 && dir_sector[i - 32 + 11] == 0x0f {
        i -= 32;
        dir_sector[i] = 0xe5;
    }
    write(sector_of(big_at), &dir_sector);
    let chain = l.chain(&head, big_first);
    let fat_at = sector_of(l.fat + chain[0] * 2);
    assert_eq!(fat_at, sector_of(l.fat + chain[chain.len() - 1] * 2));
    let mut fat_sector = read_sector(fat_at);
    for c in &chain {
        let o = l.fat + c * 2 - fat_at;
        fat_sector[o..o + 2].fill(0);
    }
    write(fat_at, &fat_sector);
    let ours = contents(&tree);
    assert_eq!(ours.len(), 2);

    // The boot sector is protected, except for the dirty flag.
    let bs = l.bootsector;
    let mut boot = head[bs..bs + 512].to_vec();
    boot[37] = 1;
    blk.pwrite(bs as u64, &boot).unwrap();
    boot[3] ^= 1;
    assert_eq!(blk.pwrite(bs as u64, &boot).unwrap_err().raw_os_error(), Some(libc::EPERM));
    drop(blk);
    drop(g);

    // The same writes through QEMU's vvfat leave the same host directory.
    let qemu_io = qemu_img().map(|p| p.with_file_name("qemu-io")).filter(|p| p.exists());
    if let Some(qemu_io) = qemu_io {
        let tree2 = dir.join("tree2");
        make_rw_tree(&tree2);
        let mut cmd = Command::new(qemu_io);
        cmd.args(["-f", "vvfat"]);
        for (i, (at, data)) in writes.iter().enumerate() {
            let src = dir.join(format!("w{i}"));
            fs::write(&src, data).unwrap();
            cmd.arg("-c").arg(format!("write -q -s {} {at} {}", src.display(), data.len()));
        }
        cmd.arg(format!("fat:16:rw:{}", tree2.display()));
        let st = cmd.output().unwrap();
        assert!(st.status.success(), "{}", String::from_utf8_lossy(&st.stderr));
        assert_eq!(ours, contents(&tree2));
    } else {
        eprintln!("qemu-io not found, not comparing");
    }
    let _ = fs::remove_dir_all(&dir);
}
