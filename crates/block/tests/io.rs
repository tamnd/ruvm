// SPDX-License-Identifier: GPL-2.0-or-later

//! The I/O path end to end: `file` and `raw` nodes made through `blockdev-add`, block backends on
//! top, and `-drive`.
//!
//! Some cases are ports of the qemu-iotests: 153 (image locking basics), and the offset and
//! size checks of 171 (raw offset/size windows). The images are small files under the target
//! directory.

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use ruvm_block::{
    BLK_PERM_CONSISTENT_READ, BLK_PERM_WRITE, BLK_PERM_WRITE_UNCHANGED, BlockBackend, BlockGraph,
    BlockInterfaceType,
};
use ruvm_qapi::json;
use ruvm_qapi::types::BlockdevOptions;
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};

const RW: u64 = BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE;
const SHARED: u64 = BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE_UNCHANGED;

/// A fresh directory for one test under the target directory.
fn scratch(test: &str) -> PathBuf {
    let base = option_env!("CARGO_TARGET_TMPDIR").map_or_else(std::env::temp_dir, PathBuf::from);
    let dir = base.join("ruvm-block-io").join(test);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// An image of `len` bytes where byte `i` is `i % 251`, so every offset is recognisable.
fn image(dir: &std::path::Path, name: &str, len: usize) -> String {
    let path = dir.join(name);
    let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
    fs::write(&path, data).unwrap();
    path.to_str().unwrap().to_string()
}

fn opts(s: &str) -> BlockdevOptions {
    let mut v = QObjectInputVisitor::new(json::from_str(s).unwrap());
    let mut o = BlockdevOptions::default();
    BlockdevOptions::visit(&mut v, None, &mut o).unwrap();
    o
}

fn add(g: &BlockGraph, s: &str) -> ruvm_base::Result<()> {
    g.blockdev_add(opts(s))
}

/// `blockdev-add` of a raw node `name` over a file node `name-file` for `path`.
fn add_raw(g: &BlockGraph, name: &str, path: &str, extra: &str) -> ruvm_base::Result<()> {
    add(
        g,
        &format!(
            r#"{{"driver": "raw", "node-name": "{name}", {extra}
                "file": {{"driver": "file", "node-name": "{name}-file", "filename": "{path}"}}}}"#
        ),
    )
}

fn errno(e: &std::io::Error) -> i32 {
    e.raw_os_error().unwrap()
}

#[test]
fn file_round_trip() {
    let dir = scratch("round-trip");
    let path = image(&dir, "a.img", 65536);
    let g = BlockGraph::new();
    add_raw(&g, "r", &path, "").unwrap();
    assert_eq!(g.node("r").unwrap().children, ["r-file"]);
    let blk = BlockBackend::new(&g, "r", RW, SHARED).unwrap();
    assert_eq!(blk.getlength().unwrap(), 65536);
    assert!(blk.is_writable() && blk.enable_write_cache());

    blk.pwrite(1000, b"hello, world").unwrap();
    let mut buf = [0u8; 12];
    blk.pread(1000, &mut buf).unwrap();
    assert_eq!(&buf, b"hello, world");
    blk.flush().unwrap();
    // Writethrough flushes after every write.
    blk.set_enable_write_cache(false);
    blk.pwrite(65536 - 4, b"tail").unwrap();
    let on_disk = fs::read(&path).unwrap();
    assert_eq!(&on_disk[1000..1012], b"hello, world");
    assert_eq!(&on_disk[65532..], b"tail");
    assert_eq!(on_disk[999], (999 % 251) as u8);

    // blk_check_byte_request(): nothing past the end.
    assert_eq!(errno(&blk.pwrite(65536 - 2, b"xyz").unwrap_err()), libc::EIO);
    assert_eq!(errno(&blk.pread(65537, &mut buf).unwrap_err()), libc::EIO);

    // The backend holds the node, blockdev-del has to wait.
    assert_eq!(g.blockdev_del("r").unwrap_err().message(), "Node r is in use");
    drop(blk);
    g.blockdev_del("r").unwrap();
    assert!(g.nodes().is_empty());
}

#[test]
fn file_direct_and_truncate() {
    let dir = scratch("direct");
    let path = image(&dir, "a.img", 16384);
    let g = BlockGraph::new();
    let r = add(
        &g,
        &format!(
            r#"{{"driver": "file", "node-name": "f", "filename": "{path}", "cache": {{"direct": true}}}}"#
        ),
    );
    if let Err(e) = &r {
        // tmpfs and some container file systems have no O_DIRECT.
        assert!(e.message().ends_with("filesystem does not support O_DIRECT"), "{e}");
        return;
    }
    let blk = BlockBackend::new(&g, "f", RW | ruvm_block::BLK_PERM_RESIZE, SHARED).unwrap();
    // Unaligned requests go through a bounce buffer on Linux.
    blk.pwrite(4000, b"across a boundary").unwrap();
    let mut buf = [0u8; 17];
    blk.pread(4000, &mut buf).unwrap();
    assert_eq!(&buf, b"across a boundary");
    blk.pwrite(16384 - 3, b"end").unwrap();
    assert_eq!(fs::metadata(&path).unwrap().len(), 16384);
    blk.truncate(8192).unwrap();
    assert_eq!(blk.getlength().unwrap(), 8192);
    // A length that is not a whole sector is reported rounded up.
    blk.truncate(1000).unwrap();
    assert_eq!(blk.getlength().unwrap(), 1024);
    let mut tail = [0xffu8; 24];
    blk.pread(1000, &mut tail).unwrap();
    assert_eq!(tail, [0; 24]);
}

#[test]
fn raw_window() {
    let dir = scratch("raw-window");
    let path = image(&dir, "a.img", 8192);
    let g = BlockGraph::new();
    add_raw(&g, "w", &path, r#""offset": 1024, "size": 2048,"#).unwrap();
    assert_eq!(g.node("w").unwrap().size, 2048);
    let blk = BlockBackend::new(&g, "w", RW, SHARED).unwrap();
    let mut buf = [0u8; 4];
    blk.pread(0, &mut buf).unwrap();
    assert_eq!(buf, [1024 % 251, 1025 % 251, 1026 % 251, 1027 % 251].map(|v| v as u8));
    blk.pwrite(2044, b"last").unwrap();
    assert_eq!(&fs::read(&path).unwrap()[3068..3072], b"last");
    assert_eq!(errno(&blk.pwrite(2045, b"last").unwrap_err()), libc::EIO);
    assert_eq!(
        blk.truncate(4096).unwrap_err().message(),
        "blk_truncate() needs the 'resize' permission"
    );
    drop(blk);
    let blk = BlockBackend::new(&g, "w", RW | ruvm_block::BLK_PERM_RESIZE, SHARED).unwrap();
    assert_eq!(blk.truncate(4096).unwrap_err().message(), "Cannot resize fixed-size raw disks");
    // A window does not let anybody else resize the file under it.
    let e = BlockBackend::new(&g, "w-file", ruvm_block::BLK_PERM_RESIZE, SHARED).unwrap_err();
    assert!(e.message().starts_with("Permission conflict on node 'w-file'"), "{e}");
    drop(blk);

    // iotest 171: offset and size against the size of the file.
    let err = |extra: &str| add_raw(&g, "x", &path, extra).unwrap_err().message().to_string();
    assert_eq!(
        err(r#""offset": 8193,"#),
        "Offset (8193) cannot be greater than size of the containing file (8192)"
    );
    assert_eq!(
        err(r#""offset": 4096, "size": 4608,"#),
        "The sum of offset (4096) and size (4608) has to be smaller or equal to the  actual size \
         of the containing file (8192)"
    );
    assert_eq!(err(r#""size": 1000,"#), "Specified size is not multiple of 512");
    assert!(g.node("x").is_none() && g.node("x-file").is_none());

    // Without a size the window runs to the end of the file and follows it.
    add_raw(&g, "tail", &path, r#""offset": 4096,"#).unwrap();
    let blk = BlockBackend::new(&g, "tail", RW | ruvm_block::BLK_PERM_RESIZE, SHARED).unwrap();
    assert_eq!(blk.getlength().unwrap(), 4096);
    blk.truncate(8192).unwrap();
    assert_eq!(fs::metadata(&path).unwrap().len(), 12288);
}

#[test]
fn read_only() {
    let dir = scratch("read-only");
    let path = image(&dir, "a.img", 4096);
    let g = BlockGraph::new();
    add_raw(&g, "ro", &path, r#""read-only": true,"#).unwrap();
    assert!(g.node("ro-file").unwrap().read_only);
    let e = BlockBackend::new(&g, "ro", RW, SHARED).unwrap_err();
    assert_eq!(e.message(), "Block node is read-only");
    let blk = BlockBackend::new(&g, "ro", BLK_PERM_CONSISTENT_READ, SHARED).unwrap();
    assert!(blk.is_read_only());
    assert_eq!(errno(&blk.pwrite(0, b"x").unwrap_err()), libc::EPERM);
    assert_eq!(errno(&blk.pwrite_zeroes(0, 512, false).unwrap_err()), libc::EPERM);
    let mut b = [0u8; 2];
    blk.pread(250, &mut b).unwrap();
    assert_eq!(b, [250, 0]);
    // A backend without the write permission cannot write to a writable node either.
    add_raw(&g, "rw", &path, "").unwrap();
    let blk = BlockBackend::new(&g, "rw", BLK_PERM_CONSISTENT_READ, SHARED).unwrap();
    assert_eq!(errno(&blk.pwrite(0, b"x").unwrap_err()), libc::EPERM);
    blk.set_perm(RW, SHARED).unwrap();
    blk.pwrite(0, b"x").unwrap();

    // force-share needs read-only.
    let e = add(
        &g,
        &format!(
            r#"{{"driver": "file", "node-name": "fs", "filename": "{path}", "force-share": true}}"#
        ),
    )
    .unwrap_err();
    assert_eq!(e.message(), "force-share=on can only be used with read-only images");
}

#[test]
fn permission_conflict_in_one_graph() {
    let dir = scratch("perm");
    let path = image(&dir, "a.img", 4096);
    let g = BlockGraph::new();
    add_raw(&g, "r", &path, "").unwrap();
    let a = BlockBackend::new(&g, "r", RW, SHARED).unwrap();
    let e = BlockBackend::new(&g, "r", RW, SHARED).unwrap_err();
    assert_eq!(
        e.message(),
        "Permission conflict on node 'r': permissions 'write' are both required by an unnamed \
         block device (uses node 'r' as 'root' child) and unshared by an unnamed block device \
         (uses node 'r' as 'root' child)."
    );
    // share-rw=on lets two writers in.
    a.set_perm(RW, SHARED | BLK_PERM_WRITE).unwrap();
    let b = BlockBackend::new(&g, "r", RW, SHARED | BLK_PERM_WRITE).unwrap();
    a.attach_dev("disk0").unwrap();
    let e = b.set_perm(RW, SHARED).unwrap_err();
    assert!(e.message().contains("and unshared by an unnamed block device"), "{e}");
    assert!(e.message().contains("required by block device 'disk0'"), "{e}");
}

/// iotest 153: two opens of one image, each as if from its own process. OFD locks belong to
/// the open file, so they conflict inside one process as well. Elsewhere QEMU falls back to
/// POSIX locks, which a process never conflicts with itself on, so there is nothing to see.
#[cfg(target_os = "linux")]
#[test]
fn image_locking() {
    let dir = scratch("locking");
    let path = image(&dir, "a.img", 4096);
    let writer = BlockGraph::new();
    add_raw(&writer, "r", &path, "").unwrap();
    let w = BlockBackend::new(&writer, "r", RW, SHARED).unwrap();

    // Another writer.
    let g = BlockGraph::new();
    add_raw(&g, "r", &path, "").unwrap();
    let e = BlockBackend::new(&g, "r", RW, SHARED).unwrap_err();
    assert_eq!(e.message(), "Failed to get \"write\" lock");
    assert_eq!(
        e.hint_text(),
        Some(format!("Is another process using the image [{path}]?\n").as_str())
    );
    // A reader that does not share writes.
    let e = BlockBackend::new(&g, "r", BLK_PERM_CONSISTENT_READ, SHARED).unwrap_err();
    assert_eq!(e.message(), "Failed to get shared \"write\" lock");
    // A reader that does, as qemu-io -r -U.
    let r = BlockBackend::new(&g, "r", BLK_PERM_CONSISTENT_READ, SHARED | BLK_PERM_WRITE).unwrap();
    // Now the first writer cannot drop sharing reads.
    let e = w.set_perm(RW, BLK_PERM_WRITE_UNCHANGED).unwrap_err();
    assert_eq!(e.message(), "Failed to get shared \"consistent read\" lock");
    drop(r);
    g.blockdev_del("r").unwrap();

    // force-share on a read-only node.
    let g = BlockGraph::new();
    add_raw(&g, "r", &path, r#""read-only": true, "force-share": true,"#).unwrap();
    BlockBackend::new(&g, "r", BLK_PERM_CONSISTENT_READ, SHARED).unwrap();

    // locking=off takes no locks and so sees none.
    let g = BlockGraph::new();
    add(
        &g,
        &format!(
            r#"{{"driver": "raw", "node-name": "r",
                "file": {{"driver": "file", "filename": "{path}", "locking": "off"}}}}"#
        ),
    )
    .unwrap();
    BlockBackend::new(&g, "r", RW, SHARED).unwrap();

    // Once the writer is gone, the image is free again.
    drop(w);
    writer.blockdev_del("r").unwrap();
    let g = BlockGraph::new();
    add_raw(&g, "r", &path, "").unwrap();
    BlockBackend::new(&g, "r", RW, SHARED).unwrap();
}

#[test]
fn discard_and_write_zeroes() {
    let dir = scratch("zeroes");
    let path = image(&dir, "a.img", 65536);
    let g = BlockGraph::new();
    add_raw(&g, "r", &path, r#""discard": "unmap","#).unwrap();
    let blk = BlockBackend::new(&g, "r", RW, SHARED).unwrap();
    let read = |off: u64, len: usize| {
        let mut b = vec![0xaa; len];
        blk.pread(off, &mut b).unwrap();
        b
    };

    blk.pwrite_zeroes(100, 300, false).unwrap();
    assert!(read(100, 300).iter().all(|&b| b == 0));
    assert_eq!(read(99, 1)[0], 99);
    assert_eq!(read(400, 1)[0], (400 % 251) as u8);
    blk.pwrite_zeroes(8192, 8192, true).unwrap();
    assert!(read(8192, 8192).iter().all(|&b| b == 0));
    assert_eq!(fs::metadata(&path).unwrap().len(), 65536);

    // Discard is only advice, but where the host punches holes the data reads back as zeroes.
    blk.pdiscard(32768, 16384).unwrap();
    if cfg!(any(target_os = "linux", target_os = "macos")) {
        assert!(read(32768, 16384).iter().all(|&b| b == 0));
    }
    assert_eq!(read(32767, 1)[0], (32767 % 251) as u8);
    assert_eq!(fs::metadata(&path).unwrap().len(), 65536);

    // Without discard=unmap on the raw node discard does nothing at all.
    let g = BlockGraph::new();
    add_raw(&g, "r", &path, "").unwrap();
    let blk = BlockBackend::new(&g, "r", RW, SHARED).unwrap();
    blk.pdiscard(0, 4096).unwrap();
    let mut b = [0u8; 1];
    blk.pread(1, &mut b).unwrap();
    assert_eq!(b[0], 1);
}

#[test]
fn blockdev_add_errors() {
    let dir = scratch("errors");
    let g = BlockGraph::new();
    let missing = dir.join("missing.img");
    let missing = missing.to_str().unwrap();
    let e = add(&g, &format!(r#"{{"driver": "file", "node-name": "f", "filename": "{missing}"}}"#))
        .unwrap_err();
    assert_eq!(e.message(), format!("Could not open '{missing}': No such file or directory"));

    let d = dir.to_str().unwrap();
    let e = add(
        &g,
        &format!(r#"{{"driver": "file", "node-name": "f", "filename": "{d}", "read-only": true}}"#),
    )
    .unwrap_err();
    assert_eq!(e.message(), format!("'file' driver requires '{d}' to be a regular file"));

    let path = image(&dir, "a.img", 4096);
    let e = add(
        &g,
        &format!(
            r#"{{"driver": "file", "node-name": "f", "filename": "{path}", "aio": "native"}}"#
        ),
    )
    .unwrap_err();
    if cfg!(target_os = "linux") {
        assert_eq!(
            e.message(),
            "aio=native was specified, but it requires cache.direct=on, which was not specified."
        );
    } else {
        assert_eq!(e.message(), "aio=native was specified, but is not supported in this build.");
    }

    let e = add(
        &g,
        &format!(
            r#"{{"driver": "file", "node-name": "f", "filename": "{path}", "detect-zeroes": "unmap"}}"#
        ),
    )
    .unwrap_err();
    assert_eq!(
        e.message(),
        "setting detect-zeroes to unmap is not allowed without setting discard operation to unmap"
    );

    // A reference to an existing file node.
    add(&g, &format!(r#"{{"driver": "file", "node-name": "f", "filename": "{path}"}}"#)).unwrap();
    add(&g, r#"{"driver": "raw", "node-name": "r", "file": "f"}"#).unwrap();
    assert_eq!(g.node("f").unwrap().parents, 1);
    assert_eq!(g.blockdev_del("f").unwrap_err().message(), "Node f is in use");
    g.blockdev_del("r").unwrap();
    g.blockdev_del("f").unwrap();
}

#[test]
fn drive_new() {
    let dir = scratch("drive");
    let path = image(&dir, "a.img", 8192);
    let g = BlockGraph::new();

    // No format: probed as raw, with the warning and block 0 guarded.
    let d = g.drive_new(&format!("file={path},if=none,id=d0"), BlockInterfaceType::Ide).unwrap();
    assert_eq!(
        d.warnings,
        [format!(
            "WARNING: Image format was not specified for '{path}' and probing guessed raw.\n         \
             Automatically detecting the format is dangerous for raw images, write operations on \
             block 0 will be restricted.\n         Specify the 'raw' format explicitly to remove \
             the restrictions."
        )]
    );
    let blk = g.backend("d0").unwrap();
    assert_eq!(blk.perm(), (BLK_PERM_CONSISTENT_READ, ruvm_block::BLK_PERM_ALL));
    blk.set_perm(RW, SHARED).unwrap();
    assert_eq!(errno(&blk.pwrite(0, b"QFI\xfb\0\0\0\x03").unwrap_err()), libc::EPERM);
    blk.pwrite(0, b"plain data").unwrap();
    blk.pwrite(512, b"QFI\xfb\0\0\0\x03").unwrap();
    assert_eq!(d.node_name.as_deref(), blk.node_name().as_deref());
    blk.set_perm(BLK_PERM_CONSISTENT_READ, ruvm_block::BLK_PERM_ALL).unwrap();

    // Cache modes and interfaces.
    let d = g
        .drive_new(
            &format!("file={path},format=raw,if=virtio,cache=unsafe"),
            BlockInterfaceType::Ide,
        )
        .unwrap();
    assert!(d.warnings.is_empty());
    assert_eq!((d.id.as_str(), d.device, d.write_cache), ("virtio0", Some("virtio-blk"), true));
    let cache = d.blockdev.unwrap().cache.unwrap();
    assert_eq!((cache.direct, cache.no_flush), (Some(false), Some(true)));

    let d = g
        .drive_new(
            &format!("file={path},format=raw,readonly=on,cache=writethrough"),
            BlockInterfaceType::Ide,
        )
        .unwrap();
    assert_eq!((d.id.as_str(), d.read_only, d.write_cache), ("ide0-hd0", true, false));
    let blk = g.backend("ide0-hd0").unwrap();
    assert!(blk.is_read_only() && !blk.enable_write_cache());
    let e = blk.set_perm(RW, SHARED).unwrap_err();
    assert_eq!(e.message(), "Block node is read-only");

    // Options for the protocol node and the format node.
    let d = g
        .drive_new(
            &format!("file={path},format=raw,if=none,id=w,offset=512,size=1024,file.locking=off"),
            BlockInterfaceType::Ide,
        )
        .unwrap();
    assert_eq!(g.node(d.node_name.as_deref().unwrap()).unwrap().size, 1024);
    assert_eq!(g.backend("w").unwrap().getlength().unwrap(), 1024);

    // Errors from the layers below come through.
    let missing = dir.join("nope.img");
    let missing = missing.to_str().unwrap();
    let e =
        g.drive_new(&format!("file={missing},format=raw"), BlockInterfaceType::Ide).unwrap_err();
    assert_eq!(e.message(), format!("Could not open '{missing}': No such file or directory"));
    assert!(g.backend("ide0-hd1").is_none());

    let qcow2 = dir.join("b.qcow2");
    fs::write(&qcow2, b"QFI\xfb\0\0\0\x03").unwrap();
    let e = g
        .drive_new(&format!("file={},if=none", qcow2.to_str().unwrap()), BlockInterfaceType::Ide)
        .unwrap_err();
    assert_eq!(e.message(), "Driver 'qcow2' is not supported yet");

    // The drive table.
    assert_eq!(g.drive_get(BlockInterfaceType::Ide, 0, 0).unwrap().id, "ide0-hd0");
    assert_eq!(g.drive_get_by_index(BlockInterfaceType::Virtio, 0).unwrap().id, "virtio0");
    g.drive_del("w").unwrap();
    assert!(g.backend("w").is_none());
    let names: Vec<_> = g.drives().into_iter().map(|d| d.id).collect();
    assert_eq!(names, ["d0", "virtio0", "ide0-hd0"]);
    let _: Option<Arc<BlockBackend>> = g.backend("d0");
}
