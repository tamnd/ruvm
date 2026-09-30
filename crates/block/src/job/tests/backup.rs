// SPDX-License-Identifier: GPL-2.0-or-later

//! The backup cases of iotests 055, 056, 124, 257 and 283 and the fleecing of iotests 222
//! on test images: `sync=full`, `top`, `none` and `bitmap` with the bitmap modes, a target
//! error, `drive-backup` to a new raw file, a `copy-before-write` node with a
//! `snapshot-access` node over it, and the argument errors.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use ruvm_qapi::types::{
    BackupPerf, BitmapSyncMode, BlockDirtyBitmapAdd, BlockdevBackup, BlockdevOnError, DriveBackup,
    MirrorSyncMode,
};

use super::cow::{CowState, cow_node};
use super::stream::{chain, read, register, wait_gone, write};
use super::*;
use crate::node::Node;

/// Backup clusters are 64 KiB: the test images give no cluster size.
const C: u64 = 64 * 1024;

fn arg(device: &str, target: &str, id: &str, sync: MirrorSyncMode) -> BlockdevBackup {
    BlockdevBackup {
        device: device.into(),
        target: target.into(),
        job_id: Some(id.into()),
        sync,
        ..Default::default()
    }
}

fn target(g: &BlockGraph, name: &str, size: u64) -> (Arc<Node>, Arc<CowState>) {
    let (n, st) = cow_node(name, size);
    register(g, &n);
    (n, st)
}

fn completed(id: &str) -> Option<Option<String>> {
    take_events(id).into_iter().find_map(|e| match e {
        BlockEvent::BlockJobCompleted { error, .. } => Some(error),
        _ => None,
    })
}

fn no_filter(g: &BlockGraph) -> bool {
    g.nodes().iter().all(|n| n.driver != "copy-before-write")
}

fn bitmap(g: &BlockGraph, node: &str, name: &str) -> Arc<crate::bitmap::DirtyBitmap> {
    g.dirty_bitmap_add(&BlockDirtyBitmapAdd {
        node: node.into(),
        name: name.into(),
        granularity: Some(C as u32),
        ..Default::default()
    })
    .unwrap()
}

#[test]
fn backup_full() {
    let _s = serial();
    let g = BlockGraph::new();
    let c = chain(&g, &["bf-base", "bf-top"], 4 * C);
    write(&c[0].1, 0, 0x11, 4096);
    write(&c[1].1, 2 * C, 0x22, 4096);
    let (t, ts) = target(&g, "bf-target", 4 * C);
    g.blockdev_backup(&arg("bf-top", "bf-target", "bf1", MirrorSyncMode::Full)).unwrap();
    wait_gone(&g, "bf1");
    assert_eq!(completed("bf1"), Some(None));
    assert!(no_filter(&g));
    assert_eq!(read(&t, 0), 0x11);
    assert_eq!(read(&t, 2 * C), 0x22);
    assert!(ts.has(0) && ts.has(C) && ts.has(3 * C), "full copies every cluster");
    // The source keeps its place in the graph.
    assert!(c[1].0.parents().iter().all(|p| p.child_name != "file"));
}

#[test]
fn backup_top() {
    let _s = serial();
    let g = BlockGraph::new();
    let c = chain(&g, &["bt-base", "bt-top"], 4 * C);
    write(&c[0].1, 0, 0x11, 4096);
    write(&c[1].1, 2 * C, 0x22, 4096);
    let (t, ts) = target(&g, "bt-target", 4 * C);
    g.blockdev_backup(&arg("bt-top", "bt-target", "bt1", MirrorSyncMode::Top)).unwrap();
    wait_gone(&g, "bt1");
    assert_eq!(completed("bt1"), Some(None));
    // Only the cluster allocated in the top image is copied.
    assert!(ts.has(2 * C) && !ts.has(0) && !ts.has(C));
    assert_eq!(read(&t, 2 * C), 0x22);
}

#[test]
fn backup_none_copies_before_write() {
    let _s = serial();
    let g = BlockGraph::new();
    let (_src, ss) = target(&g, "bn-src", 4 * C);
    write(&ss, 0, 0x11, 4 * C);
    let (t, ts) = target(&g, "bn-target", 4 * C);
    let mut a = arg("bn-src", "bn-target", "bn1", MirrorSyncMode::None);
    a.filter_node_name = Some("bn-filter".into());
    g.blockdev_backup(&a).unwrap();
    assert!(!ts.has(0), "sync=none copies nothing by itself");

    // A write to the source copies the old data of its cluster to the target first.
    let filter = g.find_node("bn-filter").unwrap();
    filter.pwrite(C + 100, &[0x55; 100]).unwrap();
    drop(filter);
    assert!(ts.has(C) && !ts.has(0) && !ts.has(2 * C));
    assert_eq!(read(&t, C + 100), 0x11);
    assert_eq!(ss.data.lock().unwrap()[(C + 100) as usize], 0x55);

    g.block_job_cancel("bn1", None).unwrap();
    wait_gone(&g, "bn1");
    let evs = take_events("bn1");
    assert!(evs.iter().any(|e| matches!(e, BlockEvent::BlockJobCancelled { .. })), "{evs:?}");
    assert!(no_filter(&g));
}

#[test]
fn backup_bitmap_modes() {
    let _s = serial();
    let g = BlockGraph::new();
    let (src, ss) = target(&g, "bb-src", 4 * C);
    write(&ss, 0, 0x11, 4 * C);
    let bm = bitmap(&g, "bb-src", "bb-map");
    bm.set_range(C, C);

    // on-success: the bitmap is empty after a successful backup.
    let (_t, ts) = target(&g, "bb-target", 4 * C);
    let mut a = arg("bb-src", "bb-target", "bb1", MirrorSyncMode::Bitmap);
    a.bitmap = Some("bb-map".into());
    a.bitmap_mode = Some(BitmapSyncMode::OnSuccess);
    g.blockdev_backup(&a).unwrap();
    wait_gone(&g, "bb1");
    assert_eq!(completed("bb1"), Some(None));
    assert!(ts.has(C) && !ts.has(0) && !ts.has(2 * C));
    let bm = src.find_dirty_bitmap("bb-map").unwrap();
    assert_eq!(bm.count(), 0);

    // never: the bitmap keeps its bits.
    bm.set_range(2 * C, C);
    let (_t2, ts2) = target(&g, "bb-target2", 4 * C);
    let mut a = arg("bb-src", "bb-target2", "bb2", MirrorSyncMode::Bitmap);
    a.bitmap = Some("bb-map".into());
    a.bitmap_mode = Some(BitmapSyncMode::Never);
    g.blockdev_backup(&a).unwrap();
    wait_gone(&g, "bb2");
    assert_eq!(completed("bb2"), Some(None));
    assert!(ts2.has(2 * C) && !ts2.has(C));
    let bm = src.find_dirty_bitmap("bb-map").unwrap();
    assert_eq!(bm.count(), C);

    // incremental is bitmap with on-success.
    let (_t3, ts3) = target(&g, "bb-target3", 4 * C);
    let mut a = arg("bb-src", "bb-target3", "bb3", MirrorSyncMode::Incremental);
    a.bitmap = Some("bb-map".into());
    g.blockdev_backup(&a).unwrap();
    wait_gone(&g, "bb3");
    assert_eq!(completed("bb3"), Some(None));
    assert!(ts3.has(2 * C) && !ts3.has(C));
    assert_eq!(src.find_dirty_bitmap("bb-map").unwrap().count(), 0);
}

#[test]
fn backup_target_error() {
    let _s = serial();
    let g = BlockGraph::new();
    let (src, _) = target(&g, "be-src", 4 * C);
    let bm = bitmap(&g, "be-src", "be-map");
    bm.set_range(0, 2 * C);
    let (_t, ts) = target(&g, "be-target", 4 * C);
    ts.fail_write.store(libc::EIO, Ordering::SeqCst);

    // always: the bitmap is synced even on failure, and keeps what was not copied.
    let mut a = arg("be-src", "be-target", "be1", MirrorSyncMode::Full);
    a.bitmap = Some("be-map".into());
    a.bitmap_mode = Some(BitmapSyncMode::Always);
    a.on_target_error = Some(BlockdevOnError::Report);
    g.blockdev_backup(&a).unwrap();
    wait_gone(&g, "be1");
    let err = completed("be1").expect("the job completed");
    assert_eq!(err.as_deref(), Some("Input/output error"));
    assert!(no_filter(&g));
    let bm = src.find_dirty_bitmap("be-map").unwrap();
    assert_eq!(bm.count(), 4 * C, "nothing was copied, so every cluster stays dirty");

    // on-success: a failure leaves the bitmap as it was.
    bm.reset_range(2 * C, 2 * C);
    let mut a = arg("be-src", "be-target", "be2", MirrorSyncMode::Bitmap);
    a.bitmap = Some("be-map".into());
    a.bitmap_mode = Some(BitmapSyncMode::OnSuccess);
    g.blockdev_backup(&a).unwrap();
    wait_gone(&g, "be2");
    assert!(completed("be2").expect("the job completed").is_some());
    assert_eq!(src.find_dirty_bitmap("be-map").unwrap().count(), 2 * C);
}

#[test]
fn backup_errors() {
    let _s = serial();
    let g = BlockGraph::new();
    let (src, _) = target(&g, "bx-src", 4 * C);
    let _t = target(&g, "bx-target", 4 * C);
    let _small = target(&g, "bx-small", 2 * C);
    let _bm = bitmap(&g, "bx-src", "bx-map");
    let e = |a: BlockdevBackup| g.blockdev_backup(&a).unwrap_err().message().to_string();
    let a = |sync| arg("bx-src", "bx-target", "bx", sync);

    assert_eq!(
        e(arg("bx-src", "bx-src", "bx", MirrorSyncMode::Full)),
        "Source and target cannot be the same"
    );
    assert_eq!(
        e(arg("bx-src", "bx-small", "bx", MirrorSyncMode::Full)),
        "Source and target image have different sizes"
    );
    assert_eq!(
        e(a(MirrorSyncMode::Bitmap)),
        "must provide a valid bitmap name for 'bitmap' sync mode"
    );
    assert_eq!(
        e(a(MirrorSyncMode::Incremental)),
        "must provide a valid bitmap name for 'incremental' sync mode"
    );
    let mut b = a(MirrorSyncMode::Incremental);
    b.bitmap = Some("bx-map".into());
    b.bitmap_mode = Some(BitmapSyncMode::Always);
    assert_eq!(e(b), "Bitmap sync mode must be 'on-success' when using sync mode 'incremental'");
    let mut b = a(MirrorSyncMode::Full);
    b.bitmap = Some("nope".into());
    assert_eq!(e(b), "Bitmap 'nope' could not be found");
    let mut b = a(MirrorSyncMode::Full);
    b.bitmap = Some("bx-map".into());
    assert_eq!(e(b), "Bitmap sync mode must be given when providing a bitmap");
    let mut b = a(MirrorSyncMode::None);
    b.bitmap = Some("bx-map".into());
    b.bitmap_mode = Some(BitmapSyncMode::Always);
    assert_eq!(e(b), "sync mode 'none' does not produce meaningful bitmap outputs");
    let mut b = a(MirrorSyncMode::Full);
    b.bitmap = Some("bx-map".into());
    b.bitmap_mode = Some(BitmapSyncMode::Never);
    assert_eq!(
        e(b),
        "Bitmap sync mode 'never' has no meaningful effect when combined with sync mode 'full'"
    );
    let mut b = a(MirrorSyncMode::Full);
    b.bitmap_mode = Some(BitmapSyncMode::Always);
    assert_eq!(e(b), "Cannot specify bitmap sync mode without a bitmap");
    let mut b = a(MirrorSyncMode::Full);
    b.x_perf = Some(BackupPerf { max_workers: Some(0), ..Default::default() });
    assert_eq!(e(b), "max-workers must be between 1 and 2147483647");
    let mut b = a(MirrorSyncMode::Full);
    b.x_perf = Some(BackupPerf { max_chunk: Some(-1), ..Default::default() });
    assert_eq!(e(b), "max-chunk must be zero (which means no limit) or positive");
    let mut b = a(MirrorSyncMode::Full);
    b.x_perf = Some(BackupPerf { max_chunk: Some(4096), ..Default::default() });
    assert_eq!(e(b), "Required max-chunk (4096) is less than backup cluster size (65536)");
    let mut b = a(MirrorSyncMode::Full);
    b.x_perf = Some(BackupPerf { min_cluster_size: Some(3 * C), ..Default::default() });
    assert_eq!(
        e(b),
        "Could not create node: Cannot create block-copy-state: min-cluster-size needs to be \
         a power of 2"
    );
    let mut b = a(MirrorSyncMode::Full);
    b.compress = Some(true);
    assert_eq!(e(b), "Compression is not supported for this drive ");
    assert!(no_filter(&g));
    // The failures left the bitmap usable.
    src.find_dirty_bitmap("bx-map").unwrap().check(crate::bitmap::BDRV_BITMAP_DEFAULT).unwrap();

    // A job id in use.
    let mut b = a(MirrorSyncMode::None);
    g.blockdev_backup(&b).unwrap();
    b.target = "bx-target".into();
    let msg = e(b);
    assert!(msg.contains("bx"), "{msg}");
    g.block_job_cancel("bx", None).unwrap();
    wait_gone(&g, "bx");
    take_events("bx");
    assert!(no_filter(&g));
}

#[test]
fn backup_speed_and_pause() {
    let _s = serial();
    let g = BlockGraph::new();
    // Chunks are 1 MiB, so at 1 MiB/s the job sleeps a second after the first one.
    const M: u64 = 1024 * 1024;
    let (_src, ss) = target(&g, "bs-src", 8 * M);
    write(&ss, 0, 0x33, 8 * M);
    let (_t, ts) = target(&g, "bs-target", 8 * M);
    let mut a = arg("bs-src", "bs-target", "bs1", MirrorSyncMode::Full);
    a.speed = Some(M as i64);
    g.blockdev_backup(&a).unwrap();
    g.job_pause("bs1").unwrap();
    wait_for("paused", || {
        g.query_jobs().iter().any(|j| j.id == "bs1" && j.status == JobStatus::Paused)
    });
    // No limit any more: the job finishes quickly once resumed.
    g.block_job_set_speed("bs1", 0).unwrap();
    g.job_resume("bs1").unwrap();
    wait_gone(&g, "bs1");
    assert_eq!(completed("bs1"), Some(None));
    assert!((0..128).all(|i| ts.has(i * C)));
}

#[test]
fn drive_backup_to_new_file() {
    let _s = serial();
    let g = BlockGraph::new();
    let (_src, ss) = target(&g, "db-src", 4 * C);
    write(&ss, C, 0x5a, 4096);
    let dir = std::env::temp_dir().join(format!("ruvm-backup-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("db-target.raw");
    let a = DriveBackup {
        device: "db-src".into(),
        target: path.to_string_lossy().into_owned(),
        format: Some("raw".into()),
        job_id: Some("db1".into()),
        sync: MirrorSyncMode::Full,
        ..Default::default()
    };
    g.drive_backup(&a).unwrap();
    wait_gone(&g, "db1");
    assert_eq!(completed("db1"), Some(None));
    let data = std::fs::read(&path).unwrap();
    assert_eq!(data.len(), (4 * C) as usize);
    assert_eq!(data[(C + 10) as usize], 0x5a);
    assert_eq!(data[0], 0);

    let mut b = a.clone();
    b.device = "nope".into();
    let msg = g.drive_backup(&b).unwrap_err().message().to_string();
    assert_eq!(msg, "Cannot find device='nope' nor node-name='nope'");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fleecing_snapshot_access() {
    let _s = serial();
    let g = BlockGraph::new();
    let (src, ss) = target(&g, "fl-src", 4 * C);
    write(&ss, 0, 0x11, 4 * C);
    let (_tmp, ts) = target(&g, "fl-tmp", 4 * C);
    add(
        &g,
        r#"{"driver": "copy-before-write", "node-name": "fl-cbw", "file": "fl-src", "target": "fl-tmp"}"#,
    );
    add(
        &g,
        r#"{"driver": "snapshot-access", "node-name": "fl-acc", "file": "fl-cbw",
            "discard": "unmap"}"#,
    );
    let cbw = g.find_node("fl-cbw").unwrap();
    let acc = g.find_node("fl-acc").unwrap();

    // The snapshot reads from the source until the guest writes there.
    assert_eq!(read(&acc, C), 0x11);
    cbw.pwrite(C, &[0x22; 512]).unwrap();
    assert_eq!(read(&src, C), 0x22);
    assert!(ts.has(C));
    assert_eq!(read(&acc, C), 0x11);
    assert_eq!(read(&acc, 3 * C), 0x11);
    let st = acc.block_status(C, C).unwrap();
    assert!(st.ret & crate::node::BDRV_BLOCK_ALLOCATED != 0, "{st:?}");

    // The snapshot cannot be written.
    let e = acc.pwrite(0, &[0; 512]).unwrap_err();
    assert_eq!(e.raw_os_error(), Some(libc::ENOTSUP));

    // A discarded range cannot be read any more, and is not copied on write.
    acc.pdiscard(2 * C, C).unwrap();
    let mut b = [0u8; 512];
    assert_eq!(acc.pread(2 * C, &mut b).unwrap_err().raw_os_error(), Some(libc::EACCES));
    cbw.pwrite(2 * C, &[0x33; 512]).unwrap();
    assert!(!ts.has(2 * C));

    // Snapshot reads of a node that is not copy-before-write fail.
    let mut b = [0u8; 512];
    let e = crate::filter::copy_before_write::preadv_snapshot(&src, 0, &mut b).unwrap_err();
    assert_eq!(e.raw_os_error(), Some(libc::ENOTSUP));
    drop((cbw, acc));
    g.blockdev_del("fl-acc").unwrap();
    g.blockdev_del("fl-cbw").unwrap();
}

#[test]
fn cbw_on_error() {
    let _s = serial();
    let g = BlockGraph::new();
    let (_src, ss) = target(&g, "ce-src", 2 * C);
    write(&ss, 0, 0x11, 2 * C);
    let (_tmp, ts) = target(&g, "ce-tmp", 2 * C);
    ts.fail_write.store(libc::ENOSPC, Ordering::SeqCst);

    // break-guest-write: the guest write fails and the source is unchanged.
    add(
        &g,
        r#"{"driver": "copy-before-write", "node-name": "ce-cbw", "file": "ce-src", "target": "ce-tmp"}"#,
    );
    let cbw = g.find_node("ce-cbw").unwrap();
    let e = cbw.pwrite(0, &[0x22; 512]).unwrap_err();
    assert_eq!(e.raw_os_error(), Some(libc::ENOSPC));
    assert_eq!(ss.data.lock().unwrap()[0], 0x11);
    drop(cbw);
    g.blockdev_del("ce-cbw").unwrap();

    // break-snapshot: the guest write succeeds, and the snapshot is gone.
    add(
        &g,
        r#"{"driver": "copy-before-write", "node-name": "ce-cbw2", "file": "ce-src",
            "target": "ce-tmp", "on-cbw-error": "break-snapshot", "cbw-timeout": 5}"#,
    );
    add(&g, r#"{"driver": "snapshot-access", "node-name": "ce-acc", "file": "ce-cbw2"}"#);
    let cbw = g.find_node("ce-cbw2").unwrap();
    let acc = g.find_node("ce-acc").unwrap();
    assert_eq!(read(&acc, C), 0x11);
    cbw.pwrite(0, &[0x22; 512]).unwrap();
    assert_eq!(ss.data.lock().unwrap()[0], 0x22);
    let mut b = [0u8; 512];
    assert_eq!(acc.pread(C, &mut b).unwrap_err().raw_os_error(), Some(libc::EACCES));
    // Later writes go through without a copy.
    ts.fail_write.store(0, Ordering::SeqCst);
    cbw.pwrite(C, &[0x33; 512]).unwrap();
    assert!(!ts.has(C));
    drop((cbw, acc));
    g.blockdev_del("ce-acc").unwrap();
    g.blockdev_del("ce-cbw2").unwrap();

    // Options that cannot work.
    let bad = |s: &str| {
        let mut v = QObjectInputVisitor::new(json::from_str(s).unwrap());
        let mut o = BlockdevOptions::default();
        BlockdevOptions::visit(&mut v, None, &mut o).unwrap();
        g.blockdev_add(o).unwrap_err().message().to_string()
    };
    assert_eq!(
        bad(r#"{"driver": "copy-before-write", "node-name": "ce-x", "file": "ce-src",
               "target": "ce-tmp", "bitmap": {"node": "ce-src", "name": "nope"}}"#),
        "Dirty bitmap 'nope' not found"
    );
    assert_eq!(
        bad(r#"{"driver": "copy-before-write", "node-name": "ce-x", "file": "ce-src",
               "target": "ce-tmp", "min-cluster-size": 100000}"#),
        "Cannot create block-copy-state: min-cluster-size needs to be a power of 2"
    );
}
