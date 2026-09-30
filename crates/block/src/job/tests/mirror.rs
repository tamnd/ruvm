// SPDX-License-Identifier: GPL-2.0-or-later

//! The mirror cases of iotests 041, 151 and 185 on test images: `sync=full` and `sync=top`
//! with a pivot, `block-job-cancel` of a ready job, `write-blocking` through
//! `block-job-change`, a `drive-mirror` to a new raw file, and the argument errors.

use std::sync::Arc;

use ruvm_qapi::types::{
    BlockJobChangeOptions, BlockJobChangeOptionsMirror, BlockJobChangeOptionsU, BlockJobInfoU,
    BlockdevMirrorArg, DriveMirror, MirrorCopyMode, MirrorSyncMode,
};

use super::commit::take_ready;
use super::cow::{CowState, cow_node};
use super::stream::{chain, read, register, wait_gone, write};
use super::*;
use crate::node::Node;

fn arg(device: &str, target: &str, id: &str, sync: MirrorSyncMode) -> BlockdevMirrorArg {
    BlockdevMirrorArg {
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

fn completed_ok(id: &str) -> bool {
    take_events(id).iter().any(|e| matches!(e, BlockEvent::BlockJobCompleted { error: None, .. }))
}

#[test]
fn mirror_full_and_pivot() {
    let _s = serial();
    let g = BlockGraph::new();
    let c = chain(&g, &["mf-base", "mf-top"], 64 * 1024);
    write(&c[0].1, 0, 0x11, 4096);
    write(&c[1].1, 8192, 0x22, 4096);
    let (t, ts) = target(&g, "mf-target", 64 * 1024);
    let mut a = arg("mf-top", "mf-target", "mf1", MirrorSyncMode::Full);
    a.filter_node_name = Some("mf-filter".into());
    a.granularity = Some(4096);
    g.blockdev_mirror(&a).unwrap();
    wait_for("ready", || take_ready("mf1"));
    let info = g.query_block_jobs().unwrap().into_iter().find(|j| j.device == "mf1").unwrap();
    assert!(info.ready);
    assert!(matches!(info.u, BlockJobInfoU::Mirror(_)));

    // A write while the job is ready goes to the target before the pivot.
    let filter = g.find_node("mf-filter").unwrap();
    filter.pwrite(20480, &[0x33; 1024]).unwrap();
    drop(filter);

    g.block_job_complete("mf1").unwrap();
    wait_gone(&g, "mf1");
    assert!(completed_ok("mf1"));
    assert!(g.find_node("mf-filter").is_none());
    // sync=full copies the whole chain; the target keeps no backing node.
    assert!(t.backing().is_none());
    assert_eq!(read(&t, 0), 0x11);
    assert_eq!(read(&t, 8192), 0x22);
    assert_eq!(read(&t, 20480), 0x33);
    assert_eq!(read(&t, 40960), 0);
    assert!(ts.has(0) && ts.has(8192) && ts.has(20480));
}

#[test]
fn mirror_top_keeps_backing() {
    let _s = serial();
    let g = BlockGraph::new();
    let c = chain(&g, &["mt-base", "mt-top"], 64 * 1024);
    write(&c[0].1, 0, 0x11, 4096);
    write(&c[1].1, 8192, 0x22, 4096);
    let (t, ts) = target(&g, "mt-target", 64 * 1024);
    let mut a = arg("mt-top", "mt-target", "mt1", MirrorSyncMode::Top);
    a.granularity = Some(4096);
    g.blockdev_mirror(&a).unwrap();
    wait_for("ready", || take_ready("mt1"));
    g.block_job_complete("mt1").unwrap();
    wait_gone(&g, "mt1");
    assert!(completed_ok("mt1"));
    // Only what is above the base is copied.
    assert!(ts.has(8192) && !ts.has(0));
    assert_eq!(read(&t, 8192), 0x22);
}

#[test]
fn mirror_cancel_when_ready() {
    let _s = serial();
    let g = BlockGraph::new();
    let (src, ss) = target(&g, "mc-src", 32 * 1024);
    write(&ss, 0, 0x44, 8192);
    let (t, _) = target(&g, "mc-target", 32 * 1024);
    g.blockdev_mirror(&arg("mc-src", "mc-target", "mc1", MirrorSyncMode::Full)).unwrap();
    wait_for("ready", || take_ready("mc1"));
    // Cancelling a ready job completes it without a pivot, and the target is in sync.
    g.block_job_cancel("mc1", None).unwrap();
    wait_gone(&g, "mc1");
    assert!(completed_ok("mc1"));
    assert_eq!(read(&t, 4096), 0x44);
    assert_eq!(read(&src, 4096), 0x44);
    assert!(g.nodes().iter().all(|n| n.driver != "mirror_top"));
}

#[test]
fn mirror_write_blocking() {
    let _s = serial();
    let g = BlockGraph::new();
    let (_src, ss) = target(&g, "mw-src", 32 * 1024);
    write(&ss, 0, 0x44, 4096);
    let (_t, ts) = target(&g, "mw-target", 32 * 1024);
    let mut a = arg("mw-src", "mw-target", "mw1", MirrorSyncMode::Full);
    a.filter_node_name = Some("mw-filter".into());
    a.granularity = Some(512);
    g.blockdev_mirror(&a).unwrap();
    wait_for("ready", || take_ready("mw1"));

    let change = |mode| BlockJobChangeOptions {
        id: "mw1".into(),
        u: BlockJobChangeOptionsU::Mirror(BlockJobChangeOptionsMirror { copy_mode: mode }),
    };
    g.block_job_change(&change(MirrorCopyMode::WriteBlocking)).unwrap();
    let e = g.block_job_change(&change(MirrorCopyMode::Background)).unwrap_err();
    assert_eq!(e.message(), "Change to copy mode 'background' is not implemented");

    // In write-blocking mode a guest write is on the target when it completes.
    let filter = g.find_node("mw-filter").unwrap();
    filter.pwrite(16384, &[0x66; 512]).unwrap();
    filter.pwrite(17000, &[0x77; 100]).unwrap();
    drop(filter);
    assert!(ts.has(16384));
    assert_eq!(ts.data.lock().unwrap()[16384], 0x66);
    wait_for("actively-synced", || {
        g.query_block_jobs().unwrap().iter().any(|j| {
            j.device == "mw1" && matches!(j.u, BlockJobInfoU::Mirror(ref m) if m.actively_synced)
        })
    });
    g.block_job_complete("mw1").unwrap();
    wait_gone(&g, "mw1");
    assert!(completed_ok("mw1"));
    assert_eq!(ts.data.lock().unwrap()[17050], 0x77);
    assert_eq!(ts.data.lock().unwrap()[1000], 0x44);
}

#[test]
fn mirror_different_size_fails() {
    let _s = serial();
    let g = BlockGraph::new();
    let _src = target(&g, "md-src", 32 * 1024);
    let _t = target(&g, "md-target", 16 * 1024);
    g.blockdev_mirror(&arg("md-src", "md-target", "md1", MirrorSyncMode::Full)).unwrap();
    wait_gone(&g, "md1");
    let evs = take_events("md1");
    assert!(
        evs.iter().any(|e| matches!(e, BlockEvent::BlockJobCompleted { error: Some(m), .. }
            if m == "Source and target image have different sizes")),
        "{evs:?}"
    );
    assert!(g.nodes().iter().all(|n| n.driver != "mirror_top"));
}

#[test]
fn mirror_errors() {
    let _s = serial();
    let g = BlockGraph::new();
    let _src = target(&g, "mx-src", 4096);
    let _t = target(&g, "mx-target", 4096);
    let e = |a: BlockdevMirrorArg| g.blockdev_mirror(&a).unwrap_err().message().to_string();
    let a = || arg("mx-src", "mx-target", "mx", MirrorSyncMode::Full);
    assert_eq!(
        e(arg("mx-src", "mx-src", "mx", MirrorSyncMode::Full)),
        "Can't mirror node into itself"
    );
    let mut b = a();
    b.granularity = Some(256);
    assert_eq!(e(b), "Parameter 'granularity' expects a value in range [512B, 64MB]");
    let mut b = a();
    b.granularity = Some(1536);
    assert_eq!(e(b), "Parameter 'granularity' expects a power of 2");
    let mut b = a();
    b.buf_size = Some(-1);
    assert_eq!(e(b), "Invalid parameter 'buf-size'");
    assert_eq!(
        e(arg("mx-src", "mx-target", "mx", MirrorSyncMode::Bitmap)),
        "Sync mode 'bitmap' not supported"
    );
    let mut b = a();
    b.replaces = Some("nope".into());
    assert_eq!(e(b), "Failed to find node with node-name='nope'");
    let mut b = a();
    b.replaces = Some("mx-target".into());
    let msg = e(b);
    assert!(
        msg.starts_with("Cannot replace 'mx-target' by a node mirrored from 'mx-src'"),
        "{msg}"
    );
    assert!(g.nodes().iter().all(|n| n.driver != "mirror_top"));

    // Complete before ready.
    let mut b = a();
    b.speed = Some(1);
    g.blockdev_mirror(&b).unwrap();
    let msg = g.block_job_complete("mx").unwrap_err().message().to_string();
    assert_eq!(msg, "Job 'mx' in state 'running' cannot accept command verb 'complete'");
    g.block_job_cancel("mx", Some(true)).unwrap();
    wait_gone(&g, "mx");
    take_events("mx");
}

#[test]
fn drive_mirror_to_new_file() {
    let _s = serial();
    let g = BlockGraph::new();
    let (_src, ss) = target(&g, "dm-src", 64 * 1024);
    write(&ss, 4096, 0x5a, 4096);
    let dir = std::env::temp_dir().join(format!("ruvm-mirror-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("dm-target.raw");
    let path_s = path.to_string_lossy().into_owned();
    let a = DriveMirror {
        device: "dm-src".into(),
        target: path_s.clone(),
        format: Some("raw".into()),
        node_name: Some("dm-target".into()),
        job_id: Some("dm1".into()),
        sync: MirrorSyncMode::Full,
        ..Default::default()
    };
    g.drive_mirror(&a).unwrap();
    wait_for("ready", || take_ready("dm1"));
    let t = g.find_node("dm-target").unwrap();
    g.block_job_complete("dm1").unwrap();
    wait_gone(&g, "dm1");
    assert!(completed_ok("dm1"));
    assert_eq!(read(&t, 4096), 0x5a);
    drop(t);
    let data = std::fs::read(&path).unwrap();
    assert_eq!(data.len(), 64 * 1024);
    assert_eq!(data[5000], 0x5a);
    assert_eq!(data[0], 0);

    let mut b = a.clone();
    b.replaces = Some("dm-src".into());
    b.node_name = None;
    b.job_id = Some("dm2".into());
    let msg = g.drive_mirror(&b).unwrap_err().message().to_string();
    assert_eq!(msg, "a node-name must be provided when replacing a named node of the graph");
    let _ = std::fs::remove_dir_all(&dir);
}
