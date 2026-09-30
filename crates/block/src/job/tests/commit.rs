// SPDX-License-Identifier: GPL-2.0-or-later

//! The `block-commit` cases of iotests 040 on test images: an intermediate commit, an
//! active commit with `block-job-complete`, an active commit cancelled before it is ready,
//! and the argument errors.

use std::sync::Arc;

use ruvm_qapi::types::{BlockCommitArg, JobType};

use super::stream::{chain, read, wait_gone, write};
use super::*;

fn arg(device: &str, id: &str) -> BlockCommitArg {
    BlockCommitArg { device: device.into(), job_id: Some(id.into()), ..Default::default() }
}

#[test]
fn commit_intermediate() {
    let _s = serial();
    let g = BlockGraph::new();
    let c = chain(&g, &["ci-base", "ci-mid", "ci-top"], 64 * 1024);
    write(&c[0].1, 0, 0x11, 4096);
    write(&c[1].1, 0, 0x22, 1024);
    write(&c[1].1, 8192, 0x33, 4096);
    write(&c[2].1, 16384, 0x44, 512);
    let mut a = arg("ci-top", "ci1");
    a.top_node = Some("ci-mid".into());
    a.base_node = Some("ci-base".into());
    g.block_commit(&a).unwrap();
    wait_gone(&g, "ci1");
    let evs = take_events("ci1");
    assert!(evs.iter().any(|e| matches!(e, BlockEvent::BlockJobCompleted { error: None, .. })));
    // The top now sits on the base, which has the data of the middle image.
    assert!(Arc::ptr_eq(&c[2].0.backing().unwrap().node, &c[0].0));
    assert!(c[0].1.has(8192) && !c[0].1.has(16384));
    assert_eq!(read(&c[0].0, 0), 0x22);
    assert_eq!(read(&c[0].0, 2048), 0x11);
    assert_eq!(read(&c[2].0, 0), 0x22);
    assert_eq!(read(&c[2].0, 8192), 0x33);
    assert_eq!(read(&c[2].0, 16384), 0x44);
    assert!(g.nodes().iter().all(|n| n.driver != "commit_top"));
}

#[test]
fn commit_active() {
    let _s = serial();
    let g = BlockGraph::new();
    let c = chain(&g, &["ca-base", "ca-mid", "ca-top"], 64 * 1024);
    write(&c[0].1, 0, 0x11, 4096);
    write(&c[1].1, 8192, 0x22, 4096);
    write(&c[2].1, 16384, 0x33, 512);
    let mut a = arg("ca-top", "ca1");
    a.filter_node_name = Some("ca-filter".into());
    g.block_commit(&a).unwrap();
    let job = g.query_jobs().into_iter().find(|j| j.id == "ca1").unwrap();
    assert_eq!(job.type_, JobType::Commit);
    wait_for("ready", || take_ready("ca1"));

    // What the guest writes while the job is ready makes it into the base too.
    let filter = g.find_node("ca-filter").unwrap();
    filter.pwrite(32768, &[0x55; 512]).unwrap();
    assert!(c[2].1.has(32768));
    drop(filter);

    g.block_job_complete("ca1").unwrap();
    wait_gone(&g, "ca1");
    let evs = take_events("ca1");
    assert!(evs.iter().any(|e| matches!(e, BlockEvent::BlockJobCompleted { error: None, .. })));
    assert!(g.find_node("ca-filter").is_none(), "the filter is gone");
    let base = &c[0];
    assert!(base.1.has(8192) && base.1.has(16384) && base.1.has(32768));
    assert_eq!(read(&base.0, 0), 0x11);
    assert_eq!(read(&base.0, 8192), 0x22);
    assert_eq!(read(&base.0, 16384), 0x33);
    assert_eq!(read(&base.0, 32768), 0x55);
}

#[test]
fn commit_active_cancel() {
    let _s = serial();
    let g = BlockGraph::new();
    let c = chain(&g, &["cc-base", "cc-top"], 64 * 1024);
    write(&c[1].1, 0, 0x22, 4096);
    let mut a = arg("cc-top", "cc1");
    a.filter_node_name = Some("cc-filter".into());
    a.speed = Some(1);
    g.block_commit(&a).unwrap();
    g.block_job_cancel("cc1", Some(true)).unwrap();
    wait_gone(&g, "cc1");
    let evs = take_events("cc1");
    assert!(evs.iter().any(|e| matches!(e, BlockEvent::BlockJobCancelled { .. })), "{evs:?}");
    assert!(g.find_node("cc-filter").is_none());
    // Nothing changed in the chain.
    assert!(Arc::ptr_eq(&c[1].0.backing().unwrap().node, &c[0].0));
    assert_eq!(read(&c[1].0, 0), 0x22);
}

#[test]
fn commit_errors() {
    let _s = serial();
    let g = BlockGraph::new();
    let _c = chain(&g, &["cx-base", "cx-mid", "cx-top"], 4096);
    let (lone, _) = cow::cow_node("cx-lone", 4096);
    stream::register(&g, &lone);
    let e = |a: BlockCommitArg| g.block_commit(&a).unwrap_err().message().to_string();
    assert_eq!(e(arg("nope", "cx")), "Device 'nope' not found");
    let mut a = arg("cx-top", "cx");
    a.top_node = Some("cx-mid".into());
    a.top = Some("x".into());
    assert_eq!(e(a), "'top-node' and 'top' are mutually exclusive");
    let mut a = arg("cx-top", "cx");
    a.base_node = Some("cx-base".into());
    a.base = Some("x".into());
    assert_eq!(e(a), "'base-node' and 'base' are mutually exclusive");
    assert_eq!(e(arg("cx-mid", "cx")), "Need a root block node");
    let mut a = arg("cx-top", "cx");
    a.top_node = Some("cx-lone".into());
    assert_eq!(e(a), "'cx-lone' is not in this backing file chain");
    let mut a = arg("cx-top", "cx");
    a.top_node = Some("cx-mid".into());
    a.base_node = Some("cx-mid".into());
    assert_eq!(e(a), "cannot commit an image into itself");
    let mut a = arg("cx-top", "cx");
    a.base = Some("nope".into());
    assert_eq!(e(a), "Can't find 'nope' in the backing chain");
    // The bottom of a chain of one is the image itself.
    assert_eq!(e(arg("cx-lone", "cx")), "cannot commit an image into itself");
    let mut a = arg("cx-top", "cx");
    a.backing_file = Some("x".into());
    assert_eq!(e(a), "'backing-file' specified, but 'top' is the active layer");
    assert!(g.query_jobs().iter().all(|j| j.id != "cx"));
}

/// Takes the `BLOCK_JOB_READY` event of `id`, if it came.
pub(crate) fn take_ready(id: &str) -> bool {
    let evs = take_events(id);
    let ready = evs.iter().any(|e| matches!(e, BlockEvent::BlockJobReady { .. }));
    // Put the other events back for the checks that come later.
    for e in evs.iter().filter(|e| !matches!(e, BlockEvent::BlockJobReady { .. })) {
        record_event(e);
    }
    ready
}
