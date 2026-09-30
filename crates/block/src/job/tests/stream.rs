// SPDX-License-Identifier: GPL-2.0-or-later

//! The `block-stream` cases of iotests 030 on test images: whole chain, `base-node`,
//! `bottom`, the argument errors, and a read-only top.

use std::sync::Arc;

use ruvm_qapi::types::BlockStreamArg;

use super::cow::{CowState, SECTOR};
use super::*;
use crate::node::Node;

fn arg(device: &str, id: &str) -> BlockStreamArg {
    BlockStreamArg { device: device.into(), job_id: Some(id.into()), ..Default::default() }
}

#[test]
fn stream_whole_chain() {
    let _s = serial();
    let g = BlockGraph::new();
    let c = chain(&g, &["st-base", "st-mid", "st-top"], 64 * 1024);
    write(&c[0].1, 0, 0x11, 4096);
    write(&c[1].1, 8192, 0x22, 4096);
    write(&c[2].1, 16384, 0x33, 512);
    g.block_stream(&arg("st-top", "st1")).unwrap();
    wait_gone(&g, "st1");
    let top = &c[2];
    assert!(top.0.backing().is_none(), "the whole chain was streamed");
    assert!(top.1.has(0) && top.1.has(8192) && top.1.has(16384));
    assert!(!top.1.has(32768), "unallocated data stays unallocated");
    assert_eq!(read(&top.0, 0), 0x11);
    assert_eq!(read(&top.0, 8192), 0x22);
    assert_eq!(read(&top.0, 16384), 0x33);
    let evs = take_events("st1");
    assert!(evs.iter().any(|e| matches!(e, BlockEvent::BlockJobCompleted { error: None, .. })));
    // The filter is gone.
    assert!(g.nodes().iter().all(|n| n.driver != "copy-on-read"));
}

#[test]
fn stream_base_node_and_bottom() {
    let _s = serial();
    let g = BlockGraph::new();
    let c = chain(&g, &["sb-base", "sb-mid", "sb-top"], 64 * 1024);
    write(&c[0].1, 0, 0x11, 4096);
    write(&c[1].1, 8192, 0x22, 4096);
    let mut a = arg("sb-top", "st2");
    a.base_node = Some("sb-base".into());
    g.block_stream(&a).unwrap();
    wait_gone(&g, "st2");
    let top = &c[2];
    assert!(Arc::ptr_eq(&top.0.backing().unwrap().node, &c[0].0));
    assert!(top.1.has(8192) && !top.1.has(0), "only the data above the base");
    assert_eq!(read(&top.0, 0), 0x11);
    assert_eq!(read(&top.0, 8192), 0x22);

    // bottom: stream the base too
    let mut a = arg("sb-top", "st3");
    a.bottom = Some("sb-base".into());
    g.block_stream(&a).unwrap();
    wait_gone(&g, "st3");
    assert!(top.0.backing().is_none());
    assert!(top.1.has(0));
}

#[test]
fn stream_errors() {
    let _s = serial();
    let g = BlockGraph::new();
    let c = chain(&g, &["se-base", "se-mid", "se-top"], 4096);
    let e = |a: BlockStreamArg| g.block_stream(&a).unwrap_err().message().to_string();
    let mut a = arg("se-top", "se");
    a.base = Some("x".into());
    a.base_node = Some("se-base".into());
    assert_eq!(e(a), "'base' and 'base-node' cannot be specified at the same time");
    let mut a = arg("se-top", "se");
    a.base = Some("x".into());
    a.bottom = Some("se-base".into());
    assert_eq!(e(a), "'base' and 'bottom' cannot be specified at the same time");
    let mut a = arg("se-top", "se");
    a.bottom = Some("x".into());
    a.base_node = Some("se-base".into());
    assert_eq!(e(a), "'bottom' and 'base-node' cannot be specified at the same time");
    assert_eq!(e(arg("nope", "se")), "Cannot find device='nope' nor node-name='nope'");
    let mut a = arg("se-top", "se");
    a.base = Some("nope".into());
    assert_eq!(e(a), "Can't find 'nope' in the backing chain");
    let mut a = arg("se-top", "se");
    a.base_node = Some("nope".into());
    assert_eq!(e(a), "Cannot find device='' nor node-name='nope'");
    let mut a = arg("se-mid", "se");
    a.base_node = Some("se-top".into());
    assert_eq!(e(a), "Node 'se-top' is not a backing image of 'se-mid'");
    let mut a = arg("se-top", "se");
    a.base_node = Some("se-top".into());
    assert_eq!(e(a), "Node 'se-top' is not a backing image of 'se-top'");
    let mut a = arg("se-mid", "se");
    a.bottom = Some("se-top".into());
    assert_eq!(e(a), "Node 'se-top' is not in a chain starting from 'se-mid'");
    let mut a = arg("se-top", "se");
    a.backing_file = Some("x".into());
    assert_eq!(e(a), "backing file specified, but streaming the entire chain");
    let mut a = arg("se-top", "0se");
    a.base_node = Some("se-base".into());
    assert_eq!(e(a), "Invalid job ID '0se'");
    // A failed start leaves the graph as it was.
    assert!(Arc::ptr_eq(&c[2].0.backing().unwrap().node, &c[1].0));
    assert!(g.nodes().iter().all(|n| n.driver != "copy-on-read"));

    // A job on the chain blocks another stream.
    let mut a = arg("se-top", "se1");
    a.speed = Some(1);
    a.auto_finalize = Some(false);
    g.block_stream(&a).unwrap();
    let a = arg("se-top", "se2");
    let msg = e(a);
    assert!(
        msg.starts_with("Node 'se-top' is busy: block device is in use by block job: stream"),
        "{msg}"
    );
    g.job_cancel("se1").unwrap();
    wait_for("cancel", || g.query_jobs().iter().all(|j| j.id != "se1"));
    assert!(Arc::ptr_eq(&c[2].0.backing().unwrap().node, &c[1].0));
    take_events("se1");
}

/// A backing chain of test images, bottom first, in the graph.
pub(crate) fn chain(g: &BlockGraph, names: &[&str], size: u64) -> Vec<(Arc<Node>, Arc<CowState>)> {
    let mut out: Vec<(Arc<Node>, Arc<CowState>)> = Vec::new();
    for n in names {
        let (node, st) = cow::cow_node(n, size);
        if let Some(below) = out.last() {
            node.set_backing_hd(Some(below.0.clone())).unwrap();
            node.meta.lock().unwrap().backing_file = below.0.name.clone();
        }
        register(g, &node);
        out.push((node, st));
    }
    out
}

/// Puts a node made by a test in the name table.
pub(crate) fn register(g: &BlockGraph, node: &Arc<Node>) {
    g.commit(
        node.clone(),
        crate::graph::Pending { nodes: vec![node.clone()], warnings: Vec::new() },
        crate::graph::OpenCtx::default(),
    )
    .unwrap();
}

/// Writes `len` bytes of `byte` straight into the image.
pub(crate) fn write(st: &CowState, off: u64, byte: u8, len: u64) {
    st.data.lock().unwrap()[off as usize..(off + len) as usize].fill(byte);
    let mut a = st.allocated.lock().unwrap();
    for i in off / SECTOR..(off + len).div_ceil(SECTOR) {
        a[i as usize] = true;
    }
}

/// The byte at `off` as the node reads it.
pub(crate) fn read(n: &Node, off: u64) -> u8 {
    let mut b = [0u8; 512];
    n.pread(off - off % 512, &mut b).unwrap();
    b[(off % 512) as usize]
}

/// Waits until the job `id` is gone.
pub(crate) fn wait_gone(g: &BlockGraph, id: &str) {
    wait_for(id, || g.query_jobs().iter().all(|j| j.id != id));
}
