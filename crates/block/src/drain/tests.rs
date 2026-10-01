// SPDX-License-Identifier: GPL-2.0-or-later

//! A port of the parts of tests/unit/test-bdrv-drain.c that apply without coroutines,
//! iothreads and block jobs: driver callbacks, quiesce counters, nesting, drain-all with
//! nodes coming and going, and appending to a drained node.
//!
//! QEMU's `blk_aio_preadv()` with a request that sleeps is a read through the backend on
//! another thread here, which the driver holds up until the drain has started. Every test
//! holds [`DrainAllOwner`] so that the drain-all of a test running next to it does not change
//! the counters it checks.

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::time::Duration;

use super::{DrainAllOwner, drain_all_begin, drain_all_end};
use crate::backend::BlockBackend;
use crate::drivers::{DriverDef, OpenArgs};
use crate::node::{Driver, Node, NodeFlags, NodeMeta, NodeSpec};
use crate::perm::BLK_PERM_ALL;
use ruvm_base::Result;
use ruvm_qapi::types::BlockdevOptionsU;

fn no_open(_: &mut OpenArgs<'_>, _: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    unreachable!("test drivers are not opened from options")
}

/// `bdrv_test`.
static BDRV_TEST: DriverDef = DriverDef::format("test", no_open).with_backing();

/// `BDRVTestState`.
#[derive(Default)]
struct TestState {
    drain_count: AtomicI32,
    /// A read has started.
    read_started: AtomicBool,
    /// A read has finished in the driver. The drain waits for this, not for the reader's
    /// thread to come back from the call.
    read_done: AtomicBool,
}

struct TestDriver(Arc<TestState>);

impl Driver for TestDriver {
    fn pread(&self, _: &Node, _: u64, buf: &mut [u8]) -> io::Result<()> {
        self.0.read_started.store(true, Ordering::SeqCst);
        // Stay until the drain polls for this request.
        std::thread::sleep(Duration::from_millis(100));
        buf.fill(0);
        self.0.read_done.store(true, Ordering::SeqCst);
        Ok(())
    }

    fn pwrite(&self, _: &Node, _: u64, _: &[u8]) -> io::Result<()> {
        Ok(())
    }

    fn getlength(&self, _: &Node) -> io::Result<u64> {
        Ok(65536)
    }

    fn drain_begin(&self, _: &Node) {
        self.0.drain_count.fetch_add(1, Ordering::SeqCst);
    }

    fn drain_end(&self, _: &Node) {
        self.0.drain_count.fetch_sub(1, Ordering::SeqCst);
    }
}

/// `bdrv_new_open_driver(&bdrv_test, name, flags)`.
fn test_node(name: &str, read_only: bool) -> (Arc<Node>, Arc<TestState>) {
    let s = Arc::new(TestState::default());
    let node = Node::build(NodeSpec {
        name: name.to_string(),
        driver_name: "test",
        driver: Box::new(TestDriver(s.clone())),
        def: Some(&BDRV_TEST),
        flags: NodeFlags { read_only, ..NodeFlags::default() },
        meta: NodeMeta::default(),
        children: Vec::new(),
    })
    .unwrap();
    (node, s)
}

fn count(s: &TestState) -> i32 {
    s.drain_count.load(Ordering::SeqCst)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DrainType {
    All,
    Node,
}

fn do_drain_begin(t: DrainType, bs: &Node) {
    match t {
        DrainType::All => drain_all_begin(),
        DrainType::Node => bs.drained_begin(),
    }
}

fn do_drain_end(t: DrainType, bs: &Node) {
    match t {
        DrainType::All => drain_all_end(),
        DrainType::Node => bs.drained_end(),
    }
}

struct Setup {
    blk: BlockBackend,
    bs: Arc<Node>,
    s: Arc<TestState>,
    backing: Arc<Node>,
    backing_s: Arc<TestState>,
}

/// `test_setup()`: a backend on `test-node`, which has `backing` as its backing file.
fn test_setup() -> Setup {
    let blk = BlockBackend::new_empty(None, BLK_PERM_ALL, BLK_PERM_ALL, false);
    let (bs, s) = test_node("test-node", false);
    blk.insert(bs.clone()).unwrap();
    let (backing, backing_s) = test_node("backing", true);
    bs.set_backing_hd(Some(backing.clone())).unwrap();
    Setup { blk, bs, s, backing, backing_s }
}

/// `test_drv_cb_common()`.
fn drv_cb_common(t: DrainType, recursive: bool) {
    let _owner = DrainAllOwner::acquire();
    let Setup { blk, bs, s, backing_s, .. } = test_setup();

    // A simple begin/end pair: the callbacks are called.
    assert_eq!(count(&s), 0);
    assert_eq!(count(&backing_s), 0);
    do_drain_begin(t, &bs);
    assert_eq!(count(&s), 1);
    assert_eq!(count(&backing_s), i32::from(recursive));
    do_drain_end(t, &bs);
    assert_eq!(count(&s), 0);
    assert_eq!(count(&backing_s), 0);

    // The same while a request is pending.
    std::thread::scope(|sc| {
        sc.spawn(|| {
            let mut buf = [0u8; 512];
            blk.pread(0, &mut buf).unwrap();
        });
        while !s.read_started.load(Ordering::SeqCst) {
            std::thread::yield_now();
        }
        assert!(!s.read_done.load(Ordering::SeqCst));
        assert_eq!(count(&s), 0);
        assert_eq!(count(&backing_s), 0);

        do_drain_begin(t, &bs);
        assert!(s.read_done.load(Ordering::SeqCst));
        assert_eq!(count(&s), 1);
        assert_eq!(count(&backing_s), i32::from(recursive));
        do_drain_end(t, &bs);
    });
    assert_eq!(count(&s), 0);
    assert_eq!(count(&backing_s), 0);
}

#[test]
fn drv_cb_drain_all() {
    drv_cb_common(DrainType::All, true);
}

#[test]
fn drv_cb_drain() {
    drv_cb_common(DrainType::Node, false);
}

/// `test_quiesce_common()`.
fn quiesce_common(t: DrainType, recursive: bool) {
    let _owner = DrainAllOwner::acquire();
    let Setup { blk, bs, backing, .. } = test_setup();
    assert_eq!(bs.quiesce_count(), 0);
    assert_eq!(backing.quiesce_count(), 0);

    do_drain_begin(t, &bs);
    // With drain-all, the node is drained once itself and once as the parent of the drained
    // backing node.
    assert_eq!(bs.quiesce_count(), if t == DrainType::All { 2 } else { 1 });
    assert_eq!(backing.quiesce_count(), u32::from(recursive));
    assert_eq!(blk.quiesce_counter(), 1);

    do_drain_end(t, &bs);
    assert_eq!(bs.quiesce_count(), 0);
    assert_eq!(backing.quiesce_count(), 0);
    assert_eq!(blk.quiesce_counter(), 0);
}

#[test]
fn quiesce_drain_all() {
    quiesce_common(DrainType::All, true);
}

#[test]
fn quiesce_drain() {
    quiesce_common(DrainType::Node, false);
}

/// `test_nested()`.
#[test]
fn nested() {
    let _owner = DrainAllOwner::acquire();
    let Setup { blk: _blk, bs, s, backing, backing_s } = test_setup();
    for outer in [DrainType::All, DrainType::Node] {
        for inner in [DrainType::All, DrainType::Node] {
            let backing_quiesce =
                u32::from(outer == DrainType::All) + u32::from(inner == DrainType::All);
            assert_eq!(bs.quiesce_count(), 0);
            assert_eq!(backing.quiesce_count(), 0);
            assert_eq!(count(&s), 0);
            assert_eq!(count(&backing_s), 0);

            do_drain_begin(outer, &bs);
            do_drain_begin(inner, &bs);

            assert_eq!(bs.quiesce_count(), 2 + u32::from(backing_quiesce > 0));
            assert_eq!(backing.quiesce_count(), backing_quiesce);
            assert_eq!(count(&s), 1);
            assert_eq!(count(&backing_s), i32::from(backing_quiesce > 0));

            do_drain_end(inner, &bs);
            do_drain_end(outer, &bs);

            assert_eq!(bs.quiesce_count(), 0);
            assert_eq!(backing.quiesce_count(), 0);
            assert_eq!(count(&s), 0);
            assert_eq!(count(&backing_s), 0);
        }
    }
}

/// `test_graph_change_drain_all()`: nodes made during drain-all start out drained, and
/// nodes going away during it do not upset the end.
#[test]
fn graph_change_drain_all() {
    let _owner = DrainAllOwner::acquire();
    let blk_a = BlockBackend::new_empty(None, BLK_PERM_ALL, BLK_PERM_ALL, false);
    let (bs_a, a_s) = test_node("test-node-a", false);
    blk_a.insert(bs_a.clone()).unwrap();
    assert_eq!(bs_a.quiesce_count(), 0);
    assert_eq!(count(&a_s), 0);

    drain_all_begin();
    assert_eq!(bs_a.quiesce_count(), 1);
    assert_eq!(count(&a_s), 1);

    let blk_b = BlockBackend::new_empty(None, BLK_PERM_ALL, BLK_PERM_ALL, false);
    let (bs_b, b_s) = test_node("test-node-b", false);
    blk_b.insert(bs_b.clone()).unwrap();
    assert_eq!(bs_a.quiesce_count(), 1);
    assert_eq!(bs_b.quiesce_count(), 1);
    assert_eq!(count(&a_s), 1);
    assert_eq!(count(&b_s), 1);
    // The new backend is quiesced too.
    assert_eq!(blk_b.quiesce_counter(), 1);

    drop(blk_a);
    assert_eq!(bs_a.quiesce_count(), 1);
    assert_eq!(count(&a_s), 1);
    drop(bs_a);
    assert_eq!(bs_b.quiesce_count(), 1);
    assert_eq!(count(&b_s), 1);

    drain_all_end();
    assert_eq!(bs_b.quiesce_count(), 0);
    assert_eq!(count(&b_s), 0);
    assert_eq!(blk_b.quiesce_counter(), 0);
}

/// `test_append_to_drained()`: a node appended on top of a drained node is drained too, and
/// both come out of it together.
#[test]
fn append_to_drained() {
    let _owner = DrainAllOwner::acquire();
    let blk = BlockBackend::new_empty(None, BLK_PERM_ALL, BLK_PERM_ALL, false);
    let (base, base_s) = test_node("base", false);
    blk.insert(base.clone()).unwrap();
    let (overlay, overlay_s) = test_node("overlay", false);

    base.drained_begin();
    assert_eq!(base.quiesce_count(), 1);
    assert_eq!(count(&base_s), 1);
    assert_eq!(base.io.in_flight(), 0);

    Node::append(&overlay, &base).unwrap();
    assert!(Arc::ptr_eq(&blk.root().unwrap(), &overlay));
    assert_eq!(base.io.in_flight(), 0);
    assert_eq!(overlay.io.in_flight(), 0);
    assert_eq!(base.quiesce_count(), 1);
    assert_eq!(count(&base_s), 1);
    assert_eq!(overlay.quiesce_count(), 1);
    assert_eq!(count(&overlay_s), 1);
    assert_eq!(blk.quiesce_counter(), 1);

    base.drained_end();
    assert_eq!(base.quiesce_count(), 0);
    assert_eq!(count(&base_s), 0);
    assert_eq!(overlay.quiesce_count(), 0);
    assert_eq!(count(&overlay_s), 0);
    assert_eq!(blk.quiesce_counter(), 0);
}

/// A backend's request made while it is drained waits for the end of the section.
#[test]
fn requests_wait_while_drained() {
    let _owner = DrainAllOwner::acquire();
    let Setup { blk, bs, .. } = test_setup();
    let done = AtomicBool::new(false);
    bs.drained_begin();
    std::thread::scope(|sc| {
        sc.spawn(|| {
            blk.pwrite(0, &[1; 512]).unwrap();
            done.store(true, Ordering::SeqCst);
        });
        std::thread::sleep(Duration::from_millis(50));
        assert!(!done.load(Ordering::SeqCst));
        bs.drained_end();
    });
    assert!(done.load(Ordering::SeqCst));
}
