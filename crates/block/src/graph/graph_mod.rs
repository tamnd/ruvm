// SPDX-License-Identifier: GPL-2.0-or-later

//! Graph changes: a port of tests/unit/test-bdrv-graph-mod.c, plus checks that a failed
//! change leaves the graph as it was.
//!
//! The QEMU test drivers (`pass-through`, `no-perm`, `exclusive-writer` and
//! `write-to-selected`) are defined here with the same permission callbacks. QEMU's
//! `bdrv_new_open_driver()` becomes [`Node::build`] with the driver's [`DriverDef`], and
//! `blk_new()` plus `blk_insert_bs()` is [`BlockBackend::new_empty`] plus `insert`.

use std::io;
use std::sync::{Arc, Mutex};

use crate::backend::BlockBackend;
use crate::drivers::{DriverDef, OpenArgs};
use crate::node::{
    BDRV_CHILD_DATA, BDRV_CHILD_FILTERED, BDRV_CHILD_PRIMARY, Driver, Node, NodeFlags, NodeMeta,
    NodeSpec,
};
use crate::perm::{BLK_PERM_ALL, BLK_PERM_CONSISTENT_READ, BLK_PERM_WRITE, PermCtx, default_perms};
use ruvm_base::Result;
use ruvm_qapi::types::BlockdevOptionsU;

const SIZE: u64 = 1 << 20;

fn no_open(_: &mut OpenArgs<'_>, _: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    unreachable!("test drivers are not opened from options")
}

static PASS_THROUGH: DriverDef = DriverDef::filter("pass-through", no_open).with_filtered_backing();
static NO_PERM: DriverDef = DriverDef::format("no-perm", no_open).with_backing();
static EXCLUSIVE_WRITER: DriverDef =
    DriverDef::filter("exclusive-writer", no_open).with_filtered_backing();
static WRITE_TO_SELECTED: DriverDef = DriverDef::format("write-to-selected", no_open);

#[derive(Clone, Copy)]
enum Perms {
    /// `bdrv_default_perms`.
    Default,
    /// `no_perm_default_perms`.
    None,
    /// `exclusive_write_perms`.
    ExclusiveWrite,
}

/// A driver with no data of its own and one of the permission policies of the QEMU test.
struct TestDriver {
    perms: Perms,
    /// For `write-to-selected`: the node name of the selected child (QEMU compares the
    /// edge name; the edges `first` and `second` lead to `fl1` and `fl2`).
    selected: Option<Arc<Mutex<Option<String>>>>,
}

impl Driver for TestDriver {
    fn pread(&self, _: &Node, _: u64, buf: &mut [u8]) -> io::Result<()> {
        buf.fill(0);
        Ok(())
    }

    fn pwrite(&self, _: &Node, _: u64, _: &[u8]) -> io::Result<()> {
        Ok(())
    }

    fn getlength(&self, _: &Node) -> io::Result<u64> {
        Ok(SIZE)
    }

    fn child_perm_for(&self, ctx: &PermCtx<'_>, perm: u64, shared: u64) -> (u64, u64) {
        if let Some(sel) = &self.selected {
            return if sel.lock().unwrap().as_deref() == Some(ctx.child.name.as_str()) {
                (BLK_PERM_WRITE, BLK_PERM_ALL & !BLK_PERM_WRITE)
            } else {
                (0, BLK_PERM_ALL)
            };
        }
        match self.perms {
            Perms::Default => default_perms(ctx, perm, shared),
            Perms::None => (0, BLK_PERM_ALL),
            Perms::ExclusiveWrite => (BLK_PERM_WRITE, BLK_PERM_ALL & !BLK_PERM_WRITE),
        }
    }
}

/// `bdrv_new_open_driver(drv, name, BDRV_O_RDWR)`.
fn node(def: &'static DriverDef, name: &str, driver: TestDriver) -> Arc<Node> {
    Node::build(NodeSpec {
        name: name.to_string(),
        driver_name: def.format_name,
        driver: Box::new(driver),
        def: Some(def),
        flags: NodeFlags::default(),
        meta: NodeMeta::default(),
        children: Vec::new(),
    })
    .unwrap()
}

fn no_perm_node(name: &str) -> Arc<Node> {
    node(&NO_PERM, name, TestDriver { perms: Perms::None, selected: None })
}

fn pass_through_node(name: &str) -> Arc<Node> {
    node(&PASS_THROUGH, name, TestDriver { perms: Perms::Default, selected: None })
}

fn exclusive_writer_node(name: &str) -> Arc<Node> {
    node(&EXCLUSIVE_WRITER, name, TestDriver { perms: Perms::ExclusiveWrite, selected: None })
}

const FILTERED_PRIMARY: u32 = BDRV_CHILD_FILTERED | BDRV_CHILD_PRIMARY;

fn child_of(parent: &Node, name: &str) -> Arc<Node> {
    parent.child(name).unwrap().node
}

/// test_update_perm_tree: appending a filter that already has the node as a data child would
/// give the node two writers that do not share. It must fail, and the graph must stay as it
/// was.
#[test]
fn update_perm_tree() {
    let root = BlockBackend::new_empty(
        None,
        BLK_PERM_WRITE | BLK_PERM_CONSISTENT_READ,
        BLK_PERM_ALL & !BLK_PERM_WRITE,
        false,
    );
    let bs = no_perm_node("node");
    let filter = pass_through_node("filter");
    root.insert(bs.clone()).unwrap();
    filter.attach_child("child", bs.clone(), BDRV_CHILD_DATA).unwrap();

    let e = Node::append(&filter, &bs).unwrap_err();
    assert_eq!(
        e.message(),
        "Permission conflict on node 'node': permissions 'write' are both required by node \
         'filter' (uses node 'node' as 'child' child) and unshared by node 'filter' (uses node \
         'node' as 'backing' child)."
    );

    // Rolled back: the backend is on the node again, and the filter has only its old child.
    assert!(Arc::ptr_eq(&root.root().unwrap(), &bs));
    assert!(filter.backing().is_none());
    assert_eq!(filter.children().len(), 1);
    assert_eq!(bs.parent_count(), 2);
    assert_eq!(filter.parent_count(), 0);
    assert_eq!(bs.cumulative_perm().0, BLK_PERM_WRITE | BLK_PERM_CONSISTENT_READ);
}

/// test_should_update_child: appending must not move the backing link of a node that the new
/// top reaches through its own subtree, which would close a loop.
#[test]
fn should_update_child() {
    let root = BlockBackend::new_empty(None, 0, BLK_PERM_ALL, false);
    let bs = no_perm_node("node");
    let filter = no_perm_node("filter");
    let target = no_perm_node("target");
    root.insert(bs.clone()).unwrap();

    target.set_backing_hd(Some(bs.clone())).unwrap();
    assert!(Arc::ptr_eq(&target.backing().unwrap().node, &bs));
    filter.attach_child("target", target.clone(), BDRV_CHILD_DATA).unwrap();
    Node::append(&filter, &bs).unwrap();

    assert!(Arc::ptr_eq(&target.backing().unwrap().node, &bs));
    assert!(Arc::ptr_eq(&filter.backing().unwrap().node, &bs));
    // The backend moved up to the filter.
    assert!(Arc::ptr_eq(&root.root().unwrap(), &filter));
    assert_eq!(bs.parent_count(), 2);
}

/// test_parallel_exclusive_write: the old permissions of the node being replaced must not
/// get in the way of the replacement.
#[test]
fn parallel_exclusive_write() {
    let top = exclusive_writer_node("top");
    let base = no_perm_node("base");
    let fl1 = pass_through_node("fl1");
    let fl2 = pass_through_node("fl2");

    let _d1 = fl1.drained();
    let _d2 = fl2.drained();

    top.attach_child("backing", fl1.clone(), FILTERED_PRIMARY).unwrap();
    fl1.attach_child("backing", base.clone(), FILTERED_PRIMARY).unwrap();
    fl2.attach_child("backing", base.clone(), FILTERED_PRIMARY).unwrap();

    Node::replace_node(&fl1, &fl2).unwrap();
    assert!(Arc::ptr_eq(&child_of(&top, "backing"), &fl2));
    assert_eq!(fl1.parent_count(), 0);
    assert_eq!(fl1.child("backing").unwrap().perm().0, 0);
    assert_eq!(fl2.child("backing").unwrap().perm().0, BLK_PERM_WRITE);
}

/// test_parallel_perm_update: permissions must be updated in topological order, parents
/// before children, or switching the selected child conflicts with the not yet updated
/// other branch.
#[test]
fn parallel_perm_update() {
    let top = no_perm_node("top");
    let selected = Arc::new(Mutex::new(None));
    let ws = node(
        &WRITE_TO_SELECTED,
        "ws",
        TestDriver { perms: Perms::None, selected: Some(selected.clone()) },
    );
    let base = no_perm_node("base");
    let fl1 = pass_through_node("fl1");
    let fl2 = pass_through_node("fl2");

    let top_child = top.attach_child("file", ws.clone(), BDRV_CHILD_DATA).unwrap();
    let c_fl1 = ws.attach_child("first", fl1.clone(), BDRV_CHILD_DATA).unwrap();
    let c_fl2 = ws.attach_child("second", fl2.clone(), BDRV_CHILD_DATA).unwrap();
    fl1.attach_child("backing", base.clone(), FILTERED_PRIMARY).unwrap();
    fl2.attach_child("backing", base.clone(), FILTERED_PRIMARY).unwrap();

    let _rd = crate::graph_lock::rdlock();
    for (sel, on, off) in [("fl1", &c_fl1, &c_fl2), ("fl2", &c_fl2, &c_fl1)]
        .into_iter()
        .chain([("fl1", &c_fl1, &c_fl2)])
    {
        *selected.lock().unwrap() = Some(sel.to_string());
        top.refresh_child_perms(&top_child).unwrap();
        assert!(on.perm().0 & BLK_PERM_WRITE != 0);
        assert!(off.perm().0 & BLK_PERM_WRITE == 0);
    }
}

/// test_append_greedy_filter: a filter that may sit in the chain but not next to it as a
/// second writer is appended with all graph changes made before the permission update.
#[test]
fn append_greedy_filter() {
    let top = exclusive_writer_node("top");
    let base = no_perm_node("base");
    let fl = exclusive_writer_node("fl1");

    top.attach_child("backing", base.clone(), FILTERED_PRIMARY).unwrap();
    Node::append(&fl, &base).unwrap();
    assert!(Arc::ptr_eq(&child_of(&top, "backing"), &fl));
    assert!(Arc::ptr_eq(&child_of(&fl, "backing"), &base));
    assert_eq!(base.parent_count(), 1);
}

/// `bdrv_replace_node()` with a backend on top: the backend follows, and a failure puts it
/// back.
#[test]
fn replace_node_moves_backends() {
    let a = no_perm_node("a");
    let b = no_perm_node("b");
    let blk = BlockBackend::with_node(None, a.clone(), BLK_PERM_WRITE, BLK_PERM_ALL).unwrap();
    Node::replace_node(&a, &b).unwrap();
    assert!(Arc::ptr_eq(&blk.root().unwrap(), &b));
    assert_eq!(a.parent_count(), 0);
    assert_eq!(b.cumulative_perm().0, BLK_PERM_WRITE);
    blk.pwrite(0, &[1; 512]).unwrap();

    // A second writer that shares nothing on `a` makes moving the first backend there fail.
    let other = BlockBackend::with_node(None, a.clone(), BLK_PERM_WRITE, 0).unwrap();
    let e = Node::replace_node(&b, &a).unwrap_err();
    assert!(e.message().starts_with("Permission conflict on node 'a': "), "{}", e.message());
    assert!(Arc::ptr_eq(&blk.root().unwrap(), &b));
    assert!(Arc::ptr_eq(&other.root().unwrap(), &a));
    assert_eq!(a.parent_count(), 1);
    assert_eq!(b.parent_count(), 1);
    drop(other);
    drop(blk);
    assert_eq!(b.parent_count(), 0);
}
