// SPDX-License-Identifier: GPL-2.0-or-later

//! Operation blockers (`bdrv_op_block()` and friends) and frozen backing links
//! (`bdrv_freeze_backing_chain()`) from block.c.
//!
//! QEMU keeps both in the nodes: `bs->op_blockers` and `BdrvChild.frozen`. The graph core here
//! has neither, so they live in side tables, keyed by the node and by the edge id of the
//! child. Only the job code consults them: the job commands check the blockers and the
//! frozen links as QEMU does, but the rest of the block layer (resize, snapshots, reopen,
//! `blockdev-del`) does not. The module documentation of [`crate::job`] lists this.

use std::sync::{Arc, Mutex, Weak};

use ruvm_base::{Error, Result};

use crate::node::Node;

/// `BlockOpType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code, reason = "the whole QEMU list, not every operation checks yet")]
pub(crate) enum BlockOpType {
    BackupSource,
    BackupTarget,
    Change,
    CommitSource,
    CommitTarget,
    DriveDel,
    Eject,
    ExternalSnapshot,
    InternalSnapshot,
    InternalSnapshotDelete,
    MirrorSource,
    MirrorTarget,
    Resize,
    Stream,
    Replace,
}

impl BlockOpType {
    const ALL: [BlockOpType; 15] = [
        BlockOpType::BackupSource,
        BlockOpType::BackupTarget,
        BlockOpType::Change,
        BlockOpType::CommitSource,
        BlockOpType::CommitTarget,
        BlockOpType::DriveDel,
        BlockOpType::Eject,
        BlockOpType::ExternalSnapshot,
        BlockOpType::InternalSnapshot,
        BlockOpType::InternalSnapshotDelete,
        BlockOpType::MirrorSource,
        BlockOpType::MirrorTarget,
        BlockOpType::Resize,
        BlockOpType::Stream,
        BlockOpType::Replace,
    ];
}

/// The `Error *reason` a blocker is registered with. Blockers are told apart by identity, as
/// QEMU compares the pointers.
#[derive(Debug)]
pub(crate) struct Reason(pub String);

struct Blocker {
    node: Weak<Node>,
    op: BlockOpType,
    reason: Arc<Reason>,
}

/// Newest first, like `QLIST_INSERT_HEAD()`.
static BLOCKERS: Mutex<Vec<Blocker>> = Mutex::new(Vec::new());

fn same_node(w: &Weak<Node>, bs: &Node) -> bool {
    std::ptr::eq(w.as_ptr(), bs)
}

/// `bdrv_op_block()`.
pub(crate) fn op_block(bs: &Node, op: BlockOpType, reason: &Arc<Reason>) {
    let mut b = BLOCKERS.lock().unwrap();
    b.retain(|x| x.node.strong_count() > 0);
    b.insert(0, Blocker { node: bs.weak(), op, reason: reason.clone() });
}

/// `bdrv_op_block_all()`.
pub(crate) fn op_block_all(bs: &Node, reason: &Arc<Reason>) {
    for op in BlockOpType::ALL {
        op_block(bs, op, reason);
    }
}

/// `bdrv_op_unblock_all()`.
pub(crate) fn op_unblock_all(bs: &Node, reason: &Arc<Reason>) {
    BLOCKERS
        .lock()
        .unwrap()
        .retain(|x| !(same_node(&x.node, bs) && Arc::ptr_eq(&x.reason, reason)));
}

/// `bdrv_op_is_blocked()`: fails with the newest blocker of `op` on `bs`. `name` is what
/// `bdrv_get_device_or_node_name()` returns for `bs`.
pub(crate) fn op_is_blocked(bs: &Node, op: BlockOpType, name: &str) -> Result<()> {
    let b = BLOCKERS.lock().unwrap();
    match b.iter().find(|x| same_node(&x.node, bs) && x.op == op) {
        Some(x) => Err(Error::generic(format!("Node '{name}' is busy: {}", x.reason.0))),
        None => Ok(()),
    }
}

/// The edge ids of the frozen links.
static FROZEN: Mutex<Vec<u64>> = Mutex::new(Vec::new());

fn is_base(i: &Arc<Node>, base: Option<&Arc<Node>>) -> bool {
    base.is_some_and(|b| Arc::ptr_eq(i, b))
}

/// The filter and COW links from `bs` down to `base` (or the bottom of the chain), with the
/// node each one leaves from.
fn chain_links(bs: &Arc<Node>, base: Option<&Arc<Node>>) -> Vec<(Arc<Node>, crate::node::Child)> {
    let mut out = Vec::new();
    let mut i = bs.clone();
    while !is_base(&i, base) {
        let Some(c) = i.filter_or_cow_child() else {
            break;
        };
        let next = c.node.clone();
        out.push((i, c));
        i = next;
    }
    out
}

/// Whether the link with edge id `edge` is frozen.
pub(crate) fn link_is_frozen(edge: u64) -> bool {
    FROZEN.lock().unwrap().contains(&edge)
}

/// `bdrv_is_backing_chain_frozen()`: fails if any filter or COW link between `bs` and `base`
/// is frozen.
pub(crate) fn chain_frozen(bs: &Arc<Node>, base: Option<&Arc<Node>>) -> Result<()> {
    let frozen = FROZEN.lock().unwrap();
    for (i, c) in chain_links(bs, base) {
        if frozen.contains(&c.edge) {
            return Err(Error::generic(format!(
                "Cannot change '{}' link from '{}' to '{}'",
                c.name, i.name, c.node.name
            )));
        }
    }
    Ok(())
}

/// `bdrv_freeze_backing_chain()`.
pub(crate) fn freeze_chain(bs: &Arc<Node>, base: Option<&Arc<Node>>) -> Result<()> {
    chain_frozen(bs, base)?;
    let mut frozen = FROZEN.lock().unwrap();
    for (_, c) in chain_links(bs, base) {
        frozen.push(c.edge);
    }
    Ok(())
}

/// `bdrv_unfreeze_backing_chain()`.
pub(crate) fn unfreeze_chain(bs: &Arc<Node>, base: Option<&Arc<Node>>) {
    let mut frozen = FROZEN.lock().unwrap();
    for (_, c) in chain_links(bs, base) {
        if let Some(p) = frozen.iter().position(|e| *e == c.edge) {
            frozen.remove(p);
        }
    }
}
