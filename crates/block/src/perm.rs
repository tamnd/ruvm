// SPDX-License-Identifier: GPL-2.0-or-later

//! The permission bits from include/block/block-common.h, `bdrv_perm_names()`, and the
//! permission update from block.c: `bdrv_default_perms()`, `bdrv_list_refresh_perms()` and the
//! transaction that undoes a failed update.
//!
//! QEMU calls a driver's `.bdrv_check_perm` on prepare and `.bdrv_set_perm` on commit. Here the
//! driver's [`Driver::set_perm`](crate::node::Driver::set_perm) does both at once, and an
//! aborted transaction calls it again with the old permissions, which is what
//! `.bdrv_abort_perm` amounts to for the drivers there are.

use std::sync::{Arc, Weak};

use ruvm_base::{Error, Result};

use crate::node::{
    BDRV_CHILD_COW, BDRV_CHILD_DATA, BDRV_CHILD_FILTERED, BDRV_CHILD_METADATA, BDRV_SECTOR_SIZE,
    Child, Node, NodeFlags, Parent, filter_default_perms,
};

/// The user needs the data to be consistent while reading.
pub const BLK_PERM_CONSISTENT_READ: u64 = 0x01;
/// The user may change the data.
pub const BLK_PERM_WRITE: u64 = 0x02;
/// The user may write, but only data that is already there, as copy-on-read does.
pub const BLK_PERM_WRITE_UNCHANGED: u64 = 0x04;
/// The user may change the size of the node.
pub const BLK_PERM_RESIZE: u64 = 0x08;
/// Every permission there is.
pub const BLK_PERM_ALL: u64 = 0x0f;

/// `DEFAULT_PERM_PASSTHROUGH`: what a filter forwards from its parents to its child.
pub(crate) const DEFAULT_PERM_PASSTHROUGH: u64 =
    BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE | BLK_PERM_WRITE_UNCHANGED | BLK_PERM_RESIZE;
/// `DEFAULT_PERM_UNCHANGED`: what a filter always shares.
pub(crate) const DEFAULT_PERM_UNCHANGED: u64 = BLK_PERM_ALL & !DEFAULT_PERM_PASSTHROUGH;

/// `bdrv_perm_names()`: the names of the bits in `perm`, joined with `, `.
pub fn perm_names(perm: u64) -> String {
    const NAMES: [(u64, &str); 4] = [
        (BLK_PERM_CONSISTENT_READ, "consistent read"),
        (BLK_PERM_WRITE, "write"),
        (BLK_PERM_WRITE_UNCHANGED, "write unchanged"),
        (BLK_PERM_RESIZE, "resize"),
    ];
    NAMES.iter().filter(|(bit, _)| perm & bit != 0).map(|(_, n)| *n).collect::<Vec<_>>().join(", ")
}

/// What a driver's `child_perm` gets to look at, the arguments of `.bdrv_child_perm` other
/// than the parents' permissions.
pub(crate) struct PermCtx<'a> {
    /// The child node.
    pub child: &'a Arc<Node>,
    /// The `BDRV_CHILD_*` role of the edge.
    pub role: u32,
    /// The position of the edge among the parent's children.
    pub index: usize,
    /// `bdrv_is_writable_after_reopen(bs, q)`.
    pub writable: bool,
    /// `BDRV_O_NO_IO` after the reopen.
    pub no_io: bool,
    /// `BDRV_O_INACTIVE` as the node is now (QEMU looks at `bs->open_flags` here).
    pub inactive: bool,
}

/// `bdrv_default_perms()`: what a parent needs on a child with the role in `ctx`.
pub(crate) fn default_perms(ctx: &PermCtx<'_>, perm: u64, shared: u64) -> (u64, u64) {
    let role = ctx.role;
    if role & BDRV_CHILD_FILTERED != 0 {
        filter_default_perms(perm, shared)
    } else if role & BDRV_CHILD_COW != 0 {
        default_perms_for_cow(ctx, perm, shared)
    } else if role & (BDRV_CHILD_METADATA | BDRV_CHILD_DATA) != 0 {
        default_perms_for_storage(ctx, perm, shared)
    } else {
        // QEMU asserts. A child without a role gets nothing and shares everything.
        (0, BLK_PERM_ALL)
    }
}

/// `bdrv_default_perms_for_cow()`: a backing file is only read, and others may write to it
/// only if the parents let others write to the image.
fn default_perms_for_cow(ctx: &PermCtx<'_>, perm: u64, shared: u64) -> (u64, u64) {
    let perm = perm & BLK_PERM_CONSISTENT_READ;
    let mut s = if shared & BLK_PERM_WRITE != 0 { BLK_PERM_WRITE | BLK_PERM_RESIZE } else { 0 };
    s |= BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE_UNCHANGED;
    if ctx.inactive {
        s |= BLK_PERM_WRITE | BLK_PERM_RESIZE;
    }
    (perm, s)
}

/// `bdrv_default_perms_for_storage()`.
fn default_perms_for_storage(ctx: &PermCtx<'_>, perm: u64, shared: u64) -> (u64, u64) {
    let (mut perm, mut shared) = filter_default_perms(perm, shared);
    if ctx.role & BDRV_CHILD_METADATA != 0 {
        // Format drivers may touch metadata even if the guest does not write.
        if ctx.writable {
            perm |= BLK_PERM_WRITE | BLK_PERM_RESIZE;
        }
        if !ctx.no_io {
            perm |= BLK_PERM_CONSISTENT_READ;
        }
        shared &= !(BLK_PERM_WRITE | BLK_PERM_RESIZE);
    }
    if ctx.role & BDRV_CHILD_DATA != 0 {
        shared &= !BLK_PERM_RESIZE;
        if perm & BLK_PERM_WRITE_UNCHANGED != 0 {
            perm |= BLK_PERM_WRITE;
        }
        if perm & BLK_PERM_WRITE != 0 {
            perm |= BLK_PERM_RESIZE;
        }
    }
    if ctx.inactive {
        shared |= BLK_PERM_WRITE | BLK_PERM_RESIZE;
    }
    (perm, shared)
}

/// The flags nodes will have after a reopen, the part of a `BlockReopenQueue` the permission
/// code looks at.
#[derive(Default)]
pub(crate) struct ReopenFlags(pub Vec<(usize, NodeFlags)>);

impl ReopenFlags {
    fn get(&self, bs: &Node) -> Option<NodeFlags> {
        let key = bs as *const Node as usize;
        self.0.iter().find(|(k, _)| *k == key).map(|(_, f)| *f)
    }
}

/// `bdrv_reopen_get_flags()`.
pub(crate) fn flags_after_reopen(bs: &Node, q: Option<&ReopenFlags>) -> NodeFlags {
    q.and_then(|q| q.get(bs)).unwrap_or_else(|| bs.flags())
}

/// `bdrv_is_writable_after_reopen()`.
pub(crate) fn writable_after_reopen(bs: &Node, q: Option<&ReopenFlags>) -> bool {
    let f = flags_after_reopen(bs, q);
    !f.read_only && !f.inactive
}

/// One undoable step of a permission change.
enum Undo {
    /// An edge's permissions were changed from these.
    EdgePerm(Arc<Node>, u64, (u64, u64)),
    /// A parent edge was added.
    EdgeAdded(Arc<Node>, u64),
    /// A parent edge was removed.
    EdgeRemoved(Arc<Node>, Parent),
    /// The driver took new permissions; these are the old ones.
    DrvPerm(Arc<Node>, (u64, u64)),
    /// The child edge `edge` of the parent was moved away from the old node.
    ChildReplaced(Arc<Node>, u64, Arc<Node>, Parent),
    /// The parent got the new child edge `edge`.
    ChildAdded(Arc<Node>, u64),
    /// The parent lost this child edge, which was at this position.
    ChildRemoved(Arc<Node>, usize, Child, Parent),
    /// The node's `inherits_from` was changed from this.
    InheritsFrom(Arc<Node>, Weak<Node>),
    /// A parent that is not a node (a backend) was moved from the first node to the second.
    ParentMoved(Arc<Node>, Arc<Node>, Parent),
}

/// A permission transaction, the `Transaction` QEMU threads through the permission code. It
/// records what changed so that [`PermTran::abort`] can put it all back. Dropping it without
/// calling either method commits.
#[derive(Default)]
pub(crate) struct PermTran {
    undo: Vec<Undo>,
}

impl PermTran {
    pub(crate) fn edge_perm(&mut self, child: Arc<Node>, id: u64, old: (u64, u64)) {
        self.undo.push(Undo::EdgePerm(child, id, old));
    }

    pub(crate) fn edge_added(&mut self, child: Arc<Node>, id: u64) {
        self.undo.push(Undo::EdgeAdded(child, id));
    }

    pub(crate) fn edge_removed(&mut self, child: Arc<Node>, old: Parent) {
        self.undo.push(Undo::EdgeRemoved(child, old));
    }

    pub(crate) fn child_replaced(
        &mut self,
        parent: Arc<Node>,
        edge: u64,
        old: Arc<Node>,
        p: Parent,
    ) {
        self.undo.push(Undo::ChildReplaced(parent, edge, old, p));
    }

    pub(crate) fn child_added(&mut self, parent: Arc<Node>, edge: u64) {
        self.undo.push(Undo::ChildAdded(parent, edge));
    }

    pub(crate) fn child_removed(&mut self, parent: Arc<Node>, idx: usize, c: Child, p: Parent) {
        self.undo.push(Undo::ChildRemoved(parent, idx, c, p));
    }

    pub(crate) fn parent_moved(&mut self, from: Arc<Node>, to: Arc<Node>, p: Parent) {
        self.undo.push(Undo::ParentMoved(from, to, p));
    }

    /// Sets `inherits_from` of `bs`, to be put back on abort.
    pub(crate) fn set_inherits_from(&mut self, bs: &Arc<Node>, to: Weak<Node>) {
        let old = std::mem::replace(&mut bs.meta.lock().unwrap().inherits_from, to);
        self.undo.push(Undo::InheritsFrom(bs.clone(), old));
    }

    /// Keeps everything.
    pub(crate) fn commit(mut self) {
        self.undo.clear();
    }

    /// Puts everything back, newest change first, then gives the drivers their old
    /// permissions back.
    pub(crate) fn abort(mut self) {
        let mut drv = Vec::new();
        while let Some(u) = self.undo.pop() {
            match u {
                Undo::EdgePerm(n, id, (p, s)) => {
                    n.set_parent_perm(id, p, s);
                }
                Undo::EdgeAdded(n, id) => {
                    n.remove_parent(id);
                }
                Undo::EdgeRemoved(n, p) => n.add_parent(p),
                Undo::DrvPerm(n, old) => drv.push((n, old)),
                Undo::ChildReplaced(parent, edge, old, p) => {
                    parent.unreplace_child_node(edge, old, p);
                }
                Undo::ChildAdded(parent, edge) => parent.unattach_child(edge),
                Undo::ChildRemoved(parent, idx, c, p) => parent.unremove_child(idx, c, p),
                Undo::InheritsFrom(bs, old) => bs.meta.lock().unwrap().inherits_from = old,
                Undo::ParentMoved(from, to, p) => from.unmove_parent(&to, p),
            }
        }
        // `.bdrv_abort_perm`: in the order the nodes were checked.
        for (n, old) in drv.into_iter().rev() {
            if n.drv_perm() != old {
                let _ = n.driver.set_perm(old.0, old.1);
                n.set_drv_perm(old);
            }
        }
    }
}

/// `bdrv_topological_dfs()`: `bs` and everything below it, parents before children.
fn topological_dfs(list: &mut Vec<Arc<Node>>, found: &mut Vec<usize>, bs: &Arc<Node>) {
    let key = Arc::as_ptr(bs) as usize;
    if found.contains(&key) {
        return;
    }
    found.push(key);
    for c in bs.children() {
        topological_dfs(list, found, &c.node);
    }
    // g_slist_prepend().
    list.insert(0, bs.clone());
}

/// `bdrv_a_allow_b()`.
fn a_allow_b(bs: &Node, a: &Parent, b: &Parent) -> Result<()> {
    if b.perm & a.shared == b.perm {
        return Ok(());
    }
    let name = &bs.name;
    Err(Error::generic(format!(
        "Permission conflict on node '{name}': permissions '{}' are both required by {} (uses \
         node '{name}' as '{}' child) and unshared by {} (uses node '{name}' as '{}' child).",
        perm_names(b.perm & !a.shared),
        b.desc,
        b.child_name,
        a.desc,
        a.child_name
    )))
}

/// `bdrv_parent_perms_conflict()`: every ordered pair of parents, in both directions.
fn parent_perms_conflict(bs: &Node) -> Result<()> {
    let parents = bs.parents();
    for (i, a) in parents.iter().enumerate() {
        for (j, b) in parents.iter().enumerate() {
            if i != j {
                a_allow_b(bs, a, b)?;
            }
        }
    }
    Ok(())
}

/// `bdrv_node_refresh_perm()`.
fn node_refresh_perm(bs: &Arc<Node>, q: Option<&ReopenFlags>, tran: &mut PermTran) -> Result<()> {
    let (perm, shared) = bs.cumulative_perm();
    let wants_write = perm & (BLK_PERM_WRITE | BLK_PERM_WRITE_UNCHANGED) != 0;
    if wants_write && !writable_after_reopen(bs, q) {
        if !writable_after_reopen(bs, None) {
            return Err(Error::generic("Block node is read-only"));
        }
        return Err(Error::generic(format!(
            "Read-only block node '{}' cannot support read-write users",
            bs.name
        )));
    }
    if wants_write && perm & BLK_PERM_RESIZE == 0 {
        let size = bs.total_sectors().max(0) as u64 * BDRV_SECTOR_SIZE;
        if size % bs.request_alignment() != 0 {
            return Err(Error::generic(
                "Cannot get 'write' permission without 'resize': Image size is not a multiple \
                 of request alignment",
            ));
        }
    }
    let old = bs.drv_perm();
    if old != (perm, shared) {
        bs.driver.set_perm(perm, shared)?;
        bs.set_drv_perm((perm, shared));
        tran.undo.push(Undo::DrvPerm(bs.clone(), old));
    }
    for c in bs.children() {
        let (p, s) = bs.child_perm_of(&c, perm, shared, q);
        if let Some(old) = c.node.set_parent_perm(c.edge, p, s) {
            if old != (p, s) {
                tran.edge_perm(c.node.clone(), c.edge, old);
            }
        }
    }
    Ok(())
}

/// `bdrv_list_refresh_perms()`: check and update the permissions of `roots` and everything
/// below them. On failure the caller aborts `tran`.
pub(crate) fn refresh_perms(
    roots: &[Arc<Node>],
    q: Option<&ReopenFlags>,
    tran: &mut PermTran,
) -> Result<()> {
    let mut list = Vec::new();
    let mut found = Vec::new();
    for r in roots {
        topological_dfs(&mut list, &mut found, r);
    }
    for bs in &list {
        parent_perms_conflict(bs)?;
        node_refresh_perm(bs, q, tran)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(perm_names(BLK_PERM_WRITE), "write");
        assert_eq!(perm_names(BLK_PERM_ALL), "consistent read, write, write unchanged, resize");
        assert_eq!(perm_names(0), "");
    }
}
