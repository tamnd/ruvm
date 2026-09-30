// SPDX-License-Identifier: GPL-2.0-or-later

//! The backing chain helpers of block.c that the jobs use: `bdrv_find_overlay()`,
//! `bdrv_chain_contains()`, `bdrv_find_backing_image()`, `bdrv_insert_node()`,
//! `bdrv_drop_filter()`, `bdrv_drop_intermediate()`, and the making of the internal filter
//! nodes (`bdrv_new_open_driver()`) the commit and mirror jobs put on top of their source,
//! `check_to_replace_node()` and `bdrv_recurse_can_replace()`.

use std::sync::Arc;

use ruvm_base::report::report_error;
use ruvm_base::{Error, Result};
use ruvm_qapi::types::{BlockdevOptions, BlockdevOptionsU};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};
use ruvm_qapi::{QDict, QValue};

use super::block_job::device_or_node_name;
use super::blocker::{BlockOpType, op_is_blocked};
use crate::drivers::{DriverDef, OpenArgs};
use crate::graph::{BlockGraph, OpenCtx, Pending, id_wellformed};
use crate::node::{BDRV_CHILD_COW, Driver, Node, NodeFlags, NodeMeta, NodeSpec};
use crate::open::full_backing_filename;
use crate::tools::path_has_protocol;

/// `NODE_NAME_SIZE` of QEMU, the limit `bdrv_assign_node_name()` checks.
const NODE_NAME_SIZE: usize = 32;

/// `bdrv_find_overlay()`: the node of the chain of `active` whose backing node (filters
/// skipped) is `bs`, or the bottom-most node of the chain for `None`.
pub(crate) fn find_overlay(active: &Arc<Node>, bs: Option<&Arc<Node>>) -> Option<Arc<Node>> {
    let bs = bs.map(|b| b.skip_filters());
    let mut active = Some(active.skip_filters());
    while let Some(a) = active {
        let next = a.backing_chain_next();
        let same = match (&bs, &next) {
            (Some(b), Some(n)) => Arc::ptr_eq(b, n),
            (None, None) => true,
            _ => false,
        };
        if same {
            return Some(a);
        }
        active = next;
    }
    None
}

/// `bdrv_chain_contains()`: whether `base` is `top` or below it, filters included.
pub(crate) fn chain_contains(top: &Arc<Node>, base: &Arc<Node>) -> bool {
    let mut top = Some(top.clone());
    while let Some(t) = top {
        if Arc::ptr_eq(&t, base) {
            return true;
        }
        top = t.filter_or_cow_bs();
    }
    false
}

/// `bdrv_find_backing_image()`: the node below `bs` whose file name is `backing_file`,
/// compared the way the images name their backing files.
pub(crate) fn find_backing_image(bs: &Arc<Node>, backing_file: &str) -> Option<Arc<Node>> {
    let is_protocol = path_has_protocol(backing_file);
    let mut curr = bs.skip_filters();
    while curr.cow_child().is_some() {
        let below = curr.backing_chain_next()?;
        let curr_backing = curr.meta.lock().unwrap().backing_file.clone();
        if curr.backing_overridden(curr.backing().as_ref()) {
            // If the backing file was overridden, we can only compare directly against the
            // backing node's filename.
            below.refresh_filename();
            if below.meta.lock().unwrap().filename == backing_file {
                return Some(below);
            }
        } else if is_protocol || path_has_protocol(&curr_backing) {
            // If either of the filename paths is actually a protocol, then compare unmodified
            // paths; otherwise make paths relative.
            if backing_file == curr_backing {
                return Some(below);
            }
            // Also check against the full backing filename for the image
            if full_backing_filename(&curr, &curr_backing).is_ok_and(|f| f == backing_file) {
                return Some(below);
            }
        } else {
            // If not an absolute filename path, make it relative to the current image's
            // filename path, and compare canonicalized absolute pathnames.
            let want = full_backing_filename(&curr, backing_file)
                .ok()
                .and_then(|f| std::fs::canonicalize(f).ok());
            let have = full_backing_filename(&curr, &curr_backing)
                .ok()
                .and_then(|f| std::fs::canonicalize(f).ok());
            if let (Some(w), Some(h)) = (want, have) {
                if w == h {
                    return Some(below);
                }
            }
        }
        curr = below;
    }
    None
}

/// `bdrv_insert_node()`: opens a node from `options` and puts it in the place of `bs`,
/// whose parents then use it.
pub(crate) fn insert_node(graph: &BlockGraph, bs: &Arc<Node>, options: QDict) -> Result<Arc<Node>> {
    let Some(drvname) = options.get_str("driver").map(str::to_string) else {
        return Err(Error::generic("driver is not specified"));
    };
    if crate::drivers::find_format(&drvname).is_none() {
        return Err(Error::generic(format!("Unknown driver: '{drvname}'")));
    }
    let mut v = QObjectInputVisitor::new(QValue::Dict(options));
    let mut opts = BlockdevOptions::default();
    BlockdevOptions::visit(&mut v, None, &mut opts)
        .map_err(|e| e.prepend("Could not create node: "))?;
    let (new, _) = graph
        .open_nodes(opts, OpenCtx::default())
        .map_err(|e| e.prepend("Could not create node: "))?;
    Node::replace_node(bs, &new).map_err(|e| e.prepend("Could not replace node: "))?;
    Ok(new)
}

/// `bdrv_drop_filter()`: the parents of the filter `bs` use its child instead.
pub(crate) fn drop_filter(bs: &Arc<Node>) -> Result<()> {
    let Some(child) = bs.filter_or_cow_bs() else {
        return Ok(());
    };
    Node::replace_node_common(bs, &child, true)
}

/// `bdrv_drop_intermediate()`: the parents of `top` use `base` instead, and the images
/// among them record `backing_file_str` (or the file name of `base`) as their backing file.
/// Returns `-EIO` as QEMU does, after reporting what went wrong.
pub(crate) fn drop_intermediate(
    top: &Arc<Node>,
    base: &Arc<Node>,
    backing_file_str: Option<&str>,
    backing_mask_protocol: bool,
) -> Result<(), i32> {
    let _d = base.drained();
    // Make sure that base is in the backing chain of top
    if !chain_contains(top, base) {
        return Err(libc::EIO);
    }
    let backing_file_str = match backing_file_str {
        Some(s) => s.to_string(),
        None => {
            base.refresh_filename();
            base.meta.lock().unwrap().filename.clone()
        }
    };
    let updated: Vec<(u64, Arc<Node>)> = top
        .parents()
        .iter()
        .filter_map(|p| p.ops.as_ref().and_then(|o| o.node()).map(|n| (p.id, n)))
        .collect();

    if let Err(e) = Node::replace_node_common(top, base, false) {
        report_error(&e);
        return Err(libc::EIO);
    }

    for (edge, parent) in updated {
        // bdrv_child_cb_update_filename(): only the backing links name a file.
        let Some(c) = parent.children().into_iter().find(|c| c.edge == edge) else {
            continue;
        };
        if c.role & BDRV_CHILD_COW == 0 {
            continue;
        }
        if let Err(e) =
            backing_update_filename(&parent, base, &backing_file_str, backing_mask_protocol)
        {
            report_error(&e);
            return Err(libc::EIO);
        }
    }
    Ok(())
}

/// `bdrv_backing_update_filename()`.
fn backing_update_filename(
    parent: &Arc<Node>,
    base: &Arc<Node>,
    filename: &str,
    backing_mask_protocol: bool,
) -> Result<()> {
    let read_only = parent.read_only();
    if read_only {
        parent.reopen_set_read_only(false)?;
    }
    let format = if backing_mask_protocol && base.is_protocol() { "raw" } else { base.driver_name };
    let r = parent
        .change_backing_file(Some(filename), Some(format), false)
        .map_err(|e| Error::from_io("Could not update backing file link", e));
    if read_only {
        parent.reopen_set_read_only(true)?;
    }
    r
}

/// `bdrv_assign_node_name()` for a node the block layer makes by itself.
fn assign_node_name(graph: &BlockGraph, name: Option<&str>) -> Result<String> {
    let name = match name {
        Some(n) => n.to_string(),
        None => graph.generate_name(),
    };
    if !name.starts_with('#') && !id_wellformed(&name) {
        return Err(Error::generic(format!("Invalid node-name: '{name}'")));
    }
    if graph.backend(&name).is_some() {
        return Err(Error::generic(format!("node-name={name} is conflicting with a device id")));
    }
    if graph.find_node(&name).is_some() {
        return Err(Error::generic(format!("Duplicate nodes with node-name='{name}'")));
    }
    if name.len() >= NODE_NAME_SIZE {
        return Err(Error::generic("Node name too long"));
    }
    Ok(name)
}

/// `bdrv_new_open_driver()` of an internal filter driver followed by `bdrv_append()`: a
/// filter named `name` (or a generated name) on top of `bs`, which takes the parents of
/// `bs`. The node is in the name table while it lives.
pub(crate) fn append_filter(
    graph: &BlockGraph,
    bs: &Arc<Node>,
    name: Option<&str>,
    def: &'static DriverDef,
    driver: Box<dyn Driver>,
) -> Result<Arc<Node>> {
    let name = assign_node_name(graph, name)?;
    let flags = NodeFlags { read_only: false, ..NodeFlags::default() };
    let filter = Node::build(NodeSpec {
        name,
        driver_name: def.format_name,
        driver,
        def: Some(def),
        flags,
        meta: NodeMeta::default(),
        children: Vec::new(),
    })?;
    Node::append(&filter, bs)?;
    filter
        .refresh_total_sectors(None)
        .map_err(|e| Error::from_io("Could not refresh total sector count", e))?;
    filter.refresh_filename();
    let (filter, _) = graph.commit(
        filter.clone(),
        Pending { nodes: vec![filter], warnings: Vec::new() },
        OpenCtx::default(),
    )?;
    Ok(filter)
}

/// The open function of the internal filter drivers, which only the jobs make.
pub(crate) fn internal_open(_: &mut OpenArgs<'_>, _: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    Err(Error::generic("This driver can only be used internally"))
}

/// `bdrv_recurse_can_replace()`: whether replacing `to_replace` by a copy of `bs` does not
/// change what the parents of `to_replace` see.
pub(crate) fn recurse_can_replace(bs: &Arc<Node>, to_replace: &Arc<Node>) -> bool {
    if Arc::ptr_eq(bs, to_replace) {
        return true;
    }
    // For filters without an own implementation, we can recurse on our own
    if bs.is_filter() {
        if let Some(c) = bs.filter_child() {
            return recurse_can_replace(&c.node, to_replace);
        }
    }
    false
}

/// `check_to_replace_node()`: the node `node_name` if a mirror of `parent_bs` may replace it.
pub(crate) fn check_to_replace_node(
    graph: &BlockGraph,
    parent_bs: &Arc<Node>,
    node_name: &str,
) -> Result<Arc<Node>> {
    let Some(to_replace_bs) = graph.find_node(node_name) else {
        return Err(Error::generic(format!("Failed to find node with node-name='{node_name}'")));
    };
    op_is_blocked(
        &to_replace_bs,
        BlockOpType::Replace,
        &device_or_node_name(graph, &to_replace_bs),
    )?;
    // We don't want arbitrary node of the BDS chain to be replaced only the top most non
    // filter in order to prevent data corruption. Another benefit is that this tests
    // exclude backing files which are blocked by the backing blockers.
    if !recurse_can_replace(parent_bs, &to_replace_bs) {
        return Err(Error::generic(format!(
            "Cannot replace '{node_name}' by a node mirrored from '{}', because it cannot be \
             guaranteed that doing so would not lead to an abrupt change of visible data",
            parent_bs.name
        )));
    }
    Ok(to_replace_bs)
}
