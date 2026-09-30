// SPDX-License-Identifier: GPL-2.0-or-later

//! The node graph: `bdrv_open()`, `bdrv_assign_node_name()` and the monitor side of
//! `blockdev-add` and `blockdev-del` from blockdev.c.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{
    BlockdevOptions, BlockdevOptionsBlkdebug, BlockdevOptionsNull, BlockdevOptionsU, BlockdevRef,
};

/// `sizeof(bs->node_name)`. A name must leave room for the terminating NUL.
const NODE_NAME_SIZE: usize = 32;

/// What the graph knows about one node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeInfo {
    /// The node name, given or generated.
    pub node_name: String,
    /// The driver name, as `driver` spells it.
    pub driver: &'static str,
    /// The node names of the children, in the order they were attached.
    pub children: Vec<String>,
    /// How many parents hold this node.
    pub parents: usize,
    /// Set for nodes `blockdev-add` created at the root, which only `blockdev-del` may remove.
    pub monitor_owned: bool,
    /// The size a guest would see, in bytes.
    pub size: u64,
}

/// The block graph. One per emulator.
#[derive(Debug, Default)]
pub struct BlockGraph {
    nodes: Mutex<BTreeMap<String, NodeInfo>>,
    next_id: AtomicU64,
}

/// `id_wellformed()`: a letter first, then letters, digits, `-`, `.` and `_`.
fn id_wellformed(id: &str) -> bool {
    let mut chars = id.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_'))
}

/// `qemu_config_parse()` for the blkdebug config file. Blank lines and comments are fine. The
/// `[inject-error]` and `[set-state]` rule sections need the I/O path, which does not exist
/// yet, so they are refused rather than silently ignored.
fn read_blkdebug_config(path: &str) -> Result<()> {
    // fopen() blocks on a fifo until a writer shows up, and the QMP out-of-band test relies
    // on that. File::open() behaves the same way.
    let f = File::open(path).map_err(|e| Error::from_io(format!("Could not open '{path}'"), e))?;
    for (n, line) in BufReader::new(f).lines().enumerate() {
        let line = line.map_err(|e| Error::from_io(format!("{path}:{}", n + 1), e))?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        return Err(Error::generic(format!(
            "{path}:{}: blkdebug rules are not supported yet",
            n + 1
        )));
    }
    Ok(())
}

/// A node the open code has built but not yet put into the graph.
struct Pending {
    nodes: Vec<NodeInfo>,
    /// References to existing nodes that gained a parent.
    referenced: Vec<String>,
}

impl BlockGraph {
    /// An empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// A copy of the node named `name`.
    pub fn node(&self, name: &str) -> Option<NodeInfo> {
        self.nodes.lock().unwrap().get(name).cloned()
    }

    /// Every node, by name.
    pub fn nodes(&self) -> Vec<NodeInfo> {
        self.nodes.lock().unwrap().values().cloned().collect()
    }

    /// `id_generate(ID_BLOCK)`: `#block` followed by a counter and two digits. QEMU uses a
    /// random number for the last two, a fixed one is just as unique.
    fn generate_name(&self) -> String {
        let n = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("#block{n}{:02}", n % 100)
    }

    /// `qmp_blockdev_add()`.
    pub fn blockdev_add(&self, opts: BlockdevOptions) -> Result<()> {
        if opts.node_name.is_none() {
            return Err(Error::generic("'node-name' must be specified for the root node"));
        }
        let mut pending = Pending { nodes: Vec::new(), referenced: Vec::new() };
        // Opening can block (the blkdebug config may be a fifo), so the lock is only taken to
        // look names up and to commit.
        self.open(opts, true, &mut pending)?;
        let mut nodes = self.nodes.lock().unwrap();
        for n in &pending.nodes {
            if nodes.contains_key(&n.node_name) {
                return Err(Error::generic(format!(
                    "Duplicate nodes with node-name='{}'",
                    n.node_name
                )));
            }
        }
        for r in &pending.referenced {
            match nodes.get_mut(r) {
                Some(n) => n.parents += 1,
                None => {
                    return Err(Error::generic(format!(
                        "Cannot find device='{r}' nor node-name='{r}'"
                    )));
                }
            }
        }
        for n in pending.nodes {
            nodes.insert(n.node_name.clone(), n);
        }
        Ok(())
    }

    /// `bdrv_open_inherit()`, returning the name of the node it made.
    fn open(&self, opts: BlockdevOptions, root: bool, pending: &mut Pending) -> Result<String> {
        let name = match opts.node_name {
            Some(n) => n,
            None => self.generate_name(),
        };
        // bdrv_assign_node_name().
        if !name.starts_with('#') && !id_wellformed(&name) {
            return Err(Error::generic(format!("Invalid node-name: '{name}'")));
        }
        let taken = self.nodes.lock().unwrap().contains_key(&name)
            || pending.nodes.iter().any(|n| n.node_name == name);
        if taken {
            return Err(Error::generic(format!("Duplicate nodes with node-name='{name}'")));
        }
        if name.len() >= NODE_NAME_SIZE {
            return Err(Error::generic("Node name too long"));
        }
        let driver = opts.u.tag();
        let (children, size) = match opts.u {
            BlockdevOptionsU::NullCo(o) | BlockdevOptionsU::NullAio(o) => {
                (Vec::new(), null_open(&o)?)
            }
            BlockdevOptionsU::Blkdebug(o) => blkdebug_open(self, *o, pending)?,
            _ => {
                return Err(Error::generic(format!(
                    "Driver '{}' is not supported yet",
                    driver.as_str()
                )));
            }
        };
        pending.nodes.push(NodeInfo {
            node_name: name.clone(),
            driver: driver.as_str(),
            children,
            parents: usize::from(!root),
            monitor_owned: root,
            size,
        });
        Ok(name)
    }

    /// `qmp_blockdev_del()`.
    pub fn blockdev_del(&self, node_name: &str) -> Result<()> {
        let mut nodes = self.nodes.lock().unwrap();
        let Some(node) = nodes.get(node_name) else {
            return Err(Error::generic(format!(
                "Failed to find node with node-name='{node_name}'"
            )));
        };
        if node.parents > 0 {
            return Err(Error::generic(format!("Node {node_name} is in use")));
        }
        if !node.monitor_owned {
            return Err(Error::generic(format!("Node {node_name} is not owned by the monitor")));
        }
        let mut drop_list = vec![node_name.to_string()];
        // bdrv_unref(): a child goes away with its last parent.
        while let Some(name) = drop_list.pop() {
            let Some(n) = nodes.get(&name) else { continue };
            if n.parents > 0 || (n.monitor_owned && name != node_name) {
                continue;
            }
            let n = nodes.remove(&name).expect("looked up above");
            for c in n.children {
                if let Some(child) = nodes.get_mut(&c) {
                    child.parents -= 1;
                    drop_list.push(c);
                }
            }
        }
        Ok(())
    }
}

/// `null_file_open()`.
fn null_open(o: &BlockdevOptionsNull) -> Result<u64> {
    if o.latency_ns.is_some_and(|l| l > i64::MAX as u64) {
        return Err(Error::generic("latency-ns is invalid"));
    }
    Ok(o.size.unwrap_or(1 << 30).max(0) as u64)
}

/// `blkdebug_open()`: the config file first, then the image child, then the limits.
fn blkdebug_open(
    graph: &BlockGraph,
    o: BlockdevOptionsBlkdebug,
    pending: &mut Pending,
) -> Result<(Vec<String>, u64)> {
    if let Some(path) = &o.config {
        read_blkdebug_config(path)?;
    }
    if o.inject_error.as_ref().is_some_and(|v| !v.is_empty())
        || o.set_state.as_ref().is_some_and(|v| !v.is_empty())
    {
        return Err(Error::generic("blkdebug rules are not supported yet"));
    }
    let (child, size) = match *o.image {
        BlockdevRef::Definition(opts) => {
            let name = graph.open(*opts, false, pending)?;
            let size = pending.nodes.last().map_or(0, |n| n.size);
            (name, size)
        }
        BlockdevRef::Reference(r) => {
            let size = graph.node(&r).map(|n| n.size).ok_or_else(|| {
                Error::generic(format!("Cannot find device='{r}' nor node-name='{r}'"))
            })?;
            pending.referenced.push(r.clone());
            (r, size)
        }
    };
    if let Some(align) = o.align {
        let a = align as u64;
        if align != 0 && (align >= i64::from(i32::MAX) || !a.is_power_of_two()) {
            return Err(Error::generic(format!("Cannot meet constraints with align {a}")));
        }
    }
    Ok((vec![child], size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvm_qapi::json;
    use ruvm_qapi::visit::{QObjectInputVisitor, Visit};

    fn opts(s: &str) -> BlockdevOptions {
        let mut v = QObjectInputVisitor::new(json::from_str(s).unwrap());
        let mut o = BlockdevOptions::default();
        BlockdevOptions::visit(&mut v, None, &mut o).unwrap();
        o
    }

    #[test]
    fn add_and_del() {
        let g = BlockGraph::new();
        let e = g.blockdev_add(opts(r#"{"driver": "null-co"}"#)).unwrap_err();
        assert_eq!(e.message(), "'node-name' must be specified for the root node");
        g.blockdev_add(opts(r#"{"driver": "null-co", "node-name": "n0"}"#)).unwrap();
        assert_eq!(g.node("n0").unwrap().size, 1 << 30);
        let e = g.blockdev_add(opts(r#"{"driver": "null-aio", "node-name": "n0"}"#)).unwrap_err();
        assert_eq!(e.message(), "Duplicate nodes with node-name='n0'");
        let e = g.blockdev_add(opts(r#"{"driver": "null-aio", "node-name": "0n"}"#)).unwrap_err();
        assert_eq!(e.message(), "Invalid node-name: '0n'");
        let long = format!(r#"{{"driver": "null-aio", "node-name": "{}"}}"#, "a".repeat(32));
        assert_eq!(g.blockdev_add(opts(&long)).unwrap_err().message(), "Node name too long");

        g.blockdev_add(opts(
            r#"{"driver": "blkdebug", "node-name": "d", "image": "n0", "align": 512}"#,
        ))
        .unwrap();
        assert_eq!(g.blockdev_del("n0").unwrap_err().message(), "Node n0 is in use");
        g.blockdev_del("d").unwrap();
        g.blockdev_del("n0").unwrap();
        assert_eq!(
            g.blockdev_del("n0").unwrap_err().message(),
            "Failed to find node with node-name='n0'"
        );
    }

    #[test]
    fn nested_child_goes_with_its_parent() {
        let g = BlockGraph::new();
        g.blockdev_add(opts(
            r#"{"driver": "blkdebug", "node-name": "d",
                "image": {"driver": "null-co", "size": 4096}}"#,
        ))
        .unwrap();
        assert_eq!(g.nodes().len(), 2);
        assert_eq!(g.node("d").unwrap().size, 4096);
        let child = g.node("d").unwrap().children[0].clone();
        assert!(child.starts_with("#block"));
        assert!(g.blockdev_del(&child).unwrap_err().message().contains("in use"));
        g.blockdev_del("d").unwrap();
        assert!(g.nodes().is_empty());
    }

    #[test]
    fn blkdebug_checks() {
        let g = BlockGraph::new();
        let e = g
            .blockdev_add(opts(
                r#"{"driver": "blkdebug", "node-name": "d", "align": 3,
                    "image": {"driver": "null-co"}}"#,
            ))
            .unwrap_err();
        assert_eq!(e.message(), "Cannot meet constraints with align 3");
        assert!(g.nodes().is_empty());
        let e = g
            .blockdev_add(opts(
                r#"{"driver": "blkdebug", "node-name": "d", "config": "/nonexistent/x",
                    "image": {"driver": "null-co"}}"#,
            ))
            .unwrap_err();
        assert!(e.message().starts_with("Could not open '/nonexistent/x': "));
        let e = g
            .blockdev_add(opts(r#"{"driver": "blkdebug", "node-name": "d", "image": "nope"}"#))
            .unwrap_err();
        assert_eq!(e.message(), "Cannot find device='nope' nor node-name='nope'");
    }
}
