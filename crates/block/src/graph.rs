// SPDX-License-Identifier: GPL-2.0-or-later

//! The node graph: `bdrv_open()`, `bdrv_assign_node_name()` and the monitor side of
//! `blockdev-add` and `blockdev-del` from blockdev.c.
//!
//! The graph owns the nodes `blockdev-add` made and the named block backends. Nodes hold their
//! children, so a child lives as long as some parent does.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{
    BlockdevDetectZeroesOptions, BlockdevDiscardOptions, BlockdevOptions, BlockdevOptionsBlkdebug,
    BlockdevOptionsNull, BlockdevOptionsU, BlockdevRef,
};

use crate::backend::BlockBackend;
use crate::drive::DriveInfo;
use crate::file::file_open;
use crate::node::{Driver, Node, NodeFlags};
use crate::raw::{BLOCK_PROBE_BUF_SIZE, probe_format, raw_open};

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
    /// How many parents hold this node, nodes and block backends alike.
    pub parents: usize,
    /// Set for nodes `blockdev-add` created at the root, which only `blockdev-del` may remove.
    pub monitor_owned: bool,
    /// The size a guest would see, in bytes.
    pub size: u64,
    /// Whether the node was opened read-only.
    pub read_only: bool,
}

/// A node in the graph's name table. The table holds every node by name, but only the
/// monitor owned ones strongly: the rest live as long as their parents.
#[derive(Debug)]
struct Entry {
    node: std::sync::Weak<Node>,
    /// The reference `blockdev-add` holds for the monitor.
    owned: Option<Arc<Node>>,
}

/// The block graph. One per emulator.
#[derive(Debug, Default)]
pub struct BlockGraph {
    nodes: Mutex<BTreeMap<String, Entry>>,
    next_id: AtomicU64,
    /// The block backends the monitor knows by name, `monitor_block_backends`.
    backends: Mutex<BTreeMap<String, Arc<BlockBackend>>>,
    /// The `-drive` table, `DriveInfo` in QEMU.
    pub(crate) drives: Mutex<Vec<DriveInfo>>,
}

/// `id_wellformed()`: a letter first, then letters, digits, `-`, `.` and `_`.
pub(crate) fn id_wellformed(id: &str) -> bool {
    let mut chars = id.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_'))
}

/// `qemu_config_parse()` for the blkdebug config file. Blank lines and comments are fine. The
/// `[inject-error]` and `[set-state]` rule sections are not implemented yet, so they are
/// refused rather than silently ignored.
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

/// The nodes one `blockdev-add` opened, not yet in the name table. Dropping it closes them.
#[derive(Default)]
struct Pending {
    nodes: Vec<Arc<Node>>,
    /// Warnings printed while opening, for callers that want to show them again.
    warnings: Vec<String>,
}

impl Pending {
    fn has(&self, name: &str) -> bool {
        self.nodes.iter().any(|n| n.name == name)
    }
}

/// The options a child inherits from its parent when it does not set them itself,
/// `bdrv_inherited_options()` for `child_of_bds`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Inherited {
    pub read_only: bool,
    pub auto_read_only: bool,
    pub direct: bool,
    pub no_flush: bool,
    pub force_share: bool,
    /// Children default to `discard=unmap`, the root to `ignore`.
    pub unmap: bool,
}

/// How a node is being opened.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct OpenCtx {
    /// A `blockdev-add` root: the monitor owns it.
    pub root: bool,
    pub inherit: Inherited,
    /// The format of this node was probed rather than given (`bs->probed`).
    pub probed: bool,
    /// `detect-zeroes` as `blockdev_init()` sets it, after the open and without its check.
    pub detect_zeroes: Option<BlockdevDetectZeroesOptions>,
}

/// `null-co` and `null-aio` from block/null.c.
struct NullDriver {
    size: u64,
    read_zeroes: bool,
}

impl Driver for NullDriver {
    fn pread(&self, _bs: &Node, _offset: u64, buf: &mut [u8]) -> io::Result<()> {
        if self.read_zeroes {
            buf.fill(0);
        }
        Ok(())
    }

    fn pwrite(&self, _bs: &Node, _offset: u64, _buf: &[u8]) -> io::Result<()> {
        Ok(())
    }

    fn pwrite_zeroes(&self, _: &Node, _: u64, _: u64, _: bool) -> io::Result<()> {
        Ok(())
    }

    fn getlength(&self, _bs: &Node) -> io::Result<u64> {
        Ok(self.size)
    }
}

/// `blkdebug` without rules: every request goes to `image` unchanged.
struct BlkdebugDriver;

impl Driver for BlkdebugDriver {
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        bs.file().pread(offset, buf)
    }

    fn pwrite(&self, bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        bs.file().pwrite(offset, buf)
    }

    fn pwrite_zeroes(&self, bs: &Node, offset: u64, bytes: u64, unmap: bool) -> io::Result<()> {
        bs.file().pwrite_zeroes(offset, bytes, unmap)
    }

    fn pdiscard(&self, bs: &Node, offset: u64, bytes: u64) -> io::Result<()> {
        bs.file().pdiscard(offset, bytes)
    }

    fn getlength(&self, bs: &Node) -> io::Result<u64> {
        bs.file().getlength()
    }

    fn truncate(&self, bs: &Node, len: u64) -> Result<()> {
        bs.file().truncate(len)
    }
}

fn not_found(r: &str) -> Error {
    Error::generic(format!("Cannot find device='{r}' nor node-name='{r}'"))
}

impl BlockGraph {
    /// An empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    fn info(name: &str, node: &Node, owned: bool) -> NodeInfo {
        NodeInfo {
            node_name: name.to_string(),
            driver: node.driver_name,
            children: node.children.iter().map(|c| c.node.name.clone()).collect(),
            parents: node.parent_count(),
            monitor_owned: owned,
            size: node.getlength().unwrap_or(0),
            read_only: node.flags.read_only,
        }
    }

    /// A description of the node named `name`.
    pub fn node(&self, name: &str) -> Option<NodeInfo> {
        let nodes = self.nodes.lock().unwrap();
        let e = nodes.get(name)?;
        let node = e.node.upgrade()?;
        Some(Self::info(name, &node, e.owned.is_some()))
    }

    /// Every node, by name.
    pub fn nodes(&self) -> Vec<NodeInfo> {
        let nodes = self.nodes.lock().unwrap();
        nodes
            .iter()
            .filter_map(|(k, e)| e.node.upgrade().map(|n| Self::info(k, &n, e.owned.is_some())))
            .collect()
    }

    /// `bdrv_find_node()`.
    pub(crate) fn find_node(&self, name: &str) -> Option<Arc<Node>> {
        self.nodes.lock().unwrap().get(name).and_then(|e| e.node.upgrade())
    }

    /// `blk_by_name()`: a named block backend, like the ones `-drive` makes.
    pub fn backend(&self, name: &str) -> Option<Arc<BlockBackend>> {
        self.backends.lock().unwrap().get(name).cloned()
    }

    /// `bdrv_lookup_bs()`: the root node of the backend `name`, or else the node `name`.
    pub(crate) fn lookup_bs(&self, name: &str) -> Result<Arc<Node>> {
        if let Some(blk) = self.backend(name) {
            let Some(node_name) = blk.node_name() else {
                return Err(Error::generic(format!("Device '{name}' has no medium")));
            };
            return self.find_node(&node_name).ok_or_else(|| not_found(name));
        }
        self.find_node(name).ok_or_else(|| not_found(name))
    }

    /// `monitor_add_blk()`: give `blk` a name the monitor can find it by.
    pub(crate) fn monitor_add_blk(&self, name: &str, blk: Arc<BlockBackend>) -> Result<()> {
        if !id_wellformed(name) {
            return Err(Error::generic("Invalid device name"));
        }
        // Look the node up first, open_nodes() takes the two locks the other way round.
        let node_conflict = self.find_node(name).is_some();
        let mut backends = self.backends.lock().unwrap();
        if backends.contains_key(name) {
            return Err(Error::generic(format!("Device with id '{name}' already exists")));
        }
        if node_conflict {
            return Err(Error::generic(format!(
                "Device name '{name}' conflicts with an existing node name"
            )));
        }
        backends.insert(name.to_string(), blk);
        Ok(())
    }

    /// `drive-del` and `blockdev_mark_auto_del()`: forget the drive `id` and its backend. The
    /// nodes go once the device that holds the backend lets go of it too.
    pub fn drive_del(&self, id: &str) -> Result<()> {
        let Some(blk) = self.backends.lock().unwrap().remove(id) else {
            return Err(Error::generic(format!("Device '{id}' not found")));
        };
        self.drives.lock().unwrap().retain(|d| d.id != id);
        drop(blk);
        Ok(())
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
        let ctx = OpenCtx { root: true, ..OpenCtx::default() };
        self.open_nodes(opts, ctx).map(drop)
    }

    /// Opens a tree of nodes and puts them all in the name table, returning the root and the
    /// warnings printed on the way.
    pub(crate) fn open_nodes(
        &self,
        opts: BlockdevOptions,
        ctx: OpenCtx,
    ) -> Result<(Arc<Node>, Vec<String>)> {
        let mut pending = Pending::default();
        // Opening can block (the blkdebug config may be a fifo), so the lock is only taken to
        // look names up and to commit.
        let root = self.open(opts, ctx, &mut pending)?;
        let mut nodes = self.nodes.lock().unwrap();
        let backends = self.backends.lock().unwrap();
        for n in &pending.nodes {
            if nodes.get(&n.name).is_some_and(|e| e.node.strong_count() > 0) {
                return Err(Error::generic(format!("Duplicate nodes with node-name='{}'", n.name)));
            }
            if backends.contains_key(&n.name) {
                return Err(Error::generic(format!(
                    "node-name={} is conflicting with a device id",
                    n.name
                )));
            }
        }
        for n in pending.nodes {
            let owned = (ctx.root && Arc::ptr_eq(&n, &root)).then(|| n.clone());
            nodes.insert(n.name.clone(), Entry { node: Arc::downgrade(&n), owned });
        }
        Ok((root, pending.warnings))
    }

    /// `bdrv_open_inherit()` and `bdrv_open_common()`.
    fn open(
        &self,
        opts: BlockdevOptions,
        ctx: OpenCtx,
        pending: &mut Pending,
    ) -> Result<Arc<Node>> {
        let inh = ctx.inherit;
        let cache = opts.cache.unwrap_or_default();
        let read_only = opts.read_only.unwrap_or(inh.read_only);
        let auto_read_only = opts.auto_read_only.unwrap_or(inh.auto_read_only);
        let unmap = match opts.discard {
            Some(d) => d == BlockdevDiscardOptions::Unmap,
            None => inh.unmap,
        };
        let mut flags = NodeFlags {
            read_only,
            direct: cache.direct.unwrap_or(inh.direct),
            no_flush: cache.no_flush.unwrap_or(inh.no_flush),
            unmap,
            force_share: opts.force_share.unwrap_or(inh.force_share),
            detect_zeroes: opts.detect_zeroes.unwrap_or_default(),
        };
        if flags.force_share && !flags.read_only {
            return Err(Error::generic("force-share=on can only be used with read-only images"));
        }
        // bdrv_parse_detect_zeroes().
        if flags.detect_zeroes == BlockdevDetectZeroesOptions::Unmap && !flags.unmap {
            return Err(Error::generic(
                "setting detect-zeroes to unmap is not allowed without setting discard \
                 operation to unmap",
            ));
        }
        if let Some(dz) = ctx.detect_zeroes {
            flags.detect_zeroes = dz;
        }

        let name = match opts.node_name {
            Some(n) => n,
            None => self.generate_name(),
        };
        // bdrv_assign_node_name().
        if !name.starts_with('#') && !id_wellformed(&name) {
            return Err(Error::generic(format!("Invalid node-name: '{name}'")));
        }
        if self.backend(&name).is_some() {
            return Err(Error::generic(format!(
                "node-name={name} is conflicting with a device id"
            )));
        }
        if self.find_node(&name).is_some() || pending.has(&name) {
            return Err(Error::generic(format!("Duplicate nodes with node-name='{name}'")));
        }
        if name.len() >= NODE_NAME_SIZE {
            return Err(Error::generic("Node name too long"));
        }

        // What the children inherit, bdrv_inherited_options().
        let child_ctx = OpenCtx {
            root: false,
            inherit: Inherited {
                read_only: flags.read_only,
                auto_read_only,
                direct: flags.direct,
                no_flush: flags.no_flush,
                force_share: flags.force_share,
                unmap: true,
            },
            probed: false,
            detect_zeroes: None,
        };
        let tag = opts.u.tag();
        let driver: Box<dyn Driver>;
        let mut children = Vec::new();
        match opts.u {
            BlockdevOptionsU::NullCo(o) | BlockdevOptionsU::NullAio(o) => {
                driver = Box::new(null_open(&o)?);
            }
            BlockdevOptionsU::Blkdebug(o) => {
                let child = blkdebug_open(self, *o, child_ctx, pending)?;
                children.push(("image", child));
                driver = Box::new(BlkdebugDriver);
            }
            BlockdevOptionsU::File(o) => {
                driver = Box::new(file_open(&o, &mut flags, auto_read_only)?);
            }
            BlockdevOptionsU::Raw(o) => {
                let child = self.open_child(*o.file, child_ctx, pending)?;
                let len =
                    child.getlength().map_err(|e| Error::from_io("Could not get image size", e))?;
                if ctx.probed {
                    probe_check(&child)?;
                }
                if ctx.probed && !flags.read_only {
                    let w = format!(
                        "WARNING: Image format was not specified for '{}' and probing guessed \
                         raw.\n         Automatically detecting the format is dangerous for \
                         raw images, write operations on block 0 will be restricted.\n         \
                         Specify the 'raw' format explicitly to remove the restrictions.",
                        filename_of(&child)
                    );
                    eprintln!("{w}");
                    pending.warnings.push(w);
                }
                driver = Box::new(raw_open(o.offset, o.size, len, ctx.probed)?);
                children.push(("file", child));
            }
            _ => {
                return Err(Error::generic(format!(
                    "Driver '{}' is not supported yet",
                    tag.as_str()
                )));
            }
        }
        let node = Node::new(name, tag.as_str(), driver, flags, children)?;
        pending.nodes.push(node.clone());
        Ok(node)
    }

    /// `bdrv_open_child()`: a new node or a reference to an existing one.
    fn open_child(&self, r: BlockdevRef, ctx: OpenCtx, pending: &mut Pending) -> Result<Arc<Node>> {
        match r {
            BlockdevRef::Definition(opts) => self.open(*opts, ctx, pending),
            BlockdevRef::Reference(r) => self.lookup_bs(&r),
        }
    }

    /// `qmp_blockdev_del()`.
    pub fn blockdev_del(&self, node_name: &str) -> Result<()> {
        let mut nodes = self.nodes.lock().unwrap();
        let Some(entry) = nodes.get(node_name) else {
            return Err(Error::generic(format!(
                "Failed to find node with node-name='{node_name}'"
            )));
        };
        let Some(node) = entry.node.upgrade() else {
            nodes.remove(node_name);
            return Err(Error::generic(format!(
                "Failed to find node with node-name='{node_name}'"
            )));
        };
        if node.parent_count() > 0 {
            return Err(Error::generic(format!("Node {node_name} is in use")));
        }
        if entry.owned.is_none() {
            return Err(Error::generic(format!("Node {node_name} is not owned by the monitor")));
        }
        nodes.remove(node_name);
        // bdrv_unref(): the children go with their last parent.
        drop(node);
        nodes.retain(|_, e| e.node.strong_count() > 0);
        Ok(())
    }
}

/// The file name a node stands for, `bs->filename` for the drivers here.
fn filename_of(node: &Node) -> String {
    node.filename().unwrap_or_else(|| node.name.clone())
}

/// `find_image_format()`: only raw can be opened so far, anything else that probes as
/// another format is refused rather than exposed to the guest as raw.
fn probe_check(file: &Node) -> Result<()> {
    let len = file.getlength().map_err(|e| Error::from_io("Could not get image size", e))?;
    if len == 0 {
        return Ok(());
    }
    let mut buf = [0u8; BLOCK_PROBE_BUF_SIZE];
    let n = buf.len().min(usize::try_from(len).unwrap_or(usize::MAX));
    file.pread(0, &mut buf[..n])
        .map_err(|e| Error::from_io("Could not read image for determining its format", e))?;
    match probe_format(&buf[..n]) {
        "raw" => Ok(()),
        f => Err(Error::generic(format!("Driver '{f}' is not supported yet"))),
    }
}

/// `null_file_open()`.
fn null_open(o: &BlockdevOptionsNull) -> Result<NullDriver> {
    if o.latency_ns.is_some_and(|l| l > i64::MAX as u64) {
        return Err(Error::generic("latency-ns is invalid"));
    }
    Ok(NullDriver {
        size: o.size.unwrap_or(1 << 30).max(0) as u64,
        read_zeroes: o.read_zeroes.unwrap_or(false),
    })
}

/// `blkdebug_open()`: the config file first, then the image child, then the limits.
fn blkdebug_open(
    graph: &BlockGraph,
    o: BlockdevOptionsBlkdebug,
    ctx: OpenCtx,
    pending: &mut Pending,
) -> Result<Arc<Node>> {
    if let Some(path) = &o.config {
        read_blkdebug_config(path)?;
    }
    if o.inject_error.as_ref().is_some_and(|v| !v.is_empty())
        || o.set_state.as_ref().is_some_and(|v| !v.is_empty())
    {
        return Err(Error::generic("blkdebug rules are not supported yet"));
    }
    let child = graph.open_child(*o.image, ctx, pending)?;
    if let Some(align) = o.align {
        let a = align as u64;
        if align != 0 && (align >= i64::from(i32::MAX) || !a.is_power_of_two()) {
            return Err(Error::generic(format!("Cannot meet constraints with align {a}")));
        }
    }
    Ok(child)
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
