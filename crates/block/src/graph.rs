// SPDX-License-Identifier: GPL-2.0-or-later

//! The node graph: `bdrv_open()`, `bdrv_assign_node_name()` and the monitor side of
//! `blockdev-add` and `blockdev-del` from blockdev.c.
//!
//! The graph owns the nodes `blockdev-add` made and the named block backends. Nodes hold their
//! children, so a child lives as long as some parent does.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_base::{Error, Result};
use ruvm_qapi::QDict;
use ruvm_qapi::types::{
    BlockdevDetectZeroesOptions, BlockdevDiscardOptions, BlockdevOptions, BlockdevRef,
    BlockdevRefOrNull,
};

use crate::backend::BlockBackend;
use crate::drive::DriveInfo;
use crate::drivers::{self, OpenArgs};
use crate::node::{
    BDRV_CHILD_COW, BDRV_CHILD_DATA, BDRV_CHILD_FILTERED, BDRV_CHILD_METADATA, BDRV_CHILD_PRIMARY,
    Node, NodeFlags, NodeMeta, NodeSpec,
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
    /// When the node joined the table: `graph_bdrv_states` is a list in that order.
    seq: u64,
}

/// The block graph. One per emulator.
#[derive(Debug, Default)]
pub struct BlockGraph {
    nodes: Mutex<BTreeMap<String, Entry>>,
    next_id: AtomicU64,
    next_seq: AtomicU64,
    /// The block backends the monitor knows by name, `monitor_block_backends`.
    pub(crate) backends: Mutex<BTreeMap<String, Arc<BlockBackend>>>,
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

/// The nodes one `blockdev-add` opened, not yet in the name table. Dropping it closes them.
#[derive(Default)]
pub(crate) struct Pending {
    pub(crate) nodes: Vec<Arc<Node>>,
    /// Warnings printed while opening, for callers that want to show them again.
    pub(crate) warnings: Vec<String>,
}

impl Pending {
    fn has(&self, name: &str) -> bool {
        self.nodes.iter().any(|n| n.name == name)
    }

    fn find(&self, name: &str) -> Option<Arc<Node>> {
        self.nodes.iter().find(|n| n.name == name).cloned()
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
    /// `BDRV_O_NATIVE_AIO`: `-drive aio=native`, the default of `aio` for `file` nodes.
    pub native_aio: bool,
    /// `BDRV_O_NO_IO`: the tools open images only to look at their metadata.
    pub no_io: bool,
    /// `BDRV_O_CHECK`: opened for `qemu-img check`.
    pub check: bool,
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
    /// `BDRV_O_PROTOCOL`: a file name opens a protocol node, without format probing.
    pub protocol: bool,
}

/// `bdrv_inherited_options()` for `child_of_bds`: how a child with `role` of a node with
/// `parent` flags is opened, for what the child's own options leave open. `parent_protocol`
/// is whether the parent itself was opened as a protocol node.
pub(crate) fn child_ctx(
    parent: &NodeFlags,
    parent_is_format: bool,
    parent_protocol: bool,
    role: u32,
) -> OpenCtx {
    let cow = role & BDRV_CHILD_COW != 0;
    let mut protocol = parent_protocol;
    // Pure data children of non-format nodes are probed.
    if !parent_is_format
        && role & BDRV_CHILD_DATA != 0
        && role & (BDRV_CHILD_METADATA | BDRV_CHILD_FILTERED) == 0
    {
        protocol = false;
    }
    // Children of format nodes (except backing files) and metadata children never are.
    if (parent_is_format && !cow) || role & BDRV_CHILD_METADATA != 0 {
        protocol = true;
    }
    OpenCtx {
        root: false,
        inherit: Inherited {
            // Backing files are opened read-only by default.
            read_only: if cow { true } else { parent.read_only },
            auto_read_only: if cow { false } else { parent.auto_read_only },
            direct: parent.direct,
            no_flush: parent.no_flush,
            force_share: parent.force_share,
            unmap: true,
            native_aio: false,
            no_io: parent.no_io,
            check: parent.check,
        },
        probed: false,
        detect_zeroes: None,
        protocol,
    }
}

/// How the backing file of a node that is being opened is chosen.
#[derive(Debug)]
pub(crate) enum BackingReq {
    /// From the driver's own `backing` option, as `blockdev-add` gives it.
    Typed,
    /// `backing: null`: no backing file.
    None,
    /// `backing` names an existing node.
    Ref(String),
    /// A new node from these options.
    Definition(Box<BlockdevOptions>),
    /// The backing file the image names, with these `backing.*` options, which may name
    /// another file instead.
    Options(QDict),
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
            children: node.children().iter().map(|c| c.node.name.clone()).collect(),
            parents: node.parent_count(),
            monitor_owned: owned,
            size: node.getlength().unwrap_or(0),
            read_only: node.read_only(),
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

    /// `graph_bdrv_states`: every live named node, in the order they were added.
    pub(crate) fn named_nodes(&self) -> Vec<Arc<Node>> {
        let nodes = self.nodes.lock().unwrap();
        let mut v: Vec<(u64, Arc<Node>)> =
            nodes.values().filter_map(|e| e.node.upgrade().map(|n| (e.seq, n))).collect();
        v.sort_by_key(|e| e.0);
        v.into_iter().map(|e| e.1).collect()
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
    pub(crate) fn generate_name(&self) -> String {
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
        self.commit(root, pending, ctx)
    }

    /// Puts the nodes an open made into the name table.
    pub(crate) fn commit(
        &self,
        root: Arc<Node>,
        pending: Pending,
        ctx: OpenCtx,
    ) -> Result<(Arc<Node>, Vec<String>)> {
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
            let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
            nodes.insert(n.name.clone(), Entry { node: Arc::downgrade(&n), owned, seq });
        }
        Ok((root, pending.warnings))
    }

    /// `bdrv_open_inherit()` and `bdrv_open_common()` for typed options, as `blockdev-add`
    /// gives them.
    pub(crate) fn open(
        &self,
        opts: BlockdevOptions,
        ctx: OpenCtx,
        pending: &mut Pending,
    ) -> Result<Arc<Node>> {
        self.open_with(opts, ctx, pending, BackingReq::Typed)
    }

    /// `bdrv_open_inherit()` with the backing file chosen by `backing`.
    pub(crate) fn open_with(
        &self,
        opts: BlockdevOptions,
        ctx: OpenCtx,
        pending: &mut Pending,
        backing: BackingReq,
    ) -> Result<Arc<Node>> {
        let inh = ctx.inherit;
        let explicit_options = crate::reopen::flat_options(&opts)?;
        let cache = opts.cache.unwrap_or_default();
        let read_only = opts.read_only.unwrap_or(inh.read_only);
        let auto_read_only = opts.auto_read_only.unwrap_or(inh.auto_read_only);
        let unmap = match opts.discard {
            Some(d) => d == BlockdevDiscardOptions::Unmap,
            None => inh.unmap,
        };
        let flags = NodeFlags {
            read_only,
            direct: cache.direct.unwrap_or(inh.direct),
            no_flush: cache.no_flush.unwrap_or(inh.no_flush),
            unmap,
            force_share: opts.force_share.unwrap_or(inh.force_share),
            detect_zeroes: opts.detect_zeroes.unwrap_or_default(),
            auto_read_only,
            allow_rdwr: !read_only,
            inactive: !opts.active.unwrap_or(true),
            no_io: inh.no_io,
            check: inh.check,
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

        let tag = opts.u.tag();
        let Some(def) = drivers::find_format(tag.as_str()) else {
            return Err(Error::generic(format!("Unknown driver '{}'", tag.as_str())));
        };
        let meta = NodeMeta::default();
        let mut args = OpenArgs {
            graph: self,
            pending,
            ctx,
            def,
            node_name: name.clone(),
            flags,
            children: Vec::new(),
            meta,
            backing: None,
        };
        let driver = (def.open)(&mut args, opts.u)?;
        let OpenArgs { mut flags, children, mut meta, backing: typed_backing, .. } = args;
        if meta.auto_backing_file.is_empty() {
            meta.auto_backing_file = meta.backing_file.clone();
        }
        if let Some(dz) = ctx.detect_zeroes {
            flags.detect_zeroes = dz;
        }
        flags.allow_rdwr = !flags.read_only;
        let node = Node::build(NodeSpec {
            name,
            driver_name: def.format_name,
            driver,
            def: Some(def),
            flags,
            meta,
            children,
        })?;
        // Children this open created belong to this node: bs->inherits_from.
        for c in node.children() {
            if pending.nodes.iter().any(|n| Arc::ptr_eq(n, &c.node)) {
                let mut m = c.node.meta.lock().unwrap();
                if m.inherits_from.strong_count() == 0 {
                    m.inherits_from = Arc::downgrade(&node);
                }
            }
        }
        crate::reopen::store_open_options(&node, explicit_options);
        pending.nodes.push(node.clone());

        let backing = match backing {
            BackingReq::Typed => match typed_backing {
                None => BackingReq::Options(QDict::new()),
                Some(BlockdevRefOrNull::Null(())) => BackingReq::None,
                Some(BlockdevRefOrNull::Reference(r)) => BackingReq::Ref(r),
                Some(BlockdevRefOrNull::Definition(o)) => BackingReq::Definition(o),
            },
            b => b,
        };
        self.open_backing_file(&node, backing, pending)?;
        node.refresh_filename();
        Ok(node)
    }

    /// `bdrv_open_child()`: a new node or a reference to an existing one, which may be one
    /// this open made.
    pub(crate) fn open_child_ref(
        &self,
        r: BlockdevRef,
        ctx: OpenCtx,
        pending: &mut Pending,
    ) -> Result<Arc<Node>> {
        match r {
            BlockdevRef::Definition(opts) => self.open(*opts, ctx, pending),
            BlockdevRef::Reference(r) => self.lookup_pending(&r, pending),
        }
    }

    /// `bdrv_lookup_bs()`, also finding the nodes of the open in progress.
    pub(crate) fn lookup_pending(&self, name: &str, pending: &Pending) -> Result<Arc<Node>> {
        match pending.find(name) {
            Some(n) => Ok(n),
            None => self.lookup_bs(name),
        }
    }

    /// `bdrv_open_backing_file()`.
    fn open_backing_file(
        &self,
        bs: &Arc<Node>,
        req: BackingReq,
        pending: &mut Pending,
    ) -> Result<()> {
        let supports = bs.def.is_some_and(|d| d.supports_backing);
        let role =
            if bs.is_filter() { BDRV_CHILD_FILTERED | BDRV_CHILD_PRIMARY } else { BDRV_CHILD_COW };
        let parent_protocol = bs.def.is_some_and(|d| d.protocol_name.is_some());
        let ctx = child_ctx(&bs.flags(), bs.is_format(), parent_protocol, role);
        let no_support = || Error::generic("Driver doesn't support backing files");
        let backing_hd = match req {
            BackingReq::Typed | BackingReq::None => return Ok(()),
            BackingReq::Ref(r) => {
                if !supports {
                    return Err(no_support());
                }
                self.lookup_pending(&r, pending)
                    .map_err(|e| e.prepend("Could not open backing file: "))?
            }
            BackingReq::Definition(o) => {
                if !supports {
                    return Err(no_support());
                }
                let n = self
                    .open(*o, ctx, pending)
                    .map_err(|e| e.prepend("Could not open backing file: "))?;
                n.meta.lock().unwrap().inherits_from = Arc::downgrade(bs);
                n
            }
            BackingReq::Options(mut options) => {
                let meta = bs.meta.lock().unwrap().clone();
                let names_file = options
                    .get("file")
                    .and_then(|f| f.as_dict())
                    .is_some_and(|f| f.contains_key("filename"));
                let mut filename = None;
                let mut implicit = false;
                if names_file {
                    // Keep the file name empty, the options say which file.
                } else if meta.backing_file.is_empty() && options.is_empty() {
                    return Ok(());
                } else {
                    if options.is_empty() {
                        implicit = meta.auto_backing_file == meta.backing_file;
                    }
                    filename = Some(crate::open::full_backing_filename(bs, &meta.backing_file)?);
                }
                if !supports {
                    return Err(no_support());
                }
                if !meta.backing_format.is_empty() && !options.contains_key("driver") {
                    options.put("driver", meta.backing_format.as_str());
                }
                let n = self
                    .open_qdict(filename.as_deref(), options, ctx, pending, true)
                    .map_err(|e| e.prepend("Could not open backing file: "))?;
                n.meta.lock().unwrap().inherits_from = Arc::downgrade(bs);
                if implicit {
                    n.refresh_filename();
                    bs.meta.lock().unwrap().auto_backing_file =
                        n.meta.lock().unwrap().filename.clone();
                }
                n
            }
        };
        bs.set_backing_hd(Some(backing_hd))
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

#[cfg(test)]
mod graph_mod;

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
