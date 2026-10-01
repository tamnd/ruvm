// SPDX-License-Identifier: GPL-2.0-or-later

//! `bdrv_reopen()` and `blockdev-reopen` transactions from block.c and blockdev.c.
//!
//! A reopen builds a queue of nodes with their new options ([`ReopenQueue`], QEMU's
//! `BlockReopenQueue`). The children a node opened itself from its own options (the ones whose
//! `inherits_from` is that node) come along, with the `child.*` options and whatever they
//! inherit from the new options of the parent. [`ReopenQueue::run`] then flushes every node,
//! prepares each (generic options, the driver's `reopen_prepare`, new `backing` and `file`
//! children), checks and updates the permissions with the new flags, and commits in reverse
//! order, or aborts everything on the first error.
//!
//! Every node keeps the options it was opened with, flattened (`bs->options` and
//! `bs->explicit_options`): [`store_open_options`] records them after the open.
//!
//! Differences from QEMU:
//!
//! - Options are compared as the strings the command line would give, so `read-only: true`
//!   and `read-only: "on"` count as the same value. QEMU compares the typed values and so
//!   rejects an unchanged option given with another type as a change.
//! - `bs->explicit_options` holds the options of the typed `BlockdevOptions` the node was
//!   opened with. `bs->options` adds the effective `cache.direct`, `cache.no-flush`,
//!   `read-only`, `auto-read-only` and (for `discard=unmap`) `discard`, as
//!   `update_options_from_flags()` and `bdrv_inherited_options()` add them in QEMU.
//! - A child counts as opened by its parent (`inherits_from`) when the same open created it
//!   and no other parent claimed it first. QEMU records this inside `bdrv_open_child()`.
//! - There are no implicit nodes (block job filters) and no frozen backing links yet, so the
//!   checks for them never fire.
//! - Errors name the node, not the block backend it may be the root of
//!   (`bdrv_get_device_or_node_name()`).

use std::sync::{Arc, Weak};

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{BlockdevDetectZeroesOptions, BlockdevOptions};
use ruvm_qapi::visit::{QObjectOutputVisitor, Visit, qapi_bool_parse};
use ruvm_qapi::{QDict, QValue, json};

use crate::graph::BlockGraph;
use crate::node::{
    BDRV_CHILD_COW, BDRV_CHILD_FILTERED, BDRV_CHILD_METADATA, BDRV_CHILD_PRIMARY, Node, NodeFlags,
    ReopenState,
};
use crate::perm::{PermTran, ReopenFlags, refresh_perms};

/// `qdict_flatten()`: nested dictionaries and lists become dotted keys.
pub(crate) fn flatten(d: &QDict) -> QDict {
    fn walk(out: &mut QDict, prefix: &str, v: &QValue) {
        match v {
            QValue::Dict(d) => {
                for (k, v) in d.iter_inserted() {
                    walk(out, &join_key(prefix, k), v);
                }
            }
            QValue::List(l) => {
                for (i, v) in l.iter().enumerate() {
                    walk(out, &join_key(prefix, &i.to_string()), v);
                }
            }
            v => out.put(prefix, v.clone()),
        }
    }
    fn join_key(prefix: &str, k: &str) -> String {
        if prefix.is_empty() { k.to_string() } else { format!("{prefix}.{k}") }
    }
    let mut out = QDict::new();
    for (k, v) in d.iter_inserted() {
        match v {
            // An empty dictionary or list vanishes, as in QEMU.
            QValue::Dict(_) | QValue::List(_) => walk(&mut out, k, v),
            v => out.put(k, v.clone()),
        }
    }
    out
}

/// The typed options as a flat dictionary, what `blockdev-reopen` and the open path store.
pub(crate) fn flat_options(opts: &BlockdevOptions) -> Result<QDict> {
    let mut out = QObjectOutputVisitor::new();
    let mut o = opts.clone();
    BlockdevOptions::visit(&mut out, None, &mut o)?;
    match out.complete() {
        QValue::Dict(d) => Ok(flatten(&d)),
        _ => Ok(QDict::new()),
    }
}

/// `qdict_extract_subqdict()`: takes the `prefix*` keys out of `src`, without the prefix.
fn extract_subqdict(src: &mut QDict, prefix: &str) -> QDict {
    let keys: Vec<String> =
        src.keys().filter(|k| k.starts_with(prefix)).map(str::to_string).collect();
    let mut out = QDict::new();
    for k in keys {
        if let Some(v) = src.remove(&k) {
            out.put(&k[prefix.len()..], v);
        }
    }
    out
}

/// Removes the options of the child `name`: the reference and every `name.*` key.
fn remove_child_opts(d: &mut QDict, name: &str) {
    d.remove(name);
    extract_subqdict(d, &format!("{name}."));
}

/// `qdict_join(dst, src, false)` on flat dictionaries.
fn join_flat(dst: &mut QDict, src: &QDict) {
    for (k, v) in src.iter_inserted() {
        if !dst.contains_key(k) {
            dst.put(k, v.clone());
        }
    }
}

/// `bdrv_join_options()`: the driver's merge, or the generic one.
fn join_options(bs: &Node, options: &mut QDict, old: QDict) {
    match bs.def.and_then(|d| d.join_options) {
        Some(f) => f(options, old),
        None => join_flat(options, &old),
    }
}

/// `qdict_copy_default()`.
fn copy_default(dst: &mut QDict, src: &QDict, key: &str) {
    if !dst.contains_key(key) {
        if let Some(v) = src.get(key) {
            dst.put(key, v.clone());
        }
    }
}

/// `qdict_set_default_str()`.
fn set_default_str(dst: &mut QDict, key: &str, v: &str) {
    if !dst.contains_key(key) {
        dst.put(key, v);
    }
}

/// An option as the command line would spell it, for comparing values of different types.
fn as_string(v: &QValue) -> Option<String> {
    match v {
        QValue::Str(s) => Some(s.clone()),
        QValue::Bool(b) => Some(if *b { "on" } else { "off" }.to_string()),
        QValue::Int(i) => Some(i.to_string()),
        QValue::Uint(u) => Some(u.to_string()),
        QValue::Double(f) => Some(json::format_g17(*f)),
        QValue::Null => None,
        v => Some(v.to_json()),
    }
}

fn same_value(a: &QValue, b: &QValue) -> bool {
    a == b || (as_string(a).is_some() && as_string(a) == as_string(b))
}

/// A boolean option, `qemu_opt_get_bool()` on what `qemu_opts_absorb_qdict()` took.
fn get_bool(d: &QDict, key: &str, default: bool) -> Result<bool> {
    match d.get(key) {
        None => Ok(default),
        Some(QValue::Bool(b)) => Ok(*b),
        Some(QValue::Str(s)) => qapi_bool_parse(key, s),
        Some(v) => qapi_bool_parse(key, &as_string(v).unwrap_or_default()),
    }
}

/// `update_flags_from_options()`.
fn update_flags_from_options(flags: &mut NodeFlags, d: &QDict) -> Result<()> {
    flags.no_flush = get_bool(d, "cache.no-flush", false)?;
    flags.direct = get_bool(d, "cache.direct", false)?;
    flags.read_only = get_bool(d, "read-only", false)?;
    flags.auto_read_only = get_bool(d, "auto-read-only", false)?;
    flags.inactive = !get_bool(d, "active", true)?;
    Ok(())
}

/// Records the options `bs` was opened with, flattened, as `bs->explicit_options` and
/// `bs->options`, without the options of its children.
pub(crate) fn store_open_options(bs: &Node, mut explicit: QDict) {
    for c in bs.children() {
        remove_child_opts(&mut explicit, &c.name);
    }
    remove_child_opts(&mut explicit, "backing");
    let f = bs.flags();
    let mut options = explicit.clone();
    // update_options_from_flags().
    set_default(&mut options, "cache.direct", f.direct);
    set_default(&mut options, "cache.no-flush", f.no_flush);
    set_default(&mut options, "read-only", f.read_only);
    set_default(&mut options, "auto-read-only", f.auto_read_only);
    if f.unmap {
        set_default_str(&mut options, "discard", "unmap");
    }
    let mut meta = bs.meta.lock().unwrap();
    meta.options = options;
    meta.explicit_options = explicit;
}

fn set_default(d: &mut QDict, key: &str, v: bool) {
    if !d.contains_key(key) {
        d.put(key, v);
    }
}

/// `bdrv_inherits_from_recursive()`: whether `parent` is up the `inherits_from` chain of
/// `child`.
fn inherits_from_recursive(child: Option<&Arc<Node>>, parent: &Node) -> bool {
    let mut cur = child.cloned();
    while let Some(c) = cur {
        if std::ptr::eq(Arc::as_ptr(&c), parent) {
            return true;
        }
        cur = c.meta.lock().unwrap().inherits_from.upgrade();
    }
    false
}

/// Whether `bs` inherits from `parent` directly.
fn inherits_from(bs: &Node, parent: &Node) -> bool {
    bs.meta
        .lock()
        .unwrap()
        .inherits_from
        .upgrade()
        .is_some_and(|p| std::ptr::eq(Arc::as_ptr(&p), parent))
}

/// `bdrv_recurse_has_child()`.
fn recurse_has_child(bs: &Arc<Node>, child: &Node) -> bool {
    std::ptr::eq(Arc::as_ptr(bs), child)
        || bs.children().iter().any(|c| recurse_has_child(&c.node, child))
}

/// One node in a reopen, `BlockReopenQueueEntry` with its `BDRVReopenState`.
struct Entry {
    bs: Arc<Node>,
    options: QDict,
    explicit: QDict,
    flags: NodeFlags,
    backing_missing: bool,
    prepared: bool,
    opaque: Option<Box<dyn std::any::Any + Send + Sync>>,
    old_backing: Option<Arc<Node>>,
    old_file: Option<Arc<Node>>,
}

/// A reopen transaction, `BlockReopenQueue`. Every node is drained while the queue exists.
/// `graph` resolves the node names `backing` and `file` may give.
pub(crate) struct ReopenQueue<'g> {
    entries: Vec<Entry>,
    graph: Option<&'g BlockGraph>,
}

impl<'g> ReopenQueue<'g> {
    /// An empty queue. `bdrv_reopen_queue()` drains every node for the first entry.
    pub(crate) fn new(graph: Option<&'g BlockGraph>) -> Self {
        crate::drain::drain_all_begin();
        ReopenQueue { entries: Vec::new(), graph }
    }

    /// `bdrv_reopen_queue()`: adds `bs` with the flat `options`. With `keep_old_opts`,
    /// options not given keep their old values; without, they go back to their defaults.
    pub(crate) fn add(&mut self, bs: &Arc<Node>, options: QDict, keep_old_opts: bool) {
        self.add_child(bs, options, None, keep_old_opts);
    }

    fn find(&self, bs: &Node) -> Option<usize> {
        self.entries.iter().position(|e| std::ptr::eq(Arc::as_ptr(&e.bs), bs))
    }

    /// `bdrv_reopen_queue_child()`. `parent` is the new options and flags of the parent that
    /// opened `bs`, with the role of the edge and whether the parent is a format node.
    fn add_child(
        &mut self,
        bs: &Arc<Node>,
        mut options: QDict,
        parent: Option<(&QDict, NodeFlags, u32)>,
        keep_old_opts: bool,
    ) {
        let existing = self.find(bs);
        // Old explicitly set values, which inherited ones do not override.
        if let Some(i) = existing {
            join_options(bs, &mut options, self.entries[i].explicit.clone());
        } else if keep_old_opts {
            join_options(bs, &mut options, bs.meta.lock().unwrap().explicit_options.clone());
        }
        let explicit = options.clone();

        let old = bs.flags();
        let mut flags = match parent {
            Some((popts, pflags, role)) => {
                // bdrv_inherited_options().
                copy_default(&mut options, popts, "cache.direct");
                copy_default(&mut options, popts, "cache.no-flush");
                copy_default(&mut options, popts, "force-share");
                if role & BDRV_CHILD_COW != 0 {
                    set_default_str(&mut options, "read-only", "on");
                    set_default_str(&mut options, "auto-read-only", "off");
                } else {
                    copy_default(&mut options, popts, "read-only");
                    copy_default(&mut options, popts, "auto-read-only");
                }
                set_default_str(&mut options, "discard", "unmap");
                let mut f = pflags;
                if role & BDRV_CHILD_METADATA != 0 {
                    f.no_io = false;
                }
                f
            }
            None => old,
        };
        if keep_old_opts {
            join_options(bs, &mut options, bs.meta.lock().unwrap().options.clone());
        }
        // Errors come again from bdrv_reopen_prepare().
        let _ = update_flags_from_options(&mut flags, &options);
        flags.force_share = old.force_share;
        flags.detect_zeroes = old.detect_zeroes;
        if !flags.read_only {
            flags.allow_rdwr = true;
        }

        let backing_missing = !keep_old_opts
            && !options.contains_key("backing")
            && !options.contains_key("backing.driver");
        let entry = Entry {
            bs: bs.clone(),
            options: options.clone(),
            explicit,
            flags,
            backing_missing,
            prepared: false,
            opaque: None,
            old_backing: None,
            old_file: None,
        };
        let idx = match existing {
            Some(i) => {
                let e = &mut self.entries[i];
                e.options = entry.options;
                e.explicit = entry.explicit;
                e.flags = entry.flags;
                if !keep_old_opts {
                    e.backing_missing = entry.backing_missing;
                }
                i
            }
            None => {
                self.entries.push(entry);
                self.entries.len() - 1
            }
        };

        for child in bs.children() {
            if !inherits_from(&child.node, bs) {
                continue;
            }
            let mut child_keep_old = keep_old_opts;
            let mut child_opts = QDict::new();
            if options.contains_key(&child.name) {
                // Only a reference to the current child keeps it, with its old options.
                if options.get_str(&child.name) != Some(child.node.name.as_str()) {
                    continue;
                }
                child_keep_old = true;
            } else {
                let prefix = format!("{}.", child.name);
                extract_subqdict(&mut self.entries[idx].explicit, &prefix);
                child_opts = extract_subqdict(&mut self.entries[idx].options, &prefix);
                extract_subqdict(&mut options, &prefix);
            }
            let popts = self.entries[idx].options.clone();
            self.add_child(
                &child.node,
                child_opts,
                Some((&popts, flags, child.role)),
                child_keep_old,
            );
        }
    }

    /// `bdrv_reopen_multiple()`.
    pub(crate) fn run(mut self) -> Result<()> {
        let mut tran = PermTran::default();
        let r = self.run_inner(&mut tran);
        if r.is_err() {
            {
                let _wr = crate::graph_lock::wrlock();
                tran.abort();
            }
            for e in &mut self.entries {
                if e.prepared {
                    let mut st = e.state(QDict::new());
                    e.bs.driver.reopen_abort(&e.bs, &mut st);
                }
            }
        }
        r
    }

    fn run_inner(&mut self, tran: &mut PermTran) -> Result<()> {
        for e in &self.entries {
            e.bs.flush().map_err(|err| Error::from_io("Error flushing drive", err))?;
        }
        for i in 0..self.entries.len() {
            prepare(&mut self.entries[i], self.graph, tran)?;
            self.entries[i].prepared = true;
        }
        let mut list: Vec<Arc<Node>> = Vec::new();
        for e in &self.entries {
            list.insert(0, e.bs.clone());
            if let Some(b) = &e.old_backing {
                list.insert(0, b.clone());
            }
            if let Some(f) = &e.old_file {
                list.insert(0, f.clone());
            }
        }
        let q = ReopenFlags(
            self.entries.iter().map(|e| (Arc::as_ptr(&e.bs) as usize, e.flags)).collect(),
        );
        {
            let _rd = crate::graph_lock::rdlock();
            refresh_perms(&list, Some(&q), tran)?;
        }
        for e in self.entries.iter_mut().rev() {
            commit(e);
        }
        {
            let _wr = crate::graph_lock::wrlock();
            std::mem::take(tran).commit();
        }
        for e in self.entries.iter().rev() {
            e.bs.driver.reopen_commit_post(&e.bs);
        }
        Ok(())
    }
}

impl Drop for ReopenQueue<'_> {
    fn drop(&mut self) {
        // bdrv_reopen_queue_free().
        crate::drain::drain_all_end();
    }
}

impl Entry {
    fn state(&mut self, options: QDict) -> ReopenState {
        ReopenState { flags: self.flags, options, opaque: self.opaque.take() }
    }
}

/// `bdrv_parse_discard_flags()`.
fn parse_discard(v: &str) -> Option<bool> {
    match v {
        "off" | "ignore" => Some(false),
        "on" | "unmap" => Some(true),
        _ => None,
    }
}

/// `bdrv_reopen_prepare()`.
fn prepare(e: &mut Entry, graph: Option<&BlockGraph>, tran: &mut PermTran) -> Result<()> {
    let bs = e.bs.clone();
    let mut options = e.options.clone();

    // The generic options, bdrv_runtime_opts. `node-name`, `driver` and `force-share` stay
    // for the check at the end.
    let mut flags = e.flags;
    update_flags_from_options(&mut flags, &options)?;
    for k in ["cache.direct", "cache.no-flush", "read-only", "auto-read-only", "active"] {
        options.remove(k);
    }
    if let Some(v) = options.remove("discard") {
        let s = as_string(&v).unwrap_or_default();
        match parse_discard(&s) {
            Some(u) => flags.unmap = u,
            None => return Err(Error::generic("Invalid discard option")),
        }
    }
    // bdrv_parse_detect_zeroes().
    flags.detect_zeroes = match options.remove("detect-zeroes") {
        None => BlockdevDetectZeroesOptions::Off,
        Some(v) => {
            let s = as_string(&v).unwrap_or_default();
            BlockdevDetectZeroesOptions::from_name(&s)
                .ok_or_else(|| Error::generic(format!("invalid parameter value: {s}")))?
        }
    };
    if flags.detect_zeroes == BlockdevDetectZeroesOptions::Unmap && !flags.unmap {
        return Err(Error::generic(
            "setting detect-zeroes to unmap is not allowed without setting discard operation \
             to unmap",
        ));
    }
    e.flags = flags;

    // bdrv_can_set_read_only(bs, read_only, true).
    if flags.read_only && bs.copy_on_read.load(std::sync::atomic::Ordering::SeqCst) > 0 {
        return Err(Error::generic(format!(
            "Can't set node '{}' to r/o with copy-on-read enabled",
            bs.name
        )));
    }

    // bdrv_reset_options_allowed().
    const COMMON: [&str; 7] = [
        "node-name",
        "discard",
        "cache.direct",
        "cache.no-flush",
        "read-only",
        "auto-read-only",
        "detect-zeroes",
    ];
    let mutable = bs.def.map_or(&[][..], |d| d.mutable_opts);
    let old_options = bs.meta.lock().unwrap().options.clone();
    for k in old_options.keys() {
        if !options.contains_key(k) && !COMMON.contains(&k) && !mutable.contains(&k) {
            return Err(Error::generic(format!(
                "Option '{k}' cannot be reset to its default value"
            )));
        }
    }

    let mut st = e.state(options);
    match bs.driver.reopen_prepare(&bs, &mut st) {
        None => {
            e.opaque = st.opaque;
            return Err(Error::generic(format!(
                "Block format '{}' used by node '{}' does not support reopening files",
                bs.driver_name, bs.name
            )));
        }
        Some(Err(err)) => {
            e.opaque = st.opaque;
            return Err(err);
        }
        Some(Ok(())) => {}
    }
    let ReopenState { mut options, opaque, .. } = st;
    e.opaque = opaque;

    let r = prepare_children(e, &mut options, graph, tran);
    if r.is_err() {
        // The driver prepared, so it is aborted here: the caller only aborts nodes that
        // prepared completely.
        let mut st = e.state(QDict::new());
        bs.driver.reopen_abort(&bs, &mut st);
    }
    r
}

/// The rest of `bdrv_reopen_prepare()` after the driver: `backing` and `file`, and the
/// options nobody took.
fn prepare_children(
    e: &mut Entry,
    options: &mut QDict,
    graph: Option<&BlockGraph>,
    tran: &mut PermTran,
) -> Result<()> {
    let bs = e.bs.clone();
    let supports_backing = bs.def.is_some_and(|d| d.supports_backing);
    if supports_backing
        && e.backing_missing
        && (bs.backing().is_some() || !bs.meta.lock().unwrap().backing_file.is_empty())
    {
        return Err(Error::generic(format!("backing is missing for '{}'", bs.name)));
    }
    parse_file_or_backing(e, options, true, graph, tran)?;
    options.remove("backing");
    parse_file_or_backing(e, options, false, graph, tran)?;
    options.remove("file");

    let old = bs.meta.lock().unwrap().options.clone();
    for (k, new) in options.iter_inserted() {
        if let QValue::Str(s) = new {
            if bs.child(k).is_some_and(|c| c.node.name == *s) {
                continue;
            }
        }
        if !old.get(k).is_some_and(|o| same_value(new, o)) {
            return Err(Error::generic(format!("Cannot change the option '{k}'")));
        }
    }
    Ok(())
}

/// `bdrv_reopen_parse_file_or_backing()`.
fn parse_file_or_backing(
    e: &mut Entry,
    options: &QDict,
    is_backing: bool,
    graph: Option<&BlockGraph>,
    tran: &mut PermTran,
) -> Result<()> {
    let bs = e.bs.clone();
    let child_name = if is_backing { "backing" } else { "file" };
    let Some(value) = options.get(child_name) else {
        return Ok(());
    };
    let old_child = if is_backing { bs.backing() } else { bs.child("file") };
    let new_child = match value {
        QValue::Null => None,
        QValue::Str(s) if old_child.as_ref().is_some_and(|c| c.node.name == *s) => return Ok(()),
        QValue::Str(s) => {
            let n = lookup(graph, s)?;
            if recurse_has_child(&n, &bs) {
                return Err(Error::generic(format!(
                    "Making '{s}' a {child_name} child of '{}' would create a cycle",
                    bs.name
                )));
            }
            Some(n)
        }
        _ => {
            return Err(Error::generic(format!(
                "Invalid parameter type for '{child_name}', expected: string"
            )));
        }
    };
    let old_bs = old_child.as_ref().map(|c| c.node.clone());
    if match (&old_bs, &new_child) {
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        (None, None) => true,
        _ => false,
    } {
        return Ok(());
    }
    if bs.is_filter() && old_bs.is_none() {
        return Err(Error::generic(format!(
            "'{}' is a {} filter node that does not support a {child_name} child",
            bs.name, bs.driver_name
        )));
    }
    if is_backing {
        e.old_backing = old_bs;
    } else {
        e.old_file = old_bs;
    }
    let _wr = crate::graph_lock::wrlock();
    set_file_or_backing_noperm(&bs, new_child, is_backing, old_child, tran)
}

/// `bdrv_set_file_or_backing_noperm()`.
fn set_file_or_backing_noperm(
    bs: &Arc<Node>,
    child_bs: Option<Arc<Node>>,
    is_backing: bool,
    child: Option<crate::node::Child>,
    tran: &mut PermTran,
) -> Result<()> {
    let update_inherits_from = inherits_from_recursive(child_bs.as_ref(), bs);
    let def = bs.def;
    if is_backing && !bs.is_filter() && !def.is_some_and(|d| d.supports_backing) {
        return Err(Error::generic(format!(
            "Driver '{}' of node '{}' does not support backing files",
            bs.driver_name, bs.name
        )));
    }
    let role = if bs.is_filter() {
        BDRV_CHILD_FILTERED | BDRV_CHILD_PRIMARY
    } else if is_backing {
        BDRV_CHILD_COW
    } else {
        match &child {
            Some(c) => c.role,
            None => {
                return Err(Error::generic(
                    "Cannot set file child to format node without file child",
                ));
            }
        }
    };
    if let Some(c) = &child {
        // bdrv_unset_inherits_from().
        if inherits_from(&c.node, bs)
            && !bs.children().iter().any(|k| k.edge != c.edge && Arc::ptr_eq(&k.node, &c.node))
        {
            tran.set_inherits_from(&c.node, Weak::new());
        }
        bs.remove_child_noperm(c.edge, tran);
    }
    if let Some(n) = child_bs {
        bs.attach_child_noperm(if is_backing { "backing" } else { "file" }, n.clone(), role, tran);
        if update_inherits_from {
            tran.set_inherits_from(&n, Arc::downgrade(bs));
        }
    }
    Ok(())
}

fn lookup(graph: Option<&BlockGraph>, name: &str) -> Result<Arc<Node>> {
    match graph {
        Some(g) => g.lookup_bs(name),
        None => Err(Error::generic(format!("Cannot find device='{name}' nor node-name='{name}'"))),
    }
}

/// `bdrv_reopen_commit()`.
fn commit(e: &mut Entry) {
    let bs = e.bs.clone();
    let mut st = e.state(e.options.clone());
    bs.driver.reopen_commit(&bs, &mut st);
    {
        let mut meta = bs.meta.lock().unwrap();
        meta.options = e.options.clone();
        meta.explicit_options = e.explicit.clone();
        for c in bs.children() {
            meta.options.remove(&c.name);
            meta.explicit_options.remove(&c.name);
        }
        meta.options.remove("backing");
        meta.explicit_options.remove("backing");
    }
    let mut f = e.flags;
    let old = bs.flags();
    f.no_io = old.no_io;
    f.force_share = old.force_share;
    bs.set_flags(f);
    let _ = bs.refresh_limits();
    let _ = bs.refresh_total_sectors(Some(bs.total_sectors()));
}

impl Node {
    /// `bdrv_reopen()`: reopens this node (and the children it opened) with `options`.
    pub(crate) fn reopen(self: &Arc<Self>, options: QDict, keep_old_opts: bool) -> Result<()> {
        let mut q = ReopenQueue::new(None);
        q.add(self, options, keep_old_opts);
        q.run()
    }

    /// `bdrv_reopen_set_read_only()`.
    pub(crate) fn reopen_set_read_only(self: &Arc<Self>, read_only: bool) -> Result<()> {
        let mut o = QDict::new();
        o.put("read-only", read_only);
        self.reopen(o, true)
    }
}

impl BlockGraph {
    /// `qmp_blockdev_reopen()`: reopens every node in `list` in one transaction. Options
    /// left out go back to their defaults.
    pub fn blockdev_reopen(&self, list: Vec<BlockdevOptions>) -> Result<()> {
        let mut q = ReopenQueue::new(Some(self));
        for opts in list {
            let Some(name) = opts.node_name.clone() else {
                return Err(Error::generic("node-name not specified"));
            };
            let Some(bs) = self.find_node(&name) else {
                return Err(Error::generic(format!("Failed to find node with node-name='{name}'")));
            };
            q.add(&bs, flat_options(&opts)?, false);
        }
        q.run()
    }

    /// `bdrv_reopen()` of the node `node_name` with flat (dotted) or nested options, for the
    /// tools and tests, as `qemu-io -c reopen -o` does.
    pub fn reopen_node(&self, node_name: &str, options: QDict, keep_old_opts: bool) -> Result<()> {
        let bs = self.lookup_bs(node_name)?;
        let mut q = ReopenQueue::new(Some(self));
        q.add(&bs, flatten(&options), keep_old_opts);
        q.run()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BlockBackend;
    use crate::perm::{BLK_PERM_ALL, BLK_PERM_CONSISTENT_READ, BLK_PERM_WRITE};
    use ruvm_qapi::visit::QObjectInputVisitor;

    fn opts(s: &str) -> BlockdevOptions {
        let mut v = QObjectInputVisitor::new(json::from_str(s).unwrap());
        let mut o = BlockdevOptions::default();
        BlockdevOptions::visit(&mut v, None, &mut o).unwrap();
        o
    }

    fn graph() -> BlockGraph {
        let g = BlockGraph::new();
        g.blockdev_add(opts(r#"{"driver": "null-co", "node-name": "n0", "size": 1048576}"#))
            .unwrap();
        g.blockdev_add(opts(r#"{"driver": "null-co", "node-name": "n1", "size": 2097152}"#))
            .unwrap();
        g
    }

    fn err(r: Result<()>) -> String {
        r.unwrap_err().message().to_string()
    }

    #[test]
    fn flatten_nested() {
        let d = json::from_str(r#"{"a": {"b": 1, "c": {"d": "x"}}, "l": [{"e": true}], "z": {}}"#)
            .unwrap();
        let f = flatten(d.as_dict().unwrap());
        assert_eq!(f.get("a.b"), Some(&QValue::Int(1)));
        assert_eq!(f.get_str("a.c.d"), Some("x"));
        assert_eq!(f.get("l.0.e"), Some(&QValue::Bool(true)));
        assert!(!f.contains_key("z"));
    }

    #[test]
    fn stored_options() {
        let g = graph();
        let n0 = g.find_node("n0").unwrap();
        let m = n0.meta.lock().unwrap();
        assert_eq!(m.explicit_options.get_str("driver"), Some("null-co"));
        assert!(!m.explicit_options.contains_key("read-only"));
        assert_eq!(m.options.get("read-only"), Some(&QValue::Bool(false)));
    }

    #[test]
    fn set_read_only() {
        let g = graph();
        let n0 = g.find_node("n0").unwrap();
        n0.reopen_set_read_only(true).unwrap();
        assert!(n0.read_only());
        assert_eq!(n0.meta.lock().unwrap().options.get("read-only"), Some(&QValue::Bool(true)));
        n0.reopen_set_read_only(false).unwrap();
        assert!(!n0.read_only());
    }

    #[test]
    fn qmp_errors() {
        let g = graph();
        assert_eq!(
            err(g.blockdev_reopen(vec![opts(r#"{"driver": "null-co", "size": 1048576}"#)])),
            "node-name not specified"
        );
        assert_eq!(
            err(g.blockdev_reopen(vec![opts(r#"{"driver": "null-co", "node-name": "nope"}"#)])),
            "Failed to find node with node-name='nope'"
        );
        assert_eq!(
            err(g.blockdev_reopen(vec![opts(r#"{"driver": "null-co", "node-name": "n0"}"#)])),
            "Option 'size' cannot be reset to its default value"
        );
        assert_eq!(
            err(g.blockdev_reopen(vec![opts(
                r#"{"driver": "null-co", "node-name": "n0", "size": 4096}"#
            )])),
            "Cannot change the option 'size'"
        );
        g.blockdev_reopen(vec![opts(
            r#"{"driver": "null-co", "node-name": "n0", "size": 1048576, "discard": "unmap",
                "detect-zeroes": "unmap", "read-only": true}"#,
        )])
        .unwrap();
        let f = g.find_node("n0").unwrap().flags();
        assert!(f.read_only && f.unmap);
        assert_eq!(f.detect_zeroes, BlockdevDetectZeroesOptions::Unmap);
    }

    #[test]
    fn qmp_reopen_read_only_and_back() {
        let g = graph();
        let n0 = g.find_node("n0").unwrap();
        g.blockdev_reopen(vec![opts(
            r#"{"driver": "null-co", "node-name": "n0", "size": 1048576, "read-only": true,
                "detect-zeroes": "on"}"#,
        )])
        .unwrap();
        assert!(n0.read_only());
        assert_eq!(n0.flags().detect_zeroes, BlockdevDetectZeroesOptions::On);
        // Left out, detect-zeroes goes back to off and read-only to off.
        g.blockdev_reopen(vec![opts(
            r#"{"driver": "null-co", "node-name": "n0", "size": 1048576}"#,
        )])
        .unwrap();
        assert!(!n0.read_only());
        assert_eq!(n0.flags().detect_zeroes, BlockdevDetectZeroesOptions::Off);
    }

    #[test]
    fn generic_option_errors() {
        let g = graph();
        let mut o = QDict::new();
        o.put("discard", "bogus");
        assert_eq!(err(g.reopen_node("n0", o, true)), "Invalid discard option");
        let mut o = QDict::new();
        o.put("detect-zeroes", "unmap");
        o.put("discard", "ignore");
        assert_eq!(
            err(g.reopen_node("n0", o, true)),
            "setting detect-zeroes to unmap is not allowed without setting discard operation \
             to unmap"
        );
        let mut o = QDict::new();
        o.put("read-only", "maybe");
        assert_eq!(
            err(g.reopen_node("n0", o, true)),
            "Parameter 'read-only' expects 'on' or 'off'"
        );
        let mut o = QDict::new();
        o.put("node-name", "other");
        assert_eq!(err(g.reopen_node("n0", o, true)), "Cannot change the option 'node-name'");
        // The same value as a string is no change.
        let mut o = QDict::new();
        o.put("size", "1048576");
        g.reopen_node("n0", o, true).unwrap();
    }

    #[test]
    fn read_only_with_writer_fails_and_rolls_back() {
        let g = graph();
        let n0 = g.find_node("n0").unwrap();
        let blk = BlockBackend::with_node(
            None,
            n0.clone(),
            BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE,
            BLK_PERM_ALL,
        )
        .unwrap();
        assert_eq!(
            err(n0.reopen_set_read_only(true)),
            "Read-only block node 'n0' cannot support read-write users"
        );
        assert!(!n0.read_only());
        assert_eq!(n0.meta.lock().unwrap().options.get("read-only"), Some(&QValue::Bool(false)));
        drop(blk);
        n0.reopen_set_read_only(true).unwrap();
    }

    #[test]
    fn inherited_child_follows_parent() {
        let g = BlockGraph::new();
        g.blockdev_add(opts(
            r#"{"driver": "raw", "node-name": "r",
                "file": {"driver": "null-co", "size": 1048576}}"#,
        ))
        .unwrap();
        let r = g.find_node("r").unwrap();
        let child = r.file();
        assert!(inherits_from(&child, &r));
        r.reopen_set_read_only(true).unwrap();
        assert!(r.read_only());
        assert!(child.read_only());
        r.reopen_set_read_only(false).unwrap();
        assert!(!child.read_only());

        // Without `file`, the child is reopened with no options at all.
        assert_eq!(
            err(g.blockdev_reopen(vec![opts(
                r#"{"driver": "raw", "node-name": "r", "file": "n0"}"#
            )])),
            "Cannot find device='n0' nor node-name='n0'"
        );
        let e = err(g.blockdev_reopen(vec![opts(
            r#"{"driver": "raw", "node-name": "r", "file": {"driver": "null-co"}}"#,
        )]));
        assert_eq!(e, "Option 'size' cannot be reset to its default value");
        g.blockdev_reopen(vec![opts(
            r#"{"driver": "raw", "node-name": "r", "read-only": true,
                "file": {"driver": "null-co", "size": 1048576}}"#,
        )])
        .unwrap();
        assert!(child.read_only());
    }

    #[test]
    fn referenced_child_is_not_reopened() {
        let g = graph();
        g.blockdev_add(opts(r#"{"driver": "raw", "node-name": "r", "file": "n0"}"#)).unwrap();
        let r = g.find_node("r").unwrap();
        let n0 = g.find_node("n0").unwrap();
        assert!(!inherits_from(&n0, &r));
        r.reopen_set_read_only(true).unwrap();
        assert!(r.read_only());
        assert!(!n0.read_only());
    }

    #[test]
    fn raw_window_and_file_replacement() {
        let g = graph();
        g.blockdev_add(opts(
            r#"{"driver": "raw", "node-name": "r", "file": "n0", "offset": 512, "size": 4096}"#,
        ))
        .unwrap();
        let r = g.find_node("r").unwrap();
        assert_eq!(r.getlength().unwrap(), 4096);
        // offset and size may be left out: they go back to the whole file.
        g.blockdev_reopen(vec![opts(r#"{"driver": "raw", "node-name": "r", "file": "n0"}"#)])
            .unwrap();
        assert_eq!(r.getlength().unwrap(), 1048576);
        assert_eq!(
            err(g.blockdev_reopen(vec![opts(
                r#"{"driver": "raw", "node-name": "r", "file": "n0", "size": 100}"#
            )])),
            "Specified size is not multiple of 512"
        );
        // A new file child.
        g.blockdev_reopen(vec![opts(r#"{"driver": "raw", "node-name": "r", "file": "n1"}"#)])
            .unwrap();
        assert_eq!(r.file().name, "n1");
        assert_eq!(r.getlength().unwrap(), 2097152);
        assert_eq!(g.find_node("n0").unwrap().parent_count(), 0);
        assert_eq!(
            err(g.blockdev_reopen(vec![opts(
                r#"{"driver": "raw", "node-name": "r", "file": "r"}"#
            )])),
            "Making 'r' a file child of 'r' would create a cycle"
        );
        assert_eq!(r.file().name, "n1");
    }

    #[test]
    fn file_replacement_rolls_back_on_perm_error() {
        let g = graph();
        g.blockdev_add(opts(r#"{"driver": "raw", "node-name": "r", "file": "n0"}"#)).unwrap();
        let r = g.find_node("r").unwrap();
        let n1 = g.find_node("n1").unwrap();
        let _blk = BlockBackend::with_node(
            None,
            r.clone(),
            BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE,
            BLK_PERM_ALL,
        )
        .unwrap();
        // Someone else holds n1 and does not share write.
        let _other = BlockBackend::with_node(
            None,
            n1.clone(),
            BLK_PERM_CONSISTENT_READ,
            BLK_PERM_CONSISTENT_READ,
        )
        .unwrap();
        let e = err(
            g.blockdev_reopen(vec![opts(r#"{"driver": "raw", "node-name": "r", "file": "n1"}"#)])
        );
        assert!(e.starts_with("Permission conflict on node 'n1'"), "{e}");
        assert_eq!(r.file().name, "n0");
        assert_eq!(n1.parent_count(), 1);
    }

    #[test]
    fn filter_children() {
        let g = graph();
        g.blockdev_add(opts(r#"{"driver": "blkdebug", "node-name": "d", "image": "n0"}"#)).unwrap();
        let mut o = QDict::new();
        o.put("backing", "n1");
        assert_eq!(
            err(g.reopen_node("d", o, true)),
            "'d' is a blkdebug filter node that does not support a backing child"
        );
        let mut o = QDict::new();
        o.put("image", "n1");
        assert_eq!(err(g.reopen_node("d", o, true)), "Cannot change the option 'image'");
    }
}
