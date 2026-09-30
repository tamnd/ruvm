// SPDX-License-Identifier: GPL-2.0-or-later

//! Opening by file name and option dictionary: the `bdrv_open()` path of block.c that
//! `-drive`, backing files and the tools use, as opposed to the typed options of
//! `blockdev-add`.
//!
//! `bdrv_open_inherit()` takes a file name (which may be a `json:{...}` pseudo-protocol) and a
//! flat option dictionary, works out the driver (`bdrv_fill_options()`), opens the `file`
//! child, probes the format when none was given (`find_image_format()`), opens the node and
//! then its backing file (`bdrv_open_backing_file()`). This module does the same, and then
//! turns the dictionary into typed [`BlockdevOptions`] for the drivers.
//!
//! Differences from QEMU:
//!
//! - Drivers read typed QAPI options rather than `QemuOpts`, so an option a driver does not
//!   know fails the options visitor ("Parameter 'x' is unexpected") rather than giving
//!   "Block format '%s' does not support the option '%s'".
//! - Children other than `file` and `backing` must have a `driver` when given as a
//!   dictionary; QEMU would open them by file name too.

use std::sync::Arc;

use ruvm_base::{Error, Result, report};
use ruvm_qapi::types::BlockdevOptions;
use ruvm_qapi::visit::{QObjectInputVisitor, Visit, qapi_bool_parse};
use ruvm_qapi::{QDict, QValue, json};

use crate::drivers;
use crate::graph::{BackingReq, BlockGraph, OpenCtx, Pending, child_ctx};
use crate::node::{BDRV_CHILD_IMAGE, Node, NodeFlags};
use crate::probe::find_image_format;

/// `qdict_crumple()` for keys with dots: `file.filename` becomes `file: {filename}`. Keys
/// without dots stay as they are.
pub(crate) fn crumple(d: QDict) -> Result<QDict> {
    let mut out = QDict::new();
    for (k, v) in d.iter_inserted() {
        let v = match v {
            QValue::Dict(sub) => QValue::Dict(crumple(sub.clone())?),
            v => v.clone(),
        };
        put_path(&mut out, k, v)?;
    }
    Ok(out)
}

fn put_path(d: &mut QDict, path: &str, v: QValue) -> Result<()> {
    match path.split_once('.') {
        None => match (d.get_mut(path), v) {
            (Some(QValue::Dict(old)), QValue::Dict(new)) => {
                for (k, v) in new.iter_inserted() {
                    put_path(old, k, v.clone())?;
                }
            }
            (Some(_), _) => {
                return Err(Error::generic(format!("Parameter '{path}' used inconsistently")));
            }
            (None, v) => d.put(path, v),
        },
        Some((head, rest)) => {
            if !d.contains_key(head) {
                d.put(head, QDict::new());
            }
            let Some(sub) = d.get_mut(head).and_then(QValue::as_dict_mut) else {
                return Err(Error::generic(format!("Parameter '{head}' used inconsistently")));
            };
            put_path(sub, rest, v)?;
        }
    }
    Ok(())
}

/// `qdict_join(dst, src, false)` on crumpled dictionaries: what `dst` has wins.
pub(crate) fn join(dst: &mut QDict, src: QDict) {
    for (k, v) in src.iter_inserted() {
        match (dst.get_mut(k), v) {
            (Some(QValue::Dict(a)), QValue::Dict(b)) => join(a, b.clone()),
            (Some(_), _) => {}
            (None, v) => dst.put(k, v.clone()),
        }
    }
}

/// Turns typed scalars into the strings the keyval visitor wants, so options from JSON and
/// from the command line can be mixed as they can in QEMU.
fn to_keyval(v: QValue) -> QValue {
    match v {
        QValue::Bool(b) => QValue::str(if b { "on" } else { "off" }),
        QValue::Int(i) => QValue::str(i.to_string()),
        QValue::Uint(u) => QValue::str(u.to_string()),
        QValue::Double(f) => QValue::str(json::format_g17(f)),
        QValue::List(l) => QValue::List(l.into_iter().map(to_keyval).collect()),
        QValue::Dict(d) => QValue::Dict(
            d.iter_inserted().map(|(k, v)| (k.to_string(), to_keyval(v.clone()))).collect(),
        ),
        v => v,
    }
}

/// A boolean option as either a JSON boolean or a command line string.
fn get_bool(d: &QDict, key: &str) -> Result<Option<bool>> {
    match d.get(key) {
        None => Ok(None),
        Some(QValue::Bool(b)) => Ok(Some(*b)),
        Some(QValue::Str(s)) => qapi_bool_parse(key, s).map(Some),
        Some(_) => {
            Err(Error::generic(format!("Invalid parameter type for '{key}', expected: boolean")))
        }
    }
}

/// The flags the node will have as far as its children care, from its options and what it
/// inherits.
fn flags_of(d: &QDict, ctx: &OpenCtx) -> Result<NodeFlags> {
    let inh = ctx.inherit;
    let cache = d.get("cache").and_then(QValue::as_dict);
    let cache_bool = |k: &str| match cache {
        Some(c) => get_bool(c, k),
        None => Ok(None),
    };
    Ok(NodeFlags {
        read_only: get_bool(d, "read-only")?.unwrap_or(inh.read_only),
        auto_read_only: get_bool(d, "auto-read-only")?.unwrap_or(inh.auto_read_only),
        direct: cache_bool("direct")?.unwrap_or(inh.direct),
        no_flush: cache_bool("no-flush")?.unwrap_or(inh.no_flush),
        force_share: get_bool(d, "force-share")?.unwrap_or(inh.force_share),
        unmap: true,
        ..NodeFlags::default()
    })
}

/// `parse_json_filename()`.
fn parse_json_filename(filename: &str) -> Result<QDict> {
    let body = filename.strip_prefix("json:").expect("checked by the caller");
    let v = json::from_str(body).map_err(|e| e.prepend("Could not parse the JSON options: "))?;
    match v {
        QValue::Dict(d) => crumple(d),
        _ => Err(Error::generic("Invalid JSON object given")),
    }
}

/// `path_is_absolute()`.
fn path_is_absolute(p: &str) -> bool {
    #[cfg(windows)]
    {
        let b = p.as_bytes();
        if b.len() >= 2 && b[1] == b':' && b[0].is_ascii_alphabetic() {
            return true;
        }
        if p.starts_with('\\') {
            return true;
        }
    }
    p.starts_with('/')
}

/// `bdrv_get_full_backing_filename()`: a relative backing file name is relative to the
/// directory of the image that names it.
pub(crate) fn full_backing_filename(bs: &Node, backing: &str) -> Result<String> {
    if backing.is_empty()
        || path_is_absolute(backing)
        || drivers::protocol_prefix(backing).is_some()
    {
        return Ok(backing.to_string());
    }
    bs.refresh_filename();
    let base = bs.meta.lock().unwrap().filename.clone();
    // bdrv_dirname(): only plain file names have a directory.
    if base.is_empty() || base.starts_with("json:") {
        return Err(Error::generic(format!("Cannot use relative backing file names for '{base}'")));
    }
    let dir = match base.rfind('/') {
        Some(i) => &base[..=i],
        None => match drivers::protocol_prefix(&base) {
            Some(p) => &base[..=p.len()],
            None => "",
        },
    };
    Ok(format!("{dir}{backing}"))
}

impl BlockGraph {
    /// `bdrv_open()`: opens `filename` with `options` as the top of a new tree and puts the
    /// nodes in the name table. Returns the root, the typed options it was opened with and
    /// the warnings printed on the way.
    pub(crate) fn open_nodes_qdict(
        &self,
        filename: Option<&str>,
        options: QDict,
        ctx: OpenCtx,
    ) -> Result<(Arc<Node>, BlockdevOptions, Vec<String>)> {
        let mut pending = Pending::default();
        let (root, opts) = self.open_qdict_opts(filename, options, ctx, &mut pending, true)?;
        let (root, warnings) = self.commit(root, pending, ctx)?;
        Ok((root, opts, warnings))
    }

    /// `bdrv_open_inherit()` without the name table.
    pub(crate) fn open_qdict(
        &self,
        filename: Option<&str>,
        options: QDict,
        ctx: OpenCtx,
        pending: &mut Pending,
        parse_filename: bool,
    ) -> Result<Arc<Node>> {
        Ok(self.open_qdict_opts(filename, options, ctx, pending, parse_filename)?.0)
    }

    fn open_qdict_opts(
        &self,
        filename: Option<&str>,
        options: QDict,
        mut ctx: OpenCtx,
        pending: &mut Pending,
        parse_filename: bool,
    ) -> Result<(Arc<Node>, BlockdevOptions)> {
        let mut options = crumple(options)?;
        let mut filename = filename.map(str::to_string);

        // json: syntax counts as explicit options, with lower priority than the ones given.
        if parse_filename && filename.as_deref().is_some_and(|f| f.starts_with("json:")) {
            let j = parse_json_filename(filename.as_deref().unwrap_or_default())?;
            join(&mut options, j);
            filename = None;
        }

        // bdrv_fill_options().
        let mut protocol = ctx.protocol;
        let drvname = options.get_str("driver").map(str::to_string);
        let mut drv = None;
        if let Some(n) = &drvname {
            let Some(d) = drivers::find_format(n) else {
                return Err(Error::generic(format!("Unknown driver '{n}'")));
            };
            protocol = d.protocol_name.is_some();
            drv = Some(d);
        }
        let mut parse = false;
        if protocol {
            if let Some(f) = &filename {
                if options.contains_key("filename") {
                    return Err(Error::generic(
                        "Can't specify 'file' and 'filename' options at the same time",
                    ));
                }
                options.put("filename", f.as_str());
                parse = parse_filename;
            }
        }
        let opt_filename = options.get_str("filename").map(str::to_string);
        if drvname.is_none() && protocol {
            let Some(f) = &opt_filename else {
                return Err(Error::generic("Must specify either driver or file"));
            };
            let d = drivers::find_protocol(f, parse)?;
            options.put("driver", d.format_name);
            drv = Some(d);
        }
        if let (Some(d), true) = (drv, parse) {
            if let (Some(pf), Some(f)) = (d.parse_filename, &opt_filename) {
                pf(f, &mut options)?;
                if !d.needs_filename {
                    options.remove("filename");
                }
            }
        }

        // `backing: null`, and the deprecated empty string.
        let backing = match options.remove("backing") {
            None => BackingReq::Options(QDict::new()),
            Some(QValue::Null) => BackingReq::None,
            Some(QValue::Str(s)) if s.is_empty() => {
                report::warn_report(
                    "Use of \"backing\": \"\" is deprecated; use \"backing\": null instead",
                );
                BackingReq::None
            }
            Some(QValue::Str(s)) => BackingReq::Ref(s),
            Some(QValue::Dict(d)) => BackingReq::Options(d),
            Some(_) => {
                return Err(Error::generic(
                    "Invalid parameter type for 'backing', expected: string",
                ));
            }
        };

        // Open the image file without the format layer, for probing and as `file`.
        let mut file = None;
        if !protocol {
            let flags = flags_of(&options, &ctx)?;
            let mut cctx = child_ctx(&flags, true, false, BDRV_CHILD_IMAGE);
            cctx.inherit.native_aio = ctx.inherit.native_aio;
            let node = match options.remove("file") {
                Some(QValue::Str(r)) => {
                    if filename.is_some() {
                        return Err(Error::generic(
                            "Cannot reference an existing block device with additional \
                             options or a new filename",
                        ));
                    }
                    Some(self.lookup_pending(&r, pending)?)
                }
                Some(QValue::Dict(d)) => {
                    Some(self.open_qdict(filename.as_deref(), d, cctx, pending, true)?)
                }
                None if filename.is_some() => {
                    Some(self.open_qdict(filename.as_deref(), QDict::new(), cctx, pending, true)?)
                }
                None => None,
                Some(_) => {
                    return Err(Error::generic(
                        "Invalid parameter type for 'file', expected: string",
                    ));
                }
            };
            if let Some(n) = &node {
                options.put("file", n.name.as_str());
            }
            file = node;
        }

        // Image format probing.
        ctx.probed = drv.is_none();
        if drv.is_none() {
            let Some(f) = &file else {
                return Err(Error::generic("Must specify either driver or file"));
            };
            let d = find_image_format(f, filename.as_deref())?;
            options.put("driver", d.format_name);
        }

        let mut v = QObjectInputVisitor::new_keyval(to_keyval(QValue::Dict(options)));
        let mut opts = BlockdevOptions::default();
        BlockdevOptions::visit(&mut v, None, &mut opts)?;
        let node = self.open_with(opts.clone(), ctx, pending, backing)?;
        Ok((node, opts))
    }

    /// `bdrv_open()` for the tools and tests: opens `filename` (which may be `json:`) with
    /// `options` (flat or nested, strings or typed) and returns the root node's name.
    pub fn open_image(&self, filename: Option<&str>, options: QDict) -> Result<String> {
        let ctx = OpenCtx { root: true, ..OpenCtx::default() };
        let (root, _, _) = self.open_nodes_qdict(filename, options, ctx)?;
        Ok(root.name.clone())
    }
}
