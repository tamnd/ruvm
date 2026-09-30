// SPDX-License-Identifier: GPL-2.0-or-later

//! The driver table, `bdrv_drivers` in block.c, and what a driver's open function gets.
//!
//! To add a driver, define a `static` [`DriverDef`] next to it and add one line to
//! [`DRIVERS`]. `blockdev-add`, `-drive`, opening by file name and format probing all find
//! drivers through this table.
//!
//! A driver's open function gets an [`OpenArgs`] and the QAPI options of its branch of
//! `BlockdevOptions`. It opens its children through [`OpenArgs::open_child`], which applies
//! the options children inherit from their parent and records the child with its role, and
//! returns the driver state. The node is made from that after the function returns, and then
//! the generic code opens the backing file the driver recorded with
//! [`OpenArgs::set_backing_file`] or that the options name.

use std::sync::Arc;

use ruvm_base::{Error, Result};
use ruvm_qapi::QDict;
use ruvm_qapi::types::{BlockdevCreateOptionsU, BlockdevOptionsU, BlockdevRef, BlockdevRefOrNull};

use crate::graph::{BlockGraph, OpenCtx, Pending};
use crate::node::{Driver, Node, NodeFlags, NodeMeta};

/// Opens a node of the driver: `.bdrv_open` or `.bdrv_file_open`.
pub(crate) type OpenFn = fn(&mut OpenArgs<'_>, BlockdevOptionsU) -> Result<Box<dyn Driver>>;

/// `.bdrv_probe`: how sure the driver is that the first bytes of a file (up to 2048, fewer
/// for short files) are its format, from 0 (not at all) to 100.
pub(crate) type ProbeFn = fn(buf: &[u8], filename: Option<&str>) -> i32;

/// `.bdrv_probe_device`: how sure the driver is that `filename` is a host device it handles.
pub(crate) type ProbeDeviceFn = fn(filename: &str) -> i32;

/// `.bdrv_parse_filename`: turns a file name into options of the driver. The file name
/// without any `protocol:` prefix handling is passed in; `options` already has what the user
/// gave.
pub(crate) type ParseFilenameFn = fn(filename: &str, options: &mut QDict) -> Result<()>;

/// `.bdrv_co_create_opts`: creates an image from `qemu-img create` style options. `options`
/// holds the `-o` options as strings (and `size`); a driver takes out what it knows. What is
/// left is reported as unsupported by the caller.
pub(crate) type CreateOptsFn = fn(filename: &str, options: &mut QDict) -> Result<()>;

/// `.bdrv_co_create`: `blockdev-create` for the driver.
pub(crate) type CreateFn = fn(graph: &BlockGraph, options: BlockdevCreateOptionsU) -> Result<()>;

/// A block driver, the static part of `BlockDriver`.
#[derive(Debug)]
pub(crate) struct DriverDef {
    /// `format_name`, what `driver` says.
    pub format_name: &'static str,
    /// `protocol_name`, for protocol drivers: the `name:` prefix of file names they take.
    pub protocol_name: Option<&'static str>,
    pub open: OpenFn,
    pub probe: Option<ProbeFn>,
    pub probe_device: Option<ProbeDeviceFn>,
    pub parse_filename: Option<ParseFilenameFn>,
    pub create_opts: Option<CreateOptsFn>,
    pub create: Option<CreateFn>,
    /// `bdrv_needs_filename`: `filename` stays in the options after `parse_filename`.
    pub needs_filename: bool,
    /// `is_filter`.
    pub is_filter: bool,
    /// `is_format`.
    pub is_format: bool,
    /// `supports_backing`.
    pub supports_backing: bool,
    /// `filtered_child_is_backing`: the filtered child of the filter is called `backing`.
    pub filtered_child_is_backing: bool,
    /// `mutable_opts`: the driver options a reopen may leave out to reset them to their
    /// defaults. Any other option the node was opened with must be given again.
    pub mutable_opts: &'static [&'static str],
    /// `strong_runtime_opts`: the driver options that change what the node reads and
    /// writes. A name ending in `.` stands for every option with that prefix.
    pub strong_runtime_opts: &'static [&'static str],
}

impl DriverDef {
    const fn new(format_name: &'static str, open: OpenFn) -> Self {
        DriverDef {
            format_name,
            protocol_name: None,
            open,
            probe: None,
            probe_device: None,
            parse_filename: None,
            create_opts: None,
            create: None,
            needs_filename: false,
            is_filter: false,
            is_format: false,
            supports_backing: false,
            filtered_child_is_backing: false,
            mutable_opts: &[],
            strong_runtime_opts: &[],
        }
    }

    /// A format driver.
    pub(crate) const fn format(name: &'static str, open: OpenFn) -> Self {
        let mut d = Self::new(name, open);
        d.is_format = true;
        d
    }

    /// A protocol driver whose file names start with `protocol:`.
    pub(crate) const fn protocol(name: &'static str, protocol: &'static str, open: OpenFn) -> Self {
        let mut d = Self::new(name, open);
        d.protocol_name = Some(protocol);
        d
    }

    /// A filter driver.
    pub(crate) const fn filter(name: &'static str, open: OpenFn) -> Self {
        let mut d = Self::new(name, open);
        d.is_filter = true;
        d
    }

    /// Also takes file names that start with `protocol:`.
    pub(crate) const fn with_protocol(mut self, protocol: &'static str) -> Self {
        self.protocol_name = Some(protocol);
        self
    }

    pub(crate) const fn with_probe(mut self, f: ProbeFn) -> Self {
        self.probe = Some(f);
        self
    }

    #[cfg_attr(not(unix), allow(dead_code, reason = "only the host file drivers use it"))]
    pub(crate) const fn with_probe_device(mut self, f: ProbeDeviceFn) -> Self {
        self.probe_device = Some(f);
        self
    }

    pub(crate) const fn with_parse_filename(mut self, f: ParseFilenameFn) -> Self {
        self.parse_filename = Some(f);
        self
    }

    pub(crate) const fn with_create_opts(mut self, f: CreateOptsFn) -> Self {
        self.create_opts = Some(f);
        self
    }

    pub(crate) const fn with_create(mut self, f: CreateFn) -> Self {
        self.create = Some(f);
        self
    }

    /// The driver needs `filename` in its options, `bdrv_needs_filename`.
    #[cfg_attr(not(unix), allow(dead_code, reason = "only the host file drivers use it"))]
    pub(crate) const fn with_needs_filename(mut self) -> Self {
        self.needs_filename = true;
        self
    }

    /// Images of this format can have a backing file.
    #[allow(dead_code, reason = "for the format drivers with backing files, qcow2 first")]
    pub(crate) const fn with_backing(mut self) -> Self {
        self.supports_backing = true;
        self
    }

    /// The options a reopen may reset by leaving them out, `mutable_opts`.
    pub(crate) const fn with_mutable_opts(mut self, opts: &'static [&'static str]) -> Self {
        self.mutable_opts = opts;
        self
    }

    /// The options that change the data of the node, `strong_runtime_opts`.
    pub(crate) const fn with_strong_opts(mut self, opts: &'static [&'static str]) -> Self {
        self.strong_runtime_opts = opts;
        self
    }

    /// A filter whose child is called `backing`.
    #[allow(dead_code, reason = "for the filters block jobs insert")]
    pub(crate) const fn with_filtered_backing(mut self) -> Self {
        self.filtered_child_is_backing = true;
        self.supports_backing = true;
        self
    }
}

/// Every driver, one line each.
pub(crate) static DRIVERS: &[&DriverDef] = &[
    #[cfg(unix)]
    &crate::file::FILE,
    // host_cdrom before host_device: on equal probe scores the first driver wins.
    #[cfg(target_os = "linux")]
    &crate::protocol::host::HOST_CDROM,
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    &crate::protocol::host::HOST_DEVICE,
    &crate::raw::RAW,
    &crate::filter::null::NULL_CO,
    &crate::filter::null::NULL_AIO,
    &crate::filter::blkdebug::BLKDEBUG,
    &crate::filter::copy_on_read::COPY_ON_READ,
    &crate::filter::compress::COMPRESS,
    &crate::filter::preallocate::PREALLOCATE,
    &crate::filter::blkverify::BLKVERIFY,
    &crate::filter::throttle::THROTTLE,
    &crate::nbd::NBD,
    &crate::nbd::NBD_TCP,
    &crate::nbd::NBD_UNIX,
    &crate::luks::LUKS,
];

/// `bdrv_find_format()`.
pub(crate) fn find_format(name: &str) -> Option<&'static DriverDef> {
    DRIVERS.iter().copied().find(|d| d.format_name == name)
}

/// `bdrv_find_protocol()`: the driver for a file name, from its `protocol:` prefix, or a host
/// device driver that claims it, or `file`.
pub(crate) fn find_protocol(
    filename: &str,
    allow_protocol_prefix: bool,
) -> Result<&'static DriverDef> {
    // A host device driver claims its files first, `bdrv_find_protocol()` does that before
    // looking at the prefix.
    let mut best: Option<(&'static DriverDef, i32)> = None;
    for d in DRIVERS {
        if let Some(p) = d.probe_device {
            let score = p(filename);
            if score > best.map_or(0, |b| b.1) {
                best = Some((d, score));
            }
        }
    }
    if let Some((d, _)) = best {
        return Ok(d);
    }
    let prefix = protocol_prefix(filename);
    let Some(prefix) = prefix.filter(|_| allow_protocol_prefix) else {
        return find_format("file").ok_or_else(|| Error::generic("Unknown protocol 'file'"));
    };
    DRIVERS
        .iter()
        .copied()
        .find(|d| d.protocol_name == Some(prefix))
        .ok_or_else(|| Error::generic(format!("Unknown protocol '{prefix}'")))
}

/// `path_has_protocol()` and the prefix itself: the part before the first `:` if there is no
/// `/` before it. Windows drive letters do not count.
pub(crate) fn protocol_prefix(filename: &str) -> Option<&str> {
    let colon = filename.find(':')?;
    let slash = filename.find('/');
    if slash.is_some_and(|s| s < colon) {
        return None;
    }
    #[cfg(windows)]
    if colon == 1 && filename.as_bytes()[0].is_ascii_alphabetic() {
        return None;
    }
    Some(&filename[..colon])
}

/// What a driver's open function works with.
pub(crate) struct OpenArgs<'a> {
    pub(crate) graph: &'a BlockGraph,
    pub(crate) pending: &'a mut Pending,
    pub(crate) ctx: OpenCtx,
    /// The driver being opened.
    pub(crate) def: &'static DriverDef,
    /// The name the node will have.
    #[allow(dead_code, reason = "for drivers that name their node in messages")]
    pub node_name: String,
    /// The flags the node will have. A driver may change them, `read_only` in particular.
    pub flags: NodeFlags,
    /// The children opened so far: name, node, role.
    pub(crate) children: Vec<(String, Arc<Node>, u32)>,
    /// File names and backing file information for the node.
    pub meta: NodeMeta,
    /// The `backing` option, if the driver takes one. `None` when the option was not given.
    pub(crate) backing: Option<BlockdevRefOrNull>,
}

impl OpenArgs<'_> {
    /// `bdrv_open_child()`: opens (or looks up) the child called `name` and records it with
    /// `role`. The child inherits the options its parent has and it does not set.
    pub(crate) fn open_child(
        &mut self,
        r: BlockdevRef,
        name: &str,
        role: u32,
    ) -> Result<Arc<Node>> {
        let node = self.graph.open_child_ref(r, self.child_ctx(role), self.pending)?;
        self.children.push((name.to_string(), node.clone(), role));
        Ok(node)
    }

    /// `bdrv_open_child()` for an optional child: `None` is fine when `allow_none`, and
    /// otherwise the error QEMU gives for a missing child.
    pub(crate) fn open_child_opt(
        &mut self,
        r: Option<BlockdevRef>,
        name: &str,
        role: u32,
        allow_none: bool,
    ) -> Result<Option<Arc<Node>>> {
        match r {
            Some(r) => self.open_child(r, name, role).map(Some),
            None if allow_none => Ok(None),
            None => Err(Error::generic(format!("A block device must be specified for \"{name}\""))),
        }
    }

    /// Records a child a driver opened some other way.
    #[allow(dead_code, reason = "for drivers that open children without open_child()")]
    pub(crate) fn add_child(&mut self, name: &str, node: Arc<Node>, role: u32) {
        self.children.push((name.to_string(), node, role));
    }

    /// What a child inherits, `bdrv_inherited_options()` of `child_of_bds` for `role`.
    pub(crate) fn child_ctx(&self, role: u32) -> OpenCtx {
        let mut ctx = crate::graph::child_ctx(
            &self.flags,
            self.def.is_format,
            self.def.protocol_name.is_some(),
            role,
        );
        ctx.inherit.native_aio = self.ctx.inherit.native_aio;
        ctx
    }

    /// The backing file name and format the image header records, for the generic code to
    /// open unless the options say otherwise.
    #[allow(dead_code, reason = "for the format drivers with backing files, qcow2 first")]
    pub(crate) fn set_backing_file(&mut self, file: &str, format: Option<&str>) {
        self.meta.backing_file = file.to_string();
        self.meta.backing_format = format.unwrap_or_default().to_string();
    }

    /// The `backing` option of drivers that take one.
    #[allow(dead_code, reason = "for the format drivers with backing files, qcow2 first")]
    pub(crate) fn set_backing_option(&mut self, backing: Option<BlockdevRefOrNull>) {
        self.backing = backing;
    }

    /// Prints `msg` to stderr as it is, the way QEMU's `fprintf(stderr, ...)` warnings are,
    /// and keeps it for callers that show warnings again.
    pub(crate) fn warn(&mut self, msg: String) {
        eprintln!("{msg}");
        self.pending.warnings.push(msg);
    }

    /// `bs->probed`: the format was probed rather than given.
    pub(crate) fn is_probed(&self) -> bool {
        self.ctx.probed
    }

    /// `bdrv_apply_auto_read_only()`: a writable node that may fall back to read-only
    /// becomes read-only; otherwise this is an error.
    #[allow(dead_code, reason = "for format drivers that cannot write some images")]
    pub(crate) fn apply_auto_read_only(&mut self, errmsg: Option<&str>) -> Result<()> {
        if self.flags.read_only {
            return Ok(());
        }
        if !self.flags.auto_read_only {
            return Err(Error::generic(errmsg.unwrap_or("Image is read-only")));
        }
        self.flags.read_only = true;
        Ok(())
    }
}
