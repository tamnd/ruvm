// SPDX-License-Identifier: GPL-2.0-or-later

//! What qemu-img and qemu-io reach into the block layer for: `blk_new_open()` with the
//! `BDRV_O_*` flags, `bdrv_query_block_graph_info()`, block status, snapshots by name, driver
//! information, `bdrv_amend_options()`, `bdrv_measure()` and the driver table as
//! `bdrv_iterate_format()` and `bdrv_find_format()` show it.
//!
//! Nodes are named by their node name, as everywhere in [`BlockGraph`]; the nodes an open makes
//! get generated names and stay alive as long as the [`BlockBackend`] on top of them.
//!
//! Also here: the `create_opts` and `amend_opts` lists of `raw`, `file` and `luks`, the `raw`
//! `.bdrv_co_create_opts`, and the `.bdrv_measure` functions of `raw` and `luks`, which the
//! drivers point to.
//!
//! Differences from QEMU:
//!
//! - Errors carry messages, not errno values, so callers that print `strerror(-ret)` in QEMU
//!   print the message of the error instead.
//! - `bdrv_query_block_graph_info()` leaves out `limits` unless asked, as QEMU does, but gets
//!   them from the same place as `query-named-block-nodes`.

use std::io;
use std::sync::Arc;

use ruvm_base::{Error, Result};
use ruvm_qapi::opts::{QemuOptDesc, QemuOptType};
use ruvm_qapi::types::{
    BlockChildInfo, BlockGraphInfo, BlockMeasureInfo, BlockdevAmendOptions, ImageInfoSpecific,
    PreallocMode, SnapshotInfo,
};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};
use ruvm_qapi::{QDict, QValue};

use crate::backend::BlockBackend;
use crate::drivers;
use crate::graph::{BlockGraph, Inherited, OpenCtx};
use crate::node::{self, Node, SnapshotEntry, errno};
use crate::perm::{
    BLK_PERM_ALL, BLK_PERM_CONSISTENT_READ, BLK_PERM_RESIZE, BLK_PERM_WRITE,
    BLK_PERM_WRITE_UNCHANGED,
};

/// `BDRV_BLOCK_DATA`: reads return data from the node.
pub const BDRV_BLOCK_DATA: u32 = node::BDRV_BLOCK_DATA;
/// `BDRV_BLOCK_ZERO`: reads return zeroes.
pub const BDRV_BLOCK_ZERO: u32 = node::BDRV_BLOCK_ZERO;
/// `BDRV_BLOCK_OFFSET_VALID`: the offset in `file` is known.
pub const BDRV_BLOCK_OFFSET_VALID: u32 = node::BDRV_BLOCK_OFFSET_VALID;
/// `BDRV_BLOCK_RAW`.
pub const BDRV_BLOCK_RAW: u32 = node::BDRV_BLOCK_RAW;
/// `BDRV_BLOCK_ALLOCATED`: the content is decided by this layer.
pub const BDRV_BLOCK_ALLOCATED: u32 = node::BDRV_BLOCK_ALLOCATED;
/// `BDRV_BLOCK_EOF`: the range reaches the end of the node.
pub const BDRV_BLOCK_EOF: u32 = node::BDRV_BLOCK_EOF;
/// `BDRV_BLOCK_COMPRESSED`.
pub const BDRV_BLOCK_COMPRESSED: u32 = node::BDRV_BLOCK_COMPRESSED;

/// The `BDRV_O_*` flags of `blk_new_open()` that the tools use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OpenFlags {
    /// `BDRV_O_RDWR`.
    pub rdwr: bool,
    /// `BDRV_O_NO_BACKING`.
    pub no_backing: bool,
    /// `BDRV_O_NO_IO`: only the metadata is looked at.
    pub no_io: bool,
    /// `BDRV_O_CHECK`: the image is opened for `qemu-img check`.
    pub check: bool,
    /// `BDRV_O_NOCACHE`.
    pub nocache: bool,
    /// `BDRV_O_NO_FLUSH`.
    pub no_flush: bool,
    /// `BDRV_O_UNMAP`.
    pub unmap: bool,
    /// `BDRV_O_RESIZE`: the backend takes the resize permission.
    pub resize: bool,
    /// `BDRV_O_PROTOCOL`: the file name opens a protocol node, without probing.
    pub protocol: bool,
    /// `BDRV_O_NO_SHARE`.
    pub no_share: bool,
    /// `BDRV_O_COPY_ON_READ`.
    pub copy_on_read: bool,
    /// `BDRV_O_NATIVE_AIO`: `file` nodes default to `aio=native`.
    pub native_aio: bool,
}

/// What `bdrv_block_status_above()` reports for the start of a range.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockStatusInfo {
    /// The `BDRV_BLOCK_*` bits.
    pub ret: u32,
    /// How many bytes from the start of the range have that status.
    pub pnum: u64,
    /// With [`BDRV_BLOCK_OFFSET_VALID`], where the range is in `file`.
    pub map: u64,
    /// With [`BDRV_BLOCK_OFFSET_VALID`], the node name of the node the data is in.
    pub file: Option<String>,
    /// The layer the status comes from, 1 for the node itself.
    pub depth: u32,
}

/// `BlockDriverInfo`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DriverInfo {
    pub cluster_size: u64,
    pub subcluster_size: u64,
    pub vm_state_offset: i64,
    pub is_dirty: bool,
    pub needs_compressed_writes: bool,
}

/// The parts of a node the tools print or decide on.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NodeDetails {
    /// `bs->drv->format_name`.
    pub driver: String,
    /// `bs->filename`, refreshed.
    pub filename: String,
    /// `bs->exact_filename`.
    pub exact_filename: String,
    /// `bs->backing_file`, the name in the image header.
    pub backing_file: String,
    /// `bs->backing_format`.
    pub backing_format: String,
    /// `bs->auto_backing_file`.
    pub auto_backing_file: String,
    pub encrypted: bool,
    pub read_only: bool,
    /// `drv->is_format`.
    pub is_format: bool,
    /// `drv->is_filter`.
    pub is_filter: bool,
    /// `drv->supports_backing`.
    pub supports_backing: bool,
    /// `bs->total_sectors * 512`, or the error of `bdrv_getlength()`.
    pub length: Option<u64>,
}

/// `BlockLimits`, the parts the tools use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Limits {
    pub request_alignment: u32,
    pub opt_transfer: u32,
    pub max_transfer: u32,
    pub pdiscard_alignment: u32,
    pub max_pdiscard: u64,
}

/// A named dirty bitmap as `qemu-img convert --bitmaps` and `qemu-img bitmap` see it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BitmapDetails {
    pub name: String,
    pub persistent: bool,
    pub inconsistent: bool,
    pub granularity: u32,
    pub enabled: bool,
}

fn status_info(st: node::BlockStatus, depth: u32) -> BlockStatusInfo {
    BlockStatusInfo {
        ret: st.ret,
        pnum: st.pnum,
        map: st.map,
        file: st.file.map(|f| f.name.clone()),
        depth,
    }
}

fn snapshot_info(sn: SnapshotEntry) -> SnapshotInfo {
    SnapshotInfo {
        id: sn.id_str,
        name: sn.name,
        vm_state_size: sn.vm_state_size as i64,
        date_sec: i64::from(sn.date_sec),
        date_nsec: i64::from(sn.date_nsec),
        vm_clock_sec: (sn.vm_clock_nsec / 1_000_000_000) as i64,
        vm_clock_nsec: (sn.vm_clock_nsec % 1_000_000_000) as i64,
        icount: sn.icount.map(|i| i as i64),
    }
}

/// `block_driver_can_compress()` of the format `name` before there is a node of it: the
/// formats whose QEMU driver has `.bdrv_co_pwritev_compressed_part`. Whether the ported
/// driver writes compressed data is checked on the node with [`BlockGraph::can_compress`].
pub fn format_can_compress(name: &str) -> bool {
    format_exists(name) && matches!(name, "qcow" | "qcow2" | "vmdk" | "compress")
}

/// `bdrv_iterate_format()`: the names of every driver, sorted, for `qemu-img --help`.
pub fn format_names() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = drivers::DRIVERS.iter().map(|d| d.format_name).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// `bdrv_find_format()`: whether there is a driver called `name`.
pub fn format_exists(name: &str) -> bool {
    drivers::find_format(name).is_some()
}

/// `drv->protocol_name` of the driver called `name`, which qemu-io `info` prints.
pub fn protocol_name(name: &str) -> Option<&'static str> {
    drivers::find_format(name).and_then(|d| d.protocol_name)
}

/// Whether the driver `name` can create images, `drv->create_opts != NULL`.
pub fn format_can_create(name: &str) -> bool {
    drivers::find_format(name).is_some_and(|d| d.create_opts.is_some())
}

/// Whether images of the format `name` can have a backing file.
pub fn format_supports_backing(name: &str) -> bool {
    drivers::find_format(name).is_some_and(|d| d.supports_backing)
}

/// `bdrv_find_protocol(filename, true)`: the name of the protocol driver for `filename`.
pub fn find_protocol(filename: &str) -> Result<&'static str> {
    drivers::find_protocol(filename, true).map(|d| d.format_name)
}

/// `create_opts` of the driver `name`; `None` when the driver cannot create images. A driver
/// that creates images but has not ported its list gives an empty one.
pub fn create_opts_list(name: &str) -> Option<&'static [QemuOptDesc]> {
    let d = drivers::find_format(name)?;
    d.create_opts.map(|_| d.create_opts_list)
}

/// `amend_opts` of the driver `name`; `None` when the driver cannot amend images.
pub fn amend_opts_list(name: &str) -> Option<&'static [QemuOptDesc]> {
    let d = drivers::find_format(name)?;
    (!d.amend_opts_list.is_empty() || d.amend_opts.is_some()).then_some(d.amend_opts_list)
}

/// Whether the driver `name` measures images, `drv->bdrv_measure != NULL`.
pub fn format_can_measure(name: &str) -> bool {
    drivers::find_format(name).is_some_and(|d| d.measure.is_some())
}

/// `path_has_protocol()`.
pub fn path_has_protocol(path: &str) -> bool {
    drivers::protocol_prefix(path).is_some()
}

/// `bdrv_get_full_backing_filename_from_filename()`: `backing` relative to the image
/// `backed` that names it.
pub fn full_backing_filename_from_filename(backed: &str, backing: &str) -> Result<String> {
    if backing.is_empty() || path_has_protocol(backing) || backing.starts_with('/') {
        return Ok(backing.to_string());
    }
    if backed.is_empty() || backed.starts_with("json:") {
        return Err(Error::generic(format!(
            "Cannot use relative backing file names for '{backed}'"
        )));
    }
    // path_combine()
    let dir = match backed.rfind('/') {
        Some(i) => &backed[..=i],
        None => match drivers::protocol_prefix(backed) {
            Some(p) => &backed[..=p.len()],
            None => "",
        },
    };
    Ok(format!("{dir}{backing}"))
}

impl BlockGraph {
    /// `blk_new_open()`: opens `filename` with `options` and `flags` and puts a backend on the
    /// root node with the permissions the flags ask for.
    pub fn blk_new_open(
        &self,
        filename: Option<&str>,
        mut options: QDict,
        flags: OpenFlags,
    ) -> Result<Arc<BlockBackend>> {
        let mut perm = 0;
        let mut shared = BLK_PERM_ALL;
        if !flags.no_io {
            perm |= BLK_PERM_CONSISTENT_READ;
            if flags.rdwr {
                perm |= BLK_PERM_WRITE;
            }
        }
        if flags.resize {
            perm |= BLK_PERM_RESIZE;
        }
        if flags.no_share {
            shared = BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE_UNCHANGED;
        }
        if flags.no_backing && !options.contains_key("backing") {
            options.put("backing", QValue::Null);
        }
        if flags.copy_on_read && !options.contains_key("copy-on-read") {
            options.put("copy-on-read", true);
        }
        let ctx = OpenCtx {
            root: false,
            inherit: Inherited {
                read_only: !flags.rdwr,
                direct: flags.nocache,
                no_flush: flags.no_flush,
                unmap: flags.unmap,
                no_io: flags.no_io,
                check: flags.check,
                native_aio: flags.native_aio,
                ..Inherited::default()
            },
            protocol: flags.protocol,
            ..OpenCtx::default()
        };
        // The open already printed its warnings, the way QEMU's fprintf() does.
        let (bs, _, _warnings) = self.open_nodes_qdict(filename, options, ctx)?;
        Ok(Arc::new(BlockBackend::with_node(None, bs, perm, shared)?))
    }

    /// `bdrv_query_block_graph_info()`: the node and everything below it.
    pub fn query_block_graph_info(&self, node: &str, limits: bool) -> Result<BlockGraphInfo> {
        let bs = self.lookup_bs(node)?;
        Self::graph_info(&bs, limits)
    }

    fn graph_info(bs: &Arc<Node>, limits: bool) -> Result<BlockGraphInfo> {
        let i = bs.query_image_info(true)?;
        let mut children = Vec::new();
        for c in bs.children() {
            children.push(BlockChildInfo {
                name: c.name.clone(),
                info: Self::graph_info(&c.node, limits)?,
            });
        }
        Ok(BlockGraphInfo {
            filename: i.filename,
            format: i.format,
            dirty_flag: i.dirty_flag,
            actual_size: i.actual_size,
            virtual_size: i.virtual_size,
            cluster_size: i.cluster_size,
            encrypted: i.encrypted,
            compressed: i.compressed,
            backing_filename: i.backing_filename,
            full_backing_filename: i.full_backing_filename,
            backing_filename_format: i.backing_filename_format,
            snapshots: i.snapshots,
            limits: if limits { i.limits } else { None },
            format_specific: i.format_specific,
            children,
        })
    }

    /// The node details of `node`.
    pub fn node_details(&self, node: &str) -> Result<NodeDetails> {
        let bs = self.lookup_bs(node)?;
        bs.refresh_filename();
        let length = bs.getlength().ok();
        let m = bs.meta.lock().unwrap();
        Ok(NodeDetails {
            driver: bs.driver_name.to_string(),
            filename: m.filename.clone(),
            exact_filename: m.exact_filename.clone(),
            backing_file: m.backing_file.clone(),
            backing_format: m.backing_format.clone(),
            auto_backing_file: m.auto_backing_file.clone(),
            encrypted: m.encrypted,
            read_only: bs.read_only(),
            is_format: bs.is_format(),
            is_filter: bs.is_filter(),
            supports_backing: bs.def.is_some_and(|d| d.supports_backing),
            length,
        })
    }

    /// The `BDRV_O_NOCACHE` and `BDRV_O_NO_FLUSH` bits of `bs->open_flags` of `node`, which
    /// qemu-io `reopen` starts from.
    pub fn node_cache_flags(&self, node: &str) -> Result<(bool, bool)> {
        let bs = self.lookup_bs(node)?;
        let f = bs.flags();
        Ok((f.direct, f.no_flush))
    }

    /// `bdrv_getlength()` of `node`.
    pub fn node_getlength(&self, node: &str) -> io::Result<u64> {
        let bs = self.lookup_bs(node).map_err(|_| errno(node::ENOMEDIUM))?;
        bs.getlength()
    }

    /// `bdrv_get_full_backing_filename()` of `node`.
    pub fn full_backing_filename(&self, node: &str) -> Result<String> {
        let bs = self.lookup_bs(node)?;
        let backing = bs.meta.lock().unwrap().backing_file.clone();
        crate::open::full_backing_filename(&bs, &backing)
    }

    /// The children of `node` as (child name, node name).
    pub fn node_children(&self, node: &str) -> Result<Vec<(String, String)>> {
        let bs = self.lookup_bs(node)?;
        Ok(bs.children().into_iter().map(|c| (c.name.clone(), c.node.name.clone())).collect())
    }

    /// `bdrv_cow_bs()`: the backing node of `node`, if it has one.
    pub fn cow_bs(&self, node: &str) -> Result<Option<String>> {
        let bs = self.lookup_bs(node)?;
        Ok(bs.cow_child().map(|c| c.node.name.clone()))
    }

    /// `bdrv_filter_or_cow_bs()`.
    pub fn filter_or_cow_bs(&self, node: &str) -> Result<Option<String>> {
        let bs = self.lookup_bs(node)?;
        Ok(bs.filter_or_cow_bs().map(|n| n.name.clone()))
    }

    /// `bdrv_skip_filters()`.
    pub fn skip_filters(&self, node: &str) -> Result<String> {
        let bs = self.lookup_bs(node)?;
        Ok(bs.skip_filters().name.clone())
    }

    /// `bdrv_backing_chain_next()`.
    pub fn backing_chain_next(&self, node: &str) -> Result<Option<String>> {
        let bs = self.lookup_bs(node)?;
        Ok(bs.backing_chain_next().map(|n| n.name.clone()))
    }

    /// `bdrv_get_info()`.
    pub fn driver_info(&self, node: &str) -> io::Result<DriverInfo> {
        let bs = self.lookup_bs(node).map_err(|_| errno(node::ENOMEDIUM))?;
        let i = bs.get_info()?;
        Ok(DriverInfo {
            cluster_size: i.cluster_size,
            subcluster_size: i.subcluster_size,
            vm_state_offset: i.vm_state_offset,
            is_dirty: i.is_dirty,
            needs_compressed_writes: i.needs_compressed_writes,
        })
    }

    /// `bdrv_get_specific_info()`.
    pub fn specific_info(&self, node: &str) -> Result<Option<ImageInfoSpecific>> {
        let bs = self.lookup_bs(node)?;
        bs.driver.get_specific_info(&bs)
    }

    /// `bdrv_supports_compressed_writes()`.
    pub fn supports_compressed_writes(&self, node: &str) -> bool {
        fn go(bs: &Node) -> bool {
            if !bs.driver.can_compress() {
                return false;
            }
            match bs.filter_child() {
                Some(c) if bs.is_filter() => go(&c.node),
                _ => true,
            }
        }
        self.lookup_bs(node).is_ok_and(|bs| go(&bs))
    }

    /// `block_driver_can_compress()` of the driver of `node` itself.
    pub fn can_compress(&self, node: &str) -> bool {
        self.lookup_bs(node).is_ok_and(|bs| bs.driver.can_compress())
    }

    /// The block limits of `node`.
    pub fn limits(&self, node: &str) -> Result<Limits> {
        let bs = self.lookup_bs(node)?;
        let bl = bs.limits();
        Ok(Limits {
            request_alignment: bl.request_alignment,
            opt_transfer: bl.opt_transfer,
            max_transfer: bl.max_transfer,
            pdiscard_alignment: bl.pdiscard_alignment,
            max_pdiscard: bl.max_pdiscard,
        })
    }

    /// `bdrv_has_zero_init()`.
    pub fn has_zero_init(&self, node: &str) -> bool {
        self.lookup_bs(node).is_ok_and(|bs| bs.has_zero_init())
    }

    /// `bdrv_block_status()`: the status of `node` alone.
    pub fn block_status(&self, node: &str, offset: u64, bytes: u64) -> io::Result<BlockStatusInfo> {
        let bs = self.lookup_bs(node).map_err(|_| errno(node::ENOMEDIUM))?;
        Ok(status_info(bs.block_status(offset, bytes)?, 1))
    }

    /// `bdrv_block_status_above()` from `node` down to (not including) `base`, and with
    /// `want_zero` false the cheaper `bdrv_is_allocated_above()` style query. The depth is
    /// that of `bdrv_co_common_block_status_above()`.
    pub fn block_status_above(
        &self,
        node: &str,
        base: Option<&str>,
        include_base: bool,
        want_zero: bool,
        offset: u64,
        bytes: u64,
    ) -> io::Result<BlockStatusInfo> {
        let bs = self.lookup_bs(node).map_err(|_| errno(node::ENOMEDIUM))?;
        let base = match base {
            Some(b) => Some(self.lookup_bs(b).map_err(|_| errno(node::ENOMEDIUM))?),
            None => None,
        };
        let mode = if want_zero { node::BDRV_WANT_PRECISE } else { node::BDRV_WANT_ALLOCATED };
        let (st, depth) =
            bs.common_block_status_above(base.as_ref(), include_base, mode, offset, bytes)?;
        Ok(status_info(st, depth))
    }

    /// `bdrv_is_allocated()`.
    pub fn is_allocated(&self, node: &str, offset: u64, bytes: u64) -> io::Result<(bool, u64)> {
        let bs = self.lookup_bs(node).map_err(|_| errno(node::ENOMEDIUM))?;
        bs.is_allocated(offset, bytes)
    }

    /// `bdrv_is_allocated_above()`: the depth of the allocating layer (0 for none) and for
    /// how many bytes that holds.
    pub fn is_allocated_above(
        &self,
        node: &str,
        base: Option<&str>,
        include_base: bool,
        offset: u64,
        bytes: u64,
    ) -> io::Result<(u32, u64)> {
        let bs = self.lookup_bs(node).map_err(|_| errno(node::ENOMEDIUM))?;
        let base = match base {
            Some(b) => Some(self.lookup_bs(b).map_err(|_| errno(node::ENOMEDIUM))?),
            None => None,
        };
        bs.is_allocated_above(base.as_ref(), include_base, offset, bytes)
    }

    /// `bdrv_find_backing_image()`: the node below `node` that `backing_file` names.
    pub fn find_backing_image(&self, node: &str, backing_file: &str) -> Option<String> {
        let bs = self.lookup_bs(node).ok()?;
        crate::job::chain::find_backing_image(&bs, backing_file).map(|n| n.name.clone())
    }

    /// `bs->exact_filename` after `bdrv_refresh_filename()`, or `bs->filename` when that is
    /// empty.
    pub fn exact_or_filename(&self, node: &str) -> Result<String> {
        let bs = self.lookup_bs(node)?;
        bs.refresh_filename();
        let meta = bs.meta.lock().unwrap();
        Ok(if meta.exact_filename.is_empty() {
            meta.filename.clone()
        } else {
            meta.exact_filename.clone()
        })
    }

    /// `bdrv_get_backing_filename()`: the backing file name the image header holds.
    pub fn backing_filename(&self, node: &str) -> Result<String> {
        let bs = self.lookup_bs(node)?;
        let name = bs.meta.lock().unwrap().backing_file.clone();
        Ok(name)
    }

    /// `bdrv_change_backing_file()` of `node`.
    pub fn change_backing_file(
        &self,
        node: &str,
        file: Option<&str>,
        format: Option<&str>,
        require: bool,
    ) -> io::Result<()> {
        let bs = self.lookup_bs(node).map_err(|_| errno(node::ENOMEDIUM))?;
        bs.change_backing_file(file, format, require)
    }

    /// `bdrv_snapshot_list()`: `None` when the node does not support snapshots.
    pub fn snapshot_list(&self, node: &str) -> Result<Option<Vec<SnapshotInfo>>> {
        let bs = self.lookup_bs(node)?;
        match bs.snapshot_list() {
            None => Ok(None),
            Some(r) => Ok(Some(r?.into_iter().map(snapshot_info).collect())),
        }
    }

    /// `bdrv_snapshot_find()`: the snapshot whose id, or else whose name, is `name`.
    pub fn snapshot_find(&self, node: &str, name: &str) -> Result<Option<SnapshotInfo>> {
        let Some(list) = self.snapshot_list(node)? else {
            return Err(Error::from_io("Failed to get a snapshot list", errno(libc::ENOTSUP)));
        };
        if let Some(sn) = list.iter().find(|s| s.id == name) {
            return Ok(Some(sn.clone()));
        }
        Ok(list.into_iter().find(|s| s.name == name))
    }

    /// `bdrv_snapshot_create()` of a snapshot called `name` taken now, as `qemu-img snapshot
    /// -c` does it.
    pub fn snapshot_create(&self, node: &str, name: &str) -> Result<()> {
        let bs = self.lookup_bs(node)?;
        let now =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
        let sn = SnapshotEntry {
            name: name.to_string(),
            date_sec: now.as_secs() as u32,
            date_nsec: now.subsec_micros() * 1000,
            ..SnapshotEntry::default()
        };
        let _d = bs.drained();
        let _g = crate::graph_lock::rdlock();
        match bs.snapshot_create(&sn) {
            Some(r) => r,
            None => Err(Error::from_io("Operation not supported", errno(libc::ENOTSUP))),
        }
    }

    /// `bdrv_snapshot_delete()` of the snapshot with `id` and `name`.
    pub fn snapshot_delete(&self, node: &str, id: Option<&str>, name: Option<&str>) -> Result<()> {
        let bs = self.lookup_bs(node)?;
        let _d = bs.drained();
        let _g = crate::graph_lock::rdlock();
        bs.snapshot_delete(id, name, node)
    }

    /// `bdrv_snapshot_load_tmp()`: makes the read-only `node` read the snapshot with `id`
    /// and/or `name`.
    pub fn snapshot_load_tmp(&self, node: &str, id: Option<&str>, name: Option<&str>) -> Result<()> {
        let bs = self.lookup_bs(node)?;
        if id.is_none() && name.is_none() {
            return Err(Error::generic("snapshot_id and name are both NULL"));
        }
        if !bs.read_only() {
            return Err(Error::generic("Device is not readonly"));
        }
        match bs.driver.snapshot_load_tmp(&bs, id, name) {
            Some(r) => r,
            None => Err(Error::generic(format!(
                "Block format '{}' used by device '' does not support temporarily loading \
                 internal snapshots",
                bs.driver_name
            ))),
        }
    }

    /// `bdrv_snapshot_load_tmp_by_id_or_name()`: tries `id_or_name` as an id, then as a
    /// name. QEMU retries only on `-ENOENT` and `-EINVAL`; errors have no errno here, so any
    /// failure of the first try leads to the second.
    pub fn snapshot_load_tmp_by_id_or_name(&self, node: &str, id_or_name: &str) -> Result<()> {
        match self.snapshot_load_tmp(node, Some(id_or_name), None) {
            Ok(()) => Ok(()),
            Err(_) => self.snapshot_load_tmp(node, None, Some(id_or_name)),
        }
    }

    /// `bdrv_supports_persistent_dirty_bitmap()`.
    pub fn supports_persistent_dirty_bitmap(&self, node: &str) -> bool {
        self.lookup_bs(node).is_ok_and(|bs| bs.supports_persistent_dirty_bitmap())
    }

    /// The named dirty bitmaps of `node`, in the order `FOR_EACH_DIRTY_BITMAP` visits them.
    pub fn dirty_bitmap_details(&self, node: &str) -> Result<Vec<BitmapDetails>> {
        let bs = self.lookup_bs(node)?;
        Ok(bs
            .dirty_bitmaps()
            .iter()
            .filter_map(|bm| {
                Some(BitmapDetails {
                    name: bm.name()?,
                    persistent: bm.get_persistence(),
                    inconsistent: bm.inconsistent(),
                    granularity: bm.granularity(),
                    enabled: bm.enabled(),
                })
            })
            .collect())
    }

    /// `bdrv_reopen_set_read_only()`.
    pub fn reopen_set_read_only(&self, node: &str, read_only: bool) -> Result<()> {
        let mut o = QDict::new();
        o.put("read-only", read_only);
        self.reopen_node(node, o, true)
    }

    /// `bdrv_amend_options()`: changes the options of the image `node` from `qemu-img amend
    /// -o` options (as strings). What the driver does not take is left in `options`.
    pub fn amend_options(&self, node: &str, options: &mut QDict, force: bool) -> Result<()> {
        let bs = self.lookup_bs(node)?;
        let Some(def) = bs.def else {
            return Err(Error::generic("Driver does not support amending options"));
        };
        if let Some(f) = def.amend_opts {
            return f(&bs, options, force);
        }
        let mut d = std::mem::take(options);
        d.put("driver", def.format_name);
        // qobject_input_visitor_new_flat_confused()
        let crumpled = crate::open::crumple(d)?;
        let mut v = QObjectInputVisitor::new_keyval(QValue::Dict(crumpled));
        let mut opts = BlockdevAmendOptions::default();
        BlockdevAmendOptions::visit(&mut v, None, &mut opts)?;
        let _g = crate::graph_lock::rdlock();
        match bs.driver.amend(&bs, &opts, force) {
            Some(r) => r,
            None => Err(Error::generic("Driver does not support amending options")),
        }
    }

    /// `bdrv_measure()`: what a new image of the format `fmt` with `options` (or holding the
    /// data of `in_node`) needs.
    pub fn measure(
        &self,
        fmt: &str,
        options: &mut QDict,
        in_node: Option<&str>,
    ) -> Result<BlockMeasureInfo> {
        let Some(def) = drivers::find_format(fmt) else {
            return Err(Error::generic(format!("Unknown file format '{fmt}'")));
        };
        let Some(m) = def.measure else {
            return Err(Error::generic(format!(
                "Block driver '{}' does not support size measurement",
                def.format_name
            )));
        };
        let in_bs = match in_node {
            Some(n) => Some(self.lookup_bs(n)?),
            None => None,
        };
        m(options, in_bs.as_deref())
    }
}

impl BlockBackend {
    /// `blk_truncate()` with `exact` and a preallocation mode. Needs the `resize` permission.
    pub fn truncate_full(&self, len: u64, exact: bool, prealloc: PreallocMode) -> Result<()> {
        let Some(node) = self.root() else {
            return Err(Error::generic("No medium inserted"));
        };
        if self.perm().0 & BLK_PERM_RESIZE == 0 {
            return Err(Error::generic("blk_truncate() needs the 'resize' permission"));
        }
        node.truncate_full(len as i64, exact, prealloc, 0)
    }
}

/// `raw_co_create_opts()`: the file with the protocol options.
pub(crate) fn raw_co_create_opts(filename: &str, options: &mut QDict) -> Result<()> {
    BlockGraph::new().create_file(filename, options)
}

fn take_size(options: &mut QDict, key: &str) -> Result<u64> {
    Ok(crate::imgopts::take_size(options, key)?.unwrap_or(0))
}

/// `raw_measure()`.
pub(crate) fn raw_measure(options: &mut QDict, in_bs: Option<&Node>) -> Result<BlockMeasureInfo> {
    let required = match in_bs {
        Some(bs) => {
            bs.getlength().map_err(|e| Error::from_io("Unable to get image size", e))? as i64
        }
        None => take_size(options, "size")?.div_ceil(512) as i64 * 512,
    };
    // Unallocated sectors count towards the file size in raw images.
    Ok(BlockMeasureInfo { required, fully_allocated: required, bitmaps: None })
}

/// `block_crypto_measure()`.
pub(crate) fn luks_measure(options: &mut QDict, in_bs: Option<&Node>) -> Result<BlockMeasureInfo> {
    // Preallocation does not change what is needed, but the option is consumed.
    options.remove("preallocation");
    let mut size = take_size(options, "size")?;
    if let Some(bs) = in_bs {
        size = bs.getlength().map_err(|e| Error::from_io("Unable to get image virtual_size", e))?;
    }
    let mut crypto = QDict::new();
    for d in LUKS_CREATE_OPTS {
        if let Some(v) = options.remove(d.name) {
            crypto.put(d.name, v);
        }
    }
    let opts = crate::luks::create_opts_init(&mut crypto, "luks")?;
    let payload = ruvm_crypto::block::QCryptoBlock::calculate_payload_offset(&opts, None)?;
    // Unallocated blocks are still encrypted, so allocation makes no difference.
    let n = (payload + size) as i64;
    Ok(BlockMeasureInfo { required: n, fully_allocated: n, bitmaps: None })
}

const fn opt(name: &'static str, ty: QemuOptType, help: &'static str) -> QemuOptDesc {
    QemuOptDesc::new(name, ty).help(help)
}

const SIZE_OPT: QemuOptDesc = opt("size", QemuOptType::Size, "Virtual disk size");

/// `raw_create_opts` of block/raw-format.c.
pub(crate) static RAW_CREATE_OPTS: [QemuOptDesc; 1] = [SIZE_OPT];

/// `raw_create_opts` of block/file-posix.c.
#[cfg(unix)]
pub(crate) static FILE_CREATE_OPTS: [QemuOptDesc; 4] = [
    SIZE_OPT,
    opt("nocow", QemuOptType::Bool, "Turn off copy-on-write (valid only on btrfs)"),
    #[cfg(target_os = "linux")]
    opt(
        "preallocation",
        QemuOptType::String,
        "Preallocation mode (allowed values: off, falloc, full)",
    ),
    #[cfg(not(target_os = "linux"))]
    opt("preallocation", QemuOptType::String, "Preallocation mode (allowed values: off, full)"),
    opt("extent_size_hint", QemuOptType::Size, "Extent size hint for the image file, 0 to disable"),
];

const LUKS_ITER_TIME: QemuOptDesc =
    opt("iter-time", QemuOptType::Number, "Time to spend in PBKDF in milliseconds");

/// `block_crypto_create_opts_luks`.
pub(crate) static LUKS_CREATE_OPTS: [QemuOptDesc; 9] = [
    SIZE_OPT,
    opt("key-secret", QemuOptType::String, "ID of the secret that provides the keyslot passphrase"),
    opt("cipher-alg", QemuOptType::String, "Name of encryption cipher algorithm"),
    opt("cipher-mode", QemuOptType::String, "Name of encryption cipher mode"),
    opt("ivgen-alg", QemuOptType::String, "Name of IV generator algorithm"),
    opt("ivgen-hash-alg", QemuOptType::String, "Name of IV generator hash algorithm"),
    opt("hash-alg", QemuOptType::String, "Name of encryption hash algorithm"),
    LUKS_ITER_TIME,
    opt("detached-header", QemuOptType::Bool, "Create a detached LUKS header"),
];

/// `block_crypto_amend_opts_luks`.
pub(crate) static LUKS_AMEND_OPTS: [QemuOptDesc; 5] = [
    opt("state", QemuOptType::String, "Select new state of affected keyslots (active/inactive)"),
    opt("keyslot", QemuOptType::Number, "Select a single keyslot to modify explicitly"),
    opt("old-secret", QemuOptType::String, "Select all keyslots that match this password"),
    opt(
        "new-secret",
        QemuOptType::String,
        "New secret to set in the matching keyslots. Empty string to erase",
    ),
    LUKS_ITER_TIME,
];
