// SPDX-License-Identifier: GPL-2.0-or-later

//! `query-named-block-nodes` and `block-set-write-threshold`, from block/qapi.c, block.c and
//! block/write-threshold.c.
//!
//! Differences from QEMU:
//!
//! - There are no implicit filter nodes yet (the ones block jobs insert), so
//!   `bdrv_skip_implicit_filters()` has nothing to skip.
//! - A driver error while listing snapshots is reported as the driver gives it, not sorted
//!   into QEMU's "not inserted", "does not support internal snapshots" and "Can't list
//!   snapshots" messages, since drivers here return an [`Error`] rather than an errno.
//! - Dirty bitmaps do not exist yet, so `dirty-bitmaps` is always absent.

use std::io;

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{
    BlockDeviceInfo, BlockLimitsInfo, BlockdevCacheInfo, BlockdevChild, ImageInfo, SnapshotInfo,
};

use crate::backend::BlockBackend;
use crate::graph::BlockGraph;
use crate::node::{BDRV_CHILD_DATA, BDRV_CHILD_FILTERED, BDRV_CHILD_METADATA, Node, errno};
use crate::open::full_backing_filename;

impl Node {
    /// `bdrv_co_get_allocated_file_size()`.
    pub(crate) fn allocated_file_size(&self) -> io::Result<u64> {
        if let Some(r) = self.driver.get_allocated_file_size(self) {
            return r;
        }
        if self.is_protocol() {
            // Protocol drivers keep their data outside their children, if they have any.
            Err(errno(libc::ENOTSUP))
        } else if self.is_filter() {
            match self.filter_child() {
                Some(c) => c.node.allocated_file_size(),
                None => Err(errno(libc::ENOTSUP)),
            }
        } else {
            // bdrv_sum_allocated_file_size().
            let mut sum = 0;
            for c in self.children() {
                if c.role & (BDRV_CHILD_DATA | BDRV_CHILD_METADATA | BDRV_CHILD_FILTERED) != 0 {
                    sum += c.node.allocated_file_size()?;
                }
            }
            Ok(sum)
        }
    }

    /// `bdrv_snapshot_list()`: the driver's snapshots, or those of the child it falls back
    /// to (`bdrv_snapshot_fallback()`). `None` when nobody supports snapshots.
    fn snapshot_infos(&self) -> Option<Result<Vec<SnapshotInfo>>> {
        match self.driver.snapshot_list(self) {
            Some(r) => Some(r.map(|v| {
                v.into_iter()
                    .map(|sn| SnapshotInfo {
                        id: sn.id_str,
                        name: sn.name,
                        vm_state_size: sn.vm_state_size as i64,
                        date_sec: i64::from(sn.date_sec),
                        date_nsec: i64::from(sn.date_nsec),
                        vm_clock_sec: (sn.vm_clock_nsec / 1_000_000_000) as i64,
                        vm_clock_nsec: (sn.vm_clock_nsec % 1_000_000_000) as i64,
                        icount: sn.icount.map(|i| i as i64),
                    })
                    .collect()
            })),
            None => {
                let c = self.primary_child()?;
                if c.role & (BDRV_CHILD_DATA | BDRV_CHILD_FILTERED) == 0 {
                    return None;
                }
                c.node.snapshot_infos()
            }
        }
    }

    /// `bdrv_do_query_node_info()` and `bdrv_query_image_info()`: the image information of
    /// the node and, unless `flat`, of its backing chain.
    pub(crate) fn query_image_info(&self, flat: bool) -> Result<ImageInfo> {
        let size = self.getlength().map_err(|e| {
            let f = self.meta.lock().unwrap().exact_filename.clone();
            Error::from_io(format!("Can't get image size '{f}'"), e)
        })?;
        self.refresh_filename();
        let meta = self.meta.lock().unwrap().clone_info();
        let bl = self.limits();
        let nz32 = |v: u32| (v != 0).then_some(v);
        let nz64 = |v: u64| (v != 0).then_some(v);
        let mut info = ImageInfo {
            filename: meta.filename,
            format: self.driver_name.to_string(),
            virtual_size: size as i64,
            actual_size: self.allocated_file_size().ok().map(|s| s as i64),
            encrypted: meta.encrypted.then_some(true),
            limits: Some(BlockLimitsInfo {
                request_alignment: bl.request_alignment,
                max_discard: nz64(bl.max_pdiscard),
                discard_alignment: nz32(bl.pdiscard_alignment),
                max_write_zeroes: nz64(bl.max_pwrite_zeroes),
                write_zeroes_alignment: nz32(bl.pwrite_zeroes_alignment),
                opt_transfer: nz32(bl.opt_transfer),
                max_transfer: nz32(bl.max_transfer),
                max_hw_transfer: nz64(bl.max_hw_transfer)
                    .map(|v| v.min(u64::from(u32::MAX)) as u32),
                max_iov: i64::from(bl.max_iov),
                max_hw_iov: (bl.max_hw_iov != 0).then_some(i64::from(bl.max_hw_iov)),
                min_mem_alignment: bl.min_mem_alignment as u64,
                opt_mem_alignment: bl.opt_mem_alignment as u64,
            }),
            ..ImageInfo::default()
        };
        if let Ok(bdi) = self.get_info() {
            if bdi.cluster_size != 0 {
                info.cluster_size = Some(bdi.cluster_size as i64);
            }
            info.dirty_flag = Some(bdi.is_dirty);
        }
        info.format_specific = self.driver.get_specific_info(self)?;
        if !meta.backing_file.is_empty() {
            info.full_backing_filename = full_backing_filename(self, &meta.backing_file).ok();
            info.backing_filename = Some(meta.backing_file);
            if !meta.backing_format.is_empty() {
                info.backing_filename_format = Some(meta.backing_format);
            }
        }
        match self.snapshot_infos() {
            Some(Ok(v)) if !v.is_empty() => info.snapshots = Some(v),
            // -ENOMEDIUM and -ENOTSUP are recoverable: the image just has no snapshots to show.
            Some(Err(e)) => {
                let code = std::error::Error::source(&e)
                    .and_then(|c| c.downcast_ref::<io::Error>())
                    .and_then(io::Error::raw_os_error);
                if code != Some(libc::ENOTSUP) && code != Some(crate::node::ENOMEDIUM) {
                    return Err(e);
                }
            }
            _ => {}
        }
        if !flat {
            // Any filtered child, for compatibility with when this always took bs->backing.
            if let Some(b) = self.filter_or_cow_bs() {
                info.backing_image = Some(Box::new(b.query_image_info(false)?));
            }
        }
        Ok(info)
    }

    /// `bdrv_block_device_info()`. `blk` is the backend for `query-block`, `None` for
    /// `query-named-block-nodes`.
    pub(crate) fn block_device_info(
        &self,
        blk: Option<&BlockBackend>,
        flat: bool,
    ) -> Result<BlockDeviceInfo> {
        self.refresh_filename();
        let flags = self.flags();
        let meta = self.meta.lock().unwrap().clone_info();
        let image = self.query_image_info(flat)?;
        let mut depth = 0;
        let mut b = image.backing_image.as_deref();
        while let Some(i) = b {
            depth += 1;
            b = i.backing_image.as_deref();
        }
        Ok(BlockDeviceInfo {
            file: meta.filename,
            node_name: self.name.clone(),
            ro: self.read_only(),
            drv: self.driver_name.to_string(),
            backing_file: self.cow_child().and_then(|c| c.node.filename()),
            backing_file_depth: depth,
            children: self
                .children()
                .into_iter()
                .map(|c| BlockdevChild { child: c.name.clone(), node_name: c.node.name.clone() })
                .collect(),
            active: !self.is_inactive(),
            encrypted: meta.encrypted,
            detect_zeroes: flags.detect_zeroes,
            image,
            cache: BlockdevCacheInfo {
                writeback: blk.is_none_or(BlockBackend::enable_write_cache),
                direct: flags.direct,
                no_flush: flags.no_flush,
            },
            write_threshold: self.write_threshold() as i64,
            ..BlockDeviceInfo::default()
        })
    }
}

/// The parts of `NodeMeta` the queries read, copied out so the lock is not held while the
/// driver is asked for more.
struct MetaInfo {
    filename: String,
    backing_file: String,
    backing_format: String,
    encrypted: bool,
}

impl crate::node::NodeMeta {
    fn clone_info(&self) -> MetaInfo {
        MetaInfo {
            filename: self.filename.clone(),
            backing_file: self.backing_file.clone(),
            backing_format: self.backing_format.clone(),
            encrypted: self.encrypted,
        }
    }
}

impl BlockGraph {
    /// `qmp_query_named_block_nodes()`: every named node, the newest first, as
    /// `bdrv_named_nodes_list()` prepends.
    pub fn query_named_block_nodes(&self, flat: Option<bool>) -> Result<Vec<BlockDeviceInfo>> {
        let flat = flat.unwrap_or(false);
        let mut list = Vec::new();
        for bs in self.named_nodes().iter().rev() {
            list.push(bs.block_device_info(None, flat)?);
        }
        Ok(list)
    }

    /// `qmp_block_set_write_threshold()`.
    pub fn block_set_write_threshold(&self, node_name: &str, threshold_bytes: u64) -> Result<()> {
        let Some(bs) = self.find_node(node_name) else {
            return Err(Error::generic(format!("Device '{node_name}' not found")));
        };
        bs.set_write_threshold(threshold_bytes);
        Ok(())
    }

    /// `bdrv_write_threshold_get()` for a node by name.
    pub fn write_threshold(&self, node_name: &str) -> Option<u64> {
        self.find_node(node_name).map(|bs| bs.write_threshold())
    }
}

#[cfg(test)]
mod tests {
    use ruvm_qapi::QDict;

    use super::*;

    #[test]
    fn named_block_nodes() {
        let g = BlockGraph::new();
        let top = g
            .open_image(
                Some(r#"json:{"driver": "raw", "offset": 512, "file": {"driver": "null-co"}}"#),
                QDict::new().with("node-name", "qn1").with("file.node-name", "qn0"),
            )
            .unwrap();
        assert_eq!(top, "qn1");
        let list = g.query_named_block_nodes(Some(true)).unwrap();
        let names: Vec<_> = list.iter().map(|i| i.node_name.as_str()).collect();
        // The newest node first.
        assert_eq!(names, ["qn1", "qn0"]);
        let raw = &list[0];
        assert_eq!(raw.drv, "raw");
        assert_eq!(
            raw.file,
            r#"json:{"offset": 512, "driver": "raw", "file": {"driver": "null-co"}}"#
        );
        assert_eq!(raw.image.virtual_size, (1 << 30) - 512);
        assert_eq!(raw.children.len(), 1);
        assert_eq!(
            (raw.children[0].child.as_str(), raw.children[0].node_name.as_str()),
            ("file", "qn0")
        );
        assert_eq!(list[1].file, "null-co://");
        assert!(list[1].children.is_empty());

        // The json: name opens the same tree again.
        let again = g.open_image(Some(&raw.file), QDict::new()).unwrap();
        let bs = g.find_node(&again).unwrap();
        assert_eq!(bs.getlength().unwrap(), (1 << 30) - 512);

        assert_eq!(
            g.block_set_write_threshold("nope", 1).unwrap_err().message(),
            "Device 'nope' not found"
        );
        g.block_set_write_threshold("qn1", 4096).unwrap();
        assert_eq!(g.write_threshold("qn1"), Some(4096));
        let list = g.query_named_block_nodes(None).unwrap();
        let qn1 = list.iter().find(|i| i.node_name == "qn1").unwrap();
        assert_eq!(qn1.write_threshold, 4096);
    }
}
