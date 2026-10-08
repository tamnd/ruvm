// SPDX-License-Identifier: GPL-2.0-or-later

//! Whole-node operations for the monitor and the tools. This covers image checks, internal
//! and external snapshots, backing file changes, emptying and zeroing images, the VM state
//! area, `qemu-img map` entries, medium control and the blkdebug breakpoints of qemu-io.
//! It comes from block.c, block/snapshot.c, block/io.c, block/block-backend.c, blockdev.c and
//! qemu-img.c.
//!
//! Differences from QEMU:
//!
//! - `bdrv_can_snapshot()` asks whether the driver can list snapshots. A [`Driver`] cannot
//!   say whether it implements `snapshot_create` without being called.
//! - `bdrv_snapshot_goto()` falls back to the primary child without closing and reopening
//!   the node around it. No driver that falls back here keeps state that the snapshot would
//!   change.
//! - There are no op blockers and no dirty bitmaps. `bdrv_op_is_blocked()` never refuses,
//!   and a snapshot goto never fails with "Device has active dirty bitmaps".
//! - `blockdev-snapshot-internal-sync` and `blockdev-snapshot` run on their own, not as
//!   `transaction` actions. When they fail nothing has changed, as after the abort of a
//!   one-action transaction.
//! - `x-blockdev-amend` is a job in QEMU. [`BlockGraph::x_blockdev_amend`] runs the
//!   driver's amend function to the end on the caller's thread and takes no job id. The
//!   driver's pre-run and clean-up steps happen inside [`Driver::amend`].
//! - An internal snapshot records a `vm-clock` of 0 and no `icount`. The virtual clock and
//!   record/replay belong to the machine, which is not in this crate.
//! - Block backends have no `activate` and `inactivate` callbacks. They keep their
//!   permissions, so a node under a backend that may write cannot be inactivated
//!   (`EPERM`), where QEMU drops the permissions of a backend with a name or without a
//!   device first.
//!
//! [`Driver`]: crate::node::Driver

use std::io;
use std::sync::Arc;

use ruvm_base::error::strerror;
use ruvm_base::{Error, Result};
use ruvm_qapi::types::SnapshotInfo;

use crate::backend::BlockBackend;
use crate::graph::BlockGraph;
use crate::node::{
    BDRV_BLOCK_ALLOCATED, BDRV_BLOCK_COMPRESSED, BDRV_BLOCK_DATA, BDRV_BLOCK_OFFSET_VALID,
    BDRV_BLOCK_ZERO, BDRV_CHILD_DATA, BDRV_CHILD_FILTERED, BDRV_CHILD_METADATA, BDRV_REQ_MAY_UNMAP,
    CheckResult, ENOMEDIUM, Node, SnapshotEntry, errno,
};
use crate::perm::{BLK_PERM_CONSISTENT_READ, BLK_PERM_WRITE, BLK_PERM_WRITE_UNCHANGED};

/// One entry of `qemu-img map`, `MapEntry` in qemu-img.c: how a range of the image reads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MapEntry {
    /// Where the range starts.
    pub start: u64,
    /// How long it is.
    pub length: u64,
    /// Reads return data (`BDRV_BLOCK_DATA`).
    pub data: bool,
    /// Reads return zeroes (`BDRV_BLOCK_ZERO`).
    pub zero: bool,
    /// The data is compressed (`BDRV_BLOCK_COMPRESSED`).
    pub compressed: bool,
    /// Where the data is in [`MapEntry::filename`], when that is known.
    pub offset: Option<u64>,
    /// How many backing layers down the range comes from, 0 for the top.
    pub depth: u32,
    /// Some layer of the chain decides the content (`BDRV_BLOCK_ALLOCATED`).
    pub present: bool,
    /// The file the data is in, when `offset` is known.
    pub filename: Option<String>,
}

impl Node {
    /// `bdrv_snapshot_fallback()`: the primary child, if it is the only child with data.
    fn snapshot_fallback(&self) -> Option<Arc<Node>> {
        let fallback = self.primary_child()?;
        let others = self.children().into_iter().any(|c| {
            c.edge != fallback.edge
                && c.role & (BDRV_CHILD_DATA | BDRV_CHILD_METADATA | BDRV_CHILD_FILTERED) != 0
        });
        if others { None } else { Some(fallback.node) }
    }

    /// `bdrv_can_snapshot()`.
    pub(crate) fn can_snapshot(&self) -> bool {
        if !self.is_inserted() || self.read_only() {
            return false;
        }
        if self.driver.snapshot_list(self).is_none() {
            return self.snapshot_fallback().is_some_and(|f| f.can_snapshot());
        }
        true
    }

    /// `bdrv_snapshot_list()` with `bdrv_snapshot_fallback()`. `None` is `-ENOTSUP`.
    pub(crate) fn snapshot_list(&self) -> Option<Result<Vec<SnapshotEntry>>> {
        match self.driver.snapshot_list(self) {
            Some(r) => Some(r),
            None => self.snapshot_fallback()?.snapshot_list(),
        }
    }

    /// `bdrv_snapshot_find_by_id_and_name()`.
    fn snapshot_find(&self, id: Option<&str>, name: Option<&str>) -> Result<Option<SnapshotEntry>> {
        assert!(id.is_some() || name.is_some());
        let list = match self.snapshot_list() {
            Some(Ok(l)) => l,
            Some(Err(e)) => return Err(e.prepend("Failed to get a snapshot list: ")),
            None => {
                return Err(Error::from_io("Failed to get a snapshot list", errno(libc::ENOTSUP)));
            }
        };
        Ok(list
            .into_iter()
            .find(|sn| id.is_none_or(|i| sn.id_str == i) && name.is_none_or(|n| sn.name == n)))
    }

    /// `bdrv_snapshot_create()`. `None` is `-ENOTSUP`.
    pub(crate) fn snapshot_create(&self, sn: &SnapshotEntry) -> Option<Result<()>> {
        match self.driver.snapshot_create(self, sn) {
            Some(r) => Some(r),
            None => self.snapshot_fallback()?.snapshot_create(sn),
        }
    }

    /// `bdrv_snapshot_goto()`.
    pub(crate) fn snapshot_goto(&self, id: &str) -> Result<()> {
        if let Some(r) = self.driver.snapshot_goto(self, id) {
            return r.map_err(|e| e.prepend("Failed to load snapshot: "));
        }
        match self.snapshot_fallback() {
            Some(f) => f.snapshot_goto(id),
            None => Err(Error::generic("Block driver does not support snapshots")),
        }
    }

    /// `bdrv_snapshot_delete()`. `device` names the node in errors.
    pub(crate) fn snapshot_delete(
        &self,
        id: Option<&str>,
        name: Option<&str>,
        device: &str,
    ) -> Result<()> {
        assert!(self.quiesce_counter.load(std::sync::atomic::Ordering::SeqCst) > 0);
        if id.is_none() && name.is_none() {
            return Err(Error::generic("snapshot_id and name are both NULL"));
        }
        if let Some(r) = self.driver.snapshot_delete(self, id, name) {
            return r;
        }
        match self.snapshot_fallback() {
            Some(f) => f.snapshot_delete(id, name, device),
            None => Err(Error::generic(format!(
                "Block format '{}' used by device '{device}' does not support internal \
                 snapshot deletion",
                self.driver_name
            ))),
        }
    }

    /// `bdrv_has_zero_init()`: whether a newly created image reads as zeroes.
    pub(crate) fn has_zero_init(&self) -> bool {
        // A copy-on-write image starts out as its base, which may not be zeroes.
        if self.cow_child().is_some() {
            return false;
        }
        if let Some(z) = self.driver.has_zero_init(self) {
            return z;
        }
        self.filter_child().is_some_and(|c| c.node.has_zero_init())
    }

    /// `bdrv_co_check()`. `fix` is a mask of [`BDRV_FIX_LEAKS`] and [`BDRV_FIX_ERRORS`].
    ///
    /// [`BDRV_FIX_LEAKS`]: crate::node::BDRV_FIX_LEAKS
    /// [`BDRV_FIX_ERRORS`]: crate::node::BDRV_FIX_ERRORS
    pub(crate) fn check(&self, fix: u32) -> Option<Result<CheckResult>> {
        self.driver.check(self, fix)
    }

    /// `bdrv_co_change_backing_file()`: records a new backing file name in the image header.
    pub(crate) fn change_backing_file(
        &self,
        file: Option<&str>,
        format: Option<&str>,
        require: bool,
    ) -> io::Result<()> {
        // A backing file format does not make sense without a backing file.
        if format.is_some() && file.is_none() {
            return Err(errno(libc::EINVAL));
        }
        if require && file.is_some() && format.is_none() {
            return Err(errno(libc::EINVAL));
        }
        self.driver
            .change_backing_file(self, file, format)
            .unwrap_or_else(|| Err(errno(libc::ENOTSUP)))?;
        let mut meta = self.meta.lock().unwrap();
        meta.backing_file = file.unwrap_or_default().to_string();
        meta.backing_format = format.unwrap_or_default().to_string();
        meta.auto_backing_file = file.unwrap_or_default().to_string();
        Ok(())
    }

    /// `bdrv_make_empty()`: drops everything the image holds itself.
    pub(crate) fn make_empty(&self) -> Result<()> {
        let Some(r) = self.driver.make_empty(self) else {
            return Err(Error::generic(format!(
                "{} does not support emptying nodes",
                self.driver_name
            )));
        };
        r.map_err(|e| {
            let f = self.meta.lock().unwrap().filename.clone();
            Error::from_io(format!("Failed to empty {f}"), e)
        })
    }

    /// `bdrv_check_request32()` for the VM state area.
    fn check_vmstate_request(pos: u64, len: usize) -> io::Result<()> {
        match pos.checked_add(len as u64) {
            Some(end) if end <= i64::MAX as u64 && len <= i32::MAX as usize => Ok(()),
            _ => Err(errno(libc::EIO)),
        }
    }

    /// `bdrv_co_readv_vmstate()`.
    pub(crate) fn load_vmstate(&self, pos: u64, buf: &mut [u8]) -> io::Result<()> {
        Self::check_vmstate_request(pos, buf.len())?;
        self.inc_in_flight();
        let r = match self.driver.load_vmstate(self, pos, buf) {
            Some(r) => r,
            None => match self.primary_bs() {
                Some(c) => c.load_vmstate(pos, buf),
                None => Err(errno(libc::ENOTSUP)),
            },
        };
        self.dec_in_flight();
        r
    }

    /// `bdrv_co_writev_vmstate()`.
    pub(crate) fn save_vmstate(&self, pos: u64, buf: &[u8]) -> io::Result<()> {
        Self::check_vmstate_request(pos, buf.len())?;
        self.inc_in_flight();
        let r = match self.driver.save_vmstate(self, pos, buf) {
            Some(r) => r,
            None => match self.primary_bs() {
                Some(c) => c.save_vmstate(pos, buf),
                None => Err(errno(libc::ENOTSUP)),
            },
        };
        self.dec_in_flight();
        r
    }

    /// `bdrv_probe_blocksizes()`: the logical and physical block size of a host device.
    pub(crate) fn probe_blocksizes(&self) -> io::Result<(u32, u32)> {
        if let Some(r) = self.driver.probe_blocksizes(self) {
            return r;
        }
        match self.filter_child() {
            Some(c) => c.node.probe_blocksizes(),
            None => Err(errno(libc::ENOTSUP)),
        }
    }

    /// `get_block_status()` of qemu-img.c: the status of the start of the range through the
    /// whole backing chain.
    pub(crate) fn map_entry(self: &Arc<Self>, offset: u64, bytes: u64) -> io::Result<MapEntry> {
        let mut bs = self.clone();
        let mut depth = 0;
        let mut bytes = bytes;
        let st = loop {
            bs = bs.skip_filters();
            let st = bs.block_status(offset, bytes)?;
            bytes = st.pnum;
            if st.ret & (BDRV_BLOCK_ZERO | BDRV_BLOCK_DATA) != 0 {
                break st;
            }
            match bs.cow_child() {
                Some(c) => bs = c.node,
                None => break crate::node::BlockStatus { ret: 0, ..st },
            }
            depth += 1;
        };
        let has_offset = st.ret & BDRV_BLOCK_OFFSET_VALID != 0;
        let filename = match &st.file {
            Some(f) if has_offset => {
                f.refresh_filename();
                f.filename()
            }
            _ => None,
        };
        Ok(MapEntry {
            start: offset,
            length: bytes,
            data: st.ret & BDRV_BLOCK_DATA != 0,
            zero: st.ret & BDRV_BLOCK_ZERO != 0,
            compressed: st.ret & BDRV_BLOCK_COMPRESSED != 0,
            offset: has_offset.then_some(st.map),
            depth,
            present: st.ret & BDRV_BLOCK_ALLOCATED != 0,
            filename,
        })
    }
}

impl Node {
    /// `bdrv_has_bds_parent()`: whether a node (an active one, with `only_active`) sits
    /// above this one.
    fn has_bds_parent(&self, only_active: bool) -> bool {
        self.parent_nodes().iter().any(|p| !only_active || !p.is_inactive())
    }

    fn set_inactive(&self, inactive: bool) {
        let mut f = self.flags();
        f.inactive = inactive;
        self.set_flags(f);
    }

    /// `bdrv_activate()`: makes this node and everything below it active again.
    pub(crate) fn activate(&self) -> Result<()> {
        for c in self.children() {
            c.node.activate()?;
        }
        if self.is_inactive() {
            // The permissions of an inactive node are a subset of those of the active one,
            // so they can be taken before the driver reloads its state.
            self.set_inactive(false);
            let r = self
                .refresh_perms()
                .and_then(|()| self.driver.invalidate_cache(self))
                .and_then(|()| {
                    self.refresh_total_sectors(Some(self.total_sectors()))
                        .map_err(|e| Error::from_io("Could not refresh total sector count", e))
                });
            if let Err(e) = r {
                self.set_inactive(true);
                return Err(e);
            }
        }
        Ok(())
    }

    /// `bdrv_inactivate_recurse()`. The error is the bare reason, the callers say what failed.
    fn inactivate_recurse(&self, top_level: bool) -> Result<()> {
        assert!(self.quiesce_counter.load(std::sync::atomic::Ordering::SeqCst) > 0);
        // A child goes inactive with the last of its parents.
        if self.has_bds_parent(true) {
            return Ok(());
        }
        // Inactivating an inactive node on request is harmless, a child that went inactive
        // before its parent is not.
        if self.is_inactive() {
            assert!(top_level, "child inactive before its parent");
            return Ok(());
        }
        self.driver.inactivate(self)?;
        let (perm, _) = self.cumulative_perm();
        if perm & (BLK_PERM_WRITE | BLK_PERM_WRITE_UNCHANGED) != 0 {
            // The parents that are still active need write access.
            return Err(Error::generic(strerror(&errno(libc::EPERM))));
        }
        self.set_inactive(true);
        // Only loosening, errors do not matter.
        let _ = self.refresh_perms();
        for c in self.children() {
            c.node.inactivate_recurse(false)?;
        }
        Ok(())
    }

    /// `bdrv_inactivate()`: makes this node and what is only below it inactive, so another
    /// process may take the image over. Every node must be drained.
    pub(crate) fn inactivate(&self) -> Result<()> {
        if self.has_bds_parent(true) {
            return Err(Error::generic("Node has active parent node"));
        }
        self.inactivate_recurse(true).map_err(|e| e.prepend("Failed to inactivate node: "))
    }
}

impl BlockGraph {
    /// `blockdev-set-active`: activates or inactivates the node `node_name` and its
    /// children, or every node when `node_name` is `None`.
    pub fn blockdev_set_active(&self, node_name: Option<&str>, active: bool) -> Result<()> {
        let Some(node_name) = node_name else {
            return if active {
                // bdrv_activate_all()
                let _g = crate::graph_lock::rdlock();
                crate::drain::all_nodes().iter().try_for_each(|bs| bs.activate())
            } else {
                // bdrv_inactivate_all(): nodes with a node above them go inactive with it.
                crate::drain::drain_all_begin();
                let r = {
                    let _g = crate::graph_lock::rdlock();
                    crate::drain::all_nodes()
                        .iter()
                        .filter(|bs| !bs.has_bds_parent(false))
                        .try_for_each(|bs| bs.inactivate_recurse(true))
                };
                crate::drain::drain_all_end();
                r.map_err(|e| e.prepend("Failed to inactivate all nodes: "))
            };
        };
        if !active {
            crate::drain::drain_all_begin();
        }
        let r = {
            let _g = crate::graph_lock::rdlock();
            match self.find_node(node_name) {
                None => {
                    Err(Error::generic(format!("Failed to find node with node-name='{node_name}'")))
                }
                Some(bs) if active => bs.activate(),
                Some(bs) => bs.inactivate(),
            }
        };
        if !active {
            crate::drain::drain_all_end();
        }
        r
    }

    /// `bdrv_get_device_or_node_name()`: the name of the block backend `bs` is the root of,
    /// or else its node name.
    pub(crate) fn device_or_node_name(&self, bs: &Arc<Node>) -> String {
        let backends = self.backends.lock().unwrap();
        for (name, blk) in backends.iter() {
            if blk.root().is_some_and(|r| Arc::ptr_eq(&r, bs)) {
                return name.clone();
            }
        }
        bs.name.clone()
    }

    /// `qmp_get_root_bs()`: the node `name` names, which must have no node above it.
    pub(crate) fn root_bs(&self, name: &str) -> Result<Arc<Node>> {
        let bs = self.lookup_bs(name)?;
        // bdrv_is_root_node()
        if !bs.parent_nodes().is_empty() {
            return Err(Error::generic("Need a root block node"));
        }
        if !bs.is_inserted() {
            return Err(Error::generic("Device has no medium"));
        }
        Ok(bs)
    }

    /// `bdrv_drain_all()`: waits until no node has a request in flight.
    pub fn drain_all(&self) {
        crate::drain::drain_all();
    }

    /// `bdrv_replace_node()` by name: every parent of `from` that would not close a loop
    /// uses `to` instead. Nothing changes when the permissions do not allow it. The block
    /// jobs use this when they finish.
    pub fn replace_node(&self, from: &str, to: &str) -> Result<()> {
        let from = self.lookup_bs(from)?;
        let to = self.lookup_bs(to)?;
        Node::replace_node(&from, &to)
    }

    /// `qemu-img check` (`bdrv_co_check()`) of the node `node_name`. `fix` is a mask of
    /// [`BDRV_FIX_LEAKS`](crate::BDRV_FIX_LEAKS) and [`BDRV_FIX_ERRORS`](crate::BDRV_FIX_ERRORS).
    pub fn check(&self, node_name: &str, fix: u32) -> Result<CheckResult> {
        let bs = self.lookup_bs(node_name)?;
        let _g = crate::graph_lock::rdlock();
        match bs.check(fix) {
            Some(r) => r,
            None => Err(Error::generic("This image format does not support checks")),
        }
    }

    /// `qmp_x_blockdev_amend()` run to completion: changes the options of the node
    /// `node_name` in place, the key slots of a LUKS image for example.
    pub fn x_blockdev_amend(
        &self,
        node_name: &str,
        options: ruvm_qapi::types::BlockdevAmendOptions,
        force: Option<bool>,
    ) -> Result<()> {
        let fmt = options.u.tag().as_str();
        let drv = crate::drivers::find_format(fmt);
        let _g = crate::graph_lock::rdlock();
        // bdrv_lookup_bs(NULL, node_name, errp)
        let Some(bs) = self.find_node(node_name) else {
            return Err(Error::generic(format!(
                "Cannot find device='' nor node-name='{node_name}'"
            )));
        };
        let Some(drv) = drv else {
            return Err(Error::generic(format!("Block driver '{fmt}' not found or not supported")));
        };
        if bs.driver_name != drv.format_name {
            return Err(Error::generic(
                "x-blockdev-amend doesn't support changing the block driver",
            ));
        }
        match bs.driver.amend(&bs, &options, force.unwrap_or(false)) {
            Some(r) => r,
            None => Err(Error::generic("Driver does not support x-blockdev-amend")),
        }
    }

    /// `blockdev-snapshot-internal-sync`: takes the internal snapshot `name` of the root node
    /// `device`.
    pub fn blockdev_snapshot_internal_sync(&self, device: &str, name: &str) -> Result<()> {
        let bs = self.root_bs(device)?;
        let _d = bs.drained();
        // Make sure the root node did not change with the drain.
        let check_bs = self.root_bs(device)?;
        if !Arc::ptr_eq(&bs, &check_bs) {
            return Err(Error::generic(format!(
                "Block node of device '{device}' unexpectedly changed"
            )));
        }
        let _g = crate::graph_lock::rdlock();
        if bs.read_only() {
            return Err(Error::generic(format!("Device '{device}' is read only")));
        }
        if !bs.can_snapshot() {
            return Err(Error::generic(format!(
                "Block format '{}' used by device '{device}' does not support internal snapshots",
                bs.driver_name
            )));
        }
        if name.is_empty() {
            return Err(Error::generic("Name is empty"));
        }
        if bs.snapshot_find(None, Some(name))?.is_some() {
            return Err(Error::generic(format!(
                "Snapshot with name '{name}' already exists on device '{device}'"
            )));
        }
        let now =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
        let sn = SnapshotEntry {
            name: name.to_string(),
            date_sec: now.as_secs() as u32,
            // g_get_real_time() has microseconds.
            date_nsec: now.subsec_micros() * 1000,
            ..SnapshotEntry::default()
        };
        let msg = format!("Failed to create snapshot '{name}' on device '{device}'");
        match bs.snapshot_create(&sn) {
            Some(r) => r.map_err(|e| e.prepend(format!("{msg}: "))),
            None => Err(Error::from_io(msg, errno(libc::ENOTSUP))),
        }
    }

    /// `blockdev-snapshot-delete-internal-sync`: deletes the internal snapshot with `id`
    /// and/or `name` of the root node `device` and returns what it was.
    pub fn blockdev_snapshot_delete_internal_sync(
        &self,
        device: &str,
        id: Option<&str>,
        name: Option<&str>,
    ) -> Result<SnapshotInfo> {
        crate::drain::drain_all_begin();
        let r = self.snapshot_delete_drained(device, id, name);
        crate::drain::drain_all_end();
        r
    }

    fn snapshot_delete_drained(
        &self,
        device: &str,
        id: Option<&str>,
        name: Option<&str>,
    ) -> Result<SnapshotInfo> {
        let _g = crate::graph_lock::rdlock();
        let bs = self.root_bs(device)?;
        if id.is_none() && name.is_none() {
            return Err(Error::generic("Name or id must be provided"));
        }
        let Some(sn) = bs.snapshot_find(id, name)? else {
            return Err(Error::generic(format!(
                "Snapshot with id '{}' and name '{}' does not exist on device '{device}'",
                id.unwrap_or("(null)"),
                name.unwrap_or("(null)")
            )));
        };
        bs.snapshot_delete(id, name, &self.device_or_node_name(&bs))?;
        Ok(SnapshotInfo {
            id: sn.id_str,
            name: sn.name,
            vm_state_size: sn.vm_state_size as i64,
            date_sec: i64::from(sn.date_sec),
            date_nsec: i64::from(sn.date_nsec),
            vm_clock_sec: (sn.vm_clock_nsec / 1_000_000_000) as i64,
            vm_clock_nsec: (sn.vm_clock_nsec % 1_000_000_000) as i64,
            icount: sn.icount.map(|i| i as i64),
        })
    }

    /// `qemu-img snapshot -a`: `bdrv_snapshot_goto()` of the node `node_name` to the snapshot
    /// with the id or name `snapshot`.
    pub fn snapshot_goto(&self, node_name: &str, snapshot: &str) -> Result<()> {
        let bs = self.lookup_bs(node_name)?;
        let _d = bs.drained();
        let _g = crate::graph_lock::rdlock();
        bs.snapshot_goto(snapshot)
    }

    /// `blockdev-snapshot`: puts the node `overlay` on top of `node`, which becomes its
    /// backing image. The parents of `node` move to `overlay`.
    pub fn blockdev_snapshot(&self, node: &str, overlay: &str) -> Result<()> {
        let old = self.lookup_bs(node)?;
        let _d = old.drained();
        // Make sure the node did not change with the drain.
        let check_bs = self.lookup_bs(node)?;
        if !Arc::ptr_eq(&old, &check_bs) {
            return Err(Error::generic(format!(
                "Block node of device '{node}' unexpectedly changed"
            )));
        }
        if !old.is_inserted() {
            return Err(Error::generic(format!(
                "Device '{}' has no medium",
                self.device_or_node_name(&old)
            )));
        }
        if !old.read_only() {
            old.flush().map_err(|e| {
                Error::from_io(
                    format!("Write to node '{}' failed", self.device_or_node_name(&old)),
                    e,
                )
            })?;
        }
        let new = self.lookup_bs(overlay)?;
        // Only an overlay whose parents do not expect a valid image yet, like a mirror
        // target, may get a backing image while in use.
        if new.cumulative_perm().0 & BLK_PERM_CONSISTENT_READ != 0 {
            return Err(Error::generic("The overlay is already in use"));
        }
        if new.is_filter() {
            return Err(Error::generic("Filters cannot be used as overlays"));
        }
        if new.cow_child().is_some() {
            return Err(Error::generic("The overlay already has a backing image"));
        }
        if !new.def.is_some_and(|d| d.supports_backing) {
            return Err(Error::generic("The overlay does not support backing images"));
        }
        // Older QEMU versions allowed an active overlay on an inactive node, which taking a
        // snapshot of a stopped VM after migrating it to a file relies on. Inactivate the
        // overlay to keep that working.
        if old.is_inactive() && !new.is_inactive() {
            crate::drain::drain_all_begin();
            let r = new.inactivate();
            crate::drain::drain_all_end();
            r?;
        }
        Node::append(&new, &old)
    }
}

impl BlockBackend {
    fn with_root<T>(&self, f: impl FnOnce(&Arc<Node>) -> io::Result<T>) -> io::Result<T> {
        match self.root() {
            Some(bs) => f(&bs),
            None => Err(errno(ENOMEDIUM)),
        }
    }

    /// `blk_make_zero()`: zeroes the whole medium, skipping what already reads as zeroes.
    pub fn make_zero(&self, may_unmap: bool) -> io::Result<()> {
        self.with_root(|bs| {
            self.check_io_write(bs, 0, 0)?;
            bs.make_zero(if may_unmap { BDRV_REQ_MAY_UNMAP } else { 0 })
        })
    }

    /// `blk_co_is_all_zeroes()`: whether the whole medium is known to read as zeroes.
    pub fn is_all_zeroes(&self) -> io::Result<bool> {
        self.with_root(|bs| bs.is_all_zeroes())
    }

    /// `bdrv_has_zero_init(blk_bs(blk))`: whether a new image of this medium reads as zeroes.
    pub fn has_zero_init(&self) -> bool {
        self.root().is_some_and(|bs| bs.has_zero_init())
    }

    /// `blk_make_empty()`: drops everything the image holds itself. Needs the `write`
    /// permission.
    pub fn make_empty(&self) -> Result<()> {
        let Some(bs) = self.root() else {
            return Err(Error::generic("No medium inserted"));
        };
        // QEMU asserts this.
        if self.check_io_write(&bs, 0, 0).is_err() {
            return Err(Error::generic("blk_make_empty() needs the 'write' permission"));
        }
        bs.make_empty()
    }

    /// `bdrv_co_change_backing_file(blk_bs(blk), ...)`, what `qemu-img rebase` and `qemu-img
    /// commit` do to record a new backing file. With `require`, a backing file needs a
    /// format.
    pub fn change_backing_file(
        &self,
        file: Option<&str>,
        format: Option<&str>,
        require: bool,
    ) -> io::Result<()> {
        self.with_root(|bs| bs.change_backing_file(file, format, require))
    }

    /// `blk_load_vmstate()`: reads the VM state saved at `pos`.
    pub fn load_vmstate(&self, pos: u64, buf: &mut [u8]) -> io::Result<()> {
        self.with_root(|bs| bs.load_vmstate(pos, buf))
    }

    /// `blk_save_vmstate()`: saves VM state at `pos`, flushed when the write cache is off.
    pub fn save_vmstate(&self, pos: u64, buf: &[u8]) -> io::Result<()> {
        self.with_root(|bs| {
            bs.save_vmstate(pos, buf)?;
            if self.enable_write_cache() { Ok(()) } else { bs.flush() }
        })
    }

    /// `blk_probe_blocksizes()`: the logical and physical block size of a host device.
    pub fn probe_blocksizes(&self) -> io::Result<(u32, u32)> {
        self.with_root(|bs| bs.probe_blocksizes())
    }

    /// `blk_eject()`: opens (`eject_flag`) or closes the tray of the host drive, and tells
    /// the monitor the guest saw the tray move either way.
    pub fn eject(&self, eject_flag: bool) {
        if let Some(bs) = self.root() {
            bs.driver.eject(&bs, eject_flag);
        }
        crate::event::emit(crate::event::BlockEvent::DeviceTrayMoved {
            device: self.name().to_string(),
            id: self.attached_dev().unwrap_or_default(),
            tray_open: eject_flag,
        });
    }

    /// `blk_lock_medium()`: locks or unlocks the medium in the host drive.
    pub fn lock_medium(&self, locked: bool) {
        if let Some(bs) = self.root() {
            bs.driver.lock_medium(&bs, locked);
        }
    }

    /// `qemu-img map`: how the start of the range reads, through the backing chain.
    pub fn map_entry(&self, offset: u64, bytes: u64) -> io::Result<MapEntry> {
        self.with_root(|bs| bs.map_entry(offset, bytes))
    }

    /// qemu-io `break`: suspends requests at the blkdebug `event` under `tag`.
    pub fn debug_breakpoint(&self, event: &str, tag: &str) -> io::Result<()> {
        self.with_root(|bs| bs.debug_breakpoint(event, tag))
    }

    /// qemu-io `remove_break`.
    pub fn debug_remove_breakpoint(&self, tag: &str) -> io::Result<()> {
        self.with_root(|bs| bs.debug_remove_breakpoint(tag))
    }

    /// qemu-io `resume`: lets the request suspended under `tag` go on.
    pub fn debug_resume(&self, tag: &str) -> io::Result<()> {
        self.with_root(|bs| bs.debug_resume(tag))
    }

    /// qemu-io `wait_break` polls this: whether a request is suspended under `tag`.
    pub fn debug_is_suspended(&self, tag: &str) -> bool {
        self.root().is_some_and(|bs| bs.debug_is_suspended(tag))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use ruvm_qapi::json;
    use ruvm_qapi::types::{BlockdevAmendOptions, BlockdevOptions};
    use ruvm_qapi::visit::{QObjectInputVisitor, Visit};

    use super::*;
    use crate::event::{BlockEvent, set_event_hook};
    use crate::perm::BLK_PERM_ALL;

    fn add(g: &BlockGraph, s: &str) {
        let mut v = QObjectInputVisitor::new(json::from_str(s).unwrap());
        let mut o = BlockdevOptions::default();
        BlockdevOptions::visit(&mut v, None, &mut o).unwrap();
        g.blockdev_add(o).unwrap();
    }

    fn null(g: &BlockGraph, name: &str) {
        add(
            g,
            &format!(
                r#"{{"driver": "null-co", "node-name": "{name}", "size": 1048576,
                    "read-zeroes": true}}"#
            ),
        );
    }

    fn err<T: std::fmt::Debug>(r: Result<T>) -> String {
        r.unwrap_err().message().to_string()
    }

    #[test]
    fn set_active() {
        let g = BlockGraph::new();
        null(&g, "act0");
        add(&g, r#"{"driver": "raw", "node-name": "act1", "file": "act0"}"#);
        assert_eq!(
            err(g.blockdev_set_active(Some("nope"), false)),
            "Failed to find node with node-name='nope'"
        );
        assert_eq!(err(g.blockdev_set_active(Some("act0"), false)), "Node has active parent node");

        g.blockdev_set_active(Some("act1"), false).unwrap();
        let (a0, a1) = (g.find_node("act0").unwrap(), g.find_node("act1").unwrap());
        assert!(a0.is_inactive() && a1.is_inactive());
        g.blockdev_set_active(Some("act1"), true).unwrap();
        assert!(!a0.is_inactive() && !a1.is_inactive());

        // A backend that may write keeps its permissions here.
        let blk = BlockBackend::new(&g, "act1", BLK_PERM_WRITE, BLK_PERM_ALL).unwrap();
        assert_eq!(
            err(g.blockdev_set_active(Some("act1"), false)),
            "Failed to inactivate node: Operation not permitted"
        );
        assert!(!a1.is_inactive());
        drop(blk);
    }

    #[test]
    fn external_snapshot_errors() {
        let g = BlockGraph::new();
        null(&g, "ext0");
        null(&g, "ext1");
        null(&g, "ext2");
        add(&g, r#"{"driver": "copy-on-read", "node-name": "extf", "file": "ext2"}"#);
        assert_eq!(
            err(g.blockdev_snapshot("ext0", "nope")),
            "Cannot find device='nope' nor node-name='nope'"
        );
        assert_eq!(
            err(g.blockdev_snapshot("ext0", "ext1")),
            "The overlay does not support backing images"
        );
        assert_eq!(err(g.blockdev_snapshot("ext0", "extf")), "Filters cannot be used as overlays");
        let blk = BlockBackend::new(&g, "ext1", BLK_PERM_CONSISTENT_READ, BLK_PERM_ALL).unwrap();
        assert_eq!(err(g.blockdev_snapshot("ext0", "ext1")), "The overlay is already in use");
        drop(blk);
    }

    #[test]
    fn internal_snapshot_errors() {
        let g = BlockGraph::new();
        null(&g, "int0");
        add(&g, r#"{"driver": "null-co", "node-name": "int1", "read-only": true}"#);
        add(&g, r#"{"driver": "raw", "node-name": "int2", "file": "int0"}"#);
        assert_eq!(err(g.blockdev_snapshot_internal_sync("int0", "s")), "Need a root block node");
        assert_eq!(
            err(g.blockdev_snapshot_internal_sync("int1", "s")),
            "Device 'int1' is read only"
        );
        assert_eq!(
            err(g.blockdev_snapshot_internal_sync("int2", "s")),
            "Block format 'raw' used by device 'int2' does not support internal snapshots"
        );
        assert_eq!(
            err(g.blockdev_snapshot_delete_internal_sync("int2", None, None)),
            "Name or id must be provided"
        );
        assert_eq!(err(g.check("int2", 0)), "This image format does not support checks");
    }

    #[test]
    fn make_empty_and_zero() {
        let g = BlockGraph::new();
        null(&g, "mk0");
        let blk = BlockBackend::new(&g, "mk0", BLK_PERM_CONSISTENT_READ, BLK_PERM_ALL).unwrap();
        assert_eq!(err(blk.make_empty()), "blk_make_empty() needs the 'write' permission");
        drop(blk);
        let blk = BlockBackend::new(&g, "mk0", BLK_PERM_WRITE, BLK_PERM_ALL).unwrap();
        assert_eq!(err(blk.make_empty()), "null-co does not support emptying nodes");
        assert!(blk.is_all_zeroes().unwrap());
        let empty = BlockBackend::new_empty(None, 0, BLK_PERM_ALL, false);
        assert_eq!(err(empty.make_empty()), "No medium inserted");
        assert_eq!(empty.map_entry(0, 512).unwrap_err().raw_os_error(), Some(ENOMEDIUM));
    }

    #[test]
    fn map() {
        let g = BlockGraph::new();
        null(&g, "map0");
        add(&g, r#"{"driver": "raw", "node-name": "map1", "file": "map0"}"#);
        let blk = BlockBackend::new(&g, "map1", 0, BLK_PERM_ALL).unwrap();
        let e = blk.map_entry(0, 1 << 20).unwrap();
        assert_eq!((e.start, e.length, e.depth), (0, 1 << 20, 0));
        assert!(e.zero && !e.data && !e.compressed && e.present);
        assert_eq!(e.offset, Some(0));
        // qemu-img 11.1 names the node the same way.
        assert_eq!(
            e.filename.as_deref(),
            Some(r#"json:{"read-zeroes": true, "driver": "null-co", "size": 1048576}"#)
        );
    }

    /// `bdrv_refresh_filename()`: plain names where they open the same node again, `json:`
    /// otherwise, as qemu-img 11.1 prints them.
    #[test]
    fn json_filenames() {
        let g = BlockGraph::new();
        add(&g, r#"{"driver": "null-co", "node-name": "jf0"}"#);
        add(&g, r#"{"driver": "raw", "node-name": "jf1", "file": "jf0"}"#);
        add(
            &g,
            r#"{"driver": "raw", "node-name": "jf2", "offset": 512,
                "file": {"driver": "null-co", "size": 1048576}}"#,
        );
        add(&g, r#"{"driver": "copy-on-read", "node-name": "jf3", "file": "jf0"}"#);
        let name = |n: &str| g.find_node(n).unwrap().filename().unwrap();
        assert_eq!(name("jf0"), "null-co://");
        assert_eq!(name("jf1"), "null-co://");
        assert_eq!(
            name("jf2"),
            r#"json:{"offset": 512, "driver": "raw", "file": {"driver": "null-co", "size": 1048576}}"#
        );
        assert_eq!(
            name("jf3"),
            r#"json:{"driver": "copy-on-read", "file": {"driver": "null-co"}}"#
        );
    }

    #[test]
    fn eject_event() {
        let blk = BlockBackend::new_empty(Some("ruvm-eject-test".into()), 0, BLK_PERM_ALL, false);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        set_event_hook(Some(Arc::new(move |e: &BlockEvent| {
            if let BlockEvent::DeviceTrayMoved { device, tray_open, .. } = e {
                if device == "ruvm-eject-test" {
                    s.lock().unwrap().push(*tray_open);
                }
            }
        })));
        blk.eject(true);
        blk.eject(false);
        set_event_hook(None);
        assert_eq!(*seen.lock().unwrap(), [true, false]);
    }

    #[test]
    fn amend_errors() {
        let g = BlockGraph::new();
        null(&g, "am0");
        let amend = || {
            let s = r#"{"driver": "luks", "state": "inactive", "keyslot": 0}"#;
            let mut v = QObjectInputVisitor::new(json::from_str(s).unwrap());
            let mut o = BlockdevAmendOptions::default();
            BlockdevAmendOptions::visit(&mut v, None, &mut o).unwrap();
            o
        };
        assert_eq!(
            err(g.x_blockdev_amend("nope", amend(), None)),
            "Cannot find device='' nor node-name='nope'"
        );
        assert_eq!(
            err(g.x_blockdev_amend("am0", amend(), Some(true))),
            "x-blockdev-amend doesn't support changing the block driver"
        );
    }
}
