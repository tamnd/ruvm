// SPDX-License-Identifier: GPL-2.0-or-later

//! `throttle` from block/throttle.c: a filter whose requests wait for the limits of a throttle
//! group, which must exist already (`object-add throttle-group`).
//!
//! Differences from QEMU: requests wait on the calling thread, as described in
//! [`crate::throttle::groups`]; there are no AioContexts to attach and detach.

use std::io;

use ruvm_base::{Error, Result};
use ruvm_qapi::QDict;
use ruvm_qapi::types::BlockdevOptionsU;

use crate::drivers::{DriverDef, OpenArgs};
use crate::node::{
    BDRV_CHILD_FILTERED, BDRV_CHILD_PRIMARY, BDRV_REQ_WRITE_COMPRESSED, BDRV_REQ_WRITE_UNCHANGED,
    Driver, Node, ReopenState,
};
use crate::throttle::ThrottleDirection;
use crate::throttle::groups::{ThrottleGroupMember, throttle_group_exists};

/// `QEMU_OPT_THROTTLE_GROUP_NAME`.
const THROTTLE_GROUP: &str = "throttle-group";

/// `bdrv_throttle`.
pub(crate) static THROTTLE: DriverDef =
    DriverDef::filter("throttle", throttle_open).with_strong_opts(&["throttle-group"]);

struct ThrottleDriver {
    tgm: ThrottleGroupMember,
    /// `bs->supported_write_flags`.
    write_flags: u32,
    /// `bs->supported_zero_flags`.
    zero_flags: u32,
}

/// The checks of `throttle_parse_options()` on the group name.
fn check_group(group: Option<&str>) -> Result<String> {
    let Some(group) = group else {
        return Err(Error::generic("Please specify a throttle group"));
    };
    if !throttle_group_exists(group) {
        return Err(Error::generic(format!("Throttle group '{group}' does not exist")));
    }
    Ok(group.to_string())
}

/// `throttle_parse_options()` on the options of a reopen: takes `throttle-group` out.
fn parse_options(options: &mut QDict) -> Result<String> {
    let group = options.get_str(THROTTLE_GROUP).map(str::to_string);
    options.remove(THROTTLE_GROUP);
    check_group(group.as_deref())
}

/// `throttle_open()`.
fn throttle_open(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::Throttle(o) = opts else {
        unreachable!("the throttle driver gets throttle options");
    };
    let file = args.open_child(*o.file, "file", BDRV_CHILD_FILTERED | BDRV_CHILD_PRIMARY)?;
    let group = check_group(Some(&o.throttle_group))?;
    let tgm = ThrottleGroupMember::new();
    tgm.register(&group);
    Ok(Box::new(ThrottleDriver {
        tgm,
        write_flags: file.driver.supported_write_flags() | BDRV_REQ_WRITE_UNCHANGED,
        zero_flags: file.driver.supported_zero_flags() | BDRV_REQ_WRITE_UNCHANGED,
    }))
}

impl Driver for ThrottleDriver {
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.tgm.io_limits_intercept(buf.len() as u64, ThrottleDirection::Read);
        bs.file().pread(offset, buf)
    }

    fn pwrite(&self, bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        self.pwrite_flags(bs, offset, buf, 0)
    }

    fn pwrite_flags(&self, bs: &Node, offset: u64, buf: &[u8], flags: u32) -> io::Result<()> {
        self.tgm.io_limits_intercept(buf.len() as u64, ThrottleDirection::Write);
        bs.file().pwrite_flags(offset, buf, flags)
    }

    fn supported_write_flags(&self) -> u32 {
        self.write_flags
    }

    fn pwrite_zeroes(&self, bs: &Node, offset: u64, bytes: u64, may_unmap: bool) -> io::Result<()> {
        let flags = if may_unmap { crate::node::BDRV_REQ_MAY_UNMAP } else { 0 };
        self.pwrite_zeroes_flags(bs, offset, bytes, flags)
    }

    fn pwrite_zeroes_flags(
        &self,
        bs: &Node,
        offset: u64,
        bytes: u64,
        flags: u32,
    ) -> io::Result<()> {
        self.tgm.io_limits_intercept(bytes, ThrottleDirection::Write);
        bs.file().pwrite_zeroes_flags(offset, bytes, flags)
    }

    fn supported_zero_flags(&self) -> u32 {
        self.zero_flags
    }

    fn pdiscard(&self, bs: &Node, offset: u64, bytes: u64) -> io::Result<()> {
        self.tgm.io_limits_intercept(bytes, ThrottleDirection::Write);
        bs.file().pdiscard(offset, bytes)
    }

    /// `throttle_co_pwritev_compressed()`.
    fn pwrite_compressed(&self, bs: &Node, offset: u64, buf: &[u8]) -> Option<io::Result<()>> {
        Some(self.pwrite_flags(bs, offset, buf, BDRV_REQ_WRITE_COMPRESSED))
    }

    fn flush_to_disk(&self, bs: &Node) -> io::Result<()> {
        bs.file().flush()
    }

    fn getlength(&self, bs: &Node) -> io::Result<u64> {
        bs.file().getlength()
    }

    fn has_truncate(&self) -> bool {
        false
    }

    /// `throttle_close()`.
    fn close(&self, _bs: &Node) {
        self.tgm.unregister();
    }

    /// `throttle_reopen_prepare()`.
    fn reopen_prepare(&self, _bs: &Node, state: &mut ReopenState) -> Option<Result<()>> {
        let r = parse_options(&mut state.options).map(|group| {
            state.opaque = Some(Box::new(group));
        });
        Some(r)
    }

    /// `throttle_reopen_commit()`: moves to the new group if the name changed.
    fn reopen_commit(&self, _bs: &Node, state: &mut ReopenState) {
        let Some(group) = state.opaque.take().and_then(|o| o.downcast::<String>().ok()) else {
            return;
        };
        if self.tgm.group_name().as_deref() != Some(group.as_str()) {
            self.tgm.unregister();
            self.tgm.register(&group);
        }
    }

    fn reopen_abort(&self, _bs: &Node, state: &mut ReopenState) {
        state.opaque = None;
    }

    /// `throttle_drain_begin()`.
    fn drain_begin(&self, _bs: &Node) {
        self.tgm.io_limits_disable_begin();
    }

    /// `throttle_drain_end()`.
    fn drain_end(&self, _bs: &Node) {
        self.tgm.io_limits_disable_end();
    }
}

#[cfg(test)]
mod tests {
    use ruvm_qapi::types::{ThrottleGroupProperties, ThrottleLimits};

    use super::*;
    use crate::graph::BlockGraph;

    fn add_group(name: &str) {
        let props = ThrottleGroupProperties {
            limits: Some(ThrottleLimits { bps_total: Some(1 << 40), ..Default::default() }),
            ..Default::default()
        };
        crate::throttle::groups::throttle_group_add(name, &props).unwrap();
    }

    fn open(g: &BlockGraph, node: &str, group: &str) -> Result<String> {
        let mut o = QDict::new();
        o.put("driver", "throttle");
        o.put("node-name", node);
        o.put("throttle-group", group);
        o.put("file.driver", "null-co");
        g.open_image(None, o)
    }

    #[test]
    fn open_errors() {
        let g = BlockGraph::new();
        let e = open(&g, "thr-err0", "filter-nogroup").unwrap_err();
        assert_eq!(e.message(), "Throttle group 'filter-nogroup' does not exist");

        let mut o = QDict::new();
        assert_eq!(parse_options(&mut o).unwrap_err().message(), "Please specify a throttle group");
        o.put("throttle-group", "filter-nogroup");
        o.put("other", "x");
        assert!(parse_options(&mut o).is_err());
        assert!(o.get("throttle-group").is_none());
        assert!(o.get("other").is_some());
    }

    #[test]
    fn io_and_references() {
        add_group("filter-g0");
        add_group("filter-g1");
        let g = BlockGraph::new();
        assert_eq!(open(&g, "thr0", "filter-g0").unwrap(), "thr0");
        let bs = g.lookup_bs("thr0").unwrap();
        let mut buf = [1u8; 512];
        bs.pread(0, &mut buf).unwrap();
        bs.pwrite(0, &buf).unwrap();
        bs.flush().unwrap();

        let e = g.throttle_group_del("filter-g0").unwrap_err();
        assert_eq!(e.message(), "Cannot delete throttle group 'filter-g0' with active references");

        // Moving the node to the other group frees the first one.
        let mut o = QDict::new();
        o.put("throttle-group", "filter-g1");
        g.reopen_node("thr0", o, true).unwrap();
        g.throttle_group_del("filter-g0").unwrap();

        let mut o = QDict::new();
        o.put("throttle-group", "filter-g0");
        let e = g.reopen_node("thr0", o, true).unwrap_err();
        assert_eq!(e.message(), "Throttle group 'filter-g0' does not exist");

        drop(bs);
        g.blockdev_del("thr0").unwrap();
        // A drain_all() of a test running alongside may hold the node for a moment, and the
        // node lets go of its group when it goes away.
        let mut r = g.throttle_group_del("filter-g1");
        for _ in 0..500 {
            if r.is_ok() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
            r = g.throttle_group_del("filter-g1");
        }
        r.unwrap();
    }
}
