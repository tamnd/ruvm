// SPDX-License-Identifier: GPL-2.0-or-later

//! `snapshot-access` from block/snapshot-access.c: a read-only view of the snapshot a
//! `copy-before-write` node keeps, for image fleecing. Reads and block status go to the
//! snapshot API of the `file` child, a discard tells the child that the range will not be
//! read again, and writes fail with `ENOTSUP`.
//!
//! Differences from QEMU:
//!
//! - The generic read path has no per-driver read flags, so the `ENOTSUP` QEMU returns for
//!   a read with flags cannot happen.

use std::io;

use ruvm_base::Result;
use ruvm_qapi::types::BlockdevOptionsU;

use crate::drivers::{DriverDef, OpenArgs};
use crate::filter::copy_before_write::{pdiscard_snapshot, preadv_snapshot, snapshot_block_status};
use crate::node::{BDRV_CHILD_DATA, BDRV_CHILD_PRIMARY, BlockStatus, Driver, Node};
use crate::perm::{BLK_PERM_ALL, PermCtx};

/// `bdrv_snapshot_access_drv`: neither a format nor a filter.
pub(crate) static SNAPSHOT_ACCESS: DriverDef = {
    let mut d = DriverDef::format("snapshot-access", snapshot_access_open);
    d.is_format = false;
    d
};

struct SnapshotAccess;

/// `snapshot_access_open()`.
fn snapshot_access_open(
    args: &mut OpenArgs<'_>,
    opts: BlockdevOptionsU,
) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::SnapshotAccess(o) = opts else {
        unreachable!("the snapshot-access driver gets snapshot-access options");
    };
    args.open_child(*o.file, "file", BDRV_CHILD_DATA | BDRV_CHILD_PRIMARY)?;
    Ok(Box::new(SnapshotAccess))
}

fn enotsup() -> io::Error {
    io::Error::from_raw_os_error(libc::ENOTSUP)
}

impl Driver for SnapshotAccess {
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        preadv_snapshot(&bs.file(), offset, buf)
    }

    fn pwrite(&self, _bs: &Node, _offset: u64, _buf: &[u8]) -> io::Result<()> {
        Err(enotsup())
    }

    fn pwrite_flags(&self, _bs: &Node, _offset: u64, _buf: &[u8], _flags: u32) -> io::Result<()> {
        Err(enotsup())
    }

    fn pwrite_zeroes(&self, _bs: &Node, _offset: u64, _bytes: u64, _unmap: bool) -> io::Result<()> {
        Err(enotsup())
    }

    fn pwrite_zeroes_flags(
        &self,
        _bs: &Node,
        _offset: u64,
        _bytes: u64,
        _flags: u32,
    ) -> io::Result<()> {
        Err(enotsup())
    }

    fn pdiscard(&self, bs: &Node, offset: u64, bytes: u64) -> io::Result<()> {
        pdiscard_snapshot(&bs.file(), offset, bytes)
    }

    fn block_status(
        &self,
        bs: &Node,
        _want: u32,
        offset: u64,
        bytes: u64,
    ) -> Option<io::Result<BlockStatus>> {
        Some(snapshot_block_status(&bs.file(), offset, bytes))
    }

    fn getlength(&self, bs: &Node) -> io::Result<u64> {
        bs.file().getlength()
    }

    fn has_truncate(&self) -> bool {
        false
    }

    fn exact_filename(&self, bs: &Node) -> Option<String> {
        bs.file().filename()
    }

    fn child_perm_for(&self, _ctx: &PermCtx<'_>, _perm: u64, _shared: u64) -> (u64, u64) {
        // Currently, we don't need any permissions. If bs->file provides snapshot-access
        // API, we can use it.
        (0, BLK_PERM_ALL)
    }
}
