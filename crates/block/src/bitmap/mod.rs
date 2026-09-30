// SPDX-License-Identifier: GPL-2.0-or-later

//! Dirty bitmaps: [`HBitmap`] (util/hbitmap.c), the per-node [`DirtyBitmap`]
//! (block/dirty-bitmap.c), the block-dirty-bitmap-* QMP commands
//! (block/monitor/bitmap-qmp-cmds.c) and the [`PersistentBitmaps`] hooks a format implements
//! to keep bitmaps in its image.
//!
//! Every completed write, write-zeroes and discard marks its range in each enabled bitmap
//! of the node it was sent to, and a resize resizes them; the request layer calls
//! [`Node::set_dirty`](crate::node::Node) and `dirty_bitmap_truncate` for that.

#![allow(
    dead_code,
    reason = "the whole dirty bitmap API is ported; qcow2 persistent bitmaps and the NBD export use the rest"
)]

mod dirty;
mod hbitmap;
mod persist;
mod qmp;

#[allow(unused_imports, reason = "for the jobs and the format drivers")]
pub(crate) use dirty::{
    BDRV_BITMAP_ALLOW_RO, BDRV_BITMAP_BUSY, BDRV_BITMAP_DEFAULT, BDRV_BITMAP_INCONSISTENT,
    BDRV_BITMAP_MAX_NAME_SIZE, BDRV_BITMAP_RO, DirtyBitmap, DirtyBitmapIter, DirtyBitmapList,
};
pub(crate) use hbitmap::HBitmap;
pub(crate) use persist::PersistentBitmaps;

#[cfg(test)]
mod hbitmap_tests;
