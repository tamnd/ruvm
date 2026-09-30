// SPDX-License-Identifier: GPL-2.0-or-later

//! The driver hooks for persistent dirty bitmaps: `bdrv_supports_persistent_dirty_bitmap`,
//! `bdrv_co_can_store_new_dirty_bitmap` and `bdrv_co_remove_persistent_dirty_bitmap` of
//! `BlockDriver`.
//!
//! A format that stores bitmaps in the image (qcow2) returns an implementation from
//! [`Driver::persistent_bitmaps`](crate::node::Driver::persistent_bitmaps). Loading and
//! storing need no hook: the driver loads its bitmaps at open time with
//! [`Node::create_dirty_bitmap`], [`DirtyBitmap::deserialize_part`] and the flag setters
//! (`set_persistence`, `set_readonly`, `set_inconsistent`, `disable`), and stores them at
//! close or inactivation by walking [`Node::dirty_bitmaps`] and taking the ones whose
//! [`DirtyBitmap::get_persistence`] is true, reading them with
//! [`DirtyBitmap::serialize_part`].

use ruvm_base::Result;

#[allow(unused_imports, reason = "used by the doc links")]
use super::DirtyBitmap;
use crate::node::Node;

/// What a format driver that stores dirty bitmaps in its image provides.
pub(crate) trait PersistentBitmaps: Send + Sync {
    /// `bdrv_supports_persistent_dirty_bitmap()`: whether the image as opened can hold
    /// bitmaps (for qcow2, whether it is version 3).
    fn supports_persistent_dirty_bitmap(&self, bs: &Node) -> bool;

    /// `bdrv_co_can_store_new_dirty_bitmap()`: whether a new persistent bitmap `name` with
    /// `granularity` fits in the image. The error is what block-dirty-bitmap-add reports.
    fn can_store_new_dirty_bitmap(&self, bs: &Node, name: &str, granularity: u32) -> Result<()>;

    /// `bdrv_co_remove_persistent_dirty_bitmap()`: drops the stored copy of `name`, if any.
    fn remove_persistent_dirty_bitmap(&self, bs: &Node, name: &str) -> Result<()>;
}
