// SPDX-License-Identifier: GPL-2.0-or-later

//! The block-dirty-bitmap-* QMP commands from block/monitor/bitmap-qmp-cmds.c.

use std::sync::Arc;

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{
    BlockDirtyBitmap, BlockDirtyBitmapAdd, BlockDirtyBitmapMerge, BlockDirtyBitmapOrStr,
};

use super::dirty::{BDRV_BITMAP_ALLOW_RO, BDRV_BITMAP_BUSY, BDRV_BITMAP_DEFAULT, BDRV_BITMAP_RO};
use super::{DirtyBitmap, HBitmap};
use crate::graph::BlockGraph;
use crate::node::{BDRV_SECTOR_SIZE, Node};

impl BlockGraph {
    /// `block_dirty_bitmap_lookup()`: the bitmap `name` of the node or device `node`.
    pub(crate) fn dirty_bitmap_lookup(
        &self,
        node: &str,
        name: &str,
    ) -> Result<(Arc<Node>, Arc<DirtyBitmap>)> {
        let Ok(bs) = self.lookup_bs(node) else {
            return Err(Error::generic(format!("Node '{node}' not found")));
        };
        match bs.find_dirty_bitmap(name) {
            Some(bm) => Ok((bs, bm)),
            None => Err(Error::generic(format!("Dirty bitmap '{name}' not found"))),
        }
    }

    /// `block-dirty-bitmap-add`.
    pub fn block_dirty_bitmap_add(&self, arg: &BlockDirtyBitmapAdd) -> Result<()> {
        self.dirty_bitmap_add(arg).map(|_| ())
    }

    /// `block_dirty_bitmap_add()`: the command, returning the new bitmap.
    pub(crate) fn dirty_bitmap_add(&self, arg: &BlockDirtyBitmapAdd) -> Result<Arc<DirtyBitmap>> {
        let name = arg.name.as_str();
        if name.is_empty() {
            return Err(Error::generic("Bitmap name cannot be empty"));
        }
        let bs = self.lookup_bs(&arg.node)?;
        let granularity = match arg.granularity {
            Some(g) => {
                if u64::from(g) < BDRV_SECTOR_SIZE || !g.is_power_of_two() {
                    return Err(Error::generic("Granularity must be power of 2 and at least 512"));
                }
                g
            }
            None => bs.default_bitmap_granularity(),
        };
        let persistent = arg.persistent.unwrap_or(false);
        if persistent {
            bs.can_store_new_dirty_bitmap(name, granularity)?;
        }
        let bitmap = bs.create_dirty_bitmap(granularity, Some(name))?;
        if arg.disabled.unwrap_or(false) {
            bitmap.disable();
        }
        bitmap.set_persistence(persistent);
        Ok(bitmap)
    }

    /// `block-dirty-bitmap-remove`.
    pub fn block_dirty_bitmap_remove(&self, arg: &BlockDirtyBitmap) -> Result<()> {
        self.dirty_bitmap_remove(&arg.node, &arg.name, true).map(|_| ())
    }

    /// `block_dirty_bitmap_remove()`: with `release` false the bitmap stays on the node and
    /// is returned, as transactions want it.
    pub(crate) fn dirty_bitmap_remove(
        &self,
        node: &str,
        name: &str,
        release: bool,
    ) -> Result<Arc<DirtyBitmap>> {
        let (bs, bitmap) = self.dirty_bitmap_lookup(node, name)?;
        bitmap.check(BDRV_BITMAP_BUSY | BDRV_BITMAP_RO)?;
        if bitmap.get_persistence() {
            bs.remove_persistent_dirty_bitmap(name)?;
        }
        if release {
            bs.release_dirty_bitmap(&bitmap);
        }
        Ok(bitmap)
    }

    /// `block-dirty-bitmap-clear`.
    pub fn block_dirty_bitmap_clear(&self, arg: &BlockDirtyBitmap) -> Result<()> {
        let (_, bitmap) = self.dirty_bitmap_lookup(&arg.node, &arg.name)?;
        bitmap.check(BDRV_BITMAP_DEFAULT)?;
        bitmap.clear(false);
        Ok(())
    }

    /// `block-dirty-bitmap-enable`.
    pub fn block_dirty_bitmap_enable(&self, arg: &BlockDirtyBitmap) -> Result<()> {
        let (_, bitmap) = self.dirty_bitmap_lookup(&arg.node, &arg.name)?;
        bitmap.check(BDRV_BITMAP_ALLOW_RO)?;
        bitmap.enable();
        Ok(())
    }

    /// `block-dirty-bitmap-disable`.
    pub fn block_dirty_bitmap_disable(&self, arg: &BlockDirtyBitmap) -> Result<()> {
        let (_, bitmap) = self.dirty_bitmap_lookup(&arg.node, &arg.name)?;
        bitmap.check(BDRV_BITMAP_ALLOW_RO)?;
        bitmap.disable();
        Ok(())
    }

    /// `block-dirty-bitmap-merge`.
    pub fn block_dirty_bitmap_merge(&self, arg: &BlockDirtyBitmapMerge) -> Result<()> {
        self.dirty_bitmap_merge(&arg.node, &arg.target, &arg.bitmaps, false).map(|_| ())
    }

    /// `block_dirty_bitmap_merge()`: ORs `bitmaps` into `target`. The sources are merged
    /// into an anonymous bitmap first, so `target` is unchanged when one of them fails. With
    /// `backup` the old bits of `target` are returned.
    pub(crate) fn dirty_bitmap_merge(
        &self,
        node: &str,
        target: &str,
        bitmaps: &[BlockDirtyBitmapOrStr],
        backup: bool,
    ) -> Result<(Arc<DirtyBitmap>, Option<HBitmap>)> {
        let bs = self.lookup_bs(node)?;
        let Some(dst) = bs.find_dirty_bitmap(target) else {
            return Err(Error::generic(format!("Dirty bitmap '{target}' not found")));
        };
        let anon = bs.create_dirty_bitmap(dst.granularity(), None)?;
        let res = (|| {
            for b in bitmaps {
                let src = match b {
                    BlockDirtyBitmapOrStr::Local(name) => match bs.find_dirty_bitmap(name) {
                        Some(s) => s,
                        None => {
                            return Err(Error::generic(format!("Dirty bitmap '{name}' not found")));
                        }
                    },
                    BlockDirtyBitmapOrStr::External(e) => {
                        self.dirty_bitmap_lookup(&e.node, &e.name)?.1
                    }
                };
                anon.merge(&src, false)?;
            }
            dst.merge(&anon, backup)
        })();
        bs.release_dirty_bitmap(&anon);
        res.map(|old| (dst, old))
    }
}
