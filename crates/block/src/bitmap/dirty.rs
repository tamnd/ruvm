// SPDX-License-Identifier: GPL-2.0-or-later

//! `BdrvDirtyBitmap` from block/dirty-bitmap.c: the named and anonymous dirty bitmaps of a
//! node, which every write and discard marks.
//!
//! A node keeps its bitmaps in a [`DirtyBitmapList`], newest first like `bs->dirty_bitmaps`.
//! A bitmap is an `Arc<DirtyBitmap>` handle; its state sits behind its own lock, where QEMU
//! has one `dirty_bitmap_mutex` per node. A parent bitmap locks its successor while it holds
//! its own lock, never the other way round, and a merge copies the source before it locks the
//! destination, so no two bitmap locks are ever taken in both orders.
//!
//! Differences from QEMU:
//!
//! - `bdrv_get_device_or_node_name()` in "Can't store persistent bitmaps to %s" is the node
//!   name: a node here does not know the name of the block backend above it.
//! - `active_iterators` is not counted; [`DirtyBitmapIter`] does not borrow the bitmap, so
//!   releasing a bitmap under an iterator is not caught.

use std::sync::{Arc, Mutex, MutexGuard, Weak};

use ruvm_base::{Error, Result};
use ruvm_qapi::types::BlockDirtyInfo;

use super::hbitmap::{HBitmap, HBitmapIter};
use crate::node::{BDRV_SECTOR_SIZE, Node};

/// `BDRV_BITMAP_MAX_NAME_SIZE`.
pub(crate) const BDRV_BITMAP_MAX_NAME_SIZE: usize = 1023;

/// `BDRV_BITMAP_BUSY`: the bitmap must not be in use by an operation.
pub(crate) const BDRV_BITMAP_BUSY: u32 = 1;
/// `BDRV_BITMAP_RO`: the bitmap must be writable.
pub(crate) const BDRV_BITMAP_RO: u32 = 2;
/// `BDRV_BITMAP_INCONSISTENT`: the bitmap must be consistent.
pub(crate) const BDRV_BITMAP_INCONSISTENT: u32 = 4;
/// `BDRV_BITMAP_DEFAULT`.
pub(crate) const BDRV_BITMAP_DEFAULT: u32 =
    BDRV_BITMAP_BUSY | BDRV_BITMAP_RO | BDRV_BITMAP_INCONSISTENT;
/// `BDRV_BITMAP_ALLOW_RO`.
pub(crate) const BDRV_BITMAP_ALLOW_RO: u32 = BDRV_BITMAP_BUSY | BDRV_BITMAP_INCONSISTENT;

/// The bitmaps of a node, `bs->dirty_bitmaps`.
#[derive(Debug, Default)]
pub(crate) struct DirtyBitmapList {
    list: Mutex<Vec<Arc<DirtyBitmap>>>,
}

impl DirtyBitmapList {
    fn lock(&self) -> MutexGuard<'_, Vec<Arc<DirtyBitmap>>> {
        self.list.lock().unwrap()
    }
}

#[derive(Debug)]
struct State {
    bitmap: HBitmap,
    busy: bool,
    successor: Option<Arc<DirtyBitmap>>,
    name: Option<String>,
    size: u64,
    disabled: bool,
    readonly: bool,
    persistent: bool,
    inconsistent: bool,
    skip_store: bool,
}

/// A `BdrvDirtyBitmap`.
#[derive(Debug)]
pub(crate) struct DirtyBitmap {
    bs: Weak<Node>,
    state: Mutex<State>,
}

impl Node {
    /// `bdrv_find_dirty_bitmap()`.
    pub(crate) fn find_dirty_bitmap(&self, name: &str) -> Option<Arc<DirtyBitmap>> {
        self.io
            .dirty_bitmaps
            .lock()
            .iter()
            .find(|b| b.state().name.as_deref() == Some(name))
            .cloned()
    }

    /// Every bitmap of the node, newest first, `bdrv_dirty_bitmap_first()` and `_next()`.
    pub(crate) fn dirty_bitmaps(&self) -> Vec<Arc<DirtyBitmap>> {
        self.io.dirty_bitmaps.lock().clone()
    }

    /// `bdrv_create_dirty_bitmap()`: a new enabled bitmap covering the node. `granularity`
    /// is a power of two of at least 512.
    pub(crate) fn create_dirty_bitmap(
        &self,
        granularity: u32,
        name: Option<&str>,
    ) -> Result<Arc<DirtyBitmap>> {
        assert!(granularity.is_power_of_two() && u64::from(granularity) >= BDRV_SECTOR_SIZE);
        if let Some(n) = name {
            if self.find_dirty_bitmap(n).is_some() {
                return Err(Error::generic(format!("Bitmap already exists: {n}")));
            }
            if n.len() > BDRV_BITMAP_MAX_NAME_SIZE {
                return Err(Error::generic(format!("Bitmap name too long: {n}")));
            }
        }
        let size =
            self.getlength().map_err(|e| Error::from_io("could not get length of device", e))?;
        let bm = Arc::new(DirtyBitmap {
            bs: self.weak(),
            state: Mutex::new(State {
                bitmap: HBitmap::new(size, granularity.trailing_zeros()),
                busy: false,
                successor: None,
                name: name.map(str::to_string),
                size,
                disabled: false,
                readonly: false,
                persistent: false,
                inconsistent: false,
                skip_store: false,
            }),
        });
        self.io.dirty_bitmaps.lock().insert(0, bm.clone());
        Ok(bm)
    }

    /// `bdrv_release_dirty_bitmap()`: takes `bitmap` off the node. It must not be busy or
    /// have a successor.
    pub(crate) fn release_dirty_bitmap(&self, bitmap: &Arc<DirtyBitmap>) {
        {
            let s = bitmap.state();
            assert!(!s.busy, "releasing a busy bitmap");
            assert!(s.successor.is_none(), "releasing a bitmap with a successor");
        }
        self.io.dirty_bitmaps.lock().retain(|b| !Arc::ptr_eq(b, bitmap));
    }

    /// `bdrv_release_named_dirty_bitmaps()`: drops every named bitmap, for closing the node.
    /// The persistent ones stay in the image.
    pub(crate) fn release_named_dirty_bitmaps(&self) {
        self.io.dirty_bitmaps.lock().retain(|b| b.state().name.is_none());
    }

    /// `bdrv_set_dirty()`: marks `offset..offset + bytes` in every enabled bitmap. The write
    /// path calls this for every write and discard.
    pub(crate) fn set_dirty(&self, offset: u64, bytes: u64) {
        let list = self.io.dirty_bitmaps.lock();
        for bm in list.iter() {
            let mut s = bm.state();
            if s.disabled {
                continue;
            }
            assert!(!s.readonly, "a write to a node with a read-only bitmap");
            let end = (offset + bytes).min(s.size);
            if offset < end {
                s.bitmap.set(offset, end - offset);
            }
        }
    }

    /// `bdrv_dirty_bitmap_truncate()`: every bitmap follows a new length of the node.
    pub(crate) fn dirty_bitmap_truncate(&self, bytes: u64) {
        let list = self.io.dirty_bitmaps.lock();
        for bm in list.iter() {
            let mut s = bm.state();
            // QEMU asserts that no bitmap is busy here; resizing is blocked while one is.
            s.bitmap.truncate(bytes);
            s.size = bytes;
        }
    }

    /// `bdrv_has_readonly_bitmaps()`.
    pub(crate) fn has_readonly_bitmaps(&self) -> bool {
        self.io.dirty_bitmaps.lock().iter().any(|b| b.state().readonly)
    }

    /// `bdrv_has_named_bitmaps()`.
    #[allow(dead_code, reason = "for the qcow2 bitmap extension")]
    pub(crate) fn has_named_bitmaps(&self) -> bool {
        self.io.dirty_bitmaps.lock().iter().any(|b| b.state().name.is_some())
    }

    /// `bdrv_query_dirty_bitmaps()`: what `query-block` shows as `dirty-bitmaps`.
    pub(crate) fn query_dirty_bitmaps(&self) -> Vec<BlockDirtyInfo> {
        self.io
            .dirty_bitmaps
            .lock()
            .iter()
            .map(|bm| {
                let recording = bm.recording();
                let s = bm.state();
                BlockDirtyInfo {
                    name: s.name.clone(),
                    count: s.bitmap.count() as i64,
                    granularity: 1u32 << s.bitmap.granularity(),
                    recording,
                    busy: s.busy,
                    persistent: s.persistent,
                    inconsistent: s.inconsistent.then_some(true),
                }
            })
            .collect()
    }

    /// `bdrv_get_default_bitmap_granularity()`: the cluster size clamped to 4k..64k, or 64k.
    pub(crate) fn default_bitmap_granularity(&self) -> u32 {
        match self.get_info() {
            Ok(bdi) if bdi.cluster_size > 0 => (bdi.cluster_size as u32).clamp(4096, 65536),
            _ => 65536,
        }
    }

    /// `bdrv_supports_persistent_dirty_bitmap()`.
    #[allow(dead_code, reason = "for migration and the qcow2 driver")]
    pub(crate) fn supports_persistent_dirty_bitmap(&self) -> bool {
        self.driver.persistent_bitmaps().is_some_and(|p| p.supports_persistent_dirty_bitmap(self))
    }

    /// `bdrv_can_store_new_dirty_bitmap()`.
    pub(crate) fn can_store_new_dirty_bitmap(&self, name: &str, granularity: u32) -> Result<()> {
        match self.driver.persistent_bitmaps() {
            Some(p) => p.can_store_new_dirty_bitmap(self, name, granularity),
            None => Err(Error::from_io(
                format!("Can't store persistent bitmaps to {}", self.name),
                crate::node::errno(libc::ENOTSUP),
            )),
        }
    }

    /// `bdrv_remove_persistent_dirty_bitmap()`: a driver without persistent bitmaps has
    /// nothing to remove.
    pub(crate) fn remove_persistent_dirty_bitmap(&self, name: &str) -> Result<()> {
        match self.driver.persistent_bitmaps() {
            Some(p) => p.remove_persistent_dirty_bitmap(self, name),
            None => Ok(()),
        }
    }
}

impl DirtyBitmap {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap()
    }

    /// An anonymous disabled bitmap of `size` bytes that belongs to no node, for the
    /// internal bitmaps a driver makes before its node exists (the copy-before-write filter
    /// creates them with `bdrv_create_dirty_bitmap()` on its own node). Such a bitmap is not
    /// listed on any node, so writes never mark it and no query shows it.
    pub(crate) fn detached(size: u64, granularity: u32) -> Arc<DirtyBitmap> {
        assert!(granularity.is_power_of_two() && u64::from(granularity) >= BDRV_SECTOR_SIZE);
        Arc::new(DirtyBitmap {
            bs: Weak::new(),
            state: Mutex::new(State {
                bitmap: HBitmap::new(size, granularity.trailing_zeros()),
                busy: false,
                successor: None,
                name: None,
                size,
                disabled: true,
                readonly: false,
                persistent: false,
                inconsistent: false,
                skip_store: false,
            }),
        })
    }

    /// The node the bitmap belongs to.
    pub(crate) fn node(&self) -> Option<Arc<Node>> {
        self.bs.upgrade()
    }

    /// `bdrv_dirty_bitmap_name()`.
    pub(crate) fn name(&self) -> Option<String> {
        self.state().name.clone()
    }

    fn name_str(&self) -> String {
        self.name().unwrap_or_else(|| "(null)".to_string())
    }

    /// `bdrv_dirty_bitmap_size()`: the length it covers, in bytes.
    pub(crate) fn size(&self) -> u64 {
        self.state().size
    }

    /// `bdrv_dirty_bitmap_granularity()`.
    pub(crate) fn granularity(&self) -> u32 {
        1u32 << self.state().bitmap.granularity()
    }

    /// `bdrv_dirty_bitmap_has_successor()`.
    pub(crate) fn has_successor(&self) -> bool {
        self.state().successor.is_some()
    }

    /// The successor, for the jobs that work on it.
    #[allow(dead_code, reason = "for migration")]
    pub(crate) fn successor(&self) -> Option<Arc<DirtyBitmap>> {
        self.state().successor.clone()
    }

    /// Whether an operation holds the bitmap.
    pub(crate) fn busy(&self) -> bool {
        self.state().busy
    }

    /// `bdrv_dirty_bitmap_set_busy()`.
    pub(crate) fn set_busy(&self, busy: bool) {
        self.state().busy = busy;
    }

    /// `bdrv_dirty_bitmap_enabled()`.
    pub(crate) fn enabled(&self) -> bool {
        !self.state().disabled
    }

    /// `bdrv_dirty_bitmap_recording()`: enabled itself, or through an enabled successor.
    pub(crate) fn recording(&self) -> bool {
        let s = self.state();
        !s.disabled || s.successor.as_ref().is_some_and(|c| c.enabled())
    }

    /// `bdrv_dirty_bitmap_check()`: whether the bitmap may be used as `flags` say.
    pub(crate) fn check(&self, flags: u32) -> Result<()> {
        let s = self.state();
        let name = s.name.clone().unwrap_or_else(|| "(null)".to_string());
        if flags & BDRV_BITMAP_BUSY != 0 && s.busy {
            return Err(Error::generic(format!(
                "Bitmap '{name}' is currently in use by another operation and cannot be used"
            )));
        }
        if flags & BDRV_BITMAP_RO != 0 && s.readonly {
            return Err(Error::generic(format!(
                "Bitmap '{name}' is readonly and cannot be modified"
            )));
        }
        if flags & BDRV_BITMAP_INCONSISTENT != 0 && s.inconsistent {
            return Err(Error::generic(format!(
                "Bitmap '{name}' is inconsistent and cannot be used"
            ))
            .hint("Try block-dirty-bitmap-remove to delete this bitmap from disk\n"));
        }
        Ok(())
    }

    /// `bdrv_dirty_bitmap_create_successor()`: an anonymous bitmap takes over recording
    /// while an operation works on this one, which becomes busy and disabled.
    pub(crate) fn create_successor(&self) -> Result<()> {
        self.check(BDRV_BITMAP_BUSY)?;
        if self.has_successor() {
            return Err(Error::generic(
                "Cannot create a successor for a bitmap that already has one",
            ));
        }
        let bs = self.node().ok_or_else(|| Error::generic("could not get length of device"))?;
        let child = bs.create_dirty_bitmap(self.granularity(), None)?;
        let mut s = self.state();
        child.state().disabled = s.disabled;
        s.disabled = true;
        s.successor = Some(child);
        s.busy = true;
        Ok(())
    }

    /// `bdrv_dirty_bitmap_enable_successor()`.
    pub(crate) fn enable_successor(&self) {
        let succ = self.state().successor.clone().expect("bitmap has a successor");
        succ.enable();
    }

    /// `bdrv_dirty_bitmap_abdicate()`: the successor takes the name and persistence of this
    /// bitmap, which is released. Returns the successor.
    pub(crate) fn abdicate(&self) -> Result<Arc<DirtyBitmap>> {
        let (succ, me) = {
            let mut s = self.state();
            let Some(succ) = s.successor.take() else {
                return Err(Error::generic(
                    "Cannot relinquish control if there's no successor present",
                ));
            };
            {
                let mut c = succ.state();
                c.name = s.name.take();
                c.persistent = s.persistent;
            }
            s.persistent = false;
            s.busy = false;
            (succ, self.node())
        };
        if let Some(bs) = me {
            bs.io.dirty_bitmaps.lock().retain(|b| !std::ptr::eq(Arc::as_ptr(b), self));
        }
        Ok(succ)
    }

    /// `bdrv_reclaim_dirty_bitmap()`: merges the successor back in after a failed operation.
    /// The bitmap is enabled if the successor was.
    pub(crate) fn reclaim(&self) -> Result<()> {
        let succ = {
            let mut s = self.state();
            let Some(succ) = s.successor.take() else {
                return Err(Error::generic("Cannot reclaim a successor when none is present"));
            };
            let (bits, disabled) = {
                let c = succ.state();
                (c.bitmap.clone(), c.disabled)
            };
            s.bitmap.merge(&bits);
            s.disabled = disabled;
            s.busy = false;
            succ
        };
        if let Some(bs) = self.node() {
            bs.release_dirty_bitmap(&succ);
        }
        Ok(())
    }

    /// `bdrv_disable_dirty_bitmap()`.
    pub(crate) fn disable(&self) {
        self.state().disabled = true;
    }

    /// `bdrv_enable_dirty_bitmap()`.
    pub(crate) fn enable(&self) {
        self.state().disabled = false;
    }

    /// `bdrv_dirty_bitmap_get()`: whether the byte at `offset` is dirty.
    pub(crate) fn get(&self, offset: u64) -> bool {
        self.state().bitmap.get(offset)
    }

    /// `bdrv_set_dirty_bitmap()`.
    pub(crate) fn set_range(&self, offset: u64, bytes: u64) {
        let mut s = self.state();
        assert!(!s.readonly);
        s.bitmap.set(offset, bytes);
    }

    /// `bdrv_reset_dirty_bitmap()`.
    pub(crate) fn reset_range(&self, offset: u64, bytes: u64) {
        let mut s = self.state();
        assert!(!s.readonly);
        s.bitmap.reset(offset, bytes);
    }

    /// `bdrv_clear_dirty_bitmap()`: clears every bit. With `backup` the old bits are
    /// returned for [`DirtyBitmap::restore`].
    pub(crate) fn clear(&self, backup: bool) -> Option<HBitmap> {
        let mut s = self.state();
        assert!(!s.readonly);
        if backup {
            let fresh = HBitmap::new(s.size, s.bitmap.granularity());
            Some(std::mem::replace(&mut s.bitmap, fresh))
        } else {
            s.bitmap.reset_all();
            None
        }
    }

    /// `bdrv_restore_dirty_bitmap()`.
    pub(crate) fn restore(&self, backup: HBitmap) {
        let mut s = self.state();
        assert!(!s.readonly);
        s.bitmap = backup;
    }

    /// A copy of the bits.
    pub(crate) fn copy_bits(&self) -> HBitmap {
        self.state().bitmap.clone()
    }

    /// `bdrv_get_dirty_count()`: the dirty bytes.
    pub(crate) fn count(&self) -> u64 {
        self.state().bitmap.count()
    }

    /// `bdrv_dirty_bitmap_readonly()`.
    pub(crate) fn readonly(&self) -> bool {
        self.state().readonly
    }

    /// `bdrv_dirty_bitmap_set_readonly()`.
    #[allow(dead_code, reason = "for the qcow2 bitmap extension")]
    pub(crate) fn set_readonly(&self, value: bool) {
        self.state().readonly = value;
    }

    /// `bdrv_dirty_bitmap_set_persistence()`.
    pub(crate) fn set_persistence(&self, persistent: bool) {
        self.state().persistent = persistent;
    }

    /// `bdrv_dirty_bitmap_get_persistence()`: whether the bitmap is to be stored.
    pub(crate) fn get_persistence(&self) -> bool {
        let s = self.state();
        s.persistent && !s.skip_store
    }

    /// `bdrv_dirty_bitmap_set_inconsistent()`: a persistent bitmap that was not stored
    /// properly; only removing it is allowed.
    #[allow(dead_code, reason = "for the qcow2 bitmap extension")]
    pub(crate) fn set_inconsistent(&self) {
        let mut s = self.state();
        assert!(s.persistent);
        s.inconsistent = true;
        s.disabled = true;
    }

    /// `bdrv_dirty_bitmap_inconsistent()`.
    pub(crate) fn inconsistent(&self) -> bool {
        self.state().inconsistent
    }

    /// `bdrv_dirty_bitmap_skip_store()`.
    #[allow(dead_code, reason = "for migration")]
    pub(crate) fn skip_store(&self, skip: bool) {
        self.state().skip_store = skip;
    }

    /// `bdrv_dirty_bitmap_sha256()`.
    #[allow(dead_code, reason = "for the qtest command x-debug-block-dirty-bitmap-sha256")]
    pub(crate) fn sha256(&self) -> Result<String> {
        self.state().bitmap.sha256()
    }

    /// `bdrv_dirty_bitmap_next_dirty()`.
    pub(crate) fn next_dirty(&self, offset: u64, bytes: u64) -> Option<u64> {
        self.state().bitmap.next_dirty(offset, bytes)
    }

    /// `bdrv_dirty_bitmap_next_zero()`.
    pub(crate) fn next_zero(&self, offset: u64, bytes: u64) -> Option<u64> {
        self.state().bitmap.next_zero(offset, bytes)
    }

    /// `bdrv_dirty_bitmap_next_dirty_area()`.
    pub(crate) fn next_dirty_area(&self, start: u64, end: u64, max: u64) -> Option<(u64, u64)> {
        self.state().bitmap.next_dirty_area(start, end, max)
    }

    /// `bdrv_dirty_bitmap_status()`.
    pub(crate) fn status(&self, offset: u64, bytes: u64) -> (bool, u64) {
        self.state().bitmap.status(offset, bytes)
    }

    /// `bdrv_dirty_bitmap_serialization_size()`.
    pub(crate) fn serialization_size(&self, offset: u64, bytes: u64) -> u64 {
        self.state().bitmap.serialization_size(offset, bytes)
    }

    /// `bdrv_dirty_bitmap_serialization_align()`.
    pub(crate) fn serialization_align(&self) -> u64 {
        self.state().bitmap.serialization_align()
    }

    /// `bdrv_dirty_bitmap_serialization_coverage()`: the bytes of disk one chunk of
    /// `chunk_size` serialised bytes covers.
    #[allow(dead_code, reason = "for the qcow2 bitmap extension")]
    pub(crate) fn serialization_coverage(&self, chunk_size: u64) -> u64 {
        let limit = u64::from(self.granularity()) * (chunk_size << 3);
        assert!(limit % self.serialization_align() == 0);
        limit
    }

    /// `bdrv_dirty_bitmap_serialize_part()`.
    pub(crate) fn serialize_part(&self, buf: &mut [u8], offset: u64, bytes: u64) {
        self.state().bitmap.serialize_part(buf, offset, bytes);
    }

    /// `bdrv_dirty_bitmap_deserialize_part()`.
    pub(crate) fn deserialize_part(&self, buf: &[u8], offset: u64, bytes: u64, finish: bool) {
        self.state().bitmap.deserialize_part(buf, offset, bytes, finish);
    }

    /// `bdrv_dirty_bitmap_deserialize_zeroes()`.
    #[allow(dead_code, reason = "for the qcow2 bitmap extension")]
    pub(crate) fn deserialize_zeroes(&self, offset: u64, bytes: u64, finish: bool) {
        self.state().bitmap.deserialize_zeroes(offset, bytes, finish);
    }

    /// `bdrv_dirty_bitmap_deserialize_ones()`.
    #[allow(dead_code, reason = "for the qcow2 bitmap extension")]
    pub(crate) fn deserialize_ones(&self, offset: u64, bytes: u64, finish: bool) {
        self.state().bitmap.deserialize_ones(offset, bytes, finish);
    }

    /// `bdrv_dirty_bitmap_deserialize_finish()`.
    pub(crate) fn deserialize_finish(&self) {
        self.state().bitmap.deserialize_finish();
    }

    /// `bdrv_merge_dirty_bitmap()`: `self |= src` after the checks QMP needs. With `backup`
    /// the bits from before are returned. On failure nothing changes.
    pub(crate) fn merge(&self, src: &DirtyBitmap, backup: bool) -> Result<Option<HBitmap>> {
        self.check(BDRV_BITMAP_DEFAULT)?;
        src.check(BDRV_BITMAP_ALLOW_RO)?;
        let (src_size, dst_size) = (src.size(), self.size());
        if src_size != dst_size {
            return Err(Error::generic(format!(
                "Bitmaps are of different sizes (destination size is {dst_size}, source size \
                 is {src_size}) and can't be merged"
            )));
        }
        Ok(self.merge_internal(src, backup))
    }

    /// `bdrv_dirty_bitmap_merge_internal()`: `self |= src` without the checks.
    pub(crate) fn merge_internal(&self, src: &DirtyBitmap, backup: bool) -> Option<HBitmap> {
        let bits = if std::ptr::eq(self, src) { None } else { Some(src.copy_bits()) };
        let mut s = self.state();
        assert!(!s.readonly && !s.inconsistent);
        let old = backup.then(|| s.bitmap.clone());
        if let Some(b) = bits {
            s.bitmap.merge(&b);
        }
        old
    }

    /// `bdrv_dirty_iter_new()`.
    pub(crate) fn iter(&self) -> DirtyBitmapIter {
        let s = self.state();
        DirtyBitmapIter { it: HBitmapIter::new(&s.bitmap, 0) }
    }

    /// `bdrv_dirty_iter_next()`.
    pub(crate) fn iter_next(&self, it: &mut DirtyBitmapIter) -> Option<u64> {
        it.it.next(&self.state().bitmap)
    }

    /// `bdrv_set_dirty_iter()`: moves the iterator to `offset`.
    pub(crate) fn set_iter(&self, it: &mut DirtyBitmapIter, offset: u64) {
        it.it = HBitmapIter::new(&self.state().bitmap, offset);
    }

    /// The name for messages, `(null)` for an anonymous bitmap as glib prints it.
    pub(crate) fn display_name(&self) -> String {
        self.name_str()
    }
}

/// `BdrvDirtyBitmapIter`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DirtyBitmapIter {
    it: HBitmapIter,
}
