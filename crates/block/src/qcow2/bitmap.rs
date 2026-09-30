// SPDX-License-Identifier: GPL-2.0-or-later

//! Persistent dirty bitmaps, from block/qcow2-bitmap.c.
//!
//! QEMU keeps the live bitmaps in the generic dirty bitmap layer and qcow2 only loads and
//! stores them. Here the live bitmaps are the node's [`DirtyBitmap`](crate::bitmap::DirtyBitmap)
//! values too. This module works on [`Bitmap`] copies of them: loading fills
//! `State::bitmaps`, which the driver in `mod.rs` hands over to the node, and before storing,
//! resizing or checking the driver copies the node's persistent bitmaps back in and empties
//! the list again afterwards. The on-disk handling (directory, tables, `IN_USE` and `AUTO`
//! flags, the autoclear bit) follows QEMU exactly.

use std::io;

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{Qcow2BitmapInfo, Qcow2BitmapInfoFlags};

use super::cache::set_be64;
use super::check::{CheckResult, Imrt};
use super::header::{be16, be32, be64};
use super::state::*;
use crate::node::errno;

const BME_MAX_TABLE_SIZE: u64 = 0x800_0000;
const BME_MAX_PHYS_SIZE: u64 = 0x2000_0000;
const BME_MAX_GRANULARITY_BITS: u32 = 31;
const BME_MIN_GRANULARITY_BITS: u32 = 9;
const BME_MAX_NAME_SIZE: usize = 1023;
const BME_TABLE_ENTRY_SIZE: u64 = 8;
const BME_RESERVED_FLAGS: u32 = 0xffff_fffc;
pub(crate) const BME_FLAG_IN_USE: u32 = 1 << 0;
pub(crate) const BME_FLAG_AUTO: u32 = 1 << 1;
const BME_TABLE_ENTRY_RESERVED_MASK: u64 = 0xff00_0000_0000_01fe;
const BME_TABLE_ENTRY_OFFSET_MASK: u64 = 0x00ff_ffff_ffff_fe00;
const BME_TABLE_ENTRY_FLAG_ALL_ONES: u64 = 1;
const BT_DIRTY_TRACKING_BITMAP: u8 = 1;
const DIR_ENTRY_SIZE: usize = 24;

/// A live persistent dirty bitmap, the parts of `BdrvDirtyBitmap` qcow2 needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Bitmap {
    pub name: String,
    pub granularity: u32,
    /// Bytes of the disk the bitmap covers.
    pub size: u64,
    /// `AUTO`: writes are recorded.
    pub enabled: bool,
    /// Loaded with `IN_USE` set: the contents cannot be trusted and it is not stored again.
    pub inconsistent: bool,
    /// Loaded from an image that cannot be written.
    pub readonly: bool,
    /// One bit per `granularity` bytes, least significant bit first, the serialized form.
    pub bits: Vec<u8>,
}

impl Bitmap {
    pub(crate) fn new(name: &str, granularity: u32, size: u64) -> Bitmap {
        let nbits = size.div_ceil(granularity as u64);
        Bitmap {
            name: name.to_string(),
            granularity,
            size,
            enabled: true,
            inconsistent: false,
            readonly: false,
            bits: vec![0; serialization_size(nbits) as usize],
        }
    }

    fn nbits(&self) -> u64 {
        self.size.div_ceil(self.granularity as u64)
    }

    fn set_bits(&mut self, first: u64, last: u64, value: bool) {
        for i in first..=last {
            let b = &mut self.bits[(i / 8) as usize];
            if value {
                *b |= 1 << (i % 8);
            } else {
                *b &= !(1 << (i % 8));
            }
        }
    }

    /// `bdrv_dirty_bitmap_truncate()`: bits past the old end start clean.
    pub(crate) fn truncate(&mut self, size: u64) {
        let old_bits = self.nbits();
        self.size = size;
        let nbits = self.nbits();
        self.bits.resize(serialization_size(nbits) as usize, 0);
        if nbits > old_bits {
            self.set_bits(old_bits, nbits - 1, false);
        }
        for i in nbits..self.bits.len() as u64 * 8 {
            self.bits[(i / 8) as usize] &= !(1 << (i % 8));
        }
    }
}

/// `hbitmap_serialization_size()` for the whole bitmap: whole 64 bit words.
fn serialization_size(nbits: u64) -> u64 {
    nbits.div_ceil(64) * 8
}

/// `get_bitmap_bytes_needed()`.
fn bitmap_bytes_needed(len: u64, granularity: u32) -> u64 {
    len.div_ceil(granularity as u64).div_ceil(8)
}

/// `calc_dir_entry_size()`.
fn calc_dir_entry_size(name_size: usize, extra_data_size: usize) -> u64 {
    ((DIR_ENTRY_SIZE + name_size + extra_data_size) as u64).next_multiple_of(8)
}

/// `check_table_entry()`.
fn check_table_entry(entry: u64, cluster_size: u64) -> bool {
    if entry & BME_TABLE_ENTRY_RESERVED_MASK != 0 {
        return false;
    }
    let offset = entry & BME_TABLE_ENTRY_OFFSET_MASK;
    if offset != 0 && (entry & BME_TABLE_ENTRY_FLAG_ALL_ONES != 0 || offset % cluster_size != 0) {
        return false;
    }
    true
}

/// `Qcow2Bitmap`, one entry of the directory.
#[derive(Clone, Debug, Default)]
struct DirBitmap {
    table_offset: u64,
    table_size: u32,
    flags: u32,
    granularity_bits: u8,
    name: String,
    /// Index into `State::bitmaps` of the live bitmap to store.
    live: Option<usize>,
}

/// The raw fields of a directory entry.
struct DirEntry {
    table_offset: u64,
    table_size: u32,
    flags: u32,
    ty: u8,
    granularity_bits: u8,
    name_size: u16,
}

/// `qcow2_get_persistent_dirty_bitmap_size()` for bitmaps of the given names, granularities and
/// disk sizes, as if copied into an image with `cluster_size`.
pub(crate) fn persistent_bitmaps_size(bitmaps: &[(String, u32, u64)], cluster_size: u64) -> u64 {
    let mut size = 0;
    let mut dir = 0;
    for (name, granularity, len) in bitmaps {
        let clusters = bitmap_bytes_needed(*len, *granularity).div_ceil(cluster_size);
        size += clusters * cluster_size;
        size += (clusters * BME_TABLE_ENTRY_SIZE).next_multiple_of(cluster_size);
        dir += calc_dir_entry_size(name.len(), 0);
    }
    size + dir.next_multiple_of(cluster_size)
}

impl State {
    /// `can_write()`.
    fn bitmap_can_write(&self) -> bool {
        self.writable()
    }

    /// `update_header_sync()`.
    fn update_header_sync(&mut self) -> io::Result<()> {
        self.update_header()?;
        self.file.flush()
    }

    /// `check_dir_entry()`.
    fn check_dir_entry(&self, e: &DirEntry) -> bool {
        let fail = e.table_size == 0
            || e.table_offset == 0
            || e.table_offset % self.cluster_size != 0
            || e.table_size as u64 > BME_MAX_TABLE_SIZE
            || e.granularity_bits as u32 > BME_MAX_GRANULARITY_BITS
            || (e.granularity_bits as u32) < BME_MIN_GRANULARITY_BITS
            || e.flags & BME_RESERVED_FLAGS != 0
            || e.name_size as usize > BME_MAX_NAME_SIZE
            || e.ty != BT_DIRTY_TRACKING_BITMAP;
        if fail {
            return false;
        }
        let phys = e.table_size as u64 * self.cluster_size;
        if phys > BME_MAX_PHYS_SIZE {
            return false;
        }
        // A valid bitmap must be big enough for the disk; an IN_USE one may be stale.
        let len = self.disk_size_sectors();
        if e.flags & BME_FLAG_IN_USE == 0
            && (len as u128) > ((phys as u128 * 8) << e.granularity_bits)
        {
            return false;
        }
        true
    }

    /// `check_constraints_on_bitmap()`.
    fn check_constraints_on_bitmap(&self, name: &str, granularity: u32) -> Result<()> {
        assert!(granularity.is_power_of_two());
        let bits = granularity.trailing_zeros();
        if bits > BME_MAX_GRANULARITY_BITS {
            return Err(Error::generic(format!(
                "Granularity exceeds maximum ({} bytes)",
                1u64 << BME_MAX_GRANULARITY_BITS
            )));
        }
        if bits < BME_MIN_GRANULARITY_BITS {
            return Err(Error::generic(format!(
                "Granularity is under minimum ({} bytes)",
                1u64 << BME_MIN_GRANULARITY_BITS
            )));
        }
        let bytes = bitmap_bytes_needed(self.disk_size_sectors(), granularity);
        if bytes > BME_MAX_PHYS_SIZE || bytes > BME_MAX_TABLE_SIZE * self.cluster_size {
            return Err(Error::generic(
                "Too much space will be occupied by the bitmap. Use larger granularity",
            ));
        }
        if name.len() > BME_MAX_NAME_SIZE {
            return Err(Error::generic(format!(
                "Name length exceeds maximum ({BME_MAX_NAME_SIZE} characters)"
            )));
        }
        Ok(())
    }

    /// `bitmap_table_load()`.
    fn bitmap_table_load(&self, offset: u64, size: u32) -> io::Result<Vec<u64>> {
        assert!(size != 0 && size as u64 <= BME_MAX_TABLE_SIZE);
        let mut buf = vec![0u8; size as usize * 8];
        self.file.pread(offset, &mut buf)?;
        let t: Vec<u64> =
            buf.chunks_exact(8).map(|c| u64::from_be_bytes(c.try_into().unwrap())).collect();
        if t.iter().any(|&e| !check_table_entry(e, self.cluster_size)) {
            return Err(errno(libc::EINVAL));
        }
        Ok(t)
    }

    /// `clear_bitmap_table()`.
    fn clear_bitmap_table(&mut self, table: &mut [u64]) {
        for e in table.iter_mut() {
            let addr = *e & BME_TABLE_ENTRY_OFFSET_MASK;
            if addr != 0 {
                self.free_clusters(addr, self.cluster_size, DiscardType::Always);
                *e = 0;
            }
        }
    }

    /// `free_bitmap_clusters()`.
    fn free_bitmap_clusters(&mut self, offset: u64, size: u32) -> io::Result<()> {
        let mut t = self.bitmap_table_load(offset, size)?;
        self.clear_bitmap_table(&mut t);
        self.free_clusters(offset, size as u64 * BME_TABLE_ENTRY_SIZE, DiscardType::Other);
        Ok(())
    }

    /// `bitmap_list_load()`.
    fn bitmap_list_load(&self, offset: u64, size: u64) -> Result<Vec<DirBitmap>> {
        if size == 0 {
            return Err(Error::generic("Requested bitmap directory size is zero"));
        }
        if size > QCOW2_MAX_BITMAP_DIRECTORY_SIZE {
            return Err(Error::generic("Requested bitmap directory size is too big"));
        }
        let mut dir = vec![0u8; size as usize];
        self.file
            .pread(offset, &mut dir)
            .map_err(|e| Error::from_io("Failed to read bitmap directory", e))?;
        let broken = || Error::generic("Broken bitmap directory");

        let mut list = Vec::new();
        let mut pos = 0usize;
        let end = dir.len();
        while pos < end {
            if pos + DIR_ENTRY_SIZE > end {
                return Err(broken());
            }
            if list.len() as u32 + 1 > self.nb_bitmaps {
                return Err(Error::generic(
                    "More bitmaps found than specified in header extension",
                ));
            }
            let d = &dir[pos..];
            let e = DirEntry {
                table_offset: be64(d, 0),
                table_size: be32(d, 8),
                flags: be32(d, 12),
                ty: d[16],
                granularity_bits: d[17],
                name_size: be16(d, 18),
            };
            let extra = be32(d, 20);
            let esize = calc_dir_entry_size(e.name_size as usize, extra as usize) as usize;
            if pos + esize > end {
                return Err(broken());
            }
            if extra != 0 {
                return Err(Error::generic("Bitmap extra data is not supported"));
            }
            let name_bytes = &d[DIR_ENTRY_SIZE..DIR_ENTRY_SIZE + e.name_size as usize];
            let name = String::from_utf8_lossy(name_bytes).into_owned();
            if !self.check_dir_entry(&e) {
                return Err(Error::generic(format!(
                    "Bitmap '{name}' doesn't satisfy the constraints"
                )));
            }
            list.push(DirBitmap {
                table_offset: e.table_offset,
                table_size: e.table_size,
                flags: e.flags,
                granularity_bits: e.granularity_bits,
                name,
                live: None,
            });
            pos += esize;
        }
        if list.len() as u32 != self.nb_bitmaps {
            return Err(Error::generic("Less bitmaps found than specified in header extension"));
        }
        if pos != end {
            return Err(broken());
        }
        Ok(list)
    }

    /// `qcow2_check_bitmaps_refcounts()`.
    pub(crate) fn check_bitmaps_refcounts(
        &mut self,
        res: &mut CheckResult,
        imrt: &mut Imrt,
    ) -> io::Result<()> {
        if self.nb_bitmaps == 0 {
            return Ok(());
        }
        self.inc_refcounts_imrt(
            res,
            imrt,
            self.bitmap_directory_offset,
            self.bitmap_directory_size,
        )?;
        let Ok(list) =
            self.bitmap_list_load(self.bitmap_directory_offset, self.bitmap_directory_size)
        else {
            res.corruptions += 1;
            return Err(errno(libc::EINVAL));
        };
        for bm in &list {
            self.inc_refcounts_imrt(
                res,
                imrt,
                bm.table_offset,
                bm.table_size as u64 * BME_TABLE_ENTRY_SIZE,
            )?;
            let t = match self.bitmap_table_load(bm.table_offset, bm.table_size) {
                Ok(t) => t,
                Err(e) => {
                    res.corruptions += 1;
                    return Err(e);
                }
            };
            for entry in t {
                let offset = entry & BME_TABLE_ENTRY_OFFSET_MASK;
                if !check_table_entry(entry, self.cluster_size) {
                    res.corruptions += 1;
                    continue;
                }
                if offset != 0 {
                    self.inc_refcounts_imrt(res, imrt, offset, self.cluster_size)?;
                }
            }
        }
        Ok(())
    }

    /// `bitmap_list_store()`.
    fn bitmap_list_store(
        &mut self,
        list: &[DirBitmap],
        offset: &mut u64,
        size: &mut u64,
        in_place: bool,
    ) -> io::Result<()> {
        let dir_size: u64 = list.iter().map(|b| calc_dir_entry_size(b.name.len(), 0)).sum();
        if dir_size == 0 || dir_size > QCOW2_MAX_BITMAP_DIRECTORY_SIZE {
            return Err(errno(libc::EINVAL));
        }
        if in_place && (*size != dir_size || *offset == 0) {
            return Err(errno(libc::EINVAL));
        }
        let mut dir = vec![0u8; dir_size as usize];
        let mut pos = 0usize;
        for bm in list {
            let e = DirEntry {
                table_offset: bm.table_offset,
                table_size: bm.table_size,
                flags: bm.flags,
                ty: BT_DIRTY_TRACKING_BITMAP,
                granularity_bits: bm.granularity_bits,
                name_size: bm.name.len() as u16,
            };
            if !self.check_dir_entry(&e) {
                return Err(errno(libc::EINVAL));
            }
            let d = &mut dir[pos..];
            d[0..8].copy_from_slice(&e.table_offset.to_be_bytes());
            d[8..12].copy_from_slice(&e.table_size.to_be_bytes());
            d[12..16].copy_from_slice(&e.flags.to_be_bytes());
            d[16] = e.ty;
            d[17] = e.granularity_bits;
            d[18..20].copy_from_slice(&e.name_size.to_be_bytes());
            d[DIR_ENTRY_SIZE..DIR_ENTRY_SIZE + bm.name.len()].copy_from_slice(bm.name.as_bytes());
            pos += calc_dir_entry_size(bm.name.len(), 0) as usize;
        }

        let dir_offset = if in_place { *offset } else { self.alloc_clusters(dir_size)? };
        // Updating in place is safe without the directory check, because the autoclear bit is
        // off while the directory is rewritten.
        let r = self
            .pre_write_overlap_check(
                if in_place { OL_BITMAP_DIRECTORY } else { 0 },
                dir_offset,
                dir_size,
                false,
            )
            .and_then(|()| self.file.pwrite(dir_offset, &dir));
        if let Err(e) = r {
            if !in_place && dir_offset > 0 {
                self.free_clusters(dir_offset, dir_size, DiscardType::Other);
            }
            return Err(e);
        }
        if !in_place {
            *size = dir_size;
            *offset = dir_offset;
        }
        Ok(())
    }

    /// `update_ext_header_and_dir_in_place()`.
    fn update_ext_header_and_dir_in_place(&mut self, list: &[DirBitmap]) -> io::Result<()> {
        if self.autoclear_features & QCOW2_AUTOCLEAR_BITMAPS == 0
            || list.is_empty()
            || list.len() as u32 != self.nb_bitmaps
        {
            return Err(errno(libc::EINVAL));
        }
        self.autoclear_features &= !QCOW2_AUTOCLEAR_BITMAPS;
        self.update_header_sync()?;
        // With the autoclear bit off the directory can be rewritten safely; leaks from a
        // failure are for qemu-img check.
        let (mut off, mut size) = (self.bitmap_directory_offset, self.bitmap_directory_size);
        self.bitmap_list_store(list, &mut off, &mut size, true)?;
        self.update_header_sync()?;
        self.autoclear_features |= QCOW2_AUTOCLEAR_BITMAPS;
        self.update_header_sync()
    }

    /// `update_ext_header_and_dir()`.
    fn update_ext_header_and_dir(&mut self, list: &[DirBitmap]) -> io::Result<()> {
        let old = (
            self.bitmap_directory_offset,
            self.bitmap_directory_size,
            self.nb_bitmaps,
            self.autoclear_features,
        );
        let (mut new_offset, mut new_size, mut new_nb) = (0u64, 0u64, 0u32);
        if !list.is_empty() {
            new_nb = list.len() as u32;
            if new_nb > QCOW2_MAX_BITMAPS {
                return Err(errno(libc::EINVAL));
            }
            self.bitmap_list_store(list, &mut new_offset, &mut new_size, false)?;
            if let Err(e) = self.flush_caches() {
                self.free_clusters(new_offset, new_size, DiscardType::Other);
                return Err(e);
            }
            self.autoclear_features |= QCOW2_AUTOCLEAR_BITMAPS;
        } else {
            self.autoclear_features &= !QCOW2_AUTOCLEAR_BITMAPS;
        }
        self.bitmap_directory_offset = new_offset;
        self.bitmap_directory_size = new_size;
        self.nb_bitmaps = new_nb;
        if let Err(e) = self.update_header_sync() {
            if new_offset > 0 {
                self.free_clusters(new_offset, new_size, DiscardType::Other);
            }
            self.bitmap_directory_offset = old.0;
            self.bitmap_directory_size = old.1;
            self.nb_bitmaps = old.2;
            self.autoclear_features = old.3;
            return Err(e);
        }
        if old.1 > 0 {
            self.free_clusters(old.0, old.1, DiscardType::Other);
        }
        Ok(())
    }

    /// `load_bitmap()`.
    fn load_bitmap(&self, bm: &DirBitmap) -> Result<Bitmap> {
        let granularity = 1u32 << bm.granularity_bits;
        let mut bitmap = Bitmap::new(&bm.name, granularity, self.disk_size_sectors());
        if bm.flags & BME_FLAG_IN_USE != 0 {
            // The data cannot be used, do not load it.
            return Ok(bitmap);
        }
        let table = self.bitmap_table_load(bm.table_offset, bm.table_size).map_err(|e| {
            Error::from_io(
                format_args!(
                    "Could not read bitmap_table table from image for bitmap '{}'",
                    bm.name
                ),
                e,
            )
        })?;
        self.load_bitmap_data(&table, &mut bitmap).map_err(|e| {
            Error::from_io(format_args!("Could not read bitmap '{}' from image", bm.name), e)
        })?;
        Ok(bitmap)
    }

    /// `load_bitmap_data()`.
    fn load_bitmap_data(&self, table: &[u64], bitmap: &mut Bitmap) -> io::Result<()> {
        let total = bitmap.bits.len() as u64;
        let tab_size = self.size_to_clusters(total);
        if tab_size != table.len() as u64 || tab_size > BME_MAX_TABLE_SIZE {
            return Err(errno(libc::EINVAL));
        }
        let cs = self.cluster_size;
        let mut buf = vec![0u8; cs as usize];
        for (i, &entry) in table.iter().enumerate() {
            let start = i as u64 * cs;
            let count = (total - start).min(cs) as usize;
            let dst = &mut bitmap.bits[start as usize..start as usize + count];
            let data_offset = entry & BME_TABLE_ENTRY_OFFSET_MASK;
            if data_offset == 0 {
                if entry & BME_TABLE_ENTRY_FLAG_ALL_ONES != 0 {
                    dst.fill(0xff);
                }
            } else {
                self.file.pread(data_offset, &mut buf)?;
                dst.copy_from_slice(&buf[..count]);
            }
        }
        // Bits past the end of the disk are not part of the bitmap.
        bitmap.truncate(bitmap.size);
        Ok(())
    }

    /// `qcow2_load_dirty_bitmaps()`. Returns whether the header was updated.
    pub(crate) fn load_dirty_bitmaps(&mut self) -> Result<bool> {
        if self.nb_bitmaps == 0 {
            return Ok(false);
        }
        let mut list =
            self.bitmap_list_load(self.bitmap_directory_offset, self.bitmap_directory_size)?;
        let mut created = Vec::new();
        let mut needs_update = false;
        for bm in list.iter_mut() {
            if bm.flags & BME_FLAG_IN_USE != 0 && self.bitmaps.iter().any(|b| b.name == bm.name) {
                // Already live, for example migrated over shared storage.
                continue;
            }
            let mut bitmap = self.load_bitmap(bm)?;
            if bm.flags & BME_FLAG_IN_USE != 0 {
                bitmap.inconsistent = true;
            } else {
                // Only written back if the image can be written.
                bm.flags |= BME_FLAG_IN_USE;
                needs_update = true;
            }
            bitmap.enabled = bm.flags & BME_FLAG_AUTO != 0;
            created.push(bitmap);
        }
        let mut header_updated = false;
        if needs_update && self.bitmap_can_write() {
            self.update_ext_header_and_dir_in_place(&list)
                .map_err(|e| Error::from_io("Can't update bitmap directory", e))?;
            header_updated = true;
        }
        if !self.bitmap_can_write() {
            for b in &mut created {
                b.readonly = true;
            }
        }
        self.bitmaps.extend(created);
        Ok(header_updated)
    }

    /// `qcow2_get_bitmap_info_list()`.
    pub(crate) fn bitmap_info_list(&self) -> Result<Vec<Qcow2BitmapInfo>> {
        if self.nb_bitmaps == 0 {
            return Ok(Vec::new());
        }
        let list =
            self.bitmap_list_load(self.bitmap_directory_offset, self.bitmap_directory_size)?;
        Ok(list
            .into_iter()
            .map(|bm| {
                let mut flags = Vec::new();
                if bm.flags & BME_FLAG_IN_USE != 0 {
                    flags.push(Qcow2BitmapInfoFlags::InUse);
                }
                if bm.flags & BME_FLAG_AUTO != 0 {
                    flags.push(Qcow2BitmapInfoFlags::Auto);
                }
                Qcow2BitmapInfo { name: bm.name, granularity: 1 << bm.granularity_bits, flags }
            })
            .collect())
    }

    /// `qcow2_truncate_bitmaps_check()`.
    pub(crate) fn truncate_bitmaps_check(&self) -> Result<()> {
        if self.nb_bitmaps == 0 {
            return Ok(());
        }
        let list =
            self.bitmap_list_load(self.bitmap_directory_offset, self.bitmap_directory_size)?;
        for bm in &list {
            let Some(b) = self.bitmaps.iter().find(|b| b.name == bm.name) else {
                return Err(Error::generic(
                    "Cannot resize qcow2 with persistent bitmaps that were not loaded into memory",
                ));
            };
            // bdrv_dirty_bitmap_check() with BDRV_BITMAP_DEFAULT.
            if b.readonly {
                return Err(Error::generic(format!(
                    "Bitmap '{}' is readonly and cannot be modified",
                    b.name
                )));
            }
            if b.inconsistent {
                return Err(Error::generic(format!(
                    "Bitmap '{}' is inconsistent and cannot be used",
                    b.name
                ))
                .hint("Try block-dirty-bitmap-remove to delete this bitmap from disk\n"));
            }
        }
        Ok(())
    }

    /// `store_bitmap_data()`.
    fn store_bitmap_data(&mut self, idx: usize) -> Result<Vec<u64>> {
        let name = self.bitmaps[idx].name.clone();
        let total = self.bitmaps[idx].bits.len() as u64;
        let cs = self.cluster_size;
        let tb_size = self.size_to_clusters(total);
        if tb_size > BME_MAX_TABLE_SIZE || tb_size * cs > BME_MAX_PHYS_SIZE {
            return Err(Error::generic(format!("Bitmap '{name}' is too big")));
        }
        let mut tb = vec![0u64; tb_size as usize];
        let mut buf = vec![0u8; cs as usize];
        for i in 0..tb_size as usize {
            let start = i * cs as usize;
            let end = (start + cs as usize).min(total as usize);
            if self.bitmaps[idx].bits[start..end].iter().all(|&b| b == 0) {
                continue;
            }
            let r = (|| {
                let off = self.alloc_clusters(cs).map_err(|e| {
                    Error::from_io(
                        format_args!("Failed to allocate clusters for bitmap '{name}'"),
                        e,
                    )
                })?;
                tb[i] = off;
                buf.fill(0);
                buf[..end - start].copy_from_slice(&self.bitmaps[idx].bits[start..end]);
                self.pre_write_overlap_check(0, off, cs, false)
                    .map_err(|e| Error::from_io("Qcow2 overlap check failed", e))?;
                self.file.pwrite(off, &buf).map_err(|e| {
                    Error::from_io(format_args!("Failed to write bitmap '{name}' to file"), e)
                })
            })();
            if let Err(e) = r {
                self.clear_bitmap_table(&mut tb);
                return Err(e);
            }
        }
        Ok(tb)
    }

    /// `store_bitmap()`.
    fn store_bitmap(&mut self, bm: &mut DirBitmap) -> Result<()> {
        let idx = bm.live.unwrap();
        let name = self.bitmaps[idx].name.clone();
        let mut tb = self.store_bitmap_data(idx)?;
        let bytes = tb.len() as u64 * BME_TABLE_ENTRY_SIZE;
        let mut tb_offset = 0;
        let r = (|| {
            tb_offset = self.alloc_clusters(bytes).map_err(|e| {
                Error::from_io(format_args!("Failed to allocate clusters for bitmap '{name}'"), e)
            })?;
            self.pre_write_overlap_check(0, tb_offset, bytes, false)
                .map_err(|e| Error::from_io("Qcow2 overlap check failed", e))?;
            let mut buf = vec![0u8; bytes as usize];
            for (i, v) in tb.iter().enumerate() {
                set_be64(&mut buf, i, *v);
            }
            self.file.pwrite(tb_offset, &buf).map_err(|e| {
                Error::from_io(format_args!("Failed to write bitmap '{name}' to file"), e)
            })
        })();
        if let Err(e) = r {
            self.clear_bitmap_table(&mut tb);
            if tb_offset > 0 {
                self.free_clusters(tb_offset, bytes, DiscardType::Other);
            }
            return Err(e);
        }
        bm.table_offset = tb_offset;
        bm.table_size = tb.len() as u32;
        Ok(())
    }

    /// `qcow2_co_remove_persistent_dirty_bitmap()`, and the bitmap is dropped from memory too.
    pub(crate) fn remove_persistent_dirty_bitmap(&mut self, name: &str) -> Result<()> {
        self.bitmaps.retain(|b| b.name != name);
        if self.nb_bitmaps == 0 {
            // A missing bitmap is not an error.
            return Ok(());
        }
        let mut list =
            self.bitmap_list_load(self.bitmap_directory_offset, self.bitmap_directory_size)?;
        let Some(i) = list.iter().position(|b| b.name == name) else {
            return Ok(());
        };
        let bm = list.remove(i);
        self.update_ext_header_and_dir(&list)
            .map_err(|e| Error::from_io("Failed to update bitmap extension", e))?;
        let _ = self.free_bitmap_clusters(bm.table_offset, bm.table_size);
        Ok(())
    }

    /// `qcow2_store_persistent_dirty_bitmaps()`. With `release`, the live bitmaps are dropped
    /// afterwards, as on close.
    pub(crate) fn store_persistent_dirty_bitmaps(&mut self, release: bool) -> Result<()> {
        let mut list = if self.nb_bitmaps == 0 {
            Vec::new()
        } else {
            self.bitmap_list_load(self.bitmap_directory_offset, self.bitmap_directory_size)?
        };
        let mut new_nb = self.nb_bitmaps;
        let mut new_dir_size = self.bitmap_directory_size;
        let mut drop_tables: Vec<(u64, u32)> = Vec::new();
        let mut need_write = false;

        let r: Result<()> = (|| {
            for idx in 0..self.bitmaps.len() {
                let b = &self.bitmaps[idx];
                if b.inconsistent {
                    continue;
                }
                let name = b.name.clone();
                let granularity = b.granularity;
                let enabled = b.enabled;
                if b.readonly {
                    if let Some(bm) = list.iter_mut().find(|x| x.name == name) {
                        bm.live = Some(idx);
                    }
                    continue;
                }
                need_write = true;
                self.check_constraints_on_bitmap(&name, granularity).map_err(|e| {
                    e.prepend(format!("Bitmap '{name}' doesn't satisfy the constraints: "))
                })?;
                let pos = list.iter().position(|x| x.name == name);
                let bm = match pos {
                    None => {
                        new_nb += 1;
                        if new_nb > QCOW2_MAX_BITMAPS {
                            return Err(Error::generic("Too many persistent bitmaps"));
                        }
                        new_dir_size += calc_dir_entry_size(name.len(), 0);
                        if new_dir_size > QCOW2_MAX_BITMAP_DIRECTORY_SIZE {
                            return Err(Error::generic("Bitmap directory is too large"));
                        }
                        list.push(DirBitmap { name: name.clone(), ..Default::default() });
                        list.last_mut().unwrap()
                    }
                    Some(p) => {
                        let bm = &mut list[p];
                        if bm.flags & BME_FLAG_IN_USE == 0 {
                            return Err(Error::generic(format!(
                                "Bitmap '{name}' already exists in the image"
                            )));
                        }
                        drop_tables.push((bm.table_offset, bm.table_size));
                        bm.table_offset = 0;
                        bm.table_size = 0;
                        bm
                    }
                };
                bm.flags = if enabled { BME_FLAG_AUTO } else { 0 };
                bm.granularity_bits = granularity.trailing_zeros() as u8;
                bm.live = Some(idx);
            }
            if !need_write {
                return Ok(());
            }
            if !self.bitmap_can_write() {
                return Err(Error::generic("No write access"));
            }
            for bm in &mut list {
                let Some(idx) = bm.live else { continue };
                if self.bitmaps[idx].readonly {
                    continue;
                }
                self.store_bitmap(bm)?;
            }
            self.update_ext_header_and_dir(&list)
                .map_err(|e| Error::from_io("Failed to update bitmap extension", e))?;
            // The directory is updated, so the old data can go.
            for (off, size) in drop_tables.drain(..) {
                let _ = self.free_bitmap_clusters(off, size);
            }
            Ok(())
        })();

        if let Err(e) = r {
            for bm in &list {
                let Some(idx) = bm.live else { continue };
                if bm.table_offset == 0 || self.bitmaps[idx].readonly {
                    continue;
                }
                let _ = self.free_bitmap_clusters(bm.table_offset, bm.table_size);
            }
            return Err(e);
        }
        if release {
            self.bitmaps.clear();
        }
        Ok(())
    }

    /// `qcow2_reopen_bitmaps_rw()`: marks the bitmaps of an image that was read-only as in
    /// use again. `find` looks up a live bitmap by name and gives whether it is read-only and
    /// whether it is inconsistent; `file_writable` is whether the protocol file can be written.
    /// Returns the names of the read-only bitmaps, which the caller makes writable.
    pub(crate) fn reopen_bitmaps_rw(
        &mut self,
        filename: &str,
        file_writable: bool,
        find: &dyn Fn(&str) -> Option<(bool, bool)>,
    ) -> Result<Vec<String>> {
        if self.nb_bitmaps == 0 {
            return Ok(Vec::new());
        }
        let mut list =
            self.bitmap_list_load(self.bitmap_directory_offset, self.bitmap_directory_size)?;
        let mut ro = Vec::new();
        let mut need_header_update = false;
        for bm in list.iter_mut() {
            let Some((readonly, inconsistent)) = find(&bm.name) else {
                return Err(Error::generic(format!(
                    "Unexpected bitmap '{}' in image '{filename}'",
                    bm.name
                )));
            };
            if bm.flags & BME_FLAG_IN_USE == 0 {
                if !readonly {
                    return Err(Error::generic(format!(
                        "Corruption: bitmap '{}' is not marked IN_USE in the image '{filename}' \
                         and not marked readonly in RAM",
                        bm.name
                    )));
                }
                if inconsistent {
                    return Err(Error::generic(format!(
                        "Corruption: bitmap '{}' is inconsistent but is not marked IN_USE in \
                         the image '{filename}'",
                        bm.name
                    )));
                }
                bm.flags |= BME_FLAG_IN_USE;
                need_header_update = true;
            } else if readonly && !inconsistent {
                // Loaded read-only and consistent, so someone else set the flag.
                return Err(Error::generic(format!(
                    "Corruption: bitmap '{}' is marked IN_USE in the image '{filename}' but it \
                     is readonly and consistent in RAM",
                    bm.name
                )));
            }
            if readonly {
                ro.push(bm.name.clone());
            }
        }
        if need_header_update {
            if !file_writable {
                return Err(Error::generic(
                    "Failed to reopen bitmaps rw: no write access the protocol file",
                ));
            }
            self.update_ext_header_and_dir_in_place(&list)
                .map_err(|e| Error::from_io("Cannot update bitmap directory", e))?;
        }
        Ok(ro)
    }

    /// `qcow2_co_can_store_new_dirty_bitmap()`.
    pub(crate) fn can_store_new_dirty_bitmap(
        &self,
        name: &str,
        granularity: u32,
        node_name: &str,
    ) -> Result<()> {
        if self.bitmaps.iter().any(|b| b.name == name) {
            return Err(Error::generic(format!("Bitmap already exists: {name}")));
        }
        let r: Result<()> = (|| {
            if self.qcow_version < 3 {
                // Without autoclear bits every bitmap would have to be dropped on open, as
                // some program without bitmap support might have touched the image.
                return Err(Error::generic("Cannot store dirty bitmaps in qcow2 v2 files"));
            }
            self.check_constraints_on_bitmap(name, granularity)?;
            let mut nb = 1u32;
            let mut dir = calc_dir_entry_size(name.len(), 0);
            for b in &self.bitmaps {
                nb += 1;
                dir += calc_dir_entry_size(b.name.len(), 0);
            }
            if nb > QCOW2_MAX_BITMAPS {
                return Err(Error::generic(
                    "Maximum number of persistent bitmaps is already reached",
                ));
            }
            if dir > QCOW2_MAX_BITMAP_DIRECTORY_SIZE {
                return Err(Error::generic("Not enough space in the bitmap directory"));
            }
            Ok(())
        })();
        r.map_err(|e| {
            e.prepend(format!("Can't make bitmap '{name}' persistent in '{node_name}': "))
        })
    }
}
