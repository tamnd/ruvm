// SPDX-License-Identifier: GPL-2.0-or-later

//! Guest to host mapping, from block/qcow2-cluster.c: the L1 table, L2 tables and their slices,
//! cluster allocation with copy on write, discard, zeroing and zero cluster expansion.
//!
//! QEMU runs several allocating writes at once and tracks them in `s->cluster_allocs` so that
//! overlapping ones wait for each other. Requests here are serialized by the image lock, so
//! nothing is ever in flight while another request looks at the tables and the dependency
//! handling is not needed.

use std::io;

use super::cache::{CacheId, get_be64, set_be64};
use super::header::L1_SIZE_OFFSET;
use super::state::*;
use crate::node::errno;

/// `INV_OFFSET`.
pub(crate) const INV_OFFSET: u64 = u64::MAX;

/// `BDRV_REQUEST_MAX_BYTES`.
pub(crate) const REQUEST_MAX_BYTES: u64 = (i32::MAX as u64 >> 9) << 9;

/// `Qcow2COWRegion`: offsets are relative to the start of the first cluster of the request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CowRegion {
    pub offset: u64,
    pub nb_bytes: u64,
}

/// `QCowL2Meta`: an allocation that still has to be linked into the L2 table.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct L2Meta {
    /// Guest offset of the first cluster.
    pub offset: u64,
    /// Host offset of the first allocated cluster.
    pub alloc_offset: u64,
    pub nb_clusters: u64,
    /// The clusters were allocated already and are only written in place.
    pub keep_old_clusters: bool,
    pub cow_start: CowRegion,
    pub cow_end: CowRegion,
    /// Preallocation: the L2 bitmap is not touched.
    pub prealloc: bool,
    /// The COW regions are handled already.
    pub skip_cow: bool,
    /// The guest data goes out together with the COW regions, `data_qiov`.
    pub merge_data: bool,
}

impl L2Meta {
    /// `l2meta_cow_start()`.
    pub(crate) fn cow_start_offset(&self) -> u64 {
        self.offset + self.cow_start.offset
    }
}

impl State {
    /// `get_l2_entry()` on a cached slice.
    pub(crate) fn l2_get(&self, slot: usize, idx: usize) -> u64 {
        self.l2_entry(self.l2_table_cache.table(slot), idx)
    }

    /// `get_l2_bitmap()` on a cached slice.
    pub(crate) fn l2_get_bitmap(&self, slot: usize, idx: usize) -> u64 {
        self.l2_bitmap(self.l2_table_cache.table(slot), idx)
    }

    /// `set_l2_entry()` on a cached slice. The caller marks the slice dirty.
    pub(crate) fn l2_set(&mut self, slot: usize, idx: usize, entry: u64) {
        let i = idx * (self.l2_entry_size() / 8) as usize;
        set_be64(self.l2_table_cache.table_mut(slot), i, entry);
    }

    /// `set_l2_bitmap()` on a cached slice.
    pub(crate) fn l2_set_bitmap(&mut self, slot: usize, idx: usize, bitmap: u64) {
        assert!(self.has_subclusters());
        set_be64(self.l2_table_cache.table_mut(slot), idx * 2 + 1, bitmap);
    }

    /// `qcow2_parse_compressed_l2_entry()`: host offset and length of the compressed data.
    pub(crate) fn parse_compressed_l2_entry(&self, l2_entry: u64) -> (u64, u64) {
        assert_eq!(self.cluster_type(l2_entry), ClusterType::Compressed);
        let coffset = l2_entry & self.cluster_offset_mask;
        let nb_csectors = ((l2_entry >> self.csize_shift) & self.csize_mask) + 1;
        let csize = nb_csectors * QCOW2_COMPRESSED_SECTOR_SIZE
            - (coffset & (QCOW2_COMPRESSED_SECTOR_SIZE - 1));
        (coffset, csize)
    }

    /// `qcow2_shrink_l1_table()`.
    pub(crate) fn shrink_l1_table(&mut self, exact_size: u64) -> io::Result<()> {
        if exact_size >= self.l1_size as u64 {
            return Ok(());
        }
        let new_l1_size = exact_size as usize;
        let r = self
            .file
            .pwrite_zeroes(
                self.l1_table_offset + new_l1_size as u64 * L1E_SIZE,
                (self.l1_size as u64 - new_l1_size as u64) * L1E_SIZE,
                false,
            )
            .and_then(|()| self.file.flush());
        if let Err(e) = r {
            // The table on disk may be partly overwritten, so forget the entries in memory too
            // rather than risk using them.
            for e in &mut self.l1_table[new_l1_size..] {
                *e = 0;
            }
            return Err(e);
        }
        for i in (new_l1_size..self.l1_size as usize).rev() {
            let off = self.l1_table[i] & L1E_OFFSET_MASK;
            if off == 0 {
                continue;
            }
            self.free_clusters(off, self.cluster_size, DiscardType::Always);
            self.l1_table[i] = 0;
        }
        Ok(())
    }

    /// `qcow2_grow_l1_table()`.
    pub(crate) fn grow_l1_table(&mut self, min_size: u64, exact_size: bool) -> io::Result<()> {
        if min_size <= self.l1_size as u64 {
            return Ok(());
        }
        // Keeps the growth loop below from overflowing.
        if min_size > i32::MAX as u64 / L1E_SIZE {
            return Err(errno(libc::EFBIG));
        }
        let new_l1_size = if exact_size {
            min_size
        } else {
            let mut n = (self.l1_size as u64).max(1);
            while min_size > n {
                n = (n * 3).div_ceil(2);
            }
            n
        };
        if new_l1_size > QCOW_MAX_L1_SIZE / L1E_SIZE {
            return Err(errno(libc::EFBIG));
        }
        let new_l1_size2 = new_l1_size * L1E_SIZE;
        let mut new_table = self.l1_table.clone();
        new_table.resize(new_l1_size as usize, 0);

        let new_offset = self.alloc_clusters(new_l1_size2)?;
        let r = (|| {
            self.cache_flush(CacheId::Refcount)?;
            // The L1 position is not updated yet, so these clusters must really be free.
            self.pre_write_overlap_check(0, new_offset, new_l1_size2, false)?;
            let mut buf = vec![0u8; new_l1_size2 as usize];
            for (i, v) in new_table.iter().enumerate() {
                set_be64(&mut buf, i, *v);
            }
            self.file.pwrite(new_offset, &buf)?;
            self.file.flush()?;
            let mut data = [0u8; 12];
            data[..4].copy_from_slice(&(new_l1_size as u32).to_be_bytes());
            data[4..].copy_from_slice(&new_offset.to_be_bytes());
            self.file.pwrite(L1_SIZE_OFFSET, &data)?;
            self.file.flush()
        })();
        if let Err(e) = r {
            self.free_clusters(new_offset, new_l1_size2, DiscardType::Other);
            return Err(e);
        }
        let old_offset = self.l1_table_offset;
        let old_size = self.l1_size as u64;
        self.l1_table_offset = new_offset;
        self.l1_table = new_table;
        self.l1_size = new_l1_size as u32;
        self.free_clusters(old_offset, old_size * L1E_SIZE, DiscardType::Other);
        Ok(())
    }

    /// `l2_load()`: the slice of the L2 table at `l2_offset` that maps `offset`.
    fn l2_load(&mut self, offset: u64, l2_offset: u64) -> io::Result<usize> {
        let start_of_slice = self.l2_entry_size()
            * (self.offset_to_l2_index(offset) - self.offset_to_l2_slice_index(offset)) as u64;
        self.cache_get(CacheId::L2, l2_offset + start_of_slice)
    }

    /// `qcow2_write_l1_entry()`. The file layer here has byte granularity, so one entry is
    /// written.
    pub(crate) fn write_l1_entry(&mut self, l1_index: usize) -> io::Result<()> {
        let off = self.l1_table_offset + L1E_SIZE * l1_index as u64;
        self.pre_write_overlap_check(OL_ACTIVE_L1, off, L1E_SIZE, false)?;
        self.file.pwrite(off, &self.l1_table[l1_index].to_be_bytes())?;
        self.file.flush()
    }

    /// `l2_allocate()`: gives L1 entry `l1_index` a new L2 table, copying the old one if there
    /// is one.
    fn l2_allocate(&mut self, l1_index: usize) -> io::Result<()> {
        let old_l2_offset = self.l1_table[l1_index];
        let l2_bytes = self.l2_size * self.l2_entry_size();
        let mut l2_offset = 0;
        let r = (|| {
            l2_offset = self.alloc_clusters(l2_bytes)?;
            assert_eq!(l2_offset & L1E_OFFSET_MASK, l2_offset);
            if l2_offset == 0 {
                self.signal_corruption(
                    true,
                    -1,
                    -1,
                    "Preventing invalid allocation of L2 table at offset 0",
                );
                return Err(errno(libc::EIO));
            }
            self.cache_flush(CacheId::Refcount)?;

            let slice_size2 = self.l2_slice_size * self.l2_entry_size();
            let n_slices = self.cluster_size / slice_size2;
            for slice in 0..n_slices {
                let slot = self.cache_get_empty(CacheId::L2, l2_offset + slice * slice_size2)?;
                if old_l2_offset & L1E_OFFSET_MASK == 0 {
                    self.l2_table_cache.table_mut(slot).fill(0);
                } else {
                    let old_off = (old_l2_offset & L1E_OFFSET_MASK) + slice * slice_size2;
                    let old = match self.cache_get(CacheId::L2, old_off) {
                        Ok(o) => o,
                        Err(e) => {
                            self.cache_put(CacheId::L2, slot);
                            return Err(e);
                        }
                    };
                    let data = self.l2_table_cache.table(old).to_vec();
                    self.l2_table_cache.table_mut(slot).copy_from_slice(&data);
                    self.cache_put(CacheId::L2, old);
                }
                self.l2_table_cache.mark_dirty(slot);
                self.cache_put(CacheId::L2, slot);
            }
            self.cache_flush(CacheId::L2)?;

            self.l1_table[l1_index] = l2_offset | QCOW_OFLAG_COPIED;
            self.write_l1_entry(l1_index)
        })();
        if let Err(e) = r {
            self.l1_table[l1_index] = old_l2_offset;
            if l2_offset > 0 {
                self.free_clusters(l2_offset, l2_bytes, DiscardType::Always);
            }
            return Err(e);
        }
        Ok(())
    }

    /// `qcow2_get_subcluster_range_type()`: the type of subcluster `sc_from` and how many
    /// subclusters from there on share it.
    fn get_subcluster_range_type(
        &self,
        l2_entry: u64,
        l2_bitmap: u64,
        sc_from: u32,
    ) -> (SubclusterType, Option<u32>) {
        let ty = self.subcluster_type(l2_entry, l2_bitmap, sc_from);
        if ty == SubclusterType::Invalid {
            return (ty, None);
        }
        if !self.has_subclusters() || ty == SubclusterType::Compressed {
            return (ty, Some(self.subclusters_per_cluster - sc_from));
        }
        let n = match ty {
            SubclusterType::Normal => {
                let val = (l2_bitmap | sub_alloc_range(0, sc_from)) as u32;
                val.trailing_ones() - sc_from
            }
            SubclusterType::ZeroPlain | SubclusterType::ZeroAlloc => {
                let val = ((l2_bitmap | sub_zero_range(0, sc_from)) >> 32) as u32;
                val.trailing_ones() - sc_from
            }
            SubclusterType::UnallocatedPlain | SubclusterType::UnallocatedAlloc => {
                let val = (((l2_bitmap >> 32) | l2_bitmap) & !sub_alloc_range(0, sc_from)) as u32;
                val.trailing_zeros() - sc_from
            }
            _ => unreachable!(),
        };
        (ty, Some(n))
    }

    /// `count_contiguous_subclusters()`. On an invalid entry, `Err` carries its index.
    fn count_contiguous_subclusters(
        &self,
        nb_clusters: u64,
        sc_index: u32,
        slot: usize,
        l2_index: usize,
    ) -> Result<u64, usize> {
        assert!(l2_index as u64 + nb_clusters <= self.l2_slice_size);
        let mut count = 0u64;
        let mut check_offset = false;
        let mut expected_offset = 0;
        let mut expected_type = SubclusterType::Normal;
        for i in 0..nb_clusters as usize {
            let first_sc = if i == 0 { sc_index } else { 0 };
            let l2_entry = self.l2_get(slot, l2_index + i);
            let l2_bitmap = self.l2_get_bitmap(slot, l2_index + i);
            let (ty, n) = self.get_subcluster_range_type(l2_entry, l2_bitmap, first_sc);
            let Some(n) = n else {
                return Err(l2_index + i);
            };
            if i == 0 {
                if ty == SubclusterType::Compressed {
                    // Compressed clusters always go one by one.
                    return Ok(n as u64);
                }
                expected_type = ty;
                expected_offset = l2_entry & L2E_OFFSET_MASK;
                check_offset = matches!(
                    ty,
                    SubclusterType::Normal
                        | SubclusterType::ZeroAlloc
                        | SubclusterType::UnallocatedAlloc
                );
            } else if ty != expected_type {
                break;
            } else if check_offset {
                expected_offset += self.cluster_size;
                if expected_offset != l2_entry & L2E_OFFSET_MASK {
                    break;
                }
            }
            count += n as u64;
            // Stop where the type changes inside a cluster.
            if first_sc + n < self.subclusters_per_cluster {
                break;
            }
        }
        Ok(count)
    }

    /// `qcow2_get_host_offset()`: maps guest `offset`. On return `bytes` is how many bytes from
    /// there on share the subcluster type and, where it applies, are contiguous in the file.
    /// The host offset is 0 for unallocated clusters and the whole L2 entry for compressed ones.
    pub(crate) fn get_host_offset(
        &mut self,
        offset: u64,
        bytes: &mut u64,
    ) -> io::Result<(u64, SubclusterType)> {
        let offset_in_cluster = self.offset_into_cluster(offset);
        let mut bytes_needed = *bytes + offset_in_cluster;
        // Bytes between the start of this cluster and the end of its L2 slice.
        let mut bytes_available = (self.l2_slice_size
            - self.offset_to_l2_slice_index(offset) as u64)
            << self.cluster_bits;
        bytes_needed = bytes_needed.min(bytes_available);
        let mut host_offset = 0;
        let ty;

        let l1_index = self.offset_to_l1_index(offset);
        let l2_offset = if l1_index < self.l1_size as usize {
            self.l1_table[l1_index] & L1E_OFFSET_MASK
        } else {
            0
        };
        if l2_offset == 0 {
            ty = SubclusterType::UnallocatedPlain;
        } else {
            if self.offset_into_cluster(l2_offset) != 0 {
                self.signal_corruption(
                    true,
                    -1,
                    -1,
                    &format!(
                        "L2 table offset {l2_offset:#x} unaligned (L1 index: {l1_index:#x})",
                        l2_offset = Hx(l2_offset),
                        l1_index = Hx(l1_index)
                    ),
                );
                return Err(errno(libc::EIO));
            }
            let slot = self.l2_load(offset, l2_offset)?;
            let r = self.get_host_offset_in_slice(slot, offset, l2_offset, bytes_needed);
            self.cache_put(CacheId::L2, slot);
            let (h, t, avail) = r?;
            host_offset = h;
            ty = t;
            bytes_available = avail;
        }
        bytes_available = bytes_available.min(bytes_needed);
        *bytes = bytes_available - offset_in_cluster;
        Ok((host_offset, ty))
    }

    fn get_host_offset_in_slice(
        &mut self,
        slot: usize,
        offset: u64,
        l2_offset: u64,
        bytes_needed: u64,
    ) -> io::Result<(u64, SubclusterType, u64)> {
        let offset_in_cluster = self.offset_into_cluster(offset);
        let l2_index = self.offset_to_l2_slice_index(offset);
        let sc_index = self.offset_to_sc_index(offset);
        let l2_entry = self.l2_get(slot, l2_index);
        let l2_bitmap = self.l2_get_bitmap(slot, l2_index);
        let nb_clusters = self.size_to_clusters(bytes_needed);
        let mut host_offset = 0;

        let ty = self.subcluster_type(l2_entry, l2_bitmap, sc_index);
        if self.qcow_version < 3
            && matches!(ty, SubclusterType::ZeroPlain | SubclusterType::ZeroAlloc)
        {
            self.signal_corruption(
                true,
                -1,
                -1,
                &format!(
                    "Zero cluster entry found in pre-v3 image (L2 offset: {l2_offset:#x}, L2 index: \
                     {l2_index:#x})",
                    l2_offset = Hx(l2_offset),
                    l2_index = Hx(l2_index)
                ),
            );
            return Err(errno(libc::EIO));
        }
        match ty {
            SubclusterType::Invalid => {}
            SubclusterType::Compressed => {
                if self.has_data_file() {
                    self.signal_corruption(
                        true,
                        -1,
                        -1,
                        &format!(
                            "Compressed cluster entry found in image with external data file (L2 \
                             offset: {l2_offset:#x}, L2 index: {l2_index:#x})",
                            l2_offset = Hx(l2_offset),
                            l2_index = Hx(l2_index)
                        ),
                    );
                    return Err(errno(libc::EIO));
                }
                host_offset = l2_entry;
            }
            SubclusterType::ZeroPlain | SubclusterType::UnallocatedPlain => {}
            SubclusterType::ZeroAlloc
            | SubclusterType::Normal
            | SubclusterType::UnallocatedAlloc => {
                let host_cluster_offset = l2_entry & L2E_OFFSET_MASK;
                host_offset = host_cluster_offset + offset_in_cluster;
                if self.offset_into_cluster(host_cluster_offset) != 0 {
                    self.signal_corruption(
                        true,
                        -1,
                        -1,
                        &format!(
                            "Cluster allocation offset {host_cluster_offset:#x} unaligned (L2 offset: \
                             {l2_offset:#x}, L2 index: {l2_index:#x})",
                            host_cluster_offset = Hx(host_cluster_offset),
                            l2_offset = Hx(l2_offset),
                            l2_index = Hx(l2_index)
                        ),
                    );
                    return Err(errno(libc::EIO));
                }
                if self.has_data_file() && host_offset != offset {
                    self.signal_corruption(
                        true,
                        -1,
                        -1,
                        &format!(
                            "External data file host cluster offset {host_cluster_offset:#x} does not \
                             match guest cluster offset: {guest:#x}, L2 index: {l2_index:#x})",
                            host_cluster_offset = Hx(host_cluster_offset),
                            guest = Hx(offset - offset_in_cluster),
                            l2_index = Hx(l2_index)
                        ),
                    );
                    return Err(errno(libc::EIO));
                }
            }
        }
        match self.count_contiguous_subclusters(nb_clusters, sc_index, slot, l2_index) {
            Ok(sc) => Ok((host_offset, ty, (sc + sc_index as u64) << self.subcluster_bits)),
            Err(bad_index) => {
                self.signal_corruption(
                    true,
                    -1,
                    -1,
                    &format!(
                        "Invalid cluster entry found  (L2 offset: {l2_offset:#x}, L2 index: {bad_index:#x})",
                        l2_offset = Hx(l2_offset),
                        bad_index = Hx(bad_index)
                    ),
                );
                Err(errno(libc::EIO))
            }
        }
    }

    /// `get_cluster_table()`: the L2 slice for `offset`, allocating or copying the L2 table if
    /// needed. Returns the cache slot and the index in the slice.
    pub(crate) fn get_cluster_table(&mut self, offset: u64) -> io::Result<(usize, usize)> {
        let l1_index = self.offset_to_l1_index(offset);
        if l1_index >= self.l1_size as usize {
            self.grow_l1_table(l1_index as u64 + 1, false)?;
        }
        assert!(l1_index < self.l1_size as usize);
        let mut l2_offset = self.l1_table[l1_index] & L1E_OFFSET_MASK;
        if self.offset_into_cluster(l2_offset) != 0 {
            self.signal_corruption(
                true,
                -1,
                -1,
                &format!(
                    "L2 table offset {l2_offset:#x} unaligned (L1 index: {l1_index:#x})",
                    l2_offset = Hx(l2_offset),
                    l1_index = Hx(l1_index)
                ),
            );
            return Err(errno(libc::EIO));
        }
        if self.l1_table[l1_index] & QCOW_OFLAG_COPIED == 0 {
            // Allocate a new L2 table (copying the old one), then drop the old reference.
            self.l2_allocate(l1_index)?;
            if l2_offset != 0 {
                self.free_clusters(
                    l2_offset,
                    self.l2_size * self.l2_entry_size(),
                    DiscardType::Other,
                );
            }
            l2_offset = self.l1_table[l1_index] & L1E_OFFSET_MASK;
            assert_eq!(self.offset_into_cluster(l2_offset), 0);
        }
        let slot = self.l2_load(offset, l2_offset)?;
        Ok((slot, self.offset_to_l2_slice_index(offset)))
    }

    /// `qcow2_alloc_compressed_cluster_offset()`: allocates `compressed_size` bytes for the
    /// compressed cluster at guest `offset` and points the L2 entry at them.
    pub(crate) fn alloc_compressed_cluster_offset(
        &mut self,
        offset: u64,
        compressed_size: u64,
    ) -> io::Result<u64> {
        assert!(!self.has_data_file());
        let (slot, l2_index) = self.get_cluster_table(offset)?;
        // Compression cannot overwrite anything.
        if self.l2_get(slot, l2_index) & L2E_OFFSET_MASK != 0 {
            self.cache_put(CacheId::L2, slot);
            return Err(errno(libc::EIO));
        }
        let cluster_offset = match self.alloc_bytes(compressed_size) {
            Ok(o) => o,
            Err(e) => {
                self.cache_put(CacheId::L2, slot);
                return Err(e);
            }
        };
        let nb_csectors = (cluster_offset + compressed_size - 1) / QCOW2_COMPRESSED_SECTOR_SIZE
            - cluster_offset / QCOW2_COMPRESSED_SECTOR_SIZE;
        assert_eq!(cluster_offset & self.cluster_offset_mask, cluster_offset);
        assert_eq!(nb_csectors & self.csize_mask, nb_csectors);
        let entry = cluster_offset | QCOW_OFLAG_COMPRESSED | (nb_csectors << self.csize_shift);
        // Compressed clusters never have the COPIED flag.
        self.l2_table_cache.mark_dirty(slot);
        self.l2_set(slot, l2_index, entry);
        if self.has_subclusters() {
            self.l2_set_bitmap(slot, l2_index, 0);
        }
        self.cache_put(CacheId::L2, slot);
        Ok(cluster_offset)
    }

    /// `perform_cow()`: copies the parts of the new clusters the request does not write. With
    /// `data`, the guest data is written in the same go.
    fn perform_cow(&mut self, m: &L2Meta, data: Option<&[u8]>) -> io::Result<()> {
        let start = m.cow_start;
        let end = m.cow_end;
        let data_bytes = end.offset - (start.offset + start.nb_bytes);
        assert!(start.offset + start.nb_bytes <= end.offset);
        if (start.nb_bytes == 0 && end.nb_bytes == 0) || m.skip_cow {
            return Ok(());
        }
        // Read both regions at once if the gap between them is small.
        let merge_reads = start.nb_bytes != 0 && end.nb_bytes != 0 && data_bytes <= 16384;
        let mut start_buf = vec![0u8; start.nb_bytes as usize];
        let mut end_buf = vec![0u8; end.nb_bytes as usize];
        if merge_reads {
            let mut buf = vec![0u8; (start.nb_bytes + data_bytes + end.nb_bytes) as usize];
            self.cow_read(m.offset + start.offset, &mut buf)?;
            start_buf.copy_from_slice(&buf[..start.nb_bytes as usize]);
            end_buf.copy_from_slice(&buf[buf.len() - end.nb_bytes as usize..]);
        } else {
            self.cow_read(m.offset + start.offset, &mut start_buf)?;
            self.cow_read(m.offset + end.offset, &mut end_buf)?;
        }
        if self.crypto.is_some() {
            self.encrypt(m.alloc_offset + start.offset, m.offset + start.offset, &mut start_buf)?;
            self.encrypt(m.alloc_offset + end.offset, m.offset + end.offset, &mut end_buf)?;
        }
        if let Some(data) = data {
            assert_eq!(data.len() as u64, data_bytes);
            let mut all = start_buf;
            all.extend_from_slice(data);
            all.extend_from_slice(&end_buf);
            self.cow_write(m.alloc_offset, start.offset, &all)?;
        } else {
            self.cow_write(m.alloc_offset, start.offset, &start_buf)?;
            self.cow_write(m.alloc_offset, end.offset, &end_buf)?;
        }
        // The L2 update must not reach the disk before the copied data.
        self.l2_table_cache.depends_on_flush();
        Ok(())
    }

    /// `do_perform_cow_read()`: reads through the driver itself, so the backing file and
    /// compressed clusters are handled.
    fn cow_read(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        self.check_usable()?;
        self.preadv(offset, buf)
    }

    /// `do_perform_cow_write()`.
    fn cow_write(
        &mut self,
        cluster_offset: u64,
        offset_in_cluster: u64,
        buf: &[u8],
    ) -> io::Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        let off = cluster_offset + offset_in_cluster;
        self.pre_write_overlap_check(0, off, buf.len() as u64, true)?;
        self.data().pwrite(off, buf)
    }

    /// `qcow2_alloc_cluster_link_l2()`: does the COW for `m` and points the L2 entries at the
    /// new clusters.
    pub(crate) fn alloc_cluster_link_l2(
        &mut self,
        m: &L2Meta,
        data: Option<&[u8]>,
    ) -> io::Result<()> {
        assert!(m.nb_clusters > 0);
        self.perform_cow(m, data)?;

        if self.use_lazy_refcounts {
            let _ = self.mark_dirty();
        }
        if self.need_accurate_refcounts() {
            let _ = self.cache_set_dependency(CacheId::L2, CacheId::Refcount);
        }
        let (slot, l2_index) = self.get_cluster_table(m.offset)?;
        self.l2_table_cache.mark_dirty(slot);
        assert!(l2_index as u64 + m.nb_clusters <= self.l2_slice_size);
        assert!(m.cow_end.offset + m.cow_end.nb_bytes <= m.nb_clusters << self.cluster_bits);

        let mut old_clusters = Vec::new();
        for i in 0..m.nb_clusters as usize {
            let offset = m.alloc_offset + ((i as u64) << self.cluster_bits);
            // With concurrent writes to one unallocated cluster the loser frees its copy; the
            // same code here also drops the reference of a COWed cluster.
            let old = self.l2_get(slot, l2_index + i);
            if old != 0 {
                old_clusters.push(old);
            }
            assert_eq!(offset & L2E_OFFSET_MASK, offset);
            self.l2_set(slot, l2_index + i, offset | QCOW_OFLAG_COPIED);

            // Mark the subclusters that were just written.
            if self.has_subclusters() && !m.prealloc {
                let mut l2_bitmap = self.l2_get_bitmap(slot, l2_index + i);
                let cs = self.cluster_size;
                let written_from = m.cow_start.offset.max(i as u64 * cs);
                let written_to = (m.cow_end.offset + m.cow_end.nb_bytes).min((i as u64 + 1) * cs);
                assert!(written_from < written_to);
                let first_sc = self.offset_to_sc_index(written_from);
                let last_sc = self.offset_to_sc_index(written_to - 1);
                l2_bitmap |= sub_alloc_range(first_sc, last_sc + 1);
                l2_bitmap &= !sub_zero_range(first_sc, last_sc + 1);
                self.l2_set_bitmap(slot, l2_index + i, l2_bitmap);
            }
        }
        self.cache_put(CacheId::L2, slot);

        // Drop the references of the old clusters of a COW. Clusters that reach 0 are not
        // discarded: the next write reuses them anyway.
        if !m.keep_old_clusters {
            for old in old_clusters {
                self.free_any_cluster(old, DiscardType::Never);
            }
        }
        Ok(())
    }

    /// `qcow2_alloc_cluster_abort()`: frees the clusters of a failed request.
    pub(crate) fn alloc_cluster_abort(&mut self, m: &L2Meta) {
        if !self.has_data_file() && !m.keep_old_clusters {
            self.free_clusters(
                m.alloc_offset,
                m.nb_clusters << self.cluster_bits,
                DiscardType::Never,
            );
        }
    }

    /// `calculate_l2_meta()`: works out the COW regions of a write and adds an [`L2Meta`] for
    /// it, unless nothing has to change in the L2 table.
    #[allow(clippy::too_many_arguments)]
    fn calculate_l2_meta(
        &mut self,
        host_cluster_offset: u64,
        guest_offset: u64,
        bytes: u64,
        slot: usize,
        metas: &mut Vec<L2Meta>,
        keep_old: bool,
    ) -> io::Result<()> {
        let mut l2_index = self.offset_to_l2_slice_index(guest_offset);
        let cow_start_to = self.offset_into_cluster(guest_offset);
        let cow_end_from = cow_start_to + bytes;
        let nb_clusters = self.size_to_clusters(cow_end_from);
        let mut skip_cow = keep_old;
        assert!(nb_clusters <= self.l2_slice_size - l2_index as u64);

        // Check the type of every subcluster involved.
        for i in 0..nb_clusters {
            let l2_entry = self.l2_get(slot, l2_index + i as usize);
            let l2_bitmap = self.l2_get_bitmap(slot, l2_index + i as usize);
            let ty;
            if skip_cow {
                let cs = self.cluster_size;
                let write_from = cow_start_to.max(i * cs);
                let write_to = cow_end_from.min((i + 1) * cs);
                let first_sc = self.offset_to_sc_index(write_from);
                let last_sc = self.offset_to_sc_index(write_to - 1);
                let (t, cnt) = self.get_subcluster_range_type(l2_entry, l2_bitmap, first_sc);
                ty = t;
                // Is any of the subclusters not a normal one?
                if ty != SubclusterType::Normal || first_sc + cnt.unwrap_or(0) <= last_sc {
                    skip_cow = false;
                }
            } else {
                // Even without COW, look for invalid entries.
                ty = self.subcluster_type(l2_entry, l2_bitmap, 0);
            }
            if ty == SubclusterType::Invalid {
                let l1_index = self.offset_to_l1_index(guest_offset);
                let l2_offset = self.l1_table[l1_index] & L1E_OFFSET_MASK;
                let idx = l2_index + i as usize;
                self.signal_corruption(
                    true,
                    -1,
                    -1,
                    &format!(
                        "Invalid cluster entry found (L2 offset: {l2_offset:#x}, L2 index: {idx:#x})",
                        l2_offset = Hx(l2_offset),
                        idx = Hx(idx)
                    ),
                );
                return Err(errno(libc::EIO));
            }
        }
        if skip_cow {
            return Ok(());
        }

        // The first cluster.
        let l2_entry = self.l2_get(slot, l2_index);
        let l2_bitmap = self.l2_get_bitmap(slot, l2_index);
        let sc_index = self.offset_to_sc_index(guest_offset);
        let ty = self.subcluster_type(l2_entry, l2_bitmap, sc_index);
        let sb = self.subcluster_bits;
        let cow_start_from = if !keep_old {
            match ty {
                SubclusterType::Compressed => 0,
                SubclusterType::Normal
                | SubclusterType::ZeroAlloc
                | SubclusterType::UnallocatedAlloc => {
                    if self.has_subclusters() {
                        // Skip the zero and unallocated subclusters at the start.
                        let alloc_bitmap = (l2_bitmap & QCOW_L2_BITMAP_ALL_ALLOC) as u32;
                        (sc_index.min(alloc_bitmap.trailing_zeros()) as u64) << sb
                    } else {
                        0
                    }
                }
                SubclusterType::ZeroPlain | SubclusterType::UnallocatedPlain => {
                    (sc_index as u64) << sb
                }
                SubclusterType::Invalid => unreachable!(),
            }
        } else {
            match ty {
                SubclusterType::Normal => cow_start_to,
                SubclusterType::ZeroAlloc | SubclusterType::UnallocatedAlloc => {
                    (sc_index as u64) << sb
                }
                _ => unreachable!(),
            }
        };

        // The last cluster.
        l2_index += nb_clusters as usize - 1;
        let l2_entry = self.l2_get(slot, l2_index);
        let l2_bitmap = self.l2_get_bitmap(slot, l2_index);
        let sc_index = self.offset_to_sc_index(guest_offset + bytes - 1);
        let ty = self.subcluster_type(l2_entry, l2_bitmap, sc_index);
        let cow_end_to = if !keep_old {
            match ty {
                SubclusterType::Compressed => cow_end_from.next_multiple_of(self.cluster_size),
                SubclusterType::Normal
                | SubclusterType::ZeroAlloc
                | SubclusterType::UnallocatedAlloc => {
                    let mut to = cow_end_from.next_multiple_of(self.cluster_size);
                    if self.has_subclusters() {
                        // Skip the zero and unallocated subclusters at the end.
                        let alloc_bitmap = (l2_bitmap & QCOW_L2_BITMAP_ALL_ALLOC) as u32;
                        to -= ((self.subclusters_per_cluster - sc_index - 1)
                            .min(alloc_bitmap.leading_zeros())
                            as u64)
                            << sb;
                    }
                    to
                }
                SubclusterType::ZeroPlain | SubclusterType::UnallocatedPlain => {
                    cow_end_from.next_multiple_of(self.subcluster_size)
                }
                SubclusterType::Invalid => unreachable!(),
            }
        } else {
            match ty {
                SubclusterType::Normal => cow_end_from,
                SubclusterType::ZeroAlloc | SubclusterType::UnallocatedAlloc => {
                    cow_end_from.next_multiple_of(self.subcluster_size)
                }
                _ => unreachable!(),
            }
        };

        metas.push(L2Meta {
            offset: self.start_of_cluster(guest_offset),
            alloc_offset: host_cluster_offset,
            nb_clusters,
            keep_old_clusters: keep_old,
            cow_start: CowRegion {
                offset: cow_start_from,
                nb_bytes: cow_start_to - cow_start_from,
            },
            cow_end: CowRegion { offset: cow_end_from, nb_bytes: cow_end_to - cow_end_from },
            prealloc: false,
            skip_cow: false,
            merge_data: false,
        });
        Ok(())
    }

    /// `cluster_needs_new_alloc()`: the cluster is unallocated or shared, so a write cannot go
    /// in place.
    fn cluster_needs_new_alloc(&self, l2_entry: u64) -> bool {
        match self.cluster_type(l2_entry) {
            ClusterType::Normal | ClusterType::ZeroAlloc => l2_entry & QCOW_OFLAG_COPIED == 0,
            ClusterType::Unallocated | ClusterType::Compressed | ClusterType::ZeroPlain => true,
        }
    }

    /// `count_single_write_clusters()`.
    fn count_single_write_clusters(
        &self,
        nb_clusters: u64,
        slot: usize,
        l2_index: usize,
        new_alloc: bool,
    ) -> u64 {
        let mut expected_offset = self.l2_get(slot, l2_index) & L2E_OFFSET_MASK;
        let mut i = 0;
        while i < nb_clusters {
            let l2_entry = self.l2_get(slot, l2_index + i as usize);
            if self.cluster_needs_new_alloc(l2_entry) != new_alloc {
                break;
            }
            if !new_alloc {
                if expected_offset != l2_entry & L2E_OFFSET_MASK {
                    break;
                }
                expected_offset += self.cluster_size;
            }
            i += 1;
        }
        i
    }

    /// `handle_copied()`: counts the clusters at `guest_offset` that can be written in place.
    /// Returns whether there were any.
    fn handle_copied(
        &mut self,
        guest_offset: u64,
        host_offset: &mut u64,
        bytes: &mut u64,
        metas: &mut Vec<L2Meta>,
    ) -> io::Result<bool> {
        assert!(
            *host_offset == INV_OFFSET
                || self.offset_into_cluster(guest_offset) == self.offset_into_cluster(*host_offset)
        );
        let l2_index = self.offset_to_l2_slice_index(guest_offset);
        let nb_clusters = self
            .size_to_clusters(self.offset_into_cluster(guest_offset) + *bytes)
            .min(self.l2_slice_size - l2_index as u64)
            .min(REQUEST_MAX_BYTES >> self.cluster_bits);

        let (slot, l2_index) = self.get_cluster_table(guest_offset)?;
        let l2_entry = self.l2_get(slot, l2_index);
        let cluster_offset = l2_entry & L2E_OFFSET_MASK;

        let r = (|| {
            if self.cluster_needs_new_alloc(l2_entry) {
                return Ok(false);
            }
            if self.offset_into_cluster(cluster_offset) != 0 {
                let what =
                    if l2_entry & QCOW_OFLAG_ZERO != 0 { "Preallocated zero" } else { "Data" };
                self.signal_corruption(
                    true,
                    -1,
                    -1,
                    &format!(
                        "{what} cluster offset {cluster_offset:#x} unaligned (guest offset: {guest_offset:#x})",
                        cluster_offset = Hx(cluster_offset),
                        guest_offset = Hx(guest_offset)
                    ),
                );
                return Err(errno(libc::EIO));
            }
            // A specific host offset is required: check it.
            if *host_offset != INV_OFFSET && cluster_offset != *host_offset {
                *bytes = 0;
                return Ok(false);
            }
            // Keep all the COPIED clusters.
            let keep_clusters =
                self.count_single_write_clusters(nb_clusters, slot, l2_index, false);
            assert!(keep_clusters <= nb_clusters);
            *bytes = (*bytes)
                .min(keep_clusters * self.cluster_size - self.offset_into_cluster(guest_offset));
            assert!(*bytes != 0);
            self.calculate_l2_meta(cluster_offset, guest_offset, *bytes, slot, metas, true)?;
            Ok(true)
        })();
        self.cache_put(CacheId::L2, slot);
        // Only hand out a host offset when progress was made.
        if let Ok(true) = r {
            *host_offset = cluster_offset + self.offset_into_cluster(guest_offset);
        }
        r
    }

    /// `do_alloc_cluster_offset()`.
    fn do_alloc_cluster_offset(
        &mut self,
        guest_offset: u64,
        host_offset: &mut u64,
        nb_clusters: &mut u64,
    ) -> io::Result<()> {
        if self.has_data_file() {
            assert!(
                *host_offset == INV_OFFSET || *host_offset == self.start_of_cluster(guest_offset)
            );
            *host_offset = self.start_of_cluster(guest_offset);
            return Ok(());
        }
        if *host_offset == INV_OFFSET {
            *host_offset = self.alloc_clusters(*nb_clusters * self.cluster_size)?;
        } else {
            *nb_clusters = self.alloc_clusters_at(*host_offset, *nb_clusters)?;
        }
        Ok(())
    }

    /// `handle_alloc()`: allocates new clusters for the part of the request that cannot be
    /// written in place. Returns whether any were allocated.
    fn handle_alloc(
        &mut self,
        guest_offset: u64,
        host_offset: &mut u64,
        bytes: &mut u64,
        metas: &mut Vec<L2Meta>,
    ) -> io::Result<bool> {
        assert!(*bytes > 0);
        let l2_index = self.offset_to_l2_slice_index(guest_offset);
        let nb_clusters = self
            .size_to_clusters(self.offset_into_cluster(guest_offset) + *bytes)
            .min(self.l2_slice_size - l2_index as u64)
            .min(REQUEST_MAX_BYTES >> self.cluster_bits);

        let (slot, l2_index) = self.get_cluster_table(guest_offset)?;
        let r = (|| {
            let mut nb_clusters =
                self.count_single_write_clusters(nb_clusters, slot, l2_index, true);
            // Only called when there are no in place clusters, so something must need COW.
            assert!(nb_clusters > 0);
            let mut alloc_cluster_offset = if *host_offset == INV_OFFSET {
                INV_OFFSET
            } else {
                self.start_of_cluster(*host_offset)
            };
            self.do_alloc_cluster_offset(
                guest_offset,
                &mut alloc_cluster_offset,
                &mut nb_clusters,
            )?;
            // The allocation cannot be extended contiguously.
            if nb_clusters == 0 {
                *bytes = 0;
                return Ok(false);
            }
            assert!(alloc_cluster_offset != INV_OFFSET);
            let oic = self.offset_into_cluster(guest_offset);
            let requested_bytes = *bytes + oic;
            let avail_bytes = nb_clusters << self.cluster_bits;
            let nb_bytes = requested_bytes.min(avail_bytes);
            *host_offset = alloc_cluster_offset + oic;
            *bytes = (*bytes).min(nb_bytes - oic);
            assert!(*bytes != 0);
            self.calculate_l2_meta(alloc_cluster_offset, guest_offset, *bytes, slot, metas, false)?;
            Ok(true)
        })();
        self.cache_put(CacheId::L2, slot);
        r
    }

    /// `qcow2_alloc_host_offset()`: finds, allocating where needed, a contiguous host area for
    /// the guest range. `bytes` may come back smaller than asked. Allocations that still need
    /// linking are added to `metas` even on failure; the caller links or aborts them.
    pub(crate) fn alloc_host_offset(
        &mut self,
        offset: u64,
        bytes: &mut u64,
        metas: &mut Vec<L2Meta>,
    ) -> io::Result<u64> {
        let mut start = offset;
        let mut remaining = *bytes;
        let mut cluster_offset = INV_OFFSET;
        let mut host_offset = INV_OFFSET;
        let mut cur_bytes = 0u64;
        loop {
            if host_offset == INV_OFFSET && cluster_offset != INV_OFFSET {
                host_offset = cluster_offset;
            }
            assert!(remaining >= cur_bytes);
            start += cur_bytes;
            remaining -= cur_bytes;
            if cluster_offset != INV_OFFSET {
                cluster_offset += cur_bytes;
            }
            if remaining == 0 {
                break;
            }
            cur_bytes = remaining;

            // Clusters that can be written in place.
            if self.handle_copied(start, &mut cluster_offset, &mut cur_bytes, metas)? {
                continue;
            } else if cur_bytes == 0 {
                break;
            }
            // New clusters, contiguous with what was found so far if possible.
            if self.handle_alloc(start, &mut cluster_offset, &mut cur_bytes, metas)? {
                continue;
            }
            assert_eq!(cur_bytes, 0);
            break;
        }
        *bytes -= remaining;
        assert!(*bytes > 0);
        assert!(host_offset != INV_OFFSET);
        assert_eq!(self.offset_into_cluster(host_offset), self.offset_into_cluster(offset));
        Ok(host_offset)
    }

    /// `discard_in_l2_slice()`: returns how many clusters were handled.
    fn discard_in_l2_slice(
        &mut self,
        offset: u64,
        nb_clusters: u64,
        ty: DiscardType,
        full_discard: bool,
    ) -> io::Result<u64> {
        let (slot, l2_index) = self.get_cluster_table(offset)?;
        let nb_clusters = nb_clusters.min(self.l2_slice_size - l2_index as u64);
        for i in 0..nb_clusters as usize {
            let old_l2_entry = self.l2_get(slot, l2_index + i);
            let old_l2_bitmap = self.l2_get_bitmap(slot, l2_index + i);
            let mut new_l2_entry = old_l2_entry;
            let mut new_l2_bitmap = old_l2_bitmap;
            let ctype = self.cluster_type(old_l2_entry);
            let keep_reference = ctype != ClusterType::Compressed
                && !full_discard
                && self.discard_no_unref
                && ty == DiscardType::Request;

            // A full discard falls through to the backing file. Otherwise the area must read as
            // zeroes, which v2 images can only manage by deallocating when there is no backing
            // file; unallocated clusters without a backing file read as zeroes already.
            if full_discard {
                new_l2_entry = 0;
                new_l2_bitmap = 0;
            } else if self.backing.is_some() || ctype.is_allocated() {
                if self.has_subclusters() {
                    new_l2_entry = if keep_reference { old_l2_entry } else { 0 };
                    new_l2_bitmap = QCOW_L2_BITMAP_ALL_ZEROES;
                } else if self.qcow_version >= 3 {
                    if keep_reference {
                        new_l2_entry |= QCOW_OFLAG_ZERO;
                    } else {
                        new_l2_entry = QCOW_OFLAG_ZERO;
                    }
                } else {
                    new_l2_entry = 0;
                }
            }
            if old_l2_entry == new_l2_entry && old_l2_bitmap == new_l2_bitmap {
                continue;
            }
            // First the L2 entry, then the refcount.
            self.l2_table_cache.mark_dirty(slot);
            self.l2_set(slot, l2_index + i, new_l2_entry);
            if self.has_subclusters() {
                self.l2_set_bitmap(slot, l2_index + i, new_l2_bitmap);
            }
            if !keep_reference {
                self.free_any_cluster(old_l2_entry, ty);
            } else {
                // Keeping the reference, but the discard still goes down.
                self.discard_cluster(old_l2_entry & L2E_OFFSET_MASK, self.cluster_size, ctype, ty);
            }
        }
        self.cache_put(CacheId::L2, slot);
        Ok(nb_clusters)
    }

    /// `qcow2_cluster_discard()`.
    pub(crate) fn cluster_discard(
        &mut self,
        mut offset: u64,
        bytes: u64,
        ty: DiscardType,
        full_discard: bool,
    ) -> io::Result<()> {
        let end_offset = offset + bytes;
        // Aligned except at the end of the image.
        assert_eq!(self.offset_into_cluster(offset), 0);
        assert!(
            self.offset_into_cluster(end_offset) == 0 || end_offset == self.disk_size_sectors()
        );
        let mut nb_clusters = self.size_to_clusters(bytes);
        self.cache_discards = true;
        let mut ret = Ok(());
        while nb_clusters > 0 {
            match self.discard_in_l2_slice(offset, nb_clusters, ty, full_discard) {
                Ok(cleared) => {
                    nb_clusters -= cleared;
                    offset += cleared * self.cluster_size;
                }
                Err(e) => {
                    ret = Err(e);
                    break;
                }
            }
        }
        self.cache_discards = false;
        self.process_discards(ret.is_ok());
        ret
    }

    /// `zero_in_l2_slice()`.
    fn zero_in_l2_slice(
        &mut self,
        offset: u64,
        nb_clusters: u64,
        may_unmap: bool,
    ) -> io::Result<u64> {
        let (slot, l2_index) = self.get_cluster_table(offset)?;
        let nb_clusters = nb_clusters.min(self.l2_slice_size - l2_index as u64);
        for i in 0..nb_clusters as usize {
            let old_l2_entry = self.l2_get(slot, l2_index + i);
            let old_l2_bitmap = self.l2_get_bitmap(slot, l2_index + i);
            let ctype = self.cluster_type(old_l2_entry);
            let unmap = ctype == ClusterType::Compressed || (may_unmap && ctype.is_allocated());
            let keep_reference = self.discard_no_unref && ctype != ClusterType::Compressed;
            let mut new_l2_entry = old_l2_entry;
            let mut new_l2_bitmap = old_l2_bitmap;
            if unmap && !keep_reference {
                new_l2_entry = 0;
            }
            if self.has_subclusters() {
                new_l2_bitmap = QCOW_L2_BITMAP_ALL_ZEROES;
            } else {
                new_l2_entry |= QCOW_OFLAG_ZERO;
            }
            if old_l2_entry == new_l2_entry && old_l2_bitmap == new_l2_bitmap {
                continue;
            }
            self.l2_table_cache.mark_dirty(slot);
            self.l2_set(slot, l2_index + i, new_l2_entry);
            if self.has_subclusters() {
                self.l2_set_bitmap(slot, l2_index + i, new_l2_bitmap);
            }
            if unmap {
                if !keep_reference {
                    self.free_any_cluster(old_l2_entry, DiscardType::Request);
                } else {
                    self.discard_cluster(
                        old_l2_entry & L2E_OFFSET_MASK,
                        self.cluster_size,
                        ctype,
                        DiscardType::Request,
                    );
                }
            }
        }
        self.cache_put(CacheId::L2, slot);
        Ok(nb_clusters)
    }

    /// `zero_l2_subclusters()`: zeroes part of one cluster.
    fn zero_l2_subclusters(&mut self, offset: u64, nb_subclusters: u32) -> io::Result<()> {
        let sc = self.offset_to_sc_index(offset);
        assert!(nb_subclusters > 0 && nb_subclusters < self.subclusters_per_cluster);
        assert!(sc + nb_subclusters <= self.subclusters_per_cluster);
        assert_eq!(self.offset_into_subcluster(offset), 0);
        let (slot, l2_index) = self.get_cluster_table(offset)?;
        let r = match self.cluster_type(self.l2_get(slot, l2_index)) {
            // Compressed clusters cannot be zeroed in part.
            ClusterType::Compressed => Err(errno(libc::ENOTSUP)),
            ClusterType::Normal | ClusterType::Unallocated => {
                let old = self.l2_get_bitmap(slot, l2_index);
                let mut l2_bitmap = old;
                l2_bitmap |= sub_zero_range(sc, sc + nb_subclusters);
                l2_bitmap &= !sub_alloc_range(sc, sc + nb_subclusters);
                if old != l2_bitmap {
                    self.l2_set_bitmap(slot, l2_index, l2_bitmap);
                    self.l2_table_cache.mark_dirty(slot);
                }
                Ok(())
            }
            _ => unreachable!(),
        };
        self.cache_put(CacheId::L2, slot);
        r
    }

    /// `qcow2_subcluster_zeroize()`: makes the range read as zeroes through metadata alone.
    /// ENOTSUP means the caller has to write zeroes.
    pub(crate) fn subcluster_zeroize(
        &mut self,
        mut offset: u64,
        bytes: u64,
        may_unmap: bool,
    ) -> io::Result<()> {
        let mut end_offset = offset + bytes;
        // A raw data file has to stay in sync, so zero it first.
        if self.data_file_is_raw() {
            assert!(self.has_data_file());
            self.data().pwrite_zeroes(offset, bytes, may_unmap)?;
        }
        let disk_end = self.disk_size_sectors();
        assert_eq!(self.offset_into_subcluster(offset), 0);
        assert!(self.offset_into_subcluster(end_offset) == 0 || end_offset >= disk_end);

        // Only v3 has the zero flag; without a backing file v2 can deallocate instead.
        if self.qcow_version < 3 {
            if self.backing.is_none() {
                return self.cluster_discard(offset, bytes, DiscardType::Request, false);
            }
            return Err(errno(libc::ENOTSUP));
        }

        let head = end_offset.min(offset.next_multiple_of(self.cluster_size)) - offset;
        offset += head;
        let tail = if end_offset >= disk_end {
            0
        } else {
            end_offset - offset.max(self.start_of_cluster(end_offset))
        };
        end_offset -= tail;

        self.cache_discards = true;
        let r = (|| {
            if head != 0 {
                self.zero_l2_subclusters(offset - head, self.size_to_subclusters(head) as u32)?;
            }
            let mut nb_clusters = self.size_to_clusters(end_offset - offset);
            while nb_clusters > 0 {
                let cleared = self.zero_in_l2_slice(offset, nb_clusters, may_unmap)?;
                nb_clusters -= cleared;
                offset += cleared * self.cluster_size;
            }
            if tail != 0 {
                self.zero_l2_subclusters(end_offset, self.size_to_subclusters(tail) as u32)?;
            }
            Ok(())
        })();
        self.cache_discards = false;
        self.process_discards(r.is_ok());
        r
    }

    /// `expand_zero_clusters_in_l1()`: turns zero clusters into zero-filled data clusters, or
    /// drops plain ones when there is no backing file. `l1` is `None` for the active table.
    fn expand_zero_clusters_in_l1(
        &mut self,
        l1: Option<&[u64]>,
        visited: &mut u64,
        total: u64,
        status: &mut dyn FnMut(u64, u64),
    ) -> io::Result<()> {
        // Downgrading is refused for images with subclusters.
        assert!(!self.has_subclusters());
        let is_active = l1.is_none();
        let l1_table: Vec<u64> = match l1 {
            Some(t) => t.to_vec(),
            None => self.l1_table.clone(),
        };
        let slice_size2 = self.l2_slice_size * self.l2_entry_size();
        let n_slices = self.cluster_size / slice_size2;

        for (i, &l1e) in l1_table.iter().enumerate() {
            let l2_offset = l1e & L1E_OFFSET_MASK;
            if l2_offset == 0 {
                *visited += 1;
                status(*visited, total);
                continue;
            }
            if self.offset_into_cluster(l2_offset) != 0 {
                self.signal_corruption(
                    true,
                    -1,
                    -1,
                    &format!(
                        "L2 table offset {l2_offset:#x} unaligned (L1 index: {i:#x})",
                        l2_offset = Hx(l2_offset),
                        i = Hx(i)
                    ),
                );
                return Err(errno(libc::EIO));
            }
            let l2_refcount = self.get_refcount(l2_offset >> self.cluster_bits)?;

            for slice in 0..n_slices {
                let slice_offset = l2_offset + slice * slice_size2;
                let (slot, mut buf) = if is_active {
                    (Some(self.cache_get(CacheId::L2, slice_offset)?), Vec::new())
                } else {
                    let mut b = vec![0u8; slice_size2 as usize];
                    self.file.pread(slice_offset, &mut b)?;
                    (None, b)
                };
                let r = self.expand_zero_clusters_in_slice(
                    slot,
                    &mut buf,
                    l2_offset,
                    slice,
                    l2_refcount,
                );
                let l2_dirty = match (slot, r) {
                    (Some(s), r) => {
                        if let Ok(true) = r {
                            self.l2_table_cache.mark_dirty(s);
                            self.l2_table_cache.depends_on_flush();
                        }
                        self.cache_put(CacheId::L2, s);
                        r?;
                        false
                    }
                    (None, r) => r?,
                };
                if l2_dirty {
                    self.pre_write_overlap_check(
                        OL_INACTIVE_L2 | OL_ACTIVE_L2,
                        slice_offset,
                        slice_size2,
                        false,
                    )?;
                    self.file.pwrite(slice_offset, &buf)?;
                }
            }
            *visited += 1;
            status(*visited, total);
        }
        Ok(())
    }

    /// The per-slice part of [`State::expand_zero_clusters_in_l1`]. Works on the cache slot if
    /// there is one, else on `buf`. Returns whether anything changed.
    fn expand_zero_clusters_in_slice(
        &mut self,
        slot: Option<usize>,
        buf: &mut [u8],
        l2_offset: u64,
        slice: u64,
        l2_refcount: u64,
    ) -> io::Result<bool> {
        let stride = (self.l2_entry_size() / 8) as usize;
        let mut dirty = false;
        for j in 0..self.l2_slice_size as usize {
            let l2_entry = match slot {
                Some(s) => self.l2_get(s, j),
                None => get_be64(buf, j * stride),
            };
            let mut offset = l2_entry & L2E_OFFSET_MASK;
            let ctype = self.cluster_type(l2_entry);
            if !matches!(ctype, ClusterType::ZeroPlain | ClusterType::ZeroAlloc) {
                continue;
            }
            let set = |st: &mut State, buf: &mut [u8], v: u64| match slot {
                Some(s) => st.l2_set(s, j, v),
                None => set_be64(buf, j * stride, v),
            };
            if ctype == ClusterType::ZeroPlain {
                if self.backing.is_none() {
                    // Not backed, so the cluster can just go.
                    set(self, buf, 0);
                    dirty = true;
                    continue;
                }
                offset = self.alloc_clusters(self.cluster_size)?;
                assert_eq!(offset & L2E_OFFSET_MASK, offset);
                if l2_refcount > 1 {
                    // A shared L2 table: the refcount is 1 already and must be l2_refcount.
                    if let Err(e) = self.update_cluster_refcount(
                        offset >> self.cluster_bits,
                        l2_refcount - 1,
                        false,
                        DiscardType::Other,
                    ) {
                        self.free_clusters(offset, self.cluster_size, DiscardType::Other);
                        return Err(e);
                    }
                }
            }
            let fail = |st: &mut State| {
                if ctype == ClusterType::ZeroPlain {
                    st.free_clusters(offset, st.cluster_size, DiscardType::Always);
                }
            };
            if self.offset_into_cluster(offset) != 0 {
                let l2_index = slice * self.l2_slice_size + j as u64;
                self.signal_corruption(
                    true,
                    -1,
                    -1,
                    &format!(
                        "Cluster allocation offset {offset:#x} unaligned (L2 offset: {l2_offset:#x}, L2 \
                         index: {l2_index:#x})",
                        offset = Hx(offset),
                        l2_offset = Hx(l2_offset),
                        l2_index = Hx(l2_index)
                    ),
                );
                fail(self);
                return Err(errno(libc::EIO));
            }
            let r = self
                .pre_write_overlap_check(0, offset, self.cluster_size, true)
                .and_then(|()| self.data().pwrite_zeroes(offset, self.cluster_size, false));
            if let Err(e) = r {
                fail(self);
                return Err(e);
            }
            let v = if l2_refcount == 1 { offset | QCOW_OFLAG_COPIED } else { offset };
            set(self, buf, v);
            dirty = true;
        }
        Ok(dirty)
    }

    /// `qcow2_expand_zero_clusters()`: for downgrading to v2, which has no zero clusters.
    pub(crate) fn expand_zero_clusters(
        &mut self,
        status: &mut dyn FnMut(u64, u64),
    ) -> io::Result<()> {
        let total =
            self.l1_size as u64 + self.snapshots.iter().map(|s| s.l1_size as u64).sum::<u64>();
        let mut visited = 0;
        self.expand_zero_clusters_in_l1(None, &mut visited, total, status)?;

        // Inactive L1 tables can point to active L2 tables, so the cache must be written back
        // and emptied: the tables are about to change on disk behind its back.
        self.cache_empty(CacheId::L2)?;

        for i in 0..self.snapshots.len() {
            let (off, size) = (self.snapshots[i].l1_table_offset, self.snapshots[i].l1_size);
            if let Err((e, errnum)) = self.validate_table(
                off,
                size as u64,
                L1E_SIZE,
                QCOW_MAX_L1_SIZE,
                "Snapshot L1 table",
            ) {
                ruvm_base::report::report_error(&e);
                return Err(errno(errnum));
            }
            let mut buf = vec![0u8; size as usize * 8];
            self.file.pread(off, &mut buf)?;
            let l1: Vec<u64> =
                buf.chunks_exact(8).map(|c| u64::from_be_bytes(c.try_into().unwrap())).collect();
            self.expand_zero_clusters_in_l1(Some(&l1), &mut visited, total, status)?;
        }
        Ok(())
    }
}
