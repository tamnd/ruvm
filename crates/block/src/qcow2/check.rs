// SPDX-License-Identifier: GPL-2.0-or-later

//! `qemu-img check` for qcow2, from the checking half of block/qcow2-refcount.c and
//! `qcow2_co_check_locked()` in block/qcow2.c.
//!
//! The check builds an in-memory refcount table (IMRT) from everything that references
//! clusters, compares it with the refcounts on disk, and with repair enabled fixes leaks and
//! errors or rebuilds the whole refcount structure. The messages on stderr are QEMU's.

use std::io;

use ruvm_base::Error;
use ruvm_base::error::strerror;
use ruvm_base::report::report_error;

use super::cache::{CacheId, get_be64, set_be64};
use super::header::REFCOUNT_TABLE_OFFSET_OFFSET;
use super::refcount::{get_refcount, set_refcount};
use super::state::*;
use crate::node::errno;

/// `BDRV_FIX_LEAKS`: repair leaked clusters.
pub(crate) const FIX_LEAKS: u32 = 1;
/// `BDRV_FIX_ERRORS`: repair everything else too.
pub(crate) const FIX_ERRORS: u32 = 2;

/// `BlockFragInfo`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FragInfo {
    pub allocated_clusters: u64,
    pub total_clusters: u64,
    pub fragmented_clusters: u64,
    pub compressed_clusters: u64,
}

/// `BdrvCheckResult`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CheckResult {
    pub corruptions: i64,
    pub leaks: i64,
    pub check_errors: i64,
    pub corruptions_fixed: i64,
    pub leaks_fixed: i64,
    pub image_end_offset: i64,
    pub bfi: FragInfo,
}

impl CheckResult {
    /// `qcow2_add_check_result()`.
    fn add(&mut self, src: &CheckResult, set_allocation_info: bool) {
        self.corruptions += src.corruptions;
        self.leaks += src.leaks;
        self.check_errors += src.check_errors;
        self.corruptions_fixed += src.corruptions_fixed;
        self.leaks_fixed += src.leaks_fixed;
        if set_allocation_info {
            self.image_end_offset = src.image_end_offset;
            self.bfi = src.bfi;
        }
    }
}

/// The in-memory refcount table: one refblock covering the whole image, in the on-disk
/// refcount width so its slices can be written out as refblocks directly.
#[derive(Debug, Default)]
pub(crate) struct Imrt {
    pub data: Vec<u8>,
    pub nb_clusters: u64,
}

/// `CHECK_FRAG_INFO`.
const CHECK_FRAG_INFO: u32 = 2;

fn fix_word(fix: u32) -> &'static str {
    if fix & FIX_ERRORS != 0 { "Repairing" } else { "ERROR" }
}

impl State {
    /// `refcount_array_byte_size()`.
    fn refcount_array_byte_size(&self, entries: u64) -> u64 {
        assert!(entries < 1 << (64 - 9));
        (entries << self.refcount_order).div_ceil(8)
    }

    /// `realloc_refcount_array()`: the byte size stays a multiple of the cluster size.
    fn realloc_refcount_array(&self, imrt: &mut Imrt, new_size: u64) -> io::Result<()> {
        let old_bytes = self.size_to_clusters(self.refcount_array_byte_size(imrt.nb_clusters))
            * self.cluster_size;
        let new_bytes =
            self.size_to_clusters(self.refcount_array_byte_size(new_size)) * self.cluster_size;
        if new_bytes != old_bytes {
            assert!(new_bytes > 0);
            let new_bytes = usize::try_from(new_bytes).map_err(|_| errno(libc::ENOMEM))?;
            if new_bytes > imrt.data.len() {
                imrt.data
                    .try_reserve_exact(new_bytes - imrt.data.len())
                    .map_err(|_| errno(libc::ENOMEM))?;
            }
            imrt.data.resize(new_bytes, 0);
        }
        imrt.nb_clusters = new_size;
        Ok(())
    }

    fn imrt_get(&self, imrt: &Imrt, k: u64) -> u64 {
        get_refcount(self.refcount_order, &imrt.data, k)
    }

    fn imrt_set(&self, imrt: &mut Imrt, k: u64, v: u64) {
        set_refcount(self.refcount_order, &mut imrt.data, k, v)
    }

    /// `qcow2_inc_refcounts_imrt()`.
    pub(crate) fn inc_refcounts_imrt(
        &mut self,
        res: &mut CheckResult,
        imrt: &mut Imrt,
        offset: u64,
        size: u64,
    ) -> io::Result<()> {
        if size == 0 || size > i64::MAX as u64 {
            return Ok(());
        }
        let file_len = self.file.len()?;
        // The last cluster may be only partly there, so up to a cluster past the end is fine.
        if (offset + size) as i64 - file_len as i64 >= self.cluster_size as i64 {
            eprintln!(
                "ERROR: counting reference for region exceeding the end of the file by one \
                 cluster or more: offset 0x{offset:x} size 0x{size:x}"
            );
            res.corruptions += 1;
            return Ok(());
        }
        let start = self.start_of_cluster(offset);
        let last = self.start_of_cluster(offset + size - 1);
        let mut cluster_offset = start;
        while cluster_offset <= last {
            let k = cluster_offset >> self.cluster_bits;
            if k >= imrt.nb_clusters {
                if let Err(e) = self.realloc_refcount_array(imrt, k + 1) {
                    res.check_errors += 1;
                    return Err(e);
                }
            }
            let refcount = self.imrt_get(imrt, k);
            if refcount == self.refcount_max {
                eprintln!("ERROR: overflow cluster offset=0x{cluster_offset:x}");
                eprintln!(
                    "Use qemu-img amend to increase the refcount entry width or qemu-img convert \
                     to create a clean copy if the image cannot be opened for writing"
                );
                res.corruptions += 1;
            } else {
                self.imrt_set(imrt, k, refcount + 1);
            }
            cluster_offset += self.cluster_size;
        }
        Ok(())
    }

    /// `fix_l2_entry_by_zero()`: the caller counted the corruption already. The second value
    /// tells whether the overlap check failed.
    fn fix_l2_entry_by_zero(
        &mut self,
        res: &mut CheckResult,
        l2_offset: u64,
        l2_table: &mut [u8],
        l2_index: usize,
        active: bool,
    ) -> (io::Result<()>, bool) {
        let esize = self.l2_entry_size();
        let idx = l2_index * (esize / 8) as usize;
        let l2e_offset = l2_offset + l2_index as u64 * esize;
        let ign = if active { OL_ACTIVE_L2 } else { OL_INACTIVE_L2 };
        if self.has_subclusters() {
            let mut l2_bitmap = self.l2_bitmap(l2_table, l2_index);
            // Allocated subclusters become zero ones.
            l2_bitmap |= l2_bitmap << 32;
            l2_bitmap &= QCOW_L2_BITMAP_ALL_ZEROES;
            self.set_l2_bitmap(l2_table, l2_index, l2_bitmap);
            self.set_l2_entry(l2_table, l2_index, 0);
        } else {
            self.set_l2_entry(l2_table, l2_index, QCOW_OFLAG_ZERO);
        }
        if let Err(e) = self.pre_write_overlap_check(ign, l2e_offset, esize, false) {
            eprintln!("ERROR: Overlap check failed");
            res.check_errors += 1;
            return (Err(e), true);
        }
        let bytes = &l2_table[idx * 8..idx * 8 + esize as usize];
        if let Err(e) = self.file.pwrite(l2e_offset, bytes).and_then(|()| self.file.flush()) {
            eprintln!("ERROR: Failed to overwrite L2 table entry: {}", strerror(&e));
            res.check_errors += 1;
            return (Err(e), false);
        }
        res.corruptions -= 1;
        res.corruptions_fixed += 1;
        (Ok(()), false)
    }

    /// `check_refcounts_l2()`.
    #[allow(clippy::too_many_arguments)]
    fn check_refcounts_l2(
        &mut self,
        res: &mut CheckResult,
        imrt: &mut Imrt,
        l2_offset: u64,
        flags: u32,
        fix: u32,
        active: bool,
    ) -> io::Result<()> {
        let l2_size_bytes = (self.l2_size * self.l2_entry_size()) as usize;
        let mut l2_table = vec![0u8; l2_size_bytes];
        if let Err(e) = self.file.pread(l2_offset, &mut l2_table) {
            eprintln!("ERROR: I/O error in check_refcounts_l2");
            res.check_errors += 1;
            return Err(e);
        }
        let mut next_contiguous_offset = 0u64;
        for i in 0..self.l2_size as usize {
            let mut l2_entry = self.l2_entry(&l2_table, i);
            let l2_bitmap = self.l2_bitmap(&l2_table, i);
            let ty = self.cluster_type(l2_entry);

            // Reserved bits of a standard cluster descriptor.
            if ty != ClusterType::Compressed && l2_entry & L2E_STD_RESERVED_MASK != 0 {
                eprintln!("ERROR found l2 entry with reserved bits set: {l2_entry:x}");
                res.corruptions += 1;
            }

            match ty {
                ClusterType::Compressed => {
                    if l2_entry & QCOW_OFLAG_COPIED != 0 {
                        eprintln!(
                            "ERROR: coffset=0x{:x}: copied flag must never be set for compressed \
                             clusters",
                            l2_entry & self.cluster_offset_mask
                        );
                        l2_entry &= !QCOW_OFLAG_COPIED;
                        res.corruptions += 1;
                    }
                    if self.has_data_file() {
                        eprintln!(
                            "ERROR compressed cluster {i} with data file, entry=0x{l2_entry:x}"
                        );
                        res.corruptions += 1;
                        continue;
                    }
                    if l2_bitmap != 0 {
                        eprintln!(
                            "ERROR compressed cluster {i} with non-zero subcluster allocation \
                             bitmap, entry=0x{l2_entry:x}"
                        );
                        res.corruptions += 1;
                        continue;
                    }
                    let (coffset, csize) = self.parse_compressed_l2_entry(l2_entry);
                    self.inc_refcounts_imrt(res, imrt, coffset, csize)?;
                    if flags & CHECK_FRAG_INFO != 0 {
                        res.bfi.allocated_clusters += 1;
                        res.bfi.compressed_clusters += 1;
                        // Compressed clusters always count as fragmented: neighbours share
                        // sectors, which have to be read again.
                        res.bfi.fragmented_clusters += 1;
                    }
                }
                ClusterType::ZeroAlloc | ClusterType::Normal => {
                    let offset = l2_entry & L2E_OFFSET_MASK;
                    if (l2_bitmap >> 32) & l2_bitmap != 0 {
                        res.corruptions += 1;
                        eprintln!(
                            "ERROR offset={offset:x}: Allocated cluster has corrupted subcluster \
                             allocation bitmap"
                        );
                    }
                    if self.offset_into_cluster(offset) != 0 {
                        res.corruptions += 1;
                        let contains_data = if self.has_subclusters() {
                            l2_bitmap & QCOW_L2_BITMAP_ALL_ALLOC != 0
                        } else {
                            l2_entry & QCOW_OFLAG_ZERO == 0
                        };
                        if !contains_data {
                            eprintln!(
                                "{} offset={offset:x}: Preallocated cluster is not properly \
                                 aligned; L2 entry corrupted.",
                                fix_word(fix)
                            );
                            if fix & FIX_ERRORS != 0 {
                                let (r, overlap) = self.fix_l2_entry_by_zero(
                                    res,
                                    l2_offset,
                                    &mut l2_table,
                                    i,
                                    active,
                                );
                                if overlap {
                                    // Something is badly wrong: stop with this table.
                                    return r;
                                }
                                if r.is_ok() {
                                    // The cluster is unused now.
                                    continue;
                                }
                                // Could not fix it; go on with the other entries.
                            }
                        } else {
                            eprintln!(
                                "ERROR offset={offset:x}: Data cluster is not properly aligned; L2 \
                                 entry corrupted."
                            );
                        }
                    }
                    if flags & CHECK_FRAG_INFO != 0 {
                        res.bfi.allocated_clusters += 1;
                        if next_contiguous_offset != 0 && offset != next_contiguous_offset {
                            res.bfi.fragmented_clusters += 1;
                        }
                        next_contiguous_offset = offset + self.cluster_size;
                    }
                    if !self.has_data_file() {
                        self.inc_refcounts_imrt(res, imrt, offset, self.cluster_size)?;
                    }
                }
                ClusterType::ZeroPlain => {
                    // Cannot happen with subclusters.
                    assert_eq!(l2_bitmap, 0);
                }
                ClusterType::Unallocated => {
                    if l2_bitmap & QCOW_L2_BITMAP_ALL_ALLOC != 0 {
                        res.corruptions += 1;
                        eprintln!(
                            "ERROR: Unallocated cluster has non-zero subcluster allocation map"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// `check_refcounts_l1()`.
    #[allow(clippy::too_many_arguments)]
    fn check_refcounts_l1(
        &mut self,
        res: &mut CheckResult,
        imrt: &mut Imrt,
        l1_table_offset: u64,
        l1_size: u32,
        flags: u32,
        fix: u32,
        active: bool,
    ) -> io::Result<()> {
        if l1_size == 0 {
            return Ok(());
        }
        let l1_size_bytes = l1_size as u64 * L1E_SIZE;
        self.inc_refcounts_imrt(res, imrt, l1_table_offset, l1_size_bytes)?;

        let mut buf = Vec::new();
        if buf.try_reserve_exact(l1_size_bytes as usize).is_err() {
            res.check_errors += 1;
            return Err(errno(libc::ENOMEM));
        }
        buf.resize(l1_size_bytes as usize, 0);
        if let Err(e) = self.file.pread(l1_table_offset, &mut buf) {
            eprintln!("ERROR: I/O error in check_refcounts_l1");
            res.check_errors += 1;
            return Err(e);
        }
        for i in 0..l1_size as usize {
            let l1e = get_be64(&buf, i);
            if l1e == 0 {
                continue;
            }
            if l1e & L1E_RESERVED_MASK != 0 {
                eprintln!("ERROR found L1 entry with reserved bits set: {l1e:x}");
                res.corruptions += 1;
            }
            let l2_offset = l1e & L1E_OFFSET_MASK;
            self.inc_refcounts_imrt(res, imrt, l2_offset, self.cluster_size)?;
            if self.offset_into_cluster(l2_offset) != 0 {
                eprintln!(
                    "ERROR l2_offset={l2_offset:x}: Table is not cluster aligned; L1 entry corrupted"
                );
                res.corruptions += 1;
            }
            self.check_refcounts_l2(res, imrt, l2_offset, flags, fix, active)?;
        }
        Ok(())
    }

    /// `check_oflag_copied()`: the COPIED flags of the active L1 and L2 tables. Refcount read
    /// errors were reported by the refcount check already, so they are skipped quietly here.
    fn check_oflag_copied(&mut self, res: &mut CheckResult, fix: u32) -> io::Result<()> {
        let repair = if fix & FIX_ERRORS != 0 {
            true
        } else if fix & FIX_LEAKS != 0 {
            // Safe only if the refcounts are right, which they are if their repair worked.
            res.check_errors == 0 && res.corruptions == 0 && res.leaks == 0
        } else {
            false
        };
        let word = if repair { "Repairing" } else { "ERROR" };
        let mut l2_table = vec![0u8; self.cluster_size as usize];

        for i in 0..self.l1_size as usize {
            let l1_entry = self.l1_table[i];
            let l2_offset = l1_entry & L1E_OFFSET_MASK;
            if l2_offset == 0 {
                continue;
            }
            let Ok(refcount) = self.get_refcount(l2_offset >> self.cluster_bits) else {
                continue;
            };
            if (refcount == 1) != (l1_entry & QCOW_OFLAG_COPIED != 0) {
                res.corruptions += 1;
                eprintln!(
                    "{word} OFLAG_COPIED L2 cluster: l1_index={i} l1_entry={l1_entry:x} refcount={refcount}"
                );
                if repair {
                    self.l1_table[i] = if refcount == 1 {
                        l1_entry | QCOW_OFLAG_COPIED
                    } else {
                        l1_entry & !QCOW_OFLAG_COPIED
                    };
                    if let Err(e) = self.write_l1_entry(i) {
                        res.check_errors += 1;
                        return Err(e);
                    }
                    res.corruptions -= 1;
                    res.corruptions_fixed += 1;
                }
            }

            let n = (self.l2_size * self.l2_entry_size()) as usize;
            if let Err(e) = self.file.pread(l2_offset, &mut l2_table[..n]) {
                eprintln!("ERROR: Could not read L2 table: {}", strerror(&e));
                res.check_errors += 1;
                return Err(e);
            }
            let mut l2_dirty = 0i64;
            for j in 0..self.l2_size as usize {
                let l2_entry = self.l2_entry(&l2_table, j);
                let data_offset = l2_entry & L2E_OFFSET_MASK;
                let ctype = self.cluster_type(l2_entry);
                if !matches!(ctype, ClusterType::Normal | ClusterType::ZeroAlloc) {
                    continue;
                }
                let refcount = if self.has_data_file() {
                    1
                } else {
                    match self.get_refcount(data_offset >> self.cluster_bits) {
                        Ok(r) => r,
                        Err(_) => continue,
                    }
                };
                if (refcount == 1) != (l2_entry & QCOW_OFLAG_COPIED != 0) {
                    res.corruptions += 1;
                    eprintln!(
                        "{word} OFLAG_COPIED data cluster: l2_entry={l2_entry:x} refcount={refcount}"
                    );
                    if repair {
                        let v = if refcount == 1 {
                            l2_entry | QCOW_OFLAG_COPIED
                        } else {
                            l2_entry & !QCOW_OFLAG_COPIED
                        };
                        self.set_l2_entry(&mut l2_table, j, v);
                        l2_dirty += 1;
                    }
                }
            }
            if l2_dirty > 0 {
                if let Err(e) =
                    self.pre_write_overlap_check(OL_ACTIVE_L2, l2_offset, self.cluster_size, false)
                {
                    eprintln!(
                        "ERROR: Could not write L2 table; metadata overlap check failed: {}",
                        strerror(&e)
                    );
                    res.check_errors += 1;
                    return Err(e);
                }
                if let Err(e) = self.file.pwrite(l2_offset, &l2_table) {
                    eprintln!("ERROR: Could not write L2 table: {}", strerror(&e));
                    res.check_errors += 1;
                    return Err(e);
                }
                // The cached copy, if any, is stale now.
                if let Some(t) = self.l2_table_cache.is_table_offset(l2_offset) {
                    self.l2_table_cache.discard(t);
                }
                res.corruptions -= l2_dirty;
                res.corruptions_fixed += l2_dirty;
            }
        }
        Ok(())
    }

    /// `check_refblocks()`.
    fn check_refblocks(
        &mut self,
        res: &mut CheckResult,
        fix: u32,
        rebuild: &mut bool,
        imrt: &mut Imrt,
    ) -> io::Result<()> {
        for i in 0..self.refcount_table.len() {
            let offset = self.refcount_table[i] & REFT_OFFSET_MASK;
            let cluster = offset >> self.cluster_bits;

            if self.refcount_table[i] & REFT_RESERVED_MASK != 0 {
                eprintln!("ERROR refcount table entry {i} has reserved bits set");
                res.corruptions += 1;
                *rebuild = true;
                continue;
            }
            if self.offset_into_cluster(offset) != 0 {
                eprintln!(
                    "ERROR refcount block {i} is not cluster aligned; refcount table entry corrupted"
                );
                res.corruptions += 1;
                *rebuild = true;
                continue;
            }
            if cluster >= imrt.nb_clusters {
                res.corruptions += 1;
                eprintln!("{} refcount block {i} is outside image", fix_word(fix));
                if fix & FIX_ERRORS != 0 {
                    let resized: io::Result<u64> = if offset > i64::MAX as u64 - self.cluster_size {
                        Err(errno(libc::EINVAL))
                    } else {
                        match self.file.truncate(offset + self.cluster_size) {
                            Ok(()) => self.file.len(),
                            Err(e) => {
                                report_error(&e);
                                Err(errno(libc::EIO))
                            }
                        }
                    };
                    let resized = resized.and_then(|size| {
                        let new_nb_clusters = self.size_to_clusters(size);
                        assert!(new_nb_clusters >= imrt.nb_clusters);
                        if let Err(e) = self.realloc_refcount_array(imrt, new_nb_clusters) {
                            res.check_errors += 1;
                            return Ok(Err(e));
                        }
                        if cluster >= imrt.nb_clusters {
                            return Err(errno(libc::EINVAL));
                        }
                        Ok(Ok(()))
                    });
                    match resized {
                        Ok(Err(e)) => return Err(e),
                        Ok(Ok(())) => {
                            res.corruptions -= 1;
                            res.corruptions_fixed += 1;
                            // The area was just allocated and zeroed, so its refcount can only
                            // come out as exactly 1.
                            self.inc_refcounts_imrt(res, imrt, offset, self.cluster_size)?;
                            continue;
                        }
                        Err(e) => {
                            *rebuild = true;
                            eprintln!("ERROR could not resize image: {}", strerror(&e));
                        }
                    }
                }
                continue;
            }
            if offset != 0 {
                self.inc_refcounts_imrt(res, imrt, offset, self.cluster_size)?;
                let rc = self.imrt_get(imrt, cluster);
                if rc != 1 {
                    eprintln!("ERROR refcount block {i} refcount={rc}");
                    res.corruptions += 1;
                    *rebuild = true;
                }
            }
        }
        Ok(())
    }

    /// `calculate_refcounts()`.
    fn calculate_refcounts(
        &mut self,
        res: &mut CheckResult,
        fix: u32,
        rebuild: &mut bool,
        imrt: &mut Imrt,
    ) -> io::Result<()> {
        if imrt.data.is_empty() {
            let n = imrt.nb_clusters;
            imrt.nb_clusters = 0;
            if let Err(e) = self.realloc_refcount_array(imrt, n) {
                res.check_errors += 1;
                return Err(e);
            }
        }
        // The header.
        self.inc_refcounts_imrt(res, imrt, 0, self.cluster_size)?;
        // The active L1 table.
        let (off, size) = (self.l1_table_offset, self.l1_size);
        self.check_refcounts_l1(res, imrt, off, size, CHECK_FRAG_INFO, fix, true)?;

        // Snapshots.
        if self.has_data_file() && !self.snapshots.is_empty() {
            eprintln!("ERROR {} snapshots in image with data file", self.snapshots.len());
            res.corruptions += 1;
        }
        for i in 0..self.snapshots.len() {
            let sn = &self.snapshots[i];
            let (off, size) = (sn.l1_table_offset, sn.l1_size);
            if self.offset_into_cluster(off) != 0 {
                eprintln!(
                    "ERROR snapshot {} ({}) l1_offset={off:#x}: L1 table is not cluster aligned; \
                     snapshot table entry corrupted",
                    sn.id_str,
                    sn.name,
                    off = Hx(off)
                );
                res.corruptions += 1;
                continue;
            }
            if size as u64 > QCOW_MAX_L1_SIZE / L1E_SIZE {
                eprintln!(
                    "ERROR snapshot {} ({}) l1_size={size:#x}: L1 table is too large; snapshot \
                     table entry corrupted",
                    sn.id_str,
                    sn.name,
                    size = Hx(size)
                );
                res.corruptions += 1;
                continue;
            }
            self.check_refcounts_l1(res, imrt, off, size, 0, fix, false)?;
        }
        let (so, ss) = (self.snapshots_offset, self.snapshots_size);
        self.inc_refcounts_imrt(res, imrt, so, ss)?;

        // Refcount structures.
        let (ro, rs) =
            (self.refcount_table_offset, self.refcount_table.len() as u64 * REFTABLE_ENTRY_SIZE);
        self.inc_refcounts_imrt(res, imrt, ro, rs)?;

        // Encryption header.
        if self.crypto_header.1 != 0 {
            let (co, cl) = self.crypto_header;
            self.inc_refcounts_imrt(res, imrt, co, cl)?;
        }

        // Bitmaps.
        self.check_bitmaps_refcounts(res, imrt)?;

        self.check_refblocks(res, fix, rebuild, imrt)
    }

    /// `compare_refcounts()`. Returns the highest cluster in use.
    fn compare_refcounts(
        &mut self,
        res: &mut CheckResult,
        fix: u32,
        rebuild: &mut bool,
        imrt: &Imrt,
    ) -> u64 {
        let mut highest_cluster = 0;
        for i in 0..imrt.nb_clusters {
            let refcount1 = match self.get_refcount(i) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Can't get refcount for cluster {i}: {}", strerror(&e));
                    res.check_errors += 1;
                    continue;
                }
            };
            let refcount2 = self.imrt_get(imrt, i);
            if refcount1 > 0 || refcount2 > 0 {
                highest_cluster = i;
            }
            if refcount1 == refcount2 {
                continue;
            }
            // Is this one allowed to be fixed?
            #[derive(PartialEq)]
            enum Fixed {
                No,
                Leak,
                Corruption,
            }
            let mut num_fixed = Fixed::No;
            if refcount1 == 0 {
                *rebuild = true;
            } else if refcount1 > refcount2 && fix & FIX_LEAKS != 0 {
                num_fixed = Fixed::Leak;
            } else if refcount1 < refcount2 && fix & FIX_ERRORS != 0 {
                num_fixed = Fixed::Corruption;
            }
            let word = if num_fixed != Fixed::No {
                "Repairing"
            } else if refcount1 < refcount2 {
                "ERROR"
            } else {
                "Leaked"
            };
            eprintln!("{word} cluster {i} refcount={refcount1} reference={refcount2}");

            if num_fixed != Fixed::No {
                let diff = refcount1.abs_diff(refcount2);
                if self
                    .update_refcount(
                        i << self.cluster_bits,
                        1,
                        diff,
                        refcount1 > refcount2,
                        DiscardType::Always,
                    )
                    .is_ok()
                {
                    match num_fixed {
                        Fixed::Leak => res.leaks_fixed += 1,
                        _ => res.corruptions_fixed += 1,
                    }
                    continue;
                }
            }
            if refcount1 < refcount2 {
                res.corruptions += 1;
            } else {
                res.leaks += 1;
            }
        }
        highest_cluster
    }

    /// `alloc_clusters_imrt()`: allocates in the IMRT rather than on disk.
    fn alloc_clusters_imrt(
        &self,
        cluster_count: u64,
        imrt: &mut Imrt,
        first_free_cluster: &mut u64,
    ) -> io::Result<u64> {
        let mut cluster = *first_free_cluster;
        let mut first_gap = true;
        let mut contiguous_free_clusters = 0u64;
        while cluster < imrt.nb_clusters && contiguous_free_clusters < cluster_count {
            if self.imrt_get(imrt, cluster) == 0 {
                contiguous_free_clusters += 1;
                if first_gap {
                    *first_free_cluster = cluster;
                    first_gap = false;
                }
            } else if contiguous_free_clusters != 0 {
                contiguous_free_clusters = 0;
            }
            cluster += 1;
        }
        // Not enough room: grow the table to append free clusters at the end.
        if contiguous_free_clusters < cluster_count {
            self.realloc_refcount_array(imrt, cluster + cluster_count - contiguous_free_clusters)?;
        }
        cluster -= contiguous_free_clusters;
        for i in 0..cluster_count {
            self.imrt_set(imrt, cluster + i, 1);
        }
        Ok(cluster << self.cluster_bits)
    }

    /// `rebuild_refcounts_write_refblocks()`: writes the refblocks for the allocated clusters
    /// in `[first_cluster, end_cluster)`. Returns whether the reftable had to grow.
    fn rebuild_refcounts_write_refblocks(
        &mut self,
        imrt: &mut Imrt,
        first_cluster: u64,
        mut end_cluster: u64,
        reftable: &mut Vec<u64>,
    ) -> ruvm_base::Result<bool> {
        let mut first_free_cluster = 0u64;
        let mut reftable_grown = false;
        let mut cluster = first_cluster;
        while cluster < end_cluster {
            if self.imrt_get(imrt, cluster) == 0 {
                cluster += 1;
                continue;
            }
            // The refblock is the slice of the IMRT that covers this cluster, so it holds the
            // right refcounts for all of its clusters.
            let refblock_index = cluster >> self.refcount_block_bits;
            let refblock_start = refblock_index << self.refcount_block_bits;
            let refblock_offset;
            if (reftable.len() as u64) > refblock_index && reftable[refblock_index as usize] != 0 {
                // Allocated in an earlier round.
                refblock_offset = reftable[refblock_index as usize];
            } else {
                // Keep out of refblocks that were written already.
                if first_free_cluster < refblock_start {
                    first_free_cluster = refblock_start;
                }
                refblock_offset = self
                    .alloc_clusters_imrt(1, imrt, &mut first_free_cluster)
                    .map_err(|e| Error::from_io("ERROR allocating refblock", e))?;
                let refblock_cluster_index = refblock_offset / self.cluster_size;
                if refblock_cluster_index >= end_cluster {
                    // The refblock with this refblock's own refcount must be written too.
                    end_cluster = refblock_cluster_index + 1;
                }
                if reftable.len() as u64 <= refblock_index {
                    let n = ((refblock_index + 1) * REFTABLE_ENTRY_SIZE)
                        .next_multiple_of(self.cluster_size)
                        / REFTABLE_ENTRY_SIZE;
                    if reftable.try_reserve_exact(n as usize - reftable.len()).is_err() {
                        return Err(Error::generic("ERROR allocating reftable memory"));
                    }
                    reftable.resize(n as usize, 0);
                    reftable_grown = true;
                }
                reftable[refblock_index as usize] = refblock_offset;
            }

            self.pre_write_overlap_check(0, refblock_offset, self.cluster_size, false)
                .map_err(|e| Error::from_io("ERROR writing refblock", e))?;
            let start = (refblock_index * self.cluster_size) as usize;
            let block = &imrt.data[start..start + self.cluster_size as usize];
            self.file
                .pwrite(refblock_offset, block)
                .map_err(|e| Error::from_io("ERROR writing refblock", e))?;
            // This refblock is done: skip to its end.
            cluster = refblock_start + self.refcount_block_size;
        }
        Ok(reftable_grown)
    }

    /// `rebuild_refcount_structure()`: builds new refcount structures from the IMRT alone. The
    /// old ones are leaked, which the second check run cleans up.
    fn rebuild_refcount_structure(
        &mut self,
        res: &mut CheckResult,
        imrt: &mut Imrt,
    ) -> ruvm_base::Result<()> {
        let _ = self.cache_empty(CacheId::Refcount);
        let mut reftable: Vec<u64> = Vec::new();

        let n = imrt.nb_clusters;
        let changed = match self.rebuild_refcounts_write_refblocks(imrt, 0, n, &mut reftable) {
            Ok(c) => c,
            Err(e) => {
                res.check_errors += 1;
                return Err(e);
            }
        };
        // There was no reftable, so it must have grown from nothing.
        assert!(changed);

        // Placing the reftable changes the IMRT, so the refblocks covering it are written
        // again, which may make the reftable grow again. Each round covers far more clusters
        // than the table grows by, so this ends.
        let mut reftable_offset;
        let mut reftable_clusters;
        loop {
            let mut first_free_cluster = 0;
            let reftable_length = reftable.len() as u64 * REFTABLE_ENTRY_SIZE;
            reftable_clusters = self.size_to_clusters(reftable_length);
            reftable_offset =
                match self.alloc_clusters_imrt(reftable_clusters, imrt, &mut first_free_cluster) {
                    Ok(o) => o,
                    Err(e) => {
                        res.check_errors += 1;
                        return Err(Error::from_io("ERROR allocating reftable", e));
                    }
                };
            assert_eq!(self.offset_into_cluster(reftable_offset), 0);
            let start = reftable_offset / self.cluster_size;
            match self.rebuild_refcounts_write_refblocks(
                imrt,
                start,
                start + reftable_clusters,
                &mut reftable,
            ) {
                Ok(true) => continue,
                Ok(false) => break,
                Err(e) => {
                    res.check_errors += 1;
                    return Err(e);
                }
            }
        }

        let reftable_length = reftable.len() as u64 * REFTABLE_ENTRY_SIZE;
        let mut buf = vec![0u8; reftable_length as usize];
        for (i, v) in reftable.iter().enumerate() {
            set_be64(&mut buf, i, *v);
        }
        self.pre_write_overlap_check(0, reftable_offset, reftable_length, false)
            .map_err(|e| Error::from_io("ERROR writing reftable", e))?;
        self.file
            .pwrite(reftable_offset, &buf)
            .map_err(|e| Error::from_io("ERROR writing reftable", e))?;

        // Point the header at the new reftable.
        let mut hdr = [0u8; 12];
        hdr[..8].copy_from_slice(&reftable_offset.to_be_bytes());
        hdr[8..].copy_from_slice(&(reftable_clusters as u32).to_be_bytes());
        self.file
            .pwrite(REFCOUNT_TABLE_OFFSET_OFFSET, &hdr)
            .and_then(|()| self.file.flush())
            .map_err(|e| Error::from_io("ERROR setting reftable", e))?;

        self.refcount_table = reftable;
        self.refcount_table_offset = reftable_offset;
        self.update_max_refcount_table_index();
        Ok(())
    }

    /// `qcow2_check_refcounts()`.
    pub(crate) fn check_refcounts(&mut self, res: &mut CheckResult, fix: u32) -> io::Result<()> {
        let size = match self.file.len() {
            Ok(s) => s,
            Err(e) => {
                res.check_errors += 1;
                return Err(e);
            }
        };
        let nb_clusters = self.size_to_clusters(size);
        if nb_clusters > i32::MAX as u64 {
            res.check_errors += 1;
            return Err(errno(libc::EFBIG));
        }
        res.bfi.total_clusters = self.size_to_clusters(self.total_size.next_multiple_of(512));

        let mut imrt = Imrt { data: Vec::new(), nb_clusters };
        let mut rebuild = false;
        self.calculate_refcounts(res, fix, &mut rebuild, &mut imrt)?;

        // Without a rebuild but with something to fix, the comparison runs again below and
        // this result is thrown away.
        let pre_compare_res = *res;
        let mut highest_cluster = self.compare_refcounts(res, 0, &mut rebuild, &imrt);

        if rebuild && fix & FIX_ERRORS != 0 {
            let old_res = *res;
            let mut fresh_leaks = 0;
            eprintln!("Rebuilding refcount structure");
            if let Err(e) = self.rebuild_refcount_structure(res, &mut imrt) {
                report_error(&e);
                return Err(errno(libc::EIO));
            }
            res.corruptions = 0;
            res.leaks = 0;

            // The reftable changed, so count the references again.
            rebuild = false;
            let bytes = self.refcount_array_byte_size(imrt.nb_clusters) as usize;
            imrt.data[..bytes].fill(0);
            self.calculate_refcounts(res, 0, &mut rebuild, &mut imrt)?;

            if fix & FIX_LEAKS != 0 {
                // The old refcount structures are leaked now. Fix that; only leaks the rebuild
                // itself caused and that could not be fixed matter.
                let saved_res = *res;
                *res = CheckResult::default();
                highest_cluster = self.compare_refcounts(res, FIX_LEAKS, &mut rebuild, &imrt);
                if rebuild {
                    eprintln!("ERROR rebuilt refcount structure is still broken");
                }
                fresh_leaks = res.leaks;
                *res = saved_res;
            }
            if res.corruptions < old_res.corruptions {
                res.corruptions_fixed += old_res.corruptions - res.corruptions;
            }
            if res.leaks < old_res.leaks {
                res.leaks_fixed += old_res.leaks - res.leaks;
            }
            res.leaks += fresh_leaks;
        } else if fix != 0 {
            if rebuild {
                eprintln!("ERROR need to rebuild refcount structures");
                res.check_errors += 1;
                return Err(errno(libc::EIO));
            }
            if res.leaks != 0 || res.corruptions != 0 {
                *res = pre_compare_res;
                highest_cluster = self.compare_refcounts(res, fix, &mut rebuild, &imrt);
            }
        }

        self.check_oflag_copied(res, fix)?;
        res.image_end_offset = ((highest_cluster + 1) * self.cluster_size) as i64;
        Ok(())
    }

    /// `qcow2_co_check_locked()`. The result is filled in as far as the check got even when it
    /// fails.
    pub(crate) fn check(&mut self, result: &mut CheckResult, fix: u32) -> io::Result<()> {
        *result = CheckResult::default();
        let mut snapshot_res = CheckResult::default();
        let mut refcount_res = CheckResult::default();

        if let Err(e) = self.check_read_snapshot_table(&mut snapshot_res, fix) {
            result.add(&snapshot_res, false);
            return Err(e);
        }
        let r = self.check_refcounts(&mut refcount_res, fix);
        result.add(&refcount_res, true);
        if let Err(e) = r {
            result.add(&snapshot_res, false);
            return Err(e);
        }
        let r = self.check_fix_snapshot_table(&mut snapshot_res, fix);
        result.add(&snapshot_res, false);
        r?;

        if fix != 0 && result.check_errors == 0 && result.corruptions == 0 {
            self.mark_clean()?;
            return self.mark_consistent();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_keeps_allocation_info_only_when_asked() {
        let mut a = CheckResult::default();
        let b =
            CheckResult { corruptions: 2, leaks: 1, image_end_offset: 100, ..Default::default() };
        a.add(&b, false);
        assert_eq!((a.corruptions, a.leaks, a.image_end_offset), (2, 1, 0));
        a.add(&b, true);
        assert_eq!((a.corruptions, a.leaks, a.image_end_offset), (4, 2, 100));
    }
}
