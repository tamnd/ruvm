// SPDX-License-Identifier: GPL-2.0-or-later

//! Resizing and emptying an image: `qcow2_co_truncate()`, `preallocate_co()` and
//! `qcow2_make_empty()` from block/qcow2.c.

use ruvm_base::report::warn_report;
use ruvm_base::{Error, Result};

use super::cluster::L2Meta;
use super::header::{L1_SIZE_OFFSET, SIZE_OFFSET};
use super::io::{Storage, write_zero_buffer};
use super::state::*;
use crate::node::errno;

/// The preallocation modes, QAPI's `PreallocMode`.
pub(crate) use ruvm_qapi::types::PreallocMode as Prealloc;

/// `qapi_enum_parse(&PreallocMode_lookup, ...)` with QEMU's message.
pub(crate) fn parse_prealloc(s: &str) -> Result<Prealloc> {
    match s {
        "off" => Ok(Prealloc::Off),
        "metadata" => Ok(Prealloc::Metadata),
        "falloc" => Ok(Prealloc::Falloc),
        "full" => Ok(Prealloc::Full),
        _ => Err(Error::generic(format!("Invalid parameter '{s}'"))),
    }
}

/// `bdrv_co_truncate()` on a file below qcow2. `falloc` makes the file as `off` does: there
/// is no portable way to reserve space without writing it. `full` writes zeroes.
pub(crate) fn storage_truncate(s: &dyn Storage, len: u64, prealloc: Prealloc) -> Result<()> {
    let old = s.len().map_err(|e| Error::from_io("Failed to get file length", e))?;
    s.truncate(len)?;
    if prealloc == Prealloc::Full && len > old {
        write_zero_buffer(s, old, len - old)
            .map_err(|e| Error::from_io("Could not write zeros for preallocation", e))?;
    }
    Ok(())
}

impl State {
    /// `preallocate_co()`: allocates clusters for `[offset, new_length)` and maps them.
    pub(crate) fn preallocate(
        &mut self,
        mut offset: u64,
        new_length: u64,
        mut mode: Prealloc,
    ) -> Result<()> {
        assert!(offset <= new_length);
        let mut bytes = new_length - offset;
        let mut host_offset = 0;
        let mut cur_bytes = 0;
        while bytes > 0 {
            cur_bytes = bytes.min((i32::MAX as u64) & !(self.cluster_size - 1));
            let mut metas: Vec<L2Meta> = Vec::new();
            host_offset = match self.alloc_host_offset(offset, &mut cur_bytes, &mut metas) {
                Ok(h) => h,
                Err(e) => {
                    for m in &metas {
                        self.alloc_cluster_abort(m);
                    }
                    return Err(Error::from_io("Allocating clusters failed", e));
                }
            };
            for m in &mut metas {
                m.prealloc = true;
            }
            while !metas.is_empty() {
                if let Err(e) = self.alloc_cluster_link_l2(&metas[0], None) {
                    for m in &metas {
                        self.alloc_cluster_abort(m);
                    }
                    return Err(Error::from_io("Mapping clusters failed", e));
                }
                metas.remove(0);
            }
            bytes -= cur_bytes;
            offset += cur_bytes;
        }

        // The file must hold every allocated cluster, or reads past its end would fail.
        let data = self.data();
        let file_length = data.len().map_err(|e| Error::from_io("Could not get file size", e))?;
        if host_offset + cur_bytes > file_length {
            if mode == Prealloc::Metadata {
                mode = Prealloc::Off;
            }
            storage_truncate(&*data, host_offset + cur_bytes, mode)?;
        }
        Ok(())
    }

    /// `qcow2_co_truncate()` without flags.
    pub(crate) fn truncate(&mut self, offset: u64, exact: bool, prealloc: Prealloc) -> Result<()> {
        self.truncate_flags(offset, exact, prealloc, false)
    }

    /// `qcow2_co_truncate()`. `zero_write` is `BDRV_REQ_ZERO_WRITE`: the new area must read
    /// as zeroes even with a backing file.
    pub(crate) fn truncate_flags(
        &mut self,
        offset: u64,
        exact: bool,
        mut prealloc: Prealloc,
        mut zero_write: bool,
    ) -> Result<()> {
        let _ = exact;
        if offset % BDRV_SECTOR_SIZE != 0 {
            return Err(Error::generic(format!(
                "The new size must be a multiple of {BDRV_SECTOR_SIZE}"
            )));
        }

        // The snapshot size was not needed before v3, so it cannot be trusted in v2.
        if !self.snapshots.is_empty() && self.qcow_version < 3 {
            return Err(Error::generic("Can't resize a v2 image which has snapshots"));
        }

        self.truncate_bitmaps_check()?;

        let old_length = self.total_size;
        let new_l1_size = self.size_to_l1(offset);
        let cs = self.cluster_size;

        if offset < old_length {
            if prealloc != Prealloc::Off {
                return Err(Error::generic("Preallocation can't be used for shrinking an image"));
            }
            let start = offset.next_multiple_of(cs);
            if old_length > start {
                self.cluster_discard(start, old_length - start, DiscardType::Always, true)
                    .map_err(|e| Error::from_io("Failed to discard cropped clusters", e))?;
            }
            self.shrink_l1_table(new_l1_size)
                .map_err(|e| Error::from_io("Failed to reduce the number of L2 tables", e))?;
            self.shrink_reftable()
                .map_err(|e| Error::from_io("Failed to discard unused refblocks", e))?;
            let old_file_size = self
                .file
                .len()
                .map_err(|e| Error::from_io("Failed to inquire current file length", e))?;
            let last_cluster = self
                .get_last_cluster(old_file_size)
                .map_err(|e| Error::from_io("Failed to find the last cluster", e))?;
            if (last_cluster + 1) * cs < old_file_size {
                // Not exact: shrinking the qcow2 layer should not fail because the file
                // cannot shrink.
                if let Err(e) = self.file.truncate((last_cluster + 1) * cs) {
                    warn_report(e.prepend("Failed to truncate the tail of the image: ").message());
                }
            }
        } else {
            self.grow_l1_table(new_l1_size, true)
                .map_err(|e| Error::from_io("Failed to grow the L1 table", e))?;
            if self.data_file_is_raw() && prealloc == Prealloc::Off {
                // With data-file-raw every L2 table exists, so that the data file reads the
                // same as the image; the grown part needs them too.
                prealloc = Prealloc::Metadata;
            }
        }

        match prealloc {
            Prealloc::Off => {
                if self.has_data_file() {
                    storage_truncate(&*self.data(), offset, prealloc)?;
                }
            }
            Prealloc::Metadata => self.preallocate(old_length, offset, prealloc)?,
            Prealloc::Falloc | Prealloc::Full => {
                if self.has_data_file() {
                    // Only the metadata, the data file gets the rest.
                    self.preallocate(old_length, offset, prealloc)?;
                } else {
                    self.grow_preallocated(old_length, offset, prealloc, &mut zero_write)?;
                }
            }
        }

        if zero_write && offset > old_length {
            let zero_start = old_length.next_multiple_of(self.subcluster_size);
            // Zero clusters as far as possible. The start must be aligned, the end may be
            // unaligned at the end of the image.
            if offset > zero_start {
                // The new size is not in effect yet, the zeroing must see it.
                let saved = self.total_size;
                self.total_size = offset;
                let r = self.subcluster_zeroize(zero_start, offset - zero_start, false);
                self.total_size = saved;
                r.map_err(|e| Error::from_io("Failed to zero out new clusters", e))?;
            }
            // Explicit zeroes for the unaligned head.
            if zero_start > old_length {
                let len = zero_start.min(offset) - old_length;
                let buf = vec![0u8; len as usize];
                self.pwritev(old_length, &buf)
                    .map_err(|e| Error::from_io("Failed to zero out the new area", e))?;
            }
        }

        if prealloc != Prealloc::Off {
            // The metadata goes to disk before the size changes.
            self.write_caches()
                .map_err(|e| Error::from_io("Failed to flush the preallocated area to disk", e))?;
        }

        self.total_size = offset;
        let r =
            self.file.pwrite(SIZE_OFFSET, &offset.to_be_bytes()).and_then(|()| self.file.flush());
        r.map_err(|e| Error::from_io("Failed to update the image size", e))?;
        self.l1_vm_state_index = new_l1_size;

        for b in &mut self.bitmaps {
            b.truncate(offset);
        }

        // The cache sizes depend on the disk size.
        let opts = self.options.clone();
        self.update_options(&opts)?;
        Ok(())
    }

    /// The `falloc` and `full` branch of `qcow2_co_truncate()` without a data file: the new
    /// data clusters are placed together at the end of the file.
    fn grow_preallocated(
        &mut self,
        old_length: u64,
        offset: u64,
        prealloc: Prealloc,
        zero_write: &mut bool,
    ) -> Result<()> {
        let cs = self.cluster_size;
        let mut old_file_size = self
            .file
            .len()
            .map_err(|e| Error::from_io("Failed to inquire current file length", e))?;
        old_file_size = match self.get_last_cluster(old_file_size) {
            Ok(last) => (last + 1) * cs,
            Err(_) => old_file_size.next_multiple_of(cs),
        };

        let mut nb_new_data_clusters =
            (offset.next_multiple_of(cs) - self.start_of_cluster(old_length)) >> self.cluster_bits;

        // An overestimate: the refcount structures must cover the new L2 tables wherever they
        // end up, so that entering the data clusters needs no new refblocks. One more for a
        // head or tail that is not aligned to an L2 table.
        let nb_new_l2_tables = nb_new_data_clusters.div_ceil(cs / self.l2_entry_size()) + 1;

        let allocation_start = self
            .refcount_area(old_file_size, nb_new_data_clusters + nb_new_l2_tables, true, 0, 0)
            .map_err(|e| Error::from_io("Failed to resize refcount structures", e))?;

        let allocated = self
            .alloc_clusters_at(allocation_start, nb_new_data_clusters)
            .map_err(|e| Error::from_io("Failed to allocate data clusters", e))?;
        assert_eq!(allocated, nb_new_data_clusters);

        // The file grows, so exact does not matter. A grown file reads as zeroes, which takes
        // care of a zeroing request.
        let new_file_size = allocation_start + nb_new_data_clusters * cs;
        let mut subclusters_need_allocation = false;
        if *zero_write {
            *zero_write = false;
            subclusters_need_allocation = true;
        }
        if let Err(e) = storage_truncate(&*self.file, new_file_size, prealloc) {
            self.free_clusters(allocation_start, nb_new_data_clusters * cs, DiscardType::Other);
            return Err(e.prepend("Failed to resize underlying file: "));
        }

        // The L2 entries.
        let mut host_offset = allocation_start;
        let mut guest_offset = old_length;
        while nb_new_data_clusters > 0 {
            let nb_clusters = nb_new_data_clusters
                .min(self.l2_slice_size - self.offset_to_l2_slice_index(guest_offset) as u64);
            let cow_start_length = self.offset_into_cluster(guest_offset);
            guest_offset = self.start_of_cluster(guest_offset);
            let m = L2Meta {
                offset: guest_offset,
                alloc_offset: host_offset,
                nb_clusters,
                cow_start: super::cluster::CowRegion { offset: 0, nb_bytes: cow_start_length },
                cow_end: super::cluster::CowRegion {
                    offset: nb_clusters << self.cluster_bits,
                    nb_bytes: 0,
                },
                prealloc: !subclusters_need_allocation,
                ..Default::default()
            };
            if let Err(e) = self.alloc_cluster_link_l2(&m, None) {
                self.free_clusters(host_offset, nb_new_data_clusters * cs, DiscardType::Other);
                return Err(Error::from_io("Failed to update L2 tables", e));
            }
            guest_offset += nb_clusters * cs;
            host_offset += nb_clusters * cs;
            nb_new_data_clusters -= nb_clusters;
        }
        Ok(())
    }

    /// `make_completely_empty()`.
    fn make_completely_empty(&mut self) -> std::io::Result<()> {
        use super::cache::CacheId;
        self.cache_empty(CacheId::L2)?;
        self.cache_empty(CacheId::Refcount)?;
        // The refcounts are about to be wrong.
        self.mark_dirty()?;

        let cs = self.cluster_size;
        let l1_clusters = (self.l1_size as u64).div_ceil(cs / L1E_SIZE);
        let l1_size2 = self.l1_size as u64 * L1E_SIZE;

        let broken = |s: &mut State, e: std::io::Error| {
            // Nothing can be trusted any more, the node goes away.
            s.drv_gone = true;
            e
        };

        if let Err(e) = self.file.pwrite_zeroes(self.l1_table_offset, l1_clusters * cs, false) {
            return Err(broken(self, e));
        }
        self.l1_table.iter_mut().for_each(|e| *e = 0);

        // Room for the refcount table, one refcount block and the L1 table right after the
        // header. Overwriting the old tables is fine, the dirty bit is set and all data is to
        // go anyway.
        if let Err(e) = self.file.pwrite_zeroes(cs, (2 + l1_clusters) * cs, false) {
            return Err(broken(self, e));
        }

        // The reftable goes right after the header, the L1 table three clusters after it, and
        // the cluster between them is the first refblock.
        let mut b = [0u8; 20];
        b[0..8].copy_from_slice(&(3 * cs).to_be_bytes());
        b[8..16].copy_from_slice(&cs.to_be_bytes());
        b[16..20].copy_from_slice(&1u32.to_be_bytes());
        if let Err(e) = self.file.pwrite(L1_SIZE_OFFSET + 4, &b).and_then(|()| self.file.flush()) {
            return Err(broken(self, e));
        }
        self.l1_table_offset = 3 * cs;

        self.refcount_table_offset = cs;
        self.refcount_table = vec![0; (cs / REFTABLE_ENTRY_SIZE) as usize];
        self.max_refcount_table_index = 0;

        if let Err(e) =
            self.file.pwrite(cs, &(2 * cs).to_be_bytes()).and_then(|()| self.file.flush())
        {
            return Err(broken(self, e));
        }
        self.refcount_table[0] = 2 * cs;

        self.free_cluster_index = 0;
        assert!(3 + l1_clusters <= self.refcount_block_size);
        match self.alloc_clusters(3 * cs + l1_size2) {
            Err(e) => return Err(broken(self, e)),
            Ok(0) => {}
            Ok(_) => {
                ruvm_base::report::error_report("First cluster in emptied image is in use");
                std::process::abort();
            }
        }

        // From here on the in-memory state matches the disk again.
        self.mark_clean()?;
        if let Err(e) = self.file.truncate((3 + l1_clusters) * cs) {
            ruvm_base::report::report_error(&e);
            return Err(errno(libc::EIO));
        }
        Ok(())
    }

    /// `qcow2_make_empty()`.
    pub(crate) fn make_empty(&mut self) -> std::io::Result<()> {
        let cs = self.cluster_size;
        let step = (i32::MAX as u64) & !(cs - 1);
        let l1_clusters = (self.l1_size as u64).div_ceil(cs / L1E_SIZE);

        if self.qcow_version >= 3
            && self.snapshots.is_empty()
            && self.nb_bitmaps == 0
            && 3 + l1_clusters <= self.refcount_block_size
            && self.crypt_method_header != QCOW_CRYPT_LUKS
            && !self.has_data_file()
        {
            // Only v3 has the dirty bit this needs, and nothing may reserve clusters of its
            // own: snapshots, a LUKS header and bitmaps all do. The header, the reftable, one
            // refblock and the L1 table must also fit in one refblock.
            return self.make_completely_empty();
        }

        // Discard every cluster, slow but always possible. This usually runs after
        // committing an external snapshot, hence the snapshot discard type.
        let end_offset = self.total_size;
        let mut offset = 0;
        while offset < end_offset {
            self.cluster_discard(
                offset,
                step.min(end_offset - offset),
                DiscardType::Snapshot,
                true,
            )?;
            offset += step;
        }
        Ok(())
    }
}

/// `BDRV_SECTOR_SIZE`.
const BDRV_SECTOR_SIZE: u64 = 512;
