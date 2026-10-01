// SPDX-License-Identifier: GPL-2.0-or-later

//! Reference counts, from block/qcow2-refcount.c: the refcount table and blocks, cluster
//! allocation and freeing, discards, the metadata overlap checks, and changing the refcount
//! width. The consistency check lives in `check.rs`.

use std::io;

use ruvm_base::{Error, Result};
use ruvm_qapi::types::BlkdebugEvent;

use super::cache::{CacheId, get_be64, set_be64};
use super::header::REFCOUNT_TABLE_OFFSET_OFFSET;
use super::state::*;
use crate::node::errno;

/// Reads entry `index` of a refcount array with `1 << order` bits per entry.
pub(crate) fn get_refcount(order: u32, a: &[u8], index: u64) -> u64 {
    let i = index as usize;
    match order {
        0 => ((a[i / 8] >> (i % 8)) & 0x1) as u64,
        1 => ((a[i / 4] >> (2 * (i % 4))) & 0x3) as u64,
        2 => ((a[i / 2] >> (4 * (i % 2))) & 0xf) as u64,
        3 => a[i] as u64,
        4 => u16::from_be_bytes([a[2 * i], a[2 * i + 1]]) as u64,
        5 => u32::from_be_bytes(a[4 * i..4 * i + 4].try_into().unwrap()) as u64,
        6 => u64::from_be_bytes(a[8 * i..8 * i + 8].try_into().unwrap()),
        _ => unreachable!(),
    }
}

/// Writes entry `index` of a refcount array with `1 << order` bits per entry.
pub(crate) fn set_refcount(order: u32, a: &mut [u8], index: u64, value: u64) {
    let i = index as usize;
    match order {
        0..=2 => {
            let bits = 1usize << order;
            let per_byte = 8 / bits;
            let shift = bits * (i % per_byte);
            let mask = ((1u16 << bits) - 1) as u8;
            assert!(value >> bits == 0);
            a[i / per_byte] &= !(mask << shift);
            a[i / per_byte] |= (value as u8) << shift;
        }
        3 => {
            assert!(value >> 8 == 0);
            a[i] = value as u8;
        }
        4 => {
            assert!(value >> 16 == 0);
            a[2 * i..2 * i + 2].copy_from_slice(&(value as u16).to_be_bytes());
        }
        5 => {
            assert!(value >> 32 == 0);
            a[4 * i..4 * i + 4].copy_from_slice(&(value as u32).to_be_bytes());
        }
        6 => a[8 * i..8 * i + 8].copy_from_slice(&value.to_be_bytes()),
        _ => unreachable!(),
    }
}

pub(crate) fn is_eagain(e: &io::Error) -> bool {
    e.raw_os_error() == Some(libc::EAGAIN)
}

/// `qcow2_refcount_metadata_size()`: the bytes of refcount table and blocks needed to cover
/// `clusters` clusters plus themselves. Also returns the number of refcount blocks.
pub(crate) fn refcount_metadata_size(
    mut clusters: u64,
    cluster_size: u64,
    refcount_order: u32,
    mut generous_increase: bool,
) -> (u64, u64) {
    // Every host cluster is counted, refcount metadata included, so look for the fixed point
    // where no more refcount blocks or table clusters are needed.
    let blocks_per_table_cluster = cluster_size / REFTABLE_ENTRY_SIZE;
    let refcounts_per_block = cluster_size * 8 / (1 << refcount_order);
    let mut table = 0u64;
    let mut blocks = 0u64;
    let mut n = 0u64;
    loop {
        let last = n;
        blocks = (clusters + table + blocks).div_ceil(refcounts_per_block);
        table = blocks.div_ceil(blocks_per_table_cluster);
        n = clusters + blocks + table;
        if n == last && generous_increase {
            clusters += table.div_ceil(2);
            n = 0;
            generous_increase = false;
        }
        if n == last {
            break;
        }
    }
    ((blocks + table) * cluster_size, blocks)
}

impl State {
    pub(crate) fn get_refcount_ro(&self, a: &[u8], index: u64) -> u64 {
        get_refcount(self.refcount_order, a, index)
    }

    /// `update_max_refcount_table_index()`.
    pub(crate) fn update_max_refcount_table_index(&mut self) {
        let mut i = self.refcount_table.len().saturating_sub(1);
        while i > 0 && self.refcount_table[i] & REFT_OFFSET_MASK == 0 {
            i -= 1;
        }
        self.max_refcount_table_index = i as u32;
    }

    /// `qcow2_refcount_init()`.
    pub(crate) fn refcount_init(&mut self, table_size: u64) -> io::Result<()> {
        assert!(self.refcount_order <= 6);
        let bytes = table_size * REFTABLE_ENTRY_SIZE;
        let mut buf = Vec::new();
        buf.try_reserve_exact(bytes as usize).map_err(|_| errno(libc::ENOMEM))?;
        buf.resize(bytes as usize, 0);
        if table_size > 0 {
            self.event(BlkdebugEvent::ReftableLoad);
            self.file.pread(self.refcount_table_offset, &mut buf)?;
        }
        self.refcount_table =
            buf.chunks_exact(8).map(|c| u64::from_be_bytes(c.try_into().unwrap())).collect();
        if table_size > 0 {
            self.update_max_refcount_table_index();
        }
        Ok(())
    }

    /// `qcow2_get_refcount()`.
    pub(crate) fn get_refcount(&mut self, cluster_index: u64) -> io::Result<u64> {
        let rt_index = cluster_index >> self.refcount_block_bits;
        if rt_index >= self.refcount_table.len() as u64 {
            return Ok(0);
        }
        let rb_offset = self.refcount_table[rt_index as usize] & REFT_OFFSET_MASK;
        if rb_offset == 0 {
            return Ok(0);
        }
        if self.offset_into_cluster(rb_offset) != 0 {
            self.signal_corruption(
                true,
                -1,
                -1,
                &format!(
                    "Refblock offset {rb_offset:#x} unaligned (reftable index: {rt_index:#x})",
                    rb_offset = Hx(rb_offset),
                    rt_index = Hx(rt_index)
                ),
            );
            return Err(errno(libc::EIO));
        }
        let i = self.cache_get(CacheId::Refcount, rb_offset)?;
        let block_index = cluster_index & (self.refcount_block_size - 1);
        let r = self.get_refcount_ro(self.refcount_block_cache.table(i), block_index);
        self.cache_put(CacheId::Refcount, i);
        Ok(r)
    }

    fn in_same_refcount_block(&self, a: u64, b: u64) -> bool {
        let shift = self.cluster_bits + self.refcount_block_bits;
        a >> shift == b >> shift
    }

    /// `alloc_refcount_block()`. Loads the refcount block for `cluster_index` into `rb`,
    /// allocating it and growing the refcount table first if needed. `EAGAIN` means something
    /// was allocated and the caller must search for free clusters again; `rb` may hold a block
    /// even on error and the caller must put it.
    fn alloc_refcount_block(
        &mut self,
        cluster_index: u64,
        rb: &mut Option<usize>,
    ) -> io::Result<()> {
        self.event(BlkdebugEvent::RefblockAlloc);
        let rt_index = cluster_index >> self.refcount_block_bits;
        if rt_index < self.refcount_table.len() as u64 {
            let rb_offset = self.refcount_table[rt_index as usize] & REFT_OFFSET_MASK;
            if rb_offset != 0 {
                if self.offset_into_cluster(rb_offset) != 0 {
                    self.signal_corruption(
                        true,
                        -1,
                        -1,
                        &format!(
                            "Refblock offset {rb_offset:#x} unaligned (reftable index: {rt_index:#x})", rb_offset = Hx(rb_offset), rt_index = Hx(rt_index)),
                    );
                    return Err(errno(libc::EIO));
                }
                self.event(BlkdebugEvent::RefblockLoad);
                *rb = Some(self.cache_get(CacheId::Refcount, rb_offset)?);
                return Ok(());
            }
        }

        // Something has to be allocated: at least the refcount block, maybe a new table. The
        // normal allocator would recurse, so the blocks are placed where they can describe
        // themselves.
        *rb = None;

        // The refcount table gets written, and it may depend on L2 tables.
        self.cache_flush(CacheId::L2)?;

        let new_block = self.alloc_clusters_noref(self.cluster_size, i64::MAX as u64)?;
        assert_eq!(new_block & REFT_OFFSET_MASK, new_block);
        if new_block == 0 {
            self.signal_corruption(
                true,
                -1,
                -1,
                "Preventing invalid allocation of refcount block at offset 0",
            );
            return Err(errno(libc::EIO));
        }

        if self.in_same_refcount_block(new_block, cluster_index << self.cluster_bits) {
            let i = self.cache_get_empty(CacheId::Refcount, new_block)?;
            *rb = Some(i);
            let block_index = (new_block >> self.cluster_bits) & (self.refcount_block_size - 1);
            let order = self.refcount_order;
            let t = self.refcount_block_cache.table_mut(i);
            t.fill(0);
            // The block describes itself.
            set_refcount(order, t, block_index, 1);
        } else {
            // Described somewhere else. This recurses at most twice before reaching a block that
            // describes itself.
            self.update_refcount(new_block, self.cluster_size, 1, false, DiscardType::Never)?;
            self.cache_flush(CacheId::Refcount)?;
            // Only initialise the block now, update_refcount uses the cache itself.
            let i = self.cache_get_empty(CacheId::Refcount, new_block)?;
            *rb = Some(i);
            self.refcount_block_cache.table_mut(i).fill(0);
        }

        let i = rb.unwrap();
        self.event(BlkdebugEvent::RefblockAllocWrite);
        self.refcount_block_cache.mark_dirty(i);
        self.cache_flush(CacheId::Refcount)?;

        // If the table is big enough, just hook the block up.
        if rt_index < self.refcount_table.len() as u64 {
            self.event(BlkdebugEvent::RefblockAllocHookup);
            self.file.pwrite(
                self.refcount_table_offset + rt_index * REFTABLE_ENTRY_SIZE,
                &new_block.to_be_bytes(),
            )?;
            self.file.flush()?;
            self.refcount_table[rt_index as usize] = new_block;
            // With a hole in the table the index may be below the maximum.
            self.max_refcount_table_index = self.max_refcount_table_index.max(rt_index as u32);
            // The block may be where the caller wanted its data, so let it search again.
            return Err(errno(libc::EAGAIN));
        }

        self.cache_put(CacheId::Refcount, i);
        *rb = None;

        // The table has to grow. New refcount blocks go at the end of the image and describe
        // themselves and the new table, so the switch happens at once.
        self.event(BlkdebugEvent::ReftableGrow);
        let blocks_used = (cluster_index + 1)
            .max((new_block >> self.cluster_bits) + 1)
            .div_ceil(self.refcount_block_size);
        let meta_offset = blocks_used * self.refcount_block_size * self.cluster_size;
        self.refcount_area(meta_offset, 0, false, rt_index, new_block)?;
        self.event(BlkdebugEvent::RefblockLoad);
        *rb = Some(self.cache_get(CacheId::Refcount, new_block)?);

        // The new metadata may sit where the caller wanted to allocate.
        Err(errno(libc::EAGAIN))
    }

    /// `qcow2_refcount_area()`: builds a new self-covering refcount table and blocks at
    /// `start_offset`, which must be at the end of the image, covering `additional_clusters`
    /// more clusters after them. Returns the offset after the new structures.
    pub(crate) fn refcount_area(
        &mut self,
        start_offset: u64,
        additional_clusters: u64,
        exact_size: bool,
        new_refblock_index: u64,
        new_refblock_offset: u64,
    ) -> io::Result<u64> {
        assert_eq!(start_offset % self.cluster_size, 0);
        let cs = self.cluster_size;
        let (_, total_refblock_count) = refcount_metadata_size(
            start_offset / cs + additional_clusters,
            cs,
            self.refcount_order,
            !exact_size,
        );
        if total_refblock_count > QCOW_MAX_REFTABLE_SIZE {
            return Err(errno(libc::EFBIG));
        }
        let area_reftable_index = (start_offset / cs) / self.refcount_block_size;
        let mut table_size = if exact_size {
            total_refblock_count
        } else {
            total_refblock_count + total_refblock_count.div_ceil(2)
        };
        // The header stores the table size in clusters.
        table_size = table_size.next_multiple_of(cs / REFTABLE_ENTRY_SIZE);
        let table_clusters = table_size * REFTABLE_ENTRY_SIZE / cs;
        if table_size > QCOW_MAX_REFTABLE_SIZE {
            return Err(errno(libc::EFBIG));
        }
        assert!(table_size > 0);

        let mut new_table = vec![0u64; table_size as usize];
        if table_size > self.max_refcount_table_index as u64 {
            let n = (self.max_refcount_table_index as usize + 1).min(self.refcount_table.len());
            new_table[..n].copy_from_slice(&self.refcount_table[..n]);
        } else {
            // Shrinking: the caller made sure there is nothing beyond start_offset.
            new_table.copy_from_slice(&self.refcount_table[..table_size as usize]);
        }
        if new_refblock_offset != 0 {
            assert!(new_refblock_index < total_refblock_count);
            new_table[new_refblock_index as usize] = new_refblock_offset;
        }

        let additional_refblock_count = (area_reftable_index..total_refblock_count)
            .filter(|&i| new_table[i as usize] == 0)
            .count() as u64;
        let table_offset = start_offset + additional_refblock_count * cs;
        let end_offset = table_offset + table_clusters * cs;

        let mut block_offset = start_offset;
        for i in area_reftable_index..total_refblock_count {
            let slot = if new_table[i as usize] != 0 {
                self.cache_get(CacheId::Refcount, new_table[i as usize])?
            } else {
                let slot = self.cache_get_empty(CacheId::Refcount, block_offset)?;
                self.refcount_block_cache.table_mut(slot).fill(0);
                self.refcount_block_cache.mark_dirty(slot);
                new_table[i as usize] = block_offset;
                block_offset += cs;
                slot
            };

            let first_offset_covered = i * self.refcount_block_size * cs;
            if first_offset_covered < end_offset {
                // Every new refcount structure gets refcount 1.
                let mut j = if first_offset_covered < start_offset {
                    assert_eq!(i, area_reftable_index);
                    (start_offset - first_offset_covered) / cs
                } else {
                    0
                };
                let end_index =
                    ((end_offset - first_offset_covered) / cs).min(self.refcount_block_size);
                let order = self.refcount_order;
                let t = self.refcount_block_cache.table_mut(slot);
                while j < end_index {
                    assert_eq!(get_refcount(order, t, j), 0);
                    set_refcount(order, t, j, 1);
                    j += 1;
                }
                self.refcount_block_cache.mark_dirty(slot);
            }
            self.cache_put(CacheId::Refcount, slot);
        }
        assert_eq!(block_offset, table_offset);

        self.event(BlkdebugEvent::RefblockAllocWriteBlocks);
        self.cache_flush(CacheId::Refcount)?;

        let mut buf = vec![0u8; (table_size * REFTABLE_ENTRY_SIZE) as usize];
        for (i, v) in new_table.iter().enumerate() {
            set_be64(&mut buf, i, *v);
        }
        self.event(BlkdebugEvent::RefblockAllocWriteTable);
        self.file.pwrite(table_offset, &buf)?;
        self.file.flush()?;

        let mut data = [0u8; 12];
        data[..8].copy_from_slice(&table_offset.to_be_bytes());
        data[8..].copy_from_slice(&(table_clusters as u32).to_be_bytes());
        self.event(BlkdebugEvent::RefblockAllocSwitchTable);
        self.file.pwrite(REFCOUNT_TABLE_OFFSET_OFFSET, &data)?;
        self.file.flush()?;

        let old_table_offset = self.refcount_table_offset;
        let old_table_size = self.refcount_table.len() as u64;
        self.refcount_table = new_table;
        self.refcount_table_offset = table_offset;
        self.update_max_refcount_table_index();

        self.free_clusters(
            old_table_offset,
            old_table_size * REFTABLE_ENTRY_SIZE,
            DiscardType::Other,
        );
        Ok(end_offset)
    }

    /// `qcow2_process_discards()`. Discards are advice, so their errors are ignored.
    pub(crate) fn process_discards(&mut self, ok: bool) {
        let discards = std::mem::take(&mut self.discards);
        if ok {
            for (offset, bytes) in discards {
                let _ = self.file.pdiscard(offset, bytes);
            }
        }
    }

    /// `queue_discard()`: remembers a region to discard, merging it with adjacent ones.
    pub(crate) fn queue_discard(&mut self, offset: u64, length: u64) {
        let mut found = None;
        for (k, d) in self.discards.iter_mut().enumerate() {
            let new_start = offset.min(d.0);
            let new_end = (offset + length).max(d.0 + d.1);
            if new_end - new_start <= length + d.1 {
                // Freed areas have no references left, so they never overlap.
                assert_eq!(d.1 + length, new_end - new_start);
                d.0 = new_start;
                d.1 = new_end - new_start;
                found = Some(k);
                break;
            }
        }
        let k = match found {
            Some(k) => k,
            None => {
                self.discards.push((offset, length));
                self.discards.len() - 1
            }
        };
        // Merge the requests that are adjacent now.
        let mut i = 0;
        let mut k = k;
        while i < self.discards.len() {
            let d = self.discards[k];
            let p = self.discards[i];
            if i == k || p.0 > d.0 + d.1 || d.0 > p.0 + p.1 {
                i += 1;
                continue;
            }
            assert!(p.0 == d.0 + d.1 || d.0 == p.0 + p.1);
            self.discards.remove(i);
            if i < k {
                k -= 1;
            }
            let d = &mut self.discards[k];
            d.0 = d.0.min(p.0);
            d.1 += p.1;
        }
    }

    /// `update_refcount()`: adds `addend` to (or with `decrease` subtracts it from) the
    /// refcount of every cluster in `[offset, offset + length)`.
    pub(crate) fn update_refcount(
        &mut self,
        offset: u64,
        length: u64,
        addend: u64,
        decrease: bool,
        ty: DiscardType,
    ) -> io::Result<()> {
        if length == 0 {
            return Ok(());
        }
        if decrease {
            let _ = self.cache_set_dependency(CacheId::Refcount, CacheId::L2);
        }

        let start = self.start_of_cluster(offset);
        let last = self.start_of_cluster(offset + length - 1);
        let mut rb: Option<usize> = None;
        let mut old_table_index: Option<u64> = None;
        let mut cluster_offset = start;
        let mut ret: io::Result<()> = Ok(());

        while cluster_offset <= last {
            let cluster_index = cluster_offset >> self.cluster_bits;
            let table_index = cluster_index >> self.refcount_block_bits;

            if old_table_index != Some(table_index) {
                if let Some(i) = rb.take() {
                    self.cache_put(CacheId::Refcount, i);
                }
                let r = self.alloc_refcount_block(cluster_index, &mut rb);
                if let Err(e) = r {
                    // The caller restarts its search; try the same clusters first.
                    if is_eagain(&e) && self.free_cluster_index > (start >> self.cluster_bits) {
                        self.free_cluster_index = start >> self.cluster_bits;
                    }
                    ret = Err(e);
                    break;
                }
            }
            old_table_index = Some(table_index);

            let slot = rb.unwrap();
            self.refcount_block_cache.mark_dirty(slot);
            let block_index = cluster_index & (self.refcount_block_size - 1);
            let mut refcount =
                self.get_refcount_ro(self.refcount_block_cache.table(slot), block_index);
            let bad = if decrease {
                refcount.checked_sub(addend).is_none()
            } else {
                refcount.checked_add(addend).is_none_or(|r| r > self.refcount_max)
            };
            if bad {
                ret = Err(errno(libc::EINVAL));
                break;
            }
            if decrease {
                refcount -= addend;
            } else {
                refcount += addend;
            }
            if refcount == 0 && cluster_index < self.free_cluster_index {
                self.free_cluster_index = cluster_index;
            }
            let order = self.refcount_order;
            set_refcount(order, self.refcount_block_cache.table_mut(slot), block_index, refcount);

            if refcount == 0 {
                // A freed cluster that held a table must not be written back from the cache.
                if let Some(t) = self.refcount_block_cache.is_table_offset(cluster_offset) {
                    self.cache_put(CacheId::Refcount, slot);
                    rb = None;
                    old_table_index = None;
                    self.refcount_block_cache.discard(t);
                }
                if let Some(t) = self.l2_table_cache.is_table_offset(cluster_offset) {
                    self.l2_table_cache.discard(t);
                }
                if self.discard_passthrough[ty as usize] {
                    self.queue_discard(cluster_offset, self.cluster_size);
                }
            }
            cluster_offset += self.cluster_size;
        }

        if !self.cache_discards {
            self.process_discards(ret.is_ok());
        }
        if let Some(i) = rb {
            self.cache_put(CacheId::Refcount, i);
        }
        // Try to undo what was done, which works in some cases such as ENOSPC while allocating
        // a new refcount block.
        if ret.is_err() && cluster_offset > offset {
            let _ = self.update_refcount(
                offset,
                cluster_offset - offset,
                addend,
                !decrease,
                DiscardType::Never,
            );
        }
        ret
    }

    /// `qcow2_update_cluster_refcount()`.
    pub(crate) fn update_cluster_refcount(
        &mut self,
        cluster_index: u64,
        addend: u64,
        decrease: bool,
        ty: DiscardType,
    ) -> io::Result<()> {
        self.update_refcount(cluster_index << self.cluster_bits, 1, addend, decrease, ty)
    }

    /// `alloc_clusters_noref()`: finds free clusters without taking a reference.
    pub(crate) fn alloc_clusters_noref(&mut self, size: u64, max: u64) -> io::Result<u64> {
        // Clusters queued for discard must not be handed out.
        if self.cache_discards {
            self.process_discards(true);
        }
        let nb_clusters = self.size_to_clusters(size);
        'retry: loop {
            for _ in 0..nb_clusters {
                let next = self.free_cluster_index;
                self.free_cluster_index += 1;
                if self.get_refcount(next)? != 0 {
                    continue 'retry;
                }
            }
            break;
        }
        // Every offset in the range must be representable within `max`.
        if self.free_cluster_index > 0 && self.free_cluster_index - 1 > (max >> self.cluster_bits) {
            return Err(errno(libc::EFBIG));
        }
        Ok((self.free_cluster_index - nb_clusters) << self.cluster_bits)
    }

    /// `qcow2_alloc_clusters()`.
    pub(crate) fn alloc_clusters(&mut self, size: u64) -> io::Result<u64> {
        self.event(BlkdebugEvent::ClusterAlloc);
        loop {
            let offset = self.alloc_clusters_noref(size, QCOW_MAX_CLUSTER_OFFSET)?;
            match self.update_refcount(offset, size, 1, false, DiscardType::Never) {
                Err(e) if is_eagain(&e) => continue,
                Err(e) => return Err(e),
                Ok(()) => return Ok(offset),
            }
        }
    }

    /// `qcow2_alloc_clusters_at()`: takes as many of `nb_clusters` free clusters at `offset`
    /// as are free in a row, returning how many.
    pub(crate) fn alloc_clusters_at(&mut self, offset: u64, nb_clusters: u64) -> io::Result<u64> {
        if nb_clusters == 0 {
            return Ok(0);
        }
        loop {
            let mut cluster_index = offset >> self.cluster_bits;
            let mut i = 0;
            while i < nb_clusters {
                let r = self.get_refcount(cluster_index)?;
                cluster_index += 1;
                if r != 0 {
                    break;
                }
                i += 1;
            }
            match self.update_refcount(offset, i << self.cluster_bits, 1, false, DiscardType::Never)
            {
                Err(e) if is_eagain(&e) => continue,
                Err(e) => return Err(e),
                Ok(()) => return Ok(i),
            }
        }
    }

    /// `qcow2_alloc_bytes()`: space for compressed data, packed into shared clusters.
    pub(crate) fn alloc_bytes(&mut self, size: u64) -> io::Result<u64> {
        self.event(BlkdebugEvent::ClusterAllocBytes);
        assert!(size > 0 && size <= self.cluster_size);
        assert!(self.free_byte_offset == 0 || self.offset_into_cluster(self.free_byte_offset) != 0);
        let mut offset = self.free_byte_offset;
        if offset != 0 {
            let refcount = self.get_refcount(offset >> self.cluster_bits)?;
            if refcount == self.refcount_max {
                offset = 0;
            }
        }
        let mut free_in_cluster = self.cluster_size - self.offset_into_cluster(offset);
        loop {
            if offset == 0 || free_in_cluster < size {
                let max = self.cluster_offset_mask.min(QCOW_MAX_CLUSTER_OFFSET);
                let new_cluster = self.alloc_clusters_noref(self.cluster_size, max)?;
                if new_cluster == 0 {
                    self.signal_corruption(
                        true,
                        -1,
                        -1,
                        "Preventing invalid allocation of compressed cluster at offset 0",
                    );
                    return Err(errno(libc::EIO));
                }
                if offset == 0 || offset.next_multiple_of(self.cluster_size) != new_cluster {
                    offset = new_cluster;
                    free_in_cluster = self.cluster_size;
                } else {
                    free_in_cluster += self.cluster_size;
                }
            }
            assert!(offset != 0);
            match self.update_refcount(offset, size, 1, false, DiscardType::Never) {
                Err(e) if is_eagain(&e) => {
                    offset = 0;
                    continue;
                }
                Err(e) => return Err(e),
                Ok(()) => break,
            }
        }
        // The refcount went up; refcount blocks must reach the disk before the L2 update.
        let _ = self.cache_set_dependency(CacheId::L2, CacheId::Refcount);
        self.free_byte_offset = offset + size;
        if self.offset_into_cluster(self.free_byte_offset) == 0 {
            self.free_byte_offset = 0;
        }
        Ok(offset)
    }

    /// `qcow2_free_clusters()`.
    pub(crate) fn free_clusters(&mut self, offset: u64, size: u64, ty: DiscardType) {
        self.event(BlkdebugEvent::ClusterFree);
        if let Err(e) = self.update_refcount(offset, size, 1, true, ty) {
            eprintln!("qcow2_free_clusters failed: {}", ruvm_base::error::strerror(&e));
        }
    }

    /// `qcow2_free_any_cluster()`: frees a cluster of any type given its L2 entry.
    pub(crate) fn free_any_cluster(&mut self, l2_entry: u64, ty: DiscardType) {
        let ctype = self.cluster_type(l2_entry);
        if self.has_data_file() {
            if self.discard_passthrough[ty as usize]
                && matches!(ctype, ClusterType::Normal | ClusterType::ZeroAlloc)
            {
                let _ = self.data().pdiscard(l2_entry & L2E_OFFSET_MASK, self.cluster_size);
            }
            return;
        }
        match ctype {
            ClusterType::Compressed => {
                let (coffset, csize) = self.parse_compressed_l2_entry(l2_entry);
                self.free_clusters(coffset, csize, ty);
            }
            ClusterType::Normal | ClusterType::ZeroAlloc => {
                let off = l2_entry & L2E_OFFSET_MASK;
                if self.offset_into_cluster(off) != 0 {
                    self.signal_corruption(
                        false,
                        -1,
                        -1,
                        &format!("Cannot free unaligned cluster {off:#x}", off = Hx(off)),
                    );
                } else {
                    self.free_clusters(off, self.cluster_size, ty);
                }
            }
            ClusterType::ZeroPlain | ClusterType::Unallocated => {}
        }
    }

    /// `qcow2_discard_cluster()`: passes a discard down without freeing anything.
    pub(crate) fn discard_cluster(
        &mut self,
        offset: u64,
        length: u64,
        ctype: ClusterType,
        dtype: DiscardType,
    ) {
        if self.discard_passthrough[dtype as usize]
            && matches!(ctype, ClusterType::Normal | ClusterType::ZeroAlloc)
        {
            if self.has_data_file() {
                let _ = self.data().pdiscard(offset, length);
            } else {
                self.queue_discard(offset, length);
            }
        }
    }

    /// `qcow2_update_snapshot_refcount()`: adds `addend` (-1, 0 or 1) to every cluster an L1
    /// table references and fixes the COPIED flags.
    pub(crate) fn update_snapshot_refcount(
        &mut self,
        l1_table_offset: u64,
        l1_size: u32,
        addend: i32,
    ) -> io::Result<()> {
        assert!((-1..=1).contains(&addend));
        let slice_size2 = self.l2_slice_size * self.l2_entry_size();
        let n_slices = self.cluster_size / slice_size2;
        self.cache_discards = true;

        // qcow2_snapshot_goto() relies on the active table not being read from disk here.
        let active = l1_table_offset == self.l1_table_offset;
        let mut l1_table: Vec<u64> = if !active {
            let mut buf = vec![0u8; l1_size as usize * 8];
            if let Err(e) = self.file.pread(l1_table_offset, &mut buf) {
                self.cache_discards = false;
                self.process_discards(false);
                return Err(e);
            }
            buf.chunks_exact(8).map(|c| u64::from_be_bytes(c.try_into().unwrap())).collect()
        } else {
            assert_eq!(l1_size, self.l1_size);
            self.l1_table.clone()
        };

        let mut l1_modified = false;
        let r = self.update_snapshot_refcount_walk(
            &mut l1_table,
            &mut l1_modified,
            addend,
            n_slices,
            slice_size2,
        );
        let ret = r.and_then(|()| self.flush_all());

        self.cache_discards = false;
        self.process_discards(ret.is_ok());

        if active {
            self.l1_table.clone_from(&l1_table);
        }
        // Only update the L1 table if it is not about to be deleted.
        if ret.is_ok() && addend >= 0 && l1_modified {
            let mut buf = vec![0u8; l1_size as usize * 8];
            for (i, v) in l1_table.iter().enumerate() {
                set_be64(&mut buf, i, *v);
            }
            self.file.pwrite(l1_table_offset, &buf)?;
            self.file.flush()?;
        }
        ret
    }

    fn update_snapshot_refcount_walk(
        &mut self,
        l1_table: &mut [u64],
        l1_modified: &mut bool,
        addend: i32,
        n_slices: u64,
        slice_size2: u64,
    ) -> io::Result<()> {
        let abs = addend.unsigned_abs() as u64;
        let dec = addend < 0;
        for (i, l1_entry) in l1_table.iter_mut().enumerate() {
            let old_l2_offset = *l1_entry;
            if old_l2_offset == 0 {
                continue;
            }
            let mut l2_offset = old_l2_offset & L1E_OFFSET_MASK;
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
            for slice in 0..n_slices {
                let slot = self.cache_get(CacheId::L2, l2_offset + slice * slice_size2)?;
                let r =
                    self.update_snapshot_refcount_slice(slot, l2_offset, slice, addend, abs, dec);
                self.cache_put(CacheId::L2, slot);
                r?;
            }
            if addend != 0 {
                self.update_cluster_refcount(
                    l2_offset >> self.cluster_bits,
                    abs,
                    dec,
                    DiscardType::Snapshot,
                )?;
            }
            let refcount = self.get_refcount(l2_offset >> self.cluster_bits)?;
            if refcount == 1 {
                l2_offset |= QCOW_OFLAG_COPIED;
            }
            if l2_offset != old_l2_offset {
                *l1_entry = l2_offset;
                *l1_modified = true;
            }
        }
        Ok(())
    }

    fn update_snapshot_refcount_slice(
        &mut self,
        slot: usize,
        l2_offset: u64,
        slice: u64,
        addend: i32,
        abs: u64,
        dec: bool,
    ) -> io::Result<()> {
        for j in 0..self.l2_slice_size as usize {
            let old_entry = self.l2_entry(self.l2_table_cache.table(slot), j);
            let mut entry = old_entry & !QCOW_OFLAG_COPIED;
            let offset = entry & L2E_OFFSET_MASK;
            let refcount = match self.cluster_type(entry) {
                ClusterType::Compressed => {
                    if addend != 0 {
                        let (coffset, csize) = self.parse_compressed_l2_entry(entry);
                        self.update_refcount(coffset, csize, abs, dec, DiscardType::Snapshot)?;
                    }
                    // Compressed clusters are never modified in place.
                    2
                }
                ClusterType::Normal | ClusterType::ZeroAlloc => {
                    if self.offset_into_cluster(offset) != 0 {
                        let l2_index = slice * self.l2_slice_size + j as u64;
                        self.signal_corruption(
                            true,
                            -1,
                            -1,
                            &format!(
                                "Cluster allocation offset {offset:#x} unaligned (L2 offset: \
                                 {l2_offset:#x}, L2 index: {l2_index:#x})",
                                offset = Hx(offset),
                                l2_offset = Hx(l2_offset),
                                l2_index = Hx(l2_index)
                            ),
                        );
                        return Err(errno(libc::EIO));
                    }
                    let cluster_index = offset >> self.cluster_bits;
                    assert!(cluster_index != 0);
                    if addend != 0 {
                        self.update_cluster_refcount(
                            cluster_index,
                            abs,
                            dec,
                            DiscardType::Snapshot,
                        )?;
                    }
                    self.get_refcount(cluster_index)?
                }
                ClusterType::ZeroPlain | ClusterType::Unallocated => 0,
            };
            if refcount == 1 {
                entry |= QCOW_OFLAG_COPIED;
            }
            if entry != old_entry {
                if addend > 0 {
                    let _ = self.cache_set_dependency(CacheId::L2, CacheId::Refcount);
                }
                let idx = j * (self.l2_entry_size() / 8) as usize;
                set_be64(self.l2_table_cache.table_mut(slot), idx, entry);
                self.l2_table_cache.mark_dirty(slot);
            }
        }
        Ok(())
    }

    /// `qcow2_check_metadata_overlap()`: which metadata, not counting the kinds in `ign`, a
    /// write to `[offset, offset + size)` would hit. 0 means none.
    pub(crate) fn check_metadata_overlap(
        &self,
        ign: u32,
        offset: u64,
        size: u64,
    ) -> io::Result<u32> {
        let chk = self.overlap_check & !ign;
        if size == 0 {
            return Ok(0);
        }
        if chk & OL_MAIN_HEADER != 0 && offset < self.cluster_size {
            return Ok(OL_MAIN_HEADER);
        }
        // Align the range to cluster boundaries.
        let size = (self.offset_into_cluster(offset) + size).next_multiple_of(self.cluster_size);
        let offset = self.start_of_cluster(offset);
        let overlaps = |o: u64, s: u64| s > 0 && offset < o.saturating_add(s) && o < offset + size;

        if chk & OL_ACTIVE_L1 != 0
            && self.l1_size != 0
            && overlaps(self.l1_table_offset, self.l1_size as u64 * L1E_SIZE)
        {
            return Ok(OL_ACTIVE_L1);
        }
        if chk & OL_REFCOUNT_TABLE != 0
            && !self.refcount_table.is_empty()
            && overlaps(
                self.refcount_table_offset,
                self.refcount_table.len() as u64 * REFTABLE_ENTRY_SIZE,
            )
        {
            return Ok(OL_REFCOUNT_TABLE);
        }
        if chk & OL_SNAPSHOT_TABLE != 0
            && self.snapshots_size != 0
            && overlaps(self.snapshots_offset, self.snapshots_size)
        {
            return Ok(OL_SNAPSHOT_TABLE);
        }
        if chk & OL_INACTIVE_L1 != 0 {
            for sn in &self.snapshots {
                if sn.l1_size != 0 && overlaps(sn.l1_table_offset, sn.l1_size as u64 * L1E_SIZE) {
                    return Ok(OL_INACTIVE_L1);
                }
            }
        }
        if chk & OL_ACTIVE_L2 != 0 {
            for &e in &self.l1_table {
                if e & L1E_OFFSET_MASK != 0 && overlaps(e & L1E_OFFSET_MASK, self.cluster_size) {
                    return Ok(OL_ACTIVE_L2);
                }
            }
        }
        if chk & OL_REFCOUNT_BLOCK != 0 && !self.refcount_table.is_empty() {
            let last = self.max_refcount_table_index as usize;
            for &e in &self.refcount_table[..=last] {
                if e & REFT_OFFSET_MASK != 0 && overlaps(e & REFT_OFFSET_MASK, self.cluster_size) {
                    return Ok(OL_REFCOUNT_BLOCK);
                }
            }
        }
        if chk & OL_INACTIVE_L2 != 0 {
            for sn in &self.snapshots {
                if let Err((_, code)) = self.validate_table(
                    sn.l1_table_offset,
                    sn.l1_size as u64,
                    L1E_SIZE,
                    QCOW_MAX_L1_SIZE,
                    "",
                ) {
                    return Err(errno(code));
                }
                let mut l1 = vec![0u8; sn.l1_size as usize * 8];
                self.file.pread(sn.l1_table_offset, &mut l1)?;
                for j in 0..sn.l1_size as usize {
                    let l2 = get_be64(&l1, j) & L1E_OFFSET_MASK;
                    if l2 != 0 && overlaps(l2, self.cluster_size) {
                        return Ok(OL_INACTIVE_L2);
                    }
                }
            }
        }
        if chk & OL_BITMAP_DIRECTORY != 0
            && self.autoclear_features & QCOW2_AUTOCLEAR_BITMAPS != 0
            && overlaps(self.bitmap_directory_offset, self.bitmap_directory_size)
        {
            return Ok(OL_BITMAP_DIRECTORY);
        }
        Ok(0)
    }

    /// `qcow2_pre_write_overlap_check()`: refuses a write over metadata and marks the image
    /// corrupt.
    pub(crate) fn pre_write_overlap_check(
        &mut self,
        ign: u32,
        offset: u64,
        size: u64,
        data_file: bool,
    ) -> io::Result<()> {
        if data_file && self.has_data_file() {
            return Ok(());
        }
        let r = self.check_metadata_overlap(ign, offset, size)?;
        if r > 0 {
            let bit = r.trailing_zeros();
            assert!(bit < OL_MAX_BITNR);
            self.signal_corruption(
                true,
                offset as i64,
                size as i64,
                &format!(
                    "Preventing invalid write on metadata (overlaps with {})",
                    METADATA_OL_NAMES[bit as usize]
                ),
            );
            return Err(errno(libc::EIO));
        }
        Ok(())
    }

    /// `alloc_refblock()` for `walk_over_reftable()`.
    fn walk_alloc_refblock(
        &mut self,
        reftable: &mut Vec<u64>,
        reftable_index: u64,
        refblock_empty: bool,
        allocated: &mut bool,
    ) -> Result<()> {
        if !refblock_empty && reftable_index >= reftable.len() as u64 {
            let new_size =
                (reftable_index + 1).next_multiple_of(self.cluster_size / REFTABLE_ENTRY_SIZE);
            if new_size > QCOW_MAX_REFTABLE_SIZE / REFTABLE_ENTRY_SIZE {
                return Err(Error::generic(
                    "This operation would make the refcount table grow beyond the maximum size \
                     supported by QEMU, aborting",
                ));
            }
            reftable.resize(new_size as usize, 0);
        }
        if !refblock_empty && reftable[reftable_index as usize] == 0 {
            let offset = self
                .alloc_clusters(self.cluster_size)
                .map_err(|e| Error::from_io("Failed to allocate refblock", e))?;
            reftable[reftable_index as usize] = offset;
            *allocated = true;
        }
        Ok(())
    }

    /// `flush_refblock()` for `walk_over_reftable()`.
    fn walk_flush_refblock(
        &mut self,
        reftable: &[u64],
        reftable_index: u64,
        refblock: &[u8],
        refblock_empty: bool,
    ) -> Result<()> {
        if reftable_index < reftable.len() as u64 && reftable[reftable_index as usize] != 0 {
            let offset = reftable[reftable_index as usize];
            self.pre_write_overlap_check(0, offset, self.cluster_size, false)
                .map_err(|e| Error::from_io("Overlap check failed", e))?;
            self.file
                .pwrite(offset, refblock)
                .map_err(|e| Error::from_io("Failed to write refblock", e))?;
        } else {
            assert!(refblock_empty);
        }
        Ok(())
    }

    /// `walk_over_reftable()`. With `new_order` set the new refblock is filled and written
    /// (the `flush_refblock` pass); without, only the allocation pass runs.
    #[allow(clippy::too_many_arguments)]
    fn walk_over_reftable(
        &mut self,
        new_reftable: &mut Vec<u64>,
        new_reftable_index: &mut u64,
        new_refblock: &mut [u8],
        new_refblock_size: u64,
        new_refcount_bits: u32,
        write_pass: Option<u32>,
        allocated: &mut bool,
        status: &mut dyn FnMut(u64, u64),
        index: u64,
        total: u64,
    ) -> Result<()> {
        let mut new_refblock_empty = true;
        let mut new_refblock_index = 0u64;
        let rt_size = self.refcount_table.len() as u64;

        let finish = |st: &mut State,
                      table: &mut Vec<u64>,
                      idx: u64,
                      block: &[u8],
                      empty: bool,
                      alloc: &mut bool| {
            if write_pass.is_some() {
                st.walk_flush_refblock(table, idx, block, empty)
            } else {
                st.walk_alloc_refblock(table, idx, empty, alloc)
            }
        };

        for reftable_index in 0..rt_size {
            let refblock_offset = self.refcount_table[reftable_index as usize] & REFT_OFFSET_MASK;
            status(index * rt_size + reftable_index, total * rt_size);
            let slot = if refblock_offset != 0 {
                if self.offset_into_cluster(refblock_offset) != 0 {
                    self.signal_corruption(
                        true,
                        -1,
                        -1,
                        &format!(
                            "Refblock offset {refblock_offset:#x} unaligned (reftable index: \
                             {reftable_index:#x})",
                            refblock_offset = Hx(refblock_offset),
                            reftable_index = Hx(reftable_index)
                        ),
                    );
                    return Err(Error::generic("Image is corrupt (unaligned refblock offset)"));
                }
                Some(
                    self.cache_get(CacheId::Refcount, refblock_offset)
                        .map_err(|e| Error::from_io("Failed to retrieve refblock", e))?,
                )
            } else {
                None
            };

            for refblock_index in 0..self.refcount_block_size {
                if new_refblock_index >= new_refblock_size {
                    if let Err(e) = finish(
                        self,
                        new_reftable,
                        *new_reftable_index,
                        new_refblock,
                        new_refblock_empty,
                        allocated,
                    ) {
                        if let Some(s) = slot {
                            self.cache_put(CacheId::Refcount, s);
                        }
                        return Err(e);
                    }
                    *new_reftable_index += 1;
                    new_refblock_index = 0;
                    new_refblock_empty = true;
                }
                let refcount = match slot {
                    Some(s) => {
                        self.get_refcount_ro(self.refcount_block_cache.table(s), refblock_index)
                    }
                    None => 0,
                };
                if new_refcount_bits < 64 && refcount >> new_refcount_bits != 0 {
                    if let Some(s) = slot {
                        self.cache_put(CacheId::Refcount, s);
                    }
                    let offset = ((reftable_index << self.refcount_block_bits) + refblock_index)
                        << self.cluster_bits;
                    return Err(Error::generic(format!(
                        "Cannot decrease refcount entry width to {new_refcount_bits} bits: Cluster \
                         at offset {offset:#x} has a refcount of {refcount}",
                        offset = Hx(offset)
                    )));
                }
                if let Some(order) = write_pass {
                    set_refcount(order, new_refblock, new_refblock_index, refcount);
                }
                new_refblock_index += 1;
                new_refblock_empty = new_refblock_empty && refcount == 0;
            }
            if let Some(s) = slot {
                self.cache_put(CacheId::Refcount, s);
            }
        }

        if new_refblock_index > 0 {
            // Complete the partially filled last refblock.
            if let Some(order) = write_pass {
                while new_refblock_index < new_refblock_size {
                    set_refcount(order, new_refblock, new_refblock_index, 0);
                    new_refblock_index += 1;
                }
            }
            finish(
                self,
                new_reftable,
                *new_reftable_index,
                new_refblock,
                new_refblock_empty,
                allocated,
            )?;
            *new_reftable_index += 1;
        }
        status((index + 1) * rt_size, total * rt_size);
        Ok(())
    }

    /// `qcow2_change_refcount_order()`.
    pub(crate) fn change_refcount_order(
        &mut self,
        refcount_order: u32,
        status: &mut dyn FnMut(u64, u64),
    ) -> Result<()> {
        assert!(self.qcow_version >= 3);
        assert!(refcount_order <= 6);
        let new_refblock_size = 1u64 << (self.cluster_bits + 3 - refcount_order);
        let new_refcount_bits = 1u32 << refcount_order;
        let mut new_refblock = vec![0u8; self.cluster_size as usize];
        let mut new_reftable: Vec<u64> = Vec::new();
        let mut new_reftable_index = 0u64;
        let mut new_reftable_offset = 0u64;
        let mut allocated_reftable_size = 0u64;
        let mut walk_index = 0u64;

        let ret: Result<()> = (|| {
            loop {
                let mut new_allocation = false;
                // This walk, the one writing the refblocks, and at least one more to see that
                // everything is allocated.
                let total_walks = (walk_index + 2).max(3);
                self.walk_over_reftable(
                    &mut new_reftable,
                    &mut new_reftable_index,
                    &mut new_refblock,
                    new_refblock_size,
                    new_refcount_bits,
                    None,
                    &mut new_allocation,
                    status,
                    walk_index,
                    total_walks,
                )?;
                walk_index += 1;
                new_reftable_index = 0;
                if !new_allocation {
                    break;
                }
                if new_reftable_offset != 0 {
                    self.free_clusters(
                        new_reftable_offset,
                        allocated_reftable_size * REFTABLE_ENTRY_SIZE,
                        DiscardType::Never,
                    );
                }
                new_reftable_offset = self
                    .alloc_clusters(new_reftable.len() as u64 * REFTABLE_ENTRY_SIZE)
                    .map_err(|e| Error::from_io("Failed to allocate the new reftable", e))?;
                allocated_reftable_size = new_reftable.len() as u64;
            }

            let mut new_allocation = false;
            self.walk_over_reftable(
                &mut new_reftable,
                &mut new_reftable_index,
                &mut new_refblock,
                new_refblock_size,
                new_refcount_bits,
                Some(refcount_order),
                &mut new_allocation,
                status,
                walk_index,
                walk_index + 1,
            )?;
            assert!(!new_allocation);

            let bytes = new_reftable.len() as u64 * REFTABLE_ENTRY_SIZE;
            self.pre_write_overlap_check(0, new_reftable_offset, bytes, false)
                .map_err(|e| Error::from_io("Overlap check failed", e))?;
            let mut buf = vec![0u8; bytes as usize];
            for (i, v) in new_reftable.iter().enumerate() {
                set_be64(&mut buf, i, *v);
            }
            self.file
                .pwrite(new_reftable_offset, &buf)
                .map_err(|e| Error::from_io("Failed to write the new reftable", e))?;

            self.cache_flush(CacheId::Refcount)
                .map_err(|e| Error::from_io("Failed to flush the refblock cache", e))?;

            // Point the header at the new table. Only what update_header() uses changes for now,
            // so everything can be restored if it fails.
            let old_order = self.refcount_order;
            let old_offset = self.refcount_table_offset;
            let old_table = std::mem::replace(&mut self.refcount_table, new_reftable.clone());
            self.refcount_order = refcount_order;
            self.refcount_table_offset = new_reftable_offset;
            if let Err(e) = self.update_header() {
                self.refcount_order = old_order;
                self.refcount_table_offset = old_offset;
                self.refcount_table = old_table;
                return Err(Error::from_io("Failed to update the qcow2 header", e));
            }
            self.update_max_refcount_table_index();
            self.refcount_bits = 1 << refcount_order;
            self.refcount_max = 1u64 << (self.refcount_bits - 1);
            self.refcount_max += self.refcount_max - 1;
            self.refcount_block_bits = self.cluster_bits + 3 - refcount_order;
            self.refcount_block_size = 1 << self.refcount_block_bits;
            // The refcount cache holds blocks in the old format; they were flushed above.
            for i in 0..self.refcount_block_cache.size() {
                if self.refcount_block_cache.offset_of(i) != 0 {
                    self.refcount_block_cache.discard(i);
                }
            }

            // Free the old structures below.
            new_reftable = old_table;
            new_reftable_offset = old_offset;
            Ok(())
        })();

        if !new_reftable.is_empty() || new_reftable_offset != 0 {
            // On success these are the old table and its blocks, which is what goes away.
            let table = std::mem::take(&mut new_reftable);
            for v in &table {
                let offset = v & REFT_OFFSET_MASK;
                if offset != 0 {
                    self.free_clusters(offset, self.cluster_size, DiscardType::Other);
                }
            }
            if new_reftable_offset > 0 {
                let size = if ret.is_ok() {
                    table.len() as u64
                } else {
                    allocated_reftable_size.max(table.len() as u64)
                };
                self.free_clusters(
                    new_reftable_offset,
                    size * REFTABLE_ENTRY_SIZE,
                    DiscardType::Other,
                );
            }
        }
        ret
    }

    fn get_refblock_offset(&mut self, offset: u64) -> io::Result<u64> {
        let index = self.offset_to_reftable_index(offset);
        let mut covering = 0;
        if index < self.refcount_table.len() as u64 {
            covering = self.refcount_table[index as usize] & REFT_OFFSET_MASK;
        }
        if covering == 0 {
            self.signal_corruption(
                true,
                -1,
                -1,
                &format!(
                    "Refblock at {offset:#x} is not covered by the refcount structures",
                    offset = Hx(offset)
                ),
            );
            return Err(errno(libc::EIO));
        }
        Ok(covering)
    }

    /// `qcow2_discard_refcount_block()`.
    fn discard_refcount_block(&mut self, discard_block_offs: u64) -> io::Result<()> {
        let cluster_index = discard_block_offs >> self.cluster_bits;
        let block_index = cluster_index & (self.refcount_block_size - 1);
        let refblock_offs = self.get_refblock_offset(discard_block_offs)?;
        assert!(discard_block_offs != 0);
        let slot = self.cache_get(CacheId::Refcount, refblock_offs)?;
        let rc = self.get_refcount_ro(self.refcount_block_cache.table(slot), block_index);
        if rc != 1 {
            let rti = self.offset_to_reftable_index(discard_block_offs);
            self.signal_corruption(
                true,
                -1,
                -1,
                &format!(
                    "Invalid refcount: refblock offset {refblock_offs:#x}, reftable index {rti}, \
                     block offset {discard_block_offs:#x}, refcount {rc:#x}",
                    refblock_offs = Hx(refblock_offs),
                    discard_block_offs = Hx(discard_block_offs),
                    rc = Hx(rc)
                ),
            );
            self.cache_put(CacheId::Refcount, slot);
            return Err(errno(libc::EINVAL));
        }
        let order = self.refcount_order;
        set_refcount(order, self.refcount_block_cache.table_mut(slot), block_index, 0);
        self.refcount_block_cache.mark_dirty(slot);
        self.cache_put(CacheId::Refcount, slot);
        if cluster_index < self.free_cluster_index {
            self.free_cluster_index = cluster_index;
        }
        if let Some(t) = self.refcount_block_cache.is_table_offset(discard_block_offs) {
            self.refcount_block_cache.discard(t);
        }
        self.queue_discard(discard_block_offs, self.cluster_size);
        Ok(())
    }

    /// `qcow2_shrink_reftable()`: drops refcount blocks that no longer count anything.
    pub(crate) fn shrink_reftable(&mut self) -> io::Result<()> {
        let n = self.refcount_table.len();
        let mut tmp = vec![0u64; n];
        for (i, t) in tmp.iter_mut().enumerate() {
            let refblock_offs = self.refcount_table[i] & REFT_OFFSET_MASK;
            if refblock_offs == 0 {
                continue;
            }
            let slot = self.cache_get(CacheId::Refcount, refblock_offs)?;
            let unused = if i as u64 == self.offset_to_reftable_index(refblock_offs) {
                // The block counts itself; ignore that reference.
                let bi = (refblock_offs >> self.cluster_bits) & (self.refcount_block_size - 1);
                let order = self.refcount_order;
                let t = self.refcount_block_cache.table_mut(slot);
                let rc = get_refcount(order, t, bi);
                set_refcount(order, t, bi, 0);
                let z = t.iter().all(|&b| b == 0);
                set_refcount(order, t, bi, rc);
                z
            } else {
                self.refcount_block_cache.table(slot).iter().all(|&b| b == 0)
            };
            self.cache_put(CacheId::Refcount, slot);
            *t = if unused { 0 } else { self.refcount_table[i] };
        }
        let mut buf = vec![0u8; n * 8];
        for (i, v) in tmp.iter().enumerate() {
            set_be64(&mut buf, i, *v);
        }
        let mut ret =
            self.file.pwrite(self.refcount_table_offset, &buf).and_then(|()| self.file.flush());
        // After a failed write the table on disk may be half updated, so forget the blocks in
        // memory either way.
        for (i, &t) in tmp.iter().enumerate() {
            if self.refcount_table[i] != 0 && t == 0 {
                if ret.is_ok() {
                    ret = self.discard_refcount_block(self.refcount_table[i] & REFT_OFFSET_MASK);
                }
                self.refcount_table[i] = 0;
            }
        }
        if !self.cache_discards {
            self.process_discards(ret.is_ok());
        }
        ret
    }

    /// `qcow2_get_last_cluster()`.
    pub(crate) fn get_last_cluster(&mut self, size: u64) -> io::Result<u64> {
        let mut i = self.size_to_clusters(size);
        while i > 0 {
            i -= 1;
            match self.get_refcount(i) {
                Err(e) => {
                    eprintln!(
                        "Can't get refcount for cluster {i}: {}",
                        ruvm_base::error::strerror(&e)
                    );
                    return Err(e);
                }
                Ok(r) if r > 0 => return Ok(i),
                Ok(_) => {}
            }
        }
        self.signal_corruption(true, -1, -1, "There are no references in the refcount table.");
        Err(errno(libc::EIO))
    }
}
