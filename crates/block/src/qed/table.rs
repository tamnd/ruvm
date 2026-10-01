// SPDX-License-Identifier: GPL-2.0-or-later

//! The L1 and L2 tables of a QED image, the L2 table cache and the cluster lookup: block/qed-table.c,
//! block/qed-l2-cache.c and block/qed-cluster.c.
//!
//! QEMU counts references to cached L2 tables so that a request keeps its table while other
//! requests evict it. Requests here run one at a time under the driver lock, so the cache holds
//! plain tables and a request that needs its table again looks it up by offset, reading it back
//! from the image if it was evicted in the meantime. Every change to a table is written through,
//! so the table read back is the same.

use std::io;

use ruvm_qapi::types::BlkdebugEvent;

use crate::node::{BDRV_SECTOR_SIZE, Node};

use super::State;

/// `MAX_L2_CACHE_SIZE`.
const MAX_L2_CACHE_SIZE: usize = 50;

/// `CachedL2Table`.
pub(super) struct CachedL2Table {
    /// Where the table is in the image file.
    pub offset: u64,
    /// The entries, in host byte order.
    pub table: Vec<u64>,
}

/// `L2TableCache`: the most recently loaded L2 tables, oldest first.
#[derive(Default)]
pub(super) struct L2Cache {
    entries: Vec<CachedL2Table>,
}

impl L2Cache {
    /// `qed_find_l2_cache_entry()`.
    pub(super) fn find(&mut self, offset: u64) -> Option<&mut Vec<u64>> {
        self.entries.iter_mut().find(|e| e.offset == offset).map(|e| &mut e.table)
    }

    /// `qed_commit_l2_cache_entry()`: adds `entry` unless a table at the same offset is cached
    /// already, in which case the cached one stays and `entry` is dropped. A full cache drops
    /// its oldest table first.
    pub(super) fn commit(&mut self, entry: CachedL2Table) {
        if self.entries.iter().any(|e| e.offset == entry.offset) {
            return;
        }
        if self.entries.len() >= MAX_L2_CACHE_SIZE {
            // Nothing holds a reference between requests, so the oldest entry can always go.
            self.entries.remove(0);
        }
        self.entries.push(entry);
    }

    /// `qed_read_l2_table()`: the L2 table at `offset`, from the cache or read from `file`
    /// and cached.
    pub(super) fn read(
        &mut self,
        file: &Node,
        offset: u64,
        nelems: usize,
    ) -> io::Result<&mut Vec<u64>> {
        let pos = match self.entries.iter().position(|e| e.offset == offset) {
            Some(pos) => pos,
            None => {
                let mut table = vec![0u64; nelems];
                file.debug_event(BlkdebugEvent::L2Load);
                read_table(file, offset, &mut table)?;
                self.commit(CachedL2Table { offset, table });
                self.entries.len() - 1
            }
        };
        Ok(&mut self.entries[pos].table)
    }
}

/// `qed_read_table()`: reads the whole table at `offset` into `table`.
pub(super) fn read_table(file: &Node, offset: u64, table: &mut [u64]) -> io::Result<()> {
    let mut buf = vec![0u8; table.len() * 8];
    file.pread(offset, &mut buf)?;
    for (e, b) in table.iter_mut().zip(buf.chunks_exact(8)) {
        *e = u64::from_le_bytes(b.try_into().expect("chunks of 8"));
    }
    Ok(())
}

/// `qed_write_table()`: writes `n` entries of `table` from `index` to the table at `offset`,
/// widened to whole sectors. With `flush`, the file is flushed afterwards.
pub(super) fn write_table(
    file: &Node,
    offset: u64,
    table: &[u64],
    index: usize,
    n: usize,
    flush: bool,
) -> io::Result<()> {
    let sector_mask = BDRV_SECTOR_SIZE as usize / 8 - 1;
    let start = index & !sector_mask;
    let end = (index + n + sector_mask) & !sector_mask;
    let mut buf = Vec::with_capacity((end - start) * 8);
    for e in &table[start..end] {
        buf.extend_from_slice(&e.to_le_bytes());
    }
    file.pwrite(offset + start as u64 * 8, &buf)?;
    if flush {
        // QEMU flushes the QED node, which flushes the file under it. The driver lock is
        // held here, so flush the file directly.
        file.flush()?;
    }
    Ok(())
}

/// What `qed_find_cluster()` found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Cluster {
    /// `QED_CLUSTER_FOUND`: data clusters in the image file.
    Found,
    /// `QED_CLUSTER_ZERO`: zero clusters.
    Zero,
    /// `QED_CLUSTER_L2`: the L2 table has no clusters there.
    L2,
    /// `QED_CLUSTER_L1`: there is no L2 table.
    L1,
}

/// `qed_offset_is_unalloc_cluster()`.
pub(super) fn is_unalloc_cluster(offset: u64) -> bool {
    offset == 0
}

/// `qed_offset_is_zero_cluster()`.
pub(super) fn is_zero_cluster(offset: u64) -> bool {
    offset == 1
}

/// `qed_count_contiguous_clusters()`: how many of the `n` entries from `index` are like the
/// first (unallocated, zero, or data clusters that follow each other in the file).
fn count_contiguous_clusters(
    cluster_size: u64,
    nelems: usize,
    table: &[u64],
    index: usize,
    n: usize,
) -> usize {
    let end = (index + n).min(nelems);
    let mut last = table[index];
    let mut i = index + 1;
    while i < end {
        if is_unalloc_cluster(last) {
            if !is_unalloc_cluster(table[i]) {
                break;
            }
        } else if is_zero_cluster(last) {
            if !is_zero_cluster(table[i]) {
                break;
            }
        } else {
            if table[i] != last.wrapping_add(cluster_size) {
                break;
            }
            last = table[i];
        }
        i += 1;
    }
    i - index
}

impl State {
    /// `qed_read_l1_table_sync()`.
    pub(super) fn read_l1_table(&mut self, file: &Node) -> io::Result<()> {
        read_table(file, self.header.l1_table_offset, &mut self.l1_table)
    }

    /// `qed_write_l1_table()`.
    pub(super) fn write_l1_table(&self, file: &Node, index: usize, n: usize) -> io::Result<()> {
        file.debug_event(BlkdebugEvent::L1Update);
        write_table(file, self.header.l1_table_offset, &self.l1_table, index, n, false)
    }

    /// `qed_read_l2_table()` for the table at `offset`.
    pub(super) fn read_l2_table(&mut self, file: &Node, offset: u64) -> io::Result<&mut Vec<u64>> {
        let nelems = self.table_nelems as usize;
        self.l2_cache.read(file, offset, nelems)
    }

    /// `qed_find_cluster()`: what backs the guest range of `len` bytes at `pos`. Returns the
    /// kind of cluster, its offset in the image file (for [`Cluster::Found`]) and how many
    /// bytes from `pos` are like that, never crossing into the range of another L2 table.
    pub(super) fn find_cluster(
        &mut self,
        file: &Node,
        pos: u64,
        len: u64,
    ) -> io::Result<(Cluster, u64, u64)> {
        // Requests are broken up at the L2 boundary so that a request acts on one L2 table
        // at a time.
        let mut len = len.min((((pos >> self.l1_shift) + 1) << self.l1_shift) - pos);

        let l2_offset = self.l1_table[self.l1_index(pos)];
        if is_unalloc_cluster(l2_offset) {
            return Ok((Cluster::L1, 0, len));
        }
        if !self.check_table_offset(l2_offset) {
            return Err(crate::node::errno(libc::EINVAL));
        }

        let index = self.l2_index(pos);
        let n = self.bytes_to_clusters(self.offset_into_cluster(pos) + len) as usize;
        let cluster_size = u64::from(self.header.cluster_size);
        let nelems = self.table_nelems as usize;
        let table = self.read_l2_table(file, l2_offset)?;
        let n = count_contiguous_clusters(cluster_size, nelems, table, index, n);
        let offset = table[index];

        let ret = if is_unalloc_cluster(offset) {
            Cluster::L2
        } else if is_zero_cluster(offset) {
            Cluster::Zero
        } else if self.check_cluster_offset(offset) {
            Cluster::Found
        } else {
            return Err(crate::node::errno(libc::EINVAL));
        };

        len =
            len.min(n as u64 * u64::from(self.header.cluster_size) - self.offset_into_cluster(pos));
        Ok((ret, offset, len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(offset: u64) -> CachedL2Table {
        CachedL2Table { offset, table: vec![offset; 4] }
    }

    #[test]
    fn cache_keeps_the_first_entry_and_drops_the_oldest() {
        let mut c = L2Cache::default();
        c.commit(entry(0x1000));
        c.commit(CachedL2Table { offset: 0x1000, table: vec![7; 4] });
        assert_eq!(c.find(0x1000).unwrap()[0], 0x1000);
        for i in 1..MAX_L2_CACHE_SIZE as u64 + 1 {
            c.commit(entry(0x1000 * (i + 1)));
        }
        assert_eq!(c.entries.len(), MAX_L2_CACHE_SIZE);
        assert!(c.find(0x1000).is_none());
        assert!(c.find(0x2000).is_some());
    }
}
