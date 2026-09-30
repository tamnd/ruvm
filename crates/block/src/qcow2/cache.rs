// SPDX-License-Identifier: GPL-2.0-or-later

//! The metadata caches from block/qcow2-cache.c: a fixed number of table slots for L2 tables
//! (or slices of them) and for refcount blocks, written back lazily.
//!
//! QEMU hands out pointers into the cache and takes them back with `qcow2_cache_put()`. Here a
//! `get` returns the slot index instead; the caller reads and writes the table through
//! [`Cache::table`] and [`Cache::table_mut`] and gives the slot back with [`State::cache_put`].
//! A slot with references is never evicted, exactly as in QEMU.

use std::io;

use super::state::{Hx, OL_ACTIVE_L2, OL_REFCOUNT_BLOCK, State};
use crate::node::errno;

/// Which of the two caches of an image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CacheId {
    L2,
    Refcount,
}

/// `Qcow2CachedTable`.
#[derive(Clone, Copy, Debug, Default)]
struct Entry {
    offset: u64,
    lru_counter: u64,
    refs: u32,
    dirty: bool,
}

/// `Qcow2Cache`.
#[derive(Debug)]
pub(crate) struct Cache {
    entries: Vec<Entry>,
    depends: Option<CacheId>,
    table_size: usize,
    depends_on_flush: bool,
    tables: Vec<u8>,
    lru_counter: u64,
    cache_clean_lru_counter: u64,
}

impl Cache {
    /// `qcow2_cache_create()`.
    pub(crate) fn new(num_tables: usize, table_size: usize) -> io::Result<Cache> {
        assert!(num_tables > 0 && table_size.is_power_of_two());
        let bytes = num_tables.checked_mul(table_size).ok_or_else(|| errno(libc::ENOMEM))?;
        let mut tables = Vec::new();
        tables.try_reserve_exact(bytes).map_err(|_| errno(libc::ENOMEM))?;
        tables.resize(bytes, 0);
        Ok(Cache {
            entries: vec![Entry::default(); num_tables],
            depends: None,
            table_size,
            depends_on_flush: false,
            tables,
            lru_counter: 0,
            cache_clean_lru_counter: 0,
        })
    }

    pub(crate) fn size(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn table(&self, i: usize) -> &[u8] {
        &self.tables[i * self.table_size..(i + 1) * self.table_size]
    }

    pub(crate) fn table_mut(&mut self, i: usize) -> &mut [u8] {
        &mut self.tables[i * self.table_size..(i + 1) * self.table_size]
    }

    /// The host offset of the table in slot `i`.
    pub(crate) fn offset_of(&self, i: usize) -> u64 {
        self.entries[i].offset
    }

    /// `qcow2_cache_entry_mark_dirty()`.
    pub(crate) fn mark_dirty(&mut self, i: usize) {
        assert!(self.entries[i].offset != 0);
        self.entries[i].dirty = true;
    }

    /// `qcow2_cache_depends_on_flush()`.
    pub(crate) fn depends_on_flush(&mut self) {
        self.depends_on_flush = true;
    }

    /// `qcow2_cache_is_table_offset()`.
    pub(crate) fn is_table_offset(&self, offset: u64) -> Option<usize> {
        self.entries.iter().position(|e| e.offset == offset)
    }

    /// `qcow2_cache_discard()`.
    pub(crate) fn discard(&mut self, i: usize) {
        assert_eq!(self.entries[i].refs, 0);
        self.entries[i] = Entry::default();
    }

    fn can_clean_entry(&self, i: usize) -> bool {
        let t = &self.entries[i];
        t.refs == 0 && !t.dirty && t.offset != 0 && t.lru_counter <= self.cache_clean_lru_counter
    }

    /// `qcow2_cache_clean_unused()`: forget the clean tables nobody used since the last call.
    pub(crate) fn clean_unused(&mut self) {
        for i in 0..self.entries.len() {
            if self.can_clean_entry(i) {
                self.entries[i].offset = 0;
                self.entries[i].lru_counter = 0;
            }
        }
        self.cache_clean_lru_counter = self.lru_counter;
    }
}

impl State {
    pub(crate) fn cache(&self, id: CacheId) -> &Cache {
        match id {
            CacheId::L2 => &self.l2_table_cache,
            CacheId::Refcount => &self.refcount_block_cache,
        }
    }

    pub(crate) fn cache_mut(&mut self, id: CacheId) -> &mut Cache {
        match id {
            CacheId::L2 => &mut self.l2_table_cache,
            CacheId::Refcount => &mut self.refcount_block_cache,
        }
    }

    fn cache_flush_dependency(&mut self, id: CacheId) -> io::Result<()> {
        if let Some(dep) = self.cache(id).depends {
            self.cache_flush(dep)?;
        }
        let c = self.cache_mut(id);
        c.depends = None;
        c.depends_on_flush = false;
        Ok(())
    }

    /// `qcow2_cache_entry_flush()`.
    fn cache_entry_flush(&mut self, id: CacheId, i: usize) -> io::Result<()> {
        let e = self.cache(id).entries[i];
        if !e.dirty || e.offset == 0 {
            return Ok(());
        }
        if self.cache(id).depends.is_some() {
            self.cache_flush_dependency(id)?;
        } else if self.cache(id).depends_on_flush {
            self.file.flush()?;
            self.cache_mut(id).depends_on_flush = false;
        }
        let ign = match id {
            CacheId::Refcount => OL_REFCOUNT_BLOCK,
            CacheId::L2 => OL_ACTIVE_L2,
        };
        let size = self.cache(id).table_size as u64;
        self.pre_write_overlap_check(ign, e.offset, size, false)?;
        self.file.pwrite(e.offset, self.cache(id).table(i))?;
        self.cache_mut(id).entries[i].dirty = false;
        Ok(())
    }

    /// `qcow2_cache_write()`: write back every dirty table. `ENOSPC` wins over other errors.
    pub(crate) fn cache_write(&mut self, id: CacheId) -> io::Result<()> {
        let mut result: io::Result<()> = Ok(());
        for i in 0..self.cache(id).size() {
            if let Err(e) = self.cache_entry_flush(id, i) {
                let keep = matches!(&result, Err(r) if r.raw_os_error() == Some(libc::ENOSPC));
                if !keep {
                    result = Err(e);
                }
            }
        }
        result
    }

    /// `qcow2_cache_flush()`.
    pub(crate) fn cache_flush(&mut self, id: CacheId) -> io::Result<()> {
        self.cache_write(id)?;
        self.file.flush()
    }

    /// `qcow2_cache_set_dependency()`: tables of `id` may only be written after those of `dep`.
    pub(crate) fn cache_set_dependency(&mut self, id: CacheId, dep: CacheId) -> io::Result<()> {
        if self.cache(dep).depends.is_some() {
            self.cache_flush_dependency(dep)?;
        }
        if self.cache(id).depends.is_some_and(|d| d != dep) {
            self.cache_flush_dependency(id)?;
        }
        self.cache_mut(id).depends = Some(dep);
        Ok(())
    }

    /// `qcow2_cache_empty()`.
    pub(crate) fn cache_empty(&mut self, id: CacheId) -> io::Result<()> {
        self.cache_flush(id)?;
        let c = self.cache_mut(id);
        for e in &mut c.entries {
            assert_eq!(e.refs, 0);
            e.offset = 0;
            e.lru_counter = 0;
        }
        c.lru_counter = 0;
        Ok(())
    }

    fn cache_do_get(
        &mut self,
        id: CacheId,
        offset: u64,
        read_from_disk: bool,
    ) -> io::Result<usize> {
        assert!(offset != 0);
        let table_size = self.cache(id).table_size as u64;
        if offset % table_size != 0 {
            let name = match id {
                CacheId::Refcount => "refcount block",
                CacheId::L2 => "L2 table",
            };
            self.signal_corruption(
                true,
                -1,
                -1,
                &format!(
                    "Cannot get entry from {name} cache: Offset {offset:#x} is unaligned",
                    offset = Hx(offset)
                ),
            );
            return Err(errno(libc::EIO));
        }

        let c = self.cache(id);
        let size = c.size();
        let lookup_index = ((offset / table_size * 4) % size as u64) as usize;
        let mut i = lookup_index;
        let mut min_lru_counter = u64::MAX;
        let mut min_lru_index = None;
        loop {
            let t = &c.entries[i];
            if t.offset == offset {
                let c = self.cache_mut(id);
                c.entries[i].refs += 1;
                return Ok(i);
            }
            if t.refs == 0 && t.lru_counter < min_lru_counter {
                min_lru_counter = t.lru_counter;
                min_lru_index = Some(i);
            }
            i += 1;
            if i == size {
                i = 0;
            }
            if i == lookup_index {
                break;
            }
        }

        // Every slot is in use. The synchronous code never holds that many tables at once.
        let i = min_lru_index.expect("qcow2 metadata cache has no free slot");
        self.cache_entry_flush(id, i)?;
        self.cache_mut(id).entries[i].offset = 0;
        if read_from_disk {
            let file = self.file.clone();
            file.pread(offset, self.cache_mut(id).table_mut(i))?;
        }
        let c = self.cache_mut(id);
        c.entries[i].offset = offset;
        c.entries[i].refs += 1;
        Ok(i)
    }

    /// `qcow2_cache_get()`.
    pub(crate) fn cache_get(&mut self, id: CacheId, offset: u64) -> io::Result<usize> {
        self.cache_do_get(id, offset, true)
    }

    /// `qcow2_cache_get_empty()`: a slot for a table whose contents the caller will fill.
    pub(crate) fn cache_get_empty(&mut self, id: CacheId, offset: u64) -> io::Result<usize> {
        self.cache_do_get(id, offset, false)
    }

    /// `qcow2_cache_put()`.
    pub(crate) fn cache_put(&mut self, id: CacheId, i: usize) {
        let c = self.cache_mut(id);
        assert!(c.entries[i].refs > 0);
        c.entries[i].refs -= 1;
        if c.entries[i].refs == 0 {
            c.lru_counter += 1;
            c.entries[i].lru_counter = c.lru_counter;
        }
    }
}

/// Reads the big endian 64 bit word number `i` of a table.
pub(crate) fn get_be64(t: &[u8], i: usize) -> u64 {
    u64::from_be_bytes(t[i * 8..i * 8 + 8].try_into().unwrap())
}

/// Writes the big endian 64 bit word number `i` of a table.
pub(crate) fn set_be64(t: &mut [u8], i: usize, v: u64) {
    t[i * 8..i * 8 + 8].copy_from_slice(&v.to_be_bytes());
}
