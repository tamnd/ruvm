// SPDX-License-Identifier: GPL-2.0-or-later

//! GLib's `GHashTable` with string keys, kept only for the order it iterates in.
//!
//! QEMU keeps QOM properties, the type table and the keys a visitor has not consumed yet in
//! `GHashTable`s and walks them in slot order. That order reaches clients: `qom-list` output,
//! `qom-list-types` output and which key an "is unexpected" error names. This table places keys in
//! the same slots GLib does (open addressing, `g_str_hash`, the prime modulus, quadratic probing,
//! tombstones and the in-place resize), so walking it gives the same sequence. The algorithm follows
//! glib/ghash.c from GLib 2.90, which is LGPL-2.1-or-later and so fine to carry in a GPL crate.

use std::fmt;

const MIN_SHIFT: u32 = 3;
const UNUSED: u32 = 0;
const TOMBSTONE: u32 = 1;

const PRIME_MOD: [u32; 32] = [
    1, 2, 3, 7, 13, 31, 61, 127, 251, 509, 1021, 2039, 4093, 8191, 16381, 32749, 65521, 131071,
    262139, 524287, 1048573, 2097143, 4194301, 8388593, 16777213, 33554393, 67108859, 134217689,
    268435399, 536870909, 1073741789, 2147483647,
];

/// `g_str_hash()`, the djb2 hash over signed chars.
pub fn g_str_hash(s: &str) -> u32 {
    let mut h: u32 = 5381;
    for &b in s.as_bytes() {
        h = h.wrapping_shl(5).wrapping_add(h).wrapping_add(b as i8 as i32 as u32);
    }
    h
}

fn is_real(h: u32) -> bool {
    h >= 2
}

/// A string keyed hash table that iterates in GLib's order.
#[derive(Clone)]
pub struct GHashTable<V> {
    size: usize,
    modulo: u32,
    mask: usize,
    nnodes: usize,
    noccupied: usize,
    hashes: Vec<u32>,
    slots: Vec<Option<(String, V)>>,
}

impl<V> Default for GHashTable<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: fmt::Debug> fmt::Debug for GHashTable<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<V> GHashTable<V> {
    /// `g_hash_table_new(g_str_hash, g_str_equal)`.
    pub fn new() -> Self {
        let mut t = GHashTable {
            size: 0,
            modulo: 0,
            mask: 0,
            nnodes: 0,
            noccupied: 0,
            hashes: Vec::new(),
            slots: Vec::new(),
        };
        t.set_shift(MIN_SHIFT);
        t.hashes = vec![UNUSED; t.size];
        t.slots = (0..t.size).map(|_| None).collect();
        t
    }

    pub fn len(&self) -> usize {
        self.nnodes
    }

    pub fn is_empty(&self) -> bool {
        self.nnodes == 0
    }

    fn set_shift(&mut self, shift: u32) {
        assert!(shift <= 31, "adding more entries to hash table would overflow");
        self.size = 1 << shift;
        self.modulo = PRIME_MOD[shift as usize];
        self.mask = self.size - 1;
    }

    fn set_shift_from_size(&mut self, size: usize) {
        let shift = usize::BITS - size.leading_zeros();
        self.set_shift(shift.max(MIN_SHIFT));
    }

    fn hash_to_index(&self, hash: u32) -> usize {
        (hash.wrapping_mul(11) % self.modulo) as usize
    }

    fn key_hash(key: &str) -> u32 {
        let h = g_str_hash(key);
        if is_real(h) { h } else { 2 }
    }

    /// `g_hash_table_lookup_node()`: the slot holding `key`, or the slot an insert would use.
    fn lookup_node(&self, key: &str, hash: u32) -> usize {
        let mut index = self.hash_to_index(hash);
        let mut first_tombstone = None;
        let mut step = 0;
        let mut node_hash = self.hashes[index];
        while node_hash != UNUSED {
            if node_hash == hash {
                if let Some((k, _)) = &self.slots[index] {
                    if k == key {
                        return index;
                    }
                }
            } else if node_hash == TOMBSTONE && first_tombstone.is_none() {
                first_tombstone = Some(index);
            }
            step += 1;
            index = (index + step) & self.mask;
            node_hash = self.hashes[index];
        }
        first_tombstone.unwrap_or(index)
    }

    fn find(&self, key: &str) -> Option<usize> {
        let i = self.lookup_node(key, Self::key_hash(key));
        is_real(self.hashes[i]).then_some(i)
    }

    pub fn get(&self, key: &str) -> Option<&V> {
        self.find(key).and_then(|i| self.slots[i].as_ref().map(|(_, v)| v))
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut V> {
        let i = self.find(key)?;
        self.slots[i].as_mut().map(|(_, v)| v)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.find(key).is_some()
    }

    /// `g_hash_table_insert()`. Replacing a value keeps the slot, and returns the old value.
    pub fn insert(&mut self, key: impl Into<String>, value: V) -> Option<V> {
        let key = key.into();
        let hash = Self::key_hash(&key);
        let i = self.lookup_node(&key, hash);
        let old_hash = self.hashes[i];
        if is_real(old_hash) {
            let slot = self.slots[i].as_mut().expect("real slot has an entry");
            return Some(std::mem::replace(&mut slot.1, value));
        }
        self.hashes[i] = hash;
        self.slots[i] = Some((key, value));
        self.nnodes += 1;
        if old_hash == UNUSED {
            self.noccupied += 1;
            self.maybe_resize();
        }
        None
    }

    /// `g_hash_table_remove()`, which may shrink the table afterwards.
    pub fn remove(&mut self, key: &str) -> Option<V> {
        let v = self.remove_no_resize(key)?;
        self.maybe_resize();
        Some(v)
    }

    /// Removal as `g_hash_table_iter_remove()` does it: a tombstone and no resize.
    pub fn remove_no_resize(&mut self, key: &str) -> Option<V> {
        let i = self.find(key)?;
        self.hashes[i] = TOMBSTONE;
        self.nnodes -= 1;
        self.slots[i].take().map(|(_, v)| v)
    }

    fn maybe_resize(&mut self) {
        let size = self.size;
        let noccupied = self.noccupied;
        if (size > 1 << MIN_SHIFT && (size - 1) / 4 >= self.nnodes)
            || size <= noccupied + noccupied / 16
        {
            self.resize();
        }
    }

    /// `g_hash_table_resize()`, done in place the way GLib does it, because the probe order during
    /// the move decides where colliding keys end up.
    fn resize(&mut self) {
        let old_size = self.size;
        self.set_shift_from_size(self.nnodes + self.nnodes / 3);
        if self.size > old_size {
            self.hashes.resize(self.size, UNUSED);
            self.slots.resize_with(self.size, || None);
        }
        let mut placed = vec![false; self.size.max(old_size)];

        for i in 0..old_size {
            let mut node_hash = self.hashes[i];
            if !is_real(node_hash) {
                self.hashes[i] = UNUSED;
                continue;
            }
            if placed[i] {
                continue;
            }
            self.hashes[i] = UNUSED;
            let mut entry = self.slots[i].take();
            loop {
                let mut index = self.hash_to_index(node_hash);
                let mut step = 0;
                while placed[index] {
                    step += 1;
                    index = (index + step) & self.mask;
                }
                placed[index] = true;
                let replaced = self.hashes[index];
                self.hashes[index] = node_hash;
                if !is_real(replaced) {
                    self.slots[index] = entry;
                    break;
                }
                node_hash = replaced;
                entry = std::mem::replace(&mut self.slots[index], entry);
            }
        }

        if self.size < old_size {
            self.hashes.truncate(self.size);
            self.slots.truncate(self.size);
        }
        self.noccupied = self.nnodes;
    }

    /// Entries in slot order, which is what `g_hash_table_iter_next()` and
    /// `g_hash_table_foreach()` visit.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &V)> {
        self.slots.iter().filter_map(|s| s.as_ref().map(|(k, v)| (k.as_str(), v)))
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.iter().map(|(k, _)| k)
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.iter().map(|(_, v)| v)
    }

    /// The first key in iteration order, as `check_struct` looks at it.
    pub fn first_key(&self) -> Option<&str> {
        self.keys().next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn str_hash_matches_glib() {
        assert_eq!(g_str_hash(""), 5381);
        assert_eq!(g_str_hash("type"), 2090777863);
        assert_eq!(g_str_hash("realized"), 4192236021);
    }

    #[test]
    fn grows_shrinks_and_keeps_everything() {
        let mut t = GHashTable::new();
        for i in 0..1000 {
            assert!(t.insert(format!("k{i}"), i).is_none());
        }
        assert_eq!(t.len(), 1000);
        for i in 0..1000 {
            assert_eq!(t.get(&format!("k{i}")), Some(&i));
        }
        for i in 0..990 {
            assert_eq!(t.remove(&format!("k{i}")), Some(i));
        }
        assert_eq!(t.len(), 10);
        assert_eq!(t.iter().count(), 10);
        for i in 990..1000 {
            assert_eq!(t.get(&format!("k{i}")), Some(&i));
        }
        assert_eq!(t.insert("k995", 5), Some(995));
    }
}
