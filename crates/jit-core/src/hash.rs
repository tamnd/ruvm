// SPDX-License-Identifier: MIT OR Apache-2.0

//! A fast hasher for the maps the translator and the block cache keep, keyed by small integers,
//! temps, op ids and helper names.
//!
//! The standard library's default hasher is SipHash with a random key, which resists
//! collision attacks at a cost of tens of cycles per key. Nothing here is keyed by data an
//! attacker could choose to collide on purpose, beyond what the guest can already do by
//! translating code, so the multiply and rotate hash of rustc's `FxHasher` is enough. The
//! final rotation brings the well mixed high bits of the product down to the low bits the
//! table indexes with, so that keys with zero low bits, such as page addresses, still spread.
//! QEMU uses xxhash for its block hash table and GLib's hashes elsewhere; this is neither.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

/// The multiplier of rustc's `FxHasher`, from the golden ratio.
const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

/// A multiply and rotate hasher; see the module documentation.
#[derive(Clone, Copy, Debug, Default)]
pub struct FastHasher {
    hash: u64,
}

impl FastHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FastHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            let mut w = [0u8; 8];
            w.copy_from_slice(c);
            self.add(u64::from_le_bytes(w));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut w = [0u8; 8];
            w[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(w) ^ ((rest.len() as u64) << 59));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash.rotate_left(26)
    }
}

/// Builds [`FastHasher`]s.
pub type FastBuildHasher = BuildHasherDefault<FastHasher>;

/// A [`HashMap`] with [`FastHasher`].
pub type FastHashMap<K, V> = HashMap<K, V, FastBuildHasher>;

/// A [`HashSet`] with [`FastHasher`].
pub type FastHashSet<K> = HashSet<K, FastBuildHasher>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::BuildHasher;

    #[test]
    fn page_addresses_spread_over_low_bits() {
        let b = FastBuildHasher::default();
        let mut buckets = [0u32; 64];
        for page in 0..4096u64 {
            buckets[(b.hash_one(page << 12) & 63) as usize] += 1;
        }
        // 64 keys per bucket on average; none may be empty or hold a large share.
        assert!(buckets.iter().all(|&n| n > 16 && n < 160), "{buckets:?}");
    }

    #[test]
    fn strings_of_different_lengths_differ() {
        let b = FastBuildHasher::default();
        assert_ne!(b.hash_one("ab"), b.hash_one("ab\0"));
        assert_ne!(b.hash_one("helper_a"), b.hash_one("helper_b"));
        let m: FastHashMap<String, u32> =
            ["x", "yy", "zzz"].iter().enumerate().map(|(i, s)| (s.to_string(), i as u32)).collect();
        assert_eq!(m.get("yy"), Some(&1));
    }
}
