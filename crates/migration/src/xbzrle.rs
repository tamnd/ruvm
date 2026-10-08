// SPDX-License-Identifier: GPL-2.0-or-later

//! migration/xbzrle.c, migration/page_cache.c and the XBZRLE parts of migration/ram.c: pages
//! sent again as the difference to what went out before.
//!
//! XBZRLE (Xor Based Zero Run Length Encoding) describes a page as runs of bytes that did not
//! change since the copy in the source's page cache and runs that did, the latter with their new
//! bytes. Each run length is a small uleb128 number:
//!
//! ```text
//! page  = zrun nzrun | zrun nzrun page
//! zrun  = length
//! nzrun = length byte...
//! ```
//!
//! A trailing run of unchanged bytes is left out. The destination applies the runs on top of
//! the page it already has, so it needs no cache of its own.
//!
//! The source keeps the cache the way QEMU does: a power of two number of slots indexed by page
//! address, each holding one page and the bitmap sync it was last used in. A page that maps to
//! a slot holding another page that was used within the last two syncs does not push it out.

use std::sync::atomic::{AtomicU64, Ordering};

use ruvm_base::{Error, Result};

/// `ENCODING_FLAG_XBZRLE`, the byte in front of every encoded page.
pub const ENCODING_FLAG_XBZRLE: u8 = 0x1;

/// `CACHED_PAGE_LIFETIME`: syncs during which a cached page is not replaced by another one.
const CACHED_PAGE_LIFETIME: u64 = 2;

/// `uleb128_encode_small()`: writes `n`, at most 0x3fff, in one or two bytes.
fn uleb128_encode_small(out: &mut [u8], n: usize) -> usize {
    debug_assert!(n <= 0x3fff);
    if n < 0x80 {
        out[0] = n as u8;
        1
    } else {
        out[0] = (n & 0x7f) as u8 | 0x80;
        out[1] = (n >> 7) as u8;
        2
    }
}

/// `uleb128_decode_small()`: reads a number of up to 14 bits, with the bytes it took. The
/// caller makes sure there are two bytes to look at.
fn uleb128_decode_small(input: &[u8]) -> Option<(usize, usize)> {
    if input[0] & 0x80 == 0 {
        Some((usize::from(input[0]), 1))
    } else if input[1] & 0x80 != 0 {
        // More than 14 bits.
        None
    } else {
        Some((usize::from(input[0] & 0x7f) | usize::from(input[1]) << 7, 2))
    }
}

/// The result of [`encode_buffer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoded {
    /// The page did not change.
    Unchanged,
    /// The encoding does not fit into the output buffer.
    Overflow,
    /// The encoding took this many bytes.
    Len(usize),
}

/// `xbzrle_encode_buffer()`: encodes `new` against `old`, both of the same length, into `dst`.
pub fn encode_buffer(old: &[u8], new: &[u8], dst: &mut [u8]) -> Encoded {
    let slen = new.len();
    let dlen = dst.len();
    debug_assert_eq!(old.len(), slen);
    let mut d = 0;
    let mut i = 0;
    while i < slen {
        if d + 2 > dlen {
            return Encoded::Overflow;
        }
        let zstart = i;
        while i < slen && old[i] == new[i] {
            i += 1;
        }
        let zrun = i - zstart;
        if zrun == slen {
            return Encoded::Unchanged;
        }
        // The last run of unchanged bytes is not sent.
        if i == slen {
            return Encoded::Len(d);
        }
        d += uleb128_encode_small(&mut dst[d..], zrun);
        if d + 2 > dlen {
            return Encoded::Overflow;
        }
        let nzstart = i;
        while i < slen && old[i] != new[i] {
            i += 1;
        }
        let nzrun = i - nzstart;
        d += uleb128_encode_small(&mut dst[d..], nzrun);
        if d + nzrun > dlen {
            return Encoded::Overflow;
        }
        dst[d..d + nzrun].copy_from_slice(&new[nzstart..i]);
        d += nzrun;
    }
    Encoded::Len(d)
}

/// `xbzrle_decode_buffer()`: applies the runs in `src` to `dst`. Returns the bytes of `dst`
/// covered, or None when the encoding is broken.
pub fn decode_buffer(src: &[u8], dst: &mut [u8]) -> Option<usize> {
    let slen = src.len();
    let dlen = dst.len();
    let mut i = 0;
    let mut d = 0;
    while i < slen {
        // zrun
        if slen - i < 2 {
            return None;
        }
        let (count, n) = uleb128_decode_small(&src[i..])?;
        // Only the first run of unchanged bytes may be empty.
        if i != 0 && count == 0 {
            return None;
        }
        i += n;
        d += count;
        if d > dlen {
            return None;
        }
        // nzrun
        if slen - i < 2 {
            return None;
        }
        let (count, n) = uleb128_decode_small(&src[i..])?;
        if count == 0 {
            return None;
        }
        i += n;
        if d + count > dlen || i + count > slen {
            return None;
        }
        dst[d..d + count].copy_from_slice(&src[i..i + count]);
        d += count;
        i += count;
    }
    Some(d)
}

struct CacheItem {
    addr: u64,
    age: u64,
    data: Option<Box<[u8]>>,
}

/// `PageCache`: the pages the source sent last, by address.
pub struct PageCache {
    items: Vec<CacheItem>,
    page_size: usize,
    size: u64,
}

impl std::fmt::Debug for PageCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PageCache").field("size", &self.size).finish()
    }
}

impl PageCache {
    /// `cache_init()`: a cache of `size` bytes of `page_size` pages. The pages are allocated
    /// as they are first used.
    pub fn new(size: u64, page_size: usize) -> Result<Self> {
        if size < page_size as u64 {
            return Err(Error::generic("cache size is smaller than target page size"));
        }
        let num_pages = size / page_size as u64;
        if !num_pages.is_power_of_two() {
            return Err(Error::generic("number of pages is not a power of two"));
        }
        let items = (0..num_pages).map(|_| CacheItem { addr: u64::MAX, age: 0, data: None });
        Ok(PageCache { items: items.collect(), page_size, size })
    }

    /// The size in bytes the cache was made with.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// `cache_get_cache_pos()`.
    fn pos(&self, addr: u64) -> usize {
        ((addr / self.page_size as u64) & (self.items.len() as u64 - 1)) as usize
    }

    /// `cache_is_cached()`: whether `addr` is in the cache. A hit makes the page young again.
    pub fn is_cached(&mut self, addr: u64, current_age: u64) -> bool {
        let pos = self.pos(addr);
        let it = &mut self.items[pos];
        if it.addr == addr {
            it.age = current_age;
            return true;
        }
        false
    }

    /// `get_cached_data()`: the page cached for `addr`'s slot.
    pub fn get(&mut self, addr: u64) -> Option<&mut [u8]> {
        let pos = self.pos(addr);
        self.items[pos].data.as_deref_mut()
    }

    /// `cache_insert()`: puts a copy of `data` into the slot of `addr`. Fails when another
    /// page that was used recently holds the slot.
    pub fn insert(&mut self, addr: u64, data: &[u8], current_age: u64) -> bool {
        let pos = self.pos(addr);
        let page_size = self.page_size;
        let it = &mut self.items[pos];
        if it.data.is_some() && it.addr != addr && it.age + CACHED_PAGE_LIFETIME > current_age {
            // The cached page is fresh; keep it.
            return false;
        }
        let page = it.data.get_or_insert_with(|| vec![0; page_size].into_boxed_slice());
        page.copy_from_slice(data);
        it.age = current_age;
        it.addr = addr;
        true
    }
}

/// `XBZRLECacheStats`, the counters `query-migrate` shows. Like QEMU's `xbzrle_counters`, they
/// add up over all the migrations of a process.
#[derive(Debug, Default)]
pub struct XbzrleCounters {
    /// Bytes of the encoded pages, without their 8 byte page header.
    pub bytes: AtomicU64,
    /// Pages found in the cache.
    pub pages: AtomicU64,
    /// Pages not found in the cache.
    pub cache_miss: AtomicU64,
    /// Pages whose encoding was larger than the page.
    pub overflow: AtomicU64,
    // The rates are doubles, kept as their bits.
    cache_miss_rate: AtomicU64,
    encoding_rate: AtomicU64,
}

impl XbzrleCounters {
    /// Cache misses per page sent, over the last rate period.
    pub fn cache_miss_rate(&self) -> f64 {
        f64::from_bits(self.cache_miss_rate.load(Ordering::Relaxed))
    }

    /// Bytes of the pages encoded per byte of their encoding, over the last rate period.
    pub fn encoding_rate(&self) -> f64 {
        f64::from_bits(self.encoding_rate.load(Ordering::Relaxed))
    }

    pub(crate) fn set_rates(&self, cache_miss_rate: f64, encoding_rate: f64) {
        self.cache_miss_rate.store(cache_miss_rate.to_bits(), Ordering::Relaxed);
        self.encoding_rate.store(encoding_rate.to_bits(), Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: usize = 4096;

    fn round_trip(old: &[u8], new: &[u8]) -> Encoded {
        let mut enc = vec![0u8; PAGE];
        let r = encode_buffer(old, new, &mut enc);
        if let Encoded::Len(n) = r {
            let mut page = old.to_vec();
            assert!(decode_buffer(&enc[..n], &mut page).is_some());
            assert_eq!(page, new);
        }
        r
    }

    #[test]
    fn encode_decode() {
        let old = vec![0u8; PAGE];
        assert_eq!(round_trip(&old, &old), Encoded::Unchanged);

        // One changed byte in the middle: zrun 100, nzrun 1 and the byte, no trailing zrun.
        let mut new = old.clone();
        new[100] = 7;
        assert_eq!(round_trip(&old, &new), Encoded::Len(3));
        let mut enc = vec![0u8; PAGE];
        encode_buffer(&old, &new, &mut enc);
        assert_eq!(&enc[..3], &[100, 1, 7]);

        // A change at the start has an empty first zrun; long runs take two length bytes.
        let mut new = old.clone();
        new[..200].fill(1);
        new[300] = 2;
        new[PAGE - 1] = 3;
        let r = round_trip(&old, &new);
        assert_eq!(r, Encoded::Len(1 + 2 + 200 + 1 + 1 + 1 + 2 + 1 + 1));

        // Every other byte changed does not fit.
        let mut new = old.clone();
        for b in new.iter_mut().step_by(2) {
            *b = 0xff;
        }
        assert_eq!(round_trip(&old, &new), Encoded::Overflow);

        // A whole new page does not fit either.
        assert_eq!(round_trip(&old, &[9u8; PAGE]), Encoded::Overflow);
    }

    #[test]
    fn decode_errors() {
        let mut page = vec![0u8; PAGE];
        // Fewer than two bytes left for a run length.
        assert_eq!(decode_buffer(&[5], &mut page), None);
        // An empty nzrun.
        assert_eq!(decode_buffer(&[5, 0, 0], &mut page), None);
        // An empty zrun after the first.
        assert_eq!(decode_buffer(&[0, 1, 9, 0, 1, 9], &mut page), None);
        // Past the end of the page.
        assert_eq!(decode_buffer(&[0xff, 0x7f, 1, 9], &mut page), None);
        // Past the end of the input.
        assert_eq!(decode_buffer(&[0, 3, 9], &mut page), None);
        // A length of more than 14 bits.
        assert_eq!(decode_buffer(&[0x80, 0x80, 1], &mut page), None);
        assert_eq!(decode_buffer(&[0, 2, 7, 8], &mut page), Some(2));
        assert_eq!(&page[..3], &[7, 8, 0]);
    }

    #[test]
    fn page_cache() {
        assert_eq!(
            PageCache::new(100, PAGE).unwrap_err().message(),
            "cache size is smaller than target page size"
        );
        assert_eq!(
            PageCache::new(3 * PAGE as u64, PAGE).unwrap_err().message(),
            "number of pages is not a power of two"
        );
        let mut c = PageCache::new(4 * PAGE as u64, PAGE).unwrap();
        let a = [1u8; PAGE];
        assert!(!c.is_cached(0, 1));
        assert!(c.insert(0, &a, 1));
        assert!(c.is_cached(0, 1));
        assert_eq!(c.get(0).unwrap()[0], 1);
        // Page 4 maps to the slot of page 0, which is still fresh.
        let p4 = 4 * PAGE as u64;
        assert!(!c.is_cached(p4, 2));
        assert!(!c.insert(p4, &[2u8; PAGE], 2));
        // Two syncs later it is not.
        assert!(c.insert(p4, &[2u8; PAGE], 3));
        assert!(!c.is_cached(0, 3));
        assert!(c.is_cached(p4, 3));
        assert_eq!(c.get(p4).unwrap()[0], 2);
    }
}
