// SPDX-License-Identifier: GPL-2.0-or-later

//! The buffer scanning helpers of qemu-img.c that `compare`, `convert` and `rebase` share.

/// `BDRV_SECTOR_SIZE`.
pub(crate) const SECTOR: usize = 512;

/// `IO_BUF_SIZE`.
pub(crate) const IO_BUF_SIZE: usize = 2 << 20;

/// `buffer_is_zero()`.
pub(crate) fn is_zero(buf: &[u8]) -> bool {
    buf.iter().all(|&b| b == 0)
}

/// `find_nonzero()`: the start of the first sector of `buf` with a byte that is not zero.
pub(crate) fn find_nonzero(buf: &[u8]) -> Option<usize> {
    let n = buf.len();
    let end = n / SECTOR * SECTOR;
    let mut i = 0;
    while i < end {
        if !is_zero(&buf[i..i + SECTOR]) {
            return Some(i);
        }
        i += SECTOR;
    }
    if i < n && !is_zero(&buf[i..n]) {
        return Some(i);
    }
    None
}

/// `is_allocated_sectors()`: whether the first of the `n` sectors in `buf` holds data, and
/// how many sectors are in the same state, with the end moved to `alignment` sectors where
/// that helps.
pub(crate) fn is_allocated_sectors(
    buf: &[u8],
    n: usize,
    sector_num: u64,
    alignment: usize,
) -> (bool, usize) {
    if n == 0 {
        return (false, 0);
    }
    let zero = is_zero(&buf[..SECTOR]);
    let mut i = 1;
    while i < n {
        if zero != is_zero(&buf[i * SECTOR..(i + 1) * SECTOR]) {
            break;
        }
        i += 1;
    }
    if i == n {
        // The whole buffer is the same, no reason to split it.
        return (!zero, i);
    }
    let mut zero = zero;
    let tail = ((sector_num + i as u64) & (alignment as u64 - 1)) as usize;
    if tail != 0 {
        if zero && i <= tail {
            // The next sector is data and its read-modify-write rewrites this tail anyway.
            zero = false;
        }
        if !zero {
            // Where possible, end the data on an aligned boundary.
            i += alignment - tail;
            i = i.min(n);
        } else {
            // Write zeroes up to the aligned boundary rather than doing a read-modify-write.
            i -= tail;
        }
    }
    (!zero, i)
}

/// `is_allocated_sectors_min()`: like [`is_allocated_sectors`], but after data up to `min`
/// zero sectors in a row count as data, so short holes do not split the writes.
pub(crate) fn is_allocated_sectors_min(
    buf: &[u8],
    n: usize,
    min: usize,
    sector_num: u64,
    alignment: usize,
) -> (bool, usize) {
    let min = min.min(n);
    let (ret, pnum) = is_allocated_sectors(buf, n, sector_num, alignment);
    if !ret {
        return (ret, pnum);
    }
    let mut num_used = pnum;
    let mut off = pnum;
    let mut left = n - pnum;
    let mut sector = sector_num + pnum as u64;
    let mut num_checked = num_used;
    while left > 0 {
        let (ret, pnum) = is_allocated_sectors(&buf[off * SECTOR..], left, sector, alignment);
        off += pnum;
        left -= pnum;
        sector += pnum as u64;
        num_checked += pnum;
        if ret {
            num_used = num_checked;
        } else if pnum >= min {
            break;
        }
    }
    (true, num_used)
}

/// `compare_buffers()`: whether the first chunks differ, and how long the prefix in the same
/// state is. A `chsize` of 0 means sectors.
pub(crate) fn compare_buffers(buf1: &[u8], buf2: &[u8], chsize: usize) -> (bool, usize) {
    let bytes = buf1.len();
    assert!(bytes > 0);
    let chsize = if chsize == 0 { SECTOR } else { chsize };
    let mut i = bytes.min(chsize);
    let res = buf1[..i] != buf2[..i];
    while i < bytes {
        let len = (bytes - i).min(chsize);
        if (buf1[i..i + len] != buf2[i..i + len]) != res {
            break;
        }
        i += len;
    }
    (res, i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonzero() {
        let mut b = vec![0u8; 2048];
        assert_eq!(find_nonzero(&b), None);
        b[1500] = 1;
        assert_eq!(find_nonzero(&b), Some(1024));
        let mut c = vec![0u8; 700];
        c[699] = 1;
        assert_eq!(find_nonzero(&c), Some(512));
    }

    #[test]
    fn allocated() {
        let mut b = vec![0u8; 8 * SECTOR];
        b[3 * SECTOR] = 1;
        assert_eq!(is_allocated_sectors(&b, 8, 0, 1), (false, 3));
        assert_eq!(is_allocated_sectors(&b[3 * SECTOR..], 5, 3, 1), (true, 1));
        // With 8 sector alignment the zeroes before data become data.
        assert_eq!(is_allocated_sectors(&b, 8, 0, 8), (true, 8));
        assert_eq!(is_allocated_sectors_min(&b[3 * SECTOR..], 5, 8, 3, 1), (true, 1));
        b[6 * SECTOR] = 1;
        assert_eq!(is_allocated_sectors_min(&b[3 * SECTOR..], 5, 8, 3, 1), (true, 4));
    }

    #[test]
    fn compare() {
        let a = vec![0u8; 4 * SECTOR];
        let mut b = a.clone();
        assert_eq!(compare_buffers(&a, &b, 0), (false, 4 * SECTOR));
        b[2 * SECTOR + 7] = 1;
        assert_eq!(compare_buffers(&a, &b, 0), (false, 2 * SECTOR));
        assert_eq!(compare_buffers(&a[2 * SECTOR..], &b[2 * SECTOR..], 0), (true, SECTOR));
    }
}
