// SPDX-License-Identifier: MIT OR Apache-2.0

//! `O_DIRECT`, for files read and written around the host page cache.
//!
//! The flag only exists on some hosts, and its value differs between Linux architectures, so
//! callers ask for it here instead of spelling it out. Direct I/O wants the buffer, the file
//! offset and the length aligned to the logical block size of the file system; that is up to
//! the caller.

/// The `open()` flag for direct I/O, or None on a host that has no such flag.
pub fn o_direct() -> Option<i32> {
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
    return Some(libc::O_DIRECT);
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "freebsd")))]
    return None;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag() {
        let f = o_direct();
        #[cfg(target_os = "linux")]
        assert!(f.is_some_and(|f| f != 0));
        #[cfg(target_os = "macos")]
        assert!(f.is_none());
        let _ = f;
    }
}
