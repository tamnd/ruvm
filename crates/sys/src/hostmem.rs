// SPDX-License-Identifier: MIT OR Apache-2.0

//! Anonymous host memory for guest RAM.
//!
//! Guest RAM has to be page aligned and have a stable host address so accelerators can map it
//! into the guest, and it must not be touched up front, or a 16 GiB guest costs 16 GiB of RSS
//! before it boots. On Unix it is an anonymous `MAP_NORESERVE` mapping, like
//! `qemu_ram_mmap()` gives a RAM block without a file behind it. Elsewhere it falls back to the
//! heap.
//!
//! The memory is handed out as `&[AtomicU8]` and never as `&[u8]`: vCPUs, the kernel and
//! device threads all write it concurrently, and only atomic access keeps that sound.

use std::fmt;
use std::sync::atomic::AtomicU8;

/// A zero filled, page aligned block of host memory.
pub struct HostMemory {
    #[cfg(unix)]
    ptr: std::ptr::NonNull<AtomicU8>,
    #[cfg(not(unix))]
    bytes: Box<[AtomicU8]>,
    len: usize,
}

// SAFETY rationale for both impls: the memory is only reachable through `&[AtomicU8]`, which is
// Send and Sync, and the mapping is owned by this value until drop.
#[cfg(unix)]
// SAFETY: see above.
unsafe impl Send for HostMemory {}
#[cfg(unix)]
// SAFETY: see above.
unsafe impl Sync for HostMemory {}

impl fmt::Debug for HostMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostMemory").field("len", &self.len).finish()
    }
}

impl HostMemory {
    /// Maps `len` bytes of zeroed memory. Pages are only backed once they are touched.
    #[cfg(unix)]
    pub fn new(len: usize) -> std::io::Result<Self> {
        if len == 0 {
            return Ok(HostMemory { ptr: std::ptr::NonNull::dangling(), len });
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE;
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        let flags = libc::MAP_PRIVATE | libc::MAP_ANON;
        // SAFETY: an anonymous mapping at an address the kernel picks aliases nothing.
        let p = unsafe {
            libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, flags, -1, 0)
        };
        if p == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        let ptr = std::ptr::NonNull::new(p.cast::<AtomicU8>()).expect("mmap never returns null");
        Ok(HostMemory { ptr, len })
    }

    /// Allocates `len` bytes of zeroed memory.
    #[cfg(not(unix))]
    pub fn new(len: usize) -> std::io::Result<Self> {
        Ok(HostMemory { bytes: (0..len).map(|_| AtomicU8::new(0)).collect(), len })
    }

    /// The bytes.
    #[cfg(unix)]
    pub fn as_slice(&self) -> &[AtomicU8] {
        // SAFETY: `ptr` is valid for `len` bytes (or dangling with `len` 0) for as long as
        // `self` lives, `AtomicU8` has the layout of `u8`, and fresh anonymous memory is zero,
        // which is a valid `AtomicU8`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// The bytes.
    #[cfg(not(unix))]
    pub fn as_slice(&self) -> &[AtomicU8] {
        &self.bytes
    }

    /// The host address of the first byte, what `KVM_SET_USER_MEMORY_REGION` and friends take.
    pub fn host_addr(&self) -> usize {
        self.as_slice().as_ptr() as usize
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(unix)]
impl Drop for HostMemory {
    fn drop(&mut self) {
        if self.len != 0 {
            // SAFETY: the mapping came from `mmap` in `new` with this length, and no reference
            // into it outlives `self`.
            unsafe { libc::munmap(self.ptr.as_ptr().cast(), self.len) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn zeroed_and_writable() {
        let m = HostMemory::new(1 << 20).unwrap();
        assert_eq!(m.len(), 1 << 20);
        assert_eq!(m.host_addr() % 4096, 0);
        let s = m.as_slice();
        assert!(s.iter().step_by(4096).all(|b| b.load(Ordering::Relaxed) == 0));
        s[12345].store(7, Ordering::Relaxed);
        assert_eq!(s[12345].load(Ordering::Relaxed), 7);
    }

    #[test]
    fn empty() {
        let m = HostMemory::new(0).unwrap();
        assert!(m.is_empty());
        assert!(m.as_slice().is_empty());
    }

    #[test]
    fn large_mappings_are_lazy() {
        // 64 GiB of address space costs nothing until touched.
        #[cfg(target_pointer_width = "64")]
        {
            let m = HostMemory::new(64 << 30).unwrap();
            m.as_slice()[(64 << 30) - 1].store(1, Ordering::Relaxed);
        }
    }
}
