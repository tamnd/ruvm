// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host memory for guest RAM.
//!
//! Guest RAM has to be page aligned and have a stable host address so accelerators can map it
//! into the guest, and it must not be touched up front, or a 16 GiB guest costs 16 GiB of RSS
//! before it boots. On Unix it is an anonymous `MAP_NORESERVE` mapping, like
//! `qemu_ram_mmap()` gives a RAM block without a file behind it. Elsewhere it falls back to the
//! heap.
//!
//! Shared RAM, which CPR hands to the next process, is a `MAP_SHARED` mapping of a file
//! descriptor instead: a fresh memfd, or a descriptor that came from the process before.
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
    /// The descriptor behind shared memory.
    #[cfg(unix)]
    fd: Option<std::os::fd::OwnedFd>,
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
        Self::map(len, None)
    }

    /// Maps `len` bytes of a new memfd named `name`, shared, the way `qemu_ram_alloc_internal()`
    /// makes shared RAM with `qemu_memfd_create()`. Only Linux has memfds; elsewhere this fails
    /// and the caller falls back to [`new`](Self::new).
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub fn shared(name: &str, len: usize) -> std::io::Result<Self> {
        let fd = rustix::fs::memfd_create(name, rustix::fs::MemfdFlags::CLOEXEC)?;
        Self::from_fd(fd, len)
    }

    /// Maps `len` bytes of a new memfd named `name`, shared. Only Linux has memfds; elsewhere
    /// this fails and the caller falls back to [`new`](Self::new).
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    pub fn shared(_name: &str, _len: usize) -> std::io::Result<Self> {
        Err(std::io::Error::from(std::io::ErrorKind::Unsupported))
    }

    /// Maps the first `len` bytes of `fd` shared, growing the file first when it is shorter,
    /// as `file_ram_alloc()` does with a descriptor CPR passed on. Whatever else maps the file
    /// sees the same bytes.
    #[cfg(unix)]
    pub fn from_fd(fd: std::os::fd::OwnedFd, len: usize) -> std::io::Result<Self> {
        let file = std::fs::File::from(fd);
        if file.metadata()?.len() < len as u64 {
            file.set_len(len as u64)?;
        }
        Self::map(len, Some(file.into()))
    }

    /// Maps `len` bytes of address space for the whole of a user mode guest, `reserved_va`.
    /// Unlike guest RAM it gets no huge pages, since the guest maps and unmaps it a page at a
    /// time with [`map_fixed`](Self::map_fixed).
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub fn reserve(len: usize) -> std::io::Result<Self> {
        Self::map_with(len, None, false)
    }

    /// `mmap()` of `len` bytes: anonymous and private without `fd`, shared with it.
    #[cfg(unix)]
    fn map(len: usize, fd: Option<std::os::fd::OwnedFd>) -> std::io::Result<Self> {
        Self::map_with(len, fd, true)
    }

    #[cfg(unix)]
    fn map_with(
        len: usize,
        fd: Option<std::os::fd::OwnedFd>,
        hugepage: bool,
    ) -> std::io::Result<Self> {
        use std::os::fd::AsRawFd;

        if len == 0 {
            return Ok(HostMemory { ptr: std::ptr::NonNull::dangling(), len, fd });
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        let anon = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE;
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        let anon = libc::MAP_PRIVATE | libc::MAP_ANON;
        let (flags, raw) = match &fd {
            Some(fd) => (libc::MAP_SHARED, fd.as_raw_fd()),
            None => (anon, -1),
        };
        // SAFETY: a new mapping at an address the kernel picks aliases nothing in this process.
        // A shared one may change under us from another process, which the `AtomicU8` view
        // allows for.
        let p = unsafe {
            libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, flags, raw, 0)
        };
        if p == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        // ram_block_add() asks for transparent huge pages on every RAM block. A huge page
        // takes one fault where small ones take 512, and a page nobody wrote reads from the
        // huge zero page. A missing page in a range registered with userfaultfd still goes
        // to the handler first, so postcopy sees the same faults. A user mode reservation
        // instead stays out of core dumps: walking its terabytes takes the kernel hours.
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            let advice = if hugepage { libc::MADV_HUGEPAGE } else { libc::MADV_DONTDUMP };
            // SAFETY: advice on the mapping just made, which nothing else uses yet; it only
            // changes how the kernel backs or dumps the pages, never their contents.
            unsafe {
                libc::madvise(p, len, advice);
            }
        }
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        let _ = hugepage;
        let ptr = std::ptr::NonNull::new(p.cast::<AtomicU8>()).expect("mmap never returns null");
        Ok(HostMemory { ptr, len, fd })
    }

    /// The descriptor of shared memory, which another process can map.
    #[cfg(unix)]
    pub fn fd(&self) -> Option<std::os::fd::BorrowedFd<'_>> {
        use std::os::fd::AsFd;
        self.fd.as_ref().map(|fd| fd.as_fd())
    }

    /// Whether the memory is a shared mapping of a file descriptor, `qemu_ram_is_shared()`
    /// with `block->fd` set.
    #[cfg(unix)]
    pub fn is_shared(&self) -> bool {
        self.fd.is_some()
    }

    /// Whether the memory is a shared mapping of a file descriptor, which it never is here.
    #[cfg(not(unix))]
    pub fn is_shared(&self) -> bool {
        false
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

    /// Drops the pages of `len` bytes at `offset`, page aligned, so that they read as zero
    /// again, `ram_block_discard_range()`. On Linux that is `MADV_DONTNEED`, which also leaves
    /// them unmapped for userfaultfd to report, or `MADV_REMOVE` for shared memory, which
    /// punches the hole in the file; elsewhere the bytes are zeroed.
    pub fn discard(&self, offset: usize, len: usize) -> std::io::Result<()> {
        self.check_range(offset, len)?;
        if len == 0 {
            return Ok(());
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            let advice = if self.fd.is_some() { libc::MADV_REMOVE } else { libc::MADV_DONTNEED };
            // SAFETY: the range is inside the mapping this value owns, and both kinds of advice
            // only replace its contents with zero fill pages, which every reader through the
            // `AtomicU8` view may observe at any time.
            let ret = unsafe {
                libc::madvise(self.ptr.as_ptr().cast::<u8>().add(offset).cast(), len, advice)
            };
            if ret != 0 {
                return Err(std::io::Error::last_os_error());
            }
        }
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        for b in &self.as_slice()[offset..offset + len] {
            b.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn check_range(&self, offset: usize, len: usize) -> std::io::Result<()> {
        let end = offset.checked_add(len);
        if end.is_none_or(|e| e > self.len) || offset % 4096 != 0 || len % 4096 != 0 {
            return Err(std::io::Error::from(std::io::ErrorKind::InvalidInput));
        }
        Ok(())
    }

    /// Replaces `len` bytes at `offset`, page aligned, with a new mapping: zero pages without
    /// `fd`, or else the bytes of `fd` from `file_off`, copy on write unless `shared`. Shared
    /// zero pages stay shared with the children of a later `fork()`. This is
    /// how a user mode guest's `mmap()` and `munmap()` land in the memory from
    /// [`reserve`](Self::reserve).
    ///
    /// The new pages are always readable and writable, so the `AtomicU8` view stays valid;
    /// the guest's own protection is the emulator's to enforce. Pages past the end of the file
    /// are zero pages instead of the file pages that would raise SIGBUS. A file shrunk later
    /// still raises it, as a mapped file does for any process.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub fn map_fixed(
        &self,
        offset: usize,
        len: usize,
        fd: Option<std::os::fd::RawFd>,
        file_off: u64,
        shared: bool,
    ) -> std::io::Result<()> {
        self.check_range(offset, len)?;
        if self.fd.is_some() || file_off % 4096 != 0 {
            return Err(std::io::Error::from(std::io::ErrorKind::InvalidInput));
        }
        let (file_len, raw) = match fd {
            Some(raw) => {
                // SAFETY: fstat() writes at most one `struct stat`, and the buffer is read only
                // when it succeeded. The descriptor is a plain number here; a closed or wrong one
                // makes fstat() fail with EBADF, which is the answer a user mode guest gets.
                let st = unsafe {
                    let mut st = std::mem::MaybeUninit::<libc::stat>::zeroed();
                    if libc::fstat(raw, st.as_mut_ptr()) != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    st.assume_init()
                };
                let size = st.st_size.max(0) as u64;
                let pages = size.saturating_sub(file_off).div_ceil(4096) * 4096;
                (usize::try_from(pages).unwrap_or(usize::MAX).min(len), raw)
            }
            None => (0, -1),
        };
        let prot = libc::PROT_READ | libc::PROT_WRITE;
        let private = libc::MAP_PRIVATE | libc::MAP_FIXED | libc::MAP_NORESERVE;
        let mut parts = Vec::with_capacity(2);
        if file_len > 0 {
            let flags = if shared { libc::MAP_SHARED | libc::MAP_FIXED } else { private };
            parts.push((offset, file_len, flags, raw));
        }
        if file_len < len {
            let flags = if shared && fd.is_none() {
                libc::MAP_SHARED | libc::MAP_FIXED | libc::MAP_NORESERVE
            } else {
                private
            };
            parts.push((offset + file_len, len - file_len, flags | libc::MAP_ANONYMOUS, -1));
        }
        for (off, l, flags, raw) in parts {
            let file_off = if raw < 0 { 0 } else { file_off as libc::off_t };
            // SAFETY: the range is inside the mapping this value owns (checked above), so
            // MAP_FIXED replaces only our own pages. The new pages are readable and writable
            // like the ones they replace, and the file part ends before the end of the file,
            // so every access through the `AtomicU8` view stays valid; readers only see the
            // contents change, which atomics allow.
            let p = unsafe {
                libc::mmap(
                    self.ptr.as_ptr().cast::<u8>().add(off).cast(),
                    l,
                    prot,
                    flags,
                    raw,
                    file_off,
                )
            };
            if p == libc::MAP_FAILED {
                return Err(std::io::Error::last_os_error());
            }
        }
        Ok(())
    }

    /// Moves the pages of `len` bytes at `from` to `to`, both page aligned and not
    /// overlapping, without copying them, the way `mremap()` moves a user mode guest's
    /// mapping. `from` keeps a mapping of zero pages (`MREMAP_DONTUNMAP`), so the
    /// `AtomicU8` view never has a hole. Fails on kernels before 5.13 or for memory the kernel
    /// cannot move that way; the caller then copies.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub fn move_pages(&self, from: usize, to: usize, len: usize) -> std::io::Result<()> {
        self.check_range(from, len)?;
        self.check_range(to, len)?;
        if self.fd.is_some() || from.max(to) - from.min(to) < len {
            return Err(std::io::Error::from(std::io::ErrorKind::InvalidInput));
        }
        if len == 0 {
            return Ok(());
        }
        let base = self.ptr.as_ptr().cast::<u8>();
        let flags = libc::MREMAP_MAYMOVE | libc::MREMAP_FIXED | libc::MREMAP_DONTUNMAP;
        // SAFETY: both ranges are inside the mapping this value owns and do not overlap.
        // MREMAP_FIXED only replaces our own pages at `to`, and MREMAP_DONTUNMAP leaves `from`
        // mapped, so every byte of the `AtomicU8` view stays valid memory.
        let p = unsafe {
            libc::mremap(
                base.add(from).cast(),
                len,
                len,
                flags,
                base.add(to).cast::<libc::c_void>(),
            )
        };
        if p == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
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
    fn discard_zeroes() {
        let m = HostMemory::new(4 * 4096).unwrap();
        m.as_slice()[4096 + 7].store(9, Ordering::Relaxed);
        m.as_slice()[2 * 4096].store(3, Ordering::Relaxed);
        m.discard(4096, 4096).unwrap();
        assert_eq!(m.as_slice()[4096 + 7].load(Ordering::Relaxed), 0);
        assert_eq!(m.as_slice()[2 * 4096].load(Ordering::Relaxed), 3);
        assert!(m.discard(4096, 4 * 4096).is_err());
        assert!(m.discard(1, 4096).is_err());
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn shared_memory_maps_twice() {
        let m = HostMemory::shared("ruvm-test", 3 * 4096).unwrap();
        assert!(m.is_shared());
        m.as_slice()[4096 + 5].store(42, Ordering::Relaxed);
        // A second mapping of the same file, as the next process gets after CPR, sees the
        // bytes, and a shorter file grows to the size asked for.
        let fd = m.fd().unwrap().try_clone_to_owned().unwrap();
        let again = HostMemory::from_fd(fd, 4 * 4096).unwrap();
        assert_eq!(again.as_slice()[4096 + 5].load(Ordering::Relaxed), 42);
        again.as_slice()[7].store(1, Ordering::Relaxed);
        assert_eq!(m.as_slice()[7].load(Ordering::Relaxed), 1);
        m.discard(4096, 4096).unwrap();
        assert_eq!(again.as_slice()[4096 + 5].load(Ordering::Relaxed), 0);
        assert!(!HostMemory::new(4096).unwrap().is_shared());
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn fixed_mappings_replace_pages() {
        use std::io::Write;
        use std::os::fd::AsRawFd;

        let m = HostMemory::reserve(16 * 4096).unwrap();
        m.as_slice()[4096].store(5, Ordering::Relaxed);
        m.map_fixed(4096, 4096, None, 0, false).unwrap();
        assert_eq!(m.as_slice()[4096].load(Ordering::Relaxed), 0);

        let path = std::env::temp_dir().join(format!("ruvm-hostmem-{}", std::process::id()));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&[7u8; 5000]).unwrap();
        drop(f);
        let f = std::fs::File::open(&path).unwrap();
        // Three pages over a file of two: the third reads as zero instead of raising SIGBUS.
        m.map_fixed(8 * 4096, 3 * 4096, Some(f.as_raw_fd()), 0, false).unwrap();
        std::fs::remove_file(&path).unwrap();
        let s = m.as_slice();
        assert_eq!(s[8 * 4096 + 4999].load(Ordering::Relaxed), 7);
        assert_eq!(s[8 * 4096 + 5000].load(Ordering::Relaxed), 0);
        assert_eq!(s[10 * 4096 + 1].load(Ordering::Relaxed), 0);
        // Private, so writes stay here.
        s[8 * 4096].store(1, Ordering::Relaxed);
        assert!(m.map_fixed(4095, 4096, None, 0, false).is_err());
        assert!(m.map_fixed(15 * 4096, 2 * 4096, None, 0, false).is_err());

        s[2 * 4096 + 3].store(9, Ordering::Relaxed);
        if m.move_pages(2 * 4096, 12 * 4096, 4096).is_ok() {
            assert_eq!(s[12 * 4096 + 3].load(Ordering::Relaxed), 9);
            assert_eq!(s[2 * 4096 + 3].load(Ordering::Relaxed), 0);
        }
        assert!(m.move_pages(0, 4096, 2 * 4096).is_err());
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
