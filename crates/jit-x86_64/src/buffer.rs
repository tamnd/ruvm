// SPDX-License-Identifier: GPL-2.0-or-later

//! The executable code buffer, QEMU's `tcg/region.c` allocation reduced to one region.
//!
//! On unix hosts the buffer is one RWX mapping, with `MAP_JIT` on macOS. On hosts that are not
//! x86-64 the mapping is not executable, since the code is only generated there. On Windows it is one
//! `VirtualAlloc` region with `PAGE_EXECUTE_READWRITE`. x86 keeps instruction fetch coherent
//! with stores, so unlike the AArch64 backend there is no cache flush after a write, which is
//! also why QEMU's `flush_idcache_range` is empty for this host. QEMU can also use a split
//! read-write and read-execute mapping when RWX is refused; this port does not, and reports
//! [`BufferError::Map`] instead.
//!
//! The buffer only hands out addresses. Every access to its memory goes through the few methods
//! here, so the unsafe code of the crate stays in this file and in the runtime's entry call.

use std::fmt;

/// The buffer could not be set up or used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BufferError {
    /// `mmap` or `VirtualAlloc` failed; the text is the OS error.
    Map(String),
    /// Executable memory is not supported on this host.
    Unsupported,
    /// A write would go past the end of the buffer.
    OutOfBounds,
}

impl fmt::Display for BufferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BufferError::Map(e) => write!(f, "cannot map the code buffer: {e}"),
            BufferError::Unsupported => f.write_str("executable memory is not supported here"),
            BufferError::OutOfBounds => f.write_str("write outside the code buffer"),
        }
    }
}

impl std::error::Error for BufferError {}

/// A block of memory that is both written with generated code and executed.
///
/// The address is kept as an integer so that the type is `Send` and `Sync` without an unsafe
/// impl; all accesses are bounds checked against `size`.
#[derive(Debug)]
pub struct CodeBuffer {
    addr: usize,
    size: usize,
}

#[cfg(windows)]
const MEM_COMMIT_RESERVE: u32 = 0x3000;
#[cfg(windows)]
const MEM_RELEASE: u32 = 0x8000;
#[cfg(windows)]
const PAGE_EXECUTE_READWRITE: u32 = 0x40;

// SAFETY: the declarations match `VirtualAlloc` and `VirtualFree` in `<memoryapi.h>`, which
// kernel32 exports with the system calling convention.
#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn VirtualAlloc(addr: *mut u8, size: usize, kind: u32, protect: u32) -> *mut u8;
    fn VirtualFree(addr: *mut u8, size: usize, kind: u32) -> i32;
}

#[cfg(any(unix, windows))]
impl CodeBuffer {
    /// Map `size` bytes of executable memory.
    #[cfg(unix)]
    pub fn new(size: usize) -> Result<CodeBuffer, BufferError> {
        let size = size.max(4096).next_multiple_of(4096);
        // Off an x86-64 host the code is only generated, never run, and an AArch64 macOS host
        // would refuse writes to a `MAP_JIT` mapping without the write protect toggle.
        #[cfg(target_arch = "x86_64")]
        let prot = libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC;
        #[cfg(not(target_arch = "x86_64"))]
        let prot = libc::PROT_READ | libc::PROT_WRITE;
        #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
        let flags = libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_JIT;
        #[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
        let flags = libc::MAP_PRIVATE | libc::MAP_ANON;
        // SAFETY: an anonymous private mapping at an address of the kernel's choosing does not
        // touch any existing memory; the result is checked below.
        let p = unsafe { libc::mmap(std::ptr::null_mut(), size, prot, flags, -1, 0) };
        if p == libc::MAP_FAILED {
            return Err(BufferError::Map(std::io::Error::last_os_error().to_string()));
        }
        Ok(CodeBuffer { addr: p as usize, size })
    }

    /// Allocate `size` bytes of executable memory.
    #[cfg(windows)]
    pub fn new(size: usize) -> Result<CodeBuffer, BufferError> {
        let size = size.max(4096).next_multiple_of(4096);
        // SAFETY: a fresh allocation at an address of the system's choosing does not touch any
        // existing memory; the result is checked below.
        let p = unsafe {
            VirtualAlloc(std::ptr::null_mut(), size, MEM_COMMIT_RESERVE, PAGE_EXECUTE_READWRITE)
        };
        if p.is_null() {
            return Err(BufferError::Map(std::io::Error::last_os_error().to_string()));
        }
        Ok(CodeBuffer { addr: p as usize, size })
    }

    /// Copy `data` to `offset`.
    pub fn write(&self, offset: usize, data: &[u8]) -> Result<(), BufferError> {
        let end = offset.checked_add(data.len()).ok_or(BufferError::OutOfBounds)?;
        if end > self.size {
            return Err(BufferError::OutOfBounds);
        }
        if data.is_empty() {
            return Ok(());
        }
        let dst = (self.addr + offset) as *mut u8;
        // SAFETY: `offset..end` lies inside the mapping, which stays mapped until `self` is
        // dropped, and `data` cannot overlap it because Rust never hands out references into
        // the mapping. Code running in the buffer at the same time is the caller's concern, as
        // it is in QEMU; single word patches go through `patch_u32` instead.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len()) };
        Ok(())
    }

    /// Store one aligned 32-bit word with a single write, so that another thread executing
    /// the instruction holding it sees either the old or the new one. This is the `qatomic_set`
    /// of a jump displacement in `tb_target_set_jmp_target`.
    pub fn patch_u32(&self, offset: usize, word: u32) -> Result<(), BufferError> {
        if offset % 4 != 0 || offset.checked_add(4).is_none_or(|e| e > self.size) {
            return Err(BufferError::OutOfBounds);
        }
        let p = (self.addr + offset) as *mut u32;
        // SAFETY: the word is inside the mapping and 4-byte aligned; a volatile aligned store is
        // a single write on x86.
        unsafe { std::ptr::write_volatile(p, word) };
        Ok(())
    }

    /// Copy `len` bytes at `offset` out of the buffer.
    pub fn read(&self, offset: usize, len: usize) -> Result<Vec<u8>, BufferError> {
        let end = offset.checked_add(len).ok_or(BufferError::OutOfBounds)?;
        if end > self.size {
            return Err(BufferError::OutOfBounds);
        }
        let mut out = vec![0u8; len];
        // SAFETY: the range is inside the mapping, which is always readable.
        unsafe {
            std::ptr::copy_nonoverlapping((self.addr + offset) as *const u8, out.as_mut_ptr(), len)
        };
        Ok(out)
    }
}

#[cfg(not(any(unix, windows)))]
impl CodeBuffer {
    /// Executable memory is only implemented for unix and Windows hosts.
    pub fn new(_size: usize) -> Result<CodeBuffer, BufferError> {
        Err(BufferError::Unsupported)
    }

    /// Never reached: no buffer can be made on this host.
    pub fn write(&self, _offset: usize, _data: &[u8]) -> Result<(), BufferError> {
        Err(BufferError::Unsupported)
    }

    /// Never reached: no buffer can be made on this host.
    pub fn patch_u32(&self, _offset: usize, _word: u32) -> Result<(), BufferError> {
        Err(BufferError::Unsupported)
    }

    /// Never reached: no buffer can be made on this host.
    pub fn read(&self, _offset: usize, _len: usize) -> Result<Vec<u8>, BufferError> {
        Err(BufferError::Unsupported)
    }
}

impl CodeBuffer {
    /// The address of the first byte.
    pub fn addr(&self) -> u64 {
        self.addr as u64
    }

    /// The size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}

#[cfg(unix)]
impl Drop for CodeBuffer {
    fn drop(&mut self) {
        // SAFETY: the mapping was created by `new` with this size and nothing refers to it once
        // the buffer is gone. Compiled blocks hold the region that owns the buffer, so no code
        // in it can still run.
        unsafe { libc::munmap(self.addr as *mut libc::c_void, self.size) };
    }
}

#[cfg(windows)]
impl Drop for CodeBuffer {
    fn drop(&mut self) {
        // SAFETY: the region was allocated by `new` and nothing refers to it once the buffer is
        // gone, as for the unix mapping above.
        unsafe { VirtualFree(self.addr as *mut u8, 0, MEM_RELEASE) };
    }
}
