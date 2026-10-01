// SPDX-License-Identifier: GPL-2.0-or-later

//! The executable code buffer, QEMU's `tcg/region.c` allocation reduced to one region, plus
//! `flush_idcache_range`.
//!
//! On macOS the buffer is one `MAP_JIT` mapping. Apple silicon keeps such a mapping either
//! writable or executable per thread, so every write is bracketed by
//! `pthread_jit_write_protect_np(0)` and `pthread_jit_write_protect_np(1)`, and followed by
//! `sys_icache_invalidate`. On Linux the buffer is one RWX mapping and writes are followed by
//! `__clear_cache`. QEMU can also use a split read-write and read-execute mapping when RWX is
//! refused; this port does not, and reports [`BufferError::Map`] instead.
//!
//! The buffer only hands out addresses. Every access to its memory goes through the few methods
//! here, so the unsafe code of the crate stays in this file.

use std::fmt;

/// The buffer could not be set up or used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BufferError {
    /// `mmap` failed; the text is the OS error.
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

// SAFETY: the declaration matches `sys_icache_invalidate` in macOS's `<libkern/OSCacheControl.h>`.
#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn sys_icache_invalidate(start: *mut libc::c_void, len: libc::size_t);
}

// SAFETY: the declaration matches the compiler runtime's `__clear_cache(char *, char *)`.
#[cfg(all(unix, not(target_os = "macos")))]
unsafe extern "C" {
    fn __clear_cache(start: *mut libc::c_char, end: *mut libc::c_char);
}

#[cfg(unix)]
impl CodeBuffer {
    /// Map `size` bytes of executable memory.
    pub fn new(size: usize) -> Result<CodeBuffer, BufferError> {
        let size = size.max(4096).next_multiple_of(4096);
        let prot = libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC;
        #[cfg(target_os = "macos")]
        let flags = libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_JIT;
        #[cfg(not(target_os = "macos"))]
        let flags = libc::MAP_PRIVATE | libc::MAP_ANONYMOUS;
        // SAFETY: an anonymous private mapping at an address of the kernel's choosing does not
        // touch any existing memory; the result is checked below.
        let p = unsafe { libc::mmap(std::ptr::null_mut(), size, prot, flags, -1, 0) };
        if p == libc::MAP_FAILED {
            return Err(BufferError::Map(std::io::Error::last_os_error().to_string()));
        }
        Ok(CodeBuffer { addr: p as usize, size })
    }

    /// Copy `data` to `offset` and make it visible to instruction fetch.
    pub fn write(&self, offset: usize, data: &[u8]) -> Result<(), BufferError> {
        let end = offset.checked_add(data.len()).ok_or(BufferError::OutOfBounds)?;
        if end > self.size {
            return Err(BufferError::OutOfBounds);
        }
        if data.is_empty() {
            return Ok(());
        }
        let dst = (self.addr + offset) as *mut u8;
        jit_write_protect(false);
        // SAFETY: `offset..end` lies inside the mapping, which stays mapped until `self` is
        // dropped, and `data` cannot overlap it because Rust never hands out references into
        // the mapping. Code running in the buffer at the same time is the caller's concern, as
        // it is in QEMU; single word patches go through `patch_u32` instead.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len()) };
        jit_write_protect(true);
        flush_icache(dst as usize, data.len());
        Ok(())
    }

    /// Store one aligned instruction word with a single 32-bit write, so that another thread
    /// executing the word sees either the old or the new one, then flush it. This is
    /// `qatomic_set` plus `flush_idcache_range` in `tb_target_set_jmp_target`.
    pub fn patch_u32(&self, offset: usize, word: u32) -> Result<(), BufferError> {
        if offset % 4 != 0 || offset.checked_add(4).is_none_or(|e| e > self.size) {
            return Err(BufferError::OutOfBounds);
        }
        let p = (self.addr + offset) as *mut u32;
        jit_write_protect(false);
        // SAFETY: the word is inside the mapping and 4-byte aligned; a volatile aligned store is
        // a single write on every host this runs on.
        unsafe { std::ptr::write_volatile(p, word) };
        jit_write_protect(true);
        flush_icache(p as usize, 4);
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

#[cfg(not(unix))]
impl CodeBuffer {
    /// Executable memory is only implemented for unix hosts.
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

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn jit_write_protect(on: bool) {
    // SAFETY: toggling the per-thread JIT write protection has no memory safety effect by itself.
    unsafe { libc::pthread_jit_write_protect_np(on as libc::c_int) };
}

#[cfg(all(unix, not(all(target_os = "macos", target_arch = "aarch64"))))]
fn jit_write_protect(_on: bool) {}

#[cfg(target_os = "macos")]
fn flush_icache(addr: usize, len: usize) {
    // SAFETY: invalidating the instruction cache over mapped memory has no effect on data.
    unsafe { sys_icache_invalidate(addr as *mut libc::c_void, len) };
}

#[cfg(all(unix, not(target_os = "macos")))]
fn flush_icache(addr: usize, len: usize) {
    // SAFETY: cleaning and invalidating caches over mapped memory has no effect on data.
    unsafe { __clear_cache(addr as *mut libc::c_char, (addr + len) as *mut libc::c_char) };
}
