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
use std::sync::atomic::{AtomicU64, Ordering};

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
    /// The slot of the buffer in [`LIVE`], or `None` if they were all taken.
    slot: Option<usize>,
}

/// The first bytes of the `nop dword [rax + disp32]` that [`fault_target`] looks for in front
/// of a faulting instruction; the displacement says where to go instead.
pub(crate) const FAULT_MARK: [u8; 3] = [0x0f, 0x1f, 0x80];
/// The length of the whole `nop`.
const FAULT_MARK_LEN: u64 = 7;

/// How many buffers [`fault_target`] can know of at once.
const SLOTS: usize = 64;

/// The start and end of each live buffer, 0 for a free slot, read by a signal handler without
/// taking a lock. A slot is taken by setting its start and given back by clearing its end
/// first, so that a reader that sees the same start before and after the end saw a whole
/// pair.
static LIVE: [(AtomicU64, AtomicU64); SLOTS] =
    [const { (AtomicU64::new(0), AtomicU64::new(0)) }; SLOTS];

/// Note a buffer of `size` bytes at `addr` in [`LIVE`].
fn register(addr: u64, size: usize) -> Option<usize> {
    let slot = LIVE.iter().position(|(start, _)| {
        start.compare_exchange(0, addr, Ordering::SeqCst, Ordering::SeqCst).is_ok()
    })?;
    LIVE[slot].1.store(addr + size as u64, Ordering::SeqCst);
    Some(slot)
}

/// Give back a slot taken by [`register`].
fn unregister(slot: Option<usize>) {
    if let Some(slot) = slot {
        LIVE[slot].1.store(0, Ordering::SeqCst);
        LIVE[slot].0.store(0, Ordering::SeqCst);
    }
}

/// Where to resume when the instruction at `pc` faulted, if it is in a code buffer behind a
/// fault mark: the out of line path the mark names, QEMU's `host_signal_handler` reaching the
/// slow path of a user mode access through `cpu_loop_exit_sigsegv`, done here by jumping to
/// the slow path the block already has. Safe to call from a signal handler: it takes no lock
/// and does not allocate.
pub fn fault_target(pc: u64) -> Option<u64> {
    let mark = pc.checked_sub(FAULT_MARK_LEN)?;
    let (start, end) = LIVE.iter().find_map(|(start, end)| {
        let s = start.load(Ordering::SeqCst);
        let e = end.load(Ordering::SeqCst);
        (s != 0 && e != 0 && start.load(Ordering::SeqCst) == s && s <= mark && pc < e)
            .then_some((s, e))
    })?;
    let mut bytes = [0u8; FAULT_MARK_LEN as usize];
    // SAFETY: `mark..pc` lies in a buffer that was mapped when its slot was read. The
    // instruction at `pc` was running, so the buffer holding it is still mapped, and blocks
    // never start with a faulting access, so `mark` is in the same buffer. Buffers are always
    // readable.
    unsafe {
        std::ptr::copy_nonoverlapping(
            mark as usize as *const u8,
            bytes.as_mut_ptr(),
            FAULT_MARK_LEN as usize,
        )
    };
    if bytes[..3] != FAULT_MARK {
        return None;
    }
    let rel = i32::from_le_bytes([bytes[3], bytes[4], bytes[5], bytes[6]]);
    let to = pc.wrapping_add(rel as i64 as u64);
    (start <= to && to < end).then_some(to)
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
        Ok(CodeBuffer { addr: p as usize, size, slot: register(p as u64, size) })
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
        Ok(CodeBuffer { addr: p as usize, size, slot: register(p as u64, size) })
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

    /// Whether [`fault_target`] knows of the buffer. Only so many buffers fit in its table.
    pub fn fault_targets(&self) -> bool {
        self.slot.is_some()
    }
}

#[cfg(unix)]
impl Drop for CodeBuffer {
    fn drop(&mut self) {
        unregister(self.slot);
        // SAFETY: the mapping was created by `new` with this size and nothing refers to it once
        // the buffer is gone. Compiled blocks hold the region that owns the buffer, so no code
        // in it can still run.
        unsafe { libc::munmap(self.addr as *mut libc::c_void, self.size) };
    }
}

#[cfg(windows)]
impl Drop for CodeBuffer {
    fn drop(&mut self) {
        unregister(self.slot);
        // SAFETY: the region was allocated by `new` and nothing refers to it once the buffer is
        // gone, as for the unix mapping above.
        unsafe { VirtualFree(self.addr as *mut u8, 0, MEM_RELEASE) };
    }
}
