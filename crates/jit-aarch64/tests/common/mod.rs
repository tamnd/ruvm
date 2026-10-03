// SPDX-License-Identifier: GPL-2.0-or-later

//! Guest memory backed by the same atomic words as a host window, for the memory ordering
//! tests. Generated code reaches the words directly through the window; accesses it cannot make
//! there, and the atomic helpers, come here.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use ruvm_jit_core::MemOpIdx;
use ruvm_jit_interp::{FaultKind, GuestMemory, MemFault};

/// Guest memory at `base` made of `words`, little endian, as [`ruvm_jit_aarch64::HostWindow`]
/// lays it out. `lock` makes the atomic helpers indivisible against each other.
pub(crate) struct WordMemory<'a> {
    pub(crate) words: &'a [AtomicU64],
    pub(crate) base: u64,
    pub(crate) lock: &'a Mutex<()>,
    pub(crate) guard: Option<MutexGuard<'a, ()>>,
    /// Number of reads and writes made through this memory.
    pub(crate) accesses: &'a AtomicUsize,
}

impl<'a> WordMemory<'a> {
    pub(crate) fn new(
        words: &'a [AtomicU64],
        base: u64,
        lock: &'a Mutex<()>,
        accesses: &'a AtomicUsize,
    ) -> WordMemory<'a> {
        WordMemory { words, base, lock, guard: None, accesses }
    }

    /// The offset of the `len` bytes at `addr`, alignment having been checked by the caller.
    fn check(&self, addr: u64, len: usize, oi: MemOpIdx, write: bool) -> Result<u64, MemFault> {
        let off = addr.wrapping_sub(self.base);
        let size = 8 * self.words.len() as u64;
        if off >= size || size - off < len as u64 {
            return Err(MemFault { addr, write, oi, kind: FaultKind::Unmapped });
        }
        self.accesses.fetch_add(1, Ordering::Relaxed);
        Ok(off)
    }
}

impl GuestMemory for WordMemory<'_> {
    fn read(&mut self, addr: u64, buf: &mut [u8], oi: MemOpIdx) -> Result<(), MemFault> {
        let off = self.check(addr, buf.len(), oi, false)?;
        for (k, b) in buf.iter_mut().enumerate() {
            let at = off as usize + k;
            *b = self.words[at / 8].load(Ordering::SeqCst).to_le_bytes()[at % 8];
        }
        Ok(())
    }

    fn write(&mut self, addr: u64, data: &[u8], oi: MemOpIdx) -> Result<(), MemFault> {
        let off = self.check(addr, data.len(), oi, true)?;
        for (k, b) in data.iter().enumerate() {
            let at = off as usize + k;
            let shift = 8 * (at % 8);
            let _ = self.words[at / 8].fetch_update(Ordering::SeqCst, Ordering::SeqCst, |w| {
                Some(w & !(0xff << shift) | (*b as u64) << shift)
            });
        }
        Ok(())
    }

    fn atomic_begin(&mut self) {
        self.guard = Some(self.lock.lock().unwrap_or_else(|e| e.into_inner()));
    }

    fn atomic_end(&mut self) {
        self.guard = None;
    }
}

/// `n` zeroed words.
pub(crate) fn words(n: usize) -> Vec<AtomicU64> {
    (0..n).map(|_| AtomicU64::new(0)).collect()
}
