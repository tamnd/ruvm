// SPDX-License-Identifier: GPL-2.0-or-later

//! The storage a drive reads and writes.
//!
//! QEMU's device models talk to a `BlockBackend` through asynchronous AIO calls. The models in
//! this crate complete every request before returning to the guest, so the backend here is a
//! plain synchronous interface: read, write, flush and a length. [`VecBackend`] keeps the disk
//! image in memory, which is what the tests use.

use std::fmt;
use std::io;
use std::sync::{Arc, Mutex, MutexGuard};

/// A disk image, addressed in bytes.
///
/// Requests from the device models are always inside `0..len()`: the models check the range
/// first and report an error to the guest otherwise, as QEMU does.
pub trait BlockBackend: Send + Sync + fmt::Debug {
    /// Fills `buf` with the bytes at `offset`.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()>;

    /// Writes `buf` at `offset`.
    fn write_at(&self, offset: u64, buf: &[u8]) -> io::Result<()>;

    /// Makes earlier writes durable.
    fn flush(&self) -> io::Result<()>;

    /// The size of the image in bytes.
    fn len(&self) -> u64;

    /// Whether the image has no bytes at all.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the image refuses writes, QEMU's `blk_is_writable()` turned around.
    fn is_read_only(&self) -> bool {
        false
    }

    /// Tells the backend that `len` bytes at `offset` are no longer needed. Backends that cannot
    /// release space ignore it, which is what QEMU does with `discard=ignore`.
    fn discard(&self, offset: u64, len: u64) -> io::Result<()> {
        let _ = (offset, len);
        Ok(())
    }

    /// Writes `len` zero bytes at `offset`. The default goes through [`BlockBackend::write_at`]
    /// in chunks.
    fn write_zeroes(&self, offset: u64, len: u64) -> io::Result<()> {
        const CHUNK: u64 = 64 * 1024;
        let zeroes = vec![0u8; CHUNK.min(len) as usize];
        let mut done = 0;
        while done < len {
            let n = CHUNK.min(len - done);
            self.write_at(offset + done, &zeroes[..n as usize])?;
            done += n;
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
struct VecInner {
    data: Vec<u8>,
    flushes: u64,
    failing: bool,
    read_only: bool,
}

/// An in-memory disk image.
///
/// Clones share the same bytes, so a test can keep one handle and give another to the device.
#[derive(Clone, Debug, Default)]
pub struct VecBackend {
    inner: Arc<Mutex<VecInner>>,
}

impl VecBackend {
    /// A zero filled image of `size` bytes.
    pub fn new(size: usize) -> Self {
        Self::from_vec(vec![0; size])
    }

    /// An image holding `data`.
    pub fn from_vec(data: Vec<u8>) -> Self {
        VecBackend { inner: Arc::new(Mutex::new(VecInner { data, ..VecInner::default() })) }
    }

    fn lock(&self) -> MutexGuard<'_, VecInner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// A copy of the whole image.
    pub fn contents(&self) -> Vec<u8> {
        self.lock().data.clone()
    }

    /// Overwrites part of the image directly, bypassing the device.
    pub fn fill(&self, offset: usize, bytes: &[u8]) {
        self.lock().data[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    /// How many times [`BlockBackend::flush`] has been called.
    pub fn flush_count(&self) -> u64 {
        self.lock().flushes
    }

    /// Marks the image read-only, like `-drive readonly=on`.
    pub fn set_read_only(&self, read_only: bool) {
        self.lock().read_only = read_only;
    }

    /// Makes every later request fail with an I/O error, to test error reporting.
    pub fn set_failing(&self, failing: bool) {
        self.lock().failing = failing;
    }
}

fn range(len: usize, offset: u64, count: usize) -> io::Result<std::ops::Range<usize>> {
    let start = usize::try_from(offset).ok().filter(|&s| s <= len && count <= len - s);
    match start {
        Some(s) => Ok(s..s + count),
        None => Err(io::Error::new(io::ErrorKind::InvalidInput, "request past the end")),
    }
}

fn check(inner: &VecInner) -> io::Result<()> {
    if inner.failing {
        return Err(io::Error::other("injected I/O error"));
    }
    Ok(())
}

impl BlockBackend for VecBackend {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let g = self.lock();
        check(&g)?;
        let r = range(g.data.len(), offset, buf.len())?;
        buf.copy_from_slice(&g.data[r]);
        Ok(())
    }

    fn write_at(&self, offset: u64, buf: &[u8]) -> io::Result<()> {
        let mut g = self.lock();
        check(&g)?;
        let r = range(g.data.len(), offset, buf.len())?;
        g.data[r].copy_from_slice(buf);
        Ok(())
    }

    fn flush(&self) -> io::Result<()> {
        let mut g = self.lock();
        check(&g)?;
        g.flushes += 1;
        Ok(())
    }

    fn len(&self) -> u64 {
        self.lock().data.len() as u64
    }

    fn is_read_only(&self) -> bool {
        self.lock().read_only
    }

    fn discard(&self, offset: u64, len: u64) -> io::Result<()> {
        let mut g = self.lock();
        check(&g)?;
        let count =
            usize::try_from(len).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        // Discarded blocks read back as zeros.
        let r = range(g.data.len(), offset, count)?;
        g.data[r].fill(0);
        Ok(())
    }
}
