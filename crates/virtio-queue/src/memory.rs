// SPDX-License-Identifier: MIT OR Apache-2.0

//! The guest memory interface the rings are read through, and a simple implementation for tests.

use std::sync::{Mutex, PoisonError};

use crate::error::MemoryError;

/// Byte access to guest physical memory.
///
/// Only [`read`](GuestMemory::read) and [`write`](GuestMemory::write) have to be provided. The
/// integer helpers are little endian, which is what the modern virtio rings use. Both methods take
/// `&self` because guest memory is shared with the vCPUs and other devices, and whatever
/// synchronisation it needs is the implementation's business.
///
/// Ring fields are at most 8 bytes wide and naturally aligned, so an implementation that turns a
/// 2, 4 or 8 byte access into a single load or store gives the driver a consistent view of each
/// field.
pub trait GuestMemory {
    /// Fills `buf` from guest memory starting at `addr`.
    fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemoryError>;

    /// Copies `buf` into guest memory starting at `addr`.
    fn write(&self, addr: u64, buf: &[u8]) -> Result<(), MemoryError>;

    /// Reads a little endian `u16`.
    fn read_u16(&self, addr: u64) -> Result<u16, MemoryError> {
        let mut b = [0; 2];
        self.read(addr, &mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    /// Reads a little endian `u32`.
    fn read_u32(&self, addr: u64) -> Result<u32, MemoryError> {
        let mut b = [0; 4];
        self.read(addr, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    /// Reads a little endian `u64`.
    fn read_u64(&self, addr: u64) -> Result<u64, MemoryError> {
        let mut b = [0; 8];
        self.read(addr, &mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    /// Writes a little endian `u16`.
    fn write_u16(&self, addr: u64, value: u16) -> Result<(), MemoryError> {
        self.write(addr, &value.to_le_bytes())
    }

    /// Writes a little endian `u32`.
    fn write_u32(&self, addr: u64, value: u32) -> Result<(), MemoryError> {
        self.write(addr, &value.to_le_bytes())
    }

    /// Writes a little endian `u64`.
    fn write_u64(&self, addr: u64, value: u64) -> Result<(), MemoryError> {
        self.write(addr, &value.to_le_bytes())
    }
}

impl<T: GuestMemory + ?Sized> GuestMemory for &T {
    fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemoryError> {
        (**self).read(addr, buf)
    }

    fn write(&self, addr: u64, buf: &[u8]) -> Result<(), MemoryError> {
        (**self).write(addr, buf)
    }
}

/// Guest memory backed by one `Vec<u8>`, for tests and examples.
///
/// It covers `base..base + size` and every access outside that range fails. A mutex around the
/// vector makes it usable from several threads, which is enough for tests that run a driver and a
/// device side by side.
#[derive(Debug)]
pub struct VecMemory {
    base: u64,
    data: Mutex<Vec<u8>>,
}

impl VecMemory {
    /// Zeroed memory covering `0..size`.
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self::with_base(0, size)
    }

    /// Zeroed memory covering `base..base + size`.
    #[must_use]
    pub fn with_base(base: u64, size: usize) -> Self {
        Self { base, data: Mutex::new(vec![0; size]) }
    }

    /// The first address covered.
    #[must_use]
    pub fn base(&self) -> u64 {
        self.base
    }

    /// How many bytes are covered.
    #[must_use]
    pub fn size(&self) -> usize {
        self.lock().len()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<u8>> {
        self.data.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn range(
        &self,
        len_total: usize,
        addr: u64,
        len: usize,
    ) -> Result<(usize, usize), MemoryError> {
        let err = MemoryError::OutOfRange { addr, len: len as u64 };
        let start = addr.checked_sub(self.base).ok_or_else(|| err.clone())?;
        let start = usize::try_from(start).map_err(|_| err.clone())?;
        let end = start.checked_add(len).ok_or_else(|| err.clone())?;
        if end > len_total {
            return Err(err);
        }
        Ok((start, end))
    }
}

impl GuestMemory for VecMemory {
    fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemoryError> {
        let data = self.lock();
        let (start, end) = self.range(data.len(), addr, buf.len())?;
        buf.copy_from_slice(&data[start..end]);
        Ok(())
    }

    fn write(&self, addr: u64, buf: &[u8]) -> Result<(), MemoryError> {
        let mut data = self.lock();
        let (start, end) = self.range(data.len(), addr, buf.len())?;
        data[start..end].copy_from_slice(buf);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_are_little_endian() {
        let mem = VecMemory::new(32);
        mem.write_u64(0, 0x0102_0304_0506_0708).unwrap();
        let mut b = [0; 8];
        mem.read(0, &mut b).unwrap();
        assert_eq!(b, [8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(mem.read_u16(0).unwrap(), 0x0708);
        assert_eq!(mem.read_u32(4).unwrap(), 0x0102_0304);
        mem.write_u16(10, 0xabcd).unwrap();
        mem.write_u32(12, 0xdead_beef).unwrap();
        assert_eq!(mem.read_u16(10).unwrap(), 0xabcd);
        assert_eq!(mem.read_u32(12).unwrap(), 0xdead_beef);
    }

    #[test]
    fn accesses_outside_are_refused() {
        let mem = VecMemory::with_base(0x1000, 16);
        assert_eq!(mem.base(), 0x1000);
        assert_eq!(mem.size(), 16);
        assert!(mem.read_u64(0x1008).is_ok());
        assert_eq!(mem.read_u64(0x1009), Err(MemoryError::OutOfRange { addr: 0x1009, len: 8 }));
        assert!(mem.read_u16(0xfff).is_err());
        assert!(mem.write(u64::MAX, &[1, 2]).is_err());
        // An empty access at the very end is fine.
        assert!(mem.read(0x1010, &mut []).is_ok());
    }

    #[test]
    fn references_are_memory_too() {
        fn takes<M: GuestMemory>(m: M) -> u16 {
            m.read_u16(0).unwrap()
        }
        let mem = VecMemory::new(2);
        mem.write_u16(0, 7).unwrap();
        assert_eq!(takes(&mem), 7);
    }
}
