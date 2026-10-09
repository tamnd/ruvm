// SPDX-License-Identifier: GPL-2.0-or-later

//! `PortioList` from QEMU's system/ioport.c, for the ISA devices here that list their ports
//! with old style byte handlers: the list is cut into one I/O region per run of ports, and
//! each region finds the handler for an access the way `portio_read()` and `portio_write()`
//! do.

use std::fmt;
use std::sync::Arc;

use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MemResult, MmioOps};

/// `MemoryRegionPortio`: `len` ports from `offset`, taking accesses of `size` bytes. The
/// handlers get the port number.
pub(crate) struct Portio<T> {
    pub offset: u32,
    pub len: u32,
    pub size: u32,
    pub read: Option<fn(&T, u32) -> u32>,
    pub write: Option<fn(&T, u32, u32)>,
}

impl<T> Clone for Portio<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Portio<T> {}

/// One region of a list, `MemoryRegionPortioList` with `portio_ops`.
struct PortioRegion<T> {
    dev: Arc<T>,
    /// The port of offset 0 of the region.
    base: u32,
    ports: Vec<Portio<T>>,
}

impl<T> fmt::Debug for PortioRegion<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PortioRegion").field("base", &self.base).finish_non_exhaustive()
    }
}

impl<T> PortioRegion<T> {
    /// `find_portio()`.
    fn find(&self, offset: u32, size: u32, write: bool) -> Option<Portio<T>> {
        self.ports.iter().copied().find(|p| {
            offset >= p.offset
                && offset < p.offset + p.len
                && size == p.size
                && if write { p.write.is_some() } else { p.read.is_some() }
        })
    }

    fn call_read(&self, p: Portio<T>, port: u32) -> u64 {
        p.read.map_or(0, |r| u64::from(r(&self.dev, port)))
    }

    fn call_write(&self, p: Portio<T>, port: u32, val: u32) {
        if let Some(w) = p.write {
            w(&self.dev, port, val);
        }
    }
}

impl<T: Send + Sync> MmioOps for PortioRegion<T> {
    /// `portio_read()`.
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        let off = offset as u32;
        let size = size.bytes();
        if let Some(p) = self.find(off, size, false) {
            return Ok(self.call_read(p, self.base + off));
        }
        if size == 2 {
            if let Some(p) = self.find(off, 1, false) {
                let mut data = self.call_read(p, self.base + off);
                if off + 1 < p.offset + p.len {
                    data |= self.call_read(p, self.base + off + 1) << 8;
                } else {
                    data |= 0xff00;
                }
                return Ok(data);
            }
        }
        Ok((1u64 << (size * 8)) - 1)
    }

    /// `portio_write()`.
    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        let off = offset as u32;
        let size = size.bytes();
        if let Some(p) = self.find(off, size, true) {
            self.call_write(p, self.base + off, value as u32);
        } else if size == 2 {
            if let Some(p) = self.find(off, 1, true) {
                self.call_write(p, self.base + off, (value & 0xff) as u32);
                if off + 1 < p.offset + p.len {
                    self.call_write(p, self.base + off + 1, ((value >> 8) & 0xff) as u32);
                }
            }
        }
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4).allow_unaligned()
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4).allow_unaligned()
    }
}

/// `portio_list_add()`: cuts `list`, sorted by offset, into regions at the holes. Returns the
/// port each region starts at, its size and its ops.
pub(crate) fn portio_list<T: Send + Sync + 'static>(
    dev: &Arc<T>,
    list: &[Portio<T>],
    start: u32,
) -> Vec<(u32, u64, Arc<dyn MmioOps>)> {
    let mut out = Vec::new();
    let mut push = |entries: &[Portio<T>], off_low: u32, off_high: u32| {
        let ports = entries.iter().map(|p| Portio { offset: p.offset - off_low, ..*p }).collect();
        let ops = PortioRegion { dev: Arc::clone(dev), base: start + off_low, ports };
        out.push((
            start + off_low,
            u64::from(off_high - off_low),
            Arc::new(ops) as Arc<dyn MmioOps>,
        ));
    };
    let mut first = 0;
    let mut off_low = list[0].offset;
    let mut off_high = off_low + list[0].len + list[0].size - 1;
    for (i, pio) in list.iter().enumerate().skip(1) {
        let off_last = pio.offset;
        if off_last > off_high {
            push(&list[first..i], off_low, off_high);
            first = i;
            off_low = off_last;
            off_high = off_low + pio.len + pio.size - 1;
        } else if off_last + pio.len > off_high {
            off_high = off_last + pio.len + pio.size - 1;
        }
    }
    push(&list[first..], off_low, off_high);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Dev;

    fn rd(_: &Dev, port: u32) -> u32 {
        port & 0xff
    }

    fn wr(_: &Dev, _: u32, _: u32) {}

    #[test]
    fn the_sb16_list_makes_three_regions() {
        let p = |offset, len, read: bool, write: bool| Portio::<Dev> {
            offset,
            len,
            size: 1,
            read: read.then_some(rd as fn(&Dev, u32) -> u32),
            write: write.then_some(wr as fn(&Dev, u32, u32)),
        };
        let list = [
            p(4, 1, false, true),
            p(5, 1, true, true),
            p(6, 1, true, true),
            p(10, 1, true, false),
            p(12, 1, false, true),
            p(12, 4, true, false),
        ];
        let regions: Vec<(u32, u64)> =
            portio_list(&Arc::new(Dev), &list, 0x220).iter().map(|(b, s, _)| (*b, *s)).collect();
        assert_eq!(regions, [(0x224, 3), (0x22a, 1), (0x22c, 4)]);
    }
}
