// SPDX-License-Identifier: GPL-2.0-or-later

//! The device quirks of hw/vfio/pci-quirks.c.
//!
//! A quirk is a small region laid over part of a trapped BAR with priority 1, as
//! `vfio_quirk_alloc()` and `memory_region_add_subregion_overlap()` do in QEMU, so that the
//! guest's accesses to a few registers come here first while the rest of the BAR still goes to
//! the device, or to its mmaps. Only the RTL8168 back door to the MSI-X table is here so far.
//! The ATI, NVIDIA and IGD quirks need the VGA regions or the generic window and mirror quirks,
//! and come later.

use std::sync::{Arc, Mutex, Weak};

use ruvm_base::error::Result;
use ruvm_hw_pci::PciDevice;
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, Endian, MemResult, MmioOps};

use super::{Setup, Shared, lock, mem_error};

/// `PCI_VENDOR_ID_REALTEK`.
const PCI_VENDOR_ID_REALTEK: u16 = 0x10ec;

/// The data register of the RTL8168 back door, at BAR2 offset 0x70.
const RTL_DATA: u64 = 0x70;
/// The address register of the RTL8168 back door, at BAR2 offset 0x74.
const RTL_ADDRESS: u64 = 0x74;

/// `vfio_bar_quirk_setup()`: the quirks that apply to BAR `nr` of the device.
pub(super) fn bar_quirk_setup(s: &Setup<'_>, nr: usize) -> Result<()> {
    probe_rtl8168_bar2(s, nr)
}

/// `vfio_pci_is()`.
fn pci_is(s: &Setup<'_>, vendor: u16, device: u16) -> bool {
    s.vendor_id == vendor && s.device_id == device
}

/// Lays `ops` over BAR `nr` at `offset` as a region of `size` bytes with priority 1.
fn add_quirk(
    s: &Setup<'_>,
    nr: usize,
    name: &str,
    offset: u64,
    size: u128,
    ops: Arc<dyn MmioOps>,
) -> Result<()> {
    // QEMU takes the BAR's region as there. A BAR the device does not have has none here, and
    // gets no quirk.
    let Some(bar) = s.regions[nr] else { return Ok(()) };
    let mem = s.bus.memory();
    let region = mem.new_io(name, size, ops).map_err(mem_error)?;
    mem.add_subregion_overlap(bar, offset, region, 1).map_err(mem_error)
}

/// The state of `VFIOrtl8168Quirk`.
#[derive(Debug, Default)]
struct RtlState {
    addr: u32,
    data: u32,
    enabled: bool,
}

/// What a write of the address register turns into.
#[derive(Debug, PartialEq, Eq)]
enum RtlWrite {
    /// The write goes to the device.
    Device,
    /// The latched data goes to the emulated MSI-X table at this offset instead.
    Table(u64, u32),
    /// The write goes nowhere.
    Drop,
}

impl RtlState {
    /// The address register reads back the latched address with bit 31 flipped, to say the
    /// access is done, while a table access is latched. `hw` is what the device read.
    fn address_read(&self, hw: u32) -> u32 {
        if self.enabled { self.addr ^ 0x8000_0000 } else { hw }
    }

    /// A type of 1 in bits 16 to 30 addresses the MSI-X table, and bit 31 asks for a write. A
    /// table write never reaches the device, so the device cannot write its own table with the
    /// guest's data.
    fn address_write(&mut self, data: u32, msix: bool) -> RtlWrite {
        self.enabled = false;
        if data & 0x7fff_0000 == 0x1_0000 {
            self.enabled = true;
            self.addr = data;
            if data & 0x8000_0000 != 0 {
                if msix {
                    return RtlWrite::Table(u64::from(data & 0xfff), self.data);
                }
                return RtlWrite::Drop;
            }
        }
        RtlWrite::Device
    }

    /// The offset in the MSI-X table that the data register reads from instead of the device.
    fn data_read_offset(&self, msix: bool) -> Option<u64> {
        (self.enabled && msix).then_some(u64::from(self.addr & 0xfff))
    }
}

/// `VFIOrtl8168Quirk`, shared by the two registers.
#[derive(Debug)]
struct Rtl8168 {
    shared: Arc<Shared>,
    dev: Weak<PciDevice>,
    /// Where BAR2 is in the device fd.
    offset: u64,
    state: Mutex<RtlState>,
}

impl Rtl8168 {
    fn read(&self, reg: u64) -> u32 {
        self.shared.region_read(2, self.offset, reg, 4) as u32
    }

    fn write(&self, reg: u64, value: u32) {
        self.shared.region_write(2, self.offset, reg, u64::from(value), 4);
    }

    fn msix(&self) -> Option<Arc<PciDevice>> {
        self.dev.upgrade().filter(|d| d.msix_present())
    }
}

/// `vfio_rtl_address_quirk`.
#[derive(Debug)]
struct RtlAddress(Arc<Rtl8168>);

/// `vfio_rtl_data_quirk`.
#[derive(Debug)]
struct RtlData(Arc<Rtl8168>);

impl MmioOps for RtlAddress {
    /// `vfio_rtl8168_quirk_address_read()`: the device is read either way.
    fn read(&self, _cx: &AccessCtx, _addr: u64, _size: AccessSize) -> MemResult<u64> {
        let hw = self.0.read(RTL_ADDRESS);
        Ok(u64::from(lock(&self.0.state).address_read(hw)))
    }

    /// `vfio_rtl8168_quirk_address_write()`.
    fn write(&self, _cx: &AccessCtx, _addr: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        let msix = self.0.msix();
        let w = lock(&self.0.state).address_write(value as u32, msix.is_some());
        match (w, msix) {
            (RtlWrite::Device, _) => self.0.write(RTL_ADDRESS, value as u32),
            (RtlWrite::Table(offset, data), Some(dev)) => {
                dev.msix_table_dispatch_write(offset, 4, u64::from(data), Endian::Little);
            }
            _ => {}
        }
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(4, 4)
    }
}

impl MmioOps for RtlData {
    /// `vfio_rtl8168_quirk_data_read()`: the device is read either way.
    fn read(&self, _cx: &AccessCtx, _addr: u64, _size: AccessSize) -> MemResult<u64> {
        let hw = self.0.read(RTL_DATA);
        let msix = self.0.msix();
        let offset = lock(&self.0.state).data_read_offset(msix.is_some());
        Ok(match (offset, msix) {
            (Some(offset), Some(dev)) => dev.msix_table_dispatch_read(offset, 4, Endian::Little),
            _ => u64::from(hw),
        })
    }

    /// `vfio_rtl8168_quirk_data_write()`: the data is kept for a table write, and goes to the
    /// device too.
    fn write(&self, _cx: &AccessCtx, _addr: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        lock(&self.0.state).data = value as u32;
        self.0.write(RTL_DATA, value as u32);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(4, 4)
    }
}

/// `vfio_probe_rtl8168_bar2_quirk()`.
fn probe_rtl8168_bar2(s: &Setup<'_>, nr: usize) -> Result<()> {
    if !pci_is(s, PCI_VENDOR_ID_REALTEK, 0x8168) || nr != 2 {
        return Ok(());
    }
    let rtl = Arc::new(Rtl8168 {
        shared: Arc::clone(&s.shared),
        dev: Arc::downgrade(s.dev),
        offset: s.bars[nr].offset,
        state: Mutex::default(),
    });
    let address = Arc::new(RtlAddress(Arc::clone(&rtl)));
    add_quirk(s, nr, "vfio-rtl8168-window-address-quirk", RTL_ADDRESS, 4, address)?;
    add_quirk(s, nr, "vfio-rtl8168-window-data-quirk", RTL_DATA, 4, Arc::new(RtlData(rtl)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rtl8168_latches_table_accesses() {
        let mut s = RtlState::default();
        // Nothing latched: the device answers.
        assert_eq!(s.address_read(0x1234), 0x1234);
        assert_eq!(s.data_read_offset(true), None);

        // A table write takes the data written before it, and reads back as done.
        s.data = 0xfee0_1000;
        assert_eq!(s.address_write(0x8001_0000, true), RtlWrite::Table(0, 0xfee0_1000));
        assert_eq!(s.address_read(0x1234), 0x0001_0000);
        // Without MSI-X the write is still kept from the device.
        assert_eq!(s.address_write(0x8001_0004, false), RtlWrite::Drop);
        assert_eq!(s.data_read_offset(false), None);

        // A table read goes to the device, and the data register reads the table.
        assert_eq!(s.address_write(0x0001_f00c, true), RtlWrite::Device);
        assert_eq!(s.address_read(0), 0x8001_f00c);
        assert_eq!(s.data_read_offset(true), Some(0xc));

        // Any other type lets go of the latch.
        assert_eq!(s.address_write(0x8002_0000, true), RtlWrite::Device);
        assert_eq!(s.address_read(0x55), 0x55);
        assert_eq!(s.data_read_offset(true), None);
    }
}
