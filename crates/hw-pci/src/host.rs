// SPDX-License-Identifier: GPL-2.0-or-later

//! The host side of config space access, from hw/pci/pci_host.c.
//!
//! Configuration mechanism #1 uses two 32 bit registers, normally at I/O ports 0xcf8 and 0xcfc.
//! The guest writes an address to CONFIG_ADDRESS:
//!
//! - bit 31: enable
//! - bits 23 to 16: bus number
//! - bits 15 to 11: slot
//! - bits 10 to 8: function
//! - bits 7 to 0: register offset
//!
//! and then reads or writes CONFIG_DATA. Byte and word accesses to CONFIG_DATA pick the bytes
//! at `offset + (port & 3)`. With the enable bit clear, or when nothing answers at the
//! address, reads return all ones and writes are dropped.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use ruvm_base::Error;
use ruvm_mem::{AccessCtx, AccessSize, Endian, MemResult, MemorySystem, MmioOps, RegionId};

use crate::bus::PciBus;
use crate::device::PciDevice;
use crate::regs::PCI_CONFIG_SPACE_SIZE;

/// Where the mechanism #1 registers sit in I/O space on a PC.
pub const PCI_HOST_CONFIG_ADDR_PORT: u64 = 0xcf8;
pub const PCI_HOST_CONFIG_DATA_PORT: u64 = 0xcfc;

/// `pci_host_config_write_common()`: a write of `len` bytes at `addr` below `limit`.
pub fn pci_host_config_write_common(dev: &PciDevice, addr: u32, limit: u32, val: u32, len: u32) {
    let limit = adjust_config_limit(dev, limit);
    if limit <= addr || len > 4 || !dev.is_enabled() {
        return;
    }
    dev.config_write(addr, val, len.min(limit - addr));
}

/// `pci_host_config_read_common()`: a read of `len` bytes at `addr` below `limit`. Returns all
/// ones outside the limit, for bad lengths and for disabled functions.
pub fn pci_host_config_read_common(dev: &PciDevice, addr: u32, limit: u32, len: u32) -> u32 {
    let limit = adjust_config_limit(dev, limit);
    if limit <= addr || len > 4 || !dev.is_enabled() {
        return !0;
    }
    dev.config_read(addr, len.min(limit - addr))
}

/// `pci_adjust_config_limit()`, also capped at the function's own config space size.
fn adjust_config_limit(dev: &PciDevice, limit: u32) -> u32 {
    let mut limit = limit;
    let extended = dev.bus().is_some_and(|b| b.allows_extended_config_space());
    if limit > PCI_CONFIG_SPACE_SIZE as u32 && !extended {
        limit = PCI_CONFIG_SPACE_SIZE as u32;
    }
    limit.min(dev.config_size() as u32)
}

/// `pci_dev_find_by_addr()`.
fn find_by_addr(bus: &PciBus, addr: u32) -> Option<Arc<PciDevice>> {
    bus.find_device((addr >> 16) as u8, (addr >> 8) as u8)
}

/// `pci_data_write()`: a config write addressed like CONFIG_ADDRESS (the enable bit is ignored).
pub fn pci_data_write(bus: &PciBus, addr: u32, val: u32, len: u32) {
    let Some(dev) = find_by_addr(bus, addr) else { return };
    let config_addr = addr & (PCI_CONFIG_SPACE_SIZE as u32 - 1);
    pci_host_config_write_common(&dev, config_addr, PCI_CONFIG_SPACE_SIZE as u32, val, len);
}

/// `pci_data_read()`: a config read addressed like CONFIG_ADDRESS. Nothing there reads as all
/// ones.
pub fn pci_data_read(bus: &PciBus, addr: u32, len: u32) -> u32 {
    let Some(dev) = find_by_addr(bus, addr) else { return !0 };
    let config_addr = addr & (PCI_CONFIG_SPACE_SIZE as u32 - 1);
    pci_host_config_read_common(&dev, config_addr, PCI_CONFIG_SPACE_SIZE as u32, len)
}

/// A PCI host bridge's config mechanism state, the `config_reg` part of `PCIHostState`.
pub struct PciHostState {
    bus: Arc<PciBus>,
    config_reg: AtomicU32,
}

impl fmt::Debug for PciHostState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PciHostState")
            .field("bus", &self.bus.name())
            .field("config_reg", &format_args!("{:#010x}", self.config_reg()))
            .finish()
    }
}

impl PciHostState {
    pub fn new(bus: Arc<PciBus>) -> Arc<PciHostState> {
        Arc::new(PciHostState { bus, config_reg: AtomicU32::new(0) })
    }

    pub fn bus(&self) -> &Arc<PciBus> {
        &self.bus
    }

    /// The current CONFIG_ADDRESS value.
    pub fn config_reg(&self) -> u32 {
        self.config_reg.load(Ordering::Relaxed)
    }

    pub fn set_config_reg(&self, v: u32) {
        self.config_reg.store(v, Ordering::Relaxed);
    }

    /// `pci_host_config_write()`: only aligned 32 bit writes update CONFIG_ADDRESS.
    pub fn config_write(&self, addr: u64, val: u64, len: u32) {
        if addr != 0 || len != 4 {
            return;
        }
        self.set_config_reg(val as u32);
    }

    /// `pci_host_config_read()`.
    pub fn config_read(&self, _addr: u64, _len: u32) -> u32 {
        self.config_reg()
    }

    /// `pci_host_data_write()`.
    pub fn data_write(&self, addr: u64, val: u64, len: u32) {
        let reg = self.config_reg();
        if reg & (1 << 31) != 0 {
            pci_data_write(&self.bus, reg | (addr as u32 & 3), val as u32, len);
        }
    }

    /// `pci_host_data_read()`.
    pub fn data_read(&self, addr: u64, len: u32) -> u32 {
        let reg = self.config_reg();
        if reg & (1 << 31) == 0 {
            return 0xffff_ffff;
        }
        pci_data_read(&self.bus, reg | (addr as u32 & 3), len)
    }

    /// CONFIG_ADDRESS as an MMIO region, `pci_host_conf_le_ops` or `pci_host_conf_be_ops`.
    pub fn conf_ops(self: &Arc<Self>, endian: Endian) -> Arc<dyn MmioOps> {
        Arc::new(ConfOps { host: Arc::clone(self), endian })
    }

    /// CONFIG_DATA as an MMIO region, `pci_host_data_le_ops` or `pci_host_data_be_ops`.
    pub fn data_ops(self: &Arc<Self>, endian: Endian) -> Arc<dyn MmioOps> {
        Arc::new(DataOps { host: Arc::clone(self), endian })
    }

    /// Maps CONFIG_ADDRESS at 0xcf8 and CONFIG_DATA at 0xcfc in `io`, as the PC host bridges
    /// do. Returns the two regions.
    pub fn map_ioports(
        self: &Arc<Self>,
        memory: &MemorySystem,
        io: RegionId,
    ) -> Result<(RegionId, RegionId), Error> {
        let err = |e: ruvm_mem::MemError| Error::generic(e.to_string());
        let conf = memory.new_io("pci-conf-idx", 4, self.conf_ops(Endian::Little)).map_err(err)?;
        let data = memory.new_io("pci-conf-data", 4, self.data_ops(Endian::Little)).map_err(err)?;
        memory.add_subregion(io, PCI_HOST_CONFIG_ADDR_PORT, conf).map_err(err)?;
        memory.add_subregion(io, PCI_HOST_CONFIG_DATA_PORT, data).map_err(err)?;
        Ok((conf, data))
    }
}

struct ConfOps {
    host: Arc<PciHostState>,
    endian: Endian,
}

struct DataOps {
    host: Arc<PciHostState>,
    endian: Endian,
}

impl fmt::Debug for ConfOps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PciHostConfOps").field("endian", &self.endian).finish_non_exhaustive()
    }
}

impl fmt::Debug for DataOps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PciHostDataOps").field("endian", &self.endian).finish_non_exhaustive()
    }
}

impl MmioOps for ConfOps {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.host.config_read(offset, size.bytes())))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        self.host.config_write(offset, value, size.bytes());
        Ok(())
    }

    fn endianness(&self) -> Endian {
        self.endian
    }
}

impl MmioOps for DataOps {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.host.data_read(offset, size.bytes())))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        self.host.data_write(offset, value, size.bytes());
        Ok(())
    }

    fn endianness(&self) -> Endian {
        self.endian
    }
}
