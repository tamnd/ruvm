// SPDX-License-Identifier: GPL-2.0-or-later

//! The PCI Express host bridge's MMCONFIG (ECAM) window, from hw/pci/pcie_host.c.
//!
//! ECAM gives every function 4 KiB of config space in memory. An address inside the window
//! splits into:
//!
//! - bits 27 to 20: bus number
//! - bits 19 to 12: devfn
//! - bits 11 to 0: register offset
//!
//! Accesses go to the function through [`pci_host_config_read_common`] and
//! [`pci_host_config_write_common`] with the function's own config space size as the limit.
//! Nothing at an address reads as all ones and swallows writes.
//!
//! The window is one MMIO region of up to 256 MiB that the chipset maps into system memory
//! and moves around as the guest programs it, the way `pcie_host_mmcfg_update()` does.
//!
//! Not ported: VMState and QOM registration. The `MCFG` and `mcfg_size` properties are the
//! [`PcieHost::base_addr`] and [`PcieHost::size`] getters.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use ruvm_mem::{
    AccessCtx, AccessSize, Endian, MemError, MemResult, MemorySystem, MmioOps, RegionId,
};

use crate::bus::PciBus;
use crate::host::{pci_host_config_read_common, pci_host_config_write_common};

/// `PCIE_BASE_ADDR_UNMAPPED`: the base address while the window is not mapped.
pub const PCIE_BASE_ADDR_UNMAPPED: u64 = u64::MAX;

/// The largest window, 256 buses.
pub const PCIE_MMCFG_SIZE_MAX: u64 = 1 << 28;
/// The smallest window, one bus.
pub const PCIE_MMCFG_SIZE_MIN: u64 = 1 << 20;

pub const PCIE_MMCFG_BUS_BIT: u32 = 20;
pub const PCIE_MMCFG_BUS_MASK: u64 = 0xff;
pub const PCIE_MMCFG_DEVFN_BIT: u32 = 12;
pub const PCIE_MMCFG_DEVFN_MASK: u64 = 0xff;
pub const PCIE_MMCFG_CONFOFFSET_MASK: u64 = 0xfff;

/// `PCIE_MMCFG_BUS()`.
pub fn pcie_mmcfg_bus(addr: u64) -> u8 {
    ((addr >> PCIE_MMCFG_BUS_BIT) & PCIE_MMCFG_BUS_MASK) as u8
}

/// `PCIE_MMCFG_DEVFN()`.
pub fn pcie_mmcfg_devfn(addr: u64) -> u8 {
    ((addr >> PCIE_MMCFG_DEVFN_BIT) & PCIE_MMCFG_DEVFN_MASK) as u8
}

/// `PCIE_MMCFG_CONFOFFSET()`.
pub fn pcie_mmcfg_confoffset(addr: u64) -> u32 {
    (addr & PCIE_MMCFG_CONFOFFSET_MASK) as u32
}

/// `pcie_mmcfg_data_write()`: an ECAM write at `mmcfg_addr`, relative to the window.
pub fn pcie_mmcfg_data_write(bus: &PciBus, mmcfg_addr: u64, val: u32, len: u32) {
    let Some(dev) = bus.find_device(pcie_mmcfg_bus(mmcfg_addr), pcie_mmcfg_devfn(mmcfg_addr))
    else {
        return;
    };
    let addr = pcie_mmcfg_confoffset(mmcfg_addr);
    let limit = dev.config_size() as u32;
    pci_host_config_write_common(&dev, addr, limit, val, len);
}

/// `pcie_mmcfg_data_read()`: an ECAM read at `mmcfg_addr`, relative to the window.
pub fn pcie_mmcfg_data_read(bus: &PciBus, mmcfg_addr: u64, len: u32) -> u32 {
    let Some(dev) = bus.find_device(pcie_mmcfg_bus(mmcfg_addr), pcie_mmcfg_devfn(mmcfg_addr))
    else {
        return !0;
    };
    let addr = pcie_mmcfg_confoffset(mmcfg_addr);
    let limit = dev.config_size() as u32;
    pci_host_config_read_common(&dev, addr, limit, len)
}

#[derive(Debug)]
struct Mmcfg {
    base_addr: u64,
    size: u64,
}

/// The ECAM part of `PCIExpressHost`: the window region and where it is mapped.
pub struct PcieHost {
    memory: Arc<MemorySystem>,
    system_memory: RegionId,
    mmio: RegionId,
    state: Mutex<Mmcfg>,
}

impl fmt::Debug for PcieHost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.lock();
        f.debug_struct("PcieHost")
            .field("mmio", &self.mmio)
            .field("base_addr", &format_args!("{:#x}", s.base_addr))
            .field("size", &format_args!("{:#x}", s.size))
            .finish_non_exhaustive()
    }
}

impl PcieHost {
    /// `pcie_host_init()`: creates the "pcie-mmcfg-mmio" window for `bus`, unmapped. It will be
    /// mapped into `system_memory`.
    pub fn new(
        memory: Arc<MemorySystem>,
        system_memory: RegionId,
        bus: Arc<PciBus>,
    ) -> Result<Arc<PcieHost>, MemError> {
        let ops = Arc::new(MmcfgOps { bus });
        let mmio = memory.new_io("pcie-mmcfg-mmio", u128::from(PCIE_MMCFG_SIZE_MAX), ops)?;
        Ok(Arc::new(PcieHost {
            memory,
            system_memory,
            mmio,
            state: Mutex::new(Mmcfg { base_addr: PCIE_BASE_ADDR_UNMAPPED, size: 0 }),
        }))
    }

    fn lock(&self) -> MutexGuard<'_, Mmcfg> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The window region.
    pub fn mmio(&self) -> RegionId {
        self.mmio
    }

    /// Where the window is mapped, or [`PCIE_BASE_ADDR_UNMAPPED`]. The `MCFG` property.
    pub fn base_addr(&self) -> u64 {
        self.lock().base_addr
    }

    /// The size last given to the window, the `mcfg_size` property.
    pub fn size(&self) -> u64 {
        self.lock().size
    }

    fn unmap_locked(&self, s: &mut Mmcfg) {
        if s.base_addr != PCIE_BASE_ADDR_UNMAPPED {
            self.memory
                .del_subregion(self.system_memory, self.mmio)
                .expect("the MMCONFIG window is mapped");
            s.base_addr = PCIE_BASE_ADDR_UNMAPPED;
        }
    }

    fn init_locked(&self, s: &mut Mmcfg, size: u64) {
        assert!(size.is_power_of_two(), "MMCONFIG size {size:#x} is not a power of 2");
        assert!(size >= PCIE_MMCFG_SIZE_MIN, "MMCONFIG size {size:#x} too small");
        assert!(size <= PCIE_MMCFG_SIZE_MAX, "MMCONFIG size {size:#x} too large");
        s.size = size;
        self.memory.set_size(self.mmio, u128::from(size)).expect("the MMCONFIG window exists");
    }

    /// `pcie_host_mmcfg_unmap()`.
    pub fn mmcfg_unmap(&self) {
        let mut s = self.lock();
        self.unmap_locked(&mut s);
    }

    /// `pcie_host_mmcfg_init()`: sets the window size, a power of two from 1 MiB to 256 MiB.
    pub fn mmcfg_init(&self, size: u64) {
        let mut s = self.lock();
        self.init_locked(&mut s, size);
    }

    /// `pcie_host_mmcfg_map()`: maps a `size` byte window at `addr`. It must not be mapped
    /// already.
    pub fn mmcfg_map(&self, addr: u64, size: u64) -> Result<(), MemError> {
        let mut s = self.lock();
        self.map_locked(&mut s, addr, size)
    }

    fn map_locked(&self, s: &mut Mmcfg, addr: u64, size: u64) -> Result<(), MemError> {
        self.init_locked(s, size);
        self.memory.add_subregion(self.system_memory, addr, self.mmio)?;
        s.base_addr = addr;
        Ok(())
    }

    /// `pcie_host_mmcfg_update()`: unmaps the window and, if `enable`, maps `size` bytes of it
    /// at `addr`, all in one memory transaction.
    pub fn mmcfg_update(&self, enable: bool, addr: u64, size: u64) {
        let mut s = self.lock();
        let _t = self.memory.transaction();
        self.unmap_locked(&mut s);
        if enable {
            self.map_locked(&mut s, addr, size).expect("the MMCONFIG window is unmapped");
        }
    }
}

/// `pcie_mmcfg_ops`.
struct MmcfgOps {
    bus: Arc<PciBus>,
}

impl fmt::Debug for MmcfgOps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MmcfgOps").field("bus", &self.bus.name()).finish()
    }
}

impl MmioOps for MmcfgOps {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(pcie_mmcfg_data_read(&self.bus, offset, size.bytes())))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        pcie_mmcfg_data_write(&self.bus, offset, value as u32, size.bytes());
        Ok(())
    }

    fn endianness(&self) -> Endian {
        Endian::Little
    }
}
