// SPDX-License-Identifier: GPL-2.0-or-later

//! The DMA address spaces of the q35 PCI functions, `pci_setup_iommu()` and
//! `pci_device_iommu_address_space()`.
//!
//! Without a vIOMMU every function reads and writes system memory. A vIOMMU plugged with
//! [`PciDma::set_iommu`] gives each function its own address space over an IOMMU region, the
//! way QEMU's `PCIIOMMUOps.get_address_space` does. QEMU picks the address space of the devices
//! plugged before machine init is done only once it is done, so a [`BusMaster`] looks it up on
//! its first access and keeps it from then on.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

use ruvm_mem::{AddressSpace, IommuOps, MemTxAttrs, MemorySystem};
use ruvm_virtio_queue::{GuestMemory, MemoryError};

/// Makes the IOMMU region callbacks for the function with a requester ID.
pub type IommuFactory = Box<dyn Fn(u16) -> Arc<dyn IommuOps> + Send + Sync>;

/// The vIOMMU of the root bus and the address spaces it handed out.
pub struct PciDma {
    mem: Weak<MemorySystem>,
    memory_as: Weak<AddressSpace>,
    /// The region name stem and the callbacks.
    iommu: OnceLock<(&'static str, IommuFactory)>,
    spaces: Mutex<BTreeMap<u16, Arc<AddressSpace>>>,
    /// Machine init is done, so a function's address space does not change any more.
    ready: AtomicBool,
}

impl std::fmt::Debug for PciDma {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PciDma")
            .field("iommu", &self.iommu.get().map(|i| i.0))
            .field("ready", &self.ready)
            .finish_non_exhaustive()
    }
}

impl PciDma {
    pub fn new(mem: &Arc<MemorySystem>, memory_as: &Arc<AddressSpace>) -> Arc<Self> {
        Arc::new(PciDma {
            mem: Arc::downgrade(mem),
            memory_as: Arc::downgrade(memory_as),
            iommu: OnceLock::new(),
            spaces: Mutex::new(BTreeMap::new()),
            ready: AtomicBool::new(false),
        })
    }

    /// `pci_setup_iommu()`: the functions' DMA goes through regions made by `factory`, named
    /// `<stem>-<bus>-<devfn>`. Returns false if the bus already has a vIOMMU.
    pub fn set_iommu(&self, stem: &'static str, factory: IommuFactory) -> bool {
        self.iommu.set((stem, factory)).is_ok()
    }

    /// Whether a vIOMMU was plugged.
    pub fn has_iommu(&self) -> bool {
        self.iommu.get().is_some()
    }

    /// Called once machine init is done.
    pub fn machine_done(&self) {
        self.ready.store(true, Ordering::Release);
    }

    /// `pci_device_iommu_address_space()` for requester ID `bdf`.
    pub fn address_space(&self, bdf: u16) -> Option<Arc<AddressSpace>> {
        let Some((stem, factory)) = self.iommu.get() else {
            return self.memory_as.upgrade();
        };
        let mut spaces = self.spaces.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(s) = spaces.get(&bdf) {
            return Some(Arc::clone(s));
        }
        let mem = self.mem.upgrade()?;
        let name = format!("{stem}-{}-{}", bdf >> 8, bdf & 0xff);
        let r = mem.new_iommu(&name, 1 << 64, factory(bdf)).ok()?;
        let space = mem.address_space_init(r, &name).ok()?;
        spaces.insert(bdf, Arc::clone(&space));
        Some(space)
    }

    /// The DMA of the function whose requester ID goes in `bdf` once it is known.
    pub fn bus_master(self: &Arc<Self>, bdf: Arc<OnceLock<u16>>) -> BusMaster {
        BusMaster { dma: Arc::downgrade(self), bdf, space: OnceLock::new() }
    }
}

/// One function's DMA, through [`PciDma::address_space`].
#[derive(Debug)]
pub struct BusMaster {
    dma: Weak<PciDma>,
    bdf: Arc<OnceLock<u16>>,
    space: OnceLock<Weak<AddressSpace>>,
}

impl BusMaster {
    fn space(&self) -> Option<Arc<AddressSpace>> {
        if let Some(s) = self.space.get() {
            return s.upgrade();
        }
        let dma = self.dma.upgrade()?;
        let space = dma.address_space(*self.bdf.get()?)?;
        if dma.ready.load(Ordering::Acquire) {
            let _ = self.space.set(Arc::downgrade(&space));
        }
        Some(space)
    }

    fn read_ok(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.space().is_some_and(|a| a.read(addr, MemTxAttrs::UNSPECIFIED, buf).is_ok())
    }

    fn write_ok(&self, addr: u64, buf: &[u8]) -> bool {
        self.space().is_some_and(|a| a.write(addr, MemTxAttrs::UNSPECIFIED, buf).is_ok())
    }
}

impl GuestMemory for BusMaster {
    fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemoryError> {
        if self.read_ok(addr, buf) {
            Ok(())
        } else {
            Err(MemoryError::OutOfRange { addr, len: buf.len() as u64 })
        }
    }

    fn write(&self, addr: u64, buf: &[u8]) -> Result<(), MemoryError> {
        if self.write_ok(addr, buf) {
            Ok(())
        } else {
            Err(MemoryError::OutOfRange { addr, len: buf.len() as u64 })
        }
    }
}

impl ruvm_hw_storage::DmaMemory for BusMaster {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.read_ok(addr, buf)
    }

    fn dma_write(&self, addr: u64, buf: &[u8]) -> bool {
        self.write_ok(addr, buf)
    }
}
