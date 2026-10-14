// SPDX-License-Identifier: GPL-2.0-or-later

//! The DMA address spaces of the q35 PCI functions, `pci_setup_iommu()` and
//! `pci_device_iommu_address_space()`.
//!
//! Without a vIOMMU every function reads and writes system memory. A vIOMMU plugged with
//! [`PciDma::set_iommu`] gives each function its own address space over an IOMMU region, the
//! way QEMU's `PCIIOMMUOps.get_address_space` does. QEMU picks the address space of the devices
//! plugged before machine init is done only once it is done, so a [`BusMaster`] looks it up on
//! its first access and keeps it from then on.
//!
//! An IOMMU that can be off for a function, as the Intel one is, also sets
//! [`PciDma::set_iommu_bypass`]. Each address space then has a container over the IOMMU region
//! and an alias of system memory, and [`PciDma::switch`] shows one of the two, the way
//! `vtd_switch_address_space()` does. Translating every access while the IOMMU is off works
//! too, but VFIO needs the plain RAM to map it.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

use ruvm_mem::{AddressSpace, IommuOps, MemTxAttrs, MemorySystem, RegionId};
use ruvm_virtio_queue::{GuestMemory, MemoryError};

/// Makes the IOMMU region callbacks for the function with a requester ID.
pub type IommuFactory = Box<dyn Fn(u16) -> Arc<dyn IommuOps> + Send + Sync>;

/// Tells whether the function with a requester ID goes through the IOMMU now.
pub type UseIommu = Box<dyn Fn(u16) -> bool + Send + Sync>;

/// One function's address space, with the two regions [`PciDma::switch`] picks from.
struct Slot {
    space: Arc<AddressSpace>,
    /// The alias of system memory and the IOMMU region.
    bypass: Option<(RegionId, RegionId)>,
}

/// The vIOMMU of the root bus and the address spaces it handed out.
pub struct PciDma {
    mem: Weak<MemorySystem>,
    memory_as: Weak<AddressSpace>,
    /// The region name stem and the callbacks.
    iommu: OnceLock<(&'static str, IommuFactory)>,
    /// The name of the system memory alias and whether a function uses the IOMMU.
    bypass: OnceLock<(&'static str, UseIommu)>,
    spaces: Mutex<BTreeMap<u16, Slot>>,
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
            bypass: OnceLock::new(),
            spaces: Mutex::new(BTreeMap::new()),
            ready: AtomicBool::new(false),
        })
    }

    /// `pci_setup_iommu()`: the functions' DMA goes through regions made by `factory`, named
    /// `<stem>-<bus>-<devfn>`. Returns false if the bus already has a vIOMMU.
    pub fn set_iommu(&self, stem: &'static str, factory: IommuFactory) -> bool {
        self.iommu.set((stem, factory)).is_ok()
    }

    /// Lets the functions skip the IOMMU region while `use_iommu` says so, with an alias of
    /// system memory named `alias`. Call it before the first address space is handed out.
    pub fn set_iommu_bypass(&self, alias: &'static str, use_iommu: UseIommu) -> bool {
        self.bypass.set((alias, use_iommu)).is_ok()
    }

    /// `vtd_switch_address_space()`: shows the IOMMU region of `bdf` or the alias of system
    /// memory. Like QEMU it turns one off before it turns the other on, so the listeners see
    /// the old mappings go before the new ones come.
    pub fn switch(&self, bdf: u16, use_iommu: bool) {
        let ids = self
            .spaces
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&bdf)
            .and_then(|s| s.bypass);
        let (Some((nodmar, iommu)), Some(mem)) = (ids, self.mem.upgrade()) else {
            return;
        };
        let (off, on) = if use_iommu { (nodmar, iommu) } else { (iommu, nodmar) };
        let _ = mem.set_enabled(off, false);
        let _ = mem.set_enabled(on, true);
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
            return Some(Arc::clone(&s.space));
        }
        let mem = self.mem.upgrade()?;
        let name = format!("{stem}-{}-{}", bdf >> 8, bdf & 0xff);
        let r = mem.new_iommu(&name, 1 << 64, factory(bdf)).ok()?;
        let Some((alias, use_iommu)) = self.bypass.get() else {
            let space = mem.address_space_init(r, &name).ok()?;
            spaces.insert(bdf, Slot { space: Arc::clone(&space), bypass: None });
            return Some(space);
        };
        let system = self.memory_as.upgrade()?.root();
        let root = mem.new_container(&format!("{name}-root"), 1 << 64).ok()?;
        let nodmar = mem.new_alias(alias, system, 0, 1 << 64).ok()?;
        let on = use_iommu(bdf);
        mem.set_enabled(r, on).ok()?;
        mem.set_enabled(nodmar, !on).ok()?;
        mem.add_subregion_overlap(root, 0, nodmar, 0).ok()?;
        mem.add_subregion_overlap(root, 0, r, 1).ok()?;
        let space = mem.address_space_init(root, &name).ok()?;
        spaces.insert(bdf, Slot { space: Arc::clone(&space), bypass: Some((nodmar, r)) });
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
