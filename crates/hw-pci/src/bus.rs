// SPDX-License-Identifier: GPL-2.0-or-later

//! A PCI bus: 256 device/function slots, bus numbering and INTx routing, from hw/pci/pci.c.
//!
//! A root bus is created by the host bridge with the containers its BARs map into. A secondary
//! bus hangs off a [`crate::PciBridge`] and gets its own containers, which the bridge windows
//! alias into the parent bus.
//!
//! INTx works as in QEMU. A function raising pin `n` asks its bus to map `(devfn, n)` to an IRQ
//! number. If that bus has no `set_irq` handler the result is treated as the pin of the bridge
//! above it and mapped again one level up, until a bus that has a handler is reached. That bus
//! keeps a count of asserted sources per IRQ and calls the handler with `count != 0`, so shared
//! lines behave as a wired OR.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, RwLock, Weak};

use ruvm_base::Error;
use ruvm_mem::{MemorySystem, RegionId};

use crate::device::{MsiTrigger, NewDevice, PciDevice, PciDeviceInfo};
use crate::regs::*;

/// Drives interrupt `irq` of the controller behind a bus to `level` (0 or 1), the
/// `pci_set_irq_fn` given to `pci_bus_irqs()`.
pub type PciSetIrqFn = Arc<dyn Fn(i32, i32) + Send + Sync>;

/// Maps pin `pin` (0 is INTA) of the function at `devfn` to an IRQ number, `pci_map_irq_fn`.
pub type PciMapIrqFn = Arc<dyn Fn(u8, i32) -> i32 + Send + Sync>;

/// `pci_swizzle_map_irq_fn()`: the standard swizzle used behind PCI-to-PCI bridges.
pub fn pci_swizzle_map_irq_fn(devfn: u8, pin: i32) -> i32 {
    pci_swizzle(i32::from(pci_slot(devfn)), pin)
}

struct BusInner {
    devices: Vec<Option<Arc<PciDevice>>>,
    slot_reserved_mask: u32,
    children: Vec<Weak<PciBus>>,
}

#[derive(Default)]
struct BusIrq {
    set_irq: Option<PciSetIrqFn>,
    map_irq: Option<PciMapIrqFn>,
    irq_count: Vec<i32>,
}

/// `vmstate_pcibus` (version 1): the assertion count of each interrupt line of a root bus.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PciBusVmState {
    /// `VMSTATE_INT32_EQUAL`.
    pub nirq: i32,
    /// `nirq` entries.
    pub irq_count: Vec<i32>,
}

/// A PCI bus, `PCIBus`.
pub struct PciBus {
    name: String,
    memory: Arc<MemorySystem>,
    mem_space: RegionId,
    io_space: RegionId,
    devfn_min: u8,
    parent_dev: Option<Weak<PciDevice>>,
    parent_bus: Option<Weak<PciBus>>,
    this: Weak<PciBus>,
    inner: Mutex<BusInner>,
    irq: Mutex<BusIrq>,
    msi_handler: RwLock<Option<MsiTrigger>>,
    msi_nonbroken: AtomicBool,
    allow_0_address: AtomicBool,
    extended_config: AtomicBool,
    root_path: RwLock<String>,
}

impl fmt::Debug for PciBus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PciBus")
            .field("name", &self.name)
            .field("is_root", &self.is_root())
            .field("devfn_min", &self.devfn_min)
            .finish_non_exhaustive()
    }
}

impl PciBus {
    /// `pci_root_bus_new()`: a root bus whose memory BARs map into `mem_space` and whose I/O
    /// BARs map into `io_space`. Automatic devfn allocation starts at `devfn_min`.
    pub fn new_root(
        name: &str,
        memory: Arc<MemorySystem>,
        mem_space: RegionId,
        io_space: RegionId,
        devfn_min: u8,
    ) -> Arc<PciBus> {
        Self::build(name, memory, mem_space, io_space, devfn_min, None, None)
    }

    /// The secondary bus of a bridge, `pci_bridge_initfn()`.
    pub(crate) fn new_secondary(
        name: &str,
        memory: Arc<MemorySystem>,
        mem_space: RegionId,
        io_space: RegionId,
        parent: &Arc<PciDevice>,
    ) -> Arc<PciBus> {
        let parent_bus = parent.bus();
        let bus = Self::build(
            name,
            memory,
            mem_space,
            io_space,
            0,
            Some(Arc::downgrade(parent)),
            parent_bus.as_ref().map(Arc::downgrade),
        );
        if let Some(pb) = parent_bus {
            pb.lock().children.push(Arc::downgrade(&bus));
        }
        bus
    }

    fn build(
        name: &str,
        memory: Arc<MemorySystem>,
        mem_space: RegionId,
        io_space: RegionId,
        devfn_min: u8,
        parent_dev: Option<Weak<PciDevice>>,
        parent_bus: Option<Weak<PciBus>>,
    ) -> Arc<PciBus> {
        Arc::new_cyclic(|this| PciBus {
            name: name.to_string(),
            memory,
            mem_space,
            io_space,
            devfn_min,
            parent_dev,
            parent_bus,
            this: this.clone(),
            inner: Mutex::new(BusInner {
                devices: vec![None; PCI_DEVFN_MAX],
                slot_reserved_mask: 0,
                children: Vec::new(),
            }),
            irq: Mutex::new(BusIrq::default()),
            msi_handler: RwLock::new(None),
            msi_nonbroken: AtomicBool::new(false),
            allow_0_address: AtomicBool::new(false),
            extended_config: AtomicBool::new(false),
            root_path: RwLock::new("0000:00".to_string()),
        })
    }

    fn lock(&self) -> MutexGuard<'_, BusInner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn irq_lock(&self) -> MutexGuard<'_, BusIrq> {
        self.irq.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// The container memory BARs on this bus map into.
    pub fn mem_space(&self) -> RegionId {
        self.mem_space
    }

    /// The container I/O BARs on this bus map into.
    pub fn io_space(&self) -> RegionId {
        self.io_space
    }

    pub fn memory(&self) -> &Arc<MemorySystem> {
        &self.memory
    }

    pub fn devfn_min(&self) -> u8 {
        self.devfn_min
    }

    /// `pci_bus_is_root()`.
    pub fn is_root(&self) -> bool {
        self.parent_dev.is_none()
    }

    /// The bridge this bus sits behind, `PCIBus::parent_dev`.
    pub fn parent_dev(&self) -> Option<Arc<PciDevice>> {
        self.parent_dev.as_ref().and_then(Weak::upgrade)
    }

    /// The bus the parent bridge sits on.
    pub fn parent_bus(&self) -> Option<Arc<PciBus>> {
        self.parent_bus.as_ref().and_then(Weak::upgrade)
    }

    /// The root bus above this one.
    pub fn root(&self) -> Arc<PciBus> {
        let mut bus = self.this.upgrade().expect("bus is alive while borrowed");
        while let Some(p) = bus.parent_bus() {
            bus = p;
        }
        bus
    }

    /// `pci_bus_num()`: 0 for a root bus, else the secondary bus number of the bridge above.
    pub fn bus_num(&self) -> u8 {
        match self.parent_dev() {
            Some(p) => p.lock().byte(PCI_SECONDARY_BUS),
            None => 0,
        }
    }

    /// `pci_root_bus_path()`: the domain and bus used in error messages, "0000:00" unless the
    /// host bridge says otherwise.
    pub fn root_bus_path(&self) -> String {
        self.root().root_path.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Sets what [`PciBus::root_bus_path`] returns, the host bridge's `root_bus_path` hook.
    pub fn set_root_bus_path(&self, path: &str) {
        *self.root().root_path.write().unwrap_or_else(|p| p.into_inner()) = path.to_string();
    }

    /// `pci_bus_irqs()`: routes this bus's interrupts to `set_irq`, with `nirq` counters.
    pub fn set_irqs(&self, set_irq: PciSetIrqFn, nirq: usize) {
        let mut g = self.irq_lock();
        g.set_irq = Some(set_irq);
        g.irq_count = vec![0; nirq];
    }

    /// `pci_bus_map_irqs()`. Without one pins pass through unchanged.
    pub fn set_map_irq(&self, map_irq: PciMapIrqFn) {
        self.irq_lock().map_irq = Some(map_irq);
    }

    pub(crate) fn has_set_irq(&self) -> bool {
        self.irq_lock().set_irq.is_some()
    }

    pub(crate) fn map_irq(&self, devfn: u8, pin: i32) -> i32 {
        let map = self.irq_lock().map_irq.clone();
        match map {
            Some(m) => m(devfn, pin),
            None => pin,
        }
    }

    /// `pci_bus_change_irq_level()`.
    pub(crate) fn change_irq_level(&self, irq_num: i32, change: i32) {
        let (set_irq, level) = {
            let mut g = self.irq_lock();
            let n = usize::try_from(irq_num).expect("negative PCI IRQ number");
            assert!(n < g.irq_count.len(), "PCI IRQ {irq_num} out of range");
            g.irq_count[n] += change;
            assert!(g.irq_count[n] >= 0, "PCI IRQ {irq_num} count went negative");
            (g.set_irq.clone(), i32::from(g.irq_count[n] != 0))
        };
        if let Some(f) = set_irq {
            f(irq_num, level);
        }
    }

    /// `pci_bus_get_irq_level()`.
    pub fn irq_level(&self, irq_num: usize) -> bool {
        self.irq_count(irq_num) != 0
    }

    /// The number of asserted sources on `irq_num`.
    pub fn irq_count(&self, irq_num: usize) -> i32 {
        self.irq_lock().irq_count.get(irq_num).copied().unwrap_or(0)
    }

    /// The `PCIBUS` section's `irq_count` array, one entry per `nirq`.
    pub fn irq_counts(&self) -> Vec<i32> {
        self.irq_lock().irq_count.clone()
    }

    /// Loads the `PCIBUS` section: overwrites the counts without driving the lines, as QEMU
    /// does. `nirq` is an `INT32_EQUAL` field, so the length must match.
    pub fn set_irq_counts(&self, counts: &[i32]) -> Result<(), String> {
        let mut g = self.irq_lock();
        if counts.len() != g.irq_count.len() {
            return Err(format!(
                "{}: nirq {} does not match {}",
                self.name,
                counts.len(),
                g.irq_count.len()
            ));
        }
        g.irq_count.copy_from_slice(counts);
        Ok(())
    }

    /// The `PCIBUS` section.
    pub fn vmstate_save(&self) -> PciBusVmState {
        let irq_count = self.irq_counts();
        PciBusVmState { nirq: irq_count.len() as i32, irq_count }
    }

    /// Loads the `PCIBUS` section, see [`Self::set_irq_counts`].
    pub fn vmstate_load(&self, v: &PciBusVmState) -> Result<(), String> {
        if v.nirq as usize != v.irq_count.len() {
            return Err(format!(
                "{}: nirq {} with {} counts",
                self.name,
                v.nirq,
                v.irq_count.len()
            ));
        }
        self.set_irq_counts(&v.irq_count)
    }

    /// Sets where MSI messages from functions below this root bus go, and marks MSI as working
    /// (`msi_nonbroken`). Call it on the root bus.
    pub fn set_msi_handler(&self, handler: Option<MsiTrigger>) {
        let root = self.root();
        root.msi_nonbroken.store(handler.is_some(), Ordering::Relaxed);
        *root.msi_handler.write().unwrap_or_else(|p| p.into_inner()) = handler;
    }

    pub(crate) fn msi_handler(&self) -> Option<MsiTrigger> {
        self.root().msi_handler.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// `msi_nonbroken`: whether the interrupt controller takes MSI. QEMU keeps one global flag
    /// set by the interrupt controllers; here it belongs to the root bus.
    pub fn set_msi_nonbroken(&self, ok: bool) {
        self.root().msi_nonbroken.store(ok, Ordering::Relaxed);
    }

    pub fn msi_nonbroken(&self) -> bool {
        self.root().msi_nonbroken.load(Ordering::Relaxed)
    }

    /// `MachineClass::pci_allow_0_address`: lets BARs be mapped at address 0. Affects devices
    /// registered afterwards.
    pub fn set_allow_0_address(&self, allow: bool) {
        self.root().allow_0_address.store(allow, Ordering::Relaxed);
    }

    /// Marks the root bus as PCI Express, so the 4 KiB extended config space is reachable
    /// through [`crate::pci_host_config_read_common`]. `PCI_BUS_EXTENDED_CONFIG_SPACE`.
    pub fn set_extended_config_space(&self, on: bool) {
        self.root().extended_config.store(on, Ordering::Relaxed);
    }

    /// `pci_bus_allows_extended_config_space()`.
    pub fn allows_extended_config_space(&self) -> bool {
        self.root().extended_config.load(Ordering::Relaxed)
    }

    /// `PCIBus::slot_reserved_mask`: slots automatic and explicit placement must avoid.
    pub fn slot_reserved_mask(&self) -> u32 {
        self.lock().slot_reserved_mask
    }

    /// `pci_bus_set_slot_reserved_mask()`.
    pub fn set_slot_reserved_mask(&self, mask: u32) {
        self.lock().slot_reserved_mask |= mask;
    }

    /// `pci_bus_clear_slot_reserved_mask()`.
    pub fn clear_slot_reserved_mask(&self, mask: u32) {
        self.lock().slot_reserved_mask &= !mask;
    }

    /// The device at `devfn`, `bus->devices[devfn]`.
    pub fn device(&self, devfn: u8) -> Option<Arc<PciDevice>> {
        self.lock().devices[usize::from(devfn)].clone()
    }

    /// All devices on the bus in devfn order.
    pub fn devices(&self) -> Vec<Arc<PciDevice>> {
        self.lock().devices.iter().flatten().cloned().collect()
    }

    /// The secondary buses of bridges on this bus.
    pub fn children(&self) -> Vec<Arc<PciBus>> {
        self.lock().children.iter().filter_map(Weak::upgrade).collect()
    }

    /// `do_pci_register_device()` plus `pci_init_multifunction()`: places a new function at
    /// `devfn`, or at the first free slot from `devfn_min` if `None`.
    pub fn register_device(
        &self,
        info: &PciDeviceInfo,
        devfn: Option<u8>,
    ) -> Result<Arc<PciDevice>, Error> {
        let name = &info.name;
        let mut g = self.lock();
        let reserved = |g: &BusInner, devfn: u8| g.slot_reserved_mask & (1 << pci_slot(devfn)) != 0;
        let devfn = match devfn {
            None => (usize::from(self.devfn_min)..PCI_DEVFN_MAX)
                .step_by(usize::from(PCI_FUNC_MAX))
                .map(|d| d as u8)
                .find(|&d| g.devices[usize::from(d)].is_none() && !reserved(&g, d))
                .ok_or_else(|| {
                    Error::generic(format!(
                        "PCI: no slot/function available for {name}, all in use or reserved"
                    ))
                })?,
            Some(d) if reserved(&g, d) => {
                return Err(Error::generic(format!(
                    "PCI: slot {} function {} not available for {}, reserved",
                    pci_slot(d),
                    pci_func(d),
                    name
                )));
            }
            Some(d) => {
                if let Some(other) = &g.devices[usize::from(d)] {
                    return Err(Error::generic(format!(
                        "PCI: slot {} function {} not available for {}, in use by {},id={}",
                        pci_slot(d),
                        pci_func(d),
                        name,
                        other.name(),
                        other.id().unwrap_or("(null)")
                    )));
                }
                d
            }
        };

        // pci_init_multifunction().
        let slot = pci_slot(devfn);
        if pci_func(devfn) != 0 {
            if let Some(f0) = &g.devices[usize::from(pci_devfn(slot, 0))] {
                if !f0.is_multifunction() {
                    return Err(Error::generic(format!(
                        "PCI: single function device can't be populated in function {:x}.{:x}",
                        slot,
                        pci_func(devfn)
                    )));
                }
            }
        } else if !info.multifunction {
            for func in 1..PCI_FUNC_MAX {
                if g.devices[usize::from(pci_devfn(slot, func))].is_some() {
                    return Err(Error::generic(format!(
                        "PCI: {slot:x}.0 indicates single function, but {slot:x}.{func:x} is \
                         already populated."
                    )));
                }
            }
        }

        let dev = PciDevice::new(NewDevice {
            info,
            devfn,
            bus: self.this.clone(),
            memory: Arc::clone(&self.memory),
            io_space: self.io_space,
            mem_space: self.mem_space,
            allow_0_address: self.root_allow_0(),
        });
        g.devices[usize::from(devfn)] = Some(Arc::clone(&dev));
        Ok(dev)
    }

    fn root_allow_0(&self) -> bool {
        match self.parent_bus() {
            Some(p) => p.root_allow_0(),
            None => self.allow_0_address.load(Ordering::Relaxed),
        }
    }

    /// `do_pci_unregister_device()`: unmaps the BARs, drops the INTx pins and frees the slot.
    pub fn unregister_device(&self, dev: &PciDevice) {
        dev.deassert_intx();
        dev.unregister_bars();
        let mut g = self.lock();
        let slot = &mut g.devices[usize::from(dev.devfn())];
        if slot.as_ref().is_some_and(|d| std::ptr::eq(Arc::as_ptr(d), dev)) {
            *slot = None;
        }
    }

    /// `pcibus_reset_hold()` plus the resets of the devices on the bus: every function gets its
    /// model reset and the generic PCI reset. Bridges reset their own secondary buses.
    pub fn reset(&self) {
        for dev in self.devices() {
            dev.reset();
        }
    }

    /// `pci_find_bus_nr()`: the bus numbered `bus_num` at or below this one.
    pub fn find_bus_nr(&self, bus_num: u8) -> Option<Arc<PciBus>> {
        let this = self.this.upgrade()?;
        if self.bus_num() == bus_num {
            return Some(this);
        }
        // Consider all bus numbers in use.
        if let Some(p) = self.parent_dev() {
            if !secondary_bus_in_range(&p, bus_num) {
                return None;
            }
        }
        for sec in self.children() {
            if sec.bus_num() == bus_num {
                return Some(sec);
            }
            if let Some(p) = sec.parent_dev() {
                if secondary_bus_in_range(&p, bus_num) {
                    return sec.find_bus_nr(bus_num);
                }
            }
        }
        None
    }

    /// `pci_find_device()`.
    pub fn find_device(&self, bus_num: u8, devfn: u8) -> Option<Arc<PciDevice>> {
        self.find_bus_nr(bus_num)?.device(devfn)
    }
}

/// `pci_secondary_bus_in_range()`.
fn secondary_bus_in_range(dev: &PciDevice, bus_num: u8) -> bool {
    let s = dev.lock();
    s.word(PCI_BRIDGE_CONTROL) & PCI_BRIDGE_CTL_BUS_RESET == 0
        && s.byte(PCI_SECONDARY_BUS) <= bus_num
        && bus_num <= s.byte(PCI_SUBORDINATE_BUS)
}
