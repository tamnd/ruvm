// SPDX-License-Identifier: GPL-2.0-or-later

//! One PCI function: its configuration space, BARs, capability list and INTx pin, from
//! hw/pci/pci.c.
//!
//! Each function owns a config array plus the `wmask`, `w1cmask`, `cmask` and `used` arrays of
//! the same size. `wmask` says which bits a guest write may change, `w1cmask` which bits a guest
//! clears by writing a one, `cmask` which bits the migration check compares, and `used` which
//! bytes belong to a capability. [`PciDevice::default_write_config`] applies the masks and then
//! reacts to what changed exactly like `pci_default_write_config()`: BAR and command register
//! writes remap the BARs, the INTx disable bit re-evaluates the pins, and the MSI and MSI-X hooks
//! run last.
//!
//! State lives behind one mutex per function. Interrupts (INTx level changes and MSI messages)
//! are collected while the lock is held and delivered after it is dropped, so the handlers they
//! reach may call back into the device.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, RwLock, Weak};

use ruvm_base::Error;
use ruvm_hw_core::IrqLine;
use ruvm_mem::{MemorySystem, RegionId};

use crate::bus::PciBus;
use crate::msix::Msix;
use crate::regs::*;

/// Delivers one MSI or MSI-X message: a 32 bit little endian store of `data` at `addr`, what
/// `address_space_stl_le()` on the device's bus master address space does in QEMU.
pub type MsiTrigger = Arc<dyn Fn(u64, u32) + Send + Sync>;

/// An MSI or MSI-X message, `MSIMessage`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct MsiMessage {
    pub address: u64,
    pub data: u32,
}

/// Per model hooks, the `config_read`, `config_write` and reset callbacks of `PCIDeviceClass`.
///
/// The defaults are the generic behaviour. A model that overrides a config callback normally
/// calls [`PciDevice::default_write_config`] (or the read side) and then looks at what changed.
pub trait PciDeviceOps: Send + Sync {
    /// `PCIDeviceClass::config_read`.
    fn config_read(&self, dev: &PciDevice, addr: u32, len: u32) -> u32 {
        dev.default_read_config(addr, len)
    }

    /// `PCIDeviceClass::config_write`.
    fn config_write(&self, dev: &PciDevice, addr: u32, val: u32, len: u32) {
        dev.default_write_config(addr, val, len);
    }

    /// The model's own reset, run by [`PciDevice::reset`] before the generic PCI reset.
    fn reset(&self, dev: &PciDevice) {
        let _ = dev;
    }
}

/// What a model tells the core about itself when it registers, the ID fields of
/// `PCIDeviceClass` plus the few flags that change how config space is laid out.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PciDeviceInfo {
    /// The device name used in error messages, `PCIDevice::name`.
    pub name: String,
    /// The qdev id, only used in error messages.
    pub id: Option<String>,
    pub vendor_id: u16,
    pub device_id: u16,
    pub revision: u8,
    pub class_id: u16,
    /// Header type 0 only. When both are zero QEMU's default subsystem IDs are used.
    pub subsystem_vendor_id: u16,
    pub subsystem_id: u16,
    /// A PCI-to-PCI bridge with a type 1 header. Use [`crate::PciBridge`] to create one.
    pub is_bridge: bool,
    /// `QEMU_PCI_CAP_MULTIFUNCTION`, the `multifunction` property.
    pub multifunction: bool,
    /// `QEMU_PCI_CAP_EXPRESS`: 4 KiB of config space instead of 256 bytes.
    pub express: bool,
}

/// A snapshot of one BAR, `PCIIORegion`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PciBarInfo {
    /// Where the BAR is mapped, or [`PCI_BAR_UNMAPPED`].
    pub addr: u64,
    pub size: u64,
    /// The low bits of the BAR: I/O or memory, 64 bit, prefetchable.
    pub type_: u8,
    /// The region mapped when the BAR is enabled.
    pub memory: RegionId,
}

/// Mutable access to the config arrays while a model sets itself up, for the work QEMU models do
/// by poking `dev->config`, `dev->wmask` and friends directly in their realize functions.
#[derive(Debug)]
pub struct PciConfigMut<'a> {
    pub config: &'a mut [u8],
    pub wmask: &'a mut [u8],
    pub w1cmask: &'a mut [u8],
    pub cmask: &'a mut [u8],
}

#[derive(Clone, Default)]
pub(crate) struct IoRegion {
    pub(crate) addr: u64,
    pub(crate) size: u64,
    pub(crate) type_: u8,
    pub(crate) memory: Option<RegionId>,
    pub(crate) container: Option<RegionId>,
}

/// Something to do once the device lock is dropped.
pub(crate) enum Effect {
    /// `pci_change_irq_level()` for a pin.
    Intx { pin: usize, change: i32 },
    /// `pci_msi_trigger()`.
    Msi { address: u64, data: u32 },
}

pub(crate) type Effects = Vec<Effect>;

pub(crate) struct DevState {
    pub(crate) config: Vec<u8>,
    pub(crate) wmask: Vec<u8>,
    pub(crate) w1cmask: Vec<u8>,
    pub(crate) cmask: Vec<u8>,
    pub(crate) used: Vec<u8>,
    pub(crate) io_regions: [IoRegion; PCI_NUM_REGIONS],
    pub(crate) irq_state: u8,
    pub(crate) enabled: bool,
    pub(crate) bus_master: bool,
    /// Offset of the power management capability, 0 if there is none.
    pub(crate) pm_cap: u8,
    /// Offset of the MSI capability, 0 if there is none.
    pub(crate) msi_cap: u8,
    pub(crate) msix: Option<Msix>,
}

impl DevState {
    pub(crate) fn byte(&self, off: usize) -> u8 {
        self.config[off]
    }

    pub(crate) fn word(&self, off: usize) -> u16 {
        pci_get_word(&self.config, off)
    }

    pub(crate) fn set_word(&mut self, off: usize, v: u16) {
        pci_set_word(&mut self.config, off, v);
    }

    pub(crate) fn long(&self, off: usize) -> u32 {
        pci_get_long(&self.config, off)
    }

    pub(crate) fn set_long(&mut self, off: usize, v: u32) {
        pci_set_long(&mut self.config, off, v);
    }

    fn header_type(&self) -> u8 {
        self.config[PCI_HEADER_TYPE] & !PCI_HEADER_TYPE_MULTI_FUNCTION
    }

    /// `pci_bar()`: the config offset of region `reg`.
    fn bar_offset(&self, reg: usize) -> usize {
        if reg != PCI_ROM_SLOT {
            return PCI_BASE_ADDRESS_0 + reg * 4;
        }
        if self.header_type() == PCI_HEADER_TYPE_BRIDGE {
            PCI_ROM_ADDRESS1
        } else {
            PCI_ROM_ADDRESS
        }
    }

    pub(crate) fn irq_disabled(&self) -> bool {
        self.word(PCI_COMMAND) & PCI_COMMAND_INTX_DISABLE != 0
    }

    fn irq_pin_state(&self, pin: usize) -> i32 {
        i32::from((self.irq_state >> pin) & 1)
    }

    fn update_irq_status(&mut self) {
        let status = self.word(PCI_STATUS);
        if self.irq_state != 0 {
            self.set_word(PCI_STATUS, status | PCI_STATUS_INTERRUPT);
        } else {
            self.set_word(PCI_STATUS, status & !PCI_STATUS_INTERRUPT);
        }
    }

    /// `pci_irq_handler()` minus the routing, which the caller does after unlocking.
    pub(crate) fn irq_handler(&mut self, fx: &mut Effects, pin: usize, level: i32) {
        assert!(pin < PCI_NUM_PINS, "INTx pin {pin} out of range");
        assert!(level == 0 || level == 1, "INTx level must be 0 or 1");
        let change = level - self.irq_pin_state(pin);
        if change == 0 {
            return;
        }
        self.irq_state &= !(1 << pin);
        self.irq_state |= (level as u8) << pin;
        self.update_irq_status();
        if self.irq_disabled() {
            return;
        }
        fx.push(Effect::Intx { pin, change });
    }

    /// `pci_device_deassert_intx()`.
    pub(crate) fn deassert_intx(&mut self, fx: &mut Effects) {
        for pin in 0..PCI_NUM_PINS {
            self.irq_handler(fx, pin, 0);
        }
    }

    /// `pci_update_irq_disabled()`.
    fn update_irq_disabled(&mut self, fx: &mut Effects, was_irq_disabled: bool) {
        let disabled = self.irq_disabled();
        if disabled == was_irq_disabled {
            return;
        }
        for pin in 0..PCI_NUM_PINS {
            let state = self.irq_pin_state(pin);
            fx.push(Effect::Intx { pin, change: if disabled { -state } else { state } });
        }
    }

    /// `pci_msi_trigger()`, dropped when bus mastering is off as the disabled bus master
    /// region would drop it in QEMU.
    pub(crate) fn send_message(&self, fx: &mut Effects, msg: MsiMessage) {
        if self.bus_master {
            fx.push(Effect::Msi { address: msg.address, data: msg.data });
        }
    }

    /// `pci_pm_state()`.
    fn pm_state(&self) -> u8 {
        if self.pm_cap == 0 {
            return 0;
        }
        (self.word(usize::from(self.pm_cap) + PCI_PM_CTRL) & PCI_PM_CTRL_STATE_MASK) as u8
    }

    /// `pci_pm_update()`.
    fn pm_update(&mut self, addr: u32, len: u32, old: u8) -> u8 {
        let ctrl = usize::from(self.pm_cap) + PCI_PM_CTRL;
        if self.pm_cap == 0 || !range_covers_byte(addr.into(), len.into(), ctrl as u64) {
            return old;
        }
        let new = self.pm_state();
        if new == old {
            return old;
        }
        let pmc = self.word(usize::from(self.pm_cap) + PCI_PM_PMC);
        // Transitions to D1 and D2 are only allowed if supported. Devices may only go to a
        // higher D state or back to D0.
        if (pmc & PCI_PM_CAP_D1 == 0 && new == 1)
            || (pmc & PCI_PM_CAP_D2 == 0 && new == 2)
            || (old != 0 && new != 0 && new < old)
        {
            let v = self.word(ctrl) & !PCI_PM_CTRL_STATE_MASK;
            self.set_word(ctrl, v | u16::from(old));
            return old;
        }
        new
    }

    /// `pci_config_get_bar_addr()` for a function that is not a VF.
    fn config_bar_addr(&self, reg: usize, type_: u8, size: u64) -> u64 {
        let bar = self.bar_offset(reg);
        let mut new_addr = if type_ & PCI_BASE_ADDRESS_MEM_TYPE_64 != 0 {
            pci_get_quad(&self.config, bar)
        } else {
            u64::from(self.long(bar))
        };
        // The ROM slot has a specific enable bit, keep it intact.
        if reg != PCI_ROM_SLOT {
            new_addr &= !(size - 1);
        }
        new_addr
    }

    /// `pci_bar_address()`.
    pub(crate) fn bar_address(&self, reg: usize, type_: u8, size: u64, allow_0: bool) -> u64 {
        let cmd = self.word(PCI_COMMAND);
        if type_ & PCI_BASE_ADDRESS_SPACE_IO != 0 {
            if cmd & PCI_COMMAND_IO == 0 {
                return PCI_BAR_UNMAPPED;
            }
            let new_addr = self.config_bar_addr(reg, type_, size);
            let last_addr = new_addr.wrapping_add(size).wrapping_sub(1);
            // Check if a 32 bit BAR wraps around explicitly.
            if last_addr <= new_addr
                || last_addr >= u64::from(u32::MAX)
                || (!allow_0 && new_addr == 0)
            {
                return PCI_BAR_UNMAPPED;
            }
            return new_addr;
        }

        if cmd & PCI_COMMAND_MEMORY == 0 {
            return PCI_BAR_UNMAPPED;
        }
        let mut new_addr = self.config_bar_addr(reg, type_, size);
        // The ROM slot has a specific enable bit.
        if reg == PCI_ROM_SLOT && new_addr & PCI_ROM_ADDRESS_ENABLE == 0 {
            return PCI_BAR_UNMAPPED;
        }
        new_addr &= !(size - 1);
        let last_addr = new_addr.wrapping_add(size).wrapping_sub(1);
        // Wrapping is not supported, and some values are treated as invalid mappings.
        if last_addr <= new_addr || last_addr == PCI_BAR_UNMAPPED || (!allow_0 && new_addr == 0) {
            return PCI_BAR_UNMAPPED;
        }
        // A 32 bit BAR must not wrap around 4 GiB. PC IDE needs this.
        if type_ & PCI_BASE_ADDRESS_MEM_TYPE_64 == 0 && last_addr >= u64::from(u32::MAX) {
            return PCI_BAR_UNMAPPED;
        }
        // The OS may set a BAR beyond what the machine can address, for example a 32 bit OS
        // putting a 64 bit BAR above 4 GiB. `HWADDR_MAX` is all ones for us.
        if last_addr == u64::MAX {
            return PCI_BAR_UNMAPPED;
        }
        new_addr
    }

    /// `pci_reset_regions()`.
    fn reset_regions(&mut self) {
        for r in 0..PCI_NUM_REGIONS {
            let region = &self.io_regions[r];
            if region.size == 0 {
                continue;
            }
            let type_ = region.type_;
            let bar = self.bar_offset(r);
            if type_ & PCI_BASE_ADDRESS_SPACE_IO == 0 && type_ & PCI_BASE_ADDRESS_MEM_TYPE_64 != 0 {
                pci_set_quad(&mut self.config, bar, u64::from(type_));
            } else {
                self.set_long(bar, u32::from(type_));
            }
        }
    }
}

/// A PCI function on a bus, `PCIDevice`.
pub struct PciDevice {
    name: String,
    id: Option<String>,
    devfn: u8,
    is_bridge: bool,
    multifunction: bool,
    express: bool,
    allow_0_address: bool,
    bus: Weak<PciBus>,
    memory: Arc<MemorySystem>,
    io_space: RegionId,
    mem_space: RegionId,
    this: Weak<PciDevice>,
    state: Mutex<DevState>,
    ops: RwLock<Option<Arc<dyn PciDeviceOps>>>,
    msi_trigger: RwLock<Option<MsiTrigger>>,
}

impl fmt::Debug for PciDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PciDevice")
            .field("name", &self.name)
            .field(
                "devfn",
                &format_args!("{:02x}.{:x}", pci_slot(self.devfn), pci_func(self.devfn)),
            )
            .field("is_bridge", &self.is_bridge)
            .finish_non_exhaustive()
    }
}

/// What the bus hands a new device, everything `do_pci_register_device()` takes from the bus.
pub(crate) struct NewDevice<'a> {
    pub(crate) info: &'a PciDeviceInfo,
    pub(crate) devfn: u8,
    pub(crate) bus: Weak<PciBus>,
    pub(crate) memory: Arc<MemorySystem>,
    pub(crate) io_space: RegionId,
    pub(crate) mem_space: RegionId,
    pub(crate) allow_0_address: bool,
}

impl PciDevice {
    /// The config space part of `do_pci_register_device()`.
    pub(crate) fn new(n: NewDevice<'_>) -> Arc<PciDevice> {
        let info = n.info;
        let size = pci_config_size(info.express);
        let mut s = DevState {
            config: vec![0; size],
            wmask: vec![0; size],
            w1cmask: vec![0; size],
            cmask: vec![0; size],
            used: vec![0; size],
            io_regions: Default::default(),
            irq_state: 0,
            enabled: true,
            bus_master: false,
            pm_cap: 0,
            msi_cap: 0,
            msix: None,
        };
        let c = &mut s.config;
        pci_set_word(c, PCI_VENDOR_ID, info.vendor_id);
        pci_set_word(c, PCI_DEVICE_ID, info.device_id);
        c[PCI_REVISION_ID] = info.revision;
        pci_set_word(c, PCI_CLASS_DEVICE, info.class_id);
        if !info.is_bridge {
            if info.subsystem_vendor_id != 0 || info.subsystem_id != 0 {
                pci_set_word(c, PCI_SUBSYSTEM_VENDOR_ID, info.subsystem_vendor_id);
                pci_set_word(c, PCI_SUBSYSTEM_ID, info.subsystem_id);
            } else {
                pci_set_word(c, PCI_SUBSYSTEM_VENDOR_ID, PCI_SUBVENDOR_ID_REDHAT_QUMRANET);
                pci_set_word(c, PCI_SUBSYSTEM_ID, PCI_SUBDEVICE_ID_QEMU);
            }
        }
        init_cmask(&mut s);
        init_wmask(&mut s);
        init_w1cmask(&mut s);
        if info.is_bridge {
            init_mask_bridge(&mut s);
        }
        if info.multifunction {
            s.config[PCI_HEADER_TYPE] |= PCI_HEADER_TYPE_MULTI_FUNCTION;
        }
        Arc::new_cyclic(|this| PciDevice {
            name: info.name.clone(),
            id: info.id.clone(),
            devfn: n.devfn,
            is_bridge: info.is_bridge,
            multifunction: info.multifunction,
            express: info.express,
            allow_0_address: n.allow_0_address,
            bus: n.bus,
            memory: n.memory,
            io_space: n.io_space,
            mem_space: n.mem_space,
            this: this.clone(),
            state: Mutex::new(s),
            ops: RwLock::new(None),
            msi_trigger: RwLock::new(None),
        })
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, DevState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub(crate) fn weak(&self) -> Weak<PciDevice> {
        self.this.clone()
    }

    pub(crate) fn memory(&self) -> &Arc<MemorySystem> {
        &self.memory
    }

    /// `PCIDevice::name`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The qdev id given at registration.
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    pub fn devfn(&self) -> u8 {
        self.devfn
    }

    pub fn slot(&self) -> u8 {
        pci_slot(self.devfn)
    }

    pub fn func(&self) -> u8 {
        pci_func(self.devfn)
    }

    pub fn is_bridge(&self) -> bool {
        self.is_bridge
    }

    pub fn is_multifunction(&self) -> bool {
        self.multifunction
    }

    pub fn is_express(&self) -> bool {
        self.express
    }

    /// `pci_config_size()`.
    pub fn config_size(&self) -> usize {
        pci_config_size(self.express)
    }

    /// The bus the function sits on, while it exists.
    pub fn bus(&self) -> Option<Arc<PciBus>> {
        self.bus.upgrade()
    }

    /// `pci_dev_bus_num()`.
    pub fn bus_num(&self) -> u8 {
        self.bus().map_or(0, |b| b.bus_num())
    }

    /// `pci_requester_id()` for a device that is not behind a PCIe-to-PCI bridge.
    pub fn requester_id(&self) -> u16 {
        (u16::from(self.bus_num()) << 8) | u16::from(self.devfn)
    }

    /// Installs the model's config and reset hooks.
    pub fn set_ops(&self, ops: Arc<dyn PciDeviceOps>) {
        *self.ops.write().unwrap_or_else(|p| p.into_inner()) = Some(ops);
    }

    fn ops(&self) -> Option<Arc<dyn PciDeviceOps>> {
        self.ops.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Sets where this function's MSI and MSI-X messages go. Without one they go to the handler
    /// of the root bus, see [`PciBus::set_msi_handler`].
    pub fn set_msi_trigger(&self, trigger: Option<MsiTrigger>) {
        *self.msi_trigger.write().unwrap_or_else(|p| p.into_inner()) = trigger;
    }

    /// Runs what was collected under the lock.
    pub(crate) fn run(&self, fx: Effects) {
        for e in fx {
            match e {
                Effect::Intx { pin, change } => self.change_irq_level(pin, change),
                Effect::Msi { address, data } => self.msi_deliver(address, data),
            }
        }
    }

    fn msi_deliver(&self, address: u64, data: u32) {
        let own = self.msi_trigger.read().unwrap_or_else(|p| p.into_inner()).clone();
        let trigger = own.or_else(|| self.bus().and_then(|b| b.msi_handler()));
        if let Some(t) = trigger {
            t(address, data);
        }
    }

    /// `pci_change_irq_level()`: maps the pin through each bus on the way up until one that has
    /// a `set_irq` handler, and changes that bus's count.
    fn change_irq_level(&self, pin: usize, change: i32) {
        let Some(mut bus) = self.bus() else { return };
        let mut devfn = self.devfn;
        let mut irq_num = pin as i32;
        loop {
            irq_num = bus.map_irq(devfn, irq_num);
            if bus.has_set_irq() {
                break;
            }
            let Some(parent) = bus.parent_dev() else { return };
            devfn = parent.devfn;
            let Some(next) = parent.bus() else { return };
            bus = next;
        }
        bus.change_irq_level(irq_num, change);
    }

    /// Calls `config_read` of the model, or the default.
    pub fn config_read(&self, addr: u32, len: u32) -> u32 {
        match self.ops() {
            Some(ops) => ops.config_read(self, addr, len),
            None => self.default_read_config(addr, len),
        }
    }

    /// Calls `config_write` of the model, or the default.
    pub fn config_write(&self, addr: u32, val: u32, len: u32) {
        match self.ops() {
            Some(ops) => ops.config_write(self, addr, val, len),
            None => self.default_write_config(addr, val, len),
        }
    }

    /// `pci_default_read_config()`.
    pub fn default_read_config(&self, addr: u32, len: u32) -> u32 {
        let s = self.lock();
        let (addr, len) = (addr as usize, len as usize);
        assert!(len <= 4 && addr + len <= s.config.len(), "config read out of range");
        let mut bytes = [0u8; 4];
        bytes[..len].copy_from_slice(&s.config[addr..addr + len]);
        u32::from_le_bytes(bytes)
    }

    /// `pci_default_write_config()`.
    pub fn default_write_config(&self, addr: u32, val: u32, len: u32) {
        let mut fx = Effects::new();
        {
            let mut s = self.lock();
            self.write_config_locked(&mut s, &mut fx, addr, val, len);
        }
        self.run(fx);
    }

    fn write_config_locked(
        &self,
        s: &mut DevState,
        fx: &mut Effects,
        addr: u32,
        val_in: u32,
        len: u32,
    ) {
        let old_pm_state = s.pm_state();
        let was_irq_disabled = s.irq_disabled();
        let a = addr as usize;
        assert!(len <= 4 && a + len as usize <= s.config.len(), "config write out of range");

        let mut val = val_in;
        for i in 0..len as usize {
            let wmask = s.wmask[a + i];
            let w1cmask = s.w1cmask[a + i];
            debug_assert!(wmask & w1cmask == 0);
            let v = val as u8;
            s.config[a + i] = (s.config[a + i] & !wmask) | (v & wmask);
            // W1C: write 1 to clear.
            s.config[a + i] &= !(v & w1cmask);
            val >>= 8;
        }

        let new_pm_state = s.pm_update(addr, len, old_pm_state);

        let (addr64, len64) = (u64::from(addr), u64::from(len));
        if ranges_overlap(addr64, len64, PCI_BASE_ADDRESS_0 as u64, 24)
            || ranges_overlap(addr64, len64, PCI_ROM_ADDRESS as u64, 4)
            || ranges_overlap(addr64, len64, PCI_ROM_ADDRESS1 as u64, 4)
            || range_covers_byte(addr64, len64, PCI_COMMAND as u64)
            || (new_pm_state != 0) != (old_pm_state != 0)
        {
            self.update_mappings_locked(s);
        }

        if ranges_overlap(addr64, len64, PCI_COMMAND as u64, 2) {
            s.update_irq_disabled(fx, was_irq_disabled);
            s.bus_master = s.word(PCI_COMMAND) & PCI_COMMAND_MASTER != 0 && s.enabled;
        }

        s.msi_write_config(fx, addr, val_in, len);
        s.msix_write_config(fx, addr, len);
    }

    /// A copy of config space.
    pub fn config_bytes(&self) -> Vec<u8> {
        self.lock().config.clone()
    }

    /// A copy of the write mask.
    pub fn wmask_bytes(&self) -> Vec<u8> {
        self.lock().wmask.clone()
    }

    /// A copy of the write-one-to-clear mask.
    pub fn w1cmask_bytes(&self) -> Vec<u8> {
        self.lock().w1cmask.clone()
    }

    /// A copy of the migration compare mask.
    pub fn cmask_bytes(&self) -> Vec<u8> {
        self.lock().cmask.clone()
    }

    /// Runs `f` on the config arrays. This is for model setup; nothing is remapped or
    /// re-evaluated afterwards.
    pub fn with_config<R>(&self, f: impl FnOnce(PciConfigMut<'_>) -> R) -> R {
        let mut s = self.lock();
        let s = &mut *s;
        f(PciConfigMut {
            config: &mut s.config,
            wmask: &mut s.wmask,
            w1cmask: &mut s.w1cmask,
            cmask: &mut s.cmask,
        })
    }

    /// `pci_register_bar()`. `memory` is mapped into the bus's I/O or memory space when the
    /// guest enables the BAR. Its size must be a power of two.
    pub fn register_bar(&self, region_num: usize, type_: u8, memory: RegionId) {
        assert!(region_num < PCI_NUM_REGIONS, "BAR {region_num} out of range");
        let size = self.memory.region(memory).map_or(0, |r| r.size);
        assert!(
            size.is_power_of_two() && size <= 1 << 63,
            "BAR size {size:#x} is not a power of 2"
        );
        let size = size as u64;
        let mut s = self.lock();
        // A bridge (type 1 header) has at most 2 BARs.
        assert!(s.header_type() != PCI_HEADER_TYPE_BRIDGE || region_num < 2 || region_num == 6);
        assert!(s.io_regions[region_num].size == 0, "BAR {region_num} registered twice");
        let container =
            if type_ & PCI_BASE_ADDRESS_SPACE_IO != 0 { self.io_space } else { self.mem_space };
        s.io_regions[region_num] = IoRegion {
            addr: PCI_BAR_UNMAPPED,
            size,
            type_,
            memory: Some(memory),
            container: Some(container),
        };

        let mut wmask = !(size - 1);
        if region_num == PCI_ROM_SLOT {
            // The ROM enable bit is writable.
            wmask |= PCI_ROM_ADDRESS_ENABLE;
        }
        let addr = s.bar_offset(region_num);
        s.set_long(addr, u32::from(type_));
        if type_ & PCI_BASE_ADDRESS_SPACE_IO == 0 && type_ & PCI_BASE_ADDRESS_MEM_TYPE_64 != 0 {
            pci_set_quad(&mut s.wmask, addr, wmask);
            pci_set_quad(&mut s.cmask, addr, !0);
        } else {
            pci_set_long(&mut s.wmask, addr, wmask as u32);
            pci_set_long(&mut s.cmask, addr, 0xffff_ffff);
        }
    }

    /// `pci_unregister_io_regions()`: unmaps and forgets every BAR.
    pub fn unregister_bars(&self) {
        let mut s = self.lock();
        let _t = self.memory.transaction();
        for r in s.io_regions.iter_mut() {
            if r.size == 0 {
                continue;
            }
            if r.addr != PCI_BAR_UNMAPPED {
                if let (Some(c), Some(m)) = (r.container, r.memory) {
                    // The region is ours and mapped, so this cannot fail.
                    let _ = self.memory.del_subregion(c, m);
                }
            }
            *r = IoRegion::default();
        }
    }

    /// `pci_get_bar_addr()`: where region `n` is mapped, or [`PCI_BAR_UNMAPPED`].
    pub fn bar_addr(&self, region_num: usize) -> u64 {
        self.lock().io_regions[region_num].addr
    }

    /// Region `n`, if it was registered.
    pub fn bar_info(&self, region_num: usize) -> Option<PciBarInfo> {
        let s = self.lock();
        let r = &s.io_regions[region_num];
        Some(PciBarInfo { addr: r.addr, size: r.size, type_: r.type_, memory: r.memory? })
    }

    /// `pci_bar_address()`: where region `n` would be mapped given config space as it is now.
    pub fn bar_address(&self, region_num: usize) -> u64 {
        let s = self.lock();
        let r = &s.io_regions[region_num];
        if r.size == 0 {
            return PCI_BAR_UNMAPPED;
        }
        s.bar_address(region_num, r.type_, r.size, self.allow_0_address)
    }

    /// `pci_update_mappings()`.
    pub fn update_mappings(&self) {
        let mut s = self.lock();
        self.update_mappings_locked(&mut s);
    }

    pub(crate) fn update_mappings_locked(&self, s: &mut DevState) {
        let _t = self.memory.transaction();
        let pm_off = s.pm_state() != 0;
        for i in 0..PCI_NUM_REGIONS {
            let r = &s.io_regions[i];
            if r.size == 0 {
                continue;
            }
            let mut new_addr = s.bar_address(i, r.type_, r.size, self.allow_0_address);
            if !s.enabled || pm_off {
                new_addr = PCI_BAR_UNMAPPED;
            }
            let r = &mut s.io_regions[i];
            if new_addr == r.addr {
                continue;
            }
            let (Some(container), Some(mem)) = (r.container, r.memory) else { continue };
            if r.addr != PCI_BAR_UNMAPPED {
                // Unmapping a region we mapped cannot fail.
                let _ = self.memory.del_subregion(container, mem);
            }
            r.addr = new_addr;
            if r.addr != PCI_BAR_UNMAPPED {
                // Nor can mapping one that is not mapped anywhere else.
                let _ = self.memory.add_subregion_overlap(container, r.addr, mem, 1);
            }
        }
    }

    /// `pci_add_capability()`. With `offset` 0 the first free space after the standard header
    /// is used. Returns the offset of the new capability.
    pub fn add_capability(&self, cap_id: u8, offset: u8, size: u8) -> Result<u8, Error> {
        let mut s = self.lock();
        self.add_capability_locked(&mut s, cap_id, offset, size)
    }

    pub(crate) fn add_capability_locked(
        &self,
        s: &mut DevState,
        cap_id: u8,
        offset: u8,
        size: u8,
    ) -> Result<u8, Error> {
        let offset = if offset == 0 {
            let found = find_space(s, size);
            if found == 0 {
                return Err(Error::generic(format!(
                    "PCI: no space for capability {cap_id:x} of size {size} in {}",
                    self.name
                )));
            }
            found
        } else {
            // Capabilities must not overlap. Device assignment relies on this check.
            for i in usize::from(offset)..usize::from(offset) + usize::from(size) {
                let overlapping_cap = find_capability_at_offset(s, i);
                if overlapping_cap != 0 {
                    return Err(Error::generic(format!(
                        "{}:{:02x}:{:02x}.{:x} Attempt to add PCI capability {:x} at offset {:x} \
                         overlaps existing capability {:x} at offset {:x}",
                        self.root_bus_path(),
                        self.bus_num(),
                        self.slot(),
                        self.func(),
                        cap_id,
                        offset,
                        overlapping_cap,
                        i
                    )));
                }
            }
            offset
        };

        let o = usize::from(offset);
        let size = usize::from(size);
        s.config[o + PCI_CAP_LIST_ID] = cap_id;
        s.config[o + PCI_CAP_LIST_NEXT] = s.config[PCI_CAPABILITY_LIST];
        s.config[PCI_CAPABILITY_LIST] = offset;
        let status = s.word(PCI_STATUS);
        s.set_word(PCI_STATUS, status | PCI_STATUS_CAP_LIST);
        let used_end = (o + size.next_multiple_of(4)).min(s.used.len());
        s.used[o..used_end].fill(0xff);
        // Capabilities are read-only by default, and checked on migration.
        s.wmask[o..o + size].fill(0);
        s.cmask[o..o + size].fill(0xff);
        Ok(offset)
    }

    /// `pci_del_capability()`: unlinks the capability and makes its bytes plain writable
    /// config space again.
    pub fn del_capability(&self, cap_id: u8, size: u8) {
        let mut s = self.lock();
        del_capability_locked(&mut s, cap_id, size);
    }

    /// `pci_find_capability()`: the offset of the first capability with `cap_id`, or 0.
    pub fn find_capability(&self, cap_id: u8) -> u8 {
        find_capability_list(&self.lock(), cap_id).0
    }

    /// `pci_pm_init()`: adds the power management capability. BARs are only mapped in D0.
    pub fn pm_init(&self, offset: u8) -> Result<u8, Error> {
        let mut s = self.lock();
        let cap = self.add_capability_locked(&mut s, PCI_CAP_ID_PM, offset, PCI_PM_SIZEOF)?;
        s.pm_cap = cap;
        Ok(cap)
    }

    /// `pci_root_bus_path()`.
    pub fn root_bus_path(&self) -> String {
        self.bus().map_or_else(|| "0000:00".to_string(), |b| b.root_bus_path())
    }

    /// `pci_intx()`: the 0-origin pin from the interrupt pin register.
    pub fn intx(&self) -> i32 {
        i32::from(self.lock().config[PCI_INTERRUPT_PIN]) - 1
    }

    /// `pci_set_irq()`: drives the pin named by the interrupt pin register.
    pub fn set_irq(&self, level: i32) {
        let intx = self.intx();
        assert!((0..PCI_NUM_PINS as i32).contains(&intx), "{} has no interrupt pin", self.name);
        self.irq_handler(intx as usize, level);
    }

    /// `pci_irq_handler()`: drives INTx pin `pin` (0 is INTA) to `level`.
    pub fn irq_handler(&self, pin: usize, level: i32) {
        let mut fx = Effects::new();
        self.lock().irq_handler(&mut fx, pin, level);
        self.run(fx);
    }

    /// `pci_allocate_irq()`: a line that drives this function's interrupt pin.
    pub fn allocate_irq(&self) -> IrqLine {
        let intx = self.intx();
        assert!((0..PCI_NUM_PINS as i32).contains(&intx), "{} has no interrupt pin", self.name);
        let dev = self.weak();
        IrqLine::from_fn(move |level| {
            if let Some(d) = dev.upgrade() {
                d.irq_handler(intx as usize, level);
            }
        })
    }

    /// `pci_device_deassert_intx()`.
    pub fn deassert_intx(&self) {
        let mut fx = Effects::new();
        self.lock().deassert_intx(&mut fx);
        self.run(fx);
    }

    /// `pci_irq_disabled()`.
    pub fn irq_disabled(&self) -> bool {
        self.lock().irq_disabled()
    }

    /// The asserted pins, bit 0 for INTA.
    pub fn irq_state(&self) -> u8 {
        self.lock().irq_state
    }

    /// `PCIDevice::enabled`.
    pub fn is_enabled(&self) -> bool {
        self.lock().enabled
    }

    /// `PCIDevice::is_master`: whether DMA and MSI writes reach memory.
    pub fn is_bus_master(&self) -> bool {
        self.lock().bus_master
    }

    /// `pci_set_enabled()`: a disabled function maps no BARs, does no DMA and is reset when it
    /// comes back.
    pub fn set_enabled(&self, state: bool) {
        {
            let mut s = self.lock();
            if s.enabled == state {
                return;
            }
            s.enabled = state;
            self.update_mappings_locked(&mut s);
            s.bus_master = s.word(PCI_COMMAND) & PCI_COMMAND_MASTER != 0 && s.enabled;
        }
        self.reset();
    }

    /// `pci_device_reset()`: the model's reset, then the generic one.
    pub fn reset(&self) {
        if let Some(ops) = self.ops() {
            ops.reset(self);
        }
        self.do_device_reset();
    }

    /// `pci_do_device_reset()`.
    pub fn do_device_reset(&self) {
        let mut fx = Effects::new();
        {
            let mut s = self.lock();
            s.deassert_intx(&mut fx);

            // Clear all writable bits.
            let m = pci_get_word(&s.wmask, PCI_COMMAND) | pci_get_word(&s.w1cmask, PCI_COMMAND);
            let v = s.word(PCI_COMMAND) & !m;
            s.set_word(PCI_COMMAND, v);
            let m = pci_get_word(&s.wmask, PCI_STATUS) | pci_get_word(&s.w1cmask, PCI_STATUS);
            let v = s.word(PCI_STATUS) & !m;
            s.set_word(PCI_STATUS, v);
            // Some devices make bits of the interrupt line read-only.
            let m = s.wmask[PCI_INTERRUPT_LINE] | s.w1cmask[PCI_INTERRUPT_LINE];
            s.config[PCI_INTERRUPT_LINE] &= !m;
            s.config[PCI_CACHE_LINE_SIZE] = 0;
            // The default PM state is D0.
            if s.pm_cap != 0 {
                let off = usize::from(s.pm_cap) + PCI_PM_CTRL;
                let v = s.word(off) & !PCI_PM_CTRL_STATE_MASK;
                s.set_word(off, v);
            }
            s.reset_regions();
            self.update_mappings_locked(&mut s);

            s.msi_reset();
            s.msix_reset(&mut fx);
        }
        self.run(fx);
    }
}

/// `pci_init_cmask()`.
fn init_cmask(s: &mut DevState) {
    let c = &mut s.cmask;
    pci_set_word(c, PCI_VENDOR_ID, 0xffff);
    pci_set_word(c, PCI_DEVICE_ID, 0xffff);
    c[PCI_STATUS] = PCI_STATUS_CAP_LIST as u8;
    c[PCI_REVISION_ID] = 0xff;
    c[PCI_CLASS_PROG] = 0xff;
    pci_set_word(c, PCI_CLASS_DEVICE, 0xffff);
    c[PCI_HEADER_TYPE] = 0xff;
    c[PCI_CAPABILITY_LIST] = 0xff;
}

/// `pci_init_wmask()`.
fn init_wmask(s: &mut DevState) {
    let w = &mut s.wmask;
    w[PCI_CACHE_LINE_SIZE] = 0xff;
    w[PCI_INTERRUPT_LINE] = 0xff;
    pci_set_word(
        w,
        PCI_COMMAND,
        PCI_COMMAND_IO
            | PCI_COMMAND_MEMORY
            | PCI_COMMAND_MASTER
            | PCI_COMMAND_INTX_DISABLE
            | PCI_COMMAND_SERR,
    );
    w[PCI_CONFIG_HEADER_SIZE..].fill(0xff);
}

/// `pci_init_w1cmask()`. Setting w1cmask on read-only bits is fine as long as they read as 0.
fn init_w1cmask(s: &mut DevState) {
    pci_set_word(
        &mut s.w1cmask,
        PCI_STATUS,
        PCI_STATUS_PARITY
            | PCI_STATUS_SIG_TARGET_ABORT
            | PCI_STATUS_REC_TARGET_ABORT
            | PCI_STATUS_REC_MASTER_ABORT
            | PCI_STATUS_SIG_SYSTEM_ERROR
            | PCI_STATUS_DETECTED_PARITY,
    );
}

/// `pci_init_mask_bridge()`.
fn init_mask_bridge(s: &mut DevState) {
    // Primary, secondary and subordinate bus numbers and the secondary latency timer.
    s.wmask[PCI_PRIMARY_BUS..PCI_PRIMARY_BUS + 4].fill(0xff);

    // Base and limit.
    s.wmask[PCI_IO_BASE] = PCI_IO_RANGE_MASK;
    s.wmask[PCI_IO_LIMIT] = PCI_IO_RANGE_MASK;
    pci_set_word(&mut s.wmask, PCI_MEMORY_BASE, PCI_MEMORY_RANGE_MASK);
    pci_set_word(&mut s.wmask, PCI_MEMORY_LIMIT, PCI_MEMORY_RANGE_MASK);
    pci_set_word(&mut s.wmask, PCI_PREF_MEMORY_BASE, PCI_PREF_RANGE_MASK);
    pci_set_word(&mut s.wmask, PCI_PREF_MEMORY_LIMIT, PCI_PREF_RANGE_MASK);

    // The upper 32 bits of the prefetchable base and limit.
    s.wmask[PCI_PREF_BASE_UPPER32..PCI_PREF_BASE_UPPER32 + 8].fill(0xff);

    // Supported memory and I/O types.
    s.config[PCI_IO_BASE] |= PCI_IO_RANGE_TYPE_16;
    s.config[PCI_IO_LIMIT] |= PCI_IO_RANGE_TYPE_16;
    let v = s.word(PCI_PREF_MEMORY_BASE) | PCI_PREF_RANGE_TYPE_64;
    s.set_word(PCI_PREF_MEMORY_BASE, v);
    let v = s.word(PCI_PREF_MEMORY_LIMIT) | PCI_PREF_RANGE_TYPE_64;
    s.set_word(PCI_PREF_MEMORY_LIMIT, v);

    // Bridges default to 10 bit VGA decoding but only 16 bit decoding is implemented, as in
    // QEMU.
    pci_set_word(
        &mut s.wmask,
        PCI_BRIDGE_CONTROL,
        PCI_BRIDGE_CTL_PARITY
            | PCI_BRIDGE_CTL_SERR
            | PCI_BRIDGE_CTL_ISA
            | PCI_BRIDGE_CTL_VGA
            | PCI_BRIDGE_CTL_VGA_16BIT
            | PCI_BRIDGE_CTL_MASTER_ABORT
            | PCI_BRIDGE_CTL_BUS_RESET
            | PCI_BRIDGE_CTL_FAST_BACK
            | PCI_BRIDGE_CTL_DISCARD
            | PCI_BRIDGE_CTL_SEC_DISCARD
            | PCI_BRIDGE_CTL_DISCARD_SERR,
    );
    // Never set by us, here for completeness like in QEMU.
    pci_set_word(&mut s.w1cmask, PCI_BRIDGE_CONTROL, PCI_BRIDGE_CTL_DISCARD_STATUS);
    s.cmask[PCI_IO_BASE] |= PCI_IO_RANGE_TYPE_MASK;
    s.cmask[PCI_IO_LIMIT] |= PCI_IO_RANGE_TYPE_MASK;
    let v = pci_get_word(&s.cmask, PCI_PREF_MEMORY_BASE) | PCI_PREF_RANGE_TYPE_MASK;
    pci_set_word(&mut s.cmask, PCI_PREF_MEMORY_BASE, v);
    let v = pci_get_word(&s.cmask, PCI_PREF_MEMORY_LIMIT) | PCI_PREF_RANGE_TYPE_MASK;
    pci_set_word(&mut s.cmask, PCI_PREF_MEMORY_LIMIT, v);
}

/// `pci_find_space()`: the first run of `size` unused bytes after the standard header.
fn find_space(s: &DevState, size: u8) -> u8 {
    let size = usize::from(size);
    let mut offset = PCI_CONFIG_HEADER_SIZE;
    for i in PCI_CONFIG_HEADER_SIZE..PCI_CONFIG_SPACE_SIZE {
        if s.used[i] != 0 {
            offset = i + 1;
        } else if i + 1 - offset == size {
            return offset as u8;
        }
    }
    0
}

/// The capability list is walked with a bound so that a broken list cannot hang us.
const CAP_WALK_LIMIT: usize = 64;

/// `pci_find_capability_list()`: the offset of `cap_id` and of the pointer that leads to it.
pub(crate) fn find_capability_list(s: &DevState, cap_id: u8) -> (u8, usize) {
    if s.word(PCI_STATUS) & PCI_STATUS_CAP_LIST == 0 {
        return (0, PCI_CAPABILITY_LIST);
    }
    let mut prev = PCI_CAPABILITY_LIST;
    for _ in 0..CAP_WALK_LIMIT {
        let next = s.config[prev];
        if next == 0 || s.config[usize::from(next) + PCI_CAP_LIST_ID] == cap_id {
            return (next, prev);
        }
        prev = usize::from(next) + PCI_CAP_LIST_NEXT;
    }
    (0, prev)
}

/// `pci_find_capability_at_offset()`: the start of the capability covering `offset`, or 0.
fn find_capability_at_offset(s: &DevState, offset: usize) -> u8 {
    if offset >= s.used.len() || s.used[offset] == 0 {
        return 0;
    }
    let mut found = 0u8;
    let mut prev = PCI_CAPABILITY_LIST;
    for _ in 0..CAP_WALK_LIMIT {
        let next = s.config[prev];
        if next == 0 {
            break;
        }
        if usize::from(next) <= offset && next > found {
            found = next;
        }
        prev = usize::from(next) + PCI_CAP_LIST_NEXT;
    }
    found
}

/// `pci_del_capability()`.
pub(crate) fn del_capability_locked(s: &mut DevState, cap_id: u8, size: u8) {
    let (offset, prev) = find_capability_list(s, cap_id);
    if offset == 0 {
        return;
    }
    let o = usize::from(offset);
    let size = usize::from(size);
    s.config[prev] = s.config[o + PCI_CAP_LIST_NEXT];
    // Make the capability writable again.
    s.wmask[o..o + size].fill(0xff);
    s.w1cmask[o..o + size].fill(0);
    // Device specific registers cannot be checked.
    s.cmask[o..o + size].fill(0);
    let used_end = (o + size.next_multiple_of(4)).min(s.used.len());
    s.used[o..used_end].fill(0);

    if s.config[PCI_CAPABILITY_LIST] == 0 {
        let v = s.word(PCI_STATUS) & !PCI_STATUS_CAP_LIST;
        s.set_word(PCI_STATUS, v);
    }
}
