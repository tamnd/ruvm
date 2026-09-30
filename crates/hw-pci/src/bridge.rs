// SPDX-License-Identifier: GPL-2.0-or-later

//! A basic PCI-to-PCI bridge, from hw/pci/pci_bridge.c.
//!
//! The bridge is a function with a type 1 header on its parent bus and owns a secondary bus.
//! The secondary bus has its own memory and I/O containers. Three windows, set by the base and
//! limit registers and gated by the command register, alias ranges of those containers into
//! the parent bus's containers at the same addresses: prefetchable memory, memory and I/O.
//!
//! INTx from the secondary bus is swizzled onto the bridge's own pins. Setting the secondary
//! bus reset bit in the bridge control register resets everything below.
//!
//! The VGA windows, the subsystem ID capability helper and the PCIe specific parts are not
//! ported.

use std::fmt;
use std::sync::{Arc, Mutex};

use ruvm_base::Error;
use ruvm_mem::{MemorySystem, RegionId};

use crate::bus::{PciBus, pci_swizzle_map_irq_fn};
use crate::device::{PciDevice, PciDeviceInfo, PciDeviceOps};
use crate::regs::*;

/// `pci_config_get_io_base()`.
fn io_base(c: &[u8], base: usize, upper16: usize) -> u64 {
    let mut val = u64::from(c[base] & PCI_IO_RANGE_MASK) << 8;
    if c[base] & PCI_IO_RANGE_TYPE_32 != 0 {
        val |= u64::from(pci_get_word(c, upper16)) << 16;
    }
    val
}

/// `pci_config_get_memory_base()`.
fn memory_base(c: &[u8], base: usize) -> u64 {
    u64::from(pci_get_word(c, base) & PCI_MEMORY_RANGE_MASK) << 16
}

/// `pci_config_get_pref_base()`.
fn pref_base(c: &[u8], base: usize, upper: usize) -> u64 {
    let tmp = pci_get_word(c, base);
    let mut val = u64::from(tmp & PCI_PREF_RANGE_MASK) << 16;
    if tmp & PCI_PREF_RANGE_TYPE_64 != 0 {
        val |= u64::from(pci_get_long(c, upper)) << 32;
    }
    val
}

/// `pci_bridge_get_base()` on a copy of the bridge's config space.
pub fn pci_bridge_get_base(config: &[u8], type_: u8) -> u64 {
    if type_ & PCI_BASE_ADDRESS_SPACE_IO != 0 {
        io_base(config, PCI_IO_BASE, PCI_IO_BASE_UPPER16)
    } else if type_ & PCI_BASE_ADDRESS_MEM_PREFETCH != 0 {
        pref_base(config, PCI_PREF_MEMORY_BASE, PCI_PREF_BASE_UPPER32)
    } else {
        memory_base(config, PCI_MEMORY_BASE)
    }
}

/// `pci_bridge_get_limit()` on a copy of the bridge's config space.
pub fn pci_bridge_get_limit(config: &[u8], type_: u8) -> u64 {
    if type_ & PCI_BASE_ADDRESS_SPACE_IO != 0 {
        // PCI bridge spec 3.2.5.6.
        io_base(config, PCI_IO_LIMIT, PCI_IO_LIMIT_UPPER16) | 0xfff
    } else {
        let limit = if type_ & PCI_BASE_ADDRESS_MEM_PREFETCH != 0 {
            pref_base(config, PCI_PREF_MEMORY_LIMIT, PCI_PREF_LIMIT_UPPER32)
        } else {
            memory_base(config, PCI_MEMORY_LIMIT)
        };
        // PCI bridge spec 3.2.5.1 and 3.2.5.8.
        limit | 0xfffff
    }
}

/// One forwarding window: where it sits and how big it is (0 when closed).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PciBridgeWindow {
    pub base: u64,
    pub size: u128,
}

#[derive(Default)]
struct Windows {
    /// The aliases currently mapped in the parent bus: prefetchable memory, memory, I/O.
    aliases: Vec<(RegionId, RegionId)>,
    pref_mem: PciBridgeWindow,
    mem: PciBridgeWindow,
    io: PciBridgeWindow,
}

struct BridgeInner {
    memory: Arc<MemorySystem>,
    sec_bus: Arc<PciBus>,
    /// The secondary bus's containers, `address_space_mem` and `address_space_io`.
    sec_mem: RegionId,
    sec_io: RegionId,
    /// The parent bus's containers the windows map into.
    parent_mem: RegionId,
    parent_io: RegionId,
    windows: Mutex<Windows>,
}

impl BridgeInner {
    fn windows(&self) -> std::sync::MutexGuard<'_, Windows> {
        self.windows.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// `pci_bridge_update_mappings()`: drops the old windows and maps new ones, in one
    /// transaction.
    fn update_mappings(&self, dev: &PciDevice) {
        let config = dev.config_bytes();
        let cmd = pci_get_word(&config, PCI_COMMAND);
        let mem = &self.memory;
        let mut w = self.windows();
        let _t = mem.transaction();
        for (container, alias) in w.aliases.drain(..) {
            let _ = mem.del_subregion(container, alias);
            let _ = mem.destroy_region(alias);
        }
        let specs = [
            (
                PCI_BASE_ADDRESS_MEM_PREFETCH,
                "pci_bridge_pref_mem",
                self.sec_mem,
                self.parent_mem,
                cmd & PCI_COMMAND_MEMORY != 0,
            ),
            (
                PCI_BASE_ADDRESS_SPACE_MEMORY,
                "pci_bridge_mem",
                self.sec_mem,
                self.parent_mem,
                cmd & PCI_COMMAND_MEMORY != 0,
            ),
            (
                PCI_BASE_ADDRESS_SPACE_IO,
                "pci_bridge_io",
                self.sec_io,
                self.parent_io,
                cmd & PCI_COMMAND_IO != 0,
            ),
        ];
        let mut out = [PciBridgeWindow::default(); 3];
        for (i, (type_, name, space, parent, enabled)) in specs.into_iter().enumerate() {
            // `pci_bridge_init_alias()`. Like QEMU this cannot express base 0 with a limit of
            // 2^64 - 1.
            let base = pci_bridge_get_base(&config, type_);
            let limit = pci_bridge_get_limit(&config, type_);
            let size =
                if enabled && limit >= base { u128::from(limit) + 1 - u128::from(base) } else { 0 };
            out[i] = PciBridgeWindow { base, size };
            let Ok(alias) = mem.new_alias(name, space, base, size) else { continue };
            if mem.add_subregion_overlap(parent, base, alias, 1).is_ok() {
                w.aliases.push((parent, alias));
            } else {
                let _ = mem.destroy_region(alias);
            }
        }
        [w.pref_mem, w.mem, w.io] = out;
    }
}

/// The config and reset hooks of a bridge.
struct BridgeOps(Arc<BridgeInner>);

impl PciDeviceOps for BridgeOps {
    /// `pci_bridge_write_config()`.
    fn config_write(&self, dev: &PciDevice, addr: u32, val: u32, len: u32) {
        let oldctl = pci_get_word(&dev.config_bytes(), PCI_BRIDGE_CONTROL);
        dev.default_write_config(addr, val, len);

        let (a, l) = (u64::from(addr), u64::from(len));
        if ranges_overlap(a, l, PCI_COMMAND as u64, 2)
            // I/O base and limit.
            || ranges_overlap(a, l, PCI_IO_BASE as u64, 2)
            // Memory and prefetchable base and limit, and the I/O upper 16 bits.
            || ranges_overlap(a, l, PCI_MEMORY_BASE as u64, 20)
            // VGA enable.
            || ranges_overlap(a, l, PCI_BRIDGE_CONTROL as u64, 2)
        {
            self.0.update_mappings(dev);
        }

        let newctl = pci_get_word(&dev.config_bytes(), PCI_BRIDGE_CONTROL);
        if !oldctl & newctl & PCI_BRIDGE_CTL_BUS_RESET != 0 {
            // A hot reset on the 0 to 1 transition.
            self.0.sec_bus.reset();
        }
    }

    /// `pci_bridge_reset()`, plus the reset of everything on the secondary bus.
    fn reset(&self, dev: &PciDevice) {
        dev.with_config(|c| {
            let conf = c.config;
            conf[PCI_PRIMARY_BUS] = 0;
            conf[PCI_SECONDARY_BUS] = 0;
            conf[PCI_SUBORDINATE_BUS] = 0;
            conf[PCI_SEC_LATENCY_TIMER] = 0;
            // The spec does not give defaults for base and limit. Like QEMU, clear the address
            // bits and keep the type bits.
            conf[PCI_IO_BASE] &= !PCI_IO_RANGE_MASK;
            conf[PCI_IO_LIMIT] &= !PCI_IO_RANGE_MASK;
            for off in [PCI_MEMORY_BASE, PCI_MEMORY_LIMIT] {
                let v = pci_get_word(conf, off) & !PCI_MEMORY_RANGE_MASK;
                pci_set_word(conf, off, v);
            }
            for off in [PCI_PREF_MEMORY_BASE, PCI_PREF_MEMORY_LIMIT] {
                let v = pci_get_word(conf, off) & !PCI_PREF_RANGE_MASK;
                pci_set_word(conf, off, v);
            }
            pci_set_long(conf, PCI_PREF_BASE_UPPER32, 0);
            pci_set_long(conf, PCI_PREF_LIMIT_UPPER32, 0);
            pci_set_word(conf, PCI_BRIDGE_CONTROL, 0);
            // The generic reset that follows clears the command register.
            let cmd = pci_get_word(conf, PCI_COMMAND) & !(PCI_COMMAND_IO | PCI_COMMAND_MEMORY);
            pci_set_word(conf, PCI_COMMAND, cmd);
        });
        self.0.update_mappings(dev);
        self.0.sec_bus.reset();
    }
}

/// A PCI-to-PCI bridge and its secondary bus, `PCIBridge`.
pub struct PciBridge {
    dev: Arc<PciDevice>,
    inner: Arc<BridgeInner>,
}

impl fmt::Debug for PciBridge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PciBridge")
            .field("dev", &self.dev)
            .field("sec_bus", &self.inner.sec_bus)
            .finish_non_exhaustive()
    }
}

impl PciBridge {
    /// Registers a bridge on `parent` and creates its secondary bus, `pci_bridge_initfn()`.
    /// `info.is_bridge` is forced on. The secondary bus uses the standard swizzle.
    pub fn new(
        parent: &Arc<PciBus>,
        info: &PciDeviceInfo,
        devfn: Option<u8>,
        sec_bus_name: &str,
    ) -> Result<PciBridge, Error> {
        let info = PciDeviceInfo {
            is_bridge: true,
            subsystem_vendor_id: 0,
            subsystem_id: 0,
            ..info.clone()
        };
        let dev = parent.register_device(&info, devfn)?;
        dev.with_config(|c| {
            let st = pci_get_word(c.config, PCI_STATUS) | PCI_STATUS_66MHZ | PCI_STATUS_FAST_BACK;
            pci_set_word(c.config, PCI_STATUS, st);
            pci_set_word(c.config, PCI_CLASS_DEVICE, PCI_CLASS_BRIDGE_PCI);
            c.config[PCI_HEADER_TYPE] = (c.config[PCI_HEADER_TYPE]
                & PCI_HEADER_TYPE_MULTI_FUNCTION)
                | PCI_HEADER_TYPE_BRIDGE;
            pci_set_word(c.config, PCI_SEC_STATUS, PCI_STATUS_66MHZ | PCI_STATUS_FAST_BACK);
        });

        let memory = Arc::clone(parent.memory());
        let err = |e: ruvm_mem::MemError| Error::generic(e.to_string());
        let sec_mem = memory.new_container("pci_bridge_pci", 1 << 64).map_err(err)?;
        let sec_io = memory.new_container("pci_bridge_io", 1 << 32).map_err(err)?;
        let sec_bus =
            PciBus::new_secondary(sec_bus_name, Arc::clone(&memory), sec_mem, sec_io, &dev);
        sec_bus.set_map_irq(Arc::new(pci_swizzle_map_irq_fn));

        let inner = Arc::new(BridgeInner {
            memory,
            sec_bus,
            sec_mem,
            sec_io,
            parent_mem: parent.mem_space(),
            parent_io: parent.io_space(),
            windows: Mutex::new(Windows::default()),
        });
        inner.update_mappings(&dev);
        dev.set_ops(Arc::new(BridgeOps(Arc::clone(&inner))));
        Ok(PciBridge { dev, inner })
    }

    /// The bridge function on the parent bus.
    pub fn device(&self) -> &Arc<PciDevice> {
        &self.dev
    }

    /// `pci_bridge_get_sec_bus()`.
    pub fn sec_bus(&self) -> &Arc<PciBus> {
        &self.inner.sec_bus
    }

    /// The prefetchable memory window as currently mapped.
    pub fn pref_mem_window(&self) -> PciBridgeWindow {
        self.inner.windows().pref_mem
    }

    /// The memory window as currently mapped.
    pub fn mem_window(&self) -> PciBridgeWindow {
        self.inner.windows().mem
    }

    /// The I/O window as currently mapped.
    pub fn io_window(&self) -> PciBridgeWindow {
        self.inner.windows().io
    }

    /// `pci_bridge_update_mappings()`.
    pub fn update_mappings(&self) {
        self.inner.update_mappings(&self.dev);
    }

    /// `pci_bridge_disable_base_limit()`: closes all windows by setting base above limit.
    pub fn disable_base_limit(&self) {
        self.dev.with_config(|c| {
            let conf = c.config;
            conf[PCI_IO_BASE] |= PCI_IO_RANGE_MASK;
            conf[PCI_IO_LIMIT] &= !PCI_IO_RANGE_MASK;
            let v = pci_get_word(conf, PCI_MEMORY_BASE) | PCI_MEMORY_RANGE_MASK;
            pci_set_word(conf, PCI_MEMORY_BASE, v);
            let v = pci_get_word(conf, PCI_MEMORY_LIMIT) & !PCI_MEMORY_RANGE_MASK;
            pci_set_word(conf, PCI_MEMORY_LIMIT, v);
            let v = pci_get_word(conf, PCI_PREF_MEMORY_BASE) | PCI_PREF_RANGE_MASK;
            pci_set_word(conf, PCI_PREF_MEMORY_BASE, v);
            let v = pci_get_word(conf, PCI_PREF_MEMORY_LIMIT) & !PCI_PREF_RANGE_MASK;
            pci_set_word(conf, PCI_PREF_MEMORY_LIMIT, v);
            pci_set_long(conf, PCI_PREF_BASE_UPPER32, 0);
            pci_set_long(conf, PCI_PREF_LIMIT_UPPER32, 0);
        });
    }
}
