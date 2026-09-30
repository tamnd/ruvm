// SPDX-License-Identifier: GPL-2.0-or-later

//! The pvpanic device, hw/misc/pvpanic.c, pvpanic-isa.c and pvpanic-pci.c.
//!
//! A guest kernel with a pvpanic driver writes an event bit to a one byte register when it
//! panics, when it has loaded a crash kernel, or when it wants the host to shut it down. Reading
//! the register tells the driver which events the device supports.
//!
//! The device does not know about run states or QMP events. Every guest write that carries a
//! known event is handed to a [`PvPanicHandler`], and the machine decides what to do with it:
//! `qemu_system_guest_panicked()`, `qemu_system_guest_crashloaded()` or
//! `qemu_system_guest_pvshutdown()` in QEMU.
//!
//! There are two front ends. [`PvPanicIsa`] is `-device pvpanic`, a single port at 0x505 that
//! firmware finds through the `etc/pvpanic-port` fw_cfg file. [`PvPanicPci`] is
//! `-device pvpanic-pci`, a 1b36:0011 function with the register in a two byte memory BAR.

use std::fmt;
use std::sync::Arc;

use ruvm_base::{Error, Result};
use ruvm_hw_core::fw_cfg::FwCfgState;
use ruvm_hw_pci::regs::{PCI_BASE_ADDRESS_SPACE_MEMORY, PCI_VENDOR_ID_REDHAT};
use ruvm_hw_pci::{PciBus, PciDevice, PciDeviceInfo};
use ruvm_mem::{
    AccessConstraints, AccessCtx, AccessSize, MemResult, MemorySystem, MmioOps, RegionId,
};

/// The guest panicked, `PVPANIC_PANICKED`.
pub const PVPANIC_PANICKED: u8 = 1 << 0;
/// The guest loaded a crash kernel after a panic, `PVPANIC_CRASH_LOADED`.
pub const PVPANIC_CRASH_LOADED: u8 = 1 << 1;
/// The guest asks to be shut down, `PVPANIC_SHUTDOWN`.
pub const PVPANIC_SHUTDOWN: u8 = 1 << 2;
/// Every event the device knows, and the default of the `events` property.
pub const PVPANIC_EVENTS: u8 = PVPANIC_PANICKED | PVPANIC_CRASH_LOADED | PVPANIC_SHUTDOWN;

/// The default of the ISA device's `ioport` property.
pub const PVPANIC_ISA_DEFAULT_IOPORT: u16 = 0x505;
/// The fw_cfg file that tells firmware where the ISA device lives.
pub const PVPANIC_FW_CFG_FILE: &str = "etc/pvpanic-port";

/// `PCI_DEVICE_ID_REDHAT_PVPANIC`.
pub const PCI_DEVICE_ID_REDHAT_PVPANIC: u16 = 0x0011;
/// `PCI_CLASS_SYSTEM_OTHER`, the class of the PCI device.
pub const PCI_CLASS_SYSTEM_OTHER: u16 = 0x0880;

/// Size of the register window of the ISA device, in bytes.
const PVPANIC_ISA_SIZE: u128 = 1;
/// Size of BAR 0 of the PCI device, in bytes.
const PVPANIC_PCI_BAR_SIZE: u128 = 2;

/// An event reported by the guest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PvPanicEvent {
    /// `GUEST_PANICKED`.
    Panicked,
    /// `GUEST_CRASHLOADED`.
    CrashLoaded,
    /// `GUEST_PVSHUTDOWN`, followed by a guest initiated shutdown.
    Shutdown,
}

impl PvPanicEvent {
    /// The bit of this event in the register.
    pub const fn bit(self) -> u8 {
        match self {
            PvPanicEvent::Panicked => PVPANIC_PANICKED,
            PvPanicEvent::CrashLoaded => PVPANIC_CRASH_LOADED,
            PvPanicEvent::Shutdown => PVPANIC_SHUTDOWN,
        }
    }

    /// The event a guest write of `value` reports. When several bits are set only the first one
    /// counts, in the order panicked, crash loaded, shutdown, as in `handle_event()`. Unknown
    /// bits are ignored.
    pub const fn from_value(value: u64) -> Option<PvPanicEvent> {
        let value = value as u8;
        if value & PVPANIC_PANICKED != 0 {
            Some(PvPanicEvent::Panicked)
        } else if value & PVPANIC_CRASH_LOADED != 0 {
            Some(PvPanicEvent::CrashLoaded)
        } else if value & PVPANIC_SHUTDOWN != 0 {
            Some(PvPanicEvent::Shutdown)
        } else {
            None
        }
    }
}

/// Called for every event the guest reports.
pub type PvPanicHandler = Arc<dyn Fn(PvPanicEvent) + Send + Sync>;

/// The register shared by both front ends, `PVPanicState` and `pvpanic_ops`.
pub struct PvPanicState {
    events: u8,
    handler: PvPanicHandler,
}

impl fmt::Debug for PvPanicState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PvPanicState").field("events", &self.events).finish_non_exhaustive()
    }
}

impl PvPanicState {
    /// A register that advertises `events` to the guest and reports writes to `handler`.
    pub fn new(events: u8, handler: PvPanicHandler) -> PvPanicState {
        PvPanicState { events, handler }
    }

    /// The `events` property, which is what guest reads return.
    pub fn events(&self) -> u8 {
        self.events
    }

    /// `handle_event()`: reports the event in `value`, if any.
    ///
    /// Like QEMU, the `events` property is not checked here. It only controls what the guest
    /// sees when it reads the register.
    pub fn handle_event(&self, value: u64) {
        if let Some(event) = PvPanicEvent::from_value(value) {
            (self.handler)(event);
        }
    }
}

impl MmioOps for PvPanicState {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(self.events.into())
    }

    fn write(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.handle_event(value);
        Ok(())
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 1)
    }
}

fn mem_error(e: ruvm_mem::MemError) -> Error {
    Error::generic(e.to_string())
}

/// `pvpanic_setup_io()`: the register as an I/O region of `size` bytes.
fn setup_io(mem: &MemorySystem, state: &Arc<PvPanicState>, size: u128) -> Result<RegionId> {
    let ops: Arc<dyn MmioOps> = Arc::clone(state) as Arc<dyn MmioOps>;
    mem.new_io("pvpanic", size, ops).map_err(mem_error)
}

/// The contents of the `etc/pvpanic-port` fw_cfg file for a device at `ioport`.
pub fn pvpanic_port_file(ioport: u16) -> Vec<u8> {
    ioport.to_le_bytes().to_vec()
}

/// The properties of `-device pvpanic`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PvPanicIsaConfig {
    /// `ioport`, 0x505 by default.
    pub ioport: u16,
    /// `events`, all of [`PVPANIC_EVENTS`] by default.
    pub events: u8,
}

impl Default for PvPanicIsaConfig {
    fn default() -> Self {
        PvPanicIsaConfig { ioport: PVPANIC_ISA_DEFAULT_IOPORT, events: PVPANIC_EVENTS }
    }
}

/// `-device pvpanic`, `PVPanicISAState`.
#[derive(Debug)]
pub struct PvPanicIsa {
    state: Arc<PvPanicState>,
    ioport: u16,
    region: RegionId,
    mapped: bool,
}

impl PvPanicIsa {
    /// `pvpanic_isa_initfn()` and `pvpanic_isa_realizefn()`.
    ///
    /// The register is mapped at `config.ioport` in `io_space` and the port is published as
    /// `etc/pvpanic-port`. QEMU does both only when the machine has fw_cfg, so without one the
    /// device is created but the guest cannot reach it. That is kept here: pass `None` for
    /// `fw_cfg` and [`PvPanicIsa::is_mapped`] is false.
    pub fn realize(
        mem: &MemorySystem,
        io_space: RegionId,
        fw_cfg: Option<&FwCfgState>,
        config: PvPanicIsaConfig,
        handler: PvPanicHandler,
    ) -> Result<PvPanicIsa> {
        let state = Arc::new(PvPanicState::new(config.events, handler));
        let region = setup_io(mem, &state, PVPANIC_ISA_SIZE)?;
        let mut dev = PvPanicIsa { state, ioport: config.ioport, region, mapped: false };
        let Some(fw_cfg) = fw_cfg else {
            return Ok(dev);
        };
        fw_cfg.add_file(PVPANIC_FW_CFG_FILE, pvpanic_port_file(config.ioport))?;
        mem.add_subregion(io_space, config.ioport.into(), region).map_err(mem_error)?;
        dev.mapped = true;
        Ok(dev)
    }

    /// The `ioport` property.
    pub fn ioport(&self) -> u16 {
        self.ioport
    }

    /// The `events` property.
    pub fn events(&self) -> u8 {
        self.state.events()
    }

    /// The one byte I/O region of the register.
    pub fn region(&self) -> RegionId {
        self.region
    }

    /// Whether realize mapped the register into the I/O space.
    pub fn is_mapped(&self) -> bool {
        self.mapped
    }

    /// The shared register state.
    pub fn state(&self) -> &Arc<PvPanicState> {
        &self.state
    }
}

/// `-device pvpanic-pci`, `PVPanicPCIState`.
#[derive(Debug)]
pub struct PvPanicPci {
    state: Arc<PvPanicState>,
    dev: Arc<PciDevice>,
    region: RegionId,
}

impl PvPanicPci {
    /// `pvpanic_pci_realizefn()`: plugs the function into `bus` at `devfn`, or the first free
    /// slot, with the register behind a two byte memory BAR 0.
    pub fn realize(
        bus: &PciBus,
        devfn: Option<u8>,
        events: u8,
        handler: PvPanicHandler,
    ) -> Result<PvPanicPci> {
        let info = PciDeviceInfo {
            name: "pvpanic-pci".to_string(),
            vendor_id: PCI_VENDOR_ID_REDHAT,
            device_id: PCI_DEVICE_ID_REDHAT_PVPANIC,
            revision: 1,
            class_id: PCI_CLASS_SYSTEM_OTHER,
            ..PciDeviceInfo::default()
        };
        let dev = bus.register_device(&info, devfn)?;
        let state = Arc::new(PvPanicState::new(events, handler));
        let region = setup_io(bus.memory(), &state, PVPANIC_PCI_BAR_SIZE)?;
        dev.register_bar(0, PCI_BASE_ADDRESS_SPACE_MEMORY, region);
        Ok(PvPanicPci { state, dev, region })
    }

    /// The PCI function.
    pub fn pci_device(&self) -> &Arc<PciDevice> {
        &self.dev
    }

    /// The `events` property.
    pub fn events(&self) -> u8 {
        self.state.events()
    }

    /// The region behind BAR 0.
    pub fn region(&self) -> RegionId {
        self.region
    }

    /// The shared register state.
    pub fn state(&self) -> &Arc<PvPanicState> {
        &self.state
    }
}
