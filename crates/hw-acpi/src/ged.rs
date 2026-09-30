// SPDX-License-Identifier: GPL-2.0-or-later

//! The ACPI Generic Event Device from hw/acpi/generic_event_device.c, the hardware-reduced event
//! source microvm and the Arm virt board use (ACPI 6.1, 5.6.9).
//!
//! One interrupt line carries up to 32 event kinds. Before pulsing it the device sets the
//! event's bit in the selector, which the guest's `_EVT` method reads (and so clears) through the
//! 4 byte `acpi-ged` region, [`AcpiGed::evt_ops`]. The 3 byte `acpi-ged-regs` region,
//! [`AcpiGed::regs_ops`], has the hardware-reduced sleep control, sleep status and reset
//! registers: SLP_EN with SLP_TYP 5 asks for a shutdown and writing
//! [`ACPI_GED_RESET_VALUE`] to the reset register asks for a reset. Both go to the
//! [`SystemRequestHandler`].
//!
//! Not ported: the memory, CPU and PCI hotplug containers (the event bits are accepted and
//! [`AcpiGed::send_event`] still signals them), GHES, the AML builder (that lives in
//! ruvm-firmware), VMState, trace points and QOM registration.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use ruvm_base::{Error, Result};
use ruvm_hw_core::IrqPin;
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MemResult, MmioOps};

use crate::core::{
    ACPI_CPU_HOTPLUG_STATUS, ACPI_GENERIC_ERROR, ACPI_MEMORY_HOTPLUG_STATUS,
    ACPI_NVDIMM_HOTPLUG_STATUS, ACPI_PCI_HOTPLUG_STATUS, ACPI_POWER_DOWN_STATUS, SystemRequest,
    SystemRequestHandler,
};

/// `TYPE_ACPI_GED`.
pub const TYPE_ACPI_GED: &str = "acpi-ged";

pub const ACPI_GED_EVT_SEL_OFFSET: u64 = 0x0;
pub const ACPI_GED_EVT_SEL_LEN: u64 = 0x4;

pub const ACPI_GED_REG_SLEEP_CTL: u64 = 0x00;
pub const ACPI_GED_REG_SLEEP_STS: u64 = 0x01;
pub const ACPI_GED_REG_RESET: u64 = 0x02;
pub const ACPI_GED_REG_COUNT: u64 = 0x03;

/// What the guest writes to the reset register to reset the machine.
pub const ACPI_GED_RESET_VALUE: u8 = 0x42;

/// SLP_TYPx bit offset in the sleep control register.
pub const ACPI_GED_SLP_TYP_POS: u32 = 0x2;
/// SLP_TYPx 3 bit mask.
pub const ACPI_GED_SLP_TYP_MASK: u8 = 0x07;
/// System `_S5` state, soft off.
pub const ACPI_GED_SLP_TYP_S5: u8 = 0x05;
/// SLP_EN, a write only bit.
pub const ACPI_GED_SLP_EN: u8 = 0x20;

pub const ACPI_GED_MEM_HOTPLUG_EVT: u32 = 0x1;
pub const ACPI_GED_PWR_DOWN_EVT: u32 = 0x2;
pub const ACPI_GED_NVDIMM_HOTPLUG_EVT: u32 = 0x4;
pub const ACPI_GED_CPU_HOTPLUG_EVT: u32 = 0x8;
pub const ACPI_GED_PCI_HOTPLUG_EVT: u32 = 0x10;
pub const ACPI_GED_ERROR_EVT: u32 = 0x20;

/// `ged_supported_events`.
pub const GED_SUPPORTED_EVENTS: [u32; 6] = [
    ACPI_GED_MEM_HOTPLUG_EVT,
    ACPI_GED_PWR_DOWN_EVT,
    ACPI_GED_NVDIMM_HOTPLUG_EVT,
    ACPI_GED_CPU_HOTPLUG_EVT,
    ACPI_GED_PCI_HOTPLUG_EVT,
    ACPI_GED_ERROR_EVT,
];

/// microvm's `GED_MMIO_BASE`, where the event selector goes.
pub const MICROVM_GED_MMIO_BASE: u64 = 0xfea0_0000;
/// microvm's `GED_MMIO_BASE_MEMHP`.
pub const MICROVM_GED_MMIO_BASE_MEMHP: u64 = MICROVM_GED_MMIO_BASE + 0x100;
/// microvm's `GED_MMIO_BASE_REGS`, where the sleep and reset registers go.
pub const MICROVM_GED_MMIO_BASE_REGS: u64 = MICROVM_GED_MMIO_BASE + 0x200;
/// microvm's `GED_MMIO_IRQ`.
pub const MICROVM_GED_MMIO_IRQ: u32 = 9;

/// The `acpi-ged` properties.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct AcpiGedProps {
    /// `ged-event`, the bitmap of events the board supports.
    pub ged_event: u32,
    /// `acpi-pci-hotplug-with-bridge-support`, which adds [`ACPI_GED_PCI_HOTPLUG_EVT`].
    pub pci_hotplug: bool,
}

/// `AcpiGedState` less the hotplug state.
pub struct AcpiGed {
    ged_event_bitmap: u32,
    sel: Mutex<u32>,
    irq: IrqPin,
    handler: Mutex<Option<SystemRequestHandler>>,
}

impl fmt::Debug for AcpiGed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AcpiGed")
            .field("ged_event_bitmap", &self.ged_event_bitmap)
            .field("sel", &*lock(&self.sel))
            .finish_non_exhaustive()
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

impl AcpiGed {
    /// `acpi_ged_initfn()` and `acpi_ged_realize()`. Fails where QEMU aborts, for event bits
    /// outside [`GED_SUPPORTED_EVENTS`].
    pub fn new(props: AcpiGedProps) -> Result<Arc<Self>> {
        let mut bitmap = props.ged_event;
        if props.pci_hotplug {
            bitmap |= ACPI_GED_PCI_HOTPLUG_EVT;
        }
        let supported = GED_SUPPORTED_EVENTS.iter().fold(0, |a, e| a | e);
        if bitmap & !supported != 0 {
            return Err(Error::generic("Unsupported events specified"));
        }
        Ok(Arc::new(AcpiGed {
            ged_event_bitmap: bitmap,
            sel: Mutex::new(0),
            irq: IrqPin::new(),
            handler: Mutex::new(None),
        }))
    }

    /// The `ged-event` bitmap after realize.
    pub fn ged_event_bitmap(&self) -> u32 {
        self.ged_event_bitmap
    }

    /// The sysbus IRQ, pulsed for every event.
    pub fn irq(&self) -> &IrqPin {
        &self.irq
    }

    /// Where shutdown and reset requests go.
    pub fn set_request_handler(&self, handler: SystemRequestHandler) {
        *lock(&self.handler) = Some(handler);
    }

    fn request(&self, req: SystemRequest) {
        let h = lock(&self.handler).clone();
        if let Some(h) = h {
            h(req);
        }
    }

    /// The selector without the read side effect.
    pub fn sel(&self) -> u32 {
        *lock(&self.sel)
    }

    /// `acpi_ged_send_event()`: maps `AcpiEventStatusBits` to a selector bit, sets it and
    /// pulses the IRQ. Returns false, like QEMU's warning, for events the GED cannot carry.
    pub fn send_event(&self, ev: u32) -> bool {
        let sel = if ev & ACPI_MEMORY_HOTPLUG_STATUS != 0 {
            ACPI_GED_MEM_HOTPLUG_EVT
        } else if ev & ACPI_POWER_DOWN_STATUS != 0 {
            ACPI_GED_PWR_DOWN_EVT
        } else if ev & ACPI_GENERIC_ERROR != 0 {
            ACPI_GED_ERROR_EVT
        } else if ev & ACPI_NVDIMM_HOTPLUG_STATUS != 0 {
            ACPI_GED_NVDIMM_HOTPLUG_EVT
        } else if ev & ACPI_CPU_HOTPLUG_STATUS != 0 {
            ACPI_GED_CPU_HOTPLUG_EVT
        } else if ev & ACPI_PCI_HOTPLUG_STATUS != 0 {
            ACPI_GED_PCI_HOTPLUG_EVT
        } else {
            return false;
        };
        *lock(&self.sel) |= sel;
        self.irq.pulse();
        true
    }

    /// The power button, what the board's powerdown notifier does with
    /// `acpi_send_event(ged, ACPI_POWER_DOWN_STATUS)`.
    pub fn power_down(&self) {
        self.send_event(ACPI_POWER_DOWN_STATUS);
    }

    /// `ged_evt_read()`: reading the selector returns it and clears it.
    pub fn evt_read(&self, addr: u64) -> u32 {
        match addr {
            ACPI_GED_EVT_SEL_OFFSET => std::mem::take(&mut *lock(&self.sel)),
            _ => 0,
        }
    }

    /// `ged_regs_read()`: the registers read as 0.
    pub fn regs_read(&self, _addr: u64) -> u8 {
        0
    }

    /// `ged_regs_write()`.
    pub fn regs_write(&self, addr: u64, data: u8) {
        match addr {
            ACPI_GED_REG_SLEEP_CTL => {
                let slp_typ = (data >> ACPI_GED_SLP_TYP_POS) & ACPI_GED_SLP_TYP_MASK;
                let slp_en = data & ACPI_GED_SLP_EN != 0;
                if slp_en && slp_typ == ACPI_GED_SLP_TYP_S5 {
                    self.request(SystemRequest::Shutdown);
                }
            }
            ACPI_GED_REG_RESET if data == ACPI_GED_RESET_VALUE => {
                self.request(SystemRequest::Reset);
            }
            // Sleep status writes and other reset values are ignored.
            _ => {}
        }
    }

    /// The `acpi-ged` region, [`ACPI_GED_EVT_SEL_LEN`] bytes. Sysbus MMIO 0.
    pub fn evt_ops(self: &Arc<Self>) -> Arc<GedEvtOps> {
        Arc::new(GedEvtOps(self.clone()))
    }

    /// The `acpi-ged-regs` region, [`ACPI_GED_REG_COUNT`] bytes. Sysbus MMIO 2.
    pub fn regs_ops(self: &Arc<Self>) -> Arc<GedRegsOps> {
        Arc::new(GedRegsOps(self.clone()))
    }
}

/// `ged_evt_ops`: 32 bit accesses, writes ignored.
#[derive(Debug)]
pub struct GedEvtOps(pub Arc<AcpiGed>);

impl MmioOps for GedEvtOps {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.0.evt_read(offset)))
    }

    fn write(
        &self,
        _cx: &AccessCtx,
        _offset: u64,
        _size: AccessSize,
        _value: u64,
    ) -> MemResult<()> {
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::exact(4)
    }
}

/// `ged_regs_ops`: byte accesses.
#[derive(Debug)]
pub struct GedRegsOps(pub Arc<AcpiGed>);

impl MmioOps for GedRegsOps {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.0.regs_read(offset)))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.0.regs_write(offset, value as u8);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::exact(1)
    }
}
