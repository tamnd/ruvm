// SPDX-License-Identifier: GPL-2.0-or-later

//! The generic PCI Express host bridge, `gpex-pcihost`, and its root function `gpex-root`
//! (1b36:0008), from hw/pci-host/gpex.c.
//!
//! The host has three regions a board maps where it wants them:
//!
//! - the ECAM window ([`GpexHost::ecam`]), 256 MiB of config space for buses 0 to 255;
//! - the MMIO window ([`GpexHost::mmio_window`]), the whole 64 bit PCI memory space. A board
//!   aliases the part it wants the guest to see, usually a window below 4 GiB at the same
//!   address and one above RAM;
//! - the I/O port window ([`GpexHost::ioport_window`]), 64 KiB of PCI I/O space.
//!
//! With `allow-unmapped-accesses` on (the default) the two windows are I/O regions that read
//! as all ones and ignore writes where no BAR is mapped, with the PCI address spaces mapped
//! over them. Off, the windows are the bare PCI address spaces.
//!
//! INTx goes to `num-irqs` output lines with the standard swizzle, `(slot + pin) % num_irqs`.
//! [`GpexHost::set_irq_num`] records which interrupt controller input each line ends up on,
//! as `gpex_set_irq_num()` does for ACPI and VFIO.
//!
//! Not ported: VMState and QOM registration, `pci_bus_set_route_irq_fn()` (the routing is
//! kept and can be read with [`GpexHost::route_intx_pin_to_irq`]) and the ACPI DSDT builder of
//! hw/pci-host/gpex-acpi.c.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

use ruvm_base::Error;
use ruvm_hw_core::IrqPin;
use ruvm_mem::{
    AccessConstraints, AccessCtx, AccessSize, MemResult, MemorySystem, MmioOps, RegionId,
};

use crate::bus::PciBus;
use crate::device::{PciDevice, PciDeviceInfo};
use crate::pcie_host::{PCIE_MMCFG_SIZE_MAX, PcieHost};
use crate::regs::*;

/// `TYPE_GPEX_HOST`.
pub const TYPE_GPEX_HOST: &str = "gpex-pcihost";
/// `TYPE_GPEX_ROOT_DEVICE`.
pub const TYPE_GPEX_ROOT_DEVICE: &str = "gpex-root";
/// `PCI_DEVICE_ID_REDHAT_PCIE_HOST`.
pub const PCI_DEVICE_ID_REDHAT_PCIE_HOST: u16 = 0x0008;
/// The size of the I/O port window, 64 KiB.
pub const GPEX_IOPORT_SIZE: u64 = 64 * 1024;

/// One window of the host, a `MemMapEntry` of `GPEXConfig`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpexWindow {
    pub base: u64,
    pub size: u64,
}

/// The properties of `gpex-pcihost`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpexConfig {
    /// `allow-unmapped-accesses`: unmapped parts of the windows read as all ones and ignore
    /// writes instead of failing the access.
    pub allow_unmapped_accesses: bool,
    /// `ecam-base` and `ecam-size`, used by the board and ACPI only.
    pub ecam: GpexWindow,
    /// `pio-base` and `pio-size`.
    pub pio: GpexWindow,
    /// `below-4g-mmio-base` and `below-4g-mmio-size`.
    pub mmio32: GpexWindow,
    /// `above-4g-mmio-base` and `above-4g-mmio-size`.
    pub mmio64: GpexWindow,
    /// `num-irqs`, the number of INTx output lines.
    pub num_irqs: u8,
}

impl Default for GpexConfig {
    fn default() -> GpexConfig {
        GpexConfig {
            allow_unmapped_accesses: true,
            ecam: GpexWindow::default(),
            pio: GpexWindow::default(),
            mmio32: GpexWindow::default(),
            mmio64: GpexWindow::default(),
            num_irqs: PCI_NUM_PINS as u8,
        }
    }
}

/// `GPEXIrq`: an output line and the interrupt controller input it is wired to.
#[derive(Debug)]
struct GpexIrq {
    irq: IrqPin,
    irq_num: AtomicI32,
}

/// `unassigned_io_ops`: reads as all ones, writes are ignored.
#[derive(Debug)]
struct UnassignedIo;

impl MmioOps for UnassignedIo {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::MAX)
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
        AccessConstraints::any_size(1, 4)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4)
    }
}

/// `GPEXHost` with its root function, realized.
pub struct GpexHost {
    config: GpexConfig,
    bus: Arc<PciBus>,
    pcie: Arc<PcieHost>,
    io_mmio: RegionId,
    io_ioport: RegionId,
    mmio_window: RegionId,
    ioport_window: RegionId,
    irqs: Arc<Vec<GpexIrq>>,
    root: Arc<PciDevice>,
}

impl fmt::Debug for GpexHost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpexHost")
            .field("config", &self.config)
            .field("pcie", &self.pcie)
            .finish_non_exhaustive()
    }
}

impl GpexHost {
    /// `gpex_host_initfn()` and `gpex_host_realize()`: the ECAM window, the PCI address
    /// spaces and their windows, the root bus "pcie.0" and the root function at 00:00.0.
    /// `system_memory` is where the ECAM window would be mapped by
    /// [`PcieHost::mmcfg_map`]; boards usually alias [`GpexHost::ecam`] themselves instead.
    pub fn new(
        memory: Arc<MemorySystem>,
        system_memory: RegionId,
        config: GpexConfig,
    ) -> Result<GpexHost, Error> {
        let err = |e: ruvm_mem::MemError| Error::generic(format!("{TYPE_GPEX_HOST}: {e}"));
        let io_mmio = memory.new_container("gpex_mmio", 1 << 64).map_err(err)?;
        let io_ioport =
            memory.new_container("gpex_ioport", u128::from(GPEX_IOPORT_SIZE)).map_err(err)?;
        let (mmio_window, ioport_window) = if config.allow_unmapped_accesses {
            let mw = memory.new_io("gpex_mmio_window", 1 << 64, Arc::new(UnassignedIo));
            let mw = mw.map_err(err)?;
            let size = u128::from(GPEX_IOPORT_SIZE);
            let iw = memory.new_io("gpex_ioport_window", size, Arc::new(UnassignedIo));
            let iw = iw.map_err(err)?;
            memory.add_subregion(mw, 0, io_mmio).map_err(err)?;
            memory.add_subregion(iw, 0, io_ioport).map_err(err)?;
            (mw, iw)
        } else {
            (io_mmio, io_ioport)
        };

        let irqs: Vec<GpexIrq> = (0..config.num_irqs)
            .map(|_| GpexIrq { irq: IrqPin::new(), irq_num: AtomicI32::new(-1) })
            .collect();
        let irqs = Arc::new(irqs);

        // pci_register_root_bus() with TYPE_PCIE_BUS.
        let bus = PciBus::new_root("pcie.0", Arc::clone(&memory), io_mmio, io_ioport, 0);
        bus.set_extended_config_space(true);
        // gpex_host_root_bus_path().
        bus.set_root_bus_path("0000:00");
        let set_irq = {
            let irqs = Arc::clone(&irqs);
            Arc::new(move |irq_num: i32, level: i32| {
                // gpex_set_irq()
                if let Some(i) = usize::try_from(irq_num).ok().and_then(|n| irqs.get(n)) {
                    i.irq.set(level);
                }
            })
        };
        bus.set_irqs(set_irq, usize::from(config.num_irqs));
        let nirq = i32::from(config.num_irqs);
        // gpex_swizzle_map_irq_fn()
        bus.set_map_irq(Arc::new(move |devfn: u8, pin: i32| {
            (i32::from(pci_slot(devfn)) + pin) % nirq.max(1)
        }));

        let pcie =
            PcieHost::new(Arc::clone(&memory), system_memory, Arc::clone(&bus)).map_err(err)?;
        pcie.mmcfg_init(PCIE_MMCFG_SIZE_MAX);

        // gpex_root_class_init(): a conventional PCI function, single function, at 00.0.
        let info = PciDeviceInfo {
            name: TYPE_GPEX_ROOT_DEVICE.to_string(),
            vendor_id: PCI_VENDOR_ID_REDHAT,
            device_id: PCI_DEVICE_ID_REDHAT_PCIE_HOST,
            revision: 0,
            class_id: PCI_CLASS_BRIDGE_HOST,
            ..PciDeviceInfo::default()
        };
        let root = bus.register_device(&info, Some(pci_devfn(0, 0)))?;

        Ok(GpexHost {
            config,
            bus,
            pcie,
            io_mmio,
            io_ioport,
            mmio_window,
            ioport_window,
            irqs,
            root,
        })
    }

    /// The properties the host was made with.
    pub fn config(&self) -> &GpexConfig {
        &self.config
    }

    /// The root bus, "pcie.0".
    pub fn bus(&self) -> &Arc<PciBus> {
        &self.bus
    }

    /// The root function, `gpex-root` at 00:00.0.
    pub fn root(&self) -> &Arc<PciDevice> {
        &self.root
    }

    /// The ECAM part, `PCIExpressHost`.
    pub fn pcie_host(&self) -> &Arc<PcieHost> {
        &self.pcie
    }

    /// Sysbus MMIO region 0, the 256 MiB ECAM window.
    pub fn ecam(&self) -> RegionId {
        self.pcie.mmio()
    }

    /// Sysbus MMIO region 1, the PCI memory space window.
    pub fn mmio_window(&self) -> RegionId {
        self.mmio_window
    }

    /// Sysbus MMIO region 2, the PCI I/O space window.
    pub fn ioport_window(&self) -> RegionId {
        self.ioport_window
    }

    /// The PCI memory space BARs map into, `io_mmio`.
    pub fn pci_memory(&self) -> RegionId {
        self.io_mmio
    }

    /// The PCI I/O space BARs map into, `io_ioport`.
    pub fn pci_io(&self) -> RegionId {
        self.io_ioport
    }

    /// INTx output line `index`, `sysbus_connect_irq()`.
    pub fn irq(&self, index: usize) -> Option<&IrqPin> {
        self.irqs.get(index).map(|i| &i.irq)
    }

    /// `gpex_set_irq_num()`: records that line `index` is interrupt `gsi` of the interrupt
    /// controller.
    pub fn set_irq_num(&self, index: usize, gsi: i32) -> Result<(), Error> {
        let Some(i) = self.irqs.get(index) else {
            return Err(Error::generic(format!("{TYPE_GPEX_HOST}: no INTx line {index}")));
        };
        i.irq_num.store(gsi, Ordering::Relaxed);
        Ok(())
    }

    /// `gpex_route_intx_pin_to_irq()`: the interrupt `pin` is routed to, or `None` while it
    /// is not wired.
    pub fn route_intx_pin_to_irq(&self, pin: usize) -> Option<i32> {
        let gsi = self.irqs.get(pin)?.irq_num.load(Ordering::Relaxed);
        (gsi >= 0).then_some(gsi)
    }

    /// The device reset of the host and everything on its bus.
    pub fn reset(&self) {
        self.bus.reset();
    }
}
