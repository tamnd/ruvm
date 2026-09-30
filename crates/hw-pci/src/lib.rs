// SPDX-License-Identifier: GPL-2.0-or-later

//! The PCI core: config space, BARs, buses, INTx routing, host bridge config access, basic
//! PCI-to-PCI bridges, MSI and MSI-X. Ported from hw/pci/pci.c, pci_host.c, pci_bridge.c,
//! msi.c and msix.c in QEMU.
//!
//! A board creates a root [`PciBus`] over its memory and I/O containers, wires INTx with
//! [`PciBus::set_irqs`] and [`PciBus::set_map_irq`], and exposes config space through a
//! [`PciHostState`] (mechanism #1 at 0xcf8 and 0xcfc). Device models register functions with
//! [`PciBus::register_device`], describe their BARs with [`PciDevice::register_bar`] and hook
//! config accesses through [`PciDeviceOps`].
//!
//! Differences from QEMU:
//!
//! - MSI and MSI-X messages go to a callback (per device or per root bus) instead of a store
//!   into the bus master address space. They are dropped while bus mastering is off, which is
//!   what the disabled bus master region does in QEMU.
//! - `msi_nonbroken` belongs to the root bus instead of being a global.
//! - `pci_add_capability()` returns an error when no free space is found instead of asserting.
//! - Capability list walks are bounded so a broken list cannot hang.
//! - Config accesses from the host are also capped at the function's own config space size.
//! - A bridge updates its windows after reset.
//!
//! Not ported: VMState, trace points, QOM registration and properties, hotplug, AER, SR-IOV,
//! ATS, PCIe capabilities and extended capabilities, IOMMU and bus master address spaces, VGA
//! registration and bridge VGA windows, option ROM files, `pci_route_intx_to_irq()`, the MSI-X
//! vector notifiers and the Xen paths.

#![forbid(unsafe_code)]

mod bridge;
mod bus;
mod device;
mod host;
mod msi;
mod msix;
pub mod regs;

pub use bridge::{PciBridge, PciBridgeWindow, pci_bridge_get_base, pci_bridge_get_limit};
pub use bus::{PciBus, PciMapIrqFn, PciSetIrqFn, pci_swizzle_map_irq_fn};
pub use device::{
    MsiMessage, MsiTrigger, PciBarInfo, PciConfigMut, PciDevice, PciDeviceInfo, PciDeviceOps,
};
pub use host::{
    PCI_HOST_CONFIG_ADDR_PORT, PCI_HOST_CONFIG_DATA_PORT, PciHostState, pci_data_read,
    pci_data_write, pci_host_config_read_common, pci_host_config_write_common,
};
pub use msi::PCI_MSI_VECTORS_MAX;
pub use msix::MsixLayout;
