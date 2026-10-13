// SPDX-License-Identifier: GPL-2.0-or-later

//! `riscv-iommu-pci`, hw/riscv/riscv-iommu-pci.c: the RISC-V IOMMU as a PCI function, its
//! registers in BAR0 with the MSI-X table and PBA over them.

use std::sync::{Arc, Weak};

use ruvm_base::Error;
use ruvm_hw_core::Clock;
use ruvm_hw_iommu::{RISCV_IOMMU_REG_SIZE, RiscvIommu, RiscvIommuConf, RiscvIommuIgs};
use ruvm_hw_pci::pcie::{PCI_EXP_TYPE_ENDPOINT, PCI_EXP_TYPE_RC_END, pcie_cap_init};
use ruvm_hw_pci::regs::{
    PCI_BASE_ADDRESS_MEM_TYPE_64, PCI_BASE_ADDRESS_SPACE_MEMORY, PCI_VENDOR_ID_REDHAT,
};
use ruvm_hw_pci::{MsixLayout, PciBus, PciDevice, PciDeviceInfo};
use ruvm_mem::AddressSpace;

/// `TYPE_RISCV_IOMMU_PCI`.
pub const TYPE_RISCV_IOMMU_PCI: &str = "riscv-iommu-pci";
/// `PCI_DEVICE_ID_REDHAT_RISCV_IOMMU`.
pub const PCI_DEVICE_ID_REDHAT_RISCV_IOMMU: u16 = 0x0014;
/// `PCI_CLASS_SYSTEM_IOMMU`. A function of this class does its DMA around the IOMMU, as
/// `pci_is_iommu()` makes it in `riscv_iommu_find_as()`.
pub const PCI_CLASS_SYSTEM_IOMMU: u16 = 0x0806;
/// `RISCV_IOMMU_PCI_MSIX_VECTORS`.
const MSIX_VECTORS: u32 = 5;
/// `RISCV_IOMMU_PCI_ICVEC_VECTORS`: CQ and FQ on vector 0 to 3, PM and PQ on 0 to 3.
const ICVEC_VECTORS: u64 = 0x3333;
/// `RISCV_IOMMU_REG_MSI_CONFIG`, where the MSI-X table starts in BAR0.
const MSIX_TABLE: u32 = 0x300;

/// The properties of `riscv-iommu-pci`: its own and those it aliases from `riscv-iommu`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RiscvIommuPciProps {
    /// `id`.
    pub id: Option<String>,
    /// `vendor-id`, also the subsystem vendor ID.
    pub vendor_id: u16,
    /// `device-id`, also the subsystem ID.
    pub device_id: u16,
    /// `revision`.
    pub revision: u8,
    /// The `riscv-iommu` properties. `riscv_iommu_pci_init()` makes `pas-bits` 56.
    pub conf: RiscvIommuConf,
}

impl Default for RiscvIommuPciProps {
    fn default() -> Self {
        RiscvIommuPciProps {
            id: None,
            vendor_id: PCI_VENDOR_ID_REDHAT,
            device_id: PCI_DEVICE_ID_REDHAT_RISCV_IOMMU,
            revision: 0x01,
            conf: RiscvIommuConf { pas_bits: 56, ..RiscvIommuConf::default() },
        }
    }
}

/// `riscv_iommu_pci_realize()` on `bus` at `devfn` or the first free slot. The IOMMU reads
/// its tables from `memory` and counts cycles on `clock`. The interrupt controllers of virt
/// all take MSIs, so the MSI-X capability is always there.
pub(crate) fn riscv_iommu_pci_realize(
    bus: &Arc<PciBus>,
    devfn: Option<u8>,
    props: &RiscvIommuPciProps,
    memory: &Arc<AddressSpace>,
    clock: &Arc<Clock>,
) -> Result<(Arc<RiscvIommu>, Arc<PciDevice>), Error> {
    let mem_err = |e: ruvm_mem::MemError| Error::generic(e.to_string());
    let info = PciDeviceInfo {
        name: TYPE_RISCV_IOMMU_PCI.to_string(),
        id: props.id.clone(),
        vendor_id: props.vendor_id,
        device_id: props.device_id,
        revision: props.revision,
        class_id: PCI_CLASS_SYSTEM_IOMMU,
        subsystem_vendor_id: props.vendor_id,
        subsystem_id: props.device_id,
        express: true,
        ..PciDeviceInfo::default()
    };
    let dev = bus.register_device(&info, devfn)?;
    let iommu =
        RiscvIommu::new(props.conf.clone(), RiscvIommuIgs::Msi, ICVEC_VECTORS, memory, clock);

    let mem = bus.memory();
    let bar0 =
        mem.new_container("riscv-iommu-bar0", RISCV_IOMMU_REG_SIZE.into()).map_err(mem_err)?;
    let regs = mem
        .new_io("riscv-iommu-regs", RISCV_IOMMU_REG_SIZE.into(), iommu.mmio_ops())
        .map_err(mem_err)?;
    mem.add_subregion(bar0, 0, regs).map_err(mem_err)?;

    // pcie_endpoint_cap_init()
    let ty = if bus.is_root() { PCI_EXP_TYPE_RC_END } else { PCI_EXP_TYPE_ENDPOINT };
    pcie_cap_init(&dev, 0, ty, 0)?;

    dev.register_bar(0, PCI_BASE_ADDRESS_SPACE_MEMORY | PCI_BASE_ADDRESS_MEM_TYPE_64, bar0);

    let layout = MsixLayout {
        table_bar: bar0,
        table_bar_nr: 0,
        table_offset: MSIX_TABLE,
        pba_bar: bar0,
        pba_bar_nr: 0,
        pba_offset: MSIX_TABLE + 256,
    };
    dev.msix_init(MSIX_VECTORS, layout, 0)?;
    // Mark all ICVEC MSI-X vectors as used.
    for i in 0..MSIX_VECTORS {
        dev.msix_vector_use(i);
    }
    let weak: Weak<PciDevice> = Arc::downgrade(&dev);
    iommu.set_notify(move |vector| {
        // riscv_iommu_pci_notify()
        if let Some(d) = weak.upgrade().filter(|d| d.msix_enabled()) {
            d.msix_notify(vector);
        }
    });
    Ok((iommu, dev))
}
