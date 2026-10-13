// SPDX-License-Identifier: GPL-2.0-or-later

//! intel-iommu, amd-iommu, smmuv3, virtio-iommu and riscv-iommu.
//!
//! The Arm SMMUv3, virtio-iommu, intel-iommu and amd-iommu are there so far: [`smmu_common`]
//! has the SMMU page table walks and the IOTLB, [`smmuv3`] the SMMU device, [`virtio_iommu`] the
//! paravirtual IOMMU, [`intel_iommu`] the VT-d IOMMU and [`amd_iommu`] the AMD-Vi IOMMU. The
//! plan for the rest of this crate is in `spec/24-workspace-layout.md`.

pub mod amd_iommu;
pub mod intel_iommu;
pub mod smmu_common;
pub mod smmuv3;
pub mod virtio_iommu;

pub use amd_iommu::{
    AMDVI_BASE_ADDR, AMDVI_MMIO_SIZE, AmdIommu, AmdIommuConf, AmdviPci, TYPE_AMD_IOMMU_DEVICE,
    TYPE_AMD_IOMMU_MEMORY_REGION, TYPE_AMD_IOMMU_PCI,
};
pub use intel_iommu::{
    DMAR_REG_BASE, DMAR_REG_SIZE, IntelIommu, IntelIommuConf, SID_INVALID, TYPE_INTEL_IOMMU_DEVICE,
    TYPE_INTEL_IOMMU_MEMORY_REGION,
};
pub use smmu_common::Stage as SmmuStage;
pub use smmuv3::{SMMU_NUM_IRQS, SMMU_SIZE, SmmuV3, SmmuV3Props};
pub use virtio_iommu::{
    GranuleMode, ReservedRegion, TYPE_VIRTIO_IOMMU, TYPE_VIRTIO_IOMMU_PCI, Viommu, VirtioIommu,
    VirtioIommuConf,
};
