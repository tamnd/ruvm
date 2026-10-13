// SPDX-License-Identifier: GPL-2.0-or-later

//! intel-iommu, amd-iommu, smmuv3, virtio-iommu and riscv-iommu.
//!
//! The Arm SMMUv3 and virtio-iommu are there so far: [`smmu_common`] has the page table walks
//! and the IOTLB, [`smmuv3`] the device, and [`virtio_iommu`] the paravirtual IOMMU. The plan for the rest of this crate is in
//! `spec/24-workspace-layout.md`.

pub mod smmu_common;
pub mod smmuv3;
pub mod virtio_iommu;

pub use smmu_common::Stage as SmmuStage;
pub use smmuv3::{SMMU_NUM_IRQS, SMMU_SIZE, SmmuV3};
pub use virtio_iommu::{
    GranuleMode, ReservedRegion, TYPE_VIRTIO_IOMMU, TYPE_VIRTIO_IOMMU_PCI, Viommu, VirtioIommu,
    VirtioIommuConf,
};
