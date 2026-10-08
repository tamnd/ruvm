// SPDX-License-Identifier: GPL-2.0-or-later

//! intel-iommu, amd-iommu, smmuv3, virtio-iommu and riscv-iommu.
//!
//! Only the Arm SMMUv3 is there so far: [`smmu_common`] has the page table walks and the
//! IOTLB, and [`smmuv3`] the device. The plan for the rest of this crate is in
//! `spec/24-workspace-layout.md`.

pub mod smmu_common;
pub mod smmuv3;

pub use smmu_common::Stage as SmmuStage;
pub use smmuv3::{SMMU_NUM_IRQS, SMMU_SIZE, SmmuV3};
