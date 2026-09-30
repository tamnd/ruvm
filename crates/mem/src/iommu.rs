// SPDX-License-Identifier: MIT OR Apache-2.0

//! IOMMU regions, the translate half of `IOMMUMemoryRegionClass`.
//!
//! Dispatch through an IOMMU region works: an access is translated, clamped to the translated
//! page and continued in the target address space, as `address_space_translate_iommu()` does.
//! Notifiers, replay and the IOTLB owned by the IOMMU model come with the vIOMMU work in spec/16.

use std::fmt;
use std::sync::Arc;

use crate::address_space::AddressSpace;
use crate::attrs::MemTxAttrs;

/// Access rights of a translation, `IOMMUAccessFlags`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct IommuAccessFlags(u8);

impl IommuAccessFlags {
    /// `IOMMU_NONE`.
    pub const NONE: IommuAccessFlags = IommuAccessFlags(0);
    /// `IOMMU_RO`.
    pub const RO: IommuAccessFlags = IommuAccessFlags(1);
    /// `IOMMU_WO`.
    pub const WO: IommuAccessFlags = IommuAccessFlags(2);
    /// `IOMMU_RW`.
    pub const RW: IommuAccessFlags = IommuAccessFlags(3);

    /// The flag for a read or a write, `IOMMU_ACCESS_FLAG(!is_write, is_write)`.
    pub const fn for_access(is_write: bool) -> Self {
        if is_write { Self::WO } else { Self::RO }
    }

    /// Whether every right in `other` is granted.
    pub const fn allows(self, other: IommuAccessFlags) -> bool {
        self.0 & other.0 == other.0
    }
}

/// The result of a translation, `IOMMUTLBEntry`.
#[derive(Clone)]
pub struct IommuTlbEntry {
    /// Where the translated access goes. `None` together with [`IommuAccessFlags::NONE`] is a
    /// fault.
    pub target_as: Option<Arc<AddressSpace>>,
    /// The input address, page aligned.
    pub iova: u64,
    /// The output address, page aligned.
    pub translated_addr: u64,
    /// The page offset mask, for example `0xfff` for a 4 KiB page.
    pub addr_mask: u64,
    /// What the mapping allows.
    pub perm: IommuAccessFlags,
}

impl fmt::Debug for IommuTlbEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IommuTlbEntry")
            .field("target_as", &self.target_as.as_ref().map(|a| a.name().to_string()))
            .field("iova", &self.iova)
            .field("translated_addr", &self.translated_addr)
            .field("addr_mask", &self.addr_mask)
            .field("perm", &self.perm)
            .finish()
    }
}

/// The callbacks of an IOMMU region.
pub trait IommuOps: Send + Sync {
    /// Translates `addr`, an offset into the region, for an access needing `flag`.
    fn translate(&self, addr: u64, flag: IommuAccessFlags, iommu_idx: u32) -> IommuTlbEntry;

    /// Picks the IOMMU index for an access, `attrs_to_index`. Most IOMMUs have one.
    fn attrs_to_index(&self, attrs: MemTxAttrs) -> u32 {
        let _ = attrs;
        0
    }
}
