// SPDX-License-Identifier: GPL-2.0-or-later

//! The `intel-iommu` device, from hw/i386/intel_iommu.c, hw/i386/intel_iommu_internal.h and
//! hw/i386/x86-iommu.c.
//!
//! The IOMMU translates the DMA of the PCI devices of the q35 root bus. Each device gets an
//! IOMMU region from [`IntelIommu::device_ops`], keyed by its source ID, and does its DMA
//! through an address space over that region. The root and context tables, the page tables
//! and the invalidation queue are read from the system address space given at creation.
//!
//! What is there: legacy mode translation with second stage page tables, pass-through and
//! device-IOTLB context entries, the fault recording register with its fault event interrupt,
//! the context cache and the IOTLB with register based and queued invalidation, and the
//! invalidation wait descriptor with its completion interrupt. Scalable mode reads the PASID
//! directory and table for PASID 0 and walks second stage tables, or first stage tables with
//! `fsts=on`, setting their accessed and dirty bits, and has the PASID based IOTLB
//! invalidation. Interrupt remapping takes the MSIs of the PCI devices and the IOAPIC through
//! [`IntelIommu::remap_msi`], with the remapping table, its source ID checks and its faults.
//!
//! # Differences from QEMU
//!
//! - Only requests without a PASID are translated. QEMU also gives PASID capable devices an
//!   address space per PASID, and with `svm=on` takes ATS translation requests and page
//!   requests into its page request queue. ruvm has the page request registers but never
//!   queues a request.
//! - There are no IOMMU notifiers and no replay, so nothing like vhost or VFIO can sit behind
//!   the IOMMU, and `caching-mode` only sets the capability bit.
//! - DMA to the interrupt range, 0xfee00000 to 0xfeefffff, is not translated. QEMU puts its
//!   `vtd-ir` region over the IOMMU region there; ruvm delivers MSIs without a DMA write, and
//!   the board hands them to [`IntelIommu::remap_msi`] instead. A plain DMA write there does
//!   not raise an interrupt.
//! - The interrupt entry cache invalidation is a no-op, since there is no cache and no KVM
//!   routes to update.
//! - There is no VMState.
//! - Guest errors that QEMU logs with `error_report_once()` are silent. The invalidation
//!   descriptor checks that use `error_report()` print the same lines.

use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

use ruvm_base::report::error_report;
use ruvm_mem::{
    AccessConstraints, AccessCtx, AccessSize, AddressSpace, IommuAccessFlags, IommuOps,
    IommuTlbEntry, MemResult, MemTxAttrs, MemTxResult, MmioOps,
};

/// `TYPE_INTEL_IOMMU_DEVICE`.
pub const TYPE_INTEL_IOMMU_DEVICE: &str = "intel-iommu";
/// `TYPE_INTEL_IOMMU_MEMORY_REGION`, the stem of the per-device region names.
pub const TYPE_INTEL_IOMMU_MEMORY_REGION: &str = "intel-iommu-iommu-memory-region";
/// `Q35_HOST_BRIDGE_IOMMU_ADDR`, where the register frame is mapped.
pub const DMAR_REG_BASE: u64 = 0xfed9_0000;
/// `DMAR_REG_SIZE`.
pub const DMAR_REG_SIZE: u64 = 0x230;

const REG_SIZE: usize = DMAR_REG_SIZE as usize;

const DMAR_VER_REG: u64 = 0x0;
const DMAR_CAP_REG: u64 = 0x8;
const DMAR_ECAP_REG: u64 = 0x10;
const DMAR_GCMD_REG: u64 = 0x18;
const DMAR_GSTS_REG: u64 = 0x1c;
const DMAR_RTADDR_REG: u64 = 0x20;
const DMAR_RTADDR_REG_HI: u64 = 0x24;
const DMAR_CCMD_REG: u64 = 0x28;
const DMAR_CCMD_REG_HI: u64 = 0x2c;
const DMAR_FSTS_REG: u64 = 0x34;
const DMAR_FECTL_REG: u64 = 0x38;
const DMAR_FEDATA_REG: u64 = 0x3c;
const DMAR_FEADDR_REG: u64 = 0x40;
const DMAR_FEUADDR_REG: u64 = 0x44;
const DMAR_PMEN_REG: u64 = 0x64;
const DMAR_IQH_REG: u64 = 0x80;
const DMAR_IQT_REG: u64 = 0x88;
const DMAR_IQT_REG_HI: u64 = 0x8c;
const DMAR_IQA_REG: u64 = 0x90;
const DMAR_IQA_REG_HI: u64 = 0x94;
const DMAR_ICS_REG: u64 = 0x9c;
const DMAR_IECTL_REG: u64 = 0xa0;
const DMAR_IEDATA_REG: u64 = 0xa4;
const DMAR_IEADDR_REG: u64 = 0xa8;
const DMAR_IEUADDR_REG: u64 = 0xac;
const DMAR_IRTA_REG: u64 = 0xb8;
const DMAR_IRTA_REG_HI: u64 = 0xbc;
const DMAR_PQH_REG: u64 = 0xc0;
const DMAR_PQT_REG: u64 = 0xc8;
const DMAR_PQA_REG: u64 = 0xd0;
const DMAR_PRS_REG: u64 = 0xdc;
const DMAR_PECTL_REG: u64 = 0xe0;
const DMAR_PEDATA_REG: u64 = 0xe4;
const DMAR_PEADDR_REG: u64 = 0xe8;
const DMAR_PEUADDR_REG: u64 = 0xec;
const DMAR_IVA_REG: u64 = 0xf0;
const DMAR_IVA_REG_HI: u64 = 0xf4;
const DMAR_IOTLB_REG: u64 = 0xf8;
const DMAR_IOTLB_REG_HI: u64 = 0xfc;
const DMAR_FRCD_REG_OFFSET: u64 = 0x220;
const DMAR_FRCD_REG_0_0: u64 = 0x220;
const DMAR_FRCD_REG_0_1: u64 = 0x224;
const DMAR_FRCD_REG_0_2: u64 = 0x228;
const DMAR_FRCD_REG_0_3: u64 = 0x22c;
/// `DMAR_FRCD_REG_NR`.
const FRCD_REG_NR: u16 = 1;

const GCMD_TE: u32 = 1 << 31;
const GCMD_SRTP: u32 = 1 << 30;
const GCMD_QIE: u32 = 1 << 26;
const GCMD_IRE: u32 = 1 << 25;
const GCMD_SIRTP: u32 = 1 << 24;
const GSTS_TES: u32 = 1 << 31;
const GSTS_RTPS: u32 = 1 << 30;
const GSTS_QIES: u32 = 1 << 26;
const GSTS_IRES: u32 = 1 << 25;
const GSTS_IRTPS: u32 = 1 << 24;

const CCMD_ICC: u64 = 1 << 63;
const CCMD_GLOBAL_INVL: u64 = 1 << 61;
const CCMD_DOMAIN_INVL: u64 = 2 << 61;
const CCMD_DEVICE_INVL: u64 = 3 << 61;
const CCMD_CIRG_MASK: u64 = 3 << 61;
const CCMD_GLOBAL_INVL_A: u64 = 1 << 59;
const CCMD_DEVICE_INVL_A: u64 = 3 << 59;
const CCMD_CAIG_MASK: u64 = 3 << 59;

const TLB_GLOBAL_FLUSH: u64 = 1 << 60;
const TLB_DSI_FLUSH: u64 = 2 << 60;
const TLB_PSI_FLUSH: u64 = 3 << 60;
const TLB_FLUSH_GRANU_MASK: u64 = 3 << 60;
const TLB_GLOBAL_FLUSH_A: u64 = 1 << 57;
const TLB_DSI_FLUSH_A: u64 = 2 << 57;
const TLB_PSI_FLUSH_A: u64 = 3 << 57;
const TLB_FLUSH_GRANU_MASK_A: u64 = 3 << 57;
const TLB_IVT: u64 = 1 << 63;

const IQA_QS: u64 = 0x7;
const IQA_DW_MASK: u64 = 0x800;
const IQH_QH_MASK: u64 = 0x7fff0;
const IQT_QT_256_RSV_BIT: u64 = 0x10;
const ICS_IWC: u32 = 1;
const IECTL_IM: u32 = 1 << 31;
const IECTL_IP: u32 = 1 << 30;
const PR_STATUS_PPR: u32 = 1;
const PR_STATUS_PRO: u32 = 1 << 1;
const PR_PECTL_IP: u32 = 1 << 30;
const PR_PECTL_IM: u32 = 1 << 31;
const FSTS_PFO: u32 = 1;
const FSTS_PPF: u32 = 1 << 1;
const FSTS_IQE: u32 = 1 << 4;
const FSTS_FRI_MASK: u32 = 0xff00;
const FECTL_IM: u32 = 1 << 31;
const FECTL_IP: u32 = 1 << 30;
const FRCD_F: u64 = 1 << 63;
const FRCD_T: u64 = 1 << 62;
const FRCD_SID_MASK: u64 = 0xffff;
const RTADDR_SMT: u64 = 1 << 10;
const IRTA_SIZE_MASK: u64 = 0xf;
const IRTA_EIME: u64 = 1 << 11;

const CAP_FRO: u64 = 0x220 << 20;
const CAP_ND: u64 = 6;
const CAP_MAMV: u64 = 18 << 48;
const CAP_PSI: u64 = 1 << 39;
const CAP_SSLPS: u64 = (1 << 34) | (1 << 35);
const CAP_DRAIN: u64 = (1 << 54) | (1 << 55);
const CAP_ESRTPS: u64 = 1 << 63;
const CAP_CM: u64 = 1 << 7;
const CAP_SAGAW_SHIFT: u64 = 8;
const CAP_SAGAW_MASK: u64 = 0x1f << CAP_SAGAW_SHIFT;
const CAP_SAGAW_39BIT: u64 = 0x2 << CAP_SAGAW_SHIFT;
const CAP_SAGAW_48BIT: u64 = 0x4 << CAP_SAGAW_SHIFT;
const CAP_FS1GP: u64 = 1 << 56;
const ECAP_QI: u64 = 1 << 1;
const ECAP_DT: u64 = 1 << 2;
const ECAP_IR: u64 = 1 << 3;
const ECAP_EIM: u64 = 1 << 4;
const ECAP_PT: u64 = 1 << 6;
const ECAP_SC: u64 = 1 << 7;
const ECAP_IRO: u64 = 0xf0 << 4;
const ECAP_MHMV: u64 = 15 << 20;
const ECAP_SRS: u64 = 1 << 31;
const ECAP_PASID: u64 = 1 << 40;
const ECAP_SMTS: u64 = 1 << 43;
const ECAP_SSTS: u64 = 1 << 46;
const ECAP_FSTS: u64 = 1 << 47;
const ECAP_PDS: u64 = 1 << 42;
const ECAP_PRS: u64 = 1 << 29;
const ECAP_NWFS: u64 = 1 << 33;

/// `VTD_MAMV`, the largest address mask of a page selective invalidation.
const MAMV: u64 = 18;
/// `VTD_IOTLB_MAX_SIZE`.
const IOTLB_MAX_SIZE: usize = 1024;
/// `VTD_CONTEXT_CACHE_GEN_MAX`.
const CONTEXT_CACHE_GEN_MAX: u32 = 0xffff_ffff;
const PAGE_MASK_4K: u64 = !0xfff;
const SS_IGN_COM: u64 = 0xbff0_0000_0000_0000;
const SS_TM: u64 = 1 << 62;
const SPTE_SNP: u64 = 0x800;
const PT_PAGE_SIZE_MASK: u64 = 1 << 7;
const INTERRUPT_ADDR_FIRST: u64 = 0xfee0_0000;
const INTERRUPT_ADDR_LAST: u64 = 0xfeef_ffff;

const ROOT_ENTRY_P: u64 = 1;
const CONTEXT_ENTRY_P: u64 = 1;
const CONTEXT_ENTRY_FPD: u64 = 1 << 1;
const CONTEXT_ENTRY_TT: u64 = 3 << 2;
const CONTEXT_TT_MULTI_LEVEL: u64 = 0;
const CONTEXT_TT_DEV_IOTLB: u64 = 1 << 2;
const CONTEXT_TT_PASS_THROUGH: u64 = 2 << 2;
const CONTEXT_ENTRY_RSVD_HI: u64 = 0xffff_ffff_ff00_0080;

const PASID_DIR_P: u64 = 1;
const PASID_DIR_FPD: u64 = 1 << 1;
const PASID_ENTRY_P: u64 = 1;
const PASID_ENTRY_FPD: u64 = 1 << 1;
/// The PASID granular translation types, `VTD_SM_PASID_ENTRY_FST` and the rest.
const PGTT_FST: u8 = 1;
const PGTT_SST: u8 = 2;
const PGTT_PT: u8 = 4;
const FS_P: u64 = 1;
const FS_RW: u64 = 1 << 1;
const FS_US: u64 = 1 << 2;
const FS_A: u64 = 1 << 5;
const FS_D: u64 = 1 << 6;
const FS_UPPER_IGNORED: u64 = 0xfff0_0000_0000_0000;

const INV_DESC_CC: u8 = 0x1;
const INV_DESC_IOTLB: u8 = 0x2;
const INV_DESC_DEVICE: u8 = 0x3;
const INV_DESC_IEC: u8 = 0x4;
const INV_DESC_WAIT: u8 = 0x5;
const INV_DESC_PIOTLB: u8 = 0x6;
const INV_DESC_PC: u8 = 0x7;
const INV_DESC_DEV_PIOTLB: u8 = 0x8;
const INV_DESC_PGRESP: u8 = 0x9;
/// `VTD_INV_DESC_NONE`.
const INV_DESC_NONE: u8 = 0;
const INV_DESC_ALL_ONE: u64 = u64::MAX;
const INV_DESC_WAIT_SW: u64 = 1 << 5;
const INV_DESC_WAIT_IF: u64 = 1 << 4;
const INV_DESC_WAIT_FN: u64 = 1 << 6;

const FR_ROOT_ENTRY_P: u8 = 1;
const FR_CONTEXT_ENTRY_P: u8 = 2;
const FR_CONTEXT_ENTRY_INV: u8 = 3;
const FR_ADDR_BEYOND_MGAW: u8 = 4;
const FR_WRITE: u8 = 5;
const FR_READ: u8 = 6;
const FR_PAGING_ENTRY_INV: u8 = 7;
const FR_ROOT_TABLE_INV: u8 = 8;
const FR_CONTEXT_TABLE_INV: u8 = 9;
const FR_ROOT_ENTRY_RSVD: u8 = 0xa;
const FR_CONTEXT_ENTRY_RSVD: u8 = 0xb;
const FR_PAGING_ENTRY_RSVD: u8 = 0xc;
const FR_INTERRUPT_ADDR: u8 = 0xe;
const FR_IR_REQ_RSVD: u8 = 0x20;
const FR_IR_INDEX_OVER: u8 = 0x21;
const FR_IR_ENTRY_P: u8 = 0x22;
const FR_IR_ROOT_INVAL: u8 = 0x23;
const FR_IR_IRTE_RSVD: u8 = 0x24;
const FR_IR_SID_ERR: u8 = 0x26;
const FR_PASID_DIR_ACCESS_ERR: u8 = 0x50;
const FR_PASID_DIR_ENTRY_P: u8 = 0x51;
const FR_PASID_TABLE_ACCESS_ERR: u8 = 0x58;
const FR_PASID_ENTRY_P: u8 = 0x59;
const FR_PASID_TABLE_ENTRY_INV: u8 = 0x5b;
const FR_FS_PAGING_ENTRY_INV: u8 = 0x70;
const FR_FS_PAGING_ENTRY_P: u8 = 0x71;
const FR_FS_PAGING_ENTRY_RSVD: u8 = 0x72;
const FR_PASID_ENTRY_FSPTPTR_INV: u8 = 0x73;
const FR_FS_NON_CANONICAL: u8 = 0x80;
const FR_FS_PAGING_ENTRY_US: u8 = 0x81;
const FR_SM_WRITE: u8 = 0x85;
const FR_SM_INTERRUPT_ADDR: u8 = 0x87;
const FR_FS_BIT_UPDATE_FAILED: u8 = 0x91;

/// `vtd_qualified_faults[]`: the faults that the FPD bit suppresses.
fn qualified_fault(fault: u8) -> bool {
    matches!(
        fault,
        0x2..=0x7
            | 0xc..=0xe
            | 0x47
            | 0x51
            | 0x59
            | 0x5b
            | 0x70..=0x73
            | 0x80
            | 0x81
            | 0x85
            | 0x87
            | 0x91
    )
}

/// `VTD_HAW_MASK()`.
fn haw_mask(aw: u8) -> u64 {
    (1u64 << aw) - 1
}

/// `vtd_pt_level_shift()`.
fn level_shift(level: u32) -> u32 {
    12 + (level - 1) * 9
}

/// `vtd_pt_level_page_mask()`.
fn level_page_mask(level: u32) -> u64 {
    !((1u64 << level_shift(level)) - 1)
}

/// The `OnOffAuto` QAPI type, for the `intremap` and `eim` properties.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OnOffAuto {
    #[default]
    Auto,
    On,
    Off,
}

impl OnOffAuto {
    /// The value of the string `v`, as the property setter takes it.
    pub fn parse(v: &str) -> Option<Self> {
        match v {
            "auto" => Some(Self::Auto),
            "on" => Some(Self::On),
            "off" => Some(Self::Off),
            _ => None,
        }
    }
}

/// The properties of `intel-iommu`, with those of its `x86-iommu` parent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntelIommuConf {
    /// `intremap`.
    pub intremap: OnOffAuto,
    /// `device-iotlb`.
    pub device_iotlb: bool,
    /// `dma-translation`.
    pub dma_translation: bool,
    /// `version`, which QEMU does not use either.
    pub version: u32,
    /// `eim`.
    pub eim: OnOffAuto,
    /// `aw-bits`.
    pub aw_bits: u8,
    /// `caching-mode`.
    pub caching_mode: bool,
    /// `scalable-mode`.
    pub scalable_mode: bool,
    /// `fsts`.
    pub fsts: bool,
    /// `snoop-control`.
    pub snoop_control: bool,
    /// `pasid-bits`.
    pub pasid_bits: u8,
    /// `svm`.
    pub svm: bool,
    /// `stale-tm`.
    pub stale_tm: bool,
    /// `fs1gp`.
    pub fs1gp: bool,
}

impl Default for IntelIommuConf {
    fn default() -> Self {
        IntelIommuConf {
            intremap: OnOffAuto::Auto,
            device_iotlb: false,
            dma_translation: true,
            version: 0,
            eim: OnOffAuto::Auto,
            aw_bits: 48,
            caching_mode: false,
            scalable_mode: false,
            fsts: false,
            snoop_control: false,
            pasid_bits: 0,
            svm: false,
            stale_tm: false,
            fs1gp: true,
        }
    }
}

impl IntelIommuConf {
    /// `x86_iommu_realize()` and `vtd_decide_config()`: the configuration with `intremap` and
    /// `eim` decided, or the error QEMU gives. `irq_all_kernel` is KVM with the whole irqchip
    /// in the kernel, and `kvm_irqchip` is KVM with any part of it there.
    pub fn realize(&self, irq_all_kernel: bool, kvm_irqchip: bool) -> Result<Self, String> {
        let mut c = self.clone();
        let ir = match c.intremap {
            OnOffAuto::Auto => !irq_all_kernel,
            v => v == OnOffAuto::On,
        };
        if ir && irq_all_kernel {
            return Err("Interrupt Remapping cannot work with kernel-irqchip=on, please use \
                        'split|off'."
                .into());
        }
        if c.eim == OnOffAuto::On && !ir {
            return Err("eim=on cannot be selected without intremap=on".into());
        }
        if c.eim == OnOffAuto::Auto {
            c.eim = if kvm_irqchip && ir { OnOffAuto::On } else { OnOffAuto::Off };
        }
        if !c.scalable_mode && c.fsts {
            return Err("fsts is only available in scalable mode".into());
        }
        if !c.fsts && c.aw_bits != 39 && c.aw_bits != 48 {
            let mode = if c.scalable_mode { "Scalable mode(fsts=off)" } else { "Legacy mode" };
            return Err(format!("{mode}: supported values for aw-bits are: 39, 48"));
        }
        if c.fsts && c.aw_bits != 48 {
            return Err("Scalable mode(fsts=on): supported value for aw-bits is: 48".into());
        }
        if c.pasid_bits != 0 && !c.scalable_mode {
            return Err("Need to set scalable mode for PASID".into());
        }
        if c.pasid_bits > 20 {
            return Err(format!(
                "PASID width {} exceeds Max PASID Width 20 allowed in PCI spec",
                c.pasid_bits
            ));
        }
        if c.svm {
            if !c.device_iotlb {
                return Err("Need to set device IOTLB for svm".into());
            }
            if !c.fsts {
                return Err("Need to set fsts for svm".into());
            }
            if !c.dma_translation {
                return Err("Need to set dma-translation for svm".into());
            }
            if c.pasid_bits == 0 {
                return Err("Need to set PASID support for svm".into());
            }
        }
        c.intremap = if ir { OnOffAuto::On } else { OnOffAuto::Off };
        Ok(c)
    }

    /// `x86_iommu_ir_supported()`.
    pub fn intr_supported(&self) -> bool {
        self.intremap == OnOffAuto::On
    }
}

/// Memory accesses of the IOMMU itself, through the system address space.
struct Dma<'a>(Option<&'a AddressSpace>);

impl Dma<'_> {
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.0.is_some_and(|a| a.read(addr, MemTxAttrs::UNSPECIFIED, buf).is_ok())
    }

    fn write(&self, addr: u64, buf: &[u8]) -> bool {
        self.0.is_some_and(|a| a.write(addr, MemTxAttrs::UNSPECIFIED, buf).is_ok())
    }

    /// `N` little endian quads at `addr`.
    fn read_quads<const N: usize>(&self, addr: u64) -> Option<[u64; N]> {
        let mut b = vec![0u8; N * 8];
        if !self.read(addr, &mut b) {
            return None;
        }
        let mut out = [0u64; N];
        for (o, c) in out.iter_mut().zip(b.chunks_exact(8)) {
            let mut q = [0u8; 8];
            q.copy_from_slice(c);
            *o = u64::from_le_bytes(q);
        }
        Some(out)
    }

    /// `vtd_get_pte()`: all ones when the read fails.
    fn pte(&self, base: u64, index: u64) -> u64 {
        self.read_quads::<1>(base + index * 8).map_or(u64::MAX, |[q]| q)
    }
}

/// `X86_IOMMU_SID_INVALID`: the source ID of interrupts without one, like the IOAPIC's.
pub const SID_INVALID: u16 = 0xffff;

const IRTE_P: u64 = 1 << 0;
const IRTE_FPD: u64 = 1 << 1;
/// `__reserved_0` and `__reserved_1` of `VTD_IR_TableEntry`.
const IRTE_RSVD_LO: u64 = 0xff00_7000;
/// `__reserved_2`.
const IRTE_RSVD_HI: u64 = !0xf_ffff;
const MSI_ADDR_IR_SHV: u32 = 1 << 3;
const MSI_ADDR_IR_INT_MODE: u32 = 1 << 4;

/// The source ID check of `vtd_irte_get()` on the high quad of an IRTE.
fn irte_sid_ok(hi: u64, sid: u16) -> bool {
    const SVT_MASK: [u16; 4] = [0xffff, 0xfffb, 0xfff9, 0xfff8];
    let source_id = hi as u16;
    match (hi >> 18) & 3 {
        0 => true,
        1 => {
            let mask = SVT_MASK[((hi >> 16) & 3) as usize];
            source_id & mask == sid & mask
        }
        2 => {
            let bus = (sid >> 8) as u8;
            (source_id as u8..=(source_id >> 8) as u8).contains(&bus)
        }
        _ => false,
    }
}

/// `VTDContextEntry`: lo, hi, and two more quads in scalable mode.
type ContextEntry = [u64; 4];

/// `struct vtd_iotlb_key`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct IotlbKey {
    gfn: u64,
    sid: u16,
    level: u32,
    pasid: u32,
}

/// `VTDIOTLBEntry`.
#[derive(Clone, Copy, Debug)]
struct IotlbEntry {
    gfn: u64,
    domain_id: u16,
    pte: u64,
    access_flags: u8,
    mask: u64,
    /// The PGTT the entry came from, second stage for legacy mode.
    pgtt: u8,
}

/// `IOMMU_RO` and `IOMMU_WO` as bits.
const ACCESS_R: u8 = 1;
const ACCESS_W: u8 = 2;

/// One translation, `IOMMUTLBEntry` without the address space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Xlate {
    iova: u64,
    translated: u64,
    mask: u64,
    perm: u8,
}

/// How a walk ended.
enum Walked {
    /// Pass-through: the address untranslated, 4K at a time.
    PassThrough,
    /// The leaf entry, its level, the access it allows, the domain and the PGTT.
    Page { pte: u64, level: u32, flags: u8, did: u16, pgtt: u8 },
    /// A fault, recorded already, with the mask of the entry to give.
    Fault { mask: u64 },
}

/// The state of `IntelIOMMUState` that the lock covers.
struct State {
    conf: IntelIommuConf,
    csr: [u8; REG_SIZE],
    wmask: [u8; REG_SIZE],
    w1cmask: [u8; REG_SIZE],
    womask: [u8; REG_SIZE],
    cap: u64,
    ecap: u64,
    root: u64,
    root_scalable: bool,
    dmar_enabled: bool,
    intr_enabled: bool,
    intr_size: u32,
    intr_root: u64,
    intr_eime: bool,
    iq: u64,
    iq_head: u16,
    iq_tail: u16,
    iq_size: u32,
    qi_enabled: bool,
    iq_last_desc_type: u8,
    iq_dw: bool,
    next_frcd_reg: u16,
    iotlb: HashMap<IotlbKey, IotlbEntry>,
    context_cache_gen: u32,
    /// `VTDAddressSpace.context_cache_entry` of each source ID: the generation and the entry.
    context_cache: HashMap<u16, (u32, ContextEntry)>,
    spte_rsvd: [u64; 5],
    spte_rsvd_large: [u64; 5],
    fpte_rsvd: [u64; 5],
    fpte_rsvd_large: [u64; 5],
    /// MSIs to send once the lock is dropped, as (address, data).
    msis: Vec<(u64, u32)>,
}

macro_rules! reg_access {
    ($get:ident, $set:ident, $ty:ty, $n:expr) => {
        fn $get(a: &[u8; REG_SIZE], off: u64) -> $ty {
            let off = off as usize;
            let mut b = [0u8; $n];
            b.copy_from_slice(&a[off..off + $n]);
            <$ty>::from_le_bytes(b)
        }

        fn $set(a: &mut [u8; REG_SIZE], off: u64, v: $ty) {
            let off = off as usize;
            a[off..off + $n].copy_from_slice(&v.to_le_bytes());
        }
    };
}

reg_access!(ld_long, st_long, u32, 4);
reg_access!(ld_quad, st_quad, u64, 8);

impl State {
    fn new(conf: IntelIommuConf) -> Self {
        let mut s = State {
            conf,
            csr: [0; REG_SIZE],
            wmask: [0; REG_SIZE],
            w1cmask: [0; REG_SIZE],
            womask: [0; REG_SIZE],
            cap: 0,
            ecap: 0,
            root: 0,
            root_scalable: false,
            dmar_enabled: false,
            intr_enabled: false,
            intr_size: 0,
            intr_root: 0,
            intr_eime: false,
            iq: 0,
            iq_head: 0,
            iq_tail: 0,
            iq_size: 0,
            qi_enabled: false,
            iq_last_desc_type: INV_DESC_NONE,
            iq_dw: false,
            next_frcd_reg: 0,
            iotlb: HashMap::new(),
            context_cache_gen: 1,
            context_cache: HashMap::new(),
            spte_rsvd: [0; 5],
            spte_rsvd_large: [0; 5],
            fpte_rsvd: [0; 5],
            fpte_rsvd_large: [0; 5],
            msis: Vec::new(),
        };
        s.init();
        s
    }

    fn define_long(&mut self, addr: u64, val: u32, wmask: u32, w1cmask: u32) {
        st_long(&mut self.csr, addr, val);
        st_long(&mut self.wmask, addr, wmask);
        st_long(&mut self.w1cmask, addr, w1cmask);
    }

    fn define_quad(&mut self, addr: u64, val: u64, wmask: u64, w1cmask: u64) {
        st_quad(&mut self.csr, addr, val);
        st_quad(&mut self.wmask, addr, wmask);
        st_quad(&mut self.w1cmask, addr, w1cmask);
    }

    fn get_long_raw(&self, addr: u64) -> u32 {
        ld_long(&self.csr, addr)
    }

    fn get_quad_raw(&self, addr: u64) -> u64 {
        ld_quad(&self.csr, addr)
    }

    fn set_long_raw(&mut self, addr: u64, val: u32) {
        st_long(&mut self.csr, addr, val);
    }

    fn set_quad_raw(&mut self, addr: u64, val: u64) {
        st_quad(&mut self.csr, addr, val);
    }

    /// `vtd_get_long()`: the register without its write-only bits.
    fn get_long(&self, addr: u64) -> u32 {
        ld_long(&self.csr, addr) & !ld_long(&self.womask, addr)
    }

    fn get_quad(&self, addr: u64) -> u64 {
        ld_quad(&self.csr, addr) & !ld_quad(&self.womask, addr)
    }

    /// `vtd_set_long()`: a guest write, through the write and write-1-to-clear masks.
    fn set_long(&mut self, addr: u64, val: u32) {
        let old = ld_long(&self.csr, addr);
        let wmask = ld_long(&self.wmask, addr);
        let w1c = ld_long(&self.w1cmask, addr);
        st_long(&mut self.csr, addr, ((old & !wmask) | (val & wmask)) & !(w1c & val));
    }

    fn set_quad(&mut self, addr: u64, val: u64) {
        let old = ld_quad(&self.csr, addr);
        let wmask = ld_quad(&self.wmask, addr);
        let w1c = ld_quad(&self.w1cmask, addr);
        st_quad(&mut self.csr, addr, ((old & !wmask) | (val & wmask)) & !(w1c & val));
    }

    fn set_clear_mask_long(&mut self, addr: u64, clear: u32, mask: u32) -> u32 {
        let v = (self.get_long_raw(addr) & !clear) | mask;
        self.set_long_raw(addr, v);
        v
    }

    fn set_clear_mask_quad(&mut self, addr: u64, clear: u64, mask: u64) -> u64 {
        let v = (self.get_quad_raw(addr) & !clear) | mask;
        self.set_quad_raw(addr, v);
        v
    }

    fn haw(&self) -> u64 {
        haw_mask(self.conf.aw_bits)
    }

    /// `vtd_cap_init()`.
    fn cap_init(&mut self) {
        let c = &self.conf;
        let aw = u64::from(c.aw_bits);
        let mut cap = CAP_FRO
            | (u64::from(FRCD_REG_NR) - 1) << 40
            | CAP_ND
            | CAP_MAMV
            | CAP_PSI
            | CAP_SSLPS
            | CAP_DRAIN
            | CAP_ESRTPS
            | ((aw - 1) & 0x3f) << 16;
        if c.dma_translation {
            if c.aw_bits >= 39 {
                cap |= CAP_SAGAW_39BIT;
            }
            if c.aw_bits >= 48 {
                cap |= CAP_SAGAW_48BIT;
            }
        }
        let mut ecap = ECAP_QI | ECAP_IRO | ECAP_PT;
        if c.intr_supported() {
            ecap |= ECAP_IR | ECAP_MHMV;
            if c.eim == OnOffAuto::On {
                ecap |= ECAP_EIM;
            }
        }
        if c.device_iotlb {
            ecap |= ECAP_DT;
        }
        if c.caching_mode {
            cap |= CAP_CM;
        }
        if c.fsts {
            ecap |= ECAP_SMTS | ECAP_FSTS;
            if c.fs1gp {
                cap |= CAP_FS1GP;
            }
        } else if c.scalable_mode {
            ecap |= ECAP_SMTS | ECAP_SRS | ECAP_SSTS;
        }
        if c.snoop_control {
            ecap |= ECAP_SC;
        }
        if c.pasid_bits != 0 {
            ecap |= u64::from(c.pasid_bits - 1) << 35 | ECAP_PASID;
        }
        self.cap = cap;
        self.ecap = ecap;
    }

    /// `vtd_init()`.
    fn init(&mut self) {
        self.csr = [0; REG_SIZE];
        self.wmask = [0; REG_SIZE];
        self.w1cmask = [0; REG_SIZE];
        self.womask = [0; REG_SIZE];
        self.root = 0;
        self.root_scalable = false;
        self.dmar_enabled = false;
        self.intr_enabled = false;
        self.iq_head = 0;
        self.iq_tail = 0;
        self.iq = 0;
        self.iq_size = 0;
        self.qi_enabled = false;
        self.iq_last_desc_type = INV_DESC_NONE;
        self.iq_dw = false;
        self.next_frcd_reg = 0;
        self.cap_init();

        let haw = self.haw();
        let stale = self.conf.device_iotlb && self.conf.stale_tm;
        let tm = if stale { SS_TM } else { 0 };
        let base = !(haw | SS_IGN_COM);
        self.spte_rsvd =
            [!0, 0x800 | !(haw | SS_IGN_COM | tm), 0x800 | base, 0x800 | base, 0x880 | base];
        let large = !(haw | SS_IGN_COM | tm);
        self.spte_rsvd_large = [0, 0, 0x1f_f800 | large, 0x3fff_f800 | large, 0];
        if self.conf.scalable_mode || self.conf.snoop_control {
            self.spte_rsvd[1] &= !SPTE_SNP;
            self.spte_rsvd_large[2] &= !SPTE_SNP;
            self.spte_rsvd_large[3] &= !SPTE_SNP;
        }
        let fs = !(haw | FS_UPPER_IGNORED);
        self.fpte_rsvd = [!0, fs, fs, fs, 0x80 | fs];
        self.fpte_rsvd_large = [0, 0, 0x1f_e000 | fs, 0x3fff_e000 | fs, 0];
        if self.conf.svm {
            self.ecap |= ECAP_PRS | ECAP_PDS | ECAP_NWFS;
        }
        self.reset_caches();

        self.define_long(DMAR_VER_REG, 0x10, 0, 0);
        self.define_quad(DMAR_CAP_REG, self.cap, 0, 0);
        self.define_quad(DMAR_ECAP_REG, self.ecap, 0, 0);
        self.define_long(DMAR_GCMD_REG, 0, 0xff80_0000, 0);
        st_long(&mut self.womask, DMAR_GCMD_REG, 0xff80_0000);
        self.define_long(DMAR_GSTS_REG, 0, 0, 0);
        self.define_quad(DMAR_RTADDR_REG, 0, 0xffff_ffff_ffff_fc00, 0);
        self.define_quad(DMAR_CCMD_REG, 0, 0xe000_0003_ffff_ffff, 0);
        st_quad(&mut self.womask, DMAR_CCMD_REG, 0x3_ffff_0000);
        self.define_long(DMAR_FSTS_REG, 0, 0, 0x11);
        self.define_long(DMAR_FECTL_REG, 0x8000_0000, 0x8000_0000, 0);
        self.define_long(DMAR_FEDATA_REG, 0, 0xffff, 0);
        self.define_long(DMAR_FEADDR_REG, 0, 0xffff_fffc, 0);
        self.define_long(DMAR_FEUADDR_REG, 0, 0, 0);
        self.define_long(DMAR_PMEN_REG, 0, 0, 0);
        self.define_quad(DMAR_IQH_REG, 0, 0, 0);
        self.define_quad(DMAR_IQT_REG, 0, 0x7fff0, 0);
        self.define_quad(DMAR_IQA_REG, 0, 0xffff_ffff_ffff_f807, 0);
        self.define_long(DMAR_ICS_REG, 0, 0, 0x1);
        self.define_long(DMAR_IECTL_REG, 0x8000_0000, 0x8000_0000, 0);
        self.define_long(DMAR_IEDATA_REG, 0, 0xffff_ffff, 0);
        self.define_long(DMAR_IEADDR_REG, 0, 0xffff_fffc, 0);
        self.define_long(DMAR_IEUADDR_REG, 0, 0, 0);
        self.define_quad(DMAR_IOTLB_REG, 0, 0xb003_ffff_0000_0000, 0);
        self.define_quad(DMAR_IVA_REG, 0, 0xffff_ffff_ffff_f07f, 0);
        st_quad(&mut self.womask, DMAR_IVA_REG, 0xffff_ffff_ffff_f07f);
        self.define_quad(DMAR_FRCD_REG_0_0, 0, 0, 0);
        self.define_quad(DMAR_FRCD_REG_0_2, 0, 0, 0x8000_0000_0000_0000);
        self.define_quad(DMAR_IRTA_REG, 0, 0xffff_ffff_ffff_f80f, 0);
        if self.ecap & ECAP_PRS != 0 {
            self.define_quad(DMAR_PQH_REG, 0, 0x7ffe0, 0);
            self.define_quad(DMAR_PQT_REG, 0, 0x7ffe0, 0);
            self.define_quad(DMAR_PQA_REG, 0, 0xffff_ffff_ffff_f007, 0);
            self.define_long(DMAR_PRS_REG, 0, 0, 0x3);
            self.define_long(DMAR_PECTL_REG, 0, 0x8000_0000, 0);
            self.define_long(DMAR_PEDATA_REG, 0, 0xffff, 0);
            self.define_long(DMAR_PEADDR_REG, 0, 0xffff_fffc, 0);
            self.define_long(DMAR_PEUADDR_REG, 0, 0xffff_ffff, 0);
        }
    }

    /// `vtd_reset_caches()`: the context cache and the IOTLB.
    fn reset_caches(&mut self) {
        self.reset_context_cache();
        self.iotlb.clear();
    }

    /// `vtd_reset_context_cache_locked()`.
    fn reset_context_cache(&mut self) {
        for (generation, _) in self.context_cache.values_mut() {
            *generation = 0;
        }
        self.context_cache_gen = 1;
    }

    /// `vtd_generate_interrupt()`: queues the MSI that two registers describe.
    fn generate_interrupt(&mut self, addr_reg: u64, data_reg: u64) {
        let addr = u64::from(self.get_long_raw(addr_reg));
        let data = self.get_long_raw(data_reg);
        self.msis.push((addr, data));
    }

    /// `vtd_generate_fault_event()`.
    fn generate_fault_event(&mut self, pre_fsts: u32) {
        if pre_fsts & (FSTS_PPF | FSTS_PFO | FSTS_IQE) != 0 {
            return;
        }
        self.set_clear_mask_long(DMAR_FECTL_REG, 0, FECTL_IP);
        if self.get_long_raw(DMAR_FECTL_REG) & FECTL_IM == 0 {
            self.generate_interrupt(DMAR_FEADDR_REG, DMAR_FEDATA_REG);
            self.set_clear_mask_long(DMAR_FECTL_REG, FECTL_IP, 0);
        }
    }

    fn is_frcd_set(&self, index: u16) -> bool {
        self.get_quad_raw(DMAR_FRCD_REG_OFFSET + (u64::from(index) << 4) + 8) & FRCD_F != 0
    }

    /// `vtd_update_fsts_ppf()`.
    fn update_fsts_ppf(&mut self) {
        let ppf = if (0..FRCD_REG_NR).any(|i| self.is_frcd_set(i)) { FSTS_PPF } else { 0 };
        self.set_clear_mask_long(DMAR_FSTS_REG, FSTS_PPF, ppf);
    }

    /// `vtd_set_frcd_and_update_ppf()`.
    fn set_frcd_and_update_ppf(&mut self, index: u16) {
        let addr = DMAR_FRCD_REG_OFFSET + (u64::from(index) << 4) + 8;
        self.set_clear_mask_quad(addr, 0, FRCD_F);
        self.update_fsts_ppf();
    }

    /// `vtd_try_collapse_fault()`.
    fn try_collapse_fault(&self, source_id: u64) -> bool {
        (0..u64::from(FRCD_REG_NR)).any(|i| {
            let r = self.get_quad_raw(DMAR_FRCD_REG_OFFSET + 8 + i * 16);
            r & FRCD_F != 0 && r & FRCD_SID_MASK == source_id
        })
    }

    /// `vtd_report_frcd_fault()`.
    fn report_frcd_fault(&mut self, source_id: u64, hi: u64, lo: u64) {
        let fsts = self.get_long_raw(DMAR_FSTS_REG);
        if fsts & FSTS_PFO != 0 || self.try_collapse_fault(source_id) {
            return;
        }
        let next = self.next_frcd_reg;
        if self.is_frcd_set(next) {
            self.set_clear_mask_long(DMAR_FSTS_REG, 0, FSTS_PFO);
            return;
        }
        let addr = DMAR_FRCD_REG_OFFSET + (u64::from(next) << 4);
        self.set_quad_raw(addr, lo);
        self.set_quad_raw(addr + 8, hi);
        if fsts & FSTS_PPF == 0 {
            self.set_clear_mask_long(
                DMAR_FSTS_REG,
                FSTS_FRI_MASK,
                (u32::from(next) << 8) & FSTS_FRI_MASK,
            );
        }
        self.set_frcd_and_update_ppf(next);
        self.next_frcd_reg = if next + 1 == FRCD_REG_NR { 0 } else { next + 1 };
        if fsts & FSTS_PPF == 0 {
            self.generate_fault_event(fsts);
        }
    }

    /// `vtd_report_fault()`: a DMA remapping fault, unless FPD hides it.
    #[allow(clippy::too_many_arguments)]
    fn report_fault(
        &mut self,
        fault: u8,
        is_fpd_set: bool,
        source_id: u16,
        addr: u64,
        is_write: bool,
        is_pasid: bool,
        pasid: u32,
    ) {
        if is_fpd_set && qualified_fault(fault) {
            return;
        }
        let lo = addr & !0xfff;
        let mut hi = u64::from(source_id)
            | u64::from(fault) << 32
            | (u64::from(pasid) & 0xffff) << 40
            | u64::from(is_pasid) << 31;
        if !is_write {
            hi |= FRCD_T;
        }
        self.report_frcd_fault(u64::from(source_id), hi, lo);
    }

    /// `vtd_report_ir_fault()`.
    fn report_ir_fault(&mut self, source_id: u16, fault: u8, index: u16) {
        let lo = u64::from(index) << 48;
        let hi = u64::from(source_id) | u64::from(fault) << 32;
        self.report_frcd_fault(u64::from(source_id), hi, lo);
    }

    /// `vtd_irte_get()`: the low quad of remapping table entry `index`, checked against the
    /// source ID unless that is [`SID_INVALID`].
    fn irte_get(&mut self, m: &Dma<'_>, index: u16, sid: u16, mut do_fault: bool) -> Option<u64> {
        if u32::from(index) >= self.intr_size {
            if do_fault {
                self.report_ir_fault(sid, FR_IR_INDEX_OVER, index);
            }
            return None;
        }
        let Some([lo, hi]) = m.read_quads::<2>(self.intr_root + u64::from(index) * 16) else {
            if do_fault {
                self.report_ir_fault(sid, FR_IR_ROOT_INVAL, index);
            }
            return None;
        };
        // The remaining faults are qualified by the Fault Processing Disable bit, even "not
        // present".
        if lo & IRTE_FPD != 0 {
            do_fault = false;
        }
        let fault = if lo & IRTE_P == 0 {
            Some(FR_IR_ENTRY_P)
        } else if lo & IRTE_RSVD_LO != 0 || hi & IRTE_RSVD_HI != 0 {
            Some(FR_IR_IRTE_RSVD)
        } else if sid != SID_INVALID && !irte_sid_ok(hi, sid) {
            Some(FR_IR_SID_ERR)
        } else {
            None
        };
        if let Some(f) = fault {
            if do_fault {
                self.report_ir_fault(sid, f, index);
            }
            return None;
        }
        Some(lo)
    }

    /// `vtd_interrupt_remap_msi()` with `vtd_remap_irq_get()` and
    /// `x86_iommu_irq_to_msi_message()`: the message to send in place of `addr` and `data`,
    /// or `None` to drop it.
    fn interrupt_remap_msi(
        &mut self,
        m: &Dma<'_>,
        sid: u16,
        addr: u64,
        data: u32,
        do_fault: bool,
    ) -> Option<(u64, u32)> {
        if !self.intr_enabled {
            return Some((addr, data));
        }
        if addr >> 32 != 0 || (addr >> 20) & 0xfff != 0xfee {
            if do_fault {
                self.report_ir_fault(sid, FR_IR_REQ_RSVD, 0);
            }
            return None;
        }
        let lo = addr as u32;
        if lo & MSI_ADDR_IR_INT_MODE == 0 {
            // Compatibility format.
            return Some((addr, data));
        }
        let shv = lo & MSI_ADDR_IR_SHV != 0;
        let mut index = ((((lo >> 2) & 1) << 15) | ((lo >> 5) & 0x7fff)) as u16;
        if shv {
            index = index.wrapping_add(data as u16);
        }
        let irte = self.irte_get(m, index, sid, do_fault)?;
        if shv && data & 0xffff_0000 != 0 {
            if do_fault {
                self.report_ir_fault(sid, FR_IR_REQ_RSVD, 0);
            }
            return None;
        }
        // QEMU only traces a vector or trigger mode that differs from the IRTE.
        let mut dest = (irte >> 32) as u32;
        if !self.intr_eime {
            dest = (dest & 0xff00) >> 8;
        }
        let out_addr = 0xfee0_0000
            | u64::from(dest & 0xff) << 12
            | ((irte >> 3) & 1) << 3
            | ((irte >> 2) & 1) << 2
            | u64::from(lo & 3)
            | u64::from(dest & 0xffff_ff00) << 32;
        let vector = ((irte >> 16) & 0xff) as u32;
        let delivery = ((irte >> 5) & 7) as u32;
        let trigger = ((irte >> 4) & 1) as u32;
        Some((out_addr, vector | delivery << 8 | 1 << 14 | trigger << 15))
    }

    /// `vtd_dev_to_context_entry()`: the context entry, or the fault with the low quad of the
    /// context entry as far as it was read, for the FPD bit.
    fn dev_to_context_entry(
        &self,
        m: &Dma<'_>,
        bus: u8,
        devfn: u8,
    ) -> Result<ContextEntry, (u8, u64)> {
        let haw = self.haw();
        let Some([re_lo, re_hi]) = m.read_quads::<2>(self.root + u64::from(bus) * 16) else {
            return Err((FR_ROOT_TABLE_INV, 0));
        };
        let upper = self.root_scalable && devfn > 127;
        let present = if upper { re_hi } else { re_lo } & ROOT_ENTRY_P != 0;
        if !present {
            return Err((FR_ROOT_ENTRY_P, 0));
        }
        let rsvd = 0xffe | !haw;
        let bad = if self.root_scalable {
            re_lo & rsvd != 0 || re_hi & rsvd != 0
        } else {
            re_hi != 0 || re_lo & rsvd != 0
        };
        if bad {
            return Err((FR_ROOT_ENTRY_RSVD, 0));
        }
        let (table, index) = if upper {
            (re_hi & !0xfff, u64::from(devfn & 0x7f))
        } else {
            (re_lo & !0xfff, u64::from(devfn))
        };
        let mut ce: ContextEntry = [0; 4];
        if self.root_scalable {
            let Some(v) = m.read_quads::<4>(table + index * 32) else {
                return Err((FR_CONTEXT_TABLE_INV, 0));
            };
            ce = v;
        } else {
            let Some([lo, hi]) = m.read_quads::<2>(table + index * 16) else {
                return Err((FR_CONTEXT_TABLE_INV, 0));
            };
            ce[0] = lo;
            ce[1] = hi;
        }
        if ce[0] & CONTEXT_ENTRY_P == 0 {
            return Err((FR_CONTEXT_ENTRY_P, ce[0]));
        }
        let bad = if self.root_scalable {
            ce[0] & (0x1e0 | !haw) != 0
                || ce[1] & 0xffff_ffff_ffe0_0000 != 0
                || ce[2] != 0
                || ce[3] != 0
        } else {
            ce[1] & CONTEXT_ENTRY_RSVD_HI != 0 || ce[0] & (0xff0 | !haw) != 0
        };
        if bad {
            return Err((FR_CONTEXT_ENTRY_RSVD, ce[0]));
        }
        if !self.root_scalable {
            if !self.ss_level_supported(ce_level(&ce)) {
                return Err((FR_CONTEXT_ENTRY_INV, ce[0]));
            }
            let ok = match ce[0] & CONTEXT_ENTRY_TT {
                CONTEXT_TT_MULTI_LEVEL | CONTEXT_TT_PASS_THROUGH => true,
                CONTEXT_TT_DEV_IOTLB => self.conf.device_iotlb,
                _ => false,
            };
            if !ok {
                return Err((FR_CONTEXT_ENTRY_INV, ce[0]));
            }
        } else {
            // vtd_ce_pasid_0_check(): a present scalable mode context entry needs a good
            // PASID entry for requests without a PASID.
            self.ce_get_pasid_entry(m, &ce, 0).map_err(|fr| (fr, ce[0]))?;
        }
        Ok(ce)
    }

    /// `vtd_pe_type_check()`. Nested translation is not supported, as in QEMU.
    fn pe_type_check(&self, pe: &PasidEntry) -> bool {
        match pe_pgtt(pe) {
            PGTT_FST => self.ecap & ECAP_FSTS != 0,
            PGTT_SST => self.ecap & ECAP_SSTS != 0,
            PGTT_PT => self.ecap & ECAP_PT != 0,
            _ => false,
        }
    }

    /// `vtd_get_pe_in_pasid_leaf_table()`, without a present check.
    fn pe_in_leaf_table(&self, m: &Dma<'_>, pasid: u32, table: u64) -> Result<PasidEntry, u8> {
        let addr = table + u64::from(pasid & 0x3f) * 64;
        let pe = m.read_quads::<8>(addr).ok_or(FR_PASID_TABLE_ACCESS_ERR)?;
        if !self.pe_type_check(&pe) {
            return Err(FR_PASID_TABLE_ENTRY_INV);
        }
        let ok = match pe_pgtt(&pe) {
            PGTT_SST => self.ss_level_supported(pe_ss_level(&pe)),
            PGTT_FST => pe_fs_level(&pe) == 4,
            _ => true,
        };
        if !ok {
            return Err(FR_PASID_TABLE_ENTRY_INV);
        }
        Ok(pe)
    }

    /// `vtd_ce_get_pasid_entry()`: the present PASID entry of `pasid`.
    fn ce_get_pasid_entry(
        &self,
        m: &Dma<'_>,
        ce: &ContextEntry,
        pasid: u32,
    ) -> Result<PasidEntry, u8> {
        let pdire = pdire_get(m, ce, pasid)?;
        if pdire & PASID_DIR_P == 0 {
            return Err(FR_PASID_DIR_ENTRY_P);
        }
        let pe = self.pe_in_leaf_table(m, pasid, pdire & !0xfff)?;
        if pe[0] & PASID_ENTRY_P == 0 {
            return Err(FR_PASID_ENTRY_P);
        }
        Ok(pe)
    }

    /// `vtd_ce_get_pasid_fpd()`. The FPD bits count even where the present bits are clear.
    fn ce_get_pasid_fpd(&self, m: &Dma<'_>, ce: &ContextEntry, pasid: u32) -> Result<bool, u8> {
        let pdire = pdire_get(m, ce, pasid)?;
        if pdire & PASID_DIR_FPD != 0 {
            return Ok(true);
        }
        if pdire & PASID_DIR_P == 0 {
            return Err(FR_PASID_DIR_ENTRY_P);
        }
        let pe = self.pe_in_leaf_table(m, pasid, pdire & !0xfff)?;
        Ok(pe[0] & PASID_ENTRY_FPD != 0)
    }

    /// What `vtd_get_iova_level()`, `vtd_get_iova_agaw()`, `vtd_get_iova_pgtbl_base()` and
    /// `vtd_get_domain_id()` give for PASID 0. Like QEMU they do not look at whether the PASID
    /// entry could be read; a failed read here counts as an entry of zeroes.
    fn stage(&self, m: &Dma<'_>, ce: &ContextEntry) -> Stage {
        if !self.root_scalable {
            return Stage {
                level: ce_level(ce),
                agaw: ce_agaw(ce),
                base: ce[0] & !0xfff,
                did: ((ce[1] >> 8) & 0xffff) as u16,
            };
        }
        let pe = self.ce_get_pasid_entry(m, ce, 0).unwrap_or([0; 8]);
        let (level, base) = if self.conf.fsts {
            (pe_fs_level(&pe), pe[2] & !0xfff)
        } else {
            (pe_ss_level(&pe), pe[0] & !0xfff)
        };
        Stage { level, agaw: 30 + ((pe[0] >> 2) & 7) as u32 * 9, base, did: pe[1] as u16 }
    }

    /// `vtd_dev_pt_enabled()` for PASID 0.
    fn dev_pt_enabled(&self, m: &Dma<'_>, ce: &ContextEntry) -> bool {
        if self.root_scalable {
            return self.ce_get_pasid_entry(m, ce, 0).is_ok_and(|pe| pe_pgtt(&pe) == PGTT_PT);
        }
        ce[0] & CONTEXT_ENTRY_TT == CONTEXT_TT_PASS_THROUGH
    }

    /// `vtd_is_ss_level_supported()`.
    fn ss_level_supported(&self, level: u32) -> bool {
        (2..=6).contains(&level)
            && self.cap & CAP_SAGAW_MASK & (1 << (level - 2 + CAP_SAGAW_SHIFT as u32)) != 0
    }

    /// `vtd_sspte_nonzero_rsvd()`.
    fn sspte_nonzero_rsvd(&self, pte: u64, level: u32) -> bool {
        let l = level as usize;
        let mask = if (l == 2 || l == 3) && pte & PT_PAGE_SIZE_MASK != 0 {
            self.spte_rsvd_large[l]
        } else {
            self.spte_rsvd.get(l).copied().unwrap_or(!0)
        };
        pte & mask != 0
    }

    /// `vtd_iova_to_sspte()`: the leaf entry with its level and the access the walk allows.
    fn iova_to_sspte(
        &self,
        m: &Dma<'_>,
        st: &Stage,
        iova: u64,
        is_write: bool,
    ) -> Result<(u64, u32, bool, bool), u8> {
        let aw = self.conf.aw_bits;
        let top = st.level;
        let limit = 1u64 << st.agaw.min(u32::from(aw));
        if iova & !(limit - 1) != 0 {
            return Err(FR_ADDR_BEYOND_MGAW);
        }
        let check = if is_write { 2 } else { 1 };
        let (mut reads, mut writes) = (true, true);
        let mut addr = st.base;
        let mut level = top;
        loop {
            let offset = (iova >> level_shift(level)) & 0x1ff;
            let pte = m.pte(addr, offset);
            if pte == u64::MAX {
                return Err(if level == top { FR_CONTEXT_ENTRY_INV } else { FR_PAGING_ENTRY_INV });
            }
            reads &= pte & 1 != 0;
            writes &= pte & 2 != 0;
            if pte & check == 0 {
                return Err(if is_write { FR_WRITE } else { FR_READ });
            }
            if self.sspte_nonzero_rsvd(pte, level) {
                return Err(FR_PAGING_ENTRY_RSVD);
            }
            if level == 1 || pte & PT_PAGE_SIZE_MASK != 0 {
                return Ok((pte, level, reads, writes));
            }
            addr = pte & !0xfff & haw_mask(aw);
            level -= 1;
        }
    }

    /// `vtd_fspte_nonzero_rsvd()`.
    fn fspte_nonzero_rsvd(&self, pte: u64, level: u32) -> bool {
        let l = level as usize;
        let mask = if (l == 2 || l == 3) && pte & PT_PAGE_SIZE_MASK != 0 {
            self.fpte_rsvd_large[l]
        } else {
            self.fpte_rsvd.get(l).copied().unwrap_or(!0)
        };
        pte & mask != 0
    }

    /// `vtd_iova_to_fspte()`: the leaf entry with its level and the access the walk allows,
    /// or the fault with the level the walk was at. The accessed bit is set on the way down,
    /// and the dirty bit on the leaf of a write, each only when neither bit is set yet, which
    /// is what `vtd_set_flag_in_pte()` does.
    fn iova_to_fspte(
        &self,
        m: &Dma<'_>,
        st: &Stage,
        iova: u64,
        is_write: bool,
    ) -> Result<(u64, u32, bool, bool), (u8, u32)> {
        let aw = self.conf.aw_bits;
        let top = st.level;
        let limit = 1u64 << st.agaw.min(u32::from(aw));
        let upper_mask = !(limit - 1);
        let upper = iova & upper_mask;
        let canonical = if iova & (limit >> 1) != 0 { upper == upper_mask } else { upper == 0 };
        if !canonical {
            return Err((FR_FS_NON_CANONICAL, top));
        }
        let mut writes = true;
        let mut addr = st.base;
        let mut level = top;
        loop {
            let offset = (iova >> level_shift(level)) & 0x1ff;
            let pte = m.pte(addr, offset);
            if pte == u64::MAX {
                let fr =
                    if level == top { FR_PASID_ENTRY_FSPTPTR_INV } else { FR_FS_PAGING_ENTRY_INV };
                return Err((fr, level));
            }
            if pte & FS_P == 0 {
                return Err((FR_FS_PAGING_ENTRY_P, level));
            }
            if pte & FS_US == 0 {
                return Err((FR_FS_PAGING_ENTRY_US, level));
            }
            writes &= pte & FS_RW != 0;
            if is_write && pte & FS_RW == 0 {
                return Err((FR_SM_WRITE, level));
            }
            if self.fspte_nonzero_rsvd(pte, level) {
                return Err((FR_FS_PAGING_ENTRY_RSVD, level));
            }
            let last = level == 1 || pte & PT_PAGE_SIZE_MASK != 0;
            let flag = if last && is_write { FS_A | FS_D } else { FS_A };
            if pte & flag == 0 && !m.write(addr + offset * 8, &(pte | flag).to_le_bytes()) {
                return Err((FR_FS_BIT_UPDATE_FAILED, level));
            }
            if last {
                return Ok((pte, level, true, writes));
            }
            addr = pte & !0xfff & haw_mask(aw);
            level -= 1;
        }
    }

    /// `vtd_lookup_iotlb()`.
    fn lookup_iotlb(&self, sid: u16, pasid: u32, addr: u64) -> Option<IotlbEntry> {
        (1..4).find_map(|level| {
            let gfn = (addr & level_page_mask(level)) >> 12;
            self.iotlb.get(&IotlbKey { gfn, sid, level, pasid }).copied()
        })
    }

    /// `vtd_update_iotlb()`.
    #[allow(clippy::too_many_arguments)]
    fn update_iotlb(
        &mut self,
        sid: u16,
        domain_id: u16,
        addr: u64,
        pte: u64,
        access_flags: u8,
        level: u32,
        pasid: u32,
        pgtt: u8,
    ) {
        let gfn = (addr & level_page_mask(level)) >> 12;
        if self.iotlb.len() >= IOTLB_MAX_SIZE {
            self.iotlb.clear();
        }
        let mask = level_page_mask(level);
        self.iotlb.insert(
            IotlbKey { gfn, sid, level, pasid },
            IotlbEntry { gfn, domain_id, pte, access_flags, mask, pgtt },
        );
    }

    /// `vtd_do_iommu_translate()` for a request without PASID.
    fn do_translate(&mut self, m: &Dma<'_>, sid: u16, addr: u64, is_write: bool) -> Xlate {
        let (pte, flags, page_mask) = match self.lookup_iotlb(sid, 0, addr) {
            Some(e) => (e.pte, e.access_flags, e.mask),
            None => match self.walk(m, sid, addr, is_write) {
                Walked::PassThrough => {
                    let iova = addr & PAGE_MASK_4K;
                    return Xlate { iova, translated: iova, mask: !PAGE_MASK_4K, perm: 3 };
                }
                Walked::Fault { mask } => {
                    return Xlate { iova: 0, translated: 0, mask, perm: 0 };
                }
                Walked::Page { pte, level, flags, did, pgtt } => {
                    self.update_iotlb(sid, did, addr, pte, flags, level, 0, pgtt);
                    (pte, flags, level_page_mask(level))
                }
            },
        };
        let translated = pte & !0xfff & self.haw() & page_mask;
        let perm = if is_write { flags } else { flags & !ACCESS_W };
        Xlate { iova: addr & page_mask, translated, mask: !page_mask, perm }
    }

    /// The part of `vtd_do_iommu_translate()` after an IOTLB miss.
    fn walk(&mut self, m: &Dma<'_>, sid: u16, addr: u64, is_write: bool) -> Walked {
        let fault = |s: &mut Self, fr: u8, fpd: bool| {
            s.report_fault(fr, fpd, sid, addr, is_write, false, 0);
            Walked::Fault { mask: !PAGE_MASK_4K }
        };
        let (ce, fpd) = match self.context_cache.get(&sid) {
            Some(&(generation, ce)) if generation == self.context_cache_gen => {
                let mut fpd = ce[0] & CONTEXT_ENTRY_FPD != 0;
                if !fpd && self.root_scalable {
                    match self.ce_get_pasid_fpd(m, &ce, 0) {
                        Ok(f) => fpd = f,
                        Err(fr) => return fault(self, fr, false),
                    }
                }
                (ce, fpd)
            }
            _ => {
                let (ce, mut fpd) = match self.dev_to_context_entry(m, (sid >> 8) as u8, sid as u8)
                {
                    Ok(ce) => (ce, ce[0] & CONTEXT_ENTRY_FPD != 0),
                    Err((fr, lo)) => return fault(self, fr, lo & CONTEXT_ENTRY_FPD != 0),
                };
                if !fpd && self.root_scalable {
                    match self.ce_get_pasid_fpd(m, &ce, 0) {
                        Ok(f) => fpd = f,
                        Err(fr) => return fault(self, fr, false),
                    }
                }
                self.context_cache.insert(sid, (self.context_cache_gen, ce));
                (ce, fpd)
            }
        };
        if self.dev_pt_enabled(m, &ce) {
            return Walked::PassThrough;
        }
        let st = self.stage(m, &ce);
        let walked = if self.conf.fsts && self.root_scalable {
            self.iova_to_fspte(m, &st, addr, is_write).map(|w| (w, PGTT_FST))
        } else {
            self.iova_to_sspte(m, &st, addr, is_write).map(|w| (w, PGTT_SST)).map_err(|fr| (fr, 0))
        };
        match walked {
            Ok(((pte, level, reads, writes), pgtt)) => {
                let xlat = pte & !0xfff & self.haw();
                let size = 1u64 << level_shift(level);
                if xlat <= INTERRUPT_ADDR_LAST && xlat + size > INTERRUPT_ADDR_FIRST {
                    let fr = if self.conf.scalable_mode {
                        FR_SM_INTERRUPT_ADDR
                    } else {
                        FR_INTERRUPT_ADDR
                    };
                    self.report_fault(fr, fpd, sid, addr, is_write, false, 0);
                    return Walked::Fault { mask: !level_page_mask(level) };
                }
                let flags = (u8::from(reads) * ACCESS_R) | (u8::from(writes) * ACCESS_W);
                Walked::Page { pte, level, flags, did: st.did, pgtt }
            }
            // The second stage walk leaves the level unset on a fault, the first stage
            // walk leaves it where the walk stopped.
            Err((fr, level)) => {
                self.report_fault(fr, fpd, sid, addr, is_write, false, 0);
                let mask = if level == 0 { !PAGE_MASK_4K } else { !level_page_mask(level) };
                Walked::Fault { mask }
            }
        }
    }

    /// `vtd_context_global_invalidate()`.
    fn context_global_invalidate(&mut self) {
        self.context_cache_gen = self.context_cache_gen.wrapping_add(1);
        if self.context_cache_gen == CONTEXT_CACHE_GEN_MAX {
            self.reset_context_cache();
        }
    }

    /// `vtd_context_device_invalidate()`.
    fn context_device_invalidate(&mut self, source_id: u16, func_mask: u64) {
        let mask: u16 = !match func_mask & 3 {
            0 => 0,
            1 => 4,
            2 => 6,
            _ => 7,
        };
        let (bus, devfn) = (source_id >> 8, source_id & 0xff);
        for (sid, (generation, _)) in &mut self.context_cache {
            if sid >> 8 == bus && (sid & 0xff & mask) == (devfn & mask) {
                *generation = 0;
            }
        }
    }

    fn iotlb_domain_invalidate(&mut self, domain_id: u16) {
        self.iotlb.retain(|_, e| e.domain_id != domain_id);
    }

    /// `vtd_iotlb_page_invalidate()` with `vtd_hash_remove_by_page()`.
    fn iotlb_page_invalidate(&mut self, domain_id: u16, addr: u64, am: u64) {
        let mask = !((1u64 << am) - 1);
        let gfn = (addr >> 12) & mask;
        self.iotlb.retain(|_, e| {
            let gfn_tlb = (addr & e.mask) >> 12;
            // First stage entries of the domain all go, as the spec says.
            !(e.domain_id == domain_id
                && (e.pgtt == PGTT_FST || (e.gfn & mask) == gfn || e.gfn == gfn_tlb))
        });
    }

    /// `vtd_handle_gcmd_write()`.
    fn handle_gcmd_write(&mut self, m: &Dma<'_>) {
        let status = self.get_long_raw(DMAR_GSTS_REG);
        let val = self.get_long_raw(DMAR_GCMD_REG);
        let changed = status ^ val;
        if changed & GCMD_TE != 0 && self.conf.dma_translation {
            let en = val & GCMD_TE != 0;
            if self.dmar_enabled != en {
                self.dmar_enabled = en;
                if en {
                    self.set_clear_mask_long(DMAR_GSTS_REG, 0, GSTS_TES);
                } else {
                    self.next_frcd_reg = 0;
                    self.set_clear_mask_long(DMAR_GSTS_REG, GSTS_TES, 0);
                }
                self.reset_caches();
            }
        }
        if val & GCMD_SRTP != 0 {
            let rtaddr = self.get_quad_raw(DMAR_RTADDR_REG);
            self.root = rtaddr & (self.haw() ^ 0xfff);
            if self.conf.scalable_mode {
                self.root_scalable = rtaddr & RTADDR_SMT != 0;
            }
            self.set_clear_mask_long(DMAR_GSTS_REG, 0, GSTS_RTPS);
            self.reset_caches();
        }
        if changed & GCMD_QIE != 0 {
            self.handle_gcmd_qie(m, val & GCMD_QIE != 0);
        }
        if val & GCMD_SIRTP != 0 {
            let irta = self.get_quad_raw(DMAR_IRTA_REG);
            self.intr_size = 1 << ((irta & IRTA_SIZE_MASK) + 1);
            self.intr_root = irta & (self.haw() ^ 0xfff);
            self.intr_eime = irta & IRTA_EIME != 0;
            self.set_clear_mask_long(DMAR_GSTS_REG, 0, GSTS_IRTPS);
        }
        if changed & GCMD_IRE != 0 && self.conf.intr_supported() {
            if val & GCMD_IRE != 0 {
                self.intr_enabled = true;
                self.set_clear_mask_long(DMAR_GSTS_REG, 0, GSTS_IRES);
            } else {
                self.intr_enabled = false;
                self.set_clear_mask_long(DMAR_GSTS_REG, GSTS_IRES, 0);
            }
        }
    }

    /// `vtd_handle_gcmd_qie()`.
    fn handle_gcmd_qie(&mut self, m: &Dma<'_>, en: bool) {
        let iqa = self.get_quad_raw(DMAR_IQA_REG);
        if en {
            self.iq = iqa & (self.haw() ^ 0xfff);
            self.iq_size = 1 << ((iqa & IQA_QS) + 8 - u64::from(self.iq_dw));
            self.qi_enabled = true;
            self.set_clear_mask_long(DMAR_GSTS_REG, 0, GSTS_QIES);
            if self.iq_tail != 0 && self.get_long_raw(DMAR_FSTS_REG) & FSTS_IQE == 0 {
                self.fetch_inv_desc(m);
            }
        } else if self.qi_enabled
            && self.iq_tail == self.iq_head
            && self.iq_last_desc_type == INV_DESC_WAIT
        {
            self.set_quad_raw(DMAR_IQH_REG, 0);
            self.iq_head = 0;
            self.qi_enabled = false;
            self.set_clear_mask_long(DMAR_GSTS_REG, GSTS_QIES, 0);
        }
    }

    /// `vtd_handle_ccmd_write()`.
    fn handle_ccmd_write(&mut self) {
        let val = self.get_quad_raw(DMAR_CCMD_REG);
        if val & CCMD_ICC == 0 || self.qi_enabled {
            return;
        }
        let caig = match val & CCMD_CIRG_MASK {
            CCMD_DOMAIN_INVL | CCMD_GLOBAL_INVL => {
                self.context_global_invalidate();
                CCMD_GLOBAL_INVL_A
            }
            CCMD_DEVICE_INVL => {
                self.context_device_invalidate(((val >> 16) & 0xffff) as u16, (val >> 32) & 3);
                CCMD_DEVICE_INVL_A
            }
            _ => 0,
        };
        self.set_clear_mask_quad(DMAR_CCMD_REG, CCMD_ICC, 0);
        self.set_clear_mask_quad(DMAR_CCMD_REG, CCMD_CAIG_MASK, caig);
    }

    /// `vtd_handle_iotlb_write()`.
    fn handle_iotlb_write(&mut self) {
        let val = self.get_quad_raw(DMAR_IOTLB_REG);
        if val & TLB_IVT == 0 || self.qi_enabled {
            return;
        }
        let domain_id = ((val >> 32) & 0xffff) as u16;
        let iaig = match val & TLB_FLUSH_GRANU_MASK {
            TLB_GLOBAL_FLUSH => {
                self.iotlb.clear();
                TLB_GLOBAL_FLUSH_A
            }
            TLB_DSI_FLUSH => {
                self.iotlb_domain_invalidate(domain_id);
                TLB_DSI_FLUSH_A
            }
            TLB_PSI_FLUSH => {
                let iva = self.get_quad_raw(DMAR_IVA_REG);
                let am = iva & 0x3f;
                if am > MAMV {
                    0
                } else {
                    self.iotlb_page_invalidate(domain_id, iva & !0xfff, am);
                    TLB_PSI_FLUSH_A
                }
            }
            _ => 0,
        };
        self.set_clear_mask_quad(DMAR_IOTLB_REG, TLB_IVT, 0);
        self.set_clear_mask_quad(DMAR_IOTLB_REG, TLB_FLUSH_GRANU_MASK_A, iaig);
    }

    /// `vtd_handle_inv_queue_error()`.
    fn handle_inv_queue_error(&mut self) {
        let fsts = self.get_long_raw(DMAR_FSTS_REG);
        self.set_clear_mask_long(DMAR_FSTS_REG, 0, FSTS_IQE);
        self.generate_fault_event(fsts);
    }

    /// `vtd_fetch_inv_desc()`.
    fn fetch_inv_desc(&mut self, m: &Dma<'_>) {
        let shift = if self.iq_dw { 5 } else { 4 };
        if u32::from(self.iq_tail) >= self.iq_size {
            self.handle_inv_queue_error();
            return;
        }
        while self.iq_head != self.iq_tail {
            if !self.process_inv_desc(m) {
                self.handle_inv_queue_error();
                break;
            }
            self.set_quad_raw(DMAR_IQH_REG, (u64::from(self.iq_head) << shift) & IQH_QH_MASK);
        }
    }

    /// `vtd_inv_desc_reserved_check()`.
    fn inv_desc_reserved_check(
        &self,
        d: &[u64; 4],
        mask: [u64; 4],
        dw: bool,
        func: &str,
        desc_type: &str,
    ) -> bool {
        if self.iq_dw {
            if d.iter().zip(mask).any(|(v, m)| v & m != 0) {
                error_report(&format!(
                    "{func}: invalid {desc_type} desc val[3]: 0x{:x} val[2]: 0x{:x} \
                     val[1]=0x{:x} val[0]=0x{:x} (reserved nonzero)",
                    d[3], d[2], d[1], d[0]
                ));
                return false;
            }
        } else {
            if dw {
                error_report(&format!(
                    "{func}: 256-bit {desc_type} desc in 128-bit invalidation queue"
                ));
                return false;
            }
            if d[0] & mask[0] != 0 || d[1] & mask[1] != 0 {
                error_report(&format!(
                    "{func}: invalid {desc_type} desc: hi={:x}, lo={:x} (reserved nonzero)",
                    d[1], d[0]
                ));
                return false;
            }
        }
        true
    }

    /// `vtd_process_inv_desc()`.
    fn process_inv_desc(&mut self, m: &Dma<'_>) -> bool {
        let addr = self.iq + u64::from(self.iq_head) * if self.iq_dw { 32 } else { 16 };
        let d = if self.iq_dw {
            m.read_quads::<4>(addr)
        } else {
            m.read_quads::<2>(addr).map(|[lo, hi]| [lo, hi, 0, 0])
        };
        let Some(d) = d else {
            self.iq_last_desc_type = INV_DESC_NONE;
            return false;
        };
        let ty = (((d[0] >> 5) & 0x70) | (d[0] & 0xf)) as u8;
        self.iq_last_desc_type = ty;
        let ok = match ty {
            INV_DESC_CC => self.process_context_cache_desc(&d),
            INV_DESC_IOTLB => self.process_iotlb_desc(&d),
            INV_DESC_PC => self.process_pasid_desc(&d),
            INV_DESC_PIOTLB => self.process_piotlb_desc(&d),
            INV_DESC_WAIT => self.process_wait_desc(m, &d),
            INV_DESC_IEC => self.inv_desc_reserved_check(
                &d,
                [0xffff_0000_07ff_f1e0, INV_DESC_ALL_ONE, INV_DESC_ALL_ONE, INV_DESC_ALL_ONE],
                false,
                "vtd_process_inv_iec_desc",
                "iec inv",
            ),
            INV_DESC_DEV_PIOTLB => self.inv_desc_reserved_check(
                &d,
                [0xfff0_0000_0000_f000, 0x7fe, INV_DESC_ALL_ONE, INV_DESC_ALL_ONE],
                true,
                "vtd_process_device_piotlb_desc",
                "device piotlb inv",
            ),
            INV_DESC_DEVICE => self.inv_desc_reserved_check(
                &d,
                [0xffff_0000_ffe0_f1f0, 0xffe, INV_DESC_ALL_ONE, INV_DESC_ALL_ONE],
                false,
                "vtd_process_device_iotlb_desc",
                "dev-iotlb inv",
            ),
            // vtd_process_page_group_response_desc(): no device asks for page requests.
            INV_DESC_PGRESP => {
                d[0] & 0xfff0_0000_0000_01e0 == 0
                    && d[1] & 0xffff_ffff_ffff_f003 == 0
                    && d[0] & (1 << 4) != 0
            }
            _ => false,
        };
        if !ok {
            return false;
        }
        self.iq_head += 1;
        if u32::from(self.iq_head) == self.iq_size {
            self.iq_head = 0;
        }
        true
    }

    /// `vtd_process_context_cache_desc()`.
    fn process_context_cache_desc(&mut self, d: &[u64; 4]) -> bool {
        let mask = [0xfffc_0000_0000_f1c0, INV_DESC_ALL_ONE, INV_DESC_ALL_ONE, INV_DESC_ALL_ONE];
        if !self.inv_desc_reserved_check(d, mask, false, "vtd_process_context_cache_desc", "cc inv")
        {
            return false;
        }
        match d[0] & 0x30 {
            0x20 | 0x10 => self.context_global_invalidate(),
            0x30 => {
                self.context_device_invalidate(((d[0] >> 32) & 0xffff) as u16, (d[0] >> 48) & 3)
            }
            _ => return false,
        }
        true
    }

    /// `vtd_process_iotlb_desc()`.
    fn process_iotlb_desc(&mut self, d: &[u64; 4]) -> bool {
        let mask = [0xffff_ffff_0000_f100, 0xf80, INV_DESC_ALL_ONE, INV_DESC_ALL_ONE];
        if !self.inv_desc_reserved_check(d, mask, false, "vtd_process_iotlb_desc", "iotlb inv") {
            return false;
        }
        let domain_id = ((d[0] >> 16) & 0xffff) as u16;
        match d[0] & 0x30 {
            0x10 => self.iotlb.clear(),
            0x20 => self.iotlb_domain_invalidate(domain_id),
            0x30 => {
                let am = d[1] & 0x3f;
                if am > MAMV {
                    return false;
                }
                self.iotlb_page_invalidate(domain_id, d[1] & !0xfff, am);
            }
            _ => return false,
        }
        true
    }

    /// `vtd_process_pasid_desc()`. There is no PASID cache to sync: a translation reads the
    /// PASID entry each time it walks.
    fn process_pasid_desc(&mut self, d: &[u64; 4]) -> bool {
        let mask = [0xfff0_0000_0000_f1c0, INV_DESC_ALL_ONE, INV_DESC_ALL_ONE, INV_DESC_ALL_ONE];
        if !self.inv_desc_reserved_check(d, mask, true, "vtd_process_pasid_desc", "pasid cache inv")
        {
            return false;
        }
        (d[0] >> 4) & 3 != 2
    }

    /// `vtd_process_piotlb_desc()`.
    fn process_piotlb_desc(&mut self, d: &[u64; 4]) -> bool {
        let mask = [0xfff0_0000_0000_f1c0, 0xf80, INV_DESC_ALL_ONE, INV_DESC_ALL_ONE];
        if !self.inv_desc_reserved_check(d, mask, true, "vtd_process_piotlb_desc", "piotlb inv") {
            return false;
        }
        let domain_id = ((d[0] >> 16) & 0xffff) as u16;
        let pasid = ((d[0] >> 32) & 0xf_ffff) as u32;
        match d[0] & 0x30 {
            // vtd_piotlb_pasid_invalidate()
            0x20 => self.iotlb.retain(|k, e| !(e.domain_id == domain_id && k.pasid == pasid)),
            // vtd_piotlb_page_invalidate(), which leaves second stage and pass-through entries
            0x30 => {
                let am = d[1] & 0x3f;
                if am > MAMV {
                    return false;
                }
                let addr = d[1] & !0xfff;
                let mask = !((1u64 << am) - 1);
                let gfn = (addr >> 12) & mask;
                self.iotlb.retain(|k, e| {
                    let gfn_tlb = (addr & e.mask) >> 12;
                    !(e.pgtt == PGTT_FST
                        && e.domain_id == domain_id
                        && k.pasid == pasid
                        && ((e.gfn & mask) == gfn || e.gfn == gfn_tlb))
                });
            }
            _ => return false,
        }
        true
    }

    /// `vtd_process_wait_desc()`.
    fn process_wait_desc(&mut self, m: &Dma<'_>, d: &[u64; 4]) -> bool {
        let lo_rsvd = 0xffff_f100 | if self.ecap & ECAP_PDS != 0 { 0 } else { 1 << 7 };
        let mask = [lo_rsvd, 3, INV_DESC_ALL_ONE, INV_DESC_ALL_ONE];
        if !self.inv_desc_reserved_check(d, mask, false, "vtd_process_wait_desc", "wait") {
            return false;
        }
        let mut ret = true;
        if d[0] & INV_DESC_WAIT_SW != 0 {
            let data = ((d[0] >> 32) as u32).to_le_bytes();
            if !m.write(d[1], &data) {
                ret = false;
            }
        }
        if d[0] & INV_DESC_WAIT_IF != 0 {
            self.generate_completion_event();
        }
        if d[0] & (INV_DESC_WAIT_IF | INV_DESC_WAIT_SW | INV_DESC_WAIT_FN) == 0 {
            return false;
        }
        ret
    }

    /// `vtd_generate_completion_event()`.
    fn generate_completion_event(&mut self) {
        if self.get_long_raw(DMAR_ICS_REG) & ICS_IWC != 0 {
            return;
        }
        self.set_clear_mask_long(DMAR_ICS_REG, 0, ICS_IWC);
        self.set_clear_mask_long(DMAR_IECTL_REG, 0, IECTL_IP);
        if self.get_long_raw(DMAR_IECTL_REG) & IECTL_IM == 0 {
            self.generate_interrupt(DMAR_IEADDR_REG, DMAR_IEDATA_REG);
            self.set_clear_mask_long(DMAR_IECTL_REG, IECTL_IP, 0);
        }
    }

    /// `vtd_handle_iqt_write()`.
    fn handle_iqt_write(&mut self, m: &Dma<'_>) {
        let val = self.get_quad_raw(DMAR_IQT_REG);
        if self.iq_dw && val & IQT_QT_256_RSV_BIT != 0 {
            self.handle_inv_queue_error();
            return;
        }
        self.iq_tail =
            if self.iq_dw { ((val >> 5) & 0x3fff) as u16 } else { ((val >> 4) & 0x7fff) as u16 };
        if self.qi_enabled && self.get_long_raw(DMAR_FSTS_REG) & FSTS_IQE == 0 {
            self.fetch_inv_desc(m);
        }
    }

    /// `vtd_handle_fsts_write()`.
    fn handle_fsts_write(&mut self) {
        let fsts = self.get_long_raw(DMAR_FSTS_REG);
        let fectl = self.get_long_raw(DMAR_FECTL_REG);
        if fectl & FECTL_IP != 0 && fsts & (FSTS_PFO | FSTS_PPF | FSTS_IQE) == 0 {
            self.set_clear_mask_long(DMAR_FECTL_REG, FECTL_IP, 0);
        }
    }

    /// `vtd_handle_fectl_write()`.
    fn handle_fectl_write(&mut self) {
        let fectl = self.get_long_raw(DMAR_FECTL_REG);
        if fectl & FECTL_IP != 0 && fectl & FECTL_IM == 0 {
            self.generate_interrupt(DMAR_FEADDR_REG, DMAR_FEDATA_REG);
            self.set_clear_mask_long(DMAR_FECTL_REG, FECTL_IP, 0);
        }
    }

    /// `vtd_handle_ics_write()`.
    fn handle_ics_write(&mut self) {
        let ics = self.get_long_raw(DMAR_ICS_REG);
        let iectl = self.get_long_raw(DMAR_IECTL_REG);
        if iectl & IECTL_IP != 0 && ics & ICS_IWC == 0 {
            self.set_clear_mask_long(DMAR_IECTL_REG, IECTL_IP, 0);
        }
    }

    /// `vtd_handle_iectl_write()`.
    /// `vtd_handle_prs_write()`.
    fn handle_prs_write(&mut self) {
        let prs = self.get_long_raw(DMAR_PRS_REG);
        if prs & (PR_STATUS_PPR | PR_STATUS_PRO) == 0 {
            self.set_clear_mask_long(DMAR_PECTL_REG, PR_PECTL_IP, 0);
        }
    }

    /// `vtd_handle_pectl_write()`.
    fn handle_pectl_write(&mut self) {
        let pectl = self.get_long_raw(DMAR_PECTL_REG);
        if pectl & PR_PECTL_IP != 0 && pectl & PR_PECTL_IM == 0 {
            self.set_clear_mask_long(DMAR_PECTL_REG, PR_PECTL_IP, 0);
            self.generate_interrupt(DMAR_PEADDR_REG, DMAR_PEDATA_REG);
        }
    }

    fn handle_iectl_write(&mut self) {
        let iectl = self.get_long_raw(DMAR_IECTL_REG);
        if iectl & IECTL_IP != 0 && iectl & IECTL_IM == 0 {
            self.generate_interrupt(DMAR_IEADDR_REG, DMAR_IEDATA_REG);
            self.set_clear_mask_long(DMAR_IECTL_REG, IECTL_IP, 0);
        }
    }

    /// `vtd_mem_read()`.
    fn mem_read(&self, addr: u64, size: u64) -> u64 {
        if addr + size > DMAR_REG_SIZE {
            return u64::MAX;
        }
        let lo32 = |v: u64| if size == 4 { v & 0xffff_ffff } else { v };
        match addr {
            DMAR_RTADDR_REG => lo32(self.get_quad_raw(DMAR_RTADDR_REG)),
            DMAR_RTADDR_REG_HI => self.get_quad_raw(DMAR_RTADDR_REG) >> 32,
            DMAR_IQA_REG => lo32(self.iq | (self.get_quad(DMAR_IQA_REG) & (IQA_QS | IQA_DW_MASK))),
            DMAR_IQA_REG_HI => self.iq >> 32,
            DMAR_PEUADDR_REG => u64::from(self.get_long_raw(DMAR_PEUADDR_REG)),
            _ if size == 4 => u64::from(self.get_long(addr)),
            _ => self.get_quad(addr),
        }
    }

    /// `vtd_mem_write()`.
    fn mem_write(&mut self, m: &Dma<'_>, addr: u64, val: u64, size: u64) {
        if addr + size > DMAR_REG_SIZE {
            return;
        }
        let long = val as u32;
        let sized = |s: &mut Self| {
            if size == 4 {
                s.set_long(addr, long);
            } else {
                s.set_quad(addr, val);
            }
        };
        match addr {
            DMAR_GCMD_REG => {
                self.set_long(addr, long);
                self.handle_gcmd_write(m);
            }
            DMAR_CCMD_REG if size == 4 => self.set_long(addr, long),
            DMAR_CCMD_REG => {
                self.set_quad(addr, val);
                self.handle_ccmd_write();
            }
            DMAR_CCMD_REG_HI => {
                self.set_long(addr, long);
                self.handle_ccmd_write();
            }
            DMAR_IOTLB_REG if size == 4 => self.set_long(addr, long),
            DMAR_IOTLB_REG => {
                self.set_quad(addr, val);
                self.handle_iotlb_write();
            }
            DMAR_IOTLB_REG_HI => {
                self.set_long(addr, long);
                self.handle_iotlb_write();
            }
            DMAR_FSTS_REG => {
                self.set_long(addr, long);
                self.handle_fsts_write();
            }
            DMAR_FECTL_REG => {
                self.set_long(addr, long);
                self.handle_fectl_write();
            }
            DMAR_IQT_REG => {
                sized(self);
                self.handle_iqt_write(m);
            }
            DMAR_IQA_REG => {
                sized(self);
                self.iq_dw = self.ecap & ECAP_SMTS != 0
                    && self.get_quad_raw(DMAR_IQA_REG) & IQA_DW_MASK != 0;
            }
            DMAR_ICS_REG => {
                self.set_long(addr, long);
                self.handle_ics_write();
            }
            DMAR_IECTL_REG => {
                self.set_long(addr, long);
                self.handle_iectl_write();
            }
            DMAR_PRS_REG => {
                self.set_long(addr, long);
                self.handle_prs_write();
            }
            DMAR_PECTL_REG => {
                self.set_long(addr, long);
                self.handle_pectl_write();
            }
            DMAR_FRCD_REG_0_2 if size == 4 => self.set_long(addr, long),
            DMAR_FRCD_REG_0_2 => {
                self.set_quad(addr, val);
                self.update_fsts_ppf();
            }
            DMAR_FRCD_REG_0_3 => {
                self.set_long(addr, long);
                self.update_fsts_ppf();
            }
            DMAR_FEDATA_REG | DMAR_FEUADDR_REG | DMAR_PMEN_REG | DMAR_RTADDR_REG_HI
            | DMAR_IQT_REG_HI | DMAR_IQA_REG_HI | DMAR_IEDATA_REG | DMAR_IEADDR_REG
            | DMAR_IEUADDR_REG | DMAR_FRCD_REG_0_1 | DMAR_IRTA_REG_HI | DMAR_IVA_REG_HI
            | DMAR_PEUADDR_REG => self.set_long(addr, long),
            _ => sized(self),
        }
    }
}

/// The level of the second stage table of a legacy context entry, `vtd_ce_get_level()`.
fn ce_level(ce: &ContextEntry) -> u32 {
    2 + (ce[1] & 7) as u32
}

/// `vtd_ce_get_agaw()`.
fn ce_agaw(ce: &ContextEntry) -> u32 {
    30 + (ce[1] & 7) as u32 * 9
}

/// `VTDPASIDEntry`.
type PasidEntry = [u64; 8];

/// `VTD_SM_PASID_ENTRY_PGTT()`.
fn pe_pgtt(pe: &PasidEntry) -> u8 {
    ((pe[0] >> 6) & 7) as u8
}

/// `VTD_PE_GET_SS_LEVEL()`.
fn pe_ss_level(pe: &PasidEntry) -> u32 {
    2 + ((pe[0] >> 2) & 7) as u32
}

/// `VTD_PE_GET_FS_LEVEL()`: 4 or 5 level paging.
fn pe_fs_level(pe: &PasidEntry) -> u32 {
    4 + ((pe[2] >> 2) & 3) as u32
}

/// `vtd_get_pdire_from_pdir_table()`: the PASID directory entry of `pasid`, present or not.
fn pdire_get(m: &Dma<'_>, ce: &ContextEntry, pasid: u32) -> Result<u64, u8> {
    let addr = (ce[0] & !0xfff) + u64::from((pasid >> 6) & 0x3fff) * 8;
    m.read_quads::<1>(addr).map(|[q]| q).ok_or(FR_PASID_DIR_ACCESS_ERR)
}

/// The page table a walk for PASID 0 starts from, with its level, its address width and
/// the domain its IOTLB entries go to.
struct Stage {
    level: u32,
    agaw: u32,
    base: u64,
    did: u16,
}

thread_local! {
    /// Set while this thread holds the IOMMU state, so that an access that comes back to the
    /// IOMMU, such as a queue placed over its own registers, fails instead of deadlocking.
    static ENGAGED: Cell<bool> = const { Cell::new(false) };
}

struct Engaged {
    prev: bool,
}

impl Engaged {
    fn enter() -> Self {
        Engaged { prev: ENGAGED.with(|e| e.replace(true)) }
    }
}

impl Drop for Engaged {
    fn drop(&mut self) {
        ENGAGED.with(|e| e.set(self.prev));
    }
}

fn engaged() -> bool {
    ENGAGED.with(Cell::get)
}

/// What the IOMMU sends its own interrupts with, as (address, data).
type MsiSender = Box<dyn Fn(u64, u32) + Send + Sync>;

/// The `intel-iommu` device.
pub struct IntelIommu {
    state: Mutex<State>,
    memory: Weak<AddressSpace>,
    msi: OnceLock<MsiSender>,
}

impl fmt::Debug for IntelIommu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntelIommu").finish_non_exhaustive()
    }
}

impl IntelIommu {
    /// An IOMMU with the configuration that [`IntelIommuConf::realize`] gave, reading its
    /// tables from `memory`.
    pub fn new(conf: IntelIommuConf, memory: &Arc<AddressSpace>) -> Arc<Self> {
        Arc::new(IntelIommu {
            state: Mutex::new(State::new(conf)),
            memory: Arc::downgrade(memory),
            msi: OnceLock::new(),
        })
    }

    /// Sets what the fault event and invalidation completion interrupts go to. They are
    /// plain MSIs to the APICs, which interrupt remapping does not see.
    pub fn set_msi_sender(&self, send: impl Fn(u64, u32) + Send + Sync + 'static) {
        let _ = self.msi.set(Box::new(send));
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Runs `f` on the state, then sends the MSIs it asked for once the lock is dropped.
    fn with_state<R>(&self, f: impl FnOnce(&mut State, &Dma<'_>) -> R) -> R {
        let memory = self.memory.upgrade();
        let (r, msis) = {
            let _engaged = Engaged::enter();
            let mut s = self.lock();
            let r = f(&mut s, &Dma(memory.as_deref()));
            (r, std::mem::take(&mut s.msis))
        };
        if let Some(send) = self.msi.get() {
            for (addr, data) in msis {
                send(addr, data);
            }
        }
        r
    }

    /// `vtd_reset_exit()`.
    pub fn system_reset(&self) {
        self.lock().init();
    }

    /// The configuration, for the DMAR table.
    pub fn conf(&self) -> IntelIommuConf {
        self.lock().conf.clone()
    }

    /// The register frame, to map at [`DMAR_REG_BASE`].
    pub fn mmio_ops(self: &Arc<Self>) -> Arc<dyn MmioOps> {
        Arc::new(VtdMmio(Arc::clone(self)))
    }

    /// The IOMMU region callbacks of the device with source ID `sid`, which is
    /// `PCI_BUILD_BDF(bus, devfn)`.
    pub fn device_ops(self: &Arc<Self>, sid: u16) -> Arc<dyn IommuOps> {
        Arc::new(VtdDevice { iommu: Arc::clone(self), sid })
    }

    /// `vtd_mem_ir_write()`: remaps an MSI from the device with source ID `sid`, or from the
    /// IOAPIC with [`SID_INVALID`]. `None` drops the interrupt, after recording the fault.
    pub fn remap_msi(&self, sid: u16, addr: u64, data: u32) -> Option<(u64, u32)> {
        self.with_state(|s, m| s.interrupt_remap_msi(m, sid, addr, data, true))
    }

    /// `vtd_iommu_translate()`.
    pub fn translate(&self, sid: u16, addr: u64, flag: IommuAccessFlags) -> IommuTlbEntry {
        let mut entry = IommuTlbEntry {
            target_as: self.memory.upgrade(),
            iova: addr & PAGE_MASK_4K,
            translated_addr: addr & PAGE_MASK_4K,
            addr_mask: !PAGE_MASK_4K,
            perm: IommuAccessFlags::RW,
        };
        if engaged() {
            entry.perm = IommuAccessFlags::NONE;
            return entry;
        }
        if (INTERRUPT_ADDR_FIRST..=INTERRUPT_ADDR_LAST).contains(&addr) {
            return entry;
        }
        let is_write = flag.allows(IommuAccessFlags::WO);
        let x = self.with_state(|s, m| {
            if s.dmar_enabled { Some(s.do_translate(m, sid, addr, is_write)) } else { None }
        });
        if let Some(x) = x {
            entry.iova = x.iova;
            entry.translated_addr = x.translated;
            entry.addr_mask = x.mask;
            entry.perm = match x.perm & 3 {
                1 => IommuAccessFlags::RO,
                2 => IommuAccessFlags::WO,
                3 => IommuAccessFlags::RW,
                _ => IommuAccessFlags::NONE,
            };
        }
        entry
    }
}

/// The IOMMU region of one device, `VTDAddressSpace`.
struct VtdDevice {
    iommu: Arc<IntelIommu>,
    sid: u16,
}

impl IommuOps for VtdDevice {
    fn translate(&self, addr: u64, flag: IommuAccessFlags, _iommu_idx: u32) -> IommuTlbEntry {
        self.iommu.translate(self.sid, addr, flag)
    }
}

/// The register frame, `vtd_mem_ops`.
struct VtdMmio(Arc<IntelIommu>);

impl fmt::Debug for VtdMmio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("VtdMmio")
    }
}

impl MmioOps for VtdMmio {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        if engaged() {
            return Err(MemTxResult::ACCESS_ERROR);
        }
        match size.bytes() {
            n @ (4 | 8) => Ok(self.0.lock().mem_read(offset, n as u64)),
            _ => Err(MemTxResult::ERROR),
        }
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        if engaged() {
            return Err(MemTxResult::ACCESS_ERROR);
        }
        match size.bytes() {
            n @ (4 | 8) => self.0.with_state(|s, m| s.mem_write(m, offset, value, n as u64)),
            _ => return Err(MemTxResult::ERROR),
        }
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(4, 8)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(4, 8)
    }
}

#[cfg(test)]
mod tests;
