// SPDX-License-Identifier: GPL-2.0-or-later

//! The PCI Express capability and the extended capability helpers, from hw/pci/pcie.c, with the
//! small parts of hw/pci/pcie_aer.c that a root port needs.
//!
//! QEMU keeps the capability offsets in `PCIDevice::exp`. Here the model keeps them itself and
//! passes the offset of the PCI Express capability (`exp_cap`) to every helper. All helpers
//! work on the config arrays of a [`PciDevice`] through [`PciDevice::with_config`], so they
//! must not be called with the device lock held.
//!
//! Only what downstream ports use is ported: the version 2 capability, link and slot
//! registers, native hotplug, root control, ARI forwarding, device error reporting, ACS and an
//! AER capability that sets up its registers but never records errors. FLR, ATS, SR-IOV,
//! the endpoint link fill and the CXL flit mode bits are not here.

use ruvm_base::Error;

use crate::device::PciDevice;
use crate::regs::*;

// PCI Express capability registers, from include/standard-headers/linux/pci_regs.h.

/// `PCI_EXP_VER2_SIZEOF`.
pub const PCI_EXP_VER2_SIZEOF: u8 = 0x3c;
pub const PCI_EXP_FLAGS: usize = 0x02;
pub const PCI_EXP_FLAGS_VER2: u16 = 0x0002;
pub const PCI_EXP_FLAGS_TYPE: u16 = 0x00f0;
pub const PCI_EXP_FLAGS_TYPE_SHIFT: u32 = 4;
pub const PCI_EXP_FLAGS_SLOT: u16 = 0x0100;
pub const PCI_EXP_FLAGS_IRQ: u16 = 0x3e00;
pub const PCI_EXP_FLAGS_IRQ_SHIFT: u32 = 9;
pub const PCI_EXP_TYPE_ENDPOINT: u8 = 0x0;
pub const PCI_EXP_TYPE_LEG_END: u8 = 0x1;
pub const PCI_EXP_TYPE_ROOT_PORT: u8 = 0x4;
pub const PCI_EXP_TYPE_UPSTREAM: u8 = 0x5;
pub const PCI_EXP_TYPE_DOWNSTREAM: u8 = 0x6;
pub const PCI_EXP_TYPE_RC_END: u8 = 0x9;

pub const PCI_EXP_DEVCAP: usize = 0x04;
pub const PCI_EXP_DEVCAP_EXT_TAG: u32 = 0x0000_0020;
pub const PCI_EXP_DEVCAP_RBER: u32 = 0x0000_8000;
pub const PCI_EXP_DEVCTL: usize = 0x08;
pub const PCI_EXP_DEVCTL_CERE: u16 = 0x0001;
pub const PCI_EXP_DEVCTL_NFERE: u16 = 0x0002;
pub const PCI_EXP_DEVCTL_FERE: u16 = 0x0004;
pub const PCI_EXP_DEVCTL_URRE: u16 = 0x0008;
pub const PCI_EXP_DEVSTA: usize = 0x0a;
pub const PCI_EXP_DEVSTA_CED: u16 = 0x0001;
pub const PCI_EXP_DEVSTA_NFED: u16 = 0x0002;
pub const PCI_EXP_DEVSTA_FED: u16 = 0x0004;
pub const PCI_EXP_DEVSTA_URD: u16 = 0x0008;

pub const PCI_EXP_LNKCAP: usize = 0x0c;
pub const PCI_EXP_LNKCAP_SLS: u32 = 0x0000_000f;
pub const PCI_EXP_LNKCAP_MLW: u32 = 0x0000_03f0;
pub const PCI_EXP_LNKCAP_ASPMS_0S: u32 = 0x0000_0400;
pub const PCI_EXP_LNKCAP_DLLLARC: u32 = 0x0010_0000;
pub const PCI_EXP_LNKCAP_LBNC: u32 = 0x0020_0000;
pub const PCI_EXP_LNKCAP_PN_SHIFT: u32 = 24;
pub const PCI_EXP_LNKCTL: usize = 0x10;
pub const PCI_EXP_LNKSTA: usize = 0x12;
pub const PCI_EXP_LNKSTA_CLS: u16 = 0x000f;
pub const PCI_EXP_LNKSTA_NLW: u16 = 0x03f0;
pub const PCI_EXP_LNKSTA_DLLLA: u16 = 0x2000;

pub const PCI_EXP_SLTCAP: usize = 0x14;
pub const PCI_EXP_SLTCAP_ABP: u32 = 0x0000_0001;
pub const PCI_EXP_SLTCAP_PCP: u32 = 0x0000_0002;
pub const PCI_EXP_SLTCAP_MRLSP: u32 = 0x0000_0004;
pub const PCI_EXP_SLTCAP_AIP: u32 = 0x0000_0008;
pub const PCI_EXP_SLTCAP_PIP: u32 = 0x0000_0010;
pub const PCI_EXP_SLTCAP_HPS: u32 = 0x0000_0020;
pub const PCI_EXP_SLTCAP_HPC: u32 = 0x0000_0040;
pub const PCI_EXP_SLTCAP_EIP: u32 = 0x0002_0000;
pub const PCI_EXP_SLTCAP_NCCS: u32 = 0x0004_0000;
pub const PCI_EXP_SLTCAP_PSN: u32 = 0xfff8_0000;
pub const PCI_EXP_SLTCAP_PSN_SHIFT: u32 = 19;

pub const PCI_EXP_SLTCTL: usize = 0x18;
pub const PCI_EXP_SLTCTL_ABPE: u16 = 0x0001;
pub const PCI_EXP_SLTCTL_PFDE: u16 = 0x0002;
pub const PCI_EXP_SLTCTL_MRLSCE: u16 = 0x0004;
pub const PCI_EXP_SLTCTL_PDCE: u16 = 0x0008;
pub const PCI_EXP_SLTCTL_CCIE: u16 = 0x0010;
pub const PCI_EXP_SLTCTL_HPIE: u16 = 0x0020;
pub const PCI_EXP_SLTCTL_AIC: u16 = 0x00c0;
pub const PCI_EXP_SLTCTL_ATTN_IND_ON: u16 = 0x0040;
pub const PCI_EXP_SLTCTL_ATTN_IND_BLINK: u16 = 0x0080;
pub const PCI_EXP_SLTCTL_ATTN_IND_OFF: u16 = 0x00c0;
pub const PCI_EXP_SLTCTL_PIC: u16 = 0x0300;
pub const PCI_EXP_SLTCTL_PWR_IND_ON: u16 = 0x0100;
pub const PCI_EXP_SLTCTL_PWR_IND_BLINK: u16 = 0x0200;
pub const PCI_EXP_SLTCTL_PWR_IND_OFF: u16 = 0x0300;
pub const PCI_EXP_SLTCTL_PCC: u16 = 0x0400;
pub const PCI_EXP_SLTCTL_PWR_ON: u16 = 0x0000;
pub const PCI_EXP_SLTCTL_PWR_OFF: u16 = 0x0400;
pub const PCI_EXP_SLTCTL_EIC: u16 = 0x0800;

pub const PCI_EXP_SLTSTA: usize = 0x1a;
pub const PCI_EXP_SLTSTA_ABP: u16 = 0x0001;
pub const PCI_EXP_SLTSTA_PFD: u16 = 0x0002;
pub const PCI_EXP_SLTSTA_MRLSC: u16 = 0x0004;
pub const PCI_EXP_SLTSTA_PDC: u16 = 0x0008;
pub const PCI_EXP_SLTSTA_CC: u16 = 0x0010;
pub const PCI_EXP_SLTSTA_MRLSS: u16 = 0x0020;
pub const PCI_EXP_SLTSTA_PDS: u16 = 0x0040;
pub const PCI_EXP_SLTSTA_EIS: u16 = 0x0080;

/// The hotplug events a slot reports, `PCIExpressHotPlugEvent`.
pub const PCI_EXP_HP_EV_ABP: u16 = PCI_EXP_SLTCTL_ABPE;
pub const PCI_EXP_HP_EV_PDC: u16 = PCI_EXP_SLTCTL_PDCE;
pub const PCI_EXP_HP_EV_CCI: u16 = PCI_EXP_SLTCTL_CCIE;
pub const PCI_EXP_HP_EV_SUPPORTED: u16 = PCI_EXP_HP_EV_ABP | PCI_EXP_HP_EV_PDC | PCI_EXP_HP_EV_CCI;

pub const PCI_EXP_RTCTL: usize = 0x1c;
pub const PCI_EXP_RTCTL_SECEE: u16 = 0x0001;
pub const PCI_EXP_RTCTL_SENFEE: u16 = 0x0002;
pub const PCI_EXP_RTCTL_SEFEE: u16 = 0x0004;
pub const PCI_EXP_RTCTL_PMEIE: u16 = 0x0008;
pub const PCI_EXP_RTCAP: usize = 0x1e;
pub const PCI_EXP_RTSTA: usize = 0x20;
pub const PCI_EXP_RTSTA_PME: u32 = 0x0001_0000;
pub const PCI_EXP_RTSTA_PENDING: u32 = 0x0002_0000;

pub const PCI_EXP_DEVCAP2: usize = 0x24;
pub const PCI_EXP_DEVCAP2_ARI: u32 = 0x0000_0020;
pub const PCI_EXP_DEVCAP2_EFF: u32 = 0x0010_0000;
pub const PCI_EXP_DEVCAP2_EETLPP: u32 = 0x0020_0000;
pub const PCI_EXP_DEVCTL2: usize = 0x28;
pub const PCI_EXP_DEVCTL2_ARI: u16 = 0x0020;
pub const PCI_EXP_DEVCTL2_EETLPPB: u16 = 0x8000;
pub const PCI_EXP_LNKCAP2: usize = 0x2c;
pub const PCI_EXP_LNKCAP2_SLS_2_5GB: u32 = 0x0000_0002;
pub const PCI_EXP_LNKCAP2_SLS_5_0GB: u32 = 0x0000_0004;
pub const PCI_EXP_LNKCAP2_SLS_8_0GB: u32 = 0x0000_0008;
pub const PCI_EXP_LNKCAP2_SLS_16_0GB: u32 = 0x0000_0010;
pub const PCI_EXP_LNKCAP2_SLS_32_0GB: u32 = 0x0000_0020;
pub const PCI_EXP_LNKCAP2_SLS_64_0GB: u32 = 0x0000_0040;
pub const PCI_EXP_LNKCTL2: usize = 0x30;
pub const PCI_EXP_LNKCTL2_TLS: u16 = 0x000f;
pub const PCI_EXP_LNKSTA2: usize = 0x32;

// Extended capabilities.

/// `PCI_EXT_CAP_ID_ERR`, advanced error reporting.
pub const PCI_EXT_CAP_ID_ERR: u16 = 0x01;
/// `PCI_EXT_CAP_ID_ACS`, access control services.
pub const PCI_EXT_CAP_ID_ACS: u16 = 0x0d;
/// `PCI_EXT_CAP_ID_ARI`.
pub const PCI_EXT_CAP_ID_ARI: u16 = 0x0e;
/// `PCI_EXT_CAP_ALIGN`.
pub const PCI_EXT_CAP_ALIGN: u16 = 4;

/// `PCI_EXT_CAP()`: an extended capability header.
pub const fn pci_ext_cap(id: u16, ver: u8, next: u16) -> u32 {
    id as u32 | (ver as u32) << 16 | (next as u32) << 20
}

/// `PCI_EXT_CAP_ID()`.
pub const fn pci_ext_cap_id(header: u32) -> u16 {
    (header & 0xffff) as u16
}

/// `PCI_EXT_CAP_VER()`.
pub const fn pci_ext_cap_ver(header: u32) -> u8 {
    ((header >> 16) & 0xf) as u8
}

/// `PCI_EXT_CAP_NEXT()`.
pub const fn pci_ext_cap_next(header: u32) -> u16 {
    ((header >> 20) & 0xffc) as u16
}

// AER, from include/hw/pci/pcie_regs.h and pci_regs.h.

pub const PCI_ERR_VER: u8 = 2;
pub const PCI_ERR_SIZEOF: u16 = 0x48;
pub const PCI_ERR_UNCOR_STATUS: usize = 0x04;
pub const PCI_ERR_UNCOR_MASK: usize = 0x08;
pub const PCI_ERR_UNCOR_SEVER: usize = 0x0c;
pub const PCI_ERR_COR_STATUS: usize = 0x10;
pub const PCI_ERR_COR_MASK: usize = 0x14;
pub const PCI_ERR_CAP: usize = 0x18;
pub const PCI_ERR_CAP_FEP_MASK: u32 = 0x0000_001f;
pub const PCI_ERR_CAP_ECRC_GENC: u32 = 0x0000_0020;
pub const PCI_ERR_CAP_ECRC_GENE: u32 = 0x0000_0040;
pub const PCI_ERR_CAP_ECRC_CHKC: u32 = 0x0000_0080;
pub const PCI_ERR_CAP_ECRC_CHKE: u32 = 0x0000_0100;
pub const PCI_ERR_CAP_MHRC: u32 = 0x0000_0200;
pub const PCI_ERR_CAP_MHRE: u32 = 0x0000_0400;
pub const PCI_ERR_CAP_TLP: u32 = 0x0000_0800;
pub const PCI_ERR_HEADER_LOG: usize = 0x1c;
pub const PCI_ERR_HEADER_LOG_SIZE: usize = 16;
pub const PCI_ERR_ROOT_COMMAND: usize = 0x2c;
pub const PCI_ERR_ROOT_CMD_COR_EN: u32 = 0x0000_0001;
pub const PCI_ERR_ROOT_CMD_NONFATAL_EN: u32 = 0x0000_0002;
pub const PCI_ERR_ROOT_CMD_FATAL_EN: u32 = 0x0000_0004;
pub const PCI_ERR_ROOT_CMD_EN_MASK: u32 = 0x0000_0007;
pub const PCI_ERR_ROOT_STATUS: usize = 0x30;
pub const PCI_ERR_ROOT_COR_RCV: u32 = 0x0000_0001;
pub const PCI_ERR_ROOT_NONFATAL_RCV: u32 = 0x0000_0020;
pub const PCI_ERR_ROOT_FATAL_RCV: u32 = 0x0000_0040;
pub const PCI_ERR_ROOT_STATUS_REPORT_MASK: u32 = 0x0000_007f;
pub const PCI_ERR_ROOT_IRQ: u32 = 0xf800_0000;
pub const PCI_ERR_ROOT_IRQ_SHIFT: u32 = 27;
pub const PCI_ERR_ROOT_ERR_SRC: usize = 0x34;
pub const PCI_ERR_TLP_PREFIX_LOG: usize = 0x38;
pub const PCI_ERR_TLP_PREFIX_LOG_SIZE: usize = 16;

/// `PCI_ERR_UNC_SUPPORTED`: the uncorrectable errors QEMU knows about.
pub const PCI_ERR_UNC_SUPPORTED: u32 = 0x0000_0010 // DLP
    | 0x0000_0020 // SDES
    | 0x0000_1000 // POISON_TLP
    | 0x0000_2000 // FCP
    | 0x0000_4000 // COMP_TIME
    | 0x0000_8000 // COMP_ABORT
    | 0x0001_0000 // UNX_COMP
    | 0x0002_0000 // RX_OVER
    | 0x0004_0000 // MALF_TLP
    | 0x0008_0000 // ECRC
    | 0x0010_0000 // UNSUP
    | 0x0020_0000 // ACSV
    | 0x0040_0000 // INTN
    | 0x0080_0000 // MCBTLP
    | 0x0100_0000 // ATOP_EBLOCKED
    | 0x0200_0000; // TLP_PRF_BLOCKED
/// `PCI_ERR_UNC_MASK_DEFAULT`: internal errors and blocked TLP prefixes.
pub const PCI_ERR_UNC_MASK_DEFAULT: u32 = 0x0040_0000 | 0x0200_0000;
/// `PCI_ERR_UNC_SEVERITY_DEFAULT`.
pub const PCI_ERR_UNC_SEVERITY_DEFAULT: u32 =
    0x0000_0010 | 0x0000_0020 | 0x0000_2000 | 0x0002_0000 | 0x0004_0000 | 0x0040_0000;
/// `PCI_ERR_COR_SUPPORTED`.
pub const PCI_ERR_COR_SUPPORTED: u32 =
    0x0001 | 0x0040 | 0x0080 | 0x0100 | 0x1000 | 0x2000 | 0x4000 | 0x8000;
/// `PCI_ERR_COR_MASK_DEFAULT`: advisory non-fatal, internal and header log overflow.
pub const PCI_ERR_COR_MASK_DEFAULT: u32 = 0x2000 | 0x4000 | 0x8000;
/// `PCIE_AER_LOG_MAX_LIMIT`.
pub const PCIE_AER_LOG_MAX_LIMIT: u16 = 128;
/// `PCIE_AER_LOG_MAX_DEFAULT`.
pub const PCIE_AER_LOG_MAX_DEFAULT: u16 = 8;

/// `PCI_SEC_STATUS_RCV_SYSTEM_ERROR`.
pub const PCI_SEC_STATUS_RCV_SYSTEM_ERROR: u16 = 0x4000;

// ACS.

pub const PCI_ACS_VER: u8 = 1;
pub const PCI_ACS_SIZEOF: u16 = 8;
pub const PCI_ACS_CAP: usize = 0x04;
pub const PCI_ACS_CTRL: usize = 0x06;
pub const PCI_ACS_SV: u16 = 0x01;
pub const PCI_ACS_TB: u16 = 0x02;
pub const PCI_ACS_RR: u16 = 0x04;
pub const PCI_ACS_CR: u16 = 0x08;
pub const PCI_ACS_UF: u16 = 0x10;
pub const PCI_ACS_EC: u16 = 0x20;
pub const PCI_ACS_DT: u16 = 0x40;

/// Link speeds, `PCIExpLinkSpeed`. The discriminant is the encoding used in the link
/// registers.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum PcieLinkSpeed {
    Gt2_5 = 1,
    Gt5 = 2,
    Gt8 = 3,
    Gt16 = 4,
    Gt32 = 5,
    Gt64 = 6,
}

impl PcieLinkSpeed {
    /// Parses the `x-speed` property values: "2_5", "5", "8", "16", "32" and "64".
    pub fn from_prop(s: &str) -> Option<PcieLinkSpeed> {
        Some(match s {
            "2_5" => PcieLinkSpeed::Gt2_5,
            "5" => PcieLinkSpeed::Gt5,
            "8" => PcieLinkSpeed::Gt8,
            "16" => PcieLinkSpeed::Gt16,
            "32" => PcieLinkSpeed::Gt32,
            "64" => PcieLinkSpeed::Gt64,
            _ => return None,
        })
    }

    /// The register encoding.
    pub fn bits(self) -> u8 {
        self as u8
    }
}

/// Link widths, `PCIExpLinkWidth`. The discriminant is the lane count.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum PcieLinkWidth {
    X1 = 1,
    X2 = 2,
    X4 = 4,
    X8 = 8,
    X12 = 12,
    X16 = 16,
    X32 = 32,
}

impl PcieLinkWidth {
    /// Parses the `x-width` property values: "1", "2", "4", "8", "12", "16" and "32".
    pub fn from_prop(s: &str) -> Option<PcieLinkWidth> {
        Some(match s {
            "1" => PcieLinkWidth::X1,
            "2" => PcieLinkWidth::X2,
            "4" => PcieLinkWidth::X4,
            "8" => PcieLinkWidth::X8,
            "12" => PcieLinkWidth::X12,
            "16" => PcieLinkWidth::X16,
            "32" => PcieLinkWidth::X32,
            _ => return None,
        })
    }

    /// The lane count.
    pub fn lanes(self) -> u8 {
        self as u8
    }
}

/// `QEMU_PCI_EXP_LNKCAP_MLW()`.
fn lnkcap_mlw(width: PcieLinkWidth) -> u32 {
    u32::from(width.lanes()) << 4
}

/// `QEMU_PCI_EXP_LNKSTA_NLW()`.
fn lnksta_nlw(width: PcieLinkWidth) -> u16 {
    u16::from(width.lanes()) << 4
}

fn set_word_mask(b: &mut [u8], off: usize, mask: u16) -> u16 {
    let v = pci_get_word(b, off);
    pci_set_word(b, off, v | mask);
    v & mask
}

fn clear_word_mask(b: &mut [u8], off: usize, mask: u16) -> u16 {
    let v = pci_get_word(b, off);
    pci_set_word(b, off, v & !mask);
    v & mask
}

fn set_long_mask(b: &mut [u8], off: usize, mask: u32) {
    let v = pci_get_long(b, off);
    pci_set_long(b, off, v | mask);
}

fn clear_long_mask(b: &mut [u8], off: usize, mask: u32) {
    let v = pci_get_long(b, off);
    pci_set_long(b, off, v & !mask);
}

/// `pcie_cap_init()`: adds a version 2 PCI Express capability of port type `type_` and fills
/// the parts that do not depend on the port model. Returns the offset of the capability.
pub fn pcie_cap_init(dev: &PciDevice, offset: u8, type_: u8, port: u8) -> Result<u8, Error> {
    assert!(dev.is_express(), "{} is not a PCI Express function", dev.name());
    let pos = dev.add_capability(PCI_CAP_ID_EXP, offset, PCI_EXP_VER2_SIZEOF)?;
    let p = usize::from(pos);
    dev.with_config(|c| {
        // pcie_cap_v1_fill(). QEMU_PCIE_EXT_TAG is on by default for every function.
        let flags = ((u16::from(type_) << PCI_EXP_FLAGS_TYPE_SHIFT) & PCI_EXP_FLAGS_TYPE)
            | PCI_EXP_FLAGS_VER2;
        pci_set_word(c.config, p + PCI_EXP_FLAGS, flags);
        pci_set_long(c.config, p + PCI_EXP_DEVCAP, PCI_EXP_DEVCAP_RBER | PCI_EXP_DEVCAP_EXT_TAG);
        pci_set_long(
            c.config,
            p + PCI_EXP_LNKCAP,
            (u32::from(port) << PCI_EXP_LNKCAP_PN_SHIFT)
                | PCI_EXP_LNKCAP_ASPMS_0S
                | lnkcap_mlw(PcieLinkWidth::X1)
                | u32::from(PcieLinkSpeed::Gt2_5.bits()),
        );
        pci_set_word(
            c.config,
            p + PCI_EXP_LNKSTA,
            lnksta_nlw(PcieLinkWidth::X1) | u16::from(PcieLinkSpeed::Gt2_5.bits()),
        );
        // Link status changes over time, so it is not compared on migration.
        pci_set_word(c.cmask, p + PCI_EXP_LNKSTA, 0);

        pci_set_long(c.config, p + PCI_EXP_DEVCAP2, PCI_EXP_DEVCAP2_EFF | PCI_EXP_DEVCAP2_EETLPP);
        pci_set_word(c.wmask, p + PCI_EXP_DEVCTL2, PCI_EXP_DEVCTL2_EETLPPB);
        // Read-only, so that it behaves like a null extended capability header.
        pci_set_long(c.wmask, PCI_CONFIG_SPACE_SIZE, 0);
    });
    Ok(pos)
}

/// `pcie_cap_fill_slot_lnk()` and `pcie_cap_fill_lnk()`: advertises the configured link width
/// and speed of a slot.
pub fn pcie_cap_fill_slot_lnk(
    dev: &PciDevice,
    exp_cap: u8,
    width: PcieLinkWidth,
    speed: PcieLinkSpeed,
) {
    let p = usize::from(exp_cap);
    dev.with_config(|c| {
        let cfg = c.config;
        // Link bandwidth notification is required for ports wider than x1 or with more than
        // one speed.
        if width > PcieLinkWidth::X1 || speed > PcieLinkSpeed::Gt2_5 {
            set_long_mask(cfg, p + PCI_EXP_LNKCAP, PCI_EXP_LNKCAP_LBNC);
        }
        if speed > PcieLinkSpeed::Gt2_5 {
            // Ports faster than 5GT/s must hardwire DLLLARC; the hotplug code sets DLLLA.
            set_long_mask(cfg, p + PCI_EXP_LNKCAP, PCI_EXP_LNKCAP_DLLLARC);
        }

        clear_long_mask(cfg, p + PCI_EXP_LNKCAP, PCI_EXP_LNKCAP_MLW | PCI_EXP_LNKCAP_SLS);
        set_long_mask(cfg, p + PCI_EXP_LNKCAP, lnkcap_mlw(width) | u32::from(speed.bits()));

        if speed > PcieLinkSpeed::Gt2_5 {
            // The target link speed defaults to the fastest one supported.
            clear_word_mask(cfg, p + PCI_EXP_LNKCTL2, PCI_EXP_LNKCTL2_TLS);
            set_word_mask(cfg, p + PCI_EXP_LNKCTL2, u16::from(speed.bits()) & PCI_EXP_LNKCTL2_TLS);
        }

        // Up to 5GT/s LNKCAP says it all. Above that LNKCAP only points at the highest bit of
        // the supported speeds vector, and every lower speed is assumed to work.
        if speed > PcieLinkSpeed::Gt5 {
            let mut v = PCI_EXP_LNKCAP2_SLS_2_5GB | PCI_EXP_LNKCAP2_SLS_5_0GB;
            v |= PCI_EXP_LNKCAP2_SLS_8_0GB;
            if speed > PcieLinkSpeed::Gt8 {
                v |= PCI_EXP_LNKCAP2_SLS_16_0GB;
            }
            if speed > PcieLinkSpeed::Gt16 {
                v |= PCI_EXP_LNKCAP2_SLS_32_0GB;
            }
            if speed > PcieLinkSpeed::Gt32 {
                v |= PCI_EXP_LNKCAP2_SLS_64_0GB;
            }
            pci_set_long(cfg, p + PCI_EXP_LNKCAP2, v);
        }
    });
}

/// `pcie_cap_get_type()`.
pub fn pcie_cap_get_type(dev: &PciDevice, exp_cap: u8) -> u8 {
    let flags = pci_get_word(&dev.config_bytes(), usize::from(exp_cap) + PCI_EXP_FLAGS);
    ((flags & PCI_EXP_FLAGS_TYPE) >> PCI_EXP_FLAGS_TYPE_SHIFT) as u8
}

/// `pcie_cap_flags_get_vector()`: the MSI or MSI-X vector used for hotplug and PME.
pub fn pcie_cap_flags_get_vector(dev: &PciDevice, exp_cap: u8) -> u32 {
    let flags = pci_get_word(&dev.config_bytes(), usize::from(exp_cap) + PCI_EXP_FLAGS);
    u32::from((flags & PCI_EXP_FLAGS_IRQ) >> PCI_EXP_FLAGS_IRQ_SHIFT)
}

/// `pcie_cap_deverr_init()`: error reporting enables and the matching status bits.
pub fn pcie_cap_deverr_init(dev: &PciDevice, exp_cap: u8) {
    let p = usize::from(exp_cap);
    dev.with_config(|c| {
        set_long_mask(c.config, p + PCI_EXP_DEVCAP, PCI_EXP_DEVCAP_RBER);
        set_long_mask(
            c.wmask,
            p + PCI_EXP_DEVCTL,
            u32::from(
                PCI_EXP_DEVCTL_CERE
                    | PCI_EXP_DEVCTL_NFERE
                    | PCI_EXP_DEVCTL_FERE
                    | PCI_EXP_DEVCTL_URRE,
            ),
        );
        set_long_mask(
            c.w1cmask,
            p + PCI_EXP_DEVSTA,
            u32::from(
                PCI_EXP_DEVSTA_CED | PCI_EXP_DEVSTA_NFED | PCI_EXP_DEVSTA_FED | PCI_EXP_DEVSTA_URD,
            ),
        );
    });
}

/// `pcie_cap_deverr_reset()`.
pub fn pcie_cap_deverr_reset(dev: &PciDevice, exp_cap: u8) {
    let p = usize::from(exp_cap);
    dev.with_config(|c| {
        clear_long_mask(
            c.config,
            p + PCI_EXP_DEVCTL,
            u32::from(
                PCI_EXP_DEVCTL_CERE
                    | PCI_EXP_DEVCTL_NFERE
                    | PCI_EXP_DEVCTL_FERE
                    | PCI_EXP_DEVCTL_URRE,
            ),
        );
    });
}

/// `pcie_cap_arifwd_init()`: ARI forwarding, for ports whose secondary side has ARI devices.
pub fn pcie_cap_arifwd_init(dev: &PciDevice, exp_cap: u8) {
    let p = usize::from(exp_cap);
    dev.with_config(|c| {
        set_long_mask(c.config, p + PCI_EXP_DEVCAP2, PCI_EXP_DEVCAP2_ARI);
        set_long_mask(c.wmask, p + PCI_EXP_DEVCTL2, u32::from(PCI_EXP_DEVCTL2_ARI));
    });
}

/// `pcie_cap_arifwd_reset()`.
pub fn pcie_cap_arifwd_reset(dev: &PciDevice, exp_cap: u8) {
    let p = usize::from(exp_cap);
    dev.with_config(|c| {
        clear_long_mask(c.config, p + PCI_EXP_DEVCTL2, u32::from(PCI_EXP_DEVCTL2_ARI));
    });
}

/// `pcie_cap_is_arifwd_enabled()`.
pub fn pcie_cap_is_arifwd_enabled(dev: &PciDevice, exp_cap: u8) -> bool {
    let cfg = dev.config_bytes();
    pci_get_long(&cfg, usize::from(exp_cap) + PCI_EXP_DEVCTL2) & u32::from(PCI_EXP_DEVCTL2_ARI) != 0
}

/// `pcie_cap_root_init()`: system error enables in the root control register.
pub fn pcie_cap_root_init(dev: &PciDevice, exp_cap: u8) {
    let p = usize::from(exp_cap);
    dev.with_config(|c| {
        pci_set_word(
            c.wmask,
            p + PCI_EXP_RTCTL,
            PCI_EXP_RTCTL_SECEE | PCI_EXP_RTCTL_SENFEE | PCI_EXP_RTCTL_SEFEE,
        );
    });
}

/// `pcie_cap_root_reset()`.
pub fn pcie_cap_root_reset(dev: &PciDevice, exp_cap: u8) {
    let p = usize::from(exp_cap);
    dev.with_config(|c| pci_set_word(c.config, p + PCI_EXP_RTCTL, 0));
}

/// `pcie_sltctl_powered_off()`: the power controller and the power indicator are both off.
pub fn pcie_sltctl_powered_off(sltctl: u16) -> bool {
    sltctl & PCI_EXP_SLTCTL_PCC == PCI_EXP_SLTCTL_PWR_OFF
        && sltctl & PCI_EXP_SLTCTL_PIC == PCI_EXP_SLTCTL_PWR_IND_OFF
}

/// The slot settings `pcie_cap_slot_init()` looks at, from `PCIESlot`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PcieSlotParams {
    /// The physical slot number, `slot`.
    pub slot: u16,
    /// `hotplug`.
    pub hotplug: bool,
    /// `x-do-not-expose-native-hotplug-cap`, the 6.1 compat knob.
    pub hide_native_hotplug_cap: bool,
    /// Whether the port itself was hot-plugged.
    pub hotplugged: bool,
    /// `power_controller_present`, `QEMU_PCIE_SLTCAP_PCP`.
    pub power_controller_present: bool,
}

/// `pcie_cap_slot_init()`: the slot registers of a downstream port with native hotplug.
pub fn pcie_cap_slot_init(dev: &PciDevice, exp_cap: u8, s: &PcieSlotParams) {
    let p = usize::from(exp_cap);
    dev.with_config(|c| {
        set_word_mask(c.config, p + PCI_EXP_FLAGS, PCI_EXP_FLAGS_SLOT);

        clear_long_mask(c.config, p + PCI_EXP_SLTCAP, !PCI_EXP_SLTCAP_PSN);
        set_long_mask(
            c.config,
            p + PCI_EXP_SLTCAP,
            (u32::from(s.slot) << PCI_EXP_SLTCAP_PSN_SHIFT)
                | PCI_EXP_SLTCAP_EIP
                | PCI_EXP_SLTCAP_PIP
                | PCI_EXP_SLTCAP_AIP
                | PCI_EXP_SLTCAP_ABP,
        );

        // Native hotplug is exposed whenever hotplug is on, unless the broken 6.1 ABI is asked
        // for.
        if s.hotplug && (!s.hide_native_hotplug_cap || s.hotplugged) {
            set_long_mask(c.config, p + PCI_EXP_SLTCAP, PCI_EXP_SLTCAP_HPS | PCI_EXP_SLTCAP_HPC);
        }

        if s.power_controller_present {
            set_long_mask(c.config, p + PCI_EXP_SLTCAP, PCI_EXP_SLTCAP_PCP);
            clear_word_mask(c.config, p + PCI_EXP_SLTCTL, PCI_EXP_SLTCTL_PCC);
            set_word_mask(c.wmask, p + PCI_EXP_SLTCTL, PCI_EXP_SLTCTL_PCC);
        }

        clear_word_mask(c.config, p + PCI_EXP_SLTCTL, PCI_EXP_SLTCTL_PIC | PCI_EXP_SLTCTL_AIC);
        set_word_mask(
            c.config,
            p + PCI_EXP_SLTCTL,
            PCI_EXP_SLTCTL_PWR_IND_OFF | PCI_EXP_SLTCTL_ATTN_IND_OFF,
        );
        set_word_mask(
            c.wmask,
            p + PCI_EXP_SLTCTL,
            PCI_EXP_SLTCTL_PIC
                | PCI_EXP_SLTCTL_AIC
                | PCI_EXP_SLTCTL_HPIE
                | PCI_EXP_SLTCTL_CCIE
                | PCI_EXP_SLTCTL_PDCE
                | PCI_EXP_SLTCTL_ABPE,
        );
        // EIC always reads as 0, but it has to be writable so a write of 1 can be seen.
        set_word_mask(c.wmask, p + PCI_EXP_SLTCTL, PCI_EXP_SLTCTL_EIC);

        set_word_mask(c.w1cmask, p + PCI_EXP_SLTSTA, PCI_EXP_HP_EV_SUPPORTED);
        // Presence changes when the guest removes the device, so do not compare it.
        clear_word_mask(c.cmask, p + PCI_EXP_SLTSTA, PCI_EXP_SLTSTA_PDS);
    });
}

/// `pcie_cap_slot_get()`: the slot control and slot status registers.
pub fn pcie_cap_slot_get(dev: &PciDevice, exp_cap: u8) -> (u16, u16) {
    let cfg = dev.config_bytes();
    let p = usize::from(exp_cap);
    (pci_get_word(&cfg, p + PCI_EXP_SLTCTL), pci_get_word(&cfg, p + PCI_EXP_SLTSTA))
}

/// `pcie_add_capability()`: adds an extended capability at `offset`, which must be at least
/// 0x100, and links it at the end of the list. The bytes become read-only.
pub fn pcie_add_capability(dev: &PciDevice, cap_id: u16, cap_ver: u8, offset: u16, size: u16) {
    let (o, sz) = (usize::from(offset), usize::from(size));
    assert!(o >= PCI_CONFIG_SPACE_SIZE, "extended capability below 0x100");
    assert!(size >= 8 && o + sz <= PCIE_CONFIG_SPACE_SIZE, "extended capability out of range");
    assert!(dev.is_express(), "{} is not a PCI Express function", dev.name());

    if o != PCI_CONFIG_SPACE_SIZE {
        // 0xffffffff never matches a 16 bit ID, so this finds the tail of the list.
        let (_, prev) = pcie_find_capability_list(&dev.config_bytes(), 0xffff_ffff);
        assert!(usize::from(prev) >= PCI_CONFIG_SPACE_SIZE, "extended capability list is empty");
        pcie_ext_cap_set_next(dev, prev, offset);
    }
    dev.with_config(|c| {
        pci_set_long(c.config, o, pci_ext_cap(cap_id, cap_ver, 0));
        c.wmask[o..o + sz].fill(0);
        c.w1cmask[o..o + sz].fill(0);
        c.cmask[o..o + sz].fill(0xff);
    });
}

/// `pcie_ext_cap_set_next()`.
fn pcie_ext_cap_set_next(dev: &PciDevice, pos: u16, next: u16) {
    assert!(next % PCI_EXT_CAP_ALIGN == 0, "misaligned extended capability");
    let p = usize::from(pos);
    dev.with_config(|c| {
        let header = pci_get_long(c.config, p);
        pci_set_long(
            c.config,
            p,
            pci_ext_cap(pci_ext_cap_id(header), pci_ext_cap_ver(header), next),
        );
    });
}

/// `pcie_find_capability_list()`: the offset of `cap_id` (0 when absent) and of the capability
/// before it. The walk is bounded like the one for normal capabilities.
fn pcie_find_capability_list(cfg: &[u8], cap_id: u32) -> (u16, u16) {
    let mut header = pci_get_long(cfg, PCI_CONFIG_SPACE_SIZE);
    if header == 0 {
        return (0, 0);
    }
    let mut prev = 0u16;
    let mut next = PCI_CONFIG_SPACE_SIZE as u16;
    // At most (4096 - 256) / 4 capabilities fit.
    for _ in 0..960 {
        if next == 0 || usize::from(next) > PCIE_CONFIG_SPACE_SIZE - 8 {
            return (0, prev);
        }
        header = pci_get_long(cfg, usize::from(next));
        if u32::from(pci_ext_cap_id(header)) == cap_id {
            return (next, prev);
        }
        prev = next;
        next = pci_ext_cap_next(header);
    }
    (0, prev)
}

/// `pcie_find_capability()`: the offset of extended capability `cap_id`, or 0.
pub fn pcie_find_capability(dev: &PciDevice, cap_id: u16) -> u16 {
    pcie_find_capability_list(&dev.config_bytes(), u32::from(cap_id)).0
}

/// `pcie_acs_init()` for a downstream port: source validation, translation blocking, request
/// and completion redirect, upstream forwarding and direct translated P2P, all writable.
pub fn pcie_acs_init(dev: &PciDevice, offset: u16) {
    pcie_add_capability(dev, PCI_EXT_CAP_ID_ACS, PCI_ACS_VER, offset, PCI_ACS_SIZEOF);
    let bits = PCI_ACS_SV | PCI_ACS_TB | PCI_ACS_RR | PCI_ACS_CR | PCI_ACS_UF | PCI_ACS_DT;
    let o = usize::from(offset);
    dev.with_config(|c| {
        pci_set_word(c.config, o + PCI_ACS_CAP, bits);
        pci_set_word(c.wmask, o + PCI_ACS_CTRL, bits);
    });
}

/// `pcie_acs_reset()`.
pub fn pcie_acs_reset(dev: &PciDevice, offset: u16) {
    let o = usize::from(offset);
    dev.with_config(|c| pci_set_word(c.config, o + PCI_ACS_CTRL, 0));
}

/// `pcie_aer_init()` for a port: the AER registers with QEMU's defaults.
///
/// This is a register level stub. Errors are never injected or logged, so the status
/// registers only change when the guest clears them and the header log stays empty.
pub fn pcie_aer_init(dev: &PciDevice, offset: u16, log_max: u16) -> Result<(), Error> {
    pcie_add_capability(dev, PCI_EXT_CAP_ID_ERR, PCI_ERR_VER, offset, PCI_ERR_SIZEOF);
    if log_max > PCIE_AER_LOG_MAX_LIMIT {
        return Err(Error::generic(format!(
            "Invalid aer_log_max {log_max}. The max number of aer log is {PCIE_AER_LOG_MAX_LIMIT}"
        )));
    }
    let o = usize::from(offset);
    dev.with_config(|c| {
        pci_set_long(c.w1cmask, o + PCI_ERR_UNCOR_STATUS, PCI_ERR_UNC_SUPPORTED);
        // x-pcie-err-unc-mask is on by default.
        pci_set_long(c.config, o + PCI_ERR_UNCOR_MASK, PCI_ERR_UNC_MASK_DEFAULT);
        pci_set_long(c.wmask, o + PCI_ERR_UNCOR_MASK, PCI_ERR_UNC_SUPPORTED);
        pci_set_long(c.config, o + PCI_ERR_UNCOR_SEVER, PCI_ERR_UNC_SEVERITY_DEFAULT);
        pci_set_long(c.wmask, o + PCI_ERR_UNCOR_SEVER, PCI_ERR_UNC_SUPPORTED);
        set_long_mask(c.w1cmask, o + PCI_ERR_COR_STATUS, PCI_ERR_COR_SUPPORTED);
        pci_set_long(c.config, o + PCI_ERR_COR_MASK, PCI_ERR_COR_MASK_DEFAULT);
        pci_set_long(c.wmask, o + PCI_ERR_COR_MASK, PCI_ERR_COR_SUPPORTED);

        if log_max > 0 {
            pci_set_long(
                c.config,
                o + PCI_ERR_CAP,
                PCI_ERR_CAP_ECRC_GENC | PCI_ERR_CAP_ECRC_CHKC | PCI_ERR_CAP_MHRC,
            );
            pci_set_long(
                c.wmask,
                o + PCI_ERR_CAP,
                PCI_ERR_CAP_ECRC_GENE | PCI_ERR_CAP_ECRC_CHKE | PCI_ERR_CAP_MHRE,
            );
        } else {
            pci_set_long(c.config, o + PCI_ERR_CAP, PCI_ERR_CAP_ECRC_GENC | PCI_ERR_CAP_ECRC_CHKC);
            pci_set_long(c.wmask, o + PCI_ERR_CAP, PCI_ERR_CAP_ECRC_GENE | PCI_ERR_CAP_ECRC_CHKE);
        }

        // Every port type: SERR forwarding and the received system error status bit. QEMU
        // sets this mask with a 32 bit access at PCI_STATUS, which only reaches the status.
        set_word_mask(c.wmask, PCI_BRIDGE_CONTROL, PCI_BRIDGE_CTL_SERR);
        set_long_mask(c.w1cmask, PCI_STATUS, u32::from(PCI_SEC_STATUS_RCV_SYSTEM_ERROR));
    });
    Ok(())
}

/// `pcie_aer_write_config()`. With no error ever recorded the first error pointer never names
/// a set status bit, so this always ends in `pcie_aer_clear_log()`.
pub fn pcie_aer_write_config(dev: &PciDevice, offset: u16) {
    let o = usize::from(offset);
    dev.with_config(|c| {
        let errcap = pci_get_long(c.config, o + PCI_ERR_CAP);
        let first_error = 1u32 << (errcap & PCI_ERR_CAP_FEP_MASK);
        let uncorsta = pci_get_long(c.config, o + PCI_ERR_UNCOR_STATUS);
        if uncorsta & first_error == 0 {
            clear_long_mask(c.config, o + PCI_ERR_CAP, PCI_ERR_CAP_FEP_MASK | PCI_ERR_CAP_TLP);
            c.config[o + PCI_ERR_HEADER_LOG..o + PCI_ERR_HEADER_LOG + PCI_ERR_HEADER_LOG_SIZE]
                .fill(0);
            c.config[o + PCI_ERR_TLP_PREFIX_LOG
                ..o + PCI_ERR_TLP_PREFIX_LOG + PCI_ERR_TLP_PREFIX_LOG_SIZE]
                .fill(0);
        }
    });
}

/// `pcie_aer_root_init()`.
pub fn pcie_aer_root_init(dev: &PciDevice, offset: u16) {
    let o = usize::from(offset);
    dev.with_config(|c| {
        pci_set_long(c.wmask, o + PCI_ERR_ROOT_COMMAND, PCI_ERR_ROOT_CMD_EN_MASK);
        pci_set_long(c.w1cmask, o + PCI_ERR_ROOT_STATUS, PCI_ERR_ROOT_STATUS_REPORT_MASK);
        // The interrupt message number is read-only but set by the model.
        pci_set_long(c.cmask, o + PCI_ERR_ROOT_STATUS, !PCI_ERR_ROOT_IRQ);
    });
}

/// `pcie_aer_root_reset()`.
pub fn pcie_aer_root_reset(dev: &PciDevice, offset: u16) {
    let o = usize::from(offset);
    dev.with_config(|c| pci_set_long(c.config, o + PCI_ERR_ROOT_COMMAND, 0));
}

/// `pcie_aer_root_set_vector()`.
pub fn pcie_aer_root_set_vector(dev: &PciDevice, offset: u16, vector: u32) {
    assert!(vector < 32, "AER vector out of range");
    let o = usize::from(offset);
    dev.with_config(|c| {
        clear_long_mask(c.config, o + PCI_ERR_ROOT_STATUS, PCI_ERR_ROOT_IRQ);
        set_long_mask(c.config, o + PCI_ERR_ROOT_STATUS, vector << PCI_ERR_ROOT_IRQ_SHIFT);
    });
}

/// `pcie_aer_status_to_cmd()`.
fn pcie_aer_status_to_cmd(status: u32) -> u32 {
    let mut cmd = 0;
    if status & PCI_ERR_ROOT_COR_RCV != 0 {
        cmd |= PCI_ERR_ROOT_CMD_COR_EN;
    }
    if status & PCI_ERR_ROOT_NONFATAL_RCV != 0 {
        cmd |= PCI_ERR_ROOT_CMD_NONFATAL_EN;
    }
    if status & PCI_ERR_ROOT_FATAL_RCV != 0 {
        cmd |= PCI_ERR_ROOT_CMD_FATAL_EN;
    }
    cmd
}

/// `pcie_aer_root_write_config()`.
///
/// Like QEMU, when neither MSI nor MSI-X is on this drives INTx from the AER root state on
/// every config write. Since no error is ever reported that means INTx is lowered, even if a
/// hotplug event had raised it. The MSI path would send the AER vector on a false to true
/// transition, which cannot happen here.
pub fn pcie_aer_root_write_config(dev: &PciDevice, offset: u16, _root_cmd_prev: u32) {
    let cfg = dev.config_bytes();
    let o = usize::from(offset);
    let root_status = pci_get_long(&cfg, o + PCI_ERR_ROOT_STATUS);
    let enabled_cmd = pcie_aer_status_to_cmd(root_status);
    let root_cmd = pci_get_long(&cfg, o + PCI_ERR_ROOT_COMMAND);
    if !dev.msix_enabled() && !dev.msi_enabled() && dev.intx() != -1 {
        dev.set_irq(i32::from(root_cmd & enabled_cmd != 0));
    }
}

/// `pcie_sync_bridge_lnk()` for a downstream port: mirrors the width and speed of the device
/// at function 0 of the secondary bus into the port's link status, clamped to what the port
/// supports. With nothing plugged the port reports its own maximum.
pub fn pcie_sync_bridge_lnk(bridge: &PciDevice, exp_cap: u8, target: Option<&PciDevice>) {
    let p = usize::from(exp_cap);
    let lnkcap = pci_get_word(&bridge.config_bytes(), p + PCI_EXP_LNKCAP);
    let target_cap = target.map(|t| (t, t.find_capability(PCI_CAP_ID_EXP)));
    let mut lnksta = match target_cap {
        Some((t, cap)) if cap != 0 => {
            let mut sta = t.config_read(u32::from(cap) + PCI_EXP_LNKSTA as u32, 2) as u16;
            let mlw = (lnkcap as u32 & PCI_EXP_LNKCAP_MLW) as u16;
            if sta & PCI_EXP_LNKSTA_NLW > mlw {
                sta = (sta & !PCI_EXP_LNKSTA_NLW) | mlw;
            }
            let sls = (lnkcap as u32 & PCI_EXP_LNKCAP_SLS) as u16;
            if sta & PCI_EXP_LNKSTA_CLS > sls {
                sta = (sta & !PCI_EXP_LNKSTA_CLS) | sls;
            }
            sta
        }
        _ => lnkcap,
    };
    if lnksta & PCI_EXP_LNKSTA_NLW == 0 {
        lnksta |= lnksta_nlw(PcieLinkWidth::X1);
    }
    if lnksta & PCI_EXP_LNKSTA_CLS == 0 {
        lnksta |= u16::from(PcieLinkSpeed::Gt2_5.bits());
    }
    bridge.with_config(|c| {
        clear_word_mask(c.config, p + PCI_EXP_LNKSTA, PCI_EXP_LNKSTA_CLS | PCI_EXP_LNKSTA_NLW);
        set_word_mask(
            c.config,
            p + PCI_EXP_LNKSTA,
            lnksta & (PCI_EXP_LNKSTA_CLS | PCI_EXP_LNKSTA_NLW),
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ext_cap_header() {
        let h = pci_ext_cap(PCI_EXT_CAP_ID_ACS, 1, 0x148);
        assert_eq!(h, 0x1481_000d);
        assert_eq!(pci_ext_cap_id(h), PCI_EXT_CAP_ID_ACS);
        assert_eq!(pci_ext_cap_ver(h), 1);
        assert_eq!(pci_ext_cap_next(h), 0x148);
    }

    #[test]
    fn powered_off() {
        assert!(pcie_sltctl_powered_off(PCI_EXP_SLTCTL_PWR_OFF | PCI_EXP_SLTCTL_PWR_IND_OFF));
        assert!(!pcie_sltctl_powered_off(PCI_EXP_SLTCTL_PWR_OFF | PCI_EXP_SLTCTL_PWR_IND_BLINK));
        assert!(!pcie_sltctl_powered_off(PCI_EXP_SLTCTL_PWR_IND_OFF));
    }

    #[test]
    fn link_props() {
        assert_eq!(PcieLinkSpeed::from_prop("16"), Some(PcieLinkSpeed::Gt16));
        assert_eq!(PcieLinkSpeed::from_prop("2_5").map(PcieLinkSpeed::bits), Some(1));
        assert_eq!(PcieLinkWidth::from_prop("32"), Some(PcieLinkWidth::X32));
        assert_eq!(PcieLinkWidth::from_prop("3"), None);
    }
}
