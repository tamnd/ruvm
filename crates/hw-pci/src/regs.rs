// SPDX-License-Identifier: GPL-2.0-or-later

//! Configuration space layout, from include/standard-headers/linux/pci_regs.h, plus the handful
//! of QEMU additions in include/hw/pci/pci.h, pci_bridge.h, pci_ids.h and msix.h.
//!
//! Register offsets are `usize` because they index the config arrays. Bit masks carry the width
//! of the register they belong to.

/// `PCI_CONFIG_HEADER_SIZE`, also `PCI_STD_HEADER_SIZEOF`.
pub const PCI_CONFIG_HEADER_SIZE: usize = 0x40;
/// Size of conventional PCI config space.
pub const PCI_CONFIG_SPACE_SIZE: usize = 0x100;
/// Size of PCI Express extended config space.
pub const PCIE_CONFIG_SPACE_SIZE: usize = 0x1000;
pub const PCI_STD_NUM_BARS: usize = 6;

pub const PCI_VENDOR_ID: usize = 0x00;
pub const PCI_DEVICE_ID: usize = 0x02;
pub const PCI_COMMAND: usize = 0x04;
pub const PCI_COMMAND_IO: u16 = 0x1;
pub const PCI_COMMAND_MEMORY: u16 = 0x2;
pub const PCI_COMMAND_MASTER: u16 = 0x4;
pub const PCI_COMMAND_SPECIAL: u16 = 0x8;
pub const PCI_COMMAND_INVALIDATE: u16 = 0x10;
pub const PCI_COMMAND_VGA_PALETTE: u16 = 0x20;
pub const PCI_COMMAND_PARITY: u16 = 0x40;
pub const PCI_COMMAND_WAIT: u16 = 0x80;
pub const PCI_COMMAND_SERR: u16 = 0x100;
pub const PCI_COMMAND_FAST_BACK: u16 = 0x200;
pub const PCI_COMMAND_INTX_DISABLE: u16 = 0x400;

pub const PCI_STATUS: usize = 0x06;
pub const PCI_STATUS_IMM_READY: u16 = 0x01;
pub const PCI_STATUS_INTERRUPT: u16 = 0x08;
pub const PCI_STATUS_CAP_LIST: u16 = 0x10;
pub const PCI_STATUS_66MHZ: u16 = 0x20;
pub const PCI_STATUS_UDF: u16 = 0x40;
pub const PCI_STATUS_FAST_BACK: u16 = 0x80;
pub const PCI_STATUS_PARITY: u16 = 0x100;
pub const PCI_STATUS_DEVSEL_MASK: u16 = 0x600;
pub const PCI_STATUS_SIG_TARGET_ABORT: u16 = 0x800;
pub const PCI_STATUS_REC_TARGET_ABORT: u16 = 0x1000;
pub const PCI_STATUS_REC_MASTER_ABORT: u16 = 0x2000;
pub const PCI_STATUS_SIG_SYSTEM_ERROR: u16 = 0x4000;
pub const PCI_STATUS_DETECTED_PARITY: u16 = 0x8000;

pub const PCI_CLASS_REVISION: usize = 0x08;
pub const PCI_REVISION_ID: usize = 0x08;
pub const PCI_CLASS_PROG: usize = 0x09;
pub const PCI_CLASS_DEVICE: usize = 0x0a;
pub const PCI_CACHE_LINE_SIZE: usize = 0x0c;
pub const PCI_LATENCY_TIMER: usize = 0x0d;
pub const PCI_HEADER_TYPE: usize = 0x0e;
pub const PCI_HEADER_TYPE_MASK: u8 = 0x7f;
pub const PCI_HEADER_TYPE_NORMAL: u8 = 0;
pub const PCI_HEADER_TYPE_BRIDGE: u8 = 1;
pub const PCI_HEADER_TYPE_CARDBUS: u8 = 2;
/// `PCI_HEADER_TYPE_MULTI_FUNCTION` in QEMU, `PCI_HEADER_TYPE_MFD` in Linux.
pub const PCI_HEADER_TYPE_MULTI_FUNCTION: u8 = 0x80;
pub const PCI_BIST: usize = 0x0f;

pub const PCI_BASE_ADDRESS_0: usize = 0x10;
pub const PCI_BASE_ADDRESS_1: usize = 0x14;
pub const PCI_BASE_ADDRESS_2: usize = 0x18;
pub const PCI_BASE_ADDRESS_3: usize = 0x1c;
pub const PCI_BASE_ADDRESS_4: usize = 0x20;
pub const PCI_BASE_ADDRESS_5: usize = 0x24;
pub const PCI_BASE_ADDRESS_SPACE: u8 = 0x01;
pub const PCI_BASE_ADDRESS_SPACE_IO: u8 = 0x01;
pub const PCI_BASE_ADDRESS_SPACE_MEMORY: u8 = 0x00;
pub const PCI_BASE_ADDRESS_MEM_TYPE_MASK: u8 = 0x06;
pub const PCI_BASE_ADDRESS_MEM_TYPE_32: u8 = 0x00;
pub const PCI_BASE_ADDRESS_MEM_TYPE_1M: u8 = 0x02;
pub const PCI_BASE_ADDRESS_MEM_TYPE_64: u8 = 0x04;
pub const PCI_BASE_ADDRESS_MEM_PREFETCH: u8 = 0x08;
pub const PCI_BASE_ADDRESS_MEM_MASK: u64 = !0x0f;
pub const PCI_BASE_ADDRESS_IO_MASK: u64 = !0x03;

pub const PCI_CARDBUS_CIS: usize = 0x28;
pub const PCI_SUBSYSTEM_VENDOR_ID: usize = 0x2c;
pub const PCI_SUBSYSTEM_ID: usize = 0x2e;
pub const PCI_ROM_ADDRESS: usize = 0x30;
pub const PCI_ROM_ADDRESS_ENABLE: u64 = 0x01;
pub const PCI_ROM_ADDRESS_MASK: u32 = !0x7ff;
pub const PCI_CAPABILITY_LIST: usize = 0x34;
pub const PCI_INTERRUPT_LINE: usize = 0x3c;
pub const PCI_INTERRUPT_PIN: usize = 0x3d;
pub const PCI_MIN_GNT: usize = 0x3e;
pub const PCI_MAX_LAT: usize = 0x3f;

// Header type 1 (PCI-to-PCI bridges).
pub const PCI_PRIMARY_BUS: usize = 0x18;
pub const PCI_SECONDARY_BUS: usize = 0x19;
pub const PCI_SUBORDINATE_BUS: usize = 0x1a;
pub const PCI_SEC_LATENCY_TIMER: usize = 0x1b;
pub const PCI_IO_BASE: usize = 0x1c;
pub const PCI_IO_LIMIT: usize = 0x1d;
pub const PCI_IO_RANGE_TYPE_MASK: u8 = 0x0f;
pub const PCI_IO_RANGE_TYPE_16: u8 = 0x00;
pub const PCI_IO_RANGE_TYPE_32: u8 = 0x01;
pub const PCI_IO_RANGE_MASK: u8 = !0x0f;
pub const PCI_SEC_STATUS: usize = 0x1e;
pub const PCI_MEMORY_BASE: usize = 0x20;
pub const PCI_MEMORY_LIMIT: usize = 0x22;
pub const PCI_MEMORY_RANGE_TYPE_MASK: u16 = 0x0f;
pub const PCI_MEMORY_RANGE_MASK: u16 = !0x0f;
pub const PCI_PREF_MEMORY_BASE: usize = 0x24;
pub const PCI_PREF_MEMORY_LIMIT: usize = 0x26;
pub const PCI_PREF_RANGE_TYPE_MASK: u16 = 0x0f;
pub const PCI_PREF_RANGE_TYPE_32: u16 = 0x00;
pub const PCI_PREF_RANGE_TYPE_64: u16 = 0x01;
pub const PCI_PREF_RANGE_MASK: u16 = !0x0f;
pub const PCI_PREF_BASE_UPPER32: usize = 0x28;
pub const PCI_PREF_LIMIT_UPPER32: usize = 0x2c;
pub const PCI_IO_BASE_UPPER16: usize = 0x30;
pub const PCI_IO_LIMIT_UPPER16: usize = 0x32;
pub const PCI_ROM_ADDRESS1: usize = 0x38;
pub const PCI_BRIDGE_CONTROL: usize = 0x3e;
pub const PCI_BRIDGE_CTL_PARITY: u16 = 0x01;
pub const PCI_BRIDGE_CTL_SERR: u16 = 0x02;
pub const PCI_BRIDGE_CTL_ISA: u16 = 0x04;
pub const PCI_BRIDGE_CTL_VGA: u16 = 0x08;
pub const PCI_BRIDGE_CTL_VGA_16BIT: u16 = 0x10;
pub const PCI_BRIDGE_CTL_MASTER_ABORT: u16 = 0x20;
pub const PCI_BRIDGE_CTL_BUS_RESET: u16 = 0x40;
pub const PCI_BRIDGE_CTL_FAST_BACK: u16 = 0x80;
pub const PCI_BRIDGE_CTL_DISCARD: u16 = 0x100;
pub const PCI_BRIDGE_CTL_SEC_DISCARD: u16 = 0x200;
pub const PCI_BRIDGE_CTL_DISCARD_STATUS: u16 = 0x400;
pub const PCI_BRIDGE_CTL_DISCARD_SERR: u16 = 0x800;

// Capability lists.
pub const PCI_CAP_LIST_ID: usize = 0;
pub const PCI_CAP_LIST_NEXT: usize = 1;
pub const PCI_CAP_FLAGS: usize = 2;
pub const PCI_CAP_SIZEOF: u8 = 4;
pub const PCI_CAP_ID_PM: u8 = 0x01;
pub const PCI_CAP_ID_AGP: u8 = 0x02;
pub const PCI_CAP_ID_VPD: u8 = 0x03;
pub const PCI_CAP_ID_SLOTID: u8 = 0x04;
pub const PCI_CAP_ID_MSI: u8 = 0x05;
pub const PCI_CAP_ID_CHSWP: u8 = 0x06;
pub const PCI_CAP_ID_PCIX: u8 = 0x07;
pub const PCI_CAP_ID_HT: u8 = 0x08;
pub const PCI_CAP_ID_VNDR: u8 = 0x09;
pub const PCI_CAP_ID_DBG: u8 = 0x0a;
pub const PCI_CAP_ID_CCRC: u8 = 0x0b;
pub const PCI_CAP_ID_SHPC: u8 = 0x0c;
pub const PCI_CAP_ID_SSVID: u8 = 0x0d;
pub const PCI_CAP_ID_AGP3: u8 = 0x0e;
pub const PCI_CAP_ID_SECDEV: u8 = 0x0f;
pub const PCI_CAP_ID_EXP: u8 = 0x10;
pub const PCI_CAP_ID_MSIX: u8 = 0x11;
pub const PCI_CAP_ID_SATA: u8 = 0x12;
pub const PCI_CAP_ID_AF: u8 = 0x13;
pub const PCI_CAP_ID_EA: u8 = 0x14;

// Power management.
pub const PCI_PM_PMC: usize = 2;
pub const PCI_PM_CAP_D1: u16 = 0x0200;
pub const PCI_PM_CAP_D2: u16 = 0x0400;
pub const PCI_PM_CTRL: usize = 4;
pub const PCI_PM_CTRL_STATE_MASK: u16 = 0x0003;
pub const PCI_PM_CTRL_NO_SOFT_RESET: u16 = 0x0008;
pub const PCI_PM_SIZEOF: u8 = 8;

// MSI.
pub const PCI_MSI_FLAGS: usize = 0x02;
pub const PCI_MSI_FLAGS_ENABLE: u16 = 0x0001;
pub const PCI_MSI_FLAGS_QMASK: u16 = 0x000e;
pub const PCI_MSI_FLAGS_QSIZE: u16 = 0x0070;
pub const PCI_MSI_FLAGS_64BIT: u16 = 0x0080;
pub const PCI_MSI_FLAGS_MASKBIT: u16 = 0x0100;
pub const PCI_MSI_ADDRESS_LO: usize = 0x04;
pub const PCI_MSI_ADDRESS_HI: usize = 0x08;
pub const PCI_MSI_DATA_32: usize = 0x08;
pub const PCI_MSI_MASK_32: usize = 0x0c;
pub const PCI_MSI_PENDING_32: usize = 0x10;
pub const PCI_MSI_DATA_64: usize = 0x0c;
pub const PCI_MSI_MASK_64: usize = 0x10;
pub const PCI_MSI_PENDING_64: usize = 0x14;

// MSI-X.
pub const PCI_MSIX_FLAGS: usize = 2;
pub const PCI_MSIX_FLAGS_QSIZE: u16 = 0x07ff;
pub const PCI_MSIX_FLAGS_MASKALL: u16 = 0x4000;
pub const PCI_MSIX_FLAGS_ENABLE: u16 = 0x8000;
pub const PCI_MSIX_TABLE: usize = 4;
pub const PCI_MSIX_PBA: usize = 8;
pub const PCI_MSIX_FLAGS_BIRMASK: u32 = 0x7;
pub const PCI_CAP_MSIX_SIZEOF: u8 = 12;
/// `MSIX_CAP_LENGTH` from include/hw/pci/msix.h.
pub const MSIX_CAP_LENGTH: u8 = 12;
pub const PCI_MSIX_ENTRY_SIZE: usize = 16;
pub const PCI_MSIX_ENTRY_LOWER_ADDR: usize = 0x0;
pub const PCI_MSIX_ENTRY_UPPER_ADDR: usize = 0x4;
pub const PCI_MSIX_ENTRY_DATA: usize = 0x8;
pub const PCI_MSIX_ENTRY_VECTOR_CTRL: usize = 0xc;
pub const PCI_MSIX_ENTRY_CTRL_MASKBIT: u8 = 0x1;

// From include/hw/pci/pci_ids.h and pci.h.
pub const PCI_CLASS_BRIDGE_HOST: u16 = 0x0600;
pub const PCI_CLASS_BRIDGE_ISA: u16 = 0x0601;
pub const PCI_CLASS_BRIDGE_PCI: u16 = 0x0604;
pub const PCI_CLASS_OTHERS: u16 = 0xff;
pub const PCI_VENDOR_ID_REDHAT: u16 = 0x1b36;
pub const PCI_VENDOR_ID_REDHAT_QUMRANET: u16 = 0x1af4;
pub const PCI_SUBVENDOR_ID_REDHAT_QUMRANET: u16 = 0x1af4;
pub const PCI_SUBDEVICE_ID_QEMU: u16 = 0x1100;

/// Number of INTx pins, INTA to INTD.
pub const PCI_NUM_PINS: usize = 4;
/// Six BARs plus the option ROM.
pub const PCI_NUM_REGIONS: usize = 7;
/// The region number of the expansion ROM BAR.
pub const PCI_ROM_SLOT: usize = 6;
pub const PCI_SLOT_MAX: u8 = 32;
pub const PCI_FUNC_MAX: u8 = 8;
pub const PCI_DEVFN_MAX: usize = 256;
/// The address of a BAR that is not mapped.
pub const PCI_BAR_UNMAPPED: u64 = u64::MAX;

/// `PCI_DEVFN(slot, func)`.
pub const fn pci_devfn(slot: u8, func: u8) -> u8 {
    ((slot & 0x1f) << 3) | (func & 0x07)
}

/// `PCI_SLOT(devfn)`.
pub const fn pci_slot(devfn: u8) -> u8 {
    (devfn >> 3) & 0x1f
}

/// `PCI_FUNC(devfn)`.
pub const fn pci_func(devfn: u8) -> u8 {
    devfn & 0x07
}

/// `pci_swizzle()`: the standard rotation of INTx pins across slots, 0-origin pins.
pub const fn pci_swizzle(slot: i32, pin: i32) -> i32 {
    (slot + pin) % PCI_NUM_PINS as i32
}

/// `pci_config_size()` for a conventional or an Express function.
pub const fn pci_config_size(express: bool) -> usize {
    if express { PCIE_CONFIG_SPACE_SIZE } else { PCI_CONFIG_SPACE_SIZE }
}

// Little endian accessors for config arrays, `pci_get_word()` and friends.

pub fn pci_get_word(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

pub fn pci_set_word(b: &mut [u8], off: usize, v: u16) {
    b[off..off + 2].copy_from_slice(&v.to_le_bytes());
}

pub fn pci_get_long(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

pub fn pci_set_long(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

pub fn pci_get_quad(b: &[u8], off: usize) -> u64 {
    u64::from(pci_get_long(b, off)) | (u64::from(pci_get_long(b, off + 4)) << 32)
}

pub fn pci_set_quad(b: &mut [u8], off: usize, v: u64) {
    b[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

/// `ranges_overlap()` from include/qemu/range.h.
pub(crate) fn ranges_overlap(first1: u64, len1: u64, first2: u64, len2: u64) -> bool {
    let last1 = first1 + len1 - 1;
    let last2 = first2 + len2 - 1;
    !(last2 < first1 || last1 < first2)
}

/// `range_covers_byte()`.
pub(crate) fn range_covers_byte(offset: u64, len: u64, byte: u64) -> bool {
    offset <= byte && byte < offset + len
}
