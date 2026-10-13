// SPDX-License-Identifier: GPL-2.0-or-later

//! The `amd-iommu` device, from hw/i386/amd_iommu.c, hw/i386/amd_iommu.h and
//! hw/i386/x86-iommu.c.
//!
//! The IOMMU sits on the q35 root bus as two parts: the `amd-iommu` sysbus device, which owns
//! the register frame at [`AMDVI_BASE_ADDR`] and does the translation, and the `AMDVI-PCI`
//! function, which carries its capability, its MSI and the target abort status bit. The board
//! builds the PCI function and hands it over as an [`AmdviPci`].
//!
//! Each device gets an IOMMU region from [`AmdIommu::device_ops`], keyed by its device ID. As
//! in QEMU a device starts out in pass-through, and only with `dma-remap=on` does an
//! `INVALIDATE_DEVTAB_ENTRY` command for a device table entry with a paging mode switch it to
//! translation. Translation walks the v1 page tables with the device table entry's mode, keeps
//! an IOTLB per device ID and page, and logs I/O page faults and table errors to the event
//! log. The command buffer has all of QEMU's commands. Interrupt remapping takes the MSIs of
//! the PCI devices and the IOAPIC through [`AmdIommu::remap_msi`], with the legacy and the
//! 128 bit interrupt remapping table entries and `xtsup=on` x2APIC destinations.
//!
//! # Differences from QEMU
//!
//! - There are no IOMMU notifiers and no replay, so nothing like vhost or VFIO can sit behind
//!   the IOMMU, and `INVALIDATE_IOMMU_PAGES` only drops IOTLB entries.
//! - A device's region is created when it first does DMA instead of when it is plugged, so an
//!   `INVALIDATE_DEVTAB_ENTRY` for a device that has done no DMA yet does not read its device
//!   table entry.
//! - MSIs reach [`AmdIommu::remap_msi`] without a DMA write, as with intel-iommu, so a plain
//!   DMA write to 0xfee00000 to 0xfeefffff does not raise an interrupt.
//! - A command buffer that is run before its base register is written does nothing, where
//!   QEMU loops forever, and an event log enabled before its base register is written drops
//!   the events, where QEMU divides by zero.
//! - There is no VMState.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

use ruvm_base::report::error_report;
use ruvm_mem::{
    AccessConstraints, AccessCtx, AccessSize, AddressSpace, IommuAccessFlags, IommuOps,
    IommuTlbEntry, MemResult, MemTxAttrs, MemTxResult, MmioOps,
};

use crate::intel_iommu::{OnOffAuto, SID_INVALID};

/// `TYPE_AMD_IOMMU_DEVICE`.
pub const TYPE_AMD_IOMMU_DEVICE: &str = "amd-iommu";
/// `TYPE_AMD_IOMMU_PCI`.
pub const TYPE_AMD_IOMMU_PCI: &str = "AMDVI-PCI";
/// `TYPE_AMD_IOMMU_MEMORY_REGION`, the stem of the per-device region names.
pub const TYPE_AMD_IOMMU_MEMORY_REGION: &str = "amd-iommu-iommu-memory-region";
/// `AMDVI_BASE_ADDR`, where the register frame is mapped.
pub const AMDVI_BASE_ADDR: u64 = 0xfed8_0000;
/// `AMDVI_MMIO_SIZE`.
pub const AMDVI_MMIO_SIZE: u64 = 0x4000;
/// `AMDVI_IOAPIC_SB_DEVID`, 00:14.0, the device ID of the IOAPIC's interrupts.
pub const AMDVI_IOAPIC_SB_DEVID: u16 = 0xa0;

/// `PCI_VENDOR_ID_AMD`.
pub const AMDVI_PCI_VENDOR_ID: u16 = 0x1022;
/// `PCI_DEVICE_ID_RD890_IOMMU`.
pub const AMDVI_PCI_DEVICE_ID: u16 = 0x1419;
/// `PCI_CLASS_SYSTEM_IOMMU`.
pub const AMDVI_PCI_CLASS: u16 = 0x0806;
/// `AMDVI_CAPAB_ID_SEC`, the secure device capability.
pub const AMDVI_CAPAB_ID_SEC: u8 = 0xf;
/// `AMDVI_CAPAB_SIZE`.
pub const AMDVI_CAPAB_SIZE: u8 = 0x18;
/// `AMDVI_CAPAB_REG_SIZE`, the size of the MSI and HyperTransport capabilities
/// `amdvi_pci_realize()` adds before `msi_init()`.
pub const AMDVI_CAPAB_REG_SIZE: u8 = 0x04;
/// `AMDVI_CAPAB_FEATURES`, the first dword of the capability.
pub const AMDVI_CAPAB_FEATURES: u32 = (1 << 28) // FLAT_EXT
    | (1 << 26) // NPCACHE
    | (1 << 24) // IOTLBSUP
    | AMDVI_CAPAB_ID_SEC as u32
    | (3 << 16) // INIT_TYPE
    | (1 << 25) // HTTUNNEL
    | (1 << 27); // EFR_SUP
/// The other dwords of the capability: `AMDVI_CAPAB_BAR_LOW`, `AMDVI_CAPAB_BAR_HIGH`,
/// `AMDVI_CAPAB_RANGE` and `AMDVI_CAPAB_MISC`, each with its offset in the capability.
pub const AMDVI_CAPAB_REGS: [(u8, u32); 4] = [
    (0x04, (AMDVI_BASE_ADDR & (((1 << 18) - 1) << 14)) as u32),
    (0x08, (AMDVI_BASE_ADDR >> 32) as u32),
    (0x0c, 0xff00_0000),
    // AMDVI_MAX_PH_ADDR | AMDVI_MAX_GVA_ADDR | AMDVI_MAX_VA_ADDR
    (0x10, (40 << 8) | (2 << 5) | (48 << 15)),
];

const MMIO_SIZE: usize = AMDVI_MMIO_SIZE as usize;

const MMIO_DEVICE_TABLE: u64 = 0x0000;
const MMIO_COMMAND_BASE: u64 = 0x0008;
const MMIO_EVENT_BASE: u64 = 0x0010;
const MMIO_CONTROL: u64 = 0x0018;
const MMIO_EXCL_LIMIT: u64 = 0x0028;
const MMIO_EXT_FEATURES: u64 = 0x0030;
const MMIO_PPR_BASE: u64 = 0x0038;
const MMIO_XT_GEN_INTR: u64 = 0x0170;
const MMIO_COMMAND_HEAD: u64 = 0x2000;
const MMIO_COMMAND_TAIL: u64 = 0x2008;
const MMIO_EVENT_HEAD: u64 = 0x2010;
const MMIO_EVENT_TAIL: u64 = 0x2018;
const MMIO_STATUS: u64 = 0x2020;
const MMIO_PPR_HEAD: u64 = 0x2030;
const MMIO_PPR_TAIL: u64 = 0x2038;

/// `GENMASK64(h, l)`.
const fn genmask(h: u32, l: u32) -> u64 {
    (!0u64 >> (63 - h + l)) << l
}

/// `extract64()`.
const fn extract(v: u64, start: u32, len: u32) -> u64 {
    (v >> start) & (!0u64 >> (64 - len))
}

const DEVTAB_SIZE_MASK: u64 = genmask(8, 0);
const BASE_MASK: u64 = genmask(51, 12);
const PTR_MASK: u64 = genmask(18, 4);
const DEVTAB_ENTRY_SIZE: u64 = 32;
const DEVTAB_SIZE_UNIT: u64 = 4096;

const CONTROL_AMDVIEN: u64 = 1 << 0;
const CONTROL_EVENTLOGEN: u64 = 1 << 2;
const CONTROL_EVENTINTEN: u64 = 1 << 3;
const CONTROL_CMDBUFLEN: u64 = 1 << 12;
const CONTROL_GAEN: u64 = 1 << 17;
const CONTROL_XTEN: u64 = 1 << 50;
const CONTROL_INTCAPXTEN: u64 = 1 << 51;

const STATUS_CMDBUF_RUN: u64 = 1 << 4;
const STATUS_EVT_RUN: u64 = 1 << 3;
const STATUS_COMP_INT: u64 = 1 << 2;
const STATUS_EVENT_INT: u64 = 1 << 1;
const STATUS_EVT_OVF: u64 = 1 << 0;

const CMD_COMPLETION_WAIT: u64 = 0x01;
const CMD_INVAL_DEVTAB_ENTRY: u64 = 0x02;
const CMD_INVAL_AMDVI_PAGES: u64 = 0x03;
const CMD_INVAL_IOTLB_PAGES: u64 = 0x04;
const CMD_INVAL_INTR_TABLE: u64 = 0x05;
const CMD_PREFETCH_AMDVI_PAGES: u64 = 0x06;
const CMD_COMPLETE_PPR_REQUEST: u64 = 0x07;
const CMD_INVAL_AMDVI_ALL: u64 = 0x08;
const COMMAND_SIZE: u32 = 16;

const DEV_VALID: u64 = 1 << 0;
const DEV_TRANSLATION_VALID: u64 = 1 << 1;
const DEV_MODE_MASK: u64 = 0x7;
const DEV_MODE_RSHIFT: u32 = 9;
const DEV_PERM_SHIFT: u32 = 61;
const DEV_PERM_READ: u64 = 1 << 61;
const DEV_PERM_WRITE: u64 = 1 << 62;
const DEV_DOMID_ID_MASK: u64 = genmask(15, 0);

const EVENT_ILLEGAL_DEVTAB_ENTRY: u16 = 0x1 << 12;
const EVENT_IOPF: u16 = 0x2 << 12;
const EVENT_IOPF_I: u16 = 1 << 3;
const EVENT_DEV_TAB_HW_ERROR: u16 = 0x3 << 12;
const EVENT_PAGE_TAB_HW_ERROR: u16 = 0x4 << 12;
const EVENT_ILLEGAL_COMMAND_ERROR: u16 = 0x5 << 12;
const EVENT_COMMAND_HW_ERROR: u16 = 0x6 << 12;
const EVENT_LEN: u32 = 16;

const PERM_READ: u64 = 1 << 0;
const PERM_WRITE: u64 = 1 << 1;

const FEATURE_PREFETCH: u64 = 1 << 0;
const FEATURE_PPR: u64 = 1 << 1;
const FEATURE_XT: u64 = 1 << 2;
const FEATURE_GT: u64 = 1 << 4;
const FEATURE_IA: u64 = 1 << 6;
const FEATURE_GA: u64 = 1 << 7;
const FEATURE_HE: u64 = 1 << 8;
const GATS_MODE: u64 = 2 << 12;
const HATS_MODE: u64 = 2 << 10;
const HATS_MODE_RESERVED: u64 = 3 << 10;
/// `AMDVI_DEFAULT_EXT_FEATURES`.
const DEFAULT_EXT_FEATURES: u64 = FEATURE_PREFETCH
    | FEATURE_PPR
    | FEATURE_IA
    | FEATURE_GT
    | FEATURE_HE
    | GATS_MODE
    | HATS_MODE
    | FEATURE_GA;

const DTE_QUAD0_RESERVED: u64 = genmask(6, 2) | genmask(63, 63);
const DTE_QUAD2_RESERVED: u64 = genmask(53, 52);
const DTE_QUAD3_RESERVED: u64 = genmask(14, 0) | genmask(53, 48);

const PTE_PR: u64 = 1 << 0;
const PTE_NEXT_LEVEL_MASK: u64 = genmask(11, 9);

const IOTLB_MAX_SIZE: usize = 1024;
const PAGE_MASK_4K: u64 = genmask(63, 12);
const PAGE_SIZE: u64 = 1 << 12;
const INV_ALL_PAGES: u64 = 1 << 52;

const INT_ADDR_FIRST: u64 = 0xfee0_0000;
const INT_ADDR_LAST: u64 = 0xfeef_ffff;

const IR_REMAP_ENABLE: u64 = 1;
const IR_INTCTL_SHIFT: u32 = 60;
const IR_INTCTL_PASS: u64 = 1;
const IR_INTCTL_REMAP: u64 = 2;
const IR_PHYS_ADDR_MASK: u64 = genmask(51, 6);
const IRTE_OFFSET: u32 = 0x7ff;

const IOAPIC_INT_TYPE_FIXED: u32 = 0x0;
const IOAPIC_INT_TYPE_ARBITRATED: u32 = 0x1;
const IOAPIC_INT_TYPE_SMI: u32 = 0x2;
const IOAPIC_INT_TYPE_NMI: u32 = 0x4;
const IOAPIC_INT_TYPE_INIT: u32 = 0x5;
const IOAPIC_INT_TYPE_EINT: u32 = 0x7;

const DEV_INT_PASS_MASK: u64 = 1 << 56;
const DEV_EINT_PASS_MASK: u64 = 1 << 57;
const DEV_NMI_PASS_MASK: u64 = 1 << 58;

/// `PT_LEVEL_SHIFT()`: level 1 is the 4 KiB leaf.
const fn level_shift(level: u32) -> u32 {
    12 + (level - 1) * 9
}

/// `PT_LEVEL_MAX_ADDR()`.
const fn level_max_addr(level: u32) -> u64 {
    if level > 5 { !0 } else { (1 << level_shift(level + 1)) - 1 }
}

/// `NEXT_PTE_ADDR()`.
const fn next_pte_addr(pte: u64, level: u32, iova: u64) -> u64 {
    (pte & BASE_MASK) + ((iova >> level_shift(level)) & genmask(8, 0)) * 8
}

/// `PTE_NEXT_LEVEL()`.
const fn pte_next_level(pte: u64) -> u32 {
    ((pte & PTE_NEXT_LEVEL_MASK) >> 9) as u32
}

/// `amdvi_get_perms()`.
const fn get_perms(entry: u64) -> u64 {
    (entry & (DEV_PERM_READ | DEV_PERM_WRITE)) >> DEV_PERM_SHIFT
}

fn perm_flags(perm: u64) -> IommuAccessFlags {
    match perm & 3 {
        1 => IommuAccessFlags::RO,
        2 => IommuAccessFlags::WO,
        3 => IommuAccessFlags::RW,
        _ => IommuAccessFlags::NONE,
    }
}

/// `error_report_once()`, one flag for each place that uses it.
struct Once(AtomicBool);

impl Once {
    const fn new() -> Self {
        Once(AtomicBool::new(false))
    }

    fn report(&self, msg: impl FnOnce() -> String) {
        if !self.0.swap(true, Ordering::Relaxed) {
            error_report(&msg());
        }
    }
}

static GUEST_MODE: Once = Once::new();
static RESERVED_INT_TYPE: Once = Once::new();
static RESERVED_INT_TYPE_GA: Once = Once::new();
static REMAP_FAILED: Once = Once::new();

/// The properties of `amd-iommu`, with those of its `x86-iommu` parent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AmdIommuConf {
    /// `intremap`.
    pub intremap: OnOffAuto,
    /// `device-iotlb`, which amd-iommu does not use.
    pub device_iotlb: bool,
    /// `dma-translation`.
    pub dma_translation: bool,
    /// `xtsup`.
    pub xtsup: bool,
    /// `pci-id`, the id of a user created `AMDVI-PCI` device.
    pub pci_id: Option<String>,
    /// `dma-remap`.
    pub dma_remap: bool,
}

impl Default for AmdIommuConf {
    fn default() -> Self {
        AmdIommuConf {
            intremap: OnOffAuto::Auto,
            device_iotlb: false,
            dma_translation: true,
            xtsup: false,
            pci_id: None,
            dma_remap: false,
        }
    }
}

impl AmdIommuConf {
    /// `x86_iommu_realize()`: the configuration with `intremap` decided, or the error QEMU
    /// gives. `irq_all_kernel` is KVM with the whole irqchip in the kernel.
    pub fn realize(&self, irq_all_kernel: bool) -> Result<Self, String> {
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
        c.intremap = if ir { OnOffAuto::On } else { OnOffAuto::Off };
        Ok(c)
    }

    /// `x86_iommu_ir_supported()`.
    pub fn intr_supported(&self) -> bool {
        self.intremap == OnOffAuto::On
    }

    /// `amdvi_extended_feature_register()`.
    pub fn extended_features(&self) -> u64 {
        let mut feature = DEFAULT_EXT_FEATURES;
        if self.xtsup {
            feature |= FEATURE_XT;
        }
        if !self.dma_translation {
            feature |= HATS_MODE_RESERVED;
        }
        feature
    }
}

/// The `AMDVI-PCI` function as the IOMMU uses it.
pub trait AmdviPci: Send + Sync {
    /// `msi_get_message(dev, 0)` when `msi_enabled()`, or `None`. In QEMU this message goes
    /// through the function's own DMA address space, so the board runs it through
    /// [`AmdIommu::remap_msi`].
    fn msi_message(&self) -> Option<(u64, u32)>;

    /// Sets `PCI_STATUS_SIG_TARGET_ABORT` in the status register.
    fn signal_target_abort(&self);
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
}

/// Why a device table entry gave no page table, `AMDVIFaultReason`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DteError {
    /// `AMDVI_FR_DTE_RTR_ERR`: the entry could not be read or is invalid.
    Retrieve,
    /// `AMDVI_FR_DTE_V`: DTE[V] is clear, so DMA passes untranslated.
    NotValid,
    /// `AMDVI_FR_DTE_TV`: DTE[TV] is clear.
    NoTranslation,
}

/// One cached translation, `AMDVIIOTLBEntry`.
#[derive(Clone, Copy, Debug)]
struct IotlbEntry {
    domid: u16,
    devid: u16,
    perms: u64,
    translated_addr: u64,
    page_mask: u64,
}

/// The result of a translation before it becomes an [`IommuTlbEntry`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Xlate {
    iova: u64,
    translated: u64,
    mask: u64,
    perm: u64,
}

impl Xlate {
    /// The entry `amdvi_translate()` starts from, which is a fault.
    fn fault(addr: u64) -> Self {
        Xlate { iova: addr, translated: 0, mask: !0, perm: 0 }
    }

    fn identity(addr: u64, perm: u64) -> Self {
        Xlate { iova: addr & PAGE_MASK_4K, translated: addr & PAGE_MASK_4K, mask: 0xfff, perm }
    }
}

/// An interrupt the IOMMU raises, sent once the state lock is dropped.
enum Irq {
    /// A message built from the XT IOMMU general interrupt control register.
    Msi(u64, u32),
    /// The MSI of the `AMDVI-PCI` function, if it is enabled.
    Pci,
}

/// `X86IOMMUIrq`.
#[derive(Clone, Copy, Debug, Default)]
struct X86Irq {
    vector: u32,
    delivery_mode: u32,
    dest_mode: u32,
    redir_hint: u32,
    dest: u32,
}

impl X86Irq {
    /// `x86_iommu_irq_to_msi_message()`, with a zero trigger mode and no low address bits.
    fn to_msi(self) -> (u64, u32) {
        let addr = INT_ADDR_FIRST
            | u64::from(self.dest & 0xff) << 12
            | u64::from(self.redir_hint & 1) << 3
            | u64::from(self.dest_mode & 1) << 2
            | u64::from(self.dest & 0xffff_ff00) << 32;
        let data = (self.vector & 0xff) | (self.delivery_mode & 7) << 8 | 1 << 14;
        (addr, data)
    }
}

/// `AMDVIState`, less what lives in memory regions.
struct State {
    conf: AmdIommuConf,
    enabled: bool,
    cmdbuf_enabled: bool,
    evtlog_enabled: bool,
    devtab: u64,
    devtab_len: u64,
    cmdbuf: u64,
    cmdbuf_len: u64,
    cmdbuf_head: u32,
    cmdbuf_tail: u32,
    evtlog: u64,
    evtlog_intr: bool,
    evtlog_len: u32,
    evtlog_head: u32,
    evtlog_tail: u32,
    mmior: Vec<u8>,
    w1cmask: Vec<u8>,
    romask: Vec<u8>,
    /// The IOTLB, keyed by page frame and device ID.
    iotlb: HashMap<(u64, u16), IotlbEntry>,
    ga_enabled: bool,
    xten: bool,
    intcapxten: bool,
    /// The device IDs that have an address space, `address_spaces`.
    spaces: HashSet<u16>,
    /// Those whose address space translates, `addr_translation`.
    translating: HashSet<u16>,
    /// Interrupts raised while the lock is held.
    irqs: Vec<Irq>,
    /// `PCI_STATUS_SIG_TARGET_ABORT` was signalled while the lock was held.
    target_abort: bool,
}

impl State {
    fn new(conf: AmdIommuConf) -> Self {
        let mut s = State {
            conf,
            enabled: false,
            cmdbuf_enabled: false,
            evtlog_enabled: false,
            devtab: 0,
            devtab_len: 0,
            cmdbuf: 0,
            cmdbuf_len: 0,
            cmdbuf_head: 0,
            cmdbuf_tail: 0,
            evtlog: 0,
            evtlog_intr: false,
            evtlog_len: 0,
            evtlog_head: 0,
            evtlog_tail: 0,
            mmior: vec![0; MMIO_SIZE],
            w1cmask: vec![0; MMIO_SIZE],
            romask: vec![0; MMIO_SIZE],
            iotlb: HashMap::new(),
            ga_enabled: false,
            xten: false,
            intcapxten: false,
            // amdvi_sysbus_realize() makes the IOAPIC's address space.
            spaces: HashSet::from([AMDVI_IOAPIC_SB_DEVID]),
            translating: HashSet::new(),
            irqs: Vec::new(),
            target_abort: false,
        };
        s.init();
        s
    }

    /// `amdvi_init()`.
    fn init(&mut self) {
        self.iotlb.clear();
        self.devtab_len = 0;
        self.cmdbuf_len = 0;
        self.cmdbuf_head = 0;
        self.cmdbuf_tail = 0;
        self.evtlog_head = 0;
        self.evtlog_tail = 0;
        self.enabled = false;
        self.cmdbuf_enabled = false;
        self.xten = false;
        self.intcapxten = false;

        self.mmior.fill(0);
        let efr = self.conf.extended_features();
        self.set_quad(MMIO_EXT_FEATURES, efr, 0xffff_ffff_ffff_ffef, 0);
        self.set_quad(MMIO_STATUS, 0, 0x98, 0x67);
    }

    /// `amdvi_sysbus_reset()`, less the MSI reset of the PCI function.
    fn reset(&mut self) {
        self.init();
        self.translating.clear();
    }

    /// `amdvi_set_quad()`.
    fn set_quad(&mut self, addr: u64, val: u64, romask: u64, w1cmask: u64) {
        let a = addr as usize;
        self.mmior[a..a + 8].copy_from_slice(&val.to_le_bytes());
        self.romask[a..a + 8].copy_from_slice(&romask.to_le_bytes());
        self.w1cmask[a..a + 8].copy_from_slice(&w1cmask.to_le_bytes());
    }

    fn load(buf: &[u8], addr: u64, size: usize) -> u64 {
        let a = addr as usize;
        let mut b = [0u8; 8];
        b[..size].copy_from_slice(&buf[a..a + size]);
        u64::from_le_bytes(b)
    }

    /// `amdvi_readq()`.
    fn readq(&self, addr: u64) -> u64 {
        Self::load(&self.mmior, addr, 8)
    }

    /// `amdvi_writeq_raw()`.
    fn writeq_raw(&mut self, addr: u64, val: u64) {
        let a = addr as usize;
        self.mmior[a..a + 8].copy_from_slice(&val.to_le_bytes());
    }

    /// `amdvi_writew()`, `amdvi_writel()` and `amdvi_writeq()`: a guest write through the
    /// read-only and write-1-to-clear masks.
    fn write_masked(&mut self, addr: u64, size: usize, val: u64) {
        let romask = Self::load(&self.romask, addr, size);
        let w1cmask = Self::load(&self.w1cmask, addr, size);
        let oldval = Self::load(&self.mmior, addr, size);
        let new = ((oldval & (romask | w1cmask)) | (val & !romask)) & !(val & w1cmask);
        let a = addr as usize;
        self.mmior[a..a + size].copy_from_slice(&new.to_le_bytes()[..size]);
    }

    /// `amdvi_assign_orq()`.
    fn orq(&mut self, addr: u64, val: u64) {
        let v = self.readq(addr) | val;
        self.writeq_raw(addr, v);
    }

    /// `amdvi_assign_andq()`.
    fn andq(&mut self, addr: u64, val: u64) {
        let v = self.readq(addr) & val;
        self.writeq_raw(addr, v);
    }

    /// `amdvi_generate_msi_interrupt()`.
    fn generate_msi_interrupt(&mut self) {
        if self.intcapxten {
            // amdvi_build_xt_msi_msg()
            let xt = self.readq(MMIO_XT_GEN_INTR);
            let irq = X86Irq {
                vector: extract(xt, 32, 8) as u32,
                delivery_mode: extract(xt, 40, 1) as u32,
                dest_mode: extract(xt, 2, 1) as u32,
                dest: ((extract(xt, 56, 8) << 24) | extract(xt, 8, 24)) as u32,
                redir_hint: 0,
            };
            let (addr, data) = irq.to_msi();
            self.irqs.push(Irq::Msi(addr, data));
        } else {
            self.irqs.push(Irq::Pci);
        }
    }

    /// `amdvi_log_event()`.
    fn log_event(&mut self, m: &Dma<'_>, evt: [u64; 2]) {
        if !self.evtlog_enabled || self.readq(MMIO_STATUS) & STATUS_EVT_OVF != 0 {
            return;
        }
        // get_next_eventlog_entry()
        let Some(next) = (self.evtlog_tail + EVENT_LEN).checked_rem(self.evtlog_len * EVENT_LEN)
        else {
            return;
        };
        if next == self.evtlog_head {
            if self.evtlog_intr {
                self.orq(MMIO_STATUS, STATUS_EVT_OVF);
                self.generate_msi_interrupt();
            }
            return;
        }
        let mut b = [0u8; 16];
        b[..8].copy_from_slice(&evt[0].to_le_bytes());
        b[8..].copy_from_slice(&evt[1].to_le_bytes());
        // QEMU only traces a failed write.
        let _ = m.write(self.evtlog + u64::from(self.evtlog_tail), &b);
        self.evtlog_tail = next;
        self.writeq_raw(MMIO_EVENT_TAIL, u64::from(next));
        if self.evtlog_intr {
            self.orq(MMIO_STATUS, STATUS_EVENT_INT);
            self.generate_msi_interrupt();
        }
    }

    /// `amdvi_encode_event()` and `amdvi_log_event()`.
    fn event(&mut self, m: &Dma<'_>, devid: u16, addr: u64, info: u16) {
        self.log_event(m, [u64::from(devid) | u64::from(info) << 48, addr]);
    }

    /// `amdvi_page_fault()`.
    fn page_fault(&mut self, m: &Dma<'_>, devid: u16, addr: u64, info: u16) {
        self.event(m, devid, addr, info | EVENT_IOPF_I | EVENT_IOPF);
        self.target_abort = true;
    }

    /// `amdvi_log_devtab_error()`.
    fn log_devtab_error(&mut self, m: &Dma<'_>, devid: u16, devtab: u64, info: u16) {
        self.event(m, devid, devtab, info | EVENT_DEV_TAB_HW_ERROR);
        self.target_abort = true;
    }

    /// `amdvi_log_command_error()`.
    fn log_command_error(&mut self, m: &Dma<'_>, addr: u64) {
        self.event(m, 0, addr, EVENT_COMMAND_HW_ERROR);
        self.target_abort = true;
    }

    /// `amdvi_log_illegalcom_error()` for the command at the head of the buffer.
    fn log_illegalcom_error(&mut self, m: &Dma<'_>, cmd0: u64) {
        let addr = self.cmdbuf + u64::from(self.cmdbuf_head);
        self.event(m, 0, addr, extract(cmd0, 60, 4) as u16 | EVENT_ILLEGAL_COMMAND_ERROR);
    }

    /// `amdvi_log_illegaldevtab_error()`.
    fn log_illegaldevtab_error(&mut self, m: &Dma<'_>, devid: u16, addr: u64) {
        self.event(m, devid, addr, EVENT_ILLEGAL_DEVTAB_ENTRY);
    }

    /// `amdvi_log_pagetab_error()`.
    fn log_pagetab_error(&mut self, m: &Dma<'_>, devid: u16, addr: u64) {
        self.event(m, devid, addr, EVENT_PAGE_TAB_HW_ERROR);
        self.target_abort = true;
    }

    fn iotlb_reset(&mut self) {
        self.iotlb.clear();
    }

    /// `amdvi_update_iotlb()`.
    fn update_iotlb(&mut self, devid: u16, gpa: u64, x: &Xlate, domid: u16) {
        if x.perm == 0 {
            return;
        }
        if self.iotlb.len() >= IOTLB_MAX_SIZE {
            self.iotlb_reset();
        }
        let entry = IotlbEntry {
            domid,
            devid,
            perms: x.perm,
            translated_addr: x.translated,
            page_mask: x.mask,
        };
        self.iotlb.insert((gpa >> 12, devid), entry);
    }

    /// `amdvi_validate_dte()`.
    fn validate_dte(&mut self, m: &Dma<'_>, devid: u16, dte: &[u64; 4]) -> bool {
        let addr = self.devtab + u64::from(devid) * DEVTAB_ENTRY_SIZE;
        // AMDVI_DTE_QUAD1_RESERVED is 0.
        if dte[0] & DTE_QUAD0_RESERVED != 0
            || dte[2] & DTE_QUAD2_RESERVED != 0
            || dte[3] & DTE_QUAD3_RESERVED != 0
        {
            self.log_illegaldevtab_error(m, devid, addr);
            return false;
        }
        // A page table root without host address translation is illegal.
        let root = (dte[0] & BASE_MASK) >> 12;
        if root != 0 && !self.conf.dma_translation {
            self.log_illegaldevtab_error(m, devid, addr);
            return false;
        }
        true
    }

    /// `amdvi_get_dte()`.
    fn get_dte(&mut self, m: &Dma<'_>, devid: u16) -> Option<[u64; 4]> {
        let addr = self.devtab + u64::from(devid) * DEVTAB_ENTRY_SIZE;
        let Some(dte) = m.read_quads::<4>(addr) else {
            self.log_devtab_error(m, devid, addr, 0);
            return None;
        };
        self.validate_dte(m, devid, &dte).then_some(dte)
    }

    /// `amdvi_as_to_dte()`.
    fn as_to_dte(&mut self, m: &Dma<'_>, devid: u16) -> Result<[u64; 4], DteError> {
        let dte = self.get_dte(m, devid).ok_or(DteError::Retrieve)?;
        if dte[0] & DEV_VALID == 0 {
            return Err(DteError::NotValid);
        }
        if dte[0] & DEV_TRANSLATION_VALID == 0 {
            return Err(DteError::NoTranslation);
        }
        Ok(dte)
    }

    /// `amdvi_get_pte_entry()`: all ones when the read fails.
    fn get_pte_entry(&mut self, m: &Dma<'_>, pte_addr: u64, devid: u16) -> u64 {
        match m.read_quads::<1>(pte_addr) {
            Some([pte]) => pte,
            None => {
                self.log_pagetab_error(m, devid, pte_addr);
                u64::MAX
            }
        }
    }

    /// `fetch_pte()`: the PTE and page size, or `None` when the walk fails. `devfn` is what
    /// QEMU passes as the device ID of a page table error.
    fn fetch_pte(&mut self, m: &Dma<'_>, devfn: u16, address: u64, dte: u64) -> Option<(u64, u64)> {
        // amdvi_get_top_pt_level_and_perms()
        let mut perms = get_perms(dte);
        if perms == 0 {
            return None;
        }
        let mut level = ((dte >> DEV_MODE_RSHIFT) & DEV_MODE_MASK) as u32;
        if level == 0 || level > 6 || address > level_max_addr(level) {
            return None;
        }

        let mut pte = self.get_pte_entry(m, next_pte_addr(dte, level, address), devfn);
        if pte == u64::MAX {
            return None;
        }
        let mut page_size = 1u64 << level_shift(level);
        loop {
            // Permission bits are ANDed at every level, including the DTE.
            perms &= get_perms(pte);
            if perms == 0 || pte & PTE_PR == 0 {
                return Some((pte, page_size));
            }
            let next = pte_next_level(pte);
            if next == 0 || next == 7 {
                break;
            }
            if level <= next {
                return None;
            }
            level = next;
            page_size = 1u64 << level_shift(level);
            pte = self.get_pte_entry(m, next_pte_addr(pte, level, address), devfn);
            if pte == u64::MAX {
                return None;
            }
        }
        if pte_next_level(pte) == 7 {
            // PTE_LARGE_PAGE_SIZE(), which is 0 here where QEMU shifts past 64 bits.
            page_size = 1u64.checked_shl(1 + (pte | 0xfff).trailing_ones()).unwrap_or(0);
        }
        Some((pte, page_size))
    }

    /// `amdvi_page_walk()`: the translation, or `None` for a fault.
    fn page_walk(
        &mut self,
        m: &Dma<'_>,
        devid: u16,
        dte: &[u64; 4],
        perms: u64,
        addr: u64,
    ) -> Option<Xlate> {
        if dte[0] & DEV_TRANSLATION_VALID == 0 {
            return None;
        }
        let mode = (dte[0] >> DEV_MODE_RSHIFT) & DEV_MODE_MASK;
        if mode >= 7 {
            return None;
        }
        if mode == 0 {
            return Some(Xlate::identity(addr, get_perms(dte[0])));
        }
        // QEMU walks and faults with the devfn of the address space, not its device ID.
        let devfn = devid & 0xff;
        match self.fetch_pte(m, devfn, addr, dte[0]) {
            Some((pte, pagesize)) if pte & PTE_PR != 0 && perms == perms & get_perms(pte) => {
                let page_mask = !pagesize.wrapping_sub(1);
                Some(Xlate {
                    iova: addr & page_mask,
                    translated: (pte & BASE_MASK) & page_mask,
                    mask: !page_mask,
                    perm: get_perms(pte),
                })
            }
            _ => {
                self.page_fault(m, devfn, addr, perms as u16);
                None
            }
        }
    }

    /// `amdvi_do_translate()`.
    fn do_translate(&mut self, m: &Dma<'_>, devid: u16, addr: u64, is_write: bool) -> Xlate {
        if let Some(e) = self.iotlb.get(&(addr >> 12, devid)) {
            return Xlate {
                iova: addr & !e.page_mask,
                translated: e.translated_addr,
                mask: e.page_mask,
                perm: e.perms,
            };
        }
        let dte = match self.as_to_dte(m, devid) {
            Ok(dte) => dte,
            Err(DteError::NotValid) => return Xlate::identity(addr, PERM_READ | PERM_WRITE),
            Err(_) => return Xlate::fault(addr),
        };
        let perms = if is_write { PERM_WRITE } else { PERM_READ };
        let x = self.page_walk(m, devid, &dte, perms, addr).unwrap_or(Xlate::fault(addr));
        self.update_iotlb(devid, addr, &x, (dte[1] & DEV_DOMID_ID_MASK) as u16);
        x
    }

    /// `amdvi_translate()` behind `amdvi_switch_address_space()`: pass-through unless
    /// `dma-remap=on` turned translation on for the device.
    fn translate(&mut self, m: &Dma<'_>, devid: u16, addr: u64, is_write: bool) -> Xlate {
        if !(self.conf.dma_remap && self.translating.contains(&devid)) {
            return Xlate::identity(addr, PERM_READ | PERM_WRITE);
        }
        if !self.enabled {
            return Xlate::identity(addr, PERM_READ | PERM_WRITE);
        }
        if (INT_ADDR_FIRST..=INT_ADDR_LAST).contains(&addr) {
            return Xlate::identity(addr, PERM_WRITE);
        }
        self.do_translate(m, devid, addr, is_write)
    }

    /// `amdvi_completion_wait()`.
    fn completion_wait(&mut self, m: &Dma<'_>, cmd: [u64; 2]) {
        let addr = extract(cmd[0], 3, 49) << 3;
        if extract(cmd[0], 52, 8) != 0 {
            self.log_illegalcom_error(m, cmd[0]);
        }
        if extract(cmd[0], 0, 1) != 0 {
            // QEMU only traces a failed store.
            let _ = m.write(addr, &cmd[1].to_le_bytes());
        }
        if extract(cmd[0], 1, 1) != 0 {
            self.orq(MMIO_STATUS, STATUS_COMP_INT);
            self.generate_msi_interrupt();
        }
    }

    /// `amdvi_update_addr_translation_mode()`.
    fn update_addr_translation_mode(&mut self, m: &Dma<'_>, devid: u16) {
        if !self.spaces.contains(&devid) {
            return;
        }
        match self.as_to_dte(m, devid) {
            Ok(dte) if (dte[0] >> DEV_MODE_RSHIFT) & DEV_MODE_MASK == 0 => {
                self.translating.remove(&devid);
            }
            Ok(_) => {
                self.translating.insert(devid);
            }
            Err(DteError::NotValid) => {
                self.translating.remove(&devid);
            }
            // Isolate the device when its entry cannot be read or has no page table.
            Err(DteError::Retrieve | DteError::NoTranslation) => {
                self.translating.insert(devid);
            }
        }
    }

    /// `amdvi_inval_devtab_entry()`.
    fn inval_devtab_entry(&mut self, m: &Dma<'_>, cmd: [u64; 2]) {
        let devid = extract(cmd[0], 0, 16) as u16;
        if extract(cmd[0], 16, 44) != 0 || cmd[1] != 0 {
            self.log_illegalcom_error(m, cmd[0]);
            return;
        }
        if self.conf.dma_remap {
            self.update_addr_translation_mode(m, devid);
        }
    }

    /// `amdvi_complete_ppr()`.
    fn complete_ppr(&mut self, m: &Dma<'_>, cmd: [u64; 2]) {
        if extract(cmd[0], 16, 16) != 0
            || extract(cmd[0], 52, 8) != 0
            || extract(cmd[1], 0, 2) != 0
            || extract(cmd[1], 3, 29) != 0
            || extract(cmd[1], 48, 16) != 0
        {
            self.log_illegalcom_error(m, cmd[0]);
        }
    }

    /// `amdvi_inval_all()`.
    fn inval_all(&mut self, m: &Dma<'_>, cmd: [u64; 2]) {
        if extract(cmd[0], 0, 60) != 0 || cmd[1] != 0 {
            self.log_illegalcom_error(m, cmd[0]);
        }
        self.iotlb_reset();
    }

    /// `amdvi_inval_pages()`. Without notifiers, `amdvi_sync_domain()` has nothing to do,
    /// but the size is decoded as QEMU does.
    fn inval_pages(&mut self, m: &Dma<'_>, cmd: [u64; 2]) {
        let domid = extract(cmd[0], 32, 16) as u16;
        if extract(cmd[0], 20, 12) != 0
            || extract(cmd[0], 48, 12) != 0
            || extract(cmd[1], 3, 9) != 0
        {
            self.log_illegalcom_error(m, cmd[0]);
        }
        self.iotlb.retain(|_, e| e.domid != domid);
        let _ = decode_invalidation_size(extract(cmd[1], 12, 52) << 12, extract(cmd[1], 0, 3));
    }

    /// `amdvi_prefetch_pages()`.
    fn prefetch_pages(&mut self, m: &Dma<'_>, cmd: [u64; 2]) {
        if extract(cmd[0], 16, 8) != 0
            || extract(cmd[0], 52, 8) != 0
            || extract(cmd[1], 1, 1) != 0
            || extract(cmd[1], 3, 1) != 0
            || extract(cmd[1], 5, 7) != 0
        {
            self.log_illegalcom_error(m, cmd[0]);
        }
    }

    /// `amdvi_inval_inttable()`. There is no interrupt entry cache to drop.
    fn inval_inttable(&mut self, m: &Dma<'_>, cmd: [u64; 2]) {
        if extract(cmd[0], 16, 44) != 0 || cmd[1] != 0 {
            self.log_illegalcom_error(m, cmd[0]);
        }
    }

    /// `iommu_inval_iotlb()`.
    fn inval_iotlb(&mut self, m: &Dma<'_>, cmd: [u64; 2]) {
        let devid = extract(cmd[0], 0, 16) as u16;
        if extract(cmd[1], 1, 1) != 0 || extract(cmd[1], 3, 1) != 0 || extract(cmd[1], 6, 6) != 0 {
            self.log_illegalcom_error(m, cmd[0]);
            return;
        }
        if extract(cmd[1], 0, 1) != 0 {
            self.iotlb.retain(|_, e| e.devid != devid);
        } else {
            let addr = extract(cmd[1], 12, 52) << 12;
            self.iotlb.remove(&(addr >> 12, devid));
        }
    }

    /// `amdvi_cmdbuf_exec()`.
    fn cmdbuf_exec(&mut self, m: &Dma<'_>) {
        let addr = self.cmdbuf + u64::from(self.cmdbuf_head);
        let Some(cmd) = m.read_quads::<2>(addr) else {
            self.log_command_error(m, addr);
            return;
        };
        match extract(cmd[0], 60, 4) {
            CMD_COMPLETION_WAIT => self.completion_wait(m, cmd),
            CMD_INVAL_DEVTAB_ENTRY => self.inval_devtab_entry(m, cmd),
            CMD_INVAL_AMDVI_PAGES => self.inval_pages(m, cmd),
            CMD_INVAL_IOTLB_PAGES => self.inval_iotlb(m, cmd),
            CMD_INVAL_INTR_TABLE => self.inval_inttable(m, cmd),
            CMD_PREFETCH_AMDVI_PAGES => self.prefetch_pages(m, cmd),
            CMD_COMPLETE_PPR_REQUEST => self.complete_ppr(m, cmd),
            CMD_INVAL_AMDVI_ALL => self.inval_all(m, cmd),
            _ => self.log_illegalcom_error(m, cmd[0]),
        }
    }

    /// `amdvi_cmdbuf_run()`.
    fn cmdbuf_run(&mut self, m: &Dma<'_>) {
        // An empty buffer would wrap the head to 0 forever.
        if !self.cmdbuf_enabled || self.cmdbuf_len == 0 {
            return;
        }
        while self.cmdbuf_head != self.cmdbuf_tail {
            self.cmdbuf_exec(m);
            self.cmdbuf_head += COMMAND_SIZE;
            if u64::from(self.cmdbuf_head) >= self.cmdbuf_len * u64::from(COMMAND_SIZE) {
                self.cmdbuf_head = 0;
            }
            self.writeq_raw(MMIO_COMMAND_HEAD, u64::from(self.cmdbuf_head));
        }
    }

    /// `amdvi_mmio_read()`.
    fn mmio_read(&self, addr: u64, size: usize) -> u64 {
        if addr + size as u64 > AMDVI_MMIO_SIZE {
            return u64::MAX;
        }
        match size {
            2 | 4 | 8 => Self::load(&self.mmior, addr, size),
            _ => u64::MAX,
        }
    }

    /// `amdvi_mmio_reg_write()`.
    fn reg_write(&mut self, size: usize, val: u64, addr: u64) {
        if matches!(size, 2 | 4 | 8) {
            self.write_masked(addr, size, val);
        }
    }

    /// `amdvi_handle_control_write()`.
    fn handle_control_write(&mut self, m: &Dma<'_>) {
        let control = self.readq(MMIO_CONTROL);
        self.enabled = control & CONTROL_AMDVIEN != 0;
        self.evtlog_enabled = self.enabled && control & CONTROL_EVENTLOGEN != 0;
        self.evtlog_intr = control & CONTROL_EVENTINTEN != 0;
        self.cmdbuf_enabled = self.enabled && control & CONTROL_CMDBUFLEN != 0;
        self.ga_enabled = control & CONTROL_GAEN != 0;
        self.xten = control & CONTROL_XTEN != 0 && self.conf.xtsup && self.ga_enabled;
        self.intcapxten = control & CONTROL_INTCAPXTEN != 0 && self.conf.xtsup;

        if self.cmdbuf_enabled {
            self.orq(MMIO_STATUS, STATUS_CMDBUF_RUN);
        } else {
            self.andq(MMIO_STATUS, !STATUS_CMDBUF_RUN);
        }
        if self.evtlog_enabled {
            self.orq(MMIO_STATUS, STATUS_EVT_RUN);
        } else {
            self.andq(MMIO_STATUS, !STATUS_EVT_RUN);
        }
        self.cmdbuf_run(m);
    }

    /// `amdvi_mmio_write()`.
    fn mmio_write(&mut self, m: &Dma<'_>, addr: u64, val: u64, size: usize) {
        let offset = addr & 7;
        if addr + size as u64 > AMDVI_MMIO_SIZE {
            return;
        }
        let reg = addr & !7;
        match reg {
            MMIO_CONTROL => {
                self.reg_write(size, val, addr);
                self.handle_control_write(m);
            }
            MMIO_DEVICE_TABLE => {
                self.reg_write(size, val, addr);
                // Only once the high half is written, as the Linux driver does.
                if offset != 0 || size == 8 {
                    let v = self.readq(MMIO_DEVICE_TABLE);
                    self.devtab = v & BASE_MASK;
                    self.devtab_len =
                        ((v & DEVTAB_SIZE_MASK) + 1) * (DEVTAB_SIZE_UNIT / DEVTAB_ENTRY_SIZE);
                }
            }
            MMIO_COMMAND_HEAD => {
                self.reg_write(size, val, addr);
                let len_mask = (self.cmdbuf_len * u64::from(COMMAND_SIZE)).wrapping_sub(1);
                self.cmdbuf_head = (self.readq(MMIO_COMMAND_HEAD) & PTR_MASK & len_mask) as u32;
                self.cmdbuf_run(m);
            }
            MMIO_COMMAND_BASE => {
                self.reg_write(size, val, addr);
                if offset != 0 || size == 8 {
                    self.cmdbuf = self.readq(MMIO_COMMAND_BASE) & BASE_MASK;
                    self.cmdbuf_len = 1 << (self.mmior[MMIO_COMMAND_BASE as usize + 7] & 0x0f);
                    self.cmdbuf_head = 0;
                    self.cmdbuf_tail = 0;
                }
            }
            MMIO_COMMAND_TAIL => {
                self.reg_write(size, val, addr);
                let len_mask = (self.cmdbuf_len * u64::from(COMMAND_SIZE)).wrapping_sub(1);
                self.cmdbuf_tail = (self.readq(MMIO_COMMAND_TAIL) & PTR_MASK & len_mask) as u32;
                self.cmdbuf_run(m);
            }
            MMIO_EVENT_BASE => {
                self.reg_write(size, val, addr);
                // Not while the event log interrupt is pending.
                if self.readq(MMIO_STATUS) & STATUS_EVENT_INT == 0 {
                    self.evtlog = self.readq(MMIO_EVENT_BASE) & BASE_MASK;
                    self.evtlog_len = 1 << (self.mmior[MMIO_EVENT_BASE as usize + 7] & 0x0f);
                    self.evtlog_tail = 0;
                    self.evtlog_head = 0;
                    self.writeq_raw(MMIO_EVENT_HEAD, 0);
                    self.writeq_raw(MMIO_EVENT_TAIL, 0);
                }
            }
            MMIO_EVENT_HEAD => {
                self.reg_write(size, val, addr);
                self.evtlog_head = (self.readq(MMIO_EVENT_HEAD) & PTR_MASK) as u32;
            }
            MMIO_EVENT_TAIL => {
                self.reg_write(size, val, addr);
                self.evtlog_tail = (self.readq(MMIO_EVENT_TAIL) & PTR_MASK) as u32;
            }
            // QEMU keeps the exclusion limit and the PPR log pointers but never uses them.
            MMIO_EXCL_LIMIT | MMIO_PPR_BASE | MMIO_PPR_HEAD | MMIO_PPR_TAIL | MMIO_STATUS
            | MMIO_XT_GEN_INTR => self.reg_write(size, val, addr),
            _ => {}
        }
    }

    /// `amdvi_get_irte()` and `amdvi_int_remap_legacy()`.
    fn int_remap_legacy(&mut self, m: &Dma<'_>, data: u32, dte: &[u64; 4]) -> Option<X86Irq> {
        let addr = (dte[2] & IR_PHYS_ADDR_MASK) + (u64::from(data & IRTE_OFFSET) << 2);
        let mut b = [0u8; 4];
        if !m.read(addr, &mut b) {
            return None;
        }
        let irte = u32::from_le_bytes(b);
        if irte & 1 == 0 {
            return None;
        }
        if (irte >> 7) & 1 != 0 {
            GUEST_MODE.report(|| "guest mode is not zero".into());
            return None;
        }
        let int_type = (irte >> 2) & 7;
        if int_type > IOAPIC_INT_TYPE_ARBITRATED {
            RESERVED_INT_TYPE.report(|| "reserved int_type".into());
            return None;
        }
        Some(X86Irq {
            delivery_mode: int_type,
            vector: (irte >> 16) & 0xff,
            dest_mode: (irte >> 6) & 1,
            redir_hint: (irte >> 5) & 1,
            dest: (irte >> 8) & 0xff,
        })
    }

    /// `amdvi_get_irte_ga()` and `amdvi_int_remap_ga()`.
    fn int_remap_ga(&mut self, m: &Dma<'_>, data: u32, dte: &[u64; 4]) -> Option<X86Irq> {
        let addr = (dte[2] & IR_PHYS_ADDR_MASK) + (u64::from(data & IRTE_OFFSET) << 4);
        let [lo, hi] = m.read_quads::<2>(addr)?;
        if lo & 1 == 0 {
            return None;
        }
        if (lo >> 7) & 1 != 0 {
            GUEST_MODE.report(|| "guest mode is not zero".into());
            return None;
        }
        let int_type = extract(lo, 2, 3) as u32;
        if int_type > IOAPIC_INT_TYPE_ARBITRATED {
            RESERVED_INT_TYPE_GA.report(|| "reserved int_type is set".into());
            return None;
        }
        let dest = if self.xten {
            extract(lo, 8, 24) | extract(hi, 56, 8) << 24
        } else {
            extract(lo, 8, 24) & 0xff
        };
        Some(X86Irq {
            delivery_mode: int_type,
            vector: extract(hi, 0, 8) as u32,
            dest_mode: extract(lo, 6, 1) as u32,
            redir_hint: extract(lo, 5, 1) as u32,
            dest: dest as u32,
        })
    }

    /// `amdvi_int_remap_msi()`: the message to send in place of `addr` and `data`, or `None`
    /// for an error.
    fn int_remap_msi(&mut self, m: &Dma<'_>, sid: u16, addr: u64, data: u32) -> Option<(u64, u32)> {
        let sid = if sid == SID_INVALID { AMDVI_IOAPIC_SB_DEVID } else { sid };
        if self.devtab_len == 0 {
            return Some((addr, data));
        }
        let dte = self.get_dte(m, sid)?;
        if dte[2] & IR_REMAP_ENABLE == 0 {
            return Some((addr, data));
        }
        if !self.conf.intr_supported() {
            return None;
        }
        if addr < INT_ADDR_FIRST || addr + 4 > INT_ADDR_LAST + 1 {
            return None;
        }
        let pass = match (data >> 8) & 7 {
            IOAPIC_INT_TYPE_FIXED | IOAPIC_INT_TYPE_ARBITRATED => {
                // __amdvi_int_remap_msi()
                let irq = match (dte[2] >> IR_INTCTL_SHIFT) & 3 {
                    // QEMU copies the message and then overwrites it with the empty IRQ.
                    IR_INTCTL_PASS => X86Irq::default(),
                    IR_INTCTL_REMAP if self.ga_enabled => self.int_remap_ga(m, data, &dte)?,
                    IR_INTCTL_REMAP => self.int_remap_legacy(m, data, &dte)?,
                    // IR_INTCTL_ABORT and the reserved value.
                    _ => return None,
                };
                return Some(irq.to_msi());
            }
            IOAPIC_INT_TYPE_SMI => {
                error_report("SMI is not supported!");
                return None;
            }
            IOAPIC_INT_TYPE_NMI => dte[2] & DEV_NMI_PASS_MASK,
            IOAPIC_INT_TYPE_INIT => dte[2] & DEV_INT_PASS_MASK,
            IOAPIC_INT_TYPE_EINT => dte[2] & DEV_EINT_PASS_MASK,
            _ => return None,
        };
        // Physical destination mode only.
        if (addr >> 2) & 1 != 0 || pass == 0 {
            return None;
        }
        Some((addr, data))
    }
}

/// `amdvi_decode_invalidation_size()`.
fn decode_invalidation_size(addr: u64, flags: u64) -> u64 {
    if flags & 1 == 0 {
        return PAGE_SIZE;
    }
    let fzbit = (addr | 0xfff).trailing_ones();
    if fzbit >= 51 { INV_ALL_PAGES } else { 1 << (fzbit + 1) }
}

thread_local! {
    /// Set while this thread holds the IOMMU state, so that an access that comes back to the
    /// IOMMU, such as a buffer placed over its own registers, fails instead of deadlocking.
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

/// The `amd-iommu` device.
pub struct AmdIommu {
    state: Mutex<State>,
    memory: Weak<AddressSpace>,
    msi: OnceLock<MsiSender>,
    pci: OnceLock<Arc<dyn AmdviPci>>,
}

impl fmt::Debug for AmdIommu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AmdIommu").finish_non_exhaustive()
    }
}

impl AmdIommu {
    /// An IOMMU with the configuration that [`AmdIommuConf::realize`] gave, reading its
    /// tables from `memory`.
    pub fn new(conf: AmdIommuConf, memory: &Arc<AddressSpace>) -> Arc<Self> {
        Arc::new(AmdIommu {
            state: Mutex::new(State::new(conf)),
            memory: Arc::downgrade(memory),
            msi: OnceLock::new(),
            pci: OnceLock::new(),
        })
    }

    /// Sets what the IOMMU's own interrupts go to. They are plain MSIs to the APICs, which
    /// interrupt remapping does not see.
    pub fn set_msi_sender(&self, send: impl Fn(u64, u32) + Send + Sync + 'static) {
        let _ = self.msi.set(Box::new(send));
    }

    /// Sets the `AMDVI-PCI` function.
    pub fn set_pci(&self, pci: Arc<dyn AmdviPci>) {
        let _ = self.pci.set(pci);
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Runs `f` on the state, then raises the interrupts and the target abort it asked for
    /// once the lock is dropped.
    fn with_state<R>(&self, f: impl FnOnce(&mut State, &Dma<'_>) -> R) -> R {
        let memory = self.memory.upgrade();
        let (r, irqs, abort) = {
            let _engaged = Engaged::enter();
            let mut s = self.lock();
            let r = f(&mut s, &Dma(memory.as_deref()));
            (r, std::mem::take(&mut s.irqs), std::mem::take(&mut s.target_abort))
        };
        let pci = self.pci.get();
        if abort {
            if let Some(pci) = pci {
                pci.signal_target_abort();
            }
        }
        for irq in irqs {
            let msg = match irq {
                Irq::Msi(addr, data) => Some((addr, data)),
                Irq::Pci => pci.and_then(|p| p.msi_message()),
            };
            if let (Some((addr, data)), Some(send)) = (msg, self.msi.get()) {
                send(addr, data);
            }
        }
        r
    }

    /// `amdvi_sysbus_reset()`. The board resets the MSI of the PCI function.
    pub fn system_reset(&self) {
        self.lock().reset();
    }

    /// The configuration, for the IVRS table.
    pub fn conf(&self) -> AmdIommuConf {
        self.lock().conf.clone()
    }

    /// The register frame, to map at [`AMDVI_BASE_ADDR`].
    pub fn mmio_ops(self: &Arc<Self>) -> Arc<dyn MmioOps> {
        Arc::new(AmdviMmio(Arc::clone(self)))
    }

    /// The IOMMU region callbacks of the device with device ID `devid`, which is
    /// `PCI_BUILD_BDF(bus, devfn)`, `amdvi_host_dma_iommu()`.
    pub fn device_ops(self: &Arc<Self>, devid: u16) -> Arc<dyn IommuOps> {
        self.lock().spaces.insert(devid);
        Arc::new(AmdviDevice { iommu: Arc::clone(self), devid })
    }

    /// `amdvi_mem_ir_write()`: remaps an MSI from the device with device ID `sid`, or from
    /// the IOAPIC with [`SID_INVALID`]. `None` drops the interrupt.
    pub fn remap_msi(&self, sid: u16, addr: u64, data: u32) -> Option<(u64, u32)> {
        if engaged() {
            return None;
        }
        let r = self.with_state(|s, m| s.int_remap_msi(m, sid, addr, data));
        if r.is_none() {
            let sid = if sid == SID_INVALID { AMDVI_IOAPIC_SB_DEVID } else { sid };
            REMAP_FAILED.report(|| format!("failed to remap interrupt from devid 0x{sid:x}"));
        }
        r
    }

    /// `amdvi_translate()`.
    pub fn translate(&self, devid: u16, addr: u64, flag: IommuAccessFlags) -> IommuTlbEntry {
        let target_as = self.memory.upgrade();
        if engaged() {
            return IommuTlbEntry {
                target_as,
                iova: addr,
                translated_addr: 0,
                addr_mask: !0,
                perm: IommuAccessFlags::NONE,
            };
        }
        let is_write = flag.allows(IommuAccessFlags::WO);
        let x = self.with_state(|s, m| s.translate(m, devid, addr, is_write));
        IommuTlbEntry {
            target_as,
            iova: x.iova,
            translated_addr: x.translated,
            addr_mask: x.mask,
            perm: perm_flags(x.perm),
        }
    }
}

/// The IOMMU region of one device, `AMDVIAddressSpace`.
struct AmdviDevice {
    iommu: Arc<AmdIommu>,
    devid: u16,
}

impl IommuOps for AmdviDevice {
    fn translate(&self, addr: u64, flag: IommuAccessFlags, _iommu_idx: u32) -> IommuTlbEntry {
        self.iommu.translate(self.devid, addr, flag)
    }
}

/// The register frame, `mmio_mem_ops`.
struct AmdviMmio(Arc<AmdIommu>);

impl fmt::Debug for AmdviMmio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AmdviMmio")
    }
}

impl MmioOps for AmdviMmio {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        if engaged() {
            return Err(MemTxResult::ACCESS_ERROR);
        }
        Ok(self.0.lock().mmio_read(offset, size.bytes() as usize))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        if engaged() {
            return Err(MemTxResult::ACCESS_ERROR);
        }
        self.0.with_state(|s, m| s.mmio_write(m, offset, value, size.bytes() as usize));
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 8)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 8)
    }
}

#[cfg(test)]
mod tests;
