// SPDX-License-Identifier: GPL-2.0-or-later

//! The `riscv-iommu` device, from hw/riscv/riscv-iommu.c, hw/riscv/riscv-iommu-hpm.c and
//! hw/riscv/riscv-iommu-bits.h.
//!
//! This is the IOMMU core that `riscv-iommu-pci` and `riscv-iommu-device` wrap. It has the
//! register file, the command, fault and page request queues, the device directory and
//! process directory walks, the first stage and G-stage page table walks with nested
//! translation, the MSI page tables with basic and MRIF entries, the translation request
//! (debug) registers and the performance monitor with its cycle counter. A front end picks
//! the interrupt generation support and the vectors software may use in `ICVEC` when it
//! builds the IOMMU, and gets the interrupt vectors through [`RiscvIommu::set_notify`].
//!
//! Each device gets an IOMMU region from [`RiscvIommu::device_ops`], keyed by its routing ID,
//! and the process ID of a request is its IOMMU index. The MSIs of the PCI devices come in
//! through [`RiscvIommu::remap_msi`].
//!
//! # Differences from QEMU
//!
//! - There are no IOMMU notifiers, so `ATS.INVAL` and `ATS.PRGR` are accepted when `ats` is
//!   on but do nothing.
//! - MSIs reach [`RiscvIommu::remap_msi`] without a DMA write. A plain DMA write that hits
//!   the MSI page table window gets a fault from the IOMMU region, where QEMU sends it to its
//!   trap region, which turns down writes without a requester ID anyway.
//! - There is no VMState.

use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

use ruvm_hw_core::{Clock, Timer};
use ruvm_mem::{
    AccessConstraints, AccessCtx, AccessSize, AddressSpace, IommuAccessFlags, IommuOps,
    IommuTlbEntry, MemResult, MemTxAttrs, MemTxResult, MmioOps,
};

/// `TYPE_RISCV_IOMMU`.
pub const TYPE_RISCV_IOMMU: &str = "riscv-iommu";
/// `TYPE_RISCV_IOMMU_MEMORY_REGION`.
pub const TYPE_RISCV_IOMMU_MEMORY_REGION: &str = "riscv-iommu-mr";
/// `RISCV_IOMMU_REG_SIZE`, the size of the register frame.
pub const RISCV_IOMMU_REG_SIZE: u64 = 0x1000;
/// `RISCV_IOMMU_REG_MSI_CONFIG`, where the front ends put their MSI-X table. The core
/// registers end there.
pub const RISCV_IOMMU_REG_MSI_CONFIG: u64 = 0x300;
/// `RISCV_IOMMU_SPEC_DOT_VER`, the default `version`.
pub const RISCV_IOMMU_SPEC_DOT_VER: u32 = 0x10;
/// `RISCV_IOMMU_IOCOUNT_NUM`, the most HPM counters there can be.
pub const RISCV_IOMMU_IOCOUNT_NUM: u8 = 31;

/// `riscv_iommu_igs_modes`, the interrupt generation support in `capabilities.IGS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RiscvIommuIgs {
    /// `RISCV_IOMMU_CAP_IGS_MSI`.
    Msi = 0,
    /// `RISCV_IOMMU_CAP_IGS_WSI`.
    Wsi = 1,
    /// `RISCV_IOMMU_CAP_IGS_BOTH`.
    Both = 2,
}

/// `RISCV_IOMMU_INTR_CQ`, the command queue interrupt.
pub const RISCV_IOMMU_INTR_CQ: u32 = 0;
/// `RISCV_IOMMU_INTR_FQ`, the fault queue interrupt.
pub const RISCV_IOMMU_INTR_FQ: u32 = 1;
/// `RISCV_IOMMU_INTR_PM`, the performance monitor interrupt.
pub const RISCV_IOMMU_INTR_PM: u32 = 2;
/// `RISCV_IOMMU_INTR_PQ`, the page request queue interrupt.
pub const RISCV_IOMMU_INTR_PQ: u32 = 3;

const REG_SIZE: usize = RISCV_IOMMU_REG_SIZE as usize;
const LIMIT_CACHE_CTX: usize = 1 << 7;
const LIMIT_CACHE_IOT: u32 = 1 << 20;

/// `GENMASK_ULL(h, l)`.
const fn genmask(h: u32, l: u32) -> u64 {
    (!0u64 >> (63 - h + l)) << l
}

/// `get_field()`.
const fn get_field(v: u64, mask: u64) -> u64 {
    (v & mask) >> mask.trailing_zeros()
}

/// `set_field()`.
const fn set_field(v: u64, mask: u64, val: u64) -> u64 {
    (v & !mask) | ((val << mask.trailing_zeros()) & mask)
}

/// `PPN_PHYS()`.
const fn ppn_phys(ppn: u64) -> u64 {
    ppn << 12
}

/// `PPN_DOWN()`.
const fn ppn_down(addr: u64) -> u64 {
    addr >> 12
}

const PAGE_OFFSET: u64 = 0xfff;
const PPN_FIELD: u64 = genmask(53, 10);
const ATP_PPN: u64 = genmask(43, 0);
const ATP_MODE: u64 = genmask(63, 60);

const REG_CAP: u64 = 0x0000;
const REG_FCTL: u64 = 0x0008;
const REG_DDTP: u64 = 0x0010;
const REG_CQB: u64 = 0x0018;
const REG_CQH: u64 = 0x0020;
const REG_CQT: u64 = 0x0024;
const REG_FQB: u64 = 0x0028;
const REG_FQH: u64 = 0x0030;
const REG_FQT: u64 = 0x0034;
const REG_PQB: u64 = 0x0038;
const REG_PQH: u64 = 0x0040;
const REG_PQT: u64 = 0x0044;
const REG_CQCSR: u64 = 0x0048;
const REG_FQCSR: u64 = 0x004c;
const REG_PQCSR: u64 = 0x0050;
const REG_IPSR: u64 = 0x0054;
const REG_IOCOUNTOVF: u64 = 0x0058;
const REG_IOCOUNTINH: u64 = 0x005c;
const REG_IOHPMCYCLES: u64 = 0x0060;
const REG_IOHPMCTR_BASE: u64 = 0x0068;
const REG_IOHPMEVT_BASE: u64 = 0x0160;
const REG_TR_REQ_IOVA: u64 = 0x0258;
const REG_TR_REQ_CTL: u64 = 0x0260;
const REG_TR_RESPONSE: u64 = 0x0268;
const REG_ICVEC: u64 = 0x02f8;

/// `RISCV_IOMMU_REG_IOHPMEVT(n)`.
const fn reg_iohpmevt(n: u64) -> u64 {
    REG_IOHPMEVT_BASE + n * 8
}

const CAP_SV32: u64 = 1 << 8;
const CAP_SV39: u64 = 1 << 9;
const CAP_SV48: u64 = 1 << 10;
const CAP_SV57: u64 = 1 << 11;
const CAP_SVRSW60T59B: u64 = 1 << 14;
const CAP_SV32X4: u64 = 1 << 16;
const CAP_SV39X4: u64 = 1 << 17;
const CAP_SV48X4: u64 = 1 << 18;
const CAP_SV57X4: u64 = 1 << 19;
const CAP_MSI_FLAT: u64 = 1 << 22;
const CAP_MSI_MRIF: u64 = 1 << 23;
const CAP_AMO_HWAD: u64 = 1 << 24;
const CAP_ATS: u64 = 1 << 25;
const CAP_T2GPA: u64 = 1 << 26;
const CAP_IGS: u64 = genmask(29, 28);
const CAP_HPM: u64 = 1 << 30;
const CAP_DBG: u64 = 1 << 31;
const CAP_PAS: u64 = genmask(37, 32);
const CAP_PD8: u64 = 1 << 38;
const CAP_PD17: u64 = 1 << 39;
const CAP_PD20: u64 = 1 << 40;

const FCTL_WSI: u64 = 1 << 1;
const FCTL_GXL: u64 = 1 << 2;

const DDTP_MODE: u64 = genmask(3, 0);
const DDTP_BUSY: u32 = 1 << 4;
const DDTP_MODE_OFF: u64 = 0;
const DDTP_MODE_BARE: u64 = 1;
const DDTP_MODE_1LVL: u64 = 2;
const DDTP_MODE_2LVL: u64 = 3;
const DDTP_MODE_3LVL: u64 = 4;

const QUEUE_LOGSZ: u64 = genmask(4, 0);
const QUEUE_ENABLE: u32 = 1 << 0;
const QUEUE_INTR_ENABLE: u32 = 1 << 1;
const QUEUE_MEM_FAULT: u32 = 1 << 8;
const QUEUE_OVERFLOW: u32 = 1 << 9;
const QUEUE_ACTIVE: u32 = 1 << 16;
const QUEUE_BUSY: u32 = 1 << 17;
const CQCSR_CMD_TO: u32 = 1 << 9;
const CQCSR_CMD_ILL: u32 = 1 << 10;
const CQCSR_FENCE_W_IP: u32 = 1 << 11;

const IPSR_CIP: u32 = 1 << 0;
const IPSR_FIP: u32 = 1 << 1;
const IPSR_PMIP: u32 = 1 << 2;
const IPSR_PIP: u32 = 1 << 3;

const IOCOUNTOVF_CY: u32 = 1 << 0;
const IOCOUNTINH_CY: u32 = 1 << 0;
const IOHPMCYCLES_COUNTER: u64 = genmask(62, 0);
const IOHPMCYCLES_OVF: u64 = 1 << 63;
const IOHPMEVT_EVENT_ID: u64 = genmask(14, 0);
const IOHPMEVT_DMASK: u64 = 1 << 15;
const IOHPMEVT_PID_PSCID: u64 = genmask(35, 16);
const IOHPMEVT_DID_GSCID: u64 = genmask(59, 36);
const IOHPMEVT_PV_PSCV: u64 = 1 << 60;
const IOHPMEVT_DV_GSCV: u64 = 1 << 61;
const IOHPMEVT_IDT: u64 = 1 << 62;
const IOHPMEVT_OF: u64 = 1 << 63;

const HPMEVENT_INVALID: u32 = 0;
const HPMEVENT_URQ: u32 = 1;
const HPMEVENT_ATS_RQ: u32 = 3;
const HPMEVENT_TLB_MISS: u32 = 4;
const HPMEVENT_DD_WALK: u32 = 5;
const HPMEVENT_PD_WALK: u32 = 6;
const HPMEVENT_S_VS_WALKS: u32 = 7;
const HPMEVENT_G_WALKS: u32 = 8;
const HPMEVENT_MAX: u32 = 9;

const TR_REQ_CTL_GO_BUSY: u64 = 1 << 0;
const TR_REQ_CTL_NW: u64 = 1 << 3;
const TR_REQ_CTL_PID: u64 = genmask(31, 12);
const TR_REQ_CTL_DID: u64 = genmask(63, 40);
const TR_RESPONSE_FAULT: u64 = 1 << 0;

const ICVEC_CIV: u64 = genmask(3, 0);
const ICVEC_FIV: u64 = genmask(7, 4);
const ICVEC_PMIV: u64 = genmask(11, 8);
const ICVEC_PIV: u64 = genmask(15, 12);

const DDTE_VALID: u64 = 1 << 0;

const DC_TC_V: u64 = 1 << 0;
const DC_TC_EN_ATS: u64 = 1 << 1;
const DC_TC_EN_PRI: u64 = 1 << 2;
const DC_TC_T2GPA: u64 = 1 << 3;
const DC_TC_DTF: u64 = 1 << 4;
const DC_TC_PDTV: u64 = 1 << 5;
const DC_TC_PRPR: u64 = 1 << 6;
const DC_TC_GADE: u64 = 1 << 7;
const DC_TC_SADE: u64 = 1 << 8;
const DC_TC_DPE: u64 = 1 << 9;
const DC_TC_SBE: u64 = 1 << 10;
const DC_TC_SXL: u64 = 1 << 11;
const DC_TC_RESERVED: u64 = genmask(23, 12) | genmask(63, 32);
const DC_IOHGATP_GSCID: u64 = genmask(59, 44);
const DC_TA_PSCID: u64 = genmask(31, 12);

/// `RISCV_IOMMU_DC_IOHGATP_MODE_*` and `RISCV_IOMMU_DC_FSC_IOSATP_MODE_*`. Sv32 and Sv39
/// share 8, told apart by `SXL` and `GXL`.
const ATP_MODE_BARE: u64 = 0;
const ATP_MODE_SV39: u64 = 8;
const ATP_MODE_SV48: u64 = 9;
const ATP_MODE_SV57: u64 = 10;
const PDTP_MODE_PD8: u64 = 1;
const PDTP_MODE_PD17: u64 = 2;
const PDTP_MODE_PD20: u64 = 3;

const CMD_OPCODE: u64 = genmask(6, 0);
const CMD_FUNC: u64 = genmask(9, 7);
const CMD_IOTINVAL_AV: u64 = 1 << 10;
const CMD_IOTINVAL_PSCID: u64 = genmask(31, 12);
const CMD_IOTINVAL_PSCV: u64 = 1 << 32;
const CMD_IOTINVAL_GV: u64 = 1 << 33;
const CMD_IOTINVAL_GSCID: u64 = genmask(59, 44);
const CMD_IOFENCE_AV: u64 = 1 << 10;
const CMD_IOFENCE_RESERVED: u64 = genmask(31, 14);
const CMD_IOFENCE_DATA: u64 = genmask(63, 32);
const CMD_IODIR_PID: u64 = genmask(31, 12);
const CMD_IODIR_DV: u64 = 1 << 33;
const CMD_IODIR_DID: u64 = genmask(63, 40);

/// `RISCV_IOMMU_CMD(func, op)`.
const fn cmd(func: u64, op: u64) -> u64 {
    (func << 7) | op
}

const CMD_IOTINVAL_VMA: u64 = cmd(0, 1);
const CMD_IOTINVAL_GVMA: u64 = cmd(1, 1);
const CMD_IOFENCE_C: u64 = cmd(0, 2);
const CMD_IODIR_INVAL_DDT: u64 = cmd(0, 3);
const CMD_IODIR_INVAL_PDT: u64 = cmd(1, 3);
const CMD_ATS_INVAL: u64 = cmd(0, 4);
const CMD_ATS_PRGR: u64 = cmd(1, 4);

const FQ_HDR_CAUSE: u64 = genmask(11, 0);
const FQ_HDR_PID: u64 = genmask(31, 12);
const FQ_HDR_PV: u64 = 1 << 32;
const FQ_HDR_TTYPE: u64 = genmask(39, 34);
const FQ_HDR_DID: u64 = genmask(63, 40);

const PREQ_HDR_PID: u64 = genmask(31, 12);
const PREQ_HDR_PV: u64 = 1 << 32;
const PREQ_HDR_DID: u64 = genmask(63, 40);
const PREQ_PAYLOAD_M: u64 = genmask(2, 0);

const CAUSE_RD_FAULT: u32 = 5;
const CAUSE_WR_FAULT: u32 = 7;
const CAUSE_RD_FAULT_S: u32 = 13;
const CAUSE_WR_FAULT_S: u32 = 15;
const CAUSE_RD_FAULT_VS: u32 = 21;
const CAUSE_WR_FAULT_VS: u32 = 23;
const CAUSE_DMA_DISABLED: u32 = 256;
const CAUSE_DDT_LOAD_FAULT: u32 = 257;
const CAUSE_DDT_INVALID: u32 = 258;
const CAUSE_DDT_MISCONFIGURED: u32 = 259;
const CAUSE_TTYPE_BLOCKED: u32 = 260;
const CAUSE_MSI_LOAD_FAULT: u32 = 261;
const CAUSE_MSI_INVALID: u32 = 262;
const CAUSE_MSI_MISCONFIGURED: u32 = 263;
const CAUSE_PDT_LOAD_FAULT: u32 = 265;
const CAUSE_PDT_INVALID: u32 = 266;
const CAUSE_PDT_MISCONFIGURED: u32 = 267;
const CAUSE_DDT_CORRUPTED: u32 = 268;
const CAUSE_MSI_PT_CORRUPTED: u32 = 270;
const CAUSE_INTERNAL_DP_ERROR: u32 = 272;
const CAUSE_MSI_WR_FAULT: u32 = 273;

const TTYPE_UADDR_RD: u64 = 2;
const TTYPE_UADDR_WR: u64 = 3;
const TTYPE_PCIE_ATS_REQ: u64 = 8;

const DC_MSIPTP_MODE_OFF: u64 = 0;
const DC_MSIPTP_MODE_FLAT: u64 = 1;
const PDTE_VALID: u64 = 1 << 0;
const PC_TA_V: u64 = 1 << 0;
const PC_TA_RESERVED: u64 = genmask(63, 32);
const PC_FSC_RESERVED: u64 = genmask(59, 44);

const MSI_PTE_V: u64 = 1 << 0;
const MSI_PTE_M: u64 = genmask(2, 1);
const MSI_PTE_M_MRIF: u64 = 1;
const MSI_PTE_M_BASIC: u64 = 3;
const MSI_PTE_MRIF_ADDR: u64 = genmask(53, 7);
const MSI_PTE_C: u64 = 1 << 63;
const MSI_MRIF_NID: u64 = genmask(9, 0);
const MSI_MRIF_NID_MSB: u64 = 1 << 60;

const PTE_V: u64 = 0x001;
const PTE_R: u64 = 0x002;
const PTE_W: u64 = 0x004;
const PTE_X: u64 = 0x008;
const PTE_U: u64 = 0x010;
const PTE_A: u64 = 0x040;
const PTE_D: u64 = 0x080;
const PTE_N: u64 = 1 << 63;
/// `PTE_RESERVED(false)` from target/riscv/cpu_bits.h.
const PTE_RESERVED: u64 = 0x1fc0_0000_0000_0000;
const PTE_PPN_SHIFT: u32 = 10;

/// `IOMMUAccessFlags` as bits.
const PERM_NONE: u8 = 0;
const PERM_RO: u8 = 1;
const PERM_WO: u8 = 2;
const PERM_RW: u8 = 3;

fn perm_bits(flag: IommuAccessFlags) -> u8 {
    u8::from(flag.allows(IommuAccessFlags::RO)) | (u8::from(flag.allows(IommuAccessFlags::WO)) << 1)
}

fn perm_flags(perm: u8) -> IommuAccessFlags {
    match perm & PERM_RW {
        PERM_RO => IommuAccessFlags::RO,
        PERM_WO => IommuAccessFlags::WO,
        PERM_RW => IommuAccessFlags::RW,
        _ => IommuAccessFlags::NONE,
    }
}

/// The properties of `riscv-iommu`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RiscvIommuConf {
    /// `version`, the spec version in `capabilities`.
    pub version: u32,
    /// `pas-bits`, the physical address bits in `capabilities`.
    pub pas_bits: u32,
    /// `bus`, ORed into the device IDs as bits 15:8.
    pub bus: u32,
    /// `ioatc-limit`, the most translations the IOTLB keeps. 0 turns it off.
    pub ioatc_limit: u32,
    /// `intremap`, the MSI page tables.
    pub intremap: bool,
    /// `ats`.
    pub ats: bool,
    /// `off`: DMA is turned off out of reset instead of passing through.
    pub off: bool,
    /// `s-stage`.
    pub s_stage: bool,
    /// `g-stage`.
    pub g_stage: bool,
    /// `hpm-counters`, clipped to [`RISCV_IOMMU_IOCOUNT_NUM`].
    pub hpm_counters: u8,
}

impl Default for RiscvIommuConf {
    fn default() -> Self {
        RiscvIommuConf {
            version: RISCV_IOMMU_SPEC_DOT_VER,
            pas_bits: 0,
            bus: 0,
            ioatc_limit: LIMIT_CACHE_IOT,
            intremap: true,
            ats: true,
            off: true,
            s_stage: true,
            g_stage: true,
            hpm_counters: RISCV_IOMMU_IOCOUNT_NUM,
        }
    }
}

/// Memory accesses of the IOMMU itself, through its downstream address space.
struct Dma<'a>(Option<&'a AddressSpace>);

impl Dma<'_> {
    fn read_res(&self, addr: u64, buf: &mut [u8]) -> MemTxResult {
        match self.0 {
            Some(a) => a.read(addr, MemTxAttrs::UNSPECIFIED, buf),
            None => MemTxResult::DECODE_ERROR,
        }
    }

    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.read_res(addr, buf).is_ok()
    }

    fn write(&self, addr: u64, buf: &[u8]) -> bool {
        self.0.is_some_and(|a| a.write(addr, MemTxAttrs::UNSPECIFIED, buf).is_ok())
    }

    fn read_u32(&self, addr: u64) -> Option<u32> {
        let mut b = [0u8; 4];
        self.read(addr, &mut b).then(|| u32::from_le_bytes(b))
    }

    fn read_u64(&self, addr: u64) -> Option<u64> {
        let mut b = [0u8; 8];
        self.read(addr, &mut b).then(|| u64::from_le_bytes(b))
    }
}

/// What the state works with besides itself: the downstream memory, the clock of the cycle
/// counter and its overflow timer.
struct Env<'a> {
    dma: Dma<'a>,
    clock: &'a Clock,
    timer: Option<&'a Timer>,
}

impl Env<'_> {
    /// `get_cycles()`: one cycle is one nanosecond of the virtual clock.
    fn cycles(&self) -> u64 {
        self.clock.get_ns() as u64
    }
}

/// A device translation context, `RISCVIOMMUContext`.
#[derive(Clone, Copy, Debug, Default)]
struct Ctx {
    devid: u32,
    process_id: u32,
    tc: u64,
    ta: u64,
    satp: u64,
    gatp: u64,
    msi_addr_mask: u64,
    msi_addr_pattern: u64,
    msiptp: u64,
}

/// `RISCVIOMMUTransTag`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum TransTag {
    /// Bypass.
    By,
    /// First stage only.
    Ss,
    /// G-stage only.
    Vg,
    /// Nested.
    Vn,
}

/// The key of an IOTLB entry, the fields `riscv_iommu_iot_equal()` compares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct IotKey {
    tag: TransTag,
    gscid: u32,
    pscid: u32,
    iova: u64,
}

/// What an IOTLB entry holds besides its key.
#[derive(Clone, Copy, Debug)]
struct IotEntry {
    phys: u64,
    perm: u8,
}

/// Which IOTLB entries of a tag an `IOTINVAL` drops, the `riscv_iommu_iot_inval_*()`
/// functions.
#[derive(Clone, Copy, Debug)]
struct IotMatch {
    gscid: bool,
    pscid: bool,
    iova: bool,
}

/// Which contexts an `IODIR` drops.
#[derive(Clone, Copy, Debug)]
enum CtxMatch {
    All,
    Devid,
    DevidProcid,
}

/// An `IOMMUTLBEntry` while it is being worked out, with the trap region as a flag.
#[derive(Clone, Copy, Debug)]
struct Tlb {
    iova: u64,
    translated: u64,
    mask: u64,
    perm: u8,
    trap: bool,
}

/// One stage of a page table walk.
#[derive(Clone, Copy, Debug, Default)]
struct Walk {
    step: u32,
    levels: u32,
    ptidxbits: u32,
    ptesize: u32,
}

/// What a register write goes on to do, `riscv_iommu_process_fn`.
#[derive(Clone, Copy, Debug)]
enum Process {
    Ddtp,
    CqTail,
    CqControl,
    FqControl,
    PqControl,
    Dbg,
}

/// `riscv_iommu_pext_u64()`: the bits of `val` where `ext` has ones, packed at the bottom.
fn pext(mut val: u64, mut ext: u64) -> u64 {
    let mut ret = 0;
    let mut rot = 1u64;
    while ext != 0 {
        if ext & 1 != 0 {
            if val & 1 != 0 {
                ret |= rot;
            }
            rot <<= 1;
        }
        val >>= 1;
        ext >>= 1;
    }
    ret
}

/// `riscv_iommu_napot_page_mask()`: the mask of a 64 KiB NAPOT page with its address in
/// `out`, or 0 for the other encodings.
fn napot_page_mask(ppn: u64, addr: u64, out: &mut u64) -> u64 {
    let napot_bits = ppn.trailing_zeros() + 1;
    if napot_bits != 4 {
        return 0;
    }
    let napot_mask = (1u64 << napot_bits) - 1;
    let page_mask = ppn_phys(napot_mask) | PAGE_OFFSET;
    *out = ppn_phys(ppn & !napot_mask) | (addr & page_mask);
    page_mask
}

fn ld(buf: &[u8], off: u64, size: usize) -> u64 {
    let off = off as usize;
    let mut b = [0u8; 8];
    b[..size].copy_from_slice(&buf[off..off + size]);
    u64::from_le_bytes(b)
}

fn st(buf: &mut [u8], off: u64, size: usize, v: u64) {
    let off = off as usize;
    buf[off..off + size].copy_from_slice(&v.to_le_bytes()[..size]);
}

/// The register state and the caches, `RISCVIOMMUState`.
struct State {
    conf: RiscvIommuConf,
    cap: u64,
    /// The features in use. QEMU never updates it from `fctl`, so `GXL` never applies.
    fctl: u64,
    icvec_avail: u64,
    hpm_cntrs: u32,
    has_notify: bool,

    ddtp: u64,
    cq_addr: u64,
    fq_addr: u64,
    pq_addr: u64,
    cq_mask: u32,
    fq_mask: u32,
    pq_mask: u32,

    ctx_cache: HashMap<(u32, u32), Ctx>,
    iot_cache: HashMap<IotKey, IotEntry>,

    regs: Box<[u8]>,
    ro: Box<[u8]>,
    wc: Box<[u8]>,

    hpmcycle_val: u64,
    hpmcycle_prev: u64,
    irq_overflow_left: u64,
    hpm_event_ctr_map: HashMap<u32, u32>,

    /// Interrupt vectors raised while the state was locked.
    irqs: Vec<u32>,
}

impl State {
    /// `riscv_iommu_instance_init()` and `riscv_iommu_realize()`.
    fn new(conf: RiscvIommuConf, igs: RiscvIommuIgs, icvec_avail: u64) -> Self {
        let mut cap = CAP_DBG | CAP_PD8;
        cap = set_field(cap, CAP_IGS, igs as u64);
        cap = set_field(cap, CAP_PAS, u64::from(conf.pas_bits));
        cap |= u64::from(conf.version) & 0xff;
        if conf.intremap {
            cap |= CAP_MSI_FLAT | CAP_MSI_MRIF;
        }
        if conf.ats {
            cap |= CAP_ATS;
        }
        if conf.s_stage {
            cap |= CAP_SV32 | CAP_SV39 | CAP_SV48 | CAP_SV57;
        }
        if conf.g_stage {
            cap |= CAP_SV32X4 | CAP_SV39X4 | CAP_SV48X4 | CAP_SV57X4 | CAP_SVRSW60T59B;
        }
        let hpm_cntrs = u32::from(conf.hpm_counters.min(RISCV_IOMMU_IOCOUNT_NUM));
        if hpm_cntrs > 0 {
            cap |= CAP_HPM;
        }
        let ddtp = if conf.off { DDTP_MODE_OFF } else { DDTP_MODE_BARE };

        let mut s = State {
            conf,
            cap,
            fctl: 0,
            icvec_avail,
            hpm_cntrs,
            has_notify: false,
            ddtp,
            cq_addr: 0,
            fq_addr: 0,
            pq_addr: 0,
            cq_mask: 0,
            fq_mask: 0,
            pq_mask: 0,
            ctx_cache: HashMap::new(),
            iot_cache: HashMap::new(),
            regs: vec![0u8; REG_SIZE].into_boxed_slice(),
            ro: vec![0xffu8; REG_SIZE].into_boxed_slice(),
            wc: vec![0u8; REG_SIZE].into_boxed_slice(),
            hpmcycle_val: 0,
            hpmcycle_prev: 0,
            irq_overflow_left: 0,
            hpm_event_ctr_map: HashMap::new(),
            irqs: Vec::new(),
        };

        s.set64(REG_CAP, cap);
        s.set64(REG_FCTL, 0);
        st(&mut s.ro, REG_FCTL, 8, !(FCTL_GXL | FCTL_WSI));
        st(&mut s.ro, REG_DDTP, 8, !(PPN_FIELD | DDTP_MODE));
        st(&mut s.ro, REG_CQB, 8, !(QUEUE_LOGSZ | PPN_FIELD));
        st(&mut s.ro, REG_FQB, 8, !(QUEUE_LOGSZ | PPN_FIELD));
        st(&mut s.ro, REG_PQB, 8, !(QUEUE_LOGSZ | PPN_FIELD));
        let qwc = u64::from(QUEUE_MEM_FAULT);
        st(&mut s.wc, REG_CQCSR, 4, qwc | u64::from(CQCSR_CMD_TO | CQCSR_CMD_ILL));
        st(&mut s.ro, REG_CQCSR, 4, u64::from(QUEUE_ACTIVE | QUEUE_BUSY));
        st(&mut s.wc, REG_FQCSR, 4, qwc | u64::from(QUEUE_OVERFLOW));
        st(&mut s.ro, REG_FQCSR, 4, u64::from(QUEUE_ACTIVE | QUEUE_BUSY));
        st(&mut s.wc, REG_PQCSR, 4, qwc | u64::from(QUEUE_OVERFLOW));
        st(&mut s.ro, REG_PQCSR, 4, u64::from(QUEUE_ACTIVE | QUEUE_BUSY));
        st(&mut s.wc, REG_IPSR, 4, u64::from(u32::MAX));
        st(&mut s.ro, REG_ICVEC, 4, 0);
        s.set64(REG_DDTP, s.ddtp);
        if cap & CAP_DBG != 0 {
            st(&mut s.ro, REG_TR_REQ_IOVA, 8, 0);
            st(&mut s.ro, REG_TR_REQ_CTL, 8, TR_REQ_CTL_GO_BUSY);
        }
        if cap & CAP_HPM != 0 {
            // +1 for the cycle counter bit.
            let inh = !((2u64 << hpm_cntrs) - 1) as u32;
            st(&mut s.ro, REG_IOCOUNTINH, 4, u64::from(inh));
            st(&mut s.ro, REG_IOHPMCYCLES, 8, 0);
            let n = hpm_cntrs as usize * 8;
            let ctr = REG_IOHPMCTR_BASE as usize;
            let evt = REG_IOHPMEVT_BASE as usize;
            s.ro[ctr..ctr + n].fill(0);
            s.ro[evt..evt + n].fill(0);
        }
        s
    }

    fn get32(&self, off: u64) -> u32 {
        ld(&self.regs, off, 4) as u32
    }

    fn set32(&mut self, off: u64, v: u32) {
        st(&mut self.regs, off, 4, u64::from(v));
    }

    /// `riscv_iommu_reg_mod32()`, returning the old value.
    fn mod32(&mut self, off: u64, set: u32, clr: u32) -> u32 {
        let v = self.get32(off);
        self.set32(off, (v & !clr) | set);
        v
    }

    fn get64(&self, off: u64) -> u64 {
        ld(&self.regs, off, 8)
    }

    fn set64(&mut self, off: u64, v: u64) {
        st(&mut self.regs, off, 8, v);
    }

    fn mod64(&mut self, off: u64, set: u64, clr: u64) -> u64 {
        let v = self.get64(off);
        self.set64(off, (v & !clr) | set);
        v
    }

    /// `riscv_iommu_reset()`.
    fn reset(&mut self) {
        self.ddtp = if self.conf.off { DDTP_MODE_OFF } else { DDTP_MODE_BARE };
        self.set64(REG_DDTP, self.ddtp);
        let clr = QUEUE_ENABLE | QUEUE_INTR_ENABLE | QUEUE_ACTIVE | QUEUE_BUSY;
        self.mod32(REG_CQCSR, 0, clr);
        self.mod32(REG_FQCSR, 0, clr);
        self.mod32(REG_PQCSR, 0, clr);
        self.mod64(REG_TR_REQ_CTL, 0, TR_REQ_CTL_GO_BUSY);
        self.set32(REG_IPSR, 0);
        self.ctx_cache.clear();
        self.iot_cache.clear();
    }

    /// `riscv_iommu_notify()`.
    fn notify(&mut self, vec_type: u32) {
        if !self.has_notify {
            return;
        }
        let icvec = self.get32(REG_ICVEC);
        let ipsr = self.mod32(REG_IPSR, 1 << vec_type, 0);
        if ipsr & (1 << vec_type) == 0 {
            self.irqs.push((icvec >> (vec_type * 4)) & 0xf);
        }
    }

    /// `riscv_iommu_fault()`: puts a record in the fault queue.
    fn fault(&mut self, m: &Env<'_>, hdr: u64, iotval: u64, iotval2: u64) {
        let ctrl = self.get32(REG_FQCSR);
        let head = self.get32(REG_FQH) & self.fq_mask;
        let tail = self.get32(REG_FQT) & self.fq_mask;
        let next = tail.wrapping_add(1) & self.fq_mask;

        if ctrl & QUEUE_ACTIVE == 0 || ctrl & (QUEUE_OVERFLOW | QUEUE_MEM_FAULT) != 0 {
            return;
        }

        if head == next {
            self.mod32(REG_FQCSR, QUEUE_OVERFLOW, 0);
        } else {
            let addr = self.fq_addr.wrapping_add(u64::from(tail) * 32);
            let mut rec = [0u8; 32];
            rec[..8].copy_from_slice(&hdr.to_le_bytes());
            rec[16..24].copy_from_slice(&iotval.to_le_bytes());
            rec[24..].copy_from_slice(&iotval2.to_le_bytes());
            if m.dma.write(addr, &rec) {
                self.set32(REG_FQT, next);
            } else {
                self.mod32(REG_FQCSR, QUEUE_MEM_FAULT, 0);
            }
        }

        if ctrl & QUEUE_INTR_ENABLE != 0 {
            self.notify(RISCV_IOMMU_INTR_FQ);
        }
    }

    /// `riscv_iommu_pri()`: puts a page request in the page request queue.
    fn pri(&mut self, m: &Env<'_>, hdr: u64, payload: u64) {
        let ctrl = self.get32(REG_PQCSR);
        let head = self.get32(REG_PQH) & self.pq_mask;
        let tail = self.get32(REG_PQT) & self.pq_mask;
        let next = tail.wrapping_add(1) & self.pq_mask;

        if ctrl & QUEUE_ACTIVE == 0 || ctrl & (QUEUE_OVERFLOW | QUEUE_MEM_FAULT) != 0 {
            return;
        }

        if head == next {
            self.mod32(REG_PQCSR, QUEUE_OVERFLOW, 0);
        } else {
            let addr = self.pq_addr.wrapping_add(u64::from(tail) * 16);
            let mut rec = [0u8; 16];
            rec[..8].copy_from_slice(&hdr.to_le_bytes());
            rec[8..].copy_from_slice(&payload.to_le_bytes());
            if m.dma.write(addr, &rec) {
                self.set32(REG_PQT, next);
            } else {
                self.mod32(REG_PQCSR, QUEUE_MEM_FAULT, 0);
            }
        }

        if ctrl & QUEUE_INTR_ENABLE != 0 {
            self.notify(RISCV_IOMMU_INTR_PQ);
        }
    }

    /// `riscv_iommu_report_fault()`.
    #[allow(clippy::too_many_arguments)]
    fn report_fault(
        &mut self,
        m: &Env<'_>,
        ctx: &Ctx,
        fault_type: u64,
        cause: u32,
        pv: bool,
        iotval: u64,
        iotval2: u64,
    ) {
        if ctx.tc & DC_TC_DTF != 0
            && !matches!(
                cause,
                CAUSE_DMA_DISABLED
                    | CAUSE_DDT_LOAD_FAULT
                    | CAUSE_DDT_INVALID
                    | CAUSE_DDT_MISCONFIGURED
                    | CAUSE_DDT_CORRUPTED
                    | CAUSE_INTERNAL_DP_ERROR
                    | CAUSE_MSI_WR_FAULT
            )
        {
            // DTF keeps this cause from being reported.
            return;
        }
        let mut hdr = set_field(0, FQ_HDR_CAUSE, u64::from(cause));
        hdr = set_field(hdr, FQ_HDR_TTYPE, fault_type);
        hdr = set_field(hdr, FQ_HDR_DID, u64::from(ctx.devid));
        hdr = set_field(hdr, FQ_HDR_PV, u64::from(pv));
        if pv {
            hdr = set_field(hdr, FQ_HDR_PID, u64::from(ctx.process_id));
        }
        self.fault(m, hdr, iotval, iotval2);
    }

    /// `riscv_iommu_msi_check()`: whether `gpa` is in the MSI page table window.
    fn msi_check(&self, ctx: &Ctx, gpa: u64) -> bool {
        self.conf.intremap
            && get_field(ctx.msiptp, ATP_MODE) == DC_MSIPTP_MODE_FLAT
            && (ppn_down(gpa) ^ ctx.msi_addr_pattern) & !ctx.msi_addr_mask == 0
    }

    /// The levels, index bits and PTE size of a stage in `mode`, or `None` when the mode is
    /// not there.
    fn walk_params(&self, mode: u64, g_stage: bool, narrow: bool) -> Option<Walk> {
        let (levels, ptidxbits, ptesize, cap) = match (narrow, mode) {
            (_, ATP_MODE_BARE) => return Some(Walk::default()),
            (true, ATP_MODE_SV39) => (2, 10, 4, if g_stage { CAP_SV32X4 } else { CAP_SV32 }),
            (true, _) => return None,
            (false, ATP_MODE_SV39) => (3, 9, 8, if g_stage { CAP_SV39X4 } else { CAP_SV39 }),
            (false, ATP_MODE_SV48) => (4, 9, 8, if g_stage { CAP_SV48X4 } else { CAP_SV48 }),
            (false, ATP_MODE_SV57) => (5, 9, 8, if g_stage { CAP_SV57X4 } else { CAP_SV57 }),
            (false, _) => return None,
        };
        (self.cap & cap != 0).then_some(Walk { step: 0, levels, ptidxbits, ptesize })
    }

    /// `riscv_iommu_spa_fetch()`: the first stage and G-stage page table walks. Returns 0 or
    /// the fault cause.
    fn spa_fetch(&mut self, m: &Env<'_>, ctx: &Ctx, t: &mut Tlb) -> u32 {
        let pv = ctx.process_id != 0;
        let satp_mode = get_field(ctx.satp, ATP_MODE);
        let gatp_mode = get_field(ctx.gatp, ATP_MODE);
        let en_s = satp_mode != ATP_MODE_BARE;
        let en_g = gatp_mode != ATP_MODE_BARE;

        // The early MSI check, for when the IOVA is the GPA.
        if !en_s && t.perm & PERM_WO != 0 && self.msi_check(ctx, t.iova) {
            t.trap = true;
            t.translated = t.iova;
            t.mask = PAGE_OFFSET;
            return 0;
        }

        // Pass through.
        if !(en_s || en_g) {
            t.translated = t.iova;
            t.mask = PAGE_OFFSET;
            t.perm = PERM_RW;
            return 0;
        }

        let mut sc = [Walk::default(); 2];
        let s_narrow = ctx.tc & DC_TC_SXL != 0;
        let g_narrow = self.fctl & FCTL_GXL != 0;
        match self.walk_params(satp_mode, false, s_narrow) {
            Some(w) => sc[0] = w,
            None => return CAUSE_DDT_MISCONFIGURED,
        }
        match self.walk_params(gatp_mode, true, g_narrow) {
            Some(w) => sc[1] = w,
            None => return CAUSE_DDT_MISCONFIGURED,
        }

        let gatp = ppn_phys(get_field(ctx.gatp, ATP_PPN));
        let satp = ppn_phys(get_field(ctx.satp, ATP_PPN));
        let mut addr = if en_s && en_g { satp } else { t.iova };
        let mut base = if en_g { gatp } else { satp };
        let mut pass = usize::from(en_g);

        loop {
            let w = sc[pass];
            let widened = if pass == 1 && w.step == 0 { 2 } else { 0 };
            let va_bits = widened + w.ptidxbits;
            let va_skip = 12 + w.ptidxbits * (w.levels - 1 - w.step);
            let idx = (addr >> va_skip) & ((1u64 << va_bits) - 1);
            let pte_addr = base.wrapping_add(idx * u64::from(w.ptesize));
            let ade = ctx.tc & (if pass == 1 { DC_TC_GADE } else { DC_TC_SADE }) != 0;
            let wo = t.perm & PERM_WO != 0;

            if ade && self.cap & CAP_AMO_HWAD == 0 {
                // GADE and SADE are reserved without AMO_HWAD.
                return CAUSE_DDT_MISCONFIGURED;
            }

            // The address range check before the first level.
            if w.step == 0 {
                let va_len = va_skip + va_bits;
                let va_mask = (1u64 << va_len) - 1;
                if pass == 0 && va_len > 32 {
                    let mask = !0u64 >> (64 - (64 - va_len + 1));
                    let msbs = (addr >> (va_len - 1)) & mask;
                    if msbs != 0 && msbs != mask {
                        return if wo { CAUSE_WR_FAULT_S } else { CAUSE_RD_FAULT_S };
                    }
                } else if addr & va_mask != addr {
                    return if wo { CAUSE_WR_FAULT_VS } else { CAUSE_RD_FAULT_VS };
                }
            }

            let event = if pass == 0 { HPMEVENT_S_VS_WALKS } else { HPMEVENT_G_WALKS };
            self.hpm_incr_ctr(ctx, event);

            let pte = if w.ptesize == 4 {
                m.dma.read_u32(pte_addr).map(u64::from)
            } else {
                m.dma.read_u64(pte_addr)
            };
            let Some(pte) = pte else {
                return if wo { CAUSE_WR_FAULT } else { CAUSE_RD_FAULT };
            };

            sc[pass].step += 1;
            let ppn = pte >> PTE_PPN_SHIFT;
            let rwx = pte & (PTE_R | PTE_W | PTE_X);

            if pte & PTE_V == 0
                || pte & PTE_RESERVED != 0
                // Without a valid process ID every access is a user access.
                || (pte & PTE_U == 0 && !pv)
            {
                break;
            } else if rwx == 0 {
                // An inner PTE, carry on walking.
                base = ppn_phys(ppn);
            } else if rwx == PTE_W
                || rwx == (PTE_W | PTE_X)
                || ppn & ((1u64 << (va_skip - 12)) - 1) != 0
                || (t.perm & PERM_RO != 0 && pte & PTE_R == 0)
                || (wo && pte & PTE_W == 0)
                || (!ade && pte & PTE_A == 0)
                || (wo && !ade && pte & PTE_D == 0)
                // A G-stage PTE is checked as a user access.
                || (pass == 1 && pte & PTE_U == 0)
            {
                break;
            } else {
                // A leaf PTE, the stage is done.
                sc[pass].step = sc[pass].levels;

                if pte & PTE_N != 0 {
                    let mask = napot_page_mask(ppn, addr, &mut base);
                    if mask == 0 {
                        break;
                    }
                    t.mask &= mask;
                } else {
                    base = ppn_phys(ppn) | (addr & ((1u64 << va_skip) - 1));
                    t.mask &= (1u64 << va_skip) - 1;
                }

                // Carry on with the first stage?
                if pass == 1 && sc[0].step != sc[0].levels {
                    pass = 0;
                    addr = t.iova;
                    continue;
                }
                // The GPA or SPA is done.
                t.translated = base;

                let pte_perm = if pte & PTE_W != 0 {
                    if pte & PTE_R != 0 { PERM_RW } else { PERM_WO }
                } else {
                    PERM_RO
                };
                t.perm &= pte_perm;

                // Trap MSI writes to the GPA.
                if pass == 0 && t.perm & PERM_WO != 0 && self.msi_check(ctx, base) {
                    t.trap = true;
                    t.mask = PAGE_OFFSET;
                    return 0;
                }

                // Carry on with the G-stage?
                if pass == 0 && en_g {
                    pass = 1;
                    addr = base;
                    base = gatp;
                    sc[1].step = 0;
                    continue;
                }

                return 0;
            }

            if sc[pass].step == sc[pass].levels {
                // No leaf PTE.
                break;
            }

            // Translate the next table of the first stage through the G-stage.
            if pass == 0 && en_g {
                pass = 1;
                addr = base;
                base = gatp;
                sc[1].step = 0;
            }
        }

        // The fault reports the last address we got to as iotval2.
        t.translated = addr;
        match (t.perm & PERM_WO != 0, pass == 1) {
            (true, true) => CAUSE_WR_FAULT_VS,
            (true, false) => CAUSE_WR_FAULT_S,
            (false, true) => CAUSE_RD_FAULT_VS,
            (false, false) => CAUSE_RD_FAULT_S,
        }
    }

    /// `riscv_iommu_msi_write()`: an MSI write to `gpa` through the MSI page table.
    fn msi_write(&mut self, m: &Env<'_>, ctx: &Ctx, gpa: u64, data: u64, size: usize) {
        let cause = match self.msi_redirect(m, ctx, gpa, data, size) {
            Ok(()) => return,
            Err(cause) => cause,
        };
        self.report_fault(m, ctx, TTYPE_UADDR_WR, cause, ctx.process_id != 0, 0, 0);
    }

    fn msi_redirect(
        &mut self,
        m: &Env<'_>,
        ctx: &Ctx,
        gpa: u64,
        data: u64,
        size: usize,
    ) -> Result<(), u32> {
        // The interrupt file number.
        let intn = pext(ppn_down(gpa), ctx.msi_addr_mask);
        let offset = intn.wrapping_mul(16);

        let mut addr = ppn_phys(get_field(ctx.msiptp, ATP_PPN));
        if addr & offset != 0 {
            // The interrupt file number is out of range.
            return Err(CAUSE_MSI_LOAD_FAULT);
        }
        addr |= offset;

        let mut b = [0u8; 16];
        let res = m.dma.read_res(addr, &mut b);
        if !res.is_ok() {
            return Err(if res == MemTxResult::DECODE_ERROR {
                CAUSE_MSI_PT_CORRUPTED
            } else {
                CAUSE_MSI_LOAD_FAULT
            });
        }
        let pte0 = ld(&b, 0, 8);
        let pte1 = ld(&b, 8, 8);

        if pte0 & MSI_PTE_V == 0 || pte0 & MSI_PTE_C != 0 {
            return Err(CAUSE_MSI_INVALID);
        }

        match get_field(pte0, MSI_PTE_M) {
            MSI_PTE_M_BASIC => {
                // Bits 62:54 and 9:3 are reserved in basic translate mode.
                if pte0 & (genmask(62, 54) | genmask(9, 3)) != 0 {
                    return Err(CAUSE_MSI_MISCONFIGURED);
                }
                let addr = ppn_phys(get_field(pte0, PPN_FIELD));
                if !m.dma.write(addr, &data.to_le_bytes()[..size]) {
                    return Err(CAUSE_MSI_WR_FAULT);
                }
                return Ok(());
            }
            MSI_PTE_M_MRIF => {}
            _ => return Err(CAUSE_MSI_MISCONFIGURED),
        }

        // An IMSIC interrupt file has identities up to 2047 and takes aligned words.
        if data > 2047 || gpa & 3 != 0 {
            return Err(CAUSE_MSI_MISCONFIGURED);
        }

        // The pending bits of the identity, 64 identities every 16 bytes.
        let addr = (get_field(pte0, MSI_PTE_MRIF_ADDR) << 9) | ((data & 0x7c0) >> 2);
        let bit = 1u64 << (data & 0x3f);
        let Some(pending) = m.dma.read_u64(addr) else {
            return Err(CAUSE_MSI_LOAD_FAULT);
        };
        if !m.dma.write(addr, &(pending | bit).to_le_bytes()) {
            return Err(CAUSE_MSI_WR_FAULT);
        }

        // The enable bits.
        let Some(enable) = m.dma.read_u64(addr + 8) else {
            return Err(CAUSE_MSI_LOAD_FAULT);
        };
        if enable & bit == 0 {
            return Ok(());
        }

        // The notice MSI.
        let addr = ppn_phys(get_field(pte1, PPN_FIELD));
        let n190 = get_field(pte1, MSI_MRIF_NID) | (get_field(pte1, MSI_MRIF_NID_MSB) << 10);
        if !m.dma.write(addr, &(n190 as u32).to_le_bytes()) {
            return Err(CAUSE_MSI_WR_FAULT);
        }
        Ok(())
    }

    /// `riscv_iommu_validate_device_ctx()`.
    fn validate_device_ctx(&self, ctx: &Ctx) -> bool {
        let tc = ctx.tc;
        if tc & DC_TC_RESERVED != 0 {
            return false;
        }
        if self.cap & CAP_ATS == 0 && tc & (DC_TC_EN_ATS | DC_TC_EN_PRI | DC_TC_PRPR) != 0 {
            return false;
        }
        if tc & DC_TC_EN_ATS == 0 && tc & (DC_TC_T2GPA | DC_TC_EN_PRI) != 0 {
            return false;
        }
        if tc & DC_TC_EN_PRI == 0 && tc & DC_TC_PRPR != 0 {
            return false;
        }
        if self.cap & CAP_T2GPA == 0 && tc & DC_TC_T2GPA != 0 {
            return false;
        }
        if self.cap & CAP_MSI_FLAT != 0 {
            let msi_mode = get_field(ctx.msiptp, ATP_MODE);
            if msi_mode != DC_MSIPTP_MODE_OFF && msi_mode != DC_MSIPTP_MODE_FLAT {
                return false;
            }
        }

        let gatp = get_field(ctx.gatp, ATP_MODE);
        if tc & DC_TC_T2GPA != 0 && gatp == ATP_MODE_BARE {
            return false;
        }
        // The G-stage root must be 16 KiB aligned.
        if gatp != ATP_MODE_BARE && ppn_phys(get_field(ctx.gatp, ATP_PPN)) & ((1 << 14) - 1) != 0 {
            return false;
        }

        let fsc_mode = get_field(ctx.satp, ATP_MODE);
        if tc & DC_TC_PDTV != 0 {
            let need = match fsc_mode {
                PDTP_MODE_PD8 => CAP_PD8,
                PDTP_MODE_PD17 => CAP_PD17,
                PDTP_MODE_PD20 => CAP_PD20,
                _ => 0,
            };
            if self.cap & need != need {
                return false;
            }
        } else {
            if tc & DC_TC_DPE != 0 {
                return false;
            }
            if !self.satp_mode_ok(tc, fsc_mode) {
                return false;
            }
        }

        // There is only little endian, so SBE must be clear.
        tc & DC_TC_SBE == 0
    }

    /// Whether the IOMMU has the first stage mode `mode` of a context with `tc`.
    fn satp_mode_ok(&self, tc: u64, mode: u64) -> bool {
        let need = if tc & DC_TC_SXL != 0 {
            if mode == ATP_MODE_SV39 { CAP_SV32 } else { 0 }
        } else {
            match mode {
                ATP_MODE_SV39 => CAP_SV39,
                ATP_MODE_SV48 => CAP_SV48,
                ATP_MODE_SV57 => CAP_SV57,
                _ => 0,
            }
        };
        self.cap & need == need
    }

    /// `riscv_iommu_validate_process_ctx()`.
    fn validate_process_ctx(&self, ctx: &Ctx) -> bool {
        if get_field(ctx.ta, PC_TA_RESERVED) != 0 || get_field(ctx.satp, PC_FSC_RESERVED) != 0 {
            return false;
        }
        let mode = get_field(ctx.satp, ATP_MODE);
        if !matches!(mode, ATP_MODE_BARE | ATP_MODE_SV39 | ATP_MODE_SV48 | ATP_MODE_SV57) {
            return false;
        }
        self.satp_mode_ok(ctx.tc, mode)
    }

    /// `pdt_memory_read()`: reads `N` quads of the process directory at the guest physical
    /// address `addr`, through the G-stage.
    fn pdt_read<const N: usize>(&self, m: &Env<'_>, ctx: &Ctx, addr: u64) -> Option<[u64; N]> {
        let mut base = addr;
        let gatp_mode = get_field(ctx.gatp, ATP_MODE);
        if gatp_mode != ATP_MODE_BARE {
            let narrow = self.fctl & FCTL_GXL != 0;
            let mut w = self.walk_params(gatp_mode, true, narrow)?;
            base = ppn_phys(get_field(ctx.gatp, ATP_PPN));
            loop {
                let va_bits = if w.step == 0 { 2 } else { 0 } + w.ptidxbits;
                let va_skip = 12 + w.ptidxbits * (w.levels - 1 - w.step);
                let idx = (addr >> va_skip) & ((1u64 << va_bits) - 1);
                let pte_addr = base.wrapping_add(idx * u64::from(w.ptesize));

                if w.step == 0 {
                    let va_mask = (1u64 << (va_skip + va_bits)) - 1;
                    if addr & va_mask != addr {
                        return None;
                    }
                }

                let pte = if w.ptesize == 4 {
                    m.dma.read_u32(pte_addr).map(u64::from)?
                } else {
                    m.dma.read_u64(pte_addr)?
                };

                w.step += 1;
                let ppn = pte >> PTE_PPN_SHIFT;
                let rwx = pte & (PTE_R | PTE_W | PTE_X);
                if pte & PTE_V == 0 {
                    return None;
                } else if rwx == 0 {
                    base = ppn_phys(ppn);
                } else if rwx == PTE_W
                    || rwx == (PTE_W | PTE_X)
                    || ppn & ((1u64 << (va_skip - 12)) - 1) != 0
                {
                    return None;
                } else {
                    if pte & PTE_N != 0 {
                        if napot_page_mask(ppn, addr, &mut base) == 0 {
                            return None;
                        }
                    } else {
                        base = ppn_phys(ppn) | (addr & ((1u64 << va_skip) - 1));
                    }
                    break;
                }

                if w.step == w.levels {
                    return None;
                }
            }
        }

        let mut b = vec![0u8; N * 8];
        if !m.dma.read(base, &mut b) {
            return None;
        }
        let mut out = [0u64; N];
        for (i, o) in out.iter_mut().enumerate() {
            *o = ld(&b, i as u64 * 8, 8);
        }
        Some(out)
    }

    /// `riscv_iommu_ctx_fetch()`: the device directory and process directory walks. Returns
    /// 0 or the fault cause.
    fn ctx_fetch(&mut self, m: &Env<'_>, ctx: &mut Ctx) -> u32 {
        let ddtp = self.ddtp;
        let mut addr = ppn_phys(get_field(ddtp, PPN_FIELD));
        // The device context format: 0 is extended (64 bytes), 1 is base (32 bytes).
        let dc_fmt = u32::from(!self.conf.intremap);
        let dc_len = 64u64 >> dc_fmt;

        let mut depth: u32 = match get_field(ddtp, DDTP_MODE) {
            DDTP_MODE_OFF => return CAUSE_DMA_DISABLED,
            DDTP_MODE_BARE => {
                // A pass-through context.
                ctx.gatp = 0;
                ctx.satp = 0;
                ctx.tc = DC_TC_V;
                if self.conf.ats {
                    ctx.tc |= DC_TC_EN_ATS;
                }
                ctx.ta = 0;
                ctx.msiptp = 0;
                return 0;
            }
            DDTP_MODE_1LVL => 0,
            DDTP_MODE_2LVL => 1,
            DDTP_MODE_3LVL => 2,
            _ => return CAUSE_DDT_MISCONFIGURED,
        };

        // The device ID width: 6, 15 and 24 bits with extended contexts, 7, 16 and 24 with
        // base ones.
        let width = depth * 9 + 6 + u32::from(dc_fmt != 0 && depth != 2);
        if u64::from(ctx.devid) >= 1u64 << width {
            return CAUSE_TTYPE_BLOCKED;
        }

        while depth > 0 {
            depth -= 1;
            self.hpm_incr_ctr(ctx, HPMEVENT_DD_WALK);
            let split = depth * 9 + 6 + dc_fmt;
            addr |= ((u64::from(ctx.devid) >> split) << 3) & PAGE_OFFSET;
            let Some(de) = m.dma.read_u64(addr) else {
                return CAUSE_DDT_LOAD_FAULT;
            };
            if de & DDTE_VALID == 0 {
                return CAUSE_DDT_INVALID;
            }
            if de & !(PPN_FIELD | DDTE_VALID) != 0 {
                return CAUSE_DDT_MISCONFIGURED;
            }
            addr = ppn_phys(get_field(de, PPN_FIELD));
        }

        self.hpm_incr_ctr(ctx, HPMEVENT_DD_WALK);

        // The device context in its page.
        addr |= (u64::from(ctx.devid) * dc_len) & PAGE_OFFSET;
        let mut b = [0u8; 64];
        if !m.dma.read(addr, &mut b[..dc_len as usize]) {
            return CAUSE_DDT_LOAD_FAULT;
        }
        ctx.tc = ld(&b, 0, 8);
        ctx.gatp = ld(&b, 8, 8);
        ctx.ta = ld(&b, 16, 8);
        ctx.satp = ld(&b, 24, 8);
        ctx.msiptp = ld(&b, 32, 8);
        ctx.msi_addr_mask = ld(&b, 40, 8);
        ctx.msi_addr_pattern = ld(&b, 48, 8);

        if ctx.tc & DC_TC_V == 0 {
            return CAUSE_DDT_INVALID;
        }
        if !self.validate_device_ctx(ctx) {
            return CAUSE_DDT_MISCONFIGURED;
        }

        let mode = get_field(ctx.satp, ATP_MODE);
        let mut addr = ppn_phys(get_field(ctx.satp, ATP_PPN));

        if ctx.tc & DC_TC_PDTV == 0 {
            if ctx.process_id != 0 {
                // Process IDs are off.
                return CAUSE_TTYPE_BLOCKED;
            }
            if mode > ATP_MODE_SV57 {
                return CAUSE_DDT_INVALID;
            }
            return 0;
        }

        if ctx.process_id == 0 && ctx.tc & DC_TC_DPE == 0 {
            // No default process ID, so no first stage.
            ctx.satp = 0;
            return 0;
        }

        if mode == ATP_MODE_BARE {
            return 0;
        }
        if mode > PDTP_MODE_PD20 {
            return CAUSE_PDT_MISCONFIGURED;
        }

        let mut depth = (mode - PDTP_MODE_PD8) as u32;
        while depth > 0 {
            depth -= 1;
            self.hpm_incr_ctr(ctx, HPMEVENT_PD_WALK);
            let split = depth * 9 + 8;
            addr |= ((u64::from(ctx.process_id) >> split) << 3) & PAGE_OFFSET;
            let Some([de]) = self.pdt_read::<1>(m, ctx, addr) else {
                return CAUSE_PDT_LOAD_FAULT;
            };
            if de & PDTE_VALID == 0 {
                return CAUSE_PDT_INVALID;
            }
            addr = ppn_phys(get_field(de, PPN_FIELD));
        }

        self.hpm_incr_ctr(ctx, HPMEVENT_PD_WALK);

        // The process context.
        addr |= (u64::from(ctx.process_id) << 4) & PAGE_OFFSET;
        let Some([ta, fsc]) = self.pdt_read::<2>(m, ctx, addr) else {
            return CAUSE_PDT_LOAD_FAULT;
        };
        ctx.ta = ta;
        ctx.satp = fsc;

        if ctx.ta & PC_TA_V == 0 {
            return CAUSE_PDT_INVALID;
        }
        if !self.validate_process_ctx(ctx) {
            return CAUSE_PDT_MISCONFIGURED;
        }
        0
    }

    /// `riscv_iommu_ctx()`: the context of `devid` and `process_id`, from the cache or the
    /// tables. A failed fetch reports a fault.
    fn ctx(
        &mut self,
        m: &Env<'_>,
        devid: u32,
        process_id: u32,
        perm: u8,
        iova: u64,
    ) -> Option<Ctx> {
        let mode = get_field(self.ddtp, DDTP_MODE);
        let key = (devid, process_id);
        if mode != DDTP_MODE_OFF && mode != DDTP_MODE_BARE {
            if let Some(c) = self.ctx_cache.get(&key) {
                if c.tc & DC_TC_V != 0 {
                    return Some(*c);
                }
            }
        }

        let mut ctx = Ctx { devid, process_id, ..Ctx::default() };
        let fault = self.ctx_fetch(m, &mut ctx);
        if fault == 0 {
            if mode != DDTP_MODE_BARE {
                if self.ctx_cache.len() >= LIMIT_CACHE_CTX {
                    self.ctx_cache.clear();
                }
                self.ctx_cache.insert(key, ctx);
            }
            return Some(ctx);
        }

        let fault_type = if perm & PERM_RO != 0 { TTYPE_UADDR_RD } else { TTYPE_UADDR_WR };
        self.report_fault(m, &ctx, fault_type, fault, process_id != 0, iova, 0);
        None
    }

    /// `riscv_iommu_ctx_inval()`.
    fn ctx_inval(&mut self, how: CtxMatch, devid: u32, process_id: u32) {
        for c in self.ctx_cache.values_mut() {
            let hit = match how {
                CtxMatch::All => true,
                CtxMatch::Devid => c.devid == devid,
                CtxMatch::DevidProcid => c.devid == devid && c.process_id == process_id,
            };
            if hit {
                c.tc &= !DC_TC_V;
            }
        }
    }

    /// `riscv_iommu_iot_inval()`: drops the matching entries of `tag` by clearing their
    /// permissions, as QEMU does.
    fn iot_inval(&mut self, how: IotMatch, gscid: u32, pscid: u32, iova: u64, tag: TransTag) {
        let iova = ppn_down(iova) & genmask(43, 0);
        for (k, e) in &mut self.iot_cache {
            if k.tag == tag
                && (!how.gscid || k.gscid == gscid)
                && (!how.pscid || k.pscid == pscid)
                && (!how.iova || k.iova == iova)
            {
                e.perm = PERM_NONE;
            }
        }
    }

    /// `riscv_iommu_translate()`. Returns 0 or the fault cause, which it has reported.
    fn translate(&mut self, m: &Env<'_>, ctx: &Ctx, t: &mut Tlb, enable_cache: bool) -> u32 {
        let satp = get_field(ctx.satp, ATP_MODE);
        let gatp = get_field(ctx.gatp, ATP_MODE);
        let tag = match (satp == ATP_MODE_BARE, gatp == ATP_MODE_BARE) {
            (true, true) => TransTag::By,
            (true, false) => TransTag::Vg,
            (false, true) => TransTag::Ss,
            (false, false) => TransTag::Vn,
        };

        self.hpm_incr_ctr(ctx, HPMEVENT_URQ);

        let mut enable_pri = t.perm == PERM_NONE && ctx.tc & DC_TC_EN_PRI != 0;
        let enable_pid = ctx.tc & DC_TC_PDTV != 0;

        let fault = 'done: {
            // An ATS request.
            if t.perm == PERM_NONE {
                self.hpm_incr_ctr(ctx, HPMEVENT_ATS_RQ);
                if ctx.tc & DC_TC_EN_ATS == 0 {
                    enable_pri = false;
                    break 'done CAUSE_TTYPE_BLOCKED;
                }
            }

            let key = IotKey {
                tag,
                gscid: get_field(ctx.gatp, DC_IOHGATP_GSCID) as u32,
                pscid: get_field(ctx.ta, DC_TA_PSCID) as u32,
                iova: ppn_down(t.iova) & genmask(43, 0),
            };
            if let Some(e) = self.iot_cache.get(&key) {
                if e.perm != PERM_NONE {
                    t.translated = ppn_phys(e.phys);
                    t.mask = PAGE_OFFSET;
                    t.perm = e.perm;
                    break 'done 0;
                }
            }

            self.hpm_incr_ctr(ctx, HPMEVENT_TLB_MISS);

            let fault = self.spa_fetch(m, ctx, t);
            if fault == 0 && t.trap {
                // Trapped MSIs are not cached.
                break 'done 0;
            }

            // Identity mappings are not cached either.
            if fault == 0 && t.translated != t.iova && enable_cache && self.conf.ioatc_limit != 0 {
                if self.iot_cache.len() >= self.conf.ioatc_limit as usize {
                    self.iot_cache.clear();
                }
                let phys = ppn_down(t.translated) & genmask(43, 0);
                self.iot_cache.insert(key, IotEntry { phys, perm: t.perm });
            }
            fault
        };

        if enable_pri && fault != 0 {
            let mut hdr = 0;
            if enable_pid {
                hdr = set_field(PREQ_HDR_PV, PREQ_HDR_PID, u64::from(ctx.process_id));
            }
            hdr = set_field(hdr, PREQ_HDR_DID, u64::from(ctx.devid));
            let payload = (t.iova & !PAGE_OFFSET) | PREQ_PAYLOAD_M;
            self.pri(m, hdr, payload);
            return fault;
        }

        if fault != 0 {
            let ttype = if t.perm & PERM_WO != 0 {
                TTYPE_UADDR_WR
            } else if t.perm & PERM_RO != 0 {
                TTYPE_UADDR_RD
            } else {
                TTYPE_PCIE_ATS_REQ
            };
            self.report_fault(m, ctx, ttype, fault, enable_pid, t.iova, t.translated);
        }
        fault
    }

    /// `riscv_iommu_memory_region_translate()` for the device with ID `devid`.
    fn region_translate(
        &mut self,
        m: &Env<'_>,
        devid: u32,
        iommu_idx: u32,
        addr: u64,
        perm: u8,
    ) -> Tlb {
        let mut t = Tlb { iova: addr, translated: 0, mask: !0, perm, trap: false };
        let failed = match self.ctx(m, devid, iommu_idx, perm, addr) {
            None => true,
            Some(ctx) => self.translate(m, &ctx, &mut t, true) != 0,
        };
        if failed {
            t.mask = 0;
            t.perm = PERM_NONE;
        }
        t
    }

    /// `riscv_iommu_trap_write()`: an MSI write from `devid` that hit the MSI page table.
    fn trap_write(&mut self, m: &Env<'_>, devid: u32, addr: u64, data: u64, size: usize) {
        if let Some(ctx) = self.ctx(m, devid, 0, PERM_RW, addr) {
            self.msi_write(m, &ctx, addr, data, size);
        }
    }

    /// `riscv_iommu_process_ddtp()`.
    fn process_ddtp(&mut self) {
        let old = self.ddtp;
        let new = self.get64(REG_DDTP);
        let new_mode = get_field(new, DDTP_MODE);
        let old_mode = get_field(old, DDTP_MODE);

        // {OFF, BARE} can go to any mode, {1LVL, 2LVL, 3LVL} only to OFF or BARE.
        let ok = if new_mode == old_mode || new_mode == DDTP_MODE_OFF || new_mode == DDTP_MODE_BARE
        {
            true
        } else if matches!(new_mode, DDTP_MODE_1LVL | DDTP_MODE_2LVL | DDTP_MODE_3LVL) {
            old_mode == DDTP_MODE_OFF || old_mode == DDTP_MODE_BARE
        } else {
            false
        };

        let new = if ok { set_field(new & PPN_FIELD, DDTP_MODE, new_mode) } else { old };
        self.ddtp = new;
        self.set64(REG_DDTP, new);
    }

    /// `riscv_iommu_process_cq_tail()`: runs the command queue.
    fn process_cq_tail(&mut self, m: &Env<'_>) {
        let ctrl = self.get32(REG_CQCSR);
        let tail = self.get32(REG_CQT) & self.cq_mask;
        let mut head = self.get32(REG_CQH) & self.cq_mask;

        if ctrl & QUEUE_ACTIVE == 0 || ctrl & (CQCSR_CMD_ILL | QUEUE_MEM_FAULT) != 0 {
            return;
        }

        while tail != head {
            let addr = self.cq_addr.wrapping_add(u64::from(head) * 16);
            let mut b = [0u8; 16];
            if !m.dma.read(addr, &mut b) {
                self.mod32(REG_CQCSR, QUEUE_MEM_FAULT, 0);
                return self.cq_fault(ctrl);
            }
            let dword0 = ld(&b, 0, 8);
            let dword1 = ld(&b, 8, 8);

            if !self.run_command(m, dword0, dword1) {
                return self.cq_fault(ctrl);
            }

            // Advance the head once the command is done.
            head = head.wrapping_add(1) & self.cq_mask;
            self.set32(REG_CQH, head);
        }
    }

    /// The `fault:` label of `riscv_iommu_process_cq_tail()`.
    fn cq_fault(&mut self, ctrl: u32) {
        if ctrl & QUEUE_INTR_ENABLE != 0 {
            self.notify(RISCV_IOMMU_INTR_CQ);
        }
    }

    /// Runs one command. Returns false when the queue stops, with `CQMF` or `CMD_ILL` set.
    fn run_command(&mut self, m: &Env<'_>, dword0: u64, dword1: u64) -> bool {
        let illegal = |s: &mut State| {
            // An illegal command, the head stays on it.
            s.mod32(REG_CQCSR, CQCSR_CMD_ILL, 0);
            false
        };

        match get_field(dword0, CMD_OPCODE | CMD_FUNC) {
            CMD_IOFENCE_C => {
                if dword0 & CMD_IOFENCE_RESERVED != 0 {
                    return illegal(self);
                }
                // ATS is synchronous here, so there is nothing to wait for.
                if dword0 & CMD_IOFENCE_AV != 0 {
                    let data = get_field(dword0, CMD_IOFENCE_DATA) as u32;
                    if !m.dma.write(dword1 << 2, &data.to_le_bytes()) {
                        self.mod32(REG_CQCSR, QUEUE_MEM_FAULT, 0);
                        return false;
                    }
                }
            }
            CMD_IOTINVAL_GVMA => {
                if dword0 & CMD_IOTINVAL_PSCV != 0 {
                    return illegal(self);
                }
                let gv = dword0 & CMD_IOTINVAL_GV != 0;
                let av = dword0 & CMD_IOTINVAL_AV != 0;
                let how = IotMatch { gscid: gv, pscid: false, iova: gv && av };
                let (gscid, pscid, iova) = iotinval_args(dword0, dword1);
                self.iot_inval(how, gscid, pscid, iova, TransTag::Vg);
                self.iot_inval(how, gscid, pscid, iova, TransTag::Vn);
            }
            CMD_IOTINVAL_VMA => {
                let gv = dword0 & CMD_IOTINVAL_GV != 0;
                let av = dword0 & CMD_IOTINVAL_AV != 0;
                let pscv = dword0 & CMD_IOTINVAL_PSCV != 0;
                let how = IotMatch { gscid: gv, pscid: pscv, iova: av };
                let tag = if gv { TransTag::Vn } else { TransTag::Ss };
                let (gscid, pscid, iova) = iotinval_args(dword0, dword1);
                self.iot_inval(how, gscid, pscid, iova, tag);
            }
            CMD_IODIR_INVAL_DDT => {
                let how = if dword0 & CMD_IODIR_DV == 0 { CtxMatch::All } else { CtxMatch::Devid };
                self.ctx_inval(how, get_field(dword0, CMD_IODIR_DID) as u32, 0);
            }
            CMD_IODIR_INVAL_PDT => {
                if dword0 & CMD_IODIR_DV == 0 {
                    return illegal(self);
                }
                let devid = get_field(dword0, CMD_IODIR_DID) as u32;
                let pid = get_field(dword0, CMD_IODIR_PID) as u32;
                self.ctx_inval(CtxMatch::DevidProcid, devid, pid);
            }
            CMD_ATS_INVAL | CMD_ATS_PRGR => {
                if !self.conf.ats {
                    return illegal(self);
                }
                // These go to the IOMMU notifiers, which there are none of.
            }
            _ => return illegal(self),
        }
        true
    }

    /// `riscv_iommu_process_cq_control()`.
    fn process_cq_control(&mut self, m: &Env<'_>) {
        let ctrl = self.get32(REG_CQCSR);
        let enable = ctrl & QUEUE_ENABLE != 0;
        let active = ctrl & QUEUE_ACTIVE != 0;
        let (set, clr) = if enable && !active {
            let base = self.get64(REG_CQB);
            self.cq_mask = ((2u64 << get_field(base, QUEUE_LOGSZ)) - 1) as u32;
            self.cq_addr = ppn_phys(get_field(base, PPN_FIELD));
            st(&mut self.ro, REG_CQT, 4, u64::from(!self.cq_mask));
            self.set32(REG_CQH, 0);
            self.set32(REG_CQT, 0);
            let clr = QUEUE_BUSY | QUEUE_MEM_FAULT | CQCSR_CMD_ILL | CQCSR_CMD_TO;
            (QUEUE_ACTIVE, clr | CQCSR_FENCE_W_IP)
        } else if !enable && active {
            st(&mut self.ro, REG_CQT, 4, u64::from(u32::MAX));
            (0, QUEUE_BUSY | QUEUE_ACTIVE)
        } else {
            (0, QUEUE_BUSY)
        };
        self.mod32(REG_CQCSR, set, clr);

        // With the errors cleared and the queue still on, run what is pending.
        let ctrl = self.get32(REG_CQCSR);
        if ctrl & QUEUE_ACTIVE != 0 && ctrl & (CQCSR_CMD_ILL | QUEUE_MEM_FAULT) == 0 {
            self.process_cq_tail(m);
        }
    }

    /// `riscv_iommu_process_fq_control()` and `riscv_iommu_process_pq_control()`.
    fn process_queue_control(&mut self, csr: u64, qb: u64, qh: u64, qt: u64) {
        let ctrl = self.get32(csr);
        let enable = ctrl & QUEUE_ENABLE != 0;
        let active = ctrl & QUEUE_ACTIVE != 0;
        let (set, clr) = if enable && !active {
            let base = self.get64(qb);
            let mask = ((2u64 << get_field(base, QUEUE_LOGSZ)) - 1) as u32;
            let addr = ppn_phys(get_field(base, PPN_FIELD));
            if csr == REG_FQCSR {
                self.fq_mask = mask;
                self.fq_addr = addr;
            } else {
                self.pq_mask = mask;
                self.pq_addr = addr;
            }
            st(&mut self.ro, qh, 4, u64::from(!mask));
            self.set32(qh, 0);
            self.set32(qt, 0);
            (QUEUE_ACTIVE, QUEUE_BUSY | QUEUE_MEM_FAULT | QUEUE_OVERFLOW)
        } else if !enable && active {
            st(&mut self.ro, qh, 4, u64::from(u32::MAX));
            (0, QUEUE_BUSY | QUEUE_ACTIVE)
        } else {
            (0, QUEUE_BUSY)
        };
        self.mod32(csr, set, clr);
    }

    /// `riscv_iommu_process_dbg()`: the translation request registers.
    fn process_dbg(&mut self, m: &Env<'_>) {
        let iova = self.get64(REG_TR_REQ_IOVA);
        let ctrl = self.get64(REG_TR_REQ_CTL);
        let devid = get_field(ctrl, TR_REQ_CTL_DID) as u32;
        let pid = get_field(ctrl, TR_REQ_CTL_PID) as u32;
        let perm = if ctrl & TR_REQ_CTL_NW != 0 { PERM_RO } else { PERM_RW };

        if ctrl & TR_REQ_CTL_GO_BUSY == 0 {
            return;
        }

        let resp = match self.ctx(m, devid, pid, perm, iova) {
            None => TR_RESPONSE_FAULT | (u64::from(CAUSE_DMA_DISABLED) << 10),
            Some(ctx) => {
                let mut t = Tlb { iova, translated: 0, mask: !0, perm, trap: false };
                let fault = self.translate(m, &ctx, &mut t, false);
                if fault != 0 {
                    TR_RESPONSE_FAULT | (u64::from(fault) << 10)
                } else {
                    set_field(0, PPN_FIELD, ppn_down(t.translated & !t.mask))
                }
            }
        };
        self.set64(REG_TR_RESPONSE, resp);
        self.mod64(REG_TR_REQ_CTL, 0, TR_REQ_CTL_GO_BUSY);
    }

    /// `riscv_iommu_update_icvec()`: each vector is clipped to what the front end has.
    fn update_icvec(&mut self, data: u64) {
        let mut icvec = 0;
        for f in [ICVEC_CIV, ICVEC_FIV, ICVEC_PMIV, ICVEC_PIV] {
            icvec |= (data & f).min(self.icvec_avail & f);
        }
        self.set64(REG_ICVEC, icvec);
    }

    /// `riscv_iommu_update_ipsr()`: a pending bit stays set only while its cause does.
    fn update_ipsr(&mut self, data: u64) {
        let mut set = 0;
        let mut clr = 0;
        let mut check = |bit: u32, csr: u32, causes: u32| {
            if data & u64::from(bit) != 0 && csr & QUEUE_INTR_ENABLE != 0 && csr & causes != 0 {
                set |= bit;
            } else {
                clr |= bit;
            }
        };
        let cq_causes = CQCSR_FENCE_W_IP | CQCSR_CMD_ILL | CQCSR_CMD_TO | QUEUE_MEM_FAULT;
        check(IPSR_CIP, self.get32(REG_CQCSR), cq_causes);
        check(IPSR_FIP, self.get32(REG_FQCSR), QUEUE_OVERFLOW | QUEUE_MEM_FAULT);
        check(IPSR_PIP, self.get32(REG_PQCSR), QUEUE_OVERFLOW | QUEUE_MEM_FAULT);
        if data & u64::from(IPSR_PMIP) == 0 {
            clr |= IPSR_PMIP;
        }
        self.mod32(REG_IPSR, set, clr);
    }

    /// `riscv_iommu_write_reg_val()`: `data` merged into the register at `addr` with its
    /// read-only and write-1-to-clear bits.
    fn write_reg_val(&self, addr: u64, size: usize, data: u64) -> u64 {
        let ro = ld(&self.ro, addr, size);
        let wc = ld(&self.wc, addr, size);
        let cur = ld(&self.regs, addr, size);
        let v = ((cur & ro) | (data & !ro)) & !(data & wc);
        if size == 8 { v } else { v & ((1u64 << (size * 8)) - 1) }
    }

    /// `riscv_iommu_mmio_write()`.
    fn mmio_write(&mut self, m: &Env<'_>, addr: u64, data: u64, size: usize) -> MemResult<()> {
        if addr & (size as u64 - 1) != 0 {
            return Err(MemTxResult::ERROR);
        }
        if addr + size as u64 > RISCV_IOMMU_REG_MSI_CONFIG {
            return Err(MemTxResult::ACCESS_ERROR);
        }

        let mut regb = addr & !3;
        let mut busy = 0u32;
        let mut process = None;
        let mut cy_inh = false;

        match regb {
            REG_DDTP | 0x14 => {
                process = Some(Process::Ddtp);
                regb = REG_DDTP;
                busy = DDTP_BUSY;
            }
            REG_CQT => process = Some(Process::CqTail),
            REG_CQCSR => {
                process = Some(Process::CqControl);
                busy = QUEUE_BUSY;
            }
            REG_FQCSR => {
                process = Some(Process::FqControl);
                busy = QUEUE_BUSY;
            }
            REG_PQCSR => {
                process = Some(Process::PqControl);
                busy = QUEUE_BUSY;
            }
            REG_ICVEC | REG_IPSR => {
                // These two have their own update rules.
                let val = self.write_reg_val(addr, size, data);
                if regb == REG_ICVEC {
                    self.update_icvec(val);
                } else {
                    self.update_ipsr(val);
                }
                return Ok(());
            }
            REG_TR_REQ_CTL => {
                process = Some(Process::Dbg);
                busy = TR_REQ_CTL_GO_BUSY as u32;
            }
            REG_IOCOUNTINH if addr == REG_IOCOUNTINH => {
                // The CY bit before the write.
                cy_inh = self.get32(REG_IOCOUNTINH) & IOCOUNTINH_CY != 0;
            }
            _ => {}
        }

        let v = self.write_reg_val(addr, size, data);
        st(&mut self.regs, addr, size, v);

        if busy != 0 {
            let rw = self.get32(regb);
            self.set32(regb, rw | busy);
        }

        if (REG_IOCOUNTOVF..=reg_iohpmevt(u64::from(RISCV_IOMMU_IOCOUNT_NUM)) + 4).contains(&regb) {
            self.process_hpm_writes(m, regb, cy_inh);
        }

        match process {
            Some(Process::Ddtp) => self.process_ddtp(),
            Some(Process::CqTail) => self.process_cq_tail(m),
            Some(Process::CqControl) => self.process_cq_control(m),
            Some(Process::FqControl) => {
                self.process_queue_control(REG_FQCSR, REG_FQB, REG_FQH, REG_FQT);
            }
            Some(Process::PqControl) => {
                self.process_queue_control(REG_PQCSR, REG_PQB, REG_PQH, REG_PQT);
            }
            Some(Process::Dbg) => self.process_dbg(m),
            None => {}
        }
        Ok(())
    }

    /// `riscv_iommu_mmio_read()`.
    fn mmio_read(&self, m: &Env<'_>, addr: u64, size: usize) -> MemResult<u64> {
        if addr & (size as u64 - 1) != 0 {
            return Err(MemTxResult::ERROR);
        }
        if addr + size as u64 > RISCV_IOMMU_REG_MSI_CONFIG {
            return Err(MemTxResult::ACCESS_ERROR);
        }

        if addr & !7 == REG_IOHPMCYCLES {
            let val = self.hpmcycle_read(m);
            return Ok(ld(&val.to_le_bytes(), addr & 7, size));
        }
        if addr & !3 == REG_IOCOUNTOVF {
            // The cycle counter overflow can be seen before the timer has fired.
            let mut val = self.get64(addr);
            if self.hpmcycle_read(m) & IOHPMCYCLES_OVF != 0 {
                val |= u64::from(IOCOUNTOVF_CY);
            }
            return Ok(ld(&val.to_le_bytes(), addr & 3, size));
        }
        Ok(ld(&self.regs, addr, size))
    }

    /// `riscv_iommu_process_hpm_writes()`.
    fn process_hpm_writes(&mut self, m: &Env<'_>, regb: u64, prev_cy_inh: bool) {
        match regb {
            REG_IOCOUNTINH => self.process_iocntinh_cy(m, prev_cy_inh),
            REG_IOHPMCYCLES | 0x64 => self.process_hpmcycle_write(m),
            r if (REG_IOHPMEVT_BASE..=reg_iohpmevt(u64::from(RISCV_IOMMU_IOCOUNT_NUM)) + 4)
                .contains(&r) =>
            {
                self.process_hpmevt_write(r & !7);
            }
            _ => {}
        }
    }

    /// `riscv_iommu_hpmcycle_read()`.
    fn hpmcycle_read(&self, m: &Env<'_>) -> u64 {
        let cycle = self.get64(REG_IOHPMCYCLES);
        let inhibit = self.get32(REG_IOCOUNTINH);
        if inhibit & IOCOUNTINH_CY != 0 {
            // The clock cannot be stopped, so an inhibited counter reads its last value.
            return (self.hpmcycle_val & IOHPMCYCLES_COUNTER) | (cycle & IOHPMCYCLES_OVF);
        }
        self.hpmcycle_val.wrapping_add(m.cycles()).wrapping_sub(self.hpmcycle_prev)
            | (cycle & IOHPMCYCLES_OVF)
    }

    /// `hpm_incr_ctr()`.
    fn hpm_incr(&mut self, ctr_idx: u32) {
        let off = u64::from(ctr_idx) << 3;
        let v = self.get64(REG_IOHPMCTR_BASE + off);
        self.set64(REG_IOHPMCTR_BASE + off, v.wrapping_add(1));

        if v == u64::MAX {
            // Only the first overflow interrupts. +1 skips the cycle counter bit.
            let ovf = self.mod32(REG_IOCOUNTOVF, 1 << (ctr_idx + 1), 0);
            if ovf & (1 << (ctr_idx + 1)) == 0 {
                self.mod64(REG_IOHPMEVT_BASE + off, IOHPMEVT_OF, 0);
                self.notify(RISCV_IOMMU_INTR_PM);
            }
        }
    }

    /// `riscv_iommu_hpm_incr_ctr()`: counts `event_id` for `ctx` on the counters that
    /// select it.
    fn hpm_incr_ctr(&mut self, ctx: &Ctx, event_id: u32) {
        if self.cap & CAP_HPM == 0 {
            return;
        }
        let inhibit = self.get32(REG_IOCOUNTINH);
        let Some(&ctrs) = self.hpm_event_ctr_map.get(&event_id) else {
            return;
        };

        let mut ctrs = ctrs;
        while ctrs != 0 {
            let ctr_idx = ctrs.trailing_zeros();
            ctrs &= ctrs - 1;
            if inhibit & (1 << (ctr_idx + 1)) != 0 {
                continue;
            }

            let evt = self.get64(REG_IOHPMEVT_BASE + (u64::from(ctr_idx) << 3));

            // The map may not have caught up with a new event ID yet.
            if u64::from(event_id) != get_field(evt, IOHPMEVT_EVENT_ID) {
                continue;
            }

            let (did_gscid, pid_pscid) = if evt & IOHPMEVT_IDT != 0 {
                (
                    get_field(ctx.gatp, DC_IOHGATP_GSCID) as u32,
                    get_field(ctx.ta, DC_TA_PSCID) as u32,
                )
            } else {
                (ctx.devid, ctx.process_id)
            };

            if evt & IOHPMEVT_PV_PSCV != 0
                && ctx.process_id != 0
                && get_field(evt, IOHPMEVT_PID_PSCID) as u32 != pid_pscid
            {
                continue;
            }

            if evt & IOHPMEVT_DV_GSCV != 0 {
                let mut mask = !0u32;
                if evt & IOHPMEVT_DMASK != 0 {
                    // The low ones of DID_GSCID and the zero above them are left out.
                    mask = get_field(evt, IOHPMEVT_DID_GSCID) as u32;
                    mask ^= mask.wrapping_add(1);
                    mask = !mask;
                }
                if get_field(evt, IOHPMEVT_DID_GSCID) as u32 & mask != did_gscid & mask {
                    continue;
                }
            }

            self.hpm_incr(ctr_idx);
        }
    }

    /// `riscv_iommu_hpm_timer_cb()`: the cycle counter has overflowed.
    fn hpm_timer_cb(&mut self, m: &Env<'_>) {
        if self.get32(REG_IOCOUNTINH) & IOCOUNTINH_CY != 0 {
            return;
        }

        if self.irq_overflow_left > 0 {
            let at = m.cycles().wrapping_add(self.irq_overflow_left);
            if let Some(t) = m.timer {
                t.modify_anticipate(at as i64);
            }
            self.irq_overflow_left = 0;
            return;
        }

        let ovf = self.get32(REG_IOCOUNTOVF);
        if ovf & IOCOUNTOVF_CY == 0 {
            // The counter wraps by itself, so hpmcycle_val and hpmcycle_prev stay.
            self.mod32(REG_IOCOUNTOVF, IOCOUNTOVF_CY, 0);
            self.mod64(REG_IOHPMCYCLES, IOHPMCYCLES_OVF, 0);
            self.notify(RISCV_IOMMU_INTR_PM);
        }
    }

    /// `hpm_setup_timer()`: arms the timer for when the cycle counter overflows from
    /// `value`.
    fn hpm_setup_timer(&mut self, m: &Env<'_>, value: u64) {
        if self.get32(REG_IOCOUNTINH) & IOCOUNTINH_CY != 0 {
            return;
        }

        // The counter has 63 bits, so it overflows past i64::MAX.
        let max = i64::MAX as u64;
        let overflow_ns = if value != 0 { max - value + 1 } else { max };
        let mut overflow_at = m.cycles().wrapping_add(overflow_ns);
        self.irq_overflow_left = 0;
        if overflow_at > max {
            self.irq_overflow_left = overflow_at - max;
            overflow_at = max;
        }
        if let Some(t) = m.timer {
            t.modify_anticipate(overflow_at as i64);
        }
    }

    /// `riscv_iommu_process_iocntinh_cy()`.
    fn process_iocntinh_cy(&mut self, m: &Env<'_>, prev_cy_inh: bool) {
        let cy_inh = self.get32(REG_IOCOUNTINH) & IOCOUNTINH_CY != 0;
        if cy_inh == prev_cy_inh {
            return;
        }
        if !cy_inh {
            // The counter runs again from now.
            self.hpmcycle_prev = m.cycles();
            self.hpm_setup_timer(m, self.hpmcycle_val);
        } else {
            // The counter stops at what it has counted so far.
            self.hpmcycle_val =
                self.hpmcycle_val.wrapping_add(m.cycles().wrapping_sub(self.hpmcycle_prev));
            if let Some(t) = m.timer {
                t.del();
            }
        }
    }

    /// `riscv_iommu_process_hpmcycle_write()`.
    fn process_hpmcycle_write(&mut self, m: &Env<'_>) {
        let val = self.get64(REG_IOHPMCYCLES);
        let ovf = self.get32(REG_IOCOUNTOVF);

        // Clearing OF in iohpmcycles clears it in iocountovf.
        if ovf & IOCOUNTOVF_CY != 0 && val & IOHPMCYCLES_OVF == 0 {
            self.mod32(REG_IOCOUNTOVF, 0, IOCOUNTOVF_CY);
        }

        self.hpmcycle_val = val & !IOHPMCYCLES_OVF;
        self.hpmcycle_prev = m.cycles();
        self.hpm_setup_timer(m, self.hpmcycle_val);
    }

    /// `riscv_iommu_process_hpmevt_write()`.
    fn process_hpmevt_write(&mut self, evt_reg: u64) {
        let ctr_idx = ((evt_reg - REG_IOHPMEVT_BASE) >> 3) as u32;
        let ovf = self.get32(REG_IOCOUNTOVF);
        let mut val = self.get64(evt_reg);

        if ctr_idx >= self.hpm_cntrs {
            return;
        }

        // Clearing OF in the event selector clears it in iocountovf.
        if ovf & (1 << (ctr_idx + 1)) != 0 && val & IOHPMEVT_OF == 0 {
            self.mod32(REG_IOCOUNTOVF, 0, 1 << (ctr_idx + 1));
        }

        let event_id = get_field(val, IOHPMEVT_EVENT_ID) as u32;
        if !(HPMEVENT_INVALID + 1..HPMEVENT_MAX).contains(&event_id) {
            // EventID is WARL, a bad one reads back as invalid.
            val = set_field(val, IOHPMEVT_EVENT_ID, u64::from(HPMEVENT_INVALID));
            self.set64(evt_reg, val);
        }

        self.update_event_map(val, ctr_idx);
    }

    /// `update_event_map()`.
    fn update_event_map(&mut self, value: u64, ctr_idx: u32) {
        let event_id = get_field(value, IOHPMEVT_EVENT_ID) as u32;
        let bit = 1u32 << ctr_idx;

        if event_id == HPMEVENT_INVALID {
            // Take the counter off the event it counted.
            let found = self
                .hpm_event_ctr_map
                .iter()
                .filter(|&(_, &v)| v & bit != 0)
                .map(|(&k, &v)| (k, v))
                .min();
            let (key, data) = found.unwrap_or((HPMEVENT_INVALID, 0));
            let left = data & !bit;
            if left != 0 {
                self.hpm_event_ctr_map.insert(key, left);
            } else {
                self.hpm_event_ctr_map.remove(&key);
            }
            return;
        }

        let old = self.hpm_event_ctr_map.get(&event_id).copied().unwrap_or(0);
        self.hpm_event_ctr_map.insert(event_id, old | bit);
    }
}

/// The `gscid`, `pscid` and page address of an `IOTINVAL`.
fn iotinval_args(dword0: u64, dword1: u64) -> (u32, u32, u64) {
    let gscid = get_field(dword0, CMD_IOTINVAL_GSCID) as u32;
    let pscid = get_field(dword0, CMD_IOTINVAL_PSCID) as u32;
    let iova = (dword1 << 2) & !PAGE_OFFSET;
    (gscid, pscid, iova)
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

/// What the IOMMU raises its interrupt vectors with.
type Notifier = Box<dyn Fn(u32) + Send + Sync>;

/// The `riscv-iommu` device.
pub struct RiscvIommu {
    state: Mutex<State>,
    memory: Weak<AddressSpace>,
    clock: Arc<Clock>,
    hpm_timer: Option<Timer>,
    notify: OnceLock<Notifier>,
}

impl fmt::Debug for RiscvIommu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RiscvIommu").finish_non_exhaustive()
    }
}

impl RiscvIommu {
    /// An IOMMU with `conf`, reading its tables from `memory` and counting cycles on
    /// `clock`, the virtual clock. The front end gives its interrupt generation support and
    /// the vectors it has, `icvec_avail_vectors`.
    pub fn new(
        conf: RiscvIommuConf,
        igs: RiscvIommuIgs,
        icvec_avail: u64,
        memory: &Arc<AddressSpace>,
        clock: &Arc<Clock>,
    ) -> Arc<Self> {
        let state = State::new(conf, igs, icvec_avail);
        let hpm = state.cap & CAP_HPM != 0;
        Arc::new_cyclic(|me: &Weak<RiscvIommu>| {
            let hpm_timer = hpm.then(|| {
                let me = me.clone();
                clock.new_timer(move || {
                    if let Some(s) = me.upgrade() {
                        s.with_state(|s, m| s.hpm_timer_cb(m));
                    }
                })
            });
            RiscvIommu {
                state: Mutex::new(state),
                memory: Arc::downgrade(memory),
                clock: Arc::clone(clock),
                hpm_timer,
                notify: OnceLock::new(),
            }
        })
    }

    /// Sets what the interrupt vectors go to, the `notify` hook of the front end. Without
    /// it no interrupt is pending in `ipsr`.
    pub fn set_notify(&self, notify: impl Fn(u32) + Send + Sync + 'static) {
        if self.notify.set(Box::new(notify)).is_ok() {
            self.lock().has_notify = true;
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Runs `f` on the state, then raises the interrupt vectors it asked for once the lock
    /// is dropped.
    fn with_state<R>(&self, f: impl FnOnce(&mut State, &Env<'_>) -> R) -> R {
        let memory = self.memory.upgrade();
        let env =
            Env { dma: Dma(memory.as_deref()), clock: &self.clock, timer: self.hpm_timer.as_ref() };
        let (r, irqs) = {
            let _engaged = Engaged::enter();
            let mut s = self.lock();
            let r = f(&mut s, &env);
            (r, std::mem::take(&mut s.irqs))
        };
        if let Some(notify) = self.notify.get() {
            for v in irqs {
                notify(v);
            }
        }
        r
    }

    /// `riscv_iommu_reset()`.
    pub fn reset(&self) {
        self.lock().reset();
    }

    /// The configuration.
    pub fn conf(&self) -> RiscvIommuConf {
        self.lock().conf.clone()
    }

    /// The `capabilities` register.
    pub fn cap(&self) -> u64 {
        self.lock().cap
    }

    /// The `fctl` register, which says whether the front end should use wired interrupts.
    pub fn fctl(&self) -> u64 {
        self.lock().get64(REG_FCTL)
    }

    /// The register frame, [`RISCV_IOMMU_REG_SIZE`] bytes. The front end puts its MSI-X
    /// table over it from [`RISCV_IOMMU_REG_MSI_CONFIG`].
    pub fn mmio_ops(self: &Arc<Self>) -> Arc<dyn MmioOps> {
        Arc::new(RiscvIommuMmio(Arc::clone(self)))
    }

    /// The IOMMU region callbacks of the PCI function `bdf`, `riscv_iommu_space()`. Its
    /// device ID has the `bus` property in bits 15:8.
    pub fn device_ops(self: &Arc<Self>, bdf: u16) -> Arc<dyn IommuOps> {
        Arc::new(RiscvIommuDevice { iommu: Arc::clone(self), bdf })
    }

    fn devid(&self, bdf: u16) -> u32 {
        u32::from(bdf) | (self.lock().conf.bus << 8)
    }

    /// `riscv_iommu_memory_region_translate()` for the PCI function `bdf` with the process
    /// ID `pid`.
    pub fn translate(
        &self,
        bdf: u16,
        pid: u32,
        addr: u64,
        flag: IommuAccessFlags,
    ) -> IommuTlbEntry {
        let target_as = self.memory.upgrade();
        let fail = |target_as| IommuTlbEntry {
            target_as,
            iova: addr,
            translated_addr: 0,
            addr_mask: 0,
            perm: IommuAccessFlags::NONE,
        };
        if engaged() {
            return fail(target_as);
        }
        let devid = self.devid(bdf);
        let t = self.with_state(|s, m| s.region_translate(m, devid, pid, addr, perm_bits(flag)));
        if t.trap {
            // QEMU sends this to its trap region, which only takes MSIs.
            return fail(target_as);
        }
        IommuTlbEntry {
            target_as,
            iova: t.iova,
            translated_addr: t.translated,
            addr_mask: t.mask,
            perm: perm_flags(t.perm),
        }
    }

    /// An MSI write of `data` to `addr` from the PCI function `requester_id`. It is
    /// translated like any write, and one that lands in the MSI page table window goes
    /// through `riscv_iommu_trap_write()`, which does the write itself. Returns the address
    /// and data for the board to write, or `None` when the IOMMU has taken care of the MSI
    /// or dropped it.
    pub fn remap_msi(&self, requester_id: u16, addr: u64, data: u32) -> Option<(u64, u32)> {
        if engaged() {
            return None;
        }
        let devid = self.devid(requester_id);
        self.with_state(|s, m| {
            let t = s.region_translate(m, devid, 0, addr, PERM_WO);
            if t.perm & PERM_WO == 0 {
                return None;
            }
            let out = (t.translated & !t.mask) | (addr & t.mask);
            if t.trap {
                s.trap_write(m, devid, out, u64::from(data), 4);
                return None;
            }
            Some((out, data))
        })
    }
}

/// The IOMMU region of one PCI function, `RISCVIOMMUSpace`.
struct RiscvIommuDevice {
    iommu: Arc<RiscvIommu>,
    bdf: u16,
}

impl IommuOps for RiscvIommuDevice {
    fn translate(&self, addr: u64, flag: IommuAccessFlags, iommu_idx: u32) -> IommuTlbEntry {
        self.iommu.translate(self.bdf, iommu_idx, addr, flag)
    }

    /// `riscv_iommu_memory_region_index()`: the process ID.
    fn attrs_to_index(&self, attrs: MemTxAttrs) -> u32 {
        if attrs.unspecified() { 0 } else { u32::from(attrs.pid()) }
    }
}

/// The register frame, `riscv_iommu_mmio_ops`.
struct RiscvIommuMmio(Arc<RiscvIommu>);

impl fmt::Debug for RiscvIommuMmio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RiscvIommuMmio")
    }
}

impl MmioOps for RiscvIommuMmio {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        if engaged() {
            return Err(MemTxResult::ACCESS_ERROR);
        }
        self.0.with_state(|s, m| s.mmio_read(m, offset, size.bytes() as usize))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        if engaged() {
            return Err(MemTxResult::ACCESS_ERROR);
        }
        self.0.with_state(|s, m| s.mmio_write(m, offset, value, size.bytes() as usize))
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
