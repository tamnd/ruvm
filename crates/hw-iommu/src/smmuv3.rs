// SPDX-License-Identifier: GPL-2.0-or-later

//! The `arm-smmuv3` device, from hw/arm/smmuv3.c and hw/arm/smmuv3-internal.h.
//!
//! The SMMU translates the DMA of the PCI devices of one root bus. Each device gets an IOMMU
//! region from [`SmmuV3::device_ops`], keyed by its stream ID, and does its DMA through an
//! address space over that region. The stream table, the context descriptors, the page tables
//! and the queues are read from and written to the system address space given at creation.
//!
//! What is there: stage 1, stage 2 and nested translation with 4K, 16K and 64K granules,
//! linear and 2-level stream tables, the command queue with the configuration and TLB
//! invalidation commands including range invalidation, the event queue, GERROR, GBPA, and the
//! four wired interrupts.
//!
//! # Differences from QEMU
//!
//! - There are no IOMMU notifiers and no replay, so nothing like vhost or VFIO can sit behind
//!   the SMMU. The `accel` mode and CMDQV are not there either.
//! - There is no secure state and no VMState.
//! - The SMMU covers one root bus, and the stream ID of a device is fixed when its region is
//!   made, where QEMU computes it from the bus number at each translation.
//! - Guest errors that QEMU logs with `LOG_GUEST_ERROR` are silent.

use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use ruvm_hw_core::IrqLine;
use ruvm_mem::{
    AccessConstraints, AccessCtx, AccessSize, AddressSpace, IommuAccessFlags, IommuOps,
    IommuTlbEntry, MemResult, MemTxAttrs, MemTxResult, MmioOps,
};

use crate::smmu_common::{
    Iotlb, PERM_RO, PERM_RW, PERM_WO, PtwError, PtwEventInfo, Stage, TlbEntry, TransCfg,
    VMSA_MAX_S2_CONCAT, WalkMemory, extract64, get_start_level, mask64, pgd_concat_idx,
    smmu_translate,
};

/// The size of the register frame, two 64K pages.
pub const SMMU_SIZE: u64 = 0x20000;
/// The number of interrupt lines: eventq, priq, cmdq-sync and gerror.
pub const SMMU_NUM_IRQS: usize = 4;

/// `SMMU_IRQ_EVTQ`.
const IRQ_EVTQ: usize = 0;
/// `SMMU_IRQ_CMD_SYNC`.
const IRQ_CMD_SYNC: usize = 2;
/// `SMMU_IRQ_GERROR`.
const IRQ_GERROR: usize = 3;

const A_IDR0: u64 = 0x0;
const A_IDR5: u64 = 0x14;
const A_IIDR: u64 = 0x18;
const A_AIDR: u64 = 0x1c;
const A_CR0: u64 = 0x20;
const A_CR0ACK: u64 = 0x24;
const A_CR1: u64 = 0x28;
const A_CR2: u64 = 0x2c;
const A_STATUSR: u64 = 0x40;
const A_GBPA: u64 = 0x44;
const A_IRQ_CTRL: u64 = 0x50;
const A_IRQ_CTRL_ACK: u64 = 0x54;
const A_GERROR: u64 = 0x60;
const A_GERRORN: u64 = 0x64;
const A_GERROR_IRQ_CFG0: u64 = 0x68;
const A_GERROR_IRQ_CFG1: u64 = 0x70;
const A_GERROR_IRQ_CFG2: u64 = 0x74;
const A_STRTAB_BASE: u64 = 0x80;
const A_STRTAB_BASE_CFG: u64 = 0x88;
const A_CMDQ_BASE: u64 = 0x90;
const A_CMDQ_PROD: u64 = 0x98;
const A_CMDQ_CONS: u64 = 0x9c;
const A_EVENTQ_BASE: u64 = 0xa0;
const A_EVENTQ_PROD: u64 = 0xa8;
const A_EVENTQ_CONS: u64 = 0xac;
const A_EVENTQ_IRQ_CFG0: u64 = 0xb0;
const A_EVENTQ_IRQ_CFG1: u64 = 0xb8;
const A_EVENTQ_IRQ_CFG2: u64 = 0xbc;
const A_IDREGS: u64 = 0xfd0;

/// The ID registers at 0xfd0, `smmuv3_idreg`.
const SMMUV3_IDREGS: [u8; 12] = [0x04, 0, 0, 0, 0x84, 0xB4, 0xF0, 0x10, 0x0D, 0xF0, 0x05, 0xB1];

const CR0_SMMUEN: u32 = 1 << 0;
const CR0_EVENTQEN: u32 = 1 << 2;
const CR0_CMDQEN: u32 = 1 << 3;
/// `SMMU_CR0_RESERVED`.
const CR0_RESERVED: u32 = 0xFFFF_FA20;
const GBPA_ABORT: u32 = 1 << 20;
const GBPA_UPDATE: u32 = 1 << 31;
/// `SMMU_GBPA_RESET_VAL`.
const GBPA_RESET_VAL: u32 = 0x1000;
const IRQ_CTRL_GERROR_IRQEN: u32 = 1 << 0;
const IRQ_CTRL_EVENTQ_IRQEN: u32 = 1 << 2;
const GERROR_CMDQ_ERR: u32 = 1 << 0;
const GERROR_EVENTQ_ABT_ERR: u32 = 1 << 2;

/// `SMMU_BASE_ADDR_MASK`.
const SMMU_BASE_ADDR_MASK: u64 = 0xfffffffffffc0;
/// `SMMU_CMDQS`.
const SMMU_CMDQS: u8 = 19;
/// `SMMU_EVENTQS`.
const SMMU_EVENTQS: u8 = 19;
/// `SMMU_IDR1_SIDSIZE`.
const SMMU_IDR1_SIDSIZE: u32 = 16;
/// `SMMU_IDR5_OAS`: 44 bits.
const SMMU_IDR5_OAS: u32 = 4;
/// `SMMU_FEATURE_2LVL_STE`.
const FEATURE_2LVL_STE: u32 = 1 << 0;

/// `SMMUCmdError`.
const CERROR_ILL: u32 = 1;
const CERROR_ABT: u32 = 2;

/// `SMMUEventType`.
mod evt {
    pub(super) const NONE: u8 = 0;
    pub(super) const C_BAD_STREAMID: u8 = 0x02;
    pub(super) const F_STE_FETCH: u8 = 0x03;
    pub(super) const C_BAD_STE: u8 = 0x04;
    pub(super) const F_CD_FETCH: u8 = 0x09;
    pub(super) const C_BAD_CD: u8 = 0x0a;
    pub(super) const F_WALK_EABT: u8 = 0x0b;
    pub(super) const F_TRANSLATION: u8 = 0x10;
    pub(super) const F_ADDR_SIZE: u8 = 0x11;
    pub(super) const F_ACCESS: u8 = 0x12;
    pub(super) const F_PERMISSION: u8 = 0x13;
}

/// `SMMUTranslationClass`.
const CLASS_CD: u8 = 0;
const CLASS_TT: u8 = 1;
const CLASS_IN: u8 = 2;

/// `SMMUCommandType`.
mod cmd {
    pub(super) const PREFETCH_CONFIG: u8 = 0x01;
    pub(super) const PREFETCH_ADDR: u8 = 0x02;
    pub(super) const CFGI_STE: u8 = 0x03;
    pub(super) const CFGI_STE_RANGE: u8 = 0x04;
    pub(super) const CFGI_CD: u8 = 0x05;
    pub(super) const CFGI_CD_ALL: u8 = 0x06;
    pub(super) const TLBI_NH_ALL: u8 = 0x10;
    pub(super) const TLBI_NH_ASID: u8 = 0x11;
    pub(super) const TLBI_NH_VA: u8 = 0x12;
    pub(super) const TLBI_NH_VAA: u8 = 0x13;
    pub(super) const TLBI_EL3_ALL: u8 = 0x18;
    pub(super) const TLBI_EL3_VA: u8 = 0x1a;
    pub(super) const TLBI_EL2_ALL: u8 = 0x20;
    pub(super) const TLBI_EL2_ASID: u8 = 0x21;
    pub(super) const TLBI_EL2_VA: u8 = 0x22;
    pub(super) const TLBI_EL2_VAA: u8 = 0x23;
    pub(super) const TLBI_S12_VMALL: u8 = 0x28;
    pub(super) const TLBI_S2_IPA: u8 = 0x2a;
    pub(super) const TLBI_NSNH_ALL: u8 = 0x30;
    pub(super) const ATC_INV: u8 = 0x40;
    pub(super) const PRI_RESP: u8 = 0x41;
    pub(super) const RESUME: u8 = 0x44;
    pub(super) const STALL_TERM: u8 = 0x45;
    pub(super) const SYNC: u8 = 0x46;
}

/// `extract32()`.
fn ex32(v: u32, start: u32, len: u32) -> u32 {
    (v >> start) & (mask64(len) as u32)
}

/// `deposit64()`.
fn deposit64(v: u64, start: u32, len: u32, field: u64) -> u64 {
    let mask = mask64(len) << start;
    (v & !mask) | ((field << start) & mask)
}

/// `oas2bits()`.
fn oas2bits(oas: u32) -> u8 {
    match oas {
        0 => 32,
        1 => 36,
        2 => 40,
        3 => 42,
        4 => 44,
        _ => 48,
    }
}

/// `tg2granule()`: the granule shift of a CD TGx field, or 0 for a reserved value.
fn tg2granule(bits: u32, ttbr: usize) -> u8 {
    match (bits, ttbr) {
        (0, 0) => 12,
        (1, 0) => 16,
        (2, 0) => 14,
        (1, _) => 14,
        (2, _) => 12,
        (3, _) => 16,
        _ => 0,
    }
}

/// `dma_aligned_pow2_mask()`: the mask of the largest naturally aligned power of two range
/// that starts at `start` and stays within `end`.
fn dma_aligned_pow2_mask(start: u64, end: u64) -> u64 {
    let addr_mask = end.wrapping_sub(start);
    let alignment_mask = if start != 0 { (start & start.wrapping_neg()) - 1 } else { u64::MAX };
    if alignment_mask <= addr_mask {
        return alignment_mask;
    }
    if addr_mask == u64::MAX {
        return u64::MAX;
    }
    (1u64 << (63 - (addr_mask + 1).leading_zeros())) - 1
}

/// What a translation came to, `SMMUTranslationStatus`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Disable,
    Abort,
    Bypass,
    Error,
    Success,
}

/// The details of an event, the fields of `SMMUEventInfo` that are ever set.
#[derive(Clone, Copy, Debug, Default)]
struct EventInfo {
    ty: u8,
    sid: u32,
    s2: bool,
    /// ADDR: the input address of a walk fault, or the CD address of `F_CD_FETCH`.
    addr: u64,
    rnw: bool,
    class: u8,
    /// ADDR2: the IPA or descriptor address of a walk fault, or the STE address of
    /// `F_STE_FETCH`.
    addr2: u64,
}

/// `SMMUQueue`.
#[derive(Clone, Copy, Debug, Default)]
struct Queue {
    base: u64,
    prod: u32,
    cons: u32,
    entry_size: u32,
    log2size: u8,
}

impl Queue {
    fn wrap_mask(&self) -> u32 {
        1 << self.log2size
    }

    fn wrap_index_mask(&self) -> u32 {
        (1u32 << (self.log2size + 1)) - 1
    }

    /// `smmuv3_q_full()`.
    fn full(&self) -> bool {
        ((self.cons ^ self.prod) & self.wrap_index_mask()) == self.wrap_mask()
    }

    /// `smmuv3_q_empty()`.
    fn empty(&self) -> bool {
        (self.cons & self.wrap_index_mask()) == (self.prod & self.wrap_index_mask())
    }

    fn entry(&self, p: u32) -> u64 {
        let idx = p & (self.wrap_mask() - 1);
        (self.base & SMMU_BASE_ADDR_MASK) + u64::from(self.entry_size) * u64::from(idx)
    }

    /// `queue_prod_incr()`.
    fn prod_incr(&mut self) {
        self.prod = (self.prod + 1) & self.wrap_index_mask();
    }

    /// `queue_cons_incr()`: the error field above the index is kept.
    fn cons_incr(&mut self) {
        let m = self.wrap_index_mask();
        self.cons = (self.cons & !m) | (self.cons.wrapping_add(1) & m);
    }
}

/// Memory accesses of the SMMU itself, through the system address space.
struct Dma<'a>(Option<&'a AddressSpace>);

impl Dma<'_> {
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.0.is_some_and(|a| a.read(addr, MemTxAttrs::UNSPECIFIED, buf).is_ok())
    }

    fn write(&self, addr: u64, buf: &[u8]) -> bool {
        self.0.is_some_and(|a| a.write(addr, MemTxAttrs::UNSPECIFIED, buf).is_ok())
    }

    fn read_words<const N: usize>(&self, addr: u64) -> Option<[u32; N]> {
        let mut b = vec![0u8; N * 4];
        if !self.read(addr, &mut b) {
            return None;
        }
        let mut w = [0u32; N];
        for (i, c) in b.chunks_exact(4).enumerate() {
            w[i] = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
        Some(w)
    }
}

impl WalkMemory for Dma<'_> {
    fn ldq_le(&self, addr: u64) -> Option<u64> {
        let mut b = [0u8; 8];
        self.read(addr, &mut b).then(|| u64::from_le_bytes(b))
    }
}

type Ste = [u32; 16];
type Cd = [u32; 16];

fn ste_s2t0sz(s: &Ste) -> u32 {
    ex32(s[5], 0, 6)
}

fn ste_s2ttb(s: &Ste) -> u64 {
    (u64::from(ex32(s[7], 0, 20)) << 32) | (u64::from(ex32(s[6], 4, 28)) << 4)
}

fn ste_ctxptr(s: &Ste) -> u64 {
    (u64::from(ex32(s[1], 0, 24)) << 32) | (u64::from(ex32(s[0], 6, 26)) << 6)
}

fn cd_tsz(cd: &Cd, i: usize) -> u32 {
    ex32(cd[0], 16 * i as u32, 6)
}

fn cd_tg(cd: &Cd, i: usize) -> u32 {
    ex32(cd[0], 16 * i as u32 + 6, 2)
}

fn cd_epd(cd: &Cd, i: usize) -> bool {
    ex32(cd[0], 16 * i as u32 + 14, 1) != 0
}

fn cd_ttb(cd: &Cd, i: usize) -> u64 {
    let lo = cd[2 + 2 * i];
    let hi = cd[3 + 2 * i];
    (u64::from(ex32(hi, 0, 20)) << 32) | (u64::from(ex32(lo, 4, 28)) << 4)
}

fn cd_had(cd: &Cd, i: usize) -> bool {
    ex32(cd[2 + 2 * i], 1, 1) != 0
}

type Cmd = [u32; 4];

fn cmd_type(c: &Cmd) -> u8 {
    ex32(c[0], 0, 8) as u8
}

fn cmd_ssec(c: &Cmd) -> bool {
    ex32(c[0], 10, 1) != 0
}

fn cmd_vmid(c: &Cmd) -> i32 {
    ex32(c[1], 0, 16) as i32
}

fn cmd_asid(c: &Cmd) -> i32 {
    ex32(c[1], 16, 16) as i32
}

fn cmd_addr(c: &Cmd) -> u64 {
    (u64::from(c[3]) << 32) | (u64::from(ex32(c[2], 12, 20)) << 12)
}

/// The registers and caches of the SMMU, `SMMUv3State` with its `SMMUState`.
struct State {
    idr: [u32; 6],
    iidr: u32,
    aidr: u32,
    cr: [u32; 3],
    cr0ack: u32,
    statusr: u32,
    gbpa: u32,
    irq_ctrl: u32,
    gerror: u32,
    gerrorn: u32,
    gerror_irq_cfg0: u64,
    gerror_irq_cfg1: u32,
    gerror_irq_cfg2: u32,
    strtab_base: u64,
    strtab_base_cfg: u32,
    eventq_irq_cfg0: u64,
    eventq_irq_cfg1: u32,
    eventq_irq_cfg2: u32,
    features: u32,
    sid_split: u8,
    cmdq: Queue,
    eventq: Queue,
    /// The decoded configurations by stream ID, `SMMUState.configs`.
    configs: HashMap<u32, TransCfg>,
    iotlb: Iotlb,
    /// The interrupt lines to pulse once the lock is dropped.
    pulses: Vec<usize>,
}

impl State {
    fn new(stages: Stage) -> Self {
        let mut idr = [0u32; 6];
        // S2P is bit 0 and S1P bit 1.
        idr[0] = match stages {
            Stage::S1 => 1 << 1,
            Stage::S2 => 1 << 0,
            Stage::Nested => (1 << 1) | (1 << 0),
        };
        // TTF = AArch64, COHACC, ASID16, VMID16, TTENDIAN = little, STALL_MODEL = no stall,
        // TERM_MODEL = abort, STLEVEL = 2-level stream tables.
        idr[0] |= (2 << 2) | (1 << 4) | (1 << 12) | (1 << 18) | (2 << 21) | (1 << 24) | (1 << 26);
        idr[0] |= 1 << 27;
        idr[1] =
            SMMU_IDR1_SIDSIZE | (u32::from(SMMU_EVENTQS) << 16) | (u32::from(SMMU_CMDQS) << 21);
        // HAD, XNX with stage 2, RIL and BBML = 2.
        idr[3] = 1 << 2;
        if idr[0] & 1 != 0 {
            idr[3] |= 1 << 4;
        }
        idr[3] |= (1 << 10) | (2 << 11);
        // OAS = 44 bits, and the 4K, 16K and 64K granules.
        idr[5] = SMMU_IDR5_OAS | (1 << 4) | (1 << 5) | (1 << 6);
        let mut s = State {
            idr,
            iidr: 0,
            aidr: 1,
            cr: [0; 3],
            cr0ack: 0,
            statusr: 0,
            gbpa: 0,
            irq_ctrl: 0,
            gerror: 0,
            gerrorn: 0,
            gerror_irq_cfg0: 0,
            gerror_irq_cfg1: 0,
            gerror_irq_cfg2: 0,
            strtab_base: 0,
            strtab_base_cfg: 0,
            eventq_irq_cfg0: 0,
            eventq_irq_cfg1: 0,
            eventq_irq_cfg2: 0,
            features: 0,
            sid_split: 0,
            cmdq: Queue::default(),
            eventq: Queue::default(),
            configs: HashMap::new(),
            iotlb: Iotlb::new(),
            pulses: Vec::new(),
        };
        s.reset();
        s
    }

    /// `smmu_base_reset_exit()` and `smmuv3_reset()`. The table bases, the IRQ configuration
    /// and CR1 and CR2 keep their values, as in QEMU.
    fn reset(&mut self) {
        self.configs.clear();
        self.iotlb.inv_all();
        self.cmdq.base = deposit64(self.cmdq.base, 0, 5, u64::from(SMMU_CMDQS));
        self.cmdq.prod = 0;
        self.cmdq.cons = 0;
        self.cmdq.entry_size = 16;
        self.eventq.base = deposit64(self.eventq.base, 0, 5, u64::from(SMMU_EVENTQS));
        self.eventq.prod = 0;
        self.eventq.cons = 0;
        self.eventq.entry_size = 32;
        self.features = 0;
        self.sid_split = 0;
        self.cr[0] = 0;
        self.cr0ack = 0;
        self.irq_ctrl = 0;
        self.gerror = 0;
        self.gerrorn = 0;
        self.statusr = 0;
        self.gbpa = GBPA_RESET_VAL;
    }

    fn stage1_supported(&self) -> bool {
        self.idr[0] & (1 << 1) != 0
    }

    fn stage2_supported(&self) -> bool {
        self.idr[0] & 1 != 0
    }

    fn idr5_oas(&self) -> u32 {
        ex32(self.idr[5], 0, 3)
    }

    /// `smmuv3_trigger_irq()`.
    fn trigger_irq(&mut self, irq: usize, gerror_mask: u32) {
        let pulse = match irq {
            IRQ_EVTQ => self.irq_ctrl & IRQ_CTRL_EVENTQ_IRQEN != 0,
            IRQ_CMD_SYNC => true,
            IRQ_GERROR => {
                let pending = self.gerror ^ self.gerrorn;
                let new = !pending & gerror_mask;
                if new == 0 {
                    // Only errors that are not pending toggle.
                    return;
                }
                self.gerror ^= new;
                self.irq_ctrl & IRQ_CTRL_GERROR_IRQEN != 0
            }
            // PRI is not supported.
            _ => false,
        };
        if pulse {
            self.pulses.push(irq);
        }
    }

    /// `smmuv3_write_eventq()`.
    fn write_eventq(&mut self, m: &Dma<'_>, words: &[u32; 8]) -> bool {
        if self.cr[0] & CR0_EVENTQEN == 0 || self.eventq.full() {
            return false;
        }
        let mut b = [0u8; 32];
        for (i, w) in words.iter().enumerate() {
            b[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        if !m.write(self.eventq.entry(self.eventq.prod), &b) {
            return false;
        }
        self.eventq.prod_incr();
        if !self.eventq.empty() {
            self.trigger_irq(IRQ_EVTQ, 0);
        }
        true
    }

    /// `smmuv3_record_event()` with `smmuv3_propagate_event()`.
    fn record_event(&mut self, m: &Dma<'_>, info: &EventInfo) {
        if self.cr[0] & CR0_EVENTQEN == 0 || info.ty == evt::NONE {
            return;
        }
        let mut w = [0u32; 8];
        w[0] = u32::from(info.ty);
        w[1] = info.sid;
        let set_addr = |w: &mut [u32; 8], i: usize, a: u64| {
            w[i] = a as u32;
            w[i + 1] = (a >> 32) as u32;
        };
        match info.ty {
            evt::F_STE_FETCH => set_addr(&mut w, 6, info.addr2),
            evt::F_CD_FETCH => set_addr(&mut w, 4, info.addr),
            evt::F_WALK_EABT
            | evt::F_TRANSLATION
            | evt::F_ADDR_SIZE
            | evt::F_ACCESS
            | evt::F_PERMISSION => {
                w[3] = (u32::from(info.s2) << 7)
                    | (u32::from(info.rnw) << 3)
                    | (u32::from(info.class & 3) << 8);
                set_addr(&mut w, 4, info.addr);
                set_addr(&mut w, 6, info.addr2);
            }
            // The rest only carry the SSID, which is always 0 here.
            _ => {}
        }
        if !self.write_eventq(m, &w) {
            self.trigger_irq(IRQ_GERROR, GERROR_EVENTQ_ABT_ERR);
        }
    }

    /// `smmu_find_ste()`.
    fn find_ste(&self, m: &Dma<'_>, sid: u32, ev: &mut EventInfo) -> Option<Ste> {
        let log2size = ex32(self.strtab_base_cfg, 0, 6);
        if u64::from(sid) >= 1u64 << log2size.min(SMMU_IDR1_SIDSIZE) {
            ev.ty = evt::C_BAD_STREAMID;
            return None;
        }
        let addr = if self.features & FEATURE_2LVL_STE != 0 {
            let split = u32::from(self.sid_split);
            // Align the base to the size of the first level, ignoring SIDSIZE.
            let strtab_size = (log2size as i32 - split as i32 + 3).max(6) as u32;
            let base = self.strtab_base & SMMU_BASE_ADDR_MASK & !mask64(strtab_size);
            let l1ptr = base + u64::from(sid >> split) * 8;
            let Some(l1) = m.read_words::<2>(l1ptr) else {
                ev.ty = evt::F_STE_FETCH;
                ev.addr2 = l1ptr;
                return None;
            };
            let span = ex32(l1[0], 0, 5);
            if span == 0 || span > 11 || span > split + 1 {
                // The L2 pointer is not valid.
                ev.ty = evt::C_BAD_STREAMID;
                return None;
            }
            let max_l2_ste = (1u32 << span) - 1;
            let l2ptr =
                ((u64::from(l1[1]) << 32) | u64::from(l1[0] & !0x1f)) & !mask64(6 + span - 1);
            let l2_off = sid & ((1u32 << split) - 1);
            if l2_off > max_l2_ste {
                ev.ty = evt::C_BAD_STE;
                return None;
            }
            l2ptr.wrapping_add(u64::from(l2_off) * 64)
        } else {
            let strtab_size = (log2size + 6).min(64);
            let base = self.strtab_base & SMMU_BASE_ADDR_MASK & !mask64(strtab_size);
            base.wrapping_add(u64::from(sid) * 64)
        };
        match m.read_words::<16>(addr) {
            Some(s) => Some(s),
            None => {
                ev.ty = evt::F_STE_FETCH;
                ev.addr2 = addr;
                None
            }
        }
    }

    /// `decode_ste_s2_cfg()`.
    fn decode_ste_s2_cfg(&self, cfg: &mut TransCfg, ste: &Ste) -> Option<()> {
        let oas = self.idr5_oas();
        // S2AA64 = 0 is an assertion in QEMU, as STE.S2AA64 is RES1 without AArch32.
        if ex32(ste[5], 19, 1) == 0 {
            return None;
        }
        let s2 = &mut cfg.s2cfg;
        s2.granule_sz = match ex32(ste[5], 14, 2) {
            0 => 12,
            1 => 16,
            2 => 14,
            _ => return None,
        };
        s2.vttb = ste_s2ttb(ste);
        s2.sl0 = ex32(ste[5], 6, 2) as u8;
        // FEAT_TTST is not supported.
        if s2.sl0 == 3 {
            return None;
        }
        // For AArch64 the effective S2PS is capped to the OAS, and to 48 bits unless the
        // granule is 64K.
        s2.eff_ps = oas2bits(ex32(ste[5], 16, 3).min(oas));
        if s2.granule_sz != 16 {
            s2.eff_ps = s2.eff_ps.min(48);
        }
        if s2.vttb & !mask64(u32::from(s2.eff_ps)) != 0 {
            return None;
        }
        s2.tsz = ste_s2t0sz(ste) as u8;
        // s2t0sz_valid(): 39 at most, and at least 64 - IAS, and 16 for small granules.
        let min = 64 - s2.eff_ps;
        let ok =
            s2.tsz <= 39 && if s2.granule_sz == 16 { s2.tsz >= min } else { s2.tsz >= min.max(16) };
        if !ok {
            return None;
        }
        // s2_pgtable_config_valid(): at most 16 concatenated tables at the start level.
        let gran = i32::from(s2.granule_sz);
        let level = get_start_level(i32::from(s2.sl0), gran);
        let max_ipa = mask64(64 - u32::from(s2.tsz));
        if pgd_concat_idx(level, gran, max_ipa) + 1 > VMSA_MAX_S2_CONCAT {
            return None;
        }
        // Only little endian tables (IDR0.TTENDIAN).
        if ex32(ste[5], 20, 1) != 0 {
            return None;
        }
        s2.affd = ex32(ste[5], 21, 1) != 0;
        s2.record_faults = ex32(ste[5], 26, 1) != 0;
        // Stall is not supported.
        if ex32(ste[5], 25, 1) != 0 {
            return None;
        }
        Some(())
    }

    /// `decode_ste()`.
    fn decode_ste(&self, cfg: &mut TransCfg, ste: &Ste, ev: &mut EventInfo) -> Option<()> {
        let r = self.decode_ste_inner(cfg, ste);
        if r.is_none() {
            ev.ty = evt::C_BAD_STE;
        }
        r
    }

    fn decode_ste_inner(&self, cfg: &mut TransCfg, ste: &Ste) -> Option<()> {
        if ste[0] & 1 == 0 {
            return None;
        }
        let config = ex32(ste[0], 1, 3);
        // decode_ste_config()
        if config & 4 == 0 {
            cfg.aborted = true;
            return Some(());
        }
        if config == 4 {
            cfg.bypassed = true;
            return Some(());
        }
        let s1 = config & 1 != 0;
        let s2 = config & 2 != 0;
        cfg.stage = match (s1, s2) {
            (true, true) => Stage::Nested,
            (false, true) => Stage::S2,
            _ => Stage::S1,
        };
        // A stage that is enabled but not advertised is a bad STE.
        if (!self.stage1_supported() && s1) || (!self.stage2_supported() && s2) {
            return None;
        }
        // The VMID counts even with stage 2 disabled.
        cfg.s2cfg.vmid = if self.stage2_supported() { ex32(ste[4], 0, 16) as i32 } else { -1 };
        if s2 {
            // The stage 1 OAS is used for the input check of stage 2 even without stage 1.
            cfg.oas = oas2bits(self.idr5_oas());
            self.decode_ste_s2_cfg(cfg, ste)?;
        }
        // Several CDs need substream support, which is not there.
        if ex32(ste[1], 27, 5) != 0 {
            return None;
        }
        // S1STALLD.
        if ex32(ste[2], 27, 1) != 0 {
            return None;
        }
        Some(())
    }

    /// `smmu_get_cd()` for SSID 0.
    fn get_cd(&mut self, m: &Dma<'_>, ste: &Ste, cfg: &TransCfg, ev: &mut EventInfo) -> Option<Cd> {
        let mut addr = ste_ctxptr(ste);
        if cfg.stage == Stage::Nested {
            // The same walk faults are reported, with the CD class.
            let e = self.do_translate(m, addr, cfg, ev, PERM_RO, CLASS_CD)?;
            addr = e.to_addr(addr);
        }
        match m.read_words::<16>(addr) {
            Some(cd) => Some(cd),
            None => {
                ev.ty = evt::F_CD_FETCH;
                ev.addr = addr;
                None
            }
        }
    }

    /// `decode_cd()`.
    fn decode_cd(
        &mut self,
        m: &Dma<'_>,
        cfg: &mut TransCfg,
        cd: &Cd,
        ev: &mut EventInfo,
    ) -> Option<()> {
        let bad = |ev: &mut EventInfo| {
            ev.ty = evt::C_BAD_CD;
            None
        };
        let valid = ex32(cd[0], 31, 1) != 0;
        let aarch64 = ex32(cd[1], 9, 1) != 0;
        // A = 0 is not allowed with TERM_MODEL = 1, S = 1 not with STALL_MODEL = 1, and there
        // is no HTTU.
        let a = ex32(cd[1], 14, 1) != 0;
        let s = ex32(cd[1], 12, 1) != 0;
        let ha_hd = ex32(cd[1], 10, 2) != 0;
        if !valid || !aarch64 || !a || s || ha_hd {
            return bad(ev);
        }
        cfg.aa64 = true;
        cfg.oas = oas2bits(ex32(cd[1], 0, 3)).min(oas2bits(self.idr5_oas()));
        cfg.tbi = ex32(cd[1], 6, 2) as u8;
        cfg.asid = ex32(cd[1], 16, 16) as i32;
        cfg.affd = ex32(cd[1], 3, 1) != 0;

        for i in 0..2 {
            cfg.tt[i].disabled = cd_epd(cd, i);
            if cfg.tt[i].disabled {
                continue;
            }
            let tsz = cd_tsz(cd, i);
            if !(16..=39).contains(&tsz) {
                return bad(ev);
            }
            let granule = tg2granule(cd_tg(cd, i), i);
            cfg.tt[i].granule_sz = granule;
            if !matches!(granule, 12 | 14 | 16) || ex32(cd[0], 15, 1) != 0 {
                return bad(ev);
            }
            // An output above 48 bits needs a 64K granule.
            if granule != 16 {
                cfg.oas = cfg.oas.min(48);
            }
            cfg.tt[i].tsz = tsz as u8;
            cfg.tt[i].ttb = cd_ttb(cd, i);
            if cfg.tt[i].ttb & !mask64(u32::from(cfg.oas)) != 0 {
                return bad(ev);
            }
            // With nesting the TTB is an IPA, which goes through stage 2 here.
            if cfg.stage == Stage::Nested {
                let ttb = cfg.tt[i].ttb;
                let e = self.do_translate(m, ttb, cfg, ev, PERM_RO, CLASS_TT)?;
                cfg.tt[i].ttb = e.to_addr(ttb);
            }
            cfg.tt[i].had = cd_had(cd, i);
        }
        cfg.record_faults = ex32(cd[1], 13, 1) != 0;
        Some(())
    }

    /// `smmuv3_decode_config()`.
    fn decode_config(&mut self, m: &Dma<'_>, sid: u32, ev: &mut EventInfo) -> Option<TransCfg> {
        let mut cfg = TransCfg { asid: -1, ..TransCfg::default() };
        let ste = self.find_ste(m, sid, ev)?;
        self.decode_ste(&mut cfg, &ste, ev)?;
        if cfg.aborted || cfg.bypassed || cfg.stage == Stage::S2 {
            return Some(cfg);
        }
        let cd = self.get_cd(m, &ste, &cfg, ev)?;
        self.decode_cd(m, &mut cfg, &cd, ev)?;
        Some(cfg)
    }

    /// `smmuv3_get_config()`: the cached configuration, decoded on a miss. Nothing is cached
    /// when decoding fails.
    fn get_config(&mut self, m: &Dma<'_>, sid: u32, ev: &mut EventInfo) -> Option<TransCfg> {
        if let Some(c) = self.configs.get(&sid) {
            return Some(*c);
        }
        let cfg = self.decode_config(m, sid, ev)?;
        self.configs.insert(sid, cfg);
        Some(cfg)
    }

    /// `smmuv3_do_translate()`. A class other than IN translates a descriptor address with
    /// stage 2 only.
    fn do_translate(
        &mut self,
        m: &Dma<'_>,
        addr: u64,
        cfg: &TransCfg,
        ev: &mut EventInfo,
        flag: u8,
        class: u8,
    ) -> Option<TlbEntry> {
        let mut c = *cfg;
        if class != CLASS_IN {
            c.asid = -1;
            c.stage = Stage::S2;
        }
        let mut info = PtwEventInfo::default();
        if let Some(e) = smmu_translate(&mut self.iotlb, m, &c, addr, flag, &mut info) {
            return Some(e);
        }
        // All walk faults have the S2 field.
        ev.s2 = info.stage == Stage::S2;
        let class = if info.is_ipa_descriptor { CLASS_TT } else { class };
        let record = match info.stage {
            Stage::S2 => cfg.s2cfg.record_faults,
            _ => cfg.record_faults,
        };
        let ty = match info.ty {
            PtwError::WalkEabt => {
                ev.ty = evt::F_WALK_EABT;
                ev.rnw = flag & 1 != 0;
                ev.class = if info.stage == Stage::S2 { class } else { CLASS_TT };
                ev.addr2 = info.addr;
                return None;
            }
            PtwError::Translation => evt::F_TRANSLATION,
            PtwError::AddrSize => evt::F_ADDR_SIZE,
            PtwError::Access => evt::F_ACCESS,
            PtwError::Permission => evt::F_PERMISSION,
            PtwError::None => return None,
        };
        if record {
            ev.ty = ty;
            ev.addr2 = info.addr;
            ev.class = class;
            ev.rnw = flag & 1 != 0;
        }
        None
    }

    /// The body of `smmuv3_translate()`: the status, and the entry on success.
    fn translate(
        &mut self,
        m: &Dma<'_>,
        sid: u32,
        addr: u64,
        flag: u8,
        ev: &mut EventInfo,
    ) -> (Status, Option<TlbEntry>) {
        if self.cr[0] & CR0_SMMUEN == 0 {
            let st = if self.gbpa & GBPA_ABORT != 0 { Status::Abort } else { Status::Disable };
            return (st, None);
        }
        let Some(cfg) = self.get_config(m, sid, ev) else {
            return (Status::Error, None);
        };
        if cfg.aborted {
            return (Status::Abort, None);
        }
        if cfg.bypassed {
            return (Status::Bypass, None);
        }
        match self.do_translate(m, addr, &cfg, ev, flag, CLASS_IN) {
            Some(e) => (Status::Success, Some(e)),
            None => (Status::Error, None),
        }
    }

    /// `smmuv3_range_inval()`.
    fn range_inval(&mut self, c: &Cmd, stage: Stage) {
        let mut addr = cmd_addr(c);
        let ty = cmd_type(c);
        let scale = ex32(c[0], 20, 5);
        let num = ex32(c[0], 12, 5);
        let ttl = ex32(c[2], 8, 2) as u8;
        let tg = ex32(c[2], 10, 2) as u8;
        // The VMID only counts when stage 2 is supported.
        let vmid = if self.stage2_supported() { cmd_vmid(c) } else { -1 };
        let asid = if ty == cmd::TLBI_NH_VA { cmd_asid(c) } else { -1 };
        let inv = |tlb: &mut Iotlb, addr: u64, num_pages: u64| {
            if stage == Stage::S1 {
                tlb.inv_iova(asid, vmid, addr, tg, num_pages, ttl);
            } else {
                tlb.inv_ipa(vmid, addr, tg, num_pages, ttl);
            }
        };
        if tg == 0 {
            inv(&mut self.iotlb, addr, 1);
            return;
        }
        // RIL is in use: split the range into naturally aligned powers of two.
        let num_pages = u64::from(num + 1) << scale;
        let granule = u32::from(tg) * 2 + 10;
        let end = addr.wrapping_add(num_pages << granule).wrapping_sub(1);
        while addr != end.wrapping_add(1) {
            let mask = dma_aligned_pow2_mask(addr, end);
            let n = mask.wrapping_add(1) >> granule;
            inv(&mut self.iotlb, addr, n);
            addr = addr.wrapping_add(mask).wrapping_add(1);
        }
    }

    /// `smmuv3_cmdq_consume()`.
    fn cmdq_consume(&mut self, m: &Dma<'_>) {
        if self.cr[0] & CR0_CMDQEN == 0 {
            return;
        }
        let mut cmd_error = 0;
        while !self.cmdq.empty() {
            let pending = self.gerror ^ self.gerrorn;
            if pending & GERROR_CMDQ_ERR != 0 {
                break;
            }
            let Some(c) = m.read_words::<4>(self.cmdq.entry(self.cmdq.cons)) else {
                cmd_error = CERROR_ABT;
                break;
            };
            cmd_error = self.exec_cmd(&c);
            if cmd_error != 0 {
                break;
            }
            // The index moves once the command is done, as a SYNC returns at once and does
            // not wait for the commands before it.
            self.cmdq.cons_incr();
        }
        if cmd_error != 0 {
            // CMDQ_CONS.ERR.
            self.cmdq.cons = (self.cmdq.cons & !(0x7f << 24)) | ((cmd_error & 0x7f) << 24);
            self.trigger_irq(IRQ_GERROR, GERROR_CMDQ_ERR);
        }
    }

    /// One command of [`State::cmdq_consume`], giving the command error or 0.
    fn exec_cmd(&mut self, c: &Cmd) -> u32 {
        let s1 = self.stage1_supported();
        let s2 = self.stage2_supported();
        match cmd_type(c) {
            cmd::SYNC => {
                // CS = SIG_IRQ.
                if ex32(c[0], 12, 2) & 1 != 0 {
                    self.trigger_irq(IRQ_CMD_SYNC, 0);
                }
            }
            cmd::PREFETCH_CONFIG | cmd::PREFETCH_ADDR => {}
            cmd::CFGI_STE | cmd::CFGI_CD | cmd::CFGI_CD_ALL => {
                if cmd_ssec(c) {
                    return CERROR_ILL;
                }
                let ty = cmd_type(c);
                if ty != cmd::CFGI_STE && !s1 {
                    return CERROR_ILL;
                }
                self.configs.remove(&c[1]);
            }
            cmd::CFGI_STE_RANGE => {
                if cmd_ssec(c) {
                    return CERROR_ILL;
                }
                let range = ex32(c[2], 0, 5);
                let mask = ((1u64 << (range + 1)) - 1) as u32;
                let start = c[1] & !mask;
                let end = start.wrapping_add(mask);
                self.configs.retain(|sid, _| !(start..=end).contains(sid));
            }
            cmd::TLBI_NH_ASID => {
                if !s1 {
                    return CERROR_ILL;
                }
                // The VMID only counts when stage 2 is supported.
                let vmid = if s2 { cmd_vmid(c) } else { -1 };
                self.iotlb.inv_asid_vmid(cmd_asid(c), vmid);
            }
            cmd::TLBI_NH_ALL | cmd::TLBI_NSNH_ALL => {
                let ty = cmd_type(c);
                if ty == cmd::TLBI_NH_ALL {
                    if !s1 {
                        return CERROR_ILL;
                    }
                    // With stage 2 this is for one VMID, otherwise for everything.
                    if s2 {
                        self.iotlb.inv_vmid_s1(cmd_vmid(c));
                        return 0;
                    }
                }
                self.iotlb.inv_all();
            }
            cmd::TLBI_NH_VAA | cmd::TLBI_NH_VA => {
                if !s1 {
                    return CERROR_ILL;
                }
                self.range_inval(c, Stage::S1);
            }
            cmd::TLBI_S12_VMALL => {
                if !s2 {
                    return CERROR_ILL;
                }
                self.iotlb.inv_vmid(cmd_vmid(c));
            }
            cmd::TLBI_S2_IPA => {
                if !s2 {
                    return CERROR_ILL;
                }
                self.range_inval(c, Stage::S2);
            }
            // ATS is not advertised, so there is nothing to invalidate.
            cmd::ATC_INV => {}
            cmd::TLBI_EL3_ALL
            | cmd::TLBI_EL3_VA
            | cmd::TLBI_EL2_ALL
            | cmd::TLBI_EL2_ASID
            | cmd::TLBI_EL2_VA
            | cmd::TLBI_EL2_VAA
            | cmd::PRI_RESP
            | cmd::RESUME
            | cmd::STALL_TERM => {}
            _ => return CERROR_ILL,
        }
        0
    }

    /// `smmu_writell()`.
    fn writell(&mut self, offset: u64, data: u64) {
        match offset {
            A_GERROR_IRQ_CFG0 => self.gerror_irq_cfg0 = data,
            A_STRTAB_BASE => self.strtab_base = data,
            A_CMDQ_BASE => {
                self.cmdq.base = data;
                self.cmdq.log2size = (extract64(data, 0, 5) as u8).min(SMMU_CMDQS);
            }
            A_EVENTQ_BASE => {
                self.eventq.base = data;
                self.eventq.log2size = (extract64(data, 0, 5) as u8).min(SMMU_EVENTQS);
            }
            A_EVENTQ_IRQ_CFG0 => self.eventq_irq_cfg0 = data,
            _ => {}
        }
    }

    /// `smmu_writel()`.
    fn writel(&mut self, m: &Dma<'_>, offset: u64, data: u32) {
        let d = u64::from(data);
        match offset {
            A_CR0 => {
                self.cr[0] = data;
                self.cr0ack = data & !CR0_RESERVED;
                // In case the command queue has been enabled.
                self.cmdq_consume(m);
            }
            A_CR1 => self.cr[1] = data,
            A_CR2 => self.cr[2] = data,
            A_IRQ_CTRL => self.irq_ctrl = data,
            A_GERRORN => {
                // smmuv3_write_gerrorn(). Toggling an error that is not pending is a guest
                // error that QEMU only logs.
                self.gerrorn = data;
                // Acknowledging CMDQ_ERR lets the commands run again.
                self.cmdq_consume(m);
            }
            A_GERROR_IRQ_CFG0 => self.gerror_irq_cfg0 = deposit64(self.gerror_irq_cfg0, 0, 32, d),
            0x6c => self.gerror_irq_cfg0 = deposit64(self.gerror_irq_cfg0, 32, 32, d),
            A_GERROR_IRQ_CFG1 => self.gerror_irq_cfg1 = data,
            A_GERROR_IRQ_CFG2 => self.gerror_irq_cfg2 = data,
            A_GBPA => {
                // Only a write with UPDATE set has an effect.
                if data & GBPA_UPDATE != 0 {
                    self.gbpa = data & !GBPA_UPDATE;
                }
            }
            A_STRTAB_BASE => self.strtab_base = deposit64(self.strtab_base, 0, 32, d),
            0x84 => self.strtab_base = deposit64(self.strtab_base, 32, 32, d),
            A_STRTAB_BASE_CFG => {
                self.strtab_base_cfg = data;
                if ex32(data, 16, 2) == 1 {
                    let split = ex32(data, 6, 5) as u8;
                    // Other values are reserved and behave as 6.
                    self.sid_split = if matches!(split, 6 | 8 | 10) { split } else { 6 };
                    self.features |= FEATURE_2LVL_STE;
                }
            }
            A_CMDQ_BASE => {
                self.cmdq.base = deposit64(self.cmdq.base, 0, 32, d);
                self.cmdq.log2size = (extract64(self.cmdq.base, 0, 5) as u8).min(SMMU_CMDQS);
            }
            0x94 => self.cmdq.base = deposit64(self.cmdq.base, 32, 32, d),
            A_CMDQ_PROD => {
                self.cmdq.prod = data;
                self.cmdq_consume(m);
            }
            A_CMDQ_CONS => self.cmdq.cons = data,
            A_EVENTQ_BASE => {
                self.eventq.base = deposit64(self.eventq.base, 0, 32, d);
                self.eventq.log2size = (extract64(self.eventq.base, 0, 5) as u8).min(SMMU_EVENTQS);
            }
            0xa4 => self.eventq.base = deposit64(self.eventq.base, 32, 32, d),
            A_EVENTQ_PROD => self.eventq.prod = data,
            A_EVENTQ_CONS => self.eventq.cons = data,
            A_EVENTQ_IRQ_CFG0 => self.eventq_irq_cfg0 = deposit64(self.eventq_irq_cfg0, 0, 32, d),
            0xb4 => self.eventq_irq_cfg0 = deposit64(self.eventq_irq_cfg0, 32, 32, d),
            A_EVENTQ_IRQ_CFG1 => self.eventq_irq_cfg1 = data,
            A_EVENTQ_IRQ_CFG2 => self.eventq_irq_cfg2 = data,
            _ => {}
        }
    }

    /// `smmu_readll()`.
    fn readll(&self, offset: u64) -> u64 {
        match offset {
            A_GERROR_IRQ_CFG0 => self.gerror_irq_cfg0,
            A_STRTAB_BASE => self.strtab_base,
            A_CMDQ_BASE => self.cmdq.base,
            A_EVENTQ_BASE => self.eventq.base,
            _ => 0,
        }
    }

    /// `smmu_readl()`.
    fn readl(&self, offset: u64) -> u32 {
        let lo = |v: u64| v as u32;
        let hi = |v: u64| (v >> 32) as u32;
        match offset {
            A_IDREGS..=0xfff => {
                SMMUV3_IDREGS.get(((offset - A_IDREGS) / 4) as usize).map_or(0, |&b| u32::from(b))
            }
            A_IDR0..=A_IDR5 => self.idr[(offset / 4) as usize],
            A_IIDR => self.iidr,
            A_AIDR => self.aidr,
            A_CR0 => self.cr[0],
            A_CR0ACK => self.cr0ack,
            A_CR1 => self.cr[1],
            A_CR2 => self.cr[2],
            A_STATUSR => self.statusr,
            A_GBPA => self.gbpa,
            A_IRQ_CTRL | A_IRQ_CTRL_ACK => self.irq_ctrl,
            A_GERROR => self.gerror,
            A_GERRORN => self.gerrorn,
            A_GERROR_IRQ_CFG0 => lo(self.gerror_irq_cfg0),
            0x6c => hi(self.gerror_irq_cfg0),
            A_GERROR_IRQ_CFG1 => self.gerror_irq_cfg1,
            A_GERROR_IRQ_CFG2 => self.gerror_irq_cfg2,
            A_STRTAB_BASE => lo(self.strtab_base),
            0x84 => hi(self.strtab_base),
            A_STRTAB_BASE_CFG => self.strtab_base_cfg,
            A_CMDQ_BASE => lo(self.cmdq.base),
            0x94 => hi(self.cmdq.base),
            A_CMDQ_PROD => self.cmdq.prod,
            A_CMDQ_CONS => self.cmdq.cons,
            A_EVENTQ_BASE => lo(self.eventq.base),
            0xa4 => hi(self.eventq.base),
            A_EVENTQ_PROD => self.eventq.prod,
            A_EVENTQ_CONS => self.eventq.cons,
            A_EVENTQ_IRQ_CFG0 => lo(self.eventq_irq_cfg0),
            0xb4 => hi(self.eventq_irq_cfg0),
            A_EVENTQ_IRQ_CFG1 => self.eventq_irq_cfg1,
            A_EVENTQ_IRQ_CFG2 => self.eventq_irq_cfg2,
            _ => 0,
        }
    }
}

thread_local! {
    /// Set while this thread holds the SMMU state, so that an access that comes back to the
    /// SMMU, such as a queue placed over its own registers, fails instead of deadlocking.
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

/// The `arm-smmuv3` device.
pub struct SmmuV3 {
    state: Mutex<State>,
    irqs: [IrqLine; SMMU_NUM_IRQS],
    memory: Weak<AddressSpace>,
}

impl fmt::Debug for SmmuV3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SmmuV3").finish_non_exhaustive()
    }
}

impl SmmuV3 {
    /// An SMMU with the given `stages`, which the `stage` property picks in QEMU, reading its
    /// tables from `memory`. The lines are eventq, priq, cmdq-sync and gerror.
    pub fn new(
        stages: Stage,
        memory: &Arc<AddressSpace>,
        irqs: [IrqLine; SMMU_NUM_IRQS],
    ) -> Arc<Self> {
        Arc::new(SmmuV3 {
            state: Mutex::new(State::new(stages)),
            irqs,
            memory: Arc::downgrade(memory),
        })
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Runs `f` on the state, then pulses the lines it asked for once the lock is dropped.
    fn with_state<R>(&self, f: impl FnOnce(&mut State, &Dma<'_>) -> R) -> R {
        let memory = self.memory.upgrade();
        let (r, pulses) = {
            let _engaged = Engaged::enter();
            let mut s = self.lock();
            let r = f(&mut s, &Dma(memory.as_deref()));
            (r, std::mem::take(&mut s.pulses))
        };
        for irq in pulses {
            self.irqs[irq].pulse();
        }
        r
    }

    /// The device reset.
    pub fn reset(&self) {
        self.lock().reset();
    }

    /// The register frame, to map at the base address.
    pub fn mmio_ops(self: &Arc<Self>) -> Arc<dyn MmioOps> {
        Arc::new(SmmuMmio(Arc::clone(self)))
    }

    /// The IOMMU region callbacks of the device with stream ID `sid`, which is
    /// `PCI_BUILD_BDF(bus, devfn)`.
    pub fn device_ops(self: &Arc<Self>, sid: u32) -> Arc<dyn IommuOps> {
        Arc::new(SmmuDevice { smmu: Arc::clone(self), sid })
    }

    /// `smmuv3_translate()`.
    pub fn translate(&self, sid: u32, addr: u64, flag: IommuAccessFlags) -> IommuTlbEntry {
        let target = self.memory.upgrade();
        let mut entry = IommuTlbEntry {
            target_as: target,
            iova: addr,
            translated_addr: addr,
            addr_mask: u64::MAX,
            perm: IommuAccessFlags::NONE,
        };
        if engaged() {
            return entry;
        }
        let flag_bits = (u8::from(flag.allows(IommuAccessFlags::RO)) * PERM_RO)
            | (u8::from(flag.allows(IommuAccessFlags::WO)) * PERM_WO);
        let (status, e) = self.with_state(|s, m| {
            let mut ev = EventInfo { sid, ..EventInfo::default() };
            let (status, e) = s.translate(m, sid, addr, flag_bits, &mut ev);
            if status == Status::Error {
                // smmuv3_fixup_event(): the input address of a walk fault is only known here.
                if matches!(
                    ev.ty,
                    evt::F_WALK_EABT
                        | evt::F_TRANSLATION
                        | evt::F_ADDR_SIZE
                        | evt::F_ACCESS
                        | evt::F_PERMISSION
                ) {
                    ev.addr = addr;
                }
                s.record_event(m, &ev);
            }
            (status, e)
        });
        match (status, e) {
            (Status::Success, Some(e)) => {
                entry.perm = perm_flags(e.perm);
                entry.translated_addr = e.to_addr(addr);
                entry.addr_mask = e.addr_mask;
            }
            (Status::Disable | Status::Bypass, _) => {
                entry.perm = flag;
                entry.addr_mask = 0xfff;
            }
            // An abort records no event, and an error has recorded one.
            _ => {}
        }
        entry
    }
}

/// The access flags for SMMU permission bits.
fn perm_flags(p: u8) -> IommuAccessFlags {
    match p & PERM_RW {
        PERM_RO => IommuAccessFlags::RO,
        PERM_WO => IommuAccessFlags::WO,
        PERM_RW => IommuAccessFlags::RW,
        _ => IommuAccessFlags::NONE,
    }
}

/// The IOMMU region of one device, `SMMUDevice`.
struct SmmuDevice {
    smmu: Arc<SmmuV3>,
    sid: u32,
}

impl IommuOps for SmmuDevice {
    fn translate(&self, addr: u64, flag: IommuAccessFlags, _iommu_idx: u32) -> IommuTlbEntry {
        self.smmu.translate(self.sid, addr, flag)
    }
}

/// The register frame, `smmu_mem_ops`.
struct SmmuMmio(Arc<SmmuV3>);

impl fmt::Debug for SmmuMmio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SmmuMmio")
    }
}

impl MmioOps for SmmuMmio {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        if engaged() {
            return Err(MemTxResult::ACCESS_ERROR);
        }
        // Page 0 and page 1 are exact aliases, a CONSTRAINED UNPREDICTABLE choice.
        let offset = offset & !0x10000;
        let s = self.0.lock();
        match size.bytes() {
            8 => Ok(s.readll(offset)),
            4 => Ok(u64::from(s.readl(offset))),
            _ => Err(MemTxResult::ERROR),
        }
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        if engaged() {
            return Err(MemTxResult::ACCESS_ERROR);
        }
        let offset = offset & !0x10000;
        match size.bytes() {
            8 => self.0.with_state(|s, _| s.writell(offset, value)),
            4 => self.0.with_state(|s, m| s.writel(m, offset, value as u32)),
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
