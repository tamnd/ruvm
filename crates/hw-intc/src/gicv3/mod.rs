// SPDX-License-Identifier: GPL-2.0-or-later

//! The emulated GICv3, from hw/intc/arm_gicv3_common.c, hw/intc/arm_gicv3.c,
//! hw/intc/arm_gicv3_dist.c, hw/intc/arm_gicv3_redist.c, hw/intc/arm_gicv3_cpuif.c,
//! hw/intc/gicv3_internal.h and include/hw/intc/arm_gicv3_common.h.
//!
//! The model has the distributor, one redistributor per CPU spread over one or more MMIO regions,
//! and the physical CPU interface (the ICC_* system registers). Affinity routing is always on, as
//! in QEMU, so the legacy GICv2 style registers (GICD_ITARGETSR, GICD_SGIR, GICD_CPENDSGIR and
//! GICD_SPENDSGIR) read as zero and ignore writes.
//!
//! # Wiring
//!
//! [`GicV3::gpio_in`] numbers the inputs like QEMU's GPIO array: SPIs first (input `n` is INTID
//! `n + 32`), then 32 inputs per CPU for its PPIs. [`GicV3::spi`] and [`GicV3::ppi`] are the same
//! lines by a friendlier name. Each CPU has four output pins, IRQ, FIQ, vIRQ and vFIQ. The two
//! virtual pins exist so the board can wire them, but nothing drives them yet.
//!
//! The output pins are always driven after the state lock is dropped, so a pin handler may call
//! back into the GIC. Each CPU's wanted levels are published with a sequence number before the
//! lock is released, and the driver repeats until the sequence is stable, so two threads racing
//! to drive the same CPU always leave the pins at the latest levels.
//!
//! # The CPU interface
//!
//! The ICC_* registers are reached through [`GicV3::icc_access`], [`GicV3::icc_read`] and
//! [`GicV3::icc_write`]. The caller (target-arm) checks the static PLx access rights from the
//! QEMU register table first and then calls `icc_access`, which implements the QEMU access
//! functions (`gicv3_irqfiq_access`, `gicv3_fiq_access`, `gicv3_irq_access`, `gicv3_dir_access`
//! and `gicv3_sgi_access`). [`IccCpuCtx::hcr_el2`] must be the effective HCR_EL2, as
//! `arm_hcr_el2_eff()` returns it.
//!
//! The GIC needs the CPU's exception level and security state to route a pending interrupt to
//! IRQ or FIQ, and it routes from places where no CPU is asking (an MMIO write from another CPU,
//! an input line). So it keeps the last [`IccCpuCtx`] it saw for each CPU, taken from every
//! `icc_*` call and from [`GicV3::cpu_state_changed`], which is the stand-in for QEMU's EL change
//! hook. If an `icc_read` or `icc_write` brings a context that differs from the stored one, the
//! CPU's outputs are recomputed first, as the hook would have done. Before any of that, the
//! stored context is EL1 Non-secure if the GIC has no security extensions and EL3 Secure if it
//! has them, matching where a CPU of that kind comes out of reset.
//!
//! The CPU interface state belongs to the CPU's reset domain, as in QEMU, so [`GicV3::reset`]
//! leaves it alone and [`GicV3::cpuif_reset`] (QEMU's `icc_reset`) resets it.
//!
//! # Differences from QEMU
//!
//! - Where QEMU logs `LOG_GUEST_ERROR` (reserved offsets, writes to read only registers, bad
//!   access sizes, EOI with nothing active) the model stays silent. The access behaves the same,
//!   reading as zero and ignoring writes.
//! - [`GicV3::reset`] and [`GicV3::cpuif_reset`] drive the output pins to the recomputed levels.
//!   QEMU leaves the lines alone and relies on the CPU reset clearing its pending interrupts.
//! - Revision 4 is refused with QEMU's "unsupported GIC revision" message, since GICv4 is not
//!   modelled.
//! - `mp_affinity` comes from the props instead of each CPU's `mp-affinity` property, and an
//!   unset `pribits` (zero) means 5, the `gic_pribits` default.
//! - `gicv3_full_update_noirqset` is not asked to scan an empty SPI range when `num_irq` is 32,
//!   which would trip an assertion in QEMU.
//! - ICC_CTLR_EL3 reads report the EL1S CBPR and EOImode bits from the Non-secure bank, which is
//!   what QEMU 11.1 does. This is kept on purpose so guests see the same values.
//! - [`GicV3::icc_access`] answers `Undefined` for EL0 and for the ICC_AP*R<n> registers the
//!   priority bits do not provide. QEMU never registers those, so it never gets asked.
//! - The AArch32 views (ICC_SGI1R and friends through MCRR) are left to target-arm, and the model
//!   assumes EL3 is AArch64 wherever QEMU asks `arm_el_is_aa64(env, 3)`.
//!
//! # Not done yet (M6 seams)
//!
//! - LPIs and the ITS. GICD_TYPER.LPIS and GICR_TYPER.PLPIS are 0, GICR_CTLR is read only and
//!   GICR_PROPBASER and GICR_PENDBASER are plain storage.
//! - The virtual CPU interface (ICH_* and ICV_*). HCR_EL2.IMO and FMO do not redirect EL1 accesses
//!   to ICV registers, the ICH_HCR_EL2 traps are absent and the vIRQ and vFIQ pins stay low.
//! - NMI (FEAT_GICv3_NMI): GICD_INMIR and GICR_INMIR0 read as zero, ICC_NMIAR1_EL1 is absent and
//!   no NMI pin exists.
//! - GICv4 and its VLPI redistributor frames.
//! - VMState and KVM.

mod cpuif;
mod dist;
mod redist;

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use ruvm_hw_core::irq::{IrqLine, IrqPin};
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MemResult, MmioOps};

/// The size of the distributor MMIO region.
pub const GICV3_DIST_SIZE: u64 = 0x10000;
/// The size of one CPU's redistributor (the RD_base and SGI_base frames).
pub const GICV3_REDIST_SIZE: u64 = 0x20000;
/// `GICV3_MAXIRQ`: the highest number of interrupt lines, SGIs and PPIs included.
pub const GICV3_MAXIRQ: u32 = 1020;
/// `GIC_INTERNAL`: SGIs and PPIs.
pub const GIC_INTERNAL: u32 = 32;
/// `GIC_NR_SGIS`.
pub const GIC_NR_SGIS: u32 = 16;

pub const INTID_SECURE: u64 = 1020;
pub const INTID_NONSECURE: u64 = 1021;
pub const INTID_SPURIOUS: u64 = 1023;

// Interrupt groups, indexing the per group CPU interface arrays.
const G0: usize = 0;
const G1: usize = 1;
const G1NS: usize = 2;

// Security banks of ICC_CTLR_EL1.
const BANK_S: usize = 0;
const BANK_NS: usize = 1;

const GICD_CTLR_EN_GRP0: u32 = 1 << 0;
const GICD_CTLR_EN_GRP1NS: u32 = 1 << 1;
const GICD_CTLR_EN_GRP1S: u32 = 1 << 2;
const GICD_CTLR_EN_GRP1_ALL: u32 = GICD_CTLR_EN_GRP1NS | GICD_CTLR_EN_GRP1S;
const GICD_CTLR_ARE: u32 = 1 << 4;
const GICD_CTLR_ARE_S: u32 = 1 << 4;
const GICD_CTLR_ARE_NS: u32 = 1 << 5;
const GICD_CTLR_DS: u32 = 1 << 6;
const GICD_CTLR_RWP: u32 = 1 << 31;

const GICR_WAKER_PROCESSOR_SLEEP: u32 = 1 << 1;
const GICR_WAKER_CHILDREN_ASLEEP: u32 = 1 << 2;
const GICR_TYPER_LAST: u64 = 1 << 4;

/// `gicv3_iidr()`: an Arm r0p0 with a zero ProductID, like an r0p0 GIC-500.
const GICV3_IIDR: u32 = 0x43b;
const GICV3_PIDR0_DIST: u32 = 0x92;
const GICV3_PIDR0_REDIST: u32 = 0x93;

/// Words in a bitmap with one bit per interrupt.
const BMP_WORDS: usize = 32;

/// The configuration of a GICv3, the QEMU device properties.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GicV3Props {
    /// `num-cpu`.
    pub num_cpu: usize,
    /// `num-irq`, SGIs and PPIs included.
    pub num_irq: u32,
    /// `revision`. Only 3 is accepted.
    pub revision: u32,
    /// `has-security-extensions`.
    pub security_extn: bool,
    /// `redist-region-count`: how many redistributors each MMIO region holds. They must add up to
    /// `num_cpu`.
    pub redist_region_count: Vec<u32>,
    /// Each CPU's MPIDR affinity value (`mp-affinity`), one per CPU.
    pub mp_affinity: Vec<u64>,
    /// The CPU's `gic_pribits`. Zero means 5, the default.
    pub pribits: u8,
}

impl Default for GicV3Props {
    /// QEMU's property defaults.
    fn default() -> Self {
        GicV3Props {
            num_cpu: 1,
            num_irq: 32,
            revision: 3,
            security_extn: false,
            redist_region_count: Vec::new(),
            mp_affinity: Vec::new(),
            pribits: 0,
        }
    }
}

/// The ICC_* system registers of the physical CPU interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IccReg {
    Pmr,
    Iar0,
    Eoir0,
    Hppir0,
    Bpr0,
    /// ICC_AP0R<n>_EL1, n in 0..4.
    Ap0r(u8),
    /// ICC_AP1R<n>_EL1, n in 0..4.
    Ap1r(u8),
    Dir,
    Rpr,
    Sgi1r,
    Asgi1r,
    Sgi0r,
    Iar1,
    Eoir1,
    Hppir1,
    Bpr1,
    CtlrEl1,
    SreEl1,
    Igrpen0,
    Igrpen1,
    SreEl2,
    CtlrEl3,
    SreEl3,
    Igrpen1El3,
}

impl IccReg {
    /// Every register, in the order of QEMU's `gicv3_cpuif_reginfo` with the extra APR
    /// registers after their first one.
    pub const ALL: [IccReg; 30] = [
        IccReg::Pmr,
        IccReg::Iar0,
        IccReg::Eoir0,
        IccReg::Hppir0,
        IccReg::Bpr0,
        IccReg::Ap0r(0),
        IccReg::Ap0r(1),
        IccReg::Ap0r(2),
        IccReg::Ap0r(3),
        IccReg::Ap1r(0),
        IccReg::Ap1r(1),
        IccReg::Ap1r(2),
        IccReg::Ap1r(3),
        IccReg::Dir,
        IccReg::Rpr,
        IccReg::Sgi1r,
        IccReg::Asgi1r,
        IccReg::Sgi0r,
        IccReg::Iar1,
        IccReg::Eoir1,
        IccReg::Hppir1,
        IccReg::Bpr1,
        IccReg::CtlrEl1,
        IccReg::SreEl1,
        IccReg::Igrpen0,
        IccReg::Igrpen1,
        IccReg::SreEl2,
        IccReg::CtlrEl3,
        IccReg::SreEl3,
        IccReg::Igrpen1El3,
    ];

    /// The register with the AArch64 encoding `op0, op1, CRn, CRm, op2`.
    pub fn from_encoding(op0: u32, op1: u32, crn: u32, crm: u32, op2: u32) -> Option<IccReg> {
        let reg = match (op0, op1, crn, crm, op2) {
            (3, 0, 4, 6, 0) => IccReg::Pmr,
            (3, 0, 12, 8, 0) => IccReg::Iar0,
            (3, 0, 12, 8, 1) => IccReg::Eoir0,
            (3, 0, 12, 8, 2) => IccReg::Hppir0,
            (3, 0, 12, 8, 3) => IccReg::Bpr0,
            (3, 0, 12, 8, n @ 4..=7) => IccReg::Ap0r((n - 4) as u8),
            (3, 0, 12, 9, n @ 0..=3) => IccReg::Ap1r(n as u8),
            (3, 0, 12, 11, 1) => IccReg::Dir,
            (3, 0, 12, 11, 3) => IccReg::Rpr,
            (3, 0, 12, 11, 5) => IccReg::Sgi1r,
            (3, 0, 12, 11, 6) => IccReg::Asgi1r,
            (3, 0, 12, 11, 7) => IccReg::Sgi0r,
            (3, 0, 12, 12, 0) => IccReg::Iar1,
            (3, 0, 12, 12, 1) => IccReg::Eoir1,
            (3, 0, 12, 12, 2) => IccReg::Hppir1,
            (3, 0, 12, 12, 3) => IccReg::Bpr1,
            (3, 0, 12, 12, 4) => IccReg::CtlrEl1,
            (3, 0, 12, 12, 5) => IccReg::SreEl1,
            (3, 0, 12, 12, 6) => IccReg::Igrpen0,
            (3, 0, 12, 12, 7) => IccReg::Igrpen1,
            (3, 4, 12, 9, 5) => IccReg::SreEl2,
            (3, 6, 12, 12, 4) => IccReg::CtlrEl3,
            (3, 6, 12, 12, 5) => IccReg::SreEl3,
            (3, 6, 12, 12, 7) => IccReg::Igrpen1El3,
            _ => return None,
        };
        Some(reg)
    }

    /// The AArch64 encoding `(op0, op1, CRn, CRm, op2)`.
    pub fn encoding(self) -> (u32, u32, u32, u32, u32) {
        match self {
            IccReg::Pmr => (3, 0, 4, 6, 0),
            IccReg::Iar0 => (3, 0, 12, 8, 0),
            IccReg::Eoir0 => (3, 0, 12, 8, 1),
            IccReg::Hppir0 => (3, 0, 12, 8, 2),
            IccReg::Bpr0 => (3, 0, 12, 8, 3),
            IccReg::Ap0r(n) => (3, 0, 12, 8, 4 + u32::from(n & 3)),
            IccReg::Ap1r(n) => (3, 0, 12, 9, u32::from(n & 3)),
            IccReg::Dir => (3, 0, 12, 11, 1),
            IccReg::Rpr => (3, 0, 12, 11, 3),
            IccReg::Sgi1r => (3, 0, 12, 11, 5),
            IccReg::Asgi1r => (3, 0, 12, 11, 6),
            IccReg::Sgi0r => (3, 0, 12, 11, 7),
            IccReg::Iar1 => (3, 0, 12, 12, 0),
            IccReg::Eoir1 => (3, 0, 12, 12, 1),
            IccReg::Hppir1 => (3, 0, 12, 12, 2),
            IccReg::Bpr1 => (3, 0, 12, 12, 3),
            IccReg::CtlrEl1 => (3, 0, 12, 12, 4),
            IccReg::SreEl1 => (3, 0, 12, 12, 5),
            IccReg::Igrpen0 => (3, 0, 12, 12, 6),
            IccReg::Igrpen1 => (3, 0, 12, 12, 7),
            IccReg::SreEl2 => (3, 4, 12, 9, 5),
            IccReg::CtlrEl3 => (3, 6, 12, 12, 4),
            IccReg::SreEl3 => (3, 6, 12, 12, 5),
            IccReg::Igrpen1El3 => (3, 6, 12, 12, 7),
        }
    }

    /// Whether the register exists for a CPU with `pribits` bits of priority (zero means 5).
    /// ICC_AP*R1 need 6 or more preemption bits and ICC_AP*R2/3 need 7, as in
    /// `gicv3_init_cpuif`. Registers of a missing EL are the caller's business.
    pub fn exists(self, pribits: u8) -> bool {
        let prebits = prebits_for(pribits);
        match self {
            IccReg::Ap0r(n) | IccReg::Ap1r(n) => match n {
                0 => true,
                1 => prebits >= 6,
                2 | 3 => prebits == 7,
                _ => false,
            },
            _ => true,
        }
    }

    /// The register's name as QEMU spells it.
    pub fn name(self) -> &'static str {
        match self {
            IccReg::Pmr => "ICC_PMR_EL1",
            IccReg::Iar0 => "ICC_IAR0_EL1",
            IccReg::Eoir0 => "ICC_EOIR0_EL1",
            IccReg::Hppir0 => "ICC_HPPIR0_EL1",
            IccReg::Bpr0 => "ICC_BPR0_EL1",
            IccReg::Ap0r(0) => "ICC_AP0R0_EL1",
            IccReg::Ap0r(1) => "ICC_AP0R1_EL1",
            IccReg::Ap0r(2) => "ICC_AP0R2_EL1",
            IccReg::Ap0r(_) => "ICC_AP0R3_EL1",
            IccReg::Ap1r(0) => "ICC_AP1R0_EL1",
            IccReg::Ap1r(1) => "ICC_AP1R1_EL1",
            IccReg::Ap1r(2) => "ICC_AP1R2_EL1",
            IccReg::Ap1r(_) => "ICC_AP1R3_EL1",
            IccReg::Dir => "ICC_DIR_EL1",
            IccReg::Rpr => "ICC_RPR_EL1",
            IccReg::Sgi1r => "ICC_SGI1R_EL1",
            IccReg::Asgi1r => "ICC_ASGI1R_EL1",
            IccReg::Sgi0r => "ICC_SGI0R_EL1",
            IccReg::Iar1 => "ICC_IAR1_EL1",
            IccReg::Eoir1 => "ICC_EOIR1_EL1",
            IccReg::Hppir1 => "ICC_HPPIR1_EL1",
            IccReg::Bpr1 => "ICC_BPR1_EL1",
            IccReg::CtlrEl1 => "ICC_CTLR_EL1",
            IccReg::SreEl1 => "ICC_SRE_EL1",
            IccReg::Igrpen0 => "ICC_IGRPEN0_EL1",
            IccReg::Igrpen1 => "ICC_IGRPEN1_EL1",
            IccReg::SreEl2 => "ICC_SRE_EL2",
            IccReg::CtlrEl3 => "ICC_CTLR_EL3",
            IccReg::SreEl3 => "ICC_SRE_EL3",
            IccReg::Igrpen1El3 => "ICC_IGRPEN1_EL3",
        }
    }
}

/// What the CPU interface needs to know about the CPU making an access, or about where a CPU is
/// now. This is the part of `CPUARMState` that the QEMU code reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IccCpuCtx {
    /// `arm_current_el()`.
    pub el: u32,
    /// `arm_feature(env, ARM_FEATURE_EL2)`.
    pub has_el2: bool,
    /// `arm_feature(env, ARM_FEATURE_EL3)`.
    pub has_el3: bool,
    /// `arm_is_secure()`.
    pub secure: bool,
    /// `arm_is_secure_below_el3()`.
    pub secure_below_el3: bool,
    /// `arm_hcr_el2_eff()`.
    pub hcr_el2: u64,
    /// `env->cp15.scr_el3`.
    pub scr_el3: u64,
}

/// The result of an ICC access check, `CPAccessResult`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IccAccess {
    Ok,
    TrapEl1,
    TrapEl2,
    TrapEl3,
    Undefined,
}

/// The `gic_pribits` default when the CPU does not give one.
fn effective_pribits(pribits: u8) -> u8 {
    if pribits == 0 { 5 } else { pribits }
}

/// The preemption bits: the priority bits, except that 8 bits of priority means 7 preemption
/// bits.
fn prebits_for(pribits: u8) -> u8 {
    effective_pribits(pribits).min(7)
}

/// The highest priority pending interrupt cached for a CPU.
#[derive(Clone, Copy, Debug)]
struct Pending {
    irq: u32,
    prio: u8,
    grp: usize,
}

/// One CPU's redistributor and CPU interface, `GICv3CPUState`.
#[derive(Debug)]
struct CpuState {
    // Redistributor.
    level: u32,
    gicr_typer: u64,
    gicr_ctlr: u32,
    gicr_waker: u32,
    gicr_propbaser: u64,
    gicr_pendbaser: u64,
    gicr_igroupr0: u32,
    gicr_ienabler0: u32,
    gicr_ipendr0: u32,
    gicr_iactiver0: u32,
    edge_trigger: u32,
    gicr_igrpmodr0: u32,
    gicr_nsacr: u32,
    gicr_ipriorityr: [u8; GIC_INTERNAL as usize],

    // CPU interface.
    icc_pmr_el1: u64,
    icc_bpr: [u64; 3],
    icc_apr: [[u64; 4]; 3],
    icc_igrpen: [u64; 3],
    icc_ctlr_el1: [u64; 2],
    icc_ctlr_el3: u64,
    pribits: u8,
    prebits: u8,

    hppi: Pending,
    seenbetter: bool,

    /// The last CPU context seen for this CPU.
    ctx: IccCpuCtx,
    /// The IRQ (bit 0) and FIQ (bit 1) levels the last `gicv3_cpuif_update` wanted.
    out: u64,
}

/// The whole GIC, `GICv3State`.
#[derive(Debug)]
struct GicState {
    num_irq: u32,
    security_extn: bool,
    irq_reset_nonsecure: bool,
    revision: u32,

    gicd_ctlr: u32,
    group: [u32; BMP_WORDS],
    grpmod: [u32; BMP_WORDS],
    enabled: [u32; BMP_WORDS],
    pending: [u32; BMP_WORDS],
    active: [u32; BMP_WORDS],
    level: [u32; BMP_WORDS],
    edge_trigger: [u32; BMP_WORDS],
    gicd_ipriority: Vec<u8>,
    gicd_irouter: Vec<u64>,
    /// `gicd_irouter_target`: the CPU each SPI routes to, if any.
    irouter_target: Vec<Option<usize>>,
    gicd_nsacr: Vec<u32>,

    cpu: Vec<CpuState>,
    /// CPUs whose `out` was recomputed since the last time the pins were driven.
    dirty: Vec<bool>,
}

fn bmp_word(bmp: &[u32; BMP_WORDS], irq: u32) -> u32 {
    bmp[(irq / 32) as usize]
}

fn bmp_word_mut(bmp: &mut [u32; BMP_WORDS], irq: u32) -> &mut u32 {
    &mut bmp[(irq / 32) as usize]
}

fn bmp_test(bmp: &[u32; BMP_WORDS], irq: u32) -> bool {
    (bmp_word(bmp, irq) >> (irq % 32)) & 1 != 0
}

fn bmp_replace(bmp: &mut [u32; BMP_WORDS], irq: u32, v: bool) {
    let w = bmp_word_mut(bmp, irq);
    let bit = 1u32 << (irq % 32);
    if v {
        *w |= bit;
    } else {
        *w &= !bit;
    }
}

/// `half_shuffle32()`: spread the low 16 bits out into the even bits.
fn half_shuffle32(x: u32) -> u32 {
    let mut x = x & 0xffff;
    x = ((x & 0xff00) << 8) | (x & 0x00ff);
    x = ((x << 4) | x) & 0x0f0f_0f0f;
    x = ((x << 2) | x) & 0x3333_3333;
    ((x << 1) | x) & 0x5555_5555
}

/// `half_unshuffle32()`: gather the even bits into the low 16 bits.
fn half_unshuffle32(x: u32) -> u32 {
    let mut x = x & 0x5555_5555;
    x = ((x >> 1) | x) & 0x3333_3333;
    x = ((x >> 2) | x) & 0x0f0f_0f0f;
    x = ((x >> 4) | x) & 0x00ff_00ff;
    ((x >> 8) | x) & 0x0000_ffff
}

/// `half_unshuffle64()`: gather the even bits into the low 32 bits.
fn half_unshuffle64(x: u64) -> u32 {
    let mut x = x & 0x5555_5555_5555_5555;
    x = ((x >> 1) | x) & 0x3333_3333_3333_3333;
    x = ((x >> 2) | x) & 0x0f0f_0f0f_0f0f_0f0f;
    x = ((x >> 4) | x) & 0x00ff_00ff_00ff_00ff;
    x = ((x >> 8) | x) & 0x0000_ffff_0000_ffff;
    (((x >> 16) | x) & 0x0000_0000_ffff_ffff) as u32
}

/// `deposit64(v, start, 32, field)` for the two halves of a 64-bit register.
fn deposit_half(v: u64, high: bool, field: u64) -> u64 {
    let field = field & 0xffff_ffff;
    if high { (v & 0xffff_ffff) | (field << 32) } else { (v & !0xffff_ffff) | field }
}

impl CpuState {
    fn new(gicr_typer: u64, pribits: u8) -> CpuState {
        let mut cs = CpuState {
            level: 0,
            gicr_typer,
            gicr_ctlr: 0,
            gicr_waker: 0,
            gicr_propbaser: 0,
            gicr_pendbaser: 0,
            gicr_igroupr0: 0,
            gicr_ienabler0: 0,
            gicr_ipendr0: 0,
            gicr_iactiver0: 0,
            edge_trigger: 0,
            gicr_igrpmodr0: 0,
            gicr_nsacr: 0,
            gicr_ipriorityr: [0; GIC_INTERNAL as usize],
            icc_pmr_el1: 0,
            icc_bpr: [0; 3],
            icc_apr: [[0; 4]; 3],
            icc_igrpen: [0; 3],
            icc_ctlr_el1: [0; 2],
            icc_ctlr_el3: 0,
            pribits,
            prebits: prebits_for(pribits),
            hppi: Pending { irq: 0, prio: 0xff, grp: G0 },
            seenbetter: false,
            ctx: IccCpuCtx::default(),
            out: 0,
        };
        cs.icc_reset();
        cs
    }
}

impl GicState {
    /// `gicv3_irq_group()`.
    fn irq_group(&self, cpu: usize, irq: u32) -> usize {
        let (grpbit, grpmodbit) = if irq < GIC_INTERNAL {
            let cs = &self.cpu[cpu];
            ((cs.gicr_igroupr0 >> irq) & 1 != 0, (cs.gicr_igrpmodr0 >> irq) & 1 != 0)
        } else {
            (bmp_test(&self.group, irq), bmp_test(&self.grpmod, irq))
        };
        if grpbit {
            return G1NS;
        }
        if self.gicd_ctlr & GICD_CTLR_DS != 0 {
            return G0;
        }
        if grpmodbit { G1 } else { G0 }
    }

    fn ds(&self) -> bool {
        self.gicd_ctlr & GICD_CTLR_DS != 0
    }

    /// `gicv3_cache_target_cpustate()`.
    fn cache_target_cpustate(&mut self, irq: u32) {
        let r = self.gicd_irouter[irq as usize];
        let tgtaff = (r & 0xff_ffff) | (((r >> 32) & 0xff) << 24);
        self.irouter_target[irq as usize] =
            self.cpu.iter().position(|cs| cs.gicr_typer >> 32 == tgtaff);
    }

    /// `gicv3_cache_all_target_cpustates()`.
    fn cache_all_target_cpustates(&mut self) {
        for irq in GIC_INTERNAL..GICV3_MAXIRQ {
            self.cache_target_cpustate(irq);
        }
    }

    /// `irqbetter()`.
    fn irqbetter(&self, cpu: usize, irq: u32, prio: u8) -> bool {
        let hppi = &self.cpu[cpu].hppi;
        if prio != hppi.prio {
            return prio < hppi.prio;
        }
        irq <= hppi.irq
    }

    /// `gicd_int_pending()`: the eligible interrupts in the 32 starting at `irq`.
    fn gicd_int_pending(&self, irq: u32) -> u32 {
        let pending = bmp_word(&self.pending, irq);
        let edge_trigger = bmp_word(&self.edge_trigger, irq);
        let level = bmp_word(&self.level, irq);
        let group = bmp_word(&self.group, irq);
        let mut grpmod = bmp_word(&self.grpmod, irq);
        let enable = bmp_word(&self.enabled, irq);
        let active = bmp_word(&self.active, irq);

        let mut pend = pending | (!edge_trigger & level);
        pend &= enable;
        pend &= !active;

        if self.ds() {
            grpmod = 0;
        }
        let mut grpmask = 0;
        if self.gicd_ctlr & GICD_CTLR_EN_GRP1NS != 0 {
            grpmask |= group;
        }
        if self.gicd_ctlr & GICD_CTLR_EN_GRP1S != 0 {
            grpmask |= !group & grpmod;
        }
        if self.gicd_ctlr & GICD_CTLR_EN_GRP0 != 0 {
            grpmask |= !group & !grpmod;
        }
        pend & grpmask
    }

    /// `gicr_int_pending()`.
    fn gicr_int_pending(&self, cpu: usize) -> u32 {
        let cs = &self.cpu[cpu];
        let mut pend = cs.gicr_ipendr0 | (!cs.edge_trigger & cs.level);
        pend &= cs.gicr_ienabler0;
        pend &= !cs.gicr_iactiver0;

        let grpmod = if self.ds() { 0 } else { cs.gicr_igrpmodr0 };
        let mut grpmask = 0;
        if self.gicd_ctlr & GICD_CTLR_EN_GRP1NS != 0 {
            grpmask |= cs.gicr_igroupr0;
        }
        if self.gicd_ctlr & GICD_CTLR_EN_GRP1S != 0 {
            grpmask |= !cs.gicr_igroupr0 & grpmod;
        }
        if self.gicd_ctlr & GICD_CTLR_EN_GRP0 != 0 {
            grpmask |= !cs.gicr_igroupr0 & !grpmod;
        }
        pend & grpmask
    }

    /// `gicv3_redist_update_noirqset()`: find the best SGI or PPI for `cpu`, without telling
    /// the CPU interface.
    fn redist_update_noirqset(&mut self, cpu: usize) {
        let mut seenbetter = false;
        let pend = self.gicr_int_pending(cpu);
        if pend != 0 {
            for i in 0..GIC_INTERNAL {
                if pend & (1 << i) == 0 {
                    continue;
                }
                let prio = self.cpu[cpu].gicr_ipriorityr[i as usize];
                if self.irqbetter(cpu, i, prio) {
                    let hppi = &mut self.cpu[cpu].hppi;
                    hppi.irq = i;
                    hppi.prio = prio;
                    seenbetter = true;
                }
            }
        }

        if seenbetter {
            let irq = self.cpu[cpu].hppi.irq;
            self.cpu[cpu].hppi.grp = self.irq_group(cpu, irq);
        }

        // If nothing beat the previous best and the previous best was one of ours, it may have
        // dropped in priority and anything could be the best now.
        let hppi = self.cpu[cpu].hppi;
        if !seenbetter && hppi.prio != 0xff && hppi.irq < GIC_INTERNAL {
            self.full_update_noirqset();
        }
    }

    /// `gicv3_redist_update()`.
    fn redist_update(&mut self, cpu: usize) {
        self.redist_update_noirqset(cpu);
        self.cpuif_update(cpu);
    }

    /// `gicv3_update_noirqset()`: rescan the SPIs `start..start + len`.
    fn update_noirqset(&mut self, start: u32, len: u32) {
        assert!(start >= GIC_INTERNAL);
        assert!(len > 0);

        for cs in &mut self.cpu {
            cs.seenbetter = false;
        }

        let mut pend = 0;
        for i in start..start + len {
            if i == start || i & 0x1f == 0 {
                pend = self.gicd_int_pending(i & !0x1f);
            }
            if pend & (1 << (i & 0x1f)) == 0 {
                continue;
            }
            // Interrupts routed to no implemented CPU stay pending and go nowhere.
            let Some(cpu) = self.irouter_target[i as usize] else {
                continue;
            };
            let prio = self.gicd_ipriority[i as usize];
            if self.irqbetter(cpu, i, prio) {
                let cs = &mut self.cpu[cpu];
                cs.hppi.irq = i;
                cs.hppi.prio = prio;
                cs.seenbetter = true;
            }
        }

        for cpu in 0..self.cpu.len() {
            if self.cpu[cpu].seenbetter {
                let irq = self.cpu[cpu].hppi.irq;
                self.cpu[cpu].hppi.grp = self.irq_group(cpu, irq);
            }
            let cs = &self.cpu[cpu];
            if !cs.seenbetter
                && cs.hppi.prio != 0xff
                && cs.hppi.irq >= start
                && cs.hppi.irq < start + len
            {
                self.full_update_noirqset();
                break;
            }
        }
    }

    /// `gicv3_update()`.
    fn update(&mut self, start: u32, len: u32) {
        self.update_noirqset(start, len);
        for cpu in 0..self.cpu.len() {
            self.cpuif_update(cpu);
        }
    }

    /// `gicv3_full_update_noirqset()`.
    fn full_update_noirqset(&mut self) {
        for cs in &mut self.cpu {
            cs.hppi.prio = 0xff;
        }
        if self.num_irq > GIC_INTERNAL {
            self.update_noirqset(GIC_INTERNAL, self.num_irq - GIC_INTERNAL);
        }
        for cpu in 0..self.cpu.len() {
            self.redist_update_noirqset(cpu);
        }
    }

    /// `gicv3_full_update()`.
    fn full_update(&mut self) {
        self.full_update_noirqset();
        for cpu in 0..self.cpu.len() {
            self.cpuif_update(cpu);
        }
    }

    /// `gicv3_dist_set_irq()`.
    fn dist_set_irq(&mut self, irq: u32, level: bool) {
        if level == bmp_test(&self.level, irq) {
            return;
        }
        bmp_replace(&mut self.level, irq, level);
        // A rising edge latches the pending bit of an edge triggered interrupt.
        if level && bmp_test(&self.edge_trigger, irq) {
            bmp_replace(&mut self.pending, irq, true);
        }
        self.update(irq, 1);
    }

    /// `gicv3_redist_set_irq()`.
    fn redist_set_irq(&mut self, cpu: usize, irq: u32, level: bool) {
        let cs = &mut self.cpu[cpu];
        if level == ((cs.level >> irq) & 1 != 0) {
            return;
        }
        if level {
            cs.level |= 1 << irq;
            if (cs.edge_trigger >> irq) & 1 != 0 {
                cs.gicr_ipendr0 |= 1 << irq;
            }
        } else {
            cs.level &= !(1 << irq);
        }
        self.redist_update(cpu);
    }

    /// `gicv3_set_irq()`: input `irq` of the GPIO array.
    fn set_irq(&mut self, irq: u32, level: bool) {
        let nspi = self.num_irq - GIC_INTERNAL;
        if irq < nspi {
            self.dist_set_irq(irq + GIC_INTERNAL, level);
        } else {
            let irq = irq - nspi;
            let cpu = (irq / GIC_INTERNAL) as usize;
            let ppi = irq % GIC_INTERNAL;
            assert!(cpu < self.cpu.len());
            // Raising an SGI through a line would be a board wiring bug.
            assert!(ppi >= GIC_NR_SGIS);
            self.redist_set_irq(cpu, ppi, level);
        }
    }

    /// `gicv3_redist_send_sgi()`.
    fn redist_send_sgi(&mut self, cpu: usize, grp: usize, irq: u32, ns: bool) {
        let irqgrp = self.irq_group(cpu, irq);
        let mut grp = grp;
        // A Secure Group 1 SGI to an interrupt configured as Secure Group 0 is fine, subject to
        // the NSACR checks.
        if grp == G1 && irqgrp == G0 {
            grp = G0;
        }
        if grp != irqgrp {
            return;
        }
        if ns && !self.ds() {
            let nsaccess = (self.cpu[cpu].gicr_nsacr >> (irq * 2)) & 3;
            if (irqgrp == G0 && nsaccess < 1) || (irqgrp == G1 && nsaccess < 2) {
                return;
            }
        }
        self.cpu[cpu].gicr_ipendr0 |= 1 << irq;
        self.redist_update(cpu);
    }

    /// `arm_gicv3_common_reset_hold()`. The CPU interfaces are not touched.
    fn reset(&mut self) {
        let irq_reset_nonsecure = self.irq_reset_nonsecure;
        for cs in &mut self.cpu {
            cs.level = 0;
            cs.gicr_ctlr = 0;
            cs.gicr_waker = GICR_WAKER_PROCESSOR_SLEEP | GICR_WAKER_CHILDREN_ASLEEP;
            cs.gicr_propbaser = 0;
            cs.gicr_pendbaser = 0;
            // A TZ aware GIC reset as if Secure firmware had readied it for a Non-secure kernel
            // puts every interrupt in Group 1.
            cs.gicr_igroupr0 = if irq_reset_nonsecure { 0xffff_ffff } else { 0 };
            cs.gicr_ienabler0 = 0;
            cs.gicr_ipendr0 = 0;
            cs.gicr_iactiver0 = 0;
            cs.edge_trigger = 0xffff;
            cs.gicr_igrpmodr0 = 0;
            cs.gicr_nsacr = 0;
            cs.gicr_ipriorityr = [0; GIC_INTERNAL as usize];
            cs.hppi.prio = 0xff;
        }

        // Affinity routing is always enabled.
        self.gicd_ctlr = if self.security_extn {
            GICD_CTLR_ARE_S | GICD_CTLR_ARE_NS
        } else {
            GICD_CTLR_DS | GICD_CTLR_ARE
        };

        self.group = [0; BMP_WORDS];
        self.grpmod = [0; BMP_WORDS];
        self.enabled = [0; BMP_WORDS];
        self.pending = [0; BMP_WORDS];
        self.active = [0; BMP_WORDS];
        self.level = [0; BMP_WORDS];
        self.edge_trigger = [0; BMP_WORDS];
        self.gicd_ipriority.fill(0);
        self.gicd_irouter.fill(0);
        self.gicd_nsacr.fill(0);
        self.cache_all_target_cpustates();

        if irq_reset_nonsecure {
            for irq in GIC_INTERNAL..self.num_irq {
                bmp_replace(&mut self.group, irq, true);
            }
        }
    }

    /// `gicv3_idreg()`: the CoreSight ID register at `regoffset` from the first one.
    fn idreg(&self, regoffset: u64, pidr0: u32) -> u32 {
        const GICD_IDS: [u8; 12] =
            [0x44, 0x00, 0x00, 0x00, 0x92, 0xB4, 0x0B, 0x00, 0x0D, 0xF0, 0x05, 0xB1];
        let regoffset = (regoffset / 4) as usize;
        if regoffset == 4 {
            return pidr0;
        }
        let mut id = u32::from(GICD_IDS[regoffset]);
        if regoffset == 6 {
            // PIDR2 bits [7:4] are the architecture revision.
            id |= self.revision << 4;
        }
        id
    }
}

/// The GICv3 device.
pub struct GicV3 {
    num_cpu: usize,
    num_irq: u32,
    pribits: u8,
    /// The first CPU and the CPU count of each redistributor region.
    regions: Vec<(usize, u32)>,
    state: Mutex<GicState>,
    /// Per CPU: the wanted IRQ (bit 0) and FIQ (bit 1) levels, above a sequence number.
    out: Vec<AtomicU64>,
    cpu_irq: Vec<IrqPin>,
    cpu_fiq: Vec<IrqPin>,
    cpu_virq: Vec<IrqPin>,
    cpu_vfiq: Vec<IrqPin>,
}

impl fmt::Debug for GicV3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GicV3")
            .field("num_cpu", &self.num_cpu)
            .field("num_irq", &self.num_irq)
            .field("regions", &self.regions)
            .finish_non_exhaustive()
    }
}

impl GicV3 {
    /// Check the props like `arm_gicv3_common_realize()` and build the device, already reset.
    pub fn new(props: GicV3Props) -> Result<Arc<GicV3>, String> {
        if props.revision != 3 {
            return Err(format!("unsupported GIC revision {}", props.revision));
        }
        if props.num_irq > GICV3_MAXIRQ {
            return Err(format!(
                "requested {} interrupt lines exceeds GIC maximum {}",
                props.num_irq, GICV3_MAXIRQ
            ));
        }
        if props.num_irq < GIC_INTERNAL {
            return Err(format!(
                "requested {} interrupt lines is below GIC minimum {}",
                props.num_irq, GIC_INTERNAL
            ));
        }
        if props.num_cpu == 0 {
            return Err("num-cpu must be at least 1".to_string());
        }
        if props.num_irq % 32 != 0 {
            return Err(format!(
                "{} interrupt lines unsupported: not divisible by 32",
                props.num_irq
            ));
        }
        let capacity: u64 = props.redist_region_count.iter().map(|&c| u64::from(c)).sum();
        if capacity != props.num_cpu as u64 {
            return Err(format!(
                "Capacity of the redist regions({}) does not match the number of vcpus({})",
                capacity, props.num_cpu
            ));
        }
        if props.mp_affinity.len() != props.num_cpu {
            return Err(format!(
                "mp-affinity has {} entries but num-cpu is {}",
                props.mp_affinity.len(),
                props.num_cpu
            ));
        }
        let pribits = effective_pribits(props.pribits);
        if !(4..=8).contains(&pribits) {
            return Err(format!("pribits {pribits} is outside the range 4 to 8"));
        }

        let mut cpu = Vec::with_capacity(props.num_cpu);
        for (i, &mpidr) in props.mp_affinity.iter().enumerate() {
            // Squash the MPIDR affinity bytes into the 32 bits GICR_TYPER has room for.
            let affid = ((mpidr & 0xff_0000_0000) >> 8) | (mpidr & 0xff_ffff);
            let typer = (affid << 32) | (1 << 24) | ((i as u64) << 8);
            cpu.push(CpuState::new(typer, pribits));
        }
        // GICR_TYPER.Last marks the final redistributor of each region.
        let mut regions = Vec::with_capacity(props.redist_region_count.len());
        let mut cpuidx = 0usize;
        for &count in &props.redist_region_count {
            regions.push((cpuidx, count));
            cpuidx += count as usize;
            if count > 0 {
                cpu[cpuidx - 1].gicr_typer |= GICR_TYPER_LAST;
            }
        }

        let initial_ctx = if props.security_extn {
            IccCpuCtx { el: 3, has_el3: true, secure: true, ..IccCpuCtx::default() }
        } else {
            IccCpuCtx { el: 1, ..IccCpuCtx::default() }
        };
        for cs in &mut cpu {
            cs.ctx = initial_ctx;
        }

        let n = GICV3_MAXIRQ as usize;
        let mut state = GicState {
            num_irq: props.num_irq,
            security_extn: props.security_extn,
            irq_reset_nonsecure: false,
            revision: props.revision,
            gicd_ctlr: 0,
            group: [0; BMP_WORDS],
            grpmod: [0; BMP_WORDS],
            enabled: [0; BMP_WORDS],
            pending: [0; BMP_WORDS],
            active: [0; BMP_WORDS],
            level: [0; BMP_WORDS],
            edge_trigger: [0; BMP_WORDS],
            gicd_ipriority: vec![0; n],
            gicd_irouter: vec![0; n],
            irouter_target: vec![None; n],
            gicd_nsacr: vec![0; n.div_ceil(16)],
            cpu,
            dirty: vec![false; props.num_cpu],
        };
        state.reset();

        let pins = || (0..props.num_cpu).map(|_| IrqPin::new()).collect::<Vec<_>>();
        Ok(Arc::new(GicV3 {
            num_cpu: props.num_cpu,
            num_irq: props.num_irq,
            pribits,
            regions,
            state: Mutex::new(state),
            out: (0..props.num_cpu).map(|_| AtomicU64::new(0)).collect(),
            cpu_irq: pins(),
            cpu_fiq: pins(),
            cpu_virq: pins(),
            cpu_vfiq: pins(),
        }))
    }

    fn lock(&self) -> MutexGuard<'_, GicState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Run `f` on the state, then drive the output pins of every CPU it touched, after the lock
    /// is released.
    fn with_state<R>(&self, f: impl FnOnce(&mut GicState) -> R) -> R {
        let mut s = self.lock();
        let r = f(&mut s);
        let mut touched = Vec::new();
        for cpu in 0..self.num_cpu {
            if !s.dirty[cpu] {
                continue;
            }
            s.dirty[cpu] = false;
            // Publish under the lock so the sequence numbers follow the order of the updates.
            let old = self.out[cpu].load(Ordering::Relaxed);
            let next = ((old >> 2).wrapping_add(1) << 2) | s.cpu[cpu].out;
            self.out[cpu].store(next, Ordering::Release);
            touched.push(cpu);
        }
        drop(s);
        for cpu in touched {
            self.drive(cpu);
        }
        r
    }

    /// Set the pins of `cpu` to the latest published levels. If another thread publishes while
    /// we are driving, go round again so the last levels driven are the latest ones.
    fn drive(&self, cpu: usize) {
        loop {
            let v = self.out[cpu].load(Ordering::Acquire);
            self.cpu_fiq[cpu].set_bool(v & 2 != 0);
            self.cpu_irq[cpu].set_bool(v & 1 != 0);
            if self.out[cpu].load(Ordering::Acquire) == v {
                break;
            }
        }
    }

    /// `num-cpu`.
    pub fn num_cpu(&self) -> usize {
        self.num_cpu
    }

    /// `num-irq`.
    pub fn num_irq(&self) -> u32 {
        self.num_irq
    }

    /// The priority bits in use, with the default applied.
    pub fn pribits(&self) -> u8 {
        self.pribits
    }

    /// How many GPIO inputs there are: the SPIs, then 32 per CPU.
    pub fn num_gpio_in(&self) -> u32 {
        self.num_irq - GIC_INTERNAL + GIC_INTERNAL * self.num_cpu as u32
    }

    /// GPIO input `n`: SPIs `0..num_irq - 32`, then 32 per CPU for its PPIs.
    pub fn gpio_in(self: &Arc<Self>, n: u32) -> IrqLine {
        assert!(n < self.num_gpio_in(), "GICv3 input {n} out of range");
        let w = Arc::downgrade(self);
        IrqLine::new(
            Arc::new(move |n, level| {
                if let Some(s) = w.upgrade() {
                    s.with_state(|st| st.set_irq(n, level != 0));
                }
            }),
            n,
        )
    }

    /// SPI `n`, which is INTID `n + 32`.
    pub fn spi(self: &Arc<Self>, n: u32) -> IrqLine {
        assert!(n < self.num_irq - GIC_INTERNAL, "GICv3 SPI {n} out of range");
        self.gpio_in(n)
    }

    /// The PPI with INTID `n` (16 to 31) of `cpu`.
    pub fn ppi(self: &Arc<Self>, cpu: usize, n: u32) -> IrqLine {
        assert!(cpu < self.num_cpu, "GICv3 CPU {cpu} out of range");
        assert!((GIC_NR_SGIS..GIC_INTERNAL).contains(&n), "INTID {n} is not a PPI");
        self.gpio_in(self.num_irq - GIC_INTERNAL + GIC_INTERNAL * cpu as u32 + n)
    }

    /// The IRQ output of `cpu`.
    pub fn cpu_irq(&self, cpu: usize) -> &IrqPin {
        &self.cpu_irq[cpu]
    }

    /// The FIQ output of `cpu`.
    pub fn cpu_fiq(&self, cpu: usize) -> &IrqPin {
        &self.cpu_fiq[cpu]
    }

    /// The virtual IRQ output of `cpu`. Never driven until ICH_* is modelled.
    pub fn cpu_virq(&self, cpu: usize) -> &IrqPin {
        &self.cpu_virq[cpu]
    }

    /// The virtual FIQ output of `cpu`. Never driven until ICH_* is modelled.
    pub fn cpu_vfiq(&self, cpu: usize) -> &IrqPin {
        &self.cpu_vfiq[cpu]
    }

    /// The distributor registers, [`GICV3_DIST_SIZE`] bytes.
    pub fn dist_ops(self: &Arc<Self>) -> Arc<dyn MmioOps> {
        Arc::new(GicV3Dist { gic: self.clone() })
    }

    /// The redistributors of region `region`, [`GicV3::redist_region_size`] bytes.
    pub fn redist_ops(self: &Arc<Self>, region: usize) -> Arc<dyn MmioOps> {
        let (cpuidx, count) = self.regions[region];
        Arc::new(GicV3Redist { gic: self.clone(), cpuidx, count: count as usize })
    }

    /// The size of redistributor region `region`.
    pub fn redist_region_size(&self, region: usize) -> u64 {
        u64::from(self.regions[region].1) * GICV3_REDIST_SIZE
    }

    /// The device reset, `arm_gicv3_common_reset_hold()`.
    pub fn reset(&self) {
        self.with_state(|s| {
            s.reset();
            for cpu in 0..s.cpu.len() {
                s.cpuif_update(cpu);
            }
        });
    }

    /// `arm_gic_common_linux_init()`: when booting a kernel straight into Non-secure state, make
    /// the next reset put every interrupt in Group 1, as Secure firmware would have.
    pub fn arm_linux_init(&self, secure_boot: bool) {
        let mut s = self.lock();
        if s.security_extn && !secure_boot {
            s.irq_reset_nonsecure = true;
        }
    }

    /// The CPU interface reset of `cpu`, `icc_reset()`. Called from the CPU's reset.
    pub fn cpuif_reset(&self, cpu: usize) {
        self.with_state(|s| {
            s.cpu[cpu].icc_reset();
            s.cpuif_update(cpu);
        });
    }

    /// The access check for `reg`, the QEMU `accessfn`. The static PLx rights are the caller's.
    pub fn icc_access(&self, cpu: usize, reg: IccReg, ctx: &IccCpuCtx, isread: bool) -> IccAccess {
        let _ = isread;
        assert!(cpu < self.num_cpu);
        if !reg.exists(self.pribits) {
            return IccAccess::Undefined;
        }
        cpuif::access(reg, ctx)
    }

    /// Read `reg` for `cpu`, which is running in `ctx`.
    pub fn icc_read(&self, cpu: usize, reg: IccReg, ctx: &IccCpuCtx) -> u64 {
        self.with_state(|s| {
            s.note_ctx(cpu, ctx);
            s.icc_read(cpu, reg)
        })
    }

    /// Write `value` to `reg` for `cpu`, which is running in `ctx`.
    pub fn icc_write(&self, cpu: usize, reg: IccReg, ctx: &IccCpuCtx, value: u64) {
        self.with_state(|s| {
            s.note_ctx(cpu, ctx);
            s.icc_write(cpu, reg, value);
        });
    }

    /// `cpu` changed exception level or security state: remember the new context and reroute,
    /// as QEMU's `gicv3_cpuif_el_change_hook` does.
    pub fn cpu_state_changed(&self, cpu: usize, ctx: &IccCpuCtx) {
        self.with_state(|s| {
            s.cpu[cpu].ctx = *ctx;
            s.cpuif_update(cpu);
        });
    }
}

/// The distributor MMIO region.
struct GicV3Dist {
    gic: Arc<GicV3>,
}

impl fmt::Debug for GicV3Dist {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("GicV3Dist")
    }
}

impl MmioOps for GicV3Dist {
    fn read(&self, cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        let secure = cx.attrs.secure();
        Ok(self.gic.with_state(|s| s.dist_read(secure, offset, size.bytes())))
    }

    fn write(&self, cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        let secure = cx.attrs.secure();
        self.gic.with_state(|s| s.dist_write(secure, offset, size.bytes(), value));
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 8)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 8)
    }
}

/// One redistributor MMIO region, `GICv3RedistRegion`.
struct GicV3Redist {
    gic: Arc<GicV3>,
    /// The first CPU of the region.
    cpuidx: usize,
    /// How many redistributors the region holds.
    count: usize,
}

impl fmt::Debug for GicV3Redist {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GicV3Redist")
            .field("cpuidx", &self.cpuidx)
            .field("count", &self.count)
            .finish()
    }
}

impl GicV3Redist {
    /// The CPU and the offset within its redistributor. Past the last redistributor of the
    /// region there is nothing.
    fn locate(&self, offset: u64) -> Option<(usize, u64)> {
        let idx = (offset / GICV3_REDIST_SIZE) as usize;
        if idx >= self.count {
            return None;
        }
        Some((self.cpuidx + idx, offset % GICV3_REDIST_SIZE))
    }
}

impl MmioOps for GicV3Redist {
    fn read(&self, cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        let Some((cpu, offset)) = self.locate(offset) else {
            return Ok(0);
        };
        let secure = cx.attrs.secure();
        Ok(self.gic.with_state(|s| s.redist_read(secure, cpu, offset, size.bytes())))
    }

    fn write(&self, cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        let Some((cpu, offset)) = self.locate(offset) else {
            return Ok(());
        };
        let secure = cx.attrs.secure();
        self.gic.with_state(|s| s.redist_write(secure, cpu, offset, size.bytes(), value));
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
mod tests {
    use super::*;

    #[test]
    fn shuffles_round_trip() {
        assert_eq!(half_shuffle32(0xffff), 0x5555_5555);
        assert_eq!(half_shuffle32(0x8001), 0x4000_0001);
        assert_eq!(half_unshuffle32(0x5555_5555), 0xffff);
        assert_eq!(half_unshuffle32(0xaaaa_aaaa), 0);
        assert_eq!(half_unshuffle64(0x5555_5555_5555_5555), 0xffff_ffff);
        assert_eq!(half_unshuffle64(0x4000_0000_0000_0001), 0x8000_0001);
        for v in [0u32, 1, 0x1234, 0xffff, 0x8000] {
            assert_eq!(half_unshuffle32(half_shuffle32(v)), v);
        }
    }

    #[test]
    fn encodings_round_trip() {
        for reg in IccReg::ALL {
            let (op0, op1, crn, crm, op2) = reg.encoding();
            assert_eq!(IccReg::from_encoding(op0, op1, crn, crm, op2), Some(reg), "{reg:?}");
        }
        assert_eq!(IccReg::from_encoding(3, 0, 12, 9, 5), None);
    }
}
