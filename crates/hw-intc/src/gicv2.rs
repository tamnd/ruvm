// SPDX-License-Identifier: GPL-2.0-or-later

//! The emulated GICv2, from hw/intc/arm_gic.c, hw/intc/arm_gic_common.c,
//! hw/intc/gic_internal.h and include/hw/intc/arm_gic_common.h, with the GICv2m MSI frame of
//! hw/intc/arm_gicv2m.c in [`crate::gicv2m::GicV2m`].
//!
//! The model is a revision 2 GIC without the security extensions and without the
//! virtualization extensions, which is what the virt board creates with `gic-version=2` when
//! `secure` and `virtualization` are off, and what KVM needs for `kernel-irqchip=off`. Every
//! access then behaves as a Secure access to a GIC that has the security extensions, so the
//! group 0 interrupts are signalled as FIQ when GICC_CTLR.FIQEn is set and as IRQ otherwise.
//!
//! # Wiring
//!
//! [`GicV2::gpio_in`] numbers the inputs like QEMU's GPIO array: SPIs first (input `n` is
//! interrupt `n + 32`), then 32 inputs per CPU for its PPIs. [`GicV2::spi`] and [`GicV2::ppi`]
//! are the same lines by a friendlier name. Each CPU has an IRQ and a FIQ output pin. They are
//! driven after the state lock is dropped, the same way as the GICv3 drives its pins.
//!
//! # The current CPU
//!
//! The CPU interface at [`GicV2::cpu_ops`] and a few distributor registers (the banked SGI and
//! PPI bits, GICD_SGIR) depend on which CPU makes the access, which QEMU reads from
//! `current_cpu`. Here the board says how to find it with [`GicV2::set_current_cpu_fn`]. With
//! one CPU, or before the board has said, it is CPU 0, as under qtest.
//!
//! # Differences from QEMU
//!
//! - Where QEMU logs `LOG_GUEST_ERROR` (reserved offsets, bad access sizes) the model stays
//!   silent. The access behaves the same.
//! - Acknowledging an SGI that GICD_SPENDSGIR made pending with no source bits (a write of 0)
//!   trips an assertion in QEMU. Here it reports source CPU 0.
//! - [`GicV2::reset`] drives the output pins to the recomputed levels. QEMU leaves the lines
//!   alone and relies on the CPU reset clearing its pending interrupts.
//! - Revision 1, the 11MPCore, the security extensions and the virtualization extensions are
//!   refused with a "not supported by ruvm yet" message.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, RwLock};

use ruvm_hw_core::irq::{IrqLine, IrqPin};
use ruvm_mem::{AccessCtx, AccessSize, MemResult, MmioOps};

/// The size of the distributor MMIO region.
pub const GICV2_DIST_SIZE: u64 = 0x1000;
/// The size of the CPU interface MMIO region of a revision 2 GIC.
pub const GICV2_CPU_SIZE: u64 = 0x2000;
/// `GIC_NCPU`: the most CPUs a GICv2 serves.
pub const GIC_NCPU: usize = 8;
/// `GIC_MAXIRQ`: the most interrupt lines, SGIs and PPIs included.
pub const GIC_MAXIRQ: u32 = 1020;
/// `GIC_INTERNAL`: SGIs and PPIs.
pub const GIC_INTERNAL: u32 = 32;
/// `GIC_NR_SGIS`.
pub const GIC_NR_SGIS: u32 = 16;
/// `GIC_MIN_PRIORITY_BITS`.
pub const GIC_MIN_PRIORITY_BITS: u8 = 4;
/// `GIC_MAX_PRIORITY_BITS`.
pub const GIC_MAX_PRIORITY_BITS: u8 = 8;

/// `ALL_CPU_MASK`.
const ALL_CPU_MASK: u8 = 0xff;
/// `GIC_NR_APRS`: 128 group priorities over 32 bit registers.
const GIC_NR_APRS: usize = 4;
const GIC_MIN_BPR: u8 = 0;
const GIC_MIN_ABPR: u8 = GIC_MIN_BPR + 1;
/// The idle running priority.
const IDLE_PRIORITY: u16 = 0x100;
const SPURIOUS: u16 = 1023;

const GICD_CTLR_EN_GRP0: u32 = 1 << 0;
const GICD_CTLR_EN_GRP1: u32 = 1 << 1;

const GICC_CTLR_ACK_CTL: u32 = 1 << 2;
const GICC_CTLR_FIQ_EN: u32 = 1 << 3;
const GICC_CTLR_CBPR: u32 = 1 << 4;
const GICC_CTLR_EOIMODE: u32 = 1 << 9;
/// `GICC_CTLR_V2_MASK`: the bits a revision 2 GIC without the security extensions keeps.
const GICC_CTLR_V2_MASK: u32 = 0x21f;

/// `gic_id_gicv2`: the CoreSight ID registers at 0xfd0.
const GIC_ID_GICV2: [u8; 12] =
    [0x04, 0x00, 0x00, 0x00, 0x90, 0xb4, 0x2b, 0x00, 0x0d, 0xf0, 0x05, 0xb1];

// The bits of a CPU's wanted output levels.
const OUT_IRQ: u64 = 1 << 0;
const OUT_FIQ: u64 = 1 << 1;
/// Where the sequence number starts in [`GicV2::out`].
const OUT_SEQ_SHIFT: u32 = 8;

/// Finds the index of the CPU making an access, QEMU's `current_cpu->cpu_index`. `None` means
/// the access does not come from a CPU and is taken as CPU 0.
pub type CurrentCpuFn = Arc<dyn Fn() -> Option<usize> + Send + Sync>;

/// The configuration of a GICv2, the QEMU device properties.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GicV2Props {
    /// `num-cpu`.
    pub num_cpu: usize,
    /// `num-irq`, SGIs and PPIs included.
    pub num_irq: u32,
    /// `revision`. Only 2 is accepted.
    pub revision: u32,
    /// `has-security-extensions`. Not modelled, so it must be off.
    pub security_extn: bool,
    /// `has-virtualization-extensions`. Not modelled, so it must be off.
    pub virt_extn: bool,
    /// `num-priority-bits`.
    pub n_prio_bits: u8,
}

impl Default for GicV2Props {
    /// QEMU's property defaults, with revision 2.
    fn default() -> Self {
        GicV2Props {
            num_cpu: 1,
            num_irq: 32,
            revision: 2,
            security_extn: false,
            virt_extn: false,
            n_prio_bits: GIC_MAX_PRIORITY_BITS,
        }
    }
}

/// `gic_irq_state`. The masks have one bit per CPU.
#[derive(Clone, Copy, Debug, Default)]
struct IrqState {
    enabled: u8,
    pending: u8,
    active: u8,
    level: u8,
    /// `model`: the 1-N model, which only the 11MPCore can set.
    model: bool,
    edge_trigger: bool,
    group: u8,
}

/// The whole GIC, `GICState`.
#[derive(Debug)]
struct GicState {
    num_cpu: usize,
    num_irq: u32,
    n_prio_bits: u8,

    ctlr: u32,
    cpu_ctlr: [u32; GIC_NCPU],
    irq_state: Vec<IrqState>,
    irq_target: Vec<u8>,
    priority1: [[u8; GIC_NCPU]; GIC_INTERNAL as usize],
    priority2: Vec<u8>,
    /// For each SGI and target CPU, the CPUs that sent it.
    sgi_pending: [[u8; GIC_NCPU]; GIC_NR_SGIS as usize],
    priority_mask: [u8; GIC_NCPU],
    running_priority: [u16; GIC_NCPU],
    current_pending: [u16; GIC_NCPU],
    bpr: [u8; GIC_NCPU],
    abpr: [u8; GIC_NCPU],
    apr: [[u32; GIC_NCPU]; GIC_NR_APRS],
    nsapr: [[u32; GIC_NCPU]; GIC_NR_APRS],

    /// The output levels last worked out by `gic_update`, the `OUT_*` bits.
    out: [u64; GIC_NCPU],
    /// CPUs whose `out` was recomputed since the pins were last driven.
    dirty: [bool; GIC_NCPU],
}

impl GicState {
    fn new(props: &GicV2Props) -> GicState {
        let n = GIC_MAXIRQ as usize;
        GicState {
            num_cpu: props.num_cpu,
            num_irq: props.num_irq,
            n_prio_bits: props.n_prio_bits,
            ctlr: 0,
            cpu_ctlr: [0; GIC_NCPU],
            irq_state: vec![IrqState::default(); n],
            irq_target: vec![0; n],
            priority1: [[0; GIC_NCPU]; GIC_INTERNAL as usize],
            priority2: vec![0; n - GIC_INTERNAL as usize],
            sgi_pending: [[0; GIC_NCPU]; GIC_NR_SGIS as usize],
            priority_mask: [0; GIC_NCPU],
            running_priority: [IDLE_PRIORITY; GIC_NCPU],
            current_pending: [SPURIOUS; GIC_NCPU],
            bpr: [GIC_MIN_BPR; GIC_NCPU],
            abpr: [GIC_MIN_ABPR; GIC_NCPU],
            apr: [[0; GIC_NCPU]; GIC_NR_APRS],
            nsapr: [[0; GIC_NCPU]; GIC_NR_APRS],
            out: [0; GIC_NCPU],
            dirty: [false; GIC_NCPU],
        }
    }

    fn irq(&self, irq: u32) -> &IrqState {
        &self.irq_state[irq as usize]
    }

    fn irq_mut(&mut self, irq: u32) -> &mut IrqState {
        &mut self.irq_state[irq as usize]
    }

    /// `GIC_DIST_GET_PRIORITY()`, which is also `gic_get_priority()` without a vCPU.
    fn priority(&self, irq: u32, cpu: usize) -> u8 {
        if irq < GIC_INTERNAL {
            self.priority1[irq as usize][cpu]
        } else {
            self.priority2[(irq - GIC_INTERNAL) as usize]
        }
    }

    /// `gic_test_group()`.
    fn test_group(&self, irq: u32, cpu: usize) -> bool {
        self.irq(irq).group & (1 << cpu) != 0
    }

    /// `gic_test_pending()` for a revision 2 GIC: an edge latched, or a level interrupt whose
    /// line is up.
    fn test_pending(&self, irq: u32, cm: u8) -> bool {
        let s = self.irq(irq);
        s.pending & cm != 0 || (!s.edge_trigger && s.level & cm != 0)
    }

    /// `gic_get_best_irq()`: the pending interrupt with the lowest priority value, or
    /// `(1023, 0x100)` when there is none.
    fn best_irq(&self, cpu: usize) -> (u32, u16, bool) {
        let cm = 1u8 << cpu;
        let mut best_irq = SPURIOUS as u32;
        let mut best_prio = IDLE_PRIORITY;
        for irq in 0..self.num_irq {
            let s = self.irq(irq);
            if s.enabled & cm != 0
                && self.test_pending(irq, cm)
                && s.active & cm == 0
                && (irq < GIC_INTERNAL || self.irq_target[irq as usize] & cm != 0)
            {
                let prio = u16::from(self.priority(irq, cpu));
                if prio < best_prio {
                    best_prio = prio;
                    best_irq = irq;
                }
            }
        }
        let group = best_irq < SPURIOUS as u32 && self.test_group(best_irq, cpu);
        (best_irq, best_prio, group)
    }

    /// `gic_irq_signaling_enabled()` for the physical interface.
    fn signaling_enabled(&self, cpu: usize, group_mask: u32) -> bool {
        self.ctlr & group_mask != 0 && self.cpu_ctlr[cpu] & group_mask != 0
    }

    /// `gic_update()`: work out each CPU's highest pending interrupt and its IRQ and FIQ
    /// levels.
    fn update(&mut self) {
        for cpu in 0..self.num_cpu {
            self.current_pending[cpu] = SPURIOUS;
            let mut out = 0;
            if self.signaling_enabled(cpu, GICD_CTLR_EN_GRP0 | GICD_CTLR_EN_GRP1) {
                let (best_irq, best_prio, group) = self.best_irq(cpu);
                if best_prio < u16::from(self.priority_mask[cpu]) {
                    self.current_pending[cpu] = best_irq as u16;
                    if best_prio < self.running_priority[cpu]
                        && self.signaling_enabled(cpu, 1 << u32::from(group))
                    {
                        if !group && self.cpu_ctlr[cpu] & GICC_CTLR_FIQ_EN != 0 {
                            out |= OUT_FIQ;
                        } else {
                            out |= OUT_IRQ;
                        }
                    }
                }
            }
            self.out[cpu] = out;
            self.dirty[cpu] = true;
        }
    }

    /// `gic_set_irq()`: input `n` of the GPIO array.
    fn set_irq(&mut self, n: u32, level: bool) {
        let nspi = self.num_irq - GIC_INTERNAL;
        let (irq, cm, target) = if n < nspi {
            let irq = n + GIC_INTERNAL;
            (irq, ALL_CPU_MASK, self.irq_target[irq as usize])
        } else {
            let n = n - nspi;
            let cpu = n / GIC_INTERNAL;
            let cm = 1u8 << cpu;
            (n % GIC_INTERNAL, cm, cm)
        };
        // Raising an SGI through a line would be a board wiring bug.
        assert!(irq >= GIC_NR_SGIS);

        let s = self.irq_mut(irq);
        if level == (s.level & cm != 0) {
            return;
        }
        if level {
            s.level |= cm;
            if s.edge_trigger {
                s.pending |= target;
            }
        } else {
            s.level &= !cm;
        }
        self.update();
    }

    /// `gic_get_current_pending_irq()` for a Secure access: a group 1 interrupt is only seen
    /// with GICC_CTLR.AckCtl set.
    fn current_pending_irq(&self, cpu: usize) -> u16 {
        let pending = self.current_pending[cpu];
        if u32::from(pending) < GIC_MAXIRQ
            && self.test_group(u32::from(pending), cpu)
            && self.cpu_ctlr[cpu] & GICC_CTLR_ACK_CTL == 0
        {
            return 1022;
        }
        pending
    }

    /// `gic_get_group_priority()`.
    fn group_priority(&self, cpu: usize, irq: u32) -> u32 {
        let bpr = if self.cpu_ctlr[cpu] & GICC_CTLR_CBPR == 0 && self.test_group(irq, cpu) {
            self.abpr[cpu] - 1
        } else {
            self.bpr[cpu]
        };
        let mask = !0u32 << ((bpr & 7) + 1);
        u32::from(self.priority(irq, cpu)) & mask
    }

    /// `gic_activate_irq()`.
    fn activate(&mut self, cpu: usize, irq: u32) {
        let prio = self.group_priority(cpu, irq);
        let preemption_level = prio >> (GIC_MIN_BPR + 1);
        let regno = (preemption_level / 32) as usize;
        let bitno = preemption_level % 32;
        if self.test_group(irq, cpu) {
            self.nsapr[regno][cpu] |= 1 << bitno;
        } else {
            self.apr[regno][cpu] |= 1 << bitno;
        }
        self.running_priority[cpu] = prio as u16;
        self.irq_mut(irq).active |= 1 << cpu;
    }

    /// `gic_get_prio_from_apr_bits()`.
    fn prio_from_apr_bits(&self, cpu: usize) -> u16 {
        for i in 0..GIC_NR_APRS {
            let apr = self.apr[i][cpu] | self.nsapr[i][cpu];
            if apr != 0 {
                return ((i as u32 * 32 + apr.trailing_zeros()) << (GIC_MIN_BPR + 1)) as u16;
            }
        }
        IDLE_PRIORITY
    }

    /// `gic_drop_prio()`: clear the lowest set active priority bit of `group`.
    fn drop_prio(&mut self, cpu: usize, group: bool) {
        for i in 0..GIC_NR_APRS {
            let apr = if group { &mut self.nsapr[i][cpu] } else { &mut self.apr[i][cpu] };
            if *apr != 0 {
                *apr &= *apr - 1;
                break;
            }
        }
        self.running_priority[cpu] = self.prio_from_apr_bits(cpu);
    }

    /// `gic_clear_pending()`.
    fn clear_pending(&mut self, irq: u32, cpu: usize) {
        let s = self.irq_mut(irq);
        s.pending &= !(if s.model { ALL_CPU_MASK } else { 1 << cpu });
    }

    /// `gic_clear_active()`.
    fn clear_active(&mut self, irq: u32, cpu: usize) {
        let cm = if irq < GIC_INTERNAL { 1 << cpu } else { ALL_CPU_MASK };
        self.irq_mut(irq).active &= !cm;
    }

    /// `gic_clear_pending_sgi()`: take the lowest source CPU and return the GICC_IAR value.
    fn clear_pending_sgi(&mut self, irq: u32, cpu: usize) -> u32 {
        let srcs = &mut self.sgi_pending[irq as usize][cpu];
        let src = if *srcs == 0 { 0 } else { srcs.trailing_zeros() };
        *srcs &= !(1u8.checked_shl(src).unwrap_or(0));
        if *srcs == 0 {
            self.clear_pending(irq, cpu);
        }
        irq | ((src & 7) << 10)
    }

    /// `gic_acknowledge_irq()`: a GICC_IAR read.
    fn acknowledge(&mut self, cpu: usize) -> u32 {
        let irq = u32::from(self.current_pending_irq(cpu));
        if irq >= GIC_MAXIRQ {
            return irq;
        }
        if u16::from(self.priority(irq, cpu)) >= self.running_priority[cpu] {
            return u32::from(SPURIOUS);
        }
        self.activate(cpu, irq);
        let ret = if irq < GIC_NR_SGIS {
            self.clear_pending_sgi(irq, cpu)
        } else {
            self.clear_pending(irq, cpu);
            irq
        };
        self.update();
        ret
    }

    /// `gic_fullprio_mask()`: clears the priority bits that are not implemented.
    fn fullprio_mask(&self) -> u8 {
        (0xffu32 << (8 - u32::from(self.n_prio_bits))) as u8
    }

    /// `gic_dist_set_priority()`.
    fn set_priority(&mut self, cpu: usize, irq: u32, val: u8) {
        let val = val & self.fullprio_mask();
        if irq < GIC_INTERNAL {
            self.priority1[irq as usize][cpu] = val;
        } else {
            self.priority2[(irq - GIC_INTERNAL) as usize] = val;
        }
    }

    /// `gic_eoi_split()`.
    fn eoi_split(&self, cpu: usize) -> bool {
        self.cpu_ctlr[cpu] & GICC_CTLR_EOIMODE != 0
    }

    /// `gic_deactivate_irq()`: a GICC_DIR write.
    fn deactivate(&mut self, cpu: usize, irq: u32) {
        // A spurious or missing interrupt is ignored, and so is a GICC_DIR write with
        // EOImode clear, which is UNPREDICTABLE.
        if irq >= GIC_MAXIRQ || irq >= self.num_irq || !self.eoi_split(cpu) {
            return;
        }
        self.clear_active(irq, cpu);
    }

    /// `gic_complete_irq()`: a GICC_EOIR write.
    fn complete(&mut self, cpu: usize, irq: u32) {
        if irq >= self.num_irq || self.running_priority[cpu] == IDLE_PRIORITY {
            return;
        }
        let group = self.test_group(irq, cpu);
        self.drop_prio(cpu, group);
        if !self.eoi_split(cpu) {
            self.clear_active(irq, cpu);
        }
        self.update();
    }

    /// `gic_dist_readb()`.
    fn dist_readb(&self, cpu: usize, offset: u32) -> u8 {
        let cm = 1u8 << cpu;
        let num_irq = self.num_irq;
        // Eight interrupts a byte from `base`, or `None` past the last interrupt.
        let irq8 = |base: u32| {
            let irq = (offset - base) * 8;
            (irq < num_irq).then_some(irq)
        };
        let bits8 = |irq: u32, f: &dyn Fn(u32) -> bool| {
            (0..8).filter(|&i| f(irq + i)).fold(0u8, |acc, i| acc | (1 << i))
        };
        match offset {
            0 => self.ctlr as u8,
            // GICD_TYPER.
            4 => ((num_irq / 32) - 1) as u8 | (((self.num_cpu - 1) as u8) << 5),
            // GICD_IIDR, the Arm JEP106 identity.
            8 => 0x3b,
            9 => 0x04,
            0x80..0x100 => match irq8(0x80) {
                Some(irq) => bits8(irq, &|i| self.irq(i).group & cm != 0),
                None => 0,
            },
            0x100..0x200 => {
                let base = if offset < 0x180 { 0x100 } else { 0x180 };
                match irq8(base) {
                    Some(irq) => bits8(irq, &|i| self.irq(i).enabled & cm != 0),
                    None => 0,
                }
            }
            0x200..0x300 => {
                let base = if offset < 0x280 { 0x200 } else { 0x280 };
                match irq8(base) {
                    Some(irq) => {
                        let mask = if irq < GIC_INTERNAL { cm } else { ALL_CPU_MASK };
                        bits8(irq, &|i| self.test_pending(i, mask))
                    }
                    None => 0,
                }
            }
            0x300..0x400 => {
                let base = if offset < 0x380 { 0x300 } else { 0x380 };
                match irq8(base) {
                    Some(irq) => {
                        let mask = if irq < GIC_INTERNAL { cm } else { ALL_CPU_MASK };
                        bits8(irq, &|i| self.irq(i).active & mask != 0)
                    }
                    None => 0,
                }
            }
            0x400..0x800 => {
                let irq = offset - 0x400;
                if irq >= num_irq { 0 } else { self.priority(irq, cpu) & self.fullprio_mask() }
            }
            0x800..0xc00 => {
                let irq = offset - 0x800;
                // For uniprocessor GICs these are RAZ/WI.
                if self.num_cpu == 1 || irq >= num_irq {
                    0
                } else if irq < GIC_INTERNAL {
                    cm
                } else {
                    self.irq_target[irq as usize]
                }
            }
            0xc00..0xf00 => {
                let irq = (offset - 0xc00) * 4;
                if irq >= num_irq {
                    return 0;
                }
                let mut res = 0u8;
                for i in 0..4 {
                    let s = self.irq(irq + i);
                    if s.model {
                        res |= 1 << (i * 2);
                    }
                    if s.edge_trigger {
                        res |= 2 << (i * 2);
                    }
                }
                res
            }
            // GICD_CPENDSGIRn and GICD_SPENDSGIRn.
            0xf10..0xf30 => {
                let irq = if offset < 0xf20 { offset - 0xf10 } else { offset - 0xf20 };
                self.sgi_pending[irq as usize][cpu]
            }
            0xfd0..0x1000 if offset & 3 == 0 => GIC_ID_GICV2[((offset - 0xfd0) >> 2) as usize],
            _ => 0,
        }
    }

    /// `gic_dist_writeb()`, without the final `gic_update()`. Returns whether the write hit a
    /// register, which is when QEMU updates.
    fn dist_writeb(&mut self, cpu: usize, offset: u32, value: u8) -> bool {
        let num_irq = self.num_irq;
        let irq8 = |base: u32| {
            let irq = (offset - base) * 8;
            (irq < num_irq).then_some(irq)
        };
        match offset {
            0 => self.ctlr = u32::from(value) & (GICD_CTLR_EN_GRP0 | GICD_CTLR_EN_GRP1),
            1..4 => {}
            0x80..0x100 => {
                let Some(irq) = irq8(0x80) else { return false };
                for i in 0..8 {
                    // Group bits are banked for private interrupts.
                    let cm = if irq < GIC_INTERNAL { 1 << cpu } else { ALL_CPU_MASK };
                    let s = self.irq_mut(irq + i);
                    if value & (1 << i) != 0 {
                        s.group |= cm;
                    } else {
                        s.group &= !cm;
                    }
                }
            }
            0x100..0x180 => {
                let Some(irq) = irq8(0x100) else { return false };
                let value = if irq < GIC_NR_SGIS { 0xff } else { value };
                for i in 0..8 {
                    if value & (1 << i) != 0 {
                        let cm = if irq < GIC_INTERNAL { 1 << cpu } else { ALL_CPU_MASK };
                        self.irq_mut(irq + i).enabled |= cm;
                    }
                }
            }
            0x180..0x200 => {
                let Some(irq) = irq8(0x180) else { return false };
                let value = if irq < GIC_NR_SGIS { 0 } else { value };
                for i in 0..8 {
                    if value & (1 << i) != 0 {
                        let cm = if irq < GIC_INTERNAL { 1 << cpu } else { ALL_CPU_MASK };
                        self.irq_mut(irq + i).enabled &= !cm;
                    }
                }
            }
            0x200..0x280 => {
                let Some(irq) = irq8(0x200) else { return false };
                let value = if irq < GIC_NR_SGIS { 0 } else { value };
                for i in 0..8 {
                    if value & (1 << i) != 0 {
                        let mask = if irq < GIC_INTERNAL {
                            1 << cpu
                        } else {
                            self.irq_target[(irq + i) as usize]
                        };
                        self.irq_mut(irq + i).pending |= mask;
                    }
                }
            }
            0x280..0x300 => {
                let Some(irq) = irq8(0x280) else { return false };
                let value = if irq < GIC_NR_SGIS { 0 } else { value };
                for i in 0..8 {
                    // This clears the pending bit for all CPUs, even for per-CPU interrupts,
                    // as QEMU does.
                    if value & (1 << i) != 0 {
                        self.irq_mut(irq + i).pending = 0;
                    }
                }
            }
            0x300..0x400 => {
                let set = offset < 0x380;
                let Some(irq) = irq8(if set { 0x300 } else { 0x380 }) else { return false };
                // These registers are banked per CPU for PPIs.
                let cm = if irq < GIC_INTERNAL { 1 << cpu } else { ALL_CPU_MASK };
                for i in 0..8 {
                    if value & (1 << i) != 0 {
                        let s = self.irq_mut(irq + i);
                        if set {
                            s.active |= cm;
                        } else {
                            s.active &= !cm;
                        }
                    }
                }
            }
            0x400..0x800 => {
                let irq = offset - 0x400;
                if irq >= num_irq {
                    return false;
                }
                self.set_priority(cpu, irq, value);
            }
            0x800..0xc00 => {
                // RAZ/WI on uniprocessor GICs.
                if self.num_cpu != 1 {
                    let irq = offset - 0x800;
                    if irq >= num_irq {
                        return false;
                    }
                    let value = if irq < GIC_INTERNAL { ALL_CPU_MASK } else { value };
                    self.irq_target[irq as usize] = value & ALL_CPU_MASK;
                    let s = self.irq_mut(irq);
                    // Changing the target of a pending interrupt moves where it is pending.
                    if irq >= GIC_INTERNAL && s.pending != 0 {
                        s.pending = value & ALL_CPU_MASK;
                    }
                }
            }
            0xc00..0xf00 => {
                let irq = (offset - 0xc00) * 4;
                if irq >= num_irq {
                    return false;
                }
                let value = if irq < GIC_NR_SGIS { value | 0xaa } else { value };
                for i in 0..4 {
                    self.irq_mut(irq + i).edge_trigger = value & (2 << (i * 2)) != 0;
                }
            }
            0xf10..0xf20 => {
                let irq = offset - 0xf10;
                let srcs = &mut self.sgi_pending[irq as usize][cpu];
                *srcs &= !value;
                if *srcs == 0 {
                    self.irq_mut(irq).pending &= !(1 << cpu);
                }
            }
            0xf20..0xf30 => {
                let irq = offset - 0xf20;
                self.irq_mut(irq).pending |= 1 << cpu;
                self.sgi_pending[irq as usize][cpu] |= value;
            }
            // 0xf00 is only handled for 32-bit writes.
            _ => return false,
        }
        true
    }

    /// `gic_dist_writel()` at 0xf00: GICD_SGIR.
    fn write_sgir(&mut self, cpu: usize, value: u32) {
        let irq = value & 0xf;
        let mask = match (value >> 24) & 3 {
            0 => ((value >> 16) as u8) & ALL_CPU_MASK,
            1 => ALL_CPU_MASK ^ (1 << cpu),
            2 => 1 << cpu,
            _ => ALL_CPU_MASK,
        };
        self.irq_mut(irq).pending |= mask;
        for target in 0..GIC_NCPU {
            if mask & (1 << target) != 0 {
                self.sgi_pending[irq as usize][target] |= 1 << cpu;
            }
        }
        self.update();
    }

    /// `gic_dist_read()`: 1, 2 or 4 bytes put together from byte reads.
    fn dist_read(&self, cpu: usize, offset: u64, size: u32) -> u64 {
        let offset = offset as u32;
        (0..size)
            .fold(0u64, |acc, i| acc | (u64::from(self.dist_readb(cpu, offset + i)) << (8 * i)))
    }

    /// `gic_dist_write()`.
    fn dist_write(&mut self, cpu: usize, offset: u64, size: u32, value: u64) {
        let offset = offset as u32;
        if size == 4 && offset == 0xf00 {
            self.write_sgir(cpu, value as u32);
            return;
        }
        for i in 0..size {
            // Each byte write updates the outputs, as gic_dist_writeb() does.
            if self.dist_writeb(cpu, offset + i, (value >> (8 * i)) as u8) {
                self.update();
            }
        }
    }

    /// `gic_cpu_read()`.
    fn cpu_read(&mut self, cpu: usize, offset: u64) -> u32 {
        match offset {
            0x00 => self.cpu_ctlr[cpu],
            0x04 => u32::from(self.priority_mask[cpu]),
            0x08 => u32::from(self.bpr[cpu]),
            0x0c => self.acknowledge(cpu),
            // GICC_RPR reads the idle priority as 0xff.
            0x14 => u32::from(self.running_priority[cpu].min(0xff)),
            0x18 => u32::from(self.current_pending_irq(cpu)),
            0x1c => u32::from(self.abpr[cpu]),
            0xd0 | 0xd4 | 0xd8 | 0xdc => self.apr[((offset - 0xd0) / 4) as usize][cpu],
            0xe0 | 0xe4 | 0xe8 | 0xec => self.nsapr[((offset - 0xe0) / 4) as usize][cpu],
            // GICC_IIDR: an Arm GICv2.
            0xfc => (2 << 16) | 0x43b,
            _ => 0,
        }
    }

    /// `gic_cpu_write()`.
    fn cpu_write(&mut self, cpu: usize, offset: u64, value: u32) {
        match offset {
            0x00 => self.cpu_ctlr[cpu] = value & GICC_CTLR_V2_MASK,
            0x04 => self.priority_mask[cpu] = (value as u8) & self.fullprio_mask(),
            // GIC_MIN_BPR is 0, so QEMU's MAX() with it does nothing.
            0x08 => self.bpr[cpu] = (value & 7) as u8,
            0x10 => {
                // gic_complete_irq() updates when it has something to do.
                self.complete(cpu, value & 0x3ff);
                return;
            }
            0x1c => self.abpr[cpu] = ((value & 7) as u8).max(GIC_MIN_ABPR),
            0xd0 | 0xd4 | 0xd8 | 0xdc => {
                self.apr[((offset - 0xd0) / 4) as usize][cpu] = value;
                self.running_priority[cpu] = self.prio_from_apr_bits(cpu);
            }
            0xe0 | 0xe4 | 0xe8 | 0xec => {
                self.nsapr[((offset - 0xe0) / 4) as usize][cpu] = value;
                self.running_priority[cpu] = self.prio_from_apr_bits(cpu);
            }
            0x1000 => self.deactivate(cpu, value & 0x3ff),
            _ => return,
        }
        self.update();
    }

    /// `arm_gic_common_reset_hold()`. The active priority registers are left alone, as in
    /// QEMU.
    fn reset(&mut self) {
        self.irq_state.fill(IrqState::default());
        for cpu in 0..self.num_cpu {
            self.priority_mask[cpu] = 0;
            self.current_pending[cpu] = SPURIOUS;
            self.running_priority[cpu] = IDLE_PRIORITY;
            self.cpu_ctlr[cpu] = 0;
            self.bpr[cpu] = GIC_MIN_BPR;
            self.abpr[cpu] = GIC_MIN_ABPR;
            for p in &mut self.priority1 {
                p[cpu] = 0;
            }
            for s in &mut self.sgi_pending {
                s[cpu] = 0;
            }
        }
        for irq in 0..GIC_NR_SGIS {
            let s = self.irq_mut(irq);
            s.enabled = ALL_CPU_MASK;
            s.edge_trigger = true;
        }
        self.priority2.fill(0);
        // For uniprocessor GICs all interrupts always target the sole CPU.
        let target = if self.num_cpu == 1 { 1 } else { 0 };
        self.irq_target.fill(target);
        self.ctlr = 0;
    }
}

/// The GICv2 device.
pub struct GicV2 {
    num_cpu: usize,
    num_irq: u32,
    state: Mutex<GicState>,
    /// Per CPU: the wanted levels (the `OUT_*` bits), below a sequence number.
    out: Vec<AtomicU64>,
    cpu_irq: Vec<IrqPin>,
    cpu_fiq: Vec<IrqPin>,
    current_cpu: RwLock<Option<CurrentCpuFn>>,
}

impl fmt::Debug for GicV2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GicV2")
            .field("num_cpu", &self.num_cpu)
            .field("num_irq", &self.num_irq)
            .finish_non_exhaustive()
    }
}

impl GicV2 {
    /// Check the props like `arm_gic_common_realize()` and `arm_gic_realize()` and build the
    /// device, already reset.
    pub fn new(props: GicV2Props) -> Result<Arc<GicV2>, String> {
        if props.num_cpu > GIC_NCPU {
            return Err(format!(
                "requested {} CPUs exceeds GIC maximum {}",
                props.num_cpu, GIC_NCPU
            ));
        }
        if props.num_irq > GIC_MAXIRQ {
            return Err(format!(
                "requested {} interrupt lines exceeds GIC maximum {}",
                props.num_irq, GIC_MAXIRQ
            ));
        }
        // ITLinesNumber is represented as (N / 32) - 1, so this is an implementation
        // imposed restriction, not an architectural one.
        if props.num_irq < 32 || props.num_irq % 32 != 0 {
            return Err(format!(
                "{} interrupt lines unsupported: not divisible by 32",
                props.num_irq
            ));
        }
        if props.virt_extn && props.revision != 2 {
            return Err(
                "GIC virtualization extensions are only supported by revision 2".to_string()
            );
        }
        if props.revision != 2 {
            return Err(format!("GIC revision {} is not supported by ruvm yet", props.revision));
        }
        if props.security_extn {
            return Err("GICv2 security extensions are not supported by ruvm yet".to_string());
        }
        if props.virt_extn {
            return Err("GICv2 virtualization extensions are not supported by ruvm yet".to_string());
        }
        if !(GIC_MIN_PRIORITY_BITS..=GIC_MAX_PRIORITY_BITS).contains(&props.n_prio_bits) {
            return Err(format!(
                "num-priority-bits cannot be greater than {GIC_MAX_PRIORITY_BITS} or less than \
                 {GIC_MIN_PRIORITY_BITS}"
            ));
        }
        if props.num_cpu == 0 {
            return Err("num-cpu must be at least 1".to_string());
        }

        let mut state = GicState::new(&props);
        state.reset();
        let pins = || (0..props.num_cpu).map(|_| IrqPin::new()).collect::<Vec<_>>();
        Ok(Arc::new(GicV2 {
            num_cpu: props.num_cpu,
            num_irq: props.num_irq,
            state: Mutex::new(state),
            out: (0..props.num_cpu).map(|_| AtomicU64::new(0)).collect(),
            cpu_irq: pins(),
            cpu_fiq: pins(),
            current_cpu: RwLock::new(None),
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
        let mut touched = [false; GIC_NCPU];
        for (cpu, touched) in touched.iter_mut().enumerate().take(self.num_cpu) {
            if !s.dirty[cpu] {
                continue;
            }
            s.dirty[cpu] = false;
            // Publish under the lock so the sequence numbers follow the order of the updates.
            let old = self.out[cpu].load(Ordering::Relaxed);
            let next = ((old >> OUT_SEQ_SHIFT).wrapping_add(1) << OUT_SEQ_SHIFT) | s.out[cpu];
            self.out[cpu].store(next, Ordering::Release);
            *touched = true;
        }
        drop(s);
        for cpu in (0..self.num_cpu).filter(|&c| touched[c]) {
            self.drive(cpu);
        }
        r
    }

    /// Set the pins of `cpu` to the latest published levels. If another thread publishes while
    /// we are driving, go round again so the last levels driven are the latest ones.
    fn drive(&self, cpu: usize) {
        loop {
            let v = self.out[cpu].load(Ordering::Acquire);
            self.cpu_irq[cpu].set_bool(v & OUT_IRQ != 0);
            self.cpu_fiq[cpu].set_bool(v & OUT_FIQ != 0);
            if self.out[cpu].load(Ordering::Acquire) == v {
                break;
            }
        }
    }

    /// `gic_get_current_cpu()`.
    fn current_cpu(&self) -> usize {
        if self.num_cpu == 1 {
            return 0;
        }
        let f = self.current_cpu.read().unwrap_or_else(|e| e.into_inner()).clone();
        // An index past the GIC's CPUs would be a board bug; take it as CPU 0 rather than
        // index out of the arrays.
        f.and_then(|f| f()).filter(|&c| c < self.num_cpu).unwrap_or(0)
    }

    /// Tell the GIC how to find the CPU making an access, replacing the previous way.
    pub fn set_current_cpu_fn(&self, f: Option<CurrentCpuFn>) {
        *self.current_cpu.write().unwrap_or_else(|e| e.into_inner()) = f;
    }

    /// `num-cpu`.
    pub fn num_cpu(&self) -> usize {
        self.num_cpu
    }

    /// `num-irq`.
    pub fn num_irq(&self) -> u32 {
        self.num_irq
    }

    /// How many GPIO inputs there are: the SPIs, then 32 per CPU.
    pub fn num_gpio_in(&self) -> u32 {
        self.num_irq - GIC_INTERNAL + GIC_INTERNAL * self.num_cpu as u32
    }

    /// GPIO input `n`: SPIs `0..num_irq - 32`, then 32 per CPU for its PPIs.
    pub fn gpio_in(self: &Arc<Self>, n: u32) -> IrqLine {
        assert!(n < self.num_gpio_in(), "GICv2 input {n} out of range");
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

    /// SPI `n`, which is interrupt `n + 32`.
    pub fn spi(self: &Arc<Self>, n: u32) -> IrqLine {
        assert!(n < self.num_irq - GIC_INTERNAL, "GICv2 SPI {n} out of range");
        self.gpio_in(n)
    }

    /// The PPI with interrupt ID `n` (16 to 31) of `cpu`.
    pub fn ppi(self: &Arc<Self>, cpu: usize, n: u32) -> IrqLine {
        assert!(cpu < self.num_cpu, "GICv2 CPU {cpu} out of range");
        assert!((GIC_NR_SGIS..GIC_INTERNAL).contains(&n), "interrupt {n} is not a PPI");
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

    /// The distributor registers, [`GICV2_DIST_SIZE`] bytes.
    pub fn dist_ops(self: &Arc<Self>) -> Arc<dyn MmioOps> {
        Arc::new(GicV2Dist { gic: self.clone() })
    }

    /// The CPU interface of the CPU making the access, [`GICV2_CPU_SIZE`] bytes.
    pub fn cpu_ops(self: &Arc<Self>) -> Arc<dyn MmioOps> {
        Arc::new(GicV2Cpu { gic: self.clone(), cpu: None })
    }

    /// The CPU interface of `cpu` whoever accesses it, QEMU's per CPU `gic_cpu` regions.
    pub fn cpu_ops_for(self: &Arc<Self>, cpu: usize) -> Arc<dyn MmioOps> {
        assert!(cpu < self.num_cpu, "GICv2 CPU {cpu} out of range");
        Arc::new(GicV2Cpu { gic: self.clone(), cpu: Some(cpu) })
    }

    /// The device reset, `arm_gic_common_reset_hold()`.
    pub fn reset(&self) {
        self.with_state(|s| {
            s.reset();
            s.update();
        });
    }
}

/// The distributor MMIO region.
struct GicV2Dist {
    gic: Arc<GicV2>,
}

impl fmt::Debug for GicV2Dist {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("GicV2Dist")
    }
}

impl MmioOps for GicV2Dist {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        let cpu = self.gic.current_cpu();
        Ok(self.gic.with_state(|s| s.dist_read(cpu, offset, size.bytes())))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        let cpu = self.gic.current_cpu();
        self.gic.with_state(|s| s.dist_write(cpu, offset, size.bytes(), value));
        Ok(())
    }
}

/// A CPU interface MMIO region: the current CPU's, or a fixed one.
struct GicV2Cpu {
    gic: Arc<GicV2>,
    cpu: Option<usize>,
}

impl fmt::Debug for GicV2Cpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GicV2Cpu").field("cpu", &self.cpu).finish()
    }
}

impl MmioOps for GicV2Cpu {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        let cpu = self.cpu.unwrap_or_else(|| self.gic.current_cpu());
        Ok(u64::from(self.gic.with_state(|s| s.cpu_read(cpu, offset))))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        let cpu = self.cpu.unwrap_or_else(|| self.gic.current_cpu());
        self.gic.with_state(|s| s.cpu_write(cpu, offset, value as u32));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(num_cpu: usize, num_irq: u32) -> GicState {
        let props = GicV2Props { num_cpu, num_irq, ..GicV2Props::default() };
        let mut s = GicState::new(&props);
        s.reset();
        s
    }

    #[test]
    fn group_priority_follows_the_binary_points() {
        let mut s = state(1, 64);
        s.set_priority(0, 40, 0xb7);
        // BPR 0 keeps bits [7:1].
        assert_eq!(s.group_priority(0, 40), 0xb6);
        s.bpr[0] = 3;
        assert_eq!(s.group_priority(0, 40), 0xb0);
        // A group 1 interrupt uses ABPR - 1 unless CBPR is set.
        s.irq_mut(40).group = ALL_CPU_MASK;
        s.abpr[0] = 2;
        assert_eq!(s.group_priority(0, 40), 0xb4);
        s.cpu_ctlr[0] = GICC_CTLR_CBPR;
        assert_eq!(s.group_priority(0, 40), 0xb0);
    }

    #[test]
    fn active_priorities_nest_and_drop() {
        let mut s = state(1, 64);
        s.set_priority(0, 40, 0x80);
        s.set_priority(0, 41, 0x40);
        s.activate(0, 40);
        assert_eq!(s.running_priority[0], 0x80);
        assert_eq!(s.apr[2][0], 1);
        s.activate(0, 41);
        assert_eq!(s.running_priority[0], 0x40);
        assert_eq!(s.apr[1][0], 1);
        s.drop_prio(0, false);
        assert_eq!(s.running_priority[0], 0x80);
        s.drop_prio(0, false);
        assert_eq!(s.running_priority[0], IDLE_PRIORITY);
    }

    #[test]
    fn fewer_priority_bits_mask_the_low_ones() {
        let props = GicV2Props { num_cpu: 1, num_irq: 64, n_prio_bits: 5, ..GicV2Props::default() };
        let mut s = GicState::new(&props);
        s.reset();
        assert_eq!(s.fullprio_mask(), 0xf8);
        s.set_priority(0, 33, 0xff);
        assert_eq!(s.priority(33, 0), 0xf8);
    }

    #[test]
    fn sgi_sources_are_acknowledged_lowest_first() {
        let mut s = state(4, 64);
        s.ctlr = 1;
        s.cpu_ctlr[0] = 1;
        s.priority_mask[0] = 0xff;
        // CPUs 2 and 1 send SGI 3 to CPU 0.
        s.write_sgir(2, 3 | (1 << 16));
        s.write_sgir(1, 3 | (1 << 16));
        assert_eq!(s.sgi_pending[3][0], 0b110);
        assert_eq!(s.acknowledge(0), 3 | (1 << 10));
        // Still pending from CPU 2, but active now.
        assert_eq!(s.irq(3).pending & 1, 1);
        s.complete(0, 3);
        assert_eq!(s.acknowledge(0), 3 | (2 << 10));
        assert_eq!(s.irq(3).pending & 1, 0);
    }

    #[test]
    fn empty_spendsgir_write_acknowledges_from_cpu_0() {
        let mut s = state(2, 64);
        s.ctlr = 1;
        s.cpu_ctlr[1] = 1;
        s.priority_mask[1] = 0xff;
        assert!(s.dist_writeb(1, 0xf25, 0));
        s.update();
        assert_eq!(s.acknowledge(1), 5);
        assert_eq!(s.irq(5).pending, 0);
    }
}
