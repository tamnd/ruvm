// SPDX-License-Identifier: GPL-2.0-or-later

//! The physical CPU interface, the ICC_* registers, from hw/intc/arm_gicv3_cpuif.c.

use super::{
    BANK_NS, BANK_S, CpuState, G0, G1, G1NS, GicState, INTID_NONSECURE, INTID_SECURE,
    INTID_SPURIOUS, IccAccess, IccCpuCtx, IccReg,
};

const ICC_CTLR_EL1_CBPR: u64 = 1 << 0;
const ICC_CTLR_EL1_EOIMODE: u64 = 1 << 1;
const ICC_CTLR_EL1_PRIBITS_SHIFT: u32 = 8;
const ICC_CTLR_EL1_IDBITS_SHIFT: u32 = 11;
const ICC_CTLR_EL1_A3V: u64 = 1 << 15;

const ICC_CTLR_EL3_CBPR_EL1S: u64 = 1 << 0;
const ICC_CTLR_EL3_CBPR_EL1NS: u64 = 1 << 1;
const ICC_CTLR_EL3_EOIMODE_EL3: u64 = 1 << 2;
const ICC_CTLR_EL3_EOIMODE_EL1S: u64 = 1 << 3;
const ICC_CTLR_EL3_EOIMODE_EL1NS: u64 = 1 << 4;
const ICC_CTLR_EL3_PRIBITS_SHIFT: u32 = 8;
const ICC_CTLR_EL3_IDBITS_SHIFT: u32 = 11;
const ICC_CTLR_EL3_A3V: u64 = 1 << 15;
const ICC_CTLR_EL3_NDS: u64 = 1 << 17;

/// ICC_SRE_EL1 reads SRE, DFB and DIB set: system registers only.
const ICC_SRE_EL1_VALUE: u64 = 0x7;
/// ICC_SRE_EL2 and ICC_SRE_EL3 add the Enable bit.
const ICC_SRE_EL2_EL3_VALUE: u64 = 0xf;

const ICC_IGRPEN_ENABLE: u64 = 1;

const HCR_FMO: u64 = 1 << 3;
const HCR_IMO: u64 = 1 << 4;
const SCR_IRQ: u64 = 1 << 1;
const SCR_FIQ: u64 = 1 << 2;

/// `gicv3_intid_is_special()`.
fn intid_is_special(intid: u64) -> bool {
    (INTID_SECURE..=INTID_SPURIOUS).contains(&intid)
}

/// The QEMU access function attached to each register.
#[derive(Clone, Copy)]
enum AccessFn {
    None,
    IrqFiq,
    Fiq,
    Irq,
    Dir,
    Sgi,
}

fn accessfn(reg: IccReg) -> AccessFn {
    match reg {
        IccReg::Pmr | IccReg::Rpr | IccReg::CtlrEl1 => AccessFn::IrqFiq,
        IccReg::Iar0 | IccReg::Eoir0 | IccReg::Hppir0 | IccReg::Bpr0 | IccReg::Ap0r(_) => {
            AccessFn::Fiq
        }
        IccReg::Igrpen0 => AccessFn::Fiq,
        IccReg::Ap1r(_) | IccReg::Iar1 | IccReg::Eoir1 | IccReg::Hppir1 | IccReg::Bpr1 => {
            AccessFn::Irq
        }
        IccReg::Igrpen1 => AccessFn::Irq,
        IccReg::Dir => AccessFn::Dir,
        IccReg::Sgi0r | IccReg::Sgi1r | IccReg::Asgi1r => AccessFn::Sgi,
        IccReg::SreEl1 | IccReg::SreEl2 | IccReg::CtlrEl3 | IccReg::SreEl3 | IccReg::Igrpen1El3 => {
            AccessFn::None
        }
    }
}

/// `gicv3_irqfiq_access()`. The ICH_HCR_EL2.TC trap is absent with the virtual interface.
fn irqfiq_access(ctx: &IccCpuCtx) -> IccAccess {
    if ctx.scr_el3 & (SCR_FIQ | SCR_IRQ) == (SCR_FIQ | SCR_IRQ) {
        match ctx.el {
            1 if ctx.hcr_el2 & (HCR_IMO | HCR_FMO) == 0 => return IccAccess::TrapEl3,
            2 => return IccAccess::TrapEl3,
            _ => {}
        }
    }
    IccAccess::Ok
}

/// `gicv3_fiq_access()` and `gicv3_irq_access()`.
fn one_access(ctx: &IccCpuCtx, scr_bit: u64, hcr_bit: u64) -> IccAccess {
    if ctx.scr_el3 & scr_bit != 0 {
        match ctx.el {
            1 if ctx.hcr_el2 & hcr_bit == 0 => return IccAccess::TrapEl3,
            2 => return IccAccess::TrapEl3,
            _ => {}
        }
    }
    IccAccess::Ok
}

/// The access check for `reg` from a CPU in `ctx`.
pub(super) fn access(reg: IccReg, ctx: &IccCpuCtx) -> IccAccess {
    if ctx.el == 0 {
        return IccAccess::Undefined;
    }
    match accessfn(reg) {
        AccessFn::None => IccAccess::Ok,
        AccessFn::IrqFiq | AccessFn::Dir => irqfiq_access(ctx),
        AccessFn::Fiq => one_access(ctx, SCR_FIQ, HCR_FMO),
        AccessFn::Irq => one_access(ctx, SCR_IRQ, HCR_IMO),
        AccessFn::Sgi => {
            // This takes priority over a possible EL3 trap.
            if ctx.el == 1 && ctx.hcr_el2 & (HCR_IMO | HCR_FMO) != 0 {
                return IccAccess::TrapEl2;
            }
            irqfiq_access(ctx)
        }
    }
}

impl CpuState {
    /// `icc_reset()`.
    pub(super) fn icc_reset(&mut self) {
        let pribits = u64::from(self.pribits - 1);
        self.icc_apr = [[0; 4]; 3];
        let ctlr_el1 = ICC_CTLR_EL1_A3V
            | (1 << ICC_CTLR_EL1_IDBITS_SHIFT)
            | (pribits << ICC_CTLR_EL1_PRIBITS_SHIFT);
        self.icc_ctlr_el1 = [ctlr_el1; 2];
        self.icc_pmr_el1 = 0;
        self.icc_bpr[G0] = self.min_bpr();
        self.icc_bpr[G1] = self.min_bpr();
        self.icc_bpr[G1NS] = self.min_bpr_ns();
        self.icc_igrpen = [0; 3];
        self.icc_ctlr_el3 = ICC_CTLR_EL3_NDS
            | ICC_CTLR_EL3_A3V
            | (1 << ICC_CTLR_EL3_IDBITS_SHIFT)
            | (pribits << ICC_CTLR_EL3_PRIBITS_SHIFT);
    }

    /// `icc_fullprio_mask()`: the priority bits that are implemented.
    fn fullprio_mask(&self) -> u64 {
        (!0u64 << (8 - self.pribits)) & 0xff
    }

    /// `icc_min_bpr()`.
    fn min_bpr(&self) -> u64 {
        7 - u64::from(self.prebits)
    }

    /// `icc_min_bpr_ns()`.
    fn min_bpr_ns(&self) -> u64 {
        self.min_bpr() + 1
    }

    /// `icc_num_aprs()`.
    fn num_aprs(&self) -> usize {
        1 << self.prebits.saturating_sub(5)
    }

    /// `icc_highest_active_prio()`: the running priority.
    fn highest_active_prio(&self) -> u64 {
        for i in 0..self.num_aprs() {
            let apr = self.icc_apr[G0][i] | self.icc_apr[G1][i] | self.icc_apr[G1NS][i];
            if apr == 0 {
                continue;
            }
            let bit = i as u64 * 32 + u64::from(apr.trailing_zeros());
            return bit << (self.min_bpr() + 1);
        }
        0xff
    }

    /// `icc_gprio_mask()`: the group priority bits for `group`, after the CBPR redirect.
    fn gprio_mask(&self, group: usize) -> u32 {
        let mut group = group;
        if (group == G1 && self.icc_ctlr_el1[BANK_S] & ICC_CTLR_EL1_CBPR != 0)
            || (group == G1NS && self.icc_ctlr_el1[BANK_NS] & ICC_CTLR_EL1_CBPR != 0)
        {
            group = G0;
        }
        let mut bpr = (self.icc_bpr[group] & 7) as u32;
        if group == G1NS {
            bpr = bpr.saturating_sub(1);
        }
        !0u32 << (bpr + 1)
    }

    /// `icc_no_enabled_hppi()`.
    fn no_enabled_hppi(&self) -> bool {
        self.hppi.prio == 0xff || self.icc_igrpen[self.hppi.grp] == 0
    }

    /// `icc_hppi_can_preempt()`.
    fn hppi_can_preempt(&self) -> bool {
        if self.no_enabled_hppi() {
            return false;
        }
        if u64::from(self.hppi.prio) >= self.icc_pmr_el1 {
            // Masked by the priority mask.
            return false;
        }
        let rprio = self.highest_active_prio();
        if rprio == 0xff {
            // Nothing is active, so anything unmasked can preempt.
            return true;
        }
        let mask = u64::from(self.gprio_mask(self.hppi.grp));
        (u64::from(self.hppi.prio) & mask) < (rprio & mask)
    }

    /// `icc_eoi_split()`: whether EOI only drops the priority and DIR deactivates.
    fn eoi_split(&self) -> bool {
        if self.ctx.el == 3 {
            return self.icc_ctlr_el3 & ICC_CTLR_EL3_EOIMODE_EL3 != 0;
        }
        let bank = if self.ctx.secure { BANK_S } else { BANK_NS };
        self.icc_ctlr_el1[bank] & ICC_CTLR_EL1_EOIMODE != 0
    }

    /// `gicv3_use_ns_bank()`.
    fn use_ns_bank(&self) -> bool {
        !self.ctx.secure_below_el3
    }

    /// Whether Non-secure state sees the restricted view of priorities: EL3 exists, the CPU is
    /// Non-secure and Group 0 is routed to EL3.
    fn ns_prio_view(&self) -> bool {
        self.ctx.has_el3 && !self.ctx.secure && self.ctx.scr_el3 & SCR_FIQ != 0
    }

    /// `icc_highest_active_group()`.
    fn highest_active_group(&self) -> Option<usize> {
        for i in 0..4 {
            let g0ctz = (self.icc_apr[G0][i] as u32).trailing_zeros();
            let g1ctz = (self.icc_apr[G1][i] as u32).trailing_zeros();
            let g1nsctz = (self.icc_apr[G1NS][i] as u32).trailing_zeros();
            if g1nsctz < g0ctz && g1nsctz < g1ctz {
                return Some(G1NS);
            }
            if g1ctz < g0ctz {
                return Some(G1);
            }
            if g0ctz < 32 {
                return Some(G0);
            }
        }
        None
    }

    /// `icc_hppir0_value()`.
    fn hppir0_value(&self, ds: bool) -> u64 {
        if self.no_enabled_hppi() {
            return INTID_SPURIOUS;
        }
        let irq_is_secure = !ds && self.hppi.grp != G1NS;
        if self.hppi.grp != G0 && self.ctx.el != 3 {
            return INTID_SPURIOUS;
        }
        if irq_is_secure && !self.ctx.secure {
            // Secure interrupts are invisible to Non-secure.
            return INTID_SPURIOUS;
        }
        if self.hppi.grp != G0 {
            // EL3 reading the Group 0 view of a Group 1 interrupt.
            return if irq_is_secure { INTID_SECURE } else { INTID_NONSECURE };
        }
        u64::from(self.hppi.irq)
    }

    /// `icc_hppir1_value()`.
    fn hppir1_value(&self, ds: bool) -> u64 {
        if self.no_enabled_hppi() || self.hppi.grp == G0 {
            return INTID_SPURIOUS;
        }
        let irq_is_secure = !ds && self.hppi.grp != G1NS;
        if irq_is_secure {
            if !self.ctx.secure {
                return INTID_SPURIOUS;
            }
        } else if self.ctx.el != 3 && self.ctx.secure {
            // A Non-secure interrupt is not visible to Secure EL1.
            return INTID_SPURIOUS;
        }
        u64::from(self.hppi.irq)
    }

    /// `icc_drop_prio()` without the update: clear the highest priority active bit of `grp`.
    fn drop_prio(&mut self, grp: usize) {
        for i in 0..self.num_aprs() {
            let apr = &mut self.icc_apr[grp][i];
            if *apr == 0 {
                continue;
            }
            *apr &= *apr - 1;
            break;
        }
    }
}

impl GicState {
    /// Remember the context of a CPU making an ICC access and reroute if it moved.
    pub(super) fn note_ctx(&mut self, cpu: usize, ctx: &IccCpuCtx) {
        if self.cpu[cpu].ctx != *ctx {
            self.cpu[cpu].ctx = *ctx;
            self.cpuif_update(cpu);
        }
    }

    /// `gicv3_cpuif_update()`: work out the IRQ and FIQ levels for `cpu`. The pins are driven
    /// once the lock is dropped.
    pub(super) fn cpuif_update(&mut self, cpu: usize) {
        let cs = &mut self.cpu[cpu];
        if cs.hppi.grp == G1 && !cs.ctx.has_el3 {
            // Without EL3 a Secure Group 1 interrupt behaves as Group 0.
            cs.hppi.grp = G0;
        }
        let mut out = 0;
        if cs.hppi_can_preempt() {
            let isfiq = match cs.hppi.grp {
                G0 => true,
                G1 => !cs.ctx.secure || cs.ctx.el == 3,
                _ => cs.ctx.secure,
            };
            out = if isfiq { 2 } else { 1 };
        }
        cs.out = out;
        self.dirty[cpu] = true;
    }

    /// `icc_activate_irq()`.
    fn icc_activate_irq(&mut self, cpu: usize, irq: u32) {
        let cs = &mut self.cpu[cpu];
        let mask = cs.gprio_mask(cs.hppi.grp);
        let prio = u32::from(cs.hppi.prio) & mask;
        let aprbit = prio >> (8 - u32::from(cs.prebits));
        cs.icc_apr[cs.hppi.grp][(aprbit / 32) as usize] |= 1 << (aprbit % 32);

        if irq < super::GIC_INTERNAL {
            cs.gicr_iactiver0 |= 1 << irq;
            cs.gicr_ipendr0 &= !(1 << irq);
            self.redist_update(cpu);
        } else {
            super::bmp_replace(&mut self.active, irq, true);
            super::bmp_replace(&mut self.pending, irq, false);
            self.update(irq, 1);
        }
    }

    /// `icc_deactivate_irq()`.
    fn icc_deactivate_irq(&mut self, cpu: usize, irq: u32) {
        if irq < super::GIC_INTERNAL {
            self.cpu[cpu].gicr_iactiver0 &= !(1 << irq);
            self.redist_update(cpu);
        } else {
            super::bmp_replace(&mut self.active, irq, false);
            self.update(irq, 1);
        }
    }

    /// `icc_iar0_read()` and `icc_iar1_read()`.
    fn icc_iar(&mut self, cpu: usize, group0: bool) -> u64 {
        let ds = self.ds();
        let cs = &self.cpu[cpu];
        let intid = if !cs.hppi_can_preempt() {
            INTID_SPURIOUS
        } else if group0 {
            cs.hppir0_value(ds)
        } else {
            cs.hppir1_value(ds)
        };
        if !intid_is_special(intid) {
            self.icc_activate_irq(cpu, intid as u32);
        }
        intid
    }

    /// `icc_eoir_write()`.
    fn icc_eoir_write(&mut self, cpu: usize, is_eoir0: bool, value: u64) {
        let irq = (value & 0xff_ffff) as u32;
        if irq >= self.num_irq {
            return;
        }
        let ds = self.ds();
        let cs = &self.cpu[cpu];
        let Some(grp) = cs.highest_active_group() else {
            return;
        };
        let ok = match grp {
            G0 => is_eoir0 && !(!ds && cs.ctx.has_el3 && !cs.ctx.secure),
            G1 => !is_eoir0 && cs.ctx.secure,
            _ => !is_eoir0 && !(cs.ctx.el != 3 && cs.ctx.secure),
        };
        if !ok {
            return;
        }
        self.cpu[cpu].drop_prio(grp);
        self.cpuif_update(cpu);
        if !self.cpu[cpu].eoi_split() {
            // Priority drop and deactivation are not split, so deactivate now.
            self.icc_deactivate_irq(cpu, irq);
        }
    }

    /// `icc_dir_write()`.
    fn icc_dir_write(&mut self, cpu: usize, value: u64) {
        let irq = (value & 0xff_ffff) as u32;
        if irq >= self.num_irq || !self.cpu[cpu].eoi_split() {
            return;
        }
        let grp = self.irq_group(cpu, irq);
        let ctx = self.cpu[cpu].ctx;
        let single_sec_state = self.ds();
        let irq_is_secure = !single_sec_state && grp != G1NS;
        let irq_is_grp0 = grp == G0;
        let route_fiq_to_el3 = ctx.scr_el3 & SCR_FIQ != 0;
        let route_irq_to_el3 = ctx.scr_el3 & SCR_IRQ != 0;
        let route_fiq_to_el2 = ctx.hcr_el2 & HCR_FMO != 0;
        let route_irq_to_el2 = ctx.hcr_el2 & HCR_IMO != 0;

        // DIR only deactivates interrupts this exception level could have taken.
        let ok = match ctx.el {
            3 => true,
            2 => {
                (single_sec_state && irq_is_grp0 && !route_fiq_to_el3)
                    || (!irq_is_secure && !irq_is_grp0 && !route_irq_to_el3)
            }
            1 if !ctx.secure_below_el3 => {
                (single_sec_state && irq_is_grp0 && !route_fiq_to_el3 && !route_fiq_to_el2)
                    || (!irq_is_secure && !irq_is_grp0 && !route_irq_to_el3 && !route_irq_to_el2)
            }
            1 => {
                (irq_is_grp0 && !route_fiq_to_el3)
                    || (!irq_is_grp0 && (!irq_is_secure || !single_sec_state) && !route_irq_to_el3)
            }
            _ => false,
        };
        if ok {
            self.icc_deactivate_irq(cpu, irq);
        }
    }

    /// `icc_generate_sgi()`.
    fn icc_generate_sgi(&mut self, cpu: usize, value: u64, grp: usize, ns: bool) {
        let aff =
            (((value >> 48) & 0xff) << 16) | (((value >> 32) & 0xff) << 8) | ((value >> 16) & 0xff);
        let targetlist = (value & 0xffff) as u32;
        let irq = ((value >> 24) & 0xf) as u32;
        let irm = (value >> 40) & 1 != 0;
        let mut grp = grp;

        if grp == G1 && self.ds() {
            // With one security state the Distributor treats Secure Group 1 as Group 0.
            grp = G0;
        }

        for i in 0..self.cpu.len() {
            if irm {
                // Every CPU but this one.
                if i == cpu {
                    continue;
                }
            } else {
                // Aff3.Aff2.Aff1.n for every n set in the target list.
                let typer = self.cpu[i].gicr_typer;
                if typer >> 40 != aff {
                    continue;
                }
                let aff0 = (typer >> 32) & 0xff;
                if aff0 > 15 || (targetlist >> aff0) & 1 == 0 {
                    continue;
                }
            }
            // The redistributor checks its own GICR_NSACR.
            self.redist_send_sgi(i, grp, irq, ns);
        }
    }

    /// `icc_bpr_read()` and `icc_bpr_write()` share this choice of register.
    fn bpr_group(&self, cpu: usize, group0: bool) -> usize {
        let cs = &self.cpu[cpu];
        let mut grp = if group0 { G0 } else { G1 };
        if grp == G1 && cs.use_ns_bank() {
            grp = G1NS;
        }
        if grp == G1 && cs.ctx.el != 3 && cs.icc_ctlr_el1[BANK_S] & ICC_CTLR_EL1_CBPR != 0 {
            // CBPR_EL1S makes Secure EL1 BPR1 accesses reach BPR0.
            grp = G0;
        }
        grp
    }

    fn ap_group(&self, cpu: usize, group0: bool) -> usize {
        if group0 {
            G0
        } else if self.cpu[cpu].use_ns_bank() {
            G1NS
        } else {
            G1
        }
    }

    /// Read `reg`. Write only registers read as zero.
    pub(super) fn icc_read(&mut self, cpu: usize, reg: IccReg) -> u64 {
        let ds = self.ds();
        match reg {
            IccReg::Pmr => {
                let cs = &self.cpu[cpu];
                let mut value = cs.icc_pmr_el1;
                if cs.ns_prio_view() {
                    // The Non-secure view of the priority.
                    if value & 0x80 == 0 {
                        value = 0;
                    } else if value != 0xff {
                        value = (value << 1) & 0xff;
                    }
                }
                value
            }
            IccReg::Rpr => {
                let cs = &self.cpu[cpu];
                let mut prio = cs.highest_active_prio();
                if cs.ns_prio_view() {
                    if prio & 0x80 == 0 {
                        prio = 0;
                    } else if prio != 0xff {
                        prio = (prio << 1) & 0xff;
                    }
                }
                prio
            }
            IccReg::Iar0 => self.icc_iar(cpu, true),
            IccReg::Iar1 => self.icc_iar(cpu, false),
            IccReg::Hppir0 => self.cpu[cpu].hppir0_value(ds),
            IccReg::Hppir1 => self.cpu[cpu].hppir1_value(ds),
            IccReg::Bpr0 | IccReg::Bpr1 => {
                let mut grp = self.bpr_group(cpu, reg == IccReg::Bpr0);
                let cs = &self.cpu[cpu];
                let mut satinc = false;
                if grp == G1NS && cs.ctx.el < 3 && cs.icc_ctlr_el1[BANK_NS] & ICC_CTLR_EL1_CBPR != 0
                {
                    // Reads give BPR0 plus one, saturating at 7.
                    grp = G0;
                    satinc = true;
                }
                let bpr = cs.icc_bpr[grp];
                if satinc { (bpr + 1).min(7) } else { bpr }
            }
            IccReg::Ap0r(n) | IccReg::Ap1r(n) => {
                let grp = self.ap_group(cpu, matches!(reg, IccReg::Ap0r(_)));
                self.cpu[cpu].icc_apr[grp][usize::from(n & 3)]
            }
            IccReg::CtlrEl1 => {
                let cs = &self.cpu[cpu];
                let bank = if cs.use_ns_bank() { BANK_NS } else { BANK_S };
                cs.icc_ctlr_el1[bank]
            }
            IccReg::CtlrEl3 => {
                // QEMU takes the EL1S aliases from the Non-secure bank too. Kept as is.
                let cs = &self.cpu[cpu];
                let ns = cs.icc_ctlr_el1[BANK_NS];
                let mut value = cs.icc_ctlr_el3;
                if ns & ICC_CTLR_EL1_EOIMODE != 0 {
                    value |= ICC_CTLR_EL3_EOIMODE_EL1NS | ICC_CTLR_EL3_EOIMODE_EL1S;
                }
                if ns & ICC_CTLR_EL1_CBPR != 0 {
                    value |= ICC_CTLR_EL3_CBPR_EL1NS | ICC_CTLR_EL3_CBPR_EL1S;
                }
                value
            }
            IccReg::Igrpen0 => self.cpu[cpu].icc_igrpen[G0],
            IccReg::Igrpen1 => {
                let cs = &self.cpu[cpu];
                let grp = if cs.use_ns_bank() { G1NS } else { G1 };
                cs.icc_igrpen[grp]
            }
            IccReg::Igrpen1El3 => {
                let cs = &self.cpu[cpu];
                cs.icc_igrpen[G1NS] | (cs.icc_igrpen[G1] << 1)
            }
            IccReg::SreEl1 => ICC_SRE_EL1_VALUE,
            IccReg::SreEl2 | IccReg::SreEl3 => ICC_SRE_EL2_EL3_VALUE,
            IccReg::Eoir0
            | IccReg::Eoir1
            | IccReg::Dir
            | IccReg::Sgi0r
            | IccReg::Sgi1r
            | IccReg::Asgi1r => 0,
        }
    }

    /// Write `reg`. Read only registers ignore writes.
    pub(super) fn icc_write(&mut self, cpu: usize, reg: IccReg, value: u64) {
        match reg {
            IccReg::Pmr => {
                let cs = &mut self.cpu[cpu];
                let mut value = value & 0xff;
                if cs.ns_prio_view() {
                    // Non-secure may not move a mask that is in the Secure range.
                    if cs.icc_pmr_el1 & 0x80 == 0 {
                        return;
                    }
                    value = (value >> 1) | 0x80;
                }
                cs.icc_pmr_el1 = value & cs.fullprio_mask();
                self.cpuif_update(cpu);
            }
            IccReg::Eoir0 => self.icc_eoir_write(cpu, true, value),
            IccReg::Eoir1 => self.icc_eoir_write(cpu, false, value),
            IccReg::Dir => self.icc_dir_write(cpu, value),
            IccReg::Sgi0r => {
                let ns = !self.cpu[cpu].ctx.secure;
                self.icc_generate_sgi(cpu, value, G0, ns);
            }
            IccReg::Sgi1r => {
                let secure = self.cpu[cpu].ctx.secure;
                let grp = if secure { G1 } else { G1NS };
                self.icc_generate_sgi(cpu, value, grp, !secure);
            }
            IccReg::Asgi1r => {
                let secure = self.cpu[cpu].ctx.secure;
                let grp = if secure { G1NS } else { G1 };
                self.icc_generate_sgi(cpu, value, grp, !secure);
            }
            IccReg::Bpr0 | IccReg::Bpr1 => {
                let grp = self.bpr_group(cpu, reg == IccReg::Bpr0);
                let cs = &mut self.cpu[cpu];
                if grp == G1NS && cs.ctx.el < 3 && cs.icc_ctlr_el1[BANK_NS] & ICC_CTLR_EL1_CBPR != 0
                {
                    // CBPR_EL1NS makes Non-secure BPR1 writes ignored.
                    return;
                }
                let minval = if grp == G1NS { cs.min_bpr_ns() } else { cs.min_bpr() };
                cs.icc_bpr[grp] = value.max(minval) & 7;
                self.cpuif_update(cpu);
            }
            IccReg::Ap0r(n) | IccReg::Ap1r(n) => {
                let grp = self.ap_group(cpu, matches!(reg, IccReg::Ap0r(_)));
                let regno = usize::from(n & 3);
                let cs = &mut self.cpu[cpu];
                let mut value = value & 0xffff_ffff;
                if grp == G1NS && cs.ctx.has_el3 {
                    // Non-secure may not claim an active priority in the Secure range, or it
                    // could block Secure interrupts.
                    let ns_start_bit = 0x80usize >> (8 - usize::from(cs.prebits));
                    let ns_start_regno = ns_start_bit / 32;
                    let ns_start_bitno = ns_start_bit % 32;
                    if regno < ns_start_regno {
                        return;
                    }
                    if regno == ns_start_regno && ns_start_bitno != 0 {
                        let keep = (1u64 << ns_start_bitno) - 1;
                        value = (value & !keep) | (cs.icc_apr[grp][regno] & keep);
                    }
                }
                cs.icc_apr[grp][regno] = value;
                self.cpuif_update(cpu);
            }
            IccReg::CtlrEl1 => {
                let ds = self.ds();
                let cs = &mut self.cpu[cpu];
                let bank = if cs.use_ns_bank() { BANK_NS } else { BANK_S };
                let mask = if cs.ctx.has_el3 && !ds {
                    // CBPR is controlled from ICC_CTLR_EL3.
                    ICC_CTLR_EL1_EOIMODE
                } else {
                    ICC_CTLR_EL1_CBPR | ICC_CTLR_EL1_EOIMODE
                };
                cs.icc_ctlr_el1[bank] = (cs.icc_ctlr_el1[bank] & !mask) | (value & mask);
                self.cpuif_update(cpu);
            }
            IccReg::CtlrEl3 => {
                let cs = &mut self.cpu[cpu];
                let both = ICC_CTLR_EL1_CBPR | ICC_CTLR_EL1_EOIMODE;
                // The EL1NS and EL1S bits are aliases of the two ICC_CTLR_EL1 banks.
                let mut ns = cs.icc_ctlr_el1[BANK_NS] & !both;
                if value & ICC_CTLR_EL3_EOIMODE_EL1NS != 0 {
                    ns |= ICC_CTLR_EL1_EOIMODE;
                }
                if value & ICC_CTLR_EL3_CBPR_EL1NS != 0 {
                    ns |= ICC_CTLR_EL1_CBPR;
                }
                cs.icc_ctlr_el1[BANK_NS] = ns;
                let mut s = cs.icc_ctlr_el1[BANK_S] & !both;
                if value & ICC_CTLR_EL3_EOIMODE_EL1S != 0 {
                    s |= ICC_CTLR_EL1_EOIMODE;
                }
                if value & ICC_CTLR_EL3_CBPR_EL1S != 0 {
                    s |= ICC_CTLR_EL1_CBPR;
                }
                cs.icc_ctlr_el1[BANK_S] = s;
                let mask = ICC_CTLR_EL3_EOIMODE_EL3;
                cs.icc_ctlr_el3 = (cs.icc_ctlr_el3 & !mask) | (value & mask);
                self.cpuif_update(cpu);
            }
            IccReg::Igrpen0 => {
                self.cpu[cpu].icc_igrpen[G0] = value & ICC_IGRPEN_ENABLE;
                self.cpuif_update(cpu);
            }
            IccReg::Igrpen1 => {
                let cs = &mut self.cpu[cpu];
                let grp = if cs.use_ns_bank() { G1NS } else { G1 };
                cs.icc_igrpen[grp] = value & ICC_IGRPEN_ENABLE;
                self.cpuif_update(cpu);
            }
            IccReg::Igrpen1El3 => {
                let cs = &mut self.cpu[cpu];
                cs.icc_igrpen[G1NS] = value & 1;
                cs.icc_igrpen[G1] = (value >> 1) & 1;
                self.cpuif_update(cpu);
            }
            IccReg::Iar0
            | IccReg::Iar1
            | IccReg::Hppir0
            | IccReg::Hppir1
            | IccReg::Rpr
            | IccReg::SreEl1
            | IccReg::SreEl2
            | IccReg::SreEl3 => {}
        }
    }
}
