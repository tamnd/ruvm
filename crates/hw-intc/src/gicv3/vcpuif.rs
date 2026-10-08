// SPDX-License-Identifier: GPL-2.0-or-later

//! The virtual CPU interface, the ICH_* and ICV_* registers, from hw/intc/arm_gicv3_cpuif.c.
//!
//! There are no vLPIs (GICv4) and no NMIs (FEAT_GICv3_NMI), so the highest priority pending
//! virtual interrupt always comes from a list register and ICH_LR<n>_EL2.NMI is RES0.

use super::cpuif::{
    HCR_FMO, HCR_IMO, ICC_CTLR_EL1_A3V, ICC_CTLR_EL1_CBPR, ICC_CTLR_EL1_EOIMODE,
    ICC_CTLR_EL1_IDBITS_SHIFT, ICC_CTLR_EL1_PRIBITS_SHIFT, intid_is_special,
};
use super::{
    CpuState, G0, G1, G1NS, GICV3_LPI_INTID_START, GICV3_MAXIRQ, GicState, INTID_SECURE,
    INTID_SPURIOUS, IccReg, OUT_MAINT, OUT_VFIQ, OUT_VIRQ,
};

/// `gic_num_lrs`, `gic_vpribits` and `gic_vprebits` as QEMU defaults them, which is what
/// every CPU modelled here sets.
pub(super) const ICH_NUM_LRS: usize = 4;
pub(super) const ICH_VPRIBITS: u8 = 5;
pub(super) const ICH_VPREBITS: u8 = 5;

const ICH_VMCR_EL2_VENG0: u64 = 1 << 0;
const ICH_VMCR_EL2_VENG1: u64 = 1 << 1;
const ICH_VMCR_EL2_VFIQEN: u64 = 1 << 3;
const ICH_VMCR_EL2_VCBPR: u64 = 1 << 4;
const ICH_VMCR_EL2_VEOIM: u64 = 1 << 9;
const ICH_VMCR_EL2_VBPR1_SHIFT: u32 = 18;
const ICH_VMCR_EL2_VBPR1_MASK: u64 = 7 << ICH_VMCR_EL2_VBPR1_SHIFT;
const ICH_VMCR_EL2_VBPR0_SHIFT: u32 = 21;
const ICH_VMCR_EL2_VBPR0_MASK: u64 = 7 << ICH_VMCR_EL2_VBPR0_SHIFT;
const ICH_VMCR_EL2_VPMR_SHIFT: u32 = 24;
const ICH_VMCR_EL2_VPMR_MASK: u64 = 0xff << ICH_VMCR_EL2_VPMR_SHIFT;

pub(super) const ICH_HCR_EL2_EN: u64 = 1 << 0;
const ICH_HCR_EL2_UIE: u64 = 1 << 1;
const ICH_HCR_EL2_LRENPIE: u64 = 1 << 2;
const ICH_HCR_EL2_NPIE: u64 = 1 << 3;
const ICH_HCR_EL2_VGRP0EIE: u64 = 1 << 4;
const ICH_HCR_EL2_VGRP0DIE: u64 = 1 << 5;
const ICH_HCR_EL2_VGRP1EIE: u64 = 1 << 6;
const ICH_HCR_EL2_VGRP1DIE: u64 = 1 << 7;
pub(super) const ICH_HCR_EL2_TC: u64 = 1 << 10;
pub(super) const ICH_HCR_EL2_TALL0: u64 = 1 << 11;
pub(super) const ICH_HCR_EL2_TALL1: u64 = 1 << 12;
const ICH_HCR_EL2_TSEI: u64 = 1 << 13;
pub(super) const ICH_HCR_EL2_TDIR: u64 = 1 << 14;
const ICH_HCR_EL2_EOICOUNT_SHIFT: u32 = 27;
const ICH_HCR_EL2_EOICOUNT_MASK: u64 = 0x1f << ICH_HCR_EL2_EOICOUNT_SHIFT;

const ICH_LR_EL2_EOI: u64 = 1 << 41;
const ICH_LR_EL2_PRIORITY_SHIFT: u32 = 48;
const ICH_LR_EL2_NMI: u64 = 1 << 59;
const ICH_LR_EL2_GROUP: u64 = 1 << 60;
const ICH_LR_EL2_HW: u64 = 1 << 61;
const ICH_LR_EL2_STATE_SHIFT: u32 = 62;
const ICH_LR_EL2_STATE_MASK: u64 = 3 << ICH_LR_EL2_STATE_SHIFT;
const ICH_LR_EL2_STATE_PENDING: u64 = 1;
const ICH_LR_EL2_STATE_PENDING_BIT: u64 = 1 << ICH_LR_EL2_STATE_SHIFT;
const ICH_LR_EL2_STATE_ACTIVE_BIT: u64 = 2 << ICH_LR_EL2_STATE_SHIFT;

const ICH_MISR_EL2_EOI: u64 = 1 << 0;
const ICH_MISR_EL2_U: u64 = 1 << 1;
const ICH_MISR_EL2_LRENP: u64 = 1 << 2;
const ICH_MISR_EL2_NP: u64 = 1 << 3;
const ICH_MISR_EL2_VGRP0E: u64 = 1 << 4;
const ICH_MISR_EL2_VGRP0D: u64 = 1 << 5;
const ICH_MISR_EL2_VGRP1E: u64 = 1 << 6;
const ICH_MISR_EL2_VGRP1D: u64 = 1 << 7;

const ICH_VTR_EL2_TDS: u64 = 1 << 19;
const ICH_VTR_EL2_NV4: u64 = 1 << 20;
const ICH_VTR_EL2_A3V: u64 = 1 << 21;
const ICH_VTR_EL2_IDBITS_SHIFT: u32 = 23;
const ICH_VTR_EL2_PREBITS_SHIFT: u32 = 26;
const ICH_VTR_EL2_PRIBITS_SHIFT: u32 = 29;

fn lr_vintid(lr: u64) -> u64 {
    lr & 0xffff_ffff
}

fn lr_pintid(lr: u64) -> u64 {
    (lr >> 32) & 0x3ff
}

fn lr_prio(lr: u64) -> u64 {
    (lr >> ICH_LR_EL2_PRIORITY_SHIFT) & 0xff
}

fn lr_state(lr: u64) -> u64 {
    (lr >> ICH_LR_EL2_STATE_SHIFT) & 3
}

fn lr_group(lr: u64) -> usize {
    if lr & ICH_LR_EL2_GROUP != 0 { G1NS } else { G0 }
}

/// The HCR_EL2 bits that send an EL1 access to `reg` to its ICV_* twin, as the `icv_access()`
/// calls in the ICC read and write functions choose them. Zero for registers that never go.
fn icv_flags(reg: IccReg) -> u64 {
    match reg {
        IccReg::Pmr | IccReg::Rpr | IccReg::CtlrEl1 | IccReg::Dir => HCR_IMO | HCR_FMO,
        IccReg::Iar0
        | IccReg::Eoir0
        | IccReg::Hppir0
        | IccReg::Bpr0
        | IccReg::Ap0r(_)
        | IccReg::Igrpen0 => HCR_FMO,
        IccReg::Iar1
        | IccReg::Eoir1
        | IccReg::Hppir1
        | IccReg::Bpr1
        | IccReg::Ap1r(_)
        | IccReg::Igrpen1 => HCR_IMO,
        _ => 0,
    }
}

impl CpuState {
    /// The ICH state part of `icc_reset()`.
    pub(super) fn ich_reset(&mut self) {
        self.ich_apr = [[0; 4]; 3];
        self.ich_hcr_el2 = 0;
        self.ich_lr_el2 = [0; 16];
        self.ich_vmcr_el2 = ICH_VMCR_EL2_VFIQEN
            | ((self.min_vbpr() + 1) << ICH_VMCR_EL2_VBPR1_SHIFT)
            | (self.min_vbpr() << ICH_VMCR_EL2_VBPR0_SHIFT);
    }

    /// `icv_access()`: whether an access to `reg` in the current context goes to the ICV_*
    /// register instead, which is so at Non-secure EL1 with HCR_EL2.IMO or FMO set as `reg`
    /// asks.
    pub(super) fn icv_access(&self, reg: IccReg) -> bool {
        let ctx = &self.ctx;
        ctx.hcr_el2 & icv_flags(reg) != 0 && ctx.el == 1 && !ctx.secure_below_el3
    }

    /// `icv_min_vbpr()`.
    fn min_vbpr(&self) -> u64 {
        7 - u64::from(self.vprebits)
    }

    /// `ich_num_aprs()`.
    fn ich_num_aprs(&self) -> usize {
        1 << (self.vprebits - 5)
    }

    /// `read_vbpr()`.
    fn read_vbpr(&self, grp: usize) -> u64 {
        if grp == G0 {
            (self.ich_vmcr_el2 & ICH_VMCR_EL2_VBPR0_MASK) >> ICH_VMCR_EL2_VBPR0_SHIFT
        } else {
            (self.ich_vmcr_el2 & ICH_VMCR_EL2_VBPR1_MASK) >> ICH_VMCR_EL2_VBPR1_SHIFT
        }
    }

    /// `write_vbpr()`: values below the minimum set the minimum.
    fn write_vbpr(&mut self, grp: usize, value: u64) {
        let mut min = self.min_vbpr();
        if grp != G0 {
            min += 1;
        }
        let value = value.max(min) & 7;
        if grp == G0 {
            self.ich_vmcr_el2 = (self.ich_vmcr_el2 & !ICH_VMCR_EL2_VBPR0_MASK)
                | (value << ICH_VMCR_EL2_VBPR0_SHIFT);
        } else {
            self.ich_vmcr_el2 = (self.ich_vmcr_el2 & !ICH_VMCR_EL2_VBPR1_MASK)
                | (value << ICH_VMCR_EL2_VBPR1_SHIFT);
        }
    }

    /// `icv_fullprio_mask()`.
    fn icv_fullprio_mask(&self) -> u64 {
        (!0u64 << (8 - self.vpribits)) & 0xff
    }

    fn vpmr(&self) -> u64 {
        (self.ich_vmcr_el2 & ICH_VMCR_EL2_VPMR_MASK) >> ICH_VMCR_EL2_VPMR_SHIFT
    }

    /// `ich_highest_active_virt_prio()`: the virtual running priority.
    fn highest_active_virt_prio(&self) -> u64 {
        for i in 0..self.ich_num_aprs() {
            let apr = (self.ich_apr[G0][i] | self.ich_apr[G1NS][i]) as u32;
            if apr == 0 {
                continue;
            }
            return (i as u64 * 32 + u64::from(apr.trailing_zeros())) << (self.min_vbpr() + 1);
        }
        0xff
    }

    /// `hppvi_index()`: the list register of the highest priority pending virtual interrupt.
    fn hppvi_index(&self) -> Option<usize> {
        if self.ich_vmcr_el2 & (ICH_VMCR_EL2_VENG0 | ICH_VMCR_EL2_VENG1) == 0 {
            return None;
        }
        let mut idx = None;
        // A priority of 0xff is never reported, which is what the architecture wants.
        let mut prio = 0xff;
        for (i, &lr) in self.ich_lr_el2[..self.num_list_regs].iter().enumerate() {
            if lr_state(lr) != ICH_LR_EL2_STATE_PENDING {
                continue;
            }
            let enable =
                if lr & ICH_LR_EL2_GROUP != 0 { ICH_VMCR_EL2_VENG1 } else { ICH_VMCR_EL2_VENG0 };
            if self.ich_vmcr_el2 & enable == 0 {
                continue;
            }
            let thisprio = lr_prio(lr);
            if thisprio < prio {
                prio = thisprio;
                idx = Some(i);
            }
        }
        idx
    }

    /// `icv_gprio_mask()`: the group priority bits for a virtual interrupt of `group`.
    fn icv_gprio_mask(&self, group: usize) -> u64 {
        let mut group = group;
        if group == G1NS && self.ich_vmcr_el2 & ICH_VMCR_EL2_VCBPR != 0 {
            group = G0;
        }
        let mut bpr = self.read_vbpr(group);
        if group == G1NS {
            bpr = bpr.saturating_sub(1);
        }
        u64::from(!0u32 << (bpr + 1))
    }

    /// `icv_hppi_can_preempt()`.
    fn icv_hppi_can_preempt(&self, lr: u64) -> bool {
        if self.ich_hcr_el2 & ICH_HCR_EL2_EN == 0 {
            // The virtual interface is disabled.
            return false;
        }
        let prio = lr_prio(lr);
        if prio >= self.vpmr() {
            return false;
        }
        let rprio = self.highest_active_virt_prio();
        if rprio == 0xff {
            return true;
        }
        let mask = self.icv_gprio_mask(lr_group(lr));
        (prio & mask) < (rprio & mask)
    }

    /// `eoi_maintenance_interrupt_state()`: the ICH_EISR_EL2 bits, and the EOI, NP and U bits
    /// of ICH_MISR_EL2 in `misr`.
    fn eoi_maintenance_interrupt_state(&self, misr: Option<&mut u64>) -> u64 {
        let mut value = 0;
        let mut validcount = 0;
        let mut seenpending = false;
        for (i, &lr) in self.ich_lr_el2[..self.num_list_regs].iter().enumerate() {
            if lr & (ICH_LR_EL2_STATE_MASK | ICH_LR_EL2_HW | ICH_LR_EL2_EOI) == ICH_LR_EL2_EOI {
                value |= 1 << i;
            }
            if lr & ICH_LR_EL2_STATE_MASK != 0 {
                validcount += 1;
            }
            if lr_state(lr) == ICH_LR_EL2_STATE_PENDING {
                seenpending = true;
            }
        }
        if let Some(misr) = misr {
            if validcount < 2 && self.ich_hcr_el2 & ICH_HCR_EL2_UIE != 0 {
                *misr |= ICH_MISR_EL2_U;
            }
            if !seenpending && self.ich_hcr_el2 & ICH_HCR_EL2_NPIE != 0 {
                *misr |= ICH_MISR_EL2_NP;
            }
            if value != 0 {
                *misr |= ICH_MISR_EL2_EOI;
            }
        }
        value
    }

    /// `maintenance_interrupt_state()`: ICH_MISR_EL2.
    fn maintenance_interrupt_state(&self) -> u64 {
        let hcr = self.ich_hcr_el2;
        let vmcr = self.ich_vmcr_el2;
        let mut value = 0;
        self.eoi_maintenance_interrupt_state(Some(&mut value));
        if hcr & ICH_HCR_EL2_LRENPIE != 0 && hcr & ICH_HCR_EL2_EOICOUNT_MASK != 0 {
            value |= ICH_MISR_EL2_LRENP;
        }
        if hcr & ICH_HCR_EL2_VGRP0EIE != 0 && vmcr & ICH_VMCR_EL2_VENG0 != 0 {
            value |= ICH_MISR_EL2_VGRP0E;
        }
        // QEMU tests VENG1 for the Group 0 disabled condition. Kept as is.
        if hcr & ICH_HCR_EL2_VGRP0DIE != 0 && vmcr & ICH_VMCR_EL2_VENG1 == 0 {
            value |= ICH_MISR_EL2_VGRP0D;
        }
        if hcr & ICH_HCR_EL2_VGRP1EIE != 0 && vmcr & ICH_VMCR_EL2_VENG1 != 0 {
            value |= ICH_MISR_EL2_VGRP1E;
        }
        if hcr & ICH_HCR_EL2_VGRP1DIE != 0 && vmcr & ICH_VMCR_EL2_VENG1 == 0 {
            value |= ICH_MISR_EL2_VGRP1D;
        }
        value
    }

    /// The vIRQ and vFIQ levels of `gicv3_cpuif_virt_irq_fiq_update()`: Group 0 is always a
    /// vFIQ and Group 1 a vIRQ.
    fn virt_levels(&self) -> u64 {
        match self.hppvi_index() {
            Some(idx) => {
                let lr = self.ich_lr_el2[idx];
                if !self.icv_hppi_can_preempt(lr) {
                    0
                } else if lr & ICH_LR_EL2_GROUP != 0 {
                    OUT_VIRQ
                } else {
                    OUT_VFIQ
                }
            }
            None => 0,
        }
    }

    /// `icv_activate_irq()`: move list register `idx` from Pending to Active and set the
    /// active priority bit.
    fn icv_activate_irq(&mut self, idx: usize, grp: usize) {
        let mask = self.icv_gprio_mask(grp);
        let prio = lr_prio(self.ich_lr_el2[idx]) & mask;
        let aprbit = prio >> (8 - self.vprebits);
        self.ich_lr_el2[idx] &= !ICH_LR_EL2_STATE_PENDING_BIT;
        self.ich_lr_el2[idx] |= ICH_LR_EL2_STATE_ACTIVE_BIT;
        self.ich_apr[grp][(aprbit / 32) as usize] |= 1 << (aprbit % 32);
    }

    /// `icv_eoi_split()`.
    fn icv_eoi_split(&self) -> bool {
        self.ich_vmcr_el2 & ICH_VMCR_EL2_VEOIM != 0
    }

    /// `icv_find_active()`: the list register holding active interrupt `irq`.
    fn icv_find_active(&self, irq: u64) -> Option<usize> {
        self.ich_lr_el2[..self.num_list_regs]
            .iter()
            .position(|&lr| lr & ICH_LR_EL2_STATE_ACTIVE_BIT != 0 && lr_vintid(lr) == irq)
    }

    /// `icv_increment_eoicount()`. The count wraps within its five bits.
    fn icv_increment_eoicount(&mut self) {
        let count = (self.ich_hcr_el2 & ICH_HCR_EL2_EOICOUNT_MASK) >> ICH_HCR_EL2_EOICOUNT_SHIFT;
        let count = ((count + 1) << ICH_HCR_EL2_EOICOUNT_SHIFT) & ICH_HCR_EL2_EOICOUNT_MASK;
        self.ich_hcr_el2 = (self.ich_hcr_el2 & !ICH_HCR_EL2_EOICOUNT_MASK) | count;
    }

    /// `icv_drop_prio()`: clear the highest active priority bit, Group 0 first on a tie, and
    /// return the priority it stood for, or 0xff if none was set.
    fn icv_drop_prio(&mut self) -> u64 {
        for i in 0..self.ich_num_aprs() {
            let apr0 = self.ich_apr[G0][i];
            let apr1 = self.ich_apr[G1NS][i];
            if apr0 == 0 && apr1 == 0 {
                continue;
            }
            let count0 = (apr0 as u32).trailing_zeros();
            let count1 = (apr1 as u32).trailing_zeros();
            let count = if count0 <= count1 {
                self.ich_apr[G0][i] &= apr0 - 1;
                count0
            } else {
                self.ich_apr[G1NS][i] &= apr1 - 1;
                count1
            };
            return (u64::from(count) + i as u64 * 32) << (self.min_vbpr() + 1);
        }
        0xff
    }

    /// `ich_vtr_read()`.
    fn ich_vtr(&self) -> u64 {
        (self.num_list_regs as u64 - 1)
            | ICH_VTR_EL2_TDS
            | ICH_VTR_EL2_A3V
            | (1 << ICH_VTR_EL2_IDBITS_SHIFT)
            | (u64::from(self.vprebits - 1) << ICH_VTR_EL2_PREBITS_SHIFT)
            | (u64::from(self.vpribits - 1) << ICH_VTR_EL2_PRIBITS_SHIFT)
            // Revision 3, without GICv4.
            | ICH_VTR_EL2_NV4
    }

    /// `ich_elrsr_read()`.
    fn ich_elrsr(&self) -> u64 {
        let mut value = 0;
        for (i, &lr) in self.ich_lr_el2[..self.num_list_regs].iter().enumerate() {
            if lr & ICH_LR_EL2_STATE_MASK == 0
                && (lr & ICH_LR_EL2_HW != 0 || lr & ICH_LR_EL2_EOI == 0)
            {
                value |= 1 << i;
            }
        }
        value
    }
}

impl GicState {
    /// `gicv3_cpuif_virt_irq_fiq_update()`: work out the vIRQ and vFIQ levels for `cpu`.
    pub(super) fn cpuif_virt_irq_fiq_update(&mut self, cpu: usize) {
        let cs = &mut self.cpu[cpu];
        cs.out = (cs.out & !(OUT_VIRQ | OUT_VFIQ)) | cs.virt_levels();
        self.dirty[cpu] = true;
    }

    /// `gicv3_cpuif_virt_update()`: the vIRQ and vFIQ levels and the maintenance interrupt.
    /// The maintenance interrupt goes to a PPI of this GIC once the pins are driven, outside
    /// the lock.
    pub(super) fn cpuif_virt_update(&mut self, cpu: usize) {
        self.cpuif_virt_irq_fiq_update(cpu);
        let cs = &mut self.cpu[cpu];
        let maint = cs.ich_hcr_el2 & ICH_HCR_EL2_EN != 0 && cs.maintenance_interrupt_state() != 0;
        cs.out = (cs.out & !OUT_MAINT) | if maint { OUT_MAINT } else { 0 };
    }

    /// `icv_deactivate_irq()`: deactivate list register `idx` and, for a hardware interrupt,
    /// the physical interrupt behind it.
    fn icv_deactivate_irq(&mut self, cpu: usize, idx: usize) {
        let lr = self.cpu[cpu].ich_lr_el2[idx];
        if lr & ICH_LR_EL2_HW != 0 {
            let pirq = lr_pintid(lr);
            if pirq < INTID_SECURE {
                self.icc_deactivate_irq(cpu, pirq as u32);
            }
        }
        // Active and pending becomes pending, and active becomes invalid.
        self.cpu[cpu].ich_lr_el2[idx] = lr & !ICH_LR_EL2_STATE_ACTIVE_BIT;
    }

    /// `icv_iar_read()`.
    fn icv_iar(&mut self, cpu: usize, grp: usize) -> u64 {
        let cs = &mut self.cpu[cpu];
        let mut intid = INTID_SPURIOUS;
        if let Some(idx) = cs.hppvi_index() {
            let lr = cs.ich_lr_el2[idx];
            if lr_group(lr) == grp && cs.icv_hppi_can_preempt(lr) {
                intid = lr_vintid(lr);
                if !intid_is_special(intid) {
                    cs.icv_activate_irq(idx, grp);
                } else {
                    // The interrupt goes from Pending to Invalid and the bogus ID is returned,
                    // as the pseudocode says.
                    cs.ich_lr_el2[idx] &= !ICH_LR_EL2_STATE_PENDING_BIT;
                }
            }
        }
        self.cpuif_virt_update(cpu);
        intid
    }

    /// `icv_hppir_read()`.
    fn icv_hppir(&self, cpu: usize, grp: usize) -> u64 {
        let cs = &self.cpu[cpu];
        match cs.hppvi_index() {
            Some(idx) if lr_group(cs.ich_lr_el2[idx]) == grp => lr_vintid(cs.ich_lr_el2[idx]),
            _ => INTID_SPURIOUS,
        }
    }

    /// `icv_eoir_write()`. The priority is dropped before the checks, QEMU's IMPDEF choice.
    fn icv_eoir_write(&mut self, cpu: usize, grp: usize, value: u64) {
        let irq = value & 0xff_ffff;
        if intid_is_special(irq) {
            return;
        }
        let cs = &mut self.cpu[cpu];
        let dropprio = cs.icv_drop_prio();
        if dropprio == 0xff {
            // Nothing is active. Whether the list registers are checked is CONSTRAINED
            // UNPREDICTABLE, and they are not.
            return;
        }
        match cs.icv_find_active(irq) {
            None => {
                // A vLPI that is not in the list registers needs nothing.
                if irq < u64::from(GICV3_LPI_INTID_START) {
                    cs.icv_increment_eoicount();
                }
            }
            Some(idx) => {
                let lr = cs.ich_lr_el2[idx];
                let lr_gprio = lr_prio(lr) & cs.icv_gprio_mask(grp);
                // LPIs lose their active state at once, as no deactivation is expected.
                if lr_group(lr) == grp
                    && lr_gprio == dropprio
                    && (!cs.icv_eoi_split() || irq >= u64::from(GICV3_LPI_INTID_START))
                {
                    self.icv_deactivate_irq(cpu, idx);
                }
            }
        }
        self.cpuif_virt_update(cpu);
    }

    /// `icv_dir_write()`.
    fn icv_dir_write(&mut self, cpu: usize, value: u64) {
        let irq = value & 0xff_ffff;
        // This also catches the special interrupt numbers and LPIs.
        if irq >= u64::from(GICV3_MAXIRQ) || !self.cpu[cpu].icv_eoi_split() {
            return;
        }
        match self.cpu[cpu].icv_find_active(irq) {
            // No list register matches, so count the EOI, which may raise a maintenance
            // interrupt.
            None => self.cpu[cpu].icv_increment_eoicount(),
            Some(idx) => self.icv_deactivate_irq(cpu, idx),
        }
        self.cpuif_virt_update(cpu);
    }

    /// The ICV_* read behind an ICC_* register.
    pub(super) fn icv_read(&mut self, cpu: usize, reg: IccReg) -> u64 {
        let cs = &self.cpu[cpu];
        match reg {
            IccReg::Pmr => cs.vpmr(),
            IccReg::Rpr => cs.highest_active_virt_prio(),
            IccReg::Iar0 => self.icv_iar(cpu, G0),
            IccReg::Iar1 => self.icv_iar(cpu, G1NS),
            IccReg::Hppir0 => self.icv_hppir(cpu, G0),
            IccReg::Hppir1 => self.icv_hppir(cpu, G1NS),
            IccReg::Bpr0 => cs.read_vbpr(G0),
            IccReg::Bpr1 => {
                if cs.ich_vmcr_el2 & ICH_VMCR_EL2_VCBPR != 0 {
                    // Reads give VBPR0 plus one, saturating at 7.
                    (cs.read_vbpr(G0) + 1).min(7)
                } else {
                    cs.read_vbpr(G1NS)
                }
            }
            IccReg::Ap0r(n) => cs.ich_apr[G0][usize::from(n & 3)],
            IccReg::Ap1r(n) => cs.ich_apr[G1NS][usize::from(n & 3)],
            IccReg::Igrpen0 => cs.ich_vmcr_el2 & ICH_VMCR_EL2_VENG0,
            IccReg::Igrpen1 => (cs.ich_vmcr_el2 & ICH_VMCR_EL2_VENG1) >> 1,
            IccReg::CtlrEl1 => {
                // The fixed fields match the ones ICH_VTR_EL2 reports.
                let mut value = ICC_CTLR_EL1_A3V
                    | (1 << ICC_CTLR_EL1_IDBITS_SHIFT)
                    | (u64::from(cs.vpribits - 1) << ICC_CTLR_EL1_PRIBITS_SHIFT);
                if cs.ich_vmcr_el2 & ICH_VMCR_EL2_VEOIM != 0 {
                    value |= ICC_CTLR_EL1_EOIMODE;
                }
                if cs.ich_vmcr_el2 & ICH_VMCR_EL2_VCBPR != 0 {
                    value |= ICC_CTLR_EL1_CBPR;
                }
                value
            }
            _ => 0,
        }
    }

    /// The ICV_* write behind an ICC_* register.
    pub(super) fn icv_write(&mut self, cpu: usize, reg: IccReg, value: u64) {
        let cs = &mut self.cpu[cpu];
        match reg {
            IccReg::Pmr => {
                let value = value & cs.icv_fullprio_mask();
                cs.ich_vmcr_el2 = (cs.ich_vmcr_el2 & !ICH_VMCR_EL2_VPMR_MASK)
                    | (value << ICH_VMCR_EL2_VPMR_SHIFT);
                self.cpuif_virt_irq_fiq_update(cpu);
            }
            IccReg::Eoir0 => self.icv_eoir_write(cpu, G0, value),
            IccReg::Eoir1 => self.icv_eoir_write(cpu, G1NS, value),
            IccReg::Dir => self.icv_dir_write(cpu, value),
            IccReg::Bpr0 | IccReg::Bpr1 => {
                let grp = if reg == IccReg::Bpr0 { G0 } else { G1NS };
                if grp == G1NS && cs.ich_vmcr_el2 & ICH_VMCR_EL2_VCBPR != 0 {
                    // VCBPR makes VBPR1 writes ignored.
                    return;
                }
                cs.write_vbpr(grp, value);
                self.cpuif_virt_irq_fiq_update(cpu);
            }
            IccReg::Ap0r(n) | IccReg::Ap1r(n) => {
                let grp = if matches!(reg, IccReg::Ap0r(_)) { G0 } else { G1NS };
                cs.ich_apr[grp][usize::from(n & 3)] = value & 0xffff_ffff;
                self.cpuif_virt_irq_fiq_update(cpu);
            }
            IccReg::Igrpen0 | IccReg::Igrpen1 => {
                let enbit = if reg == IccReg::Igrpen0 { 0 } else { 1 };
                cs.ich_vmcr_el2 = (cs.ich_vmcr_el2 & !(1 << enbit)) | ((value & 1) << enbit);
                self.cpuif_virt_update(cpu);
            }
            IccReg::CtlrEl1 => {
                let mut vmcr = cs.ich_vmcr_el2 & !(ICH_VMCR_EL2_VCBPR | ICH_VMCR_EL2_VEOIM);
                if value & ICC_CTLR_EL1_CBPR != 0 {
                    vmcr |= ICH_VMCR_EL2_VCBPR;
                }
                if value & ICC_CTLR_EL1_EOIMODE != 0 {
                    vmcr |= ICH_VMCR_EL2_VEOIM;
                }
                cs.ich_vmcr_el2 = vmcr;
                self.cpuif_virt_irq_fiq_update(cpu);
            }
            _ => {}
        }
    }

    /// Read an ICH_* register.
    pub(super) fn ich_read(&self, cpu: usize, reg: IccReg) -> u64 {
        let cs = &self.cpu[cpu];
        match reg {
            IccReg::IchAp0r(n) => cs.ich_apr[G0][usize::from(n & 3)],
            IccReg::IchAp1r(n) => cs.ich_apr[G1NS][usize::from(n & 3)],
            IccReg::IchHcr => cs.ich_hcr_el2,
            IccReg::IchVtr => cs.ich_vtr(),
            IccReg::IchMisr => cs.maintenance_interrupt_state(),
            IccReg::IchEisr => cs.eoi_maintenance_interrupt_state(None),
            IccReg::IchElrsr => cs.ich_elrsr(),
            IccReg::IchVmcr => cs.ich_vmcr_el2,
            IccReg::IchLr(n) => cs.ich_lr_el2[usize::from(n & 15)],
            _ => 0,
        }
    }

    /// Write an ICH_* register. The read only ones ignore writes.
    pub(super) fn ich_write(&mut self, cpu: usize, reg: IccReg, value: u64) {
        let cs = &mut self.cpu[cpu];
        match reg {
            IccReg::IchAp0r(n) | IccReg::IchAp1r(n) => {
                let grp = if matches!(reg, IccReg::IchAp0r(_)) { G0 } else { G1NS };
                cs.ich_apr[grp][usize::from(n & 3)] = value & 0xffff_ffff;
                self.cpuif_virt_irq_fiq_update(cpu);
            }
            IccReg::IchHcr => {
                cs.ich_hcr_el2 = value
                    & (ICH_HCR_EL2_EN
                        | ICH_HCR_EL2_UIE
                        | ICH_HCR_EL2_LRENPIE
                        | ICH_HCR_EL2_NPIE
                        | ICH_HCR_EL2_VGRP0EIE
                        | ICH_HCR_EL2_VGRP0DIE
                        | ICH_HCR_EL2_VGRP1EIE
                        | ICH_HCR_EL2_VGRP1DIE
                        | ICH_HCR_EL2_TC
                        | ICH_HCR_EL2_TALL0
                        | ICH_HCR_EL2_TALL1
                        | ICH_HCR_EL2_TSEI
                        | ICH_HCR_EL2_TDIR
                        | ICH_HCR_EL2_EOICOUNT_MASK);
                self.cpuif_virt_update(cpu);
            }
            IccReg::IchVmcr => {
                cs.ich_vmcr_el2 = (value
                    & (ICH_VMCR_EL2_VENG0
                        | ICH_VMCR_EL2_VENG1
                        | ICH_VMCR_EL2_VCBPR
                        | ICH_VMCR_EL2_VEOIM
                        | ICH_VMCR_EL2_VBPR1_MASK
                        | ICH_VMCR_EL2_VBPR0_MASK
                        | ICH_VMCR_EL2_VPMR_MASK))
                    | ICH_VMCR_EL2_VFIQEN;
                // Writing a BPR below the minimum sets the minimum.
                cs.write_vbpr(G0, cs.read_vbpr(G0));
                cs.write_vbpr(G1, cs.read_vbpr(G1));
                self.cpuif_virt_update(cpu);
            }
            IccReg::IchLr(n) => {
                let mut value = value;
                // The unimplemented priority bits are RES0, and so is NMI without
                // FEAT_GICv3_NMI.
                if cs.vpribits < 8 {
                    let low = (1u64 << (8 - cs.vpribits)) - 1;
                    value &= !(low << ICH_LR_EL2_PRIORITY_SHIFT);
                }
                value &= !ICH_LR_EL2_NMI;
                cs.ich_lr_el2[usize::from(n & 15)] = value;
                self.cpuif_virt_update(cpu);
            }
            _ => {}
        }
    }
}
