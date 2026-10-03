// SPDX-License-Identifier: GPL-2.0-or-later

//! The redistributor registers, from hw/intc/arm_gicv3_redist.c.

use super::{
    GICR_WAKER_CHILDREN_ASLEEP, GICR_WAKER_PROCESSOR_SLEEP, GICV3_IIDR, GICV3_PIDR0_REDIST,
    GicState, deposit_half, half_shuffle32, half_unshuffle32,
};

const GICR_SGI_OFFSET: u64 = 0x10000;

const GICR_CTLR: u64 = 0x0000;
const GICR_IIDR: u64 = 0x0004;
const GICR_TYPER: u64 = 0x0008;
const GICR_STATUSR: u64 = 0x0010;
const GICR_WAKER: u64 = 0x0014;
const GICR_PROPBASER: u64 = 0x0070;
const GICR_PENDBASER: u64 = 0x0078;
const GICR_IDREGS: u64 = 0xffd0;

const GICR_IGROUPR0: u64 = GICR_SGI_OFFSET + 0x0080;
const GICR_ISENABLER0: u64 = GICR_SGI_OFFSET + 0x0100;
const GICR_ICENABLER0: u64 = GICR_SGI_OFFSET + 0x0180;
const GICR_ISPENDR0: u64 = GICR_SGI_OFFSET + 0x0200;
const GICR_ICPENDR0: u64 = GICR_SGI_OFFSET + 0x0280;
const GICR_ISACTIVER0: u64 = GICR_SGI_OFFSET + 0x0300;
const GICR_ICACTIVER0: u64 = GICR_SGI_OFFSET + 0x0380;
const GICR_IPRIORITYR: u64 = GICR_SGI_OFFSET + 0x0400;
const GICR_IPRIORITYR_END: u64 = GICR_IPRIORITYR + 0x1f;
const GICR_ICFGR0: u64 = GICR_SGI_OFFSET + 0x0c00;
const GICR_ICFGR1: u64 = GICR_SGI_OFFSET + 0x0c04;
const GICR_IGRPMODR0: u64 = GICR_SGI_OFFSET + 0x0d00;
const GICR_NSACR: u64 = GICR_SGI_OFFSET + 0x0e00;
const GICR_INMIR0: u64 = GICR_SGI_OFFSET + 0x0f80;

/// Which redistributor register a set/clear pair reaches.
#[derive(Clone, Copy)]
enum Reg0 {
    Enable,
    Pend,
    Active,
}

impl GicState {
    /// `mask_group()`: the SGIs and PPIs this access may see or change. GICR_NSACR does not
    /// affect these registers, unlike GICD_NSACR.
    fn gicr_mask_group(&self, cpu: usize, secure: bool) -> u32 {
        if !secure && !self.ds() { self.cpu[cpu].gicr_igroupr0 } else { 0xffff_ffff }
    }

    /// `gicr_read_ipriorityr()`.
    fn gicr_read_ipriorityr(&self, cpu: usize, secure: bool, irq: u32) -> u32 {
        let cs = &self.cpu[cpu];
        let prio = u32::from(cs.gicr_ipriorityr[irq as usize]);
        if !secure && !self.ds() {
            if cs.gicr_igroupr0 & (1 << irq) == 0 {
                return 0;
            }
            return (prio << 1) & 0xff;
        }
        prio
    }

    /// `gicr_write_ipriorityr()`.
    fn gicr_write_ipriorityr(&mut self, cpu: usize, secure: bool, irq: u32, value: u8) {
        let ns_view = !secure && !self.ds();
        let cs = &mut self.cpu[cpu];
        let mut value = value;
        if ns_view {
            if cs.gicr_igroupr0 & (1 << irq) == 0 {
                return;
            }
            value = 0x80 | (value >> 1);
        }
        cs.gicr_ipriorityr[irq as usize] = value;
    }

    fn gicr_reg0_mut(&mut self, cpu: usize, which: Reg0) -> &mut u32 {
        let cs = &mut self.cpu[cpu];
        match which {
            Reg0::Enable => &mut cs.gicr_ienabler0,
            Reg0::Pend => &mut cs.gicr_ipendr0,
            Reg0::Active => &mut cs.gicr_iactiver0,
        }
    }

    /// `gicr_write_set_bitmap_reg()` and `gicr_write_clear_bitmap_reg()`.
    fn gicr_write_bitmap_reg(&mut self, cpu: usize, secure: bool, which: Reg0, v: u32, set: bool) {
        let v = v & self.gicr_mask_group(cpu, secure);
        let reg = self.gicr_reg0_mut(cpu, which);
        if set {
            *reg |= v;
        } else {
            *reg &= !v;
        }
        self.redist_update(cpu);
    }

    /// `gicv3_redist_read()` for one CPU's frames. Anything unhandled reads as zero.
    pub(super) fn redist_read(&self, secure: bool, cpu: usize, offset: u64, size: u32) -> u64 {
        let r = match size {
            1 => self.gicr_readb(cpu, secure, offset),
            4 => self.gicr_readl(cpu, secure, offset).map(u64::from),
            8 => self.gicr_readll(cpu, offset),
            _ => None,
        };
        r.unwrap_or(0)
    }

    /// `gicv3_redist_write()` for one CPU's frames. Anything unhandled is ignored.
    pub(super) fn redist_write(
        &mut self,
        secure: bool,
        cpu: usize,
        offset: u64,
        size: u32,
        value: u64,
    ) {
        match size {
            1 => self.gicr_writeb(cpu, secure, offset, value),
            4 => self.gicr_writel(cpu, secure, offset, value),
            8 => self.gicr_writell(cpu, offset, value),
            _ => {}
        }
    }

    fn gicr_readb(&self, cpu: usize, secure: bool, offset: u64) -> Option<u64> {
        match offset {
            GICR_IPRIORITYR..=GICR_IPRIORITYR_END => {
                let irq = (offset - GICR_IPRIORITYR) as u32;
                Some(u64::from(self.gicr_read_ipriorityr(cpu, secure, irq)))
            }
            _ => None,
        }
    }

    fn gicr_writeb(&mut self, cpu: usize, secure: bool, offset: u64, value: u64) {
        if let GICR_IPRIORITYR..=GICR_IPRIORITYR_END = offset {
            let irq = (offset - GICR_IPRIORITYR) as u32;
            self.gicr_write_ipriorityr(cpu, secure, irq, value as u8);
            self.redist_update(cpu);
        }
    }

    fn gicr_readl(&self, cpu: usize, secure: bool, offset: u64) -> Option<u32> {
        let cs = &self.cpu[cpu];
        let v = match offset {
            GICR_CTLR => cs.gicr_ctlr,
            GICR_IIDR => GICV3_IIDR,
            GICR_TYPER => cs.gicr_typer as u32,
            0x000c => (cs.gicr_typer >> 32) as u32,
            GICR_STATUSR => 0,
            GICR_WAKER => cs.gicr_waker,
            GICR_PROPBASER => cs.gicr_propbaser as u32,
            0x0074 => (cs.gicr_propbaser >> 32) as u32,
            GICR_PENDBASER => cs.gicr_pendbaser as u32,
            0x007c => (cs.gicr_pendbaser >> 32) as u32,
            GICR_IGROUPR0 => {
                if !secure && !self.ds() {
                    0
                } else {
                    cs.gicr_igroupr0
                }
            }
            GICR_ISENABLER0 | GICR_ICENABLER0 => {
                cs.gicr_ienabler0 & self.gicr_mask_group(cpu, secure)
            }
            GICR_ISPENDR0 | GICR_ICPENDR0 => {
                // A level triggered interrupt is pending while its line is high.
                let val = cs.gicr_ipendr0 | (!cs.edge_trigger & cs.level);
                val & self.gicr_mask_group(cpu, secure)
            }
            GICR_ISACTIVER0 | GICR_ICACTIVER0 => {
                cs.gicr_iactiver0 & self.gicr_mask_group(cpu, secure)
            }
            GICR_IPRIORITYR..=GICR_IPRIORITYR_END => {
                let irq = (offset - GICR_IPRIORITYR) as u32;
                let mut value = 0;
                for i in (irq..irq + 4).rev() {
                    value = (value << 8) | self.gicr_read_ipriorityr(cpu, secure, i);
                }
                value
            }
            GICR_INMIR0 => 0,
            GICR_ICFGR0 | GICR_ICFGR1 => {
                // One bit per interrupt, spread out into the odd bits.
                let value = cs.edge_trigger & self.gicr_mask_group(cpu, secure);
                let value = if offset == GICR_ICFGR1 { value >> 16 } else { value & 0xffff };
                half_shuffle32(value) << 1
            }
            GICR_IGRPMODR0 => {
                if self.ds() || !secure {
                    0
                } else {
                    cs.gicr_igrpmodr0
                }
            }
            GICR_NSACR => {
                if self.ds() || !secure {
                    0
                } else {
                    cs.gicr_nsacr
                }
            }
            GICR_IDREGS..=0xffff => self.idreg(offset - GICR_IDREGS, GICV3_PIDR0_REDIST),
            _ => return None,
        };
        Some(v)
    }

    fn gicr_writel(&mut self, cpu: usize, secure: bool, offset: u64, value: u64) {
        let v32 = value as u32;
        match offset {
            GICR_WAKER => {
                // The interface is never asynchronous, so ChildrenAsleep follows ProcessorSleep
                // at once.
                let mut v = v32 & GICR_WAKER_PROCESSOR_SLEEP;
                if v != 0 {
                    v |= GICR_WAKER_CHILDREN_ASLEEP;
                }
                self.cpu[cpu].gicr_waker = v;
            }
            GICR_PROPBASER | 0x0074 => {
                let cs = &mut self.cpu[cpu];
                cs.gicr_propbaser = deposit_half(cs.gicr_propbaser, offset & 4 != 0, value);
            }
            GICR_PENDBASER | 0x007c => {
                let cs = &mut self.cpu[cpu];
                cs.gicr_pendbaser = deposit_half(cs.gicr_pendbaser, offset & 4 != 0, value);
            }
            GICR_IGROUPR0 => {
                if !secure && !self.ds() {
                    return;
                }
                self.cpu[cpu].gicr_igroupr0 = v32;
                self.redist_update(cpu);
            }
            GICR_ISENABLER0 => self.gicr_write_bitmap_reg(cpu, secure, Reg0::Enable, v32, true),
            GICR_ICENABLER0 => self.gicr_write_bitmap_reg(cpu, secure, Reg0::Enable, v32, false),
            GICR_ISACTIVER0 => self.gicr_write_bitmap_reg(cpu, secure, Reg0::Active, v32, true),
            GICR_ICACTIVER0 => self.gicr_write_bitmap_reg(cpu, secure, Reg0::Active, v32, false),
            GICR_ISPENDR0 => self.gicr_write_bitmap_reg(cpu, secure, Reg0::Pend, v32, true),
            GICR_ICPENDR0 => self.gicr_write_bitmap_reg(cpu, secure, Reg0::Pend, v32, false),
            GICR_IPRIORITYR..=GICR_IPRIORITYR_END => {
                let irq = (offset - GICR_IPRIORITYR) as u32;
                for (n, i) in (irq..irq + 4).enumerate() {
                    self.gicr_write_ipriorityr(cpu, secure, i, (v32 >> (n * 8)) as u8);
                }
                self.redist_update(cpu);
            }
            GICR_ICFGR1 => {
                // SGIs are always edge triggered, so only the PPI half is writable.
                let value = half_unshuffle32(v32 >> 1) << 16;
                let mask = self.gicr_mask_group(cpu, secure) & 0xffff_0000;
                let cs = &mut self.cpu[cpu];
                cs.edge_trigger = (cs.edge_trigger & !mask) | (value & mask);
                self.redist_update(cpu);
            }
            GICR_IGRPMODR0 => {
                if self.ds() || !secure {
                    return;
                }
                self.cpu[cpu].gicr_igrpmodr0 = v32;
                self.redist_update(cpu);
            }
            GICR_NSACR => {
                if self.ds() || !secure {
                    return;
                }
                // Only access checks read this, so nothing needs recomputing.
                self.cpu[cpu].gicr_nsacr = v32;
            }
            // GICR_CTLR has nothing writable without LPIs. GICR_STATUSR, GICR_ICFGR0 and
            // GICR_INMIR0 ignore writes, and so do the read only registers.
            _ => {}
        }
    }

    fn gicr_readll(&self, cpu: usize, offset: u64) -> Option<u64> {
        let cs = &self.cpu[cpu];
        match offset {
            GICR_TYPER => Some(cs.gicr_typer),
            GICR_PROPBASER => Some(cs.gicr_propbaser),
            GICR_PENDBASER => Some(cs.gicr_pendbaser),
            _ => None,
        }
    }

    fn gicr_writell(&mut self, cpu: usize, offset: u64, value: u64) {
        let cs = &mut self.cpu[cpu];
        match offset {
            GICR_PROPBASER => cs.gicr_propbaser = value,
            GICR_PENDBASER => cs.gicr_pendbaser = value,
            _ => {}
        }
    }
}
