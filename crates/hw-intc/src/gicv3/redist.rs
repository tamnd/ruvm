// SPDX-License-Identifier: GPL-2.0-or-later

//! The redistributor registers and the physical LPIs, from hw/intc/arm_gicv3_redist.c.

use std::sync::Weak;

use ruvm_mem::MemTxAttrs;

use super::{
    G1NS, GICD_TYPER_IDBITS, GICR_CTLR_ENABLE_LPIS, GICR_TYPER_PLPIS, GICR_WAKER_CHILDREN_ASLEEP,
    GICR_WAKER_PROCESSOR_SLEEP, GICV3_IIDR, GICV3_LPI_INTID_START, GICV3_PIDR0_REDIST, GicState,
    Pending, deposit_half, half_shuffle32, half_unshuffle32,
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

/// `LPI_CTE_ENABLED`: the enable bit of an LPI configuration table entry.
const LPI_CTE_ENABLED: u8 = 1 << 0;
/// `LPI_PRIORITY_MASK`: the priority bits of an LPI configuration table entry.
const LPI_PRIORITY_MASK: u8 = 0xfc;
/// GICR_PROPBASER.IDBITS.
const GICR_PROPBASER_IDBITS_MASK: u64 = 0x1f;
/// GICR_PROPBASER.PHYADDR, bits 12 to 51.
const GICR_PROPBASER_PHYADDR_MASK: u64 = ((1 << 40) - 1) << 12;
/// GICR_PENDBASER.PHYADDR, bits 16 to 51.
const GICR_PENDBASER_PHYADDR_MASK: u64 = ((1 << 36) - 1) << 16;
/// The largest pending table, for 16 bits of interrupt ID, less the bytes for the INTIDs below
/// the first LPI, which are never read.
const PENDT_SCAN_MAX: usize = ((1 << (GICD_TYPER_IDBITS + 1)) - GICV3_LPI_INTID_START as usize) / 8;

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
            // GICR_TYPER.DPGS is 0, so the DPG bits are RAZ/WI, and nothing happens
            // asynchronously, so UWP and RWP are too. With LPIs, EnableLPIs is writable.
            GICR_CTLR if self.cpu[cpu].gicr_typer & GICR_TYPER_PLPIS != 0 => {
                if v32 & GICR_CTLR_ENABLE_LPIS != 0 {
                    self.cpu[cpu].gicr_ctlr |= GICR_CTLR_ENABLE_LPIS;
                    // Pick up whatever is already pending in the pending table.
                    self.update_lpi(cpu);
                } else {
                    self.cpu[cpu].gicr_ctlr &= !GICR_CTLR_ENABLE_LPIS;
                    // The best interrupt may have been an LPI.
                    self.redist_update(cpu);
                }
            }
            // GICR_STATUSR, GICR_ICFGR0 and GICR_INMIR0 ignore writes, and so do the read only
            // registers.
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

impl GicState {
    /// `address_space_read()` from the LPI table memory. What cannot be read reads as zero.
    pub(super) fn dma_read(&self, addr: u64, buf: &mut [u8]) {
        buf.fill(0);
        if let Some(a) = self.dma.as_ref().and_then(Weak::upgrade) {
            let _ = a.read(addr, MemTxAttrs::UNSPECIFIED, buf);
        }
    }

    /// `address_space_write()` to the LPI table memory.
    fn dma_write(&self, addr: u64, buf: &[u8]) {
        if let Some(a) = self.dma.as_ref().and_then(Weak::upgrade) {
            let _ = a.write(addr, MemTxAttrs::UNSPECIFIED, buf);
        }
    }

    /// GICR_PROPBASER.IDBITS of `cpu`, capped at what GICD_TYPER reports.
    fn lpi_idbits(&self, cpu: usize) -> u64 {
        (self.cpu[cpu].gicr_propbaser & GICR_PROPBASER_IDBITS_MASK).min(GICD_TYPER_IDBITS)
    }

    fn lpis_enabled(&self, cpu: usize) -> bool {
        self.cpu[cpu].gicr_ctlr & GICR_CTLR_ENABLE_LPIS != 0
    }

    /// `update_for_one_lpi()`: if LPI `irq`, whose configuration table is at `ctbase`, is
    /// enabled and beats `hpp`, make it `hpp`.
    fn update_for_one_lpi(&self, irq: u32, ctbase: u64, ds: bool, hpp: &mut Pending) {
        let mut lpite = [0u8];
        let index = i64::from(irq) - i64::from(GICV3_LPI_INTID_START);
        self.dma_read(ctbase.wrapping_add(index as u64), &mut lpite);
        let lpite = lpite[0];
        if lpite & LPI_CTE_ENABLED == 0 {
            return;
        }
        let prio =
            if ds { lpite & LPI_PRIORITY_MASK } else { ((lpite & LPI_PRIORITY_MASK) >> 1) | 0x80 };
        if prio < hpp.prio || (prio == hpp.prio && irq <= hpp.irq) {
            hpp.irq = irq;
            hpp.prio = prio;
            // LPIs are always Non-secure Group 1.
            hpp.grp = G1NS;
        }
    }

    /// `update_for_all_lpis()`: find the best pending LPI from scratch, scanning the pending
    /// table at `ptbase` and looking each pending LPI up in the configuration table at `ctbase`.
    /// `ptsizebits` is the number of interrupt ID bits less one.
    fn update_for_all_lpis(
        &self,
        ptbase: u64,
        ctbase: u64,
        ptsizebits: u64,
        ds: bool,
        hpp: &mut Pending,
    ) {
        hpp.prio = 0xff;
        let start = (GICV3_LPI_INTID_START / 8) as usize;
        let end = ((1u64 << (ptsizebits + 1)) / 8) as usize;
        if end <= start {
            return;
        }
        let mut pend = [0u8; PENDT_SCAN_MAX];
        let pend = &mut pend[..end - start];
        self.dma_read(ptbase.wrapping_add(start as u64), pend);
        for (i, &byte) in pend.iter().enumerate() {
            let mut byte = byte;
            while byte != 0 {
                let bit = byte.trailing_zeros();
                self.update_for_one_lpi(((start + i) * 8) as u32 + bit, ctbase, ds, hpp);
                byte &= !(1 << bit);
            }
        }
    }

    /// `set_pending_table_bit()`: set the pending bit of `irq` in the table at `ptbase` to
    /// `level`. False if it was already there.
    fn set_pending_table_bit(&self, ptbase: u64, irq: u32, level: bool) -> bool {
        let addr = ptbase.wrapping_add(u64::from(irq / 8));
        let mut pend = [0u8];
        self.dma_read(addr, &mut pend);
        let bit = 1u8 << (irq % 8);
        if (pend[0] & bit != 0) == level {
            return false;
        }
        if level {
            pend[0] |= bit;
        } else {
            pend[0] &= !bit;
        }
        self.dma_write(addr, &pend);
        true
    }

    /// `gicv3_redist_check_lpi_priority()`.
    fn check_lpi_priority(&mut self, cpu: usize, irq: u32) {
        let ctbase = self.cpu[cpu].gicr_propbaser & GICR_PROPBASER_PHYADDR_MASK;
        let mut hpp = self.cpu[cpu].hpplpi;
        self.update_for_one_lpi(irq, ctbase, self.ds(), &mut hpp);
        self.cpu[cpu].hpplpi = hpp;
    }

    /// `gicv3_redist_update_lpi_only()`: rescan the pending table of `cpu` for its best LPI.
    pub(super) fn update_lpi_only(&mut self, cpu: usize) {
        let idbits = self.lpi_idbits(cpu);
        if !self.lpis_enabled(cpu) {
            return;
        }
        let cs = &self.cpu[cpu];
        let ptbase = cs.gicr_pendbaser & GICR_PENDBASER_PHYADDR_MASK;
        let ctbase = cs.gicr_propbaser & GICR_PROPBASER_PHYADDR_MASK;
        let mut hpp = cs.hpplpi;
        self.update_for_all_lpis(ptbase, ctbase, idbits, self.ds(), &mut hpp);
        self.cpu[cpu].hpplpi = hpp;
    }

    /// `gicv3_redist_update_lpi()`.
    pub(super) fn update_lpi(&mut self, cpu: usize) {
        self.update_lpi_only(cpu);
        self.redist_update(cpu);
    }

    /// `gicv3_redist_lpi_pending()`: set or clear the pending bit of LPI `irq` of `cpu`.
    pub(super) fn lpi_pending(&mut self, cpu: usize, irq: u32, level: bool) {
        let ptbase = self.cpu[cpu].gicr_pendbaser & GICR_PENDBASER_PHYADDR_MASK;
        if !self.set_pending_table_bit(ptbase, irq, level) {
            return;
        }
        if level {
            // A newly pending LPI only has to be compared with the best one.
            self.check_lpi_priority(cpu, irq);
            self.redist_update(cpu);
        } else if irq == self.cpu[cpu].hpplpi.irq {
            self.update_lpi(cpu);
        }
    }

    /// `gicv3_redist_process_lpi()`: an LPI translated by the ITS, if `cpu` takes it.
    pub(super) fn process_lpi(&mut self, cpu: usize, irq: u32, level: bool) {
        let idbits = self.lpi_idbits(cpu);
        if !self.lpis_enabled(cpu)
            || u64::from(irq) > (1u64 << (idbits + 1)) - 1
            || irq < GICV3_LPI_INTID_START
        {
            return;
        }
        self.lpi_pending(cpu, irq, level);
    }

    /// `gicv3_redist_inv_lpi()`. Only the best LPI is cached, so this rescans.
    pub(super) fn inv_lpi(&mut self, cpu: usize) {
        self.update_lpi(cpu);
    }

    /// `gicv3_redist_mov_lpi()`: move the pending state of LPI `irq` from `src` to `dest`.
    /// With LPIs off on either side, nothing happens.
    pub(super) fn mov_lpi(&mut self, src: usize, dest: usize, irq: u32) {
        if !self.lpis_enabled(src) || !self.lpis_enabled(dest) {
            return;
        }
        let idbits = self.lpi_idbits(src).min(self.cpu[dest].gicr_propbaser & 0x1f);
        let pendt_size = 1u64 << (idbits + 1);
        // QEMU compares the byte index with the size in bits, so this is the same check.
        if u64::from(irq / 8) >= pendt_size {
            return;
        }
        let src_baddr = self.cpu[src].gicr_pendbaser & GICR_PENDBASER_PHYADDR_MASK;
        if !self.set_pending_table_bit(src_baddr, irq, false) {
            return;
        }
        if irq == self.cpu[src].hpplpi.irq {
            self.update_lpi(src);
        }
        self.lpi_pending(dest, irq, true);
    }

    /// `gicv3_redist_movall_lpis()`: move every pending LPI of `src` to `dest`. LPIs already
    /// pending on `dest` stay pending.
    pub(super) fn movall_lpis(&mut self, src: usize, dest: usize) {
        if !self.lpis_enabled(src) || !self.lpis_enabled(dest) {
            return;
        }
        let idbits = self.lpi_idbits(src).min(self.cpu[dest].gicr_propbaser & 0x1f);
        let pendt_size = 1u64 << (idbits + 1);
        let src_baddr = self.cpu[src].gicr_pendbaser & GICR_PENDBASER_PHYADDR_MASK;
        let dest_baddr = self.cpu[dest].gicr_pendbaser & GICR_PENDBASER_PHYADDR_MASK;
        for i in u64::from(GICV3_LPI_INTID_START / 8)..pendt_size / 8 {
            let mut src_pend = [0u8];
            self.dma_read(src_baddr.wrapping_add(i), &mut src_pend);
            if src_pend[0] == 0 {
                continue;
            }
            let mut dest_pend = [0u8];
            self.dma_read(dest_baddr.wrapping_add(i), &mut dest_pend);
            dest_pend[0] |= src_pend[0];
            self.dma_write(src_baddr.wrapping_add(i), &[0]);
            self.dma_write(dest_baddr.wrapping_add(i), &dest_pend);
        }
        self.update_lpi(src);
        self.update_lpi(dest);
    }
}
