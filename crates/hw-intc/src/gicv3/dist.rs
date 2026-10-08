// SPDX-License-Identifier: GPL-2.0-or-later

//! The distributor registers, from hw/intc/arm_gicv3_dist.c.

use super::{
    BMP_WORDS, GIC_INTERNAL, GICD_CTLR_ARE_NS, GICD_CTLR_ARE_S, GICD_CTLR_DS, GICD_CTLR_EN_GRP0,
    GICD_CTLR_EN_GRP1_ALL, GICD_CTLR_EN_GRP1NS, GICD_CTLR_EN_GRP1S, GICD_CTLR_RWP, GICV3_IIDR,
    GICV3_PIDR0_DIST, GicState, bmp_test, bmp_word, bmp_word_mut, deposit_half, half_shuffle32,
    half_unshuffle32, half_unshuffle64,
};

const GICD_CTLR: u64 = 0x0000;
const GICD_TYPER: u64 = 0x0004;
const GICD_IIDR: u64 = 0x0008;
const GICD_TYPER2: u64 = 0x000c;
const GICD_STATUSR: u64 = 0x0010;
const GICD_IGROUPR: u64 = 0x0080;
const GICD_ISENABLER: u64 = 0x0100;
const GICD_ICENABLER: u64 = 0x0180;
const GICD_ISPENDR: u64 = 0x0200;
const GICD_ICPENDR: u64 = 0x0280;
const GICD_ISACTIVER: u64 = 0x0300;
const GICD_ICACTIVER: u64 = 0x0380;
const GICD_IPRIORITYR: u64 = 0x0400;
const GICD_ITARGETSR: u64 = 0x0800;
const GICD_ICFGR: u64 = 0x0c00;
const GICD_IGRPMODR: u64 = 0x0d00;
const GICD_NSACR: u64 = 0x0e00;
const GICD_SGIR: u64 = 0x0f00;
const GICD_CPENDSGIR: u64 = 0x0f10;
const GICD_SPENDSGIR: u64 = 0x0f20;
const GICD_INMIR: u64 = 0x0f80;
const GICD_IROUTER: u64 = 0x6000;
const GICD_IDREGS: u64 = 0xffd0;

/// Which GICD_NSACR check a set/clear bitmap register applies to Non-secure accesses.
#[derive(Clone, Copy)]
enum NsacrMask {
    None,
    /// `mask_nsacr_ge1`: Non-secure may touch interrupts whose NSACR field is at least 1.
    Ge1,
    /// `mask_nsacr_ge2`: Non-secure may touch interrupts whose NSACR field is at least 2.
    Ge2,
}

/// The first interrupt of a 32 bit register `offset` bytes into a one bit per interrupt block.
fn bitmap_irq(offset: u64) -> u32 {
    (offset * 8) as u32
}

impl GicState {
    /// Whether this access sees the restricted Non-secure view.
    fn ns_view(&self, secure: bool) -> bool {
        !secure && !self.ds()
    }

    /// `gicd_ns_access()`: the GICD_NSACR field of `irq`.
    fn gicd_ns_access(&self, irq: u32) -> u32 {
        (self.gicd_nsacr[(irq / 16) as usize] >> ((irq % 16) * 2)) & 3
    }

    fn raw_nsacr(&self, irq: u32) -> u64 {
        let i = (irq / 16) as usize;
        (u64::from(self.gicd_nsacr[i + 1]) << 32) | u64::from(self.gicd_nsacr[i])
    }

    /// `mask_group_and_nsacr()`: the bits of the 32 interrupts from `irq` that this access may
    /// see or change.
    fn mask_group_and_nsacr(&self, secure: bool, maskfn: NsacrMask, irq: u32) -> u32 {
        if !self.ns_view(secure) {
            return 0xffff_ffff;
        }
        let mut mask = bmp_word(&self.group, irq);
        match maskfn {
            NsacrMask::None => {}
            NsacrMask::Ge1 => {
                let raw = self.raw_nsacr(irq);
                mask |= half_unshuffle64((raw >> 1) | raw);
            }
            NsacrMask::Ge2 => mask |= half_unshuffle64(self.raw_nsacr(irq) >> 1),
        }
        mask
    }

    fn bitmap(&self, which: Bitmap) -> &[u32; BMP_WORDS] {
        match which {
            Bitmap::Enabled => &self.enabled,
            Bitmap::Pending => &self.pending,
            Bitmap::Active => &self.active,
        }
    }

    fn bitmap_mut(&mut self, which: Bitmap) -> &mut [u32; BMP_WORDS] {
        match which {
            Bitmap::Enabled => &mut self.enabled,
            Bitmap::Pending => &mut self.pending,
            Bitmap::Active => &mut self.active,
        }
    }

    /// `gicd_read_bitmap_reg()`.
    fn gicd_read_bitmap_reg(
        &self,
        secure: bool,
        which: Bitmap,
        maskfn: NsacrMask,
        offset: u64,
    ) -> u32 {
        let irq = bitmap_irq(offset);
        if irq < GIC_INTERNAL || irq >= self.num_irq {
            return 0;
        }
        let mut val = bmp_word(self.bitmap(which), irq);
        if let Bitmap::Pending = which {
            // A level triggered interrupt is pending while its line is high.
            val |= !bmp_word(&self.edge_trigger, irq) & bmp_word(&self.level, irq);
        }
        val & self.mask_group_and_nsacr(secure, maskfn, irq)
    }

    /// `gicd_write_set_bitmap_reg()` and `gicd_write_clear_bitmap_reg()`.
    fn gicd_write_bitmap_reg(
        &mut self,
        secure: bool,
        which: Bitmap,
        maskfn: NsacrMask,
        offset: u64,
        value: u32,
        set: bool,
    ) {
        let irq = bitmap_irq(offset);
        if irq < GIC_INTERNAL || irq >= self.num_irq {
            return;
        }
        let value = value & self.mask_group_and_nsacr(secure, maskfn, irq);
        let w = bmp_word_mut(self.bitmap_mut(which), irq);
        if set {
            *w |= value;
        } else {
            *w &= !value;
        }
        self.update(irq, 32);
    }

    /// `gicd_read_ipriorityr()`.
    fn gicd_read_ipriorityr(&self, secure: bool, irq: u32) -> u32 {
        if irq < GIC_INTERNAL || irq >= self.num_irq {
            return 0;
        }
        let prio = u32::from(self.gicd_ipriority[irq as usize]);
        if self.ns_view(secure) {
            // Group 0 and Secure Group 1 fields are hidden from Non-secure.
            if !bmp_test(&self.group, irq) {
                return 0;
            }
            return (prio << 1) & 0xff;
        }
        prio
    }

    /// `gicd_write_ipriorityr()`.
    fn gicd_write_ipriorityr(&mut self, secure: bool, irq: u32, value: u8) {
        if irq < GIC_INTERNAL || irq >= self.num_irq {
            return;
        }
        let mut value = value;
        if self.ns_view(secure) {
            if !bmp_test(&self.group, irq) {
                return;
            }
            // Non-secure only reaches the lower half of the priority range.
            value = 0x80 | (value >> 1);
        }
        self.gicd_ipriority[irq as usize] = value;
    }

    /// `gicd_read_irouter()`.
    fn gicd_read_irouter(&self, secure: bool, irq: u32) -> u64 {
        if irq < GIC_INTERNAL || irq >= self.num_irq {
            return 0;
        }
        if self.ns_view(secure) && !bmp_test(&self.group, irq) && self.gicd_ns_access(irq) != 3 {
            return 0;
        }
        self.gicd_irouter[irq as usize]
    }

    /// `gicd_write_irouter()`.
    fn gicd_write_irouter(&mut self, secure: bool, irq: u32, val: u64) {
        if irq < GIC_INTERNAL || irq >= self.num_irq {
            return;
        }
        if self.ns_view(secure) && !bmp_test(&self.group, irq) && self.gicd_ns_access(irq) != 3 {
            return;
        }
        self.gicd_irouter[irq as usize] = val;
        self.cache_target_cpustate(irq);
        self.update(irq, 1);
    }

    /// `gicv3_dist_read()`. Anything unhandled reads as zero.
    pub(super) fn dist_read(&self, secure: bool, offset: u64, size: u32) -> u64 {
        let r = match size {
            1 => self.gicd_readb(secure, offset),
            4 => self.gicd_readl(secure, offset).map(u64::from),
            8 => self.gicd_readq(secure, offset),
            _ => None,
        };
        r.unwrap_or(0)
    }

    /// `gicv3_dist_write()`. Anything unhandled is ignored.
    pub(super) fn dist_write(&mut self, secure: bool, offset: u64, size: u32, value: u64) {
        match size {
            1 => self.gicd_writeb(secure, offset, value),
            4 => self.gicd_writel(secure, offset, value),
            8 => self.gicd_writeq(secure, offset, value),
            _ => {}
        }
    }

    fn gicd_readb(&self, secure: bool, offset: u64) -> Option<u64> {
        match offset {
            // Affinity routing is always on, so the GICv2 style registers are RAZ/WI.
            GICD_CPENDSGIR..=0x0f1f | GICD_SPENDSGIR..=0x0f2f | GICD_ITARGETSR..=0x0bff => Some(0),
            GICD_IPRIORITYR..=0x07ff => Some(u64::from(
                self.gicd_read_ipriorityr(secure, (offset - GICD_IPRIORITYR) as u32),
            )),
            _ => None,
        }
    }

    fn gicd_writeb(&mut self, secure: bool, offset: u64, value: u64) {
        if let GICD_IPRIORITYR..=0x07ff = offset {
            let irq = (offset - GICD_IPRIORITYR) as u32;
            if irq < GIC_INTERNAL || irq >= self.num_irq {
                return;
            }
            self.gicd_write_ipriorityr(secure, irq, value as u8);
            self.update(irq, 1);
        }
    }

    fn gicd_readq(&self, secure: bool, offset: u64) -> Option<u64> {
        match offset {
            GICD_IROUTER..=0x7fdf => {
                Some(self.gicd_read_irouter(secure, ((offset - GICD_IROUTER) / 8) as u32))
            }
            _ => None,
        }
    }

    fn gicd_writeq(&mut self, secure: bool, offset: u64, value: u64) {
        if let GICD_IROUTER..=0x7fdf = offset {
            self.gicd_write_irouter(secure, ((offset - GICD_IROUTER) / 8) as u32, value);
        }
    }

    fn gicd_readl(&self, secure: bool, offset: u64) -> Option<u32> {
        let v = match offset {
            GICD_CTLR => {
                if self.ns_view(secure) {
                    // The Non-secure view only has the aliases of a few Secure bits.
                    self.gicd_ctlr & (GICD_CTLR_ARE_S | GICD_CTLR_EN_GRP1NS | GICD_CTLR_RWP)
                } else {
                    self.gicd_ctlr
                }
            }
            GICD_TYPER => {
                // No1N, A3V, IDbits 0xf, LPIS, SecurityExtn and ITLinesNumber. NMI is 0.
                let itlinesnumber = self.num_irq / 32 - 1;
                let sec_extn = u32::from(!self.ds());
                let dvis = u32::from(self.revision >= 4);
                (1 << 25)
                    | (1 << 24)
                    | (dvis << 18)
                    | (u32::from(self.lpi_enable) << 17)
                    | (sec_extn << 10)
                    | (0xf << 19)
                    | itlinesnumber
            }
            GICD_IIDR => GICV3_IIDR,
            GICD_TYPER2 | GICD_STATUSR => 0,
            GICD_IGROUPR..=0x00ff => {
                if self.ns_view(secure) {
                    return Some(0);
                }
                let irq = bitmap_irq(offset - GICD_IGROUPR);
                if irq < GIC_INTERNAL || irq >= self.num_irq {
                    return Some(0);
                }
                bmp_word(&self.group, irq)
            }
            GICD_ISENABLER..=0x017f => self.gicd_read_bitmap_reg(
                secure,
                Bitmap::Enabled,
                NsacrMask::None,
                offset - GICD_ISENABLER,
            ),
            GICD_ICENABLER..=0x01ff => self.gicd_read_bitmap_reg(
                secure,
                Bitmap::Enabled,
                NsacrMask::None,
                offset - GICD_ICENABLER,
            ),
            GICD_ISPENDR..=0x027f => self.gicd_read_bitmap_reg(
                secure,
                Bitmap::Pending,
                NsacrMask::Ge1,
                offset - GICD_ISPENDR,
            ),
            GICD_ICPENDR..=0x02ff => self.gicd_read_bitmap_reg(
                secure,
                Bitmap::Pending,
                NsacrMask::Ge2,
                offset - GICD_ICPENDR,
            ),
            GICD_ISACTIVER..=0x037f => self.gicd_read_bitmap_reg(
                secure,
                Bitmap::Active,
                NsacrMask::Ge2,
                offset - GICD_ISACTIVER,
            ),
            GICD_ICACTIVER..=0x03ff => self.gicd_read_bitmap_reg(
                secure,
                Bitmap::Active,
                NsacrMask::Ge2,
                offset - GICD_ICACTIVER,
            ),
            GICD_IPRIORITYR..=0x07ff => {
                let irq = (offset - GICD_IPRIORITYR) as u32;
                let mut value = 0;
                for i in (irq..irq + 4).rev() {
                    value = (value << 8) | self.gicd_read_ipriorityr(secure, i);
                }
                value
            }
            GICD_ITARGETSR..=0x0bff => 0,
            GICD_ICFGR..=0x0cff => {
                // Two bits per interrupt, of which only the odd one is used.
                let irq = ((offset - GICD_ICFGR) * 4) as u32;
                if irq < GIC_INTERNAL || irq >= self.num_irq {
                    return Some(0);
                }
                let base = irq & !0x1f;
                let mut value = bmp_word(&self.edge_trigger, base);
                value &= self.mask_group_and_nsacr(secure, NsacrMask::None, base);
                value = if irq & 0x1f != 0 { value >> 16 } else { value & 0xffff };
                half_shuffle32(value) << 1
            }
            GICD_IGRPMODR..=0x0dff => {
                if self.ds() || !secure {
                    return Some(0);
                }
                let irq = bitmap_irq(offset - GICD_IGRPMODR);
                if irq < GIC_INTERNAL || irq >= self.num_irq {
                    return Some(0);
                }
                bmp_word(&self.grpmod, irq)
            }
            GICD_NSACR..=0x0eff => {
                let irq = ((offset - GICD_NSACR) * 4) as u32;
                if irq < GIC_INTERNAL || irq >= self.num_irq {
                    return Some(0);
                }
                if self.ds() || !secure {
                    return Some(0);
                }
                self.gicd_nsacr[(irq / 16) as usize]
            }
            GICD_CPENDSGIR..=0x0f1f | GICD_SPENDSGIR..=0x0f2f => 0,
            GICD_INMIR..=0x0fff => 0,
            GICD_IROUTER..=0x7fdf => {
                let r = self.gicd_read_irouter(secure, ((offset - GICD_IROUTER) / 8) as u32);
                if offset & 7 != 0 { (r >> 32) as u32 } else { r as u32 }
            }
            GICD_IDREGS..=0xffff => self.idreg(offset - GICD_IDREGS, GICV3_PIDR0_DIST),
            GICD_SGIR => 0,
            _ => return None,
        };
        Some(v)
    }

    fn gicd_writel(&mut self, secure: bool, offset: u64, value: u64) {
        let v32 = value as u32;
        match offset {
            GICD_CTLR => {
                let mask = if self.ds() {
                    // One security state: only the two group enables are writable.
                    GICD_CTLR_EN_GRP0 | GICD_CTLR_EN_GRP1NS
                } else if secure {
                    GICD_CTLR_DS | GICD_CTLR_EN_GRP0 | GICD_CTLR_EN_GRP1_ALL
                } else {
                    GICD_CTLR_EN_GRP1NS
                };
                self.gicd_ctlr = (self.gicd_ctlr & !mask) | (v32 & mask);
                if v32 & mask & GICD_CTLR_DS != 0 {
                    // Setting DS makes ARE_NS and EnableGrp1S RES0. There is no way back short
                    // of a reset, since DS is not writable once set.
                    self.gicd_ctlr &= !(GICD_CTLR_EN_GRP1S | GICD_CTLR_ARE_NS);
                }
                self.full_update();
            }
            GICD_IGROUPR..=0x00ff => {
                if self.ns_view(secure) {
                    return;
                }
                let irq = bitmap_irq(offset - GICD_IGROUPR);
                if irq < GIC_INTERNAL || irq >= self.num_irq {
                    return;
                }
                *bmp_word_mut(&mut self.group, irq) = v32;
                self.update(irq, 32);
            }
            GICD_ISENABLER..=0x017f => self.gicd_write_bitmap_reg(
                secure,
                Bitmap::Enabled,
                NsacrMask::None,
                offset - GICD_ISENABLER,
                v32,
                true,
            ),
            GICD_ICENABLER..=0x01ff => self.gicd_write_bitmap_reg(
                secure,
                Bitmap::Enabled,
                NsacrMask::None,
                offset - GICD_ICENABLER,
                v32,
                false,
            ),
            GICD_ISPENDR..=0x027f => self.gicd_write_bitmap_reg(
                secure,
                Bitmap::Pending,
                NsacrMask::Ge1,
                offset - GICD_ISPENDR,
                v32,
                true,
            ),
            GICD_ICPENDR..=0x02ff => self.gicd_write_bitmap_reg(
                secure,
                Bitmap::Pending,
                NsacrMask::Ge2,
                offset - GICD_ICPENDR,
                v32,
                false,
            ),
            GICD_ISACTIVER..=0x037f => self.gicd_write_bitmap_reg(
                secure,
                Bitmap::Active,
                NsacrMask::None,
                offset - GICD_ISACTIVER,
                v32,
                true,
            ),
            GICD_ICACTIVER..=0x03ff => self.gicd_write_bitmap_reg(
                secure,
                Bitmap::Active,
                NsacrMask::None,
                offset - GICD_ICACTIVER,
                v32,
                false,
            ),
            GICD_IPRIORITYR..=0x07ff => {
                let irq = (offset - GICD_IPRIORITYR) as u32;
                if irq < GIC_INTERNAL || irq + 3 >= self.num_irq {
                    return;
                }
                for (n, i) in (irq..irq + 4).enumerate() {
                    self.gicd_write_ipriorityr(secure, i, (v32 >> (n * 8)) as u8);
                }
                self.update(irq, 4);
            }
            GICD_ICFGR..=0x0cff => {
                let irq = ((offset - GICD_ICFGR) * 4) as u32;
                if irq < GIC_INTERNAL || irq >= self.num_irq {
                    return;
                }
                // The 32 bits written squeeze into 16 bits of the one bit per interrupt map.
                let base = irq & !0x1f;
                let mut value = half_unshuffle32(v32 >> 1);
                let mut mask = self.mask_group_and_nsacr(secure, NsacrMask::None, base);
                if irq & 0x1f != 0 {
                    value <<= 16;
                    mask &= 0xffff_0000;
                } else {
                    mask &= 0xffff;
                }
                let w = bmp_word_mut(&mut self.edge_trigger, base);
                *w = (*w & !mask) | (value & mask);
            }
            GICD_IGRPMODR..=0x0dff => {
                if self.ds() || !secure {
                    return;
                }
                let irq = bitmap_irq(offset - GICD_IGRPMODR);
                if irq < GIC_INTERNAL || irq >= self.num_irq {
                    return;
                }
                *bmp_word_mut(&mut self.grpmod, irq) = v32;
                self.update(irq, 32);
            }
            GICD_NSACR..=0x0eff => {
                let irq = ((offset - GICD_NSACR) * 4) as u32;
                if irq < GIC_INTERNAL || irq >= self.num_irq {
                    return;
                }
                if self.ds() || !secure {
                    return;
                }
                // Only access checks read this, so nothing needs recomputing.
                self.gicd_nsacr[(irq / 16) as usize] = v32;
            }
            GICD_IROUTER..=0x7fdf => {
                let irq = ((offset - GICD_IROUTER) / 8) as u32;
                let r = self.gicd_read_irouter(secure, irq);
                let r = deposit_half(r, offset & 7 != 0, value);
                self.gicd_write_irouter(secure, irq, r);
            }
            // GICD_STATUSR, GICD_ITARGETSR, GICD_SGIR, GICD_CPENDSGIR, GICD_SPENDSGIR and
            // GICD_INMIR ignore writes here, and so do the read only registers.
            _ => {}
        }
    }
}

/// The set/clear bitmaps reachable through the distributor.
#[derive(Clone, Copy)]
enum Bitmap {
    Enabled,
    Pending,
    Active,
}
