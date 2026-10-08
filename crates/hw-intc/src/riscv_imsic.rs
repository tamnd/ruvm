// SPDX-License-Identifier: GPL-2.0-or-later

//! The RISC-V incoming message signaled interrupt controller, hw/intc/riscv_imsic.c and
//! include/hw/intc/riscv_imsic.h.
//!
//! [`RiscvImsic`] is `RISCVIMSICState`: the interrupt files of one hart at one privilege level.
//! An M level IMSIC has one file. An S level IMSIC has the S file (page 0) and one file for each
//! guest (pages 1 to `num_pages - 1`). Every file has `eidelivery`, `eithreshold` and a pending
//! and an enabled bit for each of its `num_irqs - 1` identities (identity 0 does not exist).
//!
//! A device signals identity `n` of a file by writing `n` as a little endian word at offset 0
//! of the file's 4 KiB page ([`RiscvImsic::reg_write`]). The hart reaches the registers through
//! its `*iselect` and `*ireg` CSRs and reads and claims the best identity through `*topei`; the
//! CPU passes those accesses to [`RiscvImsic::rmw`] with the `AIA_IREG` encoding of
//! [`aia_make_ireg`].
//!
//! Output `page` is high while the file delivers interrupts and has an identity that is
//! pending, enabled and below the threshold. QEMU connects output 0 to the hart's `IRQ_M_EXT`
//! (M level) or `IRQ_S_EXT` (S level) input and output `i` of a guest file to the hart's guest
//! external interrupt line `IRQ_LOCAL_MAX + i - 1`; the board does the same.
//!
//! Differences from QEMU:
//!
//! - Not ported: VMState, QOM properties and registration, and the KVM side.
//! - The `qemu_log_mask()` guest error messages are not printed.
//! - QEMU keeps `eistate` in atomics. Here one mutex covers the state and the outputs are driven
//!   with it held, so the lines connected to the outputs must not call back into the IMSIC.
//! - Realize claims `MIP_MEIP` or `MIP_SEIP` on the hart, forces `ext_smaia` or `ext_ssaia` on,
//!   registers [`RiscvImsic::rmw`] as the hart's `aia_ireg_rmw_cb` for its privilege level
//!   and, for an S level IMSIC, sets the hart's GEILEN to `num_pages - 1`. All of that is CPU
//!   state the board owns, so the board must do it.
//! - QEMU accepts accesses one byte past the end of the region in its range check; the region
//!   is only `num_pages` pages long, so such accesses never reach it in either.

use std::fmt;
use std::sync::{Mutex, MutexGuard, PoisonError};

use ruvm_hw_core::irq::IrqPin;
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, Endian, MemResult, MmioOps};

/// `TYPE_RISCV_IMSIC`.
pub const TYPE_RISCV_IMSIC: &str = "riscv.imsic";

/// `IMSIC_MMIO_PAGE_SHIFT`.
pub const IMSIC_MMIO_PAGE_SHIFT: u32 = 12;
/// `IMSIC_MMIO_PAGE_SZ`.
pub const IMSIC_MMIO_PAGE_SZ: u64 = 1 << IMSIC_MMIO_PAGE_SHIFT;
/// `IMSIC_MMIO_HART_GUEST_MAX_BTIS`.
pub const IMSIC_MMIO_HART_GUEST_MAX_BITS: u32 = 6;
/// `IMSIC_MMIO_GROUP_MIN_SHIFT`.
pub const IMSIC_MMIO_GROUP_MIN_SHIFT: u32 = 24;

/// `IMSIC_MMIO_SIZE()`: the size of the region of an IMSIC with `num_pages` files.
pub const fn imsic_mmio_size(num_pages: u32) -> u64 {
    num_pages as u64 * IMSIC_MMIO_PAGE_SZ
}

/// `IMSIC_HART_SIZE()`: the room one hart takes with `guest_bits` bits of guest index.
pub const fn imsic_hart_size(guest_bits: u32) -> u64 {
    (1u64 << guest_bits) * IMSIC_MMIO_PAGE_SZ
}

/// `ISELECT_IMSIC_EIDELIVERY`.
pub const ISELECT_IMSIC_EIDELIVERY: u32 = 0x70;
/// `ISELECT_IMSIC_EITHRESHOLD`.
pub const ISELECT_IMSIC_EITHRESHOLD: u32 = 0x72;
/// `ISELECT_IMSIC_EIP0`.
pub const ISELECT_IMSIC_EIP0: u32 = 0x80;
/// `ISELECT_IMSIC_EIP63`.
pub const ISELECT_IMSIC_EIP63: u32 = 0xbf;
/// `ISELECT_IMSIC_EIE0`.
pub const ISELECT_IMSIC_EIE0: u32 = 0xc0;
/// `ISELECT_IMSIC_EIE63`.
pub const ISELECT_IMSIC_EIE63: u32 = 0xff;
/// `ISELECT_IMSIC_TOPEI`: the pseudo register number the CPU uses for `*topei`.
pub const ISELECT_IMSIC_TOPEI: u32 = 0x200;

/// `IMSIC_TOPEI_IID_SHIFT`.
pub const IMSIC_TOPEI_IID_SHIFT: u32 = 16;
/// `IMSIC_TOPEI_IID_MASK`.
pub const IMSIC_TOPEI_IID_MASK: u32 = 0x7ff;
/// `IMSIC_TOPEI_IPRIO_MASK`.
pub const IMSIC_TOPEI_IPRIO_MASK: u32 = 0x7ff;

/// `PRV_M`.
const PRV_M: u32 = 3;
/// `PRV_S`.
const PRV_S: u32 = 1;

/// `IMSIC_MAX_ID`.
const IMSIC_MAX_ID: u64 = IMSIC_TOPEI_IID_MASK as u64;
/// `IMSIC_MMIO_PAGE_LE`.
const IMSIC_MMIO_PAGE_LE: u64 = 0x00;
/// `IMSIC_EISTATE_PENDING`.
const EISTATE_PENDING: u32 = 1 << 0;
/// `IMSIC_EISTATE_ENABLED`.
const EISTATE_ENABLED: u32 = 1 << 1;
/// `IMSIC_EISTATE_ENPEND`.
const EISTATE_ENPEND: u32 = EISTATE_ENABLED | EISTATE_PENDING;

/// The error of [`RiscvImsic::rmw`]: the register does not exist at that level, QEMU's
/// `-EINVAL`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidRegister;

/// `AIA_MAKE_IREG()`: the register argument of `aia_ireg_rmw_fn`.
pub const fn aia_make_ireg(isel: u32, priv_lvl: u32, virt: bool, vgein: u32, xlen: u32) -> u32 {
    (isel & 0xffff)
        | ((priv_lvl & 0x3) << 16)
        | ((virt as u32) << 18)
        | ((vgein & 0x3f) << 20)
        | ((xlen & 0xff) << 24)
}

/// `AIA_IREG_ISEL()`.
const fn ireg_isel(r: u32) -> u32 {
    r & 0xffff
}

/// `AIA_IREG_PRIV()`.
const fn ireg_priv(r: u32) -> u32 {
    (r >> 16) & 0x3
}

/// `AIA_IREG_VIRT()`.
const fn ireg_virt(r: u32) -> bool {
    (r >> 18) & 0x1 != 0
}

/// `AIA_IREG_VGEIN()`.
const fn ireg_vgein(r: u32) -> u32 {
    (r >> 20) & 0x3f
}

/// `AIA_IREG_XLEN()`.
const fn ireg_xlen(r: u32) -> u32 {
    (r >> 24) & 0xff
}

/// The register state of `RISCVIMSICState`.
#[derive(Debug)]
struct ImsicState {
    eidelivery: Vec<u32>,
    eithreshold: Vec<u32>,
    /// `num_pages * num_irqs` entries. Entry 0 of each page is not an identity: QEMU uses it
    /// to remember whether the page's output is raised.
    eistate: Vec<u32>,
}

/// `RISCVIMSICState`, the `riscv.imsic` device.
pub struct RiscvImsic {
    mmode: bool,
    hartid: u32,
    num_pages: u32,
    num_irqs: u32,
    state: Mutex<ImsicState>,
    external_irqs: Vec<IrqPin>,
}

impl fmt::Debug for RiscvImsic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RiscvImsic")
            .field("mmode", &self.mmode)
            .field("hartid", &self.hartid)
            .field("num_pages", &self.num_pages)
            .field("num_irqs", &self.num_irqs)
            .finish_non_exhaustive()
    }
}

impl RiscvImsic {
    /// `riscv_imsic_create()` without the CPU side (see the module documentation): the IMSIC
    /// of hart `hartid`, at M level if `mmode`, with `num_pages` files of `num_ids` identities.
    /// The outputs are disconnected.
    ///
    /// # Panics
    ///
    /// On the parameters `riscv_imsic_create()` asserts against: no pages, more pages than an
    /// M level or S level IMSIC can have, or `num_ids` not one less than a multiple of 64
    /// between 63 and 2047.
    pub fn new(hartid: u32, mmode: bool, num_pages: u32, num_ids: u32) -> RiscvImsic {
        assert!(!(mmode && num_pages > 1), "an M level IMSIC has one page");
        assert!((1..=1 << IMSIC_MMIO_HART_GUEST_MAX_BITS).contains(&num_pages));
        // IMSIC_MIN_ID and IMSIC_MAX_ID.
        assert!((63..=IMSIC_TOPEI_IID_MASK).contains(&num_ids) && num_ids & 63 == 63);
        let num_irqs = num_ids + 1;
        let n = num_pages as usize;
        RiscvImsic {
            mmode,
            hartid,
            num_pages,
            num_irqs,
            state: Mutex::new(ImsicState {
                eidelivery: vec![0; n],
                eithreshold: vec![0; n],
                eistate: vec![0; n * num_irqs as usize],
            }),
            external_irqs: (0..num_pages).map(|_| IrqPin::new()).collect(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, ImsicState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Whether this is an M level IMSIC.
    pub fn mmode(&self) -> bool {
        self.mmode
    }

    /// The hart the IMSIC belongs to.
    pub fn hartid(&self) -> u32 {
        self.hartid
    }

    /// The number of interrupt files.
    pub fn num_pages(&self) -> u32 {
        self.num_pages
    }

    /// The size of the MMIO region.
    pub fn mmio_size(&self) -> u64 {
        imsic_mmio_size(self.num_pages)
    }

    /// The output of file `page`, gpio out `page`.
    pub fn external_irq(&self, page: usize) -> &IrqPin {
        &self.external_irqs[page]
    }

    /// `riscv_imsic_topei()`.
    fn topei(&self, s: &ImsicState, page: u32) -> u32 {
        let base = (page * self.num_irqs) as usize;
        let th = s.eithreshold[page as usize];
        let max_irq = if th != 0 && th <= self.num_irqs { th } else { self.num_irqs };
        for i in 1..max_irq {
            if s.eistate[base + i as usize] & EISTATE_ENPEND == EISTATE_ENPEND {
                return (i << IMSIC_TOPEI_IID_SHIFT) | i;
            }
        }
        0
    }

    /// `riscv_imsic_update()`: lowers the output if it was raised, then raises it again if the
    /// file still has something to deliver.
    fn update(&self, s: &mut ImsicState, page: u32) {
        let base = (page * self.num_irqs) as usize;
        let was = s.eistate[base];
        s.eistate[base] &= !EISTATE_ENPEND;
        if was != 0 {
            self.external_irqs[page as usize].lower();
        }
        if s.eidelivery[page as usize] != 0 && self.topei(s, page) != 0 {
            self.external_irqs[page as usize].raise();
            s.eistate[base] |= EISTATE_ENPEND;
        }
    }

    /// `riscv_imsic_eidelivery_rmw()` and `riscv_imsic_eithreshold_rmw()`.
    fn reg_rmw(&self, s: &mut ImsicState, page: u32, thresh: bool, new: u64, mask: u64) -> u64 {
        let p = page as usize;
        let (reg, mask) = if thresh {
            (&mut s.eithreshold[p], mask & IMSIC_MAX_ID)
        } else {
            (&mut s.eidelivery[p], mask & 1)
        };
        let old = *reg;
        *reg = ((u64::from(old) & !mask) | (new & mask)) as u32;
        self.update(s, page);
        u64::from(old)
    }

    /// `riscv_imsic_topei_rmw()`: a write, whatever its value, clears the pending bit of the
    /// identity read.
    fn topei_rmw(&self, s: &mut ImsicState, page: u32, wr_mask: u64) -> u64 {
        let topei = self.topei(s, page);
        if topei != 0 && wr_mask != 0 {
            let id = topei >> IMSIC_TOPEI_IID_SHIFT;
            if id != 0 {
                s.eistate[(page * self.num_irqs + id) as usize] &= !EISTATE_PENDING;
            }
        }
        self.update(s, page);
        u64::from(topei)
    }

    /// `riscv_imsic_eix_rmw()`: `eip<num>` if `pend`, else `eie<num>`.
    #[allow(clippy::too_many_arguments)]
    fn eix_rmw(
        &self,
        s: &mut ImsicState,
        xlen: u32,
        page: u32,
        mut num: u32,
        pend: bool,
        new: u64,
        wr_mask: u64,
    ) -> Result<u64, InvalidRegister> {
        let state = if pend { EISTATE_PENDING } else { EISTATE_ENABLED };
        if xlen != 32 {
            if num & 1 != 0 {
                return Err(InvalidRegister);
            }
            num >>= 1;
        }
        // An xlen of 0 is not something a CPU passes; QEMU would divide by zero.
        if xlen == 0 || num >= self.num_irqs / xlen {
            return Err(InvalidRegister);
        }
        let base = (page * self.num_irqs + num * xlen) as usize;
        let mut val = 0;
        for i in 0..xlen.min(64) {
            // Bit 0 of eip0 and eie0 is read only zero.
            if num == 0 && i == 0 {
                continue;
            }
            let m = 1u64 << i;
            let e = &mut s.eistate[base + i as usize];
            let prev = *e;
            if wr_mask & m != 0 {
                if new & m != 0 {
                    *e |= state;
                } else {
                    *e &= !state;
                }
            }
            if prev & state != 0 {
                val |= m;
            }
        }
        self.update(s, page);
        Ok(val)
    }

    /// `riscv_imsic_rmw()`, the hart's `aia_ireg_rmw_cb`: reads register `reg` (an
    /// [`aia_make_ireg`] value) and writes the bits of `wr_mask` from `new`.
    pub fn rmw(&self, reg: u32, new: u64, wr_mask: u64) -> Result<u64, InvalidRegister> {
        let (isel, priv_lvl, virt) = (ireg_isel(reg), ireg_priv(reg), ireg_virt(reg));
        let (vgein, xlen) = (ireg_vgein(reg), ireg_xlen(reg));
        let page = if self.mmode {
            if priv_lvl != PRV_M || virt {
                return Err(InvalidRegister);
            }
            0
        } else {
            if priv_lvl != PRV_S {
                return Err(InvalidRegister);
            }
            if !virt {
                0
            } else if vgein != 0 && vgein < self.num_pages {
                vgein
            } else {
                return Err(InvalidRegister);
            }
        };
        let mut s = self.lock();
        match isel {
            ISELECT_IMSIC_EIDELIVERY => Ok(self.reg_rmw(&mut s, page, false, new, wr_mask)),
            ISELECT_IMSIC_EITHRESHOLD => Ok(self.reg_rmw(&mut s, page, true, new, wr_mask)),
            ISELECT_IMSIC_TOPEI => Ok(self.topei_rmw(&mut s, page, wr_mask)),
            ISELECT_IMSIC_EIP0..=ISELECT_IMSIC_EIP63 => {
                self.eix_rmw(&mut s, xlen, page, isel - ISELECT_IMSIC_EIP0, true, new, wr_mask)
            }
            ISELECT_IMSIC_EIE0..=ISELECT_IMSIC_EIE63 => {
                self.eix_rmw(&mut s, xlen, page, isel - ISELECT_IMSIC_EIE0, false, new, wr_mask)
            }
            // QEMU logs "riscv_imsic_rmw: Invalid register priv=%d virt=%d isel=%d vgein=%d".
            _ => Err(InvalidRegister),
        }
    }

    /// `riscv_imsic_write()`: a write of identity `value` to offset 0 of a file's page sets the
    /// identity pending. Other writes are ignored.
    pub fn reg_write(&self, addr: u64, value: u64) {
        if addr & 3 != 0 || addr > self.mmio_size() {
            // QEMU logs "riscv_imsic_write: Invalid register write 0x%x".
            return;
        }
        let page = (addr >> IMSIC_MMIO_PAGE_SHIFT) as u32;
        if addr & (IMSIC_MMIO_PAGE_SZ - 1) == IMSIC_MMIO_PAGE_LE
            && value != 0
            && value < u64::from(self.num_irqs)
            && page < self.num_pages
        {
            let mut s = self.lock();
            s.eistate[(page * self.num_irqs) as usize + value as usize] |= EISTATE_PENDING;
            self.update(&mut s, page);
        }
    }

    /// `riscv_imsic_reset_enter()`: clears `eidelivery`, `eithreshold` and the enabled bits,
    /// but not the pending bits, and lowers every output.
    pub fn reset(&self) {
        let mut s = self.lock();
        s.eidelivery.fill(0);
        s.eithreshold.fill(0);
        for e in &mut s.eistate {
            *e &= !EISTATE_ENABLED;
        }
        for pin in &self.external_irqs {
            pin.lower();
        }
    }

    /// Whether identity `id` of file `page` is pending.
    pub fn is_pending(&self, page: u32, id: u32) -> bool {
        self.lock().eistate[(page * self.num_irqs + id) as usize] & EISTATE_PENDING != 0
    }
}

/// `riscv_imsic_ops`: reads give zero.
impl MmioOps for RiscvImsic {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(0)
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.reg_write(offset, value);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::exact(4)
    }

    fn endianness(&self) -> Endian {
        Endian::Little
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvm_hw_core::irq::IrqLine;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicI32, Ordering};

    fn watch(pin: &IrqPin) -> Arc<AtomicI32> {
        let level = Arc::new(AtomicI32::new(-1));
        let l = level.clone();
        pin.connect(IrqLine::from_fn(move |v| l.store(v, Ordering::SeqCst)));
        level
    }

    fn s_reg(isel: u32) -> u32 {
        aia_make_ireg(isel, PRV_S, false, 0, 64)
    }

    #[test]
    fn ireg_encoding() {
        let r = aia_make_ireg(0x1ff, 3, true, 5, 64);
        assert_eq!(r, 0x1ff | 3 << 16 | 1 << 18 | 5 << 20 | 64 << 24);
        assert_eq!((ireg_isel(r), ireg_priv(r), ireg_virt(r)), (0x1ff, 3, true));
        assert_eq!((ireg_vgein(r), ireg_xlen(r)), (5, 64));
        assert_eq!(imsic_hart_size(2), 0x4000);
        assert_eq!(imsic_mmio_size(4), 0x4000);
    }

    #[test]
    fn delivery_threshold_and_claim() {
        let imsic = RiscvImsic::new(0, false, 1, 255);
        let out = watch(imsic.external_irq(0));

        // A message to an identity that is not enabled only sets it pending.
        imsic.reg_write(0, 5);
        assert!(imsic.is_pending(0, 5));
        assert_eq!(imsic.rmw(s_reg(ISELECT_IMSIC_EIP0), 0, 0), Ok(1 << 5));
        assert_eq!(out.load(Ordering::SeqCst), -1);

        // Enable 5 and 9; still nothing without eidelivery.
        imsic.rmw(s_reg(ISELECT_IMSIC_EIE0), (1 << 5) | (1 << 9) | 1, !0).unwrap();
        assert_eq!(imsic.rmw(s_reg(ISELECT_IMSIC_EIE0), 0, 0), Ok((1 << 5) | (1 << 9)));
        assert_eq!(imsic.rmw(s_reg(ISELECT_IMSIC_TOPEI), 0, 0), Ok(5 << 16 | 5));
        // The output was never raised, so it was never driven.
        assert_eq!(out.load(Ordering::SeqCst), -1);
        assert_eq!(imsic.rmw(s_reg(ISELECT_IMSIC_EIDELIVERY), 3, !0), Ok(0));
        assert_eq!(imsic.rmw(s_reg(ISELECT_IMSIC_EIDELIVERY), 0, 0), Ok(1));
        assert_eq!(out.load(Ordering::SeqCst), 1);

        // A threshold at or below the identity masks it.
        imsic.rmw(s_reg(ISELECT_IMSIC_EITHRESHOLD), 5, !0).unwrap();
        assert_eq!(out.load(Ordering::SeqCst), 0);
        imsic.reg_write(0, 3);
        assert_eq!(out.load(Ordering::SeqCst), 0);
        imsic.rmw(s_reg(ISELECT_IMSIC_EIE0), 1 << 3, 1 << 3).unwrap();
        assert_eq!(out.load(Ordering::SeqCst), 1);
        imsic.rmw(s_reg(ISELECT_IMSIC_EITHRESHOLD), 0, !0).unwrap();

        // Claims go lowest identity first.
        assert_eq!(imsic.rmw(s_reg(ISELECT_IMSIC_TOPEI), 0, !0), Ok(3 << 16 | 3));
        assert_eq!(imsic.rmw(s_reg(ISELECT_IMSIC_TOPEI), 0, !0), Ok(5 << 16 | 5));
        assert_eq!(out.load(Ordering::SeqCst), 0);
        assert_eq!(imsic.rmw(s_reg(ISELECT_IMSIC_TOPEI), 0, !0), Ok(0));

        // Odd eip and eie numbers do not exist with a 64 bit hart, nor do ones past the last
        // identity.
        assert_eq!(imsic.rmw(s_reg(ISELECT_IMSIC_EIP0 + 1), 0, 0), Err(InvalidRegister));
        assert_eq!(imsic.rmw(s_reg(ISELECT_IMSIC_EIP0 + 8), 0, 0), Err(InvalidRegister));
        assert_eq!(imsic.rmw(s_reg(ISELECT_IMSIC_EIP0 + 6), 0, 0), Ok(0));
        assert_eq!(imsic.rmw(s_reg(0x71), 0, 0), Err(InvalidRegister));

        // Reset keeps the pending bits.
        imsic.reg_write(0, 200);
        imsic.reset();
        assert_eq!(out.load(Ordering::SeqCst), 0);
        assert!(imsic.is_pending(0, 200));
        assert_eq!(imsic.rmw(s_reg(ISELECT_IMSIC_EIE0), 0, 0), Ok(0));
    }

    #[test]
    fn guest_files_and_privilege_checks() {
        let imsic = RiscvImsic::new(1, false, 4, 255);
        let g2 = watch(imsic.external_irq(2));
        let g2_reg = |isel| aia_make_ireg(isel, PRV_S, true, 2, 64);
        imsic.rmw(g2_reg(ISELECT_IMSIC_EIDELIVERY), 1, 1).unwrap();
        // eie4 holds identities 128 to 191 of page 2.
        imsic.rmw(g2_reg(ISELECT_IMSIC_EIE0 + 4), 1, 1).unwrap();
        imsic.reg_write(2 * 0x1000, 128);
        assert_eq!(g2.load(Ordering::SeqCst), 1);
        assert!(!imsic.is_pending(0, 128));
        assert_eq!(imsic.rmw(g2_reg(ISELECT_IMSIC_TOPEI), 0, 0), Ok(128 << 16 | 128));
        // Messages away from offset 0 are ignored.
        imsic.reg_write(2 * 0x1000 + 4, 7);
        assert!(!imsic.is_pending(2, 7));

        assert_eq!(imsic.rmw(aia_make_ireg(0x70, PRV_S, true, 0, 64), 0, 0), Err(InvalidRegister));
        assert_eq!(imsic.rmw(aia_make_ireg(0x70, PRV_S, true, 4, 64), 0, 0), Err(InvalidRegister));
        assert_eq!(imsic.rmw(aia_make_ireg(0x70, PRV_M, false, 0, 64), 0, 0), Err(InvalidRegister));

        let m = RiscvImsic::new(0, true, 1, 255);
        assert_eq!(m.rmw(aia_make_ireg(0x70, PRV_M, false, 0, 64), 1, 1), Ok(0));
        assert_eq!(m.rmw(aia_make_ireg(0x70, PRV_S, false, 0, 64), 0, 0), Err(InvalidRegister));
        assert_eq!(m.valid(), AccessConstraints::exact(4));
    }
}
