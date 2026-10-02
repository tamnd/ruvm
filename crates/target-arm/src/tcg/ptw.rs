// SPDX-License-Identifier: GPL-2.0-or-later

//! The AArch64 page table walk: the port of `get_phys_addr_lpae()`,
//! `get_phys_addr_disabled()`, `get_phys_addr_twostage()`, `aa64_va_parameters()`,
//! `check_s2_mmu_setup()`, `get_S1prot()` and `get_S2prot()` from QEMU's
//! `target/arm/ptw.c`, plus `arm_cpu_tlb_fill()` and `arm_deliver_fault()` from
//! `tlb_helper.c`.
//!
//! Every AArch64 translation regime is covered: EL1&0 (with stage 2 when HCR_EL2.VM is in
//! effect), EL2&0 (VHE), EL2 and EL3, with the 4K, 16K and 64K granules the model has.
//!
//! Differences from QEMU:
//!
//! - Output and input addresses are at most 48 bits: FEAT_LPA, FEAT_LPA2 and FEAT_LVA
//!   (52 bit addresses) are not implemented, and neither is FEAT_TTST.
//! - Secure and Non-secure accesses go to the same address space, and the NS bits of the
//!   descriptors are ignored.
//! - Memory attributes are not combined across the two stages; the stage 1 attributes are
//!   reported. Stage 2 has no FEAT_S2FWB and no FEAT_XNX (TTS2UXN).
//! - HCR_EL2.PTW (stage 1 walks to Device memory) never faults, and the stage 1 walk does
//!   not check that the descriptor's stage 2 page is readable beyond the stage 2 walk
//!   itself.
//! - Stage 2 data aborts are reported without instruction syndrome (ISV is clear), as
//!   QEMU does for accesses made by helpers.

use ruvm_jit::{Cpu, CpuLoopExit, MmuAccessType, Ra, page};
use ruvm_mem::{AddressSpace, Endian, MemTxAttrs, MemTxResult};

use super::{Arm, exception_target_el, regime_el, regime_has_2_ranges};
use crate::cpu::{
    ArmFeatures, CpuArmState, EXCP_DATA_ABORT, EXCP_PREFETCH_ABORT, HCR_DC, HCR_TGE, HCR_VM,
    MMU_IDX_E10_0, MMU_IDX_E10_1, MMU_IDX_E10_1_PAN, MMU_IDX_E20_0, MMU_IDX_E20_2_PAN, SCTLR_M,
    SCTLR_WXN, pa_range_bits,
};
use crate::syndrome::{fsc, syn_data_abort_no_iss, syn_insn_abort};

/// A translation fault: the long descriptor fault status code (with the level folded in),
/// the external abort type, and for stage 2 faults the IPA.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Fault {
    /// The fault status code.
    pub(crate) fsc: u32,
    /// The EA bit.
    pub(crate) ea: bool,
    /// A stage 2 fault, taken to EL2 (`fi->stage2`).
    pub(crate) stage2: bool,
    /// A stage 2 fault on a stage 1 table walk (`fi->s1ptw`).
    pub(crate) s1ptw: bool,
    /// The faulting IPA of a stage 2 fault (`fi->s2addr`).
    pub(crate) s2addr: u64,
}

impl Fault {
    pub(crate) fn new(fsc: u32) -> Fault {
        Fault { fsc, ea: false, stage2: false, s1ptw: false, s2addr: 0 }
    }
}

/// A successful translation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Translation {
    /// The physical address (the IPA for a stage 1 only walk).
    pub(crate) pa: u64,
    /// `PAGE_READ`, `PAGE_WRITE` and `PAGE_EXEC`.
    pub(crate) prot: u32,
    /// The MAIR attribute byte.
    pub(crate) attrs: u8,
    /// The shareability field.
    pub(crate) sh: u8,
    /// The size of the page or block that maps the address.
    pub(crate) page_size: u64,
}

fn access_bit(access: MmuAccessType) -> u32 {
    match access {
        MmuAccessType::DataLoad => page::READ,
        MmuAccessType::DataStore => page::WRITE,
        MmuAccessType::InstFetch => page::EXEC,
    }
}

fn sextract64(v: u64, start: u32, len: u32) -> i64 {
    ((v << (64 - start - len)) as i64) >> (64 - len)
}

fn extract64(v: u64, start: u32, len: u32) -> u64 {
    (v >> start) & (u64::MAX >> (64 - len))
}

/// `simple_ap_to_rw_prot_is_user()`.
fn simple_ap_to_rw_prot_is_user(ap: u64, is_user: bool) -> u32 {
    match ap {
        0 if is_user => 0,
        0 => page::READ | page::WRITE,
        1 => page::READ | page::WRITE,
        2 if is_user => 0,
        2 => page::READ,
        _ => page::READ,
    }
}

/// `ARMGranuleSize`, as the page shift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gran {
    G4K = 12,
    G16K = 14,
    G64K = 16,
}

/// `sanitize_gran_size()`: an unsupported granule is treated as one the CPU has.
fn sanitize_gran(f: &ArmFeatures, gran: Option<Gran>) -> Gran {
    match gran {
        Some(Gran::G4K) => return Gran::G4K,
        Some(Gran::G16K) if f.tgran16 => return Gran::G16K,
        Some(Gran::G64K) if f.tgran64 => return Gran::G64K,
        _ => {}
    }
    // If the guest selects a granule size that isn't implemented, the architecture
    // requires that we behave as if it selected one that is (with an IMPLEMENTATION
    // DEFINED choice of which one to pick). We choose to implement the smallest supported
    // granule size.
    Gran::G4K
}

/// `tg0_to_gran_size()`.
fn tg0_gran(tg: u64) -> Option<Gran> {
    match tg {
        0 => Some(Gran::G4K),
        1 => Some(Gran::G64K),
        2 => Some(Gran::G16K),
        _ => None,
    }
}

/// `tg1_to_gran_size()`.
fn tg1_gran(tg: u64) -> Option<Gran> {
    match tg {
        1 => Some(Gran::G16K),
        2 => Some(Gran::G4K),
        3 => Some(Gran::G64K),
        _ => None,
    }
}

/// `ARMVAParameters`, the parts this port uses.
struct VaParams {
    tsz: u32,
    select: u64,
    tbi: bool,
    epd: bool,
    hpd: bool,
    ha: bool,
    hd: bool,
    ps: u32,
    gran: Gran,
}

/// `aa64_va_parameters()` for a stage 1 regime.
fn va_parameters(f: &ArmFeatures, tcr: u64, va: u64, mmu_idx: usize, data: bool) -> VaParams {
    let (select, tsz, epd, hpd, tbi, tbid, gran, ps, ha, hd);
    if !regime_has_2_ranges(mmu_idx) {
        select = 0;
        tsz = extract64(tcr, 0, 6);
        gran = tg0_gran(extract64(tcr, 14, 2));
        hpd = extract64(tcr, 24, 1);
        epd = 0; // For aarch64, this is always 0.
        tbi = extract64(tcr, 20, 1);
        tbid = extract64(tcr, 29, 1);
        ps = extract64(tcr, 16, 3);
        ha = extract64(tcr, 21, 1);
        hd = extract64(tcr, 22, 1);
    } else {
        select = extract64(va, 55, 1);
        if select == 0 {
            tsz = extract64(tcr, 0, 6);
            gran = tg0_gran(extract64(tcr, 14, 2));
            epd = extract64(tcr, 7, 1);
            hpd = extract64(tcr, 41, 1);
        } else {
            tsz = extract64(tcr, 16, 6);
            gran = tg1_gran(extract64(tcr, 30, 2));
            epd = extract64(tcr, 23, 1);
            hpd = extract64(tcr, 42, 1);
        }
        tbi = extract64(tcr, 37 + select as u32, 1);
        tbid = extract64(tcr, 51 + select as u32, 1);
        ps = extract64(tcr, 32, 3);
        ha = extract64(tcr, 39, 1);
        hd = extract64(tcr, 40, 1);
    }
    let tbi = if data { tbi } else { tbi & !tbid };
    let ha = f.hafdbs >= 1 && ha != 0;
    VaParams {
        tsz: tsz as u32,
        select,
        tbi: tbi != 0,
        epd: epd != 0,
        hpd: f.hpds && hpd != 0,
        ha,
        hd: ha && f.hafdbs >= 2 && hd != 0,
        ps: ps as u32,
        gran: sanitize_gran(f, gran),
    }
}

/// Whether a block descriptor is allowed at `level` for the granule (without FEAT_LPA2).
fn block_level_ok(gran: Gran, level: u32) -> bool {
    match gran {
        Gran::G4K => level == 1 || level == 2,
        Gran::G16K | Gran::G64K => level == 2,
    }
}

/// What a lookup needs from the CPU: its state, model and memory.
pub(crate) struct Walker<'a> {
    pub(crate) arm: &'a Arm,
    pub(crate) st: &'a CpuArmState,
    pub(crate) as_: &'a AddressSpace,
}

impl Walker<'_> {
    fn feat(&self) -> &ArmFeatures {
        self.arm.features()
    }

    /// The output size in bits of a regime whose TCR PS (or VTCR PS) field is `ps`.
    fn outputsize(&self, ps: u32) -> u32 {
        let parange = (self.arm.model().id_aa64mmfr0 & 0xf) as u32;
        pa_range_bits(parange.min(ps)).min(48)
    }

    /// Read a descriptor at `pa`.
    fn load_desc(&self, pa: u64, level: u32) -> Result<u64, Fault> {
        let (d, res) = self.as_.load(pa, 8, Endian::Little, MemTxAttrs::default());
        if res.is_ok() {
            Ok(d)
        } else {
            Err(Fault {
                ea: res != MemTxResult::DECODE_ERROR,
                ..Fault::new(fsc::sync_external_on_walk(level))
            })
        }
    }

    /// Whether stage 2 translation applies to `mmu_idx`.
    fn stage2_enabled(&self, mmu_idx: usize) -> bool {
        let hcr = self.st.hcr_el2_eff(self.feat());
        matches!(mmu_idx, MMU_IDX_E10_0 | MMU_IDX_E10_1 | MMU_IDX_E10_1_PAN)
            && hcr & HCR_VM != 0
            && hcr & HCR_TGE == 0
    }

    /// `regime_translation_disabled()` for a stage 1 regime.
    fn s1_disabled(&self, mmu_idx: usize) -> bool {
        let hcr = self.st.hcr_el2_eff(self.feat());
        if matches!(mmu_idx, MMU_IDX_E10_0 | MMU_IDX_E10_1 | MMU_IDX_E10_1_PAN) {
            // TGE means that EL0/1 act as if SCTLR_EL1.M is zero, and HCR.DC means the same.
            if hcr & (HCR_TGE | HCR_DC) != 0 {
                return true;
            }
        }
        self.st.sctlr_el[regime_el(mmu_idx) as usize] & SCTLR_M == 0
    }

    /// `get_phys_addr()`: translate `address` for an access of type `access` in `mmu_idx`.
    /// `stage1_only` stops after stage 1 (AT S1E0* and S1E1*), and `is_at` marks an AT
    /// instruction, which does not update the access flag or the dirty state.
    pub(crate) fn get_phys_addr(
        &self,
        address: u64,
        access: MmuAccessType,
        mmu_idx: usize,
        stage1_only: bool,
        is_at: bool,
    ) -> Result<Translation, Fault> {
        let s2 = !stage1_only && self.stage2_enabled(mmu_idx);
        let s1 = self.stage1(address, access, mmu_idx, s2, is_at)?;
        if !s2 {
            return Ok(s1);
        }
        // get_phys_addr_twostage(): the stage 1 output is an IPA.
        let ipa = s1.pa;
        let t2 = self.stage2(ipa, access, is_at).map_err(|mut f| {
            f.s2addr = ipa;
            f.stage2 = true;
            f
        })?;
        Ok(Translation {
            pa: t2.pa,
            prot: s1.prot & t2.prot,
            attrs: s1.attrs,
            sh: s1.sh,
            page_size: s1.page_size.min(t2.page_size),
        })
    }

    /// The stage 1 walk of `mmu_idx`; when `s2` is set the table addresses are IPAs that
    /// go through stage 2.
    fn stage1(
        &self,
        address: u64,
        access: MmuAccessType,
        mmu_idx: usize,
        s2: bool,
        is_at: bool,
    ) -> Result<Translation, Fault> {
        let st = self.st;
        let model = self.arm.model();
        let feat = self.feat();
        let rel = regime_el(mmu_idx) as usize;
        let tcr = st.tcr_el[rel];
        let data = access != MmuAccessType::InstFetch;
        let param = va_parameters(feat, tcr, address, mmu_idx, data);

        if self.s1_disabled(mmu_idx) {
            // get_phys_addr_disabled().
            let pamax = model.pamax().min(48);
            let addrtop = if param.tbi && regime_has_2_ranges(mmu_idx) { 55 } else { 63 };
            if extract64(address, pamax, addrtop - pamax + 1) != 0 {
                return Err(Fault::new(fsc::address_size(0)));
            }
            return Ok(Translation {
                pa: extract64(address, 0, 52),
                prot: page::READ | page::WRITE | page::EXEC,
                attrs: 0,
                sh: 2,
                page_size: 4096,
            });
        }

        let mut level: u32 = 0;
        // If TxSZ is programmed to a value larger than the maximum, or smaller than the
        // effective minimum, it is IMPLEMENTATION DEFINED whether we behave as if the
        // field were programmed within the valid range, or raise a level 0 Translation
        // fault; like QEMU we raise the fault.
        if !(16..=39).contains(&param.tsz) {
            return Err(Fault::new(fsc::translation(level)));
        }
        let addrsize = if param.tbi { 56 } else { 64 };
        let inputsize = 64 - param.tsz;
        let outputsize = self.outputsize(param.ps);

        if inputsize < addrsize {
            let top_bits = sextract64(address, inputsize, addrsize - inputsize);
            if top_bits.wrapping_neg() as u64 != param.select {
                // The gap between the two regions is a Translation fault.
                return Err(Fault::new(fsc::translation(level)));
            }
        }
        if param.epd {
            // Translation table walk disabled => Translation fault on TLB miss.
            return Err(Fault::new(fsc::translation(level)));
        }

        let stride = param.gran as u32 - 3;
        let ttbr = if param.select == 0 { st.ttbr0_el[rel] } else { st.ttbr1_el[rel] };

        // The starting level depends on the virtual address size (which can be up to 48
        // bits) and the translation granule size.
        level = 4 - (inputsize - 4) / stride;
        let indexmask_grainsize = (1u64 << (stride + 3)) - 1;
        let mut indexmask = (1u64 << (inputsize - stride * (4 - level))) - 1;

        let mut descaddr = extract64(ttbr, 0, 48);
        // If the base address is out of range, raise AddressSizeFault.
        if descaddr >> outputsize != 0 {
            return Err(Fault::new(fsc::address_size(0)));
        }
        // This masking clears the RES0 bits at the bottom of the TTBR and CnP.
        descaddr &= !indexmask;
        let descaddrmask = ((1u64 << 48) - 1) & !indexmask_grainsize;
        let mut tableattrs: u64 = 0;
        let mut desc_pa;

        let descriptor = loop {
            descaddr |= (address >> (stride * (4 - level))) & indexmask;
            descaddr &= !7;
            desc_pa = descaddr;
            if s2 {
                // The table address is an IPA: translate it with stage 2 (S1PTW).
                let t =
                    self.stage2(descaddr, MmuAccessType::DataLoad, is_at).map_err(|mut f| {
                        f.stage2 = true;
                        f.s1ptw = true;
                        f.s2addr = descaddr;
                        f
                    })?;
                desc_pa = t.pa;
            }
            let descriptor = self.load_desc(desc_pa, level)?;

            if descriptor & 1 == 0 || (descriptor & 2 == 0 && level == 3) {
                // Invalid, or the Reserved level 3 encoding.
                return Err(Fault::new(fsc::translation(level)));
            }
            if descriptor & 2 == 0 && !block_level_ok(param.gran, level) {
                // A block descriptor at a level that does not allow one.
                return Err(Fault::new(fsc::translation(level)));
            }

            descaddr = descriptor & descaddrmask;
            if descaddr >> outputsize != 0 {
                return Err(Fault::new(fsc::address_size(level)));
            }

            if descriptor & 2 != 0 && level < 3 {
                // Table entry. The top five bits are attributes which may propagate down
                // through lower levels of the table (and which are all arranged so that
                // 0 means "no effect", so we can gather them up by ORing in the bits at
                // each level).
                tableattrs |= extract64(descriptor, 59, 5);
                level += 1;
                indexmask = indexmask_grainsize;
                continue;
            }
            break descriptor;
        };

        // Block entry at level 1 or 2, or page entry at level 3. These are basically the
        // same thing, although the number of bits we pull in from the vaddr varies.
        let page_size = 1u64 << (stride * (4 - level) + 3);
        descaddr &= !(page_size - 1);
        descaddr |= address & (page_size - 1);

        let mut new_descriptor = descriptor;
        if descriptor & (1 << 10) == 0 && !param.ha {
            return Err(Fault::new(fsc::access_flag(level)));
        }
        // For AccessType_AT, DB is not updated, and it is IMPLEMENTATION DEFINED whether AF
        // is updated; like QEMU we choose not to.
        if !is_at {
            if descriptor & (1 << 10) == 0 && param.ha {
                new_descriptor |= 1 << 10;
            }
            if param.hd && extract64(descriptor, 51, 1) != 0 && access == MmuAccessType::DataStore {
                // Clear AP[2].
                new_descriptor &= !(1u64 << 7);
            }
        }

        let mut attrs = new_descriptor & ((((1u64 << 10) - 1) << 2) | (((1u64 << 14) - 1) << 50));
        if !param.hpd {
            // XN, PXN.
            attrs |= extract64(tableattrs, 0, 2) << 53;
            // The sense of AP[1] vs APTable[0] is reversed, as APTable[0] == 1 means
            // "force PL1 access only", which means forcing AP[1] to 0.
            attrs &= !(extract64(tableattrs, 2, 1) << 6);
            attrs |= extract64(tableattrs, 3, 1) << 7;
        }
        let two_ranges = regime_has_2_ranges(mmu_idx);
        let mut ap = extract64(attrs, 6, 2);
        if !two_ranges {
            // In a regime with one privilege level AP[1] is RES1.
            ap |= 1;
        }
        let xn = extract64(attrs, 54, 1) != 0;
        let pxn = extract64(attrs, 53, 1) != 0;

        // get_S1prot().
        let is_user = matches!(mmu_idx, MMU_IDX_E10_0 | MMU_IDX_E20_0);
        let user_rw = if two_ranges { simple_ap_to_rw_prot_is_user(ap, true) } else { 0 };
        let mut prot_rw = simple_ap_to_rw_prot_is_user(ap, false);
        if is_user {
            prot_rw = user_rw;
        } else if user_rw != 0 && matches!(mmu_idx, MMU_IDX_E10_1_PAN | MMU_IDX_E20_2_PAN) {
            // PAN forbids data accesses if EL0 has data permissions.
            prot_rw = 0;
        }
        let wxn = st.sctlr_el[rel] & SCTLR_WXN != 0;
        let xn = if two_ranges && !is_user { pxn || (user_rw & page::WRITE) != 0 } else { xn };
        let prot =
            if xn || (wxn && prot_rw & page::WRITE != 0) { prot_rw } else { prot_rw | page::EXEC };

        if prot & access_bit(access) == 0 {
            return Err(Fault::new(fsc::permission(level)));
        }

        // If FEAT_HAFDBS has made changes, update the descriptor.
        if new_descriptor != descriptor {
            // QEMU uses a compare and swap here; see the module doc of `tcg`.
            let res =
                self.as_.store(desc_pa, 8, new_descriptor, Endian::Little, MemTxAttrs::default());
            if !res.is_ok() {
                return Err(Fault {
                    ea: res != MemTxResult::DECODE_ERROR,
                    ..Fault::new(fsc::sync_external_on_walk(level))
                });
            }
        }

        let attrindx = extract64(attrs, 2, 3) as u32;
        let mair_attr = (st.mair_el[rel] >> (8 * attrindx)) as u8;
        let sh = if mair_attr & 0xf0 == 0 || mair_attr == 0x44 || mair_attr == 0x40 {
            2
        } else {
            extract64(attrs, 8, 2) as u8
        };
        Ok(Translation { pa: descaddr, prot, attrs: mair_attr, sh, page_size })
    }

    /// The stage 2 walk of `ipa` under VTCR_EL2 and VTTBR_EL2.
    fn stage2(&self, ipa: u64, access: MmuAccessType, is_at: bool) -> Result<Translation, Fault> {
        let _ = is_at;
        let st = self.st;
        let vtcr = st.vtcr_el2;
        let tsz = extract64(vtcr, 0, 6) as u32;
        let sl0 = extract64(vtcr, 6, 2) as u32;
        let gran = sanitize_gran(self.feat(), tg0_gran(extract64(vtcr, 14, 2)));
        let ps = extract64(vtcr, 16, 3) as u32;
        let outputsize = self.outputsize(ps);

        if !(16..=39).contains(&tsz) {
            return Err(Fault::new(fsc::translation(0)));
        }
        let inputsize = 64 - tsz;
        if ipa >> inputsize != 0 {
            return Err(Fault::new(fsc::translation(0)));
        }
        let stride = gran as u32 - 3;

        // check_s2_mmu_setup().
        let startlevel = match gran {
            Gran::G4K => 2i32 - sl0 as i32,
            Gran::G16K | Gran::G64K => 3i32 - sl0 as i32,
        };
        let setup_ok = startlevel >= 0
            && sl0 != 3
            && match gran {
                Gran::G64K => !(startlevel == 0 || (startlevel == 1 && outputsize <= 42)),
                Gran::G16K => !(startlevel == 0 || (startlevel == 1 && outputsize <= 40)),
                Gran::G4K => !(startlevel == 0 && outputsize <= 42),
            }
            // This is CONSTRAINED UNPREDICTABLE and we choose to fault.
            && inputsize <= outputsize
            && {
                let grainsize = stride + 3;
                let startsizecheck =
                    inputsize as i32 - ((3 - startlevel) * stride as i32 + grainsize as i32);
                // Too many or too few entries at the starting level fault.
                (1..=stride as i32 + 4).contains(&startsizecheck)
            };
        if !setup_ok {
            return Err(Fault::new(fsc::translation(0)));
        }
        let mut level = startlevel as u32;
        let indexmask_grainsize = (1u64 << (stride + 3)) - 1;
        let mut indexmask = (1u64 << (inputsize - stride * (4 - level))) - 1;
        let mut descaddr = extract64(st.vttbr_el2, 0, 48);
        if descaddr >> outputsize != 0 {
            return Err(Fault::new(fsc::address_size(level)));
        }
        descaddr &= !indexmask;
        let descaddrmask = ((1u64 << 48) - 1) & !indexmask_grainsize;

        let descriptor = loop {
            descaddr |= (ipa >> (stride * (4 - level))) & indexmask;
            descaddr &= !7;
            let descriptor = self.load_desc(descaddr, level)?;
            if descriptor & 1 == 0 || (descriptor & 2 == 0 && level == 3) {
                return Err(Fault::new(fsc::translation(level)));
            }
            if descriptor & 2 == 0 && !block_level_ok(gran, level) {
                return Err(Fault::new(fsc::translation(level)));
            }
            descaddr = descriptor & descaddrmask;
            if descaddr >> outputsize != 0 {
                return Err(Fault::new(fsc::address_size(level)));
            }
            if descriptor & 2 != 0 && level < 3 {
                // Stage 2 table descriptors have no attributes.
                level += 1;
                indexmask = indexmask_grainsize;
                continue;
            }
            break descriptor;
        };
        let page_size = 1u64 << (stride * (4 - level) + 3);
        descaddr &= !(page_size - 1);
        descaddr |= ipa & (page_size - 1);

        if descriptor & (1 << 10) == 0 {
            // Stage 2 has no hardware access flag management here.
            return Err(Fault::new(fsc::access_flag(level)));
        }
        // get_S2prot(): S2AP is bits 6 and 7, XN bit 54.
        let s2ap = extract64(descriptor, 6, 2);
        let mut prot = 0;
        if s2ap & 1 != 0 {
            prot |= page::READ;
        }
        if s2ap & 2 != 0 {
            prot |= page::WRITE;
        }
        if extract64(descriptor, 54, 1) == 0 {
            prot |= page::EXEC;
        }
        if prot & access_bit(access) == 0 {
            return Err(Fault::new(fsc::permission(level)));
        }
        Ok(Translation { pa: descaddr, prot, attrs: 0, sh: 0, page_size })
    }
}

/// `get_phys_addr()` on the vCPU `cpu`.
pub(crate) fn get_phys_addr(
    arm: &Arm,
    cpu: &mut Cpu<'_>,
    address: u64,
    access: MmuAccessType,
    mmu_idx: usize,
    stage1_only: bool,
    is_at: bool,
) -> Result<Translation, Fault> {
    let st = CpuArmState::load(cpu.env);
    let as_ = cpu.core.address_space().clone();
    let w = Walker { arm, st: &st, as_: &as_ };
    w.get_phys_addr(address, access, mmu_idx, stage1_only, is_at)
}

/// `arm_cpu_tlb_fill()`.
pub(crate) fn tlb_fill(
    arm: &Arm,
    cpu: &mut Cpu<'_>,
    address: u64,
    access: MmuAccessType,
    mmu_idx: usize,
    probe: bool,
    ra: Ra,
) -> Result<bool, CpuLoopExit> {
    match get_phys_addr(arm, cpu, address, access, mmu_idx, false, false) {
        Ok(t) => {
            // Map at least a target page; a larger page or block is recorded so that a
            // flush of any address in it flushes the whole of it.
            let size = t.page_size.max(4096);
            cpu.tlb_set_page(address & !0xfff, t.pa & !0xfff, t.prot, mmu_idx, size);
            Ok(true)
        }
        Err(_) if probe => Ok(false),
        Err(fault) => Err(deliver_fault(arm, cpu, address, access, fault, ra)),
    }
}

/// `arm_deliver_fault()`: raise the abort for `fault`; stage 2 faults go to EL2 with the
/// IPA in HPFAR_EL2.
pub(crate) fn deliver_fault(
    arm: &Arm,
    cpu: &mut Cpu<'_>,
    addr: u64,
    access: MmuAccessType,
    fault: Fault,
    ra: Ra,
) -> CpuLoopExit {
    let mut st = CpuArmState::load(cpu.env);
    let mut target_el = exception_target_el(&st);
    if fault.stage2 {
        target_el = 2;
        st.hpfar_el2 = extract64(fault.s2addr, 12, 47) << 4;
    } else if target_el == 1 && st.hcr_el2_eff(arm.features()) & HCR_TGE != 0 {
        // raise_exception() redirects the exception to EL2.
        target_el = 2;
    }
    let same_el = st.current_el() == target_el;
    let (excp, syn) = if access == MmuAccessType::InstFetch {
        (EXCP_PREFETCH_ABORT, syn_insn_abort(same_el, fault.ea, fault.s1ptw, fault.fsc))
    } else {
        let wnr = access == MmuAccessType::DataStore;
        (EXCP_DATA_ABORT, syn_data_abort_no_iss(same_el, fault.ea, fault.s1ptw, wnr, fault.fsc))
    };
    st.exception_vaddress = addr;
    st.exception_syndrome = syn;
    st.exception_target_el = target_el;
    st.store(cpu.env);
    cpu.raise_exception(excp, ra)
}
