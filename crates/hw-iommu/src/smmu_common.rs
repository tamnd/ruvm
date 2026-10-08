// SPDX-License-Identifier: GPL-2.0-or-later

//! The SMMU base, from hw/arm/smmu-common.c, hw/arm/smmu-internal.h and
//! include/hw/arm/smmu-common.h: the VMSAv8-64 page table walks of stage 1 and stage 2, their
//! nesting, and the IOTLB with its invalidations.
//!
//! The walks read the page tables through [`WalkMemory`], which the SMMUv3 model points at the
//! system address space, as QEMU reads them from `address_space_memory`.
//!
//! # Differences from QEMU
//!
//! - The IOTLB is a `HashMap` holding copies of the entries, so a lookup gives a copy rather
//!   than a pointer into the table. The hit and miss counters, which only feed traces, are gone.
//! - The configuration cache is keyed by stream ID instead of by `SMMUDevice`, which is the
//!   same thing for the devices of one root bus.

use std::collections::HashMap;

/// `IOMMU_NONE`.
pub const PERM_NONE: u8 = 0;
/// `IOMMU_RO`.
pub const PERM_RO: u8 = 1;
/// `IOMMU_WO`.
pub const PERM_WO: u8 = 2;
/// `IOMMU_RW`.
pub const PERM_RW: u8 = 3;

/// `VMSA_LEVELS`.
const VMSA_LEVELS: i32 = 4;
/// `VMSA_MAX_S2_CONCAT`.
pub const VMSA_MAX_S2_CONCAT: u64 = 16;
/// `SMMU_IOTLB_MAX_SIZE`.
pub const SMMU_IOTLB_MAX_SIZE: usize = 256;

/// `MAKE_64BIT_MASK(0, len)`.
pub(crate) fn mask64(len: u32) -> u64 {
    if len >= 64 { u64::MAX } else { (1u64 << len) - 1 }
}

/// `extract64()`.
pub(crate) fn extract64(v: u64, start: u32, len: u32) -> u64 {
    (v >> start) & mask64(len)
}

/// `sextract64()`.
fn sextract64(v: u64, start: u32, len: u32) -> i64 {
    ((v << (64 - len - start)) as i64) >> (64 - len)
}

/// Which translation stages are in use, `SMMUStage`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Stage {
    /// `SMMU_STAGE_1`.
    #[default]
    S1,
    /// `SMMU_STAGE_2`.
    S2,
    /// `SMMU_NESTED`, both stages.
    Nested,
}

/// The kind of a page table walk fault, `SMMUPTWEventType`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PtwError {
    /// `SMMU_PTW_ERR_NONE`.
    #[default]
    None,
    /// `SMMU_PTW_ERR_WALK_EABT`: a descriptor could not be read.
    WalkEabt,
    /// `SMMU_PTW_ERR_TRANSLATION`.
    Translation,
    /// `SMMU_PTW_ERR_ADDR_SIZE`.
    AddrSize,
    /// `SMMU_PTW_ERR_ACCESS`.
    Access,
    /// `SMMU_PTW_ERR_PERMISSION`.
    Permission,
}

/// What went wrong in a walk, `SMMUPTWEventInfo`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PtwEventInfo {
    /// The stage that faulted.
    pub stage: Stage,
    /// The fault.
    pub ty: PtwError,
    /// The fetched descriptor address for an external abort, or the IPA of a stage 2 fault.
    pub addr: u64,
    /// Whether the stage 2 fault happened while translating a stage 1 descriptor address.
    pub is_ipa_descriptor: bool,
}

/// One stage 1 translation table, `SMMUTransTableInfo`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TransTableInfo {
    /// The table is disabled (EPDx).
    pub disabled: bool,
    /// The translation table base.
    pub ttb: u64,
    /// TxSZ.
    pub tsz: u8,
    /// The granule as a shift: 12, 14 or 16.
    pub granule_sz: u8,
    /// Hierarchical attribute disable.
    pub had: bool,
}

/// The stage 2 configuration, `SMMUS2Cfg`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct S2Cfg {
    /// S2T0SZ.
    pub tsz: u8,
    /// S2SL0.
    pub sl0: u8,
    /// S2AFFD.
    pub affd: bool,
    /// S2R.
    pub record_faults: bool,
    /// The granule as a shift.
    pub granule_sz: u8,
    /// The effective output size in bits.
    pub eff_ps: u8,
    /// S2VMID, or -1.
    pub vmid: i32,
    /// S2TTB.
    pub vttb: u64,
}

/// A decoded STE and CD, `SMMUTransCfg`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TransCfg {
    /// The stages in use.
    pub stage: Stage,
    /// Translation is disabled.
    pub disabled: bool,
    /// Translation is bypassed.
    pub bypassed: bool,
    /// Transactions are aborted.
    pub aborted: bool,
    /// AF fault disable.
    pub affd: bool,
    /// AArch64 tables.
    pub aa64: bool,
    /// Record stage 1 faults.
    pub record_faults: bool,
    /// The output address size in bits.
    pub oas: u8,
    /// Top byte ignore.
    pub tbi: u8,
    /// The ASID, or -1.
    pub asid: i32,
    /// TTB0 and TTB1.
    pub tt: [TransTableInfo; 2],
    /// Stage 2.
    pub s2cfg: S2Cfg,
}

/// A cached translation, `SMMUTLBEntry` with its `IOMMUTLBEntry`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TlbEntry {
    /// The input address, aligned to the mapping.
    pub iova: u64,
    /// The output address, aligned to the mapping.
    pub translated_addr: u64,
    /// The offset mask of the mapping.
    pub addr_mask: u64,
    /// The rights of the first stage in use.
    pub perm: u8,
    /// The rights of the last stage in use.
    pub parent_perm: u8,
    /// The level of the leaf descriptor.
    pub level: u8,
    /// The granule as a shift.
    pub granule: u8,
}

impl TlbEntry {
    /// `CACHED_ENTRY_TO_ADDR()`.
    pub fn to_addr(&self, addr: u64) -> u64 {
        self.translated_addr.wrapping_add(addr & self.addr_mask)
    }
}

/// `SMMUIOTLBKey`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IotlbKey {
    /// The aligned input address.
    pub iova: u64,
    /// The ASID, or -1 for stage 2 entries.
    pub asid: i32,
    /// The VMID, or -1.
    pub vmid: i32,
    /// The translation granule code: 1 for 4K, 2 for 16K, 3 for 64K.
    pub tg: u8,
    /// The level.
    pub level: u8,
}

/// Reads page table descriptors for the walks.
pub trait WalkMemory {
    /// `ldq_le_dma()`: the little endian 64 bit word at `addr`, or `None` on a bus error.
    fn ldq_le(&self, addr: u64) -> Option<u64>;
}

/// The IOTLB of `SMMUState`.
#[derive(Debug, Default)]
pub struct Iotlb {
    map: HashMap<IotlbKey, TlbEntry>,
}

/// `level_shift()`.
fn level_shift(level: i32, granule_sz: i32) -> i32 {
    granule_sz + (3 - level) * (granule_sz - 3)
}

/// `iova_level_offset()`.
fn iova_level_offset(iova: u64, inputsize: i32, level: i32, gsz: i32) -> u64 {
    let shift = level_shift(level, gsz) as u32;
    (iova & mask64(inputsize as u32)).checked_shr(shift).unwrap_or(0) & mask64((gsz - 3) as u32)
}

/// `get_start_level()`. FEAT_LPA2 and FEAT_TTST are not implemented.
pub(crate) fn get_start_level(sl0: i32, granule_sz: i32) -> i32 {
    if granule_sz == 12 { 2 - sl0 } else { 3 - sl0 }
}

/// `pgd_concat_idx()`: the index of the table in a concatenated first level of stage 2.
pub(crate) fn pgd_concat_idx(start_level: i32, granule_sz: i32, ipa: u64) -> u64 {
    let shift = level_shift(start_level - 1, granule_sz) as u32;
    ipa.checked_shr(shift).unwrap_or(0)
}

/// `VMSA_IDXMSK()`.
fn vmsa_idxmsk(isz: i32, strd: i32, lvl: i32) -> u64 {
    mask64((isz - strd * (VMSA_LEVELS - lvl)) as u32)
}

/// `PTE_ADDRESS()`.
fn pte_address(pte: u64, shift: i32) -> u64 {
    if !(0..=47).contains(&shift) {
        return 0;
    }
    let shift = shift as u32;
    extract64(pte, shift, 47 - shift + 1) << shift
}

fn is_invalid_pte(pte: u64) -> bool {
    pte & 1 == 0
}

fn is_reserved_pte(pte: u64, level: i32) -> bool {
    level == 3 && pte & 3 == 1
}

fn is_table_pte(pte: u64, level: i32) -> bool {
    level < 3 && pte & 3 == 3
}

fn is_page_pte(pte: u64, level: i32) -> bool {
    level == 3 && pte & 3 == 3
}

fn pte_ap(pte: u64) -> u8 {
    extract64(pte, 6, 2) as u8
}

fn pte_aptable(pte: u64) -> u8 {
    extract64(pte, 61, 2) as u8
}

fn pte_af(pte: u64) -> bool {
    pte & (1 << 10) != 0
}

/// `is_permission_fault()`. All transactions count as privileged.
fn is_permission_fault(ap: u8, perm: u8) -> bool {
    perm & PERM_WO != 0 && ap & 2 != 0
}

/// `is_permission_fault_s2()`.
fn is_permission_fault_s2(s2ap: u8, perm: u8) -> bool {
    s2ap & perm != perm
}

/// `PTE_AP_TO_PERM()`.
fn pte_ap_to_perm(ap: u8) -> u8 {
    PERM_RO | if ap & 2 == 0 { PERM_WO } else { 0 }
}

/// `TBI0()`.
fn tbi0(tbi: u8) -> bool {
    tbi & 1 != 0
}

/// `TBI1()`. QEMU writes this as `(tbi) & 0x2 >> 1`, which by C precedence is `tbi & 1`, and
/// that is what is reproduced here.
fn tbi1(tbi: u8) -> bool {
    tbi & 1 != 0
}

/// `select_tt()`: the index of the stage 1 table that covers `iova`, or `None` in the gap
/// between the two regions.
pub fn select_tt(cfg: &TransCfg, iova: u64) -> Option<usize> {
    let tbi = if extract64(iova, 55, 1) != 0 { tbi1(cfg.tbi) } else { tbi0(cfg.tbi) };
    let tbi_byte = u32::from(tbi) * 8;
    let tsz0 = u32::from(cfg.tt[0].tsz);
    let tsz1 = u32::from(cfg.tt[1].tsz);
    if tsz0 != 0 && tsz0 > tbi_byte && extract64(iova, 64 - tsz0, tsz0 - tbi_byte) == 0 {
        // There is a TTBR0 region and the address is in it.
        Some(0)
    } else if tsz1 != 0 && tsz1 > tbi_byte && sextract64(iova, 64 - tsz1, tsz1 - tbi_byte) == -1 {
        // There is a TTBR1 region and the address is in it.
        Some(1)
    } else if tsz0 == 0 {
        // The TTBR0 region is everything not in the TTBR1 region.
        Some(0)
    } else if tsz1 == 0 {
        Some(1)
    } else {
        None
    }
}

fn get_pte(
    mem: &dyn WalkMemory,
    baseaddr: u64,
    index: u64,
    info: &mut PtwEventInfo,
) -> Option<u64> {
    let addr = baseaddr.wrapping_add(index * 8);
    let pte = mem.ldq_le(addr);
    if pte.is_none() {
        info.ty = PtwError::WalkEabt;
        info.addr = addr;
    }
    pte
}

/// The leaf address of a page or block descriptor.
fn leaf_address(pte: u64, level: i32, granule_sz: i32) -> u64 {
    if is_page_pte(pte, level) {
        pte_address(pte, granule_sz)
    } else {
        pte_address(pte, level_shift(level, granule_sz))
    }
}

impl Iotlb {
    /// An empty IOTLB.
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of cached entries.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether nothing is cached.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    fn lookup_all_levels(
        &self,
        cfg: &TransCfg,
        granule_sz: u8,
        tsz: u8,
        iova: u64,
    ) -> Option<TlbEntry> {
        let g = i32::from(granule_sz);
        let tg = (granule_sz.wrapping_sub(10)) / 2;
        let inputsize = 64 - i32::from(tsz);
        let stride = g - 3;
        if stride <= 0 {
            return None;
        }
        let mut level = 4 - (inputsize - 4) / stride;
        while level <= 3 {
            let mask = mask64(level_shift(level, g) as u32);
            let key = IotlbKey {
                iova: iova & !mask,
                asid: cfg.asid,
                vmid: cfg.s2cfg.vmid,
                tg,
                level: level as u8,
            };
            if let Some(e) = self.map.get(&key) {
                return Some(*e);
            }
            level += 1;
        }
        None
    }

    /// `smmu_iotlb_lookup()`. For nested translation it also tries the stage 2 granule, as an
    /// entry is inserted with it when the stage 2 mapping was the smaller one.
    pub fn lookup(&self, cfg: &TransCfg, granule_sz: u8, tsz: u8, iova: u64) -> Option<TlbEntry> {
        let mut entry = self.lookup_all_levels(cfg, granule_sz, tsz, iova);
        if entry.is_none() && cfg.stage == Stage::Nested && cfg.s2cfg.granule_sz != granule_sz {
            entry = self.lookup_all_levels(cfg, cfg.s2cfg.granule_sz, tsz, iova);
        }
        entry
    }

    /// `smmu_iotlb_insert()`.
    pub fn insert(&mut self, cfg: &TransCfg, new: TlbEntry) {
        if self.map.len() >= SMMU_IOTLB_MAX_SIZE {
            self.inv_all();
        }
        let key = IotlbKey {
            iova: new.iova,
            asid: cfg.asid,
            vmid: cfg.s2cfg.vmid,
            tg: new.granule.wrapping_sub(10) / 2,
            level: new.level,
        };
        self.map.insert(key, new);
    }

    /// `smmu_iotlb_inv_all()`.
    pub fn inv_all(&mut self) {
        self.map.clear();
    }

    /// `smmu_iotlb_inv_asid_vmid()`.
    pub fn inv_asid_vmid(&mut self, asid: i32, vmid: i32) {
        self.map.retain(|k, _| !(k.asid == asid && k.vmid == vmid));
    }

    /// `smmu_iotlb_inv_vmid()`.
    pub fn inv_vmid(&mut self, vmid: i32) {
        self.map.retain(|k, _| k.vmid != vmid);
    }

    /// `smmu_iotlb_inv_vmid_s1()`: the stage 1 entries of `vmid`.
    pub fn inv_vmid_s1(&mut self, vmid: i32) {
        self.map.retain(|k, _| !(k.vmid == vmid && k.asid >= 0));
    }

    /// `smmu_iotlb_inv_iova()`. A negative `asid` or `vmid` matches any.
    pub fn inv_iova(&mut self, asid: i32, vmid: i32, iova: u64, tg: u8, num_pages: u64, ttl: u8) {
        // Without a TG the range is in 4K pages.
        let granule = if tg != 0 { u32::from(tg) * 2 + 10 } else { 12 };
        if ttl != 0 && num_pages == 1 && asid >= 0 {
            let key = IotlbKey { iova, asid, vmid, tg, level: ttl };
            if self.map.remove(&key).is_some() {
                return;
            }
            // Otherwise it may be part of a larger entry.
        }
        let mask = num_pages.wrapping_shl(granule).wrapping_sub(1);
        self.map.retain(|k, e| {
            if asid >= 0 && asid != k.asid {
                return true;
            }
            if vmid >= 0 && vmid != k.vmid {
                return true;
            }
            !((iova & !e.addr_mask) == e.iova || (e.iova & !mask) == iova)
        });
    }

    /// `smmu_iotlb_inv_ipa()`: like [`Iotlb::inv_iova`] for the stage 2 entries of `vmid`.
    pub fn inv_ipa(&mut self, vmid: i32, ipa: u64, tg: u8, num_pages: u64, ttl: u8) {
        let granule = if tg != 0 { u32::from(tg) * 2 + 10 } else { 12 };
        if ttl != 0 && num_pages == 1 {
            let key = IotlbKey { iova: ipa, asid: -1, vmid, tg, level: ttl };
            if self.map.remove(&key).is_some() {
                return;
            }
        }
        let mask = num_pages.wrapping_shl(granule).wrapping_sub(1);
        self.map.retain(|k, e| {
            if k.asid >= 0 {
                // A stage 1 entry.
                return true;
            }
            if vmid != k.vmid {
                return true;
            }
            !((ipa & !e.addr_mask) == e.iova || (e.iova & !mask) == ipa)
        });
    }

    /// Whether an entry with this key is cached, for tests.
    pub fn contains(&self, key: &IotlbKey) -> bool {
        self.map.contains_key(key)
    }
}

/// `translate_table_addr_ipa()`: a stage 1 table address of a nested walk, through stage 2.
fn translate_table_addr_ipa(
    tlb: &mut Iotlb,
    mem: &dyn WalkMemory,
    cfg: &TransCfg,
    addr: u64,
    info: &mut PtwEventInfo,
) -> Option<u64> {
    let mut s2 = *cfg;
    s2.stage = Stage::S2;
    s2.asid = -1;
    if let Some(e) = smmu_translate(tlb, mem, &s2, addr, PERM_RO, info) {
        return Some(e.to_addr(addr));
    }
    info.stage = Stage::S2;
    info.addr = addr;
    info.is_ipa_descriptor = true;
    None
}

/// `smmu_ptw_64_s1()`.
fn ptw_64_s1(
    tlb: &mut Iotlb,
    mem: &dyn WalkMemory,
    cfg: &TransCfg,
    iova: u64,
    perm: u8,
    info: &mut PtwEventInfo,
) -> Option<TlbEntry> {
    let r = ptw_64_s1_inner(tlb, mem, cfg, iova, perm, info);
    if r.is_none() {
        info.stage = Stage::S1;
    }
    r
}

fn ptw_64_s1_inner(
    tlb: &mut Iotlb,
    mem: &dyn WalkMemory,
    cfg: &TransCfg,
    iova: u64,
    perm: u8,
    info: &mut PtwEventInfo,
) -> Option<TlbEntry> {
    let tt = match select_tt(cfg, iova) {
        Some(i) if !cfg.tt[i].disabled => cfg.tt[i],
        _ => {
            info.ty = PtwError::Translation;
            return None;
        }
    };
    let granule_sz = i32::from(tt.granule_sz);
    let stride = granule_sz - 3;
    let inputsize = 64 - i32::from(tt.tsz);
    if stride <= 0 {
        info.ty = PtwError::Translation;
        return None;
    }
    let mut level = 4 - (inputsize - 4) / stride;
    let indexmask = vmsa_idxmsk(inputsize, stride, level);
    let mut baseaddr = extract64(tt.ttb, 0, u32::from(cfg.oas)) & !indexmask;

    while level < VMSA_LEVELS {
        let mask = mask64(level_shift(level, granule_sz) as u32);
        let offset = iova_level_offset(iova, inputsize, level, granule_sz);
        let pte = get_pte(mem, baseaddr, offset, info)?;
        if is_invalid_pte(pte) || is_reserved_pte(pte, level) {
            break;
        }
        if is_table_pte(pte, level) {
            let ap = pte_aptable(pte);
            if is_permission_fault(ap, perm) && !tt.had {
                info.ty = PtwError::Permission;
                return None;
            }
            baseaddr = pte_address(pte, granule_sz);
            if cfg.stage == Stage::Nested {
                baseaddr = translate_table_addr_ipa(tlb, mem, cfg, baseaddr, info)?;
            }
            level += 1;
            continue;
        }
        let gpa = leaf_address(pte, level, granule_sz);
        // HTTU is not implemented, so with AFFD and AF both 0 this is an access flag fault,
        // which takes priority over a permission fault.
        if !pte_af(pte) && !cfg.affd {
            info.ty = PtwError::Access;
            return None;
        }
        let ap = pte_ap(pte);
        if is_permission_fault(ap, perm) {
            info.ty = PtwError::Permission;
            return None;
        }
        // An output beyond the effective IPA size of the CD is a stage 1 address size fault.
        if gpa >= 1u64.checked_shl(u32::from(cfg.oas)).unwrap_or(u64::MAX) {
            info.ty = PtwError::AddrSize;
            return None;
        }
        let p = pte_ap_to_perm(ap);
        return Some(TlbEntry {
            iova: iova & !mask,
            translated_addr: gpa,
            addr_mask: mask,
            perm: p,
            parent_perm: p,
            level: level as u8,
            granule: granule_sz as u8,
        });
    }
    info.ty = PtwError::Translation;
    None
}

/// `smmu_ptw_64_s2()`.
fn ptw_64_s2(
    mem: &dyn WalkMemory,
    cfg: &TransCfg,
    ipa: u64,
    perm: u8,
    info: &mut PtwEventInfo,
) -> Option<TlbEntry> {
    let r = ptw_64_s2_inner(mem, cfg, ipa, perm, info);
    if let Err(at_ipa) = r {
        if at_ipa {
            info.addr = ipa;
        }
        info.stage = Stage::S2;
        return None;
    }
    r.ok()
}

/// The walk of [`ptw_64_s2`]. An error says whether the fault reports the IPA.
fn ptw_64_s2_inner(
    mem: &dyn WalkMemory,
    cfg: &TransCfg,
    ipa: u64,
    perm: u8,
    info: &mut PtwEventInfo,
) -> Result<TlbEntry, bool> {
    let s2 = &cfg.s2cfg;
    let granule_sz = i32::from(s2.granule_sz);
    let inputsize = 64 - i32::from(s2.tsz);
    let mut level = get_start_level(i32::from(s2.sl0), granule_sz);
    let stride = granule_sz - 3;
    let idx = pgd_concat_idx(level, granule_sz, ipa);
    // The table of the concatenated first level that holds `ipa`.
    let concat = (1u64 << stride).wrapping_mul(idx).wrapping_mul(8);
    let indexmask = vmsa_idxmsk(inputsize, stride, level);
    let mut baseaddr =
        extract64(s2.vttb, 0, u32::from(s2.eff_ps)).wrapping_add(concat) & !indexmask;

    // An IPA outside the range of S2T0SZ is a stage 2 translation fault.
    if ipa >= 1u64.checked_shl(inputsize as u32).unwrap_or(u64::MAX) {
        info.ty = PtwError::Translation;
        return Err(true);
    }

    while level < VMSA_LEVELS {
        let mask = mask64(level_shift(level, granule_sz) as u32);
        let offset = iova_level_offset(ipa, inputsize, level, granule_sz);
        let Some(pte) = get_pte(mem, baseaddr, offset, info) else {
            return Err(false);
        };
        if is_invalid_pte(pte) || is_reserved_pte(pte, level) {
            break;
        }
        if is_table_pte(pte, level) {
            baseaddr = pte_address(pte, granule_sz);
            level += 1;
            continue;
        }
        let gpa = leaf_address(pte, level, granule_sz);
        // With S2AFFD and AF both 0 this is an access fault, ahead of a permission fault.
        if !pte_af(pte) && !s2.affd {
            info.ty = PtwError::Access;
            return Err(true);
        }
        let s2ap = pte_ap(pte);
        if is_permission_fault_s2(s2ap, perm) {
            info.ty = PtwError::Permission;
            return Err(true);
        }
        // An output beyond the effective PA size is a stage 2 address size fault.
        if gpa >= 1u64.checked_shl(u32::from(s2.eff_ps)).unwrap_or(u64::MAX) {
            info.ty = PtwError::AddrSize;
            return Err(true);
        }
        return Ok(TlbEntry {
            iova: ipa & !mask,
            translated_addr: gpa,
            addr_mask: mask,
            perm: s2ap,
            parent_perm: s2ap,
            level: level as u8,
            granule: granule_sz as u8,
        });
    }
    info.ty = PtwError::Translation;
    Err(true)
}

/// `combine_tlb()`: the stage 1 entry `s1` with its stage 2 entry `s2`, as one entry.
fn combine_tlb(s1: &mut TlbEntry, s2: &TlbEntry, iova: u64) {
    if s2.addr_mask < s1.addr_mask {
        s1.addr_mask = s2.addr_mask;
        s1.granule = s2.granule;
        s1.level = s2.level;
    }
    s1.translated_addr = s2.to_addr(s1.translated_addr);
    s1.iova = iova & !s1.addr_mask;
    // parent_perm has the stage 2 rights while perm keeps those of stage 1.
    s1.parent_perm = s2.perm;
}

/// `smmu_ptw()`: walks the tables of `cfg` for `iova`.
pub fn smmu_ptw(
    tlb: &mut Iotlb,
    mem: &dyn WalkMemory,
    cfg: &TransCfg,
    iova: u64,
    perm: u8,
    info: &mut PtwEventInfo,
) -> Option<TlbEntry> {
    match cfg.stage {
        Stage::S1 => ptw_64_s1(tlb, mem, cfg, iova, perm, info),
        Stage::S2 => {
            // With stage 1 bypassed the input goes to stage 2 as the IPA, and an input beyond
            // the IAS, which is the OAS for AArch64, is a stage 1 address size fault.
            if iova >= 1u64.checked_shl(u32::from(cfg.oas)).unwrap_or(u64::MAX) {
                info.ty = PtwError::AddrSize;
                info.stage = Stage::S1;
                return None;
            }
            ptw_64_s2(mem, cfg, iova, perm, info)
        }
        Stage::Nested => {
            let mut s1 = ptw_64_s1(tlb, mem, cfg, iova, perm, info)?;
            let ipa = s1.to_addr(iova);
            let s2 = ptw_64_s2(mem, cfg, ipa, perm, info)?;
            combine_tlb(&mut s1, &s2, iova);
            Some(s1)
        }
    }
}

/// `smmu_translate()`: an IOTLB lookup, then a walk that fills the IOTLB on a miss.
pub fn smmu_translate(
    tlb: &mut Iotlb,
    mem: &dyn WalkMemory,
    cfg: &TransCfg,
    addr: u64,
    flag: u8,
    info: &mut PtwEventInfo,
) -> Option<TlbEntry> {
    // The attributes of the input stage, for the lookup.
    let (granule_sz, tsz) = if cfg.stage == Stage::S2 {
        (cfg.s2cfg.granule_sz, cfg.s2cfg.tsz)
    } else {
        let Some(i) = select_tt(cfg, addr) else {
            info.ty = PtwError::Translation;
            info.stage = Stage::S1;
            return None;
        };
        (cfg.tt[i].granule_sz, cfg.tt[i].tsz)
    };

    if let Some(e) = tlb.lookup(cfg, granule_sz, tsz, addr) {
        if flag & PERM_WO != 0 && e.perm & e.parent_perm & PERM_WO == 0 {
            info.ty = PtwError::Permission;
            info.stage = if e.perm & PERM_WO == 0 { Stage::S1 } else { Stage::S2 };
            return None;
        }
        return Some(e);
    }

    let e = smmu_ptw(tlb, mem, cfg, addr, flag, info)?;
    tlb.insert(cfg, e);
    Some(e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    /// Sparse guest memory of 64 bit words.
    #[derive(Default)]
    struct Mem(RefCell<HashMap<u64, u64>>);

    impl Mem {
        fn set(&self, addr: u64, v: u64) {
            self.0.borrow_mut().insert(addr, v);
        }
    }

    impl WalkMemory for Mem {
        fn ldq_le(&self, addr: u64) -> Option<u64> {
            if addr >= 1 << 40 {
                return None;
            }
            Some(self.0.borrow().get(&addr).copied().unwrap_or(0))
        }
    }

    const AF: u64 = 1 << 10;

    /// A stage 1 configuration with a 4K granule and a 48 bit TTB0 region at 0x10000.
    fn s1_cfg() -> TransCfg {
        let mut cfg = TransCfg { stage: Stage::S1, oas: 44, asid: 1, ..TransCfg::default() };
        cfg.s2cfg.vmid = -1;
        cfg.tt[0] = TransTableInfo { ttb: 0x10000, tsz: 16, granule_sz: 12, ..Default::default() };
        cfg.tt[1].disabled = true;
        cfg
    }

    /// Maps the 4K page `iova` to `pa` with tables at 0x10000, 0x11000, 0x12000 and 0x13000.
    fn map_4k(m: &Mem, iova: u64, pa: u64, ap: u64) {
        let idx = |l: u32| (iova >> (12 + 9 * (3 - l))) & 0x1ff;
        m.set(0x10000 + idx(0) * 8, 0x11000 | 3);
        m.set(0x11000 + idx(1) * 8, 0x12000 | 3);
        m.set(0x12000 + idx(2) * 8, 0x13000 | 3);
        m.set(0x13000 + idx(3) * 8, pa | AF | (ap << 6) | 3);
    }

    #[test]
    fn helpers() {
        assert_eq!(level_shift(3, 12), 12);
        assert_eq!(level_shift(0, 12), 39);
        assert_eq!(level_shift(1, 16), 42);
        assert_eq!(vmsa_idxmsk(48, 9, 0), 0xfff);
        assert_eq!(get_start_level(1, 12), 1);
        assert_eq!(get_start_level(1, 16), 2);
        assert_eq!(pte_address(0xffff_0000_1234_5fff, 12), 0x1234_5000);
        assert_eq!(pte_ap_to_perm(0), PERM_RW);
        assert_eq!(pte_ap_to_perm(2), PERM_RO);
    }

    #[test]
    fn select_tt_regions() {
        let mut cfg = s1_cfg();
        cfg.tt[1] = TransTableInfo { tsz: 16, granule_sz: 12, ..Default::default() };
        assert_eq!(select_tt(&cfg, 0x1000), Some(0));
        assert_eq!(select_tt(&cfg, 0xffff_0000_0000_1000), Some(1));
        assert_eq!(select_tt(&cfg, 0x00f0_0000_0000_0000), None);
        // TBI on TTB0 ignores the top byte.
        cfg.tbi = 1;
        assert_eq!(select_tt(&cfg, 0x5a00_0000_0000_1000), Some(0));
    }

    #[test]
    fn stage1_walk_and_iotlb() {
        let m = Mem::default();
        map_4k(&m, 0x8000_1000, 0x4000_0000, 0);
        let cfg = s1_cfg();
        let mut tlb = Iotlb::new();
        let mut info = PtwEventInfo::default();
        let e = smmu_translate(&mut tlb, &m, &cfg, 0x8000_1234, PERM_WO, &mut info).unwrap();
        assert_eq!(e.to_addr(0x8000_1234), 0x4000_0234);
        assert_eq!((e.level, e.granule, e.perm), (3, 12, PERM_RW));
        assert_eq!(tlb.len(), 1);
        // A second lookup hits the IOTLB even when the tables are gone.
        m.0.borrow_mut().clear();
        let e = smmu_translate(&mut tlb, &m, &cfg, 0x8000_1ff0, PERM_RO, &mut info).unwrap();
        assert_eq!(e.to_addr(0x8000_1ff0), 0x4000_0ff0);
        // Invalidating the page drops it.
        tlb.inv_iova(1, -1, 0x8000_1000, 0, 1, 0);
        assert!(tlb.is_empty());
        assert!(smmu_translate(&mut tlb, &m, &cfg, 0x8000_1000, PERM_RO, &mut info).is_none());
        assert_eq!((info.ty, info.stage), (PtwError::Translation, Stage::S1));
    }

    #[test]
    fn stage1_faults() {
        let m = Mem::default();
        map_4k(&m, 0x1000, 0x5000, 2);
        let cfg = s1_cfg();
        let mut tlb = Iotlb::new();
        let mut info = PtwEventInfo::default();
        // AP[2] makes the page read only.
        assert!(smmu_translate(&mut tlb, &m, &cfg, 0x1000, PERM_WO, &mut info).is_none());
        assert_eq!(info.ty, PtwError::Permission);
        let e = smmu_translate(&mut tlb, &m, &cfg, 0x1000, PERM_RO, &mut info).unwrap();
        assert_eq!(e.perm, PERM_RO);
        // A write that hits the read only entry faults without a walk.
        assert!(smmu_translate(&mut tlb, &m, &cfg, 0x1000, PERM_WO, &mut info).is_none());
        assert_eq!((info.ty, info.stage), (PtwError::Permission, Stage::S1));
        // A leaf without AF is an access fault.
        m.set(0x13000 + 2 * 8, 0x6000 | 3);
        assert!(smmu_translate(&mut tlb, &m, &cfg, 0x2000, PERM_RO, &mut info).is_none());
        assert_eq!(info.ty, PtwError::Access);
        // An unreadable table is an external abort at the descriptor address.
        m.set(0x10000 + 8, (1 << 41) | 3);
        assert!(smmu_translate(&mut tlb, &m, &cfg, 1 << 39, PERM_RO, &mut info).is_none());
        assert_eq!((info.ty, info.addr), (PtwError::WalkEabt, 1 << 41));
    }

    #[test]
    fn block_mapping() {
        let m = Mem::default();
        m.set(0x10000, 0x11000 | 3);
        // A 2M block at level 2 for 0x20_0000.
        m.set(0x11000, 0x12000 | 3);
        m.set(0x12000 + 8, 0x4020_0000 | AF | 1);
        let cfg = s1_cfg();
        let mut tlb = Iotlb::new();
        let mut info = PtwEventInfo::default();
        let e = smmu_translate(&mut tlb, &m, &cfg, 0x2a_bcde, PERM_RO, &mut info).unwrap();
        assert_eq!(e.to_addr(0x2a_bcde), 0x402a_bcde);
        assert_eq!((e.level, e.addr_mask), (2, 0x1f_ffff));
        // A range invalidation of a page inside the block drops the block.
        tlb.inv_iova(-1, -1, 0x2b_0000, 0, 1, 0);
        assert!(tlb.is_empty());
    }

    fn s2_cfg() -> TransCfg {
        let mut cfg = TransCfg { stage: Stage::S2, oas: 44, asid: -1, ..TransCfg::default() };
        // A 40 bit IPA with 4K pages starting at level 1 (SL0 = 1), two concatenated tables.
        cfg.s2cfg = S2Cfg {
            tsz: 24,
            sl0: 1,
            granule_sz: 12,
            eff_ps: 44,
            vmid: 3,
            vttb: 0x20000,
            ..S2Cfg::default()
        };
        cfg
    }

    #[test]
    fn stage2_walk() {
        let m = Mem::default();
        let ipa = 0x80_4000_3000u64;
        // The second concatenated table at 0x21000 covers IPAs from 512G, but as in QEMU the
        // index mask of the whole concatenated table drops that offset again, so the walk
        // reads the entry from the first table.
        let l1 = 0x20000 + ((ipa >> 30) & 0x1ff) * 8;
        m.set(l1, 0x30000 | 3);
        m.set(0x30000 + ((ipa >> 21) & 0x1ff) * 8, 0x31000 | 3);
        // S2AP = 1 is read only.
        m.set(0x31000 + ((ipa >> 12) & 0x1ff) * 8, 0x7000_0000 | AF | (1 << 6) | 3);
        let cfg = s2_cfg();
        let mut tlb = Iotlb::new();
        let mut info = PtwEventInfo::default();
        let e = smmu_translate(&mut tlb, &m, &cfg, ipa + 8, PERM_RO, &mut info).unwrap();
        assert_eq!(e.to_addr(ipa + 8), 0x7000_0008);
        assert_eq!(e.perm, PERM_RO);
        assert!(tlb.contains(&IotlbKey { iova: ipa, asid: -1, vmid: 3, tg: 1, level: 3 }));
        // A stage 2 write to it is a permission fault reporting the IPA.
        tlb.inv_vmid(3);
        assert!(smmu_translate(&mut tlb, &m, &cfg, ipa, PERM_WO, &mut info).is_none());
        assert_eq!((info.ty, info.stage, info.addr), (PtwError::Permission, Stage::S2, ipa));
        // An IPA past S2T0SZ faults.
        assert!(smmu_translate(&mut tlb, &m, &cfg, 1 << 40, PERM_RO, &mut info).is_none());
        assert_eq!((info.ty, info.addr), (PtwError::Translation, 1 << 40));
        // So does an input past the OAS, as a stage 1 address size fault.
        assert!(smmu_translate(&mut tlb, &m, &cfg, 1 << 44, PERM_RO, &mut info).is_none());
        assert_eq!((info.ty, info.stage), (PtwError::AddrSize, Stage::S1));
    }

    #[test]
    fn nested_walk() {
        let m = Mem::default();
        // Stage 2 maps IPA 0 to 1G onto PA 0x1_0000_0000 with a 1G block (level 1).
        let mut cfg = s2_cfg();
        cfg.stage = Stage::Nested;
        cfg.asid = 5;
        // The CD decode has already put TTB0 through stage 2, so it holds a PA.
        cfg.tt[0] =
            TransTableInfo { ttb: 0x1_0001_0000, tsz: 16, granule_sz: 12, ..Default::default() };
        cfg.tt[1].disabled = true;
        m.set(0x20000, 0x1_0000_0000 | AF | (3 << 6) | 1);
        // The next stage 1 tables are at IPAs 0x11000 and up, so at PA 0x1_0001_1000 and up.
        let iova = 0x1000u64;
        m.set(0x1_0001_0000, 0x11000 | 3);
        m.set(0x1_0001_1000, 0x12000 | 3);
        m.set(0x1_0001_2000, 0x13000 | 3);
        m.set(0x1_0001_3000 + 8, 0x5000 | AF | 3);
        let mut tlb = Iotlb::new();
        let mut info = PtwEventInfo::default();
        let e = smmu_translate(&mut tlb, &m, &cfg, iova + 4, PERM_WO, &mut info).unwrap();
        assert_eq!(e.to_addr(iova + 4), 0x1_0000_5004);
        assert_eq!((e.perm, e.parent_perm, e.addr_mask), (PERM_RW, PERM_RW, 0xfff));
        // Stage 2 entries for the table walk and the stage 1 entry are both cached.
        assert!(tlb.contains(&IotlbKey { iova: 0, asid: -1, vmid: 3, tg: 1, level: 1 }));
        assert!(tlb.contains(&IotlbKey { iova: 0x1000, asid: 5, vmid: 3, tg: 1, level: 3 }));
        tlb.inv_vmid_s1(3);
        assert_eq!(tlb.len(), 1);
        tlb.inv_ipa(3, 0x10000, 0, 1, 0);
        assert!(tlb.is_empty());
    }

    #[test]
    fn iotlb_invalidations() {
        let mut tlb = Iotlb::new();
        let mut cfg = s1_cfg();
        let page =
            |iova| TlbEntry { iova, addr_mask: 0xfff, level: 3, granule: 12, ..Default::default() };
        for asid in 1..4 {
            cfg.asid = asid;
            tlb.insert(&cfg, page(0x1000));
            tlb.insert(&cfg, page(0x2000));
        }
        tlb.inv_asid_vmid(2, -1);
        assert_eq!(tlb.len(), 4);
        // An exact key with TTL.
        tlb.inv_iova(1, -1, 0x2000, 1, 1, 3);
        assert_eq!(tlb.len(), 3);
        // A range of two pages from 0 covers 0x1000 of every ASID.
        tlb.inv_iova(-1, -1, 0, 0, 2, 0);
        assert_eq!(tlb.len(), 1);
        // The table empties when it is full.
        for i in 0..SMMU_IOTLB_MAX_SIZE as u64 {
            tlb.insert(&cfg, page(0x10_0000 + i * 0x1000));
        }
        assert_eq!(tlb.len(), 1);
    }
}
