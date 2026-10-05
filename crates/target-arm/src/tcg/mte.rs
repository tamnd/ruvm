// SPDX-License-Identifier: GPL-2.0-or-later

//! FEAT_MTE, QEMU's `mte_helper.c`: the allocation tag storage, the tag instructions and the
//! tag checks of loads and stores.
//!
//! The board gives the CPU its tag memory with [`Arm::with_tag_memory`](super::Arm), the
//! virt board's `mte=on`: one nibble per 16 byte granule of RAM, two granules to a byte with
//! the even granule in the low nibble, which is QEMU's layout of the tag address space. QEMU
//! finds the memory type of a page in the TLB entry (`pte_attrs`); this port walks the page
//! tables again for it, as the TLB entries here carry no target data. The helpers raise
//! their faults by unwinding to the instruction, as QEMU's do with `GETPC()`.
//!
//! Not ported: FEAT_MTE_NO_ADDRESS_TAGS and FEAT_MTE_CANONICAL_TAGS (MTX), FEAT_MTE_PERM,
//! FEAT_MTE_STORE_ONLY, FEAT_MTE_TAGGED_FAR, the tag checks of the SVE and MOPS accesses, the
//! stage 2 memory attributes, and Secure tag memory; the ID registers say so.

use std::sync::atomic::{AtomicU8, Ordering};

use ruvm_jit::cputlb::probe_access;
use ruvm_jit::{Cpu, CpuLoopExit, MmuAccessType, Ra};
use ruvm_jit_core::HelperType::{I32, I64, Ptr, Void};
use ruvm_jit_interp::{HelperEnv, Unwind};

use super::helpers::{Def, def, raise_exception, run};
use super::ptw::{self, Fault};
use super::{Arm, arm_of, exception_target_el, regime_el, regime_has_2_ranges};
use crate::cpu::{CpuArmState, EXCP_DATA_ABORT, MMU_IDX_E10_0, MMU_IDX_E20_0, RGSR_EL1, TFSR_EL};
use crate::syndrome::{fsc, syn_data_abort_no_iss};

type R<T> = Result<T, CpuLoopExit>;

/// `LOG2_TAG_GRANULE`.
pub(crate) const LOG2_TAG_GRANULE: u32 = 4;
/// `TAG_GRANULE`: the bytes covered by one allocation tag.
pub(crate) const TAG_GRANULE: u64 = 1 << LOG2_TAG_GRANULE;
/// `GMID_EL1.BS` of `max`, `gm_blocksize`: LDGM and STGM move 256 bytes of tags.
pub(crate) const GM_BLOCKSIZE: u32 = 6;

/// `MTEDESC.MIDX`: the core MMU index, bits 0 to 3.
pub(crate) const MTEDESC_MIDX_SHIFT: u32 = 0;
/// `MTEDESC.TBI`: the two TBI bits of the regime, bits 4 and 5.
pub(crate) const MTEDESC_TBI_SHIFT: u32 = 4;
/// `MTEDESC.TCMA`: the two TCMA bits of the regime, bits 6 and 7.
pub(crate) const MTEDESC_TCMA_SHIFT: u32 = 6;
/// `MTEDESC.WRITE`: the access is a store.
pub(crate) const MTEDESC_WRITE: u32 = 1 << 8;
/// `MTEDESC.ALIGN`: log2 of the alignment the access needs, bits 9 to 11.
pub(crate) const MTEDESC_ALIGN_SHIFT: u32 = 9;
/// `MTEDESC.SIZEM1`: the size of the access minus one, from bit 14.
pub(crate) const MTEDESC_SIZEM1_SHIFT: u32 = 14;

/// Allocation tag storage for a range of RAM: QEMU's tag RAM (`mach-virt.tag`), one byte
/// per 32 bytes of RAM.
#[derive(Debug)]
pub struct TagMemory {
    base: u64,
    size: u64,
    tags: Box<[AtomicU8]>,
}

impl TagMemory {
    /// Tag storage for the `size` bytes of RAM at `base`, all tags zero.
    pub fn new(base: u64, size: u64) -> TagMemory {
        let n = size.div_ceil(2 * TAG_GRANULE) as usize;
        TagMemory { base, size, tags: (0..n).map(|_| AtomicU8::new(0)).collect() }
    }

    /// The tag byte of the physical address `pa`, if the RAM has tag storage. This is
    /// `address_space_translate()` in the tag address space finding RAM.
    fn index(&self, pa: u64) -> Option<usize> {
        let off = pa.checked_sub(self.base).filter(|&o| o < self.size)?;
        Some((off >> (LOG2_TAG_GRANULE + 1)) as usize)
    }

    fn byte(&self, i: usize) -> &AtomicU8 {
        &self.tags[i]
    }

    /// The allocation tag of the granule holding `pa`, for tests and debuggers.
    pub fn tag(&self, pa: u64) -> Option<u8> {
        let i = self.index(pa)?;
        Some(load_tag1(pa, self.tags[i].load(Ordering::Relaxed)))
    }
}

/// `allocation_tag_from_addr()`.
fn allocation_tag_from_addr(ptr: u64) -> u32 {
    ((ptr >> 56) & 0xf) as u32
}

/// `address_with_allocation_tag()`.
fn address_with_allocation_tag(ptr: u64, rtag: u32) -> u64 {
    (ptr & !(0xf << 56)) | (u64::from(rtag & 0xf) << 56)
}

/// `choose_nonexcluded_tag()`.
fn choose_nonexcluded_tag(mut tag: u32, mut offset: u32, exclude: u32) -> u32 {
    if exclude == 0xffff {
        return 0;
    }
    if offset == 0 {
        while exclude & (1 << tag) != 0 {
            tag = (tag + 1) & 15;
        }
    } else {
        loop {
            loop {
                tag = (tag + 1) & 15;
                if exclude & (1 << tag) == 0 {
                    break;
                }
            }
            offset -= 1;
            if offset == 0 {
                break;
            }
        }
    }
    tag
}

/// `load_tag1()`: the nibble of the granule of `ptr` in the tag byte `mem`.
fn load_tag1(ptr: u64, mem: u8) -> u8 {
    let ofs = ((ptr >> LOG2_TAG_GRANULE) & 1) * 4;
    (mem >> ofs) & 0xf
}

/// `store_tag1_parallel()`: atomically store `tag` to the nibble of the granule of `ptr`.
fn store_tag1(ptr: u64, mem: &AtomicU8, tag: u32) {
    let ofs = ((ptr >> LOG2_TAG_GRANULE) & 1) * 4;
    let tag = (tag & 0xf) as u8;
    let _ = mem.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
        Some((old & !(0xf << ofs)) | (tag << ofs))
    });
}

/// `tbi_or_mtx_check()`; MTX is never set here.
fn tbi_check(desc: u32, bit55: u32) -> bool {
    desc & ((1 << MTEDESC_TBI_SHIFT) << bit55) != 0
}

/// `tag_is_canonical()`.
fn tag_is_canonical(ptr_tag: u32, bit55: u32) -> bool {
    (ptr_tag + bit55) & 0xf == 0
}

/// `tcma_check()`.
fn tcma_check(desc: u32, bit55: u32, ptr_tag: u32) -> bool {
    let tcma = (desc >> (MTEDESC_TCMA_SHIFT + bit55)) & 1 != 0;
    tcma && tag_is_canonical(ptr_tag, bit55)
}

/// The address `ptr` with the top byte cleared as TBI does for an access through
/// `mmu_idx`, so that the probes below fill the TLB for the address the access uses.
fn clean_ptr(st: &CpuArmState, mmu_idx: usize, ptr: u64) -> u64 {
    let (tbi, _) = super::tbi_bits(st.tcr_el[regime_el(mmu_idx) as usize], mmu_idx);
    if regime_has_2_ranges(mmu_idx) {
        let bit55 = (ptr >> 55) & 1;
        if (tbi >> bit55) & 1 != 0 { ((ptr << 8) as i64 >> 8) as u64 } else { ptr }
    } else if tbi != 0 {
        ptr & ((1 << 56) - 1)
    } else {
        ptr
    }
}

/// A helper's view of the CPU for MTE: the [`Arm`], the system state and the tag memory.
struct Ctx {
    st: CpuArmState,
    tags: Option<std::sync::Arc<TagMemory>>,
}

impl Ctx {
    fn new(cpu: &Cpu<'_>, arm: &Arm) -> Ctx {
        Ctx { st: CpuArmState::load_system(cpu.env), tags: arm.tag_memory().cloned() }
    }
}

/// `allocation_tag_mem()`: the index of the tag byte of `ptr` in the tag memory, after
/// raising any fault of an access of `ptr_size` bytes at `ptr`, or `None` if the page is
/// not Tagged Normal memory or has no tag storage.
#[allow(clippy::too_many_arguments)]
fn allocation_tag_mem(
    cpu: &mut Cpu<'_>,
    arm: &Arm,
    c: &Ctx,
    mmu_idx: usize,
    ptr: u64,
    ptr_access: MmuAccessType,
    ptr_size: u64,
) -> R<Option<usize>> {
    let clean = clean_ptr(&c.st, mmu_idx, ptr);
    // Probe the first byte of the virtual address. This raises an exception for
    // inaccessible pages.
    probe_access(cpu, clean, 0, ptr_access, mmu_idx, Ra::Tb)?;
    let Ok(t) = ptw::get_phys_addr(arm, cpu, clean, ptr_access, mmu_idx, false, false) else {
        return Ok(None);
    };
    if t.attrs != 0xf0 {
        // Not Tagged.
        return Ok(None);
    }
    let ptr_paddr = t.pa | (clean & 0xfff);
    // The Normal memory access can extend to the next page. E.g. a single 8-byte access to
    // the last byte of a page will check only the last tag on the first page. Any page
    // access exception has priority over tag check exception.
    let in_page = 0x1000 - (clean & 0xfff);
    if ptr_size > in_page {
        probe_access(cpu, clean.wrapping_add(in_page), 0, ptr_access, mmu_idx, Ra::Tb)?;
    }
    // If the address has no tag storage the access is unchecked; QEMU logs this as a board
    // configuration error.
    Ok(c.tags.as_ref().and_then(|tm| tm.index(ptr_paddr)))
}

/// The tag memory of a helper that found a tag byte, which only happens with tag memory.
fn tm(c: &Ctx) -> &TagMemory {
    c.tags.as_deref().expect("a tag byte comes from tag memory")
}

/// `check_tag_aligned()`.
fn check_tag_aligned(cpu: &mut Cpu<'_>, arm: &Arm, ptr: u64) -> R<()> {
    if ptr % TAG_GRANULE != 0 {
        return Err(ptw::deliver_fault(
            arm,
            cpu,
            ptr,
            MmuAccessType::DataStore,
            Fault::new(fsc::ALIGNMENT),
            Ra::Tb,
        ));
    }
    Ok(())
}

def!(IRG, "irg", 0, I64, [Ptr, I64, I64], h_irg);
def!(ADDSUBG, "addsubg", 0, I64, [Ptr, I64, I32, I32], h_addsubg);
def!(LDG, "ldg", 0, I64, [Ptr, I64, I64], h_ldg);
def!(STG, "stg", 0, Void, [Ptr, I64, I64], h_stg);
def!(STG_STUB, "stg_stub", 0, Void, [Ptr, I64], h_stg_stub);
def!(ST2G, "st2g", 0, Void, [Ptr, I64, I64], h_st2g);
def!(ST2G_STUB, "st2g_stub", 0, Void, [Ptr, I64], h_st2g_stub);
def!(LDGM, "ldgm", 0, I64, [Ptr, I64], h_ldgm);
def!(STGM, "stgm", 0, Void, [Ptr, I64, I64], h_stgm);
def!(STZGM_TAGS, "stzgm_tags", 0, Void, [Ptr, I64, I64], h_stzgm_tags);
def!(MTE_CHECK, "mte_check", 0, I64, [Ptr, I32, I64], h_mte_check);
def!(MTE_CHECK_ZVA, "mte_check_zva", 0, I64, [Ptr, I32, I64], h_mte_check_zva);
def!(PROBE_ACCESS, "probe_access", 0, Void, [Ptr, I64, I32, I32, I32], h_probe_access);

/// Every MTE helper.
pub(crate) const ALL: &[Def] = &[
    IRG,
    ADDSUBG,
    LDG,
    STG,
    STG_STUB,
    ST2G,
    ST2G_STUB,
    LDGM,
    STGM,
    STZGM_TAGS,
    MTE_CHECK,
    MTE_CHECK_ZVA,
    PROBE_ACCESS,
];

/// The core MMU index of the current regime.
fn mmu_index(cpu: &Cpu<'_>) -> usize {
    ((super::ld32(cpu.env, crate::cpu::HFLAGS) >> super::TB_MMUIDX_SHIFT) & 0xf) as usize
}

/// `HELPER(irg)`.
fn h_irg(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let st = CpuArmState::load_system(cpu.env);
        let (rn, rm) = (a[1], a[2]);
        let exclude = ((rm | st.gcr_el1) & 0xffff) as u32;
        let rrnd = (st.gcr_el1 >> 16) & 1 != 0;
        let start = (st.rgsr_el1 & 0xf) as u32;
        let mut seed = ((st.rgsr_el1 >> 8) & 0xffff) as u32;
        // Our IMPDEF choice for GCR_EL1.RRND==1 is to continue to use the deterministic
        // algorithm. Except that with RRND==1 the kernel is not required to have set
        // RGSR_EL1.SEED != 0, which is required for the deterministic algorithm to
        // function. So we force a non-zero SEED for that case.
        if seed == 0 && rrnd {
            seed = random_seed();
        }
        // RandomTag.
        let mut offset = 0;
        for i in 0..4 {
            // NextRandomTagBit.
            let top = ((seed >> 5) ^ (seed >> 3) ^ (seed >> 2) ^ seed) & 1;
            seed = (top << 15) | (seed >> 1);
            offset |= top << i;
        }
        let rtag = choose_nonexcluded_tag(start, offset, exclude);
        let rgsr = u64::from(rtag) | (u64::from(seed) << 8);
        super::st64(cpu.env, RGSR_EL1, rgsr);
        Ok(address_with_allocation_tag(rn, rtag))
    })
}

/// A non-zero 16 bit seed, `qemu_guest_getrandom()`.
fn random_seed() -> u32 {
    use std::hash::{BuildHasher, Hasher};
    loop {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(0);
        let seed = (h.finish() & 0xffff) as u32;
        if seed != 0 {
            return seed;
        }
    }
}

/// `HELPER(addsubg)`.
fn h_addsubg(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let gcr = CpuArmState::load_system(cpu.env).gcr_el1;
        let (ptr, offset, tag_offset) = (a[1], a[2] as i32, a[3] as u32);
        let start_tag = allocation_tag_from_addr(ptr);
        let exclude = (gcr & 0xffff) as u32;
        let rtag = choose_nonexcluded_tag(start_tag, tag_offset, exclude);
        Ok(address_with_allocation_tag(ptr.wrapping_add(offset as i64 as u64), rtag))
    })
}

/// `HELPER(ldg)`.
fn h_ldg(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let ops = cpu.ops();
        let arm = arm_of(&ops);
        let c = Ctx::new(cpu, arm);
        let (ptr, xt) = (a[1], a[2]);
        let mmu_idx = mmu_index(cpu);
        // Trap if accessing an invalid page.
        let mem = allocation_tag_mem(cpu, arm, &c, mmu_idx, ptr, MmuAccessType::DataLoad, 1)?;
        // Load if page supports tags.
        let rtag = mem.map_or(0, |i| load_tag1(ptr, tm(&c).byte(i).load(Ordering::Relaxed)));
        Ok(address_with_allocation_tag(xt, u32::from(rtag)))
    })
}

/// `HELPER(stg)` and `HELPER(stg_parallel)`; the nibble is always stored atomically.
fn h_stg(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let ops = cpu.ops();
        let arm = arm_of(&ops);
        let c = Ctx::new(cpu, arm);
        let (ptr, xt) = (a[1], a[2]);
        let mmu_idx = mmu_index(cpu);
        check_tag_aligned(cpu, arm, ptr)?;
        // Trap if accessing an invalid page.
        let mem =
            allocation_tag_mem(cpu, arm, &c, mmu_idx, ptr, MmuAccessType::DataStore, TAG_GRANULE)?;
        // Store if page supports tags.
        if let Some(i) = mem {
            store_tag1(ptr, tm(&c).byte(i), allocation_tag_from_addr(xt));
        }
        Ok(0)
    })
}

/// `probe_write()` of `size` bytes, which must be on one page.
fn probe_write(cpu: &mut Cpu<'_>, c: &Ctx, ptr: u64, size: usize, mmu_idx: usize) -> R<()> {
    let clean = clean_ptr(&c.st, mmu_idx, ptr);
    probe_access(cpu, clean, size, MmuAccessType::DataStore, mmu_idx, Ra::Tb).map(|_| ())
}

/// `HELPER(stg_stub)`: STG without tag access, which still checks the alignment and the
/// access.
fn h_stg_stub(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let ops = cpu.ops();
        let arm = arm_of(&ops);
        let c = Ctx::new(cpu, arm);
        let ptr = a[1];
        check_tag_aligned(cpu, arm, ptr)?;
        probe_write(cpu, &c, ptr, TAG_GRANULE as usize, mmu_index(cpu))?;
        Ok(0)
    })
}

/// `HELPER(st2g)` and `HELPER(st2g_parallel)`.
fn h_st2g(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let ops = cpu.ops();
        let arm = arm_of(&ops);
        let c = Ctx::new(cpu, arm);
        let (ptr, xt) = (a[1], a[2]);
        let mmu_idx = mmu_index(cpu);
        let tag = allocation_tag_from_addr(xt);
        let at = MmuAccessType::DataStore;
        check_tag_aligned(cpu, arm, ptr)?;
        // Trap if accessing an invalid page(s). This takes priority over
        // !allocation_tag_access_enabled.
        if ptr & TAG_GRANULE != 0 {
            // Two stores unaligned mod TAG_GRANULE*2 -- modify two bytes.
            let mem1 = allocation_tag_mem(cpu, arm, &c, mmu_idx, ptr, at, TAG_GRANULE)?;
            let ptr2 = ptr.wrapping_add(TAG_GRANULE);
            let mem2 = allocation_tag_mem(cpu, arm, &c, mmu_idx, ptr2, at, TAG_GRANULE)?;
            // Store if page(s) support tags.
            if let Some(i) = mem1 {
                store_tag1(TAG_GRANULE, tm(&c).byte(i), tag);
            }
            if let Some(i) = mem2 {
                store_tag1(0, tm(&c).byte(i), tag);
            }
        } else {
            // Two stores aligned mod TAG_GRANULE*2 -- modify one byte.
            let mem1 = allocation_tag_mem(cpu, arm, &c, mmu_idx, ptr, at, 2 * TAG_GRANULE)?;
            if let Some(i) = mem1 {
                tm(&c).byte(i).store((tag | (tag << 4)) as u8, Ordering::Relaxed);
            }
        }
        Ok(0)
    })
}

/// `HELPER(st2g_stub)`.
fn h_st2g_stub(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let ops = cpu.ops();
        let arm = arm_of(&ops);
        let c = Ctx::new(cpu, arm);
        let ptr = a[1];
        let mmu_idx = mmu_index(cpu);
        check_tag_aligned(cpu, arm, ptr)?;
        let in_page = 0x1000 - (ptr & 0xfff);
        if in_page >= 2 * TAG_GRANULE {
            probe_write(cpu, &c, ptr, 2 * TAG_GRANULE as usize, mmu_idx)?;
        } else {
            probe_write(cpu, &c, ptr, TAG_GRANULE as usize, mmu_idx)?;
            probe_write(cpu, &c, ptr.wrapping_add(TAG_GRANULE), TAG_GRANULE as usize, mmu_idx)?;
        }
        Ok(0)
    })
}

/// The bytes LDGM and STGM cover, `4 << gm_blocksize`.
const GM_BS_BYTES: u64 = 4 << GM_BLOCKSIZE;

/// The 8 tag bytes of a 256 byte LDGM or STGM block, little endian.
fn gm_bytes(tm: &TagMemory, i: usize) -> &[AtomicU8] {
    &tm.tags[i..i + (GM_BS_BYTES >> (LOG2_TAG_GRANULE + 1)) as usize]
}

/// `HELPER(ldgm)`.
fn h_ldgm(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let ops = cpu.ops();
        let arm = arm_of(&ops);
        let c = Ctx::new(cpu, arm);
        let ptr = a[1] & !(GM_BS_BYTES - 1);
        let mmu_idx = mmu_index(cpu);
        // Trap if accessing an invalid page.
        let at = MmuAccessType::DataLoad;
        let mem = allocation_tag_mem(cpu, arm, &c, mmu_idx, ptr, at, GM_BS_BYTES)?;
        // The tag is squashed to zero if the page does not support tags. The ordering of
        // elements within the word corresponds to a little-endian operation; with BS=6 the
        // block is a whole tag word.
        Ok(mem.map_or(0, |i| {
            gm_bytes(tm(&c), i)
                .iter()
                .enumerate()
                .fold(0, |v, (n, b)| v | (u64::from(b.load(Ordering::Relaxed)) << (8 * n)))
        }))
    })
}

/// `HELPER(stgm)`.
fn h_stgm(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let ops = cpu.ops();
        let arm = arm_of(&ops);
        let c = Ctx::new(cpu, arm);
        let ptr = a[1] & !(GM_BS_BYTES - 1);
        let val = a[2];
        let mmu_idx = mmu_index(cpu);
        // Trap if accessing an invalid page.
        let at = MmuAccessType::DataStore;
        let mem = allocation_tag_mem(cpu, arm, &c, mmu_idx, ptr, at, GM_BS_BYTES)?;
        // Tag store only happens if the page support tags, and if the OS has enabled access
        // to the tags.
        if let Some(i) = mem {
            for (n, b) in gm_bytes(tm(&c), i).iter().enumerate() {
                b.store((val >> (8 * n)) as u8, Ordering::Relaxed);
            }
        }
        Ok(0)
    })
}

/// log2 of the DC ZVA block size in bytes.
fn log2_dcz_bytes(arm: &Arm) -> u32 {
    (arm.model().dczid & 0xf) as u32 + 2
}

/// `HELPER(stzgm_tags)`.
fn h_stzgm_tags(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let ops = cpu.ops();
        let arm = arm_of(&ops);
        let c = Ctx::new(cpu, arm);
        let mmu_idx = mmu_index(cpu);
        // The DC ZVA block size is at least 2 * TAG_GRANULE, so whole tag bytes are set.
        let log2_dcz = log2_dcz_bytes(arm);
        let dcz_bytes = 1u64 << log2_dcz;
        let tag_bytes = 1usize << (log2_dcz - (LOG2_TAG_GRANULE + 1));
        let ptr = a[1] & !(dcz_bytes - 1);
        let at = MmuAccessType::DataStore;
        let mem = allocation_tag_mem(cpu, arm, &c, mmu_idx, ptr, at, dcz_bytes)?;
        if let Some(i) = mem {
            let tag_pair = ((a[2] & 0xf) * 0x11) as u8;
            for b in &tm(&c).tags[i..i + tag_bytes] {
                b.store(tag_pair, Ordering::Relaxed);
            }
        }
        Ok(0)
    })
}

/// `mte_check_fail()`: record a tag check failure at `dirty_ptr`.
fn mte_check_fail(cpu: &mut Cpu<'_>, desc: u32, dirty_ptr: u64) -> R<()> {
    let mmu_idx = ((desc >> MTEDESC_MIDX_SHIFT) & 0xf) as usize;
    let reg_el = regime_el(mmu_idx);
    let st = CpuArmState::load_system(cpu.env);
    let sctlr = st.sctlr_el[reg_el as usize];
    let (el, tcf) = match mmu_idx {
        MMU_IDX_E10_0 | MMU_IDX_E20_0 => (0, (sctlr >> 38) & 3),
        _ => (reg_el, (sctlr >> 40) & 3),
    };
    let is_write = desc & MTEDESC_WRITE != 0;
    // TCF 0 never gets here: MTE_ACTIVE is clear. 1 is a synchronous exception, 2 sets the
    // asynchronous flag, and 3 is asynchronous for stores and synchronous for loads.
    if tcf == 1 || (tcf == 3 && !is_write) {
        let mut st = CpuArmState::load(cpu.env);
        st.exception_vaddress = dirty_ptr;
        let syn = syn_data_abort_no_iss(st.current_el() != 0, false, false, is_write, 0x11);
        let target_el = exception_target_el(&st);
        super::commit(cpu, &mut st);
        return Err(raise_exception(cpu, EXCP_DATA_ABORT, syn, target_el, Ra::Tb));
    }
    // mte_async_check_fail().
    let select = if regime_has_2_ranges(mmu_idx) { (dirty_ptr >> 55) & 1 } else { 0 };
    let off = TFSR_EL + 8 * el as usize;
    let v = super::ld64(cpu.env, off) | (1 << select);
    super::st64(cpu.env, off, v);
    Ok(())
}

/// `checkN()`: the number of the `count` tags from the tag byte `i` (from its odd nibble if
/// `odd`) that match `cmp`.
fn check_n(tm: &TagMemory, mut i: usize, odd: bool, cmp: u32, count: u64) -> u64 {
    let cmp = (cmp * 0x11) as u8;
    let mut n = 0;
    let mut odd = odd;
    loop {
        let diff = tm.tags[i].load(Ordering::Relaxed) ^ cmp;
        if !odd {
            // Test even tag.
            if diff & 0x0f != 0 {
                return n;
            }
            n += 1;
            if n == count {
                return n;
            }
        }
        // Test odd tag.
        if diff & 0xf0 != 0 {
            return n;
        }
        n += 1;
        if n == count {
            return n;
        }
        odd = false;
        i += 1;
    }
}

/// `mte_probe_int()`: `Ok(None)` when the check passes, `Ok(Some(fault))` with the address
/// of the first failing granule.
fn mte_probe_int(cpu: &mut Cpu<'_>, arm: &Arm, c: &Ctx, desc: u32, ptr: u64) -> R<Option<u64>> {
    let bit55 = ((ptr >> 55) & 1) as u32;
    // If TBI is disabled, the access is unchecked.
    if !tbi_check(desc, bit55) {
        return Ok(None);
    }
    let ptr_tag = allocation_tag_from_addr(ptr);
    if tcma_check(desc, bit55, ptr_tag) {
        return Ok(None);
    }
    let mmu_idx = ((desc >> MTEDESC_MIDX_SHIFT) & 0xf) as usize;
    let at =
        if desc & MTEDESC_WRITE != 0 { MmuAccessType::DataStore } else { MmuAccessType::DataLoad };
    let sizem1 = u64::from(desc >> MTEDESC_SIZEM1_SHIFT);
    // Find the addr of the end of the access.
    let ptr_last = ptr.wrapping_add(sizem1);
    // Round the bounds to the tag granule, and compute the number of tags.
    let tag_first = ptr & !(TAG_GRANULE - 1);
    let tag_last = ptr_last & !(TAG_GRANULE - 1);
    let tag_count = (tag_last.wrapping_sub(tag_first) / TAG_GRANULE) + 1;
    // Locate the page boundaries.
    let prev_page = ptr & !0xfff;
    let next_page = prev_page.wrapping_add(0x1000);
    let odd = ptr & TAG_GRANULE != 0;
    let n = if tag_last.wrapping_sub(prev_page) < 0x1000 {
        // Memory access stays on one page.
        let Some(i) = allocation_tag_mem(cpu, arm, c, mmu_idx, ptr, at, sizem1 + 1)? else {
            // Untagged.
            return Ok(None);
        };
        check_n(tm(c), i, odd, ptr_tag, tag_count)
    } else {
        // Memory access crosses to next page.
        let mem1 = allocation_tag_mem(cpu, arm, c, mmu_idx, ptr, at, next_page - ptr)?;
        let size2 = ptr_last.wrapping_sub(next_page) + 1;
        let mem2 = allocation_tag_mem(cpu, arm, c, mmu_idx, next_page, at, size2)?;
        // Perform all of the comparisons. Note the possible but unlikely case of the
        // operation spanning two pages that do not both have allocation tagging enabled.
        let cnt = (next_page - tag_first) / TAG_GRANULE;
        let mut n = cnt;
        if let Some(i) = mem1 {
            n = check_n(tm(c), i, odd, ptr_tag, cnt);
        }
        if n == cnt {
            match mem2 {
                Some(i) => n += check_n(tm(c), i, false, ptr_tag, tag_count - cnt),
                None => return Ok(None),
            }
        }
        n
    };
    if n == tag_count {
        return Ok(None);
    }
    // If we failed, we know which granule. For the first granule, the failure address is
    // @ptr, the first byte accessed. Otherwise the failure address is the first byte of the
    // nth granule.
    Ok(Some(if n > 0 { tag_first + n * TAG_GRANULE } else { ptr }))
}

/// `HELPER(mte_check)`: check the tags of an access and return its address with the top
/// byte cleaned, as the translator's `clean_data_tbi()` would.
fn h_mte_check(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let ops = cpu.ops();
        let arm = arm_of(&ops);
        let (desc, ptr) = (a[1] as u32, a[2]);
        let mmu_idx = ((desc >> MTEDESC_MIDX_SHIFT) & 0xf) as usize;
        let write = desc & MTEDESC_WRITE != 0;
        // R_XCHFJ: Alignment check not caused by memory type is priority 1, higher than any
        // translation fault. When MTE is disabled, the generated code performs the
        // alignment check during the memory access. With MTE enabled, we must check this
        // here before raising any translation fault in allocation_tag_mem.
        let align = (desc >> MTEDESC_ALIGN_SHIFT) & 7;
        if align != 0 && ptr & ((1 << align) - 1) != 0 {
            let at = if write { MmuAccessType::DataStore } else { MmuAccessType::DataLoad };
            return Err(ptw::deliver_fault(arm, cpu, ptr, at, Fault::new(fsc::ALIGNMENT), Ra::Tb));
        }
        let c = Ctx::new(cpu, arm);
        if let Some(fault) = mte_probe_int(cpu, arm, &c, desc, ptr)? {
            mte_check_fail(cpu, desc, fault)?;
        }
        Ok(clean_ptr(&c.st, mmu_idx, ptr))
    })
}

/// `HELPER(mte_check_zva)`: the tag check of DC ZVA, which returns the address cleaned.
fn h_mte_check_zva(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let ops = cpu.ops();
        let arm = arm_of(&ops);
        let (desc, ptr) = (a[1] as u32, a[2]);
        let mmu_idx = ((desc >> MTEDESC_MIDX_SHIFT) & 0xf) as usize;
        let c = Ctx::new(cpu, arm);
        let clean = clean_ptr(&c.st, mmu_idx, ptr);
        let bit55 = ((ptr >> 55) & 1) as u32;
        // If TBI is disabled, the access is unchecked.
        if !tbi_check(desc, bit55) {
            return Ok(clean);
        }
        let ptr_tag = allocation_tag_from_addr(ptr);
        if tcma_check(desc, bit55, ptr_tag) {
            return Ok(clean);
        }
        let log2_dcz = log2_dcz_bytes(arm);
        let dcz_bytes = 1u64 << log2_dcz;
        let tag_bytes = 1usize << (log2_dcz - (LOG2_TAG_GRANULE + 1));
        let align_ptr = ptr & !(dcz_bytes - 1);
        // Trap if accessing an invalid page. DC_ZVA requires that we supply the original
        // pointer for an invalid page.
        probe_write(cpu, &c, ptr, 1, mmu_idx)?;
        let at = MmuAccessType::DataStore;
        let Some(i) = allocation_tag_mem(cpu, arm, &c, mmu_idx, align_ptr, at, dcz_bytes)? else {
            return Ok(clean);
        };
        // DC ZVA is always aligned, so every tag byte of the block holds two tags of it.
        let want = (ptr_tag * 0x11) as u8;
        let tags = &tm(&c).tags[i..i + tag_bytes];
        if let Some((n, b)) =
            tags.iter().enumerate().find(|(_, b)| b.load(Ordering::Relaxed) != want)
        {
            // Locate the first nibble that differs.
            let diff = b.load(Ordering::Relaxed) ^ want;
            let nib = if diff & 0xf != 0 { 0 } else { 1 };
            mte_check_fail(cpu, desc, align_ptr + (2 * n as u64 + nib) * TAG_GRANULE)?;
        }
        Ok(clean)
    })
}

/// `HELPER(probe_access)`: fault an access of `size` bytes, which may cross a page.
fn h_probe_access(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let (ptr, mmu_idx, size) = (a[1], a[3] as usize, a[4] as usize);
        let at = if a[2] == 1 { MmuAccessType::DataStore } else { MmuAccessType::DataLoad };
        let in_page = (0x1000 - (ptr & 0xfff)) as usize;
        if size <= in_page {
            probe_access(cpu, ptr, size, at, mmu_idx, Ra::Tb)?;
        } else {
            probe_access(cpu, ptr, in_page, at, mmu_idx, Ra::Tb)?;
            let next = ptr.wrapping_add(in_page as u64);
            probe_access(cpu, next, size - in_page, at, mmu_idx, Ra::Tb)?;
        }
        Ok(0)
    })
}

/// `allocation_tag_access_enabled()`.
pub(crate) fn allocation_tag_access_enabled(
    f: &crate::cpu::ArmFeatures,
    st: &CpuArmState,
    el: u32,
    sctlr: u64,
) -> bool {
    use crate::cpu::{HCR_ATA, HCR_E2H, HCR_TGE, SCR_ATA, SCTLR_ATA, SCTLR_ATA0};
    if el < 3 && f.el3 && st.scr_el3 & SCR_ATA == 0 {
        return false;
    }
    if el < 2 && st.is_el2_enabled(f) {
        let hcr = st.hcr_el2_eff(f);
        if hcr & HCR_ATA == 0 && (hcr & HCR_E2H == 0 || hcr & HCR_TGE == 0) {
            return false;
        }
    }
    sctlr & (if el == 0 { SCTLR_ATA0 } else { SCTLR_ATA }) != 0
}

#[cfg(test)]
mod tests {
    use super::{TagMemory, check_n, choose_nonexcluded_tag};

    #[test]
    fn nonexcluded_tags() {
        assert_eq!(choose_nonexcluded_tag(0, 0, 0), 0);
        assert_eq!(choose_nonexcluded_tag(0, 0, 1), 1);
        assert_eq!(choose_nonexcluded_tag(3, 2, 0), 5);
        assert_eq!(choose_nonexcluded_tag(15, 1, 1), 1);
        assert_eq!(choose_nonexcluded_tag(5, 3, 0xffff), 0);
    }

    #[test]
    fn tag_layout_and_check() {
        let tm = TagMemory::new(0x4000_0000, 0x1000);
        // Granule 0 in the low nibble, granule 1 in the high nibble.
        tm.tags[0].store(0x21, std::sync::atomic::Ordering::Relaxed);
        tm.tags[1].store(0x22, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(tm.tag(0x4000_0000), Some(1));
        assert_eq!(tm.tag(0x4000_0010), Some(2));
        assert_eq!(tm.tag(0x4000_0020), Some(2));
        assert_eq!(tm.tag(0x3fff_fff0), None);
        assert_eq!(tm.tag(0x4000_1000), None);
        assert_eq!(check_n(&tm, 0, true, 2, 3), 3);
        assert_eq!(check_n(&tm, 0, false, 2, 3), 0);
        assert_eq!(check_n(&tm, 0, true, 2, 4), 3);
    }
}
