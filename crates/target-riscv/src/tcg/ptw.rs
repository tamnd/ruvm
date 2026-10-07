// SPDX-License-Identifier: GPL-2.0-or-later

//! The RISC-V page walk and TLB fill: `get_physical_address()`,
//! `get_physical_address_pmp()`, `riscv_cpu_tlb_fill()` and `raise_mmu_exception()` of
//! QEMU's `target/riscv/tcg/cpu_helper.c`, for Sv39, Sv48 or Sv57 translation with Svadu
//! and, with the H extension, the two stage translation of VS and VU mode: the VS stage
//! through `vsatp` and the G stage through `hgatp` (Sv39x4, Sv48x4 or Sv57x4). The model
//! has no Svpbmt, Svnapot or shadow stacks, so PTEs using them are invalid.

use ruvm_jit::{Cpu, CpuLoopExit, MmuAccessType, Ra, page};
use ruvm_mem::{AddressSpace, Endian, MemTxAttrs, RegionType};

use super::{pmp, st64};
use crate::cpu::{
    BADADDR, CpuRiscvState, EXCP_INST_ACCESS_FAULT, EXCP_INST_GUEST_PAGE_FAULT,
    EXCP_INST_PAGE_FAULT, EXCP_LOAD_ACCESS_FAULT, EXCP_LOAD_GUEST_ACCESS_FAULT,
    EXCP_LOAD_PAGE_FAULT, EXCP_STORE_AMO_ACCESS_FAULT, EXCP_STORE_GUEST_AMO_ACCESS_FAULT,
    EXCP_STORE_PAGE_FAULT, GUEST_PHYS_FAULT_ADDR, HGATP64_MODE, HGATP64_PPN, MENVCFG_ADUE,
    MMU_2STAGE_BIT, MMU_IDX_S_SUM, MMU_IDX_U, MSTATUS_MXR, PRV_M, PRV_S, PRV_U, PTE_A, PTE_ATTR,
    PTE_D, PTE_N, PTE_PBMT, PTE_PPN_MASK, PTE_PPN_SHIFT, PTE_R, PTE_RESERVED, PTE_U, PTE_V, PTE_W,
    PTE_X, SATP64_MODE, SATP64_PPN, TWO_STAGE_INDIRECT_LOOKUP, TWO_STAGE_LOOKUP, VM_MBARE, VM_SV39,
    VM_SV48, VM_SV57, get_field,
};

const PGSHIFT: u32 = 12;

/// Why a translation failed, `TRANSLATE_*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fail {
    /// `TRANSLATE_FAIL`: a page fault.
    Page,
    /// `TRANSLATE_PMP_FAIL` or `TRANSLATE_PMA_FAIL`: an access fault.
    Access,
    /// `TRANSLATE_G_STAGE_FAIL`: the G stage translation of a VS stage page table
    /// address failed.
    GStage,
}

/// A successful translation: the physical address and the page protection.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Translation {
    pub(crate) pa: u64,
    pub(crate) prot: u32,
}

/// `mmuidx_priv()`: the privilege level an MMU index translates for.
pub(crate) fn mmuidx_priv(mmu_idx: usize) -> u64 {
    let p = mmu_idx & 3;
    if p == MMU_IDX_S_SUM { PRV_S } else { p as u64 }
}

/// `mmuidx_sum()`: whether an MMU index is S mode with `mstatus.SUM`.
fn mmuidx_sum(mmu_idx: usize) -> bool {
    mmu_idx & 3 == MMU_IDX_S_SUM
}

/// `mmuidx_2stage()`: whether an MMU index translates in two stages.
pub(crate) fn mmuidx_2stage(mmu_idx: usize) -> bool {
    mmu_idx & MMU_2STAGE_BIT != 0
}

/// `get_physical_address_pmp()`: the PMP check of `size` bytes at `addr` for `access` in
/// privilege `mode`, giving the protection PMP allows.
pub(crate) fn pmp_check(
    st: &CpuRiscvState,
    addr: u64,
    size: u64,
    access: MmuAccessType,
    mode: u64,
) -> Result<u32, Fail> {
    let privs = 1u32 << access as u32;
    pmp::hart_has_privs(st, addr, size, privs, mode).ok_or(Fail::Access)
}

/// `memory_region_is_ram()` of what `addr` maps to.
fn is_ram(as_: &AddressSpace, addr: u64) -> bool {
    as_.flatview().lookup(addr).is_some_and(|r| r.region_type() == RegionType::Ram)
}

/// `get_physical_address()`: translate `addr` for one stage, with `is_debug` (no PTE
/// updates) when `debug` is set. `first_stage` selects the VS (or only) stage through
/// `satp` (or `vsatp` when two stage translation is forced from HS or M mode), else the G
/// stage through `hgatp`. With `two_stage` and `first_stage`, the page table addresses
/// are guest physical and go through the G stage; when that fails, `fault_pte_addr`
/// gets the guest physical address of the PTE shifted right by 2.
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_physical_address(
    st: &CpuRiscvState,
    as_: &AddressSpace,
    addr: u64,
    access: MmuAccessType,
    mmu_idx: usize,
    first_stage: bool,
    two_stage: bool,
    debug: bool,
    fault_pte_addr: Option<&mut u64>,
) -> Result<Translation, Fail> {
    let rwx = page::READ | page::WRITE | page::EXEC;
    let mode = mmuidx_priv(mmu_idx);
    // The background registers serve a two stage translation forced on from HS or M mode
    // (MPRV with MPV, or a hypervisor load or store).
    let use_background = !st.virt() && two_stage;
    if mode == PRV_M {
        return Ok(Translation { pa: addr, prot: rwx });
    }
    let (base_root, vm, widened) = if first_stage {
        let satp = if use_background { st.vsatp } else { st.satp };
        (get_field(satp, SATP64_PPN) << PGSHIFT, get_field(satp, SATP64_MODE), 0)
    } else {
        (get_field(st.hgatp, HGATP64_PPN) << PGSHIFT, get_field(st.hgatp, HGATP64_MODE), 2)
    };
    let levels: u32 = match vm {
        VM_SV39 => 3,
        VM_SV48 => 4,
        VM_SV57 => 5,
        VM_MBARE => return Ok(Translation { pa: addr, prot: rwx }),
        _ => unreachable!("satp or hgatp holds a bad mode {vm}"),
    };
    let ptidxbits = 9;
    let va_bits = PGSHIFT + levels * ptidxbits + widened;
    if first_stage {
        // The upper bits of the address must be a sign extension of bit va_bits - 1.
        let mask = (1u64 << (64 - (va_bits - 1))) - 1;
        let masked_msbs = (addr >> (va_bits - 1)) & mask;
        if masked_msbs != 0 && masked_msbs != mask {
            return Err(Fail::Page);
        }
    } else if addr >> va_bits != 0 {
        // A guest physical address has no bits above the widened size.
        return Err(Fail::Page);
    }
    let mut adue = st.menvcfg & MENVCFG_ADUE != 0;
    if first_stage && two_stage && st.virt() {
        adue = adue && st.henvcfg & MENVCFG_ADUE != 0;
    }
    let mut fault_pte_addr = fault_pte_addr;

    'restart: loop {
        let mut ptshift = (levels - 1) * ptidxbits;
        let mut base = base_root;
        let mut leaf = None;
        for i in 0..levels {
            let bits = if i == 0 { ptidxbits + widened } else { ptidxbits };
            let idx = (addr >> (PGSHIFT + ptshift)) & ((1 << bits) - 1);
            let pte_addr = if two_stage && first_stage {
                // Do the G stage translation of the page table address.
                let g = get_physical_address(
                    st,
                    as_,
                    base,
                    MmuAccessType::DataLoad,
                    MMU_IDX_U,
                    false,
                    true,
                    debug,
                    None,
                );
                match g {
                    Ok(t) => t.pa + idx * 8,
                    Err(_) => {
                        if let Some(f) = fault_pte_addr.as_deref_mut() {
                            *f = (base + idx * 8) >> 2;
                        }
                        return Err(Fail::GStage);
                    }
                }
            } else {
                base + idx * 8
            };
            pmp_check(st, pte_addr, 8, MmuAccessType::DataLoad, PRV_S)?;
            let (pte, res) = as_.load(pte_addr, 8, Endian::Little, MemTxAttrs::default());
            if !res.is_ok() {
                return Err(Fail::Access);
            }
            if pte & PTE_RESERVED != 0 {
                return Err(Fail::Page);
            }
            // Neither Svpbmt nor Svnapot is there.
            if pte & (PTE_PBMT | PTE_N) != 0 {
                return Err(Fail::Page);
            }
            let ppn = (pte & PTE_PPN_MASK) >> PTE_PPN_SHIFT;
            if pte & PTE_V == 0 {
                return Err(Fail::Page);
            }
            if pte & (PTE_R | PTE_W | PTE_X) != 0 {
                leaf = Some((pte, pte_addr, ppn, ptshift));
                break;
            }
            // D, A and U are reserved in non-leaf PTEs, and so are the attribute bits.
            if pte & (PTE_D | PTE_A | PTE_U | PTE_ATTR) != 0 {
                return Err(Fail::Page);
            }
            base = ppn << PGSHIFT;
            ptshift = ptshift.wrapping_sub(ptidxbits);
        }
        // Ran out of levels without a leaf.
        let Some((mut pte, pte_addr, ppn, ptshift)) = leaf else { return Err(Fail::Page) };

        // A misaligned superpage.
        if ppn & ((1u64 << ptshift) - 1) != 0 {
            return Err(Fail::Page);
        }
        let rwx_bits = pte & (PTE_R | PTE_W | PTE_X);
        // Write-only and write-execute pages are reserved (without shadow stacks).
        if rwx_bits == PTE_W || rwx_bits == (PTE_W | PTE_X) {
            return Err(Fail::Page);
        }
        let mut prot = 0;
        if rwx_bits & PTE_R != 0 {
            prot |= page::READ;
        }
        if rwx_bits & PTE_W != 0 {
            prot |= page::WRITE;
        }
        if rwx_bits & PTE_X != 0 {
            // mstatus serves the first stage, and the second stage without V (MPRV with
            // MPV); vsstatus adds to it in that case. The HS level MXR overrides both
            // stages.
            let mut mxr = (first_stage || !st.virt()) && st.mstatus & MSTATUS_MXR != 0;
            if first_stage && two_stage && !st.virt() {
                mxr |= st.vsstatus & MSTATUS_MXR != 0;
            }
            if st.virt() {
                mxr |= st.mstatus_hs & MSTATUS_MXR != 0;
            }
            if mxr {
                prot |= page::READ;
            }
            prot |= page::EXEC;
        }
        if pte & PTE_U != 0 {
            if mode != PRV_U {
                if !mmuidx_sum(mmu_idx) {
                    return Err(Fail::Page);
                }
                // SUM allows only data access to user pages.
                prot &= page::READ | page::WRITE;
            }
        } else if mode != PRV_S {
            return Err(Fail::Page);
        }
        if (prot >> access as u32) & 1 == 0 {
            return Err(Fail::Page);
        }

        let mut updated_pte = pte;
        if adue {
            updated_pte |= PTE_A | if access == MmuAccessType::DataStore { PTE_D } else { 0 };
        } else if pte & PTE_A == 0 || (access == MmuAccessType::DataStore && pte & PTE_D == 0) {
            return Err(Fail::Page);
        }
        if updated_pte != pte && !debug {
            pmp_check(st, pte_addr, 8, MmuAccessType::DataStore, PRV_S)?;
            // QEMU swaps the PTE in RAM with a compare and swap and restarts the walk if it
            // changed; see the module doc of `tcg`. A PTE outside RAM fails the walk.
            let (now, res) = as_.load(pte_addr, 8, Endian::Little, MemTxAttrs::default());
            if !res.is_ok() || !is_ram(as_, pte_addr) {
                return Err(Fail::Page);
            }
            if now != pte {
                continue 'restart;
            }
            let res = as_.store(pte_addr, 8, updated_pte, Endian::Little, MemTxAttrs::default());
            if !res.is_ok() {
                return Err(Fail::Page);
            }
            pte = updated_pte;
        }

        // For superpages, the low bits of the PPN come from the virtual address.
        let vpn = addr >> PGSHIFT;
        let pa = ((ppn | (vpn & ((1u64 << ptshift) - 1))) << PGSHIFT) | (addr & 0xfff);
        // Mark the page writable only after a store, so that the first store sets D.
        if access != MmuAccessType::DataStore && pte & PTE_D == 0 {
            prot &= !page::WRITE;
        }
        return Ok(Translation { pa, prot });
    }
}

/// `riscv_cpu_translate_for_debug()`: the physical address of `addr` for the debugger
/// and semihosting, through both stages under V=1, without PTE updates.
pub(crate) fn translate_debug(
    st: &CpuRiscvState,
    as_: &AddressSpace,
    addr: u64,
    mmu_idx: usize,
) -> Option<u64> {
    let load = MmuAccessType::DataLoad;
    let t = get_physical_address(st, as_, addr, load, mmu_idx, true, st.virt(), true, None);
    let mut pa = t.ok()?.pa;
    if st.virt() {
        let g = get_physical_address(st, as_, pa, load, MMU_IDX_U, false, true, true, None);
        pa = g.ok()?.pa;
    }
    Some(pa)
}

/// `riscv_cpu_tlb_fill()`.
pub(crate) fn tlb_fill(
    cpu: &mut Cpu<'_>,
    address: u64,
    size: usize,
    access: MmuAccessType,
    mmu_idx: usize,
    probe: bool,
    ra: Ra,
) -> Result<bool, CpuLoopExit> {
    let st = CpuRiscvState::load(cpu.env);
    let as_ = cpu.core.address_space().clone();
    let mode = mmuidx_priv(mmu_idx);
    let two_stage = mmuidx_2stage(mmu_idx);
    let mut tlb_size = 4096;
    let mut first_stage_error = true;
    let mut indirect = false;
    let mut gpfa = 0;
    let ret = if two_stage {
        // The VS stage.
        let r = get_physical_address(
            &st,
            &as_,
            address,
            access,
            mmu_idx,
            true,
            true,
            false,
            Some(&mut gpfa),
        );
        match r {
            Err(Fail::GStage) => {
                // A G stage fault during the VS stage walk; gpfa is already set.
                first_stage_error = false;
                indirect = true;
                r
            }
            Err(_) => r,
            Ok(t) => {
                // The G stage.
                let im = t.pa;
                match get_physical_address(
                    &st, &as_, im, access, MMU_IDX_U, false, true, false, None,
                ) {
                    Ok(t2) => pmp_check(&st, t2.pa, size as u64, access, mode).map(|prot_pmp| {
                        tlb_size = pmp::get_tlb_size(&st, t2.pa);
                        Translation { pa: t2.pa, prot: t.prot & t2.prot & prot_pmp }
                    }),
                    Err(fail) => {
                        // A guest physical address translation fault, an HS level
                        // exception.
                        first_stage_error = false;
                        if fail != Fail::Access {
                            gpfa = (im | (address & 0xfff)) >> 2;
                        }
                        Err(fail)
                    }
                }
            }
        }
    } else {
        get_physical_address(&st, &as_, address, access, mmu_idx, true, false, false, None)
            .and_then(|t| {
                let prot_pmp = pmp_check(&st, t.pa, size as u64, access, mode)?;
                tlb_size = pmp::get_tlb_size(&st, t.pa);
                Ok(Translation { pa: t.pa, prot: t.prot & prot_pmp })
            })
    };
    st64(cpu.env, GUEST_PHYS_FAULT_ADDR, gpfa);
    match ret {
        Ok(t) => {
            cpu.tlb_set_page(
                address & !(tlb_size - 1),
                t.pa & !(tlb_size - 1),
                t.prot,
                mmu_idx,
                tlb_size,
            );
            Ok(true)
        }
        Err(_) if probe => Ok(false),
        Err(fail) => {
            // raise_mmu_exception().
            let access_fault = fail == Fail::Access;
            let excp = match access {
                MmuAccessType::InstFetch if access_fault => EXCP_INST_ACCESS_FAULT,
                MmuAccessType::InstFetch if st.virt() && !first_stage_error => {
                    EXCP_INST_GUEST_PAGE_FAULT
                }
                MmuAccessType::InstFetch => EXCP_INST_PAGE_FAULT,
                MmuAccessType::DataLoad if access_fault => EXCP_LOAD_ACCESS_FAULT,
                MmuAccessType::DataLoad if two_stage && !first_stage_error => {
                    EXCP_LOAD_GUEST_ACCESS_FAULT
                }
                MmuAccessType::DataLoad => EXCP_LOAD_PAGE_FAULT,
                MmuAccessType::DataStore if access_fault => EXCP_STORE_AMO_ACCESS_FAULT,
                MmuAccessType::DataStore if two_stage && !first_stage_error => {
                    EXCP_STORE_GUEST_AMO_ACCESS_FAULT
                }
                MmuAccessType::DataStore => EXCP_STORE_PAGE_FAULT,
            };
            st64(cpu.env, BADADDR, address);
            st64(cpu.env, TWO_STAGE_LOOKUP, u64::from(two_stage));
            st64(cpu.env, TWO_STAGE_INDIRECT_LOOKUP, u64::from(indirect));
            Err(cpu.raise_exception(excp, ra))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ruvm_mem::MemorySystem;

    use super::*;
    use crate::cpu::MMU_IDX_S;

    const RAM: u64 = 0x8000_0000;
    const RWX: u32 = page::READ | page::WRITE | page::EXEC;
    const LEAF: u64 = PTE_V | PTE_R | PTE_W | PTE_X | PTE_A | PTE_D;
    /// The host address of the 16 KiB G stage root table.
    const G_ROOT: u64 = RAM + 0x10_0000;
    /// The guest physical address of the VS stage root table.
    const VS_ROOT: u64 = 0x20_0000;

    /// 16 MiB of RAM at 0x8000_0000, the guest physical address space of the tests.
    fn space() -> Arc<AddressSpace> {
        let mem = MemorySystem::new();
        let root = mem.new_container("system", 1 << 64).unwrap();
        let ram = mem.new_ram("ram", 16 << 20).unwrap();
        mem.add_subregion(root, RAM, ram).unwrap();
        mem.address_space_init(root, "memory").unwrap()
    }

    fn wr(as_: &AddressSpace, addr: u64, val: u64) {
        assert!(as_.store(addr, 8, val, Endian::Little, MemTxAttrs::default()).is_ok());
    }

    fn pte(pa: u64, flags: u64) -> u64 {
        ((pa >> 12) << PTE_PPN_SHIFT) | flags
    }

    /// A VS mode hart: Sv39x4 maps guest physical gigabyte 0 to the RAM base and guest
    /// physical 1 TiB (index 1024 of the widened root) to the RAM base too; Sv39 maps the
    /// guest virtual page 0x4000_0000 to guest physical 0x30_0000, 0x4000_1000 to guest
    /// physical 0x4000_0000 (outside the G stage map) and the gigabyte at 0x8000_0000 to
    /// guest physical 0; a PMP entry opens all of memory.
    fn setup() -> (CpuRiscvState, Arc<AddressSpace>) {
        let as_ = space();
        wr(&as_, G_ROOT, pte(RAM, LEAF | PTE_U));
        wr(&as_, G_ROOT + 1024 * 8, pte(RAM, LEAF | PTE_U));
        let host = |gpa: u64| RAM + gpa;
        wr(&as_, host(VS_ROOT) + 8, pte(0x20_1000, PTE_V));
        wr(&as_, host(VS_ROOT) + 2 * 8, pte(0, LEAF));
        wr(&as_, host(0x20_1000), pte(0x20_2000, PTE_V));
        wr(&as_, host(0x20_2000), pte(0x30_0000, LEAF));
        wr(&as_, host(0x20_2000) + 8, pte(0x4000_0000, LEAF));
        let mut st = CpuRiscvState {
            priv_lvl: PRV_S,
            virt_enabled: 1,
            hgatp: (VM_SV39 << 60) | (G_ROOT >> 12),
            vsatp: (VM_SV39 << 60) | (VS_ROOT >> 12),
            ..CpuRiscvState::default()
        };
        st.pmpcfg[0] = (3 << 3) | 7;
        st.pmpaddr[0] = u64::MAX >> 10;
        pmp::update_rules(&mut st);
        st.satp = st.vsatp;
        (st, as_)
    }

    fn walk(
        st: &CpuRiscvState,
        as_: &AddressSpace,
        addr: u64,
        first_stage: bool,
        fault: Option<&mut u64>,
    ) -> Result<Translation, Fail> {
        let idx = if first_stage { MMU_IDX_S | MMU_2STAGE_BIT } else { MMU_IDX_U };
        let load = MmuAccessType::DataLoad;
        get_physical_address(st, as_, addr, load, idx, first_stage, true, true, fault)
    }

    #[test]
    fn mmu_index_fields() {
        assert_eq!(mmuidx_priv(MMU_IDX_S_SUM | MMU_2STAGE_BIT), PRV_S);
        assert_eq!(mmuidx_priv(MMU_IDX_U | MMU_2STAGE_BIT), PRV_U);
        assert_eq!(mmuidx_priv(3), PRV_M);
        assert!(mmuidx_sum(MMU_IDX_S_SUM | MMU_2STAGE_BIT));
        assert!(!mmuidx_sum(MMU_IDX_S));
        assert!(mmuidx_2stage(MMU_IDX_U | MMU_2STAGE_BIT));
        assert!(!mmuidx_2stage(MMU_IDX_S_SUM));
    }

    #[test]
    fn g_stage_widened_root() {
        let (st, as_) = setup();
        // Sv39x4 takes 41 bit guest physical addresses with an 11 bit root index.
        let t = walk(&st, &as_, 0x1234, false, None).unwrap();
        assert_eq!(t.pa, RAM + 0x1234);
        assert_eq!(t.prot, RWX);
        let t = walk(&st, &as_, (1 << 40) + 0x5678, false, None).unwrap();
        assert_eq!(t.pa, RAM + 0x5678);
        // Bits above 41 fault; no sign extension applies to guest physical addresses.
        assert_eq!(walk(&st, &as_, 1 << 41, false, None).unwrap_err(), Fail::Page);
        assert_eq!(walk(&st, &as_, u64::MAX, false, None).unwrap_err(), Fail::Page);
        // An unmapped guest physical gigabyte.
        assert_eq!(walk(&st, &as_, 1 << 30, false, None).unwrap_err(), Fail::Page);
    }

    #[test]
    fn g_stage_needs_user_pages() {
        let (st, as_) = setup();
        wr(&as_, G_ROOT, pte(RAM, LEAF));
        assert_eq!(walk(&st, &as_, 0x1000, false, None).unwrap_err(), Fail::Page);
    }

    #[test]
    fn g_stage_bare_is_identity() {
        let (mut st, as_) = setup();
        st.hgatp = 0;
        let t = walk(&st, &as_, 0x1_2345_6789, false, None).unwrap();
        assert_eq!(t.pa, 0x1_2345_6789);
        assert_eq!(t.prot, RWX);
    }

    #[test]
    fn two_stage_translation() {
        let (st, as_) = setup();
        // The VS stage gives the guest physical address; its page table reads go
        // through the G stage.
        let t = walk(&st, &as_, 0x4000_0abc, true, None).unwrap();
        assert_eq!(t.pa, 0x30_0abc);
        let pa = translate_debug(&st, &as_, 0x4000_0abc, MMU_IDX_S | MMU_2STAGE_BIT);
        assert_eq!(pa, Some(RAM + 0x30_0abc));
        // A VS stage gigapage.
        let pa = translate_debug(&st, &as_, 0x8000_1234, MMU_IDX_S | MMU_2STAGE_BIT);
        assert_eq!(pa, Some(RAM + 0x1234));
        // The guest physical page 0x4000_0000 has no G stage mapping.
        let t = walk(&st, &as_, 0x4000_1000, true, None).unwrap();
        assert_eq!(t.pa, 0x4000_0000);
        assert_eq!(translate_debug(&st, &as_, 0x4000_1000, MMU_IDX_S | MMU_2STAGE_BIT), None);
    }

    #[test]
    fn g_stage_fault_on_page_table_address() {
        let (mut st, as_) = setup();
        // A VS root table at a guest physical address the G stage does not map.
        // With V=1 the guest's satp is the live one.
        st.vsatp = (VM_SV39 << 60) | (0x5000_0000 >> 12);
        st.satp = st.vsatp;
        let mut gpfa = 0;
        let r = walk(&st, &as_, 0x4000_0000, true, Some(&mut gpfa));
        assert_eq!(r.unwrap_err(), Fail::GStage);
        assert_eq!(gpfa, (0x5000_0000 + 8) >> 2);
    }

    #[test]
    fn vs_stage_uses_vsatp_from_hs_mode() {
        let (mut st, as_) = setup();
        // HLV from HS mode: V=0, two stage, the first stage through vsatp and not satp.
        st.virt_enabled = 0;
        st.satp = 0;
        let idx = MMU_IDX_S | MMU_2STAGE_BIT;
        let load = MmuAccessType::DataLoad;
        let t = get_physical_address(&st, &as_, 0x4000_0abc, load, idx, true, true, true, None);
        assert_eq!(t.unwrap().pa, 0x30_0abc);
    }

    #[test]
    fn vs_stage_mxr_from_hs_status() {
        let (st, as_) = setup();
        // An execute only VS stage page is readable with the HS level mstatus.MXR.
        wr(&as_, RAM + 0x20_2000, pte(0x30_0000, PTE_V | PTE_X | PTE_A | PTE_D));
        assert_eq!(walk(&st, &as_, 0x4000_0000, true, None).unwrap_err(), Fail::Page);
        let mut st2 = st.clone();
        st2.mstatus_hs = MSTATUS_MXR;
        assert_eq!(walk(&st2, &as_, 0x4000_0000, true, None).unwrap().pa, 0x30_0000);
    }
}
