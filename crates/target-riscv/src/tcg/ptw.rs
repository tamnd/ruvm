// SPDX-License-Identifier: GPL-2.0-or-later

//! The RISC-V page walk and TLB fill: `get_physical_address()`,
//! `get_physical_address_pmp()`, `riscv_cpu_tlb_fill()` and `raise_mmu_exception()` of
//! QEMU's `target/riscv/tcg/cpu_helper.c`, for one stage of Sv39, Sv48 or Sv57 translation
//! with Svadu. The model has no Svpbmt, Svnapot or shadow stacks, so PTEs using them are
//! invalid.

use ruvm_jit::{Cpu, CpuLoopExit, MmuAccessType, Ra, page};
use ruvm_mem::{AddressSpace, Endian, MemTxAttrs, RegionType};

use super::pmp;
use crate::cpu::{
    BADADDR, CpuRiscvState, EXCP_INST_ACCESS_FAULT, EXCP_INST_PAGE_FAULT, EXCP_LOAD_ACCESS_FAULT,
    EXCP_LOAD_PAGE_FAULT, EXCP_STORE_AMO_ACCESS_FAULT, EXCP_STORE_PAGE_FAULT, MENVCFG_ADUE,
    MMU_IDX_S_SUM, MSTATUS_MXR, PRV_M, PRV_S, PRV_U, PTE_A, PTE_ATTR, PTE_D, PTE_N, PTE_PBMT,
    PTE_PPN_MASK, PTE_PPN_SHIFT, PTE_R, PTE_RESERVED, PTE_U, PTE_V, PTE_W, PTE_X, SATP64_MODE,
    SATP64_PPN, VM_MBARE, VM_SV39, VM_SV48, VM_SV57, get_field,
};

const PGSHIFT: u32 = 12;

/// Why a translation failed, `TRANSLATE_*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fail {
    /// `TRANSLATE_FAIL`: a page fault.
    Page,
    /// `TRANSLATE_PMP_FAIL` or `TRANSLATE_PMA_FAIL`: an access fault.
    Access,
}

/// A successful translation: the physical address and the page protection.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Translation {
    pub(crate) pa: u64,
    pub(crate) prot: u32,
}

/// `mmuidx_priv()`: the privilege level an MMU index translates for.
fn mmuidx_priv(mmu_idx: usize) -> u64 {
    if mmu_idx == MMU_IDX_S_SUM { PRV_S } else { mmu_idx as u64 }
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

/// `get_physical_address()` for one stage of translation, with `is_debug` (no PTE
/// updates) when `debug` is set.
pub(crate) fn get_physical_address(
    st: &CpuRiscvState,
    as_: &AddressSpace,
    addr: u64,
    access: MmuAccessType,
    mmu_idx: usize,
    debug: bool,
) -> Result<Translation, Fail> {
    let rwx = page::READ | page::WRITE | page::EXEC;
    let mode = mmuidx_priv(mmu_idx);
    if mode == PRV_M {
        return Ok(Translation { pa: addr, prot: rwx });
    }
    let base_root = get_field(st.satp, SATP64_PPN) << PGSHIFT;
    let vm = get_field(st.satp, SATP64_MODE);
    let levels: u32 = match vm {
        VM_SV39 => 3,
        VM_SV48 => 4,
        VM_SV57 => 5,
        VM_MBARE => return Ok(Translation { pa: addr, prot: rwx }),
        _ => unreachable!("satp holds a bad mode {vm}"),
    };
    let ptidxbits = 9;
    let va_bits = PGSHIFT + levels * ptidxbits;
    // The upper bits of the address must be a sign extension of bit va_bits - 1.
    let mask = (1u64 << (64 - (va_bits - 1))) - 1;
    let masked_msbs = (addr >> (va_bits - 1)) & mask;
    if masked_msbs != 0 && masked_msbs != mask {
        return Err(Fail::Page);
    }
    let adue = st.menvcfg & MENVCFG_ADUE != 0;

    'restart: loop {
        let mut ptshift = (levels - 1) * ptidxbits;
        let mut base = base_root;
        let mut leaf = None;
        for _ in 0..levels {
            let idx = (addr >> (PGSHIFT + ptshift)) & ((1 << ptidxbits) - 1);
            let pte_addr = base + idx * 8;
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
            if get_field(st.mstatus, MSTATUS_MXR) != 0 {
                prot |= page::READ;
            }
            prot |= page::EXEC;
        }
        if pte & PTE_U != 0 {
            if mode != PRV_U {
                if mmu_idx != MMU_IDX_S_SUM {
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
    let mut tlb_size = 4096;
    let ret = get_physical_address(&st, &as_, address, access, mmu_idx, false).and_then(|t| {
        let prot_pmp = pmp_check(&st, t.pa, size as u64, access, mode)?;
        tlb_size = pmp::get_tlb_size(&st, t.pa);
        Ok(Translation { pa: t.pa, prot: t.prot & prot_pmp })
    });
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
            let excp = match (access, access_fault) {
                (MmuAccessType::InstFetch, true) => EXCP_INST_ACCESS_FAULT,
                (MmuAccessType::InstFetch, false) => EXCP_INST_PAGE_FAULT,
                (MmuAccessType::DataLoad, true) => EXCP_LOAD_ACCESS_FAULT,
                (MmuAccessType::DataLoad, false) => EXCP_LOAD_PAGE_FAULT,
                (MmuAccessType::DataStore, true) => EXCP_STORE_AMO_ACCESS_FAULT,
                (MmuAccessType::DataStore, false) => EXCP_STORE_PAGE_FAULT,
            };
            super::st64(cpu.env, BADADDR, address);
            Err(cpu.raise_exception(excp, ra))
        }
    }
}
