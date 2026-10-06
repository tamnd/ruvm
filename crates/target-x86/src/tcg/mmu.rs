// SPDX-License-Identifier: GPL-2.0-or-later

//! The page walk, the port of `target/i386/tcg/system/excp_helper.c`: 32-bit paging with
//! PSE, PAE paging, and 4 and 5 level long mode paging.
//!
//! Page table entries are read and written in physical memory through the vCPU's address
//! space. The accessed and dirty bits are set with a plain store rather than a compare and
//! swap (see the module doc of `tcg`).

use ruvm_jit::cputlb::tlb_set_page;
use ruvm_jit::{Cpu, CpuLoopExit, MmuAccessType, Ra, page};
use ruvm_mem::{Endian, MemTxAttrs};

use super::env::{A20_MASK, EFER, HFLAGS, PKRS, PKRU, cr, ld32, ld64, st64};
use super::seg::raise_exception_err_ra;
use super::{
    EXCP0D_GPF, EXCP0E_PAGE, MMU_KSMAP32_IDX, MMU_KSMAP64_IDX, MMU_NESTED_IDX, MMU_PHYS_IDX,
    MMU_USER32_IDX, MMU_USER64_IDX, X86,
};
use crate::state::{
    CR0_PG_MASK, CR0_WP_MASK, CR4_LA57_MASK, CR4_PAE_MASK, CR4_PKE_MASK, CR4_PKS_MASK,
    CR4_PSE_MASK, CR4_SMEP_MASK, HF_LMA_MASK, MSR_EFER_NXE,
};

const PG_PRESENT_MASK: u64 = 1 << 0;
const PG_RW_MASK: u64 = 1 << 1;
const PG_USER_MASK: u64 = 1 << 2;
const PG_ACCESSED_MASK: u64 = 1 << 5;
const PG_DIRTY_MASK: u64 = 1 << 6;
const PG_PSE_MASK: u64 = 1 << 7;
const PG_PSE_PAT_MASK: u64 = 1 << 12;
const PG_ADDRESS_MASK: u64 = 0x000f_ffff_ffff_f000;
const PG_HI_USER_MASK: u64 = 0x7ff0_0000_0000_0000;
const PG_NX_MASK: u64 = 1 << 63;
const PG_PKRU_BIT: u32 = 59;
const PG_PKRU_MASK: u64 = 15 << PG_PKRU_BIT;

/// Page fault error code bits.
pub(crate) const PG_ERROR_P_MASK: u32 = 0x01;
pub(crate) const PG_ERROR_W_MASK: u32 = 0x02;
pub(crate) const PG_ERROR_U_MASK: u32 = 0x04;
pub(crate) const PG_ERROR_RSVD_MASK: u32 = 0x08;
pub(crate) const PG_ERROR_I_D_MASK: u32 = 0x10;
pub(crate) const PG_ERROR_PK_MASK: u32 = 0x20;

const PG_MODE_PAE: u32 = 1 << 0;
const PG_MODE_LMA: u32 = 1 << 1;
const PG_MODE_NXE: u32 = 1 << 2;
const PG_MODE_PSE: u32 = 1 << 3;
const PG_MODE_LA57: u32 = 1 << 4;
const PG_MODE_WP: u32 = 1 << 16;
const PG_MODE_PKE: u32 = 1 << 17;
const PG_MODE_PKS: u32 = 1 << 18;
const PG_MODE_SMEP: u32 = 1 << 19;
const PG_MODE_PG: u32 = 1 << 20;

/// Whether 5 level paging is on.
pub(crate) fn la57(env: &[u8]) -> bool {
    ld32(env, HFLAGS) & HF_LMA_MASK != 0 && ld64(env, cr(4)) & CR4_LA57_MASK != 0
}

/// `get_pg_mode()`.
fn get_pg_mode(env: &[u8]) -> u32 {
    let cr0 = ld64(env, cr(0));
    let cr4 = ld64(env, cr(4));
    if cr0 & CR0_PG_MASK == 0 {
        return 0;
    }
    let mut m = PG_MODE_PG;
    if cr0 & CR0_WP_MASK != 0 {
        m |= PG_MODE_WP;
    }
    if cr4 & CR4_PAE_MASK != 0 {
        m |= PG_MODE_PAE;
        if ld64(env, EFER) & MSR_EFER_NXE != 0 {
            m |= PG_MODE_NXE;
        }
    }
    if cr4 & CR4_PSE_MASK != 0 {
        m |= PG_MODE_PSE;
    }
    if cr4 & CR4_SMEP_MASK != 0 {
        m |= PG_MODE_SMEP;
    }
    if ld32(env, HFLAGS) & HF_LMA_MASK != 0 {
        m |= PG_MODE_LMA;
        if cr4 & CR4_PKE_MASK != 0 {
            m |= PG_MODE_PKE;
        }
        if cr4 & CR4_PKS_MASK != 0 {
            m |= PG_MODE_PKS;
        }
        if cr4 & CR4_LA57_MASK != 0 {
            m |= PG_MODE_LA57;
        }
    }
    m
}

/// A successful translation.
struct Translation {
    paddr: u64,
    prot: u32,
    page_size: u64,
}

/// A failed one.
struct Fault {
    exception_index: i32,
    error_code: u32,
    cr2: u64,
}

fn ptw_ld(cpu: &Cpu<'_>, addr: u64, size: u32) -> u64 {
    // A read of unassigned memory returns all ones, as in QEMU.
    let (v, _) = cpu.core.address_space().load(addr, size, Endian::Little, MemTxAttrs::UNSPECIFIED);
    v
}

/// `ptw_setl()`: set bits in the low half of an entry.
fn ptw_setl(cpu: &Cpu<'_>, addr: u64, old: u64, set: u64) {
    if set & !old != 0 {
        let new = (old as u32) | (set as u32);
        let _ = cpu.core.address_space().store(
            addr,
            4,
            u64::from(new),
            Endian::Little,
            MemTxAttrs::UNSPECIFIED,
        );
    }
}

fn is_user(mmu_idx: usize) -> bool {
    mmu_idx == MMU_USER64_IDX || mmu_idx == MMU_USER32_IDX
}

fn is_smap(mmu_idx: usize) -> bool {
    mmu_idx == MMU_KSMAP64_IDX || mmu_idx == MMU_KSMAP32_IDX
}

enum Walk {
    Fault(u32),
    Leaf { pte: u64, pte_addr: u64, ptep: u64, page_size: u64, rsvd_mask: u64 },
}

/// A leaf at `do_check_protect`, which adds the page offset bits to the reserved mask.
fn leaf(pte: u64, pte_addr: u64, ptep: u64, page_size: u64, rsvd_mask: u64) -> Walk {
    let rsvd_mask = rsvd_mask | ((page_size - 1) & PG_ADDRESS_MASK & !PG_PSE_PAT_MASK);
    Walk::Leaf { pte, pte_addr, ptep, page_size, rsvd_mask }
}

/// The table walk of `mmu_translate()`, up to the leaf entry.
fn walk(cpu: &Cpu<'_>, addr: u64, cr3: u64, pg_mode: u32, phys_bits: u32) -> Walk {
    let mut rsvd_mask = !((1u64 << phys_bits) - 1) & PG_ADDRESS_MASK;
    if pg_mode & PG_MODE_NXE == 0 {
        rsvd_mask |= PG_NX_MASK;
    }
    let fault = Walk::Fault(0);
    let fault_rsvd = Walk::Fault(PG_ERROR_RSVD_MASK);

    if pg_mode & PG_MODE_PAE != 0 {
        let mut pte;
        let mut ptep;
        if pg_mode & PG_MODE_LMA != 0 {
            if pg_mode & PG_MODE_LA57 != 0 {
                // Page table level 5.
                let pte_addr = (cr3 & !0xfff) + (((addr >> 48) & 0x1ff) << 3);
                pte = ptw_ld(cpu, pte_addr, 8);
                if pte & PG_PRESENT_MASK == 0 {
                    return fault;
                }
                if pte & (rsvd_mask | PG_PSE_MASK) != 0 {
                    return fault_rsvd;
                }
                ptw_setl(cpu, pte_addr, pte, PG_ACCESSED_MASK);
                ptep = pte ^ PG_NX_MASK;
            } else {
                pte = cr3;
                ptep = PG_NX_MASK | PG_USER_MASK | PG_RW_MASK;
            }
            // Page table level 4.
            let pte_addr = (pte & PG_ADDRESS_MASK) + (((addr >> 39) & 0x1ff) << 3);
            pte = ptw_ld(cpu, pte_addr, 8);
            if pte & PG_PRESENT_MASK == 0 {
                return fault;
            }
            if pte & (rsvd_mask | PG_PSE_MASK) != 0 {
                return fault_rsvd;
            }
            ptw_setl(cpu, pte_addr, pte, PG_ACCESSED_MASK);
            ptep &= pte ^ PG_NX_MASK;
            // Page table level 3.
            let pte_addr = (pte & PG_ADDRESS_MASK) + (((addr >> 30) & 0x1ff) << 3);
            pte = ptw_ld(cpu, pte_addr, 8);
            if pte & PG_PRESENT_MASK == 0 {
                return fault;
            }
            if pte & rsvd_mask != 0 {
                return fault_rsvd;
            }
            ptw_setl(cpu, pte_addr, pte, PG_ACCESSED_MASK);
            ptep &= pte ^ PG_NX_MASK;
            if pte & PG_PSE_MASK != 0 {
                // 1 GB page.
                return leaf(pte, pte_addr, ptep, 1 << 30, rsvd_mask);
            }
        } else {
            // Page table level 3.
            let pte_addr = (cr3 & 0xffff_ffe0) + ((addr >> 27) & 0x18);
            rsvd_mask |= PG_HI_USER_MASK;
            pte = ptw_ld(cpu, pte_addr, 8);
            if pte & PG_PRESENT_MASK == 0 {
                return fault;
            }
            if pte & (rsvd_mask | PG_NX_MASK) != 0 {
                return fault_rsvd;
            }
            ptw_setl(cpu, pte_addr, pte, PG_ACCESSED_MASK);
            ptep = PG_NX_MASK | PG_USER_MASK | PG_RW_MASK;
        }
        // Page table level 2.
        let pte_addr = (pte & PG_ADDRESS_MASK) + (((addr >> 21) & 0x1ff) << 3);
        pte = ptw_ld(cpu, pte_addr, 8);
        if pte & PG_PRESENT_MASK == 0 {
            return fault;
        }
        if pte & rsvd_mask != 0 {
            return fault_rsvd;
        }
        if pte & PG_PSE_MASK != 0 {
            // 2 MB page.
            ptep &= pte ^ PG_NX_MASK;
            return leaf(pte, pte_addr, ptep, 2 << 20, rsvd_mask);
        }
        ptw_setl(cpu, pte_addr, pte, PG_ACCESSED_MASK);
        ptep &= pte ^ PG_NX_MASK;
        // Page table level 1.
        let pte_addr = (pte & PG_ADDRESS_MASK) + (((addr >> 12) & 0x1ff) << 3);
        pte = ptw_ld(cpu, pte_addr, 8);
        if pte & PG_PRESENT_MASK == 0 {
            return fault;
        }
        if pte & rsvd_mask != 0 {
            return fault_rsvd;
        }
        // Combine the PDE and PTE NX, user and RW protections.
        ptep &= pte ^ PG_NX_MASK;
        leaf(pte, pte_addr, ptep, 4096, rsvd_mask)
    } else {
        // Page table level 2.
        let pte_addr = (cr3 & 0xffff_f000) + ((addr >> 20) & 0xffc);
        let mut pte = ptw_ld(cpu, pte_addr, 4);
        if pte & PG_PRESENT_MASK == 0 {
            return fault;
        }
        let mut ptep = pte | PG_NX_MASK;
        // If the PSE bit is set, this is a 4 MB page.
        if pte & PG_PSE_MASK != 0 && pg_mode & PG_MODE_PSE != 0 {
            // Bits 20-13 give bits 39-32 of the address, bit 21 is reserved. Bits 20-13 stay
            // in place for setting the accessed and dirty bits below.
            pte = (pte & 0xffff_ffff) | ((pte & 0x1f_e000) << (32 - 13));
            // `do_check_protect_pse36` does not add the page offset bits to the mask.
            return Walk::Leaf { pte, pte_addr, ptep, page_size: 4 << 20, rsvd_mask: 0x20_0000 };
        }
        ptw_setl(cpu, pte_addr, pte, PG_ACCESSED_MASK);
        // Page table level 1.
        let pte_addr = (pte & !0xfff & 0xffff_ffff) + ((addr >> 10) & 0xffc);
        pte = ptw_ld(cpu, pte_addr, 4);
        if pte & PG_PRESENT_MASK == 0 {
            return fault;
        }
        // Combine the PDE and PTE user and RW protections.
        ptep &= pte | PG_NX_MASK;
        Walk::Leaf { pte, pte_addr, ptep, page_size: 4096, rsvd_mask: 0 }
    }
}

/// `mmu_translate()` for the paging modes, without nested paging or protection keys.
fn mmu_translate(
    cpu: &Cpu<'_>,
    x: &X86,
    addr: u64,
    access_type: MmuAccessType,
    mmu_idx: usize,
    pg_mode: u32,
) -> Result<Translation, Fault> {
    let user = is_user(mmu_idx);
    let cr3 = ld64(cpu.env, cr(3));
    let walked = walk(cpu, addr, cr3, pg_mode, x.model().phys_bits());
    let fault_code = match walked {
        Walk::Fault(code) => code,
        Walk::Leaf { pte, pte_addr, mut ptep, page_size, rsvd_mask } => {
            if pte & rsvd_mask != 0 {
                PG_ERROR_RSVD_MASK
            } else {
                ptep ^= PG_NX_MASK;
                // Can the page be put in the TLB? `prot` tells.
                if user && ptep & PG_USER_MASK == 0 {
                    PG_ERROR_P_MASK
                } else {
                    let mut prot = 0;
                    if !is_smap(mmu_idx) || ptep & PG_USER_MASK == 0 {
                        prot |= page::READ;
                        if ptep & PG_RW_MASK != 0 || !(user || pg_mode & PG_MODE_WP != 0) {
                            prot |= page::WRITE;
                        }
                    }
                    if ptep & PG_NX_MASK == 0
                        && (user || !(pg_mode & PG_MODE_SMEP != 0 && ptep & PG_USER_MASK != 0))
                    {
                        prot |= page::EXEC;
                    }
                    // Protection keys: PKRU for user pages, PKRS for supervisor ones.
                    let pkr = match (ptep & PG_USER_MASK != 0, pg_mode) {
                        (true, m) if m & PG_MODE_PKE != 0 => ld32(cpu.env, PKRU),
                        (false, m) if m & PG_MODE_PKS != 0 => ld32(cpu.env, PKRS),
                        _ => 0,
                    };
                    let mut pk_fault = false;
                    if pkr != 0 {
                        let pk = ((pte & PG_PKRU_MASK) >> PG_PKRU_BIT) as u32;
                        let mut pkr_prot = page::READ | page::WRITE | page::EXEC;
                        if (pkr >> (pk * 2)) & 1 != 0 {
                            pkr_prot &= !(page::READ | page::WRITE);
                        } else if (pkr >> (pk * 2)) & 2 != 0 && (user || pg_mode & PG_MODE_WP != 0)
                        {
                            pkr_prot &= !page::WRITE;
                        }
                        pk_fault = pkr_prot & (1 << access_type as u32) == 0;
                        prot &= pkr_prot;
                    }
                    if pk_fault {
                        PG_ERROR_PK_MASK | PG_ERROR_P_MASK
                    } else if prot & (1 << access_type as u32) == 0 {
                        PG_ERROR_P_MASK
                    } else {
                        let mut set = PG_ACCESSED_MASK;
                        if access_type == MmuAccessType::DataStore {
                            set |= PG_DIRTY_MASK;
                        } else if pte & PG_DIRTY_MASK == 0 {
                            // Only allow writes if already dirty, otherwise wait for the
                            // dirtying access.
                            prot &= !page::WRITE;
                        }
                        ptw_setl(cpu, pte_addr, pte, set);
                        // Merge the offset within the page.
                        let paddr =
                            (pte & PG_ADDRESS_MASK & !(page_size - 1)) | (addr & (page_size - 1));
                        let a20 = ld64(cpu.env, A20_MASK);
                        return Ok(Translation { paddr: paddr & a20, prot, page_size });
                    }
                }
            }
        }
    };
    let mut error_code = fault_code;
    if user {
        error_code |= PG_ERROR_U_MASK;
    }
    match access_type {
        MmuAccessType::DataLoad => {}
        MmuAccessType::DataStore => error_code |= PG_ERROR_W_MASK,
        MmuAccessType::InstFetch => {
            if pg_mode & (PG_MODE_NXE | PG_MODE_SMEP) != 0 {
                error_code |= PG_ERROR_I_D_MASK;
            }
        }
    }
    Err(Fault { exception_index: EXCP0E_PAGE, error_code, cr2: addr })
}

/// `get_physical_address()`.
fn get_physical_address(
    cpu: &Cpu<'_>,
    x: &X86,
    addr: u64,
    access_type: MmuAccessType,
    mmu_idx: usize,
) -> Result<Translation, Fault> {
    let mut addr = addr;
    if mmu_idx != MMU_PHYS_IDX && mmu_idx != MMU_NESTED_IDX {
        if mmu_idx & 1 != 0 {
            addr &= 0xffff_ffff;
        }
        if ld64(cpu.env, cr(0)) & CR0_PG_MASK != 0 {
            let pg_mode = get_pg_mode(cpu.env);
            if pg_mode & PG_MODE_LMA != 0 {
                // Test the virtual address sign extension.
                let shift = if pg_mode & PG_MODE_LA57 != 0 { 56 } else { 47 };
                let sext = (addr as i64) >> shift;
                if sext != 0 && sext != -1 {
                    // A non-canonical #GP does not change CR2.
                    return Err(Fault {
                        exception_index: EXCP0D_GPF,
                        error_code: 0,
                        cr2: ld64(cpu.env, cr(2)),
                    });
                }
            }
            return mmu_translate(cpu, x, addr, access_type, mmu_idx, pg_mode);
        }
    }
    // No translation needed.
    Ok(Translation {
        paddr: addr & ld64(cpu.env, A20_MASK),
        prot: page::READ | page::WRITE | page::EXEC,
        page_size: 4096,
    })
}

/// `x86_cpu_tlb_fill()`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tlb_fill(
    cpu: &mut Cpu<'_>,
    x: &X86,
    addr: u64,
    _size: usize,
    access_type: MmuAccessType,
    mmu_idx: usize,
    probe: bool,
    ra: Ra,
) -> Result<bool, CpuLoopExit> {
    match get_physical_address(cpu, x, addr, access_type, mmu_idx) {
        Ok(t) => {
            // Even for large pages only one 4 KiB page goes in the TLB, so it does not fill
            // too fast.
            tlb_set_page(cpu, addr & !0xfff, t.paddr & !0xfff, t.prot, mmu_idx, t.page_size);
            Ok(true)
        }
        Err(f) => {
            if probe {
                return Ok(false);
            }
            st64(cpu.env, cr(2), f.cr2);
            Err(raise_exception_err_ra(cpu, f.exception_index, f.error_code, ra))
        }
    }
}
