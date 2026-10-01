// SPDX-License-Identifier: GPL-2.0-or-later

//! The stage 1 page table walk for the EL1&0 regime: the port of the AArch64, 4K granule
//! parts of `get_phys_addr_lpae()`, `get_phys_addr_disabled()`, `aa64_va_parameters()` and
//! `get_S1prot()` from QEMU's `target/arm/ptw.c`, plus `arm_cpu_tlb_fill()` and
//! `arm_deliver_fault()` from `tlb_helper.c`.

use ruvm_jit::{Cpu, CpuLoopExit, MmuAccessType, Ra, page};
use ruvm_mem::{Endian, MemTxAttrs, MemTxResult};

use super::Arm;
use crate::cpu::{
    CpuArmState, EXCP_DATA_ABORT, EXCP_PREFETCH_ABORT, MMU_IDX_E10_0, MMU_IDX_E10_1_PAN, SCTLR_M,
    SCTLR_WXN, pa_range_bits,
};
use crate::syndrome::{fsc, syn_data_abort_no_iss, syn_insn_abort};

/// A translation fault: the long descriptor fault status code (with the level folded in) and
/// the external abort type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Fault {
    /// The fault status code.
    pub(crate) fsc: u32,
    /// The EA bit.
    pub(crate) ea: bool,
}

impl Fault {
    fn new(fsc: u32) -> Fault {
        Fault { fsc, ea: false }
    }
}

/// A successful translation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Translation {
    /// The physical address.
    pub(crate) pa: u64,
    /// `PAGE_READ`, `PAGE_WRITE` and `PAGE_EXEC`.
    pub(crate) prot: u32,
    /// The MAIR attribute byte.
    pub(crate) attrs: u8,
    /// The shareability field.
    pub(crate) sh: u8,
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

/// `get_phys_addr()` for a stage 1 EL1&0 regime access. `is_at` is set for the AT
/// instructions, which do not update the access flag or the dirty state.
pub(crate) fn get_phys_addr(
    arm: &Arm,
    cpu: &mut Cpu<'_>,
    address: u64,
    access: MmuAccessType,
    mmu_idx: usize,
    is_at: bool,
) -> Result<Translation, Fault> {
    let st = CpuArmState::load(cpu.env);
    let model = arm.model();
    let feat = model.features;
    let tcr = st.tcr_el[1];
    let select = extract64(address, 55, 1);
    let data = access != MmuAccessType::InstFetch;

    let (tbi, tbid) = if select == 0 {
        (extract64(tcr, 37, 1), extract64(tcr, 51, 1))
    } else {
        (extract64(tcr, 38, 1), extract64(tcr, 52, 1))
    };
    let tbi = if data { tbi } else { tbi & !tbid };

    if st.sctlr_el[1] & SCTLR_M == 0 {
        // get_phys_addr_disabled().
        let pamax = model.pamax();
        let addrtop = if tbi != 0 { 55 } else { 63 };
        if extract64(address, pamax, addrtop - pamax + 1) != 0 {
            return Err(Fault::new(fsc::address_size(0)));
        }
        return Ok(Translation {
            pa: extract64(address, 0, 52),
            prot: page::READ | page::WRITE | page::EXEC,
            attrs: 0,
            sh: 2,
        });
    }

    // aa64_va_parameters().
    let (tsz, epd, hpd) = if select == 0 {
        (extract64(tcr, 0, 6), extract64(tcr, 7, 1), extract64(tcr, 41, 1))
    } else {
        (extract64(tcr, 16, 6), extract64(tcr, 23, 1), extract64(tcr, 42, 1))
    };
    let hpd = feat.hpds && hpd != 0;
    let ps = extract64(tcr, 32, 3) as u32;
    let ha = feat.hafdbs >= 1 && extract64(tcr, 39, 1) != 0;
    let hd = ha && feat.hafdbs >= 2 && extract64(tcr, 40, 1) != 0;

    let mut level: u32 = 0;
    // If TxSZ is out of range, it is IMPLEMENTATION DEFINED whether we behave as if it were
    // in range or raise a level 0 Translation fault; like QEMU we raise the fault.
    if !(16..=39).contains(&tsz) {
        return Err(Fault::new(fsc::translation(level)));
    }
    let tsz = tsz as u32;
    let addrsize = 64 - 8 * tbi as u32;
    let inputsize = 64 - tsz;

    let parange = (model.id_aa64mmfr0 & 0xf) as u32;
    let outputsize = pa_range_bits(parange.min(ps)).min(48);

    if inputsize < addrsize {
        let top_bits = sextract64(address, inputsize, addrsize - inputsize);
        if top_bits.wrapping_neg() as u64 != select {
            // The gap between the two regions is a Translation fault.
            return Err(Fault::new(fsc::translation(level)));
        }
    }

    let stride: u32 = 9;
    let ttbr = if select == 0 { st.ttbr0_el[1] } else { st.ttbr1_el[1] };
    if epd != 0 {
        // Translation table walk disabled: Translation fault on TLB miss.
        return Err(Fault::new(fsc::translation(level)));
    }

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
    let as_ = cpu.core.address_space().clone();
    let mut desc_pa;

    let descriptor = loop {
        descaddr |= (address >> (stride * (4 - level))) & indexmask;
        descaddr &= !7;
        desc_pa = descaddr;

        let (descriptor, res) = as_.load(descaddr, 8, Endian::Little, MemTxAttrs::default());
        if !res.is_ok() {
            return Err(Fault {
                fsc: fsc::sync_external_on_walk(level),
                ea: res != MemTxResult::DECODE_ERROR,
            });
        }

        // Invalid, or a block descriptor at an invalid level (level 0 for 4K pages).
        if descriptor & 1 == 0 || (descriptor & 2 == 0 && (level == 0 || level == 3)) {
            return Err(Fault::new(fsc::translation(level)));
        }

        descaddr = descriptor & descaddrmask;
        if descaddr >> outputsize != 0 {
            return Err(Fault::new(fsc::address_size(level)));
        }

        if descriptor & 2 != 0 && level < 3 {
            // Table entry. The top five bits are attributes which propagate down.
            tableattrs |= extract64(descriptor, 59, 5);
            level += 1;
            indexmask = indexmask_grainsize;
            continue;
        }
        break descriptor;
    };

    // Block entry at level 1 or 2, or page entry at level 3.
    let page_size = 1u64 << (stride * (4 - level) + 3);
    descaddr &= !(page_size - 1);
    descaddr |= address & (page_size - 1);

    let mut new_descriptor = descriptor;
    // Check the descriptor AF bit.
    if descriptor & (1 << 10) == 0 && !ha {
        return Err(Fault::new(fsc::access_flag(level)));
    }
    // For AccessType_AT, DB is not updated, and it is IMPLEMENTATION DEFINED whether AF is
    // updated; like QEMU we choose not to.
    if !is_at {
        if descriptor & (1 << 10) == 0 && ha {
            new_descriptor |= 1 << 10;
        }
        if hd && extract64(descriptor, 51, 1) != 0 && access == MmuAccessType::DataStore {
            // Clear AP[2].
            new_descriptor &= !(1u64 << 7);
        }
    }

    let mut attrs = new_descriptor & ((((1u64 << 10) - 1) << 2) | (((1u64 << 14) - 1) << 50));
    if !hpd {
        // XN, PXN.
        attrs |= extract64(tableattrs, 0, 2) << 53;
        // The sense of AP[1] vs APTable[0] is reversed: APTable[0] == 1 forces AP[1] to 0.
        attrs &= !(extract64(tableattrs, 2, 1) << 6);
        attrs |= extract64(tableattrs, 3, 1) << 7;
    }
    let ap = extract64(attrs, 6, 2);
    let xn = extract64(attrs, 54, 1) != 0;
    let pxn = extract64(attrs, 53, 1) != 0;

    // get_S1prot().
    let is_user = mmu_idx == MMU_IDX_E10_0;
    let user_rw = simple_ap_to_rw_prot_is_user(ap, true);
    let mut prot_rw = simple_ap_to_rw_prot_is_user(ap, false);
    if is_user {
        prot_rw = user_rw;
    } else if user_rw != 0 && mmu_idx == MMU_IDX_E10_1_PAN {
        // PAN forbids data accesses if EL0 has data permissions.
        prot_rw = 0;
    }
    let wxn = st.sctlr_el[1] & SCTLR_WXN != 0;
    let xn = if is_user { xn } else { pxn || (user_rw & page::WRITE) != 0 };
    let prot =
        if xn || (wxn && prot_rw & page::WRITE != 0) { prot_rw } else { prot_rw | page::EXEC };

    if prot & access_bit(access) == 0 {
        return Err(Fault::new(fsc::permission(level)));
    }

    // If FEAT_HAFDBS has made changes, update the descriptor.
    if new_descriptor != descriptor {
        // QEMU uses a compare and swap here; see the module doc of `tcg`.
        let res = as_.store(desc_pa, 8, new_descriptor, Endian::Little, MemTxAttrs::default());
        if !res.is_ok() {
            return Err(Fault {
                fsc: fsc::sync_external_on_walk(level),
                ea: res != MemTxResult::DECODE_ERROR,
            });
        }
    }

    let attrindx = extract64(attrs, 2, 3) as u32;
    let mair_attr = (st.mair_el[1] >> (8 * attrindx)) as u8;
    let sh = if mair_attr & 0xf0 == 0 || mair_attr == 0x44 || mair_attr == 0x40 {
        2
    } else {
        extract64(attrs, 8, 2) as u8
    };
    Ok(Translation { pa: descaddr, prot, attrs: mair_attr, sh })
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
    match get_phys_addr(arm, cpu, address, access, mmu_idx, false) {
        Ok(t) => {
            cpu.tlb_set_page(address & !0xfff, t.pa & !0xfff, t.prot, mmu_idx, 4096);
            Ok(true)
        }
        Err(_) if probe => Ok(false),
        Err(fault) => Err(deliver_fault(cpu, address, access, fault, ra)),
    }
}

/// `arm_deliver_fault()` for a fault taken to EL1.
pub(crate) fn deliver_fault(
    cpu: &mut Cpu<'_>,
    addr: u64,
    access: MmuAccessType,
    fault: Fault,
    ra: Ra,
) -> CpuLoopExit {
    let mut st = CpuArmState::load(cpu.env);
    let target_el = 1;
    let same_el = st.current_el() == target_el;
    let (excp, syn) = if access == MmuAccessType::InstFetch {
        (EXCP_PREFETCH_ABORT, syn_insn_abort(same_el, fault.ea, false, fault.fsc))
    } else {
        let wnr = access == MmuAccessType::DataStore;
        (EXCP_DATA_ABORT, syn_data_abort_no_iss(same_el, fault.ea, false, wnr, fault.fsc))
    };
    st.exception_vaddress = addr;
    st.exception_syndrome = syn;
    st.exception_target_el = target_el;
    st.store(cpu.env);
    cpu.raise_exception(excp, ra)
}
