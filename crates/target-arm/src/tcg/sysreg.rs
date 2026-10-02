// SPDX-License-Identifier: GPL-2.0-or-later

//! The AArch64 system registers of EL0 to EL3: the port of the parts of QEMU's
//! `target/arm/helper.c` (`v8_cp_reginfo`, `vmsa_cp_reginfo`, `el2_cp_reginfo`,
//! `el3_cp_reginfo`, `generic_timer_cp_reginfo`, the VHE redirections of
//! `define_arm_vh_e2h_redirect_aliases()`, the ID registers and the TLBI, AT and cache
//! maintenance operations) and `debug_helper.c` that this port needs.
//!
//! Each register is a [`Reg`]: its encoding, the static access rights QEMU keeps in
//! `ARMCPRegInfo.access`, the trap that its `accessfn` checks at run time, and how it is read
//! and written. Plain storage is accessed by generated code at an `env` offset; the rest goes
//! through the `get_sysreg` and `set_sysreg` helpers, which call [`read`] and [`write`].
//!
//! The VHE redirections are resolved at translation time by [`resolve`], which is exact
//! because the TB flags hold the EL and HCR_EL2.E2H: at EL2 with E2H set an EL1 register
//! encoding names the EL2 register, and the `_EL12` and `_EL02` encodings (op1 5) name the
//! EL1 and EL0 registers; without E2H those encodings are UNDEFINED.
//!
//! Differences from QEMU:
//!
//! - On a CPU with EL3 but no EL2 every EL2 register reads as zero and ignores writes, where
//!   QEMU lists the registers that do so one by one in `el3_no_el2_cp_reginfo`.
//! - The debug registers are the few the EL1 slice had; MDCR_EL2 and MDCR_EL3 are stored but
//!   their debug traps are not checked, nor are the HSTR_EL2 and HCR_EL2.TIDCP traps.
//! - TLBI IPAS2E1 and IPAS2LE1 have no effect: this port caches no stage 2 walk results
//!   apart from the combined stage 1 and 2 entries, which the architecture requires the
//!   hypervisor to invalidate with a stage 1 TLBI (VMALLE1 or VMALLS12E1) afterwards.
//! - The TLB has no VMID or ASID tags, so a change of VTTBR_EL2 flushes the EL1&0 regime
//!   and the by-ASID TLBIs flush the whole regime, as QEMU does.
//! - A change of SCR_EL3.NS flushes the whole TLB; there is no Secure address space.

use std::mem::offset_of;

use ruvm_jit::cputlb::{
    tlb_flush, tlb_flush_by_mmuidx, tlb_flush_by_mmuidx_all_cpus_synced,
    tlb_flush_page_bits_by_mmuidx, tlb_flush_page_bits_by_mmuidx_all_cpus_synced,
};
use ruvm_jit::{Cpu, MmuAccessType};

use super::{Arm, PsciConduit, arm_of, gtimer, ptw, regime_has_2_ranges, vfp};
use crate::cpu::{
    ArmCpuModel, ArmFeatures, CpuArmState, GTIMER_HYP, GTIMER_HYPVIRT, GTIMER_PHYS, GTIMER_SEC,
    GTIMER_VIRT, HCR_DC, HCR_E2H, HCR_FWB, HCR_NV, HCR_NV1, HCR_PTW, HCR_TACR, HCR_TDZ, HCR_TGE,
    HCR_TID1, HCR_TID2, HCR_TID3, HCR_TPCP, HCR_TPU, HCR_TRVM, HCR_TSW, HCR_TTLB, HCR_TVM, HCR_VM,
    MMU_IDX_E2, MMU_IDX_E3, MMU_IDX_E10_0, MMU_IDX_E10_1, MMU_IDX_E10_1_PAN, MMU_IDX_E20_0,
    MMU_IDX_E20_2, MMU_IDX_E20_2_PAN, PSTATE_DAIF, PSTATE_PAN, PSTATE_SP, PSTATE_UAO, SCR_NS,
    SCTLR_DZE, SCTLR_UCI, SCTLR_UCT, SCTLR_UMA, env_off,
};

/// `PL3_R`.
pub(crate) const PL3_R: u8 = 0x80;
/// `PL3_W`.
pub(crate) const PL3_W: u8 = 0x40;
/// `PL2_R`: readable at EL2 (and so at EL3).
pub(crate) const PL2_R: u8 = 0x20 | PL3_R;
/// `PL2_W`.
pub(crate) const PL2_W: u8 = 0x10 | PL3_W;
/// `PL1_R`.
pub(crate) const PL1_R: u8 = 0x08 | PL2_R;
/// `PL1_W`.
pub(crate) const PL1_W: u8 = 0x04 | PL2_W;
/// `PL0_R`: readable at EL0 (and so at every EL).
pub(crate) const PL0_R: u8 = 0x02 | PL1_R;
/// `PL0_W`.
pub(crate) const PL0_W: u8 = 0x01 | PL1_W;
/// `PL3_RW`.
pub(crate) const PL3_RW: u8 = PL3_R | PL3_W;
/// `PL2_RW`.
pub(crate) const PL2_RW: u8 = PL2_R | PL2_W;
/// `PL1_RW`.
pub(crate) const PL1_RW: u8 = PL1_R | PL1_W;
/// `PL0_RW`.
pub(crate) const PL0_RW: u8 = PL0_R | PL0_W;

/// The encoding of a system register as one number, `ENCODE_AA64_CP_REG()` without the
/// coprocessor bits.
pub(crate) const fn key(op0: u32, op1: u32, crn: u32, crm: u32, op2: u32) -> u32 {
    (op0 << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2
}

const fn key_op1(k: u32) -> u32 {
    (k >> 11) & 7
}

const fn with_op1(k: u32, op1: u32) -> u32 {
    (k & !(7 << 11)) | (op1 << 11)
}

/// `cp_access_ok()`: whether the static access rights allow the access at `el`.
pub(crate) fn access_ok(access: u8, el: u32, isread: bool) -> bool {
    (access >> (el * 2 + u32::from(isread))) & 1 != 0
}

/// The run time access check of a register, its `accessfn`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Trap {
    /// No check.
    None,
    /// `ctr_el0_access()`: SCTLR.UCT and HCR_EL2.TID2.
    Uct,
    /// `aa64_zva_access()`: SCTLR.DZE and HCR_EL2.TDZ.
    Dze,
    /// `aa64_daif_access()`: SCTLR.UMA.
    Uma,
    /// `aa64_cacheop_poc_access()`: SCTLR.UCI and HCR_EL2.TPCP.
    Poc,
    /// `aa64_cacheop_pou_access()`: SCTLR.UCI and HCR_EL2.TPU.
    Pou,
    /// `access_tsw()`: HCR_EL2.TSW for the set/way operations.
    Tsw,
    /// `gt_cntfrq_access()`.
    CntFrq,
    /// `gt_pct_access()`.
    CntPct,
    /// `gt_vct_access()`.
    CntVct,
    /// `gt_ptimer_access()`.
    CntPTimer,
    /// `gt_vtimer_access()`.
    CntVTimer,
    /// `gt_stimer_access()`.
    CntSTimer,
    /// `sp_el0_access()`: SP_EL0 is not accessible while it is the current SP.
    SpEl0,
    /// `access_tvm_trvm()`: HCR_EL2.TVM for writes and TRVM for reads at EL1.
    Tvm,
    /// `access_aa64_tid1()`.
    Tid1,
    /// `access_aa64_tid2()`.
    Tid2,
    /// `access_aa64_tid3()`.
    Tid3,
    /// `access_tacr()`.
    Tacr,
    /// `access_ttlb()`.
    Ttlb,
    /// `cpacr_access()`: CPTR_EL2.TCPAC and CPTR_EL3.TCPAC.
    Cpacr,
    /// `cptr_access()`: CPTR_EL3.TCPAC for CPTR_EL2.
    Cptr,
}

/// What a run time access check decides, `CPAccessResult`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Access {
    /// `CP_ACCESS_OK`.
    Ok,
    /// `CP_ACCESS_TRAP_EL1`: a system register trap to EL1 (EL2 under HCR_EL2.TGE).
    TrapEl1,
    /// `CP_ACCESS_TRAP_EL2`.
    TrapEl2,
    /// `CP_ACCESS_TRAP_EL3`.
    TrapEl3,
    /// `CP_ACCESS_UNDEFINED`: an uncategorized UNDEF.
    Undefined,
}

/// `arm_sctlr(env, 0)`: the SCTLR that controls EL0.
pub(crate) fn sctlr_el0(f: &ArmFeatures, st: &CpuArmState) -> u64 {
    let hcr = st.hcr_el2_eff(f);
    if hcr & (HCR_E2H | HCR_TGE) == HCR_E2H | HCR_TGE { st.sctlr_el[2] } else { st.sctlr_el[1] }
}

/// `aa64_zva_access()`, also used for the DZP bit of DCZID_EL0.
pub(crate) fn zva_access(f: &ArmFeatures, st: &CpuArmState) -> Access {
    let cur_el = st.current_el();
    if cur_el < 2 {
        let hcr = st.hcr_el2_eff(f);
        if cur_el == 0 {
            if hcr & (HCR_E2H | HCR_TGE) == HCR_E2H | HCR_TGE {
                if st.sctlr_el[2] & SCTLR_DZE == 0 {
                    return Access::TrapEl2;
                }
            } else {
                if st.sctlr_el[1] & SCTLR_DZE == 0 {
                    return Access::TrapEl1;
                }
                if hcr & HCR_TDZ != 0 {
                    return Access::TrapEl2;
                }
            }
        } else if hcr & HCR_TDZ != 0 {
            return Access::TrapEl2;
        }
    }
    Access::Ok
}

impl Trap {
    /// Whether the check can fail at `el`, so that the translator has to call it.
    pub(crate) fn applies(self, el: u32) -> bool {
        match self {
            Trap::None => false,
            Trap::SpEl0 | Trap::CntSTimer => true,
            Trap::Uma => el == 0,
            _ => el < 3,
        }
    }

    /// Run the check.
    pub(crate) fn check(self, f: &ArmFeatures, st: &CpuArmState, isread: bool) -> Access {
        let el = st.current_el();
        if !self.applies(el) {
            return Access::Ok;
        }
        let hcr = st.hcr_el2_eff(f);
        let el1_hcr = |bit: u64| {
            if el == 1 && hcr & bit != 0 { Access::TrapEl2 } else { Access::Ok }
        };
        match self {
            Trap::None => Access::Ok,
            Trap::Uct => {
                if el == 0 {
                    if hcr & (HCR_E2H | HCR_TGE) == HCR_E2H | HCR_TGE {
                        if st.sctlr_el[2] & SCTLR_UCT == 0 {
                            return Access::TrapEl2;
                        }
                    } else if st.sctlr_el[1] & SCTLR_UCT == 0 {
                        return Access::TrapEl1;
                    } else if hcr & HCR_TID2 != 0 {
                        return Access::TrapEl2;
                    }
                } else if el == 1 && hcr & HCR_TID2 != 0 {
                    return Access::TrapEl2;
                }
                Access::Ok
            }
            Trap::Dze => zva_access(f, st),
            Trap::Uma => {
                if sctlr_el0(f, st) & SCTLR_UMA == 0 {
                    Access::TrapEl1
                } else {
                    Access::Ok
                }
            }
            Trap::Poc | Trap::Pou => {
                if el == 0 && sctlr_el0(f, st) & SCTLR_UCI == 0 {
                    return Access::TrapEl1;
                }
                let bit = if self == Trap::Poc { HCR_TPCP } else { HCR_TPU };
                if el < 2 && hcr & bit != 0 { Access::TrapEl2 } else { Access::Ok }
            }
            Trap::Tsw => el1_hcr(HCR_TSW),
            Trap::CntFrq => gtimer::cntfrq_access(f, st, isread),
            Trap::CntPct => gtimer::counter_access(f, st, GTIMER_PHYS),
            Trap::CntVct => gtimer::counter_access(f, st, GTIMER_VIRT),
            Trap::CntPTimer => gtimer::timer_access(f, st, GTIMER_PHYS),
            Trap::CntVTimer => gtimer::timer_access(f, st, GTIMER_VIRT),
            Trap::CntSTimer => gtimer::stimer_access(f, st),
            Trap::SpEl0 => {
                if st.pstate & PSTATE_SP == 0 {
                    // When SPSel is 0, SP_EL0 is the current SP and is inaccessible.
                    Access::Undefined
                } else {
                    Access::Ok
                }
            }
            Trap::Tvm => el1_hcr(if isread { HCR_TRVM } else { HCR_TVM }),
            Trap::Tid1 => el1_hcr(HCR_TID1),
            Trap::Tid2 => el1_hcr(HCR_TID2),
            Trap::Tid3 => el1_hcr(HCR_TID3),
            Trap::Tacr => el1_hcr(HCR_TACR),
            Trap::Ttlb => el1_hcr(HCR_TTLB),
            Trap::Cpacr => {
                // Check if CPACR accesses are to be trapped to EL2, then EL3 (TCPAC is bit
                // 31 of both CPTR registers).
                if el == 1 && st.is_el2_enabled(f) && st.cptr_el[2] & (1 << 31) != 0 {
                    Access::TrapEl2
                } else if el < 3 && f.el3 && st.cptr_el[3] & (1 << 31) != 0 {
                    Access::TrapEl3
                } else {
                    Access::Ok
                }
            }
            Trap::Cptr => {
                if el == 2 && f.el3 && st.cptr_el[3] & (1 << 31) != 0 {
                    Access::TrapEl3
                } else {
                    Access::Ok
                }
            }
        }
    }
}

/// How a register is read and written.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Kind {
    /// CurrentEL: the EL, known at translation time.
    CurrentEl,
    /// A constant of the CPU model.
    Model(fn(&ArmCpuModel) -> u64),
    /// Storage in `env` at `off`; writes keep the bits in `mask`.
    Field {
        /// The `env` offset.
        off: usize,
        /// The writable bits.
        mask: u64,
    },
    /// Reads as zero, writes ignored (`ARM_CP_CONST` with value 0).
    Zero,
    /// An operation with no effect here (`ARM_CP_NOP`).
    Nop,
    /// Read and written by [`read`] and [`write`] through helpers.
    Special,
    /// DC ZVA (`ARM_CP_DC_ZVA`).
    DcZva,
}

/// A system register, the parts of `ARMCPRegInfo` this port uses.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Reg {
    /// The name, as in QEMU's tables; only the table check below reads it.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) name: &'static str,
    /// The encoding, see [`key`].
    pub(crate) key: u32,
    /// The static access rights.
    pub(crate) access: u8,
    /// The run time check.
    pub(crate) trap: Trap,
    /// How it is accessed.
    pub(crate) kind: Kind,
    /// Whether the model has it.
    pub(crate) feat: fn(&ArmFeatures) -> bool,
}

fn always(_: &ArmFeatures) -> bool {
    true
}

fn has_pan(f: &ArmFeatures) -> bool {
    f.pan
}

fn has_uao(f: &ArmFeatures) -> bool {
    f.uao
}

fn has_lor(f: &ArmFeatures) -> bool {
    f.lor
}

fn has_dpb(f: &ArmFeatures) -> bool {
    f.dpb
}

fn has_vh(f: &ArmFeatures) -> bool {
    f.vh
}

fn has_sve(f: &ArmFeatures) -> bool {
    f.sve
}

/// The `env` offset of element `i` of an array field.
macro_rules! off {
    ($f:ident) => {
        env_off(offset_of!(CpuArmState, $f))
    };
    ($f:ident[$i:expr]) => {
        env_off(offset_of!(CpuArmState, $f)) + 8 * $i
    };
}

macro_rules! r {
    ($name:literal, ($op0:expr, $op1:expr, $crn:expr, $crm:expr, $op2:expr), $access:expr,
     $trap:ident, $kind:expr) => {
        r!($name, ($op0, $op1, $crn, $crm, $op2), $access, $trap, $kind, always)
    };
    ($name:literal, ($op0:expr, $op1:expr, $crn:expr, $crm:expr, $op2:expr), $access:expr,
     $trap:ident, $kind:expr, $feat:expr) => {
        Reg {
            name: $name,
            key: key($op0, $op1, $crn, $crm, $op2),
            access: $access,
            trap: Trap::$trap,
            kind: $kind,
            feat: $feat,
        }
    };
}

const fn field(off: usize) -> Kind {
    Kind::Field { off, mask: u64::MAX }
}

const fn field32(off: usize) -> Kind {
    Kind::Field { off, mask: 0xffff_ffff }
}

/// MIDR_EL1.
pub(crate) const MIDR_EL1: u32 = key(3, 0, 0, 0, 0);
/// SCTLR_EL1.
pub(crate) const SCTLR_EL1: u32 = key(3, 0, 1, 0, 0);
/// TTBR0_EL1.
pub(crate) const TTBR0_EL1: u32 = key(3, 0, 2, 0, 0);
/// TTBR1_EL1.
pub(crate) const TTBR1_EL1: u32 = key(3, 0, 2, 0, 1);
/// TCR_EL1.
pub(crate) const TCR_EL1: u32 = key(3, 0, 2, 0, 2);
/// NZCV.
pub(crate) const NZCV: u32 = key(3, 3, 4, 2, 0);
/// DAIF.
pub(crate) const DAIF: u32 = key(3, 3, 4, 2, 1);
/// FPCR (`ARM_CP_FPU`).
pub(crate) const FPCR: u32 = key(3, 3, 4, 4, 0);
/// FPSR (`ARM_CP_FPU | ARM_CP_SUPPRESS_TB_END`).
pub(crate) const FPSR: u32 = key(3, 3, 4, 4, 1);
/// ZCR_EL1 (`ARM_CP_SVE`).
pub(crate) const ZCR_EL1: u32 = key(3, 0, 1, 2, 0);
/// ZCR_EL2 (`ARM_CP_SVE`).
pub(crate) const ZCR_EL2: u32 = key(3, 4, 1, 2, 0);
/// ZCR_EL3 (`ARM_CP_SVE`).
pub(crate) const ZCR_EL3: u32 = key(3, 6, 1, 2, 0);
/// SPSel.
pub(crate) const SPSEL: u32 = key(3, 0, 4, 2, 0);
/// PAN.
pub(crate) const PAN: u32 = key(3, 0, 4, 2, 3);
/// UAO.
pub(crate) const UAO: u32 = key(3, 0, 4, 2, 4);
/// MPIDR_EL1.
pub(crate) const MPIDR_EL1: u32 = key(3, 0, 0, 0, 5);
/// CCSIDR_EL1.
pub(crate) const CCSIDR_EL1: u32 = key(3, 1, 0, 0, 0);
/// CTR_EL0.
pub(crate) const CTR_EL0: u32 = key(3, 3, 0, 0, 1);
/// DCZID_EL0.
pub(crate) const DCZID_EL0: u32 = key(3, 3, 0, 0, 7);
/// OSLAR_EL1.
pub(crate) const OSLAR_EL1: u32 = key(2, 0, 1, 0, 4);
/// OSLSR_EL1.
pub(crate) const OSLSR_EL1: u32 = key(2, 0, 1, 1, 4);
/// CNTPCT_EL0.
pub(crate) const CNTPCT_EL0: u32 = key(3, 3, 14, 0, 1);
/// CNTVCT_EL0.
pub(crate) const CNTVCT_EL0: u32 = key(3, 3, 14, 0, 2);
/// SCTLR_EL2.
pub(crate) const SCTLR_EL2: u32 = key(3, 4, 1, 0, 0);
/// HCR_EL2.
pub(crate) const HCR_EL2: u32 = key(3, 4, 1, 1, 0);
/// TTBR0_EL2.
pub(crate) const TTBR0_EL2: u32 = key(3, 4, 2, 0, 0);
/// TTBR1_EL2.
pub(crate) const TTBR1_EL2: u32 = key(3, 4, 2, 0, 1);
/// TCR_EL2.
pub(crate) const TCR_EL2: u32 = key(3, 4, 2, 0, 2);
/// VTTBR_EL2.
pub(crate) const VTTBR_EL2: u32 = key(3, 4, 2, 1, 0);
/// VTCR_EL2.
pub(crate) const VTCR_EL2: u32 = key(3, 4, 2, 1, 2);
/// CNTVOFF_EL2.
pub(crate) const CNTVOFF_EL2: u32 = key(3, 4, 14, 0, 3);
/// CNTHCTL_EL2.
pub(crate) const CNTHCTL_EL2: u32 = key(3, 4, 14, 1, 0);
/// SCTLR_EL3.
pub(crate) const SCTLR_EL3: u32 = key(3, 6, 1, 0, 0);
/// SCR_EL3.
pub(crate) const SCR_EL3: u32 = key(3, 6, 1, 1, 0);
/// TTBR0_EL3.
pub(crate) const TTBR0_EL3: u32 = key(3, 6, 2, 0, 0);
/// TCR_EL3.
pub(crate) const TCR_EL3: u32 = key(3, 6, 2, 0, 2);

/// The timer registers: op0 3, CRn 14, CRm 2 (physical) or 3 (virtual), op2 0 (TVAL), 1
/// (CTL) or 2 (CVAL). op1 3 is the EL0 view, 4 the EL2 timers, 5 the EL02 aliases and 7 the
/// Secure timer.
fn timer_reg(k: u32) -> Option<(u32, u32, u32)> {
    let op0 = k >> 14;
    let crn = (k >> 7) & 0xf;
    let crm = (k >> 3) & 0xf;
    let op2 = k & 7;
    if op0 == 3 && crn == 14 && (crm == 2 || crm == 3) && op2 <= 2 {
        Some((key_op1(k), crm, op2))
    } else {
        None
    }
}

const AT_S1E1R: u32 = key(1, 0, 7, 8, 0);
const AT_S1E1W: u32 = key(1, 0, 7, 8, 1);
const AT_S1E0R: u32 = key(1, 0, 7, 8, 2);
const AT_S1E0W: u32 = key(1, 0, 7, 8, 3);
const AT_S1E2R: u32 = key(1, 4, 7, 8, 0);
const AT_S1E2W: u32 = key(1, 4, 7, 8, 1);
const AT_S12E1R: u32 = key(1, 4, 7, 8, 4);
const AT_S12E1W: u32 = key(1, 4, 7, 8, 5);
const AT_S12E0R: u32 = key(1, 4, 7, 8, 6);
const AT_S12E0W: u32 = key(1, 4, 7, 8, 7);
const AT_S1E3R: u32 = key(1, 6, 7, 8, 0);
const AT_S1E3W: u32 = key(1, 6, 7, 8, 1);

macro_rules! tlbi {
    ($name:literal, $op1:expr, $crm:expr, $op2:expr, $access:expr, $trap:ident) => {
        r!($name, (1, $op1, 8, $crm, $op2), $access, $trap, Kind::Special)
    };
}

macro_rules! at {
    ($name:literal, $op1:expr, $op2:expr, $access:expr) => {
        r!($name, (1, $op1, 7, 8, $op2), $access, None, Kind::Special)
    };
}

/// The EL0 and EL1 registers, in the order of the Arm ARM's register index.
static REGS: &[Reg] = &[
    // Identification.
    r!("MIDR_EL1", (3, 0, 0, 0, 0), PL1_R, None, Kind::Special),
    r!("MPIDR_EL1", (3, 0, 0, 0, 5), PL1_R, None, Kind::Special),
    r!("REVIDR_EL1", (3, 0, 0, 0, 6), PL1_R, Tid1, Kind::Model(|m| m.revidr)),
    r!("ID_AA64PFR0_EL1", (3, 0, 0, 4, 0), PL1_R, Tid3, Kind::Model(|m| m.id_aa64pfr0)),
    r!("ID_AA64PFR1_EL1", (3, 0, 0, 4, 1), PL1_R, Tid3, Kind::Model(|m| m.id_aa64pfr1)),
    r!("ID_AA64ZFR0_EL1", (3, 0, 0, 4, 4), PL1_R, Tid3, Kind::Model(|m| m.id_aa64zfr0)),
    r!("ID_AA64DFR0_EL1", (3, 0, 0, 5, 0), PL1_R, Tid3, Kind::Model(|m| m.id_aa64dfr0)),
    r!("ID_AA64ISAR0_EL1", (3, 0, 0, 6, 0), PL1_R, Tid3, Kind::Model(|m| m.id_aa64isar0)),
    r!("ID_AA64ISAR1_EL1", (3, 0, 0, 6, 1), PL1_R, Tid3, Kind::Model(|m| m.id_aa64isar1)),
    r!("ID_AA64MMFR0_EL1", (3, 0, 0, 7, 0), PL1_R, Tid3, Kind::Model(|m| m.id_aa64mmfr0)),
    r!("ID_AA64MMFR1_EL1", (3, 0, 0, 7, 1), PL1_R, Tid3, Kind::Model(|m| m.id_aa64mmfr1)),
    r!("ID_AA64MMFR2_EL1", (3, 0, 0, 7, 2), PL1_R, Tid3, Kind::Model(|m| m.id_aa64mmfr2)),
    r!("CCSIDR_EL1", (3, 1, 0, 0, 0), PL1_R, Tid2, Kind::Special),
    r!("CLIDR_EL1", (3, 1, 0, 0, 1), PL1_R, Tid2, Kind::Model(|m| m.clidr)),
    r!("AIDR_EL1", (3, 1, 0, 0, 7), PL1_R, Tid1, Kind::Zero),
    r!("CSSELR_EL1", (3, 2, 0, 0, 0), PL1_RW, Tid2, field32(off!(csselr_el1))),
    r!("CTR_EL0", (3, 3, 0, 0, 1), PL0_R, Uct, Kind::Special),
    r!("DCZID_EL0", (3, 3, 0, 0, 7), PL0_R, None, Kind::Special),
    // System control.
    r!("SCTLR_EL1", (3, 0, 1, 0, 0), PL1_RW, Tvm, Kind::Special),
    r!("ACTLR_EL1", (3, 0, 1, 0, 1), PL1_RW, Tacr, Kind::Zero),
    r!(
        "CPACR_EL1",
        (3, 0, 1, 0, 2),
        PL1_RW,
        Cpacr,
        // cpacr_write() keeps every bit in ARMv8.
        field(off!(cpacr_el1))
    ),
    // zcr_reginfo, all three in this table as QEMU registers them whatever the ELs.
    r!("ZCR_EL1", (3, 0, 1, 2, 0), PL1_RW, None, Kind::Special, has_sve),
    r!("ZCR_EL2", (3, 4, 1, 2, 0), PL2_RW, None, Kind::Special, has_sve),
    r!("ZCR_EL3", (3, 6, 1, 2, 0), PL3_RW, None, Kind::Special, has_sve),
    // Memory management.
    r!("TTBR0_EL1", (3, 0, 2, 0, 0), PL1_RW, Tvm, Kind::Special),
    r!("TTBR1_EL1", (3, 0, 2, 0, 1), PL1_RW, Tvm, Kind::Special),
    r!("TCR_EL1", (3, 0, 2, 0, 2), PL1_RW, Tvm, Kind::Special),
    // Exception handling.
    r!("SPSR_EL1", (3, 0, 4, 0, 0), PL1_RW, None, field(off!(spsr_el[1]))),
    r!("ELR_EL1", (3, 0, 4, 0, 1), PL1_RW, None, field(off!(elr_el[1]))),
    r!("SP_EL0", (3, 0, 4, 1, 0), PL1_RW, SpEl0, field(off!(sp_el[0]))),
    r!("SPSel", (3, 0, 4, 2, 0), PL1_RW, None, Kind::Special),
    r!("CurrentEL", (3, 0, 4, 2, 2), PL1_R, None, Kind::CurrentEl),
    r!("PAN", (3, 0, 4, 2, 3), PL1_RW, None, Kind::Special, has_pan),
    r!("UAO", (3, 0, 4, 2, 4), PL1_RW, None, Kind::Special, has_uao),
    r!("NZCV", (3, 3, 4, 2, 0), PL0_RW, None, Kind::Special),
    r!("DAIF", (3, 3, 4, 2, 1), PL0_RW, Uma, Kind::Special),
    r!("FPCR", (3, 3, 4, 4, 0), PL0_RW, None, Kind::Special),
    r!("FPSR", (3, 3, 4, 4, 1), PL0_RW, None, Kind::Special),
    r!("AFSR0_EL1", (3, 0, 5, 1, 0), PL1_RW, Tvm, field(off!(afsr0_el1))),
    r!("AFSR1_EL1", (3, 0, 5, 1, 1), PL1_RW, Tvm, field(off!(afsr1_el1))),
    r!("ESR_EL1", (3, 0, 5, 2, 0), PL1_RW, Tvm, field(off!(esr_el[1]))),
    r!("FAR_EL1", (3, 0, 6, 0, 0), PL1_RW, Tvm, field(off!(far_el[1]))),
    r!("PAR_EL1", (3, 0, 7, 4, 0), PL1_RW, None, field(off!(par_el1))),
    r!("MAIR_EL1", (3, 0, 10, 2, 0), PL1_RW, Tvm, field(off!(mair_el[1]))),
    r!("AMAIR_EL1", (3, 0, 10, 3, 0), PL1_RW, Tvm, field(off!(amair_el1))),
    r!("LORSA_EL1", (3, 0, 10, 4, 0), PL1_RW, None, Kind::Zero, has_lor),
    r!("LOREA_EL1", (3, 0, 10, 4, 1), PL1_RW, None, Kind::Zero, has_lor),
    r!("LORN_EL1", (3, 0, 10, 4, 2), PL1_RW, None, Kind::Zero, has_lor),
    r!("LORC_EL1", (3, 0, 10, 4, 3), PL1_RW, None, Kind::Zero, has_lor),
    r!("LORID_EL1", (3, 0, 10, 4, 7), PL1_R, None, Kind::Zero, has_lor),
    r!(
        "VBAR_EL1",
        (3, 0, 12, 0, 0),
        PL1_RW,
        None,
        Kind::Field { off: off!(vbar_el[1]), mask: !0x1f }
    ),
    r!("CONTEXTIDR_EL1", (3, 0, 13, 0, 1), PL1_RW, Tvm, field32(off!(contextidr_el1))),
    r!("TPIDR_EL1", (3, 0, 13, 0, 4), PL1_RW, None, field(off!(tpidr_el[1]))),
    r!("TPIDR_EL0", (3, 3, 13, 0, 2), PL0_RW, None, field(off!(tpidr_el[0]))),
    r!("TPIDRRO_EL0", (3, 3, 13, 0, 3), PL0_R | PL1_W, None, field(off!(tpidrro_el0))),
    // Generic timer.
    r!("CNTKCTL_EL1", (3, 0, 14, 1, 0), PL1_RW, None, field32(off!(cntkctl_el1))),
    r!("CNTFRQ_EL0", (3, 3, 14, 0, 0), PL0_R | PL1_W, CntFrq, field32(off!(cntfrq_el0))),
    r!("CNTPCT_EL0", (3, 3, 14, 0, 1), PL0_R, CntPct, Kind::Special),
    r!("CNTVCT_EL0", (3, 3, 14, 0, 2), PL0_R, CntVct, Kind::Special),
    r!("CNTP_TVAL_EL0", (3, 3, 14, 2, 0), PL0_RW, CntPTimer, Kind::Special),
    r!("CNTP_CTL_EL0", (3, 3, 14, 2, 1), PL0_RW, CntPTimer, Kind::Special),
    r!("CNTP_CVAL_EL0", (3, 3, 14, 2, 2), PL0_RW, CntPTimer, Kind::Special),
    r!("CNTV_TVAL_EL0", (3, 3, 14, 3, 0), PL0_RW, CntVTimer, Kind::Special),
    r!("CNTV_CTL_EL0", (3, 3, 14, 3, 1), PL0_RW, CntVTimer, Kind::Special),
    r!("CNTV_CVAL_EL0", (3, 3, 14, 3, 2), PL0_RW, CntVTimer, Kind::Special),
    // Debug.
    r!("MDCCINT_EL1", (2, 0, 0, 2, 0), PL1_RW, None, Kind::Zero),
    r!("MDSCR_EL1", (2, 0, 0, 2, 2), PL1_RW, None, field32(off!(mdscr_el1))),
    r!("DBGBVR0_EL1", (2, 0, 0, 0, 4), PL1_RW, None, Kind::Zero),
    r!("DBGBCR0_EL1", (2, 0, 0, 0, 5), PL1_RW, None, Kind::Zero),
    r!("DBGWVR0_EL1", (2, 0, 0, 0, 6), PL1_RW, None, Kind::Zero),
    r!("DBGWCR0_EL1", (2, 0, 0, 0, 7), PL1_RW, None, Kind::Zero),
    r!("MDRAR_EL1", (2, 0, 1, 0, 0), PL1_R, None, Kind::Zero),
    r!("OSLAR_EL1", (2, 0, 1, 0, 4), PL1_W, None, Kind::Special),
    r!("OSLSR_EL1", (2, 0, 1, 1, 4), PL1_R, None, Kind::Special),
    r!("OSDLR_EL1", (2, 0, 1, 3, 4), PL1_RW, None, Kind::Field { off: off!(osdlr_el1), mask: 1 }),
    // Cache maintenance.
    r!("IC_IALLUIS", (1, 0, 7, 1, 0), PL1_W, Pou, Kind::Nop),
    r!("IC_IALLU", (1, 0, 7, 5, 0), PL1_W, Pou, Kind::Nop),
    r!("IC_IVAU", (1, 3, 7, 5, 1), PL0_W, Pou, Kind::Nop),
    r!("DC_IVAC", (1, 0, 7, 6, 1), PL1_W, Poc, Kind::Nop),
    r!("DC_ISW", (1, 0, 7, 6, 2), PL1_W, Tsw, Kind::Nop),
    r!("DC_CSW", (1, 0, 7, 10, 2), PL1_W, Tsw, Kind::Nop),
    r!("DC_CISW", (1, 0, 7, 14, 2), PL1_W, Tsw, Kind::Nop),
    r!("DC_ZVA", (1, 3, 7, 4, 1), PL0_W, Dze, Kind::DcZva),
    r!("DC_CVAC", (1, 3, 7, 10, 1), PL0_W, Poc, Kind::Nop),
    r!("DC_CVAU", (1, 3, 7, 11, 1), PL0_W, Pou, Kind::Nop),
    r!("DC_CVAP", (1, 3, 7, 12, 1), PL0_W, Poc, Kind::Nop, has_dpb),
    r!("DC_CIVAC", (1, 3, 7, 14, 1), PL0_W, Poc, Kind::Nop),
    // Address translation.
    at!("AT_S1E1R", 0, 0, PL1_W),
    at!("AT_S1E1W", 0, 1, PL1_W),
    at!("AT_S1E0R", 0, 2, PL1_W),
    at!("AT_S1E0W", 0, 3, PL1_W),
    // TLB maintenance.
    tlbi!("TLBI_VMALLE1IS", 0, 3, 0, PL1_W, Ttlb),
    tlbi!("TLBI_VAE1IS", 0, 3, 1, PL1_W, Ttlb),
    tlbi!("TLBI_ASIDE1IS", 0, 3, 2, PL1_W, Ttlb),
    tlbi!("TLBI_VAAE1IS", 0, 3, 3, PL1_W, Ttlb),
    tlbi!("TLBI_VALE1IS", 0, 3, 5, PL1_W, Ttlb),
    tlbi!("TLBI_VAALE1IS", 0, 3, 7, PL1_W, Ttlb),
    tlbi!("TLBI_VMALLE1", 0, 7, 0, PL1_W, Ttlb),
    tlbi!("TLBI_VAE1", 0, 7, 1, PL1_W, Ttlb),
    tlbi!("TLBI_ASIDE1", 0, 7, 2, PL1_W, Ttlb),
    tlbi!("TLBI_VAAE1", 0, 7, 3, PL1_W, Ttlb),
    tlbi!("TLBI_VALE1", 0, 7, 5, PL1_W, Ttlb),
    tlbi!("TLBI_VAALE1", 0, 7, 7, PL1_W, Ttlb),
];

/// The EL2 registers, `el2_cp_reginfo` and the VHE additions.
static EL2_REGS: &[Reg] = &[
    r!("VPIDR_EL2", (3, 4, 0, 0, 0), PL2_RW, None, field(off!(vpidr_el2))),
    r!("VMPIDR_EL2", (3, 4, 0, 0, 5), PL2_RW, None, field(off!(vmpidr_el2))),
    r!("SCTLR_EL2", (3, 4, 1, 0, 0), PL2_RW, None, Kind::Special),
    r!("ACTLR_EL2", (3, 4, 1, 0, 1), PL2_RW, None, Kind::Zero),
    r!("HCR_EL2", (3, 4, 1, 1, 0), PL2_RW, None, Kind::Special),
    r!("MDCR_EL2", (3, 4, 1, 1, 1), PL2_RW, None, field(off!(mdcr_el2))),
    r!("CPTR_EL2", (3, 4, 1, 1, 2), PL2_RW, Cptr, field(off!(cptr_el[2]))),
    r!("HSTR_EL2", (3, 4, 1, 1, 3), PL2_RW, None, field32(off!(hstr_el2))),
    r!("HACR_EL2", (3, 4, 1, 1, 7), PL2_RW, None, Kind::Zero),
    r!("TTBR0_EL2", (3, 4, 2, 0, 0), PL2_RW, None, Kind::Special),
    r!("TTBR1_EL2", (3, 4, 2, 0, 1), PL2_RW, None, Kind::Special, has_vh),
    r!("TCR_EL2", (3, 4, 2, 0, 2), PL2_RW, None, Kind::Special),
    r!("VTTBR_EL2", (3, 4, 2, 1, 0), PL2_RW, None, Kind::Special),
    r!("VTCR_EL2", (3, 4, 2, 1, 2), PL2_RW, None, Kind::Special),
    r!("SPSR_EL2", (3, 4, 4, 0, 0), PL2_RW, None, field(off!(spsr_el[2]))),
    r!("ELR_EL2", (3, 4, 4, 0, 1), PL2_RW, None, field(off!(elr_el[2]))),
    r!("SP_EL1", (3, 4, 4, 1, 0), PL2_RW, None, field(off!(sp_el[1]))),
    r!("AFSR0_EL2", (3, 4, 5, 1, 0), PL2_RW, None, Kind::Zero),
    r!("AFSR1_EL2", (3, 4, 5, 1, 1), PL2_RW, None, Kind::Zero),
    r!("ESR_EL2", (3, 4, 5, 2, 0), PL2_RW, None, field(off!(esr_el[2]))),
    r!("VSESR_EL2", (3, 4, 5, 2, 3), PL2_RW, None, field(off!(vsesr_el2))),
    r!("FAR_EL2", (3, 4, 6, 0, 0), PL2_RW, None, field(off!(far_el[2]))),
    r!("HPFAR_EL2", (3, 4, 6, 0, 4), PL2_RW, None, field(off!(hpfar_el2))),
    r!("MAIR_EL2", (3, 4, 10, 2, 0), PL2_RW, None, field(off!(mair_el[2]))),
    r!("AMAIR_EL2", (3, 4, 10, 3, 0), PL2_RW, None, Kind::Zero),
    r!(
        "VBAR_EL2",
        (3, 4, 12, 0, 0),
        PL2_RW,
        None,
        Kind::Field { off: off!(vbar_el[2]), mask: !0x1f }
    ),
    r!("CONTEXTIDR_EL2", (3, 4, 13, 0, 1), PL2_RW, None, field(off!(contextidr_el2)), has_vh),
    r!("TPIDR_EL2", (3, 4, 13, 0, 2), PL2_RW, None, field(off!(tpidr_el[2]))),
    r!("CNTVOFF_EL2", (3, 4, 14, 0, 3), PL2_RW, None, Kind::Special),
    r!("CNTHCTL_EL2", (3, 4, 14, 1, 0), PL2_RW, None, Kind::Special),
    r!("CNTHP_TVAL_EL2", (3, 4, 14, 2, 0), PL2_RW, None, Kind::Special),
    r!("CNTHP_CTL_EL2", (3, 4, 14, 2, 1), PL2_RW, None, Kind::Special),
    r!("CNTHP_CVAL_EL2", (3, 4, 14, 2, 2), PL2_RW, None, Kind::Special),
    r!("CNTHV_TVAL_EL2", (3, 4, 14, 3, 0), PL2_RW, None, Kind::Special, has_vh),
    r!("CNTHV_CTL_EL2", (3, 4, 14, 3, 1), PL2_RW, None, Kind::Special, has_vh),
    r!("CNTHV_CVAL_EL2", (3, 4, 14, 3, 2), PL2_RW, None, Kind::Special, has_vh),
    // The EL02 timer aliases; resolve() only lets them through at EL2 and EL3 with E2H.
    r!("CNTP_TVAL_EL02", (3, 5, 14, 2, 0), PL2_RW, None, Kind::Special, has_vh),
    r!("CNTP_CTL_EL02", (3, 5, 14, 2, 1), PL2_RW, None, Kind::Special, has_vh),
    r!("CNTP_CVAL_EL02", (3, 5, 14, 2, 2), PL2_RW, None, Kind::Special, has_vh),
    r!("CNTV_TVAL_EL02", (3, 5, 14, 3, 0), PL2_RW, None, Kind::Special, has_vh),
    r!("CNTV_CTL_EL02", (3, 5, 14, 3, 1), PL2_RW, None, Kind::Special, has_vh),
    r!("CNTV_CVAL_EL02", (3, 5, 14, 3, 2), PL2_RW, None, Kind::Special, has_vh),
    at!("AT_S1E2R", 4, 0, PL2_W),
    at!("AT_S1E2W", 4, 1, PL2_W),
    at!("AT_S12E1R", 4, 4, PL2_W),
    at!("AT_S12E1W", 4, 5, PL2_W),
    at!("AT_S12E0R", 4, 6, PL2_W),
    at!("AT_S12E0W", 4, 7, PL2_W),
    tlbi!("TLBI_IPAS2E1IS", 4, 0, 1, PL2_W, None),
    tlbi!("TLBI_IPAS2LE1IS", 4, 0, 5, PL2_W, None),
    tlbi!("TLBI_ALLE2IS", 4, 3, 0, PL2_W, None),
    tlbi!("TLBI_VAE2IS", 4, 3, 1, PL2_W, None),
    tlbi!("TLBI_ALLE1IS", 4, 3, 4, PL2_W, None),
    tlbi!("TLBI_VALE2IS", 4, 3, 5, PL2_W, None),
    tlbi!("TLBI_VMALLS12E1IS", 4, 3, 6, PL2_W, None),
    tlbi!("TLBI_IPAS2E1", 4, 4, 1, PL2_W, None),
    tlbi!("TLBI_IPAS2LE1", 4, 4, 5, PL2_W, None),
    tlbi!("TLBI_ALLE2", 4, 7, 0, PL2_W, None),
    tlbi!("TLBI_VAE2", 4, 7, 1, PL2_W, None),
    tlbi!("TLBI_ALLE1", 4, 7, 4, PL2_W, None),
    tlbi!("TLBI_VALE2", 4, 7, 5, PL2_W, None),
    tlbi!("TLBI_VMALLS12E1", 4, 7, 6, PL2_W, None),
];

/// The EL3 registers, `el3_cp_reginfo` and the Secure timer.
static EL3_REGS: &[Reg] = &[
    r!("SCTLR_EL3", (3, 6, 1, 0, 0), PL3_RW, None, Kind::Special),
    r!("ACTLR_EL3", (3, 6, 1, 0, 1), PL3_RW, None, Kind::Zero),
    r!("SCR_EL3", (3, 6, 1, 1, 0), PL3_RW, None, Kind::Special),
    r!("CPTR_EL3", (3, 6, 1, 1, 2), PL3_RW, None, field(off!(cptr_el[3]))),
    r!("MDCR_EL3", (3, 6, 1, 3, 1), PL3_RW, None, field(off!(mdcr_el3))),
    r!("TTBR0_EL3", (3, 6, 2, 0, 0), PL3_RW, None, Kind::Special),
    r!("TCR_EL3", (3, 6, 2, 0, 2), PL3_RW, None, Kind::Special),
    r!("SPSR_EL3", (3, 6, 4, 0, 0), PL3_RW, None, field(off!(spsr_el[3]))),
    r!("ELR_EL3", (3, 6, 4, 0, 1), PL3_RW, None, field(off!(elr_el[3]))),
    r!("SP_EL2", (3, 6, 4, 1, 0), PL3_RW, None, field(off!(sp_el[2]))),
    r!("AFSR0_EL3", (3, 6, 5, 1, 0), PL3_RW, None, Kind::Zero),
    r!("AFSR1_EL3", (3, 6, 5, 1, 1), PL3_RW, None, Kind::Zero),
    r!("ESR_EL3", (3, 6, 5, 2, 0), PL3_RW, None, field(off!(esr_el[3]))),
    r!("FAR_EL3", (3, 6, 6, 0, 0), PL3_RW, None, field(off!(far_el[3]))),
    r!("MAIR_EL3", (3, 6, 10, 2, 0), PL3_RW, None, field(off!(mair_el[3]))),
    r!("AMAIR_EL3", (3, 6, 10, 3, 0), PL3_RW, None, Kind::Zero),
    r!(
        "VBAR_EL3",
        (3, 6, 12, 0, 0),
        PL3_RW,
        None,
        Kind::Field { off: off!(vbar_el[3]), mask: !0x1f }
    ),
    r!("TPIDR_EL3", (3, 6, 13, 0, 2), PL3_RW, None, field(off!(tpidr_el[3]))),
    r!("CNTPS_TVAL_EL1", (3, 7, 14, 2, 0), PL1_RW, CntSTimer, Kind::Special),
    r!("CNTPS_CTL_EL1", (3, 7, 14, 2, 1), PL1_RW, CntSTimer, Kind::Special),
    r!("CNTPS_CVAL_EL1", (3, 7, 14, 2, 2), PL1_RW, CntSTimer, Kind::Special),
    at!("AT_S1E3R", 6, 0, PL3_W),
    at!("AT_S1E3W", 6, 1, PL3_W),
    tlbi!("TLBI_ALLE3IS", 6, 3, 0, PL3_W, None),
    tlbi!("TLBI_VAE3IS", 6, 3, 1, PL3_W, None),
    tlbi!("TLBI_VALE3IS", 6, 3, 5, PL3_W, None),
    tlbi!("TLBI_ALLE3", 6, 7, 0, PL3_W, None),
    tlbi!("TLBI_VAE3", 6, 7, 1, PL3_W, None),
    tlbi!("TLBI_VALE3", 6, 7, 5, PL3_W, None),
];

/// The EL1 registers that VHE redirects, and the EL2 register each one names at EL2 with
/// E2H set; their `_EL12` encodings (op1 5) name the EL1 register.
static E2H_REDIRECTS: &[(u32, u32)] = &[
    (key(3, 0, 1, 0, 0), key(3, 4, 1, 0, 0)),   // SCTLR
    (key(3, 0, 1, 0, 2), key(3, 4, 1, 1, 2)),   // CPACR, CPTR_EL2
    (key(3, 0, 1, 2, 0), key(3, 4, 1, 2, 0)),   // ZCR
    (key(3, 0, 2, 0, 0), key(3, 4, 2, 0, 0)),   // TTBR0
    (key(3, 0, 2, 0, 1), key(3, 4, 2, 0, 1)),   // TTBR1
    (key(3, 0, 2, 0, 2), key(3, 4, 2, 0, 2)),   // TCR
    (key(3, 0, 4, 0, 0), key(3, 4, 4, 0, 0)),   // SPSR
    (key(3, 0, 4, 0, 1), key(3, 4, 4, 0, 1)),   // ELR
    (key(3, 0, 5, 1, 0), key(3, 4, 5, 1, 0)),   // AFSR0
    (key(3, 0, 5, 1, 1), key(3, 4, 5, 1, 1)),   // AFSR1
    (key(3, 0, 5, 2, 0), key(3, 4, 5, 2, 0)),   // ESR
    (key(3, 0, 6, 0, 0), key(3, 4, 6, 0, 0)),   // FAR
    (key(3, 0, 10, 2, 0), key(3, 4, 10, 2, 0)), // MAIR
    (key(3, 0, 10, 3, 0), key(3, 4, 10, 3, 0)), // AMAIR
    (key(3, 0, 12, 0, 0), key(3, 4, 12, 0, 0)), // VBAR
    (key(3, 0, 13, 0, 1), key(3, 4, 13, 0, 1)), // CONTEXTIDR
    (key(3, 0, 14, 1, 0), key(3, 4, 14, 1, 0)), // CNTKCTL, CNTHCTL_EL2
];

/// The encoding a system instruction at `el` really names, applying the VHE redirections
/// (`e2h` is HCR_EL2.E2H from the TB flags), or `None` when it is UNDEFINED because an
/// `_EL12` or `_EL02` encoding is used without E2H.
pub(crate) fn resolve(k: u32, el: u32, e2h: bool, feat: &ArmFeatures) -> Option<u32> {
    if k >> 14 == 3 && key_op1(k) == 5 {
        // FOO_EL12 aliases only exist when E2H is 1; otherwise they UNDEF.
        if !(feat.vh && e2h && el >= 2) {
            return None;
        }
        let el1 = with_op1(k, 0);
        if E2H_REDIRECTS.iter().any(|&(a, _)| a == el1) {
            return Some(el1);
        }
        // The EL02 timer registers have their own entries.
        return Some(k);
    }
    if el == 2 && e2h {
        if let Some(&(_, el2)) = E2H_REDIRECTS.iter().find(|&&(a, _)| a == k) {
            return Some(el2);
        }
    }
    Some(k)
}

/// The register with encoding `key` on a CPU with `feat`, as `get_arm_cp_reginfo()` finds
/// it. The ID register space that is not listed reads as zero at EL1.
pub(crate) fn lookup(key_: u32, feat: &ArmFeatures) -> Option<Reg> {
    if let Some(r) = REGS.iter().find(|r| r.key == key_ && (r.feat)(feat)) {
        return Some(*r);
    }
    if feat.el2 || feat.el3 {
        if let Some(r) = EL2_REGS.iter().find(|r| r.key == key_ && (r.feat)(feat)) {
            if feat.el2 {
                return Some(*r);
            }
            // EL3 without EL2: the EL2 registers are RES0.
            return Some(Reg { kind: Kind::Zero, trap: Trap::None, ..*r });
        }
    }
    if feat.el3 {
        if let Some(r) = EL3_REGS.iter().find(|r| r.key == key_ && (r.feat)(feat)) {
            return Some(*r);
        }
    }
    let op0 = key_ >> 14;
    let op1 = key_op1(key_);
    let crn = (key_ >> 7) & 0xf;
    let crm = (key_ >> 3) & 0xf;
    if op0 == 3 && op1 == 0 && crn == 0 && (1..=7).contains(&crm) {
        return Some(Reg {
            name: "ID_RAZ",
            key: key_,
            access: PL1_R,
            trap: Trap::Tid3,
            kind: Kind::Zero,
            feat: always,
        });
    }
    None
}

/// The cache geometry CCSIDR_EL1 reports for the cache CSSELR_EL1 selects: a 32 KiB L1 data
/// cache, a 48 KiB L1 instruction cache and a 2 MiB L2, with 64 byte lines.
fn ccsidr(csselr: u64) -> u64 {
    match csselr {
        0 => 0x701f_e00a,
        1 => 0x201f_e012,
        2 => 0x70ff_e07a,
        _ => 0,
    }
}

const fn bit(idx: usize) -> u32 {
    1 << idx
}

/// The EL1&0 regime MMU indexes, `alle1_tlbmask()`.
const E10_MASK: u32 = bit(MMU_IDX_E10_0) | bit(MMU_IDX_E10_1) | bit(MMU_IDX_E10_1_PAN);
/// The EL2&0 regime MMU indexes.
const E20_MASK: u32 = bit(MMU_IDX_E20_0) | bit(MMU_IDX_E20_2) | bit(MMU_IDX_E20_2_PAN);
/// `alle2_tlbmask()`.
const ALLE2_MASK: u32 = E20_MASK | bit(MMU_IDX_E2);

/// `tlbbits_for_regime()`: 56 when TBI applies to `addr` in the regime of `mmu_idx`,
/// otherwise 64.
fn tlbbits(st: &CpuArmState, mmu_idx: usize, addr: u64) -> u32 {
    let tcr = st.tcr_el[super::regime_el(mmu_idx) as usize];
    let (tbi, _) = super::tbi_bits(tcr, mmu_idx);
    let select = if regime_has_2_ranges(mmu_idx) { (addr >> 55) & 1 } else { 0 };
    if (tbi >> select) & 1 != 0 { 56 } else { 64 }
}

/// The index into the per EL arrays of an EL1, EL2 or EL3 register encoding.
fn el_of(k: u32) -> usize {
    match key_op1(k) {
        4 => 2,
        6 => 3,
        _ => 1,
    }
}

/// The timer and whether the access is direct for a timer register at `(op1, crm)`.
fn timer_of(f: &ArmFeatures, st: &CpuArmState, op1: u32, crm: u32) -> (usize, bool) {
    let base = if crm == 2 { GTIMER_PHYS } else { GTIMER_VIRT };
    match op1 {
        3 => (gtimer::redirect(f, st, base), true),
        4 => (if crm == 2 { GTIMER_HYP } else { GTIMER_HYPVIRT }, true),
        5 => (base, false),
        _ => (GTIMER_SEC, true),
    }
}

/// Read a [`Kind::Special`] register.
pub(crate) fn read(cpu: &mut Cpu<'_>, key_: u32) -> u64 {
    let ops = cpu.ops();
    let arm: &Arm = arm_of(&ops);
    let model = arm.model();
    let f = arm.features();
    let st = CpuArmState::load(cpu.env);
    if let Some((op1, crm, op2)) = timer_reg(key_) {
        let (timer, direct) = timer_of(f, &st, op1, crm);
        return match op2 {
            0 => gtimer::tval_read(arm, &st, timer, direct),
            1 => st.gt_ctl[timer],
            _ => st.gt_cval[timer],
        };
    }
    let el1_with_el2 = st.current_el() == 1 && st.is_el2_enabled(f);
    match key_ {
        MIDR_EL1 => {
            if el1_with_el2 {
                st.vpidr_el2
            } else {
                model.midr
            }
        }
        MPIDR_EL1 => {
            if el1_with_el2 {
                st.vmpidr_el2
            } else {
                // mpidr_read_val(): the affinity with bit 31 RES1.
                (1 << 31) | arm.mp_affinity(cpu.core.shared().cpu_index)
            }
        }
        CCSIDR_EL1 => ccsidr(st.csselr_el1),
        CTR_EL0 => model.ctr,
        DCZID_EL0 => {
            // aa64_dczid_read(): DZP is set when DC ZVA is not allowed.
            let prohibited = zva_access(f, &st) != Access::Ok;
            model.dczid | (u64::from(prohibited) << 4)
        }
        SCTLR_EL1 | SCTLR_EL2 | SCTLR_EL3 => st.sctlr_el[el_of(key_)],
        TTBR0_EL1 | TTBR0_EL2 | TTBR0_EL3 => st.ttbr0_el[el_of(key_)],
        TTBR1_EL1 | TTBR1_EL2 => st.ttbr1_el[el_of(key_)],
        TCR_EL1 | TCR_EL2 | TCR_EL3 => st.tcr_el[el_of(key_)],
        HCR_EL2 => st.hcr_el2,
        SCR_EL3 => st.scr_el3,
        VTTBR_EL2 => st.vttbr_el2,
        VTCR_EL2 => st.vtcr_el2,
        CNTVOFF_EL2 => st.cntvoff_el2,
        CNTHCTL_EL2 => st.cnthctl_el2,
        NZCV => u64::from(st.nzcv()),
        DAIF => u64::from(st.daif & PSTATE_DAIF),
        FPCR => u64::from(vfp::get_fpcr(&st)),
        FPSR => u64::from(vfp::get_fpsr(&st)),
        SPSEL => u64::from(st.pstate & PSTATE_SP),
        PAN => u64::from(st.pstate & PSTATE_PAN),
        UAO => u64::from(st.pstate & PSTATE_UAO),
        OSLSR_EL1 => st.oslsr_el1,
        ZCR_EL1 | ZCR_EL2 | ZCR_EL3 => st.zcr_el[el_of(key_)],
        CNTPCT_EL0 => gtimer::phys_count(arm, &st),
        CNTVCT_EL0 => gtimer::virt_count(arm, &st),
        _ => panic!("no read for system register key 0x{key_:x}"),
    }
}

/// Write a [`Kind::Special`] register.
pub(crate) fn write(cpu: &mut Cpu<'_>, key_: u32, value: u64) {
    let ops = cpu.ops();
    let arm: &Arm = arm_of(&ops);
    let f = arm.features();
    let mut st = CpuArmState::load(cpu.env);
    if let Some((op1, crm, op2)) = timer_reg(key_) {
        let (timer, direct) = timer_of(f, &st, op1, crm);
        match op2 {
            0 => gtimer::tval_write(arm, cpu, &mut st, timer, direct, value),
            1 => gtimer::ctl_write(arm, cpu, &mut st, timer, value),
            _ => gtimer::cval_write(arm, cpu, &mut st, timer, value),
        }
        st.store(cpu.env);
        return;
    }
    match key_ {
        SCTLR_EL1 | SCTLR_EL2 | SCTLR_EL3 => {
            st.sctlr_el[el_of(key_)] = value;
            st.store(cpu.env);
            // This may enable or disable the MMU, so do a TLB flush.
            tlb_flush(cpu);
        }
        TCR_EL1 | TCR_EL2 | TCR_EL3 => {
            // vmsa_tcr_el12_write(): flush, then write.
            tlb_flush(cpu);
            st.tcr_el[el_of(key_)] = value;
            st.store(cpu.env);
        }
        TTBR0_EL1 | TTBR1_EL1 | TTBR0_EL2 | TTBR1_EL2 | TTBR0_EL3 => {
            let el = el_of(key_);
            let slot = if key_ == TTBR1_EL1 || key_ == TTBR1_EL2 {
                &mut st.ttbr1_el[el]
            } else {
                &mut st.ttbr0_el[el]
            };
            let old = *slot;
            *slot = value;
            let e2h = st.hcr_el2_eff(f) & HCR_E2H != 0;
            st.store(cpu.env);
            // If the ASID changes we must flush the TLB; only the EL1&0 and, with E2H, the
            // EL2&0 regimes have one.
            if (old ^ value) >> 48 != 0 {
                match el {
                    1 => tlb_flush_by_mmuidx(cpu, E10_MASK),
                    2 if e2h => tlb_flush_by_mmuidx(cpu, E20_MASK),
                    _ => {}
                }
            }
        }
        HCR_EL2 => {
            let old = st.hcr_el2;
            st.hcr_write(f, arm.psci_conduit() == PsciConduit::Smc, value);
            let changed = old ^ st.hcr_el2;
            st.store(cpu.env);
            // These bits change the stage 2 setup or the regime, so flush.
            if changed & (HCR_VM | HCR_PTW | HCR_DC | HCR_FWB | HCR_NV | HCR_NV1) != 0 {
                tlb_flush(cpu);
            }
            // Updates to VI and VF require us to update the status of virtual interrupts,
            // which are the logical OR of these bits and the state of the input lines from
            // the GIC.
            arm.update_virt_lines(cpu, &st);
        }
        SCR_EL3 => {
            let old = st.scr_el3;
            st.scr_write(f, value);
            st.store(cpu.env);
            // If the NS bit changes, the Security state of EL2 and below changes.
            if (old ^ st.scr_el3) & SCR_NS != 0 {
                tlb_flush(cpu);
            }
        }
        VTTBR_EL2 | VTCR_EL2 => {
            let slot = if key_ == VTTBR_EL2 { &mut st.vttbr_el2 } else { &mut st.vtcr_el2 };
            let old = *slot;
            *slot = value;
            st.store(cpu.env);
            if old != value {
                // The TLB has no VMID tags, so a new stage 2 setup flushes EL1&0.
                tlb_flush_by_mmuidx(cpu, E10_MASK);
            }
        }
        CNTVOFF_EL2 => {
            gtimer::cntvoff_write(arm, cpu, &mut st, value);
            st.store(cpu.env);
        }
        CNTHCTL_EL2 => {
            st.cnthctl_el2 = value & 0xfff;
            st.store(cpu.env);
        }
        NZCV => {
            st.set_nzcv(value as u32);
            st.store(cpu.env);
        }
        DAIF => {
            st.daif = value as u32 & PSTATE_DAIF;
            st.store(cpu.env);
        }
        FPCR => {
            vfp::set_fpcr(&mut st, value as u32, f);
            st.store(cpu.env);
        }
        FPSR => {
            vfp::set_fpsr(&mut st, value as u32);
            st.store(cpu.env);
        }
        SPSEL => {
            st.update_spsel(value as u32);
            st.store(cpu.env);
        }
        PAN => {
            st.pstate = (st.pstate & !PSTATE_PAN) | (value as u32 & PSTATE_PAN);
            st.store(cpu.env);
        }
        UAO => {
            st.pstate = (st.pstate & !PSTATE_UAO) | (value as u32 & PSTATE_UAO);
            st.store(cpu.env);
        }
        OSLAR_EL1 => {
            // oslar_write(): OSLSR_EL1.OSLK follows bit 0.
            st.oslsr_el1 = (st.oslsr_el1 & !2) | ((value & 1) << 1);
            st.store(cpu.env);
        }
        ZCR_EL1 | ZCR_EL2 | ZCR_EL3 => {
            // zcr_write(): bits other than [3:0] are RAZ/WI. Because we arrived here, we
            // know both FP and SVE are enabled; otherwise we would have trapped access to the
            // ZCR_ELn register.
            let cur_el = st.current_el();
            let old_len = super::sve_vqm1_for_el(f, &st, cur_el);
            st.zcr_el[el_of(key_)] = value & 0xf;
            let new_len = super::sve_vqm1_for_el(f, &st, cur_el);
            if new_len < old_len {
                super::sve_narrow_vq(&mut st, new_len as usize + 1);
            }
            st.store(cpu.env);
        }
        _ if key_ >> 14 == 1 && (key_ >> 7) & 0xf == 7 => at(arm, cpu, &st, key_, value),
        _ if key_ >> 14 == 1 && (key_ >> 7) & 0xf == 8 => tlbi(cpu, f, &st, key_, value),
        _ => panic!("no write for system register key 0x{key_:x}"),
    }
}

/// The AT instructions: translate `value` and write the result to PAR_EL1.
fn at(arm: &Arm, cpu: &mut Cpu<'_>, st: &CpuArmState, key_: u32, value: u64) {
    let f = arm.features();
    let access = if key_ & 1 != 0 { MmuAccessType::DataStore } else { MmuAccessType::DataLoad };
    let e2h = st.hcr_el2_eff(f) & HCR_E2H != 0;
    let el = st.current_el();
    let (mmu_idx, stage1_only) = match key_ {
        AT_S1E1R | AT_S1E1W => {
            // At EL2 with E2H the EL1 forms translate in the EL2&0 regime.
            let idx = if el == 2 && e2h {
                MMU_IDX_E20_2
            } else if el == 1 && st.pstate & PSTATE_PAN != 0 {
                MMU_IDX_E10_1_PAN
            } else {
                MMU_IDX_E10_1
            };
            (idx, true)
        }
        AT_S1E0R | AT_S1E0W => (if el == 2 && e2h { MMU_IDX_E20_0 } else { MMU_IDX_E10_0 }, true),
        AT_S12E1R | AT_S12E1W => (MMU_IDX_E10_1, false),
        AT_S12E0R | AT_S12E0W => (MMU_IDX_E10_0, false),
        AT_S1E2R | AT_S1E2W => (if e2h { MMU_IDX_E20_2 } else { MMU_IDX_E2 }, true),
        AT_S1E3R | AT_S1E3W => (MMU_IDX_E3, true),
        _ => unreachable!("AT key 0x{key_:x} is not in the table"),
    };
    let par = match ptw::get_phys_addr(arm, cpu, value, access, mmu_idx, stage1_only, true) {
        // The LPAE bit is always set; NS is set because there is no Secure address space.
        Ok(t) => {
            (1 << 11)
                | (t.pa & !0xfff & ((1 << 52) - 1))
                | (1 << 9)
                | (u64::from(t.attrs) << 56)
                | (u64::from(t.sh) << 7)
        }
        Err(fault) => {
            // PAR.S is set for a stage 2 fault and PAR.PTW for one on a stage 1 walk.
            (1 << 11)
                | 1
                | (u64::from(fault.fsc & 0x3f) << 1)
                | (u64::from(fault.s1ptw) << 8)
                | (u64::from(fault.stage2) << 9)
        }
    };
    let mut st = CpuArmState::load(cpu.env);
    st.par_el1 = par;
    st.store(cpu.env);
}

/// `vae1_tlbmask()`: the regime the EL1 TLBIs operate on, and an MMU index in it.
fn vae1_tlbmask(f: &ArmFeatures, st: &CpuArmState) -> (u32, usize) {
    let hcr = st.hcr_el2_eff(f);
    if hcr & (HCR_E2H | HCR_TGE) == HCR_E2H | HCR_TGE {
        (E20_MASK, MMU_IDX_E20_2)
    } else {
        (E10_MASK, MMU_IDX_E10_1)
    }
}

/// The TLBI operations. This TLB has no ASIDs or VMIDs, so the by-ASID forms flush every
/// entry of the regime, and the by-VA forms flush the page for every ASID, as QEMU does.
fn tlbi(cpu: &mut Cpu<'_>, f: &ArmFeatures, st: &CpuArmState, key_: u32, value: u64) {
    let op1 = key_op1(key_);
    let crm = (key_ >> 3) & 0xf;
    let op2 = key_ & 7;
    // CRm 3 and 0 are the Inner Shareable forms, which are broadcast.
    let shareable = crm == 3 || crm == 0;
    let (mask, va_idx) = match (op1, op2) {
        // VMALLE1, ASIDE1, VAE1 and friends.
        (0, _) => vae1_tlbmask(f, st),
        // ALLE2.
        (4, 0) => (ALLE2_MASK, MMU_IDX_E2),
        // IPAS2E1 and IPAS2LE1: there is no separate stage 2 TLB.
        (4, 1 | 5) if crm == 0 || crm == 4 => return,
        // VAE2 and VALE2, `vae2_tlbmask()`.
        (4, 1 | 5) => {
            if st.hcr_el2_eff(f) & HCR_E2H != 0 {
                (E20_MASK, MMU_IDX_E20_2)
            } else {
                (bit(MMU_IDX_E2), MMU_IDX_E2)
            }
        }
        // ALLE1, VMALLS12E1.
        (4, _) => (E10_MASK, MMU_IDX_E10_1),
        // ALLE3, VAE3, VALE3.
        _ => (bit(MMU_IDX_E3), MMU_IDX_E3),
    };
    let by_va = matches!((op1, op2), (0, 1 | 3 | 5 | 7) | (4 | 6, 1 | 5));
    if by_va {
        let pageaddr = (((value << 12) << 8) as i64 >> 8) as u64;
        let bits = tlbbits(st, va_idx, pageaddr);
        if shareable {
            tlb_flush_page_bits_by_mmuidx_all_cpus_synced(cpu, pageaddr, mask, bits);
        } else {
            tlb_flush_page_bits_by_mmuidx(cpu, pageaddr, mask, bits);
        }
    } else if shareable {
        tlb_flush_by_mmuidx_all_cpus_synced(cpu, mask);
    } else {
        tlb_flush_by_mmuidx(cpu, mask);
    }
}

#[cfg(test)]
mod tests {
    use super::{EL2_REGS, EL3_REGS, REGS};

    #[test]
    fn keys_and_names_are_unique() {
        let all: Vec<_> = REGS.iter().chain(EL2_REGS).chain(EL3_REGS).collect();
        for (i, a) in all.iter().enumerate() {
            assert!(!a.name.is_empty());
            for b in &all[i + 1..] {
                assert!(a.key != b.key, "{} and {} share an encoding", a.name, b.name);
                assert!(a.name != b.name, "{} is listed twice", a.name);
            }
        }
    }
}
