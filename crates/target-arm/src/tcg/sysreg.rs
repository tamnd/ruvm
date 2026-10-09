// SPDX-License-Identifier: GPL-2.0-or-later

//! The AArch64 system registers of EL0 to EL3: the port of the parts of QEMU's
//! `target/arm/helper.c` (`v8_cp_reginfo`, `vmsa_cp_reginfo`, `el2_cp_reginfo`,
//! `el3_cp_reginfo`, `generic_timer_cp_reginfo`, the VHE redirections of
//! `define_arm_vh_e2h_redirect_aliases()`, the ID registers and the TLBI, AT and cache
//! maintenance operations) and `debug_helper.c` that this port needs, with the
//! IMPLEMENTATION DEFINED registers of `cortex-regs.c` and `define_neoverse_n1_cp_reginfo()`.
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
//! - The debug registers are the few the EL1 slice had; of MDCR_EL2 and MDCR_EL3 only the
//!   PMU controls take effect (see `pmu.rs`), their debug traps are not checked, nor are the
//!   HSTR_EL2 and HCR_EL2.TIDCP traps.
//! - TLBI IPAS2E1 and IPAS2LE1 have no effect: this port caches no stage 2 walk results
//!   apart from the combined stage 1 and 2 entries, which the architecture requires the
//!   hypervisor to invalidate with a stage 1 TLBI (VMALLE1 or VMALLS12E1) afterwards.
//! - The TLB has no VMID or ASID tags, so a change of VTTBR_EL2 flushes the EL1&0 regime
//!   and the by-ASID TLBIs flush the whole regime, as QEMU does.
//! - A change of SCR_EL3.NS flushes the whole TLB; there is no Secure address space.
//! - TCR2_EL1 and TCR2_EL2 keep only the FEAT_ASID2 bits (A2, FNG0 and FNG1), which change
//!   nothing in translation as the TLB has no ASIDs; there is no HCRX_EL2, so EL1 accesses
//!   never trap on HCRX_EL2.TCR2En.
//! - The TLBI nXS forms (CRn 9) do what their plain forms do, and the TLBI OS forms are
//!   broadcast like the IS ones. There are no FEAT_TLBIRANGE operations.

use std::mem::offset_of;

use ruvm_jit::cputlb::{
    tlb_flush, tlb_flush_by_mmuidx, tlb_flush_by_mmuidx_all_cpus_synced,
    tlb_flush_page_bits_by_mmuidx, tlb_flush_page_bits_by_mmuidx_all_cpus_synced,
};
use ruvm_jit::{Cpu, MmuAccessType, interrupt};

use super::pmu::{self, Pmu, PmuTrap};
use super::{Arm, PsciConduit, arm_of, gic, gtimer, ptw, regime_has_2_ranges, vfp};
use crate::cpu::{
    ArmCpuModel, ArmFeatures, CpuArmState, GTIMER_HYP, GTIMER_HYPVIRT, GTIMER_PHYS, GTIMER_SEC,
    GTIMER_VIRT, HCR_AMO, HCR_APK, HCR_ATA, HCR_DC, HCR_E2H, HCR_FMO, HCR_FWB, HCR_IMO, HCR_NV,
    HCR_NV1, HCR_PTW, HCR_TACR, HCR_TDZ, HCR_TGE, HCR_TID1, HCR_TID2, HCR_TID3, HCR_TID5, HCR_TPCP,
    HCR_TPU, HCR_TRVM, HCR_TSW, HCR_TTLB, HCR_TVM, HCR_VM, ImpdefRegs, MMU_IDX_E2, MMU_IDX_E3,
    MMU_IDX_E10_0, MMU_IDX_E10_1, MMU_IDX_E10_1_PAN, MMU_IDX_E20_0, MMU_IDX_E20_2,
    MMU_IDX_E20_2_PAN, PSTATE_A, PSTATE_DAIF, PSTATE_F, PSTATE_I, PSTATE_PAN, PSTATE_SP,
    PSTATE_SSBS, PSTATE_TCO, PSTATE_UAO, SCR_APK, SCR_ATA, SCR_NS, SCR_NSE, SCR_TCR2EN, SCTLR_ATA,
    SCTLR_ATA0, SCTLR_DZE, SCTLR_ITFSB, SCTLR_TCF, SCTLR_TCF0, SCTLR_TCSO, SCTLR_TCSO0, SCTLR_UCI,
    SCTLR_UCT, SCTLR_UMA, env_off,
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
    /// `access_ttlb()`, also `access_ttlbis()` and `access_ttlbos()`: HCR_EL2.TTLBIS and
    /// TTLBOS are RES0 here.
    Ttlb,
    /// `tcr2_el1_access()` and `tcr2_el2_access()`: the TVM and TRVM checks at EL1 and
    /// SCR_EL3.TCR2En below EL3. There is no HCRX_EL2, so HCRX_EL2.TCR2En reads as one,
    /// as it does when EL2 is disabled.
    Tcr2,
    /// `cpacr_access()`: CPTR_EL2.TCPAC and CPTR_EL3.TCPAC.
    Cpacr,
    /// `cptr_access()`: CPTR_EL3.TCPAC for CPTR_EL2.
    Cptr,
    /// `access_pauth()`: HCR_EL2.APK and SCR_EL3.APK.
    Apk,
    /// `access_mte()`: HCR_EL2.ATA and SCR_EL3.ATA. Without FEAT_NV this is also
    /// `access_tfsr_el1()` and `access_tfsr_el2()`.
    Mte,
    /// `access_tid5()`: HCR_EL2.TID5.
    Tid5,
    /// `access_actlr_w()`: ACTLR_EL2 and ACTLR_EL3 are constant 0, so writes below EL2 trap
    /// to EL2 when it is enabled and writes below EL3 trap to EL3.
    Actlr,
    /// The `accessfn` of a GICv3 CPU interface register (`gicv3_irqfiq_access()` and
    /// friends), which the interface decides: see `gic.rs`.
    Gic,
    /// The checks of the PMU registers, see `pmu.rs`.
    Pmu(PmuTrap),
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
            Trap::Tid5 => {
                if el < 2 && hcr & HCR_TID5 != 0 {
                    Access::TrapEl2
                } else {
                    Access::Ok
                }
            }
            Trap::Mte => {
                if el < 2
                    && st.is_el2_enabled(f)
                    && hcr & HCR_ATA == 0
                    && hcr & (HCR_E2H | HCR_TGE) != HCR_E2H | HCR_TGE
                {
                    Access::TrapEl2
                } else if el < 3 && f.el3 && st.scr_el3 & SCR_ATA == 0 {
                    Access::TrapEl3
                } else {
                    Access::Ok
                }
            }
            Trap::Tacr => el1_hcr(HCR_TACR),
            Trap::Actlr => {
                if isread {
                    Access::Ok
                } else if el < 2 && st.is_el2_enabled(f) {
                    Access::TrapEl2
                } else if el < 3 && f.el3 {
                    Access::TrapEl3
                } else {
                    Access::Ok
                }
            }
            Trap::Ttlb => el1_hcr(HCR_TTLB),
            Trap::Tcr2 => {
                let tvm = el1_hcr(if isread { HCR_TRVM } else { HCR_TVM });
                if tvm != Access::Ok {
                    tvm
                } else if el < 3 && f.el3 && st.scr_el3 & SCR_TCR2EN == 0 {
                    Access::TrapEl3
                } else {
                    Access::Ok
                }
            }
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
            Trap::Apk => {
                if el < 2 && st.is_el2_enabled(f) && hcr & HCR_APK == 0 {
                    Access::TrapEl2
                } else if el < 3 && f.el3 && st.scr_el3 & SCR_APK == 0 {
                    Access::TrapEl3
                } else {
                    Access::Ok
                }
            }
            // The interface is asked by the access check helper, which knows the vCPU.
            Trap::Gic => Access::Ok,
            Trap::Pmu(t) => t.check(f, st, isread),
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
    /// Reads as the value (`ARM_CP_CONST`).
    Const(u64),
    /// An operation with no effect here (`ARM_CP_NOP`).
    Nop,
    /// Read and written by [`read`] and [`write`] through helpers.
    Special,
    /// DC ZVA (`ARM_CP_DC_ZVA`).
    DcZva,
    /// DC GVA (`ARM_CP_DC_GVA`).
    DcGva,
    /// DC GZVA (`ARM_CP_DC_GZVA`).
    DcGzva,
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

fn has_pmu(f: &ArmFeatures) -> bool {
    f.pmu != 0
}

fn has_pmuv3p4(f: &ArmFeatures) -> bool {
    f.pmu != 0 && pmu::pmuv3p4(f)
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

fn has_tlbios(f: &ArmFeatures) -> bool {
    f.tlbios
}

fn has_pauth(f: &ArmFeatures) -> bool {
    f.pauth != 0
}

fn has_tcr2(f: &ArmFeatures) -> bool {
    f.tcr2
}

fn has_rme(f: &ArmFeatures) -> bool {
    f.rme
}

fn has_ssbs(f: &ArmFeatures) -> bool {
    f.ssbs
}

fn has_rng(f: &ArmFeatures) -> bool {
    f.rng
}

/// `aa64_mte_insn_reg`: the MTE instructions and the EL0 cache operations.
fn has_mte_insn_reg(f: &ArmFeatures) -> bool {
    f.mte >= 1
}

/// `aa64_mte`: the tag storage and the tag check registers.
fn has_mte(f: &ArmFeatures) -> bool {
    f.mte >= 2
}

fn has_rme_mte(f: &ArmFeatures) -> bool {
    f.rme && f.mte >= 2
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

/// A PMU register: its check is one of [`PmuTrap`] and it exists when the CPU has a PMU.
macro_rules! pm {
    ($name:literal, ($op0:expr, $op1:expr, $crn:expr, $crm:expr, $op2:expr), $access:expr,
     $trap:ident) => {
        pm!($name, ($op0, $op1, $crn, $crm, $op2), $access, $trap, Kind::Special, has_pmu)
    };
    ($name:literal, ($op0:expr, $op1:expr, $crn:expr, $crm:expr, $op2:expr), $access:expr,
     $trap:ident, $kind:expr, $feat:expr) => {
        Reg {
            name: $name,
            key: key($op0, $op1, $crn, $crm, $op2),
            access: $access,
            trap: Trap::Pmu(PmuTrap::$trap),
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
/// ISR_EL1.
const ISR_EL1: u32 = key(3, 0, 12, 1, 0);
/// SCTLR_EL1.
pub(crate) const SCTLR_EL1: u32 = key(3, 0, 1, 0, 0);
/// TTBR0_EL1.
pub(crate) const TTBR0_EL1: u32 = key(3, 0, 2, 0, 0);
/// TTBR1_EL1.
pub(crate) const TTBR1_EL1: u32 = key(3, 0, 2, 0, 1);
/// TCR_EL1.
pub(crate) const TCR_EL1: u32 = key(3, 0, 2, 0, 2);
/// TCR2_EL1.
pub(crate) const TCR2_EL1: u32 = key(3, 0, 2, 0, 3);
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
/// TCO.
pub(crate) const TCO: u32 = key(3, 3, 4, 2, 7);
/// SSBS.
pub(crate) const SSBS: u32 = key(3, 3, 4, 2, 6);
/// L2CTLR_EL1 of the Cortex-A57 and A72.
pub(crate) const L2CTLR_EL1: u32 = key(3, 1, 11, 0, 2);
pub(crate) const RNDR: u32 = key(3, 3, 2, 4, 0);
pub(crate) const RNDRRS: u32 = key(3, 3, 2, 4, 1);
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
/// TCR2_EL2.
pub(crate) const TCR2_EL2: u32 = key(3, 4, 2, 0, 3);
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
/// GPCBW_EL3.
pub(crate) const GPCBW_EL3: u32 = key(3, 6, 2, 1, 5);
/// GPCCR_EL3.
pub(crate) const GPCCR_EL3: u32 = key(3, 6, 2, 1, 6);
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
    ($name:literal, $op1:expr, $crm:expr, $op2:expr, $access:expr, $trap:ident, $feat:expr) => {
        r!($name, (1, $op1, 8, $crm, $op2), $access, $trap, Kind::Special, $feat)
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
    r!("ID_AA64ISAR2_EL1", (3, 0, 0, 6, 2), PL1_R, Tid3, Kind::Model(|m| m.id_aa64isar2)),
    r!("ID_AA64MMFR0_EL1", (3, 0, 0, 7, 0), PL1_R, Tid3, Kind::Model(|m| m.id_aa64mmfr0)),
    r!("ID_AA64MMFR1_EL1", (3, 0, 0, 7, 1), PL1_R, Tid3, Kind::Model(|m| m.id_aa64mmfr1)),
    r!("ID_AA64MMFR2_EL1", (3, 0, 0, 7, 2), PL1_R, Tid3, Kind::Model(|m| m.id_aa64mmfr2)),
    r!("ID_AA64MMFR3_EL1", (3, 0, 0, 7, 3), PL1_R, Tid3, Kind::Model(|m| m.id_aa64mmfr3)),
    r!("ID_AA64MMFR4_EL1", (3, 0, 0, 7, 4), PL1_R, Tid3, Kind::Model(|m| m.id_aa64mmfr4)),
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
    r!("TCR2_EL1", (3, 0, 2, 0, 3), PL1_RW, Tcr2, Kind::Special, has_tcr2),
    // pauth_reginfo.
    r!("APIAKEYLO_EL1", (3, 0, 2, 1, 0), PL1_RW, Apk, field(off!(pac_keys[0])), has_pauth),
    r!("APIAKEYHI_EL1", (3, 0, 2, 1, 1), PL1_RW, Apk, field(off!(pac_keys[1])), has_pauth),
    r!("APIBKEYLO_EL1", (3, 0, 2, 1, 2), PL1_RW, Apk, field(off!(pac_keys[2])), has_pauth),
    r!("APIBKEYHI_EL1", (3, 0, 2, 1, 3), PL1_RW, Apk, field(off!(pac_keys[3])), has_pauth),
    r!("APDAKEYLO_EL1", (3, 0, 2, 2, 0), PL1_RW, Apk, field(off!(pac_keys[4])), has_pauth),
    r!("APDAKEYHI_EL1", (3, 0, 2, 2, 1), PL1_RW, Apk, field(off!(pac_keys[5])), has_pauth),
    r!("APDBKEYLO_EL1", (3, 0, 2, 2, 2), PL1_RW, Apk, field(off!(pac_keys[6])), has_pauth),
    r!("APDBKEYHI_EL1", (3, 0, 2, 2, 3), PL1_RW, Apk, field(off!(pac_keys[7])), has_pauth),
    r!("APGAKEYLO_EL1", (3, 0, 2, 3, 0), PL1_RW, Apk, field(off!(pac_keys[8])), has_pauth),
    r!("APGAKEYHI_EL1", (3, 0, 2, 3, 1), PL1_RW, Apk, field(off!(pac_keys[9])), has_pauth),
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
    r!("ISR_EL1", (3, 0, 12, 1, 0), PL1_R, None, Kind::Special),
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
    // Performance monitors, cpregs-pmu.c; PMEVCNTR<n>_EL0 and PMEVTYPER<n>_EL0 are found by
    // lookup().
    pm!("PMCR_EL0", (3, 3, 9, 12, 0), PL0_RW, Pmcr),
    pm!("PMCNTENSET_EL0", (3, 3, 9, 12, 1), PL0_RW, Reg),
    pm!("PMCNTENCLR_EL0", (3, 3, 9, 12, 2), PL0_RW, Reg),
    pm!("PMOVSCLR_EL0", (3, 3, 9, 12, 3), PL0_RW, Reg),
    pm!("PMSWINC_EL0", (3, 3, 9, 12, 4), PL0_W, Swinc),
    pm!("PMSELR_EL0", (3, 3, 9, 12, 5), PL0_RW, Selr),
    pm!(
        "PMCEID0_EL0",
        (3, 3, 9, 12, 6),
        PL0_R,
        Reg,
        Kind::Model(|m| pmu::pmceid(&m.features).0),
        has_pmu
    ),
    pm!(
        "PMCEID1_EL0",
        (3, 3, 9, 12, 7),
        PL0_R,
        Reg,
        Kind::Model(|m| pmu::pmceid(&m.features).1),
        has_pmu
    ),
    pm!("PMCCNTR_EL0", (3, 3, 9, 13, 0), PL0_RW, Ccntr),
    pm!("PMXEVTYPER_EL0", (3, 3, 9, 13, 1), PL0_RW, Reg),
    pm!("PMXEVCNTR_EL0", (3, 3, 9, 13, 2), PL0_RW, Xevcntr),
    pm!("PMUSERENR_EL0", (3, 3, 9, 14, 0), PL0_R | PL1_RW, Tpm),
    pm!("PMINTENSET_EL1", (3, 0, 9, 14, 1), PL1_RW, Tpm),
    pm!("PMINTENCLR_EL1", (3, 0, 9, 14, 2), PL1_RW, Tpm),
    pm!("PMOVSSET_EL0", (3, 3, 9, 14, 3), PL0_RW, Reg),
    pm!("PMMIR_EL1", (3, 0, 9, 14, 6), PL1_R, Reg, Kind::Zero, has_pmuv3p4),
    pm!("PMCCFILTR_EL0", (3, 3, 14, 15, 7), PL0_RW, Reg),
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
    // GMID_EL1 and mte_reginfo.
    r!(
        "GMID_EL1",
        (3, 1, 0, 0, 4),
        PL1_R,
        Tid5,
        Kind::Model(|_| super::mte::GM_BLOCKSIZE as u64),
        has_mte
    ),
    r!("TFSRE0_EL1", (3, 0, 5, 6, 1), PL1_RW, Mte, field(off!(tfsr_el[0])), has_mte),
    r!("TFSR_EL1", (3, 0, 5, 6, 0), PL1_RW, Mte, field(off!(tfsr_el[1])), has_mte),
    r!("TFSR_EL2", (3, 4, 5, 6, 0), PL2_RW, Mte, field(off!(tfsr_el[2])), has_mte),
    r!("TFSR_EL3", (3, 6, 5, 6, 0), PL3_RW, None, field(off!(tfsr_el[3])), has_mte),
    r!("RGSR_EL1", (3, 0, 1, 0, 5), PL1_RW, Mte, field(off!(rgsr_el1)), has_mte),
    r!("GCR_EL1", (3, 0, 1, 0, 6), PL1_RW, Mte, field(off!(gcr_el1)), has_mte),
    // mte_reginfo's TCO, and mte_tco_ro_reginfo's RAZ/WI one without FEAT_MTE2 (see read()
    // and write()).
    r!("TCO", (3, 3, 4, 2, 7), PL0_RW, None, Kind::Special, has_mte_insn_reg),
    // ssbs_reginfo.
    r!("SSBS", (3, 3, 4, 2, 6), PL0_RW, None, Kind::Special, has_ssbs),
    // rndr_reginfo, without FEAT_RNG_TRAP.
    r!("RNDR", (3, 3, 2, 4, 0), PL0_R, None, Kind::Special, has_rng),
    r!("RNDRRS", (3, 3, 2, 4, 1), PL0_R, None, Kind::Special, has_rng),
    r!("DC_IGVAC", (1, 0, 7, 6, 3), PL1_W, Poc, Kind::Nop, has_mte),
    r!("DC_IGSW", (1, 0, 7, 6, 4), PL1_W, Tsw, Kind::Nop, has_mte),
    r!("DC_IGDVAC", (1, 0, 7, 6, 5), PL1_W, Poc, Kind::Nop, has_mte),
    r!("DC_IGDSW", (1, 0, 7, 6, 6), PL1_W, Tsw, Kind::Nop, has_mte),
    r!("DC_CGSW", (1, 0, 7, 10, 4), PL1_W, Tsw, Kind::Nop, has_mte),
    r!("DC_CGDSW", (1, 0, 7, 10, 6), PL1_W, Tsw, Kind::Nop, has_mte),
    r!("DC_CIGSW", (1, 0, 7, 14, 4), PL1_W, Tsw, Kind::Nop, has_mte),
    r!("DC_CIGDSW", (1, 0, 7, 14, 6), PL1_W, Tsw, Kind::Nop, has_mte),
    // mte_el0_cacheop_reginfo.
    r!("DC_CGVAC", (1, 3, 7, 10, 3), PL0_W, Poc, Kind::Nop, has_mte_insn_reg),
    r!("DC_CGDVAC", (1, 3, 7, 10, 5), PL0_W, Poc, Kind::Nop, has_mte_insn_reg),
    r!("DC_CGVAP", (1, 3, 7, 12, 3), PL0_W, Poc, Kind::Nop, has_mte_insn_reg),
    r!("DC_CGDVAP", (1, 3, 7, 12, 5), PL0_W, Poc, Kind::Nop, has_mte_insn_reg),
    r!("DC_CGVADP", (1, 3, 7, 13, 3), PL0_W, Poc, Kind::Nop, has_mte_insn_reg),
    r!("DC_CGDVADP", (1, 3, 7, 13, 5), PL0_W, Poc, Kind::Nop, has_mte_insn_reg),
    r!("DC_CIGVAC", (1, 3, 7, 14, 3), PL0_W, Poc, Kind::Nop, has_mte_insn_reg),
    r!("DC_CIGDVAC", (1, 3, 7, 14, 5), PL0_W, Poc, Kind::Nop, has_mte_insn_reg),
    r!("DC_GVA", (1, 3, 7, 4, 3), PL0_W, Dze, Kind::DcGva, has_mte_insn_reg),
    r!("DC_GZVA", (1, 3, 7, 4, 4), PL0_W, Dze, Kind::DcGzva, has_mte_insn_reg),
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
    // tlbios_reginfo.
    tlbi!("TLBI_VMALLE1OS", 0, 1, 0, PL1_W, Ttlb, has_tlbios),
    tlbi!("TLBI_VAE1OS", 0, 1, 1, PL1_W, Ttlb, has_tlbios),
    tlbi!("TLBI_ASIDE1OS", 0, 1, 2, PL1_W, Ttlb, has_tlbios),
    tlbi!("TLBI_VAAE1OS", 0, 1, 3, PL1_W, Ttlb, has_tlbios),
    tlbi!("TLBI_VALE1OS", 0, 1, 5, PL1_W, Ttlb, has_tlbios),
    tlbi!("TLBI_VAALE1OS", 0, 1, 7, PL1_W, Ttlb, has_tlbios),
];

/// The EL2 registers, `el2_cp_reginfo` and the VHE additions.
static EL2_REGS: &[Reg] = &[
    r!("VPIDR_EL2", (3, 4, 0, 0, 0), PL2_RW, None, field(off!(vpidr_el2))),
    r!("VMPIDR_EL2", (3, 4, 0, 0, 5), PL2_RW, None, field(off!(vmpidr_el2))),
    r!("SCTLR_EL2", (3, 4, 1, 0, 0), PL2_RW, None, Kind::Special),
    r!("ACTLR_EL2", (3, 4, 1, 0, 1), PL2_RW, None, Kind::Zero),
    r!("HCR_EL2", (3, 4, 1, 1, 0), PL2_RW, None, Kind::Special),
    r!("MDCR_EL2", (3, 4, 1, 1, 1), PL2_RW, None, Kind::Special),
    r!("CPTR_EL2", (3, 4, 1, 1, 2), PL2_RW, Cptr, field(off!(cptr_el[2]))),
    r!("HSTR_EL2", (3, 4, 1, 1, 3), PL2_RW, None, field32(off!(hstr_el2))),
    r!("HACR_EL2", (3, 4, 1, 1, 7), PL2_RW, None, Kind::Zero),
    r!("TTBR0_EL2", (3, 4, 2, 0, 0), PL2_RW, None, Kind::Special),
    r!("TTBR1_EL2", (3, 4, 2, 0, 1), PL2_RW, None, Kind::Special, has_vh),
    r!("TCR_EL2", (3, 4, 2, 0, 2), PL2_RW, None, Kind::Special),
    r!("TCR2_EL2", (3, 4, 2, 0, 3), PL2_RW, Tcr2, Kind::Special, has_tcr2),
    r!("VTTBR_EL2", (3, 4, 2, 1, 0), PL2_RW, None, Kind::Special),
    r!("VTCR_EL2", (3, 4, 2, 1, 2), PL2_RW, None, Kind::Special),
    r!("SPSR_EL2", (3, 4, 4, 0, 0), PL2_RW, None, field(off!(spsr_el[2]))),
    r!("ELR_EL2", (3, 4, 4, 0, 1), PL2_RW, None, field(off!(elr_el[2]))),
    r!("SP_EL1", (3, 4, 4, 1, 0), PL2_RW, None, field(off!(sp_el[1]))),
    r!("SPSR_IRQ", (3, 4, 4, 3, 0), PL2_RW, None, field32(off!(spsr_aarch32[2]))),
    r!("SPSR_ABT", (3, 4, 4, 3, 1), PL2_RW, None, field32(off!(spsr_aarch32[0]))),
    r!("SPSR_UND", (3, 4, 4, 3, 2), PL2_RW, None, field32(off!(spsr_aarch32[1]))),
    r!("SPSR_FIQ", (3, 4, 4, 3, 3), PL2_RW, None, field32(off!(spsr_aarch32[3]))),
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
    tlbi!("TLBI_ALLE2OS", 4, 1, 0, PL2_W, None, has_tlbios),
    tlbi!("TLBI_VAE2OS", 4, 1, 1, PL2_W, None, has_tlbios),
    tlbi!("TLBI_ALLE1OS", 4, 1, 4, PL2_W, None, has_tlbios),
    tlbi!("TLBI_VALE2OS", 4, 1, 5, PL2_W, None, has_tlbios),
    tlbi!("TLBI_VMALLS12E1OS", 4, 1, 6, PL2_W, None, has_tlbios),
    // There is no separate stage 2 TLB, so these are ARM_CP_NOP as in QEMU.
    r!("TLBI_IPAS2E1OS", (1, 4, 8, 4, 0), PL2_W, None, Kind::Nop, has_tlbios),
    r!("TLBI_RIPAS2E1OS", (1, 4, 8, 4, 3), PL2_W, None, Kind::Nop, has_tlbios),
    r!("TLBI_IPAS2LE1OS", (1, 4, 8, 4, 4), PL2_W, None, Kind::Nop, has_tlbios),
    r!("TLBI_RIPAS2LE1OS", (1, 4, 8, 4, 7), PL2_W, None, Kind::Nop, has_tlbios),
];

/// The EL3 registers, `el3_cp_reginfo` and the Secure timer.
static EL3_REGS: &[Reg] = &[
    r!("SCTLR_EL3", (3, 6, 1, 0, 0), PL3_RW, None, Kind::Special),
    r!("ACTLR_EL3", (3, 6, 1, 0, 1), PL3_RW, None, Kind::Zero),
    r!("SCR_EL3", (3, 6, 1, 1, 0), PL3_RW, None, Kind::Special),
    r!("CPTR_EL3", (3, 6, 1, 1, 2), PL3_RW, None, field(off!(cptr_el[3]))),
    r!("MDCR_EL3", (3, 6, 1, 3, 1), PL3_RW, None, Kind::Special),
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
    // rme_reginfo and rme_mte_reginfo.
    r!("GPCCR_EL3", (3, 6, 2, 1, 6), PL3_RW, None, Kind::Special, has_rme),
    r!("GPCBW_EL3", (3, 6, 2, 1, 5), PL3_RW, None, Kind::Special, has_rme),
    r!("GPTBR_EL3", (3, 6, 2, 1, 4), PL3_RW, None, field(off!(gptbr_el3)), has_rme),
    r!("MFAR_EL3", (3, 6, 6, 0, 5), PL3_RW, None, field(off!(mfar_el3)), has_rme),
    r!("DC_CIPAPA", (1, 6, 7, 14, 1), PL3_W, None, Kind::Nop, has_rme),
    r!("DC_CIGDPAPA", (1, 6, 7, 14, 5), PL3_W, None, Kind::Nop, has_rme_mte),
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
    tlbi!("TLBI_ALLE3OS", 6, 1, 0, PL3_W, None, has_tlbios),
    tlbi!("TLBI_VAE3OS", 6, 1, 1, PL3_W, None, has_tlbios),
    tlbi!("TLBI_VALE3OS", 6, 1, 5, PL3_W, None, has_tlbios),
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
    (key(3, 0, 2, 0, 3), key(3, 4, 2, 0, 3)),   // TCR2
    (key(3, 0, 4, 0, 0), key(3, 4, 4, 0, 0)),   // SPSR
    (key(3, 0, 4, 0, 1), key(3, 4, 4, 0, 1)),   // ELR
    (key(3, 0, 5, 1, 0), key(3, 4, 5, 1, 0)),   // AFSR0
    (key(3, 0, 5, 1, 1), key(3, 4, 5, 1, 1)),   // AFSR1
    (key(3, 0, 5, 2, 0), key(3, 4, 5, 2, 0)),   // ESR
    (key(3, 0, 5, 6, 0), key(3, 4, 5, 6, 0)),   // TFSR
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

/// The registers a user mode CPU defines differently, `CONFIG_USER_ONLY`: the ID registers
/// read at EL0 as Linux emulates them (`modify_arm_cp_regs()` with the fields the kernel
/// exports), and of the generic timer only CNTFRQ_EL0 and CNTVCT_EL0. `None` means the usual
/// [`lookup`]; `Some(None)` means the register does not exist.
pub(crate) fn lookup_user(key_: u32, model: &ArmCpuModel) -> Option<Option<Reg>> {
    const CNTFRQ_EL0: u32 = key(3, 3, 14, 0, 0);
    let (op0, op1, crn, crm, op2) =
        (key_ >> 14, (key_ >> 11) & 7, (key_ >> 7) & 0xf, (key_ >> 3) & 0xf, key_ & 7);
    let reg = |name, value| {
        Some(Some(Reg {
            name,
            key: key_,
            access: PL0_R,
            trap: Trap::None,
            kind: Kind::Const(value),
            feat: always,
        }))
    };
    match (op0, op1, crn) {
        (3, 0, 0) => match (crm, op2) {
            (0, 0) => reg("MIDR_EL1", model.midr),
            (0, 5) => reg("MPIDR_EL1", 0x8000_0000),
            (0, 6) => reg("REVIDR_EL1", 0),
            (4..=7, _) => reg("ID_AA64*", super::user::user_id_reg(model, crm, op2)),
            _ => None,
        },
        (3, 3, 14) => match key_ {
            CNTFRQ_EL0 => reg("CNTFRQ_EL0", model.cntfrq),
            CNTVCT_EL0 => Some(Some(Reg {
                name: "CNTVCT_EL0",
                key: key_,
                access: PL0_R,
                trap: Trap::None,
                kind: Kind::Special,
                feat: always,
            })),
            _ => Some(None),
        },
        _ => None,
    }
}

/// The register with encoding `key` on a CPU with `feat`, as `get_arm_cp_reginfo()` finds
/// it. The ID register space that is not listed reads as zero at EL1.
pub(crate) fn lookup(key_: u32, feat: &ArmFeatures) -> Option<Reg> {
    if key_ >> 14 == 1 && (key_ >> 7) & 0xf == 9 {
        // ARM_CP_ADD_TLBI_NXS: with FEAT_XS every TLBI operation has an nXS form, encoded
        // with CRn 9 instead of 8, that does the same here.
        if !feat.xs {
            return None;
        }
        let r = lookup((key_ & !(0xf << 7)) | (8 << 7), feat)?;
        return Some(Reg { key: key_, ..r });
    }
    if let Some(r) = REGS.iter().find(|r| r.key == key_ && (r.feat)(feat)) {
        return Some(*r);
    }
    if let Some((typer, n)) = pmu::evreg(key_) {
        // The PMEVCNTR<n>_EL0 and PMEVTYPER<n>_EL0 that define_pm_cpregs() defines, one
        // pair per counter.
        if feat.pmu != 0 && n < pmu::num_counters(feat) {
            let (name, trap) = if typer {
                ("PMEVTYPER<n>_EL0", PmuTrap::Reg)
            } else {
                ("PMEVCNTR<n>_EL0", PmuTrap::Xevcntr)
            };
            return Some(Reg {
                name,
                key: key_,
                access: PL0_RW,
                trap: Trap::Pmu(trap),
                kind: Kind::Special,
                feat: has_pmu,
            });
        }
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
    if feat.gicv3 {
        if let Some(r) = icc_lookup(key_, feat) {
            return Some(r);
        }
    }
    let impdef = match feat.impdef {
        ImpdefRegs::None => &[][..],
        ImpdefRegs::CortexA57 => CORTEX_A57_REGS,
        ImpdefRegs::NeoverseN1 => NEOVERSE_N1_REGS,
    };
    if let Some(r) = impdef.iter().find(|r| r.key == key_) {
        return Some(*r);
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

/// `cortex_a72_a57_a53_cp_reginfo`, the AArch64 half.
const CORTEX_A57_REGS: &[Reg] = &[
    r!("L2CTLR_EL1", (3, 1, 11, 0, 2), PL1_RW, None, Kind::Special),
    r!("L2ECTLR_EL1", (3, 1, 11, 0, 3), PL1_RW, None, Kind::Zero),
    r!("L2ACTLR", (3, 1, 15, 0, 0), PL1_RW, None, Kind::Zero),
    r!("CPUACTLR_EL1", (3, 1, 15, 2, 0), PL1_RW, None, Kind::Zero),
    r!("CPUECTLR_EL1", (3, 1, 15, 2, 1), PL1_RW, None, Kind::Zero),
    r!("CPUMERRSR_EL1", (3, 1, 15, 2, 2), PL1_RW, None, Kind::Zero),
    r!("L2MERRSR_EL1", (3, 1, 15, 2, 3), PL1_RW, None, Kind::Zero),
];

/// `neoverse_n1_cp_reginfo`. ATCR_EL1 traps like TCR_EL1; there are no fine grained traps
/// here.
const NEOVERSE_N1_REGS: &[Reg] = &[
    r!("ATCR_EL1", (3, 0, 15, 7, 0), PL1_RW, Tvm, Kind::Zero),
    r!("ATCR_EL2", (3, 4, 15, 7, 0), PL2_RW, None, Kind::Zero),
    r!("ATCR_EL3", (3, 6, 15, 7, 0), PL3_RW, None, Kind::Zero),
    r!("ATCR_EL12", (3, 5, 15, 7, 0), PL2_RW, None, Kind::Zero),
    r!("AVTCR_EL2", (3, 4, 15, 7, 1), PL2_RW, None, Kind::Zero),
    r!("CPUACTLR_EL1", (3, 0, 15, 1, 0), PL1_RW, Actlr, Kind::Zero),
    r!("CPUACTLR2_EL1", (3, 0, 15, 1, 1), PL1_RW, Actlr, Kind::Zero),
    r!("CPUACTLR3_EL1", (3, 0, 15, 1, 2), PL1_RW, Actlr, Kind::Zero),
    // Report CPUCFR_EL1.SCU as 1, as we do not implement the DSU (and in particular its
    // system registers).
    r!("CPUCFR_EL1", (3, 0, 15, 0, 0), PL1_R, None, Kind::Model(|_| 4)),
    r!("CPUECTLR_EL1", (3, 0, 15, 1, 4), PL1_RW, Actlr, Kind::Model(|_| 0x9_6156_3010)),
    r!("CPUPCR_EL3", (3, 6, 15, 8, 1), PL3_RW, None, Kind::Zero),
    r!("CPUPMR_EL3", (3, 6, 15, 8, 3), PL3_RW, None, Kind::Zero),
    r!("CPUPOR_EL3", (3, 6, 15, 8, 2), PL3_RW, None, Kind::Zero),
    r!("CPUPSELR_EL3", (3, 6, 15, 8, 0), PL3_RW, None, Kind::Zero),
    r!("CPUPWRCTLR_EL1", (3, 0, 15, 2, 7), PL1_RW, Actlr, Kind::Zero),
    r!("ERXPFGCDN_EL1", (3, 0, 15, 2, 2), PL1_RW, Actlr, Kind::Zero),
    r!("ERXPFGCTL_EL1", (3, 0, 15, 2, 1), PL1_RW, Actlr, Kind::Zero),
    r!("ERXPFGF_EL1", (3, 0, 15, 2, 0), PL1_RW, Actlr, Kind::Zero),
];

/// The GICv3 CPU interface register with encoding `key_`, as `gicv3_init_cpuif()` defines
/// them for a CPU whose interface has `feat.gic_prebits` preemption bits. The ICH registers
/// exist only with EL2.
fn icc_lookup(key_: u32, feat: &ArmFeatures) -> Option<Reg> {
    // We don't support IRQ/FIQ bypass and system registers are always enabled, so all the
    // SRE bits are RAZ/WI or RAO/WI.
    let sre = |name: &'static str, access: u8, kind: Kind| Reg {
        name,
        key: key_,
        access,
        trap: Trap::None,
        kind,
        feat: always,
    };
    match gic::encoding(key_) {
        (3, 0, 12, 12, 5) => return Some(sre("ICC_SRE_EL1", PL1_RW, Kind::Model(|_| 0x7))),
        (3, 4, 12, 9, 5) => return Some(sre("ICC_SRE_EL2", PL2_RW, Kind::Model(|_| 0xf))),
        (3, 6, 12, 12, 5) => return Some(sre("ICC_SRE_EL3", PL3_RW, Kind::Model(|_| 0xf))),
        _ => {}
    }
    let enc = gic::encoding(key_);
    if feat.el2 {
        if let Some(r) = gic::ICH_REGS.iter().find(|r| r.enc == enc) {
            let (name, access) = (r.name, r.access);
            return Some(Reg {
                name,
                key: key_,
                access,
                trap: Trap::None,
                kind: Kind::Special,
                feat: always,
            });
        }
    }
    let r = gic::ICC_REGS.iter().find(|r| r.enc == enc && r.prebits <= feat.gic_prebits)?;
    // The EL3 registers have no accessfn.
    let trap = if r.access == PL3_RW { Trap::None } else { Trap::Gic };
    Some(Reg { name: r.name, key: key_, access: r.access, trap, kind: Kind::Special, feat: always })
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

/// 64 random bits, `qemu_guest_getrandom()` without `-seed`.
fn guest_random_u64() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(0);
    h.finish()
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
    if let Some(v) = arm.gic_read(cpu.core.shared().cpu_index, key_, &st) {
        return v;
    }
    if pmu::is_pmu_key(key_) {
        let mut st = st;
        let v = Pmu::new(arm, cpu.core.shared()).read(&mut st, key_);
        super::commit(cpu, &mut st);
        return v;
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
        // isr_read() without FEAT_NMI: the pending physical or virtual IRQ, FIQ and SError.
        ISR_EL1 => {
            let hcr = if st.current_el() == 1 { st.hcr_el2_eff(f) } else { 0 };
            let pending = cpu.core.shared().interrupt_request();
            let (irq, fiq) = (
                if hcr & HCR_IMO != 0 { super::INTERRUPT_VIRQ } else { interrupt::HARD },
                if hcr & HCR_FMO != 0 { super::INTERRUPT_VFIQ } else { super::INTERRUPT_FIQ },
            );
            let mut v = 0;
            if pending & irq != 0 {
                v |= u64::from(PSTATE_I);
            }
            if pending & fiq != 0 {
                v |= u64::from(PSTATE_F);
            }
            if hcr & HCR_AMO != 0 && pending & super::INTERRUPT_VSERR != 0 {
                v |= u64::from(PSTATE_A);
            }
            v
        }
        MPIDR_EL1 => {
            if el1_with_el2 {
                st.vmpidr_el2
            } else {
                // mpidr_read_val(): the affinity with bit 31 RES1.
                (1 << 31) | arm.mp_affinity(cpu.core.shared().cpu_index)
            }
        }
        // ccsidr_read(): QEMU's array has 16 entries and the index is masked to fit; the
        // entries past the 8 a model gives are all zero.
        CCSIDR_EL1 => model.ccsidr.get((st.csselr_el1 & 15) as usize).copied().unwrap_or(0),
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
        TCR2_EL1 | TCR2_EL2 => st.tcr2_el[el_of(key_)],
        HCR_EL2 => st.hcr_el2,
        SCR_EL3 => st.scr_el3,
        GPCCR_EL3 => st.gpccr_el3,
        GPCBW_EL3 => st.gpcbw_el3,
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
        TCO if f.mte >= 2 => u64::from(st.pstate & PSTATE_TCO),
        TCO => 0,
        SSBS => u64::from(st.pstate & PSTATE_SSBS),
        // l2ctlr_read(): the number of cores less one in bits 25:24.
        L2CTLR_EL1 => (arm.core_count().clamp(1, 4) as u64 - 1) << 24,
        RNDR | RNDRRS => {
            // rndr_readfn(): NZCV is 0b0000 for a good number; getting one never fails here.
            let mut st = st;
            st.set_nzcv(0);
            super::commit(cpu, &mut st);
            guest_random_u64()
        }
        OSLSR_EL1 => st.oslsr_el1,
        ZCR_EL1 | ZCR_EL2 | ZCR_EL3 => st.zcr_el[el_of(key_)],
        CNTPCT_EL0 => gtimer::phys_count(arm, &st),
        CNTVCT_EL0 if f.user_only => super::user::user_cntvct(model.cntfrq),
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
        super::commit(cpu, &mut st);
        return;
    }
    if arm.gic_write(cpu.core.shared().cpu_index, key_, &st, value) {
        return;
    }
    if pmu::is_pmu_key(key_) {
        Pmu::new(arm, cpu.core.shared()).write(&mut st, key_, value);
        super::commit(cpu, &mut st);
        return;
    }
    match key_ {
        SCTLR_EL1 | SCTLR_EL2 | SCTLR_EL3 => {
            let el3 = key_ == SCTLR_EL3;
            let value = if f.mte < 2 {
                if el3 {
                    value & !(SCTLR_ITFSB | SCTLR_TCF | SCTLR_ATA | SCTLR_TCSO)
                } else {
                    value
                        & !(SCTLR_ITFSB
                            | SCTLR_TCF0
                            | SCTLR_TCF
                            | SCTLR_ATA0
                            | SCTLR_ATA
                            | SCTLR_TCSO
                            | SCTLR_TCSO0)
                }
            } else if el3 {
                // No FEAT_MTE_STORE_ONLY.
                value & !SCTLR_TCSO
            } else {
                value & !(SCTLR_TCSO | SCTLR_TCSO0)
            };
            st.sctlr_el[el_of(key_)] = value;
            super::commit(cpu, &mut st);
            // This may enable or disable the MMU, so do a TLB flush.
            tlb_flush(cpu);
        }
        TCR_EL1 | TCR_EL2 | TCR_EL3 => {
            // vmsa_tcr_el12_write(): flush, then write.
            tlb_flush(cpu);
            st.tcr_el[el_of(key_)] = value;
            super::commit(cpu, &mut st);
        }
        TCR2_EL1 | TCR2_EL2 => {
            // tcr2_el1_write() and tcr2_el2_write(): of the fields only ASID2's are
            // implemented, and they change nothing but the TLB flush on a change of A2, as
            // this TLB has no ASIDs.
            const TCR2_A2: u64 = 1 << 16;
            let el = el_of(key_);
            let valid = if f.asid2 { 0x7 << 16 } else { 0 };
            let old = st.tcr2_el[el];
            st.tcr2_el[el] = value & valid;
            super::commit(cpu, &mut st);
            if f.asid2 && (old ^ value) & TCR2_A2 != 0 {
                tlb_flush_by_mmuidx(cpu, if el == 1 { E10_MASK } else { ALLE2_MASK });
            }
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
            super::commit(cpu, &mut st);
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
            super::commit(cpu, &mut st);
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
            super::commit(cpu, &mut st);
            // If the NS or NSE bit changes, the Security state of EL2 and below changes.
            if (old ^ st.scr_el3) & (SCR_NS | SCR_NSE) != 0 {
                tlb_flush(cpu);
            }
        }
        GPCCR_EL3 => {
            // gpccr_write(): L0GPTSZ is read only and the bits not mentioned are RES0. The
            // model has FEAT_RME_GPC3, so the GPC2 and GPC3 fields are writable. As in
            // QEMU, this does not flush the TLB.
            let rw_mask = 0x7 // PPS
                | (0xf << 8) // IRGN, ORGN
                | (0xf << 12) // SH, PGS
                | (0x3 << 16) // GPC, GPCP
                | (1 << 24) | (1 << 19) | (0x7 << 5) // APPSAA, NSO, SPAD, NSPAD, RLPAD
                | (1 << 29); // GPCBW
            st.gpccr_el3 = (value & rw_mask) | (st.gpccr_el3 & !rw_mask);
            super::commit(cpu, &mut st);
        }
        GPCBW_EL3 => {
            // gpcbw_write(): BWADDR, BWSTRIDE and BWSIZE.
            tlb_flush(cpu);
            let rw_mask = ((1u64 << 25) - 1) | (0x1f << 32) | (0x7 << 37);
            st.gpcbw_el3 = value & rw_mask;
            super::commit(cpu, &mut st);
        }
        VTTBR_EL2 | VTCR_EL2 => {
            let slot = if key_ == VTTBR_EL2 { &mut st.vttbr_el2 } else { &mut st.vtcr_el2 };
            let old = *slot;
            *slot = value;
            super::commit(cpu, &mut st);
            if old != value {
                // The TLB has no VMID tags, so a new stage 2 setup flushes EL1&0.
                tlb_flush_by_mmuidx(cpu, E10_MASK);
            }
        }
        CNTVOFF_EL2 => {
            gtimer::cntvoff_write(arm, cpu, &mut st, value);
            super::commit(cpu, &mut st);
        }
        CNTHCTL_EL2 => {
            st.cnthctl_el2 = value & 0xfff;
            super::commit(cpu, &mut st);
        }
        NZCV => {
            st.set_nzcv(value as u32);
            super::commit(cpu, &mut st);
        }
        DAIF => {
            st.daif = value as u32 & PSTATE_DAIF;
            super::commit(cpu, &mut st);
        }
        FPCR => {
            vfp::set_fpcr(&mut st, value as u32, f);
            super::commit(cpu, &mut st);
        }
        FPSR => {
            vfp::set_fpsr(&mut st, value as u32);
            super::commit(cpu, &mut st);
        }
        SPSEL => {
            st.update_spsel(value as u32);
            super::commit(cpu, &mut st);
        }
        PAN => {
            st.pstate = (st.pstate & !PSTATE_PAN) | (value as u32 & PSTATE_PAN);
            super::commit(cpu, &mut st);
        }
        UAO => {
            st.pstate = (st.pstate & !PSTATE_UAO) | (value as u32 & PSTATE_UAO);
            super::commit(cpu, &mut st);
        }
        SSBS => {
            st.pstate = (st.pstate & !PSTATE_SSBS) | (value as u32 & PSTATE_SSBS);
            super::commit(cpu, &mut st);
        }
        // arm_cp_write_ignore.
        L2CTLR_EL1 => {}
        TCO if f.mte < 2 => {}
        TCO => {
            st.pstate = (st.pstate & !PSTATE_TCO) | (value as u32 & PSTATE_TCO);
            super::commit(cpu, &mut st);
        }
        OSLAR_EL1 => {
            // oslar_write(): OSLSR_EL1.OSLK follows bit 0.
            st.oslsr_el1 = (st.oslsr_el1 & !2) | ((value & 1) << 1);
            super::commit(cpu, &mut st);
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
            super::commit(cpu, &mut st);
        }
        _ if key_ >> 14 == 1 && (key_ >> 7) & 0xf == 7 => at(arm, cpu, &st, key_, value),
        // CRn 9 is the nXS forms.
        _ if key_ >> 14 == 1 && matches!((key_ >> 7) & 0xf, 8 | 9) => {
            tlbi(cpu, f, &st, key_, value);
        }
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
    super::commit(cpu, &mut st);
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
    // CRm 3 and 0 are the Inner Shareable forms and CRm 1 the Outer Shareable ones, which
    // are broadcast.
    let shareable = matches!(crm, 0 | 1 | 3);
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
    use super::{
        CORTEX_A57_REGS, EL2_REGS, EL3_REGS, NEOVERSE_N1_REGS, REGS, TCR2_EL1, TCR2_EL2, key,
        lookup,
    };
    use crate::cpu::ArmCpuModel;

    #[test]
    fn feature_gated_registers() {
        let max = ArmCpuModel::max().with_el2().features;
        let a76 = ArmCpuModel::cortex_a76().with_el2().features;
        // TLBI VMALLE1NXS, VMALLE1OSNXS, ALLE2OSNXS and IPAS2E1OSNXS.
        for k in [key(1, 0, 9, 7, 0), key(1, 0, 9, 1, 0), key(1, 4, 9, 1, 0), key(1, 4, 9, 4, 0)] {
            assert_eq!(lookup(k, &max).map(|r| r.key), Some(k), "0x{k:x}");
            assert!(lookup(k, &a76).is_none(), "0x{k:x}");
        }
        // TLBI VMALLE1OS and VAE3OS (no EL3 on either).
        assert!(lookup(key(1, 0, 8, 1, 0), &max).is_some());
        assert!(lookup(key(1, 0, 8, 1, 0), &a76).is_none());
        assert!(lookup(key(1, 6, 8, 1, 1), &max).is_none());
        for k in [TCR2_EL1, TCR2_EL2] {
            assert!(lookup(k, &max).is_some());
            assert!(lookup(k, &a76).is_none());
        }
    }

    #[test]
    fn impdef_registers() {
        let a57 = ArmCpuModel::cortex_a57().features;
        let n1 = ArmCpuModel::neoverse_n1().features;
        let max = ArmCpuModel::max().features;
        // CPUECTLR_EL1 of the A57 and of the N1.
        let a57_ectlr = key(3, 1, 15, 2, 1);
        let n1_ectlr = key(3, 0, 15, 1, 4);
        assert!(lookup(a57_ectlr, &a57).is_some());
        assert!(lookup(a57_ectlr, &n1).is_none());
        assert!(lookup(n1_ectlr, &n1).is_some());
        assert!(lookup(n1_ectlr, &a57).is_none());
        assert!(lookup(n1_ectlr, &max).is_none());
        for t in [CORTEX_A57_REGS, NEOVERSE_N1_REGS] {
            for (i, a) in t.iter().enumerate() {
                for b in &t[i + 1..] {
                    assert!(a.key != b.key, "{} and {} share an encoding", a.name, b.name);
                }
            }
        }
        // SSBS.
        assert!(lookup(key(3, 3, 4, 2, 6), &n1).is_some());
        assert!(lookup(key(3, 3, 4, 2, 6), &a57).is_none());
    }

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
