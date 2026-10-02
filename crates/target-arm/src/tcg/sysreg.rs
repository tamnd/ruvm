// SPDX-License-Identifier: GPL-2.0-or-later

//! The AArch64 system registers of EL0 and EL1: the port of the parts of QEMU's
//! `target/arm/helper.c` (`v8_cp_reginfo`, `vmsa_cp_reginfo`, `generic_timer_cp_reginfo`,
//! the ID registers and the TLBI, AT and cache maintenance operations) and `debug_helper.c`
//! that this slice needs.
//!
//! Each register is a [`Reg`]: its encoding, the static access rights QEMU keeps in
//! `ARMCPRegInfo.access`, the EL0 trap that its `accessfn` checks at run time, and how it is
//! read and written. Plain storage is accessed by generated code at an `env` offset; the
//! rest goes through the `get_sysreg` and `set_sysreg` helpers, which call [`read`] and
//! [`write`].

use std::mem::offset_of;

use ruvm_jit::cputlb::{
    tlb_flush, tlb_flush_by_mmuidx, tlb_flush_by_mmuidx_all_cpus_synced,
    tlb_flush_page_bits_by_mmuidx, tlb_flush_page_bits_by_mmuidx_all_cpus_synced,
};
use ruvm_jit::{Cpu, MmuAccessType};

use super::{Arm, arm_of, ptw, vfp};
use crate::cpu::{
    ArmCpuModel, ArmFeatures, CpuArmState, MMU_IDX_E10_0, MMU_IDX_E10_1, MMU_IDX_E10_1_PAN,
    PSTATE_DAIF, PSTATE_PAN, PSTATE_SP, PSTATE_UAO, SCTLR_DZE, SCTLR_UCI, SCTLR_UCT, SCTLR_UMA,
    env_off,
};

/// `PL0_W`: writable at EL0 (and so at EL1).
pub(crate) const PL0_W: u8 = 1 | PL1_W;
/// `PL0_R`: readable at EL0 (and so at EL1).
pub(crate) const PL0_R: u8 = 2 | PL1_R;
/// `PL1_W`.
pub(crate) const PL1_W: u8 = 4;
/// `PL1_R`.
pub(crate) const PL1_R: u8 = 8;
/// `PL1_RW`.
pub(crate) const PL1_RW: u8 = PL1_R | PL1_W;
/// `PL0_RW`.
pub(crate) const PL0_RW: u8 = PL0_R | PL0_W;

/// The encoding of a system register as one number, `ENCODE_AA64_CP_REG()` without the
/// coprocessor bits.
pub(crate) const fn key(op0: u32, op1: u32, crn: u32, crm: u32, op2: u32) -> u32 {
    (op0 << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2
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
    /// `ctr_el0_access()`: SCTLR_EL1.UCT.
    Uct,
    /// `aa64_zva_access()`: SCTLR_EL1.DZE.
    Dze,
    /// `aa64_daif_access()`: SCTLR_EL1.UMA.
    Uma,
    /// `aa64_cacheop_poc_access()` and `aa64_cacheop_pou_access()`: SCTLR_EL1.UCI.
    Uci,
    /// `gt_cntfrq_access()`: CNTKCTL_EL1.EL0PCTEN or EL0VCTEN.
    CntFrq,
    /// `gt_pct_access()`: CNTKCTL_EL1.EL0PCTEN.
    CntPct,
    /// `gt_vct_access()`: CNTKCTL_EL1.EL0VCTEN.
    CntVct,
    /// `gt_ptimer_access()`: CNTKCTL_EL1.EL0PTEN.
    CntPTimer,
    /// `gt_vtimer_access()`: CNTKCTL_EL1.EL0VTEN.
    CntVTimer,
    /// `sp_el0_access()`: SP_EL0 is not accessible while it is the current SP.
    SpEl0,
}

/// What a run time access check decides, `CPAccessResult`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Access {
    /// `CP_ACCESS_OK`.
    Ok,
    /// `CP_ACCESS_TRAP_EL1`: a system register trap to EL1.
    TrapEl1,
    /// `CP_ACCESS_UNDEFINED`: an uncategorized UNDEF.
    Undefined,
}

impl Trap {
    /// Whether the check can fail at `el`, so that the translator has to call it.
    pub(crate) fn applies(self, el: u32) -> bool {
        match self {
            Trap::None => false,
            Trap::SpEl0 => el == 1,
            _ => el == 0,
        }
    }

    /// Run the check.
    pub(crate) fn check(self, st: &CpuArmState) -> Access {
        let el = st.current_el();
        let sctlr = st.sctlr_el[1];
        let cntkctl = st.cntkctl_el1;
        let trap_unless = |ok: bool| if ok { Access::Ok } else { Access::TrapEl1 };
        if !self.applies(el) {
            return Access::Ok;
        }
        match self {
            Trap::None => Access::Ok,
            Trap::Uct => trap_unless(sctlr & SCTLR_UCT != 0),
            Trap::Dze => trap_unless(sctlr & SCTLR_DZE != 0),
            Trap::Uma => trap_unless(sctlr & SCTLR_UMA != 0),
            Trap::Uci => trap_unless(sctlr & SCTLR_UCI != 0),
            Trap::CntFrq => trap_unless(cntkctl & 3 != 0),
            Trap::CntPct => trap_unless(cntkctl & 1 != 0),
            Trap::CntVct => trap_unless(cntkctl & 2 != 0),
            Trap::CntPTimer => trap_unless(cntkctl & (1 << 9) != 0),
            Trap::CntVTimer => trap_unless(cntkctl & (1 << 8) != 0),
            Trap::SpEl0 => {
                if st.pstate & PSTATE_SP == 0 {
                    // When SPSel is 0, SP_EL0 is the current SP and is inaccessible.
                    Access::Undefined
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

/// A system register, the parts of `ARMCPRegInfo` this slice uses.
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
/// CNTP_TVAL_EL0.
pub(crate) const CNTP_TVAL_EL0: u32 = key(3, 3, 14, 2, 0);
/// CNTP_CTL_EL0.
pub(crate) const CNTP_CTL_EL0: u32 = key(3, 3, 14, 2, 1);
/// CNTP_CVAL_EL0.
pub(crate) const CNTP_CVAL_EL0: u32 = key(3, 3, 14, 2, 2);
/// CNTV_TVAL_EL0.
pub(crate) const CNTV_TVAL_EL0: u32 = key(3, 3, 14, 3, 0);
/// CNTV_CTL_EL0.
pub(crate) const CNTV_CTL_EL0: u32 = key(3, 3, 14, 3, 1);
/// CNTV_CVAL_EL0.
pub(crate) const CNTV_CVAL_EL0: u32 = key(3, 3, 14, 3, 2);
/// AT S1E1R.
pub(crate) const AT_S1E1R: u32 = key(1, 0, 7, 8, 0);
/// AT S1E1W.
pub(crate) const AT_S1E1W: u32 = key(1, 0, 7, 8, 1);
/// AT S1E0R.
pub(crate) const AT_S1E0R: u32 = key(1, 0, 7, 8, 2);
/// AT S1E0W.
pub(crate) const AT_S1E0W: u32 = key(1, 0, 7, 8, 3);

/// The TLBI operations: op0 1, op1 0, CRn 8, CRm 3 (Inner Shareable) or 7, and op2.
const TLBI_VMALLE1: u32 = 0;
const TLBI_VAE1: u32 = 1;
const TLBI_ASIDE1: u32 = 2;
const TLBI_VAAE1: u32 = 3;
const TLBI_VALE1: u32 = 5;
const TLBI_VAALE1: u32 = 7;

macro_rules! tlbi {
    ($name:literal, $crm:expr, $op2:expr) => {
        r!($name, (1, 0, 8, $crm, $op2), PL1_W, None, Kind::Special)
    };
}

/// The registers, in the order of the Arm ARM's register index.
static REGS: &[Reg] = &[
    // Identification.
    r!("MIDR_EL1", (3, 0, 0, 0, 0), PL1_R, None, Kind::Model(|m| m.midr)),
    r!("MPIDR_EL1", (3, 0, 0, 0, 5), PL1_R, None, Kind::Special),
    r!("REVIDR_EL1", (3, 0, 0, 0, 6), PL1_R, None, Kind::Model(|m| m.revidr)),
    r!("ID_AA64PFR0_EL1", (3, 0, 0, 4, 0), PL1_R, None, Kind::Model(|m| m.id_aa64pfr0)),
    r!("ID_AA64PFR1_EL1", (3, 0, 0, 4, 1), PL1_R, None, Kind::Model(|m| m.id_aa64pfr1)),
    r!("ID_AA64DFR0_EL1", (3, 0, 0, 5, 0), PL1_R, None, Kind::Model(|m| m.id_aa64dfr0)),
    r!("ID_AA64ISAR0_EL1", (3, 0, 0, 6, 0), PL1_R, None, Kind::Model(|m| m.id_aa64isar0)),
    r!("ID_AA64ISAR1_EL1", (3, 0, 0, 6, 1), PL1_R, None, Kind::Model(|m| m.id_aa64isar1)),
    r!("ID_AA64MMFR0_EL1", (3, 0, 0, 7, 0), PL1_R, None, Kind::Model(|m| m.id_aa64mmfr0)),
    r!("ID_AA64MMFR1_EL1", (3, 0, 0, 7, 1), PL1_R, None, Kind::Model(|m| m.id_aa64mmfr1)),
    r!("ID_AA64MMFR2_EL1", (3, 0, 0, 7, 2), PL1_R, None, Kind::Model(|m| m.id_aa64mmfr2)),
    r!("CCSIDR_EL1", (3, 1, 0, 0, 0), PL1_R, None, Kind::Special),
    r!("CLIDR_EL1", (3, 1, 0, 0, 1), PL1_R, None, Kind::Model(|m| m.clidr)),
    r!("AIDR_EL1", (3, 1, 0, 0, 7), PL1_R, None, Kind::Zero),
    r!("CSSELR_EL1", (3, 2, 0, 0, 0), PL1_RW, None, field32(off!(csselr_el1))),
    r!("CTR_EL0", (3, 3, 0, 0, 1), PL0_R, Uct, Kind::Special),
    r!("DCZID_EL0", (3, 3, 0, 0, 7), PL0_R, None, Kind::Special),
    // System control.
    r!("SCTLR_EL1", (3, 0, 1, 0, 0), PL1_RW, None, Kind::Special),
    r!("ACTLR_EL1", (3, 0, 1, 0, 1), PL1_RW, None, Kind::Zero),
    r!(
        "CPACR_EL1",
        (3, 0, 1, 0, 2),
        PL1_RW,
        None,
        // cpacr_write() keeps every bit in ARMv8.
        field(off!(cpacr_el1))
    ),
    // Memory management.
    r!("TTBR0_EL1", (3, 0, 2, 0, 0), PL1_RW, None, Kind::Special),
    r!("TTBR1_EL1", (3, 0, 2, 0, 1), PL1_RW, None, Kind::Special),
    r!("TCR_EL1", (3, 0, 2, 0, 2), PL1_RW, None, Kind::Special),
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
    r!("AFSR0_EL1", (3, 0, 5, 1, 0), PL1_RW, None, field(off!(afsr0_el1))),
    r!("AFSR1_EL1", (3, 0, 5, 1, 1), PL1_RW, None, field(off!(afsr1_el1))),
    r!("ESR_EL1", (3, 0, 5, 2, 0), PL1_RW, None, field(off!(esr_el[1]))),
    r!("FAR_EL1", (3, 0, 6, 0, 0), PL1_RW, None, field(off!(far_el[1]))),
    r!("PAR_EL1", (3, 0, 7, 4, 0), PL1_RW, None, field(off!(par_el1))),
    r!("MAIR_EL1", (3, 0, 10, 2, 0), PL1_RW, None, field(off!(mair_el[1]))),
    r!("AMAIR_EL1", (3, 0, 10, 3, 0), PL1_RW, None, field(off!(amair_el1))),
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
    r!("CONTEXTIDR_EL1", (3, 0, 13, 0, 1), PL1_RW, None, field32(off!(contextidr_el1))),
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
    r!("IC_IALLUIS", (1, 0, 7, 1, 0), PL1_W, None, Kind::Nop),
    r!("IC_IALLU", (1, 0, 7, 5, 0), PL1_W, None, Kind::Nop),
    r!("IC_IVAU", (1, 3, 7, 5, 1), PL0_W, Uci, Kind::Nop),
    r!("DC_IVAC", (1, 0, 7, 6, 1), PL1_W, None, Kind::Nop),
    r!("DC_ISW", (1, 0, 7, 6, 2), PL1_W, None, Kind::Nop),
    r!("DC_CSW", (1, 0, 7, 10, 2), PL1_W, None, Kind::Nop),
    r!("DC_CISW", (1, 0, 7, 14, 2), PL1_W, None, Kind::Nop),
    r!("DC_ZVA", (1, 3, 7, 4, 1), PL0_W, Dze, Kind::DcZva),
    r!("DC_CVAC", (1, 3, 7, 10, 1), PL0_W, Uci, Kind::Nop),
    r!("DC_CVAU", (1, 3, 7, 11, 1), PL0_W, Uci, Kind::Nop),
    r!("DC_CVAP", (1, 3, 7, 12, 1), PL0_W, Uci, Kind::Nop, has_dpb),
    r!("DC_CIVAC", (1, 3, 7, 14, 1), PL0_W, Uci, Kind::Nop),
    // Address translation.
    r!("AT_S1E1R", (1, 0, 7, 8, 0), PL1_W, None, Kind::Special),
    r!("AT_S1E1W", (1, 0, 7, 8, 1), PL1_W, None, Kind::Special),
    r!("AT_S1E0R", (1, 0, 7, 8, 2), PL1_W, None, Kind::Special),
    r!("AT_S1E0W", (1, 0, 7, 8, 3), PL1_W, None, Kind::Special),
    // TLB maintenance.
    tlbi!("TLBI_VMALLE1IS", 3, 0),
    tlbi!("TLBI_VAE1IS", 3, 1),
    tlbi!("TLBI_ASIDE1IS", 3, 2),
    tlbi!("TLBI_VAAE1IS", 3, 3),
    tlbi!("TLBI_VALE1IS", 3, 5),
    tlbi!("TLBI_VAALE1IS", 3, 7),
    tlbi!("TLBI_VMALLE1", 7, 0),
    tlbi!("TLBI_VAE1", 7, 1),
    tlbi!("TLBI_ASIDE1", 7, 2),
    tlbi!("TLBI_VAAE1", 7, 3),
    tlbi!("TLBI_VALE1", 7, 5),
    tlbi!("TLBI_VAALE1", 7, 7),
];

/// The register with encoding `key` on a CPU with `feat`, as `get_arm_cp_reginfo()` finds
/// it. The ID register space that is not listed reads as zero at EL1.
pub(crate) fn lookup(key_: u32, feat: &ArmFeatures) -> Option<Reg> {
    if let Some(r) = REGS.iter().find(|r| r.key == key_ && (r.feat)(feat)) {
        return Some(*r);
    }
    let op0 = key_ >> 14;
    let op1 = (key_ >> 11) & 7;
    let crn = (key_ >> 7) & 0xf;
    let crm = (key_ >> 3) & 0xf;
    if op0 == 3 && op1 == 0 && crn == 0 && (1..=7).contains(&crm) {
        return Some(Reg {
            name: "ID_RAZ",
            key: key_,
            access: PL1_R,
            trap: Trap::None,
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

/// The EL1&0 regime MMU indexes.
const E10_MASK: u32 = (1 << MMU_IDX_E10_0) | (1 << MMU_IDX_E10_1) | (1 << MMU_IDX_E10_1_PAN);

/// `tlbbits_for_regime()`: 56 when TBI applies to `addr`, otherwise 64.
fn tlbbits(st: &CpuArmState, addr: u64) -> u32 {
    let tcr = st.tcr_el[1];
    let tbi = (tcr >> 37) & 3;
    if (tbi >> ((addr >> 55) & 1)) & 1 != 0 { 56 } else { 64 }
}

/// The value of a generic timer CTL register with ISTATUS computed, `gt_recalc_timer()`.
fn timer_ctl(ctl: u64, cval: u64, count: u64) -> u64 {
    if ctl & 1 != 0 {
        let istatus = u64::from(count >= cval);
        (ctl & !4) | (istatus << 2)
    } else {
        // Timer disabled: ISTATUS and the timer output are always clear.
        ctl & !4
    }
}

/// Read a [`Kind::Special`] register.
pub(crate) fn read(cpu: &mut Cpu<'_>, key_: u32) -> u64 {
    let ops = cpu.ops();
    let arm: &Arm = arm_of(&ops);
    let model = arm.model();
    let st = CpuArmState::load(cpu.env);
    let freq = st.cntfrq_el0;
    let count = || arm.counter(freq);
    match key_ {
        MPIDR_EL1 => {
            // mpidr_read_val(): Aff0 is the index within a cluster of eight, Aff1 the
            // cluster, and bit 31 is RES1.
            let idx = cpu.core.shared().cpu_index as u64;
            (1 << 31) | ((idx / 8) << 8) | (idx % 8)
        }
        CCSIDR_EL1 => ccsidr(st.csselr_el1),
        CTR_EL0 => model.ctr,
        DCZID_EL0 => {
            // aa64_dczid_read(): DZP is set when DC ZVA is not allowed.
            let prohibited = st.current_el() == 0 && st.sctlr_el[1] & SCTLR_DZE == 0;
            model.dczid | (u64::from(prohibited) << 4)
        }
        SCTLR_EL1 => st.sctlr_el[1],
        TTBR0_EL1 => st.ttbr0_el[1],
        TTBR1_EL1 => st.ttbr1_el[1],
        TCR_EL1 => st.tcr_el[1],
        NZCV => u64::from(st.nzcv()),
        DAIF => u64::from(st.daif & PSTATE_DAIF),
        FPCR => u64::from(vfp::get_fpcr(&st)),
        FPSR => u64::from(vfp::get_fpsr(&st)),
        SPSEL => u64::from(st.pstate & PSTATE_SP),
        PAN => u64::from(st.pstate & PSTATE_PAN),
        UAO => u64::from(st.pstate & PSTATE_UAO),
        OSLSR_EL1 => st.oslsr_el1,
        CNTPCT_EL0 | CNTVCT_EL0 => count(),
        CNTP_CTL_EL0 => timer_ctl(st.cntp_ctl_el0, st.cntp_cval_el0, count()),
        CNTV_CTL_EL0 => timer_ctl(st.cntv_ctl_el0, st.cntv_cval_el0, count()),
        CNTP_CVAL_EL0 => st.cntp_cval_el0,
        CNTV_CVAL_EL0 => st.cntv_cval_el0,
        CNTP_TVAL_EL0 => u64::from(st.cntp_cval_el0.wrapping_sub(count()) as u32),
        CNTV_TVAL_EL0 => u64::from(st.cntv_cval_el0.wrapping_sub(count()) as u32),
        _ => panic!("no read for system register key 0x{key_:x}"),
    }
}

/// Write a [`Kind::Special`] register.
pub(crate) fn write(cpu: &mut Cpu<'_>, key_: u32, value: u64) {
    let ops = cpu.ops();
    let arm: &Arm = arm_of(&ops);
    let mut st = CpuArmState::load(cpu.env);
    let freq = st.cntfrq_el0;
    let count = || arm.counter(freq);
    let sext32 = |v: u64| v as u32 as i32 as i64 as u64;
    match key_ {
        SCTLR_EL1 => {
            st.sctlr_el[1] = value;
            st.store(cpu.env);
            // This may enable or disable the MMU, so do a TLB flush.
            tlb_flush(cpu);
        }
        TCR_EL1 => {
            // vmsa_tcr_el12_write(): flush, then write.
            tlb_flush(cpu);
            st.tcr_el[1] = value;
            st.store(cpu.env);
        }
        TTBR0_EL1 | TTBR1_EL1 => {
            let slot = if key_ == TTBR0_EL1 { &mut st.ttbr0_el[1] } else { &mut st.ttbr1_el[1] };
            let old = *slot;
            *slot = value;
            st.store(cpu.env);
            // If the ASID changes we must flush the TLB.
            if (old ^ value) >> 48 != 0 {
                tlb_flush_by_mmuidx(cpu, E10_MASK);
            }
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
            vfp::set_fpcr(&mut st, value as u32, &arm.model().features);
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
        CNTP_CTL_EL0 => {
            st.cntp_ctl_el0 = (st.cntp_ctl_el0 & !3) | (value & 3);
            st.store(cpu.env);
        }
        CNTV_CTL_EL0 => {
            st.cntv_ctl_el0 = (st.cntv_ctl_el0 & !3) | (value & 3);
            st.store(cpu.env);
        }
        CNTP_CVAL_EL0 => {
            st.cntp_cval_el0 = value;
            st.store(cpu.env);
        }
        CNTV_CVAL_EL0 => {
            st.cntv_cval_el0 = value;
            st.store(cpu.env);
        }
        CNTP_TVAL_EL0 => {
            st.cntp_cval_el0 = count().wrapping_add(sext32(value));
            st.store(cpu.env);
        }
        CNTV_TVAL_EL0 => {
            st.cntv_cval_el0 = count().wrapping_add(sext32(value));
            st.store(cpu.env);
        }
        AT_S1E1R | AT_S1E1W | AT_S1E0R | AT_S1E0W => {
            let access = if key_ == AT_S1E1W || key_ == AT_S1E0W {
                MmuAccessType::DataStore
            } else {
                MmuAccessType::DataLoad
            };
            let mmu_idx =
                if key_ == AT_S1E0R || key_ == AT_S1E0W { MMU_IDX_E10_0 } else { MMU_IDX_E10_1 };
            let par = match ptw::get_phys_addr(arm, cpu, value, access, mmu_idx, true) {
                // The LPAE bit is always set; NS is set because there is no Secure state.
                Ok(t) => {
                    (1 << 11)
                        | (t.pa & !0xfff & ((1 << 52) - 1))
                        | (1 << 9)
                        | (u64::from(t.attrs) << 56)
                        | (u64::from(t.sh) << 7)
                }
                Err(f) => (1 << 11) | 1 | (u64::from(f.fsc & 0x3f) << 1),
            };
            let mut st = CpuArmState::load(cpu.env);
            st.par_el1 = par;
            st.store(cpu.env);
        }
        _ if key_ >> 3 == key(1, 0, 8, 3, 0) >> 3 || key_ >> 3 == key(1, 0, 8, 7, 0) >> 3 => {
            let shareable = (key_ >> 3) & 0xf == 3;
            tlbi(cpu, &st, key_ & 7, shareable, value);
        }
        _ => panic!("no write for system register key 0x{key_:x}"),
    }
}

/// The TLBI operations of the EL1&0 regime. This TLB has no ASIDs, so the by-ASID forms
/// flush every entry of the regime, and the by-VA forms flush the page for every ASID, as
/// QEMU does.
fn tlbi(cpu: &mut Cpu<'_>, st: &CpuArmState, op2: u32, shareable: bool, value: u64) {
    match op2 {
        TLBI_VMALLE1 | TLBI_ASIDE1 => {
            if shareable {
                tlb_flush_by_mmuidx_all_cpus_synced(cpu, E10_MASK);
            } else {
                tlb_flush_by_mmuidx(cpu, E10_MASK);
            }
        }
        TLBI_VAE1 | TLBI_VAAE1 | TLBI_VALE1 | TLBI_VAALE1 => {
            let pageaddr = (((value << 12) << 8) as i64 >> 8) as u64;
            let bits = tlbbits(st, pageaddr);
            if shareable {
                tlb_flush_page_bits_by_mmuidx_all_cpus_synced(cpu, pageaddr, E10_MASK, bits);
            } else {
                tlb_flush_page_bits_by_mmuidx(cpu, pageaddr, E10_MASK, bits);
            }
        }
        _ => unreachable!("TLBI op2 {op2} is not in the table"),
    }
}

#[cfg(test)]
mod tests {
    use super::REGS;

    #[test]
    fn keys_and_names_are_unique() {
        for (i, a) in REGS.iter().enumerate() {
            assert!(!a.name.is_empty());
            for b in &REGS[i + 1..] {
                assert!(a.key != b.key, "{} and {} share an encoding", a.name, b.name);
                assert!(a.name != b.name, "{} is listed twice", a.name);
            }
        }
    }
}
