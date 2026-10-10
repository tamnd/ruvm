// SPDX-License-Identifier: GPL-2.0-or-later

//! System register encodings: the `hv_sys_reg_t` ids the framework takes, the trapped register
//! a sysreg exit names, and the list `hvf_get_registers()` and `hvf_put_registers()` walk, from
//! target/arm/hvf/sysreg.c.inc.

/// An `hv_sys_reg_t` id from its encoding. The framework uses the KVM layout:
/// `op0 << 14 | op1 << 11 | crn << 7 | crm << 3 | op2`.
pub const fn hv(op0: u16, op1: u16, crn: u16, crm: u16, op2: u16) -> u16 {
    (op0 << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2
}

/// The register a system register trap names, kept in the ISS layout QEMU's `SYSREG()` macro
/// uses so the trap's `syndrome & SYSREG_MASK` compares directly.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TrapReg(pub u32);

impl TrapReg {
    /// `SYSREG(op0, op1, crn, crm, op2)`.
    pub const fn new(op0: u32, op1: u32, crn: u32, crm: u32, op2: u32) -> TrapReg {
        TrapReg((op0 << 20) | (op1 << 14) | (crn << 10) | (crm << 1) | (op2 << 17))
    }

    /// Masks the register out of an `EC_SYSTEMREGISTERTRAP` ISS, `SYSREG_MASK`.
    pub const fn from_iss(iss: u32) -> TrapReg {
        TrapReg(iss & Self::new(3, 7, 15, 15, 7).0)
    }

    /// `SYSREG_OP0()`.
    pub const fn op0(self) -> u32 {
        (self.0 >> 20) & 3
    }

    /// `SYSREG_OP1()`.
    pub const fn op1(self) -> u32 {
        (self.0 >> 14) & 7
    }

    /// `SYSREG_CRN()`.
    pub const fn crn(self) -> u32 {
        (self.0 >> 10) & 15
    }

    /// `SYSREG_CRM()`.
    pub const fn crm(self) -> u32 {
        (self.0 >> 1) & 15
    }

    /// `SYSREG_OP2()`.
    pub const fn op2(self) -> u32 {
        (self.0 >> 17) & 7
    }

    /// The same register as an `hv_sys_reg_t` id.
    pub const fn hv_id(self) -> u16 {
        hv(
            self.op0() as u16,
            self.op1() as u16,
            self.crn() as u16,
            self.crm() as u16,
            self.op2() as u16,
        )
    }

    /// The ID register space QEMU reads as zero: op0 3, op1 0, CRn 0, CRm 1 to 7.
    pub const fn is_id_space(self) -> bool {
        self.op0() == 3 && self.op1() == 0 && self.crn() == 0 && self.crm() >= 1 && self.crm() < 8
    }

    /// `DBGBVRn_EL1`, `DBGBCRn_EL1`, `DBGWVRn_EL1` or `DBGWCRn_EL1`: which of the four
    /// (op2 4 to 7) and n, the CRm.
    pub const fn debug_reg(self) -> Option<(DebugReg, usize)> {
        if self.op0() != 2 || self.op1() != 0 || self.crn() != 0 {
            return None;
        }
        let kind = match self.op2() {
            4 => DebugReg::Bvr,
            5 => DebugReg::Bcr,
            6 => DebugReg::Wvr,
            7 => DebugReg::Wcr,
            _ => return None,
        };
        Some((kind, self.crm() as usize))
    }
}

/// One of the four debug register banks.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DebugReg {
    /// `DBGBVRn_EL1`.
    Bvr,
    /// `DBGBCRn_EL1`.
    Bcr,
    /// `DBGWVRn_EL1`.
    Wvr,
    /// `DBGWCRn_EL1`.
    Wcr,
}

/// The trapped registers `hvf_sysreg_read()` and `hvf_sysreg_write()` name, as `SYSREG_*`.
pub mod trap {
    use super::TrapReg;

    pub const OSLAR_EL1: TrapReg = TrapReg::new(2, 0, 1, 0, 4);
    pub const OSLSR_EL1: TrapReg = TrapReg::new(2, 0, 1, 1, 4);
    pub const OSDLR_EL1: TrapReg = TrapReg::new(2, 0, 1, 3, 4);
    pub const LORC_EL1: TrapReg = TrapReg::new(3, 0, 10, 4, 3);
    pub const CNTPCT_EL0: TrapReg = TrapReg::new(3, 3, 14, 0, 1);
    pub const CNTP_TVAL_EL0: TrapReg = TrapReg::new(3, 3, 14, 2, 0);
    pub const CNTP_CTL_EL0: TrapReg = TrapReg::new(3, 3, 14, 2, 1);
    pub const CNTP_CVAL_EL0: TrapReg = TrapReg::new(3, 3, 14, 2, 2);
    pub const CNTHCTL_EL2: TrapReg = TrapReg::new(3, 4, 14, 1, 0);
    pub const MDCCINT_EL1: TrapReg = TrapReg::new(2, 0, 0, 2, 0);
    pub const MDSCR_EL1: TrapReg = TrapReg::new(2, 0, 0, 2, 2);
    pub const PMCR_EL0: TrapReg = TrapReg::new(3, 3, 9, 12, 0);
    pub const PMUSERENR_EL0: TrapReg = TrapReg::new(3, 3, 9, 14, 0);
    pub const PMCNTENSET_EL0: TrapReg = TrapReg::new(3, 3, 9, 12, 1);
    pub const PMCNTENCLR_EL0: TrapReg = TrapReg::new(3, 3, 9, 12, 2);
    pub const PMINTENCLR_EL1: TrapReg = TrapReg::new(3, 0, 9, 14, 2);
    pub const PMOVSCLR_EL0: TrapReg = TrapReg::new(3, 3, 9, 12, 3);
    pub const PMSWINC_EL0: TrapReg = TrapReg::new(3, 3, 9, 12, 4);
    pub const PMSELR_EL0: TrapReg = TrapReg::new(3, 3, 9, 12, 5);
    pub const PMCEID0_EL0: TrapReg = TrapReg::new(3, 3, 9, 12, 6);
    pub const PMCEID1_EL0: TrapReg = TrapReg::new(3, 3, 9, 12, 7);
    pub const PMCCNTR_EL0: TrapReg = TrapReg::new(3, 3, 9, 13, 0);
    pub const PMCCFILTR_EL0: TrapReg = TrapReg::new(3, 3, 14, 15, 7);
    pub const ICC_AP0R0_EL1: TrapReg = TrapReg::new(3, 0, 12, 8, 4);
    pub const ICC_AP0R1_EL1: TrapReg = TrapReg::new(3, 0, 12, 8, 5);
    pub const ICC_AP0R2_EL1: TrapReg = TrapReg::new(3, 0, 12, 8, 6);
    pub const ICC_AP0R3_EL1: TrapReg = TrapReg::new(3, 0, 12, 8, 7);
    pub const ICC_AP1R0_EL1: TrapReg = TrapReg::new(3, 0, 12, 9, 0);
    pub const ICC_AP1R1_EL1: TrapReg = TrapReg::new(3, 0, 12, 9, 1);
    pub const ICC_AP1R2_EL1: TrapReg = TrapReg::new(3, 0, 12, 9, 2);
    pub const ICC_AP1R3_EL1: TrapReg = TrapReg::new(3, 0, 12, 9, 3);
    pub const ICC_ASGI1R_EL1: TrapReg = TrapReg::new(3, 0, 12, 11, 6);
    pub const ICC_BPR0_EL1: TrapReg = TrapReg::new(3, 0, 12, 8, 3);
    pub const ICC_BPR1_EL1: TrapReg = TrapReg::new(3, 0, 12, 12, 3);
    pub const ICC_CTLR_EL1: TrapReg = TrapReg::new(3, 0, 12, 12, 4);
    pub const ICC_DIR_EL1: TrapReg = TrapReg::new(3, 0, 12, 11, 1);
    pub const ICC_EOIR0_EL1: TrapReg = TrapReg::new(3, 0, 12, 8, 1);
    pub const ICC_EOIR1_EL1: TrapReg = TrapReg::new(3, 0, 12, 12, 1);
    pub const ICC_HPPIR0_EL1: TrapReg = TrapReg::new(3, 0, 12, 8, 2);
    pub const ICC_HPPIR1_EL1: TrapReg = TrapReg::new(3, 0, 12, 12, 2);
    pub const ICC_IAR0_EL1: TrapReg = TrapReg::new(3, 0, 12, 8, 0);
    pub const ICC_IAR1_EL1: TrapReg = TrapReg::new(3, 0, 12, 12, 0);
    pub const ICC_IGRPEN0_EL1: TrapReg = TrapReg::new(3, 0, 12, 12, 6);
    pub const ICC_IGRPEN1_EL1: TrapReg = TrapReg::new(3, 0, 12, 12, 7);
    pub const ICC_PMR_EL1: TrapReg = TrapReg::new(3, 0, 4, 6, 0);
    pub const ICC_RPR_EL1: TrapReg = TrapReg::new(3, 0, 12, 11, 3);
    pub const ICC_SGI0R_EL1: TrapReg = TrapReg::new(3, 0, 12, 11, 7);
    pub const ICC_SGI1R_EL1: TrapReg = TrapReg::new(3, 0, 12, 11, 5);
    pub const ICC_SRE_EL1: TrapReg = TrapReg::new(3, 0, 12, 12, 5);
}

/// Who answers a trapped register when the accelerator does not.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TrapOwner {
    /// The PMU registers, `pmu_*` in target/arm/helper.c, with `kernel-irqchip=off`.
    Pmu,
    /// The physical timer, which QEMU runs as an emulated `gt_timer`.
    PhysTimer,
    /// The GICv3 CPU interface of the emulated GIC.
    Gic,
}

impl TrapOwner {
    /// Which emulated block a trapped register belongs to, if any. QEMU only sends these on
    /// when the GIC is not in the framework.
    pub fn of(reg: TrapReg) -> Option<TrapOwner> {
        use trap::*;
        const PMU: [TrapReg; 12] = [
            PMCR_EL0,
            PMUSERENR_EL0,
            PMCNTENSET_EL0,
            PMCNTENCLR_EL0,
            PMINTENCLR_EL1,
            PMOVSCLR_EL0,
            PMSWINC_EL0,
            PMSELR_EL0,
            PMCCNTR_EL0,
            PMCCFILTR_EL0,
            PMCEID0_EL0,
            PMCEID1_EL0,
        ];
        const TIMER: [TrapReg; 4] = [CNTPCT_EL0, CNTP_TVAL_EL0, CNTP_CTL_EL0, CNTP_CVAL_EL0];
        const GIC: [TrapReg; 25] = [
            ICC_AP0R0_EL1,
            ICC_AP0R1_EL1,
            ICC_AP0R2_EL1,
            ICC_AP0R3_EL1,
            ICC_AP1R0_EL1,
            ICC_AP1R1_EL1,
            ICC_AP1R2_EL1,
            ICC_AP1R3_EL1,
            ICC_ASGI1R_EL1,
            ICC_BPR0_EL1,
            ICC_BPR1_EL1,
            ICC_CTLR_EL1,
            ICC_DIR_EL1,
            ICC_EOIR0_EL1,
            ICC_EOIR1_EL1,
            ICC_HPPIR0_EL1,
            ICC_HPPIR1_EL1,
            ICC_IAR0_EL1,
            ICC_IAR1_EL1,
            ICC_IGRPEN0_EL1,
            ICC_IGRPEN1_EL1,
            ICC_PMR_EL1,
            ICC_RPR_EL1,
            ICC_SGI0R_EL1,
            ICC_SGI1R_EL1,
        ];
        if PMU.contains(&reg) {
            Some(TrapOwner::Pmu)
        } else if TIMER.contains(&reg) {
            Some(TrapOwner::PhysTimer)
        } else if GIC.contains(&reg) || reg == ICC_SRE_EL1 {
            Some(TrapOwner::Gic)
        } else {
            None
        }
    }
}

/// `hv_sys_reg_t` ids for the registers this crate and ruvm-target-arm name.
pub mod id {
    use super::hv;

    pub const MIDR_EL1: u16 = hv(3, 0, 0, 0, 0);
    pub const MPIDR_EL1: u16 = hv(3, 0, 0, 0, 5);
    pub const ID_AA64PFR0_EL1: u16 = hv(3, 0, 0, 4, 0);
    pub const ID_AA64PFR1_EL1: u16 = hv(3, 0, 0, 4, 1);
    pub const ID_AA64ZFR0_EL1: u16 = hv(3, 0, 0, 4, 4);
    pub const ID_AA64SMFR0_EL1: u16 = hv(3, 0, 0, 4, 5);
    pub const ID_AA64DFR0_EL1: u16 = hv(3, 0, 0, 5, 0);
    pub const ID_AA64DFR1_EL1: u16 = hv(3, 0, 0, 5, 1);
    pub const ID_AA64ISAR0_EL1: u16 = hv(3, 0, 0, 6, 0);
    pub const ID_AA64ISAR1_EL1: u16 = hv(3, 0, 0, 6, 1);
    pub const ID_AA64MMFR0_EL1: u16 = hv(3, 0, 0, 7, 0);
    pub const ID_AA64MMFR1_EL1: u16 = hv(3, 0, 0, 7, 1);
    pub const ID_AA64MMFR2_EL1: u16 = hv(3, 0, 0, 7, 2);
    pub const MDCCINT_EL1: u16 = hv(2, 0, 0, 2, 0);
    pub const MDSCR_EL1: u16 = hv(2, 0, 0, 2, 2);
    pub const SCTLR_EL1: u16 = hv(3, 0, 1, 0, 0);
    pub const CPACR_EL1: u16 = hv(3, 0, 1, 0, 2);
    pub const SMPRI_EL1: u16 = hv(3, 0, 1, 2, 4);
    pub const SMCR_EL1: u16 = hv(3, 0, 1, 2, 6);
    pub const TTBR0_EL1: u16 = hv(3, 0, 2, 0, 0);
    pub const TTBR1_EL1: u16 = hv(3, 0, 2, 0, 1);
    pub const TCR_EL1: u16 = hv(3, 0, 2, 0, 2);
    pub const APIAKEYLO_EL1: u16 = hv(3, 0, 2, 1, 0);
    pub const APIAKEYHI_EL1: u16 = hv(3, 0, 2, 1, 1);
    pub const APIBKEYLO_EL1: u16 = hv(3, 0, 2, 1, 2);
    pub const APIBKEYHI_EL1: u16 = hv(3, 0, 2, 1, 3);
    pub const APDAKEYLO_EL1: u16 = hv(3, 0, 2, 2, 0);
    pub const APDAKEYHI_EL1: u16 = hv(3, 0, 2, 2, 1);
    pub const APDBKEYLO_EL1: u16 = hv(3, 0, 2, 2, 2);
    pub const APDBKEYHI_EL1: u16 = hv(3, 0, 2, 2, 3);
    pub const APGAKEYLO_EL1: u16 = hv(3, 0, 2, 3, 0);
    pub const APGAKEYHI_EL1: u16 = hv(3, 0, 2, 3, 1);
    pub const SPSR_EL1: u16 = hv(3, 0, 4, 0, 0);
    pub const ELR_EL1: u16 = hv(3, 0, 4, 0, 1);
    pub const SP_EL0: u16 = hv(3, 0, 4, 1, 0);
    pub const AFSR0_EL1: u16 = hv(3, 0, 5, 1, 0);
    pub const AFSR1_EL1: u16 = hv(3, 0, 5, 1, 1);
    pub const ESR_EL1: u16 = hv(3, 0, 5, 2, 0);
    pub const FAR_EL1: u16 = hv(3, 0, 6, 0, 0);
    pub const PAR_EL1: u16 = hv(3, 0, 7, 4, 0);
    pub const MAIR_EL1: u16 = hv(3, 0, 10, 2, 0);
    pub const AMAIR_EL1: u16 = hv(3, 0, 10, 3, 0);
    pub const VBAR_EL1: u16 = hv(3, 0, 12, 0, 0);
    pub const CONTEXTIDR_EL1: u16 = hv(3, 0, 13, 0, 1);
    pub const TPIDR_EL1: u16 = hv(3, 0, 13, 0, 4);
    pub const CNTKCTL_EL1: u16 = hv(3, 0, 14, 1, 0);
    pub const CSSELR_EL1: u16 = hv(3, 2, 0, 0, 0);
    pub const TPIDR_EL0: u16 = hv(3, 3, 13, 0, 2);
    pub const TPIDRRO_EL0: u16 = hv(3, 3, 13, 0, 3);
    pub const TPIDR2_EL0: u16 = hv(3, 3, 13, 0, 5);
    pub const CNTV_CTL_EL0: u16 = hv(3, 3, 14, 3, 1);
    pub const CNTV_CVAL_EL0: u16 = hv(3, 3, 14, 3, 2);
    pub const SP_EL1: u16 = hv(3, 4, 4, 1, 0);
    pub const VPIDR_EL2: u16 = hv(3, 4, 0, 0, 0);
    pub const VMPIDR_EL2: u16 = hv(3, 4, 0, 0, 5);
    pub const SCTLR_EL2: u16 = hv(3, 4, 1, 0, 0);
    pub const HCR_EL2: u16 = hv(3, 4, 1, 1, 0);
    pub const MDCR_EL2: u16 = hv(3, 4, 1, 1, 1);
    pub const CPTR_EL2: u16 = hv(3, 4, 1, 1, 2);
    pub const TTBR0_EL2: u16 = hv(3, 4, 2, 0, 0);
    pub const TTBR1_EL2: u16 = hv(3, 4, 2, 0, 1);
    pub const TCR_EL2: u16 = hv(3, 4, 2, 0, 2);
    pub const VTTBR_EL2: u16 = hv(3, 4, 2, 1, 0);
    pub const VTCR_EL2: u16 = hv(3, 4, 2, 1, 2);
    pub const SPSR_EL2: u16 = hv(3, 4, 4, 0, 0);
    pub const ELR_EL2: u16 = hv(3, 4, 4, 0, 1);
    pub const ESR_EL2: u16 = hv(3, 4, 5, 2, 0);
    pub const FAR_EL2: u16 = hv(3, 4, 6, 0, 0);
    pub const HPFAR_EL2: u16 = hv(3, 4, 6, 0, 4);
    pub const MAIR_EL2: u16 = hv(3, 4, 10, 2, 0);
    pub const VBAR_EL2: u16 = hv(3, 4, 12, 0, 0);
    pub const TPIDR_EL2: u16 = hv(3, 4, 13, 0, 2);
    pub const CNTVOFF_EL2: u16 = hv(3, 4, 14, 0, 3);
    pub const CNTHCTL_EL2: u16 = hv(3, 4, 14, 1, 0);
    pub const SP_EL2: u16 = hv(3, 6, 4, 1, 0);

    /// `DBGBVRn_EL1`.
    pub const fn dbgbvr(n: u16) -> u16 {
        hv(2, 0, 0, n, 4)
    }

    /// `DBGBCRn_EL1`.
    pub const fn dbgbcr(n: u16) -> u16 {
        hv(2, 0, 0, n, 5)
    }

    /// `DBGWVRn_EL1`.
    pub const fn dbgwvr(n: u16) -> u16 {
        hv(2, 0, 0, n, 6)
    }

    /// `DBGWCRn_EL1`.
    pub const fn dbgwcr(n: u16) -> u16 {
        hv(2, 0, 0, n, 7)
    }
}

/// When a synced register applies, the `#ifdef` sections of sysreg.c.inc.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Gate {
    /// Always.
    Always,
    /// Only on the way into a fresh vCPU, `SYNC_NO_RAW_REGS`: QEMU writes them at init and
    /// never reads them back.
    Init,
    /// Only when the GIC is the framework's and the guest has EL2, `SYNC_VGIC_EL2_REGS`.
    VgicEl2,
    /// Only when the guest has EL2.
    El2,
    /// Only on macOS 15.2 and later with SME, not ported yet.
    Sme,
}

/// The registers QEMU syncs between the framework and the CPU state, in sysreg.c.inc order.
pub fn synced() -> Vec<(u16, Gate)> {
    use Gate::*;
    use id::*;
    let mut v = Vec::with_capacity(128);
    for n in 0..16 {
        v.push((dbgbvr(n), Always));
        v.push((dbgbcr(n), Always));
        v.push((dbgwvr(n), Always));
        v.push((dbgwcr(n), Always));
    }
    v.extend([
        (MDCCINT_EL1, Init),
        (MIDR_EL1, Init),
        (MPIDR_EL1, Init),
        (ID_AA64PFR0_EL1, Init),
        (ID_AA64PFR1_EL1, Always),
        (ID_AA64DFR0_EL1, Always),
        (ID_AA64DFR1_EL1, Always),
        (ID_AA64ISAR0_EL1, Init),
        (ID_AA64ISAR1_EL1, Always),
        (ID_AA64MMFR1_EL1, Always),
        (ID_AA64MMFR2_EL1, Always),
        (MDSCR_EL1, Always),
        (SCTLR_EL1, Always),
        (CPACR_EL1, Always),
        (TTBR0_EL1, Always),
        (TTBR1_EL1, Always),
        (TCR_EL1, Always),
        (APIAKEYLO_EL1, Always),
        (APIAKEYHI_EL1, Always),
        (APIBKEYLO_EL1, Always),
        (APIBKEYHI_EL1, Always),
        (APDAKEYLO_EL1, Always),
        (APDAKEYHI_EL1, Always),
        (APDBKEYLO_EL1, Always),
        (APDBKEYHI_EL1, Always),
        (APGAKEYLO_EL1, Always),
        (APGAKEYHI_EL1, Always),
        (SPSR_EL1, Always),
        (ELR_EL1, Always),
        (SP_EL0, Always),
        (AFSR0_EL1, Always),
        (AFSR1_EL1, Always),
        (ESR_EL1, Always),
        (FAR_EL1, Always),
        (PAR_EL1, Always),
        (MAIR_EL1, Always),
        (AMAIR_EL1, Always),
        (VBAR_EL1, Always),
        (CONTEXTIDR_EL1, Always),
        (TPIDR_EL1, Always),
        (CNTKCTL_EL1, Always),
        (CSSELR_EL1, Always),
        (TPIDR_EL0, Always),
        (TPIDRRO_EL0, Always),
        (CNTV_CTL_EL0, Always),
        (CNTV_CVAL_EL0, Always),
        (SP_EL1, Always),
        (SMCR_EL1, Sme),
        (SMPRI_EL1, Sme),
        (TPIDR2_EL0, Sme),
        (ID_AA64ZFR0_EL1, Sme),
        (ID_AA64SMFR0_EL1, Sme),
        (CNTHCTL_EL2, VgicEl2),
        (CNTVOFF_EL2, VgicEl2),
        (CPTR_EL2, El2),
        (ELR_EL2, El2),
        (ESR_EL2, El2),
        (FAR_EL2, El2),
        (HCR_EL2, El2),
        (HPFAR_EL2, El2),
        (MAIR_EL2, El2),
        (MDCR_EL2, El2),
        (SCTLR_EL2, El2),
        (SPSR_EL2, El2),
        (SP_EL2, El2),
        (TCR_EL2, El2),
        (TPIDR_EL2, El2),
        (TTBR0_EL2, El2),
        (TTBR1_EL2, El2),
        (VBAR_EL2, El2),
        (VMPIDR_EL2, El2),
        (VPIDR_EL2, El2),
        (VTCR_EL2, El2),
        (VTTBR_EL2, El2),
    ]);
    v
}

/// The registers a running vCPU syncs, the [`synced`] list filtered the way
/// `hvf_arch_init_vcpu()` builds `hvf_sreg_match`: no init only registers, no SME, the vGIC
/// and EL2 sections only when they apply.
pub fn sync_list(vgic: bool, el2: bool) -> Vec<u16> {
    synced()
        .into_iter()
        .filter(|&(_, g)| match g {
            Gate::Always => true,
            Gate::Init | Gate::Sme => false,
            Gate::VgicEl2 => vgic && el2,
            Gate::El2 => el2,
        })
        .map(|(r, _)| r)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_match_the_sdk() {
        assert_eq!(id::MIDR_EL1, 0xc000);
        assert_eq!(id::MPIDR_EL1, 0xc005);
        assert_eq!(id::ID_AA64PFR0_EL1, 0xc020);
        assert_eq!(id::ID_AA64ISAR0_EL1, 0xc030);
        assert_eq!(id::ID_AA64MMFR0_EL1, 0xc038);
        assert_eq!(id::SCTLR_EL1, 0xc080);
        assert_eq!(id::CNTV_CTL_EL0, 0xdf19);
        assert_eq!(id::CNTV_CVAL_EL0, 0xdf1a);
        assert_eq!(id::SP_EL1, 0xe208);
        assert_eq!(id::MDCCINT_EL1, 0x8010);
        assert_eq!(id::MDSCR_EL1, 0x8012);
        assert_eq!(id::MDCR_EL2, 0xe089);
        assert_eq!(id::dbgbvr(0), 0x8004);
        assert_eq!(id::dbgwcr(15), 0x807f);
    }

    #[test]
    fn trap_reg_round_trips() {
        let r = trap::CNTP_CVAL_EL0;
        assert_eq!((r.op0(), r.op1(), r.crn(), r.crm(), r.op2()), (3, 3, 14, 2, 2));
        assert_eq!(r.hv_id(), hv(3, 3, 14, 2, 2));
        // Rt and the direction bit are not part of the register.
        assert_eq!(TrapReg::from_iss(r.0 | (7 << 5) | 1), r);
        assert_eq!(trap::MDSCR_EL1.hv_id(), id::MDSCR_EL1);
    }

    #[test]
    fn id_space_and_debug_regs() {
        assert!(TrapReg::new(3, 0, 0, 4, 0).is_id_space());
        assert!(!TrapReg::new(3, 0, 0, 0, 5).is_id_space());
        assert!(!TrapReg::new(3, 0, 0, 8, 0).is_id_space());
        assert_eq!(TrapReg::new(2, 0, 0, 9, 6).debug_reg(), Some((DebugReg::Wvr, 9)));
        assert_eq!(trap::MDSCR_EL1.debug_reg(), None);
        assert_eq!(trap::OSLAR_EL1.debug_reg(), None);
    }

    #[test]
    fn owners() {
        assert_eq!(TrapOwner::of(trap::PMCCNTR_EL0), Some(TrapOwner::Pmu));
        assert_eq!(TrapOwner::of(trap::CNTPCT_EL0), Some(TrapOwner::PhysTimer));
        assert_eq!(TrapOwner::of(trap::ICC_SRE_EL1), Some(TrapOwner::Gic));
        assert_eq!(TrapOwner::of(trap::OSLAR_EL1), None);
    }

    #[test]
    fn sync_list_sections() {
        let base = sync_list(false, false);
        assert_eq!(base.len(), 64 + 6 + 36);
        assert!(!base.contains(&id::MIDR_EL1));
        assert!(!base.contains(&id::HCR_EL2));
        assert_eq!(sync_list(false, true).len(), base.len() + 20);
        assert_eq!(sync_list(true, true).len(), base.len() + 22);
        assert_eq!(sync_list(true, false).len(), base.len());
        let mut sorted = sync_list(true, true);
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), base.len() + 22);
    }
}
