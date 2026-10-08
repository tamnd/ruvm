// SPDX-License-Identifier: GPL-2.0-or-later

//! The CPU side of the GICv3 CPU interface: the ICC system registers that QEMU's
//! `gicv3_init_cpuif()` adds to a CPU wired to a GICv3, and the EL change hook it registers.
//!
//! The registers themselves live in the interrupt controller (`ruvm-hw-intc`), which this
//! crate does not depend on; the board gives the CPU a [`GicCpuInterface`] with
//! [`Arm::with_gicv3`](super::Arm::with_gicv3) that forwards to it. Every access passes the
//! CPU state the QEMU code reads from `env` as a [`GicCpuState`]. The static access rights
//! (`PL1_RW` and so on) are checked by the translator as for any system register; the
//! `accessfn` checks (`gicv3_irqfiq_access()` and friends) are asked of the interface.
//!
//! ICC_SRE_EL1 (SRE, DFB and DIB set, 0x7) and ICC_SRE_EL2 and ICC_SRE_EL3 (Enable too,
//! 0xf) are constants, as in QEMU. A CPU with EL2 also gets the ICH registers of the
//! virtualization extension, for QEMU's default shape of 4 list registers and 5 bits of
//! virtual priority and preemption, which every CPU modelled here has. Like QEMU's, they have
//! no `accessfn`. The interface redirects ICC accesses to their ICV twins under HCR_EL2.IMO
//! and FMO. ICC_NMIAR1_EL1 is not defined.

use super::sysreg::{PL1_R, PL1_RW, PL1_W, PL2_R, PL2_RW, PL3_RW};
use super::{Arm, is_secure};
use crate::cpu::{ArmFeatures, CpuArmState};

/// The CPU state the GICv3 CPU interface looks at, what QEMU's code reads from `env` with
/// `arm_current_el()`, `arm_is_secure()` and friends.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GicCpuState {
    /// `arm_current_el()`.
    pub el: u32,
    /// `arm_feature(env, ARM_FEATURE_EL2)`.
    pub has_el2: bool,
    /// `arm_feature(env, ARM_FEATURE_EL3)`.
    pub has_el3: bool,
    /// `arm_is_secure()`.
    pub secure: bool,
    /// `arm_is_secure_below_el3()`.
    pub secure_below_el3: bool,
    /// `arm_hcr_el2_eff()`.
    pub hcr_el2: u64,
    /// SCR_EL3.
    pub scr_el3: u64,
}

impl GicCpuState {
    /// The state of a CPU with `f` in `st`.
    pub(crate) fn of(f: &ArmFeatures, st: &CpuArmState) -> GicCpuState {
        GicCpuState {
            el: st.current_el(),
            has_el2: f.el2,
            has_el3: f.el3,
            secure: is_secure(f, st),
            secure_below_el3: st.is_secure_below_el3(f),
            hcr_el2: st.hcr_el2_eff(f),
            scr_el3: st.scr_el3,
        }
    }
}

/// The result of an ICC register `accessfn`, `CPAccessResult`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GicAccess {
    /// `CP_ACCESS_OK`.
    Ok,
    /// `CP_ACCESS_TRAP_EL1`.
    TrapEl1,
    /// `CP_ACCESS_TRAP_EL2`.
    TrapEl2,
    /// `CP_ACCESS_TRAP_EL3`.
    TrapEl3,
    /// `CP_ACCESS_UNDEFINED`.
    Undefined,
}

/// An ICC register by its encoding: `(op0, op1, CRn, CRm, op2)`.
pub type IccEncoding = (u32, u32, u32, u32, u32);

/// The GICv3 CPU interfaces a board wires its CPUs to. Each call names the vCPU by its
/// index (`CPUState.cpu_index`) and is made on that vCPU's thread.
pub trait GicCpuInterface: Send + Sync {
    /// The `accessfn` of the register `reg` for an access from a CPU in `state`.
    fn access(&self, cpu: usize, reg: IccEncoding, state: &GicCpuState, isread: bool) -> GicAccess;

    /// Read the register `reg`.
    fn read(&self, cpu: usize, reg: IccEncoding, state: &GicCpuState) -> u64;

    /// Write `value` to the register `reg`.
    fn write(&self, cpu: usize, reg: IccEncoding, state: &GicCpuState, value: u64);

    /// The CPU changed exception level or was reset, `arm_gicv3_cpuif_el_change_hook()`:
    /// whether a pending interrupt is an IRQ or an FIQ depends on the EL and Security
    /// state, so the interface recomputes its outputs.
    fn state_changed(&self, cpu: usize, state: &GicCpuState);

    /// The CPU was reset: put its ICC registers back to their reset values, as the
    /// `resetfn` of QEMU's ICC register table (`icc_reset()`) does from `cpu_reset()`.
    fn reset(&self, cpu: usize) {
        let _ = cpu;
    }
}

/// An ICC register whose `accessfn` and value come from the interface.
pub(crate) struct IccReg {
    /// The name, as in QEMU's tables.
    pub(crate) name: &'static str,
    /// The encoding.
    pub(crate) enc: IccEncoding,
    /// The static access rights.
    pub(crate) access: u8,
    /// The number of preemption bits the register needs to exist.
    pub(crate) prebits: u8,
}

/// The ICC registers whose `accessfn` and value come from the interface: everything in
/// `gicv3_cpuif_reginfo` and the AP*R registers of `gicv3_init_cpuif()` but the SRE
/// constants.
pub(crate) const ICC_REGS: &[IccReg] = &[
    IccReg { name: "ICC_PMR_EL1", enc: (3, 0, 4, 6, 0), access: PL1_RW, prebits: 0 },
    IccReg { name: "ICC_IAR0_EL1", enc: (3, 0, 12, 8, 0), access: PL1_R, prebits: 0 },
    IccReg { name: "ICC_EOIR0_EL1", enc: (3, 0, 12, 8, 1), access: PL1_W, prebits: 0 },
    IccReg { name: "ICC_HPPIR0_EL1", enc: (3, 0, 12, 8, 2), access: PL1_R, prebits: 0 },
    IccReg { name: "ICC_BPR0_EL1", enc: (3, 0, 12, 8, 3), access: PL1_RW, prebits: 0 },
    IccReg { name: "ICC_AP0R0_EL1", enc: (3, 0, 12, 8, 4), access: PL1_RW, prebits: 0 },
    IccReg { name: "ICC_AP0R1_EL1", enc: (3, 0, 12, 8, 5), access: PL1_RW, prebits: 6 },
    IccReg { name: "ICC_AP0R2_EL1", enc: (3, 0, 12, 8, 6), access: PL1_RW, prebits: 7 },
    IccReg { name: "ICC_AP0R3_EL1", enc: (3, 0, 12, 8, 7), access: PL1_RW, prebits: 7 },
    IccReg { name: "ICC_AP1R0_EL1", enc: (3, 0, 12, 9, 0), access: PL1_RW, prebits: 0 },
    IccReg { name: "ICC_AP1R1_EL1", enc: (3, 0, 12, 9, 1), access: PL1_RW, prebits: 6 },
    IccReg { name: "ICC_AP1R2_EL1", enc: (3, 0, 12, 9, 2), access: PL1_RW, prebits: 7 },
    IccReg { name: "ICC_AP1R3_EL1", enc: (3, 0, 12, 9, 3), access: PL1_RW, prebits: 7 },
    IccReg { name: "ICC_DIR_EL1", enc: (3, 0, 12, 11, 1), access: PL1_W, prebits: 0 },
    IccReg { name: "ICC_RPR_EL1", enc: (3, 0, 12, 11, 3), access: PL1_R, prebits: 0 },
    IccReg { name: "ICC_SGI1R_EL1", enc: (3, 0, 12, 11, 5), access: PL1_W, prebits: 0 },
    IccReg { name: "ICC_ASGI1R_EL1", enc: (3, 1, 12, 11, 6), access: PL1_W, prebits: 0 },
    IccReg { name: "ICC_SGI0R_EL1", enc: (3, 2, 12, 11, 7), access: PL1_W, prebits: 0 },
    IccReg { name: "ICC_IAR1_EL1", enc: (3, 0, 12, 12, 0), access: PL1_R, prebits: 0 },
    IccReg { name: "ICC_EOIR1_EL1", enc: (3, 0, 12, 12, 1), access: PL1_W, prebits: 0 },
    IccReg { name: "ICC_HPPIR1_EL1", enc: (3, 0, 12, 12, 2), access: PL1_R, prebits: 0 },
    IccReg { name: "ICC_BPR1_EL1", enc: (3, 0, 12, 12, 3), access: PL1_RW, prebits: 0 },
    IccReg { name: "ICC_CTLR_EL1", enc: (3, 0, 12, 12, 4), access: PL1_RW, prebits: 0 },
    IccReg { name: "ICC_IGRPEN0_EL1", enc: (3, 0, 12, 12, 6), access: PL1_RW, prebits: 0 },
    IccReg { name: "ICC_IGRPEN1_EL1", enc: (3, 0, 12, 12, 7), access: PL1_RW, prebits: 0 },
    IccReg { name: "ICC_CTLR_EL3", enc: (3, 6, 12, 12, 4), access: PL3_RW, prebits: 0 },
    IccReg { name: "ICC_IGRPEN1_EL3", enc: (3, 6, 12, 12, 7), access: PL3_RW, prebits: 0 },
];

/// The ICH registers `gicv3_init_cpuif()` adds to a CPU with EL2: `gicv3_cpuif_hcr_reginfo`
/// and the list registers, for 4 list registers and 5 bits of virtual preemption.
pub(crate) const ICH_REGS: &[IccReg] = &[
    IccReg { name: "ICH_AP0R0_EL2", enc: (3, 4, 12, 8, 0), access: PL2_RW, prebits: 0 },
    IccReg { name: "ICH_AP1R0_EL2", enc: (3, 4, 12, 9, 0), access: PL2_RW, prebits: 0 },
    IccReg { name: "ICH_HCR_EL2", enc: (3, 4, 12, 11, 0), access: PL2_RW, prebits: 0 },
    IccReg { name: "ICH_VTR_EL2", enc: (3, 4, 12, 11, 1), access: PL2_R, prebits: 0 },
    IccReg { name: "ICH_MISR_EL2", enc: (3, 4, 12, 11, 2), access: PL2_R, prebits: 0 },
    IccReg { name: "ICH_EISR_EL2", enc: (3, 4, 12, 11, 3), access: PL2_R, prebits: 0 },
    IccReg { name: "ICH_ELRSR_EL2", enc: (3, 4, 12, 11, 5), access: PL2_R, prebits: 0 },
    IccReg { name: "ICH_VMCR_EL2", enc: (3, 4, 12, 11, 7), access: PL2_RW, prebits: 0 },
    IccReg { name: "ICH_LR0_EL2", enc: (3, 4, 12, 12, 0), access: PL2_RW, prebits: 0 },
    IccReg { name: "ICH_LR1_EL2", enc: (3, 4, 12, 12, 1), access: PL2_RW, prebits: 0 },
    IccReg { name: "ICH_LR2_EL2", enc: (3, 4, 12, 12, 2), access: PL2_RW, prebits: 0 },
    IccReg { name: "ICH_LR3_EL2", enc: (3, 4, 12, 12, 3), access: PL2_RW, prebits: 0 },
];

/// The encoding of a system register key.
pub(crate) fn encoding(key: u32) -> IccEncoding {
    (key >> 14, (key >> 11) & 7, (key >> 7) & 0xf, (key >> 3) & 0xf, key & 7)
}

impl Arm {
    /// The GICv3 CPU state of a vCPU in `st`.
    pub(crate) fn gic_state(&self, st: &CpuArmState) -> GicCpuState {
        GicCpuState::of(self.features(), st)
    }

    /// The interface register with system register key `key`, when the CPU has one.
    fn icc(&self, key: u32) -> Option<(&dyn GicCpuInterface, IccEncoding)> {
        let gic = self.gic.as_deref()?;
        let enc = encoding(key);
        let ich = self.features().el2 && ICH_REGS.iter().any(|r| r.enc == enc);
        (ich || ICC_REGS.iter().any(|r| r.enc == enc)).then_some((gic, enc))
    }

    /// Read the ICC register `key` of the vCPU `cpu_index`, or `None` if it is not one.
    pub(crate) fn gic_read(&self, cpu_index: usize, key: u32, st: &CpuArmState) -> Option<u64> {
        let (gic, enc) = self.icc(key)?;
        Some(gic.read(cpu_index, enc, &self.gic_state(st)))
    }

    /// Write the ICC register `key` of the vCPU `cpu_index`; false if it is not one.
    pub(crate) fn gic_write(&self, cpu_index: usize, key: u32, st: &CpuArmState, v: u64) -> bool {
        match self.icc(key) {
            Some((gic, enc)) => {
                gic.write(cpu_index, enc, &self.gic_state(st), v);
                true
            }
            None => false,
        }
    }

    /// The `accessfn` of the ICC register `key` for the vCPU `cpu_index`.
    pub(crate) fn gic_access(
        &self,
        cpu_index: usize,
        key: u32,
        st: &CpuArmState,
        isread: bool,
    ) -> GicAccess {
        match self.icc(key) {
            Some((gic, enc)) => gic.access(cpu_index, enc, &self.gic_state(st), isread),
            None => GicAccess::Ok,
        }
    }

    /// Tell the GIC, if there is one, that the vCPU `cpu_index` changed EL or was reset,
    /// as `arm_call_el_change_hook()` does.
    pub(crate) fn gic_el_change(&self, cpu_index: usize, st: &CpuArmState) {
        if let Some(gic) = &self.gic {
            gic.state_changed(cpu_index, &self.gic_state(st));
        }
    }

    /// `cpu_reset()` as seen by the GIC CPU interface of the vCPU `cpu_index`, now in `st`:
    /// reset its ICC registers, then recompute its outputs for the new EL.
    pub fn gic_reset(&self, cpu_index: usize, st: &CpuArmState) {
        if let Some(gic) = &self.gic {
            gic.reset(cpu_index);
            gic.state_changed(cpu_index, &self.gic_state(st));
        }
    }
}
