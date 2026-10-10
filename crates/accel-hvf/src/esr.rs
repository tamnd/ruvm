// SPDX-License-Identifier: GPL-2.0-or-later

//! Decoding the exception syndrome a vCPU exits with, the switch at the top of
//! `hvf_handle_exception()`, and the exception entry `hvf_raise_exception()` needs.
//!
//! The field layouts are the ESR_ELx ones from the Arm ARM, as target/arm/syndrome.h and the
//! `DABORT_ISS` fields in target/arm/internals.h name them.

use crate::sysreg::TrapReg;

/// `EC_UNCATEGORIZED`.
pub const EC_UNCATEGORIZED: u32 = 0x00;
/// `EC_WFX_TRAP`.
pub const EC_WFX_TRAP: u32 = 0x01;
/// `EC_AA64_HVC`.
pub const EC_AA64_HVC: u32 = 0x16;
/// `EC_AA64_SMC`.
pub const EC_AA64_SMC: u32 = 0x17;
/// `EC_SYSTEMREGISTERTRAP`.
pub const EC_SYSTEMREGISTERTRAP: u32 = 0x18;
/// `EC_INSNABORT`.
pub const EC_INSNABORT: u32 = 0x20;
/// `EC_DATAABORT`.
pub const EC_DATAABORT: u32 = 0x24;
/// `EC_BREAKPOINT`.
pub const EC_BREAKPOINT: u32 = 0x30;
/// `EC_SOFTWARESTEP`.
pub const EC_SOFTWARESTEP: u32 = 0x32;
/// `EC_WATCHPOINT`.
pub const EC_WATCHPOINT: u32 = 0x34;
/// `EC_AA64_BKPT`.
pub const EC_AA64_BKPT: u32 = 0x3c;

/// `ARM_EL_IL`: the instruction was 32 bits.
const ARM_EL_IL: u32 = 1 << 25;

/// `syn_get_ec()`.
pub fn ec(syndrome: u64) -> u32 {
    ((syndrome >> 26) & 0x3f) as u32
}

/// `syn_uncategorized()`, the syndrome of the UNDEF QEMU injects for a trap it cannot handle.
pub fn syn_uncategorized() -> u32 {
    (EC_UNCATEGORIZED << 26) | ARM_EL_IL
}

/// The `DABORT_ISS` fields of a data abort, the ones QEMU looks at.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DataAbort {
    /// ISV: the rest of the fields are valid. It is clear for pair, SIMD and SVE accesses.
    pub isv: bool,
    /// The access size in bytes, `1 << SAS`.
    pub len: u32,
    /// SSE: a load sign extends.
    pub sse: bool,
    /// SRT: the general register of the access. 31 is XZR here, never SP.
    pub srt: u32,
    /// SF: the register is 64 bits wide.
    pub sf: bool,
    /// CM: a cache maintenance operation.
    pub cm: bool,
    /// S1PTW: the fault came from a stage 1 table walk.
    pub s1ptw: bool,
    /// WnR: a write.
    pub write: bool,
}

impl DataAbort {
    /// Pulls the fields out of an `EC_DATAABORT` syndrome.
    pub fn decode(syndrome: u64) -> DataAbort {
        let s = syndrome as u32;
        DataAbort {
            isv: s & (1 << 24) != 0,
            len: 1 << ((s >> 22) & 3),
            sse: s & (1 << 21) != 0,
            srt: (s >> 16) & 0x1f,
            sf: s & (1 << 15) != 0,
            cm: s & (1 << 8) != 0,
            s1ptw: s & (1 << 7) != 0,
            write: s & (1 << 6) != 0,
        }
    }

    /// The value a load puts in the register, `sextract64()` when SSE asks for it. QEMU does
    /// not look at SF, so a sign extended 32 bit load fills all 64 bits like it does.
    pub fn load_value(&self, raw: u64) -> u64 {
        if self.sse && self.len < 8 {
            let shift = 64 - self.len * 8;
            (((raw << shift) as i64) >> shift) as u64
        } else {
            raw
        }
    }
}

/// What an exception exit asks for, `hvf_handle_exception()` cut into cases.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Exception {
    /// A data abort on the IPA, for MMIO or a dirty tracked page.
    DataAbort(DataAbort),
    /// MRS or MSR on a trapped system register: the register, the general register and the
    /// direction.
    SysReg { reg: TrapReg, rt: u32, read: bool },
    /// WFI, or WFE when `wfe` is set.
    Wfx { wfe: bool },
    /// HVC with its immediate.
    Hvc(u16),
    /// SMC with its immediate.
    Smc(u16),
    /// BRK, a hardware breakpoint, a watchpoint or a software step.
    Debug(u32),
    /// An instruction abort or anything else QEMU only reports.
    Other(u32),
}

impl Exception {
    /// Sorts a syndrome by its exception class.
    pub fn decode(syndrome: u64) -> Exception {
        let s = syndrome as u32;
        match ec(syndrome) {
            EC_DATAABORT => Exception::DataAbort(DataAbort::decode(syndrome)),
            EC_SYSTEMREGISTERTRAP => Exception::SysReg {
                reg: TrapReg::from_iss(s),
                rt: (s >> 5) & 0x1f,
                read: s & 1 != 0,
            },
            EC_WFX_TRAP => Exception::Wfx { wfe: s & 1 != 0 },
            EC_AA64_HVC => Exception::Hvc(s as u16),
            EC_AA64_SMC => Exception::Smc(s as u16),
            e @ (EC_SOFTWARESTEP | EC_AA64_BKPT | EC_BREAKPOINT | EC_WATCHPOINT) => {
                Exception::Debug(e)
            }
            e => Exception::Other(e),
        }
    }
}

/// `PSTATE.D`, `A`, `I` and `F`.
const PSTATE_DAIF: u64 = 0xf << 6;
/// `PSTATE.SP`.
const PSTATE_SP: u64 = 1;
/// `PSTATE.PAN`.
const PSTATE_PAN: u64 = 1 << 22;
/// `PSTATE.UAO`.
const PSTATE_UAO: u64 = 1 << 23;
/// `PSTATE.SSBS`.
const PSTATE_SSBS: u64 = 1 << 12;
/// `PSTATE.nRW`: AArch32.
const PSTATE_NRW: u64 = 1 << 4;
/// `PSTATE_MODE_EL1h`.
const MODE_EL1H: u64 = 0b0101;
/// `SCTLR_ELx.SPAN`.
const SCTLR_SPAN: u64 = 1 << 23;
/// `SCTLR_ELx.DSSBS`.
const SCTLR_DSSBS: u64 = 1 << 44;

/// The registers a synchronous exception to EL1 changes, worked out by [`take_to_el1`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct El1Entry {
    /// The new PC: VBAR_EL1 plus the vector offset.
    pub pc: u64,
    /// The new PSTATE.
    pub cpsr: u64,
    /// ELR_EL1: where the exception was taken.
    pub elr: u64,
    /// SPSR_EL1: the PSTATE it was taken from.
    pub spsr: u64,
    /// ESR_EL1.
    pub esr: u64,
}

/// The part of `arm_cpu_do_interrupt_aarch64()` that `hvf_raise_exception(EXCP_UDEF, ..., 1)`
/// reaches: a synchronous exception taken to EL1 from EL0 or EL1 in AArch64. The vector is
/// picked from the source EL and stack pointer, DAIF is masked, PAN follows SCTLR_EL1.SPAN,
/// UAO is cleared and SSBS is set from SCTLR_EL1.DSSBS, as the Arm ARM says for FEAT_PAN,
/// FEAT_UAO and FEAT_SSBS, which every Apple core has.
pub fn take_to_el1(pc: u64, cpsr: u64, vbar: u64, sctlr: u64, syndrome: u32) -> El1Entry {
    let from_el = (cpsr >> 2) & 3;
    let offset = if cpsr & PSTATE_NRW != 0 {
        0x600
    } else if from_el == 0 {
        0x400
    } else if cpsr & PSTATE_SP != 0 {
        0x200
    } else {
        0
    };
    let mut new = MODE_EL1H | PSTATE_DAIF;
    if sctlr & SCTLR_SPAN == 0 {
        new |= PSTATE_PAN;
    } else {
        new |= cpsr & PSTATE_PAN;
    }
    new &= !PSTATE_UAO;
    if sctlr & SCTLR_DSSBS != 0 {
        new |= PSTATE_SSBS;
    }
    El1Entry { pc: vbar + offset, cpsr: new, elr: pc, spsr: cpsr, esr: u64::from(syndrome) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dabort(iss: u32) -> u64 {
        (u64::from(EC_DATAABORT) << 26) | u64::from(ARM_EL_IL) | u64::from(iss)
    }

    #[test]
    fn mmio_store_word_from_x3() {
        // STR W3, [X0]: ISV, SAS=2, SRT=3, WnR.
        let s = dabort((1 << 24) | (2 << 22) | (3 << 16) | (1 << 6));
        let Exception::DataAbort(d) = Exception::decode(s) else { panic!() };
        assert!(d.isv && d.write && !d.sse && !d.cm && !d.s1ptw);
        assert_eq!((d.len, d.srt), (4, 3));
    }

    #[test]
    fn signed_byte_load_extends() {
        // LDRSB X1, [X0]: ISV, SAS=0, SSE, SRT=1, SF.
        let d = DataAbort::decode(dabort((1 << 24) | (1 << 21) | (1 << 16) | (1 << 15)));
        assert_eq!(d.len, 1);
        assert_eq!(d.load_value(0x80), 0xffff_ffff_ffff_ff80);
        assert_eq!(d.load_value(0x7f), 0x7f);
        let plain = DataAbort { sse: false, ..d };
        assert_eq!(plain.load_value(0x80), 0x80);
        let dword = DataAbort { len: 8, ..d };
        assert_eq!(dword.load_value(u64::MAX), u64::MAX);
    }

    #[test]
    fn cache_maintenance_is_flagged() {
        let d = DataAbort::decode(dabort((1 << 8) | (1 << 6)));
        assert!(d.cm && !d.isv);
    }

    #[test]
    fn sysreg_trap_fields() {
        // MRS X5, CNTP_CTL_EL0: op0=3 op1=3 crn=14 crm=2 op2=1, read.
        let iss: u32 = (3 << 20) | (1 << 17) | (3 << 14) | (14 << 10) | (5 << 5) | (2 << 1) | 1;
        let s = (u64::from(EC_SYSTEMREGISTERTRAP) << 26) | u64::from(iss);
        let Exception::SysReg { reg, rt, read } = Exception::decode(s) else { panic!() };
        assert_eq!(reg, TrapReg::new(3, 3, 14, 2, 1));
        assert_eq!((rt, read), (5, true));
    }

    #[test]
    fn calls_and_waits() {
        let hvc = (u64::from(EC_AA64_HVC) << 26) | 0x1234;
        assert_eq!(Exception::decode(hvc), Exception::Hvc(0x1234));
        assert_eq!(Exception::decode(u64::from(EC_AA64_SMC) << 26), Exception::Smc(0));
        assert_eq!(Exception::decode(u64::from(EC_WFX_TRAP) << 26), Exception::Wfx { wfe: false });
        assert_eq!(
            Exception::decode((u64::from(EC_WFX_TRAP) << 26) | 1),
            Exception::Wfx { wfe: true }
        );
        assert_eq!(
            Exception::decode(u64::from(EC_AA64_BKPT) << 26),
            Exception::Debug(EC_AA64_BKPT)
        );
        assert_eq!(
            Exception::decode(u64::from(EC_INSNABORT) << 26),
            Exception::Other(EC_INSNABORT)
        );
    }

    #[test]
    fn undef_from_el1h_uses_the_current_el_spx_vector() {
        let cpsr = 0x3c5; // EL1h, DAIF masked
        let e = take_to_el1(0x4000_1000, cpsr, 0x8000_0000, 1 << 23, syn_uncategorized());
        assert_eq!(e.pc, 0x8000_0200);
        assert_eq!((e.elr, e.spsr), (0x4000_1000, cpsr));
        assert_eq!(e.esr, 0x0200_0000);
        assert_eq!(e.cpsr, 0x3c5);
    }

    #[test]
    fn undef_from_el0_sets_pan_when_span_is_clear() {
        let e = take_to_el1(0x1000, 0x0, 0x8000_0000, 0, syn_uncategorized());
        assert_eq!(e.pc, 0x8000_0400);
        assert_eq!(e.cpsr, 0x3c5 | PSTATE_PAN);
        let t = take_to_el1(0x1000, 0x4, 0, SCTLR_DSSBS | SCTLR_SPAN, 0);
        assert_eq!(t.pc, 0);
        assert_eq!(t.cpsr, 0x3c5 | PSTATE_SSBS);
    }
}
