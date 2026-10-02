// SPDX-License-Identifier: GPL-2.0-or-later

//! Exception syndromes for ESR_ELx: the AArch64 parts of QEMU's `target/arm/syndrome.h`.

/// `ARM_EL_EC_SHIFT`.
pub const EC_SHIFT: u32 = 26;
/// `ARM_EL_IL`: the instruction length bit, set for 32-bit instructions.
pub const IL: u32 = 1 << 25;

/// `EC_UNCATEGORIZED`.
pub const EC_UNCATEGORIZED: u32 = 0x00;
/// `EC_WFX_TRAP`.
pub const EC_WFX_TRAP: u32 = 0x01;
/// `EC_ADVSIMDFPACCESSTRAP`.
pub const EC_ADVSIMDFPACCESSTRAP: u32 = 0x07;
/// `EC_ILLEGALSTATE`.
pub const EC_ILLEGALSTATE: u32 = 0x0e;
/// `EC_AA64_SVC`.
pub const EC_AA64_SVC: u32 = 0x15;
/// `EC_AA64_HVC`.
pub const EC_AA64_HVC: u32 = 0x16;
/// `EC_AA64_SMC`.
pub const EC_AA64_SMC: u32 = 0x17;
/// `EC_SYSTEMREGISTERTRAP`.
pub const EC_SYSTEMREGISTERTRAP: u32 = 0x18;
/// `EC_INSNABORT`; add one for an abort taken to the same EL.
pub const EC_INSNABORT: u32 = 0x20;
/// `EC_PCALIGNMENT`.
pub const EC_PCALIGNMENT: u32 = 0x22;
/// `EC_DATAABORT`; add one for an abort taken to the same EL.
pub const EC_DATAABORT: u32 = 0x24;
/// `EC_AA64_BKPT`.
pub const EC_AA64_BKPT: u32 = 0x3c;

/// The exception class of a syndrome, `syn_get_ec()`.
pub const fn syn_get_ec(syn: u32) -> u32 {
    syn >> EC_SHIFT
}

/// `syn_uncategorized()`: the syndrome of an UNDEFINED instruction.
pub const fn syn_uncategorized() -> u32 {
    (EC_UNCATEGORIZED << EC_SHIFT) | IL
}

/// `syn_a64_fp_access_trap()`: an AArch64 FP or SIMD access trapped by CPACR.
pub const fn syn_a64_fp_access_trap(cv: u32, cond: u32) -> u32 {
    (EC_ADVSIMDFPACCESSTRAP << EC_SHIFT) | IL | ((cv & 1) << 24) | ((cond & 0xf) << 20)
}

/// `syn_aa64_svc()`.
pub const fn syn_aa64_svc(imm16: u32) -> u32 {
    (EC_AA64_SVC << EC_SHIFT) | IL | (imm16 & 0xffff)
}

/// `syn_aa64_hvc()`.
pub const fn syn_aa64_hvc(imm16: u32) -> u32 {
    (EC_AA64_HVC << EC_SHIFT) | IL | (imm16 & 0xffff)
}

/// `syn_aa64_smc()`.
pub const fn syn_aa64_smc(imm16: u32) -> u32 {
    (EC_AA64_SMC << EC_SHIFT) | IL | (imm16 & 0xffff)
}

/// `syn_aa64_bkpt()`.
pub const fn syn_aa64_bkpt(imm16: u32) -> u32 {
    (EC_AA64_BKPT << EC_SHIFT) | IL | (imm16 & 0xffff)
}

/// `syn_aa64_sysregtrap()`.
pub const fn syn_aa64_sysregtrap(
    op0: u32,
    op1: u32,
    op2: u32,
    crn: u32,
    crm: u32,
    rt: u32,
    isread: bool,
) -> u32 {
    (EC_SYSTEMREGISTERTRAP << EC_SHIFT)
        | IL
        | ((op0 & 3) << 20)
        | ((op2 & 7) << 17)
        | ((op1 & 7) << 14)
        | ((crn & 0xf) << 10)
        | ((rt & 0x1f) << 5)
        | ((crm & 0xf) << 1)
        | isread as u32
}

/// `syn_insn_abort()`.
pub const fn syn_insn_abort(same_el: bool, ea: bool, s1ptw: bool, fsc: u32) -> u32 {
    ((EC_INSNABORT + same_el as u32) << EC_SHIFT)
        | IL
        | ((ea as u32) << 9)
        | ((s1ptw as u32) << 7)
        | (fsc & 0x3f)
}

/// `syn_data_abort_no_iss()` with FnV and CM clear.
pub const fn syn_data_abort_no_iss(
    same_el: bool,
    ea: bool,
    s1ptw: bool,
    wnr: bool,
    fsc: u32,
) -> u32 {
    ((EC_DATAABORT + same_el as u32) << EC_SHIFT)
        | IL
        | ((ea as u32) << 9)
        | ((s1ptw as u32) << 7)
        | ((wnr as u32) << 6)
        | (fsc & 0x3f)
}

/// `syn_wfx()` for a 32-bit instruction with no register operand.
pub const fn syn_wfx(cv: u32, cond: u32, ti: u32) -> u32 {
    (EC_WFX_TRAP << EC_SHIFT) | IL | ((cv & 1) << 24) | ((cond & 0xf) << 20) | (ti & 3)
}

/// `syn_illegalstate()`.
pub const fn syn_illegalstate() -> u32 {
    (EC_ILLEGALSTATE << EC_SHIFT) | IL
}

/// `syn_pcalignment()`.
pub const fn syn_pcalignment() -> u32 {
    (EC_PCALIGNMENT << EC_SHIFT) | IL
}

/// Fault status codes in the long descriptor format (`ARMFaultType` as reported by
/// `arm_fi_to_lfsc()`).
pub mod fsc {
    /// Address size fault at `level`.
    pub const fn address_size(level: u32) -> u32 {
        level
    }
    /// Translation fault at `level`.
    pub const fn translation(level: u32) -> u32 {
        0x04 | level
    }
    /// Access flag fault at `level`.
    pub const fn access_flag(level: u32) -> u32 {
        0x08 | level
    }
    /// Permission fault at `level`.
    pub const fn permission(level: u32) -> u32 {
        0x0c | level
    }
    /// Synchronous external abort, not on a walk.
    pub const SYNC_EXTERNAL: u32 = 0x10;
    /// Synchronous external abort on the walk at `level`.
    pub const fn sync_external_on_walk(level: u32) -> u32 {
        0x14 | level
    }
    /// Alignment fault.
    pub const ALIGNMENT: u32 = 0x21;
}
