// SPDX-License-Identifier: GPL-2.0-or-later

//! The register file moved in and out of a vCPU, what `hvf_arch_get_registers()` reads and
//! `hvf_arch_put_registers()` writes, without the CPU model it lands in.

/// One vCPU's registers as the framework holds them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArmRegs {
    /// X0 to X30.
    pub x: [u64; 31],
    /// PC.
    pub pc: u64,
    /// CPSR, the guest's PSTATE in SPSR layout.
    pub cpsr: u64,
    /// FPCR.
    pub fpcr: u64,
    /// FPSR.
    pub fpsr: u64,
    /// Q0 to Q31, the low 128 bits of each SIMD and FP register.
    pub q: [u128; 32],
    /// The synced system registers as `hv_sys_reg_t` ids and values, in sync list order.
    pub sysregs: Vec<(u16, u64)>,
}

impl ArmRegs {
    /// A register file with zeroes for every register in `list`.
    pub fn with_sysregs(list: &[u16]) -> ArmRegs {
        ArmRegs { sysregs: list.iter().map(|&r| (r, 0)).collect(), ..ArmRegs::default() }
    }

    /// The value of system register `id`, if it is in the list.
    pub fn sysreg(&self, id: u16) -> Option<u64> {
        self.sysregs.iter().find(|&&(r, _)| r == id).map(|&(_, v)| v)
    }

    /// Sets system register `id`, adding it to the list when it is not there.
    pub fn set_sysreg(&mut self, id: u16, val: u64) {
        match self.sysregs.iter_mut().find(|(r, _)| *r == id) {
            Some(e) => e.1 = val,
            None => self.sysregs.push((id, val)),
        }
    }

    /// General register `rt` as an instruction names it: 31 is XZR, reading zero.
    pub fn xzr(&self, rt: u32) -> u64 {
        self.x.get(rt as usize).copied().unwrap_or(0)
    }

    /// Writes general register `rt`, dropping writes to XZR.
    pub fn set_xzr(&mut self, rt: u32, val: u64) {
        if let Some(r) = self.x.get_mut(rt as usize) {
            *r = val;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sysreg::id;

    #[test]
    fn sysreg_lookup_and_insert() {
        let mut r = ArmRegs::with_sysregs(&[id::SCTLR_EL1, id::TCR_EL1]);
        assert_eq!(r.sysreg(id::TCR_EL1), Some(0));
        r.set_sysreg(id::TCR_EL1, 5);
        r.set_sysreg(id::VBAR_EL1, 0x8000);
        assert_eq!(r.sysregs, vec![(id::SCTLR_EL1, 0), (id::TCR_EL1, 5), (id::VBAR_EL1, 0x8000)]);
        assert_eq!(r.sysreg(id::ESR_EL1), None);
    }

    #[test]
    fn register_31_is_zero() {
        let mut r = ArmRegs::default();
        r.set_xzr(30, 7);
        r.set_xzr(31, 9);
        assert_eq!((r.xzr(30), r.xzr(31)), (7, 0));
    }
}
