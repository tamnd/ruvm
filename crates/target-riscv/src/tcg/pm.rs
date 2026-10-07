// SPDX-License-Identifier: GPL-2.0-or-later

//! Pointer masking (Zjpm: Smmpm, Smnpm, Ssnpm, Sspm and Supm), a port of
//! `riscv_pm_get_pmm()`, `riscv_pm_get_vm_ldst_pmm()`, `riscv_pm_get_pmlen()` and
//! `riscv_cpu_virt_mem_enabled()` of QEMU's `target/riscv/tcg/cpu_helper.c` and of
//! `adjust_addr_body()` of `target/riscv/internals.h`.
//!
//! The PMM field of `mseccfg`, `menvcfg`, `henvcfg` or `senvcfg`, picked by the effective
//! privilege level, says how many high bits of a data address are ignored: none, 7 or 16.
//! The rest of the address is sign extended when address translation is on, and zero
//! extended when it is off. The translator masks the addresses of the scalar loads, stores,
//! atomics and cache block operations from two TB flags, as QEMU's `get_address()` does.
//! The vector helpers mask every element address, as `adjust_addr()` does, and the HLV and
//! HSV helpers their address with the PMM of the guest, as `adjust_addr_virt()` does. HLVX
//! is not masked, as in QEMU.
//!
//! Differences from QEMU:
//!
//! - The reserved PMM value 1 means no masking here. QEMU can keep it in `mseccfg` when the
//!   hart has Smepmp, and then stops on the assertion in `riscv_pm_get_pmlen()`.
//! - An access that crosses a page goes on at the next address as it is. QEMU's
//!   `riscv_pointer_wrap()` masks the address of the second page, which only matters for
//!   an access at the very top of the masked address space.

use ruvm_jit::Cpu;

use super::{TB_PM_PMM_SHIFT, TB_PM_SIGNEXTEND, ld64};
use crate::cpu::{
    HENVCFG, HSTATUS, HSTATUS_HUPMM, HSTATUS_SPVP, MENVCFG, MENVCFG_PMM, MISA, MSECCFG, MSTATUS,
    MSTATUS_MPP, MSTATUS_MPRV, MSTATUS_MPV, MSTATUS_MXR, PRIV, PRV_M, PRV_S, PRV_U, RVS, RiscvCfg,
    SATP, SATP64_MODE, SENVCFG, VIRT_ENABLED, VSATP, VSSTATUS, get_field,
};

/// `MSECCFG_PMM`: the pointer masking mode of M mode, with Smmpm.
pub(crate) const MSECCFG_PMM: u64 = 3 << 32;

/// `PMM_FIELD_RESERVED`: the PMM value no CSR write may set.
pub(crate) const PMM_FIELD_RESERVED: u64 = 1;

/// The state pointer masking depends on, read from a vCPU's `env` or from a
/// [`crate::cpu::CpuRiscvState`].
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PmState {
    pub(crate) priv_lvl: u64,
    pub(crate) virt: bool,
    pub(crate) misa: u64,
    pub(crate) mstatus: u64,
    pub(crate) vsstatus: u64,
    pub(crate) hstatus: u64,
    pub(crate) mseccfg: u64,
    pub(crate) menvcfg: u64,
    pub(crate) senvcfg: u64,
    pub(crate) henvcfg: u64,
    pub(crate) satp: u64,
    pub(crate) vsatp: u64,
}

impl PmState {
    /// The state in a vCPU's `env`.
    pub(crate) fn from_env(env: &[u8]) -> PmState {
        PmState {
            priv_lvl: ld64(env, PRIV),
            virt: ld64(env, VIRT_ENABLED) != 0,
            misa: ld64(env, MISA),
            mstatus: ld64(env, MSTATUS),
            vsstatus: ld64(env, VSSTATUS),
            hstatus: ld64(env, HSTATUS),
            mseccfg: ld64(env, MSECCFG),
            menvcfg: ld64(env, MENVCFG),
            senvcfg: ld64(env, SENVCFG),
            henvcfg: ld64(env, HENVCFG),
            satp: ld64(env, SATP),
            vsatp: ld64(env, VSATP),
        }
    }

    /// `riscv_cpu_eff_priv()`: the privilege level and virtualization mode of data
    /// accesses, which `mstatus.MPRV` changes in M mode.
    fn eff_priv(&self) -> (u64, bool) {
        if self.priv_lvl == PRV_M && self.mstatus & MSTATUS_MPRV != 0 {
            let mode = get_field(self.mstatus, MSTATUS_MPP);
            (mode, self.mstatus & MSTATUS_MPV != 0 && mode != PRV_M)
        } else {
            (self.priv_lvl, self.virt)
        }
    }
}

/// Whether the hart has any pointer masking extension, so that [`data_mask`] and
/// [`vm_ldst_mask`] can only return no masking without it.
pub(crate) fn any(cfg: &RiscvCfg) -> bool {
    cfg.ext_smmpm || cfg.ext_smnpm || cfg.ext_ssnpm
}

/// `riscv_pm_get_pmm()`: the PMM field of the effective privilege level.
fn get_pmm(cfg: &RiscvCfg, st: &PmState) -> u64 {
    let (mode, virt) = st.eff_priv();
    if (mode != PRV_M && st.mstatus & MSTATUS_MXR != 0) || (virt && st.vsstatus & MSTATUS_MXR != 0)
    {
        return 0;
    }
    match mode {
        PRV_M if cfg.ext_smmpm => get_field(st.mseccfg, MSECCFG_PMM),
        PRV_S if !virt && cfg.ext_smnpm => get_field(st.menvcfg, MENVCFG_PMM),
        PRV_S if virt && cfg.ext_ssnpm => get_field(st.henvcfg, MENVCFG_PMM),
        PRV_U if st.misa & RVS != 0 => {
            if cfg.ext_ssnpm {
                get_field(st.senvcfg, MENVCFG_PMM)
            } else {
                0
            }
        }
        PRV_U if cfg.ext_smnpm => get_field(st.menvcfg, MENVCFG_PMM),
        _ => 0,
    }
}

/// `riscv_pm_get_vm_ldst_pmm()`: the PMM field of an HLV or HSV, which accesses memory with
/// the privilege level of `hstatus.SPVP` in the guest.
fn get_vm_ldst_pmm(cfg: &RiscvCfg, st: &PmState) -> u64 {
    if !cfg.ext_ssnpm || st.mstatus & MSTATUS_MXR != 0 || st.vsstatus & MSTATUS_MXR != 0 {
        return 0;
    }
    if get_field(st.hstatus, HSTATUS_SPVP) == PRV_S {
        // The effective privilege level is VS.
        get_field(st.henvcfg, MENVCFG_PMM)
    } else if st.priv_lvl == PRV_U {
        // VU, from U mode.
        get_field(st.hstatus, HSTATUS_HUPMM)
    } else {
        get_field(st.senvcfg, MENVCFG_PMM)
    }
}

/// `riscv_cpu_virt_mem_enabled()`: whether data accesses, or with `is_vm_ldst` the accesses
/// of HLV and HSV, go through address translation.
fn virt_mem_enabled(st: &PmState, is_vm_ldst: bool) -> bool {
    let (mode, virt) =
        if is_vm_ldst { (get_field(st.hstatus, HSTATUS_SPVP), true) } else { st.eff_priv() };
    let satp = if virt { st.vsatp } else { st.satp };
    get_field(satp, SATP64_MODE) != 0 && mode != PRV_M
}

/// The masking of data addresses in effect: `pmm` picks how many high bits are ignored,
/// and the rest is sign extended if `signext` and zero extended if not.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PointerMask {
    /// The PMM field, 0 (no masking), 2 or 3; the reserved value 1 is taken as 0.
    pub(crate) pmm: u64,
    /// Whether the masked address is sign extended.
    pub(crate) signext: bool,
}

impl PointerMask {
    /// A mask from the PMM field `pmm` and whether address translation is on.
    fn new(pmm: u64, virt_mem: bool) -> PointerMask {
        let pmm = if pmm == PMM_FIELD_RESERVED { 0 } else { pmm };
        PointerMask { pmm, signext: pmm != 0 && virt_mem }
    }

    /// `riscv_pm_get_pmlen()`: the number of high bits ignored.
    pub(crate) fn pmlen(self) -> u32 {
        match self.pmm {
            2 => 7,
            3 => 16,
            _ => 0,
        }
    }

    /// `adjust_addr_body()` with the mask known.
    pub(crate) fn adjust(self, addr: u64) -> u64 {
        let pmlen = self.pmlen();
        if pmlen == 0 {
            return addr;
        }
        let addr = addr << pmlen;
        if self.signext { ((addr as i64) >> pmlen) as u64 } else { addr >> pmlen }
    }

    /// The `PM_PMM` and `PM_SIGNEXTEND` TB flags.
    pub(crate) fn tb_flags(self) -> u32 {
        ((self.pmm as u32) << TB_PM_PMM_SHIFT) | if self.signext { TB_PM_SIGNEXTEND } else { 0 }
    }

    /// The mask the TB flags `flags` hold.
    pub(crate) fn from_tb_flags(flags: u32) -> PointerMask {
        PointerMask {
            pmm: u64::from((flags >> TB_PM_PMM_SHIFT) & 3),
            signext: flags & TB_PM_SIGNEXTEND != 0,
        }
    }
}

/// The mask of ordinary data accesses, `adjust_addr()`.
pub(crate) fn data_mask(cfg: &RiscvCfg, st: &PmState) -> PointerMask {
    if !any(cfg) {
        return PointerMask::default();
    }
    PointerMask::new(get_pmm(cfg, st), virt_mem_enabled(st, false))
}

/// The mask of HLV and HSV, `adjust_addr_virt()`.
pub(crate) fn vm_ldst_mask(cfg: &RiscvCfg, st: &PmState) -> PointerMask {
    if !any(cfg) {
        return PointerMask::default();
    }
    PointerMask::new(get_vm_ldst_pmm(cfg, st), virt_mem_enabled(st, true))
}

/// [`data_mask`] of a vCPU.
pub(crate) fn cpu_data_mask(cpu: &Cpu<'_>) -> PointerMask {
    let ops = cpu.ops();
    let cfg = super::riscv_of(&ops).cfg();
    if !any(cfg) {
        return PointerMask::default();
    }
    data_mask(cfg, &PmState::from_env(cpu.env))
}

/// [`vm_ldst_mask`] of a vCPU.
pub(crate) fn cpu_vm_ldst_mask(cpu: &Cpu<'_>) -> PointerMask {
    let ops = cpu.ops();
    let cfg = super::riscv_of(&ops).cfg();
    if !any(cfg) {
        return PointerMask::default();
    }
    vm_ldst_mask(cfg, &PmState::from_env(cpu.env))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{MSTATUS_SUM, RVH};

    const SV39: u64 = 8 << 60;

    fn cfg() -> RiscvCfg {
        RiscvCfg::max()
    }

    #[test]
    fn adjust() {
        let pm = PointerMask::new(2, false);
        assert_eq!(pm.pmlen(), 7);
        assert_eq!(pm.adjust(0xff00_0000_0000_1234), 0x0100_0000_0000_1234);
        let pm = PointerMask::new(3, true);
        assert_eq!(pm.adjust(0x1234_8000_0000_0010), 0xffff_8000_0000_0010);
        assert_eq!(pm.adjust(0x1234_7000_0000_0010), 0x0000_7000_0000_0010);
        // The reserved value masks nothing.
        let pm = PointerMask::new(PMM_FIELD_RESERVED, true);
        assert_eq!(pm, PointerMask::default());
        assert_eq!(pm.adjust(0xdead_0000_0000_0000), 0xdead_0000_0000_0000);
        // The TB flags round trip.
        let pm = PointerMask::new(3, true);
        assert_eq!(PointerMask::from_tb_flags(pm.tb_flags()), pm);
    }

    #[test]
    fn pmm_by_privilege_level() {
        let cfg = cfg();
        let mut st = PmState {
            misa: RVS | RVH,
            mseccfg: 2 << 32,
            menvcfg: 3 << 32,
            senvcfg: 2 << 32,
            henvcfg: 3 << 32,
            hstatus: 2 << 48,
            satp: SV39,
            ..PmState::default()
        };
        st.priv_lvl = PRV_M;
        assert_eq!(data_mask(&cfg, &st), PointerMask { pmm: 2, signext: false });
        st.priv_lvl = PRV_S;
        assert_eq!(data_mask(&cfg, &st), PointerMask { pmm: 3, signext: true });
        st.priv_lvl = PRV_U;
        assert_eq!(data_mask(&cfg, &st), PointerMask { pmm: 2, signext: true });
        st.priv_lvl = PRV_S;
        st.virt = true;
        // VS with vsatp bare zero extends.
        assert_eq!(data_mask(&cfg, &st), PointerMask { pmm: 3, signext: false });
        // MXR turns masking off.
        st.vsstatus = MSTATUS_MXR;
        assert_eq!(data_mask(&cfg, &st), PointerMask::default());
        st.vsstatus = MSTATUS_SUM;
        // MPRV in M mode takes the level of MPP.
        st.virt = false;
        st.priv_lvl = PRV_M;
        st.mstatus = MSTATUS_MPRV | (PRV_S << 11);
        assert_eq!(data_mask(&cfg, &st), PointerMask { pmm: 3, signext: true });
        // Without S mode, U mode takes menvcfg.
        st.mstatus = 0;
        st.priv_lvl = PRV_U;
        st.misa = 0;
        assert_eq!(data_mask(&cfg, &st), PointerMask { pmm: 3, signext: true });
        // Without the extensions nothing is masked.
        let none = RiscvCfg { ext_smmpm: false, ext_smnpm: false, ext_ssnpm: false, ..cfg };
        assert_eq!(data_mask(&none, &st), PointerMask::default());
    }

    #[test]
    fn vm_ldst_pmm() {
        let cfg = cfg();
        let mut st = PmState {
            misa: RVS | RVH,
            priv_lvl: PRV_S,
            senvcfg: 2 << 32,
            henvcfg: 3 << 32,
            hstatus: (2 << 48) | HSTATUS_SPVP,
            vsatp: SV39,
            ..PmState::default()
        };
        assert_eq!(vm_ldst_mask(&cfg, &st), PointerMask { pmm: 3, signext: true });
        st.hstatus &= !HSTATUS_SPVP;
        assert_eq!(vm_ldst_mask(&cfg, &st), PointerMask { pmm: 2, signext: true });
        st.priv_lvl = PRV_U;
        assert_eq!(vm_ldst_mask(&cfg, &st), PointerMask { pmm: 2, signext: true });
        st.hstatus = 3 << 48;
        st.vsatp = 0;
        assert_eq!(vm_ldst_mask(&cfg, &st), PointerMask { pmm: 3, signext: false });
    }
}
