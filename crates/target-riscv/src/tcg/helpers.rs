// SPDX-License-Identifier: GPL-2.0-or-later

//! The integer and system helpers called from generated code: the parts of QEMU's
//! `target/riscv/tcg/op_helper.c`, `bitmanip_helper.c` and the XLRBR CRC helper that the
//! translator uses.
//!
//! Every helper is described by a [`Def`] with the name, flags and signature the translator
//! declares, so the declaration and the registration cannot drift apart. Helpers that touch
//! the CPU take `env` as their first argument, so their own arguments start at `args[1]`.
//! The translator records the instruction with `decode_save_opc()` before calling a helper
//! that can fault, so those helpers raise with [`Ra::Tb`] as QEMU's `GETPC()` callers do;
//! `raise_exception` runs after the PC is written and raises with [`Ra::None`].
//!
//! The hypervisor helpers (`hfence.vvma`, `hfence.gvma` and the HLV, HLVX and HSV
//! accesses) are here too. QEMU's `adjust_addr()` is the identity, as there is no pointer
//! masking.

use std::sync::atomic::Ordering;

use ruvm_jit::cputlb::{
    self, cpu_ld_code, cpu_ld_mmu, cpu_st_mmu, probe_access, probe_access_nonfault,
};
use ruvm_jit::{Cpu, CpuLoopExit, MmuAccessType, Ra, excp};
use ruvm_jit_core::types::call_flags::NO_RWG_SE;
use ruvm_jit_core::{HelperInfo, HelperType, MemOp, MemOpIdx};
use ruvm_jit_interp::{HelperEnv, HelperRegistry, Unwind};

use super::{csr, mmu_index, set_mode, swap_hypervisor_regs};
use crate::cpu::{
    BADADDR, CpuRiscvState, EXCP_ILLEGAL_INST, EXCP_INST_ACCESS_FAULT, EXCP_INST_ADDR_MIS,
    EXCP_STORE_AMO_ADDR_MIS, EXCP_VIRT_INSTRUCTION_FAULT, HSTATUS_HU, HSTATUS_SPV, HSTATUS_SPVP,
    HSTATUS_VTSR, HSTATUS_VTVM, HSTATUS_VTW, MENVCFG_CBCFE, MENVCFG_CBIE, MENVCFG_CBZE,
    MMU_2STAGE_BIT, MMU_IDX_S_SUM, MSTATUS_MIE, MSTATUS_MPIE, MSTATUS_MPP, MSTATUS_MPRV,
    MSTATUS_MPV, MSTATUS_SIE, MSTATUS_SPIE, MSTATUS_SPP, MSTATUS_SUM, MSTATUS_TSR, MSTATUS_TVM,
    MSTATUS_TW, PRV_M, PRV_S, PRV_U, RVS, RVU, RiscvCfg, get_field, set_field,
};

type R<T> = Result<T, CpuLoopExit>;

/// A helper's declaration and implementation.
pub(crate) struct Def {
    /// The name.
    pub(crate) name: &'static str,
    /// `call_flags`.
    pub(crate) flags: u32,
    /// The return type.
    pub(crate) ret: HelperType,
    /// The argument types.
    pub(crate) args: &'static [HelperType],
    pub(crate) f: ruvm_jit_interp::HelperFn,
}

impl Def {
    /// The [`HelperInfo`] the translator declares.
    pub(crate) fn info(&self) -> HelperInfo {
        HelperInfo::new(self.name, self.flags, self.ret, self.args)
    }
}

use HelperType::{I32, I64, Ptr, Void};

macro_rules! def {
    ($id:ident, $name:literal, $flags:expr, $ret:expr, [$($a:expr),*], $f:expr) => {
        pub(crate) const $id: Def =
            Def { name: $name, flags: $flags, ret: $ret, args: &[$($a),*], f: $f };
    };
}

/// Run `f` on the vCPU behind `h`, turning a guest exception into an [`Unwind`].
pub(crate) fn run(
    h: &mut HelperEnv<'_>,
    f: impl FnOnce(&mut Cpu<'_>) -> R<u64>,
) -> Result<u128, Unwind> {
    let mut cpu = Cpu::from_helper_env(h).expect("riscv helpers run under the runtime");
    match f(&mut cpu) {
        Ok(v) => Ok(u128::from(v)),
        Err(e) => Err(cpu.unwind(e)),
    }
}

/// `riscv_raise_exception(env, RISCV_EXCP_ILLEGAL_INST, GETPC())`.
fn illegal(cpu: &mut Cpu<'_>) -> CpuLoopExit {
    cpu.raise_exception(EXCP_ILLEGAL_INST, Ra::Tb)
}

/// `riscv_raise_exception(env, RISCV_EXCP_VIRT_INSTRUCTION_FAULT, GETPC())`.
fn virt_fault(cpu: &mut Cpu<'_>) -> CpuLoopExit {
    cpu.raise_exception(EXCP_VIRT_INSTRUCTION_FAULT, Ra::Tb)
}

def!(RAISE_EXCEPTION, "raise_exception", 0, Void, [Ptr, I32], h_raise_exception);
def!(CSRR, "csrr", 0, I64, [Ptr, I32], h_csrr);
def!(CSRW, "csrw", 0, Void, [Ptr, I32, I64], h_csrw);
def!(CSRRW, "csrrw", 0, I64, [Ptr, I32, I64, I64], h_csrrw);
def!(SRET, "sret", 0, I64, [Ptr], h_sret);
def!(MRET, "mret", 0, I64, [Ptr], h_mret);
def!(WFI, "wfi", 0, Void, [Ptr], h_wfi);
def!(TLB_FLUSH, "tlb_flush", 0, Void, [Ptr], h_tlb_flush);
def!(SC_PROBE_WRITE, "sc_probe_write", 0, Void, [Ptr, I64, I32], h_sc_probe_write);
def!(CBO_ZERO, "cbo_zero", 0, Void, [Ptr, I64], h_cbo_zero);
def!(CBO_CLEAN_FLUSH, "cbo_clean_flush", 0, Void, [Ptr, I64], h_cbo_clean_flush);
def!(CBO_INVAL, "cbo_inval", 0, Void, [Ptr, I64], h_cbo_inval);
def!(WRS_NTO, "wrs_nto", 0, Void, [Ptr], h_wrs_nto);
def!(HYP_TLB_FLUSH, "hyp_tlb_flush", 0, Void, [Ptr], h_hyp_tlb_flush);
def!(HYP_GVMA_TLB_FLUSH, "hyp_gvma_tlb_flush", 0, Void, [Ptr], h_hyp_gvma_tlb_flush);
def!(HYP_HLV_BU, "hyp_hlv_bu", 0, I64, [Ptr, I64], h_hyp_hlv_bu);
def!(HYP_HLV_HU, "hyp_hlv_hu", 0, I64, [Ptr, I64], h_hyp_hlv_hu);
def!(HYP_HLV_WU, "hyp_hlv_wu", 0, I64, [Ptr, I64], h_hyp_hlv_wu);
def!(HYP_HLV_D, "hyp_hlv_d", 0, I64, [Ptr, I64], h_hyp_hlv_d);
def!(HYP_HLVX_HU, "hyp_hlvx_hu", 0, I64, [Ptr, I64], h_hyp_hlvx_hu);
def!(HYP_HLVX_WU, "hyp_hlvx_wu", 0, I64, [Ptr, I64], h_hyp_hlvx_wu);
def!(HYP_HSV_B, "hyp_hsv_b", 0, Void, [Ptr, I64, I64], h_hyp_hsv_b);
def!(HYP_HSV_H, "hyp_hsv_h", 0, Void, [Ptr, I64, I64], h_hyp_hsv_h);
def!(HYP_HSV_W, "hyp_hsv_w", 0, Void, [Ptr, I64, I64], h_hyp_hsv_w);
def!(HYP_HSV_D, "hyp_hsv_d", 0, Void, [Ptr, I64, I64], h_hyp_hsv_d);
def!(CLMUL, "clmul", NO_RWG_SE, I64, [I64, I64], h_clmul);
def!(CLMULR, "clmulr", NO_RWG_SE, I64, [I64, I64], h_clmulr);
def!(CRC32, "crc32", NO_RWG_SE, I64, [I64, I32], h_crc32);
def!(CRC32C, "crc32c", NO_RWG_SE, I64, [I64, I32], h_crc32c);
def!(BREV8, "brev8", NO_RWG_SE, I64, [I64], h_brev8);
def!(XPERM4, "xperm4", NO_RWG_SE, I64, [I64, I64], h_xperm4);
def!(XPERM8, "xperm8", NO_RWG_SE, I64, [I64, I64], h_xperm8);

/// The helpers of this module.
pub(crate) const ALL: &[Def] = &[
    RAISE_EXCEPTION,
    CSRR,
    CSRW,
    CSRRW,
    SRET,
    MRET,
    WFI,
    TLB_FLUSH,
    SC_PROBE_WRITE,
    CBO_ZERO,
    CBO_CLEAN_FLUSH,
    CBO_INVAL,
    WRS_NTO,
    HYP_TLB_FLUSH,
    HYP_GVMA_TLB_FLUSH,
    HYP_HLV_BU,
    HYP_HLV_HU,
    HYP_HLV_WU,
    HYP_HLV_D,
    HYP_HLVX_HU,
    HYP_HLVX_WU,
    HYP_HSV_B,
    HYP_HSV_H,
    HYP_HSV_W,
    HYP_HSV_D,
    CLMUL,
    CLMULR,
    CRC32,
    CRC32C,
    BREV8,
    XPERM4,
    XPERM8,
];

/// Register every helper of the riscv front end.
pub(crate) fn register(r: &mut HelperRegistry) {
    for d in [
        ALL,
        super::fpu::ALL,
        super::vector::ALL,
        super::vector_int::ALL,
        super::vector_fp::ALL,
        super::vector_perm::ALL,
        super::vcrypto::ALL,
        super::crypto::ALL,
    ]
    .iter()
    .copied()
    .flatten()
    {
        r.register_info(&d.info(), d.f);
    }
}

/// `HELPER(raise_exception)`.
fn h_raise_exception(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| Err(cpu.raise_exception(a[1] as i32, Ra::None)))
}

/// `HELPER(csrr)`.
fn h_csrr(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        // seed must be accessed with a read-write instruction; csrrs and csrrc with x0,
        // and csrrsi and csrrci with 0, raise an illegal instruction exception.
        if a[1] as u32 == csr::CSR_SEED {
            return Err(illegal(cpu));
        }
        csr::csrr(cpu, a[1] as u32).map_err(|e| cpu.raise_exception(e, Ra::Tb))
    })
}

/// `HELPER(csrw)`.
fn h_csrw(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        csr::csrrw(cpu, a[1] as u32, a[2], u64::MAX)
            .map(|_| 0)
            .map_err(|e| cpu.raise_exception(e, Ra::Tb))
    })
}

/// `HELPER(csrrw)`.
fn h_csrrw(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        csr::csrrw(cpu, a[1] as u32, a[2], a[3]).map_err(|e| cpu.raise_exception(e, Ra::Tb))
    })
}

/// `HELPER(sret)`.
fn h_sret(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let mut st = CpuRiscvState::load(cpu.env);
        if st.priv_lvl < PRV_S {
            return Err(illegal(cpu));
        }
        let cfg = riscv_cfg(cpu);
        let retpc = st.sepc & cfg.xepc_mask();
        if !cfg.allow_16bit_insn() && retpc & 3 != 0 {
            return Err(cpu.raise_exception(EXCP_INST_ADDR_MIS, Ra::Tb));
        }
        if get_field(st.mstatus, MSTATUS_TSR) != 0 && st.priv_lvl < PRV_M {
            return Err(illegal(cpu));
        }
        if st.virt() && get_field(st.hstatus, HSTATUS_VTSR) != 0 {
            return Err(virt_fault(cpu));
        }
        let mut prev_virt = st.virt();
        let mut mstatus = st.mstatus;
        let prev_priv = get_field(mstatus, MSTATUS_SPP);
        mstatus = set_field(mstatus, MSTATUS_SIE, get_field(mstatus, MSTATUS_SPIE));
        mstatus = set_field(mstatus, MSTATUS_SPIE, 1);
        mstatus = set_field(mstatus, MSTATUS_SPP, PRV_U);
        mstatus = set_field(mstatus, MSTATUS_MPRV, 0);
        st.mstatus = mstatus;
        if st.has_h() && !st.virt() {
            // We support Hypervisor extensions and virtualisation is disabled.
            prev_virt = get_field(st.hstatus, HSTATUS_SPV) != 0;
            st.hstatus = set_field(st.hstatus, HSTATUS_SPV, 0);
            if prev_virt {
                swap_hypervisor_regs(&mut st);
            }
        }
        let flush = set_mode(&mut st, prev_priv, prev_virt, host_ticks(cpu));
        st.store(cpu.env);
        if flush {
            cputlb::tlb_flush(cpu);
        }
        Ok(retpc)
    })
}

/// `HELPER(mret)`.
fn h_mret(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let mut st = CpuRiscvState::load(cpu.env);
        if st.priv_lvl < PRV_M {
            return Err(illegal(cpu));
        }
        let cfg = riscv_cfg(cpu);
        let retpc = st.mepc & cfg.xepc_mask();
        let mut mstatus = st.mstatus;
        let prev_priv = get_field(mstatus, MSTATUS_MPP);
        if !cfg.allow_16bit_insn() && retpc & 3 != 0 {
            return Err(cpu.raise_exception(EXCP_INST_ADDR_MIS, Ra::Tb));
        }
        if cfg.pmp && st.pmp_num_rules == 0 && prev_priv != PRV_M {
            return Err(cpu.raise_exception(EXCP_INST_ACCESS_FAULT, Ra::Tb));
        }
        let prev_virt = get_field(mstatus, MSTATUS_MPV) != 0 && prev_priv != PRV_M;
        mstatus = set_field(mstatus, MSTATUS_MIE, get_field(mstatus, MSTATUS_MPIE));
        mstatus = set_field(mstatus, MSTATUS_MPIE, 1);
        let mpp = if st.misa & RVU != 0 { PRV_U } else { PRV_M };
        mstatus = set_field(mstatus, MSTATUS_MPP, mpp);
        mstatus = set_field(mstatus, MSTATUS_MPV, 0);
        if prev_priv != PRV_M {
            mstatus = set_field(mstatus, MSTATUS_MPRV, 0);
        }
        st.mstatus = mstatus;
        if st.has_h() && prev_virt {
            swap_hypervisor_regs(&mut st);
        }
        let flush = set_mode(&mut st, prev_priv, prev_virt, host_ticks(cpu));
        st.store(cpu.env);
        if flush {
            cputlb::tlb_flush(cpu);
        }
        Ok(retpc)
    })
}

/// `HELPER(wfi)`.
fn h_wfi(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let st = CpuRiscvState::load(cpu.env);
        let rvs = st.misa & RVS != 0;
        let prv_u = st.priv_lvl == PRV_U;
        let prv_s = st.priv_lvl == PRV_S;
        let tw = get_field(st.mstatus, MSTATUS_TW) != 0;
        if ((prv_s || (!rvs && prv_u)) && tw) || (rvs && prv_u && !st.virt()) {
            return Err(illegal(cpu));
        }
        if st.virt() && (prv_u || (prv_s && get_field(st.hstatus, HSTATUS_VTW) != 0)) {
            return Err(virt_fault(cpu));
        }
        cpu.core.exception_index = excp::HLT;
        cpu.shared().halted.store(1, Ordering::Release);
        Err(cpu.cpu_loop_exit())
    })
}

/// `HELPER(tlb_flush)`.
fn h_tlb_flush(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let st = CpuRiscvState::load(cpu.env);
        let tvm = get_field(st.mstatus, MSTATUS_TVM) != 0;
        if !st.virt() && (st.priv_lvl == PRV_U || (st.priv_lvl == PRV_S && tvm)) {
            return Err(illegal(cpu));
        }
        if st.virt() && (st.priv_lvl == PRV_U || get_field(st.hstatus, HSTATUS_VTVM) != 0) {
            return Err(virt_fault(cpu));
        }
        cputlb::tlb_flush(cpu);
        Ok(0)
    })
}

/// `helper_hyp_tlb_flush()`, for `hfence.vvma`.
fn hyp_tlb_flush(cpu: &mut Cpu<'_>) -> R<u64> {
    let st = CpuRiscvState::load(cpu.env);
    if st.virt() {
        return Err(virt_fault(cpu));
    }
    if st.priv_lvl == PRV_M || (st.priv_lvl == PRV_S && !st.virt()) {
        cputlb::tlb_flush(cpu);
        return Ok(0);
    }
    Err(illegal(cpu))
}

/// `HELPER(hyp_tlb_flush)`.
fn h_hyp_tlb_flush(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, hyp_tlb_flush)
}

/// `HELPER(hyp_gvma_tlb_flush)`.
fn h_hyp_gvma_tlb_flush(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let st = CpuRiscvState::load(cpu.env);
        if st.priv_lvl == PRV_S && !st.virt() && get_field(st.mstatus, MSTATUS_TVM) != 0 {
            return Err(illegal(cpu));
        }
        hyp_tlb_flush(cpu)
    })
}

/// `check_access_hlsv()`: the MMU index of a hypervisor load or store, which always goes
/// through both stages with the privilege level of `hstatus.SPVP`. `x` is for HLVX, which
/// ignores `vsstatus.SUM`.
fn check_access_hlsv(cpu: &mut Cpu<'_>, x: bool) -> R<usize> {
    let st = CpuRiscvState::load(cpu.env);
    if st.priv_lvl == PRV_M {
        // Always allowed.
    } else if st.virt() {
        return Err(virt_fault(cpu));
    } else if st.priv_lvl == PRV_U && get_field(st.hstatus, HSTATUS_HU) == 0 {
        return Err(illegal(cpu));
    }
    let mut mode = get_field(st.hstatus, HSTATUS_SPVP) as usize;
    if !x && mode == PRV_S as usize && get_field(st.vsstatus, MSTATUS_SUM) != 0 {
        mode = MMU_IDX_S_SUM;
    }
    Ok(mode | MMU_2STAGE_BIT)
}

/// `helper_hyp_hlv_*()`: an HLV load of `mop`, zero extended, at `addr` masked with the
/// pointer mask of the guest, `adjust_addr_virt()`.
fn hlv(h: &mut HelperEnv<'_>, addr: u64, mop: MemOp) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let idx = check_access_hlsv(cpu, false)?;
        let addr = super::pm::cpu_vm_ldst_mask(cpu).adjust(addr);
        cpu_ld_mmu(cpu, addr, MemOpIdx::new(mop, idx as u32), Ra::Tb)
    })
}

/// `helper_hyp_hlvx_*()`: an HLVX load of `n` bytes, which needs execute permission. QEMU
/// does not mask its address.
fn hlvx(h: &mut HelperEnv<'_>, addr: u64, n: usize) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let idx = check_access_hlsv(cpu, true)?;
        let mut b = [0u8; 8];
        cpu_ld_code(cpu, addr, &mut b[..n], idx, Ra::Tb)?;
        Ok(u64::from_le_bytes(b))
    })
}

/// `helper_hyp_hsv_*()`: an HSV store of `mop`, at `addr` masked as [`hlv`] does.
fn hsv(h: &mut HelperEnv<'_>, addr: u64, val: u64, mop: MemOp) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let idx = check_access_hlsv(cpu, false)?;
        let addr = super::pm::cpu_vm_ldst_mask(cpu).adjust(addr);
        cpu_st_mmu(cpu, addr, val, MemOpIdx::new(mop, idx as u32), Ra::Tb)?;
        Ok(0)
    })
}

/// `HELPER(hyp_hlv_bu)`.
fn h_hyp_hlv_bu(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    hlv(h, a[1], MemOp::UB)
}

/// `HELPER(hyp_hlv_hu)`.
fn h_hyp_hlv_hu(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    hlv(h, a[1], MemOp::LEUW)
}

/// `HELPER(hyp_hlv_wu)`.
fn h_hyp_hlv_wu(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    hlv(h, a[1], MemOp::LEUL)
}

/// `HELPER(hyp_hlv_d)`.
fn h_hyp_hlv_d(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    hlv(h, a[1], MemOp::LEUQ)
}

/// `HELPER(hyp_hlvx_hu)`.
fn h_hyp_hlvx_hu(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    hlvx(h, a[1], 2)
}

/// `HELPER(hyp_hlvx_wu)`.
fn h_hyp_hlvx_wu(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    hlvx(h, a[1], 4)
}

/// `HELPER(hyp_hsv_b)`.
fn h_hyp_hsv_b(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    hsv(h, a[1], a[2], MemOp::UB)
}

/// `HELPER(hyp_hsv_h)`.
fn h_hyp_hsv_h(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    hsv(h, a[1], a[2], MemOp::LEUW)
}

/// `HELPER(hyp_hsv_w)`.
fn h_hyp_hsv_w(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    hsv(h, a[1], a[2], MemOp::LEUL)
}

/// `HELPER(hyp_hsv_d)`.
fn h_hyp_hsv_d(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    hsv(h, a[1], a[2], MemOp::LEUQ)
}

/// The data MMU index of the vCPU, `riscv_env_mmu_index(env, false)`.
fn data_mmu_idx(cpu: &Cpu<'_>) -> usize {
    mmu_index(cpu.env, false)
}

/// `HELPER(sc_probe_write)`: a failed SC still checks that it could have stored.
fn h_sc_probe_write(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let (addr, size) = (a[1], a[2] as u32 as usize);
        if addr & (size as u64 - 1) != 0 {
            super::st64(cpu.env, BADADDR, addr);
            return Err(cpu.raise_exception(EXCP_STORE_AMO_ADDR_MIS, Ra::Tb));
        }
        let idx = data_mmu_idx(cpu);
        probe_access(cpu, addr, size, MmuAccessType::DataStore, idx, Ra::Tb)?;
        Ok(0)
    })
}

/// `cfg.cbom_blocksize` and `cfg.cboz_blocksize` of the vCPU. The properties take a power
/// of 2 from 8 to 4096 only, so a block is whole 8 byte words in one page.
fn cbo_blocksizes(cpu: &Cpu<'_>) -> (u64, u64) {
    let cfg = riscv_cfg(cpu);
    (u64::from(cfg.cbom_blocksize), u64::from(cfg.cboz_blocksize))
}

/// The configuration of the hart, `riscv_cpu_cfg()`.
fn riscv_cfg(cpu: &Cpu<'_>) -> RiscvCfg {
    let ops = cpu.ops();
    *super::riscv_of(&ops).cfg()
}

/// `cpu_get_host_ticks()`, for the per mode counts of the PMU.
fn host_ticks(cpu: &Cpu<'_>) -> u64 {
    let ops = cpu.ops();
    super::riscv_of(&ops).host_ticks()
}

/// `check_zicbo_envcfg()`.
fn check_zicbo_envcfg(cpu: &mut Cpu<'_>, envbits: u64) -> R<()> {
    let st = CpuRiscvState::load(cpu.env);
    if st.priv_lvl < PRV_M && get_field(st.menvcfg, envbits) == 0 {
        return Err(illegal(cpu));
    }
    if st.virt()
        && ((st.priv_lvl <= PRV_S && get_field(st.henvcfg, envbits) == 0)
            || (st.priv_lvl < PRV_S && get_field(st.senvcfg, envbits) == 0))
    {
        return Err(virt_fault(cpu));
    }
    if st.priv_lvl < PRV_S && get_field(st.senvcfg, envbits) == 0 {
        return Err(illegal(cpu));
    }
    Ok(())
}

/// `HELPER(cbo_zero)`.
fn h_cbo_zero(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        check_zicbo_envcfg(cpu, MENVCFG_CBZE)?;
        let cbozlen = cbo_blocksizes(cpu).1;
        // Mask off low-bits to align-down to the cache-block.
        let address = a[1] & !(cbozlen - 1);
        let idx = data_mmu_idx(cpu);
        // cbo.zero requires MMU_DATA_STORE access. Do a probe_write() to raise any
        // exceptions, including PMP.
        probe_access(cpu, address, cbozlen as usize, MmuAccessType::DataStore, idx, Ra::Tb)?;
        let oi = MemOpIdx::new(MemOp::LEUQ, idx as u32);
        for i in (0..cbozlen).step_by(8) {
            cpu_st_mmu(cpu, address + i, 0, oi, Ra::Tb)?;
        }
        Ok(0)
    })
}

/// `check_zicbom_access()`: the block must be loadable or storable.
fn check_zicbom_access(cpu: &mut Cpu<'_>, address: u64) -> R<()> {
    let cbomlen = cbo_blocksizes(cpu).0;
    // Mask off low-bits to align-down to the cache-block.
    let address = address & !(cbomlen - 1);
    let idx = data_mmu_idx(cpu);
    // A cache-block management instruction is permitted to access the specified cache
    // block whenever a load instruction or store instruction is permitted to access the
    // corresponding physical addresses.
    if probe_access_nonfault(cpu, address, MmuAccessType::DataLoad, idx, Ra::Tb)?.is_some() {
        return Ok(());
    }
    probe_access(cpu, address, cbomlen as usize, MmuAccessType::DataStore, idx, Ra::Tb)?;
    Ok(())
}

/// `HELPER(cbo_clean_flush)`.
fn h_cbo_clean_flush(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        check_zicbo_envcfg(cpu, MENVCFG_CBCFE)?;
        check_zicbom_access(cpu, a[1])?;
        // We don't emulate the cache-hierarchy, so we're done.
        Ok(0)
    })
}

/// `HELPER(cbo_inval)`.
fn h_cbo_inval(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        check_zicbo_envcfg(cpu, MENVCFG_CBIE)?;
        check_zicbom_access(cpu, a[1])?;
        // We don't emulate the cache-hierarchy, so we're done.
        Ok(0)
    })
}

/// `HELPER(wrs_nto)`.
fn h_wrs_nto(h: &mut HelperEnv<'_>, _a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let st = CpuRiscvState::load(cpu.env);
        let tw = get_field(st.mstatus, MSTATUS_TW) != 0;
        if st.virt()
            && (st.priv_lvl == PRV_S || st.priv_lvl == PRV_U)
            && get_field(st.hstatus, HSTATUS_VTW) != 0
            && !tw
        {
            return Err(virt_fault(cpu));
        }
        if st.priv_lvl != PRV_M && tw {
            return Err(illegal(cpu));
        }
        Ok(0)
    })
}

/// `HELPER(clmul)`.
fn clmul(rs1: u64, rs2: u64) -> u64 {
    (0..64).filter(|i| (rs2 >> i) & 1 != 0).fold(0, |r, i| r ^ (rs1 << i))
}

/// `HELPER(clmulr)`.
fn clmulr(rs1: u64, rs2: u64) -> u64 {
    (0..64).filter(|i| (rs2 >> i) & 1 != 0).fold(0, |r, i| r ^ (rs1 >> (63 - i)))
}

fn h_clmul(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok(u128::from(clmul(a[0], a[1])))
}

fn h_clmulr(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok(u128::from(clmulr(a[0], a[1])))
}

/// `HELPER(brev8)`: reverse the bits of every byte.
fn h_brev8(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok(u128::from(super::vcrypto::brev8(a[0])))
}

/// `do_xperm()`: look up the `1 << sz_log2` bit elements of `rs2` as indices into `rs1`,
/// an index out of range giving 0.
fn xperm(rs1: u64, rs2: u64, sz_log2: u32) -> u64 {
    let sz = 1u32 << sz_log2;
    let mask = (1u64 << sz) - 1;
    let mut r = 0;
    for i in (0..64).step_by(sz as usize) {
        let pos = ((rs2 >> i) & mask) << sz_log2;
        if pos < 64 {
            r |= ((rs1 >> pos) & mask) << i;
        }
    }
    r
}

/// `HELPER(xperm4)`.
fn h_xperm4(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok(u128::from(xperm(a[0], a[1], 2)))
}

/// `HELPER(xperm8)`.
fn h_xperm8(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok(u128::from(xperm(a[0], a[1], 3)))
}

/// The XLRBR CRC step over the low `sz` bytes of `val`: the reflected table update
/// `val = table[val & 0xff] ^ (val >> 8)` per byte, without inversion.
fn crc(poly: u64, mut val: u64, sz: u32) -> u64 {
    for _ in 0..sz * 8 {
        val = if val & 1 != 0 { (val >> 1) ^ poly } else { val >> 1 };
    }
    val
}

fn h_crc32(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok(u128::from(crc(0xedb8_8320, a[0], a[1] as u32)))
}

fn h_crc32c(_h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok(u128::from(crc(0x82f6_3b78, a[0], a[1] as u32)))
}

#[cfg(test)]
mod tests {
    use super::{clmul, clmulr, crc};

    #[test]
    fn carryless() {
        assert_eq!(clmul(3, 3), 5);
        assert_eq!(clmul(1 << 63, 2), 0);
        // clmulr is the bit reversed product of the bit reversed operands.
        let (a, b) = (0x1234_5678_9abc_def0u64, 0x0fed_cba9_8765_4321u64);
        assert_eq!(clmulr(a, b), clmul(a.reverse_bits(), b.reverse_bits()).reverse_bits());
    }

    #[test]
    fn crc_matches_the_table_form() {
        let table = |poly: u64, b: u64| crc(poly, b, 1);
        let mut v = 0xdead_beef_cafe_f00du64;
        let mut t = v;
        for _ in 0..8 {
            t = table(0xedb8_8320, t & 0xff) ^ (t >> 8);
        }
        v = crc(0xedb8_8320, v, 8);
        assert_eq!(v, t);
    }
}
