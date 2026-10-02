// SPDX-License-Identifier: GPL-2.0-or-later

//! The AArch64 TCG front end: the port of the integer, scalar FP and AdvSIMD parts of QEMU's
//! `target/arm/tcg/translate-a64.c`, `helper-a64.c` and `op_helper.c`, the AArch64 EL0 and
//! EL1 parts of `helper.c` (system registers, exception entry) and `ptw.c` (the stage 1 page
//! walk), run by `ruvm-jit`.
//!
//! [`Arm`] is the [`CpuOps`] of an AArch64 vCPU. Its state is a [`CpuArmState`] kept in the
//! runtime's `env` buffer, so generated code reaches every register by offset. The translator
//! (`translate.rs`) is driven by the decoder that `ruvm-decode` generates from QEMU's
//! `a64.decode` at build time. Helpers called from generated code are in `helpers.rs`, the
//! system register table is in `sysreg.rs` and the page walk used by `tlb_fill` and the AT
//! instructions is in `ptw.rs`.
//!
//! What is covered: every A64 base integer instruction (add and subtract with immediates,
//! shifted and extended registers and carry, logical immediates and registers, MOVZ, MOVN,
//! MOVK, ADR, ADRP, the bitfield instructions, EXTR, the shifts by register, multiply and
//! multiply-accumulate in every width, SMULH, UMULH, UDIV, SDIV, RBIT, REV, CLZ, CLS, CRC32 and
//! CRC32C, CSEL and its friends, CCMP and CCMN), every branch (B, BL, B.cond, CBZ, CBNZ, TBZ,
//! TBNZ, BR, BLR, RET), loads and stores in every addressing mode (unsigned offset, unscaled,
//! pre and post index, register offset with extension, literal, the unprivileged LDTR and STTR
//! forms, pairs including LDPSW and LDNP and STNP), the exclusives (LDXR, STXR, LDAXR, STLXR,
//! LDXP, STXP and their acquire and release forms, CLREX), LDAR and STLR, the LSE atomics (CAS,
//! CASP, LDADD, LDCLR, LDEOR, LDSET, LDSMAX, LDSMIN, LDUMAX, LDUMIN, SWP and their ordering
//! variants), LDAPR, the barriers, the hints, SVC, BRK, ERET, WFI, MSR to DAIFSet, DAIFClr,
//! SPSel, PAN and UAO, and MRS, MSR and SYS for the EL0 and EL1 system registers listed in
//! `sysreg.rs` (the ID registers, SCTLR, ACTLR, CPACR, TTBR0, TTBR1, TCR, MAIR, AMAIR, VBAR,
//! ESR, FAR, AFSR0, AFSR1, PAR, ELR, SPSR, SP_EL0, SPSel, CurrentEL, DAIF, NZCV, PAN, UAO,
//! TPIDR_EL0, TPIDRRO_EL0, TPIDR_EL1, CONTEXTIDR, the generic timer counters and timers,
//! CNTKCTL, the debug OS lock registers, cache maintenance, DC ZVA, AT and TLBI). Exceptions
//! are taken to EL1 through VBAR_EL1 with ESR_EL1, FAR_EL1, ELR_EL1 and SPSR_EL1 set as
//! `arm_cpu_do_interrupt_aarch64()` does, and IRQs from [`Arm::set_irq`] are delivered when
//! PSTATE.I is clear. The MMU does the stage 1 VMSAv8-64 walk with the 4K granule, input sizes
//! up to 48 bits, both TTBRs, TBI, hierarchical permissions, PAN, WXN, and the hardware access
//! flag and dirty state when the model has them.
//!
//! Scalar floating point and AdvSIMD (the second slice) are in `translate_simd.rs`, with the
//! helpers in `vfp.rs` (QEMU's `vfp_helper.c` and the FP parts of `helper-a64.c`),
//! `vec_helper.rs` (`vec_helper.c`, `neon_helper.c` and the AdvSIMD parts of
//! `helper-a64.c`) and `crypto.rs` (`crypto_helper.c`). Every FP operation goes through
//! `ruvm-softfloat` with the float_status QEMU derives from FPCR, so the rounding modes, FZ,
//! FZ16, DN, AHP and the cumulative FPSR flags match QEMU. Covered: every scalar FP data
//! processing, compare, conditional select, FMOV (register, general and immediate), FRINT*,
//! FCVT* with saturation, SCVTF and UCVTF (integer and fixed point), half precision when the model has FEAT_FP16, the AdvSIMD integer, widening,
//! narrowing, saturating, rounding doubling (RDM), dot product, shift, permute, EXT, table
//! lookup, across-lanes, copy, modified immediate, by-element and FP vector instructions, the
//! FP and SIMD loads and stores in every addressing mode with the structure loads and stores
//! (LD1 to LD4, ST1 to ST4, LD1R to LD4R), and AES, SHA1, SHA256 and PMULL (64 bit) as the
//! models' ID_AA64ISAR0_EL1 says. FP and SIMD accesses trap to EL1 with EC 0x07 when
//! CPACR_EL1.FPEN asks for it. SVE, SME, EL2 and EL3 are later slices.
//!
//! Deliberate differences from QEMU:
//!
//! - FP and AdvSIMD instructions call one out of line helper per instruction (per element
//!   loop) where QEMU expands TCG gvec operations inline; the results and flags are the same.
//! - FEAT_AFP (FPCR.AH, FPCR.NEP, FPCR.FIZ), FEAT_RPRES and FEAT_FPRCVT are not implemented,
//!   matching the models' ID registers; the FPRCVT forms (FCVT* to and from a SIMD register,
//!   SCVTF and UCVTF from a SIMD register of the other size) are unallocated.
//! - The FP access check looks only at CPACR_EL1.FPEN, since there is no EL2 or EL3 (CPTR_EL2
//!   and CPTR_EL3 cannot trap), and SME streaming mode does not exist.
//! - FP and SIMD data accesses are little endian only (SCTLR.EE and E0E are ignored, as for
//!   the integer loads and stores) and there are no MTE tag checks or SP alignment checks.
//! - FHM, FCMA, JSCVT (FJCVTZS), FRINTTS (FRINT32 and FRINT64), BF16, I8MM, SHA512, SHA3, SM3
//!   and SM4 are absent, as in the models: they raise UNDEF.
//! - The translator keeps no TCG globals for X0 to X30, SP, the PC, NZCV or the exclusive
//!   monitor: every access is a load from or store to `env` at the field's offset. The
//!   generated code computes the same values; it only does more memory traffic.
//! - The PC is written to `env` before every helper that can raise an exception, so those
//!   helpers raise without unwinding through the host return address.
//! - The SDIV and UDIV instructions call the `sdiv64` and `udiv64` helpers for both widths,
//!   with the 32-bit operands extended first, as QEMU does.
//! - Alignment checks come only from SCTLR.A and the instructions that always need alignment
//!   (exclusives, LDAR, STLR, the atomics). QEMU also asks for `MO_ALIGN_TLB_ONLY` on pairs
//!   and similar accesses so that Device memory faults unaligned accesses; the runtime treats
//!   that flag as a plain alignment check, so this port leaves it out, and with it the
//!   alignment fault for unaligned accesses to Device memory (including all data accesses
//!   while the MMU is off).
//! - Only the 4K translation granule is implemented, and ID_AA64MMFR0_EL1 says so. A TCR that
//!   asks for 16K or 64K pages is walked as 4K, the IMPLEMENTATION DEFINED choice for an
//!   unsupported granule. FEAT_LPA, FEAT_LPA2, FEAT_TTST and stage 2 are not implemented.
//! - The hardware access flag and dirty state updates are written back with a plain store of
//!   the descriptor, not a compare and swap, so another vCPU changing the same descriptor at
//!   the same moment can lose an update.
//! - The generic timers count host time since [`Arm`] was made, at CNTFRQ_EL0. The timer
//!   registers can be read and written, and CTL shows ISTATUS, but the timers raise no
//!   interrupt: there is no interrupt controller in this crate. IRQs come only from
//!   [`Arm::set_irq`]; FIQ, SError and virtual interrupts are not modelled.
//! - There is no AArch32, no EL2 or EL3 and no PSCI: HVC and SMC are UNDEFINED at every EL, as
//!   QEMU makes them when neither EL2 nor EL3 exists and PSCI is off. An ERET to AArch32 is an
//!   illegal return.
//! - Self-hosted debug (breakpoints, watchpoints, single step) and the PMU are not
//!   implemented. MDSCR_EL1 and the OS lock registers are only storage.
//! - WFE and YIELD are no-ops rather than leaving the execution loop. The pointer
//!   authentication hints (PACIASP and friends) are no-ops because the models have no
//!   FEAT_PAuth; QEMU does the same. BTI, MTE, FlagM, LRCPC2, CSSC, MOPS, SB, WFET and the
//!   128-bit atomics are UNDEFINED because the models do not have them.
//! - The ID registers in the ID space that this port does not model read as zero, and CCSIDR
//!   reports a fixed cache geometry. FEAT_IDST is not implemented, so unknown registers are an
//!   uncategorized UNDEF.
//! - Every write to a system register ends the translation block, which QEMU also does unless
//!   a register opts out.
//! - QEMU logs accesses to unknown system registers with `LOG_UNIMP` and illegal exception
//!   returns with `LOG_GUEST_ERROR`; this crate has no logging and stays silent.

mod crypto;
mod helpers;
mod ptw;
mod sysreg;
mod translate;
mod vec_helper;
mod vfp;

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use ruvm_jit::translate::TbBuild;
use ruvm_jit::{
    Cpu, CpuLoopExit, CpuOps, CpuShared, InterpBackend, Jit, JitConfig, MmuAccessType, Ra, Tb,
    TbCpuState, Vcpu, interrupt, translator_loop,
};
use ruvm_jit_interp::HelperRegistry;
use ruvm_mem::{AddressSpace, MemTxAttrs, MemTxResult};

use crate::cpu::{
    ArmCpuModel, CpuArmState, ENV_SIZE, EXCP_IRQ, NB_MMU_MODES, PC, PSTATE_I, PSTATE_IL,
    PSTATE_PAN, PSTATE_UAO, SCTLR_A,
};
use crate::syndrome::fsc;

/// TB flags: the current EL in bits 0 and 1.
pub const TB_EL_MASK: u32 = 3;
/// TB flags: PSTATE.IL.
pub const TB_PSTATE_IL: u32 = 1 << 2;
/// TB flags: data accesses use the PAN MMU index (EL1 with PSTATE.PAN set).
pub const TB_PAN: u32 = 1 << 3;
/// TB flags: LDTR and STTR are unprivileged (EL1 with PSTATE.UAO clear).
pub const TB_UNPRIV: u32 = 1 << 4;
/// TB flags: SCTLR.A, every access must be aligned.
pub const TB_ALIGN_MEM: u32 = 1 << 5;
/// TB flags: the shift of the two TBII bits, top byte ignore for instruction addresses.
pub const TB_TBII_SHIFT: u32 = 6;
/// TB flags: the shift of the two TBID bits, top byte ignore for data addresses.
pub const TB_TBID_SHIFT: u32 = 8;
/// TB flags: the shift of the two bits of `fp_excp_el`, the EL that FP and AdvSIMD accesses
/// trap to (0 when they do not trap).
pub const TB_FPEXC_EL_SHIFT: u32 = 10;

/// The AArch64 CPU: the [`CpuOps`] of vCPUs translated by the A64 front end.
pub struct Arm {
    model: ArmCpuModel,
    start: Instant,
}

impl fmt::Debug for Arm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Arm").field("model", &self.model.name).finish_non_exhaustive()
    }
}

impl Arm {
    /// A CPU of the given model.
    pub fn new(model: ArmCpuModel) -> Arm {
        Arm { model, start: Instant::now() }
    }

    /// A `cortex-a57`.
    pub fn cortex_a57() -> Arm {
        Arm::new(ArmCpuModel::cortex_a57())
    }

    /// A `cortex-a76`.
    pub fn cortex_a76() -> Arm {
        Arm::new(ArmCpuModel::cortex_a76())
    }

    /// The CPU model.
    pub fn model(&self) -> &ArmCpuModel {
        &self.model
    }

    /// Raise (`level` true) or lower the IRQ line of the vCPU whose shared half is `shared`,
    /// as the GIC does with `ARM_CPU_IRQ`.
    pub fn set_irq(&self, shared: &CpuShared, level: bool) {
        if level {
            shared.cpu_interrupt(interrupt::HARD);
        } else {
            shared.reset_interrupt(interrupt::HARD);
        }
    }

    /// The generic timer count at `freq` Hz: host time since the CPU was made.
    pub(crate) fn counter(&self, freq: u64) -> u64 {
        let ns = self.start.elapsed().as_nanos();
        (ns * u128::from(freq) / 1_000_000_000) as u64
    }
}

/// The [`Arm`] behind a vCPU's ops.
pub(crate) fn arm_of(ops: &Arc<dyn CpuOps>) -> &Arm {
    ops.as_any().and_then(|a| a.downcast_ref::<Arm>()).expect("the vCPU is an AArch64 vCPU")
}

/// Read a little endian `u64` from `env`.
pub(crate) fn ld64(env: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(env[off..off + 8].try_into().expect("8 bytes"))
}

/// Write a little endian `u64` to `env`.
pub(crate) fn st64(env: &mut [u8], off: usize, v: u64) {
    env[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

/// The TB flags of a state, the parts of `rebuild_hflags_a64()` this slice uses.
pub(crate) fn tb_flags(st: &CpuArmState) -> u32 {
    let el = st.current_el();
    let mut flags = el;
    if st.pstate & PSTATE_IL != 0 {
        flags |= TB_PSTATE_IL;
    }
    if el == 1 {
        if st.pstate & PSTATE_PAN != 0 {
            flags |= TB_PAN;
        }
        if st.pstate & PSTATE_UAO == 0 {
            flags |= TB_UNPRIV;
        }
    }
    if st.sctlr_el[1] & SCTLR_A != 0 {
        flags |= TB_ALIGN_MEM;
    }
    let tcr = st.tcr_el[1];
    let tbid = ((tcr >> 37) & 3) as u32;
    let tbii = tbid & !(((tcr >> 51) & 3) as u32);
    flags
        | (tbii << TB_TBII_SHIFT)
        | (tbid << TB_TBID_SHIFT)
        | (fp_exception_el(st) << TB_FPEXC_EL_SHIFT)
}

/// `fp_exception_el()` without EL2 and EL3: the EL that FP and AdvSIMD instructions trap to
/// under CPACR_EL1.FPEN, or 0 when they are enabled.
pub(crate) fn fp_exception_el(st: &CpuArmState) -> u32 {
    let el = st.current_el();
    match (st.cpacr_el1 >> 20) & 3 {
        // Trap from EL0 only.
        1 if el == 0 => 1,
        1 | 3 => 0,
        // 0 and 2: trap from EL0 and EL1.
        _ => 1,
    }
}

/// `exception_target_el()` without EL2 and EL3.
pub(crate) fn exception_target_el(st: &CpuArmState) -> u32 {
    st.current_el().max(1)
}

impl Arm {
    /// `arm_cpu_do_interrupt_aarch64()`: take the exception in `cpu.core.exception_index` to
    /// the EL in `exception_target_el`.
    fn do_interrupt_aarch64(&self, cpu: &mut Cpu<'_>) {
        let mut st = CpuArmState::load(cpu.env);
        let excp = cpu.core.exception_index;
        // Without EL2 and EL3 every exception is taken to EL1.
        let new_el: u32 = 1;
        let cur_el = st.current_el();
        let mut addr = st.vbar_el[new_el as usize];
        let old_mode = st.pstate_read();

        if cur_el < new_el {
            // Entry vector offset depends on whether the implemented EL immediately lower
            // than the target level is using AArch32 or AArch64; here it is always AArch64.
            addr += 0x400;
        } else if st.pstate & crate::cpu::PSTATE_SP != 0 {
            addr += 0x200;
        }

        match excp {
            crate::cpu::EXCP_PREFETCH_ABORT | crate::cpu::EXCP_DATA_ABORT => {
                st.far_el[new_el as usize] = st.exception_vaddress;
                st.esr_el[new_el as usize] = u64::from(st.exception_syndrome);
            }
            crate::cpu::EXCP_BKPT
            | crate::cpu::EXCP_UDEF
            | crate::cpu::EXCP_SWI
            | crate::cpu::EXCP_HVC
            | crate::cpu::EXCP_SMC => {
                st.esr_el[new_el as usize] = u64::from(st.exception_syndrome);
            }
            EXCP_IRQ => addr += 0x80,
            crate::cpu::EXCP_FIQ => addr += 0x100,
            _ => panic!("Unhandled exception 0x{excp:x}"),
        }

        st.save_sp(cur_el);
        st.elr_el[new_el as usize] = st.pc;
        st.spsr_el[new_el as usize] = u64::from(old_mode);

        let mut new_mode = (new_el << 2) | 1;
        if self.model.features.pan {
            // The value of PSTATE.PAN is normally preserved, except when an exception is
            // taken to EL1 with SCTLR_EL1.SPAN clear, which sets it.
            new_mode |= old_mode & PSTATE_PAN;
            if st.sctlr_el[new_el as usize] & crate::cpu::SCTLR_SPAN == 0 {
                new_mode |= PSTATE_PAN;
            }
        }
        st.pstate_write(crate::cpu::PSTATE_DAIF | new_mode);
        st.restore_sp(new_el);
        st.pc = addr;
        st.store(cpu.env);
        cpu.core.shared().set_interrupt(interrupt::EXITTB);
    }
}

impl CpuOps for Arm {
    fn translate_code(&self, cpu: &mut Cpu<'_>, tb: &mut TbBuild) -> Result<(), CpuLoopExit> {
        let mut dc = translate::DisasContext::new(&self.model);
        translator_loop(cpu, tb, &mut dc)
    }

    fn get_tb_cpu_state(&self, cpu: &Cpu<'_>) -> TbCpuState {
        let st = CpuArmState::load(cpu.env);
        TbCpuState { pc: st.pc, flags: tb_flags(&st), cflags: 0, cs_base: 0 }
    }

    fn restore_state_to_opc(&self, cpu: &mut Cpu<'_>, _tb: &Tb, data: &[u64; 3]) {
        st64(cpu.env, PC, data[0]);
    }

    fn set_pc(&self, cpu: &mut Cpu<'_>, pc: u64) {
        st64(cpu.env, PC, pc);
    }

    fn get_pc(&self, cpu: &Cpu<'_>) -> u64 {
        ld64(cpu.env, PC)
    }

    fn cpu_exec_interrupt(&self, cpu: &mut Cpu<'_>, interrupt_request: u32) -> bool {
        if interrupt_request & interrupt::HARD == 0 {
            return false;
        }
        let mut st = CpuArmState::load(cpu.env);
        let cur_el = st.current_el();
        let target_el = 1;
        // arm_excp_unmasked(): never to a lower EL, and only with PSTATE.I clear.
        if cur_el > target_el || st.daif & PSTATE_I != 0 {
            return false;
        }
        cpu.core.exception_index = EXCP_IRQ;
        st.exception_target_el = target_el;
        st.store(cpu.env);
        self.do_interrupt_aarch64(cpu);
        true
    }

    fn do_interrupt(&self, cpu: &mut Cpu<'_>) {
        self.do_interrupt_aarch64(cpu);
    }

    fn has_work(&self, cpu: &Cpu<'_>) -> bool {
        cpu.core.shared().interrupt_request() & (interrupt::HARD | interrupt::EXITTB) != 0
    }

    fn tlb_fill(
        &self,
        cpu: &mut Cpu<'_>,
        addr: u64,
        size: usize,
        access_type: MmuAccessType,
        mmu_idx: usize,
        probe: bool,
        ra: Ra,
    ) -> Result<bool, CpuLoopExit> {
        let _ = size;
        ptw::tlb_fill(self, cpu, addr, access_type, mmu_idx, probe, ra)
    }

    fn do_unaligned_access(
        &self,
        cpu: &mut Cpu<'_>,
        addr: u64,
        access_type: MmuAccessType,
        _mmu_idx: usize,
        ra: Ra,
    ) -> CpuLoopExit {
        let fault = ptw::Fault { fsc: fsc::ALIGNMENT, ea: false };
        ptw::deliver_fault(cpu, addr, access_type, fault, ra)
    }

    fn do_transaction_failed(
        &self,
        cpu: &mut Cpu<'_>,
        _physaddr: u64,
        addr: u64,
        _size: usize,
        access_type: MmuAccessType,
        _mmu_idx: usize,
        _attrs: MemTxAttrs,
        response: MemTxResult,
        ra: Ra,
    ) -> Result<(), CpuLoopExit> {
        // arm_cpu_do_transaction_failed(): a synchronous external abort, with EA set unless
        // it was a decode error (arm_extabort_type()).
        let fault =
            ptw::Fault { fsc: fsc::SYNC_EXTERNAL, ea: response != MemTxResult::DECODE_ERROR };
        Err(ptw::deliver_fault(cpu, addr, access_type, fault, ra))
    }

    fn mmu_index(&self, cpu: &Cpu<'_>, _ifetch: bool) -> usize {
        CpuArmState::load(cpu.env).mmu_idx()
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

/// The runtime configuration the AArch64 front end needs: 4 KiB pages and the AArch64 MMU
/// index space.
pub fn jit_config() -> JitConfig {
    JitConfig { page_bits: 12, nb_mmu_modes: NB_MMU_MODES, ..JitConfig::default() }
}

/// The runtime's built-in helpers plus every AArch64 helper.
pub fn helper_registry() -> HelperRegistry {
    let mut r = HelperRegistry::new();
    helpers::register(&mut r);
    r
}

/// An interpreter backend that knows the AArch64 helpers.
pub fn interp_backend() -> Arc<InterpBackend> {
    Arc::new(InterpBackend::with_helpers(helper_registry()))
}

/// A runtime for AArch64 guests running on the interpreter backend.
pub fn new_jit() -> Arc<Jit> {
    Jit::new(jit_config(), interp_backend())
}

/// Make a vCPU on `jit` running `ops`, with memory `as_` and registers from `state`.
pub fn create_vcpu(
    jit: &Arc<Jit>,
    ops: Arc<Arm>,
    as_: Arc<AddressSpace>,
    state: &CpuArmState,
) -> Vcpu {
    let mut v = jit.create_vcpu(ops, as_, ENV_SIZE);
    state.store(&mut v.env);
    v
}

/// The registers of `v`.
pub fn save_vcpu(v: &Vcpu) -> CpuArmState {
    CpuArmState::load(&v.env)
}

/// Whether `v` is halted in WFI.
pub fn vcpu_halted(v: &Vcpu) -> bool {
    v.shared().halted.load(Ordering::Acquire) != 0
}
