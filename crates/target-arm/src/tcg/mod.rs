// SPDX-License-Identifier: GPL-2.0-or-later

//! The AArch64 TCG front end: the port of the integer, scalar FP and AdvSIMD parts of QEMU's
//! `target/arm/tcg/translate-a64.c`, `helper-a64.c` and `op_helper.c`, the AArch64 parts of
//! `helper.c` (system registers, exception entry and routing, the generic timers),
//! `tlb_helper.c` and `ptw.c` (both translation stages), and `psci.c`, run by `ruvm-jit`.
//!
//! [`Arm`] is the [`CpuOps`] of an AArch64 vCPU. Its state is a [`CpuArmState`] kept in the
//! runtime's `env` buffer, so generated code reaches every register by offset. The translator
//! (`translate.rs`) is driven by the decoder that `ruvm-decode` generates from QEMU's
//! `a64.decode` at build time. Helpers called from generated code are in `helpers.rs`, the
//! system register table is in `sysreg.rs`, the page walk used by `tlb_fill` and the AT
//! instructions is in `ptw.rs`, the generic timers are in `gtimer.rs` and the emulated PSCI
//! firmware is in `psci.rs`.
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
//! CNTKCTL, the debug OS lock registers, cache maintenance, DC ZVA, AT and TLBI), plus the EL2
//! and EL3 banks (HCR, SCR, the EL2 and EL3 copies of SCTLR, TCR, TTBR, MAIR, VBAR, ESR, FAR,
//! ELR, SPSR and TPIDR, CPTR, HPFAR, VTCR, VTTBR, VPIDR, VMPIDR, MDCR, HSTR, CNTHCTL, CNTVOFF,
//! the EL2 and Secure physical timers) with their HCR_EL2 and SCR_EL3 traps, the VHE EL12 and
//! EL02 aliases and the E2H redirection of the EL1 names to their EL2 registers, resolved at
//! translation time. Exceptions are routed to EL1, EL2 or EL3 as `arm_phys_excp_target_el()`
//! and `raise_exception()` route them (including HCR_EL2.TGE, IMO, FMO and AMO and SCR_EL3.IRQ,
//! FIQ and EA) and taken through VBAR_ELx with ESR, FAR, ELR and SPSR set as
//! `arm_cpu_do_interrupt_aarch64()` does; ERET returns to any enabled EL with QEMU's illegal
//! return checks. HVC and SMC follow `pre_hvc` and `pre_smc` (SCR_EL3.HCE and SMD,
//! HCR_EL2.HCD and TSC), and when the board asks for it with [`Arm::with_psci`] a PSCI call on
//! the conduit is handled by `psci.rs`. IRQ and FIQ come from [`Arm::set_irq`] and
//! [`Arm::set_fiq`], virtual IRQ, FIQ and SError from HCR_EL2 and the virtual lines. The MMU
//! walks the EL1&0, EL2&0, EL2 and EL3 regimes with the 4K, 16K and 64K granules the model
//! has, input sizes up to 48 bits, TBI, hierarchical permissions, PAN, WXN, the hardware
//! access flag and dirty state when the model has them, and stage 2 from VTCR_EL2 and
//! VTTBR_EL2 with faults reported in ESR_EL2 and HPFAR_EL2. TLBI, including the Inner
//! Shareable broadcast forms, uses the `ruvm-jit` flush family. The generic timers drive their
//! outputs through [`ArmBoard::gt_timer_update`].
//!
//! Scalar floating point and AdvSIMD (the second slice) are in `translate_simd.rs`, with the
//! helpers in `vfp.rs` (QEMU's `vfp_helper.c` and the FP parts of `helper-a64.c`), `vec_helper.rs`
//! (`vec_helper.c`, `neon_helper.c` and the AdvSIMD parts of `helper-a64.c`) and `crypto.rs`
//! (`crypto_helper.c`). Every FP operation goes through `ruvm-softfloat` with the float_status QEMU
//! derives from FPCR, so the rounding modes, FZ, FZ16, DN, AHP and the cumulative FPSR flags match
//! QEMU. Covered: every scalar FP data processing, compare, conditional select, FMOV (register,
//! general and immediate), FRINT*, FCVT* with saturation, SCVTF and UCVTF (integer and fixed
//! point), half precision when the model has FEAT_FP16, the AdvSIMD integer, widening, narrowing,
//! saturating, rounding doubling (RDM), dot product, shift, permute, EXT, table lookup,
//! across-lanes, copy, modified immediate, by-element and FP vector instructions, the FP and SIMD
//! loads and stores in every addressing mode with the structure loads and stores (LD1 to LD4, ST1
//! to ST4, LD1R to LD4R), and AES, SHA1, SHA256 and PMULL (64 bit) as the models' ID_AA64ISAR0_EL1
//! says. FP and SIMD accesses trap with EC 0x07 as CPACR_EL1.FPEN, CPTR_EL2 (in either format) and
//! CPTR_EL3.TFP ask, as `fp_exception_el()` does.
//!
//! SVE and SVE2 (`translate_sve.rs`, `sve_helper.rs`) cover the integer, predicate, permute,
//! element count and memory groups, including first fault and non fault loads, gathers and
//! scatters. The vector length comes from ZCR_EL1 to ZCR_EL3 and the model's `sve-max-vq`
//! (`sve_vqm1_for_el()`) and is a TB flag, as is the EL SVE instructions trap to under
//! CPACR_EL1.ZEN, CPTR_EL2 and CPTR_EL3 (`sve_exception_el()`). A ZCR_ELx write or an EL
//! change that shortens the vector length zeroes the bits above it. SME is a later slice.
//!
//! Deliberate differences from QEMU:
//!
//! - FP and AdvSIMD instructions call one out of line helper per instruction (per element
//!   loop) where QEMU expands TCG gvec operations inline; the results and flags are the same.
//! - FEAT_AFP (FPCR.AH, FPCR.NEP, FPCR.FIZ), FEAT_RPRES and FEAT_FPRCVT are not implemented,
//!   matching the models' ID registers; the FPRCVT forms (FCVT* to and from a SIMD register,
//!   SCVTF and UCVTF from a SIMD register of the other size) are unallocated.
//! - SME streaming mode does not exist, so the FP access check has no streaming case.
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
//! - FEAT_LPA, FEAT_LPA2, FEAT_LVA and FEAT_TTST (52 bit addresses) are not implemented; see
//!   `ptw.rs` for the other page walk differences.
//! - The hardware access flag and dirty state updates are written back with a plain store of
//!   the descriptor, not a compare and swap, so another vCPU changing the same descriptor at
//!   the same moment can lose an update.
//! - The generic timers count host time since [`Arm`] was made, at CNTFRQ_EL0; there is no
//!   interrupt controller in this crate, so their outputs go to the board, which wires them
//!   to its GIC. See `gtimer.rs` and `psci.rs` for their own differences.
//! - The models start without EL2 and EL3 ([`ArmCpuModel::with_el2`] and
//!   [`ArmCpuModel::with_el3`] add them), where QEMU's models have them and the virt board
//!   clears them unless `virtualization` or `secure` is on. The result is the same CPU.
//! - There is no AArch32 at any EL, no Secure EL2 (FEAT_SEL2) and no FEAT_NV: an ERET to
//!   AArch32 is an illegal return, and HCR_EL2.NV and NV1 are only storage. Secure and
//!   Non-secure accesses use one address space and share the TLB, so a write that flips
//!   SCR_EL3.NS flushes everything rather than switching TLB banks.
//! - The TLB is not tagged by VMID or ASID: TLBI by ASID or VMID and writes that change them
//!   flush every entry of the affected regimes, and the IPAS2 forms flush nothing, as QEMU
//!   does, since stage 2 results are only cached combined with stage 1. The traps that
//!   MDCR_EL2, MDCR_EL3, HSTR_EL2 and HCR_EL2.TIDCP control are not checked; see `sysreg.rs`.
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
mod gtimer;
mod helpers;
mod psci;
mod ptw;
mod sve_helper;
mod sysreg;
mod translate;
mod vec_helper;
mod vfp;

use std::fmt;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use ruvm_jit::translate::TbBuild;
use ruvm_jit::{
    Cpu, CpuLoopExit, CpuOps, CpuShared, InterpBackend, Jit, JitConfig, MmuAccessType, Ra, Tb,
    TbCpuState, Vcpu, interrupt, translator_loop,
};
use ruvm_jit_interp::HelperRegistry;
use ruvm_mem::{AddressSpace, MemTxAttrs, MemTxResult};

use crate::cpu::{
    ArmCpuModel, ArmFeatures, CpuArmState, ENV_SIZE, EXCP_BKPT, EXCP_DATA_ABORT, EXCP_FIQ,
    EXCP_HVC, EXCP_HYP_TRAP, EXCP_IRQ, EXCP_PREFETCH_ABORT, EXCP_SMC, EXCP_SWI, EXCP_UDEF,
    EXCP_VFIQ, EXCP_VIRQ, EXCP_VSERR, HCR_AMO, HCR_E2H, HCR_FMO, HCR_IMO, HCR_TGE, HCR_VF, HCR_VI,
    HCR_VSE, MMU_IDX_E2, MMU_IDX_E3, MMU_IDX_E10_1, MMU_IDX_E10_1_PAN, MMU_IDX_E20_2,
    MMU_IDX_E20_2_PAN, NB_MMU_MODES, PC, PSTATE_A, PSTATE_DAIF, PSTATE_F, PSTATE_I, PSTATE_IL,
    PSTATE_PAN, PSTATE_SP, PSTATE_UAO, SCR_EA, SCR_FIQ, SCR_IRQ, SCTLR_A, SCTLR_SPAN,
};
use crate::syndrome::{EC_ADVSIMDFPACCESSTRAP, fsc, syn_get_ec, syn_serror};

pub use gtimer::GTIMER_NAMES;
pub use psci::{
    PSCI_OFF, PSCI_ON, PSCI_ON_PENDING, PSCI_RET_ALREADY_ON, PSCI_RET_DENIED,
    PSCI_RET_INTERNAL_FAILURE, PSCI_RET_INVALID_PARAMS, PSCI_RET_NOT_SUPPORTED,
    PSCI_RET_ON_PENDING, PSCI_RET_SUCCESS,
};

/// TB flags: the current EL in bits 0 and 1.
pub const TB_EL_MASK: u32 = 3;
/// TB flags: PSTATE.IL.
pub const TB_PSTATE_IL: u32 = 1 << 2;
/// TB flags: HCR_EL2.E2H is in effect (`TBFLAG_A64.E2H`).
pub const TB_E2H: u32 = 1 << 3;
/// TB flags: LDTR and STTR are unprivileged (PSTATE.UAO clear at EL1, or at EL2 in the
/// EL2&0 regime with TGE set).
pub const TB_UNPRIV: u32 = 1 << 4;
/// TB flags: SCTLR.A of the current regime, every access must be aligned.
pub const TB_ALIGN_MEM: u32 = 1 << 5;
/// TB flags: the shift of the two TBII bits, top byte ignore for instruction addresses.
pub const TB_TBII_SHIFT: u32 = 6;
/// TB flags: the shift of the two TBID bits, top byte ignore for data addresses.
pub const TB_TBID_SHIFT: u32 = 8;
/// TB flags: the shift of the two bits of `fp_excp_el`, the EL that FP and AdvSIMD accesses
/// trap to (0 when they do not trap).
pub const TB_FPEXC_EL_SHIFT: u32 = 10;
/// TB flags: the shift of the four bits of the core MMU index (`TBFLAG_ANY.MMUIDX`).
pub const TB_MMUIDX_SHIFT: u32 = 12;
/// Where the SVE exception EL (`SVEEXC_EL`, 2 bits) sits in the TB flags.
pub const TB_SVEEXC_EL_SHIFT: u32 = 16;
/// Where the SVE vector length in quadwords minus one (`VL`, 4 bits) sits in the TB flags.
pub const TB_VL_SHIFT: u32 = 18;

/// `CPU_INTERRUPT_FIQ`.
pub const INTERRUPT_FIQ: u32 = 0x0010;
/// `CPU_INTERRUPT_VIRQ`.
pub const INTERRUPT_VIRQ: u32 = 0x0040;
/// `CPU_INTERRUPT_VFIQ`.
pub const INTERRUPT_VFIQ: u32 = 0x0200;
/// `CPU_INTERRUPT_VSERR`.
pub const INTERRUPT_VSERR: u32 = interrupt::TGT_INT_0;

/// Which instruction calls into the emulated PSCI firmware, QEMU's `psci-conduit` property.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PsciConduit {
    /// `QEMU_PSCI_CONDUIT_DISABLED`: HVC and SMC behave as the architecture says.
    #[default]
    Disabled,
    /// `QEMU_PSCI_CONDUIT_HVC`.
    Hvc,
    /// `QEMU_PSCI_CONDUIT_SMC`.
    Smc,
}

/// The board side of an AArch64 CPU: where the generic timer outputs go and what the PSCI
/// calls that need the rest of the machine do. Every method has a default that does
/// nothing, so a board implements only what it wires up.
pub trait ArmBoard: Send + Sync {
    /// The output of generic timer `timer` (a `GTIMER_*` index) of the vCPU `shared` is now
    /// `level`, as `gt_update_irq()` drives `gt_timer_outputs[]`. `deadline` is when the
    /// timer next needs recalculating, as QEMU arms `gt_timer[]`; the board calls
    /// [`Arm::gt_timer_expired`] then. `None` means the timer is off or will not change.
    fn gt_timer_update(
        &self,
        shared: &CpuShared,
        timer: usize,
        level: bool,
        deadline: Option<Instant>,
    ) {
        let _ = (shared, timer, level, deadline);
    }

    /// PSCI CPU_ON, `arm_set_cpu_on()`: start the CPU whose MPIDR is `mpidr` at `entry` in
    /// `target_el` with `context_id` in X0. A board finds the vCPU and calls
    /// [`Arm::cpu_on`], returning its result. The default says no such CPU.
    fn psci_cpu_on(&self, mpidr: u64, entry: u64, context_id: u64, target_el: u32) -> i64 {
        let _ = (mpidr, entry, context_id, target_el);
        PSCI_RET_INVALID_PARAMS
    }

    /// The PSCI power state (`PSCI_ON`, `PSCI_OFF` or `PSCI_ON_PENDING`) of the CPU whose
    /// MPIDR is `mpidr`, or `None` if there is none, for AFFINITY_INFO. A board usually
    /// answers with [`Arm::power_state`].
    fn psci_power_state(&self, mpidr: u64) -> Option<u32> {
        let _ = mpidr;
        None
    }

    /// PSCI SYSTEM_OFF: `qemu_system_shutdown_request(SHUTDOWN_CAUSE_GUEST_SHUTDOWN)`.
    fn psci_system_off(&self) {}

    /// PSCI SYSTEM_RESET: `qemu_system_reset_request(SHUTDOWN_CAUSE_GUEST_RESET)`.
    fn psci_system_reset(&self) {}
}

/// The per vCPU state that lives outside `env` because other threads change it: the
/// virtual interrupt lines from the interrupt controller (`irq_line_state`), the PSCI power
/// state and the MPIDR the board gave the vCPU.
#[derive(Clone, Copy, Debug, Default)]
struct CpuLines {
    /// `INTERRUPT_VIRQ` and `INTERRUPT_VFIQ` as the GIC drives them.
    gic: u32,
    /// `power_state`.
    power: u32,
    /// `mp_affinity`, when the board set it.
    mpidr: Option<u64>,
}

/// The AArch64 CPU: the [`CpuOps`] of vCPUs translated by the A64 front end.
pub struct Arm {
    model: ArmCpuModel,
    start: Instant,
    psci_conduit: PsciConduit,
    board: Option<Arc<dyn ArmBoard>>,
    lines: Mutex<Vec<CpuLines>>,
}

impl fmt::Debug for Arm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Arm")
            .field("model", &self.model.name)
            .field("psci_conduit", &self.psci_conduit)
            .finish_non_exhaustive()
    }
}

impl Arm {
    /// A CPU of the given model.
    pub fn new(model: ArmCpuModel) -> Arm {
        Arm {
            model,
            start: Instant::now(),
            psci_conduit: PsciConduit::Disabled,
            board: None,
            lines: Mutex::new(Vec::new()),
        }
    }

    /// A `cortex-a57`.
    pub fn cortex_a57() -> Arm {
        Arm::new(ArmCpuModel::cortex_a57())
    }

    /// A `cortex-a72`.
    pub fn cortex_a72() -> Arm {
        Arm::new(ArmCpuModel::cortex_a72())
    }

    /// A `cortex-a76`.
    pub fn cortex_a76() -> Arm {
        Arm::new(ArmCpuModel::cortex_a76())
    }

    /// The same CPU with HVC or SMC calling the emulated PSCI firmware, as the virt board
    /// sets `psci-conduit` when it does not run guest firmware at EL3 (or EL2).
    pub fn with_psci(mut self, conduit: PsciConduit) -> Arm {
        self.psci_conduit = conduit;
        self
    }

    /// The same CPU wired to `board`.
    pub fn with_board(mut self, board: Arc<dyn ArmBoard>) -> Arm {
        self.board = Some(board);
        self
    }

    /// The CPU model.
    pub fn model(&self) -> &ArmCpuModel {
        &self.model
    }

    /// The model's features.
    pub(crate) fn features(&self) -> &ArmFeatures {
        &self.model.features
    }

    /// The PSCI conduit.
    pub fn psci_conduit(&self) -> PsciConduit {
        self.psci_conduit
    }

    fn lines(&self, cpu_index: usize) -> (MutexGuard<'_, Vec<CpuLines>>, usize) {
        let mut g = self.lines.lock().unwrap_or_else(|e| e.into_inner());
        if g.len() <= cpu_index {
            g.resize(cpu_index + 1, CpuLines::default());
        }
        (g, cpu_index)
    }

    /// Set the MPIDR_EL1 affinity value of the vCPU with index `cpu_index`, the
    /// `mp-affinity` property. Without it a vCPU reports QEMU's default
    /// `arm_build_mp_affinity()` layout of eight CPUs per cluster.
    pub fn set_mpidr(&self, cpu_index: usize, mpidr: u64) {
        let (mut g, i) = self.lines(cpu_index);
        g[i].mpidr = Some(mpidr & 0xff_00ff_ffff);
    }

    /// `arm_cpu_mp_affinity()` of the vCPU with index `cpu_index`.
    pub fn mp_affinity(&self, cpu_index: usize) -> u64 {
        let (g, i) = self.lines(cpu_index);
        g[i].mpidr.unwrap_or(((cpu_index as u64 / 8) << 8) | (cpu_index as u64 % 8))
    }

    /// The PSCI power state of the vCPU with index `cpu_index`.
    pub fn power_state(&self, cpu_index: usize) -> u32 {
        let (g, i) = self.lines(cpu_index);
        g[i].power
    }

    /// Mark the vCPU `shared` as powered off, the `start-powered-off` property: it stays
    /// halted until a PSCI CPU_ON (through [`Arm::cpu_on`]) starts it.
    pub fn set_powered_off(&self, shared: &CpuShared) {
        let (mut g, i) = self.lines(shared.cpu_index);
        g[i].power = PSCI_OFF;
        shared.halted.store(1, Ordering::Release);
    }

    /// `arm_set_cpu_on()` for the vCPU `shared`: check its power state, then reset it into
    /// `target_el` at `entry` with `context_id` in X0, as `arm_set_cpu_on_async_work()`
    /// does on the vCPU's thread. Returns a PSCI return code.
    pub fn cpu_on(&self, shared: &CpuShared, entry: u64, context_id: u64, target_el: u32) -> i64 {
        assert!((1..4).contains(&target_el), "requested EL must be in the 1 to 3 range");
        if entry & 3 != 0 {
            // If we are booting in AArch64 mode then "entry" needs to be 4 bytes aligned.
            return PSCI_RET_INVALID_PARAMS;
        }
        let f = self.features();
        if (target_el == 3 && !f.el3) || (target_el == 2 && !f.el2) {
            // The CPU does not support requested level.
            return PSCI_RET_INVALID_PARAMS;
        }
        {
            let (mut g, i) = self.lines(shared.cpu_index);
            match g[i].power {
                PSCI_ON => return PSCI_RET_ALREADY_ON,
                PSCI_ON_PENDING => return PSCI_RET_ON_PENDING,
                _ => g[i].power = PSCI_ON_PENDING,
            }
        }
        shared.async_run_on_cpu(move |cpu| {
            let ops = cpu.ops();
            let arm = arm_of(&ops);
            let mut st = CpuArmState::reset(arm.model());
            let index = cpu.core.shared().cpu_index;
            st.vmpidr_el2 = (1 << 31) | arm.mp_affinity(index);
            st.emulate_firmware_reset(arm.features(), target_el);
            st.pc = entry;
            st.xregs[0] = context_id;
            st.store(cpu.env);
            ruvm_jit::cputlb::tlb_flush(cpu);
            {
                let (mut g, i) = arm.lines(index);
                g[i].power = PSCI_ON;
            }
            let shared = cpu.core.shared();
            shared.halted.store(0, Ordering::Release);
            shared.set_interrupt(interrupt::EXITTB);
        });
        PSCI_RET_SUCCESS
    }

    /// `arm_set_cpu_off()` for the running vCPU: mark it off and halt it.
    pub(crate) fn cpu_off(&self, cpu: &mut Cpu<'_>) {
        let shared = cpu.core.shared();
        {
            let (mut g, i) = self.lines(shared.cpu_index);
            g[i].power = PSCI_OFF;
        }
        shared.set_interrupt(interrupt::HALT);
    }

    /// Raise (`level` true) or lower the IRQ line of the vCPU whose shared half is `shared`,
    /// as the GIC does with `ARM_CPU_IRQ`.
    pub fn set_irq(&self, shared: &CpuShared, level: bool) {
        set_line(shared, interrupt::HARD, level);
    }

    /// Raise or lower the FIQ line, `ARM_CPU_FIQ`.
    pub fn set_fiq(&self, shared: &CpuShared, level: bool) {
        set_line(shared, INTERRUPT_FIQ, level);
    }

    /// Raise or lower the virtual IRQ line, `ARM_CPU_VIRQ`. The interrupt pending is the OR
    /// of this line and HCR_EL2.VI, as `arm_cpu_update_virq()` computes it.
    pub fn set_virq(&self, shared: &CpuShared, level: bool) {
        self.set_virtual_line(shared, INTERRUPT_VIRQ, level);
    }

    /// Raise or lower the virtual FIQ line, `ARM_CPU_VFIQ`.
    pub fn set_vfiq(&self, shared: &CpuShared, level: bool) {
        self.set_virtual_line(shared, INTERRUPT_VFIQ, level);
    }

    fn set_virtual_line(&self, shared: &CpuShared, bit: u32, level: bool) {
        if !self.features().el2 {
            // The GIC might tell us about VIRQ and VFIQ state, but if we don't have EL2
            // support we don't care.
            return;
        }
        let (mut g, i) = self.lines(shared.cpu_index);
        if level {
            g[i].gic |= bit;
            shared.cpu_interrupt(bit);
        } else {
            g[i].gic &= !bit;
            // The HCR_EL2 half is folded in by the vCPU itself on its next HCR_EL2 write;
            // here only a line that nothing else holds can be dropped. Ask the vCPU to
            // recompute instead of guessing.
            let shared2 = shared;
            shared2.async_run_on_cpu(move |cpu| {
                let ops = cpu.ops();
                let st = CpuArmState::load(cpu.env);
                arm_of(&ops).update_virt_lines(cpu, &st);
            });
        }
    }

    /// `arm_cpu_update_virq()`, `arm_cpu_update_vfiq()` and `arm_cpu_update_vserr()`: set
    /// the virtual interrupt bits from HCR_EL2 and the GIC lines.
    pub(crate) fn update_virt_lines(&self, cpu: &Cpu<'_>, st: &CpuArmState) {
        let hcr = st.hcr_el2_eff(self.features());
        let shared = cpu.core.shared();
        let (g, i) = self.lines(shared.cpu_index);
        let gic = g[i].gic;
        let want = |hcr_bit: u64, line: u32| hcr & hcr_bit != 0 || gic & line != 0;
        for (bit, level) in [
            (INTERRUPT_VIRQ, want(HCR_VI, INTERRUPT_VIRQ)),
            (INTERRUPT_VFIQ, want(HCR_VF, INTERRUPT_VFIQ)),
            (INTERRUPT_VSERR, hcr & HCR_VSE != 0),
        ] {
            if level != (shared.interrupt_request() & bit != 0) {
                set_line(shared, bit, level);
            }
        }
        drop(g);
    }

    /// The generic timer `timer` of the vCPU `shared` reached the deadline the board was
    /// given: recalculate it on the vCPU's thread, as `arm_gt_ptimer_cb()` and friends do.
    pub fn gt_timer_expired(shared: &CpuShared, timer: usize) {
        shared.async_run_on_cpu(move |cpu| {
            let ops = cpu.ops();
            let arm = arm_of(&ops);
            let mut st = CpuArmState::load(cpu.env);
            gtimer::recalc(arm, cpu, &mut st, timer);
            st.store(cpu.env);
        });
    }

    /// The generic timer count at `freq` Hz: host time since the CPU was made.
    pub(crate) fn counter(&self, freq: u64) -> u64 {
        let ns = self.start.elapsed().as_nanos();
        (ns * u128::from(freq) / 1_000_000_000) as u64
    }

    /// The host time at which the counter at `freq` Hz reaches `count`, or `None` when that
    /// is too far away to represent.
    pub(crate) fn counter_deadline(&self, freq: u64, count: u64) -> Option<Instant> {
        if freq == 0 {
            return None;
        }
        let ns = u128::from(count) * 1_000_000_000 / u128::from(freq);
        let ns = u64::try_from(ns).ok()?;
        self.start.checked_add(std::time::Duration::from_nanos(ns))
    }
}

/// Set or clear an interrupt request bit, kicking the vCPU when it is set.
fn set_line(shared: &CpuShared, bit: u32, level: bool) {
    if level {
        shared.cpu_interrupt(bit);
    } else {
        shared.reset_interrupt(bit);
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

/// `regime_el()`: the EL whose registers control the translation regime of `mmu_idx`.
pub(crate) fn regime_el(mmu_idx: usize) -> u32 {
    match mmu_idx {
        MMU_IDX_E3 => 3,
        crate::cpu::MMU_IDX_E20_0 | MMU_IDX_E20_2 | MMU_IDX_E20_2_PAN | MMU_IDX_E2 => 2,
        _ => 1,
    }
}

/// `regime_has_2_ranges()`.
pub(crate) fn regime_has_2_ranges(mmu_idx: usize) -> bool {
    regime_el(mmu_idx) == 1
        || matches!(mmu_idx, crate::cpu::MMU_IDX_E20_0 | MMU_IDX_E20_2 | MMU_IDX_E20_2_PAN)
}

/// `aa64_va_parameter_tbi()` and `aa64_va_parameter_tbid()`: the two TBI bits and the two
/// TBID bits of the regime's TCR, each replicated for a one range regime.
pub(crate) fn tbi_bits(tcr: u64, mmu_idx: usize) -> (u32, u32) {
    if regime_has_2_ranges(mmu_idx) {
        (((tcr >> 37) & 3) as u32, ((tcr >> 51) & 3) as u32)
    } else {
        (((tcr >> 20) & 1) as u32 * 3, ((tcr >> 29) & 1) as u32 * 3)
    }
}

/// The TB flags of a state, the parts of `rebuild_hflags_a64()` this port uses.
pub(crate) fn tb_flags(f: &ArmFeatures, st: &CpuArmState) -> u32 {
    let el = st.current_el();
    let mmu_idx = st.mmu_idx(f);
    let hcr = st.hcr_el2_eff(f);
    let mut flags = el | ((mmu_idx as u32) << TB_MMUIDX_SHIFT);
    if st.pstate & PSTATE_IL != 0 {
        flags |= TB_PSTATE_IL;
    }
    if hcr & HCR_E2H != 0 {
        flags |= TB_E2H;
    }
    // Compute the condition for using AccType_UNPRIV for LDTR et al.
    if st.pstate & PSTATE_UAO == 0 {
        match mmu_idx {
            MMU_IDX_E10_1 | MMU_IDX_E10_1_PAN => flags |= TB_UNPRIV,
            MMU_IDX_E20_2 | MMU_IDX_E20_2_PAN if st.hcr_el2 & HCR_TGE != 0 => flags |= TB_UNPRIV,
            _ => {}
        }
    }
    let rel = regime_el(mmu_idx) as usize;
    if st.sctlr_el[rel] & SCTLR_A != 0 {
        flags |= TB_ALIGN_MEM;
    }
    let (tbid, tbid_bits) = tbi_bits(st.tcr_el[rel], mmu_idx);
    let tbii = tbid & !tbid_bits;
    let fp_el = fp_exception_el(f, st);
    if f.sve {
        let mut sve_el = sve_exception_el(f, st, el);
        // If either FP or SVE are disabled, translator does not need len. If SVE EL >
        // FP EL, FP exception has precedence, and translator does not need SVE EL. Save
        // potential re-translations by forcing the unneeded data to zero.
        if fp_el != 0 {
            if sve_el > fp_el {
                sve_el = 0;
            }
        } else if sve_el == 0 {
            flags |= sve_vqm1_for_el(f, st, el) << TB_VL_SHIFT;
        }
        flags |= sve_el << TB_SVEEXC_EL_SHIFT;
    }
    flags | (tbii << TB_TBII_SHIFT) | (tbid << TB_TBID_SHIFT) | (fp_el << TB_FPEXC_EL_SHIFT)
}

/// `fp_exception_el()`: the EL that FP and AdvSIMD instructions trap to under CPACR_EL1,
/// CPTR_EL2 and CPTR_EL3, or 0 when they are enabled.
pub(crate) fn fp_exception_el(f: &ArmFeatures, st: &CpuArmState) -> u32 {
    fp_exception_el_at(f, st, st.current_el())
}

/// `fp_exception_el()` for code running at `cur_el`.
pub(crate) fn fp_exception_el_at(f: &ArmFeatures, st: &CpuArmState, cur_el: u32) -> u32 {
    let hcr = st.hcr_el2_eff(f);
    // The CPACR controls traps to EL1: 0 and 2 trap EL0 and EL1 accesses, 1 traps only EL0
    // accesses and 3 traps nothing. It is ignored if E2H and TGE are both set.
    if hcr & (HCR_E2H | HCR_TGE) != HCR_E2H | HCR_TGE {
        match (st.cpacr_el1 >> 20) & 3 {
            1 if cur_el != 0 => {}
            3 => {}
            _ if cur_el <= 1 => return 1,
            _ => {}
        }
    }
    // CPTR_EL2 changes format with HCR_EL2.E2H (regardless of TGE).
    if cur_el <= 2 {
        if hcr & HCR_E2H != 0 {
            match (st.cptr_el[2] >> 20) & 3 {
                1 if cur_el != 0 || hcr & HCR_TGE == 0 => {}
                3 => {}
                _ => return 2,
            }
        } else if st.is_el2_enabled(f) && st.cptr_el[2] & (1 << 10) != 0 {
            return 2;
        }
    }
    // CPTR_EL3: trap all FP ops to EL3.
    if st.cptr_el[3] & (1 << 10) != 0 {
        return 3;
    }
    0
}

/// `el_is_in_host()`: whether `el` runs in the EL2&0 host regime.
fn el_is_in_host(f: &ArmFeatures, st: &CpuArmState, el: u32) -> bool {
    if el & 1 != 0 {
        return false;
    }
    let mask = if el != 0 { HCR_E2H } else { HCR_E2H | HCR_TGE };
    st.hcr_el2 & mask == mask && st.is_el2_enabled(f)
}

/// `sve_exception_el()`: the EL that SVE instructions at `el` trap to under CPACR_EL1.ZEN,
/// CPTR_EL2 and CPTR_EL3.EZ, or 0 when they are enabled.
pub(crate) fn sve_exception_el(f: &ArmFeatures, st: &CpuArmState, el: u32) -> u32 {
    if el <= 1 && !el_is_in_host(f, st, el) {
        match (st.cpacr_el1 >> 16) & 3 {
            1 if el != 0 => {}
            3 => {}
            _ => return 1,
        }
    }
    if el <= 2 && st.is_el2_enabled(f) {
        // CPTR_EL2 changes format with HCR_EL2.E2H (regardless of TGE).
        if st.hcr_el2 & HCR_E2H != 0 {
            match (st.cptr_el[2] >> 16) & 3 {
                1 if el != 0 || st.hcr_el2 & HCR_TGE == 0 => {}
                3 => {}
                _ => return 2,
            }
        } else if st.cptr_el[2] & (1 << 8) != 0 {
            return 2;
        }
    }
    // CPTR_EL3. Since EZ is negative we must check for EL3.
    if f.el3 && st.cptr_el[3] & (1 << 8) == 0 {
        return 3;
    }
    0
}

/// `sve_vqm1_for_el()`: the vector length at `el` in quadwords minus one, from the ZCR_ELx
/// LEN fields and the supported lengths (every length up to `sve-max-vq`).
pub(crate) fn sve_vqm1_for_el(f: &ArmFeatures, st: &CpuArmState, el: u32) -> u32 {
    let mut len = crate::cpu::ARM_MAX_VQ as u32 - 1;
    if el <= 1 && !el_is_in_host(f, st, el) {
        len = len.min(st.zcr_el[1] as u32 & 0xf);
    }
    if el <= 2 && st.is_el2_enabled(f) {
        len = len.min(st.zcr_el[2] as u32 & 0xf);
    }
    if f.el3 {
        len = len.min(st.zcr_el[3] as u32 & 0xf);
    }
    len.min(f.sve_max_vq.max(1) - 1)
}

/// `aarch64_sve_narrow_vq()`: zero the parts of the Z and P registers and FFR above `vq`
/// quadwords.
pub(crate) fn sve_narrow_vq(st: &mut CpuArmState, vq: usize) {
    for z in st.zregs.iter_mut() {
        z[2 * vq..].fill(0);
    }
    let mut pmask = if vq & 3 != 0 { !(u64::MAX << (16 * (vq & 3))) } else { 0 };
    for j in vq / 4..crate::cpu::ARM_MAX_VQ / 4 {
        for p in st.pregs.iter_mut() {
            p[j] &= pmask;
        }
        pmask = 0;
    }
}

/// `aarch64_sve_change_el()`: clear the state an exception entry or return from `old_el`
/// to `new_el` makes inaccessible by shortening the vector length. Every EL is AArch64.
pub(crate) fn sve_change_el(f: &ArmFeatures, st: &mut CpuArmState, old_el: u32, new_el: u32) {
    if !f.sve {
        return;
    }
    // Nothing to do if FP is disabled in either EL.
    if fp_exception_el_at(f, st, old_el) != 0 || fp_exception_el_at(f, st, new_el) != 0 {
        return;
    }
    let len = |el| {
        if sve_exception_el(f, st, el) != 0 { 0 } else { sve_vqm1_for_el(f, st, el) }
    };
    let (old_len, new_len) = (len(old_el), len(new_el));
    // When changing vector length, clear inaccessible state.
    if new_len < old_len {
        sve_narrow_vq(st, new_len as usize + 1);
    }
}

/// `exception_target_el()`: the EL synchronous exceptions from the current EL go to before
/// the HCR_EL2.TGE redirection.
pub(crate) fn exception_target_el(st: &CpuArmState) -> u32 {
    st.current_el().max(1)
}

/// `arm_is_secure()`.
fn is_secure(f: &ArmFeatures, st: &CpuArmState) -> bool {
    st.current_el() == 3 || st.is_secure_below_el3(f)
}

/// `arm_phys_excp_target_el()` for a CPU whose EL3 (if any) is AArch64 with SCR_EL3.RW set.
pub(crate) fn phys_excp_target_el(f: &ArmFeatures, st: &CpuArmState, excp: i32) -> u32 {
    let cur_el = st.current_el() as usize;
    let secure = is_secure(f, st);
    let hcr_el2 = st.hcr_el2_eff(f);
    let (scr, hcr) = match excp {
        EXCP_IRQ => (st.scr_el3 & SCR_IRQ != 0, hcr_el2 & HCR_IMO != 0),
        EXCP_FIQ => (st.scr_el3 & SCR_FIQ != 0, hcr_el2 & HCR_FMO != 0),
        _ => (st.scr_el3 & SCR_EA != 0, hcr_el2 & HCR_AMO != 0),
    };
    // For these purposes, TGE and AMO/IMO/FMO both force the interrupt to EL2.
    let hcr = hcr || hcr_el2 & HCR_TGE != 0;
    // The 64-bit EL3, SCR_EL3.RW = 1 half of target_el_table[]; -1 entries become 1.
    const TABLE: [[[[u32; 4]; 2]; 2]; 2] = [
        [[[1, 1, 2, 1], [1, 1, 1, 1]], [[2, 2, 2, 1], [2, 2, 2, 1]]],
        [[[3, 3, 3, 1], [3, 3, 3, 3]], [[3, 3, 3, 1], [3, 3, 3, 3]]],
    ];
    TABLE[usize::from(scr)][usize::from(hcr)][usize::from(secure)][cur_el]
}

/// `arm_excp_unmasked()` without FEAT_NMI.
fn excp_unmasked(st: &CpuArmState, excp: i32, target_el: u32, hcr_el2: u64) -> bool {
    let cur_el = st.current_el();
    // Don't take exceptions if they target a lower EL.
    if cur_el > target_el {
        return false;
    }
    let hypervized = |bit: u64| hcr_el2 & bit != 0 && hcr_el2 & HCR_TGE == 0;
    let pstate_unmasked = match excp {
        EXCP_FIQ => st.daif & PSTATE_F == 0,
        EXCP_IRQ => st.daif & PSTATE_I == 0,
        // Virtual interrupts are only taken when hypervized.
        EXCP_VFIQ => return hypervized(HCR_FMO) && st.daif & PSTATE_F == 0,
        EXCP_VIRQ => return hypervized(HCR_IMO) && st.daif & PSTATE_I == 0,
        EXCP_VSERR => return hypervized(HCR_AMO) && st.daif & PSTATE_A == 0,
        _ => unreachable!("not an interrupt"),
    };
    // Exceptions targeting a higher EL may not be maskable.
    let unmasked = target_el > cur_el
        && target_el != 1
        && match target_el {
            // An interrupt can be masked when HCR_E2H and HCR_TGE are both set regardless of
            // the current Security state.
            2 => hcr_el2 & (HCR_E2H | HCR_TGE) != HCR_E2H | HCR_TGE,
            // Interrupt cannot be masked when the target EL is 3.
            _ => true,
        };
    unmasked || pstate_unmasked
}

impl Arm {
    /// `arm_cpu_do_interrupt_aarch64()`: take the exception in `cpu.core.exception_index` to
    /// the EL in `exception_target_el`.
    fn do_interrupt_aarch64(&self, cpu: &mut Cpu<'_>) {
        let mut st = CpuArmState::load(cpu.env);
        let excp = cpu.core.exception_index;
        let new_el = st.exception_target_el;
        let ne = new_el as usize;
        let cur_el = st.current_el();
        let mut addr = st.vbar_el[ne];
        let old_mode = st.pstate_read();

        sve_change_el(self.features(), &mut st, cur_el, new_el);

        if cur_el < new_el {
            // Entry vector offset depends on whether the implemented EL immediately lower
            // than the target level is using AArch32 or AArch64; here it is always AArch64.
            addr += 0x400;
        } else if st.pstate & PSTATE_SP != 0 {
            addr += 0x200;
        }

        match excp {
            EXCP_PREFETCH_ABORT | EXCP_DATA_ABORT | EXCP_BKPT | EXCP_UDEF | EXCP_SWI | EXCP_HVC
            | EXCP_HYP_TRAP | EXCP_SMC => {
                if excp == EXCP_PREFETCH_ABORT || excp == EXCP_DATA_ABORT {
                    st.far_el[ne] = st.exception_vaddress;
                }
                if syn_get_ec(st.exception_syndrome) == EC_ADVSIMDFPACCESSTRAP {
                    // Mask out the AArch32 only fields to get a valid AArch64 syndrome.
                    st.exception_syndrome &= !0xf_ffff;
                }
                st.esr_el[ne] = u64::from(st.exception_syndrome);
            }
            EXCP_IRQ | EXCP_VIRQ => addr += 0x80,
            EXCP_FIQ | EXCP_VFIQ => addr += 0x100,
            EXCP_VSERR => {
                addr += 0x180;
                // Construct the SError syndrome from IDS and ISS fields.
                st.esr_el[ne] = u64::from(syn_serror((st.vsesr_el2 & 0x1ff_ffff) as u32));
            }
            _ => panic!("Unhandled exception 0x{excp:x}"),
        }

        st.save_sp(cur_el);
        st.elr_el[ne] = st.pc;
        st.spsr_el[ne] = u64::from(old_mode);

        let mut new_mode = (new_el << 2) | 1;
        if self.features().pan {
            // The value of PSTATE.PAN is normally preserved, except when an exception is
            // taken to an EL that uses the PAN bit with SCTLR_ELx.SPAN clear, which sets
            // it.
            new_mode |= old_mode & PSTATE_PAN;
            let hcr = st.hcr_el2_eff(self.features());
            let pan_el = match new_el {
                1 => true,
                2 => hcr & (HCR_E2H | HCR_TGE) == HCR_E2H | HCR_TGE,
                _ => false,
            };
            if pan_el && st.sctlr_el[ne] & SCTLR_SPAN == 0 {
                new_mode |= PSTATE_PAN;
            }
        }
        st.pstate_write(PSTATE_DAIF | new_mode);
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
        TbCpuState { pc: st.pc, flags: tb_flags(self.features(), &st), cflags: 0, cs_base: 0 }
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
        // arm_cpu_exec_interrupt(); the prioritization of interrupts is IMPLEMENTATION
        // DEFINED.
        let mut st = CpuArmState::load(cpu.env);
        let f = self.features();
        let hcr_el2 = st.hcr_el2_eff(f);
        let candidates = [
            (INTERRUPT_FIQ, EXCP_FIQ),
            (interrupt::HARD, EXCP_IRQ),
            (INTERRUPT_VIRQ, EXCP_VIRQ),
            (INTERRUPT_VFIQ, EXCP_VFIQ),
            (INTERRUPT_VSERR, EXCP_VSERR),
        ];
        for (bit, excp) in candidates {
            if interrupt_request & bit == 0 {
                continue;
            }
            let target_el = match excp {
                EXCP_FIQ | EXCP_IRQ => phys_excp_target_el(f, &st, excp),
                _ => 1,
            };
            if !excp_unmasked(&st, excp, target_el, hcr_el2) {
                continue;
            }
            if excp == EXCP_VSERR {
                // Taking a virtual abort clears HCR_EL2.VSE.
                st.hcr_el2 &= !HCR_VSE;
                cpu.core.shared().reset_interrupt(INTERRUPT_VSERR);
            }
            cpu.core.exception_index = excp;
            st.exception_target_el = target_el;
            st.store(cpu.env);
            self.do_interrupt_aarch64(cpu);
            return true;
        }
        false
    }

    fn do_interrupt(&self, cpu: &mut Cpu<'_>) {
        // arm_cpu_do_interrupt(): PSCI calls are handled before the exception is taken.
        if psci::is_psci_call(self, cpu, cpu.core.exception_index) {
            psci::handle_psci_call(self, cpu);
            return;
        }
        self.do_interrupt_aarch64(cpu);
    }

    fn has_work(&self, cpu: &Cpu<'_>) -> bool {
        // arm_cpu_has_work(): a powered off CPU has no work.
        let shared = cpu.core.shared();
        if self.power_state(shared.cpu_index) == PSCI_OFF {
            return false;
        }
        shared.interrupt_request()
            & (INTERRUPT_FIQ
                | interrupt::HARD
                | INTERRUPT_VIRQ
                | INTERRUPT_VFIQ
                | INTERRUPT_VSERR
                | interrupt::EXITTB)
            != 0
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
        ptw::deliver_fault(self, cpu, addr, access_type, ptw::Fault::new(fsc::ALIGNMENT), ra)
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
        let fault = ptw::Fault {
            ea: response != MemTxResult::DECODE_ERROR,
            ..ptw::Fault::new(fsc::SYNC_EXTERNAL)
        };
        Err(ptw::deliver_fault(self, cpu, addr, access_type, fault, ra))
    }

    fn mmu_index(&self, cpu: &Cpu<'_>, _ifetch: bool) -> usize {
        CpuArmState::load(cpu.env).mmu_idx(self.features())
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
    let el2 = ops.features().el2;
    let mut v = jit.create_vcpu(ops.clone(), as_, ENV_SIZE);
    let mut st = state.clone();
    if el2 && st.vmpidr_el2 == 0 {
        // VMPIDR_EL2 resets to the MPIDR_EL1 value, which depends on the vCPU index.
        st.vmpidr_el2 = (1 << 31) | ops.mp_affinity(v.shared().cpu_index);
    }
    st.store(&mut v.env);
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
