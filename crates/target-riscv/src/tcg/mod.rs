// SPDX-License-Identifier: GPL-2.0-or-later

//! The RV64 TCG front end: the translator ([`translate`]), its helpers, the CSRs, PMP, the
//! page walk and the [`CpuOps`] glue, after QEMU's `target/riscv/tcg`, `cpu_helper.c`,
//! `csr.c`, `pmp.c` and `op_helper.c`.
//!
//! The `mip` register and the interrupt lines that drive it live in [`Riscv`] rather than
//! in `env`, because devices on other threads change them (QEMU changes `env->mip` under
//! the BQL). The vCPU thread reads them under the same lock. So do the guest external
//! interrupt lines of H (`hgeip`), with the copies of `hgeie` and `hstatus.VGEIN` they need,
//! and whether an interrupt `mvien` or `hvien` injects is pending. The board gives GEILEN
//! ([`Riscv::with_geilen`]) and the registers of the IMSIC ([`RiscvBoard::aia_ireg_rmw`]).
//!
//! Differences from QEMU:
//!
//! - Only RV64. `misa` and the `mstatus` UXL and SXL fields are read only, so the XLEN
//!   never changes.
//! - The extensions come from the CPU model and its properties ([`crate::cfg`]), as in
//!   QEMU. Extensions this port does not have yet cannot be turned on, and CPU models
//!   that need them are refused. VLEN is fixed at 128 bits, QEMU's default, so other `vlen`
//!   values are refused too.
//! - `vsstatus` keeps every bit a write gives it, as in QEMU, but only the bits that
//!   `riscv_cpu_swap_hypervisor_regs()` swaps reach `mstatus` when V becomes 1; QEMU ORs
//!   the whole of `vsstatus` into `mstatus`.
//! - `mcycle` and `minstret` both count host time in nanoseconds since the CPU was made,
//!   where QEMU reads `cpu_get_host_ticks()`, and so do the `mhpmcounter` registers whose
//!   `mhpmevent` selects the cycle or instruction event ([`pmu`]).
//! - The page walk updates the A and D bits of a PTE with a plain store, where QEMU uses a
//!   compare and swap on RAM and restarts the walk if the PTE changed under it.
//! - There is no Smrnmi, Smdbltrp, Ssdbltrp, Smctr or control flow integrity and no icount.
//! - The debug triggers (`tselect`, `tdata1` to `tdata3`) hold values but never fire.
//! - Conditional branches may continue the block on the fall-through path (a superblock),
//!   with the taken edge emitted out of line at the end of the block; QEMU ends the block
//!   at every branch.

mod crypto;
mod csr;
mod fpu;
mod helpers;
mod pm;
mod pmp;
mod pmu;
mod ptw;
mod semihost;
mod translate;
mod translate_fp;
mod translate_rvh;
mod translate_rvv;
mod translate_rvv_fp;
mod translate_rvv_int;
mod translate_rvv_perm;
mod translate_rvvk;
mod vcrypto;
mod vector;
mod vector_fp;
mod vector_int;
mod vector_perm;

use std::fmt;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use ruvm_jit::translate::TbBuild;
use ruvm_jit::{
    Cpu, CpuLoopExit, CpuOps, CpuShared, InterpBackend, Jit, JitConfig, MmuAccessType, Ra, Tb,
    TbCpuState, Vcpu, cputlb, interrupt, translator_loop,
};
use ruvm_jit_interp::HelperRegistry;
use ruvm_mem::{AddressSpace, MemTxAttrs, MemTxResult};

use crate::cpu::{
    CpuRiscvState, ENV_SIZE, EXCP_BREAKPOINT, EXCP_ILLEGAL_INST, EXCP_INST_ACCESS_FAULT,
    EXCP_INST_ADDR_MIS, EXCP_INST_GUEST_PAGE_FAULT, EXCP_INST_PAGE_FAULT, EXCP_INT_FLAG,
    EXCP_LOAD_ACCESS_FAULT, EXCP_LOAD_ADDR_MIS, EXCP_LOAD_GUEST_ACCESS_FAULT, EXCP_LOAD_PAGE_FAULT,
    EXCP_M_ECALL, EXCP_S_ECALL, EXCP_SEMIHOST, EXCP_STORE_AMO_ACCESS_FAULT,
    EXCP_STORE_AMO_ADDR_MIS, EXCP_STORE_GUEST_AMO_ACCESS_FAULT, EXCP_STORE_PAGE_FAULT,
    EXCP_U_ECALL, EXCP_VIRT_INSTRUCTION_FAULT, EXCP_VS_ECALL, HSTATUS_GVA, HSTATUS_SPV,
    HSTATUS_SPVP, IPRIO_DEFAULT_M, IPRIO_DEFAULT_S, IPRIO_MMAXIPRIO, IRQ_LOCAL_GUEST_MAX,
    IRQ_LOCAL_MAX, IRQ_M_EXT, IRQ_S_EXT, IRQ_VS_EXT, IRQ_VS_SOFT, IRQ_VS_TIMER, MENVCFG_STCE,
    MIP_LCOFIP, MIP_SEIP, MIP_SGEIP, MIP_STIP, MIP_VSEIP, MIP_VSTIP, MMU_2STAGE_BIT, MMU_IDX_S_SUM,
    MSTATUS, MSTATUS_FS, MSTATUS_GVA, MSTATUS_HS, MSTATUS_MIE, MSTATUS_MPIE, MSTATUS_MPP,
    MSTATUS_MPRV, MSTATUS_MPV, MSTATUS_MXR, MSTATUS_SIE, MSTATUS_SPIE, MSTATUS_SPP, MSTATUS_SUM,
    MSTATUS_VS, MSTATUS64_UXL, NB_MMU_MODES, PC, PRIV, PRV_M, PRV_S, RVF, RiscvCfg,
    UW2_ALWAYS_STORE_AMO, VILL, VIRT_ENABLED, VS_MODE_INTERRUPTS, VSSTATUS, VSTART, VTYPE,
    VTYPE_VLMUL, VTYPE_VMA, VTYPE_VSEW, VTYPE_VTA, default_priority, get_field, iprio, set_field,
};

use csr::vs_bits_down;
pub use semihost::{ADP_STOPPED_APPLICATION_EXIT, SemihostingHost};

/// The TB flags: the MMU index of data accesses in bits 0 to 2.
pub const TB_MEM_IDX_MASK: u32 = 7;
/// The TB flags: the privilege level in bits 3 and 4.
pub const TB_PRIV_SHIFT: u32 = 3;
/// The TB flags: `mstatus.FS` in bits 5 and 6.
pub const TB_FS_SHIFT: u32 = 5;
/// The TB flags: `mstatus.VS` in bits 7 and 8.
pub const TB_VS_SHIFT: u32 = 7;
/// The TB flags: `vtype.vlmul` in bits 9 to 11.
pub const TB_LMUL_SHIFT: u32 = 9;
/// The TB flags: `vtype.vsew` in bits 12 to 14.
pub const TB_SEW_SHIFT: u32 = 12;
/// The TB flags: `vill` in bit 15.
pub const TB_VILL: u32 = 1 << 15;
/// The TB flags: `vstart == 0` in bit 16.
pub const TB_VSTART_EQ_ZERO: u32 = 1 << 16;
/// The TB flags: `vtype.vta` in bit 17.
pub const TB_VTA: u32 = 1 << 17;
/// The TB flags: `vtype.vma` in bit 18.
pub const TB_VMA: u32 = 1 << 18;
/// The TB flags: `virt_enabled` in bit 19.
pub const TB_VIRT: u32 = 1 << 19;
/// The shift of the 2 bit `PM_PMM` field: the PMM of data accesses, 0 if none.
pub const TB_PM_PMM_SHIFT: u32 = 20;
/// `PM_SIGNEXTEND`: masked data addresses are sign extended.
pub const TB_PM_SIGNEXTEND: u32 = 1 << 22;

/// The board side of a RISC-V CPU: the ACLINT timer the `time` CSR and Sstc read, and the
/// timer behind `stimecmp`.
pub trait RiscvBoard: Send + Sync {
    /// `rdtime_fn`: the current value of the `mtime` counter, or `None` if the board has no
    /// timer (then the `time` CSR is an illegal instruction, as in QEMU).
    fn rdtime(&self) -> Option<u64>;

    /// The frequency `mtime` counts at, in Hz.
    fn timebase_freq(&self) -> u64;

    /// Arm the Sstc supervisor timer of the vCPU `shared` for `deadline`, or cancel it with
    /// `None`, as `riscv_timer_write_timecmp()` arms `env->stimer`. When the deadline
    /// passes, the board calls [`Riscv::stimer_expired`].
    fn stimer_update(&self, shared: &CpuShared, deadline: Option<Instant>) {
        let _ = (shared, deadline);
    }

    /// `timer_mod_anticipate_ns()` on the PMU timer of the vCPU `shared` (`pmu_timer`):
    /// fire it in `delay_ns` nanoseconds of the virtual clock, or keep it if it is armed
    /// for an earlier time. When it fires, the board calls [`Riscv::pmu_timer_expired`].
    fn pmu_timer_anticipate(&self, shared: &CpuShared, delay_ns: u64) {
        let _ = (shared, delay_ns);
    }

    /// Arm the VS mode Sstc timer of the vCPU `shared` (`env->vstimer`) for `deadline`,
    /// or cancel it with `None`. When the deadline passes, the board calls
    /// [`Riscv::vstimer_expired`].
    fn vstimer_update(&self, shared: &CpuShared, deadline: Option<Instant>) {
        let _ = (shared, deadline);
    }

    /// The `aia_ireg_rmw_fn` the IMSIC of level `priv_lvl` (`PRV_M` or `PRV_S`) of the
    /// vCPU `shared` registers, `riscv_imsic_rmw()`: read the IMSIC register `reg` (the
    /// `AIA_MAKE_IREG()` encoding) and replace its `wr_mask` bits with those of `new`.
    /// `None` if the hart has no IMSIC at that level, `Some(Err(()))` if the IMSIC
    /// refuses the access.
    fn aia_ireg_rmw(
        &self,
        shared: &CpuShared,
        priv_lvl: u64,
        reg: u32,
        new: u64,
        wr_mask: u64,
    ) -> Option<Result<u64, ()>> {
        let _ = (shared, priv_lvl, reg, new, wr_mask);
        None
    }
}

/// Which Sstc timer a write is for: the `timer_irq` argument of QEMU's timer helpers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SstcTimer {
    /// `stimecmp`, raising STIP.
    S,
    /// `vstimecmp`, raising VSTIP through `vstime_irq`.
    Vs,
}

/// The per vCPU interrupt state that lives outside `env` because other threads change it.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CpuLines {
    /// `mip`.
    pub(crate) mip: u64,
    /// The SEIP input from the interrupt controller, `external_seip`.
    pub(crate) external_seip: bool,
    /// The SEIP bit M mode software wrote, `software_seip`.
    pub(crate) software_seip: bool,
    /// `vstime_irq`: the VS mode Sstc timer has fired, which shows as VSTIP.
    pub(crate) vstime_irq: bool,
    /// The PMU timer has fired and the vCPU has not yet run `riscv_pmu_timer_cb()`, which
    /// needs `env`.
    pub(crate) pmu_timer: bool,
    /// `hgeip`: bit `i` is the guest external interrupt line `i`, 1 to GEILEN.
    pub(crate) hgeip: u64,
    /// A copy of `hgeie`, which decides SGEIP when a guest external interrupt line moves.
    pub(crate) hgeie: u64,
    /// A copy of `hstatus.VGEIN`, the guest external interrupt that shows as VSEIP.
    pub(crate) vgein: u32,
    /// Whether an interrupt injected through `mvien` and `mvip`, or `hvien` and `hvip`,
    /// is pending and enabled in `sie` or `vsie`.
    pub(crate) irqf: bool,
}

impl CpuLines {
    /// VSEIP if the guest external interrupt `hstatus.VGEIN` selects is pending, the
    /// `vsgein` of `riscv_cpu_all_pending()`.
    pub(crate) fn vsgein(&self) -> u64 {
        if self.vgein < 64 && (self.hgeip >> self.vgein) & 1 != 0 { MIP_VSEIP } else { 0 }
    }

    /// Set SGEIP in `mip` to whether a guest external interrupt is pending and enabled.
    pub(crate) fn update_sgeip(&mut self) {
        let v = if self.hgeie & self.hgeip != 0 { MIP_SGEIP } else { 0 };
        self.mip = (self.mip & !MIP_SGEIP) | v;
    }

    /// `riscv_cpu_all_pending()` with the enables `mie`.
    pub(crate) fn all_pending(&self, mie: u64) -> u64 {
        let vstip = if self.vstime_irq { MIP_VSTIP } else { 0 };
        (self.mip | self.vsgein() | vstip) & mie
    }

    /// Whether `riscv_cpu_interrupt()` raises the hard interrupt request: an interrupt
    /// is pending in `mip`, through VGEIN, the VS timer or the injected ones.
    fn hard(&self) -> bool {
        self.mip != 0 || self.vsgein() != 0 || self.vstime_irq || self.irqf
    }
}

/// The RV64 CPU: the [`CpuOps`] of vCPUs translated by the RISC-V front end.
pub struct Riscv {
    start: Instant,
    board: OnceLock<Arc<dyn RiscvBoard>>,
    semihost: Option<semihost::Semihosting>,
    cfg: RiscvCfg,
    geilen: u32,
    lines: Mutex<Vec<CpuLines>>,
}

impl fmt::Debug for Riscv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Riscv")
            .field("semihosting", &self.semihost.is_some())
            .field("cfg", &self.cfg)
            .finish_non_exhaustive()
    }
}

impl Default for Riscv {
    fn default() -> Riscv {
        Riscv::new()
    }
}

impl Riscv {
    /// QEMU's default `rv64` CPU.
    pub fn new() -> Riscv {
        Riscv {
            start: Instant::now(),
            board: OnceLock::new(),
            semihost: None,
            cfg: RiscvCfg::default(),
            geilen: 0,
            lines: Mutex::new(Vec::new()),
        }
    }

    /// The same CPU with semihosting on, QEMU's `-semihosting`: the `slli; ebreak; srai`
    /// sequence makes a call to `host`, from M and S mode and also from U mode when
    /// `userspace` is set (`-semihosting-config userspace=on`).
    pub fn with_semihosting(mut self, host: Arc<dyn SemihostingHost>, userspace: bool) -> Riscv {
        self.semihost = Some(semihost::Semihosting::new(host, userspace));
        self
    }

    /// The same CPU with the XLRBR vendor extension (`-cpu rv64,xlrbr=true`): the CRC32
    /// instructions, and X in `misa`.
    pub fn with_xlrbr(mut self, on: bool) -> Riscv {
        self.cfg.ext_xlrbr = on;
        if on {
            self.cfg.misa_ext |= crate::cfg::RVX;
        } else {
            self.cfg.misa_ext &= !crate::cfg::RVX;
        }
        self
    }

    /// The same CPU with the configuration `cfg`, as [`crate::cfg::CpuBuilder::finalize`]
    /// gives it. The `misa` of the harts must come from [`CpuRiscvState::reset_cfg`] with
    /// the same `cfg`.
    pub fn with_cfg(mut self, cfg: RiscvCfg) -> Riscv {
        self.cfg = cfg;
        self
    }

    /// The configuration of the CPU.
    pub fn cfg(&self) -> &RiscvCfg {
        &self.cfg
    }

    /// The same CPU with `geilen` guest external interrupts, as an S level IMSIC with
    /// `geilen + 1` interrupt files sets with `riscv_cpu_set_geilen()`. Without H, or for
    /// more than `IRQ_LOCAL_GUEST_MAX`, nothing changes, as in QEMU. Call it after
    /// [`Riscv::with_cfg`].
    pub fn with_geilen(mut self, geilen: u32) -> Riscv {
        if self.cfg.ext_h() && geilen <= IRQ_LOCAL_GUEST_MAX {
            self.geilen = geilen;
        }
        self
    }

    /// `GEILEN`: the number of guest external interrupts of each hart.
    pub fn geilen(&self) -> u32 {
        self.geilen
    }

    /// Wire the CPU to `board`. Only the first call has an effect; the board usually
    /// holds a weak reference back to the CPU to raise interrupts.
    pub fn set_board(&self, board: Arc<dyn RiscvBoard>) {
        let _ = self.board.set(board);
    }

    fn board(&self) -> Option<&Arc<dyn RiscvBoard>> {
        self.board.get()
    }

    /// Whether semihosting calls are on, and whether also from U mode.
    pub(crate) fn semihosting(&self) -> Option<bool> {
        self.semihost.as_ref().map(|s| s.userspace)
    }

    /// Whether the XLRBR extension is on.
    pub(crate) fn xlrbr(&self) -> bool {
        self.cfg.ext_xlrbr
    }

    fn lines(&self, cpu_index: usize) -> (MutexGuard<'_, Vec<CpuLines>>, usize) {
        let mut g = self.lines.lock().unwrap_or_else(|e| e.into_inner());
        if g.len() <= cpu_index {
            g.resize(cpu_index + 1, CpuLines::default());
        }
        (g, cpu_index)
    }

    /// Run `f` on the interrupt lines of the vCPU `shared` with the lock held, then raise
    /// or clear its interrupt request to match `mip`, as every change of `mip` does in
    /// `riscv_cpu_update_mip()`.
    pub(crate) fn with_lines<R>(
        &self,
        shared: &CpuShared,
        f: impl FnOnce(&mut CpuLines) -> R,
    ) -> R {
        let (mut g, i) = self.lines(shared.cpu_index);
        let r = f(&mut g[i]);
        // riscv_cpu_interrupt(). The request bit follows mip under the lock, so a racing
        // update can never leave it clear while an interrupt is pending.
        let raise = g[i].hard() || g[i].pmu_timer;
        if raise {
            shared.set_interrupt(interrupt::HARD);
        } else {
            shared.reset_interrupt(interrupt::HARD);
        }
        drop(g);
        // The kick takes the vCPU's halt lock, and a halted vCPU holds that lock while it
        // reads mip through this one, so it happens after the lines are unlocked. Setting
        // the bit again is harmless; if mip was cleared meanwhile the vCPU finds nothing.
        if raise {
            shared.cpu_interrupt(interrupt::HARD);
        }
        r
    }

    /// `env->mip` of the vCPU `cpu_index`.
    pub fn mip(&self, cpu_index: usize) -> u64 {
        let (g, i) = self.lines(cpu_index);
        g[i].mip
    }

    /// `riscv_cpu_update_mip()`: replace the `mask` bits of `mip` with those of `value`,
    /// returning the old `mip`.
    pub fn update_mip(&self, shared: &CpuShared, mask: u64, value: u64) -> u64 {
        self.with_lines(shared, |l| {
            let old = l.mip;
            // No need to update mip for VSTIP while the VS timer drives it.
            let mask = if mask == MIP_VSTIP && l.vstime_irq { 0 } else { mask };
            l.mip = (l.mip & !mask) | (value & mask);
            old
        })
    }

    /// Set `vstime_irq` and update the interrupt request.
    fn set_vstime_irq(&self, shared: &CpuShared, level: bool) {
        self.with_lines(shared, |l| l.vstime_irq = level);
    }

    /// `riscv_cpu_set_irq()`: drive the local interrupt input `irq` (an `IRQ_*` number) of
    /// the vCPU `shared` to `level`, as the interrupt controllers and timers do. Inputs
    /// `IRQ_LOCAL_MAX + i - 1` are the guest external interrupt lines, bit `i` of `hgeip`,
    /// which need H and `i` no more than GEILEN.
    pub fn set_irq(&self, shared: &CpuShared, irq: u32, level: bool) {
        if irq == IRQ_S_EXT {
            self.with_lines(shared, |l| {
                l.external_seip = level;
                let v = level || l.software_seip;
                l.mip = (l.mip & !MIP_SEIP) | if v { MIP_SEIP } else { 0 };
            });
        } else if irq < IRQ_LOCAL_MAX {
            let bit = 1u64 << irq;
            self.update_mip(shared, bit, if level { bit } else { 0 });
        } else {
            assert!(
                irq < IRQ_LOCAL_MAX + IRQ_LOCAL_GUEST_MAX && self.cfg.ext_h(),
                "riscv: bad interrupt line {irq}"
            );
            let gein = irq - IRQ_LOCAL_MAX + 1;
            assert!(gein <= self.geilen, "riscv: guest external interrupt {gein} past GEILEN");
            self.with_lines(shared, |l| {
                l.hgeip = (l.hgeip & !(1 << gein)) | (u64::from(level) << gein);
                l.update_sgeip();
            });
        }
    }

    /// `hgeip` of the vCPU `cpu_index`.
    pub fn hgeip(&self, cpu_index: usize) -> u64 {
        let (g, i) = self.lines(cpu_index);
        g[i].hgeip
    }

    /// Bring the copies of `hgeie` and `hstatus.VGEIN` the lines keep back to their reset
    /// values, and drop the injected interrupts, as the reset of `env` does. Call it with
    /// every reset of the vCPU `shared`.
    pub fn reset_lines(&self, shared: &CpuShared) {
        self.with_lines(shared, |l| {
            l.hgeie = 0;
            l.vgein = 0;
            l.irqf = false;
            l.update_sgeip();
        });
    }

    /// The IMSIC callback of [`RiscvBoard::aia_ireg_rmw`], `None` without a board or IMSIC.
    pub(crate) fn aia_ireg_rmw(
        &self,
        shared: &CpuShared,
        priv_lvl: u64,
        reg: u32,
        new: u64,
        wr_mask: u64,
    ) -> Option<Result<u64, ()>> {
        self.board()?.aia_ireg_rmw(shared, priv_lvl, reg, new, wr_mask)
    }

    /// `rdtime_fn()`: the board's `mtime`, if it has a timer.
    pub(crate) fn rdtime(&self) -> Option<u64> {
        self.board().and_then(|b| b.rdtime())
    }

    /// The value `mcycle` and `minstret` count from, `cpu_get_host_ticks()`.
    pub(crate) fn host_ticks(&self) -> u64 {
        u64::try_from(self.start.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    /// `riscv_timer_write_timecmp()`: raise the interrupt of `timer` now if its compare
    /// value (`stimecmp`, or `vstimecmp` against `time + htimedelta`) has passed, else
    /// clear it and arm the board's timer for when it passes. Nothing happens without a
    /// board timer or with `menvcfg.STCE` clear, and for the VS timer also without H or
    /// with `henvcfg.STCE` clear.
    pub(crate) fn write_timecmp(&self, shared: &CpuShared, st: &CpuRiscvState, timer: SstcTimer) {
        let Some(board) = self.board() else { return };
        if st.menvcfg & MENVCFG_STCE == 0 {
            return;
        }
        if timer == SstcTimer::Vs && (!st.has_h() || st.henvcfg & MENVCFG_STCE == 0) {
            return;
        }
        let Some(time) = board.rdtime() else { return };
        let (timecmp, delta) = match timer {
            SstcTimer::S => (st.stimecmp, 0),
            SstcTimer::Vs => (st.vstimecmp, st.htimedelta),
        };
        let now = time.wrapping_add(delta);
        if timecmp <= now {
            // A compare value in the past raises the timer interrupt right away.
            match timer {
                SstcTimer::S => {
                    self.update_mip(shared, MIP_STIP, MIP_STIP);
                }
                SstcTimer::Vs => self.set_vstime_irq(shared, true),
            }
            return;
        }
        let update = |d| match timer {
            SstcTimer::S => board.stimer_update(shared, d),
            SstcTimer::Vs => board.vstimer_update(shared, d),
        };
        match timer {
            SstcTimer::S => {
                self.update_mip(shared, MIP_STIP, 0);
            }
            SstcTimer::Vs => self.set_vstime_irq(shared, false),
        }
        if timecmp == u64::MAX {
            update(None);
            return;
        }
        let diff = timecmp - now;
        let freq = u128::from(board.timebase_freq().max(1));
        let ns = u128::from(diff) * 1_000_000_000 / freq;
        let deadline = u64::try_from(ns)
            .ok()
            .and_then(|ns| Instant::now().checked_add(Duration::from_nanos(ns)));
        update(deadline);
    }

    /// `riscv_timer_disable_timecmp()`: stop `timer` and drop its interrupt, if the STCE
    /// bits that enable it are clear.
    pub(crate) fn disable_timecmp(&self, shared: &CpuShared, st: &CpuRiscvState, timer: SstcTimer) {
        let m_stce = st.menvcfg & MENVCFG_STCE != 0;
        let h_stce = st.henvcfg & MENVCFG_STCE != 0;
        match timer {
            SstcTimer::S if !m_stce => {
                self.update_mip(shared, MIP_STIP, 0);
                if let Some(board) = self.board() {
                    board.stimer_update(shared, None);
                }
            }
            SstcTimer::Vs if !m_stce || !h_stce => {
                self.set_vstime_irq(shared, false);
                if let Some(board) = self.board() {
                    board.vstimer_update(shared, None);
                }
            }
            _ => {}
        }
    }

    /// `riscv_timer_stce_changed()`: an STCE bit of `menvcfg` (`is_m`) or `henvcfg`
    /// flipped to `enable`.
    pub(crate) fn stce_changed(
        &self,
        shared: &CpuShared,
        st: &CpuRiscvState,
        is_m: bool,
        enable: bool,
    ) {
        if enable {
            self.write_timecmp(shared, st, SstcTimer::Vs);
        } else {
            self.disable_timecmp(shared, st, SstcTimer::Vs);
        }
        if is_m {
            if enable {
                self.write_timecmp(shared, st, SstcTimer::S);
            } else {
                self.disable_timecmp(shared, st, SstcTimer::S);
            }
        }
    }

    /// `timer_mod_anticipate_ns()` on the PMU timer of the vCPU `shared`: fire it in
    /// `delay_ns`, or earlier if it is already armed for an earlier time.
    pub(crate) fn pmu_timer(&self, shared: &CpuShared, delay_ns: u64) {
        if let Some(board) = self.board() {
            board.pmu_timer_anticipate(shared, delay_ns);
        }
    }

    /// The PMU timer the board was given by [`RiscvBoard::pmu_timer_anticipate`] fired.
    /// The vCPU runs `riscv_pmu_timer_cb()` before it next looks for interrupts.
    pub fn pmu_timer_expired(&self, shared: &CpuShared) {
        self.with_lines(shared, |l| l.pmu_timer = true);
    }

    /// `riscv_pmu_timer_cb()` on the vCPU thread: mark the counters that overflowed and
    /// raise LCOFIP, or arm the timer again for those that have not yet.
    fn pmu_timer_cb(&self, cpu: &mut Cpu<'_>, shared: &CpuShared) {
        let mut st = CpuRiscvState::load(cpu.env);
        let raise = pmu::timer_cb(&mut st, &self.cfg, self.host_ticks(), &mut |d| {
            self.pmu_timer(shared, d);
        });
        st.store(cpu.env);
        if raise {
            self.update_mip(shared, MIP_LCOFIP, MIP_LCOFIP);
        }
    }

    /// `riscv_stimer_cb()`: the deadline the board was given by
    /// [`RiscvBoard::stimer_update`] has passed; raise STIP.
    pub fn stimer_expired(&self, shared: &CpuShared) {
        self.update_mip(shared, MIP_STIP, MIP_STIP);
    }

    /// `riscv_vstimer_cb()`: the deadline the board was given by
    /// [`RiscvBoard::vstimer_update`] has passed; raise VSTIP.
    pub fn vstimer_expired(&self, shared: &CpuShared) {
        self.set_vstime_irq(shared, true);
    }

    /// `riscv_cpu_all_pending()`: the enabled pending interrupts, with VSEIP while the
    /// guest external interrupt VGEIN selects is pending and VSTIP while the VS timer has
    /// fired. The interrupts injected through `mvip` and `hvip` are not in it.
    fn all_pending(&self, cpu_index: usize, mie: u64) -> u64 {
        let (g, i) = self.lines(cpu_index);
        g[i].all_pending(mie)
    }

    /// `riscv_cpu_local_irq_pending()`: the interrupt to take now, if any.
    fn local_irq_pending(&self, cpu_index: usize, st: &CpuRiscvState) -> Option<u32> {
        let sie = st.mstatus & MSTATUS_SIE != 0;
        let (mie, hsie, vsie) = if st.virt() {
            (true, true, st.priv_lvl < PRV_S || (st.priv_lvl == PRV_S && sie))
        } else {
            (
                st.priv_lvl < PRV_M || st.mstatus & MSTATUS_MIE != 0,
                st.priv_lvl < PRV_S || (st.priv_lvl == PRV_S && sie),
                false,
            )
        };
        let pending = self.all_pending(cpu_index, st.mie);
        // M mode interrupts first, then HS mode, then VS mode ones; within a level
        // riscv_cpu_pending_to_irq() picks one.
        let irqs = pending & !st.mideleg;
        if mie && irqs != 0 {
            return pending_to_irq(&self.cfg, IRQ_M_EXT, IPRIO_DEFAULT_M, irqs, &st.miprio);
        }
        // The S level interrupts M mode injects through mvien and mvip.
        let irqs_f = st.mvip & st.mvien & !st.mideleg & st.sie;
        let irqs = (pending & st.mideleg & !st.hideleg) | irqs_f;
        if hsie && irqs != 0 {
            return pending_to_irq(&self.cfg, IRQ_S_EXT, IPRIO_DEFAULT_S, irqs, &st.siprio);
        }
        // The VS level interrupts HS mode injects through hvien and hvip, and the
        // delegated ones with the VS level bits brought down to their S level positions.
        let irqs_f_vs = st.hvip & st.hvien & !st.hideleg & st.vsie;
        let irqs = vs_bits_down(pending & st.mideleg & st.hideleg) | irqs_f_vs;
        if vsie && irqs != 0 {
            let virq = pending_to_irq(&self.cfg, IRQ_S_EXT, IPRIO_DEFAULT_S, irqs, &st.hviprio)?;
            return Some(if virq == 0 || virq > 12 { virq } else { virq + 1 });
        }
        None
    }

    /// `riscv_cpu_do_interrupt()`: take the exception or interrupt in
    /// `cpu.core.exception_index`.
    fn do_interrupt_riscv(&self, cpu: &mut Cpu<'_>) {
        let index = cpu.core.exception_index;
        let mut st = CpuRiscvState::load(cpu.env);
        let mut virt = st.virt();
        let mut write_gva = false;
        let always_storeamo = st.excp_uw2 & UW2_ALWAYS_STORE_AMO != 0;
        let is_async = index & EXCP_INT_FLAG != 0;
        let mut cause = (index & !EXCP_INT_FLAG) as u64;
        let deleg = if is_async { st.mideleg } else { st.medeleg };
        let hdeleg = if is_async { st.hideleg } else { st.hedeleg };
        let mut tval = 0;
        let mut tinst = 0;
        let mut htval = 0;
        let mut mtval2 = 0;

        if !is_async {
            // Set tval to badaddr for traps with address information.
            match cause as i32 {
                EXCP_SEMIHOST => {
                    if let Some(sh) = &self.semihost {
                        semihost::handle(self, sh, cpu);
                    }
                    let mut st = CpuRiscvState::load(cpu.env);
                    st.pc = st.pc.wrapping_add(4);
                    st.store(cpu.env);
                    cpu.core.exception_index = -1;
                    return;
                }
                EXCP_LOAD_GUEST_ACCESS_FAULT
                | EXCP_STORE_GUEST_AMO_ACCESS_FAULT
                | EXCP_LOAD_ADDR_MIS
                | EXCP_STORE_AMO_ADDR_MIS
                | EXCP_LOAD_ACCESS_FAULT
                | EXCP_STORE_AMO_ACCESS_FAULT
                | EXCP_LOAD_PAGE_FAULT
                | EXCP_STORE_PAGE_FAULT => {
                    if always_storeamo {
                        cause = promote_load_fault(cause as i32) as u64;
                    }
                    write_gva = st.two_stage_lookup != 0;
                    tval = st.badaddr;
                    tinst = if st.two_stage_indirect_lookup != 0 {
                        // The pseudoinstruction of a G stage fault taken while walking
                        // the VS stage page table.
                        0x3000
                    } else {
                        // The address offset field is non-zero only for misaligned
                        // accesses.
                        transformed_insn(&st, st.bins, tval)
                    };
                }
                EXCP_INST_GUEST_PAGE_FAULT
                | EXCP_INST_ADDR_MIS
                | EXCP_INST_ACCESS_FAULT
                | EXCP_INST_PAGE_FAULT => {
                    write_gva = st.two_stage_lookup != 0;
                    tval = st.badaddr;
                    if st.two_stage_indirect_lookup != 0 {
                        tinst = 0x3000;
                    }
                }
                EXCP_ILLEGAL_INST | EXCP_VIRT_INSTRUCTION_FAULT => tval = st.bins,
                EXCP_BREAKPOINT => tval = st.badaddr,
                _ => {}
            }
            // ecall is dispatched as one cause so translate based on mode.
            if cause == EXCP_U_ECALL as u64 {
                if st.priv_lvl == PRV_M {
                    cause = EXCP_M_ECALL as u64;
                } else if st.priv_lvl == PRV_S && st.virt() {
                    cause = EXCP_VS_ECALL as u64;
                } else if st.priv_lvl == PRV_S {
                    cause = EXCP_S_ECALL as u64;
                }
            }
        }

        let to_s = st.priv_lvl <= PRV_S && cause < 64 && (deleg >> cause) & 1 != 0;
        let vsmode_exc = st.virt() && cause < 64 && (hdeleg >> cause) & 1 != 0;
        let mut flush = false;
        if to_s {
            // Handle the trap in S mode.
            if st.has_h() {
                if vsmode_exc {
                    // Trap to VS mode. A VS level interrupt is reported with its S level
                    // cause.
                    if is_async && matches!(cause as u32, IRQ_VS_TIMER | IRQ_VS_SOFT | IRQ_VS_EXT) {
                        cause -= 1;
                    }
                    write_gva = false;
                } else if st.virt() {
                    // Trap into HS mode, from virt.
                    swap_hypervisor_regs(&mut st);
                    st.hstatus = set_field(st.hstatus, HSTATUS_SPVP, st.priv_lvl);
                    st.hstatus = set_field(st.hstatus, HSTATUS_SPV, 1);
                    htval = st.guest_phys_fault_addr;
                    virt = false;
                } else {
                    // Trap into HS mode.
                    st.hstatus = set_field(st.hstatus, HSTATUS_SPV, 0);
                    htval = st.guest_phys_fault_addr;
                }
                st.hstatus = set_field(st.hstatus, HSTATUS_GVA, u64::from(write_gva));
            }
            let mut s = st.mstatus;
            s = set_field(s, MSTATUS_SPIE, get_field(s, MSTATUS_SIE));
            s = set_field(s, MSTATUS_SPP, st.priv_lvl);
            s = set_field(s, MSTATUS_SIE, 0);
            st.mstatus = s;
            st.scause = cause | (u64::from(is_async) << 63);
            st.sepc = st.pc;
            st.stval = tval;
            st.htval = htval;
            st.htinst = tinst;
            let vec = if is_async && st.stvec & 3 == 1 { cause * 4 } else { 0 };
            st.pc = (st.stvec >> 2 << 2).wrapping_add(vec);
            flush = set_mode(&mut st, PRV_S, virt, self.host_ticks());
        } else {
            // Handle the trap in M mode.
            if st.has_h() {
                if st.virt() {
                    swap_hypervisor_regs(&mut st);
                }
                st.mstatus = set_field(st.mstatus, MSTATUS_MPV, st.virt_enabled);
                if st.virt() && tval != 0 {
                    st.mstatus = set_field(st.mstatus, MSTATUS_GVA, 1);
                }
                mtval2 = st.guest_phys_fault_addr;
                // Trapping to M mode, virt is disabled.
                virt = false;
            }
            let mut s = st.mstatus;
            s = set_field(s, MSTATUS_MPIE, get_field(s, MSTATUS_MIE));
            s = set_field(s, MSTATUS_MPP, st.priv_lvl);
            s = set_field(s, MSTATUS_MIE, 0);
            st.mstatus = s;
            st.mcause = cause | (u64::from(is_async) << 63);
            st.mtval2 = mtval2;
            st.mepc = st.pc;
            st.mtval = tval;
            st.mtinst = tinst;
            let vec = if is_async && st.mtvec & 3 == 1 { cause * 4 } else { 0 };
            st.pc = (st.mtvec >> 2 << 2).wrapping_add(vec);
            flush |= set_mode(&mut st, PRV_M, virt, self.host_ticks());
        }
        // The fault information is used up: a later trap without a two stage lookup must
        // not see it.
        st.two_stage_lookup = 0;
        st.two_stage_indirect_lookup = 0;
        st.store(cpu.env);
        if flush {
            cputlb::tlb_flush(cpu);
        }
        cpu.core.exception_index = -1;
    }
}

/// `riscv_cpu_pending_to_irq()`: the interrupt of `pending` to take first. Without Smaia
/// (for M mode, `extirq` `IRQ_M_EXT`) or Ssaia it is the lowest numbered one; with it the
/// one with the lowest priority in `prios`, where 0 means the default order around the
/// external interrupt `extirq`, whose default is `def_prio`. A tie goes to the higher
/// numbered interrupt.
pub(crate) fn pending_to_irq(
    cfg: &RiscvCfg,
    extirq: u32,
    def_prio: u8,
    pending: u64,
    prios: &[u64; 8],
) -> Option<u32> {
    if pending == 0 {
        return None;
    }
    let aia = if extirq == IRQ_M_EXT { cfg.ext_smaia } else { cfg.ext_ssaia };
    if !aia {
        return Some(pending.trailing_zeros());
    }
    let mut best = None;
    let mut best_prio = u32::MAX;
    for irq in pending.trailing_zeros()..64 {
        if (pending >> irq) & 1 == 0 {
            continue;
        }
        let mut prio = u32::from(iprio(prios, irq));
        if prio == 0 {
            prio = if irq == extirq {
                u32::from(def_prio)
            } else if default_priority(irq) < def_prio {
                1
            } else {
                u32::from(IPRIO_MMAXIPRIO)
            };
        }
        if prio <= best_prio {
            best = Some(irq);
            best_prio = prio;
        }
    }
    best
}

/// `riscv_cpu_mirq_pending()`: the M level interrupt `mtopi` reports, from the
/// `riscv_cpu_all_pending()` value `all`.
pub(crate) fn mirq_pending(cfg: &RiscvCfg, all: u64, st: &CpuRiscvState) -> Option<u32> {
    let irqs = all & !st.mideleg & !(MIP_SGEIP | VS_MODE_INTERRUPTS);
    pending_to_irq(cfg, IRQ_M_EXT, IPRIO_DEFAULT_M, irqs, &st.miprio)
}

/// `riscv_cpu_sirq_pending()`: the S level interrupt `stopi` reports.
pub(crate) fn sirq_pending(cfg: &RiscvCfg, all: u64, st: &CpuRiscvState) -> Option<u32> {
    let irqs = all & st.mideleg & !st.hideleg;
    let irqs_f = st.mvip & st.mvien & !st.mideleg & st.sie;
    pending_to_irq(cfg, IRQ_S_EXT, IPRIO_DEFAULT_S, irqs | irqs_f, &st.siprio)
}

/// `riscv_cpu_vsirq_pending()`: the VS level interrupt, at its S level number.
pub(crate) fn vsirq_pending(cfg: &RiscvCfg, all: u64, st: &CpuRiscvState) -> Option<u32> {
    let irqs = vs_bits_down(all & st.mideleg & st.hideleg);
    let irqs_f_vs = st.hvip & st.hvien & !st.hideleg & st.vsie;
    pending_to_irq(cfg, IRQ_S_EXT, IPRIO_DEFAULT_S, irqs | irqs_f_vs, &st.hviprio)
}

/// `promote_load_fault()`: an AMO that faults on its load reports a store fault.
fn promote_load_fault(cause: i32) -> i32 {
    match cause {
        EXCP_LOAD_GUEST_ACCESS_FAULT => EXCP_STORE_GUEST_AMO_ACCESS_FAULT,
        EXCP_LOAD_ACCESS_FAULT => EXCP_STORE_AMO_ACCESS_FAULT,
        EXCP_LOAD_PAGE_FAULT => EXCP_STORE_PAGE_FAULT,
        EXCP_LOAD_ADDR_MIS => EXCP_STORE_AMO_ADDR_MIS,
        c => c,
    }
}

/// `riscv_transformed_insn()`: the transformed instruction `mtinst` and `htinst` report
/// for a load or store fault at `taddr` caused by `insn`. Compressed loads and stores are
/// expanded (with bit 1 clear to tell they were compressed), the immediate is cleared and
/// the rs1 field holds the offset of `taddr` from the base address for misaligned
/// accesses.
fn transformed_insn(st: &CpuRiscvState, insn: u64, taddr: u64) -> u64 {
    let ins = insn as u32;
    let bits = |pos: u32, len: u32| (ins >> pos) & ((1 << len) - 1);
    let mut xinsn: u32 = 0;
    let mut rs1 = 0usize;
    let mut imm: u64 = 0;
    let mut size: u64 = 0;
    let set_rd = |x: u32, v: u32| (x & !(0x1f << 7)) | ((v & 0x1f) << 7);
    let set_rs2 = |x: u32, v: u32| (x & !(0x1f << 20)) | ((v & 0x1f) << 20);
    if ins & 3 != 3 {
        let rs1s = 8 + bits(7, 3) as usize;
        let rs2s = 8 + bits(2, 3);
        let c_rd = bits(7, 5);
        let c_rs2 = bits(2, 5);
        let lw_imm = u64::from((bits(6, 1) << 2) | (bits(10, 3) << 3) | (bits(5, 1) << 6));
        let ld_imm = u64::from((bits(10, 3) << 3) | (bits(5, 2) << 6));
        let lwsp_imm = u64::from((bits(4, 3) << 2) | (bits(12, 1) << 5) | (bits(2, 2) << 6));
        let ldsp_imm = u64::from((bits(5, 2) << 3) | (bits(12, 1) << 5) | (bits(2, 3) << 6));
        let swsp_imm = u64::from((bits(9, 4) << 2) | (bits(7, 2) << 6));
        let sdsp_imm = u64::from((bits(10, 3) << 3) | (bits(7, 3) << 6));
        // OPC_RISC_FLD, LW, LD, FSD, SW and SD.
        const FLD: u32 = 0x3007;
        const LW: u32 = 0x2003;
        const LD: u32 = 0x3003;
        const FSD: u32 = 0x3027;
        const SW: u32 = 0x2023;
        const SD: u32 = 0x3023;
        match (bits(0, 2), bits(13, 3)) {
            // Quadrant 0: C.FLD, C.LW, C.LD, C.FSD, C.SW and C.SD.
            (0, 1) => (xinsn, rs1, imm, size) = (set_rd(FLD, rs2s), rs1s, ld_imm, 8),
            (0, 2) => (xinsn, rs1, imm, size) = (set_rd(LW, rs2s), rs1s, lw_imm, 4),
            (0, 3) => (xinsn, rs1, imm, size) = (set_rd(LD, rs2s), rs1s, ld_imm, 8),
            (0, 5) => (xinsn, rs1, imm, size) = (set_rs2(FSD, rs2s), rs1s, ld_imm, 8),
            (0, 6) => (xinsn, rs1, imm, size) = (set_rs2(SW, rs2s), rs1s, lw_imm, 4),
            (0, 7) => (xinsn, rs1, imm, size) = (set_rs2(SD, rs2s), rs1s, ld_imm, 8),
            // Quadrant 2: C.FLDSP, C.LWSP, C.LDSP, C.FSDSP, C.SWSP and C.SDSP.
            (2, 1) => (xinsn, rs1, imm, size) = (set_rd(FLD, c_rd), 2, ldsp_imm, 8),
            (2, 2) => (xinsn, rs1, imm, size) = (set_rd(LW, c_rd), 2, lwsp_imm, 4),
            (2, 3) => (xinsn, rs1, imm, size) = (set_rd(LD, c_rd), 2, ldsp_imm, 8),
            (2, 5) => (xinsn, rs1, imm, size) = (set_rs2(FSD, c_rs2), 2, sdsp_imm, 8),
            (2, 6) => (xinsn, rs1, imm, size) = (set_rs2(SW, c_rs2), 2, swsp_imm, 4),
            (2, 7) => (xinsn, rs1, imm, size) = (set_rs2(SD, c_rs2), 2, sdsp_imm, 8),
            _ => {}
        }
        // Bit 1 clear tells that the original instruction was 16 bits.
        xinsn &= !2;
    } else {
        let funct3 = bits(12, 3);
        match ins & 0x7f {
            // OPC_RISC_ATOMIC.
            0x2f => {
                (xinsn, rs1, size) = (ins, bits(15, 5) as usize, 1 << funct3);
            }
            // OPC_RISC_LOAD and OPC_RISC_FP_LOAD: the I immediate is cleared.
            0x03 | 0x07 => {
                xinsn = ins & 0x000f_ffff;
                rs1 = bits(15, 5) as usize;
                imm = (i64::from(ins as i32) >> 20) as u64;
                size = 1 << funct3;
            }
            // OPC_RISC_STORE and OPC_RISC_FP_STORE: the S immediate is cleared.
            0x23 | 0x27 => {
                xinsn = ins & 0x01ff_f07f;
                rs1 = bits(15, 5) as usize;
                imm = u64::from(bits(7, 5)) | (((i64::from(ins as i32) >> 25) as u64) << 5);
                size = 1 << funct3;
            }
            // OPC_RISC_HLVHSV: QEMU takes the size from funct7 with one shift too many.
            0x73 if funct3 == 4 => {
                xinsn = ins;
                rs1 = bits(15, 5) as usize;
                size = 1 << (1u64 << ((bits(25, 7) >> 1) & 3));
            }
            _ => {}
        }
    }
    if size != 0 {
        let off = taddr.wrapping_sub(st.x(rs1).wrapping_add(imm)) & (size - 1);
        xinsn = (xinsn & !(0x1f << 15)) | ((off as u32 & 0x1f) << 15);
    }
    u64::from(xinsn)
}

/// `riscv_cpu_swap_hypervisor_regs()`: exchange the HS mode supervisor registers with the
/// VS mode ones, when V is about to change.
pub(crate) fn swap_hypervisor_regs(st: &mut CpuRiscvState) {
    let mut mask = MSTATUS_MXR
        | MSTATUS_SUM
        | MSTATUS_SPP
        | MSTATUS_SPIE
        | MSTATUS_SIE
        | MSTATUS64_UXL
        | MSTATUS_VS;
    if st.misa & RVF != 0 {
        mask |= MSTATUS_FS;
    }
    if st.virt() {
        // V is 1 and about to become 0.
        st.vsstatus = st.mstatus & mask;
        st.mstatus = (st.mstatus & !mask) | st.mstatus_hs;
        st.vstvec = std::mem::replace(&mut st.stvec, st.stvec_hs);
        st.vsscratch = std::mem::replace(&mut st.sscratch, st.sscratch_hs);
        st.vsepc = std::mem::replace(&mut st.sepc, st.sepc_hs);
        st.vscause = std::mem::replace(&mut st.scause, st.scause_hs);
        st.vstval = std::mem::replace(&mut st.stval, st.stval_hs);
        st.vsatp = std::mem::replace(&mut st.satp, st.satp_hs);
    } else {
        // V is 0 and about to become 1. Only the swapped bits of vsstatus reach mstatus;
        // see the module doc.
        st.mstatus_hs = st.mstatus & mask;
        st.mstatus = (st.mstatus & !mask) | (st.vsstatus & mask);
        st.stvec_hs = std::mem::replace(&mut st.stvec, st.vstvec);
        st.sscratch_hs = std::mem::replace(&mut st.sscratch, st.vsscratch);
        st.sepc_hs = std::mem::replace(&mut st.sepc, st.vsepc);
        st.scause_hs = std::mem::replace(&mut st.scause, st.vscause);
        st.stval_hs = std::mem::replace(&mut st.stval, st.vstval);
        st.satp_hs = std::mem::replace(&mut st.satp, st.vsatp);
    }
}

/// `riscv_cpu_set_mode()`: change the privilege level and, with H, the virtualization mode, at host
/// tick `now` for the per mode counts of the PMU. The load reservation is dropped, so that a
/// reservation placed in one context cannot make an SC in another succeed. Gives whether V changed,
/// in which case the caller must flush the TLB as QEMU does.
#[must_use]
pub(crate) fn set_mode(st: &mut CpuRiscvState, newpriv: u64, virt: bool, now: u64) -> bool {
    let virt = virt && st.has_h();
    if newpriv != st.priv_lvl || virt != st.virt() {
        pmu::update_fixed_ctrs(st, now, newpriv, virt);
    }
    st.priv_lvl = newpriv;
    st.load_res = u64::MAX;
    if !st.has_h() {
        return false;
    }
    let flush = st.virt() != virt;
    st.virt_enabled = u64::from(virt);
    flush
}

/// `env->two_stage_lookup = mmuidx_2stage(mmu_idx)` and `two_stage_indirect_lookup =
/// false`, for the faults that do not come from a page walk.
fn set_fault_lookup(env: &mut [u8], mmu_idx: usize) {
    let two_stage = u64::from(mmu_idx & MMU_2STAGE_BIT != 0);
    st64(env, crate::cpu::TWO_STAGE_LOOKUP, two_stage);
    st64(env, crate::cpu::TWO_STAGE_INDIRECT_LOOKUP, 0);
}

/// The [`Riscv`] behind a vCPU's ops.
pub(crate) fn riscv_of(ops: &Arc<dyn CpuOps>) -> &Riscv {
    ops.as_any().and_then(|a| a.downcast_ref::<Riscv>()).expect("the vCPU is a RISC-V vCPU")
}

/// Read a little endian `u64` from `env`.
pub(crate) fn ld64(env: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(env[off..off + 8].try_into().expect("8 bytes"))
}

/// Write a little endian `u64` to `env`.
pub(crate) fn st64(env: &mut [u8], off: usize, v: u64) {
    env[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

/// `riscv_env_mmu_index()`: the MMU index of fetches (`ifetch`) or data accesses, from the
/// privilege level, the virtualization mode and `mstatus` (or `vsstatus` for an M mode
/// access with MPRV into a guest).
pub(crate) fn mmu_index_of(
    priv_lvl: u64,
    virt: bool,
    mstatus: u64,
    vsstatus: u64,
    ifetch: bool,
) -> usize {
    let mut mode = priv_lvl;
    let mut virt = virt;
    if !ifetch {
        // riscv_cpu_eff_priv().
        let mut modified = false;
        if mode == PRV_M && mstatus & MSTATUS_MPRV != 0 {
            mode = get_field(mstatus, MSTATUS_MPP);
            virt = mstatus & MSTATUS_MPV != 0 && mode != PRV_M;
            modified = true;
        }
        let status = if modified && virt { vsstatus } else { mstatus };
        if mode == PRV_S && status & MSTATUS_SUM != 0 {
            mode = MMU_IDX_S_SUM as u64;
        }
    }
    mode as usize | if virt { MMU_2STAGE_BIT } else { 0 }
}

/// [`mmu_index_of`] for the state in `env`.
pub(crate) fn mmu_index(env: &[u8], ifetch: bool) -> usize {
    mmu_index_of(
        ld64(env, PRIV),
        ld64(env, VIRT_ENABLED) != 0,
        ld64(env, MSTATUS),
        ld64(env, VSSTATUS),
        ifetch,
    )
}

/// [`mmu_index_of`] for `st`.
pub(crate) fn mmu_index_st(st: &CpuRiscvState, ifetch: bool) -> usize {
    mmu_index_of(st.priv_lvl, st.virt(), st.mstatus, st.vsstatus, ifetch)
}

/// The TB flags of a vCPU, `riscv_get_tb_cpu_state()` cut down to what this front end
/// reads: the data MMU index, the privilege level, `mstatus.FS` and `mstatus.VS` (the
/// lower of the guest's and the hypervisor's under V=1), the vector state, V and the
/// pointer mask of data accesses. There is no `VL_EQ_VLMAX`, which QEMU only reads to use
/// gvec.
pub(crate) fn tb_flags(cfg: &RiscvCfg, env: &[u8]) -> u32 {
    let priv_lvl = ld64(env, PRIV);
    let mstatus = ld64(env, MSTATUS);
    let virt = ld64(env, VIRT_ENABLED) != 0;
    let mem_idx = mmu_index(env, false) as u32;
    let mut fs = get_field(mstatus, MSTATUS_FS) as u32;
    let mut vs = get_field(mstatus, MSTATUS_VS) as u32;
    if virt {
        let hs = ld64(env, MSTATUS_HS);
        fs = fs.min(get_field(hs, MSTATUS_FS) as u32);
        vs = vs.min(get_field(hs, MSTATUS_VS) as u32);
    }
    let mut flags =
        mem_idx | ((priv_lvl as u32) << TB_PRIV_SHIFT) | (fs << TB_FS_SHIFT) | (vs << TB_VS_SHIFT);
    if virt {
        flags |= TB_VIRT;
    }
    if cfg.ext_zve32x {
        let vtype = ld64(env, VTYPE);
        flags |= ((vtype & VTYPE_VLMUL) as u32) << TB_LMUL_SHIFT;
        flags |= (get_field(vtype, VTYPE_VSEW) as u32) << TB_SEW_SHIFT;
        if ld64(env, VILL) != 0 {
            flags |= TB_VILL;
        }
        if ld64(env, VSTART) == 0 {
            flags |= TB_VSTART_EQ_ZERO;
        }
        if vtype & VTYPE_VTA != 0 {
            flags |= TB_VTA;
        }
        if vtype & VTYPE_VMA != 0 {
            flags |= TB_VMA;
        }
    } else {
        flags |= TB_VILL;
    }
    if pm::any(cfg) {
        flags |= pm::data_mask(cfg, &pm::PmState::from_env(env)).tb_flags();
    }
    flags
}

impl CpuOps for Riscv {
    fn translate_code(&self, cpu: &mut Cpu<'_>, tb: &mut TbBuild) -> Result<(), CpuLoopExit> {
        let mut dc = translate::DisasContext::new(self.semihosting(), self.xlrbr(), self.cfg);
        translator_loop(cpu, tb, &mut dc)
    }

    fn get_tb_cpu_state(&self, cpu: &Cpu<'_>) -> TbCpuState {
        let flags = tb_flags(&self.cfg, cpu.env);
        TbCpuState { pc: ld64(cpu.env, PC), flags, cflags: 0, cs_base: 0 }
    }

    fn restore_state_to_opc(&self, cpu: &mut Cpu<'_>, _tb: &Tb, data: &[u64; 3]) {
        // riscv_restore_state_to_opc().
        st64(cpu.env, PC, data[0]);
        st64(cpu.env, crate::cpu::BINS, data[1]);
        st64(cpu.env, crate::cpu::EXCP_UW2, data[2]);
    }

    fn set_pc(&self, cpu: &mut Cpu<'_>, pc: u64) {
        st64(cpu.env, PC, pc);
    }

    fn get_pc(&self, cpu: &Cpu<'_>) -> u64 {
        ld64(cpu.env, PC)
    }

    fn cpu_exec_interrupt(&self, cpu: &mut Cpu<'_>, interrupt_request: u32) -> bool {
        // riscv_cpu_exec_interrupt().
        if interrupt_request & interrupt::HARD == 0 {
            return false;
        }
        let shared = Arc::clone(cpu.core.shared());
        // Not with_lines(): that kicks the vCPU whenever mip is nonzero, and a kick from
        // here would bring the vCPU straight back without running a single instruction.
        let fired = {
            let (mut g, i) = self.lines(shared.cpu_index);
            let fired = std::mem::take(&mut g[i].pmu_timer);
            if fired && !g[i].hard() {
                shared.reset_interrupt(interrupt::HARD);
            }
            fired
        };
        if fired {
            self.pmu_timer_cb(cpu, &shared);
        }
        let st = CpuRiscvState::load(cpu.env);
        let cpu_index = shared.cpu_index;
        match self.local_irq_pending(cpu_index, &st) {
            Some(irq) => {
                cpu.core.exception_index = EXCP_INT_FLAG | irq as i32;
                self.do_interrupt_riscv(cpu);
                true
            }
            None => false,
        }
    }

    fn do_interrupt(&self, cpu: &mut Cpu<'_>) {
        self.do_interrupt_riscv(cpu);
    }

    fn has_work(&self, cpu: &Cpu<'_>) -> bool {
        // riscv_cpu_has_work(): WFI ignores the privilege level and the delegation, but
        // respects the individual enables.
        let mie = ld64(cpu.env, crate::cpu::MIE);
        let i = cpu.core.shared().cpu_index;
        // A fired PMU timer wakes the vCPU to run its callback, which may raise LCOFIP.
        if self.all_pending(i, mie) != 0 || self.lines(i).0[i].pmu_timer {
            return true;
        }
        // The interrupts injected through mvip and hvip, which only exist with the AIA.
        if !self.cfg.ext_smaia && !self.cfg.ext_ssaia {
            return false;
        }
        let st = CpuRiscvState::load(cpu.env);
        st.mvip & st.mvien & !st.mideleg & st.sie != 0
            || st.hvip & st.hvien & !st.hideleg & st.vsie != 0
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
        ptw::tlb_fill(cpu, addr, size, access_type, mmu_idx, probe, ra)
    }

    fn do_unaligned_access(
        &self,
        cpu: &mut Cpu<'_>,
        addr: u64,
        access_type: MmuAccessType,
        mmu_idx: usize,
        ra: Ra,
    ) -> CpuLoopExit {
        // riscv_cpu_do_unaligned_access().
        let excp = match access_type {
            MmuAccessType::InstFetch => EXCP_INST_ADDR_MIS,
            MmuAccessType::DataLoad => EXCP_LOAD_ADDR_MIS,
            MmuAccessType::DataStore => EXCP_STORE_AMO_ADDR_MIS,
        };
        set_fault_lookup(cpu.env, mmu_idx);
        st64(cpu.env, crate::cpu::BADADDR, addr);
        cpu.raise_exception(excp, ra)
    }

    fn do_transaction_failed(
        &self,
        cpu: &mut Cpu<'_>,
        _physaddr: u64,
        addr: u64,
        _size: usize,
        access_type: MmuAccessType,
        mmu_idx: usize,
        _attrs: MemTxAttrs,
        _response: MemTxResult,
        ra: Ra,
    ) -> Result<(), CpuLoopExit> {
        // riscv_cpu_do_transaction_failed().
        let excp = match access_type {
            MmuAccessType::DataStore => EXCP_STORE_AMO_ACCESS_FAULT,
            MmuAccessType::DataLoad => EXCP_LOAD_ACCESS_FAULT,
            MmuAccessType::InstFetch => EXCP_INST_ACCESS_FAULT,
        };
        set_fault_lookup(cpu.env, mmu_idx);
        st64(cpu.env, crate::cpu::BADADDR, addr);
        Err(cpu.raise_exception(excp, ra))
    }

    fn mmu_index(&self, cpu: &Cpu<'_>, ifetch: bool) -> usize {
        mmu_index(cpu.env, ifetch)
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

/// The runtime configuration the RISC-V front end needs: 4 KiB pages and the eight MMU
/// indexes (U, S, S with SUM and M, each with and without two stage translation).
pub fn jit_config() -> JitConfig {
    JitConfig { page_bits: 12, nb_mmu_modes: NB_MMU_MODES, ..JitConfig::default() }
}

/// The runtime's built-in helpers plus every RISC-V helper.
pub fn helper_registry() -> HelperRegistry {
    let mut r = HelperRegistry::new();
    helpers::register(&mut r);
    r
}

/// An interpreter backend that knows the RISC-V helpers.
pub fn interp_backend() -> Arc<InterpBackend> {
    Arc::new(InterpBackend::with_helpers(helper_registry()))
}

/// A runtime for RISC-V guests running on the interpreter backend.
pub fn new_jit() -> Arc<Jit> {
    Jit::new(jit_config(), interp_backend())
}

/// Make a vCPU on `jit` running `ops`, with memory `as_` and registers from `state`.
pub fn create_vcpu(
    jit: &Arc<Jit>,
    ops: Arc<Riscv>,
    as_: Arc<AddressSpace>,
    state: &CpuRiscvState,
) -> Vcpu {
    let mut v = jit.create_vcpu(ops, as_, ENV_SIZE);
    let mut st = state.clone();
    pmp::update_rules(&mut st);
    st.store(&mut v.env);
    v
}

/// The registers of `v`.
pub fn save_vcpu(v: &Vcpu) -> CpuRiscvState {
    CpuRiscvState::load(&v.env)
}

/// Whether `v` is halted in WFI.
pub fn vcpu_halted(v: &Vcpu) -> bool {
    v.shared().halted.load(Ordering::Acquire) != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{IPRIO_DEFAULT_UPPER, MIP_MEIP, MIP_MSIP, MIP_MTIP, MIP_SSIP, set_iprio};

    #[test]
    fn default_priorities() {
        assert_eq!(default_priority(IRQ_M_EXT), IPRIO_DEFAULT_M);
        assert_eq!(default_priority(3), IPRIO_DEFAULT_M + 1);
        assert_eq!(default_priority(IRQ_S_EXT), IPRIO_DEFAULT_S);
        assert_eq!(default_priority(47), IPRIO_DEFAULT_UPPER);
        // Interrupts without a default order, and those past 63, come last.
        assert_eq!(default_priority(0), IPRIO_MMAXIPRIO);
        assert_eq!(default_priority(13), IPRIO_MMAXIPRIO);
        assert_eq!(default_priority(64), IPRIO_MMAXIPRIO);
    }

    #[test]
    fn pending_interrupt_order() {
        let mut cfg = RiscvCfg::default();
        let mut prios = [0u64; 8];
        let pending = MIP_MSIP | MIP_MTIP | MIP_MEIP;
        let m = |cfg: &RiscvCfg, pending: u64, prios: &[u64; 8]| {
            pending_to_irq(cfg, IRQ_M_EXT, IPRIO_DEFAULT_M, pending, prios)
        };
        // Without Smaia the lowest numbered interrupt goes first.
        assert_eq!(m(&cfg, 0, &prios), None);
        assert_eq!(m(&cfg, pending, &prios), Some(3));
        // With it, priority 0 puts the external interrupt at its default, the interrupts
        // that come before it by default at 1 and the others at 255; a tie goes to the
        // higher number.
        cfg.ext_smaia = true;
        assert_eq!(m(&cfg, pending, &prios), Some(IRQ_M_EXT));
        assert_eq!(m(&cfg, MIP_MSIP | MIP_MTIP, &prios), Some(7));
        assert_eq!(m(&cfg, MIP_MEIP | (1 << 47), &prios), Some(47));
        set_iprio(&mut prios, 7, 1);
        set_iprio(&mut prios, 3, 2);
        assert_eq!(m(&cfg, pending, &prios), Some(7));
        set_iprio(&mut prios, IRQ_M_EXT, 1);
        assert_eq!(m(&cfg, pending, &prios), Some(IRQ_M_EXT));
        // The S level follows Ssaia, not Smaia.
        let s = pending_to_irq(&cfg, IRQ_S_EXT, IPRIO_DEFAULT_S, MIP_SSIP | (1 << 5), &prios);
        assert_eq!(s, Some(1));
    }

    #[test]
    fn guest_external_lines() {
        let mut l = CpuLines { hgeip: 0b100, ..CpuLines::default() };
        assert_eq!(l.vsgein(), 0);
        l.vgein = 2;
        assert_eq!(l.vsgein(), MIP_VSEIP);
        assert_eq!(l.all_pending(MIP_VSEIP), MIP_VSEIP);
        assert!(l.hard());
        l.update_sgeip();
        assert_eq!(l.mip, 0);
        l.hgeie = 0b110;
        l.update_sgeip();
        assert_eq!(l.mip, MIP_SGEIP);
    }
}
