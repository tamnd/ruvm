// SPDX-License-Identifier: GPL-2.0-or-later

//! The RV64 TCG front end: the translator ([`translate`]), its helpers, the CSRs, PMP, the
//! page walk and the [`CpuOps`] glue, after QEMU's `target/riscv/tcg`, `cpu_helper.c`,
//! `csr.c`, `pmp.c` and `op_helper.c`.
//!
//! The `mip` register and the interrupt lines that drive it live in [`Riscv`] rather than
//! in `env`, because devices on other threads change them (QEMU changes `env->mip` under
//! the BQL). The vCPU thread reads them under the same lock.
//!
//! Differences from QEMU:
//!
//! - Only RV64 without the H and V extensions. `misa` and the `mstatus` UXL and SXL fields
//!   are read only, so the XLEN never changes.
//! - `mcycle` and `minstret` both count host time in nanoseconds since the CPU was made,
//!   where QEMU reads `cpu_get_host_ticks()`; the `mhpmcounter` registers hold their value
//!   but do not count.
//! - The page walk updates the A and D bits of a PTE with a plain store, where QEMU uses a
//!   compare and swap on RAM and restarts the walk if the PTE changed under it.
//! - There is no AIA, no Smrnmi, Smdbltrp, Ssdbltrp, Smctr or control flow integrity, no
//!   pointer masking and no icount.
//! - The debug triggers (`tselect`, `tdata1` to `tdata3`) hold values but never fire.
//! - Conditional branches may continue the block on the fall-through path (a superblock),
//!   with the taken edge emitted out of line at the end of the block; QEMU ends the block
//!   at every branch.

mod csr;
mod fpu;
mod helpers;
mod pmp;
mod ptw;
mod semihost;
mod translate;
mod translate_fp;

use std::fmt;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use ruvm_jit::translate::TbBuild;
use ruvm_jit::{
    Cpu, CpuLoopExit, CpuOps, CpuShared, InterpBackend, Jit, JitConfig, MmuAccessType, Ra, Tb,
    TbCpuState, Vcpu, interrupt, translator_loop,
};
use ruvm_jit_interp::HelperRegistry;
use ruvm_mem::{AddressSpace, MemTxAttrs, MemTxResult};

use crate::cpu::{
    CpuRiscvState, ENV_SIZE, EXCP_BREAKPOINT, EXCP_ILLEGAL_INST, EXCP_INST_ACCESS_FAULT,
    EXCP_INST_ADDR_MIS, EXCP_INST_PAGE_FAULT, EXCP_INT_FLAG, EXCP_LOAD_ACCESS_FAULT,
    EXCP_LOAD_ADDR_MIS, EXCP_LOAD_PAGE_FAULT, EXCP_M_ECALL, EXCP_S_ECALL, EXCP_SEMIHOST,
    EXCP_STORE_AMO_ACCESS_FAULT, EXCP_STORE_AMO_ADDR_MIS, EXCP_STORE_PAGE_FAULT, EXCP_U_ECALL,
    IRQ_S_EXT, MENVCFG_STCE, MIP_SEIP, MIP_STIP, MSTATUS, MSTATUS_FS, MSTATUS_MIE, MSTATUS_MPIE,
    MSTATUS_MPP, MSTATUS_MPRV, MSTATUS_SIE, MSTATUS_SPIE, MSTATUS_SPP, MSTATUS_SUM, NB_MMU_MODES,
    PC, PRIV, PRV_M, PRV_S, UW2_ALWAYS_STORE_AMO, get_field, set_field,
};

pub use semihost::{ADP_STOPPED_APPLICATION_EXIT, SemihostingHost};

/// The TB flags: the MMU index of data accesses in bits 0 and 1.
pub const TB_MEM_IDX_MASK: u32 = 3;
/// The TB flags: the privilege level in bits 2 and 3.
pub const TB_PRIV_SHIFT: u32 = 2;
/// The TB flags: `mstatus.FS` in bits 4 and 5.
pub const TB_FS_SHIFT: u32 = 4;

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
}

/// The RV64 CPU: the [`CpuOps`] of vCPUs translated by the RISC-V front end.
pub struct Riscv {
    start: Instant,
    board: OnceLock<Arc<dyn RiscvBoard>>,
    semihost: Option<semihost::Semihosting>,
    xlrbr: bool,
    lines: Mutex<Vec<CpuLines>>,
}

impl fmt::Debug for Riscv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Riscv")
            .field("semihosting", &self.semihost.is_some())
            .field("xlrbr", &self.xlrbr)
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
            xlrbr: false,
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
    /// instructions.
    pub fn with_xlrbr(mut self, on: bool) -> Riscv {
        self.xlrbr = on;
        self
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
        self.xlrbr
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
        let raise = g[i].mip != 0;
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
            l.mip = (l.mip & !mask) | (value & mask);
            old
        })
    }

    /// `riscv_cpu_set_irq()`: drive the local interrupt input `irq` (an `IRQ_*` number) of
    /// the vCPU `shared` to `level`, as the interrupt controllers and timers do.
    pub fn set_irq(&self, shared: &CpuShared, irq: u32, level: bool) {
        assert!(irq < 64, "riscv: bad interrupt line {irq}");
        if irq == IRQ_S_EXT {
            self.with_lines(shared, |l| {
                l.external_seip = level;
                let v = level || l.software_seip;
                l.mip = (l.mip & !MIP_SEIP) | if v { MIP_SEIP } else { 0 };
            });
        } else {
            let bit = 1u64 << irq;
            self.update_mip(shared, bit, if level { bit } else { 0 });
        }
    }

    /// `rdtime_fn()`: the board's `mtime`, if it has a timer.
    pub(crate) fn rdtime(&self) -> Option<u64> {
        self.board().and_then(|b| b.rdtime())
    }

    /// The value `mcycle` and `minstret` count from, `cpu_get_host_ticks()`.
    pub(crate) fn host_ticks(&self) -> u64 {
        u64::try_from(self.start.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    /// `riscv_timer_write_timecmp()` for `stimecmp`: raise STIP now if `timecmp` has
    /// passed, else clear it and arm the board's timer for when it passes. Nothing happens
    /// without a board timer or with `menvcfg.STCE` clear.
    pub(crate) fn write_timecmp(&self, shared: &CpuShared, menvcfg: u64, timecmp: u64) {
        let Some(board) = self.board() else { return };
        if menvcfg & MENVCFG_STCE == 0 {
            return;
        }
        let Some(now) = board.rdtime() else { return };
        if timecmp <= now {
            // A stimecmp value in the past raises the timer interrupt right away.
            self.update_mip(shared, MIP_STIP, MIP_STIP);
            return;
        }
        self.update_mip(shared, MIP_STIP, 0);
        if timecmp == u64::MAX {
            board.stimer_update(shared, None);
            return;
        }
        let diff = timecmp - now;
        let freq = u128::from(board.timebase_freq().max(1));
        let ns = u128::from(diff) * 1_000_000_000 / freq;
        let deadline = u64::try_from(ns)
            .ok()
            .and_then(|ns| Instant::now().checked_add(Duration::from_nanos(ns)));
        board.stimer_update(shared, deadline);
    }

    /// `riscv_timer_disable_timecmp()` for `stimecmp`: stop the timer and drop STIP, as
    /// clearing `menvcfg.STCE` does.
    pub(crate) fn disable_timecmp(&self, shared: &CpuShared) {
        if let Some(board) = self.board() {
            board.stimer_update(shared, None);
        }
        self.update_mip(shared, MIP_STIP, 0);
    }

    /// `riscv_stimer_cb()`: the deadline the board was given by
    /// [`RiscvBoard::stimer_update`] has passed; raise STIP.
    pub fn stimer_expired(&self, shared: &CpuShared) {
        self.update_mip(shared, MIP_STIP, MIP_STIP);
    }

    /// `riscv_cpu_all_pending()`: the enabled pending interrupts.
    fn all_pending(&self, cpu_index: usize, mie: u64) -> u64 {
        self.mip(cpu_index) & mie
    }

    /// `riscv_cpu_local_irq_pending()`: the interrupt to take now, if any.
    fn local_irq_pending(&self, cpu_index: usize, st: &CpuRiscvState) -> Option<u32> {
        let mie = st.priv_lvl < PRV_M || st.mstatus & MSTATUS_MIE != 0;
        let hsie = st.priv_lvl < PRV_S || (st.priv_lvl == PRV_S && st.mstatus & MSTATUS_SIE != 0);
        let pending = self.all_pending(cpu_index, st.mie);
        // M mode interrupts first, then HS mode ones; without AIA the lowest numbered
        // pending interrupt wins (riscv_cpu_pending_to_irq()).
        let irqs = pending & !st.mideleg;
        if mie && irqs != 0 {
            return Some(irqs.trailing_zeros());
        }
        let irqs = pending & st.mideleg;
        if hsie && irqs != 0 {
            return Some(irqs.trailing_zeros());
        }
        None
    }

    /// `riscv_cpu_do_interrupt()`: take the exception or interrupt in
    /// `cpu.core.exception_index`.
    fn do_interrupt_riscv(&self, cpu: &mut Cpu<'_>) {
        let index = cpu.core.exception_index;
        let mut st = CpuRiscvState::load(cpu.env);
        let always_storeamo = st.excp_uw2 & UW2_ALWAYS_STORE_AMO != 0;
        let is_async = index & EXCP_INT_FLAG != 0;
        let mut cause = (index & !EXCP_INT_FLAG) as u64;
        let deleg = if is_async { st.mideleg } else { st.medeleg };
        let mut tval = 0;

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
                EXCP_LOAD_ADDR_MIS
                | EXCP_STORE_AMO_ADDR_MIS
                | EXCP_LOAD_ACCESS_FAULT
                | EXCP_STORE_AMO_ACCESS_FAULT
                | EXCP_LOAD_PAGE_FAULT
                | EXCP_STORE_PAGE_FAULT => {
                    if always_storeamo {
                        cause = promote_load_fault(cause as i32) as u64;
                    }
                    tval = st.badaddr;
                }
                EXCP_INST_ADDR_MIS
                | EXCP_INST_ACCESS_FAULT
                | EXCP_INST_PAGE_FAULT
                | EXCP_BREAKPOINT => tval = st.badaddr,
                EXCP_ILLEGAL_INST => tval = st.bins,
                _ => {}
            }
            // ecall is dispatched as one cause so translate based on mode.
            if cause == EXCP_U_ECALL as u64 {
                if st.priv_lvl == PRV_M {
                    cause = EXCP_M_ECALL as u64;
                } else if st.priv_lvl == PRV_S {
                    cause = EXCP_S_ECALL as u64;
                }
            }
        }

        let to_s = st.priv_lvl <= PRV_S && cause < 64 && (deleg >> cause) & 1 != 0;
        if to_s {
            // Handle the trap in S mode.
            let mut s = st.mstatus;
            s = set_field(s, MSTATUS_SPIE, get_field(s, MSTATUS_SIE));
            s = set_field(s, MSTATUS_SPP, st.priv_lvl);
            s = set_field(s, MSTATUS_SIE, 0);
            st.mstatus = s;
            st.scause = cause | (u64::from(is_async) << 63);
            st.sepc = st.pc;
            st.stval = tval;
            let vec = if is_async && st.stvec & 3 == 1 { cause * 4 } else { 0 };
            st.pc = (st.stvec >> 2 << 2).wrapping_add(vec);
            set_mode(&mut st, PRV_S);
        } else {
            // Handle the trap in M mode.
            let mut s = st.mstatus;
            s = set_field(s, MSTATUS_MPIE, get_field(s, MSTATUS_MIE));
            s = set_field(s, MSTATUS_MPP, st.priv_lvl);
            s = set_field(s, MSTATUS_MIE, 0);
            st.mstatus = s;
            st.mcause = cause | (u64::from(is_async) << 63);
            st.mepc = st.pc;
            st.mtval = tval;
            let vec = if is_async && st.mtvec & 3 == 1 { cause * 4 } else { 0 };
            st.pc = (st.mtvec >> 2 << 2).wrapping_add(vec);
            set_mode(&mut st, PRV_M);
        }
        st.store(cpu.env);
        cpu.core.exception_index = -1;
    }
}

/// `promote_load_fault()`: an AMO that faults on its load reports a store fault.
fn promote_load_fault(cause: i32) -> i32 {
    match cause {
        EXCP_LOAD_ACCESS_FAULT => EXCP_STORE_AMO_ACCESS_FAULT,
        EXCP_LOAD_PAGE_FAULT => EXCP_STORE_PAGE_FAULT,
        EXCP_LOAD_ADDR_MIS => EXCP_STORE_AMO_ADDR_MIS,
        c => c,
    }
}

/// `riscv_cpu_set_mode()`: change the privilege level. The load reservation is dropped,
/// so that a reservation placed in one context cannot make an SC in another succeed.
pub(crate) fn set_mode(st: &mut CpuRiscvState, newpriv: u64) {
    st.priv_lvl = newpriv;
    st.load_res = u64::MAX;
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
/// privilege level and `mstatus`.
pub(crate) fn mmu_index(priv_lvl: u64, mstatus: u64, ifetch: bool) -> usize {
    let mut mode = priv_lvl;
    if !ifetch {
        if mode == PRV_M && mstatus & MSTATUS_MPRV != 0 {
            mode = get_field(mstatus, MSTATUS_MPP);
        }
        if mode == PRV_S && mstatus & MSTATUS_SUM != 0 {
            return crate::cpu::MMU_IDX_S_SUM;
        }
    }
    mode as usize
}

/// The TB flags of a vCPU, `riscv_get_tb_cpu_state()` cut down to what this front end
/// reads: the data MMU index, the privilege level and `mstatus.FS`.
pub(crate) fn tb_flags(priv_lvl: u64, mstatus: u64) -> u32 {
    let mem_idx = mmu_index(priv_lvl, mstatus, false) as u32;
    let fs = get_field(mstatus, MSTATUS_FS) as u32;
    mem_idx | ((priv_lvl as u32) << TB_PRIV_SHIFT) | (fs << TB_FS_SHIFT)
}

impl CpuOps for Riscv {
    fn translate_code(&self, cpu: &mut Cpu<'_>, tb: &mut TbBuild) -> Result<(), CpuLoopExit> {
        let mut dc = translate::DisasContext::new(self.semihosting(), self.xlrbr());
        translator_loop(cpu, tb, &mut dc)
    }

    fn get_tb_cpu_state(&self, cpu: &Cpu<'_>) -> TbCpuState {
        let flags = tb_flags(ld64(cpu.env, PRIV), ld64(cpu.env, MSTATUS));
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
        let st = CpuRiscvState::load(cpu.env);
        let cpu_index = cpu.core.shared().cpu_index;
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
        self.all_pending(cpu.core.shared().cpu_index, mie) != 0
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
        _mmu_idx: usize,
        ra: Ra,
    ) -> CpuLoopExit {
        // riscv_cpu_do_unaligned_access().
        let excp = match access_type {
            MmuAccessType::InstFetch => EXCP_INST_ADDR_MIS,
            MmuAccessType::DataLoad => EXCP_LOAD_ADDR_MIS,
            MmuAccessType::DataStore => EXCP_STORE_AMO_ADDR_MIS,
        };
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
        _mmu_idx: usize,
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
        st64(cpu.env, crate::cpu::BADADDR, addr);
        Err(cpu.raise_exception(excp, ra))
    }

    fn mmu_index(&self, cpu: &Cpu<'_>, ifetch: bool) -> usize {
        mmu_index(ld64(cpu.env, PRIV), ld64(cpu.env, MSTATUS), ifetch)
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

/// The runtime configuration the RISC-V front end needs: 4 KiB pages and the four MMU
/// indexes.
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
