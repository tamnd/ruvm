// SPDX-License-Identifier: GPL-2.0-or-later

//! The control and status registers, a port of QEMU's `target/riscv/csr.c` (with the
//! trigger CSRs of `debug.c` and the counters of `pmu.c`) for the CSRs of QEMU's default
//! `rv64` CPU without the H and V extensions.
//!
//! The CSRs are `fflags`, `frm` and `fcsr`; `cycle`, `time`, `instret` and
//! `hpmcounter3` to `hpmcounter31`; the machine information registers; `mstatus`,
//! `misa`, `medeleg`, `mideleg`, `mie`, `mtvec`, `mcounteren`, `menvcfg`,
//! `mcountinhibit`, `mhpmevent3` to `mhpmevent31`, `mscratch`, `mepc`, `mcause`,
//! `mtval`, `mip`, `mcycle`, `minstret` and `mhpmcounter3` to `mhpmcounter31`; `sstatus`,
//! `sie`, `stvec`, `scounteren`, `senvcfg`, `sscratch`, `sepc`, `scause`, `stval`, `sip`,
//! `stimecmp` and `satp`; the even `pmpcfg` registers up to `pmpcfg14` and `pmpaddr0` to
//! `pmpaddr63`; and `tselect`, `tdata1` to `tdata3`, `tinfo` and `mcontext`. Every
//! other CSR raises an illegal instruction exception, as the extensions behind them
//! (H, V, AIA, Smstateen, Sscofpmf, Zkr, Smepmp, Smrnmi, Smctr, control flow integrity,
//! pointer masking) are not in the model.
//!
//! Differences from QEMU:
//!
//! - `mstatus.UXL` cannot be written (QEMU lets M and S mode switch U mode to RV32).
//! - `hpmcounter3` to `hpmcounter18` and their machine mode aliases hold the value
//!   written but never count, whatever event `mhpmevent` selects.
//! - A `csrw` (rd = x0) reads the old value before the write, where QEMU skips the read.
//!   No read in this model has a side effect or can fail when the write would succeed,
//!   so the result is the same.
//! - Writing `tdata1` with a trigger type QEMU does not know (1 or 8 to 14) is ignored;
//!   QEMU 11.1 hits an assertion.
//! - Triggers hold their values but never fire, and an instruction count trigger does
//!   not count.

use ruvm_jit::{Cpu, CpuShared, cputlb};

use super::{CpuLines, Riscv, pmp};
use crate::cpu::{
    COUNTEREN_CY, COUNTEREN_IR, COUNTEREN_TM, CpuRiscvState, EXCP_BREAKPOINT, EXCP_ILLEGAL_INST,
    EXCP_INST_ACCESS_FAULT, EXCP_INST_ADDR_MIS, EXCP_INST_PAGE_FAULT, EXCP_LOAD_ACCESS_FAULT,
    EXCP_LOAD_ADDR_MIS, EXCP_LOAD_PAGE_FAULT, EXCP_S_ECALL, EXCP_STORE_AMO_ACCESS_FAULT,
    EXCP_STORE_AMO_ADDR_MIS, EXCP_STORE_PAGE_FAULT, EXCP_U_ECALL, FFLAGS_MASK, M_MODE_INTERRUPTS,
    MENVCFG_ADUE, MENVCFG_CBCFE, MENVCFG_CBIE, MENVCFG_CBZE, MENVCFG_FIOM, MENVCFG_STCE,
    MIP_LCOFIP, MIP_SEIP, MIP_SGEIP, MIP_SSIP, MIP_STIP, MIP_VS_BITS, MSTATUS_FS, MSTATUS_MIE,
    MSTATUS_MPIE, MSTATUS_MPP, MSTATUS_MPRV, MSTATUS_MXR, MSTATUS_SIE, MSTATUS_SPIE, MSTATUS_SPP,
    MSTATUS_SUM, MSTATUS_TSR, MSTATUS_TVM, MSTATUS_TW, MSTATUS64_UXL, NUM_TRIGGERS, PMU_AVAIL_CTRS,
    PRV_M, PRV_S, PRV_U, S_MODE_INTERRUPTS, SATP64_ASID, SATP64_MODE, SATP64_PPN, SSTATUS_MASK,
    VM_MBARE, VM_SV39, VM_SV48, VM_SV57, add_status_sd, get_field, set_field,
};

// The CSR numbers, from `cpu_bits.h`.
const CSR_FFLAGS: u32 = 0x001;
const CSR_FRM: u32 = 0x002;
const CSR_FCSR: u32 = 0x003;
const CSR_CYCLE: u32 = 0xc00;
const CSR_TIME: u32 = 0xc01;
const CSR_INSTRET: u32 = 0xc02;
const CSR_HPMCOUNTER31: u32 = 0xc1f;
const CSR_SSTATUS: u32 = 0x100;
const CSR_SIE: u32 = 0x104;
const CSR_STVEC: u32 = 0x105;
const CSR_SCOUNTEREN: u32 = 0x106;
const CSR_SENVCFG: u32 = 0x10a;
const CSR_SSCRATCH: u32 = 0x140;
const CSR_SEPC: u32 = 0x141;
const CSR_SCAUSE: u32 = 0x142;
const CSR_STVAL: u32 = 0x143;
const CSR_SIP: u32 = 0x144;
const CSR_STIMECMP: u32 = 0x14d;
const CSR_SATP: u32 = 0x180;
const CSR_MSTATUS: u32 = 0x300;
const CSR_MISA: u32 = 0x301;
const CSR_MEDELEG: u32 = 0x302;
const CSR_MIDELEG: u32 = 0x303;
const CSR_MIE: u32 = 0x304;
const CSR_MTVEC: u32 = 0x305;
const CSR_MCOUNTEREN: u32 = 0x306;
const CSR_MENVCFG: u32 = 0x30a;
const CSR_MCOUNTINHIBIT: u32 = 0x320;
const CSR_MHPMEVENT3: u32 = 0x323;
const CSR_MHPMEVENT31: u32 = 0x33f;
const CSR_MSCRATCH: u32 = 0x340;
const CSR_MEPC: u32 = 0x341;
const CSR_MCAUSE: u32 = 0x342;
const CSR_MTVAL: u32 = 0x343;
const CSR_MIP: u32 = 0x344;
const CSR_PMPCFG0: u32 = 0x3a0;
const CSR_PMPCFG15: u32 = 0x3af;
const CSR_PMPADDR0: u32 = 0x3b0;
const CSR_PMPADDR63: u32 = 0x3ef;
const CSR_TSELECT: u32 = 0x7a0;
const CSR_TDATA1: u32 = 0x7a1;
const CSR_TDATA2: u32 = 0x7a2;
const CSR_TDATA3: u32 = 0x7a3;
const CSR_TINFO: u32 = 0x7a4;
const CSR_MCONTEXT: u32 = 0x7a8;
const CSR_MCYCLE: u32 = 0xb00;
const CSR_MINSTRET: u32 = 0xb02;
const CSR_MHPMCOUNTER3: u32 = 0xb03;
const CSR_MHPMCOUNTER31: u32 = 0xb1f;
const CSR_MVENDORID: u32 = 0xf11;
const CSR_MARCHID: u32 = 0xf12;
const CSR_MIMPID: u32 = 0xf13;
const CSR_MHARTID: u32 = 0xf14;
const CSR_MCONFIGPTR: u32 = 0xf15;

/// `RISCV_CPU_MARCHID`.
const MARCHID: u64 = 42;

/// `LOCAL_INTERRUPTS`: interrupts 16 and up.
const LOCAL_INTERRUPTS: u64 = !0xffff;
/// `HS_MODE_INTERRUPTS`.
const HS_MODE_INTERRUPTS: u64 = MIP_SGEIP | MIP_VS_BITS;
/// `delegable_ints`.
const DELEGABLE_INTS: u64 = S_MODE_INTERRUPTS | MIP_VS_BITS | MIP_LCOFIP;
/// `all_ints`.
const ALL_INTS: u64 = M_MODE_INTERRUPTS | S_MODE_INTERRUPTS | HS_MODE_INTERRUPTS | LOCAL_INTERRUPTS;
/// `mvip_writable_mask`.
const MVIP_WRITABLE_MASK: u64 = MIP_SSIP | MIP_STIP | MIP_SEIP | LOCAL_INTERRUPTS;
/// `sip_writable_mask`.
const SIP_WRITABLE_MASK: u64 = MIP_SSIP | LOCAL_INTERRUPTS;

/// `RISCV_EXCP_VS_ECALL`.
const EXCP_VS_ECALL: i32 = 10;
/// `RISCV_EXCP_SW_CHECK`.
const EXCP_SW_CHECK: i32 = 18;
/// `RISCV_EXCP_INST_GUEST_PAGE_FAULT`.
const EXCP_INST_GUEST_PAGE_FAULT: i32 = 20;
/// `RISCV_EXCP_LOAD_GUEST_ACCESS_FAULT`.
const EXCP_LOAD_GUEST_ACCESS_FAULT: i32 = 21;
/// `RISCV_EXCP_VIRT_INSTRUCTION_FAULT`.
const EXCP_VIRT_INSTRUCTION_FAULT: i32 = 22;
/// `RISCV_EXCP_STORE_GUEST_AMO_ACCESS_FAULT`.
const EXCP_STORE_GUEST_AMO_ACCESS_FAULT: i32 = 23;

/// `DELEGABLE_EXCPS`: the `medeleg` bits that can be set. QEMU keeps the hypervisor
/// causes writable even without H.
const DELEGABLE_EXCPS: u64 = (1 << EXCP_INST_ADDR_MIS)
    | (1 << EXCP_INST_ACCESS_FAULT)
    | (1 << EXCP_ILLEGAL_INST)
    | (1 << EXCP_BREAKPOINT)
    | (1 << EXCP_LOAD_ADDR_MIS)
    | (1 << EXCP_LOAD_ACCESS_FAULT)
    | (1 << EXCP_STORE_AMO_ADDR_MIS)
    | (1 << EXCP_STORE_AMO_ACCESS_FAULT)
    | (1 << EXCP_U_ECALL)
    | (1 << EXCP_S_ECALL)
    | (1 << EXCP_VS_ECALL)
    | (1 << EXCP_INST_PAGE_FAULT)
    | (1 << EXCP_LOAD_PAGE_FAULT)
    | (1 << EXCP_STORE_PAGE_FAULT)
    | (1 << EXCP_SW_CHECK)
    | (1 << EXCP_INST_GUEST_PAGE_FAULT)
    | (1 << EXCP_LOAD_GUEST_ACCESS_FAULT)
    | (1 << EXCP_VIRT_INSTRUCTION_FAULT)
    | (1 << EXCP_STORE_GUEST_AMO_ACCESS_FAULT);

/// The `mstatus` bits a write changes: `write_mstatus()` with F and without V, Smdbltrp,
/// Ssdbltrp or Zicfilp.
const MSTATUS_WRITE_MASK: u64 = MSTATUS_SIE
    | MSTATUS_SPIE
    | MSTATUS_MIE
    | MSTATUS_MPIE
    | MSTATUS_SPP
    | MSTATUS_MPRV
    | MSTATUS_SUM
    | MSTATUS_MPP
    | MSTATUS_MXR
    | MSTATUS_TVM
    | MSTATUS_TSR
    | MSTATUS_TW
    | MSTATUS_FS;

/// The `menvcfg` bits a write changes: Sstc and Svadu are present, Svpbmt, Smcdeleg and
/// Ssdbltrp are not.
const MENVCFG_WRITE_MASK: u64 =
    MENVCFG_FIOM | MENVCFG_CBIE | MENVCFG_CBCFE | MENVCFG_CBZE | MENVCFG_STCE | MENVCFG_ADUE;
/// The `senvcfg` bits a write changes.
const SENVCFG_WRITE_MASK: u64 = MENVCFG_FIOM | MENVCFG_CBIE | MENVCFG_CBCFE | MENVCFG_CBZE;

/// The counters `mcounteren` and `scounteren` can enable.
const COUNTEREN_MASK: u64 = PMU_AVAIL_CTRS | COUNTEREN_CY | COUNTEREN_TM | COUNTEREN_IR;

/// `MHPMEVENT_BIT_VSINH` and `MHPMEVENT_BIT_VUINH`, which need H.
const MHPMEVENT_VINH: u64 = (1 << 59) | (1 << 58);

/// `MCONTEXT64`.
const MCONTEXT64: u64 = 0x1fff;

/// The read only `pmpaddr` bits on RV64, 54 and up.
const PMPADDR_MASK: u64 = (1 << 54) - 1;

// The debug triggers, from `debug.h`.
/// `TRIGGER_TYPE_AD_MATCH`.
const TRIGGER_TYPE_AD_MATCH: u64 = 2;
/// `TRIGGER_TYPE_INST_CNT`.
const TRIGGER_TYPE_INST_CNT: u64 = 3;
/// `TRIGGER_TYPE_AD_MATCH6`.
const TRIGGER_TYPE_AD_MATCH6: u64 = 6;
/// The `TYPE2_*` and `TYPE6_*` LOAD, STORE, EXEC, U, S and M bits.
const TYPE_MODE_RWX: u64 = 0x7 | (1 << 3) | (1 << 4) | (1 << 6);
/// `TYPE2_SIZELO`.
const TYPE2_SIZELO: u64 = 3 << 16;
/// `TYPE2_SIZEHI`.
const TYPE2_SIZEHI: u64 = 3 << 21;
/// `TYPE6_SIZE`.
const TYPE6_SIZE: u64 = 0xf << 16;
/// `TYPE6_VU` and `TYPE6_VS`.
const TYPE6_VU_VS: u64 = (1 << 23) | (1 << 24);
/// The `ITRIGGER_*` U, S, M, VU, VS and COUNT bits.
const ITRIGGER_KEEP: u64 = (1 << 6) | (1 << 7) | (1 << 9) | (1 << 25) | (1 << 26) | (0x3fff << 10);
/// `TEXTRA64_MHVALUE`.
const TEXTRA64_MHVALUE: u64 = 0xfff8_0000_0000_0000;
/// `TEXTRA64_MHSELECT`.
const TEXTRA64_MHSELECT: u64 = 0x0007_0000_0000_0000;

/// What the CSRs reach outside `env`: the interrupt lines, the timer and the clock.
trait Hw {
    /// `env->rdtime_fn()`, if the board has a timer.
    fn rdtime(&self) -> Option<u64>;
    /// The value `mcycle` and `minstret` count from.
    fn host_ticks(&self) -> u64;
    /// Run `f` on the interrupt lines with their lock held, then update the interrupt
    /// request from `mip`.
    fn with_lines(&self, f: &mut dyn FnMut(&mut CpuLines) -> u64) -> u64;
    /// `riscv_timer_write_timecmp()` for `stimecmp`.
    fn write_timecmp(&self, menvcfg: u64, timecmp: u64);
    /// `riscv_timer_disable_timecmp()` for `stimecmp`.
    fn disable_timecmp(&self);
}

/// The [`Hw`] of a vCPU.
struct CpuHw<'a> {
    rv: &'a Riscv,
    shared: &'a CpuShared,
}

impl Hw for CpuHw<'_> {
    fn rdtime(&self) -> Option<u64> {
        self.rv.rdtime()
    }

    fn host_ticks(&self) -> u64 {
        self.rv.host_ticks()
    }

    fn with_lines(&self, f: &mut dyn FnMut(&mut CpuLines) -> u64) -> u64 {
        self.rv.with_lines(self.shared, f)
    }

    fn write_timecmp(&self, menvcfg: u64, timecmp: u64) {
        self.rv.write_timecmp(self.shared, menvcfg, timecmp);
    }

    fn disable_timecmp(&self) {
        self.rv.disable_timecmp(self.shared);
    }
}

/// The CSR file of one access: the registers in `env`, the outside world, and whether the
/// access has to flush the TLB.
struct Csrs<'h> {
    st: CpuRiscvState,
    hw: &'h dyn Hw,
    flush: bool,
}

/// `riscv_csrr()`: read CSR `csrno`, or the exception to raise.
pub(crate) fn csrr(cpu: &mut Cpu<'_>, csrno: u32) -> Result<u64, i32> {
    access(cpu, csrno, false, 0, 0)
}

/// `riscv_csrrw()`: write the bits of `new_value` in `write_mask` to CSR `csrno` and give
/// the old value, or the exception to raise. A zero mask still checks for write access.
pub(crate) fn csrrw(
    cpu: &mut Cpu<'_>,
    csrno: u32,
    new_value: u64,
    write_mask: u64,
) -> Result<u64, i32> {
    access(cpu, csrno, true, new_value, write_mask)
}

/// One CSR access of the vCPU `cpu`.
fn access(cpu: &mut Cpu<'_>, csrno: u32, write: bool, new: u64, mask: u64) -> Result<u64, i32> {
    let ops = cpu.ops();
    let shared = cpu.shared();
    let hw = CpuHw { rv: super::riscv_of(&ops), shared: &shared };
    let mut c = Csrs { st: CpuRiscvState::load(cpu.env), hw: &hw, flush: false };
    let r = c.rw(csrno, write, new, mask);
    if write {
        c.st.store(cpu.env);
    }
    if c.flush {
        cputlb::tlb_flush(cpu);
    }
    r
}

/// The counter index of `cycle` to `hpmcounter31` or `mcycle` to `mhpmcounter31`.
fn ctr_index(csrno: u32) -> usize {
    (csrno & 0x1f) as usize
}

/// `legalize_mpp()`: keep the old MPP if the new one is not a privilege level the hart
/// has (2 is reserved).
fn legalize_mpp(old_mpp: u64, val: u64) -> u64 {
    match get_field(val, MSTATUS_MPP) {
        PRV_M | PRV_S | PRV_U => val,
        _ => set_field(val, MSTATUS_MPP, old_mpp),
    }
}

/// `legalize_xatp()` without the flush: the new `satp`, or `None` to keep the old one.
fn legalize_satp(old: u64, val: u64) -> Option<u64> {
    let vm = get_field(val, SATP64_MODE);
    let valid = matches!(vm, VM_MBARE | VM_SV39 | VM_SV48 | VM_SV57);
    let changed = (val ^ old) & (SATP64_MODE | SATP64_ASID | SATP64_PPN) != 0;
    (valid && changed).then_some(val)
}

/// `access_size[size] != -1`: the trigger access sizes QEMU supports (any, 1, 2, 4 and
/// 8 bytes).
fn trigger_size_ok(size: u64) -> bool {
    matches!(size, 0..=3 | 5)
}

/// `type2_mcontrol_validate()`.
fn type2_mcontrol_validate(ctrl: u64) -> u64 {
    let mut val = TRIGGER_TYPE_AD_MATCH << 60;
    let size = (get_field(ctrl, TYPE2_SIZEHI) << 2) | get_field(ctrl, TYPE2_SIZELO);
    if trigger_size_ok(size) {
        val |= ctrl & (TYPE2_SIZELO | TYPE2_SIZEHI);
    }
    val | (ctrl & TYPE_MODE_RWX)
}

/// `type6_mcontrol6_validate()`.
fn type6_mcontrol6_validate(ctrl: u64) -> u64 {
    let mut val = TRIGGER_TYPE_AD_MATCH6 << 60;
    if trigger_size_ok(get_field(ctrl, TYPE6_SIZE)) {
        val |= ctrl & TYPE6_SIZE;
    }
    val | (ctrl & (TYPE6_VU_VS | TYPE_MODE_RWX))
}

/// `itrigger_validate()`.
fn itrigger_validate(ctrl: u64) -> u64 {
    (TRIGGER_TYPE_INST_CNT << 60) | (ctrl & ITRIGGER_KEEP)
}

/// `textra_validate()`: only `mhselect` 0 and 4 are supported.
fn textra_validate(tdata3: u64) -> u64 {
    const MHSELECT_NO_RVH: [u64; 8] = [0, 0, 0, 0, 4, 4, 4, 4];
    let mhvalue = get_field(tdata3, TEXTRA64_MHVALUE);
    let mhselect = get_field(tdata3, TEXTRA64_MHSELECT) as usize;
    let textra = set_field(0, TEXTRA64_MHVALUE, mhvalue);
    set_field(textra, TEXTRA64_MHSELECT, MHSELECT_NO_RVH[mhselect])
}

/// `tdata_mapping[type][index]`: whether `tdata<index + 1>` exists for a trigger of
/// type `ttype`.
fn tdata_mapped(ttype: u64, index: usize) -> bool {
    match ttype {
        2 | 4 | 5 | 6 | 15 => true,
        3 => index != 1,
        7 => index == 0,
        _ => false,
    }
}

impl Csrs<'_> {
    /// `riscv_csrrw_check()` then `riscv_csrrw_do64()`.
    fn rw(&mut self, csrno: u32, write: bool, new: u64, mask: u64) -> Result<u64, i32> {
        self.check(csrno, write)?;
        match csrno {
            // The CSRs with a combined read-modify-write operation.
            CSR_MIDELEG => return Ok(self.rmw_mideleg(new, mask)),
            CSR_MIE => return Ok(self.rmw_mie(new, mask)),
            CSR_MIP => return Ok(self.rmw_mip(new, mask)),
            CSR_SIE => return Ok(self.rmw_sie(new, mask)),
            CSR_SIP => return Ok(self.rmw_sip(new, mask)),
            _ => {}
        }
        let old = self.read(csrno)?;
        if mask != 0 {
            self.write(csrno, (old & !mask) | (new & mask))?;
        }
        Ok(old)
    }

    /// `riscv_csrrw_check()`: whether the CSR exists and the hart may access it.
    fn check(&self, csrno: u32, write: bool) -> Result<(), i32> {
        if csrno > 0xfff {
            return Err(EXCP_ILLEGAL_INST);
        }
        let read_only = (csrno >> 10) & 3 == 3;
        if write && read_only {
            return Err(EXCP_ILLEGAL_INST);
        }
        if !self.predicate(csrno) {
            return Err(EXCP_ILLEGAL_INST);
        }
        let csr_priv = u64::from((csrno >> 8) & 3);
        if self.st.priv_lvl < csr_priv {
            return Err(EXCP_ILLEGAL_INST);
        }
        Ok(())
    }

    /// The `predicate` of `csr_ops[csrno]`: whether the CSR exists and the current state
    /// allows the access. CSRs without an entry do not exist.
    fn predicate(&self, csrno: u32) -> bool {
        match csrno {
            CSR_FFLAGS | CSR_FRM | CSR_FCSR => self.fs(),
            CSR_CYCLE..=CSR_HPMCOUNTER31 => self.ctr(csrno),
            CSR_MCYCLE | CSR_MINSTRET => true,
            CSR_MHPMCOUNTER3..=CSR_MHPMCOUNTER31 => PMU_AVAIL_CTRS & (1 << ctr_index(csrno)) != 0,
            CSR_MVENDORID..=CSR_MCONFIGPTR => true,
            CSR_MSTATUS | CSR_MISA | CSR_MEDELEG | CSR_MIDELEG | CSR_MIE | CSR_MTVEC
            | CSR_MCOUNTEREN | CSR_MENVCFG | CSR_MCOUNTINHIBIT => true,
            CSR_MHPMEVENT3..=CSR_MHPMEVENT31 => true,
            CSR_MSCRATCH | CSR_MEPC | CSR_MCAUSE | CSR_MTVAL | CSR_MIP => true,
            CSR_SSTATUS | CSR_SIE | CSR_STVEC | CSR_SCOUNTEREN | CSR_SENVCFG | CSR_SSCRATCH
            | CSR_SEPC | CSR_SCAUSE | CSR_STVAL | CSR_SIP => true,
            CSR_STIMECMP => self.sstc(),
            // satp(): S mode with mstatus.TVM set may not touch satp.
            CSR_SATP => !(self.st.priv_lvl == PRV_S && self.st.mstatus & MSTATUS_TVM != 0),
            // pmp(): the odd pmpcfg registers do not exist on RV64.
            CSR_PMPCFG0..=CSR_PMPCFG15 => (csrno - CSR_PMPCFG0) & 1 == 0,
            CSR_PMPADDR0..=CSR_PMPADDR63 => true,
            CSR_TSELECT | CSR_TDATA1 | CSR_TDATA2 | CSR_TDATA3 | CSR_TINFO | CSR_MCONTEXT => true,
            _ => false,
        }
    }

    /// `fs()`: the FP CSRs need `mstatus.FS` on.
    fn fs(&self) -> bool {
        self.st.mstatus & MSTATUS_FS != 0
    }

    /// `ctr()`: the unprivileged counters, enabled by `mcounteren` and `scounteren`.
    fn ctr(&self, csrno: u32) -> bool {
        let bit = 1u64 << ctr_index(csrno);
        // cycle, time and instret come with Zicntr, the others with the PMU.
        if csrno > CSR_INSTRET && PMU_AVAIL_CTRS & bit == 0 {
            return false;
        }
        if self.st.priv_lvl < PRV_M && self.st.mcounteren & bit == 0 {
            return false;
        }
        !(self.st.priv_lvl == PRV_U && self.st.scounteren & bit == 0)
    }

    /// `sstc()` for `stimecmp`.
    fn sstc(&self) -> bool {
        if self.hw.rdtime().is_none() {
            return false;
        }
        if self.st.priv_lvl == PRV_M {
            return true;
        }
        self.st.mcounteren & COUNTEREN_TM != 0 && self.st.menvcfg & MENVCFG_STCE != 0
    }

    /// The `read` operation of `csr_ops[csrno]`.
    fn read(&self, csrno: u32) -> Result<u64, i32> {
        let st = &self.st;
        Ok(match csrno {
            CSR_FFLAGS => st.fflags & FFLAGS_MASK,
            CSR_FRM => st.frm,
            CSR_FCSR => (st.fflags & FFLAGS_MASK) | (st.frm << 5),
            CSR_TIME => self.hw.rdtime().ok_or(EXCP_ILLEGAL_INST)?,
            CSR_CYCLE..=CSR_HPMCOUNTER31 | CSR_MCYCLE..=CSR_MHPMCOUNTER31 => {
                self.read_ctr(ctr_index(csrno))
            }
            CSR_MVENDORID | CSR_MIMPID | CSR_MCONFIGPTR => 0,
            CSR_MARCHID => MARCHID,
            CSR_MHARTID => st.mhartid,
            CSR_MSTATUS => add_status_sd(st.mstatus),
            CSR_MISA => st.misa,
            CSR_MEDELEG => st.medeleg,
            CSR_MTVEC => st.mtvec,
            CSR_MCOUNTEREN => st.mcounteren,
            CSR_MENVCFG => st.menvcfg,
            CSR_MCOUNTINHIBIT => st.mcountinhibit,
            CSR_MHPMEVENT3..=CSR_MHPMEVENT31 => st.mhpmevent[ctr_index(csrno)],
            CSR_MSCRATCH => st.mscratch,
            CSR_MEPC => st.mepc & !1,
            CSR_MCAUSE => st.mcause,
            CSR_MTVAL => st.mtval,
            CSR_SSTATUS => add_status_sd(st.mstatus & (SSTATUS_MASK | MSTATUS64_UXL)),
            CSR_STVEC => st.stvec,
            CSR_SCOUNTEREN => st.scounteren,
            CSR_SENVCFG => st.senvcfg,
            CSR_SSCRATCH => st.sscratch,
            CSR_SEPC => st.sepc & !1,
            CSR_SCAUSE => st.scause,
            CSR_STVAL => st.stval,
            CSR_STIMECMP => st.stimecmp,
            CSR_SATP => st.satp,
            CSR_PMPCFG0..=CSR_PMPCFG15 => pmp::pmpcfg_csr_read(st, (csrno - CSR_PMPCFG0) as usize),
            CSR_PMPADDR0..=CSR_PMPADDR63 => {
                pmp::pmpaddr_csr_read(st, (csrno - CSR_PMPADDR0) as usize) & PMPADDR_MASK
            }
            CSR_TSELECT => st.tselect,
            CSR_TDATA1 | CSR_TDATA2 | CSR_TDATA3 => {
                self.read_tdata((csrno - CSR_TDATA1) as usize)?
            }
            // tinfo_csr_read(): every trigger can be a type 2 or a type 6 trigger.
            CSR_TINFO => (1 << TRIGGER_TYPE_AD_MATCH) | (1 << TRIGGER_TYPE_AD_MATCH6),
            CSR_MCONTEXT => st.mcontext,
            _ => return Err(EXCP_ILLEGAL_INST),
        })
    }

    /// The `write` operation of `csr_ops[csrno]`, given the merged new value.
    fn write(&mut self, csrno: u32, val: u64) -> Result<(), i32> {
        match csrno {
            CSR_FFLAGS => {
                self.st.mstatus |= MSTATUS_FS;
                self.st.fflags = val & FFLAGS_MASK;
            }
            CSR_FRM => {
                self.st.mstatus |= MSTATUS_FS;
                self.st.frm = val & 7;
            }
            CSR_FCSR => {
                self.st.mstatus |= MSTATUS_FS;
                self.st.frm = (val >> 5) & 7;
                self.st.fflags = val & FFLAGS_MASK;
            }
            CSR_MCYCLE..=CSR_MHPMCOUNTER31 => self.write_ctr(ctr_index(csrno), val),
            CSR_MSTATUS => self.write_mstatus(val),
            // misa is not writable.
            CSR_MISA => {}
            CSR_MEDELEG => {
                self.st.medeleg = (self.st.medeleg & !DELEGABLE_EXCPS) | (val & DELEGABLE_EXCPS);
            }
            // Modes 2 and 3 are reserved; such a write is ignored.
            CSR_MTVEC => {
                if val & 3 < 2 {
                    self.st.mtvec = val;
                }
            }
            CSR_MCOUNTEREN => self.st.mcounteren = val & COUNTEREN_MASK,
            CSR_MENVCFG => self.write_menvcfg(val),
            CSR_MCOUNTINHIBIT => self.write_mcountinhibit(val),
            CSR_MHPMEVENT3..=CSR_MHPMEVENT31 => {
                self.st.mhpmevent[ctr_index(csrno)] = val & !MHPMEVENT_VINH;
            }
            CSR_MSCRATCH => self.st.mscratch = val,
            CSR_MEPC => self.st.mepc = val & !1,
            CSR_MCAUSE => self.st.mcause = val,
            CSR_MTVAL => self.st.mtval = val,
            CSR_SSTATUS => {
                let v = (self.st.mstatus & !SSTATUS_MASK) | (val & SSTATUS_MASK);
                self.write_mstatus(v);
            }
            CSR_STVEC => {
                if val & 3 < 2 {
                    self.st.stvec = val;
                }
            }
            CSR_SCOUNTEREN => self.st.scounteren = val & COUNTEREN_MASK,
            CSR_SENVCFG => {
                let m = SENVCFG_WRITE_MASK;
                self.st.senvcfg = (self.st.senvcfg & !m) | (val & m);
            }
            CSR_SSCRATCH => self.st.sscratch = val,
            CSR_SEPC => self.st.sepc = val & !1,
            CSR_SCAUSE => self.st.scause = val,
            CSR_STVAL => self.st.stval = val,
            CSR_STIMECMP => {
                self.st.stimecmp = val;
                self.hw.write_timecmp(self.st.menvcfg, val);
            }
            CSR_SATP => {
                if let Some(v) = legalize_satp(self.st.satp, val) {
                    self.flush = true;
                    self.st.satp = v;
                }
            }
            CSR_PMPCFG0..=CSR_PMPCFG15 => {
                let i = (csrno - CSR_PMPCFG0) as usize;
                self.flush |= pmp::pmpcfg_csr_write(&mut self.st, i, val);
            }
            CSR_PMPADDR0..=CSR_PMPADDR63 => {
                let i = (csrno - CSR_PMPADDR0) as usize;
                self.flush |= pmp::pmpaddr_csr_write(&mut self.st, i, val);
            }
            // tselect_csr_write(): a trigger that does not exist cannot be selected.
            CSR_TSELECT => {
                if val < NUM_TRIGGERS as u64 {
                    self.st.tselect = val;
                }
            }
            CSR_TDATA1 | CSR_TDATA2 | CSR_TDATA3 => {
                self.write_tdata((csrno - CSR_TDATA1) as usize, val)?;
            }
            CSR_MCONTEXT => self.st.mcontext = val & MCONTEXT64,
            // The read only CSRs, and tinfo whose writes are ignored.
            _ => {}
        }
        Ok(())
    }

    /// `write_mstatus()`.
    fn write_mstatus(&mut self, val: u64) {
        let mstatus = self.st.mstatus;
        let val = legalize_mpp(get_field(mstatus, MSTATUS_MPP), val);
        // MXR changes what a page permits; MPRV and SUM select another MMU index.
        if (val ^ mstatus) & MSTATUS_MXR != 0 {
            self.flush = true;
        }
        self.st.mstatus = (mstatus & !MSTATUS_WRITE_MASK) | (val & MSTATUS_WRITE_MASK);
    }

    /// `write_menvcfg()`, with `riscv_timer_stce_changed()` when STCE flips.
    fn write_menvcfg(&mut self, val: u64) {
        let m = MENVCFG_WRITE_MASK;
        let stce_changed = (self.st.menvcfg ^ val) & MENVCFG_STCE != 0;
        self.st.menvcfg = (self.st.menvcfg & !m) | (val & m);
        if stce_changed {
            if val & MENVCFG_STCE != 0 {
                self.hw.write_timecmp(self.st.menvcfg, self.st.stimecmp);
            } else {
                self.hw.disable_timecmp();
            }
        }
    }

    /// Whether counter `idx` follows the clock: `riscv_pmu_ctr_monitor_cycles()` and
    /// `riscv_pmu_ctr_monitor_instructions()`, true only for `mcycle` and `minstret`.
    fn ctr_counts(idx: usize) -> bool {
        idx == 0 || idx == 2
    }

    /// `riscv_pmu_read_ctr()`.
    fn read_ctr(&self, idx: usize) -> u64 {
        let val = self.st.mhpmcounter_val[idx];
        if self.st.mcountinhibit & (1 << idx) != 0 || !Self::ctr_counts(idx) {
            return val;
        }
        self.hw.host_ticks().wrapping_sub(self.st.mhpmcounter_prev[idx]).wrapping_add(val)
    }

    /// `riscv_pmu_write_ctr()`.
    fn write_ctr(&mut self, idx: usize, val: u64) {
        self.st.mhpmcounter_val[idx] = val;
        self.st.mhpmcounter_prev[idx] =
            if self.st.mcountinhibit & (1 << idx) == 0 && Self::ctr_counts(idx) {
                self.hw.host_ticks()
            } else {
                val
            };
    }

    /// `write_mcountinhibit()`: stopping a counter folds the time it ran into its value;
    /// starting it again restarts the count from now.
    fn write_mcountinhibit(&mut self, val: u64) {
        let present = PMU_AVAIL_CTRS | COUNTEREN_CY | COUNTEREN_IR;
        let updated = (self.st.mcountinhibit ^ val) & present;
        self.st.mcountinhibit = val & present;
        for idx in [0, 2] {
            if updated & (1 << idx) == 0 {
                continue;
            }
            let now = self.hw.host_ticks();
            if self.st.mcountinhibit & (1 << idx) == 0 {
                self.st.mhpmcounter_prev[idx] = now;
            } else {
                let prev = self.st.mhpmcounter_prev[idx];
                let v = self.st.mhpmcounter_val[idx];
                self.st.mhpmcounter_val[idx] = now.wrapping_sub(prev).wrapping_add(v);
            }
        }
    }

    /// `rmw_mideleg64()`.
    fn rmw_mideleg(&mut self, new: u64, wr_mask: u64) -> u64 {
        let m = wr_mask & DELEGABLE_INTS;
        let old = self.st.mideleg;
        self.st.mideleg = (old & !m) | (new & m);
        old
    }

    /// `rmw_mie64()`: without H the hypervisor interrupt enables stay clear.
    fn rmw_mie(&mut self, new: u64, wr_mask: u64) -> u64 {
        let m = wr_mask & ALL_INTS;
        let old = self.st.mie;
        self.st.mie = ((old & !m) | (new & m)) & !HS_MODE_INTERRUPTS;
        old
    }

    /// `rmw_sie64()`: `sie` shows the delegated bits of `mie`.
    fn rmw_sie(&mut self, new: u64, wr_mask: u64) -> u64 {
        let alias = (S_MODE_INTERRUPTS | LOCAL_INTERRUPTS) & self.st.mideleg;
        self.rmw_mie(new, wr_mask & alias) & alias
    }

    /// Whether STIP is driven by `stimecmp` alone for this access: Sstc with
    /// `menvcfg.STCE` set, from M mode.
    fn stip_from_sstc(&self) -> bool {
        self.st.priv_lvl == PRV_M && self.st.menvcfg & MENVCFG_STCE != 0
    }

    /// `rmw_mip64()`: the SEIP bit software writes is kept apart from the interrupt
    /// controller's input, and STIP belongs to Sstc when it is on.
    fn rmw_mip(&mut self, new: u64, wr_mask: u64) -> u64 {
        let mut m = wr_mask & DELEGABLE_INTS;
        if self.stip_from_sstc() {
            m &= !MIP_STIP;
        }
        self.hw.with_lines(&mut |l| {
            let mut new = new;
            if m & MIP_SEIP != 0 {
                l.software_seip = new & MIP_SEIP != 0;
                if l.external_seip {
                    new |= MIP_SEIP;
                }
            }
            let old = l.mip;
            if m != 0 {
                l.mip = (old & !m) | (new & m);
            }
            old
        })
    }

    /// `rmw_sip64()` and `rmw_mvip64()` for `sip`: the delegated bits of `mip`, of which
    /// SSIP and the local interrupts are writable.
    fn rmw_sip(&mut self, new: u64, wr_mask: u64) -> u64 {
        let wr_mask = wr_mask & self.st.mideleg & SIP_WRITABLE_MASK;
        let mut alias = (S_MODE_INTERRUPTS | LOCAL_INTERRUPTS | MIP_STIP) & self.st.mideleg;
        if self.stip_from_sstc() {
            alias &= !MIP_STIP;
        }
        let old = self.rmw_mip(new, wr_mask & alias & MVIP_WRITABLE_MASK) & alias;
        old & self.st.mideleg & (S_MODE_INTERRUPTS | LOCAL_INTERRUPTS)
    }

    /// The type of the selected trigger.
    fn cur_trigger_type(&self) -> u64 {
        self.st.tdata1[self.st.tselect as usize % NUM_TRIGGERS] >> 60
    }

    /// `read_tdata()` and `tdata_csr_read()`.
    fn read_tdata(&self, index: usize) -> Result<u64, i32> {
        if !tdata_mapped(self.cur_trigger_type(), index) {
            return Err(EXCP_ILLEGAL_INST);
        }
        let t = self.st.tselect as usize % NUM_TRIGGERS;
        Ok(match index {
            0 => self.st.tdata1[t],
            1 => self.st.tdata2[t],
            _ => self.st.tdata3[t],
        })
    }

    /// `write_tdata()` and `tdata_csr_write()`: a `tdata1` write may change the trigger
    /// type; the other registers follow the current type.
    fn write_tdata(&mut self, index: usize, val: u64) -> Result<(), i32> {
        if !tdata_mapped(self.cur_trigger_type(), index) {
            return Err(EXCP_ILLEGAL_INST);
        }
        let t = self.st.tselect as usize % NUM_TRIGGERS;
        let ttype = if index == 0 { val >> 60 } else { self.cur_trigger_type() };
        match (ttype, index) {
            (TRIGGER_TYPE_AD_MATCH, 0) => self.st.tdata1[t] = type2_mcontrol_validate(val),
            (TRIGGER_TYPE_AD_MATCH6, 0) => self.st.tdata1[t] = type6_mcontrol6_validate(val),
            (TRIGGER_TYPE_INST_CNT, 0) => self.st.tdata1[t] = itrigger_validate(val),
            (TRIGGER_TYPE_AD_MATCH | TRIGGER_TYPE_AD_MATCH6, 1) => self.st.tdata2[t] = val,
            (TRIGGER_TYPE_AD_MATCH | TRIGGER_TYPE_AD_MATCH6 | TRIGGER_TYPE_INST_CNT, 2) => {
                self.st.tdata3[t] = textra_validate(val);
            }
            // The other types are not supported or do not exist; the write is dropped.
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use super::*;
    use crate::cpu::{MIP_MEIP, MIP_MSIP, MIP_MTIP, MSTATUS_UBE, MSTATUS64_SD};

    #[derive(Default)]
    struct FakeHw {
        lines: RefCell<CpuLines>,
        time: Option<u64>,
        ticks: Cell<u64>,
        timecmp: RefCell<Vec<(u64, u64)>>,
        disabled: Cell<u32>,
    }

    impl Hw for FakeHw {
        fn rdtime(&self) -> Option<u64> {
            self.time
        }

        fn host_ticks(&self) -> u64 {
            self.ticks.get()
        }

        fn with_lines(&self, f: &mut dyn FnMut(&mut CpuLines) -> u64) -> u64 {
            f(&mut self.lines.borrow_mut())
        }

        fn write_timecmp(&self, menvcfg: u64, timecmp: u64) {
            self.timecmp.borrow_mut().push((menvcfg, timecmp));
        }

        fn disable_timecmp(&self) {
            self.disabled.set(self.disabled.get() + 1);
        }
    }

    fn csrs(hw: &FakeHw) -> Csrs<'_> {
        Csrs { st: CpuRiscvState::reset(0, 0x1000), hw, flush: false }
    }

    fn r(c: &mut Csrs<'_>, csrno: u32) -> Result<u64, i32> {
        c.rw(csrno, false, 0, 0)
    }

    fn w(c: &mut Csrs<'_>, csrno: u32, v: u64) -> Result<u64, i32> {
        c.rw(csrno, true, v, u64::MAX)
    }

    const ILL: Result<u64, i32> = Err(EXCP_ILLEGAL_INST);

    #[test]
    fn mstatus_write_mask() {
        let hw = FakeHw::default();
        let mut c = csrs(&hw);
        let old = r(&mut c, CSR_MSTATUS).unwrap();
        assert_eq!(old, (2 << 34) | (2 << 32));
        w(&mut c, CSR_MSTATUS, u64::MAX).unwrap();
        // UXL and SXL stay at 2; VS, XS, UBE and SD are not writable; FS dirty sets SD.
        let v = r(&mut c, CSR_MSTATUS).unwrap();
        assert_eq!(v, MSTATUS_WRITE_MASK | (2 << 34) | (2 << 32) | MSTATUS64_SD);
        assert_eq!(v & MSTATUS_UBE, 0);
        assert!(c.flush, "MXR changed");
    }

    #[test]
    fn mstatus_mpp_is_warl() {
        let hw = FakeHw::default();
        let mut c = csrs(&hw);
        w(&mut c, CSR_MSTATUS, set_field(0, MSTATUS_MPP, PRV_S)).unwrap();
        assert_eq!(get_field(c.st.mstatus, MSTATUS_MPP), PRV_S);
        // MPP = 2 is reserved; the old value stays.
        w(&mut c, CSR_MSTATUS, set_field(0, MSTATUS_MPP, 2)).unwrap();
        assert_eq!(get_field(c.st.mstatus, MSTATUS_MPP), PRV_S);
        assert!(!c.flush);
    }

    #[test]
    fn sstatus_view() {
        let hw = FakeHw::default();
        let mut c = csrs(&hw);
        c.st.mstatus |= MSTATUS_MIE | MSTATUS_MPP | MSTATUS_TVM;
        w(&mut c, CSR_SSTATUS, u64::MAX).unwrap();
        // M mode fields are untouched by an sstatus write.
        assert_eq!(
            c.st.mstatus & (MSTATUS_MIE | MSTATUS_MPP | MSTATUS_TVM),
            MSTATUS_MIE | MSTATUS_MPP | MSTATUS_TVM
        );
        let s = r(&mut c, CSR_SSTATUS).unwrap();
        assert_eq!(
            s,
            MSTATUS_SIE
                | MSTATUS_SPIE
                | MSTATUS_SPP
                | MSTATUS_FS
                | MSTATUS_SUM
                | MSTATUS_MXR
                | (2 << 32)
                | MSTATUS64_SD
        );
    }

    #[test]
    fn delegation_and_enable_masks() {
        let hw = FakeHw::default();
        let mut c = csrs(&hw);
        w(&mut c, CSR_MEDELEG, u64::MAX).unwrap();
        assert_eq!(c.st.medeleg, DELEGABLE_EXCPS);
        assert_eq!(c.st.medeleg & (1 << 11), 0, "M mode ecall cannot be delegated");
        w(&mut c, CSR_MIDELEG, u64::MAX).unwrap();
        assert_eq!(
            c.st.mideleg,
            (1 << 1) | (1 << 2) | (1 << 5) | (1 << 6) | (1 << 9) | (1 << 10) | (1 << 13)
        );
        w(&mut c, CSR_MIE, u64::MAX).unwrap();
        assert_eq!(c.st.mie, M_MODE_INTERRUPTS | S_MODE_INTERRUPTS | LOCAL_INTERRUPTS);
        // sie shows and writes only the delegated S mode and local bits.
        c.st.mideleg = MIP_SSIP | MIP_STIP;
        w(&mut c, CSR_MIE, 0).unwrap();
        assert_eq!(w(&mut c, CSR_SIE, u64::MAX), Ok(0));
        assert_eq!(c.st.mie, MIP_SSIP | MIP_STIP);
        assert_eq!(r(&mut c, CSR_SIE), Ok(MIP_SSIP | MIP_STIP));
    }

    #[test]
    fn mip_and_sip() {
        let hw = FakeHw::default();
        let mut c = csrs(&hw);
        hw.lines.borrow_mut().mip = MIP_MTIP | MIP_MEIP;
        // M mode interrupts are read only in mip.
        assert_eq!(w(&mut c, CSR_MIP, u64::MAX), Ok(MIP_MTIP | MIP_MEIP));
        let mip = hw.lines.borrow().mip;
        assert_eq!(mip & MIP_MSIP, 0);
        assert_eq!(mip & S_MODE_INTERRUPTS, S_MODE_INTERRUPTS);
        assert!(hw.lines.borrow().software_seip);
        // Clearing SEIP in software leaves the controller's input.
        hw.lines.borrow_mut().external_seip = true;
        w(&mut c, CSR_MIP, 0).unwrap();
        assert_eq!(hw.lines.borrow().mip & S_MODE_INTERRUPTS, MIP_SEIP);
        assert!(!hw.lines.borrow().software_seip);
        // sip: only SSIP is writable, and only when delegated.
        hw.lines.borrow_mut().mip = MIP_STIP;
        assert_eq!(w(&mut c, CSR_SIP, u64::MAX), Ok(0));
        assert_eq!(hw.lines.borrow().mip, MIP_STIP);
        c.st.mideleg = S_MODE_INTERRUPTS;
        assert_eq!(w(&mut c, CSR_SIP, u64::MAX), Ok(MIP_STIP));
        assert_eq!(hw.lines.borrow().mip, MIP_STIP | MIP_SSIP);
    }

    #[test]
    fn stip_belongs_to_sstc() {
        let hw = FakeHw { time: Some(0), ..FakeHw::default() };
        let mut c = csrs(&hw);
        w(&mut c, CSR_MENVCFG, MENVCFG_STCE).unwrap();
        assert_eq!(hw.timecmp.borrow().as_slice(), &[(MENVCFG_STCE, 0)]);
        w(&mut c, CSR_MIP, MIP_STIP).unwrap();
        assert_eq!(hw.lines.borrow().mip & MIP_STIP, 0);
        w(&mut c, CSR_STIMECMP, 1234).unwrap();
        assert_eq!(hw.timecmp.borrow().last(), Some(&(MENVCFG_STCE, 1234)));
        w(&mut c, CSR_MENVCFG, 0).unwrap();
        assert_eq!(hw.disabled.get(), 1);
        // ADUE was cleared by the write too.
        assert_eq!(c.st.menvcfg, 0);
    }

    #[test]
    fn vector_and_epc_masks() {
        let hw = FakeHw::default();
        let mut c = csrs(&hw);
        w(&mut c, CSR_MTVEC, 0x8000_0001).unwrap();
        assert_eq!(c.st.mtvec, 0x8000_0001);
        // Mode 2 is reserved; the write is dropped.
        w(&mut c, CSR_MTVEC, 0x9000_0002).unwrap();
        assert_eq!(c.st.mtvec, 0x8000_0001);
        w(&mut c, CSR_MEPC, 0x1003).unwrap();
        assert_eq!(r(&mut c, CSR_MEPC), Ok(0x1002));
        w(&mut c, CSR_MCOUNTEREN, u64::MAX).unwrap();
        assert_eq!(c.st.mcounteren, 0x7ffff);
        w(&mut c, CSR_MHPMEVENT3, u64::MAX).unwrap();
        assert_eq!(c.st.mhpmevent[3], !MHPMEVENT_VINH);
        w(&mut c, CSR_MCONTEXT, u64::MAX).unwrap();
        assert_eq!(c.st.mcontext, 0x1fff);
    }

    #[test]
    fn satp_modes() {
        let hw = FakeHw::default();
        let mut c = csrs(&hw);
        let sv39 = (VM_SV39 << 60) | 0x1234;
        w(&mut c, CSR_SATP, sv39).unwrap();
        assert_eq!(c.st.satp, sv39);
        assert!(c.flush);
        c.flush = false;
        // Sv64 (11) does not exist; the write is dropped.
        w(&mut c, CSR_SATP, 11 << 60).unwrap();
        assert_eq!(c.st.satp, sv39);
        assert!(!c.flush);
        // S mode with TVM may not access satp.
        c.st.priv_lvl = PRV_S;
        c.st.mstatus |= MSTATUS_TVM;
        assert_eq!(r(&mut c, CSR_SATP), ILL);
    }

    #[test]
    fn access_checks() {
        let hw = FakeHw::default();
        let mut c = csrs(&hw);
        // Writes to read only CSRs are illegal even with an empty mask.
        assert_eq!(c.rw(CSR_MHARTID, true, 0, 0), ILL);
        assert_eq!(r(&mut c, CSR_MARCHID), Ok(42));
        // Unknown and absent CSRs.
        assert_eq!(r(&mut c, 0x747), ILL, "mseccfg needs Smepmp");
        assert_eq!(r(&mut c, 0x3a1), ILL, "odd pmpcfg on RV64");
        assert_eq!(r(&mut c, 0xb01), ILL);
        assert_eq!(r(&mut c, 0xb13), ILL, "mhpmcounter19 is not implemented");
        assert_eq!(r(&mut c, 0x015), ILL, "seed needs Zkr");
        // FS off makes the FP CSRs illegal.
        assert_eq!(r(&mut c, CSR_FCSR), ILL);
        c.st.mstatus |= 1 << 13;
        w(&mut c, CSR_FCSR, 0xff).unwrap();
        assert_eq!((c.st.frm, c.st.fflags), (7, 0x1f));
        assert_eq!(c.st.mstatus & MSTATUS_FS, MSTATUS_FS);
        assert_eq!(r(&mut c, CSR_FRM), Ok(7));
        // Privilege.
        c.st.priv_lvl = PRV_S;
        assert_eq!(r(&mut c, CSR_MSTATUS), ILL);
        assert_eq!(r(&mut c, CSR_SSCRATCH), Ok(0));
        c.st.priv_lvl = PRV_U;
        assert_eq!(r(&mut c, CSR_SSCRATCH), ILL);
        assert_eq!(r(&mut c, CSR_FFLAGS), Ok(0x1f));
    }

    #[test]
    fn counter_access() {
        let hw = FakeHw::default();
        let mut c = csrs(&hw);
        // No timer: time is illegal even in M mode.
        assert_eq!(r(&mut c, CSR_TIME), ILL);
        assert_eq!(r(&mut c, CSR_STIMECMP), ILL);
        // hpmcounter19 and up do not exist.
        assert_eq!(r(&mut c, 0xc13), ILL);
        assert_eq!(r(&mut c, 0xc12), Ok(0));
        c.st.priv_lvl = PRV_S;
        assert_eq!(r(&mut c, CSR_CYCLE), ILL);
        c.st.mcounteren = COUNTEREN_CY;
        assert!(r(&mut c, CSR_CYCLE).is_ok());
        c.st.priv_lvl = PRV_U;
        assert_eq!(r(&mut c, CSR_CYCLE), ILL);
        c.st.scounteren = COUNTEREN_CY;
        assert!(r(&mut c, CSR_CYCLE).is_ok());
        assert_eq!(r(&mut c, CSR_INSTRET), ILL);
    }

    #[test]
    fn counters_count_and_inhibit() {
        let hw = FakeHw::default();
        let mut c = csrs(&hw);
        hw.ticks.set(100);
        w(&mut c, CSR_MCYCLE, 5).unwrap();
        hw.ticks.set(130);
        assert_eq!(r(&mut c, CSR_MCYCLE), Ok(35));
        // Inhibiting folds the elapsed count into the value.
        w(&mut c, CSR_MCOUNTINHIBIT, u64::MAX).unwrap();
        assert_eq!(c.st.mcountinhibit, PMU_AVAIL_CTRS | COUNTEREN_CY | COUNTEREN_IR);
        hw.ticks.set(1000);
        assert_eq!(r(&mut c, CSR_MCYCLE), Ok(35));
        w(&mut c, CSR_MCOUNTINHIBIT, 0).unwrap();
        hw.ticks.set(1010);
        assert_eq!(r(&mut c, CSR_MCYCLE), Ok(45));
        // The hpm counters hold their value.
        w(&mut c, CSR_MHPMCOUNTER3, 77).unwrap();
        hw.ticks.set(5000);
        assert_eq!(r(&mut c, CSR_MHPMCOUNTER3), Ok(77));
        assert_eq!(r(&mut c, 0xc03), Ok(77));
    }

    #[test]
    fn triggers() {
        let hw = FakeHw::default();
        let mut c = csrs(&hw);
        assert_eq!(r(&mut c, CSR_TINFO), Ok(0x44));
        // A type 2 write keeps the mode, access and size bits only.
        let ctrl = (2 << 60) | (1 << 59) | (0xf << 7) | (1 << 11) | (1 << 16) | TYPE_MODE_RWX;
        w(&mut c, CSR_TDATA1, ctrl).unwrap();
        assert_eq!(c.st.tdata1[0], (2 << 60) | (1 << 16) | TYPE_MODE_RWX);
        // Size 4 (6 bytes) is not supported and becomes any size.
        w(&mut c, CSR_TDATA1, (2 << 60) | (1 << 21)).unwrap();
        assert_eq!(c.st.tdata1[0], 2 << 60);
        w(&mut c, CSR_TDATA2, 0x8000_0000).unwrap();
        assert_eq!(r(&mut c, CSR_TDATA2), Ok(0x8000_0000));
        // tdata3: mhselect 5 becomes 4.
        w(&mut c, CSR_TDATA3, (5 << 48) | (3 << 51) | 0xff).unwrap();
        assert_eq!(c.st.tdata3[0], (4 << 48) | (3 << 51));
        // A type 0 write is ignored.
        w(&mut c, CSR_TDATA1, 0).unwrap();
        assert_eq!(c.st.tdata1[0], 2 << 60);
        // An instruction count trigger has no tdata2.
        w(&mut c, CSR_TDATA1, (3 << 60) | (1 << 10) | (1 << 9)).unwrap();
        assert_eq!(c.st.tdata1[0], (3 << 60) | (1 << 10) | (1 << 9));
        assert_eq!(r(&mut c, CSR_TDATA2), ILL);
        // tselect only selects triggers that exist.
        w(&mut c, CSR_TSELECT, 1).unwrap();
        w(&mut c, CSR_TSELECT, 2).unwrap();
        assert_eq!(r(&mut c, CSR_TSELECT), Ok(1));
        assert_eq!(r(&mut c, CSR_TDATA1), Ok(2 << 60));
        // Triggers are M mode only.
        c.st.priv_lvl = PRV_S;
        assert_eq!(r(&mut c, CSR_TDATA1), ILL);
    }

    #[test]
    fn pmp_csrs() {
        let hw = FakeHw::default();
        let mut c = csrs(&hw);
        w(&mut c, CSR_PMPADDR0, u64::MAX).unwrap();
        assert!(c.flush);
        assert_eq!(r(&mut c, CSR_PMPADDR0), Ok(PMPADDR_MASK));
        c.flush = false;
        w(&mut c, CSR_PMPCFG0 + 2, 0x1f).unwrap();
        assert!(c.flush);
        assert_eq!(c.st.pmp_num_rules, 1);
        assert_eq!(r(&mut c, CSR_PMPCFG0 + 2), Ok(0x1f));
        // pmpaddr16 and up exist but read as zero and ignore writes.
        c.flush = false;
        w(&mut c, CSR_PMPADDR0 + 20, 5).unwrap();
        assert!(!c.flush);
        assert_eq!(r(&mut c, CSR_PMPADDR0 + 20), Ok(0));
    }
}
