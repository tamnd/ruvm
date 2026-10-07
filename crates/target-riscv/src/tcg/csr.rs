// SPDX-License-Identifier: GPL-2.0-or-later

//! The control and status registers, a port of QEMU's `target/riscv/csr.c` (with the
//! trigger CSRs of `debug.c`) for the CSRs of the extensions this port has.
//!
//! The CSRs are `fflags`, `frm` and `fcsr`; `cycle`, `time`, `instret` and `hpmcounter3` to
//! `hpmcounter31`; the machine information registers; `mstatus`, `misa`, `medeleg`, `mideleg`,
//! `mie`, `mtvec`, `mcounteren`, `menvcfg`, `mcountinhibit`, `mhpmevent3` to `mhpmevent31`,
//! `mscratch`, `mepc`, `mcause`, `mtval`, `mip`, `mcycle`, `minstret` and `mhpmcounter3` to
//! `mhpmcounter31`; `sstatus`, `sie`, `stvec`, `scounteren`, `senvcfg`, `sscratch`, `sepc`,
//! `scause`, `stval`, `sip`, `stimecmp` and `satp`; the even `pmpcfg` registers up to `pmpcfg14`
//! and `pmpaddr0` to `pmpaddr63`; `tselect`, `tdata1` to `tdata3`, `tinfo` and `mcontext`; the
//! vector CSRs with Zve32x; and with H, `hstatus`, `hedeleg`, `hideleg`, `hie`, `htimedelta`,
//! `hcounteren`, `hgeie`, `henvcfg`, `htval`, `hip`, `hvip`, `htinst`, `hgatp`, `hgeip`,
//! `vsstatus`, `vsie`, `vstvec`, `vsscratch`, `vsepc`, `vscause`, `vstval`, `vsip`, `vstimecmp`,
//! `vsatp`, `mtval2` and `mtinst`; `seed` with Zkr; `mseccfg` with Smepmp, Zkr or Smmpm; and
//! `mstateen0` to `mstateen3`, `hstateen0` to `hstateen3` and `sstateen0` to `sstateen3` with
//! Smstateen; `scountovf` with Sscofpmf; and `mcyclecfg` and `minstretcfg` with Smcntrpmf;
//! `miselect`, `mireg` to `mireg6`, `siselect`, `sireg` to `sireg6`, `vsiselect` and `vsireg` to
//! `vsireg6` with Smcsrind and Sscsrind, which reach the counters delegated with Smcdeleg and
//! Ssccfg; and `scountinhibit` with Ssccfg. Every other CSR raises an illegal instruction
//! exception, as the extensions behind them (AIA, Smrnmi, Smctr, control flow integrity) are not in
//! the model, and so do the AIA ranges of the indirect registers. The counters themselves are in
//! [`super::pmu`]. The PMM fields of `menvcfg`, `senvcfg`, `henvcfg`, `hstatus` and `mseccfg` take
//! the pointer masking modes of Smnpm, Ssnpm and Smmpm, which [`super::pm`] applies.
//!
//! In VS and VU mode the S mode CSRs reach the VS registers, which the trap and return
//! paths swap into the S mode slots (`riscv_cpu_swap_hypervisor_regs()`), while `sie`,
//! `sip` and `stimecmp` redirect to `vsie`, `vsip` and `vstimecmp` like QEMU's. GEILEN is
//! 0, so `hgeie` and `hgeip` read as zero, and without AIA `hvien` is zero, so `hvip`
//! only holds the VS level bits of `mip`.
//!
//! Differences from QEMU:
//!
//! - `mstatus.UXL` and `vsstatus.UXL` cannot be written (QEMU lets them switch U and VU
//!   mode to RV32), and `hstatus.VSXL` reads as 2 like QEMU's.
//! - QEMU logs the `LOG_UNIMP` messages "QEMU does not support mixed HSXLEN options.",
//!   "QEMU does not support big endian guests." and "CSR_VSTVEC: reserved mode not
//!   supported" for such `hstatus` and `vstvec` writes; this crate has no logging and
//!   stays silent, with the same effect on the registers.
//! - A `csrw` (rd = x0) reads the old value before the write, where QEMU skips the read.
//!   No read in this model has a side effect or can fail when the write would succeed,
//!   so the result is the same.
//! - Writing `tdata1` with a trigger type QEMU does not know (1 or 8 to 14) is ignored;
//!   QEMU 11.1 hits an assertion.
//! - Triggers hold their values but never fire, and an instruction count trigger does
//!   not count.

use ruvm_jit::{Cpu, CpuShared, cputlb};

use super::pm::PMM_FIELD_RESERVED;
use super::{CpuLines, Riscv, SstcTimer, pmp, pmu};
use crate::cfg::PrivVer;
use crate::cpu::{
    COUNTEREN_CY, COUNTEREN_IR, COUNTEREN_TM, CpuRiscvState, EXCP_BREAKPOINT, EXCP_ILLEGAL_INST,
    EXCP_INST_ACCESS_FAULT, EXCP_INST_ADDR_MIS, EXCP_INST_GUEST_PAGE_FAULT, EXCP_INST_PAGE_FAULT,
    EXCP_LOAD_ACCESS_FAULT, EXCP_LOAD_ADDR_MIS, EXCP_LOAD_GUEST_ACCESS_FAULT, EXCP_LOAD_PAGE_FAULT,
    EXCP_M_ECALL, EXCP_S_ECALL, EXCP_STORE_AMO_ACCESS_FAULT, EXCP_STORE_AMO_ADDR_MIS,
    EXCP_STORE_GUEST_AMO_ACCESS_FAULT, EXCP_STORE_PAGE_FAULT, EXCP_U_ECALL,
    EXCP_VIRT_INSTRUCTION_FAULT, EXCP_VS_ECALL, FFLAGS_MASK, HS_MODE_INTERRUPTS, HSTATUS_HUKTE,
    HSTATUS_HUPMM, HSTATUS_VSBE, HSTATUS_VSXL, HSTATUS_VTVM, M_MODE_INTERRUPTS, MENVCFG_ADUE,
    MENVCFG_CBCFE, MENVCFG_CBIE, MENVCFG_CBZE, MENVCFG_CDE, MENVCFG_DTE, MENVCFG_FIOM,
    MENVCFG_PBMTE, MENVCFG_STCE, MIP_LCOFIP, MIP_SEIP, MIP_SGEIP, MIP_SSIP, MIP_STIP, MIP_VSEIP,
    MIP_VSSIP, MIP_VSTIP, MSTATUS_FS, MSTATUS_MIE, MSTATUS_MPIE, MSTATUS_MPP, MSTATUS_MPRV,
    MSTATUS_MXR, MSTATUS_SIE, MSTATUS_SPIE, MSTATUS_SPP, MSTATUS_SUM, MSTATUS_TSR, MSTATUS_TVM,
    MSTATUS_TW, MSTATUS64_UXL, NUM_TRIGGERS, PRV_M, PRV_S, PRV_U, RVF, RVS, RVU, S_MODE_INTERRUPTS,
    SATP64_ASID, SATP64_MODE, SATP64_PPN, SSTATUS_MASK, VS_MODE_INTERRUPTS, VSSTATUS64_UXL,
    add_status_sd, get_field, set_field,
};
use crate::cpu::{MENVCFG_PMM, MSTATUS_VS, RiscvCfg};

/// `ISELECT_*`: the `xiselect` values of Smcdeleg's counters, and the bits `xiselect`
/// holds with Smcsrind or Sscsrind, or with AIA alone.
const ISELECT_IPRIO0: u64 = 0x30;
const ISELECT_IPRIO15: u64 = 0x3f;
const ISELECT_CD_FIRST: u64 = 0x40;
const ISELECT_CD_LAST: u64 = 0x5f;
const ISELECT_IMSIC_FIRST: u64 = 0x70;
const ISELECT_IMSIC_LAST: u64 = 0xff;
const ISELECT_MASK_AIA: u64 = 0x1ff;
const ISELECT_MASK_SXCSRIND: u64 = 0xfff;

// The CSR numbers, from `cpu_bits.h`.
const CSR_FFLAGS: u32 = 0x001;
const CSR_FRM: u32 = 0x002;
const CSR_FCSR: u32 = 0x003;
const CSR_VSTART: u32 = 0x008;
const CSR_VXSAT: u32 = 0x009;
const CSR_VXRM: u32 = 0x00a;
const CSR_VCSR: u32 = 0x00f;
const CSR_VL: u32 = 0xc20;
const CSR_VTYPE: u32 = 0xc21;
const CSR_VLENB: u32 = 0xc22;
const CSR_CYCLE: u32 = 0xc00;
const CSR_TIME: u32 = 0xc01;
const CSR_INSTRET: u32 = 0xc02;
const CSR_HPMCOUNTER31: u32 = 0xc1f;
const CSR_SSTATUS: u32 = 0x100;
const CSR_SIE: u32 = 0x104;
const CSR_STVEC: u32 = 0x105;
const CSR_SCOUNTEREN: u32 = 0x106;
const CSR_SCOUNTOVF: u32 = 0xda0;
const CSR_SCOUNTINHIBIT: u32 = 0x120;
const CSR_SISELECT: u32 = 0x150;
const CSR_SIREG: u32 = 0x151;
const CSR_SIREG2: u32 = 0x152;
const CSR_SIREG4: u32 = 0x155;
const CSR_SIREG5: u32 = 0x156;
const CSR_SIREG6: u32 = 0x157;
const CSR_VSISELECT: u32 = 0x250;
const CSR_VSIREG: u32 = 0x251;
const CSR_VSIREG2: u32 = 0x252;
const CSR_VSIREG4: u32 = 0x255;
const CSR_VSIREG6: u32 = 0x257;
const CSR_MISELECT: u32 = 0x350;
const CSR_MIREG: u32 = 0x351;
const CSR_MIREG2: u32 = 0x352;
const CSR_MIREG4: u32 = 0x355;
const CSR_MIREG6: u32 = 0x357;
const CSR_SENVCFG: u32 = 0x10a;
const CSR_SSCRATCH: u32 = 0x140;
const CSR_SEPC: u32 = 0x141;
const CSR_SCAUSE: u32 = 0x142;
const CSR_STVAL: u32 = 0x143;
const CSR_SIP: u32 = 0x144;
const CSR_STIMECMP: u32 = 0x14d;
const CSR_SATP: u32 = 0x180;
const CSR_VSSTATUS: u32 = 0x200;
const CSR_VSIE: u32 = 0x204;
const CSR_VSTVEC: u32 = 0x205;
const CSR_VSSCRATCH: u32 = 0x240;
const CSR_VSEPC: u32 = 0x241;
const CSR_VSCAUSE: u32 = 0x242;
const CSR_VSTVAL: u32 = 0x243;
const CSR_VSIP: u32 = 0x244;
const CSR_VSTIMECMP: u32 = 0x24d;
const CSR_VSATP: u32 = 0x280;
const CSR_HSTATUS: u32 = 0x600;
const CSR_HEDELEG: u32 = 0x602;
const CSR_HIDELEG: u32 = 0x603;
const CSR_HIE: u32 = 0x604;
const CSR_HTIMEDELTA: u32 = 0x605;
const CSR_HCOUNTEREN: u32 = 0x606;
const CSR_HGEIE: u32 = 0x607;
const CSR_HENVCFG: u32 = 0x60a;
const CSR_HTVAL: u32 = 0x643;
const CSR_HIP: u32 = 0x644;
const CSR_HVIP: u32 = 0x645;
const CSR_HTINST: u32 = 0x64a;
const CSR_HGATP: u32 = 0x680;
const CSR_HGEIP: u32 = 0xe12;
const CSR_MSTATUS: u32 = 0x300;
const CSR_MISA: u32 = 0x301;
const CSR_MEDELEG: u32 = 0x302;
const CSR_MIDELEG: u32 = 0x303;
const CSR_MIE: u32 = 0x304;
const CSR_MTVEC: u32 = 0x305;
const CSR_MCOUNTEREN: u32 = 0x306;
const CSR_MENVCFG: u32 = 0x30a;
const CSR_MCOUNTINHIBIT: u32 = 0x320;
const CSR_MCYCLECFG: u32 = 0x321;
const CSR_MINSTRETCFG: u32 = 0x322;
const CSR_MHPMEVENT3: u32 = 0x323;
const CSR_MHPMEVENT31: u32 = 0x33f;
const CSR_MSCRATCH: u32 = 0x340;
const CSR_MEPC: u32 = 0x341;
const CSR_MCAUSE: u32 = 0x342;
const CSR_MTVAL: u32 = 0x343;
const CSR_MIP: u32 = 0x344;
const CSR_MTINST: u32 = 0x34a;
const CSR_MTVAL2: u32 = 0x34b;
const CSR_PMPCFG0: u32 = 0x3a0;
const CSR_PMPCFG3: u32 = 0x3a3;
const CSR_PMPCFG4: u32 = 0x3a4;
const CSR_PMPCFG15: u32 = 0x3af;
const CSR_PMPADDR0: u32 = 0x3b0;
const CSR_PMPADDR16: u32 = 0x3c0;
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
/// `seed`, the entropy source of Zkr.
pub(crate) const CSR_SEED: u32 = 0x015;
const CSR_MSECCFG: u32 = 0x747;
const CSR_MSTATEEN0: u32 = 0x30c;
const CSR_MSTATEEN3: u32 = 0x30f;
const CSR_HSTATEEN0: u32 = 0x60c;
const CSR_HSTATEEN3: u32 = 0x60f;
const CSR_SSTATEEN0: u32 = 0x10c;
const CSR_SSTATEEN3: u32 = 0x10f;

// The Smstateen bits, `SMSTATEEN0_*`.
const SMSTATEEN0_FCSR: u64 = 1 << 1;
const SMSTATEEN0_CTR: u64 = 1 << 54;
const SMSTATEEN0_P1P13: u64 = 1 << 56;
const SMSTATEEN0_IMSIC: u64 = 1 << 58;
const SMSTATEEN0_AIA: u64 = 1 << 59;
const SMSTATEEN0_SVSLCT: u64 = 1 << 60;
const SMSTATEEN0_HSENVCFG: u64 = 1 << 62;
const SMSTATEEN_STATEEN: u64 = 1 << 63;

/// `SEED_OPST_ES16`: `seed` holds 16 bits of entropy. QEMU's macro is a C `int`, so on
/// RV64 the value read is sign extended from bit 31 and bits 63 to 31 are all set.
const SEED_OPST_ES16: u64 = 0xffff_ffff_8000_0000;

/// `LOCAL_INTERRUPTS`: interrupts 16 and up.
const LOCAL_INTERRUPTS: u64 = !0xffff;
/// `delegable_ints`.
const DELEGABLE_INTS: u64 = S_MODE_INTERRUPTS | VS_MODE_INTERRUPTS | MIP_LCOFIP;
/// `vs_delegable_ints`.
const VS_DELEGABLE_INTS: u64 = (VS_MODE_INTERRUPTS | LOCAL_INTERRUPTS) & !MIP_LCOFIP;
/// `all_ints`.
const ALL_INTS: u64 = M_MODE_INTERRUPTS | S_MODE_INTERRUPTS | HS_MODE_INTERRUPTS | LOCAL_INTERRUPTS;
/// `mvip_writable_mask`.
const MVIP_WRITABLE_MASK: u64 = MIP_SSIP | MIP_STIP | MIP_SEIP | LOCAL_INTERRUPTS;
/// `sip_writable_mask`.
const SIP_WRITABLE_MASK: u64 = MIP_SSIP | LOCAL_INTERRUPTS;
/// `hip_writable_mask`.
const HIP_WRITABLE_MASK: u64 = MIP_VSSIP;
/// `hvip_writable_mask`.
const HVIP_WRITABLE_MASK: u64 = MIP_VSSIP | MIP_VSTIP | MIP_VSEIP | LOCAL_INTERRUPTS;
/// `vsip_writable_mask`.
const VSIP_WRITABLE_MASK: u64 = MIP_VSSIP | LOCAL_INTERRUPTS;

/// `RISCV_EXCP_SW_CHECK`.
const EXCP_SW_CHECK: i32 = 18;

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

/// `vs_delegable_excps`: the `hedeleg` bits that can be set.
const VS_DELEGABLE_EXCPS: u64 = DELEGABLE_EXCPS
    & !((1 << EXCP_S_ECALL)
        | (1 << EXCP_VS_ECALL)
        | (1 << EXCP_M_ECALL)
        | (1 << EXCP_INST_GUEST_PAGE_FAULT)
        | (1 << EXCP_LOAD_GUEST_ACCESS_FAULT)
        | (1 << EXCP_VIRT_INSTRUCTION_FAULT)
        | (1 << EXCP_STORE_GUEST_AMO_ACCESS_FAULT));

/// The `mstatus` bits a write changes: `write_mstatus()` without Smdbltrp, Ssdbltrp or
/// Zicfilp. FS is added with F and VS with Zve32x.
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
    | MSTATUS_TW;

/// The `senvcfg` bits a write changes, and the `henvcfg` bits that do not follow
/// `menvcfg`.
const SENVCFG_WRITE_MASK: u64 = MENVCFG_FIOM | MENVCFG_CBIE | MENVCFG_CBCFE | MENVCFG_CBZE;
/// The `henvcfg` bits that read as zero and cannot be set while the same `menvcfg` bit is
/// clear.
const HENVCFG_FOLLOWS_M: u64 = MENVCFG_PBMTE | MENVCFG_STCE | MENVCFG_ADUE | MENVCFG_DTE;

/// The PMM field of an `envcfg` write of `val`, if the hart has the extension `ext` that
/// makes it writable and `val` does not hold the reserved value 1.
fn pmm_mask(ext: bool, val: u64) -> u64 {
    if ext && get_field(val, MENVCFG_PMM) != PMM_FIELD_RESERVED { MENVCFG_PMM } else { 0 }
}

/// `SSTATUS_SDT`.
const SSTATUS_SDT: u64 = 1 << 24;

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
    /// `riscv_timer_write_timecmp()` for `stimecmp` or `vstimecmp`, with the CSRs of `st`.
    fn write_timecmp(&self, st: &CpuRiscvState, timer: SstcTimer);
    /// `riscv_timer_stce_changed()`: the STCE bit of `menvcfg` (`is_m`) or `henvcfg`
    /// flipped to `enable`.
    fn stce_changed(&self, st: &CpuRiscvState, is_m: bool, enable: bool);
    /// `timer_mod_anticipate_ns()` of the PMU timer: fire it in `delay_ns` nanoseconds, or
    /// earlier if it is already set for an earlier time.
    fn pmu_timer(&self, delay_ns: u64);
    /// The configuration of the CPU.
    fn cfg(&self) -> &RiscvCfg;
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

    fn write_timecmp(&self, st: &CpuRiscvState, timer: SstcTimer) {
        self.rv.write_timecmp(self.shared, st, timer);
    }

    fn stce_changed(&self, st: &CpuRiscvState, is_m: bool, enable: bool) {
        self.rv.stce_changed(self.shared, st, is_m, enable);
    }

    fn pmu_timer(&self, delay_ns: u64) {
        self.rv.pmu_timer(self.shared, delay_ns);
    }

    fn cfg(&self) -> &RiscvCfg {
        self.rv.cfg()
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

/// `riscv_new_csr_seed()`: 16 random bits, `qemu_guest_getrandom()` without `-seed`.
fn seed_value() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(0);
    (h.finish() & 0xffff) | SEED_OPST_ES16
}

/// `legalize_mpp()`: keep the old MPP if the new one is not a privilege level the hart
/// with extensions `misa` has (2 is reserved).
fn legalize_mpp(misa: u64, old_mpp: u64, val: u64) -> u64 {
    let valid = match get_field(val, MSTATUS_MPP) {
        PRV_M => true,
        PRV_S => misa & RVS != 0,
        PRV_U => misa & RVU != 0,
        _ => false,
    };
    if valid { val } else { set_field(val, MSTATUS_MPP, old_mpp) }
}

/// `legalize_xatp()` without the flush: the new `satp`, or `None` to keep the old one.
fn legalize_satp(cfg: &RiscvCfg, old: u64, val: u64) -> Option<u64> {
    let valid = cfg.satp_mode_ok(get_field(val, SATP64_MODE));
    let changed = (val ^ old) & (SATP64_MODE | SATP64_ASID | SATP64_PPN) != 0;
    (valid && changed).then_some(val)
}

/// `csr_ops[csrno].min_priv_ver`: the oldest privileged version that has CSR `csrno`.
fn min_priv_ver(csrno: u32) -> PrivVer {
    match csrno {
        CSR_MCOUNTINHIBIT | CSR_MSECCFG => PrivVer::V1_11,
        CSR_HSTATUS | CSR_HEDELEG | CSR_HIDELEG | CSR_HIE | CSR_HTIMEDELTA | CSR_HCOUNTEREN
        | CSR_HGEIE | CSR_HENVCFG | CSR_HTVAL | CSR_HIP | CSR_HVIP | CSR_HTINST | CSR_HGATP
        | CSR_HGEIP | CSR_VSSTATUS | CSR_VSIE | CSR_VSTVEC | CSR_VSSCRATCH | CSR_VSEPC
        | CSR_VSCAUSE | CSR_VSTVAL | CSR_VSIP | CSR_VSATP | CSR_MTVAL2 | CSR_MTINST
        | CSR_MENVCFG | CSR_SENVCFG | CSR_STIMECMP | CSR_VSTIMECMP | CSR_MCONFIGPTR => {
            PrivVer::V1_12
        }
        CSR_PMPCFG4..=CSR_PMPCFG15 | CSR_PMPADDR16..=CSR_PMPADDR63 => PrivVer::V1_12,
        CSR_MSTATEEN0..=CSR_MSTATEEN3
        | CSR_HSTATEEN0..=CSR_HSTATEEN3
        | CSR_SSTATEEN0..=CSR_SSTATEEN3
        | CSR_MCYCLECFG
        | CSR_MINSTRETCFG
        | CSR_SCOUNTOVF
        | CSR_SCOUNTINHIBIT => PrivVer::V1_12,
        // The aliases mireg2 to mireg6 and the like, but not mireg itself.
        CSR_MIREG2..=CSR_MIREG6 | CSR_SIREG2..=CSR_SIREG6 | CSR_VSIREG2..=CSR_VSIREG6 => {
            PrivVer::V1_12
        }
        _ => PrivVer::V1_10,
    }
}

/// Move the VS level interrupt bits of a `vsie` or `vsip` value up from their S level
/// positions to their `mie` or `mip` positions.
fn vs_bits_up(v: u64) -> u64 {
    let vsbits = v & (VS_MODE_INTERRUPTS >> 1);
    (v & !(VS_MODE_INTERRUPTS >> 1)) | (vsbits << 1)
}

/// Move the VS level interrupt bits of an `mie` or `mip` value down to their S level
/// positions in `vsie` or `vsip`.
fn vs_bits_down(v: u64) -> u64 {
    let vsbits = v & VS_MODE_INTERRUPTS;
    (v & !VS_MODE_INTERRUPTS) | (vsbits >> 1)
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
            CSR_HIDELEG => return Ok(self.rmw_hideleg(new, mask)),
            CSR_HVIP => return Ok(self.rmw_hvip(CSR_HVIP, new, mask)),
            CSR_HIP => {
                let old = self.rmw_mip64(CSR_HIP, new, mask & HIP_WRITABLE_MASK);
                return Ok(old & HS_MODE_INTERRUPTS);
            }
            CSR_HIE => {
                let old = self.rmw_mie(new, mask & HS_MODE_INTERRUPTS);
                return Ok(old & HS_MODE_INTERRUPTS);
            }
            CSR_VSIE => return Ok(self.rmw_vsie(new, mask)),
            CSR_VSIP => return Ok(self.rmw_vsip(new, mask)),
            CSR_SEED => return Ok(seed_value()),
            CSR_MISELECT | CSR_SISELECT | CSR_VSISELECT => {
                return self.rmw_xiselect(csrno, new, mask);
            }
            CSR_MIREG | CSR_SIREG | CSR_VSIREG => return self.rmw_xireg(csrno, new, mask),
            CSR_MIREG2..=CSR_MIREG6 | CSR_SIREG2..=CSR_SIREG6 | CSR_VSIREG2..=CSR_VSIREG6 => {
                return self.rmw_xiregi(csrno, new, mask);
            }
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
        // A CSR without an entry in csr_ops fails in predicate(), which gives the same
        // exception as QEMU's earlier check for the entry.
        let cfg = self.hw.cfg();
        if csrno > 0xfff || !cfg.ext_zicsr || !cfg.priv_at_least(min_priv_ver(csrno)) {
            return Err(EXCP_ILLEGAL_INST);
        }
        let read_only = (csrno >> 10) & 3 == 3;
        if write && read_only {
            return Err(EXCP_ILLEGAL_INST);
        }
        // The predicate may raise a virtual instruction exception, so it comes after the
        // read only check.
        self.predicate(csrno)?;
        let mut effective_priv = self.st.priv_lvl;
        if self.st.has_h() && self.st.priv_lvl == PRV_S && !self.st.virt() {
            // HS mode reaches the hypervisor CSRs.
            effective_priv += 1;
        }
        let csr_priv = u64::from((csrno >> 8) & 3);
        if effective_priv < csr_priv {
            if csr_priv <= PRV_S + 1 && self.st.virt() {
                return Err(EXCP_VIRT_INSTRUCTION_FAULT);
            }
            return Err(EXCP_ILLEGAL_INST);
        }
        Ok(())
    }

    /// The `predicate` of `csr_ops[csrno]`: whether the CSR exists and the current state
    /// allows the access, or the exception to raise. CSRs without an entry do not exist.
    fn predicate(&self, csrno: u32) -> Result<(), i32> {
        let cfg = self.hw.cfg();
        let smode = self.st.misa & RVS != 0;
        let umode = self.st.misa & RVU != 0;
        let ok = match csrno {
            // fs(): with Zfinx and FS off the FP CSRs depend on sstateen0.FCSR.
            CSR_FFLAGS | CSR_FRM | CSR_FCSR => {
                if !self.fs_enabled() && cfg.ext_zfinx {
                    return self.stateen_ok(0, SMSTATEEN0_FCSR);
                }
                self.fs_enabled()
            }
            CSR_VSTART | CSR_VXSAT | CSR_VXRM | CSR_VCSR | CSR_VL | CSR_VTYPE | CSR_VLENB => {
                self.vs()
            }
            CSR_CYCLE..=CSR_HPMCOUNTER31 => return self.ctr(csrno),
            CSR_MCYCLE | CSR_MINSTRET => true,
            // mctr().
            CSR_MHPMCOUNTER3..=CSR_MHPMCOUNTER31 => {
                u64::from(cfg.pmu_mask) & (1 << ctr_index(csrno)) != 0
            }
            CSR_MVENDORID..=CSR_MCONFIGPTR => true,
            CSR_MSTATUS | CSR_MISA | CSR_MIE | CSR_MTVEC | CSR_MCOUNTINHIBIT => true,
            CSR_MEDELEG | CSR_MIDELEG => smode,
            CSR_MCOUNTEREN | CSR_MENVCFG => umode,
            CSR_MHPMEVENT3..=CSR_MHPMEVENT31 => true,
            CSR_MSCRATCH | CSR_MEPC | CSR_MCAUSE | CSR_MTVAL | CSR_MIP => true,
            CSR_SSTATUS | CSR_SIE | CSR_STVEC | CSR_SCOUNTEREN | CSR_SENVCFG | CSR_SSCRATCH
            | CSR_SEPC | CSR_SCAUSE | CSR_STVAL | CSR_SIP => smode,
            CSR_STIMECMP => return self.sstc(false),
            CSR_VSTIMECMP => return self.sstc(true),
            // satp(): S mode with mstatus.TVM set may not touch satp, nor VS mode with
            // hstatus.VTVM.
            CSR_SATP => {
                let s = self.st.priv_lvl == PRV_S;
                if s && !self.st.virt() && self.st.mstatus & MSTATUS_TVM != 0 {
                    return Err(EXCP_ILLEGAL_INST);
                }
                if s && self.st.virt() && self.st.hstatus & HSTATUS_VTVM != 0 {
                    return Err(EXCP_VIRT_INSTRUCTION_FAULT);
                }
                smode
            }
            // hgatp(): HS mode with mstatus.TVM set may not touch hgatp.
            CSR_HGATP => {
                let s = self.st.priv_lvl == PRV_S;
                if s && !self.st.virt() && self.st.mstatus & MSTATUS_TVM != 0 {
                    return Err(EXCP_ILLEGAL_INST);
                }
                self.st.has_h()
            }
            // hmode().
            CSR_HSTATUS | CSR_HEDELEG | CSR_HIDELEG | CSR_HIE | CSR_HTIMEDELTA | CSR_HCOUNTEREN
            | CSR_HGEIE | CSR_HENVCFG | CSR_HTVAL | CSR_HIP | CSR_HVIP | CSR_HTINST | CSR_HGEIP
            | CSR_VSSTATUS | CSR_VSIE | CSR_VSTVEC | CSR_VSSCRATCH | CSR_VSEPC | CSR_VSCAUSE
            | CSR_VSTVAL | CSR_VSIP | CSR_VSATP | CSR_MTINST => self.st.has_h(),
            // dbltrp_hmode().
            CSR_MTVAL2 => cfg.ext_ssdbltrp || self.st.has_h(),
            // pmp(): the odd pmpcfg registers do not exist on RV64. Before 1.12 the
            // registers past pmpcfg3 fail the version check instead.
            CSR_PMPCFG0..=CSR_PMPCFG15 => {
                let max =
                    if cfg.priv_at_least(PrivVer::V1_12) { CSR_PMPCFG15 } else { CSR_PMPCFG3 };
                cfg.pmp && (csrno > max || (csrno - CSR_PMPCFG0) & 1 == 0)
            }
            CSR_PMPADDR0..=CSR_PMPADDR63 => cfg.pmp,
            // debug().
            CSR_TSELECT | CSR_TDATA1 | CSR_TDATA2 | CSR_TDATA3 | CSR_TINFO | CSR_MCONTEXT => {
                cfg.debug
            }
            CSR_SEED => return self.seed(),
            // have_mseccfg().
            CSR_MSECCFG => cfg.ext_smepmp || cfg.ext_zkr || cfg.ext_smmpm || cfg.ext_zicfilp,
            // mstateen().
            CSR_MSTATEEN0..=CSR_MSTATEEN3 => cfg.ext_smstateen,
            CSR_MCYCLECFG | CSR_MINSTRETCFG => cfg.ext_smcntrpmf,
            CSR_SCOUNTOVF => cfg.ext_sscofpmf,
            // scountinhibit_pred().
            CSR_SCOUNTINHIBIT => {
                if !cfg.ext_ssccfg || !cfg.ext_smcdeleg || self.st.menvcfg & MENVCFG_CDE == 0 {
                    return Err(EXCP_ILLEGAL_INST);
                }
                if self.st.virt() {
                    return Err(EXCP_VIRT_INSTRUCTION_FAULT);
                }
                smode
            }
            // csrind_or_aia_any(), csrind_or_aia_smode() and csrind_or_aia_hmode().
            CSR_MISELECT | CSR_MIREG => cfg.ext_smaia || cfg.ext_smcsrind,
            CSR_SISELECT | CSR_SIREG => {
                (cfg.ext_smcsrind || cfg.ext_sscsrind || cfg.ext_smaia || cfg.ext_ssaia) && smode
            }
            CSR_VSISELECT | CSR_VSIREG => {
                (cfg.ext_smcsrind || cfg.ext_sscsrind || cfg.ext_smaia || cfg.ext_ssaia)
                    && self.st.has_h()
            }
            // csrind_any(), csrind_smode() and csrind_hmode(): mireg2, mireg3 and mireg4 to
            // mireg6 (0x354 is not a CSR), and the same for sireg and vsireg.
            CSR_MIREG2..=CSR_MIREG6 => csrno != CSR_MIREG4 - 1 && cfg.ext_smcsrind,
            CSR_SIREG2..=CSR_SIREG6 => {
                csrno != CSR_SIREG4 - 1 && (cfg.ext_smcsrind || cfg.ext_sscsrind) && smode
            }
            CSR_VSIREG2..=CSR_VSIREG6 => {
                csrno != CSR_VSIREG4 - 1
                    && (cfg.ext_smcsrind || cfg.ext_sscsrind)
                    && self.st.has_h()
            }
            // hstateen(): below M mode, mstateen.SE0 must be set too.
            CSR_HSTATEEN0..=CSR_HSTATEEN3 => {
                let i = (csrno - CSR_HSTATEEN0) as usize;
                if !cfg.ext_smstateen || !self.st.has_h() {
                    return Err(EXCP_ILLEGAL_INST);
                }
                self.st.priv_lvl == PRV_M || self.st.mstateen[i] & SMSTATEEN_STATEEN != 0
            }
            CSR_SSTATEEN0..=CSR_SSTATEEN3 => return self.sstateen(csrno),
            _ => false,
        };
        if ok { Ok(()) } else { Err(EXCP_ILLEGAL_INST) }
    }

    /// `smstateen_acc_ok()`: whether bit `bit` of the `index`th `mstateen`, and in VS,
    /// VU and U mode of `hstateen` and `sstateen`, lets the hart reach what it guards.
    fn stateen_ok(&self, index: usize, bit: u64) -> Result<(), i32> {
        let st = &self.st;
        if st.priv_lvl == PRV_M || !self.hw.cfg().ext_smstateen {
            return Ok(());
        }
        if st.mstateen[index] & bit == 0 {
            return Err(EXCP_ILLEGAL_INST);
        }
        if st.virt() {
            if st.hstateen[index] & bit == 0 {
                return Err(EXCP_VIRT_INSTRUCTION_FAULT);
            }
            if st.priv_lvl == PRV_U && st.sstateen[index] & bit == 0 {
                return Err(EXCP_VIRT_INSTRUCTION_FAULT);
            }
        }
        if st.priv_lvl == PRV_U && st.misa & RVS != 0 && st.sstateen[index] & bit == 0 {
            return Err(EXCP_ILLEGAL_INST);
        }
        Ok(())
    }

    /// `sstateen()`: S mode and Smstateen, and below M mode the SE0 bit of `mstateen` and
    /// in VS mode of `hstateen`.
    fn sstateen(&self, csrno: u32) -> Result<(), i32> {
        let i = (csrno - CSR_SSTATEEN0) as usize;
        if !self.hw.cfg().ext_smstateen || self.st.misa & RVS == 0 {
            return Err(EXCP_ILLEGAL_INST);
        }
        if self.st.priv_lvl < PRV_M {
            if self.st.mstateen[i] & SMSTATEEN_STATEEN == 0 {
                return Err(EXCP_ILLEGAL_INST);
            }
            if self.st.virt() && self.st.hstateen[i] & SMSTATEEN_STATEEN == 0 {
                return Err(EXCP_VIRT_INSTRUCTION_FAULT);
            }
        }
        Ok(())
    }

    /// `seed()`: Zkr, and below M mode `mseccfg.SSEED` or `mseccfg.USEED`. VS and VU mode
    /// never reach `seed`.
    fn seed(&self) -> Result<(), i32> {
        if !self.hw.cfg().ext_zkr {
            return Err(EXCP_ILLEGAL_INST);
        }
        let st = &self.st;
        let sseed = st.mseccfg & pmp::MSECCFG_SSEED != 0;
        let useed = st.mseccfg & pmp::MSECCFG_USEED != 0;
        if st.priv_lvl == PRV_M {
            Ok(())
        } else if st.virt() {
            Err(if sseed { EXCP_VIRT_INSTRUCTION_FAULT } else { EXCP_ILLEGAL_INST })
        } else if (st.priv_lvl == PRV_S && sseed) || (st.priv_lvl == PRV_U && useed) {
            Ok(())
        } else {
            Err(EXCP_ILLEGAL_INST)
        }
    }

    /// `riscv_cpu_fp_enabled()`: `mstatus.FS` is on, and in VS or VU mode the HS level FS
    /// too. The FP CSRs need it, or Zfinx (`fs()`).
    fn fs_enabled(&self) -> bool {
        let hs = !self.st.virt() || self.st.mstatus_hs & MSTATUS_FS != 0;
        self.st.mstatus & MSTATUS_FS != 0 && hs
    }

    /// `vs()`: the vector CSRs need Zve32x and `mstatus.VS` on, and in VS or VU mode the
    /// HS level VS too (`riscv_cpu_vector_enabled()`).
    fn vs(&self) -> bool {
        let hs = !self.st.virt() || self.st.mstatus_hs & MSTATUS_VS != 0;
        self.hw.cfg().ext_zve32x && self.st.mstatus & MSTATUS_VS != 0 && hs
    }

    /// `ctr()`: the unprivileged counters, enabled by `mcounteren`, `hcounteren` and
    /// `scounteren`.
    fn ctr(&self, csrno: u32) -> Result<(), i32> {
        let bit = 1u64 << ctr_index(csrno);
        // cycle, time and instret come with Zicntr, the others with the PMU.
        let present = if csrno <= CSR_INSTRET {
            self.hw.cfg().ext_zicntr
        } else {
            u64::from(self.hw.cfg().pmu_mask) & bit != 0
        };
        if !present {
            return Err(EXCP_ILLEGAL_INST);
        }
        if self.st.priv_lvl < PRV_M && self.st.mcounteren & bit == 0 {
            return Err(EXCP_ILLEGAL_INST);
        }
        let s_off = self.st.priv_lvl == PRV_U && self.st.scounteren & bit == 0;
        if self.st.virt() && (self.st.hcounteren & bit == 0 || s_off) {
            return Err(EXCP_VIRT_INSTRUCTION_FAULT);
        }
        if s_off {
            return Err(EXCP_ILLEGAL_INST);
        }
        Ok(())
    }

    /// `sstc()` for `stimecmp`, or for `vstimecmp` (which needs H) when `vs`.
    fn sstc(&self, vs: bool) -> Result<(), i32> {
        let mode = if vs { self.st.has_h() } else { self.st.misa & RVS != 0 };
        if !self.hw.cfg().ext_sstc || !mode {
            return Err(EXCP_ILLEGAL_INST);
        }
        if self.hw.rdtime().is_none() {
            return Err(EXCP_ILLEGAL_INST);
        }
        if self.st.priv_lvl == PRV_M {
            return Ok(());
        }
        if self.st.mcounteren & COUNTEREN_TM == 0 || self.st.menvcfg & MENVCFG_STCE == 0 {
            return Err(EXCP_ILLEGAL_INST);
        }
        if self.st.virt()
            && (self.st.hcounteren & COUNTEREN_TM == 0 || self.st.henvcfg & MENVCFG_STCE == 0)
        {
            return Err(EXCP_VIRT_INSTRUCTION_FAULT);
        }
        Ok(())
    }

    /// The `read` operation of `csr_ops[csrno]`.
    fn read(&self, csrno: u32) -> Result<u64, i32> {
        let st = &self.st;
        Ok(match csrno {
            CSR_FFLAGS => st.fflags & FFLAGS_MASK,
            CSR_FRM => st.frm,
            CSR_FCSR => (st.fflags & FFLAGS_MASK) | (st.frm << 5),
            CSR_VSTART => st.vstart,
            CSR_VXSAT => st.vxsat & 1,
            CSR_VXRM => st.vxrm,
            CSR_VCSR => (st.vxrm << 1) | st.vxsat,
            CSR_VL => st.vl,
            CSR_VTYPE => (st.vill << 63) | st.vtype,
            CSR_VLENB => u64::from(self.hw.cfg().vlenb),
            CSR_TIME => {
                let delta = if st.virt() { st.htimedelta } else { 0 };
                self.hw.rdtime().ok_or(EXCP_ILLEGAL_INST)?.wrapping_add(delta)
            }
            CSR_CYCLE..=CSR_HPMCOUNTER31 | CSR_MCYCLE..=CSR_MHPMCOUNTER31 => {
                self.read_ctr(ctr_index(csrno))
            }
            CSR_MVENDORID => u64::from(self.hw.cfg().mvendorid),
            CSR_MARCHID => self.hw.cfg().marchid,
            CSR_MIMPID => self.hw.cfg().mimpid,
            CSR_MCONFIGPTR => 0,
            CSR_MHARTID => st.mhartid,
            CSR_MSTATUS => add_status_sd(st.mstatus),
            CSR_MISA => st.misa,
            CSR_MEDELEG => st.medeleg,
            CSR_MTVEC => st.mtvec,
            CSR_MCOUNTEREN => st.mcounteren,
            CSR_MENVCFG => st.menvcfg,
            CSR_MCOUNTINHIBIT => st.mcountinhibit,
            CSR_MHPMEVENT3..=CSR_MHPMEVENT31 => st.mhpmevent[ctr_index(csrno)],
            CSR_MCYCLECFG => st.mcyclecfg,
            CSR_MINSTRETCFG => st.minstretcfg,
            CSR_SCOUNTOVF => {
                // Counter delegation keeps scountovf from VS mode.
                let cfg = self.hw.cfg();
                if cfg.ext_sscofpmf && cfg.ext_ssccfg && st.menvcfg & MENVCFG_CDE != 0 && st.virt()
                {
                    return Err(EXCP_VIRT_INSTRUCTION_FAULT);
                }
                pmu::scountovf(st)
            }
            // read_scountinhibit(): the bits delegated by mcounteren.
            CSR_SCOUNTINHIBIT => st.mcountinhibit & st.mcounteren,
            CSR_MSCRATCH => st.mscratch,
            CSR_MEPC => st.mepc & self.hw.cfg().xepc_mask(),
            CSR_MCAUSE => st.mcause,
            CSR_MTVAL => st.mtval,
            CSR_SSTATUS => add_status_sd(st.mstatus & (SSTATUS_MASK | MSTATUS64_UXL)),
            CSR_STVEC => st.stvec,
            CSR_SCOUNTEREN => st.scounteren,
            CSR_SENVCFG => {
                self.stateen_ok(0, SMSTATEEN0_HSENVCFG)?;
                st.senvcfg
            }
            CSR_SSCRATCH => st.sscratch,
            CSR_SEPC => st.sepc & self.hw.cfg().xepc_mask(),
            CSR_SCAUSE => st.scause,
            CSR_STVAL => st.stval,
            CSR_STIMECMP if st.virt() => st.vstimecmp,
            CSR_STIMECMP => st.stimecmp,
            CSR_SATP => st.satp,
            CSR_MTVAL2 => st.mtval2,
            CSR_MTINST => st.mtinst,
            // hstatus: only a 64 bit, little endian VS mode.
            CSR_HSTATUS => set_field(set_field(st.hstatus, HSTATUS_VSXL, 2), HSTATUS_VSBE, 0),
            CSR_HEDELEG => st.hedeleg,
            CSR_HTIMEDELTA => {
                self.hw.rdtime().ok_or(EXCP_ILLEGAL_INST)?;
                st.htimedelta
            }
            CSR_HCOUNTEREN => st.hcounteren,
            CSR_HGEIE => st.hgeie,
            CSR_HENVCFG => {
                self.stateen_ok(0, SMSTATEEN0_HSENVCFG)?;
                st.henvcfg & (!HENVCFG_FOLLOWS_M | st.menvcfg)
            }
            CSR_HTVAL => st.htval,
            CSR_HTINST => st.htinst,
            CSR_HGATP => st.hgatp,
            // GEILEN is 0: no guest external interrupt is ever pending.
            CSR_HGEIP => 0,
            CSR_VSSTATUS => st.vsstatus,
            CSR_VSTVEC => st.vstvec,
            CSR_VSSCRATCH => st.vsscratch,
            CSR_VSEPC => st.vsepc,
            CSR_VSCAUSE => st.vscause,
            CSR_VSTVAL => st.vstval,
            CSR_VSTIMECMP => st.vstimecmp,
            CSR_VSATP => st.vsatp,
            CSR_PMPCFG0..=CSR_PMPCFG15 => {
                let n = usize::from(self.hw.cfg().pmp_regions);
                pmp::pmpcfg_csr_read(st, n, (csrno - CSR_PMPCFG0) as usize)
            }
            CSR_PMPADDR0..=CSR_PMPADDR63 => {
                let n = usize::from(self.hw.cfg().pmp_regions);
                pmp::pmpaddr_csr_read(st, n, (csrno - CSR_PMPADDR0) as usize) & PMPADDR_MASK
            }
            CSR_TSELECT => st.tselect,
            CSR_TDATA1 | CSR_TDATA2 | CSR_TDATA3 => {
                self.read_tdata((csrno - CSR_TDATA1) as usize)?
            }
            // tinfo_csr_read(): every trigger can be a type 2 or a type 6 trigger.
            CSR_TINFO => (1 << TRIGGER_TYPE_AD_MATCH) | (1 << TRIGGER_TYPE_AD_MATCH6),
            CSR_MCONTEXT => st.mcontext,
            CSR_MSECCFG => st.mseccfg,
            CSR_MSTATEEN0..=CSR_MSTATEEN3 => st.mstateen[(csrno - CSR_MSTATEEN0) as usize],
            CSR_HSTATEEN0..=CSR_HSTATEEN3 => {
                let i = (csrno - CSR_HSTATEEN0) as usize;
                st.hstateen[i] & st.mstateen[i]
            }
            CSR_SSTATEEN0..=CSR_SSTATEEN3 => {
                let i = (csrno - CSR_SSTATEEN0) as usize;
                let h = if st.virt() { st.hstateen[i] } else { !0 };
                st.sstateen[i] & st.mstateen[i] & h
            }
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
            CSR_VSTART => {
                self.st.mstatus |= MSTATUS_VS;
                // Only enough bits to hold the largest element index, lg2(VLEN).
                let bits = (u64::from(self.hw.cfg().vlenb) << 3).trailing_zeros();
                self.st.vstart = val & !(!0u64 << bits);
            }
            CSR_VXSAT => {
                self.st.mstatus |= MSTATUS_VS;
                self.st.vxsat = val & 1;
            }
            CSR_VXRM => {
                self.st.mstatus |= MSTATUS_VS;
                self.st.vxrm = val & 3;
            }
            CSR_VCSR => {
                self.st.mstatus |= MSTATUS_VS;
                self.st.vxrm = (val >> 1) & 3;
                self.st.vxsat = val & 1;
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
            CSR_MCOUNTEREN => self.st.mcounteren = val & self.counteren_mask(),
            CSR_MENVCFG => self.write_menvcfg(val),
            CSR_MCOUNTINHIBIT => self.write_mcountinhibit(val),
            CSR_MHPMEVENT3..=CSR_MHPMEVENT31 => {
                let idx = ctr_index(csrno);
                let v = val & pmu::inh_avail_mask(&self.st);
                self.st.mhpmevent[idx] = v;
                pmu::update_event_map(&mut self.st, self.hw.cfg(), v, idx);
            }
            CSR_MCYCLECFG => self.st.mcyclecfg = val & pmu::inh_avail_mask(&self.st),
            CSR_MINSTRETCFG => self.st.minstretcfg = val & pmu::inh_avail_mask(&self.st),
            CSR_SCOUNTINHIBIT => self.write_mcountinhibit(val & self.st.mcounteren),
            CSR_MSCRATCH => self.st.mscratch = val,
            CSR_MEPC => self.st.mepc = val & self.hw.cfg().xepc_mask(),
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
            CSR_SCOUNTEREN => self.st.scounteren = val & self.counteren_mask(),
            CSR_SENVCFG => {
                self.stateen_ok(0, SMSTATEEN0_HSENVCFG)?;
                let m = SENVCFG_WRITE_MASK | pmm_mask(self.hw.cfg().ext_ssnpm, val);
                self.st.senvcfg = (self.st.senvcfg & !m) | (val & m);
            }
            CSR_SSCRATCH => self.st.sscratch = val,
            CSR_SEPC => self.st.sepc = val & self.hw.cfg().xepc_mask(),
            CSR_SCAUSE => self.st.scause = val,
            CSR_STVAL => self.st.stval = val,
            CSR_STIMECMP | CSR_VSTIMECMP if csrno == CSR_VSTIMECMP || self.st.virt() => {
                self.st.vstimecmp = val;
                self.hw.write_timecmp(&self.st, SstcTimer::Vs);
            }
            CSR_STIMECMP => {
                self.st.stimecmp = val;
                self.hw.write_timecmp(&self.st, SstcTimer::S);
            }
            // write_satp(): without an MMU the write is ignored.
            CSR_SATP => {
                if self.hw.cfg().mmu {
                    self.st.satp = self.legalize_xatp(self.st.satp, val);
                }
            }
            CSR_MTVAL2 => self.st.mtval2 = val,
            CSR_MTINST => self.st.mtinst = val,
            CSR_HSTATUS => {
                // Svukte is not there; HUPMM needs Ssnpm and keeps its value when the
                // reserved 1 is written.
                let mut mask = !HSTATUS_HUKTE;
                if !self.hw.cfg().ext_ssnpm || get_field(val, HSTATUS_HUPMM) == PMM_FIELD_RESERVED {
                    mask &= !HSTATUS_HUPMM;
                }
                self.st.hstatus = (self.st.hstatus & !mask) | (val & mask);
                // QEMU logs "QEMU does not support mixed HSXLEN options." for a VSXL other
                // than 2 and "QEMU does not support big endian guests." for VSBE set.
            }
            CSR_HEDELEG => self.st.hedeleg = val & VS_DELEGABLE_EXCPS,
            CSR_HTIMEDELTA => {
                self.hw.rdtime().ok_or(EXCP_ILLEGAL_INST)?;
                self.st.htimedelta = val;
                self.hw.write_timecmp(&self.st, SstcTimer::Vs);
            }
            CSR_HCOUNTEREN => self.st.hcounteren = val & self.counteren_mask(),
            CSR_HGEIE => {
                // Only bits 1 to GEILEN exist, and GEILEN is 0; mip.SGEIP follows
                // hgeie & hgeip, which is 0.
                self.st.hgeie = 0;
                self.hw.with_lines(&mut |l| {
                    l.mip &= !MIP_SGEIP;
                    0
                });
            }
            CSR_HENVCFG => {
                self.stateen_ok(0, SMSTATEEN0_HSENVCFG)?;
                self.write_henvcfg(val);
            }
            CSR_HTVAL => self.st.htval = val,
            // htinst writes are ignored.
            CSR_HTINST => {}
            CSR_HGATP => self.st.hgatp = self.legalize_xatp(self.st.hgatp, val),
            CSR_VSSTATUS => {
                // UXL stays 64 bit; SDT needs Ssdbltrp (henvcfg.DTE).
                let val = set_field(val, VSSTATUS64_UXL, 2);
                self.st.vsstatus =
                    if self.st.henvcfg & MENVCFG_DTE != 0 { val } else { val & !SSTATUS_SDT };
            }
            CSR_VSTVEC => {
                // Modes 2 and 3 are reserved; QEMU logs "CSR_VSTVEC: reserved mode not
                // supported" and drops the write.
                if val & 3 < 2 {
                    self.st.vstvec = val;
                }
            }
            CSR_VSSCRATCH => self.st.vsscratch = val,
            CSR_VSEPC => self.st.vsepc = val,
            CSR_VSCAUSE => self.st.vscause = val,
            CSR_VSTVAL => self.st.vstval = val,
            CSR_VSATP => self.st.vsatp = self.legalize_xatp(self.st.vsatp, val),
            CSR_PMPCFG0..=CSR_PMPCFG15 => {
                let i = (csrno - CSR_PMPCFG0) as usize;
                let cfg = self.hw.cfg();
                let n = usize::from(cfg.pmp_regions);
                self.flush |= pmp::pmpcfg_csr_write(&mut self.st, n, cfg.ext_smpmpmt, i, val);
            }
            CSR_PMPADDR0..=CSR_PMPADDR63 => {
                let i = (csrno - CSR_PMPADDR0) as usize;
                let n = usize::from(self.hw.cfg().pmp_regions);
                self.flush |= pmp::pmpaddr_csr_write(&mut self.st, n, i, val);
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
            CSR_MSECCFG => {
                let n = usize::from(self.hw.cfg().pmp_regions);
                let (smepmp, smmpm) = (self.hw.cfg().ext_smepmp, self.hw.cfg().ext_smmpm);
                self.flush |= pmp::mseccfg_csr_write(&mut self.st, n, smepmp, smmpm, val);
            }
            CSR_MSTATEEN0..=CSR_MSTATEEN3 => {
                let i = (csrno - CSR_MSTATEEN0) as usize;
                let m = if i == 0 { self.mstateen0_mask() } else { SMSTATEEN_STATEEN };
                self.st.mstateen[i] = (self.st.mstateen[i] & !m) | (val & m);
            }
            CSR_HSTATEEN0..=CSR_HSTATEEN3 => {
                let i = (csrno - CSR_HSTATEEN0) as usize;
                let m = if i == 0 { self.hstateen0_mask() } else { SMSTATEEN_STATEEN };
                let m = m & self.st.mstateen[i];
                self.st.hstateen[i] = (self.st.hstateen[i] & !m) | (val & m);
            }
            CSR_SSTATEEN0..=CSR_SSTATEEN3 => {
                let i = (csrno - CSR_SSTATEEN0) as usize;
                let mut m = if i == 0 {
                    if self.st.misa & RVF == 0 { SMSTATEEN0_FCSR } else { 0 }
                } else {
                    SMSTATEEN_STATEEN
                };
                m &= self.st.mstateen[i];
                if self.st.virt() {
                    m &= self.st.hstateen[i];
                }
                self.st.sstateen[i] = (self.st.sstateen[i] & !m) | (val & m);
            }
            // The read only CSRs, and tinfo whose writes are ignored.
            _ => {}
        }
        Ok(())
    }

    /// The writable bits of `mstateen0`, `write_mstateen0()`.
    fn mstateen0_mask(&self) -> u64 {
        let cfg = self.hw.cfg();
        let mut m = SMSTATEEN_STATEEN | SMSTATEEN0_HSENVCFG;
        if self.st.misa & RVF == 0 {
            m |= SMSTATEEN0_FCSR;
        }
        if cfg.priv_at_least(PrivVer::V1_13) {
            m |= SMSTATEEN0_P1P13;
        }
        if cfg.ext_smaia || cfg.ext_smcsrind {
            m |= SMSTATEEN0_SVSLCT;
        }
        if cfg.ext_smaia {
            m |= SMSTATEEN0_AIA | SMSTATEEN0_IMSIC;
        }
        if cfg.ext_ssctr {
            m |= SMSTATEEN0_CTR;
        }
        m
    }

    /// The bits of `hstateen0` that `mstateen0` lets through, `write_hstateen0()`.
    fn hstateen0_mask(&self) -> u64 {
        let cfg = self.hw.cfg();
        let mut m = SMSTATEEN_STATEEN | SMSTATEEN0_HSENVCFG;
        if self.st.misa & RVF == 0 {
            m |= SMSTATEEN0_FCSR;
        }
        if cfg.ext_ssaia || cfg.ext_sscsrind {
            m |= SMSTATEEN0_SVSLCT;
        }
        if cfg.ext_ssaia {
            m |= SMSTATEEN0_AIA | SMSTATEEN0_IMSIC;
        }
        if cfg.ext_ssctr {
            m |= SMSTATEEN0_CTR;
        }
        m
    }

    /// `write_mstatus()`.
    fn write_mstatus(&mut self, val: u64) {
        let mstatus = self.st.mstatus;
        let val = legalize_mpp(self.st.misa, get_field(mstatus, MSTATUS_MPP), val);
        // MXR changes what a page permits; MPRV and SUM select another MMU index.
        if (val ^ mstatus) & MSTATUS_MXR != 0 {
            self.flush = true;
        }
        let mut mask = MSTATUS_WRITE_MASK;
        if self.st.misa & RVF != 0 {
            mask |= MSTATUS_FS;
        }
        if self.hw.cfg().ext_zve32x {
            mask |= MSTATUS_VS;
        }
        self.st.mstatus = (mstatus & !mask) | (val & mask);
    }

    /// `legalize_xatp()` for `satp`, `vsatp` and `hgatp`: the new value, flushing the
    /// TLB, or the old one if the mode is not supported or nothing changed.
    fn legalize_xatp(&mut self, old: u64, val: u64) -> u64 {
        match legalize_satp(self.hw.cfg(), old, val) {
            Some(v) => {
                self.flush = true;
                v
            }
            None => old,
        }
    }

    /// The counters `mcounteren`, `scounteren` and `hcounteren` can enable.
    fn counteren_mask(&self) -> u64 {
        u64::from(self.hw.cfg().pmu_mask) | COUNTEREN_CY | COUNTEREN_TM | COUNTEREN_IR
    }

    /// `write_menvcfg()`, with `riscv_timer_stce_changed()` when STCE flips, then
    /// `write_henvcfg()` to drop the `henvcfg` bits that follow `menvcfg`.
    fn write_menvcfg(&mut self, val: u64) {
        let cfg = self.hw.cfg();
        let mut m = SENVCFG_WRITE_MASK;
        if cfg.ext_svpbmt {
            m |= MENVCFG_PBMTE;
        }
        if cfg.ext_sstc {
            m |= MENVCFG_STCE;
        }
        if cfg.ext_svadu {
            m |= MENVCFG_ADUE;
        }
        if cfg.ext_smcdeleg {
            m |= MENVCFG_CDE;
        }
        m |= pmm_mask(cfg.ext_smnpm, val);
        let stce_changed = cfg.ext_sstc && (self.st.menvcfg ^ val) & MENVCFG_STCE != 0;
        self.st.menvcfg = (self.st.menvcfg & !m) | (val & m);
        if stce_changed {
            self.hw.stce_changed(&self.st, true, val & MENVCFG_STCE != 0);
        }
        self.write_henvcfg(self.st.henvcfg);
    }

    /// `write_henvcfg()`: PBMTE, STCE, ADUE and DTE can only be set when `menvcfg` has
    /// them. PMM needs Ssnpm, and the reserved value 1 clears it, as in QEMU.
    fn write_henvcfg(&mut self, val: u64) {
        let mask = SENVCFG_WRITE_MASK
            | (self.st.menvcfg & HENVCFG_FOLLOWS_M)
            | pmm_mask(self.hw.cfg().ext_ssnpm, val);
        let stce_changed = self.hw.cfg().ext_sstc && (self.st.henvcfg ^ val) & MENVCFG_STCE != 0;
        self.st.henvcfg = val & mask;
        if self.st.henvcfg & MENVCFG_DTE == 0 {
            self.st.vsstatus &= !SSTATUS_SDT;
        }
        if stce_changed {
            self.hw.stce_changed(&self.st, false, val & MENVCFG_STCE != 0);
        }
    }

    /// `riscv_pmu_read_ctr()`.
    fn read_ctr(&self, idx: usize) -> u64 {
        pmu::read_ctr(&self.st, self.hw.host_ticks(), idx)
    }

    /// `riscv_pmu_write_ctr()`.
    fn write_ctr(&mut self, idx: usize, val: u64) {
        let hw = self.hw;
        pmu::write_ctr(&mut self.st, hw.cfg(), hw.host_ticks(), idx, val, &mut |d| hw.pmu_timer(d));
    }

    /// `write_mcountinhibit()`.
    fn write_mcountinhibit(&mut self, val: u64) {
        let hw = self.hw;
        pmu::write_mcountinhibit(&mut self.st, hw.cfg(), hw.host_ticks(), val, &mut |d| {
            hw.pmu_timer(d)
        });
    }

    /// `csrind_xlate_vs_csrno()`: in VS mode `siselect` and `sireg` to `sireg6` reach the
    /// VS registers.
    fn csrind_xlate_vs(&self, csrno: u32) -> u32 {
        if !self.st.virt() {
            return csrno;
        }
        match csrno {
            CSR_SISELECT => CSR_VSISELECT,
            CSR_SIREG..=CSR_SIREG6 => CSR_VSIREG + (csrno - CSR_SIREG),
            _ => csrno,
        }
    }

    /// `rmw_xiselect()`.
    fn rmw_xiselect(&mut self, csrno: u32, new: u64, mask: u64) -> Result<u64, i32> {
        self.stateen_ok(0, SMSTATEEN0_SVSLCT)?;
        let cfg = self.hw.cfg();
        let held = if cfg.ext_smcsrind || cfg.ext_sscsrind {
            ISELECT_MASK_SXCSRIND
        } else {
            ISELECT_MASK_AIA
        };
        let iselect = match self.csrind_xlate_vs(csrno) {
            CSR_MISELECT => &mut self.st.miselect,
            CSR_SISELECT => &mut self.st.siselect,
            CSR_VSISELECT => &mut self.st.vsiselect,
            _ => return Err(EXCP_ILLEGAL_INST),
        };
        let old = *iselect;
        let mask = mask & held;
        *iselect = (old & !mask) | (new & mask);
        Ok(old)
    }

    /// `rmw_xireg()`: `mireg`, `sireg` and `vsireg`.
    fn rmw_xireg(&mut self, csrno: u32, new: u64, mask: u64) -> Result<u64, i32> {
        self.stateen_ok(0, SMSTATEEN0_SVSLCT)?;
        let csrno = self.csrind_xlate_vs(csrno);
        let isel = match csrno {
            CSR_MIREG => self.st.miselect,
            CSR_SIREG => self.st.siselect,
            CSR_VSIREG => self.st.vsiselect,
            _ => return Err(EXCP_ILLEGAL_INST),
        };
        let aia = (ISELECT_IPRIO0..=ISELECT_IPRIO15).contains(&isel)
            || (ISELECT_IMSIC_FIRST..=ISELECT_IMSIC_LAST).contains(&isel);
        let cfg = self.hw.cfg();
        if aia {
            // rmw_xireg_aia() without Smaia and Ssaia.
            return Err(EXCP_ILLEGAL_INST);
        }
        if cfg.ext_smcsrind || cfg.ext_sscsrind {
            return self.rmw_xireg_csrind(csrno, isel, new, mask);
        }
        Err(EXCP_ILLEGAL_INST)
    }

    /// `rmw_xiregi()`: the aliases `mireg2` to `mireg6` and the like.
    fn rmw_xiregi(&mut self, csrno: u32, new: u64, mask: u64) -> Result<u64, i32> {
        self.stateen_ok(0, SMSTATEEN0_SVSLCT)?;
        let csrno = self.csrind_xlate_vs(csrno);
        let isel = match csrno {
            CSR_MIREG..=CSR_MIREG6 => self.st.miselect,
            CSR_SIREG..=CSR_SIREG6 => self.st.siselect,
            CSR_VSIREG..=CSR_VSIREG6 => self.st.vsiselect,
            _ => return Err(EXCP_ILLEGAL_INST),
        };
        self.rmw_xireg_csrind(csrno, isel, new, mask)
    }

    /// `rmw_xireg_csrind()`: the counters of Smcdeleg are the only registers behind the
    /// indirect CSRs (Smctr's are not in the model).
    fn rmw_xireg_csrind(&mut self, csrno: u32, isel: u64, new: u64, mask: u64) -> Result<u64, i32> {
        if !(ISELECT_CD_FIRST..=ISELECT_CD_LAST).contains(&isel) {
            return Err(EXCP_ILLEGAL_INST);
        }
        // Only vsireg itself, not its aliases, gives the virtual instruction exception.
        self.rmw_xireg_cd(csrno, isel, new, mask).ok_or(if self.st.virt() && csrno == CSR_VSIREG {
            EXCP_VIRT_INSTRUCTION_FAULT
        } else {
            EXCP_ILLEGAL_INST
        })
    }

    /// `rmw_xireg_cd()`: Smcdeleg's view of counter `isel - 0x40` through `sireg` (the
    /// counter) and `sireg2` (its event or configuration), or `None` for an exception. The
    /// translated CSR number of VS mode never matches, as in QEMU.
    fn rmw_xireg_cd(&mut self, csrno: u32, isel: u64, new: u64, mask: u64) -> Option<u64> {
        let cfg = self.hw.cfg();
        let idx = (isel - ISELECT_CD_FIRST) as usize;
        if !cfg.ext_smcdeleg || !cfg.ext_ssccfg || idx == 1 {
            return None;
        }
        // sireg4 and sireg5 hold the upper halves on RV32 only.
        if csrno == CSR_SIREG4 || csrno == CSR_SIREG5 {
            return None;
        }
        if !cfg.ext_smcntrpmf && csrno == CSR_SIREG2 && idx < 3 {
            return None;
        }
        if self.st.mcounteren & (1 << idx) == 0 || self.st.menvcfg & MENVCFG_CDE == 0 {
            return None;
        }
        // The counter and event views take whole register writes only.
        let whole = mask == 0 || mask == u64::MAX;
        match csrno {
            CSR_SIREG if whole => {
                // rmw_cd_mhpmcounter().
                if mask == 0 {
                    return Some(self.read_ctr(idx));
                }
                self.write_ctr(idx, new);
                Some(0)
            }
            CSR_SIREG2 if idx <= 2 => Some(self.rmw_cd_ctr_cfg(idx, new, mask)),
            CSR_SIREG2 if whole => {
                // rmw_cd_mhpmevent(): S mode cannot see or set MINH.
                let ev = self.st.mhpmevent[idx];
                if mask == 0 {
                    let minh = if cfg.ext_sscofpmf { pmu::MHPMEVENT_MINH } else { 0 };
                    return Some(ev & !minh);
                }
                let m = mask & !pmu::MHPMEVENT_MINH;
                let v = (new & m) | (ev & !m);
                self.st.mhpmevent[idx] = v;
                pmu::update_event_map(&mut self.st, cfg, v, idx);
                Some(0)
            }
            _ => None,
        }
    }

    /// `rmw_cd_ctr_cfg()`: `mcyclecfg` (counter 0) or `minstretcfg` (counter 2) without
    /// MINH. Like QEMU, a read clears MINH in the register itself.
    fn rmw_cd_ctr_cfg(&mut self, idx: usize, new: u64, mask: u64) -> u64 {
        let reg = if idx == 0 { &mut self.st.mcyclecfg } else { &mut self.st.minstretcfg };
        if mask != 0 {
            let m = mask & !pmu::MHPMEVENT_MINH;
            *reg = (new & m) | (*reg & !m);
            0
        } else {
            *reg &= !pmu::MHPMEVENT_MINH;
            *reg
        }
    }

    /// `rmw_mideleg64()`: with H the VS level and guest external interrupts are always
    /// delegated.
    fn rmw_mideleg(&mut self, new: u64, wr_mask: u64) -> u64 {
        let m = wr_mask & DELEGABLE_INTS;
        let old = self.st.mideleg;
        self.st.mideleg = (old & !m) | (new & m);
        if self.st.has_h() {
            self.st.mideleg |= HS_MODE_INTERRUPTS;
        }
        old
    }

    /// `rmw_mie64()`: without H the hypervisor interrupt enables stay clear.
    fn rmw_mie(&mut self, new: u64, wr_mask: u64) -> u64 {
        let m = wr_mask & ALL_INTS;
        let old = self.st.mie;
        self.st.mie = (old & !m) | (new & m);
        if !self.st.has_h() {
            self.st.mie &= !HS_MODE_INTERRUPTS;
        }
        old
    }

    /// `rmw_sie64()`: `sie` shows the delegated bits of `mie`, or of `vsie` in VS mode.
    fn rmw_sie(&mut self, new: u64, wr_mask: u64) -> u64 {
        let alias = (S_MODE_INTERRUPTS | LOCAL_INTERRUPTS) & self.st.mideleg;
        if self.st.virt() {
            return self.rmw_vsie(new, wr_mask) & alias;
        }
        self.rmw_mie(new, wr_mask & alias) & alias
    }

    /// `rmw_vsie64()`: the bits of `mie` delegated to VS mode, with the VS level bits at
    /// their S level positions.
    fn rmw_vsie(&mut self, new: u64, wr_mask: u64) -> u64 {
        let alias = (LOCAL_INTERRUPTS | VS_MODE_INTERRUPTS) & self.st.hideleg;
        let old = self.rmw_mie(vs_bits_up(new), vs_bits_up(wr_mask) & alias);
        vs_bits_down(old & alias)
    }

    /// `rmw_hideleg64()`.
    fn rmw_hideleg(&mut self, new: u64, wr_mask: u64) -> u64 {
        let m = wr_mask & VS_DELEGABLE_INTS;
        let old = self.st.hideleg & VS_DELEGABLE_INTS;
        self.st.hideleg = (self.st.hideleg & !m) | (new & m);
        old
    }

    /// Whether STIP is driven by `stimecmp` alone for this access: Sstc with
    /// `menvcfg.STCE` set, from M mode.
    fn stip_from_sstc(&self) -> bool {
        self.st.priv_lvl == PRV_M && self.st.menvcfg & MENVCFG_STCE != 0
    }

    /// `rmw_mip()`.
    fn rmw_mip(&mut self, new: u64, wr_mask: u64) -> u64 {
        self.rmw_mip64(CSR_MIP, new, wr_mask)
    }

    /// `rmw_mip64()` for `csrno`: the SEIP bit software writes is kept apart from the
    /// interrupt controller's input, STIP (and VSTIP with `henvcfg.STCE`) belong to Sstc
    /// when it is on, and the old value shows VSTIP while the VS timer has fired, except
    /// for `hvip`.
    fn rmw_mip64(&mut self, csrno: u32, new: u64, wr_mask: u64) -> u64 {
        let mut m = wr_mask & DELEGABLE_INTS;
        if self.stip_from_sstc() {
            m &= !MIP_STIP;
            if self.st.henvcfg & MENVCFG_STCE != 0 {
                m &= !MIP_VSTIP;
            }
        }
        self.hw.with_lines(&mut |l| {
            let mut new = new;
            if m & MIP_SEIP != 0 {
                l.software_seip = new & MIP_SEIP != 0;
                if l.external_seip {
                    new |= MIP_SEIP;
                }
            }
            let mut old = l.mip;
            // riscv_cpu_update_mip(): VSTIP is left alone while the VS timer drives it.
            let m = if m == MIP_VSTIP && l.vstime_irq { 0 } else { m };
            if m != 0 {
                l.mip = (old & !m) | (new & m);
            }
            if csrno != CSR_HVIP && l.vstime_irq {
                old |= MIP_VSTIP;
            }
            old
        })
    }

    /// `rmw_hvip64()` for `hvip`, or for `vsip` as `csrno`. Without AIA `hvien` is zero,
    /// so every bit aliases `mip`; for `vsip` only the bits delegated by `hideleg`. Like
    /// QEMU, reading `hvip` gives all of `mip`.
    fn rmw_hvip(&mut self, csrno: u32, new: u64, wr_mask: u64) -> u64 {
        let alias = if csrno == CSR_VSIP { self.st.hideleg } else { !0 };
        let old = self.rmw_mip64(csrno, new, wr_mask & alias & HVIP_WRITABLE_MASK);
        old & alias
    }

    /// `rmw_vsip64()`: the bits of `mip` delegated to VS mode, with the VS level bits at
    /// their S level positions; VSSIP is writable.
    fn rmw_vsip(&mut self, new: u64, wr_mask: u64) -> u64 {
        let mask = self.st.hideleg & VS_MODE_INTERRUPTS;
        let wr_mask = vs_bits_up(wr_mask) & mask & VSIP_WRITABLE_MASK;
        let old = self.rmw_hvip(CSR_VSIP, vs_bits_up(new), wr_mask);
        vs_bits_down(old & mask)
    }

    /// `rmw_sip64()` and `rmw_mvip64()` for `sip`: the delegated bits of `mip`, of which
    /// SSIP and the local interrupts are writable; in VS mode `vsip`.
    fn rmw_sip(&mut self, new: u64, wr_mask: u64) -> u64 {
        if self.st.virt() {
            let old = self.rmw_vsip(new, wr_mask);
            return old & self.st.mideleg & (S_MODE_INTERRUPTS | LOCAL_INTERRUPTS);
        }
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
    use crate::cpu::{MIP_MEIP, MIP_MSIP, MIP_MTIP, MSTATUS_UBE, MSTATUS64_SD, VM_SV39, VM_SV48};

    #[derive(Default)]
    struct FakeHw {
        lines: RefCell<CpuLines>,
        time: Option<u64>,
        ticks: Cell<u64>,
        /// The timer and its compare value at each `write_timecmp()`.
        timecmp: RefCell<Vec<(SstcTimer, u64)>>,
        /// The `is_m` and `enable` of each `stce_changed()`.
        stce: RefCell<Vec<(bool, bool)>>,
        /// The delay of each `pmu_timer()`.
        pmu_timer: RefCell<Vec<u64>>,
        cfg: RiscvCfg,
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

        fn write_timecmp(&self, st: &CpuRiscvState, timer: SstcTimer) {
            let v = if timer == SstcTimer::S { st.stimecmp } else { st.vstimecmp };
            self.timecmp.borrow_mut().push((timer, v));
        }

        fn stce_changed(&self, _st: &CpuRiscvState, is_m: bool, enable: bool) {
            self.stce.borrow_mut().push((is_m, enable));
        }

        fn pmu_timer(&self, delay_ns: u64) {
            self.pmu_timer.borrow_mut().push(delay_ns);
        }

        fn cfg(&self) -> &RiscvCfg {
            &self.cfg
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
        // UXL and SXL stay at 2; FS is writable with F, VS only with V; XS, UBE and SD are
        // not writable; FS dirty sets SD.
        let v = r(&mut c, CSR_MSTATUS).unwrap();
        let want = MSTATUS_WRITE_MASK | MSTATUS_FS | (2 << 34) | (2 << 32) | MSTATUS64_SD;
        assert_eq!(v, want);
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
        // With H the VS level and guest external interrupts are always delegated.
        assert_eq!(c.st.mideleg, HS_MODE_INTERRUPTS);
        w(&mut c, CSR_MIDELEG, 0).unwrap();
        assert_eq!(c.st.mideleg, HS_MODE_INTERRUPTS);
        w(&mut c, CSR_MIDELEG, u64::MAX).unwrap();
        assert_eq!(
            c.st.mideleg,
            (1 << 1)
                | (1 << 2)
                | (1 << 5)
                | (1 << 6)
                | (1 << 9)
                | (1 << 10)
                | (1 << 12)
                | (1 << 13)
        );
        w(&mut c, CSR_MIE, u64::MAX).unwrap();
        assert_eq!(c.st.mie, ALL_INTS);
        // Without H the hypervisor interrupt enables stay clear.
        c.st.misa &= !crate::cpu::RVH;
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
        assert_eq!(hw.stce.borrow().as_slice(), &[(true, true)]);
        w(&mut c, CSR_MIP, MIP_STIP).unwrap();
        assert_eq!(hw.lines.borrow().mip & MIP_STIP, 0);
        w(&mut c, CSR_STIMECMP, 1234).unwrap();
        assert_eq!(hw.timecmp.borrow().last(), Some(&(SstcTimer::S, 1234)));
        // VSTIP is writable in mip until henvcfg.STCE is set too.
        w(&mut c, CSR_MIP, MIP_VSTIP).unwrap();
        assert_eq!(hw.lines.borrow().mip & MIP_VSTIP, MIP_VSTIP);
        w(&mut c, CSR_HENVCFG, MENVCFG_STCE).unwrap();
        assert_eq!(hw.stce.borrow().last(), Some(&(false, true)));
        w(&mut c, CSR_MIP, 0).unwrap();
        assert_eq!(hw.lines.borrow().mip & MIP_VSTIP, MIP_VSTIP);
        w(&mut c, CSR_MENVCFG, 0).unwrap();
        // Clearing menvcfg.STCE clears henvcfg.STCE too, but as in QEMU only the S
        // timer hears about it: write_henvcfg() compares the old henvcfg with itself.
        assert_eq!(hw.stce.borrow().as_slice(), &[(true, true), (false, true), (true, false)]);
        assert_eq!(c.st.henvcfg, 0);
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
        assert_eq!(c.st.mhpmevent[3], u64::MAX, "VSINH and VUINH exist with H");
        c.st.misa &= !crate::cpu::RVH;
        w(&mut c, CSR_MHPMEVENT3, u64::MAX).unwrap();
        assert_eq!(c.st.mhpmevent[3], !(pmu::MHPMEVENT_VSINH | pmu::MHPMEVENT_VUINH));
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
        assert_eq!(r(&mut c, 0x747), ILL, "mseccfg needs Smepmp, Zkr, Smmpm or Zicfilp");
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
    fn seed_and_mseccfg() {
        let hw = FakeHw { cfg: RiscvCfg::max(), ..FakeHw::default() };
        let mut c = csrs(&hw);
        // M mode reads 16 bits of entropy with the ES16 status.
        let v = c.rw(CSR_SEED, true, 0, 0).unwrap();
        assert_eq!(v & !0xffff, SEED_OPST_ES16);
        // Below M mode seed needs mseccfg.SSEED or USEED; VS mode never reaches it.
        c.st.priv_lvl = PRV_S;
        assert_eq!(c.rw(CSR_SEED, true, 0, 0), ILL);
        c.st.mseccfg = pmp::MSECCFG_SSEED;
        assert!(c.rw(CSR_SEED, true, 0, 0).is_ok());
        c.st.virt_enabled = 1;
        assert_eq!(c.rw(CSR_SEED, true, 0, 0), Err(EXCP_VIRT_INSTRUCTION_FAULT));
        c.st.virt_enabled = 0;
        c.st.priv_lvl = PRV_U;
        assert_eq!(c.rw(CSR_SEED, true, 0, 0), ILL);
        c.st.mseccfg = pmp::MSECCFG_USEED;
        assert!(c.rw(CSR_SEED, true, 0, 0).is_ok());
        // mseccfg: with Smepmp, MML and MMWP are sticky and flush the TLB.
        c.st.priv_lvl = PRV_M;
        w(&mut c, CSR_MSECCFG, pmp::MSECCFG_MML).unwrap();
        assert!(c.flush);
        w(&mut c, CSR_MSECCFG, 0).unwrap();
        assert_eq!(r(&mut c, CSR_MSECCFG), Ok(pmp::MSECCFG_MML));
    }

    #[test]
    fn smstateen() {
        let hw = FakeHw { cfg: RiscvCfg::max(), ..FakeHw::default() };
        let mut c = csrs(&hw);
        // The defaults have F, so FCSR is not writable.
        w(&mut c, CSR_MSTATEEN0, u64::MAX).unwrap();
        let m = c.st.mstateen[0];
        assert_eq!(
            m & (SMSTATEEN_STATEEN | SMSTATEEN0_HSENVCFG),
            SMSTATEEN_STATEEN | SMSTATEEN0_HSENVCFG
        );
        assert_eq!(m & SMSTATEEN0_FCSR, 0);
        w(&mut c, CSR_MSTATEEN0 + 1, u64::MAX).unwrap();
        assert_eq!(c.st.mstateen[1], SMSTATEEN_STATEEN);
        // hstateen only gets the bits mstateen has.
        w(&mut c, CSR_MSTATEEN0, SMSTATEEN0_HSENVCFG).unwrap();
        w(&mut c, CSR_HSTATEEN0, u64::MAX).unwrap();
        assert_eq!(r(&mut c, CSR_HSTATEEN0), Ok(SMSTATEEN0_HSENVCFG));
        // Below M mode, senvcfg follows mstateen0.ENVCFG and in VS mode hstateen0.ENVCFG.
        c.st.priv_lvl = PRV_S;
        assert_eq!(r(&mut c, CSR_SENVCFG), Ok(0));
        assert_eq!(r(&mut c, CSR_HSTATEEN0), ILL, "mstateen0.SE0 is clear");
        assert_eq!(r(&mut c, CSR_SSTATEEN0), ILL, "mstateen0.SE0 is clear");
        c.st.virt_enabled = 1;
        c.st.hstateen[0] = 0;
        assert_eq!(r(&mut c, CSR_SENVCFG), Err(EXCP_VIRT_INSTRUCTION_FAULT));
        c.st.virt_enabled = 0;
        c.st.mstateen[0] = 0;
        assert_eq!(r(&mut c, CSR_SENVCFG), ILL);
        assert_eq!(w(&mut c, CSR_HENVCFG, 0), ILL);
        // Without Smstateen the registers do not exist.
        let hw = FakeHw::default();
        let mut c = csrs(&hw);
        for csrno in [CSR_MSTATEEN0, CSR_HSTATEEN0, CSR_SSTATEEN0] {
            assert_eq!(r(&mut c, csrno), ILL, "{csrno:#x}");
        }
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
        assert_eq!(c.st.mcountinhibit, 0x7fff8 | COUNTEREN_CY | COUNTEREN_IR);
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
        // An hpm counter given the cycle event counts ticks.
        w(&mut c, CSR_MHPMEVENT3 + 1, pmu::EVENT_HW_CPU_CYCLES).unwrap();
        w(&mut c, CSR_MHPMCOUNTER3 + 1, 10).unwrap();
        hw.ticks.set(5007);
        assert_eq!(r(&mut c, CSR_MHPMCOUNTER3 + 1), Ok(17));
        assert!(hw.pmu_timer.borrow().is_empty(), "no overflow timer without Sscofpmf");
        assert_eq!(r(&mut c, CSR_SCOUNTOVF), ILL);
        assert_eq!(r(&mut c, CSR_MCYCLECFG), ILL);
    }

    #[test]
    fn sscofpmf_and_smcntrpmf() {
        let hw = FakeHw { cfg: RiscvCfg::max(), ..FakeHw::default() };
        let mut c = csrs(&hw);
        w(&mut c, CSR_MHPMEVENT3, pmu::EVENT_HW_CPU_CYCLES).unwrap();
        hw.ticks.set(100);
        w(&mut c, CSR_MHPMCOUNTER3, u64::MAX - 49).unwrap();
        assert_eq!(*hw.pmu_timer.borrow(), [50]);
        // Inhibiting and starting it again arms the timer for the rest.
        w(&mut c, CSR_MCOUNTINHIBIT, 1 << 3).unwrap();
        hw.ticks.set(120);
        w(&mut c, CSR_MCOUNTINHIBIT, 0).unwrap();
        assert_eq!(*hw.pmu_timer.borrow(), [50, 50]);
        c.st.mhpmevent[3] |= pmu::MHPMEVENT_OF;
        c.st.mhpmevent[5] |= pmu::MHPMEVENT_OF;
        assert_eq!(r(&mut c, CSR_SCOUNTOVF), Ok((1 << 3) | (1 << 5)));
        c.st.priv_lvl = PRV_S;
        c.st.mcounteren = 1 << 5;
        assert_eq!(r(&mut c, CSR_SCOUNTOVF), Ok(1 << 5));
        c.st.priv_lvl = PRV_M;
        // mcyclecfg keeps the inhibit bits of the modes the hart has.
        w(&mut c, CSR_MCYCLECFG, u64::MAX).unwrap();
        assert_eq!(r(&mut c, CSR_MCYCLECFG), Ok(u64::MAX));
        c.st.misa &= !crate::cpu::RVH;
        w(&mut c, CSR_MINSTRETCFG, u64::MAX).unwrap();
        assert_eq!(c.st.minstretcfg, !(pmu::MHPMEVENT_VSINH | pmu::MHPMEVENT_VUINH));
        // With M mode inhibited mcycle only counts the ticks below M mode.
        c.st.misa |= crate::cpu::RVH;
        w(&mut c, CSR_MCYCLECFG, pmu::MHPMEVENT_MINH).unwrap();
        w(&mut c, CSR_MCYCLE, 0).unwrap();
        hw.ticks.set(500);
        assert_eq!(r(&mut c, CSR_MCYCLE), Ok(0));
        pmu::update_fixed_ctrs(&mut c.st, 500, PRV_S, false);
        c.st.priv_lvl = PRV_S;
        hw.ticks.set(530);
        pmu::update_fixed_ctrs(&mut c.st, 530, PRV_M, false);
        c.st.priv_lvl = PRV_M;
        hw.ticks.set(900);
        assert_eq!(r(&mut c, CSR_MCYCLE), Ok(30));
    }

    #[test]
    fn indirect_csrs_and_counter_delegation() {
        let hw = FakeHw { cfg: RiscvCfg::max(), ..FakeHw::default() };
        let mut c = csrs(&hw);
        // With Smcsrind the select registers hold 12 bits; 0x354 is not an alias.
        assert_eq!(w(&mut c, CSR_MISELECT, u64::MAX), Ok(0));
        assert_eq!(r(&mut c, CSR_MISELECT), Ok(0xfff));
        assert_eq!(r(&mut c, CSR_MIREG4 - 1), ILL);
        // Unimplemented and AIA ranges raise illegal instruction exceptions.
        assert_eq!(r(&mut c, CSR_MIREG), ILL);
        w(&mut c, CSR_SISELECT, ISELECT_IPRIO0).unwrap();
        assert_eq!(r(&mut c, CSR_SIREG), ILL);
        // The counters need menvcfg.CDE and their mcounteren bit.
        w(&mut c, CSR_SISELECT, ISELECT_CD_FIRST + 3).unwrap();
        assert_eq!(r(&mut c, CSR_SIREG), ILL);
        assert_eq!(r(&mut c, CSR_SCOUNTINHIBIT), ILL);
        w(&mut c, CSR_MENVCFG, MENVCFG_CDE).unwrap();
        assert_eq!(r(&mut c, CSR_MENVCFG), Ok(MENVCFG_CDE));
        assert_eq!(r(&mut c, CSR_SIREG), ILL);
        w(&mut c, CSR_MCOUNTEREN, u64::MAX).unwrap();
        w(&mut c, CSR_MCOUNTINHIBIT, u64::MAX).unwrap();
        c.st.priv_lvl = PRV_S;
        // Below M mode mstateen0.SVSLCT guards the indirect registers.
        assert_eq!(r(&mut c, CSR_SIREG), ILL);
        c.st.mstateen[0] = u64::MAX;
        c.st.hstateen[0] = u64::MAX;
        assert_eq!(w(&mut c, CSR_SIREG, 77), Ok(0));
        assert_eq!(r(&mut c, CSR_SIREG), Ok(77));
        assert_eq!(c.st.mhpmcounter_val[3], 77);
        // Only whole register writes reach the counter.
        assert_eq!(c.rw(CSR_SIREG, true, u64::MAX, 0xff), ILL);
        // sireg2 is the event, without MINH; sireg3 to sireg6 are not.
        assert_eq!(w(&mut c, CSR_SIREG2, u64::MAX), Ok(0));
        assert_eq!(c.st.mhpmevent[3], !pmu::MHPMEVENT_MINH);
        c.st.mhpmevent[3] |= pmu::MHPMEVENT_MINH;
        assert_eq!(r(&mut c, CSR_SIREG2), Ok(!pmu::MHPMEVENT_MINH));
        for csrno in [CSR_SIREG2 + 1, CSR_SIREG4, CSR_SIREG5, CSR_SIREG6] {
            assert_eq!(r(&mut c, csrno), ILL, "{csrno:#x}");
        }
        // For mcycle sireg2 is mcyclecfg, whose MINH a read clears; time has nothing.
        c.st.mcyclecfg = u64::MAX;
        w(&mut c, CSR_SISELECT, ISELECT_CD_FIRST).unwrap();
        assert_eq!(r(&mut c, CSR_SIREG2), Ok(!pmu::MHPMEVENT_MINH));
        assert_eq!(c.st.mcyclecfg, !pmu::MHPMEVENT_MINH);
        w(&mut c, CSR_SISELECT, ISELECT_CD_FIRST + 1).unwrap();
        assert_eq!(r(&mut c, CSR_SIREG), ILL);
        // scountinhibit is the delegated part of mcountinhibit.
        c.st.mcounteren = 0b1101;
        assert_eq!(r(&mut c, CSR_SCOUNTINHIBIT), Ok(0b1101));
        w(&mut c, CSR_SCOUNTINHIBIT, 0b1000).unwrap();
        assert_eq!(c.st.mcountinhibit, 0b1000);
        // In VS mode sireg is vsireg, which never reaches a counter, and siselect is
        // vsiselect; scountinhibit and scountovf are virtual instruction faults.
        c.st.virt_enabled = 1;
        c.st.mcounteren = u64::MAX;
        w(&mut c, CSR_SISELECT, ISELECT_CD_FIRST + 3).unwrap();
        assert_eq!(c.st.vsiselect, ISELECT_CD_FIRST + 3);
        assert_eq!(r(&mut c, CSR_SIREG), Err(EXCP_VIRT_INSTRUCTION_FAULT));
        assert_eq!(r(&mut c, CSR_SIREG2), ILL);
        assert_eq!(r(&mut c, CSR_SCOUNTINHIBIT), Err(EXCP_VIRT_INSTRUCTION_FAULT));
        assert_eq!(r(&mut c, CSR_SCOUNTOVF), Err(EXCP_VIRT_INSTRUCTION_FAULT));
        // Without the extensions the registers do not exist.
        let hw = FakeHw::default();
        let mut c = csrs(&hw);
        for csrno in [CSR_MISELECT, CSR_MIREG, CSR_SIREG2, CSR_VSIREG, CSR_SCOUNTINHIBIT] {
            assert_eq!(r(&mut c, csrno), ILL, "{csrno:#x}");
        }
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

    #[test]
    fn vector_csrs() {
        // Without Zve32x the vector CSRs do not exist and mstatus.VS is read only zero.
        let hw = FakeHw::default();
        let mut c = csrs(&hw);
        assert_eq!(r(&mut c, CSR_VLENB), ILL);
        w(&mut c, CSR_MSTATUS, MSTATUS_VS).unwrap();
        assert_eq!(c.st.mstatus & MSTATUS_VS, 0);

        let hw = FakeHw { cfg: RiscvCfg::max(), ..FakeHw::default() };
        let mut c = csrs(&hw);
        // mstatus.VS starts off, so the CSRs are illegal.
        assert_eq!(r(&mut c, CSR_VL), ILL);
        w(&mut c, CSR_MSTATUS, 1 << 9).unwrap();
        assert_eq!(c.st.mstatus & MSTATUS_VS, 1 << 9);
        assert_eq!(r(&mut c, CSR_VLENB), Ok(16));
        assert_eq!(r(&mut c, CSR_VTYPE), Ok(1 << 63), "vill is set at reset");
        assert_eq!(c.rw(CSR_VL, true, 0, 0), ILL, "vl is read only");
        w(&mut c, CSR_VSTART, u64::MAX).unwrap();
        assert_eq!(c.st.vstart, 0x7f);
        // Any write makes the vector state dirty, and SD follows.
        assert_eq!(c.st.mstatus & MSTATUS_VS, MSTATUS_VS);
        assert_ne!(r(&mut c, CSR_MSTATUS).unwrap() & MSTATUS64_SD, 0);
        w(&mut c, CSR_VCSR, 0x7).unwrap();
        assert_eq!((c.st.vxrm, c.st.vxsat), (3, 1));
        w(&mut c, CSR_VXRM, 0x6).unwrap();
        assert_eq!(r(&mut c, CSR_VCSR), Ok(0x5));
        w(&mut c, CSR_VXSAT, 0x2).unwrap();
        assert_eq!(r(&mut c, CSR_VXSAT), Ok(0));
        // sstatus.VS is writable too.
        w(&mut c, CSR_SSTATUS, 0).unwrap();
        assert_eq!(c.st.mstatus & MSTATUS_VS, 0);
    }

    #[test]
    fn hypervisor_csrs_need_h() {
        let hw = FakeHw::default();
        let mut c = csrs(&hw);
        c.st.misa &= !crate::cpu::RVH;
        for csrno in [CSR_HSTATUS, CSR_HGATP, CSR_VSSTATUS, CSR_VSATP, CSR_MTVAL2, CSR_HGEIP] {
            assert_eq!(r(&mut c, csrno), ILL, "{csrno:#x}");
        }
    }

    #[test]
    fn hypervisor_csr_masks() {
        let hw = FakeHw::default();
        let mut c = csrs(&hw);
        // hstatus: VSXL reads as 2 and VSBE as 0; HUKTE and HUPMM are not writable.
        w(&mut c, CSR_HSTATUS, u64::MAX).unwrap();
        assert_eq!(c.st.hstatus, !(HSTATUS_HUKTE | HSTATUS_HUPMM));
        let h = r(&mut c, CSR_HSTATUS).unwrap();
        assert_eq!(get_field(h, HSTATUS_VSXL), 2);
        assert_eq!(h & HSTATUS_VSBE, 0);
        // hedeleg: no ecalls from S, VS or M mode, and no guest faults.
        w(&mut c, CSR_HEDELEG, u64::MAX).unwrap();
        assert_eq!(c.st.hedeleg, VS_DELEGABLE_EXCPS);
        assert_eq!(c.st.hedeleg & ((0xf << 20) | (7 << 9)), 0);
        // hideleg: only the VS level interrupts.
        assert_eq!(w(&mut c, CSR_HIDELEG, u64::MAX), Ok(0));
        assert_eq!(c.st.hideleg, VS_MODE_INTERRUPTS | (LOCAL_INTERRUPTS & !MIP_LCOFIP));
        assert_eq!(r(&mut c, CSR_HIDELEG), Ok(c.st.hideleg));
        // hcounteren like mcounteren.
        w(&mut c, CSR_HCOUNTEREN, u64::MAX).unwrap();
        assert_eq!(c.st.hcounteren, 0x7fff8 | COUNTEREN_CY | COUNTEREN_TM | COUNTEREN_IR);
        // GEILEN is 0.
        w(&mut c, CSR_HGEIE, u64::MAX).unwrap();
        assert_eq!(r(&mut c, CSR_HGEIE), Ok(0));
        assert_eq!(r(&mut c, CSR_HGEIP), Ok(0));
        assert_eq!(c.rw(CSR_HGEIP, true, 0, 0), ILL, "hgeip is read only");
        // htinst ignores writes; mtinst and mtval2 hold them.
        w(&mut c, CSR_HTINST, 0x1234).unwrap();
        assert_eq!(r(&mut c, CSR_HTINST), Ok(0));
        w(&mut c, CSR_MTINST, 0x1234).unwrap();
        w(&mut c, CSR_MTVAL2, 0x5678).unwrap();
        assert_eq!((c.st.mtinst, c.st.mtval2), (0x1234, 0x5678));
        // hgatp and vsatp take the supported modes and flush.
        w(&mut c, CSR_HGATP, (VM_SV48 << 60) | 0x80000).unwrap();
        assert_eq!(c.st.hgatp, (VM_SV48 << 60) | 0x80000);
        assert!(c.flush);
        c.flush = false;
        w(&mut c, CSR_VSATP, 7 << 60).unwrap();
        assert_eq!(c.st.vsatp, 0);
        assert!(!c.flush);
        // vstvec drops reserved modes.
        w(&mut c, CSR_VSTVEC, 0x8000_0001).unwrap();
        w(&mut c, CSR_VSTVEC, 0x8000_0002).unwrap();
        assert_eq!(c.st.vstvec, 0x8000_0001);
        // vsstatus is stored as written, with UXL 64 bit and no SDT.
        w(&mut c, CSR_VSSTATUS, u64::MAX).unwrap();
        assert_eq!(c.st.vsstatus, set_field(!SSTATUS_SDT, VSSTATUS64_UXL, 2));
        // henvcfg: STCE and ADUE follow menvcfg, which has ADUE set at reset.
        w(&mut c, CSR_MENVCFG, 0).unwrap();
        w(&mut c, CSR_HENVCFG, u64::MAX).unwrap();
        assert_eq!(c.st.henvcfg, SENVCFG_WRITE_MASK);
        w(&mut c, CSR_MENVCFG, MENVCFG_ADUE).unwrap();
        w(&mut c, CSR_HENVCFG, u64::MAX).unwrap();
        assert_eq!(r(&mut c, CSR_HENVCFG), Ok(SENVCFG_WRITE_MASK | MENVCFG_ADUE));
    }

    #[test]
    fn hypervisor_interrupt_csrs() {
        let hw = FakeHw::default();
        let mut c = csrs(&hw);
        // hvip writes the VS level bits of mip; hip only VSSIP.
        w(&mut c, CSR_HVIP, u64::MAX).unwrap();
        assert_eq!(hw.lines.borrow().mip, VS_MODE_INTERRUPTS);
        w(&mut c, CSR_HIP, 0).unwrap();
        assert_eq!(hw.lines.borrow().mip, MIP_VSTIP | MIP_VSEIP);
        assert_eq!(r(&mut c, CSR_HIP), Ok(MIP_VSTIP | MIP_VSEIP));
        // hie: the HS mode bits of mie.
        assert_eq!(w(&mut c, CSR_HIE, u64::MAX), Ok(0));
        assert_eq!(c.st.mie, HS_MODE_INTERRUPTS);
        // vsie and vsip show the delegated VS bits at their S level positions.
        assert_eq!(r(&mut c, CSR_VSIE), Ok(0));
        c.st.hideleg = MIP_VSTIP | MIP_VSSIP;
        assert_eq!(r(&mut c, CSR_VSIE), Ok(MIP_STIP | MIP_SSIP));
        assert_eq!(r(&mut c, CSR_VSIP), Ok(MIP_STIP));
        w(&mut c, CSR_VSIP, MIP_SSIP).unwrap();
        assert_eq!(hw.lines.borrow().mip, MIP_VSSIP | MIP_VSTIP | MIP_VSEIP);
        // The fired VS timer shows in vsip and hip but not in hvip.
        hw.lines.borrow_mut().mip = 0;
        hw.lines.borrow_mut().vstime_irq = true;
        assert_eq!(r(&mut c, CSR_VSIP), Ok(MIP_STIP));
        assert_eq!(r(&mut c, CSR_HIP), Ok(MIP_VSTIP));
        assert_eq!(r(&mut c, CSR_HVIP), Ok(0));
        // In VS mode sie and sip are vsie and vsip, within mideleg.
        c.st.virt_enabled = 1;
        c.st.priv_lvl = PRV_S;
        c.st.mideleg |= S_MODE_INTERRUPTS;
        assert_eq!(r(&mut c, CSR_SIP), Ok(MIP_STIP));
        assert_eq!(w(&mut c, CSR_SIE, 0), Ok(MIP_STIP | MIP_SSIP));
        assert_eq!(c.st.mie, MIP_SGEIP | MIP_VSEIP);
    }

    #[test]
    fn virtual_instruction_faults() {
        let hw = FakeHw { time: Some(100), ..FakeHw::default() };
        let mut c = csrs(&hw);
        c.st.htimedelta = 5;
        // HS mode reaches the hypervisor CSRs; VS mode faults on them.
        c.st.priv_lvl = PRV_S;
        assert_eq!(r(&mut c, CSR_HSTATUS), Ok(2 << 32));
        assert_eq!(r(&mut c, CSR_MSTATUS), ILL);
        c.st.virt_enabled = 1;
        assert_eq!(r(&mut c, CSR_HSTATUS), Err(EXCP_VIRT_INSTRUCTION_FAULT));
        assert_eq!(r(&mut c, CSR_VSSTATUS), Err(EXCP_VIRT_INSTRUCTION_FAULT));
        assert_eq!(r(&mut c, CSR_MSTATUS), ILL);
        // VU mode faults on S mode CSRs too.
        c.st.priv_lvl = PRV_U;
        assert_eq!(r(&mut c, CSR_SSCRATCH), Err(EXCP_VIRT_INSTRUCTION_FAULT));
        // satp with hstatus.VTVM.
        c.st.priv_lvl = PRV_S;
        c.st.hstatus = HSTATUS_VTVM;
        assert_eq!(r(&mut c, CSR_SATP), Err(EXCP_VIRT_INSTRUCTION_FAULT));
        // Counters: mcounteren gives illegal, hcounteren virtual instruction.
        assert_eq!(r(&mut c, CSR_TIME), ILL);
        c.st.mcounteren = COUNTEREN_TM;
        assert_eq!(r(&mut c, CSR_TIME), Err(EXCP_VIRT_INSTRUCTION_FAULT));
        c.st.hcounteren = COUNTEREN_TM;
        assert_eq!(r(&mut c, CSR_TIME), Ok(105), "time adds htimedelta in VS mode");
        c.st.priv_lvl = PRV_U;
        assert_eq!(r(&mut c, CSR_TIME), Err(EXCP_VIRT_INSTRUCTION_FAULT));
        // stimecmp needs hcounteren.TM and henvcfg.STCE in VS mode.
        c.st.priv_lvl = PRV_S;
        c.st.menvcfg = MENVCFG_STCE;
        c.st.hcounteren = 0;
        assert_eq!(r(&mut c, CSR_STIMECMP), Err(EXCP_VIRT_INSTRUCTION_FAULT));
        c.st.hcounteren = COUNTEREN_TM;
        c.st.henvcfg = MENVCFG_STCE;
        // ... and is vstimecmp there.
        w(&mut c, CSR_STIMECMP, 77).unwrap();
        assert_eq!((c.st.stimecmp, c.st.vstimecmp), (0, 77));
        assert_eq!(hw.timecmp.borrow().last(), Some(&(SstcTimer::Vs, 77)));
        // HS mode with TVM may not touch hgatp.
        c.st.virt_enabled = 0;
        c.st.mstatus |= MSTATUS_TVM;
        assert_eq!(r(&mut c, CSR_HGATP), ILL);
        assert_eq!(r(&mut c, CSR_VSATP), Ok(0));
    }
}
