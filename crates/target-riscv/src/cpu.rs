// SPDX-License-Identifier: GPL-2.0-or-later

//! `CPURISCVState` for RV64 and the CPU model: the parts of QEMU's `target/riscv/cpu.h`,
//! `cpu_bits.h` and `cpu.c` this crate needs.
//!
//! The one model is QEMU's default `rv64` CPU without the H extension: RV64GC (I, M, A, F,
//! D, C) with S and U modes, Sv39, Sv48 and Sv57, 16 PMP regions, Zicsr, Zifencei, Zicntr,
//! Zihpm, Zicbom, Zicbop, Zicboz, Zihintntl, Zihintpause, Zawrs, Zfa, Zba, Zbb, Zbc,
//! Zbs, Sstc, Svadu, Svvptc and Sdtrig, as `riscv_cpu_extensions[]` and the `rv64` class
//! enable them.
//!
//! [`CpuRiscvState`] is a plain struct with `#[repr(C)]`, so `offset_of!` gives the offset
//! of every field. The runtime keeps the state of a vCPU in a byte buffer (`env`), with the
//! struct's fields at [`ENV_TARGET_OFFSET`] plus their offset, little endian. Generated code
//! reaches registers through the offset constants below, and helpers copy the whole struct
//! in and out with [`CpuRiscvState::load`] and [`CpuRiscvState::store`]. No unsafe code is
//! involved.
//!
//! `mip` is not part of this state: devices on other threads change it, so it lives with the
//! interrupt lines in [`crate::tcg::Riscv`].

use std::mem::{offset_of, size_of};

use ruvm_jit::ENV_TARGET_OFFSET;

/// A field of [`CpuRiscvState`] that can be copied to and from `env`.
trait Field {
    fn put(&self, b: &mut [u8]);
    fn get(&mut self, b: &[u8]);
}

impl Field for u64 {
    fn put(&self, b: &mut [u8]) {
        b[..8].copy_from_slice(&self.to_le_bytes());
    }
    fn get(&mut self, b: &[u8]) {
        *self = u64::from_le_bytes(b[..8].try_into().expect("8 bytes"));
    }
}

impl<T: Field, const N: usize> Field for [T; N] {
    fn put(&self, b: &mut [u8]) {
        for (i, v) in self.iter().enumerate() {
            v.put(&mut b[size_of::<T>() * i..]);
        }
    }
    fn get(&mut self, b: &[u8]) {
        for (i, v) in self.iter_mut().enumerate() {
            v.get(&b[size_of::<T>() * i..]);
        }
    }
}

macro_rules! riscv_state {
    ($(#[$m:meta])* pub struct $name:ident { $($(#[$fm:meta])* pub $f:ident: $t:ty,)* }) => {
        $(#[$m])*
        #[repr(C)]
        #[derive(Clone, Debug, Default, PartialEq, Eq)]
        pub struct $name {
            $($(#[$fm])* pub $f: $t,)*
        }

        impl $name {
            /// Read the state from a vCPU's `env` buffer.
            pub fn load(env: &[u8]) -> $name {
                let mut s = $name::default();
                $(s.$f.get(&env[ENV_TARGET_OFFSET + offset_of!($name, $f)..]);)*
                s
            }

            /// Write the state into a vCPU's `env` buffer.
            pub fn store(&self, env: &mut [u8]) {
                $(self.$f.put(&mut env[ENV_TARGET_OFFSET + offset_of!($name, $f)..]);)*
            }
        }
    };
}

/// The number of PMP regions of the model (`pmp_regions`).
pub const PMP_REGIONS: usize = 16;
/// The number of debug triggers (`RV_MAX_TRIGGERS`).
pub const NUM_TRIGGERS: usize = 2;

riscv_state! {
    /// The RV64 register state, `CPURISCVState` cut down to RV64 without H and V.
    pub struct CpuRiscvState {
        /// x0 to x31; x0 is always zero.
        pub gpr: [u64; 32],
        /// f0 to f31, NaN boxed.
        pub fpr: [u64; 32],
        /// The PC.
        pub pc: u64,
        /// The address of the LR reservation, -1 when there is none.
        pub load_res: u64,
        /// The value LR loaded.
        pub load_val: u64,
        /// The dynamic rounding mode, `frm`.
        pub frm: u64,
        /// The accrued exception flags, `fflags`, in the RISC-V bit order.
        pub fflags: u64,
        /// The rounding mode the current instruction uses, as `set_rounding_mode()` leaves
        /// it in QEMU's `fp_status` (a RISC-V `rm` value, 0 to 4).
        pub fp_round: u64,
        /// The faulting address of the pending exception.
        pub badaddr: u64,
        /// The bits of the instruction that raised the pending exception.
        pub bins: u64,
        /// `excp_uw2`: the second unwind word of the instruction.
        pub excp_uw2: u64,
        /// The privilege level, `PRV_U`, `PRV_S` or `PRV_M`.
        pub priv_lvl: u64,
        /// `misa` with MXL in the top bits.
        pub misa: u64,
        /// `mstatus` without SD, which is computed when it is read.
        pub mstatus: u64,
        /// `mie`.
        pub mie: u64,
        /// `mideleg`.
        pub mideleg: u64,
        /// `medeleg`.
        pub medeleg: u64,
        /// `mtvec`.
        pub mtvec: u64,
        /// `stvec`.
        pub stvec: u64,
        /// `mepc`.
        pub mepc: u64,
        /// `sepc`.
        pub sepc: u64,
        /// `mcause`.
        pub mcause: u64,
        /// `scause`.
        pub scause: u64,
        /// `mtval`.
        pub mtval: u64,
        /// `stval`.
        pub stval: u64,
        /// `mscratch`.
        pub mscratch: u64,
        /// `sscratch`.
        pub sscratch: u64,
        /// `satp`.
        pub satp: u64,
        /// `mcounteren`.
        pub mcounteren: u64,
        /// `scounteren`.
        pub scounteren: u64,
        /// `mcountinhibit`.
        pub mcountinhibit: u64,
        /// `menvcfg`.
        pub menvcfg: u64,
        /// `senvcfg`.
        pub senvcfg: u64,
        /// `stimecmp`.
        pub stimecmp: u64,
        /// `mhartid`.
        pub mhartid: u64,
        /// The reset vector.
        pub resetvec: u64,
        /// The counter values as last written (`pmu_ctrs[].mhpmcounter_val`).
        pub mhpmcounter_val: [u64; 32],
        /// The free running count when they were written (`pmu_ctrs[].mhpmcounter_prev`).
        pub mhpmcounter_prev: [u64; 32],
        /// `mhpmevent3` to `mhpmevent31`, at their counter index.
        pub mhpmevent: [u64; 32],
        /// The PMP configuration bytes.
        pub pmpcfg: [u64; PMP_REGIONS],
        /// The PMP address registers.
        pub pmpaddr: [u64; PMP_REGIONS],
        /// The first address each PMP rule matches (`pmp_state.addr[].sa`).
        pub pmp_sa: [u64; PMP_REGIONS],
        /// The last address each PMP rule matches (`pmp_state.addr[].ea`).
        pub pmp_ea: [u64; PMP_REGIONS],
        /// The number of PMP rules that are not off.
        pub pmp_num_rules: u64,
        /// `tselect`.
        pub tselect: u64,
        /// `tdata1` of each trigger.
        pub tdata1: [u64; NUM_TRIGGERS],
        /// `tdata2` of each trigger.
        pub tdata2: [u64; NUM_TRIGGERS],
        /// `tdata3` of each trigger.
        pub tdata3: [u64; NUM_TRIGGERS],
        /// `mcontext`.
        pub mcontext: u64,
    }
}

/// The size of the `env` buffer a RISC-V vCPU needs.
pub const ENV_SIZE: usize = ENV_TARGET_OFFSET + size_of::<CpuRiscvState>();

/// The offset in `env` of the field at `off` in [`CpuRiscvState`].
pub const fn env_off(off: usize) -> usize {
    ENV_TARGET_OFFSET + off
}

/// The offset in `env` of x`r`.
pub const fn gpr_off(r: usize) -> usize {
    env_off(offset_of!(CpuRiscvState, gpr)) + 8 * r
}

/// The offset in `env` of f`r`.
pub const fn fpr_off(r: usize) -> usize {
    env_off(offset_of!(CpuRiscvState, fpr)) + 8 * r
}

/// `env` offsets of the fields generated code and the hot paths reach directly.
pub const PC: usize = env_off(offset_of!(CpuRiscvState, pc));
/// `load_res`.
pub const LOAD_RES: usize = env_off(offset_of!(CpuRiscvState, load_res));
/// `load_val`.
pub const LOAD_VAL: usize = env_off(offset_of!(CpuRiscvState, load_val));
/// `frm`.
pub const FRM: usize = env_off(offset_of!(CpuRiscvState, frm));
/// `fflags`.
pub const FFLAGS: usize = env_off(offset_of!(CpuRiscvState, fflags));
/// `fp_round`.
pub const FP_ROUND: usize = env_off(offset_of!(CpuRiscvState, fp_round));
/// `badaddr`.
pub const BADADDR: usize = env_off(offset_of!(CpuRiscvState, badaddr));
/// `bins`.
pub const BINS: usize = env_off(offset_of!(CpuRiscvState, bins));
/// `excp_uw2`.
pub const EXCP_UW2: usize = env_off(offset_of!(CpuRiscvState, excp_uw2));
/// `priv`.
pub const PRIV: usize = env_off(offset_of!(CpuRiscvState, priv_lvl));
/// `mstatus`.
pub const MSTATUS: usize = env_off(offset_of!(CpuRiscvState, mstatus));
/// `mie`.
pub const MIE: usize = env_off(offset_of!(CpuRiscvState, mie));
/// `mideleg`.
pub const MIDELEG: usize = env_off(offset_of!(CpuRiscvState, mideleg));
/// `satp`.
pub const SATP: usize = env_off(offset_of!(CpuRiscvState, satp));

// Privilege levels.

/// User mode.
pub const PRV_U: u64 = 0;
/// Supervisor mode.
pub const PRV_S: u64 = 1;
/// Machine mode.
pub const PRV_M: u64 = 3;

// MMU indexes, `MMUIdx_*`.

/// U mode.
pub const MMU_IDX_U: usize = 0;
/// S mode.
pub const MMU_IDX_S: usize = 1;
/// S mode with SUM set.
pub const MMU_IDX_S_SUM: usize = 2;
/// M mode.
pub const MMU_IDX_M: usize = 3;
/// The number of MMU indexes this port uses.
pub const NB_MMU_MODES: usize = 4;

// misa.

/// `MXL_RV64` in the top bits of `misa`.
pub const MISA_MXL_RV64: u64 = 2 << 62;

/// The `misa` bit of extension letter `c`.
pub const fn rvx(c: u8) -> u64 {
    1 << (c - b'A')
}

/// RVA.
pub const RVA: u64 = rvx(b'A');
/// RVC.
pub const RVC: u64 = rvx(b'C');
/// RVD.
pub const RVD: u64 = rvx(b'D');
/// RVF.
pub const RVF: u64 = rvx(b'F');
/// RVI.
pub const RVI: u64 = rvx(b'I');
/// RVM.
pub const RVM: u64 = rvx(b'M');
/// RVS.
pub const RVS: u64 = rvx(b'S');
/// RVU.
pub const RVU: u64 = rvx(b'U');

/// The extensions of the model in `misa`.
pub const MISA_EXT: u64 = RVI | RVM | RVA | RVF | RVD | RVC | RVS | RVU;

// mstatus.

/// SIE.
pub const MSTATUS_SIE: u64 = 0x2;
/// MIE.
pub const MSTATUS_MIE: u64 = 0x8;
/// SPIE.
pub const MSTATUS_SPIE: u64 = 0x20;
/// UBE.
pub const MSTATUS_UBE: u64 = 0x40;
/// MPIE.
pub const MSTATUS_MPIE: u64 = 0x80;
/// SPP.
pub const MSTATUS_SPP: u64 = 0x100;
/// VS.
pub const MSTATUS_VS: u64 = 0x600;
/// MPP.
pub const MSTATUS_MPP: u64 = 0x1800;
/// FS.
pub const MSTATUS_FS: u64 = 0x6000;
/// XS.
pub const MSTATUS_XS: u64 = 0x18000;
/// MPRV.
pub const MSTATUS_MPRV: u64 = 0x20000;
/// SUM.
pub const MSTATUS_SUM: u64 = 0x40000;
/// MXR.
pub const MSTATUS_MXR: u64 = 0x80000;
/// TVM.
pub const MSTATUS_TVM: u64 = 0x100000;
/// TW.
pub const MSTATUS_TW: u64 = 0x200000;
/// TSR.
pub const MSTATUS_TSR: u64 = 0x400000;
/// UXL.
pub const MSTATUS64_UXL: u64 = 3 << 32;
/// SXL.
pub const MSTATUS64_SXL: u64 = 3 << 34;
/// SD.
pub const MSTATUS64_SD: u64 = 1 << 63;

/// The `mstatus` bits `sstatus` shows, `sstatus_v1_10_mask` (UIE and UPIE are bits 0 and 4).
/// A read also shows UXL.
pub const SSTATUS_MASK: u64 = 0x1
    | MSTATUS_SIE
    | 0x10
    | MSTATUS_SPIE
    | MSTATUS_SPP
    | MSTATUS_VS
    | MSTATUS_FS
    | MSTATUS_XS
    | MSTATUS_SUM
    | MSTATUS_MXR;

/// The FS and VS field values.
pub const EXT_STATUS_DISABLED: u64 = 0;
/// Initial.
pub const EXT_STATUS_INITIAL: u64 = 1;
/// Clean.
pub const EXT_STATUS_CLEAN: u64 = 2;
/// Dirty.
pub const EXT_STATUS_DIRTY: u64 = 3;

// Interrupts.

/// Supervisor software interrupt.
pub const IRQ_S_SOFT: u32 = 1;
/// Machine software interrupt.
pub const IRQ_M_SOFT: u32 = 3;
/// Supervisor timer interrupt.
pub const IRQ_S_TIMER: u32 = 5;
/// Machine timer interrupt.
pub const IRQ_M_TIMER: u32 = 7;
/// Supervisor external interrupt.
pub const IRQ_S_EXT: u32 = 9;
/// Machine external interrupt.
pub const IRQ_M_EXT: u32 = 11;

/// `mip` bit of SSIP.
pub const MIP_SSIP: u64 = 1 << IRQ_S_SOFT;
/// MSIP.
pub const MIP_MSIP: u64 = 1 << IRQ_M_SOFT;
/// STIP.
pub const MIP_STIP: u64 = 1 << IRQ_S_TIMER;
/// MTIP.
pub const MIP_MTIP: u64 = 1 << IRQ_M_TIMER;
/// SEIP.
pub const MIP_SEIP: u64 = 1 << IRQ_S_EXT;
/// MEIP.
pub const MIP_MEIP: u64 = 1 << IRQ_M_EXT;
/// The VS interrupts (VSSIP, VSTIP, VSEIP), present in `mideleg` as read only ones.
pub const MIP_VS_BITS: u64 = (1 << 2) | (1 << 6) | (1 << 10);
/// LCOFIP.
pub const MIP_LCOFIP: u64 = 1 << 13;
/// SGEIP.
pub const MIP_SGEIP: u64 = 1 << 12;

/// The supervisor interrupts, `S_MODE_INTERRUPTS`.
pub const S_MODE_INTERRUPTS: u64 = MIP_SSIP | MIP_STIP | MIP_SEIP;
/// The machine interrupts, `M_MODE_INTERRUPTS`.
pub const M_MODE_INTERRUPTS: u64 = MIP_MSIP | MIP_MTIP | MIP_MEIP;

// Exceptions, `RISCV_EXCP_*`.

/// Instruction address misaligned.
pub const EXCP_INST_ADDR_MIS: i32 = 0;
/// Instruction access fault.
pub const EXCP_INST_ACCESS_FAULT: i32 = 1;
/// Illegal instruction.
pub const EXCP_ILLEGAL_INST: i32 = 2;
/// Breakpoint.
pub const EXCP_BREAKPOINT: i32 = 3;
/// Load address misaligned.
pub const EXCP_LOAD_ADDR_MIS: i32 = 4;
/// Load access fault.
pub const EXCP_LOAD_ACCESS_FAULT: i32 = 5;
/// Store/AMO address misaligned.
pub const EXCP_STORE_AMO_ADDR_MIS: i32 = 6;
/// Store/AMO access fault.
pub const EXCP_STORE_AMO_ACCESS_FAULT: i32 = 7;
/// Environment call from U mode.
pub const EXCP_U_ECALL: i32 = 8;
/// Environment call from S mode.
pub const EXCP_S_ECALL: i32 = 9;
/// Environment call from M mode.
pub const EXCP_M_ECALL: i32 = 11;
/// Instruction page fault.
pub const EXCP_INST_PAGE_FAULT: i32 = 12;
/// Load page fault.
pub const EXCP_LOAD_PAGE_FAULT: i32 = 13;
/// Store/AMO page fault.
pub const EXCP_STORE_PAGE_FAULT: i32 = 15;
/// A semihosting call (QEMU's internal `RISCV_EXCP_SEMIHOST`).
pub const EXCP_SEMIHOST: i32 = 0x3f;
/// The interrupt flag in an exception index.
pub const EXCP_INT_FLAG: i32 = i32::MIN;

/// `RISCV_UW2_ALWAYS_STORE_AMO`: a load fault of this instruction is reported as a store
/// fault.
pub const UW2_ALWAYS_STORE_AMO: u64 = 1;

// satp.

/// The MODE field of `satp`.
pub const SATP64_MODE: u64 = 0xF << 60;
/// The ASID field.
pub const SATP64_ASID: u64 = 0xFFFF << 44;
/// The PPN field.
pub const SATP64_PPN: u64 = 0x0FFF_FFFF_FFFF;
/// Bare.
pub const VM_MBARE: u64 = 0;
/// Sv39.
pub const VM_SV39: u64 = 8;
/// Sv48.
pub const VM_SV48: u64 = 9;
/// Sv57.
pub const VM_SV57: u64 = 10;

// menvcfg and senvcfg.

/// FIOM.
pub const MENVCFG_FIOM: u64 = 1;
/// CBIE.
pub const MENVCFG_CBIE: u64 = 3 << 4;
/// CBCFE.
pub const MENVCFG_CBCFE: u64 = 1 << 6;
/// CBZE.
pub const MENVCFG_CBZE: u64 = 1 << 7;
/// ADUE.
pub const MENVCFG_ADUE: u64 = 1 << 61;
/// STCE.
pub const MENVCFG_STCE: u64 = 1 << 63;

// Page table entries.

/// Valid.
pub const PTE_V: u64 = 0x001;
/// Read.
pub const PTE_R: u64 = 0x002;
/// Write.
pub const PTE_W: u64 = 0x004;
/// Execute.
pub const PTE_X: u64 = 0x008;
/// User.
pub const PTE_U: u64 = 0x010;
/// Global.
pub const PTE_G: u64 = 0x020;
/// Accessed.
pub const PTE_A: u64 = 0x040;
/// Dirty.
pub const PTE_D: u64 = 0x080;
/// Page based memory types.
pub const PTE_PBMT: u64 = 0x6000_0000_0000_0000;
/// NAPOT translation.
pub const PTE_N: u64 = 0x8000_0000_0000_0000;
/// The reserved bits, `PTE_RESERVED(false)` (the model has no Svrsw60t59b).
pub const PTE_RESERVED: u64 = 0x1FC0_0000_0000_0000;
/// All attribute bits.
pub const PTE_ATTR: u64 = PTE_N | PTE_PBMT;
/// The shift of the PPN.
pub const PTE_PPN_SHIFT: u32 = 10;
/// The PPN field.
pub const PTE_PPN_MASK: u64 = 0x003F_FFFF_FFFF_FC00;

// Counters.

/// The counters the PMU has besides cycle, time and instret (`pmu-mask`, 3 to 18).
pub const PMU_AVAIL_CTRS: u64 = 0x7fff8;
/// CY in the counter enable registers.
pub const COUNTEREN_CY: u64 = 1;
/// TM.
pub const COUNTEREN_TM: u64 = 1 << 1;
/// IR.
pub const COUNTEREN_IR: u64 = 1 << 2;

// Floating point.

/// The `fflags` bits.
pub const FFLAGS_MASK: u64 = 0x1f;
/// Round to nearest, ties to even.
pub const RISCV_FRM_RNE: u64 = 0;
/// Round towards zero.
pub const RISCV_FRM_RTZ: u64 = 1;
/// Round down.
pub const RISCV_FRM_RDN: u64 = 2;
/// Round up.
pub const RISCV_FRM_RUP: u64 = 3;
/// Round to nearest, ties to max magnitude.
pub const RISCV_FRM_RMM: u64 = 4;
/// Round to odd, QEMU's internal `RISCV_FRM_ROD`.
pub const RISCV_FRM_ROD: u64 = 8;
/// Dynamic: the rounding mode in `frm`.
pub const RISCV_FRM_DYN: u64 = 7;

impl CpuRiscvState {
    /// The state after `riscv_cpu_reset_hold()` for hart `hartid` with reset vector
    /// `resetvec`.
    pub fn reset(hartid: u64, resetvec: u64) -> CpuRiscvState {
        let mut s = CpuRiscvState {
            misa: MISA_MXL_RV64 | MISA_EXT,
            priv_lvl: PRV_M,
            mhartid: hartid,
            resetvec,
            pc: resetvec,
            load_res: u64::MAX,
            ..CpuRiscvState::default()
        };
        // mstatus: MIE and MPRV clear, SXL and UXL fixed at RV64.
        s.mstatus = (2 << 34) | (2 << 32);
        // The model has Svadu without Svade, so ADUE starts set.
        s.menvcfg = MENVCFG_ADUE;
        // Debug triggers: every trigger is a disabled type 2 match control.
        for i in 0..NUM_TRIGGERS {
            s.tdata1[i] = 2 << 60;
        }
        s
    }

    /// x`r`, with x0 reading as zero.
    pub fn x(&self, r: usize) -> u64 {
        if r == 0 { 0 } else { self.gpr[r] }
    }

    /// `mstatus` as a CSR read sees it, with SD computed.
    pub fn mstatus_sd(&self) -> u64 {
        add_status_sd(self.mstatus)
    }
}

/// `add_status_sd()`: SD is set when FS, VS or XS is dirty.
pub fn add_status_sd(status: u64) -> u64 {
    if status & MSTATUS_FS == MSTATUS_FS
        || status & MSTATUS_VS == MSTATUS_VS
        || status & MSTATUS_XS == MSTATUS_XS
    {
        status | MSTATUS64_SD
    } else {
        status
    }
}

/// `get_field()`: the value of the field `mask` of `reg`.
pub const fn get_field(reg: u64, mask: u64) -> u64 {
    (reg & mask) >> mask.trailing_zeros()
}

/// `set_field()`: `reg` with the field `mask` set to `val`.
pub const fn set_field(reg: u64, mask: u64, val: u64) -> u64 {
    (reg & !mask) | ((val << mask.trailing_zeros()) & mask)
}
