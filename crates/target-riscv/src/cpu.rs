// SPDX-License-Identifier: GPL-2.0-or-later

//! `CPURISCVState` for RV64 and the CPU model: the parts of QEMU's `target/riscv/cpu.h`,
//! `cpu_bits.h` and `cpu.c` this crate needs.
//!
//! The one model is QEMU's default `rv64` CPU: RV64GC (I, M, A, F, D, C) with S and U modes,
//! the H extension (on by default, `MISA_CFG(RVH, true)` in `target/riscv/tcg/tcg-cpu.c`,
//! with GEILEN 0), Sv39, Sv48 and Sv57, 16 PMP regions, Zicsr, Zifencei, Zicntr,
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
    /// The RV64 register state, `CPURISCVState` cut down to RV64.
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
        /// `vtype` without `vill`, which is kept apart.
        pub vtype: u64,
        /// `vl`.
        pub vl: u64,
        /// `vstart`.
        pub vstart: u64,
        /// `vxrm`.
        pub vxrm: u64,
        /// `vxsat`.
        pub vxsat: u64,
        /// `vill`, 0 or 1.
        pub vill: u64,
        /// `virt_enabled`: the hart runs in VS or VU mode, 0 or 1.
        pub virt_enabled: u64,
        /// `hstatus`.
        pub hstatus: u64,
        /// `hedeleg`.
        pub hedeleg: u64,
        /// `hideleg`.
        pub hideleg: u64,
        /// `hcounteren`.
        pub hcounteren: u64,
        /// `htval`.
        pub htval: u64,
        /// `htinst`.
        pub htinst: u64,
        /// `hgatp`.
        pub hgatp: u64,
        /// `henvcfg`.
        pub henvcfg: u64,
        /// `htimedelta`.
        pub htimedelta: u64,
        /// `hgeie`.
        pub hgeie: u64,
        /// `vstimecmp`.
        pub vstimecmp: u64,
        /// `mtval2`.
        pub mtval2: u64,
        /// `mtinst`.
        pub mtinst: u64,
        /// `vsstatus`, the guest's `sstatus` while V=0.
        pub vsstatus: u64,
        /// `vstvec`.
        pub vstvec: u64,
        /// `vsscratch`.
        pub vsscratch: u64,
        /// `vsepc`.
        pub vsepc: u64,
        /// `vscause`.
        pub vscause: u64,
        /// `vstval`.
        pub vstval: u64,
        /// `vsatp`.
        pub vsatp: u64,
        /// The HS `mstatus` bits while V=1, `mstatus_hs`.
        pub mstatus_hs: u64,
        /// `stvec_hs`.
        pub stvec_hs: u64,
        /// `sscratch_hs`.
        pub sscratch_hs: u64,
        /// `sepc_hs`.
        pub sepc_hs: u64,
        /// `scause_hs`.
        pub scause_hs: u64,
        /// `stval_hs`.
        pub stval_hs: u64,
        /// `satp_hs`.
        pub satp_hs: u64,
        /// `two_stage_lookup`: the pending fault came from a two stage translation.
        pub two_stage_lookup: u64,
        /// `two_stage_indirect_lookup`: it came from the G-stage walk of a VS-stage PTE.
        pub two_stage_indirect_lookup: u64,
        /// `guest_phys_fault_addr`: the guest physical address of a guest page fault,
        /// shifted right by 2.
        pub guest_phys_fault_addr: u64,
    }
}

/// `vlenb`: the vector registers are 128 bits long (QEMU's default `vlen`).
pub const VLENB: usize = 16;

/// The offset in `env` of the vector register file, `vreg`: v0 to v31, `VLENB` bytes each,
/// with element `i` of a register at byte `i * esz`, little endian whatever the host is.
///
/// The registers live in `env` after [`CpuRiscvState`] rather than in it, so that the
/// helpers that copy the whole struct in and out (CSR accesses, the page walk) do not
/// copy 512 more bytes each time. Only the vector helpers reach them.
pub const VREG: usize = (ENV_TARGET_OFFSET + size_of::<CpuRiscvState>() + 15) & !15;

/// The size of the `env` buffer a RISC-V vCPU needs.
pub const ENV_SIZE: usize = VREG + 32 * VLENB;

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
/// `vtype`.
pub const VTYPE: usize = env_off(offset_of!(CpuRiscvState, vtype));
/// `vl`.
pub const VL: usize = env_off(offset_of!(CpuRiscvState, vl));
/// `vstart`.
pub const VSTART: usize = env_off(offset_of!(CpuRiscvState, vstart));
/// `vxrm`.
pub const VXRM: usize = env_off(offset_of!(CpuRiscvState, vxrm));
/// `vxsat`.
pub const VXSAT: usize = env_off(offset_of!(CpuRiscvState, vxsat));
/// `vill`.
pub const VILL: usize = env_off(offset_of!(CpuRiscvState, vill));
/// `virt_enabled`.
pub const VIRT_ENABLED: usize = env_off(offset_of!(CpuRiscvState, virt_enabled));
/// `vsstatus`.
pub const VSSTATUS: usize = env_off(offset_of!(CpuRiscvState, vsstatus));
/// `mstatus_hs`.
pub const MSTATUS_HS: usize = env_off(offset_of!(CpuRiscvState, mstatus_hs));
/// `misa`.
pub const MISA: usize = env_off(offset_of!(CpuRiscvState, misa));
/// `two_stage_lookup`.
pub const TWO_STAGE_LOOKUP: usize = env_off(offset_of!(CpuRiscvState, two_stage_lookup));
/// `two_stage_indirect_lookup`.
pub const TWO_STAGE_INDIRECT_LOOKUP: usize =
    env_off(offset_of!(CpuRiscvState, two_stage_indirect_lookup));
/// `guest_phys_fault_addr`.
pub const GUEST_PHYS_FAULT_ADDR: usize = env_off(offset_of!(CpuRiscvState, guest_phys_fault_addr));

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
/// `MMU_2STAGE_BIT`: the index translates a guest (VS or VU mode, or HLV and HSV) access
/// through both stages.
pub const MMU_2STAGE_BIT: usize = 1 << 2;
/// The number of MMU indexes this port uses: U, S, S with SUM and M, each with and
/// without the two stage bit.
pub const NB_MMU_MODES: usize = 8;

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
/// RVH.
pub const RVH: u64 = rvx(b'H');
/// RVI.
pub const RVI: u64 = rvx(b'I');
/// RVM.
pub const RVM: u64 = rvx(b'M');
/// RVS.
pub const RVS: u64 = rvx(b'S');
/// RVU.
pub const RVU: u64 = rvx(b'U');
/// RVV.
pub const RVV: u64 = rvx(b'V');

/// The extensions of the model in `misa` that cannot be turned off.
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
/// GVA.
pub const MSTATUS_GVA: u64 = 1 << 38;
/// MPV.
pub const MSTATUS_MPV: u64 = 1 << 39;
/// SD.
pub const MSTATUS64_SD: u64 = 1 << 63;

// hstatus.

/// VSBE.
pub const HSTATUS_VSBE: u64 = 0x20;
/// GVA.
pub const HSTATUS_GVA: u64 = 0x40;
/// SPV.
pub const HSTATUS_SPV: u64 = 0x80;
/// SPVP.
pub const HSTATUS_SPVP: u64 = 0x100;
/// HU.
pub const HSTATUS_HU: u64 = 0x200;
/// VGEIN.
pub const HSTATUS_VGEIN: u64 = 0x3F000;
/// VTVM.
pub const HSTATUS_VTVM: u64 = 0x100000;
/// VTW.
pub const HSTATUS_VTW: u64 = 0x200000;
/// VTSR.
pub const HSTATUS_VTSR: u64 = 0x400000;
/// HUKTE.
pub const HSTATUS_HUKTE: u64 = 0x1000000;
/// VSXL.
pub const HSTATUS_VSXL: u64 = 0x300000000;
/// HUPMM.
pub const HSTATUS_HUPMM: u64 = 0x3000000000000;
/// UXL of `vsstatus`.
pub const VSSTATUS64_UXL: u64 = 0x300000000;

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
/// Virtual supervisor software interrupt.
pub const IRQ_VS_SOFT: u32 = 2;
/// Virtual supervisor timer interrupt.
pub const IRQ_VS_TIMER: u32 = 6;
/// Virtual supervisor external interrupt.
pub const IRQ_VS_EXT: u32 = 10;
/// Supervisor guest external interrupt.
pub const IRQ_S_GEXT: u32 = 12;

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
/// VSSIP.
pub const MIP_VSSIP: u64 = 1 << IRQ_VS_SOFT;
/// VSTIP.
pub const MIP_VSTIP: u64 = 1 << IRQ_VS_TIMER;
/// VSEIP.
pub const MIP_VSEIP: u64 = 1 << IRQ_VS_EXT;
/// The VS interrupts, `VS_MODE_INTERRUPTS`.
pub const VS_MODE_INTERRUPTS: u64 = MIP_VSSIP | MIP_VSTIP | MIP_VSEIP;
/// LCOFIP.
pub const MIP_LCOFIP: u64 = 1 << 13;
/// SGEIP.
pub const MIP_SGEIP: u64 = 1 << 12;

/// The supervisor interrupts, `S_MODE_INTERRUPTS`.
pub const S_MODE_INTERRUPTS: u64 = MIP_SSIP | MIP_STIP | MIP_SEIP;
/// The machine interrupts, `M_MODE_INTERRUPTS`.
pub const M_MODE_INTERRUPTS: u64 = MIP_MSIP | MIP_MTIP | MIP_MEIP;
/// The hypervisor interrupts, `HS_MODE_INTERRUPTS`.
pub const HS_MODE_INTERRUPTS: u64 = MIP_SGEIP | VS_MODE_INTERRUPTS;

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
/// Environment call from VS mode.
pub const EXCP_VS_ECALL: i32 = 10;
/// Environment call from M mode.
pub const EXCP_M_ECALL: i32 = 11;
/// Instruction page fault.
pub const EXCP_INST_PAGE_FAULT: i32 = 12;
/// Load page fault.
pub const EXCP_LOAD_PAGE_FAULT: i32 = 13;
/// Store/AMO page fault.
pub const EXCP_STORE_PAGE_FAULT: i32 = 15;
/// Instruction guest page fault.
pub const EXCP_INST_GUEST_PAGE_FAULT: i32 = 20;
/// Load guest page fault.
pub const EXCP_LOAD_GUEST_ACCESS_FAULT: i32 = 21;
/// Virtual instruction.
pub const EXCP_VIRT_INSTRUCTION_FAULT: i32 = 22;
/// Store/AMO guest page fault.
pub const EXCP_STORE_GUEST_AMO_ACCESS_FAULT: i32 = 23;
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
/// The MODE field of `hgatp`.
pub const HGATP64_MODE: u64 = 0xF << 60;
/// The VMID field.
pub const HGATP64_VMID: u64 = 0x3FFF << 44;
/// The PPN field.
pub const HGATP64_PPN: u64 = 0x0FFF_FFFF_FFFF;

// menvcfg and senvcfg.

/// FIOM.
pub const MENVCFG_FIOM: u64 = 1;
/// CBIE.
pub const MENVCFG_CBIE: u64 = 3 << 4;
/// CBCFE.
pub const MENVCFG_CBCFE: u64 = 1 << 6;
/// CBZE.
pub const MENVCFG_CBZE: u64 = 1 << 7;
/// PBMTE.
pub const MENVCFG_PBMTE: u64 = 1 << 62;
/// DTE.
pub const MENVCFG_DTE: u64 = 1 << 59;
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

// vtype.

/// `R_VTYPE_VLMUL_MASK`.
pub const VTYPE_VLMUL: u64 = 0x7;
/// `R_VTYPE_VSEW_MASK`.
pub const VTYPE_VSEW: u64 = 0x38;
/// `R_VTYPE_VTA_MASK`.
pub const VTYPE_VTA: u64 = 0x40;
/// `R_VTYPE_VMA_MASK`.
pub const VTYPE_VMA: u64 = 0x80;
/// `R_VTYPE_ALTFMT_MASK`.
pub const VTYPE_ALTFMT: u64 = 0x100;

/// The CPU configuration: QEMU's `RISCVCPUConfig` cut down to the switches that are not
/// fixed in this port. The default is QEMU's `rv64`, which has H on (`MISA_CFG(RVH, true)`
/// in `target/riscv/tcg/tcg-cpu.c`) and V and every vector extension off
/// (`MISA_CFG(RVV, false)`, and none of the `Zv*` are in the `rv64` defaults of
/// `riscv_cpu_extensions[]` in `cpu.c`).
///
/// [`RiscvCfg::max`] is what `-cpu max` enables of the vector extensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct RiscvCfg {
    /// `misa.H`.
    pub ext_h: bool,
    /// `misa.V`.
    pub ext_v: bool,
    /// Zve32x.
    pub ext_zve32x: bool,
    /// Zve32f.
    pub ext_zve32f: bool,
    /// Zve64x.
    pub ext_zve64x: bool,
    /// Zve64f.
    pub ext_zve64f: bool,
    /// Zve64d.
    pub ext_zve64d: bool,
    /// Zvfh.
    pub ext_zvfh: bool,
    /// Zvfhmin.
    pub ext_zvfhmin: bool,
    /// Zfhmin, implied by Zvfh.
    pub ext_zfhmin: bool,
    /// Zfbfmin, implied by Zvfbfwma.
    pub ext_zfbfmin: bool,
    /// Zvfbfmin.
    pub ext_zvfbfmin: bool,
    /// Zvfbfwma.
    pub ext_zvfbfwma: bool,
    /// Zvbb.
    pub ext_zvbb: bool,
    /// Zvbc.
    pub ext_zvbc: bool,
    /// Zvkb.
    pub ext_zvkb: bool,
    /// Zvkg.
    pub ext_zvkg: bool,
    /// Zvkned.
    pub ext_zvkned: bool,
    /// Zvknha.
    pub ext_zvknha: bool,
    /// Zvknhb.
    pub ext_zvknhb: bool,
    /// Zvksed.
    pub ext_zvksed: bool,
    /// Zvksh.
    pub ext_zvksh: bool,
    /// Zvkt.
    pub ext_zvkt: bool,
    /// Zvkn, a name for Zvkned, Zvknhb, Zvkb and Zvkt.
    pub ext_zvkn: bool,
    /// Zvknc, a name for Zvkn and Zvbc.
    pub ext_zvknc: bool,
    /// Zvkng, a name for Zvkn and Zvkg.
    pub ext_zvkng: bool,
    /// Zvks, a name for Zvksed, Zvksh, Zvkb and Zvkt.
    pub ext_zvks: bool,
    /// Zvksc, a name for Zvks and Zvbc.
    pub ext_zvksc: bool,
    /// Zvksg, a name for Zvks and Zvkg.
    pub ext_zvksg: bool,
    /// `vlenb`. Only 16 is supported.
    pub vlenb: u32,
    /// `elen`.
    pub elen: u32,
    /// `rvv_ta_all_1s`: tail agnostic elements are set to all ones.
    pub rvv_ta_all_1s: bool,
    /// `rvv_ma_all_1s`: mask agnostic elements are set to all ones.
    pub rvv_ma_all_1s: bool,
    /// `rvv_vl_half_avl`.
    pub rvv_vl_half_avl: bool,
    /// `rvv_vsetvl_x0_vill`.
    pub rvv_vsetvl_x0_vill: bool,
}

impl Default for RiscvCfg {
    fn default() -> RiscvCfg {
        RiscvCfg {
            ext_h: true,
            ext_v: false,
            ext_zve32x: false,
            ext_zve32f: false,
            ext_zve64x: false,
            ext_zve64f: false,
            ext_zve64d: false,
            ext_zvfh: false,
            ext_zvfhmin: false,
            ext_zfhmin: false,
            ext_zfbfmin: false,
            ext_zvfbfmin: false,
            ext_zvfbfwma: false,
            ext_zvbb: false,
            ext_zvbc: false,
            ext_zvkb: false,
            ext_zvkg: false,
            ext_zvkned: false,
            ext_zvknha: false,
            ext_zvknhb: false,
            ext_zvksed: false,
            ext_zvksh: false,
            ext_zvkt: false,
            ext_zvkn: false,
            ext_zvknc: false,
            ext_zvkng: false,
            ext_zvks: false,
            ext_zvksc: false,
            ext_zvksg: false,
            vlenb: VLENB as u32,
            elen: 64,
            rvv_ta_all_1s: false,
            rvv_ma_all_1s: false,
            rvv_vl_half_avl: false,
            rvv_vsetvl_x0_vill: false,
        }
    }
}

impl RiscvCfg {
    /// The vector extensions of `-cpu max` (`riscv_init_max_cpu_extensions()`): V and every
    /// `Zv*` extension this port has, with what they imply.
    pub fn max() -> RiscvCfg {
        let mut c = RiscvCfg {
            ext_v: true,
            ext_zvfh: true,
            ext_zvfbfmin: true,
            ext_zvfbfwma: true,
            ext_zvbb: true,
            ext_zvbc: true,
            ext_zvkb: true,
            ext_zvkg: true,
            ext_zvkned: true,
            ext_zvknha: true,
            ext_zvknhb: true,
            ext_zvksed: true,
            ext_zvksh: true,
            ext_zvkt: true,
            ext_zvkn: true,
            ext_zvknc: true,
            ext_zvkng: true,
            ext_zvks: true,
            ext_zvksc: true,
            ext_zvksg: true,
            ..RiscvCfg::default()
        };
        c.apply_implied();
        c
    }

    /// Sets the boolean `-cpu` property `name` (`v`, `zve32x`, `zvfh`, ...), as QEMU's
    /// `riscv_cpu_extensions[]`, `riscv_cpu_named_features[]` and `riscv_cpu_options[]`
    /// spell them. Gives false when `name` is not a property of this configuration.
    pub fn set_prop(&mut self, name: &str, value: bool) -> bool {
        let field = match name {
            "h" => &mut self.ext_h,
            "v" => &mut self.ext_v,
            "zve32x" => &mut self.ext_zve32x,
            "zve32f" => &mut self.ext_zve32f,
            "zve64x" => &mut self.ext_zve64x,
            "zve64f" => &mut self.ext_zve64f,
            "zve64d" => &mut self.ext_zve64d,
            "zvfh" => &mut self.ext_zvfh,
            "zvfhmin" => &mut self.ext_zvfhmin,
            "zfhmin" => &mut self.ext_zfhmin,
            "zfbfmin" => &mut self.ext_zfbfmin,
            "zvfbfmin" => &mut self.ext_zvfbfmin,
            "zvfbfwma" => &mut self.ext_zvfbfwma,
            "zvbb" => &mut self.ext_zvbb,
            "zvbc" => &mut self.ext_zvbc,
            "zvkb" => &mut self.ext_zvkb,
            "zvkg" => &mut self.ext_zvkg,
            "zvkned" => &mut self.ext_zvkned,
            "zvknha" => &mut self.ext_zvknha,
            "zvknhb" => &mut self.ext_zvknhb,
            "zvksed" => &mut self.ext_zvksed,
            "zvksh" => &mut self.ext_zvksh,
            "zvkt" => &mut self.ext_zvkt,
            "zvkn" => &mut self.ext_zvkn,
            "zvknc" => &mut self.ext_zvknc,
            "zvkng" => &mut self.ext_zvkng,
            "zvks" => &mut self.ext_zvks,
            "zvksc" => &mut self.ext_zvksc,
            "zvksg" => &mut self.ext_zvksg,
            "rvv_ta_all_1s" => &mut self.rvv_ta_all_1s,
            "rvv_ma_all_1s" => &mut self.rvv_ma_all_1s,
            "rvv_vl_half_avl" => &mut self.rvv_vl_half_avl,
            "rvv_vsetvl_x0_vill" => &mut self.rvv_vsetvl_x0_vill,
            _ => return false,
        };
        *field = value;
        true
    }

    /// `riscv_cpu_enable_implied_rules()` for the vector extensions, applied until nothing
    /// changes: V implies Zve64d; Zve64d implies Zve64f; Zve64f implies Zve32f and Zve64x;
    /// Zve32f and Zve64x imply Zve32x; Zvfh implies Zvfhmin and Zfhmin; Zvfhmin and Zvfbfmin
    /// imply Zve32f; Zvfbfwma implies Zvfbfmin and Zfbfmin; Zvbb implies Zvkb; Zvknhb
    /// implies Zve64x and Zvknha; Zvkn, Zvks and the names made of them imply their parts.
    pub fn apply_implied(&mut self) {
        loop {
            let before = *self;
            if self.ext_v {
                self.ext_zve64d = true;
            }
            if self.ext_zve64d {
                self.ext_zve64f = true;
            }
            if self.ext_zve64f {
                self.ext_zve32f = true;
                self.ext_zve64x = true;
            }
            if self.ext_zve32f || self.ext_zve64x {
                self.ext_zve32x = true;
            }
            if self.ext_zvfh {
                self.ext_zvfhmin = true;
                self.ext_zfhmin = true;
            }
            if self.ext_zvfhmin || self.ext_zvfbfmin {
                self.ext_zve32f = true;
            }
            if self.ext_zvfbfwma {
                self.ext_zvfbfmin = true;
                self.ext_zfbfmin = true;
            }
            if self.ext_zvbb {
                self.ext_zvkb = true;
            }
            if self.ext_zvknhb {
                self.ext_zve64x = true;
                self.ext_zvknha = true;
            }
            if self.ext_zvknc || self.ext_zvkng {
                self.ext_zvkn = true;
            }
            if self.ext_zvksc || self.ext_zvksg {
                self.ext_zvks = true;
            }
            if self.ext_zvknc || self.ext_zvksc {
                self.ext_zvbc = true;
            }
            if self.ext_zvkng || self.ext_zvksg {
                self.ext_zvkg = true;
            }
            if self.ext_zvkn {
                self.ext_zvkned = true;
                self.ext_zvknhb = true;
                self.ext_zvkb = true;
                self.ext_zvkt = true;
            }
            if self.ext_zvks {
                self.ext_zvksed = true;
                self.ext_zvksh = true;
                self.ext_zvkb = true;
                self.ext_zvkt = true;
            }
            if *self == before {
                break;
            }
        }
    }

    /// The checks of `riscv_cpu_validate_set_extensions()` and `riscv_cpu_validate_v()`
    /// for the vector extensions, with QEMU's messages. The model always has F and D.
    pub fn validate(&self) -> Result<(), String> {
        // riscv_cpu_validate_v().
        let min_vlen = if self.ext_v {
            128
        } else if self.ext_zve64x {
            64
        } else if self.ext_zve32x {
            32
        } else {
            0
        };
        if min_vlen != 0 {
            let vlen = self.vlenb << 3;
            // Only VLEN=128 is supported by this port: VLENB is a constant.
            if vlen != VLENB as u32 * 8 || vlen < min_vlen {
                return Err(format!(
                    "Vector extension implementation only supports VLEN in the range [{min_vlen}, {}]",
                    VLENB * 8
                ));
            }
            if self.elen > 64 || self.elen < 8 {
                return Err(
                    "Vector extension implementation only supports ELEN in the range [8, 64]"
                        .into(),
                );
            }
        }
        if (self.ext_zvfh || self.ext_zvfhmin) && !self.ext_zve32f {
            return Err("Zvfh/Zvfhmin extensions require Zve32f extension".into());
        }
        if self.ext_zvfh && !self.ext_zfhmin {
            return Err("Zvfh extensions requires Zfhmin extension".into());
        }
        if self.ext_zvfbfmin && !self.ext_zve32f {
            return Err("Zvfbfmin extension depends on Zve32f extension".into());
        }
        if self.ext_zvfbfwma && !self.ext_zvfbfmin {
            return Err("Zvfbfwma extension depends on Zvfbfmin extension".into());
        }
        if (self.ext_zvbb
            || self.ext_zvkb
            || self.ext_zvkg
            || self.ext_zvkned
            || self.ext_zvknha
            || self.ext_zvksed
            || self.ext_zvksh)
            && !self.ext_zve32x
        {
            return Err("Vector crypto extensions require V or Zve* extensions".into());
        }
        if (self.ext_zvbc || self.ext_zvknhb) && !self.ext_zve64x {
            return Err("Zvbc and Zvknhb extensions require V or Zve64x extensions".into());
        }
        Ok(())
    }

    /// The `misa` extension bits: [`MISA_EXT`], with H and V when they are on.
    pub fn misa_ext(&self) -> u64 {
        let mut ext = MISA_EXT;
        if self.ext_h {
            ext |= RVH;
        }
        if self.ext_v {
            ext |= RVV;
        }
        ext
    }
}

impl CpuRiscvState {
    /// The state after `riscv_cpu_reset_hold()` for hart `hartid` with reset vector
    /// `resetvec`, for the default CPU (with H, without V).
    pub fn reset(hartid: u64, resetvec: u64) -> CpuRiscvState {
        CpuRiscvState::reset_cfg(hartid, resetvec, &RiscvCfg::default())
    }

    /// [`CpuRiscvState::reset`] for a CPU with the extensions of `cfg`: `misa` has H and V
    /// when `cfg` has them.
    pub fn reset_cfg(hartid: u64, resetvec: u64, cfg: &RiscvCfg) -> CpuRiscvState {
        let mut s = CpuRiscvState {
            misa: MISA_MXL_RV64 | cfg.misa_ext(),
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
        s.vill = 1;
        if cfg.ext_h {
            s.vsstatus = (2 << 34) | (2 << 32);
            s.mstatus_hs = (2 << 34) | (2 << 32);
            // Bits 10, 6, 2 and 12 of mideleg are read only 1 with the H extension.
            s.mideleg |= HS_MODE_INTERRUPTS;
        }
        s
    }

    /// Whether the hart has the H extension, `riscv_has_ext(env, RVH)`.
    pub fn has_h(&self) -> bool {
        self.misa & RVH != 0
    }

    /// Whether the hart runs in VS or VU mode.
    pub fn virt(&self) -> bool {
        self.virt_enabled != 0
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
