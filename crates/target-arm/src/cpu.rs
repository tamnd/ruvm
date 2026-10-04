// SPDX-License-Identifier: GPL-2.0-or-later

//! `CPUARMState` for AArch64 and the CPU models: the parts of QEMU's `target/arm/cpu.h`,
//! `cpu.c`, `cpu64.c` and `cpu-max.c` this crate needs, including the EL2 and EL3 state, the
//! HCR_EL2 and SCR_EL3 write masks and `arm_hcr_el2_eff()`.
//!
//! The models are `cortex-a57`, `cortex-a72` and `cortex-a76` with their QEMU ID register
//! values (the A57 reports the 4K granule only, which this crate walks; the A72 adds 64K
//! and the A76, which has VHE, has all three). They start without EL2 and EL3, the virt
//! board's default, and [`ArmCpuModel::with_el2`] and [`ArmCpuModel::with_el3`] add them as
//! `virtualization=on` and `secure=on` do.
//!
//! [`CpuArmState`] is a plain struct with `#[repr(C)]`, so `offset_of!` gives the offset of
//! every field. The runtime keeps the state of a vCPU in a byte buffer (`env`), with the
//! struct's fields at [`ENV_TARGET_OFFSET`] plus their offset, little endian. Generated code
//! reaches registers through the offset constants below, and helpers copy the whole struct in
//! and out with [`CpuArmState::load`] and [`CpuArmState::store`]. No unsafe code is involved.

use std::mem::{offset_of, size_of};

use ruvm_jit::ENV_TARGET_OFFSET;

/// A field of [`CpuArmState`] that can be copied to and from `env`.
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

impl Field for u32 {
    fn put(&self, b: &mut [u8]) {
        b[..4].copy_from_slice(&self.to_le_bytes());
    }
    fn get(&mut self, b: &[u8]) {
        *self = u32::from_le_bytes(b[..4].try_into().expect("4 bytes"));
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

macro_rules! arm_state {
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

            /// Read the state without the SVE registers `zregs` and `pregs`, which are left
            /// zero, for the hot paths that only look at the system state (the TB flags,
            /// the MMU index and the page table walk). The result must not be stored back.
            pub fn load_system(env: &[u8]) -> $name {
                let mut s = $name::default();
                $(if !matches!(stringify!($f), "zregs" | "pregs") {
                    s.$f.get(&env[ENV_TARGET_OFFSET + offset_of!($name, $f)..]);
                })*
                s
            }

            /// Write the state into a vCPU's `env` buffer.
            pub fn store(&self, env: &mut [u8]) {
                $(self.$f.put(&mut env[ENV_TARGET_OFFSET + offset_of!($name, $f)..]);)*
            }
        }
    };
}

arm_state! {
    /// The AArch64 register state, `CPUARMState` cut down to AArch64 at EL0 to EL3.
    ///
    /// As in QEMU, `xregs[31]` is the current stack pointer, the banked stack pointers are in
    /// `sp_el`, and the condition flags are kept apart from `pstate`: `nf` and `vf` hold the
    /// flag in bit 31, `zf` is zero exactly when Z is set, and `cf` is 0 or 1. `daif` holds
    /// the D, A, I and F bits at their PSTATE positions. Per exception level arrays are
    /// indexed by EL.
    pub struct CpuArmState {
        /// X0 to X30, and the current SP in slot 31.
        pub xregs: [u64; 32],
        /// The PC.
        pub pc: u64,
        /// PSTATE without NZCV and DAIF.
        pub pstate: u32,
        /// PSTATE.{D,A,I,F}.
        pub daif: u32,
        /// N in bit 31.
        pub nf: u32,
        /// Zero when Z is set.
        pub zf: u32,
        /// C, 0 or 1.
        pub cf: u32,
        /// V in bit 31.
        pub vf: u32,
        /// SP_ELx while it is not the current SP.
        pub sp_el: [u64; 4],
        /// ELR_ELx.
        pub elr_el: [u64; 4],
        /// SPSR_ELx.
        pub spsr_el: [u64; 4],
        /// SCTLR_ELx.
        pub sctlr_el: [u64; 4],
        /// TCR_ELx.
        pub tcr_el: [u64; 4],
        /// TCR2_EL1 (index 1) and TCR2_EL2 (index 2).
        pub tcr2_el: [u64; 4],
        /// TTBR0_ELx.
        pub ttbr0_el: [u64; 4],
        /// TTBR1_ELx.
        pub ttbr1_el: [u64; 4],
        /// MAIR_ELx.
        pub mair_el: [u64; 4],
        /// VBAR_ELx.
        pub vbar_el: [u64; 4],
        /// ESR_ELx.
        pub esr_el: [u64; 4],
        /// FAR_ELx.
        pub far_el: [u64; 4],
        /// TPIDR_ELx.
        pub tpidr_el: [u64; 4],
        /// TPIDRRO_EL0.
        pub tpidrro_el0: u64,
        /// CONTEXTIDR_EL1.
        pub contextidr_el1: u64,
        /// CPACR_EL1.
        pub cpacr_el1: u64,
        /// PAR_EL1.
        pub par_el1: u64,
        /// CSSELR_EL1.
        pub csselr_el1: u64,
        /// AFSR0_EL1.
        pub afsr0_el1: u64,
        /// AFSR1_EL1.
        pub afsr1_el1: u64,
        /// AMAIR_EL1.
        pub amair_el1: u64,
        /// MDSCR_EL1.
        pub mdscr_el1: u64,
        /// OSDLR_EL1.
        pub osdlr_el1: u64,
        /// OSLSR_EL1.
        pub oslsr_el1: u64,
        /// CNTKCTL_EL1.
        pub cntkctl_el1: u64,
        /// CNTFRQ_EL0.
        pub cntfrq_el0: u64,
        /// The generic timers' CNT*_CTL registers, indexed by `GTIMER_*`. ISTATUS (bit 2) is
        /// kept up to date by `gt_recalc_timer()`.
        pub gt_ctl: [u64; 5],
        /// The generic timers' CNT*_CVAL registers, indexed by `GTIMER_*`.
        pub gt_cval: [u64; 5],
        /// SCR_EL3.
        pub scr_el3: u64,
        /// HCR_EL2.
        pub hcr_el2: u64,
        /// CPTR_EL2 (index 2) and CPTR_EL3 (index 3).
        pub cptr_el: [u64; 4],
        /// MDCR_EL2.
        pub mdcr_el2: u64,
        /// MDCR_EL3.
        pub mdcr_el3: u64,
        /// HSTR_EL2.
        pub hstr_el2: u64,
        /// HACR_EL2 is constant zero; this is VPIDR_EL2.
        pub vpidr_el2: u64,
        /// VMPIDR_EL2.
        pub vmpidr_el2: u64,
        /// VTCR_EL2.
        pub vtcr_el2: u64,
        /// VTTBR_EL2.
        pub vttbr_el2: u64,
        /// HPFAR_EL2.
        pub hpfar_el2: u64,
        /// CONTEXTIDR_EL2.
        pub contextidr_el2: u64,
        /// CNTHCTL_EL2.
        pub cnthctl_el2: u64,
        /// CNTVOFF_EL2.
        pub cntvoff_el2: u64,
        /// VSESR_EL2.
        pub vsesr_el2: u64,
        /// The address the exclusive monitor watches, or all ones when it is open.
        pub exclusive_addr: u64,
        /// The value loaded by the last load exclusive.
        pub exclusive_val: u64,
        /// The high half loaded by the last 128-bit load exclusive pair.
        pub exclusive_high: u64,
        /// `exception.syndrome`.
        pub exception_syndrome: u32,
        /// `exception.target_el`.
        pub exception_target_el: u32,
        /// `exception.vaddress`.
        pub exception_vaddress: u64,
        /// The SVE registers Z0 to Z31, QEMU's `vfp.zregs`, each [`ARM_MAX_VQ`] quadwords
        /// long. The AdvSIMD and FP register Vn is the low 128 bits of Zn: `zregs[n][0]` is
        /// its low half and `zregs[n][1]` its high half. Bytes beyond the current vector
        /// length are kept zero, as `aarch64_sve_narrow_vq()` leaves them.
        pub zregs: [[u64; 32]; 32],
        /// The SVE predicate registers P0 to P15 and FFR (index 16), QEMU's `vfp.pregs`: one
        /// bit per byte of a Z register, element `i` of a predicate on `1 << esz` byte
        /// elements being bit `i << esz`.
        pub pregs: [[u64; 4]; 17],
        /// ZCR_ELx, indexed by EL (index 0 unused): only the LEN field, bits 3 to 0.
        pub zcr_el: [u64; 4],
        /// FPCR: AHP, DN, FZ, RMode and FZ16 as QEMU keeps them in `vfp.fpcr`, plus the Len
        /// and Stride bits QEMU keeps apart in `vfp.vec_len` and `vfp.vec_stride`.
        pub fpcr: u32,
        /// FPSR: NZCV and the cumulative exception bits that have been folded in. The flags
        /// still in `fp_status` and the QC bit in `qc` are or-ed in when FPSR is read.
        pub fpsr: u32,
        /// `vfp.qc`: FPSR.QC is set while any bit of this is set.
        pub qc: [u64; 2],
        /// The packed `float_status` of `FPST_A64` (index 0) and `FPST_A64_F16` (index 1),
        /// see `tcg::vfp`.
        pub fp_status: [u64; 2],
    }
}

/// The size of a vCPU's `env` buffer.
pub const ENV_SIZE: usize = ENV_TARGET_OFFSET + size_of::<CpuArmState>();

/// The `env` offset of a byte offset into [`CpuArmState`].
pub const fn env_off(off: usize) -> usize {
    ENV_TARGET_OFFSET + off
}

/// The `env` offset of `xregs[n]`.
pub const fn xreg_off(n: usize) -> usize {
    env_off(offset_of!(CpuArmState, xregs)) + 8 * n
}
/// The `env` offset of the PC.
pub const PC: usize = env_off(offset_of!(CpuArmState, pc));
/// The `env` offset of `pstate`.
pub const PSTATE: usize = env_off(offset_of!(CpuArmState, pstate));
/// The `env` offset of `daif`.
pub const DAIF: usize = env_off(offset_of!(CpuArmState, daif));
/// The `env` offset of `nf`.
pub const NF: usize = env_off(offset_of!(CpuArmState, nf));
/// The `env` offset of `zf`.
pub const ZF: usize = env_off(offset_of!(CpuArmState, zf));
/// The `env` offset of `cf`.
pub const CF: usize = env_off(offset_of!(CpuArmState, cf));
/// The `env` offset of `vf`.
pub const VF: usize = env_off(offset_of!(CpuArmState, vf));
/// The `env` offset of `exclusive_addr`.
pub const EXCLUSIVE_ADDR: usize = env_off(offset_of!(CpuArmState, exclusive_addr));
/// The `env` offset of `exclusive_val`.
pub const EXCLUSIVE_VAL: usize = env_off(offset_of!(CpuArmState, exclusive_val));
/// The `env` offset of `exclusive_high`.
pub const EXCLUSIVE_HIGH: usize = env_off(offset_of!(CpuArmState, exclusive_high));

/// `ARM_MAX_VQ`: the largest SVE vector length this port supports, in quadwords.
pub const ARM_MAX_VQ: usize = 16;

/// The size in bytes of one Z register in `env`.
pub const ZREG_SIZE: usize = 16 * ARM_MAX_VQ;

/// The size in bytes of one predicate register in `env`.
pub const PREG_SIZE: usize = ZREG_SIZE / 8;

/// The index of FFR among the predicate registers.
pub const FFR: usize = 16;

/// The `env` offset of the low 64 bits of Vn, which is also the start of Zn.
pub const fn vreg_off(n: usize) -> usize {
    env_off(offset_of!(CpuArmState, zregs)) + ZREG_SIZE * n
}

/// The `env` offset of predicate register `n` (16 is FFR).
pub const fn preg_off(n: usize) -> usize {
    env_off(offset_of!(CpuArmState, pregs)) + PREG_SIZE * n
}
/// The `env` offset of `fpcr`.
pub const FPCR: usize = env_off(offset_of!(CpuArmState, fpcr));
/// The `env` offset of `fpsr`.
pub const FPSR: usize = env_off(offset_of!(CpuArmState, fpsr));
/// The `env` offset of `qc`.
pub const QC: usize = env_off(offset_of!(CpuArmState, qc));
/// The `env` offset of the packed `FPST_A64` status.
pub const FPST_A64: usize = env_off(offset_of!(CpuArmState, fp_status));
/// The `env` offset of the packed `FPST_A64_F16` status.
pub const FPST_A64_F16: usize = env_off(offset_of!(CpuArmState, fp_status)) + 8;

/// `PSTATE_SP`: SPSel.
pub const PSTATE_SP: u32 = 1;
/// `PSTATE_M`: the mode field.
pub const PSTATE_M: u32 = 0xf;
/// `PSTATE_nRW`: AArch32 when set.
pub const PSTATE_NRW: u32 = 0x10;
/// `PSTATE_F`.
pub const PSTATE_F: u32 = 1 << 6;
/// `PSTATE_I`.
pub const PSTATE_I: u32 = 1 << 7;
/// `PSTATE_A`.
pub const PSTATE_A: u32 = 1 << 8;
/// `PSTATE_D`.
pub const PSTATE_D: u32 = 1 << 9;
/// `PSTATE_DAIF`.
pub const PSTATE_DAIF: u32 = PSTATE_D | PSTATE_A | PSTATE_I | PSTATE_F;
/// `PSTATE_IL`.
pub const PSTATE_IL: u32 = 1 << 20;
/// `PSTATE_SS`.
pub const PSTATE_SS: u32 = 1 << 21;
/// `PSTATE_PAN`.
pub const PSTATE_PAN: u32 = 1 << 22;
/// `PSTATE_UAO`.
pub const PSTATE_UAO: u32 = 1 << 23;
/// `PSTATE_V`.
pub const PSTATE_V: u32 = 1 << 28;
/// `PSTATE_C`.
pub const PSTATE_C: u32 = 1 << 29;
/// `PSTATE_Z`.
pub const PSTATE_Z: u32 = 1 << 30;
/// `PSTATE_N`.
pub const PSTATE_N: u32 = 1 << 31;
/// `PSTATE_NZCV`.
pub const PSTATE_NZCV: u32 = PSTATE_N | PSTATE_Z | PSTATE_C | PSTATE_V;
/// `PSTATE_MODE_EL0t`.
pub const PSTATE_MODE_EL0T: u32 = 0;
/// `PSTATE_MODE_EL1t`.
pub const PSTATE_MODE_EL1T: u32 = 4;
/// `PSTATE_MODE_EL1h`.
pub const PSTATE_MODE_EL1H: u32 = 5;

/// `SCTLR_M`: stage 1 MMU enable.
pub const SCTLR_M: u64 = 1 << 0;
/// `SCTLR_A`: alignment checking.
pub const SCTLR_A: u64 = 1 << 1;
/// `SCTLR_UMA`: EL0 access to DAIF.
pub const SCTLR_UMA: u64 = 1 << 9;
/// `SCTLR_DZE`: EL0 access to DC ZVA.
pub const SCTLR_DZE: u64 = 1 << 14;
/// `SCTLR_UCT`: EL0 access to CTR_EL0.
pub const SCTLR_UCT: u64 = 1 << 15;
/// `SCTLR_nTWI`: WFI at EL0 does not trap.
pub const SCTLR_NTWI: u64 = 1 << 16;
/// `SCTLR_WXN`.
pub const SCTLR_WXN: u64 = 1 << 19;
/// `SCTLR_SPAN`.
pub const SCTLR_SPAN: u64 = 1 << 23;
/// `SCTLR_UCI`: EL0 access to cache maintenance by VA.
pub const SCTLR_UCI: u64 = 1 << 26;

/// `EXCP_UDEF`.
pub const EXCP_UDEF: i32 = 1;
/// `EXCP_SWI`.
pub const EXCP_SWI: i32 = 2;
/// `EXCP_PREFETCH_ABORT`.
pub const EXCP_PREFETCH_ABORT: i32 = 3;
/// `EXCP_DATA_ABORT`.
pub const EXCP_DATA_ABORT: i32 = 4;
/// `EXCP_IRQ`.
pub const EXCP_IRQ: i32 = 5;
/// `EXCP_FIQ`.
pub const EXCP_FIQ: i32 = 6;
/// `EXCP_BKPT`.
pub const EXCP_BKPT: i32 = 7;
/// `EXCP_HVC`.
pub const EXCP_HVC: i32 = 11;
/// `EXCP_SMC`.
pub const EXCP_SMC: i32 = 13;
/// `EXCP_HYP_TRAP`.
pub const EXCP_HYP_TRAP: i32 = 12;
/// `EXCP_VIRQ`.
pub const EXCP_VIRQ: i32 = 14;
/// `EXCP_VFIQ`.
pub const EXCP_VFIQ: i32 = 15;
/// `EXCP_SEMIHOST`: a semihosting call, handled without taking an exception.
pub const EXCP_SEMIHOST: i32 = 16;
/// `EXCP_VSERR`.
pub const EXCP_VSERR: i32 = 24;

/// `PSTATE_MODE_EL2t`.
pub const PSTATE_MODE_EL2T: u32 = 8;
/// `PSTATE_MODE_EL2h`.
pub const PSTATE_MODE_EL2H: u32 = 9;
/// `PSTATE_MODE_EL3t`.
pub const PSTATE_MODE_EL3T: u32 = 12;
/// `PSTATE_MODE_EL3h`.
pub const PSTATE_MODE_EL3H: u32 = 13;

/// `SCR_NS`.
pub const SCR_NS: u64 = 1 << 0;
/// `SCR_IRQ`.
pub const SCR_IRQ: u64 = 1 << 1;
/// `SCR_FIQ`.
pub const SCR_FIQ: u64 = 1 << 2;
/// `SCR_EA`.
pub const SCR_EA: u64 = 1 << 3;
/// `SCR_FW`.
pub const SCR_FW: u64 = 1 << 4;
/// `SCR_AW`.
pub const SCR_AW: u64 = 1 << 5;
/// `SCR_NET`.
pub const SCR_NET: u64 = 1 << 6;
/// `SCR_SMD`.
pub const SCR_SMD: u64 = 1 << 7;
/// `SCR_HCE`.
pub const SCR_HCE: u64 = 1 << 8;
/// `SCR_SIF`.
pub const SCR_SIF: u64 = 1 << 9;
/// `SCR_RW`.
pub const SCR_RW: u64 = 1 << 10;
/// `SCR_ST`.
pub const SCR_ST: u64 = 1 << 11;
/// `SCR_TWI`.
pub const SCR_TWI: u64 = 1 << 12;
/// `SCR_TWE`.
pub const SCR_TWE: u64 = 1 << 13;
/// `SCR_TLOR`.
pub const SCR_TLOR: u64 = 1 << 14;
/// `SCR_TCR2EN`.
pub const SCR_TCR2EN: u64 = 1 << 43;

/// `HCR_VM`.
pub const HCR_VM: u64 = 1 << 0;
/// `HCR_SWIO`.
pub const HCR_SWIO: u64 = 1 << 1;
/// `HCR_PTW`.
pub const HCR_PTW: u64 = 1 << 2;
/// `HCR_FMO`.
pub const HCR_FMO: u64 = 1 << 3;
/// `HCR_IMO`.
pub const HCR_IMO: u64 = 1 << 4;
/// `HCR_AMO`.
pub const HCR_AMO: u64 = 1 << 5;
/// `HCR_VF`.
pub const HCR_VF: u64 = 1 << 6;
/// `HCR_VI`.
pub const HCR_VI: u64 = 1 << 7;
/// `HCR_VSE`.
pub const HCR_VSE: u64 = 1 << 8;
/// `HCR_FB`.
pub const HCR_FB: u64 = 1 << 9;
/// `HCR_BSU_MASK`.
pub const HCR_BSU_MASK: u64 = 3 << 10;
/// `HCR_DC`.
pub const HCR_DC: u64 = 1 << 12;
/// `HCR_TWI`.
pub const HCR_TWI: u64 = 1 << 13;
/// `HCR_TWE`.
pub const HCR_TWE: u64 = 1 << 14;
/// `HCR_TID0`.
pub const HCR_TID0: u64 = 1 << 15;
/// `HCR_TID1`.
pub const HCR_TID1: u64 = 1 << 16;
/// `HCR_TID2`.
pub const HCR_TID2: u64 = 1 << 17;
/// `HCR_TID3`.
pub const HCR_TID3: u64 = 1 << 18;
/// `HCR_TSC`.
pub const HCR_TSC: u64 = 1 << 19;
/// `HCR_TIDCP`.
pub const HCR_TIDCP: u64 = 1 << 20;
/// `HCR_TACR`.
pub const HCR_TACR: u64 = 1 << 21;
/// `HCR_TSW`.
pub const HCR_TSW: u64 = 1 << 22;
/// `HCR_TPCP`.
pub const HCR_TPCP: u64 = 1 << 23;
/// `HCR_TPU`.
pub const HCR_TPU: u64 = 1 << 24;
/// `HCR_TTLB`.
pub const HCR_TTLB: u64 = 1 << 25;
/// `HCR_TVM`.
pub const HCR_TVM: u64 = 1 << 26;
/// `HCR_TGE`.
pub const HCR_TGE: u64 = 1 << 27;
/// `HCR_TDZ`.
pub const HCR_TDZ: u64 = 1 << 28;
/// `HCR_HCD`.
pub const HCR_HCD: u64 = 1 << 29;
/// `HCR_TRVM`.
pub const HCR_TRVM: u64 = 1 << 30;
/// `HCR_RW`.
pub const HCR_RW: u64 = 1 << 31;
/// `HCR_CD`.
pub const HCR_CD: u64 = 1 << 32;
/// `HCR_ID`.
pub const HCR_ID: u64 = 1 << 33;
/// `HCR_E2H`.
pub const HCR_E2H: u64 = 1 << 34;
/// `HCR_TLOR`.
pub const HCR_TLOR: u64 = 1 << 35;
/// `HCR_MIOCNCE`.
pub const HCR_MIOCNCE: u64 = 1 << 38;
/// `HCR_NV`.
pub const HCR_NV: u64 = 1 << 42;
/// `HCR_NV1`.
pub const HCR_NV1: u64 = 1 << 43;
/// `HCR_FWB`.
pub const HCR_FWB: u64 = 1 << 46;
/// `HCR_TID4`.
pub const HCR_TID4: u64 = 1 << 49;
/// `HCR_TICAB`.
pub const HCR_TICAB: u64 = 1 << 50;
/// `HCR_TOCU`.
pub const HCR_TOCU: u64 = 1 << 52;
/// `HCR_ENSCXT`.
pub const HCR_ENSCXT: u64 = 1 << 53;
/// `HCR_TTLBIS`.
pub const HCR_TTLBIS: u64 = 1 << 54;
/// `HCR_TTLBOS`.
pub const HCR_TTLBOS: u64 = 1 << 55;
/// `HCR_TID5`.
pub const HCR_TID5: u64 = 1 << 58;

/// `GTIMER_PHYS`: the EL1 physical timer.
pub const GTIMER_PHYS: usize = 0;
/// `GTIMER_VIRT`: the EL1 virtual timer.
pub const GTIMER_VIRT: usize = 1;
/// `GTIMER_HYP`: the EL2 physical timer.
pub const GTIMER_HYP: usize = 2;
/// `GTIMER_SEC`: the EL3 (Secure EL1) physical timer.
pub const GTIMER_SEC: usize = 3;
/// `GTIMER_HYPVIRT`: the EL2 virtual timer.
pub const GTIMER_HYPVIRT: usize = 4;
/// The number of generic timers.
pub const NUM_GTIMERS: usize = 5;

/// `ARMMMUIdx_E10_0`: EL0 accesses in the EL1&0 regime.
pub const MMU_IDX_E10_0: usize = 0;
/// `ARMMMUIdx_E10_1`: EL1 accesses in the EL1&0 regime.
pub const MMU_IDX_E10_1: usize = 2;
/// `ARMMMUIdx_E10_1_PAN`: EL1 accesses with PSTATE.PAN set.
pub const MMU_IDX_E10_1_PAN: usize = 3;
/// `ARMMMUIdx_E20_0`: EL0 accesses in the EL2&0 regime (E2H and TGE set).
pub const MMU_IDX_E20_0: usize = 5;
/// `ARMMMUIdx_E20_2`: EL2 accesses in the EL2&0 regime (E2H set).
pub const MMU_IDX_E20_2: usize = 7;
/// `ARMMMUIdx_E20_2_PAN`: EL2 accesses in the EL2&0 regime with PSTATE.PAN set.
pub const MMU_IDX_E20_2_PAN: usize = 8;
/// `ARMMMUIdx_E2`: EL2 accesses in the EL2 regime (E2H clear).
pub const MMU_IDX_E2: usize = 10;
/// `ARMMMUIdx_E3`: EL3 accesses.
pub const MMU_IDX_E3: usize = 12;
/// The number of MMU indexes the runtime is configured with.
pub const NB_MMU_MODES: usize = 16;

impl CpuArmState {
    /// `arm_current_el()`.
    pub fn current_el(&self) -> u32 {
        (self.pstate >> 2) & 3
    }

    /// `pstate_read()`: PSTATE with NZCV and DAIF folded in.
    pub fn pstate_read(&self) -> u32 {
        let z = u32::from(self.zf == 0);
        (self.pstate & !(PSTATE_NZCV | PSTATE_DAIF))
            | (self.nf & 0x8000_0000)
            | (z << 30)
            | ((self.cf & 1) << 29)
            | ((self.vf & 0x8000_0000) >> 3)
            | (self.daif & PSTATE_DAIF)
    }

    /// `pstate_write()`.
    pub fn pstate_write(&mut self, val: u32) {
        self.zf = !val & PSTATE_Z;
        self.nf = val;
        self.cf = (val >> 29) & 1;
        self.vf = (val << 3) & 0x8000_0000;
        self.daif = val & PSTATE_DAIF;
        self.pstate = val & !(PSTATE_NZCV | PSTATE_DAIF);
    }

    /// The NZCV flags in bits 31 to 28.
    pub fn nzcv(&self) -> u32 {
        self.pstate_read() & PSTATE_NZCV
    }

    /// Set the NZCV flags from bits 31 to 28 of `v`.
    pub fn set_nzcv(&mut self, v: u32) {
        self.nf = v & PSTATE_N;
        self.zf = !v & PSTATE_Z;
        self.cf = (v >> 29) & 1;
        self.vf = (v << 3) & 0x8000_0000;
    }

    /// `aarch64_save_sp()`: copy the current SP to its banked slot.
    pub fn save_sp(&mut self, el: u32) {
        if self.pstate & PSTATE_SP != 0 {
            self.sp_el[el as usize] = self.xregs[31];
        } else {
            self.sp_el[0] = self.xregs[31];
        }
    }

    /// `aarch64_restore_sp()`: load the current SP from its banked slot.
    pub fn restore_sp(&mut self, el: u32) {
        if self.pstate & PSTATE_SP != 0 {
            self.xregs[31] = self.sp_el[el as usize];
        } else {
            self.xregs[31] = self.sp_el[0];
        }
    }

    /// `update_spsel()`.
    pub fn update_spsel(&mut self, imm: u32) {
        let cur_el = self.current_el();
        // Update PSTATE SPSel bit; this requires us to update the working stack pointer in
        // xregs[31].
        if (self.pstate & PSTATE_SP) == (imm & 1) {
            return;
        }
        self.save_sp(cur_el);
        self.pstate = (self.pstate & !PSTATE_SP) | (imm & 1);
        self.restore_sp(cur_el);
    }

    /// `arm_is_secure_below_el3()`: EL3 exists and SCR_EL3.NS is clear. There is no
    /// Secure EL2.
    pub fn is_secure_below_el3(&self, f: &ArmFeatures) -> bool {
        f.el3 && self.scr_el3 & SCR_NS == 0
    }

    /// `arm_is_el2_enabled()`: EL2 exists and the CPU is in Non-secure state below EL3.
    pub fn is_el2_enabled(&self, f: &ArmFeatures) -> bool {
        f.el2 && !self.is_secure_below_el3(f)
    }

    /// `arm_hcr_el2_eff()`: HCR_EL2 as it affects the CPU, with the bits TGE forces.
    pub fn hcr_el2_eff(&self, f: &ArmFeatures) -> u64 {
        if !self.is_el2_enabled(f) {
            return 0;
        }
        let mut ret = self.hcr_el2;
        if ret & HCR_TGE != 0 {
            if ret & HCR_E2H != 0 {
                ret &= !(HCR_VM
                    | HCR_FMO
                    | HCR_IMO
                    | HCR_AMO
                    | HCR_BSU_MASK
                    | HCR_DC
                    | HCR_TWI
                    | HCR_TWE
                    | HCR_TID0
                    | HCR_TID2
                    | HCR_TPCP
                    | HCR_TPU
                    | HCR_TDZ
                    | HCR_CD
                    | HCR_ID
                    | HCR_MIOCNCE
                    | HCR_TID4
                    | HCR_TICAB
                    | HCR_TOCU
                    | HCR_ENSCXT
                    | HCR_TTLBIS
                    | HCR_TTLBOS
                    | HCR_TID5);
            } else {
                ret |= HCR_FMO | HCR_IMO | HCR_AMO;
            }
            ret &= !(HCR_SWIO
                | HCR_PTW
                | HCR_VF
                | HCR_VI
                | HCR_VSE
                | HCR_FB
                | HCR_TID1
                | HCR_TID3
                | HCR_TSC
                | HCR_TACR
                | HCR_TSW
                | HCR_TTLB
                | HCR_TVM
                | HCR_HCD
                | HCR_TRVM
                | HCR_TLOR);
        }
        ret
    }

    /// The E2H bit of HCR_EL2 when EL2 is enabled, `el_is_in_host()` without TGE.
    pub fn e2h(&self, f: &ArmFeatures) -> bool {
        self.hcr_el2_eff(f) & HCR_E2H != 0
    }

    /// `arm_mmu_idx_el()`: the MMU index for data accesses at `el`.
    pub fn mmu_idx_el(&self, f: &ArmFeatures, el: u32) -> usize {
        let hcr = self.hcr_el2_eff(f);
        let pan = self.pstate & PSTATE_PAN != 0;
        match el {
            0 if hcr & (HCR_E2H | HCR_TGE) == HCR_E2H | HCR_TGE => MMU_IDX_E20_0,
            0 => MMU_IDX_E10_0,
            1 if pan => MMU_IDX_E10_1_PAN,
            1 => MMU_IDX_E10_1,
            2 if hcr & HCR_E2H != 0 && pan => MMU_IDX_E20_2_PAN,
            2 if hcr & HCR_E2H != 0 => MMU_IDX_E20_2,
            2 => MMU_IDX_E2,
            _ => MMU_IDX_E3,
        }
    }

    /// The MMU index for data accesses at the current EL, `arm_mmu_idx()`.
    pub fn mmu_idx(&self, f: &ArmFeatures) -> usize {
        self.mmu_idx_el(f, self.current_el())
    }

    /// `scr_write()` for an AArch64 only CPU.
    pub fn scr_write(&mut self, f: &ArmFeatures, value: u64) {
        let mut valid = 0x3fff & !SCR_NET;
        if f.lor {
            valid |= SCR_TLOR;
        }
        if f.tcr2 {
            valid |= SCR_TCR2EN;
        }
        if !f.el2 {
            valid &= !SCR_HCE;
        }
        self.scr_el3 = (value | SCR_FW | SCR_AW | SCR_RW) & valid;
    }

    /// `do_hcr_write()` without the interrupt line updates, which `Arm` does after it.
    /// `psci_smc` is whether the PSCI conduit is SMC.
    pub fn hcr_write(&mut self, f: &ArmFeatures, psci_smc: bool, value: u64) {
        let mut valid = (1u64 << 34) - 1;
        if f.el3 {
            valid &= !HCR_HCD;
        } else if !psci_smc {
            // Without EL3 SMC is only useful for PSCI, so TSC is RES0 unless SMC is the
            // conduit.
            valid &= !HCR_TSC;
        }
        if f.vh {
            valid |= HCR_E2H;
        }
        if f.lor {
            valid |= HCR_TLOR;
        }
        self.hcr_el2 = (value & valid) | HCR_RW;
    }

    /// The state after a cold reset of `model`, as `arm_cpu_reset_hold()` leaves an AArch64
    /// CPU: the highest implemented EL in its h mode with DAIF masked, the MMU off and the PC
    /// at zero.
    pub fn reset(model: &ArmCpuModel) -> CpuArmState {
        let f = &model.features;
        let mut s = CpuArmState::default();
        let mode = if f.el3 {
            PSTATE_MODE_EL3H
        } else if f.el2 {
            PSTATE_MODE_EL2H
        } else {
            PSTATE_MODE_EL1H
        };
        s.pstate_write(mode | PSTATE_DAIF);
        s.sctlr_el[1] = model.reset_sctlr;
        // As in QEMU, SCTLR_EL2 resets to zero and SCTLR_EL3 to the model's SCTLR value.
        if f.el2 {
            s.hcr_write(f, false, 0);
            // VPIDR_EL2 resets to MIDR_EL1. VMPIDR_EL2 resets to MPIDR_EL1, which depends on
            // the CPU index, so `create_vcpu()` fills it in.
            s.vpidr_el2 = model.midr;
        }
        if f.el3 {
            s.sctlr_el[3] = model.reset_sctlr;
            s.scr_write(f, 0);
        }
        s.cntfrq_el0 = model.cntfrq;
        s.exclusive_addr = u64::MAX;
        // The OS lock is locked out of reset.
        s.oslsr_el1 = 10;
        s
    }

    /// `arm_emulate_firmware_reset()`: put a CPU that has just been reset into the state
    /// firmware would leave it in for code entered at `target_el`.
    pub fn emulate_firmware_reset(&mut self, f: &ArmFeatures, target_el: u32) {
        match target_el {
            3 => return,
            2 if !f.el3 => return,
            1 if !f.el3 && !f.el2 => return,
            _ => {}
        }
        if f.el3 {
            self.scr_el3 |= SCR_RW;
            if target_el == 2 {
                // If the guest is at EL2 then Linux expects the HVC insn to work.
                self.scr_el3 |= SCR_HCE;
            }
            // Put CPU into non-secure state.
            self.scr_el3 |= SCR_NS;
        }
        if f.el2 && target_el < 2 {
            self.hcr_el2 |= HCR_RW;
        }
        self.pstate_write((target_el << 2) | PSTATE_SP | (self.pstate_read() & !0x1f));
    }
}

/// The optional architecture features this slice cares about, the `isar_feature_aa64_*`
/// predicates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArmFeatures {
    /// FEAT_LSE: CAS, CASP, LDADD and friends, SWP.
    pub lse: bool,
    /// FEAT_CRC32.
    pub crc32: bool,
    /// FEAT_PAN.
    pub pan: bool,
    /// FEAT_UAO.
    pub uao: bool,
    /// FEAT_LOR: LDLAR and STLLR.
    pub lor: bool,
    /// FEAT_LRCPC: LDAPR.
    pub rcpc: bool,
    /// FEAT_HAFDBS level: 1 for the access flag, 2 for the dirty state too.
    pub hafdbs: u8,
    /// FEAT_HPDS: TCR_EL1.HPD0 and HPD1.
    pub hpds: bool,
    /// FEAT_DPB: DC CVAP.
    pub dpb: bool,
    /// FEAT_FP16: half precision arithmetic (`aa64_fp16`).
    pub fp16: bool,
    /// FEAT_RDM: SQRDMLAH and SQRDMLSH.
    pub rdm: bool,
    /// FEAT_DotProd: SDOT and UDOT.
    pub dotprod: bool,
    /// FEAT_AES: AESE, AESD, AESMC and AESIMC.
    pub aes: bool,
    /// FEAT_PMULL: PMULL and PMULL2 with 64-bit elements.
    pub pmull: bool,
    /// FEAT_SHA1.
    pub sha1: bool,
    /// FEAT_SHA256.
    pub sha256: bool,
    /// `ARM_FEATURE_EL2`.
    pub el2: bool,
    /// `ARM_FEATURE_EL3`.
    pub el3: bool,
    /// FEAT_VHE: HCR_EL2.E2H.
    pub vh: bool,
    /// The 16K translation granule (`aa64_tgran16`).
    pub tgran16: bool,
    /// The 64K translation granule (`aa64_tgran64`).
    pub tgran64: bool,
    /// FEAT_SVE (`aa64_sve`).
    pub sve: bool,
    /// FEAT_SVE2 (`aa64_sve2`).
    pub sve2: bool,
    /// FEAT_SVE_AES with FEAT_SVE_PMULL128 (`aa64_sve2_aes` and `aa64_sve2_pmull128`).
    pub sve_aes: bool,
    /// FEAT_SVE_BitPerm (`aa64_sve2_bitperm`).
    pub sve_bitperm: bool,
    /// FEAT_SVE_SHA3 (`aa64_sve2_sha3`).
    pub sve_sha3: bool,
    /// FEAT_SVE_SM4 (`aa64_sve2_sm4`).
    pub sve_sm4: bool,
    /// FEAT_F32MM (`aa64_sve_f32mm`).
    pub sve_f32mm: bool,
    /// FEAT_F64MM (`aa64_sve_f64mm`).
    pub sve_f64mm: bool,
    /// FEAT_BF16 for SVE (`aa64_sve_bf16`): BFCVT, BFCVTNT, BFDOT, BFMMLA and BFMLAL.
    pub sve_bf16: bool,
    /// FEAT_I8MM for SVE (`aa64_sve_i8mm`): SMMLA, UMMLA, USMMLA, USDOT and SUDOT.
    pub sve_i8mm: bool,
    /// FEAT_SVE2p1 (`aa64_sve2p1`). Only DUPQ looks at it; no model sets it, as the rest of
    /// SVE2.1 is not implemented, so it exists for testing DUPQ.
    pub sve2p1: bool,
    /// The largest vector length in quadwords, QEMU's `sve-max-vq` (every length from 1 to
    /// this is supported, as for TCG). Zero without SVE.
    pub sve_max_vq: u32,
    /// FEAT_TLBIOS: the Outer Shareable TLBI operations (`aa64_tlbios`).
    pub tlbios: bool,
    /// FEAT_XS: the TLBI nXS operations and DSB nXS (`aa64_xs`).
    pub xs: bool,
    /// FEAT_TCR2: TCR2_EL1 and TCR2_EL2 (`aa64_tcr2`).
    pub tcr2: bool,
    /// FEAT_ASID2: TCR2_ELx.A2, FNG0 and FNG1 (`aa64_asid2`).
    pub asid2: bool,
    /// A GICv3 CPU interface is attached ([`Arm::with_gicv3`](crate::tcg::Arm::with_gicv3)),
    /// so the ICC system registers exist, as `gicv3_init_cpuif()` defines them.
    pub gicv3: bool,
    /// The preemption bits of the attached GICv3 CPU interface (`cs->prebits`), which decide
    /// whether ICC_AP0R1_EL1 to ICC_AP1R3_EL1 exist.
    pub gic_prebits: u8,
}

/// A CPU model: the identification registers and reset values of `aarch64_*_initfn()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArmCpuModel {
    /// The QOM type name without the `-arm-cpu` suffix.
    pub name: &'static str,
    /// MIDR_EL1.
    pub midr: u64,
    /// REVIDR_EL1.
    pub revidr: u64,
    /// CTR_EL0.
    pub ctr: u64,
    /// DCZID_EL0.
    pub dczid: u64,
    /// CLIDR_EL1.
    pub clidr: u64,
    /// The CCSIDR_EL1 of each cache CSSELR_EL1 can select, indexed by CSSELR_EL1.{Level, InD}
    /// as QEMU's `ccsidr[]` is. Zero for a cache that does not exist.
    pub ccsidr: [u64; 8],
    /// The `compatible` string of the CPU's device tree node, QEMU's `dtb_compatible`.
    pub dtb_compatible: &'static str,
    /// SCTLR_EL1 out of reset.
    pub reset_sctlr: u64,
    /// CNTFRQ_EL0 out of reset, in Hz.
    pub cntfrq: u64,
    /// The priority bits of the GICv3 CPU interface, QEMU's `gic_pribits` (5 on every
    /// model here).
    pub gic_pribits: u8,
    /// ID_AA64PFR0_EL1.
    pub id_aa64pfr0: u64,
    /// ID_AA64PFR1_EL1.
    pub id_aa64pfr1: u64,
    /// ID_AA64DFR0_EL1.
    pub id_aa64dfr0: u64,
    /// ID_AA64ZFR0_EL1.
    pub id_aa64zfr0: u64,
    /// ID_AA64ISAR0_EL1.
    pub id_aa64isar0: u64,
    /// ID_AA64ISAR1_EL1.
    pub id_aa64isar1: u64,
    /// ID_AA64MMFR0_EL1.
    pub id_aa64mmfr0: u64,
    /// ID_AA64MMFR1_EL1.
    pub id_aa64mmfr1: u64,
    /// ID_AA64MMFR2_EL1.
    pub id_aa64mmfr2: u64,
    /// ID_AA64MMFR3_EL1.
    pub id_aa64mmfr3: u64,
    /// ID_AA64MMFR4_EL1.
    pub id_aa64mmfr4: u64,
    /// The features, derived from the ID registers.
    pub features: ArmFeatures,
}

/// `ID_AA64PFR0_EL1` for a CPU with AArch64 only EL0 and EL1, no EL2 or EL3 (until
/// [`ArmCpuModel::with_el2`] and [`ArmCpuModel::with_el3`] add them), and FP and AdvSIMD
/// without half precision (both fields 0).
const PFR0_EL01: u64 = 0x0000_0011;

/// `ID_AA64PFR0_EL1` FP and AdvSIMD fields at 1: implemented with half precision.
const PFR0_FP16: u64 = 0x0011_0000;

/// `ID_AA64MMFR0_EL1` fields shared by both models: 16 bit ASIDs, Secure and Non-secure
/// memory distinguished, 4K granule only (TGran64 reported as 0xf, TGran16 as 0), and no
/// mixed endian support.
const MMFR0_4K_ONLY: u64 = 0x0f00_1020;

/// `make_ccsidr(CCSIDR_FORMAT_LEGACY, assoc, linesize, cachesize, flags)`: a 32 bit CCSIDR
/// for a cache of `size` bytes in lines of `line` bytes, `assoc` ways, with `flags` in the
/// top bits (WT, WB, RA and WA, the bits QEMU still sets).
pub const fn make_ccsidr(assoc: u64, line: u64, size: u64, flags: u64) -> u64 {
    let sets = size / (assoc * line);
    let lg_line = line.trailing_zeros() as u64;
    (flags << 28) | ((sets - 1) << 13) | ((assoc - 1) << 3) | (lg_line - 4)
}

const KIB: u64 = 1024;
const MIB: u64 = 1024 * 1024;

impl ArmCpuModel {
    /// `cortex-a57`: ARMv8.0 with CRC32 and the AES, PMULL, SHA1 and SHA256 crypto
    /// extensions, and no LSE.
    pub fn cortex_a57() -> ArmCpuModel {
        ArmCpuModel {
            name: "cortex-a57",
            midr: 0x411f_d070,
            revidr: 0,
            ctr: 0x8444_c004,
            dczid: 4,
            clidr: 0x0a20_0023,
            // 32 KiB L1 D, 48 KiB L1 I and 2 MiB L2, `aarch64_aa32_a57_init()`.
            ccsidr: [
                make_ccsidr(4, 64, 32 * KIB, 7),
                make_ccsidr(3, 64, 48 * KIB, 2),
                make_ccsidr(16, 64, 2 * MIB, 7),
                0,
                0,
                0,
                0,
                0,
            ],
            dtb_compatible: "arm,cortex-a57",
            reset_sctlr: 0x00c5_0838,
            cntfrq: 62_500_000,
            gic_pribits: 5,
            id_aa64pfr0: PFR0_EL01,
            id_aa64pfr1: 0,
            id_aa64dfr0: 0x6,
            id_aa64zfr0: 0,
            // AES 2 (with PMULL), SHA1 1, SHA2 1 and CRC32 1, QEMU's 0x00011120.
            id_aa64isar0: 0x0001_1120,
            id_aa64isar1: 0,
            // PARange 4, 44 bits.
            id_aa64mmfr0: MMFR0_4K_ONLY | 4,
            id_aa64mmfr1: 0,
            id_aa64mmfr2: 0,
            id_aa64mmfr3: 0,
            id_aa64mmfr4: 0,
            features: ArmFeatures {
                crc32: true,
                aes: true,
                pmull: true,
                sha1: true,
                sha256: true,
                ..ArmFeatures::default()
            },
        }
    }

    /// `cortex-a76`: ARMv8.2 with LSE, PAN, UAO, LOR, LRCPC and hardware access flag and
    /// dirty state management, half precision FP, RDM, the dot product and the AES, PMULL,
    /// SHA1 and SHA256 crypto extensions.
    pub fn cortex_a76() -> ArmCpuModel {
        ArmCpuModel {
            name: "cortex-a76",
            midr: 0x414f_d0b1,
            revidr: 0,
            ctr: 0x8444_c004,
            dczid: 4,
            clidr: 0x8200_0023,
            // 64 KiB L1 D, 64 KiB L1 I and 512 KiB L2.
            ccsidr: [
                make_ccsidr(4, 64, 64 * KIB, 7),
                make_ccsidr(4, 64, 64 * KIB, 2),
                make_ccsidr(8, 64, 512 * KIB, 7),
                0,
                0,
                0,
                0,
                0,
            ],
            dtb_compatible: "arm,cortex-a76",
            reset_sctlr: 0x30c5_0838,
            cntfrq: 62_500_000,
            gic_pribits: 5,
            // CSV2 and CSV3 as in QEMU; RAS is not modelled.
            id_aa64pfr0: 0x1100_0000_0000_0000 | PFR0_FP16 | PFR0_EL01,
            id_aa64pfr1: 0,
            id_aa64dfr0: 0x6,
            id_aa64zfr0: 0,
            // DP 1, RDM 1, Atomic 2, CRC32 1, SHA2 1, SHA1 1 and AES 2, QEMU's
            // 0x0000100010211120.
            id_aa64isar0: 0x0000_1000_1021_1120,
            // LRCPC 1 and DPB 1.
            id_aa64isar1: 0x0010_0001,
            // PARange 2, 40 bits, and all three granules: QEMU's 0x00101122 without
            // BigEnd.
            id_aa64mmfr0: 0x0010_1022,
            // PAN 1, LO 1, HPDS 1, VH 1, HAFDBS 2.
            id_aa64mmfr1: 0x0011_1102,
            // UAO 1, CnP 1.
            id_aa64mmfr2: 0x11,
            id_aa64mmfr3: 0,
            id_aa64mmfr4: 0,
            features: ArmFeatures {
                lse: true,
                crc32: true,
                pan: true,
                uao: true,
                lor: true,
                rcpc: true,
                hafdbs: 2,
                hpds: true,
                dpb: true,
                fp16: true,
                rdm: true,
                dotprod: true,
                aes: true,
                pmull: true,
                sha1: true,
                sha256: true,
                vh: true,
                tgran16: true,
                tgran64: true,
                ..ArmFeatures::default()
            },
        }
    }

    /// `cortex-a72`: ARMv8.0 like the A57 with CRC32 and the crypto extensions, and the 4K
    /// and 64K granules.
    pub fn cortex_a72() -> ArmCpuModel {
        ArmCpuModel {
            name: "cortex-a72",
            midr: 0x410f_d083,
            revidr: 0,
            ctr: 0x8444_c004,
            dczid: 4,
            clidr: 0x0a20_0023,
            // 32 KiB L1 D, 48 KiB L1 I and 1 MiB L2.
            ccsidr: [
                make_ccsidr(4, 64, 32 * KIB, 7),
                make_ccsidr(3, 64, 48 * KIB, 2),
                make_ccsidr(16, 64, MIB, 7),
                0,
                0,
                0,
                0,
                0,
            ],
            dtb_compatible: "arm,cortex-a72",
            reset_sctlr: 0x00c5_0838,
            cntfrq: 62_500_000,
            gic_pribits: 5,
            id_aa64pfr0: PFR0_EL01,
            id_aa64pfr1: 0,
            id_aa64dfr0: 0x1030_5106,
            id_aa64zfr0: 0,
            id_aa64isar0: 0x0001_1120,
            id_aa64isar1: 0,
            // QEMU's 0x00001124 without BigEnd: PARange 4, 16 bit ASIDs, 4K and 64K
            // granules.
            id_aa64mmfr0: 0x0000_1024,
            id_aa64mmfr1: 0,
            id_aa64mmfr2: 0,
            id_aa64mmfr3: 0,
            id_aa64mmfr4: 0,
            features: ArmFeatures {
                crc32: true,
                aes: true,
                pmull: true,
                sha1: true,
                sha256: true,
                tgran64: true,
                ..ArmFeatures::default()
            },
        }
    }

    /// `max` cut down to what this port implements: the `cortex-a76` feature set with QEMU's
    /// `max` MIDR, CTR_EL0.IDC and DIC, and SVE2 with the AES, PMULL128, BitPerm, SHA3 and
    /// SM4 extensions, F32MM, F64MM, BF16 and I8MM, at vector lengths up to 2048 bits. The
    /// vector length limit is [`ArmCpuModel::with_sve_max_vq`], `-cpu max,sve-max-vq=N`.
    /// It also has FEAT_TLBIOS, FEAT_XS, FEAT_TCR2 and FEAT_ASID2, which are only maintenance
    /// operations and register bits here. QEMU's `max` has many more features (SVE2p1, EBF16,
    /// SVE_B16B16, SME, MTE, PAuth and so on) that this port does not; their ID register
    /// fields read as zero here.
    pub fn max() -> ArmCpuModel {
        let a76 = ArmCpuModel::cortex_a76();
        ArmCpuModel {
            name: "max",
            midr: 0x000f_0510,
            ctr: a76.ctr | (1 << 28) | (1 << 29),
            // QEMU's 0x8200123 with LoUU and LoUIS cleared for FEAT_S2FWB.
            clidr: 0x0000_0123,
            // 64 KiB L1 D, 64 KiB L1 I, 1 MiB L2 and 2 MiB L3. `max` starts from the A57, so
            // its device tree node says it is one.
            ccsidr: [
                make_ccsidr(4, 64, 64 * KIB, 7),
                make_ccsidr(4, 64, 64 * KIB, 2),
                make_ccsidr(8, 64, MIB, 7),
                0,
                make_ccsidr(8, 64, 2 * MIB, 7),
                0,
                0,
                0,
            ],
            dtb_compatible: "arm,cortex-a57",
            // ID_AA64PFR0_EL1.SVE = 1.
            id_aa64pfr0: a76.id_aa64pfr0 | (1 << 32),
            // SVEver 1 (SVE2), AES 2 (with PMULL128), BitPerm 1, BF16 1, SHA3 1, SM4 1,
            // I8MM 1, F32MM 1 and F64MM 1. QEMU's `max` sets SVEver 2 (SVE2p1), BF16 2
            // (FEAT_EBF16) and B16B16 1; those fields come up when their instructions land.
            id_aa64zfr0: 0x0110_1101_0011_0021,
            // ID_AA64ISAR0_EL1.TLB = 1 (FEAT_TLBIOS). QEMU's `max` has 2, FEAT_TLBIRANGE.
            id_aa64isar0: a76.id_aa64isar0 | (1 << 56),
            // ID_AA64ISAR1_EL1.XS = 1.
            id_aa64isar1: a76.id_aa64isar1 | (1 << 56),
            // ID_AA64MMFR3_EL1.TCRX = 1.
            id_aa64mmfr3: 1,
            // ID_AA64MMFR4_EL1.ASID2 = 1.
            id_aa64mmfr4: 1 << 8,
            features: ArmFeatures {
                tlbios: true,
                xs: true,
                tcr2: true,
                asid2: true,
                sve: true,
                sve2: true,
                sve_aes: true,
                sve_bitperm: true,
                sve_sha3: true,
                sve_sm4: true,
                sve_f32mm: true,
                sve_f64mm: true,
                sve_bf16: true,
                sve_i8mm: true,
                sve_max_vq: ARM_MAX_VQ as u32,
                ..a76.features
            },
            ..a76
        }
    }

    /// The model with vector lengths limited to `vq` quadwords (1 to 16), QEMU's
    /// `sve-max-vq` property. It has no effect on a model without SVE.
    pub fn with_sve_max_vq(mut self, vq: u32) -> ArmCpuModel {
        assert!((1..=ARM_MAX_VQ as u32).contains(&vq), "sve-max-vq must be from 1 to 16");
        if self.features.sve {
            self.features.sve_max_vq = vq;
        }
        self
    }

    /// The model with EL2 implemented (AArch64 only), as the virt board's
    /// `virtualization=on` leaves `ARM_FEATURE_EL2` set. The models start without EL2 and
    /// EL3, as the virt board's defaults leave them.
    pub fn with_el2(mut self) -> ArmCpuModel {
        self.features.el2 = true;
        self.id_aa64pfr0 = (self.id_aa64pfr0 & !0xf00) | 0x100;
        self
    }

    /// The model with EL3 implemented (AArch64 only), as the virt board's `secure=on`.
    pub fn with_el3(mut self) -> ArmCpuModel {
        self.features.el3 = true;
        self.id_aa64pfr0 = (self.id_aa64pfr0 & !0xf000) | 0x1000;
        self
    }

    /// The model called `name`, `cortex-a57`, `cortex-a72`, `cortex-a76` or `max`.
    pub fn by_name(name: &str) -> Option<ArmCpuModel> {
        match name {
            "max" => Some(ArmCpuModel::max()),
            "cortex-a57" => Some(ArmCpuModel::cortex_a57()),
            "cortex-a72" => Some(ArmCpuModel::cortex_a72()),
            "cortex-a76" => Some(ArmCpuModel::cortex_a76()),
            _ => None,
        }
    }

    /// The physical address size in bits, from ID_AA64MMFR0_EL1.PARange.
    pub fn pamax(&self) -> u32 {
        pa_range_bits((self.id_aa64mmfr0 & 0xf) as u32)
    }
}

/// `arm_pamax()` for a PARange value.
pub fn pa_range_bits(parange: u32) -> u32 {
    match parange {
        0 => 32,
        1 => 36,
        2 => 40,
        3 => 42,
        4 => 44,
        5 => 48,
        _ => 52,
    }
}
