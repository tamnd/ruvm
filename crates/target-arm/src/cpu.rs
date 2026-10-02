// SPDX-License-Identifier: GPL-2.0-or-later

//! `CPUARMState` for AArch64 and the CPU models: the parts of QEMU's `target/arm/cpu.h`,
//! `cpu64.c` and `cpu-max.c` this slice needs.
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

            /// Write the state into a vCPU's `env` buffer.
            pub fn store(&self, env: &mut [u8]) {
                $(self.$f.put(&mut env[ENV_TARGET_OFFSET + offset_of!($name, $f)..]);)*
            }
        }
    };
}

arm_state! {
    /// The AArch64 register state, `CPUARMState` cut down to EL0 and EL1.
    ///
    /// As in QEMU, `xregs[31]` is the current stack pointer, the banked stack pointers are in
    /// `sp_el`, and the condition flags are kept apart from `pstate`: `nf` and `vf` hold the
    /// flag in bit 31, `zf` is zero exactly when Z is set, and `cf` is 0 or 1. `daif` holds
    /// the D, A, I and F bits at their PSTATE positions. Per exception level arrays are
    /// indexed by EL; only index 1 (and 0 for `sp_el` and `tpidr_el`) is used.
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
        /// SP_EL0 and SP_EL1 while they are not the current SP.
        pub sp_el: [u64; 4],
        /// ELR_ELx.
        pub elr_el: [u64; 4],
        /// SPSR_ELx.
        pub spsr_el: [u64; 4],
        /// SCTLR_ELx.
        pub sctlr_el: [u64; 4],
        /// TCR_ELx.
        pub tcr_el: [u64; 4],
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
        /// TPIDR_EL0 and TPIDR_EL1.
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
        /// CNTP_CTL_EL0.
        pub cntp_ctl_el0: u64,
        /// CNTP_CVAL_EL0.
        pub cntp_cval_el0: u64,
        /// CNTV_CTL_EL0.
        pub cntv_ctl_el0: u64,
        /// CNTV_CVAL_EL0.
        pub cntv_cval_el0: u64,
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
        /// The AdvSIMD and FP registers V0 to V31, QEMU's `vfp.zregs` cut down to 128 bits:
        /// `zregs[n][0]` is the low half of Vn and `zregs[n][1]` the high half.
        pub zregs: [[u64; 2]; 32],
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

/// The `env` offset of the low 64 bits of Vn.
pub const fn vreg_off(n: usize) -> usize {
    env_off(offset_of!(CpuArmState, zregs)) + 16 * n
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

/// `ARMMMUIdx_E10_0`: EL0 accesses in the EL1&0 regime.
pub const MMU_IDX_E10_0: usize = 0;
/// `ARMMMUIdx_E10_1`: EL1 accesses in the EL1&0 regime.
pub const MMU_IDX_E10_1: usize = 2;
/// `ARMMMUIdx_E10_1_PAN`: EL1 accesses with PSTATE.PAN set.
pub const MMU_IDX_E10_1_PAN: usize = 3;
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

    /// The MMU index for data accesses at the current EL, `arm_mmu_idx()`.
    pub fn mmu_idx(&self) -> usize {
        match self.current_el() {
            0 => MMU_IDX_E10_0,
            _ if self.pstate & PSTATE_PAN != 0 => MMU_IDX_E10_1_PAN,
            _ => MMU_IDX_E10_1,
        }
    }

    /// The state after a cold reset of `model`: EL1h with DAIF masked, the MMU off and the PC
    /// at zero, as `arm_cpu_reset_hold()` leaves an AArch64 CPU without EL2 and EL3.
    pub fn reset(model: &ArmCpuModel) -> CpuArmState {
        let mut s = CpuArmState::default();
        s.pstate_write(PSTATE_MODE_EL1H | PSTATE_DAIF);
        s.sctlr_el[1] = model.reset_sctlr;
        s.cntfrq_el0 = model.cntfrq;
        s.exclusive_addr = u64::MAX;
        // The OS lock is locked out of reset.
        s.oslsr_el1 = 10;
        s
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
    /// SCTLR_EL1 out of reset.
    pub reset_sctlr: u64,
    /// CNTFRQ_EL0 out of reset, in Hz.
    pub cntfrq: u64,
    /// ID_AA64PFR0_EL1.
    pub id_aa64pfr0: u64,
    /// ID_AA64PFR1_EL1.
    pub id_aa64pfr1: u64,
    /// ID_AA64DFR0_EL1.
    pub id_aa64dfr0: u64,
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
    /// The features, derived from the ID registers.
    pub features: ArmFeatures,
}

/// `ID_AA64PFR0_EL1` for a CPU with AArch64 only EL0 and EL1, no EL2 or EL3, and FP and
/// AdvSIMD without half precision (both fields 0).
const PFR0_EL01: u64 = 0x0000_0011;

/// `ID_AA64PFR0_EL1` FP and AdvSIMD fields at 1: implemented with half precision.
const PFR0_FP16: u64 = 0x0011_0000;

/// `ID_AA64MMFR0_EL1` fields shared by both models: 16 bit ASIDs, Secure and Non-secure
/// memory distinguished, 4K granule only (TGran64 reported as 0xf, TGran16 as 0), and no
/// mixed endian support.
const MMFR0_4K_ONLY: u64 = 0x0f00_1020;

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
            reset_sctlr: 0x00c5_0838,
            cntfrq: 62_500_000,
            id_aa64pfr0: PFR0_EL01,
            id_aa64pfr1: 0,
            id_aa64dfr0: 0x6,
            // AES 2 (with PMULL), SHA1 1, SHA2 1 and CRC32 1, QEMU's 0x00011120.
            id_aa64isar0: 0x0001_1120,
            id_aa64isar1: 0,
            // PARange 4, 44 bits.
            id_aa64mmfr0: MMFR0_4K_ONLY | 4,
            id_aa64mmfr1: 0,
            id_aa64mmfr2: 0,
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
            reset_sctlr: 0x30c5_0838,
            cntfrq: 62_500_000,
            // CSV2 and CSV3 as in QEMU; RAS is not modelled.
            id_aa64pfr0: 0x1100_0000_0000_0000 | PFR0_FP16 | PFR0_EL01,
            id_aa64pfr1: 0,
            id_aa64dfr0: 0x6,
            // DP 1, RDM 1, Atomic 2, CRC32 1, SHA2 1, SHA1 1 and AES 2, QEMU's
            // 0x0000100010211120.
            id_aa64isar0: 0x0000_1000_1021_1120,
            // LRCPC 1 and DPB 1.
            id_aa64isar1: 0x0010_0001,
            // PARange 2, 40 bits.
            id_aa64mmfr0: MMFR0_4K_ONLY | 2,
            // PAN 1, LO 1, HPDS 1, HAFDBS 2.
            id_aa64mmfr1: 0x0011_1002,
            // UAO 1, CnP 1.
            id_aa64mmfr2: 0x11,
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
            },
        }
    }

    /// The model called `name`, `cortex-a57` or `cortex-a76`.
    pub fn by_name(name: &str) -> Option<ArmCpuModel> {
        match name {
            "cortex-a57" => Some(ArmCpuModel::cortex_a57()),
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
