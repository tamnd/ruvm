// SPDX-License-Identifier: GPL-2.0-or-later

//! Arm vCPU setup and register sync from target/arm/kvm.c, with the cpreg list merge from
//! target/arm/machine.c.
//!
//! A KVM Arm vCPU is set up with `KVM_ARM_VCPU_INIT` and then read and written one register at a
//! time with `KVM_GET_ONE_REG` and `KVM_SET_ONE_REG`. The core registers have fixed ids. Every
//! other register the kernel knows about comes from `KVM_GET_REG_LIST` and lives in a sorted list
//! of ids and values, the cpreg list, which is also what migrates. Every sequence here is written
//! against [`ArmVcpuOps`], so the order of the accesses and the encoding of each id can be checked
//! with a fake vCPU on any host. On AArch64 Linux, `KvmVcpu` implements the trait.
//!
//! ruvm has no TCG CPU behind a KVM vCPU, so the cpreg list is the only copy of the system
//! registers and [`CoreRegs`] the only copy of the core ones. Where QEMU syncs those with
//! `CPUARMState`, this code does nothing.

use std::fmt;
use std::io;

use crate::strerror;

/// `KVM_REG_ARM64`: the architecture field of every AArch64 register id.
pub const KVM_REG_ARM64: u64 = 0x6000_0000_0000_0000;
/// `KVM_REG_SIZE_SHIFT`.
pub const KVM_REG_SIZE_SHIFT: u32 = 52;
/// `KVM_REG_SIZE_MASK`.
pub const KVM_REG_SIZE_MASK: u64 = 0x00f0_0000_0000_0000;
/// `KVM_REG_SIZE_U32`.
pub const KVM_REG_SIZE_U32: u64 = 0x0020_0000_0000_0000;
/// `KVM_REG_SIZE_U64`.
pub const KVM_REG_SIZE_U64: u64 = 0x0030_0000_0000_0000;
/// `KVM_REG_SIZE_U128`.
pub const KVM_REG_SIZE_U128: u64 = 0x0040_0000_0000_0000;
/// `KVM_REG_SIZE_U256`.
pub const KVM_REG_SIZE_U256: u64 = 0x0050_0000_0000_0000;
/// `KVM_REG_SIZE_U512`.
pub const KVM_REG_SIZE_U512: u64 = 0x0060_0000_0000_0000;
/// `KVM_REG_SIZE_U2048`.
pub const KVM_REG_SIZE_U2048: u64 = 0x0080_0000_0000_0000;

/// `KVM_REG_ARM_COPROC_MASK`: which bank a register id belongs to.
pub const KVM_REG_ARM_COPROC_MASK: u64 = 0x0fff_0000;
/// `KVM_REG_ARM_CORE`: the `struct kvm_regs` fields, addressed by offset.
pub const KVM_REG_ARM_CORE: u64 = 0x0010 << 16;
/// `KVM_REG_ARM_DEMUX`: the cache size registers.
pub const KVM_REG_ARM_DEMUX: u64 = 0x0011 << 16;
/// `KVM_REG_ARM64_SYSREG`: system registers, addressed by their encoding.
pub const KVM_REG_ARM64_SYSREG: u64 = 0x0013 << 16;
/// `KVM_REG_ARM_FW`: firmware pseudo registers such as the PSCI version.
pub const KVM_REG_ARM_FW: u64 = 0x0014 << 16;
/// `KVM_REG_ARM64_SVE`: the SVE registers.
pub const KVM_REG_ARM64_SVE: u64 = 0x0015 << 16;
/// `KVM_REG_ARM_FW_FEAT_BMAP`: the firmware feature bitmaps.
pub const KVM_REG_ARM_FW_FEAT_BMAP: u64 = 0x0016 << 16;

/// `KVM_ARM_VCPU_POWER_OFF`: start the vCPU stopped, waiting for PSCI `CPU_ON`.
pub const KVM_ARM_VCPU_POWER_OFF: u32 = 0;
/// `KVM_ARM_VCPU_EL1_32BIT`: EL1 runs AArch32.
pub const KVM_ARM_VCPU_EL1_32BIT: u32 = 1;
/// `KVM_ARM_VCPU_PSCI_0_2`: PSCI 0.2 or later instead of 0.1.
pub const KVM_ARM_VCPU_PSCI_0_2: u32 = 2;
/// `KVM_ARM_VCPU_PMU_V3`.
pub const KVM_ARM_VCPU_PMU_V3: u32 = 3;
/// `KVM_ARM_VCPU_SVE`.
pub const KVM_ARM_VCPU_SVE: u32 = 4;
/// `KVM_ARM_VCPU_PTRAUTH_ADDRESS`.
pub const KVM_ARM_VCPU_PTRAUTH_ADDRESS: u32 = 5;
/// `KVM_ARM_VCPU_PTRAUTH_GENERIC`.
pub const KVM_ARM_VCPU_PTRAUTH_GENERIC: u32 = 6;
/// `KVM_ARM_VCPU_HAS_EL2`.
pub const KVM_ARM_VCPU_HAS_EL2: u32 = 7;

/// `KVM_ARM_VCPU_PMU_V3_CTRL`: the vCPU attribute group of the PMU.
pub const KVM_ARM_VCPU_PMU_V3_CTRL: u32 = 0;
/// `KVM_ARM_VCPU_PMU_V3_IRQ`: the PPI the PMU overflow interrupt goes to.
pub const KVM_ARM_VCPU_PMU_V3_IRQ: u64 = 0;
/// `KVM_ARM_VCPU_PMU_V3_INIT`.
pub const KVM_ARM_VCPU_PMU_V3_INIT: u64 = 1;

/// `KVM_MP_STATE_RUNNABLE`.
pub const KVM_MP_STATE_RUNNABLE: u32 = 0;
/// `KVM_MP_STATE_STOPPED`: powered off by PSCI.
pub const KVM_MP_STATE_STOPPED: u32 = 5;

/// `KVM_CAP_ARM_EL2`, which kvm-bindings does not have yet.
pub const KVM_CAP_ARM_EL2: u32 = 240;

/// `ARM64_AFFINITY_MASK`: Aff3 in bits 39:32 and Aff2 to Aff0 in bits 23:0.
pub const ARM64_AFFINITY_MASK: u64 = 0xff_00ff_ffff;

/// `QEMU_PSCI_VERSION_0_1`.
pub const PSCI_VERSION_0_1: u32 = 0x1;
/// `QEMU_PSCI_VERSION_0_2`.
pub const PSCI_VERSION_0_2: u32 = 0x2;

/// `PSCI_VERSION(major, minor)`.
pub const fn psci_version(major: u16, minor: u16) -> u32 {
    ((major as u32) << 16) | minor as u32
}

/// Parses the `kvm-psci-version` CPU property, which takes `major.minor`.
pub fn parse_psci_version(value: &str) -> Result<u32, String> {
    // QEMU reads it with sscanf("%hu.%hu"), which skips leading blanks and stops at the first
    // character that is not a digit.
    fn number(s: &str) -> Option<(u16, &str)> {
        let s = s.trim_start();
        let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
        Some((s[..end].parse().ok()?, &s[end..]))
    }
    let parsed = number(value).and_then(|(major, rest)| {
        let (minor, _) = number(rest.strip_prefix('.')?)?;
        Some(psci_version(major, minor))
    });
    parsed.ok_or_else(|| "Invalid PSCI version.".to_string())
}

/// Prints a PSCI version as `kvm_get_psci_version()` does.
pub fn format_psci_version(version: u32) -> String {
    format!("{}.{}", version >> 16, version & 0xffff)
}

/// `ARM64_SYS_REG(op0, op1, crn, crm, op2)`: the id of a 64-bit system register.
pub const fn sys_reg(op0: u64, op1: u64, crn: u64, crm: u64, op2: u64) -> u64 {
    KVM_REG_ARM64
        | KVM_REG_SIZE_U64
        | KVM_REG_ARM64_SYSREG
        | ((op0 << 14) & 0xc000)
        | ((op1 << 11) & 0x3800)
        | ((crn << 7) & 0x0780)
        | ((crm << 3) & 0x0078)
        | (op2 & 0x0007)
}

/// `MPIDR_EL1`.
pub const MPIDR_EL1: u64 = sys_reg(3, 0, 0, 0, 5);
/// `KVM_REG_ARM_TIMER_CTL`, `CNTV_CTL_EL0`.
pub const KVM_REG_ARM_TIMER_CTL: u64 = sys_reg(3, 3, 14, 3, 1);
/// `KVM_REG_ARM_TIMER_CNT`, the virtual count. The kernel header swaps the encodings of this
/// one and [`KVM_REG_ARM_TIMER_CVAL`] by mistake, and that mistake is now the ABI.
pub const KVM_REG_ARM_TIMER_CNT: u64 = sys_reg(3, 3, 14, 3, 2);
/// `KVM_REG_ARM_TIMER_CVAL`, `CNTV_CVAL_EL0`.
pub const KVM_REG_ARM_TIMER_CVAL: u64 = sys_reg(3, 3, 14, 0, 2);
/// `KVM_REG_ARM_PTIMER_CTL`, `CNTP_CTL_EL0`.
pub const KVM_REG_ARM_PTIMER_CTL: u64 = sys_reg(3, 3, 14, 2, 1);
/// `KVM_REG_ARM_PTIMER_CVAL`, `CNTP_CVAL_EL0`.
pub const KVM_REG_ARM_PTIMER_CVAL: u64 = sys_reg(3, 3, 14, 2, 2);
/// `KVM_REG_ARM_PTIMER_CNT`, the physical count.
pub const KVM_REG_ARM_PTIMER_CNT: u64 = sys_reg(3, 3, 14, 0, 1);

/// `KVM_REG_ARM_FW_REG(r)`.
pub const fn fw_reg(r: u64) -> u64 {
    KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM_FW | (r & 0xffff)
}

/// `KVM_REG_ARM_PSCI_VERSION`.
pub const KVM_REG_ARM_PSCI_VERSION: u64 = fw_reg(0);

/// `KVM_REG_ARM_FW_FEAT_BMAP_REG(r)`.
pub const fn fw_feat_bmap_reg(r: u64) -> u64 {
    KVM_REG_ARM64 | KVM_REG_SIZE_U64 | KVM_REG_ARM_FW_FEAT_BMAP | (r & 0xffff)
}

/// The id of the `struct kvm_regs` field at byte `offset`, `AARCH64_CORE_REG()`.
pub const fn core_reg(offset: u64, size: u64) -> u64 {
    KVM_REG_ARM64 | size | KVM_REG_ARM_CORE | (offset / 4)
}

/// The byte offsets of the `struct kvm_regs` fields. The AArch64 Linux code checks them against
/// kvm-bindings.
pub mod core_offset {
    /// `regs.regs[0]`; register n is 8n bytes further.
    pub const X0: u64 = 0;
    /// `regs.sp`, which is SP_EL0.
    pub const SP: u64 = 248;
    /// `regs.pc`.
    pub const PC: u64 = 256;
    /// `regs.pstate`.
    pub const PSTATE: u64 = 264;
    /// `sp_el1`.
    pub const SP_EL1: u64 = 272;
    /// `elr_el1`.
    pub const ELR_EL1: u64 = 280;
    /// `spsr[0]`; bank n is 8n bytes further.
    pub const SPSR: u64 = 288;
    /// `fp_regs.vregs[0]`; register n is 16n bytes further.
    pub const VREGS: u64 = 336;
    /// `fp_regs.fpsr`.
    pub const FPSR: u64 = 848;
    /// `fp_regs.fpcr`.
    pub const FPCR: u64 = 852;
}

/// `KVM_NR_SPSR`.
pub const KVM_NR_SPSR: usize = 5;
/// `KVM_ARM64_SVE_NUM_ZREGS`.
pub const SVE_NUM_ZREGS: usize = 32;
/// `KVM_ARM64_SVE_NUM_PREGS`.
pub const SVE_NUM_PREGS: usize = 16;

/// `KVM_REG_ARM64_SVE_ZREG(n, i)`: slice `i` of Z register `n`, 2048 bits.
pub const fn sve_zreg(n: u64, i: u64) -> u64 {
    KVM_REG_ARM64 | KVM_REG_ARM64_SVE | KVM_REG_SIZE_U2048 | ((n & 31) << 5) | (i & 31)
}

/// `KVM_REG_ARM64_SVE_PREG(n, i)`: slice `i` of P register `n`, 256 bits.
pub const fn sve_preg(n: u64, i: u64) -> u64 {
    KVM_REG_ARM64 | KVM_REG_ARM64_SVE | 0x400 | KVM_REG_SIZE_U256 | ((n & 15) << 5) | (i & 31)
}

/// `KVM_REG_ARM64_SVE_FFR(i)`: slice `i` of the first fault register, 256 bits.
pub const fn sve_ffr(i: u64) -> u64 {
    KVM_REG_ARM64 | KVM_REG_ARM64_SVE | 0x600 | KVM_REG_SIZE_U256 | (i & 31)
}

/// `KVM_REG_ARM64_SVE_VLS`: the set of vector lengths, one bit per quadword count.
pub const KVM_REG_ARM64_SVE_VLS: u64 =
    KVM_REG_ARM64 | KVM_REG_ARM64_SVE | KVM_REG_SIZE_U512 | 0xffff;

/// The size in bytes of the register `id`, from its size field.
pub const fn reg_size(id: u64) -> usize {
    1 << ((id & KVM_REG_SIZE_MASK) >> KVM_REG_SIZE_SHIFT)
}

/// `kvm_arm_reg_syncs_via_cpreg_list()`: whether `id` belongs in the cpreg list rather than in
/// the registers [`CoreRegs`] syncs by hand.
pub fn syncs_via_cpreg_list(id: u64) -> bool {
    !matches!(id & KVM_REG_ARM_COPROC_MASK, KVM_REG_ARM_CORE | KVM_REG_ARM64_SVE)
}

/// `kvm_print_register_name()`.
pub fn register_name(id: u64) -> String {
    let field = |mask: u64, shift: u32| (id & mask) >> shift;
    match id & KVM_REG_ARM_COPROC_MASK {
        KVM_REG_ARM_CORE => format!("core reg {id:x}"),
        KVM_REG_ARM_DEMUX => format!("demuxed reg {id:x}"),
        KVM_REG_ARM64_SYSREG => format!(
            "system register op0:{} op1:{} crn:{} crm:{} op2:{}",
            field(0xc000, 14),
            field(0x3800, 11),
            field(0x0780, 7),
            field(0x0078, 3),
            field(0x0007, 0)
        ),
        KVM_REG_ARM_FW => format!("fw reg {}", id & 0xffff),
        KVM_REG_ARM64_SVE => sve_register_name(id),
        KVM_REG_ARM_FW_FEAT_BMAP => format!("fw feat reg {}", id & 0xffff),
        _ => format!("{id:x}"),
    }
}

/// `kvm_print_sve_register_name()`. QEMU masks the id with 0xfc00 and then compares it with
/// 0x04 and 0x06, so it names every P register and the FFR "SVE ???". This uses the real
/// bases, 0x400 and 0x600.
fn sve_register_name(id: u64) -> String {
    if id == KVM_REG_ARM64_SVE_VLS {
        return "SVE VLS".to_string();
    }
    let reg = id & 0xffff;
    match reg {
        0..0x400 => format!("SVE zreg n:{} slice:{}", (reg & 0x03e0) >> 5, reg & 0x1f),
        0x400..0x600 => format!("SVE preg n:{} slice:{}", (reg & 0x01e0) >> 5, reg & 0x1f),
        0x600..0x620 => format!("SVE ffr slice:{}", reg & 0x1f),
        _ => "SVE ???".to_string(),
    }
}

/// How much state a register put carries, `KVM_PUT_RUNTIME_STATE` and its friends.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PutLevel {
    /// The state that changes while the guest runs, put before every resume.
    Runtime = 1,
    /// Also the state a reset sets.
    Reset = 2,
    /// Everything, after an incoming migration or a load.
    Full = 3,
}

/// `kvm_arm_cpreg_level()`: the counters are only written with [`PutLevel::Full`], since writing
/// them at runtime would make guest time jump.
pub fn cpreg_level(id: u64) -> PutLevel {
    match id {
        KVM_REG_ARM_TIMER_CNT | KVM_REG_ARM_PTIMER_CNT => PutLevel::Full,
        _ => PutLevel::Runtime,
    }
}

/// The pending SError of `struct kvm_vcpu_events`, `env->serror`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct SErrorEvents {
    /// An SError is pending.
    pub pending: bool,
    /// It comes with the syndrome in `esr`.
    pub has_esr: bool,
    /// The syndrome.
    pub esr: u64,
}

/// The vCPU ioctls the Arm setup and sync sequences use. Every method is one ioctl.
pub trait ArmVcpuOps {
    /// `KVM_ARM_VCPU_INIT`.
    fn vcpu_init(&mut self, target: u32, features: &[u32; 7]) -> io::Result<()>;
    /// `KVM_ARM_VCPU_FINALIZE`.
    fn vcpu_finalize(&mut self, feature: u32) -> io::Result<()>;
    /// `KVM_GET_REG_LIST`: every register id of the vCPU, in the kernel's order.
    fn reg_list(&mut self) -> io::Result<Vec<u64>>;
    /// `KVM_GET_ONE_REG`. `data` is [`reg_size`] bytes long, in host byte order.
    fn get_one_reg(&mut self, id: u64, data: &mut [u8]) -> io::Result<()>;
    /// `KVM_SET_ONE_REG`. `data` is [`reg_size`] bytes long, in host byte order.
    fn set_one_reg(&mut self, id: u64, data: &[u8]) -> io::Result<()>;
    /// `KVM_GET_MP_STATE`.
    fn get_mp_state(&mut self) -> io::Result<u32>;
    /// `KVM_SET_MP_STATE`.
    fn set_mp_state(&mut self, state: u32) -> io::Result<()>;
    /// `KVM_GET_VCPU_EVENTS`.
    fn get_vcpu_events(&mut self) -> io::Result<SErrorEvents>;
    /// `KVM_SET_VCPU_EVENTS`, with every other field zero.
    fn set_vcpu_events(&mut self, events: &SErrorEvents) -> io::Result<()>;
    /// `KVM_HAS_DEVICE_ATTR` on the vCPU. `value`, when given, is the `int` the attribute's
    /// address points to.
    fn has_device_attr(&mut self, group: u32, attr: u64, value: Option<i32>) -> io::Result<()>;
    /// `KVM_SET_DEVICE_ATTR` on the vCPU, with `value` as for `has_device_attr`.
    fn set_device_attr(&mut self, group: u32, attr: u64, value: Option<i32>) -> io::Result<()>;
}

/// Why an Arm vCPU step failed. The messages are QEMU's.
#[derive(Debug)]
pub enum ArmVcpuError {
    /// A vCPU ioctl failed during setup.
    Ioctl(&'static str, io::Error),
    /// One register could not be read or written.
    OneReg {
        /// The register id.
        id: u64,
        /// `KVM_SET_ONE_REG` rather than `KVM_GET_ONE_REG`.
        write: bool,
        /// What the kernel said.
        err: io::Error,
    },
    /// `KVM_GET_REG_LIST` named a register that is neither 32 nor 64 bits wide.
    RegSize(u64),
    /// The cpreg list could not be read right after it was built.
    InitialRead,
    /// `write_list_to_kvmstate()` failed: one message per register the kernel refused.
    SetList(Vec<String>),
    /// The kernel does not do the PSCI version asked for.
    PsciVersion(u32, io::Error),
    /// A PMU attribute could not be set. `what` is the line QEMU prints after the ioctl's.
    Pmu {
        /// `KVM_HAS_DEVICE_ATTR` or `KVM_SET_DEVICE_ATTR`.
        ioctl: &'static str,
        /// The step that failed.
        what: &'static str,
        /// What the kernel said.
        err: io::Error,
    },
    /// The virtual counter could not be read (`write` false) or written.
    VirtualTime {
        /// The write failed rather than the read.
        write: bool,
    },
    /// `KVM_GET_VCPU_EVENTS` or `KVM_SET_VCPU_EVENTS` failed.
    VcpuEvents {
        /// The set failed rather than the get.
        write: bool,
        /// What the kernel said.
        err: io::Error,
    },
    /// The incoming cpreg list has registers this vCPU does not have. One message each.
    Migration(Vec<String>),
    /// The incoming cpreg index and value arrays have different lengths.
    MigrationLength,
    /// More than 256 vCPUs on a kernel without the second `KVM_IRQ_LINE` layout.
    TooManyVcpus,
}

impl fmt::Display for ArmVcpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ioctl(what, err) => write!(f, "{what} failed: {}", strerror(err)),
            Self::OneReg { id, write, err } => write!(
                f,
                "{} of {} failed: {}",
                if *write { "KVM_SET_ONE_REG" } else { "KVM_GET_ONE_REG" },
                register_name(*id),
                strerror(err)
            ),
            Self::RegSize(_) => f.write_str("Can't handle size of register in kernel list"),
            Self::InitialRead => f.write_str("Initial read of kernel register state failed"),
            Self::SetList(lines) | Self::Migration(lines) => f.write_str(&lines.join("\n")),
            Self::PsciVersion(version, _) => write!(
                f,
                "KVM in this kernel does not support PSCI version {}.{}\nConsider setting the \
                 kvm-psci-version property on the migration source.",
                version >> 16,
                version & 0xffff
            ),
            Self::Pmu { ioctl, what, err } => write!(f, "PMU: {ioctl}: {}\n{what}", strerror(err)),
            Self::VirtualTime { write } => {
                write!(f, "Failed to {} KVM_REG_ARM_TIMER_CNT", if *write { "set" } else { "get" })
            }
            Self::VcpuEvents { write, .. } => {
                write!(f, "failed to {} vcpu events", if *write { "put" } else { "get" })
            }
            Self::MigrationLength => {
                f.write_str("cpreg migration: index and value arrays differ in length")
            }
            Self::TooManyVcpus => f.write_str(
                "Using more than 256 vcpus requires a host kernel with \
                 KVM_CAP_ARM_IRQ_LINE_LAYOUT_2",
            ),
        }
    }
}

impl std::error::Error for ArmVcpuError {}

/// The `kvm_arch_init()` vCPU count check: past 256 vCPUs the interrupt line encoding needs
/// `KVM_CAP_ARM_IRQ_LINE_LAYOUT_2`.
pub fn check_vcpu_count(cpus: u32, irq_line_layout_2: bool) -> Result<(), ArmVcpuError> {
    if cpus > 256 && !irq_line_layout_2 {
        return Err(ArmVcpuError::TooManyVcpus);
    }
    Ok(())
}

fn get_u64(ops: &mut impl ArmVcpuOps, id: u64) -> Result<u64, ArmVcpuError> {
    let mut buf = [0u8; 8];
    ops.get_one_reg(id, &mut buf).map_err(|err| ArmVcpuError::OneReg { id, write: false, err })?;
    Ok(u64::from_ne_bytes(buf))
}

fn set_u64(ops: &mut impl ArmVcpuOps, id: u64, value: u64) -> Result<(), ArmVcpuError> {
    set_bytes(ops, id, &value.to_ne_bytes())
}

fn get_u32(ops: &mut impl ArmVcpuOps, id: u64) -> Result<u32, ArmVcpuError> {
    let mut buf = [0u8; 4];
    ops.get_one_reg(id, &mut buf).map_err(|err| ArmVcpuError::OneReg { id, write: false, err })?;
    Ok(u32::from_ne_bytes(buf))
}

fn set_bytes(ops: &mut impl ArmVcpuOps, id: u64, data: &[u8]) -> Result<(), ArmVcpuError> {
    ops.set_one_reg(id, data).map_err(|err| ArmVcpuError::OneReg { id, write: true, err })
}

fn get_words(ops: &mut impl ArmVcpuOps, id: u64, words: &mut [u64]) -> Result<(), ArmVcpuError> {
    let mut buf = vec![0u8; words.len() * 8];
    ops.get_one_reg(id, &mut buf).map_err(|err| ArmVcpuError::OneReg { id, write: false, err })?;
    for (w, b) in words.iter_mut().zip(buf.chunks_exact(8)) {
        *w = u64::from_ne_bytes(b.try_into().expect("8 byte chunk"));
    }
    Ok(())
}

fn set_words(ops: &mut impl ArmVcpuOps, id: u64, words: &[u64]) -> Result<(), ArmVcpuError> {
    let buf: Vec<u8> = words.iter().flat_map(|w| w.to_ne_bytes()).collect();
    set_bytes(ops, id, &buf)
}

/// A migration tolerance for one cpreg, the two kinds `arm_register_cpreg_mig_tolerance()`
/// accepts.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CpregTolerance {
    /// The register may be missing on either end.
    NotOnBothEnds,
    /// The register may come in a stream for a destination without it, if its value masked
    /// with `mask` is `value`.
    OnlySrcTestValue {
        /// The bits to compare.
        mask: u64,
        /// What they must be.
        value: u64,
    },
}

/// The tolerances `kvm_arm_set_cpreg_mig_tolerances()` registers for `-cpu host`: registers
/// older kernels exposed by mistake, and a firmware bitmap newer kernels added.
pub const KVM_CPREG_TOLERANCES: &[(u64, CpregTolerance)] = &[
    // TCR2_EL1, exposed even when disabled before Linux 6.13.
    (sys_reg(3, 0, 2, 0, 3), CpregTolerance::NotOnBothEnds),
    // PIRE0_EL1 and PIR_EL1, likewise.
    (sys_reg(3, 0, 10, 2, 2), CpregTolerance::NotOnBothEnds),
    (sys_reg(3, 0, 10, 2, 3), CpregTolerance::NotOnBothEnds),
    // KVM_REG_ARM_VENDOR_HYP_BMAP_2, new in Linux 6.15, fine to drop while it is 0.
    (fw_feat_bmap_reg(3), CpregTolerance::OnlySrcTestValue { mask: u64::MAX, value: 0 }),
];

fn tolerance(tolerances: &[(u64, CpregTolerance)], id: u64) -> Option<CpregTolerance> {
    tolerances.iter().find(|t| t.0 == id).map(|t| t.1)
}

/// The cpreg list, `cpreg_indexes` and `cpreg_values`: every register the kernel lists that
/// [`CoreRegs`] does not sync, sorted by id. 32-bit registers keep their value zero extended.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CpregList {
    indexes: Vec<u64>,
    values: Vec<u64>,
}

impl CpregList {
    /// The list part of `kvm_arm_init_cpreg_list()`: sorts what `KVM_GET_REG_LIST` returned and
    /// keeps the registers that sync through the list. Every value starts at 0.
    pub fn from_kernel(mut ids: Vec<u64>) -> Result<Self, ArmVcpuError> {
        ids.sort_unstable();
        ids.retain(|&id| syncs_via_cpreg_list(id));
        if let Some(&id) = ids
            .iter()
            .find(|&&id| !matches!(id & KVM_REG_SIZE_MASK, KVM_REG_SIZE_U32 | KVM_REG_SIZE_U64))
        {
            return Err(ArmVcpuError::RegSize(id));
        }
        let values = vec![0; ids.len()];
        Ok(CpregList { indexes: ids, values })
    }

    /// The register ids, sorted.
    pub fn indexes(&self) -> &[u64] {
        &self.indexes
    }

    /// The values, in the order of [`CpregList::indexes`].
    pub fn values(&self) -> &[u64] {
        &self.values
    }

    /// The number of registers.
    pub fn len(&self) -> usize {
        self.indexes.len()
    }

    /// Whether the list is empty, which it only is before the vCPU is set up.
    pub fn is_empty(&self) -> bool {
        self.indexes.is_empty()
    }

    /// The value of register `id`, if the list has it.
    pub fn get(&self, id: u64) -> Option<u64> {
        self.indexes.binary_search(&id).ok().map(|i| self.values[i])
    }

    /// `kvm_arm_get_cpreg_ptr()`: the value of register `id`, if the list has it.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut u64> {
        self.indexes.binary_search(&id).ok().map(|i| &mut self.values[i])
    }

    /// `write_kvmstate_to_list()`: reads every register from the kernel. A register that fails
    /// keeps its old value and the rest are still read; the first failure is returned.
    pub fn read(&mut self, ops: &mut impl ArmVcpuOps) -> Result<(), ArmVcpuError> {
        let mut first = None;
        for (&id, value) in self.indexes.iter().zip(self.values.iter_mut()) {
            let got = if id & KVM_REG_SIZE_MASK == KVM_REG_SIZE_U32 {
                get_u32(ops, id).map(u64::from)
            } else {
                get_u64(ops, id)
            };
            match got {
                Ok(v) => *value = v,
                Err(e) => {
                    first.get_or_insert(e);
                }
            }
        }
        first.map_or(Ok(()), Err)
    }

    /// `write_list_to_kvmstate()`: writes every register whose [`cpreg_level`] is at most
    /// `level`. A refused register is reported the way QEMU does, after reading back what the
    /// kernel has when it said `EINVAL`, and the rest are still written.
    pub fn write(&self, ops: &mut impl ArmVcpuOps, level: PutLevel) -> Result<(), ArmVcpuError> {
        let mut errors = Vec::new();
        for (&id, &value) in self.indexes.iter().zip(&self.values) {
            if cpreg_level(id) > level {
                continue;
            }
            let u32_reg = id & KVM_REG_SIZE_MASK == KVM_REG_SIZE_U32;
            let set = if u32_reg {
                ops.set_one_reg(id, &(value as u32).to_ne_bytes())
            } else {
                ops.set_one_reg(id, &value.to_ne_bytes())
            };
            let Err(err) = set else { continue };
            let name = register_name(id);
            errors.push(match err.raw_os_error() {
                Some(e) if e == ENOENT => {
                    format!("Could not set register {name}: unknown to KVM")
                }
                Some(e) if e == EINVAL && u32_reg => match get_u32(ops, id) {
                    Ok(is) => format!("Could not set register {name} to {:x} (is {is:x})", value),
                    Err(_) => format!("Could not set register {name} to {:x}", value as u32),
                },
                Some(e) if e == EINVAL => match get_u64(ops, id) {
                    Ok(is) => format!("Could not set register {name} to {value:x} (is {is:x})"),
                    Err(_) => format!("Could not set register {name} to {value:x}"),
                },
                _ => format!("Could not set register {name}: {}", strerror(&err)),
            });
        }
        if errors.is_empty() { Ok(()) } else { Err(ArmVcpuError::SetList(errors)) }
    }

    /// The cpreg part of `cpu_post_load()`: takes the values of an incoming stream. A register
    /// the stream lacks keeps its value with a warning, unless a tolerance covers it, and a
    /// register only the stream has fails the load unless a tolerance covers it. Gives back the
    /// warnings. QEMU lets a later tolerated register clear an earlier failure; here any failure
    /// fails the load.
    pub fn merge_incoming(
        &mut self,
        indexes: &[u64],
        values: &[u64],
        tolerances: &[(u64, CpregTolerance)],
    ) -> Result<Vec<String>, ArmVcpuError> {
        if indexes.len() != values.len() {
            return Err(ArmVcpuError::MigrationLength);
        }
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        let missing = |id: u64, warnings: &mut Vec<String>| {
            if tolerance(tolerances, id) != Some(CpregTolerance::NotOnBothEnds) {
                warnings.push(format!(
                    "handle_cpreg_missing_in_incoming_stream: {} expected by the destination but \
                     not in the incoming stream: skip it",
                    register_name(id)
                ));
            }
        };
        let only_incoming = |id: u64, value: u64, errors: &mut Vec<String>| {
            let tolerated = match tolerance(tolerances, id) {
                Some(CpregTolerance::NotOnBothEnds) => true,
                Some(CpregTolerance::OnlySrcTestValue { mask, value: want }) => {
                    value & mask == want
                }
                None => false,
            };
            if !tolerated {
                errors.push(format!(
                    "handle_cpreg_only_in_incoming_stream: {} in the incoming stream but unknown \
                     on the destination: fail migration",
                    register_name(id)
                ));
            }
        };
        let (mut i, mut v) = (0, 0);
        while i < self.indexes.len() && v < indexes.len() {
            if indexes[v] > self.indexes[i] {
                missing(self.indexes[i], &mut warnings);
                i += 1;
            } else if indexes[v] < self.indexes[i] {
                only_incoming(indexes[v], values[v], &mut errors);
                v += 1;
            } else {
                self.values[i] = values[v];
                i += 1;
                v += 1;
            }
        }
        for &id in &self.indexes[i..] {
            missing(id, &mut warnings);
        }
        for (&id, &value) in indexes[v..].iter().zip(&values[v..]) {
            only_incoming(id, value, &mut errors);
        }
        if errors.is_empty() { Ok(warnings) } else { Err(ArmVcpuError::Migration(errors)) }
    }
}

const ENOENT: i32 = 2;
const EINVAL: i32 = 22;

/// The SVE registers, slice 0 of each, as the kernel lays them out: little endian 64-bit words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SveRegs {
    /// Z0 to Z31, 2048 bits each.
    pub z: [[u64; 32]; SVE_NUM_ZREGS],
    /// P0 to P15, 256 bits each.
    pub p: [[u64; 4]; SVE_NUM_PREGS],
    /// The first fault register.
    pub ffr: [u64; 4],
}

impl Default for SveRegs {
    fn default() -> Self {
        SveRegs { z: [[0; 32]; SVE_NUM_ZREGS], p: [[0; 4]; SVE_NUM_PREGS], ffr: [0; 4] }
    }
}

/// The vector registers: the FP and SIMD ones, or the SVE ones once SVE is finalized.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VectorRegs {
    /// V0 to V31.
    Fpsimd(Box<[u128; 32]>),
    /// The SVE state, which contains the V registers.
    Sve(Box<SveRegs>),
}

impl Default for VectorRegs {
    fn default() -> Self {
        VectorRegs::Fpsimd(Box::new([0; 32]))
    }
}

/// The registers `kvm_arch_get_registers()` and `kvm_arch_put_registers()` sync by hand: the
/// `struct kvm_regs` fields and the SVE ones. They hold what the kernel has, so `sp_el0` is
/// `regs.sp` and `pstate` is the raw `regs.pstate`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CoreRegs {
    /// X0 to X30.
    pub x: [u64; 31],
    /// SP_EL0, `regs.sp`.
    pub sp_el0: u64,
    /// SP_EL1.
    pub sp_el1: u64,
    /// PSTATE as the kernel keeps it, including the AArch32 mode bits.
    pub pstate: u64,
    /// The PC.
    pub pc: u64,
    /// ELR_EL1.
    pub elr_el1: u64,
    /// The five banked SPSRs: EL1, then the AArch32 abort, undefined, IRQ and FIQ banks.
    pub spsr: [u64; KVM_NR_SPSR],
    /// The vector registers.
    pub vregs: VectorRegs,
    /// FPSR.
    pub fpsr: u32,
    /// FPCR.
    pub fpcr: u32,
}

impl CoreRegs {
    /// Registers that start with the SVE state rather than the FP and SIMD one.
    pub fn with_sve() -> Self {
        CoreRegs { vregs: VectorRegs::Sve(Box::default()), ..CoreRegs::default() }
    }

    /// The core half of `kvm_arch_put_registers()`, in QEMU's order: X0 to X30, SP_EL0, SP_EL1,
    /// PSTATE, PC, ELR_EL1, the SPSRs, the vector registers, FPSR and FPCR.
    pub fn put(&self, ops: &mut impl ArmVcpuOps) -> Result<(), ArmVcpuError> {
        for (n, &x) in self.x.iter().enumerate() {
            set_u64(ops, core_reg(core_offset::X0 + 8 * n as u64, KVM_REG_SIZE_U64), x)?;
        }
        set_u64(ops, core_reg(core_offset::SP, KVM_REG_SIZE_U64), self.sp_el0)?;
        set_u64(ops, core_reg(core_offset::SP_EL1, KVM_REG_SIZE_U64), self.sp_el1)?;
        set_u64(ops, core_reg(core_offset::PSTATE, KVM_REG_SIZE_U64), self.pstate)?;
        set_u64(ops, core_reg(core_offset::PC, KVM_REG_SIZE_U64), self.pc)?;
        set_u64(ops, core_reg(core_offset::ELR_EL1, KVM_REG_SIZE_U64), self.elr_el1)?;
        for (n, &spsr) in self.spsr.iter().enumerate() {
            set_u64(ops, core_reg(core_offset::SPSR + 8 * n as u64, KVM_REG_SIZE_U64), spsr)?;
        }
        match &self.vregs {
            VectorRegs::Fpsimd(v) => {
                for (n, q) in v.iter().enumerate() {
                    let id = core_reg(core_offset::VREGS + 16 * n as u64, KVM_REG_SIZE_U128);
                    set_bytes(ops, id, &q.to_ne_bytes())?;
                }
            }
            VectorRegs::Sve(sve) => {
                for (n, z) in sve.z.iter().enumerate() {
                    set_words(ops, sve_zreg(n as u64, 0), z)?;
                }
                for (n, p) in sve.p.iter().enumerate() {
                    set_words(ops, sve_preg(n as u64, 0), p)?;
                }
                set_words(ops, sve_ffr(0), &sve.ffr)?;
            }
        }
        set_bytes(ops, core_reg(core_offset::FPSR, KVM_REG_SIZE_U32), &self.fpsr.to_ne_bytes())?;
        set_bytes(ops, core_reg(core_offset::FPCR, KVM_REG_SIZE_U32), &self.fpcr.to_ne_bytes())
    }

    /// The core half of `kvm_arch_get_registers()`, in the same order as [`CoreRegs::put`].
    pub fn get(&mut self, ops: &mut impl ArmVcpuOps) -> Result<(), ArmVcpuError> {
        for (n, x) in self.x.iter_mut().enumerate() {
            *x = get_u64(ops, core_reg(core_offset::X0 + 8 * n as u64, KVM_REG_SIZE_U64))?;
        }
        self.sp_el0 = get_u64(ops, core_reg(core_offset::SP, KVM_REG_SIZE_U64))?;
        self.sp_el1 = get_u64(ops, core_reg(core_offset::SP_EL1, KVM_REG_SIZE_U64))?;
        self.pstate = get_u64(ops, core_reg(core_offset::PSTATE, KVM_REG_SIZE_U64))?;
        self.pc = get_u64(ops, core_reg(core_offset::PC, KVM_REG_SIZE_U64))?;
        self.elr_el1 = get_u64(ops, core_reg(core_offset::ELR_EL1, KVM_REG_SIZE_U64))?;
        for (n, spsr) in self.spsr.iter_mut().enumerate() {
            *spsr = get_u64(ops, core_reg(core_offset::SPSR + 8 * n as u64, KVM_REG_SIZE_U64))?;
        }
        match &mut self.vregs {
            VectorRegs::Fpsimd(v) => {
                for (n, q) in v.iter_mut().enumerate() {
                    let id = core_reg(core_offset::VREGS + 16 * n as u64, KVM_REG_SIZE_U128);
                    let mut buf = [0u8; 16];
                    ops.get_one_reg(id, &mut buf).map_err(|err| ArmVcpuError::OneReg {
                        id,
                        write: false,
                        err,
                    })?;
                    *q = u128::from_ne_bytes(buf);
                }
            }
            VectorRegs::Sve(sve) => {
                for (n, z) in sve.z.iter_mut().enumerate() {
                    get_words(ops, sve_zreg(n as u64, 0), z)?;
                }
                for (n, p) in sve.p.iter_mut().enumerate() {
                    get_words(ops, sve_preg(n as u64, 0), p)?;
                }
                get_words(ops, sve_ffr(0), &mut sve.ffr)?;
            }
        }
        self.fpsr = get_u32(ops, core_reg(core_offset::FPSR, KVM_REG_SIZE_U32))?;
        self.fpcr = get_u32(ops, core_reg(core_offset::FPCR, KVM_REG_SIZE_U32))?;
        Ok(())
    }
}

/// The capabilities the Arm vCPU code looks at, read once from the kernel.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ArmKvmCaps {
    /// `KVM_CAP_ARM_PSCI_0_2`.
    pub psci_0_2: bool,
    /// `KVM_CAP_ARM_EL1_32BIT`.
    pub el1_32bit: bool,
    /// `KVM_CAP_ARM_PMU_V3`.
    pub pmu_v3: bool,
    /// `KVM_CAP_ARM_SVE`.
    pub sve: bool,
    /// `KVM_CAP_ARM_PTRAUTH_ADDRESS` and `KVM_CAP_ARM_PTRAUTH_GENERIC` both.
    pub ptrauth: bool,
    /// [`KVM_CAP_ARM_EL2`].
    pub el2: bool,
    /// `KVM_CAP_MP_STATE`, `cap_has_mp_state`.
    pub mp_state: bool,
    /// `KVM_CAP_VCPU_EVENTS`.
    pub vcpu_events: bool,
    /// `KVM_CAP_ARM_INJECT_SERROR_ESR`, `cap_has_inject_serror_esr`.
    pub inject_serror_esr: bool,
    /// `KVM_CAP_ARM_IRQ_LINE_LAYOUT_2`.
    pub irq_line_layout_2: bool,
}

/// What the CPU model and the board ask of one vCPU.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ArmVcpuConfig {
    /// The `KVM_ARM_VCPU_INIT` target, the one `KVM_ARM_PREFERRED_TARGET` gives.
    pub target: u32,
    /// `start_powered_off`: a secondary CPU that waits for PSCI `CPU_ON`.
    pub start_powered_off: bool,
    /// The `kvm-psci-version` property, or 0 for the newest the kernel has.
    pub psci_version: u32,
    /// The CPU runs AArch64 at EL1. False gives an AArch32 EL1.
    pub aarch64: bool,
    /// The CPU has a PMU, `has_pmu`.
    pub pmu: bool,
    /// The SVE vector lengths, one bit per quadword count less one, or `None` without SVE.
    pub sve_vq_map: Option<u64>,
    /// The CPU has pointer authentication.
    pub pauth: bool,
    /// The CPU has EL2, `has_el2`.
    pub el2: bool,
    /// Stop the virtual counter while the VM is stopped, the opposite of `kvm-no-adjvtime`.
    pub adjvtime: bool,
}

impl ArmVcpuConfig {
    /// `-cpu host` on a kernel with `caps`: a PMU when the kernel has one, and no SVE, pointer
    /// authentication or EL2 until ruvm reads the host's ID registers.
    pub fn host(target: u32, caps: &ArmKvmCaps) -> Self {
        ArmVcpuConfig {
            target,
            start_powered_off: false,
            psci_version: 0,
            aarch64: true,
            pmu: caps.pmu_v3,
            sve_vq_map: None,
            pauth: false,
            el2: false,
            adjvtime: true,
        }
    }
}

/// The `kvm_init_features` part of `kvm_arch_init_vcpu()`.
pub fn init_features(cfg: &ArmVcpuConfig, caps: &ArmKvmCaps) -> [u32; 7] {
    let mut w = 0u32;
    if cfg.start_powered_off {
        w |= 1 << KVM_ARM_VCPU_POWER_OFF;
    }
    // 0.2 and later are compatible with 0.2, so only 0.1 leaves the flag out.
    if cfg.psci_version != PSCI_VERSION_0_1 && caps.psci_0_2 {
        w |= 1 << KVM_ARM_VCPU_PSCI_0_2;
    }
    if !cfg.aarch64 {
        w |= 1 << KVM_ARM_VCPU_EL1_32BIT;
    }
    if cfg.pmu {
        w |= 1 << KVM_ARM_VCPU_PMU_V3;
    }
    if cfg.sve_vq_map.is_some() {
        w |= 1 << KVM_ARM_VCPU_SVE;
    }
    if cfg.pauth {
        w |= (1 << KVM_ARM_VCPU_PTRAUTH_ADDRESS) | (1 << KVM_ARM_VCPU_PTRAUTH_GENERIC);
    }
    if cfg.el2 && caps.el2 {
        w |= 1 << KVM_ARM_VCPU_HAS_EL2;
    }
    [w, 0, 0, 0, 0, 0, 0]
}

/// The virtual counter adjustment, `kvm_vtime` and `kvm_vtime_dirty`. While the VM is stopped
/// the counter is saved, and it is written back when the VM runs again, so the guest does not
/// see the stopped time.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct VirtualTime {
    /// `kvm_adjvtime`: the adjustment is on.
    pub enabled: bool,
    /// The saved counter.
    pub vtime: u64,
    /// The saved counter has not been written back yet.
    pub dirty: bool,
}

impl VirtualTime {
    /// `kvm_arm_get_virtual_time()`.
    fn save(&mut self, ops: &mut impl ArmVcpuOps) -> Result<(), ArmVcpuError> {
        if self.dirty {
            return Ok(());
        }
        self.vtime = get_u64(ops, KVM_REG_ARM_TIMER_CNT)
            .map_err(|_| ArmVcpuError::VirtualTime { write: false })?;
        self.dirty = true;
        Ok(())
    }

    /// `kvm_arm_put_virtual_time()`.
    fn restore(&mut self, ops: &mut impl ArmVcpuOps) -> Result<(), ArmVcpuError> {
        if !self.dirty {
            return Ok(());
        }
        set_u64(ops, KVM_REG_ARM_TIMER_CNT, self.vtime)
            .map_err(|_| ArmVcpuError::VirtualTime { write: true })?;
        self.dirty = false;
        Ok(())
    }
}

/// The KVM side of one Arm vCPU, the KVM fields of `ARMCPU` with the registers it syncs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArmVcpu {
    /// The `KVM_ARM_VCPU_INIT` target.
    pub target: u32,
    /// The `KVM_ARM_VCPU_INIT` features, used again on every reset.
    pub features: [u32; 7],
    /// The capabilities the sync sequences depend on.
    pub caps: ArmKvmCaps,
    /// The PSCI version the kernel implements for this vCPU, which goes into the device tree.
    pub psci_version: u32,
    /// The affinity fields of the MPIDR KVM gave the vCPU, `mp_affinity`.
    pub mp_affinity: u64,
    /// The vCPU has a PMU.
    pub pmu: bool,
    /// The core and vector registers.
    pub regs: CoreRegs,
    /// Every other register.
    pub cpregs: CpregList,
    /// The pending SError.
    pub serror: SErrorEvents,
    /// The vCPU is off, `power_state == PSCI_OFF`.
    pub powered_off: bool,
    /// The virtual counter adjustment.
    pub vtime: VirtualTime,
}

impl ArmVcpu {
    /// `kvm_arch_init_vcpu()`: `KVM_ARM_VCPU_INIT` with the features `cfg` asks for, the SVE
    /// vector lengths and `KVM_ARM_VCPU_FINALIZE`, the PSCI version, the MPIDR KVM picked, then
    /// the cpreg list and its first read.
    pub fn init(
        ops: &mut impl ArmVcpuOps,
        cfg: &ArmVcpuConfig,
        caps: &ArmKvmCaps,
    ) -> Result<Self, ArmVcpuError> {
        let features = init_features(cfg, caps);
        ops.vcpu_init(cfg.target, &features)
            .map_err(|e| ArmVcpuError::Ioctl("KVM_ARM_VCPU_INIT", e))?;
        if let Some(map) = cfg.sve_vq_map {
            let mut vls = [0u64; 8];
            vls[0] = map;
            set_words(ops, KVM_REG_ARM64_SVE_VLS, &vls)?;
            ops.vcpu_finalize(KVM_ARM_VCPU_SVE)
                .map_err(|e| ArmVcpuError::Ioctl("KVM_ARM_VCPU_FINALIZE", e))?;
        }
        let mut psci_version = cfg.psci_version;
        if psci_version != 0 {
            set_u64(ops, KVM_REG_ARM_PSCI_VERSION, u64::from(psci_version)).map_err(
                |e| match e {
                    ArmVcpuError::OneReg { err, .. } => {
                        ArmVcpuError::PsciVersion(psci_version, err)
                    }
                    other => other,
                },
            )?;
        }
        // KVM reports the version it implements through the same register, when it has it.
        if let Ok(v) = get_u64(ops, KVM_REG_ARM_PSCI_VERSION) {
            psci_version = v as u32;
        }
        let mp_affinity = get_u64(ops, MPIDR_EL1)? & ARM64_AFFINITY_MASK;
        let ids = ops.reg_list().map_err(|e| ArmVcpuError::Ioctl("KVM_GET_REG_LIST", e))?;
        let mut cpregs = CpregList::from_kernel(ids)?;
        cpregs.read(ops).map_err(|_| ArmVcpuError::InitialRead)?;
        let regs =
            if cfg.sve_vq_map.is_some() { CoreRegs::with_sve() } else { CoreRegs::default() };
        Ok(ArmVcpu {
            target: cfg.target,
            features,
            caps: *caps,
            psci_version,
            mp_affinity,
            pmu: cfg.pmu,
            regs,
            cpregs,
            serror: SErrorEvents::default(),
            powered_off: cfg.start_powered_off,
            vtime: VirtualTime { enabled: cfg.adjvtime, ..VirtualTime::default() },
        })
    }

    /// `kvm_arm_reset_vcpu()`: `KVM_ARM_VCPU_INIT` again, which puts every register at its
    /// reset value, then reads the cpreg list. QEMU takes the core registers from its own CPU
    /// reset; ruvm has none behind a KVM vCPU, so it reads KVM's, and the board then sets the
    /// boot PC and arguments before the next put.
    pub fn reset(&mut self, ops: &mut impl ArmVcpuOps) -> Result<(), ArmVcpuError> {
        ops.vcpu_init(self.target, &self.features)
            .map_err(|e| ArmVcpuError::Ioctl("KVM_ARM_VCPU_INIT", e))?;
        self.cpregs.read(ops)?;
        self.regs.get(ops)?;
        self.serror = SErrorEvents::default();
        self.powered_off = self.features[0] & (1 << KVM_ARM_VCPU_POWER_OFF) != 0;
        Ok(())
    }

    /// `kvm_arch_get_registers()`: the core registers, the SError, the cpreg list and the power
    /// state.
    pub fn get_registers(&mut self, ops: &mut impl ArmVcpuOps) -> Result<(), ArmVcpuError> {
        self.regs.get(ops)?;
        if self.caps.vcpu_events {
            self.serror = ops
                .get_vcpu_events()
                .map_err(|err| ArmVcpuError::VcpuEvents { write: false, err })?;
        }
        self.cpregs.read(ops)?;
        if self.caps.mp_state {
            let state =
                ops.get_mp_state().map_err(|e| ArmVcpuError::Ioctl("KVM_GET_MP_STATE", e))?;
            self.powered_off = state == KVM_MP_STATE_STOPPED;
        }
        Ok(())
    }

    /// `kvm_arch_put_registers()`: the core registers, the cpreg list at `level`, then the
    /// SError, after the registers so KVM's changes to them are not overwritten, then the power
    /// state.
    pub fn put_registers(
        &mut self,
        ops: &mut impl ArmVcpuOps,
        level: PutLevel,
    ) -> Result<(), ArmVcpuError> {
        self.regs.put(ops)?;
        self.cpregs.write(ops, level)?;
        if self.caps.vcpu_events {
            let mut events = SErrorEvents { pending: self.serror.pending, ..Default::default() };
            if self.caps.inject_serror_esr {
                events.has_esr = self.serror.has_esr;
                events.esr = self.serror.esr;
            }
            ops.set_vcpu_events(&events)
                .map_err(|err| ArmVcpuError::VcpuEvents { write: true, err })?;
        }
        if self.caps.mp_state {
            let state = if self.powered_off { KVM_MP_STATE_STOPPED } else { KVM_MP_STATE_RUNNABLE };
            ops.set_mp_state(state).map_err(|e| ArmVcpuError::Ioctl("KVM_SET_MP_STATE", e))?;
        }
        Ok(())
    }

    /// `kvm_arm_vm_state_change()`: saves the virtual counter when the VM stops and writes it
    /// back when it runs again.
    pub fn vm_state_change(
        &mut self,
        ops: &mut impl ArmVcpuOps,
        running: bool,
    ) -> Result<(), ArmVcpuError> {
        if !self.vtime.enabled {
            return Ok(());
        }
        if running { self.vtime.restore(ops) } else { self.vtime.save(ops) }
    }

    /// `kvm_arm_set_device_attr()` for the PMU group.
    fn pmu_attr(
        ops: &mut impl ArmVcpuOps,
        attr: u64,
        value: Option<i32>,
        what: &'static str,
    ) -> Result<(), ArmVcpuError> {
        ops.has_device_attr(KVM_ARM_VCPU_PMU_V3_CTRL, attr, value)
            .map_err(|err| ArmVcpuError::Pmu { ioctl: "KVM_HAS_DEVICE_ATTR", what, err })?;
        ops.set_device_attr(KVM_ARM_VCPU_PMU_V3_CTRL, attr, value)
            .map_err(|err| ArmVcpuError::Pmu { ioctl: "KVM_SET_DEVICE_ATTR", what, err })
    }

    /// `kvm_arm_pmu_set_irq()`: the PPI of the PMU overflow interrupt. Only with the in-kernel
    /// vGIC, and before [`ArmVcpu::pmu_init`].
    pub fn pmu_set_irq(&self, ops: &mut impl ArmVcpuOps, irq: u32) -> Result<(), ArmVcpuError> {
        if !self.pmu {
            return Ok(());
        }
        Self::pmu_attr(ops, KVM_ARM_VCPU_PMU_V3_IRQ, Some(irq as i32), "failed to set irq for PMU")
    }

    /// `kvm_arm_pmu_init()`.
    pub fn pmu_init(&self, ops: &mut impl ArmVcpuOps) -> Result<(), ArmVcpuError> {
        if !self.pmu {
            return Ok(());
        }
        Self::pmu_attr(ops, KVM_ARM_VCPU_PMU_V3_INIT, None, "failed to init PMU")
    }

    /// The KVM half of `cpu_pre_save()`: reads the cpreg list and, while the counter is held,
    /// puts the held value in it. Returns the ids and values to send.
    pub fn pre_save(
        &mut self,
        ops: &mut impl ArmVcpuOps,
    ) -> Result<(Vec<u64>, Vec<u64>), ArmVcpuError> {
        self.cpregs.read(ops)?;
        if self.vtime.dirty {
            if let Some(cnt) = self.cpregs.get_mut(KVM_REG_ARM_TIMER_CNT) {
                *cnt = self.vtime.vtime;
            }
        }
        Ok((self.cpregs.indexes.clone(), self.cpregs.values.clone()))
    }

    /// The KVM half of `cpu_post_load()`: merges the incoming list, writes all of it to the
    /// kernel and holds the incoming counter until the VM runs. Returns the merge warnings.
    pub fn post_load(
        &mut self,
        ops: &mut impl ArmVcpuOps,
        indexes: &[u64],
        values: &[u64],
        tolerances: &[(u64, CpregTolerance)],
    ) -> Result<Vec<String>, ArmVcpuError> {
        let warnings = self.cpregs.merge_incoming(indexes, values, tolerances)?;
        self.cpregs.write(ops, PutLevel::Full)?;
        if self.vtime.enabled {
            if let Some(cnt) = self.cpregs.get(KVM_REG_ARM_TIMER_CNT) {
                self.vtime.vtime = cnt;
                self.vtime.dirty = true;
            }
        }
        Ok(warnings)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::*;

    /// A vCPU that keeps registers in a map, refuses the ids it is told to, and logs every ioctl.
    #[derive(Default)]
    struct Fake {
        regs: HashMap<u64, Vec<u8>>,
        list: Vec<u64>,
        /// Ids whose set fails, with the errno.
        refuse: HashMap<u64, i32>,
        /// Ids whose get fails.
        unreadable: HashSet<u64>,
        /// Ids the kernel does not have at all, for get and set.
        absent: HashSet<u64>,
        mp_state: u32,
        events: SErrorEvents,
        no_pmu_attr: bool,
        log: Vec<String>,
    }

    impl Fake {
        fn with_list(list: &[u64]) -> Self {
            let mut f = Fake { list: list.to_vec(), ..Fake::default() };
            for &id in list {
                f.regs.insert(id, vec![0; reg_size(id)]);
            }
            f
        }

        fn set64(&mut self, id: u64, v: u64) {
            self.regs.insert(id, v.to_ne_bytes().to_vec());
        }

        fn get64(&self, id: u64) -> u64 {
            u64::from_ne_bytes(self.regs[&id][..8].try_into().unwrap())
        }

        fn sets(&self) -> Vec<&str> {
            self.log.iter().filter(|l| l.starts_with("set ")).map(String::as_str).collect()
        }
    }

    impl ArmVcpuOps for Fake {
        fn vcpu_init(&mut self, target: u32, features: &[u32; 7]) -> io::Result<()> {
            self.log.push(format!("init {target} {:#x}", features[0]));
            Ok(())
        }

        fn vcpu_finalize(&mut self, feature: u32) -> io::Result<()> {
            self.log.push(format!("finalize {feature}"));
            Ok(())
        }

        fn reg_list(&mut self) -> io::Result<Vec<u64>> {
            self.log.push("reg_list".to_string());
            Ok(self.list.clone())
        }

        fn get_one_reg(&mut self, id: u64, data: &mut [u8]) -> io::Result<()> {
            assert_eq!(data.len(), reg_size(id), "get of {id:#x}");
            self.log.push(format!("get {id:#x}"));
            if self.absent.contains(&id) || self.unreadable.contains(&id) {
                return Err(io::Error::from_raw_os_error(ENOENT));
            }
            let v = self.regs.entry(id).or_insert_with(|| vec![0; data.len()]);
            data.copy_from_slice(v);
            Ok(())
        }

        fn set_one_reg(&mut self, id: u64, data: &[u8]) -> io::Result<()> {
            assert_eq!(data.len(), reg_size(id), "set of {id:#x}");
            self.log.push(format!("set {id:#x}"));
            if self.absent.contains(&id) {
                return Err(io::Error::from_raw_os_error(ENOENT));
            }
            if let Some(&e) = self.refuse.get(&id) {
                return Err(io::Error::from_raw_os_error(e));
            }
            self.regs.insert(id, data.to_vec());
            Ok(())
        }

        fn get_mp_state(&mut self) -> io::Result<u32> {
            self.log.push("get_mp_state".to_string());
            Ok(self.mp_state)
        }

        fn set_mp_state(&mut self, state: u32) -> io::Result<()> {
            self.log.push(format!("set_mp_state {state}"));
            self.mp_state = state;
            Ok(())
        }

        fn get_vcpu_events(&mut self) -> io::Result<SErrorEvents> {
            self.log.push("get_events".to_string());
            Ok(self.events)
        }

        fn set_vcpu_events(&mut self, events: &SErrorEvents) -> io::Result<()> {
            self.log.push(format!("set_events {events:?}"));
            self.events = *events;
            Ok(())
        }

        fn has_device_attr(&mut self, group: u32, attr: u64, value: Option<i32>) -> io::Result<()> {
            self.log.push(format!("has_attr {group} {attr} {value:?}"));
            if self.no_pmu_attr {
                return Err(io::Error::from_raw_os_error(6));
            }
            Ok(())
        }

        fn set_device_attr(&mut self, group: u32, attr: u64, value: Option<i32>) -> io::Result<()> {
            self.log.push(format!("set_attr {group} {attr} {value:?}"));
            Ok(())
        }
    }

    // Some ids from a real kernel's KVM_GET_REG_LIST.
    const SCTLR_EL1: u64 = sys_reg(3, 0, 1, 0, 0);
    const TTBR0_EL1: u64 = sys_reg(3, 0, 2, 0, 0);
    const CCSIDR0: u64 = KVM_REG_ARM64 | KVM_REG_SIZE_U32 | KVM_REG_ARM_DEMUX;
    const X0: u64 = core_reg(0, KVM_REG_SIZE_U64);

    #[test]
    fn ids_match_the_kernel_headers() {
        assert_eq!(MPIDR_EL1, 0x6030_0000_0013_c005);
        assert_eq!(KVM_REG_ARM_TIMER_CNT, 0x6030_0000_0013_df1a);
        assert_eq!(KVM_REG_ARM_TIMER_CVAL, 0x6030_0000_0013_df02);
        assert_eq!(KVM_REG_ARM_TIMER_CTL, 0x6030_0000_0013_df19);
        assert_eq!(KVM_REG_ARM_PTIMER_CNT, 0x6030_0000_0013_df01);
        assert_eq!(KVM_REG_ARM_PSCI_VERSION, 0x6030_0000_0014_0000);
        assert_eq!(fw_feat_bmap_reg(3), 0x6030_0000_0016_0003);
        // X0, PC and the vregs as QEMU's AARCH64_CORE_REG() and AARCH64_SIMD_CORE_REG() give.
        assert_eq!(X0, 0x6030_0000_0010_0000);
        assert_eq!(core_reg(core_offset::PC, KVM_REG_SIZE_U64), 0x6030_0000_0010_0040);
        assert_eq!(core_reg(core_offset::PSTATE, KVM_REG_SIZE_U64), 0x6030_0000_0010_0042);
        assert_eq!(core_reg(core_offset::VREGS, KVM_REG_SIZE_U128), 0x6040_0000_0010_0054);
        assert_eq!(core_reg(core_offset::FPSR, KVM_REG_SIZE_U32), 0x6020_0000_0010_00d4);
        assert_eq!(core_reg(core_offset::FPCR, KVM_REG_SIZE_U32), 0x6020_0000_0010_00d5);
        assert_eq!(sve_zreg(31, 0), 0x6080_0000_0015_03e0);
        assert_eq!(sve_preg(15, 0), 0x6050_0000_0015_05e0);
        assert_eq!(sve_ffr(0), 0x6050_0000_0015_0600);
        assert_eq!(KVM_REG_ARM64_SVE_VLS, 0x6060_0000_0015_ffff);
        assert_eq!(reg_size(sve_zreg(0, 0)), 256);
        assert_eq!(reg_size(sve_preg(0, 0)), 32);
        assert_eq!(reg_size(KVM_REG_ARM64_SVE_VLS), 64);
        assert_eq!(reg_size(CCSIDR0), 4);
    }

    #[test]
    fn register_names() {
        assert_eq!(register_name(X0), "core reg 6030000000100000");
        assert_eq!(register_name(CCSIDR0), "demuxed reg 6020000000110000");
        assert_eq!(
            register_name(KVM_REG_ARM_TIMER_CNT),
            "system register op0:3 op1:3 crn:14 crm:3 op2:2"
        );
        assert_eq!(register_name(KVM_REG_ARM_PSCI_VERSION), "fw reg 0");
        assert_eq!(register_name(fw_feat_bmap_reg(2)), "fw feat reg 2");
        assert_eq!(register_name(KVM_REG_ARM64_SVE_VLS), "SVE VLS");
        assert_eq!(register_name(sve_zreg(3, 1)), "SVE zreg n:3 slice:1");
        assert_eq!(register_name(sve_preg(15, 0)), "SVE preg n:15 slice:0");
        assert_eq!(register_name(sve_ffr(2)), "SVE ffr slice:2");
        assert_eq!(register_name(0x6030_0000_0099_0001), "6030000000990001");
    }

    #[test]
    fn psci_versions() {
        assert_eq!(parse_psci_version("0.2"), Ok(PSCI_VERSION_0_2));
        assert_eq!(parse_psci_version("1.1"), Ok(0x10001));
        assert_eq!(parse_psci_version(" 1. 3x"), Ok(0x10003));
        assert_eq!(parse_psci_version("1"), Err("Invalid PSCI version.".to_string()));
        assert_eq!(parse_psci_version("a.b"), Err("Invalid PSCI version.".to_string()));
        assert_eq!(parse_psci_version("70000.0"), Err("Invalid PSCI version.".to_string()));
        assert_eq!(format_psci_version(0x10002), "1.2");
    }

    #[test]
    fn cpreg_list_from_kernel() {
        let list = CpregList::from_kernel(vec![TTBR0_EL1, X0, SCTLR_EL1, CCSIDR0, sve_zreg(0, 0)])
            .unwrap();
        // Sorted, core and SVE registers dropped.
        assert_eq!(list.indexes(), &[CCSIDR0, SCTLR_EL1, TTBR0_EL1]);
        assert_eq!(list.values(), &[0, 0, 0]);
        let wide = KVM_REG_ARM64 | KVM_REG_SIZE_U128 | KVM_REG_ARM64_SYSREG;
        let err = CpregList::from_kernel(vec![SCTLR_EL1, wide]).unwrap_err();
        assert_eq!(err.to_string(), "Can't handle size of register in kernel list");
    }

    #[test]
    fn cpreg_list_read_and_write() {
        let mut fake = Fake::with_list(&[SCTLR_EL1, CCSIDR0, KVM_REG_ARM_TIMER_CNT]);
        fake.set64(SCTLR_EL1, 0x30d0_0980);
        fake.regs.insert(CCSIDR0, 0x7012_e01au32.to_ne_bytes().to_vec());
        fake.set64(KVM_REG_ARM_TIMER_CNT, 1234);
        let mut list = CpregList::from_kernel(fake.list.clone()).unwrap();
        list.read(&mut fake).unwrap();
        assert_eq!(list.get(SCTLR_EL1), Some(0x30d0_0980));
        assert_eq!(list.get(CCSIDR0), Some(0x7012_e01a));
        assert_eq!(list.get(KVM_REG_ARM_TIMER_CNT), Some(1234));
        assert_eq!(list.get(TTBR0_EL1), None);

        // The counter is only written with the full level.
        fake.log.clear();
        list.write(&mut fake, PutLevel::Runtime).unwrap();
        assert_eq!(fake.sets().len(), 2);
        assert!(!fake.log.contains(&format!("set {KVM_REG_ARM_TIMER_CNT:#x}")));
        fake.log.clear();
        list.write(&mut fake, PutLevel::Full).unwrap();
        assert_eq!(fake.sets().len(), 3);
    }

    #[test]
    fn cpreg_read_keeps_going_after_a_failure() {
        let mut fake = Fake::with_list(&[SCTLR_EL1, TTBR0_EL1]);
        fake.set64(TTBR0_EL1, 7);
        fake.unreadable.insert(SCTLR_EL1);
        let mut list = CpregList::from_kernel(fake.list.clone()).unwrap();
        let err = list.read(&mut fake).unwrap_err();
        assert!(matches!(err, ArmVcpuError::OneReg { id: SCTLR_EL1, write: false, .. }));
        assert_eq!(list.get(TTBR0_EL1), Some(7));
    }

    #[test]
    fn cpreg_write_errors_match_qemu() {
        let mut fake = Fake::with_list(&[CCSIDR0, SCTLR_EL1, TTBR0_EL1, KVM_REG_ARM_PSCI_VERSION]);
        let mut list = CpregList::from_kernel(fake.list.clone()).unwrap();
        *list.get_mut(SCTLR_EL1).unwrap() = 0xabc;
        *list.get_mut(CCSIDR0).unwrap() = 0x12;
        fake.set64(SCTLR_EL1, 0xc50838);
        fake.regs.insert(CCSIDR0, 0x34u32.to_ne_bytes().to_vec());
        fake.refuse.insert(SCTLR_EL1, EINVAL);
        fake.refuse.insert(CCSIDR0, EINVAL);
        fake.absent.insert(TTBR0_EL1);
        fake.refuse.insert(KVM_REG_ARM_PSCI_VERSION, 16);
        let ArmVcpuError::SetList(lines) = list.write(&mut fake, PutLevel::Full).unwrap_err()
        else {
            panic!("wrong error");
        };
        // The text of EBUSY differs between hosts.
        let busy = strerror(&io::Error::from_raw_os_error(16));
        assert_eq!(
            lines,
            [
                "Could not set register demuxed reg 6020000000110000 to 12 (is 34)".to_string(),
                "Could not set register system register op0:3 op1:0 crn:1 crm:0 op2:0 to abc \
                 (is c50838)"
                    .to_string(),
                "Could not set register system register op0:3 op1:0 crn:2 crm:0 op2:0: unknown \
                 to KVM"
                    .to_string(),
                format!("Could not set register fw reg 0: {busy}"),
            ]
        );
    }

    #[test]
    fn merge_follows_cpu_post_load() {
        let mut list = CpregList::from_kernel(vec![CCSIDR0, SCTLR_EL1, TTBR0_EL1]).unwrap();
        *list.get_mut(TTBR0_EL1).unwrap() = 99;
        // The stream lacks TTBR0_EL1: it keeps its value, with a warning.
        let warnings = list.merge_incoming(&[CCSIDR0, SCTLR_EL1], &[5, 6], &[]).unwrap();
        assert_eq!(list.values(), &[5, 6, 99]);
        assert_eq!(
            warnings,
            ["handle_cpreg_missing_in_incoming_stream: system register op0:3 op1:0 crn:2 crm:0 \
              op2:0 expected by the destination but not in the incoming stream: skip it"]
        );

        // A register only the stream has fails, wherever it sits in the order.
        let tcr2 = sys_reg(3, 0, 2, 0, 3);
        for (ids, vals) in [
            (vec![CCSIDR0, SCTLR_EL1, TTBR0_EL1, tcr2], vec![1, 2, 3, 4]),
            (vec![CCSIDR0, SCTLR_EL1, TTBR0_EL1, sys_reg(3, 0, 10, 2, 0)], vec![1, 2, 3, 4]),
        ] {
            let err = list.clone().merge_incoming(&ids, &vals, &[]).unwrap_err();
            let msg = err.to_string();
            assert!(msg.starts_with("handle_cpreg_only_in_incoming_stream: "), "{msg}");
            assert!(msg.ends_with("unknown on the destination: fail migration"), "{msg}");
        }

        // The KVM tolerances let TCR2_EL1 come and go silently.
        let ids = [CCSIDR0, SCTLR_EL1, TTBR0_EL1, tcr2];
        let warnings = list.merge_incoming(&ids, &[1, 2, 3, 4], KVM_CPREG_TOLERANCES).unwrap();
        assert!(warnings.is_empty());
        assert_eq!(list.values(), &[1, 2, 3]);
        let mut with_tcr2 = CpregList::from_kernel(vec![SCTLR_EL1, tcr2]).unwrap();
        let warnings = with_tcr2.merge_incoming(&[SCTLR_EL1], &[1], KVM_CPREG_TOLERANCES);
        assert!(warnings.unwrap().is_empty());

        // VENDOR_HYP_BMAP_2 may only be dropped while it is 0.
        let bmap2 = fw_feat_bmap_reg(3);
        assert!(list.clone().merge_incoming(&[bmap2], &[0], KVM_CPREG_TOLERANCES).is_ok());
        assert!(list.clone().merge_incoming(&[bmap2], &[1], KVM_CPREG_TOLERANCES).is_err());

        assert!(matches!(
            list.merge_incoming(&[CCSIDR0], &[], &[]),
            Err(ArmVcpuError::MigrationLength)
        ));
    }

    #[test]
    fn features_follow_kvm_arch_init_vcpu() {
        let caps = ArmKvmCaps { psci_0_2: true, pmu_v3: true, el2: false, ..ArmKvmCaps::default() };
        let mut cfg = ArmVcpuConfig::host(5, &caps);
        assert_eq!(init_features(&cfg, &caps)[0], (1 << 2) | (1 << 3));
        cfg.start_powered_off = true;
        cfg.psci_version = PSCI_VERSION_0_1;
        cfg.aarch64 = false;
        cfg.pmu = false;
        cfg.sve_vq_map = Some(0b1011);
        cfg.pauth = true;
        cfg.el2 = true;
        // No PSCI 0.2 for 0.1, and no EL2 without the capability.
        assert_eq!(init_features(&cfg, &caps)[0], 1 | (1 << 1) | (1 << 4) | (1 << 5) | (1 << 6));
        let caps = ArmKvmCaps { el2: true, ..caps };
        assert_eq!(init_features(&cfg, &caps)[0] & (1 << 7), 1 << 7);
        let caps = ArmKvmCaps { psci_0_2: false, ..caps };
        cfg.psci_version = 0;
        assert_eq!(init_features(&cfg, &caps)[0] & (1 << 2), 0);
    }

    fn caps() -> ArmKvmCaps {
        ArmKvmCaps {
            psci_0_2: true,
            pmu_v3: true,
            mp_state: true,
            vcpu_events: true,
            ..ArmKvmCaps::default()
        }
    }

    #[test]
    fn init_sequence() {
        let mut fake = Fake::with_list(&[X0, SCTLR_EL1, KVM_REG_ARM_TIMER_CNT, sve_zreg(0, 0)]);
        // Aff3 1, the U bit, Aff2 1, Aff1 2, Aff0 3.
        fake.set64(MPIDR_EL1, 0x1_c001_0203);
        fake.set64(KVM_REG_ARM_PSCI_VERSION, 0x10001);
        fake.set64(SCTLR_EL1, 0x30d0_0980);
        let mut cfg = ArmVcpuConfig::host(5, &caps());
        cfg.start_powered_off = true;
        cfg.sve_vq_map = Some(0xf);
        let vcpu = ArmVcpu::init(&mut fake, &cfg, &caps()).unwrap();
        assert_eq!(
            fake.log[..6],
            [
                "init 5 0x1d".to_string(),
                format!("set {KVM_REG_ARM64_SVE_VLS:#x}"),
                "finalize 4".to_string(),
                format!("get {KVM_REG_ARM_PSCI_VERSION:#x}"),
                format!("get {MPIDR_EL1:#x}"),
                "reg_list".to_string(),
            ]
        );
        assert_eq!(fake.regs[&KVM_REG_ARM64_SVE_VLS][..8], 0xfu64.to_ne_bytes());
        assert_eq!(vcpu.psci_version, 0x10001);
        assert_eq!(vcpu.mp_affinity, 0x1_0001_0203);
        assert_eq!(vcpu.cpregs.indexes(), &[SCTLR_EL1, KVM_REG_ARM_TIMER_CNT]);
        assert_eq!(vcpu.cpregs.get(SCTLR_EL1), Some(0x30d0_0980));
        assert!(vcpu.powered_off);
        assert!(matches!(vcpu.regs.vregs, VectorRegs::Sve(_)));
    }

    #[test]
    fn init_with_a_psci_version() {
        let mut fake = Fake::with_list(&[SCTLR_EL1]);
        let mut cfg = ArmVcpuConfig::host(5, &caps());
        cfg.psci_version = 0x10000;
        let vcpu = ArmVcpu::init(&mut fake, &cfg, &caps()).unwrap();
        assert_eq!(fake.get64(KVM_REG_ARM_PSCI_VERSION), 0x10000);
        assert_eq!(vcpu.psci_version, 0x10000);

        let mut fake = Fake::with_list(&[SCTLR_EL1]);
        fake.refuse.insert(KVM_REG_ARM_PSCI_VERSION, EINVAL);
        cfg.psci_version = 0x10003;
        let err = ArmVcpu::init(&mut fake, &cfg, &caps()).unwrap_err();
        assert_eq!(
            err.to_string(),
            "KVM in this kernel does not support PSCI version 1.3\nConsider setting the \
             kvm-psci-version property on the migration source."
        );

        // A kernel without the register keeps the configured version, 0 here.
        let mut fake = Fake::with_list(&[SCTLR_EL1]);
        fake.absent.insert(KVM_REG_ARM_PSCI_VERSION);
        let vcpu = ArmVcpu::init(&mut fake, &ArmVcpuConfig::host(5, &caps()), &caps()).unwrap();
        assert_eq!(vcpu.psci_version, 0);

        let mut fake = Fake::with_list(&[SCTLR_EL1]);
        fake.unreadable.insert(SCTLR_EL1);
        let err = ArmVcpu::init(&mut fake, &ArmVcpuConfig::host(5, &caps()), &caps());
        assert_eq!(err.unwrap_err().to_string(), "Initial read of kernel register state failed");
    }

    #[test]
    fn put_and_get_core_registers_in_qemu_order() {
        let mut fake = Fake::default();
        let mut regs = CoreRegs::default();
        for (n, x) in regs.x.iter_mut().enumerate() {
            *x = 0x1000 + n as u64;
        }
        regs.sp_el0 = 0x10;
        regs.sp_el1 = 0x20;
        regs.pstate = 0x3c5;
        regs.pc = 0x4000_0000;
        regs.elr_el1 = 0x30;
        regs.spsr = [1, 2, 3, 4, 5];
        let VectorRegs::Fpsimd(v) = &mut regs.vregs else { unreachable!() };
        v[31] = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210;
        regs.fpsr = 0x0800_0000;
        regs.fpcr = 0x0300_0000;
        regs.put(&mut fake).unwrap();

        let ids: Vec<u64> = fake
            .log
            .iter()
            .map(|l| u64::from_str_radix(l.trim_start_matches("set 0x"), 16).unwrap())
            .collect();
        assert_eq!(ids.len(), 31 + 5 + 5 + 32 + 2);
        assert_eq!(ids[0], X0);
        assert_eq!(ids[30], core_reg(240, KVM_REG_SIZE_U64));
        assert_eq!(
            &ids[31..36],
            &[
                core_reg(core_offset::SP, KVM_REG_SIZE_U64),
                core_reg(core_offset::SP_EL1, KVM_REG_SIZE_U64),
                core_reg(core_offset::PSTATE, KVM_REG_SIZE_U64),
                core_reg(core_offset::PC, KVM_REG_SIZE_U64),
                core_reg(core_offset::ELR_EL1, KVM_REG_SIZE_U64),
            ]
        );
        assert_eq!(ids[36], core_reg(core_offset::SPSR, KVM_REG_SIZE_U64));
        assert_eq!(ids[41], core_reg(core_offset::VREGS, KVM_REG_SIZE_U128));
        assert_eq!(
            ids[73..],
            [
                core_reg(core_offset::FPSR, KVM_REG_SIZE_U32),
                core_reg(core_offset::FPCR, KVM_REG_SIZE_U32),
            ]
        );
        assert_eq!(fake.get64(core_reg(core_offset::PC, KVM_REG_SIZE_U64)), 0x4000_0000);

        let mut back = CoreRegs::default();
        back.get(&mut fake).unwrap();
        assert_eq!(back, regs);
    }

    #[test]
    fn sve_registers_round_trip() {
        let mut fake = Fake::default();
        let mut regs = CoreRegs::with_sve();
        let VectorRegs::Sve(sve) = &mut regs.vregs else { unreachable!() };
        sve.z[5][31] = 0xdead;
        sve.p[15][3] = 0xbeef;
        sve.ffr[0] = 0xffff;
        regs.put(&mut fake).unwrap();
        assert!(fake.log.contains(&format!("set {:#x}", sve_ffr(0))));
        let v0 = core_reg(core_offset::VREGS, KVM_REG_SIZE_U128);
        assert!(!fake.log.contains(&format!("set {v0:#x}")));
        let mut back = CoreRegs::with_sve();
        back.get(&mut fake).unwrap();
        assert_eq!(back, regs);
    }

    #[test]
    fn put_and_get_registers_order() {
        let mut fake = Fake::with_list(&[SCTLR_EL1]);
        let caps = ArmKvmCaps { inject_serror_esr: false, ..caps() };
        let mut vcpu = ArmVcpu::init(&mut fake, &ArmVcpuConfig::host(5, &caps), &caps).unwrap();
        vcpu.serror = SErrorEvents { pending: true, has_esr: true, esr: 0x1234 };
        vcpu.powered_off = true;
        fake.log.clear();
        vcpu.put_registers(&mut fake, PutLevel::Runtime).unwrap();
        let tail: Vec<&str> = fake.log.iter().rev().take(3).rev().map(String::as_str).collect();
        // The list, then the SError without its syndrome, then the power state.
        assert_eq!(tail[0], format!("set {SCTLR_EL1:#x}"));
        assert_eq!(tail[1], "set_events SErrorEvents { pending: true, has_esr: false, esr: 0 }");
        assert_eq!(tail[2], "set_mp_state 5");

        fake.mp_state = KVM_MP_STATE_RUNNABLE;
        fake.events = SErrorEvents { pending: false, has_esr: true, esr: 7 };
        fake.log.clear();
        vcpu.get_registers(&mut fake).unwrap();
        let tail: Vec<&str> = fake.log.iter().rev().take(3).rev().map(String::as_str).collect();
        assert_eq!(tail, ["get_events", &format!("get {SCTLR_EL1:#x}"), "get_mp_state"]);
        assert!(!vcpu.powered_off);
        assert_eq!(vcpu.serror, fake.events);

        // Without the capabilities neither ioctl is used.
        vcpu.caps = ArmKvmCaps::default();
        fake.log.clear();
        vcpu.put_registers(&mut fake, PutLevel::Runtime).unwrap();
        vcpu.get_registers(&mut fake).unwrap();
        assert!(!fake.log.iter().any(|l| l.contains("events") || l.contains("mp_state")));
    }

    #[test]
    fn reset_reinits_and_reads_back() {
        let mut fake = Fake::with_list(&[SCTLR_EL1]);
        let mut cfg = ArmVcpuConfig::host(5, &caps());
        cfg.start_powered_off = true;
        let mut vcpu = ArmVcpu::init(&mut fake, &cfg, &caps()).unwrap();
        vcpu.powered_off = false;
        vcpu.serror.pending = true;
        fake.set64(SCTLR_EL1, 0xc50838);
        fake.set64(core_reg(core_offset::PSTATE, KVM_REG_SIZE_U64), 0x3c5);
        fake.log.clear();
        vcpu.reset(&mut fake).unwrap();
        assert_eq!(fake.log[0], "init 5 0xd");
        assert_eq!(vcpu.cpregs.get(SCTLR_EL1), Some(0xc50838));
        assert_eq!(vcpu.regs.pstate, 0x3c5);
        assert!(vcpu.powered_off);
        assert!(!vcpu.serror.pending);
    }

    #[test]
    fn pmu_attributes() {
        let mut fake = Fake::with_list(&[]);
        let vcpu = ArmVcpu::init(&mut fake, &ArmVcpuConfig::host(5, &caps()), &caps()).unwrap();
        fake.log.clear();
        vcpu.pmu_set_irq(&mut fake, 23).unwrap();
        vcpu.pmu_init(&mut fake).unwrap();
        assert_eq!(
            fake.log,
            [
                "has_attr 0 0 Some(23)",
                "set_attr 0 0 Some(23)",
                "has_attr 0 1 None",
                "set_attr 0 1 None"
            ]
        );
        fake.no_pmu_attr = true;
        let err = vcpu.pmu_init(&mut fake).unwrap_err();
        let nxio = strerror(&io::Error::from_raw_os_error(6));
        assert_eq!(
            err.to_string(),
            format!("PMU: KVM_HAS_DEVICE_ATTR: {nxio}\nfailed to init PMU")
        );

        let no_pmu = ArmVcpu { pmu: false, ..vcpu };
        fake.log.clear();
        no_pmu.pmu_init(&mut fake).unwrap();
        assert!(fake.log.is_empty());
    }

    #[test]
    fn virtual_time_and_migration() {
        let mut fake = Fake::with_list(&[SCTLR_EL1, KVM_REG_ARM_TIMER_CNT]);
        let mut vcpu = ArmVcpu::init(&mut fake, &ArmVcpuConfig::host(5, &caps()), &caps()).unwrap();

        // Stop: the counter is held. A second stop keeps the first value.
        fake.set64(KVM_REG_ARM_TIMER_CNT, 1000);
        vcpu.vm_state_change(&mut fake, false).unwrap();
        fake.set64(KVM_REG_ARM_TIMER_CNT, 1500);
        vcpu.vm_state_change(&mut fake, false).unwrap();
        assert_eq!(vcpu.vtime, VirtualTime { enabled: true, vtime: 1000, dirty: true });

        // A save while stopped sends the held value, not the running one.
        let (ids, vals) = vcpu.pre_save(&mut fake).unwrap();
        assert_eq!(ids, [SCTLR_EL1, KVM_REG_ARM_TIMER_CNT]);
        assert_eq!(vals[1], 1000);

        // Run: the held value goes back.
        fake.set64(KVM_REG_ARM_TIMER_CNT, 9999);
        vcpu.vm_state_change(&mut fake, true).unwrap();
        assert_eq!(fake.get64(KVM_REG_ARM_TIMER_CNT), 1000);
        assert!(!vcpu.vtime.dirty);

        // A load writes the full list, counter included, and holds the counter.
        fake.log.clear();
        let warnings =
            vcpu.post_load(&mut fake, &ids, &[0x30d0_0980, 4242], KVM_CPREG_TOLERANCES).unwrap();
        assert!(warnings.is_empty());
        assert_eq!(fake.get64(SCTLR_EL1), 0x30d0_0980);
        assert_eq!(fake.get64(KVM_REG_ARM_TIMER_CNT), 4242);
        assert_eq!(vcpu.vtime, VirtualTime { enabled: true, vtime: 4242, dirty: true });

        // With kvm-no-adjvtime nothing is held.
        vcpu.vtime = VirtualTime::default();
        fake.log.clear();
        vcpu.vm_state_change(&mut fake, false).unwrap();
        assert!(fake.log.is_empty());

        fake.unreadable.insert(KVM_REG_ARM_TIMER_CNT);
        vcpu.vtime.enabled = true;
        let err = vcpu.vm_state_change(&mut fake, false).unwrap_err();
        assert_eq!(err.to_string(), "Failed to get KVM_REG_ARM_TIMER_CNT");
    }

    #[test]
    fn vcpu_count() {
        assert!(check_vcpu_count(256, false).is_ok());
        assert!(check_vcpu_count(512, true).is_ok());
        assert_eq!(
            check_vcpu_count(257, false).unwrap_err().to_string(),
            "Using more than 256 vcpus requires a host kernel with KVM_CAP_ARM_IRQ_LINE_LAYOUT_2"
        );
    }
}
