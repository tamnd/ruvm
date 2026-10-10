// SPDX-License-Identifier: GPL-2.0-or-later

//! The Arm side of KVM that needs no host: the in-kernel vGIC from hw/intc/arm_gicv3_kvm.c,
//! hw/intc/arm_gic_kvm.c and hw/intc/arm_gicv3_its_kvm.c, the interrupt line numbers and the
//! irqchip choice from target/arm/kvm.c, and the GIC version choice from hw/arm/virt.c.
//!
//! The kernel exposes the vGIC as a device whose state is read and written one attribute at a
//! time. Every sequence here, the setup after the device is created, the save and the restore of
//! a GICv3, a GICv2 and an ITS, is written against [`VgicDevice`], so the order of the accesses
//! and the encoding of each attribute can be checked with a fake device on any host. The Linux
//! code in `linux/vgic.rs` runs the same sequences against the real device file descriptors.
//!
//! The state structs hold what QEMU's `GICv3State`, `GICState` and `GICv3ITSState` save for
//! migration and nothing of the TCG model, which lives in ruvm-hw-intc. That crate sits above
//! this one in the layering, so the state is described here and a machine that runs a vGIC
//! keeps it next to the device.
//!
//! The vCPU setup and register sync from target/arm/kvm.c are in [`vcpu`].

pub mod vcpu;

use std::fmt;
use std::io;

use crate::{KernelIrqchip, KvmError, strerror};

/// `GIC_INTERNAL`: the SGIs and PPIs, which are banked per CPU.
pub const GIC_INTERNAL: u32 = 32;
/// `GIC_NR_SGIS`.
pub const GIC_NR_SGIS: u32 = 16;
/// `GIC_MAXIRQ`, the most interrupts a GICv2 has.
pub const GIC_MAXIRQ: u32 = 1020;
/// `GIC_NCPU`, the most CPU interfaces a GICv2 has.
pub const GIC_NCPU: u32 = 8;
/// `ALL_CPU_MASK` in hw/intc/gic_internal.h.
const ALL_CPU_MASK: u8 = 0xff;

/// `KVM_DEV_TYPE_ARM_VGIC_V2`.
pub const KVM_DEV_TYPE_ARM_VGIC_V2: u32 = 5;
/// `KVM_DEV_TYPE_ARM_VGIC_V3`.
pub const KVM_DEV_TYPE_ARM_VGIC_V3: u32 = 7;
/// `KVM_DEV_TYPE_ARM_VGIC_ITS`.
pub const KVM_DEV_TYPE_ARM_VGIC_ITS: u32 = 8;
/// `KVM_CREATE_DEVICE_TEST`: ask whether the device could be created without creating it.
pub const KVM_CREATE_DEVICE_TEST: u32 = 1;

/// `KVM_DEV_ARM_VGIC_GRP_ADDR`: where the guest sees the frames. 64-bit values.
pub const KVM_DEV_ARM_VGIC_GRP_ADDR: u32 = 0;
/// `KVM_DEV_ARM_VGIC_GRP_DIST_REGS`: distributor registers. 32-bit values.
pub const KVM_DEV_ARM_VGIC_GRP_DIST_REGS: u32 = 1;
/// `KVM_DEV_ARM_VGIC_GRP_CPU_REGS`: GICv2 CPU interface registers. 32-bit values.
pub const KVM_DEV_ARM_VGIC_GRP_CPU_REGS: u32 = 2;
/// `KVM_DEV_ARM_VGIC_GRP_NR_IRQS`. A 32-bit value.
pub const KVM_DEV_ARM_VGIC_GRP_NR_IRQS: u32 = 3;
/// `KVM_DEV_ARM_VGIC_GRP_CTRL`: commands, with no value.
pub const KVM_DEV_ARM_VGIC_GRP_CTRL: u32 = 4;
/// `KVM_DEV_ARM_VGIC_GRP_REDIST_REGS`: GICv3 redistributor registers. 32-bit values.
pub const KVM_DEV_ARM_VGIC_GRP_REDIST_REGS: u32 = 5;
/// `KVM_DEV_ARM_VGIC_GRP_CPU_SYSREGS`: GICv3 CPU interface system registers. 64-bit values.
pub const KVM_DEV_ARM_VGIC_GRP_CPU_SYSREGS: u32 = 6;
/// `KVM_DEV_ARM_VGIC_GRP_LEVEL_INFO`: the line level of level triggered interrupts.
pub const KVM_DEV_ARM_VGIC_GRP_LEVEL_INFO: u32 = 7;
/// `KVM_DEV_ARM_VGIC_GRP_ITS_REGS`: ITS registers. 64-bit values.
pub const KVM_DEV_ARM_VGIC_GRP_ITS_REGS: u32 = 8;
/// `KVM_DEV_ARM_VGIC_GRP_MAINT_IRQ`: the maintenance interrupt for nested virtualization.
pub const KVM_DEV_ARM_VGIC_GRP_MAINT_IRQ: u32 = 9;

/// `KVM_DEV_ARM_VGIC_CTRL_INIT`.
pub const KVM_DEV_ARM_VGIC_CTRL_INIT: u64 = 0;
/// `KVM_DEV_ARM_ITS_SAVE_TABLES`.
pub const KVM_DEV_ARM_ITS_SAVE_TABLES: u64 = 1;
/// `KVM_DEV_ARM_ITS_RESTORE_TABLES`.
pub const KVM_DEV_ARM_ITS_RESTORE_TABLES: u64 = 2;
/// `KVM_DEV_ARM_VGIC_SAVE_PENDING_TABLES`.
pub const KVM_DEV_ARM_VGIC_SAVE_PENDING_TABLES: u64 = 3;
/// `KVM_DEV_ARM_ITS_CTRL_RESET`.
pub const KVM_DEV_ARM_ITS_CTRL_RESET: u64 = 4;

/// `KVM_VGIC_V2_ADDR_TYPE_DIST`.
pub const KVM_VGIC_V2_ADDR_TYPE_DIST: u64 = 0;
/// `KVM_VGIC_V2_ADDR_TYPE_CPU`.
pub const KVM_VGIC_V2_ADDR_TYPE_CPU: u64 = 1;
/// `KVM_VGIC_V3_ADDR_TYPE_DIST`.
pub const KVM_VGIC_V3_ADDR_TYPE_DIST: u64 = 2;
/// `KVM_VGIC_V3_ADDR_TYPE_REDIST`.
pub const KVM_VGIC_V3_ADDR_TYPE_REDIST: u64 = 3;
/// `KVM_VGIC_ITS_ADDR_TYPE`.
pub const KVM_VGIC_ITS_ADDR_TYPE: u64 = 4;
/// `KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION`.
pub const KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION: u64 = 5;

/// `KVM_DEV_ARM_VGIC_CPUID_SHIFT`.
pub const KVM_DEV_ARM_VGIC_CPUID_SHIFT: u32 = 32;
/// `KVM_DEV_ARM_VGIC_CPUID_MASK`.
pub const KVM_DEV_ARM_VGIC_CPUID_MASK: u64 = 0xff << KVM_DEV_ARM_VGIC_CPUID_SHIFT;
/// `KVM_DEV_ARM_VGIC_V3_MPIDR_MASK`.
pub const KVM_DEV_ARM_VGIC_V3_MPIDR_MASK: u64 = 0xffff_ffff << 32;
/// `KVM_DEV_ARM_VGIC_OFFSET_MASK`.
pub const KVM_DEV_ARM_VGIC_OFFSET_MASK: u64 = 0xffff_ffff;
/// `KVM_DEV_ARM_VGIC_LINE_LEVEL_INFO_SHIFT`.
pub const KVM_DEV_ARM_VGIC_LINE_LEVEL_INFO_SHIFT: u32 = 10;
/// `VGIC_LEVEL_INFO_LINE_LEVEL`.
pub const VGIC_LEVEL_INFO_LINE_LEVEL: u64 = 0;

/// `KVM_ARM_IRQ_VCPU2_SHIFT`.
pub const KVM_ARM_IRQ_VCPU2_SHIFT: u32 = 28;
/// `KVM_ARM_IRQ_TYPE_SHIFT`.
pub const KVM_ARM_IRQ_TYPE_SHIFT: u32 = 24;
/// `KVM_ARM_IRQ_VCPU_SHIFT`.
pub const KVM_ARM_IRQ_VCPU_SHIFT: u32 = 16;
/// `KVM_ARM_IRQ_TYPE_CPU`: the IRQ or FIQ line of a vCPU, for a userspace GIC.
pub const KVM_ARM_IRQ_TYPE_CPU: u32 = 0;
/// `KVM_ARM_IRQ_TYPE_SPI`.
pub const KVM_ARM_IRQ_TYPE_SPI: u32 = 1;
/// `KVM_ARM_IRQ_TYPE_PPI`.
pub const KVM_ARM_IRQ_TYPE_PPI: u32 = 2;
/// `KVM_ARM_IRQ_CPU_IRQ`.
pub const KVM_ARM_IRQ_CPU_IRQ: u32 = 0;
/// `KVM_ARM_IRQ_CPU_FIQ`.
pub const KVM_ARM_IRQ_CPU_FIQ: u32 = 1;

/// `KVM_ARM_DEV_EL1_VTIMER` in `kvm_run.s.regs.device_irq_level`.
pub const KVM_ARM_DEV_EL1_VTIMER: u64 = 1 << 0;
/// `KVM_ARM_DEV_EL1_PTIMER`.
pub const KVM_ARM_DEV_EL1_PTIMER: u64 = 1 << 1;
/// `KVM_ARM_DEV_PMU`.
pub const KVM_ARM_DEV_PMU: u64 = 1 << 2;

/// `KVM_MSI_VALID_DEVID`.
pub const KVM_MSI_VALID_DEVID: u32 = 1;
/// `KVM_CAP_DEVICE_CTRL`.
pub const KVM_CAP_DEVICE_CTRL: u32 = 89;

/// `KVM_ARM_VGIC_V2` in the `vgic_probe` bitmap.
pub const KVM_ARM_VGIC_V2: u32 = 1 << 0;
/// `KVM_ARM_VGIC_V3` in the probe bitmap.
pub const KVM_ARM_VGIC_V3: u32 = 1 << 1;

/// Distributor register offsets.
pub const GICD_CTLR: u32 = 0x0000;
/// `GICD_TYPER`.
pub const GICD_TYPER: u32 = 0x0004;
/// `GICD_IIDR`.
pub const GICD_IIDR: u32 = 0x0008;
/// `GICD_STATUSR`.
pub const GICD_STATUSR: u32 = 0x0010;
/// `GICD_IGROUPR`.
pub const GICD_IGROUPR: u32 = 0x0080;
/// `GICD_ISENABLER`.
pub const GICD_ISENABLER: u32 = 0x0100;
/// `GICD_ICENABLER`.
pub const GICD_ICENABLER: u32 = 0x0180;
/// `GICD_ISPENDR`.
pub const GICD_ISPENDR: u32 = 0x0200;
/// `GICD_ICPENDR`.
pub const GICD_ICPENDR: u32 = 0x0280;
/// `GICD_ISACTIVER`.
pub const GICD_ISACTIVER: u32 = 0x0300;
/// `GICD_ICACTIVER`.
pub const GICD_ICACTIVER: u32 = 0x0380;
/// `GICD_IPRIORITYR`.
pub const GICD_IPRIORITYR: u32 = 0x0400;
/// `GICD_ITARGETSR`.
pub const GICD_ITARGETSR: u32 = 0x0800;
/// `GICD_ICFGR`.
pub const GICD_ICFGR: u32 = 0x0c00;
/// `GICD_CPENDSGIR`.
pub const GICD_CPENDSGIR: u32 = 0x0f10;
/// `GICD_SPENDSGIR`.
pub const GICD_SPENDSGIR: u32 = 0x0f20;
/// `GICD_IROUTER`.
pub const GICD_IROUTER: u32 = 0x6000;
/// `GICD_CTLR_ARE`.
pub const GICD_CTLR_ARE: u32 = 1 << 4;
/// `GICD_CTLR_DS`.
pub const GICD_CTLR_DS: u32 = 1 << 6;

/// `GICR_SGI_OFFSET`, the second redistributor frame.
pub const GICR_SGI_OFFSET: u32 = 0x10000;
/// `GICR_CTLR`.
pub const GICR_CTLR: u32 = 0x0000;
/// `GICR_TYPER`.
pub const GICR_TYPER: u32 = 0x0008;
/// `GICR_STATUSR`.
pub const GICR_STATUSR: u32 = 0x0010;
/// `GICR_WAKER`.
pub const GICR_WAKER: u32 = 0x0014;
/// `GICR_PROPBASER`.
pub const GICR_PROPBASER: u32 = 0x0070;
/// `GICR_PENDBASER`.
pub const GICR_PENDBASER: u32 = 0x0078;
/// `GICR_IGROUPR0`.
pub const GICR_IGROUPR0: u32 = GICR_SGI_OFFSET + 0x0080;
/// `GICR_ISENABLER0`.
pub const GICR_ISENABLER0: u32 = GICR_SGI_OFFSET + 0x0100;
/// `GICR_ICENABLER0`.
pub const GICR_ICENABLER0: u32 = GICR_SGI_OFFSET + 0x0180;
/// `GICR_ISPENDR0`.
pub const GICR_ISPENDR0: u32 = GICR_SGI_OFFSET + 0x0200;
/// `GICR_ISACTIVER0`.
pub const GICR_ISACTIVER0: u32 = GICR_SGI_OFFSET + 0x0300;
/// `GICR_ICACTIVER0`.
pub const GICR_ICACTIVER0: u32 = GICR_SGI_OFFSET + 0x0380;
/// `GICR_IPRIORITYR`.
pub const GICR_IPRIORITYR: u32 = GICR_SGI_OFFSET + 0x0400;
/// `GICR_ICFGR1`.
pub const GICR_ICFGR1: u32 = GICR_SGI_OFFSET + 0x0c04;
/// `GICR_CTLR_CES`.
pub const GICR_CTLR_CES: u32 = 1 << 1;
/// `GICR_TYPER_PLPIS`.
pub const GICR_TYPER_PLPIS: u64 = 1 << 0;
/// `GICR_WAKER_ProcessorSleep`.
pub const GICR_WAKER_PROCESSOR_SLEEP: u32 = 1 << 1;
/// `GICR_WAKER_ChildrenAsleep`.
pub const GICR_WAKER_CHILDREN_ASLEEP: u32 = 1 << 2;

/// GICv2 CPU interface offsets: `GICC_CTLR`.
pub const GICC_CTLR: u32 = 0x00;
/// `GICC_PMR`.
pub const GICC_PMR: u32 = 0x04;
/// `GICC_BPR`.
pub const GICC_BPR: u32 = 0x08;
/// `GICC_ABPR`.
pub const GICC_ABPR: u32 = 0x1c;
/// `GICC_APR`, four of them.
pub const GICC_APR: u32 = 0xd0;

/// ITS register offsets: `GITS_CTLR`.
pub const GITS_CTLR: u64 = 0x0000;
/// `GITS_IIDR`.
pub const GITS_IIDR: u64 = 0x0004;
/// `GITS_CBASER`.
pub const GITS_CBASER: u64 = 0x0080;
/// `GITS_CWRITER`.
pub const GITS_CWRITER: u64 = 0x0088;
/// `GITS_CREADR`.
pub const GITS_CREADR: u64 = 0x0090;
/// `GITS_BASER`, eight of them.
pub const GITS_BASER: u64 = 0x0100;
/// `ITS_CONTROL_SIZE`: the control frame, followed by the translation frame.
pub const ITS_CONTROL_SIZE: u64 = 0x10000;
/// `GITS_TRANSLATER` in the translation frame.
pub const GITS_TRANSLATER: u64 = 0x0040;

/// `ICC_CTLR_EL1_PRIBITS_SHIFT`.
const ICC_CTLR_EL1_PRIBITS_SHIFT: u32 = 8;

/// `KVM_DEV_ARM_VGIC_SYSREG()`: the attribute of a CPU interface system register.
pub const fn sysreg(op0: u64, op1: u64, crn: u64, crm: u64, op2: u64) -> u64 {
    (op0 << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2
}

/// `ICC_PMR_EL1`.
pub const ICC_PMR_EL1: u64 = sysreg(3, 0, 4, 6, 0);
/// `ICC_BPR0_EL1`.
pub const ICC_BPR0_EL1: u64 = sysreg(3, 0, 12, 8, 3);
/// `ICC_BPR1_EL1`.
pub const ICC_BPR1_EL1: u64 = sysreg(3, 0, 12, 12, 3);
/// `ICC_CTLR_EL1`.
pub const ICC_CTLR_EL1: u64 = sysreg(3, 0, 12, 12, 4);
/// `ICC_SRE_EL1`.
pub const ICC_SRE_EL1: u64 = sysreg(3, 0, 12, 12, 5);
/// `ICC_IGRPEN0_EL1`.
pub const ICC_IGRPEN0_EL1: u64 = sysreg(3, 0, 12, 12, 6);
/// `ICC_IGRPEN1_EL1`.
pub const ICC_IGRPEN1_EL1: u64 = sysreg(3, 0, 12, 12, 7);

/// `ICC_AP0R<n>_EL1`.
pub const fn icc_ap0r(n: u64) -> u64 {
    sysreg(3, 0, 12, 8, 4 | n)
}

/// `ICC_AP1R<n>_EL1`.
pub const fn icc_ap1r(n: u64) -> u64 {
    sysreg(3, 0, 12, 9, n)
}

/// Whether an attribute of `group` carries a 64-bit value rather than a 32-bit one.
pub const fn attr_is_64bit(group: u32) -> bool {
    matches!(
        group,
        KVM_DEV_ARM_VGIC_GRP_ADDR
            | KVM_DEV_ARM_VGIC_GRP_CPU_SYSREGS
            | KVM_DEV_ARM_VGIC_GRP_ITS_REGS
    )
}

/// `KVM_VGIC_ATTR()` in arm_gicv3_kvm.c: a register of the redistributor or CPU interface whose
/// affinity is in the top half of `gicr_typer`. The distributor uses 0.
pub const fn v3_attr(reg: u64, gicr_typer: u64) -> u64 {
    (gicr_typer & KVM_DEV_ARM_VGIC_V3_MPIDR_MASK) | reg
}

/// The line level attribute of the 32 interrupts from `irq`, `kvm_gic_line_level_access()`.
pub const fn v3_line_level_attr(irq: u32, gicr_typer: u64) -> u64 {
    v3_attr(irq as u64, gicr_typer)
        | (VGIC_LEVEL_INFO_LINE_LEVEL << KVM_DEV_ARM_VGIC_LINE_LEVEL_INFO_SHIFT)
}

/// `KVM_VGIC_ATTR()` in arm_gic_kvm.c: a GICv2 register as seen from CPU `cpu`.
pub const fn v2_attr(offset: u32, cpu: u32) -> u64 {
    (((cpu as u64) << KVM_DEV_ARM_VGIC_CPUID_SHIFT) & KVM_DEV_ARM_VGIC_CPUID_MASK)
        | (offset as u64 & KVM_DEV_ARM_VGIC_OFFSET_MASK)
}

/// The `GICR_TYPER` that arm_gicv3_common.c builds for CPU `index` from its MPIDR affinity. Only
/// the affinity in the top half matters to KVM, which picks the redistributor by it.
pub const fn gicr_typer(mp_affinity: u64, index: u32, lpis: bool) -> u64 {
    let affid = ((mp_affinity & 0xff_0000_0000) >> 8) | (mp_affinity & 0xff_ffff);
    (affid << 32) | (1 << 24) | ((index as u64) << 8) | if lpis { GICR_TYPER_PLPIS } else { 0 }
}

/// The value written to `KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION`: the base with the region index
/// in the low bits and the number of redistributors from bit 52.
pub const fn redist_region_attr(base: u64, index: u32, count: u32) -> u64 {
    base | index as u64 | ((count as u64) << 52)
}

/// The doorbell address an MSI through the in-kernel ITS targets: `GITS_TRANSLATER` in the
/// translation frame that follows the control frame at `its_base`.
pub const fn its_translater_gpa(its_base: u64) -> u64 {
    its_base + ITS_CONTROL_SIZE + GITS_TRANSLATER
}

/// The number of interrupts a `GICD_TYPER` value describes.
pub const fn typer_num_irqs(typer: u32) -> u32 {
    ((typer & 0x1f) + 1) * 32
}

/// The number of CPU interfaces a GICv2 `GICD_TYPER` value describes.
pub const fn typer_num_cpus(typer: u32) -> u32 {
    ((typer & 0xe0) >> 5) + 1
}

/// `kvm_arm_set_irq()`: the `KVM_IRQ_LINE` number of interrupt `irq` of type `irq_type` on
/// vCPU `cpu`. vCPU numbers past 255 spill into the `VCPU2` field.
pub const fn irq_line(cpu: u32, irq_type: u32, irq: u32) -> u32 {
    (irq_type << KVM_ARM_IRQ_TYPE_SHIFT)
        | irq
        | ((cpu % 256) << KVM_ARM_IRQ_VCPU_SHIFT)
        | ((cpu / 256) << KVM_ARM_IRQ_VCPU2_SHIFT)
}

/// `kvm_arm_gic_set_irq()`: the line of GIC input `irq`. Inputs below `num_irq - 32` are SPIs,
/// and after them come 32 PPIs for each CPU in turn.
pub const fn gic_irq_line(num_irq: u32, irq: u32) -> u32 {
    if irq < num_irq - GIC_INTERNAL {
        irq_line(0, KVM_ARM_IRQ_TYPE_SPI, irq + GIC_INTERNAL)
    } else {
        let ppi = irq - (num_irq - GIC_INTERNAL);
        irq_line(ppi / GIC_INTERNAL, KVM_ARM_IRQ_TYPE_PPI, ppi % GIC_INTERNAL)
    }
}

/// `arm_cpu_kvm_set_irq()`: the IRQ or FIQ line of vCPU `cpu`, used when the GIC is in
/// userspace.
pub const fn cpu_irq_line(cpu: u32, fiq: bool) -> u32 {
    irq_line(cpu, KVM_ARM_IRQ_TYPE_CPU, if fiq { KVM_ARM_IRQ_CPU_FIQ } else { KVM_ARM_IRQ_CPU_IRQ })
}

/// `kvm_arch_msi_data_to_gsi()`.
pub const fn msi_data_to_gsi(data: u32) -> u32 {
    data.wrapping_sub(GIC_INTERNAL) & 0xffff
}

/// `half_shuffle32()`: spreads the low 16 bits onto the even bits.
pub const fn half_shuffle32(x: u32) -> u32 {
    let x = ((x & 0xff00) << 8) | (x & 0x00ff);
    let x = ((x << 4) | x) & 0x0f0f_0f0f;
    let x = ((x << 2) | x) & 0x3333_3333;
    ((x << 1) | x) & 0x5555_5555
}

/// `half_unshuffle32()`: gathers the even bits into the low 16.
pub const fn half_unshuffle32(x: u32) -> u32 {
    let x = x & 0x5555_5555;
    let x = ((x >> 1) | x) & 0x3333_3333;
    let x = ((x >> 2) | x) & 0x0f0f_0f0f;
    let x = ((x >> 4) | x) & 0x00ff_00ff;
    ((x >> 8) | x) & 0x0000_ffff
}

/// `kvm_arch_irqchip_create()` for Arm, after the generic `KVM_CAP_IRQCHIP` and `KVM_CAP_IRQFD`
/// checks. Split is refused. With `KVM_CAP_DEVICE_CTRL` the vGIC device creates itself later,
/// so the answer is false; without it the caller must issue `KVM_CREATE_IRQCHIP`.
pub fn irqchip_needs_create(mode: KernelIrqchip, device_ctrl: bool) -> Result<bool, KvmError> {
    if mode == KernelIrqchip::Split {
        return Err(KvmError::ArmSplitIrqchip);
    }
    Ok(!device_ctrl)
}

/// A device output that KVM raises in `device_irq_level` when the GIC is in userspace.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DeviceIrq {
    /// The EL1 virtual timer, `GTIMER_VIRT`.
    VirtTimer,
    /// The EL1 physical timer, `GTIMER_PHYS`.
    PhysTimer,
    /// The PMU overflow interrupt.
    Pmu,
}

/// What changed in `device_irq_level` since the last exit, from `kvm_arch_post_run()`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeviceIrqChanges {
    /// Each output that changed and its new level, timers first as QEMU does.
    pub changed: Vec<(DeviceIrq, bool)>,
    /// Bits that changed and that ruvm has no output for. QEMU logs them as
    /// `unhandled in-kernel device IRQ %x`.
    pub unhandled: u64,
}

/// Compares the previous `device_irq_level` with the one from `kvm_run`.
pub fn device_irq_changes(old: u64, new: u64) -> DeviceIrqChanges {
    let mut switched = old ^ new;
    let mut changed = Vec::new();
    for (bit, irq) in [
        (KVM_ARM_DEV_EL1_VTIMER, DeviceIrq::VirtTimer),
        (KVM_ARM_DEV_EL1_PTIMER, DeviceIrq::PhysTimer),
        (KVM_ARM_DEV_PMU, DeviceIrq::Pmu),
    ] {
        if switched & bit != 0 {
            changed.push((irq, new & bit != 0));
            switched &= !bit;
        }
    }
    DeviceIrqChanges { changed, unhandled: switched }
}

/// The `gic-version` machine property, `VirtGICType`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GicVersion {
    /// `host`: whatever the host does best, KVM only.
    Host,
    /// `max`.
    Max,
    /// No property given.
    NoSel,
    /// GICv2.
    V2,
    /// GICv3.
    V3,
    /// GICv4.
    V4,
    /// GICv5.
    V5,
}

/// `VIRT_GIC_VERSION_2_MASK` and friends.
pub const GIC_V2_MASK: u32 = 1 << 0;
/// GICv3 is available.
pub const GIC_V3_MASK: u32 = 1 << 1;
/// GICv4 is available.
pub const GIC_V4_MASK: u32 = 1 << 2;
/// GICv5 is available.
pub const GIC_V5_MASK: u32 = 1 << 3;

/// The KVM half of `finalize_gic_version()`: which GICs KVM can give the guest and the name to
/// use in errors. `probe` is the `vgic_probe` bitmap, only looked at with
/// the irqchip in the kernel.
pub fn kvm_gics_supported(
    irqchip_in_kernel: bool,
    probe: u32,
) -> Result<(u32, &'static str), VgicError> {
    if !irqchip_in_kernel {
        return Ok((GIC_V2_MASK, "KVM with kernel-irqchip=off"));
    }
    if probe == 0 {
        return Err(VgicError::config("Unable to determine GIC version supported by host"));
    }
    let mut supported = 0;
    if probe & KVM_ARM_VGIC_V2 != 0 {
        supported |= GIC_V2_MASK;
    }
    if probe & KVM_ARM_VGIC_V3 != 0 {
        supported |= GIC_V3_MASK;
    }
    Ok((supported, "KVM"))
}

/// `finalize_gic_version_do()`: turns `host`, `max` and no choice into a version and checks the
/// accelerator supports it. `kvm` says whether the accelerator is KVM, for `host`.
pub fn finalize_gic_version(
    accel_name: &str,
    kvm: bool,
    version: GicVersion,
    supported: u32,
    max_cpus: u32,
) -> Result<GicVersion, VgicError> {
    let version = match version {
        GicVersion::Host if !kvm => {
            return Err(VgicError::config("gic-version=host requires KVM"));
        }
        GicVersion::Host | GicVersion::Max => {
            if supported & GIC_V4_MASK != 0 {
                GicVersion::V4
            } else if supported & GIC_V3_MASK != 0 {
                GicVersion::V3
            } else {
                GicVersion::V2
            }
        }
        GicVersion::NoSel => {
            if supported & GIC_V2_MASK != 0 && max_cpus <= GIC_NCPU {
                GicVersion::V2
            } else if supported & GIC_V3_MASK != 0 {
                GicVersion::V3
            } else if max_cpus > GIC_NCPU {
                return Err(VgicError::config(format!(
                    "{accel_name} only supports GICv2 emulation but more than 8 vcpus are requested"
                )));
            } else {
                GicVersion::NoSel
            }
        }
        v => v,
    };
    let (mask, n) = match version {
        GicVersion::V2 => (GIC_V2_MASK, 2),
        GicVersion::V3 => (GIC_V3_MASK, 3),
        GicVersion::V4 => (GIC_V4_MASK, 4),
        GicVersion::V5 => (GIC_V5_MASK, 5),
        _ => return Err(VgicError::config("logic error in finalize_gic_version")),
    };
    if supported & mask == 0 {
        let extra = if n == 4 { ", is virtualization=on?" } else { "" };
        return Err(VgicError::config(format!(
            "{accel_name} does not support GICv{n} emulation{extra}"
        )));
    }
    Ok(version)
}

/// Why a vGIC step failed. The messages are QEMU's.
#[derive(Debug)]
pub enum VgicError {
    /// `kvm_device_access()` failed.
    Access {
        /// `KVM_SET_DEVICE_ATTR` rather than `KVM_GET_DEVICE_ATTR`.
        write: bool,
        /// The attribute group.
        group: u32,
        /// The attribute.
        attr: u64,
        /// What the kernel said.
        err: io::Error,
    },
    /// A setup step failed with an errno, `error_setg_errno()`.
    Os(&'static str, io::Error),
    /// The configuration asks for something the kernel or the model cannot do.
    Config(String, Option<String>),
}

impl VgicError {
    fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into(), None)
    }

    /// The hint QEMU appends on the next line, if any.
    pub fn hint(&self) -> Option<&str> {
        match self {
            Self::Config(_, hint) => hint.as_deref(),
            _ => None,
        }
    }

    /// The errno behind the failure, if the kernel gave one.
    pub fn errno(&self) -> Option<i32> {
        match self {
            Self::Access { err, .. } | Self::Os(_, err) => err.raw_os_error(),
            Self::Config(..) => None,
        }
    }

    /// Rewords a failed access as the setup step it was part of.
    fn during(self, what: &'static str) -> Self {
        match self {
            Self::Access { err, .. } => Self::Os(what, err),
            other => other,
        }
    }
}

impl fmt::Display for VgicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Access { write, group, attr, err } => write!(
                f,
                "KVM_{}_DEVICE_ATTR failed: Group {group} attr 0x{attr:016x}: {}",
                if *write { "SET" } else { "GET" },
                strerror(err)
            ),
            Self::Os(what, err) => write!(f, "{what}: {}", strerror(err)),
            Self::Config(msg, _) => f.write_str(msg),
        }
    }
}

impl std::error::Error for VgicError {}

/// A vGIC or ITS device file descriptor, or a fake one in tests. Values travel as `u64` and the
/// device stores 32 or 64 bits as [`attr_is_64bit`] says.
pub trait VgicDevice {
    /// `KVM_GET_DEVICE_ATTR`.
    fn get(&mut self, group: u32, attr: u64) -> Result<u64, VgicError>;
    /// `KVM_SET_DEVICE_ATTR`. Commands in [`KVM_DEV_ARM_VGIC_GRP_CTRL`] ignore `value`.
    fn set(&mut self, group: u32, attr: u64, value: u64) -> Result<(), VgicError>;
    /// `KVM_HAS_DEVICE_ATTR`.
    fn has(&self, group: u32, attr: u64) -> io::Result<()>;
}

fn gicd_get(dev: &mut impl VgicDevice, offset: u32) -> Result<u32, VgicError> {
    Ok(dev.get(KVM_DEV_ARM_VGIC_GRP_DIST_REGS, v3_attr(u64::from(offset), 0))? as u32)
}

fn gicd_set(dev: &mut impl VgicDevice, offset: u32, value: u32) -> Result<(), VgicError> {
    dev.set(KVM_DEV_ARM_VGIC_GRP_DIST_REGS, v3_attr(u64::from(offset), 0), u64::from(value))
}

fn gicr_get(dev: &mut impl VgicDevice, offset: u32, typer: u64) -> Result<u32, VgicError> {
    Ok(dev.get(KVM_DEV_ARM_VGIC_GRP_REDIST_REGS, v3_attr(u64::from(offset), typer))? as u32)
}

fn gicr_set(
    dev: &mut impl VgicDevice,
    offset: u32,
    typer: u64,
    value: u32,
) -> Result<(), VgicError> {
    dev.set(KVM_DEV_ARM_VGIC_GRP_REDIST_REGS, v3_attr(u64::from(offset), typer), u64::from(value))
}

/// A 64-bit redistributor register, which KVM takes as two 32-bit halves.
fn gicr_get64(dev: &mut impl VgicDevice, offset: u32, typer: u64) -> Result<u64, VgicError> {
    let lo = gicr_get(dev, offset, typer)?;
    let hi = gicr_get(dev, offset + 4, typer)?;
    Ok((u64::from(hi) << 32) | u64::from(lo))
}

fn gicr_set64(
    dev: &mut impl VgicDevice,
    offset: u32,
    typer: u64,
    value: u64,
) -> Result<(), VgicError> {
    gicr_set(dev, offset, typer, value as u32)?;
    gicr_set(dev, offset + 4, typer, (value >> 32) as u32)
}

fn gicc_get(dev: &mut impl VgicDevice, reg: u64, typer: u64) -> Result<u64, VgicError> {
    dev.get(KVM_DEV_ARM_VGIC_GRP_CPU_SYSREGS, v3_attr(reg, typer))
}

fn gicc_set(dev: &mut impl VgicDevice, reg: u64, typer: u64, value: u64) -> Result<(), VgicError> {
    dev.set(KVM_DEV_ARM_VGIC_GRP_CPU_SYSREGS, v3_attr(reg, typer), value)
}

fn level_get(dev: &mut impl VgicDevice, irq: u32, typer: u64) -> Result<u32, VgicError> {
    Ok(dev.get(KVM_DEV_ARM_VGIC_GRP_LEVEL_INFO, v3_line_level_attr(irq, typer))? as u32)
}

fn level_set(dev: &mut impl VgicDevice, irq: u32, typer: u64, value: u32) -> Result<(), VgicError> {
    dev.set(KVM_DEV_ARM_VGIC_GRP_LEVEL_INFO, v3_line_level_attr(irq, typer), u64::from(value))
}

/// The active priority registers a CPU interface has, highest first, from the priority bits in
/// `ICC_CTLR_EL1`: four with 7 bits, two with 6, one otherwise.
fn apr_regs(icc_ctlr: u64) -> &'static [u64] {
    match ((icc_ctlr >> ICC_CTLR_EL1_PRIBITS_SHIFT) & 7) + 1 {
        7 => &[3, 2, 1, 0],
        6 => &[1, 0],
        _ => &[0],
    }
}

/// One GICv3 redistributor and CPU interface, the fields of `GICv3CPUState` that the KVM save
/// and restore touch. Index 0 of the paired arrays is group 0 and index 1 is non-secure group 1.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GicV3Cpu {
    /// `GICR_TYPER`. Its top half, the affinity, names this CPU to KVM.
    pub gicr_typer: u64,
    /// `GICR_CTLR`.
    pub gicr_ctlr: u32,
    /// The non-secure `GICR_STATUSR`.
    pub gicr_statusr: u32,
    /// `GICR_WAKER`.
    pub gicr_waker: u32,
    /// `GICR_PROPBASER`.
    pub gicr_propbaser: u64,
    /// `GICR_PENDBASER`.
    pub gicr_pendbaser: u64,
    /// `GICR_IGROUPR0`.
    pub gicr_igroupr0: u32,
    /// `GICR_ISENABLER0`.
    pub gicr_ienabler0: u32,
    /// `GICR_ISPENDR0`.
    pub gicr_ipendr0: u32,
    /// `GICR_ISACTIVER0`.
    pub gicr_iactiver0: u32,
    /// One bit per SGI and PPI, set when edge triggered.
    pub edge_trigger: u32,
    /// The line levels of the PPIs.
    pub level: u32,
    /// `GICR_IPRIORITYR<n>`, one byte per interrupt.
    pub gicr_ipriorityr: [u8; 32],
    /// `ICC_SRE_EL1`.
    pub icc_sre_el1: u64,
    /// The non-secure `ICC_CTLR_EL1`.
    pub icc_ctlr_el1: u64,
    /// `ICC_IGRPEN0_EL1` and `ICC_IGRPEN1_EL1`.
    pub icc_igrpen: [u64; 2],
    /// `ICC_PMR_EL1`.
    pub icc_pmr_el1: u64,
    /// `ICC_BPR0_EL1` and `ICC_BPR1_EL1`.
    pub icc_bpr: [u64; 2],
    /// `ICC_AP0R<n>_EL1` and `ICC_AP1R<n>_EL1`.
    pub icc_apr: [[u64; 4]; 2],
    /// The kernel's `ICC_CTLR_EL1` after reset, read once at setup.
    pub kvm_reset_icc_ctlr_el1: u64,
}

impl GicV3Cpu {
    /// `arm_gicv3_icc_reset()`, run on each CPU reset. The kernel cannot be asked while only one
    /// vCPU is paused, so this assumes its reset values, which are zero apart from `ICC_SRE_EL1`
    /// and the `ICC_CTLR_EL1` read at setup.
    pub fn icc_reset(&mut self) {
        self.icc_pmr_el1 = 0;
        self.icc_bpr = [0; 2];
        self.icc_sre_el1 = 0x7;
        self.icc_apr = [[0; 4]; 2];
        self.icc_igrpen = [0; 2];
        self.icc_ctlr_el1 = self.kvm_reset_icc_ctlr_el1;
    }
}

/// A GICv3 as KVM saves it, the fields of `GICv3State` that arm_gicv3_kvm.c reads and writes.
/// The bitmaps hold one bit per interrupt in 32-bit words, and only the words from interrupt 32
/// on mean anything: the SGIs and PPIs live in each [`GicV3Cpu`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GicV3State {
    /// The number of interrupts, a multiple of 32.
    pub num_irq: u32,
    /// `GICD_CTLR`.
    pub gicd_ctlr: u32,
    /// The non-secure `GICD_STATUSR`.
    pub gicd_statusr: u32,
    /// `GICD_IGROUPR<n>`.
    pub group: Vec<u32>,
    /// `GICD_ISENABLER<n>`.
    pub enabled: Vec<u32>,
    /// `GICD_ISPENDR<n>`.
    pub pending: Vec<u32>,
    /// `GICD_ISACTIVER<n>`.
    pub active: Vec<u32>,
    /// The SPI line levels.
    pub level: Vec<u32>,
    /// One bit per SPI, set when edge triggered.
    pub edge_trigger: Vec<u32>,
    /// `GICD_IPRIORITYR<n>`, one byte per interrupt.
    pub gicd_ipriority: Vec<u8>,
    /// `GICD_IROUTER<n>`, one per interrupt.
    pub gicd_irouter: Vec<u64>,
    /// One per vCPU.
    pub cpus: Vec<GicV3Cpu>,
}

impl GicV3State {
    /// A GIC with `num_irq` interrupts and a CPU for each `GICR_TYPER` in `typers`, all zero.
    pub fn new(num_irq: u32, typers: &[u64]) -> Self {
        let words = (num_irq / 32) as usize;
        GicV3State {
            num_irq,
            group: vec![0; words],
            enabled: vec![0; words],
            pending: vec![0; words],
            active: vec![0; words],
            level: vec![0; words],
            edge_trigger: vec![0; words],
            gicd_ipriority: vec![0; num_irq as usize],
            gicd_irouter: vec![0; num_irq as usize],
            cpus: typers
                .iter()
                .map(|&t| GicV3Cpu { gicr_typer: t, ..Default::default() })
                .collect(),
            ..Default::default()
        }
    }

    /// `arm_gicv3_common_reset_hold()` for a GIC without the security extensions, which is the
    /// only kind KVM has. The CPU interfaces are not touched: they reset with their CPU.
    pub fn reset(&mut self, lpi_enable: bool, irq_reset_nonsecure: bool) {
        for c in &mut self.cpus {
            c.level = 0;
            c.gicr_ctlr = if lpi_enable { GICR_CTLR_CES } else { 0 };
            c.gicr_statusr = 0;
            c.gicr_waker = GICR_WAKER_PROCESSOR_SLEEP | GICR_WAKER_CHILDREN_ASLEEP;
            c.gicr_propbaser = 0;
            c.gicr_pendbaser = 0;
            c.gicr_igroupr0 = if irq_reset_nonsecure { 0xffff_ffff } else { 0 };
            c.gicr_ienabler0 = 0;
            c.gicr_ipendr0 = 0;
            c.gicr_iactiver0 = 0;
            c.edge_trigger = 0xffff;
            c.gicr_ipriorityr = [0; 32];
        }
        self.gicd_ctlr = GICD_CTLR_DS | GICD_CTLR_ARE;
        self.gicd_statusr = 0;
        for bmp in [
            &mut self.group,
            &mut self.enabled,
            &mut self.pending,
            &mut self.active,
            &mut self.level,
            &mut self.edge_trigger,
        ] {
            bmp.fill(0);
        }
        self.gicd_ipriority.fill(0);
        self.gicd_irouter.fill(0);
        if irq_reset_nonsecure {
            for w in self.group.iter_mut().skip(1) {
                *w = 0xffff_ffff;
            }
        }
    }

    fn typer0(&self) -> u64 {
        self.cpus.first().map_or(0, |c| c.gicr_typer)
    }

    fn word(irq: u32) -> usize {
        (irq / 32) as usize
    }
}

/// `kvm_arm_gicv3_check()`.
fn gicv3_check(dev: &mut impl VgicDevice, num_irq: u32) -> Result<(), VgicError> {
    let kernel = typer_num_irqs(gicd_get(dev, GICD_TYPER)?);
    if kernel < num_irq {
        return Err(VgicError::config(format!(
            "Model requests {num_irq} IRQs, but kernel supports max {kernel}"
        )));
    }
    Ok(())
}

/// `kvm_dist_putbmp()`: a bitmap into the distributor from interrupt 32, clearing every bit
/// through `clear` first when the register has a clear twin.
fn dist_put_bmp(
    dev: &mut impl VgicDevice,
    num_irq: u32,
    set: u32,
    clear: Option<u32>,
    bmp: &[u32],
) -> Result<(), VgicError> {
    let mut offset = set + GIC_INTERNAL / 8;
    let mut clear = clear.map(|c| c + GIC_INTERNAL / 8);
    for irq in (GIC_INTERNAL..num_irq).step_by(32) {
        if let Some(c) = clear.as_mut() {
            gicd_set(dev, *c, !0)?;
            *c += 4;
        }
        gicd_set(dev, offset, bmp[GicV3State::word(irq)])?;
        offset += 4;
    }
    Ok(())
}

/// `kvm_dist_getbmp()`.
fn dist_get_bmp(
    dev: &mut impl VgicDevice,
    num_irq: u32,
    offset: u32,
    bmp: &mut [u32],
) -> Result<(), VgicError> {
    let mut offset = offset + GIC_INTERNAL / 8;
    for irq in (GIC_INTERNAL..num_irq).step_by(32) {
        bmp[GicV3State::word(irq)] = gicd_get(dev, offset)?;
        offset += 4;
    }
    Ok(())
}

/// `kvm_dist_put_edge_trigger()`: two bits per interrupt in `GICD_ICFGR<n>`, the upper one set
/// for edge.
fn dist_put_edge(dev: &mut impl VgicDevice, num_irq: u32, bmp: &[u32]) -> Result<(), VgicError> {
    let mut offset = GICD_ICFGR + GIC_INTERNAL * 2 / 8;
    for irq in (GIC_INTERNAL..num_irq).step_by(16) {
        let word = bmp[GicV3State::word(irq)];
        let half = if irq % 32 != 0 { word >> 16 } else { word & 0xffff };
        gicd_set(dev, offset, half_shuffle32(half) << 1)?;
        offset += 4;
    }
    Ok(())
}

/// `kvm_dist_get_edge_trigger()`. QEMU ORs into whatever the bitmap held, so the words are
/// cleared first here to make a read give exactly what the kernel has.
fn dist_get_edge(
    dev: &mut impl VgicDevice,
    num_irq: u32,
    bmp: &mut [u32],
) -> Result<(), VgicError> {
    for w in bmp.iter_mut().skip(1) {
        *w = 0;
    }
    let mut offset = GICD_ICFGR + GIC_INTERNAL * 2 / 8;
    for irq in (GIC_INTERNAL..num_irq).step_by(16) {
        let mut reg = half_unshuffle32(gicd_get(dev, offset)? >> 1);
        if irq % 32 != 0 {
            reg <<= 16;
        }
        bmp[GicV3State::word(irq)] |= reg;
        offset += 4;
    }
    Ok(())
}

/// `kvm_dist_put_priority()`: four priority bytes per register from interrupt 32.
fn dist_put_priority(
    dev: &mut impl VgicDevice,
    num_irq: u32,
    prio: &[u8],
) -> Result<(), VgicError> {
    let mut offset = GICD_IPRIORITYR + GIC_INTERNAL;
    for irq in (GIC_INTERNAL..num_irq).step_by(4) {
        let i = irq as usize;
        let reg = u32::from_le_bytes([prio[i], prio[i + 1], prio[i + 2], prio[i + 3]]);
        gicd_set(dev, offset, reg)?;
        offset += 4;
    }
    Ok(())
}

/// `kvm_dist_get_priority()`.
fn dist_get_priority(
    dev: &mut impl VgicDevice,
    num_irq: u32,
    prio: &mut [u8],
) -> Result<(), VgicError> {
    let mut offset = GICD_IPRIORITYR + GIC_INTERNAL;
    for irq in (GIC_INTERNAL..num_irq).step_by(4) {
        let i = irq as usize;
        prio[i..i + 4].copy_from_slice(&gicd_get(dev, offset)?.to_le_bytes());
        offset += 4;
    }
    Ok(())
}

/// The offset of `GICD_IROUTER<irq>`. QEMU steps by four bytes per interrupt, which lands on
/// the wrong register because each router is eight bytes wide; ruvm steps by eight, which is
/// where the kernel looks.
fn irouter_offset(irq: u32) -> u32 {
    GICD_IROUTER + 8 * irq
}

/// `kvm_arm_gicv3_put()`: writes the whole GIC into the kernel, in QEMU's order.
pub fn gicv3_put(dev: &mut impl VgicDevice, s: &GicV3State) -> Result<(), VgicError> {
    gicv3_check(dev, s.num_irq)?;
    let typer0 = s.typer0();
    let redist_typer = gicr_get64(dev, GICR_TYPER, typer0)?;
    gicd_set(dev, GICD_CTLR, s.gicd_ctlr)?;

    if redist_typer & GICR_TYPER_PLPIS != 0 {
        // The base addresses go first, before a GICR_CTLR write can turn LPIs on.
        for c in &s.cpus {
            gicr_set64(dev, GICR_PROPBASER, c.gicr_typer, c.gicr_propbaser)?;
            gicr_set64(dev, GICR_PENDBASER, c.gicr_typer, c.gicr_pendbaser)?;
        }
    }

    for c in &s.cpus {
        let t = c.gicr_typer;
        gicr_set(dev, GICR_CTLR, t, c.gicr_ctlr)?;
        gicr_set(dev, GICR_STATUSR, t, c.gicr_statusr)?;
        gicr_set(dev, GICR_WAKER, t, c.gicr_waker)?;
        gicr_set(dev, GICR_IGROUPR0, t, c.gicr_igroupr0)?;
        gicr_set(dev, GICR_ICENABLER0, t, !0)?;
        gicr_set(dev, GICR_ISENABLER0, t, c.gicr_ienabler0)?;
        // The configuration goes before the pending state so level and edge are right.
        gicr_set(dev, GICR_ICFGR1, t, half_shuffle32(c.edge_trigger >> 16) << 1)?;
        level_set(dev, 0, t, c.level)?;
        gicr_set(dev, GICR_ISPENDR0, t, c.gicr_ipendr0)?;
        gicr_set(dev, GICR_ICACTIVER0, t, !0)?;
        gicr_set(dev, GICR_ISACTIVER0, t, c.gicr_iactiver0)?;
        for (i, p) in c.gicr_ipriorityr.chunks_exact(4).enumerate() {
            let reg = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
            gicr_set(dev, GICR_IPRIORITYR + 4 * i as u32, t, reg)?;
        }
    }

    gicd_set(dev, GICD_STATUSR, s.gicd_statusr)?;
    dist_put_bmp(dev, s.num_irq, GICD_ISENABLER, Some(GICD_ICENABLER), &s.enabled)?;
    dist_put_bmp(dev, s.num_irq, GICD_IGROUPR, None, &s.group)?;
    // The routing goes before the pending state so it lands on the right CPU interface.
    for irq in GIC_INTERNAL..s.num_irq {
        let r = s.gicd_irouter[irq as usize];
        gicd_set(dev, irouter_offset(irq), r as u32)?;
        gicd_set(dev, irouter_offset(irq) + 4, (r >> 32) as u32)?;
    }
    dist_put_edge(dev, s.num_irq, &s.edge_trigger)?;
    for irq in (GIC_INTERNAL..s.num_irq).step_by(32) {
        level_set(dev, irq, typer0, s.level[GicV3State::word(irq)])?;
    }
    dist_put_bmp(dev, s.num_irq, GICD_ISPENDR, None, &s.pending)?;
    dist_put_bmp(dev, s.num_irq, GICD_ISACTIVER, Some(GICD_ICACTIVER), &s.active)?;
    dist_put_priority(dev, s.num_irq, &s.gicd_ipriority)?;

    for c in &s.cpus {
        let t = c.gicr_typer;
        gicc_set(dev, ICC_SRE_EL1, t, c.icc_sre_el1)?;
        gicc_set(dev, ICC_CTLR_EL1, t, c.icc_ctlr_el1)?;
        gicc_set(dev, ICC_IGRPEN0_EL1, t, c.icc_igrpen[0])?;
        gicc_set(dev, ICC_IGRPEN1_EL1, t, c.icc_igrpen[1])?;
        gicc_set(dev, ICC_PMR_EL1, t, c.icc_pmr_el1)?;
        gicc_set(dev, ICC_BPR0_EL1, t, c.icc_bpr[0])?;
        gicc_set(dev, ICC_BPR1_EL1, t, c.icc_bpr[1])?;
        for &n in apr_regs(c.icc_ctlr_el1) {
            gicc_set(dev, icc_ap0r(n), t, c.icc_apr[0][n as usize])?;
        }
        for &n in apr_regs(c.icc_ctlr_el1) {
            gicc_set(dev, icc_ap1r(n), t, c.icc_apr[1][n as usize])?;
        }
    }
    Ok(())
}

/// `kvm_arm_gicv3_get()`: reads the whole GIC out of the kernel.
pub fn gicv3_get(dev: &mut impl VgicDevice, s: &mut GicV3State) -> Result<(), VgicError> {
    gicv3_check(dev, s.num_irq)?;
    let typer0 = s.typer0();
    let redist_typer = gicr_get64(dev, GICR_TYPER, typer0)?;
    s.gicd_ctlr = gicd_get(dev, GICD_CTLR)?;

    for c in &mut s.cpus {
        let t = c.gicr_typer;
        c.gicr_ctlr = gicr_get(dev, GICR_CTLR, t)?;
        c.gicr_statusr = gicr_get(dev, GICR_STATUSR, t)?;
        c.gicr_waker = gicr_get(dev, GICR_WAKER, t)?;
        c.gicr_igroupr0 = gicr_get(dev, GICR_IGROUPR0, t)?;
        c.gicr_ienabler0 = gicr_get(dev, GICR_ISENABLER0, t)?;
        c.edge_trigger = half_unshuffle32(gicr_get(dev, GICR_ICFGR1, t)? >> 1) << 16;
        c.level = level_get(dev, 0, t)?;
        c.gicr_ipendr0 = gicr_get(dev, GICR_ISPENDR0, t)?;
        c.gicr_iactiver0 = gicr_get(dev, GICR_ISACTIVER0, t)?;
        for i in 0..8 {
            let reg = gicr_get(dev, GICR_IPRIORITYR + 4 * i, t)?;
            c.gicr_ipriorityr[4 * i as usize..4 * i as usize + 4]
                .copy_from_slice(&reg.to_le_bytes());
        }
    }

    if redist_typer & GICR_TYPER_PLPIS != 0 {
        for c in &mut s.cpus {
            c.gicr_propbaser = gicr_get64(dev, GICR_PROPBASER, c.gicr_typer)?;
            c.gicr_pendbaser = gicr_get64(dev, GICR_PENDBASER, c.gicr_typer)?;
        }
    }

    s.gicd_statusr = gicd_get(dev, GICD_STATUSR)?;
    dist_get_bmp(dev, s.num_irq, GICD_IGROUPR, &mut s.group)?;
    dist_get_bmp(dev, s.num_irq, GICD_ISENABLER, &mut s.enabled)?;
    for irq in (GIC_INTERNAL..s.num_irq).step_by(32) {
        s.level[GicV3State::word(irq)] = level_get(dev, irq, typer0)?;
    }
    dist_get_bmp(dev, s.num_irq, GICD_ISPENDR, &mut s.pending)?;
    dist_get_bmp(dev, s.num_irq, GICD_ISACTIVER, &mut s.active)?;
    dist_get_edge(dev, s.num_irq, &mut s.edge_trigger)?;
    dist_get_priority(dev, s.num_irq, &mut s.gicd_ipriority)?;
    for irq in GIC_INTERNAL..s.num_irq {
        let lo = gicd_get(dev, irouter_offset(irq))?;
        let hi = gicd_get(dev, irouter_offset(irq) + 4)?;
        s.gicd_irouter[irq as usize] = (u64::from(hi) << 32) | u64::from(lo);
    }

    for c in &mut s.cpus {
        let t = c.gicr_typer;
        c.icc_sre_el1 = gicc_get(dev, ICC_SRE_EL1, t)?;
        c.icc_ctlr_el1 = gicc_get(dev, ICC_CTLR_EL1, t)?;
        c.icc_igrpen[0] = gicc_get(dev, ICC_IGRPEN0_EL1, t)?;
        c.icc_igrpen[1] = gicc_get(dev, ICC_IGRPEN1_EL1, t)?;
        c.icc_pmr_el1 = gicc_get(dev, ICC_PMR_EL1, t)?;
        c.icc_bpr[0] = gicc_get(dev, ICC_BPR0_EL1, t)?;
        c.icc_bpr[1] = gicc_get(dev, ICC_BPR1_EL1, t)?;
        for &n in apr_regs(c.icc_ctlr_el1) {
            c.icc_apr[0][n as usize] = gicc_get(dev, icc_ap0r(n), t)?;
        }
        for &n in apr_regs(c.icc_ctlr_el1) {
            c.icc_apr[1][n as usize] = gicc_get(dev, icc_ap1r(n), t)?;
        }
    }
    Ok(())
}

/// The properties of a GICv3 that `kvm_arm_gicv3_realize()` looks at.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GicV3Config {
    /// The `revision` property: 3, or 4 which KVM does not do.
    pub revision: u32,
    /// The `has-security-extensions` property.
    pub security_extn: bool,
    /// The `has-nmi` property.
    pub nmi: bool,
    /// The `first-cpu-index` property.
    pub first_cpu_idx: u32,
    /// The `num-irq` property, SPIs plus 32.
    pub num_irq: u32,
    /// The distributor base.
    pub dist_base: u64,
    /// Each redistributor region as its base and the number of redistributors it holds.
    pub redist_regions: Vec<(u64, u32)>,
    /// The `maintenance-interrupt-id` property, 0 for none.
    pub maint_irq: u32,
}

/// The property checks at the top of `kvm_arm_gicv3_realize()`, before the device exists.
pub fn gicv3_check_config(cfg: &GicV3Config) -> Result<(), VgicError> {
    if cfg.revision != 3 {
        return Err(VgicError::config(format!(
            "unsupported GIC revision {} for in-kernel GIC",
            cfg.revision
        )));
    }
    if cfg.security_extn {
        return Err(VgicError::config(
            "the in-kernel VGICv3 does not implement the security extensions",
        ));
    }
    if cfg.nmi {
        return Err(VgicError::config("NMI is not supported with the in-kernel GIC"));
    }
    if cfg.first_cpu_idx != 0 {
        return Err(VgicError::config(
            "Non-zero first-cpu-idx is unsupported with the in-kernel GIC",
        ));
    }
    Ok(())
}

/// What the setup of an in-kernel GIC or ITS learned about the host kernel.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VgicSetup {
    /// The reasons migration cannot work, for the migration blockers.
    pub migration_blockers: Vec<String>,
    /// Whether the kernel can save and restore the device's registers. Without it a reset only
    /// resets ruvm's copy.
    pub can_save: bool,
    /// Whether the kernel has a flush command to run when the VM stops:
    /// `KVM_DEV_ARM_VGIC_SAVE_PENDING_TABLES` for the GIC, `KVM_DEV_ARM_ITS_SAVE_TABLES` for
    /// the ITS.
    pub flush_on_stop: bool,
}

/// The rest of `kvm_arm_gicv3_realize()` once the device exists, without the GSI routes, which
/// are a VM ioctl: the maintenance interrupt, the region check, the interrupt count, the init
/// command, the frame addresses and what the kernel can save. It also fills each CPU's
/// `kvm_reset_icc_ctlr_el1` from the kernel.
///
/// QEMU sets the addresses later, once the machine is done, through
/// `kvm_arm_register_device()`. ruvm knows them at this point and the kernel takes them any time
/// before the first `KVM_RUN`, so they are set here, regions in index order, which is the order
/// QEMU's list ends up applying them in.
pub fn gicv3_setup(
    dev: &mut impl VgicDevice,
    cfg: &GicV3Config,
    s: &mut GicV3State,
) -> Result<VgicSetup, VgicError> {
    let mut setup = VgicSetup::default();
    if cfg.maint_irq != 0 {
        setup
            .migration_blockers
            .push("Live migration disabled because KVM nested virt is enabled".to_string());
        dev.has(KVM_DEV_ARM_VGIC_GRP_MAINT_IRQ, 0).map_err(|e| {
            VgicError::Os("VGICv3 setting maintenance IRQ is not supported by this host kernel", e)
        })?;
        dev.set(KVM_DEV_ARM_VGIC_GRP_MAINT_IRQ, 0, u64::from(cfg.maint_irq))
            .map_err(|e| e.during("Failed to set VGIC maintenance IRQ"))?;
    }

    let multiple = dev.has(KVM_DEV_ARM_VGIC_GRP_ADDR, KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION).is_ok();
    if !multiple && cfg.redist_regions.len() > 1 {
        return Err(VgicError::Config(
            "Multiple VGICv3 redistributor regions are not supported by this host kernel"
                .to_string(),
            Some(format!("A maximum of {} VCPUs can be used", cfg.redist_regions[0].1)),
        ));
    }

    dev.set(KVM_DEV_ARM_VGIC_GRP_NR_IRQS, 0, u64::from(cfg.num_irq))?;
    dev.set(KVM_DEV_ARM_VGIC_GRP_CTRL, KVM_DEV_ARM_VGIC_CTRL_INIT, 0)?;

    let addr = |dev: &mut _, kind, value| {
        VgicDevice::set(dev, KVM_DEV_ARM_VGIC_GRP_ADDR, kind, value)
            .map_err(|e: VgicError| e.during("Failed to set device address"))
    };
    addr(dev, KVM_VGIC_V3_ADDR_TYPE_DIST, cfg.dist_base)?;
    if !multiple {
        if let Some(&(base, _)) = cfg.redist_regions.first() {
            addr(dev, KVM_VGIC_V3_ADDR_TYPE_REDIST, base)?;
        }
    } else {
        for (i, &(base, count)) in cfg.redist_regions.iter().enumerate() {
            addr(
                dev,
                KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION,
                redist_region_attr(base, i as u32, count),
            )?;
        }
    }

    setup.can_save = dev.has(KVM_DEV_ARM_VGIC_GRP_DIST_REGS, u64::from(GICD_CTLR)).is_ok();
    if !setup.can_save {
        setup
            .migration_blockers
            .push("This operating system kernel does not support vGICv3 migration".to_string());
    }
    setup.flush_on_stop =
        dev.has(KVM_DEV_ARM_VGIC_GRP_CTRL, KVM_DEV_ARM_VGIC_SAVE_PENDING_TABLES).is_ok();

    if setup.can_save {
        for c in &mut s.cpus {
            c.kvm_reset_icc_ctlr_el1 = gicc_get(dev, ICC_CTLR_EL1, c.gicr_typer)?;
        }
    }
    Ok(setup)
}

/// `vm_change_state_handler()`: flushes the LPI pending tables into guest memory when the VM
/// stops. QEMU reports a failure and carries on only for `EFAULT`, which the caller can tell
/// with [`VgicError::errno`].
pub fn gicv3_flush(dev: &mut impl VgicDevice) -> Result<(), VgicError> {
    dev.set(KVM_DEV_ARM_VGIC_GRP_CTRL, KVM_DEV_ARM_VGIC_SAVE_PENDING_TABLES, 0)
}

/// One GICv2 CPU interface, plus the banked state of its SGIs and PPIs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GicV2Cpu {
    /// `GICC_CTLR`.
    pub ctlr: u32,
    /// `GICC_PMR`.
    pub priority_mask: u32,
    /// `GICC_BPR`.
    pub bpr: u32,
    /// `GICC_ABPR`.
    pub abpr: u32,
    /// `GICC_APR<n>`.
    pub apr: [u32; 4],
    /// The priorities of this CPU's SGIs and PPIs, `priority1[irq][cpu]`.
    pub priority: [u8; 32],
    /// For each SGI, the CPUs it is pending from, `sgi_pending[irq][cpu]`.
    pub sgi_pending: [u8; 16],
}

/// A GICv2 as arm_gic_kvm.c saves it. The per interrupt flags are CPU masks as in QEMU's
/// `irq_state`: a bit per CPU for the SGIs and PPIs and all bits for an SPI.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GicV2State {
    /// The number of interrupts.
    pub num_irq: u32,
    /// `GICD_CTLR`.
    pub ctlr: u32,
    /// Group 1 membership.
    pub group: Vec<u8>,
    /// Enabled.
    pub enabled: Vec<u8>,
    /// Pending.
    pub pending: Vec<u8>,
    /// Active.
    pub active: Vec<u8>,
    /// Edge triggered.
    pub edge_trigger: Vec<bool>,
    /// The SPI priorities, indexed by interrupt; the first 32 are unused.
    pub priority: Vec<u8>,
    /// `GICD_ITARGETSR<n>`, one byte per interrupt.
    pub targets: Vec<u8>,
    /// One per CPU interface.
    pub cpus: Vec<GicV2Cpu>,
}

impl GicV2State {
    /// A GIC with `num_irq` interrupts and `num_cpu` interfaces, all zero.
    pub fn new(num_irq: u32, num_cpu: u32) -> Self {
        let mut s = GicV2State::default();
        s.resize(num_irq, num_cpu);
        s
    }

    fn resize(&mut self, num_irq: u32, num_cpu: u32) {
        let n = num_irq as usize;
        self.num_irq = num_irq;
        self.group.resize(n, 0);
        self.enabled.resize(n, 0);
        self.pending.resize(n, 0);
        self.active.resize(n, 0);
        self.edge_trigger.resize(n, false);
        self.priority.resize(n, 0);
        self.targets.resize(n, 0);
        self.cpus.resize(num_cpu as usize, GicV2Cpu::default());
    }

    fn num_cpu(&self) -> u32 {
        self.cpus.len() as u32
    }
}

/// The CPU mask of interrupt `irq` as seen from `cpu`.
fn cpu_mask(irq: u32, cpu: u32) -> u8 {
    if irq < GIC_INTERNAL { 1 << cpu } else { ALL_CPU_MASK }
}

fn v2_gicd_get(dev: &mut impl VgicDevice, offset: u32, cpu: u32) -> Result<u32, VgicError> {
    Ok(dev.get(KVM_DEV_ARM_VGIC_GRP_DIST_REGS, v2_attr(offset, cpu))? as u32)
}

fn v2_gicd_set(
    dev: &mut impl VgicDevice,
    offset: u32,
    cpu: u32,
    value: u32,
) -> Result<(), VgicError> {
    dev.set(KVM_DEV_ARM_VGIC_GRP_DIST_REGS, v2_attr(offset, cpu), u64::from(value))
}

fn v2_gicc_get(dev: &mut impl VgicDevice, offset: u32, cpu: u32) -> Result<u32, VgicError> {
    Ok(dev.get(KVM_DEV_ARM_VGIC_GRP_CPU_REGS, v2_attr(offset, cpu))? as u32)
}

fn v2_gicc_set(
    dev: &mut impl VgicDevice,
    offset: u32,
    cpu: u32,
    value: u32,
) -> Result<(), VgicError> {
    dev.set(KVM_DEV_ARM_VGIC_GRP_CPU_REGS, v2_attr(offset, cpu), u64::from(value))
}

/// `kvm_dist_put()`: a register group of `width` bits per interrupt for the first `max_irq`
/// interrupts. Registers that cover banked interrupts are written once per CPU.
fn v2_dist_put(
    dev: &mut impl VgicDevice,
    num_cpu: u32,
    offset: u32,
    width: u32,
    max_irq: u32,
    mut field: impl FnMut(u32, u32) -> u32,
) -> Result<(), VgicError> {
    let per_reg = 32 / width;
    let mask = (1u32 << width) - 1;
    for i in 0..max_irq / per_reg {
        let irq = i * per_reg;
        let mut cpu = 0;
        while (cpu < num_cpu && irq < GIC_INTERNAL) || cpu == 0 {
            let mut reg = 0;
            for j in 0..per_reg {
                reg |= (field(irq + j, cpu) & mask) << (j * width);
            }
            v2_gicd_set(dev, offset + 4 * i, cpu, reg)?;
            cpu += 1;
        }
    }
    Ok(())
}

/// `kvm_dist_get()`.
fn v2_dist_get(
    dev: &mut impl VgicDevice,
    num_cpu: u32,
    offset: u32,
    width: u32,
    max_irq: u32,
    mut field: impl FnMut(u32, u32, u32),
) -> Result<(), VgicError> {
    let per_reg = 32 / width;
    let mask = (1u32 << width) - 1;
    for i in 0..max_irq / per_reg {
        let irq = i * per_reg;
        let mut cpu = 0;
        while (cpu < num_cpu && irq < GIC_INTERNAL) || cpu == 0 {
            let reg = v2_gicd_get(dev, offset + 4 * i, cpu)?;
            for j in 0..per_reg {
                field(irq + j, cpu, (reg >> (j * width)) & mask);
            }
            cpu += 1;
        }
    }
    Ok(())
}

fn v2_priority(s: &GicV2State, irq: u32, cpu: u32) -> u8 {
    if irq < GIC_INTERNAL {
        s.cpus[cpu as usize].priority[irq as usize]
    } else {
        s.priority[irq as usize]
    }
}

/// `kvm_arm_gic_put()`.
pub fn gicv2_put(dev: &mut impl VgicDevice, s: &GicV2State) -> Result<(), VgicError> {
    v2_gicd_set(dev, GICD_CTLR, 0, s.ctlr)?;
    let typer = v2_gicd_get(dev, GICD_TYPER, 0)?;
    let (num_irq, num_cpu) = (typer_num_irqs(typer), typer_num_cpus(typer));
    if num_irq < s.num_irq {
        return Err(VgicError::config(format!(
            "Restoring {} IRQs, but kernel supports max {num_irq}",
            s.num_irq
        )));
    }
    if num_cpu != s.num_cpu() {
        return Err(VgicError::config(format!(
            "Restoring {} CPU interfaces, kernel only has {num_cpu}",
            s.num_cpu()
        )));
    }
    let (n, cpus) = (s.num_irq, s.num_cpu());
    let flag = |v: &[u8], irq: u32, cpu: u32| u32::from(v[irq as usize] & cpu_mask(irq, cpu) != 0);

    v2_dist_put(dev, cpus, GICD_ICENABLER, 1, n, |_, _| !0)?;
    v2_dist_put(dev, cpus, GICD_ISENABLER, 1, n, |irq, cpu| flag(&s.enabled, irq, cpu))?;
    v2_dist_put(dev, cpus, GICD_IGROUPR, 1, n, |irq, cpu| flag(&s.group, irq, cpu))?;
    // Targets go before pending so the pending state lands on the right CPU interfaces, and the
    // configuration goes before it so level and edge are right.
    v2_dist_put(dev, cpus, GICD_ITARGETSR, 8, n, |irq, _| u32::from(s.targets[irq as usize]))?;
    v2_dist_put(
        dev,
        cpus,
        GICD_ICFGR,
        2,
        n,
        |irq, _| if s.edge_trigger[irq as usize] { 2 } else { 0 },
    )?;
    v2_dist_put(dev, cpus, GICD_ICPENDR, 1, n, |_, _| !0)?;
    v2_dist_put(dev, cpus, GICD_ISPENDR, 1, n, |irq, cpu| flag(&s.pending, irq, cpu))?;
    v2_dist_put(dev, cpus, GICD_ICACTIVER, 1, n, |_, _| !0)?;
    v2_dist_put(dev, cpus, GICD_ISACTIVER, 1, n, |irq, cpu| flag(&s.active, irq, cpu))?;
    v2_dist_put(dev, cpus, GICD_IPRIORITYR, 8, n, |irq, cpu| u32::from(v2_priority(s, irq, cpu)))?;
    v2_dist_put(dev, cpus, GICD_CPENDSGIR, 8, GIC_NR_SGIS, |_, _| !0)?;
    v2_dist_put(dev, cpus, GICD_SPENDSGIR, 8, GIC_NR_SGIS, |irq, cpu| {
        u32::from(s.cpus[cpu as usize].sgi_pending[irq as usize])
    })?;

    for (cpu, c) in s.cpus.iter().enumerate() {
        let cpu = cpu as u32;
        v2_gicc_set(dev, GICC_CTLR, cpu, c.ctlr)?;
        v2_gicc_set(dev, GICC_PMR, cpu, c.priority_mask & 0xff)?;
        v2_gicc_set(dev, GICC_BPR, cpu, c.bpr & 0x7)?;
        v2_gicc_set(dev, GICC_ABPR, cpu, c.abpr & 0x7)?;
        for (i, &apr) in c.apr.iter().enumerate() {
            v2_gicc_set(dev, GICC_APR + 4 * i as u32, cpu, apr)?;
        }
    }
    Ok(())
}

/// `kvm_arm_gic_get()`. Like QEMU it takes the interrupt and CPU counts from the kernel.
pub fn gicv2_get(dev: &mut impl VgicDevice, s: &mut GicV2State) -> Result<(), VgicError> {
    s.ctlr = v2_gicd_get(dev, GICD_CTLR, 0)?;
    let typer = v2_gicd_get(dev, GICD_TYPER, 0)?;
    let num_irq = typer_num_irqs(typer);
    if num_irq > GIC_MAXIRQ {
        return Err(VgicError::config(format!(
            "Too many IRQs reported from the kernel: {num_irq}"
        )));
    }
    s.resize(num_irq, typer_num_cpus(typer));
    v2_gicd_get(dev, GICD_IIDR, 0)?;
    for v in [&mut s.group, &mut s.enabled, &mut s.pending, &mut s.active] {
        v.fill(0);
    }
    s.edge_trigger.fill(false);

    let (n, cpus) = (s.num_irq, s.num_cpu());
    let set = |v: &mut [u8], irq: u32, cpu: u32, bit: u32| {
        if bit & 1 != 0 {
            v[irq as usize] |= cpu_mask(irq, cpu);
        }
    };
    v2_dist_get(dev, cpus, GICD_IGROUPR, 1, n, |irq, cpu, f| set(&mut s.group, irq, cpu, f))?;
    v2_dist_get(dev, cpus, GICD_ISENABLER, 1, n, |irq, cpu, f| set(&mut s.enabled, irq, cpu, f))?;
    v2_dist_get(dev, cpus, GICD_ISPENDR, 1, n, |irq, cpu, f| set(&mut s.pending, irq, cpu, f))?;
    v2_dist_get(dev, cpus, GICD_ISACTIVER, 1, n, |irq, cpu, f| set(&mut s.active, irq, cpu, f))?;
    v2_dist_get(dev, cpus, GICD_ICFGR, 2, n, |irq, _, f| {
        if f & 2 != 0 {
            s.edge_trigger[irq as usize] = true;
        }
    })?;
    v2_dist_get(dev, cpus, GICD_IPRIORITYR, 8, n, |irq, cpu, f| {
        if irq < GIC_INTERNAL {
            s.cpus[cpu as usize].priority[irq as usize] = f as u8;
        } else {
            s.priority[irq as usize] = f as u8;
        }
    })?;
    v2_dist_get(dev, cpus, GICD_ITARGETSR, 8, n, |irq, _, f| s.targets[irq as usize] = f as u8)?;
    v2_dist_get(dev, cpus, GICD_CPENDSGIR, 8, GIC_NR_SGIS, |irq, cpu, f| {
        s.cpus[cpu as usize].sgi_pending[irq as usize] = f as u8;
    })?;

    for (cpu, c) in s.cpus.iter_mut().enumerate() {
        let cpu = cpu as u32;
        c.ctlr = v2_gicc_get(dev, GICC_CTLR, cpu)?;
        c.priority_mask = v2_gicc_get(dev, GICC_PMR, cpu)? & 0xff;
        c.bpr = v2_gicc_get(dev, GICC_BPR, cpu)? & 0x7;
        c.abpr = v2_gicc_get(dev, GICC_ABPR, cpu)? & 0x7;
        for i in 0..4 {
            c.apr[i] = v2_gicc_get(dev, GICC_APR + 4 * i as u32, cpu)?;
        }
    }
    Ok(())
}

/// The property checks at the top of `kvm_arm_gic_realize()`.
pub fn gicv2_check_config(security_extn: bool, virt_extn: bool) -> Result<(), VgicError> {
    if security_extn {
        return Err(VgicError::config(
            "the in-kernel VGIC does not implement the security extensions",
        ));
    }
    if virt_extn {
        return Err(VgicError::config(
            "the in-kernel VGIC does not implement the virtualization extensions",
        ));
    }
    Ok(())
}

/// The rest of `kvm_arm_gic_realize()` once the device exists: the interrupt count and the init
/// command where the kernel has them, then the distributor and CPU interface addresses.
pub fn gicv2_setup(
    dev: &mut impl VgicDevice,
    num_irq: u32,
    dist_base: u64,
    cpu_base: u64,
) -> Result<(), VgicError> {
    if dev.has(KVM_DEV_ARM_VGIC_GRP_NR_IRQS, 0).is_ok() {
        dev.set(KVM_DEV_ARM_VGIC_GRP_NR_IRQS, 0, u64::from(num_irq))?;
    }
    if dev.has(KVM_DEV_ARM_VGIC_GRP_CTRL, KVM_DEV_ARM_VGIC_CTRL_INIT).is_ok() {
        dev.set(KVM_DEV_ARM_VGIC_GRP_CTRL, KVM_DEV_ARM_VGIC_CTRL_INIT, 0)?;
    }
    for (kind, base) in
        [(KVM_VGIC_V2_ADDR_TYPE_DIST, dist_base), (KVM_VGIC_V2_ADDR_TYPE_CPU, cpu_base)]
    {
        dev.set(KVM_DEV_ARM_VGIC_GRP_ADDR, kind, base)
            .map_err(|e| e.during("Failed to set device address"))?;
    }
    Ok(())
}

/// The ITS registers QEMU migrates, `GICv3ITSState`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ItsState {
    /// `GITS_CTLR`.
    pub ctlr: u64,
    /// `GITS_IIDR`.
    pub iidr: u64,
    /// `GITS_CBASER`.
    pub cbaser: u64,
    /// `GITS_CREADR`.
    pub creadr: u64,
    /// `GITS_CWRITER`.
    pub cwriter: u64,
    /// `GITS_BASER<n>`.
    pub baser: [u64; 8],
}

fn its_get(dev: &mut impl VgicDevice, reg: u64) -> Result<u64, VgicError> {
    dev.get(KVM_DEV_ARM_VGIC_GRP_ITS_REGS, reg)
}

fn its_set(dev: &mut impl VgicDevice, reg: u64, value: u64) -> Result<(), VgicError> {
    dev.set(KVM_DEV_ARM_VGIC_GRP_ITS_REGS, reg, value)
}

/// The rest of `kvm_arm_its_realize()` once the device exists: the init command, the base
/// address and what the kernel can save.
pub fn its_setup(dev: &mut impl VgicDevice, base: u64) -> Result<VgicSetup, VgicError> {
    dev.set(KVM_DEV_ARM_VGIC_GRP_CTRL, KVM_DEV_ARM_VGIC_CTRL_INIT, 0)?;
    dev.set(KVM_DEV_ARM_VGIC_GRP_ADDR, KVM_VGIC_ITS_ADDR_TYPE, base)
        .map_err(|e| e.during("Failed to set device address"))?;
    let can_save = dev.has(KVM_DEV_ARM_VGIC_GRP_ITS_REGS, GITS_CTLR).is_ok();
    let mut setup = VgicSetup { can_save, flush_on_stop: can_save, ..Default::default() };
    if !can_save {
        setup
            .migration_blockers
            .push("This operating system kernel does not support vITS migration".to_string());
    }
    Ok(setup)
}

/// `kvm_arm_its_pre_save()`.
pub fn its_save(dev: &mut impl VgicDevice, s: &mut ItsState) -> Result<(), VgicError> {
    for (i, b) in s.baser.iter_mut().enumerate() {
        *b = its_get(dev, GITS_BASER + 8 * i as u64)?;
    }
    s.ctlr = its_get(dev, GITS_CTLR)?;
    s.cbaser = its_get(dev, GITS_CBASER)?;
    s.creadr = its_get(dev, GITS_CREADR)?;
    s.cwriter = its_get(dev, GITS_CWRITER)?;
    s.iidr = its_get(dev, GITS_IIDR)?;
    Ok(())
}

/// `kvm_arm_its_post_load()`. `GITS_CBASER` goes before `GITS_CREADR` because writing it resets
/// the read pointer, and `GITS_CTLR` goes last, after the tables are back.
pub fn its_restore(dev: &mut impl VgicDevice, s: &ItsState) -> Result<(), VgicError> {
    its_set(dev, GITS_IIDR, s.iidr)?;
    its_set(dev, GITS_CBASER, s.cbaser)?;
    its_set(dev, GITS_CREADR, s.creadr)?;
    its_set(dev, GITS_CWRITER, s.cwriter)?;
    for (i, &b) in s.baser.iter().enumerate() {
        its_set(dev, GITS_BASER + 8 * i as u64, b)?;
    }
    dev.set(KVM_DEV_ARM_VGIC_GRP_CTRL, KVM_DEV_ARM_ITS_RESTORE_TABLES, 0)?;
    its_set(dev, GITS_CTLR, s.ctlr)
}

/// `kvm_arm_its_reset_hold()`. `s` is the ITS after ruvm's own reset. Gives back the warning
/// QEMU prints when the kernel has no reset command.
pub fn its_reset(
    dev: &mut impl VgicDevice,
    s: &ItsState,
) -> Result<Option<&'static str>, VgicError> {
    if dev.has(KVM_DEV_ARM_VGIC_GRP_CTRL, KVM_DEV_ARM_ITS_CTRL_RESET).is_ok() {
        dev.set(KVM_DEV_ARM_VGIC_GRP_CTRL, KVM_DEV_ARM_ITS_CTRL_RESET, 0)?;
        return Ok(None);
    }
    let warning = Some("ITS KVM: full reset is not supported by the host kernel");
    if dev.has(KVM_DEV_ARM_VGIC_GRP_ITS_REGS, GITS_CTLR).is_err() {
        return Ok(warning);
    }
    its_set(dev, GITS_CTLR, s.ctlr)?;
    its_set(dev, GITS_CBASER, s.cbaser)?;
    for (i, &b) in s.baser.iter().enumerate() {
        its_set(dev, GITS_BASER + 8 * i as u64, b)?;
    }
    Ok(warning)
}

/// The ITS `vm_change_state_handler()`: writes the ITS tables into guest memory when the VM
/// stops. QEMU reports a failure and carries on.
pub fn its_flush(dev: &mut impl VgicDevice) -> Result<(), VgicError> {
    dev.set(KVM_DEV_ARM_VGIC_GRP_CTRL, KVM_DEV_ARM_ITS_SAVE_TABLES, 0)
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::*;

    /// A device that remembers what was written, answers reads from that, and logs every access.
    #[derive(Default)]
    struct Fake {
        regs: HashMap<(u32, u64), u64>,
        has: HashSet<(u32, u64)>,
        log: Vec<(bool, u32, u64, u64)>,
    }

    impl Fake {
        fn with_typer(typer: u32) -> Self {
            let mut f = Fake::default();
            f.regs
                .insert((KVM_DEV_ARM_VGIC_GRP_DIST_REGS, u64::from(GICD_TYPER)), u64::from(typer));
            f
        }

        fn writes(&self) -> Vec<(u32, u64, u64)> {
            self.log.iter().filter(|a| a.0).map(|&(_, g, a, v)| (g, a, v)).collect()
        }
    }

    impl VgicDevice for Fake {
        fn get(&mut self, group: u32, attr: u64) -> Result<u64, VgicError> {
            let v = self.regs.get(&(group, attr)).copied().unwrap_or(0);
            self.log.push((false, group, attr, v));
            Ok(v)
        }

        fn set(&mut self, group: u32, attr: u64, value: u64) -> Result<(), VgicError> {
            let value = if attr_is_64bit(group) { value } else { value & 0xffff_ffff };
            self.log.push((true, group, attr, value));
            self.regs.insert((group, attr), value);
            // The GICv2 SGI pending registers are one state seen through a set and a clear
            // register, and a read of either gives it.
            let off = (attr & KVM_DEV_ARM_VGIC_OFFSET_MASK) as u32;
            if group == KVM_DEV_ARM_VGIC_GRP_DIST_REGS
                && (GICD_CPENDSGIR..GICD_SPENDSGIR + 16).contains(&off)
            {
                let base = attr - u64::from(off) + u64::from(GICD_CPENDSGIR + off % 16);
                let old = self.regs.get(&(group, base)).copied().unwrap_or(0);
                let new = if off < GICD_SPENDSGIR { old & !value } else { old | value };
                self.regs.insert((group, base), new);
            }
            Ok(())
        }

        fn has(&self, group: u32, attr: u64) -> io::Result<()> {
            if self.has.contains(&(group, attr)) {
                Ok(())
            } else {
                Err(io::Error::from_raw_os_error(6))
            }
        }
    }

    #[test]
    fn shuffles_are_inverse() {
        assert_eq!(half_shuffle32(0xffff), 0x5555_5555);
        assert_eq!(half_shuffle32(0x8001), 0x4000_0001);
        for x in [0u32, 1, 0x1234, 0xffff, 0xa5a5] {
            assert_eq!(half_unshuffle32(half_shuffle32(x)), x);
        }
        assert_eq!(half_unshuffle32(0xaaaa_aaaa >> 1), 0xffff);
    }

    #[test]
    fn irq_lines() {
        // 96 inputs: 64 SPIs, then 32 PPIs per CPU.
        assert_eq!(gic_irq_line(96, 0), 0x0100_0020);
        assert_eq!(gic_irq_line(96, 63), 0x0100_005f);
        assert_eq!(gic_irq_line(96, 64 + 27), 0x0200_001b);
        assert_eq!(gic_irq_line(96, 64 + 32 + 30), 0x0201_001e);
        assert_eq!(irq_line(300, KVM_ARM_IRQ_TYPE_PPI, 27), 0x1200_0000 | (44 << 16) | 27);
        assert_eq!(cpu_irq_line(1, false), 0x0001_0000);
        assert_eq!(cpu_irq_line(1, true), 0x0001_0001);
        assert_eq!(msi_data_to_gsi(32 + 5), 5);
    }

    #[test]
    fn attributes() {
        assert_eq!(ICC_PMR_EL1, 0xc230);
        assert_eq!(ICC_CTLR_EL1, 0xc664);
        assert_eq!(icc_ap0r(2), 0xc646);
        assert_eq!(icc_ap1r(3), 0xc64b);
        let typer = gicr_typer(0x0100_0203, 5, true);
        assert_eq!(typer, 0x0000_0203_0100_0501);
        assert_eq!(v3_attr(0x14, typer), 0x0000_0203_0000_0014);
        // Aff3 sits at bit 32 of the MPIDR and moves down next to the others.
        assert_eq!(gicr_typer(0x01_0000_0000, 0, false), 0x0100_0000_0100_0000);
        assert_eq!(v3_line_level_attr(32, 0), 32);
        assert_eq!(v2_attr(0x100, 3), 0x0000_0003_0000_0100);
        assert_eq!(redist_region_attr(0x80a_0000, 1, 123), 0x07b0_0000_080a_0001);
        assert_eq!(its_translater_gpa(0x808_0000), 0x809_0040);
        assert_eq!(typer_num_irqs(0x7), 256);
        assert_eq!(typer_num_cpus(0xe0), 8);
    }

    #[test]
    fn irqchip_choice() {
        assert!(matches!(
            irqchip_needs_create(KernelIrqchip::Split, true),
            Err(KvmError::ArmSplitIrqchip)
        ));
        assert_eq!(
            KvmError::ArmSplitIrqchip.to_string(),
            "-machine kernel_irqchip=split is not supported on ARM."
        );
        assert!(!irqchip_needs_create(KernelIrqchip::On, true).unwrap());
        assert!(irqchip_needs_create(KernelIrqchip::On, false).unwrap());
    }

    #[test]
    fn device_irqs() {
        let c = device_irq_changes(0, KVM_ARM_DEV_EL1_VTIMER | 8);
        assert_eq!(c.changed, vec![(DeviceIrq::VirtTimer, true)]);
        assert_eq!(c.unhandled, 8);
        let c = device_irq_changes(KVM_ARM_DEV_EL1_PTIMER | KVM_ARM_DEV_PMU, KVM_ARM_DEV_PMU);
        assert_eq!(c.changed, vec![(DeviceIrq::PhysTimer, false)]);
        assert_eq!(c.unhandled, 0);
    }

    #[test]
    fn gic_version_choice() {
        let (both, name) = kvm_gics_supported(true, KVM_ARM_VGIC_V2 | KVM_ARM_VGIC_V3).unwrap();
        assert_eq!(name, "KVM");
        assert_eq!(
            finalize_gic_version(name, true, GicVersion::NoSel, both, 4).unwrap(),
            GicVersion::V2
        );
        assert_eq!(
            finalize_gic_version(name, true, GicVersion::NoSel, both, 9).unwrap(),
            GicVersion::V3
        );
        assert_eq!(
            finalize_gic_version(name, true, GicVersion::Host, both, 1).unwrap(),
            GicVersion::V3
        );
        assert_eq!(
            kvm_gics_supported(true, 0).unwrap_err().to_string(),
            "Unable to determine GIC version supported by host"
        );
        let (v2, off) = kvm_gics_supported(false, 0).unwrap();
        assert_eq!(
            finalize_gic_version(off, true, GicVersion::V3, v2, 1).unwrap_err().to_string(),
            "KVM with kernel-irqchip=off does not support GICv3 emulation"
        );
        assert_eq!(
            finalize_gic_version(off, true, GicVersion::NoSel, v2, 9).unwrap_err().to_string(),
            "KVM with kernel-irqchip=off only supports GICv2 emulation but more than 8 vcpus are requested"
        );
        assert_eq!(
            finalize_gic_version(name, true, GicVersion::V4, both, 1).unwrap_err().to_string(),
            "KVM does not support GICv4 emulation, is virtualization=on?"
        );
        assert_eq!(
            finalize_gic_version("TCG", false, GicVersion::Host, both, 1).unwrap_err().to_string(),
            "gic-version=host requires KVM"
        );
    }

    fn sample_v3() -> GicV3State {
        let typers = [gicr_typer(0, 0, true), gicr_typer(1, 1, true)];
        let mut s = GicV3State::new(96, &typers);
        s.reset(true, true);
        s.gicd_statusr = 3;
        s.enabled[1] = 0x8000_0001;
        s.enabled[2] = 0x10;
        s.pending[2] = 0x4;
        s.active[1] = 0x2;
        s.level[1] = 0x100;
        s.edge_trigger[1] = 0x0001_8000;
        s.edge_trigger[2] = 0xffff_0000;
        for (i, p) in s.gicd_ipriority.iter_mut().enumerate().skip(32) {
            *p = (i * 3) as u8;
        }
        s.gicd_irouter[40] = 0x0000_0001_0000_0203;
        for (n, c) in s.cpus.iter_mut().enumerate() {
            c.gicr_propbaser = 0x1234_5678_9000 + n as u64;
            c.gicr_pendbaser = 0x4000_0000_0000_0000 | n as u64;
            c.gicr_ienabler0 = 0xffff;
            c.edge_trigger = 0xffff | (1 << 27);
            c.level = 1 << 27;
            c.gicr_ipriorityr[27] = 0xa0;
            // Six priority bits.
            c.icc_ctlr_el1 = 5 << 8;
            c.icc_sre_el1 = 7;
            c.icc_igrpen = [0, 1];
            c.icc_pmr_el1 = 0xf0;
            c.icc_bpr = [2, 3];
            c.icc_apr = [[1, 2, 3, 4], [5, 6, 7, 8]];
        }
        s
    }

    #[test]
    fn gicv3_round_trip() {
        let s = sample_v3();
        let mut dev = Fake::with_typer(2);
        // The redistributor reports LPIs.
        dev.regs.insert(
            (
                KVM_DEV_ARM_VGIC_GRP_REDIST_REGS,
                v3_attr(u64::from(GICR_TYPER), s.cpus[0].gicr_typer),
            ),
            1,
        );
        gicv3_put(&mut dev, &s).unwrap();

        let typers: Vec<u64> = s.cpus.iter().map(|c| c.gicr_typer).collect();
        let mut back = GicV3State::new(96, &typers);
        gicv3_get(&mut dev, &mut back).unwrap();
        // The SGI and PPI configuration only has room for the PPIs, and the APRs past the
        // priority bits are not there.
        let mut want = s.clone();
        for c in &mut want.cpus {
            c.edge_trigger &= 0xffff_0000;
            c.icc_apr = [[1, 2, 0, 0], [5, 6, 0, 0]];
        }
        assert_eq!(back, want);
    }

    #[test]
    fn gicv3_put_order() {
        let s = sample_v3();
        let mut dev = Fake::with_typer(2);
        gicv3_put(&mut dev, &s).unwrap();
        let w = dev.writes();
        let dist = |off: u32| (KVM_DEV_ARM_VGIC_GRP_DIST_REGS, u64::from(off));
        assert_eq!((w[0].0, w[0].1), dist(GICD_CTLR));
        // No LPIs, so the first redistributor write is GICR_CTLR.
        assert_eq!(w[1].1, v3_attr(u64::from(GICR_CTLR), s.cpus[0].gicr_typer));
        // The enable bitmap clears before it sets, per register, from interrupt 32.
        let at = w.iter().position(|a| (a.0, a.1) == dist(GICD_ICENABLER + 4)).unwrap();
        assert_eq!(w[at].2, 0xffff_ffff);
        assert_eq!(
            w[at + 1],
            (KVM_DEV_ARM_VGIC_GRP_DIST_REGS, u64::from(GICD_ISENABLER + 4), 0x8000_0001)
        );
        // Routers are eight bytes apart.
        assert!(w.contains(&(
            KVM_DEV_ARM_VGIC_GRP_DIST_REGS,
            u64::from(GICD_IROUTER + 8 * 40),
            0x203
        )));
        assert!(w.contains(&(
            KVM_DEV_ARM_VGIC_GRP_DIST_REGS,
            u64::from(GICD_IROUTER + 8 * 40 + 4),
            1
        )));
        // Interrupts 47 and 48 are edge triggered, the last of one register and the first of the
        // next.
        assert!(w.contains(&(
            KVM_DEV_ARM_VGIC_GRP_DIST_REGS,
            u64::from(GICD_ICFGR + 8),
            0x8000_0000
        )));
        assert!(w.contains(&(KVM_DEV_ARM_VGIC_GRP_DIST_REGS, u64::from(GICD_ICFGR + 12), 0x2)));
        // Six priority bits means two of each APR, highest first.
        let aprs: Vec<u64> = w
            .iter()
            .filter(|a| a.0 == KVM_DEV_ARM_VGIC_GRP_CPU_SYSREGS && a.1 & 0xffff_ffff != ICC_SRE_EL1)
            .map(|a| a.1 & 0xffff_ffff)
            .skip(6)
            .take(4)
            .collect();
        assert_eq!(aprs, vec![icc_ap0r(1), icc_ap0r(0), icc_ap1r(1), icc_ap1r(0)]);
    }

    #[test]
    fn gicv3_checks_irqs() {
        let s = sample_v3();
        let mut dev = Fake::with_typer(1);
        assert_eq!(
            gicv3_put(&mut dev, &s).unwrap_err().to_string(),
            "Model requests 96 IRQs, but kernel supports max 64"
        );
    }

    #[test]
    fn gicv3_setup_sequence() {
        let typers = [gicr_typer(0, 0, false)];
        let mut s = GicV3State::new(288, &typers);
        let cfg = GicV3Config {
            revision: 3,
            num_irq: 288,
            dist_base: 0x800_0000,
            redist_regions: vec![(0x80a_0000, 123), (0x40_0000_0000, 4)],
            ..Default::default()
        };
        gicv3_check_config(&cfg).unwrap();

        let mut dev = Fake::default();
        let e = gicv3_setup(&mut dev, &cfg, &mut s).unwrap_err();
        assert_eq!(
            e.to_string(),
            "Multiple VGICv3 redistributor regions are not supported by this host kernel"
        );
        assert_eq!(e.hint(), Some("A maximum of 123 VCPUs can be used"));

        let mut dev = Fake::default();
        dev.has.insert((KVM_DEV_ARM_VGIC_GRP_ADDR, KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION));
        dev.has.insert((KVM_DEV_ARM_VGIC_GRP_DIST_REGS, 0));
        dev.regs.insert((KVM_DEV_ARM_VGIC_GRP_CPU_SYSREGS, ICC_CTLR_EL1), 0x0040_0700);
        let setup = gicv3_setup(&mut dev, &cfg, &mut s).unwrap();
        assert!(setup.can_save);
        assert!(!setup.flush_on_stop);
        assert!(setup.migration_blockers.is_empty());
        assert_eq!(s.cpus[0].kvm_reset_icc_ctlr_el1, 0x0040_0700);
        assert_eq!(
            dev.writes(),
            vec![
                (KVM_DEV_ARM_VGIC_GRP_NR_IRQS, 0, 288),
                (KVM_DEV_ARM_VGIC_GRP_CTRL, KVM_DEV_ARM_VGIC_CTRL_INIT, 0),
                (KVM_DEV_ARM_VGIC_GRP_ADDR, KVM_VGIC_V3_ADDR_TYPE_DIST, 0x800_0000),
                (
                    KVM_DEV_ARM_VGIC_GRP_ADDR,
                    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION,
                    0x07b0_0000_080a_0000
                ),
                (
                    KVM_DEV_ARM_VGIC_GRP_ADDR,
                    KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION,
                    0x0040_0040_0000_0001
                ),
            ]
        );
        s.cpus[0].icc_reset();
        assert_eq!(s.cpus[0].icc_ctlr_el1, 0x0040_0700);
        assert_eq!(s.cpus[0].icc_sre_el1, 7);

        let mut nv = cfg.clone();
        nv.maint_irq = 25;
        let e = gicv3_setup(&mut Fake::default(), &nv, &mut s).unwrap_err();
        assert_eq!(
            e.to_string(),
            "VGICv3 setting maintenance IRQ is not supported by this host kernel: No such device or address"
        );
        let mut bad = cfg;
        bad.security_extn = true;
        assert_eq!(
            gicv3_check_config(&bad).unwrap_err().to_string(),
            "the in-kernel VGICv3 does not implement the security extensions"
        );
    }

    #[test]
    fn gicv2_round_trip() {
        let mut s = GicV2State::new(64, 2);
        s.ctlr = 1;
        s.enabled[1] = 0b10;
        s.enabled[40] = ALL_CPU_MASK;
        s.pending[33] = ALL_CPU_MASK;
        s.group[27] = 0b11;
        s.edge_trigger[35] = true;
        s.priority[40] = 0xa0;
        s.targets[40] = 0b10;
        s.cpus[1].priority[27] = 0x80;
        s.cpus[1].sgi_pending[3] = 0b01;
        s.cpus[0].priority_mask = 0xf0;
        s.cpus[0].apr = [1, 2, 3, 4];
        let mut dev = Fake::with_typer(1 | (1 << 5));
        gicv2_put(&mut dev, &s).unwrap();
        // Banked registers are written once per CPU, the SPI ones once.
        let w = dev.writes();
        assert!(w.contains(&(KVM_DEV_ARM_VGIC_GRP_DIST_REGS, v2_attr(GICD_ISENABLER, 1), 0b10)));
        assert!(w.contains(&(KVM_DEV_ARM_VGIC_GRP_DIST_REGS, v2_attr(GICD_ISENABLER, 0), 0)));
        assert!(!w.iter().any(|a| a.1 == v2_attr(GICD_ISENABLER + 4, 1)));

        let mut back = GicV2State::default();
        gicv2_get(&mut dev, &mut back).unwrap();
        assert_eq!(back, s);

        let mut dev = Fake::with_typer(1);
        assert_eq!(
            gicv2_put(&mut dev, &s).unwrap_err().to_string(),
            "Restoring 2 CPU interfaces, kernel only has 1"
        );
    }

    #[test]
    fn its_sequences() {
        let s = ItsState { ctlr: 1, iidr: 2, cbaser: 3, creadr: 4, cwriter: 5, baser: [6; 8] };
        let mut dev = Fake::default();
        its_restore(&mut dev, &s).unwrap();
        let w = dev.writes();
        assert_eq!(w[0], (KVM_DEV_ARM_VGIC_GRP_ITS_REGS, GITS_IIDR, 2));
        assert_eq!(w[1], (KVM_DEV_ARM_VGIC_GRP_ITS_REGS, GITS_CBASER, 3));
        assert_eq!(w[12], (KVM_DEV_ARM_VGIC_GRP_CTRL, KVM_DEV_ARM_ITS_RESTORE_TABLES, 0));
        assert_eq!(w[13], (KVM_DEV_ARM_VGIC_GRP_ITS_REGS, GITS_CTLR, 1));
        let mut back = ItsState::default();
        its_save(&mut dev, &mut back).unwrap();
        assert_eq!(back, s);

        let mut dev = Fake::default();
        assert_eq!(
            its_reset(&mut dev, &s).unwrap(),
            Some("ITS KVM: full reset is not supported by the host kernel")
        );
        assert!(dev.writes().is_empty());
        dev.has.insert((KVM_DEV_ARM_VGIC_GRP_CTRL, KVM_DEV_ARM_ITS_CTRL_RESET));
        assert_eq!(its_reset(&mut dev, &s).unwrap(), None);
        assert_eq!(dev.writes(), vec![(KVM_DEV_ARM_VGIC_GRP_CTRL, KVM_DEV_ARM_ITS_CTRL_RESET, 0)]);

        let setup = its_setup(&mut Fake::default(), 0x808_0000).unwrap();
        assert_eq!(
            setup.migration_blockers,
            vec!["This operating system kernel does not support vITS migration"]
        );
    }

    #[test]
    fn access_error_text() {
        let e = VgicError::Access {
            write: true,
            group: 1,
            attr: 0x14,
            err: io::Error::from_raw_os_error(22),
        };
        assert_eq!(
            e.to_string(),
            "KVM_SET_DEVICE_ATTR failed: Group 1 attr 0x0000000000000014: Invalid argument"
        );
        assert_eq!(e.errno(), Some(22));
    }
}
