// SPDX-License-Identifier: GPL-2.0-or-later

//! The virt board on KVM: the KVM parts of hw/arm/virt.c, and the CPU reset of hw/arm/boot.c
//! for a vCPU whose registers live in the kernel.
//!
//! The board is built as for TCG, with one change when KVM keeps the GIC in the kernel: the
//! device interrupts go to a [`KvmSpis`] through [`VirtConfig::spi_sink`] instead of to the
//! board's GIC. The board's own GIC and ITS are still created but never reached, because the
//! kernel claims their frames. With `kernel-irqchip=off` the board's GICv2 is the GIC: KVM
//! reports the timer and PMU outputs after each exit, and the GIC drives the IRQ and FIQ lines
//! of the vCPUs.
//!
//! The run loop, `KvmVirtMachine`, only builds on Linux on AArch64. What can be checked on any
//! host is here.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use ruvm_accel_kvm::KernelIrqchip;
use ruvm_accel_kvm::arm::{DeviceIrq, GicV3Config, gicr_typer};
use ruvm_target_arm::cpu::{GTIMER_PHYS, GTIMER_VIRT};

use super::boot::BootInfo;
use super::cpus::{PMU_PPI, TIMER_PPIS};
use super::{
    SpiSink, VIRT_GIC_DIST, VIRT_GIC_NUM_IRQ, VIRT_GIC_REDIST, VIRT_GICV3_MAX_CPUS, VirtConfig,
    VirtGicVersion, virt_cpu_mp_affinity,
};

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
mod run;
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
pub use run::{KvmVirtMachine, prepare_config};

/// The `machvirt_init()` checks that depend on the accelerator being KVM, with QEMU's text.
/// EL2 for the guest is refused for now.
pub fn check_config(cfg: &VirtConfig, irqchip: KernelIrqchip) -> Result<(), String> {
    if cfg.secure {
        return Err("mach-virt: KVM does not support providing Security extensions \
                    (TrustZone) to the guest CPU"
            .to_string());
    }
    if cfg.virtualization {
        return Err("mach-virt: virtualization=on with KVM is not supported yet".to_string());
    }
    // finalize_gic_version() has the same check; this keeps the board safe from a caller
    // that skipped it.
    if irqchip == KernelIrqchip::Off && cfg.gic_version != VirtGicVersion::V2 {
        return Err("KVM with kernel-irqchip=off does not support GICv3 emulation".to_string());
    }
    Ok(())
}

/// The GIC interrupt ID a KVM device output drives when the GIC is in userspace: the timer
/// PPIs the board wires for TCG, and the PMU's.
pub fn device_irq_ppi(irq: DeviceIrq) -> u32 {
    match irq {
        DeviceIrq::VirtTimer => TIMER_PPIS[GTIMER_VIRT],
        DeviceIrq::PhysTimer => TIMER_PPIS[GTIMER_PHYS],
        DeviceIrq::Pmu => PMU_PPI,
    }
}

/// The PC `do_cpu_reset()` gives vCPU `idx`, or `None` to keep the one KVM resets it to.
/// A direct boot of anything but Linux enters it on every CPU; Linux enters the boot stub at
/// the start of RAM on CPU 0 and the others wait for PSCI `CPU_ON`.
pub fn boot_pc(info: &BootInfo, idx: usize) -> Option<u64> {
    if !info.direct {
        return None;
    }
    if !info.is_linux {
        return Some(info.entry);
    }
    (idx == 0).then_some(info.loader_start)
}

/// The in-kernel GICv3 of a virt board with `smp` CPUs, the properties `create_gicv3()` sets.
/// `redist2` is the base of the second redistributor region, used when the CPUs do not fit in
/// the first.
pub fn gicv3_config(smp: usize, redist2: Option<u64>) -> GicV3Config {
    let first = smp.min(VIRT_GICV3_MAX_CPUS);
    let mut redist_regions = vec![(VIRT_GIC_REDIST, first as u32)];
    if let Some(base) = redist2.filter(|_| smp > first) {
        redist_regions.push((base, (smp - first) as u32));
    }
    GicV3Config {
        revision: 3,
        num_irq: VIRT_GIC_NUM_IRQ,
        dist_base: VIRT_GIC_DIST,
        redist_regions,
        ..GicV3Config::default()
    }
}

/// The `GICR_TYPER` of each CPU, which tells KVM the redistributor by its affinity.
pub fn gicv3_typers(smp: usize) -> Vec<u64> {
    (0..smp).map(|i| gicr_typer(virt_cpu_mp_affinity(i), i as u32, false)).collect()
}

/// The board's SPIs on their way to an in-kernel GIC. KVM creates its GIC after the vCPUs,
/// which is after the board, so levels set before then are kept and replayed once the GIC is
/// connected.
#[derive(Clone, Default)]
pub struct KvmSpis(Arc<SpisInner>);

#[derive(Default)]
struct SpisInner {
    target: OnceLock<SpiSink>,
    early: Mutex<BTreeMap<u32, bool>>,
}

impl fmt::Debug for KvmSpis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KvmSpis").field("connected", &self.0.target.get().is_some()).finish()
    }
}

impl KvmSpis {
    /// Nothing connected yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// The sink to put in [`VirtConfig::spi_sink`].
    pub fn sink(&self) -> SpiSink {
        let inner = Arc::clone(&self.0);
        Arc::new(move |n, level| inner.set(n, level))
    }

    /// Sends every SPI to `target` from now on, starting with the levels set before. The GIC
    /// starts with every line low, so only the raised ones are replayed. Only the first call
    /// does anything.
    pub fn connect(&self, target: SpiSink) {
        let mut early = self.0.early.lock().unwrap_or_else(PoisonError::into_inner);
        if self.0.target.set(Arc::clone(&target)).is_err() {
            return;
        }
        for (&n, _) in early.iter().filter(|(_, l)| **l) {
            target(n, true);
        }
        early.clear();
    }
}

impl SpisInner {
    fn set(&self, n: u32, level: bool) {
        if let Some(t) = self.target.get() {
            return t(n, level);
        }
        let mut early = self.early.lock().unwrap_or_else(PoisonError::into_inner);
        // connect() may have finished while this waited for the lock.
        match self.target.get() {
            Some(t) => t(n, level),
            None => {
                early.insert(n, level);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(direct: bool, is_linux: bool) -> BootInfo {
        BootInfo {
            loader_start: 0x4000_0000,
            entry: 0x4020_0000,
            direct,
            is_linux,
            ..BootInfo::default()
        }
    }

    #[test]
    fn boot_pcs_follow_do_cpu_reset() {
        assert_eq!(boot_pc(&info(false, false), 0), None);
        assert_eq!(boot_pc(&info(true, false), 0), Some(0x4020_0000));
        assert_eq!(boot_pc(&info(true, false), 3), Some(0x4020_0000));
        assert_eq!(boot_pc(&info(true, true), 0), Some(0x4000_0000));
        assert_eq!(boot_pc(&info(true, true), 1), None);
    }

    #[test]
    fn device_outputs_hit_the_board_ppis() {
        assert_eq!(device_irq_ppi(DeviceIrq::VirtTimer), 27);
        assert_eq!(device_irq_ppi(DeviceIrq::PhysTimer), 30);
        assert_eq!(device_irq_ppi(DeviceIrq::Pmu), 23);
    }

    #[test]
    fn gicv3_config_splits_the_redistributors() {
        let c = gicv3_config(4, Some(0x40_0000_0000));
        assert_eq!(c.redist_regions, vec![(VIRT_GIC_REDIST, 4)]);
        assert_eq!(c.num_irq, 288);
        assert_eq!(c.dist_base, 0x0800_0000);
        assert_eq!(c.revision, 3);
        assert!(ruvm_accel_kvm::arm::gicv3_check_config(&c).is_ok());

        let c = gicv3_config(VIRT_GICV3_MAX_CPUS + 5, Some(0x40_0000_0000));
        assert_eq!(
            c.redist_regions,
            vec![(VIRT_GIC_REDIST, VIRT_GICV3_MAX_CPUS as u32), (0x40_0000_0000, 5)]
        );
    }

    #[test]
    fn typers_carry_the_cluster_affinity() {
        let t = gicv3_typers(18);
        assert_eq!(t[0] >> 32, 0);
        assert_eq!(t[15] >> 32, 15);
        // CPU 16 is the first of the second cluster, Aff1 = 1.
        assert_eq!(t[16] >> 32, 0x100);
        assert_eq!((t[17] >> 8) & 0xffff, 17);
    }

    #[test]
    fn config_checks_use_qemu_text() {
        let mut cfg = VirtConfig::new(ruvm_target_arm::cpu::ArmCpuModel::max());
        assert!(check_config(&cfg, KernelIrqchip::On).is_ok());
        assert_eq!(
            check_config(&cfg, KernelIrqchip::Off).unwrap_err(),
            "KVM with kernel-irqchip=off does not support GICv3 emulation"
        );
        cfg.gic_version = VirtGicVersion::V2;
        assert!(check_config(&cfg, KernelIrqchip::Off).is_ok());
        cfg.secure = true;
        assert!(check_config(&cfg, KernelIrqchip::On).unwrap_err().contains("TrustZone"));
    }

    #[test]
    fn early_levels_are_replayed() {
        let spis = KvmSpis::new();
        let sink = spis.sink();
        sink(3, true);
        sink(5, true);
        sink(5, false);
        sink(1, true);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = Arc::clone(&seen);
        spis.connect(Arc::new(move |n, l| s.lock().unwrap().push((n, l))));
        sink(7, true);
        sink(3, false);
        assert_eq!(*seen.lock().unwrap(), vec![(1, true), (3, true), (7, true), (3, false)]);
        // A second connect is ignored.
        spis.connect(Arc::new(|_, _| panic!("second target")));
        sink(2, true);
    }
}
