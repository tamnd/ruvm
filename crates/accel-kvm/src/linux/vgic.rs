// SPDX-License-Identifier: GPL-2.0-or-later

//! The in-kernel vGIC and ITS on AArch64 Linux: the device file descriptors behind the sequences
//! in [`crate::arm`], the GSI routes, the interrupt lines and MSIs, from hw/intc/arm_gicv3_kvm.c,
//! hw/intc/arm_gic_kvm.c, hw/intc/arm_gicv3_its_kvm.c and target/arm/kvm.c.

use std::io;
use std::os::fd::AsRawFd;
use std::sync::Arc;

use kvm_bindings::{
    KVM_IRQ_ROUTING_IRQCHIP, KvmIrqRouting, kvm_create_device, kvm_device_attr,
    kvm_irq_routing_entry, kvm_msi,
};
use kvm_ioctls::{DeviceFd, VmFd};

use super::{KvmAccel, os_error};
use crate::arm::{
    self, GIC_INTERNAL, GicV2State, GicV3Config, GicV3State, ItsState, VgicDevice, VgicError,
    VgicSetup,
};
use crate::strerror;

const _: () = {
    assert!(
        arm::KVM_DEV_TYPE_ARM_VGIC_V2 == kvm_bindings::kvm_device_type_KVM_DEV_TYPE_ARM_VGIC_V2
    );
    assert!(
        arm::KVM_DEV_TYPE_ARM_VGIC_V3 == kvm_bindings::kvm_device_type_KVM_DEV_TYPE_ARM_VGIC_V3
    );
    assert!(
        arm::KVM_DEV_TYPE_ARM_VGIC_ITS == kvm_bindings::kvm_device_type_KVM_DEV_TYPE_ARM_VGIC_ITS
    );
    assert!(arm::KVM_CREATE_DEVICE_TEST == kvm_bindings::KVM_CREATE_DEVICE_TEST);
    assert!(arm::KVM_DEV_ARM_VGIC_GRP_ADDR == kvm_bindings::KVM_DEV_ARM_VGIC_GRP_ADDR);
    assert!(arm::KVM_DEV_ARM_VGIC_GRP_DIST_REGS == kvm_bindings::KVM_DEV_ARM_VGIC_GRP_DIST_REGS);
    assert!(arm::KVM_DEV_ARM_VGIC_GRP_CPU_REGS == kvm_bindings::KVM_DEV_ARM_VGIC_GRP_CPU_REGS);
    assert!(arm::KVM_DEV_ARM_VGIC_GRP_NR_IRQS == kvm_bindings::KVM_DEV_ARM_VGIC_GRP_NR_IRQS);
    assert!(arm::KVM_DEV_ARM_VGIC_GRP_CTRL == kvm_bindings::KVM_DEV_ARM_VGIC_GRP_CTRL);
    assert!(
        arm::KVM_DEV_ARM_VGIC_GRP_REDIST_REGS == kvm_bindings::KVM_DEV_ARM_VGIC_GRP_REDIST_REGS
    );
    assert!(
        arm::KVM_DEV_ARM_VGIC_GRP_CPU_SYSREGS == kvm_bindings::KVM_DEV_ARM_VGIC_GRP_CPU_SYSREGS
    );
    assert!(arm::KVM_DEV_ARM_VGIC_GRP_LEVEL_INFO == kvm_bindings::KVM_DEV_ARM_VGIC_GRP_LEVEL_INFO);
    assert!(arm::KVM_DEV_ARM_VGIC_GRP_ITS_REGS == kvm_bindings::KVM_DEV_ARM_VGIC_GRP_ITS_REGS);
    assert!(arm::KVM_DEV_ARM_VGIC_GRP_MAINT_IRQ == kvm_bindings::KVM_DEV_ARM_VGIC_GRP_MAINT_IRQ);
    assert!(arm::KVM_DEV_ARM_VGIC_CTRL_INIT == kvm_bindings::KVM_DEV_ARM_VGIC_CTRL_INIT as u64);
    assert!(arm::KVM_DEV_ARM_ITS_SAVE_TABLES == kvm_bindings::KVM_DEV_ARM_ITS_SAVE_TABLES as u64);
    assert!(
        arm::KVM_DEV_ARM_ITS_RESTORE_TABLES == kvm_bindings::KVM_DEV_ARM_ITS_RESTORE_TABLES as u64
    );
    assert!(
        arm::KVM_DEV_ARM_VGIC_SAVE_PENDING_TABLES
            == kvm_bindings::KVM_DEV_ARM_VGIC_SAVE_PENDING_TABLES as u64
    );
    assert!(arm::KVM_DEV_ARM_ITS_CTRL_RESET == kvm_bindings::KVM_DEV_ARM_ITS_CTRL_RESET as u64);
    assert!(arm::KVM_VGIC_V2_ADDR_TYPE_DIST == kvm_bindings::KVM_VGIC_V2_ADDR_TYPE_DIST as u64);
    assert!(arm::KVM_VGIC_V2_ADDR_TYPE_CPU == kvm_bindings::KVM_VGIC_V2_ADDR_TYPE_CPU as u64);
    assert!(arm::KVM_VGIC_V3_ADDR_TYPE_DIST == kvm_bindings::KVM_VGIC_V3_ADDR_TYPE_DIST as u64);
    assert!(arm::KVM_VGIC_V3_ADDR_TYPE_REDIST == kvm_bindings::KVM_VGIC_V3_ADDR_TYPE_REDIST as u64);
    assert!(arm::KVM_VGIC_ITS_ADDR_TYPE == kvm_bindings::KVM_VGIC_ITS_ADDR_TYPE as u64);
    assert!(
        arm::KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION
            == kvm_bindings::KVM_VGIC_V3_ADDR_TYPE_REDIST_REGION as u64
    );
    assert!(arm::KVM_DEV_ARM_VGIC_CPUID_SHIFT == kvm_bindings::KVM_DEV_ARM_VGIC_CPUID_SHIFT);
    assert!(
        arm::KVM_DEV_ARM_VGIC_LINE_LEVEL_INFO_SHIFT
            == kvm_bindings::KVM_DEV_ARM_VGIC_LINE_LEVEL_INFO_SHIFT
    );
    assert!(arm::KVM_ARM_IRQ_VCPU2_SHIFT == kvm_bindings::KVM_ARM_IRQ_VCPU2_SHIFT);
    assert!(arm::KVM_ARM_IRQ_TYPE_SHIFT == kvm_bindings::KVM_ARM_IRQ_TYPE_SHIFT);
    assert!(arm::KVM_ARM_IRQ_VCPU_SHIFT == kvm_bindings::KVM_ARM_IRQ_VCPU_SHIFT);
    assert!(arm::KVM_ARM_IRQ_TYPE_CPU == kvm_bindings::KVM_ARM_IRQ_TYPE_CPU);
    assert!(arm::KVM_ARM_IRQ_TYPE_SPI == kvm_bindings::KVM_ARM_IRQ_TYPE_SPI);
    assert!(arm::KVM_ARM_IRQ_TYPE_PPI == kvm_bindings::KVM_ARM_IRQ_TYPE_PPI);
    assert!(arm::KVM_ARM_IRQ_CPU_IRQ == kvm_bindings::KVM_ARM_IRQ_CPU_IRQ);
    assert!(arm::KVM_ARM_IRQ_CPU_FIQ == kvm_bindings::KVM_ARM_IRQ_CPU_FIQ);
    assert!(arm::KVM_ARM_DEV_EL1_VTIMER == kvm_bindings::KVM_ARM_DEV_EL1_VTIMER as u64);
    assert!(arm::KVM_ARM_DEV_EL1_PTIMER == kvm_bindings::KVM_ARM_DEV_EL1_PTIMER as u64);
    assert!(arm::KVM_ARM_DEV_PMU == kvm_bindings::KVM_ARM_DEV_PMU as u64);
    assert!(arm::KVM_MSI_VALID_DEVID == kvm_bindings::KVM_MSI_VALID_DEVID);
    assert!(arm::KVM_CAP_DEVICE_CTRL == kvm_bindings::KVM_CAP_DEVICE_CTRL);
    assert!(size_of::<kvm_create_device>() == 12);
};

/// `KVM_CREATE_DEVICE`, `_IOWR(KVMIO, 0xe0, struct kvm_create_device)`. kvm-ioctls only issues
/// it to create a device, and the probe needs the test flag, which makes no file descriptor.
const KVM_CREATE_DEVICE: u64 = (3 << 30) | (12 << 16) | (0xae << 8) | 0xe0;

/// A vGIC or ITS device file descriptor.
#[derive(Debug)]
struct VgicFd(DeviceFd);

impl VgicFd {
    fn create(vm: &VmFd, kind: u32) -> io::Result<Self> {
        let mut dev = kvm_create_device { type_: kind, fd: 0, flags: 0 };
        vm.create_device(&mut dev).map(VgicFd).map_err(os_error)
    }
}

fn access_error(write: bool, group: u32, attr: u64, e: kvm_ioctls::Error) -> VgicError {
    VgicError::Access { write, group, attr, err: os_error(e) }
}

impl VgicDevice for VgicFd {
    fn get(&mut self, group: u32, attr: u64) -> Result<u64, VgicError> {
        let mut value = 0u64;
        let mut a = kvm_device_attr { flags: 0, group, attr, addr: &raw mut value as u64 };
        // SAFETY: `addr` points at `value`, which outlives the call and has room for the 32 or
        // 64 bits the kernel writes for any vGIC group. On little endian AArch64 a 32-bit value
        // lands in the low half.
        unsafe { self.0.get_device_attr(&mut a) }
            .map_err(|e| access_error(false, group, attr, e))?;
        Ok(if arm::attr_is_64bit(group) { value } else { value & 0xffff_ffff })
    }

    fn set(&mut self, group: u32, attr: u64, value: u64) -> Result<(), VgicError> {
        let wide = value;
        let narrow = value as u32;
        let addr = if group == arm::KVM_DEV_ARM_VGIC_GRP_CTRL {
            0
        } else if arm::attr_is_64bit(group) {
            &raw const wide as u64
        } else {
            &raw const narrow as u64
        };
        let a = kvm_device_attr { flags: 0, group, attr, addr };
        self.0.set_device_attr(&a).map_err(|e| access_error(true, group, attr, e))
    }

    fn has(&self, group: u32, attr: u64) -> io::Result<()> {
        let a = kvm_device_attr { flags: 0, group, attr, addr: 0 };
        self.0.has_device_attr(&a).map_err(os_error)
    }
}

/// `kvm_arm_vgic_probe()`: which vGICs the host can create, as the `KVM_ARM_VGIC_V2` and
/// `KVM_ARM_VGIC_V3` bits. Nothing is created.
pub fn vgic_probe(accel: &KvmAccel) -> u32 {
    let device_ctrl =
        accel.kvm().check_extension_raw(libc::c_ulong::from(arm::KVM_CAP_DEVICE_CTRL)) > 0;
    if !device_ctrl {
        return 0;
    }
    let test = |kind: u32| {
        let mut dev = kvm_create_device { type_: kind, fd: 0, flags: arm::KVM_CREATE_DEVICE_TEST };
        // SAFETY: `dev` is a valid `kvm_create_device` that outlives the call. With the test flag
        // the kernel only checks the type and creates no file descriptor.
        unsafe { libc::ioctl(accel.vm().as_raw_fd(), KVM_CREATE_DEVICE as _, &raw mut dev) == 0 }
    };
    let mut val = 0;
    if test(arm::KVM_DEV_TYPE_ARM_VGIC_V3) {
        val |= arm::KVM_ARM_VGIC_V3;
    }
    if test(arm::KVM_DEV_TYPE_ARM_VGIC_V2) {
        val |= arm::KVM_ARM_VGIC_V2;
    }
    val
}

/// `kvm_irqchip_add_irq_route()` for every SPI and a commit: GSI `n` is SPI `n`, so irqfds and
/// MSIs that name a GSI reach the right input.
fn set_gsi_routes(vm: &VmFd, kvm: &kvm_ioctls::Kvm, num_irq: u32) -> Result<(), VgicError> {
    if kvm.check_extension_raw(libc::c_ulong::from(kvm_bindings::KVM_CAP_IRQ_ROUTING)) <= 0 {
        return Ok(());
    }
    let entries: Vec<kvm_irq_routing_entry> = (0..num_irq - GIC_INTERNAL)
        .map(|i| {
            let mut e = kvm_irq_routing_entry {
                gsi: i,
                type_: KVM_IRQ_ROUTING_IRQCHIP,
                ..Default::default()
            };
            e.u.irqchip.irqchip = 0;
            e.u.irqchip.pin = i;
            e
        })
        .collect();
    let routing = KvmIrqRouting::from_entries(&entries).map_err(|_| {
        VgicError::Os("KVM_SET_GSI_ROUTING", io::Error::from_raw_os_error(libc::ENOMEM))
    })?;
    vm.set_gsi_routing(&routing).map_err(|e| VgicError::Os("KVM_SET_GSI_ROUTING", os_error(e)))
}

/// An in-kernel GICv3, `KVMARMGICv3Class`. ruvm keeps the state, a [`GicV3State`], and moves it
/// in and out of the kernel for migration and reset.
#[derive(Debug)]
pub struct KvmGicV3 {
    dev: VgicFd,
    vm: Arc<VmFd>,
    num_irq: u32,
    setup: VgicSetup,
}

impl KvmGicV3 {
    /// `kvm_arm_gicv3_realize()`: checks the configuration, creates the device, sets it up and
    /// routes the GSIs. `state` holds a CPU for each vCPU, whose `kvm_reset_icc_ctlr_el1` this
    /// fills.
    pub fn new(
        accel: &KvmAccel,
        cfg: &GicV3Config,
        state: &mut GicV3State,
    ) -> Result<Self, VgicError> {
        arm::gicv3_check_config(cfg)?;
        let mut dev = VgicFd::create(accel.vm(), arm::KVM_DEV_TYPE_ARM_VGIC_V3)
            .map_err(|e| VgicError::Os("error creating in-kernel VGIC", e))?;
        let setup = arm::gicv3_setup(&mut dev, cfg, state)?;
        set_gsi_routes(accel.vm(), accel.kvm(), cfg.num_irq)?;
        Ok(KvmGicV3 { dev, vm: Arc::clone(accel.vm()), num_irq: cfg.num_irq, setup })
    }

    /// What the setup learned: the migration blockers and whether to flush on VM stop.
    pub fn setup(&self) -> &VgicSetup {
        &self.setup
    }

    /// `kvm_arm_gicv3_put()`.
    pub fn put(&mut self, state: &GicV3State) -> Result<(), VgicError> {
        arm::gicv3_put(&mut self.dev, state)
    }

    /// `kvm_arm_gicv3_get()`.
    pub fn get(&mut self, state: &mut GicV3State) -> Result<(), VgicError> {
        arm::gicv3_get(&mut self.dev, state)
    }

    /// `kvm_arm_gicv3_reset_hold()`: `state` has been through [`GicV3State::reset`] and goes
    /// into the kernel, unless the kernel cannot take it.
    pub fn reset(&mut self, state: &GicV3State) -> Result<(), VgicError> {
        if !self.setup.can_save {
            return Ok(());
        }
        self.put(state)
    }

    /// Flushes the LPI pending tables when the VM stops. See [`arm::gicv3_flush`].
    pub fn vm_stopped(&mut self) -> Result<(), VgicError> {
        if !self.setup.flush_on_stop {
            return Ok(());
        }
        arm::gicv3_flush(&mut self.dev)
    }

    /// `kvm_arm_gicv3_set_irq()`: drives GIC input `irq`, an SPI or a CPU's PPI.
    pub fn set_irq(&self, irq: u32, level: bool) -> io::Result<()> {
        self.vm.set_irq_line(arm::gic_irq_line(self.num_irq, irq), level).map_err(os_error)
    }
}

/// An in-kernel GICv2, `KVMARMGICClass`.
#[derive(Debug)]
pub struct KvmGicV2 {
    dev: VgicFd,
    vm: Arc<VmFd>,
    num_irq: u32,
}

impl KvmGicV2 {
    /// `kvm_arm_gic_realize()`: creates the device for `num_irq` interrupts with the
    /// distributor and CPU interface at the given addresses, and routes the GSIs.
    pub fn new(
        accel: &KvmAccel,
        num_irq: u32,
        dist_base: u64,
        cpu_base: u64,
        security_extn: bool,
        virt_extn: bool,
    ) -> Result<Self, VgicError> {
        arm::gicv2_check_config(security_extn, virt_extn)?;
        let mut dev = VgicFd::create(accel.vm(), arm::KVM_DEV_TYPE_ARM_VGIC_V2).map_err(|e| {
            VgicError::Config(
                format!("error creating in-kernel VGIC: {}", strerror(&e)),
                Some("Perhaps the host CPU does not support GICv2?".to_string()),
            )
        })?;
        arm::gicv2_setup(&mut dev, num_irq, dist_base, cpu_base)?;
        set_gsi_routes(accel.vm(), accel.kvm(), num_irq)?;
        Ok(KvmGicV2 { dev, vm: Arc::clone(accel.vm()), num_irq })
    }

    /// `kvm_arm_gic_put()`.
    pub fn put(&mut self, state: &GicV2State) -> Result<(), VgicError> {
        arm::gicv2_put(&mut self.dev, state)
    }

    /// `kvm_arm_gic_get()`.
    pub fn get(&mut self, state: &mut GicV2State) -> Result<(), VgicError> {
        arm::gicv2_get(&mut self.dev, state)
    }

    /// `kvm_arm_gic_reset_hold()`: the reset state goes into the kernel.
    pub fn reset(&mut self, state: &GicV2State) -> Result<(), VgicError> {
        self.put(state)
    }

    /// `kvm_arm_gic_set_irq()`.
    pub fn set_irq(&self, irq: u32, level: bool) -> io::Result<()> {
        self.vm.set_irq_line(arm::gic_irq_line(self.num_irq, irq), level).map_err(os_error)
    }
}

/// An in-kernel ITS, `KVMARMITSClass`.
#[derive(Debug)]
pub struct KvmIts {
    dev: VgicFd,
    vm: Arc<VmFd>,
    translater: u64,
    setup: VgicSetup,
}

impl KvmIts {
    /// `kvm_arm_its_realize()`: creates the ITS with its control frame at `base`. MSIs then
    /// carry a device ID and go through [`KvmIts::send_msi`].
    pub fn new(accel: &KvmAccel, base: u64) -> Result<Self, VgicError> {
        let mut dev = VgicFd::create(accel.vm(), arm::KVM_DEV_TYPE_ARM_VGIC_ITS)
            .map_err(|e| VgicError::Os("error creating in-kernel ITS", e))?;
        let setup = arm::its_setup(&mut dev, base)?;
        Ok(KvmIts {
            dev,
            vm: Arc::clone(accel.vm()),
            translater: arm::its_translater_gpa(base),
            setup,
        })
    }

    /// What the setup learned.
    pub fn setup(&self) -> &VgicSetup {
        &self.setup
    }

    /// `kvm_arm_its_pre_save()`.
    pub fn save(&mut self, state: &mut ItsState) -> Result<(), VgicError> {
        arm::its_save(&mut self.dev, state)
    }

    /// `kvm_arm_its_post_load()`.
    pub fn restore(&mut self, state: &ItsState) -> Result<(), VgicError> {
        arm::its_restore(&mut self.dev, state)
    }

    /// `kvm_arm_its_reset_hold()`. See [`arm::its_reset`].
    pub fn reset(&mut self, state: &ItsState) -> Result<Option<&'static str>, VgicError> {
        arm::its_reset(&mut self.dev, state)
    }

    /// Writes the ITS tables to guest memory when the VM stops.
    pub fn vm_stopped(&mut self) -> Result<(), VgicError> {
        if !self.setup.flush_on_stop {
            return Ok(());
        }
        arm::its_flush(&mut self.dev)
    }

    /// `kvm_its_send_msi()`: a write of `data` to `GITS_TRANSLATER` from device `devid`.
    pub fn send_msi(&self, data: u32, devid: u32) -> io::Result<()> {
        let msi = kvm_msi {
            address_lo: self.translater as u32,
            address_hi: (self.translater >> 32) as u32,
            data,
            flags: arm::KVM_MSI_VALID_DEVID,
            devid,
            ..Default::default()
        };
        self.vm.signal_msi(msi).map(drop).map_err(os_error)
    }
}

impl KvmAccel {
    /// `arm_cpu_kvm_set_irq()`: drives the IRQ or FIQ line of vCPU `cpu` when the GIC is a ruvm
    /// device rather than the kernel's.
    pub fn set_cpu_irq(&self, cpu: u32, fiq: bool, level: bool) -> io::Result<()> {
        self.vm().set_irq_line(arm::cpu_irq_line(cpu, fiq), level).map_err(os_error)
    }
}
