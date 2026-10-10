// SPDX-License-Identifier: GPL-2.0-or-later

//! The accelerator on a Mac: the VM, `hvf_accel_init()` and `hvf_arch_vm_create()`, and the
//! framework's GIC.

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::cpu::{self, HostIdRegs, feature};
use crate::{HvfError, HvfOptions, KernelIrqchip, hv};

mod ffi;
mod slots;
mod vcpu;

pub use slots::SlotListener;
pub use vcpu::{HvfExits, HvfKick, HvfVcpu};

/// Where the framework's GIC sits and how big it is, for the machine's device tree.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GicInfo {
    /// The distributor base.
    pub dist_base: u64,
    /// The distributor size.
    pub dist_size: u64,
    /// The redistributor region base.
    pub redist_base: u64,
    /// The size of the whole redistributor region.
    pub redist_region_size: u64,
    /// The size of one vCPU's redistributor.
    pub redist_size: u64,
    /// The first SPI.
    pub spi_base: u32,
    /// How many SPIs there are.
    pub spi_count: u32,
}

/// Only one VM per process: the framework's VM calls take no handle.
static VM_EXISTS: AtomicBool = AtomicBool::new(false);

/// An HVF virtual machine. Dropping it destroys the VM, so the vCPUs must be gone first.
#[derive(Debug)]
pub struct HvfAccel {
    irqchip: bool,
    el2: bool,
    ipa_bits: u32,
    gic: Option<GicInfo>,
    host: HostIdRegs,
    vtimer_offset: Arc<AtomicU64>,
    slots: Arc<SlotListener>,
}

/// Releases an OS object when dropped.
struct Released(ffi::OsObject);

impl Drop for Released {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: the object came from a framework create call and this is its only
            // reference.
            unsafe { ffi::os_release(self.0) };
        }
    }
}

/// `hvf_arch_get_max_ipa_bit_size()`: the largest IPA size the host allows, rounded down to
/// a PARange.
pub fn max_ipa_bits() -> Result<u32, HvfError> {
    let mut bits = 0;
    // SAFETY: the call writes one u32 through a valid pointer.
    let r = unsafe { ffi::hv_vm_config_get_max_ipa_size(&mut bits) };
    HvfError::check("hv_vm_config_get_max_ipa_size", r)?;
    Ok(cpu::parange_bits(bits))
}

/// `hvf_arch_get_default_ipa_bit_size()`.
pub fn default_ipa_bits() -> Result<u32, HvfError> {
    let mut bits = 0;
    // SAFETY: the call writes one u32 through a valid pointer.
    let r = unsafe { ffi::hv_vm_config_get_default_ipa_size(&mut bits) };
    HvfError::check("hv_vm_config_get_default_ipa_size", r)?;
    Ok(bits)
}

/// `hvf_arm_el2_supported()`: whether this Mac and macOS can run a guest at EL2.
pub fn el2_supported() -> bool {
    let Some(f) = ffi::macos15() else { return false };
    let mut ok = false;
    // SAFETY: the call writes one bool through a valid pointer.
    let r = unsafe { (f.vm_config_get_el2_supported)(&mut ok) };
    r == hv::SUCCESS && ok
}

/// The ID registers the framework reports for a vCPU, before any adjustment.
fn host_id_regs() -> Result<HostIdRegs, HvfError> {
    // SAFETY: no arguments; the result is an owned OS object or null.
    let config = Released(unsafe { ffi::hv_vcpu_config_create() });
    let get = |reg| {
        let mut v = 0;
        // SAFETY: the config is live for the closure's lifetime and the call writes one u64.
        let r = unsafe { ffi::hv_vcpu_config_get_feature_reg(config.0, reg, &mut v) };
        HvfError::check("hv_vcpu_config_get_feature_reg", r).map(|()| v)
    };
    Ok(HostIdRegs {
        pfr0: get(feature::ID_AA64PFR0_EL1)?,
        pfr1: get(feature::ID_AA64PFR1_EL1)?,
        dfr0: get(feature::ID_AA64DFR0_EL1)?,
        dfr1: get(feature::ID_AA64DFR1_EL1)?,
        isar0: get(feature::ID_AA64ISAR0_EL1)?,
        isar1: get(feature::ID_AA64ISAR1_EL1)?,
        mmfr0: get(feature::ID_AA64MMFR0_EL1)?,
        mmfr1: get(feature::ID_AA64MMFR1_EL1)?,
        mmfr2: get(feature::ID_AA64MMFR2_EL1)?,
    })
}

impl HvfAccel {
    /// Creates the VM, `hvf_accel_init()`. `irqchip_default` is the machine's
    /// `get_kernel_irqchip_default()`, used when the option does not say.
    pub fn new(opts: &HvfOptions, irqchip_default: bool) -> Result<HvfAccel, HvfError> {
        if VM_EXISTS.swap(true, Ordering::AcqRel) {
            return Err(HvfError::Call("hv_vm_create", hv::BUSY));
        }
        let r = Self::create(opts, irqchip_default);
        if r.is_err() {
            VM_EXISTS.store(false, Ordering::Release);
        }
        r
    }

    fn create(opts: &HvfOptions, irqchip_default: bool) -> Result<HvfAccel, HvfError> {
        let irqchip = match opts.kernel_irqchip {
            Some(k) => k == KernelIrqchip::On,
            None => irqchip_default,
        };
        // SAFETY: no arguments; the result is an owned OS object.
        let config = Released(unsafe { ffi::hv_vm_config_create() });
        // SAFETY: the config is live.
        let r = unsafe { ffi::hv_vm_config_set_ipa_size(config.0, opts.ipa_bits) };
        HvfError::check("hv_vm_config_set_ipa_size", r)?;
        if opts.el2 {
            if !el2_supported() {
                return Err(HvfError::NoNestedVirt);
            }
            let f = ffi::macos15().ok_or(HvfError::NoNestedVirt)?;
            // SAFETY: the config is live.
            if unsafe { (f.vm_config_set_el2_enabled)(config.0, true) } != hv::SUCCESS {
                return Err(HvfError::EnableNestedVirt);
            }
        }
        // SAFETY: the config is live; the framework copies what it needs from it.
        let r = unsafe { ffi::hv_vm_create(config.0) };
        if r == hv::DENIED {
            return Err(HvfError::Denied);
        }
        HvfError::check("hv_vm_create", r)?;
        let gic = if irqchip {
            match Self::create_gic(opts) {
                Ok(g) => Some(g),
                Err(e) => {
                    // SAFETY: the VM was created above and has no vCPUs yet.
                    unsafe { ffi::hv_vm_destroy() };
                    return Err(e);
                }
            }
        } else {
            None
        };
        let host = match host_id_regs() {
            Ok(h) => h,
            Err(e) => {
                // SAFETY: as above.
                unsafe { ffi::hv_vm_destroy() };
                return Err(e);
            }
        };
        // SAFETY: no arguments.
        let now = unsafe { ffi::mach_absolute_time() };
        Ok(HvfAccel {
            irqchip,
            el2: opts.el2,
            ipa_bits: opts.ipa_bits,
            gic,
            host,
            vtimer_offset: Arc::new(AtomicU64::new(now)),
            slots: Arc::new(SlotListener::new()),
        })
    }

    /// The GIC part of `hvf_arch_vm_create()`. It has to come after `hv_vm_create()` and
    /// before the first vCPU.
    fn create_gic(opts: &HvfOptions) -> Result<GicInfo, HvfError> {
        let f = ffi::macos15().ok_or(HvfError::Gic("HVF: Unsupported OS for platform vGIC."))?;
        // SAFETY: no arguments; the result is an owned OS object.
        let cfg = Released(unsafe { (f.gic_config_create)() });
        // SAFETY: the config is live and each call takes it with a plain address.
        let r = unsafe {
            (f.gic_config_set_distributor_base)(cfg.0, opts.gic_dist_base)
                | (f.gic_config_set_redistributor_base)(cfg.0, opts.gic_redist_base)
                | (f.gic_create)(cfg.0)
        };
        if r != hv::SUCCESS {
            return Err(HvfError::Gic("error creating platform VGIC"));
        }
        let (mut dist, mut region, mut redist) = (0usize, 0usize, 0usize);
        let (mut spi_base, mut spi_count) = (0u32, 0u32);
        // SAFETY: each call writes through one valid pointer.
        let r = unsafe {
            (f.gic_get_distributor_size)(&mut dist)
                | (f.gic_get_redistributor_region_size)(&mut region)
                | (f.gic_get_redistributor_size)(&mut redist)
                | (f.gic_get_spi_interrupt_range)(&mut spi_base, &mut spi_count)
        };
        HvfError::check("hv_gic_get_distributor_size", r)?;
        Ok(GicInfo {
            dist_base: opts.gic_dist_base,
            dist_size: dist as u64,
            redist_base: opts.gic_redist_base,
            redist_region_size: region as u64,
            redist_size: redist as u64,
            spi_base,
            spi_count,
        })
    }

    /// Whether the GIC is the framework's.
    pub fn irqchip_in_kernel(&self) -> bool {
        self.irqchip
    }

    /// Whether the guest runs at EL2.
    pub fn el2(&self) -> bool {
        self.el2
    }

    /// The VM's IPA size in bits.
    pub fn ipa_bits(&self) -> u32 {
        self.ipa_bits
    }

    /// The framework's GIC, when there is one.
    pub fn gic(&self) -> Option<&GicInfo> {
        self.gic.as_ref()
    }

    /// The host ID registers as the guest should see them, or `None` when the host CPU is
    /// one QEMU refuses.
    pub fn host_id_regs(&self) -> Option<HostIdRegs> {
        self.host.adjust(self.ipa_bits, self.irqchip, self.el2)
    }

    /// The listener that maps the guest's RAM. Register it on the system address space.
    pub fn slot_listener(&self) -> Arc<SlotListener> {
        Arc::clone(&self.slots)
    }

    /// The guest's CNTVCT, `hvf_vtimer_val_raw()`.
    pub fn vtimer_val(&self) -> u64 {
        // SAFETY: no arguments.
        unsafe { ffi::mach_absolute_time() }
            .wrapping_sub(self.vtimer_offset.load(Ordering::Relaxed))
    }

    /// Restarts the guest counter at `val`, as `hvf_vm_state_change()` does on resume. The
    /// vCPUs pick up the new offset on their next register put.
    pub fn set_vtimer_val(&self, val: u64) {
        // SAFETY: no arguments.
        let now = unsafe { ffi::mach_absolute_time() };
        self.vtimer_offset.store(now.wrapping_sub(val), Ordering::Relaxed);
    }

    /// Sets the level of an SPI on the framework's GIC.
    pub fn set_spi(&self, intid: u32, level: bool) -> Result<(), HvfError> {
        let f = self.gic_fns()?;
        // SAFETY: plain values.
        HvfError::check("hv_gic_set_spi", unsafe { (f.gic_set_spi)(intid, level) })
    }

    /// Sends an MSI to the framework's GIC.
    pub fn send_msi(&self, addr: u64, data: u32) -> Result<(), HvfError> {
        let f = self.gic_fns()?;
        // SAFETY: plain values.
        HvfError::check("hv_gic_send_msi", unsafe { (f.gic_send_msi)(addr, data) })
    }

    /// Resets the framework's GIC.
    pub fn gic_reset(&self) -> Result<(), HvfError> {
        let f = self.gic_fns()?;
        // SAFETY: no arguments.
        HvfError::check("hv_gic_reset", unsafe { (f.gic_reset)() })
    }

    /// The framework's GIC state as an opaque blob, for migration. The vCPUs must be stopped.
    pub fn gic_save(&self) -> Result<Vec<u8>, HvfError> {
        let f = self.gic_fns()?;
        // SAFETY: no arguments; the result is an owned OS object.
        let state = Released(unsafe { (f.gic_state_create)() });
        if state.0.is_null() {
            return Err(HvfError::Call("hv_gic_state_create", hv::ERROR));
        }
        let mut size = 0usize;
        // SAFETY: the state is live and the call writes one usize.
        HvfError::check("hv_gic_state_get_size", unsafe {
            (f.gic_state_get_size)(state.0, &mut size)
        })?;
        let mut data = vec![0u8; size];
        // SAFETY: the buffer holds exactly the size the framework just reported.
        HvfError::check("hv_gic_state_get_data", unsafe {
            (f.gic_state_get_data)(state.0, data.as_mut_ptr().cast::<c_void>())
        })?;
        Ok(data)
    }

    /// Loads a blob from [`HvfAccel::gic_save`] back into the framework's GIC.
    pub fn gic_restore(&self, data: &[u8]) -> Result<(), HvfError> {
        let f = self.gic_fns()?;
        // SAFETY: the framework reads `data.len()` bytes from the slice.
        HvfError::check("hv_gic_set_state", unsafe {
            (f.gic_set_state)(data.as_ptr().cast::<c_void>(), data.len())
        })
    }

    fn gic_fns(&self) -> Result<&'static ffi::Macos15, HvfError> {
        if self.gic.is_none() {
            return Err(HvfError::Call("hv_gic", hv::NO_DEVICE));
        }
        ffi::macos15().ok_or(HvfError::Call("hv_gic", hv::UNSUPPORTED))
    }

    pub(crate) fn vtimer_offset(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.vtimer_offset)
    }
}

impl Drop for HvfAccel {
    fn drop(&mut self) {
        // SAFETY: the VM exists. If a vCPU is still alive the framework refuses with HV_BUSY
        // and nothing changes, which is all a drop can do about it.
        unsafe { ffi::hv_vm_destroy() };
        VM_EXISTS.store(false, Ordering::Release);
    }
}
