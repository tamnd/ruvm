// SPDX-License-Identifier: GPL-2.0-or-later

//! The accelerator on Windows: the partition and `whpx_accel_init()`.

use std::ffi::c_void;
use std::sync::Arc;

use windows_sys::Win32::System::Hypervisor::{
    WHV_EXTENDED_VM_EXITS, WHV_INTERRUPT_CONTROL, WHV_PARTITION_HANDLE, WHV_PARTITION_PROPERTY,
    WHV_X64_MSR_EXIT_BITMAP, WHvCapabilityCodeFeatures, WHvCapabilityCodeHypervisorPresent,
    WHvCapabilityCodeProcessorFeaturesBanks, WHvCapabilityCodeProcessorPerfmonFeatures,
    WHvPartitionPropertyCodeCpuidExitList, WHvPartitionPropertyCodeExceptionExitBitmap,
    WHvPartitionPropertyCodeExtendedVmExits, WHvPartitionPropertyCodeLocalApicEmulationMode,
    WHvPartitionPropertyCodeNestedVirtualization, WHvPartitionPropertyCodeProcessorCount,
    WHvPartitionPropertyCodeProcessorFeaturesBanks,
    WHvPartitionPropertyCodeProcessorPerfmonFeatures,
    WHvPartitionPropertyCodeProcessorXsaveFeatures, WHvPartitionPropertyCodeSeparateSecurityDomain,
    WHvPartitionPropertyCodeSyntheticProcessorFeaturesBanks,
    WHvPartitionPropertyCodeX64MsrExitBitmap,
};

use crate::features::{self, HostCaps, bank0};
use crate::{WhpxError, WhpxOptions, apic, cpuid};

mod dispatch;
mod slots;
mod vcpu;

use dispatch::{Dispatch, Reg};
pub use slots::SlotListener;
pub use vcpu::{WhpxExits, WhpxKick, WhpxVcpu};

/// `WHV_PROCESSOR_FEATURES_BANKS` and `WHV_SYNTHETIC_PROCESSOR_FEATURES_BANKS`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Banks<const N: usize> {
    count: u32,
    reserved: u32,
    banks: [u64; N],
}

/// A partition. Dropping the last reference deletes it, so the vCPUs and the memory listener
/// each hold one.
#[derive(Debug)]
pub(crate) struct Partition {
    handle: WHV_PARTITION_HANDLE,
    d: &'static Dispatch,
}

impl Drop for Partition {
    fn drop(&mut self) {
        // SAFETY: the handle came from WHvCreatePartition() and nothing uses it after this.
        unsafe { (self.d.delete_partition)(self.handle) };
    }
}

impl Partition {
    pub(crate) fn d(&self) -> &'static Dispatch {
        self.d
    }

    pub(crate) fn handle(&self) -> WHV_PARTITION_HANDLE {
        self.handle
    }

    /// `WHvSetPartitionProperty()` with `val` as the whole buffer.
    fn set<T>(&self, code: i32, val: &T, what: &'static str) -> Result<(), WhpxError> {
        // SAFETY: the buffer is `size_of::<T>()` readable bytes, and every `T` used here is
        // plain data laid out as the property expects.
        let hr = unsafe {
            (self.d.set_partition_property)(
                self.handle,
                code,
                (val as *const T).cast(),
                size_of::<T>() as u32,
            )
        };
        WhpxError::check(what, hr)
    }

    /// `WHvSetPartitionProperty()` with a whole `WHV_PARTITION_PROPERTY`, as QEMU passes most
    /// of them.
    fn set_prop(
        &self,
        code: i32,
        prop: WHV_PARTITION_PROPERTY,
        what: &'static str,
    ) -> Result<(), WhpxError> {
        self.set(code, &prop, what)
    }

    /// `WHvGetPartitionProperty()` into a `T`.
    fn get<T: Copy + Default>(&self, code: i32) -> Result<T, i32> {
        let mut v = T::default();
        let mut written = 0;
        // SAFETY: the buffer is `size_of::<T>()` writable bytes of plain data.
        let hr = unsafe {
            (self.d.get_partition_property)(
                self.handle,
                code,
                (&mut v as *mut T).cast(),
                size_of::<T>() as u32,
                &mut written,
            )
        };
        if hr < 0 { Err(hr) } else { Ok(v) }
    }

    /// `WHvGetVirtualProcessorRegisters()`.
    pub(crate) fn get_regs(
        &self,
        vp: u32,
        names: &[i32],
        vals: &mut [Reg],
    ) -> Result<(), WhpxError> {
        assert_eq!(names.len(), vals.len());
        // SAFETY: both arrays hold `names.len()` elements and `Reg` has the layout of
        // `WHV_REGISTER_VALUE`.
        let hr = unsafe {
            (self.d.get_vp_registers)(
                self.handle,
                vp,
                names.as_ptr(),
                names.len() as u32,
                vals.as_mut_ptr(),
            )
        };
        WhpxError::check("Failed to get virtual processor registers", hr)
    }

    /// `WHvSetVirtualProcessorRegisters()`.
    pub(crate) fn set_regs(&self, vp: u32, names: &[i32], vals: &[Reg]) -> Result<(), WhpxError> {
        assert_eq!(names.len(), vals.len());
        // SAFETY: as for get_regs(), read only.
        let hr = unsafe {
            (self.d.set_vp_registers)(
                self.handle,
                vp,
                names.as_ptr(),
                names.len() as u32,
                vals.as_ptr(),
            )
        };
        WhpxError::check("Failed to set virtual processor registers", hr)
    }

    /// `whpx_vcpu_kick()`.
    pub(crate) fn cancel_run(&self, vp: u32) {
        // SAFETY: any thread may cancel a run, and a vCPU that is not running only has the
        // cancel pending for its next run.
        unsafe { (self.d.cancel_run_vp)(self.handle, vp, 0) };
    }
}

/// `WHvGetCapability()` into a `T` that starts as `init`.
fn capability<T: Copy>(d: &Dispatch, code: i32, init: T) -> Result<T, i32> {
    let mut v = init;
    let mut written = 0;
    // SAFETY: the buffer is `size_of::<T>()` writable bytes of plain data.
    let hr = unsafe {
        (d.get_capability)(
            code,
            (&mut v as *mut T).cast::<c_void>(),
            size_of::<T>() as u32,
            &mut written,
        )
    };
    if hr < 0 { Err(hr) } else { Ok(v) }
}

/// `WHvGetCapability()` of a frequency, `None` when this Windows does not know the code or
/// the query fails, which QEMU only prints.
pub(crate) fn frequency(d: &Dispatch, code: i32) -> Option<u64> {
    capability(d, code, 0u64).ok()
}

/// A WHPX partition, `whpx_global`.
#[derive(Debug)]
pub struct WhpxAccel {
    part: Arc<Partition>,
    irqchip: bool,
    hyperv: bool,
    ignore_unknown_msr: bool,
    rdtscp: bool,
    invpcid: bool,
    xsave: u64,
    slots: Arc<SlotListener>,
}

impl WhpxAccel {
    /// Creates and sets up the partition, `whpx_accel_init()`. `pic` is the machine's `pic`
    /// property being on or auto, and `isapc` drops the Hyper-V LAPIC and enlightenments as
    /// QEMU does for that machine.
    pub fn new(
        opts: &WhpxOptions,
        cpus: u32,
        pic: bool,
        isapc: bool,
    ) -> Result<WhpxAccel, WhpxError> {
        let mut opts = opts.clone();
        if isapc {
            opts.kernel_irqchip = Some(crate::KernelIrqchip::Off);
            opts.hyperv = crate::OnOffAuto::Off;
        }
        let d = dispatch::dispatch()?;

        let present = capability(d, WHvCapabilityCodeHypervisorPresent, 0i32);
        match present {
            Ok(p) if p != 0 => {}
            Ok(_) => return Err(WhpxError::NoAccelerator(0)),
            Err(hr) => return Err(WhpxError::NoAccelerator(hr)),
        }
        let caps_features = capability(d, WHvCapabilityCodeFeatures, 0u64)
            .map_err(|hr| WhpxError::Call("Failed to query capabilities", hr))?;

        let mut handle: WHV_PARTITION_HANDLE = 0;
        // SAFETY: the call writes one handle through a valid pointer.
        let hr = unsafe { (d.create_partition)(&mut handle) };
        WhpxError::check("Failed to create partition", hr)?;
        // From here dropping `part` deletes the partition, which is QEMU's error path.
        let part = Arc::new(Partition { handle, d });

        // Not fatal: older Windows does not know the property.
        let xsave = part.get::<u64>(WHvPartitionPropertyCodeProcessorXsaveFeatures).unwrap_or(0);

        let mut prop = WHV_PARTITION_PROPERTY::default();
        prop.ProcessorCount = cpus;
        part.set_prop(
            WHvPartitionPropertyCodeProcessorCount,
            prop,
            "Failed to set partition processor count",
        )?;

        // Relying on this is QEMU's crutch for Windows 10: the perfmon capability and the
        // synthetic feature banks both came with Windows Server 2022.
        let modern_os = match capability(d, WHvCapabilityCodeProcessorPerfmonFeatures, 0u64) {
            Ok(perfmon) => {
                part.set(
                    WHvPartitionPropertyCodeProcessorPerfmonFeatures,
                    &perfmon,
                    "Failed to set performance monitoring features",
                )?;
                true
            }
            Err(_) => false,
        };

        let caps = HostCaps {
            features: caps_features,
            has_lapic_state2: d.set_lapic_state2.is_some(),
            modern_os,
        };
        let mut irqchip = false;
        if features::want_kernel_irqchip(&opts, &caps, pic)? {
            let mode = features::LOCAL_APIC_EMULATION_X2APIC;
            match part.set(
                WHvPartitionPropertyCodeLocalApicEmulationMode,
                &mode,
                "Failed to enable kernel irqchip",
            ) {
                Ok(()) => irqchip = true,
                Err(_) if opts.kernel_irqchip_required() => {
                    return Err(WhpxError::IrqchipUnavailable);
                }
                Err(_) => {}
            }
        }

        let host = capability(
            d,
            WHvCapabilityCodeProcessorFeaturesBanks,
            Banks::<2> { count: 2, reserved: 0, banks: [0; 2] },
        )
        .map_err(|hr| WhpxError::Call("Failed to get processor features", hr))?;
        let rdtscp = host.banks[0] & bank0::RDTSCP != 0;
        let invpcid = host.banks[0] & bank0::INVPCID != 0;
        let (banks, nested) = features::processor_banks(host.banks, &opts, irqchip);
        if nested {
            let mut prop = WHV_PARTITION_PROPERTY::default();
            prop.NestedVirtualization = 1;
            part.set_prop(
                WHvPartitionPropertyCodeNestedVirtualization,
                prop,
                "Failed to enable nested virtualization",
            )?;
        }
        if !opts.separate_security_domain() {
            let mut prop = WHV_PARTITION_PROPERTY::default();
            prop.SeparateSecurityDomain = 0;
            // Some old Windows 10 releases do not have it, so this is not fatal.
            let _ = part.set_prop(
                WHvPartitionPropertyCodeSeparateSecurityDomain,
                prop,
                "failed to unset separate security domain",
            );
        }
        part.set(
            WHvPartitionPropertyCodeProcessorFeaturesBanks,
            &Banks { count: 2, reserved: 0, banks },
            "Failed to set processor features",
        )?;

        let hyperv = features::want_hyperv(&opts, modern_os)?;
        if hyperv {
            let synthetic =
                Banks { count: 1, reserved: 0, banks: [features::synthetic_bank(irqchip)] };
            part.set(
                WHvPartitionPropertyCodeSyntheticProcessorFeaturesBanks,
                &synthetic,
                "Failed to set synthetic features",
            )?;
        }

        let mut prop = WHV_PARTITION_PROPERTY::default();
        prop.X64MsrExitBitmap = WHV_X64_MSR_EXIT_BITMAP { AsUINT64: features::MSR_EXIT_BITMAP };
        part.set_prop(
            WHvPartitionPropertyCodeX64MsrExitBitmap,
            prop,
            "Failed to set MSR exit bitmap",
        )?;

        part.set(
            WHvPartitionPropertyCodeCpuidExitList,
            &cpuid::EXIT_LIST,
            "Failed to set partition CpuidExitList",
        )?;

        // No exceptions are intercepted until a debugger asks, apart from #GP for
        // intercept-msr-gp.
        let bitmap = features::exception_bitmap(&opts, 0);
        let mut prop = WHV_PARTITION_PROPERTY::default();
        prop.ExtendedVmExits = WHV_EXTENDED_VM_EXITS { AsUINT64: features::extended_exits(bitmap) };
        part.set_prop(
            WHvPartitionPropertyCodeExtendedVmExits,
            prop,
            "Failed to enable extended VM exits",
        )?;
        let mut prop = WHV_PARTITION_PROPERTY::default();
        prop.ExceptionExitBitmap = bitmap;
        part.set_prop(
            WHvPartitionPropertyCodeExceptionExitBitmap,
            prop,
            "Failed to set exception exit bitmap",
        )?;

        // SAFETY: the handle is a live partition.
        let hr = unsafe { (d.setup_partition)(part.handle) };
        WhpxError::check("Failed to setup partition", hr)?;

        let slots = Arc::new(SlotListener::new(Arc::clone(&part)));
        Ok(WhpxAccel {
            part,
            irqchip,
            hyperv,
            ignore_unknown_msr: opts.ignore_unknown_msr(),
            rdtscp,
            invpcid,
            xsave,
            slots,
        })
    }

    pub(crate) fn partition(&self) -> &Arc<Partition> {
        &self.part
    }

    /// The Hyper-V LAPIC is in use, `whpx_irqchip_in_kernel()`.
    pub fn irqchip_in_kernel(&self) -> bool {
        self.irqchip
    }

    /// The Hyper-V enlightenments are on.
    pub fn hyperv_enabled(&self) -> bool {
        self.hyperv
    }

    /// `ignore-unknown-msr` as it resolved.
    pub fn ignore_unknown_msr(&self) -> bool {
        self.ignore_unknown_msr
    }

    /// `whpx_has_rdtscp()`.
    pub fn has_rdtscp(&self) -> bool {
        self.rdtscp
    }

    /// `whpx_has_invpcid()`.
    pub fn has_invpcid(&self) -> bool {
        self.invpcid
    }

    /// `WHvPartitionPropertyCodeProcessorXsaveFeatures`, zero when the host does not say.
    pub fn xsave_features(&self) -> u64 {
        self.xsave
    }

    /// The memory listener to register on the system address space.
    pub fn slot_listener(&self) -> Arc<SlotListener> {
        Arc::clone(&self.slots)
    }

    /// `whpx_send_msi()`: delivers an MSI to the Hyper-V LAPIC. `Ok(false)` means the
    /// request named vector 0 and was dropped, as QEMU does with a warning.
    pub fn send_msi(&self, addr: u64, data: u32) -> Result<bool, WhpxError> {
        let Ok(c) = apic::decode_msi(addr, data) else { return Ok(false) };
        let f = self.part.d.request_interrupt.ok_or(WhpxError::Function("WHvRequestInterrupt"))?;
        let ctl = WHV_INTERRUPT_CONTROL {
            _bitfield: c.bits(),
            Destination: c.destination,
            Vector: c.vector,
        };
        // SAFETY: the control block is valid for the call and its size is passed along.
        let hr = unsafe { f(self.part.handle, &ctl, size_of::<WHV_INTERRUPT_CONTROL>() as u32) };
        WhpxError::check("Failed to request interrupt", hr)?;
        Ok(true)
    }
}
