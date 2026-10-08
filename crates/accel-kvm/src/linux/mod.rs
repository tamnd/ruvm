// SPDX-License-Identifier: GPL-2.0-or-later

//! The Linux side: `kvm_init()` from accel/kvm/kvm-all.c and `kvm_arch_init()` from
//! target/i386/kvm/kvm.c, cut down to what a first x86 guest needs.

use std::ffi::CString;
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::sync::Arc;

use kvm_bindings::{KVM_CAP_SPLIT_IRQCHIP, kvm_enable_cap};
use kvm_ioctls::{Kvm, VmFd};

use crate::{KVM_API_VERSION, KVM_IDENTITY_BASE, KernelIrqchip, KvmError, KvmOptions};

mod slots;
mod vcpu;

pub use slots::SlotListener;
pub use vcpu::{KvmVcpu, VcpuKick, VcpuStop, spawn_vcpu_thread};

/// The capability numbers behind [`crate::REQUIRED_CAPS`] and [`crate::X86_REQUIRED_CAPS`].
const REQUIRED: &[(&str, u32)] = &[
    ("KVM_CAP_USER_MEMORY", kvm_bindings::KVM_CAP_USER_MEMORY),
    ("KVM_CAP_DESTROY_MEMORY_REGION_WORKS", kvm_bindings::KVM_CAP_DESTROY_MEMORY_REGION_WORKS),
    ("KVM_CAP_JOIN_MEMORY_REGIONS_WORKS", kvm_bindings::KVM_CAP_JOIN_MEMORY_REGIONS_WORKS),
    ("KVM_CAP_INTERNAL_ERROR_DATA", kvm_bindings::KVM_CAP_INTERNAL_ERROR_DATA),
    ("KVM_CAP_IOEVENTFD", kvm_bindings::KVM_CAP_IOEVENTFD),
    ("KVM_CAP_IOEVENTFD_ANY_LENGTH", kvm_bindings::KVM_CAP_IOEVENTFD_ANY_LENGTH),
    ("KVM_CAP_IMMEDIATE_EXIT", kvm_bindings::KVM_CAP_IMMEDIATE_EXIT),
    ("KVM_CAP_IRQFD", kvm_bindings::KVM_CAP_IRQFD),
    ("KVM_CAP_SET_TSS_ADDR", kvm_bindings::KVM_CAP_SET_TSS_ADDR),
    ("KVM_CAP_EXT_CPUID", kvm_bindings::KVM_CAP_EXT_CPUID),
    ("KVM_CAP_MP_STATE", kvm_bindings::KVM_CAP_MP_STATE),
    ("KVM_CAP_SIGNAL_MSI", kvm_bindings::KVM_CAP_SIGNAL_MSI),
    ("KVM_CAP_IRQ_ROUTING", kvm_bindings::KVM_CAP_IRQ_ROUTING),
    ("KVM_CAP_DEBUGREGS", kvm_bindings::KVM_CAP_DEBUGREGS),
    ("KVM_CAP_XSAVE", kvm_bindings::KVM_CAP_XSAVE),
    ("KVM_CAP_VCPU_EVENTS", kvm_bindings::KVM_CAP_VCPU_EVENTS),
    ("KVM_CAP_X86_ROBUST_SINGLESTEP", kvm_bindings::KVM_CAP_X86_ROBUST_SINGLESTEP),
    ("KVM_CAP_MCE", kvm_bindings::KVM_CAP_MCE),
];

/// The number of pins on the userspace IOAPIC in split mode, `KVM_IOAPIC_NUM_PINS`.
const SPLIT_IRQCHIP_PINS: u64 = 24;

pub(crate) fn os_error(e: kvm_ioctls::Error) -> io::Error {
    io::Error::from_raw_os_error(e.errno())
}

/// An initialized KVM VM with no vCPUs yet, `KVMState`.
#[derive(Debug)]
pub struct KvmAccel {
    kvm: Kvm,
    vm: Arc<VmFd>,
    irqchip: KernelIrqchip,
    readonly_mem: bool,
    nr_slots: usize,
    /// `kvm_dirty_ring_size`: 0 when the dirty bitmap is in use.
    dirty_ring_size: u32,
    /// `kvm_dirty_ring_with_bitmap`.
    dirty_ring_with_bitmap: bool,
    warnings: Vec<String>,
}

impl KvmAccel {
    /// `kvm_init()`: opens the device, checks the API version and the required capabilities,
    /// creates the VM, places the EPT identity map and the TSS, and creates the irqchip.
    /// `default_split` is the machine class's `default_kernel_irqchip_split`, used when the
    /// property was not given.
    pub fn new(opts: &KvmOptions, default_split: bool) -> Result<Self, KvmError> {
        let path = opts.device.as_deref().unwrap_or(Path::new("/dev/kvm"));
        let path = CString::new(path.as_os_str().as_bytes())
            .map_err(|_| KvmError::Open(io::Error::from_raw_os_error(libc::EINVAL)))?;
        let kvm = Kvm::new_with_path(&path).map_err(|e| KvmError::Open(os_error(e)))?;
        let version = kvm.get_api_version();
        if version < KVM_API_VERSION {
            return Err(KvmError::VersionTooOld);
        }
        if version > KVM_API_VERSION {
            return Err(KvmError::VersionNotSupported);
        }

        let vm = loop {
            match kvm.create_vm() {
                Err(e) if e.errno() == libc::EINTR => continue,
                r => break r.map_err(|e| KvmError::Ioctl("KVM_CREATE_VM", os_error(e)))?,
            }
        };

        for &(name, cap) in REQUIRED {
            if kvm.check_extension_raw(libc::c_ulong::from(cap)) <= 0 {
                return Err(KvmError::MissingCap(name));
            }
        }
        let mut warnings = Vec::new();
        let (dirty_ring_size, dirty_ring_with_bitmap) =
            dirty_ring_init(&vm, opts.dirty_ring_size, &mut warnings)?;
        let readonly_mem =
            vm.check_extension_raw(libc::c_ulong::from(kvm_bindings::KVM_CAP_READONLY_MEM)) > 0;
        let nr_slots = kvm.get_nr_memslots();

        vm.set_identity_map_address(KVM_IDENTITY_BASE)
            .map_err(|e| KvmError::Ioctl("KVM_SET_IDENTITY_MAP_ADDR", os_error(e)))?;
        vm.set_tss_address((KVM_IDENTITY_BASE + 0x1000) as usize)
            .map_err(|e| KvmError::Ioctl("KVM_SET_TSS_ADDR", os_error(e)))?;

        let irqchip = opts.kernel_irqchip.unwrap_or(if default_split {
            KernelIrqchip::Split
        } else {
            KernelIrqchip::On
        });
        if irqchip != KernelIrqchip::Off {
            create_irqchip(&kvm, &vm, irqchip)?;
        }

        Ok(KvmAccel {
            kvm,
            vm: Arc::new(vm),
            irqchip,
            readonly_mem,
            nr_slots,
            dirty_ring_size,
            dirty_ring_with_bitmap,
            warnings,
        })
    }

    /// What `kvm_init()` warned about, for the caller to report.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// `kvm_dirty_ring_size()`: the entries of each vCPU's dirty ring, 0 without one.
    pub fn dirty_ring_size(&self) -> u32 {
        self.dirty_ring_size
    }

    /// Whether the dirty ring comes with a backup bitmap, `kvm_dirty_ring_with_bitmap`.
    pub fn dirty_ring_with_bitmap(&self) -> bool {
        self.dirty_ring_with_bitmap
    }

    /// The VM file descriptor, for the device and CPU code that issues its own ioctls.
    pub fn vm(&self) -> &Arc<VmFd> {
        &self.vm
    }

    /// The system file descriptor.
    pub fn kvm(&self) -> &Kvm {
        &self.kvm
    }

    /// Where the interrupt controllers live.
    pub fn kernel_irqchip(&self) -> KernelIrqchip {
        self.irqchip
    }

    /// Whether `KVM_MEM_READONLY` slots are available, `kvm_readonly_mem_allowed`.
    pub fn readonly_mem(&self) -> bool {
        self.readonly_mem
    }

    /// A memory listener for this VM. Register it on the memory address space and it keeps the
    /// KVM slots in step with the guest physical map.
    pub fn slot_listener(&self) -> Arc<SlotListener> {
        Arc::new(SlotListener::new(Arc::clone(&self.vm), self.readonly_mem, self.nr_slots))
    }

    /// Creates vCPU `index`, `kvm_create_vcpu()`.
    pub fn create_vcpu(&self, index: u32) -> Result<KvmVcpu, KvmError> {
        let fd = self
            .vm
            .create_vcpu(u64::from(index))
            .map_err(|e| KvmError::Ioctl("KVM_CREATE_VCPU", os_error(e)))?;
        Ok(KvmVcpu::new(fd, index))
    }
}

/// `kvm_dirty_ring_init()`: turns the dirty ring on when `size` asks for one and the kernel
/// has it, and gives the ring size in use with whether there is a backup bitmap. Without the
/// capability this warns and stays with the bitmap.
fn dirty_ring_init(
    vm: &VmFd,
    size: u32,
    warnings: &mut Vec<String>,
) -> Result<(u32, bool), KvmError> {
    if size == 0 {
        return Ok((0, false));
    }
    let bytes = u64::from(size) * crate::KVM_DIRTY_GFN_SIZE;
    let check = |cap: u32| vm.check_extension_raw(libc::c_ulong::from(cap));
    let mut cap = kvm_bindings::KVM_CAP_DIRTY_LOG_RING;
    let mut max = check(cap);
    if max <= 0 {
        cap = kvm_bindings::KVM_CAP_DIRTY_LOG_RING_ACQ_REL;
        max = check(cap);
    }
    if max <= 0 {
        warnings.push("KVM dirty ring not available, using bitmap method".to_string());
        return Ok((0, false));
    }
    let max = max as u64;
    if bytes > max {
        return Err(KvmError::DirtyRingTooBig { size, max: max / crate::KVM_DIRTY_GFN_SIZE });
    }
    let mut enable = kvm_enable_cap { cap, ..Default::default() };
    enable.args[0] = bytes;
    vm.enable_cap(&enable).map_err(|e| KvmError::DirtyRing(os_error(e)))?;
    let with_bitmap = check(kvm_bindings::KVM_CAP_DIRTY_LOG_RING_WITH_BITMAP) > 0;
    if with_bitmap {
        let bitmap = kvm_enable_cap {
            cap: kvm_bindings::KVM_CAP_DIRTY_LOG_RING_WITH_BITMAP,
            ..Default::default()
        };
        vm.enable_cap(&bitmap).map_err(|e| KvmError::DirtyRingBitmap(os_error(e)))?;
    }
    Ok((size, with_bitmap))
}

/// `do_kvm_irqchip_create()` with the x86 `kvm_arch_irqchip_create()` folded in.
fn create_irqchip(kvm: &Kvm, vm: &VmFd, mode: KernelIrqchip) -> Result<(), KvmError> {
    let missing = || KvmError::CreateIrqchip(io::Error::from_raw_os_error(libc::EOPNOTSUPP));
    if kvm.check_extension_raw(libc::c_ulong::from(kvm_bindings::KVM_CAP_IRQCHIP)) <= 0 {
        return Err(missing());
    }
    if kvm.check_extension_raw(libc::c_ulong::from(kvm_bindings::KVM_CAP_IRQFD)) <= 0 {
        return Err(KvmError::NoIrqfd);
    }
    if mode == KernelIrqchip::Split {
        let mut cap = kvm_enable_cap { cap: KVM_CAP_SPLIT_IRQCHIP, ..Default::default() };
        cap.args[0] = SPLIT_IRQCHIP_PINS;
        return vm.enable_cap(&cap).map_err(|e| KvmError::SplitIrqchip(os_error(e)));
    }
    vm.create_irq_chip().map_err(|e| KvmError::CreateIrqchip(os_error(e)))
}
