// SPDX-License-Identifier: GPL-2.0-or-later

//! The KVM accelerator, from accel/kvm/kvm-all.c and target/i386/kvm/kvm.c.
//!
//! This is the x86 slice that the first guests need: opening `/dev/kvm` with QEMU's checks and
//! error text, the in-kernel and split irqchip, a memory listener that keeps KVM's memory slots in
//! step with the guest physical map, and a vCPU run loop that sends port and MMIO exits into the
//! I/O and memory address spaces. The option types and errors build on every host so the command
//! line can name the accelerator anywhere. The accelerator itself exists only on x86-64 Linux for
//! now.
//!
//! The `dirty-ring-size` property turns the dirty ring on with QEMU's checks and messages, but
//! nothing reaps it yet.
//!
//! Everything else in `spec/06-accelerators.md` comes later: register sync levels, CPUID and MSR
//! setup (which belong to ruvm-target-x86), GSI routing, irqfd, ioeventfd, dirty logging and the
//! other architectures.

use std::fmt;
use std::io;
use std::path::PathBuf;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod linux;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub use linux::{KvmAccel, KvmVcpu, SlotListener, VcpuKick, VcpuStop, spawn_vcpu_thread};

/// `KVM_API_VERSION`. Anything else is refused, like QEMU does.
pub const KVM_API_VERSION: i32 = 12;

/// `KVM_IDENTITY_BASE` in target/i386/kvm/kvm.c. The EPT identity map goes here and the TSS one
/// page later, and the machine reserves 0x4000 bytes from it in the e820 table.
pub const KVM_IDENTITY_BASE: u64 = 0xfeff_c000;

/// The `kernel-irqchip` accelerator property.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KernelIrqchip {
    /// PIC, IOAPIC and LAPIC all in the kernel.
    On,
    /// Only the LAPIC in the kernel. The PIC, IOAPIC and PIT are ruvm devices.
    Split,
    /// Everything in userspace.
    Off,
}

impl KernelIrqchip {
    /// Parses the property value. It is the QAPI enum `OnOffSplit`, so the error is the one the
    /// QAPI visitor gives for an unknown enum value.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "on" => Ok(Self::On),
            "off" => Ok(Self::Off),
            "split" => Ok(Self::Split),
            _ => Err(format!("Parameter 'kernel-irqchip' does not accept value '{value}'")),
        }
    }
}

/// The accelerator properties this crate knows about.
#[derive(Clone, Debug, Default)]
pub struct KvmOptions {
    /// The `device` property. `None` means `/dev/kvm`.
    pub device: Option<PathBuf>,
    /// The `kernel-irqchip` property. `None` leaves it to the machine, which picks split when
    /// its class sets `default_kernel_irqchip_split` and on otherwise.
    pub kernel_irqchip: Option<KernelIrqchip>,
    /// The `dirty-ring-size` property: the entries of each vCPU's dirty ring, a power of two,
    /// or 0 for the dirty bitmap.
    pub dirty_ring_size: u32,
}

/// `sizeof(struct kvm_dirty_gfn)`, one entry of a dirty ring.
pub const KVM_DIRTY_GFN_SIZE: u64 = 16;

/// Capabilities every KVM host must have, from `kvm_required_capabilities[]` in kvm-all.c plus
/// the two ruvm adds, in the order QEMU checks them. The name is what the error message shows.
pub const REQUIRED_CAPS: &[&str] = &[
    "KVM_CAP_USER_MEMORY",
    "KVM_CAP_DESTROY_MEMORY_REGION_WORKS",
    "KVM_CAP_JOIN_MEMORY_REGIONS_WORKS",
    "KVM_CAP_INTERNAL_ERROR_DATA",
    "KVM_CAP_IOEVENTFD",
    "KVM_CAP_IOEVENTFD_ANY_LENGTH",
    "KVM_CAP_IMMEDIATE_EXIT",
    "KVM_CAP_IRQFD",
];

/// The x86 list, `kvm_arch_required_capabilities[]` in target/i386/kvm/kvm.c.
pub const X86_REQUIRED_CAPS: &[&str] = &[
    "KVM_CAP_SET_TSS_ADDR",
    "KVM_CAP_EXT_CPUID",
    "KVM_CAP_MP_STATE",
    "KVM_CAP_SIGNAL_MSI",
    "KVM_CAP_IRQ_ROUTING",
    "KVM_CAP_DEBUGREGS",
    "KVM_CAP_XSAVE",
    "KVM_CAP_VCPU_EVENTS",
    "KVM_CAP_X86_ROBUST_SINGLESTEP",
    "KVM_CAP_MCE",
];

/// Why the accelerator could not start or a vCPU stopped. The messages are QEMU's.
#[derive(Debug)]
pub enum KvmError {
    /// `/dev/kvm` could not be opened.
    Open(io::Error),
    /// `KVM_GET_API_VERSION` returned less than 12.
    VersionTooOld,
    /// `KVM_GET_API_VERSION` returned more than 12.
    VersionNotSupported,
    /// A required capability is missing.
    MissingCap(&'static str),
    /// `KVM_CAP_IRQFD` is missing while creating the irqchip.
    NoIrqfd,
    /// `KVM_CAP_SPLIT_IRQCHIP` could not be enabled.
    SplitIrqchip(io::Error),
    /// `KVM_CREATE_IRQCHIP` failed.
    CreateIrqchip(io::Error),
    /// Some other VM or vCPU ioctl failed during setup.
    Ioctl(&'static str, io::Error),
    /// `KVM_RUN` failed with something other than an interruption.
    Run(io::Error),
    /// The host has no KVM at all.
    Unavailable,
    /// `dirty-ring-size` is more than the kernel takes, which is `max` entries.
    DirtyRingTooBig { size: u32, max: u64 },
    /// `KVM_CAP_DIRTY_LOG_RING` could not be enabled.
    DirtyRing(io::Error),
    /// `KVM_CAP_DIRTY_LOG_RING_WITH_BITMAP` could not be enabled.
    DirtyRingBitmap(io::Error),
}

/// `strerror()` text, without the `(os error N)` that Rust appends.
pub(crate) fn strerror(e: &io::Error) -> String {
    let s = e.to_string();
    match s.rfind(" (os error ") {
        Some(at) => s[..at].to_string(),
        None => s,
    }
}

impl fmt::Display for KvmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open(e) => write!(f, "Could not access KVM kernel module: {}", strerror(e)),
            Self::VersionTooOld => f.write_str("kvm version too old"),
            Self::VersionNotSupported => f.write_str("kvm version not supported"),
            Self::MissingCap(name) => write!(f, "kvm does not support {name}"),
            Self::NoIrqfd => f.write_str("kvm: irqfd not implemented"),
            Self::SplitIrqchip(e) => {
                write!(f, "Could not enable split irqchip mode: {}", strerror(e))
            }
            Self::CreateIrqchip(e) => write!(f, "Create kernel irqchip failed: {}", strerror(e)),
            Self::Ioctl(what, e) => write!(f, "{what} failed: {}", strerror(e)),
            Self::Run(e) => write!(f, "error: kvm run failed {}", strerror(e)),
            Self::Unavailable => f.write_str("-accel kvm: KVM is not available on this host"),
            Self::DirtyRingTooBig { size, max } => write!(
                f,
                "KVM dirty ring size {size} too big (maximum is {max}).  Please use a smaller value."
            ),
            Self::DirtyRing(e) => write!(
                f,
                "Enabling of KVM dirty ring failed: {}. Suggested minimum value is 1024.",
                strerror(e)
            ),
            Self::DirtyRingBitmap(e) => {
                write!(f, "Enabling of KVM dirty ring's backup bitmap failed: {}. ", strerror(e))
            }
        }
    }
}

impl std::error::Error for KvmError {}

/// The vCPU thread name, `CPU n/KVM`, which libvirt and `query-cpus-fast` users look for.
pub fn vcpu_thread_name(index: u32) -> String {
    format!("CPU {index}/KVM")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_match_qemu() {
        let enoent = io::Error::from_raw_os_error(2);
        assert_eq!(
            KvmError::Open(enoent).to_string(),
            "Could not access KVM kernel module: No such file or directory"
        );
        assert_eq!(KvmError::VersionTooOld.to_string(), "kvm version too old");
        assert_eq!(
            KvmError::MissingCap("KVM_CAP_IRQFD").to_string(),
            "kvm does not support KVM_CAP_IRQFD"
        );
        assert_eq!(
            KvmError::Run(io::Error::from_raw_os_error(14)).to_string(),
            "error: kvm run failed Bad address"
        );
        assert_eq!(
            KvmError::DirtyRingTooBig { size: 1 << 20, max: 65536 }.to_string(),
            "KVM dirty ring size 1048576 too big (maximum is 65536).  Please use a smaller value."
        );
        assert_eq!(
            KvmError::DirtyRing(io::Error::from_raw_os_error(22)).to_string(),
            "Enabling of KVM dirty ring failed: Invalid argument. Suggested minimum value is 1024."
        );
    }

    #[test]
    fn irqchip_values() {
        assert_eq!(KernelIrqchip::parse("split"), Ok(KernelIrqchip::Split));
        assert_eq!(KernelIrqchip::parse("on"), Ok(KernelIrqchip::On));
        assert_eq!(KernelIrqchip::parse("off"), Ok(KernelIrqchip::Off));
        assert_eq!(
            KernelIrqchip::parse("maybe"),
            Err("Parameter 'kernel-irqchip' does not accept value 'maybe'".to_string())
        );
        assert_eq!(vcpu_thread_name(3), "CPU 3/KVM");
    }
}
