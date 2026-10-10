// SPDX-License-Identifier: GPL-2.0-or-later

//! The Windows Hypervisor Platform accelerator, from accel/whpx and target/i386/whpx.
//!
//! The decisions QEMU makes around the platform calls are plain Rust here and build on every
//! host: the accelerator properties and their errors, the CPUID exit list and how each CPUID
//! exit is patched, the MSR exits QEMU answers itself, the interrupt injection state of the
//! userspace irqchip, the processor and synthetic feature banks, the LAPIC state page of the
//! Hyper-V APIC and the segment register conversion.
//!
//! The partition, its vCPUs and the memory listener sit in `windows` and exist only on Windows.
//! WinHvPlatform.dll and WinHvEmulation.dll are loaded at run time, like `init_whp_dispatch()`
//! does, so the binary starts on a host without the Windows Hypervisor Platform feature and
//! only `-accel whpx` fails there. The types come from windows-sys.
//!
//! One difference from QEMU 11.1: MMIO and string port I/O go through Microsoft's instruction
//! emulator in WinHvEmulation.dll, which is what QEMU used before it moved to its own x86
//! decoder. ruvm has no x86 decoder below the target crates yet.
//!
//! Not ported: the gdbstub breakpoints and single stepping, XSAVE state, the TSC and XCR sync,
//! and the register mapping to and from the target-x86 CPU state, which belongs to
//! ruvm-target-x86.

use std::fmt;

pub mod apic;
pub mod cpuid;
pub mod exit;
pub mod features;
pub mod irq;
pub mod msr;
pub mod seg;

#[cfg(windows)]
mod windows;
#[cfg(windows)]
pub use windows::{SlotListener, WhpxAccel, WhpxExits, WhpxKick, WhpxVcpu};

/// `HYPERV_APIC_BUS_FREQUENCY`: the APIC bus the Hyper-V LAPIC runs at when the host does not
/// say.
pub const HYPERV_APIC_BUS_FREQUENCY: u64 = 200_000_000;

/// The APIC bus frequency of the userspace APIC.
pub const USERSPACE_APIC_BUS_FREQUENCY: u64 = 1_000_000_000;

/// Why `WhpxVcpu::run` came back. QEMU sets `exception_index` for these and goes round its
/// outer loop, which is the caller's loop here.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum VcpuStop {
    /// The run was cancelled by the kick handle, `EXCP_INTERRUPT`.
    Kicked,
    /// HLT, or the guest idle MSR, with nothing to take: wait until the kick handle has work,
    /// `EXCP_HLT`.
    Halted,
    /// `CPU_INTERRUPT_INIT`. The caller runs `do_cpu_init()` on the register state, which
    /// halts an AP, and calls run again.
    Init,
    /// `CPU_INTERRUPT_SIPI`. The caller runs `do_cpu_sipi()` and calls run again.
    Sipi,
    /// An intercepted exception other than the #GP the MSR code handles. Only the gdbstub
    /// intercepts these, and it is not ported, so this is the exception type for the log.
    Exception(u8),
    /// `WHPX: Unexpected VP exit code %d`. QEMU pauses the VM.
    Unexpected(i32),
}

/// The `kernel-irqchip` accelerator property.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KernelIrqchip {
    /// The Hyper-V LAPIC is required.
    On,
    /// The APIC, PIC and IOAPIC are all ruvm devices.
    Off,
}

impl KernelIrqchip {
    /// `whpx_set_kernel_irqchip()`. The value is the QAPI enum `OnOffSplit`, and WHPX refuses
    /// split.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "on" => Ok(Self::On),
            "off" => Ok(Self::Off),
            "split" => Err("WHPX: split irqchip currently not supported\n\
                            Try without kernel-irqchip or with kernel-irqchip=on|off"
                .to_string()),
            _ => Err(format!("Parameter 'kernel-irqchip' does not accept value '{value}'")),
        }
    }
}

/// The QAPI `OnOffAuto` enum the other WHPX properties take.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum OnOffAuto {
    /// Let the accelerator decide.
    #[default]
    Auto,
    /// Forced on.
    On,
    /// Forced off.
    Off,
}

impl OnOffAuto {
    /// Parses the value of property `name`.
    pub fn parse(name: &str, value: &str) -> Result<Self, String> {
        match value {
            "auto" => Ok(Self::Auto),
            "on" => Ok(Self::On),
            "off" => Ok(Self::Off),
            _ => Err(format!("Parameter '{name}' does not accept value '{value}'")),
        }
    }

    fn or(self, auto: bool) -> bool {
        match self {
            Self::On => true,
            Self::Off => false,
            Self::Auto => auto,
        }
    }
}

/// The accelerator properties, as `whpx_accel_instance_init()` and the setters leave them.
#[derive(Clone, Debug, Default)]
pub struct WhpxOptions {
    /// `kernel-irqchip`. `None` allows the Hyper-V APIC without requiring it.
    pub kernel_irqchip: Option<KernelIrqchip>,
    /// `hyperv`: the Hyper-V enlightenments.
    pub hyperv: OnOffAuto,
    /// `ignore-unknown-msr`.
    pub ignore_unknown_msr: OnOffAuto,
    /// `intercept-msr-gp`.
    pub intercept_msr_gp: OnOffAuto,
    /// `ssd`: separate security domain.
    pub ssd: OnOffAuto,
}

impl WhpxOptions {
    /// `kernel_irqchip_allowed`.
    pub fn kernel_irqchip_allowed(&self) -> bool {
        self.kernel_irqchip != Some(KernelIrqchip::Off)
    }

    /// `kernel_irqchip_required`.
    pub fn kernel_irqchip_required(&self) -> bool {
        self.kernel_irqchip == Some(KernelIrqchip::On)
    }

    /// `hyperv_enlightenments_allowed`.
    pub fn hyperv_allowed(&self) -> bool {
        self.hyperv != OnOffAuto::Off
    }

    /// `hyperv_enlightenments_required`.
    pub fn hyperv_required(&self) -> bool {
        self.hyperv == OnOffAuto::On
    }

    /// `ignore_unknown_msr`: auto means on.
    pub fn ignore_unknown_msr(&self) -> bool {
        self.ignore_unknown_msr.or(true)
    }

    /// `intercept_msr_gp`: auto means off.
    pub fn intercept_msr_gp(&self) -> bool {
        self.intercept_msr_gp.or(false)
    }

    /// `separate_security_domain`: auto means on.
    pub fn separate_security_domain(&self) -> bool {
        self.ssd.or(true)
    }
}

/// HRESULTs QEMU tests for by value.
pub mod hr {
    /// `WHV_E_UNKNOWN_CAPABILITY`.
    pub const UNKNOWN_CAPABILITY: i32 = 0x8037_0300_u32 as i32;
    /// `WHV_E_UNKNOWN_PROPERTY`.
    pub const UNKNOWN_PROPERTY: i32 = 0x8037_0302_u32 as i32;
}

/// Why the accelerator could not start or a vCPU stopped. The messages are QEMU's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WhpxError {
    /// `LoadLibrary()` failed.
    Library(&'static str),
    /// A function the accelerator needs is missing from WinHvPlatform.dll.
    Function(&'static str),
    /// `WHvCapabilityCodeHypervisorPresent` said no, or the query failed.
    NoAccelerator(i32),
    /// `kernel-irqchip=on` on a host without LAPIC emulation.
    IrqchipUnavailable,
    /// `hyperv=on` on a Windows without the synthetic feature banks.
    HypervUnavailable,
    /// A platform call failed. The text is QEMU's message before `, hr=`.
    Call(&'static str, i32),
    /// The instruction emulator could not finish an MMIO or string I/O access. The value is
    /// the `WHV_EMULATOR_STATUS` bits.
    Emulation(&'static str, u32),
    /// The host is not Windows.
    Unavailable,
}

impl WhpxError {
    /// Turns an HRESULT into `Ok` or [`WhpxError::Call`] with `what`.
    pub fn check(what: &'static str, hr: i32) -> Result<(), WhpxError> {
        if hr < 0 { Err(WhpxError::Call(what, hr)) } else { Ok(()) }
    }
}

impl fmt::Display for WhpxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Library(name) => write!(f, "Could not load library {name}."),
            Self::Function(name) => write!(f, "Could not load function {name}"),
            Self::NoAccelerator(hr) => {
                write!(f, "WHPX: No accelerator found, hr={:08x}", *hr as u32)
            }
            Self::IrqchipUnavailable => f.write_str(
                "WHPX: kernel irqchip requested, but unavailable. Try without kernel-irqchip or \
                 with kernel-irqchip=off",
            ),
            Self::HypervUnavailable => {
                f.write_str("Hyper-V enlightenments not available on legacy Windows")
            }
            Self::Call(what, hr) => write!(f, "WHPX: {what}, hr={:08x}", *hr as u32),
            Self::Emulation(what, status) => {
                write!(f, "WHPX: Failed to emulate {what} with EmulatorReturnStatus: {status}")
            }
            Self::Unavailable => f.write_str("-accel whpx: WHPX is not available on this host"),
        }
    }
}

impl std::error::Error for WhpxError {}

/// The vCPU thread name, `CPU n/WHPX`.
pub fn vcpu_thread_name(index: u32) -> String {
    format!("CPU {index}/WHPX")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn irqchip_values() {
        assert_eq!(KernelIrqchip::parse("on"), Ok(KernelIrqchip::On));
        assert_eq!(KernelIrqchip::parse("off"), Ok(KernelIrqchip::Off));
        assert!(KernelIrqchip::parse("split").unwrap_err().starts_with("WHPX: split irqchip"));
        assert_eq!(
            KernelIrqchip::parse("x"),
            Err("Parameter 'kernel-irqchip' does not accept value 'x'".to_string())
        );
    }

    #[test]
    fn defaults_match_instance_init() {
        let o = WhpxOptions::default();
        assert!(o.kernel_irqchip_allowed() && !o.kernel_irqchip_required());
        assert!(o.hyperv_allowed() && !o.hyperv_required());
        assert!(o.ignore_unknown_msr());
        assert!(!o.intercept_msr_gp());
        assert!(o.separate_security_domain());
        let o = WhpxOptions {
            kernel_irqchip: Some(KernelIrqchip::Off),
            hyperv: OnOffAuto::On,
            ignore_unknown_msr: OnOffAuto::Off,
            ..WhpxOptions::default()
        };
        assert!(!o.kernel_irqchip_allowed() && o.hyperv_required() && !o.ignore_unknown_msr());
        assert_eq!(
            OnOffAuto::parse("ssd", "maybe").unwrap_err(),
            "Parameter 'ssd' does not accept value 'maybe'"
        );
    }

    #[test]
    fn messages_match_qemu() {
        assert_eq!(
            WhpxError::Call("Failed to create partition", 0x8007_0005_u32 as i32).to_string(),
            "WHPX: Failed to create partition, hr=80070005"
        );
        assert_eq!(
            WhpxError::Library("WinHvPlatform.dll").to_string(),
            "Could not load library WinHvPlatform.dll."
        );
        assert_eq!(WhpxError::check("x", 0), Ok(()));
        assert_eq!(vcpu_thread_name(2), "CPU 2/WHPX");
    }
}
