// SPDX-License-Identifier: GPL-2.0-or-later

//! The Hypervisor.framework accelerator for Apple Silicon Macs, from QEMU's accel/hvf and
//! target/arm/hvf.
//!
//! The framework is bound by hand in `macos/ffi.rs`, only the calls QEMU makes. On top of it
//! sit [`HvfAccel`], which creates the VM with its IPA size, nested virtualization and the
//! in-kernel GIC (`hv_gic`), and keeps the guest physical map in step with an address space,
//! and [`HvfVcpu`], which runs one vCPU on its own thread and handles the exits the
//! accelerator can handle alone: MMIO data aborts, the system register traps QEMU answers
//! itself, WFI with the vtimer, and PSCI calls on the HVC or SMC conduit.
//!
//! The decisions live in modules that build on every host, so they are tested everywhere:
//! [`esr`] decodes the exception syndromes, [`sysreg`] has the system register encodings and
//! the list QEMU syncs, [`psci`] decodes PSCI calls, [`vtimer`] holds the generic timer
//! arithmetic, and [`regs`] is the register file moved in and out of a vCPU. The vCPU side of
//! the CPU model, mapping [`regs::ArmRegs`] onto the Arm CPU state, is in ruvm-target-arm.
//!
//! Not ported yet: SME and SME2 state, the gdbstub breakpoints and single stepping, the PMU
//! cycle counter emulation, the physical timer and GICv3 CPU interface traps with
//! `kernel-irqchip=off` (they are handed to the caller), and the per register GIC save and
//! restore of hw/intc/arm_gicv3_hvf.c (the opaque `hv_gic_state` blob is used instead).

use std::fmt;
use std::time::Duration;

use crate::psci::PsciCall;

pub mod cpu;
pub mod esr;
pub mod psci;
pub mod regs;
pub mod sysreg;
pub mod vtimer;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod macos;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub use macos::{
    GicInfo, HvfAccel, HvfExits, HvfKick, HvfVcpu, SlotListener, default_ipa_bits, el2_supported,
    max_ipa_bits,
};

/// The PSCI conduit, the CPU's `psci-conduit` property.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Conduit {
    /// No PSCI: HVC and SMC are UNDEF.
    None,
    /// PSCI calls come by HVC.
    Hvc,
    /// PSCI calls come by SMC.
    Smc,
}

/// Why `HvfVcpu::run` came back.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum VcpuStop {
    /// Kicked by `HvfKick::kick`, or the framework cancelled the run.
    Kicked,
    /// WFI with nothing to do: halt until an interrupt line goes up or the kick, and at most
    /// for the deadline when there is one.
    Wfi(Option<Duration>),
    /// A PSCI call that reaches outside the vCPU. For CPU_ON and AFFINITY_INFO the caller
    /// writes the result to X0 with `HvfVcpu::set_x` and [`psci::x0`]. For CPU_SUSPEND X0
    /// already holds zero and the vCPU should halt as for WFI. The PC is already past an SMC.
    Psci(PsciCall),
    /// A debug exception: BRK, a breakpoint, a watchpoint or a software step.
    Debug(u32),
    /// An exception QEMU only reports, `unhandled exception ec=0x%x`, or a data abort it
    /// asserts on.
    Unhandled { ec: u32, syndrome: u64, pc: u64, far: u64 },
}

/// The `kernel-irqchip` property of `-accel hvf`. HVF has no split mode.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KernelIrqchip {
    /// The GIC is the framework's (`hv_gic_create`, macOS 15 and later).
    On,
    /// The GIC is a ruvm device and the vtimer is driven from the exits.
    Off,
}

impl KernelIrqchip {
    /// Parses the property value, `hvf_set_kernel_irqchip()`. The value is the QAPI enum
    /// `OnOffSplit`, so an unknown value gets the visitor's error and `split` gets HVF's own.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "on" => Ok(Self::On),
            "off" => Ok(Self::Off),
            "split" => Err("HVF: split irqchip is not supported on HVF.".to_string()),
            _ => Err(format!("Parameter 'kernel-irqchip' does not accept value '{value}'")),
        }
    }
}

/// The `-accel hvf` options and what the machine adds to them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HvfOptions {
    /// The `kernel-irqchip` property. `None` takes the machine's default.
    pub kernel_irqchip: Option<KernelIrqchip>,
    /// The IPA size in bits the machine asked for, `get_physical_address_range()`. QEMU uses
    /// 36 when the machine has no opinion.
    pub ipa_bits: u32,
    /// Run the guest at EL2, the virt machine's `virtualization=on`.
    pub el2: bool,
    /// Where the in-kernel GIC's distributor goes.
    pub gic_dist_base: u64,
    /// Where the in-kernel GIC's redistributor region goes.
    pub gic_redist_base: u64,
}

impl Default for HvfOptions {
    fn default() -> Self {
        // The bases are what hvf_arch_vm_create() hard codes: the virt machine's
        // VIRT_GIC_DIST and VIRT_GIC_REDIST.
        HvfOptions {
            kernel_irqchip: None,
            ipa_bits: 36,
            el2: false,
            gic_dist_base: 0x0800_0000,
            gic_redist_base: 0x080a_0000,
        }
    }
}

/// `HV_SUCCESS` and the framework's error codes, from `hv_error.h`.
pub mod hv {
    /// `HV_SUCCESS`.
    pub const SUCCESS: i32 = 0;
    /// `HV_ERROR`.
    pub const ERROR: i32 = 0xfae9_4001_u32 as i32;
    /// `HV_BUSY`.
    pub const BUSY: i32 = 0xfae9_4002_u32 as i32;
    /// `HV_BAD_ARGUMENT`.
    pub const BAD_ARGUMENT: i32 = 0xfae9_4003_u32 as i32;
    /// `HV_ILLEGAL_GUEST_STATE`.
    pub const ILLEGAL_GUEST_STATE: i32 = 0xfae9_4004_u32 as i32;
    /// `HV_NO_RESOURCES`.
    pub const NO_RESOURCES: i32 = 0xfae9_4005_u32 as i32;
    /// `HV_NO_DEVICE`.
    pub const NO_DEVICE: i32 = 0xfae9_4006_u32 as i32;
    /// `HV_DENIED`.
    pub const DENIED: i32 = 0xfae9_4007_u32 as i32;
    /// `HV_FAULT`.
    pub const FAULT: i32 = 0xfae9_4008_u32 as i32;
    /// `HV_UNSUPPORTED`.
    pub const UNSUPPORTED: i32 = 0xfae9_400f_u32 as i32;

    /// `hvf_return_string()`.
    pub fn return_string(ret: i32) -> &'static str {
        match ret {
            SUCCESS => "HV_SUCCESS",
            ERROR => "HV_ERROR",
            BUSY => "HV_BUSY",
            BAD_ARGUMENT => "HV_BAD_ARGUMENT",
            NO_RESOURCES => "HV_NO_RESOURCES",
            NO_DEVICE => "HV_NO_DEVICE",
            UNSUPPORTED => "HV_UNSUPPORTED",
            DENIED => "HV_DENIED",
            _ => "[unknown hv_return value]",
        }
    }
}

/// Why the accelerator could not start or a call failed. The messages are QEMU's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HvfError {
    /// `hv_vm_create()` said `HV_DENIED`: the binary lacks the entitlement.
    Denied,
    /// `virtualization=on` on a Mac or macOS without EL2 support.
    NoNestedVirt,
    /// `hv_vm_config_set_el2_enabled()` failed.
    EnableNestedVirt,
    /// `kernel-irqchip=on` before macOS 15, or `hv_gic_create()` failed.
    Gic(&'static str),
    /// Any other call that did not return `HV_SUCCESS`: what was called and the code.
    Call(&'static str, i32),
    /// The guest did something the accelerator cannot go on from.
    Guest(String),
}

impl HvfError {
    /// Checks a return code, `assert_hvf_ok()` without the abort.
    pub fn check(what: &'static str, ret: i32) -> Result<(), HvfError> {
        if ret == hv::SUCCESS { Ok(()) } else { Err(HvfError::Call(what, ret)) }
    }
}

impl fmt::Display for HvfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HvfError::Denied => f.write_str(
                "Could not access HVF. Is the executable signed with \
                 com.apple.security.hypervisor entitlement?",
            ),
            HvfError::NoNestedVirt => {
                f.write_str("Nested virtualization not supported on this system.")
            }
            HvfError::EnableNestedVirt => f.write_str("Failed to enable nested virtualization."),
            HvfError::Gic(msg) => f.write_str(msg),
            HvfError::Call(what, ret) => {
                write!(f, "Error: {what} = {} (0x{:x})", hv::return_string(*ret), *ret as u32)
            }
            HvfError::Guest(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for HvfError {}

/// The name of vCPU `index`'s thread, `CPU n/HVF`.
pub fn vcpu_thread_name(index: u32) -> String {
    format!("CPU {index}/HVF")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_irqchip_takes_on_and_off_only() {
        assert_eq!(KernelIrqchip::parse("on"), Ok(KernelIrqchip::On));
        assert_eq!(KernelIrqchip::parse("off"), Ok(KernelIrqchip::Off));
        assert_eq!(
            KernelIrqchip::parse("split"),
            Err("HVF: split irqchip is not supported on HVF.".to_string())
        );
        assert_eq!(
            KernelIrqchip::parse("yes"),
            Err("Parameter 'kernel-irqchip' does not accept value 'yes'".to_string())
        );
    }

    #[test]
    fn errors_read_like_qemu() {
        assert_eq!(hv::return_string(hv::DENIED), "HV_DENIED");
        assert_eq!(hv::return_string(hv::FAULT), "[unknown hv_return value]");
        assert_eq!(
            HvfError::Call("hv_vcpu_run", hv::ERROR).to_string(),
            "Error: hv_vcpu_run = HV_ERROR (0xfae94001)"
        );
        assert!(HvfError::check("x", hv::SUCCESS).is_ok());
        assert!(HvfError::Denied.to_string().starts_with("Could not access HVF."));
    }

    #[test]
    fn defaults_match_hvf_arch_vm_create() {
        let o = HvfOptions::default();
        assert_eq!(
            (o.ipa_bits, o.gic_dist_base, o.gic_redist_base),
            (36, 0x0800_0000, 0x080a_0000)
        );
        assert_eq!(vcpu_thread_name(3), "CPU 3/HVF");
    }
}
