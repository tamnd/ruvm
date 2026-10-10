// SPDX-License-Identifier: GPL-2.0-or-later

//! The partition properties `whpx_accel_init()` sets and the choices behind them: whether the
//! Hyper-V LAPIC is used, which processor and synthetic features the guest gets and which
//! exits come back to userspace.

use crate::{WhpxError, WhpxOptions};

/// `WHV_CAPABILITY_FEATURES.LocalApicEmulation`.
pub const CAP_LOCAL_APIC_EMULATION: u64 = 1 << 1;
/// `WHvX64LocalApicEmulationModeX2Apic`.
pub const LOCAL_APIC_EMULATION_X2APIC: i32 = 2;

/// Bits of processor features bank 0.
pub mod bank0 {
    /// `RdtscpSupport`.
    pub const RDTSCP: u64 = 1 << 38;
    /// `InvpcidSupport`.
    pub const INVPCID: u64 = 1 << 43;
    /// `IbrsSupport`.
    pub const IBRS: u64 = 1 << 44;
    /// `StibpSupport`.
    pub const STIBP: u64 = 1 << 45;
    /// `IbpbSupport`.
    pub const IBPB: u64 = 1 << 46;
    /// `SsbdSupport`.
    pub const SSBD: u64 = 1 << 48;
    /// `IbrsAllSupport`.
    pub const IBRS_ALL: u64 = 1 << 52;
}

/// Bits of processor features bank 1.
pub mod bank1 {
    /// `NestedVirtSupport`.
    pub const NESTED_VIRT: u64 = 1 << 6;
    /// `PsfdSupport`.
    pub const PSFD: u64 = 1 << 7;
}

/// Bits of synthetic processor features bank 0.
pub mod synthetic {
    /// `HypervisorPresent`.
    pub const HYPERVISOR_PRESENT: u64 = 1 << 0;
    /// `Hv1`.
    pub const HV1: u64 = 1 << 1;
    /// `AccessVpRunTimeReg`.
    pub const VP_RUNTIME: u64 = 1 << 2;
    /// `AccessPartitionReferenceCounter`.
    pub const REFERENCE_COUNTER: u64 = 1 << 3;
    /// `AccessSynicRegs`.
    pub const SYNIC: u64 = 1 << 4;
    /// `AccessSyntheticTimerRegs`.
    pub const SYNTHETIC_TIMERS: u64 = 1 << 5;
    /// `AccessIntrCtrlRegs`.
    pub const INTR_CTRL: u64 = 1 << 6;
    /// `AccessHypercallRegs`.
    pub const HYPERCALL: u64 = 1 << 7;
    /// `AccessVpIndex`.
    pub const VP_INDEX: u64 = 1 << 8;
    /// `AccessPartitionReferenceTsc`.
    pub const REFERENCE_TSC: u64 = 1 << 9;
    /// `AccessGuestIdleReg`.
    pub const GUEST_IDLE: u64 = 1 << 10;
    /// `AccessFrequencyRegs`.
    pub const FREQUENCY: u64 = 1 << 11;
    /// `EnableExtendedGvaRangesForFlushVirtualAddressList`.
    pub const EXTENDED_GVA_RANGES: u64 = 1 << 15;
    /// `DirectSyntheticTimers`.
    pub const DIRECT_TIMERS: u64 = 1 << 22;
    /// `TbFlushHypercalls`.
    pub const TB_FLUSH: u64 = 1 << 25;
    /// `SyntheticClusterIpi`.
    pub const CLUSTER_IPI: u64 = 1 << 26;
}

/// `WHV_X64_MSR_EXIT_BITMAP` with `UnhandledMsrs` and `ApicBaseMsrWrite`.
pub const MSR_EXIT_BITMAP: u64 = 1 << 0 | 1 << 3;

/// `WHvX64ExceptionTypeGeneralProtectionFault`.
pub const EXCEPTION_GP: u32 = 13;

/// What the host offers, from `WHvGetCapability()` and the optional exports.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct HostCaps {
    /// `WHvCapabilityCodeFeatures`.
    pub features: u64,
    /// WinHvPlatform.dll exports `WHvSetVirtualProcessorInterruptControllerState2`.
    pub has_lapic_state2: bool,
    /// The perfmon capability query worked, which is Windows Server 2022 and later. QEMU
    /// calls the older ones legacy.
    pub modern_os: bool,
}

/// Whether to try `WHvPartitionPropertyCodeLocalApicEmulationMode`. Fails when
/// `kernel-irqchip=on` cannot be honoured. A legacy Windows with a PIC keeps the userspace
/// irqchip unless the user insisted.
pub fn want_kernel_irqchip(
    opts: &WhpxOptions,
    caps: &HostCaps,
    pic: bool,
) -> Result<bool, WhpxError> {
    let usable = caps.features & CAP_LOCAL_APIC_EMULATION != 0 && caps.has_lapic_state2;
    let required = opts.kernel_irqchip_required();
    if required && !usable {
        return Err(WhpxError::IrqchipUnavailable);
    }
    Ok(opts.kernel_irqchip_allowed() && !(!caps.modern_os && pic && !required) && usable)
}

/// The processor feature banks after the host's answer, and whether nested virtualization
/// gets turned on. With `ssd=off` the speculation controls go away so the guest skips the
/// mitigations, which makes exits much cheaper.
pub fn processor_banks(
    host: [u64; 2],
    opts: &WhpxOptions,
    irqchip_in_kernel: bool,
) -> ([u64; 2], bool) {
    let mut b = host;
    let nested = irqchip_in_kernel && b[1] & bank1::NESTED_VIRT != 0;
    if !opts.separate_security_domain() {
        b[0] &= !(bank0::IBRS | bank0::STIBP | bank0::IBPB | bank0::SSBD | bank0::IBRS_ALL);
        b[1] &= !bank1::PSFD;
    }
    (b, nested)
}

/// Synthetic features bank 0. The SynIC, timers and the rest need the Hyper-V LAPIC, and the
/// TLB flush hypercalls misbehave on SMP guests without it.
pub fn synthetic_bank(irqchip_in_kernel: bool) -> u64 {
    use synthetic::*;
    let mut b = HYPERVISOR_PRESENT
        | HV1
        | VP_RUNTIME
        | REFERENCE_COUNTER
        | REFERENCE_TSC
        | HYPERCALL
        | FREQUENCY
        | VP_INDEX;
    if irqchip_in_kernel {
        b |= SYNIC
            | SYNTHETIC_TIMERS
            | INTR_CTRL
            | CLUSTER_IPI
            | DIRECT_TIMERS
            | GUEST_IDLE
            | TB_FLUSH
            | EXTENDED_GVA_RANGES;
    }
    b
}

/// Whether the synthetic features are set, which is what turns the Hyper-V enlightenments on.
pub fn want_hyperv(opts: &WhpxOptions, modern_os: bool) -> Result<bool, WhpxError> {
    if modern_os && opts.hyperv_allowed() {
        Ok(true)
    } else if !modern_os && opts.hyperv_required() {
        Err(WhpxError::HypervUnavailable)
    } else {
        Ok(false)
    }
}

/// `whpx_get_default_exceptions()` plus `extra`: the exception exit bitmap.
pub fn exception_bitmap(opts: &WhpxOptions, extra: u64) -> u64 {
    let gp = if opts.intercept_msr_gp() { 1 << EXCEPTION_GP } else { 0 };
    extra | gp
}

/// `WHV_EXTENDED_VM_EXITS`: CPUID and MSR exits always, exception exits when any exception
/// is intercepted.
pub fn extended_exits(exception_bitmap: u64) -> u64 {
    let exceptions = if exception_bitmap != 0 { 1 << 2 } else { 0 };
    1 << 0 | 1 << 1 | exceptions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KernelIrqchip, OnOffAuto};

    const MODERN: HostCaps =
        HostCaps { features: CAP_LOCAL_APIC_EMULATION, has_lapic_state2: true, modern_os: true };

    #[test]
    fn irqchip_choice() {
        let auto = WhpxOptions::default();
        assert_eq!(want_kernel_irqchip(&auto, &MODERN, true), Ok(true));
        let legacy = HostCaps { modern_os: false, ..MODERN };
        assert_eq!(want_kernel_irqchip(&auto, &legacy, true), Ok(false));
        assert_eq!(want_kernel_irqchip(&auto, &legacy, false), Ok(true));
        let on = WhpxOptions { kernel_irqchip: Some(KernelIrqchip::On), ..auto.clone() };
        assert_eq!(want_kernel_irqchip(&on, &legacy, true), Ok(true));
        let old = HostCaps { has_lapic_state2: false, ..MODERN };
        assert_eq!(want_kernel_irqchip(&on, &old, false), Err(WhpxError::IrqchipUnavailable));
        assert_eq!(want_kernel_irqchip(&auto, &old, false), Ok(false));
        let off = WhpxOptions { kernel_irqchip: Some(KernelIrqchip::Off), ..auto };
        assert_eq!(want_kernel_irqchip(&off, &MODERN, false), Ok(false));
    }

    #[test]
    fn banks() {
        let host = [bank0::RDTSCP | bank0::IBRS | bank0::SSBD, bank1::NESTED_VIRT | bank1::PSFD];
        let opts = WhpxOptions::default();
        assert_eq!(processor_banks(host, &opts, true), (host, true));
        assert_eq!(processor_banks(host, &opts, false), (host, false));
        let fast = WhpxOptions { ssd: OnOffAuto::Off, ..opts };
        assert_eq!(processor_banks(host, &fast, false).0, [bank0::RDTSCP, bank1::NESTED_VIRT]);
    }

    #[test]
    fn synthetic_and_exits() {
        assert_eq!(synthetic_bank(false), 0xb8f);
        assert_eq!(synthetic_bank(true), 0x0640_8fff);
        let opts = WhpxOptions::default();
        assert_eq!(want_hyperv(&opts, true), Ok(true));
        assert_eq!(want_hyperv(&opts, false), Ok(false));
        let on = WhpxOptions { hyperv: OnOffAuto::On, ..opts.clone() };
        assert_eq!(want_hyperv(&on, false), Err(WhpxError::HypervUnavailable));
        assert_eq!(exception_bitmap(&opts, 0), 0);
        let gp = WhpxOptions { intercept_msr_gp: OnOffAuto::On, ..opts };
        assert_eq!(exception_bitmap(&gp, 0), 1 << 13);
        assert_eq!(extended_exits(0), 3);
        assert_eq!(extended_exits(1 << 13), 7);
    }
}
