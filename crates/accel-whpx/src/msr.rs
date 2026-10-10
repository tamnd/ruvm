// SPDX-License-Identifier: GPL-2.0-or-later

//! MSR exits: which MSR accesses QEMU answers itself, the `WHvRunVpExitReasonX64MsrAccess`
//! case of `whpx_vcpu_run()`. Only MSRs in the partition's MSR exit bitmap get here, which is
//! the unhandled ones and writes to the APIC base.

/// `MSR_IA32_APICBASE`.
pub const IA32_APICBASE: u32 = 0x1b;
/// The BSP flag of the APIC base.
pub const APICBASE_BSP: u64 = 1 << 8;
/// The x2APIC enable flag.
pub const APICBASE_EXTD: u64 = 1 << 10;
/// The global enable flag.
pub const APICBASE_ENABLE: u64 = 1 << 11;
/// The base address bits.
pub const APICBASE_BASE: u64 = 0xfffff << 12;
/// `MSR_IA32_APICBASE_RESERVED`.
pub const APICBASE_RESERVED: u64 =
    !(APICBASE_BSP | APICBASE_ENABLE | APICBASE_EXTD | APICBASE_BASE);
/// `MSR_APIC_START`, the first x2APIC MSR.
pub const APIC_START: u32 = 0x800;
/// `MSR_APIC_END`, the last x2APIC MSR.
pub const APIC_END: u32 = 0x8ff;
/// `HV_X64_MSR_APIC_FREQUENCY`.
pub const HV_APIC_FREQUENCY: u32 = 0x4000_0023;
/// `HV_X64_MSR_VP_ASSIST_PAGE`.
pub const HV_VP_ASSIST_PAGE: u32 = 0x4000_0073;
/// `HV_X64_MSR_GUEST_IDLE`.
pub const HV_GUEST_IDLE: u32 = 0x4000_00f0;

/// One MSR exit, from `WHV_X64_MSR_ACCESS_CONTEXT`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MsrExit {
    /// `MsrNumber`.
    pub msr: u32,
    /// `AccessInfo.IsWrite`.
    pub write: bool,
    /// The value a write stores, EDX:EAX. Zero for a read.
    pub value: u64,
}

impl MsrExit {
    /// Builds the exit from the context's RAX and RDX.
    pub fn new(msr: u32, write: bool, rax: u64, rdx: u64) -> MsrExit {
        let value = if write { (rax & 0xffff_ffff) | (rdx << 32) } else { 0 };
        MsrExit { msr, write, value }
    }
}

/// The partition state the decision depends on.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct MsrContext {
    /// The Hyper-V LAPIC is in use.
    pub irqchip_in_kernel: bool,
    /// The partition has the Hyper-V synthetic features.
    pub hyperv: bool,
    /// `ignore-unknown-msr`.
    pub ignore_unknown_msr: bool,
    /// `env->apic_bus_freq`.
    pub apic_bus_freq: u64,
}

/// What the run loop does with an MSR exit.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MsrAction {
    /// Finish a read with this value in EDX:EAX and step over the instruction.
    Read(u64),
    /// Step over a write and drop it.
    Ignore,
    /// Inject #GP(0) and leave RIP on the instruction.
    Gpf,
    /// Give the value to `cpu_set_apic_base()`. When the APIC device takes it, write it to
    /// `WHvX64RegisterApicBase` and step over. When it refuses, #GP(0).
    SetApicBase(u64),
    /// `apic_msr_read()` of this x2APIC register index. An error is #GP(0).
    ApicRead(u32),
    /// `apic_msr_write()` of this index and value. An error is #GP(0).
    ApicWrite(u32, u64),
    /// Step over and handle the read as a HLT, `whpx_handle_hyperv_guestidle()`.
    GuestIdle,
    /// An MSR QEMU does not know. Reads give zero and writes are dropped, but the run loop
    /// traces it first.
    Unknown,
}

/// Decides how to answer `exit`.
pub fn classify(exit: &MsrExit, ctx: &MsrContext) -> MsrAction {
    let userspace_apic = !ctx.irqchip_in_kernel;
    match exit.msr {
        HV_APIC_FREQUENCY if !exit.write && userspace_apic => MsrAction::Read(ctx.apic_bus_freq),
        // Hyper-V never sends reads of the APIC base out, and QEMU aborts on one. Here it reads
        // as an unknown MSR. QEMU also calls cpu_set_apic_base() after raising #GP for reserved
        // bits, which the APIC refuses anyway.
        IA32_APICBASE if exit.write && exit.value & APICBASE_RESERVED != 0 => MsrAction::Gpf,
        IA32_APICBASE if exit.write => MsrAction::SetApicBase(exit.value),
        APIC_START..=APIC_END if userspace_apic => {
            let index = exit.msr - APIC_START;
            if exit.write {
                MsrAction::ApicWrite(index, exit.value)
            } else {
                MsrAction::ApicRead(index)
            }
        }
        // Windows and Linux both use this one, Windows 11 25H2 even when it is not advertised.
        HV_GUEST_IDLE if !exit.write && userspace_apic && ctx.hyperv => MsrAction::GuestIdle,
        // Linux writes the VP assist page even when it is not exposed, and it is not used.
        HV_VP_ASSIST_PAGE if exit.write && userspace_apic && ctx.hyperv => MsrAction::Ignore,
        _ if !ctx.ignore_unknown_msr => MsrAction::Gpf,
        _ => MsrAction::Unknown,
    }
}

/// The registers written back after the exit: RIP, then for a read EAX and EDX.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MsrResult {
    /// The new RIP.
    pub rip: u64,
    /// EAX and EDX for a read, `None` for a write.
    pub rax_rdx: Option<(u64, u64)>,
}

/// `rip` and `len` are `VpContext.Rip` and `InstructionLength`, and `value` is what a read
/// returns. A #GP keeps RIP on the instruction.
pub fn finish(exit: &MsrExit, rip: u64, len: u8, value: u64, gpf: bool) -> MsrResult {
    let rip = if gpf { rip } else { rip.wrapping_add(u64::from(len)) };
    let rax_rdx = (!exit.write).then_some((value & 0xffff_ffff, value >> 32));
    MsrResult { rip, rax_rdx }
}

#[cfg(test)]
mod tests {
    use super::*;

    const USER: MsrContext = MsrContext {
        irqchip_in_kernel: false,
        hyperv: true,
        ignore_unknown_msr: true,
        apic_bus_freq: 1_000_000_000,
    };

    #[test]
    fn apic_base() {
        let w = MsrExit::new(IA32_APICBASE, true, 0xfee0_0900, 0);
        assert_eq!(classify(&w, &USER), MsrAction::SetApicBase(0xfee0_0900));
        let bad = MsrExit::new(IA32_APICBASE, true, 0xfee0_0901, 0);
        assert_eq!(classify(&bad, &USER), MsrAction::Gpf);
        assert_eq!(APICBASE_RESERVED & 0xfff, 0x2ff);
    }

    #[test]
    fn userspace_apic_only() {
        let kernel = MsrContext { irqchip_in_kernel: true, ..USER };
        let r = MsrExit::new(0x830, false, 0, 0);
        assert_eq!(classify(&r, &USER), MsrAction::ApicRead(0x30));
        assert_eq!(classify(&r, &kernel), MsrAction::Unknown);
        let w = MsrExit::new(0x80b, true, 0, 0);
        assert_eq!(classify(&w, &USER), MsrAction::ApicWrite(0xb, 0));
        let f = MsrExit::new(HV_APIC_FREQUENCY, false, 0, 0);
        assert_eq!(classify(&f, &USER), MsrAction::Read(1_000_000_000));
        assert_eq!(classify(&f, &kernel), MsrAction::Unknown);
    }

    #[test]
    fn hyperv_msrs() {
        let idle = MsrExit::new(HV_GUEST_IDLE, false, 0, 0);
        assert_eq!(classify(&idle, &USER), MsrAction::GuestIdle);
        let plain = MsrContext { hyperv: false, ..USER };
        assert_eq!(classify(&idle, &plain), MsrAction::Unknown);
        let vp = MsrExit::new(HV_VP_ASSIST_PAGE, true, 1, 0);
        assert_eq!(classify(&vp, &USER), MsrAction::Ignore);
    }

    #[test]
    fn unknown_and_results() {
        let strict = MsrContext { ignore_unknown_msr: false, ..USER };
        let r = MsrExit::new(0xc000_0103, false, 0, 0);
        assert_eq!(classify(&r, &strict), MsrAction::Gpf);
        assert_eq!(classify(&r, &USER), MsrAction::Unknown);
        let w = MsrExit::new(0x10, true, 0x1_2345_6789, 0xab);
        assert_eq!(w.value, 0xab_2345_6789);
        assert_eq!(finish(&w, 0x1000, 2, 0, false), MsrResult { rip: 0x1002, rax_rdx: None });
        let res = finish(&r, 0x1000, 2, 0x1_0000_0002, false);
        assert_eq!(res.rax_rdx, Some((2, 1)));
        assert_eq!(finish(&r, 0x1000, 2, 0, true).rip, 0x1000);
    }
}
