// SPDX-License-Identifier: GPL-2.0-or-later

//! CPUID exits: the leaves the partition sends to userspace and how the answer from the CPU
//! model is patched before it goes back, the `WHvRunVpExitReasonX64Cpuid` case of
//! `whpx_vcpu_run()`.

/// `cpuidExitList` in `whpx_accel_init()`: the leaves that exit to userspace.
pub const EXIT_LIST: [u32; 25] = [
    0x0,
    0x1,
    0x6,
    0x7,
    0xb,
    0xd,
    0x14,
    0x24,
    0x29,
    0x1e,
    0x4000_0000,
    0x4000_0001,
    0x4000_0010,
    0x8000_0000,
    0x8000_0001,
    0x8000_0002,
    0x8000_0003,
    0x8000_0004,
    0x8000_0007,
    0x8000_0008,
    0x8000_000a,
    0x8000_0021,
    0x8000_0022,
    0xc000_0000,
    0xc000_0001,
];

/// CPUID.1:ECX.HYPERVISOR.
pub const EXT_HYPERVISOR: u32 = 1 << 31;
/// CPUID.1:ECX.X2APIC.
pub const EXT_X2APIC: u32 = 1 << 21;
/// CPUID.1:ECX.OSXSAVE.
pub const EXT_OSXSAVE: u32 = 1 << 27;
/// CPUID.1:EDX.APIC.
pub const APIC: u32 = 1 << 9;
/// CPUID.7.0:EDX.CET_IBT.
pub const C7_EDX_CET_IBT: u32 = 1 << 20;
/// CPUID.7.0:ECX.CET_SHSTK.
pub const C7_ECX_CET_SHSTK: u32 = 1 << 7;
/// CPUID.7.0:ECX.OSPKE.
pub const C7_ECX_OSPKE: u32 = 1 << 4;

/// One CPUID result, in register order.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Regs {
    /// EAX.
    pub eax: u32,
    /// EBX.
    pub ebx: u32,
    /// ECX.
    pub ecx: u32,
    /// EDX.
    pub edx: u32,
}

/// What `WHV_X64_CPUID_ACCESS_CONTEXT` carries: the leaf, the subleaf and what the hypervisor
/// would answer.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CpuidExit {
    /// RAX at the exit.
    pub leaf: u32,
    /// RCX at the exit.
    pub subleaf: u32,
    /// `DefaultResultRax` and the others.
    pub default: Regs,
}

/// The parts of the vCPU's model the patching looks at.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Model {
    /// The partition has the Hyper-V synthetic features.
    pub hyperv: bool,
    /// `vmware-cpuid-freq` is on.
    pub vmware_cpuid_freq: bool,
    /// `env->tsc_khz`.
    pub tsc_khz: u32,
    /// `env->apic_bus_freq` in Hz.
    pub apic_bus_freq: u64,
    /// The CPU has x2APIC.
    pub x2apic: bool,
    /// `FEAT_1_EDX` has APIC, which goes away while the APIC is disabled.
    pub apic: bool,
}

/// Patches `model`, which is `cpu_x86_cpuid()` for the exit's leaf, into the answer the guest
/// sees.
pub fn answer(exit: &CpuidExit, model: Regs, m: &Model) -> Regs {
    let mut r = model;
    let d = exit.default;
    if !m.hyperv {
        match exit.leaf {
            1 => r.ecx |= EXT_HYPERVISOR,
            // vmware-cpuid-freq stands in for VMware so Linux gets the TSC and APIC
            // frequencies from leaf 0x40000010. Otherwise the leaf reports KVM.
            0x4000_0000 if m.vmware_cpuid_freq => {
                r = Regs { eax: 0x4000_0010, ebx: 0x6177_4d56, ecx: 0x4d56_6572, edx: 0x6577_6172 };
            }
            0x4000_0000 => {
                r = Regs { eax: 0x4000_0001, ebx: 0x4b4d_564b, ecx: 0x564b_4d56, edx: 0x4d };
            }
            // KVM's x2APIC bit.
            0x4000_0001 if !m.vmware_cpuid_freq => {
                r.eax = 1 << 15;
                r.ebx = 1 << 15;
                r.ecx = 1 << 15;
            }
            0x4000_0010 if m.vmware_cpuid_freq => {
                r.eax = m.tsc_khz;
                r.ebx = (m.apic_bus_freq / 1000) as u32;
            }
            _ => {}
        }
    } else if matches!(exit.leaf, 0x4000_0000 | 0x4000_0001 | 0x4000_0010) {
        r = d;
    }

    if exit.leaf == 1 {
        set(&mut r.ecx, EXT_X2APIC, m.x2apic);
        set(&mut r.edx, APIC, m.apic);
        // OSXSAVE follows CR4, which the default result already knows.
        set(&mut r.ecx, EXT_OSXSAVE, d.ecx & EXT_OSXSAVE != 0);
    }
    // These follow XCR0 and XSS.
    if exit.leaf == 7 && exit.subleaf == 0 {
        set(&mut r.edx, C7_EDX_CET_IBT, d.edx & C7_EDX_CET_IBT != 0);
        set(&mut r.ecx, C7_ECX_CET_SHSTK, d.ecx & C7_ECX_CET_SHSTK != 0);
        set(&mut r.ecx, C7_ECX_OSPKE, d.ecx & C7_ECX_OSPKE != 0);
    }
    if exit.leaf == 0xd && matches!(exit.subleaf, 1 | 2) {
        r.ebx = d.ebx;
    }
    r
}

fn set(reg: &mut u32, bit: u32, on: bool) {
    if on {
        *reg |= bit;
    } else {
        *reg &= !bit;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exit(leaf: u32, subleaf: u32, default: Regs) -> CpuidExit {
        CpuidExit { leaf, subleaf, default }
    }

    #[test]
    fn leaf1_bits() {
        let m = Model { x2apic: true, apic: false, ..Model::default() };
        let d = Regs { ecx: EXT_OSXSAVE, ..Regs::default() };
        let r = answer(&exit(1, 0, d), Regs { edx: APIC, ..Regs::default() }, &m);
        assert_eq!(r.ecx, EXT_HYPERVISOR | EXT_X2APIC | EXT_OSXSAVE);
        assert_eq!(r.edx, 0);
        let r = answer(
            &exit(1, 0, Regs::default()),
            Regs { ecx: EXT_OSXSAVE, ..Regs::default() },
            &Model { hyperv: true, ..m },
        );
        assert_eq!(r.ecx, EXT_X2APIC);
    }

    #[test]
    fn hypervisor_leaves() {
        let kvm =
            answer(&exit(0x4000_0000, 0, Regs::default()), Regs::default(), &Model::default());
        assert_eq!((kvm.ebx, kvm.ecx, kvm.edx), (0x4b4d_564b, 0x564b_4d56, 0x4d));
        let m = Model {
            vmware_cpuid_freq: true,
            tsc_khz: 2_000_000,
            apic_bus_freq: 1_000_000_000,
            ..Model::default()
        };
        let vmw = answer(&exit(0x4000_0010, 0, Regs::default()), Regs::default(), &m);
        assert_eq!((vmw.eax, vmw.ebx), (2_000_000, 1_000_000));
        let d = Regs { eax: 0x4000_000b, ebx: 1, ecx: 2, edx: 3 };
        let hv = answer(&exit(0x4000_0000, 0, d), Regs::default(), &Model { hyperv: true, ..m });
        assert_eq!(hv, d);
    }

    #[test]
    fn dynamic_bits_follow_default() {
        let d = Regs { ecx: C7_ECX_OSPKE, ..Regs::default() };
        let model = Regs { ecx: C7_ECX_CET_SHSTK, edx: C7_EDX_CET_IBT, ..Regs::default() };
        let r = answer(&exit(7, 0, d), model, &Model::default());
        assert_eq!((r.ecx, r.edx), (C7_ECX_OSPKE, 0));
        let r = answer(
            &exit(0xd, 1, Regs { ebx: 0x340, ..Regs::default() }),
            Regs::default(),
            &Model::default(),
        );
        assert_eq!(r.ebx, 0x340);
        assert_eq!(EXIT_LIST.len(), 25);
    }
}
