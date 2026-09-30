// SPDX-License-Identifier: GPL-2.0-or-later

//! What the host and KVM can provide, as a plain data snapshot.
//!
//! QEMU asks KVM directly (`KVM_GET_SUPPORTED_CPUID`, `KVM_GET_MSRS` on the
//! system fd, `host_cpuid()`). This crate has no ioctl code, so the KVM
//! accelerator fills in [`HostCpuid`] once and the functions here apply the
//! same fixups as `kvm_arch_get_supported_cpuid()` and
//! `kvm_arch_get_supported_msr_feature()` in `target/i386/kvm/kvm.c`.

use super::cache::Regs;
use super::words::{
    CPUID_6_EAX_ARAT, CPUID_7_0_EBX_ERMS, CPUID_7_0_EBX_HLE, CPUID_7_0_EBX_INVPCID,
    CPUID_7_0_EBX_RDSEED, CPUID_7_0_EBX_RTM, CPUID_7_0_EDX_ARCH_CAPABILITIES, CPUID_7_0_EDX_FSRM,
    CPUID_7_1_EAX_FSRC, CPUID_7_1_EAX_FSRS, CPUID_7_1_EAX_FZRM, CPUID_7_2_EDX_MCDT_NO,
    CPUID_8000_0007_EBX_OVERFLOW_RECOV, CPUID_8000_0007_EBX_SUCCOR, CPUID_EXT_HYPERVISOR,
    CPUID_EXT_RDRAND, CPUID_EXT_TSC_DEADLINE_TIMER, CPUID_EXT_X2APIC, CPUID_EXT2_AMD_ALIASES,
    CPUID_EXT2_RDTSCP, CPUID_EXT3_TOPOEXT, CPUID_HT, CPUID_MCA, CPUID_MCE, CPUID_MTRR, CPUID_PAT,
    CPUID_XSAVE_XSAVES, KVM_CPUID_FEATURES, MSR_ARCH_CAP_FB_CLEAR, MSR_ARCH_CAP_FBSDP_NO,
    MSR_ARCH_CAP_MDS_NO, MSR_ARCH_CAP_PSDP_NO, MSR_ARCH_CAP_SBDR_SSDP_NO, MSR_ARCH_CAP_TAA_NO, Reg,
    VMX_SECONDARY_EXEC_ENABLE_INVPCID, VMX_SECONDARY_EXEC_RDRAND_EXITING,
    VMX_SECONDARY_EXEC_RDSEED_EXITING, VMX_SECONDARY_EXEC_RDTSCP, VMX_SECONDARY_EXEC_XSAVES,
};
use crate::msr::{
    MSR_IA32_ARCH_CAPABILITIES, MSR_IA32_VMX_PROCBASED_CTLS2, MSR_IA32_VMX_TRUE_ENTRY_CTLS,
    MSR_IA32_VMX_TRUE_EXIT_CTLS, MSR_IA32_VMX_TRUE_PINBASED_CTLS, MSR_IA32_VMX_TRUE_PROCBASED_CTLS,
};
use crate::state::IrqchipMode;

/// `CPUID_KVM_PV_UNHALT`, `CPUID[4000_0001].EAX` bit 7.
pub const CPUID_KVM_PV_UNHALT: u32 = 1 << 7;
/// `CPUID_KVM_MSI_EXT_DEST_ID`, `CPUID[4000_0001].EAX` bit 15.
pub const CPUID_KVM_MSI_EXT_DEST_ID: u32 = 1 << 15;
/// `CPUID_KVM_HINTS_REALTIME`, `CPUID[4000_0001].EDX` bit 0.
pub const CPUID_KVM_HINTS_REALTIME: u32 = 1 << 0;

/// One CPUID entry, like `struct kvm_cpuid_entry2` without the flags.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuidEntry {
    /// Leaf (EAX input).
    pub function: u32,
    /// Subleaf (ECX input).
    pub index: u32,
    /// EAX, EBX, ECX, EDX.
    pub regs: Regs,
}

impl CpuidEntry {
    /// Builds an entry.
    pub fn new(function: u32, index: u32, regs: Regs) -> Self {
        Self { function, index, regs }
    }

    fn reg(&self, reg: Reg) -> u32 {
        match reg {
            Reg::Eax => self.regs[0],
            Reg::Ebx => self.regs[1],
            Reg::Ecx => self.regs[2],
            Reg::Edx => self.regs[3],
        }
    }
}

/// A snapshot of the host and of KVM's capabilities.
#[derive(Debug, Clone, Default)]
pub struct HostCpuid {
    /// Output of `KVM_GET_SUPPORTED_CPUID`.
    pub supported: Vec<CpuidEntry>,
    /// Output of the CPUID instruction on the host (`host_cpuid()`).
    pub host: Vec<CpuidEntry>,
    /// Feature MSRs (`KVM_GET_MSR_FEATURE_INDEX_LIST`) with the raw values
    /// `KVM_GET_MSRS` returns for them.
    pub feature_msrs: Vec<(u32, u64)>,
    /// Interrupt controller mode (`kernel-irqchip`).
    pub irqchip: IrqchipMode,
    /// `KVM_CAP_TSC_DEADLINE_TIMER`.
    pub has_tsc_deadline: bool,
    /// `MSR_IA32_ARCH_CAPABILITIES` is in `KVM_GET_MSR_INDEX_LIST`.
    pub has_msr_arch_capabs: bool,
    /// `MSR_IA32_VMX_PROCBASED_CTLS2` is in `KVM_GET_MSR_INDEX_LIST`.
    pub has_msr_vmx_procbased_ctls2: bool,
    /// `KVM_X86_XCOMP_GUEST_SUPP`, when `KVM_GET_DEVICE_ATTR` provides it.
    pub xcomp_guest_supp: Option<u64>,
    /// `MCG_LMCE_P` is set in `KVM_X86_GET_MCE_CAP_SUPPORTED`.
    pub lmce_supported: bool,
}

fn find(entries: &[CpuidEntry], function: u32, index: u32) -> Option<&CpuidEntry> {
    entries.iter().find(|e| e.function == function && e.index == index)
}

/// `x86_cpu_family()`.
pub fn cpu_family(version: u32) -> u32 {
    ((version >> 8) & 0xf) + ((version >> 20) & 0xff)
}

/// `x86_cpu_model()`.
pub fn cpu_model(version: u32) -> u32 {
    ((version >> 4) & 0xf) | ((version >> 12) & 0xf0)
}

/// `x86_cpu_stepping()`.
pub fn cpu_stepping(version: u32) -> u32 {
    version & 0xf
}

impl HostCpuid {
    /// `host_cpuid()`: the raw host registers, zero if the leaf is absent.
    pub fn host_cpuid(&self, function: u32, index: u32) -> Regs {
        find(&self.host, function, index).map_or([0; 4], |e| e.regs)
    }

    /// Host vendor string, as `host_cpu_vendor_fms()`.
    pub fn vendor(&self) -> String {
        let r = self.host_cpuid(0, 0);
        let mut b = Vec::with_capacity(12);
        for w in [r[1], r[3], r[2]] {
            b.extend_from_slice(&w.to_le_bytes());
        }
        String::from_utf8_lossy(&b).trim_end_matches('\0').to_string()
    }

    /// Host family, model and stepping.
    pub fn fms(&self) -> (u32, u32, u32) {
        let eax = self.host_cpuid(1, 0)[0];
        (cpu_family(eax), cpu_model(eax), cpu_stepping(eax))
    }

    /// Host brand string, as `host_cpu_fill_model_id()`. Stops at the
    /// first NUL, the way the C string is then consumed.
    pub fn model_id(&self) -> String {
        let mut b = Vec::with_capacity(48);
        for i in 0..3 {
            for w in self.host_cpuid(0x8000_0002 + i, 0) {
                b.extend_from_slice(&w.to_le_bytes());
            }
        }
        let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
        String::from_utf8_lossy(&b[..end]).into_owned()
    }

    /// `host_cpu_phys_bits()`.
    pub fn phys_bits(&self) -> u32 {
        if self.host_cpuid(0x8000_0000, 0)[0] >= 0x8000_0008 {
            self.host_cpuid(0x8000_0008, 0)[0] & 0xff
        } else {
            36
        }
    }

    /// `host_tsx_broken()`: Haswell parts with broken TSX.
    pub fn tsx_broken(&self) -> bool {
        let (family, model, stepping) = self.fms();
        self.vendor() == "GenuineIntel"
            && family == 6
            && ((model == 63 && stepping < 4) || model == 60 || model == 69 || model == 70)
    }

    /// `kvm_arch_get_supported_cpuid()` with all of its fixups.
    pub fn supported_cpuid(&self, function: u32, index: u32, reg: Reg) -> u32 {
        let mut ret = find(&self.supported, function, index).map_or(0, |e| e.reg(reg));
        let in_kernel = self.irqchip.in_kernel();
        match (function, reg) {
            (1, Reg::Edx) => {
                ret |= (CPUID_MTRR | CPUID_PAT | CPUID_MCE | CPUID_MCA | CPUID_HT) as u32;
            }
            (1, Reg::Ecx) => {
                ret |= CPUID_EXT_HYPERVISOR as u32;
                if in_kernel && self.has_tsc_deadline {
                    ret |= CPUID_EXT_TSC_DEADLINE_TIMER as u32;
                }
                if !in_kernel {
                    ret &= !(CPUID_EXT_X2APIC as u32);
                }
            }
            (6, Reg::Eax) => ret |= CPUID_6_EAX_ARAT as u32,
            (7, Reg::Ebx) if index == 0 => {
                ret |= self.host_cpuid(7, 0)[1] & CPUID_7_0_EBX_ERMS as u32;
                if self.tsx_broken() {
                    ret &= !((CPUID_7_0_EBX_RTM | CPUID_7_0_EBX_HLE) as u32);
                }
            }
            (7, Reg::Edx) if index == 0 => {
                ret |= self.host_cpuid(7, 0)[3] & CPUID_7_0_EDX_FSRM as u32;
                if !self.has_msr_arch_capabs {
                    ret &= !(CPUID_7_0_EDX_ARCH_CAPABILITIES as u32);
                }
            }
            (7, Reg::Eax) if index == 1 => {
                ret |= self.host_cpuid(7, 1)[0]
                    & (CPUID_7_1_EAX_FZRM | CPUID_7_1_EAX_FSRS | CPUID_7_1_EAX_FSRC) as u32;
            }
            (7, Reg::Edx) if index == 2 => {
                ret |= self.host_cpuid(7, 2)[3] & CPUID_7_2_EDX_MCDT_NO as u32;
            }
            (0xd, Reg::Eax) | (0xd, Reg::Edx) if index == 0 => {
                if let Some(mask) = self.xcomp_guest_supp {
                    ret = if reg == Reg::Eax { mask as u32 } else { (mask >> 32) as u32 };
                }
            }
            (0x8000_0001, Reg::Ecx) => ret |= CPUID_EXT3_TOPOEXT as u32,
            (0x8000_0001, Reg::Edx) => {
                ret |= self.supported_cpuid(1, 0, Reg::Edx) & CPUID_EXT2_AMD_ALIASES as u32;
            }
            (0x8000_0007, Reg::Ebx) => {
                ret |= (CPUID_8000_0007_EBX_OVERFLOW_RECOV | CPUID_8000_0007_EBX_SUCCOR) as u32;
            }
            (KVM_CPUID_FEATURES, Reg::Eax) => {
                if !in_kernel {
                    ret &= !CPUID_KVM_PV_UNHALT;
                }
                if self.irqchip == IrqchipMode::Split {
                    ret |= CPUID_KVM_MSI_EXT_DEST_ID;
                }
            }
            (KVM_CPUID_FEATURES, Reg::Edx) => ret |= CPUID_KVM_HINTS_REALTIME,
            _ => {}
        }
        ret
    }

    /// `kvm_arch_get_supported_msr_feature()`.
    pub fn supported_msr_feature(&self, index: u32) -> u64 {
        let Some(&(_, raw)) = self.feature_msrs.iter().find(|(i, _)| *i == index) else {
            return 0;
        };
        let mut value = raw;
        match index {
            MSR_IA32_VMX_PROCBASED_CTLS2
            | MSR_IA32_VMX_TRUE_PINBASED_CTLS
            | MSR_IA32_VMX_TRUE_PROCBASED_CTLS
            | MSR_IA32_VMX_TRUE_ENTRY_CTLS
            | MSR_IA32_VMX_TRUE_EXIT_CTLS => {
                if index == MSR_IA32_VMX_PROCBASED_CTLS2 && !self.has_msr_vmx_procbased_ctls2 {
                    // KVM forgot to add these bits for some time.
                    let add = [
                        (0xd, 1, Reg::Ecx, CPUID_XSAVE_XSAVES, VMX_SECONDARY_EXEC_XSAVES),
                        (1, 0, Reg::Ecx, CPUID_EXT_RDRAND, VMX_SECONDARY_EXEC_RDRAND_EXITING),
                        (7, 0, Reg::Ebx, CPUID_7_0_EBX_INVPCID, VMX_SECONDARY_EXEC_ENABLE_INVPCID),
                        (7, 0, Reg::Ebx, CPUID_7_0_EBX_RDSEED, VMX_SECONDARY_EXEC_RDSEED_EXITING),
                        (0x8000_0001, 0, Reg::Edx, CPUID_EXT2_RDTSCP, VMX_SECONDARY_EXEC_RDTSCP),
                    ];
                    for (f, i, r, bit, ctl) in add {
                        if u64::from(self.supported_cpuid(f, i, r)) & bit != 0 {
                            value |= ctl << 32;
                        }
                    }
                }
                // Bits that can be one but do not have to be.
                let must_be_one = value as u32;
                let can_be_one = (value >> 32) as u32;
                u64::from(can_be_one & !must_be_one)
            }
            MSR_IA32_ARCH_CAPABILITIES => {
                let all = MSR_ARCH_CAP_MDS_NO
                    | MSR_ARCH_CAP_TAA_NO
                    | MSR_ARCH_CAP_SBDR_SSDP_NO
                    | MSR_ARCH_CAP_FBSDP_NO
                    | MSR_ARCH_CAP_PSDP_NO;
                if value & all == all {
                    value |= MSR_ARCH_CAP_FB_CLEAR;
                }
                value
            }
            _ => value,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixups() {
        let h = HostCpuid {
            supported: vec![CpuidEntry::new(1, 0, [0, 0, 1 << 21, 1])],
            irqchip: IrqchipMode::Off,
            ..Default::default()
        };
        // x2apic dropped without an in-kernel irqchip, hypervisor added.
        assert_eq!(h.supported_cpuid(1, 0, Reg::Ecx), 0x8000_0000);
        assert_eq!(
            h.supported_cpuid(1, 0, Reg::Edx),
            1 | 0x1000 | 0x10000 | 0x80 | 0x4000 | 0x1000_0000
        );
        // AMD aliases copied from leaf 1 EDX.
        assert_eq!(
            h.supported_cpuid(0x8000_0001, 0, Reg::Edx),
            (1 | 0x1000 | 0x10000 | 0x80 | 0x4000) & 0x183_f3ff
        );
    }

    #[test]
    fn vmx_ctls_decode() {
        let h = HostCpuid {
            feature_msrs: vec![(MSR_IA32_VMX_TRUE_PINBASED_CTLS, 0x0000_007f_0000_0016)],
            ..Default::default()
        };
        assert_eq!(h.supported_msr_feature(MSR_IA32_VMX_TRUE_PINBASED_CTLS), 0x69);
        assert_eq!(h.supported_msr_feature(0x48b), 0);
    }
}
