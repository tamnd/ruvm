// SPDX-License-Identifier: GPL-2.0-or-later

//! The host CPU model HVF gives the guest, `hvf_arm_get_host_cpu_features()`, and the IPA size
//! rounding around it.

/// The PARange values, `pamax_map[]`: the physical address sizes ID_AA64MMFR0_EL1 can name.
const PAMAX: [u32; 7] = [32, 36, 40, 42, 44, 48, 52];

/// `round_down_to_parange_index()`: the largest PARange that fits in `bits`.
pub fn parange_index(bits: u32) -> u64 {
    PAMAX.iter().rposition(|&p| bits >= p).unwrap_or(0) as u64
}

/// `round_down_to_parange_bit_size()`.
pub fn parange_bits(bits: u32) -> u32 {
    PAMAX[parange_index(bits) as usize]
}

/// `clamp_id_aa64mmfr0_parange_to_ipa_size()`: ID_AA64MMFR0_EL1 with PARange cut down to
/// what the VM's IPA size can address.
pub fn clamp_mmfr0(mmfr0: u64, ipa_bits: u32) -> u64 {
    (mmfr0 & !0xf) | parange_index(ipa_bits)
}

/// The MIDR QEMU gives every HVF vCPU: Apple, architecture 0xf, everything else zero, since
/// Apple does not expose a per model MIDR to guests.
pub const MIDR: u64 = (0x61 << 24) | (0xf << 16);

/// The SCTLR_EL1 a vCPU resets to: the m1n1 boot value with SPAN set, so PAN is not set on
/// exception entry until the guest asks for it.
pub const RESET_SCTLR: u64 = 0x3010_0180 | 0x0080_0000;

/// `HV_FEATURE_REG_*`, the order of `hv_feature_reg_t`.
pub mod feature {
    pub const ID_AA64DFR0_EL1: u32 = 0;
    pub const ID_AA64DFR1_EL1: u32 = 1;
    pub const ID_AA64ISAR0_EL1: u32 = 2;
    pub const ID_AA64ISAR1_EL1: u32 = 3;
    pub const ID_AA64MMFR0_EL1: u32 = 4;
    pub const ID_AA64MMFR1_EL1: u32 = 5;
    pub const ID_AA64MMFR2_EL1: u32 = 6;
    pub const ID_AA64PFR0_EL1: u32 = 7;
    pub const ID_AA64PFR1_EL1: u32 = 8;
    pub const CTR_EL0: u32 = 9;
    pub const CLIDR_EL1: u32 = 10;
    pub const DCZID_EL0: u32 = 11;
}

/// The ID registers of the host CPU model.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct HostIdRegs {
    pub pfr0: u64,
    pub pfr1: u64,
    pub dfr0: u64,
    pub dfr1: u64,
    pub isar0: u64,
    pub isar1: u64,
    pub mmfr0: u64,
    pub mmfr1: u64,
    pub mmfr2: u64,
}

impl HostIdRegs {
    /// The adjustments `hvf_arm_get_host_cpu_features()` makes to what the framework reports:
    /// PARange clamped to the IPA size, PMUVer 1 when the framework's GIC brings its cycle
    /// counter, and no SME with nested virtualization, which Apple does not do together.
    /// `None` when the host would advertise AArch32 at EL0 or EL1, which QEMU refuses.
    pub fn adjust(mut self, ipa_bits: u32, irqchip: bool, el2: bool) -> Option<HostIdRegs> {
        self.mmfr0 = clamp_mmfr0(self.mmfr0, ipa_bits);
        if irqchip {
            self.dfr0 = (self.dfr0 & !(0xf << 8)) | (1 << 8);
        }
        if el2 {
            self.pfr1 &= !(0xf << 24);
        }
        if self.pfr0 & 0xff != 0x11 {
            return None;
        }
        Some(self)
    }

    /// ID_AA64PFR0_EL1 as `hvf_arch_init_vcpu()` sets it: GIC field 1 when a GICv3 CPU
    /// interface is there.
    pub fn pfr0_for_vcpu(&self, gicv3: bool) -> u64 {
        if gicv3 { self.pfr0 | (1 << 24) } else { self.pfr0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parange_rounds_down() {
        assert_eq!(parange_index(31), 0);
        assert_eq!(parange_index(36), 1);
        assert_eq!(parange_index(41), 2);
        assert_eq!(parange_index(47), 4);
        assert_eq!(parange_index(64), 6);
        assert_eq!(parange_bits(39), 36);
        assert_eq!(clamp_mmfr0(0x12_0f15, 40), 0x12_0f12);
    }

    #[test]
    fn midr_and_sctlr() {
        assert_eq!(MIDR, 0x610f_0000);
        assert_eq!(RESET_SCTLR, 0x30900180);
    }

    #[test]
    fn host_adjustments() {
        let host = HostIdRegs {
            pfr0: 0x1111,
            pfr1: 1 << 24 | 0x20,
            dfr0: 0xf << 8,
            mmfr0: 0x5,
            ..Default::default()
        };
        let a = host.adjust(36, true, true).unwrap();
        assert_eq!(a.mmfr0, 0x1);
        assert_eq!(a.dfr0, 1 << 8);
        assert_eq!(a.pfr1, 0x20);
        let b = host.adjust(48, false, false).unwrap();
        assert_eq!((b.dfr0, b.pfr1, b.mmfr0), (0xf << 8, 1 << 24 | 0x20, 0x5));
        assert_eq!(a.pfr0_for_vcpu(true), 0x0100_1111);
        let aarch32 = HostIdRegs { pfr0: 0x22, ..host };
        assert_eq!(aarch32.adjust(36, false, false), None);
    }
}
