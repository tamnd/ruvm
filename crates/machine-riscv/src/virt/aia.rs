// SPDX-License-Identifier: GPL-2.0-or-later

//! The AIA interrupt controllers of the virt board, hw/riscv/aia.c: `riscv_create_aia()` and
//! `imsic_num_bits()`.
//!
//! With `aia=aplic` the board has two APLIC domains in direct mode: the M level one, which
//! takes the device interrupts and delivers to each hart's M external interrupt, and its S
//! level child, which delivers to the S external interrupt. With `aia=aplic-imsic` both
//! domains are in MSI mode and each hart has an M level IMSIC (one file) and an S level IMSIC
//! (the S file and one file for each of the `aia-guests` guests); the domains send their
//! interrupts as messages into the IMSICs through system memory.
//!
//! Differences from QEMU:
//!
//! - Only socket 0 and only TCG: the KVM split irqchip side (`riscv_aplic_set_kvm_msicfgaddr()`)
//!   is not there.
//! - The CPU side of the IMSIC realize (`ext_smaia`, `ext_ssaia`, GEILEN and the
//!   `aia_ireg_rmw_fn`) is done by the board when it makes the harts.

use std::sync::Arc;

use ruvm_hw_core::IrqLine;
use ruvm_hw_intc::riscv_aplic::{RiscvAplic, RiscvAplicConfig, TYPE_RISCV_APLIC};
use ruvm_hw_intc::riscv_imsic::{
    IMSIC_MMIO_GROUP_MIN_SHIFT, RiscvImsic, TYPE_RISCV_IMSIC, imsic_hart_size,
};
use ruvm_mem::{MemorySystem, RegionId};

use super::map_io;

/// `imsic_num_bits()`: the smallest `n` with `1 << n` at least `count`.
pub fn imsic_num_bits(count: u32) -> u32 {
    let mut ret = 0;
    while (1u64 << ret) < u64::from(count) {
        ret += 1;
    }
    ret
}

/// A `MemMapEntry`: a base address and a size.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MemMapEntry {
    pub(crate) base: u64,
    pub(crate) size: u64,
}

/// The arguments of `riscv_create_aia()`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AiaParams {
    pub(crate) msimode: bool,
    pub(crate) aia_guests: u32,
    pub(crate) m_imsic_stride: u64,
    pub(crate) num_sources: u32,
    pub(crate) aplic_m: MemMapEntry,
    pub(crate) aplic_s: MemMapEntry,
    pub(crate) imsic_m: MemMapEntry,
    pub(crate) imsic_s: MemMapEntry,
    pub(crate) socket: u32,
    pub(crate) base_hartid: u32,
    pub(crate) hart_count: u32,
    pub(crate) num_msis: u32,
    pub(crate) num_prio_bits: u32,
}

/// The APLIC domains and IMSICs of one socket.
#[derive(Debug)]
pub struct Aia {
    aplic_m: Arc<RiscvAplic>,
    aplic_s: Arc<RiscvAplic>,
    imsic_m: Vec<Arc<RiscvImsic>>,
    imsic_s: Vec<Arc<RiscvImsic>>,
}

/// `riscv_create_aia()`: makes the IMSICs (in MSI mode) and the M and S level APLIC domains
/// and maps them into `system`. The outputs are left for the board to connect.
pub(crate) fn riscv_create_aia(
    mem: &MemorySystem,
    system: RegionId,
    p: &AiaParams,
) -> Result<Aia, String> {
    // The RISC-V Advanced Interrupt Architecture, Chapter 1.2. Limits
    assert!(p.num_sources <= 1023);
    let group = u64::from(p.socket) << IMSIC_MMIO_GROUP_MIN_SHIFT;
    let mut imsic_m = Vec::new();
    let mut imsic_s = Vec::new();
    if p.msimode {
        // Per-socket M-level IMSICs
        let addr = p.imsic_m.base + group;
        for i in 0..p.hart_count {
            let imsic = Arc::new(RiscvImsic::new(p.base_hartid + i, true, 1, p.num_msis));
            let at = addr + u64::from(i) * p.m_imsic_stride;
            map_io(mem, system, TYPE_RISCV_IMSIC, at, imsic.mmio_size(), imsic.clone())?;
            imsic_m.push(imsic);
        }

        // Per-socket S-level IMSICs
        let guest_bits = imsic_num_bits(p.aia_guests + 1);
        let addr = p.imsic_s.base + group;
        let stride = imsic_hart_size(guest_bits);
        for i in 0..p.hart_count {
            let pages = 1 + p.aia_guests;
            let imsic = Arc::new(RiscvImsic::new(p.base_hartid + i, false, pages, p.num_msis));
            let at = addr + u64::from(i) * stride;
            map_io(mem, system, TYPE_RISCV_IMSIC, at, imsic.mmio_size(), imsic.clone())?;
            imsic_s.push(imsic);
        }
    }

    let domain = |entry: MemMapEntry, mmode: bool| RiscvAplicConfig {
        aperture_size: entry.size,
        hartid_base: if p.msimode { 0 } else { p.base_hartid },
        num_harts: if p.msimode { 0 } else { p.hart_count },
        num_sources: p.num_sources,
        iprio_bits: p.num_prio_bits,
        msimode: p.msimode,
        mmode,
    };
    // Per-socket M-level APLIC
    let aplic_m = RiscvAplic::new(domain(p.aplic_m, true), None);
    let at = p.aplic_m.base + u64::from(p.socket) * p.aplic_m.size;
    map_io(mem, system, TYPE_RISCV_APLIC, at, aplic_m.mmio_size(), aplic_m.clone())?;

    // Per-socket S-level APLIC
    let aplic_s = RiscvAplic::new(domain(p.aplic_s, false), Some(&aplic_m));
    let at = p.aplic_s.base + u64::from(p.socket) * p.aplic_s.size;
    map_io(mem, system, TYPE_RISCV_APLIC, at, aplic_s.mmio_size(), aplic_s.clone())?;

    Ok(Aia { aplic_m, aplic_s, imsic_m, imsic_s })
}

impl Aia {
    /// The M level APLIC domain, the root that takes the device interrupts.
    pub fn aplic_m(&self) -> &Arc<RiscvAplic> {
        &self.aplic_m
    }

    /// The S level APLIC domain, the child of the M level one.
    pub fn aplic_s(&self) -> &Arc<RiscvAplic> {
        &self.aplic_s
    }

    /// The M level IMSICs, by hart; empty without `aplic-imsic`.
    pub fn imsic_m(&self) -> &[Arc<RiscvImsic>] {
        &self.imsic_m
    }

    /// The S level IMSICs, by hart; empty without `aplic-imsic`.
    pub fn imsic_s(&self) -> &[Arc<RiscvImsic>] {
        &self.imsic_s
    }

    /// Whether the domains are in MSI mode.
    pub fn msimode(&self) -> bool {
        self.aplic_m.config().msimode
    }

    /// Input `n` of the irqchip `riscv_create_aia()` returns, the M level domain.
    pub(crate) fn input(&self, n: u32) -> IrqLine {
        self.aplic_m.input(n)
    }

    /// The device resets of the domains and IMSICs.
    pub(crate) fn reset(&self) {
        for imsic in self.imsic_m.iter().chain(&self.imsic_s) {
            imsic.reset();
        }
        self.aplic_m.reset();
        self.aplic_s.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_bits() {
        assert_eq!(imsic_num_bits(0), 0);
        assert_eq!(imsic_num_bits(1), 0);
        assert_eq!(imsic_num_bits(2), 1);
        assert_eq!(imsic_num_bits(3), 2);
        assert_eq!(imsic_num_bits(4), 2);
        assert_eq!(imsic_num_bits(8), 3);
    }
}
