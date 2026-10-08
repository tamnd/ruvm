// SPDX-License-Identifier: GPL-2.0-or-later

//! The high part of the virt memory map, `virt_set_memmap()` and `virt_set_high_memmap()` of
//! hw/arm/virt.c.
//!
//! The regions above RAM are the second redistributor region, the CXL host registers, the
//! 256 MiB ECAM and the high PCIe MMIO window, in that order. Each one is aligned on its size
//! and none of them goes below 256 GiB, so that a guest with less than 255 GiB of RAM keeps the
//! legacy layout. A region that does not fit the physical address space is turned off. With
//! the compact layout (`compact-highmem=on`, the default) a region that is off takes no space,
//! so the next one moves down.

/// `VIRT_PCIE_ECAM`, the low ECAM window used when `highmem-ecam` is off.
pub const VIRT_PCIE_ECAM: u64 = 0x3f00_0000;
/// Its size, room for 16 buses.
pub const VIRT_PCIE_ECAM_SIZE: u64 = 0x0100_0000;
/// The size of `VIRT_HIGH_GIC_REDIST2`, room for 512 redistributors.
pub const VIRT_HIGH_GIC_REDIST2_SIZE: u64 = 64 << 20;
/// The size of `VIRT_CXL_HOST`, 16 host bridges of 64 KiB.
pub const VIRT_CXL_HOST_SIZE: u64 = 64 << 10 << 4;
/// The size of `VIRT_HIGH_PCIE_ECAM`, room for 256 buses.
pub const VIRT_HIGH_PCIE_ECAM_SIZE: u64 = 256 << 20;
/// `DEFAULT_HIGH_PCIE_MMIO_SIZE`, the default size of `VIRT_HIGH_PCIE_MMIO`.
pub const DEFAULT_HIGH_PCIE_MMIO_SIZE: u64 = 512 << 30;
/// `LEGACY_RAMLIMIT_BYTES`: the high IO regions never start below `VIRT_MEM` plus this.
pub const LEGACY_RAMLIMIT_BYTES: u64 = 255 << 30;
/// `PCIE_MMCFG_SIZE_MIN`, the ECAM space of one bus.
pub const PCIE_MMCFG_SIZE_MIN: u64 = 1 << 20;

/// One entry of the memory map, `MemMapEntry`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemMapEntry {
    /// The base address.
    pub base: u64,
    /// The size in bytes.
    pub size: u64,
}

/// The `highmem*` machine properties.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Highmem {
    /// `highmem`: use the physical address space above 4 GiB. Off, it is as if the CPU had 32
    /// physical address bits.
    pub highmem: bool,
    /// `compact-highmem`.
    pub compact: bool,
    /// `highmem-redists`: the second redistributor region.
    pub redists: bool,
    /// `highmem-ecam`: the 256 MiB ECAM above RAM rather than the 16 MiB one at 0x3f000000.
    pub ecam: bool,
    /// `highmem-mmio`: the second PCIe MMIO window.
    pub mmio: bool,
    /// `highmem-mmio-size`, the size of that window.
    pub mmio_size: u64,
}

impl Default for Highmem {
    fn default() -> Highmem {
        Highmem {
            highmem: true,
            compact: true,
            redists: true,
            ecam: true,
            mmio: true,
            mmio_size: DEFAULT_HIGH_PCIE_MMIO_SIZE,
        }
    }
}

/// `highmem-mmio-size` as `virt_set_highmem_mmio_size()` checks it.
pub fn check_highmem_mmio_size(size: u64) -> Result<(), String> {
    if !size.is_power_of_two() {
        return Err("highmem-mmio-size is not a power of 2".to_string());
    }
    if size < DEFAULT_HIGH_PCIE_MMIO_SIZE {
        return Err("highmem-mmio-size cannot be set to a lower value than the default (512 GiB)"
            .to_string());
    }
    Ok(())
}

/// The memory map of one board: where the high regions went and which ones are on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VirtMemmap {
    /// `VIRT_HIGH_GIC_REDIST2`, when `highmem-redists` is on and it fits.
    pub high_redist2: Option<MemMapEntry>,
    /// The ECAM the PCIe host uses: `VIRT_HIGH_PCIE_ECAM` when `highmem-ecam` is on and it
    /// fits, `VIRT_PCIE_ECAM` otherwise.
    pub ecam: MemMapEntry,
    /// Whether [`VirtMemmap::ecam`] is the high one.
    pub highmem_ecam: bool,
    /// `VIRT_HIGH_PCIE_MMIO`, when `highmem-mmio` is on and it fits.
    pub high_mmio: Option<MemMapEntry>,
    /// The highest guest physical address in use, `highest_gpa`.
    pub highest_gpa: u64,
}

impl VirtMemmap {
    /// The number of PCIe buses the ECAM has room for.
    pub fn nr_pcie_buses(&self) -> u64 {
        self.ecam.size / PCIE_MMCFG_SIZE_MIN
    }
}

/// `virt_set_memmap()` for RAM at `mem_base` of `ram_size` bytes and a CPU with `pa_bits`
/// physical address bits. There is no device memory, so the high regions start right above
/// RAM, rounded up to 1 GiB, or at 256 GiB if that is higher.
pub fn virt_set_memmap(
    mem_base: u64,
    ram_size: u64,
    pa_bits: u32,
    hm: &Highmem,
) -> Result<VirtMemmap, String> {
    // !highmem is exactly the same as limiting the PA space to 32bit, irrespective of the
    // underlying capabilities of the HW.
    let pa_bits = if hm.highmem { pa_bits } else { 32 };
    let limit = if pa_bits >= 64 { u128::from(u64::MAX) + 1 } else { 1u128 << pa_bits };

    let device_memory_base = (mem_base + ram_size).next_multiple_of(1 << 30);
    let memtop = device_memory_base;
    if u128::from(memtop) > limit {
        return Err(format!(
            "Addressing limited to {pa_bits} bits, but memory exceeds it by {} bytes",
            u128::from(memtop) - limit
        ));
    }
    let base = memtop.max(mem_base + LEGACY_RAMLIMIT_BYTES);

    // virt_set_high_memmap().
    let mut highest_gpa = memtop - 1;
    let mut base = u128::from(base);
    let regions = [
        (hm.redists, VIRT_HIGH_GIC_REDIST2_SIZE),
        // highmem_cxl is off until the CXL host region is created, which is after this.
        (false, VIRT_CXL_HOST_SIZE),
        (hm.ecam, VIRT_HIGH_PCIE_ECAM_SIZE),
        (hm.mmio, hm.mmio_size),
    ];
    let mut placed = [None; 4];
    for (i, &(enabled, size)) in regions.iter().enumerate() {
        let size = u128::from(size);
        let region_base = base.next_multiple_of(size);
        // Check each device to see if it fits in the PA space, moving highest_gpa as we go.
        // For compatibility, move highest_gpa for disabled fitting devices as well, if the
        // compact layout has been disabled. For each device that doesn't fit, disable it.
        let fits = region_base + size <= limit;
        let enabled = enabled && fits;
        if enabled {
            placed[i] = Some(MemMapEntry { base: region_base as u64, size: size as u64 });
        }
        if hm.compact && !enabled {
            continue;
        }
        base = region_base + size;
        if fits {
            highest_gpa = (base - 1) as u64;
        }
    }

    let (ecam, highmem_ecam) = match placed[2] {
        Some(e) => (e, true),
        None => (MemMapEntry { base: VIRT_PCIE_ECAM, size: VIRT_PCIE_ECAM_SIZE }, false),
    };
    Ok(VirtMemmap {
        high_redist2: placed[0],
        ecam,
        highmem_ecam,
        high_mmio: placed[3],
        highest_gpa,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MEM: u64 = 0x4000_0000;

    #[test]
    fn default_layout() {
        let m = virt_set_memmap(MEM, 128 << 20, 44, &Highmem::default()).unwrap();
        assert_eq!(m.high_redist2, Some(MemMapEntry { base: 256 << 30, size: 64 << 20 }));
        assert_eq!(m.ecam, MemMapEntry { base: 0x40_1000_0000, size: 256 << 20 });
        assert!(m.highmem_ecam);
        assert_eq!(m.nr_pcie_buses(), 256);
        assert_eq!(m.high_mmio, Some(MemMapEntry { base: 512 << 30, size: 512 << 30 }));
        assert_eq!(m.highest_gpa, (1 << 40) - 1);
    }

    #[test]
    fn small_pa_space() {
        // 40 bits: everything fits.
        let m = virt_set_memmap(MEM, 1 << 30, 40, &Highmem::default()).unwrap();
        assert!(m.high_mmio.is_some());
        // 39 bits: the high MMIO window does not.
        let m = virt_set_memmap(MEM, 1 << 30, 39, &Highmem::default()).unwrap();
        assert_eq!(m.high_mmio, None);
        assert!(m.highmem_ecam);
        assert_eq!(m.highest_gpa, 0x40_2000_0000 - 1);
        // highmem=off: none of the high regions.
        let hm = Highmem { highmem: false, ..Highmem::default() };
        let m = virt_set_memmap(MEM, 1 << 30, 44, &hm).unwrap();
        assert_eq!((m.high_redist2, m.high_mmio), (None, None));
        assert_eq!(m.ecam, MemMapEntry { base: VIRT_PCIE_ECAM, size: VIRT_PCIE_ECAM_SIZE });
        assert_eq!(m.nr_pcie_buses(), 16);
        let e = virt_set_memmap(MEM, 4 << 30, 44, &hm).unwrap_err();
        assert_eq!(e, "Addressing limited to 32 bits, but memory exceeds it by 1073741824 bytes");
    }

    #[test]
    fn legacy_layout() {
        // Without the compact layout a region that is off still takes its space.
        let hm = Highmem { compact: false, redists: false, ..Highmem::default() };
        let m = virt_set_memmap(MEM, 1 << 30, 44, &hm).unwrap();
        assert_eq!(m.high_redist2, None);
        assert_eq!(m.ecam.base, 0x40_1000_0000);
        let hm = Highmem { redists: false, ..Highmem::default() };
        let m = virt_set_memmap(MEM, 1 << 30, 44, &hm).unwrap();
        assert_eq!(m.ecam.base, 256 << 30);
    }

    #[test]
    fn mmio_size() {
        assert!(check_highmem_mmio_size(1 << 40).is_ok());
        assert_eq!(
            check_highmem_mmio_size(3 << 39).unwrap_err(),
            "highmem-mmio-size is not a power of 2"
        );
        assert_eq!(
            check_highmem_mmio_size(1 << 30).unwrap_err(),
            "highmem-mmio-size cannot be set to a lower value than the default (512 GiB)"
        );
    }
}
