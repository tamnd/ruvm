// SPDX-License-Identifier: GPL-2.0-or-later

//! The Q35 host bridge: the MCH at 00:00.0 and the PCI Express host around it, from
//! hw/pci-host/q35.c.
//!
//! [`Q35PciHost::new`] does what `q35_host_initfn()`, `q35_host_realize()` and `mch_realize()`
//! do together. It maps CONFIG_ADDRESS and CONFIG_DATA at 0xcf8 and 0xcfc, creates the root bus
//! "pcie.0" over the PCI memory and I/O spaces, maps the PCI memory space under system memory at
//! priority -1 (`pc_pci_as_mapping_init()`), places the MCH at devfn 0 and builds its memory
//! regions:
//!
//! - the 13 PAM segments over 0xc0000 to 0xfffff (see [`crate::PamMemoryRegion`]),
//! - "smram-region", an alias of the PCI space over 0xa0000 to 0xbffff that hides low SMRAM
//!   from normal accesses,
//! - with SMM ranges on: "smram-open-high" at 0xfeda0000, the 4 GiB "smram" container the
//!   board puts into the SMM address space of the CPUs (with the "smram-low", "smram-high",
//!   "tseg-window" and "smbase-window" aliases in it), and the "tseg-blackhole" and
//!   "smbase-blackhole" regions that read as all ones and drop writes for normal accesses.
//!
//! Config writes to the MCH run `pci_default_write_config()` and then update the PAM
//! segments, the MMCONFIG window (PCIEXBAR), SMRAM, extended TSEG and SMBASE state, turning
//! aliases on and off exactly where QEMU calls `memory_region_set_enabled()`.
//!
//! As in QEMU nothing is decoded until the first reset: call [`Q35PciHost::reset`] (a system
//! reset) before running the guest.
//!
//! Differences from QEMU:
//!
//! - The QOM properties are fields of [`Q35Config`] and plain getters on [`Q35PciHost`].
//! - `pc_pci_hole64_start()` belongs to the PC machine, so the board passes its value in
//!   [`Q35Config::pc_pci_hole64_start`].
//! - A reserved PCIEXBAR length is ignored silently instead of being logged as a guest error.
//! - The `MCFG` property is only a getter: the window is placed by PCIEXBAR at reset.
//!
//! Not ported: VMState (`mch_post_load()` would be [`Mch::update`]), trace points, QOM type
//! registration, the "smram" link on the machine object and coalesced I/O on 0xcf8.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use ruvm_base::Error;
use ruvm_mem::AccessConstraints;
use ruvm_mem::{AccessCtx, AccessSize, MemError, MemResult, MemorySystem, MmioOps, RegionId};

use crate::bridge::{pci_bridge_get_base, pci_bridge_get_limit};
use crate::bus::PciBus;
use crate::device::{PciDevice, PciDeviceInfo, PciDeviceOps, PciDeviceVmState};
use crate::host::PciHostState;
use crate::pam::{
    PAM_BIOS_BASE, PAM_BIOS_SIZE, PAM_EXPAN_BASE, PAM_EXPAN_SIZE, PAM_REGIONS_COUNT,
    PamMemoryRegion, SMRAM_D_OPEN, SMRAM_G_SMRAME,
};
use crate::pcie_host::PcieHost;
use crate::regs::*;

/// `TYPE_Q35_HOST_DEVICE`.
pub const TYPE_Q35_HOST_DEVICE: &str = "q35-pcihost";
/// `TYPE_MCH_PCI_DEVICE`.
pub const TYPE_MCH_PCI_DEVICE: &str = "mch";

/// The property names QEMU uses, for boards and monitors that look them up by name.
pub const PCI_HOST_PROP_PCI_HOLE_START: &str = "pci-hole-start";
pub const PCI_HOST_PROP_PCI_HOLE_END: &str = "pci-hole-end";
pub const PCI_HOST_PROP_PCI_HOLE64_START: &str = "pci-hole64-start";
pub const PCI_HOST_PROP_PCI_HOLE64_END: &str = "pci-hole64-end";
pub const PCI_HOST_PROP_PCI_HOLE64_SIZE: &str = "pci-hole64-size";
pub const PCI_HOST_BELOW_4G_MEM_SIZE: &str = "below-4g-mem-size";
pub const PCI_HOST_ABOVE_4G_MEM_SIZE: &str = "above-4g-mem-size";
pub const PCI_HOST_PROP_SMM_RANGES: &str = "smm-ranges";
pub const PCIE_HOST_MCFG_BASE: &str = "MCFG";
pub const PCIE_HOST_MCFG_SIZE: &str = "mcfg_size";

/// `PCI_VENDOR_ID_INTEL`.
pub const PCI_VENDOR_ID_INTEL: u16 = 0x8086;
/// `PCI_DEVICE_ID_INTEL_P35_MCH`: QEMU uses the ID of the 82P35 MCH, which has no integrated
/// graphics.
pub const PCI_DEVICE_ID_INTEL_P35_MCH: u16 = 0x29c0;

/// `IO_APIC_DEFAULT_ADDRESS`, where the 32 bit PCI hole ends.
pub const IO_APIC_DEFAULT_ADDRESS: u64 = 0xfec0_0000;

/// `Q35_PCI_HOST_HOLE64_SIZE_DEFAULT`.
pub const Q35_PCI_HOST_HOLE64_SIZE_DEFAULT: u64 = 1 << 35;

pub const MCH_HOST_BRIDGE_CONFIG_ADDR: u64 = 0xcf8;
pub const MCH_HOST_BRIDGE_CONFIG_DATA: u64 = 0xcfc;

pub const MCH_HOST_BRIDGE_REVISION_DEFAULT: u8 = 0x0;

pub const MCH_HOST_BRIDGE_EXT_TSEG_MBYTES: usize = 0x50;
pub const MCH_HOST_BRIDGE_EXT_TSEG_MBYTES_SIZE: u32 = 2;
pub const MCH_HOST_BRIDGE_EXT_TSEG_MBYTES_QUERY: u16 = 0xffff;
pub const MCH_HOST_BRIDGE_EXT_TSEG_MBYTES_MAX: u16 = 0xfff;

pub const MCH_HOST_BRIDGE_SMBASE_SIZE: u64 = 128 * 1024;
pub const MCH_HOST_BRIDGE_SMBASE_ADDR: u64 = 0x30000;
pub const MCH_HOST_BRIDGE_F_SMBASE: usize = 0x9c;
pub const MCH_HOST_BRIDGE_F_SMBASE_QUERY: u8 = 0xff;
pub const MCH_HOST_BRIDGE_F_SMBASE_IN_RAM: u8 = 0x01;
pub const MCH_HOST_BRIDGE_F_SMBASE_LCK: u8 = 0x02;

/// PCIEXBAR, a 64 bit register.
pub const MCH_HOST_BRIDGE_PCIEXBAR: usize = 0x60;
pub const MCH_HOST_BRIDGE_PCIEXBAR_SIZE: u32 = 8;
pub const MCH_HOST_BRIDGE_PCIEXBAR_DEFAULT: u64 = 0xb000_0000;
/// 256 MiB.
pub const MCH_HOST_BRIDGE_PCIEXBAR_MAX: u64 = 0x1000_0000;
/// Bits 35 to 28.
pub const MCH_HOST_BRIDGE_PCIEXBAR_ADMSK: u64 = ((1 << 36) - 1) & !((1 << 28) - 1);
pub const MCH_HOST_BRIDGE_PCIEXBAR_128ADMSK: u64 = 1 << 27;
pub const MCH_HOST_BRIDGE_PCIEXBAR_64ADMSK: u64 = 1 << 26;
pub const MCH_HOST_BRIDGE_PCIEXBAR_LENGTH_MASK: u64 = 0x3 << 1;
pub const MCH_HOST_BRIDGE_PCIEXBAR_LENGTH_256M: u64 = 0x0 << 1;
pub const MCH_HOST_BRIDGE_PCIEXBAR_LENGTH_128M: u64 = 0x1 << 1;
pub const MCH_HOST_BRIDGE_PCIEXBAR_LENGTH_64M: u64 = 0x2 << 1;
pub const MCH_HOST_BRIDGE_PCIEXBAR_LENGTH_RVD: u64 = 0x3 << 1;
pub const MCH_HOST_BRIDGE_PCIEXBAREN: u64 = 1;

pub const MCH_HOST_BRIDGE_PAM_NB: usize = 7;
pub const MCH_HOST_BRIDGE_PAM_SIZE: u32 = 7;
pub const MCH_HOST_BRIDGE_PAM0: usize = 0x90;
pub const MCH_HOST_BRIDGE_PAM_BIOS_AREA: u64 = 0xf0000;
pub const MCH_HOST_BRIDGE_PAM_AREA_SIZE: u64 = 0x10000;
pub const MCH_HOST_BRIDGE_PAM1: usize = 0x91;
pub const MCH_HOST_BRIDGE_PAM_EXPAN_AREA: u64 = 0xc0000;
pub const MCH_HOST_BRIDGE_PAM_EXPAN_SIZE: u64 = 0x04000;
pub const MCH_HOST_BRIDGE_PAM2: usize = 0x92;
pub const MCH_HOST_BRIDGE_PAM3: usize = 0x93;
pub const MCH_HOST_BRIDGE_PAM4: usize = 0x94;
pub const MCH_HOST_BRIDGE_PAM_EXBIOS_AREA: u64 = 0xe0000;
pub const MCH_HOST_BRIDGE_PAM_EXBIOS_SIZE: u64 = 0x04000;
pub const MCH_HOST_BRIDGE_PAM5: usize = 0x95;
pub const MCH_HOST_BRIDGE_PAM6: usize = 0x96;
pub const MCH_HOST_BRIDGE_PAM_WE_HI: u8 = 0x2 << 4;
pub const MCH_HOST_BRIDGE_PAM_RE_HI: u8 = 0x1 << 4;
pub const MCH_HOST_BRIDGE_PAM_HI_MASK: u8 = 0x3 << 4;
pub const MCH_HOST_BRIDGE_PAM_WE_LO: u8 = 0x2;
pub const MCH_HOST_BRIDGE_PAM_RE_LO: u8 = 0x1;
pub const MCH_HOST_BRIDGE_PAM_LO_MASK: u8 = 0x3;
pub const MCH_HOST_BRIDGE_PAM_WE: u8 = 0x2;
pub const MCH_HOST_BRIDGE_PAM_RE: u8 = 0x1;
pub const MCH_HOST_BRIDGE_PAM_MASK: u8 = 0x3;

pub const MCH_HOST_BRIDGE_SMRAM: usize = 0x9d;
pub const MCH_HOST_BRIDGE_SMRAM_SIZE: u32 = 2;
pub const MCH_HOST_BRIDGE_SMRAM_D_OPEN: u8 = 1 << 6;
pub const MCH_HOST_BRIDGE_SMRAM_D_CLS: u8 = 1 << 5;
pub const MCH_HOST_BRIDGE_SMRAM_D_LCK: u8 = 1 << 4;
pub const MCH_HOST_BRIDGE_SMRAM_G_SMRAME: u8 = 1 << 3;
pub const MCH_HOST_BRIDGE_SMRAM_C_BASE_SEG_MASK: u8 = 0x7;
/// Hardwired to 0b010.
pub const MCH_HOST_BRIDGE_SMRAM_C_BASE_SEG: u8 = 0x2;
pub const MCH_HOST_BRIDGE_SMRAM_C_BASE: u64 = 0xa0000;
pub const MCH_HOST_BRIDGE_SMRAM_C_END: u64 = 0xc0000;
pub const MCH_HOST_BRIDGE_SMRAM_C_SIZE: u64 = 0x20000;
pub const MCH_HOST_BRIDGE_UPPER_SYSTEM_BIOS_END: u64 = 0x100000;
pub const MCH_HOST_BRIDGE_SMRAM_DEFAULT: u8 = MCH_HOST_BRIDGE_SMRAM_C_BASE_SEG;
pub const MCH_HOST_BRIDGE_SMRAM_WMASK: u8 = MCH_HOST_BRIDGE_SMRAM_D_OPEN
    | MCH_HOST_BRIDGE_SMRAM_D_CLS
    | MCH_HOST_BRIDGE_SMRAM_D_LCK
    | MCH_HOST_BRIDGE_SMRAM_G_SMRAME;
pub const MCH_HOST_BRIDGE_SMRAM_WMASK_LCK: u8 = MCH_HOST_BRIDGE_SMRAM_D_CLS;

pub const MCH_HOST_BRIDGE_ESMRAMC: usize = 0x9e;
pub const MCH_HOST_BRIDGE_ESMRAMC_H_SMRAME: u8 = 1 << 7;
pub const MCH_HOST_BRIDGE_ESMRAMC_E_SMERR: u8 = 1 << 6;
pub const MCH_HOST_BRIDGE_ESMRAMC_SM_CACHE: u8 = 1 << 5;
pub const MCH_HOST_BRIDGE_ESMRAMC_SM_L1: u8 = 1 << 4;
pub const MCH_HOST_BRIDGE_ESMRAMC_SM_L2: u8 = 1 << 3;
pub const MCH_HOST_BRIDGE_ESMRAMC_TSEG_SZ_MASK: u8 = 0x3 << 1;
pub const MCH_HOST_BRIDGE_ESMRAMC_TSEG_SZ_1MB: u8 = 0x0 << 1;
pub const MCH_HOST_BRIDGE_ESMRAMC_TSEG_SZ_2MB: u8 = 0x1 << 1;
pub const MCH_HOST_BRIDGE_ESMRAMC_TSEG_SZ_8MB: u8 = 0x2 << 1;
pub const MCH_HOST_BRIDGE_ESMRAMC_T_EN: u8 = 1;
pub const MCH_HOST_BRIDGE_ESMRAMC_DEFAULT: u8 = MCH_HOST_BRIDGE_ESMRAMC_SM_CACHE
    | MCH_HOST_BRIDGE_ESMRAMC_SM_L1
    | MCH_HOST_BRIDGE_ESMRAMC_SM_L2;
pub const MCH_HOST_BRIDGE_ESMRAMC_WMASK: u8 = MCH_HOST_BRIDGE_ESMRAMC_H_SMRAME
    | MCH_HOST_BRIDGE_ESMRAMC_TSEG_SZ_MASK
    | MCH_HOST_BRIDGE_ESMRAMC_T_EN;
pub const MCH_HOST_BRIDGE_ESMRAMC_WMASK_LCK: u8 = 0;

/// The PCIe root port at D1:F0.
pub const MCH_PCIE_DEV: u8 = 1;
pub const MCH_PCIE_FUNC: u8 = 0;

/// Where high SMRAM shows up.
const HIGH_SMRAM_BASE: u64 = 0xfeda_0000;

const GIB: u64 = 1 << 30;

/// The regions and properties the board wires into the Q35 host: the link properties
/// (`ram-mem`, `pci-mem`, `system-mem`, `io-mem`) and the value properties of `q35-pcihost` and
/// `mch`.
#[derive(Clone, Debug)]
pub struct Q35Config {
    /// `ram-mem`: all of guest RAM, what PAM, SMRAM and TSEG alias.
    pub ram_memory: RegionId,
    /// `pci-mem`: the PCI memory space. The MCH maps it under `system_memory` at priority -1.
    pub pci_address_space: RegionId,
    /// `system-mem`.
    pub system_memory: RegionId,
    /// `io-mem`: the I/O space, where 0xcf8 and 0xcfc go.
    pub address_space_io: RegionId,
    /// `below-4g-mem-size`: where RAM below 4 GiB ends, the start of TSEG's end and of the PCI
    /// hole.
    pub below_4g_mem_size: u64,
    /// `above-4g-mem-size`.
    pub above_4g_mem_size: u64,
    /// `pci-hole64-size`, 32 GiB by default.
    pub pci_hole64_size: u64,
    /// `smm-ranges`, on by default.
    pub has_smm_ranges: bool,
    /// `x-pci-hole64-fix`, on by default.
    pub pci_hole64_fix: bool,
    /// `mch.extended-tseg-mbytes`, 64 by default, at most 0xfff.
    pub ext_tseg_mbytes: u16,
    /// `mch.smbase-smram`, on by default.
    pub has_smram_at_smbase: bool,
    /// What `pc_pci_hole64_start()` returns for the machine: where the 64 bit hole starts when
    /// no device has a 64 bit BAR yet.
    pub pc_pci_hole64_start: u64,
}

impl Q35Config {
    /// The given regions with QEMU's default properties and no memory sizes.
    pub fn new(
        ram_memory: RegionId,
        pci_address_space: RegionId,
        system_memory: RegionId,
        address_space_io: RegionId,
    ) -> Q35Config {
        Q35Config {
            ram_memory,
            pci_address_space,
            system_memory,
            address_space_io,
            below_4g_mem_size: 0,
            above_4g_mem_size: 0,
            pci_hole64_size: Q35_PCI_HOST_HOLE64_SIZE_DEFAULT,
            has_smm_ranges: true,
            pci_hole64_fix: true,
            ext_tseg_mbytes: 64,
            has_smram_at_smbase: true,
            pc_pci_hole64_start: 0,
        }
    }
}

/// The regions only present with SMM ranges, from `mch_init_smram_regions()`.
#[derive(Debug)]
struct SmmRegions {
    open_high_smram: RegionId,
    smram: RegionId,
    low_smram: RegionId,
    high_smram: RegionId,
    tseg_blackhole: RegionId,
    tseg_window: RegionId,
    smbase_blackhole: RegionId,
    smbase_window: RegionId,
}

/// The MCH at 00:00.0, `MCHPCIState`. It is the [`PciDeviceOps`] of that function.
pub struct Mch {
    memory: Arc<MemorySystem>,
    pcie: Arc<PcieHost>,
    system_memory: RegionId,
    pam_regions: Vec<PamMemoryRegion>,
    smram_region: RegionId,
    smm: Option<SmmRegions>,
    has_smram_at_smbase: bool,
    below_4g_mem_size: u64,
    above_4g_mem_size: u64,
    pci_hole64_size: u64,
    ext_tseg_mbytes: u16,
    /// `pci_hole` as inclusive bounds, `None` while empty.
    pci_hole: Mutex<Option<(u64, u64)>>,
}

impl fmt::Debug for Mch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mch")
            .field("pam_regions", &self.pam_regions)
            .field("smram_region", &self.smram_region)
            .field("smm", &self.smm)
            .field("below_4g_mem_size", &self.below_4g_mem_size)
            .field("above_4g_mem_size", &self.above_4g_mem_size)
            .field("ext_tseg_mbytes", &self.ext_tseg_mbytes)
            .field("pci_hole", &self.pci_hole())
            .finish_non_exhaustive()
    }
}

/// `blackhole_ops`: reads all ones, drops writes.
#[derive(Debug)]
struct Blackhole;

impl MmioOps for Blackhole {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(0xffff_ffff)
    }

    fn write(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize, _v: u64) -> MemResult<()> {
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::exact(4)
    }
}

fn expect_mem(r: Result<(), MemError>) {
    if let Err(e) = r {
        panic!("q35: MCH memory region update failed: {e}");
    }
}

fn ranges_overlap(first1: u32, len1: u32, first2: u32, len2: u32) -> bool {
    let last1 = u64::from(first1) + u64::from(len1) - 1;
    let last2 = u64::from(first2) + u64::from(len2) - 1;
    !(last2 < u64::from(first1) || last1 < u64::from(first2))
}

impl Mch {
    /// `mch_realize()` minus the device registration.
    fn realize(
        memory: &Arc<MemorySystem>,
        pcie: Arc<PcieHost>,
        c: &Q35Config,
    ) -> Result<Mch, Error> {
        let err = |e: MemError| Error::generic(format!("mch: {e}"));
        if c.ext_tseg_mbytes > MCH_HOST_BRIDGE_EXT_TSEG_MBYTES_MAX {
            return Err(Error::generic(format!(
                "invalid extended-tseg-mbytes value: {}",
                c.ext_tseg_mbytes
            )));
        }

        // pc_pci_as_mapping_init(): below RAM.
        memory.add_subregion_overlap(c.system_memory, 0, c.pci_address_space, -1).map_err(err)?;

        let mut pam_regions = Vec::with_capacity(PAM_REGIONS_COUNT);
        let new_pam = |start, size| {
            PamMemoryRegion::new(
                memory,
                c.ram_memory,
                c.system_memory,
                c.pci_address_space,
                start,
                size,
            )
        };
        pam_regions.push(new_pam(PAM_BIOS_BASE, PAM_BIOS_SIZE).map_err(err)?);
        for i in 0..PAM_REGIONS_COUNT as u64 - 1 {
            pam_regions
                .push(new_pam(PAM_EXPAN_BASE + i * PAM_EXPAN_SIZE, PAM_EXPAN_SIZE).map_err(err)?);
        }

        // Despite the name this is not SMM specific: it makes the PCI space appear over the
        // low SMRAM range, and is disabled when low SMRAM should be visible. Without SMM
        // ranges it simply stays enabled.
        let smram_region = memory
            .new_alias(
                "smram-region",
                c.pci_address_space,
                MCH_HOST_BRIDGE_SMRAM_C_BASE,
                u128::from(MCH_HOST_BRIDGE_SMRAM_C_SIZE),
            )
            .map_err(err)?;
        memory
            .add_subregion_overlap(c.system_memory, MCH_HOST_BRIDGE_SMRAM_C_BASE, smram_region, 1)
            .map_err(err)?;
        memory.set_enabled(smram_region, true).map_err(err)?;

        let smm = if c.has_smm_ranges {
            Some(Self::init_smram_regions(memory, c).map_err(err)?)
        } else {
            None
        };

        Ok(Mch {
            memory: Arc::clone(memory),
            pcie,
            system_memory: c.system_memory,
            pam_regions,
            smram_region,
            smm,
            has_smram_at_smbase: c.has_smram_at_smbase,
            below_4g_mem_size: c.below_4g_mem_size,
            above_4g_mem_size: c.above_4g_mem_size,
            pci_hole64_size: c.pci_hole64_size,
            ext_tseg_mbytes: c.ext_tseg_mbytes,
            pci_hole: Mutex::new(None),
        })
    }

    /// `mch_init_smram_regions()`.
    fn init_smram_regions(memory: &MemorySystem, c: &Q35Config) -> Result<SmmRegions, MemError> {
        let ram = c.ram_memory;
        let sys = c.system_memory;
        let c_base = MCH_HOST_BRIDGE_SMRAM_C_BASE;
        let c_size = u128::from(MCH_HOST_BRIDGE_SMRAM_C_SIZE);

        let open_high_smram = memory.new_alias("smram-open-high", ram, c_base, c_size)?;
        memory.add_subregion_overlap(sys, HIGH_SMRAM_BASE, open_high_smram, 1)?;
        memory.set_enabled(open_high_smram, false)?;

        // SMRAM as seen by CPUs in SMM.
        let smram = memory.new_container("smram", u128::from(4 * GIB))?;
        memory.set_enabled(smram, true)?;
        let low_smram = memory.new_alias("smram-low", ram, c_base, c_size)?;
        memory.set_enabled(low_smram, true)?;
        memory.add_subregion(smram, c_base, low_smram)?;
        let high_smram = memory.new_alias("smram-high", ram, c_base, c_size)?;
        memory.set_enabled(high_smram, true)?;
        memory.add_subregion(smram, HIGH_SMRAM_BASE, high_smram)?;

        let tseg_blackhole = memory.new_io("tseg-blackhole", 0, Arc::new(Blackhole))?;
        memory.set_enabled(tseg_blackhole, false)?;
        memory.add_subregion_overlap(sys, c.below_4g_mem_size, tseg_blackhole, 1)?;

        let tseg_window = memory.new_alias("tseg-window", ram, c.below_4g_mem_size, 0)?;
        memory.set_enabled(tseg_window, false)?;
        memory.add_subregion(smram, c.below_4g_mem_size, tseg_window)?;

        // Not what hardware does, a QEMU specific hack.
        let smbase_size = u128::from(MCH_HOST_BRIDGE_SMBASE_SIZE);
        let smbase_blackhole =
            memory.new_io("smbase-blackhole", smbase_size, Arc::new(Blackhole))?;
        memory.set_enabled(smbase_blackhole, false)?;
        memory.add_subregion_overlap(sys, MCH_HOST_BRIDGE_SMBASE_ADDR, smbase_blackhole, 1)?;

        let smbase_window =
            memory.new_alias("smbase-window", ram, MCH_HOST_BRIDGE_SMBASE_ADDR, smbase_size)?;
        memory.set_enabled(smbase_window, false)?;
        memory.add_subregion(smram, MCH_HOST_BRIDGE_SMBASE_ADDR, smbase_window)?;

        Ok(SmmRegions {
            open_high_smram,
            smram,
            low_smram,
            high_smram,
            tseg_blackhole,
            tseg_window,
            smbase_blackhole,
            smbase_window,
        })
    }

    fn lock_hole(&self) -> MutexGuard<'_, Option<(u64, u64)>> {
        self.pci_hole.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// `has_smm_ranges`.
    pub fn has_smm_ranges(&self) -> bool {
        self.smm.is_some()
    }

    /// The "smram" container, SMRAM as CPUs in SMM see it. The board maps it over system
    /// memory in the SMM address space. `None` without SMM ranges.
    pub fn smram(&self) -> Option<RegionId> {
        self.smm.as_ref().map(|s| s.smram)
    }

    /// The "smram-region" alias that shows the PCI space over 0xa0000 to 0xbffff.
    pub fn smram_region(&self) -> RegionId {
        self.smram_region
    }

    /// PAM segment `i`: 0 is the BIOS area at 0xf0000, 1 to 12 the 16 KiB segments from
    /// 0xc0000.
    pub fn pam_region(&self, i: usize) -> &PamMemoryRegion {
        &self.pam_regions[i]
    }

    /// `pci_hole` as inclusive bounds, `None` while it is empty (before the first reset).
    pub fn pci_hole(&self) -> Option<(u64, u64)> {
        *self.lock_hole()
    }

    pub fn below_4g_mem_size(&self) -> u64 {
        self.below_4g_mem_size
    }

    pub fn above_4g_mem_size(&self) -> u64 {
        self.above_4g_mem_size
    }

    pub fn pci_hole64_size(&self) -> u64 {
        self.pci_hole64_size
    }

    pub fn ext_tseg_mbytes(&self) -> u16 {
        self.ext_tseg_mbytes
    }

    /// `mch_update_pciexbar()`.
    fn update_pciexbar(&self, dev: &PciDevice) {
        let pciexbar = dev.with_config(|c| pci_get_quad(c.config, MCH_HOST_BRIDGE_PCIEXBAR));
        let enable = pciexbar & MCH_HOST_BRIDGE_PCIEXBAREN != 0;
        let mut addr_mask = MCH_HOST_BRIDGE_PCIEXBAR_ADMSK;
        let length = match pciexbar & MCH_HOST_BRIDGE_PCIEXBAR_LENGTH_MASK {
            MCH_HOST_BRIDGE_PCIEXBAR_LENGTH_256M => 256 << 20,
            MCH_HOST_BRIDGE_PCIEXBAR_LENGTH_128M => {
                addr_mask |= MCH_HOST_BRIDGE_PCIEXBAR_128ADMSK;
                128 << 20
            }
            MCH_HOST_BRIDGE_PCIEXBAR_LENGTH_64M => {
                addr_mask |= MCH_HOST_BRIDGE_PCIEXBAR_64ADMSK | MCH_HOST_BRIDGE_PCIEXBAR_128ADMSK;
                64 << 20
            }
            // Reserved length: QEMU logs a guest error and leaves the window alone.
            _ => return,
        };
        let addr = pciexbar & addr_mask;
        self.pcie.mmcfg_update(enable, addr, length);
    }

    /// `mch_update_pam()`.
    fn update_pam(&self, dev: &PciDevice) {
        let pam: [u8; MCH_HOST_BRIDGE_PAM_NB] = dev.with_config(|c| {
            let mut v = [0u8; MCH_HOST_BRIDGE_PAM_NB];
            v.copy_from_slice(&c.config[MCH_HOST_BRIDGE_PAM0..][..MCH_HOST_BRIDGE_PAM_NB]);
            v
        });
        let _t = self.memory.transaction();
        for (i, r) in self.pam_regions.iter().enumerate() {
            r.update(i, pam[i.div_ceil(2)]);
        }
    }

    /// `mch_update_smram()`.
    fn update_smram(&self, dev: &PciDevice) {
        let Some(smm) = &self.smm else { return };
        let (smram, esmramc) = dev.with_config(|c| {
            // SMRAM.D_LCK.
            if c.config[MCH_HOST_BRIDGE_SMRAM] & MCH_HOST_BRIDGE_SMRAM_D_LCK != 0 {
                c.config[MCH_HOST_BRIDGE_SMRAM] &= !MCH_HOST_BRIDGE_SMRAM_D_OPEN;
                c.wmask[MCH_HOST_BRIDGE_SMRAM] = MCH_HOST_BRIDGE_SMRAM_WMASK_LCK;
                c.wmask[MCH_HOST_BRIDGE_ESMRAMC] = MCH_HOST_BRIDGE_ESMRAMC_WMASK_LCK;
            }
            (c.config[MCH_HOST_BRIDGE_SMRAM], c.config[MCH_HOST_BRIDGE_ESMRAMC])
        });
        let h_smrame = esmramc & MCH_HOST_BRIDGE_ESMRAMC_H_SMRAME != 0;
        let m = &*self.memory;

        let _t = m.transaction();

        if smram & SMRAM_D_OPEN != 0 {
            // Hide (!) low SMRAM if H_SMRAME is set, and show high SMRAM.
            expect_mem(m.set_enabled(self.smram_region, h_smrame));
            expect_mem(m.set_enabled(smm.open_high_smram, h_smrame));
        } else {
            // Hide both high and low SMRAM.
            expect_mem(m.set_enabled(self.smram_region, true));
            expect_mem(m.set_enabled(smm.open_high_smram, false));
        }

        if smram & SMRAM_G_SMRAME != 0 {
            expect_mem(m.set_enabled(smm.low_smram, !h_smrame));
            expect_mem(m.set_enabled(smm.high_smram, h_smrame));
        } else {
            expect_mem(m.set_enabled(smm.low_smram, false));
            expect_mem(m.set_enabled(smm.high_smram, false));
        }

        let tseg_size: u64 =
            if esmramc & MCH_HOST_BRIDGE_ESMRAMC_T_EN != 0 && smram & SMRAM_G_SMRAME != 0 {
                match esmramc & MCH_HOST_BRIDGE_ESMRAMC_TSEG_SZ_MASK {
                    MCH_HOST_BRIDGE_ESMRAMC_TSEG_SZ_1MB => 1 << 20,
                    MCH_HOST_BRIDGE_ESMRAMC_TSEG_SZ_2MB => 2 << 20,
                    MCH_HOST_BRIDGE_ESMRAMC_TSEG_SZ_8MB => 8 << 20,
                    _ => u64::from(self.ext_tseg_mbytes) << 20,
                }
            } else {
                0
            };
        let tseg_base = self.below_4g_mem_size.wrapping_sub(tseg_size);

        expect_mem(m.del_subregion(self.system_memory, smm.tseg_blackhole));
        expect_mem(m.set_enabled(smm.tseg_blackhole, tseg_size != 0));
        expect_mem(m.set_size(smm.tseg_blackhole, u128::from(tseg_size)));
        expect_mem(m.add_subregion_overlap(self.system_memory, tseg_base, smm.tseg_blackhole, 1));

        expect_mem(m.set_enabled(smm.tseg_window, tseg_size != 0));
        expect_mem(m.set_size(smm.tseg_window, u128::from(tseg_size)));
        expect_mem(m.set_address(smm.tseg_window, tseg_base));
        expect_mem(m.set_alias_offset(smm.tseg_window, tseg_base));
    }

    /// `mch_update_ext_tseg_mbytes()`.
    fn update_ext_tseg_mbytes(&self, dev: &PciDevice) {
        let ext = self.ext_tseg_mbytes;
        dev.with_config(|c| {
            if ext > 0
                && pci_get_word(c.config, MCH_HOST_BRIDGE_EXT_TSEG_MBYTES)
                    == MCH_HOST_BRIDGE_EXT_TSEG_MBYTES_QUERY
            {
                pci_set_word(c.config, MCH_HOST_BRIDGE_EXT_TSEG_MBYTES, ext);
            }
        });
    }

    /// `mch_update_smbase_smram()`.
    fn update_smbase_smram(&self, dev: &PciDevice) {
        let Some(smm) = &self.smm else { return };
        if !self.has_smram_at_smbase {
            return;
        }
        let lck = dev.with_config(|c| {
            let reg = &mut c.config[MCH_HOST_BRIDGE_F_SMBASE];
            if *reg == MCH_HOST_BRIDGE_F_SMBASE_QUERY {
                c.wmask[MCH_HOST_BRIDGE_F_SMBASE] = MCH_HOST_BRIDGE_F_SMBASE_LCK;
                *reg = MCH_HOST_BRIDGE_F_SMBASE_IN_RAM;
                return None;
            }
            // The value may come from a register write, reset or migration; keep wmask in
            // sync with it whatever the source.
            if *reg == MCH_HOST_BRIDGE_F_SMBASE_IN_RAM {
                c.wmask[MCH_HOST_BRIDGE_F_SMBASE] = MCH_HOST_BRIDGE_F_SMBASE_LCK;
                return None;
            }
            if *reg & MCH_HOST_BRIDGE_F_SMBASE_LCK != 0 {
                // Lock the register at 0x2 and refuse all writes.
                c.wmask[MCH_HOST_BRIDGE_F_SMBASE] = 0;
                *reg = MCH_HOST_BRIDGE_F_SMBASE_LCK;
            }
            Some(*reg & MCH_HOST_BRIDGE_F_SMBASE_LCK != 0)
        });
        let Some(lck) = lck else { return };
        let _t = self.memory.transaction();
        expect_mem(self.memory.set_enabled(smm.smbase_blackhole, lck));
        expect_mem(self.memory.set_enabled(smm.smbase_window, lck));
    }

    /// `mch_update()`: brings every region in line with config space, as after reset or
    /// migration.
    pub fn update(&self, dev: &PciDevice) {
        self.update_pciexbar(dev);
        self.update_pam(dev);
        if self.smm.is_some() {
            self.update_smram(dev);
            self.update_ext_tseg_mbytes(dev);
            self.update_smbase_smram(dev);
        }
        // The PCI hole goes from the end of low RAM to the IOAPIC. MMCONFIG is left out by
        // the DSDT builder.
        *self.lock_hole() = Some((self.below_4g_mem_size, IO_APIC_DEFAULT_ADDRESS - 1));
    }
}

impl PciDeviceOps for Mch {
    /// `mch_write_config()`.
    fn config_write(&self, dev: &PciDevice, addr: u32, val: u32, len: u32) {
        dev.default_write_config(addr, val, len);

        if ranges_overlap(addr, len, MCH_HOST_BRIDGE_PAM0 as u32, MCH_HOST_BRIDGE_PAM_SIZE) {
            self.update_pam(dev);
        }
        if ranges_overlap(addr, len, MCH_HOST_BRIDGE_PCIEXBAR as u32, MCH_HOST_BRIDGE_PCIEXBAR_SIZE)
        {
            self.update_pciexbar(dev);
        }
        if self.smm.is_some() {
            if ranges_overlap(addr, len, MCH_HOST_BRIDGE_SMRAM as u32, MCH_HOST_BRIDGE_SMRAM_SIZE) {
                self.update_smram(dev);
            }
            if ranges_overlap(
                addr,
                len,
                MCH_HOST_BRIDGE_EXT_TSEG_MBYTES as u32,
                MCH_HOST_BRIDGE_EXT_TSEG_MBYTES_SIZE,
            ) {
                self.update_ext_tseg_mbytes(dev);
            }
            if ranges_overlap(addr, len, MCH_HOST_BRIDGE_F_SMBASE as u32, 1) {
                self.update_smbase_smram(dev);
            }
        }
    }

    /// `mch_reset()`.
    fn reset(&self, dev: &PciDevice) {
        let smm = self.smm.is_some();
        let ext = self.ext_tseg_mbytes;
        dev.with_config(|c| {
            pci_set_quad(c.config, MCH_HOST_BRIDGE_PCIEXBAR, MCH_HOST_BRIDGE_PCIEXBAR_DEFAULT);
            if smm {
                c.config[MCH_HOST_BRIDGE_SMRAM] = MCH_HOST_BRIDGE_SMRAM_DEFAULT;
                c.config[MCH_HOST_BRIDGE_ESMRAMC] = MCH_HOST_BRIDGE_ESMRAMC_DEFAULT;
                c.wmask[MCH_HOST_BRIDGE_SMRAM] = MCH_HOST_BRIDGE_SMRAM_WMASK;
                c.wmask[MCH_HOST_BRIDGE_ESMRAMC] = MCH_HOST_BRIDGE_ESMRAMC_WMASK;
                if ext > 0 {
                    pci_set_word(
                        c.config,
                        MCH_HOST_BRIDGE_EXT_TSEG_MBYTES,
                        MCH_HOST_BRIDGE_EXT_TSEG_MBYTES_QUERY,
                    );
                }
                c.config[MCH_HOST_BRIDGE_F_SMBASE] = 0;
                c.wmask[MCH_HOST_BRIDGE_F_SMBASE] = 0xff;
            }
        });
        self.update(dev);
    }
}

/// `vmstate_mch` (version 1): the MCH's config space. A byte that used to be `smm_enabled`
/// follows it on the wire.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MchVmState {
    pub parent_obj: PciDeviceVmState,
}

/// The Q35 PCI Express host bridge, `Q35PCIHost`, with its MCH.
pub struct Q35PciHost {
    host: Arc<PciHostState>,
    pcie: Arc<PcieHost>,
    mch: Arc<Mch>,
    mch_dev: Arc<PciDevice>,
    conf_mem: RegionId,
    data_mem: RegionId,
    pci_hole64_fix: bool,
    pc_pci_hole64_start: u64,
}

impl fmt::Debug for Q35PciHost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Q35PciHost")
            .field("host", &self.host)
            .field("pcie", &self.pcie)
            .field("mch", &self.mch)
            .field("pci_hole64_fix", &self.pci_hole64_fix)
            .finish_non_exhaustive()
    }
}

impl Q35PciHost {
    /// Creates and realizes the host and its MCH, `q35_host_initfn()` plus
    /// `q35_host_realize()` plus `mch_realize()`. Nothing is decoded until [`Self::reset`].
    pub fn new(memory: Arc<MemorySystem>, config: Q35Config) -> Result<Q35PciHost, Error> {
        let bus = PciBus::new_root(
            "pcie.0",
            Arc::clone(&memory),
            config.pci_address_space,
            config.address_space_io,
            0,
        );
        // TYPE_PCIE_BUS.
        bus.set_extended_config_space(true);
        // q35_host_root_bus_path().
        bus.set_root_bus_path("0000:00");

        let host = PciHostState::new(Arc::clone(&bus));
        let (conf_mem, data_mem) = host.map_ioports(&memory, config.address_space_io)?;

        let pcie = PcieHost::new(Arc::clone(&memory), config.system_memory, Arc::clone(&bus))
            .map_err(|e| Error::generic(format!("q35: {e}")))?;

        let mch = Arc::new(Mch::realize(&memory, Arc::clone(&pcie), &config)?);
        let info = PciDeviceInfo {
            name: TYPE_MCH_PCI_DEVICE.to_string(),
            vendor_id: PCI_VENDOR_ID_INTEL,
            device_id: PCI_DEVICE_ID_INTEL_P35_MCH,
            revision: MCH_HOST_BRIDGE_REVISION_DEFAULT,
            class_id: PCI_CLASS_BRIDGE_HOST,
            ..PciDeviceInfo::default()
        };
        let mch_dev = bus.register_device(&info, Some(pci_devfn(0, 0)))?;
        mch_dev.set_ops(Arc::clone(&mch) as Arc<dyn PciDeviceOps>);

        Ok(Q35PciHost {
            host,
            pcie,
            mch,
            mch_dev,
            conf_mem,
            data_mem,
            pci_hole64_fix: config.pci_hole64_fix,
            pc_pci_hole64_start: config.pc_pci_hole64_start,
        })
    }

    /// The root bus, "pcie.0".
    pub fn bus(&self) -> &Arc<PciBus> {
        self.host.bus()
    }

    /// The 0xcf8/0xcfc config mechanism.
    pub fn host_state(&self) -> &Arc<PciHostState> {
        &self.host
    }

    /// The MMCONFIG window.
    pub fn pcie_host(&self) -> &Arc<PcieHost> {
        &self.pcie
    }

    pub fn mch(&self) -> &Arc<Mch> {
        &self.mch
    }

    /// The MCH function, 00:00.0.
    pub fn mch_device(&self) -> &Arc<PciDevice> {
        &self.mch_dev
    }

    /// The MCH's `mch` section.
    pub fn mch_vmstate_save(&self) -> MchVmState {
        MchVmState { parent_obj: self.mch_dev.vmstate_save() }
    }

    /// Loads the MCH's `mch` section, then `mch_post_load()`: PAM, SMRAM and MMCONFIG follow the
    /// loaded config space.
    pub fn mch_vmstate_load(&self, v: &MchVmState) -> Result<(), String> {
        self.mch_dev.vmstate_load(&v.parent_obj)?;
        self.mch.update(&self.mch_dev);
        Ok(())
    }

    /// The "pci-conf-idx" and "pci-conf-data" I/O regions.
    pub fn conf_regions(&self) -> (RegionId, RegionId) {
        (self.conf_mem, self.data_mem)
    }

    /// A system reset of the root bus, which runs `mch_reset()` among others.
    pub fn reset(&self) {
        self.bus().reset();
    }

    /// `pci-hole-start`: 0 while the hole is empty.
    pub fn pci_hole_start(&self) -> u32 {
        let v = self.mch.pci_hole().map_or(0, |(lob, _)| lob);
        u32::try_from(v).expect("the 32 bit PCI hole starts below 4 GiB")
    }

    /// `pci-hole-end`: one past the last byte, 0 while the hole is empty.
    pub fn pci_hole_end(&self) -> u32 {
        let v = self.mch.pci_hole().map_or(0, |(_, upb)| upb + 1);
        u32::try_from(v).expect("the 32 bit PCI hole ends below 4 GiB")
    }

    /// `q35_host_get_pci_hole64_start_value()`: the lowest 64 bit BAR or bridge prefetch
    /// window above 4 GiB, or with `x-pci-hole64-fix` the machine's default when there is none.
    fn pci_hole64_start_value(&self) -> u64 {
        let value = pci_bus_get_w64_range(self.bus()).map_or(0, |(lob, _)| lob);
        if value == 0 && self.pci_hole64_fix { self.pc_pci_hole64_start } else { value }
    }

    /// `pci-hole64-start`.
    pub fn pci_hole64_start(&self) -> u64 {
        self.pci_hole64_start_value()
    }

    /// `pci-hole64-end`: past the highest 64 bit resource, or with `x-pci-hole64-fix` at least
    /// `pci-hole64-size` past the start, rounded up to 1 GiB.
    pub fn pci_hole64_end(&self) -> u64 {
        let hole64_start = self.pci_hole64_start_value();
        let value = pci_bus_get_w64_range(self.bus()).map_or(0, |(_, upb)| upb + 1);
        let hole64_end = hole64_start.wrapping_add(self.mch.pci_hole64_size).next_multiple_of(GIB);
        if self.pci_hole64_fix && value < hole64_end { hole64_end } else { value }
    }

    /// `pci-hole64-size`.
    pub fn pci_hole64_size(&self) -> u64 {
        self.mch.pci_hole64_size
    }

    /// `below-4g-mem-size`.
    pub fn below_4g_mem_size(&self) -> u64 {
        self.mch.below_4g_mem_size
    }

    /// `above-4g-mem-size`.
    pub fn above_4g_mem_size(&self) -> u64 {
        self.mch.above_4g_mem_size
    }

    /// `MCFG`: where the MMCONFIG window is mapped, or
    /// [`crate::PCIE_BASE_ADDR_UNMAPPED`].
    pub fn mcfg_base(&self) -> u64 {
        self.pcie.base_addr()
    }

    /// `mcfg_size`.
    pub fn mcfg_size(&self) -> u64 {
        self.pcie.size()
    }

    /// `smm-ranges`.
    pub fn has_smm_ranges(&self) -> bool {
        self.mch.has_smm_ranges()
    }

    /// The "smram" container for the SMM address space, see [`Mch::smram`].
    pub fn smram(&self) -> Option<RegionId> {
        self.mch.smram()
    }
}

/// `pci_bus_get_w64_range()`: the span of 64 bit memory BARs and bridge prefetch windows above
/// 4 GiB of the functions on `bus` (not below it, as in QEMU), as inclusive bounds.
pub fn pci_bus_get_w64_range(bus: &PciBus) -> Option<(u64, u64)> {
    let mut range = None;
    w64_walk(bus, &mut range);
    range
}

fn extend(range: &mut Option<(u64, u64)>, lob: u64, upb: u64) {
    *range = Some(match *range {
        None => (lob, upb),
        Some((l, u)) => (l.min(lob), u.max(upb)),
    });
}

/// `pci_for_each_device_under_bus()` with `pci_dev_get_w64()`.
fn w64_walk(bus: &PciBus, range: &mut Option<(u64, u64)>) {
    for dev in bus.devices() {
        let cmd = dev.default_read_config(PCI_COMMAND as u32, 2) as u16;
        if cmd & PCI_COMMAND_MEMORY == 0 {
            continue;
        }
        if dev.is_bridge() {
            let config = dev.config_bytes();
            let base = pci_bridge_get_base(&config, PCI_BASE_ADDRESS_MEM_PREFETCH).max(1 << 32);
            let limit = pci_bridge_get_limit(&config, PCI_BASE_ADDRESS_MEM_PREFETCH);
            if limit >= base {
                extend(range, base, limit);
            }
        }
        for i in 0..PCI_NUM_REGIONS {
            let Some(r) = dev.bar_info(i) else { continue };
            if r.size == 0
                || r.type_ & PCI_BASE_ADDRESS_SPACE_IO != 0
                || r.type_ & PCI_BASE_ADDRESS_MEM_TYPE_64 == 0
            {
                continue;
            }
            let lob = dev.bar_address(i);
            if lob == PCI_BAR_UNMAPPED {
                continue;
            }
            let upb = lob.wrapping_add(r.size - 1);
            let lob = lob.max(1 << 32);
            if upb >= lob {
                extend(range, lob, upb);
            }
        }
    }
}
