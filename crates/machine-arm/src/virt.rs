// SPDX-License-Identifier: GPL-2.0-or-later

//! The `virt` board, hw/arm/virt.c, with a GICv3 or a GICv2 and AArch64 CPUs.
//!
//! # What is there
//!
//! RAM (`mach-virt.ram`) at 0x40000000, the GICv3 distributor at 0x08000000 and the
//! redistributors at 0x080a0000, with those of the CPUs past 123 in the high memory region (see
//! [`memmap`]), the ITS at 0x08080000 with LPIs in the GIC unless `msi=off`, the PCIe functions
//! sending their MSIs to it with their requester ID (`msi-map`), the generic timers wired to
//! their PPIs, the PL011 UART at
//! 0x09000000 (SPI 1), the PL031 RTC at 0x09010000 (SPI 2), the generic PCIe host bridge
//! (`gpex-pcihost`) with its MMIO window at 0x10000000, its I/O port window at 0x3eff0000, its
//! ECAM and its high MMIO window above RAM (see [`memmap`]) and INTx on SPIs 3 to 6, fw_cfg
//! with DMA at 0x09020000, 32 virtio-mmio transports from 0x0a000000 (SPIs 16 to 47) and an
//! empty platform bus window at 0x0c000000, and the two CFI flashes at 0 and 0x04000000, the
//! first one holding `-bios`. `iommu=smmuv3` puts the SMMUv3 (stage 1, stage 2 and nested) at
//! 0x09050000 (SPIs 74 to 77) in front of the root bus: each PCIe function does its DMA and
//! sends its MSIs through an address space of its own that the SMMU translates. A
//! second `-serial` adds the second PL011 at 0x09040000 (SPI 8). `virtualization=on` keeps EL2
//! and `secure=on` keeps EL3 and adds the secure UART at 0x09040000 and the secure RAM at
//! 0x0e000000. `gic-version=2` puts a GICv2 in place of the GICv3, its distributor at
//! 0x08000000 and its CPU interface at 0x08010000, for at most 8 CPUs, with the GICv2m MSI
//! frame at 0x08020000 (SPIs 48 to 111) unless `msi=off`. PSCI goes through HVC, or SMC with `virtualization=on`, and is left to the
//! firmware when `secure=on` has a firmware or the boot EL is at or above the conduit's EL, as
//! hw/arm/virt.c and hw/arm/boot.c decide. The device tree is built with the same libfdt calls
//! in the same order as QEMU, so it matches `-M virt,dumpdtb=` byte for byte. `-kernel` takes
//! an arm64 Image (raw, gzipped or EFI zboot with gzip) or an AArch64 ELF, with `-initrd`,
//! `-append` and `-dtb`, as hw/arm/boot.c loads them, and the ROM list is checked for overlaps
//! and copied into RAM at every reset. With firmware in the first flash, `-kernel`, `-initrd`
//! and `-append` go to the firmware through fw_cfg instead, the kernel inflated if it is gzip.
//!
//! Unless `acpi=off`, `etc/acpi/tables`, `etc/table-loader`, an empty `etc/tpm/log` and
//! `etc/acpi/rsdp` in fw_cfg carry the tables of hw/arm/virt-acpi-build.c (see
//! [`ruvm_firmware::acpi::arm_virt`]): DSDT, FADT, MADT, PPTT, GTDT, MCFG, SPCR, DBG2 and IORT.
//! When firmware boots with ACPI, the ACPI GED at 0x09080000 (SPI 9) carries the power down
//! and error events. Otherwise the PL061 GPIO at 0x09030000 (SPI 7) has the power key
//! (`gpio-keys`) on pin 3. `secure=on` adds the secure PL061 at 0x090b0000, whose pins 0 and 1
//! power the machine off and reset it (`gpio-pwr`).
//!
//! # Using it
//!
//! [`VirtMachine::new`] builds the board and loads the kernel. Plug virtio devices with
//! [`VirtMachine::attach_virtio`] (virtio-mmio) or [`VirtMachine::attach_virtio_pci`] (a PCI
//! function on the root bus), then call [`VirtMachine::machine_done`], which finishes and
//! loads the device tree and resets the board. [`VirtMachine::create_vcpus`] then makes the
//! vCPUs on a [`Jit`], wired to the GIC, and resets them. On a guest reset request
//! ([`VirtMachine::take_request`]) the runner calls [`VirtMachine::system_reset`] and, on each
//! vCPU's thread, [`VirtMachine::reset_cpu`]. The runner also runs the timers of
//! [`VirtMachine::clock`]: the generic timers arm their deadlines there.
//!
//! # Not modelled yet
//!
//! These leave a seam for M6:
//!
//! - The virtualization and security extensions of the GICv2, so `gic-version=2` with
//!   `virtualization=on` or `secure=on` fails. Without `gic-version`, QEMU picks a GICv2 for
//!   TCG when there are at most 8 CPUs; ruvm keeps its GICv3.
//! - CXL, so the empty `cxl_host_reg` container QEMU maps above the redistributors is not
//!   there either.
//! - SMBIOS (`virt_build_smbios()`), so firmware finds no SMBIOS tables in fw_cfg.
//! - The ACPI tables of the missing devices: HEST (`ras=on`), HMAT, the watchdog's GTDT entry
//!   and WDAT, TPM2, VIOT, CEDT, NFIT, the PPTT cache nodes of `smp-cache`, the memory and
//!   ACPI PCI hotplug AML, and SRAT and SLIT, since there is no `-numa`.
//!
//! Also missing: the GED's memory hotplug container at 0x09070000, NUMA, GICv2 and GICv5, the
//! tag memory of the secure RAM with `mte=on` (only the RAM has tags), `-shim`, uImage and
//! u-boot ramdisks, zstd EFI zboot payloads, big-endian and ELF32 kernels, memory hotplug and
//! device memory, and `dtb-randomness` (the board behaves as with `dtb-randomness=off`: no
//! `kaslr-seed` and no `rng-seed`).
//!
//! # Differences from QEMU
//!
//! - There is one address space, so the secure-only devices of `secure=on` (the first flash,
//!   the secure UART, the secure RAM and the secure PL061) are visible to non-secure accesses
//!   too.
//! - Errors come back as `Err` strings without the `qemu-system-aarch64: ` prefix instead of
//!   exiting. Messages QEMU prints and carries on after go to standard error and are kept in
//!   [`VirtMachine::messages`].
//! - The default CPU is cortex-a57; QEMU's is the 32-bit cortex-a15, which is not modelled.
//! - A gzip stream that ends early reports `inflate()` error -5 and corrupt data reports -3,
//!   as zlib would, but other zlib codes are not reproduced.
//! - The generic timer deadlines are host [`Instant`]s converted onto the board clock.
//! - Cache sizes in the CPU nodes come from the legacy CCSIDR layout of each model, which is
//!   what the four models use.

pub(crate) mod boot;
pub(crate) mod cpus;
mod dt;
pub mod kvm;
pub mod memmap;

use std::fmt;
use std::sync::{Arc, Mutex, OnceLock, PoisonError, RwLock, Weak};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use ruvm_base::ClockType;
use ruvm_firmware::acpi::BuildTables;
use ruvm_firmware::acpi::arm_virt::{
    self, ArmVirtAcpi, GicV2Bases, GicV2mFrame, IortSmmu, PsciConduit as AcpiPsci, VirtIrqs,
    VirtMemmap as AcpiMemmap,
};
use ruvm_firmware::acpi::gpex::Window;
use ruvm_firmware::acpi::q35::{PciDevice as AcpiPciDevice, PciDeviceAml};
use ruvm_firmware::acpi::table::{
    APPNAME6, APPNAME8, LOADER_FILE, RSDP_FILE, TABLE_FILE, TPMLOG_FILE,
};
use ruvm_hw_acpi::ged::{ACPI_GED_ERROR_EVT, ACPI_GED_EVT_SEL_LEN, ACPI_GED_PWR_DOWN_EVT};
use ruvm_hw_acpi::{AcpiGed, AcpiGedProps};
use ruvm_hw_char::pl011::{PL011_MMIO_SIZE, Pl011};
use ruvm_hw_char::serial::SerialBackend;
use ruvm_hw_core::fw_cfg::{
    DmaMemory, FW_CFG_CTL_SIZE, FW_CFG_DMA_SIZE, FW_CFG_NB_CPUS, FwCfgMachineConfig, FwCfgMem,
    fw_cfg_init_mem_dma,
};
use ruvm_hw_core::timer::TimeSource;
use ruvm_hw_core::{Clock, IrqLine};
use ruvm_hw_intc::gicv2::{GIC_NCPU, GICV2_CPU_SIZE, GICV2_DIST_SIZE, GicV2, GicV2Props};
use ruvm_hw_intc::gicv2m::{GICV2M_SIZE, GicV2m};
use ruvm_hw_intc::gicv3::{
    GICV3_DIST_SIZE, GICV3_REDIST_SIZE, GicV3, GicV3Its, GicV3Props, ITS_CONTROL_SIZE, ITS_SIZE,
    ITS_TRANS_SIZE,
};
use ruvm_hw_iommu::{SMMU_SIZE, SmmuStage, SmmuV3};
use ruvm_hw_misc::pl061::PL061_MMIO_SIZE;
use ruvm_hw_misc::{GpioKey, Pl061, Pl061Props};
use ruvm_hw_pci::regs::PCI_NUM_PINS;
use ruvm_hw_pci::{GpexConfig, GpexHost, GpexWindow, MsiTrigger};
use ruvm_hw_timer::pl031::{PL031_MMIO_SIZE, Pl031};
use ruvm_hw_virtio::mmio::{VIRTIO_MMIO_FORCE_LEGACY_DEFAULT, VIRTIO_MMIO_REGION_SIZE};
use ruvm_hw_virtio::virtio::VIRTIO_F_IOMMU_PLATFORM;
use ruvm_hw_virtio::{VirtioBackend, VirtioDeviceClass, VirtioMmio, VirtioPci, VirtioPciProps};
use ruvm_jit::{Cpu, CpuShared, Jit, Vcpu};
use ruvm_mem::{
    AccessConstraints, AccessCtx, AccessSize, AddressSpace, Endian, MemResult, MemTxAttrs,
    MemorySystem, MmioOps, RegionId, RegionType,
};
use ruvm_target_arm::cpu::{ArmCpuModel, CpuArmState};
use ruvm_target_arm::tcg::{Arm, PsciConduit, SemihostingHost, TagMemory, create_vcpu};
use ruvm_virtio_queue::{GuestMemory, MemoryError};

pub use boot::{BootInfo, RamRange, Rom};
pub use memmap::{Highmem, MemMapEntry, VirtMemmap};

use crate::fdt::Fdt;
use crate::pflash::{Pflash, PflashBacking, PflashProps};
use boot::{BootFiles, Loader};
use cpus::{CpuHub, GicCpuIf, VirtSemihost};

/// `VIRT_FLASH`, the two flashes.
pub const VIRT_FLASH: u64 = 0;
/// The size of the `VIRT_FLASH` window, half of it for each flash.
pub const VIRT_FLASH_SIZE: u64 = 0x0800_0000;
/// `VIRT_GIC_DIST`.
pub const VIRT_GIC_DIST: u64 = 0x0800_0000;
/// `VIRT_GIC_REDIST`.
pub const VIRT_GIC_REDIST: u64 = 0x080a_0000;
/// The size of the `VIRT_GIC_REDIST` window.
pub const VIRT_GIC_REDIST_SIZE: u64 = 0x00f6_0000;
/// The most CPUs the low GICv3 redistributor window has room for, `virt_redist_capacity()` of
/// `VIRT_GIC_REDIST`. The high memory region takes the CPUs past these.
pub const VIRT_GICV3_MAX_CPUS: usize = (VIRT_GIC_REDIST_SIZE / GICV3_REDIST_SIZE) as usize;
/// `VIRT_GIC_ITS`.
pub const VIRT_GIC_ITS: u64 = 0x0808_0000;
/// `VIRT_GIC_CPU`, the GICv2 CPU interface.
pub const VIRT_GIC_CPU: u64 = 0x0801_0000;
/// `VIRT_GIC_HYP`, the GICv2 virtual interface control, which ruvm does not model yet.
pub const VIRT_GIC_HYP: u64 = 0x0803_0000;
/// `VIRT_GIC_VCPU`, the GICv2 virtual CPU interface, which ruvm does not model yet.
pub const VIRT_GIC_VCPU: u64 = 0x0804_0000;
/// `VIRT_GIC_V2M`, the GICv2m MSI frame.
pub const VIRT_GIC_V2M: u64 = 0x0802_0000;
/// The first SPI of the GICv2m frame, `irqmap[VIRT_GIC_V2M]`.
pub const VIRT_GIC_V2M_IRQ: u32 = 48;
/// `NUM_GICV2M_SPIS`.
pub const NUM_GICV2M_SPIS: u32 = 64;
/// `VIRT_UART0`.
pub const VIRT_UART: u64 = 0x0900_0000;
/// The size of the UART window.
pub const VIRT_UART_SIZE: u64 = 0x1000;
/// `VIRT_UART1`, the secure UART with `secure=on`.
pub const VIRT_UART1: u64 = 0x0904_0000;
/// `VIRT_RTC`.
pub const VIRT_RTC: u64 = 0x0901_0000;
/// The size of the RTC window.
pub const VIRT_RTC_SIZE: u64 = 0x1000;
/// `VIRT_GPIO`, the PL061 with the power key.
pub const VIRT_GPIO: u64 = 0x0903_0000;
/// `VIRT_SECURE_GPIO`, the secure PL061 of `secure=on`.
pub const VIRT_SECURE_GPIO: u64 = 0x090b_0000;
/// The size of the GPIO window.
pub const VIRT_GPIO_SIZE: u64 = 0x1000;
/// `VIRT_ACPI_GED`.
pub const VIRT_ACPI_GED: u64 = 0x0908_0000;
/// `VIRT_SMMU`, where the SMMUv3 of `iommu=smmuv3` sits.
pub const VIRT_SMMU: u64 = 0x0905_0000;
/// `VIRT_FW_CFG`.
pub const VIRT_FW_CFG: u64 = 0x0902_0000;
/// The size of the fw_cfg window: data, control and DMA.
pub const VIRT_FW_CFG_SIZE: u64 = 0x18;
/// `VIRT_MMIO`, the first virtio-mmio transport.
pub const VIRT_MMIO: u64 = 0x0a00_0000;
/// The stride and size of each virtio-mmio transport.
pub const VIRT_MMIO_SIZE: u64 = 0x200;
/// `VIRT_PLATFORM_BUS`.
pub const VIRT_PLATFORM_BUS: u64 = 0x0c00_0000;
/// The size of the platform bus window.
pub const VIRT_PLATFORM_BUS_SIZE: u64 = 0x0200_0000;
/// `VIRT_SECURE_MEM`, the secure RAM of `secure=on`.
pub const VIRT_SECURE_MEM: u64 = 0x0e00_0000;
/// The size of the secure RAM.
pub const VIRT_SECURE_MEM_SIZE: u64 = 0x0100_0000;
/// `VIRT_MEM`, the base of RAM.
pub const VIRT_MEM: u64 = 0x4000_0000;
/// The PCIe MMIO window, `VIRT_PCIE_MMIO`.
pub const VIRT_PCIE_MMIO: u64 = 0x1000_0000;
/// Its size.
pub const VIRT_PCIE_MMIO_SIZE: u64 = 0x2eff_0000;
/// The PCIe I/O port window, `VIRT_PCIE_PIO`.
pub const VIRT_PCIE_PIO: u64 = 0x3eff_0000;
/// Its size.
pub const VIRT_PCIE_PIO_SIZE: u64 = 0x1_0000;
/// The SPI of the UART.
pub const VIRT_UART_IRQ: u32 = 1;
/// The SPI of PCIe INTA, followed by those of INTB, INTC and INTD.
pub const VIRT_PCIE_IRQ: u32 = 3;
/// The SPI of the RTC.
pub const VIRT_RTC_IRQ: u32 = 2;
/// The SPI of the second UART.
pub const VIRT_UART1_IRQ: u32 = 8;
/// The SPI of the first virtio-mmio transport.
pub const VIRT_MMIO_IRQ: u32 = 16;
/// The SPI of the PL061 GPIO.
pub const VIRT_GPIO_IRQ: u32 = 7;
/// The SPI of the secure PL061, 0 since `VIRT_SECURE_GPIO` has no entry in the IRQ map.
pub const VIRT_SECURE_GPIO_IRQ: u32 = 0;
/// The SPI of the ACPI GED.
pub const VIRT_ACPI_GED_IRQ: u32 = 9;
/// The first of the four SPIs of the SMMUv3.
pub const VIRT_SMMU_IRQ: u32 = 74;
/// `ARM_SPI_BASE`, the INTID of SPI 0.
pub const ARM_SPI_BASE: u32 = 32;
/// The first SPI of the platform bus.
pub const VIRT_PLATFORM_BUS_IRQ: u32 = 112;
/// `NUM_VIRTIO_TRANSPORTS`.
pub const VIRTIO_TRANSPORTS: usize = 32;
/// `NUM_IRQS` plus the 32 internal interrupts: the `num-irq` of the GIC.
pub const VIRT_GIC_NUM_IRQ: u32 = 256 + 32;
/// The default RAM size of the machine class.
pub const VIRT_DEFAULT_RAM_SIZE: u64 = 128 << 20;
/// The RAM region name, `default_ram_id`.
pub const VIRT_RAM_ID: &str = "mach-virt.ram";

/// The `msi` machine property, which `its` sets too: the MSI controller of the board.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VirtMsi {
    /// `msi=auto`, the default: the ITS with a GICv3 and the GICv2m with a GICv2.
    #[default]
    Auto,
    /// `msi=its` or `its=on`.
    Its,
    /// `msi=gicv2m`.
    Gicv2m,
    /// `msi=off`.
    Off,
    /// `its=off`, `VIRT_MSI_LEGACY_OPT_ITS_OFF`: the GICv2m with a GICv2, and no MSI
    /// controller with a GICv3.
    ItsOff,
}

/// The `gic-version` machine property, once `finalize_gic_version()` has made it a number.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VirtGicVersion {
    /// `gic-version=2`.
    V2,
    /// `gic-version=3`. QEMU picks a GICv2 when the property is not given and there are at
    /// most 8 CPUs; ruvm keeps the GICv3 it has always had.
    #[default]
    V3,
}

/// The GIC of a virt board.
#[derive(Clone, Debug)]
pub enum VirtGic {
    /// A GICv2, with `gic-version=2`.
    V2(Arc<GicV2>),
    /// A GICv3.
    V3(Arc<GicV3>),
}

impl VirtGic {
    /// SPI `n`, interrupt `n + 32`.
    pub fn spi(&self, n: u32) -> IrqLine {
        match self {
            VirtGic::V2(g) => g.spi(n),
            VirtGic::V3(g) => g.spi(n),
        }
    }

    /// The PPI with interrupt ID `n` of `cpu`.
    pub fn ppi(&self, cpu: usize, n: u32) -> IrqLine {
        match self {
            VirtGic::V2(g) => g.ppi(cpu, n),
            VirtGic::V3(g) => g.ppi(cpu, n),
        }
    }

    /// The device reset.
    pub fn reset(&self) {
        match self {
            VirtGic::V2(g) => g.reset(),
            VirtGic::V3(g) => g.reset(),
        }
    }
}

/// The `iommu` machine property: the IOMMU in front of the PCIe root bus.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VirtIommu {
    /// `iommu=none`, the default.
    #[default]
    None,
    /// `iommu=smmuv3`.
    SmmuV3,
}

/// The `-smp` topology, `ms->smp`, which the possible CPUs are numbered by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuTopology {
    pub sockets: u32,
    pub clusters: u32,
    pub cores: u32,
    pub threads: u32,
    /// `smp_props.has_clusters`: `-smp` named the clusters.
    pub has_clusters: bool,
}

impl CpuTopology {
    /// The topology `-smp N,maxcpus=M` gives: one socket of `max_cpus` cores.
    pub fn flat(max_cpus: u32) -> CpuTopology {
        CpuTopology { sockets: 1, clusters: 1, cores: max_cpus, threads: 1, has_clusters: false }
    }

    /// The number of possible CPUs.
    pub fn max_cpus(&self) -> u32 {
        self.sockets * self.clusters * self.cores * self.threads
    }
}

/// What the board is built from: the `-cpu`, `-smp`, `-m`, `-kernel`, `-initrd`, `-append`,
/// `-dtb`, `-bios`, `-serial` and `-semihosting` options and the `secure`, `virtualization`,
/// `msi`, `iommu`, `default-bus-bypass-iommu`, `acpi`, `spcr`, `x-oem-id` and `x-oem-table-id` machine properties.
#[derive(Clone)]
pub struct VirtConfig {
    /// The CPU model.
    pub cpu: ArmCpuModel,
    /// The number of CPUs.
    pub smp: usize,
    /// `maxcpus` of `-smp`, which the redistributor space must have room for. `None` means
    /// `smp`.
    pub max_cpus: Option<usize>,
    /// The RAM size in bytes.
    pub ram_size: u64,
    /// `-kernel`.
    pub kernel: Option<String>,
    /// `-initrd`.
    pub initrd: Option<String>,
    /// `-append`.
    pub append: Option<String>,
    /// `-dtb`.
    pub dtb: Option<String>,
    /// `virtualization=on`: the CPUs keep EL2 and PSCI goes through SMC.
    pub virtualization: bool,
    /// `secure=on`: the CPUs keep EL3, and the secure UART, RAM and flash appear.
    pub secure: bool,
    /// `mte=on`: the RAM gets tag memory, which makes the CPU's FEAT_MTE a full one.
    pub mte: bool,
    /// The `highmem*` properties, which place the regions above RAM.
    pub highmem: Highmem,
    /// The `msi` property.
    pub msi: VirtMsi,
    /// The `gic-version` property.
    pub gic_version: VirtGicVersion,
    /// The `iommu` property.
    pub iommu: VirtIommu,
    /// `default-bus-bypass-iommu`: the root bus is not behind the IOMMU.
    pub default_bus_bypass_iommu: bool,
    /// `-bios`: the firmware image, loaded into the first flash.
    pub firmware: Option<String>,
    /// The drives of the two flashes, `pflash0` and `pflash1` (`-drive if=pflash`).
    pub pflash: [PflashBacking; 2],
    /// The chardev of the UART, `serial_hd(0)`.
    pub serial: Option<Arc<dyn SerialBackend>>,
    /// The chardev of the second UART, `serial_hd(1)`. With it, or with `secure=on`, the
    /// second UART exists.
    pub serial1: Option<Arc<dyn SerialBackend>>,
    /// Semihosting, when enabled.
    pub semihosting: Option<Arc<dyn SemihostingHost>>,
    /// `-semihosting-config userspace=on`.
    pub semihosting_userspace: bool,
    /// The clock the generic timers run on. The default follows the host's monotonic time.
    pub clock: Option<Arc<Clock>>,
    /// The clock of the RTC, `rtc_clock`. The default is the host wall clock.
    pub rtc_clock: Option<Arc<Clock>>,
    /// The machine options fw_cfg exposes.
    pub fw_cfg: FwCfgMachineConfig,
    /// `acpi`: off leaves out the ACPI tables and the GED. `auto`, the default, is the same as
    /// `on`.
    pub acpi: bool,
    /// `spcr`: off leaves the SPCR out of the ACPI tables.
    pub spcr: bool,
    /// `x-oem-id`, at most 6 bytes.
    pub oem_id: String,
    /// `x-oem-table-id`, at most 8 bytes.
    pub oem_table_id: String,
    /// The `-smp` topology. `None` is one socket with a core for each possible CPU.
    pub topology: Option<CpuTopology>,
    /// Where the SPIs go instead of the board's GIC, for a GIC that KVM keeps in the kernel.
    pub spi_sink: Option<SpiSink>,
}

/// Takes SPI `n` and its new level, for [`VirtConfig::spi_sink`].
pub type SpiSink = Arc<dyn Fn(u32, bool) + Send + Sync>;

/// The line of SPI `n`: the sink's when there is one, the GIC's otherwise.
fn spi_line(gic: &VirtGic, sink: Option<&SpiSink>, n: u32) -> IrqLine {
    match sink {
        Some(sink) => {
            let sink = Arc::clone(sink);
            IrqLine::from_fn(move |level| sink(n, level != 0))
        }
        None => gic.spi(n),
    }
}

impl fmt::Debug for VirtConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VirtConfig")
            .field("cpu", &self.cpu.name)
            .field("smp", &self.smp)
            .field("max_cpus", &self.max_cpus)
            .field("ram_size", &self.ram_size)
            .field("kernel", &self.kernel)
            .field("initrd", &self.initrd)
            .field("append", &self.append)
            .field("dtb", &self.dtb)
            .field("virtualization", &self.virtualization)
            .field("secure", &self.secure)
            .field("mte", &self.mte)
            .field("highmem", &self.highmem)
            .field("msi", &self.msi)
            .field("iommu", &self.iommu)
            .field("default_bus_bypass_iommu", &self.default_bus_bypass_iommu)
            .field("firmware", &self.firmware)
            .field("pflash", &self.pflash)
            .field("serial", &self.serial.is_some())
            .field("serial1", &self.serial1.is_some())
            .field("semihosting", &self.semihosting.is_some())
            .field("semihosting_userspace", &self.semihosting_userspace)
            .field("acpi", &self.acpi)
            .field("spcr", &self.spcr)
            .field("oem_id", &self.oem_id)
            .field("oem_table_id", &self.oem_table_id)
            .field("topology", &self.topology)
            .finish_non_exhaustive()
    }
}

impl VirtConfig {
    /// One CPU of model `cpu`, the default RAM size and nothing to load.
    pub fn new(cpu: ArmCpuModel) -> VirtConfig {
        VirtConfig {
            cpu,
            smp: 1,
            max_cpus: None,
            ram_size: VIRT_DEFAULT_RAM_SIZE,
            kernel: None,
            initrd: None,
            append: None,
            dtb: None,
            virtualization: false,
            secure: false,
            mte: false,
            highmem: Highmem::default(),
            msi: VirtMsi::Auto,
            gic_version: VirtGicVersion::V3,
            iommu: VirtIommu::None,
            default_bus_bypass_iommu: false,
            firmware: None,
            pflash: [PflashBacking::None, PflashBacking::None],
            serial: None,
            serial1: None,
            semihosting: None,
            semihosting_userspace: false,
            clock: None,
            rtc_clock: None,
            fw_cfg: FwCfgMachineConfig::default(),
            acpi: true,
            spcr: true,
            oem_id: APPNAME6.to_string(),
            oem_table_id: APPNAME8.to_string(),
            topology: None,
            spi_sink: None,
        }
    }
}

impl Default for VirtConfig {
    fn default() -> VirtConfig {
        VirtConfig::new(ArmCpuModel::by_name("cortex-a57").expect("cortex-a57 exists"))
    }
}

/// What the guest asked the machine to do through PSCI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VirtRequest {
    /// SYSTEM_OFF.
    Shutdown,
    /// SYSTEM_RESET.
    Reset,
}

/// Receives the [`VirtRequest`]s on the thread of the vCPU that made them.
pub type VirtRequestHandler = Arc<dyn Fn(VirtRequest) + Send + Sync>;

pub(crate) fn err<E: fmt::Display>(e: E) -> String {
    e.to_string()
}

/// fw_cfg DMA through a weak reference, so the address space that maps fw_cfg does not keep
/// itself alive.
struct WeakDma(Weak<AddressSpace>);

impl DmaMemory for WeakDma {
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.0.upgrade().is_some_and(|a| a.read(addr, MemTxAttrs::UNSPECIFIED, buf).is_ok())
    }

    fn write(&self, addr: u64, buf: &[u8]) -> bool {
        self.0.upgrade().is_some_and(|a| a.write(addr, MemTxAttrs::UNSPECIFIED, buf).is_ok())
    }
}

/// Guest memory for virtio devices, weak for the same reason as [`WeakDma`].
pub(crate) struct WeakGuestMemory(pub(crate) Weak<AddressSpace>);

impl GuestMemory for WeakGuestMemory {
    fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemoryError> {
        match self.0.upgrade() {
            Some(a) if a.read(addr, MemTxAttrs::UNSPECIFIED, buf).is_ok() => Ok(()),
            _ => Err(MemoryError::OutOfRange { addr, len: buf.len() as u64 }),
        }
    }

    fn write(&self, addr: u64, buf: &[u8]) -> Result<(), MemoryError> {
        match self.0.upgrade() {
            Some(a) if a.write(addr, MemTxAttrs::UNSPECIFIED, buf).is_ok() => Ok(()),
            _ => Err(MemoryError::OutOfRange { addr, len: buf.len() as u64 }),
        }
    }
}

/// The DMA of a PCI function behind the SMMU with `iommu_platform` on: its own address space,
/// which only exists once the function has its devfn, through a weak reference.
pub(crate) struct LateGuestMemory(pub(crate) Arc<OnceLock<Weak<AddressSpace>>>);

impl LateGuestMemory {
    fn space(&self) -> Option<Arc<AddressSpace>> {
        self.0.get().and_then(Weak::upgrade)
    }
}

impl GuestMemory for LateGuestMemory {
    fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemoryError> {
        match self.space() {
            Some(a) if a.read(addr, MemTxAttrs::UNSPECIFIED, buf).is_ok() => Ok(()),
            _ => Err(MemoryError::OutOfRange { addr, len: buf.len() as u64 }),
        }
    }

    fn write(&self, addr: u64, buf: &[u8]) -> Result<(), MemoryError> {
        match self.space() {
            Some(a) if a.write(addr, MemTxAttrs::UNSPECIFIED, buf).is_ok() => Ok(()),
            _ => Err(MemoryError::OutOfRange { addr, len: buf.len() as u64 }),
        }
    }
}

/// One virtio-mmio transport. The device behind a `VirtioMmio` is fixed when it is created,
/// so the region forwards to whichever transport is current and plugging a device swaps it.
struct VirtioSlot {
    gsi: IrqLine,
    transport: RwLock<Arc<VirtioMmio>>,
    plugged: RwLock<bool>,
}

impl VirtioSlot {
    fn current(&self) -> Arc<VirtioMmio> {
        Arc::clone(&self.transport.read().unwrap_or_else(PoisonError::into_inner))
    }

    fn is_plugged(&self) -> bool {
        *self.plugged.read().unwrap_or_else(PoisonError::into_inner)
    }
}

impl MmioOps for VirtioSlot {
    fn read(&self, cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        self.current().read(cx, offset, size)
    }

    fn write(&self, cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        self.current().write(cx, offset, size, value)
    }

    fn valid(&self) -> AccessConstraints {
        self.current().valid()
    }

    fn impl_constraints(&self) -> AccessConstraints {
        self.current().impl_constraints()
    }

    fn endianness(&self) -> Endian {
        self.current().endianness()
    }
}

/// Where a board puts the generic PCIe host bridge.
pub(crate) struct PcieLayout {
    pub(crate) ecam: MemMapEntry,
    pub(crate) mmio: MemMapEntry,
    pub(crate) high_mmio: Option<MemMapEntry>,
    pub(crate) pio: MemMapEntry,
    /// The SPI of INTA, followed by those of INTB, INTC and INTD.
    pub(crate) irq: u32,
}

/// The PCIe part of `create_pcie()`: the generic host bridge with the ECAM of `layout`, the
/// low MMIO window mapped 1:1, the high one when there is one, the I/O port window, and its
/// INTx lines on four SPIs from `layout.irq`.
pub(crate) fn create_pcie(
    mem: &Arc<MemorySystem>,
    system: RegionId,
    layout: &PcieLayout,
    spi: &dyn Fn(u32) -> IrqLine,
    memory_as: &Arc<AddressSpace>,
) -> Result<GpexHost, String> {
    let w = |e: MemMapEntry| GpexWindow { base: e.base, size: e.size };
    let ecam = layout.ecam;
    let mmio = layout.mmio;
    let config = GpexConfig {
        ecam: w(ecam),
        mmio32: w(mmio),
        mmio64: layout.high_mmio.map_or(GpexWindow::default(), w),
        pio: w(layout.pio),
        ..GpexConfig::default()
    };
    let gpex = GpexHost::new(Arc::clone(mem), system, config).map_err(err)?;
    // `mc->pci_allow_0_address`: EDK2 puts the first I/O BAR at port 0, so a BAR at address 0
    // has to be mapped. It only affects functions plugged after this.
    gpex.bus().set_allow_0_address(true);
    // Map only the first size_ecam bytes of ECAM space, and the MMIO windows at the same
    // address in PCI memory space as in the system's.
    let mut aliases = vec![
        ("pcie-ecam", gpex.ecam(), 0, ecam.base, ecam.size),
        ("pcie-mmio", gpex.mmio_window(), mmio.base, mmio.base, mmio.size),
    ];
    if let Some(h) = layout.high_mmio {
        aliases.push(("pcie-mmio-high", gpex.mmio_window(), h.base, h.base, h.size));
    }
    for (name, target, offset, addr, size) in aliases {
        let alias = mem.new_alias(name, target, offset, size.into()).map_err(err)?;
        mem.add_subregion(system, addr, alias).map_err(err)?;
    }
    mem.add_subregion(system, layout.pio.base, gpex.ioport_window()).map_err(err)?;
    for i in 0..PCI_NUM_PINS {
        let irq = layout.irq + i as u32;
        if let Some(pin) = gpex.irq(i) {
            pin.connect(spi(irq));
        }
        gpex.set_irq_num(i, irq as i32).map_err(err)?;
    }
    // msi_nonbroken is a global in QEMU, and in a qemu-system-aarch64 with the Aspeed boards
    // built in, the class init of aspeed-pcie-rc sets it before the machine is even chosen. So
    // functions on this bus get their MSI-X capability with or without an MSI controller. A
    // message is a plain 32 bit store into system memory, msi_send_message(), which goes nowhere
    // unless something is mapped at its address.
    let msi: MsiTrigger = Arc::new(msi_store(memory_as));
    gpex.bus().set_msi_handler(Some(msi));
    Ok(gpex)
}

/// A message signaled interrupt as `address_space_stl_le()` into `memory_as` sends it, through a
/// weak reference so that the bus does not keep the address space alive.
pub(crate) fn msi_store(
    memory_as: &Arc<AddressSpace>,
) -> impl Fn(u64, u32) + Send + Sync + 'static {
    let weak = Arc::downgrade(memory_as);
    move |address, data| {
        if let Some(a) = weak.upgrade() {
            let _ = a.store(address, 4, data.into(), Endian::Little, MemTxAttrs::UNSPECIFIED);
        }
    }
}

/// What plugging a PCI function needs from the board: the PCIe host, and the SMMU in front of
/// its root bus with the address spaces it gives the functions.
pub(crate) struct PciPlug<'a> {
    pub(crate) mem: &'a Arc<MemorySystem>,
    pub(crate) memory_as: &'a Arc<AddressSpace>,
    pub(crate) gpex: &'a GpexHost,
    /// The SMMU, unless there is none or the root bus bypasses it.
    pub(crate) smmu: Option<&'a Arc<SmmuV3>>,
    pub(crate) iommu_spaces: &'a Mutex<Vec<Arc<AddressSpace>>>,
}

impl PciPlug<'_> {
    /// Plug a virtio device into a new function on the root bus, at `devfn` or the first free
    /// slot. As in `virtio_bus_device_plugged()`, only a device with `iommu_platform` on does
    /// its DMA through the SMMU; the others use system memory, and only their MSIs go through
    /// the SMMU.
    pub(crate) fn virtio_pci(
        &self,
        class: Box<dyn VirtioDeviceClass>,
        devfn: Option<u8>,
        props: &VirtioPciProps,
        iommu_platform: bool,
    ) -> Result<VirtioPci, String> {
        // pci_device_iommu_address_space(): the SMMU unless the root bus bypasses it.
        let smmu = self.smmu;
        let late = Arc::new(OnceLock::new());
        let memory: Arc<dyn GuestMemory + Send + Sync> = match smmu {
            Some(_) if iommu_platform => Arc::new(LateGuestMemory(Arc::clone(&late))),
            _ => Arc::new(WeakGuestMemory(Arc::downgrade(self.memory_as))),
        };
        let mut backend = VirtioBackend::new(class, memory).map_err(err)?;
        if iommu_platform {
            backend.vdev_mut().set_host_feature(VIRTIO_F_IOMMU_PLATFORM, true);
        }
        let dev = VirtioPci::new(self.gpex.bus(), devfn, backend, props).map_err(err)?;
        let mut memory_as = Arc::downgrade(self.memory_as);
        if let Some(smmu) = smmu {
            // smmu_find_add_as(): an IOMMU region and an address space of the same name for
            // the function, whose stream ID is its requester ID on bus 0.
            let devfn = dev.pci_dev().devfn();
            let mut spaces = self.iommu_spaces.lock().unwrap_or_else(PoisonError::into_inner);
            let name = format!("smmuv3-iommu-memory-region-{devfn}-{}", spaces.len());
            let ops = smmu.device_ops(u32::from(devfn));
            let r = self.mem.new_iommu(&name, 1 << 64, ops).map_err(err)?;
            let space = self.mem.address_space_init(r, &name).map_err(err)?;
            memory_as = Arc::downgrade(&space);
            let _ = late.set(Arc::downgrade(&space));
            spaces.push(space);
        }
        // msi_send_message() stores with the function's requester ID, which is the device ID
        // the ITS translates, into the function's address space.
        let pci_dev = Arc::downgrade(dev.pci_dev());
        dev.pci_dev().set_msi_trigger(Some(Arc::new(move |address, data| {
            let (Some(d), Some(a)) = (pci_dev.upgrade(), memory_as.upgrade()) else { return };
            let attrs = MemTxAttrs::new().with_requester_id(d.requester_id());
            let _ = a.store(address, 4, data.into(), Endian::Little, attrs);
        })));
        Ok(dev)
    }
}

/// The RAM ranges the system address space renders to, for the ROM copies and semihosting.
pub(crate) fn ram_ranges(mem: &MemorySystem, system: RegionId) -> Result<Vec<RamRange>, String> {
    let view = mem.render(system).map_err(err)?;
    Ok(view
        .ranges()
        .iter()
        .filter_map(|r| {
            let block = r.ram_block()?.clone();
            Some(RamRange {
                addr: r.addr(),
                size: u64::try_from(r.size()).unwrap_or(u64::MAX),
                readonly: r.readonly(),
                block,
                offset: r.offset_in_region(),
                rom_device: r.region_type() == RegionType::RomDevice,
            })
        })
        .collect())
}

/// Connect the four GIC outputs of CPU `i` to the vCPU `shared`, and its maintenance interrupt
/// to its PPI 25, as QEMU does for a GICv3 on both virt and sbsa-ref.
pub(crate) fn wire_cpu(gic: &Arc<GicV3>, arm: &Arc<Arm>, i: usize, shared: &Arc<CpuShared>) {
    type SetLine = fn(&Arm, &CpuShared, bool);
    let lines: [(&ruvm_hw_core::IrqPin, SetLine); 4] = [
        (gic.cpu_irq(i), Arm::set_irq),
        (gic.cpu_fiq(i), Arm::set_fiq),
        (gic.cpu_virq(i), Arm::set_virq),
        (gic.cpu_vfiq(i), Arm::set_vfiq),
    ];
    for (pin, set) in lines {
        let arm = Arc::downgrade(arm);
        let cpu = Arc::downgrade(shared);
        pin.connect(IrqLine::from_fn(move |level| {
            if let (Some(a), Some(c)) = (arm.upgrade(), cpu.upgrade()) {
                set(&a, &c, level != 0);
            }
        }));
    }
    // The maintenance interrupt of the virtual CPU interface goes to its own PPI.
    gic.maintenance_irq(i).connect(gic.ppi(i, 16 + dt::ARCH_GIC_MAINT_IRQ));
}

/// Connect the IRQ and FIQ outputs of a GICv2 for CPU `i` to the vCPU `shared`. Without the
/// virtualization extensions the GICv2 has no virtual outputs and no maintenance interrupt.
fn wire_cpu_v2(gic: &Arc<GicV2>, arm: &Arc<Arm>, i: usize, shared: &Arc<CpuShared>) {
    type SetLine = fn(&Arm, &CpuShared, bool);
    let lines: [(&ruvm_hw_core::IrqPin, SetLine); 2] =
        [(gic.cpu_irq(i), Arm::set_irq), (gic.cpu_fiq(i), Arm::set_fiq)];
    for (pin, set) in lines {
        let arm = Arc::downgrade(arm);
        let cpu = Arc::downgrade(shared);
        pin.connect(IrqLine::from_fn(move |level| {
            if let (Some(a), Some(c)) = (arm.upgrade(), cpu.upgrade()) {
                set(&a, &c, level != 0);
            }
        }));
    }
}

/// `virt_cpu_mp_affinity()` for a GICv3: 16 CPUs per Aff1 cluster.
pub fn virt_cpu_mp_affinity(idx: usize) -> u64 {
    let idx = idx as u64;
    ((idx / 16) << 8) | (idx % 16)
}

/// The virt board.
pub struct VirtMachine {
    model: ArmCpuModel,
    smp: usize,
    ram_size: u64,
    dtb_filename: Option<String>,
    cmdline: String,
    mem: Arc<MemorySystem>,
    system: RegionId,
    memory_as: Arc<AddressSpace>,
    arm: Arc<Arm>,
    hub: Arc<CpuHub>,
    gic: VirtGic,
    its: Option<Arc<GicV3Its>>,
    v2m: bool,
    uart: Arc<Pl011>,
    uart1: Option<Arc<Pl011>>,
    flash: [Arc<Pflash>; 2],
    /// Whether the secondaries start powered off, from the conduit the board picked before
    /// `arm_load_kernel()` adjusted it.
    secondaries_off: bool,
    /// The PSCI conduit before `arm_load_kernel()` adjusted it, `vms->psci_conduit`.
    vms_conduit: PsciConduit,
    acpi: bool,
    spcr: bool,
    oem_id: String,
    oem_table_id: String,
    topology: CpuTopology,
    ns_el2_virt_timer_irq: bool,
    virtualization: bool,
    redist2: Option<MemMapEntry>,
    /// Whether UART1 is the non-secure second UART.
    uart1_ns: bool,
    ged: Option<Arc<AcpiGed>>,
    /// The PL061 and its power key, there when the GED is not.
    gpio: Option<(Arc<Pl061>, Arc<GpioKey>)>,
    /// The secure PL061 of `secure=on`.
    secure_gpio: Option<Arc<Pl061>>,
    rtc: Arc<Pl031>,
    memmap: VirtMemmap,
    gpex: GpexHost,
    /// The SMMUv3 of `iommu=smmuv3`.
    smmu: Option<Arc<SmmuV3>>,
    /// `default-bus-bypass-iommu`: the root bus is not behind the SMMU.
    iommu_bypass: bool,
    /// The address spaces of the functions behind the SMMU, `SMMUDevice.as`.
    iommu_spaces: Mutex<Vec<Arc<AddressSpace>>>,
    pci_devices: Mutex<Vec<VirtioPci>>,
    virtio: Vec<Arc<VirtioSlot>>,
    fw_cfg: FwCfgMem,
    fdt: Fdt,
    loader: Loader,
    info: BootInfo,
    heap: Arc<Mutex<(u64, u64)>>,
    ram: Vec<RamRange>,
    clock: Arc<Clock>,
    done: bool,
}

impl fmt::Debug for VirtMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VirtMachine")
            .field("cpu", &self.model.name)
            .field("smp", &self.smp)
            .field("ram_size", &self.ram_size)
            .field("boot", &self.info)
            .field("roms", &self.loader.roms)
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}

impl VirtMachine {
    /// `machvirt_init()`: build the board and load the kernel.
    pub fn new(cfg: VirtConfig) -> Result<VirtMachine, String> {
        let mut model = cfg.cpu;
        // virt clears has_el3 and has_el2 when secure and virtualization are off.
        model.features.el3 = false;
        model.features.el2 = false;
        model.id_aa64pfr0 &= !0xff00;
        if cfg.secure {
            model = model.with_el3();
        } else if model.features.rme {
            // arm_cpu_realizefn() clears ID_AA64PFR0.RME along with EL3.
            model = model.with_rme(false);
        }
        if cfg.virtualization {
            model = model.with_el2();
        }
        if cfg.mte && model.features.mte == 0 {
            // The "tag-memory" property only exists if MemTag is supported.
            return Err("MTE requested, but not supported by the guest CPU".to_string());
        }
        let smp = cfg.smp;
        let ram_size = cfg.ram_size;

        let memmap = memmap::virt_set_memmap(VIRT_MEM, ram_size, model.pamax(), &cfg.highmem)?;
        // finalize_msi_controller(): auto is the ITS with a GICv3 and the GICv2m with a GICv2.
        let v2 = cfg.gic_version == VirtGicVersion::V2;
        let msi = match cfg.msi {
            VirtMsi::ItsOff if v2 => VirtMsi::Gicv2m,
            VirtMsi::ItsOff => VirtMsi::Off,
            VirtMsi::Auto if v2 => VirtMsi::Gicv2m,
            VirtMsi::Auto => VirtMsi::Its,
            m => m,
        };
        if msi == VirtMsi::Its && v2 {
            return Err("GICv2 + ITS is an invalid configuration.".to_string());
        }
        let its_on = msi == VirtMsi::Its;
        let v2m_on = msi == VirtMsi::Gicv2m;

        let mem = Arc::new(MemorySystem::new());
        let system = mem.new_container("system", 1 << 64).map_err(err)?;
        let memory_as = mem.address_space_init(system, "memory").map_err(err)?;

        // virt_flash_create() and virt_firmware_init(). There is one address space, so the
        // first flash, secure only with secure=on, is in the system memory too.
        let [pflash0, pflash1] = cfg.pflash;
        let pflash0_given = pflash0 != PflashBacking::None;
        let half = VIRT_FLASH_SIZE / 2;
        let flash0 = Pflash::new(&mem, "virt.flash0", PflashProps::virt_flash(half), pflash0)?;
        let flash1 = Pflash::new(&mem, "virt.flash1", PflashProps::virt_flash(half), pflash1)?;
        mem.add_subregion(system, VIRT_FLASH, flash0.region()).map_err(err)?;
        mem.add_subregion(system, VIRT_FLASH + half, flash1.region()).map_err(err)?;
        if let Some(bios) = &cfg.firmware {
            if pflash0_given {
                return Err("The contents of the first flash device may be specified with -bios \
                            or with -drive if=pflash... but you cannot use both options at once"
                    .to_string());
            }
            let data =
                std::fs::read(bios).map_err(|_| format!("Could not find ROM image '{bios}'"))?;
            if !flash0.load_image(&data) {
                return Err(format!("Could not load ROM image '{bios}'"));
            }
        }
        let firmware_loaded = pflash0_given || cfg.firmware.is_some();

        // If we have an EL3 boot ROM then the assumption is that it will implement PSCI
        // itself, so disable the internal implementation so it doesn't get in the way. The
        // usual case is that we do use the internal PSCI; if the guest has EL2 then SMC is
        // the conduit, and otherwise HVC.
        let vms_conduit = if cfg.secure && firmware_loaded {
            PsciConduit::Disabled
        } else if cfg.virtualization {
            PsciConduit::Smc
        } else {
            PsciConduit::Hvc
        };

        // The low redistributor region, and the high one when highmem-redists left it in the
        // memory map.
        let redist2_capacity =
            memmap.high_redist2.map_or(0, |r| (r.size / GICV3_REDIST_SIZE) as usize);
        let virt_max_cpus = if v2 { GIC_NCPU } else { VIRT_GICV3_MAX_CPUS + redist2_capacity };
        let max_cpus = cfg.max_cpus.unwrap_or(smp);
        if max_cpus > virt_max_cpus {
            let mut msg = format!(
                "Number of SMP CPUs requested ({max_cpus}) exceeds max CPUs supported by \
                 machine 'mach-virt' ({virt_max_cpus})"
            );
            if !v2 && memmap.high_redist2.is_none() {
                msg.push_str("\nTry 'highmem-redists=on' for more CPUs");
            }
            return Err(msg);
        }
        if smp == 0 {
            return Err(
                "Invalid SMP CPUs 0. The min CPUs supported by machine 'virt' is 1".to_string()
            );
        }
        let topology = cfg.topology.unwrap_or_else(|| CpuTopology::flat(max_cpus as u32));
        if topology.max_cpus() as usize != max_cpus {
            return Err(format!(
                "Invalid CPU topology: product of the hierarchy must match maxcpus: sockets \
                 ({}) * clusters ({}) * cores ({}) * threads ({}) != maxcpus ({max_cpus})",
                topology.sockets, topology.clusters, topology.cores, topology.threads
            ));
        }
        if cfg.oem_id.len() > 6 {
            return Err("User specified oem-id value is bigger than 6 bytes in size".to_string());
        }
        if cfg.oem_table_id.len() > 8 {
            return Err(
                "User specified oem-table-id value is bigger than 8 bytes in size".to_string()
            );
        }

        let mut fdt = Fdt::new();
        let clock_phandle = dt::create_fdt(&mut fdt, cfg.secure)?;

        let mpidrs: Vec<u64> = (0..smp).map(virt_cpu_mp_affinity).collect();
        // ns_el2_virt_timer_present().
        let ns_el2_virt_timer_irq = model.features.el2 && model.features.vh;
        // With a GICv2 the PPI flags carry the mask of the CPUs.
        let ppi_cpus = if v2 { Some(smp) } else { None };
        dt::add_timer_nodes(&mut fdt, ns_el2_virt_timer_irq, ppi_cpus)?;
        let psci = vms_conduit != PsciConduit::Disabled;
        dt::add_cpu_nodes(&mut fdt, &model, &mpidrs, psci, &topology)?;

        let ram = mem.new_ram(VIRT_RAM_ID, ram_size).map_err(err)?;
        mem.add_subregion(system, VIRT_MEM, ram).map_err(err)?;

        dt::virt_flash_fdt(&mut fdt, cfg.secure)?;

        // create_gic(): a GICv2 or a GICv3.
        let (gic, redist2) = if v2 {
            let gic = GicV2::new(GicV2Props {
                num_cpu: smp,
                num_irq: VIRT_GIC_NUM_IRQ,
                revision: 2,
                security_extn: cfg.secure,
                virt_extn: cfg.virtualization,
                n_prio_bits: model.gic_pribits,
            })?;
            gic.set_current_cpu_fn(Some(Arc::new(ruvm_jit::cpu_exec::current_cpu_index)));
            let r = mem.new_io("gic_dist", GICV2_DIST_SIZE.into(), gic.dist_ops()).map_err(err)?;
            mem.add_subregion(system, VIRT_GIC_DIST, r).map_err(err)?;
            let r = mem.new_io("gic_cpu", GICV2_CPU_SIZE.into(), gic.cpu_ops()).map_err(err)?;
            mem.add_subregion(system, VIRT_GIC_CPU, r).map_err(err)?;
            (VirtGic::V2(gic), None)
        } else {
            // create_gic(), with the second redistributor region when the CPUs do not fit in the
            // first.
            let redist0_count = smp.min(VIRT_GICV3_MAX_CPUS);
            let redist2 = memmap.high_redist2.filter(|_| smp > VIRT_GICV3_MAX_CPUS);
            let mut redist_region_count = vec![redist0_count as u32];
            if redist2.is_some() {
                redist_region_count.push((smp - redist0_count).min(redist2_capacity) as u32);
            }
            let gic = GicV3::with_sysmem(
                GicV3Props {
                    num_cpu: smp,
                    num_irq: VIRT_GIC_NUM_IRQ,
                    revision: 3,
                    security_extn: cfg.secure,
                    redist_region_count,
                    mp_affinity: mpidrs.clone(),
                    pribits: model.gic_pribits,
                    // The TCG ITS is on for every current machine version, so the GIC has LPIs
                    // even when msi=off leaves the ITS out.
                    has_lpi: true,
                },
                Some(&memory_as),
            )?;
            let r =
                mem.new_io("gicv3_dist", GICV3_DIST_SIZE.into(), gic.dist_ops()).map_err(err)?;
            mem.add_subregion(system, VIRT_GIC_DIST, r).map_err(err)?;
            let redist_bases = [Some(VIRT_GIC_REDIST), redist2.map(|r| r.base)];
            for (i, base) in redist_bases.into_iter().enumerate() {
                let Some(base) = base else { continue };
                let name = format!("gicv3_redist_region[{i}]");
                let r = mem
                    .new_io(&name, gic.redist_region_size(i).into(), gic.redist_ops(i))
                    .map_err(err)?;
                mem.add_subregion(system, base, r).map_err(err)?;
            }
            (VirtGic::V3(gic), redist2)
        };
        let gic_phandle = dt::add_gic_node(&mut fdt, v2, redist2, cfg.virtualization)?;

        // create_msi_controller(): the ITS, its control frame and then its translation frame
        // in one container.
        let (its, msi_phandle) = if let (true, VirtGic::V3(g)) = (its_on, &gic) {
            let its = GicV3Its::new(g)?;
            let main = mem.new_container("gicv3_its", ITS_SIZE.into()).map_err(err)?;
            let r =
                mem.new_io("control", ITS_CONTROL_SIZE.into(), its.control_ops()).map_err(err)?;
            mem.add_subregion(main, 0, r).map_err(err)?;
            let r = mem
                .new_io("translation", ITS_TRANS_SIZE.into(), its.translation_ops())
                .map_err(err)?;
            mem.add_subregion(main, ITS_CONTROL_SIZE, r).map_err(err)?;
            mem.add_subregion(system, VIRT_GIC_ITS, main).map_err(err)?;
            (Some(its), Some(dt::add_its_node(&mut fdt)?))
        } else if v2m_on {
            let (gic, sink) = (gic.clone(), cfg.spi_sink.clone());
            let v2m = GicV2m::new(VIRT_GIC_V2M_IRQ, NUM_GICV2M_SPIS, move |n| {
                spi_line(&gic, sink.as_ref(), n)
            })?;
            let r = mem.new_io("gicv2m", GICV2M_SIZE.into(), Arc::new(v2m)).map_err(err)?;
            mem.add_subregion(system, VIRT_GIC_V2M, r).map_err(err)?;
            (None, Some(dt::add_v2m_node(&mut fdt)?))
        } else {
            (None, None)
        };
        // The device interrupts, which go to KVM instead when it keeps the GIC.
        let spi = |n: u32| spi_line(&gic, cfg.spi_sink.as_ref(), n);
        if model.features.pmu != 0 {
            dt::add_pmu_node(&mut fdt, ppi_cpus)?;
        }

        // arm_load_kernel(), which decides the PSCI conduit the CPUs are created with.
        let mut loader = Loader::default();
        let mut info = BootInfo { loader_start: VIRT_MEM, ram_size, ..BootInfo::default() };
        let files = BootFiles { kernel: cfg.kernel.as_deref(), initrd: cfg.initrd.as_deref() };
        boot::arm_load_kernel(&mut loader, &mut info, files, firmware_loaded)?;
        // Boot into the highest EL, except that Linux boots in EL2 or EL1. Disable the PSCI
        // conduit if it targets the same or a lower EL than that.
        let f = &model.features;
        let boot_el = if info.is_linux {
            if f.el2 { 2 } else { 1 }
        } else if f.el3 {
            3
        } else if f.el2 {
            2
        } else {
            1
        };
        let conduit = match vms_conduit {
            PsciConduit::Hvc if boot_el >= 2 => PsciConduit::Disabled,
            PsciConduit::Smc if boot_el == 3 => PsciConduit::Disabled,
            c => c,
        };

        let clock = cfg.clock.unwrap_or_else(|| {
            Clock::new(ClockType::Virtual, TimeSource::Monotonic(Instant::now()))
        });
        let hub = Arc::new(CpuHub::new(|cpu, n| gic.ppi(cpu, n), mpidrs.clone(), clock.clone()));
        let heap = Arc::new(Mutex::new((0, 0)));
        let cmdline = cfg.append.clone().unwrap_or_default();
        let mut arm = Arm::new(model.clone()).with_psci(conduit).with_board(hub.clone());
        if let VirtGic::V3(g) = &gic {
            arm = arm.with_gicv3(Arc::new(GicCpuIf(g.clone())));
        }
        if cfg.mte {
            // The tags of the RAM, one byte per two 16-byte granules, as the
            // "mach-virt.tag" RAM is at VIRT_MEM / 32 of the tag address space.
            arm = arm.with_tag_memory(Arc::new(TagMemory::new(VIRT_MEM, ram_size)));
        }
        if let Some(host) = cfg.semihosting {
            let semi = VirtSemihost {
                host,
                heap: heap.clone(),
                cmdline: cpus::semihosting_cmdline(cfg.kernel.as_deref(), cfg.append.as_deref()),
            };
            arm = arm.with_semihosting(Arc::new(semi), cfg.semihosting_userspace);
        }
        let arm = Arc::new(arm);
        hub.set_arm(&arm);
        for (i, &m) in mpidrs.iter().enumerate() {
            arm.set_mpidr(i, m);
        }

        // create_uart(). The second UART is created first when it is non-secure, and after
        // UART0 when it is the secure one, so that UART0 comes first in the tree.
        let mut uart1 = None;
        let make_uart = |fdt: &mut Fdt, which: dt::Uart, chr| -> Result<Arc<Pl011>, String> {
            let (base, irq) = match which {
                dt::Uart::Uart0 => (VIRT_UART, VIRT_UART_IRQ),
                _ => (VIRT_UART1, VIRT_UART1_IRQ),
            };
            let uart = Pl011::new(chr);
            let r = mem.new_io("pl011", PL011_MMIO_SIZE.into(), uart.clone()).map_err(err)?;
            mem.add_subregion(system, base, r).map_err(err)?;
            uart.irq(0).connect(spi(irq));
            dt::create_uart(fdt, clock_phandle, which)?;
            Ok(uart)
        };
        let serial1 = cfg.serial1;
        if !cfg.secure && serial1.is_some() {
            uart1 = Some(make_uart(&mut fdt, dt::Uart::Uart1, serial1.clone())?);
        }
        let uart = make_uart(&mut fdt, dt::Uart::Uart0, cfg.serial)?;
        if cfg.secure {
            uart1 = Some(make_uart(&mut fdt, dt::Uart::SecureUart1, serial1)?);
        }

        if cfg.secure {
            // create_secure_ram(), in the one address space.
            let r = mem.new_ram("virt.secure-ram", VIRT_SECURE_MEM_SIZE).map_err(err)?;
            mem.add_subregion(system, VIRT_SECURE_MEM, r).map_err(err)?;
            dt::create_secure_ram(&mut fdt)?;
        }

        // create_rtc().
        let rtc_clock =
            cfg.rtc_clock.unwrap_or_else(|| Clock::new(ClockType::Host, TimeSource::Wall));
        // qemu_ref_timedate(): the host clock reads the date itself; the others count from
        // the date the machine started.
        let rtc_date =
            if rtc_clock.kind() == ClockType::Host { UNIX_EPOCH } else { SystemTime::now() };
        let rtc = Pl031::new(rtc_clock, rtc_date);
        let r = mem.new_io("pl031", PL031_MMIO_SIZE.into(), rtc.clone()).map_err(err)?;
        mem.add_subregion(system, VIRT_RTC, r).map_err(err)?;
        rtc.irq().connect(spi(VIRT_RTC_IRQ));
        dt::create_rtc(&mut fdt, clock_phandle)?;

        // create_pcie(), with the msi-map to the ITS.
        let layout = PcieLayout {
            ecam: memmap.ecam,
            mmio: MemMapEntry { base: VIRT_PCIE_MMIO, size: VIRT_PCIE_MMIO_SIZE },
            high_mmio: memmap.high_mmio,
            pio: MemMapEntry { base: VIRT_PCIE_PIO, size: VIRT_PCIE_PIO_SIZE },
            irq: VIRT_PCIE_IRQ,
        };
        let gpex = create_pcie(&mem, system, &layout, &spi, &memory_as)?;
        dt::create_pcie(&mut fdt, &memmap, gic_phandle, msi_phandle, VIRT_PCIE_IRQ)?;
        // create_smmu(), with the stage property set to nested as on every virt version that
        // has the SMMU. It reads its tables from system memory, and the functions on the root
        // bus get their address spaces from it as they are plugged.
        let smmu = if cfg.iommu == VirtIommu::SmmuV3 {
            let irqs = std::array::from_fn(|i| spi(VIRT_SMMU_IRQ + i as u32));
            let smmu = SmmuV3::new(SmmuStage::Nested, &memory_as, irqs);
            let r = mem.new_io("smmuv3", SMMU_SIZE.into(), smmu.mmio_ops()).map_err(err)?;
            mem.add_subregion(system, VIRT_SMMU, r).map_err(err)?;
            dt::create_smmu(&mut fdt, !cfg.default_bus_bypass_iommu)?;
            Some(smmu)
        } else {
            None
        };

        // create_acpi_ged(), for firmware that boots with ACPI, or else the PL061 with the
        // power key.
        let acpi = cfg.acpi;
        let ged = if firmware_loaded && acpi {
            let ged = AcpiGed::new(AcpiGedProps {
                ged_event: ACPI_GED_PWR_DOWN_EVT | ACPI_GED_ERROR_EVT,
                pci_hotplug: false,
            })
            .map_err(err)?;
            let r = mem.new_io("acpi-ged", ACPI_GED_EVT_SEL_LEN.into(), ged.evt_ops());
            mem.add_subregion(system, VIRT_ACPI_GED, r.map_err(err)?).map_err(err)?;
            ged.irq().connect(spi(VIRT_ACPI_GED_IRQ));
            Some(ged)
        } else {
            None
        };
        let make_gpio = |fdt: &mut Fdt, base, irq, secure| -> Result<Arc<Pl061>, String> {
            // Pull lines down to 0 if not driven by the PL061.
            let pl061 = Pl061::new(Pl061Props { pullups: 0, pulldowns: 0xff })?;
            let r = mem.new_io("pl061", PL061_MMIO_SIZE.into(), pl061.clone()).map_err(err)?;
            mem.add_subregion(system, base, r).map_err(err)?;
            pl061.irq().connect(spi(irq));
            dt::create_gpio(fdt, clock_phandle, base, irq, secure)?;
            Ok(pl061)
        };
        let gpio = if ged.is_none() {
            let pl061 = make_gpio(&mut fdt, VIRT_GPIO, VIRT_GPIO_IRQ, false)?;
            // create_gpio_keys().
            let key = GpioKey::new(clock.clone());
            key.irq().connect(pl061.gpio_in(dt::GPIO_PIN_POWER_BUTTON));
            Some((pl061, key))
        } else {
            None
        };
        // create_secure_gpio_pwr(): the secure PL061 drives the gpio-pwr device, which asks
        // for a reset or a shutdown when its line goes high. There is one address space, so
        // this PL061 is in the system memory too.
        let secure_gpio = if cfg.secure {
            let pl061 = make_gpio(&mut fdt, VIRT_SECURE_GPIO, VIRT_SECURE_GPIO_IRQ, true)?;
            for (pin, req) in [
                (dt::SECURE_GPIO_RESET, VirtRequest::Reset),
                (dt::SECURE_GPIO_POWEROFF, VirtRequest::Shutdown),
            ] {
                let hub = Arc::downgrade(&hub);
                pl061.out(pin as usize).connect(IrqLine::from_fn(move |level| {
                    if let (true, Some(h)) = (level != 0, hub.upgrade()) {
                        h.request(req);
                    }
                }));
            }
            Some(pl061)
        } else {
            None
        };

        // create_virtio_devices().
        let mut virtio = Vec::with_capacity(VIRTIO_TRANSPORTS);
        for i in 0..VIRTIO_TRANSPORTS {
            let t = VirtioMmio::new(None, VIRTIO_MMIO_FORCE_LEGACY_DEFAULT).map_err(err)?;
            let slot = Arc::new(VirtioSlot {
                gsi: spi(VIRT_MMIO_IRQ + i as u32),
                transport: RwLock::new(Arc::new(t)),
                plugged: RwLock::new(false),
            });
            let r = mem
                .new_io("virtio-mmio", VIRTIO_MMIO_REGION_SIZE.into(), slot.clone())
                .map_err(err)?;
            mem.add_subregion(system, VIRT_MMIO + i as u64 * VIRT_MMIO_SIZE, r).map_err(err)?;
            virtio.push(slot);
        }
        dt::add_virtio_nodes(&mut fdt)?;

        // create_fw_cfg().
        let dma: Arc<dyn DmaMemory> = Arc::new(WeakDma(Arc::downgrade(&memory_as)));
        let fw_cfg = fw_cfg_init_mem_dma(VIRT_FW_CFG, dma, &cfg.fw_cfg).map_err(err)?;
        fw_cfg.state().add_i16(FW_CFG_NB_CPUS, smp as u16);
        let (ctl, data, dma_addr) = fw_cfg.addrs();
        let r = mem.new_io("fwcfg.ctl", FW_CFG_CTL_SIZE.into(), fw_cfg.ctl_ops().clone());
        mem.add_subregion(system, ctl, r.map_err(err)?).map_err(err)?;
        let data_size = fw_cfg.data_ops().region_size();
        let r = mem.new_io("fwcfg.data", data_size.into(), fw_cfg.data_ops().clone());
        mem.add_subregion(system, data, r.map_err(err)?).map_err(err)?;
        if let Some(d) = fw_cfg.dma_ops() {
            let r = mem.new_io("fwcfg.dma", FW_CFG_DMA_SIZE.into(), d.clone()).map_err(err)?;
            mem.add_subregion(system, dma_addr, r).map_err(err)?;
        }
        dt::add_fw_cfg_node(&mut fdt)?;
        if firmware_loaded {
            // The rest of arm_setup_firmware_boot(), now that fw_cfg exists.
            boot::arm_setup_firmware_boot(&mut loader, fw_cfg.state(), files, &cmdline)?;
        }

        // create_platform_bus(): the window, with nothing on it yet.
        let pbus = mem.new_container("platform bus", VIRT_PLATFORM_BUS_SIZE.into()).map_err(err)?;
        mem.add_subregion(system, VIRT_PLATFORM_BUS, pbus).map_err(err)?;

        // A GICv2 without the security extensions has nothing to set up for Linux.
        if let (true, VirtGic::V3(g)) = (info.is_linux, &gic) {
            g.arm_linux_init(false);
        }

        let uart1_ns = !cfg.secure && uart1.is_some();
        Ok(VirtMachine {
            model,
            smp,
            ram_size,
            dtb_filename: cfg.dtb,
            cmdline,
            mem,
            system,
            memory_as,
            arm,
            hub,
            gic,
            its,
            v2m: v2m_on,
            uart,
            uart1,
            flash: [flash0, flash1],
            secondaries_off: psci,
            vms_conduit,
            acpi,
            spcr: cfg.spcr,
            oem_id: cfg.oem_id,
            oem_table_id: cfg.oem_table_id,
            topology,
            ns_el2_virt_timer_irq,
            virtualization: cfg.virtualization,
            redist2,
            uart1_ns,
            ged,
            gpio,
            secure_gpio,
            rtc,
            memmap,
            gpex,
            smmu,
            iommu_bypass: cfg.default_bus_bypass_iommu,
            iommu_spaces: Mutex::new(Vec::new()),
            pci_devices: Mutex::new(Vec::new()),
            virtio,
            fw_cfg,
            fdt,
            loader,
            info,
            heap,
            ram: Vec::new(),
            clock,
            done: false,
        })
    }

    /// Plug a virtio device into the highest free transport, as `-device virtio-*-device`
    /// does: the guest probes the transports from the top, so the first device plugged is
    /// found first. Returns the transport index.
    pub fn attach_virtio(&self, class: Box<dyn VirtioDeviceClass>) -> Result<usize, String> {
        let Some(index) = (0..self.virtio.len()).rev().find(|&i| !self.virtio[i].is_plugged())
        else {
            return Err("No 'virtio-bus' bus found for device".to_string());
        };
        self.attach_virtio_at(index, class, VIRTIO_MMIO_FORCE_LEGACY_DEFAULT)?;
        Ok(index)
    }

    /// Plug a virtio device into transport `index`, `bus=virtio-mmio-bus.<index>`.
    pub fn attach_virtio_at(
        &self,
        index: usize,
        class: Box<dyn VirtioDeviceClass>,
        force_legacy: bool,
    ) -> Result<(), String> {
        if self.done {
            return Err("virtio-mmio devices must be plugged before machine_done".to_string());
        }
        let Some(slot) = self.virtio.get(index) else {
            return Err(format!("Bus 'virtio-mmio-bus.{index}' not found"));
        };
        if slot.is_plugged() {
            return Err(format!("Bus 'virtio-mmio-bus.{index}' does not support hotplugging"));
        }
        let memory: Arc<dyn GuestMemory + Send + Sync> =
            Arc::new(WeakGuestMemory(Arc::downgrade(&self.memory_as)));
        let backend = VirtioBackend::new(class, memory).map_err(err)?;
        let t = VirtioMmio::new(Some(backend), force_legacy).map_err(err)?;
        t.irq().connect(slot.gsi.clone());
        *slot.transport.write().unwrap_or_else(PoisonError::into_inner) = Arc::new(t);
        *slot.plugged.write().unwrap_or_else(PoisonError::into_inner) = true;
        Ok(())
    }

    /// Plug a virtio device into a new function on the PCIe root bus, as
    /// `-device virtio-*-pci` does, at `devfn` or the first free slot. Gives the function.
    pub fn attach_virtio_pci(
        &self,
        class: Box<dyn VirtioDeviceClass>,
        devfn: Option<u8>,
        props: &VirtioPciProps,
    ) -> Result<VirtioPci, String> {
        self.attach_virtio_pci_with(class, devfn, props, false)
    }

    /// [`VirtMachine::attach_virtio_pci`] with the `iommu_platform` property of the virtio
    /// device. As in `virtio_bus_device_plugged()`, only a device with it on does its DMA
    /// through the SMMU; the others use system memory, and only their MSIs go through the SMMU.
    pub fn attach_virtio_pci_with(
        &self,
        class: Box<dyn VirtioDeviceClass>,
        devfn: Option<u8>,
        props: &VirtioPciProps,
        iommu_platform: bool,
    ) -> Result<VirtioPci, String> {
        if self.done {
            return Err("PCI devices must be plugged before machine_done".to_string());
        }
        let plug = PciPlug {
            mem: &self.mem,
            memory_as: &self.memory_as,
            gpex: &self.gpex,
            smmu: self.smmu.as_ref().filter(|_| !self.iommu_bypass),
            iommu_spaces: &self.iommu_spaces,
        };
        let dev = plug.virtio_pci(class, devfn, props, iommu_platform)?;
        self.pci_devices.lock().unwrap_or_else(PoisonError::into_inner).push(dev.clone());
        Ok(dev)
    }

    /// `virt_acpi_build()`: the ACPI tables of the board as it is now.
    pub fn acpi_tables(&self) -> BuildTables {
        let w = |base, size| Window { base, size };
        let e = |m: MemMapEntry| w(m.base, m.size);
        let spi = |irq| irq + ARM_SPI_BASE;
        let topo = &self.topology;
        let pci_devices = self
            .gpex
            .bus()
            .devices()
            .iter()
            .map(|d| AcpiPciDevice { devfn: d.devfn(), acpi_index: None, aml: PciDeviceAml::Plain })
            .collect();
        let acpi = ArmVirtAcpi {
            oem_id: self.oem_id.clone(),
            oem_table_id: self.oem_table_id.clone(),
            memmap: AcpiMemmap {
                uart0: w(VIRT_UART, VIRT_UART_SIZE),
                uart1: self.uart1_ns.then_some(w(VIRT_UART1, VIRT_UART_SIZE)),
                fw_cfg: w(VIRT_FW_CFG, VIRT_FW_CFG_SIZE),
                virtio: w(VIRT_MMIO, VIRT_MMIO_SIZE),
                ecam: e(self.memmap.ecam),
                pcie_mmio: w(VIRT_PCIE_MMIO, VIRT_PCIE_MMIO_SIZE),
                pcie_pio: w(VIRT_PCIE_PIO, VIRT_PCIE_PIO_SIZE),
                pcie_mmio_high: self.memmap.high_mmio.map_or(Window::default(), e),
                gic_dist: VIRT_GIC_DIST,
                gic_redist: w(VIRT_GIC_REDIST, VIRT_GIC_REDIST_SIZE),
                gic_redist2: self.redist2.map(e),
                gic_its: self.its.as_ref().map(|_| VIRT_GIC_ITS),
                gic_v2: matches!(self.gic, VirtGic::V2(_)).then_some(GicV2Bases {
                    cpu: VIRT_GIC_CPU,
                    vcpu: VIRT_GIC_VCPU,
                    hyp: VIRT_GIC_HYP,
                }),
                gic_v2m: self.v2m.then_some(GicV2mFrame {
                    base: VIRT_GIC_V2M,
                    spi_base: (VIRT_GIC_V2M_IRQ + ARM_SPI_BASE) as u16,
                    spi_count: NUM_GICV2M_SPIS as u16,
                }),
                acpi_ged: VIRT_ACPI_GED,
                gpio: w(VIRT_GPIO, VIRT_GPIO_SIZE),
                mem: VIRT_MEM,
            },
            irqs: VirtIrqs {
                uart0: spi(VIRT_UART_IRQ),
                uart1: spi(VIRT_UART1_IRQ),
                virtio: spi(VIRT_MMIO_IRQ),
                pcie: spi(VIRT_PCIE_IRQ),
                acpi_ged: spi(VIRT_ACPI_GED_IRQ),
                gpio: spi(VIRT_GPIO_IRQ),
            },
            virtio_count: VIRTIO_TRANSPORTS as u32,
            mpidrs: (0..self.smp).map(virt_cpu_mp_affinity).collect(),
            possible_cpus: arm_virt::possible_cpus(
                topo.sockets,
                topo.clusters,
                topo.cores,
                topo.threads,
            ),
            has_clusters: topo.has_clusters,
            threads: topo.threads,
            // The GICC performance interrupt, when the CPU has a PMU.
            pmu_irq: if self.model.features.pmu != 0 { cpus::PMU_PPI } else { 0 },
            virtualization: self.virtualization,
            ns_el2_virt_timer: self.ns_el2_virt_timer_irq,
            psci: match self.vms_conduit {
                PsciConduit::Disabled => AcpiPsci::Disabled,
                PsciConduit::Hvc => AcpiPsci::Hvc,
                PsciConduit::Smc => AcpiPsci::Smc,
            },
            ged_events: self.ged.as_ref().map(|g| g.ged_event_bitmap()),
            spcr: self.spcr,
            pci_devices,
            numa: Vec::new(),
            // populate_smmuv3_legacy_dev(): the bus range of the root bus, which has no bridges
            // under it, goes through the SMMU unless the bus bypasses it.
            smmu: self.smmu.as_ref().map(|_| IortSmmu {
                base: VIRT_SMMU,
                gsi: spi(VIRT_SMMU_IRQ),
                rc_id_maps: if self.iommu_bypass { Vec::new() } else { vec![(0, 0x100)] },
            }),
        };
        arm_virt::build(&acpi)
    }

    /// The ACPI GED, there when firmware boots with ACPI.
    pub fn ged(&self) -> Option<&Arc<AcpiGed>> {
        self.ged.as_ref()
    }

    /// `virt_powerdown_req()`: the power button event of the GED for ACPI, or else a press
    /// of the PL061 power key.
    pub fn system_powerdown(&self) {
        if let Some(g) = &self.ged {
            g.power_down();
        } else if let Some((_, key)) = &self.gpio {
            key.press();
        }
    }

    /// The PL061 GPIO, there when the GED is not.
    pub fn gpio(&self) -> Option<&Arc<Pl061>> {
        self.gpio.as_ref().map(|(p, _)| p)
    }

    /// The secure PL061 of `secure=on`.
    pub fn secure_gpio(&self) -> Option<&Arc<Pl061>> {
        self.secure_gpio.as_ref()
    }

    /// The virtio-mmio transport at `index`, plugged or not.
    pub fn virtio_transport(&self, index: usize) -> Option<Arc<VirtioMmio>> {
        self.virtio.get(index).map(|slot| slot.current())
    }

    /// The PCIe host bridge.
    pub fn gpex(&self) -> &GpexHost {
        &self.gpex
    }

    /// The virtio PCI functions plugged with [`VirtMachine::attach_virtio_pci`].
    pub fn pci_devices(&self) -> Vec<VirtioPci> {
        self.pci_devices.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// Where the regions above RAM went.
    pub fn memmap(&self) -> &VirtMemmap {
        &self.memmap
    }

    /// Connect the chardev of the UART.
    pub fn set_serial_backend(&self, backend: Option<Arc<dyn SerialBackend>>) {
        self.uart.set_backend(backend);
    }

    /// `virt_machine_done()` and the rest of `qdev_machine_creation_done()`: finish and load
    /// the device tree, check the ROMs, work out the semihosting heap and reset the board.
    pub fn machine_done(&mut self) -> Result<(), String> {
        if self.done {
            return Ok(());
        }
        if self.dtb_filename.is_none() {
            dt::add_platform_bus_node(&mut self.fdt)?;
        }
        if let Some(fdt) = boot::arm_load_dtb(
            &mut self.loader,
            &self.info,
            &self.fdt,
            self.dtb_filename.as_deref(),
            &self.cmdline,
            self.arm.psci_conduit(),
        )? {
            self.fdt = fdt;
        }
        // virt_acpi_setup(). The tables never change, so unlike QEMU they are not rebuilt
        // when the firmware first reads them.
        if self.acpi {
            let tables = self.acpi_tables();
            let fwc = self.fw_cfg.state();
            fwc.add_file(TABLE_FILE, tables.table_data).map_err(err)?;
            fwc.add_file(LOADER_FILE, tables.linker.cmd_blob().to_vec()).map_err(err)?;
            // The TPM log, empty without a TPM.
            fwc.add_file(TPMLOG_FILE, Vec::new()).map_err(err)?;
            fwc.add_file(RSDP_FILE, tables.rsdp).map_err(err)?;
        }
        self.ram = ram_ranges(&self.mem, self.system)?;
        // common_semi_find_bases(): the largest gap in the largest RAM region.
        let mut best: Option<&RamRange> = None;
        for r in self.ram.iter().filter(|r| !r.readonly && !r.rom_device) {
            if best.is_none_or(|b| r.size > b.size) {
                best = Some(r);
            }
        }
        if let Some(b) = best {
            *self.heap.lock().unwrap_or_else(PoisonError::into_inner) =
                boot::largest_gap(&self.loader.roms, b.addr, b.size);
        }
        self.fw_cfg.state().machine_reset(Vec::new(), Vec::new()).map_err(err)?;
        if let Some(msg) = boot::rom_check(&self.loader.roms) {
            return Err(msg.trim_end().to_string());
        }
        self.done = true;
        self.system_reset()
    }

    /// `qemu_system_reset()` for the devices and the ROMs. Each vCPU is reset separately with
    /// [`VirtMachine::reset_cpu`] on its own thread.
    pub fn system_reset(&mut self) -> Result<(), String> {
        self.gic.reset();
        if let Some(its) = &self.its {
            its.reset();
        }
        self.uart.reset();
        if let Some(u) = &self.uart1 {
            u.reset();
        }
        for f in &self.flash {
            f.reset();
        }
        if let Some((pl061, key)) = &self.gpio {
            pl061.reset();
            key.reset();
        }
        if let Some(pl061) = &self.secure_gpio {
            pl061.reset();
        }
        for s in &self.virtio {
            s.current().reset();
        }
        self.gpex.reset();
        if let Some(s) = &self.smmu {
            s.reset();
        }
        let fwc = self.fw_cfg.state();
        fwc.reset();
        fwc.machine_reset(Vec::new(), Vec::new()).map_err(err)?;
        boot::rom_reset(&self.loader.roms, &self.ram);
        Ok(())
    }

    /// Make the vCPUs on `jit`, wire their interrupt lines to the GIC and reset them. Call it
    /// once, after [`VirtMachine::machine_done`], on a `jit` that has no vCPUs yet.
    pub fn create_vcpus(&self, jit: &Arc<Jit>) -> Result<Vec<Vcpu>, String> {
        if !self.done {
            return Err("the vCPUs are created after machine_done".to_string());
        }
        let mut vcpus = Vec::with_capacity(self.smp);
        for i in 0..self.smp {
            let st = CpuArmState::reset(&self.model);
            let mut v = create_vcpu(jit, self.arm.clone(), self.memory_as.clone(), &st);
            let shared = v.shared().clone();
            if shared.cpu_index != i {
                return Err(format!("vCPU {i} got index {}", shared.cpu_index));
            }
            match &self.gic {
                VirtGic::V2(g) => wire_cpu_v2(g, &self.arm, i, &shared),
                VirtGic::V3(g) => wire_cpu(g, &self.arm, i, &shared),
            }
            self.hub.register(i, &shared);
            self.reset_cpu(&mut v.cpu());
            vcpus.push(v);
        }
        Ok(vcpus)
    }

    /// `arm_cpu_reset_hold()` and `do_cpu_reset()` for the vCPU `cpu`: the register reset,
    /// the power state (the secondaries start off, for PSCI to start them) and the entry
    /// into the kernel.
    pub fn reset_cpu(&self, cpu: &mut Cpu<'_>) {
        let shared = cpu.core.shared().clone();
        let idx = shared.cpu_index;
        let mut st = CpuArmState::reset(&self.model);
        let info = &self.info;
        if info.direct {
            if !info.is_linux {
                // Jump to the entry point.
                st.pc = info.entry;
            } else {
                let el = if self.model.features.el2 { 2 } else { 1 };
                st.emulate_firmware_reset(&self.model.features, el);
                if idx == 0 {
                    st.pc = info.loader_start;
                }
            }
        }
        st.rebuild_hflags(&self.model.features);
        st.store(cpu.env);
        ruvm_jit::cputlb::tlb_flush(cpu);
        self.hub.reset_timers(idx);
        // Without PSCI the boot ROM sorts the secondaries out, so they all start running.
        self.arm.reset_power_state(&shared, idx != 0 && self.secondaries_off);
        self.arm.gic_reset(idx, &st);
    }

    /// The pending PSCI shutdown or reset request, taken. Requests made while a handler is
    /// set ([`VirtMachine::set_request_handler`]) go to the handler instead.
    pub fn take_request(&self) -> Option<VirtRequest> {
        self.hub.take_request()
    }

    /// Send the PSCI shutdown and reset requests to `handler`, on the thread of the vCPU
    /// that made them, rather than keeping them for [`VirtMachine::take_request`].
    pub fn set_request_handler(&self, handler: Option<VirtRequestHandler>) {
        self.hub.set_request_handler(handler);
    }

    /// The CPU operations every vCPU runs.
    pub fn arm(&self) -> &Arc<Arm> {
        &self.arm
    }

    /// The CPU model, with EL2 and EL3 taken away unless `virtualization=on` and `secure=on`
    /// keep them.
    pub fn cpu_model(&self) -> &ArmCpuModel {
        &self.model
    }

    /// The memory system.
    pub fn memory_system(&self) -> &Arc<MemorySystem> {
        &self.mem
    }

    /// The root region of the system address space.
    pub fn system_region(&self) -> RegionId {
        self.system
    }

    /// The system address space, `address_space_memory`.
    pub fn memory_as(&self) -> &Arc<AddressSpace> {
        &self.memory_as
    }

    /// The GIC.
    pub fn gic(&self) -> &VirtGic {
        &self.gic
    }

    /// The UART.
    pub fn uart(&self) -> &Arc<Pl011> {
        &self.uart
    }

    /// The second UART, if there is one: the secure UART with `secure=on`, or the non-secure
    /// one a second `-serial` adds.
    pub fn uart1(&self) -> Option<&Arc<Pl011>> {
        self.uart1.as_ref()
    }

    /// Connect the chardev of the second UART.
    pub fn set_serial1_backend(&self, backend: Option<Arc<dyn SerialBackend>>) {
        if let Some(u) = &self.uart1 {
            u.set_backend(backend);
        }
    }

    /// The two flashes at 0, `virt.flash0` and `virt.flash1`.
    pub fn flash(&self) -> &[Arc<Pflash>; 2] {
        &self.flash
    }

    /// The RTC.
    pub fn rtc(&self) -> &Arc<Pl031> {
        &self.rtc
    }

    /// fw_cfg.
    pub fn fw_cfg(&self) -> &FwCfgMem {
        &self.fw_cfg
    }

    /// The device tree: the board's until machine_done, then the one loaded into the guest,
    /// what `dumpdtb` writes.
    pub fn fdt(&self) -> &Fdt {
        &self.fdt
    }

    /// The ROM list, sorted by address.
    pub fn roms(&self) -> &[Rom] {
        &self.loader.roms
    }

    /// The semihosting heap base and limit, set at machine_done.
    pub fn heap_info(&self) -> (u64, u64) {
        *self.heap.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The RAM ranges of the system address space, set at machine_done.
    pub fn ram_ranges(&self) -> &[RamRange] {
        &self.ram
    }

    /// Where the kernel, initrd and device tree went.
    pub fn boot_info(&self) -> &BootInfo {
        &self.info
    }

    /// The messages printed while loading.
    pub fn messages(&self) -> &[String] {
        &self.loader.messages
    }

    /// The clock the generic timers run on. The runner calls its `run_timers()`.
    pub fn clock(&self) -> &Arc<Clock> {
        &self.clock
    }

    /// The number of CPUs.
    pub fn smp(&self) -> usize {
        self.smp
    }

    /// The RAM size.
    pub fn ram_size(&self) -> u64 {
        self.ram_size
    }
}
