// SPDX-License-Identifier: GPL-2.0-or-later

//! The `sbsa-ref` board, hw/arm/sbsa-ref.c: the reference platform of the Arm Server Base
//! System Architecture, for firmware (Trusted Firmware-A and EDK2) that describes it to the
//! OS with ACPI.
//!
//! # What is there
//!
//! The two CFI flashes of 256 MiB at 0 (the first one secure, holding `-bios` or `pflash0`),
//! the secure RAM at 0x20000000, the GICv3 distributor at 0x40060000, the redistributors at
//! 0x40080000 with room for 512 CPUs, the ITS at 0x44081000, the `sbsa-ec` at 0x50000000,
//! the SBSA generic watchdog (`sbsa-gwdt`, refresh frame at 0x50010000, control frame at
//! 0x50011000, SPI 16), the PL011 UART at 0x60000000 (SPI 1), the PL031 RTC at 0x60010000
//! (SPI 2), the PL061 GPIO at 0x60020000 (SPI 7) with the power key on pin 3, the secure UART
//! at 0x60030000 (SPI 8) and the second secure UART at 0x60040000 (SPI 9), the SMMUv3 at
//! 0x60050000 (SPIs 12 to 15) in front of the PCIe root bus, the `sysbus-ahci` controller
//! with six ports at 0x60100000 (SPI 10), the generic PCIe host bridge with its I/O port
//! window at 0x7fff0000, its MMIO window at 0x80000000, its ECAM at 0xf0000000, its high MMIO
//! window at 0x100_0000_0000 and INTx on SPIs 3 to 6, and the RAM (`sbsa-ref.ram`) at
//! 0x100_0000_0000. The CPUs keep EL3 and EL2, their generic timers run at 1 GHz, and QEMU's
//! PSCI is off: the firmware implements it. The device tree only tells the firmware what
//! changes with the command line (the CPUs, their topology and the GIC), as QEMU's does.
//!
//! # Using it
//!
//! [`SbsaRefMachine::new`] builds the board. Plug drives into the AHCI ports with
//! [`SbsaRefMachine::attach_drive`] and virtio PCI functions with
//! [`SbsaRefMachine::attach_virtio_pci`], then call [`SbsaRefMachine::machine_done`]. The
//! rest is as for [`crate::virt::VirtMachine`]; [`crate::tcg_run::SbsaRefTcgMachine`] runs it.
//!
//! # Differences from QEMU
//!
//! - The xHCI controller at 0x60110000 (SPI 11) is not there, since there is no USB yet: its
//!   window reads as zero and ignores writes, so the firmware finds no controller.
//! - The default NIC (`e1000e`) and the `bochs-display` on the PCIe bus are not there, as
//!   the models do not exist yet.
//! - There is one address space, so the secure-only devices (the first flash, the secure
//!   RAM, the secure UARTs and the `sbsa-ec`) are visible to non-secure accesses too, and the
//!   SMMU has no secure view of memory.
//! - Not modelled: the secure EL2 timers, `reset-cbar`, NUMA and the `/distance-map` node.
//! - The watchdog's `pause`, `debug`, `none` and `inject-nmi` actions do nothing and no
//!   `WATCHDOG` event is sent.
//! - The default CPU is neoverse-n1 rather than neoverse-n2, which is not modelled, nor is
//!   neoverse-v1.
//! - Errors come back as `Err` strings without the `qemu-system-aarch64: ` prefix instead of
//!   exiting.

use std::fmt;
use std::sync::{Arc, Mutex, PoisonError, Weak};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use ruvm_base::ClockType;
use ruvm_hw_char::pl011::{PL011_MMIO_SIZE, Pl011};
use ruvm_hw_char::serial::SerialBackend;
use ruvm_hw_core::Clock;
use ruvm_hw_core::timer::TimeSource;
use ruvm_hw_intc::gicv3::{
    GICV3_DIST_SIZE, GICV3_REDIST_SIZE, GicV3, GicV3Its, GicV3Props, ITS_CONTROL_SIZE, ITS_SIZE,
    ITS_TRANS_SIZE,
};
use ruvm_hw_iommu::{SMMU_SIZE, SmmuStage, SmmuV3};
use ruvm_hw_misc::pl061::PL061_MMIO_SIZE;
use ruvm_hw_misc::sbsa_ec::SBSA_EC_MMIO_SIZE;
use ruvm_hw_misc::sbsa_gwdt::{SBSA_GWDT_CMMIO_SIZE, SBSA_GWDT_RMMIO_SIZE};
use ruvm_hw_misc::{
    GpioKey, Pl061, Pl061Props, SbsaEc, SbsaEcRequest, SbsaGwdt, SbsaGwdtProps, WatchdogAction,
};
use ruvm_hw_pci::GpexHost;
use ruvm_hw_storage::{BlockBackend, DmaMemory, DriveConfig, SYSBUS_AHCI_MMIO_SIZE, SysbusAhci};
use ruvm_hw_timer::pl031::{PL031_MMIO_SIZE, Pl031};
use ruvm_hw_virtio::{VirtioDeviceClass, VirtioPci, VirtioPciProps};
use ruvm_jit::{Cpu, Jit, Vcpu};
use ruvm_mem::{
    AccessConstraints, AccessCtx, AccessSize, AddressSpace, MemResult, MemTxAttrs, MemorySystem,
    MmioOps, RegionId,
};
use ruvm_target_arm::cpu::{ArmCpuModel, CpuArmState};
use ruvm_target_arm::tcg::{Arm, PsciConduit, create_vcpu};

use crate::fdt::{Fdt, sized_cells};
use crate::pflash::{Pflash, PflashBacking, PflashProps};
use crate::virt::boot::{self, BootFiles, Loader};
use crate::virt::cpus::{CpuHub, GicCpuIf};
use crate::virt::{
    BootInfo, CpuTopology, MemMapEntry, PciPlug, PcieLayout, RamRange, Rom, VirtRequest,
    VirtRequestHandler, create_pcie, err, ram_ranges, wire_cpu,
};

/// `SBSA_FLASH`: the two flashes, each half of it.
pub const SBSA_FLASH: u64 = 0;
/// The size of `SBSA_FLASH`.
pub const SBSA_FLASH_SIZE: u64 = 0x2000_0000;
/// `SBSA_SECURE_MEM`.
pub const SBSA_SECURE_MEM: u64 = 0x2000_0000;
/// The size of `SBSA_SECURE_MEM`.
pub const SBSA_SECURE_MEM_SIZE: u64 = 0x2000_0000;
/// `SBSA_CPUPERIPHS`, which only `reset-cbar` points at.
pub const SBSA_CPUPERIPHS: u64 = 0x4000_0000;
/// `SBSA_GIC_DIST`.
pub const SBSA_GIC_DIST: u64 = 0x4006_0000;
/// `SBSA_GIC_REDIST`.
pub const SBSA_GIC_REDIST: u64 = 0x4008_0000;
/// The size of `SBSA_GIC_REDIST`.
pub const SBSA_GIC_REDIST_SIZE: u64 = 0x0400_0000;
/// `SBSA_GIC_ITS`.
pub const SBSA_GIC_ITS: u64 = 0x4408_1000;
/// `SBSA_SECURE_EC`.
pub const SBSA_SECURE_EC: u64 = 0x5000_0000;
/// `SBSA_GWDT_REFRESH`.
pub const SBSA_GWDT_REFRESH: u64 = 0x5001_0000;
/// `SBSA_GWDT_CONTROL`.
pub const SBSA_GWDT_CONTROL: u64 = 0x5001_1000;
/// `SBSA_UART`.
pub const SBSA_UART: u64 = 0x6000_0000;
/// `SBSA_RTC`.
pub const SBSA_RTC: u64 = 0x6001_0000;
/// `SBSA_GPIO`.
pub const SBSA_GPIO: u64 = 0x6002_0000;
/// `SBSA_SECURE_UART`.
pub const SBSA_SECURE_UART: u64 = 0x6003_0000;
/// `SBSA_SECURE_UART_MM`.
pub const SBSA_SECURE_UART_MM: u64 = 0x6004_0000;
/// `SBSA_SMMU`.
pub const SBSA_SMMU: u64 = 0x6005_0000;
/// `SBSA_AHCI`.
pub const SBSA_AHCI: u64 = 0x6010_0000;
/// `SBSA_XHCI`, which has no controller here.
pub const SBSA_XHCI: u64 = 0x6011_0000;
/// The size of `SBSA_XHCI`.
pub const SBSA_XHCI_SIZE: u64 = 0x1_0000;
/// `SBSA_PCIE_PIO`.
pub const SBSA_PCIE_PIO: u64 = 0x7fff_0000;
/// The size of `SBSA_PCIE_PIO`.
pub const SBSA_PCIE_PIO_SIZE: u64 = 0x1_0000;
/// `SBSA_PCIE_MMIO`.
pub const SBSA_PCIE_MMIO: u64 = 0x8000_0000;
/// The size of `SBSA_PCIE_MMIO`.
pub const SBSA_PCIE_MMIO_SIZE: u64 = 0x7000_0000;
/// `SBSA_PCIE_ECAM`.
pub const SBSA_PCIE_ECAM: u64 = 0xf000_0000;
/// The size of `SBSA_PCIE_ECAM`.
pub const SBSA_PCIE_ECAM_SIZE: u64 = 0x1000_0000;
/// `SBSA_PCIE_MMIO_HIGH`.
pub const SBSA_PCIE_MMIO_HIGH: u64 = 0x1_0000_0000;
/// The size of `SBSA_PCIE_MMIO_HIGH`.
pub const SBSA_PCIE_MMIO_HIGH_SIZE: u64 = 0xff_0000_0000;
/// `SBSA_MEM`.
pub const SBSA_MEM: u64 = 0x100_0000_0000;
/// `RAMLIMIT_BYTES`, the size of `SBSA_MEM`.
pub const SBSA_RAMLIMIT: u64 = 8 << 40;

/// `sbsa_ref_irqmap[SBSA_UART]`.
pub const SBSA_UART_IRQ: u32 = 1;
/// `sbsa_ref_irqmap[SBSA_RTC]`.
pub const SBSA_RTC_IRQ: u32 = 2;
/// `sbsa_ref_irqmap[SBSA_PCIE]`, the first of four.
pub const SBSA_PCIE_IRQ: u32 = 3;
/// `sbsa_ref_irqmap[SBSA_GPIO]`.
pub const SBSA_GPIO_IRQ: u32 = 7;
/// `sbsa_ref_irqmap[SBSA_SECURE_UART]`.
pub const SBSA_SECURE_UART_IRQ: u32 = 8;
/// `sbsa_ref_irqmap[SBSA_SECURE_UART_MM]`.
pub const SBSA_SECURE_UART_MM_IRQ: u32 = 9;
/// `sbsa_ref_irqmap[SBSA_AHCI]`.
pub const SBSA_AHCI_IRQ: u32 = 10;
/// `sbsa_ref_irqmap[SBSA_XHCI]`.
pub const SBSA_XHCI_IRQ: u32 = 11;
/// `sbsa_ref_irqmap[SBSA_SMMU]`, the first of four.
pub const SBSA_SMMU_IRQ: u32 = 12;
/// `sbsa_ref_irqmap[SBSA_GWDT_WS0]`.
pub const SBSA_GWDT_WS0_IRQ: u32 = 16;

/// `NUM_IRQS + 32`, the `num-irq` of the GIC.
pub const SBSA_GIC_NUM_IRQ: u32 = 256 + 32;
/// `SBSA_GTIMER_HZ`, the generic timer and watchdog rate.
pub const SBSA_GTIMER_HZ: u64 = 1_000_000_000;
/// `NUM_SATA_PORTS`.
pub const NUM_SATA_PORTS: usize = 6;
/// The machine's `default_ram_size`.
pub const SBSA_DEFAULT_RAM_SIZE: u64 = 1 << 30;
/// The machine's `default_cpus`.
pub const SBSA_DEFAULT_CPUS: usize = 4;
/// `default_ram_id`.
pub const SBSA_RAM_ID: &str = "sbsa-ref.ram";
/// The CPUs the redistributor space has room for, also `max_cpus`.
pub const SBSA_MAX_CPUS: usize = (SBSA_GIC_REDIST_SIZE / GICV3_REDIST_SIZE) as usize;
/// The pin of the PL061 the power key drives.
const GPIO_PIN_POWER_BUTTON: u32 = 3;

/// What the board is built from: the `-cpu`, `-smp`, `-m`, `-kernel`, `-initrd`, `-append`,
/// `-dtb`, `-bios` and `-serial` options and the flash drives.
#[derive(Clone)]
pub struct SbsaRefConfig {
    /// The CPU model.
    pub cpu: ArmCpuModel,
    /// The number of CPUs.
    pub smp: usize,
    /// `maxcpus` of `-smp`. `None` means `smp`.
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
    /// `-bios`: the firmware image, loaded into the first flash.
    pub firmware: Option<String>,
    /// The drives of the two flashes, `pflash0` and `pflash1`.
    pub pflash: [PflashBacking; 2],
    /// The chardev of the UART, `serial_hd(0)`.
    pub serial: Option<Arc<dyn SerialBackend>>,
    /// The chardev of the secure UART, `serial_hd(1)`.
    pub serial1: Option<Arc<dyn SerialBackend>>,
    /// The chardev of the second secure UART, `serial_hd(2)`.
    pub serial2: Option<Arc<dyn SerialBackend>>,
    /// The clock the generic timers and the watchdog run on. The default follows the host's
    /// monotonic time.
    pub clock: Option<Arc<Clock>>,
    /// The clock of the RTC. The default is the host wall clock.
    pub rtc_clock: Option<Arc<Clock>>,
    /// The `-smp` topology, `ms->smp`. `None` is what the machine has without `-smp`: one
    /// socket, cluster, core and thread, whatever the number of CPUs.
    pub topology: Option<CpuTopology>,
}

impl fmt::Debug for SbsaRefConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SbsaRefConfig")
            .field("cpu", &self.cpu.name)
            .field("smp", &self.smp)
            .field("max_cpus", &self.max_cpus)
            .field("ram_size", &self.ram_size)
            .field("kernel", &self.kernel)
            .field("initrd", &self.initrd)
            .field("append", &self.append)
            .field("dtb", &self.dtb)
            .field("firmware", &self.firmware)
            .field("pflash", &self.pflash)
            .field("serial", &self.serial.is_some())
            .field("serial1", &self.serial1.is_some())
            .field("serial2", &self.serial2.is_some())
            .field("topology", &self.topology)
            .finish_non_exhaustive()
    }
}

impl SbsaRefConfig {
    /// The default number of CPUs of model `cpu`, the default RAM size and nothing to load.
    pub fn new(cpu: ArmCpuModel) -> SbsaRefConfig {
        SbsaRefConfig {
            cpu,
            smp: SBSA_DEFAULT_CPUS,
            max_cpus: None,
            ram_size: SBSA_DEFAULT_RAM_SIZE,
            kernel: None,
            initrd: None,
            append: None,
            dtb: None,
            firmware: None,
            pflash: [PflashBacking::None, PflashBacking::None],
            serial: None,
            serial1: None,
            serial2: None,
            clock: None,
            rtc_clock: None,
            topology: None,
        }
    }
}

impl Default for SbsaRefConfig {
    fn default() -> SbsaRefConfig {
        SbsaRefConfig::new(ArmCpuModel::by_name("neoverse-n1").expect("neoverse-n1 exists"))
    }
}

/// `arm_build_mp_affinity()` with `ARM_DEFAULT_CPUS_PER_CLUSTER`: 8 CPUs per Aff1 cluster.
pub fn sbsa_ref_cpu_mp_affinity(idx: usize) -> u64 {
    let idx = idx as u64;
    ((idx / 8) << 8) | (idx % 8)
}

/// AHCI DMA through a weak reference, so the address space that maps the controller does not
/// keep itself alive.
struct WeakAhciDma(Weak<AddressSpace>);

impl DmaMemory for WeakAhciDma {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.0.upgrade().is_some_and(|a| a.read(addr, MemTxAttrs::UNSPECIFIED, buf).is_ok())
    }

    fn dma_write(&self, addr: u64, buf: &[u8]) -> bool {
        self.0.upgrade().is_some_and(|a| a.write(addr, MemTxAttrs::UNSPECIFIED, buf).is_ok())
    }
}

/// QEMU's `unimplemented-device`, without its log: reads as zero and ignores writes.
struct Unimplemented;

impl MmioOps for Unimplemented {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(0)
    }

    fn write(
        &self,
        _cx: &AccessCtx,
        _offset: u64,
        _size: AccessSize,
        _value: u64,
    ) -> MemResult<()> {
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 8)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 8)
    }
}

/// `create_fdt()`: the root, the CPUs with their topology and the GIC with its ITS. NUMA is
/// not modelled, so there is no `/distance-map` and no `numa-node-id`.
fn create_fdt(fdt: &mut Fdt, mpidrs: &[u64], topology: &CpuTopology) -> Result<(), String> {
    fdt.setprop_string("/", "compatible", "linux,sbsa-ref")?;
    fdt.setprop_cell("/", "#address-cells", 0x2)?;
    fdt.setprop_cell("/", "#size-cells", 0x2)?;
    // This versioning scheme is for informing platform firmware only.
    fdt.setprop_cell("/", "machine-version-major", 0)?;
    fdt.setprop_cell("/", "machine-version-minor", 4)?;

    fdt.add_subnode("/cpus")?;
    fdt.setprop_cell("/cpus", "#address-cells", 2)?;
    fdt.setprop_cell("/cpus", "#size-cells", 0x0)?;
    for (cpu, &mpidr) in mpidrs.iter().enumerate().rev() {
        let nodename = format!("/cpus/cpu@{cpu}");
        fdt.add_subnode(&nodename)?;
        fdt.setprop_u64(&nodename, "reg", mpidr)?;
    }

    fdt.add_subnode("/cpus/topology")?;
    fdt.setprop_cell("/cpus/topology", "sockets", topology.sockets)?;
    fdt.setprop_cell("/cpus/topology", "clusters", topology.clusters)?;
    fdt.setprop_cell("/cpus/topology", "cores", topology.cores)?;
    fdt.setprop_cell("/cpus/topology", "threads", topology.threads)?;

    // sbsa_fdt_add_gic_node().
    let reg = |values: &[(u32, u64)], path: &str| {
        sized_cells(values).ok_or_else(|| {
            format!("qemu_fdt_setprop_sized_cells: Couldn't set {path}/reg: FDT_ERR_BADVALUE")
        })
    };
    fdt.add_subnode("/intc")?;
    let v = reg(
        &[
            (2, SBSA_GIC_DIST),
            (2, GICV3_DIST_SIZE),
            (2, SBSA_GIC_REDIST),
            (2, SBSA_GIC_REDIST_SIZE),
        ],
        "/intc",
    )?;
    fdt.setprop("/intc", "reg", &v)?;
    fdt.add_subnode("/intc/its")?;
    let v = reg(&[(2, SBSA_GIC_ITS), (2, ITS_SIZE)], "/intc/its")?;
    fdt.setprop("/intc/its", "reg", &v)
}

/// The sbsa-ref board.
pub struct SbsaRefMachine {
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
    gic: Arc<GicV3>,
    its: Arc<GicV3Its>,
    uart: Arc<Pl011>,
    secure_uart: Arc<Pl011>,
    secure_uart_mm: Arc<Pl011>,
    flash: [Arc<Pflash>; 2],
    rtc: Arc<Pl031>,
    wdt: Arc<SbsaGwdt>,
    gpio: Arc<Pl061>,
    key: Arc<GpioKey>,
    ahci: Arc<SysbusAhci>,
    gpex: GpexHost,
    smmu: Arc<SmmuV3>,
    /// The address spaces of the functions behind the SMMU, `SMMUDevice.as`.
    iommu_spaces: Mutex<Vec<Arc<AddressSpace>>>,
    pci_devices: Mutex<Vec<VirtioPci>>,
    fdt: Fdt,
    loader: Loader,
    info: BootInfo,
    ram: Vec<RamRange>,
    clock: Arc<Clock>,
    done: bool,
}

impl fmt::Debug for SbsaRefMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SbsaRefMachine")
            .field("cpu", &self.model.name)
            .field("smp", &self.smp)
            .field("ram_size", &self.ram_size)
            .field("boot", &self.info)
            .field("roms", &self.loader.roms)
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}

impl SbsaRefMachine {
    /// `sbsa_ref_init()`: build the board and load the kernel.
    pub fn new(cfg: SbsaRefConfig) -> Result<SbsaRefMachine, String> {
        // The CPUs keep EL3 and EL2, and count at SBSA_GTIMER_HZ.
        let mut model = cfg.cpu.with_el3().with_el2();
        model.cntfrq = SBSA_GTIMER_HZ;
        let smp = cfg.smp;
        let ram_size = cfg.ram_size;

        let mem = Arc::new(MemorySystem::new());
        let system = mem.new_container("system", 1 << 64).map_err(err)?;
        let memory_as = mem.address_space_init(system, "memory").map_err(err)?;

        // sbsa_firmware_init(). There is one address space, so the first flash, secure only,
        // is in the system memory too.
        let [pflash0, pflash1] = cfg.pflash;
        let pflash0_given = pflash0 != PflashBacking::None;
        let half = SBSA_FLASH_SIZE / 2;
        let flash0 = Pflash::new(&mem, "sbsa.flash0", PflashProps::virt_flash(half), pflash0)?;
        let flash1 = Pflash::new(&mem, "sbsa.flash1", PflashProps::virt_flash(half), pflash1)?;
        mem.add_subregion(system, SBSA_FLASH, flash0.region()).map_err(err)?;
        mem.add_subregion(system, SBSA_FLASH + half, flash1.region()).map_err(err)?;
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

        // This machine has EL3, and the firmware supplies PSCI, so QEMU's is disabled.
        let max_cpus = cfg.max_cpus.unwrap_or(smp);
        if max_cpus > SBSA_MAX_CPUS {
            return Err(format!(
                "Number of SMP CPUs requested ({max_cpus}) exceeds max CPUs supported by \
                 machine 'sbsa-ref' ({SBSA_MAX_CPUS})"
            ));
        }
        if smp == 0 {
            return Err(
                "Invalid SMP CPUs 0. The min CPUs supported by machine 'sbsa-ref' is 1".to_string()
            );
        }
        if ram_size > SBSA_RAMLIMIT {
            return Err("sbsa-ref: cannot model more than 8 TiB of RAM".to_string());
        }
        let topology = cfg.topology.unwrap_or(CpuTopology {
            sockets: 1,
            clusters: 1,
            cores: 1,
            threads: 1,
            has_clusters: false,
        });

        let mpidrs: Vec<u64> = (0..smp).map(sbsa_ref_cpu_mp_affinity).collect();
        let ram = mem.new_ram(SBSA_RAM_ID, ram_size).map_err(err)?;
        mem.add_subregion(system, SBSA_MEM, ram).map_err(err)?;

        let mut fdt = Fdt::new();
        create_fdt(&mut fdt, &mpidrs, &topology)?;

        // create_secure_ram(), in the one address space.
        let r = mem.new_ram("sbsa-ref.secure-ram", SBSA_SECURE_MEM_SIZE).map_err(err)?;
        mem.add_subregion(system, SBSA_SECURE_MEM, r).map_err(err)?;

        // create_gic(), then create_its().
        let gic = GicV3::with_sysmem(
            GicV3Props {
                num_cpu: smp,
                num_irq: SBSA_GIC_NUM_IRQ,
                revision: 3,
                security_extn: true,
                redist_region_count: vec![smp.min(SBSA_MAX_CPUS) as u32],
                mp_affinity: mpidrs.clone(),
                pribits: model.gic_pribits,
                has_lpi: true,
            },
            Some(&memory_as),
        )?;
        let r = mem.new_io("gicv3_dist", GICV3_DIST_SIZE.into(), gic.dist_ops()).map_err(err)?;
        mem.add_subregion(system, SBSA_GIC_DIST, r).map_err(err)?;
        let r = mem
            .new_io("gicv3_redist_region[0]", gic.redist_region_size(0).into(), gic.redist_ops(0))
            .map_err(err)?;
        mem.add_subregion(system, SBSA_GIC_REDIST, r).map_err(err)?;
        let its = GicV3Its::new(&gic)?;
        let main = mem.new_container("gicv3_its", ITS_SIZE.into()).map_err(err)?;
        let r = mem.new_io("control", ITS_CONTROL_SIZE.into(), its.control_ops()).map_err(err)?;
        mem.add_subregion(main, 0, r).map_err(err)?;
        let r =
            mem.new_io("translation", ITS_TRANS_SIZE.into(), its.translation_ops()).map_err(err)?;
        mem.add_subregion(main, ITS_CONTROL_SIZE, r).map_err(err)?;
        mem.add_subregion(system, SBSA_GIC_ITS, main).map_err(err)?;

        // arm_load_kernel(). Without fw_cfg the firmware cannot be handed a kernel.
        if firmware_loaded && cfg.kernel.is_some() {
            return Err("This machine type does not support loading both a guest firmware/BIOS \
                        image and a guest kernel at the same time. You should change your QEMU \
                        command line to specify one or the other, but not both."
                .to_string());
        }
        let mut loader = Loader::default();
        let mut info = BootInfo { loader_start: SBSA_MEM, ram_size, ..BootInfo::default() };
        let files = BootFiles { kernel: cfg.kernel.as_deref(), initrd: cfg.initrd.as_deref() };
        boot::arm_load_kernel(&mut loader, &mut info, files, firmware_loaded)?;

        let clock = cfg.clock.unwrap_or_else(|| {
            Clock::new(ClockType::Virtual, TimeSource::Monotonic(Instant::now()))
        });
        let hub = Arc::new(CpuHub::new(&gic, mpidrs.clone(), clock.clone()));
        let arm = Arc::new(
            Arm::new(model.clone())
                .with_psci(PsciConduit::Disabled)
                .with_board(hub.clone())
                .with_gicv3(Arc::new(GicCpuIf(gic.clone()))),
        );
        hub.set_arm(&arm);
        for (i, &m) in mpidrs.iter().enumerate() {
            arm.set_mpidr(i, m);
        }

        // create_uart() for the UART and the two secure ones.
        let make_uart = |base, irq, chr| -> Result<Arc<Pl011>, String> {
            let uart = Pl011::new(chr);
            let r = mem.new_io("pl011", PL011_MMIO_SIZE.into(), uart.clone()).map_err(err)?;
            mem.add_subregion(system, base, r).map_err(err)?;
            uart.irq(0).connect(gic.spi(irq));
            Ok(uart)
        };
        let uart = make_uart(SBSA_UART, SBSA_UART_IRQ, cfg.serial)?;
        let secure_uart = make_uart(SBSA_SECURE_UART, SBSA_SECURE_UART_IRQ, cfg.serial1)?;
        let secure_uart_mm = make_uart(SBSA_SECURE_UART_MM, SBSA_SECURE_UART_MM_IRQ, cfg.serial2)?;

        // create_rtc().
        let rtc_clock =
            cfg.rtc_clock.unwrap_or_else(|| Clock::new(ClockType::Host, TimeSource::Wall));
        let rtc_date =
            if rtc_clock.kind() == ClockType::Host { UNIX_EPOCH } else { SystemTime::now() };
        let rtc = Pl031::new(rtc_clock, rtc_date);
        let r = mem.new_io("pl031", PL031_MMIO_SIZE.into(), rtc.clone()).map_err(err)?;
        mem.add_subregion(system, SBSA_RTC, r).map_err(err)?;
        rtc.irq().connect(gic.spi(SBSA_RTC_IRQ));

        // The power key, which the watchdog's shutdown action presses too.
        let key = GpioKey::new(clock.clone());

        // create_wdt(). watchdog_perform_action(): reset is a reset request, shutdown a
        // power down request and poweroff a shutdown request.
        let wdt = {
            let (hub, key) = (Arc::downgrade(&hub), Arc::downgrade(&key));
            SbsaGwdt::new(
                clock.clone(),
                SbsaGwdtProps { clock_frequency: SBSA_GTIMER_HZ, wdat: false },
                Arc::new(move |action| match action {
                    WatchdogAction::Reset => {
                        if let Some(h) = hub.upgrade() {
                            h.request(VirtRequest::Reset);
                        }
                    }
                    WatchdogAction::Shutdown => {
                        if let Some(k) = key.upgrade() {
                            k.press();
                        }
                    }
                    WatchdogAction::Poweroff => {
                        if let Some(h) = hub.upgrade() {
                            h.request(VirtRequest::Shutdown);
                        }
                    }
                    _ => {}
                }),
            )
        };
        let r = mem.new_io("sbsa_gwdt.refresh", SBSA_GWDT_RMMIO_SIZE.into(), wdt.refresh_ops());
        mem.add_subregion(system, SBSA_GWDT_REFRESH, r.map_err(err)?).map_err(err)?;
        let r = mem.new_io("sbsa_gwdt.control", SBSA_GWDT_CMMIO_SIZE.into(), wdt.control_ops());
        mem.add_subregion(system, SBSA_GWDT_CONTROL, r.map_err(err)?).map_err(err)?;
        wdt.irq().connect(gic.spi(SBSA_GWDT_WS0_IRQ));

        // create_gpio(): the PL061 with its default pulls, and the power key on pin 3.
        let gpio = Pl061::new(Pl061Props::default())?;
        let r = mem.new_io("pl061", PL061_MMIO_SIZE.into(), gpio.clone()).map_err(err)?;
        mem.add_subregion(system, SBSA_GPIO, r).map_err(err)?;
        gpio.irq().connect(gic.spi(SBSA_GPIO_IRQ));
        key.irq().connect(gpio.gpio_in(GPIO_PIN_POWER_BUTTON));

        // create_ahci().
        let dma: Arc<dyn DmaMemory> = Arc::new(WeakAhciDma(Arc::downgrade(&memory_as)));
        let ahci = SysbusAhci::new(NUM_SATA_PORTS, dma);
        let r = mem.new_io("ahci", SYSBUS_AHCI_MMIO_SIZE.into(), ahci.mmio_ops()).map_err(err)?;
        mem.add_subregion(system, SBSA_AHCI, r).map_err(err)?;
        ahci.irq().connect(gic.spi(SBSA_AHCI_IRQ));

        // create_xhci() is left out: there is no xHCI model yet. Its window reads as zero and
        // ignores writes, as an `unimplemented-device` would, so that the firmware's probe
        // fails instead of taking an external abort.
        let r = mem.new_io("xhci", SBSA_XHCI_SIZE.into(), Arc::new(Unimplemented)).map_err(err)?;
        mem.add_subregion(system, SBSA_XHCI, r).map_err(err)?;

        // create_pcie(), then create_smmu() in front of its root bus.
        let layout = PcieLayout {
            ecam: MemMapEntry { base: SBSA_PCIE_ECAM, size: SBSA_PCIE_ECAM_SIZE },
            mmio: MemMapEntry { base: SBSA_PCIE_MMIO, size: SBSA_PCIE_MMIO_SIZE },
            high_mmio: Some(MemMapEntry {
                base: SBSA_PCIE_MMIO_HIGH,
                size: SBSA_PCIE_MMIO_HIGH_SIZE,
            }),
            pio: MemMapEntry { base: SBSA_PCIE_PIO, size: SBSA_PCIE_PIO_SIZE },
            irq: SBSA_PCIE_IRQ,
        };
        let gpex = create_pcie(&mem, system, &layout, &gic, &memory_as)?;
        let irqs = std::array::from_fn(|i| gic.spi(SBSA_SMMU_IRQ + i as u32));
        let smmu = SmmuV3::new(SmmuStage::Nested, &memory_as, irqs);
        let r = mem.new_io("smmuv3", SMMU_SIZE.into(), smmu.mmio_ops()).map_err(err)?;
        mem.add_subregion(system, SBSA_SMMU, r).map_err(err)?;

        // create_secure_ec().
        let ec = {
            let hub = Arc::downgrade(&hub);
            SbsaEc::new(move |req| {
                if let Some(h) = hub.upgrade() {
                    h.request(match req {
                        SbsaEcRequest::Poweroff => VirtRequest::Shutdown,
                        SbsaEcRequest::Reboot => VirtRequest::Reset,
                    });
                }
            })
        };
        let r = mem.new_io("sbsa-ec", SBSA_EC_MMIO_SIZE.into(), ec).map_err(err)?;
        mem.add_subregion(system, SBSA_SECURE_EC, r).map_err(err)?;

        if info.is_linux {
            gic.arm_linux_init(false);
        }

        Ok(SbsaRefMachine {
            model,
            smp,
            ram_size,
            dtb_filename: cfg.dtb,
            cmdline: cfg.append.unwrap_or_default(),
            mem,
            system,
            memory_as,
            arm,
            hub,
            gic,
            its,
            uart,
            secure_uart,
            secure_uart_mm,
            flash: [flash0, flash1],
            rtc,
            wdt,
            gpio,
            key,
            ahci,
            gpex,
            smmu,
            iommu_spaces: Mutex::new(Vec::new()),
            pci_devices: Mutex::new(Vec::new()),
            fdt,
            loader,
            info,
            ram: Vec::new(),
            clock,
            done: false,
        })
    }

    /// Plug a drive into AHCI port `port`, as `ahci_ide_create_devs()` does for the IDE drive
    /// with that index.
    pub fn attach_drive(
        &self,
        port: usize,
        config: DriveConfig,
        blk: Option<Arc<dyn BlockBackend>>,
    ) -> Result<(), String> {
        if self.done {
            return Err("drives are plugged before machine_done".to_string());
        }
        self.ahci.attach_drive(port, config, blk).map_err(err)
    }

    /// Plug a virtio device into a new function on the root bus, at the first free slot.
    pub fn attach_virtio_pci(&self, class: Box<dyn VirtioDeviceClass>) -> Result<(), String> {
        self.attach_virtio_pci_with(class, None, &VirtioPciProps::default(), false)
    }

    /// Plug a virtio device into a new function on the root bus, at `devfn` or the first
    /// free slot, with `props`. With `iommu_platform` its DMA goes through the SMMU.
    pub fn attach_virtio_pci_with(
        &self,
        class: Box<dyn VirtioDeviceClass>,
        devfn: Option<u8>,
        props: &VirtioPciProps,
        iommu_platform: bool,
    ) -> Result<(), String> {
        if self.done {
            return Err("devices are plugged before machine_done".to_string());
        }
        let plug = PciPlug {
            mem: &self.mem,
            memory_as: &self.memory_as,
            gpex: &self.gpex,
            smmu: Some(&self.smmu),
            iommu_spaces: &self.iommu_spaces,
        };
        let dev = plug.virtio_pci(class, devfn, props, iommu_platform)?;
        self.pci_devices.lock().unwrap_or_else(PoisonError::into_inner).push(dev);
        Ok(())
    }

    /// `sbsa_ref_powerdown_req()`: a press of the power key.
    pub fn system_powerdown(&self) {
        self.key.press();
    }

    /// The rest of `arm_load_kernel()` and `qdev_machine_creation_done()`: load the device
    /// tree, check the ROMs and reset the board.
    pub fn machine_done(&mut self) -> Result<(), String> {
        if self.done {
            return Ok(());
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
        self.ram = ram_ranges(&self.mem, self.system)?;
        if let Some(msg) = boot::rom_check(&self.loader.roms) {
            return Err(msg.trim_end().to_string());
        }
        self.done = true;
        self.system_reset()
    }

    /// `qemu_system_reset()` for the devices and the ROMs. Each vCPU is reset separately with
    /// [`SbsaRefMachine::reset_cpu`] on its own thread.
    pub fn system_reset(&mut self) -> Result<(), String> {
        self.gic.reset();
        self.its.reset();
        for u in [&self.uart, &self.secure_uart, &self.secure_uart_mm] {
            u.reset();
        }
        for f in &self.flash {
            f.reset();
        }
        self.wdt.reset();
        self.gpio.reset();
        self.key.reset();
        self.ahci.reset();
        self.gpex.reset();
        self.smmu.reset();
        boot::rom_reset(&self.loader.roms, &self.ram);
        Ok(())
    }

    /// Make the vCPUs on `jit`, wire their interrupt lines to the GIC and reset them. Call it
    /// once, after [`SbsaRefMachine::machine_done`], on a `jit` that has no vCPUs yet.
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
            wire_cpu(&self.gic, &self.arm, i, &shared);
            self.hub.register(i, &shared);
            self.reset_cpu(&mut v.cpu());
            vcpus.push(v);
        }
        Ok(vcpus)
    }

    /// `arm_cpu_reset_hold()` and `do_cpu_reset()` for the vCPU `cpu`. QEMU's PSCI is off, so
    /// every CPU starts running, for the firmware to sort the secondaries out.
    pub fn reset_cpu(&self, cpu: &mut Cpu<'_>) {
        let shared = cpu.core.shared().clone();
        let idx = shared.cpu_index;
        let mut st = CpuArmState::reset(&self.model);
        let info = &self.info;
        if info.direct {
            if !info.is_linux {
                st.pc = info.entry;
            } else {
                st.emulate_firmware_reset(&self.model.features, 2);
                if idx == 0 {
                    st.pc = info.loader_start;
                }
            }
        }
        st.rebuild_hflags(&self.model.features);
        st.store(cpu.env);
        ruvm_jit::cputlb::tlb_flush(cpu);
        self.hub.reset_timers(idx);
        self.arm.reset_power_state(&shared, false);
        self.arm.gic_reset(idx, &st);
    }

    /// The pending shutdown or reset request, taken. Requests made while a handler is set
    /// ([`SbsaRefMachine::set_request_handler`]) go to the handler instead.
    pub fn take_request(&self) -> Option<VirtRequest> {
        self.hub.take_request()
    }

    /// Send the shutdown and reset requests (of the `sbsa-ec` and the watchdog) to `handler`,
    /// on the thread that made them, rather than keeping them for
    /// [`SbsaRefMachine::take_request`].
    pub fn set_request_handler(&self, handler: Option<VirtRequestHandler>) {
        self.hub.set_request_handler(handler);
    }

    /// The CPU operations every vCPU runs.
    pub fn arm(&self) -> &Arc<Arm> {
        &self.arm
    }

    /// The CPU model, with EL3 and EL2 and the 1 GHz counter.
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
    pub fn gic(&self) -> &Arc<GicV3> {
        &self.gic
    }

    /// The UART.
    pub fn uart(&self) -> &Arc<Pl011> {
        &self.uart
    }

    /// The secure UART and the second secure UART.
    pub fn secure_uarts(&self) -> [&Arc<Pl011>; 2] {
        [&self.secure_uart, &self.secure_uart_mm]
    }

    /// Connect the chardev of the UART.
    pub fn set_serial_backend(&self, backend: Option<Arc<dyn SerialBackend>>) {
        self.uart.set_backend(backend);
    }

    /// Connect the chardev of secure UART `n`, 0 for `serial_hd(1)` and 1 for `serial_hd(2)`.
    pub fn set_secure_serial_backend(&self, n: usize, backend: Option<Arc<dyn SerialBackend>>) {
        if let Some(u) = self.secure_uarts().get(n) {
            u.set_backend(backend);
        }
    }

    /// The two flashes at 0, `sbsa.flash0` and `sbsa.flash1`.
    pub fn flash(&self) -> &[Arc<Pflash>; 2] {
        &self.flash
    }

    /// The RTC.
    pub fn rtc(&self) -> &Arc<Pl031> {
        &self.rtc
    }

    /// The watchdog.
    pub fn watchdog(&self) -> &Arc<SbsaGwdt> {
        &self.wdt
    }

    /// The PL061 GPIO.
    pub fn gpio(&self) -> &Arc<Pl061> {
        &self.gpio
    }

    /// The AHCI controller.
    pub fn ahci(&self) -> &Arc<SysbusAhci> {
        &self.ahci
    }

    /// The PCIe host bridge.
    pub fn gpex(&self) -> &GpexHost {
        &self.gpex
    }

    /// The SMMU.
    pub fn smmu(&self) -> &Arc<SmmuV3> {
        &self.smmu
    }

    /// The virtio PCI functions plugged with [`SbsaRefMachine::attach_virtio_pci`].
    pub fn pci_devices(&self) -> Vec<VirtioPci> {
        self.pci_devices.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// The device tree: the board's until machine_done, then the one loaded into the guest.
    pub fn fdt(&self) -> &Fdt {
        &self.fdt
    }

    /// The ROM list, sorted by address.
    pub fn roms(&self) -> &[Rom] {
        &self.loader.roms
    }

    /// Where the kernel, initrd and device tree went.
    pub fn boot_info(&self) -> &BootInfo {
        &self.info
    }

    /// The messages printed while loading.
    pub fn messages(&self) -> &[String] {
        &self.loader.messages
    }

    /// The clock the generic timers and the watchdog run on. The runner calls its
    /// `run_timers()`.
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
