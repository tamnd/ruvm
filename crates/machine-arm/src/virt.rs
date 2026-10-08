// SPDX-License-Identifier: GPL-2.0-or-later

//! The `virt` board, hw/arm/virt.c, with a GICv3 and AArch64 CPUs.
//!
//! # What is there
//!
//! RAM (`mach-virt.ram`) at 0x40000000, the GICv3 distributor at 0x08000000 and one
//! redistributor region at 0x080a0000, the generic timers wired to their PPIs, the PL011 UART at
//! 0x09000000 (SPI 1), the PL031 RTC at 0x09010000 (SPI 2), the generic PCIe host bridge
//! (`gpex-pcihost`) with its MMIO window at 0x10000000, its I/O port window at 0x3eff0000, its
//! ECAM and its high MMIO window above RAM (see [`memmap`]) and INTx on SPIs 3 to 6, fw_cfg
//! with DMA at 0x09020000, 32 virtio-mmio transports from 0x0a000000 (SPIs 16 to 47) and an
//! empty platform bus window at 0x0c000000, and the two CFI flashes at 0 and 0x04000000, the
//! first one holding `-bios`. A
//! second `-serial` adds the second PL011 at 0x09040000 (SPI 8). `virtualization=on` keeps EL2
//! and `secure=on` keeps EL3 and adds the secure UART at 0x09040000 and the secure RAM at
//! 0x0e000000. PSCI goes through HVC, or SMC with `virtualization=on`, and is left to the
//! firmware when `secure=on` has a firmware or the boot EL is at or above the conduit's EL, as
//! hw/arm/virt.c and hw/arm/boot.c decide. The device tree is built with the same libfdt calls in the
//! same order as QEMU, so apart from the nodes of the missing devices listed below it matches
//! `-M virt,dumpdtb=` byte for byte. `-kernel` takes an arm64 Image (raw, gzipped or EFI zboot
//! with gzip) or an AArch64 ELF, with `-initrd`, `-append` and `-dtb`, as hw/arm/boot.c loads
//! them, and the ROM list is checked for overlaps and copied into RAM at every reset.
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
//! - ITS (`its=on`): no `/intc/its` node and no MSI controller, so the PCIe node has no
//!   `msi-map` and the guest uses INTx. PCI functions still get their MSI-X capability, as in
//!   QEMU, where `msi_nonbroken` is always set (see `create_pcie()`).
//! - SMMUv3 (`iommu=smmuv3`).
//! - CXL, so the empty `cxl_host_reg` container QEMU maps above the redistributors is not
//!   there either.
//! - ACPI (`virt_acpi_setup()`, the GED device) and SMBIOS (`virt_build_smbios()`), so
//!   `virt_machine_done()` stops after `arm_load_dtb()`. Firmware that wants ACPI tables
//!   finds none in fw_cfg.
//! - The high memory redistributor region: its place is worked out, but the GIC only uses the
//!   low one, so at most 123 CPUs.
//!
//! Also missing: the PL061 GPIO with `gpio-keys` and the poweroff key and the secure PL061,
//! the PMU, NUMA, GICv2 and GICv5, the GICv3 virtual interface (`ICH_*`) and its maintenance
//! interrupt, the tag memory of the secure RAM with `mte=on` (only the RAM has tags), `-kernel`
//! together with firmware (the fw_cfg kernel path of
//! `arm_setup_firmware_boot()`), uImage and
//! u-boot ramdisks, zstd EFI zboot payloads, big-endian and ELF32 kernels, memory hotplug and
//! device memory, and `dtb-randomness` (the board behaves as with `dtb-randomness=off`: no
//! `kaslr-seed` and no `rng-seed`).
//!
//! # Differences from QEMU
//!
//! - There is one address space, so the secure-only devices of `secure=on` (the first flash,
//!   the secure UART and the secure RAM) are visible to non-secure accesses too.
//! - Errors come back as `Err` strings without the `qemu-system-aarch64: ` prefix instead of
//!   exiting. Messages QEMU prints and carries on after go to standard error and are kept in
//!   [`VirtMachine::messages`].
//! - The default CPU is cortex-a57; QEMU's is the 32-bit cortex-a15, which is not modelled.
//! - The CPU limit error leaves out "Try 'highmem-redists=on' for more CPUs", since there is
//!   no such option here.
//! - A gzip stream that ends early reports `inflate()` error -5 and corrupt data reports -3,
//!   as zlib would, but other zlib codes are not reproduced.
//! - The generic timer deadlines are host [`Instant`]s converted onto the board clock.
//! - Cache sizes in the CPU nodes come from the legacy CCSIDR layout of each model, which is
//!   what the four models use.

mod boot;
mod cpus;
mod dt;
pub mod memmap;

use std::fmt;
use std::sync::{Arc, Mutex, PoisonError, RwLock, Weak};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use ruvm_base::ClockType;
use ruvm_hw_char::pl011::{PL011_MMIO_SIZE, Pl011};
use ruvm_hw_char::serial::SerialBackend;
use ruvm_hw_core::fw_cfg::{
    DmaMemory, FW_CFG_CTL_SIZE, FW_CFG_DMA_SIZE, FW_CFG_NB_CPUS, FwCfgMachineConfig, FwCfgMem,
    fw_cfg_init_mem_dma,
};
use ruvm_hw_core::timer::TimeSource;
use ruvm_hw_core::{Clock, IrqLine};
use ruvm_hw_intc::gicv3::{GICV3_DIST_SIZE, GICV3_REDIST_SIZE, GicV3, GicV3Props};
use ruvm_hw_pci::regs::PCI_NUM_PINS;
use ruvm_hw_pci::{GpexConfig, GpexHost, GpexWindow, MsiTrigger};
use ruvm_hw_timer::pl031::{PL031_MMIO_SIZE, Pl031};
use ruvm_hw_virtio::mmio::{VIRTIO_MMIO_FORCE_LEGACY_DEFAULT, VIRTIO_MMIO_REGION_SIZE};
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
/// The most CPUs the GICv3 redistributor window has room for, `virt_max_cpus` in
/// `machvirt_init()`.
pub const VIRT_GICV3_MAX_CPUS: usize = (VIRT_GIC_REDIST_SIZE / GICV3_REDIST_SIZE) as usize;
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

/// What the board is built from: the `-cpu`, `-smp`, `-m`, `-kernel`, `-initrd`, `-append`,
/// `-dtb`, `-bios`, `-serial` and `-semihosting` options and the `secure` and
/// `virtualization` machine properties.
#[derive(Clone)]
pub struct VirtConfig {
    /// The CPU model.
    pub cpu: ArmCpuModel,
    /// The number of CPUs.
    pub smp: usize,
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
}

impl fmt::Debug for VirtConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VirtConfig")
            .field("cpu", &self.cpu.name)
            .field("smp", &self.smp)
            .field("ram_size", &self.ram_size)
            .field("kernel", &self.kernel)
            .field("initrd", &self.initrd)
            .field("append", &self.append)
            .field("dtb", &self.dtb)
            .field("virtualization", &self.virtualization)
            .field("secure", &self.secure)
            .field("mte", &self.mte)
            .field("highmem", &self.highmem)
            .field("firmware", &self.firmware)
            .field("pflash", &self.pflash)
            .field("serial", &self.serial.is_some())
            .field("serial1", &self.serial1.is_some())
            .field("semihosting", &self.semihosting.is_some())
            .field("semihosting_userspace", &self.semihosting_userspace)
            .finish_non_exhaustive()
    }
}

impl VirtConfig {
    /// One CPU of model `cpu`, the default RAM size and nothing to load.
    pub fn new(cpu: ArmCpuModel) -> VirtConfig {
        VirtConfig {
            cpu,
            smp: 1,
            ram_size: VIRT_DEFAULT_RAM_SIZE,
            kernel: None,
            initrd: None,
            append: None,
            dtb: None,
            virtualization: false,
            secure: false,
            mte: false,
            highmem: Highmem::default(),
            firmware: None,
            pflash: [PflashBacking::None, PflashBacking::None],
            serial: None,
            serial1: None,
            semihosting: None,
            semihosting_userspace: false,
            clock: None,
            rtc_clock: None,
            fw_cfg: FwCfgMachineConfig::default(),
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

fn err<E: fmt::Display>(e: E) -> String {
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
struct WeakGuestMemory(Weak<AddressSpace>);

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

/// The PCIe part of `create_pcie()`: the generic host bridge with the ECAM of `memmap`, the
/// low MMIO window mapped 1:1, the high one when there is one, the I/O port window, and its
/// INTx lines on SPIs `VIRT_PCIE_IRQ` to `VIRT_PCIE_IRQ + 3`.
fn create_pcie(
    mem: &Arc<MemorySystem>,
    system: RegionId,
    memmap: &VirtMemmap,
    gic: &Arc<GicV3>,
    memory_as: &Arc<AddressSpace>,
) -> Result<GpexHost, String> {
    let ecam = memmap.ecam;
    let config = GpexConfig {
        ecam: GpexWindow { base: ecam.base, size: ecam.size },
        mmio32: GpexWindow { base: VIRT_PCIE_MMIO, size: VIRT_PCIE_MMIO_SIZE },
        mmio64: memmap
            .high_mmio
            .map_or(GpexWindow::default(), |h| GpexWindow { base: h.base, size: h.size }),
        pio: GpexWindow { base: VIRT_PCIE_PIO, size: VIRT_PCIE_PIO_SIZE },
        ..GpexConfig::default()
    };
    let gpex = GpexHost::new(Arc::clone(mem), system, config).map_err(err)?;
    // Map only the first size_ecam bytes of ECAM space, and the MMIO windows at the same
    // address in PCI memory space as in the system's.
    let mut aliases = vec![
        ("pcie-ecam", gpex.ecam(), 0, ecam.base, ecam.size),
        ("pcie-mmio", gpex.mmio_window(), VIRT_PCIE_MMIO, VIRT_PCIE_MMIO, VIRT_PCIE_MMIO_SIZE),
    ];
    if let Some(h) = memmap.high_mmio {
        aliases.push(("pcie-mmio-high", gpex.mmio_window(), h.base, h.base, h.size));
    }
    for (name, target, offset, addr, size) in aliases {
        let alias = mem.new_alias(name, target, offset, size.into()).map_err(err)?;
        mem.add_subregion(system, addr, alias).map_err(err)?;
    }
    mem.add_subregion(system, VIRT_PCIE_PIO, gpex.ioport_window()).map_err(err)?;
    for i in 0..PCI_NUM_PINS {
        let irq = VIRT_PCIE_IRQ + i as u32;
        if let Some(pin) = gpex.irq(i) {
            pin.connect(gic.spi(irq));
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
fn msi_store(memory_as: &Arc<AddressSpace>) -> impl Fn(u64, u32) + Send + Sync + 'static {
    let weak = Arc::downgrade(memory_as);
    move |address, data| {
        if let Some(a) = weak.upgrade() {
            let _ = a.store(address, 4, data.into(), Endian::Little, MemTxAttrs::UNSPECIFIED);
        }
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
    gic: Arc<GicV3>,
    uart: Arc<Pl011>,
    uart1: Option<Arc<Pl011>>,
    flash: [Arc<Pflash>; 2],
    /// Whether the secondaries start powered off, from the conduit the board picked before
    /// `arm_load_kernel()` adjusted it.
    secondaries_off: bool,
    rtc: Arc<Pl031>,
    memmap: VirtMemmap,
    gpex: GpexHost,
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

        let max_cpus = VIRT_GICV3_MAX_CPUS;
        if smp > max_cpus {
            return Err(format!(
                "Number of SMP CPUs requested ({smp}) exceeds max CPUs supported by machine \
                 'mach-virt' ({max_cpus})"
            ));
        }
        if smp == 0 {
            return Err(
                "Invalid SMP CPUs 0. The min CPUs supported by machine 'virt' is 1".to_string()
            );
        }

        let mut fdt = Fdt::new();
        let clock_phandle = dt::create_fdt(&mut fdt, cfg.secure)?;

        let mpidrs: Vec<u64> = (0..smp).map(virt_cpu_mp_affinity).collect();
        // ns_el2_virt_timer_present().
        let ns_el2_virt_timer_irq = model.features.el2 && model.features.vh;
        dt::add_timer_nodes(&mut fdt, ns_el2_virt_timer_irq)?;
        dt::add_cpu_nodes(&mut fdt, &model, &mpidrs, vms_conduit != PsciConduit::Disabled)?;

        let ram = mem.new_ram(VIRT_RAM_ID, ram_size).map_err(err)?;
        mem.add_subregion(system, VIRT_MEM, ram).map_err(err)?;

        dt::virt_flash_fdt(&mut fdt, cfg.secure)?;

        // create_gic().
        let gic = GicV3::new(GicV3Props {
            num_cpu: smp,
            num_irq: VIRT_GIC_NUM_IRQ,
            revision: 3,
            security_extn: cfg.secure,
            redist_region_count: vec![smp as u32],
            mp_affinity: mpidrs.clone(),
            pribits: model.gic_pribits,
        })?;
        let r = mem.new_io("gicv3_dist", GICV3_DIST_SIZE.into(), gic.dist_ops()).map_err(err)?;
        mem.add_subregion(system, VIRT_GIC_DIST, r).map_err(err)?;
        let r = mem
            .new_io("gicv3_redist_region[0]", gic.redist_region_size(0).into(), gic.redist_ops(0))
            .map_err(err)?;
        mem.add_subregion(system, VIRT_GIC_REDIST, r).map_err(err)?;
        let gic_phandle = dt::add_gic_node(&mut fdt, cfg.virtualization)?;

        // arm_load_kernel(), which decides the PSCI conduit the CPUs are created with.
        let mut loader = Loader::default();
        let mut info = BootInfo { loader_start: VIRT_MEM, ram_size, ..BootInfo::default() };
        if firmware_loaded && cfg.kernel.is_some() {
            return Err("loading -kernel through fw_cfg for the firmware is not supported by \
                        ruvm yet"
                .to_string());
        }
        let files = BootFiles { kernel: cfg.kernel.as_deref(), initrd: cfg.initrd.as_deref() };
        boot::arm_load_kernel(&mut loader, &mut info, files)?;
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
        let hub = Arc::new(CpuHub::new(&gic, mpidrs.clone(), clock.clone()));
        let heap = Arc::new(Mutex::new((0, 0)));
        let cmdline = cfg.append.clone().unwrap_or_default();
        let mut arm = Arm::new(model.clone())
            .with_psci(conduit)
            .with_board(hub.clone())
            .with_gicv3(Arc::new(GicCpuIf(gic.clone())));
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
            uart.irq(0).connect(gic.spi(irq));
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
        rtc.irq().connect(gic.spi(VIRT_RTC_IRQ));
        dt::create_rtc(&mut fdt, clock_phandle)?;

        // create_pcie(). There is no MSI controller yet, so the device tree has no msi-map.
        let gpex = create_pcie(&mem, system, &memmap, &gic, &memory_as)?;
        dt::create_pcie(&mut fdt, &memmap, gic_phandle, None, VIRT_PCIE_IRQ)?;

        // create_virtio_devices().
        let mut virtio = Vec::with_capacity(VIRTIO_TRANSPORTS);
        for i in 0..VIRTIO_TRANSPORTS {
            let t = VirtioMmio::new(None, VIRTIO_MMIO_FORCE_LEGACY_DEFAULT).map_err(err)?;
            let slot = Arc::new(VirtioSlot {
                gsi: gic.spi(VIRT_MMIO_IRQ + i as u32),
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

        // create_platform_bus(): the window, with nothing on it yet.
        let pbus = mem.new_container("platform bus", VIRT_PLATFORM_BUS_SIZE.into()).map_err(err)?;
        mem.add_subregion(system, VIRT_PLATFORM_BUS, pbus).map_err(err)?;

        if info.is_linux {
            gic.arm_linux_init(false);
        }

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
            uart,
            uart1,
            flash: [flash0, flash1],
            secondaries_off: vms_conduit != PsciConduit::Disabled,
            rtc,
            memmap,
            gpex,
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
        if self.done {
            return Err("PCI devices must be plugged before machine_done".to_string());
        }
        let memory: Arc<dyn GuestMemory + Send + Sync> =
            Arc::new(WeakGuestMemory(Arc::downgrade(&self.memory_as)));
        let backend = VirtioBackend::new(class, memory).map_err(err)?;
        let dev = VirtioPci::new(self.gpex.bus(), devfn, backend, props).map_err(err)?;
        self.pci_devices.lock().unwrap_or_else(PoisonError::into_inner).push(dev.clone());
        Ok(dev)
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
        self.ram = self.ram_ranges_now()?;
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

    fn ram_ranges_now(&self) -> Result<Vec<RamRange>, String> {
        let view = self.mem.render(self.system).map_err(err)?;
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

    /// `qemu_system_reset()` for the devices and the ROMs. Each vCPU is reset separately with
    /// [`VirtMachine::reset_cpu`] on its own thread.
    pub fn system_reset(&mut self) -> Result<(), String> {
        self.gic.reset();
        self.uart.reset();
        if let Some(u) = &self.uart1 {
            u.reset();
        }
        for f in &self.flash {
            f.reset();
        }
        for s in &self.virtio {
            s.current().reset();
        }
        self.gpex.reset();
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
            self.wire(i, &shared);
            self.hub.register(i, &shared);
            self.reset_cpu(&mut v.cpu());
            vcpus.push(v);
        }
        Ok(vcpus)
    }

    /// Connect the four GIC outputs of CPU `i` to the vCPU `shared`.
    fn wire(&self, i: usize, shared: &Arc<CpuShared>) {
        type SetLine = fn(&Arm, &CpuShared, bool);
        let lines: [(&ruvm_hw_core::IrqPin, SetLine); 4] = [
            (self.gic.cpu_irq(i), Arm::set_irq),
            (self.gic.cpu_fiq(i), Arm::set_fiq),
            (self.gic.cpu_virq(i), Arm::set_virq),
            (self.gic.cpu_vfiq(i), Arm::set_vfiq),
        ];
        for (pin, set) in lines {
            let arm = Arc::downgrade(&self.arm);
            let cpu = Arc::downgrade(shared);
            pin.connect(IrqLine::from_fn(move |level| {
                if let (Some(a), Some(c)) = (arm.upgrade(), cpu.upgrade()) {
                    set(&a, &c, level != 0);
                }
            }));
        }
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
    pub fn gic(&self) -> &Arc<GicV3> {
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
