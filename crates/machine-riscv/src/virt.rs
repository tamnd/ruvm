// SPDX-License-Identifier: GPL-2.0-or-later

//! The RISC-V `virt` board, hw/riscv/virt.c, with RV64 harts, the SiFive CLINT and the PLIC
//! or the AIA interrupt controllers.
//!
//! # What is there
//!
//! The memory map is `virt_memmap`: the boot ROM (`riscv_virt_board.mrom`) at 0x1000, the
//! SiFive test device at 0x100000 (shutdown and reset), the goldfish RTC at 0x101000 (PLIC
//! source 11), the CLINT at 0x2000000 (the ACLINT MSWI and, at 0x2004000, the MTIMER at
//! 10 MHz), the PCIe I/O port window at 0x3000000, the empty platform bus window at
//! 0x4000000, the PLIC at 0xc000000 (96 sources, 7 priorities, an M and an S context per
//! hart), the 16550 UART at 0x10000000 (PLIC source 10), eight virtio-mmio transports from
//! 0x10001000 (PLIC sources 1 to 8), fw_cfg with DMA at 0x10100000, the two CFI flashes at
//! 0x20000000 and 0x22000000, the PCIe ECAM and MMIO windows at 0x30000000 and 0x40000000
//! (and the high one above RAM) and RAM (`riscv_virt_board.ram`) at 0x80000000.
//!
//! The PCIe host bridge is the generic one, `gpex-pcihost`, with its root function at
//! 00:00.0 and INTx A to D on PLIC sources 32 to 35 with the usual swizzle. Unmapped parts of
//! its windows read as all ones and ignore writes. The PLIC sets `msi_nonbroken` in QEMU, so
//! PCI functions get their MSI-X capability here too; a message is a 32 bit store into system
//! memory, where nothing takes it without AIA, and the device tree gives the bridge no
//! `msi-parent`, so Linux uses INTx.
//!
//! `aia=aplic` replaces the PLIC with two APLIC domains in direct mode ([`Aia`]): the M level
//! one at 0xc000000 takes the device interrupts on the same sources and can delegate them to
//! the S level one at 0xd000000. `aia=aplic-imsic` puts both domains in MSI mode and gives
//! each hart an M level IMSIC (at 0x24000000, a page per hart) and an S level IMSIC (at
//! 0x28000000) with `aia-guests` guest files. The harts then have Smaia and Ssaia (and GEILEN
//! is `aia-guests` if they have H), and the PCIe bridge gets the S level IMSICs as its
//! `msi-parent`. The domains and PCI functions send their messages as 32 bit stores into
//! system memory.
//!
//! Each hart's MSIP, MTIP, M external and S external lines go to `Riscv::set_irq` as
//! interrupts 3, 7, 11 and 9, and the outputs of its S level IMSIC's guest files to the
//! guest external interrupt lines 64 and up. The CPU reaches the IMSIC registers through
//! `RiscvBoard::aia_ireg_rmw`. The `time` CSR reads the MTIMER, and the Sstc `stimecmp` and
//! `vstimecmp` deadlines are timers on the board clock that call `Riscv::stimer_expired`
//! and `Riscv::vstimer_expired`.
//!
//! Booting is hw/riscv/boot.c: `-bios` (OpenSBI's `fw_dynamic` build by default, `none`
//! for no firmware, or a file; ELF or raw at 0x80000000) is loaded first, then `-kernel`
//! (ELF or raw at the firmware end rounded up to 2 MiB) with `-initrd` and `-append`, then
//! the device tree goes as high in RAM as it fits on a 2 MiB boundary, and the boot ROM
//! gets the reset vector and the `fw_dynamic_info` that tell the firmware where everything
//! is. The harts start at 0x1000 in M mode with `a0` holding `mhartid`, `a1` the device
//! tree and `a2` the dynamic info, and jump to the firmware entry. `-device loader` puts
//! its file (ELF or raw) or value into the first hart's address space and can set a hart's
//! PC, as hw/core/generic-loader.c does. The device tree is built with the same libfdt
//! calls in the same order as QEMU, so with the same CPU configuration and `rng-seed` it
//! matches `-M virt,dumpdtb=` byte for byte.
//!
//! With a firmware in the first flash (EDK2) and a `-bios`, the harts go from the `-bios`
//! firmware to the flash, and `-kernel`, `-initrd` and `-append` go to the flash firmware
//! through fw_cfg (`riscv_setup_firmware_boot()`), the kernel inflated if it is gzip.
//!
//! Unless `acpi=off`, the board puts the ACPI tables of hw/riscv/virt-acpi-build.c in fw_cfg
//! as `etc/acpi/tables`, `etc/acpi/rsdp` and `etc/table-loader` for the firmware to install
//! ([`VirtMachine::acpi_tables`]): DSDT, FADT, MADT, RHCT, SPCR (unless `spcr=off`) and
//! MCFG, which match QEMU's byte for byte for the same CPU configuration.
//!
//! # Using it
//!
//! [`VirtMachine::new`] builds the board and realizes the `-device loader`s. Plug virtio
//! devices with [`VirtMachine::attach_virtio`] (virtio-mmio) or
//! [`VirtMachine::attach_virtio_pci`] (a PCI function on the root bus), then call
//! [`VirtMachine::machine_done`], which finishes the device tree, loads the firmware,
//! kernel, device tree and boot ROM and resets the board. [`VirtMachine::create_vcpus`] then
//! makes the vCPUs on a [`Jit`], wired to the CLINT and the interrupt controllers, and resets
//! them. On a guest
//! reset request ([`VirtMachine::take_request`]) the runner calls
//! [`VirtMachine::system_reset`] and, on each vCPU's thread, [`VirtMachine::reset_cpu`]. The
//! runner also runs the timers of [`VirtMachine::clock`].
//!
//! # Not modelled
//!
//! - The ACLINT SSWI (`aclint=on`) and the RISC-V IOMMU (`iommu-sys=on`); those properties
//!   are taken only with their default values.
//! - SMBIOS (`virt_build_smbios()`).
//! - NUMA and more than one socket: every hart is in socket 0.
//! - uImage kernels, Intel HEX files for `-device loader`, u-boot ramdisks.
//! - KVM: only TCG.
//!
//! # Differences from QEMU
//!
//! - `rng-seed` is not replaced with a new one at every reset
//!   (`qemu_fdt_randomize_seeds()`); the guest sees the same seed after a reboot.
//! - A guest write to `mtime` does not re-arm the Sstc timers of the harts
//!   (`riscv_timer_write_timecmp()` from the MTIMER's time change hook), because the board
//!   cannot reach `stimecmp`. The MTIMER's own `mtimecmp` timers are re-armed.
//! - The Sstc deadlines are host [`Instant`]s converted onto the board clock.
//! - Errors come back as `Err` strings without the `qemu-system-riscv64: ` prefix instead of
//!   exiting. Messages QEMU prints and carries on after go to standard error and are kept in
//!   [`VirtMachine::messages`].
//! - The address spaces of the ROM list ("memory" and "cpu-memory-0") are both views of the
//!   one system memory; their blobs are copied in QEMU's order (all of "memory" first).
//! - With `aia=aplic-imsic`, `aia-guests` above 0 and harts without H, QEMU aborts at the
//!   first reset because the IMSIC drives guest external interrupt lines the hart does not
//!   have. Here those outputs are left unconnected.

mod aia;
mod boot;
mod cpus;
mod dt;

use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::{BuildHasher, Hasher};
use std::sync::{Arc, Mutex, PoisonError, RwLock, Weak};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use ruvm_base::ClockType;
use ruvm_firmware::acpi::BuildTables;
use ruvm_firmware::acpi::gpex::Window;
use ruvm_firmware::acpi::riscv_virt::{
    self, Cmo, Hart, MmuType, RiscvVirtAcpi, Socket, VirtAia as AcpiAia, VirtMemmap,
};
use ruvm_firmware::acpi::table::{APPNAME6, APPNAME8, LOADER_FILE, RSDP_FILE, TABLE_FILE};
use ruvm_hw_char::serial::{Serial, SerialBackend};
use ruvm_hw_core::fw_cfg::{
    DmaMemory, FW_CFG_CTL_SIZE, FW_CFG_DMA_SIZE, FW_CFG_NB_CPUS, FwCfgMachineConfig, FwCfgMem,
    fw_cfg_init_mem_dma,
};
use ruvm_hw_core::timer::TimeSource;
use ruvm_hw_core::{Clock, IrqLine, IrqPin};
use ruvm_hw_intc::riscv_aclint::{
    AclintMtimerConfig, AclintSwiConfig, RISCV_ACLINT_DEFAULT_MTIME, RISCV_ACLINT_DEFAULT_MTIMECMP,
    RISCV_ACLINT_DEFAULT_MTIMER_SIZE, RISCV_ACLINT_DEFAULT_TIMEBASE_FREQ, RISCV_ACLINT_SWI_SIZE,
    RiscvAclintMtimer, RiscvAclintSwi, TYPE_RISCV_ACLINT_MTIMER, TYPE_RISCV_ACLINT_SWI,
};
use ruvm_hw_intc::riscv_aplic::aplic_size;
use ruvm_hw_intc::riscv_imsic::imsic_hart_size;
use ruvm_hw_intc::sifive_plic::{SiFivePlic, SiFivePlicConfig, TYPE_SIFIVE_PLIC};
use ruvm_hw_misc::sifive_test::{SiFiveTest, SiFiveTestRequest, TYPE_SIFIVE_TEST};
use ruvm_hw_pci::regs::PCI_NUM_PINS;
use ruvm_hw_pci::{GpexConfig, GpexHost, GpexWindow, MsiTrigger};
use ruvm_hw_timer::goldfish_rtc::{GOLDFISH_RTC_MMIO_SIZE, GoldfishRtc, TYPE_GOLDFISH_RTC};
use ruvm_hw_virtio::mmio::{VIRTIO_MMIO_FORCE_LEGACY_DEFAULT, VIRTIO_MMIO_REGION_SIZE};
use ruvm_hw_virtio::{VirtioBackend, VirtioDeviceClass, VirtioMmio, VirtioPci, VirtioPciProps};
use ruvm_jit::{Cpu, CpuShared, Jit, Vcpu};
use ruvm_machine_arm::fdt::Fdt;
use ruvm_machine_arm::pflash::{Pflash, PflashBacking, PflashProps};
use ruvm_mem::{
    AccessConstraints, AccessCtx, AccessSize, AddressSpace, Endian, MemResult, MemTxAttrs,
    MemorySystem, MmioOps, RegionId, RegionType,
};
use ruvm_target_riscv::cfg::{VM_SV39, VM_SV48, VM_SV57};
use ruvm_target_riscv::cpu::{
    CpuRiscvState, IRQ_LOCAL_MAX, IRQ_M_EXT, IRQ_M_SOFT, IRQ_M_TIMER, IRQ_S_EXT, RiscvCfg,
};
use ruvm_target_riscv::tcg::{Riscv, SemihostingHost, create_vcpu};
use ruvm_virtio_queue::{GuestMemory, MemoryError};

pub use aia::{Aia, imsic_num_bits};
pub use boot::{
    AS_CPU0, AS_MEMORY, BootInfo, GenericLoader, RISCV64_BIOS_BIN, RamRange, Rom,
    riscv_find_firmware,
};
pub use dt::QEMU_RV64_ISA;

use boot::{KernelFiles, Loader, LoaderReset};
use cpus::{CpuHub, VirtSemihost};

/// `VIRT_MROM`, the boot ROM.
pub const VIRT_MROM: u64 = 0x1000;
/// Its size.
pub const VIRT_MROM_SIZE: u64 = 0xf000;
/// `VIRT_TEST`, the SiFive test device.
pub const VIRT_TEST: u64 = 0x10_0000;
/// Its size.
pub const VIRT_TEST_SIZE: u64 = 0x1000;
/// `VIRT_RTC`, the goldfish RTC.
pub const VIRT_RTC: u64 = 0x10_1000;
/// Its size.
pub const VIRT_RTC_SIZE: u64 = 0x1000;
/// `VIRT_CLINT`.
pub const VIRT_CLINT: u64 = 0x200_0000;
/// Its size.
pub const VIRT_CLINT_SIZE: u64 = 0x1_0000;
/// `VIRT_ACLINT_SSWI`, used with `aclint=on` only.
pub const VIRT_ACLINT_SSWI: u64 = 0x2f0_0000;
/// Its size.
pub const VIRT_ACLINT_SSWI_SIZE: u64 = 0x4000;
/// `VIRT_PCIE_PIO`.
pub const VIRT_PCIE_PIO: u64 = 0x300_0000;
/// Its size.
pub const VIRT_PCIE_PIO_SIZE: u64 = 0x1_0000;
/// `VIRT_IOMMU_SYS`, used with `iommu-sys=on` only.
pub const VIRT_IOMMU_SYS: u64 = 0x301_0000;
/// Its size.
pub const VIRT_IOMMU_SYS_SIZE: u64 = 0x1000;
/// `VIRT_PLATFORM_BUS`.
pub const VIRT_PLATFORM_BUS: u64 = 0x400_0000;
/// Its size.
pub const VIRT_PLATFORM_BUS_SIZE: u64 = 0x200_0000;
/// `VIRT_PLIC`.
pub const VIRT_PLIC: u64 = 0xc00_0000;
/// Its size, `VIRT_PLIC_SIZE(VIRT_CPUS_MAX * 2)`.
pub const VIRT_PLIC_SIZE: u64 = 0x60_0000;
/// `VIRT_APLIC_M`, the M level APLIC domain with `aia=aplic` or `aia=aplic-imsic`.
pub const VIRT_APLIC_M: u64 = 0xc00_0000;
/// `VIRT_APLIC_S`, the S level APLIC domain.
pub const VIRT_APLIC_S: u64 = 0xd00_0000;
/// Their size, `APLIC_SIZE(VIRT_CPUS_MAX)`.
pub const VIRT_APLIC_SIZE: u64 = aplic_size(VIRT_CPUS_MAX as u32);
/// `VIRT_UART0`.
pub const VIRT_UART0: u64 = 0x1000_0000;
/// Its size.
pub const VIRT_UART0_SIZE: u64 = 0x100;
/// `VIRT_VIRTIO`, the first virtio-mmio transport.
pub const VIRT_VIRTIO: u64 = 0x1000_1000;
/// The stride of the transports.
pub const VIRT_VIRTIO_SIZE: u64 = 0x1000;
/// `VIRT_FW_CFG`.
pub const VIRT_FW_CFG: u64 = 0x1010_0000;
/// Its size.
pub const VIRT_FW_CFG_SIZE: u64 = 0x18;
/// `VIRT_FLASH`, the two flashes.
pub const VIRT_FLASH: u64 = 0x2000_0000;
/// Its size, half of it for each flash.
pub const VIRT_FLASH_SIZE: u64 = 0x400_0000;
/// `VIRT_IMSIC_M`, the M level IMSICs with `aia=aplic-imsic`.
pub const VIRT_IMSIC_M: u64 = 0x2400_0000;
/// `VIRT_IMSIC_S`, the S level IMSICs.
pub const VIRT_IMSIC_S: u64 = 0x2800_0000;
/// The size of each, `VIRT_IMSIC_MAX_SIZE`: `VIRT_SOCKETS_MAX` groups of 16 MiB.
pub const VIRT_IMSIC_MAX_SIZE: u64 = 8 << 24;
/// `VIRT_PCIE_ECAM`.
pub const VIRT_PCIE_ECAM: u64 = 0x3000_0000;
/// Its size.
pub const VIRT_PCIE_ECAM_SIZE: u64 = 0x1000_0000;
/// `VIRT_PCIE_MMIO`.
pub const VIRT_PCIE_MMIO: u64 = 0x4000_0000;
/// Its size.
pub const VIRT_PCIE_MMIO_SIZE: u64 = 0x4000_0000;
/// `VIRT_DRAM`, the base of RAM.
pub const VIRT_DRAM: u64 = 0x8000_0000;

/// `UART0_IRQ`.
pub const UART0_IRQ: u32 = 10;
/// `RTC_IRQ`.
pub const RTC_IRQ: u32 = 11;
/// `VIRTIO_IRQ`, the source of the first transport.
pub const VIRTIO_IRQ: u32 = 1;
/// `VIRTIO_COUNT`.
pub const VIRTIO_COUNT: usize = 8;
/// `PCIE_IRQ`.
pub const PCIE_IRQ: u32 = 0x20;
/// `VIRT_PLATFORM_BUS_IRQ`.
pub const VIRT_PLATFORM_BUS_IRQ: u32 = 64;
/// `VIRT_IRQCHIP_NUM_SOURCES`.
pub const VIRT_IRQCHIP_NUM_SOURCES: u32 = 96;
/// `VIRT_IRQCHIP_NUM_PRIO_BITS`.
pub const VIRT_IRQCHIP_NUM_PRIO_BITS: u32 = 3;
/// `VIRT_IRQCHIP_NUM_MSIS`, the identities of each IMSIC file.
pub const VIRT_IRQCHIP_NUM_MSIS: u32 = 255;
/// `VIRT_IRQCHIP_MAX_GUESTS`, the most `aia-guests` can be.
pub const VIRT_IRQCHIP_MAX_GUESTS: u32 = 7;
/// `VIRT_PLIC_PRIORITY_BASE`.
const VIRT_PLIC_PRIORITY_BASE: u32 = 0x00;
/// `VIRT_PLIC_PENDING_BASE`.
const VIRT_PLIC_PENDING_BASE: u32 = 0x1000;
/// `VIRT_PLIC_ENABLE_BASE`.
const VIRT_PLIC_ENABLE_BASE: u32 = 0x2000;
/// `VIRT_PLIC_ENABLE_STRIDE`.
const VIRT_PLIC_ENABLE_STRIDE: u32 = 0x80;
/// `VIRT_PLIC_CONTEXT_BASE`.
const VIRT_PLIC_CONTEXT_BASE: u32 = 0x20_0000;
/// `VIRT_PLIC_CONTEXT_STRIDE`.
const VIRT_PLIC_CONTEXT_STRIDE: u32 = 0x1000;

/// `VIRT_CPUS_MAX`.
pub const VIRT_CPUS_MAX: usize = 512;
/// The default RAM size of the machine class.
pub const VIRT_DEFAULT_RAM_SIZE: u64 = 128 << 20;
/// The RAM region name, `default_ram_id`.
pub const VIRT_RAM_ID: &str = "riscv_virt_board.ram";
/// The reset vector of the harts, `DEFAULT_RSTVEC`.
pub const DEFAULT_RSTVEC: u64 = 0x1000;
/// The base clock of the UART, as `serial_mm_init()` is given it.
const UART_BAUDBASE: u32 = 399_193;
/// The size of the UART registers, `8 << regshift`.
const UART_REGION_SIZE: u64 = 8;

/// `virt_high_pcie_memmap.base` for RV64: above RAM, aligned to its 16 GiB size.
pub fn high_pcie_base(ram_size: u64) -> u64 {
    (VIRT_DRAM + ram_size).next_multiple_of(dt::VIRT64_HIGH_PCIE_MMIO_SIZE)
}

/// What the board is built from: the `-smp`, `-m`, `-kernel`, `-initrd`, `-append`, `-dtb`,
/// `-bios`, `-serial`, `-semihosting`, `-cpu` and `-device loader` options.
#[derive(Clone)]
pub struct VirtConfig {
    /// The number of harts.
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
    /// The firmware file `-bios` resolves to with [`riscv_find_firmware`], or `None` for
    /// `-bios none`.
    pub firmware: Option<String>,
    /// The drives of the two flashes, `pflash0` and `pflash1`.
    pub pflash: [PflashBacking; 2],
    /// The chardev of the UART, `serial_hd(0)`.
    pub serial: Option<Arc<dyn SerialBackend>>,
    /// Semihosting, when enabled.
    pub semihosting: Option<Arc<dyn SemihostingHost>>,
    /// `-semihosting-config userspace=on`.
    pub semihosting_userspace: bool,
    /// The configuration of the harts, `-cpu` after `riscv_cpu_finalize_features()`. The
    /// default is QEMU's default CPU, `rv64`.
    pub cpu: RiscvCfg,
    /// The `-device loader` devices, in command line order.
    pub loaders: Vec<GenericLoader>,
    /// The clock the ACLINT and the Sstc timers run on. The default follows the host's
    /// monotonic time.
    pub clock: Option<Arc<Clock>>,
    /// The clock of the RTC, `rtc_clock`. The default is the host wall clock.
    pub rtc_clock: Option<Arc<Clock>>,
    /// The machine options fw_cfg exposes.
    pub fw_cfg: FwCfgMachineConfig,
    /// The `rng-seed` of `/chosen`. The default is random, as `qemu_guest_getrandom()`.
    pub rng_seed: Option<[u8; 32]>,
    /// `aia`: the interrupt controllers.
    pub aia: VirtAia,
    /// `aia-guests`: the guest files of each S level IMSIC, at most
    /// [`VIRT_IRQCHIP_MAX_GUESTS`]. Used with [`VirtAia::AplicImsic`] only.
    pub aia_guests: u32,
    /// `acpi`: false for `acpi=off`, true for `on` and `auto` (the default).
    pub acpi: bool,
    /// The machine's `spcr` property: whether the ACPI tables include an SPCR.
    pub spcr: bool,
}

/// `RISCVVirtAIAType`, the `aia` property.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VirtAia {
    /// `none`: the SiFive PLIC.
    #[default]
    None,
    /// `aplic`: APLIC domains in direct mode.
    Aplic,
    /// `aplic-imsic`: APLIC domains in MSI mode and IMSICs.
    AplicImsic,
}

impl VirtAia {
    /// The property value, as `virt_get_aia()` gives it.
    pub fn as_str(self) -> &'static str {
        match self {
            VirtAia::None => "none",
            VirtAia::Aplic => "aplic",
            VirtAia::AplicImsic => "aplic-imsic",
        }
    }
}

impl fmt::Debug for VirtConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VirtConfig")
            .field("smp", &self.smp)
            .field("ram_size", &self.ram_size)
            .field("kernel", &self.kernel)
            .field("initrd", &self.initrd)
            .field("append", &self.append)
            .field("dtb", &self.dtb)
            .field("firmware", &self.firmware)
            .field("pflash", &self.pflash)
            .field("serial", &self.serial.is_some())
            .field("semihosting", &self.semihosting.is_some())
            .field("semihosting_userspace", &self.semihosting_userspace)
            .field("cpu", &self.cpu)
            .field("loaders", &self.loaders)
            .field("aia", &self.aia)
            .field("aia_guests", &self.aia_guests)
            .field("acpi", &self.acpi)
            .field("spcr", &self.spcr)
            .finish_non_exhaustive()
    }
}

impl Default for VirtConfig {
    /// One hart, the default RAM size, no firmware and nothing to load.
    fn default() -> VirtConfig {
        VirtConfig {
            smp: 1,
            ram_size: VIRT_DEFAULT_RAM_SIZE,
            kernel: None,
            initrd: None,
            append: None,
            dtb: None,
            firmware: None,
            pflash: [PflashBacking::None, PflashBacking::None],
            serial: None,
            semihosting: None,
            semihosting_userspace: false,
            cpu: RiscvCfg::default(),
            loaders: Vec::new(),
            clock: None,
            rtc_clock: None,
            fw_cfg: FwCfgMachineConfig::default(),
            rng_seed: None,
            aia: VirtAia::None,
            aia_guests: 0,
            acpi: true,
            spcr: true,
        }
    }
}

/// What the guest asked the machine to do through the SiFive test device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VirtRequest {
    /// `FINISHER_PASS`: shut down with this exit code.
    Shutdown(u16),
    /// `FINISHER_FAIL`: a guest panic shutdown with this exit code.
    Panic(u16),
    /// `FINISHER_RESET`.
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
    irq: IrqLine,
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

/// `serial_mm_ops` with `regshift` 0, little-endian: the 16550 registers one byte apart,
/// any access of 1 to 8 bytes reading or writing one register.
struct SerialMm(Arc<Serial>);

impl MmioOps for SerialMm {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.0.ioport_read(offset)))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.0.ioport_write(offset, value as u8);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 8)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 8)
    }

    fn endianness(&self) -> Endian {
        Endian::Little
    }
}

/// `qemu_guest_getrandom_nofail()` for the `rng-seed`.
fn random_seed() -> [u8; 32] {
    let mut seed = [0u8; 32];
    for (i, chunk) in seed.chunks_mut(8).enumerate() {
        let mut h = RandomState::new().build_hasher();
        h.write_usize(i);
        h.write_u128(SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos());
        chunk.copy_from_slice(&h.finish().to_le_bytes());
    }
    seed
}

/// The interrupt controller of the devices, `s->irqchip[0]`.
enum Irqchip {
    Plic(Arc<SiFivePlic>),
    Aia(Aia),
}

impl Irqchip {
    /// Input `n`, an interrupt source.
    fn input(&self, n: u32) -> IrqLine {
        match self {
            Irqchip::Plic(plic) => plic.input(n),
            Irqchip::Aia(aia) => aia.input(n),
        }
    }

    fn reset(&self) {
        match self {
            Irqchip::Plic(plic) => plic.reset(),
            Irqchip::Aia(aia) => aia.reset(),
        }
    }
}

/// The virt board.
pub struct VirtMachine {
    smp: usize,
    ram_size: u64,
    kernel: Option<String>,
    initrd: Option<String>,
    append: Option<String>,
    dtb_filename: Option<String>,
    firmware: Option<String>,
    pflash0_given: bool,
    mem: Arc<MemorySystem>,
    system: RegionId,
    memory_as: Arc<AddressSpace>,
    riscv: Arc<Riscv>,
    hub: Arc<CpuHub>,
    swi: Arc<RiscvAclintSwi>,
    mtimer: Arc<RiscvAclintMtimer>,
    irqchip: Irqchip,
    aia: VirtAia,
    aia_guests: u32,
    acpi: bool,
    spcr: bool,
    test: Arc<SiFiveTest>,
    uart: Arc<Serial>,
    rtc: Arc<GoldfishRtc>,
    flash: [Arc<Pflash>; 2],
    virtio: Vec<Arc<VirtioSlot>>,
    gpex: GpexHost,
    pci_devices: Mutex<Vec<VirtioPci>>,
    fw_cfg: FwCfgMem,
    fdt: Fdt,
    loader: Loader,
    loaders: Vec<LoaderReset>,
    info: BootInfo,
    heap: Arc<Mutex<(u64, u64)>>,
    ram: Vec<RamRange>,
    clock: Arc<Clock>,
    done: bool,
}

impl fmt::Debug for VirtMachine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VirtMachine")
            .field("smp", &self.smp)
            .field("ram_size", &self.ram_size)
            .field("boot", &self.info)
            .field("roms", &self.loader.roms)
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}

/// Map `ops` as the region `name` of `size` bytes at `addr`.
fn map_io(
    mem: &MemorySystem,
    system: RegionId,
    name: &str,
    addr: u64,
    size: u64,
    ops: Arc<dyn MmioOps>,
) -> Result<(), String> {
    let r = mem.new_io(name, size.into(), ops).map_err(err)?;
    mem.add_subregion(system, addr, r).map_err(err)
}

/// `gpex_pcie_init()`: the generic PCIe host bridge with its ECAM, the low and high MMIO
/// windows and the I/O port window, its INTx lines on sources 32 to 35 of the irqchip.
fn gpex_pcie_init(
    mem: &Arc<MemorySystem>,
    system: RegionId,
    memory_as: &Arc<AddressSpace>,
    irqchip: &Irqchip,
    ram_size: u64,
) -> Result<GpexHost, String> {
    let high = GpexWindow { base: high_pcie_base(ram_size), size: dt::VIRT64_HIGH_PCIE_MMIO_SIZE };
    let config = GpexConfig {
        ecam: GpexWindow { base: VIRT_PCIE_ECAM, size: VIRT_PCIE_ECAM_SIZE },
        mmio32: GpexWindow { base: VIRT_PCIE_MMIO, size: VIRT_PCIE_MMIO_SIZE },
        mmio64: high,
        pio: GpexWindow { base: VIRT_PCIE_PIO, size: VIRT_PCIE_PIO_SIZE },
        ..GpexConfig::default()
    };
    let gpex = GpexHost::new(Arc::clone(mem), system, config).map_err(err)?;
    let aliases = [
        ("pcie-ecam", gpex.ecam(), 0, VIRT_PCIE_ECAM, VIRT_PCIE_ECAM_SIZE),
        ("pcie-mmio", gpex.mmio_window(), VIRT_PCIE_MMIO, VIRT_PCIE_MMIO, VIRT_PCIE_MMIO_SIZE),
        ("pcie-mmio-high", gpex.mmio_window(), high.base, high.base, high.size),
    ];
    for (name, target, offset, addr, size) in aliases {
        let alias = mem.new_alias(name, target, offset, size.into()).map_err(err)?;
        mem.add_subregion(system, addr, alias).map_err(err)?;
    }
    mem.add_subregion(system, VIRT_PCIE_PIO, gpex.ioport_window()).map_err(err)?;
    for i in 0..PCI_NUM_PINS {
        let irq = PCIE_IRQ + i as u32;
        if let Some(pin) = gpex.irq(i) {
            pin.connect(irqchip.input(irq));
        }
        gpex.set_irq_num(i, irq as i32).map_err(err)?;
    }
    // The PLIC and the AIA devices set msi_nonbroken in QEMU, so functions get their MSI-X
    // capability. A message is a plain 32 bit store into system memory, msi_send_message(),
    // which goes nowhere unless an IMSIC is mapped at its address.
    let msi: MsiTrigger = Arc::new(msi_store(memory_as));
    gpex.bus().set_msi_handler(Some(msi));
    Ok(gpex)
}

/// A message signaled interrupt as `address_space_stl_le()` into `memory_as` sends it,
/// through a weak reference for the same reason as [`WeakDma`].
fn msi_store(memory_as: &Arc<AddressSpace>) -> impl Fn(u64, u32) + Send + Sync + 'static {
    let weak = Arc::downgrade(memory_as);
    move |address, data| {
        if let Some(a) = weak.upgrade() {
            let _ = a.store(address, 4, data.into(), Endian::Little, MemTxAttrs::UNSPECIFIED);
        }
    }
}

impl VirtMachine {
    /// `virt_machine_init()` and the realize of the `-device loader`s.
    pub fn new(cfg: VirtConfig) -> Result<VirtMachine, String> {
        let smp = cfg.smp;
        let ram_size = cfg.ram_size;
        if smp == 0 {
            return Err(
                "Invalid SMP CPUs 0. The min CPUs supported by machine 'virt' is 1".to_string()
            );
        }
        if smp > VIRT_CPUS_MAX {
            return Err(format!(
                "Invalid SMP CPUs {smp}. The max CPUs supported by machine 'virt' is \
                 {VIRT_CPUS_MAX}"
            ));
        }
        let harts = smp as u32;

        let mem = Arc::new(MemorySystem::new());
        let system = mem.new_container("system", 1 << 64).map_err(err)?;
        let memory_as = mem.address_space_init(system, "memory").map_err(err)?;
        let clock = cfg.clock.unwrap_or_else(|| {
            Clock::new(ClockType::Virtual, TimeSource::Monotonic(Instant::now()))
        });

        // The SiFive CLINT: the ACLINT MSWI, then the MTIMER after it.
        let swi = RiscvAclintSwi::new(AclintSwiConfig {
            hartid_base: 0,
            num_harts: harts,
            sswi: false,
            absent_harts: Vec::new(),
        });
        map_io(&mem, system, TYPE_RISCV_ACLINT_SWI, VIRT_CLINT, swi.mmio_size(), swi.clone())?;
        let mtimer = RiscvAclintMtimer::new(
            clock.clone(),
            AclintMtimerConfig {
                hartid_base: 0,
                num_harts: harts,
                timecmp_base: RISCV_ACLINT_DEFAULT_MTIMECMP,
                time_base: RISCV_ACLINT_DEFAULT_MTIME,
                aperture_size: RISCV_ACLINT_DEFAULT_MTIMER_SIZE,
                timebase_freq: RISCV_ACLINT_DEFAULT_TIMEBASE_FREQ,
                absent_harts: Vec::new(),
            },
        );
        let mtimer_base = VIRT_CLINT + u64::from(RISCV_ACLINT_SWI_SIZE);
        let size = mtimer.mmio_size();
        map_io(&mem, system, TYPE_RISCV_ACLINT_MTIMER, mtimer_base, size, mtimer.clone())?;

        // The per-socket interrupt controller.
        let aia = cfg.aia;
        let aia_guests = cfg.aia_guests;
        let mut cpu = cfg.cpu;
        let irqchip = if aia == VirtAia::None {
            // virt_create_plic(): an M and an S context per hart.
            let plic = SiFivePlic::new(SiFivePlicConfig {
                hart_config: vec!["MS"; smp].join(","),
                hartid_base: 0,
                num_sources: VIRT_IRQCHIP_NUM_SOURCES,
                num_priorities: (1 << VIRT_IRQCHIP_NUM_PRIO_BITS) - 1,
                priority_base: VIRT_PLIC_PRIORITY_BASE,
                pending_base: VIRT_PLIC_PENDING_BASE,
                enable_base: VIRT_PLIC_ENABLE_BASE,
                enable_stride: VIRT_PLIC_ENABLE_STRIDE,
                context_base: VIRT_PLIC_CONTEXT_BASE,
                context_stride: VIRT_PLIC_CONTEXT_STRIDE,
                aperture_size: VIRT_PLIC_SIZE as u32,
            })
            .map_err(err)?;
            map_io(&mem, system, TYPE_SIFIVE_PLIC, VIRT_PLIC, plic.mmio_size(), plic.clone())?;
            Irqchip::Plic(plic)
        } else {
            if aia_guests > VIRT_IRQCHIP_MAX_GUESTS {
                return Err("Invalid number of AIA IMSIC guests".to_string());
            }
            let msimode = aia == VirtAia::AplicImsic;
            let params = aia::AiaParams {
                msimode,
                aia_guests,
                m_imsic_stride: imsic_hart_size(0),
                num_sources: VIRT_IRQCHIP_NUM_SOURCES,
                aplic_m: aia::MemMapEntry { base: VIRT_APLIC_M, size: VIRT_APLIC_SIZE },
                aplic_s: aia::MemMapEntry { base: VIRT_APLIC_S, size: VIRT_APLIC_SIZE },
                imsic_m: aia::MemMapEntry { base: VIRT_IMSIC_M, size: VIRT_IMSIC_MAX_SIZE },
                imsic_s: aia::MemMapEntry { base: VIRT_IMSIC_S, size: VIRT_IMSIC_MAX_SIZE },
                socket: 0,
                base_hartid: 0,
                hart_count: harts,
                num_msis: VIRT_IRQCHIP_NUM_MSIS,
                num_prio_bits: VIRT_IRQCHIP_NUM_PRIO_BITS,
            };
            let a = aia::riscv_create_aia(&mem, system, &params)?;
            for aplic in [a.aplic_m(), a.aplic_s()] {
                aplic.set_msi_sink(Box::new(msi_store(&memory_as)));
            }
            if msimode {
                // riscv_imsic_realize() forces the AIA extensions on the harts.
                cpu.ext_smaia = true;
                cpu.ext_ssaia = true;
            }
            Irqchip::Aia(a)
        };

        // The harts.
        let hub = Arc::new(CpuHub::new(mtimer.clone(), smp, clock.clone()));
        if let Irqchip::Aia(a) = &irqchip {
            hub.set_imsics(a.imsic_m().to_vec(), a.imsic_s().to_vec());
        }
        let heap = Arc::new(Mutex::new((0, 0)));
        let mut riscv = Riscv::new().with_cfg(cpu);
        if aia == VirtAia::AplicImsic {
            // riscv_cpu_set_geilen() from the realize of the S level IMSICs.
            riscv = riscv.with_geilen(aia_guests);
        }
        if let Some(host) = cfg.semihosting {
            let semi = VirtSemihost {
                host,
                heap: heap.clone(),
                cmdline: cpus::semihosting_cmdline(cfg.kernel.as_deref(), cfg.append.as_deref()),
            };
            riscv = riscv.with_semihosting(Arc::new(semi), cfg.semihosting_userspace);
        }
        let riscv = Arc::new(riscv);
        riscv.set_board(hub.clone());
        hub.set_riscv(&riscv);

        // The RAM and the boot ROM.
        let ram = mem.new_ram(VIRT_RAM_ID, ram_size).map_err(err)?;
        mem.add_subregion(system, VIRT_DRAM, ram).map_err(err)?;
        let mrom = mem.new_rom("riscv_virt_board.mrom", VIRT_MROM_SIZE).map_err(err)?;
        mem.add_subregion(system, VIRT_MROM, mrom).map_err(err)?;

        // create_fw_cfg().
        let dma: Arc<dyn DmaMemory> = Arc::new(WeakDma(Arc::downgrade(&memory_as)));
        let fw_cfg = fw_cfg_init_mem_dma(VIRT_FW_CFG, dma, &cfg.fw_cfg).map_err(err)?;
        fw_cfg.state().add_i16(FW_CFG_NB_CPUS, smp as u16);
        let (ctl, data, dma_addr) = fw_cfg.addrs();
        map_io(&mem, system, "fwcfg.ctl", ctl, FW_CFG_CTL_SIZE, fw_cfg.ctl_ops().clone())?;
        let data_size = fw_cfg.data_ops().region_size();
        map_io(&mem, system, "fwcfg.data", data, data_size, fw_cfg.data_ops().clone())?;
        if let Some(d) = fw_cfg.dma_ops() {
            map_io(&mem, system, "fwcfg.dma", dma_addr, FW_CFG_DMA_SIZE, d.clone())?;
        }

        // sifive_test_create().
        let test = {
            let hub = Arc::downgrade(&hub);
            SiFiveTest::new(move |req| {
                let Some(hub) = hub.upgrade() else { return };
                hub.request(match req {
                    SiFiveTestRequest::Pass(code) => VirtRequest::Shutdown(code),
                    SiFiveTestRequest::Fail(code) => VirtRequest::Panic(code),
                    SiFiveTestRequest::Reset => VirtRequest::Reset,
                });
            })
        };
        map_io(&mem, system, TYPE_SIFIVE_TEST, VIRT_TEST, VIRT_TEST_SIZE, test.clone())?;

        // The virtio-mmio transports.
        let mut virtio = Vec::with_capacity(VIRTIO_COUNT);
        for i in 0..VIRTIO_COUNT {
            let t = VirtioMmio::new(None, VIRTIO_MMIO_FORCE_LEGACY_DEFAULT).map_err(err)?;
            let slot = Arc::new(VirtioSlot {
                irq: irqchip.input(VIRTIO_IRQ + i as u32),
                transport: RwLock::new(Arc::new(t)),
                plugged: RwLock::new(false),
            });
            let base = VIRT_VIRTIO + i as u64 * VIRT_VIRTIO_SIZE;
            map_io(&mem, system, "virtio-mmio", base, VIRTIO_MMIO_REGION_SIZE, slot.clone())?;
            virtio.push(slot);
        }

        let gpex = gpex_pcie_init(&mem, system, &memory_as, &irqchip, ram_size)?;

        // create_platform_bus(): the window, with nothing on it yet.
        let pbus = mem.new_container("platform bus", VIRT_PLATFORM_BUS_SIZE.into()).map_err(err)?;
        mem.add_subregion(system, VIRT_PLATFORM_BUS, pbus).map_err(err)?;

        // serial_mm_init().
        let uart = Serial::new(clock.clone(), UART_BAUDBASE, cfg.serial);
        uart.irq().connect(irqchip.input(UART0_IRQ));
        let ops = Arc::new(SerialMm(uart.clone()));
        map_io(&mem, system, "serial", VIRT_UART0, UART_REGION_SIZE, ops)?;

        // The goldfish RTC.
        let rtc_clock =
            cfg.rtc_clock.unwrap_or_else(|| Clock::new(ClockType::Host, TimeSource::Wall));
        // qemu_ref_timedate(): the host clock reads the date itself; the others count from
        // the date the machine started.
        let rtc_date =
            if rtc_clock.kind() == ClockType::Host { UNIX_EPOCH } else { SystemTime::now() };
        let rtc = GoldfishRtc::new(rtc_clock, rtc_date, false);
        rtc.irq().connect(irqchip.input(RTC_IRQ));
        map_io(&mem, system, TYPE_GOLDFISH_RTC, VIRT_RTC, GOLDFISH_RTC_MMIO_SIZE, rtc.clone())?;

        // virt_flash_create() and virt_flash_map().
        let [pflash0, pflash1] = cfg.pflash;
        let pflash0_given = pflash0 != PflashBacking::None;
        let half = VIRT_FLASH_SIZE / 2;
        let flash0 = Pflash::new(&mem, "virt.flash0", PflashProps::virt_flash(half), pflash0)?;
        let flash1 = Pflash::new(&mem, "virt.flash1", PflashProps::virt_flash(half), pflash1)?;
        mem.add_subregion(system, VIRT_FLASH, flash0.region()).map_err(err)?;
        mem.add_subregion(system, VIRT_FLASH + half, flash1.region()).map_err(err)?;

        // Load or create the device tree.
        let mut loader = Loader::default();
        let fdt = match &cfg.dtb {
            Some(path) => match boot::load_device_tree(&mut loader, path) {
                Some(f) => f,
                None => return Err("load_device_tree() failed".to_string()),
            },
            None => {
                let mut fdt = Fdt::new();
                let seed = cfg.rng_seed.unwrap_or_else(random_seed);
                dt::create_fdt(&mut fdt, &seed, riscv.cfg().pmu_mask)?;
                fdt
            }
        };

        // The -device loader devices, realized after the board.
        let mut loaders = Vec::with_capacity(cfg.loaders.len());
        for l in &cfg.loaders {
            loaders.push(boot::generic_loader_realize(&mut loader, l, smp, ram_size)?);
        }

        Ok(VirtMachine {
            smp,
            ram_size,
            kernel: cfg.kernel,
            initrd: cfg.initrd,
            append: cfg.append,
            dtb_filename: cfg.dtb,
            firmware: cfg.firmware,
            pflash0_given,
            mem,
            system,
            memory_as,
            riscv,
            hub,
            swi,
            mtimer,
            irqchip,
            aia,
            aia_guests,
            acpi: cfg.acpi,
            spcr: cfg.spcr,
            test,
            uart,
            rtc,
            flash: [flash0, flash1],
            virtio,
            gpex,
            pci_devices: Mutex::new(Vec::new()),
            fw_cfg,
            fdt,
            loader,
            loaders,
            info: BootInfo::default(),
            heap,
            ram: Vec::new(),
            clock,
            done: false,
        })
    }

    /// Plug a virtio device into the highest free transport, as `-device virtio-*-device`
    /// does. Returns the transport index.
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
        t.irq().connect(slot.irq.clone());
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

    /// Connect the chardev of the UART.
    pub fn set_serial_backend(&self, backend: Option<Arc<dyn SerialBackend>>) {
        self.uart.set_backend(backend);
    }

    /// `virt_machine_done()` and the rest of `qdev_machine_creation_done()`: finish the
    /// device tree, load the firmware, kernel, device tree and boot ROM, check the ROMs,
    /// work out the semihosting heap and reset the board.
    pub fn machine_done(&mut self) -> Result<(), String> {
        if self.done {
            return Ok(());
        }
        // A user provided dtb must include everything; ours needs to be finalized.
        if self.dtb_filename.is_none() {
            let riscv = self.riscv.clone();
            let args = dt::FinalizeArgs {
                smp: self.smp,
                ram_size: self.ram_size,
                cpu: riscv.cfg(),
                aia: self.aia,
                aia_guests: self.aia_guests,
            };
            dt::finalize_fdt(&mut self.fdt, args)?;
        }

        let mut info = BootInfo::default();
        // riscv_find_and_load_firmware().
        let mut start_addr = VIRT_DRAM;
        let mut firmware_end = start_addr;
        if let Some(fw) = self.firmware.clone() {
            firmware_end = boot::riscv_load_firmware(
                &mut self.loader,
                &self.memory_as,
                &fw,
                &mut start_addr,
                self.ram_size,
            )?;
        }

        let mut kernel_entry = 0;
        if self.pflash0_given {
            if self.firmware.is_none() {
                // Pflash was supplied but bios is none: jump to the base of the flash.
                start_addr = VIRT_FLASH;
            } else {
                // The flash holds an S-mode payload, which gets the kernel through fw_cfg.
                if let Some(kernel) = self.kernel.clone() {
                    let files = KernelFiles {
                        kernel: &kernel,
                        initrd: self.initrd.as_deref(),
                        cmdline: self.append.as_deref(),
                    };
                    let fwc = self.fw_cfg.state().clone();
                    boot::riscv_setup_firmware_boot(&mut self.loader, &fwc, files)?;
                }
                kernel_entry = VIRT_FLASH;
            }
        }

        if let Some(kernel) = self.kernel.clone() {
            if kernel_entry == 0 {
                let kernel_start = boot::riscv_calc_kernel_start_addr(firmware_end);
                let files = KernelFiles {
                    kernel: &kernel,
                    initrd: self.initrd.as_deref(),
                    cmdline: self.append.as_deref(),
                };
                boot::riscv_load_kernel(
                    &mut self.loader,
                    &mut info,
                    Some(&mut self.fdt),
                    files,
                    kernel_start,
                    self.ram_size,
                )?;
                kernel_entry = info.image_low_addr;
            }
        }

        // riscv_compute_fdt_addr() packs the tree first.
        self.fdt = dt::fdt_pack(&self.fdt)?;
        let fdtsize = self.fdt.as_bytes().len() as u64;
        let fdt_addr = boot::riscv_compute_fdt_addr(VIRT_DRAM, self.ram_size, fdtsize, &info)?;
        // riscv_load_fdt().
        self.loader.add(Rom::blob("fdt", fdt_addr, self.fdt.as_bytes().to_vec()));

        // riscv_setup_rom_reset_vec().
        let reset_vec = boot::reset_vec(start_addr, fdt_addr);
        self.loader.add(Rom::blob("mrom.reset", VIRT_MROM, reset_vec));
        let finfo = boot::firmware_info(VIRT_MROM_SIZE, boot::RESET_VEC_SIZE, kernel_entry)?;
        self.loader.add(Rom::blob("mrom.finfo", VIRT_MROM + boot::RESET_VEC_SIZE, finfo));

        info.start_addr = start_addr;
        info.firmware_end = firmware_end;
        info.kernel_entry = kernel_entry;
        info.fdt_addr = fdt_addr;
        self.info = info;

        // virt_acpi_setup(). The tables never change, so unlike QEMU they are not rebuilt
        // when the firmware first reads them.
        if self.acpi {
            let tables = self.acpi_tables();
            let fwc = self.fw_cfg.state();
            fwc.add_file(TABLE_FILE, tables.table_data).map_err(err)?;
            fwc.add_file(LOADER_FILE, tables.linker.cmd_blob().to_vec()).map_err(err)?;
            fwc.add_file(RSDP_FILE, tables.rsdp).map_err(err)?;
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

    /// `virt_acpi_build()`: the ACPI tables of the board as it is now.
    pub fn acpi_tables(&self) -> BuildTables {
        let cfg = self.riscv.cfg();
        let w = |base, size| Window { base, size };
        let num_harts = self.smp as u32;
        let acpi = RiscvVirtAcpi {
            oem_id: APPNAME6.to_string(),
            oem_table_id: APPNAME8.to_string(),
            memmap: VirtMemmap {
                plic: w(VIRT_PLIC, VIRT_PLIC_SIZE),
                aplic_s: w(VIRT_APLIC_S, VIRT_APLIC_SIZE),
                imsic_s: w(VIRT_IMSIC_S, VIRT_IMSIC_MAX_SIZE),
                uart0: w(VIRT_UART0, VIRT_UART0_SIZE),
                virtio: w(VIRT_VIRTIO, VIRT_VIRTIO_SIZE),
                fw_cfg: w(VIRT_FW_CFG, VIRT_FW_CFG_SIZE),
                pcie_ecam: w(VIRT_PCIE_ECAM, VIRT_PCIE_ECAM_SIZE),
                pcie_mmio: w(VIRT_PCIE_MMIO, VIRT_PCIE_MMIO_SIZE),
                pcie_pio: w(VIRT_PCIE_PIO, VIRT_PCIE_PIO_SIZE),
                pcie_mmio_high: w(high_pcie_base(self.ram_size), dt::VIRT64_HIGH_PCIE_MMIO_SIZE),
                dram: w(VIRT_DRAM, self.ram_size),
            },
            harts: (0..self.smp as u64).map(|hart_id| Hart { hart_id, socket: 0 }).collect(),
            smp_cpus: num_harts,
            sockets: vec![Socket { first_hartid: 0, num_harts }],
            aia: match self.aia {
                VirtAia::None => AcpiAia::None,
                VirtAia::Aplic => AcpiAia::Aplic,
                VirtAia::AplicImsic => AcpiAia::AplicImsic,
            },
            aia_guests: self.aia_guests,
            num_sources: VIRT_IRQCHIP_NUM_SOURCES,
            num_msis: VIRT_IRQCHIP_NUM_MSIS,
            uart_irq: UART0_IRQ,
            virtio_irq: VIRTIO_IRQ,
            virtio_count: VIRTIO_COUNT as u32,
            pcie_irq: PCIE_IRQ,
            isa: cfg.isa_string(),
            cmo: (cfg.ext_zicbom || cfg.ext_zicboz).then_some(Cmo {
                cbom_blocksize: cfg.cbom_blocksize,
                cboz_blocksize: cfg.cboz_blocksize,
            }),
            mmu: match cfg.max_satp_mode {
                VM_SV57 => Some(MmuType::Sv57),
                VM_SV48 => Some(MmuType::Sv48),
                VM_SV39 => Some(MmuType::Sv39),
                _ => None,
            },
            timebase_freq: RISCV_ACLINT_DEFAULT_TIMEBASE_FREQ.into(),
            spcr: self.spcr,
            numa: Vec::new(),
        };
        riscv_virt::build(&acpi)
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

    /// `qemu_system_reset()` for the devices, the `-device loader` values and the ROMs, in
    /// that order. Each vCPU is reset separately with [`VirtMachine::reset_cpu`] on its own
    /// thread.
    pub fn system_reset(&mut self) -> Result<(), String> {
        self.swi.reset();
        self.mtimer.reset();
        self.irqchip.reset();
        self.uart.reset();
        self.rtc.reset();
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
        // generic_loader_reset(): the values go through the CPU's address space.
        for l in self.loaders.iter().filter(|l| !l.data.is_empty()) {
            let _ = self.memory_as.write(l.addr, MemTxAttrs::UNSPECIFIED, &l.data);
        }
        boot::rom_reset(&self.loader.roms, &self.ram);
        Ok(())
    }

    /// Make the vCPUs on `jit`, wire their interrupt lines to the CLINT and the interrupt
    /// controllers and reset
    /// them. Call it once, after [`VirtMachine::machine_done`], on a `jit` that has no
    /// vCPUs yet.
    pub fn create_vcpus(&self, jit: &Arc<Jit>) -> Result<Vec<Vcpu>, String> {
        if !self.done {
            return Err("the vCPUs are created after machine_done".to_string());
        }
        let mut vcpus = Vec::with_capacity(self.smp);
        for i in 0..self.smp {
            let st = CpuRiscvState::reset_cfg(i as u64, DEFAULT_RSTVEC, self.riscv.cfg());
            let mut v = create_vcpu(jit, self.riscv.clone(), self.memory_as.clone(), &st);
            let shared = v.shared().clone();
            if shared.cpu_index != i {
                return Err(format!("vCPU {i} got index {}", shared.cpu_index));
            }
            self.wire(i, &shared);
            self.hub.register(i, &shared);
            self.reset_cpu(&mut v.cpu());
            vcpus.push(v);
        }
        // QEMU wires the harts before the first qemu_system_reset(), whose MTIMER reset
        // raises MTIP at once (every mtimecmp is 0). Here the lines did not exist yet at
        // machine_done, so reset the MTIMER again to drive them.
        self.mtimer.reset();
        Ok(vcpus)
    }

    /// Connect the MSIP, MTIP, the two external interrupt lines and the guest external
    /// interrupt lines of hart `i` to the vCPU `shared`.
    fn wire(&self, i: usize, shared: &Arc<CpuShared>) {
        let mut lines: Vec<(&IrqPin, u32)> =
            vec![(self.swi.soft_irq(i), IRQ_M_SOFT), (self.mtimer.timer_irq(i), IRQ_M_TIMER)];
        match &self.irqchip {
            Irqchip::Plic(plic) => {
                lines.push((plic.m_external_irq(i), IRQ_M_EXT));
                lines.push((plic.s_external_irq(i), IRQ_S_EXT));
            }
            Irqchip::Aia(aia) if !aia.msimode() => {
                // riscv_aplic_realize(): the direct mode domains drive the external lines.
                lines.push((aia.aplic_m().external_irq(i), IRQ_M_EXT));
                lines.push((aia.aplic_s().external_irq(i), IRQ_S_EXT));
            }
            Irqchip::Aia(aia) => {
                // riscv_imsic_realize(): page 0 is the external interrupt of the level, the
                // guest pages are the guest external interrupts. QEMU asserts when a hart
                // has fewer guest lines (no H); those are left unconnected here.
                lines.push((aia.imsic_m()[i].external_irq(0), IRQ_M_EXT));
                let s = &aia.imsic_s()[i];
                lines.push((s.external_irq(0), IRQ_S_EXT));
                for page in 1..=self.aia_guests.min(self.riscv.geilen()) {
                    lines.push((s.external_irq(page as usize), IRQ_LOCAL_MAX + page - 1));
                }
            }
        }
        for (pin, irq) in lines {
            let riscv = Arc::downgrade(&self.riscv);
            let cpu = Arc::downgrade(shared);
            pin.connect(IrqLine::from_fn(move |level| {
                if let (Some(r), Some(c)) = (riscv.upgrade(), cpu.upgrade()) {
                    r.set_irq(&c, irq, level != 0);
                }
            }));
        }
    }

    /// `riscv_cpu_reset_hold()` for the vCPU `cpu`, then the PC a `-device loader` with
    /// `cpu-num` sets (`generic_loader_reset()`).
    pub fn reset_cpu(&self, cpu: &mut Cpu<'_>) {
        let shared = cpu.core.shared().clone();
        let idx = shared.cpu_index;
        let mut st = CpuRiscvState::reset_cfg(idx as u64, DEFAULT_RSTVEC, self.riscv.cfg());
        for l in self.loaders.iter().filter(|l| l.set_pc && l.cpu == idx) {
            st.pc = l.addr;
        }
        st.store(cpu.env);
        self.riscv.reset_lines(&shared);
        ruvm_jit::cputlb::tlb_flush(cpu);
        // cpu_common_reset(): the harts start powered on.
        shared.halted.store(0, std::sync::atomic::Ordering::Release);
        self.hub.reset_timer(idx);
    }

    /// The pending shutdown or reset request, taken. Requests made while a handler is set
    /// ([`VirtMachine::set_request_handler`]) go to the handler instead.
    pub fn take_request(&self) -> Option<VirtRequest> {
        self.hub.take_request()
    }

    /// Send the shutdown and reset requests to `handler`, on the thread of the vCPU that
    /// made them, rather than keeping them for [`VirtMachine::take_request`].
    pub fn set_request_handler(&self, handler: Option<VirtRequestHandler>) {
        self.hub.set_request_handler(handler);
    }

    /// The CPU operations every vCPU runs.
    pub fn riscv(&self) -> &Arc<Riscv> {
        &self.riscv
    }

    /// The `riscv,isa` string of the harts.
    pub fn isa(&self) -> String {
        self.riscv.cfg().isa_string()
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

    /// The ACLINT MSWI of the CLINT.
    pub fn swi(&self) -> &Arc<RiscvAclintSwi> {
        &self.swi
    }

    /// The ACLINT MTIMER of the CLINT.
    pub fn mtimer(&self) -> &Arc<RiscvAclintMtimer> {
        &self.mtimer
    }

    /// The PLIC, with `aia=none`.
    pub fn plic(&self) -> Option<&Arc<SiFivePlic>> {
        match &self.irqchip {
            Irqchip::Plic(plic) => Some(plic),
            Irqchip::Aia(_) => None,
        }
    }

    /// The APLIC domains and IMSICs, with `aia=aplic` or `aia=aplic-imsic`.
    pub fn aia(&self) -> Option<&Aia> {
        match &self.irqchip {
            Irqchip::Plic(_) => None,
            Irqchip::Aia(aia) => Some(aia),
        }
    }

    /// The SiFive test device.
    pub fn test_device(&self) -> &Arc<SiFiveTest> {
        &self.test
    }

    /// The UART.
    pub fn uart(&self) -> &Arc<Serial> {
        &self.uart
    }

    /// The RTC.
    pub fn rtc(&self) -> &Arc<GoldfishRtc> {
        &self.rtc
    }

    /// The two flashes, `virt.flash0` and `virt.flash1`.
    pub fn flash(&self) -> &[Arc<Pflash>; 2] {
        &self.flash
    }

    /// fw_cfg.
    pub fn fw_cfg(&self) -> &FwCfgMem {
        &self.fw_cfg
    }

    /// The device tree: the board's until machine_done, then the packed one loaded into the
    /// guest, what `dumpdtb` writes.
    pub fn fdt(&self) -> &Fdt {
        &self.fdt
    }

    /// The ROM list, sorted by address space and address.
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

    /// Where the firmware, kernel, initrd and device tree went.
    pub fn boot_info(&self) -> &BootInfo {
        &self.info
    }

    /// The messages printed while loading.
    pub fn messages(&self) -> &[String] {
        &self.loader.messages
    }

    /// The clock the ACLINT and Sstc timers run on. The runner calls its `run_timers()`.
    pub fn clock(&self) -> &Arc<Clock> {
        &self.clock
    }

    /// The number of harts.
    pub fn smp(&self) -> usize {
        self.smp
    }

    /// The RAM size.
    pub fn ram_size(&self) -> u64 {
        self.ram_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board(cfg: VirtConfig) -> VirtMachine {
        let mut cfg = cfg;
        cfg.rng_seed = Some([7; 32]);
        let mut m = VirtMachine::new(cfg).unwrap();
        m.machine_done().unwrap();
        m
    }

    fn read(m: &VirtMachine, addr: u64, len: usize) -> Vec<u8> {
        let mut b = vec![0u8; len];
        assert!(m.memory_as().read(addr, MemTxAttrs::UNSPECIFIED, &mut b).is_ok());
        b
    }

    #[test]
    fn boot_rom_without_firmware() {
        let m = board(VirtConfig::default());
        let info = m.boot_info().clone();
        assert_eq!(info.start_addr, VIRT_DRAM);
        assert_eq!(info.fdt_addr, 0x87e0_0000);
        assert_eq!(info.kernel_entry, 0);
        let rom = read(&m, VIRT_MROM, 40);
        assert_eq!(rom, boot::reset_vec(VIRT_DRAM, 0x87e0_0000));
        let finfo = read(&m, VIRT_MROM + 40, 48);
        assert_eq!(&finfo[..8], &0x4942_534fu64.to_le_bytes());
        // The device tree is in RAM.
        assert_eq!(read(&m, 0x87e0_0000, 4), [0xd0, 0x0d, 0xfe, 0xed]);
        let names: Vec<_> = m.roms().iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, ["mrom.reset", "mrom.finfo", "fdt"]);
        let isa = m.fdt().getprop("/cpus/cpu@0", "riscv,isa").unwrap();
        assert_eq!(isa, format!("{QEMU_RV64_ISA}\0").as_bytes());
    }

    #[test]
    fn isa_string_follows_the_cpu() {
        let cpu = RiscvCfg { ext_xlrbr: true, ..RiscvCfg::default() };
        let m = board(VirtConfig { cpu, ..VirtConfig::default() });
        let isa = m.fdt().getprop("/cpus/cpu@0", "riscv,isa").unwrap();
        assert_eq!(isa, format!("{QEMU_RV64_ISA}_xlrbr\0").as_bytes());
        let ext = m.fdt().getprop("/cpus/cpu@0", "riscv,isa-extensions").unwrap();
        assert!(ext.ends_with(b"svvptc\0xlrbr\0"));
        assert_eq!(m.isa(), format!("{QEMU_RV64_ISA}_xlrbr"));
    }

    #[test]
    fn firmware_and_loader_share_the_base_of_ram() {
        let dir = std::env::temp_dir().join(format!("ruvm-riscv-virt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fw = dir.join("fw.bin");
        std::fs::write(&fw, vec![0x11u8; 0x100]).unwrap();
        let raw = dir.join("test.bin");
        std::fs::write(&raw, [0x22u8; 8]).unwrap();
        let cfg = VirtConfig {
            firmware: Some(fw.to_string_lossy().into_owned()),
            loaders: vec![GenericLoader {
                file: Some(raw.to_string_lossy().into_owned()),
                addr: VIRT_DRAM,
                force_raw: true,
                ..GenericLoader::default()
            }],
            ..VirtConfig::default()
        };
        let m = board(cfg);
        // The loader's blob is in the CPU's address space, so it does not overlap the
        // firmware in "memory", and it is copied after it.
        let as_names: Vec<_> = m.roms().iter().map(|r| r.as_name).collect();
        assert_eq!(as_names.last(), Some(&AS_CPU0));
        assert_eq!(read(&m, VIRT_DRAM, 9), [0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x11]);
        assert_eq!(m.boot_info().start_addr, VIRT_DRAM);
        assert_eq!(m.boot_info().firmware_end, VIRT_DRAM + 0x100);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_kernel() {
        let mut m = VirtMachine::new(VirtConfig {
            kernel: Some("/nonexistent/ruvm-kernel".to_string()),
            rng_seed: Some([0; 32]),
            ..VirtConfig::default()
        })
        .unwrap();
        assert_eq!(
            m.machine_done().unwrap_err(),
            "could not load kernel '/nonexistent/ruvm-kernel'"
        );
    }

    fn write32(m: &VirtMachine, addr: u64, value: u32) {
        assert!(m.memory_as().write(addr, MemTxAttrs::UNSPECIFIED, &value.to_le_bytes()).is_ok());
    }

    #[test]
    fn aplic_replaces_the_plic() {
        let m = board(VirtConfig { smp: 2, aia: VirtAia::Aplic, ..VirtConfig::default() });
        assert!(m.plic().is_none());
        let aia = m.aia().unwrap();
        assert!(!aia.msimode());
        assert!(aia.imsic_m().is_empty() && aia.imsic_s().is_empty());
        assert_eq!(aia.aplic_m().config().num_harts, 2);
        // domaincfg reads with bit 31 set, at both domains.
        assert_eq!(read(&m, VIRT_APLIC_M, 4), [0, 0, 0, 0x80]);
        assert_eq!(read(&m, VIRT_APLIC_S, 4), [0, 0, 0, 0x80]);
        assert!(m.fdt().getprop("/cpus/cpu@0", "riscv,isa").unwrap().starts_with(b"rv64"));
        assert_eq!(m.isa(), QEMU_RV64_ISA);
    }

    #[test]
    fn aplic_imsic_sends_messages() {
        let cfg =
            VirtConfig { smp: 2, aia: VirtAia::AplicImsic, aia_guests: 2, ..VirtConfig::default() };
        let m = board(cfg);
        assert!(m.riscv().cfg().ext_smaia && m.riscv().cfg().ext_ssaia);
        assert_eq!(m.riscv().geilen(), 2);
        let aia = m.aia().unwrap();
        assert_eq!(aia.imsic_s()[1].num_pages(), 3);

        // A store to the S level IMSIC of hart 1, guest file 2, sets that interrupt.
        let guest2 = VIRT_IMSIC_S + imsic_hart_size(imsic_num_bits(3)) + 2 * 0x1000;
        write32(&m, guest2, 9);
        assert!(aia.imsic_s()[1].is_pending(2, 9));

        // The M level domain in MSI mode: source 10, level high, to hart 0 with EIID 7.
        write32(&m, VIRT_APLIC_M + 0x1bc0, (VIRT_IMSIC_M >> 12) as u32);
        write32(&m, VIRT_APLIC_M + 0x1bc4, 0);
        write32(&m, VIRT_APLIC_M + 0x4 + 9 * 4, 6);
        write32(&m, VIRT_APLIC_M + 0x3004 + 9 * 4, 7);
        write32(&m, VIRT_APLIC_M + 0x1edc, 10);
        write32(&m, VIRT_APLIC_M, (1 << 8) | (1 << 2));
        assert!(!aia.imsic_m()[0].is_pending(0, 7));
        aia.input(10).raise();
        assert!(aia.imsic_m()[0].is_pending(0, 7));
        assert!(!aia.imsic_m()[1].is_pending(0, 7));
    }

    #[test]
    fn too_many_aia_guests() {
        let cfg = VirtConfig { aia: VirtAia::AplicImsic, aia_guests: 8, ..VirtConfig::default() };
        assert_eq!(VirtMachine::new(cfg).unwrap_err(), "Invalid number of AIA IMSIC guests");
    }

    /// The tables in `etc/acpi/tables` with their checksums filled in as the firmware would,
    /// and the FADT pointers zeroed as bios-tables-test does.
    fn acpi_tables(t: &BuildTables) -> Vec<(String, Vec<u8>)> {
        use ruvm_firmware::acpi::linker::Command;
        use ruvm_firmware::acpi::table::checksum;
        let mut data = t.table_data.clone();
        for cmd in t.linker.commands() {
            if let Command::AddChecksum { file, offset, start, length } = cmd {
                if file != TABLE_FILE {
                    continue;
                }
                let (o, s) = (offset as usize, start as usize);
                data[o] = 0;
                data[o] = checksum(&data[s..s + length as usize]);
            }
        }
        let mut tables = Vec::new();
        let mut at = 0;
        while at + 8 <= data.len() && data[at..at + 4] != [0; 4] {
            let len = u32::from_le_bytes(data[at + 4..at + 8].try_into().unwrap()) as usize;
            let sig = String::from_utf8(data[at..at + 4].to_vec()).unwrap();
            let mut table = data[at..at + len].to_vec();
            assert_eq!(checksum(&table), 0, "{sig} checksum");
            if sig == "FACP" {
                table[36..44].fill(0);
                table[132..148].fill(0);
                table[9] = 0;
                table[9] = checksum(&table);
            }
            tables.push((sig, table));
            at += len;
        }
        tables
    }

    /// The board's tables with `-cpu rva22s64` against QEMU's bios-tables-test blobs.
    #[test]
    fn acpi_tables_match_qemu() {
        let m = board(VirtConfig { cpu: RiscvCfg::model("rva22s64"), ..VirtConfig::default() });
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../vendor-qemu/acpi-expected/riscv64/virt");
        let tables = acpi_tables(&m.acpi_tables());
        let sigs: Vec<&str> = tables.iter().map(|(s, _)| s.as_str()).collect();
        assert_eq!(sigs, ["DSDT", "FACP", "APIC", "RHCT", "SPCR", "MCFG", "XSDT"]);
        for (sig, got) in &tables[..6] {
            let want = std::fs::read(dir.join(sig)).unwrap();
            assert!(*got == want, "{sig} differs from QEMU's");
        }
    }

    #[test]
    fn acpi_tables_go_to_fw_cfg() {
        let names = |m: &VirtMachine| -> Vec<String> {
            m.fw_cfg().state().files().into_iter().map(|(n, _, _)| n).collect()
        };
        let m = board(VirtConfig::default());
        let files = m.fw_cfg().state().files();
        for name in [TABLE_FILE, RSDP_FILE, LOADER_FILE] {
            let (_, key, size) = files.iter().find(|(n, _, _)| n == name).unwrap().clone();
            let data = m.fw_cfg().state().entry_data(key).unwrap();
            assert_eq!(data.len(), size as usize);
        }
        let tables = files.iter().find(|(n, _, _)| n == TABLE_FILE).unwrap();
        assert_eq!(tables.2, 0x20000);
        let m = board(VirtConfig { acpi: false, ..VirtConfig::default() });
        assert!(!names(&m).iter().any(|n| n.starts_with("etc/acpi")));
        // spcr=off leaves the SPCR out.
        let m = board(VirtConfig { spcr: false, ..VirtConfig::default() });
        let sigs: Vec<String> = acpi_tables(&m.acpi_tables()).into_iter().map(|(s, _)| s).collect();
        assert!(!sigs.contains(&"SPCR".to_string()));
    }

    /// With a firmware in the first flash and a `-bios`, the kernel (inflated), the initrd and
    /// the command line go through fw_cfg and the harts go to the flash.
    #[test]
    fn firmware_boot_through_fw_cfg() {
        use flate2::Compression;
        use flate2::write::GzEncoder;
        use ruvm_hw_core::fw_cfg::{
            FW_CFG_CMDLINE_DATA, FW_CFG_CMDLINE_SIZE, FW_CFG_INITRD_DATA, FW_CFG_INITRD_SIZE,
            FW_CFG_KERNEL_DATA, FW_CFG_KERNEL_SIZE,
        };
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("ruvm-riscv-fwboot-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fw = dir.join("fw.bin");
        std::fs::write(&fw, vec![0x11u8; 0x100]).unwrap();
        let image: Vec<u8> = (0..10_000u32).map(|i| (i % 251) as u8).collect();
        let mut gz = GzEncoder::new(Vec::new(), Compression::default());
        gz.write_all(&image).unwrap();
        let kernel = dir.join("Image.gz");
        std::fs::write(&kernel, gz.finish().unwrap()).unwrap();
        let initrd = dir.join("initrd");
        std::fs::write(&initrd, [0x1f, 0x8b, 1, 2, 3]).unwrap();
        let path = |p: &std::path::Path| Some(p.to_string_lossy().into_owned());
        let half = (VIRT_FLASH_SIZE / 2) as usize;
        let m = board(VirtConfig {
            firmware: path(&fw),
            pflash: [PflashBacking::Bytes(vec![0; half]), PflashBacking::None],
            kernel: path(&kernel),
            initrd: path(&initrd),
            append: Some("console=ttyS0".to_string()),
            ..VirtConfig::default()
        });
        assert_eq!(m.boot_info().kernel_entry, VIRT_FLASH);
        let fwc = m.fw_cfg().state();
        let get = |key| fwc.entry_data(key).unwrap();
        assert_eq!(get(FW_CFG_KERNEL_SIZE), 10_000u32.to_le_bytes());
        assert_eq!(get(FW_CFG_KERNEL_DATA), image);
        // The initrd is passed as it is even though it starts with the gzip magic.
        assert_eq!(get(FW_CFG_INITRD_SIZE), 5u32.to_le_bytes());
        assert_eq!(get(FW_CFG_INITRD_DATA), [0x1f, 0x8b, 1, 2, 3]);
        assert_eq!(get(FW_CFG_CMDLINE_SIZE), 14u32.to_le_bytes());
        assert_eq!(get(FW_CFG_CMDLINE_DATA), b"console=ttyS0\0");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn high_pcie_window() {
        assert_eq!(high_pcie_base(128 << 20), 16 << 30);
        assert_eq!(high_pcie_base(16 << 30), 32 << 30);
    }
}
