// SPDX-License-Identifier: GPL-2.0-or-later

//! The `microvm` machine, hw/i386/microvm.c with the parts of x86.c and x86-common.c it uses.
//!
//! microvm is a minimal board: RAM, firmware, fw_cfg, one or two IOAPICs, a row of virtio-mmio
//! transports and, depending on the properties, an 8259 pair, a PIT, an RTC, one ISA serial port
//! and an ACPI Generic Event Device. There is no PCI unless `pcie=on`, which is not supported
//! here yet.
//!
//! Building a machine goes in three steps, like QEMU's startup:
//!
//! 1. [`Microvm::new`] is `microvm_machine_state_init()`: memory, fw_cfg, the kernel and
//!    the devices.
//! 2. Devices are plugged: [`Microvm::attach_virtio`] for `-device virtio-*-device` and
//!    [`Microvm::set_serial_backend`] for the chardev of the serial port.
//! 3. [`Microvm::machine_done`] runs the machine-done notifiers (ACPI tables, the device tree,
//!    `etc/e820`, `bootorder`) and then the first system reset.
//!
//! vCPUs are not created here. An accelerator takes the address spaces, the RAM ranges, the
//! 8259 output and the IOAPIC message hook from the accessors below.

mod fdt;
pub mod props;

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, PoisonError, RwLock, Weak};
use std::time::SystemTime;

use ruvm_base::ClockType;
use ruvm_firmware::acpi::devices::IsaDevice;
use ruvm_firmware::acpi::microvm::{self as acpi_microvm, MicrovmAcpi};
use ruvm_firmware::acpi::table::{LOADER_FILE, RSDP_FILE, TABLE_FILE};
use ruvm_firmware::acpi::x86::{MadtConfig, PossibleCpu};
use ruvm_firmware::e820::{E820_FILE, E820_RAM, E820Table};
use ruvm_firmware::x86_linux::{
    FW_CFG_CMDLINE_DATA, FW_CFG_CMDLINE_SIZE, X86KernelBoot, X86LinuxInput, x86_load_linux,
};
use ruvm_hw_acpi::ged::{
    ACPI_GED_EVT_SEL_LEN, ACPI_GED_PWR_DOWN_EVT, ACPI_GED_REG_COUNT, MICROVM_GED_MMIO_BASE,
    MICROVM_GED_MMIO_BASE_REGS, MICROVM_GED_MMIO_IRQ,
};
use ruvm_hw_acpi::{AcpiGed, AcpiGedProps};
use ruvm_hw_char::serial::{SERIAL_BAUDBASE_DEFAULT, SERIAL_IO_SIZE, Serial, SerialBackend};
use ruvm_hw_core::fw_cfg::{
    DmaMemory, FW_CFG_ARCH_LOCAL, FW_CFG_CTL_SIZE, FW_CFG_DMA_SIZE, FW_CFG_IO_BASE,
    FW_CFG_MAX_CPUS, FW_CFG_NB_CPUS, FW_CFG_RAM_SIZE, FwCfgIo, FwCfgMachineConfig, FwCfgState,
    fw_cfg_init_io_dma,
};
use ruvm_hw_core::{Clock, IrqLine, IrqPin, irq};
use ruvm_hw_intc::i8259::{I8259Pair, ISA_NUM_IRQS, i8259_init};
use ruvm_hw_intc::ioapic::{
    IO_APIC_DEFAULT_ADDRESS, IO_APIC_SECONDARY_ADDRESS, IO_APIC_SECONDARY_IRQBASE, IOAPIC_NUM_PINS,
    IOAPIC_VER_DEF, IoApic, IoApicMsiHandler, IoApics,
};
use ruvm_hw_timer::i8254::I8254;
use ruvm_hw_timer::mc146818::{Mc146818Props, Mc146818Rtc};
use ruvm_hw_virtio::mmio::{VIRTIO_MMIO_FORCE_LEGACY_DEFAULT, VIRTIO_MMIO_REGION_SIZE};
use ruvm_hw_virtio::{VirtioBackend, VirtioDeviceClass, VirtioMmio};
use ruvm_mem::{
    AccessConstraints, AccessCtx, AccessSize, AddressSpace, Endian, MemResult, MemTxAttrs,
    MemorySystem, MmioOps, RamBlock, RegionId, RegionType,
};
use ruvm_virtio_queue::{GuestMemory, MemoryError};

pub use props::{MicrovmProps, OnOffAuto};

const KIB: u64 = 1 << 10;
const MIB: u64 = 1 << 20;

/// `mc->desc`.
pub const MICROVM_DESC: &str = "microvm (i386)";
/// `mc->max_cpus`.
pub const MICROVM_MAX_CPUS: u32 = 288;
/// `mc->default_ram_id`.
pub const MICROVM_RAM_ID: &str = "microvm.ram";
/// The default `-m` of x86 machines.
pub const MICROVM_DEFAULT_RAM_SIZE: u64 = 128 * MIB;
/// `MICROVM_QBOOT_FILENAME`, the default firmware with ACPI off.
pub const MICROVM_QBOOT_FILENAME: &str = "qboot.rom";
/// `MICROVM_BIOS_FILENAME`, the default firmware with ACPI on.
pub const MICROVM_BIOS_FILENAME: &str = "bios-microvm.bin";
/// RAM above this much goes above 4 GiB, the `lowmem` of `microvm_memory_init()`.
pub const MICROVM_LOWMEM: u64 = 0xc000_0000;
/// Where RAM above `lowmem` is mapped.
pub const MICROVM_ABOVE_4G_BASE: u64 = 0x1_0000_0000;
/// `VIRTIO_MMIO_BASE`, the first virtio-mmio transport.
pub const VIRTIO_MMIO_BASE: u64 = acpi_microvm::VIRTIO_MMIO_BASE;
/// The distance between two virtio-mmio transports.
pub const VIRTIO_MMIO_STRIDE: u64 = acpi_microvm::VIRTIO_MMIO_SIZE;
/// `FW_CFG_IRQ0_OVERRIDE` from hw/i386/fw_cfg.h.
pub const FW_CFG_IRQ0_OVERRIDE: u16 = FW_CFG_ARCH_LOCAL + 2;
/// The ports of the PIT.
pub const PIT_IO_BASE: u64 = 0x40;
/// The ports of the RTC.
pub const RTC_IO_BASE: u64 = 0x70;
/// The ports of COM1.
pub const SERIAL_IO_BASE: u64 = 0x3f8;
/// The ISA IRQ of COM1.
pub const SERIAL_IRQ: u32 = 4;
/// The ISA IRQ of the RTC.
pub const RTC_IRQ: u32 = 8;
/// The base year microvm gives the RTC.
pub const RTC_BASE_YEAR: i32 = 2000;
/// `VIRTIO_CMDLINE_MAXLEN`.
const VIRTIO_CMDLINE_MAXLEN: usize = 64;
/// The largest part of the firmware mapped below 1 MiB, `x86_isa_bios_init()`.
const ISA_BIOS_MAX: u64 = 128 * KIB;

/// A `-kernel` boot.
#[derive(Clone, Debug, Default)]
pub struct KernelConfig {
    /// The `-kernel` path, for messages.
    pub filename: String,
    /// The kernel file.
    pub data: Vec<u8>,
    /// `-append`.
    pub cmdline: String,
    /// The `-initrd` file.
    pub initrd: Option<Vec<u8>>,
    /// The `-dtb` path, for messages.
    pub dtb_filename: String,
    /// The `-dtb` file.
    pub dtb: Option<Vec<u8>>,
}

/// An `-option-rom` argument.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptionRom {
    /// The file name, looked up in [`MicrovmConfig::rom_files`].
    pub name: String,
    /// `bootindex`, -1 when not given.
    pub bootindex: i32,
}

/// Everything [`Microvm::new`] needs: the command line after parsing, with files already read.
pub struct MicrovmConfig {
    /// `-m`.
    pub ram_size: u64,
    /// `-smp cpus=`.
    pub cpus: u32,
    /// `-smp maxcpus=`, 0 for the same as `cpus`.
    pub max_cpus: u32,
    /// Whether the accelerator is KVM. With KVM `rtc=auto` leaves the RTC out.
    pub kvm: bool,
    /// The `-machine` properties.
    pub props: MicrovmProps,
    /// `-bios`, or `None` for the default of [`default_firmware_name`].
    pub firmware_name: Option<String>,
    /// The contents of the firmware file, `None` if it could not be found.
    pub firmware: Option<Vec<u8>>,
    /// `-kernel` and friends.
    pub kernel: Option<KernelConfig>,
    /// `-option-rom`, in command line order.
    pub option_roms: Vec<OptionRom>,
    /// ROM files by name: the `-option-rom` files and the boot ROMs the kernel loader asks
    /// for (`linuxboot_dma.bin`, `pvh.bin`). A missing file gets QEMU's warning and is
    /// skipped.
    pub rom_files: BTreeMap<String, Vec<u8>>,
    /// Whether `serial_hd(0)` exists, which is what makes `isa-serial=on` create a port.
    /// False for `-serial none` and `-nodefaults` without `-serial`.
    pub serial_hd: bool,
    /// The chardev behind the serial port. It can also be set later.
    pub serial_backend: Option<Arc<dyn SerialBackend>>,
    /// `-uuid`, `-boot` and the display options fw_cfg reports.
    pub fw_cfg: FwCfgMachineConfig,
    /// `QEMU_CLOCK_VIRTUAL`, for the PIT, serial port and IOAPIC timers.
    pub clock: Arc<Clock>,
    /// `rtc_clock`, the clock the RTC counts on.
    pub rtc_clock: Arc<Clock>,
    /// The date the RTC starts from, `-rtc base=`.
    pub rtc_date: SystemTime,
}

impl fmt::Debug for MicrovmConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MicrovmConfig")
            .field("ram_size", &self.ram_size)
            .field("cpus", &self.cpus)
            .field("max_cpus", &self.max_cpus)
            .field("kvm", &self.kvm)
            .field("props", &self.props)
            .field("firmware_name", &self.firmware_name)
            .field("kernel", &self.kernel.as_ref().map(|k| &k.filename))
            .field("option_roms", &self.option_roms)
            .field("serial_hd", &self.serial_hd)
            .finish_non_exhaustive()
    }
}

impl Default for MicrovmConfig {
    /// 128 MiB, one CPU, default properties, no firmware bytes and clocks that only move when
    /// stepped.
    fn default() -> Self {
        MicrovmConfig {
            ram_size: MICROVM_DEFAULT_RAM_SIZE,
            cpus: 1,
            max_cpus: 0,
            kvm: false,
            props: MicrovmProps::default(),
            firmware_name: None,
            firmware: None,
            kernel: None,
            option_roms: Vec::new(),
            rom_files: BTreeMap::new(),
            serial_hd: true,
            serial_backend: None,
            fw_cfg: FwCfgMachineConfig::default(),
            clock: Clock::manual(ClockType::Virtual),
            rtc_clock: Clock::manual(ClockType::Host),
            rtc_date: SystemTime::now(),
        }
    }
}

/// The firmware file microvm loads without `-bios`: `bios-microvm.bin` with ACPI and
/// `qboot.rom` without.
pub fn default_firmware_name(props: &MicrovmProps) -> &'static str {
    if props.acpi_enabled() { MICROVM_BIOS_FILENAME } else { MICROVM_QBOOT_FILENAME }
}

/// One stretch of guest physical memory backed by host RAM, for a KVM memory slot.
#[derive(Clone, Debug)]
pub struct GuestRamRange {
    /// The name of the region that maps it.
    pub name: String,
    /// Guest physical address.
    pub gpa: u64,
    /// Length in bytes.
    pub size: u64,
    /// The RAM block behind it.
    pub block: Arc<RamBlock>,
    /// Where the range starts in `block`.
    pub offset: u64,
    /// Whether guest writes must fault.
    pub readonly: bool,
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

/// `GSIState` and `gsi_handler()`: GSIs 0 to 15 go to the 8259 and the first IOAPIC, 16 to 23
/// to the first IOAPIC and 24 to 47 to the second.
struct GsiState {
    i8259: Vec<IrqLine>,
    ioapic: Vec<IrqLine>,
    ioapic2: Vec<IrqLine>,
}

impl GsiState {
    fn set(&self, n: u32, level: i32) {
        let n = n as usize;
        let base2 = IO_APIC_SECONDARY_IRQBASE as usize;
        if n < ISA_NUM_IRQS {
            if let Some(l) = self.i8259.get(n) {
                l.set(level);
            }
        }
        if n < IOAPIC_NUM_PINS {
            if let Some(l) = self.ioapic.get(n) {
                l.set(level);
            }
        } else if n >= base2 && n < base2 + IOAPIC_NUM_PINS {
            if let Some(l) = self.ioapic2.get(n - base2) {
                l.set(level);
            }
        }
    }
}

/// One virtio-mmio transport. The device behind a `VirtioMmio` is fixed when it is created, so
/// the region forwards to whichever transport is current and plugging a device swaps it.
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

/// What the kernel loader left for reset time: ELF segments of a PVH kernel.
#[derive(Clone, Debug)]
struct RomBlob {
    addr: u64,
    data: Vec<u8>,
    size: u64,
}

/// `unassigned_io_ops`, behind `get_system_io()`: ports nobody claimed read as all ones and
/// ignore writes.
#[derive(Debug)]
struct UnassignedIo;

impl MmioOps for UnassignedIo {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::MAX)
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
        AccessConstraints::any_size(1, 4).allow_unaligned()
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4).allow_unaligned()
    }
}

fn err<E: fmt::Display>(e: E) -> String {
    e.to_string()
}

/// A microvm board.
pub struct Microvm {
    ram_size: u64,
    below_4g_mem_size: u64,
    above_4g_mem_size: u64,
    cpus: u32,
    max_cpus: u32,
    props: MicrovmProps,
    kvm: bool,

    mem: Arc<MemorySystem>,
    system: RegionId,
    io: RegionId,
    ram: RegionId,
    memory_as: Arc<AddressSpace>,
    io_as: Arc<AddressSpace>,
    bios: RegionId,
    bios_data: Vec<u8>,
    roms: Vec<RomBlob>,

    fw_cfg: FwCfgIo,
    e820: E820Table,
    boot_order: Vec<(i32, String)>,
    warnings: Vec<String>,
    kernel: Option<X86KernelBoot>,
    kernel_cmdline: Option<String>,

    gsi: Vec<IrqLine>,
    ioapics: IoApics,
    ioapic: Arc<IoApic>,
    ioapic2: Option<Arc<IoApic>>,
    msi_hook: Arc<RwLock<Option<IoApicMsiHandler>>>,
    pic: Option<I8259Pair>,
    pic_output: Arc<IrqPin>,
    pit: Option<Arc<I8254>>,
    rtc: Option<Arc<Mc146818Rtc>>,
    serial: Option<Arc<Serial>>,
    ged: Option<Arc<AcpiGed>>,
    virtio: Vec<Arc<VirtioSlot>>,
    virtio_irq_base: u32,

    done: bool,
    kernel_cmdline_fixed: bool,
}

impl fmt::Debug for Microvm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Microvm")
            .field("ram_size", &self.ram_size)
            .field("below_4g_mem_size", &self.below_4g_mem_size)
            .field("above_4g_mem_size", &self.above_4g_mem_size)
            .field("cpus", &self.cpus)
            .field("max_cpus", &self.max_cpus)
            .field("props", &self.props)
            .field("virtio_irq_base", &self.virtio_irq_base)
            .field("virtio_transports", &self.virtio.len())
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}

impl Microvm {
    /// `microvm_machine_state_init()`: checks the configuration, builds memory, loads the
    /// firmware and kernel and creates the devices. Errors carry QEMU's message.
    pub fn new(cfg: MicrovmConfig) -> Result<Microvm, String> {
        let MicrovmConfig {
            ram_size,
            cpus,
            max_cpus,
            kvm,
            props,
            firmware_name,
            firmware,
            kernel,
            option_roms,
            rom_files,
            serial_hd,
            serial_backend,
            fw_cfg: fw_cfg_cfg,
            clock,
            rtc_clock,
            rtc_date,
        } = cfg;

        // machine_parse_smp_config(), for a topology given as a CPU count.
        let cpus = cpus.max(1);
        let max_cpus = if max_cpus == 0 { cpus } else { max_cpus };
        if max_cpus < cpus {
            return Err(format!(
                "Invalid CPU topology: maxcpus must be equal to or greater than smp: \
                 sockets ({max_cpus}) * cores (1) * threads (1) == maxcpus ({max_cpus}) \
                 < smp_cpus ({cpus})"
            ));
        }
        if max_cpus > MICROVM_MAX_CPUS {
            return Err(format!(
                "Invalid SMP CPUs {max_cpus}. The max CPUs supported by machine 'microvm' is \
                 {MICROVM_MAX_CPUS}"
            ));
        }
        let acpi = props.acpi_enabled();
        if acpi && props.usb {
            return Err("microvm: usb=on (sysbus XHCI) is not supported yet".to_string());
        }
        if acpi && props.pcie == OnOffAuto::On {
            return Err("microvm: pcie=on (GPEX host bridge) is not supported yet".to_string());
        }

        // microvm_memory_init()
        let mem = Arc::new(MemorySystem::new());
        let system = mem.new_container("system", 1 << 64).map_err(err)?;
        let io = mem.new_io("io", 1 << 16, Arc::new(UnassignedIo)).map_err(err)?;
        let memory_as = mem.address_space_init(system, "memory").map_err(err)?;
        let io_as = mem.address_space_init(io, "I/O").map_err(err)?;

        let (below_4g_mem_size, above_4g_mem_size) = if ram_size > MICROVM_LOWMEM {
            (MICROVM_LOWMEM, ram_size - MICROVM_LOWMEM)
        } else {
            (ram_size, 0)
        };
        let ram = mem.new_ram(MICROVM_RAM_ID, ram_size).map_err(err)?;
        let below = mem.new_alias("ram-below-4g", ram, 0, below_4g_mem_size.into()).map_err(err)?;
        mem.add_subregion(system, 0, below).map_err(err)?;
        let mut e820 = E820Table::new();
        e820.add_entry(0, below_4g_mem_size, E820_RAM);
        if above_4g_mem_size > 0 {
            let above = mem
                .new_alias("ram-above-4g", ram, below_4g_mem_size, above_4g_mem_size.into())
                .map_err(err)?;
            mem.add_subregion(system, MICROVM_ABOVE_4G_BASE, above).map_err(err)?;
            e820.add_entry(MICROVM_ABOVE_4G_BASE, above_4g_mem_size, E820_RAM);
        }

        let dma: Arc<dyn DmaMemory> = Arc::new(WeakDma(Arc::downgrade(&memory_as)));
        let fw_cfg = fw_cfg_init_io_dma(FW_CFG_IO_BASE, dma, &fw_cfg_cfg).map_err(err)?;
        {
            let comb = mem
                .new_io("fwcfg", FW_CFG_CTL_SIZE.into(), fw_cfg.comb_ops().clone())
                .map_err(err)?;
            mem.add_subregion(io, FW_CFG_IO_BASE.into(), comb).map_err(err)?;
            if let Some(d) = fw_cfg.dma_ops() {
                let dma =
                    mem.new_io("fwcfg.dma", FW_CFG_DMA_SIZE.into(), d.clone()).map_err(err)?;
                mem.add_subregion(io, u64::from(FW_CFG_IO_BASE) + 4, dma).map_err(err)?;
            }
        }
        let fwc = fw_cfg.state();
        fwc.add_i16(FW_CFG_NB_CPUS, cpus as u16);
        fwc.add_i16(FW_CFG_MAX_CPUS, max_cpus as u16);
        fwc.add_i64(FW_CFG_RAM_SIZE, ram_size);
        fwc.add_i32(FW_CFG_IRQ0_OVERRIDE, 1);

        let mut option_roms = option_roms;
        let mut roms = Vec::new();
        let mut kernel_boot = None;
        let mut kernel_cmdline = None;
        if let Some(k) = &kernel {
            let input = X86LinuxInput {
                kernel_filename: &k.filename,
                kernel: &k.data,
                cmdline: &k.cmdline,
                initrd: k.initrd.as_deref(),
                dtb_filename: &k.dtb_filename,
                dtb: k.dtb.as_deref(),
                rng_seed: None,
                below_4g_mem_size,
                acpi_data_size: 0,
                confidential_guest: false,
            };
            let boot = x86_load_linux(&input).map_err(err)?;
            let rom = match &boot {
                X86KernelBoot::Linux(l) => {
                    for item in &l.fw_cfg {
                        fwc.add_bytes(item.key, item.data.clone());
                    }
                    for file in &l.files {
                        fwc.add_file(&file.name, file.data.clone()).map_err(err)?;
                    }
                    l.option_rom
                }
                X86KernelBoot::Pvh(p) => {
                    for item in &p.fw_cfg {
                        fwc.add_bytes(item.key, item.data.clone());
                    }
                    roms.extend(p.segments.iter().map(|s| RomBlob {
                        addr: s.addr,
                        data: s.data.clone(),
                        size: s.mem_size.max(s.data.len() as u64),
                    }));
                    p.option_rom
                }
                X86KernelBoot::Multiboot(_) => {
                    return Err("multiboot kernels are not supported on microvm yet".to_string());
                }
            };
            option_roms.push(OptionRom { name: rom.to_string(), bootindex: 0 });
            kernel_cmdline = Some(k.cmdline.clone());
            kernel_boot = Some(boot);
        }

        let mut warnings = Vec::new();
        let mut boot_order = Vec::new();
        if props.option_roms {
            for rom in &option_roms {
                // rom_add_option()
                let Some(data) = rom_files.get(&rom.name) else {
                    warnings.push(format!(
                        "rom: file {:<20}: error Failed to open file \u{201c}{}\u{201d}: No such \
                         file or directory",
                        rom.name, rom.name
                    ));
                    continue;
                };
                let base = rom.name.rsplit('/').next().unwrap_or(&rom.name);
                let fw_name = format!("genroms/{base}");
                fwc.add_file(&fw_name, data.clone()).map_err(err)?;
                if rom.bootindex >= 0 {
                    boot_order.push((rom.bootindex, format!("/rom@{fw_name}")));
                }
            }
        }
        boot_order.sort_by_key(|b| b.0);

        // microvm_devices_init()
        let ioapic_count = if !acpi || props.ioapic2 == OnOffAuto::Off { 1 } else { 2 };

        let pic_output = Arc::new(IrqPin::new());
        let pic = if props.pic != OnOffAuto::Off {
            let out = Arc::clone(&pic_output);
            let pair = i8259_init(IrqLine::from_fn(move |level| out.set(level)));
            for (name, base, ops) in [
                ("pic", 0x20, pair.master.clone() as Arc<dyn MmioOps>),
                ("pic", 0xa0, pair.slave.clone() as Arc<dyn MmioOps>),
                ("elcr", 0x4d0, pair.master.elcr_io() as Arc<dyn MmioOps>),
                ("elcr", 0x4d1, pair.slave.elcr_io() as Arc<dyn MmioOps>),
            ] {
                let size = if name == "pic" { 2 } else { 1 };
                let r = mem.new_io(name, size, ops).map_err(err)?;
                mem.add_subregion(io, base, r).map_err(err)?;
            }
            Some(pair)
        } else {
            None
        };

        let msi_hook: Arc<RwLock<Option<IoApicMsiHandler>>> = Arc::new(RwLock::new(None));
        let msi: IoApicMsiHandler = {
            let hook = Arc::clone(&msi_hook);
            let weak = Arc::downgrade(&memory_as);
            Arc::new(move |addr, data| {
                let h = hook.read().unwrap_or_else(PoisonError::into_inner).clone();
                match h {
                    Some(h) => h(addr, data),
                    None => {
                        if let Some(a) = weak.upgrade() {
                            let _ = a.write_u32(addr, MemTxAttrs::UNSPECIFIED, data);
                        }
                    }
                }
            })
        };
        let ioapics = IoApics::new();
        let make_ioapic = |base: u64| -> Result<Arc<IoApic>, String> {
            let s =
                IoApic::realize(&clock, IOAPIC_VER_DEF, &ioapics, Arc::clone(&msi)).map_err(err)?;
            if let Some(p) = &pic {
                let master = Arc::clone(&p.master);
                s.set_pic_read_irq(Some(Arc::new(move || master.pic_read_irq())));
            }
            let r = mem.new_io("ioapic", 0x1000, s.clone()).map_err(err)?;
            mem.add_subregion(system, base, r).map_err(err)?;
            Ok(s)
        };
        let ioapic = make_ioapic(IO_APIC_DEFAULT_ADDRESS)?;
        let ioapic2 =
            if ioapic_count > 1 { Some(make_ioapic(IO_APIC_SECONDARY_ADDRESS)?) } else { None };

        let gsi_state = Arc::new(GsiState {
            i8259: pic.as_ref().map(|p| p.irq_set.clone()).unwrap_or_default(),
            ioapic: ioapic.inputs(),
            ioapic2: ioapic2.as_ref().map(|s| s.inputs()).unwrap_or_default(),
        });
        let gsi = irq::allocate(
            Arc::new(move |n, level| gsi_state.set(n, level)),
            (IOAPIC_NUM_PINS * ioapic_count) as u32,
        );

        let (virtio_irq_base, virtio_num_transports) = if ioapic2.is_some() {
            (IO_APIC_SECONDARY_IRQBASE, IOAPIC_NUM_PINS as u32)
        } else if acpi {
            (16, 8)
        } else {
            (5, 8)
        };
        let mut virtio = Vec::new();
        for i in 0..virtio_num_transports {
            let line = gsi[(virtio_irq_base + i) as usize].clone();
            let t = VirtioMmio::new(None, VIRTIO_MMIO_FORCE_LEGACY_DEFAULT).map_err(err)?;
            t.irq().connect(line.clone());
            let slot = Arc::new(VirtioSlot {
                gsi: line,
                transport: RwLock::new(Arc::new(t)),
                plugged: RwLock::new(false),
            });
            let r = mem
                .new_io("virtio-mmio", VIRTIO_MMIO_REGION_SIZE.into(), slot.clone())
                .map_err(err)?;
            mem.add_subregion(system, VIRTIO_MMIO_BASE + u64::from(i) * VIRTIO_MMIO_STRIDE, r)
                .map_err(err)?;
            virtio.push(slot);
        }

        let ged = if acpi {
            let g =
                AcpiGed::new(AcpiGedProps { ged_event: ACPI_GED_PWR_DOWN_EVT, pci_hotplug: false })
                    .map_err(err)?;
            let evt =
                mem.new_io("acpi-ged", ACPI_GED_EVT_SEL_LEN.into(), g.evt_ops()).map_err(err)?;
            mem.add_subregion(system, MICROVM_GED_MMIO_BASE, evt).map_err(err)?;
            let regs = mem
                .new_io("acpi-ged-regs", ACPI_GED_REG_COUNT.into(), g.regs_ops())
                .map_err(err)?;
            mem.add_subregion(system, MICROVM_GED_MMIO_BASE_REGS, regs).map_err(err)?;
            g.irq().connect(gsi[MICROVM_GED_MMIO_IRQ as usize].clone());
            Some(g)
        } else {
            None
        };

        let pit = if props.pit != OnOffAuto::Off {
            let p = I8254::new(&clock, PIT_IO_BASE as u32);
            p.irq.connect(gsi[0].clone());
            let r = mem.new_io("pit", 4, p.clone()).map_err(err)?;
            mem.add_subregion(io, PIT_IO_BASE, r).map_err(err)?;
            Some(p)
        } else {
            None
        };

        let rtc = if props.rtc == OnOffAuto::On || (props.rtc == OnOffAuto::Auto && !kvm) {
            let rtc_props = Mc146818Props {
                base_year: RTC_BASE_YEAR,
                iobase: RTC_IO_BASE as u16,
                irq: RTC_IRQ as u8,
                ..Mc146818Props::default()
            };
            let s = Arc::new(Mc146818Rtc::new(rtc_props, rtc_clock, rtc_date).map_err(err)?);
            s.connect_irq(gsi[RTC_IRQ as usize].clone());
            let r = mem.new_io("rtc", 2, s.clone()).map_err(err)?;
            mem.add_subregion(io, RTC_IO_BASE, r).map_err(err)?;
            set_rtc_cmos(&s, below_4g_mem_size, above_4g_mem_size);
            Some(s)
        } else {
            None
        };

        let serial = if props.isa_serial && serial_hd {
            let s = Serial::new(Arc::clone(&clock), SERIAL_BAUDBASE_DEFAULT, serial_backend);
            s.irq().connect(gsi[SERIAL_IRQ as usize].clone());
            let r = mem.new_io("serial", SERIAL_IO_SIZE.into(), s.clone()).map_err(err)?;
            mem.add_subregion(io, SERIAL_IO_BASE, r).map_err(err)?;
            Some(s)
        } else {
            None
        };

        // x86_bios_rom_init(x86ms, default_firmware, get_system_memory(), true)
        let bios_name = firmware_name.unwrap_or_else(|| default_firmware_name(&props).to_string());
        let bios_data = match firmware {
            Some(d) if !d.is_empty() && d.len() as u64 % 65536 == 0 => d,
            _ => return Err(format!("qemu: could not load PC BIOS '{bios_name}'")),
        };
        let bios_size = bios_data.len() as u64;
        let bios = mem.new_ram("pc.bios", bios_size).map_err(err)?;
        let isa_bios_size = bios_size.min(ISA_BIOS_MAX);
        let isa_bios = mem
            .new_alias("isa-bios", bios, bios_size - isa_bios_size, isa_bios_size.into())
            .map_err(err)?;
        mem.add_subregion_overlap(system, MIB - isa_bios_size, isa_bios, 1).map_err(err)?;
        mem.add_subregion(system, (1u64 << 32) - bios_size, bios).map_err(err)?;

        Ok(Microvm {
            ram_size,
            below_4g_mem_size,
            above_4g_mem_size,
            cpus,
            max_cpus,
            props,
            kvm,
            mem,
            system,
            io,
            ram,
            memory_as,
            io_as,
            bios,
            bios_data,
            roms,
            fw_cfg,
            e820,
            boot_order,
            warnings,
            kernel: kernel_boot,
            kernel_cmdline,
            gsi,
            ioapics,
            ioapic,
            ioapic2,
            msi_hook,
            pic,
            pic_output,
            pit,
            rtc,
            serial,
            ged,
            virtio,
            virtio_irq_base,
            done: false,
            kernel_cmdline_fixed: false,
        })
    }

    /// Plugs a virtio device the way `-device` without `bus=` does: into the free transport
    /// that was created last, so the highest index goes first. Returns the transport index.
    pub fn attach_virtio(&self, class: Box<dyn VirtioDeviceClass>) -> Result<usize, String> {
        let Some(index) = (0..self.virtio.len()).rev().find(|&i| !self.virtio[i].is_plugged())
        else {
            return Err("No 'virtio-bus' bus found for device".to_string());
        };
        self.attach_virtio_at(index, class, VIRTIO_MMIO_FORCE_LEGACY_DEFAULT)?;
        Ok(index)
    }

    /// Plugs a virtio device into transport `index` with the transport's `force-legacy`
    /// property set to `force_legacy`.
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

    /// Connects the chardev of the ISA serial port. Returns false if there is no port.
    pub fn set_serial_backend(&self, backend: Option<Arc<dyn SerialBackend>>) -> bool {
        match &self.serial {
            Some(s) => {
                s.set_backend(backend);
                true
            }
            None => false,
        }
    }

    /// The transports that have a device, newest first as QEMU walks the system bus.
    fn plugged_desc(&self) -> Vec<u32> {
        (0..self.virtio.len() as u32)
            .rev()
            .filter(|&i| self.virtio[i as usize].is_plugged())
            .collect()
    }

    /// The input of `acpi_build_microvm()`.
    pub fn acpi_input(&self) -> MicrovmAcpi {
        let mut isa = Vec::new();
        // The ISA bus lists its children newest first: the serial port, then the RTC.
        if self.serial.is_some() {
            isa.push(IsaDevice::Serial {
                index: 0,
                iobase: SERIAL_IO_BASE as u16,
                irq: SERIAL_IRQ as u8,
            });
        }
        if self.rtc.is_some() {
            isa.push(IsaDevice::Rtc { io_base: RTC_IO_BASE as u16, irq: RTC_IRQ as u8 });
        }
        MicrovmAcpi {
            oem_id: self.props.oem_id.clone(),
            oem_table_id: self.props.oem_table_id.clone(),
            madt: MadtConfig {
                cpus: (0..self.max_cpus)
                    .map(|i| PossibleCpu { arch_id: i, present: i < self.cpus })
                    .collect(),
                pic: self.props.pic != OnOffAuto::Off,
                ioapic2: self.ioapic2.is_some(),
                apic_xrupt_override: false,
                pci_irq_mask: 0,
            },
            isa,
            ged_events: ACPI_GED_PWR_DOWN_EVT,
            virtio_irq_base: self.virtio_irq_base,
            virtio_transports: self.plugged_desc(),
            usb: false,
            i8042: false,
        }
    }

    /// `dt_setup_microvm()`: the device tree edk2 reads from `etc/fdt`.
    pub fn device_tree(&self) -> Vec<u8> {
        let mut f = fdt::Fdt::new();
        f.setprop_string("/", "compatible", "linux,microvm");
        f.setprop_cell("/", "#address-cells", 2);
        f.setprop_cell("/", "#size-cells", 2);
        f.add_subnode("/chosen");

        // The system bus lists its children newest first, so the second IOAPIC comes first.
        let mut phandles = [0u32; 2];
        let mut apics = vec![(1usize, IO_APIC_SECONDARY_ADDRESS)];
        if self.ioapic2.is_none() {
            apics.clear();
        }
        apics.push((0, IO_APIC_DEFAULT_ADDRESS));
        for (index, base) in apics {
            let node = format!("/ioapic{}@{base:x}", index + 1);
            f.add_subnode(&node);
            f.setprop_string(&node, "compatible", "intel,ce4100-ioapic");
            f.setprop(&node, "interrupt-controller", &[]);
            f.setprop_cell(&node, "#interrupt-cells", 2);
            f.setprop_cell(&node, "#address-cells", 2);
            f.setprop_reg64(&node, "reg", base, 0x1000);
            let ph = f.alloc_phandle();
            f.setprop_cell(&node, "phandle", ph);
            f.setprop_cell(&node, "linux,phandle", ph);
            phandles[index] = ph;
        }

        let add_irq = |f: &mut fdt::Fdt, node: &str, irq: u32| {
            let (index, irq) = if irq >= IO_APIC_SECONDARY_IRQBASE {
                (1, irq - IO_APIC_SECONDARY_IRQBASE)
            } else {
                (0, irq)
            };
            f.setprop_cell(node, "interrupt-parent", phandles[index]);
            f.setprop_cells(node, "interrupts", &[irq, 0]);
        };
        for index in self.plugged_desc() {
            let base = VIRTIO_MMIO_BASE + u64::from(index) * VIRTIO_MMIO_STRIDE;
            let node = format!("/virtio_mmio@{base:x}");
            f.add_subnode(&node);
            f.setprop_string(&node, "compatible", "virtio,mmio");
            f.setprop_reg64(&node, "reg", base, VIRTIO_MMIO_STRIDE);
            f.setprop(&node, "dma-coherent", &[]);
            add_irq(&mut f, &node, self.virtio_irq_base + index);
        }
        if self.serial.is_some() {
            let node = format!("/serial@{SERIAL_IO_BASE:x}");
            f.add_subnode(&node);
            f.setprop(&node, "compatible", b"ns16550\0");
            f.setprop_reg64(&node, "reg", SERIAL_IO_BASE, 8);
            add_irq(&mut f, &node, SERIAL_IRQ);
            f.setprop_string("/chosen", "stdout-path", &node);
        }
        if self.rtc.is_some() {
            let node = format!("/rtc@{RTC_IO_BASE:x}");
            f.add_subnode(&node);
            f.setprop(&node, "compatible", b"motorola,mc146818\0");
            f.setprop_reg64(&node, "reg", RTC_IO_BASE, 8);
            add_irq(&mut f, &node, RTC_IRQ);
        }
        f.to_blob()
    }

    fn bootorder(&self) -> Vec<u8> {
        // get_boot_devices_list(): the paths joined by newlines, NUL terminated.
        let mut out = Vec::new();
        for (_, path) in &self.boot_order {
            if let Some(last) = out.last_mut() {
                *last = b'\n';
            }
            out.extend_from_slice(path.as_bytes());
            out.push(0);
        }
        out
    }

    /// The machine-done notifiers and the first system reset: `fw_cfg_machine_ready()`,
    /// `microvm_machine_done()` (ACPI tables, `etc/fdt`, `etc/e820`) and then
    /// `qemu_system_reset()`. Plug devices before calling this.
    pub fn machine_done(&mut self) -> Result<(), String> {
        if self.done {
            return Err("machine_done called twice".to_string());
        }
        self.done = true;
        let fwc = Arc::clone(self.fw_cfg.state());
        fwc.machine_reset(self.bootorder(), Vec::new()).map_err(err)?;

        if self.props.acpi_enabled() {
            let tables = acpi_microvm::build(&self.acpi_input());
            fwc.add_file(TABLE_FILE, tables.table_data).map_err(err)?;
            fwc.add_file(LOADER_FILE, tables.linker.cmd_blob().to_vec()).map_err(err)?;
            fwc.add_file(RSDP_FILE, tables.rsdp).map_err(err)?;
        }
        fwc.add_file("etc/fdt", self.device_tree()).map_err(err)?;
        fwc.add_file(E820_FILE, self.e820.to_blob()).map_err(err)?;

        self.system_reset()
    }

    /// `microvm_machine_reset()` without the CPUs: fixes the kernel command line once, resets
    /// the devices and reloads the firmware and PVH segments into RAM.
    pub fn system_reset(&mut self) -> Result<(), String> {
        if !self.props.acpi_enabled()
            && self.kernel.is_some()
            && self.props.auto_kernel_cmdline
            && !self.kernel_cmdline_fixed
        {
            self.fix_kernel_cmdline();
            self.kernel_cmdline_fixed = true;
        }

        // qemu_devices_reset()
        if let Some(p) = &self.pic {
            p.master.reset();
            p.slave.reset();
        }
        self.ioapic.reset();
        if let Some(s) = &self.ioapic2 {
            s.reset();
        }
        for slot in &self.virtio {
            slot.current().reset();
        }
        if let Some(p) = &self.pit {
            p.reset();
        }
        if let Some(r) = &self.rtc {
            r.reset();
        }
        if let Some(s) = &self.serial {
            s.reset();
        }
        let fwc = self.fw_cfg.state();
        fwc.reset();
        fwc.machine_reset(self.bootorder(), Vec::new()).map_err(err)?;

        // rom_reset()
        let block = self.mem.ram_block(self.bios).ok_or("pc.bios has no RAM block")?;
        block.write(0, &self.bios_data).map_err(err)?;
        for rom in &self.roms {
            let mut data = rom.data.clone();
            data.resize(rom.size as usize, 0);
            if !self.memory_as.write(rom.addr, MemTxAttrs::UNSPECIFIED, &data).is_ok() {
                return Err(format!(
                    "rom: could not write {:#x} bytes at {:#x}",
                    rom.size, rom.addr
                ));
            }
        }
        Ok(())
    }

    /// `microvm_fix_kernel_cmdline()`.
    fn fix_kernel_cmdline(&self) {
        let mut cmdline = self.kernel_cmdline.clone().unwrap_or_default();
        // QEMU works on the C string.
        if let Some(nul) = cmdline.find('\0') {
            cmdline.truncate(nul);
        }
        for index in self.plugged_desc() {
            let add = format!(
                " virtio_mmio.device=512@0x{:x}:{}",
                VIRTIO_MMIO_BASE + u64::from(index) * VIRTIO_MMIO_STRIDE,
                self.virtio_irq_base + index
            );
            if add.len() < VIRTIO_CMDLINE_MAXLEN {
                cmdline.push_str(&add);
            }
        }
        let fwc = self.fw_cfg.state();
        fwc.modify_i32(FW_CFG_CMDLINE_SIZE, cmdline.len() as u32 + 1);
        fwc.modify_string(FW_CFG_CMDLINE_DATA, &cmdline);
    }

    /// The memory system all regions live in.
    pub fn memory_system(&self) -> &Arc<MemorySystem> {
        &self.mem
    }

    /// The root of system memory, `get_system_memory()`.
    pub fn system_memory(&self) -> RegionId {
        self.system
    }

    /// The root of the I/O port space, `get_system_io()`.
    pub fn system_io(&self) -> RegionId {
        self.io
    }

    /// `address_space_memory`.
    pub fn memory_as(&self) -> &Arc<AddressSpace> {
        &self.memory_as
    }

    /// `address_space_io`.
    pub fn io_as(&self) -> &Arc<AddressSpace> {
        &self.io_as
    }

    /// The machine RAM region, `microvm.ram`.
    pub fn ram_region(&self) -> RegionId {
        self.ram
    }

    /// The RAM block behind `microvm.ram`.
    pub fn ram_block(&self) -> Option<Arc<RamBlock>> {
        self.mem.ram_block(self.ram)
    }

    /// The firmware region, `pc.bios`.
    pub fn bios_region(&self) -> RegionId {
        self.bios
    }

    /// Every RAM backed range of system memory after overlaps are resolved: the two RAM
    /// windows, the firmware at the top of 4 GiB and its copy below 1 MiB. These are the KVM
    /// memory slots.
    pub fn ram_ranges(&self) -> Result<Vec<GuestRamRange>, String> {
        let view = self.mem.render(self.system).map_err(err)?;
        let mut out = Vec::new();
        for r in view.ranges() {
            if r.region_type() != RegionType::Ram {
                continue;
            }
            let Some(block) = r.ram_block() else { continue };
            out.push(GuestRamRange {
                name: r.name().to_string(),
                gpa: r.addr(),
                size: r.size() as u64,
                block: Arc::clone(block),
                offset: r.offset_in_region(),
                readonly: r.readonly(),
            });
        }
        Ok(out)
    }

    /// `-m`.
    pub fn ram_size(&self) -> u64 {
        self.ram_size
    }

    /// RAM mapped from address 0.
    pub fn below_4g_mem_size(&self) -> u64 {
        self.below_4g_mem_size
    }

    /// RAM mapped from 4 GiB.
    pub fn above_4g_mem_size(&self) -> u64 {
        self.above_4g_mem_size
    }

    /// `smp.cpus`, the CPUs to create.
    pub fn cpus(&self) -> u32 {
        self.cpus
    }

    /// `smp.max_cpus`.
    pub fn max_cpus(&self) -> u32 {
        self.max_cpus
    }

    /// The APIC IDs of the possible CPUs, index = CPU index. Only a flat topology of sockets
    /// is modelled, so the ID is the index.
    pub fn apic_ids(&self) -> Vec<u32> {
        (0..self.max_cpus).collect()
    }

    /// Whether the machine was built for KVM.
    pub fn kvm(&self) -> bool {
        self.kvm
    }

    /// The properties the machine was built with.
    pub fn props(&self) -> &MicrovmProps {
        &self.props
    }

    /// The fw_cfg device.
    pub fn fw_cfg(&self) -> &Arc<FwCfgState> {
        self.fw_cfg.state()
    }

    /// The e820 table.
    pub fn e820(&self) -> &E820Table {
        &self.e820
    }

    /// Warnings QEMU would print on stderr, such as missing option ROMs.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// What the kernel loader produced, if `-kernel` was given. The CPUs still start at the
    /// reset vector: the firmware and the boot ROM take it from there.
    pub fn kernel_boot(&self) -> Option<&X86KernelBoot> {
        self.kernel.as_ref()
    }

    /// `x86ms->gsi`: 24 lines per IOAPIC.
    pub fn gsi(&self) -> &[IrqLine] {
        &self.gsi
    }

    /// The IOAPICs, for EOI broadcasts from the local APICs.
    pub fn ioapics(&self) -> &IoApics {
        &self.ioapics
    }

    /// The first IOAPIC, at 0xfec00000.
    pub fn ioapic(&self) -> &Arc<IoApic> {
        &self.ioapic
    }

    /// The second IOAPIC, at 0xfec10000, with `ioapic2` and ACPI.
    pub fn ioapic2(&self) -> Option<&Arc<IoApic>> {
        self.ioapic2.as_ref()
    }

    /// Where IOAPIC interrupt messages go. Without a handler they are written to system
    /// memory like `address_space_stl_le(ioapic_as, ...)`, which only reaches something once
    /// a local APIC is mapped there.
    pub fn set_msi_handler(&self, handler: Option<IoApicMsiHandler>) {
        *self.msi_hook.write().unwrap_or_else(PoisonError::into_inner) = handler;
    }

    /// The 8259 pair, with `pic` not off.
    pub fn pic(&self) -> Option<&I8259Pair> {
        self.pic.as_ref()
    }

    /// The 8259 INTR output, `x86_allocate_cpu_irq()`. Connect it to the BSP's LINT0 or the
    /// accelerator's interrupt request.
    pub fn pic_output(&self) -> &Arc<IrqPin> {
        &self.pic_output
    }

    /// The PIT, with `pit` not off.
    pub fn pit(&self) -> Option<&Arc<I8254>> {
        self.pit.as_ref()
    }

    /// The RTC, when present.
    pub fn rtc(&self) -> Option<&Arc<Mc146818Rtc>> {
        self.rtc.as_ref()
    }

    /// COM1, when present.
    pub fn serial(&self) -> Option<&Arc<Serial>> {
        self.serial.as_ref()
    }

    /// The ACPI Generic Event Device, with ACPI.
    pub fn ged(&self) -> Option<&Arc<AcpiGed>> {
        self.ged.as_ref()
    }

    /// `virtio_irq_base`.
    pub fn virtio_irq_base(&self) -> u32 {
        self.virtio_irq_base
    }

    /// `virtio_num_transports`.
    pub fn virtio_transport_count(&self) -> usize {
        self.virtio.len()
    }

    /// The transport at `index`, to reach the device model through
    /// [`VirtioMmio::with_device`].
    pub fn virtio_transport(&self, index: usize) -> Option<Arc<VirtioMmio>> {
        self.virtio.get(index).map(|s| s.current())
    }

    /// Whether transport `index` has a device.
    pub fn virtio_plugged(&self, index: usize) -> bool {
        self.virtio.get(index).is_some_and(|s| s.is_plugged())
    }
}

/// `microvm_set_rtc()`: the memory sizes in the CMOS.
fn set_rtc_cmos(s: &Mc146818Rtc, below: u64, above: u64) {
    let val = (below / KIB).min(640);
    s.set_cmos_data(0x15, val as u8);
    s.set_cmos_data(0x16, (val >> 8) as u8);
    // extended memory (next 64MiB)
    let val = if below > MIB { (below - MIB) / KIB } else { 0 }.min(65535);
    s.set_cmos_data(0x17, val as u8);
    s.set_cmos_data(0x18, (val >> 8) as u8);
    s.set_cmos_data(0x30, val as u8);
    s.set_cmos_data(0x31, (val >> 8) as u8);
    // memory between 16MiB and 4GiB
    let val = if below > 16 * MIB { (below - 16 * MIB) / (64 * KIB) } else { 0 }.min(65535);
    s.set_cmos_data(0x34, val as u8);
    s.set_cmos_data(0x35, (val >> 8) as u8);
    // memory above 4GiB
    let val = above / 65536;
    s.set_cmos_data(0x5b, val as u8);
    s.set_cmos_data(0x5c, (val >> 8) as u8);
    s.set_cmos_data(0x5d, (val >> 16) as u8);
}
