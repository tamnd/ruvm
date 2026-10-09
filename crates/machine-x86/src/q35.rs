// SPDX-License-Identifier: GPL-2.0-or-later

//! The `pc-q35-11.1` machine (alias `q35`): hw/i386/pc_q35.c with the parts of pc.c and
//! x86-common.c it uses.
//!
//! The board is the Q35 host bridge (MCH at 00:00.0), the ICH9 LPC bridge at 00:1f.0 with its
//! power management block, the ICH9 AHCI controller at 00:1f.2, the ICH9 SMBus controller at
//! 00:1f.3 with its SPD EEPROMs, an 8259 pair, one IOAPIC, the HPET, the PIT and speaker port,
//! the RTC, up to four ISA serial ports, the i8042 with port 0x92 and fw_cfg. The firmware is
//! either ROM (`bios-256k.bin` by default) or, when pflash0 has a drive, the two CFI01 system
//! flashes OVMF uses. Both sit at the top of 4 GiB with their last 128 KiB mirrored below 1 MiB
//! through the PAM registers.
//!
//! Building a machine goes in three steps, like QEMU's startup:
//!
//! 1. [`Q35::new`] is `pc_q35_init()`: memory, fw_cfg, the kernel and the devices.
//! 2. Devices are plugged: [`Q35::attach_drive`] for `-drive if=ide` and
//!    [`Q35::set_serial_backend`] for the chardevs of COM1 to COM4. More PCI devices go on
//!    [`Q35::pci_bus`].
//! 3. [`Q35::machine_done`] runs the machine-done notifiers (ACPI tables, `etc/e820`, the late
//!    CMOS setup, `bootorder`) and then the first system reset.
//!
//! fw_cfg carries what QEMU's does: the ACPI tables with their loader script, the SMBIOS
//! tables, `etc/e820`, `bootorder` and the option ROMs, byte for byte (see
//! `tests/qemu_fw_cfg.rs`). On AMD CPUs the e820 map has the HyperTransport hole and RAM above
//! 4 GiB moves past 1 TiB when it would reach it, as in `pc_memory_init()`.
//!
//! Not modelled: the default VGA (the board behaves like `-vga none`), the VMware port, the
//! parallel port, the i8257 DMA controllers, USB, `etc/msr_feature_control` and memory
//! hotplug. The ACPI tables advertise PCI and CPU hotplug as QEMU's do, but the hotplug
//! registers at 0xcc0 and 0xcd8 are not emulated.
//!
//! vCPUs are not created here. An accelerator takes the address spaces (including the SMM one),
//! the RAM ranges, the APIC IDs, the 8259 output, the A20 line and the MSI hook from the
//! accessors below.

pub mod props;

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError, RwLock, Weak};
use std::time::SystemTime;

use ruvm_base::ClockType;
use ruvm_firmware::acpi::devices::IsaDevice;
use ruvm_firmware::acpi::pci::CrsRange;
use ruvm_firmware::acpi::q35::{
    self as acpi_q35, PciDevice as AcpiPciDevice, PciDeviceAml, Q35Acpi,
};
use ruvm_firmware::acpi::table::{LOADER_FILE, McfgInfo, RSDP_FILE, TABLE_FILE, TPMLOG_FILE};
use ruvm_firmware::acpi::x86::{MadtConfig, PossibleCpu};
use ruvm_firmware::e820::{E820_FILE, E820_RAM, E820_RESERVED, E820Table};
use ruvm_firmware::smbios::{
    SMBIOS_ANCHOR_FILE, SMBIOS_TABLES_FILE, SmbiosConfig, SmbiosEntryPointType as SmbiosEp,
    SmbiosOptions, SmbiosPciDevice, SmbiosTopology, mem_array_from_e820, smbios_get_tables,
};
use ruvm_firmware::x86_linux::{X86KernelBoot, X86LinuxInput, x86_load_linux};
use ruvm_hw_acpi::{SystemRequest, SystemRequestHandler};
use ruvm_hw_char::serial::{SERIAL_BAUDBASE_DEFAULT, SERIAL_IO_SIZE, Serial, SerialBackend};
use ruvm_hw_core::fw_cfg::{
    DmaMemory, FW_CFG_CTL_SIZE, FW_CFG_DMA_SIZE, FW_CFG_IO_BASE, FW_CFG_MAX_CPUS, FW_CFG_NB_CPUS,
    FW_CFG_NUMA, FW_CFG_RAM_SIZE, FwCfgIo, FwCfgMachineConfig, FwCfgState, fw_cfg_init_io_dma,
};
use ruvm_hw_core::{Clock, IrqLine, IrqPin, irq};
use ruvm_hw_i2c::smbus_eeprom::SmbusEepromSlave;
use ruvm_hw_i2c::smbus_ich9::{Ich9Smbus, ich9_smbus_q35_init};
use ruvm_hw_input::pckbd::{I8042, I8042Props};
use ruvm_hw_intc::i8259::{I8259Pair, i8259_init};
use ruvm_hw_intc::ioapic::{
    IO_APIC_DEFAULT_ADDRESS, IOAPIC_NUM_PINS, IOAPIC_VER_DEF, IoApic, IoApicMsiHandler, IoApics,
};
use ruvm_hw_pci::q35::{Q35Config, Q35PciHost};
use ruvm_hw_pci::regs::{PCI_CLASS_DEVICE, PCI_HEADER_TYPE, PCI_HEADER_TYPE_BRIDGE};
use ruvm_hw_pci::{PCIE_BASE_ADDR_UNMAPPED, PciBus};
use ruvm_hw_storage::{BlockBackend, DriveConfig, DriveKind, ICH9_AHCI_PORTS, Ich9Ahci};
use ruvm_hw_timer::hpet::{HPET_BASE, HPET_LEN, Hpet, HpetFwConfig, HpetProperties};
use ruvm_hw_timer::i8254::I8254;
use ruvm_hw_timer::mc146818::{Mc146818Props, Mc146818Rtc};
use ruvm_mem::{AddressSpace, MemTxAttrs, MemorySystem, MmioOps, RamBlock, RegionId, RegionType};

use crate::ich9_lpc::{Ich9Lpc, Ich9LpcConfig, SmiHandler};
use crate::microvm::props::OnOffAuto;
use crate::microvm::{GuestRamRange, KernelConfig, OptionRom};
use crate::pc::{
    ACPI_BUILD_PCI_IRQS, FW_CFG_ACPI_TABLES, FW_CFG_HPET, FW_CFG_IRQ0_OVERRIDE, GIB, GsiHook,
    GsiHookSlot, GsiState, HdGeometry, ISA_BIOS_MAX, IoportF0, MIB, PC_FW_DATA, PC_ROM_MIN_VGA,
    PC_ROM_SIZE, PCSPK_IO_BASE, PORT92_IO_BASE, PcSpeaker, Port92, REG_EQUIPMENT_BYTE,
    UnassignedIo, WeakDma, boot_order_nibbles, cmos_init_disks, cmos_set_memory, err, hd_geometry,
    hd_geometry_guess, pci_hole64_start, rtc_ref_date, rtc_set_cpus_count, set_boot_dev,
};
use crate::pflash::{FlashDrive, Pflash, PflashBacking, pc_system_flash_map};

pub use props::{Q35Props, SmbiosEntryPointType};

/// `mc->name`.
pub const Q35_MACHINE_NAME: &str = "pc-q35-11.1";
/// `mc->alias`.
pub const Q35_MACHINE_ALIAS: &str = "q35";
/// The older q35 machine versions ruvm also builds, newest first. Their compat properties,
/// `hw_compat_11_0` and `hw_compat_10_2` (`pc_compat_11_0` and `pc_compat_10_2` are empty),
/// touch none of the devices ruvm has, so each is [`Q35_MACHINE_NAME`] under another name: the
/// name shows only in `-machine help` and in the configuration section of a migration stream.
pub const Q35_OLDER_MACHINE_NAMES: [&str; 2] = ["pc-q35-11.0", "pc-q35-10.2"];
/// `mc->desc`.
pub const Q35_DESC: &str = "Standard PC (Q35 + ICH9, 2009)";
/// `mc->max_cpus`.
pub const Q35_MAX_CPUS: u32 = 4096;
/// `mc->default_ram_id`.
pub const Q35_RAM_ID: &str = "pc.ram";
/// The default `-m` of x86 machines.
pub const Q35_DEFAULT_RAM_SIZE: u64 = 128 * MIB;
/// The default firmware, from `firmware=bios-256k.bin` in the default machine options.
pub const Q35_BIOS_FILENAME: &str = "bios-256k.bin";
/// The option ROM the kvmvapic device registers (`vapic_realize()`), which patches TPR
/// accesses in Windows XP era guests. It is added with bootindex -1, so it is loaded but not
/// put in the boot order.
pub const KVMVAPIC_ROM: &str = "kvmvapic.bin";
/// The default `-boot order=`.
pub const PC_DEFAULT_BOOT_ORDER: &str = "cad";
/// The default `phys-bits` of TCG CPUs.
pub const TCG_PHYS_ADDR_BITS: u32 = 40;
/// `AMD_HT_START`, the HyperTransport window below 1 TiB.
pub const AMD_HT_START: u64 = 0xfd_0000_0000;
/// `AMD_HT_END`.
pub const AMD_HT_END: u64 = 0xff_ffff_ffff;
/// `CPUID_HT`, the HTT bit of `CPUID[1].EDX`.
const CPUID_HT: u32 = 1 << 28;
/// `AMD_ABOVE_1TB_START`, where RAM above 4 GiB moves when it would overlap the
/// HyperTransport window.
pub const AMD_ABOVE_1TB_START: u64 = AMD_HT_END + 1;

/// What the board needs to know about the first CPU: `IS_AMD_CPU()` for the memory map, and
/// the CPUID signature and feature bits that SMBIOS type 4 reports.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CpuIdent {
    /// Whether the vendor is AuthenticAMD.
    pub amd: bool,
    /// `CPUID[1].EAX`.
    pub version: u32,
    /// `CPUID[1].EDX`.
    pub features_edx: u32,
}

impl CpuIdent {
    /// Works the identity out of `CPUID[0]` and `CPUID[1]`, each as EAX, EBX, ECX, EDX.
    pub fn from_cpuid(leaf0: [u32; 4], leaf1: [u32; 4]) -> CpuIdent {
        // "Auth" "enti" "cAMD" in EBX, EDX, ECX.
        let amd = leaf0[1] == 0x6874_7541 && leaf0[3] == 0x6974_6e65 && leaf0[2] == 0x444d_4163;
        CpuIdent { amd, version: leaf1[0], features_edx: leaf1[3] }
    }
}
/// The ports of the PIT.
pub const PIT_IO_BASE: u64 = 0x40;
/// The ports of the RTC.
pub const RTC_IO_BASE: u64 = 0x70;
/// The ISA IRQ of the RTC.
pub const RTC_IRQ: u32 = 8;
/// The ports of COM1.
pub const SERIAL_IO_BASE: u64 = 0x3f8;
/// The ISA IRQ of COM1.
pub const SERIAL_IRQ: u32 = 4;
/// `MAX_ISA_SERIAL_PORTS`: how many ISA serial ports `pc_superio_init()` creates at most.
pub const MAX_ISA_SERIAL_PORTS: usize = 4;
/// The ports of COM1 to COM4, `isa_serial_io[]`.
pub const ISA_SERIAL_IO: [u64; MAX_ISA_SERIAL_PORTS] = [0x3f8, 0x2f8, 0x3e8, 0x2e8];
/// The ISA IRQs of COM1 to COM4, `isa_serial_irq[]`.
pub const ISA_SERIAL_IRQ: [u32; MAX_ISA_SERIAL_PORTS] = [4, 3, 4, 3];
/// The i8042 data port.
pub const I8042_DATA_PORT: u64 = 0x60;
/// The i8042 command and status port.
pub const I8042_CMD_PORT: u64 = 0x64;
/// `hpet-intcap` for q35: IRQ 2, 8 and 16 to 23.
pub const Q35_HPET_INTCAP: u32 = 0xff_0104;
/// Where the AHCI controller goes, 00:1f.2.
pub const ICH9_SATA1_DEVFN: u8 = (0x1f << 3) | 2;
/// Where the LPC bridge sits, 00:1f.0.
pub const ICH9_LPC_DEVFN: u8 = 0x1f << 3;
/// `ACPI_PCIHP_ADDR_ICH9`.
pub const ACPI_PCIHP_ADDR_ICH9: u16 = 0x0cc0;
/// `ACPI_PCIHP_SIZE`.
pub const ACPI_PCIHP_SIZE: u16 = 0x18;
/// `ICH9_CPU_HOTPLUG_IO_BASE`.
pub const ICH9_CPU_HOTPLUG_IO_BASE: u16 = 0x0cd8;
/// `ICH9_LPC_SMI_F_CPU_HOTPLUG_BIT`.
const SMI_F_CPU_HOTPLUG_BIT: u32 = 1;
/// `ICH9_LPC_SMI_F_CPU_HOT_UNPLUG_BIT`.
const SMI_F_CPU_HOT_UNPLUG_BIT: u32 = 2;

/// Receives the A20 line: true when address bit 20 is passed through.
pub type A20Handler = Arc<dyn Fn(bool) + Send + Sync>;

/// A drive of one of the system flashes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PflashDrive {
    /// The block backend name QEMU prints in errors, "pflash0" or "pflash1".
    pub name: String,
    /// The image size as the block layer reports it, see
    /// [`raw_block_length`](crate::pflash::raw_block_length).
    pub size: u64,
    /// The image.
    pub backing: PflashBacking,
}

/// Everything [`Q35::new`] needs: the command line after parsing, with files already read.
pub struct Q35MachineConfig {
    /// `mc->name`: [`Q35_MACHINE_NAME`] or one of [`Q35_OLDER_MACHINE_NAMES`]. It is the
    /// version string of the SMBIOS system table.
    pub machine_name: &'static str,
    /// `-m`.
    pub ram_size: u64,
    /// `-smp cpus=`.
    pub cpus: u32,
    /// `-smp maxcpus=`, 0 for the same as `cpus`.
    pub max_cpus: u32,
    /// Whether the accelerator is KVM.
    pub kvm: bool,
    /// Whether the accelerator can run SMM. KVM says so through `KVM_CAP_X86_SMM`. In QEMU
    /// TCG always can; here it is [`crate::tcg_run::TCG_SMM_AVAILABLE`], since the x86 front
    /// end has no SMM yet.
    pub smm_available: bool,
    /// `phys-bits` of the CPU model, for the address space check of `pc_memory_init()`.
    pub phys_bits: u32,
    /// The first CPU. The default is a CPU that is not AMD, with zero CPUID values; note
    /// that QEMU's `qemu64` is AuthenticAMD under TCG, while KVM uses the host vendor.
    pub cpu: CpuIdent,
    /// The `-machine` properties.
    pub props: Q35Props,
    /// `-bios`, or `None` for [`Q35_BIOS_FILENAME`].
    pub firmware_name: Option<String>,
    /// The contents of the firmware file, `None` if it could not be found. Not used when
    /// pflash0 has a drive.
    pub firmware: Option<Vec<u8>>,
    /// The drives of the two system flashes, `-machine pflash0=,pflash1=` or
    /// `-drive if=pflash`. With a pflash0 drive the board maps CFI01 flashes below 4 GiB
    /// instead of loading the BIOS as ROM, which is how OVMF boots.
    pub pflash: [Option<PflashDrive>; 2],
    /// `-kernel` and friends.
    pub kernel: Option<KernelConfig>,
    /// `-option-rom`, in command line order.
    pub option_roms: Vec<OptionRom>,
    /// ROM files by name: the `-option-rom` files and the boot ROMs the kernel loader asks
    /// for (`linuxboot_dma.bin`, `pvh.bin`). A missing file gets QEMU's warning and is skipped.
    pub rom_files: BTreeMap<String, Vec<u8>>,
    /// Whether `serial_hd(i)` exists, for each `-serial` in order. `-serial none` leaves a
    /// hole. COM1 to COM4 are created for the first four that exist, and their chardevs are
    /// connected later with [`Q35::set_serial_backend`].
    pub serial_hds: Vec<bool>,
    /// `-boot order=`.
    pub boot_order: String,
    /// The `-smbios` options, in command line order. `-uuid` goes in [`Self::fw_cfg`].
    pub smbios: SmbiosOptions,
    /// The `-smp` topology the SMBIOS processor tables describe. `None` is what `-smp N`
    /// gives on current machine types: one socket with `max_cpus` cores.
    pub topology: Option<SmbiosTopology>,
    /// `-uuid`, `-boot` and the display options fw_cfg reports. `enable_graphics` is taken
    /// from the `graphics` property.
    pub fw_cfg: FwCfgMachineConfig,
    /// `QEMU_CLOCK_VIRTUAL`, for the timers.
    pub clock: Arc<Clock>,
    /// `rtc_clock`, the clock the RTC counts on.
    pub rtc_clock: Arc<Clock>,
    /// The date the RTC starts from, `-rtc base=`.
    pub rtc_date: SystemTime,
    /// `kvm_pit_in_kernel()`: KVM emulates the PIT and the speaker port, so the board leaves
    /// its own out.
    pub pit_in_kernel: bool,
    /// The RAM of `-machine memory-backend=`, used instead of a block of the board's own,
    /// with the backend's name. It has to be `ram_size` long.
    pub memdev: Option<Arc<RamBlock>>,
    /// `-machine aux-ram-share=`: the RAM and ROM the board makes are shared memory.
    pub aux_ram_share: bool,
}

impl fmt::Debug for Q35MachineConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Q35MachineConfig")
            .field("machine_name", &self.machine_name)
            .field("ram_size", &self.ram_size)
            .field("memdev", &self.memdev.as_ref().map(|b| b.name()))
            .field("aux_ram_share", &self.aux_ram_share)
            .field("cpus", &self.cpus)
            .field("max_cpus", &self.max_cpus)
            .field("kvm", &self.kvm)
            .field("props", &self.props)
            .field("firmware_name", &self.firmware_name)
            .field("pflash", &self.pflash)
            .field("kernel", &self.kernel.as_ref().map(|k| &k.filename))
            .field("option_roms", &self.option_roms)
            .field("serial_hds", &self.serial_hds)
            .field("boot_order", &self.boot_order)
            .finish_non_exhaustive()
    }
}

impl Default for Q35MachineConfig {
    /// 128 MiB, one CPU, TCG, default properties, no firmware bytes and clocks that only move
    /// when stepped.
    fn default() -> Self {
        Q35MachineConfig {
            machine_name: Q35_MACHINE_NAME,
            ram_size: Q35_DEFAULT_RAM_SIZE,
            cpus: 1,
            max_cpus: 0,
            kvm: false,
            smm_available: true,
            phys_bits: TCG_PHYS_ADDR_BITS,
            cpu: CpuIdent::default(),
            props: Q35Props::default(),
            firmware_name: None,
            firmware: None,
            pflash: [None, None],
            kernel: None,
            option_roms: Vec::new(),
            rom_files: BTreeMap::new(),
            serial_hds: vec![true],
            boot_order: PC_DEFAULT_BOOT_ORDER.to_string(),
            smbios: SmbiosOptions::new(),
            topology: None,
            fw_cfg: FwCfgMachineConfig::default(),
            clock: Clock::manual(ClockType::Virtual),
            rtc_clock: Clock::manual(ClockType::Host),
            rtc_date: SystemTime::now(),
            pit_in_kernel: false,
            memdev: None,
            aux_ram_share: false,
        }
    }
}

/// `pc_system_firmware_init()` for q35: the CFI01 flashes when pflash0 has a drive, else the
/// BIOS as ROM (`x86_bios_rom_init()`). Either way the top 128 KiB are aliased read only
/// below 1 MiB as "isa-bios". Returns `pc.bios` with its contents, or the flashes.
///
/// QEMU refuses pflash under KVM without `KVM_CAP_READONLY_MEM`. Every kernel ruvm runs on
/// has it (Linux 3.7 and later), so that check is left out.
#[allow(clippy::type_complexity)]
fn pc_system_firmware_init(
    mem: &Arc<MemorySystem>,
    rom_memory: RegionId,
    firmware_name: Option<String>,
    firmware: Option<Vec<u8>>,
    pflash: [Option<PflashDrive>; 2],
    props: &Q35Props,
) -> Result<(Option<(RegionId, Vec<u8>)>, Vec<Arc<Pflash>>), String> {
    let drives = [0, 1]
        .map(|i| pflash[i].as_ref().map(|d| FlashDrive { name: d.name.clone(), size: d.size }));
    let map = pc_system_flash_map(&drives, props.max_fw_size).map_err(|e| match e.info() {
        Some(info) => format!("{e}\ninfo: {info}"),
        None => e.to_string(),
    })?;
    if map.flashes.is_empty() {
        // x86_bios_rom_init(.., rom_memory, false)
        let bios_name = firmware_name.unwrap_or_else(|| Q35_BIOS_FILENAME.to_string());
        let bios_data = match firmware {
            Some(d) if !d.is_empty() && d.len() as u64 % 65536 == 0 => d,
            _ => return Err(format!("qemu: could not load PC BIOS '{bios_name}'")),
        };
        let bios_size = bios_data.len() as u64;
        let bios = mem.new_ram("pc.bios", bios_size).map_err(err)?;
        mem.set_readonly(bios, true).map_err(err)?;
        mem.add_subregion(rom_memory, (1u64 << 32) - bios_size, bios).map_err(err)?;
        let isa_bios_size = bios_size.min(ISA_BIOS_MAX);
        let isa_bios = mem
            .new_alias("isa-bios", bios, bios_size - isa_bios_size, isa_bios_size.into())
            .map_err(err)?;
        mem.add_subregion_overlap(rom_memory, MIB - isa_bios_size, isa_bios, 1).map_err(err)?;
        mem.set_readonly(isa_bios, true).map_err(err)?;
        return Ok((Some((bios, bios_data)), Vec::new()));
    }
    // pc_system_flash_map()
    let mut pflash = pflash;
    let mut flashes = Vec::new();
    for f in &map.flashes {
        let backing = pflash[f.index].take().map_or(PflashBacking::None, |d| d.backing);
        let dev = Pflash::new(mem, f.name, f.props, backing)?;
        mem.add_subregion(rom_memory, f.base, dev.region()).map_err(err)?;
        flashes.push(dev);
    }
    if let (Some(isa), Some(flash0)) = (map.isa_bios, flashes.first()) {
        let alias = mem
            .new_alias(isa.name, flash0.region(), isa.flash_offset, isa.size.into())
            .map_err(err)?;
        mem.set_readonly(alias, isa.readonly).map_err(err)?;
        mem.add_subregion_overlap(rom_memory, isa.addr, alias, isa.priority).map_err(err)?;
    }
    Ok((None, flashes))
}

/// The RAM split of `pc_q35_init()`: returns `(below_4g, above_4g)` and a warning QEMU would
/// print. `max_ram_below_4g` 0 means 4 GiB.
pub fn q35_ram_split(ram_size: u64, max_ram_below_4g: u64) -> (u64, u64, Option<String>) {
    let mut lowmem = if ram_size >= 0xb000_0000 { 0x8000_0000 } else { 0xb000_0000 };
    let max = if max_ram_below_4g == 0 { 4 * GIB } else { max_ram_below_4g };
    let mut warning = None;
    if lowmem > max {
        lowmem = max;
        if ram_size.saturating_sub(lowmem) > lowmem && lowmem & (GIB - 1) != 0 {
            warning = Some(format!(
                "There is possibly poor performance as the ram size  (0x{ram_size:x}) is more \
                 then twice the size of max-ram-below-4g ({max}) and max-ram-below-4g is not a \
                 multiple of 1G."
            ));
        }
    }
    if ram_size >= lowmem { (lowmem, ram_size - lowmem, warning) } else { (ram_size, 0, warning) }
}

/// What the kernel loader left for reset time: ELF segments of a PVH kernel.
#[derive(Clone, Debug)]
struct RomBlob {
    addr: u64,
    data: Vec<u8>,
    size: u64,
}

/// AHCI DMA through a weak reference to system memory.
struct WeakAhciDma(Weak<AddressSpace>);

impl ruvm_hw_storage::DmaMemory for WeakAhciDma {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.0.upgrade().is_some_and(|a| a.read(addr, MemTxAttrs::UNSPECIFIED, buf).is_ok())
    }

    fn dma_write(&self, addr: u64, buf: &[u8]) -> bool {
        self.0.upgrade().is_some_and(|a| a.write(addr, MemTxAttrs::UNSPECIFIED, buf).is_ok())
    }
}

/// `a20_line` from `pc_superio_init()`: both the i8042 and port 0x92 drive the CPU's A20 mask
/// and the last write wins.
struct A20Line {
    level: AtomicBool,
    handler: RwLock<Option<A20Handler>>,
}

impl A20Line {
    fn set(&self, level: bool) {
        self.level.store(level, Ordering::SeqCst);
        let h = self.handler.read().unwrap_or_else(PoisonError::into_inner).clone();
        if let Some(h) = h {
            h(level);
        }
    }
}

/// The live state the ACPI tables are built from, shared with the fw_cfg callbacks.
struct AcpiSource {
    host: Arc<Q35PciHost>,
    lpc: Arc<Ich9Lpc>,
    props: Q35Props,
    cpus: u32,
    max_cpus: u32,
    pic: bool,
    i8042: bool,
    hpet: bool,
    /// The indexes of the ISA serial ports that exist, COM1 being 0.
    serials: Vec<usize>,
}

impl AcpiSource {
    fn pci_devices(&self, bus: &PciBus) -> Vec<AcpiPciDevice> {
        let children = bus.children();
        let mut devs = bus.devices();
        devs.sort_by_key(|d| d.devfn());
        devs.iter()
            .map(|d| {
                let devfn = d.devfn();
                let (class, header) = d.with_config(|c| {
                    (
                        u16::from_le_bytes([
                            c.config[PCI_CLASS_DEVICE],
                            c.config[PCI_CLASS_DEVICE + 1],
                        ]),
                        c.config[PCI_HEADER_TYPE] & 0x7f,
                    )
                });
                let aml = if bus.parent_dev().is_none() && devfn == ICH9_LPC_DEVFN {
                    let mut isa = Vec::new();
                    // qbus_build_aml() walks the ISA bus newest first. pc_superio_init()
                    // creates the serial ports in index order and the i8042 after them, and
                    // the RTC came with the LPC bridge before all of them.
                    if self.i8042 {
                        isa.push(IsaDevice::I8042 { kbd_irq: 1, mouse_irq: 12 });
                    }
                    for &index in self.serials.iter().rev() {
                        isa.push(IsaDevice::Serial {
                            index: index as u32,
                            iobase: ISA_SERIAL_IO[index] as u16,
                            irq: ISA_SERIAL_IRQ[index] as u8,
                        });
                    }
                    isa.push(IsaDevice::Rtc { io_base: RTC_IO_BASE as u16, irq: RTC_IRQ as u8 });
                    PciDeviceAml::Lpc { isa }
                } else if class == 0x0300 {
                    PciDeviceAml::Vga { qxl: false }
                } else if header == PCI_HEADER_TYPE_BRIDGE {
                    let child = children
                        .iter()
                        .find(|b| b.parent_dev().is_some_and(|p| Arc::ptr_eq(&p, d)));
                    let devices = child.map(|b| self.pci_devices(b)).unwrap_or_default();
                    PciDeviceAml::Bridge { devices }
                } else {
                    PciDeviceAml::Plain
                };
                AcpiPciDevice { devfn, acpi_index: None, aml }
            })
            .collect()
    }

    fn input(&self) -> Q35Acpi {
        let pm = self.lpc.pm();
        let pm_props = *pm.props();
        let negotiated = self.lpc.smi_negotiated_features();
        let mcfg_base = self.host.mcfg_base();
        let mcfg = (mcfg_base != PCIE_BASE_ADDR_UNMAPPED)
            .then(|| McfgInfo { base: mcfg_base, size: self.host.mcfg_size() });
        let hole_start = u64::from(self.host.pci_hole_start());
        let hole_end = u64::from(self.host.pci_hole_end());
        let hole64_start = self.host.pci_hole64_start();
        let hole64_end = self.host.pci_hole64_end();
        Q35Acpi {
            oem_id: self.props.oem_id.clone(),
            oem_table_id: self.props.oem_table_id.clone(),
            madt: MadtConfig {
                cpus: (0..self.max_cpus)
                    .map(|i| PossibleCpu { arch_id: i, present: i < self.cpus })
                    .collect(),
                pic: self.pic,
                ioapic2: false,
                apic_xrupt_override: true,
                pci_irq_mask: ACPI_BUILD_PCI_IRQS,
            },
            max_cpus: self.max_cpus,
            smm: pm_props.smm_enabled || pm_props.smm_compat,
            sci_int: u16::from(self.lpc.sci_gsi()),
            pm_io_base: pm.pm_io_base() as u16,
            i8042: self.i8042,
            hpet: self.hpet,
            mcfg,
            pci_root_uid: 0,
            pcihp_bridge: true,
            pcihp_io_base: ACPI_PCIHP_ADDR_ICH9,
            pcihp_io_len: ACPI_PCIHP_SIZE,
            smi_on_cpuhp: negotiated & (1 << SMI_F_CPU_HOTPLUG_BIT) != 0,
            smi_on_cpu_unplug: negotiated & (1 << SMI_F_CPU_HOT_UNPLUG_BIT) != 0,
            cpu_hp_io_base: ICH9_CPU_HOTPLUG_IO_BASE,
            s3_disabled: pm_props.disable_s3,
            s4_disabled: pm_props.disable_s4,
            s4_val: pm_props.s4_val,
            pci_hole: CrsRange { base: hole_start, limit: hole_end.saturating_sub(1) },
            pci_hole64: (hole64_end > hole64_start)
                .then(|| CrsRange { base: hole64_start, limit: hole64_end - 1 }),
            pci_devices: self.pci_devices(self.host.bus()),
        }
    }
}

/// `AcpiBuildState`: whether the tables were rebuilt since the last reset, and the blobs.
#[derive(Default)]
struct AcpiCache {
    patched: bool,
    table: Vec<u8>,
    loader: Vec<u8>,
    rsdp: Vec<u8>,
}

/// Which blob a select callback copies.
#[derive(Copy, Clone)]
enum AcpiBlob {
    Table,
    Loader,
    Rsdp,
}

/// `acpi_build_update()` for one of the three files: rebuilds all three once after each reset
/// and copies the one that was selected.
fn acpi_build_update(src: &AcpiSource, cache: &Mutex<AcpiCache>, which: AcpiBlob, buf: &mut [u8]) {
    let mut c = cache.lock().unwrap_or_else(PoisonError::into_inner);
    if !c.patched {
        c.patched = true;
        let t = acpi_q35::build(&src.input());
        c.table = t.table_data;
        c.loader = t.linker.cmd_blob().to_vec();
        c.rsdp = t.rsdp;
    }
    let data = match which {
        AcpiBlob::Table => &c.table,
        AcpiBlob::Loader => &c.loader,
        AcpiBlob::Rsdp => &c.rsdp,
    };
    // acpi_ram_update(): the sizes do not change, the tables are padded.
    let n = buf.len().min(data.len());
    buf[..n].copy_from_slice(&data[..n]);
}

/// A drive plugged with [`Q35::attach_drive`], what the late CMOS setup needs.
#[derive(Copy, Clone, Debug)]
struct PluggedDrive {
    kind: DriveKind,
    sectors: u64,
    geometry: Option<(u32, u32, u32)>,
}

/// A q35 board.
pub struct Q35 {
    machine_name: &'static str,
    ram_size: u64,
    below_4g_mem_size: u64,
    above_4g_mem_size: u64,
    cpus: u32,
    max_cpus: u32,
    props: Q35Props,
    cpu: CpuIdent,
    /// `-uuid`, if one was given.
    uuid: Option<[u8; 16]>,
    smbios: SmbiosOptions,
    topology: SmbiosTopology,
    kvm: bool,
    smm_enabled: bool,
    vmport: bool,
    boot_devices: String,

    mem: Arc<MemorySystem>,
    system: RegionId,
    io: RegionId,
    pci: RegionId,
    ram: RegionId,
    /// `pc.rom`, the option ROM area.
    option_rom: RegionId,
    /// The PCI devices' option ROMs, `DEVICE/TYPE.rom`.
    device_roms: Vec<RegionId>,
    memory_as: Arc<AddressSpace>,
    io_as: Arc<AddressSpace>,
    smm_root: Option<RegionId>,
    smm_as: Option<Arc<AddressSpace>>,
    /// `pc.bios`, when the firmware is ROM rather than pflash.
    bios: Option<(RegionId, Vec<u8>)>,
    /// pflash0 and pflash1, when pflash0 has a drive.
    flashes: Vec<Arc<Pflash>>,
    roms: Vec<RomBlob>,

    fw_cfg: FwCfgIo,
    e820: E820Table,
    boot_order: Vec<(i32, String)>,
    warnings: Vec<String>,
    kernel: Option<X86KernelBoot>,

    host: Arc<Q35PciHost>,
    lpc: Arc<Ich9Lpc>,
    ahci: Option<Arc<Ich9Ahci>>,
    /// The ICH9 SMBus controller at 00:1f.3 and its eight empty SPD EEPROMs.
    smbus: Option<(Arc<Ich9Smbus>, Vec<Arc<SmbusEepromSlave>>)>,
    drives: Mutex<[Option<PluggedDrive>; ICH9_AHCI_PORTS]>,
    gsi: Vec<IrqLine>,
    ioapics: IoApics,
    ioapic: Arc<IoApic>,
    msi_hook: Arc<RwLock<Option<IoApicMsiHandler>>>,
    gsi_hook: GsiHookSlot,
    request_hook: Arc<RwLock<Option<SystemRequestHandler>>>,
    pic: Option<I8259Pair>,
    pic_output: Arc<IrqPin>,
    pit: Option<Arc<I8254>>,
    pcspk: Option<Arc<PcSpeaker>>,
    hpet: Option<Arc<Hpet>>,
    hpet_fw: HpetFwConfig,
    rtc: Arc<Mc146818Rtc>,
    /// COM1 to COM4 by index, `None` where `serial_hd(i)` does not exist.
    serials: Vec<Option<Arc<Serial>>>,
    i8042: Option<Arc<I8042>>,
    port92: Option<Arc<Port92>>,
    a20: Arc<A20Line>,
    acpi_cache: Arc<Mutex<AcpiCache>>,
    acpi_src: Arc<AcpiSource>,
    /// What devices outside the board reset with it.
    reset_hooks: Mutex<Vec<ResetHook>>,

    done: bool,
}

/// A device's part of a system reset, for devices the board does not own.
pub type ResetHook = Box<dyn Fn() + Send + Sync>;

impl fmt::Debug for Q35 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Q35")
            .field("ram_size", &self.ram_size)
            .field("below_4g_mem_size", &self.below_4g_mem_size)
            .field("above_4g_mem_size", &self.above_4g_mem_size)
            .field("cpus", &self.cpus)
            .field("max_cpus", &self.max_cpus)
            .field("props", &self.props)
            .field("smm_enabled", &self.smm_enabled)
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}

impl Q35 {
    /// `pc_q35_init()`: checks the configuration, builds memory, loads the firmware and kernel
    /// and creates the devices. Errors carry QEMU's message.
    pub fn new(cfg: Q35MachineConfig) -> Result<Q35, String> {
        let Q35MachineConfig {
            machine_name,
            ram_size,
            cpus,
            max_cpus,
            kvm,
            smm_available,
            phys_bits,
            cpu,
            props,
            firmware_name,
            firmware,
            pflash,
            kernel,
            option_roms,
            rom_files,
            serial_hds,
            boot_order: boot_devices,
            smbios,
            topology,
            fw_cfg: mut fw_cfg_cfg,
            clock,
            rtc_clock,
            rtc_date,
            pit_in_kernel,
            memdev,
            aux_ram_share,
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
        if max_cpus > Q35_MAX_CPUS {
            return Err(format!(
                "Invalid SMP CPUs {max_cpus}. The max CPUs supported by machine \
                 '{machine_name}' is {Q35_MAX_CPUS}"
            ));
        }
        if props.usb {
            return Err("q35: usb=on (ICH9 UHCI/EHCI) is not supported yet".to_string());
        }
        if props.wdat {
            return Err(
                "q35: wdat=on (the ICH9 TCO watchdog table) is not supported yet".to_string()
            );
        }
        let smm_enabled = props.smm_enabled(smm_available)?;
        // pc_basic_device_init(): resolve vmport.
        let vmport = match props.vmport {
            OnOffAuto::Auto => props.i8042,
            OnOffAuto::On => true,
            OnOffAuto::Off => false,
        };
        if !props.i8042 && vmport {
            return Err("vmport requires the i8042 controller to be enabled".to_string());
        }
        let mut warnings = Vec::new();

        let (below_4g_mem_size, above_4g_mem_size, warning) =
            q35_ram_split(ram_size, props.max_ram_below_4g);
        warnings.extend(warning);
        let mut above_4g_mem_start = 1u64 << 32;
        let mut hole64_start = pci_hole64_start(above_4g_mem_start, above_4g_mem_size);

        let mem = Arc::new(MemorySystem::new());
        mem.set_aux_ram_share(aux_ram_share);
        let system = mem.new_container("system", 1 << 64).map_err(err)?;
        let io = mem.new_io("io", 1 << 16, Arc::new(UnassignedIo)).map_err(err)?;
        let memory_as = mem.address_space_init(system, "memory").map_err(err)?;
        let io_as = mem.address_space_init(io, "I/O").map_err(err)?;
        let pci = mem.new_container("pci", 1 << 64).map_err(err)?;

        // pc_memory_init()
        let host_cfg_pci_hole64_size = Q35Config::new(pci, pci, system, io).pci_hole64_size;
        let mut e820 = E820Table::new();
        // The HyperTransport window near 1 TiB exists only on AMD hosts, so RAM above 4 GiB
        // moves past it (and the window is advertised) only for AMD CPUs.
        if cpu.amd {
            if hole64_start + host_cfg_pci_hole64_size > AMD_HT_START {
                above_4g_mem_start = AMD_ABOVE_1TB_START;
                hole64_start = pci_hole64_start(above_4g_mem_start, above_4g_mem_size);
            }
            if phys_bits >= 40 {
                e820.add_entry(AMD_HT_START, AMD_HT_END - AMD_HT_START + 1, E820_RESERVED);
            }
        }
        let maxusedaddr = hole64_start + host_cfg_pci_hole64_size - 1;
        let maxphysaddr = if phys_bits >= 64 { u64::MAX } else { (1u64 << phys_bits) - 1 };
        if maxphysaddr < maxusedaddr {
            return Err(format!(
                "Address space limit 0x{maxphysaddr:x} < 0x{maxusedaddr:x} phys-bits too low \
                 ({phys_bits})"
            ));
        }
        // machine_consume_memdev(): the backend's region, named after the backend.
        let ram = match memdev {
            Some(block) => mem.new_ram_from_block(block),
            None => mem.new_ram(Q35_RAM_ID, ram_size),
        }
        .map_err(err)?;
        let below = mem.new_alias("ram-below-4g", ram, 0, below_4g_mem_size.into()).map_err(err)?;
        mem.add_subregion(system, 0, below).map_err(err)?;
        e820.add_entry(0, below_4g_mem_size, E820_RAM);
        if above_4g_mem_size > 0 {
            let above = mem
                .new_alias("ram-above-4g", ram, below_4g_mem_size, above_4g_mem_size.into())
                .map_err(err)?;
            mem.add_subregion(system, above_4g_mem_start, above).map_err(err)?;
            e820.add_entry(above_4g_mem_start, above_4g_mem_size, E820_RAM);
        }

        let (bios, flashes) =
            pc_system_firmware_init(&mem, pci, firmware_name, firmware, pflash, &props)?;

        let option_rom_mr = mem.new_ram("pc.rom", PC_ROM_SIZE).map_err(err)?;
        mem.set_readonly(option_rom_mr, true).map_err(err)?;
        mem.add_subregion_overlap(pci, PC_ROM_MIN_VGA, option_rom_mr, 1).map_err(err)?;

        // fw_cfg_arch_create()
        fw_cfg_cfg.enable_graphics = props.graphics;
        // qemu_uuid_set: an all-zero UUID gives the same tables as none.
        let uuid = (fw_cfg_cfg.uuid != [0; 16]).then_some(fw_cfg_cfg.uuid);
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
        let fwc = Arc::clone(fw_cfg.state());
        // The APIC ID limit: IDs are CPU indexes with a flat topology.
        let apic_id_limit = max_cpus;
        fwc.add_i16(FW_CFG_NB_CPUS, cpus as u16);
        fwc.add_i16(FW_CFG_MAX_CPUS, apic_id_limit as u16);
        fwc.add_i64(FW_CFG_RAM_SIZE, ram_size);
        fwc.add_bytes(FW_CFG_ACPI_TABLES, Vec::new());
        fwc.add_i32(FW_CFG_IRQ0_OVERRIDE, 1);
        // FW_CFG_HPET is added once the HPET exists, so it has its final contents.
        fwc.add_bytes(FW_CFG_NUMA, vec![0; (1 + apic_id_limit as usize) * 8]);

        // x86_load_linux() and the option ROMs. The `-option-rom` files come first, then the
        // one the kvmvapic device asks for when the APIC is realized (it needs at least 1 MiB
        // of RAM to map it), then the kernel's boot ROM.
        let mut option_roms = option_roms;
        if ram_size >= 1 << 20 {
            option_roms.push(OptionRom { name: KVMVAPIC_ROM.to_string(), bootindex: -1 });
        }
        let mut roms = Vec::new();
        let mut kernel_boot = None;
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
                acpi_data_size: PC_FW_DATA,
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
                    return Err("multiboot kernels are not supported on q35 yet".to_string());
                }
            };
            option_roms.push(OptionRom { name: rom.to_string(), bootindex: 0 });
            kernel_boot = Some(boot);
        }
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

        // The host bridge, which maps "pci" under system memory and 0xcf8/0xcfc.
        let mut host_cfg = Q35Config::new(ram, pci, system, io);
        host_cfg.below_4g_mem_size = below_4g_mem_size;
        host_cfg.above_4g_mem_size = above_4g_mem_size;
        host_cfg.has_smm_ranges = smm_enabled;
        host_cfg.pc_pci_hole64_start = hole64_start;
        let host = Arc::new(Q35PciHost::new(Arc::clone(&mem), host_cfg).map_err(err)?);

        // The SMM address space of the CPUs, x86_cpu_machine_done() and
        // register_smram_listener().
        let (smm_root, smm_as) = match host.smram() {
            Some(smram) => {
                let root = mem.new_container("memory-smm", 1 << 64).map_err(err)?;
                let alias = mem.new_alias("smm-memory", system, 0, 1 << 64).map_err(err)?;
                mem.add_subregion_overlap(root, 0, alias, 0).map_err(err)?;
                mem.add_subregion_overlap(root, 0, smram, 10).map_err(err)?;
                let a = mem.address_space_init(root, "smm").map_err(err)?;
                (Some(root), Some(a))
            }
            None => (None, None),
        };

        // The interrupt controllers.
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
        let ioapic =
            IoApic::realize(&clock, IOAPIC_VER_DEF, &ioapics, Arc::clone(&msi)).map_err(err)?;
        if let Some(p) = &pic {
            let master = Arc::clone(&p.master);
            ioapic.set_pic_read_irq(Some(Arc::new(move || master.pic_read_irq())));
        }
        {
            let r = mem.new_io("ioapic", 0x1000, ioapic.clone()).map_err(err)?;
            mem.add_subregion(system, IO_APIC_DEFAULT_ADDRESS, r).map_err(err)?;
        }
        let gsi_hook: GsiHookSlot = Arc::new(RwLock::new(None));
        let gsi_state = Arc::new(GsiState {
            i8259: pic.as_ref().map(|p| p.irq_set.clone()).unwrap_or_default(),
            ioapic: ioapic.inputs(),
            ioapic2: Vec::new(),
            hook: Arc::clone(&gsi_hook),
        });
        let gsi = irq::allocate(
            Arc::new(move |n, level| gsi_state.set(n, level)),
            IOAPIC_NUM_PINS as u32,
        );
        {
            let m = Arc::clone(&msi);
            host.bus().set_msi_handler(Some(Arc::new(move |addr, data| m(addr, data))));
        }

        // System requests from the LPC, the i8042 and port 0x92.
        let request_hook: Arc<RwLock<Option<SystemRequestHandler>>> = Arc::new(RwLock::new(None));
        let request: SystemRequestHandler = {
            let hook = Arc::clone(&request_hook);
            Arc::new(move |req| {
                let h = hook.read().unwrap_or_else(PoisonError::into_inner).clone();
                if let Some(h) = h {
                    h(req);
                }
            })
        };

        // The LPC bridge and its power management block.
        let mut lpc_cfg = Ich9LpcConfig::new(system, io);
        lpc_cfg.pm.smm_enabled = smm_enabled;
        let lpc = Arc::new(
            Ich9Lpc::new(Arc::clone(&mem), host.bus(), Arc::clone(&clock), lpc_cfg, &gsi)
                .map_err(err)?,
        );
        lpc.set_request_handler(Arc::clone(&request));
        // acpi_pm1_cnt_init() and ich9_lpc_pm_init().
        fwc.add_file("etc/system-states", lpc.pm().system_states().to_vec()).map_err(err)?;
        if lpc.smi_host_features() != 0 {
            fwc.add_file("etc/smi/supported-features", lpc.smi_host_features_le().to_vec())
                .map_err(err)?;
            let (l1, l2) = (Arc::clone(&lpc), Arc::clone(&lpc));
            fwc.add_file_callback(
                "etc/smi/requested-features",
                Some(Box::new(move |buf: &mut [u8]| {
                    let v = l1.smi_guest_features_le();
                    let n = buf.len().min(v.len());
                    buf[..n].copy_from_slice(&v[..n]);
                })),
                Some(Box::new(move |data: &[u8], _off: u32, _len: u32| {
                    let mut le = [0u8; 8];
                    let n = data.len().min(8);
                    le[..n].copy_from_slice(&data[..n]);
                    l2.set_smi_guest_features(le);
                })),
                vec![0; 8],
                false,
            )
            .map_err(err)?;
            let l3 = Arc::clone(&lpc);
            fwc.add_file_callback(
                "etc/smi/features-ok",
                Some(Box::new(move |buf: &mut [u8]| {
                    l3.smi_features_ok_select();
                    if let Some(b) = buf.first_mut() {
                        *b = l3.smi_features_ok();
                    }
                })),
                None,
                vec![0],
                true,
            )
            .map_err(err)?;
        }

        // The RTC inside the LPC bridge, which ich9_lpc_realize() gives base_year 2000.
        let rtc = {
            let date = rtc_ref_date(rtc_date, &rtc_clock);
            let props = Mc146818Props { base_year: 2000, ..Mc146818Props::default() };
            let s = Arc::new(Mc146818Rtc::new(props, rtc_clock, date).map_err(err)?);
            let r = mem.new_io("rtc", 2, s.clone()).map_err(err)?;
            mem.add_subregion(io, RTC_IO_BASE, r).map_err(err)?;
            s
        };

        // pc_basic_device_init()
        let ioport80 = mem.new_io("ioport80", 1, Arc::new(UnassignedIo)).map_err(err)?;
        mem.add_subregion(io, 0x80, ioport80).map_err(err)?;
        let ioport_f0 =
            mem.new_io("ioportF0", 1, Arc::new(IoportF0::new(gsi[13].clone()))).map_err(err)?;
        mem.add_subregion(io, 0xf0, ioport_f0).map_err(err)?;

        let mut hpet_fw = HpetFwConfig::new();
        let hpet = if props.hpet {
            let hp = HpetProperties { intcap: Q35_HPET_INTCAP, ..HpetProperties::default() };
            let h = Hpet::realize(&clock, hp, &mut hpet_fw, HPET_BASE).map_err(err)?;
            let r = mem.new_io("hpet", HPET_LEN.into(), h.clone()).map_err(err)?;
            mem.add_subregion(system, HPET_BASE, r).map_err(err)?;
            for (i, line) in gsi.iter().enumerate() {
                h.irq(i).connect(line.clone());
            }
            let m = Arc::clone(&msi);
            h.set_msi_handler(Some(Arc::new(move |addr, data| m(addr, data))));
            // The table has its final contents after the first reset.
            h.reset(&mut hpet_fw);
            Some(h)
        } else {
            None
        };
        fwc.add_bytes(FW_CFG_HPET, hpet_fw.to_bytes());
        match &hpet {
            // Overwrites the connection the south bridge made.
            Some(h) => rtc.connect_irq(h.legacy_irq_in(1)),
            None => rtc.connect_irq(gsi[RTC_IRQ as usize].clone()),
        }

        // pc_superio_init(): serial_hds_isa_init(isa_bus, 0, MAX_ISA_SERIAL_PORTS).
        let mut serials = Vec::with_capacity(MAX_ISA_SERIAL_PORTS);
        for index in 0..MAX_ISA_SERIAL_PORTS {
            if !serial_hds.get(index).copied().unwrap_or(false) {
                serials.push(None);
                continue;
            }
            let s = Serial::new(Arc::clone(&clock), SERIAL_BAUDBASE_DEFAULT, None);
            s.irq().connect(gsi[ISA_SERIAL_IRQ[index] as usize].clone());
            let r = mem.new_io("serial", SERIAL_IO_SIZE.into(), s.clone()).map_err(err)?;
            mem.add_subregion(io, ISA_SERIAL_IO[index], r).map_err(err)?;
            serials.push(Some(s));
        }
        let a20 = Arc::new(A20Line { level: AtomicBool::new(true), handler: RwLock::new(None) });
        let (i8042, port92) = if props.i8042 {
            let k = I8042::new(Arc::clone(&clock), I8042Props::default()).map_err(err)?;
            let d = mem.new_io("i8042-data", 1, k.data_io()).map_err(err)?;
            mem.add_subregion(io, I8042_DATA_PORT, d).map_err(err)?;
            let c = mem.new_io("i8042-cmd", 1, k.cmd_io()).map_err(err)?;
            mem.add_subregion(io, I8042_CMD_PORT, c).map_err(err)?;
            k.kbd_irq().connect(gsi[1].clone());
            k.mouse_irq().connect(gsi[12].clone());
            let r = Arc::clone(&request);
            k.reset_out().connect(IrqLine::from_fn(move |level| {
                if level != 0 {
                    r(SystemRequest::Reset);
                }
            }));
            let a = Arc::clone(&a20);
            k.a20_out().connect(IrqLine::from_fn(move |level| a.set(level != 0)));

            let r = Arc::clone(&request);
            let p = Port92::new(Arc::new(move || r(SystemRequest::Reset)));
            let a = Arc::clone(&a20);
            p.a20_out().connect(IrqLine::from_fn(move |level| a.set(level != 0)));
            let pr = mem.new_io("port92", 1, p.clone()).map_err(err)?;
            mem.add_subregion(io, PORT92_IO_BASE, pr).map_err(err)?;
            (Some(k), Some(p))
        } else {
            (None, None)
        };

        let (pit, pcspk) = if props.pit != OnOffAuto::Off && !pit_in_kernel {
            let p = I8254::new(&clock, PIT_IO_BASE as u32);
            match &hpet {
                Some(h) => {
                    p.irq.connect(h.legacy_irq_in(0));
                    h.pit_enabled().connect(p.irq_control_in());
                }
                None => p.irq.connect(gsi[0].clone()),
            }
            let r = mem.new_io("pit", 4, p.clone()).map_err(err)?;
            mem.add_subregion(io, PIT_IO_BASE, r).map_err(err)?;
            let spk = PcSpeaker::new(Arc::clone(&p));
            let r = mem.new_io("pcspk", 1, spk.clone()).map_err(err)?;
            mem.add_subregion(io, PCSPK_IO_BASE, r).map_err(err)?;
            (Some(p), Some(spk))
        } else {
            (None, None)
        };

        // The ICH9 AHCI controller, which takes the MSI path set above.
        let ahci = if props.sata {
            let dma: Arc<dyn ruvm_hw_storage::DmaMemory> =
                Arc::new(WeakAhciDma(Arc::downgrade(&memory_as)));
            Some(
                Ich9Ahci::new_multifunction(host.bus(), dma, Some(ICH9_SATA1_DEVFN))
                    .map_err(err)?,
            )
        } else {
            None
        };
        // The SMBus controller with blank SPD EEPROMs; QEMU still leaves their data empty.
        let smbus =
            if props.smbus { Some(ich9_smbus_q35_init(host.bus()).map_err(err)?) } else { None };

        let acpi_src = Arc::new(AcpiSource {
            host: Arc::clone(&host),
            lpc: Arc::clone(&lpc),
            props: props.clone(),
            cpus,
            max_cpus,
            pic: pic.is_some(),
            i8042: i8042.is_some(),
            hpet: hpet.is_some(),
            serials: (0..MAX_ISA_SERIAL_PORTS).filter(|&i| serials[i].is_some()).collect(),
        });

        Ok(Q35 {
            machine_name,
            ram_size,
            below_4g_mem_size,
            above_4g_mem_size,
            cpus,
            max_cpus,
            props,
            cpu,
            uuid,
            smbios,
            topology: topology
                .unwrap_or(SmbiosTopology { cores: max_cpus, ..SmbiosTopology::default() }),
            kvm,
            smm_enabled,
            vmport,
            boot_devices,
            mem,
            system,
            io,
            pci,
            ram,
            option_rom: option_rom_mr,
            device_roms: Vec::new(),
            memory_as,
            io_as,
            smm_root,
            smm_as,
            bios,
            flashes,
            roms,
            fw_cfg,
            e820,
            boot_order,
            warnings,
            kernel: kernel_boot,
            host,
            lpc,
            ahci,
            smbus,
            drives: Mutex::new([None; ICH9_AHCI_PORTS]),
            gsi,
            ioapics,
            ioapic,
            msi_hook,
            gsi_hook,
            request_hook,
            pic,
            pic_output,
            pit,
            pcspk,
            hpet,
            hpet_fw,
            rtc,
            serials,
            i8042,
            port92,
            a20,
            acpi_cache: Arc::new(Mutex::new(AcpiCache::default())),
            acpi_src,
            reset_hooks: Mutex::new(Vec::new()),
            done: false,
        })
    }

    /// Plugs a drive the way `-drive if=ide,index=N` does on q35: one unit per bus, so index
    /// N is AHCI port N. A hard disk needs a backend; a CD-ROM drive without one has no medium.
    pub fn attach_drive(
        &self,
        index: usize,
        config: DriveConfig,
        blk: Option<Arc<dyn BlockBackend>>,
    ) -> Result<(), String> {
        if self.done {
            return Err("drives must be plugged before machine_done".to_string());
        }
        let Some(ahci) = &self.ahci else {
            return Err(format!("machine type does not support if=ide,bus={index},unit=0"));
        };
        if index >= ICH9_AHCI_PORTS {
            return Err(format!("machine type does not support if=ide,bus={index},unit=0"));
        }
        let (kind, geometry) = (config.kind, config.geometry);
        let sectors = blk.as_ref().map_or(0, |b| b.len() / 512);
        ahci.attach_drive(index, config, blk).map_err(err)?;
        self.drives.lock().unwrap_or_else(PoisonError::into_inner)[index] =
            Some(PluggedDrive { kind, sectors, geometry });
        Ok(())
    }

    /// Connects the chardev of the ISA serial port `index`, 0 being COM1. Returns false if
    /// there is no such port.
    pub fn set_serial_backend(
        &self,
        index: usize,
        backend: Option<Arc<dyn SerialBackend>>,
    ) -> bool {
        match self.serial(index) {
            Some(s) => {
                s.set_backend(backend);
                true
            }
            None => false,
        }
    }

    /// The input of the q35 ACPI builder, from the current device state. The tables the guest
    /// reads are rebuilt from this on the first read after each reset, `acpi_build_update()`.
    pub fn acpi_input(&self) -> Q35Acpi {
        self.acpi_src.input()
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

    /// `acpi_setup()`: adds the three ACPI files with callbacks that rebuild them on first read.
    fn acpi_setup(&self) -> Result<(), String> {
        let fwc = self.fw_cfg.state();
        let t = acpi_q35::build(&self.acpi_input());
        let loader = t.linker.cmd_blob().to_vec();
        {
            let mut c = self.acpi_cache.lock().unwrap_or_else(PoisonError::into_inner);
            c.table = t.table_data.clone();
            c.loader = loader.clone();
            c.rsdp = t.rsdp.clone();
        }
        let add = |name: &str, which: AcpiBlob, data: Vec<u8>| {
            let src = Arc::clone(&self.acpi_src);
            let cache = Arc::clone(&self.acpi_cache);
            fwc.add_file_callback(
                name,
                Some(Box::new(move |buf: &mut [u8]| acpi_build_update(&src, &cache, which, buf))),
                None,
                data,
                true,
            )
            .map_err(err)
        };
        add(TABLE_FILE, AcpiBlob::Table, t.table_data)?;
        add(LOADER_FILE, AcpiBlob::Loader, loader)?;
        // QEMU built with TPM support always adds the event log, empty without a TPM.
        fwc.add_file(TPMLOG_FILE, Vec::new()).map_err(err)?;
        add(RSDP_FILE, AcpiBlob::Rsdp, t.rsdp)?;
        Ok(())
    }

    /// `fw_cfg_build_smbios()`: the SMBIOS tables and their entry point, for the firmware to
    /// install.
    fn build_smbios(&self) -> Result<(), String> {
        let mut cfg = SmbiosConfig::q35();
        if let Some(d) = cfg.defaults.as_mut() {
            d.version = self.machine_name.to_string();
        }
        cfg.uuid = self.uuid;
        cfg.ep_type = match self.props.smbios_entry_point_type {
            SmbiosEntryPointType::Ep32 => SmbiosEp::Ep32,
            SmbiosEntryPointType::Ep64 => SmbiosEp::Ep64,
            SmbiosEntryPointType::Auto => SmbiosEp::Auto,
        };
        cfg.topology = self.topology;
        cfg.cpuid_version = self.cpu.version;
        cfg.cpuid_features = self.cpu.features_edx;
        // x86_cpu_realizefn() sets HTT whenever a package has more than one thread.
        if self.topology.threads_per_socket() > 1 {
            cfg.cpuid_features |= CPUID_HT;
        }
        cfg.ram_size = self.ram_size;
        cfg.mem_array = mem_array_from_e820(self.e820.entries());
        cfg.pci_devices = self
            .host
            .bus()
            .devices()
            .iter()
            .map(|d| SmbiosPciDevice {
                id: d.id().unwrap_or_default().to_string(),
                bus: 0,
                devfn: d.devfn(),
                on_root_bus: true,
            })
            .collect();
        let t = smbios_get_tables(&cfg, &self.smbios).map_err(|e| e.message().to_string())?;
        let fwc = self.fw_cfg.state();
        fwc.add_file(SMBIOS_TABLES_FILE, t.tables).map_err(err)?;
        fwc.add_file(SMBIOS_ANCHOR_FILE, t.anchor).map_err(err)?;
        Ok(())
    }

    /// `pc_cmos_init_late()`.
    fn cmos_init_late(&self) -> Result<(), String> {
        let s = &self.rtc;
        cmos_set_memory(s, self.below_4g_mem_size, self.above_4g_mem_size);
        // The FPU is there and a PS/2 mouse is installed.
        s.set_cmos_data(REG_EQUIPMENT_BYTE, 0x02 | 0x04);
        set_boot_dev(s, &self.boot_devices, self.props.fd_bootchk)?;

        // idebus[0] and idebus[1] are AHCI ports 0 and 1, each with one unit.
        let drives = *self.drives.lock().unwrap_or_else(PoisonError::into_inner);
        let geometry = |port: usize| -> Option<HdGeometry> {
            drives[port].filter(|d| d.kind == DriveKind::Hd).map(|d| match d.geometry {
                Some((c, h, s)) => hd_geometry(c, h, s),
                None => hd_geometry_guess(d.sectors),
            })
        };
        let hd = [geometry(0), None, geometry(1), None];
        cmos_init_disks(s, &hd);
        Ok(())
    }

    /// The machine-done notifiers and the first system reset: `fw_cfg_machine_ready()`,
    /// `ich9_lpc_machine_ready()`, `pc_machine_done()` (the RTC CPU count, ACPI tables,
    /// `etc/e820`, the late CMOS setup) and then `qemu_system_reset()`. Plug devices before
    /// calling this.
    pub fn machine_done(&mut self) -> Result<(), String> {
        if self.done {
            return Err("machine_done called twice".to_string());
        }
        // The boot order is checked when the CMOS is set up, but fail before touching
        // anything.
        boot_order_nibbles(&self.boot_devices)?;
        self.done = true;
        let fwc = Arc::clone(self.fw_cfg.state());
        fwc.machine_reset(self.bootorder(), Vec::new()).map_err(err)?;
        self.lpc.machine_ready();

        rtc_set_cpus_count(&self.rtc, self.cpus);
        if self.props.acpi_enabled() {
            self.acpi_setup()?;
        }
        self.build_smbios()?;
        fwc.add_file(E820_FILE, self.e820.to_blob()).map_err(err)?;
        fwc.modify_i16(FW_CFG_NB_CPUS, self.cpus as u16);
        self.cmos_init_late()?;

        self.system_reset()
    }

    /// `pc_machine_reset()` without the CPUs: resets the devices, lets the ACPI tables be
    /// rebuilt on their next read and reloads the firmware and PVH segments into RAM.
    pub fn system_reset(&mut self) -> Result<(), String> {
        // qemu_devices_reset()
        if let Some(p) = &self.pic {
            p.master.reset();
            p.slave.reset();
        }
        self.ioapic.reset();
        // The root bus: the MCH (PAM, SMRAM, PCIEXBAR), the LPC bridge and the AHCI.
        self.host.reset();
        if let Some(h) = &self.hpet {
            h.reset(&mut self.hpet_fw);
        }
        if let Some(p) = &self.pit {
            p.reset();
        }
        self.rtc.reset();
        for s in self.serials.iter().flatten() {
            s.reset();
        }
        if let Some(k) = &self.i8042 {
            k.reset();
        }
        if let Some(p) = &self.port92 {
            p.reset();
        }
        for hook in self.reset_hooks.lock().unwrap_or_else(PoisonError::into_inner).iter() {
            hook();
        }
        // The CPU reset turns the A20 mask off again.
        self.a20.set(true);
        let fwc = self.fw_cfg.state();
        fwc.reset();
        fwc.machine_reset(self.bootorder(), Vec::new()).map_err(err)?;
        // acpi_build_reset()
        self.acpi_cache.lock().unwrap_or_else(PoisonError::into_inner).patched = false;

        // pflash_cfi01_reset(): back to read array mode. The contents stay.
        for f in &self.flashes {
            f.reset();
        }
        // rom_reset()
        if let Some((bios, data)) = &self.bios {
            let block = self.mem.ram_block(*bios).ok_or("pc.bios has no RAM block")?;
            block.write(0, data).map_err(err)?;
        }
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

    /// The PCI memory space, `rom_memory` in `pc_memory_init()`.
    pub fn pci_memory(&self) -> RegionId {
        self.pci
    }

    /// `address_space_memory`.
    pub fn memory_as(&self) -> &Arc<AddressSpace> {
        &self.memory_as
    }

    /// `address_space_io`.
    pub fn io_as(&self) -> &Arc<AddressSpace> {
        &self.io_as
    }

    /// The address space CPUs use in SMM: system memory with the MCH's SMRAM view on top.
    /// `None` with `smm=off`.
    pub fn smm_as(&self) -> Option<&Arc<AddressSpace>> {
        self.smm_as.as_ref()
    }

    /// The root region of [`Self::smm_as`].
    pub fn smm_memory(&self) -> Option<RegionId> {
        self.smm_root
    }

    /// The machine RAM region, `pc.ram`.
    pub fn ram_region(&self) -> RegionId {
        self.ram
    }

    /// The RAM block behind `pc.ram`.
    pub fn ram_block(&self) -> Option<Arc<RamBlock>> {
        self.mem.ram_block(self.ram)
    }

    /// The RAM blocks that migrate, in QEMU's names: `pc.ram`, `pc.bios` when the firmware is
    /// ROM, `pc.rom`, and the flash blocks.
    pub fn migratable_ram_blocks(&self) -> Vec<Arc<RamBlock>> {
        let mut ids = vec![self.ram];
        ids.extend(self.bios.as_ref().map(|b| b.0));
        ids.push(self.option_rom);
        ids.extend(self.device_roms.iter().copied());
        ids.extend(self.flashes.iter().map(|f| f.region()));
        ids.into_iter().filter_map(|id| self.mem.ram_block(id)).collect()
    }

    /// `pci_add_option_rom()`: makes the option ROM `name`, `DEVICE/TYPE.rom`, from `data`,
    /// in a RAM block of the next power of two size so it migrates. The device maps it with
    /// its ROM BAR.
    pub fn add_device_rom(&mut self, name: &str, data: &[u8]) -> Result<RegionId, String> {
        let rom = self.mem.new_rom(name, (data.len() as u64).next_power_of_two()).map_err(err)?;
        self.mem.ram_block(rom).ok_or("option ROM has no RAM")?.write(0, data).map_err(err)?;
        self.device_roms.push(rom);
        Ok(rom)
    }

    /// The firmware region, `pc.bios`, or `None` when the firmware is in pflash.
    pub fn bios_region(&self) -> Option<RegionId> {
        self.bios.as_ref().map(|b| b.0)
    }

    /// The system flashes, pflash0 first. Empty unless pflash0 has a drive.
    pub fn flashes(&self) -> &[Arc<Pflash>] {
        &self.flashes
    }

    /// Every RAM backed range of system memory after overlaps and the PAM settings are
    /// resolved. These are the KVM memory slots; they change when the guest reprograms PAM or
    /// SMRAM.
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

    /// Whether SMM is on: the CPUs need an SMM address space and SMIs.
    pub fn smm_enabled(&self) -> bool {
        self.smm_enabled
    }

    /// What `vmport` resolved to. There is no VMware port device either way.
    pub fn vmport(&self) -> bool {
        self.vmport
    }

    /// The machine type name, `pc-q35-11.1` or an older version.
    pub fn machine_name(&self) -> &'static str {
        self.machine_name
    }

    /// The properties the machine was built with.
    pub fn props(&self) -> &Q35Props {
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

    /// Warnings QEMU would print on stderr.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// What the kernel loader produced, if `-kernel` was given.
    pub fn kernel_boot(&self) -> Option<&X86KernelBoot> {
        self.kernel.as_ref()
    }

    /// The Q35 host bridge.
    pub fn host(&self) -> &Arc<Q35PciHost> {
        &self.host
    }

    /// The root bus, "pcie.0".
    pub fn pci_bus(&self) -> &Arc<PciBus> {
        self.host.bus()
    }

    /// The LPC bridge at 00:1f.0.
    pub fn lpc(&self) -> &Arc<Ich9Lpc> {
        &self.lpc
    }

    /// The SMBus controller at 00:1f.3, with `smbus=on`.
    pub fn smbus(&self) -> Option<&Arc<Ich9Smbus>> {
        self.smbus.as_ref().map(|s| &s.0)
    }

    /// The AHCI controller at 00:1f.2, with `sata=on`.
    pub fn ahci(&self) -> Option<&Arc<Ich9Ahci>> {
        self.ahci.as_ref()
    }

    /// `x86ms->gsi`: 24 lines.
    pub fn gsi(&self) -> &[IrqLine] {
        &self.gsi
    }

    /// The IOAPICs, for EOI broadcasts from the local APICs.
    pub fn ioapics(&self) -> &IoApics {
        &self.ioapics
    }

    /// The IOAPIC at 0xfec00000.
    pub fn ioapic(&self) -> &Arc<IoApic> {
        &self.ioapic
    }

    /// Where MSIs go: those of the IOAPIC, the HPET and PCI devices. Without a handler they
    /// are written to system memory, which only reaches something once a local APIC is mapped
    /// there.
    pub fn set_msi_handler(&self, handler: Option<IoApicMsiHandler>) {
        *self.msi_hook.write().unwrap_or_else(PoisonError::into_inner) = handler;
    }

    /// Puts a hook in front of the GSI handler. KVM with the in-kernel irqchip uses it to send
    /// the lines to `KVM_IRQ_LINE` instead of the emulated PIC and IOAPIC.
    pub fn set_gsi_hook(&self, hook: Option<GsiHook>) {
        *self.gsi_hook.write().unwrap_or_else(PoisonError::into_inner) = hook;
    }

    /// Where guest requested resets, shutdowns, suspends and wakeups go: from the LPC's PM
    /// block and reset control register, the i8042 and port 0x92.
    pub fn set_request_handler(&self, handler: Option<SystemRequestHandler>) {
        *self.request_hook.write().unwrap_or_else(PoisonError::into_inner) = handler;
    }

    /// Where SMIs from the LPC bridge (APM writes, the SMI timers) go.
    pub fn set_smi_handler(&self, handler: Option<SmiHandler>) {
        self.lpc.set_smi_handler(handler);
    }

    /// Receives the A20 line from the i8042 and port 0x92.
    pub fn set_a20_handler(&self, handler: Option<A20Handler>) {
        *self.a20.handler.write().unwrap_or_else(PoisonError::into_inner) = handler;
    }

    /// The last level driven onto the A20 line.
    pub fn a20_enabled(&self) -> bool {
        self.a20.level.load(Ordering::SeqCst)
    }

    /// The 8259 pair, with `pic` not off.
    pub fn pic(&self) -> Option<&I8259Pair> {
        self.pic.as_ref()
    }

    /// The 8259 INTR output. Connect it to the BSP's LINT0 or the accelerator's interrupt
    /// request.
    pub fn pic_output(&self) -> &Arc<IrqPin> {
        &self.pic_output
    }

    /// The PIT, with `pit` not off.
    pub fn pit(&self) -> Option<&Arc<I8254>> {
        self.pit.as_ref()
    }

    /// The speaker port, with the PIT.
    pub fn pcspk(&self) -> Option<&Arc<PcSpeaker>> {
        self.pcspk.as_ref()
    }

    /// Has `hook` run on every system reset, after the board's own devices.
    pub fn add_reset_hook(&self, hook: ResetHook) {
        self.reset_hooks.lock().unwrap_or_else(PoisonError::into_inner).push(hook);
    }

    /// The HPET, with `hpet=on`.
    pub fn hpet(&self) -> Option<&Arc<Hpet>> {
        self.hpet.as_ref()
    }

    /// The RTC.
    pub fn rtc(&self) -> &Arc<Mc146818Rtc> {
        &self.rtc
    }

    /// The ISA serial port `index`, 0 being COM1, when present.
    pub fn serial(&self, index: usize) -> Option<&Arc<Serial>> {
        self.serials.get(index).and_then(Option::as_ref)
    }

    /// The i8042, with `i8042=on`.
    pub fn i8042(&self) -> Option<&Arc<I8042>> {
        self.i8042.as_ref()
    }

    /// Port 0x92, which comes with the i8042.
    pub fn port92(&self) -> Option<&Arc<Port92>> {
        self.port92.as_ref()
    }

    /// `AcpiBuildState.patched` for the `acpi_build` section: a getter and a setter.
    pub(crate) fn acpi_patched(
        &self,
    ) -> (impl FnMut() -> bool + Send + 'static, impl FnMut(bool) + Send + 'static) {
        let (get, set) = (Arc::clone(&self.acpi_cache), Arc::clone(&self.acpi_cache));
        (
            move || get.lock().unwrap_or_else(PoisonError::into_inner).patched,
            move |p| set.lock().unwrap_or_else(PoisonError::into_inner).patched = p,
        )
    }
}
