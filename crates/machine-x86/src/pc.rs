// SPDX-License-Identifier: GPL-2.0-or-later

//! Pieces shared by the PC family of boards: the parts of hw/i386/pc.c, x86-common.c,
//! port92.c and pcspk.c that do not depend on the chipset.
//!
//! The boards themselves live in [`crate::q35`] and [`crate::microvm`]. What is here is either
//! a small ISA device with no better home (port 0x92, the speaker port, the dummy ports 0x80 and
//! 0xf0) or one of the helpers pc.c uses to fill in the CMOS and the fw_cfg tables.

use std::fmt;
use std::sync::{Arc, Mutex, PoisonError, RwLock, Weak};

use ruvm_hw_core::IrqLine;
use ruvm_hw_core::IrqPin;
use ruvm_hw_core::fw_cfg::{DmaMemory, FW_CFG_ARCH_LOCAL};
use ruvm_hw_intc::i8259::ISA_NUM_IRQS;
use ruvm_hw_intc::ioapic::{IO_APIC_SECONDARY_IRQBASE, IOAPIC_NUM_PINS};
use ruvm_hw_timer::i8254::I8254;
use ruvm_hw_timer::mc146818::Mc146818Rtc;
use ruvm_mem::{
    AccessConstraints, AccessCtx, AccessSize, AddressSpace, MemResult, MemTxAttrs, MmioOps,
};

pub(crate) const KIB: u64 = 1 << 10;
pub(crate) const MIB: u64 = 1 << 20;
pub(crate) const GIB: u64 = 1 << 30;

/// `FW_CFG_ACPI_TABLES`: the `-acpitable` blobs, empty unless the user gave some.
pub const FW_CFG_ACPI_TABLES: u16 = FW_CFG_ARCH_LOCAL;
/// `FW_CFG_SMBIOS_ENTRIES`.
pub const FW_CFG_SMBIOS_ENTRIES: u16 = FW_CFG_ARCH_LOCAL + 1;
/// `FW_CFG_IRQ0_OVERRIDE`.
pub const FW_CFG_IRQ0_OVERRIDE: u16 = FW_CFG_ARCH_LOCAL + 2;
/// `FW_CFG_HPET`: the packed `hpet_fw_cfg`.
pub const FW_CFG_HPET: u16 = FW_CFG_ARCH_LOCAL + 4;

/// `PC_FW_DATA`: space at the top of low RAM that the kernel loader keeps free for the ACPI
/// tables the firmware copies there.
pub const PC_FW_DATA: u64 = 0x20000 + 0x8000;
/// `PC_ROM_MIN_VGA`, where the option ROM area starts.
pub const PC_ROM_MIN_VGA: u64 = 0xc0000;
/// `PC_ROM_SIZE`, the option ROM area up to the ISA BIOS.
pub const PC_ROM_SIZE: u64 = 0xe0000 - PC_ROM_MIN_VGA;
/// The largest part of the firmware mapped below 1 MiB, `x86_isa_bios_init()`.
pub const ISA_BIOS_MAX: u64 = 128 * KIB;
/// `ACPI_BUILD_PCI_IRQS`: the ISA IRQs PCI interrupts may be routed to, the default of the
/// `pci_irq_mask` x86 machine field.
pub const ACPI_BUILD_PCI_IRQS: u16 = 1 << 5 | 1 << 9 | 1 << 10 | 1 << 11;
/// `PC_MAX_BOOT_DEVICES`.
pub const PC_MAX_BOOT_DEVICES: usize = 3;
/// `REG_EQUIPMENT_BYTE` in the CMOS.
pub const REG_EQUIPMENT_BYTE: usize = 0x14;
/// The port of `port92`, the fast A20 and reset register.
pub const PORT92_IO_BASE: u64 = 0x92;
/// The port of the PC speaker.
pub const PCSPK_IO_BASE: u64 = 0x61;

/// fw_cfg DMA through a weak reference, so the address space that maps fw_cfg does not keep
/// itself alive.
pub(crate) struct WeakDma(pub(crate) Weak<AddressSpace>);

impl DmaMemory for WeakDma {
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.0.upgrade().is_some_and(|a| a.read(addr, MemTxAttrs::UNSPECIFIED, buf).is_ok())
    }

    fn write(&self, addr: u64, buf: &[u8]) -> bool {
        self.0.upgrade().is_some_and(|a| a.write(addr, MemTxAttrs::UNSPECIFIED, buf).is_ok())
    }
}

/// `unassigned_io_ops`, behind `get_system_io()`: ports nobody claimed read as all ones and
/// ignore writes. The dummy ports 0x80 and 0xf0 behave the same way.
#[derive(Debug)]
pub(crate) struct UnassignedIo;

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

pub(crate) fn err<E: fmt::Display>(e: E) -> String {
    e.to_string()
}

/// A hook in front of the GSI handler, called with the GSI number and level. It returns true
/// when it took care of the GSI, which is how an accelerator with its own interrupt controllers
/// (KVM's `KVM_IRQ_LINE`) takes the lines away from the emulated PIC and IOAPIC.
pub type GsiHook = Arc<dyn Fn(u32, i32) -> bool + Send + Sync>;

/// The slot a board keeps for its [`GsiHook`].
pub(crate) type GsiHookSlot = Arc<RwLock<Option<GsiHook>>>;

/// `GSIState` and `gsi_handler()`: GSIs 0 to 15 go to the 8259 and the first IOAPIC, 16 to 23
/// to the first IOAPIC and 24 to 47 to the second.
pub(crate) struct GsiState {
    pub(crate) i8259: Vec<IrqLine>,
    pub(crate) ioapic: Vec<IrqLine>,
    pub(crate) ioapic2: Vec<IrqLine>,
    pub(crate) hook: GsiHookSlot,
}

impl GsiState {
    pub(crate) fn set(&self, n: u32, level: i32) {
        let hook = self.hook.read().unwrap_or_else(PoisonError::into_inner).clone();
        if let Some(h) = hook {
            if h(n, level) {
                return;
            }
        }
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

/// The memory sizes of the CMOS, the part of `pc_cmos_init()` both PC boards and microvm write.
pub fn cmos_set_memory(s: &Mc146818Rtc, below: u64, above: u64) {
    // base memory (first MiB)
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

/// `x86_rtc_set_cpus_count()`: CMOS byte 0x5f holds the CPU count minus one, or 0 when it does
/// not fit and the firmware has to use `FW_CFG_NB_CPUS`.
pub fn rtc_set_cpus_count(s: &Mc146818Rtc, cpus: u32) {
    let val = if cpus > 0xff { 0 } else { cpus.saturating_sub(1) as u8 };
    s.set_cmos_data(0x5f, val);
}

/// `boot_device2nibble()`.
fn boot_device2nibble(boot_device: char) -> u8 {
    match boot_device {
        'a' | 'b' => 0x01, // floppy boot
        'c' => 0x02,       // hard drive boot
        'd' => 0x03,       // CD-ROM boot
        'n' => 0x04,       // Network boot
        _ => 0,
    }
}

/// The checks of `set_boot_dev()`: the nibbles for a `-boot order=` string.
pub fn boot_order_nibbles(order: &str) -> Result<[u8; PC_MAX_BOOT_DEVICES], String> {
    // QEMU uses strlen(), so count bytes.
    if order.len() > PC_MAX_BOOT_DEVICES {
        return Err("Too many boot devices for PC".to_string());
    }
    let mut bds = [0u8; PC_MAX_BOOT_DEVICES];
    for (i, c) in order.chars().enumerate() {
        bds[i] = boot_device2nibble(c);
        if bds[i] == 0 {
            return Err(format!("Invalid boot device for PC: '{c}'"));
        }
    }
    Ok(bds)
}

/// `set_boot_dev()`: the boot order in CMOS bytes 0x3d and 0x38, the latter also carrying the
/// "skip the floppy boot signature check" bit.
pub fn set_boot_dev(s: &Mc146818Rtc, order: &str, fd_bootchk: bool) -> Result<(), String> {
    let bds = boot_order_nibbles(order)?;
    s.set_cmos_data(0x3d, (bds[1] << 4) | bds[0]);
    s.set_cmos_data(0x38, (bds[2] << 4) | u8::from(!fd_bootchk));
    Ok(())
}

/// `BiosAtaTranslation`, as `ide_get_bios_chs_trans()` returns it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BiosAtaTranslation {
    None = 1,
    Lba = 2,
    Large = 3,
}

/// The physical geometry of a hard disk and the BIOS translation for it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct HdGeometry {
    pub cylinders: u32,
    pub heads: u32,
    pub sectors: u32,
    pub translation: BiosAtaTranslation,
}

/// `hd_geometry_guess()` for a disk of `nb_sectors` 512 byte sectors, without the guess from
/// the partition table (which the IDE model does not make either): `guess_chs_for_size()`
/// followed by `hd_bios_chs_auto_trans()`.
pub fn hd_geometry_guess(nb_sectors: u64) -> HdGeometry {
    let cylinders = (nb_sectors / (16 * 63)).clamp(2, 16383) as u32;
    let (heads, sectors) = (16, 63);
    let translation = if cylinders <= 1024 && heads <= 16 && sectors <= 63 {
        BiosAtaTranslation::None
    } else {
        BiosAtaTranslation::Lba
    };
    HdGeometry { cylinders, heads, sectors, translation }
}

/// `cmos_init_hd()`: a type 47 (user defined) drive entry.
pub fn cmos_init_hd(s: &Mc146818Rtc, type_ofs: usize, info_ofs: usize, g: &HdGeometry) {
    // QEMU passes the geometry as int16_t and int8_t.
    let cylinders = g.cylinders as u16;
    let heads = g.heads as u8;
    s.set_cmos_data(type_ofs, 47);
    s.set_cmos_data(info_ofs, cylinders as u8);
    s.set_cmos_data(info_ofs + 1, (cylinders >> 8) as u8);
    s.set_cmos_data(info_ofs + 2, heads);
    s.set_cmos_data(info_ofs + 3, 0xff);
    s.set_cmos_data(info_ofs + 4, 0xff);
    s.set_cmos_data(info_ofs + 5, 0xc0 | (u8::from(heads > 8) << 3));
    s.set_cmos_data(info_ofs + 6, cylinders as u8);
    s.set_cmos_data(info_ofs + 7, (cylinders >> 8) as u8);
    s.set_cmos_data(info_ofs + 8, g.sectors as u8);
}

/// The hard disk and floppy part of `pc_cmos_init_late()`. `hd[i]` is the disk on IDE bus
/// `i / 2`, unit `i % 2`, if it is a hard disk: the two drive entries only cover bus 0, the
/// translation bits cover all four. There is never a floppy controller here, so the floppy
/// byte is 0 and the equipment byte keeps its floppy bits clear.
pub fn cmos_init_disks(s: &Mc146818Rtc, hd: &[Option<HdGeometry>; 4]) {
    let mut val = 0u8;
    if let Some(g) = &hd[0] {
        cmos_init_hd(s, 0x19, 0x1b, g);
        val |= 0xf0;
    }
    if let Some(g) = &hd[1] {
        cmos_init_hd(s, 0x1a, 0x24, g);
        val |= 0x0f;
    }
    s.set_cmos_data(0x12, val);

    let mut val = 0u8;
    for (i, g) in hd.iter().enumerate() {
        if let Some(g) = g {
            let trans = g.translation as u8 - 1;
            val |= trans << (i * 2);
        }
    }
    s.set_cmos_data(0x39, val);

    // pc_cmos_init_floppy() without a controller: no drive types and no drive bits in the
    // equipment byte.
    s.set_cmos_data(0x10, 0);
}

/// `pc_pci_hole64_start()` without memory hotplug, CXL and SGX: the first GiB boundary past
/// the RAM above 4 GiB.
pub fn pci_hole64_start(above_4g_mem_start: u64, above_4g_mem_size: u64) -> u64 {
    (above_4g_mem_start + above_4g_mem_size).next_multiple_of(GIB)
}

/// A callback for guest requested resets, from port 0x92 and the keyboard controller.
pub type ResetRequest = Arc<dyn Fn() + Send + Sync>;

/// `Port92State`, the "System Control Port A" at 0x92: bit 1 drives A20 and a rising edge on
/// bit 0 resets the machine.
pub struct Port92 {
    outport: Mutex<u8>,
    a20_out: IrqPin,
    reset: ResetRequest,
}

impl fmt::Debug for Port92 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Port92").field("outport", &self.outport()).finish_non_exhaustive()
    }
}

impl Port92 {
    /// `port92_initfn()`. `reset` is `qemu_system_reset_request()`.
    pub fn new(reset: ResetRequest) -> Arc<Port92> {
        Arc::new(Port92 { outport: Mutex::new(0), a20_out: IrqPin::new(), reset })
    }

    /// The `a20` output, `PORT92_A20_LINE`.
    pub fn a20_out(&self) -> &IrqPin {
        &self.a20_out
    }

    /// The register.
    pub fn outport(&self) -> u8 {
        *self.outport.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `port92_reset()`: only the reset bit goes back to 0.
    pub fn reset(&self) {
        *self.outport.lock().unwrap_or_else(PoisonError::into_inner) &= !1;
    }

    /// `port92_write()`.
    pub fn write(&self, val: u8) {
        let oldval = {
            let mut o = self.outport.lock().unwrap_or_else(PoisonError::into_inner);
            let old = *o;
            *o = val;
            old
        };
        self.a20_out.set(i32::from((val >> 1) & 1));
        if val & 1 != 0 && oldval & 1 == 0 {
            (self.reset)();
        }
    }
}

impl MmioOps for Port92 {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.outport()))
    }

    fn write(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        Port92::write(self, value as u8);
        Ok(())
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 1)
    }
}

/// The I/O side of `PCSpkState` at port 0x61: the channel 2 gate, the speaker data bit, the
/// channel 2 output and the refresh toggle BIOSes use for delays. There is no audio.
pub struct PcSpeaker {
    pit: Arc<I8254>,
    state: Mutex<(u8, u8)>,
}

impl fmt::Debug for PcSpeaker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PcSpeaker").finish_non_exhaustive()
    }
}

impl PcSpeaker {
    /// `pcspk_realizefn()` with the `pit` link set.
    pub fn new(pit: Arc<I8254>) -> Arc<PcSpeaker> {
        Arc::new(PcSpeaker { pit, state: Mutex::new((0, 0)) })
    }
}

impl MmioOps for PcSpeaker {
    /// `pcspk_io_read()`.
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        let ch = self.pit.get_channel_info(2);
        let mut s = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let (data_on, refresh) = &mut *s;
        *refresh ^= 1 << 4;
        let v = (ch.gate as u32 & 1)
            | (u32::from(*data_on) << 1)
            | u32::from(*refresh)
            | ((ch.out as u32 & 1) << 5);
        Ok(u64::from(v))
    }

    /// `pcspk_io_write()`.
    fn write(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        let gate = (value & 1) as i32;
        self.state.lock().unwrap_or_else(PoisonError::into_inner).0 = ((value >> 1) & 1) as u8;
        self.pit.set_gate(2, gate);
        Ok(())
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 1)
    }
}

/// `ioportF0_io_ops`: reads as all ones, and a write clears the FPU error interrupt, IRQ 13.
pub struct IoportF0 {
    ferr: IrqLine,
}

impl fmt::Debug for IoportF0 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IoportF0").finish_non_exhaustive()
    }
}

impl IoportF0 {
    /// `ferr` is GSI 13.
    pub fn new(ferr: IrqLine) -> IoportF0 {
        IoportF0 { ferr }
    }
}

impl MmioOps for IoportF0 {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::MAX)
    }

    /// `ioportF0_write()`: `cpu_set_ignne()` and lowering FERR.
    fn write(
        &self,
        _cx: &AccessCtx,
        _offset: u64,
        _size: AccessSize,
        _value: u64,
    ) -> MemResult<()> {
        self.ferr.set(0);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4).allow_unaligned()
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4).allow_unaligned()
    }
}
