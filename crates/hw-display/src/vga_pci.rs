// SPDX-License-Identifier: GPL-2.0-or-later

//! The PCI standard VGA, the "VGA" device of QEMU's `hw/display/vga-pci.c`.
//!
//! The function is 1234:1111 with class 0x0300. BAR 0 is VRAM, prefetchable. BAR 2 is a 4 KiB
//! register window: the EDID blob at 0, the VGA ports 0x3c0 to 0x3df at 0x400, the Bochs VBE
//! registers at 0x500 and the QEMU byte order register at 0x600. The device also claims the
//! legacy window at 0xa0000 and the VGA and VBE ports in the address spaces of its bus.
//!
//! Where this differs from QEMU: there is no `secondary-vga`, no migration state, and the ACPI
//! `_S1D` to `_S3D` methods QEMU adds to the DSDT for the device are not built.

use std::fmt;
use std::sync::{Arc, Mutex};

use ruvm_base::error::{Error, Result};
use ruvm_hw_pci::regs::{PCI_BASE_ADDRESS_MEM_PREFETCH, PCI_BASE_ADDRESS_SPACE_MEMORY};
use ruvm_hw_pci::{PciBus, PciDevice, PciDeviceInfo, PciDeviceOps};
use ruvm_mem::{
    AccessConstraints, AccessCtx, AccessSize, MemResult, MemorySystem, MmioOps, RegionId,
};
use ruvm_ui::console::{ConsoleDevice, DisplayState, GraphicHwOps};

use crate::edid::{EdidInfo, EdidRegion, edid_generate};
use crate::vga::{
    PCI_VGA_BOCHS_OFFSET, PCI_VGA_BOCHS_SIZE, PCI_VGA_IOPORT_OFFSET, PCI_VGA_IOPORT_SIZE,
    PCI_VGA_MMIO_SIZE, PCI_VGA_QEXT_BIG_ENDIAN, PCI_VGA_QEXT_LITTLE_ENDIAN, PCI_VGA_QEXT_OFFSET,
    PCI_VGA_QEXT_REG_BYTEORDER, PCI_VGA_QEXT_REG_SIZE, PCI_VGA_QEXT_SIZE, VgaCommon, VgaLowmem,
    vga_vram_size_mb,
};

/// `PCI_VENDOR_ID_QEMU`.
pub const PCI_VENDOR_ID_QEMU: u16 = 0x1234;
/// `PCI_DEVICE_ID_QEMU_VGA`.
pub const PCI_DEVICE_ID_QEMU_VGA: u16 = 0x1111;
/// `PCI_CLASS_DISPLAY_VGA`.
pub const PCI_CLASS_DISPLAY_VGA: u16 = 0x0300;
/// `PCI_CLASS_DISPLAY_OTHER`.
pub const PCI_CLASS_DISPLAY_OTHER: u16 = 0x0380;

/// The option ROM of the device, `romfile`.
pub const VGA_ROMFILE: &str = "vgabios-stdvga.bin";

pub(crate) fn mem_error(e: ruvm_mem::MemError) -> Error {
    Error::generic(e.to_string())
}

/// `unassigned_io_ops`, behind the register BAR: reads all ones, ignores writes.
#[derive(Debug)]
pub(crate) struct UnassignedIo;

impl MmioOps for UnassignedIo {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::MAX)
    }

    fn write(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize, _v: u64) -> MemResult<()> {
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4).allow_unaligned()
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4).allow_unaligned()
    }
}

/// The 4 KiB register BAR, an unassigned I/O region with the register blocks on top. Here it is
/// a container with an all ones region at the bottom.
pub(crate) fn mmio_bar(mem: &MemorySystem, name: &str) -> Result<RegionId> {
    let bar = mem.new_container(name, u128::from(PCI_VGA_MMIO_SIZE)).map_err(mem_error)?;
    let bg = mem
        .new_io(name, u128::from(PCI_VGA_MMIO_SIZE), Arc::new(UnassignedIo))
        .map_err(mem_error)?;
    mem.add_subregion_overlap(bar, 0, bg, -1).map_err(mem_error)?;
    Ok(bar)
}

/// Adds the EDID window at offset 0 of `bar`, `qemu_edid_region_io()`.
pub(crate) fn add_edid(
    mem: &MemorySystem,
    bar: RegionId,
    size: usize,
    info: &EdidInfo,
) -> Result<RegionId> {
    let mut blob = vec![0u8; size];
    let mut info = info.clone();
    edid_generate(&mut blob, &mut info);
    let region = mem
        .new_io("edid", size as u128, Arc::new(EdidRegion::new(blob.into())))
        .map_err(mem_error)?;
    mem.add_subregion(bar, 0, region).map_err(mem_error)?;
    Ok(region)
}

/// The byte order register, `pci_vga_qext_ops` and `bochs_display_qext_ops`.
pub(crate) trait QextTarget: Send + Sync + fmt::Debug {
    fn big_endian_fb(&self) -> bool;
    fn set_big_endian_fb(&self, value: bool);
}

impl QextTarget for VgaCommon {
    fn big_endian_fb(&self) -> bool {
        VgaCommon::big_endian_fb(self)
    }

    fn set_big_endian_fb(&self, value: bool) {
        VgaCommon::set_big_endian_fb(self, value);
    }
}

/// "qemu extended regs".
#[derive(Debug)]
pub(crate) struct Qext<T: QextTarget>(pub(crate) Arc<T>);

impl<T: QextTarget> MmioOps for Qext<T> {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(match offset {
            PCI_VGA_QEXT_REG_SIZE => PCI_VGA_QEXT_SIZE,
            PCI_VGA_QEXT_REG_BYTEORDER => u64::from(if self.0.big_endian_fb() {
                PCI_VGA_QEXT_BIG_ENDIAN
            } else {
                PCI_VGA_QEXT_LITTLE_ENDIAN
            }),
            _ => 0,
        })
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        if offset == PCI_VGA_QEXT_REG_BYTEORDER {
            if value == u64::from(PCI_VGA_QEXT_BIG_ENDIAN) {
                self.0.set_big_endian_fb(true);
            }
            if value == u64::from(PCI_VGA_QEXT_LITTLE_ENDIAN) {
                self.0.set_big_endian_fb(false);
            }
        }
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::exact(4)
    }
}

/// "vga ioports remapped", `pci_vga_ioport_ops`: the ports 0x3c0 to 0x3df as MMIO.
#[derive(Debug)]
struct PciVgaIoports(Arc<VgaCommon>);

impl MmioOps for PciVgaIoports {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        let port = offset as u32 + 0x3c0;
        Ok(match size {
            AccessSize::B1 => u64::from(self.0.ioport_read(port)),
            AccessSize::B2 => {
                u64::from(self.0.ioport_read(port)) | (u64::from(self.0.ioport_read(port + 1)) << 8)
            }
            _ => 0,
        })
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        let port = offset as u32 + 0x3c0;
        match size {
            AccessSize::B1 => self.0.ioport_write(port, value as u32),
            AccessSize::B2 => {
                // Low byte first, so a word write sets an index and then its register.
                self.0.ioport_write(port, (value & 0xff) as u32);
                self.0.ioport_write(port + 1, ((value >> 8) & 0xff) as u32);
            }
            _ => {}
        }
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 2)
    }
}

/// `vga_reset()`, which `vga_init()` registers for the system reset. Here the PCI bus reset at
/// system reset runs it.
#[derive(Debug)]
struct VgaPciOps(Arc<VgaCommon>);

impl PciDeviceOps for VgaPciOps {
    fn reset(&self, _dev: &PciDevice) {
        self.0.reset();
    }
}

/// "bochs dispi interface", `pci_vga_bochs_ops`: VBE register n at offset 2n.
#[derive(Debug)]
struct PciVgaBochs(Arc<VgaCommon>);

impl MmioOps for PciVgaBochs {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        self.0.vbe_write_index((offset >> 1) as u32);
        Ok(u64::from(self.0.vbe_read_data()))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.0.vbe_write_index((offset >> 1) as u32);
        self.0.vbe_write_data(value as u32);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::exact(2)
    }
}

/// The properties of "VGA".
#[derive(Clone, Debug)]
pub struct VgaPciProps {
    /// The qdev `id`.
    pub id: Option<String>,
    /// `vgamem_mb`.
    pub vgamem_mb: u32,
    /// `mmio`: the register BAR.
    pub mmio: bool,
    /// `qemu-extended-regs`.
    pub qemu_extended_regs: bool,
    /// `edid`.
    pub edid: bool,
    /// `xres`, `yres`, `xmax`, `ymax` and `refresh_rate`.
    pub edid_info: EdidInfo,
    /// The machine is an x86 one, which puts VBE data at both 0x1cf and 0x1d0.
    pub x86: bool,
    /// The default framebuffer byte order, the target's.
    pub big_endian: bool,
}

impl Default for VgaPciProps {
    fn default() -> VgaPciProps {
        VgaPciProps {
            id: None,
            vgamem_mb: 16,
            mmio: true,
            qemu_extended_regs: true,
            edid: true,
            edid_info: EdidInfo::default(),
            x86: true,
            big_endian: false,
        }
    }
}

/// A realized "VGA".
pub struct VgaPci {
    vga: Arc<VgaCommon>,
    dev: Arc<PciDevice>,
    vram: RegionId,
    mmio: Option<RegionId>,
    lowmem: RegionId,
    ports: Mutex<Vec<(u16, RegionId)>>,
}

impl fmt::Debug for VgaPci {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VgaPci").field("vga", &self.vga).finish_non_exhaustive()
    }
}

impl VgaPci {
    /// `pci_std_vga_realize()`: plugs the function into `bus` at `devfn`, or the first free
    /// slot, and makes its console in `ds`.
    pub fn realize(
        bus: &PciBus,
        devfn: Option<u8>,
        props: &VgaPciProps,
        ds: &DisplayState,
    ) -> Result<VgaPci> {
        let mem = bus.memory();
        let revision = if props.mmio && props.qemu_extended_regs { 2 } else { 0 };
        let info = PciDeviceInfo {
            name: "VGA".to_string(),
            id: props.id.clone(),
            vendor_id: PCI_VENDOR_ID_QEMU,
            device_id: PCI_DEVICE_ID_QEMU_VGA,
            revision,
            class_id: PCI_CLASS_DISPLAY_VGA,
            ..PciDeviceInfo::default()
        };

        // vga_common_init()
        let size = u64::from(vga_vram_size_mb(props.vgamem_mb)) << 20;
        let vram = mem.new_ram("vga.vram", size).map_err(mem_error)?;
        let block = mem
            .ram_block(vram)
            .ok_or_else(|| Error::generic("vga.vram has no RAM block".to_string()))?;
        let vga = VgaCommon::new(block, props.big_endian);

        let dev = bus.register_device(&info, devfn)?;
        dev.set_ops(Arc::new(VgaPciOps(Arc::clone(&vga))));

        // vga_init(): the legacy window over everything else at 0xa0000, and the ports.
        let lowmem = mem
            .new_io("vga-lowmem", 0x20000, Arc::new(VgaLowmem::new(Arc::clone(&vga))))
            .map_err(mem_error)?;
        mem.add_subregion_overlap(bus.mem_space(), 0xa0000, lowmem, 1).map_err(mem_error)?;
        let mut ports = Vec::new();
        for (port, len, ops) in vga.portio_regions(props.x86) {
            let name = if port == 0x1ce { "vbe" } else { "vga" };
            let r = mem.new_io(name, u128::from(len), ops).map_err(mem_error)?;
            mem.add_subregion(bus.io_space(), u64::from(port), r).map_err(mem_error)?;
            ports.push((port, r));
        }

        let con = ds.graphic_console_create(
            Some(ConsoleDevice { id: props.id.clone(), typename: "VGA".to_string() }),
            0,
            Arc::clone(&vga) as Arc<dyn GraphicHwOps>,
        );
        vga.set_console(con);

        dev.register_bar(0, PCI_BASE_ADDRESS_MEM_PREFETCH, vram);

        let mmio = if props.mmio {
            let bar = mmio_bar(mem, "vga.mmio")?;
            let io = mem
                .new_io(
                    "vga ioports remapped",
                    u128::from(PCI_VGA_IOPORT_SIZE),
                    Arc::new(PciVgaIoports(Arc::clone(&vga))),
                )
                .map_err(mem_error)?;
            mem.add_subregion(bar, PCI_VGA_IOPORT_OFFSET, io).map_err(mem_error)?;
            let bochs = mem
                .new_io(
                    "bochs dispi interface",
                    u128::from(PCI_VGA_BOCHS_SIZE),
                    Arc::new(PciVgaBochs(Arc::clone(&vga))),
                )
                .map_err(mem_error)?;
            mem.add_subregion(bar, PCI_VGA_BOCHS_OFFSET, bochs).map_err(mem_error)?;
            if props.qemu_extended_regs {
                let qext = mem
                    .new_io(
                        "qemu extended regs",
                        u128::from(PCI_VGA_QEXT_SIZE),
                        Arc::new(Qext(Arc::clone(&vga))),
                    )
                    .map_err(mem_error)?;
                mem.add_subregion(bar, PCI_VGA_QEXT_OFFSET, qext).map_err(mem_error)?;
            }
            if props.edid {
                add_edid(mem, bar, 384, &props.edid_info)?;
            }
            dev.register_bar(2, PCI_BASE_ADDRESS_SPACE_MEMORY, bar);
            Some(bar)
        } else {
            None
        };

        Ok(VgaPci { vga, dev, vram, mmio, lowmem, ports: Mutex::new(ports) })
    }

    /// The PCI function.
    pub fn pci_device(&self) -> &Arc<PciDevice> {
        &self.dev
    }

    /// The VGA core.
    pub fn vga(&self) -> &Arc<VgaCommon> {
        &self.vga
    }

    /// The VRAM region behind BAR 0.
    pub fn vram_region(&self) -> RegionId {
        self.vram
    }

    /// The register region behind BAR 2, if the `mmio` property is on.
    pub fn mmio_region(&self) -> Option<RegionId> {
        self.mmio
    }

    /// The legacy window at 0xa0000.
    pub fn lowmem_region(&self) -> RegionId {
        self.lowmem
    }

    /// The I/O port regions, by first port.
    pub fn port_regions(&self) -> Vec<(u16, RegionId)> {
        self.ports.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }

    /// `vga_reset()`, `vga_common_reset()` on the core.
    pub fn reset(&self) {
        self.vga.reset();
    }
}
