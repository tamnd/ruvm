// SPDX-License-Identifier: GPL-2.0-or-later

//! `bochs-display`, QEMU's `hw/display/bochs-display.c`: a VGA free PCI framebuffer driven by
//! the Bochs VBE registers.
//!
//! The function is 1234:1111 with class 0x0380 and revision 2. BAR 0 is VRAM, prefetchable. BAR
//! 2 is the same 4 KiB register window as the standard VGA's, without the VGA ports: the EDID
//! blob at 0, the VBE registers at 0x500 and the byte order register at 0x600. On a PCI Express
//! bus the function gets an Express capability at 0x80.
//!
//! Where this differs from QEMU: QEMU shows VRAM directly and asks the dirty log which lines to
//! redraw. Here the surface owns its pixels and each update copies the visible part of VRAM
//! into it, reporting the lines that changed. There is no migration state.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use ruvm_base::error::{Error, Result};
use ruvm_hw_pci::pcie::{PCI_EXP_TYPE_ENDPOINT, PCI_EXP_TYPE_RC_END, pcie_cap_init};
use ruvm_hw_pci::regs::{PCI_BASE_ADDRESS_MEM_PREFETCH, PCI_BASE_ADDRESS_SPACE_MEMORY};
use ruvm_hw_pci::{PciBus, PciDevice, PciDeviceInfo};
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MemResult, MmioOps, RamBlock, RegionId};
use ruvm_ui::console::{ConsoleDevice, DisplayState, GraphicHwOps, QemuConsole};
use ruvm_ui::pixman::{PixelFormat, R5G6B5};
use ruvm_ui::surface::DisplaySurface;

use crate::edid::EdidInfo;
use crate::vga::{
    PCI_VGA_BOCHS_OFFSET, PCI_VGA_BOCHS_SIZE, PCI_VGA_QEXT_OFFSET, PCI_VGA_QEXT_SIZE,
    VBE_DISPI_ENABLED, VBE_DISPI_ID5, VBE_DISPI_INDEX_BPP, VBE_DISPI_INDEX_ENABLE,
    VBE_DISPI_INDEX_ID, VBE_DISPI_INDEX_NB, VBE_DISPI_INDEX_VIDEO_MEMORY_64K,
    VBE_DISPI_INDEX_VIRT_WIDTH, VBE_DISPI_INDEX_X_OFFSET, VBE_DISPI_INDEX_XRES,
    VBE_DISPI_INDEX_Y_OFFSET, VBE_DISPI_INDEX_YRES,
};
use crate::vga_pci::{
    PCI_CLASS_DISPLAY_OTHER, PCI_DEVICE_ID_QEMU_VGA, PCI_VENDOR_ID_QEMU, Qext, QextTarget,
    add_edid, mem_error, mmio_bar,
};

/// The option ROM of the device, `romfile`.
pub const BOCHS_DISPLAY_ROMFILE: &str = "vgabios-bochs-display.bin";

/// `BochsDisplayMode`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Mode {
    format: Option<PixelFormat>,
    bytepp: u32,
    width: u32,
    height: u32,
    stride: u32,
    offset: u64,
    size: u64,
}

#[derive(Debug, Default)]
struct Regs {
    vbe_regs: [u16; VBE_DISPI_INDEX_NB],
    big_endian_fb: bool,
    mode: Mode,
}

/// The registers and VRAM of a `bochs-display`.
pub struct BochsDisplayState {
    regs: Mutex<Regs>,
    vram: Arc<RamBlock>,
    vgamem: u64,
}

impl fmt::Debug for BochsDisplayState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BochsDisplayState").field("vgamem", &self.vgamem).finish_non_exhaustive()
    }
}

impl BochsDisplayState {
    fn lock(&self) -> MutexGuard<'_, Regs> {
        self.regs.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// `bochs_display_vbe_read()` of register `index`.
    pub fn vbe_read(&self, index: usize) -> u64 {
        match index {
            VBE_DISPI_INDEX_ID => return u64::from(VBE_DISPI_ID5),
            VBE_DISPI_INDEX_VIDEO_MEMORY_64K => return self.vgamem / (64 * 1024),
            _ => {}
        }
        let r = self.lock();
        match r.vbe_regs.get(index) {
            Some(&v) => u64::from(v),
            None => u64::MAX,
        }
    }

    /// `bochs_display_vbe_write()` of register `index`.
    pub fn vbe_write(&self, index: usize, val: u64) {
        let mut r = self.lock();
        if let Some(reg) = r.vbe_regs.get_mut(index) {
            *reg = val as u16;
        }
    }

    /// `bochs_display_get_mode()`: the mode the registers describe, if it is valid.
    fn get_mode(&self, r: &Regs) -> Option<Mode> {
        let vbe = &r.vbe_regs;
        if vbe[VBE_DISPI_INDEX_ENABLE] & VBE_DISPI_ENABLED == 0 {
            return None;
        }
        let (format, bytepp) = match vbe[VBE_DISPI_INDEX_BPP] {
            // Best effort: native byte order only.
            16 => (R5G6B5, 2),
            32 => {
                let f = if r.big_endian_fb {
                    PixelFormat::be_x8r8g8b8()
                } else {
                    PixelFormat::le_x8r8g8b8()
                };
                (f, 4)
            }
            _ => return None,
        };
        let width = u32::from(vbe[VBE_DISPI_INDEX_XRES]);
        let height = u32::from(vbe[VBE_DISPI_INDEX_YRES]);
        let virt_width = u32::from(vbe[VBE_DISPI_INDEX_VIRT_WIDTH]).max(width);
        let stride = virt_width * bytepp;
        let size = u64::from(stride) * u64::from(height);
        let offset = u64::from(vbe[VBE_DISPI_INDEX_X_OFFSET]) * u64::from(bytepp)
            + u64::from(vbe[VBE_DISPI_INDEX_Y_OFFSET]) * u64::from(stride);
        if width < 64 || height < 64 {
            return None;
        }
        if offset + size > self.vgamem {
            return None;
        }
        Some(Mode { format: Some(format), bytepp, width, height, stride, offset, size })
    }

    /// `bochs_display_update()`.
    fn update(&self, con: &QemuConsole) {
        let mut r = self.lock();
        let Some(mode) = self.get_mode(&r) else { return };
        let mut full_update = false;
        if r.mode != mode {
            r.mode = mode;
            let format = mode.format.unwrap_or_else(|| unreachable!());
            con.set_surface(Some(DisplaySurface::new_from(
                mode.width as usize,
                mode.height as usize,
                format,
                mode.stride as usize,
            )));
            full_update = true;
        }
        drop(r);

        let mut rects = Vec::new();
        con.with_surface_mut(|surf| {
            let Some(surf) = surf else { return };
            let stride = surf.stride();
            let data = surf.data_mut();
            let len = (mode.size as usize).min(data.len());
            let mut fresh = vec![0u8; len];
            let _ = self.vram.read(mode.offset, &mut fresh);
            let mut ys: i32 = -1;
            for y in 0..mode.height as usize {
                let a = (y * stride).min(len);
                let b = ((y + 1) * stride).min(len);
                let dirty = data[a..b] != fresh[a..b];
                if dirty {
                    data[a..b].copy_from_slice(&fresh[a..b]);
                }
                if dirty && ys < 0 {
                    ys = y as i32;
                }
                if !dirty && ys >= 0 {
                    rects.push((ys, y as i32 - ys));
                    ys = -1;
                }
            }
            if ys >= 0 {
                rects.push((ys, mode.height as i32 - ys));
            }
        });
        if full_update {
            con.update_full();
        } else {
            for (y, h) in rects {
                con.update(0, y, mode.width as i32, h);
            }
        }
    }
}

impl QextTarget for BochsDisplayState {
    fn big_endian_fb(&self) -> bool {
        self.lock().big_endian_fb
    }

    fn set_big_endian_fb(&self, value: bool) {
        self.lock().big_endian_fb = value;
    }
}

impl GraphicHwOps for BochsDisplayState {
    fn gfx_update(&self, con: &QemuConsole) -> bool {
        self.update(con);
        true
    }
}

/// "bochs dispi interface", `bochs_display_vbe_ops`.
#[derive(Debug)]
struct Vbe(Arc<BochsDisplayState>);

impl MmioOps for Vbe {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(self.0.vbe_read((offset >> 1) as usize))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.0.vbe_write((offset >> 1) as usize, value);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::exact(2)
    }
}

/// The properties of `bochs-display`.
#[derive(Clone, Debug)]
pub struct BochsDisplayProps {
    /// The qdev `id`.
    pub id: Option<String>,
    /// `vgamem` in bytes.
    pub vgamem: u64,
    /// `edid`.
    pub edid: bool,
    /// `xres`, `yres`, `xmax`, `ymax` and `refresh_rate`.
    pub edid_info: EdidInfo,
    /// The default framebuffer byte order, the target's.
    pub big_endian: bool,
}

impl Default for BochsDisplayProps {
    fn default() -> BochsDisplayProps {
        BochsDisplayProps {
            id: None,
            vgamem: 16 << 20,
            edid: true,
            edid_info: EdidInfo::default(),
            big_endian: false,
        }
    }
}

/// A realized `bochs-display`.
#[derive(Debug)]
pub struct BochsDisplay {
    state: Arc<BochsDisplayState>,
    dev: Arc<PciDevice>,
    vram: RegionId,
    mmio: RegionId,
}

impl BochsDisplay {
    /// `bochs_display_realize()`: plugs the function into `bus` at `devfn`, or the first free
    /// slot, and makes its console in `ds`. `express` says whether the bus is PCI Express,
    /// `pci_bus_is_express()`.
    pub fn realize(
        bus: &PciBus,
        devfn: Option<u8>,
        express: bool,
        props: &BochsDisplayProps,
        ds: &DisplayState,
    ) -> Result<BochsDisplay> {
        if props.vgamem < 4 << 20 {
            return Err(Error::generic("bochs-display: video memory too small".to_string()));
        }
        if props.vgamem > 256 << 20 {
            return Err(Error::generic("bochs-display: video memory too big".to_string()));
        }
        let vgamem = props.vgamem.next_power_of_two();
        let mem = bus.memory();
        let info = PciDeviceInfo {
            name: "bochs-display".to_string(),
            id: props.id.clone(),
            vendor_id: PCI_VENDOR_ID_QEMU,
            device_id: PCI_DEVICE_ID_QEMU_VGA,
            revision: 2,
            class_id: PCI_CLASS_DISPLAY_OTHER,
            express,
            ..PciDeviceInfo::default()
        };
        let dev = bus.register_device(&info, devfn)?;

        let vram = mem.new_ram("bochs-display-vram", vgamem).map_err(mem_error)?;
        let block = mem
            .ram_block(vram)
            .ok_or_else(|| Error::generic("bochs-display-vram has no RAM block".to_string()))?;
        let state = Arc::new(BochsDisplayState {
            regs: Mutex::new(Regs { big_endian_fb: props.big_endian, ..Regs::default() }),
            vram: block,
            vgamem,
        });

        let con = ds.graphic_console_create(
            Some(ConsoleDevice { id: props.id.clone(), typename: "bochs-display".to_string() }),
            0,
            Arc::clone(&state) as Arc<dyn GraphicHwOps>,
        );
        drop(con);

        let vbe = mem
            .new_io(
                "bochs dispi interface",
                u128::from(PCI_VGA_BOCHS_SIZE),
                Arc::new(Vbe(Arc::clone(&state))),
            )
            .map_err(mem_error)?;
        let qext = mem
            .new_io(
                "qemu extended regs",
                u128::from(PCI_VGA_QEXT_SIZE),
                Arc::new(Qext(Arc::clone(&state))),
            )
            .map_err(mem_error)?;
        let mmio = mmio_bar(mem, "bochs-display-mmio")?;
        mem.add_subregion(mmio, PCI_VGA_BOCHS_OFFSET, vbe).map_err(mem_error)?;
        mem.add_subregion(mmio, PCI_VGA_QEXT_OFFSET, qext).map_err(mem_error)?;

        dev.register_bar(0, PCI_BASE_ADDRESS_MEM_PREFETCH, vram);
        dev.register_bar(2, PCI_BASE_ADDRESS_SPACE_MEMORY, mmio);

        if props.edid {
            add_edid(mem, mmio, 256, &props.edid_info)?;
        }

        if express {
            // pcie_endpoint_cap_init()
            let ty = if bus.is_root() { PCI_EXP_TYPE_RC_END } else { PCI_EXP_TYPE_ENDPOINT };
            pcie_cap_init(&dev, 0x80, ty, 0)?;
        }

        Ok(BochsDisplay { state, dev, vram, mmio })
    }

    /// The PCI function.
    pub fn pci_device(&self) -> &Arc<PciDevice> {
        &self.dev
    }

    /// The register and VRAM state.
    pub fn state(&self) -> &Arc<BochsDisplayState> {
        &self.state
    }

    /// The VRAM region behind BAR 0.
    pub fn vram_region(&self) -> RegionId {
        self.vram
    }

    /// The register region behind BAR 2.
    pub fn mmio_region(&self) -> RegionId {
        self.mmio
    }
}
