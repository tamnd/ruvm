// SPDX-License-Identifier: GPL-2.0-or-later

//! The VGA core, `VGACommonState`, from QEMU's `hw/display/vga.c`, `vga-helpers.h` and
//! `vga-access.h`.
//!
//! This is the standard VGA plus the Bochs VBE ("dispi") extensions: the I/O ports at 0x3b0 to
//! 0x3df and 0x1ce, the legacy window at 0xa0000, the planar memory model behind it, and the
//! code that turns VRAM into a picture on the console in text, planar, 256 colour and VBE direct
//! colour modes. The PCI front end that puts it on a bus is in `vga_pci.rs`.
//!
//! Where this differs from QEMU:
//! - Retrace is always the "dumb" method, which QEMU uses unless `-global VGA.retrace=precise`.
//! - QEMU maps a "vga.chain4" alias of VRAM over the legacy window in chain 4 mode as a fast
//!   path. Here every access goes through [`VgaCommon::mem_writeb`], which stores to the same
//!   bytes.
//! - QEMU finds changed scanlines with the `DIRTY_MEMORY_VGA` log. Here every update renders all
//!   scanlines and compares them with the surface, which reports the same rectangles for writes
//!   that change pixels.
//! - QEMU builds a surface over VRAM for the direct colour modes. Here that surface owns its
//!   pixels and the update copies the visible part of VRAM into it.
//! - The text and blink cursor phases follow the host clock from device creation, where QEMU
//!   uses `QEMU_CLOCK_VIRTUAL`.
//! - The curses text console hook (`text_update`), migration state, and the hardware cursor
//!   hooks that only cirrus uses are not here.

use std::fmt;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MemResult, MmioOps, RamBlock};
use ruvm_ui::console::{GraphicHwOps, QemuConsole};
use ruvm_ui::pixman::default_pixman_format;
use ruvm_ui::surface::DisplaySurface;

/// `VBE_DISPI_MAX_XRES`.
pub const VBE_DISPI_MAX_XRES: u16 = 16000;
/// `VBE_DISPI_MAX_YRES`.
pub const VBE_DISPI_MAX_YRES: u16 = 12000;
/// `VBE_DISPI_MAX_BPP`.
pub const VBE_DISPI_MAX_BPP: u16 = 32;

pub const VBE_DISPI_INDEX_ID: usize = 0x0;
pub const VBE_DISPI_INDEX_XRES: usize = 0x1;
pub const VBE_DISPI_INDEX_YRES: usize = 0x2;
pub const VBE_DISPI_INDEX_BPP: usize = 0x3;
pub const VBE_DISPI_INDEX_ENABLE: usize = 0x4;
pub const VBE_DISPI_INDEX_BANK: usize = 0x5;
pub const VBE_DISPI_INDEX_VIRT_WIDTH: usize = 0x6;
pub const VBE_DISPI_INDEX_VIRT_HEIGHT: usize = 0x7;
pub const VBE_DISPI_INDEX_X_OFFSET: usize = 0x8;
pub const VBE_DISPI_INDEX_Y_OFFSET: usize = 0x9;
/// The size of the register file, `VBE_DISPI_INDEX_NB`.
pub const VBE_DISPI_INDEX_NB: usize = 0xa;
/// The read only VRAM size in 64 KiB units, not in the register file.
pub const VBE_DISPI_INDEX_VIDEO_MEMORY_64K: usize = 0xa;

pub const VBE_DISPI_ID0: u16 = 0xb0c0;
pub const VBE_DISPI_ID5: u16 = 0xb0c5;

pub const VBE_DISPI_DISABLED: u16 = 0x00;
pub const VBE_DISPI_ENABLED: u16 = 0x01;
pub const VBE_DISPI_GETCAPS: u16 = 0x02;
pub const VBE_DISPI_8BIT_DAC: u16 = 0x20;
pub const VBE_DISPI_LFB_ENABLED: u16 = 0x40;
pub const VBE_DISPI_NOCLEARMEM: u16 = 0x80;

/// The MMIO BAR of the PCI devices, `PCI_VGA_MMIO_SIZE`.
pub const PCI_VGA_MMIO_SIZE: u64 = 0x1000;
/// The VGA registers in the MMIO BAR, `PCI_VGA_IOPORT_OFFSET`.
pub const PCI_VGA_IOPORT_OFFSET: u64 = 0x400;
pub const PCI_VGA_IOPORT_SIZE: u64 = 0x3e0 - 0x3c0;
/// The Bochs VBE registers in the MMIO BAR, `PCI_VGA_BOCHS_OFFSET`.
pub const PCI_VGA_BOCHS_OFFSET: u64 = 0x500;
pub const PCI_VGA_BOCHS_SIZE: u64 = 0x0b * 2;
/// The QEMU extension registers in the MMIO BAR, `PCI_VGA_QEXT_OFFSET`.
pub const PCI_VGA_QEXT_OFFSET: u64 = 0x600;
pub const PCI_VGA_QEXT_SIZE: u64 = 2 * 4;
pub const PCI_VGA_QEXT_REG_SIZE: u64 = 0;
pub const PCI_VGA_QEXT_REG_BYTEORDER: u64 = 4;
pub const PCI_VGA_QEXT_LITTLE_ENDIAN: u32 = 0x1e1e_1e1e;
pub const PCI_VGA_QEXT_BIG_ENDIAN: u32 = 0xbebe_bebe;

// Ports, from vga_regs.h.
const VGA_CRT_DC: u32 = 0x3d5;
const VGA_CRT_DM: u32 = 0x3b5;
const VGA_ATT_R: u32 = 0x3c1;
const VGA_ATT_W: u32 = 0x3c0;
const VGA_GFX_D: u32 = 0x3cf;
const VGA_SEQ_D: u32 = 0x3c5;
const VGA_MIS_R: u32 = 0x3cc;
const VGA_MIS_W: u32 = 0x3c2;
const VGA_FTC_R: u32 = 0x3ca;
const VGA_IS1_RC: u32 = 0x3da;
const VGA_IS1_RM: u32 = 0x3ba;
const VGA_PEL_D: u32 = 0x3c9;
const VGA_CRT_IC: u32 = 0x3d4;
const VGA_CRT_IM: u32 = 0x3b4;
const VGA_GFX_I: u32 = 0x3ce;
const VGA_SEQ_I: u32 = 0x3c4;
const VGA_PEL_IW: u32 = 0x3c8;
const VGA_PEL_IR: u32 = 0x3c7;

const VGA_ATT_C: usize = 0x15;
const VGA_MIS_COLOR: u8 = 0x01;

const VGA_CRTC_H_DISP: usize = 1;
const VGA_CRTC_V_TOTAL: usize = 6;
const VGA_CRTC_OVERFLOW: usize = 7;
const VGA_CRTC_MAX_SCAN: usize = 9;
const VGA_CRTC_CURSOR_START: usize = 0x0a;
const VGA_CRTC_CURSOR_END: usize = 0x0b;
const VGA_CRTC_START_HI: usize = 0x0c;
const VGA_CRTC_START_LO: usize = 0x0d;
const VGA_CRTC_CURSOR_HI: usize = 0x0e;
const VGA_CRTC_CURSOR_LO: usize = 0x0f;
const VGA_CRTC_V_SYNC_END: usize = 0x11;
const VGA_CRTC_V_DISP_END: usize = 0x12;
const VGA_CRTC_OFFSET: usize = 0x13;
const VGA_CRTC_UNDERLINE: usize = 0x14;
const VGA_CRTC_MODE: usize = 0x17;
const VGA_CRTC_LINE_COMPARE: usize = 0x18;
const VGA_CR11_LOCK_CR0_CR7: u8 = 0x80;
const VGA_CR14_DW: u8 = 0x40;
const VGA_CR17_WORD_BYTE: u8 = 0x40;

const VGA_ATC_MODE: usize = 0x10;
const VGA_ATC_OVERSCAN: usize = 0x11;
const VGA_ATC_PLANE_ENABLE: usize = 0x12;
const VGA_ATC_PEL: usize = 0x13;
const VGA_ATC_COLOR_PAGE: usize = 0x14;

const VGA_SEQ_CLOCK_MODE: usize = 0x01;
const VGA_SEQ_PLANE_WRITE: usize = 0x02;
const VGA_SEQ_CHARACTER_MAP: usize = 0x03;
const VGA_SEQ_MEMORY_MODE: usize = 0x04;
const VGA_SR01_CHAR_CLK_8DOTS: u8 = 0x01;
const VGA_SR02_ALL_PLANES: u8 = 0x0f;
const VGA_SR04_SEQ_MODE: u8 = 0x04;
const VGA_SR04_CHN_4M: u8 = 0x08;

const VGA_GFX_SR_VALUE: usize = 0x00;
const VGA_GFX_SR_ENABLE: usize = 0x01;
const VGA_GFX_COMPARE_VALUE: usize = 0x02;
const VGA_GFX_DATA_ROTATE: usize = 0x03;
const VGA_GFX_PLANE_READ: usize = 0x04;
const VGA_GFX_MODE: usize = 0x05;
const VGA_GFX_MISC: usize = 0x06;
const VGA_GFX_COMPARE_MASK: usize = 0x07;
const VGA_GFX_BIT_MASK: usize = 0x08;
const VGA_GR05_HOST_ODD_EVEN: u8 = 0x10;
const VGA_GR06_GRAPHICS_MODE: u8 = 0x01;
const VGA_GR06_CHAIN_ODD_EVEN: u8 = 0x02;

const ST01_V_RETRACE: u8 = 0x08;
const ST01_DISP_ENABLE: u8 = 0x01;

/// `CH_ATTR_SIZE`: the most characters a text mode screen can have.
const CH_ATTR_SIZE: usize = 160 * 100;

/// Frame counter bit 4: the cursor blinks every 16 frames at 60 Hz.
const VGA_TEXT_CURSOR_PERIOD_MS: i64 = 1000 * 2 * 16 / 60;
/// Frame counter bit 5: characters blink every 32 frames at 60 Hz.
const VGA_TEXT_BLINK_PERIOD_MS: i64 = 1000 * 2 * 32 / 60;

/// The address mask of the non VESA modes, `VGA_VRAM_SIZE`.
const VGA_VRAM_SIZE: u32 = 256 * 1024;

/// A shift of zero pixels in every mode, `VGA_HPEL_NEUTRAL`.
const VGA_HPEL_NEUTRAL: u8 = 8;

const GMODE_TEXT: i32 = 0;
const GMODE_GRAPH: i32 = 1;
const GMODE_BLANK: i32 = 2;

/// `sr_mask`: the bits of each sequencer register that exist.
pub const SR_MASK: [u8; 8] = [0x03, 0x3d, 0x0f, 0x3f, 0x0e, 0x00, 0x00, 0xff];

/// `gr_mask`: the bits of each graphics controller register that exist.
pub const GR_MASK: [u8; 16] = [
    0x0f, 0x0f, 0x0f, 0x1f, 0x03, 0x7b, 0x0f, 0x0f, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// `mask16`: a plane mask turned into a byte mask over the four planes of a little endian dword.
const MASK16: [u32; 16] = [
    0x0000_0000,
    0x0000_00ff,
    0x0000_ff00,
    0x0000_ffff,
    0x00ff_0000,
    0x00ff_00ff,
    0x00ff_ff00,
    0x00ff_ffff,
    0xff00_0000,
    0xff00_00ff,
    0xff00_ff00,
    0xff00_ffff,
    0xffff_0000,
    0xffff_00ff,
    0xffff_ff00,
    0xffff_ffff,
];

/// `expand4`: bit j of the index moved to bit 4j.
const EXPAND4: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut v = 0u32;
        let mut j = 0;
        while j < 8 {
            v |= ((i as u32 >> j) & 1) << (j * 4);
            j += 1;
        }
        t[i] = v;
        i += 1;
    }
    t
};

/// `expand2`: bit pair j of the index moved to bits 4j and 4j + 1.
const EXPAND2: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut v = 0u32;
        let mut j = 0;
        while j < 4 {
            v |= ((i as u32 >> (2 * j)) & 3) << (j * 4);
            j += 1;
        }
        t[i] = v;
        i += 1;
    }
    t
};

/// `expand4to8`: every bit of a nibble doubled.
const EXPAND4TO8: [u8; 16] = {
    let mut t = [0u8; 16];
    let mut i = 0;
    while i < 16 {
        let mut v = 0u8;
        let mut j = 0;
        while j < 4 {
            let b = ((i >> j) & 1) as u8;
            v |= b << (2 * j);
            v |= b << (2 * j + 1);
            j += 1;
        }
        t[i] = v;
        i += 1;
    }
    t
};

/// The glyph of the text cursor: every row lit.
const CURSOR_GLYPH: [u8; 32 * 4] = [0xff; 32 * 4];

/// `c6_to_8()`: a 6 bit DAC value widened to 8 bits.
fn c6_to_8(v: u8) -> u32 {
    let v = u32::from(v & 0x3f);
    let b = v & 1;
    (v << 2) | (b << 1) | b
}

/// `rgb_to_pixel32()`.
fn rgb_to_pixel32(r: u32, g: u32, b: u32) -> u32 {
    (r << 16) | (g << 8) | b
}

/// `vga_common_init()`'s size rule for the `vgamem_mb` property: clamped to 1..512 and rounded up
/// to a power of two.
pub fn vga_vram_size_mb(mb: u32) -> u32 {
    mb.clamp(1, 512).next_power_of_two()
}

/// `VGADisplayParams`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct DisplayParams {
    line_offset: u32,
    start_addr: u32,
    line_compare: u32,
    hpel: u8,
    hpel_split: bool,
}

/// The register file and the display refresh state of `VGACommonState`.
struct VgaState {
    latch: u32,
    sr_index: u8,
    sr: [u8; 256],
    sr_vbe: [u8; 256],
    gr_index: u8,
    gr: [u8; 256],
    ar_index: u8,
    ar: [u8; 21],
    ar_flip_flop: bool,
    cr_index: u8,
    cr: [u8; 256],
    msr: u8,
    fcr: u8,
    st00: u8,
    st01: u8,
    dac_state: u8,
    dac_sub_index: u8,
    dac_read_index: u8,
    dac_write_index: u8,
    dac_cache: [u8; 3],
    dac_8bit: bool,
    palette: [u8; 768],
    bank_offset: u32,
    vbe_index: u16,
    vbe_regs: [u16; VBE_DISPI_INDEX_NB],
    vbe_start_addr: u32,
    vbe_line_offset: u32,
    vbe_bank_mask: u32,
    font_offsets: [u32; 2],
    graphic_mode: i32,
    shift_control: u8,
    double_scan: u8,
    params: DisplayParams,
    plane_updated: u32,
    last_line_offset: u32,
    last_cw: u8,
    last_ch: u8,
    last_width: u32,
    last_height: u32,
    last_scr_width: u32,
    last_scr_height: u32,
    last_depth: u32,
    last_byteswap: bool,
    cursor_start: u8,
    cursor_end: u8,
    cursor_visible_phase: bool,
    cursor_blink_time: i64,
    blink_visible_phase: bool,
    blink_time: i64,
    cursor_offset: u32,
    full_update_text: bool,
    full_update_gfx: bool,
    big_endian_fb: bool,
    last_palette: [u32; 256],
    last_ch_attr: Vec<u32>,
    /// The start address the shared surface was made for, None when the surface is the
    /// console's own. Stands in for comparing `surface_data()` with VRAM.
    shared_base: Option<u32>,
    /// Scratch rows for the graphic modes, `panning_buf` and a copy of the surface row.
    panning_buf: Vec<u32>,
    row_buf: Vec<u32>,
}

impl VgaState {
    fn new(big_endian_fb: bool) -> VgaState {
        VgaState {
            latch: 0,
            sr_index: 0,
            sr: [0; 256],
            sr_vbe: [0; 256],
            gr_index: 0,
            gr: [0; 256],
            ar_index: 0,
            ar: [0; 21],
            ar_flip_flop: false,
            cr_index: 0,
            cr: [0; 256],
            msr: 0,
            fcr: 0,
            st00: 0,
            st01: 0,
            dac_state: 0,
            dac_sub_index: 0,
            dac_read_index: 0,
            dac_write_index: 0,
            dac_cache: [0; 3],
            dac_8bit: false,
            palette: [0; 768],
            bank_offset: 0,
            vbe_index: 0,
            vbe_regs: [0; VBE_DISPI_INDEX_NB],
            vbe_start_addr: 0,
            vbe_line_offset: 0,
            vbe_bank_mask: 0,
            font_offsets: [0; 2],
            graphic_mode: -1,
            shift_control: 0,
            double_scan: 0,
            params: DisplayParams::default(),
            plane_updated: 0,
            last_line_offset: 0,
            last_cw: 0,
            last_ch: 0,
            last_width: 0,
            last_height: 0,
            last_scr_width: 0,
            last_scr_height: 0,
            last_depth: 0,
            last_byteswap: false,
            cursor_start: 0,
            cursor_end: 0,
            cursor_visible_phase: false,
            cursor_blink_time: 0,
            blink_visible_phase: false,
            blink_time: 0,
            cursor_offset: 0,
            full_update_text: false,
            full_update_gfx: false,
            big_endian_fb,
            last_palette: [0; 256],
            last_ch_attr: vec![0; CH_ATTR_SIZE],
            shared_base: None,
            panning_buf: Vec::new(),
            row_buf: Vec::new(),
        }
    }

    fn vbe_enabled(&self) -> bool {
        self.vbe_regs[VBE_DISPI_INDEX_ENABLE] & VBE_DISPI_ENABLED != 0
    }

    /// `sr()`: the sequencer register as the VBE modes override it.
    fn sr(&self, idx: usize) -> u8 {
        if self.vbe_enabled() { self.sr_vbe[idx] } else { self.sr[idx] }
    }

    fn ioport_invalid(&self, addr: u32) -> bool {
        if self.msr & VGA_MIS_COLOR != 0 {
            (0x3b0..=0x3bf).contains(&addr)
        } else {
            (0x3d0..=0x3df).contains(&addr)
        }
    }

    /// `vbe_fixup_regs()`: moves the VBE registers to the closest valid mode.
    fn vbe_fixup_regs(&mut self, vbe_size: u32) {
        if !self.vbe_enabled() {
            return;
        }
        let r = &mut self.vbe_regs;
        let bits: u32 = match r[VBE_DISPI_INDEX_BPP] {
            4 | 8 | 16 | 24 | 32 => u32::from(r[VBE_DISPI_INDEX_BPP]),
            15 => 16,
            _ => {
                r[VBE_DISPI_INDEX_BPP] = 8;
                8
            }
        };

        r[VBE_DISPI_INDEX_XRES] &= !7;
        if r[VBE_DISPI_INDEX_XRES] == 0 {
            r[VBE_DISPI_INDEX_XRES] = 8;
        }
        if r[VBE_DISPI_INDEX_XRES] > VBE_DISPI_MAX_XRES {
            r[VBE_DISPI_INDEX_XRES] = VBE_DISPI_MAX_XRES;
        }
        r[VBE_DISPI_INDEX_VIRT_WIDTH] &= !7;
        if r[VBE_DISPI_INDEX_VIRT_WIDTH] > VBE_DISPI_MAX_XRES {
            r[VBE_DISPI_INDEX_VIRT_WIDTH] = VBE_DISPI_MAX_XRES;
        }
        if r[VBE_DISPI_INDEX_VIRT_WIDTH] < r[VBE_DISPI_INDEX_XRES] {
            r[VBE_DISPI_INDEX_VIRT_WIDTH] = r[VBE_DISPI_INDEX_XRES];
        }

        let linelength = u32::from(r[VBE_DISPI_INDEX_VIRT_WIDTH]) * bits / 8;
        let maxy = vbe_size / linelength;
        if r[VBE_DISPI_INDEX_YRES] == 0 {
            r[VBE_DISPI_INDEX_YRES] = 1;
        }
        if r[VBE_DISPI_INDEX_YRES] > VBE_DISPI_MAX_YRES {
            r[VBE_DISPI_INDEX_YRES] = VBE_DISPI_MAX_YRES;
        }
        if u32::from(r[VBE_DISPI_INDEX_YRES]) > maxy {
            r[VBE_DISPI_INDEX_YRES] = maxy as u16;
        }

        if r[VBE_DISPI_INDEX_X_OFFSET] > VBE_DISPI_MAX_XRES {
            r[VBE_DISPI_INDEX_X_OFFSET] = VBE_DISPI_MAX_XRES;
        }
        if r[VBE_DISPI_INDEX_Y_OFFSET] > VBE_DISPI_MAX_YRES {
            r[VBE_DISPI_INDEX_Y_OFFSET] = VBE_DISPI_MAX_YRES;
        }
        let yres = u32::from(r[VBE_DISPI_INDEX_YRES]);
        let mut offset = u32::from(r[VBE_DISPI_INDEX_X_OFFSET]) * bits / 8;
        offset += u32::from(r[VBE_DISPI_INDEX_Y_OFFSET]) * linelength;
        if offset + yres * linelength > vbe_size {
            r[VBE_DISPI_INDEX_Y_OFFSET] = 0;
            offset = u32::from(r[VBE_DISPI_INDEX_X_OFFSET]) * bits / 8;
            if offset + yres * linelength > vbe_size {
                r[VBE_DISPI_INDEX_X_OFFSET] = 0;
                offset = 0;
            }
        }

        r[VBE_DISPI_INDEX_VIRT_HEIGHT] = maxy as u16;
        self.vbe_line_offset = linelength;
        self.vbe_start_addr = offset / 4;
    }

    /// `vbe_update_vgaregs()`: puts the VGA registers in the graphic mode the VBE mode needs.
    fn vbe_update_vgaregs(&mut self) {
        if !self.vbe_enabled() {
            return;
        }
        self.gr[VGA_GFX_MISC] = (self.gr[VGA_GFX_MISC] & !0x0c) | 0x04 | VGA_GR06_GRAPHICS_MODE;
        self.cr[VGA_CRTC_MODE] |= 3;
        self.cr[VGA_CRTC_OFFSET] = (self.vbe_line_offset >> 3) as u8;
        self.cr[VGA_CRTC_H_DISP] =
            ((self.vbe_regs[VBE_DISPI_INDEX_XRES] >> 3) as u8).wrapping_sub(1);
        let h = i32::from(self.vbe_regs[VBE_DISPI_INDEX_YRES]) - 1;
        self.cr[VGA_CRTC_V_DISP_END] = h as u8;
        self.cr[VGA_CRTC_OVERFLOW] = (self.cr[VGA_CRTC_OVERFLOW] & !0x42)
            | (((h >> 7) & 0x02) as u8)
            | (((h >> 3) & 0x40) as u8);
        self.cr[VGA_CRTC_LINE_COMPARE] = 0xff;
        self.cr[VGA_CRTC_OVERFLOW] |= 0x10;
        self.cr[VGA_CRTC_MAX_SCAN] |= 0x40;

        let shift_control: u8 = if self.vbe_regs[VBE_DISPI_INDEX_BPP] == 4 {
            self.sr_vbe[VGA_SEQ_CLOCK_MODE] &= !8;
            0
        } else {
            self.sr_vbe[VGA_SEQ_MEMORY_MODE] |= VGA_SR04_CHN_4M;
            self.sr_vbe[VGA_SEQ_PLANE_WRITE] |= VGA_SR02_ALL_PLANES;
            2
        };
        self.gr[VGA_GFX_MODE] = (self.gr[VGA_GFX_MODE] & !0x60) | (shift_control << 5);
        self.cr[VGA_CRTC_MAX_SCAN] &= !0x9f;
    }

    /// `vga_get_params()`.
    fn get_params(&self) -> DisplayParams {
        if self.vbe_enabled() {
            DisplayParams {
                line_offset: self.vbe_line_offset,
                start_addr: self.vbe_start_addr,
                line_compare: 65535,
                hpel: VGA_HPEL_NEUTRAL,
                hpel_split: false,
            }
        } else {
            let cr = &self.cr;
            DisplayParams {
                line_offset: u32::from(cr[VGA_CRTC_OFFSET]) << 3,
                start_addr: u32::from(cr[VGA_CRTC_START_LO])
                    | (u32::from(cr[VGA_CRTC_START_HI]) << 8),
                line_compare: u32::from(cr[VGA_CRTC_LINE_COMPARE])
                    | ((u32::from(cr[VGA_CRTC_OVERFLOW]) & 0x10) << 4)
                    | ((u32::from(cr[VGA_CRTC_MAX_SCAN]) & 0x40) << 3),
                hpel: self.ar[VGA_ATC_PEL],
                hpel_split: self.ar[VGA_ATC_MODE] & 0x20 != 0,
            }
        }
    }

    /// `update_basic_params()`: true when the start address, pitch or split changed.
    fn update_basic_params(&mut self) -> bool {
        let current = self.get_params();
        if current != self.params {
            self.params = current;
            return true;
        }
        false
    }

    /// `vga_get_bpp()`.
    fn get_bpp(&self) -> u32 {
        if self.vbe_enabled() { u32::from(self.vbe_regs[VBE_DISPI_INDEX_BPP]) } else { 0 }
    }

    /// `vga_get_resolution()`.
    fn get_resolution(&self) -> (u32, u32) {
        if self.vbe_enabled() {
            (
                u32::from(self.vbe_regs[VBE_DISPI_INDEX_XRES]),
                u32::from(self.vbe_regs[VBE_DISPI_INDEX_YRES]),
            )
        } else {
            let cr = &self.cr;
            let width = (u32::from(cr[VGA_CRTC_H_DISP]) + 1) * 8;
            let height = u32::from(cr[VGA_CRTC_V_DISP_END])
                | ((u32::from(cr[VGA_CRTC_OVERFLOW]) & 0x02) << 7)
                | ((u32::from(cr[VGA_CRTC_OVERFLOW]) & 0x40) << 3);
            (width, height + 1)
        }
    }

    /// `vga_get_text_resolution()`: columns, rows, character width and height.
    fn get_text_resolution(&self) -> (u32, u32, u32, u32) {
        let cheight = u32::from(self.cr[VGA_CRTC_MAX_SCAN] & 0x1f) + 1;
        let mut cwidth = 8;
        if self.sr(VGA_SEQ_CLOCK_MODE) & VGA_SR01_CHAR_CLK_8DOTS == 0 {
            cwidth = 9;
        }
        if self.sr(VGA_SEQ_CLOCK_MODE) & 0x08 != 0 {
            cwidth = 16;
        }
        let width = u32::from(self.cr[VGA_CRTC_H_DISP]) + 1;
        let height = if self.cr[VGA_CRTC_V_TOTAL] == 100 {
            100
        } else {
            let h = u32::from(self.cr[VGA_CRTC_V_DISP_END])
                | ((u32::from(self.cr[VGA_CRTC_OVERFLOW]) & 0x02) << 7)
                | ((u32::from(self.cr[VGA_CRTC_OVERFLOW]) & 0x40) << 3);
            (h + 1) / cheight
        };
        (width, height, cwidth, cheight)
    }

    /// `update_palette16()`: true when a colour changed.
    fn update_palette16(&mut self) -> bool {
        let mut full_update = false;
        for i in 0..16 {
            let mut v = usize::from(self.ar[i]);
            let page = usize::from(self.ar[VGA_ATC_COLOR_PAGE]);
            if self.ar[VGA_ATC_MODE] & 0x80 != 0 {
                v = ((page & 0xf) << 4) | (v & 0xf);
            } else {
                v = ((page & 0xc) << 4) | (v & 0x3f);
            }
            v *= 3;
            let col = rgb_to_pixel32(
                c6_to_8(self.palette[v]),
                c6_to_8(self.palette[v + 1]),
                c6_to_8(self.palette[v + 2]),
            );
            if col != self.last_palette[i] {
                full_update = true;
                self.last_palette[i] = col;
            }
        }
        full_update
    }

    /// `update_palette256()`: true when a colour changed.
    fn update_palette256(&mut self) -> bool {
        let mut full_update = false;
        for i in 0..256 {
            let v = i * 3;
            let p = &self.palette[v..v + 3];
            let col = if self.dac_8bit {
                rgb_to_pixel32(p[0].into(), p[1].into(), p[2].into())
            } else {
                rgb_to_pixel32(c6_to_8(p[0]), c6_to_8(p[1]), c6_to_8(p[2]))
            };
            if col != self.last_palette[i] {
                full_update = true;
                self.last_palette[i] = col;
            }
        }
        full_update
    }
}

/// `VGACommonState`: the registers, VRAM and the console of one VGA.
pub struct VgaCommon {
    st: Mutex<VgaState>,
    vram: Arc<RamBlock>,
    vram_size: u32,
    vbe_size: u32,
    vbe_size_mask: u32,
    default_endian_fb: bool,
    con: OnceLock<QemuConsole>,
    epoch: Instant,
}

impl fmt::Debug for VgaCommon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VgaCommon")
            .field("vram_size", &self.vram_size)
            .field("vbe_size", &self.vbe_size)
            .finish_non_exhaustive()
    }
}

impl VgaCommon {
    /// `vga_common_init()` over `vram`, whose size must be a power of two, followed by the reset
    /// QEMU does before the guest runs. `big_endian_fb` is the default framebuffer byte order,
    /// the target's in QEMU.
    pub fn new(vram: Arc<RamBlock>, big_endian_fb: bool) -> Arc<VgaCommon> {
        let vram_size = u32::try_from(vram.len()).unwrap_or(u32::MAX);
        let vga = VgaCommon {
            st: Mutex::new(VgaState::new(big_endian_fb)),
            vram,
            vram_size,
            vbe_size: vram_size,
            vbe_size_mask: vram_size.wrapping_sub(1),
            default_endian_fb: big_endian_fb,
            con: OnceLock::new(),
            epoch: Instant::now(),
        };
        vga.reset();
        Arc::new(vga)
    }

    fn lock(&self) -> MutexGuard<'_, VgaState> {
        self.st.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn bytes(&self) -> &[AtomicU8] {
        self.vram.atomic_bytes()
    }

    fn now_ms(&self) -> i64 {
        i64::try_from(self.epoch.elapsed().as_millis()).unwrap_or(i64::MAX)
    }

    /// The VRAM block.
    pub fn vram(&self) -> &Arc<RamBlock> {
        &self.vram
    }

    pub fn vram_size(&self) -> u32 {
        self.vram_size
    }

    /// The console the device draws on, once [`VgaCommon::set_console`] ran.
    pub fn console(&self) -> Option<&QemuConsole> {
        self.con.get()
    }

    /// Records the console made for this device with `qemu_graphic_console_create()`.
    pub fn set_console(&self, con: QemuConsole) {
        let _ = self.con.set(con);
    }

    /// The `big-endian-framebuffer` property.
    pub fn big_endian_fb(&self) -> bool {
        self.lock().big_endian_fb
    }

    pub fn set_big_endian_fb(&self, value: bool) {
        self.lock().big_endian_fb = value;
    }

    /// Whether the framebuffer byte order differs from the default, `vga_endian_state_needed()`.
    pub fn endian_state_needed(&self) -> bool {
        self.big_endian_fb() != self.default_endian_fb
    }

    /// `vga_common_reset()`.
    pub fn reset(&self) {
        let mut s = self.lock();
        let big_endian_fb = s.big_endian_fb;
        let mut fresh = VgaState::new(big_endian_fb);
        fresh.vbe_regs[VBE_DISPI_INDEX_ID] = VBE_DISPI_ID5;
        fresh.vbe_bank_mask = (self.vram_size >> 16).wrapping_sub(1);
        fresh.graphic_mode = -1;
        // Kept across reset as in QEMU: the phases, the depth and the scratch buffers.
        fresh.cursor_visible_phase = s.cursor_visible_phase;
        fresh.cursor_blink_time = s.cursor_blink_time;
        fresh.blink_visible_phase = s.blink_visible_phase;
        fresh.blink_time = s.blink_time;
        fresh.last_depth = s.last_depth;
        fresh.last_byteswap = s.last_byteswap;
        fresh.last_line_offset = s.last_line_offset;
        fresh.full_update_text = s.full_update_text;
        fresh.full_update_gfx = s.full_update_gfx;
        fresh.shared_base = s.shared_base;
        *s = fresh;
    }

    /// `vga_ioport_read()` for the absolute port `addr`.
    pub fn ioport_read(&self, addr: u32) -> u32 {
        let mut s = self.lock();
        if s.ioport_invalid(addr) {
            return 0xff;
        }
        let val = match addr {
            VGA_ATT_W => {
                if !s.ar_flip_flop {
                    s.ar_index
                } else {
                    0
                }
            }
            VGA_ATT_R => {
                let index = usize::from(s.ar_index & 0x1f);
                if index < VGA_ATT_C { s.ar[index] } else { 0 }
            }
            VGA_MIS_W => s.st00,
            VGA_SEQ_I => s.sr_index,
            VGA_SEQ_D => s.sr[usize::from(s.sr_index)],
            VGA_PEL_IR => s.dac_state,
            VGA_PEL_IW => s.dac_write_index,
            VGA_PEL_D => {
                let v = s.palette[usize::from(s.dac_read_index) * 3 + usize::from(s.dac_sub_index)];
                s.dac_sub_index += 1;
                if s.dac_sub_index == 3 {
                    s.dac_sub_index = 0;
                    s.dac_read_index = s.dac_read_index.wrapping_add(1);
                }
                v
            }
            VGA_FTC_R => s.fcr,
            VGA_MIS_R => s.msr,
            VGA_GFX_I => s.gr_index,
            VGA_GFX_D => s.gr[usize::from(s.gr_index)],
            VGA_CRT_IM | VGA_CRT_IC => s.cr_index,
            VGA_CRT_DM | VGA_CRT_DC => s.cr[usize::from(s.cr_index)],
            VGA_IS1_RM | VGA_IS1_RC => {
                // Just toggle to fool polling, `vga_dumb_retrace()`.
                s.st01 ^= ST01_V_RETRACE | ST01_DISP_ENABLE;
                s.ar_flip_flop = false;
                s.st01
            }
            _ => 0,
        };
        u32::from(val)
    }

    /// `vga_ioport_write()` for the absolute port `addr`.
    pub fn ioport_write(&self, addr: u32, val: u32) {
        let mut s = self.lock();
        if s.ioport_invalid(addr) {
            return;
        }
        let val8 = val as u8;
        match addr {
            VGA_ATT_W => {
                if !s.ar_flip_flop {
                    s.ar_index = val8 & 0x3f;
                } else {
                    let index = usize::from(s.ar_index & 0x1f);
                    match index {
                        0x00..=0x0f => s.ar[index] = val8 & 0x3f,
                        VGA_ATC_MODE => s.ar[index] = val8 & !0x10,
                        VGA_ATC_OVERSCAN => s.ar[index] = val8,
                        VGA_ATC_PLANE_ENABLE => s.ar[index] = val8 & !0xc0,
                        VGA_ATC_PEL => s.ar[index] = val8 & !0xf0,
                        VGA_ATC_COLOR_PAGE => s.ar[index] = val8 & !0xf0,
                        _ => {}
                    }
                }
                s.ar_flip_flop = !s.ar_flip_flop;
            }
            VGA_MIS_W => s.msr = val8 & !0x10,
            VGA_SEQ_I => s.sr_index = val8 & 7,
            VGA_SEQ_D => {
                let i = usize::from(s.sr_index);
                s.sr[i] = val8 & SR_MASK[i];
            }
            VGA_PEL_IR => {
                s.dac_read_index = val8;
                s.dac_sub_index = 0;
                s.dac_state = 3;
            }
            VGA_PEL_IW => {
                s.dac_write_index = val8;
                s.dac_sub_index = 0;
                s.dac_state = 0;
            }
            VGA_PEL_D => {
                let sub = usize::from(s.dac_sub_index);
                s.dac_cache[sub] = val8;
                s.dac_sub_index += 1;
                if s.dac_sub_index == 3 {
                    let at = usize::from(s.dac_write_index) * 3;
                    let cache = s.dac_cache;
                    s.palette[at..at + 3].copy_from_slice(&cache);
                    s.dac_sub_index = 0;
                    s.dac_write_index = s.dac_write_index.wrapping_add(1);
                }
            }
            VGA_GFX_I => s.gr_index = val8 & 0x0f,
            VGA_GFX_D => {
                let i = usize::from(s.gr_index);
                s.gr[i] = val8 & GR_MASK[i];
                s.vbe_update_vgaregs();
            }
            VGA_CRT_IM | VGA_CRT_IC => s.cr_index = val8,
            VGA_CRT_DM | VGA_CRT_DC => {
                // CR0 to CR7 are write protected, except bit 4 of CR7.
                if s.cr[VGA_CRTC_V_SYNC_END] & VGA_CR11_LOCK_CR0_CR7 != 0
                    && usize::from(s.cr_index) <= VGA_CRTC_OVERFLOW
                {
                    if usize::from(s.cr_index) == VGA_CRTC_OVERFLOW {
                        s.cr[VGA_CRTC_OVERFLOW] = (s.cr[VGA_CRTC_OVERFLOW] & !0x10) | (val8 & 0x10);
                        s.vbe_update_vgaregs();
                    }
                    return;
                }
                let i = usize::from(s.cr_index);
                s.cr[i] = val8;
                s.vbe_update_vgaregs();
            }
            VGA_IS1_RM | VGA_IS1_RC => s.fcr = val8 & 0x10,
            _ => {}
        }
    }

    /// `vbe_ioport_read_index()`.
    pub fn vbe_read_index(&self) -> u32 {
        u32::from(self.lock().vbe_index)
    }

    /// `vbe_ioport_write_index()`.
    pub fn vbe_write_index(&self, val: u32) {
        self.lock().vbe_index = val as u16;
    }

    /// `vbe_ioport_read_data()`.
    pub fn vbe_read_data(&self) -> u32 {
        let s = self.lock();
        let index = usize::from(s.vbe_index);
        if index < VBE_DISPI_INDEX_NB {
            let v = if s.vbe_regs[VBE_DISPI_INDEX_ENABLE] & VBE_DISPI_GETCAPS != 0 {
                match index {
                    VBE_DISPI_INDEX_XRES => VBE_DISPI_MAX_XRES,
                    VBE_DISPI_INDEX_YRES => VBE_DISPI_MAX_YRES,
                    VBE_DISPI_INDEX_BPP => VBE_DISPI_MAX_BPP,
                    _ => s.vbe_regs[index],
                }
            } else {
                s.vbe_regs[index]
            };
            u32::from(v)
        } else if index == VBE_DISPI_INDEX_VIDEO_MEMORY_64K {
            self.vbe_size / (64 * 1024)
        } else {
            0
        }
    }

    /// `vbe_ioport_write_data()`.
    pub fn vbe_write_data(&self, val: u32) {
        let mut s = self.lock();
        let index = usize::from(s.vbe_index);
        if index > VBE_DISPI_INDEX_NB {
            return;
        }
        let v16 = val as u16;
        match index {
            VBE_DISPI_INDEX_ID => {
                if (u32::from(VBE_DISPI_ID0)..=u32::from(VBE_DISPI_ID5)).contains(&val) {
                    s.vbe_regs[index] = v16;
                }
            }
            VBE_DISPI_INDEX_XRES
            | VBE_DISPI_INDEX_YRES
            | VBE_DISPI_INDEX_BPP
            | VBE_DISPI_INDEX_VIRT_WIDTH
            | VBE_DISPI_INDEX_X_OFFSET
            | VBE_DISPI_INDEX_Y_OFFSET => {
                s.vbe_regs[index] = v16;
                s.vbe_fixup_regs(self.vbe_size);
                s.vbe_update_vgaregs();
            }
            VBE_DISPI_INDEX_BANK => {
                let v = val & s.vbe_bank_mask;
                s.vbe_regs[index] = v as u16;
                s.bank_offset = v << 16;
            }
            VBE_DISPI_INDEX_ENABLE => {
                if v16 & VBE_DISPI_ENABLED != 0
                    && s.vbe_regs[VBE_DISPI_INDEX_ENABLE] & VBE_DISPI_ENABLED == 0
                {
                    s.vbe_regs[VBE_DISPI_INDEX_VIRT_WIDTH] = 0;
                    s.vbe_regs[VBE_DISPI_INDEX_X_OFFSET] = 0;
                    s.vbe_regs[VBE_DISPI_INDEX_Y_OFFSET] = 0;
                    s.vbe_regs[VBE_DISPI_INDEX_ENABLE] |= VBE_DISPI_ENABLED;
                    s.vbe_fixup_regs(self.vbe_size);
                    s.vbe_update_vgaregs();
                    if v16 & VBE_DISPI_NOCLEARMEM == 0 {
                        let len = u64::from(s.vbe_regs[VBE_DISPI_INDEX_YRES])
                            * u64::from(s.vbe_line_offset);
                        let len = len.min(self.vram.len());
                        let _ = self.vram.fill(0, len, 0);
                    }
                } else {
                    s.bank_offset = 0;
                }
                s.dac_8bit = v16 & VBE_DISPI_8BIT_DAC != 0;
                s.vbe_regs[index] = v16;
            }
            _ => {}
        }
    }

    fn load_dword(&self, index: u32) -> u32 {
        let b = self.bytes();
        let at = index as usize * 4;
        u32::from_le_bytes([
            b[at].load(Ordering::Relaxed),
            b[at + 1].load(Ordering::Relaxed),
            b[at + 2].load(Ordering::Relaxed),
            b[at + 3].load(Ordering::Relaxed),
        ])
    }

    fn store_dword(&self, index: u32, v: u32) {
        let b = self.bytes();
        let at = index as usize * 4;
        for (i, byte) in v.to_le_bytes().into_iter().enumerate() {
            b[at + i].store(byte, Ordering::Relaxed);
        }
    }

    /// `vga_mem_readb()`: a byte read at `addr` in the 0xa0000 to 0xbffff window.
    pub fn mem_readb(&self, addr: u64) -> u32 {
        let mut s = self.lock();
        let memory_map_mode = (s.gr[VGA_GFX_MISC] >> 2) & 3;
        let mut addr = (addr & 0x1ffff) as u32;
        match memory_map_mode {
            0 => {}
            1 => {
                if addr >= 0x10000 {
                    return 0xff;
                }
                addr = addr.wrapping_add(s.bank_offset);
            }
            2 => {
                addr = addr.wrapping_sub(0x10000);
                if addr >= 0x8000 {
                    return 0xff;
                }
            }
            _ => {
                addr = addr.wrapping_sub(0x18000);
                if addr >= 0x8000 {
                    return 0xff;
                }
            }
        }

        let plane: u32;
        if s.sr(VGA_SEQ_MEMORY_MODE) & VGA_SR04_CHN_4M != 0 {
            plane = addr & 3;
            addr &= !3;
        } else if s.gr[VGA_GFX_MODE] & VGA_GR05_HOST_ODD_EVEN != 0 {
            plane = u32::from(s.gr[VGA_GFX_PLANE_READ] & 2) | (addr & 1);
        } else {
            plane = u32::from(s.gr[VGA_GFX_PLANE_READ]);
        }

        if s.gr[VGA_GFX_MISC] & VGA_GR06_CHAIN_ODD_EVEN != 0 {
            addr &= !1;
        }

        if s.cr[VGA_CRTC_UNDERLINE] & VGA_CR14_DW != 0 {
            addr >>= 2;
        } else if s.gr[VGA_GFX_MODE] & VGA_GR05_HOST_ODD_EVEN != 0
            && s.cr[VGA_CRTC_MODE] & VGA_CR17_WORD_BYTE == 0
        {
            addr >>= 1;
        }

        if u64::from(addr) * 4 >= u64::from(self.vram_size) {
            return 0xff;
        }

        // QEMU checks the plain SR4 here, not the VBE override.
        if s.sr[VGA_SEQ_MEMORY_MODE] & VGA_SR04_CHN_4M != 0 {
            return u32::from(self.bytes()[((addr << 2) | plane) as usize].load(Ordering::Relaxed));
        }

        s.latch = self.load_dword(addr);
        if s.gr[VGA_GFX_MODE] & 0x08 == 0 {
            (s.latch >> (plane * 8)) & 0xff
        } else {
            let mut ret = (s.latch ^ MASK16[usize::from(s.gr[VGA_GFX_COMPARE_VALUE])])
                & MASK16[usize::from(s.gr[VGA_GFX_COMPARE_MASK])];
            ret |= ret >> 16;
            ret |= ret >> 8;
            (!ret) & 0xff
        }
    }

    /// `vga_mem_writeb()`: a byte write at `addr` in the 0xa0000 to 0xbffff window.
    pub fn mem_writeb(&self, addr: u64, val: u32) {
        let mut s = self.lock();
        let memory_map_mode = (s.gr[VGA_GFX_MISC] >> 2) & 3;
        let mut addr = (addr & 0x1ffff) as u32;
        match memory_map_mode {
            0 => {}
            1 => {
                if addr >= 0x10000 {
                    return;
                }
                addr = addr.wrapping_add(s.bank_offset);
            }
            2 => {
                addr = addr.wrapping_sub(0x10000);
                if addr >= 0x8000 {
                    return;
                }
            }
            _ => {
                addr = addr.wrapping_sub(0x18000);
                if addr >= 0x8000 {
                    return;
                }
            }
        }

        let mut mask = u32::from(s.sr(VGA_SEQ_PLANE_WRITE));
        let mut plane = 0;
        let chain4 = s.sr(VGA_SEQ_MEMORY_MODE) & VGA_SR04_CHN_4M != 0;
        if chain4 {
            plane = addr & 3;
            mask &= 1 << plane;
            addr &= !3;
        } else {
            if s.sr(VGA_SEQ_MEMORY_MODE) & VGA_SR04_SEQ_MODE == 0 {
                mask &= if addr & 1 != 0 { 0x0a } else { 0x05 };
            }
            if s.gr[VGA_GFX_MISC] & VGA_GR06_CHAIN_ODD_EVEN != 0 {
                addr &= !1;
            }
        }

        if s.cr[VGA_CRTC_UNDERLINE] & VGA_CR14_DW != 0 {
            addr >>= 2;
        } else if s.sr(VGA_SEQ_MEMORY_MODE) & VGA_SR04_SEQ_MODE == 0
            && s.cr[VGA_CRTC_MODE] & VGA_CR17_WORD_BYTE == 0
        {
            addr >>= 1;
        }

        if u64::from(addr) * 4 >= u64::from(self.vram_size) {
            return;
        }

        if chain4 {
            if mask != 0 {
                self.bytes()[((addr << 2) | plane) as usize].store(val as u8, Ordering::Relaxed);
                s.plane_updated |= mask;
            }
            return;
        }

        let gr = &s.gr;
        let write_mode = gr[VGA_GFX_MODE] & 3;
        let mut val = val;
        let bit_mask: u32;
        if write_mode == 1 {
            val = s.latch;
        } else {
            match write_mode {
                0 => {
                    let b = u32::from(gr[VGA_GFX_DATA_ROTATE] & 7);
                    val = ((val >> b) | val.wrapping_shl(8 - b)) & 0xff;
                    val |= val << 8;
                    val |= val << 16;
                    let set_mask = MASK16[usize::from(gr[VGA_GFX_SR_ENABLE])];
                    val =
                        (val & !set_mask) | (MASK16[usize::from(gr[VGA_GFX_SR_VALUE])] & set_mask);
                    bit_mask = u32::from(gr[VGA_GFX_BIT_MASK]);
                }
                2 => {
                    val = MASK16[(val & 0x0f) as usize];
                    bit_mask = u32::from(gr[VGA_GFX_BIT_MASK]);
                }
                _ => {
                    let b = u32::from(gr[VGA_GFX_DATA_ROTATE] & 7);
                    val = (val >> b) | val.wrapping_shl(8 - b);
                    bit_mask = u32::from(gr[VGA_GFX_BIT_MASK]) & val;
                    val = MASK16[usize::from(gr[VGA_GFX_SR_VALUE])];
                }
            }

            match gr[VGA_GFX_DATA_ROTATE] >> 3 {
                1 => val &= s.latch,
                2 => val |= s.latch,
                3 => val ^= s.latch,
                _ => {}
            }

            let mut bit_mask = bit_mask;
            bit_mask |= bit_mask << 8;
            bit_mask |= bit_mask << 16;
            val = (val & bit_mask) | (s.latch & !bit_mask);
        }

        s.plane_updated |= mask;
        let write_mask = MASK16[mask as usize];
        let old = self.load_dword(addr);
        self.store_dword(addr, (old & !write_mask) | (val & write_mask));
    }

    /// The I/O regions of the VGA and VBE ports, `vga_init_io()` and `portio_list_add()`: the
    /// port each region starts at, its size, and its ops. `x86` picks the VBE layout with data
    /// at both 0x1cf and 0x1d0, which only x86 machines get.
    pub fn portio_regions(self: &Arc<Self>, x86: bool) -> Vec<(u16, u64, Arc<dyn MmioOps>)> {
        let vga_ports = [
            PortioEntry { offset: 0x04, len: 2, size: 1, func: PortFn::Vga },
            PortioEntry { offset: 0x0a, len: 1, size: 1, func: PortFn::Vga },
            PortioEntry { offset: 0x10, len: 16, size: 1, func: PortFn::Vga },
            PortioEntry { offset: 0x24, len: 2, size: 1, func: PortFn::Vga },
            PortioEntry { offset: 0x2a, len: 1, size: 1, func: PortFn::Vga },
        ];
        let vbe_x86 = [
            PortioEntry { offset: 0, len: 1, size: 2, func: PortFn::VbeIndex },
            PortioEntry { offset: 1, len: 1, size: 2, func: PortFn::VbeData },
            PortioEntry { offset: 2, len: 1, size: 2, func: PortFn::VbeData },
        ];
        let vbe_other = [
            PortioEntry { offset: 0, len: 1, size: 2, func: PortFn::VbeIndex },
            PortioEntry { offset: 2, len: 1, size: 2, func: PortFn::VbeData },
        ];
        let mut out = Vec::new();
        portio_list_add(self, &vga_ports, 0x3b0, &mut out);
        portio_list_add(self, if x86 { &vbe_x86[..] } else { &vbe_other[..] }, 0x1ce, &mut out);
        out
    }

    /// `vga_invalidate_display()`.
    pub fn invalidate_display(&self) {
        let mut s = self.lock();
        s.last_width = u32::MAX;
        s.last_height = u32::MAX;
    }

    /// `vga_update_display()`.
    pub fn update_display(&self, con: &QemuConsole) {
        let mut s = self.lock();
        let mut rects = Vec::new();
        let mut full_update = false;
        let graphic_mode = if s.ar_index & 0x20 == 0 {
            GMODE_BLANK
        } else {
            i32::from(s.gr[VGA_GFX_MISC] & VGA_GR06_GRAPHICS_MODE)
        };
        if graphic_mode != s.graphic_mode {
            s.graphic_mode = graphic_mode;
            s.cursor_blink_time = self.now_ms();
            full_update = true;
        }
        match graphic_mode {
            GMODE_TEXT => self.draw_text(&mut s, con, full_update, &mut rects),
            GMODE_GRAPH => self.draw_graphic(&mut s, con, full_update, &mut rects),
            _ => self.draw_blank(&mut s, con, full_update, &mut rects),
        }
        drop(s);
        for (x, y, w, h) in rects {
            con.update(x, y, w, h);
        }
    }

    /// `vga_draw_text()`.
    fn draw_text(
        &self,
        s: &mut VgaState,
        con: &QemuConsole,
        full_update: bool,
        rects: &mut Vec<(i32, i32, i32, i32)>,
    ) {
        let mut full_update = full_update;
        let now = self.now_ms();

        let v = u32::from(s.sr(VGA_SEQ_CHARACTER_MAP));
        let offset = (((v >> 4) & 1) | ((v << 1) & 6)) * 8192 * 4 + 2;
        if offset != s.font_offsets[0] {
            s.font_offsets[0] = offset;
            full_update = true;
        }
        let font_base0 = offset;
        let offset = (((v >> 5) & 1) | ((v >> 1) & 6)) * 8192 * 4 + 2;
        let font_base1 = offset;
        if offset != s.font_offsets[1] {
            s.font_offsets[1] = offset;
            full_update = true;
        }
        if s.plane_updated & (1 << 2) != 0 {
            // A write to plane 2 since the last update may have changed the font.
            s.plane_updated = 0;
            full_update = true;
        }
        full_update |= s.update_basic_params();

        let line_offset = s.params.line_offset;
        let (width, height, cw, cheight) = s.get_text_resolution();
        if height * width <= 1 {
            return;
        }
        if (height * width) as usize > CH_ATTR_SIZE {
            return;
        }

        if width != s.last_width
            || height != s.last_height
            || cw != u32::from(s.last_cw)
            || cheight != u32::from(s.last_ch)
            || s.last_depth != 0
        {
            s.last_scr_width = width * cw;
            s.last_scr_height = height * cheight;
            con.resize(s.last_scr_width as usize, s.last_scr_height as usize);
            s.shared_base = None;
            s.last_depth = 0;
            s.last_width = width;
            s.last_height = height;
            s.last_ch = cheight as u8;
            s.last_cw = cw as u8;
            full_update = true;
        }
        full_update |= s.update_palette16();
        let palette = s.last_palette;

        if full_update {
            s.full_update_text = true;
        }
        if s.full_update_gfx {
            s.full_update_gfx = false;
            full_update = true;
        }

        let cursor_offset = ((u32::from(s.cr[VGA_CRTC_CURSOR_HI]) << 8)
            | u32::from(s.cr[VGA_CRTC_CURSOR_LO]))
        .wrapping_sub(s.params.start_addr);
        if cursor_offset != s.cursor_offset
            || s.cr[VGA_CRTC_CURSOR_START] != s.cursor_start
            || s.cr[VGA_CRTC_CURSOR_END] != s.cursor_end
        {
            // The cursor moved: redraw the characters under the old and the new position.
            if (s.cursor_offset as usize) < CH_ATTR_SIZE {
                let i = s.cursor_offset as usize;
                s.last_ch_attr[i] = u32::MAX;
            }
            if (cursor_offset as usize) < CH_ATTR_SIZE {
                s.last_ch_attr[cursor_offset as usize] = u32::MAX;
            }
            s.cursor_offset = cursor_offset;
            s.cursor_start = s.cr[VGA_CRTC_CURSOR_START];
            s.cursor_end = s.cr[VGA_CRTC_CURSOR_END];
        }
        let cursor_addr = s.params.start_addr.wrapping_add(cursor_offset).wrapping_mul(4);
        if now >= s.cursor_blink_time {
            s.cursor_blink_time = now + VGA_TEXT_CURSOR_PERIOD_MS / 2;
            s.cursor_visible_phase = !s.cursor_visible_phase;
        }
        if now >= s.blink_time {
            s.blink_time = now + VGA_TEXT_BLINK_PERIOD_MS / 2;
            s.blink_visible_phase = !s.blink_visible_phase;
            if s.ar[VGA_ATC_MODE] & 0x08 != 0 {
                full_update = true;
            }
        }

        let bytes = self.bytes();
        let vram_size = self.vram_size as usize;
        let byte = |a: usize| bytes.get(a).map_or(0, |b| b.load(Ordering::Relaxed));
        let cr = s.cr;
        let ar_mode = s.ar[VGA_ATC_MODE];
        let blink_visible = s.blink_visible_phase;
        let cursor_visible = s.cursor_visible_phase;
        let line_compare = s.params.line_compare;
        let start_addr = s.params.start_addr;
        let ch_attr_cache = &mut s.last_ch_attr;

        con.with_surface_mut(|surface| {
            let Some(surface) = surface else { return };
            if surface.bits_per_pixel() != 32 {
                return;
            }
            let linesize = surface.stride();
            let x_incr = cw as usize * 4;
            let mut dest = 0usize;
            let data = surface.data_mut();
            let mut ch_attr_idx = 0usize;
            let mut line = 0u32;
            let mut offset = start_addr as usize * 4;
            let mut font = [0u8; 32];
            let mut cursor_font = [0u8; 32];
            for (i, f) in cursor_font.iter_mut().enumerate() {
                *f = CURSOR_GLYPH[i * 4];
            }
            for cy in 0..height {
                let mut d1 = dest;
                let mut src = offset;
                let mut cx_min = width as i32;
                let mut cx_max = -1i32;
                for cx in 0..width {
                    if src + 2 > vram_size {
                        break;
                    }
                    let ch = byte(src);
                    let cattr = byte(src + 1);
                    let ch_attr = u32::from(ch) | (u32::from(cattr) << 8);
                    let is_cursor = src as u32 == cursor_addr;
                    if full_update || ch_attr != ch_attr_cache[ch_attr_idx] || is_cursor {
                        cx_min = cx_min.min(cx as i32);
                        cx_max = cx_max.max(cx as i32);
                        ch_attr_cache[ch_attr_idx] = ch_attr;
                        let base = if (cattr >> 3) & 1 != 0 { font_base1 } else { font_base0 };
                        let font_at = base as usize + 32 * 4 * usize::from(ch);
                        for (i, f) in font.iter_mut().enumerate() {
                            *f = byte(font_at + i * 4);
                        }
                        let (fgcol, bgcol);
                        if ar_mode & 0x08 != 0 {
                            bgcol = palette[usize::from((cattr >> 4) & 0x07)];
                            fgcol = if cattr & 0x80 != 0 && !blink_visible {
                                bgcol
                            } else {
                                palette[usize::from(cattr & 0x0f)]
                            };
                        } else {
                            bgcol = palette[usize::from(cattr >> 4)];
                            fgcol = palette[usize::from(cattr & 0x0f)];
                        }
                        let dup9 = (0xb0..=0xdf).contains(&ch) && ar_mode & 0x04 != 0;
                        draw_glyph(data, d1, linesize, &font, cheight, cw, fgcol, bgcol, dup9);
                        if is_cursor && cr[VGA_CRTC_CURSOR_START] & 0x20 == 0 && cursor_visible {
                            let line_start = u32::from(cr[VGA_CRTC_CURSOR_START] & 0x1f);
                            let mut line_last = u32::from(cr[VGA_CRTC_CURSOR_END] & 0x1f);
                            if line_last > cheight - 1 {
                                line_last = cheight - 1;
                            }
                            if line_last >= line_start && line_start < cheight {
                                let h = line_last - line_start + 1;
                                let d = d1 + linesize * line_start as usize;
                                draw_glyph(
                                    data,
                                    d,
                                    linesize,
                                    &cursor_font,
                                    h,
                                    cw,
                                    fgcol,
                                    bgcol,
                                    true,
                                );
                            }
                        }
                    }
                    d1 += x_incr;
                    src += 4;
                    ch_attr_idx += 1;
                }
                if cx_max != -1 {
                    rects.push((
                        cx_min * cw as i32,
                        (cy * cheight) as i32,
                        (cx_max - cx_min + 1) * cw as i32,
                        cheight as i32,
                    ));
                }
                dest += linesize * cheight as usize;
                let line1 = line + cheight;
                offset = offset.wrapping_add(line_offset as usize);
                if line < line_compare && line1 >= line_compare {
                    offset = 0;
                }
                line = line1;
            }
        });
    }

    /// `vga_draw_graphic()`.
    fn draw_graphic(
        &self,
        s: &mut VgaState,
        con: &QemuConsole,
        full_update: bool,
        rects: &mut Vec<(i32, i32, i32, i32)>,
    ) {
        let mut full_update = full_update;
        let byteswap = if cfg!(target_endian = "big") { !s.big_endian_fb } else { s.big_endian_fb };

        full_update |= s.update_basic_params();

        let (width, height) = s.get_resolution();
        let mut disp_width = width;
        let depth = s.get_bpp();

        let shift_control = (s.gr[VGA_GFX_MODE] >> 5) & 3;
        let double_scan = s.cr[VGA_CRTC_MAX_SCAN] >> 7;
        let multi_scan: u32 = if s.cr[VGA_CRTC_MODE] & 1 != 0 {
            ((u32::from(s.cr[VGA_CRTC_MAX_SCAN] & 0x1f) + 1) << double_scan) - 1
        } else {
            u32::from(double_scan)
        };
        let mut multi_run = multi_scan;
        if shift_control != s.shift_control || double_scan != s.double_scan {
            full_update = true;
            s.shift_control = shift_control;
            s.double_scan = double_scan;
        }

        let kind;
        let bits: u32;
        if shift_control == 0 {
            full_update |= s.update_palette16();
            if s.sr(VGA_SEQ_CLOCK_MODE) & 8 != 0 {
                disp_width <<= 1;
                kind = LineKind::Line4d2;
            } else {
                kind = LineKind::Line4;
            }
            bits = 4;
        } else if shift_control == 1 {
            full_update |= s.update_palette16();
            if s.sr(VGA_SEQ_CLOCK_MODE) & 8 != 0 {
                disp_width <<= 1;
                kind = LineKind::Line2d2;
            } else {
                kind = LineKind::Line2;
            }
            bits = 4;
        } else {
            let be = s.big_endian_fb;
            match depth {
                8 => {
                    full_update |= s.update_palette256();
                    kind = LineKind::Line8;
                    bits = 8;
                }
                15 => {
                    kind = if be { LineKind::Line15Be } else { LineKind::Line15Le };
                    bits = 16;
                }
                16 => {
                    kind = if be { LineKind::Line16Be } else { LineKind::Line16Le };
                    bits = 16;
                }
                24 => {
                    kind = if be { LineKind::Line24Be } else { LineKind::Line24Le };
                    bits = 24;
                }
                32 => {
                    kind = if be { LineKind::Line32Be } else { LineKind::Line32Le };
                    bits = 32;
                }
                _ => {
                    full_update |= s.update_palette256();
                    kind = LineKind::Line8d2;
                    bits = 4;
                }
            }
        }

        // Bit 3 of the pel panning register only matters in text mode.
        let mut hpel = if bits <= 8 { s.params.hpel & 7 } else { 0 };
        let mut bwidth = u64::from((width * bits).div_ceil(8));
        if hpel != 0 {
            bwidth += 4;
        }

        let region_start = u64::from(s.params.start_addr) * 4;
        let region_end = region_start
            + u64::from(s.params.line_offset) * u64::from(height.saturating_sub(1))
            + bwidth;
        // On wrap around QEMU takes the slow route through a shadow surface.
        let force_shadow = region_end > u64::from(self.vbe_size);

        let format = default_pixman_format(depth, !byteswap);
        let mut allocate_surface = match format {
            Some(f) => !con.check_format(f) || force_shadow,
            None => true,
        };
        // A pitch shorter than a row cannot back a surface here.
        if let Some(f) = format {
            if (s.params.line_offset as usize) < disp_width as usize * f.bytes_per_pixel() {
                allocate_surface = true;
            }
        }

        let is_allocated = con.with_surface(|surf| surf.is_none_or(|x| x.is_allocated()));
        if s.params.line_offset != s.last_line_offset
            || disp_width != s.last_width
            || height != s.last_height
            || s.last_depth != depth
            || s.last_byteswap != byteswap
            || allocate_surface != is_allocated
        {
            s.last_scr_width = disp_width;
            s.last_scr_height = height;
            s.last_width = disp_width;
            s.last_height = height;
            s.last_line_offset = s.params.line_offset;
            s.last_depth = depth;
            s.last_byteswap = byteswap;
            full_update = true;
        }

        if !is_allocated && s.shared_base != Some(s.params.start_addr) {
            // Page flip: a shared surface follows the new base address.
            full_update = true;
        }

        if full_update {
            if !allocate_surface {
                let format = format.unwrap_or_else(|| unreachable!());
                con.set_surface(Some(DisplaySurface::new_from(
                    disp_width as usize,
                    height as usize,
                    format,
                    s.params.line_offset as usize,
                )));
                s.shared_base = Some(s.params.start_addr);
            } else {
                con.resize(disp_width as usize, height as usize);
                s.shared_base = None;
            }
        }

        let shared = con.with_surface(|surf| surf.is_some_and(|x| !x.is_allocated()));
        if shared {
            self.copy_shared(s, con, full_update, rects);
            return;
        }

        let need = disp_width as usize + 16;
        s.panning_buf.resize(need, 0);
        s.row_buf.resize(need, 0);
        let mut panning = std::mem::take(&mut s.panning_buf);
        let mut row = std::mem::take(&mut s.row_buf);
        let palette = s.last_palette;
        let cx = LineCtx {
            vram: self.bytes(),
            mask: self.vbe_size_mask,
            palette: &palette,
            plane_mask: MASK16[usize::from(s.ar[VGA_ATC_PLANE_ENABLE] & 0xf)],
        };
        let cr17 = s.cr[VGA_CRTC_MODE];
        let params = s.params;

        con.with_surface_mut(|surf| {
            let Some(surf) = surf else { return };
            if surf.bits_per_pixel() != 32 {
                return;
            }
            let rows = surf.height().min(height as usize);
            let cols = surf.width().min(disp_width as usize);
            let stride = surf.stride();
            let data = surf.data_mut();
            let mut addr1 = params.start_addr.wrapping_mul(4);
            let mut y_start: i32 = -1;
            let mut y1: u32 = 0;
            let mut y = 0usize;
            while y < height as usize {
                let mut addr = addr1;
                if cr17 & 1 == 0 {
                    // CGA compatibility addressing.
                    let shift = 14 + u32::from((cr17 >> 6) & 1);
                    addr = (addr & !(1 << shift)) | ((y1 & 1) << shift);
                }
                if cr17 & 2 == 0 {
                    addr = (addr & !0x8000) | ((y1 & 2) << 14);
                }
                let mut update = full_update;
                if y < rows {
                    let line = &mut data[y * stride..y * stride + cols * 4];
                    for (p, c) in row.iter_mut().zip(line.chunks_exact(4)) {
                        *p = u32::from_ne_bytes([c[0], c[1], c[2], c[3]]);
                    }
                    let src =
                        match cx.draw(kind, &mut row, &mut panning, addr, width, u32::from(hpel)) {
                            Some(off) => &panning[off..off + cols],
                            None => &row[..cols],
                        };
                    for (c, p) in line.chunks_exact_mut(4).zip(src) {
                        let b = p.to_ne_bytes();
                        if c != b {
                            c.copy_from_slice(&b);
                            update = true;
                        }
                    }
                }
                if update {
                    if y_start < 0 {
                        y_start = y as i32;
                    }
                } else if y_start >= 0 {
                    rects.push((0, y_start, disp_width as i32, y as i32 - y_start));
                    y_start = -1;
                }
                if multi_run == 0 {
                    let mask = u32::from(cr17 & 3) ^ 3;
                    if (y1 & mask) == mask {
                        addr1 = addr1.wrapping_add(params.line_offset);
                    }
                    y1 += 1;
                    multi_run = multi_scan;
                } else {
                    multi_run -= 1;
                }
                // Line compare acts on the displayed lines.
                if y as u32 == params.line_compare {
                    if params.hpel_split {
                        hpel = VGA_HPEL_NEUTRAL;
                    }
                    addr1 = 0;
                }
                y += 1;
            }
            if y_start >= 0 {
                rects.push((0, y_start, disp_width as i32, y as i32 - y_start));
            }
        });
        s.panning_buf = panning;
        s.row_buf = row;
    }

    /// The direct colour modes with a shared surface: copies the frame out of VRAM and reports
    /// the rows that changed.
    fn copy_shared(
        &self,
        s: &VgaState,
        con: &QemuConsole,
        full_update: bool,
        rects: &mut Vec<(i32, i32, i32, i32)>,
    ) {
        let base = u64::from(s.params.start_addr) * 4;
        con.with_surface_mut(|surf| {
            let Some(surf) = surf else { return };
            let stride = surf.stride();
            let h = surf.height();
            let w = surf.width() as i32;
            let data = surf.data_mut();
            let avail = self.vram.len().saturating_sub(base) as usize;
            let len = data.len().min(avail);
            let mut fresh = vec![0u8; len];
            let _ = self.vram.read(base, &mut fresh);
            let mut y_start: i32 = -1;
            for y in 0..h {
                let a = (y * stride).min(len);
                let b = ((y + 1) * stride).min(len);
                let changed = data[a..b] != fresh[a..b];
                if changed {
                    data[a..b].copy_from_slice(&fresh[a..b]);
                }
                if full_update || changed {
                    if y_start < 0 {
                        y_start = y as i32;
                    }
                } else if y_start >= 0 {
                    rects.push((0, y_start, w, y as i32 - y_start));
                    y_start = -1;
                }
            }
            if y_start >= 0 {
                rects.push((0, y_start, w, h as i32 - y_start));
            }
        });
    }

    /// `vga_draw_blank()`.
    fn draw_blank(
        &self,
        s: &mut VgaState,
        con: &QemuConsole,
        full_update: bool,
        rects: &mut Vec<(i32, i32, i32, i32)>,
    ) {
        if !full_update {
            return;
        }
        if s.last_scr_width == 0 || s.last_scr_height == 0 {
            return;
        }
        let is_allocated = con.with_surface(|surf| surf.is_none_or(|x| x.is_allocated()));
        if !is_allocated {
            // Unshare the buffer, or blanking would wipe VRAM.
            con.set_surface(Some(DisplaySurface::new(
                s.last_scr_width as usize,
                s.last_scr_height as usize,
            )));
            s.shared_base = None;
        }
        let (w, h) = (s.last_scr_width as usize, s.last_scr_height as usize);
        con.with_surface_mut(|surf| {
            let Some(surf) = surf else { return };
            let bpp = surf.bytes_per_pixel();
            let stride = surf.stride();
            let rows = h.min(surf.height());
            let data = surf.data_mut();
            for y in 0..rows {
                let start = y * stride;
                let end = (start + w * bpp).min(start + stride).min(data.len());
                data[start..end].fill(0);
            }
        });
        let (fw, fh) = (con.width(0), con.height(0));
        rects.push((0, 0, fw, fh));
    }
}

impl GraphicHwOps for VgaCommon {
    fn invalidate(&self) {
        self.invalidate_display();
    }

    fn gfx_update(&self, con: &QemuConsole) -> bool {
        self.update_display(con);
        true
    }
}

/// Draws one character cell: `vga_draw_glyph8()`, `vga_draw_glyph9()` or `vga_draw_glyph16()`
/// by the cell width. `font` holds one byte per glyph row.
#[allow(clippy::too_many_arguments)]
fn draw_glyph(
    data: &mut [u8],
    at: usize,
    linesize: usize,
    font: &[u8; 32],
    h: u32,
    cw: u32,
    fgcol: u32,
    bgcol: u32,
    dup9: bool,
) {
    let xorcol = bgcol ^ fgcol;
    let bit = |font_data: u32, n: u32| (0u32.wrapping_sub((font_data >> n) & 1) & xorcol) ^ bgcol;
    let mut d = at;
    for &font_data in font.iter().take(h as usize) {
        let font_data = u32::from(font_data);
        let mut px = [0u32; 16];
        let n = match cw {
            16 => {
                let hi = u32::from(EXPAND4TO8[(font_data >> 4) as usize]);
                let lo = u32::from(EXPAND4TO8[(font_data & 0x0f) as usize]);
                for i in 0..8 {
                    px[i] = bit(hi, 7 - i as u32);
                    px[8 + i] = bit(lo, 7 - i as u32);
                }
                16
            }
            9 => {
                for (i, p) in px.iter_mut().take(8).enumerate() {
                    *p = bit(font_data, 7 - i as u32);
                }
                px[8] = if dup9 { px[7] } else { bgcol };
                9
            }
            _ => {
                for (i, p) in px.iter_mut().take(8).enumerate() {
                    *p = bit(font_data, 7 - i as u32);
                }
                8
            }
        };
        for (i, p) in px.iter().take(n).enumerate() {
            let o = d + i * 4;
            if let Some(dst) = data.get_mut(o..o + 4) {
                dst.copy_from_slice(&p.to_ne_bytes());
            }
        }
        d += linesize;
    }
}

/// The scanline renderers of `vga_draw_line_table`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LineKind {
    Line2,
    Line2d2,
    Line4,
    Line4d2,
    Line8d2,
    Line8,
    Line15Le,
    Line16Le,
    Line24Le,
    Line32Le,
    Line15Be,
    Line16Be,
    Line24Be,
    Line32Be,
}

/// What the scanline renderers read: VRAM through `vbe_size_mask`, the palette and the plane
/// enable mask.
struct LineCtx<'a> {
    vram: &'a [AtomicU8],
    mask: u32,
    palette: &'a [u32; 256],
    plane_mask: u32,
}

impl LineCtx<'_> {
    fn byte(&self, addr: u32) -> u32 {
        u32::from(self.vram[(addr & self.mask) as usize].load(Ordering::Relaxed))
    }

    fn word(&self, addr: u32, be: bool) -> u32 {
        let o = addr & self.mask & !1;
        let b = [
            self.vram[o as usize].load(Ordering::Relaxed),
            self.vram[o as usize + 1].load(Ordering::Relaxed),
        ];
        u32::from(if be { u16::from_be_bytes(b) } else { u16::from_le_bytes(b) })
    }

    fn dword_le(&self, addr: u32) -> u32 {
        let o = (addr & self.mask & !3) as usize;
        u32::from_le_bytes([
            self.vram[o].load(Ordering::Relaxed),
            self.vram[o + 1].load(Ordering::Relaxed),
            self.vram[o + 2].load(Ordering::Relaxed),
            self.vram[o + 3].load(Ordering::Relaxed),
        ])
    }

    /// Renders one scanline into `row`, or into `pan` when the line is panned, in which case the
    /// return value is where in `pan` the visible pixels start.
    fn draw(
        &self,
        kind: LineKind,
        row: &mut [u32],
        pan: &mut [u32],
        addr: u32,
        width: u32,
        hpel: u32,
    ) -> Option<usize> {
        let pal = self.palette;
        let mut addr = addr;
        let mut width = width as usize;
        match kind {
            LineKind::Line2 | LineKind::Line2d2 | LineKind::Line4 | LineKind::Line4d2 => {
                let hpel = (hpel & 7) as usize;
                let d: &mut [u32] = if hpel != 0 {
                    width += 8;
                    pan
                } else {
                    row
                };
                let dup = matches!(kind, LineKind::Line2d2 | LineKind::Line4d2);
                let mut o = 0;
                for _ in 0..width >> 3 {
                    let data = self.dword_le(addr & (VGA_VRAM_SIZE - 1)) & self.plane_mask;
                    let plane = |p: u32| ((data >> (p * 8)) & 0xff) as usize;
                    let mut px = [0u32; 8];
                    if matches!(kind, LineKind::Line2 | LineKind::Line2d2) {
                        let v = EXPAND2[plane(0)] | (EXPAND2[plane(2)] << 2);
                        for (i, p) in px.iter_mut().take(4).enumerate() {
                            *p = pal[((v >> (12 - 4 * i)) & 0xf) as usize];
                        }
                        let v = EXPAND2[plane(1)] | (EXPAND2[plane(3)] << 2);
                        for (i, p) in px.iter_mut().skip(4).enumerate() {
                            *p = pal[((v >> (12 - 4 * i)) & 0xf) as usize];
                        }
                    } else {
                        let v = EXPAND4[plane(0)]
                            | (EXPAND4[plane(1)] << 1)
                            | (EXPAND4[plane(2)] << 2)
                            | (EXPAND4[plane(3)] << 3);
                        for (i, p) in px.iter_mut().enumerate() {
                            *p = pal[((v >> (28 - 4 * i)) & 0xf) as usize];
                        }
                    }
                    if dup {
                        for (i, &p) in px.iter().enumerate() {
                            d[o + 2 * i] = p;
                            d[o + 2 * i + 1] = p;
                        }
                        o += 16;
                    } else {
                        d[o..o + 8].copy_from_slice(&px);
                        o += 8;
                    }
                    addr = addr.wrapping_add(4);
                }
                if hpel != 0 { Some(if dup { 2 * hpel } else { hpel }) } else { None }
            }
            LineKind::Line8d2 => {
                let mut hpel = ((hpel >> 1) & 3) as usize;
                // Panning moves the source address unless the line would wrap in a plane.
                if addr.wrapping_add((width as u32 >> 3) * 4) < VGA_VRAM_SIZE {
                    addr = addr.wrapping_add(hpel as u32 * 4);
                    hpel = 0;
                }
                let d: &mut [u32] = if hpel != 0 {
                    width += 8;
                    pan
                } else {
                    row
                };
                let mut o = 0;
                for _ in 0..width >> 3 {
                    addr &= VGA_VRAM_SIZE - 1;
                    for i in 0..4 {
                        let p = pal[self.byte(addr + i) as usize];
                        d[o + 2 * i as usize] = p;
                        d[o + 2 * i as usize + 1] = p;
                    }
                    o += 8;
                    addr += 4;
                }
                if hpel != 0 { Some(2 * hpel) } else { None }
            }
            LineKind::Line8 => {
                let hpel = ((hpel >> 1) & 3) as usize;
                let d: &mut [u32] = if hpel != 0 {
                    width += 8;
                    pan
                } else {
                    row
                };
                let mut o = 0;
                for _ in 0..width >> 3 {
                    for i in 0..8 {
                        d[o + i] = pal[self.byte(addr.wrapping_add(i as u32)) as usize];
                    }
                    o += 8;
                    addr = addr.wrapping_add(8);
                }
                if hpel != 0 { Some(hpel) } else { None }
            }
            LineKind::Line15Le | LineKind::Line15Be => {
                let be = kind == LineKind::Line15Be;
                for p in row.iter_mut().take(width) {
                    let v = self.word(addr, be);
                    let r = (v >> 7) & 0xf8;
                    let g = (v >> 2) & 0xf8;
                    let b = (v << 3) & 0xf8;
                    *p = rgb_to_pixel32(r, g, b);
                    addr = addr.wrapping_add(2);
                }
                None
            }
            LineKind::Line16Le | LineKind::Line16Be => {
                let be = kind == LineKind::Line16Be;
                for p in row.iter_mut().take(width) {
                    let v = self.word(addr, be);
                    let r = (v >> 8) & 0xf8;
                    let g = (v >> 3) & 0xfc;
                    let b = (v << 3) & 0xf8;
                    *p = rgb_to_pixel32(r, g, b);
                    addr = addr.wrapping_add(2);
                }
                None
            }
            LineKind::Line24Le | LineKind::Line24Be | LineKind::Line32Le | LineKind::Line32Be => {
                let step =
                    if matches!(kind, LineKind::Line24Le | LineKind::Line24Be) { 3 } else { 4 };
                for p in row.iter_mut().take(width) {
                    let (r, g, b) = match kind {
                        LineKind::Line24Le | LineKind::Line32Le => (
                            self.byte(addr.wrapping_add(2)),
                            self.byte(addr.wrapping_add(1)),
                            self.byte(addr),
                        ),
                        LineKind::Line24Be => (
                            self.byte(addr),
                            self.byte(addr.wrapping_add(1)),
                            self.byte(addr.wrapping_add(2)),
                        ),
                        _ => (
                            self.byte(addr.wrapping_add(1)),
                            self.byte(addr.wrapping_add(2)),
                            self.byte(addr.wrapping_add(3)),
                        ),
                    };
                    *p = rgb_to_pixel32(r, g, b);
                    addr = addr.wrapping_add(step);
                }
                None
            }
        }
    }
}

/// Which handler a port entry calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PortFn {
    Vga,
    VbeIndex,
    VbeData,
}

/// `MemoryRegionPortio`: `len` ports from `offset`, each taking accesses of `size` bytes.
#[derive(Clone, Copy, Debug)]
struct PortioEntry {
    offset: u32,
    len: u32,
    size: u32,
    func: PortFn,
}

/// One region of a `PortioList`, `MemoryRegionPortioList` with `portio_ops`.
#[derive(Debug)]
struct VgaPortio {
    vga: Arc<VgaCommon>,
    /// The port of offset 0 of the region.
    base: u32,
    ports: Vec<PortioEntry>,
}

impl VgaPortio {
    /// `find_portio()`.
    fn find(&self, offset: u32, size: u32) -> Option<PortioEntry> {
        self.ports
            .iter()
            .copied()
            .find(|p| offset >= p.offset && offset < p.offset + p.len && size == p.size)
    }

    fn call_read(&self, p: PortioEntry, port: u32) -> u64 {
        u64::from(match p.func {
            PortFn::Vga => self.vga.ioport_read(port),
            PortFn::VbeIndex => self.vga.vbe_read_index(),
            PortFn::VbeData => self.vga.vbe_read_data(),
        })
    }

    fn call_write(&self, p: PortioEntry, port: u32, val: u32) {
        match p.func {
            PortFn::Vga => self.vga.ioport_write(port, val),
            PortFn::VbeIndex => self.vga.vbe_write_index(val),
            PortFn::VbeData => self.vga.vbe_write_data(val),
        }
    }
}

impl MmioOps for VgaPortio {
    /// `portio_read()`.
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        let off = offset as u32;
        let size = size.bytes();
        if let Some(p) = self.find(off, size) {
            return Ok(self.call_read(p, self.base + off));
        }
        if size == 2 {
            if let Some(p) = self.find(off, 1) {
                let mut data = self.call_read(p, self.base + off);
                if off + 1 < p.offset + p.len {
                    data |= self.call_read(p, self.base + off + 1) << 8;
                } else {
                    data |= 0xff00;
                }
                return Ok(data);
            }
        }
        Ok((1u64 << (size * 8)) - 1)
    }

    /// `portio_write()`.
    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        let off = offset as u32;
        let size = size.bytes();
        if let Some(p) = self.find(off, size) {
            self.call_write(p, self.base + off, value as u32);
        } else if size == 2 {
            if let Some(p) = self.find(off, 1) {
                self.call_write(p, self.base + off, (value & 0xff) as u32);
                if off + 1 < p.offset + p.len {
                    self.call_write(p, self.base + off + 1, ((value >> 8) & 0xff) as u32);
                }
            }
        }
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4).allow_unaligned()
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4).allow_unaligned()
    }
}

/// `portio_list_add()`: splits `list` into regions at the holes and adds them to `out`.
fn portio_list_add(
    vga: &Arc<VgaCommon>,
    list: &[PortioEntry],
    start: u32,
    out: &mut Vec<(u16, u64, Arc<dyn MmioOps>)>,
) {
    let mut push = |entries: &[PortioEntry], off_low: u32, off_high: u32| {
        let ports =
            entries.iter().map(|p| PortioEntry { offset: p.offset - off_low, ..*p }).collect();
        let ops = VgaPortio { vga: Arc::clone(vga), base: start + off_low, ports };
        out.push((
            (start + off_low) as u16,
            u64::from(off_high - off_low),
            Arc::new(ops) as Arc<dyn MmioOps>,
        ));
    };
    let mut first = 0;
    let mut off_low = list[0].offset;
    let mut off_last = off_low;
    let mut off_high = off_low + list[0].len + list[0].size - 1;
    for (i, pio) in list.iter().enumerate().skip(1) {
        off_last = off_last.max(pio.offset);
        if off_last > off_high {
            push(&list[first..i], off_low, off_high);
            first = i;
            off_low = off_last;
            off_high = off_low + pio.len + pio.size - 1;
        } else if off_last + pio.len > off_high {
            off_high = off_last + pio.len + list[first].size - 1;
        }
    }
    push(&list[first..], off_low, off_high);
}

/// The legacy window at 0xa0000, "vga-lowmem", `vga_mem_ops`.
#[derive(Debug)]
pub struct VgaLowmem {
    vga: Arc<VgaCommon>,
}

impl VgaLowmem {
    pub fn new(vga: Arc<VgaCommon>) -> VgaLowmem {
        VgaLowmem { vga }
    }
}

impl MmioOps for VgaLowmem {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.vga.mem_readb(offset)))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.vga.mem_writeb(offset, value as u32);
        Ok(())
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::exact(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvm_ui::console::DisplayState;
    use ruvm_ui::pixman::{R5G6B5, X8R8G8B8};

    fn vga(mb: u64) -> Arc<VgaCommon> {
        let ram = Arc::new(RamBlock::new("vga.vram", mb << 20, 12).unwrap());
        VgaCommon::new(ram, false)
    }

    fn console(v: &Arc<VgaCommon>, ds: &Arc<DisplayState>) -> QemuConsole {
        let con = ds.graphic_console_create(None, 0, Arc::clone(v) as Arc<dyn GraphicHwOps>);
        v.set_console(con.clone());
        con
    }

    fn vbe(v: &VgaCommon, index: usize, val: u32) {
        v.vbe_write_index(index as u32);
        v.vbe_write_data(val);
    }

    #[test]
    fn tables_match_qemu() {
        assert_eq!(EXPAND4[0xff], 0x1111_1111);
        assert_eq!(EXPAND4[0x81], 0x1000_0001);
        assert_eq!(EXPAND2[0xff], 0x3333);
        assert_eq!(EXPAND2[0x1b], 0x0123);
        assert_eq!(EXPAND4TO8[0x5], 0x33);
        assert_eq!(EXPAND4TO8[0xf], 0xff);
        assert_eq!(c6_to_8(0x3f), 0xff);
        assert_eq!(c6_to_8(0x20), 0x80);
        assert_eq!(vga_vram_size_mb(0), 1);
        assert_eq!(vga_vram_size_mb(12), 16);
        assert_eq!(vga_vram_size_mb(4096), 512);
    }

    #[test]
    fn port_regions_follow_portio_list_add() {
        let v = vga(16);
        let x86: Vec<(u16, u64)> = v.portio_regions(true).iter().map(|r| (r.0, r.1)).collect();
        assert_eq!(x86, [(0x3b4, 2), (0x3ba, 1), (0x3c0, 16), (0x3d4, 2), (0x3da, 1), (0x1ce, 4)]);
        let other: Vec<(u16, u64)> = v.portio_regions(false).iter().map(|r| (r.0, r.1)).collect();
        assert_eq!(other.last(), Some(&(0x1ce, 4)));
    }

    #[test]
    fn port_access_sizes() {
        let v = vga(16);
        let regions = v.portio_regions(true);
        let cx = AccessCtx::default();
        let (_, _, seq) = &regions[2];
        // A word write to 0x3c4 sets the index then the data.
        seq.write(&cx, 4, AccessSize::B2, 0x0f02).unwrap();
        assert_eq!(v.ioport_read(0x3c5), 0x0f);
        assert_eq!(seq.read(&cx, 4, AccessSize::B2).unwrap(), 0x0f02);
        assert_eq!(seq.read(&cx, 0, AccessSize::B4).unwrap(), 0xffff_ffff);
        // The last port of a run reads 0xff in the high byte.
        assert_eq!(seq.read(&cx, 15, AccessSize::B2).unwrap() & 0xff00, 0xff00);
        let (_, _, vbe_ports) = &regions[5];
        vbe_ports.write(&cx, 0, AccessSize::B2, VBE_DISPI_INDEX_ID as u64).unwrap();
        assert_eq!(vbe_ports.read(&cx, 1, AccessSize::B2).unwrap(), u64::from(VBE_DISPI_ID5));
        assert_eq!(vbe_ports.read(&cx, 1, AccessSize::B1).unwrap(), 0xff);
        vbe_ports.write(&cx, 0, AccessSize::B2, 0xa).unwrap();
        assert_eq!(vbe_ports.read(&cx, 2, AccessSize::B2).unwrap(), 256);
    }

    #[test]
    fn vbe_fixup_and_vga_regs() {
        let v = vga(16);
        vbe(&v, VBE_DISPI_INDEX_XRES, 1027);
        vbe(&v, VBE_DISPI_INDEX_YRES, 768);
        vbe(&v, VBE_DISPI_INDEX_BPP, 13);
        vbe(&v, VBE_DISPI_INDEX_ENABLE, u32::from(VBE_DISPI_ENABLED | VBE_DISPI_LFB_ENABLED));
        v.vbe_write_index(VBE_DISPI_INDEX_BPP as u32);
        assert_eq!(v.vbe_read_data(), 8);
        v.vbe_write_index(VBE_DISPI_INDEX_XRES as u32);
        assert_eq!(v.vbe_read_data(), 1024);
        v.vbe_write_index(VBE_DISPI_INDEX_VIRT_HEIGHT as u32);
        assert_eq!(v.vbe_read_data(), 16384);
        let s = v.lock();
        assert_eq!(s.vbe_line_offset, 1024);
        assert_eq!(s.cr[VGA_CRTC_H_DISP], 127);
        assert_eq!(s.gr[VGA_GFX_MISC] & 0x0d, 0x05);
        assert_eq!(s.gr[VGA_GFX_MODE] >> 5, 2);
        drop(s);
        vbe(&v, VBE_DISPI_INDEX_ENABLE, u32::from(VBE_DISPI_GETCAPS));
        v.vbe_write_index(VBE_DISPI_INDEX_XRES as u32);
        assert_eq!(v.vbe_read_data(), 16000);
        vbe(&v, VBE_DISPI_INDEX_BANK, 0x1234);
        v.vbe_write_index(VBE_DISPI_INDEX_BANK as u32);
        assert_eq!(v.vbe_read_data(), 0x34);
    }

    #[test]
    fn planar_write_modes() {
        let v = vga(1);
        // Write all planes, mode 0 with no rotation: one byte lands in all four planes.
        v.ioport_write(0x3c4, 2);
        v.ioport_write(0x3c5, 0x0f);
        v.ioport_write(0x3c4, 4);
        v.ioport_write(0x3c5, 0x06);
        v.ioport_write(0x3ce, 8);
        v.ioport_write(0x3cf, 0xff);
        v.mem_writeb(0x10, 0xa5);
        assert_eq!(v.load_dword(0x10), 0xa5a5_a5a5);
        // Read mode 0 from plane 2 loads the latch.
        v.ioport_write(0x3ce, 4);
        v.ioport_write(0x3cf, 2);
        assert_eq!(v.mem_readb(0x10), 0xa5);
        // Write mode 1 copies the latch.
        v.ioport_write(0x3ce, 5);
        v.ioport_write(0x3cf, 1);
        v.mem_writeb(0x20, 0);
        assert_eq!(v.load_dword(0x20), 0xa5a5_a5a5);
        // Write mode 2 with only plane 1 enabled.
        v.ioport_write(0x3cf, 2);
        v.ioport_write(0x3c4, 2);
        v.ioport_write(0x3c5, 0x02);
        v.mem_writeb(0x30, 0x0f);
        assert_eq!(v.load_dword(0x30), 0x0000_ff00);
    }

    #[test]
    fn dac_round_trip() {
        let v = vga(1);
        v.ioport_write(0x3c8, 5);
        for c in [1, 2, 3] {
            v.ioport_write(0x3c9, c);
        }
        v.ioport_write(0x3c7, 5);
        let got: Vec<u32> = (0..3).map(|_| v.ioport_read(0x3c9)).collect();
        assert_eq!(got, [1, 2, 3]);
        assert_eq!(v.ioport_read(0x3c8), 6);
        // In mono mode the colour ports are dead.
        assert_eq!(v.ioport_read(0x3d4), 0xff);
    }

    /// Mode 3 the way SeaBIOS sets it up, cut down to what the renderer reads.
    fn text_mode(v: &VgaCommon) {
        v.ioport_write(0x3c2, 0x67);
        let seq = [(1u32, 0x00u32), (2, 0x03), (3, 0x00), (4, 0x02)];
        for (i, x) in seq {
            v.ioport_write(0x3c4, i);
            v.ioport_write(0x3c5, x);
        }
        let crtc = [
            (1u32, 0x4fu32),
            (6, 0xbf),
            (7, 0x1f),
            (9, 0x4f),
            (0x0a, 0x20),
            (0x12, 0x8f),
            (0x13, 0x28),
            (0x17, 0xa3),
            (0x18, 0xff),
        ];
        for (i, x) in crtc {
            v.ioport_write(0x3d4, i);
            v.ioport_write(0x3d5, x);
        }
        let gfx = [(5u32, 0x10u32), (6, 0x0e)];
        for (i, x) in gfx {
            v.ioport_write(0x3ce, i);
            v.ioport_write(0x3cf, x);
        }
        let _ = v.ioport_read(0x3da);
        for i in 0..16u32 {
            v.ioport_write(0x3c0, i);
            v.ioport_write(0x3c0, i);
        }
        v.ioport_write(0x3c0, 0x10);
        v.ioport_write(0x3c0, 0x0c);
        v.ioport_write(0x3c0, 0x20);
        v.ioport_write(0x3c8, 0);
        for i in 0..16u32 {
            let c = if i == 7 {
                0x2a
            } else if i == 0 {
                0
            } else {
                0x3f
            };
            for _ in 0..3 {
                v.ioport_write(0x3c9, c);
            }
        }
    }

    #[test]
    fn text_mode_draws_glyphs_from_plane_2() {
        let v = vga(16);
        let ds = DisplayState::new();
        let con = console(&v, &ds);
        text_mode(&v);
        // A solid 'A' glyph in font 0 and an 'A' at the top left, grey on black. DAC 0x2a widens to
        // 0xa8 as in QEMU.
        for row in 0..16usize {
            v.bytes()[2 + 32 * 4 * 0x41 + row * 4].store(0xff, Ordering::Relaxed);
        }
        v.bytes()[0].store(0x41, Ordering::Relaxed);
        v.bytes()[1].store(0x07, Ordering::Relaxed);
        con.hw_update();
        con.with_surface(|s| {
            let s = s.unwrap();
            assert_eq!((s.width(), s.height()), (720, 400));
            assert_eq!(s.image().pixel(0, 0), 0x00a8_a8a8);
            assert_eq!(s.image().pixel(7, 15), 0x00a8_a8a8);
            // The ninth column is background outside the line drawing range.
            assert_eq!(s.image().pixel(8, 0), 0);
            assert_eq!(s.image().pixel(9, 0), 0);
        });
    }

    #[test]
    fn vbe_32bpp_uses_a_shared_surface() {
        let v = vga(16);
        let ds = DisplayState::new();
        let con = console(&v, &ds);
        vbe(&v, VBE_DISPI_INDEX_XRES, 64);
        vbe(&v, VBE_DISPI_INDEX_YRES, 32);
        vbe(&v, VBE_DISPI_INDEX_BPP, 32);
        vbe(&v, VBE_DISPI_INDEX_ENABLE, u32::from(VBE_DISPI_ENABLED | VBE_DISPI_LFB_ENABLED));
        let _ = v.ioport_read(0x3da);
        v.ioport_write(0x3c0, 0x20);
        v.vram().write(4 * (64 * 3 + 5), &[0x11, 0x22, 0x33, 0]).unwrap();
        con.hw_update();
        con.with_surface(|s| {
            let s = s.unwrap();
            assert!(!s.is_allocated());
            assert_eq!(s.format(), X8R8G8B8);
            assert_eq!(s.image().pixel(5, 3), 0x0033_2211);
        });
        // 16 bpp is shared as r5g6b5.
        vbe(&v, VBE_DISPI_INDEX_BPP, 16);
        con.hw_update();
        con.with_surface(|s| assert_eq!(s.unwrap().format(), R5G6B5));
        // 8 bpp goes through the palette into a surface of the console's own.
        vbe(&v, VBE_DISPI_INDEX_BPP, 8);
        v.ioport_write(0x3c8, 1);
        for c in [0x3f, 0, 0] {
            v.ioport_write(0x3c9, c);
        }
        v.vram().write(64 * 2 + 3, &[1]).unwrap();
        con.hw_update();
        con.with_surface(|s| {
            let s = s.unwrap();
            assert!(s.is_allocated());
            assert_eq!(s.image().pixel(3, 2), 0x00ff_0000);
        });
    }

    #[test]
    fn blank_when_the_palette_is_off() {
        let v = vga(16);
        let ds = DisplayState::new();
        let con = console(&v, &ds);
        text_mode(&v);
        con.hw_update();
        let _ = v.ioport_read(0x3da);
        v.ioport_write(0x3c0, 0x00);
        v.bytes()[0].store(0xdb, Ordering::Relaxed);
        con.hw_update();
        con.with_surface(|s| {
            let s = s.unwrap();
            assert!(s.data().iter().all(|&b| b == 0));
        });
    }
}
