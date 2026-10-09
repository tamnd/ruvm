// SPDX-License-Identifier: GPL-2.0-or-later

//! `DisplaySurface`, from QEMU's `ui/display-surface.c` and `include/ui/surface.h`.
//!
//! A surface is what a graphic device scans out and what the UIs and `screendump` read: a pixman
//! image plus two flags. QEMU can build a surface over guest memory without a copy, and devices
//! such as the standard VGA, bochs-display and ramfb do. Here a surface always owns its pixels,
//! and those devices copy the guest framebuffer into it on every display update instead. The
//! pixels a reader sees after an update are the same.

use crate::pixman::{A8, Color, Image, PixelFormat, X8R8G8B8};
use crate::vgafont::{FONT_HEIGHT, FONT_WIDTH, VGAFONT16};

/// `QEMU_ALLOCATED_FLAG`: the console allocated the pixels, as opposed to wrapping a device's.
pub const QEMU_ALLOCATED_FLAG: u32 = 0x01;
/// `QEMU_PLACEHOLDER_FLAG`: a "nothing to show" surface with a message on it.
pub const QEMU_PLACEHOLDER_FLAG: u32 = 0x02;

/// `DisplaySurface`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisplaySurface {
    image: Image,
    flags: u32,
}

impl DisplaySurface {
    /// `qemu_create_displaysurface()`: a zeroed `x8r8g8b8` surface with the stride `width * 4`.
    pub fn new(width: usize, height: usize) -> DisplaySurface {
        DisplaySurface {
            image: Image::new(X8R8G8B8, width, height, width * 4),
            flags: QEMU_ALLOCATED_FLAG,
        }
    }

    /// `qemu_create_displaysurface_from()` with device memory: a surface in `format` with the
    /// given stride. QEMU wraps the device's buffer; this one starts zeroed and the device fills
    /// it, so it is not marked allocated, which is what `qemu_console_resize()` looks at.
    pub fn new_from(
        width: usize,
        height: usize,
        format: PixelFormat,
        stride: usize,
    ) -> DisplaySurface {
        DisplaySurface { image: Image::new(format, width, height, stride), flags: 0 }
    }

    /// `qemu_create_displaysurface_pixman()`: a surface over an existing image.
    pub fn from_image(image: Image) -> DisplaySurface {
        DisplaySurface { image, flags: 0 }
    }

    /// `qemu_create_placeholder_surface()`: a black `w` by `h` surface with `msg` in gray in the
    /// middle, in the 8x16 VGA font.
    pub fn placeholder(w: usize, h: usize, msg: &str) -> DisplaySurface {
        let mut surface = DisplaySurface::new(w, h);
        let len = msg.len() as i64;
        let x = (w as i64 / FONT_WIDTH as i64 - len) / 2;
        let y = (h as i64 / FONT_HEIGHT as i64 - 1) / 2;
        for (i, ch) in msg.bytes().enumerate() {
            let glyph = glyph_from_vgafont(ch);
            glyph_render(&mut surface.image, &glyph, Color::GRAY, Color::BLACK, x + i as i64, y);
        }
        surface.flags |= QEMU_PLACEHOLDER_FLAG;
        surface
    }

    pub fn image(&self) -> &Image {
        &self.image
    }

    pub fn image_mut(&mut self) -> &mut Image {
        &mut self.image
    }

    pub fn flags(&self) -> u32 {
        self.flags
    }

    /// `surface_width()`.
    pub fn width(&self) -> usize {
        self.image.width()
    }

    /// `surface_height()`.
    pub fn height(&self) -> usize {
        self.image.height()
    }

    /// `surface_stride()`.
    pub fn stride(&self) -> usize {
        self.image.stride()
    }

    /// `surface_format()`.
    pub fn format(&self) -> PixelFormat {
        self.image.format()
    }

    /// `surface_bits_per_pixel()`.
    pub fn bits_per_pixel(&self) -> u32 {
        self.image.format().bpp()
    }

    /// `surface_bytes_per_pixel()`.
    pub fn bytes_per_pixel(&self) -> usize {
        self.image.format().bpp().div_ceil(8) as usize
    }

    /// `surface_data()`.
    pub fn data(&self) -> &[u8] {
        self.image.data()
    }

    pub fn data_mut(&mut self) -> &mut [u8] {
        self.image.data_mut()
    }

    /// `surface_is_allocated()`.
    pub fn is_allocated(&self) -> bool {
        self.flags & QEMU_ALLOCATED_FLAG != 0
    }

    /// `surface_is_placeholder()`.
    pub fn is_placeholder(&self) -> bool {
        self.flags & QEMU_PLACEHOLDER_FLAG != 0
    }
}

/// `qemu_pixman_glyph_from_vgafont()`: the `a8` mask of one character, 0xff where the font has a
/// bit set.
pub fn glyph_from_vgafont(ch: u8) -> Image {
    let mut glyph = Image::new(A8, FONT_WIDTH, FONT_HEIGHT, 0);
    let font = &VGAFONT16[FONT_HEIGHT * usize::from(ch)..][..FONT_HEIGHT];
    for (y, &bits) in font.iter().enumerate() {
        for x in 0..FONT_WIDTH {
            let on = bits & (1 << (7 - x)) != 0;
            glyph.set_pixel(x, y, if on { 0xff } else { 0 });
        }
    }
    glyph
}

/// `qemu_pixman_glyph_render()`: fills character cell (`x`, `y`) with `bg` and paints `fg`
/// through the glyph mask.
pub fn glyph_render(surface: &mut Image, glyph: &Image, fg: Color, bg: Color, x: i64, y: i64) {
    let cw = FONT_WIDTH as i64;
    let ch = FONT_HEIGHT as i64;
    surface.fill(bg, cw * x, ch * y, cw, ch);
    surface.over_mask(fg, glyph, cw * x, ch * y);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholder_centers_the_message() {
        let msg = "Guest has not initialized the display (yet).";
        let s = DisplaySurface::placeholder(640, 480, msg);
        assert!(s.is_placeholder() && s.is_allocated());
        assert_eq!(s.stride(), 2560);
        // 44 characters in 80 columns start at column 18, row (30 - 1) / 2 = 14.
        let (x0, y0) = (18 * 8, 14 * 16);
        let mut lit = 0;
        for y in 0..480 {
            for x in 0..640 {
                let p = s.image().pixel(x, y);
                if p != 0 {
                    assert_eq!(p, 0x00aa_aaaa);
                    assert!((x0..x0 + 44 * 8).contains(&x) && (y0..y0 + 16).contains(&y));
                    lit += 1;
                }
            }
        }
        assert!(lit > 0);
    }

    #[test]
    fn a_message_wider_than_the_surface_is_clipped() {
        let s = DisplaySurface::placeholder(64, 32, "Display output is not active.");
        assert_eq!(s.width(), 64);
    }
}
