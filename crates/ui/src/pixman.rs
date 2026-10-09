// SPDX-License-Identifier: GPL-2.0-or-later

//! The small part of pixman that QEMU's display code leans on: format codes, images and the
//! conversions between them.
//!
//! QEMU links the real pixman and describes every surface with a `pixman_format_code_t`. The
//! codes here are bit for bit the same, so a format that QEMU logs or migrates means the same
//! thing on both sides, and `qemu_pixelformat_from_pixman()` and `qemu_default_pixman_format()`
//! from `ui/qemu-pixman.c` are ported on top of them.
//!
//! Only the operations the console core needs are here: a `PIXMAN_OP_SRC` copy between any two
//! of the RGB formats, a solid fill, and the `PIXMAN_OP_OVER` of a solid colour through an `a8`
//! mask that draws a glyph. Pixman widens a narrow channel by repeating its bits (a 5 bit `0x1f`
//! becomes `0xff`, not `0xf8`) and narrows one by dropping the low bits, and the conversions here
//! do the same, so a screendump of a 16 bpp surface comes out byte for byte the same as QEMU's.

/// `PIXMAN_TYPE_OTHER`.
pub const TYPE_OTHER: u32 = 0;
/// `PIXMAN_TYPE_A`.
pub const TYPE_A: u32 = 1;
/// `PIXMAN_TYPE_ARGB`.
pub const TYPE_ARGB: u32 = 2;
/// `PIXMAN_TYPE_ABGR`.
pub const TYPE_ABGR: u32 = 3;
/// `PIXMAN_TYPE_BGRA`.
pub const TYPE_BGRA: u32 = 8;
/// `PIXMAN_TYPE_RGBA`.
pub const TYPE_RGBA: u32 = 9;

/// A `pixman_format_code_t`: bits per pixel, type and the four channel widths packed into a word.
///
/// The pixel value is read in host byte order, as pixman does, so `x8r8g8b8` is the bytes
/// `b, g, r, x` on a little endian host. [`PixelFormat::be_r8g8b8`] and friends are the
/// `PIXMAN_BE_*` and `PIXMAN_LE_*` aliases, which name a byte order instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PixelFormat(pub u32);

/// `PIXMAN_FORMAT(bpp, type, a, r, g, b)`.
pub const fn format(bpp: u32, ty: u32, a: u32, r: u32, g: u32, b: u32) -> PixelFormat {
    PixelFormat((bpp << 24) | (ty << 16) | (a << 12) | (r << 8) | (g << 4) | b)
}

/// `PIXMAN_a8r8g8b8`.
pub const A8R8G8B8: PixelFormat = format(32, TYPE_ARGB, 8, 8, 8, 8);
/// `PIXMAN_x8r8g8b8`.
pub const X8R8G8B8: PixelFormat = format(32, TYPE_ARGB, 0, 8, 8, 8);
/// `PIXMAN_a8b8g8r8`.
pub const A8B8G8R8: PixelFormat = format(32, TYPE_ABGR, 8, 8, 8, 8);
/// `PIXMAN_x8b8g8r8`.
pub const X8B8G8R8: PixelFormat = format(32, TYPE_ABGR, 0, 8, 8, 8);
/// `PIXMAN_b8g8r8a8`.
pub const B8G8R8A8: PixelFormat = format(32, TYPE_BGRA, 8, 8, 8, 8);
/// `PIXMAN_b8g8r8x8`.
pub const B8G8R8X8: PixelFormat = format(32, TYPE_BGRA, 0, 8, 8, 8);
/// `PIXMAN_r8g8b8a8`.
pub const R8G8B8A8: PixelFormat = format(32, TYPE_RGBA, 8, 8, 8, 8);
/// `PIXMAN_r8g8b8x8`.
pub const R8G8B8X8: PixelFormat = format(32, TYPE_RGBA, 0, 8, 8, 8);
/// `PIXMAN_r8g8b8`.
pub const R8G8B8: PixelFormat = format(24, TYPE_ARGB, 0, 8, 8, 8);
/// `PIXMAN_b8g8r8`.
pub const B8G8R8: PixelFormat = format(24, TYPE_ABGR, 0, 8, 8, 8);
/// `PIXMAN_r5g6b5`.
pub const R5G6B5: PixelFormat = format(16, TYPE_ARGB, 0, 5, 6, 5);
/// `PIXMAN_b5g6r5`.
pub const B5G6R5: PixelFormat = format(16, TYPE_ABGR, 0, 5, 6, 5);
/// `PIXMAN_x1r5g5b5`.
pub const X1R5G5B5: PixelFormat = format(16, TYPE_ARGB, 0, 5, 5, 5);
/// `PIXMAN_a1r5g5b5`.
pub const A1R5G5B5: PixelFormat = format(16, TYPE_ARGB, 1, 5, 5, 5);
/// `PIXMAN_a8`.
pub const A8: PixelFormat = format(8, TYPE_A, 8, 0, 0, 0);

const LITTLE: bool = cfg!(target_endian = "little");

impl PixelFormat {
    /// `PIXMAN_FORMAT_BPP()`.
    pub const fn bpp(self) -> u32 {
        self.0 >> 24
    }

    /// `PIXMAN_FORMAT_TYPE()`.
    pub const fn ty(self) -> u32 {
        (self.0 >> 16) & 0xff
    }

    /// `PIXMAN_FORMAT_A()`.
    pub const fn a(self) -> u32 {
        (self.0 >> 12) & 0xf
    }

    /// `PIXMAN_FORMAT_R()`.
    pub const fn r(self) -> u32 {
        (self.0 >> 8) & 0xf
    }

    /// `PIXMAN_FORMAT_G()`.
    pub const fn g(self) -> u32 {
        (self.0 >> 4) & 0xf
    }

    /// `PIXMAN_FORMAT_B()`.
    pub const fn b(self) -> u32 {
        self.0 & 0xf
    }

    /// `PIXMAN_FORMAT_DEPTH()`.
    pub const fn depth(self) -> u32 {
        self.a() + self.r() + self.g() + self.b()
    }

    /// Bytes per pixel, rounded up.
    pub const fn bytes_per_pixel(self) -> usize {
        self.bpp().div_ceil(8) as usize
    }

    /// `PIXMAN_LE_x8r8g8b8`: the bytes `b, g, r, x` in memory.
    pub const fn le_x8r8g8b8() -> PixelFormat {
        if LITTLE { X8R8G8B8 } else { B8G8R8X8 }
    }

    /// `PIXMAN_BE_x8r8g8b8`: the bytes `x, r, g, b` in memory.
    pub const fn be_x8r8g8b8() -> PixelFormat {
        if LITTLE { B8G8R8X8 } else { X8R8G8B8 }
    }

    /// `PIXMAN_LE_a8r8g8b8`.
    pub const fn le_a8r8g8b8() -> PixelFormat {
        if LITTLE { A8R8G8B8 } else { B8G8R8A8 }
    }

    /// `PIXMAN_LE_x8b8g8r8`.
    pub const fn le_x8b8g8r8() -> PixelFormat {
        if LITTLE { X8B8G8R8 } else { R8G8B8X8 }
    }

    /// `PIXMAN_LE_a8b8g8r8`.
    pub const fn le_a8b8g8r8() -> PixelFormat {
        if LITTLE { A8B8G8R8 } else { R8G8B8A8 }
    }

    /// `PIXMAN_LE_r8g8b8`: the bytes `b, g, r` in memory.
    pub const fn le_r8g8b8() -> PixelFormat {
        if LITTLE { R8G8B8 } else { B8G8R8 }
    }

    /// `PIXMAN_BE_r8g8b8`: the bytes `r, g, b` in memory, which is what a ppm or png row holds.
    pub const fn be_r8g8b8() -> PixelFormat {
        if LITTLE { B8G8R8 } else { R8G8B8 }
    }

    /// The shifts of the four channels inside the pixel value, `qemu_pixelformat_from_pixman()`.
    /// None for a type that is not one of the four RGB orders.
    fn shifts(self) -> Option<Shifts> {
        let (r, g, b, bpp) = (self.r(), self.g(), self.b(), self.bpp());
        let s = match self.ty() {
            TYPE_ARGB => Shifts { a: b + g + r, r: b + g, g: b, b: 0 },
            TYPE_ABGR => Shifts { a: r + g + b, b: r + g, g: r, r: 0 },
            TYPE_BGRA => Shifts { b: bpp - b, g: bpp - (b + g), r: bpp - (b + g + r), a: 0 },
            TYPE_RGBA => Shifts { r: bpp - r, g: bpp - (r + g), b: bpp - (r + g + b), a: 0 },
            TYPE_A => Shifts { a: 0, r: 0, g: 0, b: 0 },
            _ => return None,
        };
        Some(s)
    }

    /// True for the formats [`Image::composite_src`] can read and write.
    pub fn is_supported(self) -> bool {
        matches!(self.bpp(), 8 | 16 | 24 | 32) && self.shifts().is_some()
    }

    /// `qemu_pixelformat_from_pixman()`.
    pub fn to_pixel_format_info(self) -> Option<PixelFormatInfo> {
        let s = self.shifts()?;
        let max = |bits: u32| ((1u32 << bits) - 1) as u8;
        Some(PixelFormatInfo {
            bits_per_pixel: self.bpp() as u8,
            bytes_per_pixel: (self.bpp() / 8) as u8,
            depth: self.depth() as u8,
            abits: self.a() as u8,
            rbits: self.r() as u8,
            gbits: self.g() as u8,
            bbits: self.b() as u8,
            ashift: s.a as u8,
            rshift: s.r as u8,
            gshift: s.g as u8,
            bshift: s.b as u8,
            amax: max(self.a()),
            rmax: max(self.r()),
            gmax: max(self.g()),
            bmax: max(self.b()),
            amask: u32::from(max(self.a())) << s.a,
            rmask: u32::from(max(self.r())) << s.r,
            gmask: u32::from(max(self.g())) << s.g,
            bmask: u32::from(max(self.b())) << s.b,
        })
    }
}

#[derive(Clone, Copy)]
struct Shifts {
    a: u32,
    r: u32,
    g: u32,
    b: u32,
}

/// QEMU's `PixelFormat` from `include/ui/console.h`, the channel layout spelled out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelFormatInfo {
    pub bits_per_pixel: u8,
    pub bytes_per_pixel: u8,
    pub depth: u8,
    pub rmask: u32,
    pub gmask: u32,
    pub bmask: u32,
    pub amask: u32,
    pub rshift: u8,
    pub gshift: u8,
    pub bshift: u8,
    pub ashift: u8,
    pub rmax: u8,
    pub gmax: u8,
    pub bmax: u8,
    pub amax: u8,
    pub rbits: u8,
    pub gbits: u8,
    pub bbits: u8,
    pub abits: u8,
}

/// `qemu_default_pixman_format()`: the format a device with `bpp` bits per pixel and the given
/// byte order hands to the console, or None where QEMU returns 0.
pub fn default_pixman_format(bpp: u32, native_endian: bool) -> Option<PixelFormat> {
    if native_endian {
        match bpp {
            15 => Some(X1R5G5B5),
            16 => Some(R5G6B5),
            24 => Some(R8G8B8),
            32 => Some(X8R8G8B8),
            _ => None,
        }
    } else {
        match bpp {
            24 => Some(B8G8R8),
            32 => Some(B8G8R8X8),
            _ => None,
        }
    }
}

/// `DRM_FORMAT_RGB888`.
pub const DRM_FORMAT_RGB888: u32 = fourcc(b"RG24");
/// `DRM_FORMAT_XRGB8888`.
pub const DRM_FORMAT_XRGB8888: u32 = fourcc(b"XR24");
/// `DRM_FORMAT_ARGB8888`.
pub const DRM_FORMAT_ARGB8888: u32 = fourcc(b"AR24");
/// `DRM_FORMAT_XBGR8888`.
pub const DRM_FORMAT_XBGR8888: u32 = fourcc(b"XB24");
/// `DRM_FORMAT_ABGR8888`.
pub const DRM_FORMAT_ABGR8888: u32 = fourcc(b"AB24");

/// `fourcc_code()`.
pub const fn fourcc(c: &[u8; 4]) -> u32 {
    (c[0] as u32) | ((c[1] as u32) << 8) | ((c[2] as u32) << 16) | ((c[3] as u32) << 24)
}

/// `qemu_drm_format_to_pixman()`. DRM formats are little endian whatever the host is.
pub fn drm_format_to_pixman(drm: u32) -> Option<PixelFormat> {
    match drm {
        DRM_FORMAT_RGB888 => Some(PixelFormat::le_r8g8b8()),
        DRM_FORMAT_ARGB8888 => Some(PixelFormat::le_a8r8g8b8()),
        DRM_FORMAT_XRGB8888 => Some(PixelFormat::le_x8r8g8b8()),
        DRM_FORMAT_XBGR8888 => Some(PixelFormat::le_x8b8g8r8()),
        DRM_FORMAT_ABGR8888 => Some(PixelFormat::le_a8b8g8r8()),
        _ => None,
    }
}

/// `qemu_pixman_to_drm_format()`.
pub fn pixman_to_drm_format(format: PixelFormat) -> Option<u32> {
    [
        DRM_FORMAT_RGB888,
        DRM_FORMAT_ARGB8888,
        DRM_FORMAT_XRGB8888,
        DRM_FORMAT_XBGR8888,
        DRM_FORMAT_ABGR8888,
    ]
    .into_iter()
    .find(|&drm| drm_format_to_pixman(drm) == Some(format))
}

/// `qemu_pixman_check_format()`: the conversions pixman is known to do well, which a UI that
/// converts with pixman accepts in `dpy_gfx_check_format`.
pub fn pixman_check_format(format: PixelFormat) -> bool {
    matches!(
        format,
        X8R8G8B8 | A8R8G8B8 | B8G8R8X8 | B8G8R8A8 | R8G8B8 | B8G8R8 | X1R5G5B5 | R5G6B5
    )
}

/// A `pixman_color_t`: four 16 bit channels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Color {
    pub red: u16,
    pub green: u16,
    pub blue: u16,
    pub alpha: u16,
}

impl Color {
    /// `QEMU_PIXMAN_COLOR(r, g, b)`: an opaque colour from 8 bit channels.
    pub const fn rgb(r: u8, g: u8, b: u8) -> Color {
        Color { red: (r as u16) << 8, green: (g as u16) << 8, blue: (b as u16) << 8, alpha: 0xffff }
    }

    /// `QEMU_PIXMAN_COLOR_BLACK`.
    pub const BLACK: Color = Color::rgb(0, 0, 0);
    /// `QEMU_PIXMAN_COLOR_GRAY`.
    pub const GRAY: Color = Color::rgb(0xaa, 0xaa, 0xaa);

    fn argb(self) -> Argb {
        Argb {
            a: (self.alpha >> 8) as u8,
            r: (self.red >> 8) as u8,
            g: (self.green >> 8) as u8,
            b: (self.blue >> 8) as u8,
        }
    }
}

/// One pixel widened to 8 bits a channel, what pixman's general path converts through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Argb {
    pub a: u8,
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

/// Widens a `bits` wide channel to 8 bits by repeating its bits, as pixman's `unorm_to_unorm()`.
fn widen(v: u32, bits: u32) -> u8 {
    match bits {
        0 => 0,
        8 => v as u8,
        _ => {
            let mut out = 0u32;
            let mut have = 0;
            while have < 8 {
                out = (out << bits) | v;
                have += bits;
            }
            (out >> (have - 8)) as u8
        }
    }
}

/// Narrows an 8 bit channel to `bits` by dropping the low bits.
fn narrow(v: u8, bits: u32) -> u32 {
    if bits == 0 { 0 } else { u32::from(v) >> (8 - bits) }
}

/// A `pixman_image_t` made of bits: a format, a size, a stride in bytes and the pixels.
///
/// QEMU can wrap guest memory in an image without copying it. An image here always owns its
/// buffer, and a device that scans out of guest memory copies the rows it is told are dirty.
#[derive(Clone, PartialEq, Eq)]
pub struct Image {
    format: PixelFormat,
    width: usize,
    height: usize,
    stride: usize,
    data: Vec<u8>,
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Image")
            .field("format", &self.format)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("stride", &self.stride)
            .finish_non_exhaustive()
    }
}

impl Image {
    /// `pixman_image_create_bits(format, width, height, NULL, stride)`: a zeroed image. A stride
    /// of 0 picks pixman's default, the row rounded up to a whole number of 32 bit words.
    pub fn new(format: PixelFormat, width: usize, height: usize, stride: usize) -> Image {
        let stride = if stride == 0 { default_stride(format, width) } else { stride };
        Image { format, width, height, stride, data: vec![0; stride * height] }
    }

    /// An image over `data`, which must hold at least `stride * (height - 1)` plus one row.
    pub fn from_data(
        format: PixelFormat,
        width: usize,
        height: usize,
        stride: usize,
        mut data: Vec<u8>,
    ) -> Image {
        let need = stride * height;
        if data.len() < need {
            data.resize(need, 0);
        }
        Image { format, width, height, stride, data }
    }

    pub fn format(&self) -> PixelFormat {
        self.format
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// The row pitch in bytes, `pixman_image_get_stride()`.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The pixels, `pixman_image_get_data()`.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// One row of pixels, without the padding at the end of the stride.
    pub fn row(&self, y: usize) -> &[u8] {
        let start = y * self.stride;
        let len = self.width * self.format.bpp() as usize / 8;
        &self.data[start..start + len]
    }

    pub fn row_mut(&mut self, y: usize) -> &mut [u8] {
        let start = y * self.stride;
        let len = self.width * self.format.bpp() as usize / 8;
        &mut self.data[start..start + len]
    }

    /// The raw pixel value at (x, y), read in host byte order.
    pub fn pixel(&self, x: usize, y: usize) -> u32 {
        read_pixel(&self.data[y * self.stride..], x, self.format.bpp())
    }

    /// Sets the raw pixel value at (x, y).
    pub fn set_pixel(&mut self, x: usize, y: usize, v: u32) {
        let bpp = self.format.bpp();
        write_pixel(&mut self.data[y * self.stride..], x, bpp, v);
    }

    /// The pixel at (x, y) widened to 8 bits a channel, alpha 0xff where the format has none.
    pub fn argb(&self, x: usize, y: usize) -> Argb {
        unpack(self.format, self.pixel(x, y))
    }

    /// `pixman_image_composite(PIXMAN_OP_SRC, src, NULL, self, sx, sy, 0, 0, dx, dy, w, h)`, clipped
    /// to both images.
    #[allow(clippy::too_many_arguments)]
    pub fn composite_src(
        &mut self,
        src: &Image,
        sx: i64,
        sy: i64,
        dx: i64,
        dy: i64,
        w: i64,
        h: i64,
    ) {
        let Some((sx, sy, dx, dy, w, h)) = clip(src, sx, sy, self, dx, dy, w, h) else {
            return;
        };
        let (sbpp, dbpp) = (src.format.bpp(), self.format.bpp());
        let same = src.format == self.format;
        for row in 0..h {
            let srow = &src.data[(sy + row) * src.stride..];
            let start = (dy + row) * self.stride;
            let drow = &mut self.data[start..];
            if same {
                let bytes = dbpp as usize / 8;
                drow[dx * bytes..(dx + w) * bytes]
                    .copy_from_slice(&srow[sx * bytes..(sx + w) * bytes]);
                continue;
            }
            for col in 0..w {
                let p = unpack(src.format, read_pixel(srow, sx + col, sbpp));
                write_pixel(drow, dx + col, dbpp, pack(self.format, p));
            }
        }
    }

    /// `PIXMAN_OP_SRC` of a solid colour over a rectangle, clipped to the image.
    pub fn fill(&mut self, color: Color, x: i64, y: i64, w: i64, h: i64) {
        let Some((x, y, w, h)) = clip_rect(self, x, y, w, h) else {
            return;
        };
        let v = pack_solid(self.format, color.argb());
        let bpp = self.format.bpp();
        for row in y..y + h {
            let drow = &mut self.data[row * self.stride..];
            for col in x..x + w {
                write_pixel(drow, col, bpp, v);
            }
        }
    }

    /// `PIXMAN_OP_OVER` of an opaque solid colour through the `a8` image `mask`, placed at (x, y).
    /// A mask value of 0xff writes the colour, 0 leaves the pixel alone, and anything between
    /// blends the way pixman does.
    pub fn over_mask(&mut self, color: Color, mask: &Image, x: i64, y: i64) {
        let w = mask.width as i64;
        let h = mask.height as i64;
        let Some((mx, my, dx, dy, w, h)) = clip(mask, 0, 0, self, x, y, w, h) else {
            return;
        };
        let fg = color.argb();
        let bpp = self.format.bpp();
        for row in 0..h {
            for col in 0..w {
                let m = mask.data[(my + row) * mask.stride + mx + col];
                if m == 0 {
                    continue;
                }
                let drow = &mut self.data[(dy + row) * self.stride..];
                let out = if m == 0xff && fg.a == 0xff {
                    fg
                } else {
                    let d = unpack(self.format, read_pixel(drow, dx + col, bpp));
                    blend_over(fg, m, d)
                };
                write_pixel(drow, dx + col, bpp, pack_solid(self.format, out));
            }
        }
    }
}

/// Packs a solid colour the way pixman's fills and its `over_n_8_8888` path store it: they
/// treat x8r8g8b8 as a8r8g8b8, so the unused byte gets the alpha. Only a client that reads that
/// byte, such as a 32 bpp VNC client, can tell.
fn pack_solid(format: PixelFormat, p: Argb) -> u32 {
    pack(if format == X8R8G8B8 { A8R8G8B8 } else { format }, p)
}

/// `src IN mask OVER dst` for one pixel, with pixman's rounding division by 255.
fn blend_over(src: Argb, m: u8, dst: Argb) -> Argb {
    let mul = |a: u8, b: u8| -> u8 {
        let t = u32::from(a) * u32::from(b) + 0x80;
        (((t >> 8) + t) >> 8) as u8
    };
    let s = Argb { a: mul(src.a, m), r: mul(src.r, m), g: mul(src.g, m), b: mul(src.b, m) };
    let ia = 255 - s.a;
    Argb {
        a: s.a.saturating_add(mul(dst.a, ia)),
        r: s.r.saturating_add(mul(dst.r, ia)),
        g: s.g.saturating_add(mul(dst.g, ia)),
        b: s.b.saturating_add(mul(dst.b, ia)),
    }
}

/// pixman's default stride: the row rounded up to 32 bits.
pub fn default_stride(format: PixelFormat, width: usize) -> usize {
    (width * format.bpp() as usize).div_ceil(32) * 4
}

#[allow(clippy::too_many_arguments)]
fn clip(
    src: &Image,
    sx: i64,
    sy: i64,
    dst: &Image,
    dx: i64,
    dy: i64,
    w: i64,
    h: i64,
) -> Option<(usize, usize, usize, usize, usize, usize)> {
    let (mut sx, mut sy, mut dx, mut dy, mut w, mut h) = (sx, sy, dx, dy, w, h);
    for (s, d, len) in [(&mut sx, &mut dx, &mut w), (&mut sy, &mut dy, &mut h)] {
        let lo = (-*s).max(-*d).max(0);
        *s += lo;
        *d += lo;
        *len -= lo;
    }
    w = w.min(src.width as i64 - sx).min(dst.width as i64 - dx);
    h = h.min(src.height as i64 - sy).min(dst.height as i64 - dy);
    if w <= 0 || h <= 0 {
        return None;
    }
    Some((sx as usize, sy as usize, dx as usize, dy as usize, w as usize, h as usize))
}

fn clip_rect(img: &Image, x: i64, y: i64, w: i64, h: i64) -> Option<(usize, usize, usize, usize)> {
    let x0 = x.max(0);
    let y0 = y.max(0);
    let x1 = (x + w).min(img.width as i64);
    let y1 = (y + h).min(img.height as i64);
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some((x0 as usize, y0 as usize, (x1 - x0) as usize, (y1 - y0) as usize))
}

fn read_pixel(row: &[u8], x: usize, bpp: u32) -> u32 {
    match bpp {
        32 => {
            let o = x * 4;
            u32::from_ne_bytes([row[o], row[o + 1], row[o + 2], row[o + 3]])
        }
        24 => {
            let o = x * 3;
            if LITTLE {
                u32::from(row[o]) | u32::from(row[o + 1]) << 8 | u32::from(row[o + 2]) << 16
            } else {
                u32::from(row[o]) << 16 | u32::from(row[o + 1]) << 8 | u32::from(row[o + 2])
            }
        }
        16 => {
            let o = x * 2;
            u32::from(u16::from_ne_bytes([row[o], row[o + 1]]))
        }
        8 => u32::from(row[x]),
        _ => 0,
    }
}

fn write_pixel(row: &mut [u8], x: usize, bpp: u32, v: u32) {
    match bpp {
        32 => {
            let o = x * 4;
            row[o..o + 4].copy_from_slice(&v.to_ne_bytes());
        }
        24 => {
            let o = x * 3;
            let b = [v as u8, (v >> 8) as u8, (v >> 16) as u8];
            if LITTLE {
                row[o..o + 3].copy_from_slice(&b);
            } else {
                row[o..o + 3].copy_from_slice(&[b[2], b[1], b[0]]);
            }
        }
        16 => {
            let o = x * 2;
            row[o..o + 2].copy_from_slice(&(v as u16).to_ne_bytes());
        }
        8 => row[x] = v as u8,
        _ => {}
    }
}

/// Reads the channels of a pixel value in `format`.
pub fn unpack(format: PixelFormat, v: u32) -> Argb {
    let Some(s) = format.shifts() else {
        return Argb { a: 0xff, r: 0, g: 0, b: 0 };
    };
    let field = |bits: u32, shift: u32| -> u32 {
        if bits == 0 { 0 } else { (v >> shift) & ((1u32 << bits) - 1) }
    };
    let a = if format.a() == 0 { 0xff } else { widen(field(format.a(), s.a), format.a()) };
    Argb {
        a,
        r: widen(field(format.r(), s.r), format.r()),
        g: widen(field(format.g(), s.g), format.g()),
        b: widen(field(format.b(), s.b), format.b()),
    }
}

/// Packs channels into a pixel value in `format`. The padding bits of an `x` format are zero.
pub fn pack(format: PixelFormat, p: Argb) -> u32 {
    let Some(s) = format.shifts() else {
        return 0;
    };
    (narrow(p.a, format.a()) << s.a)
        | (narrow(p.r, format.r()) << s.r)
        | (narrow(p.g, format.g()) << s.g)
        | (narrow(p.b, format.b()) << s.b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_match_pixman() {
        assert_eq!(X8R8G8B8.0, 0x2002_0888);
        assert_eq!(A8R8G8B8.0, 0x2002_8888);
        assert_eq!(R5G6B5.0, 0x1002_0565);
        assert_eq!(X1R5G5B5.0, 0x1002_0555);
        assert_eq!(R8G8B8.0, 0x1802_0888);
        assert_eq!(B8G8R8.0, 0x1803_0888);
        assert_eq!(B8G8R8X8.0, 0x2008_0888);
        assert_eq!(A8.0, 0x0801_8000);
    }

    #[test]
    fn narrow_channels_widen_by_bit_replication() {
        assert_eq!(widen(0x1f, 5), 0xff);
        assert_eq!(widen(0x10, 5), 0x84);
        assert_eq!(widen(0x3f, 6), 0xff);
        assert_eq!(widen(0x20, 6), 0x82);
        assert_eq!(widen(1, 1), 0xff);
        let p = unpack(R5G6B5, 0xf800);
        assert_eq!(p, Argb { a: 0xff, r: 0xff, g: 0, b: 0 });
    }

    #[test]
    fn src_to_be_r8g8b8_gives_rgb_bytes() {
        let mut fb = Image::new(X8R8G8B8, 2, 1, 0);
        fb.set_pixel(0, 0, 0x0012_3456);
        fb.set_pixel(1, 0, 0xff65_4321);
        let mut line = Image::new(PixelFormat::be_r8g8b8(), 2, 1, 0);
        line.composite_src(&fb, 0, 0, 0, 0, 2, 1);
        assert_eq!(line.stride(), 8);
        assert_eq!(&line.data()[..6], &[0x12, 0x34, 0x56, 0x65, 0x43, 0x21]);
        assert_eq!(&line.data()[6..], &[0, 0]);
    }

    #[test]
    fn default_formats() {
        assert_eq!(default_pixman_format(32, true), Some(X8R8G8B8));
        assert_eq!(default_pixman_format(15, true), Some(X1R5G5B5));
        assert_eq!(default_pixman_format(16, false), None);
        assert_eq!(default_pixman_format(32, false), Some(B8G8R8X8));
    }

    #[test]
    fn pixel_format_info_of_x8r8g8b8() {
        let pf = X8R8G8B8.to_pixel_format_info().unwrap();
        assert_eq!((pf.rshift, pf.gshift, pf.bshift), (16, 8, 0));
        assert_eq!((pf.rmask, pf.gmask, pf.bmask, pf.amask), (0xff0000, 0xff00, 0xff, 0));
        assert_eq!(pf.depth, 24);
    }

    #[test]
    fn drm_formats_round_trip() {
        assert_eq!(drm_format_to_pixman(DRM_FORMAT_XRGB8888), Some(X8R8G8B8));
        assert_eq!(pixman_to_drm_format(X8R8G8B8), Some(DRM_FORMAT_XRGB8888));
        assert_eq!(drm_format_to_pixman(0), None);
    }

    #[test]
    fn over_mask_paints_only_set_bits() {
        let mut img = Image::new(X8R8G8B8, 4, 1, 0);
        let mut mask = Image::new(A8, 2, 1, 0);
        mask.data_mut()[1] = 0xff;
        img.over_mask(Color::GRAY, &mask, 1, 0);
        assert_eq!(img.pixel(1, 0), 0);
        assert_eq!(img.pixel(2, 0), 0xffaa_aaaa);
        img.fill(Color::BLACK, 3, 0, 1, 1);
        assert_eq!(img.pixel(3, 0), 0xff00_0000);
        let mut img = Image::new(R5G6B5, 1, 1, 0);
        img.fill(Color::GRAY, 0, 0, 1, 1);
        assert_eq!(img.pixel(0, 0), 0xad55);
    }
}
