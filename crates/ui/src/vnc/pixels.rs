// SPDX-License-Identifier: GPL-2.0-or-later

//! The client pixel format and the conversion from the server surface, from
//! `set_pixel_format()`, `set_pixel_conversion()`, `vnc_convert_pixel()` and the
//! `write_pixels` functions of QEMU's ui/vnc.c.
//!
//! The server surface is `x8r8g8b8` in host byte order, one `u32` a pixel. A client whose format
//! pixman calls `x8r8g8b8` gets the pixels copied as they are, with the unused byte, and every
//! other client gets them through [`VncPixelFormat::convert`], which keeps the top bits of each
//! channel.

/// The `PixelFormat` QEMU keeps for a client, with only the fields VNC reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VncPixelFormat {
    pub bits_per_pixel: u8,
    pub bytes_per_pixel: usize,
    pub depth: u8,
    pub rmax: u32,
    pub gmax: u32,
    pub bmax: u32,
    pub rbits: u32,
    pub gbits: u32,
    pub bbits: u32,
    pub rshift: u32,
    pub gshift: u32,
    pub bshift: u32,
    /// `client_endian == G_BIG_ENDIAN`.
    pub big_endian: bool,
}

impl VncPixelFormat {
    /// `qemu_default_pixelformat(32)` in host byte order, what `pixel_format_message()` resets a
    /// client to.
    pub const fn server_default() -> VncPixelFormat {
        VncPixelFormat {
            bits_per_pixel: 32,
            bytes_per_pixel: 4,
            depth: 24,
            rmax: 255,
            gmax: 255,
            bmax: 255,
            rbits: 8,
            gbits: 8,
            bbits: 8,
            rshift: 16,
            gshift: 8,
            bshift: 0,
            big_endian: cfg!(target_endian = "big"),
        }
    }

    /// The checks and the arithmetic of `set_pixel_format()`. None is a bad format, which
    /// QEMU answers by dropping the client.
    #[allow(clippy::too_many_arguments)]
    pub fn from_client(
        mut bits_per_pixel: u8,
        big_endian_flag: u8,
        true_color_flag: u8,
        mut red_max: u16,
        mut green_max: u16,
        mut blue_max: u16,
        mut red_shift: u8,
        mut green_shift: u8,
        mut blue_shift: u8,
    ) -> Option<VncPixelFormat> {
        if true_color_flag == 0 {
            // Expose a reasonable default 256 color map.
            bits_per_pixel = 8;
            red_max = 7;
            green_max = 7;
            blue_max = 3;
            red_shift = 0;
            green_shift = 3;
            blue_shift = 6;
        }
        if !matches!(bits_per_pixel, 8 | 16 | 32) {
            return None;
        }
        if red_max > 255 || green_max > 255 || blue_max > 255 {
            return None;
        }
        if red_shift >= bits_per_pixel
            || green_shift >= bits_per_pixel
            || blue_shift >= bits_per_pixel
        {
            return None;
        }
        let max = |m: u16| if m != 0 { u32::from(m) } else { 0xff };
        Some(VncPixelFormat {
            bits_per_pixel,
            bytes_per_pixel: usize::from(bits_per_pixel / 8),
            depth: if bits_per_pixel == 32 { 24 } else { bits_per_pixel },
            rmax: max(red_max),
            gmax: max(green_max),
            bmax: max(blue_max),
            rbits: red_max.count_ones(),
            gbits: green_max.count_ones(),
            bbits: blue_max.count_ones(),
            rshift: u32::from(red_shift),
            gshift: u32::from(green_shift),
            bshift: u32::from(blue_shift),
            big_endian: big_endian_flag != 0,
        })
    }

    /// Whether `qemu_pixman_get_format()` of this format is the server's `x8r8g8b8`, which is
    /// when `set_pixel_conversion()` picks the plain copy. Like `qemu_pixman_get_type()` this
    /// looks at the order of the shifts and the channel widths, so a byte swapped client with
    /// the shifts the other way round also gets the copy.
    pub fn is_server_format(&self) -> bool {
        let native = self.big_endian == cfg!(target_endian = "big");
        let (r, g, b) = (self.rshift, self.gshift, self.bshift);
        let argb = if native { r > g && g > b && b == 0 } else { r < g && g < b && r != 0 };
        self.bits_per_pixel == 32 && argb && self.rbits == 8 && self.gbits == 8 && self.bbits == 8
    }

    /// `vnc_convert_pixel()`: one server pixel in the client format, `bytes_per_pixel` bytes
    /// written to the front of `buf`.
    pub fn convert(&self, buf: &mut [u8], v: u32) {
        let r = ((((v >> 16) & 0xff) << self.rbits) >> 8) & 0xff;
        let g = ((((v >> 8) & 0xff) << self.gbits) >> 8) & 0xff;
        let b = (((v & 0xff) << self.bbits) >> 8) & 0xff;
        let v = (r << self.rshift) | (g << self.gshift) | (b << self.bshift);
        match self.bytes_per_pixel {
            1 => buf[0] = v as u8,
            2 => {
                let b = if self.big_endian {
                    (v as u16).to_be_bytes()
                } else {
                    (v as u16).to_le_bytes()
                };
                buf[..2].copy_from_slice(&b);
            }
            _ => {
                let b = if self.big_endian { v.to_be_bytes() } else { v.to_le_bytes() };
                buf[..4].copy_from_slice(&b);
            }
        }
    }

    /// One pixel value in the client format, as a number, which is how the tight encoder's
    /// palette code sees the bytes `convert` writes when it reads them back in host order.
    pub fn convert_value(&self, v: u32) -> u32 {
        let mut b = [0u8; 4];
        self.convert(&mut b, v);
        match self.bytes_per_pixel {
            1 => u32::from(b[0]),
            2 => u32::from(u16::from_ne_bytes([b[0], b[1]])),
            _ => u32::from_ne_bytes(b),
        }
    }
}

/// How `write_pixels` turns server pixels into bytes for one client.
#[derive(Clone, Copy, Debug)]
pub struct PixelWriter {
    pub pf: VncPixelFormat,
    /// `vnc_write_pixels_generic` rather than `vnc_write_pixels_copy`. The hextile encoder
    /// switches between its generic and its 32 bit tile function on the same flag.
    pub generic: bool,
}

impl PixelWriter {
    /// What `pixel_format_message()` leaves behind: the default format and the plain copy.
    pub const fn server_default() -> PixelWriter {
        PixelWriter { pf: VncPixelFormat::server_default(), generic: false }
    }

    /// `set_pixel_conversion()`.
    pub fn for_format(pf: VncPixelFormat) -> PixelWriter {
        PixelWriter { pf, generic: !pf.is_server_format() }
    }

    /// The bytes one pixel takes on the wire.
    pub fn bytes_per_pixel(&self) -> usize {
        if self.generic { self.pf.bytes_per_pixel } else { 4 }
    }

    /// `vs->write_pixels()` for a run of server pixels.
    pub fn write(&self, out: &mut Vec<u8>, pixels: &[u32]) {
        if !self.generic {
            for p in pixels {
                out.extend_from_slice(&p.to_ne_bytes());
            }
            return;
        }
        let n = self.pf.bytes_per_pixel;
        let mut buf = [0u8; 4];
        for &p in pixels {
            self.pf.convert(&mut buf, p);
            out.extend_from_slice(&buf[..n]);
        }
    }

    /// One pixel, the way the hextile tile function stores a subrect colour: a plain copy of the
    /// four bytes, or the converted pixel.
    pub fn write_one(&self, out: &mut Vec<u8>, p: u32) {
        self.write(out, std::slice::from_ref(&p));
    }
}
