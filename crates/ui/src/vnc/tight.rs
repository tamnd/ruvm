// SPDX-License-Identifier: GPL-2.0-or-later

//! The tight encoding without JPEG and PNG, from QEMU's ui/vnc-enc-tight.c.
//!
//! Large rectangles are searched for big areas of one colour, which go out as solid fills, and
//! the rest is cut into pieces of at most `max_rect_size` pixels. Each piece is sent as a fill
//! when it has one colour, as a bitmap when it has two, through a palette when it has few enough
//! and as full colour otherwise, the last three through one of four deflate streams.
//!
//! The gradient filter and JPEG are only used for lossy displays, and like a QEMU built without
//! JPEG support this one never is, so neither is here. The pixel buffers hold the rectangle in
//! the client's format and are read back as host order numbers, which is how QEMU compares them.

use super::palette::Palette;
use super::pixels::{PixelWriter, VncPixelFormat};
use super::zlib::ZStream;
use super::{ENCODING_TIGHT, Fb, framebuffer_update};

const EXPLICIT_FILTER: u8 = 0x04;
const FILL: u8 = 0x08;
const FILTER_PALETTE: u8 = 0x01;
const MIN_TO_COMPRESS: usize = 12;
const MIN_SPLIT_RECT_SIZE: i64 = 4096;
const MIN_SOLID_SUBRECT_SIZE: i64 = 2048;
const MAX_SPLIT_TILE_SIZE: i64 = 16;

/// The columns of `tight_conf` a lossless encoder reads.
struct Conf {
    max_rect_size: i64,
    max_rect_width: i64,
    mono_min_rect_size: usize,
    idx_zlib_level: u32,
    mono_zlib_level: u32,
    raw_zlib_level: u32,
    idx_max_colors_divisor: usize,
}

const fn conf(
    max_rect_size: i64,
    max_rect_width: i64,
    mono_min_rect_size: usize,
    idx_zlib_level: u32,
    mono_zlib_level: u32,
    raw_zlib_level: u32,
    idx_max_colors_divisor: usize,
) -> Conf {
    Conf {
        max_rect_size,
        max_rect_width,
        mono_min_rect_size,
        idx_zlib_level,
        mono_zlib_level,
        raw_zlib_level,
        idx_max_colors_divisor,
    }
}

/// `tight_conf[]`, indexed by the compression level.
const TIGHT_CONF: [Conf; 10] = [
    conf(512, 32, 6, 0, 0, 0, 4),
    conf(2048, 128, 6, 1, 1, 1, 8),
    conf(6144, 256, 8, 3, 3, 2, 24),
    conf(10240, 1024, 12, 5, 5, 3, 32),
    conf(16384, 2048, 12, 6, 6, 4, 32),
    conf(32768, 2048, 12, 7, 7, 5, 32),
    conf(65536, 2048, 16, 7, 7, 6, 48),
    conf(65536, 2048, 16, 8, 8, 7, 64),
    conf(65536, 2048, 32, 9, 9, 8, 64),
    conf(65536, 2048, 32, 9, 9, 9, 96),
];

/// The tight half of `VncWorker`.
pub(crate) struct Tight {
    /// The `CompressLevel` pseudo encoding, 9 unless the client says otherwise.
    pub(crate) compression: u8,
    /// The `QualityLevel` pseudo encoding, -1 for lossless. Only lossy displays take it.
    pub(crate) quality: i32,
    streams: [Option<ZStream>; 4],
    pixel24: bool,
}

impl Default for Tight {
    fn default() -> Tight {
        Tight { compression: 9, quality: -1, streams: [None, None, None, None], pixel24: false }
    }
}

/// Everything one update needs: the output, the server pixels and the client's format.
struct Ctx<'a, 'b> {
    out: &'a mut Vec<u8>,
    fb: &'a Fb<'b>,
    pw: &'a PixelWriter,
    t: &'a mut Tight,
}

impl Ctx<'_, '_> {
    fn conf(&self) -> &'static Conf {
        &TIGHT_CONF[usize::from(self.t.compression.min(9))]
    }

    fn pf(&self) -> &VncPixelFormat {
        &self.pw.pf
    }

    /// The rectangle in the client's format, `vnc_raw_send_framebuffer_update()` into the tight
    /// buffer.
    fn raw(&self, x: i64, y: i64, w: i64, h: i64) -> Vec<u8> {
        let (x, y, w, h) = (x as usize, y as usize, w as usize, h as usize);
        let mut buf = Vec::with_capacity(w * h * self.pw.bytes_per_pixel());
        super::raw::send(&mut buf, self.fb, self.pw, x, y, w, h);
        buf
    }

    /// `check_solid_tile32()` on the server surface.
    fn check_solid_tile(
        &self,
        x: i64,
        y: i64,
        w: i64,
        h: i64,
        color: &mut u32,
        samecolor: bool,
    ) -> bool {
        let c = self.fb.pixel(x as usize, y as usize);
        if samecolor && c != *color {
            return false;
        }
        for dy in 0..h {
            let row = self.fb.row(x as usize, (y + dy) as usize, w as usize);
            if row.iter().any(|&p| p != c) {
                return false;
            }
        }
        *color = c;
        true
    }

    /// `find_best_solid_area()`.
    fn find_best_solid_area(&self, x: i64, y: i64, w: i64, h: i64, mut color: u32) -> (i64, i64) {
        let mut w_prev = w;
        let (mut w_best, mut h_best) = (0, 0);
        let mut dy = y;
        while dy < y + h {
            let dh = MAX_SPLIT_TILE_SIZE.min(y + h - dy);
            let dw = MAX_SPLIT_TILE_SIZE.min(w_prev);
            if !self.check_solid_tile(x, dy, dw, dh, &mut color, true) {
                break;
            }
            let mut dx = x + dw;
            while dx < x + w_prev {
                let dw = MAX_SPLIT_TILE_SIZE.min(x + w_prev - dx);
                if !self.check_solid_tile(dx, dy, dw, dh, &mut color, true) {
                    break;
                }
                dx += dw;
            }
            w_prev = dx - x;
            if w_prev * (dy + dh - y) > w_best * h_best {
                w_best = w_prev;
                h_best = dy + dh - y;
            }
            dy += MAX_SPLIT_TILE_SIZE;
        }
        (w_best, h_best)
    }

    /// `extend_solid_area()`.
    #[allow(clippy::too_many_arguments)]
    fn extend_solid_area(
        &self,
        x: i64,
        y: i64,
        w: i64,
        h: i64,
        mut color: u32,
        xp: &mut i64,
        yp: &mut i64,
        wp: &mut i64,
        hp: &mut i64,
    ) {
        let mut cy = *yp - 1;
        while cy >= y && self.check_solid_tile(*xp, cy, *wp, 1, &mut color, true) {
            cy -= 1;
        }
        *hp += *yp - (cy + 1);
        *yp = cy + 1;

        let mut cy = *yp + *hp;
        while cy < y + h && self.check_solid_tile(*xp, cy, *wp, 1, &mut color, true) {
            cy += 1;
        }
        *hp += cy - (*yp + *hp);

        let mut cx = *xp - 1;
        while cx >= x && self.check_solid_tile(cx, *yp, 1, *hp, &mut color, true) {
            cx -= 1;
        }
        *wp += *xp - (cx + 1);
        *xp = cx + 1;

        let mut cx = *xp + *wp;
        while cx < x + w && self.check_solid_tile(cx, *yp, 1, *hp, &mut color, true) {
            cx += 1;
        }
        *wp += cx - (*xp + *wp);
    }

    /// `tight_compress_data()`: short data as it is, the rest deflated through stream `id` with
    /// its length in front.
    fn compress_data(&mut self, id: usize, data: &[u8], level: u32) -> i64 {
        if data.len() < MIN_TO_COMPRESS {
            self.out.extend_from_slice(data);
            return data.len() as i64;
        }
        // tight_init_stream()
        let z = self.t.streams[id].get_or_insert_with(|| ZStream::new(level));
        z.set_level(level);
        let mut zbuf = Vec::new();
        if !z.deflate_sync(data, &mut zbuf) {
            return -1;
        }
        send_compact_size(self.out, zbuf.len());
        self.out.extend_from_slice(&zbuf);
        zbuf.len() as i64
    }

    /// `tight_pack24()`: four byte pixels down to their three colour bytes.
    fn pack24(&self, buf: &[u8]) -> Vec<u8> {
        let pf = self.pf();
        let (r, g, b) = if pf.big_endian == cfg!(target_endian = "big") {
            (pf.rshift as i64, pf.gshift as i64, pf.bshift as i64)
        } else {
            (24 - pf.rshift as i64, 24 - pf.gshift as i64, 24 - pf.bshift as i64)
        };
        let shr = |p: u32, s: i64| if (0..32).contains(&s) { (p >> s) as u8 } else { 0 };
        let mut out = Vec::with_capacity(buf.len() / 4 * 3);
        for px in buf.chunks_exact(4) {
            let p = u32::from_ne_bytes([px[0], px[1], px[2], px[3]]);
            out.extend_from_slice(&[shr(p, r), shr(p, g), shr(p, b)]);
        }
        out
    }

    /// `send_full_color_rect()`.
    fn send_full_color_rect(&mut self, buf: &[u8]) -> i32 {
        let level = self.conf().raw_zlib_level;
        self.out.push(0);
        let data = if self.t.pixel24 { self.pack24(buf) } else { buf.to_vec() };
        i32::from(self.compress_data(0, &data, level) >= 0)
    }

    /// `send_solid_rect()`.
    fn send_solid_rect(&mut self, buf: &[u8]) -> i32 {
        self.out.push(FILL << 4);
        if self.t.pixel24 {
            let p = self.pack24(&buf[..4]);
            self.out.extend_from_slice(&p);
        } else {
            let n = self.pf().bytes_per_pixel;
            self.out.extend_from_slice(&buf[..n]);
        }
        1
    }

    /// `send_mono_rect()`.
    fn send_mono_rect(&mut self, buf: &[u8], w: usize, h: usize, bg: u32, fg: u32) -> i32 {
        let level = self.conf().mono_zlib_level;
        self.out.push((1 | EXPLICIT_FILTER) << 4);
        self.out.push(FILTER_PALETTE);
        self.out.push(1);
        let values = pixel_values(buf, self.pf().bytes_per_pixel);
        match self.pf().bytes_per_pixel {
            4 => {
                let mut colors = Vec::with_capacity(8);
                colors.extend_from_slice(&bg.to_ne_bytes());
                colors.extend_from_slice(&fg.to_ne_bytes());
                let colors = if self.t.pixel24 { self.pack24(&colors) } else { colors };
                self.out.extend_from_slice(&colors);
            }
            2 => {
                self.out.extend_from_slice(&(bg as u16).to_ne_bytes());
                self.out.extend_from_slice(&(fg as u16).to_ne_bytes());
            }
            _ => {
                self.out.push(bg as u8);
                self.out.push(fg as u8);
            }
        }
        let data = encode_mono_rect(&values, w, h, bg);
        i32::from(self.compress_data(1, &data, level) >= 0)
    }

    /// `send_palette_rect()`.
    fn send_palette_rect(&mut self, buf: &[u8], palette: &Palette) -> i32 {
        let level = self.conf().idx_zlib_level;
        let colors = palette.size();
        self.out.push((2 | EXPLICIT_FILTER) << 4);
        self.out.push(FILTER_PALETTE);
        self.out.push((colors as u8).wrapping_sub(1));
        match self.pf().bytes_per_pixel {
            4 => {
                let mut header = Vec::with_capacity(colors * 4);
                for &c in palette.colors() {
                    header.extend_from_slice(&c.to_ne_bytes());
                }
                let header = if self.t.pixel24 { self.pack24(&header) } else { header };
                self.out.extend_from_slice(&header);
            }
            2 => {
                for &c in palette.colors() {
                    self.out.extend_from_slice(&(c as u16).to_ne_bytes());
                }
            }
            // No palette for 8 bit colours.
            _ => return -1,
        }
        let values = pixel_values(buf, self.pf().bytes_per_pixel);
        let data = encode_indexed_rect(&values, palette);
        i32::from(self.compress_data(2, &data, level) >= 0)
    }

    /// `send_sub_rect()` with `send_sub_rect_nojpeg()`.
    fn send_sub_rect(&mut self, x: i64, y: i64, w: i64, h: i64) -> i32 {
        framebuffer_update(
            self.out,
            x as usize,
            y as usize,
            w as usize,
            h as usize,
            ENCODING_TIGHT,
        );
        let buf = self.raw(x, y, w, h);
        let (colors, bg, fg, palette) = self.fill_palette(&buf, (w * h) as usize);
        match colors {
            // Without lossy updates tight_detect_smooth_image() is always false.
            0 => self.send_full_color_rect(&buf),
            1 => self.send_solid_rect(&buf),
            2 => self.send_mono_rect(&buf, w as usize, h as usize, bg, fg),
            3..=256 => match palette {
                Some(p) => self.send_palette_rect(&buf, &p),
                None => 0,
            },
            _ => 0,
        }
    }

    /// `send_sub_rect_solid()`.
    fn send_sub_rect_solid(&mut self, x: i64, y: i64, w: i64, h: i64) -> i32 {
        framebuffer_update(
            self.out,
            x as usize,
            y as usize,
            w as usize,
            h as usize,
            ENCODING_TIGHT,
        );
        let buf = self.raw(x, y, w, h);
        self.send_solid_rect(&buf)
    }

    /// `send_rect_simple()`.
    fn send_rect_simple(&mut self, x: i64, y: i64, w: i64, h: i64, split: bool) -> i32 {
        let max_size = self.conf().max_rect_size;
        let max_width = self.conf().max_rect_width;
        let mut n = 0;
        if split && (w > max_width || w * h > max_size) {
            let max_sub_width = w.min(max_width);
            let max_sub_height = max_size / max_sub_width;
            let mut dy = 0;
            while dy < h {
                let mut dx = 0;
                while dx < w {
                    let rw = max_sub_width.min(w - dx);
                    let rh = max_sub_height.min(h - dy);
                    n += self.send_sub_rect(x + dx, y + dy, rw, rh);
                    dx += max_width;
                }
                dy += max_sub_height;
            }
        } else {
            n += self.send_sub_rect(x, y, w, h);
        }
        n
    }

    /// `find_large_solid_color_rect()`.
    fn find_large_solid_color_rect(
        &mut self,
        x: i64,
        mut y: i64,
        w: i64,
        mut h: i64,
        max_rows: i64,
    ) -> i32 {
        let mut n = 0;
        let mut dy = y;
        while dy < y + h {
            if dy - y >= max_rows {
                n += self.send_rect_simple(x, y, w, max_rows, true);
                y += max_rows;
                h -= max_rows;
            }
            let dh = MAX_SPLIT_TILE_SIZE.min(y + h - dy);
            let mut dx = x;
            while dx < x + w {
                let dw = MAX_SPLIT_TILE_SIZE.min(x + w - dx);
                let mut color = 0;
                if self.check_solid_tile(dx, dy, dw, dh, &mut color, false) {
                    let (w_best, h_best) =
                        self.find_best_solid_area(dx, dy, w - (dx - x), h - (dy - y), color);
                    // Make sure a solid rectangle is large enough, or the whole rectangle is of
                    // the same colour.
                    if w_best * h_best == w * h || w_best * h_best >= MIN_SOLID_SUBRECT_SIZE {
                        let (mut xb, mut yb, mut wb, mut hb) = (dx, dy, w_best, h_best);
                        self.extend_solid_area(
                            x, y, w, h, color, &mut xb, &mut yb, &mut wb, &mut hb,
                        );
                        if yb != y {
                            n += self.send_rect_simple(x, y, w, yb - y, true);
                        }
                        if xb != x {
                            n += self.update(x, yb, xb - x, hb);
                        }
                        n += self.send_sub_rect_solid(xb, yb, wb, hb);
                        if xb + wb != x + w {
                            n += self.update(xb + wb, yb, w - (xb - x) - wb, hb);
                        }
                        if yb + hb != y + h {
                            n += self.update(x, yb + hb, w, h - (yb - y) - hb);
                        }
                        return n;
                    }
                }
                dx += MAX_SPLIT_TILE_SIZE;
            }
            dy += MAX_SPLIT_TILE_SIZE;
        }
        n + self.send_rect_simple(x, y, w, h, true)
    }

    /// `tight_send_framebuffer_update()`.
    fn update(&mut self, x: i64, y: i64, w: i64, h: i64) -> i32 {
        let pf = self.pf();
        self.t.pixel24 =
            pf.bytes_per_pixel == 4 && pf.rmax == 0xff && pf.bmax == 0xff && pf.gmax == 0xff;
        if w * h < MIN_SPLIT_RECT_SIZE {
            return self.send_rect_simple(x, y, w, h, true);
        }
        let max_rows = self.conf().max_rect_size / self.conf().max_rect_width.min(w);
        self.find_large_solid_color_rect(x, y, w, h, max_rows)
    }

    /// `tight_fill_palette()`: the number of colours, 0 when there are too many, with the
    /// background and foreground of a two colour rectangle and the palette of a bigger one.
    fn fill_palette(&self, buf: &[u8], count: usize) -> (usize, u32, u32, Option<Palette>) {
        let conf = self.conf();
        let mut max = count / conf.idx_max_colors_divisor;
        if max < 2 && count >= conf.mono_min_rect_size {
            max = 2;
        }
        if max >= 256 {
            max = 256;
        }
        let bpp = self.pf().bytes_per_pixel;
        if bpp != 4 && bpp != 2 {
            max = 2;
        }
        let data = pixel_values(buf, bpp);
        fill_palette(&data, max, count)
    }
}

/// The pixels of a buffer in the client's format as host order numbers.
fn pixel_values(buf: &[u8], bpp: usize) -> Vec<u32> {
    match bpp {
        4 => buf.chunks_exact(4).map(|p| u32::from_ne_bytes([p[0], p[1], p[2], p[3]])).collect(),
        2 => buf.chunks_exact(2).map(|p| u32::from(u16::from_ne_bytes([p[0], p[1]]))).collect(),
        _ => buf.iter().map(|&p| u32::from(p)).collect(),
    }
}

/// `tight_fill_palette8/16/32()`.
fn fill_palette(data: &[u32], max: usize, count: usize) -> (usize, u32, u32, Option<Palette>) {
    let c0 = data[0];
    let mut i = 1;
    while i < count && data[i] == c0 {
        i += 1;
    }
    if i >= count {
        return (1, c0, c0, None);
    }
    if max < 2 {
        return (0, 0, 0, None);
    }
    let mut n0 = i;
    let c1 = data[i];
    let mut n1 = 0;
    let mut ci = c1;
    i += 1;
    while i < count {
        ci = data[i];
        if ci == c0 {
            n0 += 1;
        } else if ci == c1 {
            n1 += 1;
        } else {
            break;
        }
        i += 1;
    }
    if i >= count {
        return if n0 > n1 { (2, c0, c1, None) } else { (2, c1, c0, None) };
    }
    if max == 2 {
        return (0, 0, 0, None);
    }
    let mut palette = Palette::new(max);
    palette.put(c0);
    palette.put(c1);
    palette.put(ci);
    i += 1;
    while i < count {
        if data[i] != ci {
            ci = data[i];
            if palette.put(ci) == 0 {
                return (0, 0, 0, None);
            }
        }
        i += 1;
    }
    (palette.size(), 0, 0, Some(palette))
}

/// `tight_encode_indexed_rect16/32()`: one palette index a pixel.
pub(super) fn encode_indexed_rect(data: &[u32], palette: &Palette) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let rgb = data[i];
        let mut rep = 1;
        while i + rep < data.len() && data[i + rep] == rgb {
            rep += 1;
        }
        // QEMU's index is a uint8_t that it checks against (uint8_t)-1 for a colour missing
        // from the palette, so the 256th colour of a full palette goes out as the first.
        let idx = match palette.idx(rgb) {
            Some(255) | None => 0,
            Some(i) => i,
        };
        out.extend(std::iter::repeat_n(idx, rep));
        i += rep;
    }
    out
}

/// `tight_encode_mono_rect8/16/32()`: one bit a pixel, set for the foreground, most significant
/// bit first, each row padded to a byte.
fn encode_mono_rect(data: &[u32], w: usize, h: usize, bg: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(w.div_ceil(8) * h);
    let mut ptr = 0;
    let aligned_width = w - w % 8;
    for _ in 0..h {
        let mut x = 0;
        while x < aligned_width {
            let mut bg_bits = 0;
            while bg_bits < 8 {
                let p = data[ptr];
                ptr += 1;
                if p != bg {
                    break;
                }
                bg_bits += 1;
            }
            if bg_bits == 8 {
                out.push(0);
                x += 8;
                continue;
            }
            let mut mask = 0x80u32 >> bg_bits;
            let mut value = mask;
            bg_bits += 1;
            while bg_bits < 8 {
                mask >>= 1;
                if data[ptr] != bg {
                    value |= mask;
                }
                ptr += 1;
                bg_bits += 1;
            }
            out.push(value as u8);
            x += 8;
        }
        if x >= w {
            continue;
        }
        let mut mask = 0x80u32;
        let mut value = 0;
        while x < w {
            if data[ptr] != bg {
                value |= mask;
            }
            ptr += 1;
            mask >>= 1;
            x += 1;
        }
        out.push(value as u8);
    }
    out
}

/// `tight_send_compact_size()`.
fn send_compact_size(out: &mut Vec<u8>, len: usize) {
    let mut buf = [(len & 0x7f) as u8, 0, 0];
    let mut bytes = 1;
    if len > 0x7f {
        buf[0] |= 0x80;
        buf[1] = ((len >> 7) & 0x7f) as u8;
        bytes = 2;
        if len > 0x3fff {
            buf[1] |= 0x80;
            buf[2] = ((len >> 14) & 0xff) as u8;
            bytes = 3;
        }
    }
    out.extend_from_slice(&buf[..bytes]);
}

/// `vnc_tight_send_framebuffer_update()`, headers included. Returns the number of rectangles.
#[allow(clippy::too_many_arguments)]
pub(crate) fn send(
    out: &mut Vec<u8>,
    t: &mut Tight,
    fb: &Fb<'_>,
    pw: &PixelWriter,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) -> i32 {
    let mut ctx = Ctx { out, fb, pw, t };
    ctx.update(x as i64, y as i64, w as i64, h as i64)
}
