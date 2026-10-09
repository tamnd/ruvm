// SPDX-License-Identifier: GPL-2.0-or-later

//! The hextile encoding, from QEMU's ui/vnc-enc-hextile.c and ui/vnc-enc-hextile-template.h.
//!
//! The rectangle is cut into 16x16 tiles. A tile of one colour is sent as its background, a
//! tile of two colours as a background, a foreground and runs of the foreground, and a tile of
//! more colours as runs that each carry their colour. A tile whose runs would take more room
//! than its pixels is sent raw. Background and foreground carry over from tile to tile.
//!
//! QEMU instantiates the template for 32 bit server pixels twice: once copying the pixels as
//! they are and once converting them for the client. Both are this one function, with the
//! difference inside [`PixelWriter`].

use super::Fb;
use super::pixels::PixelWriter;

const RAW: u8 = 0x01;
const BACKGROUND_SPECIFIED: u8 = 0x02;
const FOREGROUND_SPECIFIED: u8 = 0x04;
const ANY_SUBRECTS: u8 = 0x08;
const SUBRECTS_COLOURED: u8 = 0x10;

/// What carries over from one tile to the next: `last_bg`, `last_fg`, `has_bg` and `has_fg`.
#[derive(Default)]
struct Carry {
    last_bg: u32,
    last_fg: u32,
    has_bg: bool,
    has_fg: bool,
}

/// `hextile_enc_cord()`.
fn enc_cord(data: &mut Vec<u8>, x: usize, y: usize, w: usize, h: usize) {
    data.push((((x & 0x0f) << 4) | (y & 0x0f)) as u8);
    data.push(((((w - 1) & 0x0f) << 4) | ((h - 1) & 0x0f)) as u8);
}

/// `vnc_hextile_send_framebuffer_update()`, without the rectangle header.
pub(crate) fn send(
    out: &mut Vec<u8>,
    fb: &Fb<'_>,
    pw: &PixelWriter,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) -> i32 {
    let mut carry = Carry::default();
    let mut j = y;
    while j < y + h {
        let mut i = x;
        while i < x + w {
            send_tile(out, fb, pw, i, j, 16.min(x + w - i), 16.min(y + h - j), &mut carry);
            i += 16;
        }
        j += 16;
    }
    1
}

/// `send_hextile_tile_*()`.
#[allow(clippy::too_many_arguments)]
fn send_tile(
    out: &mut Vec<u8>,
    fb: &Fb<'_>,
    pw: &PixelWriter,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    carry: &mut Carry,
) {
    let mut bg = 0u32;
    let mut fg = 0u32;
    let mut n_colors = 0;
    let mut bg_count = 0;
    let mut fg_count = 0;
    let mut flags = 0u8;
    let mut data = Vec::new();
    let mut n_subtiles = 0usize;

    'rows: for j in 0..h {
        for &p in fb.row(x, y + j, w) {
            match n_colors {
                0 => {
                    bg = p;
                    n_colors = 1;
                }
                1 => {
                    if p != bg {
                        fg = p;
                        n_colors = 2;
                    }
                }
                2 => {
                    if p != bg && p != fg {
                        n_colors = 3;
                    } else if p == bg {
                        bg_count += 1;
                    } else if p == fg {
                        fg_count += 1;
                    }
                }
                _ => {}
            }
        }
        if n_colors > 2 {
            break 'rows;
        }
    }

    if n_colors > 1 && fg_count > bg_count {
        std::mem::swap(&mut fg, &mut bg);
    }

    if !carry.has_bg || carry.last_bg != bg {
        flags |= BACKGROUND_SPECIFIED;
        carry.has_bg = true;
        carry.last_bg = bg;
    }

    if n_colors < 3 && (!carry.has_fg || carry.last_fg != fg) {
        flags |= FOREGROUND_SPECIFIED;
        carry.has_fg = true;
        carry.last_fg = fg;
    }

    match n_colors {
        2 => {
            flags |= ANY_SUBRECTS;
            for j in 0..h {
                let row = fb.row(x, y + j, w);
                let mut min_x: Option<usize> = None;
                for (i, &p) in row.iter().enumerate() {
                    if p == fg {
                        if min_x.is_none() {
                            min_x = Some(i);
                        }
                    } else if let Some(m) = min_x.take() {
                        enc_cord(&mut data, m, j, i - m, 1);
                        n_subtiles += 1;
                    }
                }
                if let Some(m) = min_x {
                    enc_cord(&mut data, m, j, w - m, 1);
                    n_subtiles += 1;
                }
            }
        }
        3 => {
            flags |= ANY_SUBRECTS | SUBRECTS_COLOURED;
            if !carry.has_bg || carry.last_bg != bg {
                flags |= BACKGROUND_SPECIFIED;
            }
            for j in 0..h {
                let row = fb.row(x, y + j, w);
                let mut run: Option<(u32, usize)> = None;
                for (i, &p) in row.iter().enumerate() {
                    match run {
                        None => {
                            if p != bg {
                                run = Some((p, i));
                            }
                        }
                        Some((color, min_x)) => {
                            if p != color {
                                pw.write_one(&mut data, color);
                                enc_cord(&mut data, min_x, j, i - min_x, 1);
                                n_subtiles += 1;
                                run = if p != bg { Some((p, i)) } else { None };
                            }
                        }
                    }
                }
                if let Some((color, min_x)) = run {
                    pw.write_one(&mut data, color);
                    enc_cord(&mut data, min_x, j, w - min_x, 1);
                    n_subtiles += 1;
                }
            }
            // A SubrectsColoured subtile invalidates the foreground color.
            carry.has_fg = false;
            if data.len() > w * h * 4 {
                n_colors = 4;
                flags = RAW;
                carry.has_bg = false;
            }
        }
        _ => {}
    }

    if n_colors > 3 {
        flags = RAW;
        carry.has_fg = false;
        carry.has_bg = false;
        n_colors = 4;
    }

    out.push(flags);
    if n_colors < 4 {
        if flags & BACKGROUND_SPECIFIED != 0 {
            pw.write_one(out, carry.last_bg);
        }
        if flags & FOREGROUND_SPECIFIED != 0 {
            pw.write_one(out, carry.last_fg);
        }
        if n_subtiles != 0 {
            // A u8 on the wire, like QEMU's vnc_write_u8().
            out.push(n_subtiles as u8);
            out.extend_from_slice(&data);
        }
    } else {
        for j in 0..h {
            pw.write(out, fb.row(x, y + j, w));
        }
    }
}
