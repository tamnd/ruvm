// SPDX-License-Identifier: GPL-2.0-or-later

//! The raw encoding, from QEMU's ui/vnc-enc-raw.c: every pixel of the rectangle in the client
//! format, row by row.

use super::Fb;
use super::pixels::PixelWriter;

/// `vnc_raw_send_framebuffer_update()`, without the rectangle header.
pub(crate) fn send(
    out: &mut Vec<u8>,
    fb: &Fb<'_>,
    pw: &PixelWriter,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) -> i32 {
    for row in y..y + h {
        pw.write(out, fb.row(x, row, w));
    }
    1
}
