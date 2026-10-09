// SPDX-License-Identifier: GPL-2.0-or-later

//! The zlib encoding, from QEMU's ui/vnc-enc-zlib.c, and the deflate stream it shares with the
//! tight encoder.
//!
//! The rectangle is the raw encoding of its pixels pushed through one deflate stream per client
//! with a sync flush, so the client inflates every rectangle with the same stream. QEMU opens the
//! stream with zlib's largest `memLevel` and flate2 has no way to ask for it, so the compressed
//! bytes can differ from QEMU's. They inflate to the same pixels.
//!
//! flate2 over zlib-rs calls `deflateParams()` without an output buffer, which fails once the
//! stream has been used and the new level needs another strategy (stored for 0, fast for 1 to 3,
//! slow for 4 to 9). Where QEMU would switch, the stream keeps its old level, so the output only
//! compresses differently.

use flate2::{Compress, Compression, FlushCompress, Status};

use super::pixels::PixelWriter;
use super::{ENCODING_ZLIB, Fb, framebuffer_update};

/// A deflate stream with the level it was last set to, the `z_stream` and `level` pairs QEMU
/// keeps.
pub(crate) struct ZStream {
    c: Compress,
    level: u32,
}

impl ZStream {
    /// `deflateInit2()` with a zlib header and a 32 KiB window.
    pub(crate) fn new(level: u32) -> ZStream {
        ZStream { c: Compress::new(Compression::new(level), true), level }
    }

    /// `deflateParams()` when `level` differs from the stream's. A level zlib-rs cannot switch
    /// to leaves the stream at its old one.
    pub(crate) fn set_level(&mut self, level: u32) {
        if self.level != level && self.c.set_level(Compression::new(level)).is_ok() {
            self.level = level;
        }
    }

    /// `deflate(Z_SYNC_FLUSH)` of all of `input`, appended to `out`.
    pub(crate) fn deflate_sync(&mut self, input: &[u8], out: &mut Vec<u8>) -> bool {
        let start = self.c.total_in();
        out.reserve(input.len() + input.len() / 1000 + 64);
        loop {
            let done = (self.c.total_in() - start) as usize;
            if out.capacity() - out.len() < 64 {
                out.reserve(4096);
            }
            match self.c.compress_vec(&input[done..], out, FlushCompress::Sync) {
                Ok(Status::Ok) | Ok(Status::BufError) => {}
                Ok(Status::StreamEnd) | Err(_) => return false,
            }
            let done = (self.c.total_in() - start) as usize;
            // The flush is complete once the input is in and zlib stopped short of the end of
            // the buffer.
            if done == input.len() && out.len() < out.capacity() {
                return true;
            }
        }
    }
}

/// `vnc_zlib_send_framebuffer_update()`, with the rectangle header. `level` is the tight
/// compression level the client asked for, which QEMU uses for this encoding too.
#[allow(clippy::too_many_arguments)]
pub(crate) fn send(
    out: &mut Vec<u8>,
    stream: &mut Option<ZStream>,
    level: u32,
    fb: &Fb<'_>,
    pw: &PixelWriter,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) -> i32 {
    framebuffer_update(out, x, y, w, h, ENCODING_ZLIB);
    let len_at = out.len();
    out.extend_from_slice(&[0; 4]);
    let mut raw = Vec::with_capacity(w * h * pw.bytes_per_pixel());
    super::raw::send(&mut raw, fb, pw, x, y, w, h);
    let z = stream.get_or_insert_with(|| ZStream::new(level));
    z.set_level(level);
    // QEMU leaves the header and the zero length in the output when zlib fails.
    let before = out.len();
    if !z.deflate_sync(&raw, out) {
        out.truncate(before);
        return 0;
    }
    let written = (out.len() - before) as u32;
    out[len_at..len_at + 4].copy_from_slice(&written.to_be_bytes());
    1
}
