// SPDX-License-Identifier: GPL-2.0-or-later

//! Cluster compression, from block/qcow2-threads.c.
//!
//! Only deflate is built in. QEMU links libzstd when it is available; the zstd crate wraps the
//! C library and needs a C cross compiler for every target this workspace checks, so it is not
//! used and images with `compression_type=zstd` are refused the way a QEMU built without zstd
//! refuses them.

use std::io;

use flate2::{Compress, Compression, Decompress, FlushCompress, FlushDecompress, Status};

use crate::node::errno;

/// The deflate window QEMU uses: 4 KiB, no zlib header.
const WINDOW_BITS: u8 = 12;

/// `qcow2_zlib_compress()`: compresses `src` into at most `dest.len()` bytes and returns the
/// compressed length. ENOMEM means the result does not fit, and the caller writes the cluster
/// uncompressed instead.
pub(crate) fn zlib_compress(dest: &mut [u8], src: &[u8]) -> io::Result<usize> {
    let mut c = Compress::new_with_window_bits(Compression::default(), false, WINDOW_BITS);
    match c.compress(src, dest, FlushCompress::Finish) {
        Ok(Status::StreamEnd) => Ok(c.total_out() as usize),
        Ok(_) => Err(errno(libc::ENOMEM)),
        Err(_) => Err(errno(libc::EIO)),
    }
}

/// `qcow2_zlib_decompress()`: `dest` must be filled completely. The input may carry some
/// bytes of padding past the end of the stream, because the compressed size is only known to
/// the sector.
pub(crate) fn zlib_decompress(dest: &mut [u8], src: &[u8]) -> io::Result<()> {
    let mut d = Decompress::new_with_window_bits(false, WINDOW_BITS);
    match d.decompress(src, dest, FlushDecompress::Finish) {
        Ok(Status::StreamEnd | Status::BufError | Status::Ok)
            if d.total_out() as usize == dest.len() =>
        {
            Ok(())
        }
        _ => Err(errno(libc::EIO)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let src: Vec<u8> = (0..65536u32).map(|i| (i / 100) as u8).collect();
        let mut out = vec![0u8; 65535];
        let n = zlib_compress(&mut out, &src).unwrap();
        assert!(n < 4096);
        let mut back = vec![0u8; 65536];
        // Padding after the stream is fine.
        zlib_decompress(&mut back, &out[..n + 100]).unwrap();
        assert_eq!(back, src);
    }

    #[test]
    fn incompressible_data_does_not_fit() {
        let mut x = 0x1234_5678u32;
        let src: Vec<u8> = (0..4096)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x as u8
            })
            .collect();
        let mut out = vec![0u8; 4095];
        let e = zlib_compress(&mut out, &src).unwrap_err();
        assert_eq!(e.raw_os_error(), Some(libc::ENOMEM));
    }

    #[test]
    fn short_output_is_an_error() {
        let src = vec![7u8; 1000];
        let mut out = vec![0u8; 999];
        let n = zlib_compress(&mut out, &src).unwrap();
        let mut back = vec![0u8; 2000];
        assert!(zlib_decompress(&mut back, &out[..n]).is_err());
    }
}
