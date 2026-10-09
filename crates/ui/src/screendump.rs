// SPDX-License-Identifier: GPL-2.0-or-later

//! The QMP `screendump` command, from QEMU's `ui/ui-qmp-cmds.c`.
//!
//! [`screendump`] picks the console the way `qmp_screendump()` does, asks the device for a fresh
//! frame, and writes the surface as a binary ppm or an 8 bit RGB png.
//!
//! The ppm writer keeps a quirk of QEMU's: each row is written with the stride of a pixman
//! `PIXMAN_BE_r8g8b8` line buffer, which is the row rounded up to a multiple of 4 bytes, so a width
//! that is not a multiple of 4 leaves zero bytes at the end of every row. Readers that trust the
//! header see a skewed picture, but the file is byte for byte what QEMU writes.
//!
//! The png writer filters every row with filter type 0 and compresses with zlib through flate2.
//! libpng picks filters adaptively, so the bytes differ from a QEMU built with libpng while the
//! decoded pixels are the same.

use std::io::Write;

use ruvm_base::error::strerror;
use ruvm_base::{Error, Result};

use crate::console::{DisplayState, QemuConsole};
use crate::pixman::{Image, PixelFormat};

/// `ImageFormat`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ImageFormat {
    #[default]
    Ppm,
    Png,
}

impl ImageFormat {
    /// The QAPI name, `ppm` or `png`.
    pub fn from_name(name: &str) -> Option<ImageFormat> {
        match name {
            "ppm" => Some(ImageFormat::Ppm),
            "png" => Some(ImageFormat::Png),
            _ => None,
        }
    }
}

/// `qmp_screendump()`: dumps the console picked by `device` and `head` to `filename`.
///
/// Without a device the console with index 0 is used, as in QEMU, and a head without a device is
/// an error. The device is asked for an update first, which is what `qemu_console_co_wait_update()`
/// waits for.
pub fn screendump(
    ds: &DisplayState,
    filename: &str,
    device: Option<&str>,
    head: Option<i64>,
    format: Option<ImageFormat>,
) -> Result<()> {
    let con: QemuConsole = match device {
        Some(id) => ds.lookup_by_device_name(id, head.unwrap_or(0) as u32)?,
        None => {
            if head.is_some() {
                return Err(Error::generic("'head' must be specified together with 'device'"));
            }
            ds.lookup_by_index(0)
                .ok_or_else(|| Error::generic("There is no console to take a screendump from"))?
        }
    };

    con.hw_update();
    let image = con
        .with_surface(|s| s.map(|s| s.image().clone()))
        .ok_or_else(|| Error::generic("no surface"))?;

    let file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(filename)
        .map_err(|e| {
        Error::generic(format!("Could not create '{filename}': {}", strerror(&e)))
    })?;
    let mut out = std::io::BufWriter::new(file);
    let res = match format.unwrap_or_default() {
        ImageFormat::Ppm => ppm_save(&mut out, &image),
        ImageFormat::Png => png_save(&mut out, &image),
    }
    .and_then(|()| out.flush());
    if let Err(e) = res {
        drop(out);
        let _ = std::fs::remove_file(filename);
        return Err(Error::generic(format!("Unable to write to file: {}", strerror(&e))));
    }
    Ok(())
}

/// `ppm_save()`: a P6 header and the rows converted to `PIXMAN_BE_r8g8b8`, padding included.
pub fn ppm_save(out: &mut dyn Write, image: &Image) -> std::io::Result<()> {
    let (width, height) = (image.width(), image.height());
    write!(out, "P6\n{width} {height}\n255\n")?;
    let mut line = Image::new(PixelFormat::be_r8g8b8(), width, 1, 0);
    for y in 0..height {
        line.composite_src(image, 0, y as i64, 0, 0, width as i64, 1);
        out.write_all(line.data())?;
    }
    Ok(())
}

/// The ppm bytes of `image`, as [`ppm_save`] writes them.
pub fn ppm_bytes(image: &Image) -> Vec<u8> {
    let mut v = Vec::new();
    // Writing to a Vec cannot fail.
    let _ = ppm_save(&mut v, image);
    v
}

/// `png_save()`: an 8 bit RGB png, not interlaced.
pub fn png_save(out: &mut dyn Write, image: &Image) -> std::io::Result<()> {
    let (width, height) = (image.width(), image.height());
    out.write_all(&[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'])?;

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(width as u32).to_be_bytes());
    ihdr.extend_from_slice(&(height as u32).to_be_bytes());
    // Bit depth 8, colour type 2 (RGB), compression 0, filter 0, no interlace.
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    png_chunk(out, b"IHDR", &ihdr)?;

    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    let mut line = Image::new(PixelFormat::be_r8g8b8(), width, 1, 0);
    for y in 0..height {
        line.composite_src(image, 0, y as i64, 0, 0, width as i64, 1);
        z.write_all(&[0])?;
        z.write_all(&line.data()[..width * 3])?;
    }
    let idat = z.finish()?;
    png_chunk(out, b"IDAT", &idat)?;
    png_chunk(out, b"IEND", &[])
}

fn png_chunk(out: &mut dyn Write, ty: &[u8; 4], data: &[u8]) -> std::io::Result<()> {
    out.write_all(&(data.len() as u32).to_be_bytes())?;
    out.write_all(ty)?;
    out.write_all(data)?;
    let mut crc = Crc32::new();
    crc.update(ty);
    crc.update(data);
    out.write_all(&crc.finish().to_be_bytes())
}

/// The CRC-32 png chunks end with, the zlib polynomial.
struct Crc32(u32);

impl Crc32 {
    fn new() -> Crc32 {
        Crc32(0xffff_ffff)
    }

    fn update(&mut self, data: &[u8]) {
        for &b in data {
            self.0 ^= u32::from(b);
            for _ in 0..8 {
                let mask = (self.0 & 1).wrapping_neg();
                self.0 = (self.0 >> 1) ^ (0xedb8_8320 & mask);
            }
        }
    }

    fn finish(&self) -> u32 {
        !self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixman::{R5G6B5, X8R8G8B8};

    #[test]
    fn ppm_of_a_two_by_one_surface() {
        let mut img = Image::new(X8R8G8B8, 2, 1, 0);
        img.set_pixel(0, 0, 0x00ff_0000);
        img.set_pixel(1, 0, 0x0000_ff00);
        let ppm = ppm_bytes(&img);
        let header = b"P6\n2 1\n255\n";
        assert_eq!(&ppm[..header.len()], header);
        // Six bytes of pixels and two of pixman stride padding.
        assert_eq!(&ppm[header.len()..], &[0xff, 0, 0, 0, 0xff, 0, 0, 0]);
    }

    #[test]
    fn ppm_widens_565_like_pixman() {
        let mut img = Image::new(R5G6B5, 4, 1, 0);
        img.set_pixel(0, 0, 0xffff);
        img.set_pixel(1, 0, 0x0841);
        let ppm = ppm_bytes(&img);
        let px = &ppm[b"P6\n4 1\n255\n".len()..];
        // 0x0841 has 1 in each 5 bit channel and 2 in the 6 bit one, all of which widen to 8.
        assert_eq!(&px[..6], &[0xff, 0xff, 0xff, 0x08, 0x08, 0x08]);
    }

    #[test]
    fn crc_of_iend() {
        let mut c = Crc32::new();
        c.update(b"IEND");
        assert_eq!(c.finish(), 0xae42_6082);
    }

    #[test]
    fn png_round_trips_through_zlib() {
        let mut img = Image::new(X8R8G8B8, 3, 2, 0);
        img.set_pixel(2, 1, 0x0012_3456);
        let mut v = Vec::new();
        png_save(&mut v, &img).unwrap();
        assert_eq!(&v[1..4], b"PNG");
        // IHDR: width 3, height 2, depth 8, colour type 2.
        assert_eq!(&v[16..26], &[0, 0, 0, 3, 0, 0, 0, 2, 8, 2]);
        let idat_len = u32::from_be_bytes([v[33], v[34], v[35], v[36]]) as usize;
        assert_eq!(&v[37..41], b"IDAT");
        let mut d = flate2::read::ZlibDecoder::new(&v[41..41 + idat_len]);
        let mut raw = Vec::new();
        std::io::Read::read_to_end(&mut d, &mut raw).unwrap();
        assert_eq!(raw.len(), 2 * (1 + 9));
        assert_eq!(&raw[10 + 1 + 6..], &[0x12, 0x34, 0x56]);
    }

    #[test]
    fn screendump_errors_and_writes() {
        let dir = std::env::temp_dir().join(format!("ruvm-screendump-{}", std::process::id()));
        let missing = dir.join("missing").join("x.ppm");
        let missing = missing.to_str().unwrap();
        let ds = DisplayState::new();
        let e = screendump(&ds, missing, None, Some(1), None).unwrap_err();
        assert_eq!(e.message(), "'head' must be specified together with 'device'");
        let e = screendump(&ds, missing, None, None, None).unwrap_err();
        assert_eq!(e.message(), "There is no console to take a screendump from");
        let con = ds.graphic_console_create(None, 0, std::sync::Arc::new(NoHw));
        con.resize(4, 2);
        let e = screendump(&ds, missing, None, None, None).unwrap_err();
        assert_eq!(e.message(), format!("Could not create '{missing}': No such file or directory"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("x.ppm");
        screendump(&ds, path.to_str().unwrap(), None, None, Some(ImageFormat::Ppm)).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(bytes.len(), b"P6\n4 2\n255\n".len() + 2 * 12);
    }

    struct NoHw;
    impl crate::console::GraphicHwOps for NoHw {}
}
