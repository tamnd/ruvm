// SPDX-License-Identifier: GPL-2.0-or-later

//! The `bochs` format driver from block/bochs.c: growing redolog images of the Bochs emulator,
//! read only.
//!
//! The image starts with a 512 byte little endian header ("Bochs Virtual HD Image", type
//! "Redolog", subtype "Growing"), followed by a catalog of 32-bit entries, one per extent. An
//! entry of 0xffffffff means the extent was never written. Otherwise it is the index of the
//! extent in the data area, where each extent is a sector bitmap followed by the sectors
//! themselves. A sector whose bit is clear reads as zeroes.
//!
//! Differences from QEMU: none in behavior. QEMU serialises reads with a coroutine mutex; the
//! state here never changes after open, so reads run without a lock.

use std::io;

use ruvm_base::{Error, Result};
use ruvm_qapi::types::BlockdevOptionsU;

use crate::drivers::{DriverDef, OpenArgs};
use crate::node::{BDRV_CHILD_IMAGE, BDRV_CHILD_PRIMARY, BlockLimits, Driver, Node, errno};

const HEADER_MAGIC: &[u8] = b"Bochs Virtual HD Image";
const HEADER_VERSION: u32 = 0x0002_0000;
const HEADER_V1: u32 = 0x0001_0000;
const HEADER_SIZE: usize = 512;

const REDOLOG_TYPE: &[u8] = b"Redolog";
const GROWING_TYPE: &[u8] = b"Growing";

/// `bdrv_bochs`.
pub(crate) static BOCHS: DriverDef = DriverDef::format("bochs", open).with_probe(bochs_probe);

/// `strcmp(field, s) == 0` for a fixed size, possibly unterminated, header field.
fn field_is(field: &[u8], s: &[u8]) -> bool {
    field.len() > s.len() && field.starts_with(s) && field[s.len()] == 0
}

fn le32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(buf[off..off + 4].try_into().unwrap())
}

fn le64(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(buf[off..off + 8].try_into().unwrap())
}

/// Whether `buf` holds a header QEMU accepts: the three strings and one of the two versions.
fn header_ok(buf: &[u8]) -> bool {
    let version = le32(buf, 64);
    field_is(&buf[0..32], HEADER_MAGIC)
        && field_is(&buf[32..48], REDOLOG_TYPE)
        && field_is(&buf[48..64], GROWING_TYPE)
        && (version == HEADER_VERSION || version == HEADER_V1)
}

/// `bochs_probe()`.
fn bochs_probe(buf: &[u8], _filename: Option<&str>) -> i32 {
    if buf.len() < HEADER_SIZE {
        return 0;
    }
    if header_ok(buf) { 100 } else { 0 }
}

fn open_err(file: &Node, e: io::Error) -> Error {
    let name = file.filename().unwrap_or_default();
    Error::from_io(format!("Could not open '{name}'"), e)
}

/// `BDRVBochsState`.
struct Bochs {
    catalog_bitmap: Vec<u32>,
    data_offset: u32,
    bitmap_blocks: u32,
    extent_blocks: u32,
    extent_size: u32,
    total_sectors: u64,
}

/// `bochs_open()`.
fn open(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::Bochs(o) = opts else { unreachable!("bochs driver with other options") };

    // No write support yet
    args.apply_auto_read_only(None)?;
    let file = args.open_child(*o.file, "file", BDRV_CHILD_IMAGE | BDRV_CHILD_PRIMARY)?;

    let mut h = [0u8; HEADER_SIZE];
    file.pread(0, &mut h).map_err(|e| open_err(&file, e))?;
    if !header_ok(&h) {
        return Err(Error::generic("Image not in Bochs format"));
    }

    let disk = if le32(&h, 64) == HEADER_V1 { le64(&h, 84) } else { le64(&h, 88) };
    let total_sectors = disk / 512;

    let header = le32(&h, 68);
    let catalog_size = le32(&h, 72);
    if catalog_size > 0x10_0000 {
        return Err(Error::generic("Catalog size is too large"));
    }

    let mut raw = vec![0u8; catalog_size as usize * 4];
    file.pread(u64::from(header), &mut raw).map_err(|e| open_err(&file, e))?;
    let catalog_bitmap: Vec<u32> =
        raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();

    let data_offset = header.wrapping_add(catalog_size * 4);
    let bitmap_blocks = 1u32.wrapping_add(le32(&h, 76).wrapping_sub(1) / 512);
    let extent_blocks = 1u32.wrapping_add(le32(&h, 80).wrapping_sub(1) / 512);

    let extent_size = le32(&h, 80);
    if extent_size < 512 {
        // the bochs code never writes images like this
        return Err(Error::generic("Extent size must be at least 512"));
    } else if !extent_size.is_power_of_two() {
        return Err(Error::generic(format!("Extent size {extent_size} is not a power of two")));
    } else if extent_size > 0x80_0000 {
        return Err(Error::generic(format!("Extent size {extent_size} is too large")));
    }

    if u64::from(catalog_size) < total_sectors.div_ceil(u64::from(extent_size / 512)) {
        return Err(Error::generic("Catalog size is too small for this disk size"));
    }

    Ok(Box::new(Bochs {
        catalog_bitmap,
        data_offset,
        bitmap_blocks,
        extent_blocks,
        extent_size,
        total_sectors,
    }))
}

impl Bochs {
    /// `seek_to_sector()`: where sector `sector_num` is in the file, or 0 when it reads as
    /// zeroes.
    fn seek_to_sector(&self, file: &Node, sector_num: u64) -> io::Result<u64> {
        let offset = sector_num * 512;
        let extent_index = offset / u64::from(self.extent_size);
        let extent_offset = (offset % u64::from(self.extent_size)) / 512;

        let entry = self.catalog_bitmap[extent_index as usize];
        if entry == 0xffff_ffff {
            return Ok(0); // not allocated
        }

        let bitmap_offset = u64::from(self.data_offset)
            + 512
                * u64::from(entry)
                * u64::from(self.extent_blocks.wrapping_add(self.bitmap_blocks));

        // read in bitmap for current extent
        let mut bitmap_entry = [0u8; 1];
        file.pread(bitmap_offset + extent_offset / 8, &mut bitmap_entry)?;
        if (bitmap_entry[0] >> (extent_offset % 8)) & 1 == 0 {
            return Ok(0); // not allocated
        }

        Ok(bitmap_offset + 512 * (u64::from(self.bitmap_blocks) + extent_offset))
    }
}

impl Driver for Bochs {
    /// `bochs_co_preadv()`.
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        debug_assert!(offset % 512 == 0 && buf.len() % 512 == 0);
        let file = bs.file();
        let sector_num = offset / 512;
        for (i, sector) in buf.chunks_mut(512).enumerate() {
            let block_offset = self.seek_to_sector(&file, sector_num + i as u64)?;
            if block_offset > 0 {
                file.pread(block_offset, sector)?;
            } else {
                sector.fill(0);
            }
        }
        Ok(())
    }

    fn pwrite(&self, _bs: &Node, _offset: u64, _buf: &[u8]) -> io::Result<()> {
        Err(errno(libc::ENOTSUP))
    }

    /// `bdrv_co_getlength()` on the size from the header.
    fn getlength(&self, _bs: &Node) -> io::Result<u64> {
        if self.total_sectors > i64::MAX as u64 / 512 {
            return Err(errno(libc::EFBIG));
        }
        Ok(self.total_sectors * 512)
    }

    /// `bochs_refresh_limits()`: no sub-sector I/O.
    fn refresh_limits(&self, _bs: &Node, bl: &mut BlockLimits) -> Result<()> {
        bl.request_alignment = 512;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe() {
        let mut h = [0u8; 512];
        h[..22].copy_from_slice(HEADER_MAGIC);
        h[32..39].copy_from_slice(REDOLOG_TYPE);
        h[48..55].copy_from_slice(GROWING_TYPE);
        h[64..68].copy_from_slice(&HEADER_V1.to_le_bytes());
        assert_eq!(bochs_probe(&h, None), 100);
        h[64..68].copy_from_slice(&HEADER_VERSION.to_le_bytes());
        assert_eq!(bochs_probe(&h, None), 100);
        assert_eq!(bochs_probe(&h[..511], None), 0);
        h[55] = b'X';
        assert_eq!(bochs_probe(&h, None), 0);
    }
}
