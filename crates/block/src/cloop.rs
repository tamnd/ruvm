// SPDX-License-Identifier: GPL-2.0-or-later

//! The `cloop` format driver from block/cloop.c: Linux compressed loop images (V2.0), read only.
//!
//! The image starts with a 128 byte shell script, then the big endian block size and block
//! count, then `n_blocks + 1` big endian 64-bit offsets. Block `i` is the zlib stream between
//! `offsets[i]` and `offsets[i + 1]` and inflates to exactly one block. The last inflated block
//! is cached.
//!
//! Differences from QEMU: none in behavior. The "Could not allocate ..." errors of QEMU cannot
//! happen here, an allocation failure aborts as everywhere in Rust.

use std::io;
use std::sync::Mutex;

use flate2::{Decompress, FlushDecompress, Status};
use ruvm_base::{Error, Result};
use ruvm_qapi::types::BlockdevOptionsU;

use crate::drivers::{DriverDef, OpenArgs};
use crate::node::{BDRV_CHILD_IMAGE, BDRV_CHILD_PRIMARY, BlockLimits, Driver, Node, errno};

/// Maximum compressed block size
const MAX_BLOCK_SIZE: u32 = 64 * 1024 * 1024;

const MAGIC_VERSION_2_0: &[u8] = b"#!/bin/sh\n\
#V2.0 Format\n\
modprobe cloop file=$0 && mount -r -t iso9660 /dev/cloop $1\n";

/// `bdrv_cloop`.
pub(crate) static CLOOP: DriverDef = DriverDef::format("cloop", open).with_probe(cloop_probe);

/// `cloop_probe()`: compares as much of the script as the buffer has.
fn cloop_probe(buf: &[u8], _filename: Option<&str>) -> i32 {
    let length = MAGIC_VERSION_2_0.len().min(buf.len());
    if MAGIC_VERSION_2_0[..length] == buf[..length] { 2 } else { 0 }
}

fn open_err(file: &Node, e: io::Error) -> Error {
    let name = file.filename().unwrap_or_default();
    Error::from_io(format!("Could not open '{name}'"), e)
}

/// The block cache of `BDRVCloopState`.
struct Cache {
    current_block: u32,
    compressed_block: Vec<u8>,
    uncompressed_block: Vec<u8>,
    zstream: Decompress,
}

/// `BDRVCloopState`.
struct Cloop {
    block_size: u32,
    n_blocks: u32,
    offsets: Vec<u64>,
    sectors_per_block: u32,
    cache: Mutex<Cache>,
}

/// `cloop_open()`.
fn open(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::Cloop(o) = opts else { unreachable!("cloop driver with other options") };

    args.apply_auto_read_only(None)?;
    let file = args.open_child(*o.file, "file", BDRV_CHILD_IMAGE | BDRV_CHILD_PRIMARY)?;

    // read header
    let mut b4 = [0u8; 4];
    file.pread(128, &mut b4).map_err(|e| open_err(&file, e))?;
    let block_size = u32::from_be_bytes(b4);
    if block_size % 512 != 0 {
        return Err(Error::generic(format!("block_size {block_size} must be a multiple of 512")));
    }
    if block_size == 0 {
        return Err(Error::generic("block_size cannot be zero"));
    }

    // cloop's create_compressed_fs.c warns about block sizes beyond 256 KB but we can accept
    // more. Prevent ridiculous values like 4 GB - 1 since we need a buffer this big.
    if block_size > MAX_BLOCK_SIZE {
        return Err(Error::generic(format!(
            "block_size {block_size} must be {} MB or less",
            MAX_BLOCK_SIZE / (1024 * 1024)
        )));
    }

    file.pread(128 + 4, &mut b4).map_err(|e| open_err(&file, e))?;
    let n_blocks = u32::from_be_bytes(b4);

    // read offsets
    let max_blocks = (u32::MAX - 1) / 8;
    if n_blocks > max_blocks {
        // Prevent integer overflow
        return Err(Error::generic(format!("n_blocks {n_blocks} must be {max_blocks} or less")));
    }
    let offsets_size = (n_blocks + 1) * 8;
    if offsets_size > 512 * 1024 * 1024 {
        // Prevent ridiculous offsets_size which causes memory allocation to fail or overflows
        // bdrv_pread() size. In practice the 512 MB offsets[] limit supports 16 TB images at
        // 256 KB block size.
        return Err(Error::generic("image requires too many offsets, try increasing block size"));
    }

    let mut raw = vec![0u8; offsets_size as usize];
    file.pread(128 + 4 + 4, &mut raw).map_err(|e| open_err(&file, e))?;
    let offsets: Vec<u64> =
        raw.chunks_exact(8).map(|c| u64::from_be_bytes(c.try_into().unwrap())).collect();

    let mut max_compressed_block_size: u64 = 1;
    for i in 1..offsets.len() {
        if offsets[i] < offsets[i - 1] {
            return Err(Error::generic(format!(
                "offsets not monotonically increasing at index {i}, image file is corrupt"
            )));
        }

        let size = offsets[i] - offsets[i - 1];

        // Compressed blocks should be smaller than the uncompressed block size but maybe
        // compression performed poorly so the compressed block is actually bigger. Clamp down
        // on unrealistic values to prevent ridiculous compressed_block allocation.
        if size > 2 * u64::from(MAX_BLOCK_SIZE) {
            return Err(Error::generic(format!(
                "invalid compressed block size at index {i}, image file is corrupt"
            )));
        }

        max_compressed_block_size = max_compressed_block_size.max(size);
    }

    let cache = Cache {
        current_block: n_blocks,
        compressed_block: vec![0; max_compressed_block_size as usize + 1],
        uncompressed_block: vec![0; block_size as usize],
        zstream: Decompress::new(true),
    };

    Ok(Box::new(Cloop {
        block_size,
        n_blocks,
        offsets,
        sectors_per_block: block_size / 512,
        cache: Mutex::new(cache),
    }))
}

impl Cloop {
    /// `cloop_read_block()`: makes `block_num` the cached block.
    fn read_block(&self, file: &Node, c: &mut Cache, block_num: u32) -> bool {
        if c.current_block == block_num {
            return true;
        }
        let i = block_num as usize;
        let bytes = (self.offsets[i + 1] - self.offsets[i]) as usize;

        if file.pread(self.offsets[i], &mut c.compressed_block[..bytes]).is_err() {
            return false;
        }

        c.zstream.reset(true);
        let r = c.zstream.decompress(
            &c.compressed_block[..bytes],
            &mut c.uncompressed_block,
            FlushDecompress::Finish,
        );
        if !matches!(r, Ok(Status::StreamEnd))
            || c.zstream.total_out() != u64::from(self.block_size)
        {
            return false;
        }

        c.current_block = block_num;
        true
    }
}

impl Driver for Cloop {
    /// `cloop_co_preadv()`.
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        debug_assert!(offset % 512 == 0 && buf.len() % 512 == 0);
        let file = bs.file();
        let sector_num = offset / 512;
        let mut c = self.cache.lock().unwrap();

        for (i, sector) in buf.chunks_mut(512).enumerate() {
            let s = sector_num + i as u64;
            let sector_offset_in_block = (s % u64::from(self.sectors_per_block)) as usize;
            let block_num = (s / u64::from(self.sectors_per_block)) as u32;
            if !self.read_block(&file, &mut c, block_num) {
                return Err(errno(libc::EIO));
            }
            let data = &c.uncompressed_block[sector_offset_in_block * 512..][..512];
            sector.copy_from_slice(data);
        }
        Ok(())
    }

    fn pwrite(&self, _bs: &Node, _offset: u64, _buf: &[u8]) -> io::Result<()> {
        Err(errno(libc::ENOTSUP))
    }

    fn getlength(&self, _bs: &Node) -> io::Result<u64> {
        // Cast to u64 to prevent u32 overflow
        Ok(u64::from(self.n_blocks) * u64::from(self.sectors_per_block) * 512)
    }

    /// `cloop_refresh_limits()`: no sub-sector I/O.
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
        assert_eq!(cloop_probe(MAGIC_VERSION_2_0, None), 2);
        assert_eq!(cloop_probe(&MAGIC_VERSION_2_0[..10], None), 2);
        assert_eq!(cloop_probe(b"#!/bin/sh\n#V1.0", None), 0);
        let mut buf = [0u8; 512];
        buf[..MAGIC_VERSION_2_0.len()].copy_from_slice(MAGIC_VERSION_2_0);
        assert_eq!(cloop_probe(&buf, None), 2);
    }
}
