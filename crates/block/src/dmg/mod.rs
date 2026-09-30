// SPDX-License-Identifier: GPL-2.0-or-later

//! The `dmg` format driver from block/dmg.c, dmg-bz2.c and dmg-lzfse.c: Apple disk images
//! (UDIF), read only.
//!
//! A UDIF image ends with a 512 byte "koly" trailer that points at the data fork and at the
//! chunk tables, either in a resource fork or in an XML property list. Each table ("mish"
//! block) lists chunks of sectors that are stored raw (UDRW), zeroed (UDZE, UDIG) or compressed
//! with zlib (UDZO), bzip2 (UDBZ) or lzfse (ULFO). A read inflates the whole chunk the sector
//! is in and keeps the last chunk.
//!
//! QEMU builds the bzip2 and lzfse decompressors as the loadable modules dmg-bz2 and
//! dmg-lzfse, on top of libbz2 and liblzfse. Here they are decoders of this crate behind the
//! `dmg-bzip2` feature (on by default, as the module is in common QEMU builds) and the
//! `dmg-lzfse` feature (off by default, as QEMU needs the rarely installed liblzfse). Without
//! the feature, chunks of that kind are left out of the table with QEMU's warning about the
//! missing module, and reading them fails with EIO, as in QEMU without the module.
//!
//! Differences from QEMU:
//!
//! - With the `dmg-lzfse` feature, ULFO images are readable while a QEMU without liblzfse,
//!   such as Homebrew's qemu-img, only warns and fails the reads.
//! - The bzip2 decoder does not decode the "randomised" blocks that bzip2 stopped writing in
//!   1999; libbz2 still does. Such a chunk fails to read with EIO.
//! - QEMU serialises reads with a coroutine mutex; here a mutex guards the chunk cache.

#[cfg(feature = "dmg-bzip2")]
mod bz2;
#[cfg(feature = "dmg-lzfse")]
mod lzfse;

use std::io;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use flate2::{Decompress, FlushDecompress, Status};
use ruvm_base::{Error, Result, report};
use ruvm_qapi::types::BlockdevOptionsU;

use crate::drivers::{DriverDef, OpenArgs};
use crate::node::{BDRV_CHILD_IMAGE, BDRV_CHILD_PRIMARY, BlockLimits, Driver, Node, errno};

/// Limit chunk sizes to prevent unreasonable amounts of memory being used or truncating when
/// converting to 32-bit types
const DMG_LENGTHS_MAX: u64 = 64 * 1024 * 1024; // 64 MB
const DMG_SECTORCOUNTS_MAX: u64 = DMG_LENGTHS_MAX / 512;

// DMG Block Type
const UDZE: u32 = 0; // Zeroes
const UDRW: u32 = 1; // RAW type
const UDIG: u32 = 2; // Ignore
const UDZO: u32 = 0x8000_0005;
const UDBZ: u32 = 0x8000_0006;
const ULFO: u32 = 0x8000_0007;
const UDCM: u32 = 0x7fff_fffe; // Comments
const UDLE: u32 = 0xffff_ffff; // Last Entry

/// `bdrv_dmg`.
pub(crate) static DMG: DriverDef = DriverDef::format("dmg", open).with_probe(dmg_probe);

/// `dmg_probe()`: only the file name tells.
fn dmg_probe(_buf: &[u8], filename: Option<&str>) -> i32 {
    match filename {
        Some(f) if f.len() > 4 && f.as_bytes().ends_with(b".dmg") => 2,
        _ => 0,
    }
}

fn be64(buf: &[u8], off: usize) -> u64 {
    u64::from_be_bytes(buf[off..off + 8].try_into().unwrap())
}

fn be32(buf: &[u8], off: usize) -> u32 {
    u32::from_be_bytes(buf[off..off + 4].try_into().unwrap())
}

fn read_uint64(file: &Node, offset: u64) -> io::Result<u64> {
    let mut b = [0u8; 8];
    file.pread(offset, &mut b)?;
    Ok(u64::from_be_bytes(b))
}

fn read_uint32(file: &Node, offset: u64) -> io::Result<u32> {
    let mut b = [0u8; 4];
    file.pread(offset, &mut b)?;
    Ok(u32::from_be_bytes(b))
}

fn einval() -> io::Error {
    errno(libc::EINVAL)
}

/// `warn_report_once()`: `flag` is the call site's "already reported".
fn warn_once(flag: &AtomicBool, msg: &str) {
    if !flag.swap(true, Ordering::Relaxed) {
        report::warn_report(msg);
    }
}

static WARNED_BZ2: AtomicBool = AtomicBool::new(false);
static WARNED_LZFSE: AtomicBool = AtomicBool::new(false);
static WARNED_UNKNOWN: AtomicBool = AtomicBool::new(false);

/// One entry of the chunk table.
struct Chunk {
    type_: u32,
    /// Where the (compressed) data is in the file.
    offset: u64,
    /// How long the (compressed) data is.
    length: u64,
    /// The first guest sector.
    sector: u64,
    /// How many guest sectors.
    sectorcount: u64,
}

/// `DmgHeaderState`, used when building the sector table.
struct HeaderState {
    /// used internally by dmg_read_mish_block to remember offsets of blocks across calls
    data_fork_offset: u64,
    max_compressed_size: u32,
    max_sectors_per_chunk: u32,
}

/// `dmg_is_known_block_type()`.
fn is_known_block_type(entry_type: u32) -> bool {
    matches!(entry_type, UDZE | UDRW | UDIG | UDZO)
        || (entry_type == UDBZ && cfg!(feature = "dmg-bzip2"))
        || (entry_type == ULFO && cfg!(feature = "dmg-lzfse"))
}

/// `update_max_chunk_size()`: grows the buffer sizes for compressed/uncompressed chunk I/O.
fn update_max_chunk_size(c: &Chunk, ds: &mut HeaderState) {
    let (compressed_size, uncompressed_sectors) = match c.type_ {
        UDZO | UDBZ | ULFO => (c.length as u32, c.sectorcount as u32),
        UDRW => (0, c.length.div_ceil(512) as u32),
        // as the all-zeroes block may be large, it is treated specially: the sector is not
        // copied from a large buffer, a simple memset is used instead.
        _ => (0, 0),
    };
    ds.max_compressed_size = ds.max_compressed_size.max(compressed_size);
    ds.max_sectors_per_chunk = ds.max_sectors_per_chunk.max(uncompressed_sectors);
}

/// `dmg_read_mish_block()`: appends the chunks of one mish block to `chunks`.
fn read_mish_block(chunks: &mut Vec<Chunk>, ds: &mut HeaderState, buffer: &[u8]) -> io::Result<()> {
    let count = buffer.len();
    // skip data that is not a valid MISH block (invalid magic or too small)
    if count < 244 || be32(buffer, 0) != 0x6d69_7368 {
        // assume success for now
        return Ok(());
    }

    // chunk offsets are relative to this sector number
    let out_offset = be64(buffer, 8);
    // location in data fork for (compressed) blob (in bytes)
    let in_offset = ds.data_fork_offset.wrapping_add(be64(buffer, 0x18));

    // move to begin of chunk entries
    let n_entries = (count - 204) / 40;
    for e in 0..n_entries {
        let entry = &buffer[204 + 40 * e..204 + 40 * (e + 1)];
        let type_ = be32(entry, 0);
        if !is_known_block_type(type_) {
            match type_ {
                UDBZ => warn_once(
                    &WARNED_BZ2,
                    "dmg-bzip2 module is missing, accessing bzip2 compressed blocks will result \
                     in I/O errors",
                ),
                ULFO => warn_once(
                    &WARNED_LZFSE,
                    "dmg-lzfse module is missing, accessing lzfse compressed blocks will result \
                     in I/O errors",
                ),
                // Comments and last entry can be ignored without problems
                UDCM | UDLE => {}
                _ => warn_once(
                    &WARNED_UNKNOWN,
                    &format!(
                        "Image contains chunks of unknown type {type_:x}, accessing them will \
                         result in I/O errors"
                    ),
                ),
            }
            continue;
        }
        let i = chunks.len();

        // sector number and count
        let sector = be64(entry, 8).wrapping_add(out_offset);
        let sectorcount = be64(entry, 0x10);

        // all-zeroes sector (type UDZE and UDIG) does not need to be "uncompressed" and can
        // therefore be unbounded.
        if type_ != UDZE && type_ != UDIG && sectorcount > DMG_SECTORCOUNTS_MAX {
            report::error_report(&format!(
                "sector count {sectorcount} for chunk {i} is larger than max \
                 ({DMG_SECTORCOUNTS_MAX})"
            ));
            return Err(einval());
        }

        // offset and length in (compressed) data fork
        let offset = be64(entry, 0x18).wrapping_add(in_offset);
        let length = be64(entry, 0x20);
        if length > DMG_LENGTHS_MAX {
            report::error_report(&format!(
                "length {length} for chunk {i} is larger than max ({DMG_LENGTHS_MAX})"
            ));
            return Err(einval());
        }

        // Uncompressed chunk length must match sector count. Compressed chunks are validated
        // during dmg_read_chunk() since the uncompressed size is not known ahead of time.
        if type_ == UDRW && sectorcount != length.div_ceil(512) {
            report::error_report(&format!(
                "length {length} for chunk {i} is inconsistent with sector count {sectorcount}"
            ));
            return Err(einval());
        }

        let c = Chunk { type_, offset, length, sector, sectorcount };
        update_max_chunk_size(&c, ds);
        chunks.push(c);
    }
    Ok(())
}

/// `dmg_find_koly_offset()`: the koly magic is in the last 511 bytes of the second last
/// sector or the first 4 bytes of the last sector, since the image size need not be a
/// multiple of 512.
fn find_koly_offset(file: &Node) -> Result<u64> {
    let length = file
        .getlength()
        .map_err(|e| Error::from_io("Failed to get file size while reading UDIF trailer", e))?;
    if length < 512 {
        return Err(Error::generic("dmg file must be at least 512 bytes long"));
    }
    let offset = if length > 511 + 512 { length - 511 - 512 } else { 0 };
    let n = length.min(515) as usize;
    let mut buffer = [0u8; 515];
    file.pread(offset, &mut buffer[..n])
        .map_err(|e| Error::from_io("Failed while reading UDIF trailer", e))?;
    match buffer[..n].windows(4).position(|w| w == b"koly") {
        Some(i) => Ok(offset + i as u64),
        None => Err(Error::generic("Could not locate UDIF trailer in dmg file")),
    }
}

/// `dmg_read_resource_fork()`.
fn read_resource_fork(
    file: &Node,
    chunks: &mut Vec<Chunk>,
    ds: &mut HeaderState,
    info_begin: u64,
    info_length: u64,
) -> io::Result<()> {
    // read offset from begin of resource fork (info_begin) to resource data
    let rsrc_data_offset = read_uint32(file, info_begin)?;
    if u64::from(rsrc_data_offset) > info_length {
        return Err(einval());
    }

    // read length of resource data
    let count = read_uint32(file, info_begin + 8)?;
    if count == 0 || u64::from(rsrc_data_offset.wrapping_add(count)) > info_length {
        return Err(einval());
    }

    // begin of resource data (consisting of one or more resources)
    let mut offset = info_begin + u64::from(rsrc_data_offset);

    // end of resource data (there is possibly a following resource map which will be
    // ignored).
    let info_end = offset + u64::from(count);

    // read offsets (mish blocks) from one or more resources in resource data
    let mut buffer = Vec::new();
    while offset < info_end {
        // size of following resource
        let count = read_uint32(file, offset)?;
        if count == 0 || u64::from(count) > info_end - offset {
            return Err(einval());
        }
        offset += 4;

        buffer.resize(count as usize, 0);
        file.pread(offset, &mut buffer)?;
        read_mish_block(chunks, ds, &buffer)?;
        // advance offset by size of resource
        offset += u64::from(count);
    }
    Ok(())
}

/// `g_base64_decode()`: skips characters outside the alphabet and drops the bytes that
/// trailing padding stands for.
fn base64_decode(s: &[u8]) -> Vec<u8> {
    fn rank(c: u8) -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => 0,
            _ => return None,
        } as u32)
    }
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut v, mut i) = (0u32, 0);
    let mut last = [0u8; 2];
    for &c in s {
        let Some(r) = rank(c) else { continue };
        last[1] = last[0];
        last[0] = c;
        v = (v << 6) | r;
        i += 1;
        if i == 4 {
            out.push((v >> 16) as u8);
            if last[1] != b'=' {
                out.push((v >> 8) as u8);
            }
            if last[0] != b'=' {
                out.push(v as u8);
            }
            i = 0;
        }
    }
    out
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// `dmg_read_plist_xml()`: the mish blocks are base64 in the `<data>` elements.
fn read_plist_xml(
    file: &Node,
    chunks: &mut Vec<Chunk>,
    ds: &mut HeaderState,
    info_begin: u64,
    info_length: u64,
) -> io::Result<()> {
    // Attempt to set a safe upper cap on the data length. A test sample had a XML length of
    // about 1 MiB.
    if info_length == 0 || info_length > 16 * 1024 * 1024 {
        return Err(einval());
    }
    let mut buffer = vec![0u8; info_length as usize];
    file.pread(info_begin, &mut buffer).map_err(|_| einval())?;
    // The C code works on it as a string.
    if let Some(nul) = buffer.iter().position(|&b| b == 0) {
        buffer.truncate(nul);
    }

    // look for <data>...</data>.
    let mut pos = 0;
    while let Some(b) = find(&buffer[pos..], b"<data>") {
        let data_begin = pos + b + 6;
        // malformed XML?
        let Some(e) = find(&buffer[data_begin..], b"</data>") else {
            return Err(einval());
        };
        let data_end = data_begin + e;
        let mish = base64_decode(&buffer[data_begin..data_end]);
        // The C code passes the length as a 32-bit count.
        let n = mish.len().min(u32::MAX as usize);
        read_mish_block(chunks, ds, &mish[..n])?;
        pos = data_end + 1;
    }
    Ok(())
}

/// The chunk cache of `BDRVDMGState`.
struct Cache {
    current_chunk: usize,
    compressed_chunk: Vec<u8>,
    uncompressed_chunk: Vec<u8>,
    zstream: Decompress,
}

/// `BDRVDMGState`.
struct Dmg {
    /// Ordered by sector.
    chunks: Vec<Chunk>,
    total_sectors: u64,
    cache: Mutex<Cache>,
}

/// `dmg_open()`.
fn open(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::Dmg(o) = opts else { unreachable!("dmg driver with other options") };

    args.apply_auto_read_only(None)?;
    let file = args.open_child(*o.file, "file", BDRV_CHILD_IMAGE | BDRV_CHILD_PRIMARY)?;
    let fail = |e: io::Error| {
        let name = file.filename().unwrap_or_default();
        Error::from_io(format!("Could not open '{name}'"), e)
    };

    let mut chunks = Vec::new();
    let mut ds =
        HeaderState { data_fork_offset: 0, max_compressed_size: 1, max_sectors_per_chunk: 1 };

    // locate the UDIF trailer
    let offset = find_koly_offset(&file)?;

    // offset of data fork (DataForkOffset)
    ds.data_fork_offset = read_uint64(&file, offset + 0x18).map_err(fail)?;
    if ds.data_fork_offset > offset {
        return Err(fail(einval()));
    }

    // offset of resource fork (RsrcForkOffset)
    let rsrc_fork_offset = read_uint64(&file, offset + 0x28).map_err(fail)?;
    let rsrc_fork_length = read_uint64(&file, offset + 0x30).map_err(fail)?;
    if rsrc_fork_offset >= offset || rsrc_fork_length > offset - rsrc_fork_offset {
        return Err(fail(einval()));
    }
    // offset of property list (XMLOffset)
    let plist_xml_offset = read_uint64(&file, offset + 0xd8).map_err(fail)?;
    let plist_xml_length = read_uint64(&file, offset + 0xe0).map_err(fail)?;
    if plist_xml_offset >= offset || plist_xml_length > offset - plist_xml_offset {
        return Err(fail(einval()));
    }
    let total_sectors = read_uint64(&file, offset + 0x1ec).map_err(fail)?;
    if total_sectors > i64::MAX as u64 {
        return Err(fail(einval()));
    }
    if rsrc_fork_length != 0 {
        read_resource_fork(&file, &mut chunks, &mut ds, rsrc_fork_offset, rsrc_fork_length)
            .map_err(fail)?;
    } else if plist_xml_length != 0 {
        read_plist_xml(&file, &mut chunks, &mut ds, plist_xml_offset, plist_xml_length)
            .map_err(fail)?;
    } else {
        return Err(fail(einval()));
    }

    // There must be at least one chunk
    if chunks.is_empty() {
        return Err(fail(einval()));
    }

    let cache = Cache {
        current_chunk: chunks.len(),
        compressed_chunk: vec![0; ds.max_compressed_size as usize + 1],
        uncompressed_chunk: vec![0; 512 * ds.max_sectors_per_chunk as usize],
        zstream: Decompress::new(true),
    };
    Ok(Box::new(Dmg { chunks, total_sectors, cache: Mutex::new(cache) }))
}

impl Dmg {
    /// `is_sector_in_chunk()`.
    fn is_sector_in_chunk(&self, chunk_num: usize, sector_num: u64) -> bool {
        match self.chunks.get(chunk_num) {
            Some(c) => c.sector <= sector_num && c.sector.wrapping_add(c.sectorcount) > sector_num,
            None => false,
        }
    }

    /// `search_chunk()`: a binary search, which assumes the table is ordered; the chunk
    /// count when the sector is in none.
    fn search_chunk(&self, sector_num: u64) -> usize {
        let n = self.chunks.len();
        let (mut chunk1, mut chunk2) = (0usize, n - 1);
        while chunk1 <= chunk2 {
            let chunk3 = (chunk1 + chunk2) / 2;
            let c = &self.chunks[chunk3];
            if c.sector > sector_num {
                if chunk3 == 0 {
                    return n;
                }
                chunk2 = chunk3 - 1;
            } else if c.sector.wrapping_add(c.sectorcount) > sector_num {
                return chunk3;
            } else {
                chunk1 = chunk3 + 1;
            }
        }
        n // error
    }

    /// `dmg_read_chunk()`: makes the chunk with `sector_num` the cached one.
    fn read_chunk(&self, file: &Node, s: &mut Cache, sector_num: u64) -> bool {
        if self.is_sector_in_chunk(s.current_chunk, sector_num) {
            return true;
        }
        let chunk = self.search_chunk(sector_num);
        if chunk >= self.chunks.len() {
            return false;
        }
        let c = &self.chunks[chunk];

        s.current_chunk = self.chunks.len();
        let len = c.length as usize;
        let out_len = 512 * c.sectorcount as usize;
        match c.type_ {
            // zlib compressed
            UDZO => {
                // we need to buffer, because only the chunk as whole can be inflated.
                if file.pread(c.offset, &mut s.compressed_chunk[..len]).is_err() {
                    return false;
                }
                s.zstream.reset(true);
                let r = s.zstream.decompress(
                    &s.compressed_chunk[..len],
                    &mut s.uncompressed_chunk[..out_len],
                    FlushDecompress::Finish,
                );
                if !matches!(r, Ok(Status::StreamEnd)) || s.zstream.total_out() != out_len as u64 {
                    return false;
                }
            }
            // bzip2 compressed
            #[cfg(feature = "dmg-bzip2")]
            UDBZ => {
                if file.pread(c.offset, &mut s.compressed_chunk[..len]).is_err() {
                    return false;
                }
                let (input, output) = (&s.compressed_chunk[..len], &mut s.uncompressed_chunk);
                if bz2::uncompress(input, &mut output[..out_len]).is_err() {
                    return false;
                }
            }
            #[cfg(feature = "dmg-lzfse")]
            ULFO => {
                if file.pread(c.offset, &mut s.compressed_chunk[..len]).is_err() {
                    return false;
                }
                let (input, output) = (&s.compressed_chunk[..len], &mut s.uncompressed_chunk);
                if lzfse::uncompress(input, &mut output[..out_len]).is_none() {
                    return false;
                }
            }
            // copy
            UDRW => {
                if file.pread(c.offset, &mut s.uncompressed_chunk[..len]).is_err() {
                    return false;
                }
                // Zero the unread part of the last sector when chunk length is unaligned to
                // avoid exposing uninitialized memory.
                if len % 512 != 0 {
                    s.uncompressed_chunk[len..len.next_multiple_of(512)].fill(0);
                }
            }
            // zeros and ignore: see pread, no buffer needs to be pre-filled
            _ => {}
        }
        s.current_chunk = chunk;
        true
    }
}

impl Driver for Dmg {
    /// `dmg_co_preadv()`.
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        debug_assert!(offset % 512 == 0 && buf.len() % 512 == 0);
        let file = bs.file();
        let sector_num = offset / 512;
        let mut s = self.cache.lock().unwrap();

        for (i, sector) in buf.chunks_mut(512).enumerate() {
            let sn = sector_num + i as u64;
            if !self.read_chunk(&file, &mut s, sn) {
                return Err(errno(libc::EIO));
            }
            // Special case: current chunk is all zeroes. Do not perform a memcpy as the
            // uncompressed chunk buffer may be too small to cover the large all-zeroes
            // section.
            let c = &self.chunks[s.current_chunk];
            if c.type_ == UDZE || c.type_ == UDIG {
                sector.fill(0);
                continue;
            }
            let sector_offset_in_chunk = (sn - c.sector) as usize;
            sector.copy_from_slice(&s.uncompressed_chunk[sector_offset_in_chunk * 512..][..512]);
        }
        Ok(())
    }

    fn pwrite(&self, _bs: &Node, _offset: u64, _buf: &[u8]) -> io::Result<()> {
        Err(errno(libc::ENOTSUP))
    }

    /// `bdrv_co_getlength()` on the size from the trailer.
    fn getlength(&self, _bs: &Node) -> io::Result<u64> {
        if self.total_sectors > i64::MAX as u64 / 512 {
            return Err(errno(libc::EFBIG));
        }
        Ok(self.total_sectors * 512)
    }

    /// `dmg_refresh_limits()`: no sub-sector I/O.
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
        assert_eq!(dmg_probe(&[], Some("a.dmg")), 2);
        assert_eq!(dmg_probe(&[], Some(".dmg")), 0);
        assert_eq!(dmg_probe(&[], Some("a.DMG")), 0);
        assert_eq!(dmg_probe(&[], Some("a.img")), 0);
        assert_eq!(dmg_probe(&[], None), 0);
    }

    #[test]
    fn base64() {
        assert_eq!(base64_decode(b"aGVs\n\tbG8="), b"hello");
        assert_eq!(base64_decode(b"aGk="), b"hi");
        assert_eq!(base64_decode(b"aA=="), b"h");
        // An unfinished group is dropped, as glib does.
        assert_eq!(base64_decode(b"aGVsbG"), b"hel");
    }

    fn mish(entries: &[(u32, u64, u64, u64, u64)]) -> Vec<u8> {
        let mut b = vec![0u8; 204];
        b[..4].copy_from_slice(b"mish");
        b[8..16].copy_from_slice(&100u64.to_be_bytes());
        for &(t, sector, count, off, len) in entries {
            let mut e = [0u8; 40];
            e[..4].copy_from_slice(&t.to_be_bytes());
            e[8..16].copy_from_slice(&sector.to_be_bytes());
            e[16..24].copy_from_slice(&count.to_be_bytes());
            e[24..32].copy_from_slice(&off.to_be_bytes());
            e[32..40].copy_from_slice(&len.to_be_bytes());
            b.extend_from_slice(&e);
        }
        b
    }

    #[test]
    fn mish_block() {
        let mut ds = HeaderState {
            data_fork_offset: 1000,
            max_compressed_size: 1,
            max_sectors_per_chunk: 1,
        };
        let mut chunks = Vec::new();
        let b = mish(&[
            (UDRW, 0, 2, 0, 1000),
            (UDCM, 0, 0, 0, 0),
            (UDZO, 2, 8, 1000, 300),
            (UDZE, 10, 1 << 40, 0, 0),
            (UDLE, 0, 0, 0, 0),
        ]);
        read_mish_block(&mut chunks, &mut ds, &b).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].sector, 100);
        assert_eq!(chunks[1].offset, 2000);
        assert_eq!(chunks[2].sectorcount, 1 << 40);
        assert_eq!(ds.max_compressed_size, 300);
        assert_eq!(ds.max_sectors_per_chunk, 8);

        // Too short or not a mish block: ignored.
        read_mish_block(&mut chunks, &mut ds, &b[..243]).unwrap();
        assert_eq!(chunks.len(), 3);

        // Inconsistent raw chunks and oversized chunks fail.
        let bad = mish(&[(UDRW, 0, 3, 0, 1000)]);
        assert!(read_mish_block(&mut chunks, &mut ds, &bad).is_err());
        let bad = mish(&[(UDZO, 0, DMG_SECTORCOUNTS_MAX + 1, 0, 10)]);
        assert!(read_mish_block(&mut chunks, &mut ds, &bad).is_err());
        let bad = mish(&[(UDZO, 0, 1, 0, DMG_LENGTHS_MAX + 1)]);
        assert!(read_mish_block(&mut chunks, &mut ds, &bad).is_err());
    }
}
