// SPDX-License-Identifier: GPL-2.0-or-later

//! The `parallels` format driver from block/parallels.c and block/parallels.h: Parallels and
//! OpenVZ ploop disk images.
//!
//! An image starts with a 64 byte header followed by the block allocation table (BAT), one
//! little-endian 32-bit entry per cluster of the disk. An entry of 0 is an unallocated
//! cluster; any other value, times `off_multiplier`, is the sector in the file where the
//! cluster is. Images with the "WithoutFreeSpace" magic count BAT entries in sectors and may
//! leave `data_off` zero; images with the "WithouFreSpacExt" magic (the ones QEMU creates)
//! count them in clusters. Clusters are `tracks` sectors large. The header's `inuse` field is
//! set while an image is open read-write, and an image found with it set was not closed
//! correctly and is repaired when opened read-write.
//!
//! New clusters come from holes in the file first (the `used_bmap` of clusters BAT entries
//! point at) and otherwise from its end, which grows by the runtime option `prealloc-size`
//! (128 MiB by default) at a time, either by writing zeroes (`prealloc-mode=falloc`) or by
//! truncating (`prealloc-mode=truncate`). Closing the image cuts the file back to the end of
//! the last cluster. Parallels images have no zero flag in their BAT: writing zeroes to whole
//! clusters discards them, and without a backing file an unallocated cluster reads as zeroes.
//!
//! The optional Format Extension (see [`ext`]) is only read when the image is opened
//! read-only, as in QEMU; read-write opens warn and ignore it.
//!
//! Differences from QEMU:
//!
//! - The runtime options `prealloc-mode` and `prealloc-size` cannot be given: the typed
//!   `BlockdevOptions` of `parallels` (`BlockdevOptionsGenericFormat`) have only `file`, and
//!   QEMU only accepts them from `-drive` style option dictionaries. They are parsed from the
//!   node options exactly as QEMU parses them, which leaves their defaults (`falloc`, 128M).
//! - There is no `BDRV_O_CHECK` open flag here, so an image that needs repair and is opened
//!   read-write for `qemu-img check -r` is repaired while it is opened, and the check that
//!   follows sees the repaired image. Read-only opens are never repaired, as in QEMU.
//! - The dirty bitmaps of the Format Extension are read, checked and loaded as QEMU does, but
//!   kept in the driver ([`ParallelsDriver::bitmaps`]) rather than registered as read-only
//!   dirty bitmaps of the node, since the node has no dirty bitmap list yet. An invalid
//!   bitmap granularity, which QEMU asserts on, is an error here.
//! - QEMU writes the header with at least `bdrv_opt_mem_align()` bytes and tracks dirty BAT
//!   parts in blocks of four host pages. Both only change how the same bytes are written.
//! - The BAT changes of writes are written when the node is flushed, as in QEMU, and also
//!   when it is closed, since closing a node here does not flush it first.
//! - QEMU registers a migration blocker for every open parallels node; there is no migration
//!   here.
//! - After repairing an image while opening it, QEMU does not refill its map of the used
//!   clusters, so the next writes can be given clusters that are in use and overwrite data
//!   (seen with qemu-io 11.1.2). The map is refilled here.
//! - Without `BDRV_FIX_ERRORS`, a BAT entry pointing outside the image makes QEMU's check
//!   abort on an assertion in `parallels_check_duplicate()`. Such entries are skipped by the
//!   duplicate check here, so `qemu-img check` reports them as it reports every other error.
//! - When a check fails part way, QEMU keeps the partial counts. Here the error is returned
//!   (its message is the `strerror()` text) and the counts are lost.

mod ext;

use std::io;
use std::sync::{Mutex, MutexGuard};

use ruvm_base::{Error, Result, report};
use ruvm_qapi::QDict;
use ruvm_qapi::opts::QemuOptDesc;
use ruvm_qapi::types::{
    BlockdevCreateOptionsParallels, BlockdevCreateOptionsU, BlockdevOptionsU, BlockdevRef,
    PreallocMode,
};

use crate::drivers::{DriverDef, OpenArgs};
use crate::graph::BlockGraph;
use crate::imgopts::{create_opts_open, opt_desc, take_size, take_str, visit_create_options};
use crate::node::{
    BDRV_BLOCK_DATA, BDRV_BLOCK_OFFSET_VALID, BDRV_CHILD_IMAGE, BDRV_CHILD_PRIMARY,
    BDRV_FIX_ERRORS, BDRV_FIX_LEAKS, BDRV_REQ_ZERO_WRITE, BDRV_SECTOR_SIZE, BlockLimits,
    BlockStatus, CheckResult, Driver, Node, errno, is_enotsup,
};

pub(crate) use ext::LoadedBitmap;

/// `bdrv_parallels`.
pub(crate) static PARALLELS: DriverDef = DriverDef::format("parallels", parallels_open_node)
    .with_probe(parallels_probe)
    .with_create_opts(parallels_co_create_opts)
    .with_create_opts_list(&PARALLELS_CREATE_OPTS)
    .with_create(parallels_co_create)
    .with_backing();

/// `parallels_create_opts`, in QEMU's order.
static PARALLELS_CREATE_OPTS: [QemuOptDesc; 2] = [
    opt_desc!("size", Size, "Virtual disk size"),
    opt_desc!("cluster_size", Size, "Parallels image cluster size", "1048576"),
];

const HEADER_MAGIC: &[u8; 16] = b"WithoutFreeSpace";
const HEADER_MAGIC2: &[u8; 16] = b"WithouFreSpacExt";
const HEADER_VERSION: u32 = 2;
const HEADER_INUSE_MAGIC: u32 = 0x746F_6E59;
const MAX_PARALLELS_IMAGE_FACTOR: u64 = 1 << 32;
const PARALLELS_HEADER_READ_CHUNK: u32 = 64 * 1024 * 1024;

const HEADS_NUMBER: u32 = 16;
const SEC_IN_CYL: u64 = 32;
/// 1 MiB.
const DEFAULT_CLUSTER_SIZE: u64 = 1_048_576;

const BDRV_SECTOR_BITS: u32 = 9;

/// `sizeof(ParallelsHeader)`. The BAT follows it.
const HEADER_SIZE: usize = 64;

/// Offsets into `ParallelsHeader`, all fields little-endian.
mod hdr {
    pub(super) const MAGIC: usize = 0;
    pub(super) const VERSION: usize = 16;
    pub(super) const HEADS: usize = 20;
    pub(super) const CYLINDERS: usize = 24;
    pub(super) const TRACKS: usize = 28;
    pub(super) const BAT_ENTRIES: usize = 32;
    pub(super) const NB_SECTORS: usize = 36;
    pub(super) const INUSE: usize = 44;
    pub(super) const DATA_OFF: usize = 48;
    pub(super) const EXT_OFF: usize = 56;
}

fn le32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().expect("4 bytes"))
}

fn le64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().expect("8 bytes"))
}

fn put_le32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_le64(b: &mut [u8], off: usize, v: u64) {
    b[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

/// `bat_entry_off()`: where BAT entry `idx` is in the file.
fn bat_entry_off(idx: u32) -> u32 {
    (HEADER_SIZE as u32).wrapping_add(4u32.wrapping_mul(idx))
}

/// `ParallelsPreallocMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreallocModeP {
    Fallocate,
    Truncate,
}

/// `parallels_opts_prealloc()`: the runtime options `prealloc-size` (in sectors) and
/// `prealloc-mode`, with their defaults.
fn parallels_opts_prealloc(options: &mut QDict) -> Result<(u64, PreallocModeP)> {
    let bytes = take_size(options, "prealloc-size")?.unwrap_or(128 << 20);
    let buf = take_str(options, "prealloc-mode").unwrap_or_else(|| "falloc".to_string());
    let mode = match buf.as_str() {
        "falloc" => PreallocModeP::Fallocate,
        "truncate" => PreallocModeP::Truncate,
        _ => return Err(Error::generic(format!("invalid parameter value: {buf}"))),
    };
    Ok((bytes >> BDRV_SECTOR_BITS, mode))
}

/// A bitmap of `len` bits, what `bitmap_new()` and friends work on.
#[derive(Debug, Default, Clone)]
struct Bitmap {
    words: Vec<u64>,
    len: u64,
}

impl Bitmap {
    fn new(len: u64) -> Self {
        Bitmap { words: vec![0; len.div_ceil(64) as usize], len }
    }

    #[cfg(test)]
    fn get(&self, i: u64) -> bool {
        self.words[(i / 64) as usize] & (1 << (i % 64)) != 0
    }

    fn set(&mut self, i: u64) {
        self.words[(i / 64) as usize] |= 1 << (i % 64);
    }

    fn clear(&mut self, i: u64) {
        self.words[(i / 64) as usize] &= !(1 << (i % 64));
    }

    /// `find_next_bit()`: the first set bit at or after `start`, `len` if none.
    fn find_next(&self, start: u64) -> u64 {
        self.find(start, false)
    }

    /// `find_next_zero_bit()`.
    fn find_next_zero(&self, start: u64) -> u64 {
        self.find(start, true)
    }

    fn find(&self, start: u64, zero: bool) -> u64 {
        let mut i = start;
        while i < self.len {
            let w = self.words[(i / 64) as usize];
            let w = if zero { !w } else { w } >> (i % 64);
            if w != 0 {
                return (i + u64::from(w.trailing_zeros())).min(self.len);
            }
            i = (i / 64 + 1) * 64;
        }
        self.len
    }

    /// `bitmap_zero_extend()`.
    fn zero_extend(&mut self, new_len: u64) {
        // Bits past the old end are always clear.
        self.words.resize(new_len.div_ceil(64) as usize, 0);
        self.len = new_len;
    }

    fn zero(&mut self) {
        self.words.iter_mut().for_each(|w| *w = 0);
    }
}

/// The parts of `BDRVParallelsState` that do not change after the open.
#[derive(Debug)]
struct Consts {
    /// `bs->total_sectors`.
    total_sectors: u64,
    bat_size: u32,
    tracks: u32,
    cluster_size: u32,
    off_multiplier: u32,
    /// In sectors.
    prealloc_size: u64,
    bat_dirty_block: u32,
    /// `bdrv_opt_mem_align(bs->file->bs)`.
    mem_align: u32,
    /// The magic is "WithoutFreeSpace".
    old_magic: bool,
}

/// The parts of `BDRVParallelsState` under `s->lock`.
#[derive(Debug)]
struct State {
    /// `s->header` followed by the BAT, `header_size` bytes as they are on disk.
    header: Vec<u8>,
    header_size: u32,
    header_unclean: bool,
    bat_dirty_bmap: Bitmap,
    used_bmap: Bitmap,
    /// In sectors.
    data_start: i64,
    /// In sectors.
    data_end: i64,
    prealloc_mode: PreallocModeP,
}

/// `BDRVParallelsState`.
#[derive(Debug)]
pub(crate) struct ParallelsDriver {
    c: Consts,
    lock: Mutex<State>,
    /// The dirty bitmaps of the Format Extension, loaded when the image is opened read-only.
    #[allow(dead_code, reason = "not registered with the node yet, see the module doc")]
    pub(crate) bitmaps: Vec<LoadedBitmap>,
}

/// An error that is not an errno, for `truncate` failures.
fn io_from(e: &Error) -> io::Error {
    use std::error::Error as _;
    if let Some(n) =
        e.source().and_then(|c| c.downcast_ref::<io::Error>()).and_then(|c| c.raw_os_error())
    {
        return errno(n);
    }
    if e.message() == "Block driver does not support requested flags" {
        return errno(libc::ENOTSUP);
    }
    io::Error::other(e.message().to_string())
}

/// `bdrv_pwrite_sync()`.
fn pwrite_sync(file: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
    file.pwrite(offset, buf)?;
    file.flush()
}

fn fix_word(fix: u32, mask: u32) -> &'static str {
    if fix & mask != 0 { "Repairing" } else { "ERROR" }
}

impl State {
    fn bat(&self, idx: u32) -> u32 {
        le32(&self.header, bat_entry_off(idx) as usize)
    }

    /// `bat2sect()`.
    fn bat2sect(&self, c: &Consts, idx: u32) -> i64 {
        (u64::from(self.bat(idx)) * u64::from(c.off_multiplier)) as i64
    }

    /// `parallels_set_bat_entry()`.
    fn set_bat_entry(&mut self, c: &Consts, index: u32, offset: u32) {
        put_le32(&mut self.header, bat_entry_off(index) as usize, offset);
        self.bat_dirty_bmap.set(u64::from(bat_entry_off(index) / c.bat_dirty_block));
    }

    /// `seek_to_sector()`: the sector in the file, or -1 when unallocated or invalid.
    fn seek_to_sector(&self, c: &Consts, sector_num: i64) -> i64 {
        let index = (sector_num / i64::from(c.tracks)) as u32;
        let offset = sector_num % i64::from(c.tracks);

        // Not allocated.
        if index >= c.bat_size || self.bat(index) == 0 {
            return -1;
        }
        let cluster_off = self.bat2sect(c, index);
        if cluster_off < self.data_start || cluster_off + i64::from(c.tracks) > self.data_end {
            // The cluster is outside of the image file or overlaps the header.
            return -1;
        }
        cluster_off + offset
    }

    /// `block_status()`: the file sector of `sector_num` (-1 when unallocated) and how many
    /// sectors from there on are the same, contiguous in the file or unallocated.
    fn block_status(&self, c: &Consts, mut sector_num: i64, mut nb_sectors: i32) -> (i64, i32) {
        let mut start_off = -2i64;
        let mut prev_end_off = -2i64;
        let mut pnum = 0i32;
        while nb_sectors > 0 || start_off == -2 {
            let offset = self.seek_to_sector(c, sector_num);
            if start_off == -2 {
                start_off = offset;
                prev_end_off = offset;
            } else if offset != prev_end_off {
                break;
            }
            // cluster_remainder()
            let to_end =
                ((i64::from(c.tracks) - sector_num % i64::from(c.tracks)) as i32).min(nb_sectors);
            nb_sectors -= to_end;
            sector_num += i64::from(to_end);
            pnum += to_end;
            if offset > 0 {
                prev_end_off += i64::from(to_end);
            }
        }
        (start_off, pnum)
    }

    /// `host_cluster_index()`.
    fn host_cluster_index(&self, c: &Consts, off: i64) -> u32 {
        let off = off - (self.data_start << BDRV_SECTOR_BITS);
        (off / i64::from(c.cluster_size)) as u32
    }

    /// `mark_used()`: marks `count` clusters from the host offset `off` in `bitmap`. Fails
    /// with `E2BIG` past the end of the bitmap and `EBUSY` when one is already marked.
    fn mark_used(&self, c: &Consts, bitmap: &mut Bitmap, off: i64, count: u32) -> Result<(), i32> {
        let cluster_index = self.host_cluster_index(c, off);
        if u64::from(cluster_index) + u64::from(count) > bitmap.len {
            return Err(libc::E2BIG);
        }
        let next_used = bitmap.find_next(u64::from(cluster_index));
        if next_used < u64::from(cluster_index) + u64::from(count) {
            return Err(libc::EBUSY);
        }
        for i in 0..u64::from(count) {
            bitmap.set(u64::from(cluster_index) + i);
        }
        Ok(())
    }

    /// `parallels_fill_used_bitmap()`: the clusters the BAT points at, as far as the image
    /// allows. The bitmap is filled even when there are errors, for the repair.
    fn fill_used_bitmap(&mut self, c: &Consts, file: &Node) -> Result<(), i32> {
        let payload_bytes = file.getlength().map_err(|e| e.raw_os_error().unwrap_or(libc::EIO))?
            as i64
            - self.data_start * BDRV_SECTOR_SIZE as i64;
        if payload_bytes < 0 {
            return Err(libc::EINVAL);
        }
        let size = (payload_bytes as u64).div_ceil(u64::from(c.cluster_size));
        self.used_bmap = Bitmap::new(size);
        if size == 0 {
            return Ok(());
        }
        let mut bitmap = std::mem::take(&mut self.used_bmap);
        let mut err = Ok(());
        for i in 0..c.bat_size {
            let host_off = self.bat2sect(c, i) << BDRV_SECTOR_BITS;
            if host_off == 0 {
                continue;
            }
            let r = self.mark_used(c, &mut bitmap, host_off, 1);
            if r.is_err() && err.is_ok() {
                err = r;
            }
        }
        self.used_bmap = bitmap;
        err
    }

    /// `allocate_clusters()`: makes sure the clusters from `sector_num` on are allocated and
    /// returns the file sector of `sector_num` with how many sectors from there on are
    /// contiguous in the file.
    fn allocate_clusters(
        &mut self,
        c: &Consts,
        file: &Node,
        backing: Option<&Node>,
        sector_num: i64,
        nb_sectors: i32,
    ) -> io::Result<(i64, i32)> {
        let (pos, mut pnum) = self.block_status(c, sector_num, nb_sectors);
        if pos > 0 {
            return Ok((pos, pnum));
        }

        let tracks = i64::from(c.tracks);
        let cluster_size = i64::from(c.cluster_size);
        let idx = sector_num / tracks;
        let mut to_allocate = (sector_num + i64::from(pnum) + tracks - 1) / tracks - idx;

        // Only writes inside the disk get here, and block_status() stops at its end.
        assert!(idx < i64::from(c.bat_size) && idx + to_allocate <= i64::from(c.bat_size));

        let first_free = self.used_bmap.find_next_zero(0);
        let mut host_off;
        if first_free == self.used_bmap.len {
            let bytes =
                to_allocate * cluster_size + c.prealloc_size as i64 * BDRV_SECTOR_SIZE as i64;
            host_off = self.data_end * BDRV_SECTOR_SIZE as i64;

            // The expanded file has to read back as zeroes. Truncating does that if the user
            // allowed it and the file supports it; otherwise write the zeroes.
            let mut ret = Ok(());
            if self.prealloc_mode == PreallocModeP::Truncate {
                ret = file
                    .truncate_full(host_off + bytes, false, PreallocMode::Off, BDRV_REQ_ZERO_WRITE)
                    .map_err(|e| io_from(&e));
                if matches!(&ret, Err(e) if is_enotsup(e)) {
                    self.prealloc_mode = PreallocModeP::Fallocate;
                }
            }
            if self.prealloc_mode == PreallocModeP::Fallocate {
                ret = file.pwrite_zeroes(host_off as u64, bytes as u64, false);
            }
            ret?;

            let new_usedsize = self.used_bmap.len + (bytes / cluster_size) as u64;
            self.used_bmap.zero_extend(new_usedsize);
        } else {
            let next_used = self.used_bmap.find_next(first_free);

            // Not enough contiguous clusters in the middle, allocate fewer.
            if ((next_used - first_free) as i64) < to_allocate {
                to_allocate = (next_used - first_free) as i64;
                pnum = ((idx + to_allocate) * tracks - sector_num) as i32;
            }

            host_off = self.data_start * BDRV_SECTOR_SIZE as i64;
            host_off += first_free as i64 * cluster_size;

            // The tail area of the branch above needs no preallocation, but this is likely a
            // hole being reused. Preallocate it if prealloc_mode asks for that.
            if self.prealloc_mode == PreallocModeP::Fallocate
                && host_off < self.data_end * BDRV_SECTOR_SIZE as i64
            {
                file.pwrite_zeroes(host_off as u64, (cluster_size * to_allocate) as u64, false)?;
            }
        }

        // Fill the new clusters from the backing file. As in QEMU, this goes to data_end, and
        // most of it is overwritten by the write that follows.
        if let Some(backing) = backing {
            let nb_cow_bytes = (to_allocate * tracks) << BDRV_SECTOR_BITS;
            let mut buf = vec![0u8; nb_cow_bytes as usize];
            backing.pread((idx * tracks * BDRV_SECTOR_SIZE as i64) as u64, &mut buf)?;
            file.pwrite((self.data_end * BDRV_SECTOR_SIZE as i64) as u64, &buf)?;
        }

        let mut used = std::mem::take(&mut self.used_bmap);
        let r = self.mark_used(c, &mut used, host_off, to_allocate as u32);
        self.used_bmap = used;
        // The image is inconsistent if this fails.
        r.map_err(errno)?;

        for i in 0..to_allocate {
            let entry = (host_off / BDRV_SECTOR_SIZE as i64 / i64::from(c.off_multiplier)) as u32;
            self.set_bat_entry(c, (idx + i) as u32, entry);
            host_off += cluster_size;
        }
        if host_off > self.data_end * BDRV_SECTOR_SIZE as i64 {
            self.data_end = host_off / BDRV_SECTOR_SIZE as i64;
        }

        Ok((self.bat2sect(c, idx as u32) + sector_num % tracks, pnum))
    }

    /// `parallels_co_flush_to_os()`: writes the dirty parts of the header and BAT.
    fn flush_bat(&mut self, c: &Consts, file: &Node) -> io::Result<()> {
        let mut bit = self.bat_dirty_bmap.find_next(0);
        while bit < self.bat_dirty_bmap.len {
            let off = bit as u32 * c.bat_dirty_block;
            let to_write = c.bat_dirty_block.min(self.header_size - off);
            file.pwrite(u64::from(off), &self.header[off as usize..(off + to_write) as usize])?;
            bit = self.bat_dirty_bmap.find_next(bit + 1);
        }
        self.bat_dirty_bmap.zero();
        Ok(())
    }

    /// `parallels_update_header()`.
    fn update_header(&self, c: &Consts, file: &Node) -> io::Result<()> {
        let size = (c.mem_align.max(HEADER_SIZE as u32)).min(self.header_size);
        pwrite_sync(file, 0, &self.header[..size as usize])
    }

    /// `parallels_test_data_off()`: whether `data_off` is right, and the right value.
    fn test_data_off(&self, c: &Consts, file_nb_sectors: i64) -> (bool, u32) {
        let mut min_off = bat_entry_off(c.bat_size).div_ceil(BDRV_SECTOR_SIZE as u32);
        if !c.old_magic {
            min_off = min_off.next_multiple_of(c.cluster_size / BDRV_SECTOR_SIZE as u32);
        }
        let data_off = le32(&self.header, hdr::DATA_OFF);
        if data_off == 0 && c.old_magic {
            return (true, min_off);
        }
        if data_off < min_off || i64::from(data_off) > file_nb_sectors {
            return (false, min_off);
        }
        (true, data_off)
    }

    /// `parallels_check_unclean()`.
    fn check_unclean(&mut self, res: &mut CheckResult, fix: u32) {
        if !self.header_unclean {
            return;
        }
        eprintln!("{} image was not closed correctly", fix_word(fix, BDRV_FIX_ERRORS));
        res.corruptions += 1;
        if fix & BDRV_FIX_ERRORS != 0 {
            // Closing the image does the job.
            res.corruptions_fixed += 1;
            self.header_unclean = false;
        }
    }

    /// `parallels_check_data_off()`.
    fn check_data_off(
        &mut self,
        c: &Consts,
        file: &Node,
        res: &mut CheckResult,
        fix: u32,
    ) -> io::Result<()> {
        let file_size = match file.nb_sectors() {
            Ok(n) => n as i64,
            Err(e) => {
                res.check_errors += 1;
                return Err(e);
            }
        };
        let (ok, data_off) = self.test_data_off(c, file_size);
        if ok {
            return Ok(());
        }
        res.corruptions += 1;
        if fix & BDRV_FIX_ERRORS != 0 {
            put_le32(&mut self.header, hdr::DATA_OFF, data_off);
            self.data_start = i64::from(data_off);
            // Only running out of memory is an error in QEMU, which cannot happen here.
            let _ = self.fill_used_bitmap(c, file);
            res.corruptions_fixed += 1;
        }
        eprintln!("{} data_off field has incorrect value", fix_word(fix, BDRV_FIX_ERRORS));
        Ok(())
    }

    /// `parallels_check_outside_image()`.
    fn check_outside_image(
        &mut self,
        c: &Consts,
        file: &Node,
        res: &mut CheckResult,
        fix: u32,
    ) -> io::Result<()> {
        let size = match file.getlength() {
            Ok(n) => n as i64,
            Err(e) => {
                res.check_errors += 1;
                return Err(e);
            }
        };
        let data_start_off = self.data_start << BDRV_SECTOR_BITS;
        let cluster_size = i64::from(c.cluster_size);

        let mut high_off = 0i64;
        for i in 0..c.bat_size {
            let off = self.bat2sect(c, i) << BDRV_SECTOR_BITS;
            if off == 0 {
                continue;
            }
            if off < data_start_off || off + cluster_size > size {
                eprintln!("{} cluster {i} is outside image", fix_word(fix, BDRV_FIX_ERRORS));
                res.corruptions += 1;
                if fix & BDRV_FIX_ERRORS != 0 {
                    self.set_bat_entry(c, i, 0);
                    res.corruptions_fixed += 1;
                }
                continue;
            }
            high_off = high_off.max(off);
        }

        if high_off == 0 {
            res.image_end_offset = self.data_end << BDRV_SECTOR_BITS;
        } else {
            res.image_end_offset = high_off + cluster_size;
            self.data_end = res.image_end_offset >> BDRV_SECTOR_BITS;
        }
        Ok(())
    }

    /// `parallels_check_leak()`: the file should end with the last cluster. `explicit` says
    /// whether leaks are counted and reported, rather than just cut off.
    fn check_leak(
        &mut self,
        c: &Consts,
        file: &Node,
        res: &mut CheckResult,
        fix: u32,
        explicit: bool,
    ) -> io::Result<()> {
        let size = match file.getlength() {
            Ok(n) => n as i64,
            Err(e) => {
                res.check_errors += 1;
                return Err(e);
            }
        };
        if size > res.image_end_offset {
            let count = (size - res.image_end_offset + i64::from(c.cluster_size) - 1)
                / i64::from(c.cluster_size);
            if explicit {
                eprintln!(
                    "{} space leaked at the end of the image {}",
                    fix_word(fix, BDRV_FIX_LEAKS),
                    size - res.image_end_offset
                );
                res.leaks += count;
            }
            if fix & BDRV_FIX_LEAKS != 0 {
                // Really repairing the image means shrinking it, so exact=true.
                if let Err(e) = file.truncate_full(res.image_end_offset, true, PreallocMode::Off, 0)
                {
                    report::error_report(e.message());
                    res.check_errors += 1;
                    return Err(io_from(&e));
                }
                if explicit {
                    res.leaks_fixed += count;
                }
            }
        }
        Ok(())
    }

    /// `parallels_check_duplicate()`: BAT entries that point at the same cluster. A repair
    /// gives the later entries copies of the cluster.
    fn check_duplicate(
        &mut self,
        c: &Consts,
        file: &Node,
        backing: Option<&Node>,
        res: &mut CheckResult,
        fix: u32,
    ) -> io::Result<()> {
        // Clusters allocated for the repair are past every cluster a BAT entry points at, so
        // they need not be in the bitmap.
        let mut bitmap_size = self.host_cluster_index(c, res.image_end_offset);
        if bitmap_size == 0 {
            return Ok(());
        }
        if res.image_end_offset % i64::from(c.cluster_size) != 0 {
            // An unaligned image end makes the bitmap one shorter.
            bitmap_size += 1;
        }
        let mut bitmap = Bitmap::new(u64::from(bitmap_size));
        let mut buf = vec![0u8; c.cluster_size as usize];
        let mut fixed = false;

        for i in 0..c.bat_size {
            let mut host_off = self.bat2sect(c, i) << BDRV_SECTOR_BITS;
            if host_off == 0 {
                continue;
            }
            match self.mark_used(c, &mut bitmap, host_off, 1) {
                Ok(()) => continue,
                // QEMU asserts this cannot happen: every entry still pointing outside the
                // image was either repaired or is not a duplicate we can tell.
                Err(libc::E2BIG) => continue,
                Err(_) => {}
            }

            // This cluster duplicates another one.
            eprintln!("{} duplicate offset in BAT entry {i}", fix_word(fix, BDRV_FIX_ERRORS));
            res.corruptions += 1;
            if fix & BDRV_FIX_ERRORS == 0 {
                continue;
            }

            // Reset the entry and allocate a new cluster for its guest offset, where the
            // allocator likes, then copy the old cluster there. Keep the old entry to put
            // back if that fails.
            let bat_entry = self.bat(i);
            self.set_bat_entry(c, i, 0);

            let r = (|| -> io::Result<i64> {
                file.pread(host_off as u64, &mut buf)?;
                let guest_sector = (i64::from(i) * i64::from(c.cluster_size)) >> BDRV_SECTOR_BITS;
                let (host_sector, _) =
                    self.allocate_clusters(c, file, backing, guest_sector, c.tracks as i32)?;
                let host_off = host_sector << BDRV_SECTOR_BITS;
                file.pwrite(host_off as u64, &buf)?;
                Ok(host_off)
            })();
            match r {
                Ok(o) => host_off = o,
                Err(e) => {
                    res.check_errors += 1;
                    put_le32(&mut self.header, bat_entry_off(i) as usize, bat_entry);
                    return Err(e);
                }
            }
            if host_off + i64::from(c.cluster_size) > res.image_end_offset {
                res.image_end_offset = host_off + i64::from(c.cluster_size);
            }

            // The allocator will reuse holes inside the image, so keep the bitmap right for
            // the new cluster too. Clusters past the image are not in it, so E2BIG is fine.
            if self.mark_used(c, &mut bitmap, host_off, 1) == Err(libc::EBUSY) {
                res.check_errors += 1;
                put_le32(&mut self.header, bat_entry_off(i) as usize, bat_entry);
                return Err(errno(libc::EBUSY));
            }
            fixed = true;
            res.corruptions_fixed += 1;
        }

        if fixed {
            // New clusters grew the file by the preallocation size. Cut it to the right size
            // without counting that as a leak.
            self.check_leak(c, file, res, fix, false)?;
        }
        Ok(())
    }

    /// `parallels_collect_statistics()`.
    fn collect_statistics(&self, c: &Consts, res: &mut CheckResult) {
        res.bfi.total_clusters = u64::from(c.bat_size);
        // Compression is not supported.
        res.bfi.compressed_clusters = 0;

        let cluster_size = i64::from(c.cluster_size);
        let mut prev_off = 0i64;
        for i in 0..c.bat_size {
            let off = self.bat2sect(c, i) << BDRV_SECTOR_BITS;
            // Without BDRV_FIX_ERRORS, entries outside the image are still there. Skip them
            // along with the unallocated ones.
            if off == 0 || off + cluster_size > res.image_end_offset {
                prev_off = 0;
                continue;
            }
            if prev_off != 0 && prev_off + cluster_size != off {
                res.bfi.fragmented_clusters += 1;
            }
            prev_off = off;
            res.bfi.allocated_clusters += 1;
        }
    }

    /// The part of `parallels_co_check()` under the lock.
    fn check(
        &mut self,
        c: &Consts,
        file: &Node,
        backing: Option<&Node>,
        res: &mut CheckResult,
        fix: u32,
    ) -> io::Result<()> {
        self.check_unclean(res, fix);
        self.check_data_off(c, file, res, fix)?;
        self.check_outside_image(c, file, res, fix)?;
        self.check_leak(c, file, res, fix, true)?;
        self.check_duplicate(c, file, backing, res, fix)?;
        self.collect_statistics(c, res);
        Ok(())
    }
}

fn open_err(file: &Node, e: io::Error) -> Error {
    let name = file.filename().unwrap_or_default();
    Error::from_io(format!("Could not open '{name}'"), e)
}

/// `parallels_probe()`.
fn parallels_probe(buf: &[u8], _filename: Option<&str>) -> i32 {
    if buf.len() < HEADER_SIZE {
        return 0;
    }
    let magic = &buf[hdr::MAGIC..hdr::MAGIC + 16];
    if (magic == HEADER_MAGIC || magic == HEADER_MAGIC2)
        && le32(buf, hdr::VERSION) == HEADER_VERSION
    {
        return 100;
    }
    0
}

/// `parallels_open()`.
fn parallels_open_node(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::Parallels(o) = opts else {
        unreachable!("parallels driver with other options")
    };
    let mut runtime = args.meta.options.clone();
    let (prealloc_size, prealloc_mode) = parallels_opts_prealloc(&mut runtime)?;
    let file = args.open_child(*o.file, "file", BDRV_CHILD_IMAGE | BDRV_CHILD_PRIMARY)?;
    let rdwr = !args.flags.read_only;
    let inactive = args.flags.inactive;
    Ok(Box::new(parallels_open(&file, prealloc_size, prealloc_mode, rdwr, inactive)?))
}

fn parallels_open(
    file: &Node,
    prealloc_size: u64,
    prealloc_mode: PreallocModeP,
    rdwr: bool,
    inactive: bool,
) -> Result<ParallelsDriver> {
    let file_nb_sectors =
        file.nb_sectors().map_err(|_| open_err(file, errno(libc::EINVAL)))? as i64;

    let mut ph = [0u8; HEADER_SIZE];
    file.pread(0, &mut ph).map_err(|e| open_err(file, e))?;

    let mut total_sectors = le64(&ph, hdr::NB_SECTORS);
    let fail_format = || Error::generic("Image not in Parallels format");
    if le32(&ph, hdr::VERSION) != HEADER_VERSION {
        return Err(fail_format());
    }
    let magic = &ph[hdr::MAGIC..hdr::MAGIC + 16];
    let tracks = le32(&ph, hdr::TRACKS);
    let (off_multiplier, old_magic) = if magic == HEADER_MAGIC {
        total_sectors &= 0xffff_ffff;
        (1, true)
    } else if magic == HEADER_MAGIC2 {
        (tracks, false)
    } else {
        return Err(fail_format());
    };

    if tracks == 0 {
        return Err(Error::generic("Invalid image: Zero sectors per track"));
    }
    if tracks > i32::MAX as u32 / 513 {
        return Err(Error::generic("Invalid image: Too big cluster"));
    }
    let prealloc_size = prealloc_size.max(u64::from(tracks));
    let cluster_size = tracks << BDRV_SECTOR_BITS;

    let bat_size = le32(&ph, hdr::BAT_ENTRIES);
    if bat_size > i32::MAX as u32 / 4 {
        return Err(Error::generic("Catalog too large"));
    }
    let ext_off = le64(&ph, hdr::EXT_OFF);
    if ext_off >= (i64::MAX >> BDRV_SECTOR_BITS) as u64 {
        return Err(Error::generic("Invalid image: Too big offset"));
    }
    if u64::from(bat_size) * u64::from(tracks) < total_sectors {
        return Err(Error::generic(
            "Invalid image: Catalog size too small for advertised disk size",
        ));
    }

    let mem_align = file.limits().opt_mem_alignment.max(1) as u32;
    let size = bat_entry_off(bat_size);
    let mut header_size = size.next_multiple_of(mem_align);
    let mut header = vec![0u8; header_size as usize];
    // One request of header_size bytes may be larger than a request can be.
    let mut header_off = 0u32;
    while header_off < header_size {
        let chunk = (header_size - header_off).min(PARALLELS_HEADER_READ_CHUNK);
        file.pread(
            u64::from(header_off),
            &mut header[header_off as usize..(header_off + chunk) as usize],
        )
        .map_err(|e| open_err(file, e))?;
        header_off += chunk;
    }

    let c0 = Consts {
        total_sectors,
        bat_size,
        tracks,
        cluster_size,
        off_multiplier,
        prealloc_size,
        bat_dirty_block: 4 * 4096,
        mem_align,
        old_magic,
    };
    let mut s = State {
        header,
        header_size,
        header_unclean: false,
        bat_dirty_bmap: Bitmap::default(),
        used_bmap: Bitmap::default(),
        data_start: 0,
        data_end: 0,
        prealloc_mode,
    };

    let mut need_check = false;
    if le32(&ph, hdr::INUSE) == HEADER_INUSE_MAGIC {
        need_check = true;
        s.header_unclean = true;
    }
    let (ok, data_start) = s.test_data_off(&c0, file_nb_sectors);
    need_check = need_check || !ok;

    s.data_start = i64::from(data_start);
    s.data_end = s.data_start;
    if s.data_end < i64::from(header_size >> BDRV_SECTOR_BITS) {
        // There is no room to align the header up between the BAT and the data, so the
        // header writes are read-modify-write.
        header_size = size;
        s.header_size = size;
        s.header.truncate(size as usize);
    }

    let mut bitmaps = Vec::new();
    if ext_off != 0 {
        if rdwr {
            // Opening an image with an extension read-write is unsafe, as the extension is not
            // supported. But QEMU's parallels driver always ignored it, so warn and go on.
            report::warn_report("Format Extension ignored in RW mode");
        } else {
            let read = |off: u64, buf: &mut [u8]| file.pread(off, buf);
            let ctx = ext::ExtCtx { read: &read, cluster_size, total_sectors };
            bitmaps = ext::read_format_extension(&ctx, ext_off << BDRV_SECTOR_BITS)?;
        }
    }

    if rdwr && !inactive {
        put_le32(&mut s.header, hdr::INUSE, HEADER_INUSE_MAGIC);
        s.update_header(&c0, file).map_err(|e| open_err(file, e))?;
    }

    s.bat_dirty_bmap = Bitmap::new(u64::from(header_size.div_ceil(c0.bat_dirty_block)));

    for i in 0..bat_size {
        let sector = s.bat2sect(&c0, i);
        if sector == 0 {
            // Not allocated.
            continue;
        }
        if sector < i64::from(data_start) || sector + i64::from(tracks) > file_nb_sectors {
            // The cluster is outside of the image file or overlaps the header.
            need_check = true;
            continue;
        }
        if sector + i64::from(tracks) > s.data_end {
            s.data_end = sector + i64::from(tracks);
        }
    }

    if !need_check {
        // These are correctable errors.
        need_check = s.fill_used_bitmap(&c0, file).is_err();
    }

    // Inactive and read-only images are not repaired. (QEMU also leaves images opened for a
    // check alone, see the module doc.)
    if inactive || !rdwr {
        return Ok(ParallelsDriver { c: c0, lock: Mutex::new(s), bitmaps });
    }

    // Repair the image if it is corrupted.
    if need_check {
        let mut res = CheckResult::default();
        let r = s
            .check(&c0, file, None, &mut res, BDRV_FIX_ERRORS | BDRV_FIX_LEAKS)
            .and_then(|()| s.flush_bat(&c0, file))
            .and_then(|()| file.flush());
        if let Err(e) = r {
            return Err(Error::from_io("Could not repair corrupted image", e));
        }
        // QEMU leaves used_bmap as it was before the repair (empty, or without the clusters
        // the repair moved), so later writes can reuse clusters the BAT points at. See the
        // module doc.
        let _ = s.fill_used_bitmap(&c0, file);
    }
    Ok(ParallelsDriver { c: c0, lock: Mutex::new(s), bitmaps })
}

impl ParallelsDriver {
    fn state(&self) -> MutexGuard<'_, State> {
        self.lock.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The largest request in sectors the sector based interface of QEMU takes, an `int`.
const MAX_REQUEST_SECTORS: u64 = (i32::MAX as u64) >> BDRV_SECTOR_BITS;

impl Driver for ParallelsDriver {
    /// `parallels_co_readv()`.
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let file = bs.file();
        let backing = bs.backing().map(|c| c.node);
        let mut sector_num = (offset >> BDRV_SECTOR_BITS) as i64;
        let mut done = 0usize;
        while done < buf.len() {
            let nb_sectors =
                (((buf.len() - done) as u64 >> BDRV_SECTOR_BITS).min(MAX_REQUEST_SECTORS)) as i32;
            let (position, n) = self.state().block_status(&self.c, sector_num, nb_sectors);
            let nbytes = (n as usize) << BDRV_SECTOR_BITS;
            let chunk = &mut buf[done..done + nbytes];
            if position < 0 {
                match &backing {
                    Some(b) => b.pread((sector_num as u64) << BDRV_SECTOR_BITS, chunk)?,
                    None => chunk.fill(0),
                }
            } else {
                file.pread((position as u64) << BDRV_SECTOR_BITS, chunk)?;
            }
            sector_num += i64::from(n);
            done += nbytes;
        }
        Ok(())
    }

    /// `parallels_co_writev()`.
    fn pwrite(&self, bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        let file = bs.file();
        let backing = bs.backing().map(|c| c.node);
        let mut sector_num = (offset >> BDRV_SECTOR_BITS) as i64;
        let mut done = 0usize;
        while done < buf.len() {
            let nb_sectors =
                (((buf.len() - done) as u64 >> BDRV_SECTOR_BITS).min(MAX_REQUEST_SECTORS)) as i32;
            let (position, n) = self.state().allocate_clusters(
                &self.c,
                &file,
                backing.as_deref(),
                sector_num,
                nb_sectors,
            )?;
            let nbytes = (n as usize) << BDRV_SECTOR_BITS;
            file.pwrite((position as u64) << BDRV_SECTOR_BITS, &buf[done..done + nbytes])?;
            sector_num += i64::from(n);
            done += nbytes;
        }
        Ok(())
    }

    /// `parallels_co_pdiscard()`. There is no zero flag in the BAT, so a discarded cluster
    /// would show stale data of a backing file.
    fn pdiscard(&self, bs: &Node, offset: u64, bytes: u64) -> io::Result<()> {
        if bs.backing().is_some() {
            return Err(errno(libc::ENOTSUP));
        }
        let cluster_size = u64::from(self.c.cluster_size);
        if offset % cluster_size != 0 || bytes % cluster_size != 0 {
            return Err(errno(libc::ENOTSUP));
        }
        let file = bs.file();
        let mut cluster = (offset / cluster_size) as u32;
        let mut count = (bytes / cluster_size) as u32;

        let mut s = self.state();
        while count > 0 {
            let host_off = s.bat2sect(&self.c, cluster) << BDRV_SECTOR_BITS;
            if host_off != 0 {
                file.pdiscard(host_off as u64, cluster_size)?;
                s.set_bat_entry(&self.c, cluster, 0);
                let idx = u64::from(s.host_cluster_index(&self.c, host_off));
                if idx < s.used_bmap.len {
                    s.used_bmap.clear(idx);
                }
            }
            cluster += 1;
            count -= 1;
        }
        Ok(())
    }

    /// `parallels_co_pwrite_zeroes()`: the format has no zero flag, so this is a discard,
    /// which only works without a backing file.
    fn pwrite_zeroes(
        &self,
        bs: &Node,
        offset: u64,
        bytes: u64,
        _may_unmap: bool,
    ) -> io::Result<()> {
        self.pdiscard(bs, offset, bytes)
    }

    fn getlength(&self, _bs: &Node) -> io::Result<u64> {
        Ok(self.c.total_sectors * BDRV_SECTOR_SIZE)
    }

    /// `parallels_co_flush_to_os()`.
    fn flush_to_os(&self, bs: &Node) -> io::Result<()> {
        self.state().flush_bat(&self.c, &bs.file())
    }

    /// The sector based interface of the driver in QEMU makes the requests sector aligned.
    fn refresh_limits(&self, _bs: &Node, bl: &mut BlockLimits) -> Result<()> {
        bl.request_alignment = bl.request_alignment.max(BDRV_SECTOR_SIZE as u32);
        Ok(())
    }

    /// `parallels_co_block_status()`.
    fn block_status(
        &self,
        bs: &Node,
        _want: u32,
        offset: u64,
        bytes: u64,
    ) -> Option<io::Result<BlockStatus>> {
        let nb_sectors = (bytes >> BDRV_SECTOR_BITS).min(MAX_REQUEST_SECTORS) as i32;
        let (off, count) =
            self.state().block_status(&self.c, (offset >> BDRV_SECTOR_BITS) as i64, nb_sectors);
        let pnum = count as u64 * BDRV_SECTOR_SIZE;
        if off < 0 {
            return Some(Ok(BlockStatus { ret: 0, pnum, map: 0, file: None }));
        }
        Some(Ok(BlockStatus {
            ret: BDRV_BLOCK_DATA | BDRV_BLOCK_OFFSET_VALID,
            pnum,
            map: off as u64 * BDRV_SECTOR_SIZE,
            file: Some(bs.file()),
        }))
    }

    /// `bdrv_has_zero_init_1`.
    fn has_zero_init(&self, _bs: &Node) -> Option<bool> {
        Some(true)
    }

    /// `parallels_co_check()`.
    fn check(&self, bs: &Node, fix: u32) -> Option<Result<CheckResult>> {
        let file = bs.file();
        let backing = bs.backing().map(|c| c.node);
        let mut res = CheckResult::default();
        let r = self.state().check(&self.c, &file, backing.as_deref(), &mut res, fix);
        let r = r.and_then(|()| bs.flush().inspect_err(|_| res.check_errors += 1));
        Some(match r {
            Ok(()) => Ok(res),
            Err(e) => Err(Error::with_cause(ruvm_base::error::strerror(&e), e)),
        })
    }

    /// `parallels_close()`: clears `inuse` and cuts the file after the last cluster.
    fn close(&self, bs: &Node) {
        let flags = bs.flags();
        if flags.read_only || flags.inactive {
            return;
        }
        let file = bs.file();
        let mut s = self.state();
        // bdrv_close() flushes before the driver closes in QEMU.
        let _ = s.flush_bat(&self.c, &file);
        put_le32(&mut s.header, hdr::INUSE, 0);
        let _ = s.update_header(&self.c, &file);
        // Errors are ignored, so exact=true does no harm.
        let _ = file.truncate_full(s.data_end << BDRV_SECTOR_BITS, true, PreallocMode::Off, 0);
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

/// `parallels_co_create()`: `blockdev-create` with `driver: parallels`.
fn parallels_co_create(graph: &BlockGraph, options: BlockdevCreateOptionsU) -> Result<()> {
    let BlockdevCreateOptionsU::Parallels(o) = options else {
        unreachable!("parallels driver with other create options")
    };
    do_create(graph, o)
}

fn do_create(graph: &BlockGraph, o: BlockdevCreateOptionsParallels) -> Result<()> {
    // Sanity checks. The sizes are int64_t in QEMU.
    let total_size = o.size as i64;
    let cl_size = o.cluster_size.unwrap_or(DEFAULT_CLUSTER_SIZE) as i64;

    // Bound cl_size so that the multiplication below cannot overflow.
    if cl_size >= i64::MAX / MAX_PARALLELS_IMAGE_FACTOR as i64 {
        return Err(Error::generic("Cluster size is too large"));
    }
    if cl_size <= 0 || total_size as u64 >= MAX_PARALLELS_IMAGE_FACTOR * cl_size as u64 {
        return Err(Error::generic("Image size is too large for this cluster size"));
    }
    let bat_count = (total_size + cl_size - 1) / cl_size;
    if bat_count > i64::from(i32::MAX) / 4 {
        return Err(Error::generic("Catalog too large"));
    }
    if total_size % BDRV_SECTOR_SIZE as i64 != 0 {
        return Err(Error::generic("Image size must be a multiple of 512 bytes"));
    }
    if cl_size % BDRV_SECTOR_SIZE as i64 != 0 {
        return Err(Error::generic("Cluster size must be a multiple of 512 bytes"));
    }

    let blk = graph.open_create_blk(o.file)?;
    let node = blk.root().expect("a new backend has its node");

    let bat_entries = bat_count as u32;
    let bat_sectors = u64::from(bat_entry_off(bat_entries)).div_ceil(cl_size as u64);
    let bat_sectors = ((bat_sectors * cl_size as u64) >> BDRV_SECTOR_BITS) as u32;

    let mut tmp = [0u8; BDRV_SECTOR_SIZE as usize];
    tmp[hdr::MAGIC..hdr::MAGIC + 16].copy_from_slice(HEADER_MAGIC2);
    put_le32(&mut tmp, hdr::VERSION, HEADER_VERSION);
    // The geometry is not used at the image level, it is only there for the specification.
    put_le32(&mut tmp, hdr::HEADS, HEADS_NUMBER);
    let cylinders = (total_size as u64 / BDRV_SECTOR_SIZE / u64::from(HEADS_NUMBER) / SEC_IN_CYL)
        .min(u64::from(u32::MAX));
    put_le32(&mut tmp, hdr::CYLINDERS, cylinders as u32);
    put_le32(&mut tmp, hdr::TRACKS, (cl_size >> BDRV_SECTOR_BITS) as u32);
    put_le32(&mut tmp, hdr::BAT_ENTRIES, bat_entries);
    put_le64(&mut tmp, hdr::NB_SECTORS, (total_size as u64).div_ceil(BDRV_SECTOR_SIZE));
    put_le32(&mut tmp, hdr::DATA_OFF, bat_sectors);

    let r = node.pwrite(0, &tmp).and_then(|()| {
        node.pwrite_zeroes(
            BDRV_SECTOR_SIZE,
            u64::from(bat_sectors.saturating_sub(1)) << BDRV_SECTOR_BITS,
            false,
        )
    });
    r.map_err(|e| Error::from_io("Failed to create Parallels image", e))
}

/// `parallels_co_create_opts()`: `qemu-img create -f parallels`.
fn parallels_co_create_opts(filename: &str, options: &mut QDict) -> Result<()> {
    let (graph, _blk, qdict) = create_opts_open(
        filename,
        options,
        "parallels",
        &["size", "cluster_size"],
        &[("cluster_size", "cluster-size")],
    )?;
    let opts = visit_create_options(qdict)?;
    let BlockdevCreateOptionsU::Parallels(mut o) = opts.u else {
        unreachable!("parallels create options")
    };
    // Silently round up the sizes.
    o.size = o.size.next_multiple_of(BDRV_SECTOR_SIZE);
    o.cluster_size = o.cluster_size.map(|c| c.next_multiple_of(BDRV_SECTOR_SIZE));
    let file = match &o.file {
        BlockdevRef::Reference(n) => BlockdevRef::Reference(n.clone()),
        other => other.clone(),
    };
    do_create(&graph, BlockdevCreateOptionsParallels { file, ..o })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_opts_list() {
        let names: Vec<_> = PARALLELS_CREATE_OPTS.iter().map(|d| d.name).collect();
        assert_eq!(names, ["size", "cluster_size"]);
        assert_eq!(PARALLELS_CREATE_OPTS[0].help, Some("Virtual disk size"));
        assert_eq!(PARALLELS_CREATE_OPTS[0].def_value_str, None);
        assert_eq!(PARALLELS_CREATE_OPTS[1].help, Some("Parallels image cluster size"));
        assert_eq!(PARALLELS_CREATE_OPTS[1].def_value_str, Some("1048576"));
    }

    #[test]
    fn probe() {
        let mut h = [0u8; 64];
        h[..16].copy_from_slice(HEADER_MAGIC);
        assert_eq!(parallels_probe(&h, None), 0);
        put_le32(&mut h, hdr::VERSION, 2);
        assert_eq!(parallels_probe(&h, None), 100);
        h[..16].copy_from_slice(HEADER_MAGIC2);
        assert_eq!(parallels_probe(&h, None), 100);
        assert_eq!(parallels_probe(&h[..63], None), 0);
        h[0] = b'X';
        assert_eq!(parallels_probe(&h, None), 0);
    }

    #[test]
    fn prealloc_opts() {
        let mut o = QDict::new();
        assert_eq!(parallels_opts_prealloc(&mut o).unwrap(), (262_144, PreallocModeP::Fallocate));
        o.put("prealloc-size", "1M");
        o.put("prealloc-mode", "truncate");
        assert_eq!(parallels_opts_prealloc(&mut o).unwrap(), (2048, PreallocModeP::Truncate));
        assert!(o.is_empty());
        o.put("prealloc-mode", "bogus");
        assert_eq!(
            parallels_opts_prealloc(&mut o).unwrap_err().message(),
            "invalid parameter value: bogus"
        );
    }

    #[test]
    fn bitmap() {
        let mut b = Bitmap::new(130);
        assert_eq!(b.find_next(0), 130);
        assert_eq!(b.find_next_zero(0), 0);
        b.set(0);
        b.set(64);
        b.set(129);
        assert_eq!(b.find_next(1), 64);
        assert_eq!(b.find_next(65), 129);
        assert_eq!(b.find_next_zero(0), 1);
        for i in 0..130 {
            b.set(i);
        }
        assert_eq!(b.find_next_zero(0), 130);
        b.clear(100);
        assert!(!b.get(100));
        assert_eq!(b.find_next_zero(0), 100);
        b.zero_extend(200);
        assert_eq!(b.find_next_zero(101), 130);
    }

    fn ext_ctx_parse(
        cluster: &[u8],
        total_sectors: u64,
        read: &dyn Fn(u64, &mut [u8]) -> io::Result<()>,
    ) -> Result<Vec<LoadedBitmap>> {
        let ctx = ext::ExtCtx { read, cluster_size: cluster.len() as u32, total_sectors };
        ext::parse_format_extension(&ctx, cluster)
    }

    #[test]
    fn format_extension() {
        let no_read = |_: u64, _: &mut [u8]| -> io::Result<()> { panic!("no reads expected") };
        let id = *b"\x01\x23\x45\x67\x89\xab\xcd\xef\x01\x23\x45\x67\x89\xab\xcd\xef";

        // Nothing but the end of features.
        let c = ext::build_extension(4096, &[]);
        assert_eq!(ext_ctx_parse(&c, 2048, &no_read).unwrap(), vec![]);

        // A 1 MiB disk with a 64 KiB granularity bitmap: 16 bits, one L1 entry, all ones.
        let f = ext::bitmap_feature(2048, id, 128, &[1]);
        let c = ext::build_extension(4096, &[(ext::DIRTY_BITMAP_MAGIC, f)]);
        let b = ext_ctx_parse(&c, 2048, &no_read).unwrap();
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].name, "01234567-89ab-cdef-0123-456789abcdef");
        assert_eq!(b[0].granularity, 65536);
        assert_eq!(b[0].words, vec![0xffff]);
        assert_eq!(b[0].count(), 1 << 20);

        // A data cluster.
        let f = ext::bitmap_feature(2048, id, 128, &[8]);
        let c = ext::build_extension(4096, &[(ext::DIRTY_BITMAP_MAGIC, f)]);
        let read = |off: u64, buf: &mut [u8]| -> io::Result<()> {
            assert_eq!(off, 8 * 512);
            buf.fill(0);
            buf[0] = 0b1010_0101;
            buf[1] = 0xff;
            buf[2] = 0xff;
            Ok(())
        };
        let b = ext_ctx_parse(&c, 2048, &read).unwrap();
        assert_eq!(b[0].words, vec![0xffa5]);
        assert!(b[0].get(0));
        assert!(!b[0].get(65536));

        // The errors.
        let err =
            |c: &[u8], ts: u64| ext_ctx_parse(c, ts, &no_read).unwrap_err().message().to_string();
        let mut bad = ext::build_extension(4096, &[]);
        bad[0] ^= 1;
        assert_eq!(
            err(&bad, 2048),
            "Wrong parallels Format Extension magic: 0xab234cef23dcea86, expected: \
             0xab234cef23dcea87"
        );
        let mut bad = ext::build_extension(4096, &[]);
        bad[100] = 1;
        assert_eq!(
            err(&bad, 2048),
            "Wrong checksum in Format Extension header. Format extension is corrupted."
        );
        let c = ext::build_extension(4096, &[(0x1234, vec![0; 8])]);
        assert_eq!(err(&c, 2048), "Unknown feature: 0x1234");
        let f = ext::bitmap_feature(2048, id, 128, &[1]);
        let c = ext::build_extension(4096, &[(ext::DIRTY_BITMAP_MAGIC, f)]);
        assert_eq!(
            err(&c, 4096),
            "Bitmap size (in sectors) 2048 differs from disk size in sectors 4096"
        );
        let f = ext::bitmap_feature(2048, id, 128, &[1, 1]);
        let c = ext::build_extension(4096, &[(ext::DIRTY_BITMAP_MAGIC, f)]);
        assert_eq!(
            err(&c, 2048),
            "Bitmap table size 2 does not correspond to bitmap size and cluster size. Expected 1"
        );
        let c = ext::build_extension(4096, &[(ext::DIRTY_BITMAP_MAGIC, vec![0; 16])]);
        assert_eq!(
            err(&c, 2048),
            "Too small Bitmap Feature area in Parallels Format Extension: 16 bytes, expected at \
             least 32 bytes"
        );
        let f = ext::bitmap_feature(2048, id, 128, &[1]);
        let c = ext::build_extension(
            4096,
            &[(ext::DIRTY_BITMAP_MAGIC, f.clone()), (ext::DIRTY_BITMAP_MAGIC, f)],
        );
        assert_eq!(err(&c, 2048), "Bitmap already exists: 01234567-89ab-cdef-0123-456789abcdef");
    }
}
