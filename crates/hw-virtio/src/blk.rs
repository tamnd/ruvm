// SPDX-License-Identifier: GPL-2.0-or-later

//! virtio-blk, a port of `hw/block/virtio-blk.c`.
//!
//! The device parses requests from its queues and carries them out synchronously against a
//! [`BlockBackend`]. Supported request types are `IN`, `OUT`, `FLUSH`, `GET_ID`, `DISCARD` and
//! `WRITE_ZEROES`. Everything else, including SCSI passthrough and the zoned commands, completes
//! with `VIRTIO_BLK_S_UNSUPP`, which is what QEMU does for a non-zoned disk on a guest that did
//! not negotiate SCSI.
//!
//! Differences from QEMU:
//!
//! - Requests run to completion inside the queue notify, one at a time. There is no request
//!   merging, no coroutine or AIO layer, no iothread and no dataplane.
//! - Only the `report` error policy exists: a failed read, write, flush, discard or write
//!   zeroes completes with `VIRTIO_BLK_S_IOERR`. QEMU's `rerror` and `werror` stop and retry
//!   policies are not ported.
//! - With the write cache off, every write is followed by a flush of the backend, standing in
//!   for the block layer's writethrough mode.
//! - The default geometry is QEMU's size based guess (16 heads, 63 sectors). The guess from an
//!   MBR partition table and host geometry probing are not ported.
//! - The block limits QEMU reads from the image (optimal transfer size, discard alignment) are
//!   not known here, so only the properties set in [`VirtioBlkConf`] are reported.
//! - Requests complete inside the queue handler, so migration never has one in flight and
//!   the device's own part of the stream is only the end of the request list.
//!
//! Not ported: zoned devices, secure erase, SCSI passthrough, multiqueue with iothread mapping,
//! DMA restart after a stop, trace points, QOM registration, block accounting and the
//! `drive` property. `ruvm-block` has no I/O path yet, so the device talks to the small
//! [`BlockBackend`] trait instead, with [`MemBlockBackend`] as an in-memory implementation.

use std::any::Any;
use std::fmt;
use std::io;
use std::sync::{Arc, Mutex, PoisonError};

use ruvm_base::{Error, Result};
use ruvm_virtio_queue::{DescriptorChain, GuestMemory, Reader, Writer};

use crate::virtio::{
    VIRTIO_CONFIG_S_DRIVER_OK, VIRTIO_F_ANY_LAYOUT, VIRTIO_F_VERSION_1, VirtIODevice,
    VirtioDeviceClass, feature, has_feature,
};

/// `TYPE_VIRTIO_BLK`.
pub const TYPE_VIRTIO_BLK: &str = "virtio-blk-device";

/// `VIRTIO_ID_BLOCK`.
pub const VIRTIO_ID_BLOCK: u16 = 2;

/// `VIRTIO_BLK_F_SIZE_MAX`.
pub const VIRTIO_BLK_F_SIZE_MAX: u32 = 1;
/// `VIRTIO_BLK_F_SEG_MAX`.
pub const VIRTIO_BLK_F_SEG_MAX: u32 = 2;
/// `VIRTIO_BLK_F_GEOMETRY`.
pub const VIRTIO_BLK_F_GEOMETRY: u32 = 4;
/// `VIRTIO_BLK_F_RO`.
pub const VIRTIO_BLK_F_RO: u32 = 5;
/// `VIRTIO_BLK_F_BLK_SIZE`.
pub const VIRTIO_BLK_F_BLK_SIZE: u32 = 6;
/// `VIRTIO_BLK_F_SCSI` (legacy).
pub const VIRTIO_BLK_F_SCSI: u32 = 7;
/// `VIRTIO_BLK_F_FLUSH`, called `VIRTIO_BLK_F_WCE` in legacy headers.
pub const VIRTIO_BLK_F_FLUSH: u32 = 9;
/// `VIRTIO_BLK_F_WCE`, the legacy name of [`VIRTIO_BLK_F_FLUSH`].
pub const VIRTIO_BLK_F_WCE: u32 = VIRTIO_BLK_F_FLUSH;
/// `VIRTIO_BLK_F_TOPOLOGY`.
pub const VIRTIO_BLK_F_TOPOLOGY: u32 = 10;
/// `VIRTIO_BLK_F_CONFIG_WCE`: the writeback flag in config space is writable.
pub const VIRTIO_BLK_F_CONFIG_WCE: u32 = 11;
/// `VIRTIO_BLK_F_MQ`.
pub const VIRTIO_BLK_F_MQ: u32 = 12;
/// `VIRTIO_BLK_F_DISCARD`.
pub const VIRTIO_BLK_F_DISCARD: u32 = 13;
/// `VIRTIO_BLK_F_WRITE_ZEROES`.
pub const VIRTIO_BLK_F_WRITE_ZEROES: u32 = 14;
/// `VIRTIO_BLK_F_SECURE_ERASE`.
pub const VIRTIO_BLK_F_SECURE_ERASE: u32 = 16;
/// `VIRTIO_BLK_F_ZONED`.
pub const VIRTIO_BLK_F_ZONED: u32 = 17;

/// `VIRTIO_BLK_T_IN`: read.
pub const VIRTIO_BLK_T_IN: u32 = 0;
/// `VIRTIO_BLK_T_OUT`: write, also the direction bit of other requests.
pub const VIRTIO_BLK_T_OUT: u32 = 1;
/// `VIRTIO_BLK_T_SCSI_CMD`.
pub const VIRTIO_BLK_T_SCSI_CMD: u32 = 2;
/// `VIRTIO_BLK_T_FLUSH`.
pub const VIRTIO_BLK_T_FLUSH: u32 = 4;
/// `VIRTIO_BLK_T_GET_ID`: read the serial number.
pub const VIRTIO_BLK_T_GET_ID: u32 = 8;
/// `VIRTIO_BLK_T_DISCARD`.
pub const VIRTIO_BLK_T_DISCARD: u32 = 11;
/// `VIRTIO_BLK_T_WRITE_ZEROES`.
pub const VIRTIO_BLK_T_WRITE_ZEROES: u32 = 13;
/// `VIRTIO_BLK_T_BARRIER`: legacy flag, ignored.
pub const VIRTIO_BLK_T_BARRIER: u32 = 0x8000_0000;

/// `VIRTIO_BLK_S_OK`.
pub const VIRTIO_BLK_S_OK: u8 = 0;
/// `VIRTIO_BLK_S_IOERR`.
pub const VIRTIO_BLK_S_IOERR: u8 = 1;
/// `VIRTIO_BLK_S_UNSUPP`.
pub const VIRTIO_BLK_S_UNSUPP: u8 = 2;

/// `VIRTIO_BLK_WRITE_ZEROES_FLAG_UNMAP`.
pub const VIRTIO_BLK_WRITE_ZEROES_FLAG_UNMAP: u32 = 1;

/// `VIRTIO_BLK_ID_BYTES`: length of the serial number string.
pub const VIRTIO_BLK_ID_BYTES: usize = 20;

/// `BDRV_SECTOR_SIZE`: the unit of sector numbers in requests.
pub const BDRV_SECTOR_SIZE: u64 = 512;
const BDRV_SECTOR_BITS: u32 = 9;

/// `BDRV_REQUEST_MAX_SECTORS`: the largest request the block layer accepts.
pub const BDRV_REQUEST_MAX_SECTORS: u32 = (i32::MAX as u32) >> BDRV_SECTOR_BITS;

/// `VIRTIO_BLK_MAX_CONFIG_SIZE`: `sizeof(struct virtio_blk_config)`.
pub const VIRTIO_BLK_MAX_CONFIG_SIZE: usize = 96;

/// Config offsets, `struct virtio_blk_config`.
const CFG_CAPACITY: usize = 0;
const CFG_SIZE_MAX: usize = 8;
const CFG_SEG_MAX: usize = 12;
const CFG_CYLINDERS: usize = 16;
const CFG_HEADS: usize = 18;
const CFG_SECTORS: usize = 19;
const CFG_BLK_SIZE: usize = 20;
const CFG_PHYSICAL_BLOCK_EXP: usize = 24;
const CFG_ALIGNMENT_OFFSET: usize = 25;
const CFG_MIN_IO_SIZE: usize = 26;
const CFG_OPT_IO_SIZE: usize = 28;
const CFG_WCE: usize = 32;
const CFG_NUM_QUEUES: usize = 34;
const CFG_MAX_DISCARD_SECTORS: usize = 36;
const CFG_MAX_DISCARD_SEG: usize = 40;
const CFG_DISCARD_SECTOR_ALIGNMENT: usize = 44;
const CFG_MAX_WRITE_ZEROES_SECTORS: usize = 48;
const CFG_MAX_WRITE_ZEROES_SEG: usize = 52;
const CFG_WRITE_ZEROES_MAY_UNMAP: usize = 56;

/// `virtio_blk_cfg_size_params`: the config space always reaches up to the discard fields, and
/// each of these features extends it to the end of its fields.
const CONFIG_SIZE_MIN: usize = CFG_MAX_DISCARD_SECTORS;
const CONFIG_FEATURE_SIZES: [(u32, usize); 4] = [
    (VIRTIO_BLK_F_DISCARD, 48),
    (VIRTIO_BLK_F_WRITE_ZEROES, 57),
    (VIRTIO_BLK_F_SECURE_ERASE, 72),
    (VIRTIO_BLK_F_ZONED, 96),
];

/// How much data a request moves through a bounce buffer at a time.
const CHUNK: usize = 64 * 1024;

/// The disk image behind a virtio-blk device.
///
/// All offsets and lengths are in bytes. Errors complete the request with
/// `VIRTIO_BLK_S_IOERR`.
pub trait BlockBackend: Send + fmt::Debug {
    /// The size of the image in bytes.
    fn size(&self) -> u64;

    /// Whether writes are allowed. A read-only image makes the device offer `VIRTIO_BLK_F_RO`.
    fn is_writable(&self) -> bool {
        true
    }

    /// Fills `buf` from `offset`.
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()>;

    /// Writes `buf` at `offset`.
    fn write_at(&mut self, offset: u64, buf: &[u8]) -> io::Result<()>;

    /// Makes earlier writes durable.
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }

    /// Tells the image `len` bytes at `offset` are no longer needed. Doing nothing is valid.
    fn discard(&mut self, _offset: u64, _len: u64) -> io::Result<()> {
        Ok(())
    }

    /// Zeroes `len` bytes at `offset`. `may_unmap` allows doing it by deallocating. The default
    /// writes zero buffers.
    fn write_zeroes(&mut self, offset: u64, len: u64, _may_unmap: bool) -> io::Result<()> {
        let zeroes = vec![0; CHUNK];
        let mut done = 0;
        while done < len {
            let n = (len - done).min(CHUNK as u64) as usize;
            self.write_at(offset + done, &zeroes[..n])?;
            done += n as u64;
        }
        Ok(())
    }
}

/// A disk image held in memory, shared so that tests and the host can look at it while the
/// device owns the backend.
#[derive(Clone, Debug)]
pub struct MemBlockBackend {
    data: Arc<Mutex<Vec<u8>>>,
    writable: bool,
    flushes: Arc<Mutex<u64>>,
}

impl MemBlockBackend {
    /// A zero filled writable image of `size` bytes.
    pub fn new(size: usize) -> Self {
        Self::from_vec(vec![0; size])
    }

    /// A writable image with the given contents.
    pub fn from_vec(data: Vec<u8>) -> Self {
        MemBlockBackend {
            data: Arc::new(Mutex::new(data)),
            writable: true,
            flushes: Arc::new(Mutex::new(0)),
        }
    }

    /// The same image, read-only.
    pub fn read_only(mut self) -> Self {
        self.writable = false;
        self
    }

    /// The image contents.
    pub fn data(&self) -> Arc<Mutex<Vec<u8>>> {
        Arc::clone(&self.data)
    }

    /// How many flushes the image has seen.
    pub fn flush_count(&self) -> u64 {
        *self.flushes.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn range(&self, offset: u64, len: usize) -> io::Result<std::ops::Range<usize>> {
        let size = self.data.lock().unwrap_or_else(PoisonError::into_inner).len();
        let start = usize::try_from(offset).ok().filter(|&s| s <= size);
        match start.and_then(|s| s.checked_add(len).filter(|&e| e <= size).map(|e| s..e)) {
            Some(r) => Ok(r),
            None => Err(io::Error::new(io::ErrorKind::InvalidInput, "access past end of image")),
        }
    }
}

impl BlockBackend for MemBlockBackend {
    fn size(&self) -> u64 {
        self.data.lock().unwrap_or_else(PoisonError::into_inner).len() as u64
    }

    fn is_writable(&self) -> bool {
        self.writable
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let r = self.range(offset, buf.len())?;
        buf.copy_from_slice(&self.data.lock().unwrap_or_else(PoisonError::into_inner)[r]);
        Ok(())
    }

    fn write_at(&mut self, offset: u64, buf: &[u8]) -> io::Result<()> {
        if !self.writable {
            return Err(io::Error::new(io::ErrorKind::PermissionDenied, "read-only image"));
        }
        let r = self.range(offset, buf.len())?;
        self.data.lock().unwrap_or_else(PoisonError::into_inner)[r].copy_from_slice(buf);
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        *self.flushes.lock().unwrap_or_else(PoisonError::into_inner) += 1;
        Ok(())
    }

    fn discard(&mut self, offset: u64, len: u64) -> io::Result<()> {
        if !self.writable {
            return Err(io::Error::new(io::ErrorKind::PermissionDenied, "read-only image"));
        }
        // Discarded blocks read back as zeroes here, which the specification allows.
        self.write_zeroes(offset, len, true)
    }
}

/// The virtio-blk properties, with QEMU's defaults.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VirtioBlkConf {
    /// `serial`: what `VIRTIO_BLK_T_GET_ID` returns.
    pub serial: Option<String>,
    /// `config-wce`: offer `VIRTIO_BLK_F_CONFIG_WCE`.
    pub config_wce: bool,
    /// `discard`: offer `VIRTIO_BLK_F_DISCARD`.
    pub discard: bool,
    /// `write-zeroes`: offer `VIRTIO_BLK_F_WRITE_ZEROES`.
    pub write_zeroes: bool,
    /// `report-discard-granularity`.
    pub report_discard_granularity: bool,
    /// `num-queues`: `None` is QEMU's automatic choice, which is 1 for this transport.
    pub num_queues: Option<u16>,
    /// `queue-size`.
    pub queue_size: u16,
    /// `seg-max-adjust`: report `queue_size - 2` segments instead of 126.
    pub seg_max_adjust: bool,
    /// `max-discard-sectors`.
    pub max_discard_sectors: u32,
    /// `max-write-zeroes-sectors`.
    pub max_write_zeroes_sectors: u32,
    /// `x-enable-wce-if-config-wce`.
    pub x_enable_wce_if_config_wce: bool,
    /// `write-cache`: whether the write cache starts enabled.
    pub write_cache: bool,
    /// `logical_block_size`, 0 for 512.
    pub logical_block_size: u32,
    /// `physical_block_size`, 0 for 512.
    pub physical_block_size: u32,
    /// `min_io_size`.
    pub min_io_size: u32,
    /// `opt_io_size`.
    pub opt_io_size: u32,
    /// `discard_granularity`, `u32::MAX` (QEMU's -1) for the logical block size.
    pub discard_granularity: u32,
    /// `cyls`, 0 to guess the geometry.
    pub cyls: u32,
    /// `heads`, 0 to guess the geometry.
    pub heads: u32,
    /// `secs`, 0 to guess the geometry.
    pub secs: u32,
}

impl Default for VirtioBlkConf {
    fn default() -> Self {
        VirtioBlkConf {
            serial: None,
            config_wce: true,
            discard: true,
            write_zeroes: true,
            report_discard_granularity: true,
            num_queues: None,
            queue_size: 256,
            seg_max_adjust: true,
            max_discard_sectors: BDRV_REQUEST_MAX_SECTORS,
            max_write_zeroes_sectors: BDRV_REQUEST_MAX_SECTORS,
            x_enable_wce_if_config_wce: true,
            write_cache: true,
            logical_block_size: 0,
            physical_block_size: 0,
            min_io_size: 0,
            opt_io_size: 0,
            discard_granularity: u32::MAX,
            cyls: 0,
            heads: 0,
            secs: 0,
        }
    }
}

/// The virtio-blk device model, `VirtIOBlock`.
#[derive(Debug)]
pub struct VirtioBlk {
    conf: VirtioBlkConf,
    backend: Box<dyn BlockBackend>,
    host_features: u64,
    config_size: usize,
    sector_mask: u64,
    num_queues: u16,
    wce: bool,
    original_wce: bool,
}

/// What handling one request came to: a status to complete it with, or a device error that
/// was already reported and leaves the request detached.
type Outcome = std::result::Result<u8, ()>;

fn put_u16(cfg: &mut [u8], off: usize, v: u16) {
    cfg[off..off + 2].copy_from_slice(&v.to_le_bytes());
}

fn put_u32(cfg: &mut [u8], off: usize, v: u32) {
    cfg[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_u64(cfg: &mut [u8], off: usize, v: u64) {
    cfg[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

impl VirtioBlk {
    /// A device on `backend` with properties `conf`. The properties are checked when the device
    /// is realized.
    pub fn new(backend: Box<dyn BlockBackend>, conf: VirtioBlkConf) -> Self {
        let wce = conf.write_cache;
        VirtioBlk {
            conf,
            backend,
            host_features: 0,
            config_size: 0,
            sector_mask: 0,
            num_queues: 0,
            wce,
            original_wce: wce,
        }
    }

    /// The properties, with the block sizes and geometry filled in once realized.
    pub fn conf(&self) -> &VirtioBlkConf {
        &self.conf
    }

    /// The backend.
    pub fn backend(&self) -> &dyn BlockBackend {
        &*self.backend
    }

    /// Whether the write cache is on, `blk_enable_write_cache()`.
    pub fn write_cache_enabled(&self) -> bool {
        self.wce
    }

    fn total_sectors(&self) -> u64 {
        self.backend.size() >> BDRV_SECTOR_BITS
    }

    fn logical_block_size(&self) -> u32 {
        self.conf.logical_block_size
    }

    /// `blkconf_blocksizes()` without probing the image.
    fn check_blocksizes(&mut self) -> Result<()> {
        let c = &mut self.conf;
        if c.physical_block_size == 0 {
            c.physical_block_size = BDRV_SECTOR_SIZE as u32;
        }
        if c.logical_block_size == 0 {
            c.logical_block_size = BDRV_SECTOR_SIZE as u32;
        }
        for (name, v) in [
            ("logical_block_size", c.logical_block_size),
            ("physical_block_size", c.physical_block_size),
        ] {
            if !v.is_power_of_two() || !(512..=2 * 1024 * 1024).contains(&v) {
                return Err(Error::generic(format!(
                    "Property {TYPE_VIRTIO_BLK}.{name} doesn't take value {v} (a power of two between 512 B and 2 MiB)"
                )));
            }
        }
        let lbs = c.logical_block_size;
        if lbs > c.physical_block_size {
            return Err(Error::generic("logical_block_size > physical_block_size not supported"));
        }
        if c.min_io_size % lbs != 0 {
            return Err(Error::generic("min_io_size must be a multiple of logical_block_size"));
        }
        if c.min_io_size / lbs > u32::from(u16::MAX) {
            return Err(Error::generic(format!(
                "min_io_size must not exceed {} logical blocks",
                u16::MAX
            )));
        }
        if c.opt_io_size % lbs != 0 {
            return Err(Error::generic("opt_io_size must be a multiple of logical_block_size"));
        }
        if c.discard_granularity != u32::MAX && c.discard_granularity % lbs != 0 {
            return Err(Error::generic(
                "discard_granularity must be a multiple of logical_block_size",
            ));
        }
        Ok(())
    }

    /// `blkconf_geometry()` with the size based guess.
    fn check_geometry(&mut self) -> Result<()> {
        let total = self.total_sectors();
        let c = &mut self.conf;
        if c.cyls == 0 && c.heads == 0 && c.secs == 0 {
            c.cyls = (total / (16 * 63)).clamp(2, 16383) as u32;
            c.heads = 16;
            c.secs = 63;
        }
        if !(1..=65535).contains(&c.cyls) {
            return Err(Error::generic("cyls must be between 1 and 65535"));
        }
        if !(1..=255).contains(&c.heads) {
            return Err(Error::generic("heads must be between 1 and 255"));
        }
        if !(1..=255).contains(&c.secs) {
            return Err(Error::generic("secs must be between 1 and 255"));
        }
        Ok(())
    }

    /// `virtio_blk_update_config()`: the full config structure.
    fn update_config(&self) -> [u8; VIRTIO_BLK_MAX_CONFIG_SIZE] {
        let c = &self.conf;
        let blk_size = c.logical_block_size;
        let mut cfg = [0u8; VIRTIO_BLK_MAX_CONFIG_SIZE];
        put_u64(&mut cfg, CFG_CAPACITY, self.total_sectors());
        put_u32(&mut cfg, CFG_SIZE_MAX, 0);
        let seg_max = if c.seg_max_adjust { u32::from(c.queue_size) - 2 } else { 128 - 2 };
        put_u32(&mut cfg, CFG_SEG_MAX, seg_max);
        put_u16(&mut cfg, CFG_CYLINDERS, c.cyls as u16);
        put_u32(&mut cfg, CFG_BLK_SIZE, blk_size);
        put_u16(&mut cfg, CFG_MIN_IO_SIZE, (c.min_io_size / blk_size) as u16);
        put_u32(&mut cfg, CFG_OPT_IO_SIZE, c.opt_io_size / blk_size);
        cfg[CFG_HEADS] = c.heads as u8;
        // The capacity must be a whole number of logical blocks per track. Where it is not,
        // round the sectors per track down to a whole block.
        let length = self.backend.size();
        cfg[CFG_SECTORS] = if length > 0
            && length / u64::from(c.heads) / u64::from(c.secs) % u64::from(blk_size) != 0
        {
            (u64::from(c.secs) & !self.sector_mask) as u8
        } else {
            c.secs as u8
        };
        let mut exp = 0u8;
        let mut size = c.physical_block_size;
        while size > c.logical_block_size {
            exp += 1;
            size >>= 1;
        }
        cfg[CFG_PHYSICAL_BLOCK_EXP] = exp;
        cfg[CFG_ALIGNMENT_OFFSET] = 0;
        cfg[CFG_WCE] = u8::from(self.wce);
        put_u16(&mut cfg, CFG_NUM_QUEUES, self.num_queues);
        if has_feature(self.host_features, VIRTIO_BLK_F_DISCARD) {
            let granularity = if c.discard_granularity == u32::MAX || !c.report_discard_granularity
            {
                blk_size
            } else {
                c.discard_granularity
            };
            put_u32(&mut cfg, CFG_MAX_DISCARD_SECTORS, c.max_discard_sectors);
            put_u32(&mut cfg, CFG_DISCARD_SECTOR_ALIGNMENT, granularity >> BDRV_SECTOR_BITS);
            // Only one segment per request, as in QEMU.
            put_u32(&mut cfg, CFG_MAX_DISCARD_SEG, 1);
        }
        if has_feature(self.host_features, VIRTIO_BLK_F_WRITE_ZEROES) {
            put_u32(&mut cfg, CFG_MAX_WRITE_ZEROES_SECTORS, c.max_write_zeroes_sectors);
            cfg[CFG_WRITE_ZEROES_MAY_UNMAP] = 1;
            put_u32(&mut cfg, CFG_MAX_WRITE_ZEROES_SEG, 1);
        }
        cfg
    }

    /// `virtio_blk_sect_range_ok()`.
    fn sect_range_ok(&self, sector: u64, size: u64) -> bool {
        let nb_sectors = size >> BDRV_SECTOR_BITS;
        if nb_sectors > u64::from(BDRV_REQUEST_MAX_SECTORS) {
            return false;
        }
        if sector & self.sector_mask != 0 {
            return false;
        }
        if size % u64::from(self.logical_block_size()) != 0 {
            return false;
        }
        let total = self.total_sectors();
        sector <= total && nb_sectors <= total - sector
    }

    fn do_read<M: GuestMemory + ?Sized>(
        &mut self,
        vdev: &mut VirtIODevice,
        sector: u64,
        size: u64,
        w: &mut Writer<'_, M>,
    ) -> Outcome {
        if !self.sect_range_ok(sector, size) {
            return Ok(VIRTIO_BLK_S_IOERR);
        }
        let mut buf = vec![0; (size as usize).min(CHUNK)];
        let mut done = 0u64;
        while done < size {
            let n = (size - done).min(CHUNK as u64) as usize;
            if self.backend.read_at((sector << BDRV_SECTOR_BITS) + done, &mut buf[..n]).is_err() {
                return Ok(VIRTIO_BLK_S_IOERR);
            }
            if let Err(e) = w.write_all(&buf[..n]) {
                vdev.error(&format!("virtio-blk: {e}"));
                return Err(());
            }
            done += n as u64;
        }
        Ok(VIRTIO_BLK_S_OK)
    }

    fn do_write<M: GuestMemory + ?Sized>(
        &mut self,
        vdev: &mut VirtIODevice,
        sector: u64,
        size: u64,
        r: &mut Reader<'_, M>,
    ) -> Outcome {
        if !self.sect_range_ok(sector, size) {
            return Ok(VIRTIO_BLK_S_IOERR);
        }
        if !self.backend.is_writable() {
            return Ok(VIRTIO_BLK_S_IOERR);
        }
        let mut buf = vec![0; (size as usize).min(CHUNK)];
        let mut done = 0u64;
        while done < size {
            let n = (size - done).min(CHUNK as u64) as usize;
            if let Err(e) = r.read_exact(&mut buf[..n]) {
                vdev.error(&format!("virtio-blk: {e}"));
                return Err(());
            }
            if self.backend.write_at((sector << BDRV_SECTOR_BITS) + done, &buf[..n]).is_err() {
                return Ok(VIRTIO_BLK_S_IOERR);
            }
            done += n as u64;
        }
        if !self.wce && self.backend.flush().is_err() {
            return Ok(VIRTIO_BLK_S_IOERR);
        }
        Ok(VIRTIO_BLK_S_OK)
    }

    /// `virtio_blk_handle_discard_write_zeroes()`.
    fn do_discard_write_zeroes(&mut self, hdr: &[u8; 16], is_write_zeroes: bool) -> u8 {
        let sector = u64::from_le_bytes(hdr[0..8].try_into().unwrap_or_default());
        let num_sectors = u32::from_le_bytes(hdr[8..12].try_into().unwrap_or_default());
        let flags = u32::from_le_bytes(hdr[12..16].try_into().unwrap_or_default());
        let max_sectors = if is_write_zeroes {
            self.conf.max_write_zeroes_sectors
        } else {
            self.conf.max_discard_sectors
        };
        if num_sectors > max_sectors {
            return VIRTIO_BLK_S_IOERR;
        }
        let bytes = u64::from(num_sectors) << BDRV_SECTOR_BITS;
        if !self.sect_range_ok(sector, bytes) {
            return VIRTIO_BLK_S_IOERR;
        }
        if flags & !VIRTIO_BLK_WRITE_ZEROES_FLAG_UNMAP != 0 {
            return VIRTIO_BLK_S_UNSUPP;
        }
        let unmap = flags & VIRTIO_BLK_WRITE_ZEROES_FLAG_UNMAP != 0;
        if !is_write_zeroes && unmap {
            return VIRTIO_BLK_S_UNSUPP;
        }
        if !self.backend.is_writable() {
            return VIRTIO_BLK_S_IOERR;
        }
        let offset = sector << BDRV_SECTOR_BITS;
        let ret = if is_write_zeroes {
            self.backend.write_zeroes(offset, bytes, unmap)
        } else {
            self.backend.discard(offset, bytes)
        };
        if ret.is_ok() { VIRTIO_BLK_S_OK } else { VIRTIO_BLK_S_IOERR }
    }

    /// `virtio_blk_handle_request()` followed by `virtio_blk_req_complete()`. An `Err` means a
    /// device error was raised and the chain must be detached.
    fn handle_request(
        &mut self,
        vdev: &mut VirtIODevice,
        q: u16,
        chain: &DescriptorChain,
    ) -> std::result::Result<(), ()> {
        let (Some(_), Some(last)) = (chain.readable().first(), chain.writable().last().copied())
        else {
            vdev.error("virtio-blk missing headers");
            return Err(());
        };
        if chain.readable_len() < 16 {
            vdev.error("virtio-blk request outhdr too short");
            return Err(());
        }
        let mem = Arc::clone(vdev.mem());
        let mut reader = chain.reader(&*mem);
        let mut hdr = [0u8; 16];
        if let Err(e) = reader.read_exact(&mut hdr) {
            vdev.error(&format!("virtio-blk: {e}"));
            return Err(());
        }
        if last.is_empty() {
            vdev.error("virtio-blk request inhdr too short");
            return Err(());
        }
        // The status byte is always the last byte of the last device writable buffer.
        let in_len = chain.writable_len();
        let status_addr = last.addr().wrapping_add(u64::from(last.len()) - 1);
        let data_in = in_len - 1;
        let data_out = chain.readable_len() - 16;
        let mut writer = chain.writer(&*mem);

        let ty = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
        let sector = u64::from_le_bytes(hdr[8..16].try_into().unwrap_or_default());
        let status = match ty & !(VIRTIO_BLK_T_OUT | VIRTIO_BLK_T_BARRIER) {
            VIRTIO_BLK_T_IN => {
                if ty & VIRTIO_BLK_T_OUT != 0 {
                    self.do_write(vdev, sector, data_out, &mut reader)?
                } else {
                    self.do_read(vdev, sector, data_in, &mut writer)?
                }
            }
            VIRTIO_BLK_T_FLUSH => {
                if self.backend.flush().is_ok() {
                    VIRTIO_BLK_S_OK
                } else {
                    VIRTIO_BLK_S_IOERR
                }
            }
            VIRTIO_BLK_T_GET_ID => {
                // The string is NUL terminated only when shorter than the buffer.
                let mut id = self.conf.serial.clone().unwrap_or_default().into_bytes();
                id.push(0);
                let size = id.len().min(data_in as usize).min(VIRTIO_BLK_ID_BYTES);
                if let Err(e) = writer.write_all(&id[..size]) {
                    vdev.error(&format!("virtio-blk: {e}"));
                    return Err(());
                }
                VIRTIO_BLK_S_OK
            }
            t if t == VIRTIO_BLK_T_DISCARD & !VIRTIO_BLK_T_OUT
                || t == VIRTIO_BLK_T_WRITE_ZEROES & !VIRTIO_BLK_T_OUT =>
            {
                let is_write_zeroes = ty & !VIRTIO_BLK_T_BARRIER == VIRTIO_BLK_T_WRITE_ZEROES;
                if ty & VIRTIO_BLK_T_OUT == 0 || data_out > 16 {
                    VIRTIO_BLK_S_UNSUPP
                } else {
                    let mut dwz = [0u8; 16];
                    if data_out < 16 || reader.read_exact(&mut dwz).is_err() {
                        vdev.error("virtio-blk discard/write_zeroes header too short");
                        return Err(());
                    }
                    self.do_discard_write_zeroes(&dwz, is_write_zeroes)
                }
            }
            // SCSI passthrough, zoned commands and anything unknown.
            _ => VIRTIO_BLK_S_UNSUPP,
        };

        if let Err(e) = mem.write(status_addr, &[status]) {
            vdev.error(&format!("virtio-blk: {e}"));
        }
        vdev.push(q, chain, in_len as u32);
        vdev.notify(q);
        Ok(())
    }

    /// `virtio_blk_handle_vq()`.
    fn handle_vq(&mut self, vdev: &mut VirtIODevice, q: u16) {
        let suppress_notifications = vdev.queue_notification(q);
        loop {
            if suppress_notifications {
                vdev.set_queue_notification(q, false);
            }
            while let Some(chain) = vdev.pop(q) {
                if self.handle_request(vdev, q, &chain).is_err() {
                    vdev.detach(q, &chain);
                    break;
                }
            }
            if suppress_notifications {
                vdev.set_queue_notification(q, true);
            }
            if vdev.queue_empty(q) {
                break;
            }
        }
    }
}

impl VirtioDeviceClass for VirtioBlk {
    fn realize(&mut self, vdev: &mut VirtIODevice) -> Result<()> {
        let num_queues = self.conf.num_queues.unwrap_or(1);
        if num_queues == 0 {
            return Err(Error::generic("num-queues property must be larger than 0"));
        }
        let qs = self.conf.queue_size;
        if qs <= 2 {
            return Err(Error::generic(format!("invalid queue-size property ({qs}), must be > 2")));
        }
        if !qs.is_power_of_two() || qs > 1024 {
            return Err(Error::generic(format!(
                "invalid queue-size property ({qs}), must be a power of 2 (max 1024)"
            )));
        }
        self.wce = self.conf.write_cache;
        self.original_wce = self.wce;
        self.check_geometry()?;
        self.check_blocksizes()?;

        let mut hf = 0;
        if self.conf.config_wce {
            hf |= feature(VIRTIO_BLK_F_CONFIG_WCE);
        }
        if self.conf.discard {
            hf |= feature(VIRTIO_BLK_F_DISCARD);
        }
        if self.conf.write_zeroes {
            hf |= feature(VIRTIO_BLK_F_WRITE_ZEROES);
        }
        self.host_features = hf;
        let max = BDRV_REQUEST_MAX_SECTORS;
        let mds = self.conf.max_discard_sectors;
        if has_feature(hf, VIRTIO_BLK_F_DISCARD) && (mds == 0 || mds > max) {
            return Err(Error::generic(format!(
                "invalid max-discard-sectors property ({mds}), must be between 1 and {max}"
            )));
        }
        let mwz = self.conf.max_write_zeroes_sectors;
        if has_feature(hf, VIRTIO_BLK_F_WRITE_ZEROES) && (mwz == 0 || mwz > max) {
            return Err(Error::generic(format!(
                "invalid max-write-zeroes-sectors property ({mwz}), must be between 1 and {max}"
            )));
        }

        let mut config_size = CONFIG_SIZE_MIN;
        for (bit, end) in CONFIG_FEATURE_SIZES {
            if has_feature(hf, bit) {
                config_size = config_size.max(end);
            }
        }
        self.config_size = config_size;
        vdev.init(TYPE_VIRTIO_BLK, VIRTIO_ID_BLOCK, config_size);
        self.sector_mask = u64::from(self.conf.logical_block_size) / BDRV_SECTOR_SIZE - 1;
        for _ in 0..num_queues {
            vdev.add_queue(qs)?;
        }
        self.num_queues = num_queues;
        Ok(())
    }

    fn get_features(&mut self, _vdev: &VirtIODevice, features: u64) -> Result<u64> {
        let mut f = features | self.host_features;
        f |= feature(VIRTIO_BLK_F_SEG_MAX);
        f |= feature(VIRTIO_BLK_F_GEOMETRY);
        f |= feature(VIRTIO_BLK_F_TOPOLOGY);
        f |= feature(VIRTIO_BLK_F_BLK_SIZE);
        if !has_feature(f, VIRTIO_F_VERSION_1) {
            f &= !feature(VIRTIO_F_ANY_LAYOUT);
            // Added for historical reasons, as in QEMU.
            f |= feature(VIRTIO_BLK_F_SCSI);
        }
        if self.wce
            || (self.conf.x_enable_wce_if_config_wce && has_feature(f, VIRTIO_BLK_F_CONFIG_WCE))
        {
            f |= feature(VIRTIO_BLK_F_WCE);
        }
        if !self.backend.is_writable() {
            f |= feature(VIRTIO_BLK_F_RO);
        }
        if self.num_queues > 1 {
            f |= feature(VIRTIO_BLK_F_MQ);
        }
        Ok(f)
    }

    fn get_config(&mut self, _vdev: &VirtIODevice, config: &mut [u8]) {
        let cfg = self.update_config();
        let n = config.len().min(cfg.len());
        config[..n].copy_from_slice(&cfg[..n]);
    }

    fn set_config(&mut self, _vdev: &mut VirtIODevice, config: &mut [u8]) {
        if let Some(&wce) = config.get(CFG_WCE) {
            self.wce = wce != 0;
        }
    }

    fn set_status(&mut self, vdev: &mut VirtIODevice, status: u8) -> Result<()> {
        if status & VIRTIO_CONFIG_S_DRIVER_OK == 0 {
            return Ok(());
        }
        // A guest that supports VIRTIO_BLK_F_CONFIG_WCE must be able to send a flush, and it
        // chooses the cache mode through config space. Guests without it get writeback only
        // if they negotiated flushes.
        if !vdev.has_feature(VIRTIO_BLK_F_CONFIG_WCE) {
            self.wce = vdev.has_feature(VIRTIO_BLK_F_WCE);
        }
        Ok(())
    }

    fn reset(&mut self, _vdev: &mut VirtIODevice) {
        self.wce = self.original_wce;
    }

    fn handle_output(&mut self, vdev: &mut VirtIODevice, queue: u16) {
        self.handle_vq(vdev, queue);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
