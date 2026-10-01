// SPDX-License-Identifier: GPL-2.0-or-later

//! The `qcow` format driver from block/qcow.c: version 1 of the QEMU copy-on-write format,
//! with compression, legacy AES encryption and a backing file.
//!
//! A 48 byte big-endian header gives the virtual size, the cluster size (`cluster_bits`, 512
//! bytes to 64 KiB), the number of entries of an L2 table (`l2_bits`), the encryption method
//! and where the L1 table is. The optional backing file name follows the header. Every L1
//! entry is the file offset of an L2 table or zero; every L2 entry is the file offset of a
//! data cluster, zero for an unallocated one, or a compressed cluster: bit 63 set, the
//! compressed size in the bits above `63 - cluster_bits` and the offset below. Compressed data
//! is raw deflate with a 4 KiB window. New L2 tables and clusters are put at the end of the
//! file, rounded up to the cluster size, exactly as QEMU does, so images written here and by
//! QEMU have the same layout. Sixteen L2 tables are cached with the same least frequently
//! used replacement QEMU has, and the last decompressed cluster is kept.
//!
//! An unallocated cluster reads from the backing file, or as zeroes without one. A write to
//! part of a compressed cluster decompresses it into a new cluster first. A write to part of a
//! new cluster of an encrypted image fills the rest of the cluster with encrypted zeroes.
//!
//! Encryption is the legacy qcow AES-CBC scheme of ruvm-crypto's [`QCryptoBlock`], opened with
//! the `encrypt.key-secret` option (`encrypt.format` is `aes`, the only choice). QEMU refuses
//! to open AES images in system emulators (`bdrv_uses_whitelist()`) and only lets the tools
//! convert them. ruvm has no system emulator block whitelist yet, so [`uses_whitelist`] is
//! false as it is in `qemu-img` and `qemu-io`, and the refusal and its message are there for
//! when that changes.
//!
//! Differences from QEMU:
//!
//! - QEMU registers a migration blocker for every open qcow node; there is no migration here.
//! - QEMU drops its lock around the data transfer of a request. Here reads drop it too, while
//!   writes and compressed writes hold it to the end, which only serialises writes.
//! - Compression uses flate2 with memory level 8, where QEMU asks zlib for level 9. The stream
//!   format and the window are the same and either side reads the other's clusters, but the
//!   compressed bytes, and so the file layout after a compressed write, may differ.
//! - When an L2 table cannot be read, QEMU leaves the cache slot it was loading into with the
//!   old offset but clobbered contents. Here the table is read into a separate buffer first
//!   and the slot is only replaced on success.
//! - A failed write in `blockdev-create` gives the `strerror()` text, as QEMU's job does, and
//!   "Could not create image: ..." from `qemu-img create`, as QEMU's `bdrv_co_create()` does.
//! - Like QEMU, a compressed write to a cluster that is already allocated writes the
//!   compressed bytes over it without updating its L2 entry. Compressed writes are only meant
//!   for new images (`qemu-img convert -c`); this is kept as it is.

use std::io;
use std::sync::{Mutex, MutexGuard};

use flate2::{Compress, Compression, Decompress, FlushCompress, FlushDecompress, Status};
use ruvm_base::{Error, Result};
use ruvm_crypto::block::{QCRYPTO_BLOCK_OPEN_NO_IO, QCryptoBlock, QCryptoBlockIo};
use ruvm_qapi::QDict;
use ruvm_qapi::opts::QemuOptDesc;
use ruvm_qapi::types::{
    BlockdevCreateOptionsQcow, BlockdevCreateOptionsU, BlockdevOptionsU, BlockdevQcowEncryptionU,
    BlockdevRef, PreallocMode, QCryptoBlockFormat, QCryptoBlockOpenOptions,
    QCryptoBlockOpenOptionsU, QCryptoBlockOptionsQCow,
};

use crate::drivers::{DriverDef, OpenArgs, find_format};
use crate::graph::BlockGraph;
use crate::imgopts::{create_opts_open, opt_desc, take_str, visit_create_options};
use crate::node::{
    BDRV_BLOCK_COMPRESSED, BDRV_BLOCK_DATA, BDRV_BLOCK_OFFSET_VALID, BDRV_CHILD_IMAGE,
    BDRV_CHILD_PRIMARY, BDRV_SECTOR_SIZE, BlockDriverInfo, BlockLimits, BlockStatus, Driver, Node,
    ReopenState, errno,
};

/// `bdrv_qcow`.
pub(crate) static QCOW: DriverDef = DriverDef::format("qcow", qcow_open_node)
    .with_probe(qcow_probe)
    .with_create_opts(qcow_co_create_opts)
    .with_create_opts_list(&QCOW_CREATE_OPTS)
    .with_create(qcow_co_create)
    .with_backing()
    .with_strong_opts(&["encrypt.key-secret"]);

/// `qcow_create_opts`, in QEMU's order.
static QCOW_CREATE_OPTS: [QemuOptDesc; 6] = [
    opt_desc!("size", Size, "Virtual disk size"),
    opt_desc!("backing_file", String, "File name of a base image"),
    opt_desc!("backing_fmt", String, "Format of the backing image"),
    opt_desc!(
        "encryption",
        Bool,
        "Encrypt the image with format 'aes'. (Deprecated in favor of encrypt.format=aes)"
    ),
    opt_desc!("encrypt.format", String, "Encrypt the image, format choices: 'aes'"),
    opt_desc!(
        "encrypt.key-secret",
        String,
        "ID of the secret that provides the AES encryption key"
    ),
];

/// `QCOW_MAGIC`: "QFI\xfb".
const QCOW_MAGIC: u32 = u32::from_be_bytes([b'Q', b'F', b'I', 0xfb]);
const QCOW_VERSION: u32 = 1;

const QCOW_CRYPT_NONE: u32 = 0;
const QCOW_CRYPT_AES: u32 = 1;

const QCOW_OFLAG_COMPRESSED: u64 = 1 << 63;

const L2_CACHE_SIZE: usize = 16;

/// `sizeof(QCowHeader)`.
const HEADER_SIZE: usize = 48;

/// The fields of `QCowHeader`, as byte offsets.
mod hdr {
    pub(super) const MAGIC: usize = 0;
    pub(super) const VERSION: usize = 4;
    pub(super) const BACKING_FILE_OFFSET: usize = 8;
    pub(super) const BACKING_FILE_SIZE: usize = 16;
    pub(super) const SIZE: usize = 24;
    pub(super) const CLUSTER_BITS: usize = 32;
    pub(super) const L2_BITS: usize = 33;
    pub(super) const CRYPT_METHOD: usize = 36;
    pub(super) const L1_TABLE_OFFSET: usize = 40;
}

/// The deflate window of compressed clusters: 4 KiB, no zlib header.
const WINDOW_BITS: u8 = 12;

fn be32(b: &[u8], off: usize) -> u32 {
    u32::from_be_bytes(b[off..off + 4].try_into().unwrap())
}

fn be64(b: &[u8], off: usize) -> u64 {
    u64::from_be_bytes(b[off..off + 8].try_into().unwrap())
}

fn put_be32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_be_bytes());
}

fn put_be64(b: &mut [u8], off: usize, v: u64) {
    b[off..off + 8].copy_from_slice(&v.to_be_bytes());
}

/// `bdrv_uses_whitelist()`: whether this is a system emulator that refuses drivers and
/// features outside its whitelist. Only the tools exist here so far.
fn uses_whitelist() -> bool {
    false
}

/// `qcow_probe()`.
fn qcow_probe(buf: &[u8], _filename: Option<&str>) -> i32 {
    if buf.len() >= HEADER_SIZE
        && be32(buf, hdr::MAGIC) == QCOW_MAGIC
        && be32(buf, hdr::VERSION) == QCOW_VERSION
    {
        100
    } else {
        0
    }
}

/// The `-errno` a failed `bdrv_pread()` in `qcow_open()` returns, as `bdrv_open_driver()`
/// reports it.
fn open_err(file: &Node, e: io::Error) -> Error {
    let name = file.filename().unwrap_or_default();
    Error::from_io(format!("Could not open '{name}'"), e)
}

/// The errno of a block layer error, for the callbacks that return `-errno`.
fn to_io(e: Error) -> io::Error {
    let code = std::error::Error::source(&e)
        .and_then(|c| c.downcast_ref::<io::Error>())
        .and_then(io::Error::raw_os_error);
    match code {
        Some(c) => errno(c),
        None => io::Error::other(e.message().to_string()),
    }
}

/// `bdrv_co_pwrite_sync()`: a write and a flush of the node.
fn pwrite_sync(file: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
    file.pwrite(offset, buf)?;
    file.flush()
}

/// Header access for `qcrypto_block_open()` and `qcrypto_block_create()`, which QEMU calls
/// with NULL callbacks: the legacy AES scheme has no header of its own.
struct NoHeaderIo;

impl QCryptoBlockIo for NoHeaderIo {
    fn read(&mut self, _offset: u64, _buf: &mut [u8]) -> Result<()> {
        Err(Error::generic("Could not read encryption header"))
    }

    fn write(&mut self, _offset: u64, _buf: &[u8]) -> Result<()> {
        Err(Error::generic("Could not write encryption header"))
    }
}

/// The mutable part of `BDRVQcowState`, under `s->lock`.
struct State {
    l1_table: Vec<u64>,
    /// `L2_CACHE_SIZE` tables of `l2_size` entries, decoded.
    l2_cache: Vec<u64>,
    l2_cache_offsets: [u64; L2_CACHE_SIZE],
    l2_cache_counts: [u32; L2_CACHE_SIZE],
    /// The last decompressed cluster.
    cluster_cache: Vec<u8>,
    /// Compressed data as read from the file.
    cluster_data: Vec<u8>,
    /// The file offset of the compressed cluster in `cluster_cache`, `u64::MAX` for none.
    cluster_cache_offset: u64,
}

/// The `allocate` argument of `get_cluster_offset()`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Allocate {
    /// 0: look up only.
    No,
    /// 1: allocate a normal cluster, of which bytes `n_start..n_end` are about to be written.
    Normal { n_start: u64, n_end: u64 },
    /// 2: allocate a compressed cluster of this many bytes.
    Compressed(u64),
}

/// `BDRVQcowState`.
pub(crate) struct QcowDriver {
    cluster_bits: u32,
    cluster_size: u64,
    l2_bits: u32,
    l2_size: usize,
    l1_size: usize,
    cluster_offset_mask: u64,
    l1_table_offset: u64,
    /// `bs->total_sectors`.
    total_sectors: u64,
    /// `s->crypto`: set for AES images, even when opened without I/O.
    crypto: Option<QCryptoBlock>,
    lock: Mutex<State>,
}

impl std::fmt::Debug for QcowDriver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QcowDriver")
            .field("cluster_bits", &self.cluster_bits)
            .field("l2_bits", &self.l2_bits)
            .field("l1_size", &self.l1_size)
            .field("encrypted", &self.crypto.is_some())
            .finish_non_exhaustive()
    }
}

/// `qcow_open()`.
fn qcow_open_node(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::Qcow(o) = opts else { unreachable!("qcow driver with other options") };
    let o = *o;
    let file = args.open_child(*o.file, "file", BDRV_CHILD_IMAGE | BDRV_CHILD_PRIMARY)?;
    args.set_backing_option(o.backing.map(|b| *b));
    let (encryptfmt, key_secret) = match o.encrypt {
        Some(e) => {
            let fmt = e.u.tag().as_str();
            let BlockdevQcowEncryptionU::Aes(q) = e.u;
            (Some(fmt), q.key_secret)
        }
        None => (None, None),
    };

    let mut header = [0u8; HEADER_SIZE];
    file.pread(0, &mut header).map_err(|e| open_err(&file, e))?;
    let magic = be32(&header, hdr::MAGIC);
    let version = be32(&header, hdr::VERSION);
    let backing_file_offset = be64(&header, hdr::BACKING_FILE_OFFSET);
    let backing_file_size = be32(&header, hdr::BACKING_FILE_SIZE);
    let size = be64(&header, hdr::SIZE);
    let cluster_bits = u32::from(header[hdr::CLUSTER_BITS]);
    let l2_bits = u32::from(header[hdr::L2_BITS]);
    let crypt_method = be32(&header, hdr::CRYPT_METHOD);
    let l1_table_offset = be64(&header, hdr::L1_TABLE_OFFSET);

    if magic != QCOW_MAGIC {
        return Err(Error::generic("Image not in qcow format"));
    }
    if version != QCOW_VERSION {
        let e = Error::generic(format!(
            "qcow (v{QCOW_VERSION}) does not support qcow version {version}"
        ));
        return Err(if version == 2 || version == 3 {
            e.hint("Try the 'qcow2' driver instead.\n")
        } else {
            e
        });
    }
    if size <= 1 {
        return Err(Error::generic("Image size is too small (must be at least 2 bytes)"));
    }
    if !(9..=16).contains(&cluster_bits) {
        return Err(Error::generic("Cluster size must be between 512 and 64k"));
    }
    // l2_bits is a number of entries of 8 bytes each.
    if !(9 - 3..=16 - 3).contains(&l2_bits) {
        return Err(Error::generic("L2 table size must be between 512 and 64k"));
    }

    let mut crypto = None;
    if crypt_method != QCOW_CRYPT_NONE {
        if uses_whitelist() && crypt_method == QCOW_CRYPT_AES {
            return Err(Error::generic(
                "Use of AES-CBC encrypted qcow images is no longer supported in system emulators",
            )
            .hint(
                "You can use 'qemu-img convert' to convert your image to an alternative \
                 supported format, such as unencrypted qcow, or raw with the LUKS format \
                 instead.\n",
            ));
        }
        if crypt_method != QCOW_CRYPT_AES {
            return Err(Error::generic("invalid encryption method in qcow header"));
        }
        if let Some(f) = encryptfmt {
            if f != "aes" {
                return Err(Error::generic(format!(
                    "Header reported 'aes' encryption format but options specify '{f}'"
                )));
            }
        }
        let opts = QCryptoBlockOpenOptions {
            u: QCryptoBlockOpenOptionsU::Qcow(QCryptoBlockOptionsQCow { key_secret }),
        };
        let cflags = if args.flags.no_io { QCRYPTO_BLOCK_OPEN_NO_IO } else { 0 };
        crypto = Some(QCryptoBlock::open(&opts, Some("encrypt."), &mut NoHeaderIo, cflags)?);
        args.meta.encrypted = true;
    } else if let Some(f) = encryptfmt {
        return Err(Error::generic(format!(
            "No encryption in image header, but options specified format '{f}'"
        )));
    }

    let cluster_size = 1u64 << cluster_bits;
    let l2_size = 1usize << l2_bits;
    let shift = cluster_bits + l2_bits;
    if size > u64::MAX - (1u64 << shift) {
        return Err(Error::generic("Image too large"));
    }
    let l1_size = (size + (1u64 << shift) - 1) >> shift;
    if l1_size > (i32::MAX as u64) / 8 {
        return Err(Error::generic("Image too large"));
    }
    let l1_size = l1_size as usize;

    let mut raw = vec![0u8; l1_size * 8];
    file.pread(l1_table_offset, &mut raw).map_err(|e| open_err(&file, e))?;
    let l1_table: Vec<u64> = raw.chunks_exact(8).map(|c| be64(c, 0)).collect();

    if backing_file_offset != 0 {
        let len = backing_file_size as usize;
        if len > 1023 {
            return Err(Error::generic("Backing file name too long"));
        }
        let mut name = vec![0u8; len];
        file.pread(backing_file_offset, &mut name).map_err(|e| open_err(&file, e))?;
        // A C string: it ends at the first NUL.
        if let Some(nul) = name.iter().position(|&b| b == 0) {
            name.truncate(nul);
        }
        args.set_backing_file(&String::from_utf8_lossy(&name), None);
    }

    Ok(Box::new(QcowDriver {
        cluster_bits,
        cluster_size,
        l2_bits,
        l2_size,
        l1_size,
        cluster_offset_mask: (1u64 << (63 - cluster_bits)) - 1,
        l1_table_offset,
        total_sectors: size / BDRV_SECTOR_SIZE,
        crypto,
        lock: Mutex::new(State {
            l1_table,
            l2_cache: vec![0; l2_size * L2_CACHE_SIZE],
            l2_cache_offsets: [0; L2_CACHE_SIZE],
            l2_cache_counts: [0; L2_CACHE_SIZE],
            cluster_cache: vec![0; cluster_size as usize],
            cluster_data: vec![0; cluster_size as usize],
            cluster_cache_offset: u64::MAX,
        }),
    }))
}

/// `decompress_buffer()`: `out` must come out full. The input may have padding after the end
/// of the stream.
fn decompress_buffer(out: &mut [u8], buf: &[u8]) -> io::Result<()> {
    let mut d = Decompress::new_with_window_bits(false, WINDOW_BITS);
    match d.decompress(buf, out, FlushDecompress::Finish) {
        Ok(Status::StreamEnd | Status::BufError | Status::Ok)
            if d.total_out() as usize == out.len() =>
        {
            Ok(())
        }
        _ => Err(errno(libc::EIO)),
    }
}

/// The deflate part of `qcow_co_pwritev_compressed()`: the compressed length, or `None` when
/// the data does not get smaller than `out`.
fn compress_buffer(out: &mut [u8], buf: &[u8]) -> io::Result<Option<usize>> {
    let mut c = Compress::new_with_window_bits(Compression::default(), false, WINDOW_BITS);
    match c.compress(buf, out, FlushCompress::Finish) {
        Ok(Status::StreamEnd) if (c.total_out() as usize) < out.len() => {
            Ok(Some(c.total_out() as usize))
        }
        Ok(_) => Ok(None),
        Err(_) => Err(errno(libc::EINVAL)),
    }
}

impl QcowDriver {
    fn state(&self) -> MutexGuard<'_, State> {
        self.lock.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn align_up(&self, v: u64) -> u64 {
        v.div_ceil(self.cluster_size) * self.cluster_size
    }

    /// `get_cluster_offset()`: the L2 entry for the cluster of `offset`, 0 when it is not
    /// allocated and `allocate` is [`Allocate::No`], allocating it otherwise.
    fn get_cluster_offset(
        &self,
        s: &mut State,
        file: &Node,
        offset: u64,
        allocate: Allocate,
    ) -> io::Result<u64> {
        let l1_index = (offset >> (self.l2_bits + self.cluster_bits)) as usize;
        let mut l2_offset = s.l1_table[l1_index];
        let mut new_l2_table = false;
        if l2_offset == 0 {
            if allocate == Allocate::No {
                return Ok(0);
            }
            // Allocate a new L2 table at the end of the file.
            l2_offset = self.align_up(file.getlength()?);
            s.l1_table[l1_index] = l2_offset;
            pwrite_sync(
                file,
                self.l1_table_offset + l1_index as u64 * 8,
                &l2_offset.to_be_bytes(),
            )?;
            new_l2_table = true;
        }

        let l2_size = self.l2_size;
        let slot = match s.l2_cache_offsets.iter().position(|&o| o == l2_offset) {
            Some(i) => {
                s.l2_cache_counts[i] = s.l2_cache_counts[i].wrapping_add(1);
                if s.l2_cache_counts[i] == u32::MAX {
                    for c in &mut s.l2_cache_counts {
                        *c >>= 1;
                    }
                }
                i
            }
            None => {
                // Not cached: replace the least used table.
                let mut min_index = 0;
                let mut min_count = u32::MAX;
                for (i, &c) in s.l2_cache_counts.iter().enumerate() {
                    if c < min_count {
                        min_count = c;
                        min_index = i;
                    }
                }
                let mut raw = vec![0u8; l2_size * 8];
                if new_l2_table {
                    pwrite_sync(file, l2_offset, &raw)?;
                } else {
                    file.pread(l2_offset, &mut raw)?;
                }
                let table = &mut s.l2_cache[min_index * l2_size..(min_index + 1) * l2_size];
                for (e, c) in table.iter_mut().zip(raw.chunks_exact(8)) {
                    *e = be64(c, 0);
                }
                s.l2_cache_offsets[min_index] = l2_offset;
                s.l2_cache_counts[min_index] = 1;
                min_index
            }
        };

        let l2_index = ((offset >> self.cluster_bits) as usize) & (l2_size - 1);
        let mut cluster_offset = s.l2_cache[slot * l2_size + l2_index];
        let compressed = cluster_offset & QCOW_OFLAG_COMPRESSED != 0;
        let normal = matches!(allocate, Allocate::Normal { .. });
        if cluster_offset != 0 && !(compressed && normal) {
            return Ok(cluster_offset);
        }
        let (n_start, n_end) = match allocate {
            Allocate::No => return Ok(0),
            Allocate::Normal { n_start, n_end } => (n_start, n_end),
            Allocate::Compressed(_) => (0, 0),
        };
        debug_assert!((n_start | n_end) % BDRV_SECTOR_SIZE == 0);
        let cs = self.cluster_size;
        if compressed && n_end - n_start < cs {
            // The cluster is compressed and not overwritten completely: decompress it into
            // a new cluster first.
            self.decompress_cluster(s, file, cluster_offset).map_err(|_| errno(libc::EIO))?;
            cluster_offset = self.align_up(file.getlength()?);
            file.pwrite(cluster_offset, &s.cluster_cache)?;
        } else {
            cluster_offset = file.getlength()?;
            match allocate {
                Allocate::Normal { .. } => {
                    cluster_offset = self.align_up(cluster_offset);
                    if cluster_offset + cs > i64::MAX as u64 {
                        return Err(errno(libc::E2BIG));
                    }
                    file.truncate_full((cluster_offset + cs) as i64, false, PreallocMode::Off, 0)
                        .map_err(to_io)?;
                    // An encrypted cluster must have its unwritten part initialised.
                    if let Some(crypto) = &self.crypto {
                        if n_end - n_start < cs {
                            let start_offset = offset & !(cs - 1);
                            let mut i = 0;
                            while i < cs {
                                if i < n_start || i >= n_end {
                                    let mut sector = [0u8; BDRV_SECTOR_SIZE as usize];
                                    crypto
                                        .encrypt(start_offset + i, &mut sector)
                                        .map_err(|_| errno(libc::EIO))?;
                                    file.pwrite(cluster_offset + i, &sector)?;
                                }
                                i += BDRV_SECTOR_SIZE;
                            }
                        }
                    }
                }
                Allocate::Compressed(compressed_size) => {
                    cluster_offset |=
                        QCOW_OFLAG_COMPRESSED | compressed_size << (63 - self.cluster_bits);
                }
                Allocate::No => unreachable!("handled above"),
            }
        }
        s.l2_cache[slot * l2_size + l2_index] = cluster_offset;
        pwrite_sync(file, l2_offset + l2_index as u64 * 8, &cluster_offset.to_be_bytes())?;
        Ok(cluster_offset)
    }

    /// `decompress_cluster()`: the compressed cluster at `cluster_offset` into
    /// `cluster_cache`.
    fn decompress_cluster(
        &self,
        s: &mut State,
        file: &Node,
        cluster_offset: u64,
    ) -> io::Result<()> {
        let coffset = cluster_offset & self.cluster_offset_mask;
        if s.cluster_cache_offset != coffset {
            let csize =
                ((cluster_offset >> (63 - self.cluster_bits)) & (self.cluster_size - 1)) as usize;
            file.pread(coffset, &mut s.cluster_data[..csize])?;
            let State { cluster_cache, cluster_data, .. } = s;
            decompress_buffer(cluster_cache, &cluster_data[..csize])?;
            s.cluster_cache_offset = coffset;
        }
        Ok(())
    }

    /// `qcow_co_pwritev()` with the state locked.
    fn pwrite_locked(
        &self,
        s: &mut State,
        file: &Node,
        mut offset: u64,
        buf: &[u8],
    ) -> io::Result<()> {
        // Disable the compressed cache.
        s.cluster_cache_offset = u64::MAX;
        let cs = self.cluster_size;
        let mut done = 0usize;
        while done < buf.len() {
            let offset_in_cluster = offset & (cs - 1);
            let n = ((cs - offset_in_cluster) as usize).min(buf.len() - done);
            let alloc = Allocate::Normal {
                n_start: offset_in_cluster,
                n_end: offset_in_cluster + n as u64,
            };
            let cluster_offset = self.get_cluster_offset(s, file, offset, alloc)?;
            if cluster_offset == 0 || cluster_offset & 511 != 0 {
                return Err(errno(libc::EIO));
            }
            let chunk = &buf[done..done + n];
            match &self.crypto {
                Some(crypto) => {
                    // Encrypt a copy, the caller's buffer stays as it is.
                    let mut enc = chunk.to_vec();
                    crypto.encrypt(offset, &mut enc).map_err(|_| errno(libc::EIO))?;
                    file.pwrite(cluster_offset + offset_in_cluster, &enc)?;
                }
                None => file.pwrite(cluster_offset + offset_in_cluster, chunk)?,
            }
            done += n;
            offset += n as u64;
        }
        Ok(())
    }
}

impl Driver for QcowDriver {
    /// `qcow_co_preadv()`.
    fn pread(&self, bs: &Node, mut offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let file = bs.file();
        let cs = self.cluster_size;
        let mut done = 0usize;
        while done < buf.len() {
            let mut s = self.state();
            let cluster_offset = self.get_cluster_offset(&mut s, &file, offset, Allocate::No)?;
            let offset_in_cluster = offset & (cs - 1);
            let n = ((cs - offset_in_cluster) as usize).min(buf.len() - done);
            let chunk = &mut buf[done..done + n];
            if cluster_offset == 0 {
                drop(s);
                match bs.backing() {
                    // Read from the base image.
                    Some(b) => b.node.pread(offset, chunk)?,
                    None => chunk.fill(0),
                }
            } else if cluster_offset & QCOW_OFLAG_COMPRESSED != 0 {
                self.decompress_cluster(&mut s, &file, cluster_offset)
                    .map_err(|_| errno(libc::EIO))?;
                let start = offset_in_cluster as usize;
                chunk.copy_from_slice(&s.cluster_cache[start..start + n]);
            } else {
                if cluster_offset & 511 != 0 {
                    return Err(errno(libc::EIO));
                }
                drop(s);
                file.pread(cluster_offset + offset_in_cluster, chunk)?;
                if let Some(crypto) = &self.crypto {
                    crypto.decrypt(offset, chunk).map_err(|_| errno(libc::EIO))?;
                }
            }
            done += n;
            offset += n as u64;
        }
        Ok(())
    }

    /// `qcow_co_pwritev()`.
    fn pwrite(&self, bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        let file = bs.file();
        let mut s = self.state();
        self.pwrite_locked(&mut s, &file, offset, buf)
    }

    /// QEMU has no `.bdrv_co_pwrite_zeroes` for qcow: zeroes are written as data.
    fn has_pwrite_zeroes(&self) -> bool {
        false
    }

    fn getlength(&self, _bs: &Node) -> io::Result<u64> {
        Ok(self.total_sectors * BDRV_SECTOR_SIZE)
    }

    /// `qcow_co_pwritev_compressed()`.
    fn pwrite_compressed(&self, bs: &Node, offset: u64, buf: &[u8]) -> Option<io::Result<()>> {
        Some(self.write_compressed(bs, offset, buf))
    }

    fn can_compress(&self) -> bool {
        true
    }

    /// `qcow_co_block_status()`.
    fn block_status(
        &self,
        bs: &Node,
        _want: u32,
        offset: u64,
        bytes: u64,
    ) -> Option<io::Result<BlockStatus>> {
        let file = bs.file();
        let cluster_offset = {
            let mut s = self.state();
            match self.get_cluster_offset(&mut s, &file, offset, Allocate::No) {
                Ok(o) => o,
                Err(e) => return Some(Err(e)),
            }
        };
        let index_in_cluster = offset & (self.cluster_size - 1);
        let pnum = (self.cluster_size - index_in_cluster).min(bytes);
        let mut st = BlockStatus { ret: 0, pnum, map: 0, file: None };
        if cluster_offset == 0 {
            return Some(Ok(st));
        }
        if cluster_offset & QCOW_OFLAG_COMPRESSED != 0 {
            st.ret = BDRV_BLOCK_DATA | BDRV_BLOCK_COMPRESSED;
        } else if self.crypto.is_some() {
            st.ret = BDRV_BLOCK_DATA;
        } else {
            st.ret = BDRV_BLOCK_DATA | BDRV_BLOCK_OFFSET_VALID;
            st.map = cluster_offset | index_in_cluster;
            st.file = Some(file);
        }
        Some(Ok(st))
    }

    /// `qcow_refresh_limits()`: at least encrypted images need 512 byte requests, and QEMU
    /// asks for that on all of them.
    fn refresh_limits(&self, _bs: &Node, bl: &mut BlockLimits) -> Result<()> {
        bl.request_alignment = BDRV_SECTOR_SIZE as u32;
        Ok(())
    }

    /// `qcow_reopen_prepare()`: nothing to do.
    fn reopen_prepare(&self, _bs: &Node, _state: &mut ReopenState) -> Option<Result<()>> {
        Some(Ok(()))
    }

    /// `qcow_co_get_info()`.
    fn get_info(&self, _bs: &Node) -> Option<io::Result<BlockDriverInfo>> {
        Some(Ok(BlockDriverInfo { cluster_size: self.cluster_size, ..Default::default() }))
    }

    /// `bdrv_has_zero_init_1()`.
    fn has_zero_init(&self, _bs: &Node) -> Option<bool> {
        Some(true)
    }

    /// `qcow_make_empty()`.
    fn make_empty(&self, bs: &Node) -> Option<io::Result<()>> {
        let file = bs.file();
        let mut s = self.state();
        let l1_length = self.l1_size as u64 * 8;
        s.l1_table.fill(0);
        // QEMU returns -1, EPERM, when the L1 table cannot be written.
        if pwrite_sync(&file, self.l1_table_offset, &vec![0u8; l1_length as usize]).is_err() {
            return Some(Err(errno(libc::EPERM)));
        }
        if let Err(e) = file.truncate_full(
            (self.l1_table_offset + l1_length) as i64,
            false,
            PreallocMode::Off,
            0,
        ) {
            return Some(Err(to_io(e)));
        }
        s.l2_cache.fill(0);
        s.l2_cache_offsets = [0; L2_CACHE_SIZE];
        s.l2_cache_counts = [0; L2_CACHE_SIZE];
        Some(Ok(()))
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

impl QcowDriver {
    /// `qcow_co_pwritev_compressed()`: one whole cluster, or the tail of the image.
    fn write_compressed(&self, bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        let cs = self.cluster_size as usize;
        let bytes = buf.len();
        if bytes != cs
            && (bytes > cs || offset + bytes as u64 != self.total_sectors * BDRV_SECTOR_SIZE)
        {
            return Err(errno(libc::EINVAL));
        }
        // Zero pad the last write if the image size is not cluster aligned.
        let mut data = vec![0u8; cs];
        data[..bytes].copy_from_slice(buf);

        // Best compression, small window, no zlib header.
        let mut out = vec![0u8; cs];
        let file = bs.file();
        let mut s = self.state();
        let Some(out_len) = compress_buffer(&mut out, &data)? else {
            // Could not compress: write a normal cluster.
            return self.pwrite_locked(&mut s, &file, offset, buf);
        };
        let cluster_offset =
            self.get_cluster_offset(&mut s, &file, offset, Allocate::Compressed(out_len as u64))?;
        if cluster_offset == 0 {
            return Err(errno(libc::EIO));
        }
        let cluster_offset = cluster_offset & self.cluster_offset_mask;
        file.pwrite(cluster_offset, &out[..out_len])
    }
}

/// `qcow_co_create()`: `blockdev-create` with `driver: qcow`.
fn qcow_co_create(graph: &BlockGraph, options: BlockdevCreateOptionsU) -> Result<()> {
    let BlockdevCreateOptionsU::Qcow(o) = options else {
        unreachable!("qcow driver with other create options")
    };
    // A failed write has no message of its own; the job reports the errno.
    do_create(graph, o)?.map_err(|e| Error::with_cause(ruvm_base::error::strerror(&e), e))
}

/// The body of `qcow_co_create()`. The outer error is one QEMU sets in `errp`, the inner one
/// a failed write that only returns `-errno`.
fn do_create(
    graph: &BlockGraph,
    o: BlockdevCreateOptionsQcow,
) -> Result<std::result::Result<(), io::Error>> {
    let total_size = o.size;
    if total_size == 0 {
        return Err(Error::generic("Image size is too small, cannot be zero length"));
    }
    if o.encrypt.as_ref().is_some_and(|e| e.u.tag() != QCryptoBlockFormat::Qcow) {
        return Err(Error::generic("Unsupported encryption format"));
    }

    let blk = graph.open_create_blk(o.file)?;
    let node = blk.root().expect("a new backend has its node");

    let mut header = [0u8; HEADER_SIZE];
    put_be32(&mut header, hdr::MAGIC, QCOW_MAGIC);
    put_be32(&mut header, hdr::VERSION, QCOW_VERSION);
    put_be64(&mut header, hdr::SIZE, total_size);
    let mut header_size = HEADER_SIZE as u64;
    let mut backing_file = o.backing_file;
    if let Some(b) = &backing_file {
        if b != "fat:" {
            put_be64(&mut header, hdr::BACKING_FILE_OFFSET, header_size);
            put_be32(&mut header, hdr::BACKING_FILE_SIZE, b.len() as u32);
            header_size += b.len() as u64;
        } else {
            // The special backing file of vvfat.
            backing_file = None;
        }
        // 512 byte clusters to avoid copying unmodified sectors, 32 KiB L2 tables.
        header[hdr::CLUSTER_BITS] = 9;
        header[hdr::L2_BITS] = 12;
    } else {
        // 4 KiB clusters, 4 KiB L2 tables.
        header[hdr::CLUSTER_BITS] = 12;
        header[hdr::L2_BITS] = 9;
    }
    header_size = (header_size + 7) & !7;
    let shift = u32::from(header[hdr::CLUSTER_BITS]) + u32::from(header[hdr::L2_BITS]);
    let l1_size = ((u128::from(total_size) + (1u128 << shift) - 1) >> shift) as u64;
    put_be64(&mut header, hdr::L1_TABLE_OFFSET, header_size);

    if let Some(enc) = &o.encrypt {
        put_be32(&mut header, hdr::CRYPT_METHOD, QCOW_CRYPT_AES);
        // Only checks that the key can be had, the scheme has no header.
        QCryptoBlock::create(enc, Some("encrypt."), &mut NoHeaderIo, 0)?;
    } else {
        put_be32(&mut header, hdr::CRYPT_METHOD, QCOW_CRYPT_NONE);
    }

    let write = || -> io::Result<()> {
        node.pwrite(0, &header)?;
        if let Some(b) = &backing_file {
            node.pwrite(HEADER_SIZE as u64, b.as_bytes())?;
        }
        let zero = [0u8; BDRV_SECTOR_SIZE as usize];
        let sectors = (8 * l1_size).div_ceil(BDRV_SECTOR_SIZE);
        for i in 0..sectors {
            node.pwrite(header_size + BDRV_SECTOR_SIZE * i, &zero)?;
        }
        Ok(())
    };
    Ok(write())
}

/// The keys of `qcow_create_opts` that `qemu_opts_to_qdict_filtered()` takes, without
/// `backing_fmt`, which is taken before.
const CREATE_OPT_NAMES: &[&str] =
    &["size", "backing_file", "encryption", "encrypt.format", "encrypt.key-secret"];

/// `qdict_rename_keys()`.
fn rename_keys(qdict: &mut QDict, renames: &[(&str, &str)]) -> Result<()> {
    for (from, to) in renames {
        if let Some(v) = qdict.get(from).cloned() {
            if qdict.contains_key(to) {
                return Err(Error::generic(format!(
                    "'{to}' and its alias '{from}' can't be used at the same time"
                )));
            }
            qdict.put(*to, v);
            qdict.remove(from);
        }
    }
    Ok(())
}

/// The part of `qcow_co_create_opts()` before the file is created: the qcow options taken out
/// of `options`, the legacy syntax converted and the keys renamed.
fn create_opts_qdict(options: &mut QDict) -> Result<QDict> {
    // A backing format cannot be stored, but it must make sense.
    if let Some(fmt) = take_str(options, "backing_fmt") {
        if find_format(&fmt).is_none() {
            return Err(Error::generic(format!("unrecognized backing format '{fmt}'")));
        }
    }

    let mut qdict = QDict::new();
    for name in CREATE_OPT_NAMES {
        if let Some(v) = take_str(options, name) {
            qdict.put(*name, v);
        }
    }
    match qdict.get_str("encryption") {
        Some("on") => qdict.put("encryption", "qcow"),
        Some("off") => {
            qdict.remove("encryption");
        }
        _ => {}
    }
    if qdict.get_str("encrypt.format") == Some("aes") {
        qdict.put("encrypt.format", "qcow");
    }
    rename_keys(&mut qdict, &[("backing_file", "backing-file"), ("encryption", "encrypt.format")])?;
    Ok(qdict)
}

/// `qcow_co_create_opts()`: `qemu-img create -f qcow`.
fn qcow_co_create_opts(filename: &str, options: &mut QDict) -> Result<()> {
    let qdict = create_opts_qdict(options)?;

    // Create and open the file (protocol layer).
    let (graph, _blk, mut all) = create_opts_open(filename, options, "qcow", &[], &[])?;
    for (k, v) in qdict.iter() {
        all.put(k, v.clone());
    }
    let create_options = visit_create_options(all)?;
    let BlockdevCreateOptionsU::Qcow(mut o) = create_options.u else {
        unreachable!("driver is qcow")
    };
    // Silently round up the size.
    o.size = o.size.div_ceil(BDRV_SECTOR_SIZE) * BDRV_SECTOR_SIZE;
    // The file is in `graph` already, by node name.
    debug_assert!(matches!(o.file, BlockdevRef::Reference(_)));
    do_create(&graph, o)?.map_err(|e| Error::from_io("Could not create image", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(version: u32) -> Vec<u8> {
        let mut h = vec![0u8; HEADER_SIZE];
        put_be32(&mut h, hdr::MAGIC, QCOW_MAGIC);
        put_be32(&mut h, hdr::VERSION, version);
        h
    }

    #[test]
    fn probe() {
        assert_eq!(qcow_probe(&header(1), None), 100);
        assert_eq!(qcow_probe(&header(2), None), 0);
        assert_eq!(qcow_probe(&header(1)[..47], None), 0);
        assert_eq!(qcow_probe(&[0u8; 64], None), 0);
    }

    #[test]
    fn compression_round_trip() {
        let src: Vec<u8> = (0..4096u32).map(|i| (i / 64) as u8).collect();
        let mut out = vec![0u8; 4096];
        let n = compress_buffer(&mut out, &src).unwrap().unwrap();
        let mut back = vec![0u8; 4096];
        decompress_buffer(&mut back, &out[..n]).unwrap();
        assert_eq!(back, src);
        // Output that does not come out full is an error.
        let mut long = vec![0u8; 8192];
        assert!(decompress_buffer(&mut long, &out[..n]).is_err());
    }

    #[test]
    fn incompressible_is_none() {
        let mut x = 0x1234_5678u32;
        let src: Vec<u8> = (0..4096)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x as u8
            })
            .collect();
        let mut out = vec![0u8; 4096];
        assert_eq!(compress_buffer(&mut out, &src).unwrap(), None);
    }

    fn opts(pairs: &[(&str, &str)]) -> QDict {
        let mut d = QDict::new();
        for (k, v) in pairs {
            d.put(*k, *v);
        }
        d
    }

    #[test]
    fn create_opts_conversions() {
        let mut o = opts(&[("size", "1M"), ("backing_file", "fat:"), ("other", "x")]);
        let q = create_opts_qdict(&mut o).unwrap();
        assert_eq!(q.get_str("size"), Some("1M"));
        assert_eq!(q.get_str("backing-file"), Some("fat:"));
        assert!(!q.contains_key("backing_file"));
        // What qcow does not know stays for the protocol.
        assert_eq!(o.get_str("other"), Some("x"));
        assert!(!o.contains_key("size"));

        let mut o = opts(&[("encryption", "on"), ("encrypt.key-secret", "sec0")]);
        let q = create_opts_qdict(&mut o).unwrap();
        assert_eq!(q.get_str("encrypt.format"), Some("qcow"));
        assert!(!q.contains_key("encryption"));

        let mut o = opts(&[("encryption", "off")]);
        let q = create_opts_qdict(&mut o).unwrap();
        assert!(!q.contains_key("encrypt.format"));

        let mut o = opts(&[("encrypt.format", "aes")]);
        let q = create_opts_qdict(&mut o).unwrap();
        assert_eq!(q.get_str("encrypt.format"), Some("qcow"));

        let mut o = opts(&[("encryption", "on"), ("encrypt.format", "aes")]);
        let e = create_opts_qdict(&mut o).unwrap_err();
        assert_eq!(
            e.message(),
            "'encrypt.format' and its alias 'encryption' can't be used at the same time"
        );

        let mut o = opts(&[("backing_fmt", "nosuch")]);
        let e = create_opts_qdict(&mut o).unwrap_err();
        assert_eq!(e.message(), "unrecognized backing format 'nosuch'");
        let mut o = opts(&[("backing_fmt", "raw")]);
        assert!(create_opts_qdict(&mut o).unwrap().is_empty());
    }

    #[test]
    fn create_opts_list_order() {
        let names: Vec<&str> = QCOW_CREATE_OPTS.iter().map(|d| d.name).collect();
        assert_eq!(
            names,
            [
                "size",
                "backing_file",
                "backing_fmt",
                "encryption",
                "encrypt.format",
                "encrypt.key-secret"
            ]
        );
    }

    #[cfg(unix)]
    #[test]
    fn create_opts_for_vvfat() {
        let dir = std::env::temp_dir().join(format!("ruvm-qcow-unit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("vvfat.qcow");
        let path = path.to_str().unwrap();
        let mut o = QDict::new();
        o.put("size", 1u64 << 20);
        o.put("backing_file", "fat:");
        (QCOW.create_opts.unwrap())(path, &mut o).unwrap();
        assert!(o.is_empty());
        let data = std::fs::read(path).unwrap();
        assert_eq!(qcow_probe(&data, None), 100);
        assert_eq!(be64(&data, hdr::BACKING_FILE_OFFSET), 0);
        assert_eq!(data[hdr::CLUSTER_BITS], 9);
        assert_eq!(data[hdr::L2_BITS], 12);
        assert_eq!(be64(&data, hdr::SIZE), 1 << 20);
        assert_eq!(be64(&data, hdr::L1_TABLE_OFFSET), 48);
        // One L1 entry, padded to a sector.
        assert_eq!(data.len(), 48 + 512);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
