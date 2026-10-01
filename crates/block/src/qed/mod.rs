// SPDX-License-Identifier: GPL-2.0-or-later

//! The `qed` format driver, the QEMU Enhanced Disk format from block/qed.c and block/qed.h.
//!
//! A QED image starts with a header cluster holding the geometry, the feature bits and the
//! backing file name, then a two level table maps guest clusters to data clusters in the image
//! file: the L1 table at `l1_table_offset` points to L2 tables, and L2 entries point to data
//! clusters. An entry of 0 is unallocated (the guest reads the backing file, or zeroes without
//! one), and an L2 entry of 1 is a zero cluster. New clusters and tables are always appended
//! to the end of the file, and the end of the file is where the next allocation goes.
//!
//! An allocating write sets `QED_F_NEED_CHECK` in the header first (unless the image has a
//! backing file, where a flush before the L2 update keeps the image consistent), and an image
//! opened read-write with the flag set is checked and repaired before use.
//!
//! Differences from QEMU:
//!
//! - QEMU clears `QED_F_NEED_CHECK` from a timer five seconds after the last allocating write.
//!   There is no timer here: the flag is cleared when the node is flushed (after flushing the
//!   file, as the timer does) and when it is closed.
//!   QEMU also fires the timer when the node is drained, which on a read-only node clears the
//!   flag in memory only (the header write fails), so `qemu-img info` reports a read-only
//!   dirty image as clean. Here the flag stays set on a read-only node and `info` reports
//!   it dirty, which is what the header says.
//! - There is no `BDRV_O_CHECK` open flag, so an image with the flag set that is opened
//!   read-write for `qemu-img check -r` is repaired while it is opened, and the check that
//!   follows sees the repaired image. Read-only opens are not repaired, as in QEMU.
//! - Requests run one at a time under the driver lock, which is QEMU's `table_lock` held for
//!   the whole request including the data transfers. The queue of allocating writes and the
//!   `-EAGAIN` restart that goes with it are not needed.
//! - The L2 cache holds tables without reference counts; see the `table` module.
//! - Where QEMU flushes the QED node itself (after writing a new L2 table, and when a check
//!   marks the image clean) the file child is flushed, because the driver lock is held.
//! - The backing file name buffer is 4096 bytes, `PATH_MAX` on Linux, on every host.

mod check;
mod table;

use std::io;
use std::sync::{Mutex, MutexGuard};

use ruvm_base::error::strerror;
use ruvm_base::{Error, Result};
use ruvm_qapi::QDict;
use ruvm_qapi::opts::QemuOptDesc;
use ruvm_qapi::types::{
    BlkdebugEvent, BlockdevCreateOptionsQed, BlockdevCreateOptionsU, BlockdevOptionsU, PreallocMode,
};

use crate::drivers::{DriverDef, OpenArgs};
use crate::graph::BlockGraph;
use crate::imgopts::{create_opts_open, opt_desc, visit_create_options};
use crate::node::{
    BDRV_BLOCK_DATA, BDRV_BLOCK_OFFSET_VALID, BDRV_BLOCK_ZERO, BDRV_CHILD_IMAGE,
    BDRV_CHILD_PRIMARY, BDRV_SECTOR_SIZE, BlockDriverInfo, BlockLimits, BlockStatus, CheckResult,
    Driver, Node, ReopenState, errno,
};

use self::table::{CachedL2Table, Cluster, L2Cache, write_table};

/// `bdrv_qed`.
pub(crate) static QED: DriverDef = DriverDef::format("qed", qed_open)
    .with_probe(qed_probe)
    .with_create_opts(qed_create_opts)
    .with_create_opts_list(&QED_CREATE_OPTS)
    .with_create(qed_create)
    .with_backing();

/// `qed_create_opts`, in QEMU's order.
static QED_CREATE_OPTS: [QemuOptDesc; 5] = [
    opt_desc!("size", Size, "Virtual disk size"),
    opt_desc!("backing_file", String, "File name of a base image"),
    opt_desc!("backing_fmt", String, "Image format of the base image"),
    opt_desc!("cluster_size", Size, "Cluster size (in bytes)", "65536"),
    opt_desc!("table_size", Size, "L1/L2 table size (in clusters)"),
];

/// `QED_MAGIC`: "QED\0" read as a little-endian number.
const QED_MAGIC: u32 = u32::from_le_bytes(*b"QED\0");

/// `QED_F_BACKING_FILE`: the image has a backing file.
const F_BACKING_FILE: u64 = 0x01;
/// `QED_F_NEED_CHECK`: the image needs a consistency check before use.
const F_NEED_CHECK: u64 = 0x02;
/// `QED_F_BACKING_FORMAT_NO_PROBE`: the backing file is raw and is not probed.
const F_BACKING_FORMAT_NO_PROBE: u64 = 0x04;
/// `QED_FEATURE_MASK`.
const FEATURE_MASK: u64 = F_BACKING_FILE | F_NEED_CHECK | F_BACKING_FORMAT_NO_PROBE;
/// `QED_COMPAT_FEATURE_MASK`.
const COMPAT_FEATURE_MASK: u64 = 0;
/// `QED_AUTOCLEAR_FEATURE_MASK`.
const AUTOCLEAR_FEATURE_MASK: u64 = 0;

/// `QED_MIN_CLUSTER_SIZE`.
const MIN_CLUSTER_SIZE: u32 = 4 * 1024;
/// `QED_MAX_CLUSTER_SIZE`.
const MAX_CLUSTER_SIZE: u32 = 64 * 1024 * 1024;
/// `QED_DEFAULT_CLUSTER_SIZE`.
const DEFAULT_CLUSTER_SIZE: u32 = 64 * 1024;
/// `QED_MIN_TABLE_SIZE`, in clusters.
const MIN_TABLE_SIZE: u32 = 1;
/// `QED_MAX_TABLE_SIZE`, in clusters.
const MAX_TABLE_SIZE: u32 = 16;
/// `QED_DEFAULT_TABLE_SIZE`, in clusters.
const DEFAULT_TABLE_SIZE: u32 = 4;

/// `sizeof(QEDHeader)`.
const HEADER_SIZE: usize = 64;

/// `sizeof(bs->backing_file)`.
const BACKING_FILE_BUF: usize = 4096;

/// `QEDHeader`, in host byte order. On disk every field is little-endian.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Header {
    magic: u32,
    /// In bytes.
    cluster_size: u32,
    /// For L1 and L2 tables, in clusters.
    table_size: u32,
    /// In clusters.
    header_size: u32,
    /// Format feature flags.
    features: u64,
    /// Compatible feature flags.
    compat_features: u64,
    /// Self-resetting feature flags.
    autoclear_features: u64,
    /// In bytes.
    l1_table_offset: u64,
    /// Total logical image size, in bytes.
    image_size: u64,
    /// In bytes, from the start of the file.
    backing_filename_offset: u32,
    /// In bytes.
    backing_filename_size: u32,
}

impl Header {
    /// `qed_header_le_to_cpu()`.
    fn from_le(b: &[u8; HEADER_SIZE]) -> Header {
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().expect("4 bytes"));
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().expect("8 bytes"));
        Header {
            magic: u32_at(0),
            cluster_size: u32_at(4),
            table_size: u32_at(8),
            header_size: u32_at(12),
            features: u64_at(16),
            compat_features: u64_at(24),
            autoclear_features: u64_at(32),
            l1_table_offset: u64_at(40),
            image_size: u64_at(48),
            backing_filename_offset: u32_at(56),
            backing_filename_size: u32_at(60),
        }
    }

    /// `qed_header_cpu_to_le()`.
    fn to_le(self) -> [u8; HEADER_SIZE] {
        let mut b = [0u8; HEADER_SIZE];
        b[0..4].copy_from_slice(&self.magic.to_le_bytes());
        b[4..8].copy_from_slice(&self.cluster_size.to_le_bytes());
        b[8..12].copy_from_slice(&self.table_size.to_le_bytes());
        b[12..16].copy_from_slice(&self.header_size.to_le_bytes());
        b[16..24].copy_from_slice(&self.features.to_le_bytes());
        b[24..32].copy_from_slice(&self.compat_features.to_le_bytes());
        b[32..40].copy_from_slice(&self.autoclear_features.to_le_bytes());
        b[40..48].copy_from_slice(&self.l1_table_offset.to_le_bytes());
        b[48..56].copy_from_slice(&self.image_size.to_le_bytes());
        b[56..60].copy_from_slice(&self.backing_filename_offset.to_le_bytes());
        b[60..64].copy_from_slice(&self.backing_filename_size.to_le_bytes());
        b
    }
}

/// `qed_max_image_size()`.
fn max_image_size(cluster_size: u32, table_size: u32) -> u64 {
    // The product is computed in 32 bits in QEMU too.
    let table_entries = u64::from(table_size.wrapping_mul(cluster_size)) / 8;
    let l2_size = table_entries.wrapping_mul(u64::from(cluster_size));
    l2_size.wrapping_mul(table_entries)
}

/// `qed_is_cluster_size_valid()`.
fn is_cluster_size_valid(cluster_size: u32) -> bool {
    (MIN_CLUSTER_SIZE..=MAX_CLUSTER_SIZE).contains(&cluster_size) && cluster_size.is_power_of_two()
}

/// `qed_is_table_size_valid()`.
fn is_table_size_valid(table_size: u32) -> bool {
    (MIN_TABLE_SIZE..=MAX_TABLE_SIZE).contains(&table_size) && table_size.is_power_of_two()
}

/// `qed_is_image_size_valid()`.
fn is_image_size_valid(image_size: u64, cluster_size: u32, table_size: u32) -> bool {
    image_size % BDRV_SECTOR_SIZE == 0 && image_size <= max_image_size(cluster_size, table_size)
}

/// `qed_check_cluster_offset()` on its own: whether `offset` is a cluster in the image file
/// past the header.
fn check_cluster_offset(h: &Header, file_size: u64, offset: u64) -> bool {
    let header_size = u64::from(h.header_size) * u64::from(h.cluster_size);
    if offset & (u64::from(h.cluster_size) - 1) != 0 {
        return false;
    }
    offset >= header_size && offset < file_size
}

/// `qed_fmt_is_raw()`.
fn fmt_is_raw(fmt: Option<&str>) -> bool {
    fmt == Some("raw")
}

/// `BDRVQEDState`, what the driver lock (`table_lock`) protects.
struct State {
    header: Header,
    l1_table: Vec<u64>,
    l2_cache: L2Cache,
    /// Entries per table.
    table_nelems: u32,
    l1_shift: u32,
    l2_shift: u32,
    l2_mask: u32,
    /// The end of the image file rounded down to a cluster, where the next cluster goes.
    file_size: u64,
}

impl State {
    /// `qed_start_of_cluster()`.
    fn start_of_cluster(&self, offset: u64) -> u64 {
        offset & !(u64::from(self.header.cluster_size) - 1)
    }

    /// `qed_offset_into_cluster()`.
    fn offset_into_cluster(&self, offset: u64) -> u64 {
        offset & (u64::from(self.header.cluster_size) - 1)
    }

    /// `qed_bytes_to_clusters()`. Like QEMU this divides by one less than the cluster size,
    /// which counts one cluster too many for every `cluster_size - 1` clusters.
    fn bytes_to_clusters(&self, bytes: u64) -> u64 {
        let c = u64::from(self.header.cluster_size);
        self.start_of_cluster(bytes.wrapping_add(c - 1)) / (c - 1)
    }

    /// `qed_l1_index()`.
    fn l1_index(&self, pos: u64) -> usize {
        (pos >> self.l1_shift) as usize
    }

    /// `qed_l2_index()`.
    fn l2_index(&self, pos: u64) -> usize {
        ((pos >> self.l2_shift) as u32 & self.l2_mask) as usize
    }

    /// `qed_check_cluster_offset()`.
    fn check_cluster_offset(&self, offset: u64) -> bool {
        check_cluster_offset(&self.header, self.file_size, offset)
    }

    /// `qed_check_table_offset()`: whether a whole table fits at `offset`.
    fn check_table_offset(&self, offset: u64) -> bool {
        let end_offset = offset.wrapping_add(
            u64::from(self.header.table_size - 1) * u64::from(self.header.cluster_size),
        );
        // Overflow check
        if end_offset <= offset {
            return false;
        }
        self.check_cluster_offset(offset) && self.check_cluster_offset(end_offset)
    }

    /// `qed_alloc_clusters()`: where `n` new clusters go. Only the state changes.
    fn alloc_clusters(&mut self, n: u64) -> u64 {
        let offset = self.file_size;
        self.file_size += n * u64::from(self.header.cluster_size);
        offset
    }

    /// `qed_write_header_sync()`: writes the header fields.
    fn write_header_sync(&self, file: &Node) -> io::Result<()> {
        file.pwrite(0, &self.header.to_le())
    }

    /// `qed_write_header()`: updates the header in place, rewriting the whole sector it is in
    /// as read, so that unknown data after the header survives.
    fn write_header(&self, file: &Node) -> io::Result<()> {
        let len = HEADER_SIZE.div_ceil(BDRV_SECTOR_SIZE as usize) * BDRV_SECTOR_SIZE as usize;
        let mut buf = vec![0u8; len];
        file.pread(0, &mut buf)?;
        buf[..HEADER_SIZE].copy_from_slice(&self.header.to_le());
        file.pwrite(0, &buf)
    }

    /// `qed_update_l2_table()`: points `n` entries from `index` at `cluster` and the clusters
    /// after it, or all at the zero or unallocated marker.
    fn update_l2_table(&self, table: &mut [u64], index: usize, n: usize, mut cluster: u64) {
        for e in &mut table[index..index + n] {
            *e = cluster;
            if !table::is_unalloc_cluster(cluster) && !table::is_zero_cluster(cluster) {
                cluster += u64::from(self.header.cluster_size);
            }
        }
    }

    /// `qed_aio_write_l2_update()`: records `n` clusters at `offset` for the guest range at
    /// `pos`, in a new L2 table (and the L1 table) if `need_alloc`.
    fn write_l2_update(
        &mut self,
        file: &Node,
        need_alloc: bool,
        pos: u64,
        n: usize,
        offset: u64,
    ) -> io::Result<()> {
        let index = self.l2_index(pos);
        if need_alloc {
            // qed_new_l2_table()
            let l2_offset = self.alloc_clusters(u64::from(self.header.table_size));
            let mut table = vec![0u64; self.table_nelems as usize];
            self.update_l2_table(&mut table, index, n, offset);
            // Write out the whole new L2 table
            file.debug_event(BlkdebugEvent::L2Update);
            write_table(file, l2_offset, &table, 0, table.len(), true)?;

            // qed_aio_write_l1_update()
            let l1_index = self.l1_index(pos);
            self.l1_table[l1_index] = l2_offset;
            let r = self.write_l1_table(file, l1_index, 1);
            // Commit the current L2 table to the cache
            self.l2_cache.commit(CachedL2Table { offset: l2_offset, table });
            r
        } else {
            let l2_offset = self.l1_table[self.l1_index(pos)];
            let mut table = std::mem::take(self.read_l2_table(file, l2_offset)?);
            self.update_l2_table(&mut table, index, n, offset);
            // Write out only the updated part of the L2 table
            file.debug_event(BlkdebugEvent::L2Update);
            let r = write_table(file, l2_offset, &table, index, n, false);
            if let Some(t) = self.l2_cache.find(l2_offset) {
                *t = table;
            }
            r
        }
    }
}

/// What a request carries, `QEMUIOVector` and the `QED_AIOCB_*` flags.
enum Buf<'a> {
    Read(&'a mut [u8]),
    Write(&'a [u8]),
    /// `QED_AIOCB_WRITE | QED_AIOCB_ZERO`.
    Zero,
}

/// The QED driver state.
pub(crate) struct QedDriver {
    s: Mutex<State>,
}

/// `bdrv_qed_probe()`.
fn qed_probe(buf: &[u8], _filename: Option<&str>) -> i32 {
    if buf.len() < HEADER_SIZE {
        return 0;
    }
    if u32::from_le_bytes(buf[0..4].try_into().expect("4 bytes")) != QED_MAGIC {
        return 0;
    }
    100
}

/// `qed_read_string()`: the string of `n` bytes at `offset`, up to its first NUL.
fn read_string(file: &Node, offset: u64, n: usize, buflen: usize) -> io::Result<String> {
    if n >= buflen {
        return Err(errno(libc::EINVAL));
    }
    let mut buf = vec![0u8; n];
    file.pread(offset, &mut buf)?;
    let end = buf.iter().position(|&c| c == 0).unwrap_or(n);
    Ok(String::from_utf8_lossy(&buf[..end]).into_owned())
}

/// The backing file an image header names: the file name and whether it is raw.
type BackingInfo = Option<(String, bool)>;

/// `bdrv_qed_do_open()`: reads and checks the header and the L1 table of the image in `file`.
/// `inactive` is `BDRV_O_INACTIVE`.
fn do_open(file: &Node, inactive: bool) -> Result<(State, BackingInfo)> {
    let mut le_header = [0u8; HEADER_SIZE];
    if file.pread(0, &mut le_header).is_err() {
        return Err(Error::generic("Failed to read QED header"));
    }
    let header = Header::from_le(&le_header);

    if header.magic != QED_MAGIC {
        return Err(Error::generic("Image not in QED format"));
    }
    if header.features & !FEATURE_MASK != 0 {
        // image uses unsupported feature bits
        return Err(Error::generic(format!(
            "Unsupported QED features: {:x}",
            header.features & !FEATURE_MASK
        )));
    }
    if !is_cluster_size_valid(header.cluster_size) {
        return Err(Error::generic("QED cluster size is invalid"));
    }

    // Round down file size to the last cluster
    let Ok(file_len) = file.getlength() else {
        return Err(Error::generic("Failed to get file length"));
    };
    let file_size = file_len & !(u64::from(header.cluster_size) - 1);

    if !is_table_size_valid(header.table_size) {
        return Err(Error::generic("QED table size is invalid"));
    }
    if !is_image_size_valid(header.image_size, header.cluster_size, header.table_size) {
        return Err(Error::generic("QED image size is invalid"));
    }

    let table_nelems = header.cluster_size * header.table_size / 8;
    let l2_shift = header.cluster_size.trailing_zeros();
    let mut s = State {
        header,
        l1_table: Vec::new(),
        l2_cache: L2Cache::default(),
        table_nelems,
        l1_shift: l2_shift + table_nelems.trailing_zeros(),
        l2_shift,
        l2_mask: table_nelems - 1,
        file_size,
    };
    if !s.check_table_offset(header.l1_table_offset) {
        return Err(Error::generic("QED table offset is invalid"));
    }

    // Header size calculation must not overflow uint32_t
    if header.header_size > u32::MAX / header.cluster_size {
        return Err(Error::generic("QED header size is too large"));
    }

    let mut backing = None;
    if header.features & F_BACKING_FILE != 0 {
        if u64::from(header.backing_filename_offset) + u64::from(header.backing_filename_size)
            > u64::from(header.cluster_size * header.header_size)
        {
            return Err(Error::generic("QED backing filename offset is invalid"));
        }
        let name = read_string(
            file,
            u64::from(header.backing_filename_offset),
            header.backing_filename_size as usize,
            BACKING_FILE_BUF,
        )
        .map_err(|_| Error::generic("Failed to read backing filename"))?;
        backing = Some((name, header.features & F_BACKING_FORMAT_NO_PROBE != 0));
    }

    // Reset unknown autoclear feature bits. This is a backwards compatibility mechanism
    // that allows images to be opened by older programs, which "knock out" unknown feature
    // bits. When an image is opened by a newer program again it can detect that the
    // autoclear feature is no longer valid.
    if header.autoclear_features & !AUTOCLEAR_FEATURE_MASK != 0 && !file.read_only() && !inactive {
        s.header.autoclear_features &= AUTOCLEAR_FEATURE_MASK;
        if s.write_header_sync(file).is_err() {
            return Err(Error::generic("Failed to update header"));
        }
        // From here on only known autoclear feature bits are valid
        let _ = file.flush();
    }

    s.l1_table = vec![0u64; table_nelems as usize];
    if s.read_l1_table(file).is_err() {
        return Err(Error::generic("Failed to read L1 table"));
    }

    // If image was not closed cleanly, check consistency
    if s.header.features & F_NEED_CHECK != 0 {
        // Read-only images cannot be fixed. There is no risk of corruption since write
        // operations are not possible. Therefore, allow potentially inconsistent images to
        // be opened read-only. This can aid data recovery from an otherwise inconsistent
        // image.
        if !file.read_only() && !inactive {
            let (_, r) = check::qed_check(&mut s, file, true);
            if r.is_err() {
                return Err(Error::generic("Image corrupted"));
            }
        }
    }
    Ok((s, backing))
}

/// `bdrv_qed_open()`.
fn qed_open(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::Qed(o) = opts else { unreachable!("qed driver with other options") };
    let file = args.open_child(*o.file, "file", BDRV_CHILD_IMAGE | BDRV_CHILD_PRIMARY)?;
    args.set_backing_option(o.backing.map(|b| *b));
    let (s, backing) = do_open(&file, args.flags.inactive)?;
    if let Some((name, raw)) = backing {
        args.set_backing_file(&name, raw.then_some("raw"));
    }
    Ok(Box::new(QedDriver { s: Mutex::new(s) }))
}

impl QedDriver {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.s.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `qed_read_backing_file()`: the backing file's data at `pos`, or zeroes without one.
    fn read_backing_file(bs: &Node, pos: u64, buf: &mut [u8]) -> io::Result<()> {
        match bs.backing() {
            Some(b) => {
                bs.file().debug_event(BlkdebugEvent::ReadBackingAio);
                b.node.pread(pos, buf)
            }
            None => {
                buf.fill(0);
                Ok(())
            }
        }
    }

    /// `qed_copy_from_backing_file()`: copies `len` bytes of the backing file at `pos` (or
    /// zeroes) to `offset` in the image file.
    fn copy_from_backing_file(
        bs: &Node,
        file: &Node,
        pos: u64,
        len: u64,
        offset: u64,
    ) -> io::Result<()> {
        // Skip copy entirely if there is no work to do
        if len == 0 {
            return Ok(());
        }
        let mut buf = vec![0u8; len as usize];
        Self::read_backing_file(bs, pos, &mut buf)?;
        file.debug_event(BlkdebugEvent::CowWrite);
        file.pwrite(offset, &buf)
    }

    /// `qed_aio_write_alloc()`: a write to clusters the image does not have yet, or a zero
    /// write to clusters that are not zero clusters already. `data` is `None` for zeroes.
    fn write_alloc(
        s: &mut State,
        bs: &Node,
        file: &Node,
        found: Cluster,
        pos: u64,
        len: u64,
        data: Option<&[u8]>,
    ) -> io::Result<()> {
        let nclusters = s.bytes_to_clusters(s.offset_into_cluster(pos) + len);
        let cluster = match data {
            None => {
                // Skip ahead if the clusters are already zero
                if found == Cluster::Zero {
                    return Ok(());
                }
                1
            }
            Some(_) => s.alloc_clusters(nclusters),
        };

        // qed_should_set_need_check(): the flush before the L2 update keeps an image with a
        // backing file consistent.
        if bs.backing().is_none() && s.header.features & F_NEED_CHECK == 0 {
            s.header.features |= F_NEED_CHECK;
            s.write_header(file)?;
        }

        if let Some(data) = data {
            // qed_aio_write_cow(): populate front untouched region of new data cluster
            let start = s.start_of_cluster(pos);
            let head = s.offset_into_cluster(pos);
            Self::copy_from_backing_file(bs, file, start, head, cluster)?;

            // Populate back untouched region of new data cluster
            let start = pos + len;
            let tail = s.start_of_cluster(start + u64::from(s.header.cluster_size) - 1) - start;
            let offset = cluster + head + len;
            Self::copy_from_backing_file(bs, file, start, tail, offset)?;

            // qed_aio_write_main()
            file.debug_event(BlkdebugEvent::WriteAio);
            file.pwrite(cluster + head, data)?;

            if bs.backing().is_some() {
                // Flush new data clusters before updating the L2 table. A crash during an
                // allocating write could otherwise leave empty clusters in the image, and
                // the backing file data of the untouched regions would be lost.
                file.flush()?;
            }
        }

        s.write_l2_update(file, found == Cluster::L1, pos, nclusters as usize, cluster)
    }

    /// `qed_co_request()` and `qed_aio_next_io()`: the request `buf` at `pos`, cluster run
    /// by cluster run.
    fn request(&self, bs: &Node, pos: u64, bytes: u64, mut buf: Buf<'_>) -> io::Result<()> {
        let file = bs.file();
        let mut s = self.lock();
        let mut done = 0u64;
        while done < bytes {
            let cur_pos = pos + done;
            let (found, offset, len) = s.find_cluster(&file, cur_pos, bytes - done)?;
            let (d, l) = (done as usize, len as usize);
            // Adjust offset into cluster
            let offset = offset + s.offset_into_cluster(cur_pos);
            match &mut buf {
                // qed_aio_read_data(): zero clusters and backing file reads, otherwise the
                // data cluster.
                Buf::Read(out) => {
                    let chunk = &mut out[d..d + l];
                    match found {
                        Cluster::Zero => chunk.fill(0),
                        Cluster::Found => {
                            file.debug_event(BlkdebugEvent::ReadAio);
                            file.pread(offset, chunk)?
                        }
                        Cluster::L2 | Cluster::L1 => Self::read_backing_file(bs, cur_pos, chunk)?,
                    }
                }
                // qed_aio_write_data()
                Buf::Write(data) => {
                    let chunk = &data[d..d + l];
                    match found {
                        Cluster::Found => {
                            file.debug_event(BlkdebugEvent::WriteAio);
                            file.pwrite(offset, chunk)?
                        }
                        _ => {
                            Self::write_alloc(&mut s, bs, &file, found, cur_pos, len, Some(chunk))?
                        }
                    }
                }
                Buf::Zero => match found {
                    Cluster::Found => {
                        file.debug_event(BlkdebugEvent::WriteAio);
                        file.pwrite(offset, &vec![0u8; l])?
                    }
                    _ => Self::write_alloc(&mut s, bs, &file, found, cur_pos, len, None)?,
                },
            }
            done += len;
        }
        Ok(())
    }

    /// `bdrv_qed_do_close()` without freeing anything: flushes, and clears the need-check
    /// flag since the image was closed cleanly.
    fn do_close(&self, bs: &Node) {
        let file = bs.file();
        let mut s = self.lock();
        // Ensure writes reach stable storage
        let _ = file.flush();
        // Clean shutdown, no check required on next open
        if s.header.features & F_NEED_CHECK != 0 && !file.read_only() {
            s.header.features &= !F_NEED_CHECK;
            let _ = s.write_header_sync(&file);
        }
    }
}

impl Driver for QedDriver {
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let bytes = buf.len() as u64;
        self.request(bs, offset, bytes, Buf::Read(buf))
    }

    fn pwrite(&self, bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        self.request(bs, offset, buf.len() as u64, Buf::Write(buf))
    }

    /// `bdrv_qed_co_pwrite_zeroes()`.
    fn pwrite_zeroes(
        &self,
        bs: &Node,
        offset: u64,
        bytes: u64,
        _may_unmap: bool,
    ) -> io::Result<()> {
        let cluster_size = u64::from(self.lock().header.cluster_size);
        // Fall back if the request is not aligned
        if offset % cluster_size != 0 || bytes % cluster_size != 0 {
            return Err(errno(libc::ENOTSUP));
        }
        self.request(bs, offset, bytes, Buf::Zero)
    }

    /// Clears the need-check flag, what QEMU's `qed_need_check_timer()` does.
    fn flush_to_os(&self, bs: &Node) -> io::Result<()> {
        let file = bs.file();
        let mut s = self.lock();
        if s.header.features & F_NEED_CHECK == 0 || file.read_only() {
            return Ok(());
        }
        // Ensure writes are on disk before clearing flag
        file.flush()?;
        s.header.features &= !F_NEED_CHECK;
        let _ = s.write_header(&file);
        Ok(())
    }

    /// `bdrv_qed_co_getlength()`.
    fn getlength(&self, _bs: &Node) -> io::Result<u64> {
        Ok(self.lock().header.image_size)
    }

    fn truncate(&self, bs: &Node, len: u64) -> Result<()> {
        self.truncate_full(bs, len, false, PreallocMode::Off, 0)
    }

    /// `bdrv_qed_co_truncate()`: only grows.
    fn truncate_full(
        &self,
        bs: &Node,
        offset: u64,
        _exact: bool,
        prealloc: PreallocMode,
        _flags: u32,
    ) -> Result<()> {
        let mut s = self.lock();
        if prealloc != PreallocMode::Off {
            return Err(Error::generic(format!(
                "Unsupported preallocation mode '{}'",
                prealloc.as_str()
            )));
        }
        if !is_image_size_valid(offset, s.header.cluster_size, s.header.table_size) {
            return Err(Error::generic("Invalid image size specified"));
        }
        if offset < s.header.image_size {
            return Err(Error::generic("Shrinking images is currently not supported"));
        }
        let old_image_size = s.header.image_size;
        s.header.image_size = offset;
        if let Err(e) = s.write_header_sync(&bs.file()) {
            s.header.image_size = old_image_size;
            return Err(Error::from_io("Failed to update the image size", e));
        }
        Ok(())
    }

    /// `bdrv_qed_refresh_limits()`, and the sector alignment of a `.bdrv_co_readv` driver.
    fn refresh_limits(&self, _bs: &Node, bl: &mut BlockLimits) -> Result<()> {
        let cluster_size = self.lock().header.cluster_size;
        bl.request_alignment = BDRV_SECTOR_SIZE as u32;
        bl.pwrite_zeroes_alignment = cluster_size;
        bl.max_pwrite_zeroes = u64::from(i32::MAX as u32 & !(cluster_size - 1));
        Ok(())
    }

    /// `bdrv_qed_co_block_status()`.
    fn block_status(
        &self,
        bs: &Node,
        _want: u32,
        offset: u64,
        bytes: u64,
    ) -> Option<io::Result<BlockStatus>> {
        let file = bs.file();
        let mut s = self.lock();
        let (found, cluster, pnum) = match s.find_cluster(&file, offset, bytes) {
            Ok(r) => r,
            Err(e) => return Some(Err(e)),
        };
        let mut st = BlockStatus { ret: 0, pnum, map: 0, file: None };
        match found {
            Cluster::Found => {
                st.map = cluster | s.offset_into_cluster(offset);
                st.ret = BDRV_BLOCK_DATA | BDRV_BLOCK_OFFSET_VALID;
                st.file = Some(file.clone());
            }
            Cluster::Zero => st.ret = BDRV_BLOCK_ZERO,
            Cluster::L2 | Cluster::L1 => {}
        }
        Some(Ok(st))
    }

    /// `bdrv_qed_close()`.
    fn close(&self, bs: &Node) {
        self.do_close(bs);
    }

    /// `bdrv_qed_reopen_prepare()`: nothing to do.
    fn reopen_prepare(&self, _bs: &Node, _state: &mut ReopenState) -> Option<Result<()>> {
        Some(Ok(()))
    }

    /// `bdrv_qed_co_get_info()`.
    fn get_info(&self, _bs: &Node) -> Option<io::Result<BlockDriverInfo>> {
        let s = self.lock();
        Some(Ok(BlockDriverInfo {
            cluster_size: u64::from(s.header.cluster_size),
            is_dirty: s.header.features & F_NEED_CHECK != 0,
            ..Default::default()
        }))
    }

    /// `bdrv_has_zero_init_1()`.
    fn has_zero_init(&self, _bs: &Node) -> Option<bool> {
        Some(true)
    }

    /// `bdrv_qed_co_check()`.
    fn check(&self, bs: &Node, fix: u32) -> Option<Result<CheckResult>> {
        let file = bs.file();
        let mut s = self.lock();
        let (result, r) = check::qed_check(&mut s, &file, fix != 0);
        Some(match r {
            Ok(()) => Ok(result),
            Err(e) => Err(Error::from_io("Check failed", e)),
        })
    }

    /// `bdrv_qed_co_change_backing_file()`.
    fn change_backing_file(
        &self,
        bs: &Node,
        backing_file: Option<&str>,
        backing_fmt: Option<&str>,
    ) -> Option<io::Result<()>> {
        let mut s = self.lock();
        // Refuse to set backing filename if unknown compat feature bits are active. If the
        // image uses an unknown compat feature then we may not know the layout of data
        // following the header structure and cannot safely add a new string.
        if backing_file.is_some() && s.header.compat_features & !COMPAT_FEATURE_MASK != 0 {
            return Some(Err(errno(libc::ENOTSUP)));
        }

        let mut new_header = s.header;
        new_header.features &= !(F_BACKING_FILE | F_BACKING_FORMAT_NO_PROBE);

        // Adjust feature flags
        if backing_file.is_some() {
            new_header.features |= F_BACKING_FILE;
            if fmt_is_raw(backing_fmt) {
                new_header.features |= F_BACKING_FORMAT_NO_PROBE;
            }
        }

        // Calculate new header size
        let backing = backing_file.unwrap_or_default().as_bytes();
        new_header.backing_filename_offset = HEADER_SIZE as u32;
        new_header.backing_filename_size = backing.len() as u32;
        let buffer_len = HEADER_SIZE as u64 + backing.len() as u64;

        // Make sure we can rewrite header without failing
        if buffer_len > u64::from(new_header.header_size) * u64::from(new_header.cluster_size) {
            return Some(Err(errno(libc::ENOSPC)));
        }

        // Prepare new header
        let mut buffer = new_header.to_le().to_vec();
        buffer.extend_from_slice(backing);

        // Write new header, bdrv_co_pwrite_sync()
        let file = bs.file();
        let r = file.pwrite(0, &buffer).and_then(|()| file.flush());
        if r.is_ok() {
            s.header = new_header;
        }
        Some(r)
    }

    /// `bdrv_qed_co_invalidate_cache()`: closes and opens the image again.
    fn invalidate_cache(&self, bs: &Node) -> Result<()> {
        self.do_close(bs);
        let file = bs.file();
        let (new, backing) =
            do_open(&file, false).map_err(|e| e.prepend("Could not reopen qed layer: "))?;
        *self.lock() = new;
        if let Some((name, raw)) = backing {
            let mut meta = bs.meta.lock().unwrap_or_else(|e| e.into_inner());
            if meta.backing_file != name {
                meta.backing_file = name.clone();
                meta.auto_backing_file = name;
            }
            if raw {
                meta.backing_format = "raw".to_string();
            }
        }
        Ok(())
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

/// Why creating an image failed: an error QEMU reports with `errp`, or an I/O error it only
/// returns.
enum CreateError {
    Err(Error),
    Io(io::Error),
}

impl From<Error> for CreateError {
    fn from(e: Error) -> Self {
        CreateError::Err(e)
    }
}

impl From<io::Error> for CreateError {
    fn from(e: io::Error) -> Self {
        CreateError::Io(e)
    }
}

/// `bdrv_qed_co_create()`.
fn do_create(
    graph: &BlockGraph,
    o: BlockdevCreateOptionsQed,
) -> std::result::Result<(), CreateError> {
    // Validate options and set default values. The sizes are 32 bits wide in QEMU.
    let cluster_size = o.cluster_size.map_or(DEFAULT_CLUSTER_SIZE, |c| c as u32);
    let table_size = o.table_size.map_or(DEFAULT_TABLE_SIZE, |t| t as u32);

    if !is_cluster_size_valid(cluster_size) {
        return Err(Error::generic(format!(
            "QED cluster size must be within range [{MIN_CLUSTER_SIZE}, {MAX_CLUSTER_SIZE}] and \
             power of 2"
        ))
        .into());
    }
    if !is_table_size_valid(table_size) {
        return Err(Error::generic(format!(
            "QED table size must be within range [{MIN_TABLE_SIZE}, {MAX_TABLE_SIZE}] and power \
             of 2"
        ))
        .into());
    }
    if !is_image_size_valid(o.size, cluster_size, table_size) {
        return Err(Error::generic(format!(
            "QED image size must be a non-zero multiple of cluster size and less than {} bytes",
            max_image_size(cluster_size, table_size)
        ))
        .into());
    }

    // Create BlockBackend to write to the image
    let blk = graph.open_create_blk(o.file)?;
    let node = blk.root().expect("a new backend has its node");

    // Prepare image format
    let mut header = Header {
        magic: QED_MAGIC,
        cluster_size,
        table_size,
        header_size: 1,
        l1_table_offset: u64::from(cluster_size),
        image_size: o.size,
        ..Header::default()
    };
    let l1_size = cluster_size as usize * table_size as usize;

    // The QED format associates file length with allocation status, so a new file (which
    // is empty) must have a length of 0.
    node.truncate_full(0, true, PreallocMode::Off, 0)?;

    let backing_file_given = o.backing_file.is_some();
    let backing_file = o.backing_file.unwrap_or_default();
    if backing_file_given {
        header.features |= F_BACKING_FILE;
        header.backing_filename_offset = HEADER_SIZE as u32;
        header.backing_filename_size = backing_file.len() as u32;
        if o.backing_fmt.is_some_and(|f| fmt_is_raw(Some(f.as_str()))) {
            header.features |= F_BACKING_FORMAT_NO_PROBE;
        }
    }

    node.pwrite(0, &header.to_le())?;
    node.pwrite(HEADER_SIZE as u64, backing_file.as_bytes())?;
    node.pwrite(header.l1_table_offset, &vec![0u8; l1_size])?;
    Ok(())
}

/// `bdrv_qed_co_create()`: `blockdev-create` with `driver: qed`. An I/O error fails the job
/// with its `strerror()` text.
fn qed_create(graph: &BlockGraph, options: BlockdevCreateOptionsU) -> Result<()> {
    let BlockdevCreateOptionsU::Qed(o) = options else {
        unreachable!("qed driver with other create options")
    };
    do_create(graph, o).map_err(|e| match e {
        CreateError::Err(e) => e,
        CreateError::Io(e) => Error::generic(strerror(&e)),
    })
}

/// `bdrv_qed_co_create_opts()`: `qemu-img create -f qed`. An I/O error is reported by
/// `bdrv_co_create()` as "Could not create image".
fn qed_create_opts(filename: &str, options: &mut QDict) -> Result<()> {
    let (graph, _blk, qdict) = create_opts_open(
        filename,
        options,
        "qed",
        &["size", "backing_file", "backing_fmt", "cluster_size", "table_size"],
        &[
            ("backing_file", "backing-file"),
            ("backing_fmt", "backing-fmt"),
            ("cluster_size", "cluster-size"),
            ("table_size", "table-size"),
        ],
    )?;
    let create_options = visit_create_options(qdict)?;
    let BlockdevCreateOptionsU::Qed(mut o) = create_options.u else {
        unreachable!("driver is qed")
    };
    // Silently round up size
    o.size = o.size.wrapping_add(BDRV_SECTOR_SIZE - 1) & !(BDRV_SECTOR_SIZE - 1);
    do_create(&graph, o).map_err(|e| match e {
        CreateError::Err(e) => e,
        CreateError::Io(e) => Error::from_io("Could not create image", e),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_round_trip() {
        let h = Header {
            magic: QED_MAGIC,
            cluster_size: 65536,
            table_size: 4,
            header_size: 1,
            features: F_BACKING_FILE,
            compat_features: 2,
            autoclear_features: 3,
            l1_table_offset: 65536,
            image_size: 1 << 30,
            backing_filename_offset: 64,
            backing_filename_size: 9,
        };
        let b = h.to_le();
        assert_eq!(&b[0..4], b"QED\0");
        assert_eq!(Header::from_le(&b), h);
    }

    #[test]
    fn probe() {
        let mut buf = [0u8; 64];
        buf[0..4].copy_from_slice(b"QED\0");
        assert_eq!(qed_probe(&buf, None), 100);
        assert_eq!(qed_probe(&buf[..63], None), 0);
        assert_eq!(qed_probe(&[0u8; 64], None), 0);
    }

    #[test]
    fn sizes() {
        assert!(is_cluster_size_valid(4096));
        assert!(!is_cluster_size_valid(2048));
        assert!(!is_cluster_size_valid(65536 + 4096));
        assert!(is_table_size_valid(16));
        assert!(!is_table_size_valid(3));
        // The default geometry covers 64 TiB.
        assert_eq!(max_image_size(65536, 4), 64 << 40);
        assert!(is_image_size_valid(0, 65536, 4));
        assert!(!is_image_size_valid(511, 65536, 4));
    }

    #[test]
    fn bytes_to_clusters_keeps_the_quirk() {
        let s = State {
            header: Header { cluster_size: 4096, table_size: 1, ..Header::default() },
            l1_table: Vec::new(),
            l2_cache: L2Cache::default(),
            table_nelems: 512,
            l1_shift: 21,
            l2_shift: 12,
            l2_mask: 511,
            file_size: 0,
        };
        assert_eq!(s.bytes_to_clusters(0), 0);
        assert_eq!(s.bytes_to_clusters(1), 1);
        assert_eq!(s.bytes_to_clusters(4096), 1);
        // 4095 clusters worth of bytes counts one extra cluster.
        assert_eq!(s.bytes_to_clusters(4095 * 4096), 4096);
        assert_eq!(s.l1_index(3 << 21), 3);
        assert_eq!(s.l2_index((5 << 12) + 7), 5);
    }

    #[test]
    fn create_opts_order() {
        let names: Vec<&str> = QED_CREATE_OPTS.iter().map(|d| d.name).collect();
        assert_eq!(names, ["size", "backing_file", "backing_fmt", "cluster_size", "table_size"]);
    }
}
