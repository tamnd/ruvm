// SPDX-License-Identifier: GPL-2.0-or-later

//! Opening VMDK images: the descriptor, the extent headers and the grain directories.

use std::io;
use std::sync::{Arc, Mutex};

use ruvm_base::{Error, Result};
use ruvm_qapi::QDict;
use ruvm_qapi::types::BlockdevOptionsU;

use super::{
    DESC_SIZE, Extent, L2_CACHE_SIZE, MARKER_END_OF_STREAM, MARKER_FOOTER, SECTOR_SIZE, State,
    VMDK3_MAGIC, VMDK4_COMPRESSION_DEFLATE, VMDK4_FLAG_MARKER, VMDK4_FLAG_RGD,
    VMDK4_FLAG_ZERO_GRAIN, VMDK4_GD_AT_END, VMDK4_MAGIC, VmdkDriver, be32, le16, le32, le64,
    round_up_mask,
};
use crate::drivers::OpenArgs;
use crate::node::{
    BDRV_CHILD_DATA, BDRV_CHILD_IMAGE, BDRV_CHILD_METADATA, BDRV_CHILD_PRIMARY, Driver, Node, errno,
};

const SESPARSE_CONST_HEADER_MAGIC: u64 = 0x0000_0000_cafe_babe;
const SESPARSE_VOLATILE_HEADER_MAGIC: u64 = 0x0000_0000_cafe_cafe;

/// Offsets into `VMDK4Header`, which is read from byte 4 of the file (after the magic).
pub(super) mod header4 {
    pub(in crate::vmdk) const VERSION: usize = 0;
    pub(in crate::vmdk) const FLAGS: usize = 4;
    pub(in crate::vmdk) const CAPACITY: usize = 8;
    pub(in crate::vmdk) const GRANULARITY: usize = 16;
    pub(in crate::vmdk) const DESC_OFFSET: usize = 24;
    pub(in crate::vmdk) const DESC_SIZE: usize = 32;
    pub(in crate::vmdk) const NUM_GTES_PER_GT: usize = 40;
    pub(in crate::vmdk) const RGD_OFFSET: usize = 44;
    pub(in crate::vmdk) const GD_OFFSET: usize = 52;
    pub(in crate::vmdk) const GRAIN_OFFSET: usize = 60;
    pub(in crate::vmdk) const CHECK_BYTES: usize = 69;
    pub(in crate::vmdk) const COMPRESS_ALGORITHM: usize = 73;
    /// `sizeof(VMDK4Header)`.
    pub(in crate::vmdk) const SIZE: usize = 75;
}
use header4 as h4;

/// Offsets into `VMDK3Header`, also read from byte 4.
mod h3 {
    pub(super) const DISK_SECTORS: usize = 8;
    pub(super) const GRANULARITY: usize = 12;
    pub(super) const L1DIR_OFFSET: usize = 16;
    pub(super) const L1DIR_SIZE: usize = 20;
    pub(super) const SIZE: usize = 40;
}

/// What an open collects before the driver exists.
struct OpenState {
    /// `bs->file`.
    top: Arc<Node>,
    extents: Vec<Extent>,
    create_type: Option<String>,
    desc_offset: u64,
}

/// A C string: `buf` up to its first NUL.
pub(super) fn cstr(buf: &[u8]) -> &[u8] {
    match buf.iter().position(|&b| b == 0) {
        Some(n) => &buf[..n],
        None => buf,
    }
}

/// `strstr()`.
pub(super) fn strstr(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

/// `isspace()` in the C locale.
fn is_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\x0b' | b'\x0c' | b'\r')
}

/// `bdrv_refresh_filename()` and `bs->filename`.
fn node_filename(node: &Node) -> String {
    node.refresh_filename();
    node.filename().unwrap_or_default()
}

/// `bdrv_open_driver()` without a message from the driver: "Could not open '%s'".
fn open_err(top: &Node, e: io::Error) -> Error {
    let name = top.filename().unwrap_or_default();
    Error::from_io(format!("Could not open '{name}'"), e)
}

/// `vmdk_read_desc()`: up to 1 MiB from `desc_offset`, NUL terminated.
fn read_desc(file: &Node, desc_offset: u64) -> Result<Vec<u8>> {
    let size = file.getlength().map_err(|e| Error::from_io("Could not access file", e))?;
    if size < 4 {
        // Both a descriptor file and a sparse image are much larger than 4 bytes, and the
        // callers look at the first 4 bytes for the magic.
        return Err(Error::generic("File is too small, not a valid image"));
    }
    // Avoid unbounded allocation.
    let size = size.min((1 << 20) - 1) as usize;
    let mut buf = vec![0u8; size + 1];
    file.pread(desc_offset, &mut buf[..size])
        .map_err(|e| Error::from_io("Could not read from file", e))?;
    Ok(buf)
}

/// `vmdk_read_cid()`: the `CID` (or with `parent`, the `parentCID`) of the descriptor at
/// `desc_offset` in `file`.
pub(super) fn read_cid(file: &Node, desc_offset: u64, parent: bool) -> io::Result<u32> {
    let mut desc = vec![0u8; DESC_SIZE];
    file.pread(desc_offset, &mut desc)?;
    desc[DESC_SIZE - 1] = 0;
    read_cid_from(&desc, parent).ok_or_else(|| errno(libc::EINVAL))
}

/// The `CID` or `parentCID` in the descriptor text `desc`.
pub(super) fn read_cid_from(desc: &[u8], parent: bool) -> Option<u32> {
    let name: &[u8] = if parent { b"parentCID" } else { b"CID" };
    let s = cstr(desc);
    let pos = strstr(s, name)?;
    // sizeof("CID") skips the name and the '='.
    let start = (pos + name.len() + 1).min(s.len());
    scan_hex(&s[start..])
}

/// `sscanf(s, "%" SCNx32)`.
fn scan_hex(s: &[u8]) -> Option<u32> {
    let mut i = 0;
    while i < s.len() && is_space(s[i]) {
        i += 1;
    }
    let mut neg = false;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        neg = s[i] == b'-';
        i += 1;
    }
    let mut v: u64 = 0;
    let mut digits = 0;
    if i + 1 < s.len() && s[i] == b'0' && (s[i + 1] | 0x20) == b'x' {
        // "0x" with no hex digit after it reads as the 0.
        i += 2;
        digits = 1;
    }
    while i < s.len() {
        let Some(d) = (s[i] as char).to_digit(16) else { break };
        v = v.saturating_mul(16).saturating_add(u64::from(d));
        digits += 1;
        i += 1;
    }
    if digits == 0 {
        return None;
    }
    let v = if v > u64::from(u32::MAX) { u32::MAX } else { v as u32 };
    Some(if neg { v.wrapping_neg() } else { v })
}

/// `vmdk_parent_open()`: the parent named by `parentFileNameHint`.
fn parent_open(args: &mut OpenArgs<'_>, file: &Node, desc_offset: u64) -> io::Result<()> {
    let mut desc = vec![0u8; DESC_SIZE + 1];
    file.pread(desc_offset, &mut desc[..DESC_SIZE])?;
    let s = cstr(&desc);
    let name: &[u8] = b"parentFileNameHint";
    if let Some(pos) = strstr(s, name) {
        // sizeof("parentFileNameHint") + 1 skips the name, '=' and '"'.
        let start = (pos + name.len() + 2).min(s.len());
        let rest = &s[start..];
        let end = rest.iter().position(|&b| b == b'"').ok_or_else(|| errno(libc::EINVAL))?;
        // PATH_MAX - 1.
        if end > 4095 {
            return Err(errno(libc::EINVAL));
        }
        let parent = String::from_utf8_lossy(&rest[..end]).into_owned();
        args.set_backing_file(&parent, Some("vmdk"));
        args.meta.auto_backing_file = parent;
    }
    Ok(())
}

/// The arguments of `vmdk_add_extent()` that describe the grain tables.
struct Tables {
    l1_offset: u64,
    l1_backup_offset: u64,
    l1_size: u32,
    l2_size: u32,
    cluster_sectors: u64,
}

/// `vmdk_add_extent()`: appends an extent to `st.extents`.
fn add_extent(
    st: &mut OpenState,
    file: &Node,
    child: &str,
    flat: bool,
    sectors: u64,
    t: Tables,
) -> Result<()> {
    // Grains of 0x200000 sectors (1 GiB) are unrealistic. A grain size of 0 would divide by
    // zero later on.
    if t.cluster_sectors > 0x20_0000 || (!flat && t.cluster_sectors == 0) {
        return Err(Error::generic("Invalid granularity, image may be corrupt"));
    }
    // Enough for 8 TB of VMDK3 or VMDK4 with the smallest grains and grain tables, and for
    // 64 TB of seSparse; both formats allow less.
    if t.l1_size > 32 * 1024 * 1024 {
        return Err(Error::generic("L1 size too big"));
    }
    let nb_sectors = file.nb_sectors().map_err(|e| open_err(&st.top, e))?;
    let end_sector = st.extents.last().map_or(0, |e| e.end_sector).wrapping_add(sectors);
    st.extents.push(Extent {
        child: child.to_string(),
        flat,
        sectors,
        l1_table_offset: t.l1_offset,
        l1_backup_table_offset: t.l1_backup_offset,
        l1_size: t.l1_size,
        // An uint32_t in QEMU.
        l1_entry_sectors: u64::from(t.l2_size).wrapping_mul(t.cluster_sectors) as u32,
        l2_size: t.l2_size,
        cluster_sectors: if flat { sectors } else { t.cluster_sectors },
        next_cluster_sector: round_up_mask(nb_sectors, t.cluster_sectors),
        entry_size: 4,
        end_sector,
        ..Extent::default()
    });
    Ok(())
}

/// `vmdk_init_tables()`: reads the grain directory and its backup of the last extent.
fn init_tables(st: &mut OpenState, file: &Node) -> Result<()> {
    let e = st.extents.last_mut().expect("an extent was just added");
    let l1_bytes = e.l1_size as usize * e.entry_size as usize;
    let mut raw = vec![0u8; l1_bytes];
    if let Err(err) = file.pread(e.l1_table_offset, &mut raw) {
        let r = Error::from_io(
            format!("Could not read l1 table from extent '{}'", node_filename(file)),
            err,
        );
        st.extents.pop();
        return Err(r);
    }
    e.l1_table = if e.entry_size == 8 {
        raw.chunks_exact(8).map(|c| le64(c, 0)).collect()
    } else {
        raw.chunks_exact(4).map(|c| u64::from(le32(c, 0))).collect()
    };

    if e.l1_backup_table_offset != 0 {
        let mut raw = vec![0u8; l1_bytes];
        if let Err(err) = file.pread(e.l1_backup_table_offset, &mut raw) {
            let r = Error::from_io(
                format!("Could not read l1 backup table from extent '{}'", node_filename(file)),
                err,
            );
            st.extents.pop();
            return Err(r);
        }
        e.l1_backup_table = raw.chunks_exact(4).map(|c| le32(c, 0)).collect();
    }
    e.l2_cache = vec![0u8; e.entry_size as usize * e.l2_size as usize * L2_CACHE_SIZE];
    Ok(())
}

/// `vmdk_open_vmfs_sparse()`: a VMDK3 ("COWD") extent.
fn open_vmfs_sparse(st: &mut OpenState, file: &Node, child: &str) -> Result<()> {
    let mut h = [0u8; h3::SIZE];
    file.pread(4, &mut h).map_err(|e| {
        Error::from_io(format!("Could not read header from file '{}'", node_filename(file)), e)
    })?;
    let t = Tables {
        l1_offset: u64::from(le32(&h, h3::L1DIR_OFFSET)) << 9,
        l1_backup_offset: 0,
        l1_size: le32(&h, h3::L1DIR_SIZE),
        l2_size: 4096,
        cluster_sectors: u64::from(le32(&h, h3::GRANULARITY)),
    };
    add_extent(st, file, child, false, u64::from(le32(&h, h3::DISK_SECTORS)), t)?;
    init_tables(st, file)
}

/// `check_se_sparse_const_header()`: strict checks, the format is not documented.
fn check_se_sparse_const_header(h: &[u8]) -> Result<()> {
    let magic = le64(h, 0);
    let version = le64(h, 8);
    let grain_size = le64(h, 24);
    let grain_table_size = le64(h, 32);
    let flags = le64(h, 40);
    let reserved = [le64(h, 48), le64(h, 56), le64(h, 64), le64(h, 72)];
    if magic != SESPARSE_CONST_HEADER_MAGIC {
        return Err(Error::generic(format!("Bad const header magic: 0x{magic:016x}")));
    }
    if version != 0x0000_0002_0000_0001 {
        return Err(Error::generic(format!("Unsupported version: 0x{version:016x}")));
    }
    if grain_size != 8 {
        return Err(Error::generic(format!("Unsupported grain size: {grain_size}")));
    }
    if grain_table_size != 64 {
        return Err(Error::generic(format!("Unsupported grain table size: {grain_table_size}")));
    }
    if flags != 0 {
        return Err(Error::generic(format!("Unsupported flags: 0x{flags:016x}")));
    }
    if reserved.iter().any(|&r| r != 0) {
        return Err(Error::generic(format!(
            "Unsupported reserved bits: 0x{:016x} 0x{:016x} 0x{:016x} 0x{:016x}",
            reserved[0], reserved[1], reserved[2], reserved[3]
        )));
    }
    if h[208..512].iter().any(|&b| b != 0) {
        return Err(Error::generic("Unsupported non-zero const header padding"));
    }
    Ok(())
}

/// `check_se_sparse_volatile_header()`.
fn check_se_sparse_volatile_header(h: &[u8]) -> Result<()> {
    let magic = le64(h, 0);
    if magic != SESPARSE_VOLATILE_HEADER_MAGIC {
        return Err(Error::generic(format!("Bad volatile header magic: 0x{magic:016x}")));
    }
    if le64(h, 24) != 0 {
        return Err(Error::generic("Image is dirty, Replaying journal not supported"));
    }
    if h[32..512].iter().any(|&b| b != 0) {
        return Err(Error::generic("Unsupported non-zero volatile header padding"));
    }
    Ok(())
}

/// `vmdk_open_se_sparse()`: an ESXi seSparse extent, read only.
fn open_se_sparse(
    args: &mut OpenArgs<'_>,
    st: &mut OpenState,
    file: &Node,
    child: &str,
) -> Result<()> {
    args.apply_auto_read_only(Some("No write support for seSparse images available"))?;

    let mut ch = [0u8; 512];
    file.pread(0, &mut ch).map_err(|e| {
        Error::from_io(
            format!("Could not read const header from file '{}'", node_filename(file)),
            e,
        )
    })?;
    check_se_sparse_const_header(&ch)?;

    let volatile_header_offset = le64(&ch, 80);
    let mut vh = [0u8; 512];
    file.pread(volatile_header_offset.wrapping_mul(SECTOR_SIZE), &mut vh).map_err(|e| {
        Error::from_io(
            format!("Could not read volatile header from file '{}'", node_filename(file)),
            e,
        )
    })?;
    check_se_sparse_volatile_header(&vh)?;

    let capacity = le64(&ch, 16);
    let grain_size = le64(&ch, 24);
    let grain_table_size = le64(&ch, 32);
    let grain_dir_offset = le64(&ch, 128);
    let grain_dir_size = le64(&ch, 136);
    let grain_tables_offset = le64(&ch, 144);
    let grains_offset = le64(&ch, 192);
    let t = Tables {
        l1_offset: grain_dir_offset.wrapping_mul(SECTOR_SIZE),
        l1_backup_offset: 0,
        // An uint32_t and an int in QEMU.
        l1_size: (grain_dir_size.wrapping_mul(SECTOR_SIZE) / 8) as u32,
        l2_size: (grain_table_size * SECTOR_SIZE / 8) as u32,
        cluster_sectors: grain_size,
    };
    add_extent(st, file, child, false, capacity, t)?;
    let e = st.extents.last_mut().expect("just added");
    e.sesparse = true;
    e.sesparse_l2_tables_offset = grain_tables_offset;
    e.sesparse_clusters_offset = grains_offset;
    e.entry_size = 8;
    init_tables(st, file)
}

/// `vmdk_open_vmdk4()`: a hosted sparse ("KDMV") extent.
fn open_vmdk4(
    args: &mut OpenArgs<'_>,
    st: &mut OpenState,
    file: &Arc<Node>,
    child: &str,
) -> Result<()> {
    let mut header = [0u8; h4::SIZE];
    file.pread(4, &mut header).map_err(|e| {
        Error::from_io(format!("Could not read header from file '{}'", node_filename(file)), e)
    })?;
    if le64(&header, h4::CAPACITY) == 0 {
        let desc_offset = le64(&header, h4::DESC_OFFSET);
        if desc_offset != 0 {
            let buf = read_desc(file, desc_offset << 9)?;
            return open_desc_file(args, st, &buf);
        }
    }

    if st.create_type.is_none() {
        st.create_type = Some("monolithicSparse".to_string());
    }

    if le64(&header, h4::GD_OFFSET) == VMDK4_GD_AT_END {
        // The footer takes precedence over the header. It starts 1024 bytes before the end:
        // one sector for the footer, and another one for the end-of-stream marker.
        let mut footer = [0u8; 1536];
        let end = file
            .nb_sectors()
            .map_err(|e| Error::from_io("Failed to read footer", e))?
            .wrapping_mul(SECTOR_SIZE);
        let at = end
            .checked_sub(1536)
            .ok_or_else(|| Error::from_io("Failed to read footer", errno(libc::EIO)))?;
        file.pread(at, &mut footer).map_err(|e| Error::from_io("Failed to read footer", e))?;
        if be32(&footer, 512) != VMDK4_MAGIC
            || le32(&footer, 8) != 0
            || le32(&footer, 12) != MARKER_FOOTER
            || le64(&footer, 1024) != 0
            || le32(&footer, 1024 + 8) != 0
            || le32(&footer, 1024 + 12) != MARKER_END_OF_STREAM
        {
            return Err(Error::generic("Invalid footer"));
        }
        header.copy_from_slice(&footer[516..516 + h4::SIZE]);
    }

    let version = le32(&header, h4::VERSION);
    let flags = le32(&header, h4::FLAGS);
    let compressed = le16(&header, h4::COMPRESS_ALGORITHM) == VMDK4_COMPRESSION_DEFLATE;
    if version > 3 {
        return Err(Error::generic(format!("Unsupported VMDK version {version}")));
    } else if version == 3 && !args.flags.read_only && !compressed {
        // VMware KB 2064959: version 3 added persistent changed block tracking, which
        // backup software may ignore and read as version 1. Reading is safe.
        return Err(Error::generic("VMDK version 3 must be read only"));
    }

    let num_gtes_per_gt = le32(&header, h4::NUM_GTES_PER_GT);
    if num_gtes_per_gt > 512 {
        return Err(Error::generic("L2 table size too big"));
    }
    let granularity = le64(&header, h4::GRANULARITY);
    let l1_entry_sectors = u64::from(num_gtes_per_gt).wrapping_mul(granularity) as u32;
    if l1_entry_sectors == 0 {
        return Err(Error::generic("L1 entry size is invalid"));
    }
    let capacity = le64(&header, h4::CAPACITY);
    let l1_size = (capacity.wrapping_add(u64::from(l1_entry_sectors)).wrapping_sub(1)
        / u64::from(l1_entry_sectors)) as u32;
    let l1_backup_offset =
        if flags & VMDK4_FLAG_RGD != 0 { le64(&header, h4::RGD_OFFSET) << 9 } else { 0 };
    let grain_offset = le64(&header, h4::GRAIN_OFFSET);
    let nb_sectors = file.nb_sectors().map_err(|e| open_err(&st.top, e))?;
    if (nb_sectors as i64) < grain_offset as i64 {
        return Err(Error::generic(format!(
            "File truncated, expecting at least {} bytes",
            (grain_offset as i64).wrapping_mul(SECTOR_SIZE as i64)
        )));
    }

    let t = Tables {
        l1_offset: le64(&header, h4::GD_OFFSET) << 9,
        l1_backup_offset,
        l1_size,
        l2_size: num_gtes_per_gt,
        cluster_sectors: granularity,
    };
    add_extent(st, file, child, false, capacity, t)?;
    let e = st.extents.last_mut().expect("just added");
    e.compressed = compressed;
    if compressed {
        st.create_type = Some("streamOptimized".to_string());
    }
    e.has_marker = flags & VMDK4_FLAG_MARKER != 0;
    e.version = version;
    e.has_zero_grain = flags & VMDK4_FLAG_ZERO_GRAIN != 0;
    init_tables(st, file)
}

/// `vmdk_parse_description()`: the quoted value of `name` in the descriptor.
pub(super) fn parse_description(desc: &[u8], name: &str) -> Option<String> {
    let desc = cstr(desc);
    let pos = strstr(desc, name.as_bytes())?;
    // Skip the "=\"" after the name.
    let start = pos + name.len() + 2;
    if start >= desc.len() {
        return None;
    }
    let len = desc[start..].iter().position(|&b| b == b'"')?;
    // The buffer is 128 bytes.
    if len + 1 > 128 {
        return None;
    }
    Some(String::from_utf8_lossy(&desc[start..start + len]).into_owned())
}

/// `vmdk_open_sparse()`: a sparse extent by its magic.
fn open_sparse(
    args: &mut OpenArgs<'_>,
    st: &mut OpenState,
    file: &Arc<Node>,
    child: &str,
    buf: &[u8],
) -> Result<()> {
    match be32(buf, 0) {
        VMDK3_MAGIC => open_vmfs_sparse(st, file, child),
        VMDK4_MAGIC => open_vmdk4(args, st, file, child),
        _ => Err(Error::generic("Image not in VMDK format")),
    }
}

/// The start of the line after the one at `p`, `next_line()`.
fn next_line(s: &[u8], mut p: usize) -> usize {
    while p < s.len() {
        if s[p] == b'\n' {
            return p + 1;
        }
        p += 1;
    }
    p
}

/// An extent line as `sscanf(p, "%10s %" SCNd64 " %10s \"%511[^\n\r\"]\" %" SCNd64, ...)`
/// reads it.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct ExtentLine {
    /// The number of conversions, -1 for `EOF`.
    pub matches: i32,
    pub access: Vec<u8>,
    pub sectors: i64,
    pub ty: Vec<u8>,
    pub fname: Vec<u8>,
    pub flat_offset: i64,
}

/// A `sscanf()` over a C string.
struct Scanner<'a> {
    s: &'a [u8],
    i: usize,
}

impl Scanner<'_> {
    fn skip_space(&mut self) {
        while self.i < self.s.len() && is_space(self.s[self.i]) {
            self.i += 1;
        }
    }

    /// `%Ns`.
    fn string(&mut self, width: usize) -> Option<Vec<u8>> {
        self.skip_space();
        let start = self.i;
        while self.i < self.s.len() && self.i - start < width && !is_space(self.s[self.i]) {
            self.i += 1;
        }
        (self.i > start).then(|| self.s[start..self.i].to_vec())
    }

    /// `%ld`.
    fn int(&mut self) -> Option<i64> {
        self.skip_space();
        let mut j = self.i;
        let mut neg = false;
        if j < self.s.len() && (self.s[j] == b'+' || self.s[j] == b'-') {
            neg = self.s[j] == b'-';
            j += 1;
        }
        let digits_start = j;
        let mut v: i64 = 0;
        while j < self.s.len() && self.s[j].is_ascii_digit() {
            let d = i64::from(self.s[j] - b'0');
            v = if neg {
                v.saturating_mul(10).saturating_sub(d)
            } else {
                v.saturating_mul(10).saturating_add(d)
            };
            j += 1;
        }
        if j == digits_start {
            return None;
        }
        self.i = j;
        Some(v)
    }

    /// A literal character of the format.
    fn literal(&mut self, c: u8) -> bool {
        if self.i < self.s.len() && self.s[self.i] == c {
            self.i += 1;
            true
        } else {
            false
        }
    }

    /// `%N[^...]`.
    fn not_in(&mut self, width: usize, set: &[u8]) -> Option<Vec<u8>> {
        let start = self.i;
        while self.i < self.s.len() && self.i - start < width && !set.contains(&self.s[self.i]) {
            self.i += 1;
        }
        (self.i > start).then(|| self.s[start..self.i].to_vec())
    }
}

/// Reads an extent line starting at `s`, the rest of the descriptor.
pub(super) fn scan_extent_line(s: &[u8]) -> ExtentLine {
    let mut l = ExtentLine { flat_offset: -1, ..ExtentLine::default() };
    let mut sc = Scanner { s, i: 0 };
    sc.skip_space();
    if sc.i >= s.len() {
        l.matches = -1;
        return l;
    }
    let Some(access) = sc.string(10) else { return l };
    l.access = access;
    l.matches = 1;
    let Some(sectors) = sc.int() else { return l };
    l.sectors = sectors;
    l.matches = 2;
    let Some(ty) = sc.string(10) else { return l };
    l.ty = ty;
    l.matches = 3;
    sc.skip_space();
    if !sc.literal(b'"') {
        return l;
    }
    let Some(fname) = sc.not_in(511, b"\n\r\"") else { return l };
    l.fname = fname;
    l.matches = 4;
    if !sc.literal(b'"') {
        return l;
    }
    sc.skip_space();
    if let Some(off) = sc.int() {
        l.flat_offset = off;
        l.matches = 5;
    }
    l
}

/// `vmdk_parse_extents()`.
fn parse_extents(args: &mut OpenArgs<'_>, st: &mut OpenState, desc: &[u8]) -> Result<()> {
    let desc = cstr(desc);
    let mut desc_file_dir: Option<String> = None;
    let mut p = 0;
    while p < desc.len() {
        // RW [size in sectors] FLAT "file-name.vmdk" OFFSET
        // RW [size in sectors] SPARSE "file-name.vmdk"
        // RW [size in sectors] VMFS "file-name.vmdk"
        // RW [size in sectors] VMFSSPARSE "file-name.vmdk"
        // RW [size in sectors] SESPARSE "file-name.vmdk"
        let line = p;
        p = next_line(desc, p);
        let mut l = scan_extent_line(&desc[line..]);
        if l.matches < 4 || l.access != b"RW" {
            continue;
        }
        let ty = l.ty.as_slice();
        let invalid = if ty == b"FLAT" {
            l.matches != 5 || l.flat_offset < 0
        } else if ty == b"VMFS" {
            if l.matches == 4 {
                l.flat_offset = 0;
                false
            } else {
                true
            }
        } else {
            l.matches != 4
        };
        if invalid {
            let mut end = next_line(desc, line);
            if desc[end - 1] == b'\n' {
                end -= 1;
            }
            return Err(Error::generic(format!(
                "Invalid extent line: {}",
                String::from_utf8_lossy(&desc[line..end])
            )));
        }

        let known = [&b"FLAT"[..], b"SPARSE", b"VMFS", b"VMFSSPARSE", b"SESPARSE"];
        if l.sectors <= 0 || !known.contains(&ty) {
            continue;
        }

        let fname = String::from_utf8_lossy(&l.fname).into_owned();
        let extent_path = if crate::open::path_is_absolute(&fname) {
            fname
        } else {
            if desc_file_dir.is_none() {
                match crate::open::dirname(&st.top) {
                    Ok(d) => desc_file_dir = Some(d),
                    Err(e) => {
                        return Err(e.prepend(format!(
                            "Cannot use relative paths with VMDK descriptor file '{}': ",
                            node_filename(&st.top)
                        )));
                    }
                }
            }
            format!("{}{fname}", desc_file_dir.as_deref().unwrap_or_default())
        };

        let child = format!("extents.{}", st.extents.len());
        let flat = ty == b"FLAT" || ty == b"VMFS";
        let mut role = BDRV_CHILD_DATA;
        if !flat {
            // Non-flat extents have metadata.
            role |= BDRV_CHILD_METADATA;
        }
        let ctx = args.child_ctx(role);
        let extent_file =
            args.graph.open_qdict(Some(&extent_path), QDict::new(), ctx, args.pending, true)?;
        args.add_child(&child, extent_file.clone(), role);

        let ty_str = String::from_utf8_lossy(ty).into_owned();
        if flat {
            let t = Tables {
                l1_offset: 0,
                l1_backup_offset: 0,
                l1_size: 0,
                l2_size: 0,
                cluster_sectors: 0,
            };
            add_extent(st, &extent_file, &child, true, l.sectors as u64, t)?;
            st.extents.last_mut().expect("just added").flat_start_offset =
                (l.flat_offset as u64) << 9;
        } else if ty == b"SPARSE" || ty == b"VMFSSPARSE" {
            // Both are "COWD" sparse files.
            let buf = read_desc(&extent_file, 0)?;
            open_sparse(args, st, &extent_file, &child, &buf)?;
        } else {
            open_se_sparse(args, st, &extent_file, &child)?;
        }
        st.extents.last_mut().expect("an extent was added").ty = ty_str;
    }
    Ok(())
}

/// `vmdk_open_desc_file()`.
fn open_desc_file(args: &mut OpenArgs<'_>, st: &mut OpenState, buf: &[u8]) -> Result<()> {
    let Some(ct) = parse_description(buf, "createType") else {
        return Err(Error::generic("invalid VMDK image descriptor"));
    };
    let supported = [
        "monolithicFlat",
        "vmfs",
        "vmfsSparse",
        "seSparse",
        "twoGbMaxExtentSparse",
        "twoGbMaxExtentFlat",
    ];
    if !supported.contains(&ct.as_str()) {
        return Err(Error::generic(format!("Unsupported image type '{ct}'")));
    }
    st.create_type = Some(ct);
    st.desc_offset = 0;
    parse_extents(args, st, buf)
}

/// `vmdk_open()`.
pub(super) fn vmdk_open(
    args: &mut OpenArgs<'_>,
    opts: BlockdevOptionsU,
) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::Vmdk(o) = opts else { unreachable!("vmdk driver with other options") };
    let o = *o;
    args.set_backing_option(o.backing.map(|b| *b));
    let file = args.open_child(*o.file, "file", BDRV_CHILD_IMAGE | BDRV_CHILD_PRIMARY)?;
    let mut st =
        OpenState { top: file.clone(), extents: Vec::new(), create_type: None, desc_offset: 0 };

    let buf = read_desc(&file, 0)?;
    match be32(&buf, 0) {
        VMDK3_MAGIC | VMDK4_MAGIC => {
            open_sparse(args, &mut st, &file, "file", &buf)?;
            st.desc_offset = 0x200;
        }
        _ => {
            // No data in the descriptor file.
            if let Some(c) = args.children.iter_mut().find(|c| c.0 == "file") {
                c.2 &= !BDRV_CHILD_DATA;
            }
            open_desc_file(args, &mut st, &buf)?;
        }
    }

    // Try to open the parent image, if there is one.
    parent_open(args, &file, st.desc_offset).map_err(|e| open_err(&file, e))?;
    let cid = read_cid(&file, st.desc_offset, false).map_err(|e| open_err(&file, e))?;
    let parent_cid = read_cid(&file, st.desc_offset, true).map_err(|e| open_err(&file, e))?;

    let total_sectors = st.extents.last().map_or(0, |e| e.end_sector);
    Ok(Box::new(VmdkDriver {
        desc_offset: st.desc_offset,
        cid,
        parent_cid,
        create_type: st.create_type.unwrap_or_default(),
        total_sectors,
        lock: Mutex::new(State { extents: st.extents, cid_updated: false, cid_checked: false }),
    }))
}
