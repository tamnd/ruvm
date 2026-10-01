// SPDX-License-Identifier: GPL-2.0-or-later

//! The image header and its extensions: `QCowHeader`, `qcow2_read_extensions()`,
//! `qcow2_update_header()` and the feature name table, from block/qcow2.c.

use std::io;

use ruvm_base::report::warn_report;
use ruvm_base::{Error, Result};

use super::state::*;
use crate::node::errno;

pub(crate) const QCOW2_EXT_MAGIC_END: u32 = 0;
pub(crate) const QCOW2_EXT_MAGIC_BACKING_FORMAT: u32 = 0xe279_2aca;
pub(crate) const QCOW2_EXT_MAGIC_FEATURE_TABLE: u32 = 0x6803_f857;
pub(crate) const QCOW2_EXT_MAGIC_CRYPTO_HEADER: u32 = 0x0537_be77;
pub(crate) const QCOW2_EXT_MAGIC_BITMAPS: u32 = 0x2385_2875;
pub(crate) const QCOW2_EXT_MAGIC_DATA_FILE: u32 = 0x4441_5441;

pub(crate) const QCOW2_FEAT_TYPE_INCOMPATIBLE: u8 = 0;
pub(crate) const QCOW2_FEAT_TYPE_COMPATIBLE: u8 = 1;
pub(crate) const QCOW2_FEAT_TYPE_AUTOCLEAR: u8 = 2;

/// `sizeof(QCowHeader)`: the version 3 header with the compression type and padding.
pub(crate) const HEADER_SIZE: usize = 112;
/// `offsetof(QCowHeader, incompatible_features)`: the length of a version 2 header.
pub(crate) const HEADER_V2_SIZE: usize = 72;
/// `offsetof(QCowHeader, compression_type)`.
pub(crate) const HEADER_V3_MIN_SIZE: usize = 104;
/// `offsetof(QCowHeader, refcount_table_offset)`.
pub(crate) const REFCOUNT_TABLE_OFFSET_OFFSET: u64 = 48;
/// `offsetof(QCowHeader, l1_size)`.
pub(crate) const L1_SIZE_OFFSET: u64 = 36;
/// `offsetof(QCowHeader, nb_snapshots)`.
pub(crate) const NB_SNAPSHOTS_OFFSET: u64 = 60;
/// `offsetof(QCowHeader, size)`.
pub(crate) const SIZE_OFFSET: u64 = 24;
/// `sizeof(bs->backing_format)`.
pub(crate) const BACKING_FORMAT_SIZE: u32 = 16;

/// The feature name table QEMU writes, `(type, bit, name)`.
pub(crate) const FEATURES: [(u8, u32, &str); 8] = [
    (QCOW2_FEAT_TYPE_INCOMPATIBLE, QCOW2_INCOMPAT_DIRTY_BITNR, "dirty bit"),
    (QCOW2_FEAT_TYPE_INCOMPATIBLE, QCOW2_INCOMPAT_CORRUPT_BITNR, "corrupt bit"),
    (QCOW2_FEAT_TYPE_INCOMPATIBLE, QCOW2_INCOMPAT_DATA_FILE_BITNR, "external data file"),
    (QCOW2_FEAT_TYPE_INCOMPATIBLE, QCOW2_INCOMPAT_COMPRESSION_BITNR, "compression type"),
    (QCOW2_FEAT_TYPE_INCOMPATIBLE, QCOW2_INCOMPAT_EXTL2_BITNR, "extended L2 entries"),
    (QCOW2_FEAT_TYPE_COMPATIBLE, QCOW2_COMPAT_LAZY_REFCOUNTS_BITNR, "lazy refcounts"),
    (QCOW2_FEAT_TYPE_AUTOCLEAR, QCOW2_AUTOCLEAR_BITMAPS_BITNR, "bitmaps"),
    (QCOW2_FEAT_TYPE_AUTOCLEAR, QCOW2_AUTOCLEAR_DATA_FILE_RAW_BITNR, "raw external data"),
];

pub(crate) fn be16(b: &[u8], off: usize) -> u16 {
    u16::from_be_bytes(b[off..off + 2].try_into().unwrap())
}

pub(crate) fn be32(b: &[u8], off: usize) -> u32 {
    u32::from_be_bytes(b[off..off + 4].try_into().unwrap())
}

pub(crate) fn be64(b: &[u8], off: usize) -> u64 {
    u64::from_be_bytes(b[off..off + 8].try_into().unwrap())
}

/// `QCowHeader`, in host byte order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Header {
    pub magic: u32,
    pub version: u32,
    pub backing_file_offset: u64,
    pub backing_file_size: u32,
    pub cluster_bits: u32,
    pub size: u64,
    pub crypt_method: u32,
    pub l1_size: u32,
    pub l1_table_offset: u64,
    pub refcount_table_offset: u64,
    pub refcount_table_clusters: u32,
    pub nb_snapshots: u32,
    pub snapshots_offset: u64,
    pub incompatible_features: u64,
    pub compatible_features: u64,
    pub autoclear_features: u64,
    pub refcount_order: u32,
    pub header_length: u32,
    pub compression_type: u8,
}

impl Header {
    /// Decodes the fixed part of the header. Fields past what `b` holds read as zero.
    pub(crate) fn parse(b: &[u8]) -> Header {
        let mut buf = [0u8; HEADER_SIZE];
        let n = b.len().min(HEADER_SIZE);
        buf[..n].copy_from_slice(&b[..n]);
        let b = &buf;
        Header {
            magic: be32(b, 0),
            version: be32(b, 4),
            backing_file_offset: be64(b, 8),
            backing_file_size: be32(b, 16),
            cluster_bits: be32(b, 20),
            size: be64(b, 24),
            crypt_method: be32(b, 32),
            l1_size: be32(b, 36),
            l1_table_offset: be64(b, 40),
            refcount_table_offset: be64(b, 48),
            refcount_table_clusters: be32(b, 56),
            nb_snapshots: be32(b, 60),
            snapshots_offset: be64(b, 64),
            incompatible_features: be64(b, 72),
            compatible_features: be64(b, 80),
            autoclear_features: be64(b, 88),
            refcount_order: be32(b, 96),
            header_length: be32(b, 100),
            compression_type: b[104],
        }
    }

    /// Encodes the full 112 byte header.
    pub(crate) fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut b = [0u8; HEADER_SIZE];
        b[0..4].copy_from_slice(&self.magic.to_be_bytes());
        b[4..8].copy_from_slice(&self.version.to_be_bytes());
        b[8..16].copy_from_slice(&self.backing_file_offset.to_be_bytes());
        b[16..20].copy_from_slice(&self.backing_file_size.to_be_bytes());
        b[20..24].copy_from_slice(&self.cluster_bits.to_be_bytes());
        b[24..32].copy_from_slice(&self.size.to_be_bytes());
        b[32..36].copy_from_slice(&self.crypt_method.to_be_bytes());
        b[36..40].copy_from_slice(&self.l1_size.to_be_bytes());
        b[40..48].copy_from_slice(&self.l1_table_offset.to_be_bytes());
        b[48..56].copy_from_slice(&self.refcount_table_offset.to_be_bytes());
        b[56..60].copy_from_slice(&self.refcount_table_clusters.to_be_bytes());
        b[60..64].copy_from_slice(&self.nb_snapshots.to_be_bytes());
        b[64..72].copy_from_slice(&self.snapshots_offset.to_be_bytes());
        b[72..80].copy_from_slice(&self.incompatible_features.to_be_bytes());
        b[80..88].copy_from_slice(&self.compatible_features.to_be_bytes());
        b[88..96].copy_from_slice(&self.autoclear_features.to_be_bytes());
        b[96..100].copy_from_slice(&self.refcount_order.to_be_bytes());
        b[100..104].copy_from_slice(&self.header_length.to_be_bytes());
        b[104] = self.compression_type;
        b
    }
}

/// `qcow2_probe()`.
pub(crate) fn probe(buf: &[u8]) -> i32 {
    if buf.len() >= HEADER_SIZE && be32(buf, 0) == QCOW_MAGIC && be32(buf, 4) >= 2 {
        100
    } else {
        0
    }
}

/// `header_ext_add()`: appends an extension, failing with `ENOSPC` past `limit` bytes.
fn header_ext_add(buf: &mut Vec<u8>, magic: u32, data: &[u8], limit: usize) -> io::Result<()> {
    let ext_len = 8 + data.len().next_multiple_of(8);
    if buf.len() + ext_len > limit {
        return Err(errno(libc::ENOSPC));
    }
    buf.extend_from_slice(&magic.to_be_bytes());
    buf.extend_from_slice(&(data.len() as u32).to_be_bytes());
    buf.extend_from_slice(data);
    buf.resize(buf.len() + (data.len().next_multiple_of(8) - data.len()), 0);
    Ok(())
}

/// The feature table extension as QEMU writes it: 48 byte entries of type, bit and name.
pub(crate) fn feature_table_bytes() -> Vec<u8> {
    let mut v = Vec::with_capacity(FEATURES.len() * 48);
    for (ty, bit, name) in FEATURES {
        let mut e = [0u8; 48];
        e[0] = ty;
        e[1] = bit as u8;
        e[2..2 + name.len()].copy_from_slice(name.as_bytes());
        v.extend_from_slice(&e);
    }
    v
}

/// `report_unsupported_feature()`.
pub(crate) fn report_unsupported_feature(table: Option<&[u8]>, mut mask: u64) -> Error {
    let mut features = String::new();
    if let Some(t) = table {
        for e in t.chunks_exact(48) {
            if e[2] == 0 {
                break;
            }
            if e[0] == QCOW2_FEAT_TYPE_INCOMPATIBLE && e[1] < 64 && mask & (1u64 << e[1]) != 0 {
                if !features.is_empty() {
                    features.push_str(", ");
                }
                let name = &e[2..48];
                let end = name.iter().position(|&c| c == 0).unwrap_or(name.len());
                features.push_str(&String::from_utf8_lossy(&name[..end]));
                mask &= !(1u64 << e[1]);
            }
        }
    }
    if mask != 0 {
        if !features.is_empty() {
            features.push_str(", ");
        }
        features.push_str(&format!("Unknown incompatible feature: {mask:x}"));
    }
    Error::generic(format!("Unsupported qcow2 feature(s): {features}"))
}

/// What `qcow2_read_extensions()` found besides what it stores in the state.
#[derive(Default)]
pub(crate) struct ExtResult {
    pub need_update_header: bool,
}

impl State {
    /// `validate_compression_type()`.
    pub(crate) fn validate_compression_type(&self) -> std::result::Result<(), (Error, i32)> {
        match self.compression_type {
            QCOW2_COMPRESSION_TYPE_ZLIB => {}
            t => {
                return Err((
                    Error::generic(format!("qcow2: unknown compression type: {t}")),
                    libc::ENOTSUP,
                ));
            }
        }
        // The zlib type is the default and must not have the feature bit, anything else must.
        if self.compression_type == QCOW2_COMPRESSION_TYPE_ZLIB {
            if self.incompatible_features & QCOW2_INCOMPAT_COMPRESSION != 0 {
                return Err((
                    Error::generic(
                        "qcow2: Compression type incompatible feature bit must not be set",
                    ),
                    libc::EINVAL,
                ));
            }
        } else if self.incompatible_features & QCOW2_INCOMPAT_COMPRESSION == 0 {
            return Err((
                Error::generic("qcow2: Compression type incompatible feature bit must be set"),
                libc::EINVAL,
            ));
        }
        Ok(())
    }

    /// `qcow2_read_extensions()`. The LUKS header is not opened here: the caller does that once
    /// `crypto_header` is known, see `open.rs`.
    pub(crate) fn read_extensions(
        &mut self,
        start_offset: u64,
        end_offset: u64,
        want_feature_table: bool,
    ) -> Result<ExtResult> {
        let mut res = ExtResult::default();
        let mut offset = start_offset;
        while offset < end_offset {
            let mut ext = [0u8; 8];
            self.file.pread(offset, &mut ext).map_err(|e| {
                Error::from_io(
                    format_args!("qcow2_read_extension: ERROR: pread fail from offset {offset}"),
                    e,
                )
            })?;
            let magic = be32(&ext, 0);
            let len = be32(&ext, 4);
            offset += 8;
            if offset > end_offset || len as u64 > end_offset - offset {
                return Err(Error::generic("Header extension too large"));
            }
            let read = |st: &State, what: &str| -> Result<Vec<u8>> {
                let mut v = vec![0u8; len as usize];
                st.file.pread(offset, &mut v).map_err(|e| Error::from_io(what, e))?;
                Ok(v)
            };

            match magic {
                QCOW2_EXT_MAGIC_END => return Ok(res),
                QCOW2_EXT_MAGIC_BACKING_FORMAT => {
                    if len >= BACKING_FORMAT_SIZE {
                        return Err(Error::generic(format!(
                            "ERROR: ext_backing_format: len={len} too large (>={BACKING_FORMAT_SIZE})"
                        )));
                    }
                    let v = read(self, "ERROR: ext_backing_format: Could not read format name")?;
                    let s = c_string(&v);
                    self.backing_format = s.clone();
                    self.image_backing_format = Some(s);
                }
                QCOW2_EXT_MAGIC_FEATURE_TABLE => {
                    if want_feature_table {
                        let mut v = read(self, "ERROR: ext_feature_table: Could not read table")?;
                        // QEMU pads the table with two zeroed entries, so a short last entry
                        // still counts.
                        v.resize(v.len() + 2 * 48, 0);
                        self.feature_table = Some(v);
                    }
                }
                QCOW2_EXT_MAGIC_CRYPTO_HEADER => {
                    if self.crypt_method_header != QCOW_CRYPT_LUKS {
                        return Err(Error::generic(
                            "CRYPTO header extension only expected with LUKS encryption method",
                        ));
                    }
                    if len != 16 {
                        return Err(Error::generic(format!(
                            "CRYPTO header extension size {len}, but expected size 16"
                        )));
                    }
                    let v = read(self, "Unable to read CRYPTO header extension")?;
                    self.crypto_header = (be64(&v, 0), be64(&v, 8));
                    if self.crypto_header.0 % self.cluster_size != 0 {
                        return Err(Error::generic(format!(
                            "Encryption header offset '{}' is not a multiple of cluster size \
                             '{}'",
                            self.crypto_header.0, self.cluster_size
                        )));
                    }
                }
                QCOW2_EXT_MAGIC_BITMAPS => {
                    if len != 24 {
                        return Err(Error::generic("bitmaps_ext: Invalid extension length"));
                    }
                    if self.autoclear_features & QCOW2_AUTOCLEAR_BITMAPS == 0 {
                        if self.qcow_version < 3 {
                            warn_report(
                                "This qcow2 v2 image contains bitmaps, but they may have been \
                                 modified by a program without persistent bitmap support; so \
                                 now they must all be considered inconsistent",
                            );
                        } else {
                            warn_report(
                                "a program lacking bitmap support modified this file, so all \
                                 bitmaps are now considered inconsistent",
                            );
                        }
                        // error_printf(), without a newline, like QEMU.
                        eprint!(
                            "Some clusters may be leaked, run 'qemu-img check -r' on the image \
                             file to fix."
                        );
                        res.need_update_header = true;
                    } else {
                        let v = read(self, "bitmaps_ext: Could not read ext header")?;
                        if be32(&v, 4) != 0 {
                            return Err(Error::generic("bitmaps_ext: Reserved field is not zero"));
                        }
                        let nb = be32(&v, 0);
                        let dir_size = be64(&v, 8);
                        let dir_offset = be64(&v, 16);
                        if nb > QCOW2_MAX_BITMAPS {
                            return Err(Error::generic(format!(
                                "bitmaps_ext: Image has {nb} bitmaps, exceeding the QEMU \
                                 supported maximum of {QCOW2_MAX_BITMAPS}"
                            )));
                        }
                        if nb == 0 {
                            return Err(Error::generic(
                                "found bitmaps extension with zero bitmaps",
                            ));
                        }
                        if self.offset_into_cluster(dir_offset) != 0 {
                            return Err(Error::generic(
                                "bitmaps_ext: invalid bitmap directory offset",
                            ));
                        }
                        if dir_size > QCOW2_MAX_BITMAP_DIRECTORY_SIZE {
                            return Err(Error::generic(format!(
                                "bitmaps_ext: bitmap directory size ({dir_size}) exceeds the \
                                 maximum supported size ({QCOW2_MAX_BITMAP_DIRECTORY_SIZE})"
                            )));
                        }
                        self.nb_bitmaps = nb;
                        self.bitmap_directory_offset = dir_offset;
                        self.bitmap_directory_size = dir_size;
                    }
                }
                QCOW2_EXT_MAGIC_DATA_FILE => {
                    let v = read(self, "ERROR: Could not read data file name")?;
                    self.image_data_file = Some(c_string(&v));
                }
                _ => {
                    // Unknown extensions are kept for when the header is rewritten.
                    let v = read(self, "ERROR: unknown extension: Could not read data")?;
                    self.unknown_header_ext.insert(0, (magic, v));
                }
            }
            offset += (len as u64 + 7) & !7;
        }
        Ok(res)
    }

    /// `qcow2_update_header()`: rewrites the whole first cluster from the state.
    pub(crate) fn update_header(&mut self) -> io::Result<()> {
        let limit = self.cluster_size as usize;
        if limit < HEADER_SIZE {
            return Err(errno(libc::ENOSPC));
        }
        let header_length = HEADER_SIZE + self.unknown_header_fields.len();
        let refcount_table_clusters =
            (self.refcount_table.len() as u64 >> (self.cluster_bits - 3)) as u32;
        if self.validate_compression_type().is_err() {
            return Err(errno(libc::EINVAL));
        }

        let mut header = Header {
            magic: QCOW_MAGIC,
            version: self.qcow_version,
            backing_file_offset: 0,
            backing_file_size: 0,
            cluster_bits: self.cluster_bits,
            size: self.total_size,
            crypt_method: self.crypt_method_header,
            l1_size: self.l1_size,
            l1_table_offset: self.l1_table_offset,
            refcount_table_offset: self.refcount_table_offset,
            refcount_table_clusters,
            nb_snapshots: self.snapshots.len() as u32,
            snapshots_offset: self.snapshots_offset,
            incompatible_features: self.incompatible_features,
            compatible_features: self.compatible_features,
            autoclear_features: self.autoclear_features,
            refcount_order: self.refcount_order,
            header_length: header_length as u32,
            compression_type: self.compression_type,
        };

        let fixed = match self.qcow_version {
            2 => HEADER_V2_SIZE,
            3 => HEADER_SIZE,
            _ => return Err(errno(libc::EINVAL)),
        };
        let mut buf: Vec<u8> = Vec::with_capacity(limit);
        buf.resize(fixed, 0);

        if !self.unknown_header_fields.is_empty() {
            if limit - buf.len() < self.unknown_header_fields.len() {
                return Err(errno(libc::ENOSPC));
            }
            buf.extend_from_slice(&self.unknown_header_fields);
        }
        if let Some(f) = &self.image_backing_format {
            header_ext_add(&mut buf, QCOW2_EXT_MAGIC_BACKING_FORMAT, f.as_bytes(), limit)?;
        }
        if self.has_data_file() {
            if let Some(f) = &self.image_data_file {
                header_ext_add(&mut buf, QCOW2_EXT_MAGIC_DATA_FILE, f.as_bytes(), limit)?;
            }
        }
        if self.crypto_header.0 != 0 {
            let mut d = [0u8; 16];
            d[..8].copy_from_slice(&self.crypto_header.0.to_be_bytes());
            d[8..].copy_from_slice(&self.crypto_header.1.to_be_bytes());
            header_ext_add(&mut buf, QCOW2_EXT_MAGIC_CRYPTO_HEADER, &d, limit)?;
        }
        // A mere 8 feature names take 392 bytes, which would leave almost no room for a
        // backing file name with small clusters, so the table is left out for 4k and below.
        if self.qcow_version >= 3 && self.cluster_size > 4096 {
            header_ext_add(&mut buf, QCOW2_EXT_MAGIC_FEATURE_TABLE, &feature_table_bytes(), limit)?;
        }
        if self.nb_bitmaps > 0 {
            let mut d = [0u8; 24];
            d[..4].copy_from_slice(&self.nb_bitmaps.to_be_bytes());
            d[8..16].copy_from_slice(&self.bitmap_directory_size.to_be_bytes());
            d[16..24].copy_from_slice(&self.bitmap_directory_offset.to_be_bytes());
            header_ext_add(&mut buf, QCOW2_EXT_MAGIC_BITMAPS, &d, limit)?;
        }
        for (magic, data) in &self.unknown_header_ext {
            header_ext_add(&mut buf, *magic, data, limit)?;
        }
        header_ext_add(&mut buf, QCOW2_EXT_MAGIC_END, &[], limit)?;

        if let Some(b) = &self.image_backing_file {
            if limit - buf.len() < b.len() {
                return Err(errno(libc::ENOSPC));
            }
            header.backing_file_offset = buf.len() as u64;
            header.backing_file_size = b.len() as u32;
            buf.extend_from_slice(b.as_bytes());
        }
        buf.resize(limit, 0);
        buf[..fixed].copy_from_slice(&header.to_bytes()[..fixed]);
        self.file.pwrite(0, &buf)
    }

    /// `qcow2_co_change_backing_file()`.
    pub(crate) fn change_backing_file(
        &mut self,
        backing_file: Option<&str>,
        backing_fmt: Option<&str>,
    ) -> io::Result<()> {
        // With a backing file the external data file alone no longer makes sense.
        if backing_file.is_some() && self.data_file_is_raw() {
            return Err(errno(libc::EINVAL));
        }
        if backing_file.is_some_and(|b| b.len() > 1023) {
            return Err(errno(libc::EINVAL));
        }
        self.backing_file = backing_file.unwrap_or("").to_string();
        let fmt: String = backing_fmt.unwrap_or("").chars().take(15).collect();
        self.backing_format = fmt.clone();
        self.image_backing_file = backing_file.map(|s| s.to_string());
        self.image_backing_format = backing_fmt.map(|_| fmt);
        self.update_header()
    }
}

/// The bytes up to the first NUL, as a string.
pub(crate) fn c_string(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).into_owned()
}
