// SPDX-License-Identifier: GPL-2.0-or-later

//! The Parallels Format Extension from block/parallels-ext.c: a cluster that `ext_off` in the
//! header points at, holding an MD5 checksum and a list of features. The only feature QEMU
//! knows is the dirty bitmaps one, whose bitmaps are loaded read-only.
//!
//! The bitmaps are kept by the driver in [`LoadedBitmap`], in the layout `hbitmap` serializes
//! (64-bit little-endian words, one bit per `granularity` bytes of the disk). They are not
//! registered as dirty bitmaps of the node, see the module doc of the driver.

use std::io;

use ruvm_base::{Error, Result};
use ruvm_qapi::types::QCryptoHashAlgo;

use crate::node::BDRV_SECTOR_SIZE;

const PARALLELS_FORMAT_EXTENSION_MAGIC: u64 = 0xAB23_4CEF_23DC_EA87;

const PARALLELS_END_OF_FEATURES_MAGIC: u64 = 0;
const PARALLELS_DIRTY_BITMAP_FEATURE_MAGIC: u64 = 0x2038_5FAE_252C_B34A;

/// `sizeof(ParallelsFormatExtensionHeader)`: the magic and the MD5 of the rest of the cluster.
const EXT_HEADER_SIZE: usize = 24;
/// `sizeof(ParallelsFeatureHeader)`.
const FEATURE_HEADER_SIZE: usize = 24;
/// `sizeof(ParallelsDirtyBitmapFeature)`, without the L1 table that follows it.
const BITMAP_FEATURE_SIZE: usize = 32;

/// A dirty bitmap of the Format Extension, `BdrvDirtyBitmap` as loaded by
/// `parallels_load_bitmap()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LoadedBitmap {
    /// The UUID of the bitmap as text, which QEMU uses as its name.
    pub name: String,
    /// Bytes of the disk per bit.
    pub granularity: u32,
    /// The size of the disk in bytes.
    pub size: u64,
    /// The bits, 64 per word, the first bit being the lowest bit of the first word.
    pub words: Vec<u64>,
}

impl LoadedBitmap {
    fn new(name: String, granularity: u32, size: u64) -> Self {
        let bits = size.div_ceil(u64::from(granularity));
        LoadedBitmap { name, granularity, size, words: vec![0; bits.div_ceil(64) as usize] }
    }

    fn bits(&self) -> u64 {
        self.size.div_ceil(u64::from(self.granularity))
    }

    /// Whether the byte at `offset` is dirty.
    #[allow(dead_code, reason = "for the dirty bitmap code and tests")]
    pub(crate) fn get(&self, offset: u64) -> bool {
        let bit = offset / u64::from(self.granularity);
        bit < self.bits() && self.words[(bit / 64) as usize] & (1 << (bit % 64)) != 0
    }

    /// The number of dirty bytes, `bdrv_get_dirty_count()`.
    #[allow(dead_code, reason = "for the dirty bitmap code and tests")]
    pub(crate) fn count(&self) -> u64 {
        let bits: u64 = self.words.iter().map(|w| u64::from(w.count_ones())).sum();
        (bits * u64::from(self.granularity)).min(self.size)
    }

    /// `bdrv_dirty_bitmap_serialization_size(bitmap, 0, size)`: the words that cover the
    /// whole bitmap, in bytes.
    fn serialization_size(&self) -> u64 {
        if self.size == 0 {
            return 0;
        }
        ((self.size - 1) / u64::from(self.granularity) / 64 + 1) * 8
    }

    /// `bdrv_dirty_bitmap_deserialize_ones()`.
    fn deserialize_ones(&mut self, offset: u64, count: u64) {
        let g = u64::from(self.granularity);
        let first = offset / g;
        let last = (offset + count - 1) / g;
        for bit in first..=last.min(self.bits() - 1) {
            self.words[(bit / 64) as usize] |= 1 << (bit % 64);
        }
    }

    /// `bdrv_dirty_bitmap_deserialize_part()`: the words for `count` bytes at `offset` from
    /// `buf`.
    fn deserialize_part(&mut self, buf: &[u8], offset: u64, count: u64) {
        let g = u64::from(self.granularity);
        let first = offset / g / 64;
        let last = (offset + count - 1) / g / 64;
        for (i, w) in (first..=last).enumerate() {
            let Some(src) = buf.get(i * 8..i * 8 + 8) else { break };
            if let Some(dst) = self.words.get_mut(w as usize) {
                *dst = u64::from_le_bytes(src.try_into().expect("8 bytes"));
            }
        }
    }

    /// `bdrv_dirty_bitmap_deserialize_finish()`: here, dropping the bits past the end.
    fn deserialize_finish(&mut self) {
        let bits = self.bits();
        if bits % 64 != 0 {
            if let Some(last) = self.words.last_mut() {
                *last &= (1u64 << (bits % 64)) - 1;
            }
        }
    }
}

/// What the extension code needs of the driver state.
pub(crate) struct ExtCtx<'a> {
    /// Reads from the image file, `bdrv_pread(bs->file, ...)`.
    pub read: &'a dyn Fn(u64, &mut [u8]) -> io::Result<()>,
    pub cluster_size: u32,
    /// `bs->total_sectors`.
    pub total_sectors: u64,
}

fn le32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().expect("4 bytes"))
}

fn le64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().expect("8 bytes"))
}

/// `qemu_uuid_unparse()`.
fn uuid_unparse(id: &[u8]) -> String {
    let hex =
        |r: std::ops::Range<usize>| id[r].iter().map(|b| format!("{b:02x}")).collect::<String>();
    format!("{}-{}-{}-{}-{}", hex(0..4), hex(4..6), hex(6..8), hex(8..10), hex(10..16))
}

/// `parallels_load_bitmap_data()`: fills `bitmap` from the clusters the L1 table points at.
/// An entry of 0 is a cluster of zeroes and an entry of 1 a cluster of ones.
fn load_bitmap_data(ctx: &ExtCtx<'_>, l1_table: &[u64], bitmap: &mut LoadedBitmap) -> Result<()> {
    let mut buf = vec![0u8; ctx.cluster_size as usize];
    let bm_size = bitmap.size;
    // bdrv_dirty_bitmap_serialization_coverage(): the disk bytes one cluster covers.
    let limit = u64::from(bitmap.granularity) * (u64::from(ctx.cluster_size) << 3);
    let mut offset = 0u64;
    for &entry in l1_table {
        let count = (bm_size - offset).min(limit);
        if entry == 1 {
            bitmap.deserialize_ones(offset, count);
        } else if entry != 0 {
            (ctx.read)(entry << 9, &mut buf)
                .map_err(|e| Error::from_io("Failed to read bitmap data cluster", e))?;
            bitmap.deserialize_part(&buf, offset, count);
        }
        offset += limit;
    }
    bitmap.deserialize_finish();
    Ok(())
}

/// `parallels_load_bitmap()`: `data` is the dirty bitmaps feature, a
/// `ParallelsDirtyBitmapFeature` followed by the L1 table.
fn load_bitmap(ctx: &ExtCtx<'_>, data: &[u8], loaded: &[LoadedBitmap]) -> Result<LoadedBitmap> {
    if data.len() < BITMAP_FEATURE_SIZE {
        return Err(Error::generic(format!(
            "Too small Bitmap Feature area in Parallels Format Extension: {} bytes, expected at \
             least {BITMAP_FEATURE_SIZE} bytes",
            data.len()
        )));
    }
    let size = le64(data, 0);
    let id = &data[8..24];
    let granularity = le32(data, 24).wrapping_shl(9);
    let l1_size = le32(data, 28);
    let data = &data[BITMAP_FEATURE_SIZE..];

    if size != ctx.total_sectors {
        return Err(Error::generic(format!(
            "Bitmap size (in sectors) {} differs from disk size in sectors {}",
            size as i64, ctx.total_sectors as i64
        )));
    }
    if u64::from(l1_size) * 8 > data.len() as u64 {
        return Err(Error::generic(
            "Bitmaps feature corrupted: l1 table exceeds extension data_size",
        ));
    }

    // bdrv_create_dirty_bitmap() asserts this, a corrupted image must not get that far.
    if !granularity.is_power_of_two() || u64::from(granularity) < BDRV_SECTOR_SIZE {
        return Err(Error::generic(format!(
            "Bitmaps feature corrupted: invalid granularity {granularity}"
        )));
    }
    let name = uuid_unparse(id);
    if loaded.iter().any(|b| b.name == name) {
        return Err(Error::generic(format!("Bitmap already exists: {name}")));
    }
    let mut bitmap = LoadedBitmap::new(name, granularity, ctx.total_sectors * BDRV_SECTOR_SIZE);

    let tab_size = bitmap.serialization_size().div_ceil(u64::from(ctx.cluster_size));
    if tab_size != u64::from(l1_size) {
        return Err(Error::generic(format!(
            "Bitmap table size {l1_size} does not correspond to bitmap size and cluster size. \
             Expected {tab_size}"
        )));
    }

    if l1_size != 0 {
        let l1_table: Vec<u64> =
            data[..l1_size as usize * 8].chunks_exact(8).map(|c| le64(c, 0)).collect();
        load_bitmap_data(ctx, &l1_table, &mut bitmap)?;
    }
    Ok(bitmap)
}

/// `parallels_parse_format_extension()`: checks the extension cluster and loads its bitmaps.
pub(crate) fn parse_format_extension(
    ctx: &ExtCtx<'_>,
    ext_cluster: &[u8],
) -> Result<Vec<LoadedBitmap>> {
    // The cluster is at least a sector, larger than the header.
    let magic = le64(ext_cluster, 0);
    let check_sum = &ext_cluster[8..24];
    let mut pos = EXT_HEADER_SIZE;
    let mut remaining = ext_cluster.len() as i64 - EXT_HEADER_SIZE as i64;

    if magic != PARALLELS_FORMAT_EXTENSION_MAGIC {
        return Err(Error::generic(format!(
            "Wrong parallels Format Extension magic: 0x{magic:x}, expected: \
             0x{PARALLELS_FORMAT_EXTENSION_MAGIC:x}"
        )));
    }

    let hash = ruvm_crypto::hash::hash_bytes(QCryptoHashAlgo::Md5, &ext_cluster[pos..])?;
    if hash != check_sum {
        return Err(Error::generic(
            "Wrong checksum in Format Extension header. Format extension is corrupted.",
        ));
    }

    let mut bitmaps = Vec::new();
    loop {
        // QEMU only takes the feature headers off `remaining`, so the data of earlier
        // features can move `pos` past the cluster. Stop there rather than read beyond it.
        if remaining < FEATURE_HEADER_SIZE as i64 || pos + FEATURE_HEADER_SIZE > ext_cluster.len() {
            return Err(Error::generic(format!(
                "Can not read feature header, as remaining bytes ({remaining}) in Format \
                 Extension is less than Feature header size ({FEATURE_HEADER_SIZE})"
            )));
        }
        let fh = &ext_cluster[pos..pos + FEATURE_HEADER_SIZE];
        pos += FEATURE_HEADER_SIZE;
        remaining -= FEATURE_HEADER_SIZE as i64;

        let fmagic = le64(fh, 0);
        let flags = le64(fh, 8);
        let data_size = le32(fh, 16) as usize;

        if flags != 0 {
            return Err(Error::generic("Flags for extension feature are unsupported"));
        }
        if data_size as i64 > remaining || pos + data_size > ext_cluster.len() {
            return Err(Error::generic("Feature data_size exceedes Format Extension cluster"));
        }

        match fmagic {
            PARALLELS_END_OF_FEATURES_MAGIC => return Ok(bitmaps),
            PARALLELS_DIRTY_BITMAP_FEATURE_MAGIC => {
                let b = load_bitmap(ctx, &ext_cluster[pos..pos + data_size], &bitmaps)?;
                bitmaps.push(b);
            }
            _ => return Err(Error::generic(format!("Unknown feature: 0x{fmagic:x}"))),
        }
        pos = (pos + data_size).next_multiple_of(8);
    }
}

/// `parallels_read_format_extension()`: reads the cluster at `ext_off` (bytes) and parses it.
pub(crate) fn read_format_extension(ctx: &ExtCtx<'_>, ext_off: u64) -> Result<Vec<LoadedBitmap>> {
    assert!(ext_off > 0);
    let mut ext_cluster = vec![0u8; ctx.cluster_size as usize];
    (ctx.read)(ext_off, &mut ext_cluster)
        .map_err(|e| Error::from_io("Failed to read Format Extension cluster", e))?;
    parse_format_extension(ctx, &ext_cluster)
}

/// Builds an extension cluster, for tests: the features are `(magic, data)` pairs.
#[cfg(test)]
pub(crate) fn build_extension(cluster_size: usize, features: &[(u64, Vec<u8>)]) -> Vec<u8> {
    let mut c = vec![0u8; cluster_size];
    c[..8].copy_from_slice(&PARALLELS_FORMAT_EXTENSION_MAGIC.to_le_bytes());
    let mut pos = EXT_HEADER_SIZE;
    for (magic, data) in features {
        c[pos..pos + 8].copy_from_slice(&magic.to_le_bytes());
        c[pos + 16..pos + 20].copy_from_slice(&(data.len() as u32).to_le_bytes());
        pos += FEATURE_HEADER_SIZE;
        c[pos..pos + data.len()].copy_from_slice(data);
        pos = (pos + data.len()).next_multiple_of(8);
    }
    // The end of features header is all zeroes.
    let hash =
        ruvm_crypto::hash::hash_bytes(QCryptoHashAlgo::Md5, &c[EXT_HEADER_SIZE..]).expect("md5");
    c[8..24].copy_from_slice(&hash);
    c
}

/// A dirty bitmap feature, for tests.
#[cfg(test)]
pub(crate) fn bitmap_feature(
    size: u64,
    id: [u8; 16],
    granularity_sectors: u32,
    l1: &[u64],
) -> Vec<u8> {
    let mut d = Vec::new();
    d.extend_from_slice(&size.to_le_bytes());
    d.extend_from_slice(&id);
    d.extend_from_slice(&granularity_sectors.to_le_bytes());
    d.extend_from_slice(&(l1.len() as u32).to_le_bytes());
    for e in l1 {
        d.extend_from_slice(&e.to_le_bytes());
    }
    d
}

#[cfg(test)]
pub(crate) const DIRTY_BITMAP_MAGIC: u64 = PARALLELS_DIRTY_BITMAP_FEATURE_MAGIC;
