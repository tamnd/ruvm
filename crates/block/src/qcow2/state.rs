// SPDX-License-Identifier: GPL-2.0-or-later

//! `BDRVQcow2State` and the small helpers from block/qcow2.h, plus the pieces of block/qcow2.c
//! that everything else leans on: corruption reporting, the dirty and corrupt bits, and table
//! validation.

use std::io;
use std::sync::Arc;

use ruvm_base::Error;

use super::bitmap::Bitmap;
use super::cache::Cache;
use super::crypto::Encryption;
use super::io::{Backing, Storage};
use super::snapshot::Snapshot;
use crate::node::errno;

pub(crate) const QCOW_MAGIC: u32 = u32::from_be_bytes([b'Q', b'F', b'I', 0xfb]);

pub(crate) const QCOW_CRYPT_NONE: u32 = 0;
pub(crate) const QCOW_CRYPT_AES: u32 = 1;
pub(crate) const QCOW_CRYPT_LUKS: u32 = 2;

pub(crate) const QCOW_MAX_CRYPT_CLUSTERS: u64 = 32;
pub(crate) const QCOW_MAX_SNAPSHOTS: u64 = 65536;
pub(crate) const QCOW_MAX_CLUSTER_OFFSET: u64 = (1 << 56) - 1;
pub(crate) const QCOW_MAX_REFTABLE_SIZE: u64 = 8 << 20;
pub(crate) const QCOW_MAX_L1_SIZE: u64 = 32 << 20;
pub(crate) const QCOW_MAX_SNAPSHOTS_SIZE: u64 = 1024 * QCOW_MAX_SNAPSHOTS;
pub(crate) const QCOW_MAX_SNAPSHOT_EXTRA_DATA: u32 = 1024;
pub(crate) const QCOW2_MAX_BITMAPS: u32 = 65535;
pub(crate) const QCOW2_MAX_BITMAP_DIRECTORY_SIZE: u64 = 1024 * QCOW2_MAX_BITMAPS as u64;

pub(crate) const QCOW_OFLAG_COPIED: u64 = 1 << 63;
pub(crate) const QCOW_OFLAG_COMPRESSED: u64 = 1 << 62;
pub(crate) const QCOW_OFLAG_ZERO: u64 = 1;

pub(crate) const QCOW_EXTL2_SUBCLUSTERS_PER_CLUSTER: u32 = 32;
pub(crate) const QCOW_L2_BITMAP_ALL_ALLOC: u64 = 0xffff_ffff;
pub(crate) const QCOW_L2_BITMAP_ALL_ZEROES: u64 = 0xffff_ffff << 32;

/// `QCOW_OFLAG_SUB_ALLOC(x)`.
pub(crate) const fn sub_alloc(x: u32) -> u64 {
    1 << x
}

/// `QCOW_OFLAG_SUB_ZERO(x)`.
pub(crate) const fn sub_zero(x: u32) -> u64 {
    1 << (x + 32)
}

/// `QCOW_OFLAG_SUB_ALLOC_RANGE(x, y)`: subclusters `[x, y)`.
pub(crate) const fn sub_alloc_range(x: u32, y: u32) -> u64 {
    (1u64 << y).wrapping_sub(1u64 << x)
}

/// `QCOW_OFLAG_SUB_ZERO_RANGE(x, y)`.
pub(crate) const fn sub_zero_range(x: u32, y: u32) -> u64 {
    sub_alloc_range(x, y) << 32
}

pub(crate) const L2E_SIZE_NORMAL: u64 = 8;
pub(crate) const L2E_SIZE_EXTENDED: u64 = 16;
pub(crate) const L1E_SIZE: u64 = 8;
pub(crate) const REFTABLE_ENTRY_SIZE: u64 = 8;

pub(crate) const MIN_CLUSTER_BITS: u32 = 9;
pub(crate) const MAX_CLUSTER_BITS: u32 = 21;
pub(crate) const QCOW2_COMPRESSED_SECTOR_SIZE: u64 = 512;
pub(crate) const MIN_L2_CACHE_SIZE: u64 = 2;
pub(crate) const MIN_REFCOUNT_CACHE_SIZE: u64 = 4;

#[cfg(target_os = "linux")]
pub(crate) const DEFAULT_L2_CACHE_MAX_SIZE: u64 = 32 << 20;
#[cfg(target_os = "linux")]
pub(crate) const DEFAULT_CACHE_CLEAN_INTERVAL: u64 = 600;
#[cfg(not(target_os = "linux"))]
pub(crate) const DEFAULT_L2_CACHE_MAX_SIZE: u64 = 8 << 20;
#[cfg(not(target_os = "linux"))]
pub(crate) const DEFAULT_CACHE_CLEAN_INTERVAL: u64 = 0;

pub(crate) const DEFAULT_CLUSTER_SIZE: u64 = 65536;

pub(crate) const QCOW2_INCOMPAT_DIRTY_BITNR: u32 = 0;
pub(crate) const QCOW2_INCOMPAT_CORRUPT_BITNR: u32 = 1;
pub(crate) const QCOW2_INCOMPAT_DATA_FILE_BITNR: u32 = 2;
pub(crate) const QCOW2_INCOMPAT_COMPRESSION_BITNR: u32 = 3;
pub(crate) const QCOW2_INCOMPAT_EXTL2_BITNR: u32 = 4;
pub(crate) const QCOW2_INCOMPAT_DIRTY: u64 = 1 << QCOW2_INCOMPAT_DIRTY_BITNR;
pub(crate) const QCOW2_INCOMPAT_CORRUPT: u64 = 1 << QCOW2_INCOMPAT_CORRUPT_BITNR;
pub(crate) const QCOW2_INCOMPAT_DATA_FILE: u64 = 1 << QCOW2_INCOMPAT_DATA_FILE_BITNR;
pub(crate) const QCOW2_INCOMPAT_COMPRESSION: u64 = 1 << QCOW2_INCOMPAT_COMPRESSION_BITNR;
pub(crate) const QCOW2_INCOMPAT_EXTL2: u64 = 1 << QCOW2_INCOMPAT_EXTL2_BITNR;
pub(crate) const QCOW2_INCOMPAT_MASK: u64 = QCOW2_INCOMPAT_DIRTY
    | QCOW2_INCOMPAT_CORRUPT
    | QCOW2_INCOMPAT_DATA_FILE
    | QCOW2_INCOMPAT_COMPRESSION
    | QCOW2_INCOMPAT_EXTL2;

pub(crate) const QCOW2_COMPAT_LAZY_REFCOUNTS_BITNR: u32 = 0;
pub(crate) const QCOW2_COMPAT_LAZY_REFCOUNTS: u64 = 1 << QCOW2_COMPAT_LAZY_REFCOUNTS_BITNR;

pub(crate) const QCOW2_AUTOCLEAR_BITMAPS_BITNR: u32 = 0;
pub(crate) const QCOW2_AUTOCLEAR_DATA_FILE_RAW_BITNR: u32 = 1;
pub(crate) const QCOW2_AUTOCLEAR_BITMAPS: u64 = 1 << QCOW2_AUTOCLEAR_BITMAPS_BITNR;
pub(crate) const QCOW2_AUTOCLEAR_DATA_FILE_RAW: u64 = 1 << QCOW2_AUTOCLEAR_DATA_FILE_RAW_BITNR;
pub(crate) const QCOW2_AUTOCLEAR_MASK: u64 =
    QCOW2_AUTOCLEAR_BITMAPS | QCOW2_AUTOCLEAR_DATA_FILE_RAW;

pub(crate) const QCOW2_COMPRESSION_TYPE_ZLIB: u8 = 0;

pub(crate) const OL_MAX_BITNR: u32 = 9;
pub(crate) const OL_MAIN_HEADER: u32 = 1 << 0;
pub(crate) const OL_ACTIVE_L1: u32 = 1 << 1;
pub(crate) const OL_ACTIVE_L2: u32 = 1 << 2;
pub(crate) const OL_REFCOUNT_TABLE: u32 = 1 << 3;
pub(crate) const OL_REFCOUNT_BLOCK: u32 = 1 << 4;
pub(crate) const OL_SNAPSHOT_TABLE: u32 = 1 << 5;
pub(crate) const OL_INACTIVE_L1: u32 = 1 << 6;
pub(crate) const OL_INACTIVE_L2: u32 = 1 << 7;
pub(crate) const OL_BITMAP_DIRECTORY: u32 = 1 << 8;
pub(crate) const OL_CONSTANT: u32 =
    OL_MAIN_HEADER | OL_ACTIVE_L1 | OL_REFCOUNT_TABLE | OL_SNAPSHOT_TABLE | OL_BITMAP_DIRECTORY;
pub(crate) const OL_CACHED: u32 = OL_CONSTANT | OL_ACTIVE_L2 | OL_REFCOUNT_BLOCK | OL_INACTIVE_L1;
pub(crate) const OL_ALL: u32 = OL_CACHED | OL_INACTIVE_L2;

/// `metadata_ol_names`.
pub(crate) const METADATA_OL_NAMES: [&str; OL_MAX_BITNR as usize] = [
    "qcow2_header",
    "active L1 table",
    "active L2 table",
    "refcount table",
    "refcount block",
    "snapshot table",
    "inactive L1 table",
    "inactive L2 table",
    "bitmap directory",
];

pub(crate) const L1E_OFFSET_MASK: u64 = 0x00ff_ffff_ffff_fe00;
pub(crate) const L1E_RESERVED_MASK: u64 = 0x7f00_0000_0000_01ff;
pub(crate) const L2E_OFFSET_MASK: u64 = 0x00ff_ffff_ffff_fe00;
pub(crate) const L2E_STD_RESERVED_MASK: u64 = 0x3f00_0000_0000_01fe;
pub(crate) const REFT_OFFSET_MASK: u64 = 0xffff_ffff_ffff_fe00;
pub(crate) const REFT_RESERVED_MASK: u64 = 0x1ff;

/// `enum qcow2_discard_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DiscardType {
    Never = 0,
    Always = 1,
    Request = 2,
    Snapshot = 3,
    Other = 4,
}

/// `QCow2ClusterType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClusterType {
    Unallocated,
    ZeroPlain,
    ZeroAlloc,
    Normal,
    Compressed,
}

impl ClusterType {
    /// `qcow2_cluster_is_allocated()`.
    pub(crate) fn is_allocated(self) -> bool {
        matches!(self, ClusterType::Compressed | ClusterType::Normal | ClusterType::ZeroAlloc)
    }
}

/// `QCow2SubclusterType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SubclusterType {
    UnallocatedPlain,
    UnallocatedAlloc,
    ZeroPlain,
    ZeroAlloc,
    Normal,
    Compressed,
    Invalid,
}

/// Where a request came from, which decides what `BDRV_O_*` style behaviour applies while
/// opening: a plain open, `qemu-img check` (`BDRV_O_CHECK`) or an open that does no I/O on the
/// guest data (`BDRV_O_NO_IO`, used by `qemu-img info`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct OpenFlags {
    /// `BDRV_O_RDWR`.
    pub read_write: bool,
    /// `BDRV_O_CHECK`.
    pub check: bool,
    /// `BDRV_O_NO_IO`.
    pub no_io: bool,
    /// `BDRV_O_INACTIVE`.
    pub inactive: bool,
    /// `BDRV_O_UNMAP`: requests may be passed down as discards.
    pub unmap: bool,
}

/// The state of an open image, `BDRVQcow2State` plus the parts of `BlockDriverState` the
/// driver uses.
pub(crate) struct State {
    pub file: Arc<dyn Storage>,
    /// `s->data_file` when it is not `bs->file`.
    pub data_file: Option<Arc<dyn Storage>>,
    pub backing: Option<Arc<dyn Backing>>,
    pub flags: OpenFlags,
    /// `bs->drv == NULL` after fatal corruption: the image refuses all I/O.
    pub drv_gone: bool,
    /// `bs->total_sectors * BDRV_SECTOR_SIZE`.
    pub total_size: u64,
    /// `bs->backing_file`, what the image header names.
    pub backing_file: String,
    /// `bs->backing_format`.
    pub backing_format: String,

    pub cluster_bits: u32,
    pub cluster_size: u64,
    pub l2_slice_size: u64,
    pub subcluster_bits: u32,
    pub subcluster_size: u64,
    pub subclusters_per_cluster: u32,
    pub l2_bits: u32,
    pub l2_size: u64,
    pub l1_size: u32,
    pub l1_vm_state_index: u64,
    pub refcount_block_bits: u32,
    pub refcount_block_size: u64,
    pub csize_shift: u32,
    pub csize_mask: u64,
    pub cluster_offset_mask: u64,
    pub l1_table_offset: u64,
    pub l1_table: Vec<u64>,

    pub l2_table_cache: Cache,
    pub refcount_block_cache: Cache,
    pub cache_clean_interval: u64,
    /// When the caches were last cleaned. There is no timer thread: cleaning happens when a
    /// request finds the interval has passed.
    pub last_cache_clean: std::time::Instant,

    pub refcount_table: Vec<u64>,
    pub refcount_table_offset: u64,
    pub max_refcount_table_index: u32,
    pub free_cluster_index: u64,
    pub free_byte_offset: u64,

    /// `Qcow2CryptoHeaderExtension`: offset and length of the LUKS header.
    pub crypto_header: (u64, u64),
    pub crypto: Option<Box<dyn Encryption>>,
    pub crypt_physical_offset: bool,
    pub crypt_method_header: u32,
    pub snapshots_offset: u64,
    pub snapshots_size: u64,
    pub snapshots: Vec<Snapshot>,

    pub nb_bitmaps: u32,
    pub bitmap_directory_size: u64,
    pub bitmap_directory_offset: u64,
    /// The dirty bitmaps in memory, what QEMU keeps as `BdrvDirtyBitmap` on the node.
    pub bitmaps: Vec<Bitmap>,

    pub qcow_version: u32,
    pub use_lazy_refcounts: bool,
    pub refcount_order: u32,
    pub refcount_bits: u32,
    pub refcount_max: u64,

    pub discard_passthrough: [bool; 5],
    pub discard_no_unref: bool,
    pub overlap_check: u32,
    pub signaled_corruption: bool,

    pub incompatible_features: u64,
    pub compatible_features: u64,
    pub autoclear_features: u64,

    pub unknown_header_fields: Vec<u8>,
    pub unknown_header_ext: Vec<(u32, Vec<u8>)>,
    pub discards: Vec<(u64, u64)>,
    pub cache_discards: bool,

    pub image_backing_file: Option<String>,
    pub image_backing_format: Option<String>,
    pub image_data_file: Option<String>,

    pub metadata_preallocation_checked: bool,
    pub metadata_preallocation: bool,
    pub compression_type: u8,
    /// The raw feature table extension, kept for error messages.
    pub feature_table: Option<Vec<u8>>,
    /// The runtime options the image was opened with, `bs->options`, applied again after a
    /// resize to size the caches.
    pub options: super::open::OpenOptions,
}

impl State {
    pub(crate) fn has_subclusters(&self) -> bool {
        self.incompatible_features & QCOW2_INCOMPAT_EXTL2 != 0
    }

    pub(crate) fn l2_entry_size(&self) -> u64 {
        if self.has_subclusters() { L2E_SIZE_EXTENDED } else { L2E_SIZE_NORMAL }
    }

    /// `get_l2_entry()` on a slice.
    pub(crate) fn l2_entry(&self, slice: &[u8], idx: usize) -> u64 {
        let i = idx * (self.l2_entry_size() / 8) as usize;
        super::cache::get_be64(slice, i)
    }

    /// `get_l2_bitmap()`: always 0 without subclusters.
    pub(crate) fn l2_bitmap(&self, slice: &[u8], idx: usize) -> u64 {
        if self.has_subclusters() { super::cache::get_be64(slice, idx * 2 + 1) } else { 0 }
    }

    /// `set_l2_entry()`.
    pub(crate) fn set_l2_entry(&self, slice: &mut [u8], idx: usize, entry: u64) {
        let i = idx * (self.l2_entry_size() / 8) as usize;
        super::cache::set_be64(slice, i, entry);
    }

    /// `set_l2_bitmap()`.
    pub(crate) fn set_l2_bitmap(&self, slice: &mut [u8], idx: usize, bitmap: u64) {
        assert!(self.has_subclusters());
        super::cache::set_be64(slice, idx * 2 + 1, bitmap);
    }

    /// `has_data_file()`. An image opened without I/O has no data file open but still has
    /// one as far as the metadata is concerned.
    pub(crate) fn has_data_file(&self) -> bool {
        self.data_file.is_some()
            || (self.flags.no_io && self.incompatible_features & QCOW2_INCOMPAT_DATA_FILE != 0)
    }

    /// `s->data_file`: the external data file, or the image file itself.
    pub(crate) fn data(&self) -> Arc<dyn Storage> {
        self.data_file.clone().unwrap_or_else(|| self.file.clone())
    }

    /// `data_file_is_raw()`.
    pub(crate) fn data_file_is_raw(&self) -> bool {
        self.autoclear_features & QCOW2_AUTOCLEAR_DATA_FILE_RAW != 0
    }

    pub(crate) fn start_of_cluster(&self, offset: u64) -> u64 {
        offset & !(self.cluster_size - 1)
    }

    pub(crate) fn offset_into_cluster(&self, offset: u64) -> u64 {
        offset & (self.cluster_size - 1)
    }

    pub(crate) fn offset_into_subcluster(&self, offset: u64) -> u64 {
        offset & (self.subcluster_size - 1)
    }

    pub(crate) fn size_to_clusters(&self, size: u64) -> u64 {
        size.div_ceil(self.cluster_size)
    }

    pub(crate) fn size_to_subclusters(&self, size: u64) -> u64 {
        size.div_ceil(self.subcluster_size)
    }

    pub(crate) fn size_to_l1(&self, size: u64) -> u64 {
        let shift = self.cluster_bits + self.l2_bits;
        size.div_ceil(1 << shift)
    }

    pub(crate) fn offset_to_l1_index(&self, offset: u64) -> usize {
        (offset >> (self.l2_bits + self.cluster_bits)) as usize
    }

    pub(crate) fn offset_to_l2_index(&self, offset: u64) -> usize {
        ((offset >> self.cluster_bits) & (self.l2_size - 1)) as usize
    }

    pub(crate) fn offset_to_l2_slice_index(&self, offset: u64) -> usize {
        ((offset >> self.cluster_bits) & (self.l2_slice_size - 1)) as usize
    }

    pub(crate) fn offset_to_sc_index(&self, offset: u64) -> u32 {
        ((offset >> self.subcluster_bits) & (self.subclusters_per_cluster as u64 - 1)) as u32
    }

    /// `qcow2_vm_state_offset()`.
    pub(crate) fn vm_state_offset(&self) -> u64 {
        self.l1_vm_state_index << (self.cluster_bits + self.l2_bits)
    }

    pub(crate) fn offset_to_reftable_index(&self, offset: u64) -> u64 {
        offset >> (self.refcount_block_bits + self.cluster_bits)
    }

    /// `qcow2_get_cluster_type()`.
    pub(crate) fn cluster_type(&self, l2_entry: u64) -> ClusterType {
        if l2_entry & QCOW_OFLAG_COMPRESSED != 0 {
            ClusterType::Compressed
        } else if l2_entry & QCOW_OFLAG_ZERO != 0 && !self.has_subclusters() {
            if l2_entry & L2E_OFFSET_MASK != 0 {
                ClusterType::ZeroAlloc
            } else {
                ClusterType::ZeroPlain
            }
        } else if l2_entry & L2E_OFFSET_MASK == 0 {
            // Offset 0 is a valid offset in an external data file, and clusters there always
            // have refcount 1, so the COPIED flag tells them apart.
            if self.has_data_file() && l2_entry & QCOW_OFLAG_COPIED != 0 {
                ClusterType::Normal
            } else {
                ClusterType::Unallocated
            }
        } else {
            ClusterType::Normal
        }
    }

    /// `qcow2_get_subcluster_type()`.
    pub(crate) fn subcluster_type(&self, l2_entry: u64, l2_bitmap: u64, sc: u32) -> SubclusterType {
        let ty = self.cluster_type(l2_entry);
        assert!(sc < self.subclusters_per_cluster);
        if self.has_subclusters() {
            match ty {
                ClusterType::Compressed => SubclusterType::Compressed,
                ClusterType::Normal => {
                    if (l2_bitmap >> 32) & l2_bitmap != 0 {
                        SubclusterType::Invalid
                    } else if l2_bitmap & sub_zero(sc) != 0 {
                        SubclusterType::ZeroAlloc
                    } else if l2_bitmap & sub_alloc(sc) != 0 {
                        SubclusterType::Normal
                    } else {
                        SubclusterType::UnallocatedAlloc
                    }
                }
                ClusterType::Unallocated => {
                    if l2_bitmap & QCOW_L2_BITMAP_ALL_ALLOC != 0 {
                        SubclusterType::Invalid
                    } else if l2_bitmap & sub_zero(sc) != 0 {
                        SubclusterType::ZeroPlain
                    } else {
                        SubclusterType::UnallocatedPlain
                    }
                }
                ClusterType::ZeroPlain | ClusterType::ZeroAlloc => unreachable!(),
            }
        } else {
            match ty {
                ClusterType::Compressed => SubclusterType::Compressed,
                ClusterType::ZeroPlain => SubclusterType::ZeroPlain,
                ClusterType::ZeroAlloc => SubclusterType::ZeroAlloc,
                ClusterType::Normal => SubclusterType::Normal,
                ClusterType::Unallocated => SubclusterType::UnallocatedPlain,
            }
        }
    }

    /// `qcow2_need_accurate_refcounts()`.
    pub(crate) fn need_accurate_refcounts(&self) -> bool {
        self.incompatible_features & QCOW2_INCOMPAT_DIRTY == 0
    }

    /// `bdrv_is_writable()`.
    pub(crate) fn writable(&self) -> bool {
        self.flags.read_write && !self.flags.inactive
    }

    /// Fails requests once fatal corruption made the node unusable.
    pub(crate) fn check_usable(&self) -> io::Result<()> {
        if self.drv_gone {
            return Err(errno(crate::node::ENOMEDIUM));
        }
        Ok(())
    }

    /// `qcow2_signal_corruption()`. The `BLOCK_IMAGE_CORRUPTED` event has no monitor to go to
    /// yet, so only the message on stderr remains.
    pub(crate) fn signal_corruption(&mut self, fatal: bool, offset: i64, size: i64, msg: &str) {
        let _ = (offset, size);
        let fatal = fatal && self.writable();
        if self.signaled_corruption
            && (!fatal || self.incompatible_features & QCOW2_INCOMPAT_CORRUPT != 0)
        {
            return;
        }
        if fatal {
            eprintln!(
                "qcow2: Marking image as corrupt: {msg}; further corruption events will be \
                 suppressed"
            );
        } else {
            eprintln!(
                "qcow2: Image is corrupt: {msg}; further non-fatal corruption events will be \
                 suppressed"
            );
        }
        if fatal {
            let _ = self.mark_corrupt();
            self.drv_gone = true;
        }
        self.signaled_corruption = true;
    }

    /// `qcow2_mark_dirty()`: sets the dirty bit on disk before anything relies on it.
    pub(crate) fn mark_dirty(&mut self) -> io::Result<()> {
        assert!(self.qcow_version >= 3);
        if self.incompatible_features & QCOW2_INCOMPAT_DIRTY != 0 {
            return Ok(());
        }
        let val = (self.incompatible_features | QCOW2_INCOMPAT_DIRTY).to_be_bytes();
        self.file.pwrite(72, &val)?;
        self.file.flush()?;
        self.incompatible_features |= QCOW2_INCOMPAT_DIRTY;
        Ok(())
    }

    /// `qcow2_mark_clean()`.
    pub(crate) fn mark_clean(&mut self) -> io::Result<()> {
        if self.incompatible_features & QCOW2_INCOMPAT_DIRTY != 0 {
            self.incompatible_features &= !QCOW2_INCOMPAT_DIRTY;
            self.flush_caches()?;
            return self.update_header();
        }
        Ok(())
    }

    /// `qcow2_mark_corrupt()`.
    pub(crate) fn mark_corrupt(&mut self) -> io::Result<()> {
        self.incompatible_features |= QCOW2_INCOMPAT_CORRUPT;
        self.update_header()
    }

    /// `qcow2_mark_consistent()`.
    pub(crate) fn mark_consistent(&mut self) -> io::Result<()> {
        if self.incompatible_features & QCOW2_INCOMPAT_CORRUPT != 0 {
            self.flush_caches()?;
            self.incompatible_features &= !QCOW2_INCOMPAT_CORRUPT;
            return self.update_header();
        }
        Ok(())
    }

    /// `qcow2_validate_table()`.
    pub(crate) fn validate_table(
        &self,
        offset: u64,
        entries: u64,
        entry_len: u64,
        max_size_bytes: u64,
        table_name: &str,
    ) -> Result<(), (Error, i32)> {
        if entries > max_size_bytes / entry_len {
            return Err((Error::generic(format!("{table_name} too large")), libc::EFBIG));
        }
        // Use i64::MAX as the limit even for unsigned fields, the values end up in signed
        // arithmetic.
        let bytes = entries * entry_len;
        if (i64::MAX as u64 - bytes) < offset || self.offset_into_cluster(offset) != 0 {
            return Err((Error::generic(format!("{table_name} offset invalid")), libc::EINVAL));
        }
        Ok(())
    }

    /// `qcow2_write_caches()`.
    pub(crate) fn write_caches(&mut self) -> io::Result<()> {
        self.cache_write(super::cache::CacheId::L2)?;
        if self.need_accurate_refcounts() {
            self.cache_write(super::cache::CacheId::Refcount)?;
        }
        Ok(())
    }

    /// `qcow2_flush_caches()`.
    pub(crate) fn flush_caches(&mut self) -> io::Result<()> {
        self.write_caches()?;
        self.file.flush()
    }

    /// `bdrv_flush(bs)` from inside the driver: write the caches back, then flush the files.
    pub(crate) fn flush_all(&mut self) -> io::Result<()> {
        self.write_caches()?;
        if let Some(d) = &self.data_file {
            d.flush()?;
        }
        self.file.flush()
    }

    /// Runs the cache cleaner if its interval has passed, in place of QEMU's timer.
    pub(crate) fn maybe_clean_caches(&mut self) {
        if self.cache_clean_interval == 0 {
            return;
        }
        let now = std::time::Instant::now();
        if now.duration_since(self.last_cache_clean).as_secs() >= self.cache_clean_interval {
            self.l2_table_cache.clean_unused();
            self.refcount_block_cache.clean_unused();
            self.last_cache_clean = now;
        }
    }
}

/// `Error` from an errno value with a prefix, the shape of `error_setg_errno()`.
pub(crate) fn err_errno(msg: impl std::fmt::Display, e: io::Error) -> Error {
    Error::from_io(msg, e)
}

/// Formats like C's `%#x`: `0x` in front of non-zero values only, so 0 prints as `0`. Used as
/// a named argument, `format!("{i:#x}", i = Hx(i))`, in messages copied from QEMU.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Hx<T>(pub T);

impl<T: std::fmt::LowerHex + PartialEq + Default> std::fmt::LowerHex for Hx<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 == T::default() { f.write_str("0") } else { write!(f, "{:#x}", self.0) }
    }
}
