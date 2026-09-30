// SPDX-License-Identifier: GPL-2.0-or-later

//! Opening an image: `qcow2_do_open()`, the runtime options (`qcow2_update_options()` and
//! `read_cache_sizes()`) and the encryption header callbacks, from block/qcow2.c.

use std::sync::Arc;

use ruvm_base::{Error, Result};
use ruvm_qapi::cutils;
use ruvm_qapi::visit::{parse_option_size, qapi_bool_parse};

use super::bitmap::Bitmap;
use super::cache::Cache;
use super::check::{CheckResult, FIX_ERRORS, FIX_LEAKS};
use super::crypto::{EncryptionFormat, EncryptionOptions, EncryptionProvider, HeaderIo};
use super::header::{HEADER_SIZE, HEADER_V3_MIN_SIZE, Header, report_unsupported_feature};
use super::io::Storage;
use super::snapshot::SNAPSHOT_HEADER_SIZE;
use super::state::*;

/// The names of the per-structure overlap check options, indexed by `OL_*_BITNR`.
pub(crate) const OVERLAP_OPTION_NAMES: [&str; OL_MAX_BITNR as usize] = [
    "overlap-check.main-header",
    "overlap-check.active-l1",
    "overlap-check.active-l2",
    "overlap-check.refcount-table",
    "overlap-check.refcount-block",
    "overlap-check.snapshot-table",
    "overlap-check.inactive-l1",
    "overlap-check.inactive-l2",
    "overlap-check.bitmap-directory",
];

/// The runtime options of an image, `qcow2_runtime_opts`, together with the open flags and the
/// other images the qcow2 image is used with. `None` means the option was not given and
/// QEMU's default applies.
#[derive(Clone, Debug, Default)]
pub(crate) struct OpenOptions {
    pub flags: OpenFlags,
    /// `lazy-refcounts`. Defaults to what the image header says.
    pub lazy_refcounts: Option<bool>,
    /// `pass-discard-request`. Defaults to on with `discard=unmap` ([`OpenFlags::unmap`]).
    pub pass_discard_request: Option<bool>,
    /// `pass-discard-snapshot`, default on.
    pub pass_discard_snapshot: Option<bool>,
    /// `pass-discard-other`, default off.
    pub pass_discard_other: Option<bool>,
    /// `discard-no-unref`, default off.
    pub discard_no_unref: Option<bool>,
    /// `overlap-check`: `none`, `constant`, `cached` (the default) or `all`.
    pub overlap_check: Option<String>,
    /// `overlap-check.template`, the same thing in the flags form of the option.
    pub overlap_check_template: Option<String>,
    /// `overlap-check.main-header` and the other per-structure switches, indexed like
    /// `OL_*_BITNR`: main header, active L1, active L2, refcount table, refcount block,
    /// snapshot table, inactive L1, inactive L2, bitmap directory.
    pub overlap_flags: [Option<bool>; OL_MAX_BITNR as usize],
    /// `cache-size`: the L2 and refcount caches together, in bytes.
    pub cache_size: Option<u64>,
    /// `l2-cache-size`, in bytes.
    pub l2_cache_size: Option<u64>,
    /// `l2-cache-entry-size`, in bytes.
    pub l2_cache_entry_size: Option<u64>,
    /// `refcount-cache-size`, in bytes.
    pub refcount_cache_size: Option<u64>,
    /// `cache-clean-interval`, in seconds.
    pub cache_clean_interval: Option<u64>,
    /// `encrypt.format`.
    pub encrypt_format: Option<String>,
    /// The other `encrypt.*` options, without the prefix, `key-secret` for example.
    pub encrypt_opts: EncryptionOptions,
    /// What opens the encryption of encrypted images.
    pub encryption: Option<Arc<dyn EncryptionProvider>>,
    /// `data-file`: the external data file, when the caller opened it.
    pub data_file: Option<Arc<dyn Storage>>,
}

impl OpenOptions {
    /// Sets an option from its `-drive` or `-o` text form. Returns `Ok(false)` for a key this
    /// driver does not know, which the caller reports.
    pub(crate) fn set(&mut self, key: &str, value: &str) -> Result<bool> {
        let b = || qapi_bool_parse(key, value).map(Some);
        let size = || parse_option_size(key, value).map(Some);
        match key {
            "lazy-refcounts" => self.lazy_refcounts = b()?,
            "pass-discard-request" => self.pass_discard_request = b()?,
            "pass-discard-snapshot" => self.pass_discard_snapshot = b()?,
            "pass-discard-other" => self.pass_discard_other = b()?,
            "discard-no-unref" => self.discard_no_unref = b()?,
            "overlap-check" => self.overlap_check = Some(value.to_string()),
            "overlap-check.template" => self.overlap_check_template = Some(value.to_string()),
            "cache-size" => self.cache_size = size()?,
            "l2-cache-size" => self.l2_cache_size = size()?,
            "l2-cache-entry-size" => self.l2_cache_entry_size = size()?,
            "refcount-cache-size" => self.refcount_cache_size = size()?,
            "cache-clean-interval" => {
                self.cache_clean_interval = match cutils::strtou64(value, 0, true) {
                    Ok((v, _)) => Some(v),
                    Err((cutils::Errno::Range, _)) => {
                        return Err(Error::generic(format!(
                            "Value '{value}' is too large for parameter '{key}'"
                        )));
                    }
                    Err(_) => {
                        return Err(Error::generic(format!("Parameter '{key}' expects a number")));
                    }
                }
            }
            "encrypt.format" => self.encrypt_format = Some(value.to_string()),
            _ => {
                if let Some(i) = OVERLAP_OPTION_NAMES.iter().position(|n| *n == key) {
                    self.overlap_flags[i] = b()?;
                } else if let Some(k) = key.strip_prefix("encrypt.") {
                    self.encrypt_opts.push((k.to_string(), value.to_string()));
                } else {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
}

/// `Qcow2ReopenState`: the options checked and the caches made, not yet in effect.
pub(crate) struct Prepared {
    l2_table_cache: Cache,
    refcount_block_cache: Cache,
    l2_slice_size: u64,
    use_lazy_refcounts: bool,
    overlap_check: u32,
    discard_passthrough: [bool; 5],
    discard_no_unref: bool,
    cache_clean_interval: u64,
    /// The encryption format the header asks for, for opening it.
    pub(crate) crypto_format: Option<EncryptionFormat>,
}

/// The LUKS header area of an open image, what `qcow2_crypto_hdr_read_func()` reads.
pub(crate) struct ImageHeaderIo<'a> {
    pub file: &'a dyn Storage,
    pub offset: u64,
    pub length: u64,
}

impl HeaderIo for ImageHeaderIo<'_> {
    fn read(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        if offset + buf.len() as u64 > self.length {
            return Err(Error::generic("Request for data outside of extension header"));
        }
        self.file
            .pread(self.offset + offset, buf)
            .map_err(|e| Error::from_io("Could not read encryption header", e))
    }

    fn init(&mut self, _len: u64) -> Result<()> {
        Err(Error::generic("Cannot allocate an encryption header while opening"))
    }

    fn write(&mut self, offset: u64, buf: &[u8]) -> Result<()> {
        if offset + buf.len() as u64 > self.length {
            return Err(Error::generic("Request for data outside of extension header"));
        }
        // QEMU says "read" here too.
        self.file
            .pwrite(self.offset + offset, buf)
            .map_err(|e| Error::from_io("Could not read encryption header", e))
    }
}

/// The error for an encrypted image when nobody passed an [`EncryptionProvider`].
pub(crate) fn no_provider(format: EncryptionFormat) -> Error {
    Error::generic(format!(
        "qcow2: encrypt.format={} needs a crypto provider, none was given",
        format.as_str()
    ))
}

impl State {
    /// A state with nothing read yet, what `bs->opaque` is before `qcow2_do_open()`.
    pub(crate) fn blank(file: Arc<dyn Storage>, flags: OpenFlags) -> State {
        State {
            file,
            data_file: None,
            backing: None,
            flags,
            drv_gone: false,
            total_size: 0,
            backing_file: String::new(),
            backing_format: String::new(),
            cluster_bits: 16,
            cluster_size: 65536,
            l2_slice_size: 8192,
            subcluster_bits: 16,
            subcluster_size: 65536,
            subclusters_per_cluster: 1,
            l2_bits: 13,
            l2_size: 8192,
            l1_size: 0,
            l1_vm_state_index: 0,
            refcount_block_bits: 15,
            refcount_block_size: 32768,
            csize_shift: 0,
            csize_mask: 0,
            cluster_offset_mask: 0,
            l1_table_offset: 0,
            l1_table: Vec::new(),
            l2_table_cache: Cache::new(1, 512).expect("tiny cache"),
            refcount_block_cache: Cache::new(1, 512).expect("tiny cache"),
            cache_clean_interval: 0,
            last_cache_clean: std::time::Instant::now(),
            refcount_table: Vec::new(),
            refcount_table_offset: 0,
            max_refcount_table_index: 0,
            free_cluster_index: 0,
            free_byte_offset: 0,
            crypto_header: (0, 0),
            crypto: None,
            crypt_physical_offset: false,
            crypt_method_header: 0,
            snapshots_offset: 0,
            snapshots_size: 0,
            snapshots: Vec::new(),
            nb_bitmaps: 0,
            bitmap_directory_size: 0,
            bitmap_directory_offset: 0,
            bitmaps: Vec::<Bitmap>::new(),
            qcow_version: 3,
            use_lazy_refcounts: false,
            refcount_order: 4,
            refcount_bits: 16,
            refcount_max: 0xffff,
            discard_passthrough: [false, true, false, true, false],
            discard_no_unref: false,
            overlap_check: OL_CACHED,
            signaled_corruption: false,
            incompatible_features: 0,
            compatible_features: 0,
            autoclear_features: 0,
            unknown_header_fields: Vec::new(),
            unknown_header_ext: Vec::new(),
            discards: Vec::new(),
            cache_discards: false,
            image_backing_file: None,
            image_backing_format: None,
            image_data_file: None,
            metadata_preallocation_checked: false,
            metadata_preallocation: false,
            compression_type: QCOW2_COMPRESSION_TYPE_ZLIB,
            feature_table: None,
            options: OpenOptions::default(),
        }
    }

    /// `read_cache_sizes()`: the L2 cache size, the L2 cache entry size and the refcount
    /// cache size, in bytes.
    fn read_cache_sizes(&self, o: &OpenOptions) -> Result<(u64, u64, u64)> {
        let min_refcount_cache = MIN_REFCOUNT_CACHE_SIZE * self.cluster_size;
        let virtual_disk_size = self.total_size;
        let max_l2_entries = virtual_disk_size.div_ceil(self.cluster_size);
        // An L2 table is always one cluster, so the largest useful cache is a multiple of the
        // cluster size.
        let max_l2_cache =
            (max_l2_entries * self.l2_entry_size()).next_multiple_of(self.cluster_size);

        let combined_cache_size = o.cache_size.unwrap_or(0);
        let l2_cache_max_setting = o.l2_cache_size.unwrap_or(DEFAULT_L2_CACHE_MAX_SIZE);
        let mut refcount_cache_size = o.refcount_cache_size.unwrap_or(0);
        let mut l2_cache_entry_size = o.l2_cache_entry_size.unwrap_or(self.cluster_size);
        let mut l2_cache_size = max_l2_cache.min(l2_cache_max_setting);

        if o.cache_size.is_some() {
            if o.l2_cache_size.is_some() && o.refcount_cache_size.is_some() {
                return Err(Error::generic(
                    "cache-size, l2-cache-size and refcount-cache-size may not be set at the \
                     same time",
                ));
            } else if o.l2_cache_size.is_some() && l2_cache_max_setting > combined_cache_size {
                return Err(Error::generic("l2-cache-size may not exceed cache-size"));
            } else if refcount_cache_size > combined_cache_size {
                return Err(Error::generic("refcount-cache-size may not exceed cache-size"));
            }

            if o.l2_cache_size.is_some() {
                refcount_cache_size = combined_cache_size - l2_cache_size;
            } else if o.refcount_cache_size.is_some() {
                l2_cache_size = combined_cache_size - refcount_cache_size;
            } else if combined_cache_size >= max_l2_cache + min_refcount_cache {
                // As much as possible for the L2 cache, the rest for refcounts.
                l2_cache_size = max_l2_cache;
                refcount_cache_size = combined_cache_size - l2_cache_size;
            } else {
                refcount_cache_size = combined_cache_size.min(min_refcount_cache);
                l2_cache_size = combined_cache_size - refcount_cache_size;
            }
        }

        // When the L2 cache cannot cover the whole disk, smaller entries make loads and
        // evictions cheaper.
        if l2_cache_size < max_l2_cache && o.l2_cache_entry_size.is_none() {
            l2_cache_entry_size = self.cluster_size.min(4096);
        }

        if l2_cache_entry_size < (1 << MIN_CLUSTER_BITS)
            || l2_cache_entry_size > self.cluster_size
            || !l2_cache_entry_size.is_power_of_two()
        {
            return Err(Error::generic(format!(
                "L2 cache entry size must be a power of two between {} and the cluster size ({})",
                1 << MIN_CLUSTER_BITS,
                self.cluster_size
            )));
        }
        Ok((l2_cache_size, l2_cache_entry_size, refcount_cache_size))
    }

    /// `qcow2_update_options_prepare()`.
    pub(crate) fn update_options_prepare(&mut self, o: &OpenOptions) -> Result<Prepared> {
        let (l2_cache_size, l2_cache_entry_size, refcount_cache_size) = self.read_cache_sizes(o)?;

        let l2_cache_size = (l2_cache_size / l2_cache_entry_size).max(MIN_L2_CACHE_SIZE);
        if l2_cache_size > i32::MAX as u64 {
            return Err(Error::generic("L2 cache size too big"));
        }
        let refcount_cache_size =
            (refcount_cache_size / self.cluster_size).max(MIN_REFCOUNT_CACHE_SIZE);
        if refcount_cache_size > i32::MAX as u64 {
            return Err(Error::generic("Refcount cache size too big"));
        }

        // Flush the old caches before new ones replace them.
        self.cache_flush(super::cache::CacheId::L2)
            .map_err(|e| Error::from_io("Failed to flush the L2 table cache", e))?;
        self.cache_flush(super::cache::CacheId::Refcount)
            .map_err(|e| Error::from_io("Failed to flush the refcount block cache", e))?;

        let l2_slice_size = l2_cache_entry_size / self.l2_entry_size();
        let caches =
            Cache::new(l2_cache_size as usize, l2_cache_entry_size as usize).and_then(|l2| {
                Ok((l2, Cache::new(refcount_cache_size as usize, self.cluster_size as usize)?))
            });
        let Ok((l2_table_cache, refcount_block_cache)) = caches else {
            return Err(Error::generic("Could not allocate metadata caches"));
        };

        let cache_clean_interval = o.cache_clean_interval.unwrap_or(DEFAULT_CACHE_CLEAN_INTERVAL);
        #[cfg(not(target_os = "linux"))]
        if cache_clean_interval != 0 {
            return Err(Error::generic("cache-clean-interval not supported on this host"));
        }
        if cache_clean_interval > u32::MAX as u64 {
            return Err(Error::generic("Cache clean interval too big"));
        }

        // Lazy refcounts; going from enabled to disabled needs a clean image.
        let use_lazy_refcounts =
            o.lazy_refcounts.unwrap_or(self.compatible_features & QCOW2_COMPAT_LAZY_REFCOUNTS != 0);
        if use_lazy_refcounts && self.qcow_version < 3 {
            return Err(Error::generic(
                "Lazy refcounts require a qcow2 image with at least qemu 1.1 compatibility level",
            ));
        }
        if self.use_lazy_refcounts && !use_lazy_refcounts {
            self.mark_clean().map_err(|e| Error::from_io("Failed to disable lazy refcounts", e))?;
        }

        // Overlap checks.
        if let (Some(t), Some(c)) = (&o.overlap_check_template, &o.overlap_check) {
            if t != c {
                return Err(Error::generic(format!(
                    "Conflicting values for qcow2 options 'overlap-check' ('{c}') and \
                     'overlap-check.template' ('{t}')"
                )));
            }
        }
        let mode =
            o.overlap_check.as_deref().or(o.overlap_check_template.as_deref()).unwrap_or("cached");
        let template = match mode {
            "none" => 0,
            "constant" => OL_CONSTANT,
            "cached" => OL_CACHED,
            "all" => OL_ALL,
            _ => {
                return Err(Error::generic(format!(
                    "Unsupported value '{mode}' for qcow2 option 'overlap-check'. Allowed are any \
                     of the following: none, constant, cached, all"
                )));
            }
        };
        let mut overlap_check = 0;
        for i in 0..OL_MAX_BITNR {
            // The template gives the default, every flag can be switched on its own.
            let on = o.overlap_flags[i as usize].unwrap_or(template & (1 << i) != 0);
            overlap_check |= (on as u32) << i;
        }

        let discard_passthrough = [
            false,
            true,
            o.pass_discard_request.unwrap_or(self.flags.unmap),
            o.pass_discard_snapshot.unwrap_or(true),
            o.pass_discard_other.unwrap_or(false),
        ];

        let discard_no_unref = o.discard_no_unref.unwrap_or(false);
        if discard_no_unref && self.qcow_version < 3 {
            return Err(Error::generic("discard-no-unref is only supported since qcow2 version 3"));
        }

        let fmt = o.encrypt_format.as_deref();
        let crypto_format = match self.crypt_method_header {
            QCOW_CRYPT_NONE => {
                if let Some(f) = fmt {
                    return Err(Error::generic(format!(
                        "No encryption in image header, but options specified format '{f}'"
                    )));
                }
                None
            }
            QCOW_CRYPT_AES => {
                if let Some(f) = fmt.filter(|f| *f != "aes") {
                    return Err(Error::generic(format!(
                        "Header reported 'aes' encryption format but options specify '{f}'"
                    )));
                }
                Some(EncryptionFormat::Aes)
            }
            QCOW_CRYPT_LUKS => {
                if let Some(f) = fmt.filter(|f| *f != "luks") {
                    return Err(Error::generic(format!(
                        "Header reported 'luks' encryption format but options specify '{f}'"
                    )));
                }
                Some(EncryptionFormat::Luks)
            }
            m => return Err(Error::generic(format!("Unsupported encryption method {m}"))),
        };

        Ok(Prepared {
            l2_table_cache,
            refcount_block_cache,
            l2_slice_size,
            use_lazy_refcounts,
            overlap_check,
            discard_passthrough,
            discard_no_unref,
            cache_clean_interval,
            crypto_format,
        })
    }

    /// `qcow2_update_options_commit()`.
    pub(crate) fn update_options_commit(&mut self, r: Prepared) {
        self.l2_table_cache = r.l2_table_cache;
        self.refcount_block_cache = r.refcount_block_cache;
        self.l2_slice_size = r.l2_slice_size;
        self.overlap_check = r.overlap_check;
        self.use_lazy_refcounts = r.use_lazy_refcounts;
        self.discard_passthrough = r.discard_passthrough;
        self.discard_no_unref = r.discard_no_unref;
        self.cache_clean_interval = r.cache_clean_interval;
        self.last_cache_clean = std::time::Instant::now();
    }

    /// `qcow2_update_options()`.
    pub(crate) fn update_options(&mut self, o: &OpenOptions) -> Result<Option<EncryptionFormat>> {
        let r = self.update_options_prepare(o)?;
        let f = r.crypto_format;
        self.update_options_commit(r);
        Ok(f)
    }

    /// Opens the encryption context the header asks for, the `qcrypto_block_open()` calls in
    /// `qcow2_read_extensions()` and `qcow2_do_open()`.
    fn open_crypto(&mut self, format: EncryptionFormat, o: &OpenOptions) -> Result<()> {
        let provider = o.encryption.as_ref().ok_or_else(|| no_provider(format))?;
        let no_io = self.flags.no_io;
        let crypto = match format {
            EncryptionFormat::Luks => {
                let (offset, length) = self.crypto_header;
                let mut hio = ImageHeaderIo { file: &*self.file, offset, length };
                provider.open(format, &o.encrypt_opts, Some(&mut hio), no_io)?
            }
            EncryptionFormat::Aes => provider.open(format, &o.encrypt_opts, None, no_io)?,
        };
        self.crypto = Some(crypto);
        Ok(())
    }

    /// `qcow2_do_open()`. `open_data_file` opens the external data file the header names when
    /// the options did not give one; it gets the file name.
    pub(crate) fn do_open(
        &mut self,
        o: &OpenOptions,
        open_data_file: &mut dyn FnMut(&str) -> Result<Arc<dyn Storage>>,
    ) -> Result<()> {
        let flags = self.flags;
        self.options = o.clone();
        let mut hbuf = [0u8; HEADER_SIZE];
        self.file
            .pread(0, &mut hbuf)
            .map_err(|e| Error::from_io("Could not read qcow2 header", e))?;
        let mut header = Header::parse(&hbuf);

        if header.magic != QCOW_MAGIC {
            return Err(Error::generic("Image is not in qcow2 format"));
        }
        if header.version < 2 || header.version > 3 {
            return Err(Error::generic(format!("Unsupported qcow2 version {}", header.version)));
        }
        self.qcow_version = header.version;

        if header.cluster_bits < MIN_CLUSTER_BITS || header.cluster_bits > MAX_CLUSTER_BITS {
            return Err(Error::generic(format!(
                "Unsupported cluster size: 2^{}",
                header.cluster_bits
            )));
        }
        self.cluster_bits = header.cluster_bits;
        self.cluster_size = 1 << self.cluster_bits;

        if header.version == 2 {
            header.incompatible_features = 0;
            header.compatible_features = 0;
            header.autoclear_features = 0;
            header.refcount_order = 4;
            header.header_length = 72;
        } else if header.header_length < 104 {
            return Err(Error::generic("qcow2 header too short"));
        }

        if header.header_length as u64 > self.cluster_size {
            return Err(Error::generic("qcow2 header exceeds cluster size"));
        }

        if header.header_length as usize > HEADER_SIZE {
            let mut v = vec![0u8; header.header_length as usize - HEADER_SIZE];
            self.file
                .pread(HEADER_SIZE as u64, &mut v)
                .map_err(|e| Error::from_io("Could not read unknown qcow2 header fields", e))?;
            self.unknown_header_fields = v;
        }

        if header.backing_file_offset > self.cluster_size {
            return Err(Error::generic("Invalid backing file offset"));
        }
        let ext_end = if header.backing_file_offset != 0 {
            header.backing_file_offset
        } else {
            1 << header.cluster_bits
        };

        self.incompatible_features = header.incompatible_features;
        self.compatible_features = header.compatible_features;
        self.autoclear_features = header.autoclear_features;

        // Older images have no compression type field and can only use zlib.
        self.compression_type = if header.header_length as usize > HEADER_V3_MIN_SIZE {
            header.compression_type
        } else {
            QCOW2_COMPRESSION_TYPE_ZLIB
        };
        self.validate_compression_type().map_err(|e| e.0)?;

        if self.incompatible_features & !QCOW2_INCOMPAT_MASK != 0 {
            let _ = self.read_extensions(header.header_length as u64, ext_end, true);
            return Err(report_unsupported_feature(
                self.feature_table.as_deref(),
                self.incompatible_features & !QCOW2_INCOMPAT_MASK,
            ));
        }

        // Corrupt images may only be written to for repairs.
        if self.incompatible_features & QCOW2_INCOMPAT_CORRUPT != 0
            && flags.read_write
            && !flags.check
        {
            return Err(Error::generic("qcow2: Image is corrupt; cannot be opened read/write"));
        }

        self.subclusters_per_cluster =
            if self.has_subclusters() { QCOW_EXTL2_SUBCLUSTERS_PER_CLUSTER } else { 1 };
        self.subcluster_size = self.cluster_size / self.subclusters_per_cluster as u64;
        self.subcluster_bits = self.subcluster_size.trailing_zeros();
        if self.subcluster_size < (1 << MIN_CLUSTER_BITS) {
            return Err(Error::generic(format!(
                "Unsupported subcluster size: {}",
                self.subcluster_size
            )));
        }

        if header.refcount_order > 6 {
            return Err(Error::generic(
                "Reference count entry width too large; may not exceed 64 bits",
            ));
        }
        self.refcount_order = header.refcount_order;
        self.refcount_bits = 1 << self.refcount_order;
        self.refcount_max = 1u64 << (self.refcount_bits - 1);
        self.refcount_max += self.refcount_max - 1;

        self.crypt_method_header = header.crypt_method;
        if self.crypt_method_header != 0 {
            // AES derives its IVs from the guest offset; LUKS and anything later from the host
            // offset, the alternative being insecure.
            self.crypt_physical_offset = self.crypt_method_header != QCOW_CRYPT_AES;
        }

        self.l2_bits = self.cluster_bits - self.l2_entry_size().trailing_zeros();
        self.l2_size = 1 << self.l2_bits;
        // 2^(refcount_order - 3) is the refcount width in bytes.
        self.refcount_block_bits = self.cluster_bits + 3 - self.refcount_order;
        self.refcount_block_size = 1 << self.refcount_block_bits;
        self.total_size = header.size / BDRV_SECTOR_SIZE * BDRV_SECTOR_SIZE;
        self.csize_shift = 62 - (self.cluster_bits - 8);
        self.csize_mask = (1 << (self.cluster_bits - 8)) - 1;
        self.cluster_offset_mask = (1 << self.csize_shift) - 1;

        self.refcount_table_offset = header.refcount_table_offset;
        let refcount_table_size =
            (header.refcount_table_clusters as u64) << (self.cluster_bits - 3);

        if header.refcount_table_clusters == 0 && !flags.check {
            return Err(Error::generic("Image does not contain a reference count table"));
        }
        self.validate_table(
            self.refcount_table_offset,
            header.refcount_table_clusters as u64,
            self.cluster_size,
            QCOW_MAX_REFTABLE_SIZE,
            "Reference count table",
        )
        .map_err(|e| e.0)?;

        if !flags.check {
            // The size of the snapshot table is checked when it is read, the entries have
            // different sizes.
            self.validate_table(
                header.snapshots_offset,
                header.nb_snapshots as u64,
                SNAPSHOT_HEADER_SIZE,
                SNAPSHOT_HEADER_SIZE * QCOW_MAX_SNAPSHOTS,
                "Snapshot table",
            )
            .map_err(|e| e.0)?;
        }

        self.validate_table(
            header.l1_table_offset,
            header.l1_size as u64,
            L1E_SIZE,
            QCOW_MAX_L1_SIZE,
            "Active L1 table",
        )
        .map_err(|e| e.0)?;
        self.l1_size = header.l1_size;
        self.l1_table_offset = header.l1_table_offset;

        let l1_vm_state_index = self.size_to_l1(header.size);
        if l1_vm_state_index > i32::MAX as u64 {
            return Err(Error::generic("Image is too big"));
        }
        self.l1_vm_state_index = l1_vm_state_index;

        // The L1 table must cover at least the whole disk.
        if (self.l1_size as u64) < self.l1_vm_state_index {
            return Err(Error::generic("L1 table is too small"));
        }

        if self.l1_size > 0 {
            let mut buf = Vec::new();
            let bytes = self.l1_size as usize * L1E_SIZE as usize;
            if buf.try_reserve_exact(bytes).is_err() {
                return Err(Error::generic("Could not allocate L1 table"));
            }
            buf.resize(bytes, 0);
            self.file
                .pread(self.l1_table_offset, &mut buf)
                .map_err(|e| Error::from_io("Could not read L1 table", e))?;
            self.l1_table =
                buf.chunks_exact(8).map(|c| u64::from_be_bytes(c.try_into().unwrap())).collect();
        }

        let crypto_format = self.update_options(o)?;

        self.refcount_init(refcount_table_size)
            .map_err(|e| Error::from_io("Could not initialize refcount handling", e))?;

        let ext = self.read_extensions(header.header_length as u64, ext_end, false)?;
        let mut update_header = ext.need_update_header;

        // The LUKS header lives in the image, where the extension says.
        if crypto_format == Some(EncryptionFormat::Luks) && self.crypto_header.1 != 0 {
            self.open_crypto(EncryptionFormat::Luks, o)?;
        }

        if flags.no_io {
            // `qemu-img info` does not open the data file, so that an untrusted image can be
            // looked at without touching the files it names.
            self.data_file = None;
        } else {
            let mut data_file = o.data_file.clone();
            if self.incompatible_features & QCOW2_INCOMPAT_DATA_FILE != 0 {
                if data_file.is_none() {
                    if let Some(name) = self.image_data_file.clone() {
                        data_file = Some(open_data_file(&name)?);
                    }
                }
                if data_file.is_none() {
                    return Err(Error::generic("'data-file' is required for this image"));
                }
                self.data_file = data_file;
            } else {
                if data_file.is_some() {
                    return Err(Error::generic(
                        "'data-file' can only be set for images with an external data file",
                    ));
                }
                self.data_file = None;
                if self.data_file_is_raw() {
                    return Err(Error::generic("data-file-raw requires a data file"));
                }
            }
        }

        // Methods without a header region are opened here.
        if self.crypt_method_header != 0 && self.crypto.is_none() {
            if self.crypt_method_header == QCOW_CRYPT_AES {
                self.open_crypto(EncryptionFormat::Aes, o)?;
            } else {
                return Err(Error::generic(format!(
                    "Missing CRYPTO header for crypt method {}",
                    self.crypt_method_header
                )));
            }
        }

        if header.backing_file_offset != 0 {
            let len = header.backing_file_size as u64;
            if len > 1023u64.min(self.cluster_size - header.backing_file_offset) || len >= 4096 {
                return Err(Error::generic("Backing file name too long"));
            }
            let mut v = vec![0u8; len as usize];
            self.file
                .pread(header.backing_file_offset, &mut v)
                .map_err(|e| Error::from_io("Could not read backing file name", e))?;
            let name = super::header::c_string(&v);
            self.backing_file = name.clone();
            self.image_backing_file = Some(name);
        }

        // Snapshots are not needed by check, and a broken table must not stop it.
        if !flags.check {
            self.snapshots_offset = header.snapshots_offset;
            self.read_snapshots(header.nb_snapshots)?;
        }

        // Unknown autoclear bits are dropped.
        update_header |= self.autoclear_features & !QCOW2_AUTOCLEAR_MASK != 0;
        update_header = update_header && self.writable();
        if update_header {
            self.autoclear_features &= QCOW2_AUTOCLEAR_MASK;
        }

        if !flags.inactive {
            let header_updated = self.load_dirty_bitmaps()?;
            update_header = update_header && !header_updated;
        }

        if update_header {
            self.update_header().map_err(|e| Error::from_io("Could not update qcow2 header", e))?;
        }

        // Repair the image if it is dirty.
        if !flags.check && self.writable() && self.incompatible_features & QCOW2_INCOMPAT_DIRTY != 0
        {
            let mut result = CheckResult::default();
            let r = self.check(&mut result, FIX_ERRORS | FIX_LEAKS);
            let r = match r {
                Ok(()) if result.check_errors != 0 => Err(crate::node::errno(libc::EIO)),
                r => r,
            };
            r.map_err(|e| Error::from_io("Could not repair dirty image", e))?;
        }
        Ok(())
    }
}

/// `BDRV_SECTOR_SIZE`.
const BDRV_SECTOR_SIZE: u64 = 512;
