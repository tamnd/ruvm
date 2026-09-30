// SPDX-License-Identifier: GPL-2.0-or-later

//! Creating images and measuring what they need: `qcow2_co_create()`,
//! `qcow2_co_create_opts()`, `qcow2_set_up_encryption()` and `qcow2_measure()` from
//! block/qcow2.c, with the `-o` option lists of `qcow2_create_opts` and `qcow2_amend_opts`.

use std::io;
use std::sync::Arc;

use ruvm_base::{Error, Result};
use ruvm_qapi::cutils;
use ruvm_qapi::opts::{QemuOptDesc, QemuOptType};
use ruvm_qapi::visit::{parse_option_size, qapi_bool_parse};

use super::crypto::{EncryptionFormat, EncryptionOptions, EncryptionProvider, HeaderIo};
use super::header::HEADER_SIZE;
use super::io::Storage;
use super::open::{OpenOptions, no_provider};
use super::resize::{Prealloc, parse_prealloc};
use super::state::*;
use crate::node::{BDRV_BLOCK_ALLOCATED, BDRV_BLOCK_DATA, BDRV_BLOCK_ZERO};

const BDRV_SECTOR_SIZE: u64 = 512;

macro_rules! opt {
    ($name:expr, $ty:ident, $help:expr) => {
        QemuOptDesc::new($name, QemuOptType::$ty).help($help)
    };
    ($name:expr, $ty:ident, $help:expr, $def:expr) => {
        QemuOptDesc::new($name, QemuOptType::$ty).help($help).default_value($def)
    };
}

/// `QCOW_COMMON_OPTIONS`, followed by the list terminator so both lists can splice it in.
macro_rules! common_options {
    ($($head:expr,)*) => {
        [
            $($head,)*
            opt!("size", Size, "Virtual disk size"),
            opt!("compat", String, "Compatibility level (v2 [0.10] or v3 [1.1])"),
            opt!("backing_file", String, "File name of a base image"),
            opt!("backing_fmt", String, "Image format of the base image"),
            opt!("data_file", String, "File name of an external data file"),
            opt!(
                "data_file_raw",
                Bool,
                "The external data file must stay valid as a raw image"
            ),
            opt!("lazy_refcounts", Bool, "Postpone refcount updates", "off"),
            opt!("refcount_bits", Number, "Width of a reference count entry in bits", "16"),
        ]
    };
}

/// `qcow2_create_opts`: the options `qemu-img create -f qcow2 -o` takes, in QEMU's order.
pub(crate) static CREATE_OPTS: [QemuOptDesc; 22] = common_options!(
    opt!(
        "encryption",
        Bool,
        "Encrypt the image with format 'aes'. (Deprecated in favor of encrypt.format=aes)"
    ),
    opt!("encrypt.format", String, "Encrypt the image, format choices: 'aes', 'luks'"),
    opt!("encrypt.key-secret", String, "ID of secret providing qcow AES key or LUKS passphrase"),
    opt!("encrypt.cipher-alg", String, "Name of encryption cipher algorithm"),
    opt!("encrypt.cipher-mode", String, "Name of encryption cipher mode"),
    opt!("encrypt.ivgen-alg", String, "Name of IV generator algorithm"),
    opt!("encrypt.ivgen-hash-alg", String, "Name of IV generator hash algorithm"),
    opt!("encrypt.hash-alg", String, "Name of encryption hash algorithm"),
    opt!("encrypt.iter-time", Number, "Time to spend in PBKDF in milliseconds"),
    opt!("cluster_size", Size, "qcow2 cluster size", "65536"),
    opt!("extended_l2", Bool, "Extended L2 tables", "off"),
    opt!(
        "preallocation",
        String,
        "Preallocation mode (allowed values: off, metadata, falloc, full)"
    ),
    opt!(
        "compression_type",
        String,
        "Compression method used for image cluster compression",
        "zlib"
    ),
    opt!(
        "keep_data_file",
        Bool,
        "Assume the external data file already exists and do not overwrite it"
    ),
);

/// `qcow2_amend_opts`: the options `qemu-img amend -f qcow2 -o` takes.
pub(crate) static AMEND_OPTS: [QemuOptDesc; 13] = common_options!(
    opt!("encrypt.state", String, "Select new state of affected keyslots (active/inactive)"),
    opt!("encrypt.keyslot", Number, "Select a single keyslot to modify explicitly"),
    opt!("encrypt.old-secret", String, "Select all keyslots that match this password"),
    opt!(
        "encrypt.new-secret",
        String,
        "New secret to set in the matching keyslots. Empty string to erase"
    ),
    opt!("encrypt.iter-time", Number, "Time to spend in PBKDF in milliseconds"),
);

/// The options of a new image, `BlockdevCreateOptionsQcow2` without the `file` and
/// `data-file` references, which are passed to [`create`] as stores. `None` means QEMU's
/// default.
#[derive(Clone, Debug, Default)]
pub(crate) struct CreateOptions {
    /// `size`, the virtual disk size in bytes.
    pub size: u64,
    /// `version`: 2 (`compat=0.10`) or 3 (`compat=1.1`, the default).
    pub version: Option<u32>,
    /// `cluster-size`, 64 KiB by default.
    pub cluster_size: Option<u64>,
    /// `extended-l2`, off by default.
    pub extended_l2: Option<bool>,
    /// `preallocation`, off by default.
    pub preallocation: Option<Prealloc>,
    /// `lazy-refcounts`, off by default.
    pub lazy_refcounts: Option<bool>,
    /// `refcount-bits`, 16 by default.
    pub refcount_bits: Option<u64>,
    /// `compression-type`: 0 for zlib (the default), 1 for zstd.
    pub compression_type: Option<u8>,
    /// `backing-file`.
    pub backing_file: Option<String>,
    /// `backing-fmt`.
    pub backing_fmt: Option<String>,
    /// The external data file name to record in the image. When unset and a data file is
    /// passed to [`create`], its [`Storage::filename`] is used.
    pub data_file: Option<String>,
    /// `data-file-raw`, off by default.
    pub data_file_raw: Option<bool>,
    /// `keep_data_file`: the data file exists already and must not be recreated. Only
    /// [`create_file`] looks at it.
    pub keep_data_file: bool,
    /// `encrypt.format`.
    pub encrypt_format: Option<EncryptionFormat>,
    /// The other `encrypt.*` options, without the prefix.
    pub encrypt_opts: EncryptionOptions,
    /// What creates the encryption header of encrypted images.
    pub encryption: Option<Arc<dyn EncryptionProvider>>,
    /// Whether the legacy `encryption` option was given, to catch it together with
    /// `encrypt.format`.
    legacy_encryption: bool,
}

fn parse_number(name: &str, value: &str) -> Result<u64> {
    match cutils::strtou64(value, 0, true) {
        Ok((v, _)) => Ok(v),
        Err((cutils::Errno::Range, _)) => {
            Err(Error::generic(format!("Value '{value}' is too large for parameter '{name}'")))
        }
        Err(_) => Err(Error::generic(format!("Parameter '{name}' expects a number"))),
    }
}

fn parse_version(value: &str) -> Result<u32> {
    match value {
        "0.10" | "v2" => Ok(2),
        "1.1" | "v3" => Ok(3),
        _ => Err(Error::generic(format!("Parameter 'version' does not accept value '{value}'"))),
    }
}

/// `Qcow2CompressionType` from its name.
fn parse_compression_type(value: &str) -> Result<u8> {
    match value {
        "zlib" => Ok(QCOW2_COMPRESSION_TYPE_ZLIB),
        // This build has no zstd, and like a QEMU built without it the value is not even in
        // the enum.
        _ => Err(Error::generic(format!(
            "Parameter 'compression-type' does not accept value '{value}'"
        ))),
    }
}

impl CreateOptions {
    /// Sets an option from its `-o` form (`cluster_size`, `compat`, ...) or its QMP form
    /// (`cluster-size`, `version`, ...), applying the conversions `qcow2_co_create_opts()`
    /// makes. Returns `Ok(false)` for a key qcow2 does not know.
    pub(crate) fn set(&mut self, key: &str, value: &str) -> Result<bool> {
        let b = || qapi_bool_parse(key, value);
        match key {
            "size" => self.size = parse_option_size(key, value)?,
            "compat" | "version" => self.version = Some(parse_version(value)?),
            "cluster_size" | "cluster-size" => {
                self.cluster_size = Some(parse_option_size(key, value)?)
            }
            "extended_l2" | "extended-l2" => self.extended_l2 = Some(b()?),
            "preallocation" => {
                self.preallocation = Some(parse_prealloc(value).map_err(|_| {
                    Error::generic(format!(
                        "Parameter 'preallocation' does not accept value '{value}'"
                    ))
                })?)
            }
            "lazy_refcounts" | "lazy-refcounts" => self.lazy_refcounts = Some(b()?),
            "refcount_bits" | "refcount-bits" => {
                self.refcount_bits = Some(parse_number(key, value)?)
            }
            "compression_type" | "compression-type" => {
                self.compression_type = Some(parse_compression_type(value)?)
            }
            "backing_file" | "backing-file" => self.backing_file = Some(value.to_string()),
            "backing_fmt" | "backing-fmt" => self.backing_fmt = Some(value.to_string()),
            "data_file" | "data-file" => self.data_file = Some(value.to_string()),
            "data_file_raw" | "data-file-raw" => self.data_file_raw = Some(b()?),
            "keep_data_file" => {
                self.keep_data_file = match value {
                    "on" => true,
                    "off" => false,
                    _ => {
                        return Err(Error::generic(format!(
                            "Invalid value '{value}' for 'keep_data_file': Must be 'on' or \
                             'off'"
                        )));
                    }
                }
            }
            "encryption" => {
                if b()? {
                    if self.encrypt_format.is_some() {
                        return Err(conflict());
                    }
                    self.encrypt_format = Some(EncryptionFormat::Aes);
                    self.legacy_encryption = true;
                }
            }
            "encrypt.format" => {
                if self.legacy_encryption {
                    return Err(conflict());
                }
                let f = match value {
                    "qcow" => Some(EncryptionFormat::Aes),
                    v => EncryptionFormat::parse(v),
                };
                let Some(f) = f else {
                    return Err(Error::generic(format!(
                        "Parameter 'format' does not accept value '{value}'"
                    )));
                };
                self.encrypt_format = Some(f);
            }
            _ => {
                let Some(k) = key.strip_prefix("encrypt.") else {
                    return Ok(false);
                };
                self.encrypt_opts.push((k.to_string(), value.to_string()));
            }
        }
        Ok(true)
    }
}

fn conflict() -> Error {
    Error::generic("'encrypt.format' and its alias 'encryption' can't be used at the same time")
}

/// `validate_cluster_size()`.
pub(crate) fn validate_cluster_size(cluster_size: u64, extended_l2: bool) -> Result<()> {
    let bits = cluster_size.trailing_zeros();
    if !(MIN_CLUSTER_BITS..=MAX_CLUSTER_BITS).contains(&bits) || 1u64 << bits != cluster_size {
        return Err(Error::generic(format!(
            "Cluster size must be a power of two between {} and {}k",
            1 << MIN_CLUSTER_BITS,
            1 << (MAX_CLUSTER_BITS - 10)
        )));
    }
    if extended_l2 {
        let min = (1u64 << MIN_CLUSTER_BITS) * QCOW_EXTL2_SUBCLUSTERS_PER_CLUSTER as u64;
        if cluster_size < min {
            return Err(Error::generic(format!(
                "Extended L2 entries are only supported with cluster sizes of at least {min} \
                 bytes"
            )));
        }
    }
    Ok(())
}

/// The LUKS header area while creating, `qcow2_crypto_hdr_init_func()` and
/// `qcow2_crypto_hdr_write_func()`.
struct CreateHeaderIo<'a> {
    s: &'a mut State,
}

impl HeaderIo for CreateHeaderIo<'_> {
    fn read(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        let (hoff, hlen) = self.s.crypto_header;
        if offset + buf.len() as u64 > hlen {
            return Err(Error::generic("Request for data outside of extension header"));
        }
        self.s
            .file
            .pread(hoff + offset, buf)
            .map_err(|e| Error::from_io("Could not read encryption header", e))
    }

    fn init(&mut self, len: u64) -> Result<()> {
        let off = self.s.alloc_clusters(len).map_err(|e| {
            Error::from_io(format_args!("Cannot allocate cluster for LUKS header size {len}"), e)
        })?;
        self.s.crypto_header = (off, len);
        // Zero the whole clusters so that the parts of the header nobody writes (unused key
        // slots, for example) have predictable content.
        let clusterlen = self.s.size_to_clusters(len) * self.s.cluster_size;
        assert!(self.s.pre_write_overlap_check(0, off, clusterlen, false).is_ok());
        self.s
            .file
            .pwrite_zeroes(off, clusterlen, false)
            .map_err(|e| Error::from_io("Could not zero fill encryption header", e))
    }

    fn write(&mut self, offset: u64, buf: &[u8]) -> Result<()> {
        let (hoff, hlen) = self.s.crypto_header;
        if offset + buf.len() as u64 > hlen {
            return Err(Error::generic("Request for data outside of extension header"));
        }
        // QEMU says "read" here too.
        self.s
            .file
            .pwrite(hoff + offset, buf)
            .map_err(|e| Error::from_io("Could not read encryption header", e))
    }
}

impl State {
    /// `qcow2_set_up_encryption()`.
    pub(crate) fn set_up_encryption(
        &mut self,
        format: EncryptionFormat,
        opts: &EncryptionOptions,
        provider: Option<&Arc<dyn EncryptionProvider>>,
    ) -> Result<()> {
        let Some(provider) = provider else {
            return Err(no_provider(format));
        };
        self.crypt_method_header = match format {
            EncryptionFormat::Luks => QCOW_CRYPT_LUKS,
            EncryptionFormat::Aes => QCOW_CRYPT_AES,
        };
        let r = match format {
            EncryptionFormat::Luks => {
                let mut hio = CreateHeaderIo { s: self };
                provider.create(format, opts, Some(&mut hio))
            }
            EncryptionFormat::Aes => provider.create(format, opts, None),
        };
        // The context is only needed to write the header; it is dropped right away.
        drop(r?);
        self.update_header().map_err(|e| Error::from_io("Could not write encryption header", e))
    }

    /// `qcow2_inactivate()`: stores the bitmaps, writes the caches back and clears the dirty
    /// bit. Errors are reported, the first one is returned.
    pub(crate) fn inactivate(&mut self, node_name: &str) -> io::Result<()> {
        use ruvm_base::report::error_report;
        let mut result = Ok(());
        if let Err(e) = self.store_persistent_dirty_bitmaps(true) {
            result = Err(crate::node::errno(libc::EINVAL));
            error_report(&format!(
                "Lost persistent bitmaps during inactivation of node '{node_name}': {}",
                e.message()
            ));
        }
        if let Err(e) = self.cache_flush(super::cache::CacheId::L2) {
            error_report(&format!(
                "Failed to flush the L2 table cache: {}",
                ruvm_base::error::strerror(&e)
            ));
            result = Err(e);
        }
        if let Err(e) = self.cache_flush(super::cache::CacheId::Refcount) {
            error_report(&format!(
                "Failed to flush the refcount block cache: {}",
                ruvm_base::error::strerror(&e)
            ));
            result = Err(e);
        }
        // A read-only image cannot clear a dirty bit it inherited.
        if result.is_ok() && self.writable() {
            let _ = self.mark_clean();
        }
        result
    }
}

/// `qcow2_co_create()`: writes a new image to `file`, with `data_file` as its external data
/// file if given.
pub(crate) fn create(
    file: Arc<dyn Storage>,
    data_file: Option<Arc<dyn Storage>>,
    o: &CreateOptions,
) -> Result<()> {
    if o.size % BDRV_SECTOR_SIZE != 0 {
        return Err(Error::generic(format!(
            "Image size must be a multiple of {BDRV_SECTOR_SIZE} bytes"
        )));
    }
    let version = o.version.unwrap_or(3);
    let cluster_size = o.cluster_size.unwrap_or(DEFAULT_CLUSTER_SIZE);
    let extended_l2 = o.extended_l2.unwrap_or(false);
    if extended_l2 && version < 3 {
        return Err(Error::generic(
            "Extended L2 entries are only supported with compatibility level 1.1 and above \
             (use version=v3 or greater)",
        ));
    }
    validate_cluster_size(cluster_size, extended_l2)?;

    let mut prealloc = o.preallocation.unwrap_or(Prealloc::Off);
    if o.backing_file.is_some() && prealloc != Prealloc::Off && !extended_l2 {
        return Err(Error::generic(
            "Backing file and preallocation can only be used at the same time if extended_l2 \
             is on",
        ));
    }
    if o.backing_fmt.is_some() && o.backing_file.is_none() {
        return Err(Error::generic("Backing format cannot be used without backing file"));
    }

    let lazy_refcounts = o.lazy_refcounts.unwrap_or(false);
    if version < 3 && lazy_refcounts {
        return Err(Error::generic(
            "Lazy refcounts only supported with compatibility level 1.1 and above (use \
             version=v3 or greater)",
        ));
    }

    let refcount_bits = o.refcount_bits.unwrap_or(16);
    if refcount_bits > 64 || !refcount_bits.is_power_of_two() {
        return Err(Error::generic(
            "Refcount width must be a power of two and may not exceed 64 bits",
        ));
    }
    if version < 3 && refcount_bits != 16 {
        return Err(Error::generic(
            "Different refcount widths than 16 bits require compatibility level 1.1 or above \
             (use version=v3 or greater)",
        ));
    }
    let refcount_order = refcount_bits.trailing_zeros();

    let data_file_raw = o.data_file_raw.unwrap_or(false);
    if data_file_raw && data_file.is_none() {
        return Err(Error::generic("data-file-raw requires data-file"));
    }
    if data_file_raw && o.backing_file.is_some() {
        return Err(Error::generic(
            "Backing file and data-file-raw cannot be used at the same time",
        ));
    }
    if data_file_raw && prealloc == Prealloc::Off {
        // The metadata must map the data file 1:1, so that reading the image through qcow2
        // gives the same as reading the data file as raw.
        prealloc = Prealloc::Metadata;
    }
    if data_file.is_some() && version < 3 {
        return Err(Error::generic(
            "External data files are only supported with compatibility level 1.1 and above \
             (use version=v3 or greater)",
        ));
    }

    let mut compression_type = QCOW2_COMPRESSION_TYPE_ZLIB;
    if let Some(ct) = o.compression_type {
        if ct != QCOW2_COMPRESSION_TYPE_ZLIB {
            if version < 3 {
                return Err(Error::generic(
                    "Non-zlib compression type is only supported with compatibility level 1.1 \
                     and above (use version=v3 or greater)",
                ));
            }
            // This build has no zstd, which is what QEMU says without CONFIG_ZSTD.
            return Err(Error::generic("Unknown compression type"));
        }
        compression_type = ct;
    }

    // A minimal header with a refcount table of one cluster pointing at one empty refcount
    // block. Opening the image then fixes the refcounts of these three clusters.
    let cs = cluster_size as usize;
    let mut buf = vec![0u8; cs];
    let mut incompat = 0;
    if data_file.is_some() {
        incompat |= QCOW2_INCOMPAT_DATA_FILE;
    }
    if compression_type != QCOW2_COMPRESSION_TYPE_ZLIB {
        incompat |= QCOW2_INCOMPAT_COMPRESSION;
    }
    if extended_l2 {
        incompat |= QCOW2_INCOMPAT_EXTL2;
    }
    let header = super::header::Header {
        magic: QCOW_MAGIC,
        version,
        backing_file_offset: 0,
        backing_file_size: 0,
        cluster_bits: cluster_size.trailing_zeros(),
        size: 0,
        crypt_method: QCOW_CRYPT_NONE,
        l1_size: 0,
        l1_table_offset: 0,
        refcount_table_offset: cluster_size,
        refcount_table_clusters: 1,
        nb_snapshots: 0,
        snapshots_offset: 0,
        incompatible_features: incompat,
        compatible_features: if lazy_refcounts { QCOW2_COMPAT_LAZY_REFCOUNTS } else { 0 },
        autoclear_features: if data_file_raw { QCOW2_AUTOCLEAR_DATA_FILE_RAW } else { 0 },
        refcount_order,
        header_length: HEADER_SIZE as u32,
        compression_type,
    };
    buf[..HEADER_SIZE].copy_from_slice(&header.to_bytes());
    file.pwrite(0, &buf).map_err(|e| Error::from_io("Could not write qcow2 header", e))?;

    let mut table = vec![0u8; 2 * cs];
    table[..8].copy_from_slice(&(2 * cluster_size).to_be_bytes());
    file.pwrite(cluster_size, &table)
        .map_err(|e| Error::from_io("Could not write refcount table", e))?;

    // Open the image and make it consistent.
    let flags = OpenFlags { read_write: true, ..OpenFlags::default() };
    let mut s = State::blank(file.clone(), flags);
    let oo = OpenOptions { flags, data_file: data_file.clone(), ..OpenOptions::default() };
    s.do_open(&oo, &mut |_| Err(Error::generic("'data-file' is required for this image")))?;

    let r = (|| -> Result<()> {
        match s.alloc_clusters(3 * cluster_size) {
            Err(e) => {
                return Err(Error::from_io(
                    "Could not allocate clusters for qcow2 header and refcount table",
                    e,
                ));
            }
            Ok(0) => {}
            Ok(_) => panic!("Huh, first cluster in empty image is already in use?"),
        }

        if let Some(d) = &data_file {
            s.image_data_file = o.data_file.clone().or_else(|| d.filename());
        }

        s.update_header().map_err(|e| Error::from_io("Could not update qcow2 header", e))?;

        s.truncate(o.size, false, prealloc).map_err(|e| e.prepend("Could not resize image: "))?;

        if let Some(b) = &o.backing_file {
            s.change_backing_file(Some(b), o.backing_fmt.as_deref()).map_err(|e| {
                Error::from_io(
                    format_args!(
                        "Could not assign backing file '{b}' with format '{}'",
                        o.backing_fmt.as_deref().unwrap_or("(null)")
                    ),
                    e,
                )
            })?;
        }

        if let Some(f) = o.encrypt_format {
            s.set_up_encryption(f, &o.encrypt_opts, o.encryption.as_ref())?;
        } else if !o.encrypt_opts.is_empty() {
            return Err(Error::generic("Parameter 'encrypt.format' is missing"));
        }
        Ok(())
    })();
    let closed = s.inactivate("");
    r?;
    closed.map_err(|e| Error::from_io("Could not flush the new image", e))?;
    s.flush_all().map_err(|e| Error::from_io("Could not flush the new image", e))?;
    Ok(())
}

/// The image `qemu-img measure` looks at, the `in_bs` of `qcow2_measure()`.
pub(crate) trait MeasureSource {
    /// `bdrv_getlength()`.
    fn len(&self) -> io::Result<u64>;
    /// `bdrv_block_status_above()` down to the bottom of the chain: the `BDRV_BLOCK_*` status
    /// of the range starting at `offset` and how many bytes it covers.
    fn block_status(&self, offset: u64, bytes: u64) -> io::Result<(u32, u64)>;
    /// `qcow2_get_persistent_dirty_bitmap_size()`, or `None` when the image cannot have
    /// persistent bitmaps (`bdrv_supports_persistent_dirty_bitmap()`).
    fn bitmaps_size(&self, cluster_size: u64) -> Option<u64> {
        let _ = cluster_size;
        None
    }
}

/// `qcow2_calc_prealloc_size()`.
fn calc_prealloc_size(
    total_size: u64,
    cluster_size: u64,
    refcount_order: u32,
    extended_l2: bool,
) -> u64 {
    let aligned_total_size = total_size.next_multiple_of(cluster_size);
    let l2e_size = if extended_l2 { L2E_SIZE_EXTENDED } else { L2E_SIZE_NORMAL };
    let mut meta_size = cluster_size;

    let nl2e = (aligned_total_size / cluster_size).next_multiple_of(cluster_size / l2e_size);
    meta_size += nl2e * l2e_size;

    let nl1e = (nl2e * l2e_size / cluster_size).next_multiple_of(cluster_size / L1E_SIZE);
    meta_size += nl1e * L1E_SIZE;

    meta_size += super::refcount::refcount_metadata_size(
        (meta_size + aligned_total_size) / cluster_size,
        cluster_size,
        refcount_order,
        false,
    )
    .0;
    meta_size + aligned_total_size
}

/// The result of [`measure`], `BlockMeasureInfo`.
pub(crate) type MeasureInfo = ruvm_qapi::types::BlockMeasureInfo;

/// `qcow2_measure()`: how big an image made with `o` gets, empty or holding the data of
/// `input`. The option checks use the `-o` wording of QEMU, which differs a little from
/// [`create`]'s.
pub(crate) fn measure(o: &CreateOptions, input: Option<&dyn MeasureSource>) -> Result<MeasureInfo> {
    let extended_l2 = o.extended_l2.unwrap_or(false);
    let cluster_size = o.cluster_size.unwrap_or(DEFAULT_CLUSTER_SIZE);
    validate_cluster_size(cluster_size, extended_l2)?;
    let version = o.version.unwrap_or(3);
    let refcount_bits = o.refcount_bits.unwrap_or(16);
    if refcount_bits > 64 || !refcount_bits.is_power_of_two() {
        return Err(Error::generic(
            "Refcount width must be a power of two and may not exceed 64 bits",
        ));
    }
    if version < 3 && refcount_bits != 16 {
        return Err(Error::generic(
            "Different refcount widths than 16 bits require compatibility level 1.1 or above \
             (use compat=1.1 or greater)",
        ));
    }
    let prealloc = o.preallocation.unwrap_or(Prealloc::Off);
    let has_backing_file = o.backing_file.is_some();

    let mut luks_payload_size = 0;
    if o.encrypt_format == Some(EncryptionFormat::Luks) {
        let Some(p) = &o.encryption else {
            return Err(no_provider(EncryptionFormat::Luks));
        };
        let headerlen = p.header_len(EncryptionFormat::Luks, &o.encrypt_opts)?;
        luks_payload_size = headerlen.next_multiple_of(cluster_size);
    }

    let mut virtual_size = o.size.next_multiple_of(cluster_size);
    let l2e_size = if extended_l2 { L2E_SIZE_EXTENDED } else { L2E_SIZE_NORMAL };
    let l2_tables = (virtual_size / cluster_size).div_ceil(cluster_size / l2e_size);
    if l2_tables * L1E_SIZE > QCOW_MAX_L1_SIZE {
        return Err(Error::generic(
            "The image size is too large (try using a larger cluster size)",
        ));
    }

    let mut required = 0u64;
    if let Some(inp) = input {
        let ssize = inp.len().map_err(|e| Error::from_io("Unable to get image virtual_size", e))?;
        virtual_size = ssize.next_multiple_of(cluster_size);
        if has_backing_file {
            // How much the backing chain shares with the input is unknown, so every cluster
            // may need to be written.
            required = virtual_size;
        } else {
            let mut offset = 0;
            while offset < ssize {
                let (st, mut pnum) = inp
                    .block_status(offset, ssize - offset)
                    .map_err(|e| Error::from_io("Unable to get block status", e))?;
                if st & BDRV_BLOCK_ZERO != 0 {
                    // Zero ranges need nothing without a backing file.
                } else if st & (BDRV_BLOCK_DATA | BDRV_BLOCK_ALLOCATED)
                    == BDRV_BLOCK_DATA | BDRV_BLOCK_ALLOCATED
                {
                    pnum = (offset + pnum).next_multiple_of(cluster_size) - offset;
                    required += offset % cluster_size + pnum;
                }
                if pnum == 0 {
                    break;
                }
                offset += pnum;
            }
        }
    }

    if prealloc == Prealloc::Full || prealloc == Prealloc::Falloc {
        required = virtual_size;
    }

    let fully_allocated = luks_payload_size
        + calc_prealloc_size(
            virtual_size,
            cluster_size,
            refcount_bits.trailing_zeros(),
            extended_l2,
        );
    let bitmaps =
        if version >= 3 { input.and_then(|i| i.bitmaps_size(cluster_size)) } else { None };
    Ok(MeasureInfo {
        required: (fully_allocated - virtual_size + required) as i64,
        fully_allocated: fully_allocated as i64,
        bitmaps: bitmaps.map(|b| b as i64),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn option_lists_have_qemu_order() {
        let names: Vec<_> = CREATE_OPTS.iter().map(|d| d.name).collect();
        assert_eq!(names[0], "encryption");
        assert_eq!(names[9], "cluster_size");
        assert_eq!(*names.last().unwrap(), "refcount_bits");
        let names: Vec<_> = AMEND_OPTS.iter().map(|d| d.name).collect();
        assert_eq!(names[0], "encrypt.state");
        assert_eq!(names.len(), 13);
    }

    #[test]
    fn cluster_size_messages() {
        assert_eq!(
            validate_cluster_size(1000, false).unwrap_err().message(),
            "Cluster size must be a power of two between 512 and 2048k"
        );
        assert_eq!(
            validate_cluster_size(4096, true).unwrap_err().message(),
            "Extended L2 entries are only supported with cluster sizes of at least 16384 bytes"
        );
        assert!(validate_cluster_size(16384, true).is_ok());
    }

    #[test]
    fn legacy_options() {
        let mut o = CreateOptions::default();
        assert!(o.set("compat", "0.10").unwrap());
        assert_eq!(o.version, Some(2));
        assert!(o.set("encryption", "on").unwrap());
        assert_eq!(
            o.set("encrypt.format", "luks").unwrap_err().message(),
            "'encrypt.format' and its alias 'encryption' can't be used at the same time"
        );
        assert_eq!(
            o.set("compat", "foo").unwrap_err().message(),
            "Parameter 'version' does not accept value 'foo'"
        );
        assert!(!o.set("nonsense", "1").unwrap());
    }

    #[test]
    fn prealloc_size_matches_qemu() {
        // qemu-img measure -O qcow2 --size 1G: fully allocated 1074135040.
        assert_eq!(calc_prealloc_size(1 << 30, 65536, 4, false), 1_074_135_040);
    }
}
