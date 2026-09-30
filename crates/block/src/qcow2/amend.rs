// SPDX-License-Identifier: GPL-2.0-or-later

//! `qemu-img amend` and `x-blockdev-amend` for qcow2: `qcow2_amend_options()`,
//! `qcow2_upgrade()`, `qcow2_downgrade()` and `qcow2_co_amend()` from block/qcow2.c.

use std::io;

use ruvm_base::{Error, Result};
use ruvm_qapi::visit::{parse_option_size, qapi_bool_parse};

use super::crypto::EncryptionOptions;
use super::open::ImageHeaderIo;
use super::resize::Prealloc;
use super::state::*;

/// The `-o` options of `qemu-img amend`, `qcow2_amend_opts`. `None` means the option was not
/// given, and only given options change anything.
#[derive(Clone, Debug, Default)]
pub(crate) struct AmendOptions {
    /// `size`.
    pub size: Option<u64>,
    /// `compat`, as given: `0.10`, `v2`, `1.1` or `v3`. Checked when amending.
    pub compat: Option<String>,
    /// `backing_file`. It cannot be changed, only repeated.
    pub backing_file: Option<String>,
    /// `backing_fmt`. The same.
    pub backing_fmt: Option<String>,
    /// `data_file`.
    pub data_file: Option<String>,
    /// `data_file_raw`.
    pub data_file_raw: Option<bool>,
    /// `lazy_refcounts`.
    pub lazy_refcounts: Option<bool>,
    /// `refcount_bits`.
    pub refcount_bits: Option<u64>,
    /// The `encrypt.*` options (`state`, `keyslot`, `old-secret`, `new-secret`, `iter-time`),
    /// without the prefix.
    pub encrypt_opts: EncryptionOptions,
}

/// The `encrypt.*` options `qcow2_amend_opts` knows.
const ENCRYPT_AMEND_KEYS: [&str; 5] = ["state", "keyslot", "old-secret", "new-secret", "iter-time"];

impl AmendOptions {
    /// Sets an option from its `-o` form. Returns `Ok(false)` for a key `qemu-img amend` does
    /// not take for qcow2.
    pub(crate) fn set(&mut self, key: &str, value: &str) -> Result<bool> {
        let b = || qapi_bool_parse(key, value).map(Some);
        match key {
            "size" => self.size = Some(parse_option_size(key, value)?),
            "compat" => self.compat = Some(value.to_string()),
            "backing_file" => self.backing_file = Some(value.to_string()),
            "backing_fmt" => self.backing_fmt = Some(value.to_string()),
            "data_file" => self.data_file = Some(value.to_string()),
            "data_file_raw" => self.data_file_raw = b()?,
            "lazy_refcounts" => self.lazy_refcounts = b()?,
            "refcount_bits" => {
                self.refcount_bits = Some(match ruvm_qapi::cutils::strtou64(value, 0, true) {
                    Ok((v, _)) => v,
                    Err((ruvm_qapi::cutils::Errno::Range, _)) => {
                        return Err(Error::generic(format!(
                            "Value '{value}' is too large for parameter '{key}'"
                        )));
                    }
                    Err(_) => {
                        return Err(Error::generic(format!("Parameter '{key}' expects a number")));
                    }
                })
            }
            _ => {
                let Some(k) = key.strip_prefix("encrypt.") else {
                    return Ok(false);
                };
                if !ENCRYPT_AMEND_KEYS.contains(&k) {
                    return Ok(false);
                }
                if k == "keyslot" || k == "iter-time" {
                    // QEMU_OPT_NUMBER: checked when the option is parsed.
                    if ruvm_qapi::cutils::strtou64(value, 0, true).is_err() {
                        return Err(Error::generic(format!("Parameter '{key}' expects a number")));
                    }
                }
                self.encrypt_opts.push((k.to_string(), value.to_string()));
            }
        }
        Ok(true)
    }

    /// Builds the options from `-o` style pairs; unknown keys fail with QEMU's message.
    #[cfg(test)]
    pub(crate) fn parse<'a>(
        opts: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<AmendOptions> {
        let mut o = AmendOptions::default();
        for (k, v) in opts {
            if !o.set(k, v)? {
                return Err(Error::generic(format!("Invalid parameter '{k}'")));
            }
        }
        Ok(o)
    }
}

/// `Qcow2AmendOperation`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Operation {
    None,
    Upgrading,
    UpdatingEncryption,
    ChangingRefcountOrder,
    Downgrading,
}

/// `Qcow2AmendHelperCBInfo` and `qcow2_amend_helper_cb()`: turns the progress of each step
/// into the progress of the whole amend.
struct Progress<'a> {
    status: &'a mut dyn FnMut(u64, u64),
    current: Operation,
    total_operations: i64,
    operations_completed: i64,
    offset_completed: i64,
    last: Operation,
    last_work_size: i64,
}

impl Progress<'_> {
    fn cb(&mut self, operation_offset: u64, operation_work_size: u64) {
        if self.current != self.last {
            if self.last != Operation::None {
                self.offset_completed += self.last_work_size;
                self.operations_completed += 1;
            }
            self.last = self.current;
        }
        assert!(self.total_operations > 0);
        assert!(self.operations_completed < self.total_operations);
        self.last_work_size = operation_work_size as i64;
        let current_work_size = self.offset_completed + operation_work_size as i64;
        let projected = current_work_size * (self.total_operations - self.operations_completed - 1)
            / (self.operations_completed + 1);
        (self.status)(
            (self.offset_completed + operation_offset as i64) as u64,
            (current_work_size + projected) as u64,
        );
    }
}

fn update_header_err(e: io::Error) -> Error {
    err_errno("Failed to update the image header", e)
}

impl State {
    /// `qcow2_has_compressed_clusters()`.
    fn has_compressed_clusters(&mut self) -> io::Result<bool> {
        let mut offset = 0;
        let mut bytes = self.total_size.next_multiple_of(512);
        while bytes != 0 {
            let mut cur = bytes.min(i32::MAX as u64);
            let (_, ty) = self.get_host_offset(offset, &mut cur)?;
            if ty == SubclusterType::Compressed {
                return Ok(true);
            }
            offset += cur;
            bytes -= cur;
        }
        Ok(false)
    }

    /// `qcow2_downgrade()`: to version 2, the only older one.
    fn downgrade(&mut self, target_version: u32, status: &mut dyn FnMut(u64, u64)) -> Result<()> {
        let current_version = self.qcow_version;
        assert!(target_version < current_version);
        assert_eq!(target_version, 2);

        if self.refcount_order != 4 {
            return Err(Error::generic("compat=0.10 requires refcount_bits=16"));
        }
        if self.has_data_file() {
            return Err(Error::generic("Cannot downgrade an image with a data file"));
        }
        // Snapshots of another size, or with more VM state than 32 bits can say, need the
        // extra data v2 programs may not know is important.
        let disk_size = self.disk_size_sectors();
        if self
            .snapshots
            .iter()
            .any(|sn| sn.vm_state_size > u32::MAX as u64 || sn.disk_size != disk_size)
        {
            return Err(Error::generic("Internal snapshots prevent downgrade of image"));
        }

        if self.incompatible_features & QCOW2_INCOMPAT_DIRTY != 0 {
            self.mark_clean().map_err(|e| err_errno("Failed to make the image clean", e))?;
        }
        if self.incompatible_features & !QCOW2_INCOMPAT_COMPRESSION != 0 {
            return Err(Error::generic(format!(
                "Cannot downgrade an image with incompatible features {:#x} set",
                self.incompatible_features & !QCOW2_INCOMPAT_COMPRESSION
            )));
        }

        // Compatible features can be dropped, lazy refcounts were settled by cleaning above,
        // and autoclear features are trivial to clear.
        self.compatible_features = 0;
        self.autoclear_features = 0;

        self.expand_zero_clusters(status)
            .map_err(|e| err_errno("Failed to turn zero into data clusters", e))?;

        if self.incompatible_features & QCOW2_INCOMPAT_COMPRESSION != 0 {
            match self.has_compressed_clusters() {
                Err(_) => return Err(Error::generic("Failed to check block status")),
                Ok(true) => {
                    return Err(Error::generic(
                        "Cannot downgrade an image with zstd compression type and existing \
                         compressed clusters",
                    ));
                }
                Ok(false) => {}
            }
            // No compressed clusters, so the default zlib will do.
            self.incompatible_features &= !QCOW2_INCOMPAT_COMPRESSION;
            self.compression_type = QCOW2_COMPRESSION_TYPE_ZLIB;
        }
        assert_eq!(self.incompatible_features, 0);

        self.qcow_version = target_version;
        if let Err(e) = self.update_header() {
            self.qcow_version = current_version;
            return Err(update_header_err(e));
        }
        Ok(())
    }

    /// `qcow2_upgrade()`: to version 3.
    fn upgrade(&mut self, target_version: u32, status: &mut dyn FnMut(u64, u64)) -> Result<()> {
        let current_version = self.qcow_version;
        assert!(target_version > current_version);
        assert_eq!(target_version, 3);
        status(0, 2);

        // v3 needs the 64-bit VM state size and the disk size in every snapshot, and
        // write_snapshots() always writes them.
        let need_snapshot_update = self.snapshots.iter().any(|sn| sn.extra_data_size < 16);
        if need_snapshot_update {
            self.write_snapshots()
                .map_err(|e| err_errno("Failed to update the snapshot table", e))?;
        }
        status(1, 2);

        self.qcow_version = target_version;
        if let Err(e) = self.update_header() {
            self.qcow_version = current_version;
            return Err(update_header_err(e));
        }
        status(2, 2);
        Ok(())
    }

    /// Runs `Encryption::amend` on the LUKS header inside the image.
    fn amend_encryption(&mut self, opts: &EncryptionOptions, force: bool) -> Result<()> {
        let (offset, length) = self.crypto_header;
        let file = self.file.clone();
        let mut hio = ImageHeaderIo { file: &*file, offset, length };
        let crypto = self.crypto.as_mut().expect("image is encrypted");
        crypto.amend(&mut hio, opts, force)
    }

    /// `qcow2_amend_options()`. `status` gets the progress as (done, total) work units.
    pub(crate) fn amend_options(
        &mut self,
        o: &AmendOptions,
        status: &mut dyn FnMut(u64, u64),
        force: bool,
    ) -> Result<()> {
        let old_version = self.qcow_version;
        let mut new_version = old_version;
        let mut lazy_refcounts = self.use_lazy_refcounts;
        let mut data_file_raw = self.data_file_is_raw();
        let mut refcount_bits = self.refcount_bits as u64;
        let mut encryption_update = false;

        // In the order of qcow2_amend_opts: encrypt.*, then the common options.
        if !o.encrypt_opts.is_empty() {
            if self.crypto.is_none() {
                return Err(Error::generic(
                    "Can't amend encryption options - encryption not present",
                ));
            }
            if self.crypt_method_header != QCOW_CRYPT_LUKS {
                return Err(Error::generic("Only LUKS encryption options can be amended"));
            }
            encryption_update = true;
        }
        let new_size = o.size.unwrap_or(0);
        if let Some(compat) = &o.compat {
            new_version = match compat.as_str() {
                "0.10" | "v2" => 2,
                "1.1" | "v3" => 3,
                _ => return Err(Error::generic(format!("Unknown compatibility level {compat}"))),
            };
        }
        if o.data_file.is_some() && !self.has_data_file() {
            return Err(Error::generic(
                "data-file can only be set for images that use an external data file",
            ));
        }
        if let Some(raw) = o.data_file_raw {
            data_file_raw = raw;
            if data_file_raw && !self.data_file_is_raw() {
                return Err(Error::generic("data-file-raw cannot be set on existing images"));
            }
        }
        if let Some(l) = o.lazy_refcounts {
            lazy_refcounts = l;
        }
        if let Some(bits) = o.refcount_bits {
            // qemu_opt_get_number() goes through an int.
            let bits = bits as i32;
            if bits <= 0 || bits > 64 || !(bits as u32).is_power_of_two() {
                return Err(Error::generic(
                    "Refcount width must be a power of two and may not exceed 64 bits",
                ));
            }
            refcount_bits = bits as u64;
        }

        let mut progress = Progress {
            status,
            current: Operation::None,
            total_operations: (new_version != old_version) as i64
                + (self.refcount_bits as u64 != refcount_bits) as i64
                + encryption_update as i64,
            operations_completed: 0,
            offset_completed: 0,
            last: Operation::None,
            last_work_size: 0,
        };

        // Upgrade first, some features need compat=1.1.
        if new_version > old_version {
            progress.current = Operation::Upgrading;
            self.upgrade(new_version, &mut |a, b| progress.cb(a, b))?;
        }

        if encryption_update {
            progress.current = Operation::UpdatingEncryption;
            self.amend_encryption(&o.encrypt_opts, force)?;
        }

        if self.refcount_bits as u64 != refcount_bits {
            let refcount_order = refcount_bits.trailing_zeros();
            if new_version < 3 && refcount_bits != 16 {
                return Err(Error::generic(
                    "Refcount widths other than 16 bits require compatibility level 1.1 or \
                     above (use compat=1.1 or greater)",
                ));
            }
            progress.current = Operation::ChangingRefcountOrder;
            self.change_refcount_order(refcount_order, &mut |a, b| progress.cb(a, b))?;
        }

        // data-file-raw blocks backing files, so clear it first if requested.
        if data_file_raw {
            self.autoclear_features |= QCOW2_AUTOCLEAR_DATA_FILE_RAW;
        } else {
            self.autoclear_features &= !QCOW2_AUTOCLEAR_DATA_FILE_RAW;
        }
        if let Some(d) = &o.data_file {
            self.image_data_file = if d.is_empty() { None } else { Some(d.clone()) };
        }
        self.update_header().map_err(update_header_err)?;

        if (o.backing_file.is_some() || o.backing_fmt.is_some())
            && (o.backing_file != self.image_backing_file
                || o.backing_fmt != self.image_backing_format)
        {
            return Err(Error::generic("Cannot amend the backing file")
                .hint("You can use 'qemu-img rebase' instead.\n"));
        }

        if self.use_lazy_refcounts != lazy_refcounts {
            if lazy_refcounts {
                if new_version < 3 {
                    return Err(Error::generic(
                        "Lazy refcounts only supported with compatibility level 1.1 and above \
                         (use compat=1.1 or greater)",
                    ));
                }
                self.compatible_features |= QCOW2_COMPAT_LAZY_REFCOUNTS;
                if let Err(e) = self.update_header() {
                    self.compatible_features &= !QCOW2_COMPAT_LAZY_REFCOUNTS;
                    return Err(update_header_err(e));
                }
                self.use_lazy_refcounts = true;
            } else {
                self.mark_clean().map_err(|e| err_errno("Failed to make the image clean", e))?;
                self.compatible_features &= !QCOW2_COMPAT_LAZY_REFCOUNTS;
                if let Err(e) = self.update_header() {
                    self.compatible_features |= QCOW2_COMPAT_LAZY_REFCOUNTS;
                    return Err(update_header_err(e));
                }
                self.use_lazy_refcounts = false;
            }
        }

        if new_size != 0 {
            // Amending must give exactly the new size.
            self.truncate(new_size, true, Prealloc::Off)?;
        }

        // Downgrade last, once the features it cannot keep are gone.
        if new_version < old_version {
            progress.current = Operation::Downgrading;
            self.downgrade(new_version, &mut |a, b| progress.cb(a, b))?;
        }
        Ok(())
    }

    /// `qcow2_co_amend()`: `x-blockdev-amend`, which can only change LUKS key slots.
    /// `format_is_luks` is whether the options are for the `luks` format.
    pub(crate) fn blockdev_amend(
        &mut self,
        encrypt: Option<(bool, &EncryptionOptions)>,
        force: bool,
    ) -> Result<()> {
        let Some((format_is_luks, opts)) = encrypt else {
            return Ok(());
        };
        if self.crypto.is_none() {
            return Err(Error::generic("image is not encrypted, can't amend"));
        }
        if !format_is_luks {
            return Err(Error::generic(
                "Amend can't be used to change the qcow2 encryption format",
            ));
        }
        if self.crypt_method_header != QCOW_CRYPT_LUKS {
            return Err(Error::generic(
                "Only LUKS encryption options can be amended for qcow2 with blockdev-amend",
            ));
        }
        self.amend_encryption(opts, force)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_options() {
        let o = AmendOptions::parse([
            ("compat", "1.1"),
            ("refcount_bits", "64"),
            ("encrypt.keyslot", "1"),
        ])
        .unwrap();
        assert_eq!(o.compat.as_deref(), Some("1.1"));
        assert_eq!(o.refcount_bits, Some(64));
        assert_eq!(o.encrypt_opts, vec![("keyslot".to_string(), "1".to_string())]);
        let e = AmendOptions::parse([("cluster_size", "4k")]).unwrap_err();
        assert_eq!(e.message(), "Invalid parameter 'cluster_size'");
        let e = AmendOptions::parse([("encrypt.format", "luks")]).unwrap_err();
        assert_eq!(e.message(), "Invalid parameter 'encrypt.format'");
    }

    #[test]
    fn progress_projection() {
        let mut seen = Vec::new();
        let mut f = |a, b| seen.push((a, b));
        let mut p = Progress {
            status: &mut f,
            current: Operation::Upgrading,
            total_operations: 2,
            operations_completed: 0,
            offset_completed: 0,
            last: Operation::None,
            last_work_size: 0,
        };
        p.cb(1, 2);
        p.current = Operation::ChangingRefcountOrder;
        p.cb(5, 10);
        // One operation of two: the total is projected to twice the first.
        assert_eq!(seen, vec![(1, 4), (7, 12)]);
    }
}
