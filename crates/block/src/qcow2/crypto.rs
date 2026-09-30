// SPDX-License-Identifier: GPL-2.0-or-later

//! The seam between qcow2 and disk encryption.
//!
//! qcow2 knows two encryption methods: the legacy AES-CBC of `encrypt.format=aes` (crypt method
//! 1, no header, IVs from the guest offset) and LUKS (crypt method 2, a LUKS header stored in
//! clusters of the image and pointed to by the full disk encryption header extension, IVs from
//! the host offset). The ciphers themselves live in ruvm-crypto. qcow2 reaches them through
//! [`EncryptionProvider`], passed in `OpenOptions::encryption` or `CreateOptions::encryption`,
//! and calls it the way block/qcow2.c calls `qcrypto_block_open()` and
//! `qcrypto_block_create()`. [`CryptoProvider`] is the provider backed by ruvm-crypto's
//! `QCryptoBlock`, the one the driver uses; tests can pass their own.

use std::fmt;

use ruvm_base::Result;
use ruvm_qapi::types::QCryptoBlockInfoLUKS;

/// The `encrypt.format` values qcow2 accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EncryptionFormat {
    /// `aes`: legacy AES-CBC with plain64 style IVs from the guest offset.
    Aes,
    /// `luks`.
    Luks,
}

impl EncryptionFormat {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            EncryptionFormat::Aes => "aes",
            EncryptionFormat::Luks => "luks",
        }
    }

    pub(crate) fn parse(s: &str) -> Option<EncryptionFormat> {
        match s {
            "aes" => Some(EncryptionFormat::Aes),
            "luks" => Some(EncryptionFormat::Luks),
            _ => None,
        }
    }
}

/// An open encryption context, `QCryptoBlock`.
pub(crate) trait Encryption: Send + Sync + fmt::Debug {
    /// `qcrypto_block_get_sector_size()`. Requests are aligned to this.
    fn sector_size(&self) -> u64;

    /// `qcrypto_block_encrypt()`: encrypts `buf` in place. `offset` is the byte offset the IVs
    /// are derived from (the host offset for LUKS, the guest offset for AES).
    fn encrypt(&self, offset: u64, buf: &mut [u8]) -> Result<()>;

    /// `qcrypto_block_decrypt()`.
    fn decrypt(&self, offset: u64, buf: &mut [u8]) -> Result<()>;

    /// `qcrypto_block_get_info()` for LUKS, the details under `encrypt` in the format specific
    /// information. `None` for AES, which has none.
    fn luks_info(&self) -> Option<QCryptoBlockInfoLUKS> {
        None
    }

    /// `qcrypto_block_amend_options()`: changes LUKS key slots from the `encrypt.*` amend
    /// options (`state`, `keyslot`, `old-secret`, `new-secret`, `iter-time`).
    fn amend(
        &mut self,
        header: &mut dyn HeaderIo,
        opts: &EncryptionOptions,
        force: bool,
    ) -> Result<()> {
        let _ = (header, opts, force);
        Err(ruvm_base::Error::generic("This encryption does not support amending"))
    }
}

/// Access to the LUKS header area inside the image, the read, init and write callbacks of
/// `qcrypto_block_open()` and `qcrypto_block_create()`.
pub(crate) trait HeaderIo {
    /// Reads `buf.len()` bytes at `offset` into the header area.
    fn read(&mut self, offset: u64, buf: &mut [u8]) -> Result<()>;
    /// Allocates a zeroed header area of `len` bytes. Only called while creating.
    fn init(&mut self, len: u64) -> Result<()>;
    /// Writes `buf` at `offset` into the header area.
    fn write(&mut self, offset: u64, buf: &[u8]) -> Result<()>;
}

/// The `encrypt.*` options, without the prefix, as given by the user.
pub(crate) type EncryptionOptions = Vec<(String, String)>;

/// Creates encryption contexts. `no_io` is `QCRYPTO_BLOCK_OPEN_NO_IO`: the header is parsed but
/// no key is needed and nothing will be encrypted.
pub(crate) trait EncryptionProvider: Send + Sync + fmt::Debug {
    fn open(
        &self,
        format: EncryptionFormat,
        opts: &EncryptionOptions,
        header: Option<&mut dyn HeaderIo>,
        no_io: bool,
    ) -> Result<Box<dyn Encryption>>;

    fn create(
        &self,
        format: EncryptionFormat,
        opts: &EncryptionOptions,
        header: Option<&mut dyn HeaderIo>,
    ) -> Result<Box<dyn Encryption>>;

    /// `qcrypto_block_calculate_payload_offset()`: the length of the header a new volume made
    /// with `opts` gets, for `qemu-img measure`.
    fn header_len(&self, format: EncryptionFormat, opts: &EncryptionOptions) -> Result<u64> {
        let _ = opts;
        Err(ruvm_base::Error::generic(format!(
            "Cannot measure the {} header without a crypto provider that knows its size",
            format.as_str()
        )))
    }
}

/// The ruvm-crypto implementation of the seam.
#[derive(Debug, Default)]
pub(crate) struct CryptoProvider;

/// A `QCryptoBlock` behind the [`Encryption`] trait.
#[derive(Debug)]
struct CryptoBlock(ruvm_crypto::block::QCryptoBlock);

/// Adapts [`HeaderIo`] to ruvm-crypto's header callbacks.
struct IoAdapter<'a>(Option<&'a mut dyn HeaderIo>);

impl ruvm_crypto::block::QCryptoBlockIo for IoAdapter<'_> {
    fn read(&mut self, offset: u64, buf: &mut [u8]) -> Result<()> {
        match &mut self.0 {
            Some(h) => h.read(offset, buf),
            None => Err(ruvm_base::Error::generic("Request for data outside of extension header")),
        }
    }

    fn write(&mut self, offset: u64, buf: &[u8]) -> Result<()> {
        match &mut self.0 {
            Some(h) => h.write(offset, buf),
            None => Err(ruvm_base::Error::generic("Request for data outside of extension header")),
        }
    }

    fn init(&mut self, headerlen: u64) -> Result<()> {
        match &mut self.0 {
            Some(h) => h.init(headerlen),
            None => Ok(()),
        }
    }
}

fn opts_dict(opts: &EncryptionOptions) -> ruvm_qapi::QDict {
    let mut d = ruvm_qapi::QDict::new();
    for (k, v) in opts {
        d.put(k.clone(), ruvm_qapi::QValue::str(v.clone()));
    }
    d
}

fn key_secret(opts: &EncryptionOptions) -> Option<String> {
    opts.iter().rev().find(|(k, _)| k == "key-secret").map(|(_, v)| v.clone())
}

fn create_options(
    format: EncryptionFormat,
    opts: &EncryptionOptions,
) -> Result<ruvm_qapi::types::QCryptoBlockCreateOptions> {
    let fmt = match format {
        EncryptionFormat::Aes => "qcow",
        EncryptionFormat::Luks => "luks",
    };
    crate::luks::create_opts_init(&mut opts_dict(opts), fmt)
}

impl EncryptionProvider for CryptoProvider {
    fn open(
        &self,
        format: EncryptionFormat,
        opts: &EncryptionOptions,
        header: Option<&mut dyn HeaderIo>,
        no_io: bool,
    ) -> Result<Box<dyn Encryption>> {
        use ruvm_qapi::types::{
            QCryptoBlockOpenOptions, QCryptoBlockOpenOptionsU, QCryptoBlockOptionsLUKS,
            QCryptoBlockOptionsQCow,
        };
        let key_secret = key_secret(opts);
        let u = match format {
            EncryptionFormat::Aes => {
                QCryptoBlockOpenOptionsU::Qcow(QCryptoBlockOptionsQCow { key_secret })
            }
            EncryptionFormat::Luks => {
                QCryptoBlockOpenOptionsU::Luks(QCryptoBlockOptionsLUKS { key_secret })
            }
        };
        let flags = if no_io { ruvm_crypto::block::QCRYPTO_BLOCK_OPEN_NO_IO } else { 0 };
        let mut io = IoAdapter(header);
        let b = ruvm_crypto::block::QCryptoBlock::open(
            &QCryptoBlockOpenOptions { u },
            Some("encrypt."),
            &mut io,
            flags,
        )?;
        Ok(Box::new(CryptoBlock(b)))
    }

    fn create(
        &self,
        format: EncryptionFormat,
        opts: &EncryptionOptions,
        header: Option<&mut dyn HeaderIo>,
    ) -> Result<Box<dyn Encryption>> {
        let o = create_options(format, opts)?;
        let mut io = IoAdapter(header);
        let b = ruvm_crypto::block::QCryptoBlock::create(&o, Some("encrypt."), &mut io, 0)?;
        Ok(Box::new(CryptoBlock(b)))
    }

    fn header_len(&self, format: EncryptionFormat, opts: &EncryptionOptions) -> Result<u64> {
        let o = create_options(format, opts)?;
        ruvm_crypto::block::QCryptoBlock::calculate_payload_offset(&o, Some("encrypt."))
    }
}

impl Encryption for CryptoBlock {
    fn sector_size(&self) -> u64 {
        self.0.sector_size()
    }

    fn encrypt(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.0.encrypt(offset, buf)
    }

    fn decrypt(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.0.decrypt(offset, buf)
    }

    fn luks_info(&self) -> Option<QCryptoBlockInfoLUKS> {
        match self.0.get_info().ok()?.u {
            ruvm_qapi::types::QCryptoBlockInfoU::Luks(l) => Some(l),
            ruvm_qapi::types::QCryptoBlockInfoU::Qcow => None,
        }
    }

    fn amend(
        &mut self,
        header: &mut dyn HeaderIo,
        opts: &EncryptionOptions,
        force: bool,
    ) -> Result<()> {
        use ruvm_qapi::types::{QCryptoBlockAmendOptions, QCryptoBlockAmendOptionsU};
        use ruvm_qapi::visit::Visit;
        let mut d = opts_dict(opts);
        let fmt = match self.0.format() {
            ruvm_qapi::types::QCryptoBlockFormat::Luks => "luks",
            ruvm_qapi::types::QCryptoBlockFormat::Qcow => "qcow",
        };
        let o = if fmt == "luks" {
            d.put("format", ruvm_qapi::QValue::str(fmt));
            let mut v =
                ruvm_qapi::visit::QObjectInputVisitor::new_keyval(ruvm_qapi::QValue::Dict(d));
            let mut o = QCryptoBlockAmendOptions::default();
            QCryptoBlockAmendOptions::visit(&mut v, None, &mut o)?;
            o
        } else {
            QCryptoBlockAmendOptions { u: QCryptoBlockAmendOptionsU::Qcow }
        };
        let mut io = IoAdapter(Some(header));
        self.0.amend_options(&mut io, &o, force)
    }
}
