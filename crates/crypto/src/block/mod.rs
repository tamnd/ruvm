// SPDX-License-Identifier: GPL-2.0-or-later

//! `QCryptoBlock`: the full disk encryption layer shared by the `luks` block driver and the
//! qcow and qcow2 formats, ported from QEMU's `crypto/block.c`.
//!
//! A [`QCryptoBlock`] is opened or created from the QAPI `QCryptoBlock*Options` types. Header
//! I/O goes through a [`QCryptoBlockIo`] implementation supplied by the caller, which stands in
//! for QEMU's read, write and init function pointers plus their `opaque` argument. Once open,
//! [`QCryptoBlock::encrypt`] and [`QCryptoBlock::decrypt`] work in place on whole sectors; the
//! offset is the byte offset of the data within the encrypted payload, which is what selects the
//! IV.
//!
//! Differences from QEMU:
//!
//! * The driver table is a `match` on the format; there is no function pointer vtable.
//! * The pool of free ciphers is a `Mutex<Vec<Cipher>>`. A cipher is popped for each request and
//!   pushed back afterwards, as in QEMU, so concurrent requests do not serialize on one cipher.
//! * The ESSIV generator carries its own lock, so the block mutex is not taken around
//!   `qcrypto_ivgen_calculate()`.
//! * Header callbacks return `Result<()>` rather than a negative errno.

pub mod luks;
pub mod qcow;

use std::sync::Mutex;

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{
    QCryptoBlockAmendOptions, QCryptoBlockAmendOptionsU, QCryptoBlockCreateOptions,
    QCryptoBlockCreateOptionsU, QCryptoBlockFormat, QCryptoBlockInfo, QCryptoBlockInfoU,
    QCryptoBlockOpenOptions, QCryptoBlockOpenOptionsU, QCryptoCipherAlgo, QCryptoCipherMode,
    QCryptoHashAlgo,
};

use crate::cipher::Cipher;
use crate::ivgen::IvGen;

/// `QCRYPTO_BLOCK_OPEN_NO_IO`: only parse the header, do not unlock the master key.
pub const QCRYPTO_BLOCK_OPEN_NO_IO: u32 = 1 << 0;
/// `QCRYPTO_BLOCK_OPEN_DETACHED`: the header lives apart from the payload, so skip the payload
/// overlap checks.
pub const QCRYPTO_BLOCK_OPEN_DETACHED: u32 = 1 << 1;
/// `QCRYPTO_BLOCK_CREATE_DETACHED`: create a header with a payload offset of zero.
pub const QCRYPTO_BLOCK_CREATE_DETACHED: u32 = 1 << 0;

/// Header access for [`QCryptoBlock`]. This is QEMU's `QCryptoBlockReadFunc`,
/// `QCryptoBlockWriteFunc` and `QCryptoBlockInitFunc` together with their `opaque` pointer.
pub trait QCryptoBlockIo {
    /// Reads `buf.len()` bytes of header at byte `offset`.
    fn read(&mut self, offset: u64, buf: &mut [u8]) -> Result<()>;
    /// Writes `buf` into the header at byte `offset`.
    fn write(&mut self, offset: u64, buf: &[u8]) -> Result<()>;
    /// Called once during creation with the size of the header, before anything is written.
    fn init(&mut self, headerlen: u64) -> Result<()> {
        let _ = headerlen;
        Ok(())
    }
}

/// The per-format state.
#[derive(Debug)]
enum Format {
    Qcow,
    Luks(Box<luks::Luks>),
}

/// The parameters from which ciphers are made (`block->alg`, `block->mode`, `block->key`).
struct CipherParams {
    alg: QCryptoCipherAlgo,
    mode: QCryptoCipherMode,
    key: Vec<u8>,
}

impl Drop for CipherParams {
    fn drop(&mut self) {
        self.key.fill(0);
    }
}

/// `QCryptoBlock`: an opened or freshly created encryption header.
pub struct QCryptoBlock {
    format: Format,
    params: Option<CipherParams>,
    free_ciphers: Mutex<Vec<Cipher>>,
    ivgen: Option<IvGen>,
    kdfhash: QCryptoHashAlgo,
    niv: usize,
    payload_offset: u64,
    sector_size: u64,
    detached_header: bool,
}

impl std::fmt::Debug for QCryptoBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QCryptoBlock")
            .field("format", &self.format)
            .field("payload_offset", &self.payload_offset)
            .field("sector_size", &self.sector_size)
            .field("detached_header", &self.detached_header)
            .finish_non_exhaustive()
    }
}

/// `qcrypto_block_has_format()`: whether `buf`, the start of an image, holds a header of `format`.
pub fn has_format(format: QCryptoBlockFormat, buf: &[u8]) -> bool {
    match format {
        QCryptoBlockFormat::Qcow => qcow::has_format(buf),
        QCryptoBlockFormat::Luks => luks::has_format(buf),
    }
}

impl QCryptoBlock {
    fn empty(format: Format) -> QCryptoBlock {
        QCryptoBlock {
            format,
            params: None,
            free_ciphers: Mutex::new(Vec::new()),
            ivgen: None,
            kdfhash: QCryptoHashAlgo::Md5,
            niv: 0,
            payload_offset: 0,
            sector_size: 0,
            detached_header: false,
        }
    }

    /// `qcrypto_block_open()`. `optprefix` goes in front of option names in error messages,
    /// for example `encrypt.` for qcow2.
    pub fn open(
        options: &QCryptoBlockOpenOptions,
        optprefix: Option<&str>,
        io: &mut dyn QCryptoBlockIo,
        flags: u32,
    ) -> Result<QCryptoBlock> {
        let optprefix = optprefix.unwrap_or("");
        match &options.u {
            QCryptoBlockOpenOptionsU::Qcow(o) => {
                let mut block = QCryptoBlock::empty(Format::Qcow);
                qcow::open(&mut block, o, optprefix, flags)?;
                Ok(block)
            }
            QCryptoBlockOpenOptionsU::Luks(o) => {
                let mut block = QCryptoBlock::empty(Format::Qcow);
                let l = luks::open(&mut block, o, optprefix, io, flags)?;
                block.format = Format::Luks(Box::new(l));
                Ok(block)
            }
        }
    }

    /// `qcrypto_block_create()`.
    pub fn create(
        options: &QCryptoBlockCreateOptions,
        optprefix: Option<&str>,
        io: &mut dyn QCryptoBlockIo,
        flags: u32,
    ) -> Result<QCryptoBlock> {
        let optprefix = optprefix.unwrap_or("");
        match &options.u {
            QCryptoBlockCreateOptionsU::Qcow(o) => {
                let mut block = QCryptoBlock::empty(Format::Qcow);
                block.detached_header = flags & QCRYPTO_BLOCK_CREATE_DETACHED != 0;
                qcow::create(&mut block, o, optprefix)?;
                Ok(block)
            }
            QCryptoBlockCreateOptionsU::Luks(o) => {
                let mut block = QCryptoBlock::empty(Format::Qcow);
                block.detached_header = flags & QCRYPTO_BLOCK_CREATE_DETACHED != 0;
                let l = luks::create(&mut block, o, optprefix, io)?;
                block.format = Format::Luks(Box::new(l));
                Ok(block)
            }
        }
    }

    /// `qcrypto_block_calculate_payload_offset()`: the header size a create with these options
    /// would produce, found by running the creation without writing anything.
    pub fn calculate_payload_offset(
        options: &QCryptoBlockCreateOptions,
        optprefix: Option<&str>,
    ) -> Result<u64> {
        struct HeaderLen(u64);
        impl QCryptoBlockIo for HeaderLen {
            fn read(&mut self, _offset: u64, _buf: &mut [u8]) -> Result<()> {
                Ok(())
            }
            fn write(&mut self, _offset: u64, _buf: &[u8]) -> Result<()> {
                Ok(())
            }
            fn init(&mut self, headerlen: u64) -> Result<()> {
                self.0 = headerlen;
                Ok(())
            }
        }
        let mut hl = HeaderLen(0);
        QCryptoBlock::create(options, optprefix, &mut hl, 0)?;
        Ok(hl.0)
    }

    /// `qcrypto_block_amend_options()`: adds or erases LUKS keyslots.
    pub fn amend_options(
        &mut self,
        io: &mut dyn QCryptoBlockIo,
        options: &QCryptoBlockAmendOptions,
        force: bool,
    ) -> Result<()> {
        let fmt = self.format();
        let same = matches!(
            (&options.u, fmt),
            (QCryptoBlockAmendOptionsU::Luks(_), QCryptoBlockFormat::Luks)
                | (QCryptoBlockAmendOptionsU::Qcow, QCryptoBlockFormat::Qcow)
        );
        if !same {
            return Err(Error::generic("Cannot amend encryption format"));
        }
        match &options.u {
            QCryptoBlockAmendOptionsU::Luks(o) => luks::amend(self, io, o, force),
            QCryptoBlockAmendOptionsU::Qcow => Err(Error::generic(format!(
                "Crypto format {} doesn't support format options amendment",
                fmt.as_str()
            ))),
        }
    }

    /// `qcrypto_block_get_info()`.
    pub fn get_info(&self) -> Result<QCryptoBlockInfo> {
        let u = match &self.format {
            Format::Qcow => QCryptoBlockInfoU::Qcow,
            Format::Luks(l) => QCryptoBlockInfoU::Luks(luks::get_info(self, l)),
        };
        Ok(QCryptoBlockInfo { u })
    }

    /// The format of this header.
    pub fn format(&self) -> QCryptoBlockFormat {
        match self.format {
            Format::Qcow => QCryptoBlockFormat::Qcow,
            Format::Luks(_) => QCryptoBlockFormat::Luks,
        }
    }

    /// `qcrypto_block_decrypt()`: decrypts `buf` in place. `offset` is the byte offset of the
    /// data within the payload; both it and `buf.len()` must be multiples of the sector size.
    pub fn decrypt(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.encdec(offset, buf, false)
    }

    /// `qcrypto_block_encrypt()`: encrypts `buf` in place, with the same rules as
    /// [`QCryptoBlock::decrypt`].
    pub fn encrypt(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.encdec(offset, buf, true)
    }

    fn encdec(&self, offset: u64, buf: &mut [u8], enc: bool) -> Result<()> {
        // Both formats use 512 byte sectors.
        let ss = match &self.format {
            Format::Qcow => qcow::SECTOR_SIZE,
            Format::Luks(_) => luks::SECTOR_SIZE,
        };
        assert!(offset % ss == 0);
        assert!(buf.len() as u64 % ss == 0);
        let mut cipher = self.pop_cipher()?;
        let r = cipher_encdec(
            &mut cipher,
            self.niv,
            self.ivgen.as_ref(),
            ss as usize,
            offset,
            buf,
            enc,
        );
        self.push_cipher(cipher);
        r
    }

    /// `qcrypto_block_get_kdf_hash()`.
    pub fn kdf_hash(&self) -> QCryptoHashAlgo {
        self.kdfhash
    }

    /// `qcrypto_block_get_payload_offset()`: where the encrypted data starts, in bytes.
    pub fn payload_offset(&self) -> u64 {
        self.payload_offset
    }

    /// `qcrypto_block_get_sector_size()`.
    pub fn sector_size(&self) -> u64 {
        self.sector_size
    }

    /// Whether the header was created or opened as a detached header.
    pub fn detached_header(&self) -> bool {
        self.detached_header
    }

    /// `qcrypto_block_get_ivgen()`, for tests.
    pub fn ivgen(&self) -> Option<&IvGen> {
        self.ivgen.as_ref()
    }

    fn pop_cipher(&self) -> Result<Cipher> {
        if let Some(c) = self.free_ciphers.lock().unwrap_or_else(|e| e.into_inner()).pop() {
            return Ok(c);
        }
        let p = self.params.as_ref().expect("the block was opened for I/O");
        Cipher::new(p.alg, p.mode, &p.key)
    }

    fn push_cipher(&self, cipher: Cipher) {
        self.free_ciphers.lock().unwrap_or_else(|e| e.into_inner()).push(cipher);
    }

    /// `qcrypto_block_init_cipher()`.
    fn init_cipher(
        &mut self,
        alg: QCryptoCipherAlgo,
        mode: QCryptoCipherMode,
        key: &[u8],
    ) -> Result<()> {
        // Make one cipher now to validate the parameters, which makes failure at I/O time
        // unlikely.
        let c = Cipher::new(alg, mode, key)?;
        self.params = Some(CipherParams { alg, mode, key: key.to_vec() });
        self.push_cipher(c);
        Ok(())
    }
}

/// `do_qcrypto_block_cipher_encdec()`: runs the cipher over `buf` one sector at a time, setting
/// the IV for each sector from `ivgen`.
pub(crate) fn cipher_encdec(
    cipher: &mut Cipher,
    niv: usize,
    ivgen: Option<&IvGen>,
    sectorsize: usize,
    offset: u64,
    buf: &mut [u8],
    enc: bool,
) -> Result<()> {
    assert!(offset % sectorsize as u64 == 0);
    assert!(buf.len() % sectorsize == 0);
    let mut iv = vec![0u8; niv];
    let first = offset / sectorsize as u64;
    for (startsector, chunk) in (first..).zip(buf.chunks_mut(sectorsize)) {
        if niv != 0 {
            ivgen
                .expect("an IV generator when the mode needs an IV")
                .calculate(startsector, &mut iv)?;
            cipher.setiv(&iv)?;
        }
        if enc {
            cipher.encrypt(chunk)?;
        } else {
            cipher.decrypt(chunk)?;
        }
    }
    Ok(())
}
