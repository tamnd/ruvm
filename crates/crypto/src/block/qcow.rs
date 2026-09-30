// SPDX-License-Identifier: GPL-2.0-or-later

//! The legacy qcow and qcow2 AES encryption, ported from QEMU's `crypto/block-qcow.c`.
//!
//! The key is the password bytes, truncated or zero padded to 16 bytes, used directly as an
//! AES-128 key in CBC mode with a plain64 IV. There is no header, so nothing is read or written
//! and the payload offset is zero. This scheme is weak; QEMU only supports it for existing
//! images and so does ruvm.

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{
    QCryptoBlockOptionsQCow, QCryptoCipherAlgo, QCryptoCipherMode, QCryptoHashAlgo,
    QCryptoIVGenAlgo,
};

use super::QCryptoBlock;
use crate::cipher::cipher_get_iv_len;
use crate::ivgen::IvGen;
use crate::secret::secret_lookup_as_utf8;

/// `QCRYPTO_BLOCK_QCOW_SECTOR_SIZE`.
pub const SECTOR_SIZE: u64 = 512;

/// `qcrypto_block_qcow_has_format()`: qcow encryption has no header to recognize.
pub fn has_format(_buf: &[u8]) -> bool {
    false
}

fn init(block: &mut QCryptoBlock, keysecret: &str) -> Result<()> {
    let password = secret_lookup_as_utf8(keysecret)?;
    let mut keybuf = [0u8; 16];
    let n = password.len().min(16);
    keybuf[..n].copy_from_slice(&password.as_bytes()[..n]);

    block.niv = cipher_get_iv_len(QCryptoCipherAlgo::Aes128, QCryptoCipherMode::Cbc);
    block.ivgen = Some(IvGen::new(
        QCryptoIVGenAlgo::Plain64,
        QCryptoCipherAlgo::Aes128,
        QCryptoHashAlgo::Md5,
        &[],
    )?);
    let r = block.init_cipher(QCryptoCipherAlgo::Aes128, QCryptoCipherMode::Cbc, &keybuf);
    keybuf.fill(0);
    if let Err(e) = r {
        block.ivgen = None;
        return Err(e);
    }
    block.sector_size = SECTOR_SIZE;
    block.payload_offset = 0;
    Ok(())
}

fn missing_secret(optprefix: &str) -> Error {
    Error::generic(format!("Parameter '{optprefix}key-secret' is required for cipher"))
}

/// `qcrypto_block_qcow_open()`.
pub(super) fn open(
    block: &mut QCryptoBlock,
    options: &QCryptoBlockOptionsQCow,
    optprefix: &str,
    flags: u32,
) -> Result<()> {
    if flags & super::QCRYPTO_BLOCK_OPEN_NO_IO != 0 {
        block.sector_size = SECTOR_SIZE;
        block.payload_offset = 0;
        return Ok(());
    }
    let Some(secret) = &options.key_secret else {
        return Err(missing_secret(optprefix));
    };
    init(block, secret)
}

/// `qcrypto_block_qcow_create()`: the same as opening, since everything is hardwired.
pub(super) fn create(
    block: &mut QCryptoBlock,
    options: &QCryptoBlockOptionsQCow,
    optprefix: &str,
) -> Result<()> {
    let Some(secret) = &options.key_secret else {
        return Err(missing_secret(optprefix));
    };
    init(block, secret)
}
