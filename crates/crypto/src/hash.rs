// SPDX-License-Identifier: GPL-2.0-or-later

//! Hash algorithms, from crypto/hash.c.
//!
//! Every algorithm in `QCryptoHashAlgo` is supported, including SM3, which QEMU only has when
//! it is built with a library that provides it.

use digest::DynDigest;
use ruvm_base::{Error, Result};
use ruvm_qapi::types::QCryptoHashAlgo;

use crate::base64;

/// `qcrypto_hash_supports()`.
pub fn hash_supports(_alg: QCryptoHashAlgo) -> bool {
    true
}

/// `qcrypto_hash_digest_len()`.
pub fn hash_digest_len(alg: QCryptoHashAlgo) -> usize {
    match alg {
        QCryptoHashAlgo::Md5 => 16,
        QCryptoHashAlgo::Sha1 | QCryptoHashAlgo::Ripemd160 => 20,
        QCryptoHashAlgo::Sha224 => 28,
        QCryptoHashAlgo::Sha256 | QCryptoHashAlgo::Sm3 => 32,
        QCryptoHashAlgo::Sha384 => 48,
        QCryptoHashAlgo::Sha512 => 64,
    }
}

/// A hash in progress, `QCryptoHash`.
pub struct Hash {
    alg: QCryptoHashAlgo,
    ctx: Box<dyn DynDigest + Send + Sync>,
}

impl std::fmt::Debug for Hash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hash").field("alg", &self.alg).finish_non_exhaustive()
    }
}

pub(crate) fn new_digest(alg: QCryptoHashAlgo) -> Box<dyn DynDigest + Send + Sync> {
    match alg {
        QCryptoHashAlgo::Md5 => Box::new(md5::Md5::default()),
        QCryptoHashAlgo::Sha1 => Box::new(sha1::Sha1::default()),
        QCryptoHashAlgo::Sha224 => Box::new(sha2::Sha224::default()),
        QCryptoHashAlgo::Sha256 => Box::new(sha2::Sha256::default()),
        QCryptoHashAlgo::Sha384 => Box::new(sha2::Sha384::default()),
        QCryptoHashAlgo::Sha512 => Box::new(sha2::Sha512::default()),
        QCryptoHashAlgo::Ripemd160 => Box::new(ripemd::Ripemd160::default()),
        QCryptoHashAlgo::Sm3 => Box::new(sm3::Sm3::default()),
    }
}

impl Hash {
    /// `qcrypto_hash_new()`.
    pub fn new(alg: QCryptoHashAlgo) -> Result<Hash> {
        if !hash_supports(alg) {
            return Err(Error::generic(format!("Unsupported hash algorithm {}", alg.as_str())));
        }
        Ok(Hash { alg, ctx: new_digest(alg) })
    }

    pub fn alg(&self) -> QCryptoHashAlgo {
        self.alg
    }

    /// `qcrypto_hash_update()`.
    pub fn update(&mut self, buf: &[u8]) {
        self.ctx.update(buf);
    }

    /// `qcrypto_hash_updatev()`.
    pub fn updatev(&mut self, iov: &[&[u8]]) {
        for b in iov {
            self.ctx.update(b);
        }
    }

    /// `qcrypto_hash_finalize_bytes()`.
    pub fn finalize_bytes(self) -> Vec<u8> {
        self.ctx.finalize().into_vec()
    }

    /// `qcrypto_hash_finalize_digest()`: the result in lower case hex.
    pub fn finalize_digest(self) -> String {
        to_hex(&self.finalize_bytes())
    }

    /// `qcrypto_hash_finalize_base64()`.
    pub fn finalize_base64(self) -> String {
        base64::encode(&self.finalize_bytes())
    }
}

pub(crate) fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[usize::from(b >> 4)] as char);
        s.push(HEX[usize::from(b & 0xf)] as char);
    }
    s
}

/// `qcrypto_hash_bytesv()`.
pub fn hash_bytesv(alg: QCryptoHashAlgo, iov: &[&[u8]]) -> Result<Vec<u8>> {
    let mut h = Hash::new(alg)?;
    h.updatev(iov);
    Ok(h.finalize_bytes())
}

/// `qcrypto_hash_bytes()`.
pub fn hash_bytes(alg: QCryptoHashAlgo, buf: &[u8]) -> Result<Vec<u8>> {
    hash_bytesv(alg, &[buf])
}

/// `qcrypto_hash_digestv()`.
pub fn hash_digestv(alg: QCryptoHashAlgo, iov: &[&[u8]]) -> Result<String> {
    let mut h = Hash::new(alg)?;
    h.updatev(iov);
    Ok(h.finalize_digest())
}

/// `qcrypto_hash_digest()`.
pub fn hash_digest(alg: QCryptoHashAlgo, buf: &[u8]) -> Result<String> {
    hash_digestv(alg, &[buf])
}

/// `qcrypto_hash_base64v()`.
pub fn hash_base64v(alg: QCryptoHashAlgo, iov: &[&[u8]]) -> Result<String> {
    let mut h = Hash::new(alg)?;
    h.updatev(iov);
    Ok(h.finalize_base64())
}

/// `qcrypto_hash_base64()`.
pub fn hash_base64(alg: QCryptoHashAlgo, buf: &[u8]) -> Result<String> {
    hash_base64v(alg, &[buf])
}
