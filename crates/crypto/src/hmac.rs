// SPDX-License-Identifier: GPL-2.0-or-later

//! Keyed hashes, from crypto/hmac.c.
//!
//! Like QEMU's nettle and gcrypt backends, a [`Hmac`] keeps its key after each call, so it can
//! be used for many messages.

use digest::KeyInit;
use digest::Mac;
use ruvm_base::Result;
use ruvm_qapi::types::QCryptoHashAlgo;

use crate::hash::to_hex;

enum Inner {
    Md5(hmac::Hmac<md5::Md5>),
    Sha1(hmac::Hmac<sha1::Sha1>),
    Sha224(hmac::Hmac<sha2::Sha224>),
    Sha256(hmac::Hmac<sha2::Sha256>),
    Sha384(hmac::Hmac<sha2::Sha384>),
    Sha512(hmac::Hmac<sha2::Sha512>),
    Ripemd160(hmac::Hmac<ripemd::Ripemd160>),
    Sm3(hmac::Hmac<sm3::Sm3>),
}

/// `QCryptoHmac`.
pub struct Hmac {
    alg: QCryptoHashAlgo,
    inner: Inner,
}

impl std::fmt::Debug for Hmac {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hmac").field("alg", &self.alg).finish_non_exhaustive()
    }
}

/// `qcrypto_hmac_supports()`.
pub fn hmac_supports(_alg: QCryptoHashAlgo) -> bool {
    true
}

macro_rules! each {
    ($inner:expr, $m:ident => $body:expr) => {
        match $inner {
            Inner::Md5($m) => $body,
            Inner::Sha1($m) => $body,
            Inner::Sha224($m) => $body,
            Inner::Sha256($m) => $body,
            Inner::Sha384($m) => $body,
            Inner::Sha512($m) => $body,
            Inner::Ripemd160($m) => $body,
            Inner::Sm3($m) => $body,
        }
    };
}

impl Hmac {
    /// `qcrypto_hmac_new()`.
    pub fn new(alg: QCryptoHashAlgo, key: &[u8]) -> Result<Hmac> {
        // HMAC takes keys of any length, so new_from_slice cannot fail.
        let inner = match alg {
            QCryptoHashAlgo::Md5 => Inner::Md5(KeyInit::new_from_slice(key).expect("any length")),
            QCryptoHashAlgo::Sha1 => Inner::Sha1(KeyInit::new_from_slice(key).expect("any length")),
            QCryptoHashAlgo::Sha224 => {
                Inner::Sha224(KeyInit::new_from_slice(key).expect("any length"))
            }
            QCryptoHashAlgo::Sha256 => {
                Inner::Sha256(KeyInit::new_from_slice(key).expect("any length"))
            }
            QCryptoHashAlgo::Sha384 => {
                Inner::Sha384(KeyInit::new_from_slice(key).expect("any length"))
            }
            QCryptoHashAlgo::Sha512 => {
                Inner::Sha512(KeyInit::new_from_slice(key).expect("any length"))
            }
            QCryptoHashAlgo::Ripemd160 => {
                Inner::Ripemd160(KeyInit::new_from_slice(key).expect("any length"))
            }
            QCryptoHashAlgo::Sm3 => Inner::Sm3(KeyInit::new_from_slice(key).expect("any length")),
        };
        Ok(Hmac { alg, inner })
    }

    pub fn alg(&self) -> QCryptoHashAlgo {
        self.alg
    }

    /// `qcrypto_hmac_bytesv()`.
    pub fn bytesv(&mut self, iov: &[&[u8]]) -> Result<Vec<u8>> {
        // The stored state only ever holds the key, so every call starts from a freshly keyed
        // context, like nettle's digest function which resets the context after output.
        Ok(each!(&self.inner, m => {
            let mut m = m.clone();
            for b in iov {
                m.update(b);
            }
            m.finalize().into_bytes().to_vec()
        }))
    }

    /// `qcrypto_hmac_bytes()`.
    pub fn bytes(&mut self, buf: &[u8]) -> Result<Vec<u8>> {
        self.bytesv(&[buf])
    }

    /// `qcrypto_hmac_digestv()`: the result in lower case hex.
    pub fn digestv(&mut self, iov: &[&[u8]]) -> Result<String> {
        Ok(to_hex(&self.bytesv(iov)?))
    }

    /// `qcrypto_hmac_digest()`.
    pub fn digest(&mut self, buf: &[u8]) -> Result<String> {
        self.digestv(&[buf])
    }
}
