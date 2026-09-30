// SPDX-License-Identifier: GPL-2.0-or-later

//! Initialization vector generators, from crypto/ivgen.c, crypto/ivgen-plain.c,
//! crypto/ivgen-plain64.c and crypto/ivgen-essiv.c.

use std::sync::Mutex;

use ruvm_base::Result;
use ruvm_qapi::types::{QCryptoCipherAlgo, QCryptoCipherMode, QCryptoHashAlgo, QCryptoIVGenAlgo};

use crate::cipher::{Cipher, cipher_get_block_len, cipher_get_key_len};
use crate::hash::{hash_bytes, hash_digest_len};

/// `QCryptoIVGen`.
#[derive(Debug)]
pub struct IvGen {
    algorithm: QCryptoIVGenAlgo,
    cipher: QCryptoCipherAlgo,
    hash: QCryptoHashAlgo,
    essiv: Option<Mutex<Cipher>>,
}

impl IvGen {
    /// `qcrypto_ivgen_new()`. `cipheralg` and `hash` only matter for ESSIV; `key` is the master
    /// key of the volume.
    pub fn new(
        alg: QCryptoIVGenAlgo,
        cipheralg: QCryptoCipherAlgo,
        hash: QCryptoHashAlgo,
        key: &[u8],
    ) -> Result<IvGen> {
        let essiv = match alg {
            QCryptoIVGenAlgo::Plain | QCryptoIVGenAlgo::Plain64 => None,
            QCryptoIVGenAlgo::Essiv => {
                let nsalt = cipher_get_key_len(cipheralg);
                let nhash = hash_digest_len(hash);
                let salt = hash_bytes(hash, key)?;
                let c = Cipher::new(cipheralg, QCryptoCipherMode::Ecb, &salt[..nhash.min(nsalt)])?;
                Some(Mutex::new(c))
            }
        };
        Ok(IvGen { algorithm: alg, cipher: cipheralg, hash, essiv })
    }

    /// `qcrypto_ivgen_calculate()`: fills `iv` for `sector`.
    pub fn calculate(&self, sector: u64, iv: &mut [u8]) -> Result<()> {
        iv.fill(0);
        match self.algorithm {
            QCryptoIVGenAlgo::Plain => {
                let s = (sector as u32).to_le_bytes();
                let n = iv.len().min(s.len());
                iv[..n].copy_from_slice(&s[..n]);
            }
            QCryptoIVGenAlgo::Plain64 => {
                let s = sector.to_le_bytes();
                let n = iv.len().min(s.len());
                iv[..n].copy_from_slice(&s[..n]);
            }
            QCryptoIVGenAlgo::Essiv => {
                let ndata = cipher_get_block_len(self.cipher);
                let mut data = vec![0u8; ndata];
                let s = sector.to_le_bytes();
                let n = ndata.min(s.len());
                data[..n].copy_from_slice(&s[..n]);
                let mut c = self.essiv.as_ref().expect("essiv has a cipher").lock().unwrap();
                c.encrypt(&mut data)?;
                let n = ndata.min(iv.len());
                iv[..n].copy_from_slice(&data[..n]);
            }
        }
        Ok(())
    }

    /// `qcrypto_ivgen_get_algorithm()`.
    pub fn algorithm(&self) -> QCryptoIVGenAlgo {
        self.algorithm
    }

    /// `qcrypto_ivgen_get_cipher()`.
    pub fn cipher(&self) -> QCryptoCipherAlgo {
        self.cipher
    }

    /// `qcrypto_ivgen_get_hash()`.
    pub fn hash(&self) -> QCryptoHashAlgo {
        self.hash
    }
}
