// SPDX-License-Identifier: GPL-2.0-or-later

//! Block ciphers and their modes, from crypto/cipher.c and crypto/cipher-nettle.c.inc.
//!
//! The block primitives come from RustCrypto. The modes are written here so that they keep the
//! nettle backend's behaviour exactly:
//!
//! - CBC and CTR carry the IV from one call to the next, as nettle updates it in place;
//! - CTR treats the whole IV block as one big-endian counter;
//! - XTS computes the tweak from the IV on every call and does not change the stored IV, and
//!   the length must be a multiple of the block size (no ciphertext stealing, which QEMU never
//!   reaches because of its length check);
//! - SM4 only works in ECB mode, and CAST5 has no XTS mode, which are the nettle limits.
//!
//! QEMU's `qcrypto_cipher_encrypt()` takes separate input and output buffers. Here the buffer is
//! transformed in place, which is how every caller in the block layer uses it.

use cipher::{BlockDecrypt, BlockEncrypt, KeyInit};
use ruvm_base::{Error, Result};
use ruvm_qapi::types::{QCryptoCipherAlgo, QCryptoCipherMode};

/// `qcrypto_cipher_get_key_len()`.
pub fn cipher_get_key_len(alg: QCryptoCipherAlgo) -> usize {
    match alg {
        QCryptoCipherAlgo::Des => 8,
        QCryptoCipherAlgo::Aes128
        | QCryptoCipherAlgo::Cast5_128
        | QCryptoCipherAlgo::Serpent128
        | QCryptoCipherAlgo::Twofish128
        | QCryptoCipherAlgo::Sm4 => 16,
        QCryptoCipherAlgo::Aes192
        | QCryptoCipherAlgo::V3des
        | QCryptoCipherAlgo::Serpent192
        | QCryptoCipherAlgo::Twofish192 => 24,
        QCryptoCipherAlgo::Aes256
        | QCryptoCipherAlgo::Serpent256
        | QCryptoCipherAlgo::Twofish256 => 32,
    }
}

/// `qcrypto_cipher_get_block_len()`.
pub fn cipher_get_block_len(alg: QCryptoCipherAlgo) -> usize {
    match alg {
        QCryptoCipherAlgo::Des | QCryptoCipherAlgo::V3des | QCryptoCipherAlgo::Cast5_128 => 8,
        _ => 16,
    }
}

/// `qcrypto_cipher_get_iv_len()`.
pub fn cipher_get_iv_len(alg: QCryptoCipherAlgo, mode: QCryptoCipherMode) -> usize {
    match mode {
        QCryptoCipherMode::Ecb => 0,
        _ => cipher_get_block_len(alg),
    }
}

/// `qcrypto_cipher_supports()`. Like the nettle backend this says yes to every pair, even the
/// ones [`Cipher::new`] then refuses.
pub fn cipher_supports(_alg: QCryptoCipherAlgo, _mode: QCryptoCipherMode) -> bool {
    true
}

fn validate_key_length(alg: QCryptoCipherAlgo, mode: QCryptoCipherMode, nkey: usize) -> Result<()> {
    let want = cipher_get_key_len(alg);
    if mode == QCryptoCipherMode::Xts {
        if matches!(alg, QCryptoCipherAlgo::Des | QCryptoCipherAlgo::V3des) {
            return Err(Error::generic("XTS mode not compatible with DES/3DES"));
        }
        if nkey % 2 != 0 {
            return Err(Error::generic("XTS cipher key length should be a multiple of 2"));
        }
        if want != nkey / 2 {
            return Err(Error::generic(format!("Cipher key length {nkey} should be {}", want * 2)));
        }
    } else if want != nkey {
        return Err(Error::generic(format!("Cipher key length {nkey} should be {want}")));
    }
    Ok(())
}

/// One keyed block primitive.
trait BlockPrim: Send + Sync {
    fn encrypt_block(&self, block: &mut [u8]);
    fn decrypt_block(&self, block: &mut [u8]);
}

struct Prim<T>(T);

impl<T> BlockPrim for Prim<T>
where
    T: BlockEncrypt + BlockDecrypt + Send + Sync,
{
    fn encrypt_block(&self, block: &mut [u8]) {
        self.0.encrypt_block(cipher::generic_array::GenericArray::from_mut_slice(block));
    }

    fn decrypt_block(&self, block: &mut [u8]) {
        self.0.decrypt_block(cipher::generic_array::GenericArray::from_mut_slice(block));
    }
}

fn prim<T>(key: &[u8]) -> Box<dyn BlockPrim>
where
    T: KeyInit + BlockEncrypt + BlockDecrypt + Send + Sync + 'static,
{
    // The key length was validated against the algorithm already.
    Box::new(Prim(T::new_from_slice(key).expect("validated key length")))
}

fn new_prim(alg: QCryptoCipherAlgo, key: &[u8]) -> Box<dyn BlockPrim> {
    match alg {
        QCryptoCipherAlgo::Aes128 => prim::<aes::Aes128>(key),
        QCryptoCipherAlgo::Aes192 => prim::<aes::Aes192>(key),
        QCryptoCipherAlgo::Aes256 => prim::<aes::Aes256>(key),
        QCryptoCipherAlgo::Des => prim::<des::Des>(key),
        QCryptoCipherAlgo::V3des => prim::<des::TdesEde3>(key),
        QCryptoCipherAlgo::Cast5_128 => prim::<cast5::Cast5>(key),
        QCryptoCipherAlgo::Serpent128
        | QCryptoCipherAlgo::Serpent192
        | QCryptoCipherAlgo::Serpent256 => prim::<serpent::Serpent>(key),
        QCryptoCipherAlgo::Twofish128
        | QCryptoCipherAlgo::Twofish192
        | QCryptoCipherAlgo::Twofish256 => prim::<twofish::Twofish>(key),
        QCryptoCipherAlgo::Sm4 => prim::<sm4::Sm4>(key),
    }
}

/// `QCryptoCipher`: a keyed cipher in one mode, with its IV.
pub struct Cipher {
    alg: QCryptoCipherAlgo,
    mode: QCryptoCipherMode,
    blen: usize,
    key: Box<dyn BlockPrim>,
    key_xts: Option<Box<dyn BlockPrim>>,
    iv: Vec<u8>,
}

impl std::fmt::Debug for Cipher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cipher")
            .field("alg", &self.alg)
            .field("mode", &self.mode)
            .finish_non_exhaustive()
    }
}

impl Cipher {
    /// `qcrypto_cipher_new()`.
    pub fn new(alg: QCryptoCipherAlgo, mode: QCryptoCipherMode, key: &[u8]) -> Result<Cipher> {
        validate_key_length(alg, mode, key.len())?;
        let bad_mode = || Err(Error::generic(format!("Unsupported cipher mode {}", mode.as_str())));
        match (alg, mode) {
            (QCryptoCipherAlgo::Cast5_128, QCryptoCipherMode::Xts) => return bad_mode(),
            (QCryptoCipherAlgo::Sm4, m) if m != QCryptoCipherMode::Ecb => return bad_mode(),
            _ => {}
        }
        let blen = cipher_get_block_len(alg);
        let (key, key_xts) = if mode == QCryptoCipherMode::Xts {
            let (k1, k2) = key.split_at(key.len() / 2);
            (new_prim(alg, k1), Some(new_prim(alg, k2)))
        } else {
            (new_prim(alg, key), None)
        };
        Ok(Cipher { alg, mode, blen, key, key_xts, iv: vec![0; blen] })
    }

    pub fn alg(&self) -> QCryptoCipherAlgo {
        self.alg
    }

    pub fn mode(&self) -> QCryptoCipherMode {
        self.mode
    }

    /// `qcrypto_cipher_setiv()`.
    pub fn setiv(&mut self, iv: &[u8]) -> Result<()> {
        if self.mode == QCryptoCipherMode::Ecb {
            return Err(Error::generic("Setting IV is not supported"));
        }
        if iv.len() != self.blen {
            return Err(Error::generic(format!("Expected IV size {} not {}", self.blen, iv.len())));
        }
        self.iv.copy_from_slice(iv);
        Ok(())
    }

    fn length_check(&self, len: usize) -> Result<()> {
        if len & (self.blen - 1) != 0 {
            return Err(Error::generic(format!(
                "Length {len} must be a multiple of block size {}",
                self.blen
            )));
        }
        Ok(())
    }

    /// `qcrypto_cipher_encrypt()`, in place.
    pub fn encrypt(&mut self, buf: &mut [u8]) -> Result<()> {
        self.length_check(buf.len())?;
        let blen = self.blen;
        match self.mode {
            QCryptoCipherMode::Ecb => {
                for b in buf.chunks_exact_mut(blen) {
                    self.key.encrypt_block(b);
                }
            }
            QCryptoCipherMode::Cbc => {
                for b in buf.chunks_exact_mut(blen) {
                    xor(b, &self.iv);
                    self.key.encrypt_block(b);
                    self.iv.copy_from_slice(b);
                }
            }
            QCryptoCipherMode::Ctr => self.ctr(buf),
            QCryptoCipherMode::Xts => self.xts(buf, true),
        }
        Ok(())
    }

    /// `qcrypto_cipher_decrypt()`, in place.
    pub fn decrypt(&mut self, buf: &mut [u8]) -> Result<()> {
        self.length_check(buf.len())?;
        let blen = self.blen;
        match self.mode {
            QCryptoCipherMode::Ecb => {
                for b in buf.chunks_exact_mut(blen) {
                    self.key.decrypt_block(b);
                }
            }
            QCryptoCipherMode::Cbc => {
                let mut prev = [0u8; 16];
                for b in buf.chunks_exact_mut(blen) {
                    prev[..blen].copy_from_slice(b);
                    self.key.decrypt_block(b);
                    xor(b, &self.iv);
                    self.iv.copy_from_slice(&prev[..blen]);
                }
            }
            QCryptoCipherMode::Ctr => self.ctr(buf),
            QCryptoCipherMode::Xts => self.xts(buf, false),
        }
        Ok(())
    }

    fn ctr(&mut self, buf: &mut [u8]) {
        let blen = self.blen;
        let mut ks = [0u8; 16];
        for b in buf.chunks_exact_mut(blen) {
            ks[..blen].copy_from_slice(&self.iv);
            self.key.encrypt_block(&mut ks[..blen]);
            xor(b, &ks[..blen]);
            for byte in self.iv.iter_mut().rev() {
                *byte = byte.wrapping_add(1);
                if *byte != 0 {
                    break;
                }
            }
        }
    }

    fn xts(&self, buf: &mut [u8], enc: bool) {
        // Only 16-byte ciphers get this far.
        let mut t = [0u8; 16];
        t.copy_from_slice(&self.iv);
        self.key_xts.as_ref().expect("xts has a tweak key").encrypt_block(&mut t);
        for b in buf.chunks_exact_mut(16) {
            xor(b, &t);
            if enc {
                self.key.encrypt_block(b);
            } else {
                self.key.decrypt_block(b);
            }
            xor(b, &t);
            xts_mul_alpha(&mut t);
        }
    }
}

fn xor(dst: &mut [u8], src: &[u8]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d ^= s;
    }
}

/// Multiplies the XTS tweak by x in GF(2^128), little-endian as IEEE 1619 has it.
fn xts_mul_alpha(t: &mut [u8; 16]) {
    let carry = t[15] >> 7;
    for i in (1..16).rev() {
        t[i] = (t[i] << 1) | (t[i - 1] >> 7);
    }
    t[0] = (t[0] << 1) ^ (carry * 0x87);
}
