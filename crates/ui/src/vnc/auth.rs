// SPDX-License-Identifier: GPL-2.0-or-later

//! VNC password authentication, from `start_auth_vnc()` and `protocol_client_auth_vnc()` in
//! QEMU's ui/vnc.c.
//!
//! The server sends 16 random bytes, the client sends them back DES encrypted with the password
//! as the key. RFB takes the key bits in the opposite order to DES, so each byte of the password
//! is bit reversed before it becomes the key, which is what QEMU's `deskey()` callers do.

use ruvm_base::Result;
use ruvm_crypto::cipher::Cipher;
use ruvm_qapi::types::{QCryptoCipherAlgo, QCryptoCipherMode};

/// `VNC_AUTH_CHALLENGE_SIZE`.
pub const CHALLENGE_SIZE: usize = 16;

/// A fresh challenge, `make_challenge()`.
pub fn make_challenge() -> Result<[u8; CHALLENGE_SIZE]> {
    let mut challenge = [0u8; CHALLENGE_SIZE];
    ruvm_crypto::random::random_bytes(&mut challenge)?;
    Ok(challenge)
}

/// The DES key for a password: its first eight bytes, up to the first NUL as `strlen()` would
/// stop, zero padded, each byte bit reversed.
pub fn password_key(password: &[u8]) -> [u8; 8] {
    let mut key = [0u8; 8];
    for (k, &p) in key.iter_mut().zip(password.iter().take_while(|&&p| p != 0)) {
        *k = p.reverse_bits();
    }
    key
}

/// The response a client knowing `password` gives to `challenge`.
pub fn expected_response(
    password: &[u8],
    challenge: &[u8; CHALLENGE_SIZE],
) -> Result<[u8; CHALLENGE_SIZE]> {
    let key = password_key(password);
    let mut cipher = Cipher::new(QCryptoCipherAlgo::Des, QCryptoCipherMode::Ecb, &key)?;
    let mut response = *challenge;
    cipher.encrypt(&mut response)?;
    Ok(response)
}
