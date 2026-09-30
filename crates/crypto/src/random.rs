// SPDX-License-Identifier: GPL-2.0-or-later

//! Random bytes, from crypto/random-*.c, taken from the operating system through getrandom.

use ruvm_base::{Error, Result};

/// `qcrypto_random_bytes()`.
pub fn random_bytes(buf: &mut [u8]) -> Result<()> {
    getrandom::fill(buf).map_err(|e| Error::generic(format!("Unable to read random bytes: {e}")))
}
