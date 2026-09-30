// SPDX-License-Identifier: GPL-2.0-or-later

//! Ciphers, hashes, secrets and LUKS headers, compatible with QEMU's crypto subsystem.
//!
//! This crate mirrors QEMU's `crypto/` directory. The primitives come from the RustCrypto
//! project, so there is no OpenSSL, gnutls or nettle dependency. Where the QEMU backends
//! differ in behaviour, this crate follows the nettle backend, which is the one QEMU uses on
//! most distributions.
//!
//! * [`cipher`]: block ciphers in ECB, CBC, XTS and CTR mode.
//! * [`hash`], [`hmac`]: message digests and keyed digests.
//! * [`pbkdf`]: PBKDF2 with QEMU's iteration count calibration.
//! * [`ivgen`]: the plain, plain64 and essiv IV generators.
//! * [`afsplit`]: the LUKS anti-forensic splitter.
//! * [`secret`]: the `secret` user-creatable object and secret lookups.
//! * [`block`]: `QCryptoBlock`, the LUKS1 format and legacy qcow AES encryption.
//!
//! The TLS credential objects and the `tls-cipher-suites` object are not ported yet.

#![forbid(unsafe_code)]

pub mod afsplit;
pub mod base64;
pub mod block;
pub mod cipher;
pub mod hash;
pub mod hmac;
pub mod ivgen;
pub mod pbkdf;
pub mod random;
pub mod secret;
