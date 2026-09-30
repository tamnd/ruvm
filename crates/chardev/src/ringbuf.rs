// SPDX-License-Identifier: GPL-2.0-or-later

//! The `ringbuf` chardev and its old name `memory`, chardev/char-ringbuf.c: what the frontend
//! writes is kept in a ring, the oldest bytes giving way, for `ringbuf-read` to take.

use std::sync::{Mutex, MutexGuard};

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{ChardevRingbuf, DataFormat};

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Debug)]
struct Ring {
    buf: Vec<u8>,
    prod: usize,
    cons: usize,
}

/// `RingBufChardev`.
#[derive(Debug)]
pub(crate) struct Ringbuf {
    ring: Mutex<Ring>,
    /// Made as `memory`, `TYPE_CHARDEV_MEMORY`.
    pub(crate) memory: bool,
}

impl Ringbuf {
    /// `ringbuf_chr_open()`.
    pub(crate) fn open(opts: &ChardevRingbuf, memory: bool) -> Result<Ringbuf> {
        let size = opts.size.unwrap_or(65536);
        // A size of 0 is not a power of two either, whatever the bit trick in QEMU says.
        let size = usize::try_from(size)
            .ok()
            .filter(|s| s.is_power_of_two())
            .ok_or_else(|| Error::generic("size of ringbuf chardev must be power of two"))?;
        Ok(Ringbuf { ring: Mutex::new(Ring { buf: vec![0; size], prod: 0, cons: 0 }), memory })
    }

    /// `ringbuf_chr_write()`.
    pub(crate) fn write(&self, buf: &[u8]) -> usize {
        let mut r = lock(&self.ring);
        let size = r.buf.len();
        for &b in buf {
            let at = r.prod & (size - 1);
            r.buf[at] = b;
            r.prod = r.prod.wrapping_add(1);
            if r.prod.wrapping_sub(r.cons) > size {
                r.cons = r.prod.wrapping_sub(size);
            }
        }
        buf.len()
    }

    /// `ringbuf_chr_read()` of up to `len` bytes.
    fn read(&self, len: usize) -> Vec<u8> {
        let mut r = lock(&self.ring);
        let size = r.buf.len();
        let n = len.min(r.prod.wrapping_sub(r.cons));
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(r.buf[r.cons & (size - 1)]);
            r.cons = r.cons.wrapping_add(1);
        }
        out
    }

    /// The data half of `qmp_ringbuf_write()`.
    pub(crate) fn qmp_write(&self, data: &str, format: Option<DataFormat>) -> Result<()> {
        if format == Some(DataFormat::Base64) {
            self.write(&base64_decode(data)?);
        } else {
            self.write(data.as_bytes());
        }
        Ok(())
    }

    /// The data half of `qmp_ringbuf_read()`.
    pub(crate) fn qmp_read(&self, size: i64, format: Option<DataFormat>) -> Result<String> {
        if size <= 0 {
            return Err(Error::generic("size must be greater than zero"));
        }
        let data = self.read(usize::try_from(size).unwrap_or(usize::MAX));
        if format == Some(DataFormat::Base64) {
            return Ok(base64_encode(&data));
        }
        // QEMU hands back a C string, which ends at the first NUL. Bytes that are not UTF-8
        // are replaced, the FIXME in QEMU says as much.
        let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
        Ok(String::from_utf8_lossy(&data[..end]).into_owned())
    }
}

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// `g_base64_encode()`.
pub(crate) fn base64_encode(data: &[u8]) -> String {
    let mut s = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                s.push(char::from(ALPHABET[(n >> (18 - 6 * i)) as usize & 63]));
            } else {
                s.push('=');
            }
        }
    }
    s
}

/// `qbase64_decode()`: only the base64 alphabet, `=` and newlines are allowed, and then
/// decoding is as forgiving as `g_base64_decode()`.
pub(crate) fn base64_decode(s: &str) -> Result<Vec<u8>> {
    if s.bytes().any(|c| !(ALPHABET.contains(&c) || c == b'=' || c == b'\n')) {
        return Err(Error::generic("Base64 data contains invalid characters"));
    }
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes() {
        let Some(v) = ALPHABET.iter().position(|&a| a == c) else { continue };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}
