// SPDX-License-Identifier: GPL-2.0-or-later

//! Base64 as QEMU uses it: `qbase64_decode()` from util/base64.c on top of glib's
//! `g_base64_decode()`, and `g_base64_encode()`.
//!
//! The decoder checks the characters first, then decodes as leniently as glib does: newlines are
//! skipped, `=` counts as a zero digit that also drops output bytes, and a trailing partial
//! group is ignored.
//!
//! Rust slices carry their length, so the "Base64 data is not NUL terminated" case of
//! `qbase64_decode()` cannot happen here; a trailing NUL, if the caller has one, must be left
//! out of the input.

use ruvm_base::{Error, Result};

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// `g_base64_encode()`: standard alphabet with `=` padding and no line breaks.
pub fn encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let v = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(ALPHABET[(v >> 18) as usize & 63] as char);
        out.push(ALPHABET[(v >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { ALPHABET[(v >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { ALPHABET[v as usize & 63] as char } else { '=' });
    }
    out
}

fn rank(c: u8) -> Option<u32> {
    match c {
        b'A'..=b'Z' => Some(u32::from(c - b'A')),
        b'a'..=b'z' => Some(u32::from(c - b'a') + 26),
        b'0'..=b'9' => Some(u32::from(c - b'0') + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        b'=' => Some(0),
        _ => None,
    }
}

/// `g_base64_decode()`, without any validation.
fn g_base64_decode(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut v: u32 = 0;
    let mut i = 0;
    let mut last = [0u8; 2];
    for &c in input {
        if let Some(r) = rank(c) {
            last[1] = last[0];
            last[0] = c;
            v = (v << 6) | r;
            i += 1;
            if i == 4 {
                out.push((v >> 16) as u8);
                if last[1] != b'=' {
                    out.push((v >> 8) as u8);
                }
                if last[0] != b'=' {
                    out.push(v as u8);
                }
                i = 0;
            }
        }
    }
    out
}

/// `qbase64_decode()`.
pub fn decode(input: &[u8]) -> Result<Vec<u8>> {
    if input.contains(&0) {
        return Err(Error::generic("Base64 data contains embedded NUL characters"));
    }
    if !input.iter().all(|&c| rank(c).is_some() || c == b'\n') {
        return Err(Error::generic("Base64 data contains invalid characters"));
    }
    Ok(g_base64_decode(input))
}
