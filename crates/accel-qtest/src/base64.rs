// SPDX-License-Identifier: GPL-2.0-or-later

//! The two GLib base64 helpers the `b64read` and `b64write` commands use.
//!
//! The decoder follows GLib's `g_base64_decode_step()` rather than a strict RFC 4648 decoder:
//! characters outside the alphabet are skipped, `=` counts as a zero digit that suppresses output
//! bytes, and a trailing group of fewer than four digits is dropped. Clients depend on that
//! leniency, so a strict decoder would change what lands in guest memory.

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// `g_base64_encode()`: standard alphabet, padded with `=`.
pub(crate) fn encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);
        let v = (u32::from(b0) << 16) | (u32::from(b1) << 8) | u32::from(b2);
        out.push(char::from(ALPHABET[(v >> 18) as usize & 63]));
        out.push(char::from(ALPHABET[(v >> 12) as usize & 63]));
        if chunk.len() > 1 {
            out.push(char::from(ALPHABET[(v >> 6) as usize & 63]));
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(char::from(ALPHABET[v as usize & 63]));
        } else {
            out.push('=');
        }
    }
    out
}

/// The `mime_base64_rank` table: the digit value, 0 for `=`, and `None` for anything GLib skips.
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

/// `g_base64_decode_inplace()`, which is one `g_base64_decode_step()` over the whole text.
pub(crate) fn decode(text: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3 + 3);
    let mut v: u32 = 0;
    let mut i = 0;
    let mut last = [0u8; 2];
    for &c in text {
        let Some(r) = rank(c) else {
            continue;
        };
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
    out
}
