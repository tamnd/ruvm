// SPDX-License-Identifier: GPL-2.0-or-later

//! The anti-forensic information splitter of LUKS, from crypto/afsplit.c.

use ruvm_base::Result;
use ruvm_qapi::types::QCryptoHashAlgo;

use crate::hash::{Hash, hash_digest_len};
use crate::random::random_bytes;

fn xor_into(dst: &mut [u8], src: &[u8]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d ^= s;
    }
}

/// `qcrypto_afsplit_hash()`: the diffusion step, hashing each digest sized chunk of `block`
/// with its big-endian index in front.
fn diffuse(hash: QCryptoHashAlgo, block: &mut [u8]) -> Result<()> {
    let digestlen = hash_digest_len(hash);
    for (i, chunk) in block.chunks_mut(digestlen).enumerate() {
        let mut h = Hash::new(hash)?;
        h.update(&(i as u32).to_be_bytes());
        h.update(chunk);
        let out = h.finalize_bytes();
        let n = chunk.len();
        chunk.copy_from_slice(&out[..n]);
    }
    Ok(())
}

/// `qcrypto_afsplit_encode()`: splits `input` (`blocklen` bytes) into `stripes` blocks written
/// to `out`, which must hold `blocklen * stripes` bytes.
pub fn afsplit_encode(
    hash: QCryptoHashAlgo,
    blocklen: usize,
    stripes: u32,
    input: &[u8],
    out: &mut [u8],
) -> Result<()> {
    let stripes = stripes as usize;
    let mut block = vec![0u8; blocklen];
    for i in 0..stripes - 1 {
        let stripe = &mut out[i * blocklen..(i + 1) * blocklen];
        random_bytes(stripe)?;
        xor_into(&mut block, stripe);
        diffuse(hash, &mut block)?;
    }
    let last = &mut out[(stripes - 1) * blocklen..stripes * blocklen];
    for (o, (a, b)) in last.iter_mut().zip(input.iter().zip(&block)) {
        *o = a ^ b;
    }
    Ok(())
}

/// `qcrypto_afsplit_decode()`: merges `stripes` blocks of `input` back into `out`
/// (`blocklen` bytes).
pub fn afsplit_decode(
    hash: QCryptoHashAlgo,
    blocklen: usize,
    stripes: u32,
    input: &[u8],
    out: &mut [u8],
) -> Result<()> {
    let stripes = stripes as usize;
    let mut block = vec![0u8; blocklen];
    for i in 0..stripes - 1 {
        xor_into(&mut block, &input[i * blocklen..(i + 1) * blocklen]);
        diffuse(hash, &mut block)?;
    }
    let last = &input[(stripes - 1) * blocklen..stripes * blocklen];
    for (o, (a, b)) in out.iter_mut().zip(last.iter().zip(&block)) {
        *o = a ^ b;
    }
    Ok(())
}
