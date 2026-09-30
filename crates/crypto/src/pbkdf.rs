// SPDX-License-Identifier: GPL-2.0-or-later

//! PBKDF2 and its iteration count calibration, from crypto/pbkdf.c and crypto/pbkdf-gcrypt.c.
//!
//! [`pbkdf2_count_iters`] measures how many iterations the calling machine does in a second of
//! thread CPU time, as QEMU does, by running PBKDF2 on a separate thread. Two differences:
//!
//! - The thread CPU time comes from `CLOCK_THREAD_CPUTIME_ID`, which counts system time as well
//!   as user time; QEMU reads only the user time. On Windows it is the wall clock.
//! - The measurement can be replaced for tests: [`set_iters_per_second_override`] makes every
//!   calibration return a fixed rate, and [`pbkdf2_count_iters_with_clock`] runs the real loop
//!   against a clock the caller supplies.
//!
//! Iteration counts are 64 bits wide, so the `ULONG_MAX` check of the gcrypt backend can never
//! fire and is left out.

use std::sync::atomic::{AtomicU64, Ordering};

use digest::{KeyInit, Mac};
use ruvm_base::{Error, Result};
use ruvm_qapi::types::QCryptoHashAlgo;

/// `qcrypto_pbkdf2_supports()`.
pub fn pbkdf2_supports(_hash: QCryptoHashAlgo) -> bool {
    true
}

fn pbkdf2_generic<M>(key: &[u8], salt: &[u8], iterations: u64, out: &mut [u8])
where
    M: Mac + KeyInit + Clone,
{
    // HMAC accepts keys of every length.
    let prf = <M as KeyInit>::new_from_slice(key).expect("any key length");
    for (i, chunk) in out.chunks_mut(<M as digest::OutputSizeUser>::output_size()).enumerate() {
        let mut m = prf.clone();
        m.update(salt);
        m.update(&(i as u32 + 1).to_be_bytes());
        let mut u = m.finalize().into_bytes();
        let mut t = u.clone();
        for _ in 1..iterations {
            let mut m = prf.clone();
            m.update(&u);
            u = m.finalize().into_bytes();
            for (a, b) in t.iter_mut().zip(u.iter()) {
                *a ^= b;
            }
        }
        chunk.copy_from_slice(&t[..chunk.len()]);
    }
}

/// `qcrypto_pbkdf2()`: derives `out.len()` bytes from `key` and `salt`.
pub fn pbkdf2(
    hash: QCryptoHashAlgo,
    key: &[u8],
    salt: &[u8],
    iterations: u64,
    out: &mut [u8],
) -> Result<()> {
    if !pbkdf2_supports(hash) {
        return Err(Error::generic(format!(
            "PBKDF does not support hash algorithm {}",
            hash.as_str()
        )));
    }
    match hash {
        QCryptoHashAlgo::Md5 => pbkdf2_generic::<hmac::Hmac<md5::Md5>>(key, salt, iterations, out),
        QCryptoHashAlgo::Sha1 => {
            pbkdf2_generic::<hmac::Hmac<sha1::Sha1>>(key, salt, iterations, out)
        }
        QCryptoHashAlgo::Sha224 => {
            pbkdf2_generic::<hmac::Hmac<sha2::Sha224>>(key, salt, iterations, out)
        }
        QCryptoHashAlgo::Sha256 => {
            pbkdf2_generic::<hmac::Hmac<sha2::Sha256>>(key, salt, iterations, out)
        }
        QCryptoHashAlgo::Sha384 => {
            pbkdf2_generic::<hmac::Hmac<sha2::Sha384>>(key, salt, iterations, out)
        }
        QCryptoHashAlgo::Sha512 => {
            pbkdf2_generic::<hmac::Hmac<sha2::Sha512>>(key, salt, iterations, out)
        }
        QCryptoHashAlgo::Ripemd160 => {
            pbkdf2_generic::<hmac::Hmac<ripemd::Ripemd160>>(key, salt, iterations, out)
        }
        QCryptoHashAlgo::Sm3 => pbkdf2_generic::<hmac::Hmac<sm3::Sm3>>(key, salt, iterations, out),
    }
    Ok(())
}

static ITERS_OVERRIDE: AtomicU64 = AtomicU64::new(0);

/// Makes [`pbkdf2_count_iters`] return `rate` iterations per second without measuring anything,
/// or measure again with `None`. This is for tests, which want LUKS volumes created quickly and
/// with predictable iteration counts. It affects the whole process.
pub fn set_iters_per_second_override(rate: Option<u64>) {
    ITERS_OVERRIDE.store(rate.unwrap_or(0), Ordering::SeqCst);
}

/// The CPU time the calling thread has used, in milliseconds.
pub fn thread_cpu_ms() -> Result<u64> {
    #[cfg(unix)]
    {
        let ts = rustix::time::clock_gettime(rustix::time::ClockId::ThreadCPUTime);
        Ok(ts.tv_sec as u64 * 1000 + ts.tv_nsec as u64 / 1_000_000)
    }
    #[cfg(not(unix))]
    {
        use std::sync::OnceLock;
        use std::time::Instant;
        static START: OnceLock<Instant> = OnceLock::new();
        Ok(START.get_or_init(Instant::now).elapsed().as_millis() as u64)
    }
}

/// The measuring loop of `threaded_qcrypto_pbkdf2_count_iters()`, run on the calling thread with
/// `clock` giving the CPU time in milliseconds.
pub fn pbkdf2_count_iters_with_clock(
    hash: QCryptoHashAlgo,
    key: &[u8],
    salt: &[u8],
    nout: usize,
    clock: &mut dyn FnMut() -> Result<u64>,
) -> Result<u64> {
    let mut out = vec![0u8; nout];
    let mut iterations: u64 = 1 << 15;
    let mut scaled = 0;
    let delta_ms = loop {
        let start_ms = clock()?;
        pbkdf2(hash, key, salt, iterations, &mut out)?;
        let end_ms = clock()?;
        let delta_ms = end_ms.saturating_sub(start_ms);
        if scaled > 5 && delta_ms == 0 {
            return Err(Error::generic("Unable to get accurate CPU usage"));
        } else if delta_ms > 500 {
            break delta_ms;
        } else if delta_ms < 100 {
            iterations *= 10;
        } else {
            iterations = iterations * 1000 / delta_ms;
        }
        scaled += 1;
    };
    out.fill(0);
    Ok(iterations * 1000 / delta_ms)
}

/// `qcrypto_pbkdf2_count_iters()`: how many PBKDF2 iterations with this hash and these lengths
/// take one second of CPU time.
pub fn pbkdf2_count_iters(
    hash: QCryptoHashAlgo,
    key: &[u8],
    salt: &[u8],
    nout: usize,
) -> Result<u64> {
    let rate = ITERS_OVERRIDE.load(Ordering::SeqCst);
    if rate != 0 {
        return Ok(rate);
    }
    std::thread::scope(|s| {
        std::thread::Builder::new()
            .name("pbkdf2".into())
            .spawn_scoped(s, || {
                pbkdf2_count_iters_with_clock(hash, key, salt, nout, &mut thread_cpu_ms)
            })
            .map_err(|e| Error::from_io("Unable to create thread", e))?
            .join()
            .expect("the pbkdf2 thread does not panic")
    })
}
