// SPDX-License-Identifier: GPL-2.0-or-later

//! The differential harness against QEMU's own softfloat.
//!
//! `tests/c/driver.c` includes QEMU 11.1's `fpu/softfloat.c` and is built with the host C
//! compiler. Both implementations are fed the same random and edge case requests, over every
//! rounding mode and the x86, Arm, Arm FPCR.AH, legacy MIPS, PowerPC and m68k style NaN
//! configurations, and must agree bit for bit on the result and the exception flags.
//!
//! These tests need a QEMU source tree and a C compiler, so they are ignored by default:
//!
//! ```text
//! cargo test -p ruvm-softfloat --test qemu_diff -- --ignored --nocapture
//! ```
//!
//! `RUVM_QEMU_SRC` points at the QEMU tree (default `~/src/qemu-v11.1.0`), `CC` picks the
//! compiler (default `cc`), `RUVM_SOFTFLOAT_DIFF_N` sets the number of requests (default
//! 4 million) and `RUVM_SOFTFLOAT_DIFF_SEED` the seed. `regenerate_vectors` rewrites the
//! checked in `tests/vectors/qemu.bin` from QEMU's answers.

#![cfg(unix)]

mod common;

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use common::{REQ_LEN, RES_LEN, Req, Res};

/// Build the C driver and return its path.
fn build_driver() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let qemu = std::env::var_os("RUVM_QEMU_SRC").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var_os("HOME").expect("HOME")).join("src/qemu-v11.1.0")
    });
    assert!(
        qemu.join("fpu/softfloat.c").exists(),
        "no QEMU source tree at {}, set RUVM_QEMU_SRC",
        qemu.display()
    );
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("softfloat-qemu-driver");
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let c = manifest.join("tests/c");
    let status = Command::new(&cc)
        .arg("-O2")
        .arg("-w")
        .arg("-I")
        .arg(c.join("stubs"))
        .arg("-I")
        .arg(qemu.join("include"))
        .arg("-I")
        .arg(&qemu)
        .arg("-o")
        .arg(&out)
        .arg(c.join("driver.c"))
        .arg("-lm")
        .status()
        .expect("run the C compiler");
    assert!(status.success(), "building the QEMU softfloat driver failed");
    out
}

/// A running driver.
struct Driver {
    child: Child,
}

impl Driver {
    fn start() -> Self {
        let path = build_driver();
        let child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("start the driver");
        Driver { child }
    }

    /// Run a batch of requests through QEMU.
    fn run(&mut self, reqs: &[Req]) -> Vec<Res> {
        let mut bytes = Vec::with_capacity(reqs.len() * REQ_LEN);
        for r in reqs {
            r.encode(&mut bytes);
        }
        let flush = Req { op: 0xffff, ..Req::default() };
        flush.encode(&mut bytes);
        let mut out = vec![0u8; reqs.len() * RES_LEN];
        let stdin = self.child.stdin.as_mut().unwrap();
        let stdout = self.child.stdout.as_mut().unwrap();
        std::thread::scope(|scope| {
            scope.spawn(|| stdin.write_all(&bytes).expect("write to the driver"));
            stdout.read_exact(&mut out).expect("read from the driver (did it abort?)");
        });
        out.chunks_exact(RES_LEN).map(Res::decode).collect()
    }
}

impl Drop for Driver {
    fn drop(&mut self) {
        drop(self.child.stdin.take());
        let _ = self.child.wait();
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    match std::env::var(name) {
        Ok(v) => {
            let v = v.trim();
            match v.strip_prefix("0x") {
                Some(h) => u64::from_str_radix(h, 16).expect(name),
                None => v.parse().expect(name),
            }
        }
        Err(_) => default,
    }
}

#[test]
#[ignore = "needs a QEMU source tree and a C compiler"]
fn qemu_differential() {
    let total = env_u64("RUVM_SOFTFLOAT_DIFF_N", 4_000_000) as usize;
    let seed = env_u64("RUVM_SOFTFLOAT_DIFF_SEED", 0x0dd5_eed0_1234_5678);
    let ops = common::all_ops();
    let mut driver = Driver::start();
    let mut rng = common::Rng::new(seed);

    let mut per_op = std::collections::BTreeMap::<u16, (u64, u64)>::new();
    let mut shown = 0;
    let mut bad = 0u64;
    let batch = 1 << 16;
    let mut done = 0;
    while done < total {
        let n = batch.min(total - done);
        let reqs: Vec<Req> =
            (0..n).map(|i| common::gen_request(&mut rng, ops[(done + i) % ops.len()])).collect();
        let want = driver.run(&reqs);
        for (r, w) in reqs.iter().zip(&want) {
            let got = common::eval(r);
            let e = per_op.entry(r.op).or_default();
            e.0 += 1;
            if got != *w {
                e.1 += 1;
                bad += 1;
                if shown < 40 {
                    shown += 1;
                    eprintln!("MISMATCH {}", common::describe(r, w, &got));
                }
            }
        }
        done += n;
    }

    for (op, (n, b)) in &per_op {
        if *b != 0 {
            eprintln!("{:32} {b} of {n} differ", common::op_name(*op));
        }
    }
    eprintln!("{total} requests over {} ops, seed {seed:#x}: {bad} mismatches", per_op.len());
    assert_eq!(bad, 0, "ruvm-softfloat differs from QEMU");
}

#[test]
#[ignore = "needs a QEMU source tree and a C compiler; rewrites tests/vectors/qemu.bin"]
fn regenerate_vectors() {
    let reqs = common::gen_stream(common::VECTOR_SEED, common::VECTOR_COUNT);
    let mut driver = Driver::start();
    let want = driver.run(&reqs);

    let mut file = Vec::with_capacity(32 + reqs.len() * 8);
    file.extend_from_slice(common::VECTOR_MAGIC);
    file.extend_from_slice(&common::VECTOR_SEED.to_le_bytes());
    file.extend_from_slice(&(reqs.len() as u64).to_le_bytes());
    file.extend_from_slice(&common::stream_hash(&reqs).to_le_bytes());
    for w in &want {
        file.extend_from_slice(&w.hash().to_le_bytes());
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/qemu.bin");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, file).unwrap();
    eprintln!("wrote {} vectors to {}", reqs.len(), path.display());
}
