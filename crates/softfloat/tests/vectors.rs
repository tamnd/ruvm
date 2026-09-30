// SPDX-License-Identifier: GPL-2.0-or-later

//! Replays the checked in QEMU vectors: a fast, deterministic subset of the differential
//! harness in `qemu_diff.rs` that needs neither QEMU nor a C compiler.
//!
//! `tests/vectors/qemu.bin` holds, for each request of the deterministic stream
//! `common::gen_stream(VECTOR_SEED, VECTOR_COUNT)`, a hash of QEMU 11.1's result and flags. The
//! stream covers every compared operation of every format under every rounding mode and the
//! x86, Arm, Arm FPCR.AH, legacy MIPS, PowerPC and m68k style configurations. Regenerate the
//! file with `cargo test -p ruvm-softfloat --test qemu_diff -- --ignored regenerate_vectors`.

mod common;

const VECTORS: &[u8] = include_bytes!("vectors/qemu.bin");

fn u64_at(i: usize) -> u64 {
    u64::from_le_bytes(VECTORS[i..i + 8].try_into().unwrap())
}

#[test]
fn qemu_vectors() {
    assert_eq!(&VECTORS[..8], common::VECTOR_MAGIC, "not a vector file");
    let seed = u64_at(8);
    let count = u64_at(16) as usize;
    assert_eq!(seed, common::VECTOR_SEED);
    assert_eq!(count, common::VECTOR_COUNT);
    assert_eq!(VECTORS.len(), 32 + count * 8);

    let reqs = common::gen_stream(seed, count);
    assert_eq!(
        common::stream_hash(&reqs),
        u64_at(24),
        "the request generator changed; regenerate tests/vectors/qemu.bin from QEMU"
    );

    let mut bad = 0;
    for (i, r) in reqs.iter().enumerate() {
        let got = common::eval(r);
        if got.hash() != u64_at(32 + i * 8) {
            bad += 1;
            if bad <= 20 {
                eprintln!(
                    "vector {i} differs from QEMU: {}\n  ruvm result {:?} (run the qemu_diff harness for QEMU's)",
                    common::op_name(r.op),
                    got
                );
                eprintln!("  request {r:?}");
            }
        }
    }
    assert_eq!(bad, 0, "{bad} of {count} vectors differ from QEMU");
}
