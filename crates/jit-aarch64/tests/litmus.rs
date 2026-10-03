// SPDX-License-Identifier: GPL-2.0-or-later

//! Memory ordering litmus tests on the host: two threads run translated blocks of an x86 guest
//! (`guest_mo` is x86-TSO) against shared guest memory in a host window, many times, and no
//! outcome x86-TSO forbids may appear. Each fence mapping this host supports is tried.
//!
//! The shapes, with `x` and `y` starting at 0:
//!
//! - MP, message passing: `x = 1; y = 1` against `r0 = y; r1 = x`. Forbidden: `r0 = 1, r1 = 0`.
//! - SB+mfence, store buffering with fences: `x = 1; mfence; r0 = y` against
//!   `y = 1; mfence; r1 = x`. Forbidden: `r0 = 0, r1 = 0`.
//! - SB+xchg, the same with a locked exchange in place of the store and fence. Forbidden:
//!   `r0 = 0, r1 = 0`.
//! - LB, load buffering: `r0 = x; y = 1` against `r1 = y; x = 1`. Forbidden: `r0 = 1, r1 = 1`.
//! - SB without fences, where `0, 0` is allowed by x86-TSO; its outcomes are only printed.
//!
//! The same blocks built without any ordering (`guest_mo` 0) are run as a control and only
//! printed: they show whether the host reorders at all in this harness.
//!
//! Each shape runs `RUVM_LITMUS_ITERS` times (100000 by default) for each mapping, in bursts of
//! 64 runs on separate variables between synchronizations of the threads, which takes well
//! under a few seconds. Run with `--nocapture` to see the outcome counts. Generated code
//! only runs on an AArch64 host; elsewhere the blocks are only compiled.

mod common;

use std::collections::BTreeMap;
use std::hint::spin_loop;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use common::{WordMemory, words};
use ruvm_jit_aarch64::{
    CodeRegion, CodegenOptions, CompiledTb, HostWindow, TARGET_DEFAULT_MO, select_fence_mapping,
};
use ruvm_jit_core::ir::{FuncConfig, TempI64};
use ruvm_jit_core::liveness::LogMask;
use ruvm_jit_core::memory_model::X86_TSO;
use ruvm_jit_core::tcg_op_ldst::AtomicOp;
use ruvm_jit_core::types::mo;
use ruvm_jit_core::{FenceMapping, Func, MemOp};
use ruvm_jit_interp::{Exit, HelperRegistry};

const NATIVE: bool = cfg!(all(unix, target_arch = "aarch64"));

/// Guest address of the window.
const BASE: u64 = 0x10_0000;
/// Each run uses its own `x` and `y`, at these offsets from a base taken from the CPU state,
/// on different 128 byte lines.
const X: u64 = 0;
const Y: u64 = 128;
/// Bytes between the variables of one run and the next.
const STRIDE: u64 = 256;
/// Runs between two synchronizations of the threads. Each runs on fresh variables, so that the
/// two threads' runs overlap in many different ways.
const BURST: usize = 64;
const WORDS: usize = BURST * STRIDE as usize / 8;
/// Results in env words 0 and 1, the variable base in word 2.
const ENV_SIZE: usize = 24;

/// One guest step of a litmus thread.
#[derive(Clone, Copy, Debug)]
enum Step {
    /// Store 1 to the variable at this offset.
    St(u64),
    /// Load the variable at this offset into the next result register.
    Ld(u64),
    /// `mfence`.
    Mfence,
    /// `xchg` of 1 with the variable at this offset, the old value discarded.
    Xchg(u64),
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    name: &'static str,
    threads: [&'static [Step]; 2],
    /// The outcome x86-TSO forbids, `(r0, r1)` with `r0` the first load of thread 0 or, if it
    /// has none, of thread 1.
    forbidden: Option<(u64, u64)>,
}

const SHAPES: [Shape; 5] = [
    Shape {
        name: "MP",
        threads: [&[Step::St(X), Step::St(Y)], &[Step::Ld(Y), Step::Ld(X)]],
        forbidden: Some((1, 0)),
    },
    Shape {
        name: "SB+mfence",
        threads: [
            &[Step::St(X), Step::Mfence, Step::Ld(Y)],
            &[Step::St(Y), Step::Mfence, Step::Ld(X)],
        ],
        forbidden: Some((0, 0)),
    },
    Shape {
        name: "SB+xchg",
        threads: [&[Step::Xchg(X), Step::Ld(Y)], &[Step::Xchg(Y), Step::Ld(X)]],
        forbidden: Some((0, 0)),
    },
    Shape {
        name: "LB",
        threads: [&[Step::Ld(X), Step::St(Y)], &[Step::Ld(Y), Step::St(X)]],
        forbidden: Some((1, 1)),
    },
    Shape {
        name: "SB",
        threads: [&[Step::St(X), Step::Ld(Y)], &[Step::St(Y), Step::Ld(X)]],
        forbidden: None,
    },
];

/// The block for one thread: its steps, loads written to env words 0 and 1, then exit.
fn block(steps: &[Step], guest_mo: u32, m: FenceMapping) -> Func {
    let cfg = FuncConfig {
        parallel: true,
        guest_mo,
        target_default_mo: TARGET_DEFAULT_MO,
        fence_mapping: m,
        ..FuncConfig::default()
    };
    let mut f = Func::new(cfg);
    let env = f.env();
    let r: Vec<TempI64> = (0..2).map(|k| f.global_mem_new_i64(env, 8 * k, "r")).collect();
    let base = f.global_mem_new_i64(env, 16, "base");
    let one = f.constant_i64(1);
    let mut nr = 0;
    for s in steps {
        match *s {
            Step::St(off) => {
                let a = f.temp_new_i64();
                f.gen_addi_i64(a, base, off as i64);
                f.gen_qemu_st_i64(one, a, 0, MemOp::UQ);
            }
            Step::Ld(off) => {
                let a = f.temp_new_i64();
                f.gen_addi_i64(a, base, off as i64);
                f.gen_qemu_ld_i64(r[nr], a, 0, MemOp::UQ);
                nr += 1;
            }
            Step::Mfence => f.gen_mb(mo::ALL | mo::BAR_SC),
            Step::Xchg(off) => {
                let a = f.temp_new_i64();
                f.gen_addi_i64(a, base, off as i64);
                let old = f.temp_new_i64();
                f.gen_atomic_op_i64(AtomicOp::Xchg, old, a, one, 0, MemOp::UQ);
            }
        }
    }
    f.gen_exit_tb(0, 0);
    f.gen_code(true, LogMask::default());
    f
}

fn compile(r: &Arc<CodeRegion>, f: &Func) -> CompiledTb {
    match r.compile_with(f, &CodegenOptions::host()) {
        Ok(tb) => tb,
        Err(e) => panic!("compile: {e}\n{}", f.dump_ops(false)),
    }
}

/// A barrier for two threads that spins.
struct SpinBarrier {
    count: AtomicU32,
    generation: AtomicU32,
}

impl SpinBarrier {
    fn new() -> SpinBarrier {
        SpinBarrier { count: AtomicU32::new(0), generation: AtomicU32::new(0) }
    }

    fn wait(&self) {
        let g = self.generation.load(Ordering::Acquire);
        if self.count.fetch_add(1, Ordering::AcqRel) == 1 {
            self.count.store(0, Ordering::Relaxed);
            self.generation.store(g.wrapping_add(1), Ordering::Release);
        } else {
            while self.generation.load(Ordering::Acquire) == g {
                spin_loop();
            }
        }
    }
}

fn iterations() -> usize {
    std::env::var("RUVM_LITMUS_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(100_000)
}

/// Run both blocks at least `n` times at once, in bursts of [`BURST`] between synchronizations,
/// and count the outcomes `(r0, r1)`.
fn run(tbs: &[CompiledTb; 2], shape: &Shape, n: usize) -> BTreeMap<(u64, u64), usize> {
    let n = n.div_ceil(BURST) * BURST;
    let ws = words(WORDS);
    let lock = Mutex::new(());
    let slow = AtomicUsize::new(0);
    let barrier = SpinBarrier::new();
    let loads = shape.threads.map(|s| s.iter().filter(|s| matches!(s, Step::Ld(_))).count());
    let results: Vec<Vec<[u64; 2]>> = std::thread::scope(|sc| {
        let handles: Vec<_> = (0..2)
            .map(|t| {
                let (ws, lock, slow, barrier, tb) = (&ws, &lock, &slow, &barrier, &tbs[t]);
                sc.spawn(move || {
                    let reg = HelperRegistry::new();
                    let mut mem = WordMemory::new(ws, BASE, lock, slow);
                    let win = HostWindow::new(ws, BASE);
                    let mut env = vec![0u8; ENV_SIZE];
                    let mut out = Vec::with_capacity(n);
                    for i in 0..n / BURST {
                        barrier.wait();
                        // Stagger the two threads a little so their accesses overlap in
                        // different ways.
                        for _ in 0..(i * 7 + 3 * t) % 11 {
                            spin_loop();
                        }
                        for k in 0..BURST as u64 {
                            env[16..24].copy_from_slice(&(BASE + k * STRIDE).to_le_bytes());
                            let x = tb.run_with_window(&mut env, &win, &mut mem, &reg);
                            assert_eq!(x, Ok(Exit::ExitTb(0)));
                            let rd = |k: usize| {
                                u64::from_le_bytes(env[8 * k..8 * k + 8].try_into().unwrap())
                            };
                            out.push([rd(0), rd(1)]);
                        }
                        barrier.wait();
                        if t == 0 {
                            for w in ws.iter() {
                                w.store(0, Ordering::Relaxed);
                            }
                        }
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("litmus thread")).collect()
    });
    // Plain loads and stores never leave the window; only the exchanges use the service
    // routine.
    let xchgs = shape.threads.iter().flat_map(|s| s.iter()).any(|s| matches!(s, Step::Xchg(_)));
    assert!(xchgs || slow.load(Ordering::Relaxed) == 0, "{}: window not used", shape.name);
    let mut counts = BTreeMap::new();
    for (a, b) in results[0].iter().zip(&results[1]) {
        let rs: Vec<u64> = a[..loads[0]].iter().chain(&b[..loads[1]]).copied().collect();
        *counts.entry((rs[0], rs[1])).or_insert(0) += 1;
    }
    counts
}

fn show(counts: &BTreeMap<(u64, u64), usize>) -> String {
    counts.iter().map(|((a, b), n)| format!("{a}/{b}: {n}")).collect::<Vec<_>>().join(", ")
}

#[test]
fn x86_tso_litmus() {
    let n = iterations();
    let region = CodeRegion::new(1 << 20).expect("code region");
    let mut mappings: Vec<FenceMapping> =
        FenceMapping::ALL.iter().map(|m| select_fence_mapping(*m, X86_TSO)).collect();
    mappings.dedup();
    let mut failures = Vec::new();
    for shape in &SHAPES {
        for &m in &mappings {
            let tbs = shape.threads.map(|s| compile(&region, &block(s, X86_TSO, m)));
            if !NATIVE {
                continue;
            }
            let counts = run(&tbs, shape, n);
            println!("{} {m}: {}", shape.name, show(&counts));
            assert_eq!(counts.values().sum::<usize>(), n.div_ceil(BURST) * BURST);
            if let Some(bad) = shape.forbidden {
                if let Some(k) = counts.get(&bad) {
                    failures
                        .push(format!("{} with {m}: forbidden {bad:?} seen {k} times", shape.name));
                }
            }
        }
        // The control: no ordering at all.
        let tbs = shape.threads.map(|s| compile(&region, &block(s, 0, FenceMapping::Qemu)));
        if NATIVE {
            let counts = run(&tbs, shape, n);
            println!("{} unordered (control): {}", shape.name, show(&counts));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The unordered control blocks really have no barriers or ordered accesses, so the control
/// says something about the host.
#[test]
fn control_blocks_are_unordered() {
    let region = CodeRegion::new(1 << 16).expect("code region");
    for shape in &SHAPES[..2] {
        for s in shape.threads {
            let code = compile(&region, &block(s, 0, FenceMapping::Qemu)).code();
            let words: Vec<u32> =
                code.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            let mfence = s.iter().any(|s| matches!(s, Step::Mfence));
            let dmbs = words.iter().filter(|w| *w & 0xfffff0ff == 0xd50330bf).count();
            assert_eq!(dmbs, mfence as usize, "{}", shape.name);
            assert!(!words.iter().any(|w| w & 0x3ffffc00 == 0x38bfc000), "ldapr");
            assert!(!words.iter().any(|w| w & 0x3ffffc00 == 0x089ffc00), "stlr");
        }
    }
}
