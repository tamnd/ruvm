// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared harness: build a block whose inputs and outputs are globals in the env buffer, then run
//! it three ways (as built, through the optimizer and liveness, and with constant inputs through
//! the optimizer) and check that all three agree.

#![allow(dead_code)]

use ruvm_jit_core::Func;
use ruvm_jit_core::ir::{FuncConfig, TempI32, TempI64};
use ruvm_jit_core::liveness::LogMask;
use ruvm_jit_interp::{Exit, NoMemory, run_tb};

pub(crate) const ENV_SIZE: usize = 0x400;
const IN_BASE: i64 = 0x100;
const OUT_BASE: i64 = 0x200;

/// Run `f` and return the final env and the exit.
pub(crate) fn run(f: &Func, env: &[u8]) -> (Vec<u8>, Exit) {
    let mut e = env.to_vec();
    let exit = run_tb(f, &mut e, &mut NoMemory).expect("run");
    (e, exit)
}

/// Run `f` as built and after `gen_code(true)`, check both agree, and return the env.
pub(crate) fn run_both(f: &Func, env: &[u8]) -> Vec<u8> {
    if let Err(e) = f.verify() {
        panic!("verify failed: {e:?}\n{}", f.dump_ops(false));
    }
    let (e0, x0) = run(f, env);
    let mut g = f.clone();
    g.gen_code(true, LogMask::default());
    let (e1, x1) = run(&g, env);
    assert_eq!(x0, x1, "exit differs after optimization\n{}", g.dump_ops(true));
    assert_eq!(e0, e1, "env differs after optimization\n{}", g.dump_ops(true));
    e0
}

fn rd(env: &[u8], off: i64, n: usize) -> u64 {
    let mut b = [0u8; 8];
    b[..n].copy_from_slice(&env[off as usize..off as usize + n]);
    u64::from_le_bytes(b)
}

macro_rules! evaluator {
    ($name:ident, $t:ty, $temp:ident, $gnew:ident, $cst:ident, $sz:expr, $st:ty) => {
        /// Evaluate `gen` on `ins`, three ways, returning the outputs.
        pub(crate) fn $name(
            nout: usize,
            ins: &[$t],
            build: impl Fn(&mut Func, &[$temp], &[$temp]),
        ) -> Vec<$t> {
            let mut results = Vec::new();
            for constant in [false, true] {
                let mut f = Func::new(FuncConfig::default());
                let env = f.env();
                let outs: Vec<$temp> = (0..nout)
                    .map(|i| f.$gnew(env, OUT_BASE + 8 * i as i64, &format!("out{i}")))
                    .collect();
                let args: Vec<$temp> = ins
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        if constant {
                            f.$cst(v as $st)
                        } else {
                            f.$gnew(env, IN_BASE + 8 * i as i64, &format!("in{i}"))
                        }
                    })
                    .collect();
                build(&mut f, &outs, &args);
                f.gen_exit_tb(0, 0);
                let mut e = vec![0u8; ENV_SIZE];
                for (i, &v) in ins.iter().enumerate() {
                    let o = (IN_BASE + 8 * i as i64) as usize;
                    e[o..o + $sz].copy_from_slice(&v.to_le_bytes());
                }
                let e = run_both(&f, &e);
                let r: Vec<$t> =
                    (0..nout).map(|i| rd(&e, OUT_BASE + 8 * i as i64, $sz) as $t).collect();
                results.push(r);
            }
            assert_eq!(results[0], results[1], "constant inputs give a different result");
            results.pop().unwrap()
        }
    };
}

evaluator!(eval32, u32, TempI32, global_mem_new_i32, constant_i32, 4, i32);
evaluator!(eval64, u64, TempI64, global_mem_new_i64, constant_i64, 8, i64);

/// One output, 32 bits.
pub(crate) fn e32(ins: &[u32], build: impl Fn(&mut Func, TempI32, &[TempI32])) -> u32 {
    eval32(1, ins, |f, o, a| build(f, o[0], a))[0]
}

/// One output, 64 bits.
pub(crate) fn e64(ins: &[u64], build: impl Fn(&mut Func, TempI64, &[TempI64])) -> u64 {
    eval64(1, ins, |f, o, a| build(f, o[0], a))[0]
}

/// A small xorshift generator, so the tests need no extra crates.
#[derive(Clone, Debug)]
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    pub(crate) fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    pub(crate) fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// A value biased toward edge cases.
    pub(crate) fn interesting(&mut self) -> u64 {
        const EDGE: [u64; 10] = [
            0,
            1,
            2,
            0x7f,
            0x80,
            0xffff_ffff,
            0x8000_0000,
            0x7fff_ffff,
            u64::MAX,
            0x8000_0000_0000_0000,
        ];
        match self.below(4) {
            0 => EDGE[self.below(EDGE.len() as u64) as usize],
            1 => self.below(64),
            _ => self.next(),
        }
    }
}
