// SPDX-License-Identifier: GPL-2.0-or-later

//! Block chaining: `goto_tb` patched to jump straight to the next block, `lookup_and_goto_ptr`
//! jumping to the block a [`Chain`] finds, the `icount_decr` check reading the shared word, and
//! the region rules (blocks jump back into older regions of their chain, never forward and
//! never into another chain).
//!
//! Generated code only runs on an AArch64 host; elsewhere these tests only compile and link.

use std::any::Any;
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Weak};

use ruvm_jit_aarch64::{Chain, ChainExit, CodeRegion, CodegenOptions, CompileOptions, CompiledTb};
use ruvm_jit_core::ir::FuncConfig;
use ruvm_jit_core::{Cond, Func};
use ruvm_jit_interp::{Exit, FlatMemory, HelperEnv, HelperRegistry, Unwind};

const NATIVE: bool = cfg!(all(unix, target_arch = "aarch64"));
const ENV_SIZE: usize = 0x400;
const G0: i64 = 0x100;
const G1: i64 = 0x108;
const G2: i64 = 0x110;
/// Where the blocks load `icount_decr` from.
const DECR: i64 = 0x3f8;

/// A block and its name, owned the way a runtime owns its blocks.
struct Blk {
    name: &'static str,
    code: CompiledTb,
}

fn compile(r: &Arc<CodeRegion>, name: &'static str, f: &Func) -> Arc<Blk> {
    let opts = CompileOptions { icount_decr_offset: Some(DECR), ..CompileOptions::default() };
    let code = r.compile_chained(f, &CodegenOptions::default(), &opts).expect("compile");
    let b = Arc::new(Blk { name, code });
    let owner: Weak<dyn Any + Send + Sync> = Arc::downgrade(&b) as Weak<Blk>;
    b.code.set_owner(owner);
    b
}

/// `lookup_tb_ptr`: `then` while the counter at `G1` is below `limit`, then nothing.
struct TestChain {
    then: Arc<Blk>,
    limit: u64,
}

impl Chain for TestChain {
    fn lookup_tb_ptr(
        &self,
        he: &mut HelperEnv<'_>,
    ) -> Result<Option<Arc<dyn Any + Send + Sync>>, Unwind> {
        let n = rd64(he.env, G1);
        Ok((n < self.limit).then(|| Arc::clone(&self.then) as Arc<dyn Any + Send + Sync>))
    }

    fn code<'a>(&self, block: &'a (dyn Any + Send + Sync)) -> Option<&'a CompiledTb> {
        block.downcast_ref::<Blk>().map(|b| &b.code)
    }
}

fn rd64(env: &[u8], off: i64) -> u64 {
    let off = off as usize;
    u64::from_le_bytes(env[off..off + 8].try_into().unwrap())
}

fn name(b: &Option<Arc<dyn Any + Send + Sync>>) -> Option<&'static str> {
    b.as_ref().and_then(|b| b.downcast_ref::<Blk>()).map(|b| b.name)
}

/// `a`: leave with exit index 3 if `icount_decr` is negative, else count in `G0` and take
/// `goto_tb` 0.
fn block_a() -> Func {
    let mut f = Func::new(FuncConfig::default());
    let env = f.env();
    let g0 = f.global_mem_new_i64(env, G0, "g0");
    let t = f.temp_new_i32();
    f.gen_ld_i32(t, env, DECR);
    let out = f.new_label();
    f.gen_brcondi_i32(Cond::Lt, t, 0, out);
    f.gen_addi_i64(g0, g0, 1);
    f.gen_goto_tb(0);
    f.gen_exit_tb(0x1000, 0);
    f.gen_set_label(out);
    f.gen_exit_tb(0x1000, 3);
    f
}

/// `b`: count in `G1` and `lookup_and_goto_ptr`.
fn block_b() -> Func {
    let mut f = Func::new(FuncConfig::default());
    let env = f.env();
    let g1 = f.global_mem_new_i64(env, G1, "g1");
    f.gen_addi_i64(g1, g1, 1);
    f.gen_lookup_and_goto_ptr();
    f
}

/// `c`: add `G0` to `G2` and take `goto_tb` 1.
fn block_c() -> Func {
    let mut f = Func::new(FuncConfig::default());
    let env = f.env();
    let g0 = f.global_mem_new_i64(env, G0, "g0");
    let g2 = f.global_mem_new_i64(env, G2, "g2");
    f.gen_add_i64(g2, g2, g0);
    f.gen_goto_tb(1);
    f.gen_exit_tb(0x3000, 1);
    f
}

/// Just `goto_tb` 0, falling through to `exit_tb` when unlinked.
fn block_jump(tb: u64) -> Func {
    let mut f = Func::new(FuncConfig::default());
    f.gen_goto_tb(0);
    f.gen_exit_tb(tb, 0);
    f
}

fn run(b: &Blk, env: &mut [u8], chain: &dyn Chain, decr: u32) -> ChainExit {
    let mut mem = FlatMemory::new(0x1_0000, 16);
    let decr = AtomicU32::new(decr);
    b.code.run_chained(env, &mut mem, &HelperRegistry::new(), chain, &decr).expect("run")
}

#[test]
fn a_loop_of_blocks_runs_without_returning() {
    let r = CodeRegion::new(1 << 20).unwrap();
    let a = compile(&r, "a", &block_a());
    let b = compile(&r, "b", &block_b());
    let c = compile(&r, "c", &block_c());
    assert!(a.code.set_goto_tb_target(0, Some(&b.code)));
    assert!(c.code.set_goto_tb_target(1, Some(&a.code)));
    assert!(!c.code.set_goto_tb_target(0, Some(&a.code)));
    let chain = TestChain { then: Arc::clone(&c), limit: 1000 };
    if !NATIVE {
        return;
    }

    // The loop a, b, c runs until the lookup in b finds nothing. The CPU state's copy of
    // icount_decr is negative, but the blocks read the shared word, which is 0.
    let mut env = vec![0u8; ENV_SIZE];
    env[DECR as usize..DECR as usize + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    let x = run(&a, &mut env, &chain, 0);
    assert_eq!(x.exit, Exit::GotoPtr(0));
    assert_eq!(name(&x.block), Some("b"));
    assert!(x.next.is_none());
    assert_eq!((rd64(&env, G0), rd64(&env, G1), rd64(&env, G2)), (1000, 1000, 999 * 1000 / 2));

    // An exit request in the shared word stops the loop at the start of a.
    let mut env = vec![0u8; ENV_SIZE];
    let x = run(&a, &mut env, &chain, u32::MAX);
    assert_eq!(x.exit, Exit::ExitTb(0x1003));
    assert_eq!(name(&x.block), Some("a"));
    assert_eq!(rd64(&env, G0), 0);

    // Unlinking a slot makes the block fall through to its exit_tb.
    assert!(c.code.set_goto_tb_target(1, None));
    let mut env = vec![0u8; ENV_SIZE];
    let x = run(&a, &mut env, &chain, 0);
    assert_eq!(x.exit, Exit::ExitTb(0x3001));
    assert_eq!(name(&x.block), Some("c"));
    assert_eq!((rd64(&env, G0), rd64(&env, G1), rd64(&env, G2)), (1, 1, 1));

    // A slot linked only to the exit stub leaves with the slot number.
    assert!(c.code.set_goto_tb_linked(1, true));
    let x = run(&a, &mut vec![0u8; ENV_SIZE], &chain, 0);
    assert_eq!(x.exit, Exit::GotoTb(1));
    assert_eq!(name(&x.block), Some("c"));
}

#[test]
fn jumps_stay_within_a_chain_of_regions() {
    let r = CodeRegion::new(1 << 20).unwrap();
    let later = r.successor(1 << 20).unwrap();
    let other = CodeRegion::new(1 << 20).unwrap();
    let x = compile(&r, "x", &block_jump(0x5000));
    let y = compile(&later, "y", &block_jump(0x6000));
    let z = compile(&other, "z", &block_jump(0x7000));
    let b = compile(&r, "b", &block_b());
    // A later region may jump back into x; x may not jump forward into y, or into z of
    // another chain, so those slots go to the exit stub.
    assert!(y.code.set_goto_tb_target(0, Some(&x.code)));
    assert!(x.code.set_goto_tb_target(0, Some(&y.code)));
    if NATIVE {
        let none = TestChain { then: Arc::clone(&x), limit: 0 };
        let e = run(&y, &mut vec![0u8; ENV_SIZE], &none, 0);
        assert_eq!((e.exit, name(&e.block)), (Exit::GotoTb(0), Some("x")));
    }
    assert!(x.code.set_goto_tb_target(0, Some(&z.code)));
    if NATIVE {
        let none = TestChain { then: Arc::clone(&x), limit: 0 };
        let e = run(&x, &mut vec![0u8; ENV_SIZE], &none, 0);
        assert_eq!((e.exit, name(&e.block)), (Exit::GotoTb(0), Some("x")));
    }

    // goto_ptr to a block the code may not reach comes back with that block as the next one.
    if NATIVE {
        for target in [&y, &z] {
            let chain = TestChain { then: Arc::clone(target), limit: 10 };
            let e = run(&b, &mut vec![0u8; ENV_SIZE], &chain, 0);
            assert_eq!((e.exit, name(&e.block)), (Exit::GotoPtr(0), Some("b")));
            assert_eq!(name(&e.next), Some(target.name));
        }
    }
}
