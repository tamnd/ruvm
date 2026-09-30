// SPDX-License-Identifier: MIT OR Apache-2.0

//! Guest memory, helpers, exits and vector ops.

mod common;

use common::{ENV_SIZE, run_both};
use ruvm_jit_core::helpers::lookup_tb_ptr;
use ruvm_jit_core::ir::{FuncConfig, HelperType};
use ruvm_jit_core::liveness::LogMask;
use ruvm_jit_core::types::{call_flags, tb_exit};
use ruvm_jit_core::{Cond, Func, HelperInfo, MemOp, Type};
use ruvm_jit_interp::{
    Exit, FaultKind, FlatMemory, HelperEnv, HelperRegistry, InterpError, Machine, NoMemory, Unwind,
    run_tb,
};

const MEM_BASE: u64 = 0x1_0000;

fn new_func() -> Func {
    Func::new(FuncConfig::default())
}

fn run_mem(f: &Func, env: &mut [u8], mem: &mut FlatMemory) -> Exit {
    f.verify().expect("verify");
    let mut g = f.clone();
    g.gen_code(true, LogMask::default());
    let mut env2 = env.to_vec();
    let mut mem2 = mem.clone();
    let x0 = run_tb(f, env, mem).expect("run");
    let x1 = run_tb(&g, &mut env2, &mut mem2).expect("run optimized");
    assert_eq!(x0, x1);
    assert_eq!(env, &env2[..]);
    assert_eq!(mem, &mem2);
    x0
}

fn rd64(env: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(env[off..off + 8].try_into().unwrap())
}

#[test]
fn qemu_ld_st_sizes_and_endianness() {
    let mut mem = FlatMemory::new(MEM_BASE, 64);
    for (i, b) in mem.bytes.iter_mut().enumerate() {
        *b = 0x80 + i as u8;
    }
    let cases: [(MemOp, u64); 10] = [
        (MemOp::UB, 0x80),
        (MemOp::SB, 0xffff_ffff_ffff_ff80),
        (MemOp::UW, 0x8180),
        (MemOp::SW.or(MemOp::BE), 0xffff_ffff_ffff_8081),
        (MemOp::UL, 0x8382_8180),
        (MemOp::UL.or(MemOp::BE), 0x8081_8283),
        (MemOp::SL, 0xffff_ffff_8382_8180),
        (MemOp::UQ, 0x8786_8584_8382_8180),
        (MemOp::UQ.or(MemOp::BE), 0x8081_8283_8485_8687),
        (MemOp::UW.or(MemOp::BE), 0x8081),
    ];
    for (mop, want) in cases {
        let mut f = new_func();
        let env = f.env();
        let r = f.global_mem_new_i64(env, 0x100, "r");
        let a = f.constant_i64(MEM_BASE as i64);
        f.gen_qemu_ld_i64(r, a, 0, mop);
        f.gen_exit_tb(0, 0);
        let mut e = vec![0u8; ENV_SIZE];
        run_mem(&f, &mut e, &mut mem.clone());
        assert_eq!(rd64(&e, 0x100), want, "{mop:?}");
    }

    // i32 loads of a signed halfword sign extend to 32 bits only.
    let mut f = new_func();
    let env = f.env();
    let r = f.global_mem_new_i32(env, 0x100, "r");
    let a = f.constant_i64(MEM_BASE as i64);
    f.gen_qemu_ld_i32(r, a, 0, MemOp::SW);
    f.gen_exit_tb(0, 0);
    let mut e = vec![0u8; ENV_SIZE];
    run_mem(&f, &mut e, &mut mem.clone());
    assert_eq!(rd64(&e, 0x100), 0xffff_8180);

    // Stores in both byte orders.
    let mut f = new_func();
    let v = f.constant_i64(0x1122_3344_5566_7788);
    let a0 = f.constant_i64(MEM_BASE as i64);
    let a1 = f.constant_i64(MEM_BASE as i64 + 8);
    let a2 = f.constant_i64(MEM_BASE as i64 + 16);
    f.gen_qemu_st_i64(v, a0, 0, MemOp::UQ);
    f.gen_qemu_st_i64(v, a1, 0, MemOp::UQ.or(MemOp::BE));
    f.gen_qemu_st_i64(v, a2, 0, MemOp::UW.or(MemOp::BE));
    f.gen_exit_tb(0, 0);
    let mut m = FlatMemory::new(MEM_BASE, 32);
    run_mem(&f, &mut vec![0u8; ENV_SIZE], &mut m);
    assert_eq!(&m.bytes[..8], &0x1122_3344_5566_7788u64.to_le_bytes());
    assert_eq!(&m.bytes[8..16], &0x1122_3344_5566_7788u64.to_be_bytes());
    assert_eq!(&m.bytes[16..19], &[0x77, 0x88, 0]);
}

#[test]
fn qemu_ld_st_i128() {
    let mut f = new_func();
    let env = f.env();
    let lo = f.global_mem_new_i64(env, 0x100, "lo");
    let hi = f.global_mem_new_i64(env, 0x108, "hi");
    let t = f.temp_new_i128();
    let a = f.constant_i64(MEM_BASE as i64);
    let b = f.constant_i64(MEM_BASE as i64 + 16);
    f.gen_qemu_ld_i128(t, a, 0, MemOp::UO);
    f.gen_qemu_st_i128(t, b, 0, MemOp::UO.or(MemOp::BE));
    f.gen_mov_i64(lo, t.low());
    f.gen_mov_i64(hi, t.high());
    f.gen_exit_tb(0, 0);
    let mut m = FlatMemory::new(MEM_BASE, 32);
    for (i, b) in m.bytes[..16].iter_mut().enumerate() {
        *b = i as u8;
    }
    let mut e = vec![0u8; ENV_SIZE];
    run_mem(&f, &mut e, &mut m);
    let v = u128::from_le_bytes(m.bytes[..16].try_into().unwrap());
    assert_eq!(rd64(&e, 0x100), v as u64);
    assert_eq!(rd64(&e, 0x108), (v >> 64) as u64);
    let mut rev: Vec<u8> = m.bytes[..16].to_vec();
    rev.reverse();
    assert_eq!(&m.bytes[16..], &rev[..]);
}

#[test]
fn memory_faults_unwind() {
    let mut f = new_func();
    let env = f.env();
    let r = f.global_mem_new_i64(env, 0x100, "r");
    f.gen_movi_i64(r, 7);
    let a = f.constant_i64(MEM_BASE as i64 + 1);
    f.gen_qemu_ld_i64(r, a, 3, MemOp::UL.or(MemOp::ALIGN));
    f.gen_movi_i64(r, 9);
    f.gen_exit_tb(0, 0);
    let mut e = vec![0u8; ENV_SIZE];
    let mut m = FlatMemory::new(MEM_BASE, 16);
    let x = run_mem(&f, &mut e, &mut m);
    match x {
        Exit::Unwind(Unwind::Mem(fault)) => {
            assert_eq!(fault.kind, FaultKind::Unaligned);
            assert_eq!(fault.addr, MEM_BASE + 1);
            assert!(!fault.write);
            assert_eq!(fault.oi.mmu_idx(), 3);
        }
        x => panic!("unexpected exit {x:?}"),
    }
    // The global was synced before the load could fault.
    assert_eq!(rd64(&e, 0x100), 7);

    let x = run_tb(&f, &mut vec![0u8; ENV_SIZE], &mut NoMemory).unwrap();
    assert!(matches!(x, Exit::Unwind(Unwind::Mem(_))));
    let mut f = new_func();
    let a = f.constant_i64(0);
    let v = f.constant_i32(1);
    f.gen_qemu_st_i32(v, a, 0, MemOp::UL);
    f.gen_exit_tb(0, 0);
    match run_tb(&f, &mut vec![0u8; ENV_SIZE], &mut FlatMemory::new(MEM_BASE, 16)).unwrap() {
        Exit::Unwind(Unwind::Mem(fault)) => {
            assert_eq!(fault.kind, FaultKind::Unmapped);
            assert!(fault.write);
        }
        x => panic!("unexpected exit {x:?}"),
    }
}

#[test]
fn atomics() {
    for parallel in [false, true] {
        let mut f = Func::new(FuncConfig { parallel, ..FuncConfig::default() });
        let env = f.env();
        let r0 = f.global_mem_new_i64(env, 0x100, "r0");
        let r1 = f.global_mem_new_i64(env, 0x108, "r1");
        let r2 = f.global_mem_new_i64(env, 0x110, "r2");
        let r3 = f.global_mem_new_i32(env, 0x118, "r3");
        let a = f.constant_i64(MEM_BASE as i64);
        let v = f.constant_i64(5);
        f.gen_atomic_fetch_add_i64(r0, a, v, 0, MemOp::UQ);
        let cmp = f.constant_i64(15);
        let new = f.constant_i64(100);
        f.gen_atomic_cmpxchg_i64(r1, a, cmp, new, 0, MemOp::UQ);
        f.gen_atomic_xchg_i64(r2, a, v, 0, MemOp::UQ);
        let a4 = f.constant_i64(MEM_BASE as i64 + 8);
        let m = f.constant_i32(-2);
        f.gen_atomic_fetch_smax_i32(r3, a4, m, 0, MemOp::SL);
        f.gen_exit_tb(0, 0);
        let mut mem = FlatMemory::new(MEM_BASE, 16);
        mem.bytes[0] = 10;
        mem.bytes[8..12].copy_from_slice(&(-7i32).to_le_bytes());
        let mut e = vec![0u8; ENV_SIZE];
        run_mem(&f, &mut e, &mut mem);
        assert_eq!(rd64(&e, 0x100), 10);
        assert_eq!(rd64(&e, 0x108), 15);
        assert_eq!(rd64(&e, 0x110), 100);
        assert_eq!(rd64(&e, 0x118) as u32, -7i32 as u32);
        assert_eq!(&mem.bytes[..8], &5u64.to_le_bytes());
        assert_eq!(&mem.bytes[8..12], &(-2i32).to_le_bytes());
    }
}

fn helper_add3(_: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok((a[0] as u32).wrapping_add(a[1] as u32).wrapping_add(a[2] as u32) as u128)
}

fn helper_store_env(e: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    // Writes to a global behind the IR's back, which is allowed without NO_WG.
    let off = 0x100 + a[0] as usize;
    e.env[off..off + 8].copy_from_slice(&a[1].to_le_bytes());
    Ok(0)
}

fn helper_raise(_: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Err(Unwind::Exception(a[1]))
}

fn helper_wide(_: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    let v = a[0] as u128 | (a[1] as u128) << 64;
    Ok(v.rotate_left(8))
}

#[test]
fn helper_calls() {
    let add3 = HelperInfo::new(
        "add3",
        call_flags::NO_RWG | call_flags::NO_SE,
        HelperType::I32,
        &[HelperType::I32, HelperType::I32, HelperType::I32],
    );
    let store =
        HelperInfo::new("store_env", 0, HelperType::Void, &[HelperType::Ptr, HelperType::I64]);
    let wide = HelperInfo::new("wide", call_flags::NO_RWG, HelperType::I128, &[HelperType::I128]);
    let mut reg = HelperRegistry::new();
    reg.register_info(&add3, helper_add3);
    reg.register_info(&store, helper_store_env);
    reg.register_info(&wide, helper_wide);

    let mut f = new_func();
    let env = f.env();
    let g = f.global_mem_new_i64(env, 0x100, "g");
    let o = f.global_mem_new_i32(env, 0x108, "o");
    let lo = f.global_mem_new_i64(env, 0x110, "lo");
    let hi = f.global_mem_new_i64(env, 0x118, "hi");
    f.gen_movi_i64(g, 1);
    let h = f.helper(store);
    let v = f.constant_i64(0x55);
    f.gen_call(h, None, &[env.temp(), v.temp()]);
    // The helper wrote g in env, so the IR must see 0x55 here, not 1.
    f.gen_addi_i64(g, g, 1);
    let h = f.helper(add3);
    let (a, b, c) = (f.constant_i32(1), f.constant_i32(2), f.constant_i32(-4));
    f.gen_call(h, Some(o.temp()), &[a.temp(), b.temp(), c.temp()]);
    let h = f.helper(wide);
    let t = f.temp_new_i128();
    f.gen_movi_i64(t.low(), 0x0102_0304_0506_0708);
    f.gen_movi_i64(t.high(), 0x1112_1314_1516_1718);
    let t2 = f.temp_new_i128();
    f.gen_call(h, Some(t2.temp()), &[t.temp()]);
    f.gen_mov_i64(lo, t2.low());
    f.gen_mov_i64(hi, t2.high());
    f.gen_exit_tb(0, 0);
    f.verify().unwrap();

    for opt in [false, true] {
        let mut g = f.clone();
        if opt {
            g.gen_code(true, LogMask::default());
        }
        let mut e = vec![0u8; ENV_SIZE];
        let mut nm = NoMemory;
        let mut m = Machine::new(&mut e, &mut nm, &reg);
        assert_eq!(m.run(&g).unwrap(), Exit::ExitTb(0));
        assert_eq!(rd64(&e, 0x100), 0x56, "optimized: {opt}");
        assert_eq!(rd64(&e, 0x108) as u32, u32::MAX);
        assert_eq!(rd64(&e, 0x110), 0x0203_0405_0607_0811);
        assert_eq!(rd64(&e, 0x118), 0x1213_1415_1617_1801);
    }

    // An unregistered helper and a mismatched signature are setup errors.
    let empty = HelperRegistry::empty();
    let mut e = vec![0u8; ENV_SIZE];
    let err = Machine::new(&mut e, &mut NoMemory, &empty).run(&f).unwrap_err();
    assert!(matches!(err, InterpError::UnknownHelper(_)), "{err:?}");
    let mut bad = HelperRegistry::new();
    bad.register("store_env", HelperType::Void, &[HelperType::Ptr], helper_store_env);
    let err = Machine::new(&mut e, &mut NoMemory, &bad).run(&f).unwrap_err();
    assert_eq!(err, InterpError::HelperSignature("store_env".into()));
}

#[test]
fn helper_exception_unwinds() {
    let raise = HelperInfo::new(
        "raise",
        call_flags::NO_RETURN,
        HelperType::Void,
        &[HelperType::Ptr, HelperType::I32],
    );
    let mut reg = HelperRegistry::new();
    reg.register_info(&raise, helper_raise);
    let mut f = new_func();
    let env = f.env();
    let g = f.global_mem_new_i64(env, 0x100, "g");
    f.gen_movi_i64(g, 3);
    let h = f.helper(raise);
    let c = f.constant_i32(13);
    f.gen_call(h, None, &[env.temp(), c.temp()]);
    f.gen_movi_i64(g, 4);
    f.gen_exit_tb(0, 0);
    for opt in [false, true] {
        let mut g = f.clone();
        if opt {
            g.gen_code(true, LogMask::default());
        }
        let mut e = vec![0u8; ENV_SIZE];
        let x = Machine::new(&mut e, &mut NoMemory, &reg).run(&g).unwrap();
        assert_eq!(x, Exit::Unwind(Unwind::Exception(13)));
        assert_eq!(rd64(&e, 0x100), 3);
    }
}

#[test]
fn exits() {
    // exit_tb carries the block pointer plus the index.
    let mut f = new_func();
    f.gen_exit_tb(0x1000, 1);
    assert_eq!(run_tb(&f, &mut [0u8; 16], &mut NoMemory).unwrap(), Exit::ExitTb(0x1001));

    let mut f = new_func();
    f.gen_exit_tb(0x1000, tb_exit::REQUESTED);
    assert_eq!(run_tb(&f, &mut [0u8; 16], &mut NoMemory).unwrap(), Exit::ExitTb(0x1003));

    // goto_tb falls through when unlinked and leaves when linked.
    let mut f = new_func();
    f.gen_goto_tb(1);
    f.gen_exit_tb(0x2000, 1);
    assert_eq!(run_tb(&f, &mut [0u8; 16], &mut NoMemory).unwrap(), Exit::ExitTb(0x2001));
    let reg = HelperRegistry::new();
    let mut e = [0u8; 16];
    let mut nm = NoMemory;
    let mut m = Machine::new(&mut e, &mut nm, &reg);
    m.linked[1] = true;
    assert_eq!(m.run(&f).unwrap(), Exit::GotoTb(1));

    // lookup_and_goto_ptr calls lookup_tb_ptr, which finds nothing.
    let mut f = new_func();
    f.gen_lookup_and_goto_ptr();
    f.verify().unwrap();
    assert_eq!(run_tb(&f, &mut [0u8; 16], &mut NoMemory).unwrap(), Exit::GotoPtr(0));
    let mut reg = HelperRegistry::new();
    reg.register_info(&lookup_tb_ptr(), |_, _| Ok(0xabc0));
    let mut e = [0u8; 16];
    assert_eq!(Machine::new(&mut e, &mut NoMemory, &reg).run(&f).unwrap(), Exit::GotoPtr(0xabc0));

    // CF_NO_GOTO_PTR turns it into exit_tb 0.
    let mut f = Func::new(FuncConfig { no_goto_ptr: true, ..FuncConfig::default() });
    f.gen_lookup_and_goto_ptr();
    assert_eq!(run_tb(&f, &mut [0u8; 16], &mut NoMemory).unwrap(), Exit::ExitTb(0));
}

#[test]
fn run_errors_and_insn_start() {
    let mut f = new_func();
    f.gen_insn_start(&[0x40_0000, 7, 0]);
    let t = f.temp_new_i64();
    f.gen_movi_i64(t, 1);
    f.gen_insn_start(&[0x40_0004, 8, 0]);
    let reg = HelperRegistry::new();
    let mut e = [0u8; 16];
    let mut nm = NoMemory;
    let mut m = Machine::new(&mut e, &mut nm, &reg);
    assert_eq!(m.run(&f).unwrap_err(), InterpError::FellOffEnd);
    assert_eq!(m.last_insn_start, Some([0x40_0004, 8, 0]));

    let mut f = new_func();
    let l = f.new_label();
    f.gen_set_label(l);
    f.gen_br(l);
    let mut nm = NoMemory;
    let mut m = Machine::new(&mut e, &mut nm, &reg);
    m.step_limit = 1000;
    assert_eq!(m.run(&f).unwrap_err(), InterpError::StepLimit);

    let mut f = new_func();
    let env = f.env();
    let g = f.global_mem_new_i64(env, 0x1000, "far");
    f.gen_movi_i64(g, 1);
    f.gen_exit_tb(0, 0);
    let err = run_tb(&f, &mut [0u8; 16], &mut NoMemory).unwrap_err();
    assert!(matches!(err, InterpError::EnvOutOfBounds { offset: 0x1000, len: 8 }), "{err:?}");
}

#[test]
fn indirect_globals() {
    // A global based on another global pointer, lowered by liveness pass 2.
    let mut f = new_func();
    let env = f.env();
    let base = f.global_mem_new_ptr(env, 0x100, "base");
    let x = f.global_mem_new_i64(base, 0x10, "x");
    let y = f.global_mem_new_i64(env, 0x108, "y");
    f.gen_addi_i64(x, x, 5);
    f.gen_mov_i64(y, x);
    f.gen_addi_i64(x, x, 1);
    f.gen_exit_tb(0, 0);
    let mut e = vec![0u8; ENV_SIZE];
    e[0x100..0x108].copy_from_slice(&0x200u64.to_le_bytes());
    e[0x210..0x218].copy_from_slice(&10u64.to_le_bytes());
    let e = run_both(&f, &e);
    assert_eq!(rd64(&e, 0x108), 15);
    assert_eq!(rd64(&e, 0x210), 16);
}

fn vec_eval(ty: Type, build: impl Fn(&mut Func, ruvm_jit_core::ir::TempVec)) -> [u8; 32] {
    let mut f = new_func();
    let r = f.temp_new_vec(ty);
    build(&mut f, r);
    let env = f.env();
    f.gen_st_vec(r, env, 0x100);
    f.gen_exit_tb(0, 0);
    let mut e = vec![0u8; ENV_SIZE];
    for (i, b) in e[0x200..0x280].iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(37).wrapping_add(0x81);
    }
    let e = run_both(&f, &e);
    e[0x100..0x120].try_into().unwrap()
}

#[test]
fn vector_ops() {
    let src = |i: usize| (i as u8).wrapping_mul(37).wrapping_add(0x81);
    for ty in [Type::V64, Type::V128, Type::V256] {
        let n = ty.size() as usize;
        for vece in 0..4u32 {
            let es = 1usize << vece;
            let lane = |b: &[u8], i: usize| {
                let mut x = [0u8; 8];
                x[..es].copy_from_slice(&b[i * es..i * es + es]);
                u64::from_le_bytes(x)
            };
            let bits = 8 * es as u32;
            let m = if bits == 64 { !0 } else { (1u64 << bits) - 1 };
            let sx = |x: u64| ((x << (64 - bits)) as i64) >> (64 - bits);
            let a: Vec<u8> = (0..32).map(src).collect();
            let b: Vec<u8> = (32..64).map(src).collect();
            let two = |build: &dyn Fn(&mut Func, _, _, _)| {
                vec_eval(ty, |f, r| {
                    let env = f.env();
                    let x = f.temp_new_vec(ty);
                    let y = f.temp_new_vec(ty);
                    f.gen_ld_vec(x, env, 0x200);
                    f.gen_ld_vec(y, env, 0x220);
                    build(f, r, x, y);
                })
            };
            let check = |name: &str, got: [u8; 32], want: &dyn Fn(u64, u64) -> u64| {
                for i in 0..n / es {
                    let w = want(lane(&a, i), lane(&b, i)) & m;
                    assert_eq!(lane(&got, i), w, "{name} {ty:?} vece {vece} lane {i}");
                }
            };
            check("add", two(&|f, r, x, y| f.gen_add_vec(vece, r, x, y)), &|x, y| {
                x.wrapping_add(y)
            });
            check("sub", two(&|f, r, x, y| f.gen_sub_vec(vece, r, x, y)), &|x, y| {
                x.wrapping_sub(y)
            });
            check("mul", two(&|f, r, x, y| f.gen_mul_vec(vece, r, x, y)), &|x, y| {
                x.wrapping_mul(y)
            });
            check("xor", two(&|f, r, x, y| f.gen_xor_vec(vece, r, x, y)), &|x, y| x ^ y);
            check("andc", two(&|f, r, x, y| f.gen_andc_vec(vece, r, x, y)), &|x, y| x & !y);
            check("umin", two(&|f, r, x, y| f.gen_umin_vec(vece, r, x, y)), &|x, y| x.min(y));
            check("smax", two(&|f, r, x, y| f.gen_smax_vec(vece, r, x, y)), &|x, y| {
                sx(x).max(sx(y)) as u64
            });
            check("usadd", two(&|f, r, x, y| f.gen_usadd_vec(vece, r, x, y)), &|x, y| {
                x.saturating_add(y).min(m)
            });
            check("sssub", two(&|f, r, x, y| f.gen_sssub_vec(vece, r, x, y)), &|x, y| {
                let lim = 1i128 << (bits - 1);
                (sx(x) as i128 - sx(y) as i128).clamp(-lim, lim - 1) as u64
            });
            check("shlv", two(&|f, r, x, y| f.gen_shlv_vec(vece, r, x, y)), &|x, y| {
                x << (y & (bits as u64 - 1))
            });
            check("sarv", two(&|f, r, x, y| f.gen_sarv_vec(vece, r, x, y)), &|x, y| {
                (sx(x) >> (y & (bits as u64 - 1))) as u64
            });
            check("rotrv", two(&|f, r, x, y| f.gen_rotrv_vec(vece, r, x, y)), &|x, y| {
                let s = (y & (bits as u64 - 1)) as u32;
                if s == 0 { x } else { (x >> s) | (x << (bits - s)) }
            });
            let s = bits - 1;
            check("shri", two(&|f, r, x, _| f.gen_shri_vec(vece, r, x, s as i64)), &|x, _| x >> s);
            check(
                "sars",
                two(&|f, r, x, _| {
                    let c = f.constant_i32(3);
                    f.gen_sars_vec(vece, r, x, c)
                }),
                &|x, _| (sx(x) >> 3) as u64,
            );
            check("neg", two(&|f, r, x, _| f.gen_neg_vec(vece, r, x)), &|x, _| x.wrapping_neg());
            check("abs", two(&|f, r, x, _| f.gen_abs_vec(vece, r, x)), &|x, _| {
                sx(x).unsigned_abs()
            });
            for c in [Cond::Eq, Cond::Lt, Cond::Gtu, Cond::Leu] {
                let want = |x: u64, y: u64| match c {
                    Cond::Eq => x == y,
                    Cond::Lt => sx(x) < sx(y),
                    Cond::Gtu => x > y,
                    _ => x <= y,
                };
                check("cmp", two(&|f, r, x, y| f.gen_cmp_vec(c, vece, r, x, y)), &|x, y| {
                    if want(x, y) { m } else { 0 }
                });
                check(
                    "cmpsel",
                    two(&|f, r, x, y| f.gen_cmpsel_vec(c, vece, r, x, y, y, x)),
                    &|x, y| if want(x, y) { y } else { x },
                );
            }
            check(
                "dupi",
                two(&|f, r, _, _| f.gen_dupi_vec(vece, r, 0x1234_5678_9abc_def0)),
                &|_, _| 0x1234_5678_9abc_def0,
            );
            check(
                "dup_mem",
                two(&|f, r, _, _| {
                    let env = f.env();
                    f.gen_dup_mem_vec(vece, r, env, 0x200)
                }),
                &|_, _| lane(&a, 0),
            );
            check(
                "bitsel",
                two(&|f, r, x, y| {
                    let c = f.constant_vec(ty, 0, 0x0f);
                    f.gen_bitsel_vec(vece, r, c, x, y)
                }),
                &|x, y| {
                    let k = 0x0f0f_0f0f_0f0f_0f0f;
                    (x & k) | (y & !k)
                },
            );
            // The bytes past the vector's size are not written.
            let got = two(&|f, r, x, y| f.gen_or_vec(vece, r, x, y));
            assert!(got[n..].iter().all(|&b| b == 0));
        }
    }
}
