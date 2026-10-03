// SPDX-License-Identifier: GPL-2.0-or-later

//! What the fence mappings lower to: the `dmb` variants for barriers, `ldapr`, `ldar`,
//! `ldapurs*` and `stlr` for flagged accesses through the host window, the conservative
//! barriers on the service path, the barriers around helper calls and atomics, and that window
//! accesses read and write the same bytes as the reference interpreter.
//!
//! The instruction checks run on any host; the value checks only on an AArch64 one.

mod common;

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::{WordMemory, words};
use ruvm_jit_aarch64::{CodeRegion, CodegenOptions, CompiledTb, HostFeatures, HostWindow};
use ruvm_jit_core::ir::{FuncConfig, HelperType, TempI64};
use ruvm_jit_core::liveness::LogMask;
use ruvm_jit_core::memory_model::X86_TSO;
use ruvm_jit_core::tcg_op_ldst::AtomicOp;
use ruvm_jit_core::types::{call_flags, mo};
use ruvm_jit_core::{FenceMapping, Func, HelperInfo, MemOp};
use ruvm_jit_interp::{FlatMemory, HelperRegistry, Machine};

const NATIVE: bool = cfg!(all(unix, target_arch = "aarch64"));

const DMB_ISHLD: u32 = 0xd50339bf;
const DMB_ISHST: u32 = 0xd5033abf;
const DMB_ISH: u32 = 0xd5033bbf;
/// `blr x16`, the call to the service routine.
const BLR_X16: u32 = 0xd63f0200;

/// An instruction pattern: the bits under `mask` must equal `word`.
#[derive(Clone, Copy, Debug)]
struct Pat {
    word: u32,
    mask: u32,
}

/// Exactly `word`.
const fn exact(word: u32) -> Pat {
    Pat { word, mask: !0 }
}

/// `word` with any Rt and Rn.
const fn any_regs(word: u32) -> Pat {
    Pat { word, mask: !0x3ff }
}

const LDAPR_X: Pat = any_regs(0xf8bfc000);
const LDAPR_W: Pat = any_regs(0xb8bfc000);
const LDAPR_H: Pat = any_regs(0x78bfc000);
const LDAPR_B: Pat = any_regs(0x38bfc000);
const LDAR_X: Pat = any_regs(0xc8dffc00);
const STLR_X: Pat = any_regs(0xc89ffc00);
const STLR_W: Pat = any_regs(0x889ffc00);
const STLR_H: Pat = any_regs(0x489ffc00);
const STLR_B: Pat = any_regs(0x089ffc00);
/// `ldapursb xt`, offset 0.
const LDAPURSB_X: Pat = any_regs(0x19800000);
/// `ldapursh wt`, offset 0.
const LDAPURSH_W: Pat = any_regs(0x59c00000);
/// `ldapursw`, offset 0.
const LDAPURSW: Pat = any_regs(0x99800000);
/// Any `ldapr`, `ldar` or `stlr`.
const ORDERED: [Pat; 3] = [
    Pat { word: 0x38bfc000, mask: 0x3ffffc00 },
    Pat { word: 0x08dffc00, mask: 0x3ffffc00 },
    Pat { word: 0x089ffc00, mask: 0x3ffffc00 },
];

const BASE: u64 = 0x4000;
const ENV_SIZE: usize = 0x400;

fn insns(tb: &CompiledTb) -> Vec<u32> {
    tb.code().chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn count(code: &[u32], p: Pat) -> usize {
    code.iter().filter(|w| *w & p.mask == p.word & p.mask).count()
}

fn has(code: &[u32], p: Pat) -> bool {
    count(code, p) > 0
}

fn config(m: FenceMapping) -> FuncConfig {
    FuncConfig { parallel: true, guest_mo: X86_TSO, fence_mapping: m, ..FuncConfig::default() }
}

fn window(features: HostFeatures) -> CodegenOptions {
    CodegenOptions { guest_window: true, features }
}

fn compile(f: &Func, opts: &CodegenOptions) -> (CompiledTb, Vec<u32>) {
    let r = CodeRegion::new(1 << 16).expect("code region");
    let tb = match r.compile_with(f, opts) {
        Ok(tb) => tb,
        Err(e) => panic!("compile: {e}\n{}", f.dump_ops(false)),
    };
    let code = insns(&tb);
    (tb, code)
}

fn optimized(f: &Func) -> Func {
    let mut g = f.clone();
    g.gen_code(true, LogMask::default());
    g
}

/// A block loading 64 bits from guest `BASE` and storing them to `BASE + 8`.
fn load_store(m: FenceMapping) -> Func {
    let mut f = Func::new(config(m));
    let env = f.env();
    let v = f.global_mem_new_i64(env, 0, "v");
    let a = f.constant_i64(BASE as i64);
    let b = f.constant_i64(BASE as i64 + 8);
    f.gen_qemu_ld_i64(v, a, 0, MemOp::UQ);
    f.gen_qemu_st_i64(v, b, 0, MemOp::UQ);
    f.gen_exit_tb(0, 0);
    f
}

#[test]
fn barriers_lower_to_dmb() {
    let mut f = Func::new(FuncConfig::default());
    for bar in [
        mo::LD_LD,
        mo::LD_ST,
        mo::LD_LD | mo::LD_ST,
        mo::ST_ST,
        mo::ST_LD,
        mo::LD_ST | mo::ST_ST,
        mo::ALL,
        0,
    ] {
        f.gen_mb(bar | mo::BAR_SC);
    }
    f.gen_exit_tb(0, 0);
    let (_, code) = compile(&f, &CodegenOptions::default());
    let dmbs: Vec<u32> = code.iter().copied().filter(|w| *w & 0xfffff0ff == 0xd50330bf).collect();
    assert_eq!(
        dmbs,
        [DMB_ISHLD, DMB_ISHLD, DMB_ISHLD, DMB_ISHST, DMB_ISH, DMB_ISH, DMB_ISH, DMB_ISH]
    );
}

#[test]
fn qemu_mapping_fences_before_each_access() {
    let f = load_store(FenceMapping::Qemu);
    for opts in [CodegenOptions::default(), window(HostFeatures::ALL)] {
        let (_, code) = compile(&f, &opts);
        assert_eq!(count(&code, exact(DMB_ISHLD)), 1);
        assert_eq!(count(&code, exact(DMB_ISH)), 1);
        assert_eq!(count(&code, exact(DMB_ISHST)), 0);
        assert!(ORDERED.iter().all(|p| !has(&code, *p)));
    }
}

#[test]
fn risotto_folds_load_and_store_barriers() {
    let f = load_store(FenceMapping::Risotto);
    let opts = window(HostFeatures::ALL);
    let (_, code) = compile(&f, &opts);
    assert_eq!(count(&code, exact(DMB_ISHLD)), 1);
    assert_eq!(count(&code, exact(DMB_ISHST)), 1);
    assert_eq!(count(&code, exact(DMB_ISH)), 0);
    // The optimizer merges the two into one full barrier.
    let (_, code) = compile(&optimized(&f), &opts);
    assert_eq!(count(&code, exact(DMB_ISHLD)), 0);
    assert_eq!(count(&code, exact(DMB_ISHST)), 0);
    assert_eq!(count(&code, exact(DMB_ISH)), 1);
    assert!(ORDERED.iter().all(|p| !has(&code, *p)));
}

#[test]
fn rcpc_uses_ldapr_and_stlr() {
    let f = load_store(FenceMapping::AranciniRcpc);
    let (_, code) = compile(&optimized(&f), &window(HostFeatures::ALL));
    assert_eq!(count(&code, LDAPR_X), 1);
    assert_eq!(count(&code, STLR_X), 1);
    assert!(!has(&code, LDAR_X));
    // The window path needs no barrier; the service path keeps conservative ones.
    assert_eq!(count(&code, exact(DMB_ISHLD)), 1);
    assert_eq!(count(&code, exact(DMB_ISH)), 1);

    // Without FEAT_LRCPC the load is the stronger ldar.
    let (_, code) = compile(&f, &window(HostFeatures::BASELINE));
    assert_eq!(count(&code, LDAR_X), 1);
    assert!(!has(&code, LDAPR_X));
    assert_eq!(count(&code, STLR_X), 1);

    // Without the window every access goes to the service routine, fenced.
    let (_, code) = compile(&f, &CodegenOptions::default());
    assert!(ORDERED.iter().all(|p| !has(&code, *p)));
    let ld = code.iter().position(|w| *w == BLR_X16).expect("load call");
    let st = code.iter().rposition(|w| *w == BLR_X16).expect("store call");
    let ishld = code.iter().position(|w| *w == DMB_ISHLD).expect("dmb ishld");
    let ish = code.iter().position(|w| *w == DMB_ISH).expect("dmb ish");
    assert!(ld < ishld && ishld < ish && ish < st, "{ld} {ishld} {ish} {st}");
}

#[test]
fn rcpc_access_sizes() {
    let mut f = Func::new(config(FenceMapping::AranciniRcpc));
    let env = f.env();
    let v = f.global_mem_new_i64(env, 0, "v");
    let w = f.global_mem_new_i32(env, 8, "w");
    let a = f.constant_i64(BASE as i64);
    f.gen_qemu_ld_i32(w, a, 0, MemOp::UB);
    f.gen_qemu_ld_i32(w, a, 0, MemOp::UW);
    f.gen_qemu_ld_i32(w, a, 0, MemOp::UL);
    f.gen_qemu_ld_i64(v, a, 0, MemOp::SB);
    f.gen_qemu_ld_i32(w, a, 0, MemOp::SW);
    f.gen_qemu_ld_i64(v, a, 0, MemOp::SL);
    f.gen_qemu_st_i32(w, a, 0, MemOp::UB);
    f.gen_qemu_st_i32(w, a, 0, MemOp::UW);
    f.gen_qemu_st_i32(w, a, 0, MemOp::UL);
    f.gen_exit_tb(0, 0);

    let (_, code) = compile(&f, &window(HostFeatures::ALL));
    assert_eq!(count(&code, LDAPR_B), 1);
    assert_eq!(count(&code, LDAPR_H), 1);
    assert_eq!(count(&code, LDAPR_W), 1);
    assert_eq!(count(&code, LDAPURSB_X), 1);
    assert_eq!(count(&code, LDAPURSH_W), 1);
    assert_eq!(count(&code, LDAPURSW), 1);
    assert_eq!(count(&code, STLR_B), 1);
    assert_eq!(count(&code, STLR_H), 1);
    assert_eq!(count(&code, STLR_W), 1);

    // FEAT_LRCPC alone: plain ldapr, then a sign extension.
    let lrcpc = HostFeatures { lrcpc: true, lrcpc2: false };
    let (_, code) = compile(&f, &window(lrcpc));
    assert_eq!(count(&code, LDAPR_B), 2);
    assert_eq!(count(&code, LDAPR_H), 2);
    assert_eq!(count(&code, LDAPR_W), 2);
    assert!(!has(&code, LDAPURSB_X) && !has(&code, LDAPURSH_W) && !has(&code, LDAPURSW));
}

fn side_effect_helper() -> HelperInfo {
    HelperInfo::new("touch", 0, HelperType::Void, &[HelperType::Ptr])
}

fn pure_helper() -> HelperInfo {
    HelperInfo::new("pure", call_flags::NO_SIDE_EFFECTS, HelperType::I64, &[HelperType::I64])
}

#[test]
fn helper_calls_are_fenced_under_the_alternative_mappings() {
    for m in FenceMapping::ALL {
        let mut f = Func::new(config(m));
        let env = f.env();
        let h = f.helper(side_effect_helper());
        f.gen_call(h, None, &[env.temp()]);
        let p = f.helper(pure_helper());
        let v = f.global_mem_new_i64(env, 0, "v");
        f.gen_call(p, Some(v.temp()), &[v.temp()]);
        f.gen_exit_tb(0, 0);
        let (_, code) = compile(&f, &CodegenOptions::default());
        let calls: Vec<usize> =
            code.iter().enumerate().filter(|(_, w)| **w == BLR_X16).map(|(k, _)| k).collect();
        assert_eq!(calls.len(), 2);
        if m == FenceMapping::Qemu {
            assert!(!code.iter().any(|w| [DMB_ISHLD, DMB_ISHST, DMB_ISH].contains(w)));
            continue;
        }
        // The helper with side effects: ishst before, ishld right after.
        assert!(code[..calls[0]].contains(&DMB_ISHST), "{m}");
        assert_eq!(code[calls[0] + 1], DMB_ISHLD, "{m}");
        // The pure one is left alone.
        assert_eq!(count(&code, exact(DMB_ISHST)), 1, "{m}");
        assert_eq!(count(&code, exact(DMB_ISHLD)), 1, "{m}");
    }
}

/// A parallel block doing a compare and swap and an exchange at guest `BASE`.
fn atomics(m: FenceMapping) -> Func {
    let mut f = Func::new(config(m));
    let env = f.env();
    let old = f.global_mem_new_i64(env, 0, "old");
    let cmp = f.global_mem_new_i64(env, 8, "cmp");
    let new = f.global_mem_new_i64(env, 16, "new");
    let prev = f.global_mem_new_i64(env, 24, "prev");
    let a = f.constant_i64(BASE as i64);
    f.gen_atomic_cmpxchg_i64(old, a, cmp, new, 0, MemOp::UQ);
    let seven = f.constant_i64(7);
    f.gen_atomic_op_i64(AtomicOp::Xchg, prev, a, seven, 0, MemOp::UQ);
    f.gen_exit_tb(0, 0);
    f
}

#[test]
fn atomics_get_full_barriers() {
    for m in FenceMapping::ALL {
        let f = atomics(m);
        for g in [f.clone(), optimized(&f)] {
            let (tb, code) = compile(&g, &window(HostFeatures::ALL));
            let calls: Vec<usize> =
                code.iter().enumerate().filter(|(_, w)| **w == BLR_X16).map(|(k, _)| k).collect();
            assert_eq!(calls.len(), 2, "{m}");
            let fences: Vec<(usize, u32)> = code
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, w)| [DMB_ISHLD, DMB_ISHST, DMB_ISH].contains(w))
                .collect();
            if m == FenceMapping::Qemu {
                // QEMU puts no barrier around its atomic helpers.
                assert!(fences.is_empty(), "{m}: {fences:x?}");
            } else {
                // A full barrier before and after each helper, and no store barrier since the
                // full one already orders earlier stores.
                for c in &calls {
                    assert!(fences.iter().any(|&(k, w)| k < *c && w == DMB_ISH), "{m}");
                    assert!(fences.iter().any(|&(k, w)| k > *c && w == DMB_ISH), "{m}");
                }
                assert!(fences.iter().all(|&(_, w)| w != DMB_ISHST), "{m}: {fences:x?}");
            }
            if NATIVE {
                let mut env = vec![0u8; ENV_SIZE];
                env[8..16].copy_from_slice(&5u64.to_le_bytes());
                env[16..24].copy_from_slice(&9u64.to_le_bytes());
                let mut mem = FlatMemory::new(BASE, 16);
                mem.bytes[..8].copy_from_slice(&5u64.to_le_bytes());
                tb.run(&mut env, &mut mem, &HelperRegistry::new()).expect("run");
                assert_eq!(rd64(&env, 0), 5);
                assert_eq!(rd64(&env, 24), 9);
                assert_eq!(rd64(&mem.bytes, 0), 7);
            }
        }
    }
}

fn rd64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

/// Guest addresses: the window covers the first 64 bytes of a 128 byte memory.
const WINDOW_WORDS: usize = 8;
const MEM_WORDS: usize = 16;

/// A block of loads and stores of every size and extension at `offsets` from `BASE`, each
/// load result written to its own env slot.
fn accesses(m: FenceMapping, offsets: &[u64], align: MemOp) -> Func {
    let mut f = Func::new(config(m));
    let env = f.env();
    // 14 loads at each offset, and 4 after the stores.
    let n = 18 * offsets.len();
    assert!(8 * n <= ENV_SIZE);
    let regs: Vec<TempI64> = (0..n).map(|k| f.global_mem_new_i64(env, 8 * k as i64, "r")).collect();
    let mut regs = regs.into_iter();
    let mut g64 = |_: &mut Func| regs.next().expect("enough globals");
    let loads =
        [MemOp::UB, MemOp::SB, MemOp::UW, MemOp::SW, MemOp::UL, MemOp::SL, MemOp::UQ, MemOp::SQ];
    for &off in offsets {
        let a = f.constant_i64((BASE + off) as i64);
        for op in loads {
            let r = g64(&mut f);
            f.gen_qemu_ld_i64(r, a, 0, op.or(align));
            if op.size() < 3 {
                let t = f.temp_new_i32();
                f.gen_qemu_ld_i32(t, a, 0, op.or(align));
                let r = g64(&mut f);
                f.gen_ext_i32_i64(r, t);
            }
        }
    }
    // Store back a mix of values at each offset, then read them again.
    for (k, &off) in offsets.iter().enumerate() {
        let a = f.constant_i64((BASE + off) as i64);
        let v = f.constant_i64(0x8899_aabb_ccdd_eeffu64.rotate_left(8 * k as u32) as i64);
        let w = f.constant_i32(0x8182_8384u32.rotate_left(8 * k as u32) as i32);
        f.gen_qemu_st_i64(v, a, 0, MemOp::UQ.or(align));
        let r = g64(&mut f);
        f.gen_qemu_ld_i64(r, a, 0, MemOp::UQ.or(align));
        for size in [MemOp::UL, MemOp::UW, MemOp::UB] {
            f.gen_qemu_st_i32(w, a, 0, size.or(align));
            let r = g64(&mut f);
            f.gen_qemu_ld_i64(r, a, 0, MemOp::UQ.or(align));
        }
    }
    f.gen_exit_tb(0, 0);
    f
}

/// Run `f` in the interpreter and natively against a window, and check that they agree.
/// Returns the number of accesses that went to the service routine.
fn window_matches_interpreter(f: &Func, opts: &CodegenOptions) -> usize {
    let mut init = vec![0u8; 8 * MEM_WORDS];
    for (k, b) in init.iter_mut().enumerate() {
        *b = (k as u8).wrapping_mul(37) ^ 0x95;
    }
    let reg = HelperRegistry::new();
    let mut want_env = vec![0u8; ENV_SIZE];
    let mut want_mem = FlatMemory::new(BASE, init.len());
    want_mem.bytes.copy_from_slice(&init);
    let want = Machine::new(&mut want_env, &mut want_mem, &reg).run(f);

    let (tb, _) = compile(f, opts);
    if !NATIVE {
        return 0;
    }
    let ws = words(MEM_WORDS);
    for (k, w) in ws.iter().enumerate() {
        w.store(rd64(&init, 8 * k), Ordering::Relaxed);
    }
    let lock = Mutex::new(());
    let slow = AtomicUsize::new(0);
    let mut mem = WordMemory::new(&ws, BASE, &lock, &slow);
    let win = HostWindow::new(&ws[..WINDOW_WORDS], BASE);
    let mut env = vec![0u8; ENV_SIZE];
    let got = tb.run_with_window(&mut env, &win, &mut mem, &reg);
    assert_eq!(got, want, "{}", f.dump_ops(true));
    assert_eq!(env, want_env, "{}", f.dump_ops(true));
    let bytes: Vec<u8> = ws.iter().flat_map(|w| w.load(Ordering::Relaxed).to_le_bytes()).collect();
    assert_eq!(bytes, want_mem.bytes);
    slow.load(Ordering::Relaxed)
}

#[test]
fn window_accesses_match_the_interpreter() {
    let features =
        [HostFeatures::BASELINE, HostFeatures { lrcpc: true, lrcpc2: false }, HostFeatures::ALL];
    for m in FenceMapping::ALL {
        for feat in features {
            let opts = window(feat);
            // Aligned, inside the window: no service call at all.
            let f = accesses(m, &[0, 8, 48, 56], MemOp::UB);
            let slow = window_matches_interpreter(&optimized(&f), &opts);
            assert!(!NATIVE || slow == 0, "{m} {feat:?}: {slow} slow accesses");
            // Misaligned, across the end of the window and past it: some go to the service
            // routine, and the results must not change.
            let f = accesses(m, &[1, 3, 6, 58, 60, 64, 120], MemOp::UB);
            let slow = window_matches_interpreter(&f, &opts);
            assert!(!NATIVE || slow > 0);
            window_matches_interpreter(&optimized(&f), &opts);
        }
        // Without the window everything goes to the service routine.
        let f = accesses(m, &[0, 8], MemOp::UB);
        let slow = window_matches_interpreter(&f, &CodegenOptions::default());
        assert!(!NATIVE || slow > 0);
    }
}

#[test]
fn alignment_faults_still_come_from_the_service_routine() {
    for m in FenceMapping::ALL {
        let f = accesses(m, &[0, 2], MemOp::ALIGN);
        window_matches_interpreter(&f, &window(HostFeatures::detect()));
    }
}

#[test]
fn host_options() {
    let o = CodegenOptions::host();
    assert!(o.guest_window);
    assert_eq!(o.features, HostFeatures::detect());
    assert_eq!(CodegenOptions::default().features, HostFeatures::BASELINE);
}
