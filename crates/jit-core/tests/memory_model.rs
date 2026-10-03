// SPDX-License-Identifier: MIT OR Apache-2.0

//! The barriers each fence mapping emits around guest loads, stores, atomic read-modify-write
//! and compare and swap, and how the optimizer folds them. Each sequence is shown as a short
//! list: `ld` and `st` for guest accesses, with `.acq` and `.rel` for the ordering flags, `mb`
//! with its order bits and the Arm barrier QEMU's `tcg_out_mb` table picks for it, and `call`.

use ruvm_jit_core::ir::{FuncConfig, HelperInfo, HelperType, TempI64};
use ruvm_jit_core::memory_model::{FenceKind, X86_TSO, ldst_flags, needed_mo};
use ruvm_jit_core::types::mo;
use ruvm_jit_core::{FenceMapping, Func, MemOp, Opcode};

fn config(m: FenceMapping) -> FuncConfig {
    FuncConfig { parallel: true, guest_mo: X86_TSO, fence_mapping: m, ..FuncConfig::default() }
}

fn bar_name(bar: u64) -> String {
    let mut parts = Vec::new();
    for (bit, name) in [(mo::LD_LD, "rr"), (mo::ST_LD, "wr"), (mo::LD_ST, "rw"), (mo::ST_ST, "ww")]
    {
        if bar as u32 & bit != 0 {
            parts.push(name);
        }
    }
    let dmb = match FenceKind::for_bar(bar as u32) {
        FenceKind::Load => "ishld",
        FenceKind::Store => "ishst",
        FenceKind::Full => "ish",
    };
    format!("mb {} ({dmb})", if parts.len() == 4 { "all".to_string() } else { parts.join("+") })
}

/// The memory relevant ops of `f`, in order.
fn shape(f: &Func) -> Vec<String> {
    let mut out = Vec::new();
    for (_, op) in f.ops() {
        let ord = |s: &str, flag: u8, mark: &str| {
            if op.flags & flag != 0 { format!("{s}{mark}") } else { s.to_string() }
        };
        match op.opc {
            Opcode::QemuLd | Opcode::QemuLd2 => out.push(ord("ld", ldst_flags::ACQUIRE_PC, ".acq")),
            Opcode::QemuSt | Opcode::QemuSt2 => out.push(ord("st", ldst_flags::RELEASE, ".rel")),
            Opcode::Mb => out.push(bar_name(op.args[0])),
            Opcode::Call => out.push("call".to_string()),
            _ => {}
        }
    }
    out
}

fn check(f: &Func, want: &[&str]) {
    assert_eq!(shape(f), want);
}

struct Regs {
    a: TempI64,
    b: TempI64,
    v: TempI64,
}

fn setup(c: FuncConfig) -> (Func, Regs) {
    let mut f = Func::new(c);
    let env = f.env();
    let a = f.global_mem_new_i64(env, 0, "a");
    let b = f.global_mem_new_i64(env, 8, "b");
    let v = f.global_mem_new_i64(env, 16, "v");
    (f, Regs { a, b, v })
}

#[test]
fn plain_load_and_store() {
    for (m, ld, st) in [
        (FenceMapping::Qemu, &["mb rr (ishld)", "ld"][..], &["mb rw+ww (ish)", "st"][..]),
        (FenceMapping::Risotto, &["ld", "mb rr+rw (ishld)"][..], &["mb ww (ishst)", "st"][..]),
        (FenceMapping::AranciniRcpc, &["ld.acq"][..], &["st.rel"][..]),
    ] {
        let (mut f, r) = setup(config(m));
        f.gen_qemu_ld_i64(r.v, r.a, 0, MemOp::UQ);
        check(&f, ld);
        let (mut f, r) = setup(config(m));
        f.gen_qemu_st_i64(r.v, r.a, 0, MemOp::UQ);
        check(&f, st);
        // Every width and the 32-bit forms behave the same.
        let (mut f, r) = setup(config(m));
        let w = f.temp_new_i32();
        f.gen_qemu_ld_i32(w, r.a, 0, MemOp::SW);
        check(&f, ld);
        let (mut f, r) = setup(config(m));
        let w = f.temp_new_i32();
        f.gen_qemu_st_i32(w, r.a, 0, MemOp::UB);
        check(&f, st);
    }
}

#[test]
fn flags_keep_the_size() {
    let (mut f, r) = setup(config(FenceMapping::AranciniRcpc));
    f.gen_qemu_ld_i64(r.v, r.a, 0, MemOp::SW);
    f.gen_qemu_st_i64(r.v, r.a, 0, MemOp::UL);
    let flags: Vec<u8> = f.ops().map(|(_, op)| op.flags).collect();
    assert_eq!(flags, [ldst_flags::ACQUIRE_PC | 1, ldst_flags::RELEASE | 2]);
}

#[test]
fn i128_accesses() {
    for (m, want) in [
        (FenceMapping::Qemu, &["mb rr (ishld)", "ld", "mb ww (ishst)", "st"][..]),
        (FenceMapping::Risotto, &["ld", "mb rr+rw (ishld)", "mb ww (ishst)", "st"][..]),
        (FenceMapping::AranciniRcpc, &["ld.acq", "st.rel"][..]),
    ] {
        let (mut f, r) = setup(config(m));
        let t = f.temp_new_i128();
        f.gen_qemu_ld_i128(t, r.a, 0, MemOp::UO);
        f.gen_qemu_st_i128(t, r.b, 0, MemOp::UO);
        check(&f, want);
    }
}

#[test]
fn load_then_store_folds_to_one_barrier() {
    // QEMU: ldr; dmb ishld before the load, dmb ish before the store; nothing to merge.
    let (mut f, r) = setup(config(FenceMapping::Qemu));
    f.gen_qemu_ld_i64(r.v, r.a, 0, MemOp::UQ);
    f.gen_addi_i64(r.v, r.v, 1);
    f.gen_qemu_st_i64(r.v, r.b, 0, MemOp::UQ);
    check(&f, &["mb rr (ishld)", "ld", "mb rw+ww (ish)", "st"]);
    f.optimize();
    check(&f, &["mb rr (ishld)", "ld", "mb rw+ww (ish)", "st"]);

    // Risotto: ld; F(rm); F(ww); st becomes ld; F(rm+ww); st, one dmb ish, even with
    // arithmetic between the two barriers.
    let (mut f, r) = setup(config(FenceMapping::Risotto));
    f.gen_qemu_ld_i64(r.v, r.a, 0, MemOp::UQ);
    f.gen_addi_i64(r.v, r.v, 1);
    f.gen_qemu_st_i64(r.v, r.b, 0, MemOp::UQ);
    check(&f, &["ld", "mb rr+rw (ishld)", "mb ww (ishst)", "st"]);
    f.optimize();
    check(&f, &["ld", "mb rr+rw+ww (ish)", "st"]);
    f.verify().unwrap();

    // RCpc: no barriers at all.
    let (mut f, r) = setup(config(FenceMapping::AranciniRcpc));
    f.gen_qemu_ld_i64(r.v, r.a, 0, MemOp::UQ);
    f.gen_qemu_st_i64(r.v, r.b, 0, MemOp::UQ);
    f.optimize();
    check(&f, &["ld.acq", "st.rel"]);
}

#[test]
fn barriers_never_cross_accesses_or_calls() {
    // A store then a load: the store barrier and the load barrier are on opposite sides of
    // the accesses and stay apart. So do the barriers of two loads.
    let (mut f, r) = setup(config(FenceMapping::Risotto));
    f.gen_qemu_st_i64(r.v, r.a, 0, MemOp::UQ);
    f.gen_qemu_ld_i64(r.v, r.b, 0, MemOp::UQ);
    f.gen_qemu_ld_i64(r.v, r.b, 0, MemOp::UQ);
    f.optimize();
    check(&f, &["mb ww (ishst)", "st", "ld", "mb rr+rw (ishld)", "ld", "mb rr+rw (ishld)"]);

    // A helper call between a load and a store keeps both barriers.
    let (mut f, r) = setup(config(FenceMapping::Risotto));
    let h = f.helper(HelperInfo::new("h", 0, HelperType::Void, &[]));
    f.gen_qemu_ld_i64(r.v, r.a, 0, MemOp::UQ);
    f.gen_call(h, None, &[]);
    f.gen_qemu_st_i64(r.v, r.b, 0, MemOp::UQ);
    f.optimize();
    check(&f, &["ld", "mb rr+rw (ishld)", "call", "mb ww (ishst)", "st"]);

    // So does a label.
    let (mut f, r) = setup(config(FenceMapping::Risotto));
    let l = f.new_label();
    f.gen_qemu_ld_i64(r.v, r.a, 0, MemOp::UQ);
    f.gen_set_label(l);
    f.gen_qemu_st_i64(r.v, r.b, 0, MemOp::UQ);
    f.optimize();
    check(&f, &["ld", "mb rr+rw (ishld)", "mb ww (ishst)", "st"]);
}

#[test]
fn guest_fence_merges_with_load_barrier() {
    // mfence right after a load: the load barrier grows into the full one.
    let (mut f, r) = setup(config(FenceMapping::Risotto));
    f.gen_qemu_ld_i64(r.v, r.a, 0, MemOp::UQ);
    f.gen_mb(mo::ALL | mo::BAR_SC);
    f.gen_qemu_ld_i64(r.v, r.b, 0, MemOp::UQ);
    f.optimize();
    check(&f, &["ld", "mb all (ish)", "ld", "mb rr+rw (ishld)"]);

    // Store buffering with mfence under RCpc: the full barrier sits between the release store
    // and the acquire load, which RCpc alone would let pass each other.
    let (mut f, r) = setup(config(FenceMapping::AranciniRcpc));
    f.gen_qemu_st_i64(r.v, r.a, 0, MemOp::UQ);
    f.gen_mb(mo::ALL | mo::BAR_SC);
    f.gen_qemu_ld_i64(r.v, r.b, 0, MemOp::UQ);
    f.optimize();
    check(&f, &["st.rel", "mb all (ish)", "ld.acq"]);
}

#[test]
fn parallel_rmw_and_cmpxchg() {
    for m in FenceMapping::ALL {
        let around: &[&str] = if m == FenceMapping::Qemu {
            &["call"]
        } else {
            &["mb all (ish)", "call", "mb all (ish)"]
        };
        let (mut f, r) = setup(config(m));
        f.gen_atomic_fetch_add_i64(r.v, r.a, r.b, 0, MemOp::UQ);
        check(&f, around);
        let (mut f, r) = setup(config(m));
        f.gen_atomic_xchg_i64(r.v, r.a, r.b, 0, MemOp::UL);
        check(&f, around);
        let (mut f, r) = setup(config(m));
        let (x, y) = (f.temp_new_i32(), f.temp_new_i32());
        f.gen_atomic_fetch_or_i32(x, r.a, y, 0, MemOp::UB);
        check(&f, around);
        let (mut f, r) = setup(config(m));
        f.gen_atomic_cmpxchg_i64(r.v, r.a, r.b, r.v, 0, MemOp::UQ);
        check(&f, around);
        let (mut f, r) = setup(config(m));
        f.gen_atomic_cmpxchg_i64(r.v, r.a, r.b, r.v, 0, MemOp::SL);
        check(&f, around);
        let (mut f, r) = setup(config(m));
        let (x, y, z) = (f.temp_new_i32(), f.temp_new_i32(), f.temp_new_i32());
        f.gen_atomic_cmpxchg_i32(x, r.a, y, z, 0, MemOp::UL);
        check(&f, around);
        let (mut f, r) = setup(config(m));
        let (x, y, z) = (f.temp_new_i128(), f.temp_new_i128(), f.temp_new_i128());
        f.gen_atomic_cmpxchg_i128(x, r.a, y, z, 0, MemOp::UO);
        check(&f, around);
        let (mut f, r) = setup(config(m));
        let (x, y) = (f.temp_new_i128(), f.temp_new_i128());
        f.gen_atomic_op_i128(
            ruvm_jit_core::tcg_op_ldst::AtomicOp::FetchOr,
            x,
            r.a,
            y,
            0,
            MemOp::UO,
        );
        check(&f, around);
        // The barriers stay through the optimizer: a call ends a run of barriers.
        f.optimize();
        check(&f, around);
    }
}

#[test]
fn serial_rmw_and_cmpxchg() {
    let serial = |m| FuncConfig { parallel: false, ..config(m) };

    let (mut f, r) = setup(serial(FenceMapping::Qemu));
    f.gen_atomic_fetch_add_i64(r.v, r.a, r.b, 0, MemOp::UQ);
    check(&f, &["mb rr (ishld)", "ld", "mb rw+ww (ish)", "st"]);

    let (mut f, r) = setup(serial(FenceMapping::Risotto));
    f.gen_atomic_fetch_add_i64(r.v, r.a, r.b, 0, MemOp::UQ);
    check(&f, &["mb all (ish)", "ld", "mb rr+rw (ishld)", "mb ww (ishst)", "st", "mb all (ish)"]);
    f.optimize();
    check(&f, &["mb all (ish)", "ld", "mb rr+rw+ww (ish)", "st", "mb all (ish)"]);

    let (mut f, r) = setup(serial(FenceMapping::AranciniRcpc));
    f.gen_atomic_cmpxchg_i64(r.v, r.a, r.b, r.v, 0, MemOp::UQ);
    check(&f, &["mb all (ish)", "ld.acq", "st.rel", "mb all (ish)"]);

    // The non-atomic compare and swap is a plain load and store.
    let (mut f, r) = setup(config(FenceMapping::Risotto));
    f.gen_nonatomic_cmpxchg_i64(r.v, r.a, r.b, r.v, 0, MemOp::UQ);
    check(&f, &["ld", "mb rr+rw (ishld)", "mb ww (ishst)", "st"]);

    // A leading full barrier swallows the barrier of the load before it.
    let (mut f, r) = setup(config(FenceMapping::Risotto));
    f.gen_qemu_ld_i64(r.v, r.a, 0, MemOp::UQ);
    f.gen_atomic_fetch_add_i64(r.v, r.a, r.b, 0, MemOp::UQ);
    f.optimize();
    check(&f, &["ld", "mb all (ish)", "call", "mb all (ish)"]);
}

#[test]
fn mappings_that_do_not_fit_fall_back_to_qemu() {
    // A sequentially consistent guest needs ST_LD, which neither alternative enforces.
    for m in FenceMapping::ALL {
        let (mut f, r) = setup(FuncConfig { guest_mo: mo::ALL, fence_mapping: m, ..config(m) });
        f.gen_qemu_ld_i64(r.v, r.a, 0, MemOp::UQ);
        f.gen_qemu_st_i64(r.v, r.b, 0, MemOp::UQ);
        f.gen_atomic_fetch_add_i64(r.v, r.a, r.b, 0, MemOp::UQ);
        check(&f, &["mb rr+wr (ish)", "ld", "mb rw+ww (ish)", "st", "call"]);
    }
    // A host as strong as the guest (x86 on x86) and a weak guest (Arm on Arm) need nothing.
    for (g, h) in [(X86_TSO, X86_TSO), (0, 0)] {
        for m in FenceMapping::ALL {
            let (mut f, r) = setup(FuncConfig {
                guest_mo: g,
                target_default_mo: h,
                fence_mapping: m,
                ..FuncConfig::default()
            });
            f.gen_qemu_ld_i64(r.v, r.a, 0, MemOp::UQ);
            f.gen_qemu_st_i64(r.v, r.b, 0, MemOp::UQ);
            f.gen_atomic_fetch_add_i64(r.v, r.a, r.b, 0, MemOp::UQ);
            check(&f, &["ld", "st", "ld", "st"]);
        }
    }
}

#[test]
fn serial_user_mode_needs_no_order() {
    // tcg_gen_mb drops barriers in a serial user mode block; the flags follow it.
    for m in FenceMapping::ALL {
        let (mut f, r) = setup(FuncConfig { parallel: false, user_only: true, ..config(m) });
        f.gen_qemu_ld_i64(r.v, r.a, 0, MemOp::UQ);
        f.gen_qemu_st_i64(r.v, r.b, 0, MemOp::UQ);
        check(&f, &["ld", "st"]);
    }
}

#[test]
fn selection() {
    let arm = 0;
    let x86 = X86_TSO;
    assert_eq!(needed_mo(X86_TSO, arm), mo::LD_LD | mo::LD_ST | mo::ST_ST);
    assert_eq!(FenceMapping::default(), FenceMapping::Qemu);
    assert_eq!(FuncConfig::default().fence_mapping, FenceMapping::Qemu);

    // x86 on Arm.
    assert_eq!(FenceMapping::preferred(X86_TSO, arm), FenceMapping::Risotto);
    assert_eq!(FenceMapping::Qemu.select(X86_TSO, arm, true), FenceMapping::Qemu);
    assert_eq!(FenceMapping::Risotto.select(X86_TSO, arm, true), FenceMapping::Risotto);
    assert_eq!(FenceMapping::AranciniRcpc.select(X86_TSO, arm, true), FenceMapping::AranciniRcpc);
    assert_eq!(FenceMapping::AranciniRcpc.select(X86_TSO, arm, false), FenceMapping::Risotto);

    // Nothing to enforce, or ST_LD to enforce: QEMU's mapping.
    for m in FenceMapping::ALL {
        assert_eq!(m.select(X86_TSO, x86, true), FenceMapping::Qemu);
        assert_eq!(m.select(0, arm, true), FenceMapping::Qemu);
        assert_eq!(m.select(mo::ALL, arm, true), FenceMapping::Qemu);
        assert_eq!(FenceMapping::from_name(m.name()), Some(m));
        assert_eq!(m.to_string(), m.name());
    }
    assert_eq!(FenceMapping::preferred(mo::ALL, arm), FenceMapping::Qemu);
    assert_eq!(FenceMapping::from_name("arancini"), Some(FenceMapping::AranciniRcpc));
    assert_eq!(FenceMapping::from_name("tso"), None);
}

#[test]
fn host_barrier_table() {
    // QEMU's aarch64 tcg_out_mb, for every combination of order bits.
    for bar in 0..=mo::ALL {
        let want = match bar {
            8 => FenceKind::Store,
            1 | 4 | 5 => FenceKind::Load,
            _ => FenceKind::Full,
        };
        assert_eq!(FenceKind::for_bar(bar), want, "bar {bar:#x}");
        assert_eq!(FenceKind::for_bar(bar | mo::BAR_SC), want);
    }
}
