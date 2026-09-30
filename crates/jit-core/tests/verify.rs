// SPDX-License-Identifier: MIT OR Apache-2.0

//! The verifier: well formed IR passes, and each kind of mistake is reported.

use ruvm_jit_core::ir::{FuncConfig, HelperType};
use ruvm_jit_core::{Cond, Func, HelperInfo, MemOp, Opcode, Type};

fn func() -> Func {
    Func::new(FuncConfig::default())
}

fn errors(f: &Func) -> Vec<String> {
    match f.verify() {
        Ok(()) => Vec::new(),
        Err(e) => e.iter().map(|e| e.to_string()).collect(),
    }
}

fn expect_error(f: &Func, needle: &str) {
    let e = errors(f);
    assert!(e.iter().any(|m| m.contains(needle)), "no error containing {needle:?} in {e:?}");
}

#[test]
fn well_formed_block_passes() {
    let mut f = func();
    let env = f.env();
    let g = f.global_mem_new_i64(env, 0, "g");
    let t = f.temp_new_i32();
    f.gen_insn_start(&[0x1000, 0, 0]);
    f.gen_extrl_i64_i32(t, g);
    let l = f.new_label();
    f.gen_brcondi_i32(Cond::Ne, t, 0, l);
    f.gen_qemu_ld_i64(g, g, 0, MemOp::UQ);
    f.gen_set_label(l);
    // A TB temp defined before the label may be read after it.
    f.gen_ext_i32_i64(g, t);
    f.gen_goto_tb(0);
    f.gen_exit_tb(0x4000, 0);
    assert_eq!(errors(&f), Vec::<String>::new());
}

#[test]
fn type_mismatch() {
    let mut f = func();
    let a = f.temp_new_i32();
    let b = f.temp_new_i64();
    f.gen_movi_i32(a, 1);
    f.gen_movi_i64(b, 1);
    // Emit add_i32 with an i64 input by hand.
    f.emit_op(Opcode::Add, Type::I32, &[a.arg(), a.arg(), b.arg()]);
    f.gen_exit_tb(0, 0);
    expect_error(&f, "add: argument 2 (loc1) has type I64, expected Exact(I32)");
}

#[test]
fn helper_signature_mismatch() {
    let mut f = func();
    let h = f.helper(HelperInfo::new("h", 0, HelperType::I64, &[HelperType::I32]));
    let r = f.temp_new_i64();
    let a = f.temp_new_i32();
    f.gen_movi_i32(a, 0);
    let op = f.gen_call(h, Some(r.temp()), &[a.temp()]);
    // Point the argument at the i64 temp.
    f.op_mut(op).args[1] = r.arg();
    f.gen_exit_tb(0, 0);
    expect_error(&f, "call: argument 1");
}

#[test]
fn use_before_def() {
    let mut f = func();
    let env = f.env();
    let g = f.global_mem_new_i64(env, 0, "g");
    let t = f.temp_new_i64();
    f.gen_add_i64(g, g, t);
    f.gen_exit_tb(0, 0);
    expect_error(&f, "add: loc0 is read before it is written");

    // An EBB temp does not survive a label.
    let mut f = func();
    let env = f.env();
    let g = f.global_mem_new_i64(env, 0, "g");
    let t = f.temp_ebb_new_i64();
    f.gen_movi_i64(t, 1);
    let l = f.new_label();
    f.gen_brcondi_i64(Cond::Eq, g, 0, l);
    f.gen_set_label(l);
    f.gen_add_i64(g, g, t);
    f.gen_exit_tb(0, 0);
    expect_error(&f, "is read before it is written in this EBB");

    // Globals and constants are always defined; discard undefines.
    let mut f = func();
    let t = f.temp_new_i64();
    f.gen_movi_i64(t, 1);
    f.gen_discard_i64(t);
    f.gen_addi_i64(t, t, 1);
    f.gen_exit_tb(0, 0);
    expect_error(&f, "read before it is written");
}

#[test]
fn bad_labels() {
    let mut f = func();
    let l = f.new_label();
    f.gen_br(l);
    f.gen_exit_tb(0, 0);
    expect_error(&f, "branch to label $L0 which is never set");

    let mut f = func();
    let l = f.new_label();
    f.gen_set_label(l);
    f.emit_op(Opcode::SetLabel, Type::I32, &[l.arg()]);
    f.gen_exit_tb(0, 0);
    expect_error(&f, "label $L0 is set twice");

    let mut f = func();
    f.emit_op(Opcode::Br, Type::I32, &[7]);
    f.gen_exit_tb(0, 0);
    expect_error(&f, "br: label 7 does not exist");
}

#[test]
fn malformed_ops() {
    let mut f = func();
    let t = f.temp_new_i64();
    f.emit_op(Opcode::Mov, Type::I64, &[t.arg(), 999]);
    f.gen_exit_tb(0, 0);
    expect_error(&f, "mov: argument 1 is not a temp");

    let mut f = func();
    let (a, b) = (f.temp_new_i64(), f.temp_new_i64());
    f.gen_movi_i64(a, 0);
    f.gen_movi_i64(b, 0);
    f.emit_op(Opcode::Addci, Type::I64, &[a.arg(), a.arg(), b.arg()]);
    f.gen_exit_tb(0, 0);
    expect_error(&f, "carry-in op does not follow a carry-out op");

    let mut f = func();
    let t = f.temp_new_i64();
    f.gen_movi_i64(t, 0);
    f.emit_op(Opcode::Neg, Type::V128, &[t.arg(), t.arg()]);
    f.gen_exit_tb(0, 0);
    expect_error(&f, "neg: op type V128 is not I32 or I64");
}
