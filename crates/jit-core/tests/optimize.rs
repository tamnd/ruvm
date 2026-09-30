// SPDX-License-Identifier: MIT OR Apache-2.0

//! The optimizer and liveness passes, shown as IR dumps before and after. The expected text is
//! what QEMU's `-d op,op_opt` prints for the same ops on a 64-bit host, less the `pref=` sets.

use ruvm_jit_core::ir::{FuncConfig, TempI32, TempI64};
use ruvm_jit_core::liveness::LogMask;
use ruvm_jit_core::types::{bswap, mo};
use ruvm_jit_core::{Cond, Func, MemOp, Type};

fn setup() -> (Func, [TempI64; 3], [TempI32; 2]) {
    let mut f = Func::new(FuncConfig::default());
    let env = f.env();
    let r = [
        f.global_mem_new_i64(env, 0, "r0"),
        f.global_mem_new_i64(env, 8, "r1"),
        f.global_mem_new_i64(env, 16, "r2"),
    ];
    let w = [f.global_mem_new_i32(env, 24, "w0"), f.global_mem_new_i32(env, 28, "w1")];
    (f, r, w)
}

/// Check the dump before and after the optimizer alone.
fn check_opt(mut f: Func, before: &str, after: &str) {
    assert_eq!(f.dump_ops(false), before, "before");
    f.optimize();
    assert_eq!(f.dump_ops(false), after, "after");
    f.verify().expect("the optimized IR verifies");
}

/// Check the `op_opt` dump of the whole middle end.
fn check_opt_live(mut f: Func, want: &str) {
    let log = f.gen_code(true, LogMask { op: false, op_ind: false, op_opt: true });
    let got = log.strip_prefix("OP after optimization and liveness analysis:\n").unwrap();
    assert_eq!(got.trim_end_matches('\n'), want.trim_end_matches('\n'));
}

#[test]
fn constant_folding() {
    let (mut f, r, w) = setup();
    let t = f.temp_new_i64();
    f.gen_movi_i64(t, 2);
    f.gen_addi_i64(t, t, 3);
    f.gen_muli_i64(r[0], t, 4);
    let c = f.constant_i32(0x7fff_ffff);
    f.gen_addi_i32(w[1], c, 1);
    f.gen_exit_tb(0, 0);
    check_opt(
        f,
        " mov_i64 loc0,$0x2
 add_i64 loc0,loc0,$0x3
 shl_i64 r0,loc0,$0x2
 add_i32 w1,$0x7fffffff,$0x1
 exit_tb $0x0
",
        " mov_i64 loc0,$0x2
 mov_i64 loc0,$0x5
 mov_i64 r0,$0x14
 mov_i32 w1,$0x80000000
 exit_tb $0x0
",
    );
}

#[test]
fn copy_propagation_and_commutative_swap() {
    let (mut f, r, _) = setup();
    let u = f.temp_new_i64();
    f.gen_mov_i64(u, r[1]);
    f.gen_add_i64(r[2], u, r[2]);
    f.gen_sub_i64(r[0], u, r[1]);
    f.gen_exit_tb(0, 0);
    check_opt(
        f,
        " mov_i64 loc0,r1
 add_i64 r2,loc0,r2
 sub_i64 r0,loc0,r1
 exit_tb $0x0
",
        " mov_i64 loc0,r1
 add_i64 r2,r2,r1
 mov_i64 r0,$0x0
 exit_tb $0x0
",
    );
}

#[test]
fn known_bits() {
    let (mut f, r, w) = setup();
    let t = f.temp_new_i64();
    f.gen_ext8u_i64(t, r[1]);
    f.gen_andi_i64(r[1], t, 0xff);
    f.gen_shri_i64(r[0], t, 8);
    f.gen_xor_i32(w[0], w[1], w[1]);
    f.gen_setcond_i64(Cond::Eq, r[2], t, t);
    f.gen_exit_tb(0, 0);
    check_opt(
        f,
        " extract_i64 loc0,r1,$0x0,$0x8
 extract_i64 r1,loc0,$0x0,$0x8
 shr_i64 r0,loc0,$0x8
 xor_i32 w0,w1,w1
 setcond_i64 r2,loc0,loc0,eq
 exit_tb $0x0
",
        " extract_i64 loc0,r1,$0x0,$0x8
 mov_i64 r1,loc0
 mov_i64 r0,$0x0
 mov_i32 w0,$0x0
 mov_i64 r2,$0x1
 exit_tb $0x0
",
    );
}

#[test]
fn brcond_folding() {
    let (mut f, r, _) = setup();
    let t = f.temp_new_i64();
    f.gen_movi_i64(t, 5);
    let l0 = f.new_label();
    let l1 = f.new_label();
    f.gen_brcondi_i64(Cond::Gtu, t, 9, l0);
    f.gen_movi_i64(r[0], 1);
    f.gen_brcondi_i64(Cond::Lt, t, 9, l1);
    f.gen_movi_i64(r[0], 2);
    f.gen_set_label(l0);
    f.gen_movi_i64(r[1], 3);
    f.gen_set_label(l1);
    f.gen_exit_tb(0, 0);
    let g = f.clone();
    check_opt(
        f,
        " mov_i64 loc0,$0x5
 brcond_i64 loc0,$0x9,gtu,$L0
 mov_i64 r0,$0x1
 brcond_i64 loc0,$0x9,lt,$L1
 mov_i64 r0,$0x2
 set_label $L0
 mov_i64 r1,$0x3
 set_label $L1
 exit_tb $0x0
",
        " mov_i64 loc0,$0x5
 mov_i64 r0,$0x1
 br $L1
 mov_i64 r0,$0x2
 set_label $L0
 mov_i64 r1,$0x3
 set_label $L1
 exit_tb $0x0
",
    );
    // The reachability pass then drops the dead code after br, then the br to the very next
    // label, then both labels, which nothing uses any more.
    check_opt_live(
        g,
        " mov_i64 r0,$0x1                          sync: 0  dead: 0 1
 exit_tb $0x0                           
",
    );
}

#[test]
fn liveness_syncs_and_dead_temps() {
    let (mut f, r, _) = setup();
    let t = f.temp_new_i64();
    f.gen_add_i64(t, r[0], r[1]);
    f.gen_mul_i64(r[2], t, r[0]);
    f.gen_exit_tb(0, 0);
    check_opt_live(
        f,
        " add_i64 tmp0,r0,r1                       dead: 2
 mul_i64 r2,tmp0,r0                       sync: 0  dead: 0 1 2
 exit_tb $0x0                           
",
    );
}

#[test]
fn test_conditions() {
    let (mut f, r, _) = setup();
    let t = f.temp_new_i64();
    f.gen_ext8u_i64(t, r[1]);
    // TSTNE x,i is NE x,0 when i covers every bit x can have.
    f.gen_setcondi_i64(Cond::TstNe, r[0], t, 0xff);
    // TSTNE x,pow2 is an extract of that bit.
    f.gen_setcondi_i64(Cond::TstNe, r[2], r[1], i64::MIN);
    // A condition on two copies of a value is known.
    f.gen_movcond_i64(Cond::Eq, r[1], t, t, r[0], r[2]);
    f.gen_exit_tb(0, 0);
    check_opt(
        f,
        " extract_i64 loc0,r1,$0x0,$0x8
 setcond_i64 r0,loc0,$0xff,tstne
 setcond_i64 r2,r1,$0x8000000000000000,tstne
 movcond_i64 r1,loc0,loc0,r0,r2,eq
 exit_tb $0x0
",
        " extract_i64 loc0,r1,$0x0,$0x8
 setcond_i64 r0,loc0,$0x0,ne
 extract_i64 r2,r1,$0x3f,$0x1
 mov_i64 r1,r0
 exit_tb $0x0
",
    );
}

#[test]
fn indirect_globals_lowering() {
    let mut f = Func::new(FuncConfig::default());
    let env = f.env();
    let base = f.global_mem_new_ptr(env, 0x100, "base");
    let x = f.global_mem_new_i64(base, 0x10, "x");
    let y = f.global_mem_new_i64(env, 0x108, "y");
    f.gen_addi_i64(x, x, 5);
    f.gen_mov_i64(y, x);
    let l = f.new_label();
    f.gen_brcondi_i64(Cond::Eq, y, 0, l);
    f.gen_addi_i64(x, x, 1);
    f.gen_set_label(l);
    f.gen_exit_tb(0, 0);
    let log = f.gen_code(true, LogMask::ALL);
    assert_eq!(
        log,
        "OP:
 add_i64 x,x,$0x5
 mov_i64 y,x
 brcond_i64 y,$0x0,eq,$L0
 add_i64 x,x,$0x1
 set_label $L0
 exit_tb $0x0

OP before indirect lowering:
 add_i64 x,x,$0x5                         sync: 0  dead: 1 2
 mov_i64 y,x                              sync: 0
 brcond_i64 y,$0x0,eq,$L0                 dead: 0 1
 add_i64 x,x,$0x1                         sync: 0  dead: 0 1 2
 set_label $L0
 exit_tb $0x0

OP after optimization and liveness analysis:
 ld_i64 tmp3,base,$0x10                 
 add_i64 tmp3,tmp3,$0x5                   dead: 1 2
 st_i64 tmp3,base,$0x10                 
 mov_i64 y,tmp3                           sync: 0
 brcond_i64 y,$0x0,eq,$L0                 dead: 0 1
 add_i64 tmp3,tmp3,$0x1                   dead: 1 2
 st_i64 tmp3,base,$0x10                   dead: 0 1
 set_label $L0                          
 exit_tb $0x0                           

"
    );
}

#[test]
fn dump_format() {
    // Insn markers, memops, barriers, vector ops, flags and calls print as QEMU prints them.
    let mut f = Func::new(FuncConfig::default());
    let env = f.env();
    let pc = f.global_mem_new_i64(env, 0, "pc");
    let w = f.global_mem_new_i32(env, 8, "w");
    f.gen_insn_start(&[0x40_1000, 0, 0]);
    f.gen_qemu_ld_i64(pc, pc, 1, MemOp::UL.or(MemOp::BE).or(MemOp::ALIGN));
    f.gen_qemu_st_i32(w, pc, 0, MemOp::UW);
    f.gen_mb(mo::ALL | mo::BAR_SC);
    f.gen_atomic_fetch_add_i32(w, pc, w, 2, MemOp::UL);
    let v = f.temp_new_vec(Type::V128);
    f.gen_dup_i32_vec(2, v, w);
    f.gen_add_vec(1, v, v, v);
    f.gen_st_vec(v, env, 0x40);
    f.gen_bswap16_i32(w, w, bswap::OS);
    f.gen_goto_tb(0);
    f.gen_exit_tb(0x7000, 0);
    f.gen_lookup_and_goto_ptr();
    assert_eq!(
        f.dump_ops(false),
        "
 ---- 0000000000401000 0000000000000000 0000000000000000
 qemu_ld_i64 pc,pc,noat+al+beul,1
 qemu_st_i32 w,pc,noat+un+leuw,0
 mb seq:all
 qemu_ld_i32 tmp0,pc,noat+un+leul,2
 mov_i32 tmp1,w
 add_i32 tmp1,tmp0,tmp1
 qemu_st_i32 tmp1,pc,noat+un+leul,2
 mov_i32 w,tmp0
dup_vec v128,e32,tmp2,w
add_vec v128,e16,tmp2,tmp2,tmp2
st_vec v128,e8,tmp2,env,$0x40
 bswap16_i32 w,w,os
 goto_tb $0x0
 exit_tb $0x7000
 call lookup_tb_ptr,$0x6,$1,tmp3,env
 goto_ptr tmp3
"
    );
}
