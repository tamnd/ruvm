// SPDX-License-Identifier: GPL-2.0-or-later

//! The backend against the reference interpreter. Random scalar and vector blocks, blocks with
//! more live values than host registers, and hand written ones for helpers, guest memory, exits
//! and errors, are run both ways, before and after the optimizer, and must leave the same CPU
//! state, guest memory and exit. A hot loop is timed both ways.
//!
//! Generated code only runs on an AArch64 host; elsewhere these tests check that every block
//! compiles.

use std::sync::Arc;

use ruvm_jit_aarch64::{CodeRegion, CompiledTb};
use ruvm_jit_core::helpers::lookup_tb_ptr;
use ruvm_jit_core::ir::{FuncConfig, HelperType, TempI32, TempI64, TempVec};
use ruvm_jit_core::liveness::LogMask;
use ruvm_jit_core::types::{bswap, call_flags};
use ruvm_jit_core::{Cond, Func, HelperInfo, Label, MemOp, Opcode, Type};
use ruvm_jit_interp::{Exit, FlatMemory, HelperEnv, HelperRegistry, InterpError, Machine, Unwind};

const ENV_SIZE: usize = 0x400;
const MEM_BASE: u64 = 0x1_0000;
const N64: usize = 6;
const N32: usize = 4;
const G64: i64 = 0x100;
const G32: i64 = 0x180;
const SCRATCH: i64 = 0x300;

const NATIVE: bool = cfg!(all(unix, target_arch = "aarch64"));

fn region() -> Arc<CodeRegion> {
    CodeRegion::new(1 << 20).expect("code region")
}

fn compile(r: &Arc<CodeRegion>, f: &Func) -> CompiledTb {
    match r.compile(f) {
        Ok(tb) => tb,
        Err(e) => panic!("compile: {e}\n{}", f.dump_ops(false)),
    }
}

type Outcome = (Result<Exit, InterpError>, Vec<u8>, FlatMemory);

fn interp(
    f: &Func,
    env: &[u8],
    mem: &FlatMemory,
    reg: &HelperRegistry,
    linked: [bool; 2],
) -> Outcome {
    let (mut e, mut m) = (env.to_vec(), mem.clone());
    let mut mach = Machine::new(&mut e, &mut m, reg);
    mach.linked = linked;
    let x = mach.run(f);
    (x, e, m)
}

fn native(tb: &CompiledTb, env: &[u8], mem: &FlatMemory, reg: &HelperRegistry) -> Outcome {
    let (mut e, mut m) = (env.to_vec(), mem.clone());
    let x = tb.run(&mut e, &mut m, reg);
    (x, e, m)
}

/// Compile `f` as built and optimized, run both natively and in the interpreter, check that
/// all four agree, and return the result.
fn check(
    f: &Func,
    env: &[u8],
    mem: &FlatMemory,
    reg: &HelperRegistry,
    linked: [bool; 2],
) -> Outcome {
    f.verify().expect("verify");
    let r = region();
    let mut opt = f.clone();
    opt.gen_code(true, LogMask::default());
    let want = interp(f, env, mem, reg, linked);
    for g in [f, &opt] {
        let tb = compile(&r, g);
        for (slot, on) in linked.iter().enumerate() {
            tb.set_goto_tb_linked(slot as u32, *on);
        }
        if !NATIVE {
            continue;
        }
        let got = native(&tb, env, mem, reg);
        assert!(
            got.0 == want.0 && got.1 == want.1 && got.2 == want.2,
            "native run differs: {:?} vs interpreter {:?}\n{}",
            got.0,
            want.0,
            g.dump_ops(true)
        );
    }
    want
}

fn rd64(env: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(env[off..off + 8].try_into().unwrap())
}

/// A small xorshift generator, the same as the interpreter's tests use.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// A value biased toward edge cases.
    fn interesting(&mut self) -> u64 {
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

const CONDS: [Cond; 14] = [
    Cond::Never,
    Cond::Always,
    Cond::Lt,
    Cond::Ge,
    Cond::Gt,
    Cond::Le,
    Cond::Eq,
    Cond::Ne,
    Cond::Ltu,
    Cond::Geu,
    Cond::Gtu,
    Cond::Leu,
    Cond::TstEq,
    Cond::TstNe,
];

/// Random scalar blocks, after the interpreter's property tests.
struct Gen<'a> {
    f: &'a mut Func,
    rng: Rng,
    v64: Vec<TempI64>,
    v32: Vec<TempI32>,
    pending: Vec<(Label, u32)>,
}

impl Gen<'_> {
    fn a64(&mut self) -> TempI64 {
        let i = self.rng.below(self.v64.len() as u64) as usize;
        self.v64[i]
    }

    fn a32(&mut self) -> TempI32 {
        let i = self.rng.below(self.v32.len() as u64) as usize;
        self.v32[i]
    }

    fn cond(&mut self) -> Cond {
        CONDS[self.rng.below(CONDS.len() as u64) as usize]
    }

    fn imm(&mut self) -> i64 {
        self.rng.interesting() as i64
    }

    fn field(&mut self, width: u32) -> (u32, u32) {
        let ofs = self.rng.below(width as u64) as u32;
        let len = 1 + self.rng.below((width - ofs) as u64) as u32;
        (ofs, len)
    }

    fn distinct64(&mut self, r: TempI64) -> TempI64 {
        loop {
            let t = self.a64();
            if t != r {
                return t;
            }
        }
    }

    fn distinct32(&mut self, r: TempI32) -> TempI32 {
        loop {
            let t = self.a32();
            if t != r {
                return t;
            }
        }
    }

    fn step64(&mut self) {
        let (r, a, b) = (self.a64(), self.a64(), self.a64());
        let k = self.rng.below(44);
        let f = &mut *self.f;
        match k {
            0 => f.gen_add_i64(r, a, b),
            1 => f.gen_sub_i64(r, a, b),
            2 => f.gen_mul_i64(r, a, b),
            3 => f.gen_and_i64(r, a, b),
            4 => f.gen_or_i64(r, a, b),
            5 => f.gen_xor_i64(r, a, b),
            6 => f.gen_andc_i64(r, a, b),
            7 => f.gen_orc_i64(r, a, b),
            8 => f.gen_eqv_i64(r, a, b),
            9 => f.gen_nand_i64(r, a, b),
            10 => f.gen_nor_i64(r, a, b),
            11 => f.gen_shl_i64(r, a, b),
            12 => f.gen_shr_i64(r, a, b),
            13 => f.gen_sar_i64(r, a, b),
            14 => f.gen_rotl_i64(r, a, b),
            15 => f.gen_rotr_i64(r, a, b),
            16 => f.gen_divu_i64(r, a, b),
            17 => f.gen_rem_i64(r, a, b),
            18 => f.gen_muluh_i64(r, a, b),
            19 => f.gen_mulsh_i64(r, a, b),
            20 => f.gen_clz_i64(r, a, b),
            21 => f.gen_ctz_i64(r, a, b),
            22 => f.gen_ctpop_i64(r, a),
            23 => f.gen_neg_i64(r, a),
            24 => f.gen_not_i64(r, a),
            25 => f.gen_smin_i64(r, a, b),
            26 => f.gen_umax_i64(r, a, b),
            27 => {
                let c = self.cond();
                self.f.gen_setcond_i64(c, r, a, b)
            }
            28 => {
                let c = self.cond();
                let i = self.imm();
                self.f.gen_negsetcondi_i64(c, r, a, i)
            }
            29 => {
                let c = self.cond();
                let (x, y) = (self.a64(), self.a64());
                self.f.gen_movcond_i64(c, r, a, b, x, y)
            }
            30 => {
                let (ofs, len) = self.field(64);
                self.f.gen_deposit_i64(r, a, b, ofs, len)
            }
            31 => {
                let (ofs, len) = self.field(64);
                self.f.gen_extract_i64(r, a, ofs, len)
            }
            32 => {
                let (ofs, len) = self.field(64);
                self.f.gen_sextract_i64(r, a, ofs, len)
            }
            33 => {
                let ofs = self.rng.below(64) as u32;
                self.f.gen_extract2_i64(r, a, b, ofs)
            }
            34 => {
                let r2 = self.distinct64(r);
                if self.rng.below(2) == 0 {
                    self.f.gen_mulu2_i64(r, r2, a, b)
                } else {
                    self.f.gen_muls2_i64(r, r2, a, b)
                }
            }
            35 => {
                let r2 = self.distinct64(r);
                let (c, d) = (self.a64(), self.a64());
                if self.rng.below(2) == 0 {
                    self.f.gen_add2_i64(r, r2, a, b, c, d)
                } else {
                    self.f.gen_sub2_i64(r, r2, a, b, c, d)
                }
            }
            36 => {
                let flags =
                    [bswap::OZ, bswap::OS, bswap::IZ | bswap::OZ][self.rng.below(3) as usize];
                match self.rng.below(3) {
                    0 => self.f.gen_bswap16_i64(r, a, flags),
                    1 => self.f.gen_bswap32_i64(r, a, flags),
                    _ => self.f.gen_bswap64_i64(r, a),
                }
            }
            37 => {
                let i = self.imm();
                match self.rng.below(8) {
                    0 => self.f.gen_addi_i64(r, a, i),
                    1 => self.f.gen_andi_i64(r, a, i),
                    2 => self.f.gen_ori_i64(r, a, i),
                    3 => self.f.gen_xori_i64(r, a, i),
                    4 => self.f.gen_muli_i64(r, a, i),
                    5 => self.f.gen_shli_i64(r, a, i & 63),
                    6 => self.f.gen_sari_i64(r, a, i & 63),
                    _ => self.f.gen_movi_i64(r, i),
                }
            }
            38 => {
                let x = self.a32();
                match self.rng.below(4) {
                    0 => self.f.gen_ext_i32_i64(r, x),
                    1 => self.f.gen_extu_i32_i64(r, x),
                    2 => {
                        let y = self.a32();
                        self.f.gen_concat_i32_i64(r, x, y)
                    }
                    _ => self.f.gen_ext32s_i64(r, a),
                }
            }
            39 => f.gen_div_i64(r, a, b),
            40 => f.gen_remu_i64(r, a, b),
            41 => {
                // The 128 by 64 bit divisions go through the service routine.
                let r2 = self.distinct64(r);
                let (c, d) = (self.a64(), self.a64());
                self.f.gen_xor_i64(c, c, d);
                let opc = if self.rng.below(2) == 0 { Opcode::Divu2 } else { Opcode::Divs2 };
                let args = [r, r2, a, c, b].map(|t| t.temp().arg());
                self.f.emit_op(opc, Type::I64, &args);
            }
            42 => {
                let co = self.distinct64(r);
                let ci = self.a64();
                self.f.gen_addcio_i64(r, co, a, b, ci)
            }
            _ => {
                // A round trip through a host memory slot.
                let env = self.f.env();
                let o = SCRATCH + 8 * self.rng.below(4) as i64;
                self.f.gen_st_i64(a, env, o);
                match self.rng.below(4) {
                    0 => self.f.gen_ld_i64(r, env, o),
                    1 => self.f.gen_ld8s_i64(r, env, o + 1),
                    2 => self.f.gen_ld16s_i64(r, env, o + 2),
                    _ => self.f.gen_ld32u_i64(r, env, o + 4),
                }
            }
        }
    }

    fn step32(&mut self) {
        let (r, a, b) = (self.a32(), self.a32(), self.a32());
        let k = self.rng.below(32);
        let f = &mut *self.f;
        match k {
            0 => f.gen_add_i32(r, a, b),
            1 => f.gen_sub_i32(r, a, b),
            2 => f.gen_mul_i32(r, a, b),
            3 => f.gen_and_i32(r, a, b),
            4 => f.gen_or_i32(r, a, b),
            5 => f.gen_xor_i32(r, a, b),
            6 => f.gen_shl_i32(r, a, b),
            7 => f.gen_shr_i32(r, a, b),
            8 => f.gen_sar_i32(r, a, b),
            9 => f.gen_rotl_i32(r, a, b),
            10 => f.gen_rotr_i32(r, a, b),
            11 => f.gen_div_i32(r, a, b),
            12 => f.gen_remu_i32(r, a, b),
            13 => f.gen_clz_i32(r, a, b),
            14 => f.gen_ctz_i32(r, a, b),
            15 => f.gen_ctpop_i32(r, a),
            16 => f.gen_neg_i32(r, a),
            17 => f.gen_ext8s_i32(r, a),
            18 => f.gen_ext16u_i32(r, a),
            19 => {
                let c = self.cond();
                self.f.gen_setcond_i32(c, r, a, b)
            }
            20 => {
                let c = self.cond();
                let (x, y) = (self.a32(), self.a32());
                self.f.gen_movcond_i32(c, r, a, b, x, y)
            }
            21 => {
                let (ofs, len) = self.field(32);
                self.f.gen_deposit_i32(r, a, b, ofs, len)
            }
            22 => {
                let (ofs, len) = self.field(32);
                if self.rng.below(2) == 0 {
                    self.f.gen_extract_i32(r, a, ofs, len)
                } else {
                    self.f.gen_sextract_i32(r, a, ofs, len)
                }
            }
            23 => {
                let r2 = self.distinct32(r);
                match self.rng.below(4) {
                    0 => self.f.gen_mulu2_i32(r, r2, a, b),
                    1 => self.f.gen_muls2_i32(r, r2, a, b),
                    2 => {
                        let (c, d) = (self.a32(), self.a32());
                        self.f.gen_sub2_i32(r, r2, a, b, c, d)
                    }
                    _ => {
                        let (c, d) = (self.a32(), self.a32());
                        self.f.gen_add2_i32(r, r2, a, b, c, d)
                    }
                }
            }
            24 => {
                let flags = [bswap::OZ, bswap::OS][self.rng.below(2) as usize];
                if self.rng.below(2) == 0 {
                    self.f.gen_bswap16_i32(r, a, flags)
                } else {
                    self.f.gen_bswap32_i32(r, a)
                }
            }
            25 => {
                let x = self.a64();
                if self.rng.below(2) == 0 {
                    self.f.gen_extrl_i64_i32(r, x)
                } else {
                    self.f.gen_extrh_i64_i32(r, x)
                }
            }
            26 => {
                let i = self.imm() as i32;
                match self.rng.below(5) {
                    0 => self.f.gen_addi_i32(r, a, i),
                    1 => self.f.gen_andi_i32(r, a, i),
                    2 => self.f.gen_shri_i32(r, a, i & 31),
                    3 => self.f.gen_rotli_i32(r, a, i & 31),
                    _ => self.f.gen_movi_i32(r, i),
                }
            }
            27 => {
                let c = self.cond();
                let i = self.imm() as i32;
                self.f.gen_setcondi_i32(c, r, a, i)
            }
            28 => {
                let ofs = self.rng.below(32) as u32;
                self.f.gen_extract2_i32(r, a, b, ofs)
            }
            29 => f.gen_mulsh_i32(r, a, b),
            30 => {
                let env = self.f.env();
                let o = SCRATCH + 4 * self.rng.below(8) as i64;
                self.f.gen_st16_i32(a, env, o);
                if self.rng.below(2) == 0 {
                    self.f.gen_ld16s_i32(r, env, o)
                } else {
                    self.f.gen_ld8u_i32(r, env, o + 1)
                }
            }
            _ => f.gen_umin_i32(r, a, b),
        }
    }

    fn branch(&mut self) {
        let l = self.f.new_label();
        let skip = 1 + self.rng.below(4) as u32;
        if self.rng.below(2) == 0 {
            let (a, b, c) = (self.a64(), self.a64(), self.cond());
            self.f.gen_brcond_i64(c, a, b, l);
        } else {
            let (a, c, i) = (self.a32(), self.cond(), self.imm() as i32);
            self.f.gen_brcondi_i32(c, a, i, l);
        }
        self.pending.push((l, skip));
    }

    fn tick(&mut self) {
        let mut i = 0;
        while i < self.pending.len() {
            self.pending[i].1 -= 1;
            if self.pending[i].1 == 0 {
                let (l, _) = self.pending.remove(i);
                self.f.gen_set_label(l);
            } else {
                i += 1;
            }
        }
    }
}

fn build(seed: u64) -> Func {
    build_with(seed, 2, None)
}

/// A random scalar block with `extra` TB temps of each width next to the globals, all of them
/// live to the end, and calls to `call` (a helper like [`helper_add3`]) sprinkled in when given.
/// Many temps and calls make the allocator spill and reload.
fn build_with(seed: u64, extra: usize, call: Option<&HelperInfo>) -> Func {
    let mut f = Func::new(FuncConfig::default());
    let env = f.env();
    let mut v64: Vec<TempI64> =
        (0..N64).map(|i| f.global_mem_new_i64(env, G64 + 8 * i as i64, format!("r{i}"))).collect();
    let mut v32: Vec<TempI32> =
        (0..N32).map(|i| f.global_mem_new_i32(env, G32 + 4 * i as i64, format!("w{i}"))).collect();
    let mut init = Rng::new(seed ^ 0x5eed);
    for i in 0..extra {
        let t = f.temp_new_i64();
        if i < N64 {
            f.gen_mov_i64(t, v64[i]);
        } else {
            f.gen_movi_i64(t, init.interesting() as i64);
        }
        v64.push(t);
        let t = f.temp_new_i32();
        if i < N32 {
            f.gen_mov_i32(t, v32[i]);
        } else {
            f.gen_movi_i32(t, init.interesting() as i32);
        }
        v32.push(t);
    }
    let mut g = Gen { f: &mut f, rng: Rng::new(seed), v64, v32, pending: Vec::new() };
    let n = 5 + g.rng.below(if extra > 2 { 120 } else { 40 });
    for _ in 0..n {
        if let Some(info) = call {
            if g.rng.below(12) == 0 {
                let (r, a, b, c) = (g.a32(), g.a32(), g.a32(), g.a32());
                let h = g.f.helper(info.clone());
                g.f.gen_call(h, Some(r.temp()), &[a.temp(), b.temp(), c.temp()]);
            }
        }
        match g.rng.below(10) {
            0..=4 => g.step64(),
            5..=7 => g.step32(),
            8 => g.branch(),
            _ => {
                let (a, b) = (g.v64[N64], g.v64[N64 + 1]);
                let d = g.v64[g.rng.below(N64 as u64) as usize];
                g.f.gen_xor_i64(d, a, b);
            }
        }
        g.tick();
    }
    let rest: Vec<Label> = g.pending.drain(..).map(|(l, _)| l).collect();
    for l in rest {
        g.f.gen_set_label(l);
    }
    let (a, b) = (g.v64[N64], g.v64[N64 + 1]);
    let (c, d) = (g.v32[N32], g.v32[N32 + 1]);
    let (x, y) = (g.v64[0], g.v32[0]);
    for k in N64 + 2..g.v64.len() {
        let (d, t) = (g.v64[k % N64], g.v64[k]);
        g.f.gen_xor_i64(d, d, t);
    }
    for k in N32 + 2..g.v32.len() {
        let (d, t) = (g.v32[k % N32], g.v32[k]);
        g.f.gen_add_i32(d, d, t);
    }
    g.f.gen_add_i64(x, x, a);
    g.f.gen_xor_i64(x, x, b);
    g.f.gen_add_i32(y, y, c);
    g.f.gen_sub_i32(y, y, d);
    if g.rng.below(3) == 0 {
        let l = g.f.new_label();
        let c = g.cond();
        let (p, q) = (g.a64(), g.a64());
        g.f.gen_brcond_i64(c, p, q, l);
        g.f.gen_exit_tb(0x1000, 0);
        g.f.gen_set_label(l);
        g.f.gen_exit_tb(0x1000, 1);
    } else {
        f.gen_exit_tb(0, 0);
    }
    f
}

#[test]
fn random_blocks_match_the_interpreter() {
    let mut rng = Rng::new(0xa64);
    let reg = HelperRegistry::new();
    let mem = FlatMemory::new(MEM_BASE, 16);
    for _ in 0..1500 {
        let seed = rng.next();
        let mut env = vec![0u8; ENV_SIZE];
        for i in 0..N64 {
            let o = G64 as usize + 8 * i;
            env[o..o + 8].copy_from_slice(&rng.interesting().to_le_bytes());
        }
        for i in 0..N32 {
            let o = G32 as usize + 4 * i;
            env[o..o + 4].copy_from_slice(&(rng.interesting() as u32).to_le_bytes());
        }
        let f = build(seed);
        let (x, _, _) = check(&f, &env, &mem, &reg, [false; 2]);
        assert!(matches!(x, Ok(Exit::ExitTb(_))), "seed {seed:#x}: {x:?}");
    }
}

fn helper_add3(_: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    Ok((a[0] as u32).wrapping_add(a[1] as u32).wrapping_add(a[2] as u32) as u128)
}

fn helper_store_env(e: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
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

    let mut f = Func::new(FuncConfig::default());
    let env = f.env();
    let g = f.global_mem_new_i64(env, 0x100, "g");
    let o = f.global_mem_new_i32(env, 0x108, "o");
    let lo = f.global_mem_new_i64(env, 0x110, "lo");
    let hi = f.global_mem_new_i64(env, 0x118, "hi");
    f.gen_movi_i64(g, 1);
    let h = f.helper(store);
    let v = f.constant_i64(0x55);
    f.gen_call(h, None, &[env.temp(), v.temp()]);
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

    let mem = FlatMemory::new(MEM_BASE, 16);
    let (x, e, _) = check(&f, &[0u8; ENV_SIZE], &mem, &reg, [false; 2]);
    assert_eq!(x, Ok(Exit::ExitTb(0)));
    assert_eq!(rd64(&e, 0x100), 0x56);
    assert_eq!(rd64(&e, 0x108) as u32, u32::MAX);
    assert_eq!(rd64(&e, 0x110), 0x0203_0405_0607_0811);
    assert_eq!(rd64(&e, 0x118), 0x1213_1415_1617_1801);

    // Setup errors are reported the same way.
    let (x, _, _) = check(&f, &[0u8; ENV_SIZE], &mem, &HelperRegistry::empty(), [false; 2]);
    assert!(matches!(x, Err(InterpError::UnknownHelper(_))), "{x:?}");
    let mut bad = HelperRegistry::new();
    bad.register("store_env", HelperType::Void, &[HelperType::Ptr], helper_store_env);
    let (x, _, _) = check(&f, &[0u8; ENV_SIZE], &mem, &bad, [false; 2]);
    assert_eq!(x, Err(InterpError::HelperSignature("store_env".into())));
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
    let mut f = Func::new(FuncConfig::default());
    let env = f.env();
    let g = f.global_mem_new_i64(env, 0x100, "g");
    f.gen_movi_i64(g, 3);
    let h = f.helper(raise);
    let c = f.constant_i32(13);
    f.gen_call(h, None, &[env.temp(), c.temp()]);
    f.gen_movi_i64(g, 4);
    f.gen_exit_tb(0, 0);
    let (x, e, _) = check(&f, &[0u8; ENV_SIZE], &FlatMemory::new(MEM_BASE, 16), &reg, [false; 2]);
    assert_eq!(x, Ok(Exit::Unwind(Unwind::Exception(13))));
    assert_eq!(rd64(&e, 0x100), 3);
}

#[test]
fn guest_memory() {
    let mut f = Func::new(FuncConfig::default());
    let env = f.env();
    let r: Vec<TempI64> =
        (0..6).map(|i| f.global_mem_new_i64(env, 0x100 + 8 * i, format!("r{i}"))).collect();
    let w = f.global_mem_new_i32(env, 0x140, "w");
    let a = f.constant_i64(MEM_BASE as i64);
    f.gen_qemu_ld_i64(r[0], a, 0, MemOp::UQ);
    f.gen_qemu_ld_i64(r[1], a, 1, MemOp::SW.or(MemOp::BE));
    f.gen_qemu_ld_i32(w, a, 2, MemOp::SB);
    let b = f.constant_i64(MEM_BASE as i64 + 16);
    f.gen_qemu_st_i64(r[0], b, 0, MemOp::UQ.or(MemOp::BE));
    f.gen_qemu_st_i32(w, b, 0, MemOp::UW);
    let t = f.temp_new_i128();
    f.gen_qemu_ld_i128(t, a, 0, MemOp::UO);
    let c = f.constant_i64(MEM_BASE as i64 + 32);
    f.gen_qemu_st_i128(t, c, 0, MemOp::UO.or(MemOp::BE));
    f.gen_mov_i64(r[2], t.low());
    f.gen_mov_i64(r[3], t.high());
    // A fault leaves the block with the globals written so far.
    let d = f.constant_i64(MEM_BASE as i64 + 1);
    f.gen_movi_i64(r[4], 7);
    f.gen_qemu_ld_i64(r[4], d, 3, MemOp::UL.or(MemOp::ALIGN));
    f.gen_movi_i64(r[5], 9);
    f.gen_exit_tb(0, 0);

    let mut mem = FlatMemory::new(MEM_BASE, 64);
    for (i, b) in mem.bytes[..16].iter_mut().enumerate() {
        *b = 0x80 | i as u8;
    }
    let (x, e, m) = check(&f, &[0u8; ENV_SIZE], &mem, &HelperRegistry::new(), [false; 2]);
    assert!(matches!(x, Ok(Exit::Unwind(Unwind::Mem(_)))), "{x:?}");
    assert_eq!(rd64(&e, 0x100), 0x8786_8584_8382_8180);
    assert_eq!(rd64(&e, 0x120), 7);
    assert_eq!(rd64(&e, 0x128), 0);
    assert_eq!(m.bytes[16..18], [0x80, 0xff]);
}

#[test]
fn exits_and_goto_tb() {
    let reg = HelperRegistry::new();
    let mem = FlatMemory::new(MEM_BASE, 16);
    let mut f = Func::new(FuncConfig::default());
    f.gen_exit_tb(0x1000, 1);
    assert_eq!(check(&f, &[0u8; 16], &mem, &reg, [false; 2]).0, Ok(Exit::ExitTb(0x1001)));

    let mut f = Func::new(FuncConfig::default());
    f.gen_goto_tb(1);
    f.gen_exit_tb(0x2000, 1);
    assert_eq!(check(&f, &[0u8; 16], &mem, &reg, [false; 2]).0, Ok(Exit::ExitTb(0x2001)));
    assert_eq!(check(&f, &[0u8; 16], &mem, &reg, [false, true]).0, Ok(Exit::GotoTb(1)));

    // Linking can be undone.
    let r = region();
    let tb = compile(&r, &f);
    assert!(tb.set_goto_tb_linked(1, true));
    assert!(!tb.set_goto_tb_linked(0, true));
    assert!(tb.set_goto_tb_linked(1, false));
    if NATIVE {
        let (x, _, _) = native(&tb, &[0u8; 16], &mem, &reg);
        assert_eq!(x, Ok(Exit::ExitTb(0x2001)));
    }

    let mut f = Func::new(FuncConfig::default());
    f.gen_lookup_and_goto_ptr();
    assert_eq!(check(&f, &[0u8; 16], &mem, &reg, [false; 2]).0, Ok(Exit::GotoPtr(0)));
    let mut reg2 = HelperRegistry::new();
    reg2.register_info(&lookup_tb_ptr(), |_, _| Ok(0xabc0));
    assert_eq!(check(&f, &[0u8; 16], &mem, &reg2, [false; 2]).0, Ok(Exit::GotoPtr(0xabc0)));
}

#[test]
fn run_errors_and_insn_start() {
    let reg = HelperRegistry::new();
    let mem = FlatMemory::new(MEM_BASE, 16);
    let mut f = Func::new(FuncConfig::default());
    f.gen_insn_start(&[0x40_0000, 7, 0]);
    let t = f.temp_new_i64();
    f.gen_movi_i64(t, 1);
    f.gen_insn_start(&[0x40_0004, 8, 0]);
    assert_eq!(check(&f, &[0u8; 16], &mem, &reg, [false; 2]).0, Err(InterpError::FellOffEnd));
    if NATIVE {
        let r = region();
        let tb = compile(&r, &f);
        let mut last = None;
        let x = tb.run_traced(&mut [0u8; 16], &mut mem.clone(), &reg, &mut last);
        assert_eq!(x, Err(InterpError::FellOffEnd));
        assert_eq!(last, Some([0x40_0004, 8, 0]));
    }

    // A constant offset beyond env is caught on entry.
    let mut f = Func::new(FuncConfig::default());
    let env = f.env();
    let g = f.global_mem_new_i64(env, 0x1000, "far");
    f.gen_movi_i64(g, 1);
    f.gen_exit_tb(0, 0);
    let x = check(&f, &[0u8; 16], &mem, &reg, [false; 2]).0;
    assert_eq!(x, Err(InterpError::EnvOutOfBounds { offset: 0x1000, len: 8 }));

    // A pointer global is checked where it is used, after the stores before it.
    let mut f = Func::new(FuncConfig::default());
    let env = f.env();
    let base = f.global_mem_new_ptr(env, 0x0, "base");
    let early = f.global_mem_new_i64(env, 0x8, "early");
    let x = f.global_mem_new_i64(base, 0x10, "x");
    f.gen_movi_i64(early, 5);
    f.gen_addi_i64(x, x, 5);
    f.gen_exit_tb(0, 0);
    let mut e = [0u8; 0x40];
    e[0..8].copy_from_slice(&0x18u64.to_le_bytes());
    let (x, e2, _) = check(&f, &e, &mem, &reg, [false; 2]);
    assert_eq!(x, Ok(Exit::ExitTb(0)));
    assert_eq!(rd64(&e2, 0x28), 5);
    e[0..8].copy_from_slice(&0x38u64.to_le_bytes());
    let (x, e2, _) = check(&f, &e, &mem, &reg, [false; 2]);
    assert_eq!(x, Err(InterpError::EnvOutOfBounds { offset: 0x48, len: 8 }));
    assert_eq!(rd64(&e2, 0x8), 5);
    // Pointer arithmetic wraps, as in the interpreter.
    e[0..8].copy_from_slice(&u64::MAX.to_le_bytes());
    let _ = check(&f, &e, &mem, &reg, [false; 2]);
}

/// Where the vector tests keep their vectors in the CPU state: twelve 16-byte slots.
const VEC: i64 = 0x200;
const VEC_SLOTS: u64 = 12;

/// Random vector blocks: V64 and V128 temps loaded from the CPU state, every vector op at every
/// element size, and the results stored back.
struct VGen<'a> {
    f: &'a mut Func,
    rng: Rng,
    v64: Vec<TempVec>,
    v128: Vec<TempVec>,
    s64: Vec<TempI64>,
    s32: Vec<TempI32>,
}

impl VGen<'_> {
    fn pick(&mut self, ty: Type) -> TempVec {
        let pool = if ty == Type::V64 { &self.v64 } else { &self.v128 };
        pool[self.rng.below(pool.len() as u64) as usize]
    }

    fn slot(&mut self) -> i64 {
        VEC + 16 * self.rng.below(VEC_SLOTS) as i64
    }

    fn step(&mut self) {
        let ty = if self.rng.below(2) == 0 { Type::V64 } else { Type::V128 };
        let vece = self.rng.below(4) as u32;
        let bits = 8i64 << vece;
        let (r, a, b) = (self.pick(ty), self.pick(ty), self.pick(ty));
        let k = self.rng.below(46);
        let f = &mut *self.f;
        match k {
            0 => f.gen_add_vec(vece, r, a, b),
            1 => f.gen_sub_vec(vece, r, a, b),
            2 => f.gen_mul_vec(vece, r, a, b),
            3 => f.gen_ssadd_vec(vece, r, a, b),
            4 => f.gen_usadd_vec(vece, r, a, b),
            5 => f.gen_sssub_vec(vece, r, a, b),
            6 => f.gen_ussub_vec(vece, r, a, b),
            7 => f.gen_smin_vec(vece, r, a, b),
            8 => f.gen_umin_vec(vece, r, a, b),
            9 => f.gen_smax_vec(vece, r, a, b),
            10 => f.gen_umax_vec(vece, r, a, b),
            11 => f.gen_and_vec(vece, r, a, b),
            12 => f.gen_or_vec(vece, r, a, b),
            13 => f.gen_xor_vec(vece, r, a, b),
            14 => f.gen_andc_vec(vece, r, a, b),
            15 => f.gen_orc_vec(vece, r, a, b),
            16 => f.gen_nand_vec(vece, r, a, b),
            17 => f.gen_nor_vec(vece, r, a, b),
            18 => f.gen_eqv_vec(vece, r, a, b),
            19 => f.gen_not_vec(vece, r, a),
            20 => f.gen_neg_vec(vece, r, a),
            21 => f.gen_abs_vec(vece, r, a),
            22..=26 => {
                let i = self.rng.below(bits as u64) as i64;
                let f = &mut *self.f;
                match k {
                    22 => f.gen_shli_vec(vece, r, a, i),
                    23 => f.gen_shri_vec(vece, r, a, i),
                    24 => f.gen_sari_vec(vece, r, a, i),
                    25 => f.gen_rotli_vec(vece, r, a, i),
                    _ => f.gen_rotri_vec(vece, r, a, i),
                }
            }
            27..=30 => {
                let n = self.s32[self.rng.below(self.s32.len() as u64) as usize];
                let f = &mut *self.f;
                match k {
                    27 => f.gen_shls_vec(vece, r, a, n),
                    28 => f.gen_shrs_vec(vece, r, a, n),
                    29 => f.gen_sars_vec(vece, r, a, n),
                    _ => f.gen_rotls_vec(vece, r, a, n),
                }
            }
            31 => f.gen_shlv_vec(vece, r, a, b),
            32 => f.gen_shrv_vec(vece, r, a, b),
            33 => f.gen_sarv_vec(vece, r, a, b),
            34 => f.gen_rotlv_vec(vece, r, a, b),
            35 => f.gen_rotrv_vec(vece, r, a, b),
            36 | 37 => {
                let c = CONDS[self.rng.below(CONDS.len() as u64) as usize];
                let b = if k == 37 { self.f.constant_vec(ty, vece, 0) } else { b };
                self.f.gen_cmp_vec(c, vece, r, a, b)
            }
            38 => {
                let c = self.pick(ty);
                self.f.gen_bitsel_vec(vece, r, a, b, c)
            }
            39 => {
                let cnd = CONDS[self.rng.below(CONDS.len() as u64) as usize];
                let (c, d) = (self.pick(ty), self.pick(ty));
                self.f.gen_cmpsel_vec(cnd, vece, r, a, b, c, d)
            }
            40 => {
                let v = self.rng.interesting();
                self.f.gen_dupi_vec(vece, r, v)
            }
            41 => {
                if self.rng.below(2) == 0 {
                    let x = self.s64[self.rng.below(self.s64.len() as u64) as usize];
                    self.f.gen_dup_i64_vec(vece, r, x)
                } else {
                    let x = self.s32[self.rng.below(self.s32.len() as u64) as usize];
                    self.f.gen_dup_i32_vec(vece.min(2), r, x)
                }
            }
            42 => {
                let env = self.f.env();
                let o = self.slot() + (self.rng.below(16) as i64 & !((1 << vece) - 1));
                self.f.gen_dup_mem_vec(vece, r, env, o)
            }
            43 => {
                let env = self.f.env();
                let (o, p) = (self.slot(), self.slot());
                self.f.gen_st_vec(a, env, o);
                self.f.gen_ld_vec(r, env, p)
            }
            44 => {
                // A V64 op on V128 inputs reads their low halves.
                let r = self.pick(Type::V64);
                let (a, b) = (self.pick(Type::V128), self.pick(Type::V128));
                self.f.gen_add_vec(vece, r, a, b)
            }
            _ => f.gen_mov_vec(r, a),
        }
    }
}

/// A random vector block with `n64` V64 and `n128` V128 temps.
fn build_vec(seed: u64, n64: usize, n128: usize) -> Func {
    let mut f = Func::new(FuncConfig::default());
    let env = f.env();
    let s64: Vec<TempI64> =
        (0..N64).map(|i| f.global_mem_new_i64(env, G64 + 8 * i as i64, format!("r{i}"))).collect();
    let s32: Vec<TempI32> =
        (0..N32).map(|i| f.global_mem_new_i32(env, G32 + 4 * i as i64, format!("w{i}"))).collect();
    let mut g =
        VGen { f: &mut f, rng: Rng::new(seed), v64: Vec::new(), v128: Vec::new(), s64, s32 };
    for i in 0..n64 + n128 {
        let ty = if i < n64 { Type::V64 } else { Type::V128 };
        let t = g.f.temp_new_vec(ty);
        let o = g.slot();
        g.f.gen_ld_vec(t, env, o);
        if i < n64 { g.v64.push(t) } else { g.v128.push(t) }
    }
    let n = 5 + g.rng.below(60);
    for _ in 0..n {
        g.step();
    }
    let all: Vec<TempVec> = g.v64.iter().chain(g.v128.iter()).copied().collect();
    for (k, t) in all.into_iter().enumerate() {
        let o = VEC + 16 * (k as i64 % VEC_SLOTS as i64);
        if k % 5 == 4 && g.f.temp(t).base_type == Type::V128 {
            g.f.gen_stl_vec(t, env, o, Type::V64);
        } else {
            g.f.gen_st_vec(t, env, o);
        }
    }
    f.gen_exit_tb(0, 0);
    f
}

fn random_env(rng: &mut Rng) -> Vec<u8> {
    let mut env = vec![0u8; ENV_SIZE];
    for chunk in env.chunks_mut(8) {
        chunk.copy_from_slice(&rng.interesting().to_le_bytes());
    }
    env
}

#[test]
fn random_vector_blocks_match_the_interpreter() {
    let mut rng = Rng::new(0x7ec);
    let reg = HelperRegistry::new();
    let mem = FlatMemory::new(MEM_BASE, 16);
    for _ in 0..1500 {
        let seed = rng.next();
        let env = random_env(&mut rng);
        let f = build_vec(seed, 2, 3);
        let (x, _, _) = check(&f, &env, &mem, &reg, [false; 2]);
        assert_eq!(x, Ok(Exit::ExitTb(0)), "seed {seed:#x}");
    }
}

#[test]
fn register_pressure_spills_match_the_interpreter() {
    let add3 = HelperInfo::new(
        "add3",
        call_flags::NO_RWG | call_flags::NO_SE,
        HelperType::I32,
        &[HelperType::I32, HelperType::I32, HelperType::I32],
    );
    let mut reg = HelperRegistry::new();
    reg.register_info(&add3, helper_add3);
    let mem = FlatMemory::new(MEM_BASE, 16);
    let mut rng = Rng::new(0x5b11);
    for round in 0..300 {
        let seed = rng.next();
        let env = random_env(&mut rng);
        // More live values than the 22 general registers, with and without calls.
        let call = if round % 2 == 0 { Some(&add3) } else { None };
        let f = build_with(seed, 30, call);
        let (x, _, _) = check(&f, &env, &mem, &reg, [false; 2]);
        assert!(matches!(x, Ok(Exit::ExitTb(_))), "seed {seed:#x}: {x:?}");
        // More live vectors than the 22 vector registers.
        let f = build_vec(seed, 12, 16);
        let (x, _, _) = check(&f, &env, &mem, &reg, [false; 2]);
        assert_eq!(x, Ok(Exit::ExitTb(0)), "seed {seed:#x}");
    }
}

#[test]
fn vector_values_are_exact() {
    let mut f = Func::new(FuncConfig::default());
    let env = f.env();
    let (a, b) = (f.temp_new_vec(Type::V128), f.temp_new_vec(Type::V128));
    let r = f.temp_new_vec(Type::V128);
    f.gen_ld_vec(a, env, VEC);
    f.gen_ld_vec(b, env, VEC + 16);
    f.gen_add_vec(0, r, a, b);
    f.gen_st_vec(r, env, VEC + 32);
    f.gen_mul_vec(3, r, a, b);
    f.gen_st_vec(r, env, VEC + 48);
    f.gen_umin_vec(3, r, a, b);
    f.gen_st_vec(r, env, VEC + 64);
    f.gen_exit_tb(0, 0);
    let mut e = vec![0u8; ENV_SIZE];
    let va: [u64; 2] = [0x01ff_0203_0405_0607, 0x8000_0000_0000_0003];
    let vb: [u64; 2] = [0x0101_0101_0101_01ff, 0x0000_0000_0000_0005];
    for k in 0..2 {
        let (o, p) = (VEC as usize + 8 * k, VEC as usize + 16 + 8 * k);
        e[o..o + 8].copy_from_slice(&va[k].to_le_bytes());
        e[p..p + 8].copy_from_slice(&vb[k].to_le_bytes());
    }
    let mem = FlatMemory::new(MEM_BASE, 16);
    let (x, e, _) = check(&f, &e, &mem, &HelperRegistry::new(), [false; 2]);
    assert_eq!(x, Ok(Exit::ExitTb(0)));
    let base = VEC as usize;
    assert_eq!(rd64(&e, base + 32), 0x0200_0304_0506_0706);
    assert_eq!(rd64(&e, base + 48 + 8), 0x8000_0000_0000_000f);
    assert_eq!(rd64(&e, base + 64), 0x0101_0101_0101_01ff);
}

#[test]
fn vector_access_through_a_pointer() {
    let mut f = Func::new(FuncConfig::default());
    let env = f.env();
    let base = f.global_mem_new_ptr(env, 0x0, "base");
    let (v, w) = (f.temp_new_vec(Type::V128), f.temp_new_vec(Type::V64));
    f.gen_ld_vec(v, base, 0x10);
    f.gen_dup_mem_vec(1, w, base, 0x2);
    f.gen_st_vec(v, base, 0x20);
    f.gen_st_vec(w, env, 0x8);
    f.gen_exit_tb(0, 0);
    let mem = FlatMemory::new(MEM_BASE, 16);
    let reg = HelperRegistry::new();
    let mut rng = Rng::new(0xbee);
    let mut e = random_env(&mut rng);
    for (p, ok) in [(0x100u64, true), (0x3d0, true), (0x3e0, false), (u64::MAX - 8, false)] {
        e[0..8].copy_from_slice(&p.to_le_bytes());
        let (x, _, _) = check(&f, &e, &mem, &reg, [false; 2]);
        assert_eq!(x.is_ok(), ok, "{p:#x}: {x:?}");
    }
}

/// A counted loop: a multiply, add and shift chain on two globals and a 128-bit vector add
/// per iteration.
fn hot_loop(iters: i64) -> Func {
    let mut f = Func::new(FuncConfig::default());
    let env = f.env();
    let n = f.global_mem_new_i64(env, G64, "n");
    let acc = f.global_mem_new_i64(env, G64 + 8, "acc");
    let (v, w) = (f.temp_new_vec(Type::V128), f.temp_new_vec(Type::V128));
    let t = f.temp_new_i64();
    f.gen_movi_i64(n, iters);
    f.gen_ld_vec(v, env, VEC);
    f.gen_ld_vec(w, env, VEC + 16);
    let top = f.new_label();
    f.gen_set_label(top);
    f.gen_muli_i64(acc, acc, 0x5851_f42d_4c95_7f2d);
    f.gen_add_i64(acc, acc, n);
    f.gen_shri_i64(t, acc, 29);
    f.gen_xor_i64(acc, acc, t);
    f.gen_add_vec(2, v, v, w);
    f.gen_subi_i64(n, n, 1);
    f.gen_brcondi_i64(Cond::Ne, n, 0, top);
    f.gen_st_vec(v, env, VEC + 32);
    f.gen_exit_tb(0, 0);
    f
}

#[test]
fn hot_loop_native_vs_interpreter() {
    let iters = 200_000;
    let f = hot_loop(iters);
    let mut opt = f.clone();
    opt.gen_code(true, LogMask::default());
    let mut rng = Rng::new(0x100b);
    let env = random_env(&mut rng);
    let mem = FlatMemory::new(MEM_BASE, 16);
    let reg = HelperRegistry::new();
    let t0 = std::time::Instant::now();
    let want = interp(&opt, &env, &mem, &reg, [false; 2]);
    let t_interp = t0.elapsed();
    assert_eq!(want.0, Ok(Exit::ExitTb(0)));
    let tb = compile(&region(), &opt);
    if !NATIVE {
        return;
    }
    let t0 = std::time::Instant::now();
    let got = native(&tb, &env, &mem, &reg);
    let t_native = t0.elapsed();
    assert!(got.0 == want.0 && got.1 == want.1 && got.2 == want.2, "hot loop differs");
    println!(
        "hot loop, {iters} iterations: interpreter {t_interp:?}, native {t_native:?}, \
         speedup {:.1}x",
        t_interp.as_secs_f64() / t_native.as_secs_f64().max(1e-9)
    );
}

#[test]
fn blocks_are_placed_one_after_another() {
    let r = region();
    let mut f = Func::new(FuncConfig::default());
    f.gen_exit_tb(0x40, 0);
    let a = compile(&r, &f);
    let b = compile(&r, &f);
    assert_eq!(a.addr() % 16, 0);
    assert!(b.addr() >= a.addr() + a.code().len() as u64);
    assert!(r.used() <= r.size());
}
