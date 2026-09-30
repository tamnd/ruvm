// SPDX-License-Identifier: MIT OR Apache-2.0

//! Property tests: random blocks give the same result unoptimized, optimized, and optimized with
//! the inputs known as constants.

mod common;

use common::{ENV_SIZE, Rng, run};
use ruvm_jit_core::ir::{FuncConfig, TempI32, TempI64, TempVec};
use ruvm_jit_core::liveness::LogMask;
use ruvm_jit_core::types::bswap;
use ruvm_jit_core::{Cond, Func, Label, Type};

const N64: usize = 6;
const N32: usize = 4;
const G64: i64 = 0x100;
const G32: i64 = 0x180;
const SCRATCH: i64 = 0x300;
const VEC: i64 = 0x200;

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

    fn step64(&mut self) {
        let (r, a, b) = (self.a64(), self.a64(), self.a64());
        let k = self.rng.below(40);
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
            _ => {
                // A round trip through a host memory slot.
                let env = self.f.env();
                let o = SCRATCH + 8 * self.rng.below(4) as i64;
                self.f.gen_st_i64(a, env, o);
                match self.rng.below(3) {
                    0 => self.f.gen_ld_i64(r, env, o),
                    1 => self.f.gen_ld8s_i64(r, env, o + 1),
                    _ => self.f.gen_ld32u_i64(r, env, o + 4),
                }
            }
        }
    }

    fn distinct64(&mut self, r: TempI64) -> TempI64 {
        loop {
            let t = self.a64();
            if t != r {
                return t;
            }
        }
    }

    fn step32(&mut self) {
        let (r, a, b) = (self.a32(), self.a32(), self.a32());
        let k = self.rng.below(30);
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
                let r2 = loop {
                    let t = self.a32();
                    if t != r {
                        break t;
                    }
                };
                match self.rng.below(3) {
                    0 => self.f.gen_mulu2_i32(r, r2, a, b),
                    1 => self.f.gen_muls2_i32(r, r2, a, b),
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

/// Build a random block. With `consts`, the inputs are also moved into the globals as
/// constants first, so the optimizer knows them. The same seed gives the same block either way.
fn build(seed: u64, init64: &[u64], init32: &[u32], consts: bool) -> Func {
    let mut f = Func::new(FuncConfig::default());
    let env = f.env();
    let mut v64: Vec<TempI64> =
        (0..N64).map(|i| f.global_mem_new_i64(env, G64 + 8 * i as i64, &format!("r{i}"))).collect();
    let mut v32: Vec<TempI32> =
        (0..N32).map(|i| f.global_mem_new_i32(env, G32 + 4 * i as i64, &format!("w{i}"))).collect();
    if consts {
        for (t, &v) in v64.iter().zip(init64) {
            f.gen_movi_i64(*t, v as i64);
        }
        for (t, &v) in v32.iter().zip(init32) {
            f.gen_movi_i32(*t, v as i32);
        }
    }
    // Two TB temps of each width, seeded from the globals.
    for i in 0..2 {
        let t = f.temp_new_i64();
        f.gen_mov_i64(t, v64[i]);
        v64.push(t);
        let t = f.temp_new_i32();
        f.gen_mov_i32(t, v32[i]);
        v32.push(t);
    }
    let mut g = Gen { f: &mut f, rng: Rng::new(seed), v64, v32, pending: Vec::new() };
    let n = 5 + g.rng.below(40);
    for _ in 0..n {
        match g.rng.below(10) {
            0..=4 => g.step64(),
            5..=7 => g.step32(),
            8 => g.branch(),
            _ => {
                // Fold a TB temp back into a global so its value is observed.
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
    // Observe the TB temps too.
    let (a, b) = (g.v64[N64], g.v64[N64 + 1]);
    let (c, d) = (g.v32[N32], g.v32[N32 + 1]);
    let (x, y) = (g.v64[0], g.v32[0]);
    g.f.gen_add_i64(x, x, a);
    g.f.gen_xor_i64(x, x, b);
    g.f.gen_add_i32(y, y, c);
    g.f.gen_sub_i32(y, y, d);
    let exit = g.rng.below(3);
    if exit == 0 {
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
fn optimized_matches_unoptimized() {
    let mut rng = Rng::new(0x5eed);
    let (mut ops_before, mut ops_after) = (0usize, 0usize);
    for iter in 0..3000u64 {
        let seed = rng.next();
        let init64: Vec<u64> = (0..N64).map(|_| rng.interesting()).collect();
        let init32: Vec<u32> = (0..N32).map(|_| rng.interesting() as u32).collect();
        let mut env = vec![0u8; ENV_SIZE];
        for (i, v) in init64.iter().enumerate() {
            let o = G64 as usize + 8 * i;
            env[o..o + 8].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in init32.iter().enumerate() {
            let o = G32 as usize + 4 * i;
            env[o..o + 4].copy_from_slice(&v.to_le_bytes());
        }

        let base = build(seed, &init64, &init32, false);
        if let Err(e) = base.verify() {
            panic!("iteration {iter}: verify: {e:?}\n{}", base.dump_ops(false));
        }
        let want = run(&base, &env);

        let mut opt = base.clone();
        opt.gen_code(true, LogMask::default());
        let got = run(&opt, &env);
        assert!(
            got == want,
            "iteration {iter} seed {seed:#x}: optimized run differs\nbefore:\n{}\nafter:\n{}",
            base.dump_ops(false),
            opt.dump_ops(true)
        );

        let mut noopt = base.clone();
        noopt.gen_code(false, LogMask::default());
        assert!(run(&noopt, &env) == want, "iteration {iter}: liveness alone changed the result");

        let known = build(seed, &init64, &init32, true);
        let mut kopt = known.clone();
        kopt.gen_code(true, LogMask::default());
        ops_before += known.nb_ops();
        ops_after += kopt.nb_ops();
        let got = run(&kopt, &env);
        assert!(
            got == want,
            "iteration {iter} seed {seed:#x}: constant inputs differ\nbefore:\n{}\nafter:\n{}",
            known.dump_ops(false),
            kopt.dump_ops(true)
        );
    }
    // With every input known, much of each block folds away. Labels end what the optimizer
    // knows, as in QEMU, so not all of it does.
    assert!(ops_after * 4 < ops_before * 3, "only {ops_before} -> {ops_after} ops");
}

/// A random block of vector ops over four vectors loaded from env, stored back at the end.
fn build_vec(seed: u64, ty: Type) -> Func {
    let mut f = Func::new(FuncConfig::default());
    let env = f.env();
    let g = f.global_mem_new_i64(env, G64, "r0");
    let mut rng = Rng::new(seed);
    let v: Vec<TempVec> = (0..4).map(|_| f.temp_new_vec(ty)).collect();
    for (i, &t) in v.iter().enumerate() {
        f.gen_ld_vec(t, env, VEC + 32 * i as i64);
    }
    let pick = |rng: &mut Rng| v[rng.below(4) as usize];
    for _ in 0..(5 + rng.below(20)) {
        let vece = rng.below(4) as u32;
        let (r, a, b) = (pick(&mut rng), pick(&mut rng), pick(&mut rng));
        match rng.below(20) {
            0 => f.gen_add_vec(vece, r, a, b),
            1 => f.gen_sub_vec(vece, r, a, b),
            2 => f.gen_and_vec(vece, r, a, b),
            3 => f.gen_or_vec(vece, r, a, b),
            4 => f.gen_xor_vec(vece, r, a, b),
            5 => f.gen_andc_vec(vece, r, a, b),
            6 => f.gen_not_vec(vece, r, a),
            7 => f.gen_neg_vec(vece, r, a),
            8 => f.gen_shli_vec(vece, r, a, rng.below(8u64 << vece) as i64),
            9 => f.gen_sari_vec(vece, r, a, rng.below(8u64 << vece) as i64),
            10 => {
                let c = CONDS[2 + rng.below(12) as usize];
                // The tst conditions are not vector conditions.
                let c = if c.is_tst() { Cond::Eq } else { c };
                f.gen_cmp_vec(c, vece, r, a, b)
            }
            11 => {
                let c = pick(&mut rng);
                f.gen_bitsel_vec(vece, r, a, b, c)
            }
            12 => f.gen_dupi_vec(vece, r, rng.interesting()),
            13 => f.gen_dup_i64_vec(vece, r, g),
            14 => {
                let k = f.constant_vec(ty, vece, rng.interesting() as i64);
                f.gen_and_vec(vece, r, a, k)
            }
            15 => {
                let k = f.constant_vec(ty, vece, rng.interesting() as i64);
                f.gen_or_vec(vece, r, k, b)
            }
            16 => f.gen_umax_vec(vece, r, a, b),
            17 => f.gen_ssadd_vec(vece, r, a, b),
            18 => f.gen_mov_vec(r, a),
            _ => f.gen_rotlv_vec(vece, r, a, b),
        }
    }
    for (i, &t) in v.iter().enumerate() {
        f.gen_st_vec(t, env, VEC + 32 * i as i64);
    }
    f.gen_exit_tb(0, 0);
    f
}

#[test]
fn vector_optimized_matches_unoptimized() {
    let mut rng = Rng::new(0xfeed);
    for iter in 0..1500u64 {
        let seed = rng.next();
        let ty = [Type::V64, Type::V128, Type::V256][rng.below(3) as usize];
        let mut env = vec![0u8; ENV_SIZE];
        for b in env[G64 as usize..].iter_mut() {
            *b = rng.next() as u8;
        }
        let base = build_vec(seed, ty);
        if let Err(e) = base.verify() {
            panic!("iteration {iter}: verify: {e:?}\n{}", base.dump_ops(false));
        }
        let want = run(&base, &env);
        let mut opt = base.clone();
        opt.gen_code(true, LogMask::default());
        assert!(
            run(&opt, &env) == want,
            "iteration {iter}: optimized vector run differs\nbefore:\n{}\nafter:\n{}",
            base.dump_ops(false),
            opt.dump_ops(true)
        );
    }
}
