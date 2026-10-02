// SPDX-License-Identifier: GPL-2.0-or-later

//! The SVE and SVE2 translator: QEMU's `target/arm/tcg/translate-sve.c` for the integer,
//! predicate, permute, element count and memory groups.
//!
//! The decoder is generated from QEMU's `sve.decode`. Every data processing instruction is
//! one call of the `sve` helper ([`super::super::sve_helper`]) and every load and store one
//! call of `sve_mem`, with a descriptor naming the operation, the registers and the vector
//! length, which is a constant of the translation block as in QEMU. The checks QEMU does
//! before `sve_access_check()` (the element sizes and the ID register features) are done in
//! the same order, so the unallocated encodings and the traps are the same. The instructions
//! that only move constants or general registers (RDVL, ADDVL, ADDPL, CNT, INC, DEC and the
//! immediate DUPs) are expanded inline.
//!
//! Differences from QEMU:
//!
//! - The floating point SVE instructions, the widening, narrowing, pairwise and long SVE2
//!   integer groups, the indexed multiplies, the dot products, the complex integer
//!   instructions, the SVE2 crypto instructions (AESE, AESD, AESMC, AESIMC, SM4E, SM4EKEY,
//!   RAX1), DUP (indexed) of a quadword, the SVE2.1 and SME forms, BF16 and I8MM are not
//!   implemented and raise an Undefined Instruction exception. To match, the `max` model
//!   clears the ID_AA64ZFR0_EL1 AES, SHA3, SM4, BF16, I8MM, F32MM and F64MM fields that
//!   QEMU's `max` sets (and the `sve_aes`, `sve_sha3` and `sve_sm4` feature flags); they
//!   come back when the instructions land.
//! - LDR and STR of a vector or predicate register do not check the alignment when
//!   SCTLR_ELx.A asks for it.
//! - The bits of a Z register above the vector length are zeroed by every write, as they are
//!   in QEMU; the predicate registers likewise.

use ruvm_jit_core::ir::TempI64;

use super::S;
use crate::cpu::EXCP_UDEF;
use crate::syndrome::syn_sve_access_trap;
use crate::tcg::sve_helper::{Dsc, SVE, SVE_MEM, b, c, fam, mm, pl, pm, pr, pred_count, r, u};
use crate::tcg::vfp::vfp_expand_imm;

#[allow(missing_docs, unreachable_pub, dead_code, clippy::pedantic, clippy::nursery)]
mod decode {
    include!(concat!(env!("OUT_DIR"), "/sve_decode.rs"));
}

use decode::{
    arg_disas_sve30, arg_disas_sve31, arg_disas_sve32, arg_disas_sve33, arg_disas_sve34,
    arg_disas_sve35, arg_disas_sve36, arg_disas_sve37, arg_disas_sve39, arg_disas_sve40,
    arg_disas_sve41, arg_disas_sve43, arg_disas_sve45, arg_disas_sve47, arg_disas_sve54,
    arg_incdec_cnt, arg_incdec_pred, arg_incdec2_cnt, arg_incdec2_pred, arg_ptrue, arg_rpr_esz,
    arg_rpr_s, arg_rpri_esz, arg_rpri_gather_load, arg_rpri_load, arg_rpri_scatter_store,
    arg_rpri_store, arg_rprr_esz, arg_rprr_gather_load, arg_rprr_load, arg_rprr_s,
    arg_rprr_scatter_store, arg_rprr_store, arg_rprrr_esz, arg_rr_dbm, arg_rr_esz, arg_rri,
    arg_rri_esz, arg_rrr_esz, arg_rrri, arg_rrri_esz, arg_rrrr_esz, arg_while,
};

pub(super) fn disas(s: &mut S<'_, '_>, insn: u32) -> bool {
    decode::disas_sve(s, insn)
}

// The element size checks of QEMU's helper tables.

fn any(_: i32) -> bool {
    true
}

/// No 64-bit elements.
fn no_d(e: i32) -> bool {
    e != 3
}

/// No 8-bit elements.
fn no_b(e: i32) -> bool {
    e >= 1
}

/// 32 or 64-bit elements.
fn sd(e: i32) -> bool {
    e >= 2
}

fn only_d(e: i32) -> bool {
    e == 3
}

fn only_s(e: i32) -> bool {
    e == 2
}

/// 8 or 16-bit elements.
fn bh(e: i32) -> bool {
    e <= 1
}

/// A valid `tszimm` encoding.
fn tsz(e: i32) -> bool {
    e >= 0
}

/// `dtype_mop[]`: the memory element size and signedness of a load `dtype`.
fn dtype_mop(dtype: i32) -> (u32, bool) {
    const T: [(u32, bool); 16] = [
        (0, false),
        (0, false),
        (0, false),
        (0, false),
        (2, true),
        (1, false),
        (1, false),
        (1, false),
        (1, true),
        (1, true),
        (2, false),
        (2, false),
        (0, true),
        (0, true),
        (0, true),
        (3, false),
    ];
    T[dtype as usize & 15]
}

/// `dtype_esz[]`.
fn dtype_esz(dtype: i32) -> i32 {
    const T: [i32; 16] = [0, 1, 2, 3, 3, 1, 2, 3, 3, 2, 2, 3, 3, 2, 1, 3];
    T[dtype as usize & 15]
}

/// Replicate the low `8 << esz` bits of `v` across 64 bits.
fn rep(esz: i32, v: u64) -> u64 {
    match esz {
        0 => (v & 0xff) * 0x0101_0101_0101_0101,
        1 => (v & 0xffff) * 0x0001_0001_0001_0001,
        2 => (v & 0xffff_ffff) * 0x0000_0001_0000_0001,
        _ => v,
    }
}

impl S<'_, '_> {
    /// `sve_access_check()`: raise the SVE access trap if CPACR_EL1.ZEN, CPTR_EL2 or
    /// CPTR_EL3 say so, then do the FP access check. Returns true if the instruction should
    /// be translated.
    pub(super) fn sve_access_check(&mut self) -> bool {
        if self.d.sve_excp_el != 0 {
            let el = self.d.sve_excp_el;
            self.gen_exception_insn_el(0, EXCP_UDEF, syn_sve_access_trap(), el);
            return false;
        }
        self.fp_access_check()
    }

    /// Translate: false if not allocated, else the access check and `f`.
    fn sve_gen(&mut self, ok: bool, f: impl FnOnce(&mut Self)) -> bool {
        if !ok {
            return false;
        }
        if self.sve_access_check() {
            f(self);
        }
        true
    }

    /// Call the `sve` helper; `r` holds Rd, Rn, Rm, Ra and Pg. Returns its scalar result.
    #[allow(clippy::too_many_arguments)]
    fn sv(
        &mut self,
        fm: u32,
        op: u32,
        esz: i32,
        r: [i32; 5],
        data: u32,
        x: Option<TempI64>,
        y: Option<TempI64>,
    ) -> TempI64 {
        let dsc = Dsc::pack(fm, op, esz, self.d.vl, r, data);
        let env = self.env();
        let dsc = self.c64(dsc as i64);
        let x = match x {
            Some(t) => t,
            None => self.c64(0),
        };
        let y = match y {
            Some(t) => t,
            None => self.c64(0),
        };
        let ret = self.new64();
        self.call(&SVE, Some(ret.into()), &[env.into(), dsc.into(), x.into(), y.into()]);
        ret
    }

    /// [`Self::sv`] with an immediate operand.
    fn svi(&mut self, fm: u32, op: u32, esz: i32, r: [i32; 5], data: u32, imm: i64) {
        let x = self.c64(imm);
        self.sv(fm, op, esz, r, data, Some(x), None);
    }

    /// Call the `sve_mem` helper on the address `addr`.
    fn svm(&mut self, op: u32, esz: i32, r: [i32; 5], data: u32, addr: TempI64) {
        let data = data | (self.get_mem_index() << 16);
        let dsc = Dsc::pack(0, op, esz, self.d.vl, r, data);
        let env = self.env();
        let dsc = self.c64(dsc as i64);
        let y = self.c64(0);
        let ret = self.new64();
        self.call(&SVE_MEM, Some(ret.into()), &[env.into(), dsc.into(), addr.into(), y.into()]);
    }

    /// Store the 64-bit pattern `v` to every word of Zd.
    fn dup_const(&mut self, rd: i32, v: u64) {
        let t = self.c64(v as i64);
        self.dup_t(rd, t, None);
    }

    /// Store `lo` (and `hi` in the odd words, if given) to every word of Zd.
    fn dup_t(&mut self, rd: i32, lo: TempI64, hi: Option<TempI64>) {
        let off = crate::cpu::vreg_off(rd as usize);
        for i in 0..self.d.vl as usize / 8 {
            let t = match hi {
                Some(h) if i & 1 == 1 => h,
                _ => lo,
            };
            self.st_env64(t, off + 8 * i);
        }
    }

    /// `decode_pred_count()` for the vector length.
    fn numelem(&self, pat: i32, esz: i32) -> i64 {
        pred_count(self.d.vl as usize, pat as u32, esz as u32) as i64
    }

    /// `do_sat_addsub_vec()`: Zd = Zn + val (or - val, with `d`), saturating.
    fn sat_vec(&mut self, esz: i32, rd: i32, rn: i32, val: TempI64, uns: bool, d: bool) {
        let (op, v) = if uns && esz == 3 {
            (if d { b::UQSUBI } else { b::UQADDI }, val)
        } else {
            let v = if d {
                let t = self.new64();
                let z = self.c64(0);
                self.f().gen_sub_i64(t, z, val);
                t
            } else {
                val
            };
            (if uns { b::UQADDI } else { b::SQADDI }, v)
        };
        self.sv(fam::ZZI, op, esz, [rd, rn, 0, 0, 0], 0, Some(v), None);
    }

    /// Zd = Zn + val (or - val, with `d`), wrapping.
    fn add_vec(&mut self, esz: i32, rd: i32, rn: i32, val: TempI64, d: bool) {
        let op = if d { b::SUB } else { b::ADD };
        self.sv(fam::ZZI, op, esz, [rd, rn, 0, 0, 0], 0, Some(val), None);
    }

    /// `do_sat_addsub_32()` and `do_sat_addsub_64()` on Xd.
    fn sat_reg(&mut self, rd: i32, val: TempI64, uns: bool, d: bool, sf: bool) {
        let x = self.reg(rd);
        let data = u32::from(uns) | u32::from(d) << 1 | u32::from(sf) << 2;
        let t = self.sv(fam::PRED, pr::SATR, 0, [0; 5], data, Some(x), Some(val));
        self.set_reg(rd, t);
    }

    /// `do_cntp()`: the number of active elements of Pn under Pg.
    fn cntp(&mut self, esz: i32, rn: i32, pg: i32) -> TempI64 {
        self.sv(fam::PRED, pr::CNTP, esz, [0, rn, 0, 0, pg], 0, None, None)
    }

    /// The contiguous address `Xn|SP + (Xm << msz)`, or `Xn|SP + imm` with `rm` None.
    fn cont_addr(&mut self, rn: i32, rm: Option<i32>, msz: u32, imm: i64) -> TempI64 {
        let a = self.reg_sp(rn);
        match rm {
            Some(rm) => {
                let m = self.reg(rm);
                self.f().gen_shli_i64(m, m, i64::from(msz));
                self.f().gen_add_i64(a, a, m);
            }
            None => self.f().gen_addi_i64(a, a, imm),
        }
        self.clean_data_tbi(a)
    }

    /// `do_ld_zpa()` and the first fault and non fault loads.
    fn ld_cont(&mut self, rd: i32, pg: i32, addr: TempI64, dtype: i32, nreg: i32, fault: u32) {
        let (msz, sign) = dtype_mop(dtype);
        let data = msz | u32::from(sign) << 2 | (nreg as u32) << 3 | fault << 5;
        self.svm(mm::LD, dtype_esz(dtype), [rd, 0, 0, 0, pg], data, addr);
    }

    /// The gathers and scatters; `x` is the scalar part of the address.
    fn gather(&mut self, op: u32, esz: i32, r: [i32; 5], data: u32, x: TempI64) {
        self.svm(op, esz, r, data, x);
    }
}

/// The SVE data processing instructions of one helper family, all with the same arguments.
macro_rules! zpzz {
    ($($name:ident: $feat:ident, $op:expr, $chk:expr, $data:expr;)*) => {$(
        fn $name(&mut self, a: &mut arg_rprr_esz) -> bool {
            let ok = self.feat().$feat && $chk(a.esz);
            self.sve_gen(ok, |s| {
                s.sv(fam::ZPZZ, $op, a.esz, [a.rd, a.rn, a.rm, 0, a.pg], $data, None, None);
            })
        }
    )*};
}

macro_rules! zzz {
    ($($name:ident: $feat:ident, $op:expr, $chk:expr, $data:expr;)*) => {$(
        fn $name(&mut self, a: &mut arg_rrr_esz) -> bool {
            let ok = self.feat().$feat && $chk(a.esz);
            self.sve_gen(ok, |s| {
                s.sv(fam::ZZZ, $op, a.esz, [a.rd, a.rn, a.rm, 0, 0], $data, None, None);
            })
        }
    )*};
}

macro_rules! zpz {
    ($($name:ident: $feat:ident, $fam:expr, $op:expr, $chk:expr;)*) => {$(
        fn $name(&mut self, a: &mut arg_rpr_esz) -> bool {
            let ok = self.feat().$feat && $chk(a.esz);
            self.sve_gen(ok, |s| {
                s.sv($fam, $op, a.esz, [a.rd, a.rn, 0, 0, a.pg], 0, None, None);
            })
        }
    )*};
}

macro_rules! zpzi {
    ($($name:ident: $feat:ident, $op:expr;)*) => {$(
        fn $name(&mut self, a: &mut arg_rpri_esz) -> bool {
            let ok = self.feat().$feat && a.esz >= 0;
            self.sve_gen(ok, |s| {
                s.svi(fam::ZPZI, $op, a.esz, [a.rd, a.rn, 0, 0, a.pg], 0, i64::from(a.imm));
            })
        }
    )*};
}

macro_rules! zzi {
    ($($name:ident: $feat:ident, $op:expr, $chk:expr;)*) => {$(
        fn $name(&mut self, a: &mut arg_rri_esz) -> bool {
            let ok = self.feat().$feat && $chk(a.esz);
            self.sve_gen(ok, |s| {
                s.svi(fam::ZZI, $op, a.esz, [a.rd, a.rn, 0, 0, 0], 0, i64::from(a.imm));
            })
        }
    )*};
}

macro_rules! zzi_sat {
    ($($name:ident: $u:expr, $d:expr;)*) => {$(
        fn $name(&mut self, a: &mut arg_rri_esz) -> bool {
            let ok = self.feat().sve;
            self.sve_gen(ok, |s| {
                let v = s.c64(i64::from(a.imm));
                s.sat_vec(a.esz, a.rd, a.rn, v, $u, $d);
            })
        }
    )*};
}

macro_rules! cmp {
    ($($zz:ident, $zw:ident, $zi:ident: $op:expr;)*) => {$(
        fn $zz(&mut self, a: &mut arg_rprr_esz) -> bool {
            let ok = self.feat().sve;
            self.sve_gen(ok, |s| {
                s.sv(fam::CMP, $op, a.esz, [a.rd, a.rn, a.rm, 0, a.pg], 0, None, None);
            })
        }
        fn $zw(&mut self, a: &mut arg_rprr_esz) -> bool {
            let ok = self.feat().sve && a.esz != 3;
            self.sve_gen(ok, |s| {
                s.sv(fam::CMP, $op, a.esz, [a.rd, a.rn, a.rm, 0, a.pg], 1, None, None);
            })
        }
        fn $zi(&mut self, a: &mut arg_rpri_esz) -> bool {
            let ok = self.feat().sve;
            self.sve_gen(ok, |s| {
                s.svi(fam::CMP, $op, a.esz, [a.rd, a.rn, 0, 0, a.pg], 2, i64::from(a.imm));
            })
        }
    )*};
}

macro_rules! pppp {
    ($($name:ident: $op:expr;)*) => {$(
        fn $name(&mut self, a: &mut arg_rprr_s) -> bool {
            let ok = self.feat().sve;
            self.sve_gen(ok, |s| {
                let data = a.s as u32;
                s.sv(fam::PPPP, $op, 0, [a.rd, a.rn, a.rm, 0, a.pg], data, None, None);
            })
        }
    )*};
}

macro_rules! brk {
    ($($name:ident: $data:expr;)*) => {$(
        fn $name(&mut self, a: &mut arg_rpr_s) -> bool {
            let ok = self.feat().sve;
            self.sve_gen(ok, |s| {
                let data = a.s as u32 | $data;
                s.sv(fam::PRED, pr::BRK, 0, [a.rd, a.rn, 0, 0, a.pg], data, None, None);
            })
        }
    )*};
}

macro_rules! perm3 {
    ($($name:ident: $feat:ident, $fam:expr, $op:expr, $chk:expr;)*) => {$(
        fn $name(&mut self, a: &mut arg_rrr_esz) -> bool {
            let ok = self.feat().$feat && $chk(a.esz);
            self.sve_gen(ok, |s| {
                s.sv($fam, $op, a.esz, [a.rd, a.rn, a.rm, 0, 0], 0, None, None);
            })
        }
    )*};
}

macro_rules! zzzz {
    ($($name:ident: $fam:expr, $op:expr;)*) => {$(
        fn $name(&mut self, a: &mut arg_rrrr_esz) -> bool {
            let ok = self.feat().sve2;
            self.sve_gen(ok, |s| {
                s.sv($fam, $op, a.esz, [a.rd, a.rn, a.rm, a.ra, 0], 2, None, None);
            })
        }
    )*};
}

macro_rules! last {
    ($($name:ident: $op:expr, $data:expr;)*) => {$(
        fn $name(&mut self, a: &mut arg_rpr_esz) -> bool {
            let ok = self.feat().sve;
            self.sve_gen(ok, |s| {
                let to_v = $data & 1 != 0;
                let x = if $op == pm::CLAST && !to_v { Some(s.reg(a.rd)) } else { None };
                let t = s.sv(fam::PERM, $op, a.esz, [a.rd, a.rn, 0, 0, a.pg], $data, x, None);
                if !to_v {
                    s.set_reg(a.rd, t);
                }
            })
        }
    )*};
}

impl decode::DisasSve for S<'_, '_> {
    // The field functions of sve.decode.

    fn plus_8(&mut self, x: i32) -> i32 {
        x + 8
    }

    fn plus_1(&mut self, x: i32) -> i32 {
        x + 1
    }

    fn plus_12(&mut self, x: i32) -> i32 {
        x + 12
    }

    fn times_2(&mut self, x: i32) -> i32 {
        x * 2
    }

    fn times_4(&mut self, x: i32) -> i32 {
        x * 4
    }

    fn expand_imm_sh8s(&mut self, x: i32) -> i32 {
        let v = i32::from(x as u8 as i8);
        if x & 0x100 != 0 { v << 8 } else { v }
    }

    fn expand_imm_sh8u(&mut self, x: i32) -> i32 {
        let v = i32::from(x as u8);
        if x & 0x100 != 0 { v << 8 } else { v }
    }

    fn tszimm_esz(&mut self, x: i32) -> i32 {
        31 - ((x as u32) >> 3).leading_zeros() as i32
    }

    fn tszimm_shr(&mut self, x: i32) -> i32 {
        let esz = self.tszimm_esz(x);
        if esz < 0 { 0 } else { (16 << esz) - x }
    }

    fn tszimm_shl(&mut self, x: i32) -> i32 {
        let esz = self.tszimm_esz(x);
        if esz < 0 { 0 } else { x - (8 << esz) }
    }

    fn msz_dtype(&mut self, x: i32) -> i32 {
        [0, 5, 10, 15, 18][x as usize & 3]
    }

    // SVE Integer Arithmetic - Binary Predicated Group.

    zpzz! {
        trans_AND_zpzz: sve, b::AND, any, 0;
        trans_ORR_zpzz: sve, b::ORR, any, 0;
        trans_EOR_zpzz: sve, b::EOR, any, 0;
        trans_BIC_zpzz: sve, b::BIC, any, 0;
        trans_ADD_zpzz: sve, b::ADD, any, 0;
        trans_SUB_zpzz: sve, b::SUB, any, 0;
        trans_SMAX_zpzz: sve, b::SMAX, any, 0;
        trans_UMAX_zpzz: sve, b::UMAX, any, 0;
        trans_SMIN_zpzz: sve, b::SMIN, any, 0;
        trans_UMIN_zpzz: sve, b::UMIN, any, 0;
        trans_SABD_zpzz: sve, b::SABD, any, 0;
        trans_UABD_zpzz: sve, b::UABD, any, 0;
        trans_MUL_zpzz: sve, b::MUL, any, 0;
        trans_SMULH_zpzz: sve, b::SMULH, any, 0;
        trans_UMULH_zpzz: sve, b::UMULH, any, 0;
        trans_SDIV_zpzz: sve, b::SDIV, sd, 0;
        trans_UDIV_zpzz: sve, b::UDIV, sd, 0;
        trans_ASR_zpzz: sve, b::ASR, any, 0;
        trans_LSR_zpzz: sve, b::LSR, any, 0;
        trans_LSL_zpzz: sve, b::LSL, any, 0;
        trans_ASR_zpzw: sve, b::ASR, no_d, 1;
        trans_LSR_zpzw: sve, b::LSR, no_d, 1;
        trans_LSL_zpzw: sve, b::LSL, no_d, 1;
        trans_SADALP_zpzz: sve2, b::SADALP, no_b, 0;
        trans_UADALP_zpzz: sve2, b::UADALP, no_b, 0;
        trans_SRSHL: sve2, b::SRSHL, any, 0;
        trans_URSHL: sve2, b::URSHL, any, 0;
        trans_SQSHL: sve2, b::SQSHL, any, 0;
        trans_UQSHL: sve2, b::UQSHL, any, 0;
        trans_SQRSHL: sve2, b::SQRSHL, any, 0;
        trans_UQRSHL: sve2, b::UQRSHL, any, 0;
        trans_SHADD: sve2, b::SHADD, any, 0;
        trans_UHADD: sve2, b::UHADD, any, 0;
        trans_SHSUB: sve2, b::SHSUB, any, 0;
        trans_UHSUB: sve2, b::UHSUB, any, 0;
        trans_SRHADD: sve2, b::SRHADD, any, 0;
        trans_URHADD: sve2, b::URHADD, any, 0;
        trans_SQADD_zpzz: sve2, b::SQADD, any, 0;
        trans_UQADD_zpzz: sve2, b::UQADD, any, 0;
        trans_SQSUB_zpzz: sve2, b::SQSUB, any, 0;
        trans_UQSUB_zpzz: sve2, b::UQSUB, any, 0;
        trans_SUQADD: sve2, b::SUQADD, any, 0;
        trans_USQADD: sve2, b::USQADD, any, 0;
    }

    fn trans_MLA(&mut self, a: &mut arg_rprrr_esz) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.sv(fam::ZPZZ, b::MLA, a.esz, [a.rd, a.rn, a.rm, a.ra, a.pg], 2, None, None);
        })
    }

    fn trans_MLS(&mut self, a: &mut arg_rprrr_esz) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.sv(fam::ZPZZ, b::MLS, a.esz, [a.rd, a.rn, a.rm, a.ra, a.pg], 2, None, None);
        })
    }

    // SVE Integer Arithmetic - Unpredicated Group, and the SVE2 unpredicated multiplies.

    zzz! {
        trans_ADD_zzz: sve, b::ADD, any, 0;
        trans_SUB_zzz: sve, b::SUB, any, 0;
        trans_SQADD_zzz: sve, b::SQADD, any, 0;
        trans_UQADD_zzz: sve, b::UQADD, any, 0;
        trans_SQSUB_zzz: sve, b::SQSUB, any, 0;
        trans_UQSUB_zzz: sve, b::UQSUB, any, 0;
        trans_AND_zzz: sve, b::AND, any, 0;
        trans_ORR_zzz: sve, b::ORR, any, 0;
        trans_EOR_zzz: sve, b::EOR, any, 0;
        trans_BIC_zzz: sve, b::BIC, any, 0;
        trans_ASR_zzw: sve, b::ASR, no_d, 1;
        trans_LSR_zzw: sve, b::LSR, no_d, 1;
        trans_LSL_zzw: sve, b::LSL, no_d, 1;
        trans_MUL_zzz: sve2, b::MUL, any, 0;
        trans_SMULH_zzz: sve2, b::SMULH, any, 0;
        trans_UMULH_zzz: sve2, b::UMULH, any, 0;
        trans_PMUL_zzz: sve2, b::PMUL, any, 0;
        trans_SQDMULH_zzz: sve2, b::SQDMULH, any, 0;
        trans_SQRDMULH_zzz: sve2, b::SQRDMULH, any, 0;
        trans_SABA: sve2, b::SABA, any, 0;
        trans_UABA: sve2, b::UABA, any, 0;
        trans_BEXT: sve_bitperm, b::BEXT, any, 0;
        trans_BDEP: sve_bitperm, b::BDEP, any, 0;
        trans_BGRP: sve_bitperm, b::BGRP, any, 0;
    }

    zzzz! {
        trans_SQRDMLAH_zzzz: fam::ZZZ, b::SQRDMLAH;
        trans_SQRDMLSH_zzzz: fam::ZZZ, b::SQRDMLSH;
        trans_EOR3: fam::PERM, pm::EOR3;
        trans_BCAX: fam::PERM, pm::BCAX;
        trans_BSL: fam::PERM, pm::BSL;
        trans_BSL1N: fam::PERM, pm::BSL1N;
        trans_BSL2N: fam::PERM, pm::BSL2N;
        trans_NBSL: fam::PERM, pm::NBSL;
    }

    fn trans_XAR(&mut self, a: &mut arg_rrri_esz) -> bool {
        let ok = a.esz >= 0 && self.feat().sve2;
        self.sve_gen(ok, |s| {
            s.svi(fam::PERM, pm::XAR, a.esz, [a.rd, a.rn, a.rm, 0, 0], 0, i64::from(a.imm));
        })
    }

    // SVE Integer Arithmetic - Unary Predicated Group and reductions.

    zpz! {
        trans_CLS_m: sve, fam::ZPZ, u::CLS, any;
        trans_CLZ_m: sve, fam::ZPZ, u::CLZ, any;
        trans_CNT_zpz_m: sve, fam::ZPZ, u::CNT, any;
        trans_CNOT_m: sve, fam::ZPZ, u::CNOT, any;
        trans_NOT_zpz_m: sve, fam::ZPZ, u::NOT, any;
        trans_FABS_m: sve, fam::ZPZ, u::FABS, no_b;
        trans_FNEG_m: sve, fam::ZPZ, u::FNEG, no_b;
        trans_ABS_m: sve, fam::ZPZ, u::ABS, any;
        trans_NEG_m: sve, fam::ZPZ, u::NEG, any;
        trans_SXTB_m: sve, fam::ZPZ, u::SXTB, no_b;
        trans_UXTB_m: sve, fam::ZPZ, u::UXTB, no_b;
        trans_SXTH_m: sve, fam::ZPZ, u::SXTH, sd;
        trans_UXTH_m: sve, fam::ZPZ, u::UXTH, sd;
        trans_SXTW_m: sve, fam::ZPZ, u::SXTW, only_d;
        trans_UXTW_m: sve, fam::ZPZ, u::UXTW, only_d;
        trans_REVB_m: sve, fam::ZPZ, u::REVB, no_b;
        trans_REVH_m: sve, fam::ZPZ, u::REVH, sd;
        trans_REVW_m: sve, fam::ZPZ, u::REVW, only_d;
        trans_RBIT_m: sve, fam::ZPZ, u::RBIT, any;
        trans_URECPE_m: sve2, fam::ZPZ, u::URECPE, only_s;
        trans_URSQRTE_m: sve2, fam::ZPZ, u::URSQRTE, only_s;
        trans_SQABS_m: sve2, fam::ZPZ, u::SQABS, any;
        trans_SQNEG_m: sve2, fam::ZPZ, u::SQNEG, any;
        trans_ORV: sve, fam::RED, r::ORV, any;
        trans_EORV: sve, fam::RED, r::EORV, any;
        trans_ANDV: sve, fam::RED, r::ANDV, any;
        trans_UADDV: sve, fam::RED, r::UADDV, any;
        trans_SADDV: sve, fam::RED, r::SADDV, no_d;
        trans_SMAXV: sve, fam::RED, r::SMAXV, any;
        trans_UMAXV: sve, fam::RED, r::UMAXV, any;
        trans_SMINV: sve, fam::RED, r::SMINV, any;
        trans_UMINV: sve, fam::RED, r::UMINV, any;
        trans_MOVPRFX_z: sve, fam::PERM, pm::MOVPRFX_Z, any;
        trans_MOVPRFX_m: sve, fam::PERM, pm::MOVPRFX_M, any;
        trans_COMPACT: sve, fam::PERM, pm::COMPACT, sd;
        trans_SPLICE_sve2: sve2, fam::PERM, pm::SPLICE2, any;
    }

    // SVE Bitwise Shift - Predicated Group.

    zpzi! {
        trans_ASR_zpzi: sve, b::ASR;
        trans_LSR_zpzi: sve, b::LSR;
        trans_LSL_zpzi: sve, b::LSL;
        trans_ASRD: sve, b::ASRD;
        trans_SQSHL_zpzi: sve2, b::SQSHL;
        trans_UQSHL_zpzi: sve2, b::UQSHL;
        trans_SRSHR: sve2, b::SRSHR;
        trans_URSHR: sve2, b::URSHR;
        trans_SQSHLU: sve2, b::SQSHLU;
    }

    // SVE Integer Wide Immediate - Unpredicated Group and the unpredicated shifts.

    zzi! {
        trans_ADD_zzi: sve, b::ADD, tsz;
        trans_SUBR_zzi: sve, b::SUBR, tsz;
        trans_SMAX_zzi: sve, b::SMAX, any;
        trans_UMAX_zzi: sve, b::UMAX, any;
        trans_SMIN_zzi: sve, b::SMIN, any;
        trans_UMIN_zzi: sve, b::UMIN, any;
        trans_MUL_zzi: sve, b::MUL, tsz;
        trans_ASR_zzi: sve, b::ASR, tsz;
        trans_LSR_zzi: sve, b::LSR, tsz;
        trans_LSL_zzi: sve, b::LSL, tsz;
        trans_SSRA: sve2, b::SSRA, tsz;
        trans_USRA: sve2, b::USRA, tsz;
        trans_SRSRA: sve2, b::SRSRA, tsz;
        trans_URSRA: sve2, b::URSRA, tsz;
        trans_SRI: sve2, b::SRI, tsz;
        trans_SLI: sve2, b::SLI, tsz;
    }

    fn trans_SUB_zzi(&mut self, a: &mut arg_rri_esz) -> bool {
        a.imm = a.imm.wrapping_neg();
        self.trans_ADD_zzi(a)
    }

    zzi_sat! {
        trans_SQADD_zzi: false, false;
        trans_UQADD_zzi: true, false;
        trans_SQSUB_zzi: false, true;
        trans_UQSUB_zzi: true, true;
    }

    fn trans_AND_zzi(&mut self, a: &mut arg_rr_dbm) -> bool {
        self.zz_dbm(a, b::AND)
    }

    fn trans_ORR_zzi(&mut self, a: &mut arg_rr_dbm) -> bool {
        self.zz_dbm(a, b::ORR)
    }

    fn trans_EOR_zzi(&mut self, a: &mut arg_rr_dbm) -> bool {
        self.zz_dbm(a, b::EOR)
    }

    fn trans_DUPM(&mut self, a: &mut arg_disas_sve34) -> bool {
        let Some(imm) = wmask(a.dbm) else {
            return false;
        };
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| s.dup_const(a.rd, imm))
    }

    fn trans_DUP_i(&mut self, a: &mut arg_disas_sve47) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| s.dup_const(a.rd, rep(a.esz, a.imm as i64 as u64)))
    }

    fn trans_FDUP(&mut self, a: &mut arg_disas_sve47) -> bool {
        let ok = a.esz != 0 && self.feat().sve;
        self.sve_gen(ok, |s| {
            let v = vfp_expand_imm(a.esz as u32, a.imm as u32);
            s.dup_const(a.rd, rep(a.esz, v));
        })
    }

    // SVE Integer Compare Group.

    cmp! {
        trans_CMPEQ_ppzz, trans_CMPEQ_ppzw, trans_CMPEQ_ppzi: c::EQ;
        trans_CMPNE_ppzz, trans_CMPNE_ppzw, trans_CMPNE_ppzi: c::NE;
        trans_CMPGE_ppzz, trans_CMPGE_ppzw, trans_CMPGE_ppzi: c::GE;
        trans_CMPGT_ppzz, trans_CMPGT_ppzw, trans_CMPGT_ppzi: c::GT;
        trans_CMPHS_ppzz, trans_CMPHS_ppzw, trans_CMPHS_ppzi: c::HS;
        trans_CMPHI_ppzz, trans_CMPHI_ppzw, trans_CMPHI_ppzi: c::HI;
    }

    fn trans_CMPLT_ppzw(&mut self, a: &mut arg_rprr_esz) -> bool {
        self.cmp_w(a, c::LT)
    }

    fn trans_CMPLE_ppzw(&mut self, a: &mut arg_rprr_esz) -> bool {
        self.cmp_w(a, c::LE)
    }

    fn trans_CMPLO_ppzw(&mut self, a: &mut arg_rprr_esz) -> bool {
        self.cmp_w(a, c::LO)
    }

    fn trans_CMPLS_ppzw(&mut self, a: &mut arg_rprr_esz) -> bool {
        self.cmp_w(a, c::LS)
    }

    fn trans_CMPLT_ppzi(&mut self, a: &mut arg_rpri_esz) -> bool {
        self.cmp_i(a, c::LT)
    }

    fn trans_CMPLE_ppzi(&mut self, a: &mut arg_rpri_esz) -> bool {
        self.cmp_i(a, c::LE)
    }

    fn trans_CMPLO_ppzi(&mut self, a: &mut arg_rpri_esz) -> bool {
        self.cmp_i(a, c::LO)
    }

    fn trans_CMPLS_ppzi(&mut self, a: &mut arg_rpri_esz) -> bool {
        self.cmp_i(a, c::LS)
    }

    fn trans_MATCH(&mut self, a: &mut arg_rprr_esz) -> bool {
        let ok = self.feat().sve2 && bh(a.esz);
        self.sve_gen(ok, |s| {
            s.sv(fam::PERM, pm::MATCH, a.esz, [a.rd, a.rn, a.rm, 0, a.pg], 0, None, None);
        })
    }

    fn trans_NMATCH(&mut self, a: &mut arg_rprr_esz) -> bool {
        let ok = self.feat().sve2 && bh(a.esz);
        self.sve_gen(ok, |s| {
            s.sv(fam::PERM, pm::NMATCH, a.esz, [a.rd, a.rn, a.rm, 0, a.pg], 0, None, None);
        })
    }

    fn trans_HISTCNT(&mut self, a: &mut arg_rprr_esz) -> bool {
        let ok = self.feat().sve2 && sd(a.esz);
        self.sve_gen(ok, |s| {
            s.sv(fam::PERM, pm::HISTCNT, a.esz, [a.rd, a.rn, a.rm, 0, a.pg], 0, None, None);
        })
    }

    // SVE Predicate Logical Operations Group and the breaks.

    pppp! {
        trans_AND_pppp: pl::AND;
        trans_BIC_pppp: pl::BIC;
        trans_EOR_pppp: pl::EOR;
        trans_ORR_pppp: pl::ORR;
        trans_ORN_pppp: pl::ORN;
        trans_NOR_pppp: pl::NOR;
        trans_NAND_pppp: pl::NAND;
    }

    fn trans_SEL_pppp(&mut self, a: &mut arg_rprr_s) -> bool {
        let ok = a.s == 0 && self.feat().sve;
        self.sve_gen(ok, |s| {
            s.sv(fam::PPPP, pl::SEL, 0, [a.rd, a.rn, a.rm, 0, a.pg], 0, None, None);
        })
    }

    fn trans_BRKPA(&mut self, a: &mut arg_rprr_s) -> bool {
        self.brkp(a, 0)
    }

    fn trans_BRKPB(&mut self, a: &mut arg_rprr_s) -> bool {
        self.brkp(a, 4)
    }

    brk! {
        trans_BRKA_z: 0;
        trans_BRKB_z: 4;
        trans_BRKA_m: 2;
        trans_BRKB_m: 6;
    }

    fn trans_BRKN(&mut self, a: &mut arg_rpr_s) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            let data = a.s as u32;
            s.sv(fam::PRED, pr::BRKN, 0, [a.rd, a.rn, 0, 0, a.pg], data, None, None);
        })
    }

    // SVE Predicate Misc Group.

    fn trans_PTEST(&mut self, a: &mut arg_disas_sve37) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.sv(fam::PRED, pr::PTEST, 0, [0, a.rn, 0, 0, a.pg], 0, None, None);
        })
    }

    fn trans_PTRUE(&mut self, a: &mut arg_ptrue) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            let data = a.s as u32;
            s.svi(fam::PRED, pr::PTRUE, a.esz, [a.rd, 0, 0, 0, 0], data, i64::from(a.pat));
        })
    }

    fn trans_SETFFR(&mut self, _a: &mut arg_disas_sve35) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.sv(fam::PRED, pr::SETFFR, 0, [0; 5], 0, None, None);
        })
    }

    fn trans_PFALSE(&mut self, a: &mut arg_disas_sve39) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.sv(fam::PRED, pr::PFALSE, 0, [a.rd, 0, 0, 0, 0], 0, None, None);
        })
    }

    fn trans_RDFFR(&mut self, a: &mut arg_disas_sve39) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.sv(fam::PRED, pr::RDFFR, 0, [a.rd, 0, 0, 0, 0], 0, None, None);
        })
    }

    fn trans_RDFFR_p(&mut self, a: &mut arg_disas_sve40) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            let data = 2 | a.s as u32;
            s.sv(fam::PRED, pr::RDFFR, 0, [a.rd, 0, 0, 0, a.pg], data, None, None);
        })
    }

    fn trans_WRFFR(&mut self, a: &mut arg_disas_sve41) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.sv(fam::PRED, pr::WRFFR, 0, [0, a.rn, 0, 0, 0], 0, None, None);
        })
    }

    fn trans_PFIRST(&mut self, a: &mut arg_rr_esz) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.sv(fam::PRED, pr::PFIRST, a.esz, [a.rd, 0, 0, 0, a.rn], 1, None, None);
        })
    }

    fn trans_PNEXT(&mut self, a: &mut arg_rr_esz) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.sv(fam::PRED, pr::PNEXT, a.esz, [a.rd, 0, 0, 0, a.rn], 1, None, None);
        })
    }

    // SVE Element Count Group.

    fn trans_CNT_r(&mut self, a: &mut arg_incdec_cnt) -> bool {
        let n = self.numelem(a.pat, a.esz) * i64::from(a.imm);
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            let t = s.c64(n);
            s.set_reg(a.rd, t);
        })
    }

    fn trans_INCDEC_r(&mut self, a: &mut arg_incdec_cnt) -> bool {
        let inc = self.numelem(a.pat, a.esz) * i64::from(a.imm);
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            let r = s.reg(a.rd);
            s.f().gen_addi_i64(r, r, if a.d != 0 { -inc } else { inc });
            s.set_reg(a.rd, r);
        })
    }

    fn trans_SINCDEC_r_32(&mut self, a: &mut arg_incdec_cnt) -> bool {
        let inc = self.numelem(a.pat, a.esz) * i64::from(a.imm);
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            if inc == 0 {
                let r = s.reg(a.rd);
                if a.u != 0 {
                    s.f().gen_ext32u_i64(r, r);
                } else {
                    s.f().gen_ext32s_i64(r, r);
                }
                s.set_reg(a.rd, r);
            } else {
                let v = s.c64(inc);
                s.sat_reg(a.rd, v, a.u != 0, a.d != 0, false);
            }
        })
    }

    fn trans_SINCDEC_r_64(&mut self, a: &mut arg_incdec_cnt) -> bool {
        let inc = self.numelem(a.pat, a.esz) * i64::from(a.imm);
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            if inc != 0 {
                let v = s.c64(inc);
                s.sat_reg(a.rd, v, a.u != 0, a.d != 0, true);
            }
        })
    }

    fn trans_INCDEC_v(&mut self, a: &mut arg_incdec2_cnt) -> bool {
        let inc = self.numelem(a.pat, a.esz) * i64::from(a.imm);
        let ok = a.esz != 0 && self.feat().sve;
        self.sve_gen(ok, |s| {
            if inc != 0 {
                let v = s.c64(inc);
                s.add_vec(a.esz, a.rd, a.rn, v, a.d != 0);
            } else {
                s.sv(fam::PERM, pm::MOVPRFX, 0, [a.rd, a.rn, 0, 0, 0], 0, None, None);
            }
        })
    }

    fn trans_SINCDEC_v(&mut self, a: &mut arg_incdec2_cnt) -> bool {
        let inc = self.numelem(a.pat, a.esz) * i64::from(a.imm);
        let ok = a.esz != 0 && self.feat().sve;
        self.sve_gen(ok, |s| {
            if inc != 0 {
                let v = s.c64(inc);
                s.sat_vec(a.esz, a.rd, a.rn, v, a.u != 0, a.d != 0);
            } else {
                s.sv(fam::PERM, pm::MOVPRFX, 0, [a.rd, a.rn, 0, 0, 0], 0, None, None);
            }
        })
    }

    fn trans_CNTP(&mut self, a: &mut arg_rpr_esz) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            let t = s.cntp(a.esz, a.rn, a.pg);
            s.set_reg(a.rd, t);
        })
    }

    fn trans_INCDECP_r(&mut self, a: &mut arg_incdec_pred) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            let v = s.cntp(a.esz, a.pg, a.pg);
            let r = s.reg(a.rd);
            if a.d != 0 {
                s.f().gen_sub_i64(r, r, v);
            } else {
                s.f().gen_add_i64(r, r, v);
            }
            s.set_reg(a.rd, r);
        })
    }

    fn trans_INCDECP_z(&mut self, a: &mut arg_incdec2_pred) -> bool {
        let ok = a.esz != 0 && self.feat().sve;
        self.sve_gen(ok, |s| {
            let v = s.cntp(a.esz, a.pg, a.pg);
            s.add_vec(a.esz, a.rd, a.rn, v, a.d != 0);
        })
    }

    fn trans_SINCDECP_r_32(&mut self, a: &mut arg_incdec_pred) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            let v = s.cntp(a.esz, a.pg, a.pg);
            s.sat_reg(a.rd, v, a.u != 0, a.d != 0, false);
        })
    }

    fn trans_SINCDECP_r_64(&mut self, a: &mut arg_incdec_pred) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            let v = s.cntp(a.esz, a.pg, a.pg);
            s.sat_reg(a.rd, v, a.u != 0, a.d != 0, true);
        })
    }

    fn trans_SINCDECP_z(&mut self, a: &mut arg_incdec2_pred) -> bool {
        let ok = a.esz != 0 && self.feat().sve;
        self.sve_gen(ok, |s| {
            let v = s.cntp(a.esz, a.pg, a.pg);
            s.sat_vec(a.esz, a.rd, a.rn, v, a.u != 0, a.d != 0);
        })
    }

    // SVE Stack Allocation Group.

    fn trans_ADDVL(&mut self, a: &mut arg_rri) -> bool {
        let ok = self.feat().sve;
        let v = i64::from(a.imm) * i64::from(self.d.vl);
        self.sve_gen(ok, |s| {
            let t = s.reg_sp(a.rn);
            s.f().gen_addi_i64(t, t, v);
            s.set_reg_sp(a.rd, t);
        })
    }

    fn trans_ADDPL(&mut self, a: &mut arg_rri) -> bool {
        let ok = self.feat().sve;
        let v = i64::from(a.imm) * i64::from(self.d.vl / 8);
        self.sve_gen(ok, |s| {
            let t = s.reg_sp(a.rn);
            s.f().gen_addi_i64(t, t, v);
            s.set_reg_sp(a.rd, t);
        })
    }

    fn trans_RDVL(&mut self, a: &mut arg_disas_sve32) -> bool {
        let ok = self.feat().sve;
        let v = i64::from(a.imm) * i64::from(self.d.vl);
        self.sve_gen(ok, |s| {
            let t = s.c64(v);
            s.set_reg(a.rd, t);
        })
    }

    // SVE Index Generation Group.

    fn trans_INDEX_ii(&mut self, a: &mut arg_disas_sve30) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            let x = s.c64(i64::from(a.imm1));
            let y = s.c64(i64::from(a.imm2));
            s.sv(fam::PERM, pm::INDEX, a.esz, [a.rd, 0, 0, 0, 0], 0, Some(x), Some(y));
        })
    }

    fn trans_INDEX_ir(&mut self, a: &mut arg_disas_sve31) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            let x = s.c64(i64::from(a.imm));
            let y = s.reg(a.rm);
            s.sv(fam::PERM, pm::INDEX, a.esz, [a.rd, 0, 0, 0, 0], 0, Some(x), Some(y));
        })
    }

    fn trans_INDEX_ri(&mut self, a: &mut arg_rri_esz) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            let x = s.reg(a.rn);
            let y = s.c64(i64::from(a.imm));
            s.sv(fam::PERM, pm::INDEX, a.esz, [a.rd, 0, 0, 0, 0], 0, Some(x), Some(y));
        })
    }

    fn trans_INDEX_rr(&mut self, a: &mut arg_rrr_esz) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            let x = s.reg(a.rn);
            let y = s.reg(a.rm);
            s.sv(fam::PERM, pm::INDEX, a.esz, [a.rd, 0, 0, 0, 0], 0, Some(x), Some(y));
        })
    }

    // SVE Permute - Unpredicated, Predicates and Predicated Groups.

    fn trans_DUP_s(&mut self, a: &mut arg_rr_esz) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            let x = s.reg_sp(a.rn);
            s.sv(fam::PERM, pm::DUP, a.esz, [a.rd, 0, 0, 0, 0], 0, Some(x), None);
        })
    }

    fn trans_DUP_x(&mut self, a: &mut arg_rri) -> bool {
        if a.imm & 0x1f == 0 || !self.feat().sve {
            return false;
        }
        self.sve_gen(true, |s| {
            let esz = a.imm.trailing_zeros() as i32;
            let index = (a.imm as u32 >> (esz + 1)) as usize;
            let vl = s.d.vl as usize;
            if esz == 4 {
                if index * 16 < vl {
                    let off = crate::cpu::vreg_off(a.rn as usize) + index * 16;
                    let lo = s.ld_env64(off);
                    let hi = s.ld_env64(off + 8);
                    s.dup_t(a.rd, lo, Some(hi));
                } else {
                    s.dup_const(a.rd, 0);
                }
            } else {
                let x = s.c64(index as i64);
                s.sv(fam::PERM, pm::DUPX, esz, [a.rd, a.rn, 0, 0, 0], 0, Some(x), None);
            }
        })
    }

    fn trans_INSR_f(&mut self, a: &mut arg_rrr_esz) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.sv(fam::PERM, pm::INSR, a.esz, [a.rd, a.rn, a.rm, 0, 0], 1, None, None);
        })
    }

    fn trans_INSR_r(&mut self, a: &mut arg_rrr_esz) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            let x = s.reg(a.rm);
            s.sv(fam::PERM, pm::INSR, a.esz, [a.rd, a.rn, 0, 0, 0], 0, Some(x), None);
        })
    }

    fn trans_REV_v(&mut self, a: &mut arg_rr_esz) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.sv(fam::PERM, pm::REV, a.esz, [a.rd, a.rn, 0, 0, 0], 0, None, None);
        })
    }

    perm3! {
        trans_TBL: sve, fam::PERM, pm::TBL, any;
        trans_TBL_sve2: sve2, fam::PERM, pm::TBL2, any;
        trans_TBX: sve2, fam::PERM, pm::TBX, any;
        trans_ZIP1_z: sve, fam::PERM, pm::ZIP1, any;
        trans_ZIP2_z: sve, fam::PERM, pm::ZIP2, any;
        trans_UZP1_z: sve, fam::PERM, pm::UZP1, any;
        trans_UZP2_z: sve, fam::PERM, pm::UZP2, any;
        trans_TRN1_z: sve, fam::PERM, pm::TRN1, any;
        trans_TRN2_z: sve, fam::PERM, pm::TRN2, any;
        trans_ZIP1_p: sve, fam::PRED, pr::ZIP1, any;
        trans_ZIP2_p: sve, fam::PRED, pr::ZIP2, any;
        trans_UZP1_p: sve, fam::PRED, pr::UZP1, any;
        trans_UZP2_p: sve, fam::PRED, pr::UZP2, any;
        trans_TRN1_p: sve, fam::PRED, pr::TRN1, any;
        trans_TRN2_p: sve, fam::PRED, pr::TRN2, any;
        trans_HISTSEG: sve2, fam::PERM, pm::HISTSEG, |e| e == 0;
    }

    fn trans_REV_p(&mut self, a: &mut arg_rr_esz) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.sv(fam::PRED, pr::REV, a.esz, [a.rd, a.rn, 0, 0, 0], 0, None, None);
        })
    }

    fn trans_PUNPKLO(&mut self, a: &mut arg_rr_esz) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.sv(fam::PRED, pr::PUNPK, 0, [a.rd, a.rn, 0, 0, 0], 0, None, None);
        })
    }

    fn trans_PUNPKHI(&mut self, a: &mut arg_rr_esz) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.sv(fam::PRED, pr::PUNPK, 0, [a.rd, a.rn, 0, 0, 0], 1, None, None);
        })
    }

    fn trans_UNPK(&mut self, a: &mut arg_disas_sve36) -> bool {
        let ok = a.esz != 0 && self.feat().sve;
        self.sve_gen(ok, |s| {
            let data = a.h as u32 | (a.u as u32) << 1;
            s.sv(fam::PERM, pm::UNPK, a.esz, [a.rd, a.rn, 0, 0, 0], data, None, None);
        })
    }

    fn trans_EXT(&mut self, a: &mut arg_rrri) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.svi(fam::PERM, pm::EXT, 0, [a.rd, a.rn, a.rm, 0, 0], 0, i64::from(a.imm));
        })
    }

    fn trans_EXT_sve2(&mut self, a: &mut arg_rri) -> bool {
        let ok = self.feat().sve2;
        self.sve_gen(ok, |s| {
            s.svi(fam::PERM, pm::EXT2, 0, [a.rd, a.rn, 0, 0, 0], 0, i64::from(a.imm));
        })
    }

    fn trans_SPLICE(&mut self, a: &mut arg_rprr_esz) -> bool {
        self.zpzz_perm(a, pm::SPLICE)
    }

    fn trans_SEL_zpzz(&mut self, a: &mut arg_rprr_esz) -> bool {
        self.zpzz_perm(a, pm::SEL)
    }

    fn trans_CLASTA_z(&mut self, a: &mut arg_rprr_esz) -> bool {
        self.zpzz_perm(a, pm::CLASTA_Z)
    }

    fn trans_CLASTB_z(&mut self, a: &mut arg_rprr_esz) -> bool {
        self.zpzz_perm(a, pm::CLASTB_Z)
    }

    last! {
        trans_CLASTA_r: pm::CLAST, 0;
        trans_CLASTB_r: pm::CLAST, 2;
        trans_CLASTA_v: pm::CLAST, 1;
        trans_CLASTB_v: pm::CLAST, 3;
        trans_LASTA_r: pm::LAST, 0;
        trans_LASTB_r: pm::LAST, 2;
        trans_LASTA_v: pm::LAST, 1;
        trans_LASTB_v: pm::LAST, 3;
    }

    fn trans_CPY_m_v(&mut self, a: &mut arg_rpr_esz) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.sv(fam::PERM, pm::CPY_M, a.esz, [a.rd, a.rn, 0, 0, a.pg], 1, None, None);
        })
    }

    fn trans_CPY_m_r(&mut self, a: &mut arg_rpr_esz) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            let x = s.reg_sp(a.rn);
            s.sv(fam::PERM, pm::CPY_M, a.esz, [a.rd, 0, 0, 0, a.pg], 0, Some(x), None);
        })
    }

    fn trans_CPY_m_i(&mut self, a: &mut arg_rpri_esz) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.svi(fam::PERM, pm::CPY_M, a.esz, [a.rd, 0, 0, 0, a.pg], 0, i64::from(a.imm));
        })
    }

    fn trans_CPY_z_i(&mut self, a: &mut arg_rpri_esz) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.svi(fam::PERM, pm::CPY_Z, a.esz, [a.rd, 0, 0, 0, a.pg], 0, i64::from(a.imm));
        })
    }

    fn trans_FCPY(&mut self, a: &mut arg_rpri_esz) -> bool {
        let ok = a.esz != 0 && self.feat().sve;
        self.sve_gen(ok, |s| {
            let v = vfp_expand_imm(a.esz as u32, a.imm as u32) as i64;
            s.svi(fam::PERM, pm::CPY_M, a.esz, [a.rd, 0, 0, 0, a.pg], 0, v);
        })
    }

    fn trans_MOVPRFX(&mut self, a: &mut arg_disas_sve33) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.sv(fam::PERM, pm::MOVPRFX, 0, [a.rd, a.rn, 0, 0, 0], 0, None, None);
        })
    }

    // SVE Compute Vector Address Group.

    fn trans_ADR_s32(&mut self, a: &mut arg_rrri) -> bool {
        self.adr(a, 3, 0)
    }

    fn trans_ADR_u32(&mut self, a: &mut arg_rrri) -> bool {
        self.adr(a, 3, 1)
    }

    fn trans_ADR_p32(&mut self, a: &mut arg_rrri) -> bool {
        self.adr(a, 2, 2)
    }

    fn trans_ADR_p64(&mut self, a: &mut arg_rrri) -> bool {
        self.adr(a, 3, 2)
    }

    // WHILE and CTERM.

    fn trans_WHILE_lt(&mut self, a: &mut arg_while) -> bool {
        let ok = self.feat().sve;
        self.do_while(ok, a, true)
    }

    fn trans_WHILE_gt(&mut self, a: &mut arg_while) -> bool {
        let ok = self.feat().sve2;
        self.do_while(ok, a, false)
    }

    fn trans_WHILE_ptr(&mut self, a: &mut arg_disas_sve45) -> bool {
        let ok = self.feat().sve2;
        self.sve_gen(ok, |s| {
            let x = s.read_cpu_reg(a.rn, true);
            let y = s.read_cpu_reg(a.rm, true);
            let data = 16 | (a.rw as u32) << 5;
            s.sv(fam::PRED, pr::WHILE, a.esz, [a.rd, 0, 0, 0, 0], data, Some(x), Some(y));
        })
    }

    fn trans_CTERM(&mut self, a: &mut arg_disas_sve43) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            let sf = a.sf != 0;
            let x = s.read_cpu_reg(a.rn, sf);
            let y = s.read_cpu_reg(a.rm, sf);
            let data = a.ne as u32 | u32::from(sf) << 1;
            s.sv(fam::PRED, pr::CTERM, 0, [0; 5], data, Some(x), Some(y));
        })
    }

    // SVE Memory - 32-bit Gather and Unsized Contiguous Group, and the others.

    fn trans_LDR_zri(&mut self, a: &mut arg_rri) -> bool {
        let vl = i64::from(self.d.vl);
        self.ldr_str(a, mm::LDR, 0, vl)
    }

    fn trans_LDR_pri(&mut self, a: &mut arg_rri) -> bool {
        let pl = i64::from(self.d.vl / 8);
        self.ldr_str(a, mm::LDR, 8, pl)
    }

    fn trans_STR_zri(&mut self, a: &mut arg_rri) -> bool {
        let vl = i64::from(self.d.vl);
        self.ldr_str(a, mm::STR, 0, vl)
    }

    fn trans_STR_pri(&mut self, a: &mut arg_rri) -> bool {
        let pl = i64::from(self.d.vl / 8);
        self.ldr_str(a, mm::STR, 8, pl)
    }

    fn trans_LD_zprr(&mut self, a: &mut arg_rprr_load) -> bool {
        let ok = a.rm != 31 && a.dtype < 16 && self.feat().sve;
        self.sve_gen(ok, |s| {
            let addr = s.cont_addr(a.rn, Some(a.rm), dtype_mop(a.dtype).0, 0);
            s.ld_cont(a.rd, a.pg, addr, a.dtype, a.nreg, 0);
        })
    }

    fn trans_LD_zpri(&mut self, a: &mut arg_rpri_load) -> bool {
        let ok = a.dtype < 16 && self.feat().sve;
        let elements = i64::from(self.d.vl >> dtype_esz(a.dtype));
        self.sve_gen(ok, |s| {
            let (msz, _) = dtype_mop(a.dtype);
            let off = (i64::from(a.imm) * elements * i64::from(a.nreg + 1)) << msz;
            let addr = s.cont_addr(a.rn, None, msz, off);
            s.ld_cont(a.rd, a.pg, addr, a.dtype, a.nreg, 0);
        })
    }

    fn trans_LDFF1_zprr(&mut self, a: &mut arg_rprr_load) -> bool {
        let ok = a.dtype < 16 && self.feat().sve;
        self.sve_gen(ok, |s| {
            let addr = s.cont_addr(a.rn, Some(a.rm), dtype_mop(a.dtype).0, 0);
            s.ld_cont(a.rd, a.pg, addr, a.dtype, 0, mm::FF);
        })
    }

    fn trans_LDNF1_zpri(&mut self, a: &mut arg_rpri_load) -> bool {
        let ok = a.dtype < 16 && self.feat().sve;
        let elements = i64::from(self.d.vl >> dtype_esz(a.dtype));
        self.sve_gen(ok, |s| {
            let (msz, _) = dtype_mop(a.dtype);
            let off = (i64::from(a.imm) * elements) << msz;
            let addr = s.cont_addr(a.rn, None, msz, off);
            s.ld_cont(a.rd, a.pg, addr, a.dtype, 0, mm::NF);
        })
    }

    fn trans_LD1RQ_zprr(&mut self, a: &mut arg_rprr_load) -> bool {
        let ok = a.rm != 31 && self.feat().sve;
        self.sve_gen(ok, |s| {
            let (msz, sign) = dtype_mop(a.dtype);
            let addr = s.cont_addr(a.rn, Some(a.rm), msz, 0);
            let data = msz | u32::from(sign) << 2;
            s.svm(mm::LD1RQ, dtype_esz(a.dtype), [a.rd, 0, 0, 0, a.pg], data, addr);
        })
    }

    fn trans_LD1RQ_zpri(&mut self, a: &mut arg_rpri_load) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            let (msz, sign) = dtype_mop(a.dtype);
            let addr = s.cont_addr(a.rn, None, msz, i64::from(a.imm) * 16);
            let data = msz | u32::from(sign) << 2;
            s.svm(mm::LD1RQ, dtype_esz(a.dtype), [a.rd, 0, 0, 0, a.pg], data, addr);
        })
    }

    fn trans_LD1R_zpri(&mut self, a: &mut arg_rpri_load) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            let (msz, sign) = dtype_mop(a.dtype);
            let addr = s.cont_addr(a.rn, None, msz, i64::from(a.imm) << msz);
            let data = msz | u32::from(sign) << 2;
            s.svm(mm::LD1R, dtype_esz(a.dtype), [a.rd, 0, 0, 0, a.pg], data, addr);
        })
    }

    fn trans_ST_zprr(&mut self, a: &mut arg_rprr_store) -> bool {
        let ok = a.rm != 31 && a.msz <= a.esz && a.esz <= 3 && self.feat().sve;
        self.sve_gen(ok, |s| {
            let addr = s.cont_addr(a.rn, Some(a.rm), a.msz as u32, 0);
            let data = a.msz as u32 | (a.nreg as u32) << 3;
            s.svm(mm::ST, a.esz, [a.rd, 0, 0, 0, a.pg], data, addr);
        })
    }

    fn trans_ST_zpri(&mut self, a: &mut arg_rpri_store) -> bool {
        let ok = a.msz <= a.esz && a.esz <= 3 && self.feat().sve;
        let elements = i64::from(self.d.vl) >> a.esz.clamp(0, 3);
        self.sve_gen(ok, |s| {
            let off = (i64::from(a.imm) * elements * i64::from(a.nreg + 1)) << a.msz;
            let addr = s.cont_addr(a.rn, None, a.msz as u32, off);
            let data = a.msz as u32 | (a.nreg as u32) << 3;
            s.svm(mm::ST, a.esz, [a.rd, 0, 0, 0, a.pg], data, addr);
        })
    }

    fn trans_LD1_zprz(&mut self, a: &mut arg_rprr_gather_load) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            let x = s.reg_sp(a.rn);
            let data = gather_data(a.msz, a.u == 0, a.ff != 0, a.xs as u32, a.scale != 0);
            s.gather(mm::GATHER, a.esz, [a.rd, 0, a.rm, 0, a.pg], data, x);
        })
    }

    fn trans_LD1_zpiz(&mut self, a: &mut arg_rpri_gather_load) -> bool {
        let ok = !(a.esz < a.msz || (a.esz == a.msz && a.u == 0)) && self.feat().sve;
        self.sve_gen(ok, |s| {
            let x = s.c64(i64::from(a.imm) << a.msz);
            let data = gather_data(a.msz, a.u == 0, a.ff != 0, mm::OFF_VEC, false);
            s.gather(mm::GATHER, a.esz, [a.rd, 0, a.rn, 0, a.pg], data, x);
        })
    }

    fn trans_LDNT1_zprz(&mut self, a: &mut arg_rprr_gather_load) -> bool {
        let ok = a.esz >= a.msz + i32::from(a.u == 0) && self.feat().sve2;
        self.sve_gen(ok, |s| {
            let x = s.reg(a.rm);
            let data = gather_data(a.msz, a.u == 0, false, mm::OFF_VEC, false);
            s.gather(mm::GATHER, a.esz, [a.rd, 0, a.rn, 0, a.pg], data, x);
        })
    }

    fn trans_ST1_zprz(&mut self, a: &mut arg_rprr_scatter_store) -> bool {
        let ok = !(a.esz < a.msz || (a.msz == 0 && a.scale != 0)) && self.feat().sve;
        self.sve_gen(ok, |s| {
            let x = s.reg_sp(a.rn);
            let data = gather_data(a.msz, false, false, a.xs as u32, a.scale != 0);
            s.gather(mm::SCATTER, a.esz, [a.rd, 0, a.rm, 0, a.pg], data, x);
        })
    }

    fn trans_ST1_zpiz(&mut self, a: &mut arg_rpri_scatter_store) -> bool {
        let ok = a.esz >= a.msz && self.feat().sve;
        self.sve_gen(ok, |s| {
            let x = s.c64(i64::from(a.imm) << a.msz);
            let data = gather_data(a.msz, false, false, mm::OFF_VEC, false);
            s.gather(mm::SCATTER, a.esz, [a.rd, 0, a.rn, 0, a.pg], data, x);
        })
    }

    fn trans_STNT1_zprz(&mut self, a: &mut arg_rprr_scatter_store) -> bool {
        let ok = a.esz >= a.msz && self.feat().sve2;
        self.sve_gen(ok, |s| {
            let x = s.reg(a.rm);
            let data = gather_data(a.msz, false, false, mm::OFF_VEC, false);
            s.gather(mm::SCATTER, a.esz, [a.rd, 0, a.rn, 0, a.pg], data, x);
        })
    }

    fn trans_PRF(&mut self, _a: &mut arg_disas_sve35) -> bool {
        // Prefetch is a nop within QEMU.
        self.sve_gen(self.feat().sve, |_| {})
    }

    fn trans_PRF_ns(&mut self, _a: &mut arg_disas_sve35) -> bool {
        self.sve_gen(self.feat().sve, |_| {})
    }

    fn trans_PRF_rr(&mut self, a: &mut arg_disas_sve54) -> bool {
        let ok = a.rm != 31 && self.feat().sve;
        self.sve_gen(ok, |_| {})
    }
}

/// The data of a gather or scatter descriptor.
fn gather_data(msz: i32, sign: bool, ff: bool, kind: u32, scale: bool) -> u32 {
    msz as u32 | u32::from(sign) << 2 | u32::from(ff) << 3 | kind << 4 | u32::from(scale) << 6
}

/// `logic_imm_decode_wmask()` on the 13-bit `dbm` field.
fn wmask(dbm: i32) -> Option<u64> {
    let dbm = dbm as u32;
    super::logic_imm_decode_wmask((dbm >> 12) & 1, dbm & 0x3f, (dbm >> 6) & 0x3f)
}

impl S<'_, '_> {
    fn zz_dbm(&mut self, a: &arg_rr_dbm, op: u32) -> bool {
        let Some(imm) = wmask(a.dbm) else {
            return false;
        };
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.svi(fam::ZZI, op, 3, [a.rd, a.rn, 0, 0, 0], 0, imm as i64);
        })
    }

    fn cmp_w(&mut self, a: &arg_rprr_esz, op: u32) -> bool {
        let ok = self.feat().sve && a.esz != 3;
        self.sve_gen(ok, |s| {
            s.sv(fam::CMP, op, a.esz, [a.rd, a.rn, a.rm, 0, a.pg], 1, None, None);
        })
    }

    fn cmp_i(&mut self, a: &arg_rpri_esz, op: u32) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.svi(fam::CMP, op, a.esz, [a.rd, a.rn, 0, 0, a.pg], 2, i64::from(a.imm));
        })
    }

    fn brkp(&mut self, a: &arg_rprr_s, b_form: u32) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            let data = a.s as u32 | b_form;
            s.sv(fam::PRED, pr::BRKP, 0, [a.rd, a.rn, a.rm, 0, a.pg], data, None, None);
        })
    }

    fn zpzz_perm(&mut self, a: &arg_rprr_esz, op: u32) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.sv(fam::PERM, op, a.esz, [a.rd, a.rn, a.rm, 0, a.pg], 0, None, None);
        })
    }

    fn adr(&mut self, a: &arg_rrri, esz: i32, data: u32) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            s.svi(fam::PERM, pm::ADR, esz, [a.rd, a.rn, a.rm, 0, 0], data, i64::from(a.imm));
        })
    }

    /// `do_WHILE()` for WHILELT, WHILELE, WHILELO, WHILELS and the greater-than forms.
    fn do_while(&mut self, ok: bool, a: &arg_while, lt: bool) -> bool {
        self.sve_gen(ok, |s| {
            let x = s.read_cpu_reg(a.rn, true);
            let y = s.read_cpu_reg(a.rm, true);
            // GE and HS have eq clear and GT and HI have it set.
            let eq = (a.eq != 0) == lt;
            let data = u32::from(!lt) | (a.u as u32) << 1 | u32::from(eq) << 2 | (a.sf as u32) << 3;
            s.sv(fam::PRED, pr::WHILE, a.esz, [a.rd, 0, 0, 0, 0], data, Some(x), Some(y));
        })
    }

    /// LDR and STR of a vector (`data` 0) or predicate (`data` 8) register.
    fn ldr_str(&mut self, a: &arg_rri, op: u32, data: u32, size: i64) -> bool {
        let ok = self.feat().sve;
        self.sve_gen(ok, |s| {
            let addr = s.cont_addr(a.rn, None, 0, i64::from(a.imm) * size);
            s.svm(op, 0, [a.rd, 0, 0, 0, 0], data, addr);
        })
    }
}
