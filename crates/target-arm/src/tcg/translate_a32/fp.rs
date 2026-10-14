// SPDX-License-Identifier: GPL-2.0-or-later

//! The VFP instructions: QEMU's `translate-vfp.c`, with the decoders of vfp.decode and
//! vfp-uncond.decode.
//!
//! The FP enable checks of `vfp_access_check()` always pass: only the user mode emulator
//! runs AArch32 code, with FP enabled. There are no short vectors: FPSCR.Len and Stride are
//! taken as zero, which a CPU without MVFR0.FPShVec has to keep them at. The arithmetic goes
//! through the one `a32_vfp` helper, with the values in 64-bit temporaries whatever their
//! size.
//!
//! Not here yet, and so UNDEF: VCVT to BFloat16 and VJCVT.

use ruvm_jit_core::MemOp;
use ruvm_jit_core::ir::{TempI32, TempI64};
use ruvm_jit_core::types::Cond;
use ruvm_softfloat::RoundMode;

use super::{S, h};
use crate::cpu::{CPSR_NZCV, vreg_off};
use crate::tcg::a32_vfp::{self, op};
use crate::tcg::vec_helper::RMODE_FPCR;
use crate::tcg::vfp::vfp_expand_imm;

#[allow(missing_docs, unreachable_pub, clippy::pedantic, clippy::nursery)]
mod dec {
    include!(concat!(env!("OUT_DIR"), "/vfp_decode.rs"));
}

#[allow(missing_docs, unreachable_pub, clippy::pedantic, clippy::nursery)]
mod uncond {
    include!(concat!(env!("OUT_DIR"), "/vfp_uncond_decode.rs"));
}

use dec::*;
use uncond::*;

/// `disas_vfp()`.
pub(super) fn disas(s: &mut S<'_, '_>, insn: u32) -> bool {
    disas_vfp(s, insn)
}

/// `disas_vfp_uncond()`.
pub(super) fn disas_uncond(s: &mut S<'_, '_>, insn: u32) -> bool {
    disas_vfp_uncond(s, insn)
}

/// The rounding mode a softfloat status takes.
const fn rm(m: RoundMode) -> u32 {
    m as u32
}

/// `vfp_reg_offset()`: the `env` offset of D`reg`, or of S`reg` if not `dp`. D`n` is half of
/// the AdvSIMD register Q`n/2`, and S`n` half of D`n/2`.
pub(super) fn vfp_reg_offset(dp: bool, reg: i32) -> usize {
    let reg = reg as usize;
    if dp {
        vreg_off(reg >> 1) + 8 * (reg & 1)
    } else {
        vreg_off(reg >> 2) + 8 * ((reg >> 1) & 1) + 4 * (reg & 1)
    }
}

impl S<'_, '_> {
    /// `gen_aa32_ld_i64()`.
    fn ld64(&mut self, val: TempI64, addr: TempI32, op: MemOp) {
        let op = self.finalize_memop(op);
        let idx = self.d.mmu_idx;
        let a = self.aa32_addr(addr);
        self.f().gen_qemu_ld_i64(val, a, idx, op);
    }

    /// `gen_aa32_st_i64()`.
    fn st64(&mut self, val: TempI64, addr: TempI32, op: MemOp) {
        let op = self.finalize_memop(op);
        let idx = self.d.mmu_idx;
        let a = self.aa32_addr(addr);
        self.f().gen_qemu_st_i64(val, a, idx, op);
    }

    /// `vfp_load_reg32()`.
    fn vfp_load_reg32(&mut self, reg: i32) -> TempI32 {
        self.ld_env32(vfp_reg_offset(false, reg))
    }

    /// `vfp_store_reg32()`.
    fn vfp_store_reg32(&mut self, t: TempI32, reg: i32) {
        self.st_env32(t, vfp_reg_offset(false, reg));
    }

    /// `gen_set_nzcv()`: NZCV from bits 31 to 28 of `t`.
    fn set_nzcv(&mut self, t: TempI32) {
        self.set_cpsr(t, CPSR_NZCV);
    }

    /// `trans_VLDM_VSTM_sp()` and `trans_VLDM_VSTM_dp()`, with `n` registers of `size`
    /// bytes.
    fn vldm_vstm(&mut self, a: &arg_VLDM_VSTM_dp, dp: bool, n: i32) -> bool {
        // For thumb, use of PC is UNPREDICTABLE.
        let addr = self.add_reg_for_lit(a.rn, 0);
        if a.p != 0 {
            // pre-decrement
            self.f().gen_addi_i32(addr, addr, -(a.imm << 2));
        }
        let size = if dp { 8 } else { 4 };
        for i in 0..n {
            let off = vfp_reg_offset(dp, a.vd + i);
            if dp {
                let mop = MemOp::UQ | MemOp::ALIGN_4;
                if a.l != 0 {
                    let t = self.new64();
                    self.ld64(t, addr, mop);
                    self.st_env64(t, off);
                } else {
                    let t = self.ld_env64(off);
                    self.st64(t, addr, mop);
                }
            } else {
                let idx = self.d.mmu_idx;
                if a.l != 0 {
                    let t = self.new32();
                    self.ld32(t, addr, idx, MemOp::UL | MemOp::ALIGN);
                    self.st_env32(t, off);
                } else {
                    let t = self.ld_env32(off);
                    self.st32(t, addr, idx, MemOp::UL | MemOp::ALIGN);
                }
            }
            self.f().gen_addi_i32(addr, addr, size);
        }
        if a.w != 0 {
            // writeback
            let offset = if a.p != 0 {
                -size * n
            } else if dp && a.imm & 1 != 0 {
                // FLDMX and FSTMX
                4
            } else {
                0
            };
            if offset != 0 {
                self.f().gen_addi_i32(addr, addr, offset);
            }
            self.store_reg(a.rn, addr);
        }
        true
    }
}

impl S<'_, '_> {
    /// Load an operand of `esz`: D`reg` for double precision, else S`reg`, a half precision
    /// value zero extended as `vfp_load_reg16()` loads it.
    fn fload(&mut self, esz: u32, reg: i32) -> TempI64 {
        let t = self.new64();
        let env = self.env();
        let off = vfp_reg_offset(esz == 3, reg) as i64;
        match esz {
            3 => self.f().gen_ld_i64(t, env, off),
            2 => self.f().gen_ld32u_i64(t, env, off),
            _ => self.f().gen_ld16u_i64(t, env, off),
        }
        t
    }

    /// Store a result of `esz`: D`reg` for double precision, else the low 32 bits to S`reg`.
    fn fstore(&mut self, esz: u32, reg: i32, t: TempI64) {
        let env = self.env();
        let off = vfp_reg_offset(esz == 3, reg) as i64;
        if esz == 3 {
            self.f().gen_st_i64(t, env, off);
        } else {
            self.f().gen_st32_i64(t, env, off);
        }
    }

    /// `vfp_f16_offset()`: the half of S`reg` that VCVTB (`top` clear) or VCVTT uses.
    fn f16_off(reg: i32, top: i32) -> i64 {
        (vfp_reg_offset(false, reg) + if top != 0 { 2 } else { 0 }) as i64
    }

    /// Call `a32_vfp` for operation `op` on elements of `esz`, with the accumulator `d`.
    fn vfp_call(&mut self, desc: i32, n: TempI64, m: TempI64, d: TempI64) -> TempI64 {
        let r = self.new64();
        let env = self.env();
        let desc = self.c32(desc);
        let args = [env.into(), n.into(), m.into(), d.into(), desc.into()];
        self.call(&a32_vfp::VFP, Some(r.into()), &args);
        r
    }

    /// Operation `op` in the FPSCR rounding mode.
    fn fop(&mut self, esz: u32, op: u32, n: TempI64, m: TempI64) -> TempI64 {
        let z = self.c64(0);
        self.vfp_call(a32_vfp::desc(op, esz, RMODE_FPCR, 0, false), n, m, z)
    }

    /// One operand operation `op` in rounding mode `rmode`, with immediate `imm`.
    fn fop1(&mut self, esz: u32, op: u32, rmode: u32, imm: u32, i16: bool, n: TempI64) -> TempI64 {
        let z = self.c64(0);
        self.vfp_call(a32_vfp::desc(op, esz, rmode, imm, i16), n, z, z)
    }

    /// `gen_vfp_neg*()`: flip the sign bit.
    fn fneg(&mut self, esz: u32, t: TempI64) -> TempI64 {
        let r = self.new64();
        self.f().gen_xori_i64(r, t, 1 << ((8 << esz) - 1));
        r
    }

    /// `gen_vfp_abs*()`: clear the sign bit.
    fn fabs(&mut self, esz: u32, t: TempI64) -> TempI64 {
        let r = self.new64();
        self.f().gen_andi_i64(r, t, (u64::MAX >> (65 - (8 << esz))) as i64);
        r
    }

    /// `do_vfp_3op_*()`: `vd = f(vd, vn, vm)` on elements of `esz`.
    fn vfp3(
        &mut self,
        esz: u32,
        vd: i32,
        vn: i32,
        vm: i32,
        f: impl FnOnce(&mut Self, TempI64, TempI64, TempI64) -> TempI64,
    ) -> bool {
        let n = self.fload(esz, vn);
        let m = self.fload(esz, vm);
        let d = self.fload(esz, vd);
        let r = f(self, d, n, m);
        self.fstore(esz, vd, r);
        true
    }

    /// `do_vfp_2op_*()`: `vd = f(vm)` on elements of `esz`.
    fn vfp2(
        &mut self,
        esz: u32,
        vd: i32,
        vm: i32,
        f: impl FnOnce(&mut Self, TempI64) -> TempI64,
    ) -> bool {
        let m = self.fload(esz, vm);
        let r = f(self, m);
        self.fstore(esz, vd, r);
        true
    }

    /// `gen_VMLA_*()` and the other multiply accumulates that round twice:
    /// `vd = (neg_d ? -vd : vd) + (neg_p ? -(vn * vm) : vn * vm)`.
    fn mac(&mut self, esz: u32, a: &arg_disas_vfp0, neg_p: bool, neg_d: bool) -> bool {
        self.vfp3(esz, a.vd, a.vn, a.vm, |s, d, n, m| {
            let mut p = s.fop(esz, op::MUL, n, m);
            if neg_p {
                p = s.fneg(esz, p);
            }
            let d = if neg_d { s.fneg(esz, d) } else { d };
            s.fop(esz, op::ADD, d, p)
        })
    }

    /// `do_vfm_*()`: the fused `vd = (neg_d ? -vd : vd) + (neg_n ? -vn : vn) * vm`.
    fn vfm(&mut self, esz: u32, a: &arg_disas_vfp0, neg_n: bool, neg_d: bool) -> bool {
        self.vfp3(esz, a.vd, a.vn, a.vm, |s, d, n, m| {
            let n = if neg_n { s.fneg(esz, n) } else { n };
            let d = if neg_d { s.fneg(esz, d) } else { d };
            s.vfp_call(a32_vfp::desc(op::MULADD, esz, RMODE_FPCR, 0, false), n, m, d)
        })
    }

    /// `trans_VCMP_*()`: compare with Vm, or with +0.0 for the Z form.
    fn vcmp(&mut self, esz: u32, a: &arg_disas_vfp11) -> bool {
        if a.z != 0 && a.vm != 0 {
            return false;
        }
        let n = self.fload(esz, a.vd);
        let m = if a.z != 0 { self.c64(0) } else { self.fload(esz, a.vm) };
        let op = if a.e != 0 { op::CMPE } else { op::CMP };
        self.fop(esz, op, n, m);
        true
    }

    /// `trans_VCVT_int_*()`: the integer in S`vm` to a float of `esz` in `vd`.
    fn vcvt_from_int(&mut self, esz: u32, a: &arg_disas_vfp13) -> bool {
        let op = if a.s != 0 { op::SITO } else { op::UITO };
        let m = self.fload(2, a.vm);
        let r = self.fop1(esz, op, RMODE_FPCR, 0, false, m);
        self.fstore(esz, a.vd, r);
        true
    }

    /// `trans_VCVT_*_int()`: the float of `esz` in `vm` to an integer in S`vd`, rounding to
    /// zero for the RZ form and in the FPSCR mode otherwise.
    fn vcvt_to_int(&mut self, esz: u32, a: &arg_disas_vfp15) -> bool {
        let op = if a.s != 0 { op::TOSI } else { op::TOUI };
        let rmode = if a.rz != 0 { rm(RoundMode::ToZero) } else { RMODE_FPCR };
        let m = self.fload(esz, a.vm);
        let r = self.fop1(esz, op, rmode, 0, false, m);
        self.fstore(2, a.vd, r);
        true
    }

    /// `trans_VCVT_fix_*()`: between a float of `esz` and a fixed point number in the same
    /// register. `opc` is op:U:sx: to fixed point, unsigned, 32 bits.
    fn vcvt_fix(&mut self, esz: u32, a: &arg_disas_vfp14) -> bool {
        let sx = a.opc & 1 != 0;
        let frac_bits = (if sx { 32 } else { 16 }) - a.imm;
        let to_fix = a.opc & 4 != 0;
        let unsigned = a.opc & 2 != 0;
        // The integer takes the whole register, a half precision float its low half.
        let isz = if esz == 3 { 3 } else { 2 };
        let (op, rmode, from) = match (to_fix, unsigned) {
            (true, false) => (op::TOSI, RoundMode::ToZero, esz),
            (true, true) => (op::TOUI, RoundMode::ToZero, esz),
            (false, false) => (op::SITO, RoundMode::NearestEven, isz),
            (false, true) => (op::UITO, RoundMode::NearestEven, isz),
        };
        let v = self.fload(from, a.vd);
        let r = self.fop1(esz, op, rm(rmode), frac_bits as u32, !sx, v);
        let to = if to_fix { isz } else { esz.max(2) };
        self.fstore(to, a.vd, r);
        true
    }

    /// The VCVT from half precision: the half `t` of S`vm` to `to` in `vd`.
    fn vcvt_from_f16(&mut self, to: u32, a: &arg_disas_vfp12) -> bool {
        let m = self.new64();
        let env = self.env();
        self.f().gen_ld16u_i64(m, env, Self::f16_off(a.vm, a.t));
        let r = self.fop1(1, op::CVT, RMODE_FPCR, to, false, m);
        self.fstore(to, a.vd, r);
        true
    }

    /// The VCVT to half precision: the float of `from` in `vm` to the half `t` of S`vd`.
    fn vcvt_to_f16(&mut self, from: u32, a: &arg_disas_vfp12) -> bool {
        let m = self.fload(from, a.vm);
        let r = self.fop1(from, op::CVT, RMODE_FPCR, 1, false, m);
        let env = self.env();
        self.f().gen_st16_i64(r, env, Self::f16_off(a.vd, a.t));
        true
    }

    /// `trans_VMOV_imm_*()`.
    fn vmov_imm(&mut self, esz: u32, a: &arg_disas_vfp10) -> bool {
        let t = self.c64(vfp_expand_imm(esz, a.imm as u32) as i64);
        self.fstore(esz, a.vd, t);
        true
    }
}

/// The `trans_` hooks of one VFP operation in half, single and double precision.
macro_rules! hsd {
    ($($hp:ident, $sp:ident, $dp:ident, $arg:ident: |$s:ident, $a:ident, $e:ident| $body:expr;)*) => {$(
        fn $hp(&mut self, a: &mut $arg) -> bool {
            let ($s, $a, $e) = (self, &*a, 1);
            $body
        }
        fn $sp(&mut self, a: &mut $arg) -> bool {
            let ($s, $a, $e) = (self, &*a, 2);
            $body
        }
        fn $dp(&mut self, a: &mut $arg) -> bool {
            let ($s, $a, $e) = (self, &*a, 3);
            $body
        }
    )*};
}

/// The `trans_` hooks of one VFP operation in single and double precision.
macro_rules! sd {
    ($($sp:ident, $dp:ident, $arg:ident: |$s:ident, $a:ident, $e:ident| $body:expr;)*) => {$(
        fn $sp(&mut self, a: &mut $arg) -> bool {
            let ($s, $a, $e) = (self, &*a, 2);
            $body
        }
        fn $dp(&mut self, a: &mut $arg) -> bool {
            let ($s, $a, $e) = (self, &*a, 3);
            $body
        }
    )*};
}

/// The VFP instructions of `translate-vfp.c`, but for the ones the module documentation
/// lists.
impl DisasVfp for S<'_, '_> {
    fn trans_VMSR_VMRS(&mut self, a: &mut arg_VMSR_VMRS) -> bool {
        // Only FPSCR: FPSID, FPEXC and the MVFRs are UNDEF in User mode, and VPR and P0 are
        // M profile.
        if a.reg != 1 {
            return false;
        }
        let env = self.env();
        if a.l != 0 {
            // VMRS, move VFP reg to gp reg
            let t = self.new32();
            self.call(&h::VFP_GET_FPSCR, Some(t.into()), &[env.into()]);
            if a.rt == 15 {
                // Set the 4 flag bits in the CPSR.
                self.f().gen_andi_i32(t, t, 0xf000_0000_u32 as i32);
                self.set_nzcv(t);
            } else {
                self.store_reg(a.rt, t);
            }
        } else {
            // VMSR, move gp reg to VFP reg
            let t = self.load_reg(a.rt);
            self.call(&h::VFP_SET_FPSCR, None, &[env.into(), t.into()]);
            self.gen_lookup_tb();
        }
        true
    }

    fn trans_VMOV_to_gp(&mut self, a: &mut arg_VMOV_to_gp) -> bool {
        // VMOV scalar to general purpose register
        let off = (vfp_reg_offset(true, a.vn) + ((a.index as usize) << a.size)) as i64;
        let t = self.new32();
        let env = self.env();
        let f = self.f();
        match (a.size, a.u != 0) {
            (0, false) => f.gen_ld8s_i32(t, env, off),
            (0, true) => f.gen_ld8u_i32(t, env, off),
            (1, false) => f.gen_ld16s_i32(t, env, off),
            (1, true) => f.gen_ld16u_i32(t, env, off),
            _ => f.gen_ld_i32(t, env, off),
        }
        self.store_reg(a.rt, t);
        true
    }

    fn trans_VMOV_from_gp(&mut self, a: &mut arg_VMOV_from_gp) -> bool {
        // VMOV general purpose register to scalar
        let off = (vfp_reg_offset(true, a.vn) + ((a.index as usize) << a.size)) as i64;
        let t = self.load_reg(a.rt);
        let env = self.env();
        let f = self.f();
        match a.size {
            0 => f.gen_st8_i32(t, env, off),
            1 => f.gen_st16_i32(t, env, off),
            _ => f.gen_st_i32(t, env, off),
        }
        true
    }

    fn trans_VDUP(&mut self, a: &mut arg_VDUP) -> bool {
        // VDUP (general purpose register)
        if (a.b != 0 && a.e != 0) || (a.q != 0 && a.vn & 1 != 0) {
            return false;
        }
        let size = if a.b != 0 {
            0
        } else if a.e != 0 {
            1
        } else {
            2
        };
        let t = self.load_reg(a.rt);
        let x = self.dup64(t, size);
        self.st_dreg_q(x, a.vn, a.q != 0);
        true
    }

    fn trans_VMOV_single(&mut self, a: &mut arg_VMOV_single) -> bool {
        if a.l != 0 {
            // VFP to general purpose register
            let t = self.vfp_load_reg32(a.vn);
            if a.rt == 15 {
                // Set the 4 flag bits in the CPSR.
                self.set_nzcv(t);
            } else {
                self.store_reg(a.rt, t);
            }
        } else {
            // general purpose register to VFP
            let t = self.load_reg(a.rt);
            self.vfp_store_reg32(t, a.vn);
        }
        true
    }

    fn trans_VMOV_64_sp(&mut self, a: &mut arg_VMOV_64_sp) -> bool {
        if a.op != 0 {
            // fpreg to gpreg
            let t = self.vfp_load_reg32(a.vm);
            self.store_reg(a.rt, t);
            let t = self.vfp_load_reg32(a.vm + 1);
            self.store_reg(a.rt2, t);
        } else {
            // gpreg to fpreg
            let t = self.load_reg(a.rt);
            self.vfp_store_reg32(t, a.vm);
            let t = self.load_reg(a.rt2);
            self.vfp_store_reg32(t, a.vm + 1);
        }
        true
    }

    fn trans_VMOV_64_dp(&mut self, a: &mut arg_VMOV_64_dp) -> bool {
        // D16-D31 exist: the model has 32 double registers.
        if a.op != 0 {
            // fpreg to gpreg
            let t = self.vfp_load_reg32(a.vm * 2);
            self.store_reg(a.rt, t);
            let t = self.vfp_load_reg32(a.vm * 2 + 1);
            self.store_reg(a.rt2, t);
        } else {
            // gpreg to fpreg
            let t = self.load_reg(a.rt);
            self.vfp_store_reg32(t, a.vm * 2);
            let t = self.load_reg(a.rt2);
            self.vfp_store_reg32(t, a.vm * 2 + 1);
        }
        true
    }

    fn trans_VLDR_VSTR_sp(&mut self, a: &mut arg_VLDR_VSTR_sp) -> bool {
        let offset = if a.u != 0 { a.imm << 2 } else { -(a.imm << 2) };
        // For thumb, use of PC is UNPREDICTABLE.
        let addr = self.add_reg_for_lit(a.rn, offset);
        let idx = self.d.mmu_idx;
        if a.l != 0 {
            let t = self.new32();
            self.ld32(t, addr, idx, MemOp::UL | MemOp::ALIGN);
            self.vfp_store_reg32(t, a.vd);
        } else {
            let t = self.vfp_load_reg32(a.vd);
            self.st32(t, addr, idx, MemOp::UL | MemOp::ALIGN);
        }
        true
    }

    fn trans_VLDR_VSTR_dp(&mut self, a: &mut arg_VLDR_VSTR_dp) -> bool {
        let offset = if a.u != 0 { a.imm << 2 } else { -(a.imm << 2) };
        // For thumb, use of PC is UNPREDICTABLE.
        let addr = self.add_reg_for_lit(a.rn, offset);
        let off = vfp_reg_offset(true, a.vd);
        if a.l != 0 {
            let t = self.new64();
            self.ld64(t, addr, MemOp::UQ | MemOp::ALIGN_4);
            self.st_env64(t, off);
        } else {
            let t = self.ld_env64(off);
            self.st64(t, addr, MemOp::UQ | MemOp::ALIGN_4);
        }
        true
    }

    fn trans_VLDM_VSTM_sp(&mut self, a: &mut arg_VLDM_VSTM_sp) -> bool {
        let n = a.imm;
        if n == 0 || a.vd + n > 32 {
            // UNPREDICTABLE cases for bad immediates: we choose to UNDEF to avoid generating
            // huge numbers of TCG ops.
            return false;
        }
        if a.rn == 15 && a.w != 0 {
            // writeback to PC is UNPREDICTABLE, we choose to UNDEF
            return false;
        }
        self.vldm_vstm(a, false, n)
    }

    fn trans_VLDM_VSTM_dp(&mut self, a: &mut arg_VLDM_VSTM_dp) -> bool {
        let n = a.imm >> 1;
        if n == 0 || a.vd + n > 32 || n > 16 {
            // UNPREDICTABLE cases for bad immediates: we choose to UNDEF to avoid generating
            // huge numbers of TCG ops.
            return false;
        }
        if a.rn == 15 && a.w != 0 {
            // writeback to PC is UNPREDICTABLE, we choose to UNDEF
            return false;
        }
        self.vldm_vstm(a, true, n)
    }

    fn trans_VLDR_VSTR_hp(&mut self, a: &mut arg_VLDR_VSTR_hp) -> bool {
        let offset = if a.u != 0 { a.imm << 1 } else { -(a.imm << 1) };
        // For thumb, use of PC is UNPREDICTABLE.
        let addr = self.add_reg_for_lit(a.rn, offset);
        let idx = self.d.mmu_idx;
        if a.l != 0 {
            let t = self.new32();
            self.ld32(t, addr, idx, MemOp::UW | MemOp::ALIGN);
            self.vfp_store_reg32(t, a.vd);
        } else {
            let t = self.vfp_load_reg32(a.vd);
            self.st32(t, addr, idx, MemOp::UW | MemOp::ALIGN);
        }
        true
    }

    fn trans_VMOV_half(&mut self, a: &mut arg_VMOV_half) -> bool {
        if a.rt == 15 {
            // UNPREDICTABLE; we choose to UNDEF
            return false;
        }
        if a.l != 0 {
            // VFP to general purpose register
            let t = self.vfp_load_reg32(a.vn);
            self.f().gen_andi_i32(t, t, 0xffff);
            self.store_reg(a.rt, t);
        } else {
            // general purpose register to VFP
            let t = self.load_reg(a.rt);
            self.f().gen_andi_i32(t, t, 0xffff);
            self.vfp_store_reg32(t, a.vn);
        }
        true
    }

    hsd! {
        trans_VMLA_hp, trans_VMLA_sp, trans_VMLA_dp, arg_disas_vfp0:
            |s, a, e| s.mac(e, a, false, false);
        trans_VMLS_hp, trans_VMLS_sp, trans_VMLS_dp, arg_disas_vfp0:
            |s, a, e| s.mac(e, a, true, false);
        trans_VNMLS_hp, trans_VNMLS_sp, trans_VNMLS_dp, arg_disas_vfp0:
            |s, a, e| s.mac(e, a, false, true);
        trans_VNMLA_hp, trans_VNMLA_sp, trans_VNMLA_dp, arg_disas_vfp0:
            |s, a, e| s.mac(e, a, true, true);
        trans_VMUL_hp, trans_VMUL_sp, trans_VMUL_dp, arg_disas_vfp0:
            |s, a, e| s.vfp3(e, a.vd, a.vn, a.vm, |s, _, n, m| s.fop(e, op::MUL, n, m));
        trans_VNMUL_hp, trans_VNMUL_sp, trans_VNMUL_dp, arg_disas_vfp0:
            |s, a, e| s.vfp3(e, a.vd, a.vn, a.vm, |s, _, n, m| {
                let p = s.fop(e, op::MUL, n, m);
                s.fneg(e, p)
            });
        trans_VADD_hp, trans_VADD_sp, trans_VADD_dp, arg_disas_vfp0:
            |s, a, e| s.vfp3(e, a.vd, a.vn, a.vm, |s, _, n, m| s.fop(e, op::ADD, n, m));
        trans_VSUB_hp, trans_VSUB_sp, trans_VSUB_dp, arg_disas_vfp0:
            |s, a, e| s.vfp3(e, a.vd, a.vn, a.vm, |s, _, n, m| s.fop(e, op::SUB, n, m));
        trans_VDIV_hp, trans_VDIV_sp, trans_VDIV_dp, arg_disas_vfp0:
            |s, a, e| s.vfp3(e, a.vd, a.vn, a.vm, |s, _, n, m| s.fop(e, op::DIV, n, m));
        trans_VFMA_hp, trans_VFMA_sp, trans_VFMA_dp, arg_disas_vfp0:
            |s, a, e| s.vfm(e, a, false, false);
        trans_VFMS_hp, trans_VFMS_sp, trans_VFMS_dp, arg_disas_vfp0:
            |s, a, e| s.vfm(e, a, true, false);
        trans_VFNMA_hp, trans_VFNMA_sp, trans_VFNMA_dp, arg_disas_vfp0:
            |s, a, e| s.vfm(e, a, true, true);
        trans_VFNMS_hp, trans_VFNMS_sp, trans_VFNMS_dp, arg_disas_vfp0:
            |s, a, e| s.vfm(e, a, false, true);
        trans_VMOV_imm_hp, trans_VMOV_imm_sp, trans_VMOV_imm_dp, arg_disas_vfp10:
            |s, a, e| s.vmov_imm(e, a);
        trans_VABS_hp, trans_VABS_sp, trans_VABS_dp, arg_disas_vfp1:
            |s, a, e| s.vfp2(e, a.vd, a.vm, |s, m| s.fabs(e, m));
        trans_VNEG_hp, trans_VNEG_sp, trans_VNEG_dp, arg_disas_vfp1:
            |s, a, e| s.vfp2(e, a.vd, a.vm, |s, m| s.fneg(e, m));
        trans_VSQRT_hp, trans_VSQRT_sp, trans_VSQRT_dp, arg_disas_vfp1:
            |s, a, e| s.vfp2(e, a.vd, a.vm, |s, m| s.fop1(e, op::SQRT, RMODE_FPCR, 0, false, m));
        trans_VCMP_hp, trans_VCMP_sp, trans_VCMP_dp, arg_disas_vfp11:
            |s, a, e| s.vcmp(e, a);
        trans_VRINTR_hp, trans_VRINTR_sp, trans_VRINTR_dp, arg_disas_vfp1:
            |s, a, e| s.vfp2(e, a.vd, a.vm, |s, m| s.fop1(e, op::RINT, RMODE_FPCR, 0, false, m));
        trans_VRINTZ_hp, trans_VRINTZ_sp, trans_VRINTZ_dp, arg_disas_vfp1:
            |s, a, e| s.vfp2(e, a.vd, a.vm, |s, m| {
                s.fop1(e, op::RINT, rm(RoundMode::ToZero), 0, false, m)
            });
        trans_VRINTX_hp, trans_VRINTX_sp, trans_VRINTX_dp, arg_disas_vfp1:
            |s, a, e| s.vfp2(e, a.vd, a.vm, |s, m| s.fop1(e, op::RINTX, RMODE_FPCR, 0, false, m));
        trans_VCVT_int_hp, trans_VCVT_int_sp, trans_VCVT_int_dp, arg_disas_vfp13:
            |s, a, e| s.vcvt_from_int(e, a);
        trans_VCVT_hp_int, trans_VCVT_sp_int, trans_VCVT_dp_int, arg_disas_vfp15:
            |s, a, e| s.vcvt_to_int(e, a);
        trans_VCVT_fix_hp, trans_VCVT_fix_sp, trans_VCVT_fix_dp, arg_disas_vfp14:
            |s, a, e| s.vcvt_fix(e, a);
    }

    sd! {
        trans_VMOV_reg_sp, trans_VMOV_reg_dp, arg_disas_vfp1:
            |s, a, e| s.vfp2(e, a.vd, a.vm, |_, m| m);
        trans_VCVT_f32_f16, trans_VCVT_f64_f16, arg_disas_vfp12:
            |s, a, e| s.vcvt_from_f16(e, a);
        trans_VCVT_f16_f32, trans_VCVT_f16_f64, arg_disas_vfp12:
            |s, a, e| s.vcvt_to_f16(e, a);
    }

    fn trans_VCVT_sp(&mut self, a: &mut arg_VCVT_sp) -> bool {
        // VCVT.F64.F32 Dd, Sm
        let m = self.fload(2, a.vm);
        let r = self.fop1(2, op::CVT, RMODE_FPCR, 3, false, m);
        self.fstore(3, a.vd, r);
        true
    }

    fn trans_VCVT_dp(&mut self, a: &mut arg_VCVT_dp) -> bool {
        // VCVT.F32.F64 Sd, Dm
        let m = self.fload(3, a.vm);
        let r = self.fop1(3, op::CVT, RMODE_FPCR, 2, false, m);
        self.fstore(2, a.vd, r);
        true
    }
}

/// `fp_decode_rm[]`: the rounding modes of the RM field of VRINT{A,N,P,M} and VCVT{A,N,P,M}.
const FP_DECODE_RM: [RoundMode; 4] =
    [RoundMode::TiesAway, RoundMode::NearestEven, RoundMode::Up, RoundMode::Down];

/// The unconditional VFP instructions of ARMv8, from `translate-vfp.c`.
impl DisasVfpUncond for S<'_, '_> {
    fn trans_VSEL(&mut self, a: &mut arg_VSEL) -> bool {
        let esz = a.sz as u32;
        let frn = self.fload(esz, a.vn);
        let frm = self.fload(esz, a.vm);
        let g = self.g();
        let zero = self.c64(0);
        let (zf, t) = (self.new64(), self.new64());
        self.f().gen_extu_i32_i64(zf, g.zf);
        let dest = self.new64();
        match a.cc {
            0 => {
                // eq: Z
                self.f().gen_movcond_i64(Cond::Eq, dest, zf, zero, frn, frm);
            }
            1 => {
                // vs: V
                self.f().gen_ext_i32_i64(t, g.vf);
                self.f().gen_movcond_i64(Cond::Lt, dest, t, zero, frn, frm);
            }
            _ => {
                // ge: N == V, gt: !Z && N == V
                let x = self.new32();
                self.f().gen_xor_i32(x, g.vf, g.nf);
                self.f().gen_ext_i32_i64(t, x);
                if a.cc == 3 {
                    self.f().gen_movcond_i64(Cond::Ne, dest, zf, zero, frn, frm);
                    self.f().gen_movcond_i64(Cond::Ge, dest, t, zero, dest, frm);
                } else {
                    self.f().gen_movcond_i64(Cond::Ge, dest, t, zero, frn, frm);
                }
            }
        }
        // The half precision form clears the top half of Sd.
        if esz == 1 {
            self.f().gen_andi_i64(dest, dest, 0xffff);
        }
        self.fstore(esz, a.vd, dest);
        true
    }

    hsd! {
        trans_VMAXNM_hp, trans_VMAXNM_sp, trans_VMAXNM_dp, arg_disas_vfp_uncond0:
            |s, a, e| s.vfp3(e, a.vd, a.vn, a.vm, |s, _, n, m| s.fop(e, op::MAXNM, n, m));
        trans_VMINNM_hp, trans_VMINNM_sp, trans_VMINNM_dp, arg_disas_vfp_uncond0:
            |s, a, e| s.vfp3(e, a.vd, a.vn, a.vm, |s, _, n, m| s.fop(e, op::MINNM, n, m));
    }

    fn trans_VRINT(&mut self, a: &mut arg_VRINT) -> bool {
        let esz = a.sz as u32;
        let rmode = rm(FP_DECODE_RM[a.rm as usize]);
        self.vfp2(esz, a.vd, a.vm, |s, m| s.fop1(esz, op::RINT, rmode, 0, false, m))
    }

    fn trans_VCVT(&mut self, a: &mut arg_VCVT) -> bool {
        let esz = a.sz as u32;
        let rmode = rm(FP_DECODE_RM[a.rm as usize]);
        let op = if a.op != 0 { op::TOSI } else { op::TOUI };
        let m = self.fload(esz, a.vm);
        let r = self.fop1(esz, op, rmode, 0, false, m);
        self.fstore(2, a.vd, r);
        true
    }

    fn trans_VMOVX(&mut self, a: &mut arg_VMOVX) -> bool {
        let t = self.vfp_load_reg32(a.vm);
        self.f().gen_shri_i32(t, t, 16);
        self.vfp_store_reg32(t, a.vd);
        true
    }

    fn trans_VINS(&mut self, a: &mut arg_VINS) -> bool {
        let rm = self.vfp_load_reg32(a.vm);
        let rd = self.vfp_load_reg32(a.vd);
        self.f().gen_deposit_i32(rd, rd, rm, 16, 16);
        self.vfp_store_reg32(rd, a.vd);
        true
    }
}
