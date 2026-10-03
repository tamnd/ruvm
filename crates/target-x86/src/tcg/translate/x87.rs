// SPDX-License-Identifier: GPL-2.0-or-later

//! The x87 instructions, D8 to DF, and FWAIT: `gen_x87()` of `translate.c` and `gen_WAIT()`
//! of `emit.c.inc`.
//!
//! Each QEMU x87 helper call becomes a call to the one [`fpu::X87`] helper with the
//! operation packed by [`x87_op`]. Memory operands that QEMU loads or stores in generated
//! code are loaded and stored here the same way, so the helper only sees values; the
//! environment, FSAVE, FRSTOR, FLDT, FSTPT, FBLD and FBSTP forms pass the address.

use super::super::EXCP07_PREX;
use super::super::env::{FPCS, FPDP, FPDS, FPIP, SEG_SELECTOR, SEG_SIZE, SEGS};
use super::super::helpers::fpu::{self, ARITH_ST0_FT0, ARITH_STN_ST0, F, x87_op};
use super::*;
use crate::state::{HF_EM_MASK, HF_MP_MASK, HF_TS_MASK, R_CS, R_EAX, R_EDX};

/// `fcmov_cc[]`: the condition each FCMOVcc row tests, before the negation bit.
const FCMOV_CC: [u32; 4] = [JCC_B << 1, JCC_Z << 1, JCC_BE << 1, JCC_P << 1];

impl S<'_, '_, '_> {
    /// Call the x87 helper for `f` on ST(`n`) with value or address `v`, and return what it
    /// returns.
    fn x87_call(&mut self, f: F, n: u32, v: Option<TempI64>) -> TempI64 {
        let op = self.c32(x87_op(f, n));
        let v = match v {
            Some(v) => v,
            None => self.c64(0),
        };
        let r = self.new64();
        self.env_call(&fpu::X87, Some(r.into()), &[op.into(), v.into()]);
        r
    }

    /// An x87 operation with no operand.
    fn x87_do(&mut self, f: F, n: u32) {
        self.x87_call(f, n, None);
    }

    /// `gen_helper_fp_arith_ST0_FT0()`.
    fn fp_arith_st0_ft0(&mut self, op: u32) {
        self.x87_do(ARITH_ST0_FT0[op as usize], 0);
        if op == 3 {
            self.x87_do(F::Fpop, 0);
        }
    }

    /// Load `mop` from A0 into a new temp.
    fn x87_ld(&mut self, mop: MemOp) -> TempI64 {
        let (t, a0, idx) = (self.new64(), self.g.a0, self.d.mem_index);
        self.f().gen_qemu_ld_i64(t, a0, idx, mop);
        t
    }

    /// Store `t` to A0 with `mop`.
    fn x87_st(&mut self, t: TempI64, mop: MemOp) {
        let (a0, idx) = (self.g.a0, self.d.mem_index);
        self.f().gen_qemu_st_i64(t, a0, idx, mop);
    }

    /// Store the selector of segment `seg` into the 16 bit FPCS or FPDS field at `off`.
    fn x87_st_sel(&mut self, seg: usize, off: usize) {
        let sel = self.ld_env32(SEGS + seg * SEG_SIZE + SEG_SELECTOR);
        let env = self.g.env;
        self.f().gen_st16_i64(sel, env, off as i64);
    }

    /// FWAIT (9B): `gen_WAIT()`.
    pub(super) fn fwait(&mut self) {
        if self.d.flags & (HF_MP_MASK | HF_TS_MASK) == HF_MP_MASK | HF_TS_MASK {
            self.gen_exception(EXCP07_PREX);
        } else {
            // This needs to be treated as I/O because of ferr_irq.
            self.b.translator_io_start();
            self.x87_do(F::Fwait, 0);
        }
    }

    /// D8 to DF: `gen_x87()`.
    pub(super) fn x87(&mut self, b: u32) -> R {
        let (modrm, md, _, _) = self.modrm()?;
        let addr = if md != 3 { Some(self.lea_modrm_0(modrm)?) } else { None };
        if self.d.flags & (HF_EM_MASK | HF_TS_MASK) != 0 {
            // If CR0.EM or CR0.TS is set, raise an FPU exception.
            self.gen_exception(EXCP07_PREX);
            return Ok(());
        }
        let rm = modrm & 7;
        let op = ((b & 7) << 3) | ((modrm >> 3) & 7);
        let update_fip = match addr {
            Some(a) => self.x87_mem(op, a),
            None => self.x87_reg(op, rm),
        };
        let Some(update_fip) = update_fip else {
            self.gen_illegal_opcode();
            return Ok(());
        };
        if update_fip {
            self.x87_st_sel(R_CS, FPCS);
            let eip = self.eip_cur() as i64;
            let t = self.c64(eip);
            self.st_env64(t, FPIP);
        }
        Ok(())
    }

    /// The memory forms of `gen_x87()`. Returns whether to update FPCS and FPIP, or `None`
    /// for an illegal opcode. The forms that update FPCS and FPIP also update FPDS and FPDP.
    fn x87_mem(&mut self, op: u32, a: Addr) -> Option<bool> {
        use MemOp as M;
        let ea = self.lea_modrm_1(a);
        let last_addr = self.new64();
        self.f().gen_mov_i64(last_addr, ea);
        let (aflag, ovr) = (self.d.aflag, self.d.override_seg);
        self.lea_v_seg(aflag, ea, a.def_seg, ovr);
        let a0 = self.g.a0;
        let data32 = self.d.dflag - 1;
        match op {
            0x00..=0x07 => {
                let t = self.x87_ld(M::LEUL);
                self.x87_call(F::FldsFt0, 0, Some(t));
                self.fp_arith_st0_ft0(op & 7);
            }
            0x10..=0x17 => {
                let t = self.x87_ld(M::LEUL);
                self.x87_call(F::FildlFt0, 0, Some(t));
                self.fp_arith_st0_ft0(op & 7);
            }
            0x20..=0x27 => {
                let t = self.x87_ld(M::LEUQ);
                self.x87_call(F::FldlFt0, 0, Some(t));
                self.fp_arith_st0_ft0(op & 7);
            }
            0x30..=0x37 => {
                let t = self.x87_ld(M::LESW);
                self.x87_call(F::FildlFt0, 0, Some(t));
                self.fp_arith_st0_ft0(op & 7);
            }
            0x08 | 0x18 | 0x28 | 0x38 | 0x3d => {
                let (mop, f) = match op {
                    0x08 => (M::LEUL, F::FldsSt0),
                    0x18 => (M::LEUL, F::FildlSt0),
                    0x28 => (M::LEUQ, F::FldlSt0),
                    0x38 => (M::LESW, F::FildlSt0),
                    _ => (M::LEUQ, F::FildllSt0),
                };
                let t = self.x87_ld(mop);
                self.x87_call(f, 0, Some(t));
            }
            0x19 | 0x29 | 0x39 | 0x0a | 0x0b | 0x1a | 0x1b | 0x2a | 0x2b | 0x3a | 0x3b | 0x3f => {
                let (f, mop) = match op {
                    0x19 => (F::FisttlSt0, M::LEUL),
                    0x29 => (F::FisttllSt0, M::LEUQ),
                    0x39 => (F::FisttSt0, M::LEUW),
                    0x0a | 0x0b => (F::FstsSt0, M::LEUL),
                    0x1a | 0x1b => (F::FistlSt0, M::LEUL),
                    0x2a | 0x2b => (F::FstlSt0, M::LEUQ),
                    0x3a | 0x3b => (F::FistSt0, M::LEUW),
                    _ => (F::FistllSt0, M::LEUQ),
                };
                let t = self.x87_call(f, 0, None);
                self.x87_st(t, mop);
                // The FISTTP forms (/1) and FISTP m64 always pop, the rest only for /3.
                if op & 7 != 2 {
                    self.x87_do(F::Fpop, 0);
                }
            }
            0x0c | 0x0e | 0x2c | 0x2e => {
                let f = match op {
                    0x0c => F::Fldenv,
                    0x0e => F::Fstenv,
                    0x2c => F::Frstor,
                    _ => F::Fsave,
                };
                self.x87_call(f, data32, Some(a0));
                return Some(false);
            }
            0x0d => {
                let t = self.x87_ld(M::LEUW);
                self.x87_call(F::Fldcw, 0, Some(t));
                return Some(false);
            }
            0x0f | 0x2f => {
                let f = if op == 0x0f { F::Fnstcw } else { F::Fnstsw };
                let t = self.x87_call(f, 0, None);
                self.x87_st(t, M::LEUW);
                return Some(false);
            }
            0x1d => {
                self.x87_call(F::FldtSt0, 0, Some(a0));
            }
            0x3c => {
                self.x87_call(F::FbldSt0, 0, Some(a0));
            }
            0x1f | 0x3e => {
                let f = if op == 0x1f { F::FsttSt0 } else { F::FbstSt0 };
                self.x87_call(f, 0, Some(a0));
                self.x87_do(F::Fpop, 0);
            }
            _ => return None,
        }
        let ovr = self.d.override_seg;
        let last_seg = if ovr >= 0 { ovr } else { a.def_seg };
        self.x87_st_sel(last_seg as usize, FPDS);
        self.st_env64(last_addr, FPDP);
        Some(true)
    }

    /// The register forms of `gen_x87()`. Returns whether to update FPCS and FPIP, or `None`
    /// for an illegal opcode.
    fn x87_reg(&mut self, op: u32, rm: u32) -> Option<bool> {
        let pop_if = |s: &mut Self, c: bool| {
            if c {
                s.x87_do(F::Fpop, 0);
            }
        };
        match op {
            // FLD ST(i).
            0x08 => {
                self.x87_do(F::Fpush, 0);
                self.x87_do(F::FmovSt0Stn, (rm + 1) & 7);
            }
            // FXCH, and the undocumented FXCH4 and FXCH7.
            0x09 | 0x29 | 0x39 => self.x87_do(F::FxchgSt0Stn, rm),
            0x0a => {
                // FNOP checks for exceptions (the FreeBSD FPU probe) and needs to be treated
                // as I/O because of ferr_irq.
                if rm != 0 {
                    return None;
                }
                self.b.translator_io_start();
                self.x87_do(F::Fwait, 0);
                return Some(false);
            }
            0x0c => match rm {
                0 => self.x87_do(F::Fchs, 0),
                1 => self.x87_do(F::Fabs, 0),
                4 => {
                    // FTST.
                    self.x87_do(F::FldzFt0, 0);
                    self.x87_do(F::FcomSt0Ft0, 0);
                }
                5 => self.x87_do(F::Fxam, 0),
                _ => return None,
            },
            0x0d => {
                const K: [F; 7] =
                    [F::Fld1, F::Fldl2t, F::Fldl2e, F::Fldpi, F::Fldlg2, F::Fldln2, F::FldzSt0];
                let f = *K.get(rm as usize)?;
                self.x87_do(F::Fpush, 0);
                self.x87_do(f, 0);
            }
            0x0e => {
                const K: [F; 8] = [
                    F::F2xm1,
                    F::Fyl2x,
                    F::Fptan,
                    F::Fpatan,
                    F::Fxtract,
                    F::Fprem1,
                    F::Fdecstp,
                    F::Fincstp,
                ];
                self.x87_do(K[rm as usize], 0);
            }
            0x0f => {
                const K: [F; 8] = [
                    F::Fprem,
                    F::Fyl2xp1,
                    F::Fsqrt,
                    F::Fsincos,
                    F::Frndint,
                    F::Fscale,
                    F::Fsin,
                    F::Fcos,
                ];
                self.x87_do(K[rm as usize], 0);
            }
            // FADD to FDIVR ST, ST(i), and the undocumented FCOM2, FCOMP3 and FCOMP5.
            0x00..=0x07 | 0x22 | 0x23 | 0x32 => {
                self.x87_do(F::FmovFt0Stn, rm);
                self.fp_arith_st0_ft0(op & 7);
                pop_if(self, op >= 0x30);
            }
            // FADD to FDIVR ST(i), ST, and the popping forms.
            0x20 | 0x21 | 0x24..=0x27 | 0x30 | 0x31 | 0x34..=0x37 => {
                self.x87_do(ARITH_STN_ST0[(op & 7) as usize], rm);
                pop_if(self, op >= 0x30);
            }
            // FUCOMPP.
            0x15 => {
                if rm != 1 {
                    return None;
                }
                self.x87_do(F::FmovFt0Stn, 1);
                self.x87_do(F::FucomSt0Ft0, 0);
                self.x87_do(F::Fpop, 0);
                self.x87_do(F::Fpop, 0);
            }
            0x1c => match rm {
                // FENI, FDISI and FSETPM only mean something on the 287: do nothing.
                0 | 1 | 4 => {}
                2 => {
                    self.x87_do(F::Fclex, 0);
                    return Some(false);
                }
                3 => {
                    self.x87_do(F::Fninit, 0);
                    return Some(false);
                }
                _ => return None,
            },
            // FUCOMI, FUCOMIP, FCOMI and FCOMIP.
            0x1d | 0x3d | 0x1e | 0x3e => {
                if !self.d.feat.cmov {
                    return None;
                }
                self.gen_update_cc_op();
                self.x87_do(F::FmovFt0Stn, rm);
                let f = if op & 7 == 5 { F::FucomiSt0Ft0 } else { F::FcomiSt0Ft0 };
                self.x87_do(f, 0);
                pop_if(self, op >= 0x30);
                self.assume_cc_op(CC_OP_EFLAGS);
            }
            // FFREE and the undocumented FFREEP.
            0x28 | 0x38 => {
                self.x87_do(F::FfreeStn, rm);
                pop_if(self, op >= 0x30);
            }
            // FST, FSTP, and the undocumented FSTP1, FSTP8 and FSTP9.
            0x2a | 0x2b | 0x0b | 0x3a | 0x3b => {
                self.x87_do(F::FmovStnSt0, rm);
                pop_if(self, op != 0x2a);
            }
            // FUCOM and FUCOMP.
            0x2c | 0x2d => {
                self.x87_do(F::FmovFt0Stn, rm);
                self.x87_do(F::FucomSt0Ft0, 0);
                pop_if(self, op == 0x2d);
            }
            // FCOMPP.
            0x33 => {
                if rm != 1 {
                    return None;
                }
                self.x87_do(F::FmovFt0Stn, 1);
                self.x87_do(F::FcomSt0Ft0, 0);
                self.x87_do(F::Fpop, 0);
                self.x87_do(F::Fpop, 0);
            }
            // FNSTSW AX.
            0x3c => {
                if rm != 0 {
                    return None;
                }
                let t = self.x87_call(F::Fnstsw, 0, None);
                self.mov_reg_v(OT16, R_EAX, t);
            }
            // FCMOVcc.
            0x10..=0x13 | 0x18..=0x1b => {
                if !self.d.feat.cmov {
                    return None;
                }
                let op1 = FCMOV_CC[(op & 3) as usize] | (((op >> 3) & 1) ^ 1);
                let l1 = self.label();
                self.gen_jcc1(op1, l1);
                self.x87_do(F::FmovSt0Stn, rm);
                self.set_label(l1);
            }
            _ => return None,
        }
        Some(true)
    }

    /// FXSAVE, FXRSTOR, XSAVE, XRSTOR and XSAVEOPT: 0F AE /0, /1, /4, /5 and /6 with a memory
    /// operand and no prefix. XSAVEOPT calls the XSAVE helper, as QEMU does.
    pub(super) fn fxsave_xsave(&mut self, m: u32) -> R {
        let op = (m >> 3) & 7;
        let has = match op {
            0 | 1 => self.d.feat.fxsr,
            4 | 5 => self.d.feat.xsave,
            _ => self.d.feat.xsaveopt,
        };
        if !has {
            self.gen_illegal_opcode();
            return Ok(());
        }
        self.gen_lea_modrm(m)?;
        let g = self.g;
        if op < 2 {
            if self.d.flags & (HF_EM_MASK | HF_TS_MASK) != 0 {
                self.gen_exception(EXCP07_PREX);
            } else {
                let d = if op == 0 { &fpu::FXSAVE } else { &fpu::FXRSTOR };
                self.env_call(d, None, &[g.a0.into()]);
            }
            return Ok(());
        }
        let features = self.new64();
        self.f().gen_concat32_i64(features, g.regs[R_EAX], g.regs[R_EDX]);
        let d = if op == 5 { &fpu::XRSTOR } else { &fpu::XSAVE };
        self.env_call(d, None, &[g.a0.into(), features.into()]);
        Ok(())
    }
}
