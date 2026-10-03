// SPDX-License-Identifier: GPL-2.0-or-later

//! The general purpose register extensions of `decode-new.c.inc` and `emit.c.inc`: the VEX
//! prefix with the BMI1 and BMI2 instructions of the 0F 38 F0 to FF rows and RORX, ADCX and
//! ADOX, MOVBE, CRC32, RDRAND, RDSEED, RDPID, XGETBV, XSETBV, the FS and GS base
//! instructions, MOVNTI, LDMXCSR and STMXCSR.

use super::super::EXCP07_PREX;
use super::super::cc::{CC_OP_LOGICB, cc_op_has_eflags};
use super::super::env::{HF_OSFXSR_MASK, MXCSR};
use super::super::helpers::vec;
use super::*;
use crate::state::{CR4_FSGSBASE_MASK, HF_EM_MASK, HF_TS_MASK, R_EAX, R_EDX, R_FS, R_GS};

/// The VEX prefix byte `pp` field, as in `pp_prefix[]`.
const PP_PREFIX: [u32; 4] = [0, PREFIX_DATA, PREFIX_REPZ, PREFIX_REPNZ];

/// What [`S::vex_prefix`] found.
pub(super) enum Vex {
    /// Not a VEX prefix: LES or LDS.
    No,
    /// A VEX prefix for opcode map 1, 2 or 3.
    Map(u32),
    /// An exception has been raised.
    Done,
}

/// The BMI rows of `opcodes_0F38_F0toFF`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Bmi {
    Andn,
    Blsr,
    Blsmsk,
    Blsi,
    Bzhi,
    Pext,
    Pdep,
    Mulx,
    Bextr,
    Shlx,
    Sarx,
    Shrx,
    Rorx,
}

impl S<'_, '_, '_> {
    /// `decode_by_prefix()`: the column of a four entry row.
    fn prefix_col(&self) -> usize {
        let p = self.d.prefix;
        if p & PREFIX_REPNZ != 0 {
            3
        } else if p & PREFIX_REPZ != 0 {
            2
        } else if p & PREFIX_DATA != 0 {
            1
        } else {
            0
        }
    }

    /// The `y` operand size: 32 or 64 bits.
    fn ot_y(&self) -> u32 {
        if self.d.dflag == OT16 { OT32 } else { self.d.dflag }
    }

    /// Parse a C4 or C5 prefix, as the `case 0xc4`/`case 0xc5` of QEMU's `disas_insn()`.
    pub(super) fn vex_prefix(&mut self, b: u32) -> R<Vex> {
        if !self.d.pe || self.d.vm86 {
            return Ok(Vex::No);
        }
        let vex2 = self.ldub()? as u32;
        if !self.code64() && vex2 & 0xc0 != 0xc0 {
            // Rewind the byte: this is LES or LDS.
            self.d.pc -= 1;
            return Ok(Vex::No);
        }
        // No preceding LOCK, 66, F2, F3 or REX prefix.
        let bad = PREFIX_REPZ | PREFIX_REPNZ | PREFIX_LOCK | PREFIX_DATA | PREFIX_REX;
        if self.d.prefix & bad != 0 {
            self.gen_illegal_opcode();
            return Ok(Vex::Done);
        }
        let c64 = self.code64();
        if c64 {
            self.d.rex_r = ((!vex2 >> 4) & 8) as usize;
        }
        let (vex3, map) = if b == 0xc5 {
            (vex2, 1)
        } else {
            let vex3 = self.ldub()? as u32;
            if c64 {
                self.d.rex_x = ((!vex2 >> 3) & 8) as usize;
                self.d.rex_b = ((!vex2 >> 2) & 8) as usize;
            }
            self.d.vex_w = vex3 & 0x80 != 0;
            let map = vex2 & 0x1f;
            if !(1..=3).contains(&map) {
                self.gen_illegal_opcode();
                return Ok(Vex::Done);
            }
            (vex3, map)
        };
        self.d.vex_v = ((!vex3 >> 3) & if c64 { 15 } else { 7 }) as usize;
        self.d.vex_l = vex3 & 4 != 0;
        self.d.prefix |= PP_PREFIX[(vex3 & 3) as usize] | PREFIX_VEX;
        if c64 {
            self.d.rex_w = self.d.vex_w;
        }
        Ok(Vex::Map(map))
    }

    /// Decode an instruction after a VEX prefix. Only the general purpose register
    /// instructions (VEX class 13) are implemented; vector instructions raise #UD.
    pub(super) fn vex_insn(&mut self, map: u32) -> R {
        let b = self.ldub()? as u32;
        let col = self.prefix_col();
        let op = match (map, b) {
            (2, 0xf2) if col == 0 => Some((Bmi::Andn, self.d.feat.bmi1)),
            (2, 0xf3) if col == 0 => {
                let m = self.ldub()? as u32;
                self.d.pc -= 1;
                match (m >> 3) & 7 {
                    1 => Some((Bmi::Blsr, self.d.feat.bmi1)),
                    2 => Some((Bmi::Blsmsk, self.d.feat.bmi1)),
                    3 => Some((Bmi::Blsi, self.d.feat.bmi1)),
                    _ => None,
                }
            }
            (2, 0xf5) => match col {
                0 => Some((Bmi::Bzhi, self.d.feat.bmi1)),
                2 => Some((Bmi::Pext, self.d.feat.bmi2)),
                3 => Some((Bmi::Pdep, self.d.feat.bmi2)),
                _ => None,
            },
            (2, 0xf6) if col == 3 => Some((Bmi::Mulx, self.d.feat.bmi2)),
            (2, 0xf7) => {
                let op = [Bmi::Bextr, Bmi::Shlx, Bmi::Sarx, Bmi::Shrx][col];
                Some((op, self.d.feat.bmi1))
            }
            (3, 0xf0) if col == 3 => Some((Bmi::Rorx, self.d.feat.bmi2)),
            _ => None,
        };
        let op = match op {
            Some((op, true)) if !self.d.vex_l => op,
            _ => {
                self.gen_illegal_opcode();
                return Ok(());
            }
        };
        self.bmi(op)
    }

    /// Load the `E` operand of `modrm` into `t`; the address is computed if needed.
    fn load_e(&mut self, m: u32, ot: u32, t: TempI64, sign: bool) -> R {
        let md = m >> 6;
        let rm = (m & 7) as usize | self.d.rex_b;
        if md == 3 {
            self.mov_v_reg(ot, t, rm);
            if ot < OT64 {
                self.ext(ot, t, t, sign);
            }
        } else {
            self.gen_lea_modrm(m)?;
            let a0 = self.g.a0;
            self.ld_v(ot, t, a0);
            if sign {
                self.ext(ot, t, t, true);
            }
        }
        Ok(())
    }

    /// The BMI1 and BMI2 instructions; `emit.c.inc`'s `gen_ANDN()` to `gen_SHRX()`.
    fn bmi(&mut self, op: Bmi) -> R {
        let g = self.g;
        let ot = self.ot_y();
        let m = self.ldub()? as u32;
        let reg = ((m >> 3) & 7) as usize | self.d.rex_r;
        let vv = self.d.vex_v;
        let bound = self.c64(if ot == OT64 { 63 } else { 31 });
        let zero = self.c64(0);
        let mone = self.c64(-1);
        match op {
            Bmi::Andn => {
                self.mov_v_reg(ot, g.t0, vv);
                self.load_e(m, ot, g.t1, false)?;
                self.f().gen_andc_i64(g.t0, g.t1, g.t0);
                self.mov_reg_v(ot, reg, g.t0);
                self.f().gen_mov_i64(g.cc_dst, g.t0);
                self.set_cc_op(CC_OP_LOGICB + ot);
            }
            Bmi::Blsr | Bmi::Blsmsk | Bmi::Blsi => {
                self.load_e(m, ot, g.t1, false)?;
                let f = self.f();
                let cc = match op {
                    Bmi::Blsr => {
                        f.gen_subi_i64(g.t0, g.t1, 1);
                        f.gen_and_i64(g.t0, g.t0, g.t1);
                        CC_OP_BMILGB
                    }
                    Bmi::Blsmsk => {
                        f.gen_subi_i64(g.t0, g.t1, 1);
                        f.gen_xor_i64(g.t0, g.t0, g.t1);
                        CC_OP_BMILGB
                    }
                    _ => {
                        f.gen_neg_i64(g.t0, g.t1);
                        f.gen_and_i64(g.t0, g.t0, g.t1);
                        CC_OP_BLSIB
                    }
                };
                self.mov_reg_v(ot, vv, g.t0);
                let f = self.f();
                f.gen_mov_i64(g.cc_src, g.t1);
                f.gen_mov_i64(g.cc_dst, g.t0);
                self.set_cc_op(cc + ot);
            }
            Bmi::Bzhi => {
                self.load_e(m, ot, g.t0, false)?;
                self.mov_v_reg(ot, g.t1, vv);
                let f = self.f();
                f.gen_ext8u_i64(g.t1, g.t1);
                f.gen_shl_i64(g.a0, mone, g.t1);
                f.gen_movcond_i64(Cond::Leu, g.a0, g.t1, bound, g.a0, zero);
                f.gen_andc_i64(g.t0, g.t0, g.a0);
                // BMILG clears O, so C is stored inverted.
                f.gen_setcond_i64(Cond::Leu, g.t1, g.t1, bound);
                self.mov_reg_v(ot, reg, g.t0);
                let f = self.f();
                f.gen_mov_i64(g.cc_src, g.t1);
                f.gen_mov_i64(g.cc_dst, g.t0);
                self.set_cc_op(CC_OP_BMILGB + ot);
            }
            Bmi::Pext | Bmi::Pdep => {
                self.mov_v_reg(ot, g.t0, vv);
                self.load_e(m, ot, g.t1, false)?;
                let h = if op == Bmi::Pext { &vec::PEXT } else { &vec::PDEP };
                self.call(h, Some(g.t0.into()), &[g.t0.into(), g.t1.into()]);
                self.mov_reg_v(ot, reg, g.t0);
            }
            Bmi::Mulx => {
                self.load_e(m, ot, g.t0, false)?;
                self.mov_v_reg(ot, g.t1, R_EDX);
                if ot == OT64 {
                    let lo = g.regs[vv];
                    self.f().gen_mulu2_i64(lo, g.t0, g.t0, g.t1);
                } else {
                    let a = self.trunc32(g.t0);
                    let b = self.trunc32(g.t1);
                    let f = self.f();
                    f.gen_mulu2_i32(a, b, a, b);
                    f.gen_extu_i32_i64(g.regs[vv], a);
                    f.gen_extu_i32_i64(g.t0, b);
                }
                self.mov_reg_v(ot, reg, g.t0);
            }
            Bmi::Bextr => {
                self.load_e(m, ot, g.t0, false)?;
                self.mov_v_reg(ot, g.t1, vv);
                let f = self.f();
                // Shifts larger than the operand size give zero.
                f.gen_ext8u_i64(g.a0, g.t1);
                f.gen_shr_i64(g.t0, g.t0, g.a0);
                f.gen_movcond_i64(Cond::Leu, g.t0, g.a0, bound, g.t0, zero);
                // The length as an inverse mask.
                f.gen_extract_i64(g.a0, g.t1, 8, 8);
                f.gen_shl_i64(g.t1, mone, g.a0);
                f.gen_movcond_i64(Cond::Leu, g.t1, g.a0, bound, g.t1, zero);
                f.gen_andc_i64(g.t0, g.t0, g.t1);
                self.mov_reg_v(ot, reg, g.t0);
                self.f().gen_mov_i64(g.cc_dst, g.t0);
                self.set_cc_op(CC_OP_LOGICB + ot);
            }
            Bmi::Shlx | Bmi::Sarx | Bmi::Shrx => {
                self.load_e(m, ot, g.t0, op == Bmi::Sarx)?;
                self.mov_v_reg(ot, g.t1, vv);
                let mask = if ot == OT64 { 63 } else { 31 };
                let f = self.f();
                f.gen_andi_i64(g.t1, g.t1, mask);
                match op {
                    Bmi::Shlx => f.gen_shl_i64(g.t0, g.t0, g.t1),
                    Bmi::Sarx => f.gen_sar_i64(g.t0, g.t0, g.t1),
                    _ => f.gen_shr_i64(g.t0, g.t0, g.t1),
                }
                self.mov_reg_v(ot, reg, g.t0);
            }
            Bmi::Rorx => {
                self.d.rip_offset = 1;
                self.load_e(m, ot, g.t0, false)?;
                let imm = self.ldub()? as i64;
                if ot == OT64 {
                    self.f().gen_rotri_i64(g.t0, g.t0, imm & 63);
                } else {
                    let t = self.trunc32(g.t0);
                    let f = self.f();
                    f.gen_rotri_i32(t, t, (imm & 31) as i32);
                    f.gen_extu_i32_i64(g.t0, t);
                }
                self.mov_reg_v(ot, reg, g.t0);
            }
        }
        Ok(())
    }

    /// The legacy 0F 38 map: MOVBE, CRC32, ADCX and ADOX.
    pub(super) fn op_0f38(&mut self) -> R {
        let g = self.g;
        let b = self.ldub()? as u32;
        let col = self.prefix_col();
        match (b, col) {
            (0xf0 | 0xf1, 0 | 1) => {
                let (m, md, reg, _) = self.modrm()?;
                if !self.d.feat.movbe || md == 3 {
                    self.gen_illegal_opcode();
                    return Ok(());
                }
                let ot = self.d.dflag;
                self.gen_lea_modrm(m)?;
                let (idx, mop) = (self.d.mem_index, MemOp(ot) | MemOp::BE);
                if b == 0xf0 {
                    self.f().gen_qemu_ld_i64(g.t0, g.a0, idx, mop);
                    self.mov_reg_v(ot, reg, g.t0);
                } else {
                    self.mov_v_reg(ot, g.t0, reg);
                    self.f().gen_qemu_st_i64(g.t0, g.a0, idx, mop);
                }
            }
            (0xf0 | 0xf1, 3) => {
                let (m, _, reg, _) = self.modrm()?;
                if !self.d.feat.sse42 {
                    self.gen_illegal_opcode();
                    return Ok(());
                }
                let ot = if b == 0xf0 { OT8 } else { self.d.dflag };
                self.gen_ldst_modrm(m, ot, None, false)?;
                let f = self.f();
                f.gen_mov_i64(g.t1, g.t0);
                self.mov_v_reg(OT32, g.t0, reg);
                let crc = self.trunc32(g.t0);
                let bits = self.c32(8 << ot);
                self.call(&vec::CRC32, Some(g.t0.into()), &[crc.into(), g.t1.into(), bits.into()]);
                self.mov_reg_v(OT32, reg, g.t0);
            }
            (0xf6, 1 | 2) => {
                let (m, _, reg, _) = self.modrm()?;
                if !self.d.feat.adx {
                    self.gen_illegal_opcode();
                    return Ok(());
                }
                let ot = self.ot_y();
                self.mov_v_reg(ot, g.t0, reg);
                self.load_e(m, ot, g.t1, false)?;
                self.gen_adcox(ot, if col == 1 { CC_OP_ADCX } else { CC_OP_ADOX });
                self.mov_reg_v(ot, reg, g.t0);
            }
            // The vector instructions of the 0F 38 map are not implemented.
            _ => self.gen_illegal_opcode(),
        }
        Ok(())
    }

    /// `gen_ADCOX()`: T0 = T0 + T1 + CF (ADCX) or OF (ADOX).
    fn gen_adcox(&mut self, ot: u32, cc_op: u32) {
        let g = self.g;
        let adcx = cc_op == CC_OP_ADCX;
        let mut new_op = cc_op;
        let mut carry_in = None;
        if !cc_op_has_eflags(self.d.cc_op) {
            // Same result as QEMU's gen_mov_eflags() into a fresh CC_SRC.
            self.gen_compute_eflags();
        } else {
            let cur = self.d.cc_op;
            // Re-use the carry out of a previous round.
            if cur == cc_op || cur == CC_OP_ADCOX {
                carry_in = Some(if adcx { g.cc_dst } else { g.cc_src2 });
            }
            // Keep the other carry of previous rounds.
            if cur != cc_op && cur != CC_OP_EFLAGS {
                new_op = CC_OP_ADCOX;
            }
        }
        let carry_in = match carry_in {
            Some(c) => c,
            None => {
                let c = self.new64();
                let bit = if adcx { CC_C } else { CC_O }.trailing_zeros();
                self.f().gen_extract_i64(c, g.cc_src, bit, 1);
                c
            }
        };
        let out = self.new64();
        if ot == OT32 {
            let f = self.f();
            f.gen_ext32u_i64(g.t0, g.t0);
            f.gen_ext32u_i64(g.t1, g.t1);
            f.gen_add_i64(g.t0, g.t0, g.t1);
            f.gen_add_i64(g.t0, g.t0, carry_in);
            f.gen_shri_i64(out, g.t0, 32);
        } else {
            let zero = self.c64(0);
            let f = self.f();
            f.gen_add2_i64(g.t0, out, g.t0, zero, carry_in, zero);
            f.gen_add2_i64(g.t0, out, g.t0, out, g.t1, zero);
        }
        let dst = if adcx { g.cc_dst } else { g.cc_src2 };
        self.f().gen_mov_i64(dst, out);
        self.set_cc_op(new_op);
    }

    /// RDRAND (0F C7 /6), RDSEED and RDPID (0F C7 /7).
    pub(super) fn rdrand(&mut self, md: u32, op: u32, rm: usize) {
        let g = self.g;
        let p = self.d.prefix;
        let rep = PREFIX_REPZ | PREFIX_REPNZ;
        if md != 3 || p & PREFIX_LOCK != 0 {
            self.gen_illegal_opcode();
            return;
        }
        if op == 7 && p & PREFIX_REPNZ == 0 && p & PREFIX_REPZ != 0 {
            // RDPID.
            if !self.d.feat.rdpid {
                self.gen_illegal_opcode();
                return;
            }
            self.env_call(&vec::RDPID, Some(g.t0.into()), &[]);
            let ot = self.d.dflag;
            self.mov_reg_v(ot, rm, g.t0);
            return;
        }
        let ok = if op == 7 { self.d.feat.rdseed } else { self.d.feat.rdrand };
        if p & rep != 0 || !ok {
            self.gen_illegal_opcode();
            return;
        }
        self.b.translator_io_start();
        self.env_call(&vec::RDRAND, Some(g.t0.into()), &[]);
        let ot = self.d.dflag;
        self.mov_reg_v(ot, rm, g.t0);
        self.assume_cc_op(CC_OP_EFLAGS);
    }

    /// XGETBV (0F 01 D0) and XSETBV (0F 01 D1).
    pub(super) fn xgetbv_xsetbv(&mut self, m: u32) {
        let g = self.g;
        let bad = PREFIX_LOCK | PREFIX_DATA | PREFIX_REPZ | PREFIX_REPNZ;
        if !self.d.feat.xsave || self.d.prefix & bad != 0 {
            self.gen_illegal_opcode();
            return;
        }
        let ecx = self.trunc32(g.regs[R_ECX]);
        if m == 0xd0 {
            let v = self.new64();
            self.env_call(&vec::XGETBV, Some(v.into()), &[ecx.into()]);
            let f = self.f();
            f.gen_ext32u_i64(g.regs[R_EAX], v);
            f.gen_shri_i64(g.regs[R_EDX], v, 32);
            return;
        }
        if !self.check_cpl0() {
            return;
        }
        let v = self.new64();
        let f = self.f();
        f.gen_concat32_i64(v, g.regs[R_EAX], g.regs[R_EDX]);
        self.env_call(&vec::XSETBV, None, &[ecx.into(), v.into()]);
        // End the block: XCR0 changes HF_AVX_EN.
        self.b.is_jmp = DISAS_EOB_NEXT;
    }

    /// RDFSBASE, RDGSBASE, WRFSBASE and WRGSBASE: F3 0F AE /0 to /3 with mod 3.
    pub(super) fn fsgsbase(&mut self, m: u32) {
        let g = self.g;
        if !self.d.feat.fsgsbase || !self.code64() {
            self.gen_illegal_opcode();
            return;
        }
        let ot = self.ot_y();
        let rm = (m & 7) as usize | self.d.rex_b;
        let base = g.seg_base[if m & 8 != 0 { R_GS } else { R_FS }];
        // Test CR4 at run time, as QEMU does to keep the hflags bits.
        let bit = self.c32(CR4_FSGSBASE_MASK as i32);
        self.env_call(&vec::CR4_TESTBIT, None, &[bit.into()]);
        if m & 0x10 == 0 {
            self.f().gen_mov_i64(g.t0, base);
            self.mov_reg_v(ot, rm, g.t0);
        } else {
            self.mov_v_reg(ot, g.t0, rm);
            self.f().gen_mov_i64(base, g.t0);
        }
    }

    /// LDMXCSR (0F AE /2) and STMXCSR (0F AE /3) with a memory operand and no prefix.
    pub(super) fn ldst_mxcsr(&mut self, m: u32) -> R {
        let g = self.g;
        let flags = self.d.flags;
        if !self.d.feat.sse || flags & HF_OSFXSR_MASK == 0 {
            self.gen_illegal_opcode();
            return Ok(());
        }
        if flags & HF_TS_MASK != 0 {
            self.gen_exception(EXCP07_PREX);
            return Ok(());
        }
        if flags & HF_EM_MASK != 0 {
            self.gen_illegal_opcode();
            return Ok(());
        }
        self.gen_lea_modrm(m)?;
        if (m >> 3) & 7 == 2 {
            self.ld_v(OT32, g.t0, g.a0);
            self.st_env32(g.t0, MXCSR);
        } else {
            let t = self.ld_env32(MXCSR);
            self.st_v(OT32, t, g.a0);
        }
        Ok(())
    }

    /// MOVNTI (0F C3): a plain store, as QEMU does.
    pub(super) fn movnti(&mut self) -> R {
        let g = self.g;
        let (m, md, reg, _) = self.modrm()?;
        if md == 3 || !self.d.feat.sse2 {
            self.gen_illegal_opcode();
            return Ok(());
        }
        let ot = self.ot_y();
        self.gen_lea_modrm(m)?;
        self.mov_v_reg(ot, g.t0, reg);
        self.st_v(ot, g.t0, g.a0);
        Ok(())
    }
}
