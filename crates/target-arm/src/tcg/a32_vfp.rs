// SPDX-License-Identifier: GPL-2.0-or-later

//! The AArch32 VFP helpers: the scalar `vfp_*` helpers of QEMU's `vfp_helper.c` that the
//! A32 and T32 translator calls, folded into one helper that takes the operation in a
//! descriptor.
//!
//! The values are the register contents: a half or single precision value in the low bits,
//! an integer as the 32 bits of an S register or the 64 bits of a D register. Half
//! precision arithmetic runs on `FPST_A32_F16` and everything else on `FPST_A32`, which are
//! `fp_status[1]` and `fp_status[0]` here as FPCR sets them the same way for both
//! instruction sets.

use ruvm_jit_core::HelperType::{I32, I64, Ptr};
use ruvm_jit_interp::{HelperEnv, Unwind};
use ruvm_softfloat::{Float16, Float32, Float64, FloatRelation, FloatStatus, RoundMode, flags};

use super::helpers::{Def, def};
use super::vec_helper::RMODE_FPCR;
use super::vfp::{FPCR_AHP, Fp, cvt, load_status, rint, store_status};
use crate::cpu::{FPCR, FPSR};

/// The operations of `a32_vfp`.
#[allow(missing_docs)]
pub(crate) mod op {
    pub(crate) const ADD: u32 = 0;
    pub(crate) const SUB: u32 = 1;
    pub(crate) const MUL: u32 = 2;
    pub(crate) const DIV: u32 = 3;
    pub(crate) const MAXNM: u32 = 4;
    pub(crate) const MINNM: u32 = 5;
    /// Fused `n * m + d`, `vfp_muladd`.
    pub(crate) const MULADD: u32 = 6;
    pub(crate) const SQRT: u32 = 7;
    /// `rints`: round to integral without raising inexact.
    pub(crate) const RINT: u32 = 8;
    /// `rints_exact`.
    pub(crate) const RINTX: u32 = 9;
    /// Float to signed integer, the immediate being the fraction bits.
    pub(crate) const TOSI: u32 = 10;
    pub(crate) const TOUI: u32 = 11;
    /// Signed integer to float, the immediate being the fraction bits.
    pub(crate) const SITO: u32 = 12;
    pub(crate) const UITO: u32 = 13;
    /// Precision conversion to the size in the immediate.
    pub(crate) const CVT: u32 = 14;
    /// `vfp_cmp`: set FPSCR.NZCV.
    pub(crate) const CMP: u32 = 15;
    /// `vfp_cmpe`: as CMP, but a quiet NaN raises invalid too.
    pub(crate) const CMPE: u32 = 16;
}

/// The descriptor of `a32_vfp`: operation `op` on elements of `esz` (1 to 3) in rounding
/// mode `rmode` ([`RMODE_FPCR`] for the mode in FPSCR), with immediate `imm`. `i16` says
/// the integer of a conversion is 16 bits.
pub(crate) fn desc(op: u32, esz: u32, rmode: u32, imm: u32, i16: bool) -> i32 {
    (op | esz << 8 | rmode << 12 | (imm & 0xff) << 16 | u32::from(i16) << 24) as i32
}

def!(VFP, "a32_vfp", 0, I64, [Ptr, I64, I64, I64, I32], h_vfp);

/// The helpers of this module.
pub(crate) const ALL: &[Def] = &[VFP];

/// `a32_vfp(env, n, m, d, desc)`: operation `op` of `desc` on `n`, `m` and the
/// accumulator `d`.
fn h_vfp(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    let (n, m, d, desc) = (a[1], a[2], a[3], a[4] as u32);
    let op = desc & 0xff;
    let esz = (desc >> 8) & 3;
    let rmode = (desc >> 12) & 15;
    let imm = (desc >> 16) & 0xff;
    let i16 = desc & (1 << 24) != 0;
    let env = &mut *h.env;
    let idx = usize::from(esz == 1 && op != op::CVT);
    let mut s = load_status(env, idx);
    let saved = s.rounding_mode;
    if rmode != RMODE_FPCR {
        s.rounding_mode = RoundMode::from_u8(rmode as u8).unwrap_or(saved);
    }
    let r = if op == op::CVT {
        let fpcr = u32::from_le_bytes(env[FPCR..FPCR + 4].try_into().unwrap());
        cvt(n, esz, imm, fpcr & FPCR_AHP == 0, &mut s)
    } else {
        let v = Vals { op, n, m, d, imm, i16 };
        match esz {
            1 => v.run::<Float16>(&mut s),
            2 => v.run::<Float32>(&mut s),
            _ => v.run::<Float64>(&mut s),
        }
    };
    s.rounding_mode = saved;
    store_status(env, idx, &s);
    if op == op::CMP || op == op::CMPE {
        let fpsr = u32::from_le_bytes(env[FPSR..FPSR + 4].try_into().unwrap());
        let fpsr = (fpsr & 0x0fff_ffff) | (r as u32) << 28;
        env[FPSR..FPSR + 4].copy_from_slice(&fpsr.to_le_bytes());
        return Ok(0);
    }
    Ok(u128::from(r))
}

/// The arguments of one operation.
struct Vals {
    op: u32,
    n: u64,
    m: u64,
    d: u64,
    imm: u32,
    i16: bool,
}

impl Vals {
    /// The result, or NZCV in bits 3 to 0 for a comparison.
    fn run<F: Fp>(&self, s: &mut FloatStatus) -> u64 {
        let (n, m) = (F::from_bits(self.n), F::from_bits(self.m));
        let shift = self.imm as i32;
        let r = match self.op {
            op::ADD => n.add(m, s),
            op::SUB => n.sub(m, s),
            op::MUL => n.mul(m, s),
            op::DIV => n.div(m, s),
            op::MAXNM => n.maxnum(m, s),
            op::MINNM => n.minnum(m, s),
            op::MULADD => n.muladd(m, F::from_bits(self.d), 0, s),
            op::SQRT => n.sqrt(s),
            op::RINT | op::RINTX => rint(n, self.op == op::RINTX, s),
            op::TOSI | op::TOUI => {
                // VFP_CONV_FLOAT_FIX_ROUND: a NaN raises invalid and gives 0. A signed
                // result is sign extended to 64 bits, as the C helpers return it.
                if n.is_any_nan() {
                    s.raise(flags::INVALID);
                    return 0;
                }
                let bits = if self.i16 { 16 } else { 32 };
                let rm = s.rounding_mode;
                return n.to_int(bits, self.op == op::TOSI, rm, shift, s);
            }
            op::SITO | op::UITO => {
                let x = match (self.i16, self.op == op::SITO) {
                    (true, true) => self.n as i16 as i64 as u64,
                    (true, false) => self.n & 0xffff,
                    (false, true) => self.n as i32 as i64 as u64,
                    (false, false) => self.n & 0xffff_ffff,
                };
                F::from_int(x, self.op == op::SITO, -shift, s)
            }
            _ => {
                let rel = if self.op == op::CMPE { n.compare(m, s) } else { n.compare_quiet(m, s) };
                return match rel {
                    FloatRelation::Equal => 0x6,
                    FloatRelation::Less => 0x8,
                    FloatRelation::Greater => 0x2,
                    FloatRelation::Unordered => 0x3,
                };
            }
        };
        r.bits()
    }
}

#[cfg(test)]
mod tests {
    use super::{Vals, op};
    use ruvm_softfloat::{Float32, Float64, FloatStatus};

    fn run32(op: u32, n: u64, m: u64, imm: u32, i16: bool) -> u64 {
        let mut s = FloatStatus::arm();
        Vals { op, n, m, d: 0, imm, i16 }.run::<Float32>(&mut s)
    }

    #[test]
    fn conversions() {
        // -1.5 to a signed integer rounds to even, -2, sign extended.
        assert_eq!(run32(op::TOSI, 0xbfc0_0000, 0, 0, false), (-2i64) as u64);
        // A NaN converts to 0.
        assert_eq!(run32(op::TOUI, 0x7fc0_0000, 0, 0, false), 0);
        // 0x8000 as a signed 16 bit fixed point number with 1 fraction bit is -16384.0.
        assert_eq!(run32(op::SITO, 0x1234_8000, 0, 1, true), 0xc680_0000);
        // 3.0 to unsigned 16 bit fixed point with 4 fraction bits.
        assert_eq!(run32(op::TOUI, 0x4040_0000, 0, 4, true), 48);
    }

    #[test]
    fn compare() {
        let mut s = FloatStatus::arm();
        let v = |n, m| Vals { op: op::CMP, n, m, d: 0, imm: 0, i16: false };
        // 1.0 < 2.0, equal, and a NaN.
        assert_eq!(v(0x3ff0_0000_0000_0000, 0x4000_0000_0000_0000).run::<Float64>(&mut s), 0x8);
        assert_eq!(run32(op::CMP, 0x3f80_0000, 0x3f80_0000, 0, false), 0x6);
        assert_eq!(run32(op::CMP, 0x7fc0_0000, 0x3f80_0000, 0, false), 0x3);
    }
}
