// SPDX-License-Identifier: GPL-2.0-or-later

//! Lazy condition codes: the `CCOp` values and the port of `cc_helper.c` and
//! `cc_helper_template.h.inc`.
//!
//! The flags are not computed when an instruction runs. It records its operands in `cc_dst`,
//! `cc_src` and `cc_src2` and the kind of operation in `cc_op`, and the flags are computed from
//! those only when something reads them.

use super::env::{CC_A, CC_C, CC_O, CC_P, CC_S, CC_Z};

/// Flags are in `cc_src` in EFLAGS layout.
pub const CC_OP_EFLAGS: u32 = 0;
/// CF in `cc_dst`, the rest in `cc_src`.
pub const CC_OP_ADCX: u32 = 1;
/// OF in `cc_src2`, the rest in `cc_src`.
pub const CC_OP_ADOX: u32 = 2;
/// CF in `cc_dst`, OF in `cc_src2`, the rest in `cc_src`.
pub const CC_OP_ADCOX: u32 = 3;
/// `CC_OP_MULB`; add the size (0 to 3) for the other widths.
pub const CC_OP_MULB: u32 = 4;
/// `CC_OP_ADDB`.
pub const CC_OP_ADDB: u32 = 8;
/// `CC_OP_ADCB`.
pub const CC_OP_ADCB: u32 = 12;
/// `CC_OP_SUBB`.
pub const CC_OP_SUBB: u32 = 16;
/// `CC_OP_SBBB`.
pub const CC_OP_SBBB: u32 = 20;
/// `CC_OP_LOGICB`.
pub const CC_OP_LOGICB: u32 = 24;
/// `CC_OP_INCB`.
pub const CC_OP_INCB: u32 = 28;
/// `CC_OP_DECB`.
pub const CC_OP_DECB: u32 = 32;
/// `CC_OP_SHLB`.
pub const CC_OP_SHLB: u32 = 36;
/// `CC_OP_SARB`.
pub const CC_OP_SARB: u32 = 40;
/// `CC_OP_BMILGB`.
pub const CC_OP_BMILGB: u32 = 44;
/// `CC_OP_BLSIB`.
pub const CC_OP_BLSIB: u32 = 48;
/// `CC_OP_POPCNTB`; only `CC_OP_POPCNT` (size 3) is used.
pub const CC_OP_POPCNTB: u32 = 52;
/// `CC_OP_POPCNT`: Z from `cc_dst`, every other flag clear.
pub const CC_OP_POPCNT: u32 = 55;
/// `CC_OP_SBB_SELFB`.
pub const CC_OP_SBB_SELFB: u32 = 56;
/// `CC_OP_SBB_SELF`: `sbb reg, reg`; `cc_dst` holds `-CF`.
pub const CC_OP_SBB_SELF: u32 = 59;
/// The value is only known at run time and is in `env->cc_op`.
pub const CC_OP_DYNAMIC: u32 = 60;
/// Nothing is live.
pub const CC_OP_CLR: u32 = 61;
/// The number of values.
pub const CC_OP_NB: u32 = 62;

/// `cc_op_size()`.
pub const fn cc_op_size(op: u32) -> u32 {
    op & 3
}

/// `CC_OP_HAS_EFLAGS()`.
pub const fn cc_op_has_eflags(op: u32) -> bool {
    op <= CC_OP_ADCOX
}

/// The `cc_*` values an operation reads, `cc_op_live`.
pub const USES_CC_DST: u8 = 1;
/// `cc_src` is live.
pub const USES_CC_SRC: u8 = 2;
/// `cc_src2` is live.
pub const USES_CC_SRC2: u8 = 4;
/// `cc_srcT` is live.
pub const USES_CC_SRCT: u8 = 8;

/// `cc_op_live()`.
pub const fn cc_op_live(op: u32) -> u8 {
    match op {
        CC_OP_EFLAGS => USES_CC_SRC,
        CC_OP_ADCX => USES_CC_DST | USES_CC_SRC,
        CC_OP_ADOX => USES_CC_SRC | USES_CC_SRC2,
        CC_OP_ADCOX => USES_CC_DST | USES_CC_SRC | USES_CC_SRC2,
        CC_OP_DYNAMIC => USES_CC_DST | USES_CC_SRC | USES_CC_SRC2,
        CC_OP_CLR => 0,
        CC_OP_POPCNTB..=CC_OP_POPCNT => USES_CC_DST,
        _ => match op & !3 {
            CC_OP_MULB | CC_OP_ADDB | CC_OP_SHLB | CC_OP_SARB | CC_OP_BMILGB | CC_OP_BLSIB => {
                USES_CC_DST | USES_CC_SRC
            }
            CC_OP_ADCB | CC_OP_SBBB => USES_CC_DST | USES_CC_SRC | USES_CC_SRC2,
            CC_OP_SUBB => USES_CC_DST | USES_CC_SRC | USES_CC_SRCT,
            CC_OP_LOGICB => USES_CC_DST,
            CC_OP_INCB | CC_OP_DECB => USES_CC_DST | USES_CC_SRC,
            CC_OP_SBB_SELFB => USES_CC_DST,
            _ => USES_CC_DST | USES_CC_SRC | USES_CC_SRC2,
        },
    }
}

/// `parity_table[]`: `CC_P` when the byte has an even number of set bits.
pub fn parity(b: u64) -> u32 {
    if (b as u8).count_ones() % 2 == 0 { CC_P } else { 0 }
}

fn bits(size: u32) -> u32 {
    8 << size
}

fn mask(size: u32) -> u64 {
    if size == 3 { u64::MAX } else { (1u64 << bits(size)) - 1 }
}

fn sign(size: u32) -> u64 {
    1u64 << (bits(size) - 1)
}

/// Z and S from a result.
fn zs(dst: u64, size: u32) -> u32 {
    let d = dst & mask(size);
    let mut f = 0;
    if d == 0 {
        f |= CC_Z;
    }
    if d & sign(size) != 0 {
        f |= CC_S;
    }
    f
}

/// `compute_aco_cout()`: A, C and O from a vector of carries out of each bit.
fn compute_aco_cout(carries: u64, size: u32) -> u32 {
    let b = bits(size);
    let mut f = 0;
    if carries & 8 != 0 {
        f |= CC_A;
    }
    if (carries >> (b - 1)) & 1 != 0 {
        f |= CC_C;
    }
    if ((carries >> (b - 1)) ^ (carries >> (b - 2))) & 1 != 0 {
        f |= CC_O;
    }
    f
}

/// `compute_aco_add()`.
fn compute_aco_add(dst: u64, src1: u64, src2: u64, size: u32) -> u32 {
    let carries = (src1 & src2) | ((src1 | src2) & !dst);
    compute_aco_cout(carries, size)
}

/// `compute_aco_sub()`.
fn compute_aco_sub(dst: u64, src1: u64, src2: u64, size: u32) -> u32 {
    let borrows = (!src1 & src2) | ((!src1 | src2) & dst);
    compute_aco_cout(borrows, size)
}

fn compute_all_add(dst: u64, src2: u64, size: u32) -> u32 {
    let src1 = dst.wrapping_sub(src2);
    zs(dst, size) | parity(dst) | compute_aco_add(dst, src1, src2, size)
}

fn compute_all_adc(dst: u64, src2: u64, src3: u64, size: u32) -> u32 {
    let src1 = dst.wrapping_sub(src2).wrapping_sub(src3);
    let d = dst & mask(size);
    let s1 = src1 & mask(size);
    // The carry is computed separately: with a carry in, dst == src1 also means a carry out.
    let cf = if src3 != 0 { d <= s1 } else { d < s1 };
    let mut f = zs(dst, size) | parity(dst);
    f |= compute_aco_add(dst, src1, src2, size) & !CC_C;
    if cf {
        f |= CC_C;
    }
    f
}

fn compute_all_sub(dst: u64, src2: u64, size: u32) -> u32 {
    let src1 = dst.wrapping_add(src2);
    zs(dst, size) | parity(dst) | compute_aco_sub(dst, src1, src2, size)
}

fn compute_all_sbb(dst: u64, src2: u64, src3: u64, size: u32) -> u32 {
    let src1 = dst.wrapping_add(src2).wrapping_add(src3);
    let s1 = src1 & mask(size);
    let s2 = src2 & mask(size);
    let cf = if src3 != 0 { s1 <= s2 } else { s1 < s2 };
    let mut f = zs(dst, size) | parity(dst);
    f |= compute_aco_sub(dst, src1, src2, size) & !CC_C;
    if cf {
        f |= CC_C;
    }
    f
}

fn compute_all_logic(dst: u64, size: u32) -> u32 {
    zs(dst, size) | parity(dst)
}

fn compute_all_inc(dst: u64, src1: u64, size: u32) -> u32 {
    let src2 = 1;
    let src1d = dst.wrapping_sub(1);
    let mut f = src1 as u32 & CC_C;
    f |= zs(dst, size) | parity(dst);
    f |= compute_aco_add(dst, src1d, src2, size) & (CC_A | CC_O);
    f
}

fn compute_all_dec(dst: u64, src1: u64, size: u32) -> u32 {
    let src2 = 1;
    let src1d = dst.wrapping_add(1);
    let mut f = src1 as u32 & CC_C;
    f |= zs(dst, size) | parity(dst);
    f |= compute_aco_sub(dst, src1d, src2, size) & (CC_A | CC_O);
    f
}

fn compute_all_shl(dst: u64, src1: u64, size: u32) -> u32 {
    let mut f = zs(dst, size) | parity(dst);
    if (src1 >> (bits(size) - 1)) & 1 != 0 {
        f |= CC_C;
    }
    // OF is only defined for shifts by one: the top bit changed.
    if ((src1 ^ dst) >> (bits(size) - 1)) & 1 != 0 {
        f |= CC_O;
    }
    f
}

fn compute_all_sar(dst: u64, src1: u64, size: u32) -> u32 {
    let mut f = zs(dst, size) | parity(dst);
    f |= src1 as u32 & CC_C;
    // OF is only defined for shifts by one, where it is zero for SAR and the old sign for SHR.
    if ((src1 ^ dst) >> (bits(size) - 1)) & 1 != 0 {
        f |= CC_O;
    }
    f
}

fn compute_all_mul(dst: u64, src1: u64, size: u32) -> u32 {
    // compute_aco_mul(): CF and OF are set when the high half is not just the extension.
    let mut f = zs(dst, size) | parity(dst);
    if src1 != 0 {
        f |= CC_C | CC_O;
    }
    f
}

fn compute_all_bmilg(dst: u64, src1: u64, size: u32) -> u32 {
    // PF and AF are undefined and left clear.
    let mut f = zs(dst, size);
    if src1 == 0 {
        f |= CC_C;
    }
    f
}

fn compute_all_blsi(dst: u64, src1: u64, size: u32) -> u32 {
    // PF and AF are undefined and left clear.
    let mut f = zs(dst, size);
    if src1 != 0 {
        f |= CC_C;
    }
    f
}

/// `helper_cc_compute_all()`.
pub fn compute_all(dst: u64, src1: u64, src2: u64, op: u32) -> u32 {
    let size = cc_op_size(op);
    match op {
        CC_OP_EFLAGS => src1 as u32,
        CC_OP_CLR => CC_Z | CC_P,
        CC_OP_POPCNT => {
            if dst != 0 {
                0
            } else {
                CC_Z
            }
        }
        CC_OP_ADCX => (src1 as u32 & !CC_C) | (dst as u32 & CC_C),
        CC_OP_ADOX => (src1 as u32 & !CC_O) | ((src2 as u32 & 1) * CC_O),
        CC_OP_ADCOX => {
            (src1 as u32 & !(CC_C | CC_O)) | (dst as u32 & CC_C) | ((src2 as u32 & 1) * CC_O)
        }
        _ => match op & !3 {
            CC_OP_MULB => compute_all_mul(dst, src1, size),
            CC_OP_ADDB => compute_all_add(dst, src1, size),
            CC_OP_ADCB => compute_all_adc(dst, src1, src2, size),
            CC_OP_SUBB => compute_all_sub(dst, src1, size),
            CC_OP_SBBB => compute_all_sbb(dst, src1, src2, size),
            CC_OP_LOGICB => compute_all_logic(dst, size),
            CC_OP_INCB => compute_all_inc(dst, src1, size),
            CC_OP_DECB => compute_all_dec(dst, src1, size),
            CC_OP_SHLB => compute_all_shl(dst, src1, size),
            CC_OP_SARB => compute_all_sar(dst, src1, size),
            CC_OP_BMILGB => compute_all_bmilg(dst, src1, size),
            CC_OP_BLSIB => compute_all_blsi(dst, src1, size),
            // dst is either all zeros (--Z-P-) or all ones (-S-APC).
            CC_OP_SBB_SELFB => (dst as u32 & (CC_Z | CC_A | CC_C | CC_S)) ^ (CC_P | CC_Z),
            _ => panic!("unknown cc_op {op}"),
        },
    }
}

/// `helper_cc_compute_c()`: only CF, as 0 or 1.
pub fn compute_c(dst: u64, src1: u64, src2: u64, op: u32) -> u64 {
    match op {
        CC_OP_CLR | CC_OP_POPCNTB..=CC_OP_POPCNT => 0,
        CC_OP_EFLAGS | CC_OP_ADOX => src1 & 1,
        CC_OP_ADCX | CC_OP_ADCOX => dst,
        _ => match op & !3 {
            CC_OP_LOGICB => 0,
            CC_OP_SARB | CC_OP_INCB | CC_OP_DECB => src1 & 1,
            CC_OP_MULB | CC_OP_BLSIB => u64::from(src1 != 0),
            CC_OP_SBB_SELFB => dst & 1,
            CC_OP_BMILGB => u64::from(src1 == 0),
            _ => u64::from(compute_all(dst, src1, src2, op) & CC_C != 0),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_flags() {
        // 0x7f + 1 = 0x80: overflow, sign, adjust, no carry.
        let f = compute_all(0x80, 1, 0, CC_OP_ADDB);
        assert_eq!(f, CC_O | CC_S | CC_A);
        // 0xff + 1 = 0x00: carry, zero, adjust, parity.
        let f = compute_all(0x100, 1, 0, CC_OP_ADDB);
        assert_eq!(f, CC_C | CC_Z | CC_A | CC_P);
        assert_eq!(compute_c(0x100, 1, 0, CC_OP_ADDB), 1);
    }

    #[test]
    fn sub_flags() {
        // 0 - 1 = 0xffffffff: carry (borrow), sign, adjust, parity.
        let f = compute_all(0xffff_ffff, 1, 0, CC_OP_SUBL);
        assert_eq!(f, CC_C | CC_S | CC_A | CC_P);
        // 0x80000000 - 1: overflow.
        let f = compute_all(0x7fff_ffff, 1, 0, CC_OP_SUBL);
        assert_eq!(f & (CC_O | CC_C), CC_O);
    }

    const CC_OP_SUBL: u32 = CC_OP_SUBB + 2;
}
