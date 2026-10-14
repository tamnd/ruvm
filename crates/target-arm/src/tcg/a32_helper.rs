// SPDX-License-Identifier: GPL-2.0-or-later

//! The AArch32 integer helpers: the parts of QEMU's `op_helper.c`, `arith_helper.c`,
//! `op_addsub.h` and `helper.c` that the A32 and T32 translator calls.
//!
//! The helpers that set the Q flag or the GE bits take `env` and write `QF` or `GE` there;
//! neither is a TCG global. The flag setting shifts write `CF`, which is one, so they are
//! declared as touching the globals.

use ruvm_jit_core::HelperType;
use ruvm_jit_core::types::call_flags::{NO_RWG, NO_RWG_SE};
use ruvm_jit_interp::{HelperEnv, Unwind};

use super::helpers::{Def, crc_update, def, run};
use super::{arm_of, commit, vfp};
use crate::cpu::{CF, CPSR_EXEC, CpuArmState, GE, QF};

use HelperType::{I32, Ptr, Void};

type H = Result<u128, Unwind>;

/// `FPSCR_FPSR_MASK`: the FPSCR bits kept in FPSR, NZCV, QC and the cumulative exceptions.
const FPSCR_FPSR_MASK: u32 = 0xf800_009f;
/// `FPSCR_FPCR_MASK`: the FPSCR bits kept in FPCR.
const FPSCR_FPCR_MASK: u32 = 0x07ff_9f00;

fn st32(env: &mut [u8], off: usize, v: u32) {
    env[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

/// Set the Q flag.
fn set_q(env: &mut [u8]) {
    st32(env, QF, 1);
}

fn ret(v: u32) -> H {
    Ok(u128::from(v))
}

// Shifts by a register that set the carry flag.

/// `HELPER(shl_cc)`.
fn h_shl_cc(h: &mut HelperEnv<'_>, a: &[u64]) -> H {
    let (x, shift) = (a[1] as u32, a[2] as u32 & 0xff);
    if shift >= 32 {
        st32(h.env, CF, if shift == 32 { x & 1 } else { 0 });
        ret(0)
    } else if shift != 0 {
        st32(h.env, CF, (x >> (32 - shift)) & 1);
        ret(x << shift)
    } else {
        ret(x)
    }
}

/// `HELPER(shr_cc)`.
fn h_shr_cc(h: &mut HelperEnv<'_>, a: &[u64]) -> H {
    let (x, shift) = (a[1] as u32, a[2] as u32 & 0xff);
    if shift >= 32 {
        st32(h.env, CF, if shift == 32 { x >> 31 } else { 0 });
        ret(0)
    } else if shift != 0 {
        st32(h.env, CF, (x >> (shift - 1)) & 1);
        ret(x >> shift)
    } else {
        ret(x)
    }
}

/// `HELPER(sar_cc)`.
fn h_sar_cc(h: &mut HelperEnv<'_>, a: &[u64]) -> H {
    let (x, shift) = (a[1] as u32, a[2] as u32 & 0xff);
    if shift >= 32 {
        st32(h.env, CF, x >> 31);
        ret(((x as i32) >> 31) as u32)
    } else if shift != 0 {
        st32(h.env, CF, (x >> (shift - 1)) & 1);
        ret(((x as i32) >> shift) as u32)
    } else {
        ret(x)
    }
}

/// `HELPER(ror_cc)`.
fn h_ror_cc(h: &mut HelperEnv<'_>, a: &[u64]) -> H {
    let (x, shift1) = (a[1] as u32, a[2] as u32 & 0xff);
    let shift = shift1 & 0x1f;
    if shift == 0 {
        if shift1 != 0 {
            st32(h.env, CF, x >> 31);
        }
        ret(x)
    } else {
        st32(h.env, CF, (x >> (shift - 1)) & 1);
        ret(x.rotate_right(shift))
    }
}

// Saturation.

/// `HELPER(add_setq)`: the wrapping sum, setting Q on signed overflow.
fn h_add_setq(h: &mut HelperEnv<'_>, a: &[u64]) -> H {
    let (r, o) = (a[1] as i32).overflowing_add(a[2] as i32);
    if o {
        set_q(h.env);
    }
    ret(r as u32)
}

/// `HELPER(add_saturate)`.
fn h_add_saturate(h: &mut HelperEnv<'_>, a: &[u64]) -> H {
    let (x, y) = (a[1] as i32, a[2] as i32);
    match x.checked_add(y) {
        Some(r) => ret(r as u32),
        None => {
            set_q(h.env);
            ret(x.saturating_add(y) as u32)
        }
    }
}

/// `HELPER(sub_saturate)`.
fn h_sub_saturate(h: &mut HelperEnv<'_>, a: &[u64]) -> H {
    let (x, y) = (a[1] as i32, a[2] as i32);
    match x.checked_sub(y) {
        Some(r) => ret(r as u32),
        None => {
            set_q(h.env);
            ret(x.saturating_sub(y) as u32)
        }
    }
}

/// `do_ssat()`: saturate `val` to a signed `shift + 1` bit value.
fn do_ssat(env: &mut [u8], val: i32, shift: u32) -> u32 {
    let top = val >> shift;
    let mask = (1u32 << shift) - 1;
    if top > 0 {
        set_q(env);
        mask
    } else if top < -1 {
        set_q(env);
        !mask
    } else {
        val as u32
    }
}

/// `do_usat()`: saturate `val` to an unsigned `shift` bit value.
fn do_usat(env: &mut [u8], val: i32, shift: u32) -> u32 {
    let max = ((1u64 << shift) - 1) as u32;
    if val < 0 {
        set_q(env);
        0
    } else if val as u32 > max {
        set_q(env);
        max
    } else {
        val as u32
    }
}

/// `HELPER(ssat)`.
fn h_ssat(h: &mut HelperEnv<'_>, a: &[u64]) -> H {
    ret(do_ssat(h.env, a[1] as i32, a[2] as u32))
}

/// `HELPER(usat)`.
fn h_usat(h: &mut HelperEnv<'_>, a: &[u64]) -> H {
    ret(do_usat(h.env, a[1] as i32, a[2] as u32))
}

/// `HELPER(ssat16)`.
fn h_ssat16(h: &mut HelperEnv<'_>, a: &[u64]) -> H {
    let (x, shift) = (a[1] as u32, a[2] as u32);
    let lo = do_ssat(h.env, i32::from(x as i16), shift) & 0xffff;
    let hi = do_ssat(h.env, (x as i32) >> 16, shift);
    ret(lo | (hi << 16))
}

/// `HELPER(usat16)`.
fn h_usat16(h: &mut HelperEnv<'_>, a: &[u64]) -> H {
    let (x, shift) = (a[1] as u32, a[2] as u32);
    let lo = do_usat(h.env, i32::from(x as i16), shift) & 0xffff;
    let hi = do_usat(h.env, (x as i32) >> 16, shift);
    ret(lo | (hi << 16))
}

// Bytes and halfwords.

/// `HELPER(sxtb16)`.
fn h_sxtb16(_h: &mut HelperEnv<'_>, a: &[u64]) -> H {
    let x = a[0] as u32;
    ret(u32::from(i16::from(x as i8) as u16) | (u32::from(i16::from((x >> 16) as i8) as u16) << 16))
}

/// `HELPER(uxtb16)`.
fn h_uxtb16(_h: &mut HelperEnv<'_>, a: &[u64]) -> H {
    ret(a[0] as u32 & 0x00ff_00ff)
}

/// `HELPER(sel_flags)`: each byte from `a` where its GE bit is set, else from `b`.
fn h_sel_flags(_h: &mut HelperEnv<'_>, a: &[u64]) -> H {
    let (flags, x, y) = (a[0] as u32, a[1] as u32, a[2] as u32);
    let mask = (0..4).filter(|i| flags & (1 << i) != 0).fold(0u32, |m, i| m | (0xff << (8 * i)));
    ret((x & mask) | (y & !mask))
}

/// `HELPER(usad8)`: the sum of the absolute differences of the bytes.
fn h_usad8(_h: &mut HelperEnv<'_>, a: &[u64]) -> H {
    let (x, y) = ((a[0] as u32).to_le_bytes(), (a[1] as u32).to_le_bytes());
    ret(x.iter().zip(y).map(|(&p, q)| u32::from(p.abs_diff(q))).sum())
}

/// `HELPER(rbit)`.
fn h_rbit(_h: &mut HelperEnv<'_>, a: &[u64]) -> H {
    ret((a[0] as u32).reverse_bits())
}

/// `HELPER(sdiv)`: division by zero gives zero.
fn h_sdiv(_h: &mut HelperEnv<'_>, a: &[u64]) -> H {
    let (num, den) = (a[1] as i32, a[2] as i32);
    ret(if den == 0 { 0 } else { num.wrapping_div(den) as u32 })
}

/// `HELPER(udiv)`: division by zero gives zero.
fn h_udiv(_h: &mut HelperEnv<'_>, a: &[u64]) -> H {
    let (num, den) = (a[1] as u32, a[2] as u32);
    ret(num.checked_div(den).unwrap_or(0))
}

/// `HELPER(crc32)`: the CRC-32 of the low `a[2]` bytes of `a[1]`.
fn h_crc32(_h: &mut HelperEnv<'_>, a: &[u64]) -> H {
    ret(crc_update(0xedb8_8320, a[0] as u32, a[1] & 0xffff_ffff, a[2] as u32))
}

/// `HELPER(crc32c)`.
fn h_crc32c(_h: &mut HelperEnv<'_>, a: &[u64]) -> H {
    ret(crc_update(0x82f6_3b78, a[0] as u32, a[1] & 0xffff_ffff, a[2] as u32))
}

/// `HELPER(cpsr_read)`: the CPSR without the execution state bits.
fn h_cpsr_read(h: &mut HelperEnv<'_>, _a: &[u64]) -> H {
    ret(CpuArmState::load(h.env).cpsr_read() & !CPSR_EXEC)
}

/// `HELPER(cpsr_write)`: write the CPSR bits in `mask`, then rebuild the TB flags.
fn h_cpsr_write(h: &mut HelperEnv<'_>, a: &[u64]) -> H {
    let (val, mask) = (a[1] as u32, a[2] as u32);
    run(h, |cpu| {
        let mut st = CpuArmState::load(cpu.env);
        st.cpsr_write(val, mask);
        commit(cpu, &mut st);
        Ok(0)
    })
}

/// `HELPER(vfp_get_fpscr)`.
fn h_vfp_get_fpscr(h: &mut HelperEnv<'_>, _a: &[u64]) -> H {
    let st = CpuArmState::load(h.env);
    ret((vfp::get_fpcr(&st) & FPSCR_FPCR_MASK) | (vfp::get_fpsr(&st) & FPSCR_FPSR_MASK))
}

/// `HELPER(vfp_set_fpscr)`.
fn h_vfp_set_fpscr(h: &mut HelperEnv<'_>, a: &[u64]) -> H {
    let val = a[1] as u32;
    run(h, |cpu| {
        let ops = cpu.ops();
        let feat = *arm_of(&ops).features();
        let mut st = CpuArmState::load(cpu.env);
        vfp::set_fpcr(&mut st, val & FPSCR_FPCR_MASK, &feat);
        vfp::set_fpsr(&mut st, val & FPSCR_FPSR_MASK);
        commit(cpu, &mut st);
        Ok(0)
    })
}

// The parallel additions and subtractions of `op_addsub.h`.

/// What a parallel add or subtract does with each lane's result.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Par {
    /// Signed, setting GE where the result is not negative.
    S,
    /// Signed saturating.
    Q,
    /// Signed halving.
    Sh,
    /// Unsigned, setting GE where an addition carries or a subtraction does not borrow.
    U,
    /// Unsigned saturating.
    Uq,
    /// Unsigned halving.
    Uh,
}

/// The lane operations: `ADD16`, `ASX` (`addsubx`), `SAX` (`subaddx`), `SUB16`, `ADD8`,
/// `SUB8`.
#[derive(Clone, Copy)]
enum Lanes {
    Add16,
    Asx,
    Sax,
    Sub16,
    Add8,
    Sub8,
}

/// One lane: `x` plus or minus `y`, `bits` wide. Returns the lane result and its GE bit.
fn par_lane(kind: Par, add: bool, x: u32, y: u32, bits: u32) -> (u32, bool) {
    let mask = (1u32 << bits) - 1;
    let signed = matches!(kind, Par::S | Par::Q | Par::Sh);
    let (x, y) = if signed {
        let sh = 32 - bits;
        (((x << sh) as i32) >> sh, ((y << sh) as i32) >> sh)
    } else {
        ((x & mask) as i32, (y & mask) as i32)
    };
    let r = if add { x + y } else { x - y };
    let (min, max) = (-(1i32 << (bits - 1)), (1i32 << (bits - 1)) - 1);
    let v = match kind {
        Par::S | Par::U => r,
        Par::Q => r.clamp(min, max),
        Par::Uq => r.clamp(0, mask as i32),
        Par::Sh | Par::Uh => r >> 1,
    };
    let ge = match kind {
        Par::U if add => r >> bits == 1,
        _ => r >= 0,
    };
    (v as u32 & mask, ge)
}

/// A parallel add or subtract: the result and the GE bits.
fn par(kind: Par, op: Lanes, a: u32, b: u32) -> (u32, u32) {
    // (add, take y from the other halfword) for the low and the high lane.
    let lanes16 = |lo: (bool, bool), hi: (bool, bool)| {
        let mut res = 0;
        let mut ge = 0;
        for (n, (add, cross)) in [lo, hi].into_iter().enumerate() {
            let x = a >> (16 * n);
            let y = if cross { b >> (16 * (n ^ 1)) } else { b >> (16 * n) };
            let (v, g) = par_lane(kind, add, x, y, 16);
            res |= v << (16 * n);
            if g {
                ge |= 3 << (2 * n);
            }
        }
        (res, ge)
    };
    let lanes8 = |add: bool| {
        let mut res = 0;
        let mut ge = 0;
        for n in 0..4 {
            let (v, g) = par_lane(kind, add, a >> (8 * n), b >> (8 * n), 8);
            res |= v << (8 * n);
            if g {
                ge |= 1 << n;
            }
        }
        (res, ge)
    };
    match op {
        Lanes::Add16 => lanes16((true, false), (true, false)),
        Lanes::Sub16 => lanes16((false, false), (false, false)),
        Lanes::Asx => lanes16((false, true), (true, true)),
        Lanes::Sax => lanes16((true, true), (false, true)),
        Lanes::Add8 => lanes8(true),
        Lanes::Sub8 => lanes8(false),
    }
}

/// Define the parallel add and subtract helpers, in a list named `$list`. The S and U forms
/// take `env` and write GE.
macro_rules! par_helpers {
    ($list:ident, ge; $($id:ident, $f:ident, $name:literal, $kind:ident, $op:ident;)*) => {
        $(
            fn $f(h: &mut HelperEnv<'_>, a: &[u64]) -> H {
                let (r, ge) = par(Par::$kind, Lanes::$op, a[1] as u32, a[2] as u32);
                st32(h.env, GE, ge);
                ret(r)
            }
            def!($id, $name, NO_RWG, I32, [Ptr, I32, I32], $f);
        )*
        pub(crate) const $list: &[Def] = &[$($id),*];
    };
    ($list:ident; $($id:ident, $f:ident, $name:literal, $kind:ident, $op:ident;)*) => {
        $(
            fn $f(_h: &mut HelperEnv<'_>, a: &[u64]) -> H {
                ret(par(Par::$kind, Lanes::$op, a[0] as u32, a[1] as u32).0)
            }
            def!($id, $name, NO_RWG_SE, I32, [I32, I32], $f);
        )*
        pub(crate) const $list: &[Def] = &[$($id),*];
    };
}

par_helpers! {
    PAR_GE, ge;
    SADD16, h_sadd16, "sadd16", S, Add16;
    SASX, h_sasx, "saddsubx", S, Asx;
    SSAX, h_ssax, "ssubaddx", S, Sax;
    SSUB16, h_ssub16, "ssub16", S, Sub16;
    SADD8, h_sadd8, "sadd8", S, Add8;
    SSUB8, h_ssub8, "ssub8", S, Sub8;
    UADD16, h_uadd16, "uadd16", U, Add16;
    UASX, h_uasx, "uaddsubx", U, Asx;
    USAX, h_usax, "usubaddx", U, Sax;
    USUB16, h_usub16, "usub16", U, Sub16;
    UADD8, h_uadd8, "uadd8", U, Add8;
    USUB8, h_usub8, "usub8", U, Sub8;
}

par_helpers! {
    PAR;
    QADD16, h_qadd16, "qadd16", Q, Add16;
    QASX, h_qasx, "qaddsubx", Q, Asx;
    QSAX, h_qsax, "qsubaddx", Q, Sax;
    QSUB16, h_qsub16, "qsub16", Q, Sub16;
    QADD8, h_qadd8, "qadd8", Q, Add8;
    QSUB8, h_qsub8, "qsub8", Q, Sub8;
    SHADD16, h_shadd16, "shadd16", Sh, Add16;
    SHASX, h_shasx, "shaddsubx", Sh, Asx;
    SHSAX, h_shsax, "shsubaddx", Sh, Sax;
    SHSUB16, h_shsub16, "shsub16", Sh, Sub16;
    SHADD8, h_shadd8, "shadd8", Sh, Add8;
    SHSUB8, h_shsub8, "shsub8", Sh, Sub8;
    UQADD16, h_uqadd16, "uqadd16", Uq, Add16;
    UQASX, h_uqasx, "uqaddsubx", Uq, Asx;
    UQSAX, h_uqsax, "uqsubaddx", Uq, Sax;
    UQSUB16, h_uqsub16, "uqsub16", Uq, Sub16;
    UQADD8, h_uqadd8, "uqadd8", Uq, Add8;
    UQSUB8, h_uqsub8, "uqsub8", Uq, Sub8;
    UHADD16, h_uhadd16, "uhadd16", Uh, Add16;
    UHASX, h_uhasx, "uhaddsubx", Uh, Asx;
    UHSAX, h_uhsax, "uhsubaddx", Uh, Sax;
    UHSUB16, h_uhsub16, "uhsub16", Uh, Sub16;
    UHADD8, h_uhadd8, "uhadd8", Uh, Add8;
    UHSUB8, h_uhsub8, "uhsub8", Uh, Sub8;
}

def!(SHL_CC, "shl_cc", 0, I32, [Ptr, I32, I32], h_shl_cc);
def!(SHR_CC, "shr_cc", 0, I32, [Ptr, I32, I32], h_shr_cc);
def!(SAR_CC, "sar_cc", 0, I32, [Ptr, I32, I32], h_sar_cc);
def!(ROR_CC, "ror_cc", 0, I32, [Ptr, I32, I32], h_ror_cc);
def!(ADD_SETQ, "add_setq", NO_RWG, I32, [Ptr, I32, I32], h_add_setq);
def!(ADD_SATURATE, "add_saturate", NO_RWG, I32, [Ptr, I32, I32], h_add_saturate);
def!(SUB_SATURATE, "sub_saturate", NO_RWG, I32, [Ptr, I32, I32], h_sub_saturate);
def!(SSAT, "ssat", NO_RWG, I32, [Ptr, I32, I32], h_ssat);
def!(USAT, "usat", NO_RWG, I32, [Ptr, I32, I32], h_usat);
def!(SSAT16, "ssat16", NO_RWG, I32, [Ptr, I32, I32], h_ssat16);
def!(USAT16, "usat16", NO_RWG, I32, [Ptr, I32, I32], h_usat16);
def!(SXTB16, "sxtb16", NO_RWG_SE, I32, [I32], h_sxtb16);
def!(UXTB16, "uxtb16", NO_RWG_SE, I32, [I32], h_uxtb16);
def!(SEL_FLAGS, "sel_flags", NO_RWG_SE, I32, [I32, I32, I32], h_sel_flags);
def!(USAD8, "usad8", NO_RWG_SE, I32, [I32, I32], h_usad8);
def!(RBIT, "rbit", NO_RWG_SE, I32, [I32], h_rbit);
def!(SDIV, "sdiv", NO_RWG_SE, I32, [Ptr, I32, I32], h_sdiv);
def!(UDIV, "udiv", NO_RWG_SE, I32, [Ptr, I32, I32], h_udiv);
def!(CRC32, "crc32", NO_RWG_SE, I32, [I32, I32, I32], h_crc32);
def!(CRC32C, "crc32c", NO_RWG_SE, I32, [I32, I32, I32], h_crc32c);
def!(CPSR_READ, "cpsr_read", 0, I32, [Ptr], h_cpsr_read);
def!(CPSR_WRITE, "cpsr_write", 0, Void, [Ptr, I32, I32], h_cpsr_write);
def!(VFP_GET_FPSCR, "vfp_get_fpscr", 0, I32, [Ptr], h_vfp_get_fpscr);
def!(VFP_SET_FPSCR, "vfp_set_fpscr", 0, Void, [Ptr, I32], h_vfp_set_fpscr);

/// The helpers other than the parallel add and subtract ones.
pub(crate) const BASE: &[Def] = &[
    SHL_CC,
    SHR_CC,
    SAR_CC,
    ROR_CC,
    ADD_SETQ,
    ADD_SATURATE,
    SUB_SATURATE,
    SSAT,
    USAT,
    SSAT16,
    USAT16,
    SXTB16,
    UXTB16,
    SEL_FLAGS,
    USAD8,
    RBIT,
    SDIV,
    UDIV,
    CRC32,
    CRC32C,
    CPSR_READ,
    CPSR_WRITE,
    VFP_GET_FPSCR,
    VFP_SET_FPSCR,
];

#[cfg(test)]
mod tests {
    use super::{Lanes, Par, par};

    #[test]
    fn parallel_add_sub() {
        // UADD8 carries out of the top two bytes only.
        assert_eq!(par(Par::U, Lanes::Add8, 0xff80_0102, 0x0180_0001), (0x0000_0103, 0b1100));
        // SSUB16: 1 - 2 is negative, 5 - 3 is not.
        assert_eq!(par(Par::S, Lanes::Sub16, 0x0005_0001, 0x0003_0002), (0x0002_ffff, 0b1100));
        // QADD16 saturates.
        assert_eq!(par(Par::Q, Lanes::Add16, 0x7fff_8000, 0x0001_ffff).0, 0x7fff_8000);
        // UQSUB8 floors at zero.
        assert_eq!(par(Par::Uq, Lanes::Sub8, 0x0510_0000, 0x0601_0000).0, 0x000f_0000);
        // SASX: low = a.lo - b.hi, high = a.hi + b.lo.
        assert_eq!(par(Par::S, Lanes::Asx, 0x0010_0020, 0x0005_0003).0, 0x0013_001b);
        // UHADD16 halves without losing the carry.
        assert_eq!(par(Par::Uh, Lanes::Add16, 0xffff_0002, 0xffff_0004).0, 0xffff_0003);
    }
}
