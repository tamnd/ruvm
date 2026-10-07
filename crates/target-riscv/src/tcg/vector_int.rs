// SPDX-License-Identifier: GPL-2.0-or-later

//! The vector integer and fixed point helpers, the integer and fixed point parts of QEMU's
//! `target/riscv/tcg/vector_helper.c`: "Vector Integer Arithmetic Instructions" to "Vector
//! Fixed-Point Arithmetic Instructions", and the integer extensions.
//!
//! As in [`super::vector`], a helper serves every SEW (in [`Desc::esz`]) and the `.vv`,
//! `.vx` and `.vi` forms of an instruction ([`Desc::scalar`] picks the scalar argument
//! over `vs1`). The `.vi` forms get the immediate as the scalar, as QEMU passes it to the
//! `.vx` helper.
//!
//! The fixed point helpers read `vxrm` from `env` at the start and set `vxsat` when an
//! element saturates, as QEMU's do.
//!
//! Deliberate differences from QEMU:
//!
//! - QEMU has a helper per instruction form and SEW (`vadd_vv_b` to `vadd_vx_d`); here a
//!   helper serves every SEW and the `.vv`, `.vx` and `.vi` forms, and `vmv.v.v`,
//!   `vmv.v.x` and `vmv.v.i` share one helper, as do `vzext.vf2` to `vzext.vf8` and
//!   `vsext.vf2` to `vsext.vf8` (the factor is in [`Desc::x`]).
//! - QEMU sets `vxsat` as soon as an element saturates; here it is set once after the
//!   loop. The helpers cannot fault, so the state they leave is the same.

use ruvm_jit_core::HelperType::{I32, I64, Ptr, Void};
use ruvm_jit_core::types::call_flags::NO_RWG;
use ruvm_jit_interp::HelperEnv;

use super::helpers::Def;
use super::vector::{
    Desc, HR, Shape, def, for_each, set_1s, set_vstart, sext, src1, total_elems, trunc, vget, vl,
    vmask, vop_def, vset, vset_mask, vstart,
};
use super::{ld64, st64};
use crate::cpu::{VLENB, VXRM, VXSAT};

/// A helper `fn(env, desc, scalar)` that runs `$f`, a `fn(&mut [u8], &Desc, u64)`.
macro_rules! hdef {
    ($id:ident, $name:literal, $f:expr) => {
        def!($id, $name, NO_RWG, Void, [Ptr, I32, I64], {
            fn h(e: &mut HelperEnv<'_>, a: &[u64]) -> HR {
                let f: fn(&mut [u8], &Desc, u64) = $f;
                f(e.env, &Desc::decode(a[1] as u32), a[2]);
                Ok(0)
            }
            h
        });
    };
}

/// The number of bits of an element of `1 << log2` bytes.
#[inline]
fn bits(log2: u32) -> u32 {
    8 << log2
}

/// The largest signed value of an element of `1 << log2` bytes.
#[inline]
fn smax(log2: u32) -> i64 {
    (u64::MAX >> (65 - bits(log2))) as i64
}

/// The smallest signed value of an element of `1 << log2` bytes.
#[inline]
fn smin(log2: u32) -> i64 {
    -smax(log2) - 1
}

/// The largest unsigned value of an element of `1 << log2` bytes.
#[inline]
fn umax(log2: u32) -> u64 {
    u64::MAX >> (64 - bits(log2))
}

// The loops.

/// The loop of the helpers without masking, `vadc`, `vmerge` and `vmv.v`: `body` for each
/// element from `vstart` to `vl`, then the tail of SEW elements set to ones when tail
/// agnostic.
fn for_all(env: &mut [u8], d: &Desc, mut body: impl FnMut(&mut [u8], usize)) {
    let vl = vl(env);
    let start = vstart(env);
    if start >= vl {
        // VSTART_CHECK_EARLY_EXIT().
        set_vstart(env, 0);
        return;
    }
    for i in start..vl {
        body(env, i);
    }
    set_vstart(env, 0);
    let total = total_elems(env, d, d.esz);
    set_1s(env, d.vd, d.vta, vl << d.esz, total << d.esz);
}

/// The loop of the helpers with a mask result, `vmadc` and the compares: `body` returns
/// the mask bit of each element from `vstart` to `vl`, or `None` for a masked off element.
/// A mask result is always tail agnostic, so the tail bits up to VLEN are set when
/// `VTA_ALL_1S`.
fn for_mask(env: &mut [u8], d: &Desc, mut body: impl FnMut(&mut [u8], usize) -> Option<bool>) {
    let vl = vl(env);
    let start = vstart(env);
    if start >= vl {
        set_vstart(env, 0);
        return;
    }
    for i in start..vl {
        match body(env, i) {
            Some(v) => vset_mask(env, d.vd, i, v),
            // Set the masked off elements to ones.
            None if d.vma => vset_mask(env, d.vd, i, true),
            None => {}
        }
    }
    set_vstart(env, 0);
    if d.vta_all_1s {
        for i in vl..VLENB * 8 {
            vset_mask(env, d.vd, i, true);
        }
    }
}

/// `do_vext_vv()` of the `OPIVV3` operations: `vd[i] = op(vs2[i], s1, vd[i], sew)`.
fn vop3(env: &mut [u8], d: &Desc, scalar: u64, shape: Shape, op: fn(u64, u64, u64, u32) -> u64) {
    let sew = d.esz;
    let (dl, l2, l1) = shape.log2(sew);
    for_each(env, d, dl, |env, i| {
        let s2 = vget(env, d.vs2, i, l2);
        let s1 = src1(env, d, i, l1, scalar);
        let old = vget(env, d.vd, i, dl);
        vset(env, d.vd, i, dl, op(s2, s1, old, sew));
    });
}

/// `vext_vv_rm_2()` and `vext_vx_rm_2()`: `vd[i] = op(vs2[i], s1, sew, vxrm, sat)` with
/// the rounding mode of `vxrm`, and `vxsat` set when `op` sets `sat`.
fn vop_rm(
    env: &mut [u8],
    d: &Desc,
    scalar: u64,
    shape: Shape,
    op: fn(u64, u64, u32, u64, &mut bool) -> u64,
) {
    let vxrm = ld64(env, VXRM) & 3;
    let sew = d.esz;
    let (dl, l2, l1) = shape.log2(sew);
    let mut sat = false;
    for_each(env, d, dl, |env, i| {
        let s2 = vget(env, d.vs2, i, l2);
        let s1 = src1(env, d, i, l1, scalar);
        vset(env, d.vd, i, dl, op(s2, s1, sew, vxrm, &mut sat));
    });
    if sat {
        st64(env, VXSAT, 1);
    }
}

// Vector Single-Width Integer Add and Subtract.

vop_def!(VADD, "vadd", Shape::Single, |a, b, _| a.wrapping_add(b));
vop_def!(VSUB, "vsub", Shape::Single, |a, b, _| a.wrapping_sub(b));
vop_def!(VRSUB, "vrsub", Shape::Single, |a, b, _| b.wrapping_sub(a));

// Vector Widening Integer Add/Subtract.

vop_def!(VWADDU, "vwaddu", Shape::Widen, |a, b, _| a.wrapping_add(b));
vop_def!(VWADD, "vwadd", Shape::Widen, |a, b, s| { sext(a, s).wrapping_add(sext(b, s)) as u64 });
vop_def!(VWSUBU, "vwsubu", Shape::Widen, |a, b, _| a.wrapping_sub(b));
vop_def!(VWSUB, "vwsub", Shape::Widen, |a, b, s| { sext(a, s).wrapping_sub(sext(b, s)) as u64 });
vop_def!(VWADDU_W, "vwaddu_w", Shape::WidenW, |a, b, _| a.wrapping_add(b));
vop_def!(VWADD_W, "vwadd_w", Shape::WidenW, |a, b, s| a.wrapping_add(sext(b, s) as u64));
vop_def!(VWSUBU_W, "vwsubu_w", Shape::WidenW, |a, b, _| a.wrapping_sub(b));
vop_def!(VWSUB_W, "vwsub_w", Shape::WidenW, |a, b, s| a.wrapping_sub(sext(b, s) as u64));

// Vector Integer Add-with-Carry / Subtract-with-Borrow.

/// `vadc` and `vsbc`: `vs2 + s1 + v0[i]` or `vs2 - s1 - v0[i]` for every element.
fn adc(env: &mut [u8], d: &Desc, scalar: u64, sub: bool) {
    let sew = d.esz;
    for_all(env, d, |env, i| {
        let s2 = vget(env, d.vs2, i, sew);
        let s1 = src1(env, d, i, sew, scalar);
        let c = u64::from(vmask(env, 0, i));
        let v = if sub {
            s2.wrapping_sub(s1).wrapping_sub(c)
        } else {
            s2.wrapping_add(s1).wrapping_add(c)
        };
        vset(env, d.vd, i, sew, v);
    });
}

/// `vmadc` and `vmsbc`: the carry out of `vs2 + s1` or the borrow out of `vs2 - s1`, with
/// the carry or borrow in from `v0` when masked.
fn madc(env: &mut [u8], d: &Desc, scalar: u64, sub: bool) {
    let sew = d.esz;
    for_mask(env, d, |env, i| {
        let s2 = vget(env, d.vs2, i, sew);
        let s1 = src1(env, d, i, sew, scalar);
        let c = !d.vm && vmask(env, 0, i);
        Some(if sub {
            // DO_MSBC().
            if c { s2 <= s1 } else { s2 < s1 }
        } else {
            // DO_MADC().
            let sum = s2.wrapping_add(s1);
            if c { trunc(sum.wrapping_add(1), sew) <= s2 } else { trunc(sum, sew) < s2 }
        })
    });
}

hdef!(VADC, "vadc", |env, d, s| adc(env, d, s, false));
hdef!(VSBC, "vsbc", |env, d, s| adc(env, d, s, true));
hdef!(VMADC, "vmadc", |env, d, s| madc(env, d, s, false));
hdef!(VMSBC, "vmsbc", |env, d, s| madc(env, d, s, true));

// Vector Bitwise Logical.

vop_def!(VAND, "vand", Shape::Single, |a, b, _| a & b);
vop_def!(VOR, "vor", Shape::Single, |a, b, _| a | b);
vop_def!(VXOR, "vxor", Shape::Single, |a, b, _| a ^ b);

// Vector Single-Width Bit Shift.

vop_def!(VSLL, "vsll", Shape::Single, |a, b, s| a << (b & u64::from(bits(s) - 1)));
vop_def!(VSRL, "vsrl", Shape::Single, |a, b, s| a >> (b & u64::from(bits(s) - 1)));
vop_def!(VSRA, "vsra", Shape::Single, |a, b, s| {
    (sext(a, s) >> (b & u64::from(bits(s) - 1))) as u64
});

// Vector Narrowing Integer Right Shift: the shift amount has log2(2*SEW) bits.

vop_def!(VNSRL, "vnsrl", Shape::Narrow, |a, b, s| a >> (b & u64::from(2 * bits(s) - 1)));
vop_def!(VNSRA, "vnsra", Shape::Narrow, |a, b, s| {
    (sext(a, s + 1) >> (b & u64::from(2 * bits(s) - 1))) as u64
});

// Vector Integer Comparison.

/// A compare: the mask bit `op(vs2[i], s1, sew)` of each active element.
fn cmp(env: &mut [u8], d: &Desc, scalar: u64, op: fn(u64, u64, u32) -> bool) {
    let sew = d.esz;
    for_mask(env, d, |env, i| {
        let s2 = vget(env, d.vs2, i, sew);
        let s1 = src1(env, d, i, sew, scalar);
        if !d.vm && !vmask(env, 0, i) {
            return None;
        }
        Some(op(s2, s1, sew))
    });
}

hdef!(VMSEQ, "vmseq", |env, d, s| cmp(env, d, s, |a, b, _| a == b));
hdef!(VMSNE, "vmsne", |env, d, s| cmp(env, d, s, |a, b, _| a != b));
hdef!(VMSLTU, "vmsltu", |env, d, s| cmp(env, d, s, |a, b, _| a < b));
hdef!(VMSLT, "vmslt", |env, d, s| cmp(env, d, s, |a, b, w| sext(a, w) < sext(b, w)));
hdef!(VMSLEU, "vmsleu", |env, d, s| cmp(env, d, s, |a, b, _| a <= b));
hdef!(VMSLE, "vmsle", |env, d, s| cmp(env, d, s, |a, b, w| sext(a, w) <= sext(b, w)));
hdef!(VMSGTU, "vmsgtu", |env, d, s| cmp(env, d, s, |a, b, _| a > b));
hdef!(VMSGT, "vmsgt", |env, d, s| cmp(env, d, s, |a, b, w| sext(a, w) > sext(b, w)));

// Vector Integer Min/Max.

vop_def!(VMINU, "vminu", Shape::Single, |a, b, _| a.min(b));
vop_def!(VMIN, "vmin", Shape::Single, |a, b, s| sext(a, s).min(sext(b, s)) as u64);
vop_def!(VMAXU, "vmaxu", Shape::Single, |a, b, _| a.max(b));
vop_def!(VMAX, "vmax", Shape::Single, |a, b, s| sext(a, s).max(sext(b, s)) as u64);

// Vector Single-Width Integer Multiply.

/// `do_mulh_*()`: the high SEW bits of the signed product.
fn mulh(a: u64, b: u64, s: u32) -> u64 {
    let p = i128::from(sext(a, s)) * i128::from(sext(b, s));
    (p >> bits(s)) as u64
}

/// `do_mulhu_*()`: the high SEW bits of the unsigned product.
fn mulhu(a: u64, b: u64, s: u32) -> u64 {
    ((u128::from(a) * u128::from(b)) >> bits(s)) as u64
}

/// `do_mulhsu_*()`: the high SEW bits of the product of signed `vs2` and unsigned `s1`.
fn mulhsu(a: u64, b: u64, s: u32) -> u64 {
    let p = i128::from(sext(a, s)) * i128::from(b);
    (p >> bits(s)) as u64
}

vop_def!(VMUL, "vmul", Shape::Single, |a, b, _| a.wrapping_mul(b));
vop_def!(VMULH, "vmulh", Shape::Single, mulh);
vop_def!(VMULHU, "vmulhu", Shape::Single, mulhu);
vop_def!(VMULHSU, "vmulhsu", Shape::Single, mulhsu);

// Vector Integer Divide.

/// `DO_DIVU()`: all ones for a division by zero.
fn divu(a: u64, b: u64, s: u32) -> u64 {
    a.checked_div(b).unwrap_or_else(|| umax(s))
}

/// `DO_REMU()`: the dividend for a division by zero.
fn remu(a: u64, b: u64, _s: u32) -> u64 {
    if b == 0 { a } else { a % b }
}

/// `DO_DIV()`: -1 for a division by zero, the dividend for the overflow of the most
/// negative value divided by -1.
fn div(a: u64, b: u64, s: u32) -> u64 {
    let (n, m) = (sext(a, s), sext(b, s));
    if m == 0 {
        u64::MAX
    } else if n == smin(s) && m == -1 {
        a
    } else {
        (n / m) as u64
    }
}

/// `DO_REM()`: the dividend for a division by zero, 0 for the overflow.
fn rem(a: u64, b: u64, s: u32) -> u64 {
    let (n, m) = (sext(a, s), sext(b, s));
    if m == 0 {
        a
    } else if n == smin(s) && m == -1 {
        0
    } else {
        (n % m) as u64
    }
}

vop_def!(VDIVU, "vdivu", Shape::Single, divu);
vop_def!(VDIV, "vdiv", Shape::Single, div);
vop_def!(VREMU, "vremu", Shape::Single, remu);
vop_def!(VREM, "vrem", Shape::Single, rem);

// Vector Widening Integer Multiply.

vop_def!(VWMULU, "vwmulu", Shape::Widen, |a, b, _| a.wrapping_mul(b));
vop_def!(VWMUL, "vwmul", Shape::Widen, |a, b, s| { sext(a, s).wrapping_mul(sext(b, s)) as u64 });
// WOP_SUS: signed vs2, unsigned vs1.
vop_def!(VWMULSU, "vwmulsu", Shape::Widen, |a, b, s| (sext(a, s) as u64).wrapping_mul(b));

// Vector Single-Width Integer Multiply-Add: `vd[i] = op(vs2[i], s1, vd[i])`.

hdef!(VMACC, "vmacc", |env, d, s| {
    vop3(env, d, s, Shape::Single, |a, b, v, _| b.wrapping_mul(a).wrapping_add(v));
});
hdef!(VNMSAC, "vnmsac", |env, d, s| {
    vop3(env, d, s, Shape::Single, |a, b, v, _| v.wrapping_sub(b.wrapping_mul(a)));
});
hdef!(VMADD, "vmadd", |env, d, s| {
    vop3(env, d, s, Shape::Single, |a, b, v, _| b.wrapping_mul(v).wrapping_add(a));
});
hdef!(VNMSUB, "vnmsub", |env, d, s| {
    vop3(env, d, s, Shape::Single, |a, b, v, _| a.wrapping_sub(b.wrapping_mul(v)));
});

// Vector Widening Integer Multiply-Add: `vd[i] = s1 * vs2[i] + vd[i]` in 2*SEW.

hdef!(VWMACCU, "vwmaccu", |env, d, s| {
    vop3(env, d, s, Shape::Widen, |a, b, v, _| b.wrapping_mul(a).wrapping_add(v));
});
hdef!(VWMACC, "vwmacc", |env, d, s| {
    vop3(env, d, s, Shape::Widen, |a, b, v, w| {
        (sext(b, w).wrapping_mul(sext(a, w)) as u64).wrapping_add(v)
    });
});
// WOP_SSU: signed vs1, unsigned vs2.
hdef!(VWMACCSU, "vwmaccsu", |env, d, s| {
    vop3(env, d, s, Shape::Widen, |a, b, v, w| (sext(b, w) as u64).wrapping_mul(a).wrapping_add(v));
});
// WOP_SUS: unsigned rs1, signed vs2.
hdef!(VWMACCUS, "vwmaccus", |env, d, s| {
    vop3(env, d, s, Shape::Widen, |a, b, v, w| b.wrapping_mul(sext(a, w) as u64).wrapping_add(v));
});

// Vector Integer Merge and Move.

/// `vmerge`: `s1` where `v0` is set, `vs2[i]` elsewhere.
fn merge(env: &mut [u8], d: &Desc, scalar: u64) {
    let sew = d.esz;
    for_all(env, d, |env, i| {
        let v =
            if vmask(env, 0, i) { src1(env, d, i, sew, scalar) } else { vget(env, d.vs2, i, sew) };
        vset(env, d.vd, i, sew, v);
    });
}

/// `vmv.v.v`, `vmv.v.x` and `vmv.v.i`: `vd[i] = s1`.
fn mv(env: &mut [u8], d: &Desc, scalar: u64) {
    let sew = d.esz;
    for_all(env, d, |env, i| {
        let v = src1(env, d, i, sew, scalar);
        vset(env, d.vd, i, sew, v);
    });
}

hdef!(VMERGE, "vmerge", merge);
hdef!(VMV_V, "vmv_v", mv);

// Vector Fixed-Point Arithmetic.

/// `get_round()`: the rounding increment of `v` shifted right by `shift` in rounding mode
/// `vxrm`.
fn get_round(vxrm: u64, v: u64, shift: u32) -> u64 {
    if shift == 0 || shift > 64 {
        return 0;
    }
    let d = if shift < 64 { (v >> shift) & 1 } else { 0 };
    let d1 = (v >> (shift - 1)) & 1;
    let low = |n: u32| if n >= 64 { v } else { v & ((1u64 << n) - 1) };
    match vxrm {
        // Round to nearest up (add +0.5 LSB).
        0 => d1,
        // Round to nearest even.
        1 => {
            if shift > 1 {
                d1 & (u64::from(low(shift - 1) != 0) | d)
            } else {
                d1 & d
            }
        }
        // Round down (truncate).
        2 => 0,
        // Round to odd (OR the bits into the LSB, "jam").
        _ => u64::from(d == 0 && low(shift) != 0),
    }
}

/// `saddu*()`.
fn saddu(a: u64, b: u64, s: u32, _vxrm: u64, sat: &mut bool) -> u64 {
    let res = trunc(a.wrapping_add(b), s);
    if res < a {
        *sat = true;
        umax(s)
    } else {
        res
    }
}

/// `sadd*()`.
fn sadd(a: u64, b: u64, s: u32, _vxrm: u64, sat: &mut bool) -> u64 {
    let (a, b) = (sext(a, s), sext(b, s));
    let res = sext(a.wrapping_add(b) as u64, s);
    if (res ^ a) & (res ^ b) < 0 {
        *sat = true;
        (if a > 0 { smax(s) } else { smin(s) }) as u64
    } else {
        res as u64
    }
}

/// `ssubu*()`.
fn ssubu(a: u64, b: u64, s: u32, _vxrm: u64, sat: &mut bool) -> u64 {
    let res = trunc(a.wrapping_sub(b), s);
    if res > a {
        *sat = true;
        0
    } else {
        res
    }
}

/// `ssub*()`.
fn ssub(a: u64, b: u64, s: u32, _vxrm: u64, sat: &mut bool) -> u64 {
    let (a, b) = (sext(a, s), sext(b, s));
    let res = sext(a.wrapping_sub(b) as u64, s);
    if (res ^ a) & (a ^ b) < 0 {
        *sat = true;
        (if a >= 0 { smax(s) } else { smin(s) }) as u64
    } else {
        res as u64
    }
}

/// `aadd32()` for SEW 8 to 32 and `aadd64()`.
fn aadd(a: u64, b: u64, s: u32, vxrm: u64, _sat: &mut bool) -> u64 {
    let (a, b) = (sext(a, s), sext(b, s));
    if s < 3 {
        let res = a + b;
        let round = get_round(vxrm, res as u64, 1);
        ((res >> 1) as u64).wrapping_add(round)
    } else {
        let res = a.wrapping_add(b);
        let round = get_round(vxrm, res as u64, 1);
        // With signed overflow, bit 64 is the inverse of bit 63.
        let over = (res ^ a) & (res ^ b) & i64::MIN;
        (((res >> 1) ^ over) as u64).wrapping_add(round)
    }
}

/// `aaddu32()` for SEW 8 to 32 and `aaddu64()`.
fn aaddu(a: u64, b: u64, s: u32, vxrm: u64, _sat: &mut bool) -> u64 {
    let res = a.wrapping_add(b);
    let round = get_round(vxrm, res, 1);
    if s < 3 {
        (res >> 1).wrapping_add(round)
    } else {
        let over = u64::from(res < a) << 63;
        ((res >> 1) | over).wrapping_add(round)
    }
}

/// `asub32()` for SEW 8 to 32 and `asub64()`.
fn asub(a: u64, b: u64, s: u32, vxrm: u64, _sat: &mut bool) -> u64 {
    let (a, b) = (sext(a, s), sext(b, s));
    let res = a.wrapping_sub(b);
    let round = get_round(vxrm, res as u64, 1);
    if s < 3 {
        ((res >> 1) as u64).wrapping_add(round)
    } else {
        // With signed overflow, bit 64 is the inverse of bit 63.
        let over = (res ^ a) & (a ^ b) & i64::MIN;
        (((res >> 1) ^ over) as u64).wrapping_add(round)
    }
}

/// `asubu32()` for SEW 8 to 32 and `asubu64()`.
fn asubu(a: u64, b: u64, s: u32, vxrm: u64, _sat: &mut bool) -> u64 {
    if s < 3 {
        let res = a as i64 - b as i64;
        let round = get_round(vxrm, res as u64, 1);
        ((res >> 1) as u64).wrapping_add(round)
    } else {
        let res = a.wrapping_sub(b);
        let round = get_round(vxrm, res, 1);
        let over = u64::from(res > a) << 63;
        ((res >> 1) | over).wrapping_add(round)
    }
}

/// `vsmul8()` to `vsmul64()`: the signed product shifted right by SEW - 1 with rounding,
/// saturated.
fn vsmul(a: u64, b: u64, s: u32, vxrm: u64, sat: &mut bool) -> u64 {
    let (a, b) = (sext(a, s), sext(b, s));
    if s < 3 {
        let sh = bits(s) - 1;
        let res = a * b;
        let round = get_round(vxrm, res as u64, sh);
        let res = (res >> sh) + round as i64;
        if res > smax(s) {
            *sat = true;
            smax(s) as u64
        } else if res < smin(s) {
            *sat = true;
            smin(s) as u64
        } else {
            res as u64
        }
    } else {
        if a == i64::MIN && b == i64::MIN {
            *sat = true;
            return i64::MAX as u64;
        }
        let p = i128::from(a) * i128::from(b);
        let (lo, hi) = (p as u64, (p >> 64) as u64);
        let round = get_round(vxrm, lo, 63);
        // This cannot overflow, there are always two sign bits after the multiply.
        let mut res = (hi << 1) | (lo >> 63);
        if round != 0 {
            if res == i64::MAX as u64 {
                *sat = true;
            } else {
                res += 1;
            }
        }
        res
    }
}

/// `vssrl8()` to `vssrl64()`.
fn vssrl(a: u64, b: u64, s: u32, vxrm: u64, _sat: &mut bool) -> u64 {
    let shift = (b & u64::from(bits(s) - 1)) as u32;
    let round = get_round(vxrm, a, shift);
    (a >> shift).wrapping_add(round)
}

/// `vssra8()` to `vssra64()`.
fn vssra(a: u64, b: u64, s: u32, vxrm: u64, _sat: &mut bool) -> u64 {
    let a = sext(a, s);
    let shift = (b & u64::from(bits(s) - 1)) as u32;
    let round = get_round(vxrm, a as u64, shift);
    ((a >> shift) as u64).wrapping_add(round)
}

/// `vnclipu8()` to `vnclipu32()`: a 2*SEW `vs2` shifted right with rounding, saturated to
/// SEW.
fn vnclipu(a: u64, b: u64, s: u32, vxrm: u64, sat: &mut bool) -> u64 {
    let shift = (b & u64::from(2 * bits(s) - 1)) as u32;
    let round = get_round(vxrm, a, shift);
    let res = (a >> shift).wrapping_add(round);
    if res > umax(s) {
        *sat = true;
        umax(s)
    } else {
        res
    }
}

/// `vnclip8()` to `vnclip32()`.
fn vnclip(a: u64, b: u64, s: u32, vxrm: u64, sat: &mut bool) -> u64 {
    let a = sext(a, s + 1);
    let shift = (b & u64::from(2 * bits(s) - 1)) as u32;
    let round = get_round(vxrm, a as u64, shift);
    let res = (a >> shift) + round as i64;
    if res > smax(s) {
        *sat = true;
        smax(s) as u64
    } else if res < smin(s) {
        *sat = true;
        smin(s) as u64
    } else {
        res as u64
    }
}

hdef!(VSADDU, "vsaddu", |env, d, s| vop_rm(env, d, s, Shape::Single, saddu));
hdef!(VSADD, "vsadd", |env, d, s| vop_rm(env, d, s, Shape::Single, sadd));
hdef!(VSSUBU, "vssubu", |env, d, s| vop_rm(env, d, s, Shape::Single, ssubu));
hdef!(VSSUB, "vssub", |env, d, s| vop_rm(env, d, s, Shape::Single, ssub));
hdef!(VAADD, "vaadd", |env, d, s| vop_rm(env, d, s, Shape::Single, aadd));
hdef!(VAADDU, "vaaddu", |env, d, s| vop_rm(env, d, s, Shape::Single, aaddu));
hdef!(VASUB, "vasub", |env, d, s| vop_rm(env, d, s, Shape::Single, asub));
hdef!(VASUBU, "vasubu", |env, d, s| vop_rm(env, d, s, Shape::Single, asubu));
hdef!(VSMUL, "vsmul", |env, d, s| vop_rm(env, d, s, Shape::Single, vsmul));
hdef!(VSSRL, "vssrl", |env, d, s| vop_rm(env, d, s, Shape::Single, vssrl));
hdef!(VSSRA, "vssra", |env, d, s| vop_rm(env, d, s, Shape::Single, vssra));
hdef!(VNCLIPU, "vnclipu", |env, d, s| vop_rm(env, d, s, Shape::Narrow, vnclipu));
hdef!(VNCLIP, "vnclip", |env, d, s| vop_rm(env, d, s, Shape::Narrow, vnclip));

// Vector Integer Extension.

/// `vzext.vf*` and `vsext.vf*`: `vd[i]` is `vs2[i]` of SEW / 2^`Desc::x` bits, zero or sign
/// extended.
fn ext(env: &mut [u8], d: &Desc, signed: bool) {
    let sew = d.esz;
    let from = sew - d.x;
    for_each(env, d, sew, |env, i| {
        let v = vget(env, d.vs2, i, from);
        let v = if signed { sext(v, from) as u64 } else { v };
        vset(env, d.vd, i, sew, v);
    });
}

hdef!(VZEXT, "vzext_vf", |env, d, _| ext(env, d, false));
hdef!(VSEXT, "vsext_vf", |env, d, _| ext(env, d, true));

/// The helpers of this module.
pub(crate) const ALL: &[Def] = &[
    VADD, VSUB, VRSUB, VWADDU, VWADD, VWSUBU, VWSUB, VWADDU_W, VWADD_W, VWSUBU_W, VWSUB_W, VADC,
    VSBC, VMADC, VMSBC, VAND, VOR, VXOR, VSLL, VSRL, VSRA, VNSRL, VNSRA, VMSEQ, VMSNE, VMSLTU,
    VMSLT, VMSLEU, VMSLE, VMSGTU, VMSGT, VMINU, VMIN, VMAXU, VMAX, VMUL, VMULH, VMULHU, VMULHSU,
    VDIVU, VDIV, VREMU, VREM, VWMULU, VWMUL, VWMULSU, VMACC, VNMSAC, VMADD, VNMSUB, VWMACCU,
    VWMACC, VWMACCSU, VWMACCUS, VMERGE, VMV_V, VSADDU, VSADD, VSSUBU, VSSUB, VAADD, VAADDU, VASUB,
    VASUBU, VSMUL, VSSRL, VSSRA, VNCLIPU, VNCLIP, VZEXT, VSEXT,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{ENV_SIZE, VL, VSTART, VTYPE};
    use crate::tcg::vector::voff;
    use ruvm_jit_interp::NoMemory;

    /// An `env` with `vtype` SEW `1 << sew` bytes, LMUL 1, and `vl`.
    fn env(sew: u32, vl: u64) -> Vec<u8> {
        let mut env = vec![0u8; ENV_SIZE];
        st64(&mut env, VTYPE, u64::from(sew) << 3);
        st64(&mut env, VL, vl);
        env
    }

    /// Call helper `h` with descriptor `d` and scalar `s`.
    fn call(h: &Def, env: &mut [u8], d: Desc, s: u64) {
        let a = [0, u64::from(d.encode()), s];
        let mut mem = NoMemory;
        let mut he = HelperEnv { env, mem: &mut mem };
        (h.f)(&mut he, &a).expect("no unwind");
    }

    fn fill(env: &mut [u8], reg: u32, log2: u32, vals: &[u64]) {
        for (i, &v) in vals.iter().enumerate() {
            vset(env, reg, i, log2, v);
        }
    }

    fn get(env: &[u8], reg: u32, log2: u32, n: usize) -> Vec<u64> {
        (0..n).map(|i| vget(env, reg, i, log2)).collect()
    }

    /// An unmasked `.vv` descriptor: `vd` v1, `vs1` v2, `vs2` v3.
    fn vv(esz: u32) -> Desc {
        Desc { vm: true, vd: 1, vs1: 2, vs2: 3, esz, ..Desc::default() }
    }

    /// The `vd` elements of fixed point helper `h` in each `vxrm` mode, and `vxsat`.
    fn per_vxrm(h: &Def, e: &mut [u8], d: Desc, s: u64, dl: u32, n: usize) -> Vec<Vec<u64>> {
        (0..4)
            .map(|rm| {
                st64(e, VXRM, rm);
                call(h, e, d, s);
                get(e, d.vd, dl, n)
            })
            .collect()
    }

    #[test]
    fn rounding_modes() {
        // get_round() directly: 0b101 >> 1 and 0b10110 >> 2.
        assert_eq!((0..4).map(|m| get_round(m, 5, 1)).collect::<Vec<_>>(), [1, 0, 0, 1]);
        assert_eq!((0..4).map(|m| get_round(m, 22, 2)).collect::<Vec<_>>(), [1, 1, 0, 0]);
        assert_eq!(get_round(0, u64::MAX, 0), 0);

        // vaadd, SEW 8: (4 + 1) / 2, (6 + 1) / 2, (-3 + 0) / 2.
        let mut e = env(0, 3);
        fill(&mut e, 3, 0, &[4, 6, 0xfd]);
        fill(&mut e, 2, 0, &[1, 1, 0]);
        let r = per_vxrm(&VAADD, &mut e, vv(0), 0, 0, 3);
        assert_eq!(r, [[3, 4, 0xff], [2, 4, 0xfe], [2, 3, 0xfe], [3, 3, 0xff]]);
        // vaaddu, SEW 64 with a carry out: (MAX + 1) / 2 is 2^63.
        let mut e = env(3, 1);
        fill(&mut e, 3, 3, &[u64::MAX]);
        let r = per_vxrm(&VAADDU, &mut e, Desc { scalar: true, ..vv(3) }, 1, 3, 1);
        assert!(r.iter().all(|v| v[0] == 1 << 63));
        // vasub, SEW 64 with an overflow: (MIN - 1) / 2.
        fill(&mut e, 3, 3, &[1 << 63]);
        let r = per_vxrm(&VASUB, &mut e, Desc { scalar: true, ..vv(3) }, 1, 3, 1);
        let (up, down) = (0xc000_0000_0000_0000, 0xbfff_ffff_ffff_ffff);
        assert_eq!(r, [[up], [up], [down], [down]]);

        // vssrl.vx, SEW 16: 22 >> 2 and 18 >> 2; the shift is masked to 4 bits.
        let mut e = env(1, 2);
        fill(&mut e, 3, 1, &[22, 18]);
        let r = per_vxrm(&VSSRL, &mut e, Desc { scalar: true, ..vv(1) }, 0x12, 1, 2);
        assert_eq!(r, [[6, 5], [6, 4], [5, 4], [5, 5]]);
        // vssra, SEW 8: -3 >> 1.
        let mut e = env(0, 1);
        fill(&mut e, 3, 0, &[0xfd]);
        let r = per_vxrm(&VSSRA, &mut e, Desc { scalar: true, ..vv(0) }, 1, 0, 1);
        assert_eq!(r, [[0xff], [0xfe], [0xfe], [0xff]]);

        // vsmul, SEW 8: 1 * 64 >> 7 is a half.
        let mut e = env(0, 1);
        fill(&mut e, 3, 0, &[1]);
        let r = per_vxrm(&VSMUL, &mut e, Desc { scalar: true, ..vv(0) }, 64, 0, 1);
        assert_eq!(r, [[1], [0], [0], [1]]);
        assert_eq!(ld64(&e, VXSAT), 0);
        // SEW 64: 3 * 2^62 >> 63 is 1.5.
        let mut e = env(3, 1);
        fill(&mut e, 3, 3, &[1 << 62]);
        let r = per_vxrm(&VSMUL, &mut e, Desc { scalar: true, ..vv(3) }, 3, 3, 1);
        assert_eq!(r, [[2], [2], [1], [1]]);
        assert_eq!(ld64(&e, VXSAT), 0);

        // vnclip.wi, SEW 8: 0x123 >> 4 has the rounding bits 0b0011.
        let mut e = env(0, 1);
        fill(&mut e, 3, 1, &[0x123]);
        let r = per_vxrm(&VNCLIP, &mut e, Desc { scalar: true, ..vv(0) }, 4, 0, 1);
        assert_eq!(r, [[0x12], [0x12], [0x12], [0x13]]);
        let r = per_vxrm(&VNCLIPU, &mut e, Desc { scalar: true, ..vv(0) }, 4, 0, 1);
        assert_eq!(r, [[0x12], [0x12], [0x12], [0x13]]);
        assert_eq!(ld64(&e, VXSAT), 0);
    }

    #[test]
    fn saturation() {
        // vsmul, SEW 8 and 64: MIN * MIN saturates.
        let mut e = env(0, 1);
        fill(&mut e, 3, 0, &[0x80]);
        call(&VSMUL, &mut e, Desc { scalar: true, ..vv(0) }, 0x80);
        assert_eq!(vget(&e, 1, 0, 0), 0x7f);
        assert_eq!(ld64(&e, VXSAT), 1);
        let mut e = env(3, 1);
        fill(&mut e, 3, 3, &[1 << 63]);
        call(&VSMUL, &mut e, Desc { scalar: true, ..vv(3) }, 1 << 63);
        assert_eq!(vget(&e, 1, 0, 3), i64::MAX as u64);
        assert_eq!(ld64(&e, VXSAT), 1);

        // vnclip.wv, SEW 8: 0x7fff, -256 >> 1 and 0x8000 >> 4.
        let mut e = env(0, 3);
        fill(&mut e, 3, 1, &[0x7fff, 0xff00, 0x8000]);
        fill(&mut e, 2, 0, &[0, 1, 0x14]);
        call(&VNCLIP, &mut e, vv(0), 0);
        assert_eq!(get(&e, 1, 0, 3), [0x7f, 0x80, 0x80]);
        assert_eq!(ld64(&e, VXSAT), 1);
        // vnclipu: 0x7fff >> 15 rounds up, 0xff00 >> 1 saturates, 0x8000 >> (0x1c & 15) does
        // not.
        st64(&mut e, VXSAT, 0);
        fill(&mut e, 2, 0, &[15, 1, 0x1c]);
        call(&VNCLIPU, &mut e, vv(0), 0);
        assert_eq!(get(&e, 1, 0, 3), [1, 0xff, 0x08]);
        assert_eq!(ld64(&e, VXSAT), 1);

        // The saturating adds and subtracts, SEW 8.
        let mut e = env(0, 2);
        let d = Desc { scalar: true, ..vv(0) };
        fill(&mut e, 3, 0, &[0xff, 0x10]);
        call(&VSADDU, &mut e, d, 1);
        assert_eq!(get(&e, 1, 0, 2), [0xff, 0x11]);
        assert_eq!(ld64(&e, VXSAT), 1);
        st64(&mut e, VXSAT, 0);
        fill(&mut e, 3, 0, &[0x7f, 0x80]);
        call(&VSADD, &mut e, d, 1);
        assert_eq!(get(&e, 1, 0, 2), [0x7f, 0x81]);
        assert_eq!(ld64(&e, VXSAT), 1);
        st64(&mut e, VXSAT, 0);
        call(&VSSUB, &mut e, d, 1);
        assert_eq!(get(&e, 1, 0, 2), [0x7e, 0x80]);
        assert_eq!(ld64(&e, VXSAT), 1);
        st64(&mut e, VXSAT, 0);
        call(&VSSUBU, &mut e, d, 0x7f);
        assert_eq!(get(&e, 1, 0, 2), [0, 1]);
        assert_eq!(ld64(&e, VXSAT), 0);
        call(&VSSUBU, &mut e, d, 0x81);
        assert_eq!(get(&e, 1, 0, 2), [0, 0]);
        assert_eq!(ld64(&e, VXSAT), 1);
        // vsadd.vi: the immediate -1 is truncated to SEW.
        st64(&mut e, VXSAT, 0);
        call(&VSADD, &mut e, d, u64::MAX);
        assert_eq!(get(&e, 1, 0, 2), [0x7e, 0x80]);
        assert_eq!(ld64(&e, VXSAT), 1);
    }

    #[test]
    fn widen_and_narrow() {
        let mut e = env(0, 2);
        fill(&mut e, 3, 0, &[0xff, 0x7f]);
        fill(&mut e, 2, 0, &[0xff, 0x01]);
        let d = Desc { vd: 4, ..vv(0) };
        call(&VWADD, &mut e, d, 0);
        assert_eq!(get(&e, 4, 1, 2), [0xfffe, 0x80]);
        call(&VWADDU, &mut e, d, 0);
        assert_eq!(get(&e, 4, 1, 2), [0x1fe, 0x80]);
        call(&VWSUB, &mut e, d, 0);
        assert_eq!(get(&e, 4, 1, 2), [0, 0x7e]);
        call(&VWMUL, &mut e, d, 0);
        assert_eq!(get(&e, 4, 1, 2), [1, 0x7f]);
        call(&VWMULU, &mut e, d, 0);
        assert_eq!(get(&e, 4, 1, 2), [0xfe01, 0x7f]);
        // Signed vs2 times unsigned vs1.
        call(&VWMULSU, &mut e, d, 0);
        assert_eq!(get(&e, 4, 1, 2), [0xff01, 0x7f]);
        // vwmaccsu: signed vs1 times unsigned vs2, plus vd.
        fill(&mut e, 4, 1, &[1, 1]);
        call(&VWMACCSU, &mut e, d, 0);
        assert_eq!(get(&e, 4, 1, 2), [0xff02, 0x80]);
        // vwmaccus.vx: unsigned rs1 times signed vs2.
        fill(&mut e, 4, 1, &[0, 0]);
        call(&VWMACCUS, &mut e, Desc { scalar: true, ..d }, 0xff);
        assert_eq!(get(&e, 4, 1, 2), [0xff01, 0x7e81]);

        // vwadd.wv: a 16-bit vs2 and a sign extended 8-bit vs1.
        fill(&mut e, 3, 1, &[0x1000, 0xffff]);
        fill(&mut e, 2, 0, &[0x80, 0x01]);
        call(&VWADD_W, &mut e, d, 0);
        assert_eq!(get(&e, 4, 1, 2), [0x0f80, 0]);
        call(&VWADDU_W, &mut e, d, 0);
        assert_eq!(get(&e, 4, 1, 2), [0x1080, 0]);

        // vnsrl and vnsra, SEW 8: the shift is masked to 4 bits.
        fill(&mut e, 3, 1, &[0x1234, 0x8000]);
        fill(&mut e, 2, 0, &[0x14, 8]);
        call(&VNSRL, &mut e, vv(0), 0);
        assert_eq!(get(&e, 1, 0, 2), [0x23, 0x80]);
        call(&VNSRA, &mut e, Desc { scalar: true, ..vv(0) }, 12);
        assert_eq!(get(&e, 1, 0, 2), [0x01, 0xf8]);
    }

    #[test]
    fn divide_edge_cases() {
        let mut e = env(0, 4);
        fill(&mut e, 3, 0, &[7, 0x80, 0xf9, 200]);
        fill(&mut e, 2, 0, &[0, 0xff, 2, 3]);
        call(&VDIV, &mut e, vv(0), 0);
        assert_eq!(get(&e, 1, 0, 4), [0xff, 0x80, 0xfd, 0xee]);
        call(&VREM, &mut e, vv(0), 0);
        assert_eq!(get(&e, 1, 0, 4), [7, 0, 0xff, 0xfe]);
        call(&VDIVU, &mut e, vv(0), 0);
        assert_eq!(get(&e, 1, 0, 4), [0xff, 0, 0x7c, 66]);
        call(&VREMU, &mut e, vv(0), 0);
        assert_eq!(get(&e, 1, 0, 4), [7, 0x80, 1, 2]);
        let mut e = env(3, 2);
        fill(&mut e, 3, 3, &[1 << 63, 5]);
        call(&VDIV, &mut e, Desc { scalar: true, ..vv(3) }, u64::MAX);
        assert_eq!(get(&e, 1, 3, 2), [1 << 63, (-5i64) as u64]);
        call(&VREM, &mut e, Desc { scalar: true, ..vv(3) }, u64::MAX);
        assert_eq!(get(&e, 1, 3, 2), [0, 0]);
        call(&VDIVU, &mut e, Desc { scalar: true, ..vv(3) }, 0);
        assert_eq!(get(&e, 1, 3, 2), [u64::MAX, u64::MAX]);
        // The high halves of the products.
        assert_eq!(mulh(1 << 63, 1 << 63, 3), 1 << 62);
        assert_eq!(mulhu(u64::MAX, u64::MAX, 3), u64::MAX - 1);
        assert_eq!(mulhsu(u64::MAX, u64::MAX, 3), u64::MAX);
        assert_eq!(mulhsu(0x80, 0xff, 0) as u8, 0x80);
    }

    #[test]
    fn carry_and_borrow() {
        let mut e = env(0, 4);
        fill(&mut e, 3, 0, &[0xff, 0x80, 0x01, 0x00]);
        fill(&mut e, 2, 0, &[0x01, 0x7f, 0x01, 0x00]);
        e[voff(0, 0)] = 0x0f;
        let d = Desc { vta_all_1s: true, ..vv(0) };
        call(&VMADC, &mut e, d, 0);
        assert_eq!(e[voff(1, 0)], 0xf1);
        assert_eq!(e[voff(1, 15)], 0xff);
        call(&VMADC, &mut e, Desc { vm: false, ..d }, 0);
        assert_eq!(e[voff(1, 0)], 0xf3);
        call(&VMSBC, &mut e, d, 0);
        assert_eq!(e[voff(1, 0)], 0xf0);
        call(&VMSBC, &mut e, Desc { vm: false, ..d }, 0);
        assert_eq!(e[voff(1, 0)], 0xfc);
        // Without VTA_ALL_1S the tail bits are left alone.
        e[voff(1, 0)] = 0;
        call(&VMSBC, &mut e, Desc { vm: false, vta_all_1s: false, ..d }, 0);
        assert_eq!(e[voff(1, 0)], 0x0c);
        // vadc and vsbc take the carry from v0 whatever vm is.
        e[voff(0, 0)] = 0b0101;
        call(&VADC, &mut e, Desc { vd: 4, ..d }, 0);
        assert_eq!(get(&e, 4, 0, 4), [0x01, 0xff, 0x03, 0x00]);
        call(&VSBC, &mut e, Desc { vd: 4, scalar: true, ..d }, 1);
        assert_eq!(get(&e, 4, 0, 4), [0xfd, 0x7f, 0xff, 0xff]);
    }

    #[test]
    fn compare_mask_and_tail() {
        let mut e = env(1, 4);
        fill(&mut e, 3, 1, &[1, 0xffff, 5, 3]);
        e[voff(0, 0)] = 0b1101;
        e[voff(1, 0)] = 0b0010;
        let d = Desc { vm: false, vma: false, vta_all_1s: true, scalar: true, ..vv(1) };
        // Element 1 is masked off and keeps its bit, which happens to be set.
        call(&VMSLT, &mut e, d, 2);
        assert_eq!(e[voff(1, 0)], 0xf3);
        assert!((1..16).all(|b| e[voff(1, b)] == 0xff));
        e[voff(1, 0)] = 0;
        call(&VMSLT, &mut e, d, 2);
        assert_eq!(e[voff(1, 0)], 0xf1);
        // With vma the masked off bit is set.
        call(&VMSLTU, &mut e, Desc { vma: true, ..d }, 2);
        assert_eq!(e[voff(1, 0)], 0xf3);
        call(&VMSGT, &mut e, Desc { vm: true, ..d }, 1);
        assert_eq!(e[voff(1, 0)], 0xfc);
        call(&VMSEQ, &mut e, Desc { vm: true, ..d }, 0x10005);
        assert_eq!(e[voff(1, 0)], 0xf4);
        fill(&mut e, 2, 1, &[1, 1, 4, 3]);
        call(&VMSLE, &mut e, Desc { vm: true, scalar: false, ..d }, 0);
        assert_eq!(e[voff(1, 0)], 0xfb);
    }

    #[test]
    fn extensions() {
        let mut e = env(2, 2);
        fill(&mut e, 3, 0, &[0x80, 0x7f]);
        let d = Desc { x: 2, ..vv(2) };
        call(&VZEXT, &mut e, d, 0);
        assert_eq!(get(&e, 1, 2, 2), [0x80, 0x7f]);
        call(&VSEXT, &mut e, d, 0);
        assert_eq!(get(&e, 1, 2, 2), [0xffff_ff80, 0x7f]);
        let mut e = env(3, 1);
        fill(&mut e, 3, 0, &[0x80]);
        call(&VSEXT, &mut e, Desc { x: 3, ..vv(3) }, 0);
        assert_eq!(vget(&e, 1, 0, 3), 0xffff_ffff_ffff_ff80);
        fill(&mut e, 3, 2, &[0x8000_0000]);
        call(&VZEXT, &mut e, Desc { x: 1, ..vv(3) }, 0);
        assert_eq!(vget(&e, 1, 0, 3), 0x8000_0000);
    }

    #[test]
    fn masks_and_tails() {
        let mut e = env(0, 3);
        fill(&mut e, 3, 0, &[1, 2, 3]);
        fill(&mut e, 1, 0, &[9; 16]);
        e[voff(0, 0)] = 0b101;
        let d = Desc { vm: false, scalar: true, ..vv(0) };
        // Undisturbed: the masked off element and the tail keep their values.
        call(&VADD, &mut e, d, 0x110);
        assert_eq!(get(&e, 1, 0, 5), [0x11, 9, 0x13, 9, 9]);
        // Agnostic: set to all ones.
        call(&VADD, &mut e, Desc { vma: true, vta: true, ..d }, 0x10);
        assert_eq!(get(&e, 1, 0, 5), [0x11, 0xff, 0x13, 0xff, 0xff]);
        assert_eq!(e[voff(1, 15)], 0xff);
        // vstart past vl: nothing changes but vstart.
        fill(&mut e, 1, 0, &[9; 16]);
        st64(&mut e, VSTART, 3);
        call(&VADD, &mut e, Desc { vta: true, ..d }, 1);
        assert_eq!(get(&e, 1, 0, 4), [9; 4]);
        assert_eq!(ld64(&e, VSTART), 0);
        // vstart 1: element 0 is left alone.
        st64(&mut e, VSTART, 1);
        call(&VADD, &mut e, Desc { vm: true, ..d }, 1);
        assert_eq!(get(&e, 1, 0, 4), [9, 3, 4, 9]);
        // vmacc: vd += s1 * vs2.
        call(&VMACC, &mut e, Desc { vm: true, ..d }, 2);
        assert_eq!(get(&e, 1, 0, 4), [11, 7, 10, 9]);
        call(&VNMSUB, &mut e, Desc { vm: true, ..d }, 1);
        assert_eq!(get(&e, 1, 0, 3), [0xf6, 0xfb, 0xf9]);
        // vmerge ignores vm and vma and takes s1 where v0 is set.
        call(&VMERGE, &mut e, Desc { vma: true, ..d }, 0x55);
        assert_eq!(get(&e, 1, 0, 4), [0x55, 2, 0x55, 9]);
        // vmv.v.x with a tail agnostic tail.
        call(&VMV_V, &mut e, Desc { vta: true, ..d }, 0x1234);
        assert_eq!(get(&e, 1, 0, 4), [0x34, 0x34, 0x34, 0xff]);
        // Shifts take the low log2(SEW) bits of the amount.
        call(&VSLL, &mut e, Desc { vm: true, ..d }, 9);
        assert_eq!(get(&e, 1, 0, 3), [2, 4, 6]);
        call(&VSRA, &mut e, Desc { vm: true, ..d }, 0);
        assert_eq!(get(&e, 1, 0, 3), [1, 2, 3]);
        fill(&mut e, 3, 0, &[0x80]);
        call(&VSRA, &mut e, Desc { vm: true, ..d }, 0x0f);
        assert_eq!(vget(&e, 1, 0, 0), 0xff);
        call(&VSRL, &mut e, Desc { vm: true, ..d }, 7);
        assert_eq!(vget(&e, 1, 0, 0), 1);
    }
}
