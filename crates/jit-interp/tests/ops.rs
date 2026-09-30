// SPDX-License-Identifier: MIT OR Apache-2.0

//! Semantics of every integer op, each checked unoptimized, optimized and constant folded.

mod common;

use common::{Rng, e32, e64, eval32, eval64};
use ruvm_jit_core::Cond;
use ruvm_jit_core::types::bswap;

const CONDS: [Cond; 16] = [
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
    Cond::Lt,
    Cond::Eq,
];

fn cond64(c: Cond, a: u64, b: u64) -> bool {
    let (sa, sb) = (a as i64, b as i64);
    match c {
        Cond::Never => false,
        Cond::Always => true,
        Cond::Lt => sa < sb,
        Cond::Ge => sa >= sb,
        Cond::Gt => sa > sb,
        Cond::Le => sa <= sb,
        Cond::Eq => a == b,
        Cond::Ne => a != b,
        Cond::Ltu => a < b,
        Cond::Geu => a >= b,
        Cond::Gtu => a > b,
        Cond::Leu => a <= b,
        Cond::TstEq => a & b == 0,
        Cond::TstNe => a & b != 0,
    }
}

fn cond32(c: Cond, a: u32, b: u32) -> bool {
    match c {
        Cond::Lt | Cond::Ge | Cond::Gt | Cond::Le => {
            cond64(c, a as i32 as i64 as u64, b as i32 as i64 as u64)
        }
        _ => cond64(c, a as u64, b as u64),
    }
}

const V32: [u32; 10] = [0, 1, 2, 31, 32, 33, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff, 0x1234_5678];
const V64: [u64; 10] = [
    0,
    1,
    63,
    64,
    65,
    0x7fff_ffff_ffff_ffff,
    0x8000_0000_0000_0000,
    u64::MAX,
    0x0123_4567_89ab_cdef,
    0xffff_ffff,
];

macro_rules! bin_test {
    ($name:ident, $g32:ident, $g64:ident, |$a:ident, $b:ident| $r32:expr, $r64:expr) => {
        #[test]
        fn $name() {
            for &$a in &V32 {
                for &$b in &V32 {
                    let got = e32(&[$a, $b], |f, r, x| f.$g32(r, x[0], x[1]));
                    assert_eq!(got, $r32, "{} i32 {:#x} {:#x}", stringify!($name), $a, $b);
                }
            }
            for &$a in &V64 {
                for &$b in &V64 {
                    let got = e64(&[$a, $b], |f, r, x| f.$g64(r, x[0], x[1]));
                    assert_eq!(got, $r64, "{} i64 {:#x} {:#x}", stringify!($name), $a, $b);
                }
            }
        }
    };
}

bin_test!(add, gen_add_i32, gen_add_i64, |a, b| a.wrapping_add(b), a.wrapping_add(b));
bin_test!(sub, gen_sub_i32, gen_sub_i64, |a, b| a.wrapping_sub(b), a.wrapping_sub(b));
bin_test!(mul, gen_mul_i32, gen_mul_i64, |a, b| a.wrapping_mul(b), a.wrapping_mul(b));
bin_test!(and, gen_and_i32, gen_and_i64, |a, b| a & b, a & b);
bin_test!(or, gen_or_i32, gen_or_i64, |a, b| a | b, a | b);
bin_test!(xor, gen_xor_i32, gen_xor_i64, |a, b| a ^ b, a ^ b);
bin_test!(andc, gen_andc_i32, gen_andc_i64, |a, b| a & !b, a & !b);
bin_test!(orc, gen_orc_i32, gen_orc_i64, |a, b| a | !b, a | !b);
bin_test!(eqv, gen_eqv_i32, gen_eqv_i64, |a, b| !(a ^ b), !(a ^ b));
bin_test!(nand, gen_nand_i32, gen_nand_i64, |a, b| !(a & b), !(a & b));
bin_test!(nor, gen_nor_i32, gen_nor_i64, |a, b| !(a | b), !(a | b));
// Shift counts are masked to the width, as the optimizer folds them.
bin_test!(shl, gen_shl_i32, gen_shl_i64, |a, b| a << (b & 31), a << (b & 63));
bin_test!(shr, gen_shr_i32, gen_shr_i64, |a, b| a >> (b & 31), a >> (b & 63));
bin_test!(
    sar,
    gen_sar_i32,
    gen_sar_i64,
    |a, b| ((a as i32) >> (b & 31)) as u32,
    ((a as i64) >> (b & 63)) as u64
);
bin_test!(
    rotl,
    gen_rotl_i32,
    gen_rotl_i64,
    |a, b| a.rotate_left(b & 31),
    a.rotate_left((b & 63) as u32)
);
bin_test!(
    rotr,
    gen_rotr_i32,
    gen_rotr_i64,
    |a, b| a.rotate_right(b & 31),
    a.rotate_right((b & 63) as u32)
);
bin_test!(
    mulsh,
    gen_mulsh_i32,
    gen_mulsh_i64,
    |a, b| ((a as i32 as i64 * b as i32 as i64) >> 32) as u32,
    ((a as i64 as i128 * b as i64 as i128) >> 64) as u64
);
bin_test!(
    muluh,
    gen_muluh_i32,
    gen_muluh_i64,
    |a, b| ((a as u64 * b as u64) >> 32) as u32,
    ((a as u128 * b as u128) >> 64) as u64
);
// Division by zero divides by one, and the signed overflow case wraps, as the folder does.
bin_test!(
    divs,
    gen_div_i32,
    gen_div_i64,
    |a, b| (a as i32).wrapping_div(if b == 0 { 1 } else { b as i32 }) as u32,
    (a as i64).wrapping_div(if b == 0 { 1 } else { b as i64 }) as u64
);
bin_test!(
    rems,
    gen_rem_i32,
    gen_rem_i64,
    |a, b| (a as i32).wrapping_rem(if b == 0 { 1 } else { b as i32 }) as u32,
    (a as i64).wrapping_rem(if b == 0 { 1 } else { b as i64 }) as u64
);
bin_test!(
    divu,
    gen_divu_i32,
    gen_divu_i64,
    |a, b| a / if b == 0 { 1 } else { b },
    a / if b == 0 { 1 } else { b }
);
bin_test!(
    remu,
    gen_remu_i32,
    gen_remu_i64,
    |a, b| a % if b == 0 { 1 } else { b },
    a % if b == 0 { 1 } else { b }
);
bin_test!(
    clz,
    gen_clz_i32,
    gen_clz_i64,
    |a, b| if a == 0 { b } else { a.leading_zeros() },
    if a == 0 { b } else { a.leading_zeros() as u64 }
);
bin_test!(
    ctz,
    gen_ctz_i32,
    gen_ctz_i64,
    |a, b| if a == 0 { b } else { a.trailing_zeros() },
    if a == 0 { b } else { a.trailing_zeros() as u64 }
);
bin_test!(
    smin,
    gen_smin_i32,
    gen_smin_i64,
    |a, b| (a as i32).min(b as i32) as u32,
    (a as i64).min(b as i64) as u64
);
bin_test!(
    smax,
    gen_smax_i32,
    gen_smax_i64,
    |a, b| (a as i32).max(b as i32) as u32,
    (a as i64).max(b as i64) as u64
);
bin_test!(umin, gen_umin_i32, gen_umin_i64, |a, b| a.min(b), a.min(b));
bin_test!(umax, gen_umax_i32, gen_umax_i64, |a, b| a.max(b), a.max(b));

macro_rules! un_test {
    ($name:ident, $g32:ident, $g64:ident, |$a:ident| $r32:expr, $r64:expr) => {
        #[test]
        fn $name() {
            for &$a in &V32 {
                let got = e32(&[$a], |f, r, x| f.$g32(r, x[0]));
                assert_eq!(got, $r32, "{} i32 {:#x}", stringify!($name), $a);
            }
            for &$a in &V64 {
                let got = e64(&[$a], |f, r, x| f.$g64(r, x[0]));
                assert_eq!(got, $r64, "{} i64 {:#x}", stringify!($name), $a);
            }
        }
    };
}

un_test!(neg, gen_neg_i32, gen_neg_i64, |a| a.wrapping_neg(), a.wrapping_neg());
un_test!(not, gen_not_i32, gen_not_i64, |a| !a, !a);
un_test!(ctpop, gen_ctpop_i32, gen_ctpop_i64, |a| a.count_ones(), a.count_ones() as u64);
un_test!(
    abs,
    gen_abs_i32,
    gen_abs_i64,
    |a| (a as i32).wrapping_abs() as u32,
    (a as i64).wrapping_abs() as u64
);
un_test!(
    clrsb,
    gen_clrsb_i32,
    gen_clrsb_i64,
    |a| (if (a as i32) < 0 { !a } else { a }).leading_zeros() - 1,
    (if (a as i64) < 0 { !a } else { a }).leading_zeros() as u64 - 1
);
un_test!(ext8s, gen_ext8s_i32, gen_ext8s_i64, |a| a as i8 as i32 as u32, a as i8 as i64 as u64);
un_test!(ext16s, gen_ext16s_i32, gen_ext16s_i64, |a| a as i16 as i32 as u32, a as i16 as u64);
un_test!(ext8u, gen_ext8u_i32, gen_ext8u_i64, |a| a & 0xff, a & 0xff);
un_test!(ext16u, gen_ext16u_i32, gen_ext16u_i64, |a| a & 0xffff, a & 0xffff);
un_test!(bswap32, gen_bswap32_i32, gen_bswap64_i64, |a| a.swap_bytes(), a.swap_bytes());

#[test]
fn clz_ctz_defaults() {
    // tcg_gen_clzi and ctzi with the width as the default, which is what targets use.
    for &a in &V32 {
        assert_eq!(e32(&[a], |f, r, x| f.gen_clzi_i32(r, x[0], 32)), a.leading_zeros());
        assert_eq!(e32(&[a], |f, r, x| f.gen_ctzi_i32(r, x[0], 32)), a.trailing_zeros());
        assert_eq!(
            e32(&[a], |f, r, x| f.gen_ctzi_i32(r, x[0], 7)),
            if a == 0 { 7 } else { a.trailing_zeros() }
        );
    }
    for &a in &V64 {
        assert_eq!(e64(&[a], |f, r, x| f.gen_clzi_i64(r, x[0], 64)), a.leading_zeros() as u64);
        assert_eq!(e64(&[a], |f, r, x| f.gen_ctzi_i64(r, x[0], 64)), a.trailing_zeros() as u64);
        assert_eq!(
            e64(&[a], |f, r, x| f.gen_ctzi_i64(r, x[0], -1)),
            if a == 0 { u64::MAX } else { a.trailing_zeros() as u64 }
        );
    }
}

#[test]
fn shift_masking_i32() {
    // The upper bits of the count are ignored for i32, even above 32.
    let a = 0x8000_0001u32;
    assert_eq!(e32(&[a, 33], |f, r, x| f.gen_shl_i32(r, x[0], x[1])), 2);
    assert_eq!(e32(&[a, 32], |f, r, x| f.gen_shr_i32(r, x[0], x[1])), a);
    assert_eq!(e32(&[a, 63], |f, r, x| f.gen_sar_i32(r, x[0], x[1])), 0xffff_ffff);
    assert_eq!(e32(&[a, 36], |f, r, x| f.gen_rotl_i32(r, x[0], x[1])), 0x18);
    assert_eq!(e32(&[a, 1], |f, r, x| f.gen_rotr_i32(r, x[0], x[1])), 0xc000_0000);
    assert_eq!(e32(&[a], |f, r, x| f.gen_rotli_i32(r, x[0], 0)), a);
    assert_eq!(e32(&[a], |f, r, x| f.gen_rotri_i32(r, x[0], 4)), 0x1800_0000);
    assert_eq!(e32(&[a], |f, r, x| f.gen_sari_i32(r, x[0], 31)), 0xffff_ffff);
    assert_eq!(e64(&[1], |f, r, x| f.gen_shli_i64(r, x[0], 63)), 1 << 63);
}

#[test]
fn immediates() {
    let mut rng = Rng::new(1);
    for _ in 0..200 {
        let a = rng.interesting();
        let i = rng.interesting() as i64;
        let a32 = a as u32;
        let i32v = i as i32;
        assert_eq!(e64(&[a], |f, r, x| f.gen_addi_i64(r, x[0], i)), a.wrapping_add(i as u64));
        assert_eq!(e64(&[a], |f, r, x| f.gen_subi_i64(r, x[0], i)), a.wrapping_sub(i as u64));
        assert_eq!(e64(&[a], |f, r, x| f.gen_andi_i64(r, x[0], i)), a & i as u64);
        assert_eq!(e64(&[a], |f, r, x| f.gen_ori_i64(r, x[0], i)), a | i as u64);
        assert_eq!(e64(&[a], |f, r, x| f.gen_xori_i64(r, x[0], i)), a ^ i as u64);
        assert_eq!(e64(&[a], |f, r, x| f.gen_muli_i64(r, x[0], i)), a.wrapping_mul(i as u64));
        assert_eq!(
            e32(&[a32], |f, r, x| f.gen_addi_i32(r, x[0], i32v)),
            a32.wrapping_add(i32v as u32)
        );
        assert_eq!(e32(&[a32], |f, r, x| f.gen_andi_i32(r, x[0], i32v)), a32 & i32v as u32);
        assert_eq!(
            e32(&[a32], |f, r, x| f.gen_muli_i32(r, x[0], i32v)),
            a32.wrapping_mul(i32v as u32)
        );
        let s = (i & 31) as i32;
        assert_eq!(e32(&[a32], |f, r, x| f.gen_shli_i32(r, x[0], s)), a32 << s);
        assert_eq!(e32(&[a32], |f, r, x| f.gen_shri_i32(r, x[0], s)), a32 >> s);
        assert_eq!(e32(&[a32], |f, r, x| f.gen_rotli_i32(r, x[0], s)), a32.rotate_left(s as u32));
        let s = i & 63;
        assert_eq!(e64(&[a], |f, r, x| f.gen_sari_i64(r, x[0], s)), ((a as i64) >> s) as u64);
        assert_eq!(e64(&[a], |f, r, x| f.gen_rotri_i64(r, x[0], s)), a.rotate_right(s as u32));
    }
}

#[test]
fn deposit_extract_sextract() {
    let mut rng = Rng::new(2);
    for _ in 0..12 {
        let (a, b) = (rng.next(), rng.next());
        let (a32, b32) = (a as u32, b as u32);
        for ofs in 0..64u32 {
            for len in [1u32, 2, 7, 8, 16, 31, 32, 33, 63, 64] {
                if ofs + len > 64 || rng.below(4) != 0 {
                    continue;
                }
                let m = if len == 64 { !0u64 } else { (1u64 << len) - 1 };
                let dep = (a & !(m << ofs)) | ((b & m) << ofs);
                let ext = (a >> ofs) & m;
                let sext = (((a >> ofs) << (64 - len)) as i64 >> (64 - len)) as u64;
                assert_eq!(e64(&[a, b], |f, r, x| f.gen_deposit_i64(r, x[0], x[1], ofs, len)), dep);
                assert_eq!(e64(&[a], |f, r, x| f.gen_extract_i64(r, x[0], ofs, len)), ext);
                assert_eq!(e64(&[a], |f, r, x| f.gen_sextract_i64(r, x[0], ofs, len)), sext);
                if ofs + len <= 32 {
                    let m = m as u32;
                    let dep = (a32 & !(m << ofs)) | ((b32 & m) << ofs);
                    let ext = (a32 >> ofs) & m;
                    let sext = (((a32 >> ofs) << (32 - len)) as i32 >> (32 - len)) as u32;
                    assert_eq!(
                        e32(&[a32, b32], |f, r, x| f.gen_deposit_i32(r, x[0], x[1], ofs, len)),
                        dep
                    );
                    assert_eq!(e32(&[a32], |f, r, x| f.gen_extract_i32(r, x[0], ofs, len)), ext);
                    assert_eq!(e32(&[a32], |f, r, x| f.gen_sextract_i32(r, x[0], ofs, len)), sext);
                }
            }
        }
        for ofs in 0..32 {
            let want = ((((b32 as u64) << 32) | a32 as u64) >> ofs) as u32;
            assert_eq!(e32(&[a32, b32], |f, r, x| f.gen_extract2_i32(r, x[0], x[1], ofs)), want);
        }
        for ofs in 0..64 {
            let want = ((((b as u128) << 64) | a as u128) >> ofs) as u64;
            assert_eq!(e64(&[a, b], |f, r, x| f.gen_extract2_i64(r, x[0], x[1], ofs)), want);
        }
    }
}

#[test]
fn double_word() {
    let mut rng = Rng::new(3);
    for _ in 0..200 {
        let (a, b) = (rng.interesting(), rng.interesting());
        let (a32, b32) = (a as u32, b as u32);

        let p = a32 as u64 * b32 as u64;
        let r = eval32(2, &[a32, b32], |f, o, x| f.gen_mulu2_i32(o[0], o[1], x[0], x[1]));
        assert_eq!(r, [p as u32, (p >> 32) as u32]);
        let p = (a32 as i32 as i64 * b32 as i32 as i64) as u64;
        let r = eval32(2, &[a32, b32], |f, o, x| f.gen_muls2_i32(o[0], o[1], x[0], x[1]));
        assert_eq!(r, [p as u32, (p >> 32) as u32]);
        let p = a as u128 * b as u128;
        let r = eval64(2, &[a, b], |f, o, x| f.gen_mulu2_i64(o[0], o[1], x[0], x[1]));
        assert_eq!(r, [p as u64, (p >> 64) as u64]);
        let p = (a as i64 as i128 * b as i64 as i128) as u128;
        let r = eval64(2, &[a, b], |f, o, x| f.gen_muls2_i64(o[0], o[1], x[0], x[1]));
        assert_eq!(r, [p as u64, (p >> 64) as u64]);

        let (c, d) = (rng.interesting(), rng.interesting());
        let x = ((b as u128) << 64) | a as u128;
        let y = ((d as u128) << 64) | c as u128;
        let s = x.wrapping_add(y);
        let r =
            eval64(2, &[a, b, c, d], |f, o, i| f.gen_add2_i64(o[0], o[1], i[0], i[1], i[2], i[3]));
        assert_eq!(r, [s as u64, (s >> 64) as u64]);
        let s = x.wrapping_sub(y);
        let r =
            eval64(2, &[a, b, c, d], |f, o, i| f.gen_sub2_i64(o[0], o[1], i[0], i[1], i[2], i[3]));
        assert_eq!(r, [s as u64, (s >> 64) as u64]);

        let (c32, d32) = (c as u32, d as u32);
        let x = ((b32 as u64) << 32) | a32 as u64;
        let y = ((d32 as u64) << 32) | c32 as u64;
        let s = x.wrapping_add(y);
        let r = eval32(2, &[a32, b32, c32, d32], |f, o, i| {
            f.gen_add2_i32(o[0], o[1], i[0], i[1], i[2], i[3])
        });
        assert_eq!(r, [s as u32, (s >> 32) as u32]);
        let s = x.wrapping_sub(y);
        let r = eval32(2, &[a32, b32, c32, d32], |f, o, i| {
            f.gen_sub2_i32(o[0], o[1], i[0], i[1], i[2], i[3])
        });
        assert_eq!(r, [s as u32, (s >> 32) as u32]);

        // addcio: r = a + b + (ci & 1) with the carry out.
        let ci = rng.below(2);
        let s = a as u128 + b as u128 + ci as u128;
        let r = eval64(2, &[a, b, ci], |f, o, i| f.gen_addcio_i64(o[0], o[1], i[0], i[1], i[2]));
        assert_eq!(r, [s as u64, (s >> 64) as u64]);
        let s = a32 as u64 + b32 as u64 + ci;
        let r = eval32(2, &[a32, b32, ci as u32], |f, o, i| {
            f.gen_addcio_i32(o[0], o[1], i[0], i[1], i[2])
        });
        assert_eq!(r, [s as u32, (s >> 32) as u32]);
    }
}

#[test]
fn setcond_negsetcond_movcond() {
    let mut rng = Rng::new(4);
    for _ in 0..60 {
        let (a, b) = (rng.interesting(), rng.interesting());
        let (v1, v2) = (rng.next(), rng.next());
        for c in CONDS {
            let t = cond64(c, a, b);
            assert_eq!(e64(&[a, b], |f, r, x| f.gen_setcond_i64(c, r, x[0], x[1])), t as u64);
            assert_eq!(
                e64(&[a, b], |f, r, x| f.gen_negsetcond_i64(c, r, x[0], x[1])),
                (t as u64).wrapping_neg()
            );
            assert_eq!(
                e64(&[a], |f, r, x| f.gen_setcondi_i64(c, r, x[0], b as i64)),
                t as u64,
                "{c:?} {a:#x} {b:#x}"
            );
            assert_eq!(
                e64(&[a, b, v1, v2], |f, r, x| f.gen_movcond_i64(c, r, x[0], x[1], x[2], x[3])),
                if t { v1 } else { v2 }
            );
            let (a, b, v1, v2) = (a as u32, b as u32, v1 as u32, v2 as u32);
            let t = cond32(c, a, b);
            assert_eq!(e32(&[a, b], |f, r, x| f.gen_setcond_i32(c, r, x[0], x[1])), t as u32);
            assert_eq!(
                e32(&[a], |f, r, x| f.gen_negsetcondi_i32(c, r, x[0], b as i32)),
                (t as u32).wrapping_neg()
            );
            assert_eq!(
                e32(&[a, b, v1, v2], |f, r, x| f.gen_movcond_i32(c, r, x[0], x[1], x[2], x[3])),
                if t { v1 } else { v2 }
            );
        }
    }
}

#[test]
fn brcond() {
    let mut rng = Rng::new(5);
    for _ in 0..12 {
        let (a, b) = (rng.interesting(), rng.interesting());
        for c in CONDS {
            let got = e64(&[a, b], |f, r, x| {
                let l = f.new_label();
                let done = f.new_label();
                f.gen_brcond_i64(c, x[0], x[1], l);
                f.gen_movi_i64(r, 2);
                f.gen_br(done);
                f.gen_set_label(l);
                f.gen_movi_i64(r, 1);
                f.gen_set_label(done);
            });
            assert_eq!(got, if cond64(c, a, b) { 1 } else { 2 }, "{c:?} {a:#x} {b:#x}");
            let (a, b) = (a as u32, b as u32);
            let got = e32(&[a], |f, r, x| {
                let l = f.new_label();
                f.gen_movi_i32(r, 1);
                f.gen_brcondi_i32(c, x[0], b as i32, l);
                f.gen_movi_i32(r, 2);
                f.gen_set_label(l);
            });
            assert_eq!(got, if cond32(c, a, b) { 1 } else { 2 }, "{c:?} {a:#x} {b:#x}");
        }
    }
}

#[test]
fn loop_with_tb_temp() {
    // sum = 0; i = n; do { sum += i; i -= 1 } while (i != 0)
    for n in [1u64, 2, 10, 100] {
        let got = e64(&[n], |f, r, x| {
            let i = f.temp_new_i64();
            let sum = f.temp_new_i64();
            f.gen_mov_i64(i, x[0]);
            f.gen_movi_i64(sum, 0);
            let top = f.new_label();
            f.gen_set_label(top);
            f.gen_add_i64(sum, sum, i);
            f.gen_subi_i64(i, i, 1);
            f.gen_brcondi_i64(Cond::Ne, i, 0, top);
            f.gen_mov_i64(r, sum);
        });
        assert_eq!(got, n * (n + 1) / 2);
    }
}

#[test]
fn bswap_flags() {
    let a = 0x1234_80f1u32;
    let swapped = 0xf180u32;
    assert_eq!(e32(&[a], |f, r, x| f.gen_bswap16_i32(r, x[0], bswap::OZ)), swapped);
    assert_eq!(
        e32(&[a], |f, r, x| f.gen_bswap16_i32(r, x[0], bswap::OS)),
        swapped as u16 as i16 as i32 as u32
    );
    assert_eq!(
        e32(&[a & 0xffff], |f, r, x| f.gen_bswap16_i32(r, x[0], bswap::IZ | bswap::OZ)),
        swapped
    );
    let b = 0x1122_3344_8899_aabbu64;
    assert_eq!(e64(&[b], |f, r, x| f.gen_bswap16_i64(r, x[0], bswap::OZ)), 0xbbaa);
    assert_eq!(e64(&[b], |f, r, x| f.gen_bswap16_i64(r, x[0], bswap::OS)), 0xbbaau16 as i16 as u64);
    assert_eq!(e64(&[b], |f, r, x| f.gen_bswap32_i64(r, x[0], bswap::OZ)), 0xbbaa_9988);
    assert_eq!(
        e64(&[b], |f, r, x| f.gen_bswap32_i64(r, x[0], bswap::OS)),
        0xbbaa_9988u32 as i32 as u64
    );
    assert_eq!(e64(&[b], |f, r, x| f.gen_bswap64_i64(r, x[0])), 0xbbaa_9988_4433_2211);
}

#[test]
fn width_conversions() {
    for &a in &V64 {
        let lo = a as u32;
        let hi = (a >> 32) as u32;
        let got = eval32(2, &[], |f, o, _| {
            let t = f.constant_i64(a as i64);
            f.gen_extrl_i64_i32(o[0], t);
            f.gen_extrh_i64_i32(o[1], t);
        });
        assert_eq!(got, [lo, hi]);
        assert_eq!(e64(&[a], |f, r, x| f.gen_ext32s_i64(r, x[0])), lo as i32 as i64 as u64);
        assert_eq!(e64(&[a], |f, r, x| f.gen_ext32u_i64(r, x[0])), lo as u64);
        let got = e64(&[], |f, r, _| {
            let t = f.temp_new_i32();
            f.gen_movi_i32(t, lo as i32);
            f.gen_ext_i32_i64(r, t);
        });
        assert_eq!(got, lo as i32 as i64 as u64);
        let got = e64(&[], |f, r, _| {
            let t = f.temp_new_i32();
            f.gen_movi_i32(t, lo as i32);
            f.gen_extu_i32_i64(r, t);
        });
        assert_eq!(got, lo as u64);
        let got = e64(&[], |f, r, _| {
            let l = f.constant_i32(lo as i32);
            let h = f.constant_i32(hi as i32);
            f.gen_concat_i32_i64(r, l, h);
        });
        assert_eq!(got, a);
    }
}

#[test]
fn host_loads_and_stores() {
    // ld and st on env offsets, with every width and extension.
    let v = 0x8182_8384_8586_8788u64;
    let got = eval64(6, &[v], |f, o, x| {
        let env = f.env();
        f.gen_st_i64(x[0], env, 0x300);
        f.gen_ld8u_i64(o[0], env, 0x300);
        f.gen_ld8s_i64(o[1], env, 0x300);
        f.gen_ld16s_i64(o[2], env, 0x300);
        f.gen_ld32u_i64(o[3], env, 0x300);
        f.gen_ld32s_i64(o[4], env, 0x304);
        let t = f.temp_new_i64();
        f.gen_movi_i64(t, 0x55);
        f.gen_st8_i64(t, env, 0x301);
        f.gen_ld_i64(o[5], env, 0x300);
    });
    assert_eq!(
        got,
        [
            0x88,
            0xffff_ffff_ffff_ff88,
            0xffff_ffff_ffff_8788,
            0x8586_8788,
            0xffff_ffff_8182_8384,
            0x8182_8384_8586_5588
        ]
    );
}
