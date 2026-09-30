// SPDX-License-Identifier: GPL-2.0-or-later

//! Multi word fraction arithmetic, the `frac64_*`, `frac128_*` and `frac256_*` helpers of
//! `fpu/softfloat.c` and the parts of `include/fpu/softfloat-macros.h` they use.
//!
//! A fraction is one, two or four 64 bit words, most significant first, with the binary point
//! after bit 63 of the top word. [`Frac`] is the operation set every size has; [`FracN`] adds
//! the ones that need a double width type, which only the 64 and 128 bit sizes have.

use core::cmp::Ordering;

/// `DECOMPOSED_IMPLICIT_BIT`: the integer bit of a canonical fraction.
pub(crate) const IMPLICIT_BIT: u64 = 1 << 63;

/// `shl_double`: shift `l` left by `c`, filling from `r`. A count of 0 returns `l`.
#[inline]
pub(crate) fn shl_double(l: u64, r: u64, c: u32) -> u64 {
    if c == 0 { l } else { (l << c) | (r >> (64 - c)) }
}

/// `shr_double`: shift `r` right by `c`, filling from `l`. A count of 0 returns `r`.
#[inline]
pub(crate) fn shr_double(l: u64, r: u64, c: u32) -> u64 {
    if c == 0 { r } else { (r >> c) | (l << (64 - c)) }
}

/// `mul64To128`: the full product, as (high, low).
#[inline]
pub(crate) fn mul64_to_128(a: u64, b: u64) -> (u64, u64) {
    let p = u128::from(a) * u128::from(b);
    ((p >> 64) as u64, p as u64)
}

/// `add128`.
#[inline]
pub(crate) fn add128(a0: u64, a1: u64, b0: u64, b1: u64) -> (u64, u64) {
    let r = ((u128::from(a0) << 64) | u128::from(a1))
        .wrapping_add((u128::from(b0) << 64) | u128::from(b1));
    ((r >> 64) as u64, r as u64)
}

/// `sub128`.
#[inline]
pub(crate) fn sub128(a0: u64, a1: u64, b0: u64, b1: u64) -> (u64, u64) {
    let r = ((u128::from(a0) << 64) | u128::from(a1))
        .wrapping_sub((u128::from(b0) << 64) | u128::from(b1));
    ((r >> 64) as u64, r as u64)
}

/// `add192`.
#[inline]
pub(crate) fn add192(a: (u64, u64, u64), b: (u64, u64, u64)) -> (u64, u64, u64) {
    let (z2, c2) = a.2.overflowing_add(b.2);
    let (z1, c1a) = a.1.overflowing_add(b.1);
    let (z1, c1b) = z1.overflowing_add(u64::from(c2));
    let z0 = a.0.wrapping_add(b.0).wrapping_add(u64::from(c1a | c1b));
    (z0, z1, z2)
}

/// `sub192`.
#[inline]
pub(crate) fn sub192(a: (u64, u64, u64), b: (u64, u64, u64)) -> (u64, u64, u64) {
    let (z2, b2) = a.2.overflowing_sub(b.2);
    let (z1, b1a) = a.1.overflowing_sub(b.1);
    let (z1, b1b) = z1.overflowing_sub(u64::from(b2));
    let z0 = a.0.wrapping_sub(b.0).wrapping_sub(u64::from(b1a | b1b));
    (z0, z1, z2)
}

/// `mul128By64To192`.
#[inline]
pub(crate) fn mul128_by_64_to_192(a0: u64, a1: u64, b: u64) -> (u64, u64, u64) {
    let (m1, z2) = mul64_to_128(a1, b);
    let (z0, z1) = mul64_to_128(a0, b);
    let (z0, z1) = add128(z0, z1, 0, m1);
    (z0, z1, z2)
}

/// `mul128To256`: the full product of two 128 bit numbers, most significant word first.
#[inline]
pub(crate) fn mul128_to_256(a0: u64, a1: u64, b0: u64, b1: u64) -> (u64, u64, u64, u64) {
    let (m1, m2) = mul64_to_128(a1, b0);
    let (n1, n2) = mul64_to_128(a0, b1);
    let (z2, z3) = mul64_to_128(a1, b1);
    let (z0, z1) = mul64_to_128(a0, b0);
    let m = add192((0, m1, m2), (0, n1, n2));
    let z = add192(m, (z0, z1, z2));
    (z.0, z.1, z.2, z3)
}

/// `estimateDiv128To64`: an estimate of `(a0:a1) / b`, at most 2 too big. `b` must have its
/// top bit set.
pub(crate) fn estimate_div_128_to_64(a0: u64, a1: u64, b: u64) -> u64 {
    if b <= a0 {
        return u64::MAX;
    }
    let b0 = b >> 32;
    let mut z = if b0 << 32 <= a0 { 0xFFFF_FFFF_0000_0000 } else { (a0 / b0) << 32 };
    let (term0, term1) = mul64_to_128(b, z);
    let (mut rem0, mut rem1) = sub128(a0, a1, term0, term1);
    while (rem0 as i64) < 0 {
        z = z.wrapping_sub(0x1_0000_0000);
        let b1 = b << 32;
        (rem0, rem1) = add128(rem0, rem1, b0, b1);
    }
    rem0 = (rem0 << 32) | (rem1 >> 32);
    z |= if b0 << 32 <= rem0 { 0xFFFF_FFFF } else { rem0 / b0 };
    z
}

/// `shortShift128Left`, for counts 0 to 63.
#[inline]
pub(crate) fn short_shift128_left(a0: u64, a1: u64, count: u32) -> (u64, u64) {
    let z1 = a1 << count;
    let z0 = if count == 0 { a0 } else { (a0 << count) | (a1 >> ((64 - count) & 63)) };
    (z0, z1)
}

/// `shortShift192Left`, for counts 0 to 63.
#[inline]
pub(crate) fn short_shift192_left(a0: u64, a1: u64, a2: u64, count: u32) -> (u64, u64, u64) {
    let z2 = a2 << count;
    let mut z1 = a1 << count;
    let mut z0 = a0 << count;
    if count > 0 {
        let neg = (64 - count) & 63;
        z1 |= a2 >> neg;
        z0 |= a1 >> neg;
    }
    (z0, z1, z2)
}

/// `shift128Right`.
#[inline]
pub(crate) fn shift128_right(a0: u64, a1: u64, count: u32) -> (u64, u64) {
    if count == 0 {
        (a0, a1)
    } else if count < 64 {
        (a0 >> count, (a0 << ((64 - count) & 63)) | (a1 >> count))
    } else {
        (0, if count < 128 { a0 >> (count & 63) } else { 0 })
    }
}

/// `shift128Left`.
#[inline]
pub(crate) fn shift128_left(a0: u64, a1: u64, count: u32) -> (u64, u64) {
    if count < 64 { short_shift128_left(a0, a1, count) } else { (a1 << (count - 64), 0) }
}

#[inline]
fn lt128(a0: u64, a1: u64, b0: u64, b1: u64) -> bool {
    a0 < b0 || (a0 == b0 && a1 < b1)
}

#[inline]
fn le128(a0: u64, a1: u64, b0: u64, b1: u64) -> bool {
    a0 < b0 || (a0 == b0 && a1 <= b1)
}

#[inline]
fn lt192(a: (u64, u64, u64), b: (u64, u64, u64)) -> bool {
    a < b
}

#[inline]
fn le192(a: (u64, u64, u64), b: (u64, u64, u64)) -> bool {
    a <= b
}

/// The operations every fraction size has.
pub(crate) trait Frac: Copy + Default + core::fmt::Debug {
    /// The width in bits.
    const N: i32;
    /// The most significant word.
    fn hi(&self) -> u64;
    /// Set the most significant word.
    fn set_hi(&mut self, v: u64);
    /// The least significant word. For a one word fraction this is the same word as `hi`.
    fn lo(&self) -> u64;
    /// Set the least significant word.
    fn set_lo(&mut self, v: u64);
    /// `fracN_add`: the sum and the carry out.
    fn add(&self, b: &Self) -> (Self, bool);
    /// `fracN_addi`: add a word at the bottom, returning the carry out.
    fn addi(&self, c: u64) -> (Self, bool);
    /// `fracN_sub`: the difference and the borrow out.
    fn sub(&self, b: &Self) -> (Self, bool);
    /// `fracN_neg`.
    fn neg(&mut self);
    /// `fracN_normalize`: shift left until the top bit is set, returning the count, or `N` for
    /// zero.
    fn normalize(&mut self) -> i32;
    /// `fracN_shl`, for counts below `N`.
    fn shl(&mut self, c: i32);
    /// `fracN_shr`, for counts below `N`.
    fn shr(&mut self, c: i32);
    /// `fracN_shrjam`: shift right, or-ing every bit shifted out into the lowest bit.
    fn shrjam(&mut self, c: i32);
    /// `fracN_eqz`.
    fn eqz(&self) -> bool;
    /// `fracN_cmp`, as -1, 0 or 1.
    fn cmp(&self, b: &Self) -> i32;
    /// `fracN_allones`.
    fn allones(&mut self);
    /// The canonical default NaN fraction for this size from the 64 bit one.
    fn from_dnan(frac: u64) -> Self;
}

/// The operations that need a double width fraction.
pub(crate) trait FracN: Frac {
    /// The double width fraction.
    type W: Frac;
    /// `fracN_mulw`: the full product.
    fn mulw(&self, b: &Self) -> Self::W;
    /// `fracN_widen`: zero extend below.
    fn widen(&self) -> Self::W;
    /// `fracN_truncjam`: keep the top half, jamming the rest into the lowest bit.
    fn truncjam(w: &Self::W) -> Self;
    /// `fracN_div`: divide in place, returning true if the dividend fraction was the smaller
    /// one, which the caller takes off the exponent.
    fn div(&mut self, b: &Self) -> bool;
}

/// A one word fraction, `FloatParts64`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Frac64(pub u64);

/// A two word fraction, `FloatParts128`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Frac128 {
    pub hi: u64,
    pub lo: u64,
}

/// A four word fraction, `FloatParts256`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Frac256 {
    pub hi: u64,
    pub hm: u64,
    pub lm: u64,
    pub lo: u64,
}

fn ord(o: Ordering) -> i32 {
    match o {
        Ordering::Less => -1,
        Ordering::Equal => 0,
        Ordering::Greater => 1,
    }
}

impl Frac for Frac64 {
    const N: i32 = 64;
    #[inline]
    fn hi(&self) -> u64 {
        self.0
    }
    #[inline]
    fn set_hi(&mut self, v: u64) {
        self.0 = v;
    }
    #[inline]
    fn lo(&self) -> u64 {
        self.0
    }
    #[inline]
    fn set_lo(&mut self, v: u64) {
        self.0 = v;
    }
    #[inline]
    fn add(&self, b: &Self) -> (Self, bool) {
        let (r, c) = self.0.overflowing_add(b.0);
        (Frac64(r), c)
    }
    #[inline]
    fn addi(&self, c: u64) -> (Self, bool) {
        let (r, c) = self.0.overflowing_add(c);
        (Frac64(r), c)
    }
    #[inline]
    fn sub(&self, b: &Self) -> (Self, bool) {
        let (r, c) = self.0.overflowing_sub(b.0);
        (Frac64(r), c)
    }
    #[inline]
    fn neg(&mut self) {
        self.0 = self.0.wrapping_neg();
    }
    #[inline]
    fn normalize(&mut self) -> i32 {
        if self.0 != 0 {
            let shift = self.0.leading_zeros();
            self.0 <<= shift;
            shift as i32
        } else {
            64
        }
    }
    #[inline]
    fn shl(&mut self, c: i32) {
        self.0 <<= c;
    }
    #[inline]
    fn shr(&mut self, c: i32) {
        self.0 >>= c;
    }
    #[inline]
    fn shrjam(&mut self, c: i32) {
        if c != 0 {
            let a0 = self.0;
            self.0 = if c < 64 {
                (a0 >> c) | u64::from(shr_double(a0, 0, c as u32) != 0)
            } else {
                u64::from(a0 != 0)
            };
        }
    }
    #[inline]
    fn eqz(&self) -> bool {
        self.0 == 0
    }
    #[inline]
    fn cmp(&self, b: &Self) -> i32 {
        ord(self.0.cmp(&b.0))
    }
    #[inline]
    fn allones(&mut self) {
        self.0 = u64::MAX;
    }
    #[inline]
    fn from_dnan(frac: u64) -> Self {
        Frac64(frac)
    }
}

impl FracN for Frac64 {
    type W = Frac128;
    #[inline]
    fn mulw(&self, b: &Self) -> Frac128 {
        let (hi, lo) = mul64_to_128(self.0, b.0);
        Frac128 { hi, lo }
    }
    #[inline]
    fn widen(&self) -> Frac128 {
        Frac128 { hi: self.0, lo: 0 }
    }
    #[inline]
    fn truncjam(w: &Frac128) -> Self {
        Frac64(w.hi | u64::from(w.lo != 0))
    }
    #[inline]
    fn div(&mut self, b: &Self) -> bool {
        let a = self.0;
        let ret = a < b.0;
        let n = if ret {
            u128::from(a) << 64
        } else {
            (u128::from(a >> 1) << 64) | u128::from(a << 63)
        };
        let d = u128::from(b.0);
        let q = (n / d) as u64;
        let r = n % d;
        self.0 = q | u64::from(r != 0);
        ret
    }
}

impl Frac for Frac128 {
    const N: i32 = 128;
    #[inline]
    fn hi(&self) -> u64 {
        self.hi
    }
    #[inline]
    fn set_hi(&mut self, v: u64) {
        self.hi = v;
    }
    #[inline]
    fn lo(&self) -> u64 {
        self.lo
    }
    #[inline]
    fn set_lo(&mut self, v: u64) {
        self.lo = v;
    }
    #[inline]
    fn add(&self, b: &Self) -> (Self, bool) {
        let (lo, c0) = self.lo.overflowing_add(b.lo);
        let (hi, c1) = self.hi.overflowing_add(b.hi);
        let (hi, c2) = hi.overflowing_add(u64::from(c0));
        (Frac128 { hi, lo }, c1 | c2)
    }
    #[inline]
    fn addi(&self, c: u64) -> (Self, bool) {
        let (lo, c0) = self.lo.overflowing_add(c);
        let (hi, c1) = self.hi.overflowing_add(u64::from(c0));
        (Frac128 { hi, lo }, c1)
    }
    #[inline]
    fn sub(&self, b: &Self) -> (Self, bool) {
        let (lo, b0) = self.lo.overflowing_sub(b.lo);
        let (hi, b1) = self.hi.overflowing_sub(b.hi);
        let (hi, b2) = hi.overflowing_sub(u64::from(b0));
        (Frac128 { hi, lo }, b1 | b2)
    }
    #[inline]
    fn neg(&mut self) {
        let (r, _) = Frac128::default().sub(self);
        *self = r;
    }
    #[inline]
    fn normalize(&mut self) -> i32 {
        if self.hi != 0 {
            let shl = self.hi.leading_zeros();
            self.hi = shl_double(self.hi, self.lo, shl);
            self.lo <<= shl;
            shl as i32
        } else if self.lo != 0 {
            let shl = self.lo.leading_zeros();
            self.hi = self.lo << shl;
            self.lo = 0;
            shl as i32 + 64
        } else {
            128
        }
    }
    #[inline]
    fn shl(&mut self, c: i32) {
        let (mut a0, mut a1) = (self.hi, self.lo);
        if c & 64 != 0 {
            a0 = a1;
            a1 = 0;
        }
        let c = (c & 63) as u32;
        if c != 0 {
            a0 = shl_double(a0, a1, c);
            a1 <<= c;
        }
        self.hi = a0;
        self.lo = a1;
    }
    #[inline]
    fn shr(&mut self, c: i32) {
        let (mut a0, mut a1) = (self.hi, self.lo);
        if c & 64 != 0 {
            a1 = a0;
            a0 = 0;
        }
        let c = (c & 63) as u32;
        if c != 0 {
            a1 = shr_double(a0, a1, c);
            a0 >>= c;
        }
        self.hi = a0;
        self.lo = a1;
    }
    fn shrjam(&mut self, c: i32) {
        let (mut a0, mut a1) = (self.hi, self.lo);
        let mut sticky = 0u64;
        if c == 0 {
            return;
        }
        let mut c = c;
        let mut shift = true;
        if c < 64 {
            // Nothing to do before the shift.
        } else if c < 128 {
            sticky = a1;
            a1 = a0;
            a0 = 0;
            c &= 63;
            if c == 0 {
                shift = false;
            }
        } else {
            sticky = a0 | a1;
            a0 = 0;
            a1 = 0;
            shift = false;
        }
        if shift {
            let c = c as u32;
            sticky |= shr_double(a1, 0, c);
            a1 = shr_double(a0, a1, c);
            a0 >>= c;
        }
        self.lo = a1 | u64::from(sticky != 0);
        self.hi = a0;
    }
    #[inline]
    fn eqz(&self) -> bool {
        (self.hi | self.lo) == 0
    }
    #[inline]
    fn cmp(&self, b: &Self) -> i32 {
        ord((self.hi, self.lo).cmp(&(b.hi, b.lo)))
    }
    #[inline]
    fn allones(&mut self) {
        self.hi = u64::MAX;
        self.lo = u64::MAX;
    }
    #[inline]
    fn from_dnan(frac: u64) -> Self {
        Frac128 { hi: frac, lo: (frac & 1).wrapping_neg() }
    }
}

impl FracN for Frac128 {
    type W = Frac256;
    #[inline]
    fn mulw(&self, b: &Self) -> Frac256 {
        let (hi, hm, lm, lo) = mul128_to_256(self.hi, self.lo, b.hi, b.lo);
        Frac256 { hi, hm, lm, lo }
    }
    #[inline]
    fn widen(&self) -> Frac256 {
        Frac256 { hi: self.hi, hm: self.lo, lm: 0, lo: 0 }
    }
    #[inline]
    fn truncjam(w: &Frac256) -> Self {
        Frac128 { hi: w.hi, lo: w.hm | u64::from((w.lm | w.lo) != 0) }
    }
    fn div(&mut self, b: &Self) -> bool {
        let (mut a0, mut a1) = (self.hi, self.lo);
        let (b0, b1) = (b.hi, b.lo);
        let ret = lt128(a0, a1, b0, b1);
        if !ret {
            a1 = shr_double(a0, a1, 1);
            a0 >>= 1;
        }
        let mut q0 = estimate_div_128_to_64(a0, a1, b0);
        let t = mul128_by_64_to_192(b0, b1, q0);
        let (mut r0, mut r1, mut r2) = sub192((a0, a1, 0), t);
        while r0 != 0 {
            q0 = q0.wrapping_sub(1);
            (r0, r1, r2) = add192((r0, r1, r2), (0, b0, b1));
        }
        let mut q1 = estimate_div_128_to_64(r1, r2, b0);
        let t = mul128_by_64_to_192(b0, b1, q1);
        let (mut r1, mut r2, mut r3) = sub192((r1, r2, 0), t);
        while r1 != 0 {
            q1 = q1.wrapping_sub(1);
            (r1, r2, r3) = add192((r1, r2, r3), (0, b0, b1));
        }
        q1 |= u64::from((r2 | r3) != 0);
        self.hi = q0;
        self.lo = q1;
        ret
    }
}

impl Frac for Frac256 {
    const N: i32 = 256;
    #[inline]
    fn hi(&self) -> u64 {
        self.hi
    }
    #[inline]
    fn set_hi(&mut self, v: u64) {
        self.hi = v;
    }
    #[inline]
    fn lo(&self) -> u64 {
        self.lo
    }
    #[inline]
    fn set_lo(&mut self, v: u64) {
        self.lo = v;
    }
    fn add(&self, b: &Self) -> (Self, bool) {
        let (lo, c0) = self.lo.overflowing_add(b.lo);
        let (lm, c1) = carry_add(self.lm, b.lm, c0);
        let (hm, c2) = carry_add(self.hm, b.hm, c1);
        let (hi, c3) = carry_add(self.hi, b.hi, c2);
        (Frac256 { hi, hm, lm, lo }, c3)
    }
    fn addi(&self, c: u64) -> (Self, bool) {
        self.add(&Frac256 { hi: 0, hm: 0, lm: 0, lo: c })
    }
    fn sub(&self, b: &Self) -> (Self, bool) {
        let (lo, c0) = self.lo.overflowing_sub(b.lo);
        let (lm, c1) = borrow_sub(self.lm, b.lm, c0);
        let (hm, c2) = borrow_sub(self.hm, b.hm, c1);
        let (hi, c3) = borrow_sub(self.hi, b.hi, c2);
        (Frac256 { hi, hm, lm, lo }, c3)
    }
    fn neg(&mut self) {
        let (r, _) = Frac256::default().sub(self);
        *self = r;
    }
    fn normalize(&mut self) -> i32 {
        let (mut a0, mut a1, mut a2, mut a3) = (self.hi, self.hm, self.lm, self.lo);
        let mut ret;
        let shl;
        if a0 != 0 {
            shl = a0.leading_zeros();
            if shl == 0 {
                return 0;
            }
            ret = shl as i32;
        } else {
            if a1 != 0 {
                ret = 64;
                (a0, a1, a2, a3) = (a1, a2, a3, 0);
            } else if a2 != 0 {
                ret = 128;
                (a0, a1, a2, a3) = (a2, a3, 0, 0);
            } else if a3 != 0 {
                ret = 192;
                (a0, a1, a2, a3) = (a3, 0, 0, 0);
            } else {
                *self = Frac256::default();
                return 256;
            }
            shl = a0.leading_zeros();
            if shl == 0 {
                *self = Frac256 { hi: a0, hm: a1, lm: a2, lo: a3 };
                return ret;
            }
            ret += shl as i32;
        }
        a0 = shl_double(a0, a1, shl);
        a1 = shl_double(a1, a2, shl);
        a2 = shl_double(a2, a3, shl);
        a3 <<= shl;
        *self = Frac256 { hi: a0, hm: a1, lm: a2, lo: a3 };
        ret
    }
    fn shl(&mut self, c: i32) {
        let (mut a0, mut a1, mut a2, mut a3) = (self.hi, self.hm, self.lm, self.lo);
        let mut c = c;
        while c >= 64 {
            (a0, a1, a2, a3) = (a1, a2, a3, 0);
            c -= 64;
        }
        let c = c as u32;
        *self = Frac256 {
            hi: shl_double(a0, a1, c),
            hm: shl_double(a1, a2, c),
            lm: shl_double(a2, a3, c),
            lo: a3 << c,
        };
    }
    fn shr(&mut self, c: i32) {
        let (mut a0, mut a1, mut a2, mut a3) = (self.hi, self.hm, self.lm, self.lo);
        let mut c = c;
        while c >= 64 {
            (a0, a1, a2, a3) = (0, a0, a1, a2);
            c -= 64;
        }
        let c = c as u32;
        *self = Frac256 {
            hi: a0 >> c,
            hm: shr_double(a0, a1, c),
            lm: shr_double(a1, a2, c),
            lo: shr_double(a2, a3, c),
        };
    }
    fn shrjam(&mut self, c: i32) {
        let (mut a0, mut a1, mut a2, mut a3) = (self.hi, self.hm, self.lm, self.lo);
        let mut sticky = 0u64;
        if c == 0 {
            return;
        }
        let mut c = c;
        let mut shift = true;
        if c < 64 {
            // Nothing to do before the shift.
        } else if c < 256 {
            if c & 128 != 0 {
                sticky |= a2 | a3;
                (a3, a2, a1, a0) = (a1, a0, 0, 0);
            }
            if c & 64 != 0 {
                sticky |= a3;
                (a3, a2, a1, a0) = (a2, a1, a0, 0);
            }
            c &= 63;
            if c == 0 {
                shift = false;
            }
        } else {
            sticky = a0 | a1 | a2 | a3;
            (a0, a1, a2, a3) = (0, 0, 0, 0);
            shift = false;
        }
        if shift {
            let c = c as u32;
            sticky |= shr_double(a3, 0, c);
            a3 = shr_double(a2, a3, c);
            a2 = shr_double(a1, a2, c);
            a1 = shr_double(a0, a1, c);
            a0 >>= c;
        }
        self.lo = a3 | u64::from(sticky != 0);
        self.lm = a2;
        self.hm = a1;
        self.hi = a0;
    }
    fn eqz(&self) -> bool {
        (self.hi | self.hm | self.lm | self.lo) == 0
    }
    fn cmp(&self, b: &Self) -> i32 {
        ord((self.hi, self.hm, self.lm, self.lo).cmp(&(b.hi, b.hm, b.lm, b.lo)))
    }
    fn allones(&mut self) {
        *self = Frac256 { hi: u64::MAX, hm: u64::MAX, lm: u64::MAX, lo: u64::MAX };
    }
    fn from_dnan(frac: u64) -> Self {
        let lo = (frac & 1).wrapping_neg();
        Frac256 { hi: frac, hm: lo, lm: lo, lo }
    }
}

#[inline]
fn carry_add(a: u64, b: u64, c: bool) -> (u64, bool) {
    let (r, c1) = a.overflowing_add(b);
    let (r, c2) = r.overflowing_add(u64::from(c));
    (r, c1 | c2)
}

#[inline]
fn borrow_sub(a: u64, b: u64, c: bool) -> (u64, bool) {
    let (r, c1) = a.overflowing_sub(b);
    let (r, c2) = r.overflowing_sub(u64::from(c));
    (r, c1 | c2)
}

/// `frac64_modrem`. Returns the new (sign, exp, frac, is_zero); the quotient goes to
/// `mod_quot` when it is given, which also selects fmod over IEEE remainder.
pub(crate) fn frac64_modrem(
    a_sign: &mut bool,
    a_exp: &mut i32,
    a_frac: &mut u64,
    b_exp: i32,
    b_frac: u64,
    mod_quot: Option<&mut u64>,
) -> bool {
    let mut exp_diff = *a_exp - b_exp;
    let mut a0 = *a_frac;
    let mut a1 = 0u64;
    let (mut t0, mut t1);
    let mut q;
    let mut quot;

    if exp_diff < -1 {
        if let Some(m) = mod_quot {
            *m = 0;
        }
        return false;
    }
    if exp_diff == -1 {
        a0 >>= 1;
        exp_diff = 0;
    }

    let b0 = b_frac;
    q = u64::from(b0 <= a0);
    quot = q;
    if q != 0 {
        a0 = a0.wrapping_sub(b0);
    }

    exp_diff -= 64;
    while exp_diff > 0 {
        q = estimate_div_128_to_64(a0, a1, b0);
        q = q.saturating_sub(2);
        (t0, t1) = mul64_to_128(b0, q);
        (a0, a1) = sub128(a0, a1, t0, t1);
        (a0, a1) = short_shift128_left(a0, a1, 62);
        exp_diff -= 62;
        quot = (quot << 62).wrapping_add(q);
    }

    exp_diff += 64;
    if exp_diff > 0 {
        let sh = (64 - exp_diff) as u32;
        q = estimate_div_128_to_64(a0, a1, b0);
        q = if q > 2 { (q - 2) >> sh } else { 0 };
        (t0, t1) = mul64_to_128(b0, q << sh);
        (a0, a1) = sub128(a0, a1, t0, t1);
        (t0, t1) = short_shift128_left(0, b0, sh);
        while le128(t0, t1, a0, a1) {
            q = q.wrapping_add(1);
            (a0, a1) = sub128(a0, a1, t0, t1);
        }
        quot = (if exp_diff < 64 { quot << exp_diff } else { 0 }).wrapping_add(q);
    } else {
        t0 = b0;
        t1 = 0;
    }

    if let Some(m) = mod_quot {
        *m = quot;
    } else {
        (t0, t1) = sub128(t0, t1, a0, a1);
        if lt128(t0, t1, a0, a1) || ((t0, t1) == (a0, a1) && (q & 1) != 0) {
            a0 = t0;
            a1 = t1;
            *a_sign = !*a_sign;
        }
    }

    let shift;
    if a0 != 0 {
        shift = a0.leading_zeros();
        (a0, a1) = short_shift128_left(a0, a1, shift);
    } else if a1 != 0 {
        let s = a1.leading_zeros();
        a0 = a1 << s;
        a1 = 0;
        shift = s + 64;
    } else {
        return true;
    }

    *a_exp = b_exp + exp_diff - shift as i32;
    *a_frac = a0 | u64::from(a1 != 0);
    false
}

/// `frac128_modrem`, like [`frac64_modrem`] for two word fractions.
pub(crate) fn frac128_modrem(
    a_sign: &mut bool,
    a_exp: &mut i32,
    a_frac: &mut Frac128,
    b_exp: i32,
    b_frac: Frac128,
    mod_quot: Option<&mut u64>,
) -> bool {
    let mut exp_diff = *a_exp - b_exp;
    let mut a0 = a_frac.hi;
    let mut a1 = a_frac.lo;
    let mut a2 = 0u64;
    let (mut t0, mut t1, mut t2);
    let mut q;
    let mut quot;

    if exp_diff < -1 {
        if let Some(m) = mod_quot {
            *m = 0;
        }
        return false;
    }
    if exp_diff == -1 {
        (a0, a1) = shift128_right(a0, a1, 1);
        exp_diff = 0;
    }

    let b0 = b_frac.hi;
    let b1 = b_frac.lo;

    q = u64::from(le128(b0, b1, a0, a1));
    quot = q;
    if q != 0 {
        (a0, a1) = sub128(a0, a1, b0, b1);
    }

    exp_diff -= 64;
    while exp_diff > 0 {
        q = estimate_div_128_to_64(a0, a1, b0);
        q = q.saturating_sub(4);
        let t = mul128_by_64_to_192(b0, b1, q);
        (a0, a1, a2) = sub192((a0, a1, a2), t);
        (a0, a1, a2) = short_shift192_left(a0, a1, a2, 61);
        exp_diff -= 61;
        quot = (quot << 61).wrapping_add(q);
    }

    exp_diff += 64;
    if exp_diff > 0 {
        let sh = (64 - exp_diff) as u32;
        q = estimate_div_128_to_64(a0, a1, b0);
        q = if q > 4 { (q - 4) >> sh } else { 0 };
        let t = mul128_by_64_to_192(b0, b1, q << sh);
        (a0, a1, a2) = sub192((a0, a1, a2), t);
        (t0, t1, t2) = short_shift192_left(0, b0, b1, sh);
        while le192((t0, t1, t2), (a0, a1, a2)) {
            q = q.wrapping_add(1);
            (a0, a1, a2) = sub192((a0, a1, a2), (t0, t1, t2));
        }
        quot = (if exp_diff < 64 { quot << exp_diff } else { 0 }).wrapping_add(q);
    } else {
        t0 = b0;
        t1 = b1;
        t2 = 0;
    }

    if let Some(m) = mod_quot {
        *m = quot;
    } else {
        (t0, t1, t2) = sub192((t0, t1, t2), (a0, a1, a2));
        if lt192((t0, t1, t2), (a0, a1, a2)) || ((t0, t1, t2) == (a0, a1, a2) && (q & 1) != 0) {
            a0 = t0;
            a1 = t1;
            a2 = t2;
            *a_sign = !*a_sign;
        }
    }

    let shift;
    if a0 != 0 {
        shift = a0.leading_zeros();
        (a0, a1, a2) = short_shift192_left(a0, a1, a2, shift);
    } else if a1 != 0 {
        let s = a1.leading_zeros();
        (a0, a1) = short_shift128_left(a1, a2, s);
        a2 = 0;
        shift = s + 64;
    } else if a2 != 0 {
        let s = a2.leading_zeros();
        a0 = a2 << s;
        a1 = 0;
        a2 = 0;
        shift = s + 128;
    } else {
        return true;
    }

    *a_exp = b_exp + exp_diff - shift as i32;
    a_frac.hi = a0;
    a_frac.lo = a1 | u64::from(a2 != 0);
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shrjam128_collects_sticky() {
        let mut f = Frac128 { hi: 1, lo: 1 };
        f.shrjam(64);
        assert_eq!(f, Frac128 { hi: 0, lo: 1 });
        let mut f = Frac128 { hi: 0x8000_0000_0000_0000, lo: 0 };
        f.shrjam(200);
        assert_eq!(f, Frac128 { hi: 0, lo: 1 });
    }

    #[test]
    fn normalize256_counts_words() {
        let mut f = Frac256 { hi: 0, hm: 0, lm: 1, lo: 0 };
        assert_eq!(f.normalize(), 128 + 63);
        assert_eq!(f.hi, 1 << 63);
    }

    #[test]
    fn estimate_div_is_close() {
        let b = 0x8000_0000_0000_0001u64;
        let q = estimate_div_128_to_64(0x4000_0000_0000_0000, 0, b);
        let exact = ((0x4000_0000_0000_0000u128) << 64) / u128::from(b);
        assert!(u128::from(q) >= exact && u128::from(q) - exact <= 2);
    }
}
