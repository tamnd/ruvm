// SPDX-License-Identifier: MIT OR Apache-2.0

//! Vector op semantics, element by element.
//!
//! A vector value is up to 256 bits held as four little-endian 64-bit words; element `i` of
//! size `8 << vece` bits sits at bit `i << (3 + vece)`.

use ruvm_jit_core::ir::Op;
use ruvm_jit_core::opcode::Opcode;
use ruvm_jit_core::types::{Cond, dup_const};

pub(crate) type V = [u64; 4];

pub(crate) fn from_bytes(b: &[u8; 32]) -> V {
    let mut v = [0u64; 4];
    for (i, w) in v.iter_mut().enumerate() {
        let mut x = [0u8; 8];
        x.copy_from_slice(&b[8 * i..8 * i + 8]);
        *w = u64::from_le_bytes(x);
    }
    v
}

pub(crate) fn to_bytes(v: &V) -> [u8; 32] {
    let mut b = [0u8; 32];
    for (i, w) in v.iter().enumerate() {
        b[8 * i..8 * i + 8].copy_from_slice(&w.to_le_bytes());
    }
    b
}

pub(crate) fn dup(vece: u32, x: u64) -> V {
    [dup_const(vece, x); 4]
}

/// Zero every byte from `bytes` on.
pub(crate) fn clear_above(mut v: V, bytes: usize) -> V {
    for (i, w) in v.iter_mut().enumerate() {
        if 8 * i >= bytes {
            *w = 0;
        }
    }
    v
}

fn esize(vece: u32) -> u32 {
    8 << vece
}

fn emask(bits: u32) -> u64 {
    if bits == 64 { !0 } else { (1u64 << bits) - 1 }
}

fn lane(v: &V, vece: u32, i: usize) -> u64 {
    let bits = esize(vece);
    let bit = i * bits as usize;
    (v[bit / 64] >> (bit % 64)) & emask(bits)
}

fn set_lane(v: &mut V, vece: u32, i: usize, x: u64) {
    let bits = esize(vece);
    let bit = i * bits as usize;
    let m = emask(bits) << (bit % 64);
    v[bit / 64] = (v[bit / 64] & !m) | ((x << (bit % 64)) & m);
}

fn sext(x: u64, bits: u32) -> i64 {
    ((x << (64 - bits)) as i64) >> (64 - bits)
}

fn lanes(vece: u32) -> usize {
    256 / esize(vece) as usize
}

fn map1(vece: u32, a: &V, f: impl Fn(u64) -> u64) -> V {
    let mut r = [0u64; 4];
    for i in 0..lanes(vece) {
        set_lane(&mut r, vece, i, f(lane(a, vece, i)));
    }
    r
}

fn map2(vece: u32, a: &V, b: &V, f: impl Fn(u64, u64) -> u64) -> V {
    let mut r = [0u64; 4];
    for i in 0..lanes(vece) {
        set_lane(&mut r, vece, i, f(lane(a, vece, i), lane(b, vece, i)));
    }
    r
}

fn shift(opc: Opcode, bits: u32, x: u64, n: u64) -> u64 {
    let n = (n & (bits as u64 - 1)) as u32;
    let m = emask(bits);
    match opc {
        Opcode::ShliVec | Opcode::ShlsVec | Opcode::ShlvVec => (x << n) & m,
        Opcode::ShriVec | Opcode::ShrsVec | Opcode::ShrvVec => x >> n,
        Opcode::SariVec | Opcode::SarsVec | Opcode::SarvVec => (sext(x, bits) >> n) as u64 & m,
        Opcode::RotliVec | Opcode::RotlsVec | Opcode::RotlvVec => {
            if n == 0 {
                x
            } else {
                ((x << n) | (x >> (bits - n))) & m
            }
        }
        Opcode::RotrvVec => {
            if n == 0 {
                x
            } else {
                ((x >> n) | (x << (bits - n))) & m
            }
        }
        _ => unreachable!(),
    }
}

pub(crate) fn shift_scalar(opc: Opcode, vece: u32, a: &V, n: u64) -> V {
    let bits = esize(vece);
    map1(vece, a, |x| shift(opc, bits, x, n))
}

fn cond(c: Cond, bits: u32, x: u64, y: u64) -> bool {
    match c {
        Cond::Lt | Cond::Ge | Cond::Le | Cond::Gt => {
            c.eval_u64(sext(x, bits) as u64, sext(y, bits) as u64)
        }
        _ => c.eval_u64(x, y),
    }
}

/// Every vector op except the loads, stores and scalar shifts, which need the machine.
pub(crate) fn exec(op: &Op, ins: &[V; 4]) -> Result<V, String> {
    let vece = op.vece as u32;
    let bits = esize(vece);
    let m = emask(bits);
    let (a, b) = (&ins[0], &ins[1]);
    let smin = -(1i128 << (bits - 1));
    let smax = (1i128 << (bits - 1)) - 1;
    let s = |x: u64| sext(x, bits) as i128;
    let v = match op.opc {
        Opcode::MovVec => *a,
        Opcode::AddVec => map2(vece, a, b, |x, y| x.wrapping_add(y) & m),
        Opcode::SubVec => map2(vece, a, b, |x, y| x.wrapping_sub(y) & m),
        Opcode::MulVec => map2(vece, a, b, |x, y| x.wrapping_mul(y) & m),
        Opcode::NegVec => map1(vece, a, |x| x.wrapping_neg() & m),
        Opcode::AbsVec => map1(vece, a, |x| (s(x).unsigned_abs() as u64) & m),
        Opcode::SsaddVec => map2(vece, a, b, |x, y| (s(x) + s(y)).clamp(smin, smax) as u64 & m),
        Opcode::SssubVec => map2(vece, a, b, |x, y| (s(x) - s(y)).clamp(smin, smax) as u64 & m),
        Opcode::UsaddVec => map2(vece, a, b, |x, y| (x as u128 + y as u128).min(m as u128) as u64),
        Opcode::UssubVec => map2(vece, a, b, |x, y| x.saturating_sub(y)),
        Opcode::SminVec => map2(vece, a, b, |x, y| if s(x) <= s(y) { x } else { y }),
        Opcode::SmaxVec => map2(vece, a, b, |x, y| if s(x) >= s(y) { x } else { y }),
        Opcode::UminVec => map2(vece, a, b, |x, y| x.min(y)),
        Opcode::UmaxVec => map2(vece, a, b, |x, y| x.max(y)),
        Opcode::AndVec => std::array::from_fn(|i| a[i] & b[i]),
        Opcode::OrVec => std::array::from_fn(|i| a[i] | b[i]),
        Opcode::XorVec => std::array::from_fn(|i| a[i] ^ b[i]),
        Opcode::AndcVec => std::array::from_fn(|i| a[i] & !b[i]),
        Opcode::OrcVec => std::array::from_fn(|i| a[i] | !b[i]),
        Opcode::NandVec => std::array::from_fn(|i| !(a[i] & b[i])),
        Opcode::NorVec => std::array::from_fn(|i| !(a[i] | b[i])),
        Opcode::EqvVec => std::array::from_fn(|i| !(a[i] ^ b[i])),
        Opcode::NotVec => std::array::from_fn(|i| !a[i]),
        Opcode::ShliVec | Opcode::ShriVec | Opcode::SariVec | Opcode::RotliVec => {
            let n = op.args[2];
            map1(vece, a, |x| shift(op.opc, bits, x, n))
        }
        Opcode::ShlvVec
        | Opcode::ShrvVec
        | Opcode::SarvVec
        | Opcode::RotlvVec
        | Opcode::RotrvVec => map2(vece, a, b, |x, y| shift(op.opc, bits, x, y)),
        Opcode::CmpVec => {
            let c = Cond::from_u64(op.args[3]).ok_or("cmp_vec: bad condition")?;
            map2(vece, a, b, |x, y| if cond(c, bits, x, y) { m } else { 0 })
        }
        Opcode::BitselVec => {
            let c = &ins[2];
            std::array::from_fn(|i| (a[i] & b[i]) | (!a[i] & c[i]))
        }
        Opcode::CmpselVec => {
            let c = Cond::from_u64(op.args[5]).ok_or("cmpsel_vec: bad condition")?;
            let (t, e) = (&ins[2], &ins[3]);
            let mut r = [0u64; 4];
            for i in 0..lanes(vece) {
                let pick = if cond(c, bits, lane(a, vece, i), lane(b, vece, i)) { t } else { e };
                set_lane(&mut r, vece, i, lane(pick, vece, i));
            }
            r
        }
        o => return Err(format!("unhandled vector op {}", o.name())),
    };
    Ok(v)
}
