// SPDX-License-Identifier: GPL-2.0-or-later

//! The AArch32 Advanced SIMD integer helper: the operations of QEMU's `neon_helper.c` and of
//! the gvec expansions that `translate-neon.c` uses, run on D register images by the A64
//! helper [`vec_helper::neon`] where the two instruction sets agree.
//!
//! D`n` is half of the AdvSIMD register Q`n/2`, so the two D registers of a Q register are
//! contiguous but D`2k+1` and D`2k+2` are not. The helper reads each operand as the 16 bytes
//! of D`r` and D`r+1` and writes 8 or 16 bytes back the same way, leaving the rest of the
//! SVE register alone as QEMU's AArch32 code does.

use ruvm_jit_core::HelperType::{I32, Ptr, Void};
use ruvm_jit_interp::{HelperEnv, Unwind};

use super::helpers::{Def, def};
use super::vec_helper::{self, Desc, Regs, V, op, set_qc};
use crate::cpu::vreg_off;

/// Write the result to D`d` and D`d+1`, not only D`d`.
pub(crate) const OUT_Q: u32 = 1 << 20;
/// Also run operation `op + 1` and write its result to D`m` (and D`m+1`), for VTRN, VUZP
/// and VZIP.
pub(crate) const PAIR: u32 = 1 << 21;

def!(NEON, "a32_neon", 0, Void, [Ptr, I32, I32], h_neon);

/// The helpers of this module.
pub(crate) const ALL: &[Def] = &[NEON];

/// The `env` offset of D`r`.
fn d_off(r: u32) -> usize {
    let r = r as usize;
    vreg_off(r >> 1) + 8 * (r & 1)
}

/// D`r` and D`r+1` as one image.
fn dload(env: &[u8], r: u32) -> V {
    let mut v = [0; 16];
    let (lo, hi) = (d_off(r), d_off(r + 1));
    v[..8].copy_from_slice(&env[lo..lo + 8]);
    v[8..].copy_from_slice(&env[hi..hi + 8]);
    v
}

/// Write the low 8 bytes of `v` to D`r`, and the high 8 bytes to D`r+1` if `q`.
fn dstore(env: &mut [u8], r: u32, v: &V, q: bool) {
    let lo = d_off(r);
    env[lo..lo + 8].copy_from_slice(&v[..8]);
    if q {
        let hi = d_off(r + 1);
        env[hi..hi + 8].copy_from_slice(&v[8..]);
    }
}

/// `a32_neon(env, regs, desc)`: the operation in `desc` (a [`Desc`]) on D`n`, D`m` and the
/// old D`d` in `regs` (from [`Regs::pack`], with [`OUT_Q`] and [`PAIR`]).
fn h_neon(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    let r = Regs::unpack(a[1]);
    let flags = a[1] as u32;
    let q = flags & OUT_Q != 0;
    let desc = Desc::unpack(a[2]);
    let env = &mut *h.env;
    let n = dload(env, r.n);
    let m = dload(env, r.m);
    let d = dload(env, r.d);
    let mut sat = false;
    let out = if matches!(desc.op, op::TBL | op::TBX) {
        vtbl(env, &desc, r.n, &m, &d)
    } else {
        vec_helper::neon(&desc, r, env, &n, &m, &d, &mut sat)
    };
    if flags & PAIR != 0 {
        let desc2 = Desc { op: desc.op + 1, ..desc };
        let out2 = vec_helper::neon(&desc2, r, env, &n, &m, &d, &mut sat);
        // QEMU writes Dm first, so Dd wins when they are the same register, except for
        // VTRN.32, which swaps the elements one at a time.
        dstore(env, r.m, &out2, q);
        if r.d == r.m && desc.op == op::TRN1 && desc.esz == 2 {
            let rev = Desc { op: op::REV64, ..desc };
            let out = vec_helper::neon(&rev, r, env, &d, &d, &d, &mut sat);
            dstore(env, r.d, &out, q);
        } else {
            dstore(env, r.d, &out, q);
        }
    } else {
        dstore(env, r.d, &out, q);
    }
    if sat {
        set_qc(env);
    }
    Ok(0)
}

/// VTBL and VTBX: the table is the `imm + 1` registers from D`n`, the indexes the bytes of
/// D`m`. An index past the table gives 0, or the old byte of D`d` for VTBX.
fn vtbl(env: &[u8], desc: &Desc, rn: u32, m: &V, d: &V) -> V {
    let len = desc.imm as usize + 1;
    let mut table = [0u8; 32];
    for k in 0..len {
        let o = d_off(rn + k as u32);
        table[8 * k..8 * k + 8].copy_from_slice(&env[o..o + 8]);
    }
    let mut out = [0u8; 16];
    for i in 0..8 {
        let ix = m[i] as usize;
        out[i] = if ix < 8 * len {
            table[ix]
        } else if desc.op == op::TBX {
            d[i]
        } else {
            0
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d_registers_follow_the_q_registers() {
        assert_eq!(d_off(3), vreg_off(1) + 8);
        assert_eq!(d_off(4), vreg_off(2));
    }
}
