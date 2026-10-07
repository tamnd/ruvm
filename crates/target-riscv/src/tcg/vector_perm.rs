// SPDX-License-Identifier: GPL-2.0-or-later

//! The vector integer reduction, mask and permutation helpers, the parts of QEMU's
//! `target/riscv/tcg/vector_helper.c` from "Vector Reduction Operations" (the integer
//! ones) to "Vector Whole Register Move", and `vset_velem0`.
//!
//! As elsewhere, a helper serves every SEW and gets its registers in the descriptor;
//! [`Desc::x`] picks the operation where QEMU has a helper per operation.
//!
//! Deliberate differences from QEMU:
//!
//! - `vmv.x.s` and `vfmv.f.s` read element 0 with a helper, [`VMV_X_S`], where QEMU loads
//!   it inline, and `vmv.s.x` and `vfmv.s.f` check `vstart < vl` in their helper,
//!   [`VMV_S_X`], where QEMU branches around `vset_velem0`. The registers and `vstart`
//!   end up the same.

use ruvm_jit_core::HelperType::{I32, I64, Ptr, Void};
use ruvm_jit_core::types::call_flags::NO_RWG;
use ruvm_jit_interp::HelperEnv;

use super::helpers::Def;
use super::vector::{
    Desc, HR, def, for_each, max_elems, set_1s, set_vstart, sext, total_elems, vget, vl, vmask,
    vset, vset_mask, vsew, vstart,
};
use crate::cpu::VLENB;

/// The descriptor of a helper call.
fn desc(a: &[u64]) -> Desc {
    Desc::decode(a[1] as u32)
}

// Reductions.

/// [`Desc::x`] of [`VRED`]: `vredsum.vs`.
pub(super) const RED_SUM: u32 = 0;
/// `vredmaxu.vs`.
pub(super) const RED_MAXU: u32 = 1;
/// `vredmax.vs`.
pub(super) const RED_MAX: u32 = 2;
/// `vredminu.vs`.
pub(super) const RED_MINU: u32 = 3;
/// `vredmin.vs`.
pub(super) const RED_MIN: u32 = 4;
/// `vredand.vs`.
pub(super) const RED_AND: u32 = 5;
/// `vredor.vs`.
pub(super) const RED_OR: u32 = 6;
/// `vredxor.vs`.
pub(super) const RED_XOR: u32 = 7;

def!(VRED, "vred", NO_RWG, Void, [Ptr, I32], h_vred);
def!(VWRED, "vwred", NO_RWG, Void, [Ptr, I32], h_vwred);

/// `GEN_VEXT_RED()`: `vd[0] = op(vs1[0], vs2[*])` over the active elements, `vd[0]` and
/// `vs1[0]` of `1 << dlog2` bytes, the `vs2` elements of `1 << slog2` bytes.
fn reduce(env: &mut [u8], d: &Desc, dlog2: u32, slog2: u32, op: impl Fn(u64, u64) -> u64) {
    let vl = vl(env);
    let mut s1 = vget(env, d.vs1, 0, dlog2);
    let start = vstart(env);
    if start >= vl {
        set_vstart(env, 0);
        return;
    }
    for i in start..vl {
        if !d.vm && !vmask(env, 0, i) {
            continue;
        }
        s1 = op(s1, vget(env, d.vs2, i, slog2));
    }
    vset(env, d.vd, 0, dlog2, s1);
    set_vstart(env, 0);
    // Set the tail elements to ones.
    set_1s(env, d.vd, d.vta, 1 << dlog2, VLENB);
}

/// The single width integer reductions, `vredsum_vs_b` to `vredxor_vs_d`.
fn h_vred(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = desc(a);
    let sew = d.esz;
    let op = d.x;
    reduce(h.env, &d, sew, sew, |s1, s2| match op {
        RED_SUM => s1.wrapping_add(s2),
        RED_MAXU => s1.max(s2),
        RED_MAX => {
            if sext(s1, sew) >= sext(s2, sew) {
                s1
            } else {
                s2
            }
        }
        RED_MINU => s1.min(s2),
        RED_MIN => {
            if sext(s1, sew) <= sext(s2, sew) {
                s1
            } else {
                s2
            }
        }
        RED_AND => s1 & s2,
        RED_OR => s1 | s2,
        _ => s1 ^ s2,
    });
    Ok(0)
}

/// `vwredsum_vs_*` ([`Desc::x`] 0) and `vwredsumu_vs_*` ([`Desc::x`] 1): a 2*SEW sum of
/// the SEW elements extended.
fn h_vwred(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = desc(a);
    let sew = d.esz;
    let signed = d.x == 0;
    reduce(h.env, &d, sew + 1, sew, |s1, s2| {
        let s2 = if signed { sext(s2, sew) as u64 } else { s2 };
        s1.wrapping_add(s2)
    });
    Ok(0)
}

// Mask operations.

/// [`Desc::x`] of [`VMASK_MM`]: `vmand.mm`.
pub(super) const MM_AND: u32 = 0;
/// `vmnand.mm`.
pub(super) const MM_NAND: u32 = 1;
/// `vmandn.mm`.
pub(super) const MM_ANDN: u32 = 2;
/// `vmxor.mm`.
pub(super) const MM_XOR: u32 = 3;
/// `vmor.mm`.
pub(super) const MM_OR: u32 = 4;
/// `vmnor.mm`.
pub(super) const MM_NOR: u32 = 5;
/// `vmorn.mm`.
pub(super) const MM_ORN: u32 = 6;
/// `vmxnor.mm`.
pub(super) const MM_XNOR: u32 = 7;

def!(VMASK_MM, "vmask_mm", NO_RWG, Void, [Ptr, I32], h_vmask_mm);
def!(VCPOP_M, "vcpop_m", NO_RWG, I64, [Ptr, I32], h_vcpop_m);
def!(VFIRST_M, "vfirst_m", NO_RWG, I64, [Ptr, I32], h_vfirst_m);
def!(VMSETM, "vmsetm", NO_RWG, Void, [Ptr, I32], h_vmsetm);
def!(VIOTA_M, "viota_m", NO_RWG, Void, [Ptr, I32], h_viota_m);
def!(VID_V, "vid_v", NO_RWG, Void, [Ptr, I32], h_vid_v);

/// The mask bits of a mask register, `vlenb * 8`.
const MASK_BITS: usize = VLENB * 8;

/// Mask destinations are always tail agnostic: with `vta_all_1s`, set mask bits `from` to
/// the end of the register.
fn mask_tail(env: &mut [u8], d: &Desc, from: usize) {
    if d.vta_all_1s {
        for i in from..MASK_BITS {
            vset_mask(env, d.vd, i, true);
        }
    }
}

/// `GEN_VEXT_MASK_VV()`: `vd.mask[i] = op(vs2.mask[i], vs1.mask[i])`.
fn h_vmask_mm(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = desc(a);
    let env = &mut *h.env;
    let vl = vl(env);
    let start = vstart(env);
    if start >= vl {
        set_vstart(env, 0);
        return Ok(0);
    }
    for i in start..vl {
        let m = vmask(env, d.vs1, i);
        let n = vmask(env, d.vs2, i);
        let r = match d.x {
            MM_AND => n & m,
            MM_NAND => !(n & m),
            MM_ANDN => n & !m,
            MM_XOR => n ^ m,
            MM_OR => n | m,
            MM_NOR => !(n | m),
            MM_ORN => n | !m,
            _ => !(n ^ m),
        };
        vset_mask(env, d.vd, i, r);
    }
    set_vstart(env, 0);
    mask_tail(env, &d, vl);
    Ok(0)
}

/// `HELPER(vcpop_m)`: the active set bits of `vs2`.
fn h_vcpop_m(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = desc(a);
    let env = &mut *h.env;
    let mut cnt = 0u64;
    for i in vstart(env)..vl(env) {
        if (d.vm || vmask(env, 0, i)) && vmask(env, d.vs2, i) {
            cnt += 1;
        }
    }
    set_vstart(env, 0);
    Ok(u128::from(cnt))
}

/// `HELPER(vfirst_m)`: the index of the first active set bit of `vs2`, or -1.
fn h_vfirst_m(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = desc(a);
    let env = &mut *h.env;
    for i in vstart(env)..vl(env) {
        if (d.vm || vmask(env, 0, i)) && vmask(env, d.vs2, i) {
            return Ok(i as u128);
        }
    }
    set_vstart(env, 0);
    Ok(u128::from(u64::MAX))
}

/// [`Desc::x`] of [`VMSETM`], `enum set_mask_type`: `vmsof.m`.
pub(super) const ONLY_FIRST: u32 = 1;
/// `vmsif.m`.
pub(super) const INCLUDE_FIRST: u32 = 2;
/// `vmsbf.m`.
pub(super) const BEFORE_FIRST: u32 = 3;

/// `vmsetm()`: `vmsbf.m`, `vmsif.m` and `vmsof.m`.
fn h_vmsetm(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = desc(a);
    let ty = d.x;
    let env = &mut *h.env;
    let vl = vl(env);
    let start = vstart(env);
    if start >= vl {
        set_vstart(env, 0);
        return Ok(0);
    }
    let mut first_mask_bit = false;
    for i in start..vl {
        if !d.vm && !vmask(env, 0, i) {
            // Set the masked off elements to ones.
            if d.vma {
                vset_mask(env, d.vd, i, true);
            }
            continue;
        }
        // Write a zero to all the following active elements.
        if first_mask_bit {
            vset_mask(env, d.vd, i, false);
            continue;
        }
        if vmask(env, d.vs2, i) {
            first_mask_bit = true;
            vset_mask(env, d.vd, i, ty != BEFORE_FIRST);
        } else {
            vset_mask(env, d.vd, i, ty != ONLY_FIRST);
        }
    }
    set_vstart(env, 0);
    mask_tail(env, &d, vl);
    Ok(0)
}

/// `GEN_VEXT_VIOTA_M()`: `vd[i]` is the number of set bits of `vs2` below `i`.
fn h_viota_m(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = desc(a);
    let mut sum = 0u64;
    for_each(h.env, &d, d.esz, |env, i| {
        vset(env, d.vd, i, d.esz, sum);
        if vmask(env, d.vs2, i) {
            sum += 1;
        }
    });
    Ok(0)
}

/// `GEN_VEXT_VID_V()`: `vd[i] = i`.
fn h_vid_v(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = desc(a);
    for_each(h.env, &d, d.esz, |env, i| vset(env, d.vd, i, d.esz, i as u64));
    Ok(0)
}

// Scalar moves.

def!(VMV_X_S, "vmv_x_s", NO_RWG, I64, [Ptr, I32], h_vmv_x_s);
def!(VMV_S_X, "vmv_s_x", NO_RWG, Void, [Ptr, I32, I64], h_vmv_s_x);

/// `vmv.x.s` ([`Desc::x`] 0): element 0 of `vs2` sign extended; `vfmv.f.s`
/// ([`Desc::x`] 1): element 0 of `vs2` NaN boxed. Both clear `vstart`.
fn h_vmv_x_s(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = desc(a);
    let sew = d.esz;
    let v = vget(h.env, d.vs2, 0, sew);
    let r = if d.x == 0 {
        sext(v, sew) as u64
    } else if sew < 3 {
        v | (u64::MAX << (8 << sew))
    } else {
        v
    };
    set_vstart(h.env, 0);
    Ok(u128::from(r))
}

/// `vmv.s.x` and `vfmv.s.f`: unless `vstart >= vl`, `vset_velem0()`: `vd[0] = s1`, every
/// element of the register past it tail; then `vstart` is cleared.
fn h_vmv_s_x(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = desc(a);
    let env = &mut *h.env;
    if vstart(env) < vl(env) {
        vset(env, d.vd, 0, d.esz, a[2]);
        // Treat every element past vd[0] as tail for scalar to vector moves.
        set_1s(env, d.vd, d.vta, 1 << d.esz, VLENB);
    }
    set_vstart(env, 0);
    Ok(0)
}

// Slides.

def!(VSLIDEUP, "vslideup", NO_RWG, Void, [Ptr, I32, I64], h_vslideup);
def!(VSLIDEDOWN, "vslidedown", NO_RWG, Void, [Ptr, I32, I64], h_vslidedown);
def!(VSLIDE1UP, "vslide1up", NO_RWG, Void, [Ptr, I32, I64], h_vslide1up);
def!(VSLIDE1DOWN, "vslide1down", NO_RWG, Void, [Ptr, I32, I64], h_vslide1down);

/// Set masked off element `i` of `1 << log2` bytes of `vd` to ones when mask agnostic, and
/// say whether it is masked off.
fn masked_off(env: &mut [u8], d: &Desc, i: usize, log2: u32) -> bool {
    if !d.vm && !vmask(env, 0, i) {
        set_1s(env, d.vd, d.vma, i << log2, (i + 1) << log2);
        return true;
    }
    false
}

/// Clear `vstart` and set the tail of `vd` from element `from` to ones.
fn finish(env: &mut [u8], d: &Desc, log2: u32, from: usize) {
    set_vstart(env, 0);
    let total = total_elems(env, d, log2);
    set_1s(env, d.vd, d.vta, from << log2, total << log2);
}

/// `GEN_VEXT_VSLIDEUP_VX()`: `vd[i + s1] = vs2[i]`.
fn h_vslideup(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = desc(a);
    let sew = d.esz;
    let offset = a[2];
    let env = &mut *h.env;
    let vl = vl(env);
    let start = vstart(env);
    if start >= vl {
        set_vstart(env, 0);
        return Ok(0);
    }
    let i_min = (start as u64).max(offset);
    for i in i_min..vl as u64 {
        let i = i as usize;
        if masked_off(env, &d, i, sew) {
            continue;
        }
        let v = vget(env, d.vs2, i - offset as usize, sew);
        vset(env, d.vd, i, sew, v);
    }
    finish(env, &d, sew, vl);
    Ok(0)
}

/// `GEN_VEXT_VSLIDEDOWN_VX()`: `vd[i] = vs2[i + s1]`, zero past VLMAX.
fn h_vslidedown(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = desc(a);
    let sew = d.esz;
    let s1 = a[2];
    let vlmax = max_elems(&d, sew) as u64;
    let env = &mut *h.env;
    let vl = vl(env);
    let start = vstart(env);
    if start >= vl {
        set_vstart(env, 0);
        return Ok(0);
    }
    let i_min = vlmax.saturating_sub(s1).min(vl as u64) as usize;
    let i_max = i_min.max(start);
    for i in start..i_max {
        if masked_off(env, &d, i, sew) {
            continue;
        }
        let v = vget(env, d.vs2, i + s1 as usize, sew);
        vset(env, d.vd, i, sew, v);
    }
    for i in i_max..vl {
        if masked_off(env, &d, i, sew) {
            continue;
        }
        vset(env, d.vd, i, sew, 0);
    }
    finish(env, &d, sew, vl);
    Ok(0)
}

/// `vslide1up_*()`: `vd[0] = s1`, `vd[i + 1] = vs2[i]`.
fn h_vslide1up(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = desc(a);
    let sew = d.esz;
    let s1 = a[2];
    for_each(h.env, &d, sew, |env, i| {
        let v = if i == 0 { s1 } else { vget(env, d.vs2, i - 1, sew) };
        vset(env, d.vd, i, sew, v);
    });
    Ok(0)
}

/// `vslide1down_*()`: `vd[i] = vs2[i + 1]`, `vd[vl - 1] = s1`.
fn h_vslide1down(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = desc(a);
    let sew = d.esz;
    let s1 = a[2];
    let vl = vl(h.env);
    for_each(h.env, &d, sew, |env, i| {
        let v = if i == vl - 1 { s1 } else { vget(env, d.vs2, i + 1, sew) };
        vset(env, d.vd, i, sew, v);
    });
    Ok(0)
}

// Gather, compress and whole register moves.

def!(VRGATHER_VV, "vrgather_vv", NO_RWG, Void, [Ptr, I32], h_vrgather_vv);
def!(VRGATHER_VX, "vrgather_vx", NO_RWG, Void, [Ptr, I32, I64], h_vrgather_vx);
def!(VCOMPRESS_VM, "vcompress_vm", NO_RWG, Void, [Ptr, I32], h_vcompress_vm);
def!(VMVR_V, "vmvr_v", NO_RWG, Void, [Ptr, I32], h_vmvr_v);

/// `vd[i] = index >= VLMAX ? 0 : vs2[index]` for each active `i`, the index of element `i`
/// from `index`.
fn gather(env: &mut [u8], d: &Desc, index: impl Fn(&[u8], usize) -> u64) {
    let sew = d.esz;
    let vlmax = max_elems(d, sew) as u64;
    for_each(env, d, sew, |env, i| {
        let idx = index(env, i);
        let v = if idx >= vlmax { 0 } else { vget(env, d.vs2, idx as usize, sew) };
        vset(env, d.vd, i, sew, v);
    });
}

/// `GEN_VEXT_VRGATHER_VV()`: `vrgather.vv` ([`Desc::x`] 0, SEW indices) and
/// `vrgatherei16.vv` ([`Desc::x`] 1, 16-bit indices).
fn h_vrgather_vv(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = desc(a);
    let ilog2 = if d.x == 1 { 1 } else { d.esz };
    gather(h.env, &d, |env, i| vget(env, d.vs1, i, ilog2));
    Ok(0)
}

/// `GEN_VEXT_VRGATHER_VX()`: every element from `vs2[s1]`.
fn h_vrgather_vx(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = desc(a);
    let s1 = a[2];
    gather(h.env, &d, |_, _| s1);
    Ok(0)
}

/// `GEN_VEXT_VCOMPRESS_VM()`: the elements of `vs2` whose `vs1` mask bit is set, packed
/// at the start of `vd`.
fn h_vcompress_vm(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = desc(a);
    let sew = d.esz;
    let env = &mut *h.env;
    let vl = vl(env);
    let start = vstart(env);
    if start >= vl {
        set_vstart(env, 0);
        return Ok(0);
    }
    let mut num = 0;
    for i in start..vl {
        if !vmask(env, d.vs1, i) {
            continue;
        }
        let v = vget(env, d.vs2, i, sew);
        vset(env, d.vd, num, sew, v);
        num += 1;
    }
    finish(env, &d, sew, num);
    Ok(0)
}

/// `HELPER(vmvr_v)`: copy [`Desc::nf`] registers from `vs2` to `vd` from element `vstart`
/// of SEW.
fn h_vmvr_v(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = desc(a);
    let env = &mut *h.env;
    let maxsz = VLENB * d.nf as usize;
    let startb = vstart(env) << vsew(env);
    for b in startb..maxsz {
        env[super::vector::voff(d.vd, b)] = env[super::vector::voff(d.vs2, b)];
    }
    set_vstart(env, 0);
    Ok(0)
}

/// The helpers of this module.
pub(crate) const ALL: &[Def] = &[
    VRED,
    VWRED,
    VMASK_MM,
    VCPOP_M,
    VFIRST_M,
    VMSETM,
    VIOTA_M,
    VID_V,
    VMV_X_S,
    VMV_S_X,
    VSLIDEUP,
    VSLIDEDOWN,
    VSLIDE1UP,
    VSLIDE1DOWN,
    VRGATHER_VV,
    VRGATHER_VX,
    VCOMPRESS_VM,
    VMVR_V,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{ENV_SIZE, VL, VSTART, VTYPE};
    use crate::tcg::st64;
    use crate::tcg::vector::voff;
    use ruvm_jit_interp::NoMemory;

    /// An `env` with `vtype` SEW `1 << sew` bytes, LMUL 1, and `vl`.
    fn env(sew: u32, vl: u64) -> Vec<u8> {
        let mut env = vec![0u8; ENV_SIZE];
        st64(&mut env, VTYPE, u64::from(sew) << 3);
        st64(&mut env, VL, vl);
        env
    }

    /// Call helper `h` with descriptor `d` and `args` after it.
    fn call(h: &Def, env: &mut [u8], d: Desc, args: &[u64]) -> u64 {
        let mut a = vec![0, u64::from(d.encode())];
        a.extend_from_slice(args);
        let mut mem = NoMemory;
        let mut he = HelperEnv { env, mem: &mut mem };
        (h.f)(&mut he, &a).expect("no unwind") as u64
    }

    fn fill(env: &mut [u8], reg: u32, log2: u32, vals: &[u64]) {
        for (i, &v) in vals.iter().enumerate() {
            vset(env, reg, i, log2, v);
        }
    }

    #[test]
    fn reductions() {
        let mut e = env(1, 4);
        fill(&mut e, 2, 1, &[1, 0xffff, 5, 0x8000]);
        fill(&mut e, 3, 1, &[10]);
        let d = Desc { vm: true, vd: 1, vs1: 3, vs2: 2, esz: 1, vta: true, ..Desc::default() };
        call(&VRED, &mut e, Desc { x: RED_SUM, ..d }, &[]);
        assert_eq!(vget(&e, 1, 0, 1), (10 + 1 + 0xffff + 5 + 0x8000) & 0xffff);
        // The tail of vd past element 0 is all ones with vta.
        assert_eq!(vget(&e, 1, 1, 1), 0xffff);
        call(&VRED, &mut e, Desc { x: RED_MAX, ..d }, &[]);
        assert_eq!(vget(&e, 1, 0, 1), 10);
        call(&VRED, &mut e, Desc { x: RED_MIN, ..d }, &[]);
        assert_eq!(vget(&e, 1, 0, 1), 0x8000);
        call(&VRED, &mut e, Desc { x: RED_MAXU, ..d }, &[]);
        assert_eq!(vget(&e, 1, 0, 1), 0xffff);
        call(&VRED, &mut e, Desc { x: RED_MINU, ..d }, &[]);
        assert_eq!(vget(&e, 1, 0, 1), 1);
        call(&VRED, &mut e, Desc { x: RED_XOR, ..d }, &[]);
        assert_eq!(vget(&e, 1, 0, 1), 10 ^ 1 ^ 0xffff ^ 5 ^ 0x8000);
        // Masked: only elements 0 and 2.
        e[voff(0, 0)] = 0b0101;
        call(&VRED, &mut e, Desc { x: RED_SUM, vm: false, ..d }, &[]);
        assert_eq!(vget(&e, 1, 0, 1), 16);
        // Widening: signed and unsigned sums of 16-bit elements into 32 bits.
        fill(&mut e, 3, 2, &[1 << 20]);
        call(&VWRED, &mut e, Desc { x: 0, ..d }, &[]);
        assert_eq!(vget(&e, 1, 0, 2), (1 << 20) + 1 - 1 + 5 - 0x8000);
        call(&VWRED, &mut e, Desc { x: 1, ..d }, &[]);
        assert_eq!(vget(&e, 1, 0, 2), (1 << 20) + 1 + 0xffff + 5 + 0x8000);
        // vl 0 leaves vd alone.
        st64(&mut e, VL, 0);
        vset(&mut e, 1, 0, 1, 0x1234);
        call(&VRED, &mut e, Desc { x: RED_SUM, ..d }, &[]);
        assert_eq!(vget(&e, 1, 0, 1), 0x1234);
    }

    #[test]
    fn mask_ops() {
        let mut e = env(0, 10);
        e[voff(1, 0)] = 0b1100_1010;
        e[voff(1, 1)] = 0b11;
        e[voff(2, 0)] = 0b1010_0110;
        e[voff(2, 1)] = 0b01;
        let d = Desc { vd: 3, vs1: 1, vs2: 2, vta_all_1s: true, ..Desc::default() };
        call(&VMASK_MM, &mut e, Desc { x: MM_ANDN, ..d }, &[]);
        // vs2 & !vs1 over 10 bits, ones after.
        assert_eq!(e[voff(3, 0)], 0b0010_0100);
        assert_eq!(e[voff(3, 1)], 0b1111_1100);
        assert_eq!(e[voff(3, 15)], 0xff);
        call(&VMASK_MM, &mut e, Desc { x: MM_XNOR, vta_all_1s: false, ..d }, &[]);
        assert_eq!(e[voff(3, 0)], !(0b1100_1010u8 ^ 0b1010_0110));
        assert_eq!(e[voff(3, 1)] & 3, 0b01);
        // vcpop and vfirst of v2 (5 set bits in the first 10).
        let m = Desc { vm: true, vs2: 2, ..Desc::default() };
        assert_eq!(call(&VCPOP_M, &mut e, m, &[]), 5);
        assert_eq!(call(&VFIRST_M, &mut e, m, &[]), 1);
        e[voff(0, 0)] = 0b1111_1000;
        assert_eq!(call(&VFIRST_M, &mut e, Desc { vm: false, ..m }, &[]), 5);
        e[voff(2, 0)] = 0;
        e[voff(2, 1)] = 0;
        assert_eq!(call(&VFIRST_M, &mut e, m, &[]), u64::MAX);
    }

    #[test]
    fn set_first_and_iota() {
        let mut e = env(0, 8);
        e[voff(2, 0)] = 0b0011_0100;
        let d = Desc { vm: true, vd: 3, vs2: 2, ..Desc::default() };
        call(&VMSETM, &mut e, Desc { x: BEFORE_FIRST, ..d }, &[]);
        assert_eq!(e[voff(3, 0)], 0b0000_0011);
        call(&VMSETM, &mut e, Desc { x: INCLUDE_FIRST, ..d }, &[]);
        assert_eq!(e[voff(3, 0)], 0b0000_0111);
        call(&VMSETM, &mut e, Desc { x: ONLY_FIRST, ..d }, &[]);
        assert_eq!(e[voff(3, 0)], 0b0000_0100);
        // viota.m: the set bits of v2 below each element.
        call(&VIOTA_M, &mut e, Desc { vd: 4, ..d }, &[]);
        let got: Vec<u64> = (0..8).map(|i| vget(&e, 4, i, 0)).collect();
        assert_eq!(got, [0, 0, 0, 1, 1, 2, 3, 3]);
        call(&VID_V, &mut e, Desc { vd: 4, ..d }, &[]);
        assert_eq!(vget(&e, 4, 7, 0), 7);
    }

    #[test]
    fn slides() {
        let mut e = env(2, 4);
        fill(&mut e, 2, 2, &[10, 11, 12, 13]);
        fill(&mut e, 4, 2, &[90, 91, 92, 93]);
        let d = Desc { vm: true, vd: 4, vs2: 2, esz: 2, ..Desc::default() };
        call(&VSLIDEUP, &mut e, d, &[2]);
        let got: Vec<u64> = (0..4).map(|i| vget(&e, 4, i, 2)).collect();
        assert_eq!(got, [90, 91, 10, 11]);
        call(&VSLIDEDOWN, &mut e, d, &[1]);
        let got: Vec<u64> = (0..4).map(|i| vget(&e, 4, i, 2)).collect();
        assert_eq!(got, [11, 12, 13, 0]);
        // With vl 2, slidedown still reads past vl up to VLMAX.
        st64(&mut e, VL, 2);
        call(&VSLIDEDOWN, &mut e, d, &[2]);
        assert_eq!(vget(&e, 4, 0, 2), 12);
        assert_eq!(vget(&e, 4, 1, 2), 13);
        st64(&mut e, VL, 4);
        call(&VSLIDE1UP, &mut e, d, &[0xffff_ffff_8000_0001]);
        let got: Vec<u64> = (0..4).map(|i| vget(&e, 4, i, 2)).collect();
        assert_eq!(got, [0x8000_0001, 10, 11, 12]);
        call(&VSLIDE1DOWN, &mut e, d, &[7]);
        let got: Vec<u64> = (0..4).map(|i| vget(&e, 4, i, 2)).collect();
        assert_eq!(got, [11, 12, 13, 7]);
    }

    #[test]
    fn gather_and_compress() {
        let mut e = env(0, 6);
        fill(&mut e, 2, 0, &[10, 11, 12, 13, 14, 15]);
        fill(&mut e, 1, 0, &[5, 0, 200, 3, 16, 1]);
        let d = Desc { vm: true, vd: 4, vs1: 1, vs2: 2, esz: 0, vta: true, ..Desc::default() };
        call(&VRGATHER_VV, &mut e, d, &[]);
        let got: Vec<u64> = (0..6).map(|i| vget(&e, 4, i, 0)).collect();
        // VLMAX is 16, so index 200 reads zero; index 15 is past vl but below VLMAX.
        assert_eq!(got, [15, 10, 0, 13, 0, 11]);
        assert_eq!(vget(&e, 4, 6, 0), 0xff);
        // vrgatherei16.vv with SEW 8: 16-bit indices.
        fill(&mut e, 1, 1, &[2, 0x100, 4]);
        st64(&mut e, VL, 3);
        call(&VRGATHER_VV, &mut e, Desc { x: 1, ..d }, &[]);
        let got: Vec<u64> = (0..3).map(|i| vget(&e, 4, i, 0)).collect();
        assert_eq!(got, [12, 0, 14]);
        call(&VRGATHER_VX, &mut e, d, &[3]);
        let got: Vec<u64> = (0..3).map(|i| vget(&e, 4, i, 0)).collect();
        assert_eq!(got, [13, 13, 13]);
        // vcompress.vm.
        st64(&mut e, VL, 6);
        e[voff(1, 0)] = 0b10_1010;
        call(&VCOMPRESS_VM, &mut e, d, &[]);
        let got: Vec<u64> = (0..4).map(|i| vget(&e, 4, i, 0)).collect();
        assert_eq!(got, [11, 13, 15, 0xff]);
    }

    #[test]
    fn scalar_and_whole_moves() {
        let mut e = env(1, 4);
        vset(&mut e, 2, 0, 1, 0x8001);
        let d = Desc { vm: true, vd: 3, vs2: 2, esz: 1, vta: true, ..Desc::default() };
        assert_eq!(call(&VMV_X_S, &mut e, d, &[]), 0xffff_ffff_ffff_8001);
        assert_eq!(call(&VMV_X_S, &mut e, Desc { x: 1, ..d }, &[]), 0xffff_ffff_ffff_8001);
        vset(&mut e, 2, 0, 1, 0x7001);
        assert_eq!(call(&VMV_X_S, &mut e, d, &[]), 0x7001);
        assert_eq!(call(&VMV_X_S, &mut e, Desc { x: 1, ..d }, &[]), 0xffff_ffff_ffff_7001);
        call(&VMV_S_X, &mut e, d, &[0x1_2345]);
        assert_eq!(vget(&e, 3, 0, 1), 0x2345);
        assert_eq!(vget(&e, 3, 7, 1), 0xffff);
        // Nothing is written when vstart >= vl, but vstart is cleared.
        st64(&mut e, VSTART, 4);
        call(&VMV_S_X, &mut e, d, &[0x55]);
        assert_eq!(vget(&e, 3, 0, 1), 0x2345);
        assert_eq!(vstart(&e), 0);
        // vmv2r.v from element 3 of SEW 16.
        for b in 0..32 {
            e[voff(4, b)] = b as u8;
        }
        st64(&mut e, VSTART, 3);
        call(&VMVR_V, &mut e, Desc { vd: 6, vs2: 4, nf: 2, ..Desc::default() }, &[]);
        assert_eq!(e[voff(6, 5)], 0);
        assert_eq!(e[voff(6, 6)], 6);
        assert_eq!(e[voff(6, 31)], 31);
        assert_eq!(vstart(&e), 0);
    }
}
