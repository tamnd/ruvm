// SPDX-License-Identifier: GPL-2.0-or-later

//! The vector helpers: the framework of QEMU's `target/riscv/tcg/vector_helper.c` and
//! `vector_internals.c`, `vsetvl` and the loads and stores. The integer and fixed point
//! arithmetic helpers are in [`super::vector_int`], the integer reductions and the mask
//! and permutation helpers in [`super::vector_perm`], the floating point ones in
//! [`super::vector_fp`] and the crypto ones in [`super::vcrypto`].
//!
//! The vector registers live in `env` at [`VREG`], `VLENB` bytes each, element `i` of a
//! register group at byte `i * esz` from the start of the group, little endian on every
//! host (QEMU keeps host order in 64-bit words and fixes the index with the `H*` macros).
//!
//! A helper does not get pointers to its registers as in QEMU. It gets a descriptor, a
//! [`Desc`], with QEMU's `VDATA` bits, the register numbers and the element size, and it
//! reads the registers out of `env`. So a single helper serves every SEW where QEMU has
//! one per SEW (`vadd_vv_b` to `vadd_vv_d`), and the `.vx` and `.vi` forms share the
//! helper of the `.vv` form with [`Desc::scalar`] set.
//!
//! Deliberate differences from QEMU:
//!
//! - QEMU's unit stride and whole register loads and stores probe a page at a time and
//!   copy from the host page when no element can fault or hit I/O. Here every element goes
//!   through the softmmu, which leaves the same memory, registers and `vstart` behind,
//!   including when an element faults.
//! - QEMU's fault-only-first loads first probe the whole range, and only go element by
//!   element when that probe finds something other than RAM. Here every active element
//!   after the first is probed on its own, which finds the same first element that is not
//!   in RAM.
//! - QEMU asserts when a whole register load or store starts with `vstart` past the end of
//!   the registers; here the instruction does nothing but clear `vstart`.

use ruvm_jit::cputlb::{cpu_ld_mmu, cpu_st_mmu, probe_access, probe_access_nonfault};
use ruvm_jit::{Cpu, CpuLoopExit, MmuAccessType, Ra};
use ruvm_jit_core::HelperType::{I32, I64, Ptr, Void};
use ruvm_jit_core::types::call_flags::{NO_RWG, NO_WG};
use ruvm_jit_core::{MemOp, MemOpIdx};
use ruvm_jit_interp::{HelperEnv, Unwind};

use super::helpers::{Def, run};
use super::pm::{self, PointerMask};
use super::{ld64, mmu_index, st64};
use crate::cfg::RiscvCfg;
use crate::cpu::{
    VILL, VL, VLENB, VREG, VSTART, VTYPE, VTYPE_ALTFMT, VTYPE_VLMUL, VTYPE_VSEW, get_field,
};

pub(super) type HR = Result<u128, Unwind>;
type R<T> = Result<T, CpuLoopExit>;

macro_rules! def {
    ($id:ident, $name:literal, $flags:expr, $ret:expr, [$($a:expr),*], $f:expr) => {
        pub(crate) const $id: $crate::tcg::helpers::Def = $crate::tcg::helpers::Def {
            name: $name,
            flags: $flags,
            ret: $ret,
            args: &[$($a),*],
            f: $f,
        };
    };
}
pub(super) use def;

/// The size of the vector register file in bytes.
const VREGS_SIZE: usize = 32 * VLENB;

/// `RV_VLEN_MAX`.
pub(super) const RV_VLEN_MAX: u64 = 1024;

// The descriptor.

/// The descriptor of a vector helper call: QEMU's `simd_desc()` data (the `VDATA` fields)
/// plus what QEMU passes as register pointers.
///
/// The encoding: `VDATA` in bits 0 to 10 as in QEMU (VM 0, LMUL 1 to 3, VTA 4,
/// VTA_ALL_1S 5, VMA 6, NF 7 to 10), `vd` in bits 11 to 15, `vs1` in bits 16 to 20, `vs2`
/// in bits 21 to 25, log2 of the element size in bits 26 and 27, [`Desc::scalar`] in bit
/// 28 and three bits for the helper in 29 to 31.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Desc {
    /// `VDATA.VM`: unmasked.
    pub(super) vm: bool,
    /// `VDATA.LMUL`, signed.
    pub(super) lmul: i32,
    /// `VDATA.VTA`.
    pub(super) vta: bool,
    /// `VDATA.VTA_ALL_1S`.
    pub(super) vta_all_1s: bool,
    /// `VDATA.VMA`.
    pub(super) vma: bool,
    /// `VDATA.NF`.
    pub(super) nf: u32,
    /// The destination register.
    pub(super) vd: u32,
    /// The first source register.
    pub(super) vs1: u32,
    /// The second source register.
    pub(super) vs2: u32,
    /// log2 of the element size in bytes, usually SEW.
    pub(super) esz: u32,
    /// The first source is the scalar argument (`.vx`, `.vi`, `.vf`), not `vs1`.
    pub(super) scalar: bool,
    /// Three bits for the helper.
    pub(super) x: u32,
}

impl Desc {
    /// The descriptor as the helper gets it.
    pub(super) fn encode(&self) -> u32 {
        u32::from(self.vm)
            | ((self.lmul as u32 & 7) << 1)
            | (u32::from(self.vta) << 4)
            | (u32::from(self.vta_all_1s) << 5)
            | (u32::from(self.vma) << 6)
            | ((self.nf & 15) << 7)
            | ((self.vd & 31) << 11)
            | ((self.vs1 & 31) << 16)
            | ((self.vs2 & 31) << 21)
            | ((self.esz & 3) << 26)
            | (u32::from(self.scalar) << 28)
            | ((self.x & 7) << 29)
    }

    /// The descriptor of a helper call.
    pub(super) fn decode(d: u32) -> Desc {
        Desc {
            vm: d & 1 != 0,
            lmul: (((d >> 1) & 7) as i32) << 29 >> 29,
            vta: d & (1 << 4) != 0,
            vta_all_1s: d & (1 << 5) != 0,
            vma: d & (1 << 6) != 0,
            nf: (d >> 7) & 15,
            vd: (d >> 11) & 31,
            vs1: (d >> 16) & 31,
            vs2: (d >> 21) & 31,
            esz: (d >> 26) & 3,
            scalar: d & (1 << 28) != 0,
            x: (d >> 29) & 7,
        }
    }
}

// The registers.

/// The offset in `env` of byte `byte` of the register group that starts at `reg`.
#[inline]
pub(super) fn voff(reg: u32, byte: usize) -> usize {
    VREG + ((reg as usize * VLENB + byte) & (VREGS_SIZE - 1))
}

/// Element `i` of `1 << log2` bytes of the register group at `reg`, zero extended.
#[inline]
pub(super) fn vget(env: &[u8], reg: u32, i: usize, log2: u32) -> u64 {
    let o = voff(reg, i << log2);
    match log2 {
        0 => u64::from(env[o]),
        1 => u64::from(u16::from_le_bytes([env[o], env[o + 1]])),
        2 => u64::from(u32::from_le_bytes(env[o..o + 4].try_into().expect("4 bytes"))),
        _ => u64::from_le_bytes(env[o..o + 8].try_into().expect("8 bytes")),
    }
}

/// Set element `i` of `1 << log2` bytes of the register group at `reg` to the low bits of
/// `v`.
#[inline]
pub(super) fn vset(env: &mut [u8], reg: u32, i: usize, log2: u32, v: u64) {
    let o = voff(reg, i << log2);
    let n = 1 << log2;
    env[o..o + n].copy_from_slice(&v.to_le_bytes()[..n]);
}

/// `vext_elem_mask()`: bit `i` of the mask register `reg`.
#[inline]
pub(super) fn vmask(env: &[u8], reg: u32, i: usize) -> bool {
    env[voff(reg, i / 8)] >> (i % 8) & 1 != 0
}

/// `vext_set_elem_mask()`: set bit `i` of the mask register `reg`.
#[inline]
pub(super) fn vset_mask(env: &mut [u8], reg: u32, i: usize, v: bool) {
    let o = voff(reg, i / 8);
    let bit = 1u8 << (i % 8);
    if v {
        env[o] |= bit;
    } else {
        env[o] &= !bit;
    }
}

/// `vext_set_elems_1s()`: when `agnostic`, set bytes `cnt` to `tot` of the register group
/// at `reg` to all ones.
pub(super) fn set_1s(env: &mut [u8], reg: u32, agnostic: bool, cnt: usize, tot: usize) {
    if !agnostic {
        return;
    }
    for b in cnt..tot {
        env[voff(reg, b)] = 0xff;
    }
}

// The state.

/// `env->vl`.
#[inline]
pub(super) fn vl(env: &[u8]) -> usize {
    ld64(env, VL) as usize
}

/// `env->vstart`.
#[inline]
pub(super) fn vstart(env: &[u8]) -> usize {
    ld64(env, VSTART) as usize
}

/// Set `env->vstart`.
#[inline]
pub(super) fn set_vstart(env: &mut [u8], v: usize) {
    st64(env, VSTART, v as u64);
}

/// log2 of SEW in bytes, `vtype.vsew`.
#[inline]
pub(super) fn vsew(env: &[u8]) -> u32 {
    get_field(ld64(env, VTYPE), VTYPE_VSEW) as u32
}

/// `vext_max_elems()`: VLMAX for elements of `1 << log2` bytes and the LMUL of `d`.
pub(super) fn max_elems(d: &Desc, log2: u32) -> usize {
    let scale = d.lmul - log2 as i32;
    if scale < 0 { VLENB >> -scale } else { VLENB << scale }
}

/// `vext_get_total_elems()`: the elements of `1 << log2` bytes in the destination
/// register group, at least one register.
pub(super) fn total_elems(env: &[u8], d: &Desc, log2: u32) -> usize {
    let emul = (log2 as i32 - vsew(env) as i32 + d.lmul).max(0);
    (VLENB << emul) >> log2
}

/// The loop of most helpers, `do_vext_vv()` and its kind: `body` for each active element
/// from `vstart` to `vl`, masked off elements of `1 << dlog2` bytes of `vd` set to ones
/// when mask agnostic, then the tail set to ones when tail agnostic.
pub(super) fn for_each(
    env: &mut [u8],
    d: &Desc,
    dlog2: u32,
    mut body: impl FnMut(&mut [u8], usize),
) {
    let vl = vl(env);
    let start = vstart(env);
    if start >= vl {
        // VSTART_CHECK_EARLY_EXIT().
        set_vstart(env, 0);
        return;
    }
    for i in start..vl {
        if !d.vm && !vmask(env, 0, i) {
            set_1s(env, d.vd, d.vma, i << dlog2, (i + 1) << dlog2);
            continue;
        }
        body(env, i);
    }
    set_vstart(env, 0);
    let total = total_elems(env, d, dlog2);
    set_1s(env, d.vd, d.vta, vl << dlog2, total << dlog2);
}

/// Sign extend the low `8 << log2` bits of `v`.
#[inline]
pub(super) fn sext(v: u64, log2: u32) -> i64 {
    let sh = 64 - (8 << log2);
    ((v << sh) as i64) >> sh
}

/// The low `8 << log2` bits of `v`.
#[inline]
pub(super) fn trunc(v: u64, log2: u32) -> u64 {
    if log2 >= 3 { v } else { v & ((1u64 << (8 << log2)) - 1) }
}

// The element-wise arithmetic.

/// The element sizes of an element-wise operation, relative to SEW.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Shape {
    /// Every operand is SEW.
    Single,
    /// `.vv`, `.vx`: a 2*SEW result from SEW operands.
    Widen,
    /// `.wv`, `.wx`: a 2*SEW result from a 2*SEW `vs2` and a SEW `vs1`.
    WidenW,
    /// A SEW result from a 2*SEW `vs2` and a SEW `vs1`.
    Narrow,
}

impl Shape {
    /// log2 of the sizes in bytes of the destination, `vs2` and `vs1` elements.
    pub(super) fn log2(self, sew: u32) -> (u32, u32, u32) {
        match self {
            Shape::Single => (sew, sew, sew),
            Shape::Widen => (sew + 1, sew, sew),
            Shape::WidenW => (sew + 1, sew + 1, sew),
            Shape::Narrow => (sew, sew + 1, sew),
        }
    }
}

/// The first source of element `i`: the scalar of a `.vx`, `.vi` or `.vf` operation
/// truncated to `1 << log2` bytes, or element `i` of `vs1`.
#[inline]
pub(super) fn src1(env: &[u8], d: &Desc, i: usize, log2: u32, scalar: u64) -> u64 {
    if d.scalar { trunc(scalar, log2) } else { vget(env, d.vs1, i, log2) }
}

/// `do_vext_vv()` and `do_vext_vx()`: `vd[i] = op(vs2[i], s1, sew)` with the arguments of
/// a helper, `env`, the descriptor and the scalar. The operands are zero extended from
/// their size; `sew` is log2 of SEW in bytes, so that `op` can sign extend them.
pub(super) fn vop(env: &mut [u8], a: &[u64], shape: Shape, op: impl Fn(u64, u64, u32) -> u64) {
    let d = Desc::decode(a[1] as u32);
    let scalar = a[2];
    let sew = d.esz;
    let (dl, l2, l1) = shape.log2(sew);
    for_each(env, &d, dl, |env, i| {
        let s2 = vget(env, d.vs2, i, l2);
        let s1 = src1(env, &d, i, l1, scalar);
        vset(env, d.vd, i, dl, op(s2, s1, sew));
    });
}

/// A helper of the element-wise operation `op` with [`vop`]: `vop_def!(ID, "name", shape,
/// |s2, s1, sew| ...)`.
macro_rules! vop_def {
    ($id:ident, $name:literal, $shape:expr, $op:expr) => {
        $crate::tcg::vector::def!(
            $id,
            $name,
            ruvm_jit_core::types::call_flags::NO_RWG,
            ruvm_jit_core::HelperType::Void,
            [
                ruvm_jit_core::HelperType::Ptr,
                ruvm_jit_core::HelperType::I32,
                ruvm_jit_core::HelperType::I64
            ],
            {
                fn h(e: &mut ruvm_jit_interp::HelperEnv<'_>, a: &[u64]) -> $crate::tcg::vector::HR {
                    $crate::tcg::vector::vop(e.env, a, $shape, $op);
                    Ok(0)
                }
                h
            }
        );
    };
}
pub(super) use vop_def;

// vsetvl.

def!(VSETVL, "vsetvl", NO_RWG, I64, [Ptr, I64, I64, I64], h_vsetvl);

/// `reset_ill_vtype()`.
fn reset_ill_vtype(env: &mut [u8]) {
    st64(env, VILL, 1);
    st64(env, VTYPE, 0);
    st64(env, VL, 0);
    st64(env, VSTART, 0);
}

/// `vext_get_vlmax()`.
pub(super) fn get_vlmax(vsew: u32, lmul: i32) -> u64 {
    let vlen = (VLENB as u64) << 3;
    vlen >> (vsew as i32 + 3 - lmul)
}

/// The `vsetvl` flag that says both `rd` and `rs1` are `x0`.
pub(super) const VSETVL_X0: u64 = 1;
/// The `vsetvl` flag for the `rvv_vl_half_avl` property.
const VSETVL_HALF_AVL: u64 = 2;
/// The `vsetvl` flag for the `rvv_vsetvl_x0_vill` property.
const VSETVL_X0_VILL: u64 = 4;
/// The shift of ELEN in the `vsetvl` flags.
const VSETVL_ELEN_SHIFT: u32 = 8;

/// The `vsetvl` flags of configuration `cfg`, without [`VSETVL_X0`]. The helper has no
/// CPU, so the translator passes the parts of the configuration it needs as a constant.
pub(super) fn vsetvl_flags(cfg: &RiscvCfg) -> u64 {
    let mut f = u64::from(cfg.elen) << VSETVL_ELEN_SHIFT;
    if cfg.rvv_vl_half_avl {
        f |= VSETVL_HALF_AVL;
    }
    if cfg.rvv_vsetvl_x0_vill {
        f |= VSETVL_X0_VILL;
    }
    f
}

/// `HELPER(vsetvl)`: `vl` and `vtype` from AVL `s1` and the new `vtype` `s2`. The fourth
/// argument has the flags of [`vsetvl_flags`] and [`VSETVL_X0`].
fn h_vsetvl(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    Ok(u128::from(vsetvl(h.env, a[1], a[2], a[3])))
}

/// `HELPER(vsetvl)` with the flags `flags`.
pub(super) fn vsetvl(env: &mut [u8], s1: u64, s2: u64, flags: u64) -> u64 {
    let vlmul = s2 & VTYPE_VLMUL;
    let vsew = get_field(s2, VTYPE_VSEW) as u32;
    let sew = 8u64 << vsew;
    let altfmt = s2 & VTYPE_ALTFMT != 0;
    let mut vill = s2 >> 63 != 0;
    let elen = flags >> VSETVL_ELEN_SHIFT;

    if vlmul & 4 != 0 {
        // Fractional LMUL: ELEN * LMUL >= SEW.
        if vlmul == 4 || (elen >> (8 - vlmul)) < sew {
            vill = true;
        }
    }
    // There is no Zvfbfa, so altfmt is illegal at every SEW.
    if altfmt {
        vill = true;
    }
    // vtype_reserved(): bits 8 to 62 without Zvfbfa.
    let reserved = s2 & (((1u64 << 55) - 1) << 8);
    if sew > elen || vill || reserved != 0 {
        reset_ill_vtype(env);
        return 0;
    }

    let lmul = ((vlmul as i32) << 29) >> 29;
    let vlmax = get_vlmax(vsew, lmul);
    let vl = if s1 <= vlmax {
        s1
    } else if s1 < 2 * vlmax && flags & VSETVL_HALF_AVL != 0 {
        (s1 + 1) >> 1
    } else {
        vlmax
    };

    if flags & VSETVL_X0_VILL != 0 && flags & VSETVL_X0 != 0 && ld64(env, VL) != vl {
        reset_ill_vtype(env);
        return 0;
    }

    st64(env, VL, vl);
    st64(env, VTYPE, s2);
    st64(env, VSTART, 0);
    st64(env, VILL, 0);
    vl
}

// Loads and stores.

/// The memory operation of an element of `1 << log2` bytes.
fn elem_memop(log2: u32) -> MemOp {
    match log2 {
        0 => MemOp::UB,
        1 => MemOp::LEUW,
        2 => MemOp::LEUL,
        _ => MemOp::LEUQ,
    }
}

/// `riscv_env_mmu_index(env, false)`.
fn data_mmu_idx(env: &[u8]) -> usize {
    mmu_index(env, false)
}

/// Load or store element `idx` of `1 << log2` bytes of the register group at `vd` from or
/// to `addr`: `lde_*_tlb()` and `ste_*_tlb()`.
#[inline]
fn ldst_elem(
    cpu: &mut Cpu<'_>,
    store: bool,
    addr: u64,
    vd: u32,
    idx: usize,
    log2: u32,
    mmu_idx: usize,
) -> R<()> {
    let oi = MemOpIdx::new(elem_memop(log2), mmu_idx as u32);
    if store {
        let v = vget(cpu.env, vd, idx, log2);
        cpu_st_mmu(cpu, addr, v, oi, Ra::Tb)
    } else {
        let v = cpu_ld_mmu(cpu, addr, oi, Ra::Tb)?;
        vset(cpu.env, vd, idx, log2, v);
        Ok(())
    }
}

/// `vext_set_tail_elems_1s()`.
fn set_tail_elems_1s(env: &mut [u8], vl: usize, d: &Desc, nf: usize, log2: u32, max: usize) {
    if !d.vta {
        return;
    }
    for k in 0..nf {
        set_1s(env, d.vd, true, (k * max + vl) << log2, (k * max + max) << log2);
    }
}

/// The address of field `k` of element `i`.
enum Addr {
    /// Unit stride: `base + ((i * nf + k) << log2)`.
    Unit,
    /// Strided: `base + stride * i + (k << log2)`.
    Stride(u64),
    /// Indexed: `base + vs2[i] + (k << log2)`, the index of `1 << log2` bytes in register
    /// group `vs2`.
    Index(u32, u32),
}

/// `vext_ldst_stride()`, `vext_ldst_index()` and `vext_ldst_us()`: load or store the
/// active elements from `vstart` to `evl`.
fn ldst(
    cpu: &mut Cpu<'_>,
    d: &Desc,
    base: u64,
    addr_of: Addr,
    store: bool,
    evl: usize,
    vm: bool,
) -> R<u64> {
    let nf = d.nf as usize;
    let log2 = d.esz;
    let max = max_elems(d, log2);
    let mmu_idx = data_mmu_idx(cpu.env);
    let pm = pm::cpu_data_mask(cpu);
    let start = vstart(cpu.env);
    if start >= evl {
        set_vstart(cpu.env, 0);
        return Ok(0);
    }
    for i in start..evl {
        for k in 0..nf {
            if !vm && !vmask(cpu.env, 0, i) {
                let e = i + k * max;
                set_1s(cpu.env, d.vd, d.vma, e << log2, (e + 1) << log2);
                continue;
            }
            let addr = match addr_of {
                Addr::Unit => base.wrapping_add(((i * nf + k) as u64) << log2),
                Addr::Stride(stride) => base
                    .wrapping_add(stride.wrapping_mul(i as u64))
                    .wrapping_add((k as u64) << log2),
                Addr::Index(vs2, ilog2) => {
                    base.wrapping_add(vget(cpu.env, vs2, i, ilog2)).wrapping_add((k as u64) << log2)
                }
            };
            ldst_elem(cpu, store, pm.adjust(addr), d.vd, i + k * max, log2, mmu_idx)?;
        }
        set_vstart(cpu.env, i + 1);
    }
    set_vstart(cpu.env, 0);
    if !store {
        set_tail_elems_1s(cpu.env, evl, d, nf, log2, max);
    }
    Ok(0)
}

def!(VLE, "vle", NO_WG, Void, [Ptr, I32, I64], h_ldst_us::<false>);
def!(VSE, "vse", NO_WG, Void, [Ptr, I32, I64], h_ldst_us::<true>);
def!(VLM, "vlm", NO_WG, Void, [Ptr, I32, I64], h_ldst_m::<false>);
def!(VSM, "vsm", NO_WG, Void, [Ptr, I32, I64], h_ldst_m::<true>);
def!(VLSE, "vlse", NO_WG, Void, [Ptr, I32, I64, I64], h_ldst_stride::<false>);
def!(VSSE, "vsse", NO_WG, Void, [Ptr, I32, I64, I64], h_ldst_stride::<true>);
def!(VLXEI, "vlxei", NO_WG, Void, [Ptr, I32, I64], h_ldst_index::<false>);
def!(VSXEI, "vsxei", NO_WG, Void, [Ptr, I32, I64], h_ldst_index::<true>);
def!(VLEFF, "vleff", NO_WG, Void, [Ptr, I32, I64], h_ldff);
def!(VLRE, "vlre", NO_WG, Void, [Ptr, I32, I64], h_ldst_whole::<false>);
def!(VSR, "vsr", NO_WG, Void, [Ptr, I32, I64], h_ldst_whole::<true>);

/// `HELPER(vle*_v)`, `HELPER(vse*_v)` and their `_mask` forms: unit stride, the masked
/// ones as a stride of `nf << esz`, as in QEMU.
fn h_ldst_us<const STORE: bool>(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = Desc::decode(a[1] as u32);
    let base = a[2];
    run(h, |cpu| {
        let vl = vl(cpu.env);
        if d.vm {
            ldst(cpu, &d, base, Addr::Unit, STORE, vl, true)
        } else {
            let stride = u64::from(d.nf) << d.esz;
            ldst(cpu, &d, base, Addr::Stride(stride), STORE, vl, false)
        }
    })
}

/// `HELPER(vlm_v)` and `HELPER(vsm_v)`: `ceil(vl / 8)` bytes.
fn h_ldst_m<const STORE: bool>(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = Desc::decode(a[1] as u32);
    let base = a[2];
    run(h, |cpu| {
        // The evl of QEMU is a uint8_t.
        let evl = ((vl(cpu.env) + 7) >> 3) & 0xff;
        ldst(cpu, &d, base, Addr::Unit, STORE, evl, true)
    })
}

/// `HELPER(vlse*_v)` and `HELPER(vsse*_v)`.
fn h_ldst_stride<const STORE: bool>(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = Desc::decode(a[1] as u32);
    let (base, stride) = (a[2], a[3]);
    run(h, |cpu| {
        let vl = vl(cpu.env);
        ldst(cpu, &d, base, Addr::Stride(stride), STORE, vl, d.vm)
    })
}

/// `HELPER(vlxei*_v)` and `HELPER(vsxei*_v)`: the index EEW is in [`Desc::x`].
fn h_ldst_index<const STORE: bool>(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = Desc::decode(a[1] as u32);
    let base = a[2];
    run(h, |cpu| {
        let vl = vl(cpu.env);
        ldst(cpu, &d, base, Addr::Index(d.vs2, d.x), STORE, vl, d.vm)
    })
}

/// `HELPER(vle*ff_v)`: `vext_ldff()`.
fn h_ldff(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = Desc::decode(a[1] as u32);
    let base = a[2];
    run(h, |cpu| {
        let nf = d.nf as usize;
        let log2 = d.esz;
        let msize = (nf as u64) << log2;
        let mmu_idx = data_mmu_idx(cpu.env);
        let pm = pm::cpu_data_mask(cpu);
        let env_vl = vl(cpu.env);
        let start = vstart(cpu.env);
        if start >= env_vl {
            set_vstart(cpu.env, 0);
            return Ok(0);
        }
        let mut new_vl = 0;
        for i in start..env_vl {
            if !d.vm && !vmask(cpu.env, 0, i) {
                continue;
            }
            let addr = pm.adjust(base.wrapping_add(i as u64 * msize));
            if i == 0 {
                // Allow a fault on the first element.
                probe_pages(cpu, addr, msize, mmu_idx, pm)?;
            } else if !probe_pages_nonfault(cpu, addr, msize, mmu_idx, pm)? {
                // Stop at an element that is not mapped or is not RAM.
                new_vl = i;
                break;
            }
        }
        if new_vl != 0 {
            st64(cpu.env, VL, new_vl as u64);
        }
        let vl = vl(cpu.env);
        if vstart(cpu.env) < vl {
            let addr_of = if d.vm { Addr::Unit } else { Addr::Stride(msize) };
            ldst(cpu, &d, base, addr_of, false, vl, d.vm)
        } else {
            set_vstart(cpu.env, 0);
            set_tail_elems_1s(cpu.env, vl, &d, nf, log2, max_elems(&d, log2));
            Ok(0)
        }
    })
}

/// The target page size.
const PAGE_SIZE: u64 = 4096;

/// `probe_pages()` without flags: fault if `len` bytes at `addr`, on at most two pages,
/// cannot be loaded. The address of each page is masked with `pm`.
fn probe_pages(cpu: &mut Cpu<'_>, addr: u64, len: u64, mmu_idx: usize, pm: PointerMask) -> R<()> {
    let pagelen = PAGE_SIZE - (addr & (PAGE_SIZE - 1));
    let cur = pagelen.min(len);
    let first = pm.adjust(addr);
    probe_access(cpu, first, cur as usize, MmuAccessType::DataLoad, mmu_idx, Ra::Tb)?;
    if len > cur {
        let addr = pm.adjust(addr.wrapping_add(cur));
        probe_access(cpu, addr, (len - cur) as usize, MmuAccessType::DataLoad, mmu_idx, Ra::Tb)?;
    }
    Ok(())
}

/// The non faulting probe of `vext_ldff()` for an element after the first: whether every
/// page of `len` bytes at `addr` is mapped RAM. The address of each next page is masked with
/// `pm`.
fn probe_pages_nonfault(
    cpu: &mut Cpu<'_>,
    addr: u64,
    len: u64,
    mmu_idx: usize,
    pm: PointerMask,
) -> R<bool> {
    let mut addr = addr;
    let mut remain = len;
    loop {
        let offset = PAGE_SIZE - (addr & (PAGE_SIZE - 1));
        match probe_access_nonfault(cpu, addr, MmuAccessType::DataLoad, mmu_idx, Ra::Tb)? {
            Some(true) => {}
            _ => return Ok(false),
        }
        if remain <= offset {
            return Ok(true);
        }
        remain -= offset;
        addr = pm.adjust(addr.wrapping_add(offset));
    }
}

/// `HELPER(vl*re*_v)` and `HELPER(vs*r_v)`: `vext_ldst_whole()`, NF registers whatever
/// `vl` and `vtype` are.
fn h_ldst_whole<const STORE: bool>(h: &mut HelperEnv<'_>, a: &[u64]) -> HR {
    let d = Desc::decode(a[1] as u32);
    let base = a[2];
    run(h, |cpu| {
        let log2 = d.esz;
        let evl = (d.nf as usize * VLENB) >> log2;
        let mmu_idx = data_mmu_idx(cpu.env);
        let pm = pm::cpu_data_mask(cpu);
        let start = vstart(cpu.env);
        for i in start..evl {
            let addr = pm.adjust(base.wrapping_add((i as u64) << log2));
            ldst_elem(cpu, STORE, addr, d.vd, i, log2, mmu_idx)?;
            set_vstart(cpu.env, i + 1);
        }
        set_vstart(cpu.env, 0);
        Ok(0)
    })
}

/// The helpers of this module.
pub(crate) const ALL: &[Def] =
    &[VSETVL, VLE, VSE, VLM, VSM, VLSE, VSSE, VLXEI, VSXEI, VLEFF, VLRE, VSR];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{ENV_SIZE, RiscvCfg};

    #[test]
    fn desc_round_trip() {
        let d = Desc {
            vm: true,
            lmul: -3,
            vta: true,
            vta_all_1s: false,
            vma: true,
            nf: 8,
            vd: 31,
            vs1: 17,
            vs2: 2,
            esz: 3,
            scalar: true,
            x: 5,
        };
        assert_eq!(Desc::decode(d.encode()), d);
        assert_eq!(Desc::decode(Desc::default().encode()), Desc::default());
    }

    #[test]
    fn registers_are_little_endian() {
        let mut env = vec![0u8; ENV_SIZE];
        vset(&mut env, 1, 1, 2, 0x1122_3344);
        assert_eq!(&env[voff(1, 4)..voff(1, 8)], &[0x44, 0x33, 0x22, 0x11]);
        assert_eq!(vget(&env, 1, 3, 1), 0x1122);
        // Element 8 of 32 bits in the group at v1 is element 0 of v3.
        vset(&mut env, 1, 8, 2, 0xdead_beef);
        assert_eq!(vget(&env, 3, 0, 2), 0xdead_beef);
        vset_mask(&mut env, 0, 9, true);
        assert!(vmask(&env, 0, 9));
        assert_eq!(env[voff(0, 1)], 2);
        assert_eq!(sext(0x80, 0), -128);
        assert_eq!(trunc(0x1_2345, 1), 0x2345);
    }

    #[test]
    fn vsetvl_rules() {
        let cfg = vsetvl_flags(&RiscvCfg::max());
        let mut env = vec![0u8; ENV_SIZE];
        // e32, m1: VLMAX 4.
        assert_eq!(vsetvl(&mut env, 10, 0x10, cfg), 4);
        assert_eq!(ld64(&env, VTYPE), 0x10);
        assert_eq!(ld64(&env, VILL), 0);
        assert_eq!(vsetvl(&mut env, 3, 0x10, cfg), 3);
        // e8, m8: VLMAX 128.
        assert_eq!(vsetvl(&mut env, RV_VLEN_MAX, 0x03, cfg), 128);
        // e64, mf2 needs ELEN 128.
        assert_eq!(vsetvl(&mut env, 1, 0x1f, cfg), 0);
        assert_eq!(ld64(&env, VILL), 1);
        assert_eq!(ld64(&env, VTYPE), 0);
        assert_eq!(ld64(&env, VL), 0);
        // e8, mf8: VLMAX 2.
        assert_eq!(vsetvl(&mut env, 7, 0x05, cfg), 2);
        // vlmul 4 is reserved.
        assert_eq!(vsetvl(&mut env, 7, 0x04, cfg), 0);
        // SEW 128 is above ELEN.
        assert_eq!(vsetvl(&mut env, 7, 0x20, cfg), 0);
        // altfmt and the other reserved bits.
        assert_eq!(vsetvl(&mut env, 7, 0x100, cfg), 0);
        assert_eq!(vsetvl(&mut env, 7, 1 << 62, cfg), 0);
        // vill set in the new vtype.
        assert_eq!(vsetvl(&mut env, 7, 1 << 63, cfg), 0);
        // vta and vma are kept.
        assert_eq!(vsetvl(&mut env, 1, 0xc0, cfg), 1);
        assert_eq!(ld64(&env, VTYPE), 0xc0);
        assert_eq!(get_vlmax(0, 3), 128);
        assert_eq!(get_vlmax(3, -1), 1);
    }

    #[test]
    fn element_counts() {
        let mut env = vec![0u8; ENV_SIZE];
        st64(&mut env, VTYPE, 0x10);
        let d = Desc { lmul: 1, ..Desc::default() };
        assert_eq!(max_elems(&d, 2), 8);
        assert_eq!(max_elems(&Desc { lmul: -2, ..d }, 0), 4);
        // SEW 32, LMUL 2: 8 elements of 32 bits, 16 of 64 bits for a widening result.
        assert_eq!(total_elems(&env, &d, 2), 8);
        assert_eq!(total_elems(&env, &d, 3), 8);
        assert_eq!(total_elems(&env, &Desc { lmul: -1, ..d }, 2), 4);
    }

    #[test]
    fn for_each_masks_and_tails() {
        let mut env = vec![0u8; ENV_SIZE];
        st64(&mut env, VTYPE, 0x00);
        st64(&mut env, VL, 4);
        st64(&mut env, VSTART, 1);
        env[voff(0, 0)] = 0b1011;
        let d = Desc { vd: 2, vma: true, vta: true, ..Desc::default() };
        let mut seen = Vec::new();
        for_each(&mut env, &d, 0, |env, i| {
            seen.push(i);
            vset(env, 2, i, 0, 7);
        });
        assert_eq!(seen, [1, 3]);
        assert_eq!(vget(&env, 2, 0, 0), 0);
        assert_eq!(vget(&env, 2, 2, 0), 0xff);
        assert_eq!(vget(&env, 2, 3, 0), 7);
        assert_eq!(vget(&env, 2, 15, 0), 0xff);
        assert_eq!(vstart(&env), 0);
    }
}
