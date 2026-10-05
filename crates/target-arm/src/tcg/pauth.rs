// SPDX-License-Identifier: GPL-2.0-or-later

//! FEAT_PAuth, QEMU's `pauth_helper.c`: the QARMA5 and QARMA3 block ciphers, the
//! IMPLEMENTATION DEFINED `qemu_xxhash64_4()` algorithm, and the PAC, AUT, XPAC and PACGA
//! helpers. The helpers raise their traps and authentication failures by unwinding to the
//! instruction, as QEMU's do with `GETPC()`.

use ruvm_jit::{Cpu, CpuLoopExit, Ra};
use ruvm_jit_interp::{HelperEnv, Unwind};

use super::helpers::{Def, def, raise_exception, run};
use super::{arm_of, exception_target_el, ptw, sysreg};
use crate::cpu::{
    ArmFeatures, CpuArmState, EXCP_UDEF, HCR_API, HCR_E2H, HCR_TGE, PAUTH_2, PAUTH_EPAC,
    PAUTH_FPAC, PAUTH_FPACCOMBINED, PauthAlg, SCR_API, SCTLR_ENDA, SCTLR_ENDB, SCTLR_ENIA,
    SCTLR_ENIB,
};
use crate::syndrome::{syn_pacfail, syn_pactrap};

use ruvm_jit_core::HelperType::{I64, Ptr};

fn extract(v: u64, start: u32, len: u32) -> u64 {
    (v >> start) & (u64::MAX >> (64 - len))
}

fn sextract(v: u64, start: u32, len: u32) -> u64 {
    (((v << (64 - len - start)) as i64) >> (64 - len)) as u64
}

/// `MAKE_64BIT_MASK(shift, len)`.
fn mask(shift: u32, len: u32) -> u64 {
    if len == 0 { 0 } else { (u64::MAX >> (64 - len)) << shift }
}

fn deposit(v: u64, start: u32, len: u32, field: u64) -> u64 {
    let m = mask(start, len);
    (v & !m) | ((field << start) & m)
}

fn pac_cell_shuffle(i: u64) -> u64 {
    const SRC: [u32; 16] = [52, 24, 44, 0, 28, 48, 4, 40, 32, 12, 56, 20, 8, 36, 16, 60];
    SRC.iter().enumerate().fold(0, |o, (n, &b)| o | (extract(i, b, 4) << (4 * n)))
}

fn pac_cell_inv_shuffle(i: u64) -> u64 {
    const SRC: [u32; 16] = [12, 24, 48, 36, 56, 44, 4, 16, 32, 52, 28, 8, 20, 0, 40, 60];
    SRC.iter().enumerate().fold(0, |o, (n, &b)| o | (extract(i, b, 4) << (4 * n)))
}

fn sub_cells(i: u64, table: &[u8; 16]) -> u64 {
    (0..64).step_by(4).fold(0, |o, b| o | (u64::from(table[((i >> b) & 0xf) as usize]) << b))
}

const SUB: [u8; 16] =
    [0xb, 0x6, 0x8, 0xf, 0xc, 0x0, 0x9, 0xe, 0x3, 0x7, 0x4, 0x5, 0xd, 0x2, 0x1, 0xa];
const SUB1: [u8; 16] =
    [0xa, 0xd, 0xe, 0x6, 0xf, 0x7, 0x3, 0x5, 0x9, 0x8, 0x0, 0xc, 0xb, 0x1, 0x2, 0x4];
const INV_SUB: [u8; 16] =
    [0x5, 0xe, 0xd, 0x8, 0xa, 0xb, 0x1, 0x9, 0x2, 0x6, 0xf, 0x0, 0x4, 0xc, 0x7, 0x3];

/// A 4-bit rotate left by `n`.
fn rot_cell(cell: u64, n: u32) -> u64 {
    let cell = cell | (cell << 4);
    extract(cell, 4 - n, 4)
}

fn pac_mult(i: u64) -> u64 {
    let mut o = 0;
    for b in (0..16).step_by(4) {
        let i0 = extract(i, b, 4);
        let i4 = extract(i, b + 16, 4);
        let i8 = extract(i, b + 32, 4);
        let ic = extract(i, b + 48, 4);

        let t0 = rot_cell(i8, 1) ^ rot_cell(i4, 2) ^ rot_cell(i0, 1);
        let t1 = rot_cell(ic, 1) ^ rot_cell(i4, 1) ^ rot_cell(i0, 2);
        let t2 = rot_cell(ic, 2) ^ rot_cell(i8, 1) ^ rot_cell(i0, 1);
        let t3 = rot_cell(ic, 1) ^ rot_cell(i8, 2) ^ rot_cell(i4, 1);

        o |= t3 << b;
        o |= t2 << (b + 16);
        o |= t1 << (b + 32);
        o |= t0 << (b + 48);
    }
    o
}

fn tweak_cell_rot(cell: u64) -> u64 {
    (cell >> 1) | (((cell ^ (cell >> 1)) & 1) << 3)
}

fn tweak_cell_inv_rot(cell: u64) -> u64 {
    ((cell << 1) & 0xf) | ((cell & 1) ^ (cell >> 3))
}

/// The source cell of each output cell of `tweak_shuffle()`, and whether it is rotated.
const TWEAK: [(u32, bool); 16] = [
    (16, false),
    (20, false),
    (24, true),
    (28, false),
    (44, true),
    (8, false),
    (12, false),
    (32, true),
    (48, false),
    (52, false),
    (56, false),
    (60, true),
    (0, true),
    (4, false),
    (40, true),
    (36, true),
];

/// The same for `tweak_inv_shuffle()`.
const TWEAK_INV: [(u32, bool); 16] = [
    (48, true),
    (52, false),
    (20, false),
    (24, false),
    (0, false),
    (4, false),
    (8, true),
    (12, false),
    (28, true),
    (60, true),
    (56, true),
    (16, true),
    (32, false),
    (36, false),
    (40, false),
    (44, true),
];

fn tweak_shuffle(i: u64) -> u64 {
    TWEAK.iter().enumerate().fold(0, |o, (n, &(b, rot))| {
        let c = extract(i, b, 4);
        o | ((if rot { tweak_cell_rot(c) } else { c }) << (4 * n))
    })
}

fn tweak_inv_shuffle(i: u64) -> u64 {
    TWEAK_INV.iter().enumerate().fold(0, |o, (n, &(b, rot))| {
        let c = extract(i, b, 4);
        o | ((if rot { tweak_cell_inv_rot(c) } else { c }) << (4 * n))
    })
}

/// A 128-bit key, `ARMPACKey`.
#[derive(Clone, Copy)]
struct Key {
    lo: u64,
    hi: u64,
}

/// `pauth_computepac_architected()`: QARMA5, or QARMA3 when `qarma3`.
fn computepac_architected(data: u64, modifier: u64, key: Key, qarma3: bool) -> u64 {
    const RC: [u64; 5] = [
        0x0000000000000000,
        0x13198A2E03707344,
        0xA4093822299F31D0,
        0x082EFA98EC4E6C89,
        0x452821E638D01377,
    ];
    const ALPHA: u64 = 0xC0AC29B7C97C50DD;
    let iterations = if qarma3 { 2 } else { 4 };
    let fwd = |v| if qarma3 { sub_cells(v, &SUB1) } else { sub_cells(v, &SUB) };
    let inv = |v| if qarma3 { sub_cells(v, &SUB1) } else { sub_cells(v, &INV_SUB) };
    // Note that in the ARM pseudocode, key0 contains bits <127:64> and key1 contains bits
    // <63:0> of the 128-bit key.
    let (key0, key1) = (key.hi, key.lo);

    let modk0 = (key0 << 63) | ((key0 >> 1) ^ (key0 >> 63));
    let mut runningmod = modifier;
    let mut workingval = data ^ key0;

    for (i, rc) in RC.iter().enumerate().take(iterations + 1) {
        let roundkey = key1 ^ runningmod;
        workingval ^= roundkey;
        workingval ^= rc;
        if i > 0 {
            workingval = pac_cell_shuffle(workingval);
            workingval = pac_mult(workingval);
        }
        workingval = fwd(workingval);
        runningmod = tweak_shuffle(runningmod);
    }
    let roundkey = modk0 ^ runningmod;
    workingval ^= roundkey;
    workingval = pac_cell_shuffle(workingval);
    workingval = pac_mult(workingval);
    workingval = fwd(workingval);
    workingval = pac_cell_shuffle(workingval);
    workingval = pac_mult(workingval);
    workingval ^= key1;
    workingval = pac_cell_inv_shuffle(workingval);
    workingval = inv(workingval);
    workingval = pac_mult(workingval);
    workingval = pac_cell_inv_shuffle(workingval);
    workingval ^= key0;
    workingval ^= runningmod;
    for i in 0..=iterations {
        workingval = inv(workingval);
        if i < iterations {
            workingval = pac_mult(workingval);
            workingval = pac_cell_inv_shuffle(workingval);
        }
        runningmod = tweak_inv_shuffle(runningmod);
        let roundkey = key1 ^ runningmod;
        workingval ^= RC[iterations - i];
        workingval ^= roundkey;
        workingval ^= ALPHA;
    }
    workingval ^ modk0
}

const PRIME64_1: u64 = 0x9E3779B185EBCA87;
const PRIME64_2: u64 = 0xC2B2AE3D27D4EB4F;
const PRIME64_3: u64 = 0x165667B19E3779F9;
const PRIME64_4: u64 = 0x85EBCA77C2B2AE63;

fn xxh64_round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(PRIME64_2)).rotate_left(31).wrapping_mul(PRIME64_1)
}

fn xxh64_mergeround(acc: u64, val: u64) -> u64 {
    (acc ^ xxh64_round(0, val)).wrapping_mul(PRIME64_1).wrapping_add(PRIME64_4)
}

/// `qemu_xxhash64_4()`, with QEMU's seed of 1.
fn xxhash64_4(a: u64, b: u64, c: u64, d: u64) -> u64 {
    const SEED: u64 = 1;
    let v1 = xxh64_round(SEED.wrapping_add(PRIME64_1).wrapping_add(PRIME64_2), a);
    let v2 = xxh64_round(SEED.wrapping_add(PRIME64_2), b);
    let v3 = xxh64_round(SEED, c);
    let v4 = xxh64_round(SEED.wrapping_sub(PRIME64_1), d);

    let mut h = v1
        .rotate_left(1)
        .wrapping_add(v2.rotate_left(7))
        .wrapping_add(v3.rotate_left(12))
        .wrapping_add(v4.rotate_left(18));
    h = xxh64_mergeround(h, v1);
    h = xxh64_mergeround(h, v2);
    h = xxh64_mergeround(h, v3);
    h = xxh64_mergeround(h, v4);

    h ^= h >> 33;
    h = h.wrapping_mul(PRIME64_2);
    h ^= h >> 29;
    h = h.wrapping_mul(PRIME64_3);
    h ^ (h >> 32)
}

/// `pauth_computepac()`.
fn computepac(f: &ArmFeatures, data: u64, modifier: u64, key: Key) -> u64 {
    match f.pauth_alg {
        PauthAlg::Qarma5 => computepac_architected(data, modifier, key, false),
        PauthAlg::Qarma3 => computepac_architected(data, modifier, key, true),
        PauthAlg::Impdef => xxhash64_4(data, modifier, key.lo, key.hi),
    }
}

/// The parts of `ARMVAParameters` PAuth uses. FEAT_MTE_NO_ADDRESS_TAGS (`mtx`) is not
/// implemented, so it is always clear.
struct Param {
    tsz: u32,
    tbi: bool,
}

/// `aa64_va_parameters()` for `arm_stage1_mmu_idx()`.
fn va_params(f: &ArmFeatures, st: &CpuArmState, ptr: u64, data: bool) -> Param {
    let (tsz, tbi) = ptw::pauth_va_params(f, st, ptr, st.mmu_idx(f), data);
    Param { tsz, tbi }
}

/// `pauth_addpac()`.
fn addpac(f: &ArmFeatures, st: &CpuArmState, ptr: u64, modifier: u64, key: Key, data: bool) -> u64 {
    let param = va_params(f, st, ptr, data);

    // If tagged pointers are in use, use ptr<55>, otherwise ptr<63>.
    let mut ext = if param.tbi { sextract(ptr, 55, 1) } else { sextract(ptr, 63, 1) };

    // Build a pointer with known good extension bits.
    let top_bit = 64 - 8 * u32::from(param.tbi);
    let bot_bit = 64 - param.tsz;
    let ext_ptr = deposit(ptr, bot_bit, top_bit - bot_bit, ext);

    let mut pac = computepac(f, ext_ptr, modifier, key);

    // Check if the ptr has good extension bits and corrupt the pointer authentication code
    // if not.
    let test = sextract(ptr, bot_bit, top_bit - bot_bit);
    if test != 0 && test != u64::MAX {
        if f.pauth >= PAUTH_2 {
            // No action required.
        } else if f.pauth == PAUTH_EPAC {
            pac = 0;
        } else {
            // Note that our top_bit is one greater than the pseudocode's version, hence
            // "- 2" here.
            pac ^= mask(top_bit - 2, 1);
        }
    }

    // Preserve the determination between upper and lower at bit 55, and insert pointer
    // authentication code.
    if f.pauth >= PAUTH_2 {
        pac ^= ptr;
    }
    let ptr = if param.tbi {
        pac &= mask(bot_bit, 54 - bot_bit + 1);
        ptr & !mask(bot_bit, 55 - bot_bit + 1)
    } else {
        pac &= !(mask(55, 1) | mask(0, bot_bit));
        ptr & mask(0, bot_bit)
    };
    ext &= mask(55, 1);
    pac | ext | ptr
}

/// `pauth_ptr_mask()`.
fn ptr_mask(param: &Param) -> u64 {
    let bot = 64 - param.tsz;
    let top = 64 - 8 * u32::from(param.tbi);
    mask(bot, top - bot)
}

/// `pauth_original_ptr()`.
fn original_ptr(ptr: u64, param: &Param) -> u64 {
    let m = ptr_mask(param);
    // Note that bit 55 is used whether or not the regime has 2 ranges.
    if extract(ptr, 55, 1) != 0 { ptr | m } else { ptr & !m }
}

/// What an authentication helper needs from the vCPU.
struct Ctx {
    f: ArmFeatures,
    st: CpuArmState,
}

impl Ctx {
    fn new(cpu: &Cpu<'_>) -> Ctx {
        let ops = cpu.ops();
        Ctx { f: *arm_of(&ops).features(), st: CpuArmState::load_system(cpu.env) }
    }

    fn key(&self, n: usize) -> Key {
        Key { lo: self.st.pac_keys[2 * n], hi: self.st.pac_keys[2 * n + 1] }
    }

    /// `pauth_key_enabled()`.
    fn key_enabled(&self, bit: u64) -> bool {
        let el = self.st.current_el();
        let sctlr = if el == 0 {
            sysreg::sctlr_el0(&self.f, &self.st)
        } else {
            self.st.sctlr_el[el as usize]
        };
        sctlr & bit != 0
    }
}

/// `pauth_check_trap()`.
fn check_trap(cpu: &mut Cpu<'_>, c: &Ctx) -> Result<(), CpuLoopExit> {
    let (f, st) = (&c.f, &c.st);
    let el = st.current_el();
    if el < 2 && st.is_el2_enabled(f) {
        let hcr = st.hcr_el2_eff(f);
        let mut trap = hcr & HCR_API == 0;
        if el == 0 {
            // Trap only applies to EL1&0 regime.
            trap &= hcr & (HCR_E2H | HCR_TGE) != HCR_E2H | HCR_TGE;
        }
        if trap {
            return Err(raise_exception(cpu, EXCP_UDEF, syn_pactrap(), 2, Ra::Tb));
        }
    }
    if el < 3 && f.el3 && st.scr_el3 & SCR_API == 0 {
        return Err(raise_exception(cpu, EXCP_UDEF, syn_pactrap(), 3, Ra::Tb));
    }
    Ok(())
}

/// `pauth_auth()`.
#[allow(clippy::too_many_arguments)]
fn auth(
    cpu: &mut Cpu<'_>,
    c: &Ctx,
    ptr: u64,
    modifier: u64,
    keynumber: u32,
    data: bool,
    is_combined: bool,
) -> Result<u64, CpuLoopExit> {
    let (f, st) = (&c.f, &c.st);
    let param = va_params(f, st, ptr, data);
    let key = c.key(usize::from(data) * 2 + keynumber as usize);
    let orig_ptr = original_ptr(ptr, &param);
    let pac = computepac(f, orig_ptr, modifier, key);
    let bot_bit = 64 - param.tsz;
    let top_bit = 64 - 8 * u32::from(param.tbi);

    let cmp_mask = mask(bot_bit, top_bit - bot_bit) & !mask(55, 1);

    if f.pauth >= PAUTH_2 {
        let fault_feature = if is_combined { PAUTH_FPACCOMBINED } else { PAUTH_FPAC };
        let result = ptr ^ (pac & cmp_mask);

        if f.pauth >= fault_feature && (result ^ sextract(result, 55, 1)) & cmp_mask != 0 {
            let target_el = exception_target_el(st);
            return Err(raise_exception(
                cpu,
                EXCP_UDEF,
                syn_pacfail(data, keynumber),
                target_el,
                Ra::Tb,
            ));
        }
        return Ok(result);
    }

    if (pac ^ ptr) & cmp_mask != 0 {
        let error_code = u64::from((keynumber << 1) | (keynumber ^ 1));
        return Ok(if param.tbi {
            deposit(orig_ptr, 53, 2, error_code)
        } else {
            deposit(orig_ptr, 61, 2, error_code)
        });
    }
    Ok(orig_ptr)
}

/// The SCTLR enable bit of each key: IA, IB, DA and DB.
const ENABLE: [u64; 4] = [SCTLR_ENIA, SCTLR_ENIB, SCTLR_ENDA, SCTLR_ENDB];

/// `HELPER(pacia)` and friends: key `n` is IA, IB, DA or DB.
fn pac(h: &mut HelperEnv<'_>, a: &[u64], n: usize) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let c = Ctx::new(cpu);
        if !c.key_enabled(ENABLE[n]) {
            return Ok(a[1]);
        }
        check_trap(cpu, &c)?;
        Ok(addpac(&c.f, &c.st, a[1], a[2], c.key(n), n >= 2))
    })
}

/// `pauth_autia()` and friends.
fn aut(h: &mut HelperEnv<'_>, a: &[u64], n: usize, is_combined: bool) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let c = Ctx::new(cpu);
        if !c.key_enabled(ENABLE[n]) {
            return Ok(a[1]);
        }
        check_trap(cpu, &c)?;
        auth(cpu, &c, a[1], a[2], (n & 1) as u32, n >= 2, is_combined)
    })
}

fn h_pacia(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    pac(h, a, 0)
}

fn h_pacib(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    pac(h, a, 1)
}

fn h_pacda(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    pac(h, a, 2)
}

fn h_pacdb(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    pac(h, a, 3)
}

fn h_autia(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    aut(h, a, 0, false)
}

fn h_autib(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    aut(h, a, 1, false)
}

fn h_autda(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    aut(h, a, 2, false)
}

fn h_autdb(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    aut(h, a, 3, false)
}

fn h_autia_combined(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    aut(h, a, 0, true)
}

fn h_autib_combined(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    aut(h, a, 1, true)
}

fn h_autda_combined(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    aut(h, a, 2, true)
}

fn h_autdb_combined(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    aut(h, a, 3, true)
}

/// `HELPER(pacga)`.
fn h_pacga(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let c = Ctx::new(cpu);
        check_trap(cpu, &c)?;
        Ok(computepac(&c.f, a[1], a[2], c.key(4)) & 0xffff_ffff_0000_0000)
    })
}

/// `HELPER(xpaci)` and `HELPER(xpacd)`: `pauth_strip()`.
fn strip(h: &mut HelperEnv<'_>, a: &[u64], data: bool) -> Result<u128, Unwind> {
    run(h, |cpu| {
        let c = Ctx::new(cpu);
        Ok(original_ptr(a[1], &va_params(&c.f, &c.st, a[1], data)))
    })
}

fn h_xpaci(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    strip(h, a, false)
}

fn h_xpacd(h: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    strip(h, a, true)
}

def!(PACIA, "pacia", 0, I64, [Ptr, I64, I64], h_pacia);
def!(PACIB, "pacib", 0, I64, [Ptr, I64, I64], h_pacib);
def!(PACDA, "pacda", 0, I64, [Ptr, I64, I64], h_pacda);
def!(PACDB, "pacdb", 0, I64, [Ptr, I64, I64], h_pacdb);
def!(PACGA, "pacga", 0, I64, [Ptr, I64, I64], h_pacga);
def!(AUTIA, "autia", 0, I64, [Ptr, I64, I64], h_autia);
def!(AUTIB, "autib", 0, I64, [Ptr, I64, I64], h_autib);
def!(AUTDA, "autda", 0, I64, [Ptr, I64, I64], h_autda);
def!(AUTDB, "autdb", 0, I64, [Ptr, I64, I64], h_autdb);
def!(AUTIA_COMBINED, "autia_combined", 0, I64, [Ptr, I64, I64], h_autia_combined);
def!(AUTIB_COMBINED, "autib_combined", 0, I64, [Ptr, I64, I64], h_autib_combined);
def!(AUTDA_COMBINED, "autda_combined", 0, I64, [Ptr, I64, I64], h_autda_combined);
def!(AUTDB_COMBINED, "autdb_combined", 0, I64, [Ptr, I64, I64], h_autdb_combined);
def!(XPACI, "xpaci", 0, I64, [Ptr, I64], h_xpaci);
def!(XPACD, "xpacd", 0, I64, [Ptr, I64], h_xpacd);

/// Every PAuth helper.
pub(crate) const ALL: &[Def] = &[
    PACIA,
    PACIB,
    PACDA,
    PACDB,
    PACGA,
    AUTIA,
    AUTIB,
    AUTDA,
    AUTDB,
    AUTIA_COMBINED,
    AUTIB_COMBINED,
    AUTDA_COMBINED,
    AUTDB_COMBINED,
    XPACI,
    XPACD,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The QARMA5 test vector of tests/tcg/aarch64/system/pauth-3.c, from the QARMA paper.
    #[test]
    fn qarma5_vector() {
        let key = Key { lo: 0xec2802d4e0a488e9, hi: 0x84be85ce9804e94b };
        let pac = computepac_architected(0xfb623599da6e8127, 0x477d469dec0b8762, key, false);
        assert_eq!(pac, 0xc003b93999b33765);
    }

    #[test]
    fn inverse_shuffles() {
        let v = 0x0123_4567_89ab_cdef;
        assert_eq!(pac_cell_inv_shuffle(pac_cell_shuffle(v)), v);
        assert_eq!(tweak_inv_shuffle(tweak_shuffle(v)), v);
        assert_eq!(sub_cells(sub_cells(v, &SUB), &INV_SUB), v);
    }
}
