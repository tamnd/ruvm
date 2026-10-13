// SPDX-License-Identifier: GPL-2.0-or-later

//! The aarch64 registers: `target/arm/gdbstub64.c` in user mode, with `aarch64-core.xml`,
//! the SVE registers when the model has SVE and `aarch64-fpu.xml` when it does not,
//! `aarch64-pauth.xml` with pointer authentication, and the TLS registers.
//!
//! The model has no SME and no MTE2, so the SME, SME2 and `tag_ctl` features are absent, and
//! the system registers feature `org.qemu.gdb.arm.sys.regs`, which is for system mode work,
//! is left out.

use std::sync::OnceLock;

use ruvm_linux_user::gdb::{Cx, Regs};
use ruvm_target_arm::cpu::{ArmFeatures, CpuArmState};
use ruvm_target_arm::tcg::user::{commit, features as arm_features, pauth_ptr_mask, sve_vq_el0};
use ruvm_target_arm::tcg::{vfp_get_fpcr, vfp_get_fpsr, vfp_set_fpcr, vfp_set_fpsr};

use crate::user::{FeatureBuilder, Target};

/// aarch64 as GDB sees it.
pub(crate) static TARGET: Target = Target {
    name: "aarch64",
    arch: Some("aarch64"),
    features,
    g_regs: 34,
    abi_digits: 16,
    read,
    write,
    set_pc,
};

/// The register numbers each feature starts at, as `gdb_register_coprocessor()` hands
/// them out.
struct Layout {
    f: ArmFeatures,
    /// The SVE feature, or else `aarch64-fpu.xml`, starts at 34.
    sve: bool,
    pauth: Option<usize>,
    tls: usize,
}

static LAYOUT: OnceLock<Layout> = OnceLock::new();

const CORE_REGS: usize = 34;
const SVE_REGS: usize = 52;
const FPU_REGS: usize = 34;
const PAUTH_REGS: usize = 4;

/// `output_vector_union_type()`.
fn vector_union_type(b: &mut FeatureBuilder, reg_width: usize, name: &str) {
    const LANES: [(&str, usize, char, char); 13] = [
        ("uint128", 128, 'q', 'u'),
        ("int128", 128, 'q', 's'),
        ("ieee_double", 64, 'd', 'f'),
        ("uint64", 64, 'd', 'u'),
        ("int64", 64, 'd', 's'),
        ("ieee_single", 32, 's', 'f'),
        ("uint32", 32, 's', 'u'),
        ("int32", 32, 's', 's'),
        ("ieee_half", 16, 'h', 'f'),
        ("uint16", 16, 'h', 'u'),
        ("int16", 16, 'h', 's'),
        ("uint8", 8, 'b', 'u'),
        ("int8", 8, 'b', 's'),
    ];
    const SUF: [char; 5] = ['b', 'h', 's', 'd', 'q'];
    for (ty, size, sz, suffix) in LANES {
        let count = reg_width / size;
        b.tag(&format!("<vector id=\"{name}{sz}{suffix}\" type=\"{ty}\" count=\"{count}\"/>"));
    }
    for (i, s) in SUF.iter().enumerate() {
        b.tag(&format!("<union id=\"{name}n{s}\">"));
        for (_, size, sz, suffix) in LANES {
            if size == 8 << i {
                b.tag(&format!("<field name=\"{suffix}\" type=\"{name}{sz}{suffix}\"/>"));
            }
        }
        b.tag("</union>");
    }
    b.tag(&format!("<union id=\"{name}\">"));
    for s in SUF.iter().rev() {
        b.tag(&format!("<field name=\"{s}\" type=\"{name}n{s}\"/>"));
    }
    b.tag("</union>");
}

/// `arm_gen_dynamic_svereg_feature()`.
fn sve_feature(max_vq: usize, base: usize) -> (String, usize) {
    let reg_width = max_vq * 128;
    let pred_width = max_vq * 16;
    let mut b = FeatureBuilder::new("org.gnu.gdb.aarch64.sve", base);
    vector_union_type(&mut b, reg_width, "svev");
    b.tag(&format!("<vector id=\"svep\" type=\"uint8\" count=\"{}\"/>", pred_width / 8));
    for i in 0..32 {
        b.reg(&format!("z{i}"), reg_width, i, "svev", None);
    }
    b.reg("fpsr", 32, 32, "int", Some("float"));
    b.reg("fpcr", 32, 33, "int", Some("float"));
    for i in 0..16 {
        b.reg(&format!("p{i}"), pred_width, 34 + i, "svep", None);
    }
    b.reg("ffr", pred_width, 50, "svep", Some("vector"));
    b.reg("vg", 64, 51, "int", None);
    b.end()
}

/// `arm_gen_dynamic_tls_feature()`, without `tpidr2` as there is no SME.
fn tls_feature(base: usize) -> (String, usize) {
    let mut b = FeatureBuilder::new("org.gnu.gdb.aarch64.tls", base);
    b.reg("tpidr", 64, 0, "data_ptr", None);
    b.end()
}

/// `aarch64_cpu_register_gdb_regs_for_features()`.
fn features(cx: &mut Cx<'_, '_>) -> Vec<(&'static str, String)> {
    let tid = cx.tid();
    let f = match cx.regs(tid) {
        Some(Regs::Live(c)) => arm_features(c),
        _ => ArmFeatures::default(),
    };
    let mut v = vec![("aarch64-core.xml", include_str!("../gdb-xml/aarch64-core.xml").to_string())];
    let mut base = CORE_REGS;
    let sve = f.sve;
    if sve {
        let (xml, n) = sve_feature(f.sve_max_vq as usize, base);
        debug_assert_eq!(n, SVE_REGS);
        v.push(("sve-registers.xml", xml));
        base += SVE_REGS;
    } else {
        v.push(("aarch64-fpu.xml", include_str!("../gdb-xml/aarch64-fpu.xml").to_string()));
        base += FPU_REGS;
    }
    let pauth = (f.pauth != 0).then(|| {
        v.push(("aarch64-pauth.xml", include_str!("../gdb-xml/aarch64-pauth.xml").to_string()));
        base += PAUTH_REGS;
        base - PAUTH_REGS
    });
    let (xml, _) = tls_feature(base);
    v.push(("tls-registers.xml", xml));
    let _ = LAYOUT.set(Layout { f, sve, pauth, tls: base });
    v
}

fn put(out: &mut Vec<u8>, v: u64, size: usize) -> usize {
    out.extend_from_slice(&v.to_le_bytes()[..size]);
    size
}

fn get(mem: &[u8], size: usize) -> u64 {
    let mut b = [0u8; 8];
    b[..size].copy_from_slice(&mem[..size]);
    u64::from_le_bytes(b)
}

/// The register `n` counts from the start of its feature, and which feature that is.
enum Reg {
    Core(usize),
    Sve(usize),
    Fpu(usize),
    Pauth(usize),
    Tls,
}

fn which(l: &Layout, n: usize) -> Option<Reg> {
    if n < CORE_REGS {
        return Some(Reg::Core(n));
    }
    let r = n - CORE_REGS;
    if l.sve && r < SVE_REGS {
        return Some(Reg::Sve(r));
    }
    if !l.sve && r < FPU_REGS {
        return Some(Reg::Fpu(r));
    }
    if let Some(p) = l.pauth.filter(|&p| n >= p && n < p + PAUTH_REGS) {
        return Some(Reg::Pauth(n - p));
    }
    (n == l.tls).then_some(Reg::Tls)
}

/// The registers of thread `tid`.
fn state(cx: &mut Cx<'_, '_>, tid: u32) -> Option<CpuArmState> {
    match cx.regs(tid)? {
        Regs::Live(c) => Some(CpuArmState::load(c.env)),
        Regs::Saved(v) => Some(CpuArmState::load(&v)),
    }
}

/// `aarch64_cpu_gdb_read_register()` and the feature `get_reg` functions.
fn read(cx: &mut Cx<'_, '_>, tid: u32, n: usize, out: &mut Vec<u8>) -> usize {
    let Some(l) = LAYOUT.get() else { return 0 };
    let Some(reg) = which(l, n) else { return 0 };
    let Some(st) = state(cx, tid) else { return 0 };
    let max_vq = l.f.sve_max_vq as usize;
    match reg {
        Reg::Core(n @ 0..32) => put(out, st.xregs[n], 8),
        Reg::Core(32) => put(out, st.pc, 8),
        Reg::Core(_) => put(out, st.pstate_read().into(), 4),
        // aarch64_gdb_get_fpu_reg().
        Reg::Fpu(r @ 0..32) => put(out, st.zregs[r][0], 8) + put(out, st.zregs[r][1], 8),
        Reg::Fpu(32) | Reg::Sve(32) => put(out, vfp_get_fpsr(&st).into(), 4),
        Reg::Fpu(33) | Reg::Sve(33) => put(out, vfp_get_fpcr(&st).into(), 4),
        Reg::Fpu(_) => 0,
        // aarch64_gdb_get_sve_reg().
        Reg::Sve(r @ 0..32) => {
            let z = &st.zregs[r];
            (0..max_vq).map(|vq| put(out, z[vq * 2], 8) + put(out, z[vq * 2 + 1], 8)).sum()
        }
        Reg::Sve(r @ 34..51) => {
            let p = &st.pregs[r - 34];
            (0..max_vq).step_by(4).map(|vq| put(out, p[vq / 4], 8)).sum()
        }
        Reg::Sve(_) => put(out, u64::from(sve_vq_el0(&l.f, &st)) * 2, 8),
        // aarch64_gdb_get_pauth_reg().
        Reg::Pauth(r) => put(out, pauth_ptr_mask(&l.f, &st, r & 2 != 0, r & 1 == 0), 8),
        // aarch64_gdb_get_tls_reg().
        Reg::Tls => put(out, st.tpidr_el[0], 8),
    }
}

/// `aarch64_cpu_gdb_write_register()` and the feature `set_reg` functions. `mem` holds
/// enough bytes for any register.
fn write(cx: &mut Cx<'_, '_>, tid: u32, n: usize, mem: &[u8]) -> usize {
    let Some(l) = LAYOUT.get() else { return 0 };
    let Some(reg) = which(l, n) else { return 0 };
    let Some(mut r) = cx.regs(tid) else { return 0 };
    let mut st = match &r {
        Regs::Live(c) => CpuArmState::load(c.env),
        Regs::Saved(v) => CpuArmState::load(v),
    };
    let max_vq = l.f.sve_max_vq as usize;
    let len = match reg {
        Reg::Core(n @ 0..32) => {
            st.xregs[n] = get(mem, 8);
            8
        }
        Reg::Core(32) => {
            st.pc = get(mem, 8);
            8
        }
        Reg::Core(_) => {
            st.pstate_write(get(mem, 8) as u32);
            4
        }
        Reg::Fpu(r @ 0..32) => {
            st.zregs[r][0] = get(mem, 8);
            st.zregs[r][1] = get(&mem[8..], 8);
            16
        }
        Reg::Fpu(32) | Reg::Sve(32) => {
            vfp_set_fpsr(&mut st, get(mem, 4) as u32);
            4
        }
        Reg::Fpu(33) | Reg::Sve(33) => {
            vfp_set_fpcr(&mut st, get(mem, 4) as u32, &l.f);
            4
        }
        Reg::Fpu(_) => 0,
        Reg::Sve(r @ 0..32) => {
            for vq in 0..max_vq {
                st.zregs[r][vq * 2] = get(&mem[vq * 16..], 8);
                st.zregs[r][vq * 2 + 1] = get(&mem[vq * 16 + 8..], 8);
            }
            max_vq * 16
        }
        Reg::Sve(r @ 34..51) => {
            let mut len = 0;
            for vq in (0..max_vq).step_by(4) {
                st.pregs[r - 34][vq / 4] = get(&mem[len..], 8);
                len += 8;
            }
            len
        }
        // vg and the pauth masks cannot be set.
        Reg::Sve(_) | Reg::Pauth(_) => return 0,
        Reg::Tls => {
            st.tpidr_el[0] = get(mem, 8);
            8
        }
    };
    match &mut r {
        Regs::Live(c) => commit(c, &mut st),
        Regs::Saved(v) => st.store(v),
    }
    len
}

/// `aarch64_cpu_set_pc()`.
fn set_pc(env: &mut [u8], pc: u64) {
    let mut st = CpuArmState::load(env);
    st.pc = pc;
    st.store(env);
}
