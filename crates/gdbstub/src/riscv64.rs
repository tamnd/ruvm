// SPDX-License-Identifier: GPL-2.0-or-later

//! The riscv64 registers: `target/riscv/gdbstub.c` in user mode, with
//! `riscv-64bit-cpu.xml`, the FPU registers with F or D, the vector registers with Zve32x,
//! `riscv-64bit-virtual.xml` and the CSRs a user program can reach.

use std::sync::OnceLock;

use ruvm_linux_user::gdb::{Cx, Regs};
use ruvm_target_riscv::cfg::RiscvCfg;
use ruvm_target_riscv::cpu::{PC, RVD, RVF, VREG, fpr_off, gpr_off};
use ruvm_target_riscv::tcg::{cpu_cfg, csrrw_debug};

use crate::user::{FeatureBuilder, Target};

/// riscv64 as GDB sees it.
pub(crate) static TARGET: Target = Target {
    name: "riscv64",
    arch: Some("riscv:rv64"),
    features,
    g_regs: 33,
    abi_digits: 16,
    read,
    write,
    set_pc,
};

/// The register numbers each feature starts at, as `gdb_register_coprocessor()` hands
/// them out.
struct Layout {
    misa_ext: u64,
    vlenb: usize,
    fpu: Option<usize>,
    vector: Option<usize>,
    virt: usize,
    csr: usize,
    /// The CSRs of `riscv-csr.xml`, in the order of their numbers.
    csrs: Vec<u32>,
}

static LAYOUT: OnceLock<Layout> = OnceLock::new();

const CORE_REGS: usize = 33;

/// A `csr_ops[]` predicate: whether the configuration has the CSR.
type Predicate = fn(&RiscvCfg) -> bool;

/// The `csr_ops[]` entries whose predicate passes in user mode: number, name, and whether
/// the configuration has it.
const CSRS: [(u32, &str, Predicate); 15] = [
    (0x001, "fflags", fs),
    (0x002, "frm", fs),
    (0x003, "fcsr", fs),
    (0x008, "vstart", vs),
    (0x009, "vxsat", vs),
    (0x00a, "vxrm", vs),
    (0x00f, "vcsr", vs),
    (0x015, "seed", |c| c.ext_zkr),
    (0x017, "jvt", |c| c.ext_zcmt),
    (0xc00, "cycle", ctr),
    (0xc01, "time", ctr),
    (0xc02, "instret", ctr),
    (0xc20, "vl", vs),
    (0xc21, "vtype", vs),
    (0xc22, "vlenb", vs),
];

/// The `fs` predicate, which passes in user mode.
fn fs(_: &RiscvCfg) -> bool {
    true
}

/// The `vs` predicate.
fn vs(c: &RiscvCfg) -> bool {
    c.ext_zve32x
}

/// The `ctr` predicate, which lets user mode read the counters.
fn ctr(_: &RiscvCfg) -> bool {
    true
}

/// `ricsv_gen_dynamic_vector_feature()`.
fn vector_feature(vlenb: usize, base: usize) -> (String, usize) {
    const LANES: [(&str, &str, usize, char); 5] = [
        ("uint128", "quads", 128, 'q'),
        ("uint64", "longs", 64, 'l'),
        ("uint32", "words", 32, 'w'),
        ("uint16", "shorts", 16, 's'),
        ("uint8", "bytes", 8, 'b'),
    ];
    let bitsize = vlenb << 3;
    let mut b = FeatureBuilder::new("org.gnu.gdb.riscv.vector", base);
    for (ty, id, size, _) in LANES {
        b.tag(&format!("<vector id=\"{id}\" type=\"{ty}\" count=\"{}\"/>", bitsize / size));
    }
    b.tag("<union id=\"riscv_vector\">");
    for (_, id, _, suffix) in LANES {
        b.tag(&format!("<field name=\"{suffix}\" type=\"{id}\"/>"));
    }
    b.tag("</union>");
    for i in 0..32 {
        b.reg(&format!("v{i}"), bitsize, i, "riscv_vector", Some("vector"));
    }
    b.end()
}

/// `riscv_gen_dynamic_csr_feature()`.
fn csr_feature(csrs: &[u32], base: usize) -> (String, usize) {
    let mut b = FeatureBuilder::new("org.gnu.gdb.riscv.csr", base);
    for &(n, name, _) in CSRS.iter().filter(|c| csrs.contains(&c.0)) {
        b.reg(name, 64, n as usize, "int", None);
    }
    b.end()
}

/// `riscv_cpu_register_gdb_regs_for_features()`.
fn features(cx: &mut Cx<'_, '_>) -> Vec<(&'static str, String)> {
    let tid = cx.tid();
    let Some(Regs::Live(c)) = cx.regs(tid) else { return Vec::new() };
    let cfg = cpu_cfg(c);
    let mut v =
        vec![("riscv-64bit-cpu.xml", include_str!("../gdb-xml/riscv-64bit-cpu.xml").to_string())];
    let mut base = CORE_REGS;
    let mut fpu = None;
    if cfg.misa_ext & (RVD | RVF) != 0 {
        let xml = if cfg.misa_ext & RVD != 0 {
            ("riscv-64bit-fpu.xml", include_str!("../gdb-xml/riscv-64bit-fpu.xml"))
        } else {
            ("riscv-32bit-fpu.xml", include_str!("../gdb-xml/riscv-32bit-fpu.xml"))
        };
        v.push((xml.0, xml.1.to_string()));
        fpu = Some(base);
        base += 32;
    }
    let vlenb = cfg.vlenb as usize;
    let mut vector = None;
    if cfg.ext_zve32x {
        let (xml, n) = vector_feature(vlenb, base);
        v.push(("riscv-vector.xml", xml));
        vector = Some(base);
        base += n;
    }
    v.push((
        "riscv-64bit-virtual.xml",
        include_str!("../gdb-xml/riscv-64bit-virtual.xml").to_string(),
    ));
    let virt = base;
    base += 1;
    let mut csrs = Vec::new();
    if cfg.ext_zicsr {
        csrs = CSRS.iter().filter(|c| (c.2)(&cfg)).map(|c| c.0).collect();
        let (xml, _) = csr_feature(&csrs, base);
        v.push(("riscv-csr.xml", xml));
    }
    let _ =
        LAYOUT.set(Layout { misa_ext: cfg.misa_ext, vlenb, fpu, vector, virt, csr: base, csrs });
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

fn ld64(env: &[u8], off: usize) -> u64 {
    get(&env[off..], 8)
}

fn st64(env: &mut [u8], off: usize, v: u64) {
    env[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

/// The register `n` counts from the start of its feature, and which feature that is.
enum Reg {
    Core(usize),
    Fpu(usize),
    Vector(usize),
    Virt,
    Csr(u32),
}

fn which(l: &Layout, n: usize) -> Option<Reg> {
    let within = |base: Option<usize>, len: usize| base.filter(|&b| n >= b && n < b + len);
    if n < CORE_REGS {
        Some(Reg::Core(n))
    } else if let Some(b) = within(l.fpu, 32) {
        Some(Reg::Fpu(n - b))
    } else if let Some(b) = within(l.vector, 32) {
        Some(Reg::Vector(n - b))
    } else if n == l.virt {
        Some(Reg::Virt)
    } else {
        let csrno = u32::try_from(n.checked_sub(l.csr)?).ok()?;
        l.csrs.contains(&csrno).then_some(Reg::Csr(csrno))
    }
}

/// `riscv_csrrw_debug()` on thread `tid`. Another thread's CSRs are read from its saved
/// registers, with the counters of the calling thread's vCPU.
fn csr(cx: &mut Cx<'_, '_>, tid: u32, csrno: u32, write: Option<u64>) -> Option<u64> {
    let me = cx.tid();
    let saved = match cx.regs(tid)? {
        Regs::Live(c) => return csrrw_debug(c, csrno, write),
        Regs::Saved(v) => v,
    };
    let mut env = saved;
    let Some(Regs::Live(c)) = cx.regs(me) else { return None };
    let mut other = ruvm_jit::Cpu { env: &mut env, core: &mut *c.core };
    csrrw_debug(&mut other, csrno, write)
}

/// `riscv_cpu_gdb_read_register()` and the feature `get_reg` functions.
fn read(cx: &mut Cx<'_, '_>, tid: u32, n: usize, out: &mut Vec<u8>) -> usize {
    let Some(l) = LAYOUT.get() else { return 0 };
    let Some(reg) = which(l, n) else { return 0 };
    if let Reg::Csr(csrno) = reg {
        return csr(cx, tid, csrno, None).map_or(0, |v| put(out, v, 8));
    }
    let Some(r) = cx.regs(tid) else { return 0 };
    let env: &[u8] = match &r {
        Regs::Live(c) => c.env,
        Regs::Saved(v) => v,
    };
    match reg {
        Reg::Core(32) => put(out, ld64(env, PC), 8),
        Reg::Core(n) => put(out, ld64(env, gpr_off(n)), 8),
        Reg::Fpu(f) if l.misa_ext & RVD != 0 => put(out, ld64(env, fpr_off(f)), 8),
        Reg::Fpu(f) => put(out, ld64(env, fpr_off(f)), 4),
        Reg::Vector(v) => {
            (0..l.vlenb).step_by(8).map(|i| put(out, ld64(env, VREG + v * l.vlenb + i), 8)).sum()
        }
        // The privilege level, which in user mode is always 0.
        Reg::Virt => put(out, 0, 8),
        Reg::Csr(_) => 0,
    }
}

/// `riscv_cpu_gdb_write_register()` and the feature `set_reg` functions. `mem` holds
/// enough bytes for any register.
fn write(cx: &mut Cx<'_, '_>, tid: u32, n: usize, mem: &[u8]) -> usize {
    let Some(l) = LAYOUT.get() else { return 0 };
    let Some(reg) = which(l, n) else { return 0 };
    if let Reg::Csr(csrno) = reg {
        return csr(cx, tid, csrno, Some(get(mem, 8))).map_or(0, |_| 8);
    }
    let Some(mut r) = cx.regs(tid) else { return 0 };
    let env: &mut [u8] = match &mut r {
        Regs::Live(c) => c.env,
        Regs::Saved(v) => v,
    };
    match reg {
        // x0 ignores writes.
        Reg::Core(0) => 8,
        Reg::Core(32) => {
            st64(env, PC, get(mem, 8));
            8
        }
        Reg::Core(n) => {
            st64(env, gpr_off(n), get(mem, 8));
            8
        }
        // Always 64 bits.
        Reg::Fpu(f) => {
            st64(env, fpr_off(f), get(mem, 8));
            8
        }
        Reg::Vector(v) => {
            for i in (0..l.vlenb).step_by(8) {
                st64(env, VREG + v * l.vlenb + i, get(&mem[i..], 8));
            }
            l.vlenb
        }
        Reg::Virt => 8,
        Reg::Csr(_) => 0,
    }
}

/// `riscv_cpu_set_pc()`.
fn set_pc(env: &mut [u8], pc: u64) {
    st64(env, PC, pc);
}
