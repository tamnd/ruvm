// SPDX-License-Identifier: GPL-2.0-or-later

//! The x86_64 registers: `target/i386/gdbstub.c` for a 64-bit target in user mode, with
//! `i386-64bit.xml` and `i386-64bit-linux.xml`.
//!
//! As in QEMU, EFLAGS reads and writes whole, the x87 registers read from the top of the
//! stack but write to the physical register, and the control registers read as they are and
//! ignore writes.

use ruvm_linux_user::gdb::{Cx, Regs};
use ruvm_target_x86::state::{HF_CS64_MASK, R_CS, R_DS, R_ES, R_FS, R_GS, R_SS};
use ruvm_target_x86::tcg::cc::CC_OP_EFLAGS;
use ruvm_target_x86::tcg::env::{
    CC_MASK, CC_OP, CC_SRC, DF, DF_MASK, EFER, EFLAGS, EIP, FPSTT, FPUC, FPUS, HFLAGS,
    KERNELGSBASE, MXCSR, SEG_BASE, SEG_SELECTOR, compute_eflags, cr, fpreg, ld32, ld64, reg, seg,
    st32, st64, zmm,
};
use ruvm_target_x86::tcg::user::cpu_x86_load_seg;

use crate::user::Target;

/// `gpr_map`: GDB's order of the general registers.
const GPR_MAP: [usize; 16] = [0, 3, 1, 2, 6, 7, 5, 4, 8, 9, 10, 11, 12, 13, 14, 15];
/// The segment registers in GDB's order.
const SEG_MAP: [usize; 6] = [R_CS, R_SS, R_DS, R_ES, R_FS, R_GS];

const IDX_IP: usize = 16;
const IDX_FLAGS: usize = 17;
const IDX_SEG: usize = 18;
const IDX_CTL: usize = 27;
const IDX_FP: usize = 33;
const IDX_XMM: usize = 49;
const IDX_MXCSR: usize = 65;
/// `orig_rax` of `i386-64bit-linux.xml`, after the 66 core registers.
const IDX_ORIG_AX: usize = 66;

/// x86_64 as GDB sees it.
pub(crate) static TARGET: Target = Target {
    name: "x86_64",
    arch: Some("i386:x86-64"),
    features,
    g_regs: 66,
    abi_digits: 16,
    read,
    write,
    set_pc,
};

fn features(_: &mut Cx<'_, '_>) -> Vec<(&'static str, String)> {
    vec![
        ("i386-64bit.xml", include_str!("../gdb-xml/i386-64bit.xml").to_string()),
        ("i386-64bit-linux.xml", include_str!("../gdb-xml/i386-64bit-linux.xml").to_string()),
    ]
}

fn cs64(env: &[u8]) -> bool {
    ld32(env, HFLAGS) & HF_CS64_MASK != 0
}

/// `gdb_get_reg()`: a target long, cut to 32 bits outside 64-bit code.
fn long(env: &[u8], v: u64) -> u64 {
    if cs64(env) { v } else { v & 0xffff_ffff }
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

/// `x86_cpu_gdb_read_register()` and the Linux register.
fn read(cx: &mut Cx<'_, '_>, tid: u32, n: usize, out: &mut Vec<u8>) -> usize {
    let orig_ax = cx.orig_ax(tid);
    let Some(r) = cx.regs(tid) else { return 0 };
    let env: &[u8] = match &r {
        Regs::Live(c) => c.env,
        Regs::Saved(v) => v,
    };
    if n == IDX_ORIG_AX {
        return put(out, long(env, orig_ax), 8);
    }
    read_env(env, n, out)
}

fn read_env(env: &[u8], n: usize, out: &mut Vec<u8>) -> usize {
    match n {
        0..16 => put(out, long(env, ld64(env, reg(GPR_MAP[n]))), 8),
        IDX_IP => put(out, long(env, ld64(env, EIP)), 8),
        IDX_FLAGS => put(out, u64::from(compute_eflags(env)), 4),
        18..24 => put(out, ld32(env, seg(SEG_MAP[n - IDX_SEG]) + SEG_SELECTOR).into(), 4),
        24 => put(out, ld64(env, seg(R_FS) + SEG_BASE), 8),
        25 => put(out, ld64(env, seg(R_GS) + SEG_BASE), 8),
        26 => put(out, ld64(env, KERNELGSBASE), 8),
        27 => put(out, ld64(env, cr(0)), 8),
        28 => put(out, ld64(env, cr(2)), 8),
        29 => put(out, ld64(env, cr(3)), 8),
        30 => put(out, ld64(env, cr(4)), 8),
        // The TPR, which user mode does not have.
        31 => put(out, 0, 8),
        32 => put(out, ld64(env, EFER), 8),
        33..41 => {
            let r = (n - IDX_FP + ld32(env, FPSTT) as usize) % 8;
            put(out, ld64(env, fpreg(r)), 8) + put(out, ld64(env, fpreg(r) + 8), 2)
        }
        41 => put(out, ld32(env, FPUC).into(), 4),
        42 => {
            let fpus = ld32(env, FPUS) & !0x3800 | (ld32(env, FPSTT) & 7) << 11;
            put(out, fpus.into(), 4)
        }
        // ftag, fiseg, fioff, foseg, fooff and fop read as 0.
        43..49 => put(out, 0, 4),
        49..65 => {
            let x = zmm(n - IDX_XMM);
            put(out, ld64(env, x), 8) + put(out, ld64(env, x + 8), 8)
        }
        IDX_MXCSR => put(out, ld32(env, MXCSR).into(), 4),
        _ => 0,
    }
}

/// `x86_cpu_gdb_write_register()` and the Linux register. `mem` holds at least 16 bytes.
fn write(cx: &mut Cx<'_, '_>, tid: u32, n: usize, mem: &[u8]) -> usize {
    if n == IDX_ORIG_AX {
        let Some(r) = cx.regs(tid) else { return 0 };
        let wide = match &r {
            Regs::Live(c) => cs64(c.env),
            Regs::Saved(v) => cs64(v),
        };
        drop(r);
        let v = get(mem, 8);
        cx.set_orig_ax(tid, if wide { v } else { v & 0xffff_ffff });
        return 8;
    }
    let Some(mut r) = cx.regs(tid) else { return 0 };
    if (IDX_SEG..IDX_SEG + 6).contains(&n) {
        // x86_cpu_gdb_load_seg(): a new selector loads the segment from the descriptor tables.
        if let Regs::Live(c) = &mut r {
            let s = SEG_MAP[n - IDX_SEG];
            let sel = get(mem, 4) as u16 as u32;
            if sel != ld32(c.env, seg(s) + SEG_SELECTOR) {
                let _ = cpu_x86_load_seg(c, s, sel);
            }
        }
        return 4;
    }
    let env: &mut [u8] = match &mut r {
        Regs::Live(c) => c.env,
        Regs::Saved(v) => v,
    };
    write_env(env, n, mem)
}

/// `gdb_write_reg_cs64()`: 64 bits in 64-bit code, else 32.
fn cs64_value(env: &[u8], mem: &[u8]) -> (u64, usize) {
    if cs64(env) { (get(mem, 8), 8) } else { (get(mem, 4), 4) }
}

fn write_env(env: &mut [u8], n: usize, mem: &[u8]) -> usize {
    match n {
        0..16 => {
            let v = get(mem, 8);
            if cs64(env) {
                st64(env, reg(GPR_MAP[n]), v);
            } else if n < 8 {
                st64(env, reg(GPR_MAP[n]), v & 0xffff_ffff);
            }
            8
        }
        IDX_IP => {
            let v = long(env, get(mem, 8));
            st64(env, EIP, v);
            8
        }
        IDX_FLAGS => {
            // env->eflags whole, as the next cpu_exec() splits it.
            let v = get(mem, 4) as u32;
            st64(env, CC_SRC, u64::from(v & CC_MASK));
            st32(env, CC_OP, CC_OP_EFLAGS);
            st32(env, DF, if v & DF_MASK != 0 { -1i32 as u32 } else { 1 });
            st64(env, EFLAGS, u64::from(v & !(CC_MASK | DF_MASK)));
            4
        }
        24..27 => {
            let (v, len) = cs64_value(env, mem);
            let off = match n {
                24 => seg(R_FS) + SEG_BASE,
                25 => seg(R_GS) + SEG_BASE,
                _ => KERNELGSBASE,
            };
            st64(env, off, v);
            len
        }
        IDX_CTL..33 => cs64_value(env, mem).1,
        33..41 => {
            let r = fpreg(n - IDX_FP);
            st64(env, r, get(mem, 8));
            env[r + 8..r + 10].copy_from_slice(&mem[8..10]);
            10
        }
        41 => {
            st32(env, FPUC, get(mem, 4) as u32 & 0xffff);
            4
        }
        42 => {
            let v = get(mem, 4) as u32;
            st32(env, FPSTT, (v >> 11) & 7);
            st32(env, FPUS, v & !0x3800 & 0xffff);
            4
        }
        43..49 => 4,
        49..65 => {
            let x = zmm(n - IDX_XMM);
            st64(env, x, get(mem, 8));
            st64(env, x + 8, get(&mem[8..], 8));
            16
        }
        IDX_MXCSR => {
            st32(env, MXCSR, get(mem, 4) as u32);
            4
        }
        _ => 0,
    }
}

/// `x86_cpu_set_pc()`.
fn set_pc(env: &mut [u8], pc: u64) {
    st64(env, EIP, pc);
}
