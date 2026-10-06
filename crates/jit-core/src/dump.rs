// SPDX-License-Identifier: MIT OR Apache-2.0

//! The text dump of an op list, a port of `tcg_dump_ops` from `tcg/tcg.c`.
//!
//! The output is byte for byte what QEMU prints for `-d op` and `-d op_opt` on a host built with
//! plugin support, except that the `pref=` register preferences are never printed: they belong to
//! the register allocator of a real backend. A line that QEMU would end with `pref=...` still gets
//! the padding to column 40.

use std::fmt::Write;

use crate::ir::{DEAD_ARG, Func, Op, SYNC_ARG, Temp};
use crate::opcode::Opcode;
use crate::types::{Cond, MemOp, MemOpIdx, TempKind, Type, mo, opf};

const LDST_NAMES: [(u32, &str); 14] = [
    (0x00, "ub"),
    (0x08, "sb"),
    (0x01, "leuw"),
    (0x09, "lesw"),
    (0x02, "leul"),
    (0x0a, "lesl"),
    (0x03, "leq"),
    (0x11, "beuw"),
    (0x19, "besw"),
    (0x12, "beul"),
    (0x1a, "besl"),
    (0x13, "beq"),
    (0x14, "beo"),
    (0x04, "leo"),
];

const ALIGNMENT_NAMES: [&str; 8] =
    ["un+", "al2+", "al4+", "al8+", "al16+", "al32+", "al64+", "al+"];

const ATOM_NAMES: [Option<&str>; 8] =
    [Some(""), Some("pair+"), Some("w16+"), Some("w16p+"), Some("sub+"), Some("noat+"), None, None];

const PLUGIN_FROM_NAMES: [&str; 4] = ["from-tb", "from-insn", "after-insn", "after-tb"];

fn bswap_flag_name(flags: u64) -> Option<&'static str> {
    match flags {
        1 => Some("iz"),
        2 => Some("oz"),
        4 => Some("os"),
        3 => Some("iz,oz"),
        5 => Some("iz,os"),
        _ => None,
    }
}

fn mb_names(bar: u64) -> (&'static str, &'static str) {
    let b = match bar as u32 & mo::BAR_SC {
        0 => "none",
        mo::BAR_LDAQ => "acq",
        mo::BAR_STRL => "rel",
        _ => "seq",
    };
    let m = match bar as u32 & mo::ALL {
        0 => "none",
        0x1 => "rr",
        0x4 => "rw",
        0x2 => "wr",
        0x8 => "ww",
        0x5 => "rr+rw",
        0x3 => "rr+wr",
        0x9 => "rr+ww",
        0x6 => "rw+wr",
        0xc => "rw+ww",
        0xa => "wr+ww",
        0x7 => "rr+rw+wr",
        0xd => "rr+rw+ww",
        0xb => "rr+wr+ww",
        0xe => "rw+wr+ww",
        _ => "all",
    };
    (b, m)
}

impl Func {
    /// The name of a temp as printed in dumps, `tcg_get_arg_str`.
    pub fn temp_name(&self, t: Temp) -> String {
        let td = self.temp(t);
        let rel = t.index() as i64 - self.nb_globals() as i64;
        match td.kind {
            TempKind::Fixed | TempKind::Global => {
                td.name.as_deref().unwrap_or_default().to_string()
            }
            TempKind::Tb => format!("loc{rel}"),
            TempKind::Ebb => format!("tmp{rel}"),
            TempKind::Const => match td.ty {
                Type::I32 => format!("$0x{:x}", td.val as i32),
                Type::I64 => format!("$0x{:x}", td.val),
                Type::V64 | Type::V128 | Type::V256 => {
                    format!("v{}$0x{:x}", 64 << (td.ty as u32 - Type::V64 as u32), td.val)
                }
                Type::I128 => panic!("no I128 constants"),
            },
        }
    }

    fn dump_op(&self, out: &mut String, op: &Op, have_prefs: bool) {
        let start = out.len();
        let c = op.opc;
        let def = c.def();
        let nb_oargs;
        if c == Opcode::InsnStart {
            nb_oargs = 0;
            out.push_str("\n ----");
            for i in 0..crate::types::INSN_START_WORDS {
                let _ = write!(out, " {:016x}", op.args[i]);
            }
        } else if c == Opcode::Call {
            let info = self.helper_info(op.call_helper());
            nb_oargs = op.callo as usize;
            let nb_iargs = op.calli as usize;
            let _ = write!(out, " {} {},$0x{:x},${}", def.name, info.name, info.flags, nb_oargs);
            for i in 0..nb_oargs + nb_iargs {
                let _ = write!(out, ",{}", self.temp_name(op.arg_temp(i)));
            }
        } else {
            if def.flags & opf::INT != 0 {
                let _ = write!(out, " {}_i{} ", def.name, 8 * op.ty.size());
            } else if def.flags & opf::VECTOR != 0 {
                let _ = write!(out, "{} v{},e{},", def.name, 8 * op.ty.size(), 8 << op.vece);
            } else {
                let _ = write!(out, " {} ", def.name);
            }
            nb_oargs = def.nb_oargs as usize;
            let nb_iargs = def.nb_iargs as usize;
            let nb_cargs = def.nb_cargs as usize;
            let mut k = 0usize;
            for _ in 0..nb_oargs + nb_iargs {
                let sep = if k > 0 { "," } else { "" };
                let _ = write!(out, "{sep}{}", self.temp_name(op.arg_temp(k)));
                k += 1;
            }
            let mut i = match c {
                Opcode::Brcond
                | Opcode::Setcond
                | Opcode::Negsetcond
                | Opcode::Movcond
                | Opcode::CmpVec
                | Opcode::CmpselVec => {
                    match Cond::from_u64(op.args[k]) {
                        Some(cond) => {
                            let _ = write!(out, ",{}", cond.name());
                        }
                        None => {
                            let _ = write!(out, ",$0x{:x}", op.args[k]);
                        }
                    }
                    k += 1;
                    1
                }
                Opcode::QemuLd | Opcode::QemuSt | Opcode::QemuLd2 | Opcode::QemuSt2 => {
                    let oi = MemOpIdx(op.args[k] as u32);
                    k += 1;
                    let mop = oi.memop().0;
                    let ix = oi.mmu_idx();
                    let s_tlb = if mop & MemOp::ALIGN_TLB_ONLY.0 != 0 { "tlb+" } else { "" };
                    let s_al = ALIGNMENT_NAMES[((mop & MemOp::AMASK.0) >> MemOp::ASHIFT) as usize];
                    let key = mop & (MemOp::BSWAP.0 | MemOp::SSIZE.0);
                    let s_op = LDST_NAMES.iter().find(|(v, _)| *v == key).map(|(_, n)| *n);
                    let s_at =
                        ATOM_NAMES[((mop & MemOp::ATOM_MASK.0) >> MemOp::ATOM_SHIFT) as usize];
                    let rest = mop
                        & !(MemOp::AMASK.0
                            | MemOp::BSWAP.0
                            | MemOp::SSIZE.0
                            | MemOp::ATOM_MASK.0
                            | MemOp::ALIGN_TLB_ONLY.0);
                    match (rest, s_op, s_at) {
                        (0, Some(s_op), Some(s_at)) => {
                            let _ = write!(out, ",{s_at}{s_al}{s_tlb}{s_op},{ix}");
                        }
                        _ => {
                            let _ = write!(out, ",$0x{mop:x},{ix}");
                        }
                    }
                    1
                }
                Opcode::Bswap16 | Opcode::Bswap32 | Opcode::Bswap64 => {
                    let flags = op.args[k];
                    match bswap_flag_name(flags) {
                        Some(n) => {
                            let _ = write!(out, ",{n}");
                        }
                        None => {
                            let _ = write!(out, ",$0x{flags:x}");
                        }
                    }
                    k = 1;
                    1
                }
                Opcode::PluginCb => {
                    let from = op.args[k];
                    k += 1;
                    match PLUGIN_FROM_NAMES.get(from as usize) {
                        Some(n) => out.push_str(n),
                        None => {
                            let _ = write!(out, "$0x{from:x}");
                        }
                    }
                    1
                }
                _ => 0,
            };
            match c {
                Opcode::SetLabel | Opcode::Br | Opcode::Brcond => {
                    let sep = if k > 0 { "," } else { "" };
                    let _ = write!(out, "{sep}$L{}", op.arg_label(k).id());
                    i += 1;
                    k += 1;
                }
                Opcode::Mb => {
                    let sep = if k > 0 { "," } else { "" };
                    let (b, m) = mb_names(op.args[k]);
                    let _ = write!(out, "{sep}{b}:{m}");
                    i += 1;
                    k += 1;
                }
                _ => {}
            }
            while i < nb_cargs {
                let sep = if k > 0 { "," } else { "" };
                let _ = write!(out, "{sep}$0x{:x}", op.args[k]);
                i += 1;
                k += 1;
            }
        }
        let _ = nb_oargs;

        if have_prefs || op.life != 0 {
            let col = out.len() - start;
            for _ in col..40 {
                out.push(' ');
            }
        }
        if op.life != 0 {
            let mut life = op.life;
            if life & (SYNC_ARG * 3) != 0 {
                out.push_str("  sync:");
                for i in 0..2 {
                    if life & (SYNC_ARG << i) != 0 {
                        let _ = write!(out, " {i}");
                    }
                }
            }
            life /= DEAD_ARG;
            if life != 0 {
                out.push_str("  dead:");
                let mut i = 0;
                while life != 0 {
                    if life & 1 != 0 {
                        let _ = write!(out, " {i}");
                    }
                    i += 1;
                    life >>= 1;
                }
            }
        }
        out.push('\n');
    }

    /// `tcg_dump_ops`: every linked op, one per line. `have_prefs` is true for the dump taken
    /// after liveness analysis.
    pub fn dump_ops(&self, have_prefs: bool) -> String {
        let mut out = String::new();
        for (_, op) in self.ops() {
            self.dump_op(&mut out, op, have_prefs);
        }
        out
    }
}
