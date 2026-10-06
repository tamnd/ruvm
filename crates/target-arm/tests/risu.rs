// SPDX-License-Identifier: GPL-2.0-or-later

//! risu traces recorded on real AArch64 hardware, replayed through `ruvm-jit`.
//!
//! risu (<https://gitlab.com/pm215/risu>) generates an image of random instructions from an
//! instruction group, each followed by a risu op (a UDF the master catches as SIGILL). The
//! master runs the image on real hardware and writes a trace: for every op, the register state
//! (`struct reginfo`) or the 8 KiB memory block it compares. `scripts/risu/` makes the images
//! and records the traces; the small ones are under `tests/data/risu/`.
//!
//! ruvm's linux-user is a stub, so the image is not run as risu's apprentice would run it.
//! Instead this harness plays the part of both the apprentice and the Linux kernel under it:
//! the image runs at EL0 with the MMU on, every UDF and every other exception goes to an EL1
//! vector holding WFI, and the harness does what risu's signal handler does with the op
//! (`OP_SETMEMBLOCK`, `OP_GETMEMBLOCK`, compare the registers or the memory block against the
//! next trace record), then returns to EL0 after the op. Registers are compared the way risu
//! compares them, the whole `struct reginfo` except the fault address.
//!
//! Every test block is checked from the state the hardware recorded: when a block differs,
//! the harness reports it, loads the recorded registers (or memory block) and carries on, so
//! one difference does not spill into the blocks after it.
//!
//! The memory map: RAM from 0 to 16 MiB, identity mapped with 2 MiB blocks. The first block
//! is EL1 only and holds the vectors at 0 (VBAR_EL1) and the page tables at 1 MiB; the rest
//! is read, write and execute at EL0 and holds the image at 2 MiB and the EL0 stack at the
//! top.
//!
//! `RUVM_RISU_DIR` names a directory of bigger traces, `<name>.bin` (or `.bin.gz`) with
//! `<name>.trace` (or `.trace.gz`), which `big_traces` replays when it is set;
//! `scripts/risu/regen.sh` makes them.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use ruvm_jit::cpu_exec::cpu_exec;
use ruvm_jit::{Jit, Vcpu, excp};
use ruvm_mem::{AddressSpace, MemTxAttrs, MemorySystem};
use ruvm_target_arm::cpu::{ArmCpuModel, ArmFeatures, CpuArmState};
use ruvm_target_arm::tcg::{
    Arm, create_vcpu, new_jit, save_vcpu, vfp_get_fpsr, vfp_set_fpcr, vfp_set_fpsr,
};

const U: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const RAM_SIZE: u64 = 0x100_0000;
const BLOCK: u64 = 0x20_0000;
const PT: u64 = 0x10_0000;
const IMAGE: u64 = 0x20_0000;
const STACK_EL0: u64 = RAM_SIZE - 0x1000;
const STACK_EL1: u64 = 0x8000;
const WFI: u32 = 0xd503_207f;
/// Where a synchronous exception from EL0 halts: the WFI at VBAR_EL1 + 0x400, plus 4.
const HALT_PC: u64 = 0x404;

/// risu's ops (`RisuOp`), and `OP_SIGILL` for an UNDEF that is not one.
const OP_COMPARE: i32 = 0;
const OP_TESTEND: i32 = 1;
const OP_SETMEMBLOCK: i32 = 2;
const OP_GETMEMBLOCK: i32 = 3;
const OP_COMPAREMEM: i32 = 4;
const OP_SIGILL: i32 = -1;
/// An exception risu's master would have died of: anything but an UNDEF.
const OP_FAULT: i32 = -2;

const RISU_MAGIC: u32 = 0x5249_5355;
const MEMBLOCKLEN: usize = 8192;
/// `sizeof(struct reginfo)` as risu writes it without SVE: the 32 V registers in `extra`.
const REGINFO_SIZE: usize = 816;
const REGINFO_VREGS: usize = 304;

/// One trace record: the op, the PC as an offset into the image, and the payload.
struct Rec<'a> {
    op: i32,
    pc: u64,
    data: &'a [u8],
}

fn gunzip(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    flate2::read::MultiGzDecoder::new(bytes).read_to_end(&mut out).expect("gzip data");
    out
}

/// Read `path`, or `path` with `.gz` appended, uncompressing it.
fn load(path: &Path) -> Option<Vec<u8>> {
    if let Ok(b) = std::fs::read(path) {
        return Some(if b.starts_with(&[0x1f, 0x8b]) { gunzip(&b) } else { b });
    }
    let mut gz = path.as_os_str().to_owned();
    gz.push(".gz");
    std::fs::read(PathBuf::from(gz)).ok().map(|b| gunzip(&b))
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

fn parse(trace: &[u8]) -> Vec<Rec<'_>> {
    let mut recs = Vec::new();
    let mut off = 0;
    while off < trace.len() {
        assert!(off + 24 <= trace.len(), "truncated trace header at {off:#x}");
        assert_eq!(u32_at(trace, off), RISU_MAGIC, "bad trace magic at {off:#x}");
        let size = u32_at(trace, off + 4) as usize;
        let op = u32_at(trace, off + 8) as i32;
        let pc = u64_at(trace, off + 16);
        off += 24;
        let data = &trace[off..off + size];
        off += size;
        match op {
            OP_COMPARE | OP_TESTEND | OP_SIGILL => assert_eq!(
                size, REGINFO_SIZE,
                "a reginfo of {size} bytes: only traces without SVE or SME state are supported"
            ),
            OP_COMPAREMEM => assert_eq!(size, MEMBLOCKLEN),
            OP_SETMEMBLOCK | OP_GETMEMBLOCK => assert_eq!(size, 0),
            _ => panic!("unknown risu op {op}"),
        }
        recs.push(Rec { op, pc, data });
    }
    recs
}

/// The `struct reginfo` risu's `reginfo_init()` would make from `st`, stopped at `insn` at
/// offset `pc` into the image.
fn reginfo(st: &CpuArmState, pc: u64, insn: u32) -> Vec<u8> {
    let mut r = vec![0u8; REGINFO_SIZE];
    for i in 0..31 {
        r[8 + i * 8..16 + i * 8].copy_from_slice(&st.xregs[i].to_le_bytes());
    }
    r[256..264].copy_from_slice(&0xdead_beef_dead_beef_u64.to_le_bytes());
    r[264..272].copy_from_slice(&pc.to_le_bytes());
    r[272..276].copy_from_slice(&st.nzcv().to_le_bytes());
    r[276..280].copy_from_slice(&insn.to_le_bytes());
    r[280..284].copy_from_slice(&vfp_get_fpsr(st).to_le_bytes());
    r[284..288].copy_from_slice(&st.fpcr.to_le_bytes());
    for i in 0..32 {
        let o = REGINFO_VREGS + i * 16;
        r[o..o + 8].copy_from_slice(&st.zregs[i][0].to_le_bytes());
        r[o + 8..o + 16].copy_from_slice(&st.zregs[i][1].to_le_bytes());
    }
    r
}

/// Load the registers of a recorded `struct reginfo` into `st`.
fn restore(st: &mut CpuArmState, r: &[u8], f: &ArmFeatures) {
    for i in 0..31 {
        st.xregs[i] = u64_at(r, 8 + i * 8);
    }
    st.set_nzcv(u32_at(r, 272));
    vfp_set_fpcr(st, u32_at(r, 284), f);
    vfp_set_fpsr(st, u32_at(r, 280));
    for i in 0..32 {
        let o = REGINFO_VREGS + i * 16;
        st.zregs[i] = [0; 32];
        st.zregs[i][0] = u64_at(r, o);
        st.zregs[i][1] = u64_at(r, o + 8);
    }
}

/// The fields of two reginfos that differ, as risu's `reginfo_dump_mismatch()` lists them:
/// `name: ours (hardware)`.
fn diff(ours: &[u8], hw: &[u8]) -> String {
    let mut s = String::new();
    for i in 0..31 {
        let (a, b) = (u64_at(ours, 8 + i * 8), u64_at(hw, 8 + i * 8));
        if a != b {
            let _ = write!(s, " x{i}: {a:#x} ({b:#x})");
        }
    }
    let (a, b) = (u64_at(ours, 264), u64_at(hw, 264));
    if a != b {
        let _ = write!(s, " pc: {a:#x} ({b:#x})");
    }
    for (name, off) in [("nzcv", 272), ("insn", 276), ("fpsr", 280), ("fpcr", 284)] {
        let (a, b) = (u32_at(ours, off), u32_at(hw, off));
        if a != b {
            let _ = write!(s, " {name}: {a:#x} ({b:#x})");
        }
    }
    if ours[256..264] != hw[256..264] || ours[288..REGINFO_VREGS] != hw[288..REGINFO_VREGS] {
        s.push_str(" sp/sve_vl/svcr differ");
    }
    for i in 0..32 {
        let o = REGINFO_VREGS + i * 16;
        if ours[o..o + 16] != hw[o..o + 16] {
            let v = |b: &[u8]| (u64_at(b, o + 8), u64_at(b, o));
            let ((ah, al), (bh, bl)) = (v(ours), v(hw));
            let _ = write!(s, " v{i}: {ah:016x}{al:016x} ({bh:016x}{bl:016x})");
        }
    }
    s
}

fn op_name(op: i32) -> &'static str {
    match op {
        OP_COMPARE => "COMPARE",
        OP_TESTEND => "TESTEND",
        OP_SETMEMBLOCK => "SETMEMBLOCK",
        OP_GETMEMBLOCK => "GETMEMBLOCK",
        OP_COMPAREMEM => "COMPAREMEM",
        OP_SIGILL => "SIGILL",
        _ => "an exception",
    }
}

/// The CPU the traces are replayed on: `max` without EL2, EL3 or PAuth, so the image runs
/// at EL0 under an EL1 the harness stands in for, and pointer authentication (whose keys
/// differ from the hardware's) is out of the comparison.
fn model() -> ArmCpuModel {
    let mut m = ArmCpuModel::max().with_pauth(None);
    m.features.el2 = false;
    m.features.el3 = false;
    // ID_AA64PFR0_EL1.EL2 and EL3.
    m.id_aa64pfr0 &= !0xff00;
    m
}

struct World {
    _sys: MemorySystem,
    as_: Arc<AddressSpace>,
    jit: Arc<Jit>,
    arm: Arc<Arm>,
}

impl World {
    fn new(image: &[u8]) -> World {
        assert!(
            (image.len() as u64) < STACK_EL0 - 0x1_0000 - IMAGE,
            "the image does not fit in RAM"
        );
        let sys = MemorySystem::new();
        let root = sys.new_container("system", 1 << 64).unwrap();
        let as_ = sys.address_space_init(root, "memory").unwrap();
        let ram = sys.new_ram("ram", RAM_SIZE).unwrap();
        sys.add_subregion(root, 0, ram).unwrap();
        let w = World { _sys: sys, as_, jit: new_jit(), arm: Arc::new(Arm::new(model())) };
        let wfi: Vec<u8> = (0..0x200).flat_map(|_| WFI.to_le_bytes()).collect();
        w.write(0, &wfi);
        // L0 and L1 tables pointing at one L2 table of 2 MiB blocks: AttrIndx 0, inner
        // shareable, AF. The first block is EL1 only and not executable at EL0 (UXN); the
        // rest is AP 01, read and write at EL0 and EL1.
        w.write(PT, &((PT + 0x1000) | 3).to_le_bytes());
        w.write(PT + 0x1000, &((PT + 0x2000) | 3).to_le_bytes());
        w.write(PT + 0x2000, &(0x701u64 | (1 << 54)).to_le_bytes());
        for b in 1..RAM_SIZE / BLOCK {
            w.write(PT + 0x2000 + b * 8, &((b * BLOCK) | 0x741).to_le_bytes());
        }
        w.write(IMAGE, image);
        w
    }

    fn write(&self, addr: u64, bytes: &[u8]) {
        assert!(self.as_.write(addr, U, bytes).is_ok());
    }

    fn read(&self, addr: u64, len: usize) -> Vec<u8> {
        let mut b = vec![0; len];
        assert!(self.as_.read(addr, U, &mut b).is_ok(), "read of {addr:#x}");
        b
    }

    fn r32(&self, addr: u64) -> u32 {
        u32_at(&self.read(addr, 4), 0)
    }

    fn features(&self) -> &ArmFeatures {
        &self.arm.model().features
    }

    /// The state a Linux process starts the image in: EL0 with the MMU on, FP enabled, and
    /// SCTLR_EL1 as Linux sets it for user space.
    fn state(&self) -> CpuArmState {
        let m = self.arm.model();
        let mut st = CpuArmState::reset(m);
        st.mair_el[1] = 0xff;
        // T0SZ 16, write back cacheable inner shareable walks, 4K granule, EPD1, TBI0.
        st.tcr_el[1] = 16 | (1 << 8) | (1 << 10) | (3 << 12) | (1 << 23) | (1 << 37);
        st.ttbr0_el[1] = PT;
        // M, C, SA, SA0, I, DZE, UCT, nTWE and UCI, with nTWI clear as in Linux.
        st.sctlr_el[1] = m.reset_sctlr
            | 1
            | (1 << 2)
            | (1 << 3)
            | (1 << 4)
            | (1 << 12)
            | (1 << 14)
            | (1 << 15)
            | (1 << 18)
            | (1 << 26);
        st.cpacr_el1 = 3 << 20;
        st.vbar_el[1] = 0;
        st.sp_el[1] = STACK_EL1;
        st.pstate_write(0);
        st.xregs[31] = STACK_EL0;
        st.pc = IMAGE;
        st
    }
}

/// What a replay found.
#[derive(Default)]
struct Report {
    /// Ops compared: register and memory compares.
    checks: usize,
    /// Instructions the hardware executed and ruvm rejected because their feature is not in
    /// the model, by feature (see [`absent_feature`]).
    skipped: BTreeMap<&'static str, usize>,
    /// Mismatches, one line each.
    mismatches: Vec<String>,
    /// The trace ended before TESTEND, or the replay lost sync with it.
    lost: Option<String>,
}

/// The extensions the M4 has and ruvm's `max` model does not (its ID registers do not
/// advertise them, so they UNDEF there as on hardware without them): the mask and value of
/// their AdvSIMD, FP and load/store encodings. risu's broader patterns (FCVT, the by element
/// forms) still produce some of them after `scripts/risu` leaves their own patterns out.
const ABSENT: &[(u32, u32, &str)] = &[
    (0xbf20_ec00, 0x2e00_e400, "FEAT_FCMA"),
    (0xbf20_e400, 0x2e00_c400, "FEAT_FCMA"),
    (0xbf00_9400, 0x2f00_1000, "FEAT_FCMA"),
    (0xffff_fc00, 0x1e63_4000, "FEAT_BF16"),
    (0xbfff_fc00, 0x0ea1_6800, "FEAT_BF16"),
    (0xbfe0_fc00, 0x2e40_fc00, "FEAT_BF16"),
    (0xbfc0_f400, 0x0f40_f000, "FEAT_BF16"),
    (0xbfe0_fc00, 0x2ec0_fc00, "FEAT_BF16"),
    (0xbfc0_f400, 0x0fc0_f000, "FEAT_BF16"),
    (0xffe0_fc00, 0x6e40_ec00, "FEAT_BF16"),
    (0xbf60_fc00, 0x0e20_ec00, "FEAT_FHM"),
    (0xbf60_fc00, 0x2e20_cc00, "FEAT_FHM"),
    (0xbfc0_b400, 0x0f80_0000, "FEAT_FHM"),
    (0xbfc0_b400, 0x2f80_8000, "FEAT_FHM"),
    (0xffe0_f000, 0xce60_8000, "FEAT_SHA512/SHA3"),
    (0xffff_fc00, 0xcec0_8000, "FEAT_SHA512/SHA3"),
    (0xffc0_8000, 0xce00_0000, "FEAT_SHA512/SHA3"),
    (0xffe0_0000, 0xce80_0000, "FEAT_SHA512/SHA3"),
    (0x3f20_0c00, 0x1900_0000, "FEAT_LRCPC2"),
];

/// The extension of [`ABSENT`] that `insn` belongs to.
fn absent_feature(insn: u32) -> Option<&'static str> {
    ABSENT.iter().find(|&&(m, v, _)| insn & m == v).map(|&(_, _, f)| f)
}

/// Run `v` until it halts in the EL1 vector.
fn run_to_halt(v: &mut Vcpu) -> bool {
    for _ in 0..10_000 {
        let r = cpu_exec(&mut v.cpu());
        if r == excp::HLT {
            return true;
        }
    }
    false
}

/// The instruction words of the test block that ended at `pc`: from after the previous op
/// at `prev` up to `pc`, at most 8 of them.
fn block_words(w: &World, prev: u64, pc: u64) -> String {
    let start = (prev + 4).max(pc.saturating_sub(32));
    let mut s = String::new();
    let mut a = start;
    while a < pc {
        let _ = write!(s, "{:08x} ", w.r32(IMAGE + a));
        a += 4;
    }
    s.trim_end().to_string()
}

/// Replay the trace `trace` of the image `image`.
fn replay(image: &[u8], trace: &[u8]) -> Report {
    let recs = parse(trace);
    let w = World::new(image);
    let mut st = w.state();
    let mut v = create_vcpu(&w.jit, w.arm.clone(), w.as_.clone(), &st);
    let mut rep = Report::default();
    let mut memblock = 0u64;
    let mut prev_pc = 0u64;
    let mut i = 0;
    // An instruction was stepped over since the last compare, so the next compare is
    // expected to differ and only resyncs.
    let mut tainted = false;
    loop {
        if !run_to_halt(&mut v) {
            rep.lost = Some(format!("the vCPU did not stop after the op at {prev_pc:#x}"));
            return rep;
        }
        st = save_vcpu(&v);
        if st.pc != HALT_PC {
            rep.lost = Some(format!(
                "halted at {:#x}, not in the lower EL synchronous vector, ESR {:#x} ELR {:#x}",
                st.pc, st.esr_el[1], st.elr_el[1]
            ));
            return rep;
        }
        // Back to EL0 at the trapping instruction, as the kernel does before the signal. Its
        // exception return clears the local exclusive monitor.
        st.exclusive_addr = u64::MAX;
        let elr = st.elr_el[1];
        let esr = st.esr_el[1];
        st.save_sp(1);
        st.pstate_write(st.spsr_el[1] as u32);
        st.restore_sp(0);
        st.pc = elr;
        let off = elr.wrapping_sub(IMAGE);
        let insn = if off < image.len() as u64 { w.r32(elr) } else { 0 };
        let op = if esr >> 26 != 0 {
            OP_FAULT
        } else if insn & !0xf == 0x5af0 {
            (insn & 0xf) as i32
        } else {
            OP_SIGILL
        };
        let Some(rec) = recs.get(i) else {
            rep.lost = Some(format!("the trace ended before {} at {off:#x}", op_name(op)));
            return rep;
        };
        let is_reg = |op| matches!(op, OP_COMPARE | OP_TESTEND | OP_SIGILL);
        let mut resync = false;
        // Whether a register op ended the test block.
        let mut ended = false;
        if op == OP_SIGILL && off < rec.pc {
            // ruvm rejected an instruction the hardware executed. Count or report it, step
            // over it, and take the hardware's state at the next op without comparing.
            match absent_feature(insn) {
                Some(f) => *rep.skipped.entry(f).or_default() += 1,
                None => rep.mismatches.push(format!(
                    "{} at {:#x} [{}]: ruvm SIGILL at {off:#x} [{insn:08x}]",
                    op_name(rec.op),
                    rec.pc,
                    block_words(&w, prev_pc, rec.pc)
                )),
            }
            tainted = true;
        } else if is_reg(op) && is_reg(rec.op) {
            i += 1;
            ended = true;
            rep.checks += usize::from(!tainted);
            let ours = reginfo(&st, off, insn);
            if ours[8..] == rec.data[8..] {
                if op == OP_TESTEND {
                    return rep;
                }
            } else if tainted {
                resync = true;
            } else {
                let mut line = format!(
                    "{} at {:#x} [{}]:",
                    op_name(rec.op),
                    rec.pc,
                    block_words(&w, prev_pc, rec.pc.max(off))
                );
                if op != rec.op {
                    let _ = write!(line, " ruvm {} at {off:#x};", op_name(op));
                }
                line.push_str(&diff(&ours, rec.data));
                rep.mismatches.push(line);
                resync = true;
            }
            tainted = false;
        } else if op == rec.op && rec.pc == off {
            i += 1;
            match op {
                OP_SETMEMBLOCK => memblock = st.xregs[0],
                OP_GETMEMBLOCK => st.xregs[0] = st.xregs[0].wrapping_add(memblock),
                OP_COMPAREMEM => {
                    let ours = w.read(memblock, MEMBLOCKLEN);
                    if std::mem::take(&mut tainted) {
                        w.write(memblock, rec.data);
                    } else if ours == rec.data {
                        rep.checks += 1;
                    } else {
                        rep.checks += 1;
                        let at = ours.iter().zip(rec.data).position(|(a, b)| a != b).unwrap();
                        rep.mismatches.push(format!(
                            "COMPAREMEM at {:#x} [{}]: memory differs from byte {at:#x}: \
                             {:02x?} ({:02x?})",
                            rec.pc,
                            block_words(&w, prev_pc, rec.pc),
                            &ours[at..(at + 16).min(MEMBLOCKLEN)],
                            &rec.data[at..(at + 16).min(MEMBLOCKLEN)],
                        ));
                        w.write(memblock, rec.data);
                    }
                }
                _ => unreachable!(),
            }
        } else {
            i += 1;
            // A different op or place: ruvm executed what the hardware rejected, or took
            // some other exception.
            let what = if op == OP_FAULT {
                format!("ruvm took ESR {esr:#x} at {off:#x} [{insn:08x}]")
            } else {
                format!("ruvm {} at {off:#x} [{insn:08x}]", op_name(op))
            };
            rep.mismatches.push(format!(
                "{} at {:#x} [{}]: {what}",
                op_name(rec.op),
                rec.pc,
                block_words(&w, prev_pc, rec.pc.max(off))
            ));
            match rec.op {
                OP_COMPAREMEM => w.write(memblock, rec.data),
                op if is_reg(op) => {}
                _ => {
                    rep.lost = Some(format!("lost sync at {} {:#x}", op_name(rec.op), rec.pc));
                    return rep;
                }
            }
            resync = true;
        }
        if resync {
            // Carry on from the hardware's state after the record's op.
            if is_reg(rec.op) {
                restore(&mut st, rec.data, w.features());
                if rec.op == OP_TESTEND {
                    return rep;
                }
            }
            st.pc = IMAGE + rec.pc;
        }
        if ended || resync {
            prev_pc = st.pc - IMAGE;
        }
        st.pc += 4;
        st.rebuild_hflags(w.features());
        st.store(&mut v.env);
        v.shared().halted.store(0, Ordering::Release);
    }
}

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/risu")
}

/// Replay `<dir>/<name>.bin` against `<dir>/<name>.trace` and print what differed.
fn replay_file(dir: &Path, name: &str) -> Report {
    let image = load(&dir.join(format!("{name}.bin"))).expect("risu image");
    let trace = load(&dir.join(format!("{name}.trace"))).expect("risu trace");
    let rep = replay(&image, &trace);
    println!("{name}: {} checks, {} mismatches", rep.checks, rep.mismatches.len());
    for (f, n) in &rep.skipped {
        println!("  {n} instructions of {f} skipped: not in the model");
    }
    for m in &rep.mismatches {
        println!("  {m}");
    }
    if let Some(l) = &rep.lost {
        println!("  {l}");
    }
    rep
}

#[track_caller]
fn check(name: &str) {
    let rep = replay_file(&data_dir(), name);
    assert!(rep.lost.is_none(), "{name}: {}", rep.lost.unwrap());
    assert!(rep.checks > 0, "{name}: nothing was compared");
    assert!(rep.mismatches.is_empty(), "{name}: {} mismatches", rep.mismatches.len());
}

/// The base A64 group: integer data processing, loads and stores, exclusives, and the
/// reserved encodings that must UNDEF.
#[test]
fn risu_a64() {
    check("a64");
}

/// FP and AdvSIMD, with their loads and stores and the AES and SHA instructions.
#[test]
fn risu_a64_v() {
    check("a64_v");
}

/// The v8.1 and v8.2 groups: SQRDMLAH and SQRDMLSH, and the half precision FP instructions.
#[test]
fn risu_a64_ext() {
    check("a64_ext");
}

/// The traces in `RUVM_RISU_DIR`, if it is set.
#[test]
fn big_traces() {
    let Some(dir) = std::env::var_os("RUVM_RISU_DIR") else {
        return;
    };
    let dir = PathBuf::from(dir);
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .expect("RUVM_RISU_DIR")
        .filter_map(|e| {
            let n = e.ok()?.file_name().into_string().ok()?;
            let n = n.strip_suffix(".gz").unwrap_or(&n);
            n.strip_suffix(".trace").map(str::to_string)
        })
        .filter(|n| n.starts_with("a64"))
        .collect();
    names.sort();
    names.dedup();
    let mut bad = Vec::new();
    for n in &names {
        let rep = replay_file(&dir, n);
        if rep.lost.is_some() || !rep.mismatches.is_empty() {
            bad.push(n.clone());
        }
    }
    assert!(bad.is_empty(), "traces that differ: {bad:?}");
}
