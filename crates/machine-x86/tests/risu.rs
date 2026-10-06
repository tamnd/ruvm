// SPDX-License-Identifier: GPL-2.0-or-later

//! risu traces recorded on real x86-64 hardware (an AMD EPYC), replayed through `ruvm-jit`
//! and the x86 translator.
//!
//! risu (<https://gitlab.com/pm215/risu>, with Jan Bobek's x86 support) generates an image of
//! random instructions from an instruction group, each followed by a risu op (a UD1 the
//! master catches as SIGILL). The master runs the image on real hardware and writes a trace:
//! for every op, the register state (`struct reginfo`) or the 8 KiB memory block it compares.
//! `scripts/risu/` makes the images and records the traces; the small ones are under
//! `tests/data/risu/`.
//!
//! ruvm's linux-user is a stub, so the image is not run as risu's apprentice would run it.
//! This harness plays the part of both the apprentice and the Linux kernel under it: the
//! image runs in 64-bit mode with paging on, every exception goes to an IDT handler that
//! halts, and the harness does what risu's signal handler does with the op
//! (`OP_SETMEMBLOCK`, `OP_GETMEMBLOCK`, compare the registers or the memory block against the
//! next trace record), then returns after the op. Registers are compared the way risu
//! compares them: the general purpose registers but RSP, RIP as an offset into the image,
//! the arithmetic flags, the four bytes at RIP, MXCSR, XMM0 to XMM15 and, in traces recorded
//! with AVX state, the upper halves of YMM0 to YMM15.
//!
//! Every test block is checked from the state the hardware recorded: when a block differs,
//! the harness reports it, loads the recorded registers (or memory block) and carries on.
//! When ruvm raises an exception the hardware did not, the harness reports it and carries on
//! from the hardware's state at the next register compare.
//!
//! The image runs at CPL 0 rather than 3; none of the generated instructions behave
//! differently there. The undefined flags of the integer instructions are cleared in the
//! image after each instruction (see `scripts/risu/patches`), so the traces are exact.
//!
//! The memory map: RAM from 0 to 64 MiB, identity mapped with 2 MiB pages, with the GDT, the
//! IDT and its handlers, the stack and the page tables in the first 2 MiB and the image at
//! 2 MiB, mapped again at `IMAGE_VA` where it runs. The image holds the memory block, so
//! stores to it run through the translator's self-modifying code handling.
//!
//! Instructions the EPYC model advertises but the translator leaves out (it raises #UD for
//! them, see `ruvm_target_x86::tcg::translate::sse`) are counted as skipped rather than
//! reported, and so are the differences listed in `KNOWN`: ones the hardware traces found
//! in the translator, which these tests do not fail on until they are fixed.
//!
//! `RUVM_RISU_DIR` names a directory of bigger traces, `x86_<name>.bin` (or `.bin.gz`) with
//! `x86_<name>.trace` (or `.trace.gz`), which `big_traces` replays when it is set;
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
use ruvm_target_x86::cpuid::{Accel, X86Cpu};
use ruvm_target_x86::state::{
    CR0_ET_MASK, CR0_MP_MASK, CR0_NE_MASK, CR0_PE_MASK, CR0_PG_MASK, CR0_WP_MASK, CR4_OSFXSR_MASK,
    CR4_OSXMMEXCPT_MASK, CR4_OSXSAVE_MASK, CR4_PAE_MASK, DESC_A_MASK, DESC_B_MASK, DESC_CS_MASK,
    DESC_G_MASK, DESC_L_MASK, DESC_P_MASK, DESC_R_MASK, DESC_S_MASK, DESC_W_MASK, HF_LMA_MASK,
    IrqchipMode, MSR_EFER_LME, R_CS, R_EAX, R_EBP, R_EBX, R_ECX, R_EDI, R_EDX, R_ESI, R_ESP,
    ResetConfig, SegmentCache, X86CpuState,
};
use ruvm_target_x86::tcg::{X86, create_vcpu, env, new_jit, save_vcpu};

const U: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const RAM_SIZE: u64 = 0x400_0000;
const GDT: u64 = 0x3000;
const IDT: u64 = 0x4000;
const HANDLERS: u64 = 0x5000;
const PML4: u64 = 0x1_0000;
const STACK: u64 = 0x8_0000;
const IMAGE: u64 = 0x20_0000;
/// Where the image is mapped, a high user address as Linux gives risu's `mmap()`. The flags
/// of the address arithmetic in the memory setup of a test block depend on it.
const IMAGE_VA: u64 = 0x7f80_0000_0000;
/// `HF_OSFXSR_MASK`, which `cpu_x86_update_cr4()` derives from CR4.OSFXSR.
const HF_OSFXSR: u32 = 1 << 22;
/// The flags risu records: OF, SF, ZF, AF, PF and CF.
const FLAGS: u64 = 0x8d5;
const UD: u64 = 6;

const CODE64: u32 = DESC_P_MASK
    | DESC_S_MASK
    | DESC_CS_MASK
    | DESC_R_MASK
    | DESC_A_MASK
    | DESC_L_MASK
    | DESC_G_MASK;
const DATA32: u32 =
    DESC_P_MASK | DESC_S_MASK | DESC_W_MASK | DESC_A_MASK | DESC_B_MASK | DESC_G_MASK;

/// risu's ops (`RisuOp`), and `OP_SIGILL` for a #UD that is not one.
const OP_COMPARE: i32 = 0;
const OP_TESTEND: i32 = 1;
const OP_SETMEMBLOCK: i32 = 2;
const OP_GETMEMBLOCK: i32 = 3;
const OP_COMPAREMEM: i32 = 4;
const OP_SIGILL: i32 = -1;
/// An exception risu's master would have died of: anything but a #UD.
const OP_FAULT: i32 = -2;

const RISU_MAGIC: u32 = 0x5249_5355;
const MEMBLOCKLEN: usize = 8192;
/// `sizeof(struct reginfo)` on x86-64.
const REGINFO_SIZE: usize = 2312;
const RI_MXCSR: usize = 4;
const RI_XFEATURES: usize = 8;
const RI_GREGS: usize = 16;
const RI_VREGS: usize = RI_GREGS + 23 * 8;
/// `XFEAT_AVX` in `xfeatures`.
const XFEAT_AVX: u64 = 4;
/// The `gregset_t` slots of the general purpose registers, `REG_R8` to `REG_RCX`, and ruvm's
/// numbers for them; then `REG_RSP`, `REG_RIP` and `REG_EFL`.
const GREGS: [usize; 15] =
    [8, 9, 10, 11, 12, 13, 14, 15, R_EDI, R_ESI, R_EBP, R_EBX, R_EDX, R_EAX, R_ECX];
const REG_RSP: usize = 15;
const REG_RIP: usize = 16;
const REG_EFL: usize = 17;
const GREG_NAMES: [&str; 18] = [
    "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "rdi", "rsi", "rbp", "rbx", "rdx", "rax",
    "rcx", "rsp", "rip", "eflags",
];

/// One trace record: the op, the RIP as an offset into the image, and the payload.
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
            OP_COMPARE | OP_TESTEND | OP_SIGILL => assert_eq!(size, REGINFO_SIZE),
            OP_COMPAREMEM => assert_eq!(size, MEMBLOCKLEN),
            OP_SETMEMBLOCK | OP_GETMEMBLOCK => assert_eq!(size, 0),
            _ => panic!("unknown risu op {op}"),
        }
        recs.push(Rec { op, pc, data });
    }
    recs
}

fn is_reg(op: i32) -> bool {
    matches!(op, OP_COMPARE | OP_TESTEND | OP_SIGILL)
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

/// The `struct reginfo` risu's `reginfo_init()` would make from `st`, stopped at `insn` at
/// offset `pc` into the image, with the `xfeatures` of the trace.
fn reginfo(st: &X86CpuState, pc: u64, insn: u32, xfeatures: u64) -> Vec<u8> {
    let mut r = vec![0u8; REGINFO_SIZE];
    r[0..4].copy_from_slice(&insn.to_le_bytes());
    r[RI_MXCSR..RI_MXCSR + 4].copy_from_slice(&st.mxcsr.to_le_bytes());
    r[RI_XFEATURES..RI_XFEATURES + 8].copy_from_slice(&xfeatures.to_le_bytes());
    let mut put = |slot: usize, v: u64| {
        r[RI_GREGS + slot * 8..RI_GREGS + slot * 8 + 8].copy_from_slice(&v.to_le_bytes());
    };
    for (slot, &reg) in GREGS.iter().enumerate() {
        put(slot, st.regs[reg]);
    }
    put(REG_RSP, 0xdead_beef);
    put(REG_RIP, pc);
    put(REG_EFL, st.rflags & FLAGS);
    let quads = if xfeatures & XFEAT_AVX != 0 { 4 } else { 2 };
    for i in 0..16 {
        for q in 0..quads {
            let o = RI_VREGS + i * 64 + q * 8;
            r[o..o + 8].copy_from_slice(&st.xmm_regs[i][q].to_le_bytes());
        }
    }
    r
}

/// Load the registers of the reginfo `ri` into `st`.
fn restore(st: &mut X86CpuState, ri: &[u8]) {
    let greg = |slot: usize| u64_at(ri, RI_GREGS + slot * 8);
    for (slot, &reg) in GREGS.iter().enumerate() {
        st.regs[reg] = greg(slot);
    }
    st.rflags = (st.rflags & !FLAGS) | greg(REG_EFL);
    st.mxcsr = u32_at(ri, RI_MXCSR);
    let quads = if u64_at(ri, RI_XFEATURES) & XFEAT_AVX != 0 { 4 } else { 2 };
    for i in 0..16 {
        for q in 0..quads {
            st.xmm_regs[i][q] = u64_at(ri, RI_VREGS + i * 64 + q * 8);
        }
    }
}

/// The fields of two reginfos that differ, as `name: ours (hardware)`.
fn diff(ours: &[u8], hw: &[u8]) -> String {
    let mut s = String::new();
    if ours[0..4] != hw[0..4] {
        let _ = write!(s, " insn: {:#x} ({:#x})", u32_at(ours, 0), u32_at(hw, 0));
    }
    if ours[RI_MXCSR..RI_MXCSR + 4] != hw[RI_MXCSR..RI_MXCSR + 4] {
        let _ = write!(s, " mxcsr: {:#x} ({:#x})", u32_at(ours, RI_MXCSR), u32_at(hw, RI_MXCSR));
    }
    for (slot, name) in GREG_NAMES.iter().enumerate() {
        let o = RI_GREGS + slot * 8;
        if ours[o..o + 8] != hw[o..o + 8] {
            let _ = write!(s, " {name}: {:#x} ({:#x})", u64_at(ours, o), u64_at(hw, o));
        }
    }
    for i in 0..16 {
        let o = RI_VREGS + i * 64;
        if ours[o..o + 32] != hw[o..o + 32] {
            let ymm = |b: &[u8]| {
                (0..4).rev().map(|q| format!("{:016x}", u64_at(b, o + q * 8))).collect::<String>()
            };
            let _ = write!(s, " ymm{i}: {} ({})", ymm(ours), ymm(hw));
        }
    }
    s
}

struct World {
    _sys: MemorySystem,
    as_: Arc<AddressSpace>,
    jit: Arc<Jit>,
    x86: Arc<X86>,
    image_len: u64,
}

impl World {
    fn new(image: &[u8]) -> World {
        assert!((image.len() as u64) < RAM_SIZE - IMAGE, "the image does not fit in RAM");
        let mut m = X86Cpu::new("EPYC", Accel::Tcg).unwrap();
        let _ = m.realize();
        let sys = MemorySystem::new();
        let root = sys.new_container("system", 1 << 64).unwrap();
        let as_ = sys.address_space_init(root, "memory").unwrap();
        let ram = sys.new_ram("ram", RAM_SIZE).unwrap();
        sys.add_subregion(root, 0, ram).unwrap();
        let w = World {
            _sys: sys,
            as_,
            jit: new_jit(),
            x86: Arc::new(X86::new(m)),
            image_len: image.len() as u64,
        };
        w.w64(GDT, 0);
        w.w64(GDT + 8, 0x00af_9b00_0000_ffff);
        w.w64(GDT + 16, 0x00cf_9300_0000_ffff);
        for n in 0..32u64 {
            let h = HANDLERS + n * 16;
            let lo = (h & 0xffff) | (0x08 << 16) | (0x8e00 << 32) | (((h >> 16) & 0xffff) << 48);
            w.w64(IDT + n * 16, lo);
            w.w64(IDT + n * 16 + 8, h >> 32);
            w.write(h, &[0xf4]);
        }
        w.w64(PML4, (PML4 + 0x1000) | 3);
        w.w64(PML4 + 0x1000, (PML4 + 0x2000) | 3);
        for p in 0..RAM_SIZE >> 21 {
            w.w64(PML4 + 0x2000 + p * 8, (p << 21) | 0x83);
        }
        // IMAGE_VA, in PML4 slot 255, to the RAM from IMAGE up.
        w.w64(PML4 + (IMAGE_VA >> 39) * 8, (PML4 + 0x3000) | 3);
        w.w64(PML4 + 0x3000, (PML4 + 0x4000) | 3);
        for p in 0..(RAM_SIZE - IMAGE) >> 21 {
            w.w64(PML4 + 0x4000 + p * 8, (IMAGE + (p << 21)) | 0x83);
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

    fn w64(&self, addr: u64, v: u64) {
        self.write(addr, &v.to_le_bytes());
    }

    /// The four bytes at offset `off` into the image, as risu's `faulting_insn`.
    fn insn(&self, off: u64) -> u32 {
        if off < self.image_len { u32_at(&self.read(IMAGE + off, 4), 0) } else { 0 }
    }

    /// The physical address of the image address `va`, the memory block's.
    fn phys(&self, va: u64) -> u64 {
        let off = va.wrapping_sub(IMAGE_VA);
        assert!(off < RAM_SIZE - IMAGE, "memory block at {va:#x}, not in the image");
        IMAGE + off
    }

    /// The bytes of the image from `from` to `to`, at most 24 of them, in hex.
    fn bytes(&self, from: u64, to: u64) -> String {
        let from = from.max(to.saturating_sub(24));
        let to = to.min(self.image_len);
        if from >= to {
            return String::new();
        }
        self.read(IMAGE + from, (to - from) as usize).iter().map(|b| format!("{b:02x}")).collect()
    }

    /// The state the image starts in, as a Linux process sees it: 64-bit mode with SSE and
    /// AVX state enabled, MXCSR 0x1f80 and the x87 control word 0x37f.
    fn state(&self) -> X86CpuState {
        let mut st = X86CpuState::new_reset(&ResetConfig {
            is_bsp: true,
            kvm: false,
            irqchip: IrqchipMode::Off,
            cpuid_version: 0x0080_0f12,
            has_monitor: false,
        });
        st.rflags = 2;
        st.rip = IMAGE_VA;
        st.cr4 = CR4_PAE_MASK | CR4_OSFXSR_MASK | CR4_OSXMMEXCPT_MASK | CR4_OSXSAVE_MASK;
        st.xcr0 = 7;
        st.efer = MSR_EFER_LME;
        st.cr3 = PML4;
        st.update_cr0(
            CR0_PE_MASK | CR0_MP_MASK | CR0_ET_MASK | CR0_NE_MASK | CR0_WP_MASK | CR0_PG_MASK,
        );
        assert_ne!(st.hflags & HF_LMA_MASK, 0);
        st.hflags |= HF_OSFXSR;
        st.gdt = SegmentCache { selector: 0, base: GDT, limit: 0x17, flags: 0 };
        st.idt = SegmentCache { selector: 0, base: IDT, limit: 0x1ff, flags: 0 };
        st.load_seg_cache(R_CS, 0x08, 0, 0xffff_ffff, CODE64);
        for s in [0, 2, 3, 4, 5] {
            st.load_seg_cache(s, 0x10, 0, 0xffff_ffff, DATA32);
        }
        st.regs[R_ESP] = STACK;
        st.mxcsr = 0x1f80;
        st.fpuc = 0x37f;
        st
    }
}

/// The instruction groups the translator decodes but raises #UD for, although the EPYC
/// model advertises them, by opcode map (1 for 0F, 2 for 0F38, 3 for 0F3A, legacy or VEX
/// encoded) and opcode.
fn absent_feature(map: u8, op: u8) -> Option<&'static str> {
    match (map, op) {
        (2, 0xdb..=0xdf) | (3, 0xdf) => Some("AES-NI"),
        (3, 0x60..=0x63) => Some("SSE4.2 PCMPxSTRx"),
        (2, 0x90..=0x93) => Some("AVX2 gathers"),
        (2, 0xc8..=0xcd) | (3, 0xcc) => Some("SHA"),
        _ => None,
    }
}

/// The opcode map and opcode of the instruction starting with `b`, if it is in map 0F, 0F38
/// or 0F3A.
fn opcode(b: &[u8]) -> Option<(u8, u8)> {
    let start = b.iter().position(|&x| !matches!(x, 0x66 | 0xf2 | 0xf3 | 0x40..=0x4f))?;
    match b[start..] {
        [0xc4, m, _, op, ..] => Some((m & 0x1f, op)),
        [0xc5, _, op, ..] => Some((1, op)),
        [0x0f, 0x38, op, ..] => Some((2, op)),
        [0x0f, 0x3a, op, ..] => Some((3, op)),
        [0x0f, op, ..] => Some((1, op)),
        _ => None,
    }
}

/// Differences between the translator and the EPYC the traces found, by trace and the
/// image offset of the compare, which the tests report but do not fail on. They are bugs or
/// QEMU behaviours of `ruvm-target-x86`; take an entry out when it is fixed.
const KNOWN: &[(&str, u64, &str)] = &[
    // CMPXCHG r/m8 with AL as r/m: the compare succeeds, so AL gets the source (r15b);
    // ruvm then writes the old value back to AL.
    ("x86_int", 0x7804, "CMPXCHG AL, r8 writes the old AL over the stored source"),
    // CMPXCHG r/m32 failing with a register r/m: the hardware leaves r/m alone, ruvm writes
    // it back zero extended.
    ("x86_int", 0xa5f4, "a failing CMPXCHG r32, r32 zero extends the destination"),
    // RCPPS, RCPSS, RSQRTPS and RSQRTSS: the hardware returns its 12 bit approximation, ruvm
    // (as QEMU) the exact result; for a denormal input the hardware returns infinity.
    ("x86_sse", 0x250c, "RCPSS approximation"),
    ("x86_sse", 0x2737, "RSQRTPS approximation and a denormal input"),
    ("x86_sse", 0x2e83, "RSQRTSS approximation"),
    ("x86_sse", 0x42a3, "RSQRTSS approximation"),
    ("x86_sse", 0x438a, "RCPPS approximation"),
    ("x86_sse", 0x568d, "RSQRTPS approximation"),
    ("x86_sse", 0x5c82, "RCPSS approximation"),
    ("x86_avx", 0x28dd, "VRSQRTPS approximation"),
    ("x86_avx", 0x4977, "VRCPPS approximation"),
    ("x86_avx", 0x4fc3, "VRCPSS of a denormal input"),
    // Two quiet NaNs: SSE returns the first source operand, ruvm the NaN with the larger
    // significand (the x87 rule).
    ("x86_sse", 0x4470, "MULPD of two NaNs returns the second"),
];

#[derive(Default)]
struct Report {
    /// Ops compared: register and memory compares.
    checks: usize,
    /// Mismatches, one line each.
    mismatches: Vec<String>,
    /// Mismatches listed in `KNOWN`.
    known: Vec<String>,
    /// Instructions the translator leaves out, by group.
    skipped: BTreeMap<&'static str, usize>,
    /// The trace ended before TESTEND, or the replay lost sync with it.
    lost: Option<String>,
}

impl Report {
    /// Report the mismatch `line` at the compare at `pc` of the trace `name`.
    fn differ(&mut self, name: &str, pc: u64, line: String) {
        match KNOWN.iter().find(|k| k.0 == name && k.1 == pc) {
            Some(k) => self.known.push(format!("{line} [known: {}]", k.2)),
            None => self.mismatches.push(line),
        }
    }
}

/// Run `v` until it halts in an exception handler.
fn run_to_halt(v: &mut Vcpu) -> bool {
    for _ in 0..10_000 {
        if cpu_exec(&mut v.cpu()) == excp::HLT {
            return true;
        }
    }
    false
}

/// Replay the trace `trace`, named `name`, of the image `image`.
fn replay(name: &str, image: &[u8], trace: &[u8]) -> Report {
    let recs = parse(trace);
    let w = World::new(image);
    let mut st = w.state();
    let mut v = create_vcpu(&w.jit, w.x86.clone(), w.as_.clone(), &st);
    let mut rep = Report::default();
    let xfeatures = recs.iter().find(|r| is_reg(r.op)).map_or(3, |r| u64_at(r.data, RI_XFEATURES));
    let mut memblock = 0u64;
    // Where the test block being run starts: after the last register op.
    let mut block = 0u64;
    let mut i = 0;
    loop {
        if !run_to_halt(&mut v) {
            rep.lost = Some(format!("the vCPU did not stop in the block at {block:#x}"));
            return rep;
        }
        save_vcpu(&v, &mut st);
        let vector = st.rip.wrapping_sub(HANDLERS + 1) / 16;
        if st.rip <= HANDLERS || vector >= 32 {
            rep.lost = Some(format!("halted at {:#x}, not in an exception handler", st.rip));
            return rep;
        }
        // Back to the trapping instruction, as the kernel does before the signal: pop the
        // interrupt frame (with an error code for #DF, #TS to #PF, #AC and #CP).
        let errcode = matches!(vector, 8 | 10..=14 | 17 | 21);
        let frame = w.read(st.regs[R_ESP] + if errcode { 8 } else { 0 }, 40);
        st.rip = u64_at(&frame, 0);
        st.rflags = u64_at(&frame, 16);
        st.regs[R_ESP] = u64_at(&frame, 24);
        let off = st.rip.wrapping_sub(IMAGE_VA);
        let insn = w.insn(off);
        let op = if vector != UD {
            OP_FAULT
        } else if insn & 0xf8_ffff == 0xc0_b90f {
            ((insn >> 16) & 7) as i32
        } else {
            OP_SIGILL
        };
        let Some(rec) = recs.get(i) else {
            rep.lost = Some(format!("the trace ended before {} at {off:#x}", op_name(op)));
            return rep;
        };
        i += 1;
        let mut resync = false;
        if is_reg(op) && is_reg(rec.op) && off == rec.pc {
            rep.checks += 1;
            let ours = reginfo(&st, off, insn, xfeatures);
            if ours == rec.data {
                if op == OP_TESTEND {
                    return rep;
                }
            } else {
                let line = format!(
                    "{} at {:#x} [{}]:{}",
                    op_name(rec.op),
                    rec.pc,
                    w.bytes(block, rec.pc),
                    diff(&ours, rec.data)
                );
                rep.differ(name, rec.pc, line);
                resync = true;
            }
        } else if op == rec.op && rec.pc == off {
            match op {
                OP_SETMEMBLOCK => memblock = st.regs[R_EAX],
                OP_GETMEMBLOCK => st.regs[R_EAX] = st.regs[R_EAX].wrapping_add(memblock),
                OP_COMPAREMEM => {
                    rep.checks += 1;
                    let ours = w.read(w.phys(memblock), MEMBLOCKLEN);
                    if ours != rec.data {
                        let at = ours.iter().zip(rec.data).position(|(a, b)| a != b).unwrap();
                        let line = format!(
                            "COMPAREMEM at {:#x} [{}]: memory differs from byte {at:#x}: \
                             {:02x?} ({:02x?})",
                            rec.pc,
                            w.bytes(block, rec.pc),
                            &ours[at..(at + 16).min(MEMBLOCKLEN)],
                            &rec.data[at..(at + 16).min(MEMBLOCKLEN)],
                        );
                        rep.differ(name, rec.pc, line);
                        w.write(w.phys(memblock), rec.data);
                    }
                }
                _ => unreachable!(),
            }
        } else {
            // A different op or place: ruvm raised an exception the hardware did not, or
            // ran past an op. An x86 instruction cannot be stepped over without decoding it,
            // so carry on from the hardware's state at the next register compare, taking
            // the memory blocks of the compares on the way.
            let absent = if op == OP_SIGILL && off < w.image_len {
                opcode(&w.read(IMAGE + off, 16)).and_then(|(map, op)| absent_feature(map, op))
            } else {
                None
            };
            if let Some(feature) = absent {
                *rep.skipped.entry(feature).or_default() += 1;
            } else {
                let what = if op == OP_FAULT {
                    format!("ruvm took vector {vector} at {off:#x}")
                } else {
                    format!("ruvm {} at {off:#x}", op_name(op))
                };
                let line = format!(
                    "{} at {:#x} [{}]: {what} [{}]",
                    op_name(rec.op),
                    rec.pc,
                    w.bytes(block, rec.pc),
                    w.bytes(off, off + 8)
                );
                rep.differ(name, rec.pc, line);
            }
            let mut r = rec;
            loop {
                match r.op {
                    OP_COMPAREMEM => w.write(w.phys(memblock), r.data),
                    OP_SETMEMBLOCK => {
                        rep.lost = Some(format!("lost sync at SETMEMBLOCK {:#x}", r.pc));
                        return rep;
                    }
                    _ => {}
                }
                if is_reg(r.op) {
                    break;
                }
                let Some(next) = recs.get(i) else {
                    rep.lost = Some("the trace ended while resyncing".to_string());
                    return rep;
                };
                i += 1;
                r = next;
            }
            restore(&mut st, r.data);
            if r.op == OP_TESTEND {
                return rep;
            }
            st.rip = IMAGE_VA + r.pc;
        }
        if resync {
            // Carry on from the hardware's state after the record's op.
            restore(&mut st, rec.data);
            if rec.op == OP_TESTEND {
                return rep;
            }
            st.rip = IMAGE_VA + rec.pc;
        }
        if is_reg(op) || st.rip != IMAGE_VA + off {
            block = st.rip - IMAGE_VA + 3;
        }
        st.rip += 3;
        env::load_state(&mut v.env, &st);
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
    let rep = replay(name, &image, &trace);
    println!(
        "{name}: {} checks, {} mismatches, {} known differences",
        rep.checks,
        rep.mismatches.len(),
        rep.known.len()
    );
    for m in rep.mismatches.iter().chain(&rep.known) {
        println!("  {m}");
    }
    for (feature, n) in &rep.skipped {
        println!("  {n} instructions of {feature} skipped: not in the translator");
    }
    for k in KNOWN.iter().filter(|k| k.0 == name) {
        if !rep.known.iter().any(|m| m.contains(&format!(" at {:#x} ", k.1))) {
            println!(
                "  the known difference at {:#x} ({}) matches now; drop it from KNOWN",
                k.1, k.2
            );
        }
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

/// The integer group: ALU, shifts and rotates, multiply and divide, bit tests and scans,
/// SETcc and CMOVcc, moves, exchanges, BSWAP and the flag instructions.
#[test]
fn risu_x86_int() {
    check("x86_int");
}

/// BMI1, BMI2, ABM (LZCNT), POPCNT, ADX, MOVBE and CRC32.
#[test]
fn risu_x86_bmi() {
    check("x86_bmi");
}

/// MMX, SSE to SSE4.2, AES and PCLMULQDQ, legacy encoded.
#[test]
fn risu_x86_sse() {
    check("x86_sse");
}

/// AVX and AVX2, with the VEX encoded AES and PCLMULQDQ, comparing all of YMM0 to YMM15.
#[test]
fn risu_x86_avx() {
    check("x86_avx");
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
        .filter(|n| n.starts_with("x86"))
        .collect();
    names.sort();
    names.dedup();
    let mut bad = BTreeMap::new();
    for n in &names {
        let rep = replay_file(&dir, n);
        if rep.lost.is_some() || !rep.mismatches.is_empty() {
            bad.insert(n.clone(), rep.mismatches.len());
        }
    }
    assert!(bad.is_empty(), "traces that differ: {bad:?}");
}
