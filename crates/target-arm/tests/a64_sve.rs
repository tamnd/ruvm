// SPDX-License-Identifier: GPL-2.0-or-later

//! SVE and SVE2 instructions run through `ruvm-jit` on the interpreter backend, against
//! results taken from QEMU.
//!
//! The Apple M4 the other tests were generated on has no SVE, so `tests/data/a64_sve.txt` comes
//! from QEMU 11.1 itself: `tests/data/gen_sve.py` runs every instruction word in a bare metal
//! program under `qemu-system-aarch64 -M virt -cpu max,sve-max-vq=N -accel tcg` at EL1 and
//! records what it changed. Each case starts from the state [`init_state`] builds (the same
//! one `gen_sve.py` builds), and the test checks every register the instruction may touch and
//! the whole data buffer, so a write QEMU does not do is caught as well as a wrong value.
//!
//! The memory map follows the virt board where it matters: RAM at 0x4000_0000, the data
//! buffer around [`MID`], so that the addresses held in vector registers are the same as in
//! QEMU, and 16 MiB of it as `-m 16M` gives. Nothing is mapped outside RAM, so an access
//! there is an external abort in both, and the first fault and non fault loads stop there.

use std::sync::Arc;

use ruvm_jit::cpu_exec::cpu_exec;
use ruvm_jit::{Jit, Vcpu, excp};
use ruvm_mem::{AddressSpace, MemTxAttrs, MemorySystem};
use ruvm_target_arm::cpu::{ArmCpuModel, CpuArmState};
use ruvm_target_arm::tcg::{Arm, create_vcpu, new_jit, save_vcpu};

const U: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const RAM: u64 = 0x4000_0000;
const RAM_SIZE: u64 = 0x100_0000;
const CODE: u64 = 0x4020_0000;
const MID: u64 = 0x4030_0000;
const BUF: usize = 1024;
const WFI: u32 = 0xd503_207f;
/// The number of Z registers in the state.
const NZ: usize = 14;
/// `mrs x28, fpsr`, run after the instruction to read the flags it raised.
const MRS_X28_FPSR: u32 = 0xd53b_443c;
/// CPACR_EL1.FPEN = 3 and ZEN = 3: FP, AdvSIMD and SVE enabled at EL0 and EL1.
const FPEN_ZEN: u64 = (3 << 20) | (3 << 16);

struct World {
    _sys: MemorySystem,
    as_: Arc<AddressSpace>,
    jit: Arc<Jit>,
    arm: Arc<Arm>,
}

impl World {
    fn new(model: ArmCpuModel) -> World {
        let sys = MemorySystem::new();
        let root = sys.new_container("system", 1 << 64).unwrap();
        let as_ = sys.address_space_init(root, "memory").unwrap();
        let ram = sys.new_ram("ram", RAM_SIZE).unwrap();
        sys.add_subregion(root, RAM, ram).unwrap();
        let w = World { _sys: sys, as_, jit: new_jit(), arm: Arc::new(Arm::new(model)) };
        w.code(RAM, &[WFI; 0x200]);
        w
    }

    fn max(vq: u32) -> World {
        World::new(ArmCpuModel::max().with_sve_max_vq(vq))
    }

    fn write(&self, addr: u64, bytes: &[u8]) {
        assert!(self.as_.write(addr, U, bytes).is_ok());
    }

    fn read(&self, addr: u64, len: usize) -> Vec<u8> {
        let mut b = vec![0; len];
        assert!(self.as_.read(addr, U, &mut b).is_ok());
        b
    }

    fn code(&self, addr: u64, words: &[u32]) {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.write(addr, &bytes);
    }

    /// The reset state at EL1 with FP and SVE enabled, ZCR_EL1.LEN = `vq` - 1 and the
    /// vectors at the start of RAM.
    fn state(&self, vq: u32) -> CpuArmState {
        let mut st = CpuArmState::reset(self.arm.model());
        st.cpacr_el1 = FPEN_ZEN;
        st.zcr_el[1] = u64::from(vq - 1);
        st.vbar_el[1] = RAM;
        st.xregs[31] = MID + 0x1_0000;
        st
    }

    /// Run `code` at `pc` from `st` until the vCPU halts in WFI.
    fn run(&self, mut st: CpuArmState, pc: u64, code: &[u32]) -> CpuArmState {
        self.code(pc, code);
        st.pc = pc;
        let mut v: Vcpu = create_vcpu(&self.jit, self.arm.clone(), self.as_.clone(), &st);
        for _ in 0..1000 {
            let r = cpu_exec(&mut v.cpu());
            if r == excp::HLT {
                return save_vcpu(&v);
            }
        }
        panic!("the vCPU did not halt");
    }
}

/// The xorshift generator `gen_sve.py` fills the registers from.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// The state every case starts from, `init_state()` in `gen_sve.py`.
struct Init {
    /// Z0 to Z13, `vl / 8` words each.
    z: Vec<Vec<u64>>,
    /// P0 to P7, as `vl / 8` bits.
    p: Vec<u64>,
    x: [u64; 16],
    nzcv: u32,
    mem: Vec<u8>,
}

/// Z0 (but for a first 64-bit lane of 8), Z2, Z3 and Z4 are random, Z1 holds small unsigned
/// byte offsets in its 64-bit lanes, Z5 small signed offsets in its 32-bit lanes, Z6 and Z7 64
/// and 32-bit addresses into the buffer. P1 is all true and the rest random. X8 points to the
/// middle of the 1 KiB buffer and X9 to X15 hold small and edge values, X14 16 bytes before the
/// end of RAM. Z8 to Z13 hold half, single and double precision values from [`FP_H`],
/// [`FP_S`] and [`FP_D`].
fn init_state(vq: usize) -> Init {
    let vl = 16 * vq;
    let mut r = Rng(0x9E37_79B9_7F4A_7C15);
    let pair = |lo: u64, hi: u64| (lo & 0xffff_ffff) | (hi << 32);
    let mut z = Vec::new();
    for n in 0..8 {
        let w: Vec<u64> = match n {
            1 => (0..vl / 8).map(|_| (r.next() % 7) * 8).collect(),
            5 => {
                let lanes: Vec<u64> = (0..vl / 4)
                    .map(|_| (((r.next() % 11) as i64 - 4) * 8) as u64 & 0xffff_ffff)
                    .collect();
                (0..vl / 8).map(|i| pair(lanes[2 * i], lanes[2 * i + 1])).collect()
            }
            6 => (0..vl as u64 / 8).map(|i| MID - 64 + 24 * i).collect(),
            7 => (0..vl as u64 / 8)
                .map(|i| pair(MID - 64 + 24 * i, MID - 64 + 24 * i + 12))
                .collect(),
            _ => {
                let mut w: Vec<u64> = (0..vl / 8).map(|_| r.next()).collect();
                if n == 0 {
                    w[0] = 8;
                }
                w
            }
        };
        z.push(w);
    }
    for (mul, add) in [(3, 0), (5, 1)] {
        z.push(fp_reg(vl, &FP_H.map(u64::from), 2, mul, add));
        z.push(fp_reg(vl, &FP_S.map(u64::from), 4, mul, add));
        z.push(fp_reg(vl, &FP_D, 8, mul, add));
    }
    let pmask = (1u64 << (vl / 8 * 8)) - 1;
    let p = (0..8).map(|n| if n == 1 { pmask } else { r.next() & pmask }).collect();
    let mut x = [0; 16];
    for v in &mut x[..8] {
        *v = r.next();
    }
    x[8..].copy_from_slice(&[
        MID,
        3,
        (-16i64) as u64,
        5,
        17,
        0xffff_fffe,
        RAM + RAM_SIZE - 16,
        0x8000_0000_0000_0003,
    ]);
    let mem = (0..BUF).map(|j| (j * 37 + 11) as u8).collect();
    Init { z, p, x, nzcv: 0xa, mem }
}

/// The floating point values of Z8 to Z13, as in `gen_sve.py`: ones, twos, halves, pi,
/// zeros of both signs, infinities, quiet and signaling NaNs, denormals, the largest finite
/// values, a third and values out of the integer ranges.
const FP_H: [u16; 16] = [
    0x3c00, 0xc000, 0x3800, 0x4248, 0x0000, 0x8000, 0x7c00, 0x7e00, 0x0001, 0x7bff, 0xbe00, 0x5640,
    0x7d00, 0xfc00, 0x3555, 0xd140,
];
const FP_S: [u32; 16] = [
    0x3f80_0000,
    0xc000_0000,
    0x3f00_0000,
    0x4049_0fdb,
    0x0000_0000,
    0x8000_0000,
    0x7f80_0000,
    0x7fc0_0000,
    0x0000_0001,
    0x7f7f_ffff,
    0xbfc0_0000,
    0x42c8_0000,
    0x7fa0_0000,
    0xcf00_0001,
    0x3eaa_aaab,
    0x4f80_0000,
];
const FP_D: [u64; 16] = [
    0x3ff0_0000_0000_0000,
    0xc000_0000_0000_0000,
    0x3fe0_0000_0000_0000,
    0x4009_21fb_5444_2d18,
    0,
    0x8000_0000_0000_0000,
    0x7ff0_0000_0000_0000,
    0x7ff8_0000_0000_0000,
    1,
    0x7fef_ffff_ffff_ffff,
    0xbff8_0000_0000_0000,
    0x4059_0000_0000_0000,
    0x7ff4_0000_0000_0000,
    0xc3e0_0000_0000_0001,
    0x3fd5_5555_5555_5555,
    0x43f0_0000_0000_0000,
];

/// `fp_reg()` of `gen_sve.py`: lane `i` (of `size` bytes) is `table[(i * mul + add) % 16]`.
fn fp_reg(vl: usize, table: &[u64; 16], size: usize, mul: usize, add: usize) -> Vec<u64> {
    let mut bytes = Vec::with_capacity(vl);
    for i in 0..vl / size {
        bytes.extend_from_slice(&table[(i * mul + add) % 16].to_le_bytes()[..size]);
    }
    bytes.chunks(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect()
}

/// A hex number of any length as little endian 64-bit words, `n` of them.
fn words(s: &str, n: usize) -> Vec<u64> {
    assert!(s.len() <= 16 * n, "{s}");
    let s = format!("{s:0>width$}", width = 16 * n);
    (0..n).rev().map(|i| u64::from_str_radix(&s[16 * i..16 * i + 16], 16).unwrap()).collect()
}

fn hex(s: &str) -> u64 {
    u64::from_str_radix(s, 16).unwrap_or_else(|_| panic!("bad hex {s:?}"))
}

/// Run the cases of one vector length; returns the lines that differ.
fn run_vq(vq: usize, lines: &[&str]) -> Vec<String> {
    let vl = 16 * vq;
    let pl = vl / 8;
    let w = World::max(vq as u32);
    let init = init_state(vq);
    let mut bad = Vec::new();
    for (k, line) in lines.iter().enumerate() {
        let (fields, asm) = line.split_once(" ; ").expect("case line");
        let (insn, outs) = fields.split_once(" =>").expect("case line");
        let insn = hex(insn) as u32;

        let mut want_z = init.z.clone();
        let mut want_p = init.p.clone();
        let mut want_ffr = (1u64 << (pl * 8)) - 1;
        let mut want_x = init.x;
        let mut want_nzcv = init.nzcv;
        let mut want_esr = 0;
        let mut want_fpsr = 0;
        let mut want_mem = init.mem.clone();
        for o in outs.split_whitespace() {
            let (name, v) = o.split_once('=').expect("name=value");
            if let Some(off) = name.strip_prefix('m') {
                let off = (off.parse::<i64>().unwrap() + BUF as i64 / 2) as usize;
                want_mem[off..off + 8].copy_from_slice(&hex(v).to_le_bytes());
            } else if name == "ffr" {
                want_ffr = hex(v);
            } else if name == "nzcv" {
                want_nzcv = hex(v) as u32;
            } else if name == "esr" {
                want_esr = hex(v);
            } else if name == "fpsr" {
                want_fpsr = hex(v);
            } else {
                let n: usize = name[1..].parse().unwrap();
                match &name[..1] {
                    "z" => want_z[n] = words(v, vl / 8),
                    "p" => want_p[n] = hex(v),
                    "x" => want_x[n] = hex(v),
                    _ => panic!("bad field {o}"),
                }
            }
        }

        let mut st = w.state(vq as u32);
        for n in 0..NZ {
            st.zregs[n][..vl / 8].copy_from_slice(&init.z[n]);
        }
        for n in 0..8 {
            st.pregs[n][0] = init.p[n];
        }
        st.pregs[16][0] = (1u64 << (pl * 8)) - 1;
        st.xregs[..16].copy_from_slice(&init.x);
        st.set_nzcv(init.nzcv << 28);
        w.write(MID - BUF as u64 / 2, &init.mem);
        let pc = CODE + 16 * k as u64;
        let st = w.run(st, pc, &[insn, MRS_X28_FPSR, WFI]);

        let mut diffs = Vec::new();
        for (n, want) in want_z.iter().enumerate() {
            let z = &st.zregs[n];
            if z[..vl / 8] != want[..] || z[vl / 8..].iter().any(|&v| v != 0) {
                diffs.push(format!("z{n}={:x?}", &z[..vl / 8]));
            }
        }
        for (n, &want) in want_p.iter().enumerate() {
            if st.pregs[n][0] != want {
                diffs.push(format!("p{n}={:x}", st.pregs[n][0]));
            }
        }
        if st.pregs[16][0] != want_ffr {
            diffs.push(format!("ffr={:x}", st.pregs[16][0]));
        }
        for (n, (&got, &want)) in st.xregs[..16].iter().zip(&want_x).enumerate() {
            if got != want {
                diffs.push(format!("x{n}={got:x}"));
            }
        }
        // The QEMU harness returns from an exception, restoring NZCV from SPSR_EL1, where
        // this test stops in the vector with NZCV cleared by the exception entry.
        let nzcv = if want_esr != 0 { (st.spsr_el[1] >> 28) as u32 } else { st.nzcv() >> 28 };
        if nzcv != want_nzcv {
            diffs.push(format!("nzcv={nzcv:x}"));
        }
        if st.esr_el[1] != want_esr {
            diffs.push(format!("esr={:x}", st.esr_el[1]));
        }
        // FPSR as `mrs x28, fpsr` read it after the instruction; an exception skips the
        // read, leaving the zero X28 starts with, as the flags QEMU dumps are then zero.
        if st.xregs[28] != want_fpsr {
            diffs.push(format!("fpsr={:x}", st.xregs[28]));
        }
        let mem = w.read(MID - BUF as u64 / 2, BUF);
        for j in (0..BUF).step_by(8) {
            if mem[j..j + 8] != want_mem[j..j + 8] {
                let v = u64::from_le_bytes(mem[j..j + 8].try_into().unwrap());
                diffs.push(format!("m{}={v:x}", j as i64 - BUF as i64 / 2));
            }
        }
        if !diffs.is_empty() {
            bad.push(format!("vq {vq} {asm}: got {}\n    want {}", diffs.join(" "), outs.trim()));
        }
    }
    bad
}

#[test]
fn sve_matches_qemu() {
    let text = include_str!("data/a64_sve.txt");
    let mut sections: Vec<(usize, Vec<&str>)> = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if let Some(vq) = line.strip_prefix("vq ") {
            sections.push((vq.parse().unwrap(), Vec::new()));
        } else {
            sections.last_mut().expect("a vq line first").1.push(line);
        }
    }
    assert!(sections.len() >= 3);
    let mut bad = Vec::new();
    let mut n = 0;
    for (vq, lines) in &sections {
        n += lines.len();
        bad.extend(run_vq(*vq, lines));
    }
    assert!(n > 3000);
    assert!(
        bad.is_empty(),
        "{} of {n} cases differ:\n{}",
        bad.len(),
        bad.iter().take(60).cloned().collect::<Vec<_>>().join("\n")
    );
}

const RDVL_X0: u32 = 0x04bf_5020;
const MSR_ZCR_EL1_X1: u32 = 0xd518_1201;
const MRS_X2_ZCR_EL1: u32 = 0xd538_1202;
const MRS_X3_ID_AA64ZFR0: u32 = 0xd538_0483;
const MRS_X4_ID_AA64PFR0: u32 = 0xd538_0404;
const ADD_Z0_Z1_Z2: u32 = 0x0422_0020;
const ISB: u32 = 0xd503_3fdf;

/// RDVL reads the vector length that ZCR_EL1.LEN and `sve-max-vq` give, every length up to
/// the maximum being supported as in QEMU's TCG `max`.
#[test]
fn vector_length_from_zcr_and_max_vq() {
    for (max_vq, len, want) in [(4, 0, 16), (4, 2, 48), (4, 15, 64), (3, 15, 48), (16, 15, 256)] {
        let w = World::max(max_vq);
        let mut st = w.state(1);
        st.zcr_el[1] = len;
        let st = w.run(st, CODE, &[RDVL_X0, WFI]);
        assert_eq!(st.xregs[0], want, "sve-max-vq {max_vq} ZCR_EL1.LEN {len}");
    }
}

/// A write to ZCR_EL1 changes the vector length of the following instructions, reads back
/// the LEN field and zeroes the Z register bits above the new length.
#[test]
fn zcr_write_narrows_the_registers() {
    let w = World::max(4);
    let mut st = w.state(4);
    st.zregs[5] = [!0; 32];
    st.xregs[1] = 0xffff_fff0;
    let st = w.run(st, CODE, &[MSR_ZCR_EL1_X1, ISB, MRS_X2_ZCR_EL1, RDVL_X0, WFI]);
    assert_eq!(st.xregs[0], 16);
    assert_eq!(st.xregs[2], 0);
    assert_eq!(st.zregs[5][..2], [!0, !0]);
    assert!(st.zregs[5][2..].iter().all(|&v| v == 0));
}

/// The ID registers of `max` advertise SVE2 and its extensions.
#[test]
fn max_id_registers() {
    let w = World::max(4);
    let st = w.run(w.state(4), CODE, &[MRS_X3_ID_AA64ZFR0, MRS_X4_ID_AA64PFR0, WFI]);
    assert_eq!(st.xregs[3], 0x0110_0101_0001_0021);
    assert_eq!((st.xregs[4] >> 32) & 0xf, 1);
}

/// CPACR_EL1.ZEN traps SVE instructions and ZCR_EL1 with the SVE access syndrome (EC 0x19);
/// with ZEN set but FPEN clear they trap as FP accesses (EC 7). Without SVE the instruction
/// is undefined.
#[test]
fn sve_access_traps() {
    const ESR_SVE_TRAP: u64 = 0x6600_0000;
    const ESR_FP_TRAP: u64 = 0x1fe0_0000;
    const ESR_UNDEF: u64 = 0x0200_0000;
    for (cpacr, insn, want) in [
        (3 << 20, ADD_Z0_Z1_Z2, ESR_SVE_TRAP),
        ((3 << 20) | (1 << 16), ADD_Z0_Z1_Z2, 0),
        (3 << 16, ADD_Z0_Z1_Z2, ESR_FP_TRAP),
        (3 << 20, MSR_ZCR_EL1_X1, ESR_SVE_TRAP),
        (3 << 20, RDVL_X0, ESR_SVE_TRAP),
        (FPEN_ZEN, RDVL_X0, 0),
    ] {
        let w = World::max(2);
        let mut st = w.state(2);
        st.cpacr_el1 = cpacr;
        let st = w.run(st, CODE, &[insn, WFI]);
        assert_eq!(st.esr_el[1], want, "CPACR_EL1 {cpacr:#x} insn {insn:#x}");
        if want != 0 {
            assert_eq!(st.elr_el[1], CODE);
            assert_eq!(st.pc, RAM + 0x204);
        } else {
            assert_eq!(st.pc, CODE + 8);
        }
    }
    let w = World::new(ArmCpuModel::cortex_a76());
    let st = w.run(w.state(1), CODE, &[ADD_Z0_Z1_Z2, WFI]);
    assert_eq!(st.esr_el[1], ESR_UNDEF);
}
