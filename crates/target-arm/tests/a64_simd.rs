// SPDX-License-Identifier: GPL-2.0-or-later

//! Hand assembled A64 FP and AdvSIMD snippets run through `ruvm-jit` on the interpreter
//! backend.
//!
//! The cases live in `tests/data`. Each one is an instruction word with its inputs and the
//! results it gave when run natively on an Apple M4 (see the header of each file), so the
//! expected values come from hardware and not from this crate's model of it. The test itself
//! runs nothing natively, since this crate's unsafe budget is zero.
//!
//! `a64_simd.txt` holds every form of the generated set (every arrangement and element size
//! that assembles) with up to three inputs each, preferring inputs that raise FPSR flags. The
//! full set the subset was cut from can be checked by pointing `RUVM_A64_SIMD_CASES` at it.
//!
//! The memory map follows `tests/a64.rs`: RAM from 0 to 4 MiB, a WFI in every EL1 vector slot,
//! and each case at its own address from [`CODE`] so that one world runs them all.

use std::sync::Arc;

use ruvm_jit::cpu_exec::cpu_exec;
use ruvm_jit::{Jit, Vcpu, excp};
use ruvm_mem::{AddressSpace, MemTxAttrs, MemorySystem};
use ruvm_target_arm::cpu::{ArmCpuModel, CpuArmState};
use ruvm_target_arm::tcg::{Arm, create_vcpu, new_jit, save_vcpu};

const U: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const RAM_SIZE: u64 = 0x40_0000;
const CODE: u64 = 0x10_0000;
const STACK: u64 = 0x3_0000;
const DATA: u64 = 0x8100;
const WFI: u32 = 0xd503_207f;
/// `msr fpcr, x9`, `msr fpsr, xzr` and `mrs x10, fpsr`.
const MSR_FPCR_X9: u32 = 0xd51b_4409;
const MSR_FPSR_XZR: u32 = 0xd51b_443f;
const MRS_X10_FPSR: u32 = 0xd53b_442a;
/// CPACR_EL1.FPEN = 3: FP and AdvSIMD enabled at EL0 and EL1.
const FPEN: u64 = 3 << 20;

/// V0 to V3 on entry to the load and store cases.
const SIMD_MEM_V: [u128; 4] = [
    0x807f_01ff_7e02_fd03_ff00_8001_7fff_8000,
    0x01ff_7f80_80ff_0102_7f7f_8080_0000_ffff,
    0x8000_ffff_7fff_0001_ffff_0001_8000_fffe,
    0x0000_0001_ffff_ffff_8000_0000_7fff_ffff,
];

struct World {
    _sys: MemorySystem,
    as_: Arc<AddressSpace>,
    jit: Arc<Jit>,
    arm: Arc<Arm>,
}

impl World {
    fn a76() -> World {
        let sys = MemorySystem::new();
        let root = sys.new_container("system", 1 << 64).unwrap();
        let as_ = sys.address_space_init(root, "memory").unwrap();
        let ram = sys.new_ram("ram", RAM_SIZE).unwrap();
        sys.add_subregion(root, 0, ram).unwrap();
        let model = ArmCpuModel::cortex_a76();
        let w = World { _sys: sys, as_, jit: new_jit(), arm: Arc::new(Arm::new(model)) };
        w.code(0, &[WFI; 0x200]);
        w
    }

    fn write(&self, addr: u64, bytes: &[u8]) {
        assert!(self.as_.write(addr, U, bytes).is_ok());
    }

    fn code(&self, addr: u64, words: &[u32]) {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.write(addr, &bytes);
    }

    fn r64(&self, addr: u64) -> u64 {
        let mut b = [0; 8];
        assert!(self.as_.read(addr, U, &mut b).is_ok());
        u64::from_le_bytes(b)
    }

    /// The reset state at EL1 with FP enabled and the PC at `pc`.
    fn state(&self, pc: u64) -> CpuArmState {
        let mut st = CpuArmState::reset(self.arm.model());
        st.pc = pc;
        st.xregs[31] = STACK;
        st.cpacr_el1 = FPEN;
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

fn set_v(st: &mut CpuArmState, n: usize, v: u128) {
    st.zregs[n][0] = v as u64;
    st.zregs[n][1] = (v >> 64) as u64;
}

fn get_v(st: &CpuArmState, n: usize) -> u128 {
    u128::from(st.zregs[n][0]) | (u128::from(st.zregs[n][1]) << 64)
}

fn hex(s: &str) -> u128 {
    u128::from_str_radix(s, 16).unwrap_or_else(|_| panic!("bad hex {s:?}"))
}

/// One register case: the instruction, FPCR, NZCV, V0 to V3 and X0 to X3 on entry, and V0,
/// X0, FPSR and NZCV after.
struct Case {
    line: String,
    insn: u32,
    fpcr: u64,
    nzcv: u32,
    v: [u128; 4],
    x: [u64; 4],
    out_v0: u128,
    out_x0: u64,
    fpsr: u64,
    nzcv_out: u32,
}

/// Parse a case file: `$n = value` alias lines, then one case per line.
fn parse(text: &str) -> Vec<Case> {
    let mut alias = std::collections::HashMap::new();
    let mut cases = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix('$') {
            let (n, v) = rest.split_once(" = ").expect("alias line");
            alias.insert(format!("${n}"), hex(v));
            continue;
        }
        let (fields, _asm) = line.split_once(" ; ").expect("case line");
        let (ins, outs) = fields.split_once(" => ").expect("case line");
        let val = |s: &str| if s.starts_with('$') { alias[s] } else { hex(s) };
        let i: Vec<u128> = ins.split(' ').map(val).collect();
        let o: Vec<u128> = outs.split(' ').map(hex).collect();
        assert_eq!((i.len(), o.len()), (11, 4), "{line}");
        cases.push(Case {
            line: line.to_string(),
            insn: i[0] as u32,
            fpcr: i[1] as u64,
            nzcv: i[2] as u32,
            v: [i[3], i[4], i[5], i[6]],
            x: [i[7] as u64, i[8] as u64, i[9] as u64, i[10] as u64],
            out_v0: o[0],
            out_x0: o[1] as u64,
            fpsr: o[2] as u64,
            nzcv_out: o[3] as u32,
        });
    }
    cases
}

/// Run every case, each at its own address, and report all the mismatches at once.
fn run_cases(cases: &[Case]) {
    let w = World::a76();
    let mut bad = Vec::new();
    for (k, c) in cases.iter().enumerate() {
        let pc = CODE + 32 * k as u64;
        let mut st = w.state(pc);
        for n in 0..4 {
            set_v(&mut st, n, c.v[n]);
        }
        st.xregs[..4].copy_from_slice(&c.x);
        st.xregs[9] = c.fpcr;
        st.set_nzcv(c.nzcv << 28);
        let code = [MSR_FPCR_X9, MSR_FPSR_XZR, c.insn, MRS_X10_FPSR, WFI];
        let st = w.run(st, pc, &code);
        let got = (get_v(&st, 0), st.xregs[0], st.xregs[10], st.nzcv() >> 28);
        let want = (c.out_v0, c.out_x0, c.fpsr, c.nzcv_out);
        if got != want {
            bad.push(format!(
                "{}\n    got  v0 {:x} x0 {:x} fpsr {:x} nzcv {:x}",
                c.line, got.0, got.1, got.2, got.3
            ));
        }
    }
    // Each case's vCPU is gone with its TLB, so a long run does not pile them up.
    assert!(w.jit.cpu_list().is_empty());
    assert!(
        bad.is_empty(),
        "{} of {} cases differ:\n{}",
        bad.len(),
        cases.len(),
        bad.iter().take(60).cloned().collect::<Vec<_>>().join("\n")
    );
}

#[test]
fn fp_and_advsimd_match_hardware() {
    let cases = parse(include_str!("data/a64_simd.txt"));
    assert!(cases.len() > 2000);
    run_cases(&cases);
}

/// The full generated set, when `RUVM_A64_SIMD_CASES` names it.
#[test]
fn full_case_set_from_env() {
    let Ok(path) = std::env::var("RUVM_A64_SIMD_CASES") else {
        return;
    };
    let text = std::fs::read_to_string(path).unwrap();
    run_cases(&parse(&text));
}

#[test]
fn loads_and_stores_match_hardware() {
    let text = include_str!("data/a64_simd_mem.txt");
    let w = World::a76();
    let mut n = 0;
    for (k, line) in text.lines().filter(|l| !l.starts_with('#')).enumerate() {
        let (fields, asm) = line.split_once(" ; ").unwrap();
        let (ins, outs) = fields.split_once(" => ").unwrap();
        let mut ins = ins.split(' ');
        let insn = hex(ins.next().unwrap()) as u32;
        let x2 = hex(ins.next().unwrap()) as u64;
        let outs: Vec<&str> = outs.split(' ').filter(|s| !s.is_empty()).collect();
        // The buffer: byte DATA + i is i mod 256 for i from -128 to 383.
        let bytes: Vec<u8> = (-128i32..384).map(|i| i as u8).collect();
        w.write(DATA - 128, &bytes);
        let pc = CODE + 32 * k as u64;
        let mut st = w.state(pc);
        for (r, v) in SIMD_MEM_V.iter().enumerate() {
            set_v(&mut st, r, *v);
        }
        st.xregs[1] = DATA;
        st.xregs[2] = x2;
        let st = w.run(st, pc, &[insn, WFI]);
        for (r, out) in outs.iter().take(4).enumerate() {
            assert_eq!(get_v(&st, r), hex(out), "{asm}: V{r}");
        }
        let wb: i64 = outs[4].parse().unwrap();
        assert_eq!(st.xregs[1], DATA.wrapping_add_signed(wb), "{asm}: X1");
        let mut changed = std::collections::HashMap::new();
        for m in &outs[5..] {
            let (off, v) = m.split_once(':').unwrap();
            changed.insert(off.parse::<i64>().unwrap(), hex(v) as u64);
        }
        for off in (-128i64..384).step_by(8) {
            let orig = (0..8).rev().fold(0u64, |a, j| (a << 8) | u64::from((off + j) as u8));
            let want = changed.get(&off).copied().unwrap_or(orig);
            assert_eq!(w.r64(DATA.wrapping_add_signed(off)), want, "{asm}: memory at {off}");
        }
        n += 1;
    }
    assert!(n >= 40);
}

#[test]
fn ldr_literal() {
    let w = World::a76();
    // ldr q0, #16; ldr d1, #20; ldr s2, #24; wfi; then the literal pool.
    let code = [
        0x9c00_0080,
        0x5c00_00a1,
        0x1c00_00c2,
        WFI,
        0x0302_0100,
        0x0706_0504,
        0x0b0a_0908,
        0x0f0e_0d0c,
        0x1312_1110,
        0x1716_1514,
    ];
    let mut st = w.state(CODE);
    set_v(&mut st, 1, u128::MAX);
    set_v(&mut st, 2, u128::MAX);
    let st = w.run(st, CODE, &code);
    assert_eq!(get_v(&st, 0), 0x0f0e_0d0c_0b0a_0908_0706_0504_0302_0100);
    assert_eq!(get_v(&st, 1), 0x0f0e_0d0c_0b0a_0908);
    assert_eq!(get_v(&st, 2), 0x1312_1110);
}

/// FADD D0, D1, D2 with CPACR_EL1.FPEN clear traps to EL1 with the Advanced SIMD and FP
/// access syndrome: EC 7, IL, CV and COND 0xe. FPEN = 1 traps EL0 only.
#[test]
fn fp_access_traps() {
    const FADD: u32 = 0x1e62_2820;
    const ESR_FP_TRAP: u64 = 0x1fe0_0000;
    for (fpen, el0, traps) in
        [(0, false, true), (1, false, false), (1, true, true), (3, true, false)]
    {
        let w = World::a76();
        let mut st = w.state(CODE);
        st.cpacr_el1 = fpen << 20;
        if el0 {
            st.pstate_write(0);
            st.sp_el[1] = STACK;
        }
        let st = w.run(st, CODE, &[FADD, WFI]);
        if traps {
            assert_eq!(st.esr_el[1], ESR_FP_TRAP, "fpen {fpen} el0 {el0}");
            assert_eq!(st.elr_el[1], CODE);
            assert_eq!(st.pc, if el0 { 0x404 } else { 0x204 });
        } else {
            assert_eq!(st.pc, CODE + 8, "fpen {fpen} el0 {el0}");
        }
    }
}
