// SPDX-License-Identifier: GPL-2.0-or-later

//! Hand assembled A64 snippets run through `ruvm-jit` on the interpreter backend, checked
//! against the Arm ARM.
//!
//! Every snippet is a list of instruction words with the assembly they encode beside them.
//! The expected results of the user level integer cases in [`ALU`] were taken from running the
//! same instruction words natively on an AArch64 host when the table was written; the test
//! itself runs nothing natively, since this crate's unsafe budget is zero.
//!
//! The memory map of every test: RAM from 0 to 4 MiB, the EL1 vector table at 0 (VBAR_EL1 is
//! 0) with WFI in every slot unless a test puts a handler there, code at 0x1000, data at
//! 0x8000 and the stack below 0x30000. A snippet ends in WFI, which halts the vCPU because no
//! interrupt is pending, and the test then looks at the registers and memory.

use std::sync::Arc;

use ruvm_jit::cpu_exec::cpu_exec;
use ruvm_jit::{Jit, Vcpu, bp, excp};
use ruvm_mem::{AddressSpace, MemTxAttrs, MemorySystem};
use ruvm_target_arm::cpu::{ArmCpuModel, CpuArmState};
use ruvm_target_arm::tcg::{Arm, create_vcpu, new_jit, save_vcpu, vcpu_halted};

const U: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const RAM_SIZE: u64 = 0x40_0000;
const CODE: u64 = 0x1000;
const DATA: u64 = 0x8000;
const STACK: u64 = 0x3_0000;
const WFI: u32 = 0xd503_207f;

/// NZCV values, bits 31 to 28 shifted down to 3 to 0.
const N: u32 = 8;
const Z: u32 = 4;
const C: u32 = 2;
const V: u32 = 1;

/// ESR_EL1 of an Undefined Instruction exception: EC 0 (uncategorized) and IL.
const ESR_UNDEF: u64 = 0x0200_0000;

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
        sys.add_subregion(root, 0, ram).unwrap();
        let w = World { _sys: sys, as_, jit: new_jit(), arm: Arc::new(Arm::new(model)) };
        // Every vector slot halts.
        w.code(0, &[WFI; 0x200]);
        w
    }

    fn a57() -> World {
        World::new(ArmCpuModel::cortex_a57())
    }

    fn a76() -> World {
        World::new(ArmCpuModel::cortex_a76())
    }

    fn write(&self, addr: u64, bytes: &[u8]) {
        assert!(self.as_.write(addr, U, bytes).is_ok());
    }

    fn code(&self, addr: u64, words: &[u32]) {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.write(addr, &bytes);
    }

    fn w64(&self, addr: u64, v: u64) {
        self.write(addr, &v.to_le_bytes());
    }

    fn r64(&self, addr: u64) -> u64 {
        let mut b = [0; 8];
        assert!(self.as_.read(addr, U, &mut b).is_ok());
        u64::from_le_bytes(b)
    }

    /// The reset state of the model with the PC at [`CODE`], VBAR_EL1 at 0 and SP_EL1 at
    /// [`STACK`].
    fn state(&self) -> CpuArmState {
        let mut st = CpuArmState::reset(self.arm.model());
        st.pc = CODE;
        st.xregs[31] = STACK;
        st
    }

    fn vcpu(&self, st: &CpuArmState) -> Vcpu {
        create_vcpu(&self.jit, self.arm.clone(), self.as_.clone(), st)
    }

    /// Run `code` from [`CODE`] in state `st` until the vCPU halts in WFI. The world keeps
    /// the blocks it translated, so each world runs one snippet at [`CODE`].
    fn run(&self, st: &CpuArmState, code: &[u32]) -> CpuArmState {
        self.code(CODE, code);
        let mut v = self.vcpu(st);
        run_to_halt(&mut v);
        save_vcpu(&v)
    }
}

/// Run until the vCPU halts in WFI or stops at a breakpoint, returning the reason.
fn run_to_halt(v: &mut Vcpu) -> i32 {
    for _ in 0..1000 {
        let r = cpu_exec(&mut v.cpu());
        if r == excp::HLT || r == excp::DEBUG {
            return r;
        }
    }
    panic!("the vCPU did not halt");
}

/// Check that an exception from `pc` at EL1 was taken to the current EL, SP_ELx vector.
#[track_caller]
fn assert_sync_el1(st: &CpuArmState, pc: u64, esr: u64) {
    assert_eq!(st.esr_el[1], esr, "ESR_EL1 {:#x}", st.esr_el[1]);
    assert_eq!(st.elr_el[1], pc, "ELR_EL1");
    // Halted in the WFI at VBAR_EL1 + 0x200.
    assert_eq!(st.pc, 0x204);
    assert_eq!(st.current_el(), 1);
}

/// `ConditionHolds()` from the Arm ARM.
fn condition_holds(cond: u32, nzcv: u32) -> bool {
    let (n, z, c, v) = (nzcv & N != 0, nzcv & Z != 0, nzcv & C != 0, nzcv & V != 0);
    let r = match cond >> 1 {
        0 => z,
        1 => c,
        2 => n,
        3 => v,
        4 => c && !z,
        5 => n == v,
        6 => n == v && !z,
        _ => true,
    };
    if cond & 1 != 0 && cond != 15 { !r } else { r }
}

/// One instruction and its operands: X0 to X3 and NZCV on entry, X0 and NZCV after.
struct Case {
    insn: u32,
    asm: &'static str,
    x: [u64; 4],
    nzcv: u32,
    x0: u64,
    nzcv_out: u32,
}

const fn c(insn: u32, asm: &'static str, x: [u64; 4], nzcv: u32, x0: u64, nzcv_out: u32) -> Case {
    Case { insn, asm, x, nzcv, x0, nzcv_out }
}

/// Run one instruction followed by WFI at EL1 from `st`.
fn run_one(w: &World, mut st: CpuArmState, insn: u32, x: [u64; 4], nzcv: u32) -> CpuArmState {
    st.xregs[..4].copy_from_slice(&x);
    st.set_nzcv(nzcv << 28);
    w.run(&st, &[insn, WFI])
}

/// The user level integer instructions: flags edge cases for every flag setting form, then
/// the data processing instructions that leave the flags alone.
static ALU: &[Case] = &[
    c(
        0xab020020,
        "adds x0, x1, x2",
        [0x0, 0x7fffffffffffffff, 0x1, 0x0],
        0x0,
        0x8000000000000000,
        0x9,
    ),
    c(0xab020020, "adds x0, x1, x2", [0x0, 0xffffffffffffffff, 0x1, 0x0], 0x0, 0x0, 0x6),
    c(
        0xab020020,
        "adds x0, x1, x2",
        [0x0, 0x8000000000000000, 0x8000000000000000, 0x0],
        0x0,
        0x0,
        0x7,
    ),
    c(0xab020020, "adds x0, x1, x2", [0x0, 0x1, 0x2, 0x0], 0xf, 0x3, 0x0),
    c(0x2b020020, "adds w0, w1, w2", [0x0, 0x12345678ffffffff, 0x1, 0x0], 0x0, 0x0, 0x6),
    c(0x2b020020, "adds w0, w1, w2", [0x0, 0x7fffffff, 0x1, 0x0], 0x0, 0x80000000, 0x9),
    c(0x2b020020, "adds w0, w1, w2", [0x0, 0xffffffff, 0xffffffff, 0x0], 0x0, 0xfffffffe, 0xa),
    c(0xeb020020, "subs x0, x1, x2", [0x0, 0x0, 0x0, 0x0], 0x0, 0x0, 0x6),
    c(0xeb020020, "subs x0, x1, x2", [0x0, 0x0, 0x1, 0x0], 0x0, 0xffffffffffffffff, 0x8),
    c(
        0xeb020020,
        "subs x0, x1, x2",
        [0x0, 0x8000000000000000, 0x1, 0x0],
        0x0,
        0x7fffffffffffffff,
        0x3,
    ),
    c(0xeb020020, "subs x0, x1, x2", [0x0, 0x5, 0x3, 0x0], 0x0, 0x2, 0x2),
    c(
        0xeb020020,
        "subs x0, x1, x2",
        [0x0, 0x7fffffffffffffff, 0xffffffffffffffff, 0x0],
        0x0,
        0x8000000000000000,
        0x9,
    ),
    c(0x6b020020, "subs w0, w1, w2", [0x0, 0x80000000, 0x1, 0x0], 0x0, 0x7fffffff, 0x3),
    c(0x6b020020, "subs w0, w1, w2", [0x0, 0x1, 0x2, 0x0], 0x0, 0xffffffff, 0x8),
    c(0x6b020020, "subs w0, w1, w2", [0x0, 0xffffffff00000005, 0x5, 0x0], 0x0, 0x0, 0x6),
    c(0xba020020, "adcs x0, x1, x2", [0x0, 0xffffffffffffffff, 0x0, 0x0], 0x2, 0x0, 0x6),
    c(
        0xba020020,
        "adcs x0, x1, x2",
        [0x0, 0xffffffffffffffff, 0x0, 0x0],
        0x0,
        0xffffffffffffffff,
        0x8,
    ),
    c(
        0xba020020,
        "adcs x0, x1, x2",
        [0x0, 0x7fffffffffffffff, 0x0, 0x0],
        0x2,
        0x8000000000000000,
        0x9,
    ),
    c(
        0xba020020,
        "adcs x0, x1, x2",
        [0x0, 0xffffffffffffffff, 0xffffffffffffffff, 0x0],
        0x2,
        0xffffffffffffffff,
        0xa,
    ),
    c(0x3a020020, "adcs w0, w1, w2", [0x0, 0x7fffffff, 0x0, 0x0], 0x2, 0x80000000, 0x9),
    c(0x3a020020, "adcs w0, w1, w2", [0x0, 0xffffffff, 0xffffffff, 0x0], 0x2, 0xffffffff, 0xa),
    c(0x3a020020, "adcs w0, w1, w2", [0x0, 0xffffffff, 0x0, 0x0], 0x2, 0x0, 0x6),
    c(0xfa020020, "sbcs x0, x1, x2", [0x0, 0x5, 0x5, 0x0], 0x0, 0xffffffffffffffff, 0x8),
    c(0xfa020020, "sbcs x0, x1, x2", [0x0, 0x5, 0x5, 0x0], 0x2, 0x0, 0x6),
    c(0xfa020020, "sbcs x0, x1, x2", [0x0, 0x0, 0x0, 0x0], 0x0, 0xffffffffffffffff, 0x8),
    c(
        0xfa020020,
        "sbcs x0, x1, x2",
        [0x0, 0x8000000000000000, 0x0, 0x0],
        0x0,
        0x7fffffffffffffff,
        0x3,
    ),
    c(0x7a020020, "sbcs w0, w1, w2", [0x0, 0x0, 0x0, 0x0], 0x0, 0xffffffff, 0x8),
    c(0x7a020020, "sbcs w0, w1, w2", [0x0, 0x80000000, 0x0, 0x0], 0x2, 0x80000000, 0xa),
    c(0x7a020020, "sbcs w0, w1, w2", [0x0, 0x80000000, 0x0, 0x0], 0x0, 0x7fffffff, 0x3),
    c(
        0xea020020,
        "ands x0, x1, x2",
        [0x0, 0x8000000000000001, 0xffffffffffffffff, 0x0],
        0xf,
        0x8000000000000001,
        0x8,
    ),
    c(0x6a020020, "ands w0, w1, w2", [0x0, 0x100000000, 0xffffffffffffffff, 0x0], 0x3, 0x0, 0x4),
    c(0x6a020020, "ands w0, w1, w2", [0x0, 0x80000000, 0xffffffff, 0x0], 0x7, 0x80000000, 0x8),
    c(0xea220020, "bics x0, x1, x2", [0x0, 0xff, 0xff, 0x0], 0x3, 0x0, 0x4),
    c(0xea220020, "bics x0, x1, x2", [0x0, 0xff, 0xf, 0x0], 0x0, 0xf0, 0x0),
    c(0xb13ffc20, "adds x0, x1, #0xfff", [0x0, 0xfffffffffffff001, 0x0, 0x0], 0x0, 0x0, 0x6),
    c(0xb13ffc20, "adds x0, x1, #0xfff", [0x0, 0x0, 0x0, 0x0], 0xf, 0xfff, 0x0),
    c(0x71400420, "subs w0, w1, #0x1, lsl #12", [0x0, 0x1000, 0x0, 0x0], 0x0, 0x0, 0x6),
    c(0x71400420, "subs w0, w1, #0x1, lsl #12", [0x0, 0x0, 0x0, 0x0], 0x0, 0xfffff000, 0x8),
    c(
        0xf2089c20,
        "ands x0, x1, #0xff00ff00ff00ff00",
        [0x0, 0xffffffffffffffff, 0x0, 0x0],
        0x0,
        0xff00ff00ff00ff00,
        0x8,
    ),
    c(
        0xf2089c20,
        "ands x0, x1, #0xff00ff00ff00ff00",
        [0x0, 0xff00ff00ff00ff, 0x0, 0x0],
        0xb,
        0x0,
        0x4,
    ),
    c(0xab22c820, "adds x0, x1, w2, sxtw #2", [0x0, 0x100, 0xffffffff, 0x0], 0x0, 0xfc, 0x2),
    c(
        0xab22c820,
        "adds x0, x1, w2, sxtw #2",
        [0x0, 0x100, 0x1234567800000004, 0x0],
        0x0,
        0x110,
        0x0,
    ),
    c(
        0xeb820c20,
        "subs x0, x1, x2, asr #3",
        [0x0, 0x0, 0x8000000000000000, 0x0],
        0x0,
        0x1000000000000000,
        0x0,
    ),
    c(0xeb820c20, "subs x0, x1, x2, asr #3", [0x0, 0x1, 0x8, 0x0], 0x0, 0x0, 0x6),
    c(0xfa420025, "ccmp x1, x2, #0x5, eq", [0x0, 0x3, 0x3, 0x0], 0x4, 0x0, 0x6),
    c(0xfa420025, "ccmp x1, x2, #0x5, eq", [0x0, 0x3, 0x3, 0x0], 0x0, 0x0, 0x5),
    c(0xfa420025, "ccmp x1, x2, #0x5, eq", [0x0, 0x3, 0x4, 0x0], 0x4, 0x0, 0x8),
    c(0x3a431828, "ccmn w1, #0x3, #0x8, ne", [0x0, 0xfffffffd, 0x0, 0x0], 0x0, 0x0, 0x6),
    c(0x3a431828, "ccmn w1, #0x3, #0x8, ne", [0x0, 0xfffffffd, 0x0, 0x0], 0x4, 0x0, 0x8),
    c(0x3a431828, "ccmn w1, #0x3, #0x8, ne", [0x0, 0x7ffffffd, 0x0, 0x0], 0x0, 0x0, 0x9),
    c(0x7201003f, "tst w1, #0x80000000", [0x0, 0x80000000, 0x0, 0x0], 0xf, 0x0, 0x8),
    c(0x7201003f, "tst w1, #0x80000000", [0x0, 0x7fffffff, 0x0, 0x0], 0xb, 0x0, 0x4),
    c(0xeb0103e0, "negs x0, x1", [0x0, 0x8000000000000000, 0x0, 0x0], 0x0, 0x8000000000000000, 0x9),
    c(0xeb0103e0, "negs x0, x1", [0x0, 0x0, 0x0, 0x0], 0x0, 0x0, 0x6),
    c(0xeb0103e0, "negs x0, x1", [0x0, 0x1, 0x0, 0x0], 0x0, 0xffffffffffffffff, 0x8),
    c(0x7a0103e0, "ngcs w0, w1", [0x0, 0x0, 0x0, 0x0], 0x2, 0x0, 0x6),
    c(0x7a0103e0, "ngcs w0, w1", [0x0, 0x0, 0x0, 0x0], 0x0, 0xffffffff, 0x8),
    c(0x9a020020, "adc x0, x1, x2", [0x0, 0x1, 0x2, 0x0], 0xa, 0x4, 0xa),
    c(0x5a020020, "sbc w0, w1, w2", [0x0, 0x5, 0x3, 0x0], 0x4, 0x1, 0x4),
    c(0x0b027c20, "add w0, w1, w2, lsl #31", [0x0, 0x1, 0x1, 0x0], 0x0, 0x80000001, 0x0),
    c(0x9a820020, "csel x0, x1, x2, eq", [0x0, 0x1, 0x2, 0x0], 0x4, 0x1, 0x4),
    c(0x9a820020, "csel x0, x1, x2, eq", [0x0, 0x1, 0x2, 0x0], 0x0, 0x2, 0x0),
    c(0x1a82a420, "csinc w0, w1, w2, ge", [0x0, 0x100000005, 0xffffffff, 0x0], 0x9, 0x5, 0x9),
    c(0x1a82a420, "csinc w0, w1, w2, ge", [0x0, 0x100000005, 0xffffffff, 0x0], 0x8, 0x0, 0x8),
    c(0xda82b020, "csinv x0, x1, x2, lt", [0x0, 0x7, 0x10, 0x0], 0x8, 0x7, 0x8),
    c(0xda82b020, "csinv x0, x1, x2, lt", [0x0, 0x7, 0x10, 0x0], 0x0, 0xffffffffffffffef, 0x0),
    c(0x5a828420, "csneg w0, w1, w2, hi", [0x0, 0x7, 0x1, 0x0], 0x2, 0x7, 0x2),
    c(0x5a828420, "csneg w0, w1, w2, hi", [0x0, 0x7, 0x1, 0x0], 0x6, 0xffffffff, 0x6),
    c(0x9a9f07e0, "cset x0, ne", [0x0, 0x0, 0x0, 0x0], 0x0, 0x1, 0x0),
    c(0x9a9f07e0, "cset x0, ne", [0x0, 0x0, 0x0, 0x0], 0x4, 0x0, 0x4),
    c(0x5a9f33e0, "csetm w0, hs", [0x0, 0x0, 0x0, 0x0], 0x2, 0xffffffff, 0x2),
    c(0x5a9f33e0, "csetm w0, hs", [0x0, 0x0, 0x0, 0x0], 0x0, 0x0, 0x0),
    c(0xd3485c20, "ubfx x0, x1, #8, #16", [0x0, 0x123456789abcdef0, 0x0, 0x0], 0x0, 0xbcde, 0x0),
    c(0x13042c20, "sbfx w0, w1, #4, #8", [0x0, 0xf80, 0x0, 0x0], 0x0, 0xfffffff8, 0x0),
    c(0x13042c20, "sbfx w0, w1, #4, #8", [0x0, 0x7f0, 0x0, 0x0], 0x0, 0x7f, 0x0),
    c(
        0xb3440c20,
        "bfi x0, x1, #60, #4",
        [0x123456789abcdef0, 0xa5, 0x0, 0x0],
        0x0,
        0x523456789abcdef0,
        0x0,
    ),
    c(
        0x33187c20,
        "bfxil w0, w1, #24, #8",
        [0xffffffffffffffff, 0xab000000, 0x0, 0x0],
        0x0,
        0xffffffab,
        0x0,
    ),
    c(0x53010020, "lsl w0, w1, #31", [0x0, 0x3, 0x0, 0x0], 0x0, 0x80000000, 0x0),
    c(0xd37ffc20, "lsr x0, x1, #63", [0x0, 0x8000000000000000, 0x0, 0x0], 0x0, 0x1, 0x0),
    c(0x13017c20, "asr w0, w1, #1", [0x0, 0x80000000, 0x0, 0x0], 0x0, 0xc0000000, 0x0),
    c(0x93401c20, "sxtb x0, w1", [0x0, 0x180, 0x0, 0x0], 0x0, 0xffffffffffffff80, 0x0),
    c(0x53003c20, "uxth w0, w1", [0x0, 0x12345678, 0x0, 0x0], 0x0, 0x5678, 0x0),
    c(0x93407c20, "sxtw x0, w1", [0x0, 0x80000000, 0x0, 0x0], 0x0, 0xffffffff80000000, 0x0),
    c(0x93760c20, "sbfiz x0, x1, #10, #4", [0x0, 0xf, 0x0, 0x0], 0x0, 0xfffffffffffffc00, 0x0),
    c(0x93c21020, "extr x0, x1, x2, #0x4", [0x0, 0x1, 0xf0, 0x0], 0x0, 0x100000000000000f, 0x0),
    c(0x13827c20, "extr w0, w1, w2, #0x1f", [0x0, 0x1, 0x80000000, 0x0], 0x0, 0x3, 0x0),
    c(0x93c10420, "ror x0, x1, #0x1", [0x0, 0x3, 0x0, 0x0], 0x0, 0x8000000000000001, 0x0),
    c(0x1ac22020, "lsl w0, w1, w2", [0x0, 0x1, 0x21, 0x0], 0x0, 0x2, 0x0),
    c(
        0x9ac22420,
        "lsr x0, x1, x2",
        [0x0, 0xffffffffffffffff, 0x41, 0x0],
        0x0,
        0x7fffffffffffffff,
        0x0,
    ),
    c(
        0x9ac22820,
        "asr x0, x1, x2",
        [0x0, 0x8000000000000000, 0x3f, 0x0],
        0x0,
        0xffffffffffffffff,
        0x0,
    ),
    c(0x1ac22c20, "ror w0, w1, w2", [0x0, 0x1, 0x1, 0x0], 0x0, 0x80000000, 0x0),
    c(0x1ac22c20, "ror w0, w1, w2", [0x0, 0x12345678, 0x24, 0x0], 0x0, 0x81234567, 0x0),
    c(
        0x9b027c20,
        "mul x0, x1, x2",
        [0x0, 0x3, 0xfffffffffffffffe, 0x0],
        0x0,
        0xfffffffffffffffa,
        0x0,
    ),
    c(0x1b020c20, "madd w0, w1, w2, w3", [0x0, 0x10000, 0x10000, 0x5], 0x0, 0x5, 0x0),
    c(0x9b028c20, "msub x0, x1, x2, x3", [0x0, 0x3, 0x4, 0xa], 0x0, 0xfffffffffffffffe, 0x0),
    c(0x9b220c20, "smaddl x0, w1, w2, x3", [0x0, 0xfffffffe, 0x3, 0xa], 0x0, 0x4, 0x0),
    c(0x9ba28c20, "umsubl x0, w1, w2, x3", [0x0, 0xffffffff, 0x2, 0x200000000], 0x0, 0x2, 0x0),
    c(
        0x9b427c20,
        "smulh x0, x1, x2",
        [0x0, 0xffffffffffffffff, 0x2, 0x0],
        0x0,
        0xffffffffffffffff,
        0x0,
    ),
    c(
        0x9b427c20,
        "smulh x0, x1, x2",
        [0x0, 0x8000000000000000, 0x8000000000000000, 0x0],
        0x0,
        0x4000000000000000,
        0x0,
    ),
    c(
        0x9bc27c20,
        "umulh x0, x1, x2",
        [0x0, 0xffffffffffffffff, 0xffffffffffffffff, 0x0],
        0x0,
        0xfffffffffffffffe,
        0x0,
    ),
    c(0x9ac20820, "udiv x0, x1, x2", [0x0, 0xa, 0x0, 0x0], 0x0, 0x0, 0x0),
    c(
        0x9ac20820,
        "udiv x0, x1, x2",
        [0x0, 0xffffffffffffffff, 0x3, 0x0],
        0x0,
        0x5555555555555555,
        0x0,
    ),
    c(
        0x9ac20c20,
        "sdiv x0, x1, x2",
        [0x0, 0x8000000000000000, 0xffffffffffffffff, 0x0],
        0x0,
        0x8000000000000000,
        0x0,
    ),
    c(
        0x9ac20c20,
        "sdiv x0, x1, x2",
        [0x0, 0xfffffffffffffff9, 0x2, 0x0],
        0x0,
        0xfffffffffffffffd,
        0x0,
    ),
    c(0x9ac20c20, "sdiv x0, x1, x2", [0x0, 0x5, 0x0, 0x0], 0x0, 0x0, 0x0),
    c(0x1ac20c20, "sdiv w0, w1, w2", [0x0, 0x80000000, 0xffffffff, 0x0], 0x0, 0x80000000, 0x0),
    c(0x1ac20c20, "sdiv w0, w1, w2", [0x0, 0xfffffff9, 0x2, 0x0], 0x0, 0xfffffffd, 0x0),
    c(0x1ac20820, "udiv w0, w1, w2", [0x0, 0x1ffffffff, 0x2, 0x0], 0x0, 0x7fffffff, 0x0),
    c(0xdac00020, "rbit x0, x1", [0x0, 0x1, 0x0, 0x0], 0x0, 0x8000000000000000, 0x0),
    c(0x5ac00020, "rbit w0, w1", [0x0, 0x100000001, 0x0, 0x0], 0x0, 0x80000000, 0x0),
    c(0xdac00c20, "rev x0, x1", [0x0, 0x1122334455667788, 0x0, 0x0], 0x0, 0x8877665544332211, 0x0),
    c(0x5ac00820, "rev w0, w1", [0x0, 0x1122334455667788, 0x0, 0x0], 0x0, 0x88776655, 0x0),
    c(
        0xdac00420,
        "rev16 x0, x1",
        [0x0, 0x1122334455667788, 0x0, 0x0],
        0x0,
        0x2211443366558877,
        0x0,
    ),
    c(0x5ac00420, "rev16 w0, w1", [0x0, 0x1122334455667788, 0x0, 0x0], 0x0, 0x66558877, 0x0),
    c(
        0xdac00820,
        "rev32 x0, x1",
        [0x0, 0x1122334455667788, 0x0, 0x0],
        0x0,
        0x4433221188776655,
        0x0,
    ),
    c(0xdac01020, "clz x0, x1", [0x0, 0x0, 0x0, 0x0], 0x0, 0x40, 0x0),
    c(0xdac01020, "clz x0, x1", [0x0, 0x10000, 0x0, 0x0], 0x0, 0x2f, 0x0),
    c(0x5ac01020, "clz w0, w1", [0x0, 0x100010000, 0x0, 0x0], 0x0, 0xf, 0x0),
    c(0x5ac01020, "clz w0, w1", [0x0, 0x100000000, 0x0, 0x0], 0x0, 0x20, 0x0),
    c(0xdac01420, "cls x0, x1", [0x0, 0x0, 0x0, 0x0], 0x0, 0x3f, 0x0),
    c(0xdac01420, "cls x0, x1", [0x0, 0xff00000000000000, 0x0, 0x0], 0x0, 0x7, 0x0),
    c(0x5ac01420, "cls w0, w1", [0x0, 0xffffffff, 0x0, 0x0], 0x0, 0x1f, 0x0),
    c(0x5ac01420, "cls w0, w1", [0x0, 0x1, 0x0, 0x0], 0x0, 0x1e, 0x0),
    c(
        0xd2e24680,
        "mov x0, #0x1234000000000000",
        [0x5, 0x0, 0x0, 0x0],
        0x0,
        0x1234000000000000,
        0x0,
    ),
    c(0x12a24680, "mov w0, #-0x12340001", [0x5, 0x0, 0x0, 0x0], 0x0, 0xedcbffff, 0x0),
    c(0x92800000, "mov x0, #-0x1", [0x5, 0x0, 0x0, 0x0], 0x0, 0xffffffffffffffff, 0x0),
    c(
        0xf2b7dde0,
        "movk x0, #0xbeef, lsl #16",
        [0xffffffffffffffff, 0x0, 0x0, 0x0],
        0x0,
        0xffffffffbeefffff,
        0x0,
    ),
    c(
        0x72b7dde0,
        "movk w0, #0xbeef, lsl #16",
        [0xffffffffffffffff, 0x0, 0x0, 0x0],
        0x0,
        0xbeefffff,
        0x0,
    ),
    c(
        0xb200f3e0,
        "mov x0, #0x5555555555555555",
        [0x0, 0x0, 0x0, 0x0],
        0x0,
        0x5555555555555555,
        0x0,
    ),
    c(0x52001c20, "eor w0, w1, #0xff", [0x0, 0xffffffff00001234, 0x0, 0x0], 0x0, 0x12cb, 0x0),
    c(0xaa221020, "orn x0, x1, x2, lsl #4", [0x0, 0x1, 0xf, 0x0], 0x0, 0xffffffffffffff0f, 0x0),
    c(0x4a220020, "eon w0, w1, w2", [0x0, 0xf0f0f0f0, 0xff00ff00, 0x0], 0x0, 0xf00ff00f, 0x0),
    c(0x1ac24020, "crc32b w0, w1, w2", [0x0, 0xffffffff, 0x61, 0x0], 0x0, 0x174841bc, 0x0),
    c(0x1ac24420, "crc32h w0, w1, w2", [0x0, 0x12345678, 0xabcd1234, 0x0], 0x0, 0x2d7d97b4, 0x0),
    c(0x1ac24820, "crc32w w0, w1, w2", [0x0, 0x0, 0xdeadbeef, 0x0], 0x0, 0x3b1ebf03, 0x0),
    c(
        0x9ac24c20,
        "crc32x w0, w1, x2",
        [0x0, 0xffffffff, 0x123456789abcdef, 0x0],
        0x0,
        0xbbc41db8,
        0x0,
    ),
    c(0x1ac25020, "crc32cb w0, w1, w2", [0x0, 0xffffffff, 0x61, 0x0], 0x0, 0x3e2fbccf, 0x0),
    c(0x1ac25420, "crc32ch w0, w1, w2", [0x0, 0x12345678, 0xabcd1234, 0x0], 0x0, 0xaa68fcf7, 0x0),
    c(0x1ac25820, "crc32cw w0, w1, w2", [0x0, 0x0, 0xdeadbeef, 0x0], 0x0, 0x9991d14, 0x0),
    c(
        0x9ac25c20,
        "crc32cx w0, w1, x2",
        [0x0, 0xffffffff, 0x123456789abcdef, 0x0],
        0x0,
        0x9a4f27dc,
        0x0,
    ),
    c(0xd53b4200, "mrs x0, NZCV", [0x0, 0x0, 0x0, 0x0], 0x9, 0x90000000, 0x9),
    c(0xd51b4201, "msr NZCV, x1", [0x0, 0x60000000, 0x0, 0x0], 0x9, 0x0, 0x6),
];

#[test]
fn alu_matches_the_arm_arm() {
    for case in ALU {
        let w = World::a76();
        let st = run_one(&w, w.state(), case.insn, case.x, case.nzcv);
        assert_eq!(st.pc, CODE + 8, "{}: did not run to the WFI", case.asm);
        assert_eq!(st.xregs[0], case.x0, "{} with {:#x?}: X0", case.asm, case.x);
        assert_eq!(st.nzcv() >> 28, case.nzcv_out, "{} with {:#x?}: NZCV", case.asm, case.x);
    }
}

#[test]
fn every_condition_against_every_flag_combination() {
    for cond in 0..16 {
        for nzcv in 0..16 {
            // b.<cond> .+12; mov x0, #0; wfi; mov x0, #1; wfi
            let w = World::a57();
            let mut st = w.state();
            st.set_nzcv(nzcv << 28);
            st.xregs[0] = 7;
            let st = w.run(&st, &[0x5400_0060 | cond, 0xd280_0000, WFI, 0xd280_0020, WFI]);
            let taken = st.xregs[0] == 1;
            assert_eq!(taken, condition_holds(cond, nzcv), "b.cond {cond} with nzcv {nzcv:#x}");

            // csel x0, x1, x2, <cond>
            let w = World::a57();
            let insn = 0x9a82_0020 | (cond << 12);
            let st = run_one(&w, w.state(), insn, [0, 1, 2, 0], nzcv);
            let want = if condition_holds(cond, nzcv) { 1 } else { 2 };
            assert_eq!(st.xregs[0], want, "csel cond {cond} with nzcv {nzcv:#x}");
            assert_eq!(st.nzcv() >> 28, nzcv);
        }
    }
}

#[test]
fn sp_forms() {
    let w = World::a57();
    // cmp sp, x2 (subs xzr, sp, x2, uxtx): the flag setting form writes XZR, not SP.
    let mut st = w.state();
    st.xregs[31] = 0x100;
    let st = run_one(&w, st, 0xeb22_63ff, [0, 0, 0x100, 0], 0);
    assert_eq!(st.nzcv() >> 28, Z | C);
    assert_eq!(st.xregs[31], 0x100);
    // sub x0, sp, #16
    let w = World::a57();
    let st = run_one(&w, w.state(), 0xd100_43e0, [0; 4], 0);
    assert_eq!(st.xregs[0], STACK - 16);
    // mov sp, x1 (add sp, x1, #0)
    let w = World::a57();
    let st = run_one(&w, w.state(), 0x9100_003f, [0, 0x1234_5670, 0, 0], 0);
    assert_eq!(st.xregs[31], 0x1234_5670);
}

#[test]
fn pc_relative() {
    let w = World::a57();
    // adr x0, .+0x40
    let st = run_one(&w, w.state(), 0x1000_0200, [0; 4], 0);
    assert_eq!(st.xregs[0], CODE + 0x40);
    // adrp x0, .+0x5000 (immlo 1, immhi 1)
    let w = World::a57();
    let mut st = w.state();
    st.pc = 0x1234 & !3;
    w.code(st.pc, &[0xb000_0020, WFI]);
    let mut v = w.vcpu(&st);
    run_to_halt(&mut v);
    assert_eq!(save_vcpu(&v).xregs[0], 0x6000);
}

#[test]
fn undefined_encodings() {
    let undef: &[(u32, &str)] = &[
        (0x8bc2_2020, "add x0, x1, x2, ror #8 (reserved shift)"),
        (0x1e62_2820, "fadd d0, d1, d2 (FP is the second slice)"),
        (0xd400_0002, "hvc #0 (no EL2)"),
        (0xd400_0003, "smc #0 (no EL3)"),
        (0xd440_0000, "hlt #0 (no semihosting)"),
        (0x0000_0001, "udf #1"),
        (0xd538_f000, "mrs x0, s3_0_c15_c0_0 (IMPLEMENTATION DEFINED)"),
        (0xd69f_03e0, "eret at EL0"),
    ];
    for &(insn, asm) in undef {
        let w = World::a76();
        let mut st = w.state();
        if asm.ends_with("at EL0") {
            st.pstate_write(0);
            st.sp_el[1] = STACK;
        }
        let st = run_one(&w, st, insn, [0; 4], 0);
        assert_eq!(st.esr_el[1], ESR_UNDEF, "{asm}");
        assert_eq!(st.elr_el[1], CODE, "{asm}");
        let vector = if asm.ends_with("at EL0") { 0x404 } else { 0x204 };
        assert_eq!(st.pc, vector, "{asm}");
    }
}

#[test]
fn lse_is_undefined_without_feat_lse() {
    // ldadd x4, x5, [x2] on a cortex-a57, which is ARMv8.0.
    let w = World::a57();
    let st = run_one(&w, w.state(), 0xf824_0045, [0, 0, DATA, 0], 0);
    assert_sync_el1(&st, CODE, ESR_UNDEF);
}

#[test]
fn brk_reports_its_immediate() {
    let w = World::a57();
    // brk #7: EC 0x3c, IL, the immediate in the ISS. The preferred return is the BRK.
    let st = run_one(&w, w.state(), 0xd420_00e0, [0; 4], 0);
    assert_sync_el1(&st, CODE, 0xf200_0007);
}

#[test]
fn el0_cannot_read_el1_registers() {
    let w = World::a57();
    let mut st = w.state();
    st.pstate_write(0);
    st.sp_el[1] = STACK;
    // mrs x0, sctlr_el1 at EL0
    let st = run_one(&w, st, 0xd538_1000, [0; 4], 0);
    assert_eq!(st.esr_el[1], ESR_UNDEF);
    assert_eq!(st.elr_el[1], CODE);
    assert_eq!(st.pc, 0x404);
    assert_eq!(st.spsr_el[1], 0);
}

#[test]
fn system_registers() {
    let w = World::a76();
    // mrs x0, currentel
    let st = run_one(&w, w.state(), 0xd538_4240, [0; 4], 0);
    assert_eq!(st.xregs[0], 4);
    // mrs x0, midr_el1
    let w = World::a76();
    let st = run_one(&w, w.state(), 0xd538_0000, [0; 4], 0);
    assert_eq!(st.xregs[0], 0x414f_d0b1);
    // mrs x0, ctr_el0
    let w = World::a76();
    let st = run_one(&w, w.state(), 0xd53b_0020, [0; 4], 0);
    assert_eq!(st.xregs[0], 0x8444_c004);
    // msr tpidr_el0, x1; mrs x0, tpidr_el0
    let w = World::a76();
    let mut st = w.state();
    st.xregs[1] = 0xfeed_f00d;
    let st = w.run(&st, &[0xd51b_d041, 0xd53b_d040, 0xd503_201f, 0xd503_203f, WFI]);
    assert_eq!(st.xregs[0], 0xfeed_f00d);
    assert_eq!(st.tpidr_el[0], 0xfeed_f00d);
    assert_eq!(st.pc, CODE + 20);
}

#[test]
fn branches() {
    let w = World::a57();
    let code = [
        0xd280_0000, // 0x00: mov x0, #0
        0xd280_00a1, // 0x04: mov x1, #5
        0x9100_0800, // 0x08: 1: add x0, x0, #2
        0xf100_0421, // 0x0c: subs x1, x1, #1
        0x54ff_ffc1, // 0x10: b.ne 1b
        0xb400_0041, // 0x14: cbz x1, 2f
        0xd280_7ce0, // 0x18: mov x0, #999
        0x9400_000f, // 0x1c: 2: bl 9f
        0x3700_0040, // 0x20: tbnz w0, #0, 3f
        0xd280_0025, // 0x24: mov x5, #1
        0x3608_0140, // 0x28: 3: tbz w0, #1, 8f
        0xb500_0121, // 0x2c: cbnz x1, 8f
        0x3400_0047, // 0x30: cbz w7, 4f
        0xd280_0045, // 0x34: mov x5, #2
        0x1000_0062, // 0x38: 4: adr x2, 5f
        0xd61f_0040, // 0x3c: br x2
        0xd280_0065, // 0x40: mov x5, #3
        0x1000_00e3, // 0x44: 5: adr x3, 10f
        0xd63f_0060, // 0x48: blr x3
        0x1400_0002, // 0x4c: b 6f
        0xd280_0085, // 0x50: 8: mov x5, #4
        WFI,         // 0x54: 6: wfi
        0x9100_0400, // 0x58: 9: add x0, x0, #1
        0xd65f_03c0, // 0x5c: ret
        0xaa1e_03e6, // 0x60: 10: mov x6, x30
        0xd65f_03c0, // 0x64: ret
    ];
    let mut st = w.state();
    // W7 is zero even though X7 is not.
    st.xregs[7] = 0x1_0000_0000;
    let st = w.run(&st, &code);
    assert_eq!(st.xregs[0], 11);
    assert_eq!(st.xregs[1], 0);
    assert_eq!(st.xregs[5], 0, "a branch went the wrong way");
    assert_eq!(st.xregs[6], CODE + 0x4c);
    assert_eq!(st.xregs[30], CODE + 0x4c);
    assert_eq!(st.pc, CODE + 0x58);
}

#[test]
fn loads_and_stores() {
    let w = World::a57();
    let x1: u64 = 0x1122_3344_5566_7788;
    let code = [
        0xf900_0041, // str x1, [x2]
        0xf800_8c41, // str x1, [x2, #8]!
        0xf85f_8443, // ldr x3, [x2], #-8
        0x3900_4041, // strb w1, [x2, #16]
        0x7900_2441, // strh w1, [x2, #18]
        0xb900_1441, // str w1, [x2, #20]
        0x3980_4044, // ldrsb x4, [x2, #16]
        0x39c0_4045, // ldrsb w5, [x2, #16]
        0x7980_2446, // ldrsh x6, [x2, #18]
        0xb980_0447, // ldrsw x7, [x2, #4]
        0x7940_0c48, // ldrh w8, [x2, #6]
        0xf81f_f141, // stur x1, [x10, #-1]
        0xf85f_f14b, // ldur x11, [x10, #-1]
        0xd280_004c, // mov x12, #2
        0xf86c_784d, // ldr x13, [x2, x12, lsl #3]
        0x1280_000e, // mov w14, #-1
        0xb86e_d84f, // ldr w15, [x2, w14, sxtw #2]
        0x386c_6850, // ldrb w16, [x2, x12]
        0xa9bf_0be1, // stp x1, x2, [sp, #-16]!
        0xa8c1_4bf1, // ldp x17, x18, [sp], #16
        0x2904_1441, // stp w1, w5, [x2, #32]
        0x6944_5053, // ldpsw x19, x20, [x2, #32]
        0x2944_5855, // ldp w21, w22, [x2, #32]
        0x5800_0157, // ldr x23, 1f
        0x9800_0178, // ldrsw x24, 2f
        0xf840_0859, // ldtr x25, [x2]
        0x3804_0841, // sttrb w1, [x2, #64]
        0xf840_305a, // ldur x26, [x2, #3]
        0xa805_2841, // stnp x1, x10, [x2, #80]
        0xa945_705b, // ldp x27, x28, [x2, #80]
        0xf81f_8c41, // str x1, [x2, #-8]!
        0x78c0_8c5d, // ldrsh w29, [x2, #8]!
        WFI,         // wfi
        0x1234_5678, // 1: .quad 0xcafebabe12345678
        0xcafe_babe,
        0x8000_0000, // 2: .word 0x80000000
    ];
    w.w64(DATA - 8, 0xdead_beef_0000_0000);
    let mut st = w.state();
    st.xregs[1] = x1;
    st.xregs[2] = DATA;
    st.xregs[10] = DATA + 0x100;
    let st = w.run(&st, &code);
    let x = &st.xregs;

    assert_eq!(w.r64(DATA), x1);
    assert_eq!(w.r64(DATA + 8), x1);
    assert_eq!(x[3], x1, "post index load");
    // DATA + 16: 88 xx 88 77 88 77 66 55
    assert_eq!(w.r64(DATA + 16), 0x5566_7788_7788_0088);
    assert_eq!(x[4], 0xffff_ffff_ffff_ff88, "ldrsb x");
    assert_eq!(x[5], 0xffff_ff88, "ldrsb w zero extends to 64 bits");
    assert_eq!(x[6], 0x7788, "ldrsh x");
    assert_eq!(x[7], 0x1122_3344, "ldrsw");
    assert_eq!(x[8], 0x1122, "ldrh");
    assert_eq!(w.r64(DATA + 0xff), x1, "unaligned stur");
    assert_eq!(x[11], x1, "unaligned ldur");
    assert_eq!(x[13], 0x5566_7788_7788_0088, "register offset, lsl #3");
    assert_eq!(x[15], 0xdead_beef, "register offset, sxtw #2 of -1");
    assert_eq!(x[16], 0x66, "register offset byte");
    assert_eq!(w.r64(STACK - 16), x1, "stp pre index");
    assert_eq!(w.r64(STACK - 8), DATA);
    assert_eq!((x[17], x[18]), (x1, DATA), "ldp post index");
    assert_eq!(w.r64(DATA + 32), 0xffff_ff88_5566_7788, "stp w");
    assert_eq!((x[19], x[20]), (0x5566_7788, 0xffff_ffff_ffff_ff88), "ldpsw");
    assert_eq!((x[21], x[22]), (0x5566_7788, 0xffff_ff88), "ldp w");
    assert_eq!(x[23], 0xcafe_babe_1234_5678, "ldr literal");
    assert_eq!(x[24], 0xffff_ffff_8000_0000, "ldrsw literal");
    assert_eq!(x[25], x1, "ldtr");
    assert_eq!(w.r64(DATA + 64) & 0xff, 0x88, "sttrb");
    assert_eq!(x[26], 0x6677_8811_2233_4455, "ldur unaligned");
    assert_eq!((x[27], x[28]), (x1, DATA + 0x100), "stnp then ldp");
    assert_eq!(w.r64(DATA - 8), x1, "str pre index with a negative offset");
    // ldrsh w29, [x2, #8]! loads the low half of X1 from DATA.
    assert_eq!(x[29], 0x7788, "ldrsh w with writeback");
    assert_eq!(x[2], DATA, "the base after the last two writebacks");
    assert_eq!(x[31], STACK, "SP after the pair push and pop");
    assert_eq!(st.pc, CODE + 33 * 4);
}

#[test]
fn exclusives() {
    let w = World::a57();
    let code = [
        0xc85f_7c40, // ldxr x0, [x2]
        0x9100_0400, // add x0, x0, #1
        0xc801_7c40, // stxr w1, x0, [x2]
        0xc803_7c40, // stxr w3, x0, [x2]
        0x885f_fc44, // ldaxr w4, [x2]
        0xd503_3f5f, // clrex
        0x8805_fc44, // stlxr w5, w4, [x2]
        0xc87f_1c46, // ldxp x6, x7, [x2]
        0xc828_1847, // stxp w8, x7, x6, [x2]
        0x887f_2849, // ldxp w9, w10, [x2]
        0x882b_244a, // stxp w11, w10, w9, [x2]
        0xc8df_fc4e, // ldar x14, [x2]
        0xc89f_fe0f, // stlr x15, [x16]
        0x085f_7e51, // ldxrb w17, [x18]
        0x0813_fe4c, // stlxrb w19, w12, [x18]
        0x485f_fe54, // ldaxrh w20, [x18]
        WFI,
    ];
    w.w64(DATA, 5);
    w.w64(DATA + 8, 7);
    w.w64(DATA + 0x20, 0xaabb);
    let mut st = w.state();
    st.xregs[2] = DATA;
    st.xregs[15] = 0x1515;
    st.xregs[16] = DATA + 0x10;
    st.xregs[12] = 0x1cc;
    st.xregs[18] = DATA + 0x20;
    for r in [1, 3, 5, 8, 11, 19] {
        st.xregs[r] = 0x55;
    }
    let st = w.run(&st, &code);
    let x = &st.xregs;
    assert_eq!(x[0], 6);
    assert_eq!(x[1], 0, "stxr after ldxr succeeds");
    assert_eq!(x[3], 1, "stxr without a monitor fails");
    assert_eq!(x[4], 6, "ldaxr w");
    assert_eq!(x[5], 1, "stlxr after clrex fails");
    assert_eq!((x[6], x[7]), (6, 7), "ldxp x");
    assert_eq!(x[8], 0, "stxp x succeeds");
    assert_eq!((x[9], x[10]), (7, 0), "ldxp w of the swapped pair");
    assert_eq!(x[11], 0, "stxp w succeeds");
    assert_eq!(w.r64(DATA), 0x7_0000_0000);
    assert_eq!(w.r64(DATA + 8), 6);
    assert_eq!(x[14], 0x7_0000_0000, "ldar");
    assert_eq!(w.r64(DATA + 0x10), 0x1515, "stlr");
    assert_eq!(x[17], 0xbb, "ldxrb");
    assert_eq!(x[19], 0, "stlxrb succeeds");
    assert_eq!(w.r64(DATA + 0x20), 0xaacc);
    assert_eq!(x[20], 0xaacc, "ldaxrh");
    assert_eq!(st.exclusive_addr, DATA + 0x20, "the monitor is open on the last address");
}

#[test]
fn exclusives_need_alignment() {
    // ldxr x0, [x2] with X2 not a multiple of 8: an alignment fault, DFSC 0x21.
    let w = World::a57();
    let st = run_one(&w, w.state(), 0xc85f_7c40, [0, 0, DATA + 1, 0], 0);
    assert_sync_el1(&st, CODE, 0x9600_0021);
    assert_eq!(st.far_el[1], DATA + 1);
}

#[test]
fn lse_atomics() {
    let w = World::a76();
    let code = [
        0xd280_0140, // mov x0, #10
        0xd280_0281, // mov x1, #20
        0xc8a0_7c41, // cas x0, x1, [x2]
        0xd280_0c63, // mov x3, #99
        0xc8a3_7c41, // cas x3, x1, [x2]
        0xd280_00a4, // mov x4, #5
        0xf824_0045, // ldadd x4, x5, [x2]
        0xb8e4_0046, // ldaddal w4, w6, [x2]
        0xd280_1fe7, // mov x7, #0xff
        0xf827_8048, // swp x7, x8, [x2]
        0xd280_1e09, // mov x9, #0xf0
        0xf829_104a, // ldclr x9, x10, [x2]
        0xf829_204b, // ldeor x9, x11, [x2]
        0xd280_200c, // mov x12, #0x100
        0xf82c_304d, // ldset x12, x13, [x2]
        0x9280_000e, // mov x14, #-1
        0xf82e_404f, // ldsmax x14, x15, [x2]
        0xf82e_5050, // ldsmin x14, x16, [x2]
        0xf824_7051, // ldumin x4, x17, [x2]
        0x3821_8052, // swpb w1, w18, [x2]
        0x5280_1013, // mov w19, #0x80
        0x3833_4054, // ldsmaxb w19, w20, [x2]
        0x5290_0015, // mov w21, #0x8000
        0x7835_5056, // ldsminh w21, w22, [x2]
        0x7824_4057, // ldsmaxh w4, w23, [x2]
        0x4878_ff9a, // caspal x24, x25, x26, x27, [x28]
        0xf8bf_c39d, // ldapr x29, [x28]
        0x08a4_7c44, // casb w4, w4, [x2]
        0xf824_005f, // stadd x4, [x2]
        WFI,
    ];
    w.w64(DATA, 10);
    w.w64(DATA + 0x10, 1);
    w.w64(DATA + 0x18, 2);
    let mut st = w.state();
    st.xregs[2] = DATA;
    st.xregs[24] = 1;
    st.xregs[25] = 2;
    st.xregs[26] = 3;
    st.xregs[27] = 4;
    st.xregs[28] = DATA + 0x10;
    let st = w.run(&st, &code);
    let x = &st.xregs;
    assert_eq!(x[0], 10, "cas that matches returns the old value");
    assert_eq!(x[3], 20, "cas that does not match returns the memory value");
    assert_eq!(x[5], 20, "ldadd");
    assert_eq!(x[6], 25, "ldaddal w");
    assert_eq!(x[8], 30, "swp");
    assert_eq!(x[10], 0xff, "ldclr");
    assert_eq!(x[11], 0x0f, "ldeor");
    assert_eq!(x[13], 0xff, "ldset");
    assert_eq!(x[15], 0x1ff, "ldsmax keeps the larger signed value");
    assert_eq!(x[16], 0x1ff, "ldsmin");
    assert_eq!(x[17], u64::MAX, "ldumin");
    assert_eq!(x[18], 5, "swpb");
    assert_eq!(x[20], 20, "ldsmaxb");
    assert_eq!(x[22], 20, "ldsminh");
    assert_eq!(x[23], 0x8000, "ldsmaxh returns the old value zero extended");
    assert_eq!((x[24], x[25]), (1, 2), "caspal returns the old pair");
    assert_eq!(w.r64(DATA + 0x10), 3);
    assert_eq!(w.r64(DATA + 0x18), 4);
    assert_eq!(x[29], 3, "ldapr");
    assert_eq!(x[4], 5, "casb returns the old byte");
    // casb wrote 5 over 5, then stadd added 5.
    assert_eq!(w.r64(DATA), 10);
}

/// Identity map the first 4 MiB with 2 MiB blocks through a four level table at 1 MiB: the
/// first block read/write, the second read only at EL1. Returns MAIR, TCR and TTBR0.
fn page_tables(w: &World) -> (u64, u64, u64) {
    let l0 = 0x10_0000;
    let l1 = l0 + 0x1000;
    let l2 = l0 + 0x2000;
    w.w64(l0, l1 | 3);
    w.w64(l1, l2 | 3);
    // Block, AttrIndx 0, inner shareable, AF.
    w.w64(l2, 0x701);
    // The same with AP[2], read only.
    w.w64(l2 + 8, 0x20_0000 | 0x781);
    let mair = 0xff;
    // T0SZ 16 (48 bit VAs), write back cacheable walks, inner shareable, 4K granule, EPD1.
    let tcr = 16 | (1 << 8) | (1 << 10) | (3 << 12) | (1 << 23);
    (mair, tcr, l0)
}

#[test]
fn mmu_faults_report_esr_and_far() {
    let code = [
        0xd518_a201, // msr mair_el1, x1
        0xd518_2042, // msr tcr_el1, x2
        0xd518_2003, // msr ttbr0_el1, x3
        0xd503_3fdf, // isb
        0xd538_1004, // mrs x4, sctlr_el1
        0xb240_0084, // orr x4, x4, #1
        0xd518_1004, // msr sctlr_el1, x4
        0xd503_3fdf, // isb
        0xf940_00c5, // ldr x5, [x6]
        0xf900_0107, // str x7, [x8]
        WFI,
    ];
    // The faulting address, and the ESR: EC 0x25 (data abort, same EL), IL, WnR, and the
    // DFSC.
    let cases = [
        (0x4000_0000, 0x9600_0045, "translation fault, level 1"),
        (0x8000_0000_0000, 0x9600_0044, "translation fault, level 0"),
        (0x20_0010, 0x9600_004e, "permission fault, level 2"),
    ];
    for (addr, esr, what) in cases {
        let w = World::a57();
        let (mair, tcr, ttbr0) = page_tables(&w);
        w.w64(DATA, 0x0123_4567);
        let mut st = w.state();
        st.xregs[1] = mair;
        st.xregs[2] = tcr;
        st.xregs[3] = ttbr0;
        st.xregs[6] = DATA;
        st.xregs[8] = addr;
        let st = w.run(&st, &code);
        assert_eq!(st.sctlr_el[1] & 1, 1, "{what}: the MMU is on");
        assert_eq!(st.xregs[5], 0x0123_4567, "{what}: the load through the MMU");
        assert_sync_el1(&st, CODE + 0x24, esr);
        assert_eq!(st.far_el[1], addr, "{what}: FAR_EL1");
    }

    // The read only block can still be read.
    let w = World::a57();
    let (mair, tcr, ttbr0) = page_tables(&w);
    w.w64(0x20_0010, 0xabcd);
    let mut st = w.state();
    st.xregs[1] = mair;
    st.xregs[2] = tcr;
    st.xregs[3] = ttbr0;
    st.xregs[6] = 0x20_0010;
    st.xregs[8] = DATA;
    let st = w.run(&st, &code);
    assert_eq!(st.xregs[5], 0xabcd);
    assert_eq!(st.pc, CODE + 0x2c);
}

#[test]
fn svc_round_trip_through_vbar_el1() {
    let w = World::a57();
    // EL0 code at 0x2000.
    w.code(
        0x2000,
        &[
            0xd280_0020, // mov x0, #1
            0xd400_0841, // svc #0x42
            0x9100_2800, // add x0, x0, #10
            WFI,
        ],
    );
    // The EL1 handler for synchronous exceptions from a lower EL, VBAR_EL1 + 0x400.
    w.code(
        0x400,
        &[
            0xd538_5201, // mrs x1, esr_el1
            0xd538_4022, // mrs x2, elr_el1
            0xd538_4003, // mrs x3, spsr_el1
            0xd538_4244, // mrs x4, currentel
            0x9101_9000, // add x0, x0, #100
            0xd69f_03e0, // eret
        ],
    );
    let mut st = w.state();
    st.xregs[9] = 0x2000;
    st.sp_el[0] = 0x2_0000;
    let st = w.run(
        &st,
        &[
            0xd518_4029, // msr elr_el1, x9
            0xd518_401f, // msr spsr_el1, xzr
            0xd69f_03e0, // eret
        ],
    );
    assert_eq!(st.xregs[0], 111);
    assert_eq!(st.xregs[1], 0x5600_0042, "ESR_EL1: EC 0x15, IL, imm16");
    assert_eq!(st.xregs[2], 0x2008, "ELR_EL1: the instruction after the SVC");
    assert_eq!(st.xregs[3], 0, "SPSR_EL1: EL0t with nothing masked");
    assert_eq!(st.xregs[4], 4, "CurrentEL in the handler");
    assert_eq!(st.current_el(), 0, "back at EL0");
    assert_eq!(st.xregs[31], 0x2_0000, "SP_EL0 is the stack at EL0");
    assert_eq!(st.sp_el[1], STACK, "SP_EL1 is banked");
    assert_eq!(st.pc, 0x2010);
}

#[test]
fn irq_is_delivered_when_unmasked() {
    let w = World::a57();
    // The EL1 handler for IRQs at the current EL with SP_EL1, VBAR_EL1 + 0x280.
    w.code(
        0x280,
        &[
            0xd538_4021, // mrs x1, elr_el1
            0xd538_4002, // mrs x2, spsr_el1
            0x1400_0000, // b .
        ],
    );
    w.code(
        CODE,
        &[
            0xd280_0020, // mov x0, #1
            0xd503_42ff, // msr daifclr, #2
            0xd280_0040, // mov x0, #2
            WFI,
        ],
    );
    let mut v = w.vcpu(&w.state());
    w.arm.set_irq(v.shared(), true);
    v.cpu().breakpoint_insert(0x288, bp::GDB);
    assert_eq!(run_to_halt(&mut v), excp::DEBUG);
    let st = save_vcpu(&v);
    assert_eq!(st.pc, 0x288);
    assert_eq!(st.xregs[0], 1, "the IRQ is taken right after the MSR");
    assert_eq!(st.xregs[1], CODE + 8);
    // EL1h with D, A and F still masked and NZCV clear.
    assert_eq!(st.xregs[2], 0x345);
    assert_eq!(st.daif, 0x3c0, "every exception is masked on entry");

    // While PSTATE.I is set, a pending IRQ only wakes WFI.
    let w = World::a57();
    let mut v = w.vcpu(&w.state());
    w.code(CODE, &[0xd280_0020, WFI, 0xd280_0060, WFI]);
    w.arm.set_irq(v.shared(), true);
    v.cpu().breakpoint_insert(CODE + 12, bp::GDB);
    assert_eq!(run_to_halt(&mut v), excp::DEBUG);
    let st = save_vcpu(&v);
    assert_eq!(st.xregs[0], 3);
    assert!(!vcpu_halted(&v));
}

#[test]
fn wfi_halts() {
    let w = World::a57();
    w.code(CODE, &[WFI]);
    let mut v = w.vcpu(&w.state());
    assert_eq!(run_to_halt(&mut v), excp::HLT);
    assert!(vcpu_halted(&v));
    assert_eq!(save_vcpu(&v).pc, CODE + 4);
}

#[test]
fn misaligned_pc_faults() {
    let w = World::a57();
    let mut st = w.state();
    st.pc = CODE + 2;
    let mut v = w.vcpu(&st);
    run_to_halt(&mut v);
    let st = save_vcpu(&v);
    // EC 0x22, PC alignment fault, with the PC in FAR_EL1.
    assert_sync_el1(&st, CODE + 2, 0x8a00_0000);
    assert_eq!(st.far_el[1], CODE + 2);
}
