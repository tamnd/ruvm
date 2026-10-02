// SPDX-License-Identifier: GPL-2.0-or-later

//! Hand assembled A64 programs for EL2 and EL3, run through `ruvm-jit` on the interpreter
//! backend and checked against the Arm ARM and QEMU: the walk down from EL3 to EL0 and back,
//! stage 2 faults, HVC and SMC routing, PSCI, VHE register redirection and the generic timer
//! interrupts.
//!
//! The memory map: RAM from 0 to 4 MiB, the EL1 vector table at 0, the EL2 one at 0x4000 and
//! the EL3 one at 0x5000, all with WFI in every slot unless a test puts a handler there. Code
//! starts at 0x1000, with code for the lower ELs at 0x2000. A program ends in WFI, which halts
//! the vCPU because no interrupt is pending, and the test then looks at the registers.

use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Instant;

use ruvm_jit::cpu_exec::cpu_exec;
use ruvm_jit::{CpuShared, Jit, Vcpu, excp};
use ruvm_mem::{AddressSpace, MemTxAttrs, MemorySystem};
use ruvm_target_arm::cpu::{
    ArmCpuModel, CpuArmState, GTIMER_HYP, GTIMER_PHYS, HCR_E2H, HCR_HCD, HCR_RW, HCR_TGE, HCR_TSC,
    HCR_VM, SCR_HCE, SCR_NS, SCR_RW, SCR_SMD,
};
use ruvm_target_arm::tcg::{
    Arm, ArmBoard, PsciConduit, create_vcpu, new_jit, save_vcpu, vcpu_halted,
};

const U: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const RAM_SIZE: u64 = 0x40_0000;
const CODE: u64 = 0x1000;
const LOWER: u64 = 0x2000;
const VBAR2: u64 = 0x4000;
const VBAR3: u64 = 0x5000;
const STACK: u64 = 0x3_0000;
const WFI: u32 = 0xd503_207f;
const ERET: u32 = 0xd69f_03e0;
const ISB: u32 = 0xd503_3fdf;
const DSB_ISH: u32 = 0xd503_3b9f;

/// ESR_ELx of an Undefined Instruction exception: EC 0 (uncategorized) and IL.
const ESR_UNDEF: u64 = 0x0200_0000;

/// PSTATE values: ELx with SP_ELx and DAIF masked.
const EL1H: u64 = 0x3c5;
const EL2H: u64 = 0x3c9;
const EL3H: u64 = 0x3cd;

/// Vector offsets: synchronous from the current EL with SP_ELx, from a lower EL, and IRQ
/// from the current EL with SP_ELx.
const CUR_SYNC: u64 = 0x200;
const LOW_SYNC: u64 = 0x400;
const CUR_IRQ: u64 = 0x280;

/// System register encodings, (op0, op1, CRn, CRm, op2).
type Key = (u32, u32, u32, u32, u32);
const SCR_EL3: Key = (3, 6, 1, 1, 0);
const SPSR_EL3: Key = (3, 6, 4, 0, 0);
const ELR_EL3: Key = (3, 6, 4, 0, 1);
const ESR_EL3: Key = (3, 6, 5, 2, 0);
const HCR_EL2: Key = (3, 4, 1, 1, 0);
const SPSR_EL2: Key = (3, 4, 4, 0, 0);
const ELR_EL2: Key = (3, 4, 4, 0, 1);
const ESR_EL2: Key = (3, 4, 5, 2, 0);
const FAR_EL2: Key = (3, 4, 6, 0, 0);
const HPFAR_EL2: Key = (3, 4, 6, 0, 4);
const VTCR_EL2: Key = (3, 4, 2, 1, 2);
const VTTBR_EL2: Key = (3, 4, 2, 1, 0);
const SPSR_EL1: Key = (3, 0, 4, 0, 0);
const ELR_EL1: Key = (3, 0, 4, 0, 1);
const ESR_EL1: Key = (3, 0, 5, 2, 0);
const SCTLR_EL1: Key = (3, 0, 1, 0, 0);
const SCTLR_EL12: Key = (3, 5, 1, 0, 0);
const TCR_EL1: Key = (3, 0, 2, 0, 2);
const TTBR0_EL1: Key = (3, 0, 2, 0, 0);
const MAIR_EL1: Key = (3, 0, 10, 2, 0);
const CURRENTEL: Key = (3, 0, 4, 2, 2);
const CNTP_CTL_EL0: Key = (3, 3, 14, 2, 1);
const CNTP_CVAL_EL0: Key = (3, 3, 14, 2, 2);

fn sysreg_bits(k: Key, rt: u32) -> u32 {
    let (op0, op1, crn, crm, op2) = k;
    ((op0 - 2) << 19) | (op1 << 16) | (crn << 12) | (crm << 8) | (op2 << 5) | rt
}

/// `msr <k>, x<rt>`.
fn msr(k: Key, rt: u32) -> u32 {
    0xd510_0000 | sysreg_bits(k, rt)
}

/// `mrs x<rt>, <k>`.
fn mrs(rt: u32, k: Key) -> u32 {
    0xd530_0000 | sysreg_bits(k, rt)
}

/// `sys #op1, Cn, Cm, #op2, x<rt>`, the TLBI forms.
fn sys(op1: u32, crn: u32, crm: u32, op2: u32, rt: u32) -> u32 {
    0xd508_0000 | (op1 << 16) | (crn << 12) | (crm << 8) | (op2 << 5) | rt
}

fn svc(imm: u32) -> u32 {
    0xd400_0001 | (imm << 5)
}

fn hvc(imm: u32) -> u32 {
    0xd400_0002 | (imm << 5)
}

fn smc(imm: u32) -> u32 {
    0xd400_0003 | (imm << 5)
}

/// `movz x<rd>, #imm`.
fn movz(rd: u32, imm: u32) -> u32 {
    0xd280_0000 | (imm << 5) | rd
}

/// What the test board saw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Event {
    Timer(usize, bool),
    CpuOn(u64, u64, u64, u32),
}

/// A board that records the timer outputs and PSCI calls, and wires every timer output to
/// the IRQ line as the virt board's GIC would with the PPIs enabled.
#[derive(Default)]
struct Board {
    log: Mutex<Vec<Event>>,
    arm: OnceLock<Weak<Arm>>,
}

impl Board {
    fn events(&self) -> Vec<Event> {
        self.log.lock().unwrap().clone()
    }
}

impl ArmBoard for Board {
    fn gt_timer_update(
        &self,
        shared: &CpuShared,
        timer: usize,
        level: bool,
        _deadline: Option<Instant>,
    ) {
        self.log.lock().unwrap().push(Event::Timer(timer, level));
        if let Some(arm) = self.arm.get().and_then(Weak::upgrade) {
            arm.set_irq(shared, level);
        }
    }

    fn psci_cpu_on(&self, mpidr: u64, entry: u64, context_id: u64, target_el: u32) -> i64 {
        self.log.lock().unwrap().push(Event::CpuOn(mpidr, entry, context_id, target_el));
        0
    }
}

struct World {
    _sys: MemorySystem,
    as_: Arc<AddressSpace>,
    jit: Arc<Jit>,
    arm: Arc<Arm>,
    board: Arc<Board>,
}

impl World {
    fn new(model: ArmCpuModel, psci: PsciConduit) -> World {
        let sys = MemorySystem::new();
        let root = sys.new_container("system", 1 << 64).unwrap();
        let as_ = sys.address_space_init(root, "memory").unwrap();
        let ram = sys.new_ram("ram", RAM_SIZE).unwrap();
        sys.add_subregion(root, 0, ram).unwrap();
        let board = Arc::new(Board::default());
        let arm = Arc::new(Arm::new(model).with_psci(psci).with_board(board.clone()));
        assert!(board.arm.set(Arc::downgrade(&arm)).is_ok());
        let w = World { _sys: sys, as_, jit: new_jit(), arm, board };
        // Every vector slot of the three tables halts.
        w.code(0, &[WFI; 0x200]);
        w.code(VBAR2, &[WFI; 0x200]);
        w.code(VBAR3, &[WFI; 0x200]);
        w
    }

    /// A Cortex-A57 with EL2 and EL3.
    fn el3() -> World {
        World::new(ArmCpuModel::cortex_a57().with_el2().with_el3(), PsciConduit::Disabled)
    }

    /// A Cortex-A57 with EL2 and no EL3.
    fn el2() -> World {
        World::new(ArmCpuModel::cortex_a57().with_el2(), PsciConduit::Disabled)
    }

    /// A Cortex-A76, which has VHE, with EL2.
    fn vhe() -> World {
        World::new(ArmCpuModel::cortex_a76().with_el2(), PsciConduit::Disabled)
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

    /// The reset state of the model, at the highest EL, with the PC at [`CODE`] and the
    /// vector tables in place.
    fn state(&self) -> CpuArmState {
        let mut st = CpuArmState::reset(self.arm.model());
        st.pc = CODE;
        st.xregs[31] = STACK;
        st.vbar_el[2] = VBAR2;
        st.vbar_el[3] = VBAR3;
        st
    }

    /// [`World::state`] moved to `pstate` (an ELxh value) as firmware would leave it: SCR_EL3
    /// with NS, RW and `scr` set, and HCR_EL2 with RW and `hcr` set.
    fn state_at(&self, pstate: u64, scr: u64, hcr: u64) -> CpuArmState {
        let f = self.arm.model().features;
        let mut st = self.state();
        if f.el3 {
            st.scr_write(&f, SCR_NS | SCR_RW | scr);
        }
        if f.el2 {
            st.hcr_write(&f, false, HCR_RW | hcr);
        }
        st.pstate_write(pstate as u32);
        st
    }

    fn vcpu(&self, st: &CpuArmState) -> Vcpu {
        create_vcpu(&self.jit, self.arm.clone(), self.as_.clone(), st)
    }

    /// Run `code` from [`CODE`] in state `st` until the vCPU halts in WFI.
    fn run(&self, st: &CpuArmState, code: &[u32]) -> CpuArmState {
        self.code(CODE, code);
        let mut v = self.vcpu(st);
        run_to_halt(&mut v);
        assert!(vcpu_halted(&v));
        save_vcpu(&v)
    }
}

/// Run until the vCPU halts in WFI.
fn run_to_halt(v: &mut Vcpu) {
    for _ in 0..1000 {
        let r = cpu_exec(&mut v.cpu());
        if r == excp::HLT {
            return;
        }
        assert_ne!(r, excp::DEBUG);
    }
    panic!("the vCPU did not halt");
}

/// Check that the vCPU halted in the WFI of vector `offset` of EL `el`'s table.
#[track_caller]
fn assert_vector(st: &CpuArmState, el: u32, offset: u64) {
    let vbar = [0, 0, VBAR2, VBAR3][el as usize];
    assert_eq!(st.current_el(), el, "the exception went to EL{el}");
    assert_eq!(st.pc, vbar + offset + 4, "halted at the vector");
}

#[test]
fn walk_from_el3_to_el0_and_back() {
    let w = World::el3();
    // EL2 at LOWER: enter EL1 at LOWER + 0x100.
    w.code(LOWER, &[msr(HCR_EL2, 4), msr(ELR_EL2, 5), msr(SPSR_EL2, 6), ERET]);
    // EL1: enter EL0 at LOWER + 0x200.
    w.code(LOWER + 0x100, &[msr(ELR_EL1, 7), msr(SPSR_EL1, 31), ERET]);
    // EL0: SVC up to EL1, then finish once everything has returned.
    w.code(LOWER + 0x200, &[svc(1), movz(13, 7), WFI]);
    // EL1 sync from EL0: HVC up to EL2, then return to EL0.
    w.code(LOW_SYNC, &[hvc(2), mrs(12, CURRENTEL), mrs(21, ESR_EL1), ERET]);
    // EL2 sync from EL1: SMC up to EL3, then return to EL1.
    w.code(VBAR2 + LOW_SYNC, &[smc(3), mrs(11, CURRENTEL), mrs(20, ESR_EL2), ERET]);
    // EL3 sync from EL2: return to EL2.
    w.code(VBAR3 + LOW_SYNC, &[mrs(10, ESR_EL3), mrs(22, SPSR_EL3), ERET]);

    let mut st = w.state();
    assert_eq!(st.current_el(), 3, "the CPU resets into EL3");
    st.xregs[1] = SCR_NS | SCR_HCE | SCR_RW;
    st.xregs[2] = LOWER;
    st.xregs[3] = EL2H;
    st.xregs[4] = HCR_RW;
    st.xregs[5] = LOWER + 0x100;
    st.xregs[6] = EL1H;
    st.xregs[7] = LOWER + 0x200;
    let st = w.run(&st, &[msr(SCR_EL3, 1), msr(ELR_EL3, 2), msr(SPSR_EL3, 3), ERET]);

    assert_eq!(st.current_el(), 0, "back at EL0");
    assert_eq!(st.pc, LOWER + 0x20c);
    assert_eq!(st.xregs[13], 7);
    assert_eq!(st.xregs[21], 0x5600_0001, "ESR_EL1: SVC #1");
    assert_eq!(st.xregs[20], 0x5a00_0002, "ESR_EL2: HVC #2");
    assert_eq!(st.xregs[10], 0x5e00_0003, "ESR_EL3: SMC #3");
    assert_eq!(st.xregs[22], EL2H, "SPSR_EL3: taken from EL2h");
    assert_eq!(st.spsr_el[2], EL1H, "SPSR_EL2: taken from EL1h");
    assert_eq!(st.spsr_el[1], 0, "SPSR_EL1: taken from EL0t");
    assert_eq!(st.elr_el[1], LOWER + 0x204, "ELR_EL1: after the SVC");
    assert_eq!(st.elr_el[2], LOW_SYNC + 4, "ELR_EL2: after the HVC");
    assert_eq!(st.elr_el[3], VBAR2 + LOW_SYNC + 4, "ELR_EL3: after the SMC");
    assert_eq!(st.xregs[11], 2 << 2, "CurrentEL after the return to EL2");
    assert_eq!(st.xregs[12], 1 << 2, "CurrentEL after the return to EL1");
}

/// Stage 2 tables at 1 MiB for a 32 bit IPA space starting at level 1: IPA 0 to 2 MiB is
/// identity mapped read/write by a level 2 block and everything above is unmapped. Returns
/// VTCR_EL2 and VTTBR_EL2, and the address of the level 2 entry for 2 MiB to 4 MiB.
fn stage2_tables(w: &World) -> (u64, u64, u64) {
    let l1 = 0x10_0000;
    let l2 = l1 + 0x1000;
    w.w64(l1, l2 | 3);
    w.w64(l2, S2_BLOCK);
    // T0SZ 32, SL0 1 (start at level 1), inner write back walks, inner shareable, 4K
    // granule, PS 0 (32 bits).
    let vtcr = 32 | (1 << 6) | (1 << 8) | (1 << 10) | (3 << 12);
    (vtcr, l1, l2 + 8)
}

/// A stage 2 level 2 block: MemAttr normal write back, S2AP read/write, inner shareable, AF.
const S2_BLOCK: u64 = 0x7fd;

#[test]
fn stage2_data_abort_reports_esr_and_hpfar_and_retries() {
    let w = World::el2();
    let (vtcr, vttbr, l2_hole) = stage2_tables(&w);
    w.w64(0x20_0010, 0xfeed);
    // EL1 at LOWER, MMU off so the VA is the IPA.
    w.code(LOWER, &[0xf940_00e6 /* ldr x6, [x7] */, WFI]);
    // The EL2 handler maps the hole, invalidates and returns to the load.
    w.code(
        VBAR2 + LOW_SYNC,
        &[
            0x9100_05ef, // add x15, x15, #1
            mrs(10, ESR_EL2),
            mrs(11, HPFAR_EL2),
            mrs(12, FAR_EL2),
            0xf900_01cd, // str x13, [x14]
            DSB_ISH,
            sys(4, 8, 3, 6, 31), // tlbi vmalls12e1is
            DSB_ISH,
            ERET,
        ],
    );
    let mut st = w.state();
    assert_eq!(st.current_el(), 2, "the CPU resets into EL2");
    st.xregs[1] = vtcr;
    st.xregs[2] = vttbr;
    st.xregs[3] = HCR_RW | HCR_VM;
    st.xregs[4] = LOWER;
    st.xregs[5] = EL1H;
    st.xregs[7] = 0x20_0010;
    st.xregs[13] = 0x20_0000 | S2_BLOCK;
    st.xregs[14] = l2_hole;
    let st = w.run(
        &st,
        &[
            msr(VTCR_EL2, 1),
            msr(VTTBR_EL2, 2),
            msr(HCR_EL2, 3),
            msr(ELR_EL2, 4),
            msr(SPSR_EL2, 5),
            ERET,
        ],
    );
    assert_eq!(st.xregs[15], 1, "one stage 2 fault");
    assert_eq!(st.xregs[10], 0x9200_0006, "ESR_EL2: EC 0x24, IL, level 2 translation fault");
    assert_eq!(st.xregs[11], 0x200 << 4, "HPFAR_EL2: IPA 0x200000 >> 12 in FIPA");
    assert_eq!(st.xregs[12], 0x20_0010, "FAR_EL2: the VA");
    assert_eq!(st.elr_el[2], LOWER, "ELR_EL2: the load");
    assert_eq!(st.xregs[6], 0xfeed, "the load succeeds once the IPA is mapped");
    assert_eq!(st.current_el(), 1);
    assert_eq!(st.pc, LOWER + 8);
}

#[test]
fn stage2_fault_on_a_stage1_walk_sets_s1ptw() {
    let w = World::el2();
    let (vtcr, vttbr, _) = stage2_tables(&w);
    // EL1 turns its MMU on with TTBR0_EL1 in the unmapped IPA range, so the fetch of the
    // instruction after the SCTLR_EL1 write faults walking stage 1.
    w.code(
        LOWER,
        &[
            msr(MAIR_EL1, 8),
            msr(TCR_EL1, 9),
            msr(TTBR0_EL1, 10),
            ISB,
            mrs(11, SCTLR_EL1),
            0xb240_016b, // orr x11, x11, #1
            msr(SCTLR_EL1, 11),
            ISB,
            movz(12, 1),
            WFI,
        ],
    );
    let mut st = w.state();
    st.xregs[1] = vtcr;
    st.xregs[2] = vttbr;
    st.xregs[3] = HCR_RW | HCR_VM;
    st.xregs[4] = LOWER;
    st.xregs[5] = EL1H;
    st.xregs[8] = 0xff;
    // T0SZ 16, 4K granule, EPD1.
    st.xregs[9] = 16 | (1 << 8) | (1 << 10) | (3 << 12) | (1 << 23);
    st.xregs[10] = 0x30_0000;
    let st = w.run(
        &st,
        &[
            msr(VTCR_EL2, 1),
            msr(VTTBR_EL2, 2),
            msr(HCR_EL2, 3),
            msr(ELR_EL2, 4),
            msr(SPSR_EL2, 5),
            ERET,
        ],
    );
    assert_vector(&st, 2, LOW_SYNC);
    assert_eq!(st.esr_el[2], 0x8200_0086, "ESR_EL2: EC 0x20, IL, S1PTW, level 2 translation fault");
    assert_eq!(st.hpfar_el2, 0x300 << 4, "HPFAR_EL2: the stage 1 table's IPA");
    assert_eq!(
        st.far_el[2],
        LOWER + 0x1c,
        "FAR_EL2: the fetch of the ISB after the SCTLR_EL1 write"
    );
    assert_eq!(st.elr_el[2], LOWER + 0x1c);
    assert_eq!(st.xregs[12], 0, "the MOV never ran");
}

/// Run `insn` at `pstate` with SCR_EL3 bits `scr` and HCR_EL2 bits `hcr` set, returning the
/// state once it halts in a vector.
fn run_call(w: &World, pstate: u64, scr: u64, hcr: u64, insn: u32) -> CpuArmState {
    let st = w.state_at(pstate, scr, hcr);
    w.run(&st, &[insn, WFI])
}

#[test]
fn hvc_routing() {
    // HVC from EL1 goes to EL2 when SCR_EL3.HCE is set.
    let st = run_call(&World::el3(), EL1H, SCR_HCE, 0, hvc(0x12));
    assert_vector(&st, 2, LOW_SYNC);
    assert_eq!(st.esr_el[2], 0x5a00_0012);
    assert_eq!(st.elr_el[2], CODE + 4);

    // HVC from EL2 goes to EL2.
    let st = run_call(&World::el3(), EL2H, SCR_HCE, 0, hvc(1));
    assert_vector(&st, 2, CUR_SYNC);
    assert_eq!(st.esr_el[2], 0x5a00_0001);

    // HVC from EL3 goes to EL3.
    let w = World::el3();
    let st = w.run(&w.state_at(EL3H, SCR_HCE, 0), &[hvc(3), WFI]);
    assert_vector(&st, 3, CUR_SYNC);
    assert_eq!(st.esr_el[3], 0x5a00_0003);

    // SCR_EL3.HCE clear: HVC is UNDEFINED, at EL1 and at EL2.
    let st = run_call(&World::el3(), EL1H, 0, 0, hvc(0));
    assert_vector(&st, 1, CUR_SYNC);
    assert_eq!(st.esr_el[1], ESR_UNDEF);
    assert_eq!(st.elr_el[1], CODE);
    let st = run_call(&World::el3(), EL2H, 0, 0, hvc(0));
    assert_vector(&st, 2, CUR_SYNC);
    assert_eq!(st.esr_el[2], ESR_UNDEF);

    // Without EL3, HCR_EL2.HCD disables HVC.
    let st = run_call(&World::el2(), EL1H, 0, HCR_HCD, hvc(0));
    assert_vector(&st, 1, CUR_SYNC);
    assert_eq!(st.esr_el[1], ESR_UNDEF);
    let st = run_call(&World::el2(), EL1H, 0, 0, hvc(5));
    assert_vector(&st, 2, LOW_SYNC);
    assert_eq!(st.esr_el[2], 0x5a00_0005);

    // Without EL2, HVC is UNDEFINED.
    let w = World::new(ArmCpuModel::cortex_a57(), PsciConduit::Disabled);
    let st = w.run(&w.state(), &[hvc(0), WFI]);
    assert_vector(&st, 1, CUR_SYNC);
    assert_eq!(st.esr_el[1], ESR_UNDEF);

    // HVC at EL0 is always UNDEFINED, and under TGE that goes to EL2.
    let st = run_call(&World::el2(), 0, 0, 0, hvc(0));
    assert_vector(&st, 1, LOW_SYNC);
    assert_eq!(st.esr_el[1], ESR_UNDEF);
    let st = run_call(&World::el2(), 0, 0, HCR_TGE, hvc(0));
    assert_vector(&st, 2, LOW_SYNC);
    assert_eq!(st.esr_el[2], ESR_UNDEF);
}

#[test]
fn smc_routing() {
    // SMC from EL1 and EL2 goes to EL3.
    let st = run_call(&World::el3(), EL1H, 0, 0, smc(0x34));
    assert_vector(&st, 3, LOW_SYNC);
    assert_eq!(st.esr_el[3], 0x5e00_0034);
    assert_eq!(st.elr_el[3], CODE + 4, "ELR_EL3: after the SMC");
    assert_eq!(st.spsr_el[3], EL1H);
    let st = run_call(&World::el3(), EL2H, 0, 0, smc(1));
    assert_vector(&st, 3, LOW_SYNC);
    assert_eq!(st.esr_el[3], 0x5e00_0001);

    // HCR_EL2.TSC traps SMC from EL1 to EL2, with ELR_EL2 at the SMC itself.
    let st = run_call(&World::el3(), EL1H, 0, HCR_TSC, smc(0x56));
    assert_vector(&st, 2, LOW_SYNC);
    assert_eq!(st.esr_el[2], 0x5e00_0056);
    assert_eq!(st.elr_el[2], CODE, "ELR_EL2: the trapped SMC");

    // TSC does not apply at EL2.
    let st = run_call(&World::el3(), EL2H, 0, HCR_TSC, smc(2));
    assert_vector(&st, 3, LOW_SYNC);

    // SCR_EL3.SMD makes SMC UNDEFINED, but TSC still wins at EL1.
    let st = run_call(&World::el3(), EL1H, SCR_SMD, 0, smc(0));
    assert_vector(&st, 1, CUR_SYNC);
    assert_eq!(st.esr_el[1], ESR_UNDEF);
    let st = run_call(&World::el3(), EL2H, SCR_SMD, 0, smc(0));
    assert_vector(&st, 2, CUR_SYNC);
    assert_eq!(st.esr_el[2], ESR_UNDEF);
    let st = run_call(&World::el3(), EL1H, SCR_SMD, HCR_TSC, smc(7));
    assert_vector(&st, 2, LOW_SYNC);
    assert_eq!(st.esr_el[2], 0x5e00_0007);

    // Without EL3 SMC is UNDEFINED.
    let st = run_call(&World::el2(), EL1H, 0, 0, smc(0));
    assert_vector(&st, 1, CUR_SYNC);
    assert_eq!(st.esr_el[1], ESR_UNDEF);

    // SMC at EL0 is UNDEFINED.
    let st = run_call(&World::el3(), 0, 0, 0, smc(0));
    assert_vector(&st, 1, LOW_SYNC);
    assert_eq!(st.esr_el[1], ESR_UNDEF);
}

#[test]
fn psci_through_the_conduit() {
    let code = [
        0xd503_201f, // nop, x0 to x3 are set by the test
        hvc(0),
        0xaa00_03f4,       // mov x20, x0
        msr(SPSR_EL1, 31), // a marker that the code went on
        WFI,
    ];
    let run = |conduit: PsciConduit, model: ArmCpuModel, x: [u64; 4], insn: u32| {
        let w = World::new(model, conduit);
        let mut st = w.state();
        st.xregs[..4].copy_from_slice(&x);
        let mut code = code;
        code[1] = insn;
        let st = w.run(&st, &code);
        (st, w.board.events())
    };
    let a57 = ArmCpuModel::cortex_a57;

    // PSCI_VERSION over HVC on a CPU without EL2: 1.1, and execution goes on.
    let (st, _) = run(PsciConduit::Hvc, a57(), [0x8400_0000, 0, 0, 0], hvc(0));
    assert_eq!(st.xregs[20], 0x10001);
    assert_eq!(st.pc, CODE + 0x14);
    assert_eq!(st.current_el(), 1);

    // CPU_ON (64 bit) goes to the board; the target EL is EL1 without EL2.
    let (st, ev) = run(PsciConduit::Hvc, a57(), [0xc400_0003, 0x101, 0x8_0000, 0x55], hvc(0));
    assert_eq!(st.xregs[20], 0);
    assert_eq!(ev, [Event::CpuOn(0x101, 0x8_0000, 0x55, 1)]);

    // Over SMC on a CPU with EL2 the new CPU starts at EL2.
    let (st, ev) = run(PsciConduit::Smc, a57().with_el2(), [0x8400_0003, 1, 0x8_0000, 0], smc(0));
    assert_eq!(st.xregs[20], 0);
    assert_eq!(ev, [Event::CpuOn(1, 0x8_0000, 0, 2)]);

    // PSCI_FEATURES of an unknown function and MIGRATE_INFO_TYPE.
    let (st, _) = run(PsciConduit::Hvc, a57(), [0x8400_000a, 0x8400_0010, 0, 0], hvc(0));
    assert_eq!(st.xregs[20] as i64, -1);
    let (st, _) = run(PsciConduit::Hvc, a57(), [0x8400_0006, 0, 0, 0], hvc(0));
    assert_eq!(st.xregs[20], 2);

    // A function ID that is not PSCI is an ordinary HVC: UNDEFINED without EL2.
    let (st, _) = run(PsciConduit::Hvc, a57(), [0x1234, 0, 0, 0], hvc(0));
    assert_vector(&st, 1, CUR_SYNC);
    assert_eq!(st.esr_el[1], ESR_UNDEF);

    // The other conduit is not PSCI: SMC without EL3 is UNDEFINED.
    let (st, _) = run(PsciConduit::Hvc, a57(), [0x8400_0000, 0, 0, 0], smc(0));
    assert_vector(&st, 1, CUR_SYNC);
    assert_eq!(st.esr_el[1], ESR_UNDEF);
}

#[test]
fn vhe_redirects_el1_registers_to_el2() {
    let w = World::vhe();
    let st = w.state_at(EL2H, 0, HCR_E2H);
    let mut st = st;
    st.xregs[1] = 0x30c5_1838;
    st.xregs[2] = 0x30c5_0838;
    st.xregs[4] = 3;
    let st = w.run(
        &st,
        &[
            msr(SCTLR_EL1, 1),  // SCTLR_EL2 under E2H
            msr(SCTLR_EL12, 2), // the real SCTLR_EL1
            mrs(3, SCTLR_EL1),
            mrs(5, SCTLR_EL12),
            msr(CNTP_CVAL_EL0, 31), // CNTHP_CVAL_EL2 under E2H
            msr(CNTP_CTL_EL0, 4),   // enabled and masked
            mrs(6, CNTP_CTL_EL0),
            WFI,
        ],
    );
    assert_eq!(st.sctlr_el[2], 0x30c5_1838, "MSR SCTLR_EL1 at EL2 with E2H writes SCTLR_EL2");
    assert_eq!(st.sctlr_el[1], 0x30c5_0838, "MSR SCTLR_EL12 writes SCTLR_EL1");
    assert_eq!(st.xregs[3], 0x30c5_1838);
    assert_eq!(st.xregs[5], 0x30c5_0838);
    assert_eq!(st.xregs[6], 7, "CNTHP_CTL_EL2: enabled, masked, ISTATUS");
    assert_eq!(st.gt_ctl[GTIMER_HYP], 7);
    assert_eq!(st.gt_ctl[GTIMER_PHYS], 0, "the EL1 timer is untouched");
    assert!(
        w.board.events().iter().all(|e| matches!(e, Event::Timer(GTIMER_HYP, false))),
        "only the EL2 timer output changes, and it stays low: {:?}",
        w.board.events()
    );

    // Without E2H the EL12 aliases are UNDEFINED and SCTLR_EL1 is itself.
    let w = World::vhe();
    let mut st = w.state_at(EL2H, 0, 0);
    st.xregs[1] = 0x30c5_1838;
    let st = w.run(&st, &[msr(SCTLR_EL1, 1), mrs(5, SCTLR_EL12), WFI]);
    assert_eq!(st.sctlr_el[1], 0x30c5_1838);
    assert_vector(&st, 2, CUR_SYNC);
    assert_eq!(st.esr_el[2], ESR_UNDEF);
    assert_eq!(st.elr_el[2], CODE + 4);

    // A CPU without VHE has no EL12 aliases even with the bit written: E2H is RES0.
    let w = World::el2();
    let st = w.state_at(EL2H, 0, HCR_E2H);
    assert_eq!(st.hcr_el2 & HCR_E2H, 0);
    let st = w.run(&st, &[mrs(5, SCTLR_EL12), WFI]);
    assert_vector(&st, 2, CUR_SYNC);
    assert_eq!(st.esr_el[2], ESR_UNDEF);
}

#[test]
fn vhe_host_el0_traps_to_el2() {
    let w = World::vhe();
    w.code(LOWER, &[svc(9), WFI]);
    let mut st = w.state_at(EL2H, 0, HCR_E2H | HCR_TGE);
    st.xregs[1] = LOWER;
    let st = w.run(&st, &[msr(ELR_EL2, 1), msr(SPSR_EL2, 31), ERET]);
    assert_vector(&st, 2, LOW_SYNC);
    assert_eq!(st.esr_el[2], 0x5600_0009, "SVC from EL0 under TGE goes to EL2");
    assert_eq!(st.elr_el[2], LOWER + 4);
    assert_eq!(st.esr_el[1], 0, "EL1 saw nothing");
}

#[test]
fn timer_interrupt_reaches_the_cpu() {
    let w = World::new(ArmCpuModel::cortex_a57(), PsciConduit::Disabled);
    // The IRQ handler at EL1: read the control, mask the timer, halt.
    w.code(CUR_IRQ, &[mrs(1, CNTP_CTL_EL0), msr(CNTP_CTL_EL0, 2), mrs(3, ELR_EL1), WFI]);
    let mut st = w.state();
    st.xregs[2] = 3;
    st.xregs[4] = 1;
    let st = w.run(
        &st,
        &[
            msr(CNTP_CVAL_EL0, 31), // the compare value is already reached
            msr(CNTP_CTL_EL0, 4),   // enable
            0xd503_42ff,            // msr daifclr, #2
            movz(5, 1),
            WFI,
        ],
    );
    assert_vector(&st, 1, CUR_IRQ + 0xc);
    assert_eq!(st.xregs[1], 5, "CNTP_CTL_EL0: enabled with ISTATUS");
    assert_eq!(st.xregs[3], CODE + 0xc, "the IRQ is taken right after the DAIFClr");
    assert_eq!(st.xregs[5], 0);
    assert_eq!(
        w.board.events(),
        [
            Event::Timer(GTIMER_PHYS, false),
            Event::Timer(GTIMER_PHYS, true),
            Event::Timer(GTIMER_PHYS, false)
        ],
        "the CVAL write updates the low output, it rises on enable and falls when masked"
    );

    // With EL2 the guest's timer access is trapped by CNTHCTL_EL2.EL1PCEN.
    let w = World::el2();
    let mut st = w.state_at(EL1H, 0, 0);
    st.cnthctl_el2 = 0;
    st.xregs[4] = 1;
    let st = w.run(&st, &[msr(CNTP_CTL_EL0, 4), WFI]);
    assert_vector(&st, 2, LOW_SYNC);
    // EC 0x18 (MSR/MRS trap), IL, then the ISS: Op0 3, Op2 1, Op1 3, CRn 14, Rt 4, CRm 2,
    // write.
    assert_eq!(
        st.esr_el[2],
        0x6200_0000 | (3 << 20) | (1 << 17) | (3 << 14) | (14 << 10) | (4 << 5) | (2 << 1)
    );
    assert!(w.board.events().is_empty());
}

#[test]
fn eret_to_an_el_that_is_not_enabled_is_illegal() {
    // An ERET from EL3 to EL2 while SCR_EL3.NS is clear (Secure, no Secure EL2) is an
    // illegal return: the PC goes to ELR, the EL stays and PSTATE.IL is set, so the next
    // instruction takes an Illegal Execution State exception.
    let w = World::el3();
    w.code(LOWER, &[WFI]);
    let mut st = w.state();
    st.xregs[1] = LOWER;
    st.xregs[2] = EL2H;
    let st = w.run(&st, &[msr(ELR_EL3, 1), msr(SPSR_EL3, 2), ERET]);
    assert_vector(&st, 3, CUR_SYNC);
    assert_eq!(st.esr_el[3], 0x3a00_0000, "EC 0x0e and IL");
    assert_eq!(st.elr_el[3], LOWER);
    assert_eq!(st.spsr_el[3] & (1 << 20), 1 << 20, "PSTATE.IL was set");
}

#[test]
fn ttlb_and_tvm_trap_el1() {
    // HCR_EL2.TTLB traps TLBI at EL1 and TVM traps writes to the VM controls.
    let w = World::el2();
    let st = run_call(&w, EL1H, 0, 1 << 25, sys(0, 8, 7, 0, 31)); // tlbi vmalle1
    assert_vector(&st, 2, LOW_SYNC);
    assert_eq!(st.esr_el[2] >> 26, 0x18);
    let w = World::el2();
    let st = run_call(&w, EL1H, 0, 1 << 26, msr(SCTLR_EL1, 0));
    assert_vector(&st, 2, LOW_SYNC);
    assert_eq!(st.esr_el[2] >> 26, 0x18);
    // TVM does not trap reads.
    let w = World::el2();
    let st = run_call(&w, EL1H, 0, 1 << 26, mrs(0, SCTLR_EL1));
    assert_eq!(st.current_el(), 1);
    assert_eq!(st.pc, CODE + 8);
}

#[test]
fn el1_reads_the_virtual_midr_and_mpidr() {
    const MIDR_EL1: Key = (3, 0, 0, 0, 0);
    const MPIDR_EL1: Key = (3, 0, 0, 0, 5);
    const VPIDR_EL2: Key = (3, 4, 0, 0, 0);
    let w = World::el2();
    let st = w.state();
    assert_eq!(st.vpidr_el2, w.arm.model().midr, "VPIDR_EL2 resets to MIDR_EL1");
    w.code(LOWER, &[mrs(6, MIDR_EL1), mrs(7, MPIDR_EL1), WFI]);
    let mut st = st;
    st.xregs[1] = 0x4100_1234;
    st.xregs[2] = LOWER;
    st.xregs[3] = EL1H;
    let st =
        w.run(&st, &[mrs(5, MIDR_EL1), msr(VPIDR_EL2, 1), msr(ELR_EL2, 2), msr(SPSR_EL2, 3), ERET]);
    assert_eq!(st.xregs[5], w.arm.model().midr, "EL2 reads the real MIDR_EL1");
    assert_eq!(st.xregs[6], 0x4100_1234, "EL1 reads VPIDR_EL2");
    assert_eq!(st.xregs[7], 1 << 31, "EL1 reads VMPIDR_EL2, reset from MPIDR_EL1");
}
