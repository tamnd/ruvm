// SPDX-License-Identifier: GPL-2.0-or-later

//! Hand assembled A64 programs for the GICv3 CPU interface registers and semihosting, run
//! through `ruvm-jit` on the interpreter backend: the ICC registers exist only with a GICv3
//! attached and go to it, and `HLT #0xf000` makes the semihosting calls QEMU's
//! `arm-compat-semi.c` makes.
//!
//! RAM is from 0 to 4 MiB with the EL1 vector table at 0 (WFI in every slot) and code at
//! 0x1000. A program ends in WFI, which halts the vCPU.

use std::sync::{Arc, Mutex};

use ruvm_jit::cpu_exec::cpu_exec;
use ruvm_jit::{Vcpu, excp};
use ruvm_mem::{AddressSpace, MemTxAttrs, MemorySystem};
use ruvm_target_arm::cpu::CpuArmState;
use ruvm_target_arm::tcg::{
    ADP_STOPPED_APPLICATION_EXIT, Arm, GicAccess, GicCpuInterface, GicCpuState, IccEncoding,
    SemihostingHost, create_vcpu, new_jit, save_vcpu, vcpu_halted,
};

const U: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const RAM_SIZE: u64 = 0x40_0000;
const CODE: u64 = 0x1000;
const DATA: u64 = 0x8000;
const STACK: u64 = 0x3_0000;
const WFI: u32 = 0xd503_207f;
const ERET: u32 = 0xd69f_03e0;

/// ESR_EL1 of an Undefined Instruction exception: EC 0 (uncategorized) and IL.
const ESR_UNDEF: u64 = 0x0200_0000;
/// The vector of a synchronous exception from the current EL with SP_EL1.
const CUR_SYNC: u64 = 0x200;
/// The vector of a synchronous exception from EL0.
const LOW_SYNC: u64 = 0x400;

type Key = (u32, u32, u32, u32, u32);
const ICC_PMR_EL1: Key = (3, 0, 4, 6, 0);
const ICC_IAR1_EL1: Key = (3, 0, 12, 12, 0);
const ICC_EOIR1_EL1: Key = (3, 0, 12, 12, 1);
const ICC_SRE_EL1: Key = (3, 0, 12, 12, 5);
const ICC_IGRPEN1_EL1: Key = (3, 0, 12, 12, 7);
const ICC_AP0R1_EL1: Key = (3, 0, 12, 8, 5);
const ICC_SGI1R_EL1: Key = (3, 0, 12, 11, 5);
const ID_AA64PFR0_EL1: Key = (3, 0, 0, 4, 0);
const SPSR_EL1: Key = (3, 0, 4, 0, 0);
const ELR_EL1: Key = (3, 0, 4, 0, 1);

fn sysreg_bits(k: Key, rt: u32) -> u32 {
    let (op0, op1, crn, crm, op2) = k;
    ((op0 - 2) << 19) | (op1 << 16) | (crn << 12) | (crm << 8) | (op2 << 5) | rt
}

fn msr(k: Key, rt: u32) -> u32 {
    0xd510_0000 | sysreg_bits(k, rt)
}

fn mrs(rt: u32, k: Key) -> u32 {
    0xd530_0000 | sysreg_bits(k, rt)
}

fn movz(rd: u32, imm: u32) -> u32 {
    0xd280_0000 | (imm << 5) | rd
}

fn svc(imm: u32) -> u32 {
    0xd400_0001 | (imm << 5)
}

fn hlt(imm: u32) -> u32 {
    0xd440_0000 | (imm << 5)
}

/// What the test GIC saw.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Call {
    Read(IccEncoding, u32),
    Write(IccEncoding, u64),
    State(u32),
}

/// A GIC CPU interface that logs every call, reads every register as 0x40 plus its op2, and
/// traps accesses to ICC_SGI1R_EL1 to EL1.
#[derive(Default)]
struct Gic {
    log: Mutex<Vec<Call>>,
}

impl GicCpuInterface for Gic {
    fn access(&self, cpu: usize, reg: IccEncoding, _state: &GicCpuState, _r: bool) -> GicAccess {
        assert_eq!(cpu, 0);
        if reg == ICC_SGI1R_EL1 { GicAccess::TrapEl1 } else { GicAccess::Ok }
    }

    fn read(&self, _cpu: usize, reg: IccEncoding, state: &GicCpuState) -> u64 {
        self.log.lock().unwrap().push(Call::Read(reg, state.el));
        0x40 + u64::from(reg.4)
    }

    fn write(&self, _cpu: usize, reg: IccEncoding, _state: &GicCpuState, value: u64) {
        self.log.lock().unwrap().push(Call::Write(reg, value));
    }

    fn state_changed(&self, _cpu: usize, state: &GicCpuState) {
        self.log.lock().unwrap().push(Call::State(state.el));
    }
}

/// A semihosting host that collects the console output and the exit code.
#[derive(Default)]
struct Host {
    console: Mutex<Vec<u8>>,
    exit: Mutex<Option<u32>>,
}

impl SemihostingHost for Host {
    fn console_write(&self, buf: &[u8]) -> usize {
        self.console.lock().unwrap().extend_from_slice(buf);
        buf.len()
    }

    fn console_read(&self) -> u8 {
        b'q'
    }

    fn exit(&self, code: u32) {
        *self.exit.lock().unwrap() = Some(code);
    }

    fn cmdline(&self) -> Option<String> {
        Some("kernel console=ttyAMA0".into())
    }

    fn heap_info(&self) -> (u64, u64) {
        (0x4100_0000, 0x4800_0000)
    }
}

struct World {
    _sys: MemorySystem,
    as_: Arc<AddressSpace>,
    arm: Arc<Arm>,
}

impl World {
    fn new(arm: Arm) -> World {
        let sys = MemorySystem::new();
        let root = sys.new_container("system", 1 << 64).unwrap();
        let as_ = sys.address_space_init(root, "memory").unwrap();
        let ram = sys.new_ram("ram", RAM_SIZE).unwrap();
        sys.add_subregion(root, 0, ram).unwrap();
        let w = World { _sys: sys, as_, arm: Arc::new(arm) };
        w.code(0, &[WFI; 0x200]);
        w
    }

    fn write(&self, addr: u64, bytes: &[u8]) {
        assert!(self.as_.write(addr, U, bytes).is_ok());
    }

    fn read(&self, addr: u64, len: usize) -> Vec<u8> {
        let mut b = vec![0; len];
        assert!(self.as_.read(addr, U, &mut b).is_ok());
        b
    }

    fn r64(&self, addr: u64) -> u64 {
        u64::from_le_bytes(self.read(addr, 8).try_into().unwrap())
    }

    fn w64(&self, addr: u64, v: u64) {
        self.write(addr, &v.to_le_bytes());
    }

    fn code(&self, addr: u64, words: &[u32]) {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.write(addr, &bytes);
    }

    /// Run `code` at EL1 from [`CODE`] with `regs` in X0 upwards until the vCPU halts.
    fn run(&self, regs: &[u64], code: &[u32]) -> CpuArmState {
        self.code(CODE, code);
        let mut st = CpuArmState::reset(self.arm.model());
        st.pc = CODE;
        st.xregs[31] = STACK;
        st.xregs[..regs.len()].copy_from_slice(regs);
        // A new runtime each time, so no translation of an earlier program is reused.
        let jit = new_jit();
        let mut v = create_vcpu(&jit, self.arm.clone(), self.as_.clone(), &st);
        run_to_halt(&mut v);
        assert!(vcpu_halted(&v));
        save_vcpu(&v)
    }
}

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

/// Check that the program took an UNDEF at the instruction at `CODE + 4 * insn`.
#[track_caller]
fn assert_undef(st: &CpuArmState, insn: u64) {
    assert_eq!(st.pc, CUR_SYNC + 4, "halted at the synchronous vector");
    assert_eq!(st.esr_el[1], ESR_UNDEF);
    assert_eq!(st.elr_el[1], CODE + 4 * insn);
}

fn gic_world() -> (World, Arc<Gic>) {
    let gic = Arc::new(Gic::default());
    let w = World::new(Arm::cortex_a57().with_gicv3(gic.clone()));
    (w, gic)
}

#[test]
fn icc_registers_are_undefined_without_a_gic() {
    let w = World::new(Arm::cortex_a57());
    let st = w.run(&[], &[mrs(1, ICC_PMR_EL1), WFI]);
    assert_undef(&st, 0);
    let st = w.run(&[], &[mrs(1, ICC_SRE_EL1), WFI]);
    assert_undef(&st, 0);
    let st = w.run(&[], &[mrs(1, ID_AA64PFR0_EL1), WFI]);
    assert_eq!((st.xregs[1] >> 24) & 0xf, 0, "ID_AA64PFR0_EL1.GIC is 0");
}

#[test]
fn icc_registers_go_to_the_gic() {
    let (w, gic) = gic_world();
    let st = w.run(
        &[0, 0xf0, 0x23],
        &[
            msr(ICC_PMR_EL1, 1),
            mrs(2, ICC_IAR1_EL1),
            msr(ICC_EOIR1_EL1, 2),
            msr(ICC_IGRPEN1_EL1, 0),
            mrs(3, ICC_SRE_EL1),
            msr(ICC_SRE_EL1, 1),
            mrs(4, ID_AA64PFR0_EL1),
            WFI,
        ],
    );
    assert_eq!(st.pc, CODE + 8 * 4, "ran to the WFI");
    assert_eq!(st.xregs[2], 0x40, "ICC_IAR1_EL1 came from the GIC");
    assert_eq!(st.xregs[3], 7, "ICC_SRE_EL1 is the constant 7");
    assert_eq!((st.xregs[4] >> 24) & 0xf, 1, "ID_AA64PFR0_EL1.GIC is 1");
    assert_eq!(
        *gic.log.lock().unwrap(),
        [
            Call::Write(ICC_PMR_EL1, 0xf0),
            Call::Read(ICC_IAR1_EL1, 1),
            Call::Write(ICC_EOIR1_EL1, 0x40),
            Call::Write(ICC_IGRPEN1_EL1, 0),
        ]
    );
}

#[test]
fn icc_access_checks() {
    let (w, gic) = gic_world();
    // The A57 has 5 priority bits, so there is only one AP0R register.
    let st = w.run(&[], &[mrs(1, ICC_AP0R1_EL1), WFI]);
    assert_undef(&st, 0);
    // IAR1 is read only.
    let st = w.run(&[], &[msr(ICC_IAR1_EL1, 1), WFI]);
    assert_undef(&st, 0);
    // The interface traps SGI1R to EL1: a system register trap, EC 0x18.
    let st = w.run(&[0, 5], &[msr(ICC_SGI1R_EL1, 1), WFI]);
    assert_eq!(st.pc, CUR_SYNC + 4);
    assert_eq!(st.esr_el[1] >> 26, 0x18);
    assert!(gic.log.lock().unwrap().iter().all(|c| matches!(c, Call::State(_))));
}

#[test]
fn el0_access_to_icc_is_undefined_and_el_changes_reach_the_gic() {
    let (w, gic) = gic_world();
    w.code(0x2000, &[mrs(0, ICC_PMR_EL1), WFI]);
    let st = w.run(&[0x2000, 0], &[msr(ELR_EL1, 0), msr(SPSR_EL1, 1), ERET]);
    assert_eq!(st.pc, LOW_SYNC + 4, "EL0 took an exception to EL1");
    assert_eq!(st.esr_el[1], ESR_UNDEF);
    assert_eq!(st.elr_el[1], 0x2000);
    // ERET to EL0, then the UNDEF back to EL1.
    assert_eq!(*gic.log.lock().unwrap(), [Call::State(0), Call::State(1)]);

    let (w, gic) = gic_world();
    let st = w.run(&[], &[svc(0)]);
    assert_eq!(st.pc, CUR_SYNC + 4);
    assert_eq!(*gic.log.lock().unwrap(), [Call::State(1)]);
}

fn semi_world(userspace: bool) -> (World, Arc<Host>) {
    let host = Arc::new(Host::default());
    let w = World::new(Arm::cortex_a57().with_semihosting(host.clone(), userspace));
    (w, host)
}

/// Make semihosting call `nr` with X1 `arg`, then halt; returns X0.
fn call(w: &World, nr: u64, arg: u64) -> CpuArmState {
    let st = w.run(&[nr, arg], &[hlt(0xf000), WFI]);
    assert_eq!(st.pc, CODE + 8, "the call returned past the HLT");
    st
}

#[test]
fn hlt_without_semihosting_is_undefined() {
    let w = World::new(Arm::cortex_a57());
    let st = w.run(&[4, DATA], &[hlt(0xf000), WFI]);
    assert_undef(&st, 0);
    let (w, _) = semi_world(false);
    let st = w.run(&[4, DATA], &[hlt(0xf001), WFI]);
    assert_undef(&st, 0);
}

#[test]
fn hlt_at_el0_needs_userspace_semihosting() {
    let (w, host) = semi_world(false);
    w.write(DATA, b"hi\0");
    w.code(0x2000, &[movz(0, 4), movz(1, DATA as u32), hlt(0xf000), WFI]);
    let st = w.run(&[0x2000, 0], &[msr(ELR_EL1, 0), msr(SPSR_EL1, 1), ERET]);
    assert_eq!(st.pc, LOW_SYNC + 4);
    assert_eq!(st.elr_el[1], 0x2008);
    assert!(host.console.lock().unwrap().is_empty());

    let (w, host) = semi_world(true);
    w.write(DATA, b"hi\0");
    w.code(0x2000, &[movz(0, 4), movz(1, DATA as u32), hlt(0xf000), WFI]);
    let st = w.run(&[0x2000, 0], &[msr(ELR_EL1, 0), msr(SPSR_EL1, 1), ERET]);
    assert_eq!(st.current_el(), 0);
    assert_eq!(st.pc, 0x2010);
    assert_eq!(*host.console.lock().unwrap(), b"hi");
}

#[test]
fn console_calls() {
    let (w, host) = semi_world(false);
    w.write(DATA, b"Hello World\n\0");
    let st = call(&w, 0x04, DATA);
    assert_eq!(st.xregs[0], 0xdead_beef);
    let st = call(&w, 0x03, DATA + 4);
    assert_eq!(st.xregs[0], 0xdead_beef);
    assert_eq!(*host.console.lock().unwrap(), b"Hello World\no");
    // SYS_READC returns the byte and leaves it below the stack pointer.
    let st = call(&w, 0x07, 0);
    assert_eq!(st.xregs[0], u64::from(b'q'));
    assert_eq!(w.read(STACK - 1, 1), b"q");
}

#[test]
fn info_calls() {
    let (w, _) = semi_world(false);
    // SYS_HEAPINFO: heap base, heap limit, stack base, stack limit.
    w.w64(DATA, DATA + 0x100);
    let st = call(&w, 0x16, DATA);
    assert_eq!(st.xregs[0], 0);
    let block: Vec<u64> = (0..4).map(|i| w.r64(DATA + 0x100 + 8 * i)).collect();
    assert_eq!(block, [0x4100_0000, 0x4800_0000, 0x4800_0000, 0x4100_0000]);

    // SYS_GET_CMDLINE: the buffer, its size, then the length written back.
    w.w64(DATA, DATA + 0x200);
    w.w64(DATA + 8, 0x100);
    let st = call(&w, 0x15, DATA);
    assert_eq!(st.xregs[0], 0);
    assert_eq!(w.r64(DATA + 8), 22);
    assert_eq!(w.read(DATA + 0x200, 23), b"kernel console=ttyAMA0\0");
    // A buffer that is too small fails with E2BIG.
    w.w64(DATA + 8, 22);
    let st = call(&w, 0x15, DATA);
    assert_eq!(st.xregs[0], u64::MAX);
    let st = call(&w, 0x13, 0);
    assert_eq!(st.xregs[0], 7, "SYS_ERRNO is E2BIG");

    assert_eq!(call(&w, 0x31, 0).xregs[0], 1_000_000_000, "SYS_TICKFREQ");
    assert_eq!(call(&w, 0x19, DATA).xregs[0], 0, "SYS_SYNCCACHE");
    let st = call(&w, 0x30, DATA + 0x300);
    assert_eq!(st.xregs[0], 0);
    assert_ne!(w.r64(DATA + 0x300), 0, "SYS_ELAPSED counts");
    w.w64(DATA, (-5i64) as u64);
    assert_eq!(call(&w, 0x08, DATA).xregs[0], 1, "SYS_ISERROR");
}

#[test]
fn feature_file() {
    let (w, _) = semi_world(false);
    w.write(DATA + 0x100, b":semihosting-features\0");
    // SYS_OPEN in mode "rb".
    w.w64(DATA, DATA + 0x100);
    w.w64(DATA + 8, 1);
    w.w64(DATA + 16, 21);
    let fd = call(&w, 0x01, DATA).xregs[0];
    assert_eq!(fd, 1, "guest fds start at 1");
    // SYS_FLEN.
    w.w64(DATA, fd);
    assert_eq!(call(&w, 0x0c, DATA).xregs[0], 5);
    // SYS_READ of 8 bytes returns the 3 not read.
    w.w64(DATA + 8, DATA + 0x200);
    w.w64(DATA + 16, 8);
    assert_eq!(call(&w, 0x06, DATA).xregs[0], 3);
    assert_eq!(w.read(DATA + 0x200, 5), [0x53, 0x48, 0x46, 0x42, 0x03]);
    // SYS_SEEK past the end fails; to the start works and returns 0.
    w.w64(DATA + 8, 6);
    assert_eq!(call(&w, 0x0a, DATA).xregs[0], u64::MAX);
    w.w64(DATA + 8, 0);
    assert_eq!(call(&w, 0x0a, DATA).xregs[0], 0);
    // SYS_CLOSE, then closing again fails with EBADF.
    assert_eq!(call(&w, 0x02, DATA).xregs[0], 0);
    assert_eq!(call(&w, 0x02, DATA).xregs[0], u64::MAX);
    assert_eq!(call(&w, 0x13, 0).xregs[0], 9);
    // Opening it for writing fails with EACCES.
    w.w64(DATA, DATA + 0x100);
    w.w64(DATA + 8, 4);
    assert_eq!(call(&w, 0x01, DATA).xregs[0], u64::MAX);
    assert_eq!(call(&w, 0x13, 0).xregs[0], 13);
}

#[test]
fn host_files() {
    let (w, _) = semi_world(false);
    let dir = std::env::temp_dir().join(format!("ruvm-semihost-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("out.txt");
    let name = path.to_str().unwrap().as_bytes();
    w.write(DATA + 0x100, name);
    w.write(DATA + 0x100 + name.len() as u64, &[0]);
    // SYS_OPEN in mode "w".
    w.w64(DATA, DATA + 0x100);
    w.w64(DATA + 8, 4);
    w.w64(DATA + 16, name.len() as u64);
    let fd = call(&w, 0x01, DATA).xregs[0];
    assert!((fd as i64) > 0);
    // SYS_WRITE returns the number of bytes not written.
    w.write(DATA + 0x200, b"semihosted");
    w.w64(DATA, fd);
    w.w64(DATA + 8, DATA + 0x200);
    w.w64(DATA + 16, 10);
    assert_eq!(call(&w, 0x05, DATA).xregs[0], 0);
    assert_eq!(call(&w, 0x0c, DATA).xregs[0], 10, "SYS_FLEN");
    assert_eq!(call(&w, 0x09, DATA).xregs[0], 0, "SYS_ISTTY");
    assert_eq!(call(&w, 0x02, DATA).xregs[0], 0);
    assert_eq!(std::fs::read(&path).unwrap(), b"semihosted");
    // SYS_REMOVE.
    w.w64(DATA, DATA + 0x100);
    w.w64(DATA + 8, name.len() as u64);
    assert_eq!(call(&w, 0x0e, DATA).xregs[0], 0);
    assert!(!path.exists());
    std::fs::remove_dir(&dir).unwrap();
}

#[test]
fn exit_stops_the_cpu() {
    let (w, host) = semi_world(false);
    w.w64(DATA, ADP_STOPPED_APPLICATION_EXIT);
    w.w64(DATA + 8, 3);
    // The CPU stops at the call: nothing after the HLT runs.
    let st = w.run(&[0x18, DATA], &[hlt(0xf000), movz(5, 1), WFI]);
    assert_eq!(st.xregs[5], 0);
    assert_eq!(*host.exit.lock().unwrap(), Some(3));
    // Any other reason is exit status 1.
    w.w64(DATA, 0x20023);
    let _ = w.run(&[0x20, DATA], &[hlt(0xf000), movz(5, 1), WFI]);
    assert_eq!(*host.exit.lock().unwrap(), Some(1));
}

#[test]
fn bad_parameter_block_is_efault() {
    let (w, _) = semi_world(false);
    // The parameter block is outside RAM.
    let st = call(&w, 0x16, 0x1000_0000);
    assert_eq!(st.xregs[0], u64::MAX);
    assert_eq!(call(&w, 0x13, 0).xregs[0], 14);
}
