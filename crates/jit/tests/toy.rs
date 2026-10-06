// SPDX-License-Identifier: GPL-2.0-or-later

//! End to end tests of the runtime with a toy guest: a front end, a CPU with a simple page
//! table, and RAM and a device from `ruvm-mem`.
//!
//! The toy ISA has eight 64-bit registers and 32-bit little endian instructions:
//! `op | a << 8 | b << 12 | imm16 << 16`, with `imm16` signed.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ruvm_jit::accel::start_vcpus;
use ruvm_jit::cpu_exec::cpu_exec;
use ruvm_jit::translate::TbBuild;
use ruvm_jit::{
    Cpu, CpuLoopExit, CpuOps, DisasContextBase, DisasJumpType, ENV_TARGET_OFFSET, InterpBackend,
    Jit, JitConfig, MmuAccessType, Ra, Tb, TbCpuState, TranslatorOps, Vcpu, cputlb, excp,
    interrupt, page, translator_loop,
};
use ruvm_jit_core::types::Cond;
use ruvm_jit_core::{Func, HelperInfo, HelperType, MemOp, Temp};
use ruvm_jit_interp::{HelperEnv, HelperRegistry, Unwind};
use ruvm_mem::{
    AccessCtx, AccessSize, AddressSpace, Endian, MemResult, MemTxAttrs, MemorySystem, MmioOps,
};

const U: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const PAGE: u64 = 0x1000;

// Opcodes.
const LI: u32 = 1;
const ADDI: u32 = 2;
const ADD: u32 = 3;
const LD: u32 = 4;
const ST: u32 = 5;
const BNEZ: u32 = 6;
const J: u32 = 7;
const JR: u32 = 8;
const STOP: u32 = 9;
const HLT: u32 = 10;
const SWI: u32 = 11;
const EI: u32 = 12;
const INC: u32 = 13;
const NOP: u32 = 14;
const XADD: u32 = 15;
const SUB: u32 = 16;
const ST16: u32 = 17;
const LD16: u32 = 18;

const EXCP_SWI: i32 = 3;
const EXCP_PAGEFAULT: i32 = 14;
const EXCP_ILLEGAL: i32 = 4;

const IRQ_VECTOR: u64 = 0xe000;
const FAULT_VECTOR: u64 = 0xf000;

const fn reg_off(r: usize) -> usize {
    ENV_TARGET_OFFSET + 8 * r
}
const PC_OFF: usize = ENV_TARGET_OFFSET + 64;
const IRQ_EN_OFF: usize = ENV_TARGET_OFFSET + 72;
const FAULT_ADDR_OFF: usize = ENV_TARGET_OFFSET + 80;
const ENV_SIZE: usize = 96;

fn enc(op: u32, a: u32, b: u32, imm: i32) -> u32 {
    assert!((-0x8000..0x8000).contains(&imm));
    op | (a << 8) | (b << 12) | ((imm as u32 & 0xffff) << 16)
}

/// A tiny assembler. Branch offsets are relative to the next instruction.
struct Asm {
    base: u64,
    words: Vec<u32>,
}

impl Asm {
    fn new(base: u64) -> Asm {
        Asm { base, words: Vec::new() }
    }
    fn pc(&self) -> u64 {
        self.base + 4 * self.words.len() as u64
    }
    fn i(&mut self, op: u32, a: u32, b: u32, imm: i32) -> &mut Asm {
        self.words.push(enc(op, a, b, imm));
        self
    }
    fn li(&mut self, a: u32, imm: i32) -> &mut Asm {
        self.i(LI, a, 0, imm)
    }
    fn addi(&mut self, a: u32, b: u32, imm: i32) -> &mut Asm {
        self.i(ADDI, a, b, imm)
    }
    fn jump_to(&mut self, op: u32, a: u32, target: u64) -> &mut Asm {
        let off = target as i64 - (self.pc() as i64 + 4);
        self.i(op, a, 0, off as i32)
    }
    fn bnez(&mut self, a: u32, target: u64) -> &mut Asm {
        self.jump_to(BNEZ, a, target)
    }
    fn j(&mut self, target: u64) -> &mut Asm {
        self.jump_to(J, 0, target)
    }
    fn stop(&mut self) -> &mut Asm {
        self.i(STOP, 0, 0, 0)
    }
    fn load(&self, as_: &AddressSpace) {
        let bytes: Vec<u8> = self.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert!(as_.write(self.base, U, &bytes).is_ok());
    }
}

fn rd64(env: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(env[off..off + 8].try_into().unwrap())
}

fn wr64(env: &mut [u8], off: usize, v: u64) {
    env[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

fn reg(v: &Vcpu, r: usize) -> u64 {
    rd64(&v.env, reg_off(r))
}

fn pc(v: &Vcpu) -> u64 {
    rd64(&v.env, PC_OFF)
}

/// The toy CPU.
#[derive(Debug, Default)]
struct ToyOps {
    /// Virtual page to physical page and protection. `None` means paging is off.
    pt: Mutex<Option<HashMap<u64, (u64, u32)>>>,
    tb_cpu_state_calls: AtomicU64,
    faults: AtomicU64,
    swis: AtomicU64,
    irqs: AtomicU64,
    fills: AtomicU64,
}

impl ToyOps {
    fn map(&self, vaddr: u64, paddr: u64, prot: u32) {
        self.pt
            .lock()
            .unwrap()
            .get_or_insert_with(HashMap::new)
            .insert(vaddr / PAGE, (paddr, prot));
    }
}

impl CpuOps for ToyOps {
    fn translate_code(&self, cpu: &mut Cpu<'_>, tb: &mut TbBuild) -> Result<(), CpuLoopExit> {
        translator_loop(cpu, tb, &mut ToyDisas)
    }

    fn get_tb_cpu_state(&self, cpu: &Cpu<'_>) -> TbCpuState {
        self.tb_cpu_state_calls.fetch_add(1, Ordering::Relaxed);
        TbCpuState { pc: rd64(cpu.env, PC_OFF), flags: 0, cflags: 0, cs_base: 0 }
    }

    fn restore_state_to_opc(&self, cpu: &mut Cpu<'_>, _tb: &Tb, data: &[u64; 3]) {
        wr64(cpu.env, PC_OFF, data[0]);
    }

    fn set_pc(&self, cpu: &mut Cpu<'_>, pc: u64) {
        wr64(cpu.env, PC_OFF, pc);
    }

    fn get_pc(&self, cpu: &Cpu<'_>) -> u64 {
        rd64(cpu.env, PC_OFF)
    }

    fn cpu_exec_interrupt(&self, cpu: &mut Cpu<'_>, interrupt_request: u32) -> bool {
        if interrupt_request & interrupt::HARD != 0 && rd64(cpu.env, IRQ_EN_OFF) != 0 {
            cpu.core.shared().reset_interrupt(interrupt::HARD);
            let pc = rd64(cpu.env, PC_OFF);
            wr64(cpu.env, reg_off(7), pc);
            wr64(cpu.env, PC_OFF, IRQ_VECTOR);
            wr64(cpu.env, IRQ_EN_OFF, 0);
            self.irqs.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        false
    }

    fn do_interrupt(&self, cpu: &mut Cpu<'_>) {
        match cpu.core.exception_index {
            EXCP_SWI => {
                self.swis.fetch_add(1, Ordering::Relaxed);
                let pc = rd64(cpu.env, PC_OFF);
                wr64(cpu.env, PC_OFF, pc + 4);
            }
            EXCP_PAGEFAULT => {
                self.faults.fetch_add(1, Ordering::Relaxed);
                let a = rd64(cpu.env, FAULT_ADDR_OFF);
                wr64(cpu.env, reg_off(6), a);
                let pc = rd64(cpu.env, PC_OFF);
                wr64(cpu.env, reg_off(7), pc);
                wr64(cpu.env, PC_OFF, FAULT_VECTOR);
            }
            e => panic!("unexpected exception {e}"),
        }
    }

    fn tlb_fill(
        &self,
        cpu: &mut Cpu<'_>,
        addr: u64,
        _size: usize,
        access_type: MmuAccessType,
        mmu_idx: usize,
        probe: bool,
        ra: Ra,
    ) -> Result<bool, CpuLoopExit> {
        self.fills.fetch_add(1, Ordering::Relaxed);
        let vpage = addr & !(PAGE - 1);
        let entry = match &*self.pt.lock().unwrap() {
            None => Some((vpage, page::RWX)),
            Some(pt) => pt.get(&(addr / PAGE)).copied(),
        };
        let need = match access_type {
            MmuAccessType::DataLoad => page::READ,
            MmuAccessType::DataStore => page::WRITE,
            MmuAccessType::InstFetch => page::EXEC,
        };
        match entry {
            Some((ppage, prot)) if prot & need != 0 => {
                cpu.tlb_set_page(vpage, ppage, prot, mmu_idx, PAGE);
                Ok(true)
            }
            _ if probe => Ok(false),
            _ => {
                wr64(cpu.env, FAULT_ADDR_OFF, addr);
                Err(cpu.raise_exception(EXCP_PAGEFAULT, ra))
            }
        }
    }

    fn do_unaligned_access(
        &self,
        cpu: &mut Cpu<'_>,
        _addr: u64,
        _access_type: MmuAccessType,
        _mmu_idx: usize,
        ra: Ra,
    ) -> CpuLoopExit {
        cpu.raise_exception(EXCP_ILLEGAL, ra)
    }

    fn mmu_index(&self, _cpu: &Cpu<'_>, _ifetch: bool) -> usize {
        0
    }
}

/// `toy_raise(env, excp)`.
fn helper_raise(_h: &mut HelperEnv<'_>, args: &[u64]) -> Result<u128, Unwind> {
    Err(Unwind::Exception(args[1] & 0xffff_ffff))
}

/// `toy_hlt(env)`: halt and leave the loop with the PC already past the instruction.
fn helper_hlt(h: &mut HelperEnv<'_>, _args: &[u64]) -> Result<u128, Unwind> {
    let mut cpu = Cpu::from_helper_env(h).expect("runs under the runtime");
    cpu.core.shared().halted.store(1, Ordering::Release);
    cpu.core.exception_index = excp::HLT;
    let e = cpu.cpu_loop_exit();
    Err(cpu.unwind(e))
}

fn raise_info() -> HelperInfo {
    HelperInfo::new("toy_raise", 0, HelperType::Void, &[HelperType::Ptr, HelperType::I32])
}

fn hlt_info() -> HelperInfo {
    HelperInfo::new("toy_hlt", 0, HelperType::Void, &[HelperType::Ptr])
}

fn backend() -> Arc<InterpBackend> {
    let mut reg = HelperRegistry::new();
    reg.register_info(&raise_info(), helper_raise);
    reg.register_info(&hlt_info(), helper_hlt);
    Arc::new(InterpBackend::with_helpers(reg))
}

/// The front end.
struct ToyDisas;

fn ld_reg(f: &mut Func, r: u32) -> ruvm_jit_core::ir::TempI64 {
    let t = f.temp_new_i64();
    let env = f.env();
    f.gen_ld_i64(t, env, reg_off(r as usize) as i64);
    t
}

fn st_reg(f: &mut Func, r: u32, t: ruvm_jit_core::ir::TempI64) {
    let env = f.env();
    f.gen_st_i64(t, env, reg_off(r as usize) as i64);
}

fn set_pc(f: &mut Func, pc: u64) {
    let t = f.constant_i64(pc as i64);
    let env = f.env();
    f.gen_st_i64(t, env, PC_OFF as i64);
}

fn gen_goto(db: &mut DisasContextBase<'_>, n: u64, dest: u64) {
    let use_goto_tb = db.translator_use_goto_tb(dest);
    let id = db.tb.id;
    let f = &mut db.tb.f;
    if use_goto_tb {
        f.gen_goto_tb(n);
        set_pc(f, dest);
        f.gen_exit_tb(id, n);
    } else {
        set_pc(f, dest);
        f.gen_lookup_and_goto_ptr();
    }
}

fn gen_raise(f: &mut Func, excp: i32) {
    let h = f.helper(raise_info());
    let env = f.env();
    let c = f.constant_i32(excp);
    f.gen_call(h, None, &[Temp::from(env), Temp::from(c)]);
}

impl TranslatorOps for ToyDisas {
    fn insn_start(&mut self, db: &mut DisasContextBase<'_>, _cpu: &mut Cpu<'_>) {
        let pc = db.pc_next;
        db.tb.f.gen_insn_start(&[pc]);
    }

    fn translate_insn(
        &mut self,
        db: &mut DisasContextBase<'_>,
        cpu: &mut Cpu<'_>,
    ) -> Result<(), CpuLoopExit> {
        let pc = db.pc_next;
        let w = db.translator_ldl(cpu, pc, Endian::Little)?;
        db.pc_next = pc + 4;
        let next = pc + 4;
        let op = w & 0xff;
        let a = (w >> 8) & 7;
        let b = (w >> 12) & 7;
        let imm = i64::from((w >> 16) as u16 as i16);
        let target = next.wrapping_add(imm as u64);
        let parallel = db.tb.cflags() & ruvm_jit::cf::PARALLEL != 0;
        let f = &mut db.tb.f;
        match op {
            LI => {
                let t = f.constant_i64(imm);
                st_reg(f, a, t);
            }
            ADDI => {
                let t = ld_reg(f, b);
                f.gen_addi_i64(t, t, imm);
                st_reg(f, a, t);
            }
            ADD => {
                let x = ld_reg(f, a);
                let y = ld_reg(f, b);
                f.gen_add_i64(x, x, y);
                st_reg(f, a, x);
            }
            LD => {
                let addr = ld_reg(f, b);
                f.gen_addi_i64(addr, addr, imm);
                let v = f.temp_new_i64();
                f.gen_qemu_ld_i64(v, addr, 0, MemOp::UL);
                st_reg(f, a, v);
            }
            ST => {
                let addr = ld_reg(f, b);
                f.gen_addi_i64(addr, addr, imm);
                let v = ld_reg(f, a);
                f.gen_qemu_st_i64(v, addr, 0, MemOp::UL);
            }
            INC => {
                if parallel {
                    // Not supported in parallel: do it with the world stopped.
                    let h = f.helper(ruvm_jit_core::helpers::exit_atomic());
                    let env = f.env();
                    f.gen_call(h, None, &[Temp::from(env)]);
                    db.is_jmp = DisasJumpType::NoReturn;
                } else {
                    let addr = ld_reg(f, b);
                    let v = f.temp_new_i64();
                    f.gen_qemu_ld_i64(v, addr, 0, MemOp::UL);
                    f.gen_addi_i64(v, v, 1);
                    f.gen_qemu_st_i64(v, addr, 0, MemOp::UL);
                }
            }
            XADD => {
                // a = fetch_add([b], a), 32 bits.
                let addr = ld_reg(f, b);
                let v = ld_reg(f, a);
                f.gen_atomic_fetch_add_i64(v, addr, v, 0, MemOp::UL);
                st_reg(f, a, v);
            }
            SUB => {
                let x = ld_reg(f, a);
                let y = ld_reg(f, b);
                f.gen_sub_i64(x, x, y);
                st_reg(f, a, x);
            }
            ST16 => {
                let addr = ld_reg(f, b);
                f.gen_addi_i64(addr, addr, imm);
                let v = ld_reg(f, a);
                f.gen_qemu_st_i64(v, addr, 0, MemOp::UW);
            }
            LD16 => {
                let addr = ld_reg(f, b);
                f.gen_addi_i64(addr, addr, imm);
                let v = f.temp_new_i64();
                f.gen_qemu_ld_i64(v, addr, 0, MemOp::UW);
                st_reg(f, a, v);
            }
            BNEZ => {
                let t = ld_reg(f, a);
                let l = f.new_label();
                f.gen_brcondi_i64(Cond::Ne, t, 0, l);
                gen_goto(db, 1, next);
                db.tb.f.gen_set_label(l);
                gen_goto(db, 0, target);
                db.is_jmp = DisasJumpType::NoReturn;
            }
            J => {
                gen_goto(db, 0, target);
                db.is_jmp = DisasJumpType::NoReturn;
            }
            JR => {
                let t = ld_reg(f, a);
                let env = f.env();
                f.gen_st_i64(t, env, PC_OFF as i64);
                f.gen_lookup_and_goto_ptr();
                db.is_jmp = DisasJumpType::NoReturn;
            }
            STOP => {
                gen_raise(f, excp::DEBUG);
                db.is_jmp = DisasJumpType::NoReturn;
            }
            SWI => {
                gen_raise(f, EXCP_SWI);
                db.is_jmp = DisasJumpType::NoReturn;
            }
            HLT => {
                set_pc(f, next);
                let h = f.helper(hlt_info());
                let env = f.env();
                f.gen_call(h, None, &[Temp::from(env)]);
                db.is_jmp = DisasJumpType::NoReturn;
            }
            EI => {
                let one = f.constant_i64(1);
                let env = f.env();
                f.gen_st_i64(one, env, IRQ_EN_OFF as i64);
                // Interrupts may be pending: end the block.
                db.is_jmp = DisasJumpType::TooMany;
            }
            NOP => {}
            _ => {
                gen_raise(f, EXCP_ILLEGAL);
                db.is_jmp = DisasJumpType::NoReturn;
            }
        }
        Ok(())
    }

    fn tb_stop(&mut self, db: &mut DisasContextBase<'_>, _cpu: &mut Cpu<'_>) {
        match db.is_jmp {
            DisasJumpType::Next | DisasJumpType::TooMany => {
                let next = db.pc_next;
                gen_goto(db, 0, next);
            }
            _ => {}
        }
    }
}

/// A device that counts reads and returns `0x40 + offset`.
#[derive(Default)]
struct Dev {
    reads: AtomicUsize,
    writes: Mutex<Vec<(u64, u64)>>,
}

impl MmioOps for Dev {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        Ok(0x40 + offset)
    }
    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.writes.lock().unwrap().push((offset, value));
        Ok(())
    }
}

const DEV_BASE: u64 = 0x10_0000;

struct World {
    _sys: MemorySystem,
    as_: Arc<AddressSpace>,
    jit: Arc<Jit>,
    ops: Arc<ToyOps>,
    dev: Arc<Dev>,
}

fn world(config: JitConfig) -> World {
    let sys = MemorySystem::new();
    let root = sys.new_container("system", 1 << 64).unwrap();
    let as_ = sys.address_space_init(root, "memory").unwrap();
    let ram = sys.new_ram("ram", 0x10_0000).unwrap();
    sys.add_subregion(root, 0, ram).unwrap();
    let dev = Arc::new(Dev::default());
    let io = sys.new_io("dev", 0x1000, dev.clone()).unwrap();
    sys.add_subregion(root, DEV_BASE, io).unwrap();
    let jit = Jit::new(config, backend());
    World { _sys: sys, as_, jit, ops: Arc::new(ToyOps::default()), dev }
}

impl World {
    fn vcpu(&self, pc: u64) -> Vcpu {
        let mut v = self.jit.create_vcpu(self.ops.clone(), self.as_.clone(), ENV_SIZE);
        wr64(&mut v.env, PC_OFF, pc);
        v
    }

    fn read32(&self, addr: u64) -> u32 {
        self.as_.read_u32(addr, U).0
    }

    fn write32(&self, addr: u64, v: u32) {
        assert!(self.as_.write_u32(addr, U, v).is_ok());
    }
}

fn run(v: &mut Vcpu) -> i32 {
    cpu_exec(&mut v.cpu())
}

#[test]
fn straight_line_and_stop() {
    let w = world(JitConfig::default());
    let mut a = Asm::new(0x1000);
    a.li(1, 5).li(2, 7).i(ADD, 1, 2, 0).addi(3, 1, -2).stop();
    a.load(&w.as_);
    let mut v = w.vcpu(0x1000);
    assert_eq!(run(&mut v), excp::DEBUG);
    assert_eq!(reg(&v, 1), 12);
    assert_eq!(reg(&v, 3), 10);
    // The state is restored to the STOP instruction.
    assert_eq!(pc(&v), 0x1010);
    assert_eq!(w.jit.tb_count(), 1);
}

/// A vCPU leaves the CPU list, and frees its TLB, when it is dropped, so creating one per test
/// case does not grow without bound. A new one takes the lowest free index.
#[test]
fn dropped_vcpus_leave_the_cpu_list() {
    let w = world(JitConfig::default());
    let mut a = Asm::new(0x1000);
    a.li(1, 5).stop();
    a.load(&w.as_);
    for _ in 0..500 {
        let mut v = w.vcpu(0x1000);
        assert_eq!(run(&mut v), excp::DEBUG);
        assert_eq!(v.shared().cpu_index, 0);
        assert_eq!(w.jit.cpu_list().len(), 1);
    }
    assert!(w.jit.cpu_list().is_empty());
    let a = w.vcpu(0x1000);
    let b = w.vcpu(0x1000);
    let c = w.vcpu(0x1000);
    drop(b);
    assert_eq!(w.jit.cpu_list().len(), 2);
    let d = w.vcpu(0x1000);
    assert_eq!([a.shared().cpu_index, c.shared().cpu_index, d.shared().cpu_index], [0, 2, 3]);
    assert_eq!(w.jit.tb_count(), 1);
}

fn counting_loop(w: &World, n: i32) -> Vcpu {
    let mut a = Asm::new(0x1000);
    a.li(1, 0).li(2, n);
    let head = a.pc();
    a.addi(1, 1, 2).addi(2, 2, -1).bnez(2, head).stop();
    a.load(&w.as_);
    w.vcpu(0x1000)
}

#[test]
fn goto_tb_chaining() {
    let w = world(JitConfig::default());
    let mut v = counting_loop(&w, 1000);
    assert_eq!(run(&mut v), excp::DEBUG);
    assert_eq!(reg(&v, 1), 2000);
    // The loop block chains to itself, so the execution loop runs only a handful of times.
    let lookups = w.ops.tb_cpu_state_calls.load(Ordering::Relaxed);
    assert!(lookups < 10, "{lookups} lookups");

    let nochain = world(JitConfig { nochain: true, ..JitConfig::default() });
    let mut v = counting_loop(&nochain, 1000);
    assert_eq!(run(&mut v), excp::DEBUG);
    assert_eq!(reg(&v, 1), 2000);
    let lookups = nochain.ops.tb_cpu_state_calls.load(Ordering::Relaxed);
    assert!(lookups >= 1000, "{lookups} lookups");
}

#[test]
fn lookup_and_goto_ptr_chaining() {
    let w = world(JitConfig::default());
    // An indirect jump back to the loop head.
    let mut a = Asm::new(0x1000);
    a.li(1, 0).li(2, 300);
    let head = a.pc();
    a.li(5, head as i32).addi(1, 1, 1).addi(2, 2, -1);
    let over = a.pc() + 8;
    a.bnez(2, over).stop().i(JR, 5, 0, 0);
    a.load(&w.as_);
    let mut v = w.vcpu(0x1000);
    assert_eq!(run(&mut v), excp::DEBUG);
    assert_eq!(reg(&v, 1), 300);
    // helper_lookup_tb_ptr finds the blocks, the main loop is entered rarely.
    assert!(w.jit.tb_count() <= 4);
}

#[test]
fn self_modifying_code() {
    let w = world(JitConfig::default());
    let patched = 0x3000;
    // The block that will be patched lives on its own page.
    let mut p = Asm::new(patched);
    p.addi(1, 1, 1).j(0x2000);
    p.load(&w.as_);
    let mut b = Asm::new(0x2000);
    b.jump_to(BNEZ, 4, 0x2008).stop().li(4, 0).j(0x1008);
    b.load(&w.as_);
    // The new instruction is data at 0x5000.
    w.write32(0x5000, enc(ADDI, 1, 1, 100));
    let mut a = Asm::new(0x1000);
    a.li(4, 1).j(patched);
    // 0x1008: patch the first instruction of the block, then run it again.
    a.li(5, 0x5000).i(LD, 6, 5, 0).li(5, patched as i32).i(ST, 6, 5, 0).j(patched);
    a.load(&w.as_);
    let mut v = w.vcpu(0x1000);
    assert_eq!(run(&mut v), excp::DEBUG);
    assert_eq!(reg(&v, 1), 101);
    assert!(w.jit.tb_phys_invalidate_count() >= 1);
}

#[test]
fn smc_from_outside_needs_invalidate() {
    let w = world(JitConfig::default());
    let mut a = Asm::new(0x1000);
    a.li(1, 1).stop();
    a.load(&w.as_);
    let mut v = w.vcpu(0x1000);
    assert_eq!(run(&mut v), excp::DEBUG);
    assert_eq!(reg(&v, 1), 1);
    // A write that bypasses the softmmu, like DMA.
    w.write32(0x1000, enc(LI, 1, 0, 2));
    w.jit.tb_invalidate_phys_range(0x1000, 0x1003);
    wr64(&mut v.env, PC_OFF, 0x1000);
    assert_eq!(run(&mut v), excp::DEBUG);
    assert_eq!(reg(&v, 1), 2);
}

#[test]
fn tlb_fill_fault_and_flush() {
    let w = world(JitConfig::default());
    // Identity map the code pages, map 0x40000 to 0x5000 read only.
    for p in [0x1000, FAULT_VECTOR] {
        w.ops.map(p, p, page::READ | page::EXEC);
    }
    w.ops.map(0x40000, 0x5000, page::READ);
    w.write32(0x5000, 1234);
    w.write32(0x7000, 5678);
    let mut a = Asm::new(0x1000);
    a.li(5, 0x4000).addi(5, 5, 0x4000).addi(5, 5, 0x4000).addi(5, 5, 0x4000);
    // r5 = 0x10000; make it 0x40000 by adding r5 to itself twice.
    a.i(ADD, 5, 5, 0).i(ADD, 5, 5, 0);
    a.i(LD, 1, 5, 0).stop();
    // 0x1020: store to the read only page.
    a.i(ST, 1, 5, 0).stop();
    a.load(&w.as_);
    let mut f = Asm::new(FAULT_VECTOR);
    f.stop();
    f.load(&w.as_);

    let mut v = w.vcpu(0x1000);
    assert_eq!(run(&mut v), excp::DEBUG);
    assert_eq!(reg(&v, 1), 1234);
    assert_eq!(pc(&v), 0x101c);

    // Remap and flush: the next load sees the new page.
    w.ops.map(0x40000, 0x7000, page::READ);
    cputlb::tlb_flush_page(&mut v.cpu(), 0x40000);
    wr64(&mut v.env, PC_OFF, 0x1018);
    assert_eq!(run(&mut v), excp::DEBUG);
    assert_eq!(reg(&v, 1), 5678);

    // A store to the read only page faults, with the state restored to the store.
    let fills = w.ops.fills.load(Ordering::Relaxed);
    wr64(&mut v.env, PC_OFF, 0x1020);
    assert_eq!(run(&mut v), excp::DEBUG);
    assert_eq!(w.ops.faults.load(Ordering::Relaxed), 1);
    assert_eq!(reg(&v, 6), 0x40000);
    assert_eq!(reg(&v, 7), 0x1020);
    assert_eq!(pc(&v), FAULT_VECTOR);
    assert!(w.ops.fills.load(Ordering::Relaxed) > fills);

    // An instruction fetch from an unmapped page faults too.
    wr64(&mut v.env, PC_OFF, 0x9000);
    assert_eq!(run(&mut v), excp::DEBUG);
    assert_eq!(w.ops.faults.load(Ordering::Relaxed), 2);
    assert_eq!(reg(&v, 6), 0x9000);

    // Only the dirty MMU index is flushed, which counts as a partial flush as in QEMU.
    let (full, part) = cputlb::tlb_flush_counts(&v.cpu());
    cputlb::tlb_flush(&mut v.cpu());
    assert_eq!(cputlb::tlb_flush_counts(&v.cpu()), (full, part + 1));
}

#[test]
fn exceptions_are_delivered() {
    let w = world(JitConfig::default());
    let mut a = Asm::new(0x1000);
    a.li(1, 0).i(SWI, 0, 0, 0).addi(1, 1, 1).i(SWI, 0, 0, 0).addi(1, 1, 1).stop();
    a.load(&w.as_);
    let mut v = w.vcpu(0x1000);
    assert_eq!(run(&mut v), excp::DEBUG);
    assert_eq!(w.ops.swis.load(Ordering::Relaxed), 2);
    assert_eq!(reg(&v, 1), 2);
}

#[test]
fn mmio_in_the_middle_of_a_block_is_recompiled() {
    let w = world(JitConfig::default());
    let mut a = Asm::new(0x1000);
    // r5 = DEV_BASE (0x100000) = 0x4000 * 64 built by doubling.
    a.li(5, 0x4000);
    for _ in 0..6 {
        a.i(ADD, 5, 5, 0);
    }
    a.li(2, 0).i(LD, 1, 5, 8).addi(2, 2, 3).i(ST, 2, 5, 0x10).stop();
    a.load(&w.as_);
    let mut v = w.vcpu(0x1000);
    assert_eq!(run(&mut v), excp::DEBUG);
    assert_eq!(reg(&v, 1), 0x48);
    assert_eq!(reg(&v, 2), 3);
    // The device saw each access exactly once.
    assert_eq!(w.dev.reads.load(Ordering::SeqCst), 1);
    assert_eq!(*w.dev.writes.lock().unwrap(), vec![(0x10, 3)]);
}

#[test]
fn breakpoints() {
    let w = world(JitConfig::default());
    let mut a = Asm::new(0x1000);
    a.li(1, 1).li(2, 2).li(3, 3).stop();
    a.load(&w.as_);
    let mut v = w.vcpu(0x1000);
    v.cpu().breakpoint_insert(0x1008, ruvm_jit::bp::GDB);
    assert_eq!(run(&mut v), excp::DEBUG);
    assert_eq!(pc(&v), 0x1008);
    assert_eq!(reg(&v, 2), 2);
    assert_eq!(reg(&v, 3), 0);
    assert!(v.cpu().breakpoint_remove(0x1008, ruvm_jit::bp::GDB));
    assert_eq!(run(&mut v), excp::DEBUG);
    assert_eq!(pc(&v), 0x100c);
    assert_eq!(reg(&v, 3), 3);
}

#[test]
fn single_step() {
    let w = world(JitConfig::default());
    let mut a = Asm::new(0x1000);
    a.li(1, 1).li(2, 2).stop();
    a.load(&w.as_);
    let mut v = w.vcpu(0x1000);
    v.core.singlestep_enabled = true;
    assert_eq!(run(&mut v), excp::DEBUG);
    assert_eq!(pc(&v), 0x1004);
    assert_eq!(run(&mut v), excp::DEBUG);
    assert_eq!(pc(&v), 0x1008);
    assert_eq!(reg(&v, 2), 2);
}

/// Code that spins forever at 0x1004 after enabling interrupts, and an interrupt vector that
/// stops.
fn spin_with_irq(w: &World) {
    let mut a = Asm::new(0x1000);
    a.i(EI, 0, 0, 0);
    let spin = a.pc();
    a.j(spin);
    a.load(&w.as_);
    let mut h = Asm::new(IRQ_VECTOR);
    h.li(3, 42).stop();
    h.load(&w.as_);
}

#[test]
fn interrupt_breaks_a_chained_loop() {
    let w = world(JitConfig { mttcg: true, ..JitConfig::default() });
    spin_with_irq(&w);
    let mut v = w.vcpu(0x1000);
    let shared = v.shared().clone();
    let t = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(30));
        shared.cpu_interrupt(interrupt::HARD);
    });
    // The first cpu_exec may return EXCP_INTERRUPT for the kick before the vector runs.
    let mut r = run(&mut v);
    while r == excp::INTERRUPT {
        r = run(&mut v);
    }
    t.join().unwrap();
    assert_eq!(r, excp::DEBUG);
    assert_eq!(reg(&v, 3), 42);
    assert_eq!(reg(&v, 7), 0x1004);
    assert_eq!(w.ops.irqs.load(Ordering::Relaxed), 1);
}

#[test]
fn halt_waits_for_an_interrupt() {
    let w = world(JitConfig { mttcg: true, ..JitConfig::default() });
    let mut a = Asm::new(0x1000);
    a.i(EI, 0, 0, 0).i(HLT, 0, 0, 0).stop();
    a.load(&w.as_);
    let mut h = Asm::new(IRQ_VECTOR);
    h.li(3, 7).stop();
    h.load(&w.as_);
    let v = w.vcpu(0x1000);
    let threads = start_vcpus(&w.jit, vec![v]);
    let shared = threads.cpus()[0].clone();
    let start = std::time::Instant::now();
    while shared.halted.load(Ordering::Acquire) == 0 {
        assert!(start.elapsed() < Duration::from_secs(10), "the vCPU never halted");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(!shared.stopped.load(Ordering::Acquire));
    shared.cpu_interrupt(interrupt::HARD);
    assert!(threads.wait_all_stopped(Duration::from_secs(10)));
    let v = threads.stop_and_join().pop().unwrap();
    assert_eq!(reg(&v, 3), 7);
    assert_eq!(reg(&v, 7), 0x1008);
}

/// Each vCPU adds 1 to the word at 0x5000, `n` times, with the INC instruction.
fn atomic_counter_program(w: &World, n: i32) {
    let mut a = Asm::new(0x1000);
    a.li(5, 0x5000).li(2, n);
    let head = a.pc();
    a.i(INC, 0, 5, 0).addi(2, 2, -1).bnez(2, head).stop();
    a.load(&w.as_);
}

#[test]
fn mttcg_step_atomic_under_contention() {
    let w = world(JitConfig { mttcg: true, ..JitConfig::default() });
    atomic_counter_program(&w, 200);
    let vcpus: Vec<Vcpu> = (0..4).map(|_| w.vcpu(0x1000)).collect();
    let threads = start_vcpus(&w.jit, vcpus);
    assert!(threads.wait_all_stopped(Duration::from_secs(60)));
    let vcpus = threads.stop_and_join();
    assert_eq!(vcpus.len(), 4);
    assert_eq!(w.read32(0x5000), 800);
    for v in &vcpus {
        assert_eq!(reg(v, 2), 0);
    }
}

/// Several vCPUs add `1 << 16` to the word at 0x5000 with an atomic fetch-and-add while
/// another stores to its low half with plain 16-bit stores, as a spinlock's owner releases the
/// lock byte while other vCPUs queue on the same word. The atomics must not undo a plain store
/// and no increment may be lost.
#[test]
fn mttcg_atomics_do_not_lose_plain_stores() {
    const N: i32 = 3000;
    const ADDERS: usize = 3;
    let w = world(JitConfig { mttcg: true, ..JitConfig::default() });
    // Adders, at 0x1000: r1 = 0x10000, then N times fetch_add([r5], r1).
    let mut a = Asm::new(0x1000);
    a.li(5, 0x5000).li(1, 0x4000).i(ADD, 1, 1, 0).i(ADD, 1, 1, 0).li(2, N);
    let head = a.pc();
    a.addi(3, 1, 0).i(XADD, 3, 5, 0).addi(2, 2, -1).bnez(2, head).stop();
    a.load(&w.as_);
    // The storer, at 0x2000: for r4 in 1..=N, store r4 to the low half and read it back,
    // counting mismatches in r6.
    let mut b = Asm::new(0x2000);
    b.li(5, 0x5000).li(2, N).li(4, 0).li(6, 0);
    let head = b.pc();
    b.addi(4, 4, 1).i(ST16, 4, 5, 0).i(LD16, 3, 5, 0).i(SUB, 3, 4, 0);
    let check = b.pc();
    b.bnez(3, check + 16).addi(2, 2, -1).bnez(2, head).stop();
    b.addi(6, 6, 1).addi(2, 2, -1).bnez(2, head).stop();
    b.load(&w.as_);

    let mut vcpus: Vec<Vcpu> = (0..ADDERS).map(|_| w.vcpu(0x1000)).collect();
    vcpus.push(w.vcpu(0x2000));
    let threads = start_vcpus(&w.jit, vcpus);
    assert!(threads.wait_all_stopped(Duration::from_secs(120)));
    let vcpus = threads.stop_and_join();
    let word = w.read32(0x5000);
    assert_eq!(word >> 16, (ADDERS as u32 * N as u32) & 0xffff, "lost atomic increments");
    assert_eq!(word & 0xffff, N as u32, "the last plain store was undone");
    let storer = vcpus.iter().find(|v| reg(v, 4) == N as u64).expect("the storer finished");
    assert_eq!(reg(storer, 6), 0, "plain stores were undone by an atomic");
}

#[test]
fn exclusive_work_stops_every_other_vcpu() {
    let w = world(JitConfig { mttcg: true, ..JitConfig::default() });
    // Every vCPU increments its own counter forever, non atomically.
    let mut a = Asm::new(0x1000);
    let head = a.pc();
    a.i(LD, 1, 5, 0).addi(1, 1, 1).i(ST, 1, 5, 0).j(head);
    a.load(&w.as_);
    let vcpus: Vec<Vcpu> = (0..4u64)
        .map(|i| {
            let mut v = w.vcpu(0x1000);
            wr64(&mut v.env, reg_off(5), 0x5000 + 8 * i);
            v
        })
        .collect();
    let threads = start_vcpus(&w.jit, vcpus);
    let as_ = w.as_.clone();
    let snapshot =
        move || -> Vec<u32> { (0..4).map(|i| as_.read_u32(0x5000 + 8 * i, U).0).collect() };

    // Let them run.
    let start = std::time::Instant::now();
    while snapshot().contains(&0) {
        assert!(start.elapsed() < Duration::from_secs(10), "the vCPUs do not run");
        std::thread::sleep(Duration::from_millis(1));
    }

    // From an outside thread.
    for _ in 0..5 {
        w.jit.start_exclusive();
        let before = snapshot();
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(snapshot(), before);
        w.jit.end_exclusive();
    }

    // From work items on the vCPUs themselves.
    let inside = Arc::new(AtomicUsize::new(0));
    let done = Arc::new(AtomicUsize::new(0));
    for c in threads.cpus() {
        for _ in 0..3 {
            let inside = inside.clone();
            let done = done.clone();
            let snap = snapshot.clone();
            c.async_safe_run_on_cpu(move |_cpu| {
                assert_eq!(inside.fetch_add(1, Ordering::SeqCst), 0);
                let before = snap();
                std::thread::sleep(Duration::from_millis(2));
                assert_eq!(snap(), before);
                inside.fetch_sub(1, Ordering::SeqCst);
                done.fetch_add(1, Ordering::SeqCst);
            });
        }
    }
    let start = std::time::Instant::now();
    while done.load(Ordering::SeqCst) < 12 {
        assert!(start.elapsed() < Duration::from_secs(20), "work items did not run");
        std::thread::sleep(Duration::from_millis(1));
    }

    // run_on_cpu waits for the item.
    let seen = Arc::new(AtomicU64::new(0));
    let s = seen.clone();
    threads.cpus()[2].run_on_cpu(move |cpu| {
        s.store(rd64(cpu.env, reg_off(5)), Ordering::SeqCst);
    });
    assert_eq!(seen.load(Ordering::SeqCst), 0x5010);

    let vcpus = threads.stop_and_join();
    assert_eq!(vcpus.len(), 4);
}

/// A program of `blocks` distinct blocks run `loops` times, so that a small code buffer fills.
fn many_blocks_program(w: &World, blocks: u64, loops: i32) {
    let mut a = Asm::new(0x1000);
    a.li(1, 0).li(2, loops);
    let head = a.pc();
    for _ in 0..blocks {
        let next = a.pc() + 8;
        a.addi(1, 1, 1).j(next);
    }
    a.addi(2, 2, -1).bnez(2, head).stop();
    a.load(&w.as_);
}

#[test]
fn tb_flush_under_mttcg() {
    let config = JitConfig { mttcg: true, code_gen_buffer_size: 16 << 10, ..JitConfig::default() };
    let w = world(config);
    many_blocks_program(&w, 60, 40);
    let vcpus: Vec<Vcpu> = (0..4).map(|_| w.vcpu(0x1000)).collect();
    let threads = start_vcpus(&w.jit, vcpus);
    assert!(threads.wait_all_stopped(Duration::from_secs(120)));
    let vcpus = threads.stop_and_join();
    for v in &vcpus {
        assert_eq!(reg(v, 1), 60 * 40);
    }
    assert!(w.jit.tb_flush_count() > 0, "the code buffer never filled");
    assert!(w.jit.code_gen_used() <= 16 << 10);
}

#[test]
fn tb_flush_serial() {
    let config = JitConfig { code_gen_buffer_size: 16 << 10, ..JitConfig::default() };
    let w = world(config);
    many_blocks_program(&w, 60, 5);
    let mut v = w.vcpu(0x1000);
    let mut r = run(&mut v);
    while r == excp::INTERRUPT {
        r = run(&mut v);
    }
    assert_eq!(r, excp::DEBUG);
    assert_eq!(reg(&v, 1), 300);
    assert!(w.jit.tb_flush_count() > 0);
}

#[test]
fn round_robin_kicks_a_spinning_vcpu() {
    let w = world(JitConfig::default());
    // vCPU 0 spins until the flag at 0x5000 is set; vCPU 1 sets it.
    let mut a = Asm::new(0x1000);
    a.li(5, 0x5000);
    let head = a.pc();
    a.i(LD, 1, 5, 0);
    let over = a.pc() + 8;
    a.bnez(1, over).j(head).stop();
    a.load(&w.as_);
    let mut b = Asm::new(0x2000);
    b.li(5, 0x5000).li(1, 1).i(ST, 1, 5, 0).stop();
    b.load(&w.as_);
    let v0 = w.vcpu(0x1000);
    let v1 = w.vcpu(0x2000);
    let threads = start_vcpus(&w.jit, vec![v0, v1]);
    assert!(threads.wait_all_stopped(Duration::from_secs(20)));
    let vcpus = threads.stop_and_join();
    assert_eq!(reg(&vcpus[0], 1), 1);
    assert_eq!(pc(&vcpus[0]), 0x1010);
    assert_eq!(pc(&vcpus[1]), 0x200c);
}

#[test]
fn round_robin_step_atomic_and_flush_all_cpus() {
    let w = world(JitConfig::default());
    atomic_counter_program(&w, 50);
    let vcpus: Vec<Vcpu> = (0..3).map(|_| w.vcpu(0x1000)).collect();
    let threads = start_vcpus(&w.jit, vcpus);
    assert!(threads.wait_all_stopped(Duration::from_secs(20)));
    let mut vcpus = threads.stop_and_join();
    assert_eq!(w.read32(0x5000), 150);
    let count = |v: &mut Vcpu| {
        let (full, part) = cputlb::tlb_flush_counts(&v.cpu());
        full + part
    };
    let before: Vec<u64> = vcpus.iter_mut().map(count).collect();
    cputlb::tlb_flush_all_cpus_synced(&mut vcpus[0].cpu());
    // Every vCPU flushes when it runs its queued work, the source one as safe work.
    for v in vcpus.iter_mut() {
        v.cpu().process_queued_cpu_work();
    }
    for (v, b) in vcpus.iter_mut().zip(before) {
        assert_eq!(count(v), b + 1);
    }
}
