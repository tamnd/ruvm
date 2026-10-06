// SPDX-License-Identifier: GPL-2.0-or-later

//! Hand assembled x86 snippets run through `ruvm-jit` on the interpreter backend.
//!
//! Every snippet is a byte list with the assembly it encodes beside it, and ends in HLT,
//! which stops `cpu_exec()` with `EXCP_HLT`. The expected values follow the Intel SDM.
//!
//! The memory map: RAM from 0 to 4 MiB; the real mode IVT at 0 with a HLT handler per
//! vector at 0x500 + n; code at 0x1000; the GDT at 0x3000; the long mode IDT at 0x4000 with
//! a HLT handler per vector at 0x5000 + 16 n; data at 0x8000; the stack below 0x30000 and
//! the page tables at 0x10000. The page tables map 0 to 2 MiB read/write and 2 to 4 MiB
//! read only, both with 2 MiB pages; everything above is not present.

use std::sync::Arc;

use ruvm_jit::cpu_exec::cpu_exec;
use ruvm_jit::{Jit, Vcpu, excp};
use ruvm_mem::{AddressSpace, MemTxAttrs, MemorySystem};
use ruvm_target_x86::cpuid::{Accel, X86Cpu};
use ruvm_target_x86::state::{
    CR0_ET_MASK, CR0_NE_MASK, CR0_PE_MASK, CR0_PG_MASK, CR0_WP_MASK, CR4_PAE_MASK, CR4_PKE_MASK,
    CR4_PKS_MASK, DESC_A_MASK, DESC_B_MASK, DESC_CS_MASK, DESC_G_MASK, DESC_L_MASK, DESC_P_MASK,
    DESC_R_MASK, DESC_S_MASK, DESC_W_MASK, HF_CS64_MASK, HF_LMA_MASK, IrqchipMode, MSR_EFER_LMA,
    MSR_EFER_LME, MSR_EFER_SCE, R_CS, R_EAX, R_EBX, R_ECX, R_EDI, R_EDX, R_ESI, R_ESP, R_SS,
    ResetConfig, SegmentCache, X86CpuState,
};
use ruvm_target_x86::tcg::{X86, create_vcpu, new_jit, save_vcpu};

const U: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const RAM_SIZE: u64 = 0x40_0000;
const CODE: u64 = 0x1000;
const GDT: u64 = 0x3000;
const IDT: u64 = 0x4000;
const HANDLERS: u64 = 0x5000;
const REAL_HANDLERS: u64 = 0x500;
const DATA: u64 = 0x8000;
const STACK: u64 = 0x3_0000;
const PML4: u64 = 0x1_0000;

const CF: u64 = 0x001;
const PF: u64 = 0x004;
const AF: u64 = 0x010;
const ZF: u64 = 0x040;
const SF: u64 = 0x080;
const DF: u64 = 0x400;
const OF: u64 = 0x800;
const ARITH: u64 = CF | PF | AF | ZF | SF | OF;

const CODE64_FLAGS: u32 = DESC_P_MASK
    | DESC_S_MASK
    | DESC_CS_MASK
    | DESC_R_MASK
    | DESC_A_MASK
    | DESC_L_MASK
    | DESC_G_MASK;
const CODE32_FLAGS: u32 = DESC_P_MASK
    | DESC_S_MASK
    | DESC_CS_MASK
    | DESC_R_MASK
    | DESC_A_MASK
    | DESC_B_MASK
    | DESC_G_MASK;
const DATA32_FLAGS: u32 =
    DESC_P_MASK | DESC_S_MASK | DESC_W_MASK | DESC_A_MASK | DESC_B_MASK | DESC_G_MASK;

struct World {
    _sys: MemorySystem,
    as_: Arc<AddressSpace>,
    jit: Arc<Jit>,
    x86: Arc<X86>,
}

impl World {
    fn new() -> World {
        World::with_model(X86::qemu64())
    }

    fn with_model(x86: X86) -> World {
        let sys = MemorySystem::new();
        let root = sys.new_container("system", 1 << 64).unwrap();
        let as_ = sys.address_space_init(root, "memory").unwrap();
        let ram = sys.new_ram("ram", RAM_SIZE).unwrap();
        sys.add_subregion(root, 0, ram).unwrap();
        let w = World { _sys: sys, as_, jit: new_jit(), x86: Arc::new(x86) };
        // Real mode: IVT entry n points at a HLT at 0x500 + n.
        for n in 0..32u64 {
            let ent = (REAL_HANDLERS + n) as u32;
            w.write(n * 4, &ent.to_le_bytes());
            w.write(REAL_HANDLERS + n, &[0xf4]);
        }
        // GDT: null, 0x08 64-bit code, 0x10 data, 0x18 32-bit code.
        w.w64(GDT, 0);
        w.w64(GDT + 8, 0x00af_9b00_0000_ffff);
        w.w64(GDT + 16, 0x00cf_9300_0000_ffff);
        w.w64(GDT + 24, 0x00cf_9b00_0000_ffff);
        // Long mode IDT: vector n jumps to a HLT at 0x5000 + 16 n.
        for n in 0..32u64 {
            let h = HANDLERS + n * 16;
            let lo = (h & 0xffff) | (0x08 << 16) | (0x8e00 << 32) | ((h >> 16) & 0xffff) << 48;
            w.w64(IDT + n * 16, lo);
            w.w64(IDT + n * 16 + 8, h >> 32);
            w.write(h, &[0xf4]);
        }
        // Page tables.
        w.w64(PML4, (PML4 + 0x1000) | 3);
        w.w64(PML4 + 0x1000, (PML4 + 0x2000) | 3);
        w.w64(PML4 + 0x2000, 0x83);
        w.w64(PML4 + 0x2008, 0x20_0000 | 0x81);
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

    fn w64(&self, addr: u64, v: u64) {
        self.write(addr, &v.to_le_bytes());
    }

    fn r64(&self, addr: u64) -> u64 {
        u64::from_le_bytes(self.read(addr, 8).try_into().unwrap())
    }

    fn reset() -> X86CpuState {
        let mut st = X86CpuState::new_reset(&ResetConfig {
            is_bsp: true,
            kvm: false,
            irqchip: IrqchipMode::Off,
            cpuid_version: 0x0006_0fb1,
            has_monitor: false,
        });
        st.rflags = 2;
        st.rip = CODE;
        st
    }

    /// Real mode with every segment at 0.
    fn real() -> X86CpuState {
        let mut st = World::reset();
        let data = DESC_P_MASK | DESC_S_MASK | DESC_W_MASK | DESC_A_MASK;
        let code = DESC_P_MASK | DESC_S_MASK | DESC_CS_MASK | DESC_R_MASK | DESC_A_MASK;
        for s in 0..6 {
            st.load_seg_cache(s, 0, 0, 0xffff, if s == R_CS { code } else { data });
        }
        st.idt = SegmentCache { selector: 0, base: 0, limit: 0x3ff, flags: 0 };
        st.regs[R_ESP] = 0x7000;
        st
    }

    /// Flat 32-bit protected mode without paging.
    fn prot32() -> X86CpuState {
        let mut st = World::reset();
        st.update_cr0(CR0_PE_MASK | CR0_ET_MASK | CR0_NE_MASK);
        st.gdt = SegmentCache { selector: 0, base: GDT, limit: 0x1f, flags: 0 };
        st.load_seg_cache(R_CS, 0x18, 0, 0xffff_ffff, CODE32_FLAGS);
        for s in [0, 2, 3, 4, 5] {
            st.load_seg_cache(s, 0x10, 0, 0xffff_ffff, DATA32_FLAGS);
        }
        st.regs[R_ESP] = STACK;
        st
    }

    /// 64-bit mode at CPL 0 with paging.
    fn long64() -> X86CpuState {
        let mut st = World::reset();
        st.cr4 = CR4_PAE_MASK;
        st.efer = MSR_EFER_LME;
        st.cr3 = PML4;
        st.update_cr0(CR0_PE_MASK | CR0_ET_MASK | CR0_NE_MASK | CR0_WP_MASK | CR0_PG_MASK);
        assert_ne!(st.hflags & HF_LMA_MASK, 0);
        st.gdt = SegmentCache { selector: 0, base: GDT, limit: 0x1f, flags: 0 };
        st.idt = SegmentCache { selector: 0, base: IDT, limit: 0x1ff, flags: 0 };
        st.load_seg_cache(R_CS, 0x08, 0, 0xffff_ffff, CODE64_FLAGS);
        for s in [0, 2, 3, 4, 5] {
            st.load_seg_cache(s, 0x10, 0, 0xffff_ffff, DATA32_FLAGS);
        }
        st.regs[R_ESP] = STACK;
        st
    }

    /// Run `code` at [`CODE`] from `st` with `regs` set, until the vCPU halts.
    fn run(&self, mut st: X86CpuState, regs: &[(usize, u64)], code: &[u8]) -> X86CpuState {
        self.write(CODE, code);
        for &(r, v) in regs {
            st.regs[r] = v;
        }
        let mut v: Vcpu = create_vcpu(&self.jit, self.x86.clone(), self.as_.clone(), &st);
        let mut halted = false;
        for _ in 0..10_000 {
            if cpu_exec(&mut v.cpu()) == excp::HLT {
                halted = true;
                break;
            }
        }
        assert!(halted, "the vCPU did not halt");
        save_vcpu(&v, &mut st);
        st
    }
}

/// Run a 64-bit snippet in a fresh world.
fn run64(regs: &[(usize, u64)], code: &[u8]) -> X86CpuState {
    World::new().run(World::long64(), regs, code)
}

/// Run a 64-bit snippet on an EPYC model, which has POPCNT and ABM.
fn run_epyc(regs: &[(usize, u64)], code: &[u8]) -> X86CpuState {
    let mut m = X86Cpu::new("EPYC", Accel::Tcg).unwrap();
    let _ = m.realize();
    assert!(m.has_feature("popcnt") && m.has_feature("abm"));
    World::with_model(X86::new(m)).run(World::long64(), regs, code)
}

/// The vector of the long mode exception handler a vCPU halted in.
fn vector64(st: &X86CpuState) -> u64 {
    assert!(st.rip > HANDLERS && st.rip < HANDLERS + 32 * 16, "rip {:#x}", st.rip);
    (st.rip - HANDLERS - 1) / 16
}

#[track_caller]
fn check(st: &X86CpuState, rax: u64, mask: u64, flags: u64) {
    assert_eq!(st.regs[R_EAX], rax, "rax {:#x}", st.regs[R_EAX]);
    assert_eq!(st.rflags & mask, flags, "rflags {:#x}", st.rflags);
}

#[test]
fn add_flags_8_16_32_64() {
    // add al, 1
    check(&run64(&[(R_EAX, 0x7f)], &[0x04, 0x01, 0xf4]), 0x80, ARITH, OF | SF | AF);
    // add ax, 1: the upper bits of RAX stay.
    let st = run64(&[(R_EAX, 0x1_ffff)], &[0x66, 0x05, 0x01, 0x00, 0xf4]);
    check(&st, 0x1_0000, ARITH, CF | ZF | PF | AF);
    // sub eax, 1: a 32-bit result is zero extended.
    let st = run64(&[(R_EAX, 0xffff_0000_0000_0000)], &[0x2d, 1, 0, 0, 0, 0xf4]);
    check(&st, 0xffff_ffff, ARITH, CF | SF | AF | PF);
    // add rax, 1
    let st = run64(&[(R_EAX, i64::MAX as u64)], &[0x48, 0x83, 0xc0, 0x01, 0xf4]);
    check(&st, 1 << 63, ARITH, OF | SF | AF | PF);
    // cmp rax, rbx with equal values.
    let st = run64(&[(R_EAX, 5), (R_EBX, 5)], &[0x48, 0x39, 0xd8, 0xf4]);
    check(&st, 5, ARITH, ZF | PF);
}

#[test]
fn adc_sbb_inc_dec() {
    // stc; adc al, 0
    check(&run64(&[(R_EAX, 0xff)], &[0xf9, 0x14, 0x00, 0xf4]), 0, ARITH, CF | ZF | AF | PF);
    // stc; sbb eax, 0
    let st = run64(&[(R_EAX, 0)], &[0xf9, 0x83, 0xd8, 0x00, 0xf4]);
    check(&st, 0xffff_ffff, ARITH, CF | SF | AF | PF);
    // stc; inc al: CF is kept.
    check(&run64(&[(R_EAX, 0xff)], &[0xf9, 0xfe, 0xc0, 0xf4]), 0, ARITH, CF | ZF | AF | PF);
    // clc; dec word [rbx]
    let w = World::new();
    w.write(DATA, &[0x00, 0x80]);
    let st = w.run(World::long64(), &[(R_EBX, DATA)], &[0xf8, 0x66, 0xff, 0x0b, 0xf4]);
    assert_eq!(w.read(DATA, 2), [0xff, 0x7f]);
    assert_eq!(st.rflags & ARITH, OF | AF | PF);
}

#[test]
fn logic_and_xor_clear() {
    // xor eax, eax
    check(&run64(&[(R_EAX, u64::MAX)], &[0x31, 0xc0, 0xf4]), 0, ARITH, ZF | PF);
    // stc; and rax, rbx: CF and OF are cleared.
    let st = run64(&[(R_EAX, 0xf0), (R_EBX, 0x3c)], &[0xf9, 0x48, 0x21, 0xd8, 0xf4]);
    check(&st, 0x30, ARITH, PF);
    // test al, 0x80
    check(&run64(&[(R_EAX, 0x80)], &[0xa8, 0x80, 0xf4]), 0x80, ARITH, SF);
}

#[test]
fn shifts_and_rotates() {
    // shl al, 1
    check(&run64(&[(R_EAX, 0x81)], &[0xd0, 0xe0, 0xf4]), 0x02, CF | OF | ZF | SF, CF | OF);
    // sar eax, 4
    let st = run64(&[(R_EAX, 0x8000_0010)], &[0xc1, 0xf8, 0x04, 0xf4]);
    check(&st, 0xf800_0001, CF | ZF | SF, SF);
    // shr rax, cl with cl = 0: nothing changes, flags included.
    let st = run64(&[(R_EAX, 3), (R_ECX, 0)], &[0xf9, 0x48, 0xd3, 0xe8, 0xf4]);
    check(&st, 3, CF, CF);
    // rol rax, cl
    let st = run64(&[(R_EAX, 0x8000_0000_0000_0001), (R_ECX, 4)], &[0x48, 0xd3, 0xc0, 0xf4]);
    check(&st, 0x18, CF, 0);
    // ror al, 1
    check(&run64(&[(R_EAX, 0x01)], &[0xd0, 0xc8, 0xf4]), 0x80, CF | OF, CF | OF);
    // stc; rcl al, 1
    check(&run64(&[(R_EAX, 0x80)], &[0xf9, 0xd0, 0xd0, 0xf4]), 0x01, CF | OF, CF | OF);
    // clc; rcr ax, 3
    let st = run64(&[(R_EAX, 0x0005)], &[0xf8, 0x66, 0xc1, 0xd8, 0x03, 0xf4]);
    check(&st, 0x4000, CF, CF);
    // shrd rax, rbx, 4
    let st = run64(&[(R_EAX, 0x10), (R_EBX, 0xf)], &[0x48, 0x0f, 0xac, 0xd8, 0x04, 0xf4]);
    check(&st, 0xf000_0000_0000_0001, CF, 0);
    // shld eax, ebx, cl
    let st =
        run64(&[(R_EAX, 0x8000_0001), (R_EBX, 0xf000_0000), (R_ECX, 4)], &[0x0f, 0xa5, 0xd8, 0xf4]);
    check(&st, 0x1f, CF, 0);
}

#[test]
fn mul_imul_div() {
    // mul rbx
    let st = run64(&[(R_EAX, 1 << 63), (R_EBX, 4)], &[0x48, 0xf7, 0xe3, 0xf4]);
    check(&st, 0, CF | OF, CF | OF);
    assert_eq!(st.regs[R_EDX], 2);
    // mul bl: the result fits in AL.
    check(&run64(&[(R_EAX, 7), (R_EBX, 3)], &[0xf6, 0xe3, 0xf4]), 21, CF | OF, 0);
    // imul rax, rbx, 7
    check(
        &run64(&[(R_EBX, (-6i64) as u64)], &[0x48, 0x6b, 0xc3, 0x07, 0xf4]),
        (-42i64) as u64,
        CF | OF,
        0,
    );
    // imul ecx: EDX:EAX = -1 * 0x40000000 * 2 does not fit in EAX.
    let st = run64(&[(R_EAX, 0x8000_0000), (R_ECX, 2)], &[0xf7, 0xe9, 0xf4]);
    check(&st, 0, CF | OF, CF | OF);
    assert_eq!(st.regs[R_EDX], 0xffff_ffff);
    // div ecx
    let st = run64(&[(R_EAX, 100), (R_EDX, 1), (R_ECX, 7)], &[0xf7, 0xf1, 0xf4]);
    assert_eq!(st.regs[R_EAX], 0x1_0000_0064 / 7);
    assert_eq!(st.regs[R_EDX], 0x1_0000_0064 % 7);
    // idiv bl
    let st = run64(&[(R_EAX, (-7i16) as u16 as u64), (R_EBX, 2)], &[0xf6, 0xfb, 0xf4]);
    assert_eq!(st.regs[R_EAX] & 0xffff, 0xfffd);
}

#[test]
fn divide_error_long_mode() {
    // div rbx with rbx = 0 raises #DE with the address of the DIV.
    let w = World::new();
    let st = w.run(World::long64(), &[(R_EAX, 1), (R_EBX, 0)], &[0x48, 0xf7, 0xf3, 0xf4]);
    assert_eq!(vector64(&st), 0);
    assert_eq!(w.r64(st.regs[R_ESP]), CODE);
    assert_eq!(w.r64(st.regs[R_ESP] + 8), 0x08);
    // idiv cl overflowing the quotient is #DE too.
    let w = World::new();
    let st = w.run(World::long64(), &[(R_EAX, 0x8000), (R_ECX, 1)], &[0x90, 0xf6, 0xf9, 0xf4]);
    assert_eq!(vector64(&st), 0);
    assert_eq!(w.r64(st.regs[R_ESP]), CODE + 1);
}

#[test]
fn div_idiv_64() {
    const DIV_RBX: [u8; 4] = [0x48, 0xf7, 0xf3, 0xf4];
    const IDIV_RBX: [u8; 4] = [0x48, 0xf7, 0xfb, 0xf4];
    let div = |hi: u64, lo: u64, d: u64| run64(&[(R_EAX, lo), (R_EDX, hi), (R_EBX, d)], &DIV_RBX);
    let idiv = |hi: i64, lo: i64, d: i64| -> X86CpuState {
        run64(&[(R_EAX, lo as u64), (R_EDX, hi as u64), (R_EBX, d as u64)], &IDIV_RBX)
    };
    // The high half is zero, and then not.
    let st = div(0, 100, 7);
    assert_eq!((st.regs[R_EAX], st.regs[R_EDX]), (14, 2));
    let n = 3u128 << 64 | 5;
    let st = div(3, 5, 10);
    assert_eq!((st.regs[R_EAX], st.regs[R_EDX]), ((n / 10) as u64, (n % 10) as u64));
    // The quotient does not fit, and a zero divisor.
    assert_eq!(vector64(&div(10, 0, 10)), 0);
    assert_eq!(vector64(&div(0, 1, 0)), 0);
    // A sign extended dividend.
    let st = idiv(-1, -100, 7);
    assert_eq!((st.regs[R_EAX] as i64, st.regs[R_EDX] as i64), (-14, -2));
    let st = idiv(0, 100, -7);
    assert_eq!((st.regs[R_EAX] as i64, st.regs[R_EDX] as i64), (-14, 2));
    assert_eq!(vector64(&idiv(-1, i64::MIN, -1)), 0);
    assert_eq!(vector64(&idiv(0, 5, 0)), 0);
    // A dividend that needs both halves: -2^64 / -3.
    let n = -(1i128 << 64);
    let st = idiv(-1, 0, -3);
    assert_eq!(st.regs[R_EAX] as i64 as i128, n / -3);
    assert_eq!(st.regs[R_EDX] as i64 as i128, n % -3);
    // 2^64 / 1 does not fit, nor does anything over zero.
    assert_eq!(vector64(&idiv(1, 0, 1)), 0);
    assert_eq!(vector64(&idiv(5, 0, 0)), 0);
}

#[test]
fn div_idiv_32() {
    // The upper halves of RAX and RDX are ignored and the results zero extended.
    const JUNK: u64 = 0xdead_beef << 32;
    let run = |code: &[u8], hi: u32, lo: u32, d: u32| {
        let regs = [
            (R_EAX, JUNK | u64::from(lo)),
            (R_EDX, JUNK | u64::from(hi)),
            (R_EBX, JUNK | u64::from(d)),
        ];
        run64(&regs, code)
    };
    let div = |hi: u32, lo: u32, d: u32| run(&[0xf7, 0xf3, 0xf4], hi, lo, d);
    let idiv = |hi: i32, lo: i32, d: i32| run(&[0xf7, 0xfb, 0xf4], hi as u32, lo as u32, d as u32);
    let res = |st: &X86CpuState| (st.regs[R_EAX], st.regs[R_EDX]);
    let n = 3u64 << 32 | 5;
    assert_eq!(res(&div(0, 100, 7)), (14, 2));
    assert_eq!(res(&div(3, 5, 10)), (n / 10, n % 10));
    assert_eq!(res(&div(9, u32::MAX, 10)), (0xffff_ffff, 9));
    assert_eq!(vector64(&div(10, 0, 10)), 0);
    assert_eq!(vector64(&div(0, 1, 0)), 0);
    let u = |v: i64| u64::from(v as u32);
    assert_eq!(res(&idiv(-1, -100, 7)), (u(-14), u(-2)));
    assert_eq!(res(&idiv(0, 100, -7)), (u(-14), u(2)));
    // -2^32 / -3, and -2^32 / 2, the most negative quotient.
    assert_eq!(res(&idiv(-1, 0, -3)), (u(-(1i64 << 32) / -3), u(-(1i64 << 32) % -3)));
    assert_eq!(res(&idiv(-1, 0, 2)), (u(i64::from(i32::MIN)), 0));
    // Overflow, -1 and zero divisors.
    assert_eq!(vector64(&idiv(1, 0, 1)), 0);
    assert_eq!(vector64(&idiv(-1, i32::MIN, -1)), 0);
    assert_eq!(res(&idiv(-1, -6, -1)), (6, 0));
    assert_eq!(vector64(&idiv(0, 5, 0)), 0);
}

#[test]
fn divide_error_real_mode() {
    // div bl with bl = 0: the IVT handler runs with FLAGS, CS and IP pushed.
    let w = World::new();
    let st = w.run(World::real(), &[(R_EAX, 0x10)], &[0xf6, 0xf3, 0xf4]);
    assert_eq!(st.rip, REAL_HANDLERS + 1);
    let sp = st.regs[R_ESP];
    assert_eq!(sp, 0x7000 - 6);
    assert_eq!(w.read(sp, 4), [0x00, 0x10, 0x00, 0x00]);
}

#[test]
fn flags_16_bit_real_mode() {
    let w = World::new();
    // add ax, 0x7fff
    let st = w.run(World::real(), &[(R_EAX, 1)], &[0x05, 0xff, 0x7f, 0xf4]);
    check(&st, 0x8000, ARITH, OF | SF | AF | PF);
    // 66 prefix gives a 32-bit add in real mode: add eax, 1
    let w = World::new();
    let st = w.run(World::real(), &[(R_EAX, 0xffff)], &[0x66, 0x83, 0xc0, 0x01, 0xf4]);
    check(&st, 0x1_0000, ARITH, AF | PF);
}

#[test]
fn bit_instructions() {
    // bt rax, rbx
    check(&run64(&[(R_EAX, 0x10), (R_EBX, 4)], &[0x48, 0x0f, 0xa3, 0xd8, 0xf4]), 0x10, CF, CF);
    // btc rax, rbx
    check(&run64(&[(R_EAX, 0x10), (R_EBX, 5)], &[0x48, 0x0f, 0xbb, 0xd8, 0xf4]), 0x30, CF, 0);
    // btr eax, 4
    check(&run64(&[(R_EAX, 0x10)], &[0x0f, 0xba, 0xf0, 0x04, 0xf4]), 0, CF, CF);
    // bts dword [rbx], ecx with a bit offset past the dword.
    let w = World::new();
    w.write(DATA, &[0; 8]);
    w.run(World::long64(), &[(R_EBX, DATA), (R_ECX, 33)], &[0x0f, 0xab, 0x0b, 0xf4]);
    assert_eq!(w.r64(DATA), 2 << 32);
    // bsf rax, rbx
    check(&run64(&[(R_EBX, 0x100)], &[0x48, 0x0f, 0xbc, 0xc3, 0xf4]), 8, ZF, 0);
    // bsr eax, ebx with ebx = 0: ZF and the destination is unchanged.
    check(&run64(&[(R_EAX, 77), (R_EBX, 0)], &[0x0f, 0xbd, 0xc3, 0xf4]), 77, ZF, ZF);
    // A zero 32-bit source leaves all 64 bits of the destination, as on hardware
    // (risu x86_int: bsf ebx, r14d).
    let st = run64(&[(R_EBX, 0xb6ab_e996_aa1b_f040), (14, 0)], &[0x41, 0x0f, 0xbc, 0xde, 0xf4]);
    assert_eq!(st.regs[R_EBX], 0xb6ab_e996_aa1b_f040);
    assert_ne!(st.rflags & ZF, 0);
    let st = run64(&[(R_EAX, 0xffff_ffff_0000_0005), (R_EBX, 0)], &[0x0f, 0xbd, 0xc3, 0xf4]);
    assert_eq!(st.regs[R_EAX], 0xffff_ffff_0000_0005);
    // A nonzero 32-bit source zero extends.
    check(&run64(&[(R_EAX, u64::MAX), (R_EBX, 0x8000)], &[0x0f, 0xbd, 0xc3, 0xf4]), 15, ZF, 0);
    // popcnt rax, rbx: qemu64 does not have POPCNT, EPYC does.
    let popcnt = [0xf3, 0x48, 0x0f, 0xb8, 0xc3, 0xf4];
    assert_eq!(vector64(&run64(&[], &popcnt)), 6);
    check(&run_epyc(&[(R_EBX, 0xff_00ff)], &popcnt), 16, ARITH, 0);
    // lzcnt rax, rbx needs ABM; without it the F3 prefix is ignored and it is BSR.
    let lzcnt = [0xf3, 0x48, 0x0f, 0xbd, 0xc3, 0xf4];
    check(&run64(&[(R_EBX, 0x10)], &lzcnt), 4, ZF, 0);
    check(&run_epyc(&[(R_EBX, 1)], &lzcnt), 63, CF | ZF, 0);
    check(&run_epyc(&[(R_EBX, 0)], &lzcnt), 64, CF | ZF, CF);
    // lzcnt ax, bx
    check(&run_epyc(&[(R_EBX, 0x00ff)], &[0x66, 0xf3, 0x0f, 0xbd, 0xc3, 0xf4]), 8, CF | ZF, 0);
    // bswap rax and bswap eax
    let st = run64(&[(R_EAX, 0x0102_0304_0506_0708)], &[0x48, 0x0f, 0xc8, 0xf4]);
    assert_eq!(st.regs[R_EAX], 0x0807_0605_0403_0201);
    let st = run64(&[(R_EAX, 0x1122_3344_5566_7788)], &[0x0f, 0xc8, 0xf4]);
    assert_eq!(st.regs[R_EAX], 0x8877_6655);
}

#[test]
fn conditions() {
    // cmp rax, rbx; setb cl; cmovl rdx, rbx; setg sil (REX); jae +1; hlt; hlt
    let code = [
        0x48, 0x39, 0xd8, 0x0f, 0x92, 0xc1, 0x48, 0x0f, 0x4c, 0xd3, 0x40, 0x0f, 0x9f, 0xc6, 0x73,
        0x01, 0xf4, 0xf4,
    ];
    let st = run64(&[(R_EAX, 1), (R_EBX, 2), (R_ECX, 0xff00), (R_ESI, 0x55)], &code);
    assert_eq!(st.regs[R_ECX], 0xff01);
    assert_eq!(st.regs[R_EDX], 2);
    assert_eq!(st.regs[R_ESI], 0);
    // jae not taken: halted in the first HLT.
    assert_eq!(st.rip, CODE + code.len() as u64 - 1);
    // Signed against unsigned: -1 < 1 but 0xff..ff > 1.
    let code = [0x48, 0x39, 0xd8, 0x0f, 0x9c, 0xc1, 0x0f, 0x97, 0xc2, 0xf4];
    let st = run64(&[(R_EAX, u64::MAX), (R_EBX, 1)], &code);
    assert_eq!(st.regs[R_ECX] & 0xff, 1);
    assert_eq!(st.regs[R_EDX] & 0xff, 1);
    // loop: rcx = 5, add rax, 2 each time.
    let code = [0x48, 0x83, 0xc0, 0x02, 0xe2, 0xfa, 0xf4];
    let st = run64(&[(R_ECX, 5)], &code);
    assert_eq!(st.regs[R_EAX], 10);
    assert_eq!(st.regs[R_ECX], 0);
}

#[test]
fn branches_inside_a_block() {
    // A loop whose body has more forward branches than a block has direct exits, some
    // taken and some not, with flags read after a branch that was not taken.
    let code = [
        0x89, 0xca, // top: mov edx, ecx
        0x83, 0xe2, 0x07, // and edx, 7
        0x83, 0xfa, 0x03, // cmp edx, 3
        0x75, 0x02, // jne +2
        0x01, 0xc8, // add eax, ecx
        0x83, 0xfa, 0x05, // cmp edx, 5
        0x72, 0x03, // jb +3
        0x83, 0xc0, 0x64, // add eax, 100
        0x83, 0xfa, 0x06, // cmp edx, 6
        0x73, 0x03, // jae +3
        0x83, 0xd3, 0x00, // adc ebx, 0
        0xf7, 0xc1, 0x01, 0x00, 0x00, 0x00, // test ecx, 1
        0x74, 0x02, // je +2
        0xff, 0xc6, // inc esi
        0x83, 0xfa, 0x00, // cmp edx, 0
        0x75, 0x03, // jne +3
        0x83, 0xc7, 0x07, // add edi, 7
        0xff, 0xc9, // dec ecx
        0x75, 0xce, // jnz top
        0xf4, // hlt
    ];
    let (mut a, mut b, mut si, mut di) = (0u64, 0u64, 0u64, 0u64);
    for c in (1..=100u64).rev() {
        let d = c & 7;
        if d == 3 {
            a += c;
        }
        if d >= 5 {
            a += 100;
        }
        if d < 6 {
            b += 1;
        }
        if c & 1 != 0 {
            si += 1;
        }
        if d == 0 {
            di += 7;
        }
    }
    let st = run64(&[(R_ECX, 100)], &code);
    assert_eq!(st.rip, CODE + code.len() as u64);
    assert_eq!(
        (st.regs[R_EAX], st.regs[R_EBX], st.regs[R_ESI], st.regs[R_EDI], st.regs[R_ECX]),
        (a, b, si, di, 0)
    );

    // A fault after a branch that was not taken sees the flags from before it.
    // cmp eax, ebx; jne +2; div ecx; hlt
    let code = [0x39, 0xd8, 0x75, 0x02, 0xf7, 0xf1, 0xf4];
    let st = run64(&[(R_EAX, 5), (R_EBX, 5), (R_ECX, 0)], &code);
    assert_eq!(vector64(&st), 0);
    assert_eq!(st.rflags & ARITH, ZF | PF);
}

#[test]
fn rep_movs_overlap_and_df() {
    // rep movsb forward with the destination one byte above the source.
    let w = World::new();
    w.write(DATA, b"abcdefghij");
    let st = w.run(
        World::long64(),
        &[(R_ESI, DATA), (R_EDI, DATA + 1), (R_ECX, 8)],
        &[0xf3, 0xa4, 0xf4],
    );
    assert_eq!(w.read(DATA, 10), b"aaaaaaaaaj");
    assert_eq!(st.regs[R_ECX], 0);
    assert_eq!(st.regs[R_ESI], DATA + 8);
    assert_eq!(st.regs[R_EDI], DATA + 9);
    // std; rep movsb backwards does a correct overlapping move up by one.
    let w = World::new();
    w.write(DATA, b"abcdefghij");
    let st = w.run(
        World::long64(),
        &[(R_ESI, DATA + 7), (R_EDI, DATA + 8), (R_ECX, 8)],
        &[0xfd, 0xf3, 0xa4, 0xf4],
    );
    assert_eq!(w.read(DATA, 10), b"aabcdefghj");
    assert_eq!(st.regs[R_ESI], DATA - 1);
    assert_eq!(st.regs[R_EDI], DATA);
    assert_ne!(st.rflags & DF, 0);
    // rep movsd with a zero count does nothing.
    let w = World::new();
    w.write(DATA, b"abcdefgh");
    let st = w.run(
        World::long64(),
        &[(R_ESI, DATA), (R_EDI, DATA + 4), (R_ECX, 0)],
        &[0xf3, 0xa5, 0xf4],
    );
    assert_eq!(w.read(DATA, 8), b"abcdefgh");
    assert_eq!(st.regs[R_ESI], DATA);
}

#[test]
fn rep_stos_cmps_scas() {
    // rep stosq
    let w = World::new();
    let st = w.run(
        World::long64(),
        &[(R_EAX, 0x1122_3344_5566_7788), (R_EDI, DATA), (R_ECX, 3)],
        &[0xf3, 0x48, 0xab, 0xf4],
    );
    for i in 0..3 {
        assert_eq!(w.r64(DATA + 8 * i), 0x1122_3344_5566_7788);
    }
    assert_eq!(st.regs[R_EDI], DATA + 24);
    // repe cmpsb stops at the first difference.
    let w = World::new();
    w.write(DATA, b"abcXefgh");
    w.write(DATA + 0x100, b"abcYefgh");
    let st = w.run(
        World::long64(),
        &[(R_ESI, DATA), (R_EDI, DATA + 0x100), (R_ECX, 8)],
        &[0xf3, 0xa6, 0xf4],
    );
    assert_eq!(st.regs[R_ECX], 4);
    assert_eq!(st.regs[R_ESI], DATA + 4);
    assert_eq!(st.rflags & (ZF | CF), CF);
    // repne scasb finds a zero byte.
    let w = World::new();
    w.write(DATA, b"hello\0world");
    let st = w.run(
        World::long64(),
        &[(R_EAX, 0), (R_EDI, DATA), (R_ECX, u64::MAX)],
        &[0xf2, 0xae, 0xf4],
    );
    assert_eq!(st.regs[R_EDI], DATA + 6);
    assert_eq!(st.regs[R_ECX], u64::MAX - 6);
    assert_ne!(st.rflags & ZF, 0);
    // A 32-bit address size wraps ECX only and leaves the upper half of RCX clear.
    let w = World::new();
    let st = w.run(
        World::long64(),
        &[(R_EAX, 0x5a), (R_EDI, DATA), (R_ECX, 0xffff_ffff_0000_0002)],
        &[0x67, 0xf3, 0xaa, 0xf4],
    );
    assert_eq!(st.regs[R_ECX], 0);
    assert_eq!(w.read(DATA, 3), [0x5a, 0x5a, 0]);
}

#[test]
fn exchange_and_atomics() {
    // cmpxchg [rbx], rcx succeeds.
    let w = World::new();
    w.w64(DATA, 7);
    let st = w.run(
        World::long64(),
        &[(R_EAX, 7), (R_EBX, DATA), (R_ECX, 9)],
        &[0x48, 0x0f, 0xb1, 0x0b, 0xf4],
    );
    assert_eq!(w.r64(DATA), 9);
    assert_ne!(st.rflags & ZF, 0);
    // lock cmpxchg [rbx], ecx fails and loads EAX.
    let w = World::new();
    w.w64(DATA, 0x1234);
    let st = w.run(
        World::long64(),
        &[(R_EAX, 0xffff_0000_0000_0007), (R_EBX, DATA), (R_ECX, 9)],
        &[0xf0, 0x0f, 0xb1, 0x0b, 0xf4],
    );
    assert_eq!(w.r64(DATA), 0x1234);
    assert_eq!(st.regs[R_EAX], 0x1234);
    assert_eq!(st.rflags & ZF, 0);
    // lock xadd [rbx], rax
    let w = World::new();
    w.w64(DATA, 5);
    let st =
        w.run(World::long64(), &[(R_EAX, 3), (R_EBX, DATA)], &[0xf0, 0x48, 0x0f, 0xc1, 0x03, 0xf4]);
    assert_eq!(w.r64(DATA), 8);
    assert_eq!(st.regs[R_EAX], 5);
    // lock add dword [rbx], 1; lock neg byte [rbx+4]; lock not word [rbx+6]; xchg [rbx+8], rcx
    let w = World::new();
    w.w64(DATA, 0x0000_0001_ffff_ffff);
    w.w64(DATA + 8, 0xaaaa);
    let code = [
        0xf0, 0x83, 0x03, 0x01, // lock add dword [rbx], 1
        0xf0, 0xf6, 0x5b, 0x04, // lock neg byte [rbx + 4]
        0xf0, 0x66, 0xf7, 0x53, 0x06, // lock not word [rbx + 6]
        0x48, 0x87, 0x4b, 0x08, // xchg [rbx + 8], rcx
        0xf4,
    ];
    let st = w.run(World::long64(), &[(R_EBX, DATA), (R_ECX, 0x5555)], &code);
    assert_eq!(w.r64(DATA), 0xffff_00ff_0000_0000);
    assert_eq!(w.r64(DATA + 8), 0x5555);
    assert_eq!(st.regs[R_ECX], 0xaaaa);
    // lock bts dword [rbx], 3
    let w = World::new();
    w.w64(DATA, 0);
    let st = w.run(World::long64(), &[(R_EBX, DATA)], &[0xf0, 0x0f, 0xba, 0x2b, 0x03, 0xf4]);
    assert_eq!(w.r64(DATA), 8);
    assert_eq!(st.rflags & CF, 0);
    // cmpxchg8b [rdi] succeeds; cmpxchg16b [rdi] fails and loads RDX:RAX.
    let w = World::new();
    w.w64(DATA, 0x0000_0002_0000_0001);
    let st = w.run(
        World::long64(),
        &[(R_EAX, 1), (R_EDX, 2), (R_EBX, 3), (R_ECX, 4), (R_EDI, DATA)],
        &[0x0f, 0xc7, 0x0f, 0xf4],
    );
    assert_eq!(w.r64(DATA), 0x0000_0004_0000_0003);
    assert_ne!(st.rflags & ZF, 0);
    let w = World::new();
    w.w64(DATA, 10);
    w.w64(DATA + 8, 20);
    let st = w.run(
        World::long64(),
        &[(R_EAX, 1), (R_EDX, 2), (R_EDI, DATA)],
        &[0xf0, 0x48, 0x0f, 0xc7, 0x0f, 0xf4],
    );
    assert_eq!(st.regs[R_EAX], 10);
    assert_eq!(st.regs[R_EDX], 20);
    assert_eq!(st.rflags & ZF, 0);
    // cmpxchg al, cl: the compare with itself succeeds and AL takes CL, not the old AL.
    let st = run64(&[(R_EAX, 0x1122_3344_5566_7705), (R_ECX, 0x99)], &[0x0f, 0xb0, 0xc8, 0xf4]);
    check(&st, 0x1122_3344_5566_7799, ZF, ZF);
    // cmpxchg ebx, ecx fails: EBX keeps its upper half, RAX takes the zero extended EBX.
    let st =
        run64(&[(R_EAX, 1), (R_EBX, 0xffff_ffff_0000_0002), (R_ECX, 9)], &[0x0f, 0xb1, 0xcb, 0xf4]);
    check(&st, 2, ZF, 0);
    assert_eq!(st.regs[R_EBX], 0xffff_ffff_0000_0002);
    // cmpxchg ebx, ecx succeeds: EBX is zero extended, RAX is not written.
    let st = run64(
        &[(R_EAX, 0xaaaa_0000_0000_0002), (R_EBX, 0xffff_ffff_0000_0002), (R_ECX, 9)],
        &[0x0f, 0xb1, 0xcb, 0xf4],
    );
    check(&st, 0xaaaa_0000_0000_0002, ZF, ZF);
    assert_eq!(st.regs[R_EBX], 9);
    // cmpxchg ah, bl fails: AL takes AH.
    let st = run64(&[(R_EAX, 0x1234), (R_EBX, 0x55)], &[0x0f, 0xb0, 0xdc, 0xf4]);
    check(&st, 0x1212, ZF, 0);
    // LOCK on a register destination is #UD.
    let st = run64(&[], &[0xf0, 0x01, 0xc0, 0xf4]);
    assert_eq!(vector64(&st), 6);
}

#[test]
fn stack_call_enter_leave() {
    // call sub; hlt; sub: enter 16, 0; mov rax, rbp; leave; ret
    let code = [
        0xe8, 0x01, 0x00, 0x00, 0x00, // call +1
        0xf4, // hlt
        0xc8, 0x10, 0x00, 0x00, // enter 16, 0
        0x48, 0x89, 0xe8, // mov rax, rbp
        0x48, 0x89, 0xe1, // mov rcx, rsp
        0xc9, // leave
        0xc3, // ret
    ];
    let st = run64(&[(R_EBX, 0)], &code);
    assert_eq!(st.rip, CODE + 6);
    assert_eq!(st.regs[R_ESP], STACK);
    assert_eq!(st.regs[R_EAX], STACK - 16);
    assert_eq!(st.regs[R_ECX], STACK - 32);
    // push rax; push 0x7f; pop rbx; pop rcx; push word 5; pop dx
    let code = [0x50, 0x6a, 0x7f, 0x5b, 0x59, 0x66, 0x6a, 0x05, 0x66, 0x5a, 0xf4];
    let st = run64(&[(R_EAX, 0x1234_5678_9abc), (R_EDX, 0xffff_0000)], &code);
    assert_eq!(st.regs[R_EBX], 0x7f);
    assert_eq!(st.regs[R_ECX], 0x1234_5678_9abc);
    assert_eq!(st.regs[R_EDX], 0xffff_0005);
    assert_eq!(st.regs[R_ESP], STACK);
}

#[test]
fn protected_mode_32() {
    // pushad; add eax, ebx; popad leaves EAX restored; pushfd; pop ecx
    let w = World::new();
    let code = [0x60, 0x01, 0xd8, 0x61, 0x01, 0xd8, 0x9c, 0x59, 0xf4];
    let st = w.run(World::prot32(), &[(R_EAX, 0xffff_ffff), (R_EBX, 1)], &code);
    assert_eq!(st.regs[R_EAX], 0);
    assert_eq!(st.regs[R_ECX] & ARITH, CF | ZF | PF | AF);
    assert_eq!(st.regs[R_ESP], STACK);
    // movzx, movsx and lea with a SIB byte.
    let code = [
        0x0f, 0xb6, 0xc3, // movzx eax, bl
        0x0f, 0xbf, 0xcb, // movsx ecx, bx
        0x8d, 0x54, 0x98, 0x10, // lea edx, [eax + ebx * 4 + 0x10]
        0xf4,
    ];
    let w = World::new();
    let st = w.run(World::prot32(), &[(R_EBX, 0x8081)], &code);
    assert_eq!(st.regs[R_EAX], 0x81);
    assert_eq!(st.regs[R_ECX], 0xffff_8081);
    assert_eq!(st.regs[R_EDX], 0x81 + 0x8081 * 4 + 0x10);
}

#[test]
fn page_fault_error_code_and_cr2() {
    // mov [rbx], rax to a page that is not present: P = 0, W = 1.
    let w = World::new();
    let st = w.run(World::long64(), &[(R_EBX, 0x60_0008)], &[0x48, 0x89, 0x03, 0xf4]);
    assert_eq!(vector64(&st), 14);
    assert_eq!(st.cr2, 0x60_0008);
    assert_eq!(w.r64(st.regs[R_ESP]), 2);
    assert_eq!(w.r64(st.regs[R_ESP] + 8), CODE);
    // A read of the same page: error code 0.
    let w = World::new();
    let st = w.run(World::long64(), &[(R_EBX, 0x60_0000)], &[0x90, 0x48, 0x8b, 0x03, 0xf4]);
    assert_eq!(vector64(&st), 14);
    assert_eq!(w.r64(st.regs[R_ESP]), 0);
    assert_eq!(w.r64(st.regs[R_ESP] + 8), CODE + 1);
    // A supervisor write to a read only page with CR0.WP: P = 1, W = 1.
    let w = World::new();
    let st = w.run(World::long64(), &[(R_EBX, 0x20_0010)], &[0x88, 0x03, 0xf4]);
    assert_eq!(vector64(&st), 14);
    assert_eq!(st.cr2, 0x20_0010);
    assert_eq!(w.r64(st.regs[R_ESP]), 3);
    // A rep stosb that faults part way keeps the progress made in RCX and RDI.
    let w = World::new();
    let st = w.run(World::long64(), &[(R_EDI, 0x1f_fffe), (R_ECX, 4)], &[0xf3, 0xaa, 0xf4]);
    assert_eq!(vector64(&st), 14);
    assert_eq!(st.cr2, 0x20_0000);
    assert_eq!(st.regs[R_ECX], 2);
    assert_eq!(st.regs[R_EDI], 0x20_0000);
    assert_eq!(w.r64(st.regs[R_ESP] + 8), CODE);
}

#[test]
fn protection_keys() {
    let max = || {
        let mut m = X86Cpu::new("max", Accel::Tcg).unwrap();
        m.realize().unwrap();
        assert!(m.has_feature("pku") && m.has_feature("pks"));
        World::with_model(X86::new(m))
    };
    let pk = || {
        let mut st = World::long64();
        st.cr4 |= CR4_PKE_MASK | CR4_PKS_MASK;
        st
    };
    // rdpkru reads PKRU into EDX:EAX.
    let mut st = pk();
    st.pkru = 0xc;
    let st = max().run(st, &[(R_EAX, u64::MAX), (R_EDX, u64::MAX)], &[0x0f, 0x01, 0xee, 0xf4]);
    assert_eq!((st.regs[R_EAX], st.regs[R_EDX]), (0xc, 0));
    // Without CR4.PKE both are #UD; ECX other than 0 or EDX other than 0 is #GP.
    for code in [[0x0f, 0x01, 0xee, 0xf4], [0x0f, 0x01, 0xef, 0xf4]] {
        assert_eq!(vector64(&max().run(World::long64(), &[], &code)), 6);
    }
    assert_eq!(vector64(&max().run(pk(), &[(R_ECX, 1)], &[0x0f, 0x01, 0xee, 0xf4])), 13);
    assert_eq!(vector64(&max().run(pk(), &[(R_EDX, 1)], &[0x0f, 0x01, 0xef, 0xf4])), 13);
    // The 2 to 4 MiB page gets key 1, writable, and user when `user`; the code, stack and
    // tables stay on key 0.
    let keyed = |user: bool| {
        let w = max();
        let u = if user { 4 } else { 0 };
        w.w64(PML4, (PML4 + 0x1000) | 3 | u);
        w.w64(PML4 + 0x1000, (PML4 + 0x2000) | 3 | u);
        w.w64(PML4 + 0x2008, 0x20_0000 | 0x83 | u | 1 << 59);
        w
    };
    let addr = 0x20_0100;
    // PKRS covers the supervisor pages: access disable on key 1 faults a read with PK | P,
    // write disable faults a write (CR0.WP is set) with PK | W | P but lets a read through.
    // mov ecx, 0x6e1; mov eax, imm; xor edx, edx; wrmsr; mov al, [rbx]; mov [rbx], al
    let pkrs = |v: u8| [0xb9, 0xe1, 6, 0, 0, 0xb8, v, 0, 0, 0, 0x31, 0xd2, 0x0f, 0x30];
    let w = keyed(false);
    let code = [&pkrs(4)[..], &[0x8a, 0x03, 0xf4]].concat();
    let st = w.run(pk(), &[(R_EBX, addr)], &code);
    assert_eq!(vector64(&st), 14);
    assert_eq!((st.cr2, w.r64(st.regs[R_ESP])), (addr, 0x21));
    let w = keyed(false);
    let code = [&pkrs(8)[..], &[0x8a, 0x03, 0x88, 0x03, 0xf4]].concat();
    let st = w.run(pk(), &[(R_EBX, addr)], &code);
    assert_eq!(vector64(&st), 14);
    assert_eq!(w.r64(st.regs[R_ESP]), 0x23);
    assert_eq!(w.r64(st.regs[R_ESP] + 8), CODE + 16);
    assert_eq!(st.pkrs, 8);
    // PKRU covers the user pages, for supervisor accesses too (SMAP is off).
    let w = keyed(true);
    let regs = [(R_EAX, 4), (R_ECX, 0), (R_EDX, 0), (R_EBX, addr)];
    let st = w.run(pk(), &regs, &[0x0f, 0x01, 0xef, 0x8a, 0x03, 0xf4]);
    assert_eq!(vector64(&st), 14);
    assert_eq!(w.r64(st.regs[R_ESP]), 0x21);
    assert_eq!(st.pkru, 4);
    // Without CR4.PKE the key is ignored.
    let w = keyed(true);
    let mut st = World::long64();
    st.pkru = 0xc;
    let st = w.run(st, &[(R_EBX, addr)], &[0x8a, 0x03, 0xf4]);
    assert_eq!(st.rip, CODE + 3);
}

#[test]
fn undefined_and_fpu_raise_ud() {
    // addps xmm0, xmm0 and vaddps without CR4.OSFXSR and CR4.OSXSAVE, the invalid x87 form
    // D9 D1, and ud2 are #UD.
    for code in [&[0x0f, 0x58, 0xc0][..], &[0xc5, 0xf8, 0x58, 0xc0], &[0xd9, 0xd1], &[0x0f, 0x0b]] {
        let st = run64(&[], code);
        assert_eq!(vector64(&st), 6, "{code:x?}");
        assert_eq!(st.rip, HANDLERS + 6 * 16 + 1);
    }
    // push es is not valid in 64-bit mode.
    assert_eq!(vector64(&run64(&[], &[0x06])), 6);
}

#[test]
fn cpuid_rdtsc_and_system() {
    // cpuid leaf 0: qemu64 is "AuthenticAMD".
    let st = run64(&[(R_EAX, 0)], &[0x0f, 0xa2, 0xf4]);
    assert_eq!(st.regs[R_EBX], u32::from_le_bytes(*b"Auth") as u64);
    assert_eq!(st.regs[R_EDX], u32::from_le_bytes(*b"enti") as u64);
    assert_eq!(st.regs[R_ECX], u32::from_le_bytes(*b"cAMD") as u64);
    // rdtsc leaves the upper halves clear.
    let st = run64(&[(R_EAX, u64::MAX), (R_EDX, u64::MAX)], &[0x0f, 0x31, 0xf4]);
    assert_eq!(st.regs[R_EAX] >> 32, 0);
    assert_eq!(st.regs[R_EDX] >> 32, 0);
    // mov rax, cr0; mov rcx, cr3; mov dr0, rbx; mov rdx, dr0
    let code = [0x0f, 0x20, 0xc0, 0x0f, 0x20, 0xd9, 0x0f, 0x23, 0xc3, 0x0f, 0x21, 0xc2, 0xf4];
    let st = run64(&[(R_EBX, 0x1234)], &code);
    assert_eq!(st.regs[R_EAX] & 0x8000_0001, 0x8000_0001);
    assert_eq!(st.regs[R_ECX], PML4);
    assert_eq!(st.regs[R_EDX], 0x1234);
    // int3 goes through the IDT; the pushed RIP is after the INT3.
    let w = World::new();
    let st = w.run(World::long64(), &[], &[0xcc, 0xf4]);
    assert_eq!(vector64(&st), 3);
    assert_eq!(w.r64(st.regs[R_ESP]), CODE + 1);
}

#[test]
fn syscall_and_iretq() {
    // syscall jumps to LSTAR with RCX = the next RIP and R11 = RFLAGS.
    let w = World::new();
    w.write(0x6000, &[0xf4]);
    let mut st = World::long64();
    st.efer |= MSR_EFER_SCE;
    st.star = 0x0008_0000_0000_0000 | (0x08u64 << 32);
    st.lstar = 0x6000;
    st.fmask = 0x200;
    let st = w.run(st, &[], &[0x0f, 0x05, 0xf4]);
    assert_eq!(st.rip, 0x6001);
    assert_eq!(st.regs[R_ECX], CODE + 2);
    assert_eq!(st.regs[11], 2);
    assert_eq!(st.segs[R_CS].selector, 0x08);
    assert_eq!(st.segs[R_SS].selector, 0x10);
    // iretq to the same privilege level.
    let code = [
        0x48, 0x89, 0xe0, // mov rax, rsp
        0x6a, 0x10, // push 0x10
        0x50, // push rax
        0x6a, 0x02, // push 2 (rflags)
        0x6a, 0x08, // push 0x08
        0x68, 0x40, 0x10, 0x00, 0x00, // push 0x1040
        0x48, 0xcf, // iretq
    ];
    let w = World::new();
    w.write(CODE + 0x40, &[0xf4]);
    let st = w.run(World::long64(), &[], &code);
    assert_eq!(st.rip, CODE + 0x41);
    assert_eq!(st.regs[R_ESP], STACK);
}

#[test]
fn real_to_long_mode_switch() {
    let w = World::new();
    // GDTR image at 0x3100.
    w.write(0x3100, &[0x1f, 0x00, 0x00, 0x30, 0x00, 0x00]);
    let mut code = vec![
        0x0f, 0x01, 0x16, 0x00, 0x31, // lgdt [0x3100]
        0x66, 0xb8, 0x20, 0x00, 0x00, 0x00, // mov eax, CR4.PAE
        0x0f, 0x22, 0xe0, // mov cr4, eax
        0x66, 0xb8, 0x00, 0x00, 0x01, 0x00, // mov eax, 0x10000
        0x0f, 0x22, 0xd8, // mov cr3, eax
        0x66, 0xb9, 0x80, 0x00, 0x00, 0xc0, // mov ecx, MSR_EFER
        0x0f, 0x32, // rdmsr
        0x66, 0x0d, 0x00, 0x01, 0x00, 0x00, // or eax, EFER.LME
        0x0f, 0x30, // wrmsr
        0x0f, 0x20, 0xc0, // mov eax, cr0
        0x66, 0x0d, 0x01, 0x00, 0x00, 0x80, // or eax, CR0.PG | CR0.PE
        0x0f, 0x22, 0xc0, // mov cr0, eax
        0x66, 0xea, 0x00, 0x11, 0x00, 0x00, 0x08, 0x00, // jmp dword 0x08:0x1100
    ];
    code.resize(0x100, 0xf4);
    // 64-bit code at 0x1100: mov rax, imm64; mov ebx, 0x10; mov ds, ebx; push rax; pop rcx;
    // hlt
    code.extend_from_slice(&[0x48, 0xb8, 0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12]);
    code.extend_from_slice(&[0xbb, 0x10, 0x00, 0x00, 0x00, 0x8e, 0xdb]);
    code.extend_from_slice(&[0x50, 0x59, 0xf4]);
    let st = w.run(World::real(), &[], &code);
    assert_ne!(st.efer & MSR_EFER_LMA, 0);
    assert_ne!(st.hflags & HF_CS64_MASK, 0);
    assert_eq!(st.segs[R_CS].selector, 0x08);
    assert_eq!(st.segs[3].selector, 0x10);
    assert_eq!(st.regs[R_EAX], 0x1234_5678_9abc_def0);
    assert_eq!(st.regs[R_ECX], 0x1234_5678_9abc_def0);
    assert_eq!(st.rip, 0x1100 + 20);
    assert_eq!(st.gdt.base, GDT);
}

#[test]
fn string_and_misc_16_bit() {
    // Real mode: rep stosw with ES:DI; cbw; cwd; xlat
    let w = World::new();
    w.write(0x8100, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let code = [
        0xb8, 0x34, 0x12, // mov ax, 0x1234
        0xbf, 0x00, 0x80, // mov di, 0x8000
        0xb9, 0x03, 0x00, // mov cx, 3
        0xf3, 0xab, // rep stosw
        0xb0, 0x85, // mov al, 0x85
        0x98, // cbw
        0x99, // cwd
        0xbb, 0x00, 0x81, // mov bx, 0x8100
        0x88, 0xce, // mov dh, cl
        0xb0, 0x07, // mov al, 7
        0xd7, // xlat
        0xf4,
    ];
    let st = w.run(World::real(), &[(R_EDI, 0xffff_0000)], &code);
    assert_eq!(w.read(DATA, 6), [0x34, 0x12, 0x34, 0x12, 0x34, 0x12]);
    assert_eq!(st.regs[R_EDI], 0xffff_8006);
    assert_eq!(st.regs[R_EAX], 0xff07);
    // The upper half of EDX keeps the CPUID version loaded at reset.
    assert_eq!(st.regs[R_EDX] & 0xffff, 0x00ff);
}

/// x86 blocks carry the TSO order of QEMU's i386 `TCGCPUOps`: `TCG_MO_ALL & ~TCG_MO_ST_LD`,
/// so only a store followed by a load may be reordered.
#[test]
fn guest_memory_order_is_tso() {
    use ruvm_jit::CpuOps;
    let x = X86::new(X86Cpu::new("qemu64", Accel::Tcg).unwrap());
    assert_eq!(x.guest_default_memory_order(), 0x0d);
}
