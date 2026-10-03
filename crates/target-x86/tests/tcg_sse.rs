// SPDX-License-Identifier: GPL-2.0-or-later

//! The x87, SSE and AVX slice of the translator: for now the general purpose register
//! extensions that come with it (BMI1, BMI2, ADX, MOVBE, CRC32, RDRAND, RDSEED, RDPID,
//! XGETBV, the FS and GS base instructions, LDMXCSR and STMXCSR) and the VEX prefix.
//!
//! `data/tcg_bmi.txt` holds the results of the same instruction bytes run natively on an
//! AMD EPYC host; each case runs in 64-bit mode with RDI pointing at 32 bytes of data. RSP,
//! RDI and R15 are not compared (the native harness uses R15 as its base register).
//! `data/native` holds the case generator and the C harness that produced the file.
//!
//! The memory map follows `tcg.rs`: code at 0x1000, the GDT at 0x3000, the long mode IDT at
//! 0x4000 with a HLT handler per vector at 0x5000 + 16 n, data at 0x8000, the stack below
//! 0x30000 and the page tables at 0x10000, which identity map the first 2 MiB.

use std::sync::Arc;

use ruvm_jit::cpu_exec::cpu_exec;
use ruvm_jit::{Jit, Vcpu, excp};
use ruvm_mem::{AddressSpace, MemTxAttrs, MemorySystem};
use ruvm_target_x86::cpuid::{Accel, X86Cpu};
use ruvm_target_x86::state::{
    CR0_ET_MASK, CR0_NE_MASK, CR0_PE_MASK, CR0_PG_MASK, CR0_TS_MASK, CR0_WP_MASK,
    CR4_FSGSBASE_MASK, CR4_OSFXSR_MASK, CR4_OSXSAVE_MASK, CR4_PAE_MASK, DESC_A_MASK, DESC_B_MASK,
    DESC_CS_MASK, DESC_G_MASK, DESC_L_MASK, DESC_P_MASK, DESC_R_MASK, DESC_S_MASK, DESC_W_MASK,
    HF_LMA_MASK, IrqchipMode, MSR_EFER_LME, R_CS, R_EAX, R_EBX, R_ECX, R_EDI, R_EDX, R_ESP, R_FS,
    R_GS, ResetConfig, SegmentCache, X86CpuState,
};
use ruvm_target_x86::tcg::{X86, create_vcpu, new_jit, save_vcpu};

const U: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const RAM_SIZE: u64 = 0x40_0000;
const CODE: u64 = 0x1000;
const GDT: u64 = 0x3000;
const IDT: u64 = 0x4000;
const HANDLERS: u64 = 0x5000;
const DATA: u64 = 0x8000;
const STACK: u64 = 0x3_0000;
const PML4: u64 = 0x1_0000;
const CF: u64 = 0x001;
const ARITH: u64 = 0x8d5;
/// `HF_OSFXSR_MASK`, which `cpu_x86_update_cr4()` derives from CR4.OSFXSR.
const HF_OSFXSR: u32 = 1 << 22;

const CODE64: u32 = DESC_P_MASK
    | DESC_S_MASK
    | DESC_CS_MASK
    | DESC_R_MASK
    | DESC_A_MASK
    | DESC_L_MASK
    | DESC_G_MASK;
const DATA32: u32 =
    DESC_P_MASK | DESC_S_MASK | DESC_W_MASK | DESC_A_MASK | DESC_B_MASK | DESC_G_MASK;

struct World {
    _sys: MemorySystem,
    as_: Arc<AddressSpace>,
    jit: Arc<Jit>,
    x86: Arc<X86>,
}

impl World {
    fn new(model: &str) -> World {
        let mut m = X86Cpu::new(model, Accel::Tcg).unwrap();
        let _ = m.realize();
        let sys = MemorySystem::new();
        let root = sys.new_container("system", 1 << 64).unwrap();
        let as_ = sys.address_space_init(root, "memory").unwrap();
        let ram = sys.new_ram("ram", RAM_SIZE).unwrap();
        sys.add_subregion(root, 0, ram).unwrap();
        let w = World { _sys: sys, as_, jit: new_jit(), x86: Arc::new(X86::new(m)) };
        w.w64(GDT, 0);
        w.w64(GDT + 8, 0x00af_9b00_0000_ffff);
        w.w64(GDT + 16, 0x00cf_9300_0000_ffff);
        for n in 0..32u64 {
            let h = HANDLERS + n * 16;
            let lo = (h & 0xffff) | (0x08 << 16) | (0x8e00 << 32) | ((h >> 16) & 0xffff) << 48;
            w.w64(IDT + n * 16, lo);
            w.w64(IDT + n * 16 + 8, h >> 32);
            w.write(h, &[0xf4]);
        }
        w.w64(PML4, (PML4 + 0x1000) | 3);
        w.w64(PML4 + 0x1000, (PML4 + 0x2000) | 3);
        w.w64(PML4 + 0x2000, 0x83);
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

    /// 64-bit mode at CPL 0 with paging and `cr4` added to CR4.PAE.
    fn long64(cr4: u64) -> X86CpuState {
        let mut st = X86CpuState::new_reset(&ResetConfig {
            is_bsp: true,
            kvm: false,
            irqchip: IrqchipMode::Off,
            cpuid_version: 0x0080_0f12,
            has_monitor: false,
        });
        st.rflags = 2;
        st.rip = CODE;
        st.cr4 = CR4_PAE_MASK | cr4;
        st.efer = MSR_EFER_LME;
        st.cr3 = PML4;
        st.update_cr0(CR0_PE_MASK | CR0_ET_MASK | CR0_NE_MASK | CR0_WP_MASK | CR0_PG_MASK);
        assert_ne!(st.hflags & HF_LMA_MASK, 0);
        if cr4 & CR4_OSFXSR_MASK != 0 {
            st.hflags |= HF_OSFXSR;
        }
        st.gdt = SegmentCache { selector: 0, base: GDT, limit: 0x17, flags: 0 };
        st.idt = SegmentCache { selector: 0, base: IDT, limit: 0x1ff, flags: 0 };
        st.load_seg_cache(R_CS, 0x08, 0, 0xffff_ffff, CODE64);
        for s in [0, 2, 3, 4, 5] {
            st.load_seg_cache(s, 0x10, 0, 0xffff_ffff, DATA32);
        }
        st.regs[R_ESP] = STACK;
        st
    }

    /// Run `code` followed by HLT until the vCPU halts. Writes from outside the vCPU do not
    /// flush the translation block cache, so every world runs code only once.
    fn run(&self, mut st: X86CpuState, code: &[u8]) -> X86CpuState {
        let mut c = code.to_vec();
        c.push(0xf4);
        self.write(CODE, &c);
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

/// The vector of the exception handler a vCPU halted in, or `None` after the snippet's HLT.
fn vector(st: &X86CpuState) -> Option<u64> {
    if st.rip > HANDLERS && st.rip < HANDLERS + 32 * 16 {
        Some((st.rip - HANDLERS - 1) / 16)
    } else {
        None
    }
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn regs(s: &str) -> Vec<u64> {
    s.split(',').map(|v| u64::from_str_radix(v, 16).unwrap()).collect()
}

#[test]
fn native_results_match() {
    let w0 = World::new("EPYC");
    for f in ["bmi1", "bmi2", "adx", "movbe", "sse4.2"] {
        assert!(w0.x86.model().has_feature(f), "{f}");
    }
    let data = include_str!("data/tcg_bmi.txt");
    let mut n = 0;
    for line in data.lines().filter(|l| !l.starts_with('#')) {
        let p: Vec<&str> = line.split(';').collect();
        assert_eq!(p.len(), 9, "{line}");
        let (name, code) = (p[0], hex(p[1]));
        let (rin, fin, min) = (regs(p[2]), u64::from_str_radix(p[3], 16).unwrap(), hex(p[4]));
        let mask = u64::from_str_radix(p[5], 16).unwrap();
        let (rout, fout, mout) = (regs(p[6]), u64::from_str_radix(p[7], 16).unwrap(), hex(p[8]));
        let w = World::new("EPYC");
        let mut st = World::long64(0);
        for (i, &v) in rin.iter().enumerate() {
            if i != R_ESP {
                st.regs[i] = v;
            }
        }
        st.regs[R_EDI] = DATA;
        st.rflags = fin;
        w.write(DATA, &min);
        let st = w.run(st, &code);
        assert_eq!(vector(&st), None, "{name}: exception");
        for (i, &v) in rout.iter().enumerate() {
            if i != R_ESP && i != R_EDI && i != 15 {
                assert_eq!(st.regs[i], v, "{name}: register {i}: {:#x} != {v:#x}", st.regs[i]);
            }
        }
        assert_eq!(st.rflags & mask, fout & mask, "{name}: rflags {:#x}", st.rflags);
        assert_eq!(w.read(DATA, 32), mout, "{name}: memory");
        n += 1;
    }
    assert!(n > 400);
}

#[test]
fn vex_and_bmi_raise_ud() {
    let ud = |m: &str, code: &[u8]| {
        let st = World::new(m).run(World::long64(0), code);
        assert_eq!(vector(&st), Some(6), "{code:x?}");
        assert_eq!(st.rip, HANDLERS + 6 * 16 + 1);
    };
    // andn rax, rbx, rcx with VEX.L = 1.
    ud("EPYC", &[0xc4, 0xe2, 0xe4, 0xf2, 0xc1]);
    // A 66 prefix before VEX.
    ud("EPYC", &[0x66, 0xc4, 0xe2, 0xe0, 0xf2, 0xc1]);
    // VEX map 4 is reserved.
    ud("EPYC", &[0xc4, 0xe4, 0xe0, 0xf2, 0xc1]);
    // movbe with a register operand.
    ud("EPYC", &[0x0f, 0x38, 0xf0, 0xc1]);
    // F3 0F 38 F0 is empty.
    ud("EPYC", &[0xf3, 0x0f, 0x38, 0xf0, 0x07]);
    // blsr with modrm.reg 0 has no instruction.
    ud("EPYC", &[0xc4, 0xe2, 0xb0, 0xf3, 0xc1]);
    // Vector instructions are not implemented yet: vaddps.
    ud("EPYC", &[0xc5, 0xf8, 0x58, 0xc0]);
    // qemu64 has no BMI1, BMI2 or ADX.
    ud("qemu64", &[0xc4, 0xe2, 0xe0, 0xf2, 0xc1]);
    ud("qemu64", &[0xc4, 0xe2, 0xe3, 0xf6, 0xc1]);
    ud("qemu64", &[0x66, 0x48, 0x0f, 0x38, 0xf6, 0xc3]);
}

#[test]
fn rdrand_rdseed_rdpid() {
    let w = World::new("EPYC");
    // rdrand rax; rdrand ecx; rdseed rdx.
    let mut st = World::long64(0);
    st.rflags = 2 | ARITH;
    let st = World::new("EPYC")
        .run(st, &[0x48, 0x0f, 0xc7, 0xf0, 0x0f, 0xc7, 0xf1, 0x48, 0x0f, 0xc7, 0xfa]);
    assert_eq!(vector(&st), None);
    assert_eq!(st.rflags & ARITH, CF);
    assert_eq!(st.regs[R_ECX] >> 32, 0);
    // A memory operand is #UD.
    let st = World::new("EPYC").run(World::long64(0), &[0x0f, 0xc7, 0x37]);
    assert_eq!(vector(&st), Some(6));
    // rdpid rax.
    let st = World::new("EPYC").run(World::long64(0), &[0xf3, 0x0f, 0xc7, 0xf8]);
    let has = w.x86.model().has_feature("rdpid");
    assert_eq!(vector(&st), if has { None } else { Some(6) });
}

#[test]
fn xgetbv_and_xsetbv() {
    // xgetbv with ECX = 0 returns XCR0.
    let mut st = World::long64(CR4_OSXSAVE_MASK);
    st.xcr0 = 7;
    st.regs[R_ECX] = 0;
    st.regs[R_EAX] = !0;
    st.regs[R_EDX] = !0;
    let st = World::new("EPYC").run(st, &[0x0f, 0x01, 0xd0]);
    assert_eq!((vector(&st), st.regs[R_EAX], st.regs[R_EDX]), (None, 7, 0));
    // Without CR4.OSXSAVE it is #UD; ECX = 2 is #GP.
    assert_eq!(vector(&World::new("EPYC").run(World::long64(0), &[0x0f, 0x01, 0xd0])), Some(6));
    let mut st = World::long64(CR4_OSXSAVE_MASK);
    st.regs[R_ECX] = 2;
    assert_eq!(vector(&World::new("EPYC").run(st, &[0x0f, 0x01, 0xd0])), Some(13));
    // xsetbv XCR0 = 3, then read it back.
    let mut st = World::long64(CR4_OSXSAVE_MASK);
    st.xcr0 = 1;
    st.regs[R_ECX] = 0;
    st.regs[R_EAX] = 3;
    st.regs[R_EDX] = 0;
    let st = World::new("EPYC").run(st, &[0x0f, 0x01, 0xd1, 0x31, 0xc0, 0x0f, 0x01, 0xd0]);
    assert_eq!((vector(&st), st.xcr0, st.regs[R_EAX]), (None, 3, 3));
    // XCR0 without x87 state is #GP.
    let mut st = World::long64(CR4_OSXSAVE_MASK);
    st.regs[R_ECX] = 0;
    st.regs[R_EAX] = 2;
    st.regs[R_EDX] = 0;
    assert_eq!(vector(&World::new("EPYC").run(st, &[0x0f, 0x01, 0xd1])), Some(13));
}

#[test]
fn fs_gs_base() {
    let w = World::new("EPYC");
    // wrfsbase rbx; wrgsbase ecx; rdfsbase rax; rdgsbase edx.
    let code = [
        0xf3, 0x48, 0x0f, 0xae, 0xd3, 0xf3, 0x0f, 0xae, 0xd9, 0xf3, 0x48, 0x0f, 0xae, 0xc0, 0xf3,
        0x0f, 0xae, 0xca,
    ];
    let mut st = World::long64(CR4_FSGSBASE_MASK);
    st.regs[R_EBX] = 0x1234_5678_9abc_def0;
    st.regs[R_ECX] = 0xffff_ffff_8765_4321;
    let st = w.run(st, &code);
    assert_eq!(vector(&st), None);
    assert_eq!(st.segs[R_FS].base, 0x1234_5678_9abc_def0);
    assert_eq!(st.segs[R_GS].base, 0x8765_4321);
    assert_eq!(st.regs[R_EAX], 0x1234_5678_9abc_def0);
    assert_eq!(st.regs[R_EDX], 0x8765_4321);
    // Without CR4.FSGSBASE they are #UD.
    assert_eq!(vector(&w.run(World::long64(0), &code)), Some(6));
}

#[test]
fn ldmxcsr_stmxcsr() {
    let w = World::new("EPYC");
    // ldmxcsr [rdi]; stmxcsr [rdi + 8].
    let code = [0x0f, 0xae, 0x17, 0x0f, 0xae, 0x5f, 0x08];
    w.w64(DATA, 0x7f80);
    w.w64(DATA + 8, !0);
    let mut st = World::long64(CR4_OSFXSR_MASK);
    st.regs[R_EDI] = DATA;
    let st = w.run(st, &code);
    assert_eq!(vector(&st), None);
    assert_eq!(st.mxcsr, 0x7f80);
    assert_eq!(w.read(DATA + 8, 8), [0x80, 0x7f, 0, 0, 0xff, 0xff, 0xff, 0xff]);
    // Without CR4.OSFXSR it is #UD; with CR0.TS it is #NM.
    let mut st = World::long64(0);
    st.regs[R_EDI] = DATA;
    assert_eq!(vector(&w.run(st, &code)), Some(6));
    let mut st = World::long64(CR4_OSFXSR_MASK);
    st.regs[R_EDI] = DATA;
    st.update_cr0(st.cr0 | CR0_TS_MASK);
    assert_eq!(vector(&w.run(st, &code)), Some(7));
}

#[test]
fn movnti() {
    let w = World::new("EPYC");
    // movnti [rdi], rax; movnti [rdi + 8], ecx.
    let mut st = World::long64(0);
    st.regs[R_EDI] = DATA;
    st.regs[R_EAX] = 0x1122_3344_5566_7788;
    st.regs[R_ECX] = 0x99aa_bbcc_ddee_ff00;
    w.write(DATA, &[0; 16]);
    let st = w.run(st, &[0x48, 0x0f, 0xc3, 0x07, 0x0f, 0xc3, 0x4f, 0x08]);
    assert_eq!(vector(&st), None);
    assert_eq!(w.read(DATA, 16), hex("887766554433221100ffeedd00000000"));
}
