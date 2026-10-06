// SPDX-License-Identifier: GPL-2.0-or-later

//! The x87, SSE and AVX slice of the translator: MMX, SSE to SSE4.2, AVX, AVX2, FMA and F16C,
//! the general purpose register extensions that come with them (BMI1, BMI2, ADX, MOVBE,
//! CRC32, RDRAND, RDSEED, RDPID, XGETBV, the FS and GS base instructions, LDMXCSR and STMXCSR)
//! and the VEX prefix.
//!
//! `data/tcg_bmi.txt` holds the results of the same instruction bytes run natively on an
//! AMD EPYC host; each case runs in 64-bit mode with RDI pointing at 32 bytes of data. RSP,
//! RDI and R15 are not compared (the native harness uses R15 as its base register).
//!
//! `data/tcg_vec.txt` does the same for the vector instructions. To keep the file small, it
//! holds the kind and the seed of each case's input, which [`Rng`] expands as the generator
//! does, and only the 8-byte chunks of the state that changed. The state is YMM0 to YMM3, MM0
//! and MM1, RAX, RCX, RDX and the 32 bytes RDI points to, plus RFLAGS and MXCSR.
//!
//! `data/native` holds the case generators and the C harnesses that produced the files.
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
    CR0_EM_MASK, CR0_ET_MASK, CR0_MP_MASK, CR0_NE_MASK, CR0_PE_MASK, CR0_PG_MASK, CR0_TS_MASK,
    CR0_WP_MASK, CR4_FSGSBASE_MASK, CR4_OSFXSR_MASK, CR4_OSXSAVE_MASK, CR4_PAE_MASK, DESC_A_MASK,
    DESC_B_MASK, DESC_CS_MASK, DESC_G_MASK, DESC_L_MASK, DESC_P_MASK, DESC_R_MASK, DESC_S_MASK,
    DESC_W_MASK, HF_LMA_MASK, IrqchipMode, MSR_EFER_LME, R_CS, R_EAX, R_EBX, R_ECX, R_EDI, R_EDX,
    R_ESP, R_FS, R_GS, ResetConfig, SegmentCache, X86CpuState,
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
    // Without CR4.OSXSAVE and XCR0, VEX encoded vector instructions are #UD: vaddps.
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

/// splitmix64 and the input kinds of `data/native/gen_vec.py`; keep the two in sync.
struct Rng(u64);

const F32: [u32; 19] = [
    0,
    0x8000_0000,
    0x3f80_0000,
    0xbfc0_0000,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0xffc0_0000,
    0x7fa0_0000,
    0x0000_0001,
    0x807f_ffff,
    0x7f7f_ffff,
    0x4f00_0000,
    0xcf00_0000,
    0x3f00_0000,
    0x4020_0000,
    0xc020_0000,
    0x4040_0000,
    0x3fc0_0000,
];
const F64: [u64; 19] = [
    0,
    0x8000_0000_0000_0000,
    0x3ff0_0000_0000_0000,
    0xbff8_0000_0000_0000,
    0x7ff0_0000_0000_0000,
    0xfff0_0000_0000_0000,
    0x7ff8_0000_0000_0000,
    0x7ff4_0000_0000_0000,
    0x0000_0000_0000_0001,
    0x800f_ffff_ffff_ffff,
    0x7fef_ffff_ffff_ffff,
    0x41e0_0000_0000_0000,
    0xc1e0_0000_0000_0000,
    0x3fe0_0000_0000_0000,
    0x4004_0000_0000_0000,
    0xc004_0000_0000_0000,
    0x4008_0000_0000_0000,
    0x43e0_0000_0000_0000,
    0x3ff8_0000_0000_0000,
];
const H16: [u16; 10] = [0, 0x8000, 0x3c00, 0x7c00, 0xfc00, 0x7e00, 0x7d00, 0x0001, 0x03ff, 0x7bff];
const W16: [u16; 6] = [0, 0x7fff, 0x8000, 0xffff, 0x80, 0x7f];
const TEXT: [u8; 10] = [0, b'a', b'a', b'b', b'b', b'c', b'z', 0x7f, 0x80, 0xff];
const MXCSR: [u32; 6] = [0x1f80, 0x1f80, 0x3f80, 0x5f80, 0x7f80, 0x9fc0];

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    fn bits(&mut self, k: u32) -> u64 {
        self.next() >> (64 - k)
    }

    fn f32(&mut self) -> u32 {
        let k = self.below(10);
        if k < 3 {
            F32[self.below(F32.len() as u64) as usize]
        } else if k < 5 {
            ((self.below(600) as i64 - 300) as f32).to_bits()
        } else {
            (self.bits(1) << 31 | (100 + self.below(60)) << 23 | self.bits(23)) as u32
        }
    }

    fn f64(&mut self) -> u64 {
        let k = self.below(10);
        if k < 3 {
            F64[self.below(F64.len() as u64) as usize]
        } else if k < 5 {
            ((self.below(600) as i64 - 300) as f64).to_bits()
        } else {
            self.bits(1) << 63 | (990 + self.below(70)) << 52 | self.bits(52)
        }
    }

    fn val(&mut self, kind: &str, n: usize) -> Vec<u8> {
        let mut b = Vec::new();
        while b.len() < n {
            match kind {
                "s" => {
                    let v = self.f32();
                    b.extend_from_slice(&v.to_le_bytes());
                }
                "d" => {
                    let v = self.f64();
                    b.extend_from_slice(&v.to_le_bytes());
                }
                "h" => {
                    let k = self.below(12) as usize;
                    let v = if k < 10 { H16[k] } else { self.bits(16) as u16 };
                    b.extend_from_slice(&v.to_le_bytes());
                }
                "w" => {
                    let k = self.below(7) as usize;
                    let v = if k < 6 { W16[k] } else { self.bits(16) as u16 };
                    b.extend_from_slice(&v.to_le_bytes());
                }
                "x" => {
                    let v = if self.below(2) == 0 { self.below(70) } else { self.next() };
                    b.extend_from_slice(&v.to_le_bytes());
                }
                "t" => b.push(TEXT[self.below(10) as usize]),
                _ => {
                    let v = self.next();
                    b.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        b
    }

    fn gpr(&mut self, kind: &str) -> u64 {
        if kind == "t" {
            return match self.below(4) {
                0 => 1 << 32 | self.below(20),
                _ => (self.below(41) as i64 - 20) as u64,
            };
        }
        if kind != "s" {
            return self.next();
        }
        match self.below(4) {
            0 => self.next(),
            1 => self.bits(31),
            2 => self.bits(20).wrapping_neg(),
            _ => 0,
        }
    }

    /// The 200 bytes of state, RFLAGS and MXCSR of a case.
    fn state(&mut self, kind: &str, mx: bool) -> (Vec<u8>, u64, u32) {
        let mut st = Vec::new();
        for _ in 0..4 {
            st.extend(self.val(kind, 32));
        }
        st.extend(self.val("i", 16));
        for _ in 0..3 {
            let v = self.gpr(kind);
            st.extend_from_slice(&v.to_le_bytes());
        }
        st.extend(self.val(kind, 32));
        let fl = 0x202 | (self.bits(12) & ARITH);
        let mxcsr = if mx { MXCSR[self.below(6) as usize] } else { 0x1f80 };
        (st, fl, mxcsr)
    }
}

fn chunk(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap())
}

/// The vector state of the native harness: YMM0 to YMM3, MM0 and MM1, RAX, RCX, RDX and the
/// memory at DATA.
fn vec_state(w: &World, st: &X86CpuState) -> Vec<u8> {
    let mut b = Vec::new();
    for r in &st.xmm_regs[..4] {
        for q in &r[..4] {
            b.extend_from_slice(&q.to_le_bytes());
        }
    }
    for r in &st.fpregs[..2] {
        b.extend_from_slice(&r[0].to_le_bytes());
    }
    for r in [R_EAX, R_ECX, R_EDX] {
        b.extend_from_slice(&st.regs[r].to_le_bytes());
    }
    b.extend(w.read(DATA, 32));
    b
}

/// 64-bit mode with SSE and AVX state enabled.
fn long64_avx() -> X86CpuState {
    let mut st = World::long64(CR4_OSFXSR_MASK | CR4_OSXSAVE_MASK);
    st.xcr0 = 7;
    st
}

/// Cases where QEMU and the hardware give different results, and ruvm follows QEMU.
/// maxpd_0: with MXCSR.DAZ set, QEMU's FPU_MAX compares the flushed inputs but returns the
/// unflushed denormal operand, while the hardware returns zero.
const QEMU_RESULT_DIFFS: &[&str] = &["maxpd_0"];

#[test]
fn native_vector_results_match() {
    let w0 = World::new("EPYC");
    for f in
        ["avx", "avx2", "fma", "f16c", "pclmulqdq", "sse4.1", "ssse3", "aes", "sha-ni", "sse4.2"]
    {
        assert!(w0.x86.model().has_feature(f), "{f}");
    }
    let data = include_str!("data/tcg_vec.txt");
    let mut n = 0;
    let mut bad = Vec::new();
    for line in data.lines().filter(|l| !l.starts_with('#')) {
        let p: Vec<&str> = line.split(';').collect();
        assert_eq!(p.len(), 6, "{line}");
        let (name, code) = (p[0], hex(p[1]));
        let d: Vec<&str> = p[2].split(':').collect();
        let (kind, seed, mx) = (d[0], d[1].parse().unwrap(), d[2] == "1");
        let (input, fin, mxin) = Rng(seed).state(kind, mx);
        let mut want = input.clone();
        for c in p[3].split(',').filter(|c| !c.is_empty()) {
            let (i, v) = c.split_once(':').unwrap();
            let i: usize = i.parse().unwrap();
            want[8 * i..8 * i + 8].copy_from_slice(&hex(v));
        }
        let (fout, mxout) = (u64::from_str_radix(p[4], 16).unwrap(), p[5]);
        let mxout = u32::from_str_radix(mxout, 16).unwrap();

        let w = World::new("EPYC");
        let mut st = long64_avx();
        for i in 0..4 {
            for j in 0..4 {
                st.xmm_regs[i][j] = chunk(&input, 4 * i + j);
            }
        }
        for i in 0..2 {
            st.fpregs[i][0] = chunk(&input, 16 + i);
        }
        for (i, r) in [R_EAX, R_ECX, R_EDX].into_iter().enumerate() {
            st.regs[r] = chunk(&input, 18 + i);
        }
        st.regs[R_EDI] = DATA;
        st.rflags = fin;
        st.mxcsr = mxin;
        w.write(DATA, &input[168..]);
        let st = w.run(st, &code);
        n += 1;
        if vector(&st).is_some() {
            bad.push(format!("{name}: exception {:?}", vector(&st)));
            continue;
        }
        let got = vec_state(&w, &st);
        for i in 0..25 {
            if chunk(&got, i) != chunk(&want, i) && !QEMU_RESULT_DIFFS.contains(&name) {
                bad.push(format!(
                    "{name}: chunk {i}: {:016x} != {:016x} (input {:016x})",
                    chunk(&got, i),
                    chunk(&want, i),
                    chunk(&input, i)
                ));
            }
        }
        if st.rflags & ARITH != fout & ARITH {
            bad.push(format!("{name}: rflags {:#x} != {fout:#x}", st.rflags));
        }
        // QEMU raises DE when a half precision input is a denormal; AMD hardware does not.
        let de = if name.starts_with("vcvtph2ps") { 2 } else { 0 };
        if st.mxcsr & !de != mxout & !de {
            bad.push(format!("{name}: mxcsr {:#x} != {mxout:#x} (input {mxin:#x})", st.mxcsr));
        }
    }
    assert!(bad.is_empty(), "{} mismatches:\n{}", bad.len(), bad.join("\n"));
    assert!(n > 2000);
}

#[test]
fn vector_exceptions() {
    let run = |st: X86CpuState, code: &[u8]| vector(&World::new("EPYC").run(st, code));
    let with_cr0 = |cr4: u64, cr0: u64| {
        let mut st = World::long64(cr4);
        st.update_cr0(st.cr0 | cr0);
        st.regs[R_EDI] = DATA;
        st
    };
    // addps xmm0, xmm1 needs CR4.OSFXSR, which QEMU checks before CR0.TS and CR0.EM.
    let addps = [0x0f, 0x58, 0xc1];
    assert_eq!(run(with_cr0(CR4_OSFXSR_MASK, 0), &addps), None);
    assert_eq!(run(with_cr0(0, 0), &addps), Some(6));
    assert_eq!(run(with_cr0(0, CR0_TS_MASK), &addps), Some(6));
    assert_eq!(run(with_cr0(CR4_OSFXSR_MASK, CR0_TS_MASK), &addps), Some(7));
    assert_eq!(run(with_cr0(CR4_OSFXSR_MASK, CR0_EM_MASK), &addps), Some(6));
    // paddb mm0, mm1 does not need CR4.OSFXSR, but the 66 form paddb xmm0, xmm1 does.
    assert_eq!(run(with_cr0(0, 0), &[0x0f, 0xfc, 0xc1]), None);
    assert_eq!(run(with_cr0(0, CR0_TS_MASK), &[0x0f, 0xfc, 0xc1]), Some(7));
    assert_eq!(run(with_cr0(0, 0), &[0x66, 0x0f, 0xfc, 0xc1]), Some(6));
    // A LOCK prefix is #UD.
    assert_eq!(run(with_cr0(CR4_OSFXSR_MASK, 0), &[0xf0, 0x0f, 0x58, 0xc1]), Some(6));
    // movaps and addps need an aligned memory operand, movups does not.
    let mis = |code: &[u8]| run(with_cr0(CR4_OSFXSR_MASK, 0), code);
    assert_eq!(mis(&[0x0f, 0x28, 0x47, 0x01]), Some(13));
    assert_eq!(mis(&[0x0f, 0x58, 0x47, 0x01]), Some(13));
    assert_eq!(mis(&[0x0f, 0x10, 0x47, 0x01]), None);
    // The VEX forms: vmovaps still needs alignment, vaddps does not.
    let mut st = long64_avx();
    st.regs[R_EDI] = DATA;
    assert_eq!(run(st.clone(), &[0xc5, 0xf8, 0x28, 0x47, 0x01]), Some(13));
    assert_eq!(run(st.clone(), &[0xc5, 0xf8, 0x58, 0x47, 0x01]), None);
    // XCR0 without the YMM state makes VEX #UD, as does a LOCK or 66 prefix before VEX.
    let mut no_ymm = st.clone();
    no_ymm.xcr0 = 3;
    assert_eq!(run(no_ymm, &[0xc5, 0xf8, 0x58, 0xc1]), Some(6));
    assert_eq!(run(st.clone(), &[0xf0, 0xc5, 0xf8, 0x58, 0xc1]), Some(6));
    assert_eq!(run(st.clone(), &[0x66, 0xc5, 0xf8, 0x58, 0xc1]), Some(6));
    // VEX.L = 1 is #UD for the 128-bit only vmovd.
    assert_eq!(run(st.clone(), &[0xc5, 0xfd, 0x6e, 0xc0]), Some(6));
    // With AVX2, integer ops take VEX.L = 1: vpaddb ymm0, ymm1, ymm2.
    assert_eq!(run(st, &[0xc5, 0xf5, 0xfc, 0xc2]), None);
}

#[test]
fn vzeroupper_and_vzeroall() {
    let mut st = long64_avx();
    for (i, r) in st.xmm_regs.iter_mut().enumerate() {
        for (j, q) in r.iter_mut().take(4).enumerate() {
            *q = (i * 4 + j + 1) as u64;
        }
    }
    let up = World::new("EPYC").run(st.clone(), &[0xc5, 0xf8, 0x77]);
    assert_eq!(vector(&up), None);
    for (i, r) in up.xmm_regs.iter().take(16).enumerate() {
        assert_eq!(r[..4], [(i * 4 + 1) as u64, (i * 4 + 2) as u64, 0, 0], "{i}");
    }
    let all = World::new("EPYC").run(st, &[0xc5, 0xfc, 0x77]);
    assert_eq!(vector(&all), None);
    for (i, r) in all.xmm_regs.iter().take(16).enumerate() {
        assert_eq!(r[..4], [0; 4], "{i}");
    }
}

#[test]
fn mmx_tags_and_emms() {
    // movq mm0, rax switches to MMX mode: TOP = 0 and every tag valid.
    let mut st = World::long64(0);
    st.regs[R_EAX] = 0x1122_3344_5566_7788;
    st.fpstt = 3;
    st.fptags = [1; 8];
    let mmx = World::new("EPYC").run(st.clone(), &[0x48, 0x0f, 0x6e, 0xc0]);
    assert_eq!(vector(&mmx), None);
    assert_eq!((mmx.fpstt, mmx.fptags, mmx.fpregs[0][0]), (0, [0; 8], 0x1122_3344_5566_7788));
    // emms then marks every register empty.
    let emms = World::new("EPYC").run(st, &[0x48, 0x0f, 0x6e, 0xc0, 0x0f, 0x77]);
    assert_eq!(vector(&emms), None);
    assert_eq!((emms.fpstt, emms.fptags), (0, [1; 8]));
}

#[test]
fn vldmxcsr_vstmxcsr() {
    let w = World::new("EPYC");
    w.w64(DATA, 0x3f80);
    w.w64(DATA + 8, !0);
    let mut st = long64_avx();
    st.regs[R_EDI] = DATA;
    // vldmxcsr [rdi]; vstmxcsr [rdi + 8].
    let st = w.run(st, &[0xc5, 0xf8, 0xae, 0x17, 0xc5, 0xf8, 0xae, 0x5f, 0x08]);
    assert_eq!(vector(&st), None);
    assert_eq!(st.mxcsr, 0x3f80);
    assert_eq!(w.read(DATA + 8, 8), [0x80, 0x3f, 0, 0, 0xff, 0xff, 0xff, 0xff]);
    // VEX.L = 1 is #UD.
    let mut st = long64_avx();
    st.regs[R_EDI] = DATA;
    assert_eq!(vector(&World::new("EPYC").run(st, &[0xc5, 0xfc, 0xae, 0x17])), Some(6));
}

#[test]
fn rip_relative_with_immediate() {
    // pshufd xmm0, [rip + 7], 0x1b: the displacement is relative to the end of the
    // instruction, after the immediate, so the operand is the aligned 16 bytes at 0x1010.
    let w = World::new("EPYC");
    w.write(CODE + 0x10, &hex("01000000020000000300000004000000"));
    let st = w.run(World::long64(CR4_OSFXSR_MASK), &[0x66, 0x0f, 0x70, 0x05, 7, 0, 0, 0, 0x1b]);
    assert_eq!(vector(&st), None);
    assert_eq!(st.xmm_regs[0][..2], [0x0000_0003_0000_0004, 0x0000_0001_0000_0002]);
}

#[test]
fn rcp_and_rsqrt_are_exact() {
    // QEMU computes RCPPS and RSQRTPS exactly and leaves MXCSR alone, where hardware gives
    // 12-bit approximations: rcpps xmm0, xmm1; rsqrtps xmm2, xmm1 on 4, 0.25, 16 and 3.
    let mut st = World::long64(CR4_OSFXSR_MASK);
    st.xmm_regs[1][0] = 0x3e80_0000_4080_0000;
    st.xmm_regs[1][1] = 0x4040_0000_4180_0000;
    let st = World::new("EPYC").run(st, &[0x0f, 0x53, 0xc1, 0x0f, 0x52, 0xd1]);
    assert_eq!(vector(&st), None);
    assert_eq!(st.xmm_regs[0][..2], [0x4080_0000_3e80_0000, 0x3eaa_aaab_3d80_0000]);
    assert_eq!(st.xmm_regs[2][..2], [0x4000_0000_3f00_0000, 0x3f13_cd3a_3e80_0000]);
    assert_eq!(st.mxcsr, 0x1f80);
}

/// 1.0 as a floatx80 register: mantissa and sign plus exponent.
const X80_ONE: [u64; 2] = [0x8000_0000_0000_0000, 0x3fff];

#[test]
fn x87_arith_and_stores() {
    let w = World::new("EPYC");
    w.w64(DATA, 0x3ff8_0000_0000_0000);
    w.w64(DATA + 8, 3);
    w.write(DATA + 16, &[0xff; 16]);
    let mut st = World::long64(0);
    st.regs[R_EDI] = DATA;
    st.regs[R_EAX] = !0;
    // fld qword [rdi]; fiadd dword [rdi + 8]; fstp qword [rdi + 16]; fild dword [rdi + 8];
    // fsqrt; fstp dword [rdi + 24]; fnstsw ax.
    let code = [
        0xdd, 0x07, 0xda, 0x47, 0x08, 0xdd, 0x5f, 0x10, 0xdb, 0x47, 0x08, 0xd9, 0xfa, 0xd9, 0x5f,
        0x18, 0xdf, 0xe0,
    ];
    let st = w.run(st, &code);
    assert_eq!(vector(&st), None);
    assert_eq!(chunk(&w.read(DATA, 32), 2), 0x4012_0000_0000_0000);
    // sqrt(3) rounded to single precision; the precision exception sets FPUS.PE.
    assert_eq!(w.read(DATA + 24, 8), [0xd7, 0xb3, 0xdd, 0x3f, 0xff, 0xff, 0xff, 0xff]);
    assert_eq!((st.fpstt, st.fptags), (0, [1; 8]));
    assert_eq!(st.regs[R_EAX], 0xffff_ffff_ffff_0020);
}

#[test]
fn x87_fcomi_and_fcmov() {
    // fld1; fldz; fcomi st0, st1; setc al; fcmovb st0, st1; fucomip st0, st1.
    let code = [0xd9, 0xe8, 0xd9, 0xee, 0xdb, 0xf1, 0x0f, 0x92, 0xc0, 0xda, 0xc1, 0xdf, 0xe9];
    let mut st = World::long64(0);
    st.regs[R_EAX] = 0;
    st.rflags = 2 | ARITH;
    let st = World::new("EPYC").run(st, &code);
    assert_eq!(vector(&st), None);
    assert_eq!(st.regs[R_EAX], 1);
    // QEMU's FCOMI and FUCOMI leave OF, SF and AF alone, where the hardware clears them.
    assert_eq!(st.rflags & ARITH, ARITH & !(CF | 0x04));
    assert_eq!(st.fpstt, 7);
    assert_eq!(st.fpregs[6], X80_ONE);
    assert_eq!(st.fpregs[7], X80_ONE);
}

#[test]
fn x87_fbst_fbld_round_trip() {
    let w = World::new("EPYC");
    w.w64(DATA, (-1_234_567_890_123i64) as u64);
    let mut st = World::long64(0);
    st.regs[R_EDI] = DATA;
    // fild qword [rdi]; fbstp [rdi + 16]; fbld [rdi + 16]; fistp qword [rdi + 32].
    let code = [0xdf, 0x2f, 0xdf, 0x77, 0x10, 0xdf, 0x67, 0x10, 0xdf, 0x7f, 0x20];
    let st = w.run(st, &code);
    assert_eq!(vector(&st), None);
    assert_eq!(w.read(DATA + 16, 10), [0x23, 0x01, 0x89, 0x67, 0x45, 0x23, 0x01, 0, 0, 0x80]);
    assert_eq!(chunk(&w.read(DATA + 32, 8), 0), (-1_234_567_890_123i64) as u64);
    assert_eq!(st.fpstt, 0);
}

#[test]
fn x87_fnstenv_and_last_pointers() {
    let w = World::new("EPYC");
    w.w64(DATA + 64, 0x4000_0000_0000_0000);
    let mut st = World::long64(0);
    st.regs[R_EDI] = DATA;
    // fld1; fld qword [rdi + 64]; fnstenv [rdi]; rex.w fnstenv [rdi + 32].
    let code = [0xd9, 0xe8, 0xdd, 0x47, 0x40, 0xd9, 0x37, 0x48, 0xd9, 0x77, 0x20];
    let st = w.run(st, &code);
    assert_eq!(vector(&st), None);
    let env = w.read(DATA, 28);
    let l = |i: usize| u32::from_le_bytes(env[i..i + 4].try_into().unwrap());
    assert_eq!((l(0), l(4), l(8)), (0x37f, 0x3000, 0x0fff));
    assert_eq!((l(12), l(16), l(20), l(24)), (CODE as u32 + 2, 8, DATA as u32 + 64, 0x10));
    // QEMU passes dflag - 1 = 2 with REX.W, which still writes the 32-bit layout.
    assert_eq!(w.read(DATA + 32, 28), env);
    assert_eq!((st.fpip, st.fpcs, st.fpdp, st.fpds), (CODE + 2, 8, DATA + 64, 0x10));
    assert_eq!(st.fpuc, 0x37f);
}

#[test]
fn x87_device_not_available() {
    let nm = |cr0: u64, code: &[u8]| {
        let mut st = World::long64(0);
        st.update_cr0(st.cr0 | cr0);
        vector(&World::new("EPYC").run(st, code))
    };
    // fld1 with CR0.TS or CR0.EM set.
    assert_eq!(nm(CR0_TS_MASK, &[0xd9, 0xe8]), Some(7));
    assert_eq!(nm(CR0_EM_MASK, &[0xd9, 0xe8]), Some(7));
    // fwait only checks CR0.TS when CR0.MP is set.
    assert_eq!(nm(CR0_TS_MASK, &[0x9b]), None);
    assert_eq!(nm(CR0_TS_MASK | CR0_MP_MASK, &[0x9b]), Some(7));
    // fnop is the one valid D9 /2 register form.
    assert_eq!(nm(0, &[0xd9, 0xd0]), None);
    assert_eq!(nm(0, &[0xd9, 0xd1]), Some(6));
}

#[test]
fn fxsave_fxrstor() {
    let w = World::new("EPYC");
    w.write(DATA, &[0xff; 512]);
    let mut st = World::long64(CR4_OSFXSR_MASK);
    st.regs[R_EDI] = DATA;
    st.xmm_regs[0][0] = 0x1122_3344_5566_7788;
    // fld1; fxsave [rdi]; fninit; fxrstor [rdi].
    let code = [0xd9, 0xe8, 0x0f, 0xae, 0x07, 0xdb, 0xe3, 0x0f, 0xae, 0x0f];
    let st = w.run(st, &code);
    assert_eq!(vector(&st), None);
    let a = w.read(DATA, 512);
    assert_eq!(a[..6], [0x7f, 0x03, 0x00, 0x38, 0x80, 0x00]);
    assert_eq!(a[8..24], [0; 16]);
    assert_eq!(u32::from_le_bytes(a[24..28].try_into().unwrap()), 0x1f80);
    // ST0 is register 7.
    assert_eq!((chunk(&a, 4), chunk(&a, 5) & 0xffff), (X80_ONE[0], X80_ONE[1]));
    assert_eq!(chunk(&a, 20), 0x1122_3344_5566_7788);
    assert_eq!((st.fpstt, st.fptags[7], st.fpregs[7]), (7, 0, X80_ONE));
    // CR0.TS is #NM.
    let mut st = World::long64(CR4_OSFXSR_MASK);
    st.regs[R_EDI] = DATA;
    st.update_cr0(st.cr0 | CR0_TS_MASK);
    assert_eq!(vector(&World::new("EPYC").run(st, &[0x0f, 0xae, 0x07])), Some(7));
}

#[test]
fn xsave_xrstor() {
    let w = World::new("EPYC");
    w.write(DATA, &[0; 1024]);
    let mut st = long64_avx();
    st.xcr0 = 3;
    st.regs[R_EDI] = DATA;
    st.regs[R_EAX] = 3;
    st.regs[R_EDX] = 0;
    st.xmm_regs[0][1] = 0x99aa_bbcc_ddee_ff00;
    // fld1; xsave [rdi]; fninit; xrstor [rdi].
    let code = [0xd9, 0xe8, 0x0f, 0xae, 0x27, 0xdb, 0xe3, 0x0f, 0xae, 0x2f];
    let st = w.run(st, &code);
    assert_eq!(vector(&st), None);
    let a = w.read(DATA, 576);
    assert_eq!(a[..2], [0x7f, 0x03]);
    assert_eq!(chunk(&a, 21), 0x99aa_bbcc_ddee_ff00);
    assert_eq!(chunk(&a, 64), 3);
    assert_eq!((st.fpstt, st.fptags[7], st.fpregs[7]), (7, 0, X80_ONE));
    // XSAVEOPT writes the same area.
    if w.x86.model().has_feature("xsaveopt") {
        let w2 = World::new("EPYC");
        w2.write(DATA, &[0; 1024]);
        let mut st = long64_avx();
        st.xcr0 = 3;
        st.regs[R_EDI] = DATA;
        st.regs[R_EAX] = 3;
        st.regs[R_EDX] = 0;
        let st = w2.run(st, &[0xd9, 0xe8, 0x0f, 0xae, 0x37]);
        assert_eq!(vector(&st), None);
        assert_eq!(w2.read(DATA, 576)[..32], a[..32]);
        assert_eq!(chunk(&w2.read(DATA, 576), 64), 3);
    }
    // Without CR4.OSXSAVE, XSAVE is #UD.
    let mut st = World::long64(CR4_OSFXSR_MASK);
    st.regs[R_EDI] = DATA;
    assert_eq!(vector(&World::new("EPYC").run(st, &[0x0f, 0xae, 0x27])), Some(6));
}
