// SPDX-License-Identifier: GPL-2.0-or-later

//! System management mode on the x86 TCG front end: SMI entry, the state save area, RSM and
//! SMBASE relocation, checked against QEMU's `smm_helper.c`.
//!
//! The memory map: RAM from 0 to 4 MiB, and SMRAM from 0x38000 to 0x40000 in the vCPU's SMM
//! address space only, over the RAM there. With the reset SMBASE of 0x30000 the SMI handler
//! starts at 0x38000 and the save area is at 0x3fe00, both in SMRAM. The real mode IVT points
//! vector n at a HLT at 0x500 + n; code runs from 0x1000, the GDT is at 0x3000 and the long
//! mode page tables at 0x10000 identity map the first 2 MiB.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use ruvm_jit::cpu_exec::cpu_exec;
use ruvm_jit::{Jit, Vcpu, excp};
use ruvm_mem::{AddressSpace, MemTxAttrs, MemorySystem};
use ruvm_target_x86::cpuid::{Accel, X86Cpu};
use ruvm_target_x86::state::{
    CR0_ET_MASK, CR0_NE_MASK, CR0_PE_MASK, CR0_PG_MASK, CR4_PAE_MASK, DESC_A_MASK, DESC_B_MASK,
    DESC_CS_MASK, DESC_G_MASK, DESC_L_MASK, DESC_P_MASK, DESC_R_MASK, DESC_S_MASK, DESC_W_MASK,
    HF_CS64_MASK, HF_LMA_MASK, HF_SMM_MASK, HF2_NMI_MASK, IrqchipMode, MSR_EFER_LMA, MSR_EFER_LME,
    R_CS, R_DS, R_EAX, R_EBX, R_ECX, R_EDX, R_ESP, ResetConfig, SegmentCache, X86CpuState,
};
use ruvm_target_x86::tcg::env::HF2_SMM_INSIDE_NMI_MASK;
use ruvm_target_x86::tcg::{CPU_INTERRUPT_SMI, X86, create_vcpu, new_jit, save_vcpu};

const U: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const RAM_SIZE: u64 = 0x40_0000;
const CODE: u64 = 0x1000;
const GDT: u64 = 0x3000;
const PML4: u64 = 0x1_0000;
const SMBASE: u64 = 0x3_0000;
const HANDLER: u64 = SMBASE + 0x8000;
/// The save area: SMBASE + 0x8000 + 0x7e00, so offsets below are from SMBASE + 0x8000.
const SM: u64 = SMBASE + 0x8000;

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
/// The segment flags SMI entry loads.
const SMM_FLAGS: u32 = DESC_P_MASK | DESC_S_MASK | DESC_W_MASK | DESC_G_MASK | DESC_A_MASK;

struct World {
    _sys: MemorySystem,
    as_: Arc<AddressSpace>,
    smm: Arc<AddressSpace>,
    jit: Arc<Jit>,
    x86: Arc<X86>,
}

impl World {
    fn new() -> World {
        World::with_model(X86::qemu64())
    }

    /// A model without long mode, which uses the 32-bit save area.
    fn new32() -> World {
        let mut m = X86Cpu::new("qemu64", Accel::Tcg).unwrap();
        m.parse_features("-lm").unwrap();
        let _ = m.realize();
        assert!(!m.has_feature("lm"));
        World::with_model(X86::new(m))
    }

    fn with_model(x86: X86) -> World {
        let sys = MemorySystem::new();
        let root = sys.new_container("system", 1 << 64).unwrap();
        let as_ = sys.address_space_init(root, "memory").unwrap();
        let ram = sys.new_ram("ram", RAM_SIZE).unwrap();
        sys.add_subregion(root, 0, ram).unwrap();
        // The SMM address space: system memory with SMRAM on top, as QEMU's "cpu-smm".
        let smm_root = sys.new_container("memory-smm", 1 << 64).unwrap();
        let alias = sys.new_alias("smm-memory", root, 0, 1 << 64).unwrap();
        sys.add_subregion_overlap(smm_root, 0, alias, 0).unwrap();
        let smram = sys.new_ram("smram", 0x8000).unwrap();
        sys.add_subregion_overlap(smm_root, HANDLER, smram, 1).unwrap();
        let smm = sys.address_space_init(smm_root, "cpu-smm").unwrap();
        let w = World { _sys: sys, as_, smm, jit: new_jit(), x86: Arc::new(x86) };
        for n in 0..32u64 {
            let ent = (0x500 + n) as u32;
            w.write(n * 4, &ent.to_le_bytes());
            w.write(0x500 + n, &[0xf4]);
        }
        // GDT: null, 0x08 64-bit code, 0x10 data, 0x18 32-bit code.
        w.w64(GDT, 0);
        w.w64(GDT + 8, 0x00af_9b00_0000_ffff);
        w.w64(GDT + 16, 0x00cf_9300_0000_ffff);
        w.w64(GDT + 24, 0x00cf_9b00_0000_ffff);
        w.w64(PML4, (PML4 + 0x1000) | 3);
        w.w64(PML4 + 0x1000, (PML4 + 0x2000) | 3);
        w.w64(PML4 + 0x2000, 0x83);
        w
    }

    fn write(&self, addr: u64, bytes: &[u8]) {
        assert!(self.as_.write(addr, U, bytes).is_ok());
    }

    fn w64(&self, addr: u64, v: u64) {
        self.write(addr, &v.to_le_bytes());
    }

    fn r32(&self, addr: u64) -> u32 {
        let mut b = [0; 4];
        assert!(self.as_.read(addr, U, &mut b).is_ok());
        u32::from_le_bytes(b)
    }

    /// Write `bytes` to SMRAM.
    fn smm_write(&self, addr: u64, bytes: &[u8]) {
        assert!(self.smm.write(addr, U, bytes).is_ok());
    }

    fn smm_read(&self, addr: u64, len: usize) -> u64 {
        let mut b = [0; 8];
        assert!(self.smm.read(addr, U, &mut b[..len]).is_ok());
        u64::from_le_bytes(b)
    }

    fn vcpu(&self, st: &X86CpuState) -> Vcpu {
        let mut v = create_vcpu(&self.jit, self.x86.clone(), self.as_.clone(), st);
        v.core.set_address_space(1, self.smm.clone());
        v
    }
}

/// Run `v` until it executes HLT, and return its state.
fn run(v: &mut Vcpu) -> X86CpuState {
    v.shared().halted.store(0, Ordering::Release);
    let mut halted = false;
    for _ in 0..10_000 {
        if cpu_exec(&mut v.cpu()) == excp::HLT {
            halted = true;
            break;
        }
    }
    assert!(halted, "the vCPU did not halt");
    let mut st = X86CpuState::default();
    save_vcpu(v, &mut st);
    st
}

/// Raise an SMI on `v` and run it until it halts.
fn smi(v: &mut Vcpu) -> X86CpuState {
    v.shared().cpu_interrupt(CPU_INTERRUPT_SMI);
    run(v)
}

fn reset() -> X86CpuState {
    let mut st = X86CpuState::new_reset(&ResetConfig {
        is_bsp: true,
        kvm: false,
        irqchip: IrqchipMode::Off,
        cpuid_version: 0x0006_0fb1,
        has_monitor: false,
    });
    assert_eq!(st.smbase, SMBASE as u32);
    st.rflags = 2;
    st.rip = CODE;
    st
}

fn real() -> X86CpuState {
    let mut st = reset();
    let data = DESC_P_MASK | DESC_S_MASK | DESC_W_MASK | DESC_A_MASK;
    let code = DESC_P_MASK | DESC_S_MASK | DESC_CS_MASK | DESC_R_MASK | DESC_A_MASK;
    for s in 0..6 {
        st.load_seg_cache(s, 0, 0, 0xffff, if s == R_CS { code } else { data });
    }
    st.idt = SegmentCache { selector: 0, base: 0, limit: 0x3ff, flags: 0 };
    st.regs[R_ESP] = 0x7000;
    st
}

fn prot32() -> X86CpuState {
    let mut st = reset();
    st.update_cr0(CR0_PE_MASK | CR0_ET_MASK | CR0_NE_MASK);
    st.gdt = SegmentCache { selector: 0, base: GDT, limit: 0x1f, flags: 0 };
    st.load_seg_cache(R_CS, 0x18, 0, 0xffff_ffff, CODE32_FLAGS);
    for s in [0, 2, 3, 4, 5] {
        st.load_seg_cache(s, 0x10, 0, 0xffff_ffff, DATA32_FLAGS);
    }
    st.regs[R_ESP] = 0x2_0000;
    st
}

fn long64() -> X86CpuState {
    let mut st = reset();
    st.cr4 = CR4_PAE_MASK;
    st.efer = MSR_EFER_LME;
    st.cr3 = PML4;
    st.update_cr0(CR0_PE_MASK | CR0_ET_MASK | CR0_NE_MASK | CR0_PG_MASK);
    assert_ne!(st.hflags & HF_LMA_MASK, 0);
    st.gdt = SegmentCache { selector: 0, base: GDT, limit: 0x1f, flags: 0 };
    st.load_seg_cache(R_CS, 0x08, 0, 0xffff_ffff, CODE64_FLAGS);
    for s in [0, 2, 3, 4, 5] {
        st.load_seg_cache(s, 0x10, 0, 0xffff_ffff, DATA32_FLAGS);
    }
    st.regs[R_ESP] = 0x2_0000;
    st
}

/// The state SMI entry sets up, as `do_smm_enter()`.
#[track_caller]
fn check_smm_entry(st: &X86CpuState, smbase: u64) {
    assert_ne!(st.hflags & HF_SMM_MASK, 0);
    assert_eq!(st.hflags & (HF_LMA_MASK | HF_CS64_MASK), 0);
    assert_eq!(st.cr0 & (CR0_PE_MASK | CR0_PG_MASK), 0);
    assert_eq!(st.cr4, 0);
    assert_eq!(st.dr[7], 0x400);
    assert_eq!(st.rflags, 2);
    let cs = &st.segs[R_CS];
    assert_eq!((cs.selector, cs.base, cs.limit), ((smbase >> 4) as u32, smbase, 0xffff_ffff));
    assert_eq!(cs.flags, SMM_FLAGS);
    for s in [0, 2, 3, 4, 5] {
        let d = &st.segs[s];
        assert_eq!((d.selector, d.base, d.limit, d.flags), (0, 0, 0xffff_ffff, SMM_FLAGS));
    }
}

#[test]
fn smi_from_long_mode_saves_the_amd64_area_and_rsm_restores_it() {
    let w = World::new();
    // SMI handler: hlt; rsm. Normal RAM under SMRAM has int3, which would end at 0x504.
    w.write(HANDLER, &[0xcc, 0xcc, 0xcc]);
    w.smm_write(HANDLER, &[0xf4, 0x0f, 0xaa]);
    // wrmsr; rdmsr; hlt, with ECX = MSR_SMI_COUNT.
    w.write(CODE, &[0x0f, 0x30, 0x0f, 0x32, 0xf4]);
    let mut st = long64();
    let vals: Vec<u64> = (0..16u64).map(|i| 0x1111_0000_0000_0000 * (i % 15 + 1) + i).collect();
    st.regs.copy_from_slice(&vals);
    st.regs[R_ECX] = 0x34;
    st.regs[R_EAX] = 0x55;
    st.regs[R_EDX] = 0;
    st.regs[R_ESP] = 0x2_0000;
    st.rflags = 2 | 0x8c5; // CF PF ZF SF OF
    st.dr[6] = 0xffff_0ff1;
    let before = st.clone();
    let mut v = w.vcpu(&st);

    let s = smi(&mut v);
    check_smm_entry(&s, SMBASE);
    assert_eq!(s.rip, 0x8001);
    assert_eq!(s.efer, 0);
    assert_eq!(s.msr_smi_count, 1);
    assert_ne!(s.hflags2 & HF2_NMI_MASK, 0);
    assert_eq!(s.hflags2 & HF2_SMM_INSIDE_NMI_MASK, 0);

    // The save area went to SMRAM; the RAM below is untouched.
    assert_eq!(w.r32(SM + 0x7ff8), 0);
    for i in 0..16 {
        assert_eq!(w.smm_read(SM + 0x7ff8 - i * 8, 8), before.regs[i as usize], "reg {i}");
    }
    assert_eq!(w.smm_read(SM + 0x7f78, 8), CODE);
    assert_eq!(w.smm_read(SM + 0x7f70, 4), 2 | 0x8c5);
    assert_eq!(w.smm_read(SM + 0x7f68, 4), 0xffff_0ff1);
    assert_eq!(w.smm_read(SM + 0x7f60, 4), 0x400);
    assert_eq!(w.smm_read(SM + 0x7f58, 4), before.cr0);
    assert_eq!(w.smm_read(SM + 0x7f50, 8), PML4);
    assert_eq!(w.smm_read(SM + 0x7f48, 4), CR4_PAE_MASK);
    assert_eq!(w.smm_read(SM + 0x7ed0, 8), MSR_EFER_LME | MSR_EFER_LMA);
    assert_eq!(w.smm_read(SM + 0x7efc, 4), 0x0002_0064);
    assert_eq!(w.smm_read(SM + 0x7f00, 4), SMBASE);
    assert_eq!(w.smm_read(SM + 0x7e68, 8), GDT);
    assert_eq!(w.smm_read(SM + 0x7e64, 4), 0x1f);
    // CS, segment 1: selector, attributes, limit, base.
    assert_eq!(w.smm_read(SM + 0x7e10, 2), 0x08);
    assert_eq!(w.smm_read(SM + 0x7e12, 2), u64::from((CODE64_FLAGS >> 8) & 0xf0ff));
    assert_eq!(w.smm_read(SM + 0x7e14, 4), 0xffff_ffff);
    assert_eq!(w.smm_read(SM + 0x7e18, 8), 0);

    let s = run(&mut v);
    assert_eq!(s.hflags & HF_SMM_MASK, 0);
    assert_eq!(s.hflags2 & HF2_NMI_MASK, 0);
    assert_eq!(s.rip, CODE + 5);
    // WRMSR to MSR_SMI_COUNT is ignored; RDMSR reads the count.
    assert_eq!((s.regs[R_EAX], s.regs[R_EDX]), (1, 0));
    for i in 0..16 {
        if i != R_EAX && i != R_EDX {
            assert_eq!(s.regs[i], before.regs[i], "reg {i}");
        }
    }
    assert_eq!(s.rflags, before.rflags);
    assert_eq!((s.cr0, s.cr3, s.cr4, s.efer), (before.cr0, PML4, CR4_PAE_MASK, before.efer));
    assert_eq!(s.dr[6], 0xffff_0ff1);
    assert_eq!(s.dr[7], 0x400);
    assert_ne!(s.hflags & HF_CS64_MASK, 0);
    assert_eq!(s.segs, before.segs);
    assert_eq!(s.gdt, before.gdt);
    assert_eq!(s.smbase, SMBASE as u32);
}

#[test]
fn smi_without_long_mode_saves_the_32_bit_area() {
    let w = World::new32();
    w.smm_write(HANDLER, &[0xf4, 0x0f, 0xaa]);
    w.write(CODE, &[0xf4]);
    let mut st = prot32();
    for (i, r) in st.regs.iter_mut().enumerate().take(8) {
        *r = 0x1010_1010 * (i as u64 + 1);
    }
    st.regs[R_ESP] = 0x2_0000;
    let before = st.clone();
    let mut v = w.vcpu(&st);

    let s = smi(&mut v);
    check_smm_entry(&s, SMBASE);
    assert_eq!(s.rip, 0x8001);
    assert_eq!(w.smm_read(SM + 0x7ffc, 4), before.cr0);
    assert_eq!(w.smm_read(SM + 0x7ff4, 4), 2);
    assert_eq!(w.smm_read(SM + 0x7ff0, 4), CODE);
    // EAX at 0x7fd0 up to EDI at 0x7fec.
    for i in 0..8 {
        assert_eq!(w.smm_read(SM + 0x7fd0 + i * 4, 4), before.regs[i as usize], "reg {i}");
    }
    assert_eq!(w.smm_read(SM + 0x7fc8, 4), 0x400);
    assert_eq!(w.smm_read(SM + 0x7f74, 4), GDT);
    assert_eq!(w.smm_read(SM + 0x7f70, 4), 0x1f);
    // CS: the selector at 0x7fac, then attributes, limit and base from 0x7f90.
    assert_eq!(w.smm_read(SM + 0x7fac, 4), 0x18);
    assert_eq!(w.smm_read(SM + 0x7f90, 4), u64::from((CODE32_FLAGS >> 8) & 0xf0ff));
    assert_eq!(w.smm_read(SM + 0x7f94, 4), 0xffff_ffff);
    assert_eq!(w.smm_read(SM + 0x7f98, 4), 0);
    // DS, segment 3: the selector at 0x7fb4, the rest from 0x7f2c.
    assert_eq!(w.smm_read(SM + 0x7fb4, 4), 0x10);
    assert_eq!(w.smm_read(SM + 0x7f2c, 4), u64::from((DATA32_FLAGS >> 8) & 0xf0ff));
    assert_eq!(w.smm_read(SM + 0x7efc, 4), 0x0002_0000);
    assert_eq!(w.smm_read(SM + 0x7ef8, 4), SMBASE);

    let s = run(&mut v);
    assert_eq!(s.hflags & HF_SMM_MASK, 0);
    assert_eq!(s.rip, CODE + 1);
    assert_eq!(s.regs[..8], before.regs[..8]);
    assert_eq!(s.cr0, before.cr0);
    assert_eq!(s.segs, before.segs);
    assert_eq!(s.hflags, before.hflags);
}

#[test]
fn rsm_relocates_smbase() {
    let w = World::new();
    // mov dword [0x3ff00], 0x50000 (addr32, data32); rsm.
    w.smm_write(
        HANDLER,
        &[0x66, 0x67, 0xc7, 0x05, 0x00, 0xff, 0x03, 0x00, 0x00, 0x00, 0x05, 0x00, 0x0f, 0xaa],
    );
    // The relocated handler, in normal RAM: hlt.
    w.write(0x5_8000, &[0xf4]);
    w.write(CODE, &[0xf4]);
    let mut v = w.vcpu(&long64());
    let s = smi(&mut v);
    assert_eq!(s.hflags & HF_SMM_MASK, 0);
    assert_eq!(s.rip, CODE + 1);
    assert_eq!(s.smbase, 0x5_0000);

    let s = smi(&mut v);
    check_smm_entry(&s, 0x5_0000);
    assert_eq!(s.rip, 0x8001);
    assert_eq!(s.msr_smi_count, 2);
    assert_eq!(w.r32(0x5_8000 + 0x7efc), 0x0002_0064);
}

#[test]
fn smm_reads_smram_and_flushes_the_tlb() {
    let w = World::new();
    w.write(HANDLER, &0x1122_3344u32.to_le_bytes());
    // With DS at 0x30000: mov eax, [0x8000]; mov ebx, eax; hlt; mov eax, [0x8000]; hlt
    w.write(CODE, &[0x66, 0xa1, 0x00, 0x80, 0x66, 0x89, 0xc3, 0xf4, 0x66, 0xa1, 0x00, 0x80, 0xf4]);
    // mov eax, [0x38000]; mov [0x9000], eax; rsm, with 32-bit addresses
    let handler = [
        0x66, 0x67, 0xa1, 0x00, 0x80, 0x03, 0x00, 0x66, 0x67, 0xa3, 0x00, 0x90, 0x00, 0x00, 0x0f,
        0xaa,
    ];
    w.smm_write(HANDLER, &handler);
    let mut st = real();
    let data = st.segs[R_DS].flags;
    st.load_seg_cache(R_DS, 0x3000, SMBASE, 0xffff, data);
    let mut v = w.vcpu(&st);
    let s = run(&mut v);
    assert_eq!(s.regs[R_EBX], 0x1122_3344);
    // The handler read SMRAM, although the TLB had the page from outside SMM.
    let s = smi(&mut v);
    assert_eq!(w.r32(0x9000), u32::from_le_bytes(handler[..4].try_into().unwrap()));
    // And outside SMM the same address is RAM again.
    assert_eq!(s.rip, CODE + 13);
    assert_eq!(s.regs[R_EAX], 0x1122_3344);
    assert_eq!(s.regs[R_EBX], 0x1122_3344);
}

#[test]
fn rsm_outside_smm_is_ud() {
    let w = World::new();
    w.write(CODE, &[0x0f, 0xaa]);
    let mut v = w.vcpu(&real());
    let s = run(&mut v);
    // Vector 6 halts at 0x506.
    assert_eq!(s.rip, 0x507);
    assert_eq!(s.msr_smi_count, 0);
}

#[test]
fn smi_masks_nmis_and_smis_until_rsm() {
    let w = World::new();
    w.smm_write(HANDLER, &[0xf4, 0x0f, 0xaa]);
    w.write(CODE, &[0xf4]);

    // An SMI while an NMI is being handled leaves NMIs blocked after RSM.
    let mut st = long64();
    st.hflags2 |= HF2_NMI_MASK;
    let mut v = w.vcpu(&st);
    let s = smi(&mut v);
    assert_ne!(s.hflags2 & HF2_SMM_INSIDE_NMI_MASK, 0);
    let s = run(&mut v);
    assert_eq!(s.hflags2 & (HF2_NMI_MASK | HF2_SMM_INSIDE_NMI_MASK), HF2_NMI_MASK);

    // A second SMI raised in SMM waits for RSM, then enters SMM again at once.
    let mut v = w.vcpu(&long64());
    let s = smi(&mut v);
    assert_eq!((s.rip, s.msr_smi_count), (0x8001, 1));
    v.shared().cpu_interrupt(CPU_INTERRUPT_SMI);
    let s = run(&mut v);
    assert_ne!(s.hflags & HF_SMM_MASK, 0);
    assert_eq!((s.rip, s.msr_smi_count), (0x8001, 2));
    // The second save area holds the state at the point RSM returned to.
    assert_eq!(w.smm_read(SM + 0x7f78, 8), CODE);
    let s = run(&mut v);
    assert_eq!(s.hflags & HF_SMM_MASK, 0);
    assert_eq!(s.rip, CODE + 1);
}
