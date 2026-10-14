// SPDX-License-Identifier: GPL-2.0-or-later

//! System management mode, the port of `target/i386/tcg/system/smm_helper.c`: SMI entry,
//! RSM and the state save area at SMBASE + 0xfe00.
//!
//! In SMM every access carries `MemTxAttrs::secure`, as `cpu_get_mem_attrs()` sets it, and
//! [`X86`](super::X86) sends such accesses to address space 1 (`X86ASIdx_SMM`), where the
//! machine puts SMRAM over system memory. Without that address space they go to the normal
//! one. The A20 gate is open in SMM, as `x86_get_a20_mask()` has it.
//!
//! The save area layout is the AMD64 one (revision 0x20064) when the model has long mode, and
//! the 32-bit one (revision 0x20000) otherwise, as in QEMU. Both have SMBASE relocation.

use std::sync::Arc;

use ruvm_jit::Cpu;
use ruvm_jit::cputlb::tlb_flush;
use ruvm_mem::{AddressSpace, Endian, MemTxAttrs};

use super::env::{
    A20_MASK, CC_A, CC_C, CC_O, CC_P, CC_S, CC_Z, DF_MASK, EFER, EIP, GDT, HF2_SMM_INSIDE_NMI_MASK,
    HFLAGS, HFLAGS2, IDT, LDT, SEG_BASE, SEG_FLAGS, SEG_LIMIT, SEG_SELECTOR, SMBASE, SMI_COUNT, TR,
    compute_eflags, cr, dr, ld32, ld64, load_eflags, reg, seg, st32, st64,
};
use super::helpers::{load_efer, update_cr0, update_cr3, update_cr4};
use super::seg::load_seg_cache;
use super::{X86, x86_of};
use crate::state::{
    CR0_EM_MASK, CR0_PE_MASK, CR0_PG_MASK, CR0_TS_MASK, DESC_A_MASK, DESC_G_MASK, DESC_P_MASK,
    DESC_S_MASK, DESC_W_MASK, DR6_FIXED_1, DR7_FIXED_1, HF_SMM_MASK, HF2_NMI_MASK, R_CS,
};

/// The SMM revision identifier of the AMD64 save area.
const REVISION_64: u32 = 0x0002_0064;
/// The SMM revision identifier of the 32-bit save area.
const REVISION_32: u32 = 0x0002_0000;
/// The revision bit that says SMBASE relocation is supported.
const REVISION_SMBASE_RELOC: u32 = 0x0002_0000;

/// The EFLAGS bits that SMI entry and RSM load: all but the arithmetic flags and DF, which
/// come through `CC_SRC` and `DF`.
const EFLAGS_UPDATE: u32 = !(CC_O | CC_S | CC_Z | CC_A | CC_P | CC_C | DF_MASK);

/// The segment flags SMI entry loads: present, writable, 4 GiB, accessed.
const SMM_SEG_FLAGS: u32 = DESC_P_MASK | DESC_S_MASK | DESC_W_MASK | DESC_G_MASK | DESC_A_MASK;

/// `cpu_get_mem_attrs()`: accesses made in SMM are secure.
pub(crate) fn mem_attrs(env: &[u8]) -> MemTxAttrs {
    MemTxAttrs::default().with_secure(ld32(env, HFLAGS) & HF_SMM_MASK != 0)
}

/// `x86_get_a20_mask()`: A20 is always enabled in SMM.
pub(crate) fn a20_mask(env: &[u8]) -> u64 {
    if ld32(env, HFLAGS) & HF_SMM_MASK != 0 { !0 } else { ld64(env, A20_MASK) }
}

/// The state save area of a vCPU, reached with `x86_ld*_phys()` and `x86_st*_phys()`: the
/// address space and attributes of the vCPU's current mode, at SMBASE + 0x8000.
struct SmState {
    as_: Arc<AddressSpace>,
    attrs: MemTxAttrs,
    base: u64,
}

impl SmState {
    fn new(cpu: &Cpu<'_>) -> SmState {
        let attrs = mem_attrs(cpu.env);
        SmState {
            as_: Arc::clone(cpu.core.address_space_for(attrs)),
            attrs,
            base: u64::from(ld32(cpu.env, SMBASE)) + 0x8000,
        }
    }

    fn ld(&self, off: u64, size: u32) -> u64 {
        self.as_.load(self.base + off, size, Endian::Little, self.attrs).0
    }

    fn st(&self, off: u64, size: u32, v: u64) {
        let _ = self.as_.store(self.base + off, size, v, Endian::Little, self.attrs);
    }

    fn lduw(&self, off: u64) -> u32 {
        self.ld(off, 2) as u32
    }

    fn ldl(&self, off: u64) -> u32 {
        self.ld(off, 4) as u32
    }

    fn ldq(&self, off: u64) -> u64 {
        self.ld(off, 8)
    }

    fn stw(&self, off: u64, v: u32) {
        self.st(off, 2, u64::from(v & 0xffff));
    }

    fn stl(&self, off: u64, v: u32) {
        self.st(off, 4, u64::from(v));
    }

    fn stq(&self, off: u64, v: u64) {
        self.st(off, 8, v);
    }
}

/// The selector, base, limit and flags of the segment register at `off` in the state buffer.
fn seg_parts(env: &[u8], off: usize) -> (u32, u64, u32, u32) {
    (
        ld32(env, off + SEG_SELECTOR),
        ld64(env, off + SEG_BASE),
        ld32(env, off + SEG_LIMIT),
        ld32(env, off + SEG_FLAGS),
    )
}

fn set_seg_parts(env: &mut [u8], off: usize, selector: u32, base: u64, limit: u32, flags: u32) {
    st32(env, off + SEG_SELECTOR, selector);
    st64(env, off + SEG_BASE, base);
    st32(env, off + SEG_LIMIT, limit);
    st32(env, off + SEG_FLAGS, flags);
}

/// The save area attribute word of segment flags, and back.
fn attr_of(flags: u32) -> u32 {
    (flags >> 8) & 0xf0ff
}

fn flags_of(attr: u32) -> u32 {
    (attr & 0xf0ff) << 8
}

/// Where the 32-bit save area keeps the attributes, limit and base of segment `i`.
fn seg_offset_32(i: usize) -> u64 {
    if i < 3 { 0x7f84 + i as u64 * 12 } else { 0x7f2c + (i as u64 - 3) * 12 }
}

/// `sm_state_init_64()`.
fn sm_state_init_64(cpu: &Cpu<'_>, sm: &SmState) {
    let env = &*cpu.env;
    for i in 0..6 {
        let off = 0x7e00 + i as u64 * 16;
        let (sel, base, limit, flags) = seg_parts(env, seg(i));
        sm.stw(off, sel);
        sm.stw(off + 2, attr_of(flags));
        sm.stl(off + 4, limit);
        sm.stq(off + 8, base);
    }

    sm.stq(0x7e68, ld64(env, GDT + SEG_BASE));
    sm.stl(0x7e64, ld32(env, GDT + SEG_LIMIT));

    let (sel, base, limit, flags) = seg_parts(env, LDT);
    sm.stw(0x7e70, sel);
    sm.stq(0x7e78, base);
    sm.stl(0x7e74, limit);
    sm.stw(0x7e72, attr_of(flags));

    sm.stq(0x7e88, ld64(env, IDT + SEG_BASE));
    sm.stl(0x7e84, ld32(env, IDT + SEG_LIMIT));

    let (sel, base, limit, flags) = seg_parts(env, TR);
    sm.stw(0x7e90, sel);
    sm.stq(0x7e98, base);
    sm.stl(0x7e94, limit);
    sm.stw(0x7e92, attr_of(flags));

    // QEMU keeps EFER where the SDM has IA32_BNDCFGS or a reserved area.
    sm.stq(0x7ed0, ld64(env, EFER));

    for i in 0..16 {
        sm.stq(0x7ff8 - i as u64 * 8, ld64(env, reg(i)));
    }
    sm.stq(0x7f78, ld64(env, EIP));
    sm.stl(0x7f70, compute_eflags(env));
    sm.stl(0x7f68, ld64(env, dr(6)) as u32);
    sm.stl(0x7f60, ld64(env, dr(7)) as u32);

    sm.stl(0x7f48, ld64(env, cr(4)) as u32);
    sm.stq(0x7f50, ld64(env, cr(3)));
    sm.stl(0x7f58, ld64(env, cr(0)) as u32);

    sm.stl(0x7efc, REVISION_64);
    sm.stl(0x7f00, ld32(env, SMBASE));
}

/// `sm_state_init_32()`.
fn sm_state_init_32(cpu: &Cpu<'_>, sm: &SmState) {
    let env = &*cpu.env;
    sm.stl(0x7ffc, ld64(env, cr(0)) as u32);
    sm.stl(0x7ff8, ld64(env, cr(3)) as u32);
    sm.stl(0x7ff4, compute_eflags(env));
    sm.stl(0x7ff0, ld64(env, EIP) as u32);
    // EAX at 0x7fd0 up to EDI at 0x7fec.
    for i in 0..8 {
        sm.stl(0x7fd0 + i as u64 * 4, ld64(env, reg(i)) as u32);
    }
    sm.stl(0x7fcc, ld64(env, dr(6)) as u32);
    sm.stl(0x7fc8, ld64(env, dr(7)) as u32);

    let (sel, base, limit, flags) = seg_parts(env, TR);
    sm.stl(0x7fc4, sel);
    sm.stl(0x7f64, base as u32);
    sm.stl(0x7f60, limit);
    sm.stl(0x7f5c, attr_of(flags));

    let (sel, base, limit, flags) = seg_parts(env, LDT);
    sm.stl(0x7fc0, sel);
    sm.stl(0x7f80, base as u32);
    sm.stl(0x7f7c, limit);
    sm.stl(0x7f78, attr_of(flags));

    sm.stl(0x7f74, ld64(env, GDT + SEG_BASE) as u32);
    sm.stl(0x7f70, ld32(env, GDT + SEG_LIMIT));

    sm.stl(0x7f58, ld64(env, IDT + SEG_BASE) as u32);
    sm.stl(0x7f54, ld32(env, IDT + SEG_LIMIT));

    for i in 0..6 {
        let off = seg_offset_32(i);
        let (sel, base, limit, flags) = seg_parts(env, seg(i));
        sm.stl(0x7fa8 + i as u64 * 4, sel);
        sm.stl(off + 8, base as u32);
        sm.stl(off + 4, limit);
        sm.stl(off, attr_of(flags));
    }
    sm.stl(0x7f14, ld64(env, cr(4)) as u32);

    sm.stl(0x7efc, REVISION_32);
    sm.stl(0x7ef8, ld32(env, SMBASE));
}

/// `do_smm_enter()`: take an SMI, saving the state at SMBASE + 0xfe00 and starting the
/// handler at SMBASE + 0x8000 in a real mode like state with 4 GiB segments.
///
/// The TLB is flushed once the vCPU is in SMM, since its entries carry the memory
/// attributes of the mode they were filled in.
pub(crate) fn do_smm_enter(cpu: &mut Cpu<'_>, x: &X86) {
    let count = ld64(cpu.env, SMI_COUNT).wrapping_add(1);
    st64(cpu.env, SMI_COUNT, count);
    let hf = ld32(cpu.env, HFLAGS) | HF_SMM_MASK;
    st32(cpu.env, HFLAGS, hf);
    let mut hf2 = ld32(cpu.env, HFLAGS2);
    if hf2 & HF2_NMI_MASK != 0 {
        hf2 |= HF2_SMM_INSIDE_NMI_MASK;
    } else {
        hf2 |= HF2_NMI_MASK;
    }
    st32(cpu.env, HFLAGS2, hf2);
    tlb_flush(cpu);

    let sm = SmState::new(cpu);
    if x.model().has_feature("lm") {
        sm_state_init_64(cpu, &sm);
        load_efer(cpu, 0);
    } else {
        sm_state_init_32(cpu, &sm);
    }

    load_eflags(cpu.env, 0, EFLAGS_UPDATE);
    st64(cpu.env, EIP, 0x8000);
    let cr0 = ld64(cpu.env, cr(0)) & !(CR0_PE_MASK | CR0_EM_MASK | CR0_TS_MASK | CR0_PG_MASK);
    update_cr0(cpu, cr0);
    update_cr4(cpu, 0);
    // helper_set_dr(env, 7, 0x400).
    st64(cpu.env, dr(7), DR7_FIXED_1);

    let smbase = ld32(cpu.env, SMBASE);
    load_seg_cache(
        cpu.env,
        R_CS,
        (smbase >> 4) & 0xffff,
        u64::from(smbase),
        0xffff_ffff,
        SMM_SEG_FLAGS,
    );
    for s in 0..6 {
        if s != R_CS {
            load_seg_cache(cpu.env, s, 0, 0, 0xffff_ffff, SMM_SEG_FLAGS);
        }
    }
}

/// `rsm_load_regs_64()`.
fn rsm_load_regs_64(cpu: &mut Cpu<'_>, sm: &SmState) {
    load_efer(cpu, sm.ldq(0x7ed0));

    let env = &mut *cpu.env;
    st64(env, GDT + SEG_BASE, sm.ldq(0x7e68));
    st32(env, GDT + SEG_LIMIT, sm.ldl(0x7e64));

    set_seg_parts(
        env,
        LDT,
        sm.lduw(0x7e70),
        sm.ldq(0x7e78),
        sm.ldl(0x7e74),
        flags_of(sm.lduw(0x7e72)),
    );

    st64(env, IDT + SEG_BASE, sm.ldq(0x7e88));
    st32(env, IDT + SEG_LIMIT, sm.ldl(0x7e84));

    set_seg_parts(
        env,
        TR,
        sm.lduw(0x7e90),
        sm.ldq(0x7e98),
        sm.ldl(0x7e94),
        flags_of(sm.lduw(0x7e92)),
    );

    for i in 0..16 {
        st64(env, reg(i), sm.ldq(0x7ff8 - i as u64 * 8));
    }
    st64(env, EIP, sm.ldq(0x7f78));
    load_eflags(env, sm.ldl(0x7f70), EFLAGS_UPDATE);
    st64(env, dr(6), u64::from(sm.ldl(0x7f68)) | DR6_FIXED_1);
    st64(env, dr(7), u64::from(sm.ldl(0x7f60)) | DR7_FIXED_1);

    update_cr4(cpu, u64::from(sm.ldl(0x7f48)));
    update_cr3(cpu, sm.ldq(0x7f50));
    update_cr0(cpu, u64::from(sm.ldl(0x7f58)));

    for i in 0..6 {
        let off = 0x7e00 + i as u64 * 16;
        load_seg_cache(
            cpu.env,
            i,
            sm.lduw(off),
            sm.ldq(off + 8),
            sm.ldl(off + 4),
            flags_of(sm.lduw(off + 2)),
        );
    }

    if sm.ldl(0x7efc) & REVISION_SMBASE_RELOC != 0 {
        st32(cpu.env, SMBASE, sm.ldl(0x7f00));
    }
}

/// `rsm_load_regs_32()`.
fn rsm_load_regs_32(cpu: &mut Cpu<'_>, sm: &SmState) {
    update_cr0(cpu, u64::from(sm.ldl(0x7ffc)));
    update_cr3(cpu, u64::from(sm.ldl(0x7ff8)));

    let env = &mut *cpu.env;
    load_eflags(env, sm.ldl(0x7ff4), EFLAGS_UPDATE);
    st64(env, EIP, u64::from(sm.ldl(0x7ff0)));
    for i in 0..8 {
        st64(env, reg(i), u64::from(sm.ldl(0x7fd0 + i as u64 * 4)));
    }
    st64(env, dr(6), u64::from(sm.ldl(0x7fcc)) | DR6_FIXED_1);
    st64(env, dr(7), u64::from(sm.ldl(0x7fc8)) | DR7_FIXED_1);

    set_seg_parts(
        env,
        TR,
        sm.ldl(0x7fc4) & 0xffff,
        u64::from(sm.ldl(0x7f64)),
        sm.ldl(0x7f60),
        flags_of(sm.ldl(0x7f5c)),
    );
    set_seg_parts(
        env,
        LDT,
        sm.ldl(0x7fc0) & 0xffff,
        u64::from(sm.ldl(0x7f80)),
        sm.ldl(0x7f7c),
        flags_of(sm.ldl(0x7f78)),
    );

    st64(env, GDT + SEG_BASE, u64::from(sm.ldl(0x7f74)));
    st32(env, GDT + SEG_LIMIT, sm.ldl(0x7f70));

    st64(env, IDT + SEG_BASE, u64::from(sm.ldl(0x7f58)));
    st32(env, IDT + SEG_LIMIT, sm.ldl(0x7f54));

    for i in 0..6 {
        let off = seg_offset_32(i);
        load_seg_cache(
            env,
            i,
            sm.ldl(0x7fa8 + i as u64 * 4) & 0xffff,
            u64::from(sm.ldl(off + 8)),
            sm.ldl(off + 4),
            flags_of(sm.ldl(off)),
        );
    }
    update_cr4(cpu, u64::from(sm.ldl(0x7f14)));

    if sm.ldl(0x7efc) & REVISION_SMBASE_RELOC != 0 {
        st32(cpu.env, SMBASE, sm.ldl(0x7ef8));
    }
}

/// `helper_rsm()`: reload the state saved at SMI entry and leave SMM.
///
/// The TLB is flushed after leaving SMM, as on entry.
pub(crate) fn helper_rsm(cpu: &mut Cpu<'_>) {
    let ops = cpu.ops();
    let lm = x86_of(&ops).model().has_feature("lm");
    let sm = SmState::new(cpu);
    if lm {
        rsm_load_regs_64(cpu, &sm);
    } else {
        rsm_load_regs_32(cpu, &sm);
    }

    let mut hf2 = ld32(cpu.env, HFLAGS2);
    if hf2 & HF2_SMM_INSIDE_NMI_MASK == 0 {
        hf2 &= !HF2_NMI_MASK;
    }
    hf2 &= !HF2_SMM_INSIDE_NMI_MASK;
    st32(cpu.env, HFLAGS2, hf2);
    let hf = ld32(cpu.env, HFLAGS) & !HF_SMM_MASK;
    st32(cpu.env, HFLAGS, hf);
    tlb_flush(cpu);
}
