// SPDX-License-Identifier: GPL-2.0-or-later

//! The layout of `CPUX86State` in the runtime's CPU state buffer, the part the generated code
//! addresses by offset, and the conversion to and from [`X86CpuState`].
//!
//! Every offset here is absolute in the buffer, so it already includes
//! [`ENV_TARGET_OFFSET`]. Integers are little endian, as on every host the runtime supports.

use ruvm_jit::ENV_TARGET_OFFSET;

use crate::state::{CPU_NB_REGS, SegmentCache, X86CpuState};

const B: usize = ENV_TARGET_OFFSET;

/// `regs[16]`, 64 bits each.
pub const REGS: usize = B;
/// `eip`.
pub const EIP: usize = B + 128;
/// `eflags` without the arithmetic flags and DF, which live in `cc_*` and `df`.
pub const EFLAGS: usize = B + 136;
/// `cc_dst`.
pub const CC_DST: usize = B + 144;
/// `cc_src`.
pub const CC_SRC: usize = B + 152;
/// `cc_src2`.
pub const CC_SRC2: usize = B + 160;
/// `cc_op`, 32 bits.
pub const CC_OP: usize = B + 168;
/// `df`, 32 bits: 1 or -1.
pub const DF: usize = B + 172;
/// `hflags`, 32 bits.
pub const HFLAGS: usize = B + 176;
/// `hflags2`, 32 bits.
pub const HFLAGS2: usize = B + 180;
/// `segs[6]`, [`SEG_SIZE`] bytes each.
pub const SEGS: usize = B + 184;
/// The size of one segment cache: selector (32), pad (32), base (64), limit (32), flags (32).
pub const SEG_SIZE: usize = 24;
/// Selector offset in a segment cache.
pub const SEG_SELECTOR: usize = 0;
/// Base offset in a segment cache.
pub const SEG_BASE: usize = 8;
/// Limit offset in a segment cache.
pub const SEG_LIMIT: usize = 16;
/// Flags offset in a segment cache.
pub const SEG_FLAGS: usize = 20;
/// `ldt`.
pub const LDT: usize = SEGS + 6 * SEG_SIZE;
/// `tr`.
pub const TR: usize = LDT + SEG_SIZE;
/// `gdt` (base and limit only).
pub const GDT: usize = TR + SEG_SIZE;
/// `idt` (base and limit only).
pub const IDT: usize = GDT + SEG_SIZE;
/// `cr[0..5]`, 64 bits each. `cr[1]` is unused.
pub const CR: usize = IDT + SEG_SIZE;
/// The task priority register, which is CR8.
pub const CR8: usize = CR + 40;
/// `efer`.
pub const EFER: usize = CR8 + 8;
/// `star`.
pub const STAR: usize = EFER + 8;
/// `lstar`.
pub const LSTAR: usize = STAR + 8;
/// `cstar`.
pub const CSTAR: usize = LSTAR + 8;
/// `fmask`.
pub const FMASK: usize = CSTAR + 8;
/// `kernelgsbase`.
pub const KERNELGSBASE: usize = FMASK + 8;
/// `sysenter_cs`.
pub const SYSENTER_CS: usize = KERNELGSBASE + 8;
/// `sysenter_esp`.
pub const SYSENTER_ESP: usize = SYSENTER_CS + 8;
/// `sysenter_eip`.
pub const SYSENTER_EIP: usize = SYSENTER_ESP + 8;
/// `dr[8]`.
pub const DR: usize = SYSENTER_EIP + 8;
/// `a20_mask`, 64 bits here.
pub const A20_MASK: usize = DR + 64;
/// `error_code`, 32 bits.
pub const ERROR_CODE: usize = A20_MASK + 8;
/// `exception_is_int`, 32 bits.
pub const EXCEPTION_IS_INT: usize = ERROR_CODE + 4;
/// `exception_next_eip`.
pub const EXCEPTION_NEXT_EIP: usize = EXCEPTION_IS_INT + 4;
/// `old_exception`, 32 bits.
pub const OLD_EXCEPTION: usize = EXCEPTION_NEXT_EIP + 8;
/// Unused, keeps the next field aligned.
pub const PAD0: usize = OLD_EXCEPTION + 4;
/// `tsc_offset`.
pub const TSC_OFFSET: usize = PAD0 + 4;
/// `pat`.
pub const PAT: usize = TSC_OFFSET + 8;
/// `apic_base`.
pub const APIC_BASE: usize = PAT + 8;
/// `tsc_aux`.
pub const TSC_AUX: usize = APIC_BASE + 8;
/// `msr_ia32_misc_enable`.
pub const MISC_ENABLE: usize = TSC_AUX + 8;
/// `xcr0`.
pub const XCR0: usize = MISC_ENABLE + 8;
/// `fpstt`, 32 bits: the x87 top of stack.
pub const FPSTT: usize = XCR0 + 8;
/// `fpus`, 32 bits: the x87 status word. TOP lives in [`FPSTT`].
pub const FPUS: usize = FPSTT + 4;
/// `fpuc`, 32 bits: the x87 control word.
pub const FPUC: usize = FPUS + 4;
/// `fpop`, 32 bits: the last x87 opcode.
pub const FPOP: usize = FPUC + 4;
/// `fptags[8]`, one byte each, 1 meaning empty.
pub const FPTAGS: usize = FPOP + 4;
/// `fpip`: the last x87 instruction pointer.
pub const FPIP: usize = FPTAGS + 8;
/// `fpdp`: the last x87 data pointer.
pub const FPDP: usize = FPIP + 8;
/// `fpcs`, 32 bits.
pub const FPCS: usize = FPDP + 8;
/// `fpds`, 32 bits.
pub const FPDS: usize = FPCS + 4;
/// `fpregs[8]`, [`FPREG_SIZE`] bytes each: the significand (also the MMX register), then
/// the sign and exponent.
pub const FPREGS: usize = FPDS + 4;
/// The size of one x87 register slot.
pub const FPREG_SIZE: usize = 16;
/// `ft0`, the x87 temporary, laid out like an x87 register.
pub const FT0: usize = FPREGS + 8 * FPREG_SIZE;
/// `mxcsr`, 32 bits.
pub const MXCSR: usize = FT0 + FPREG_SIZE;
/// `pkru`, 32 bits.
pub const PKRU: usize = MXCSR + 4;
/// `xmm_regs[32]`, [`ZMM_SIZE`] bytes each.
pub const XMM_REGS: usize = PKRU + 4;
/// The size of one vector register.
pub const ZMM_SIZE: usize = 64;
/// `xmm_t0`, the vector operand loaded from memory.
pub const XMM_T0: usize = XMM_REGS + 32 * ZMM_SIZE;
/// `mmx_t0`, the MMX operand loaded from memory.
pub const MMX_T0: usize = XMM_T0 + ZMM_SIZE;
/// The size of the whole buffer.
pub const ENV_SIZE: usize = MMX_T0 + 8;

/// The offset of general purpose register `r`.
pub const fn reg(r: usize) -> usize {
    REGS + 8 * r
}

/// The offset of the segment cache of segment register `s`.
pub const fn seg(s: usize) -> usize {
    SEGS + SEG_SIZE * s
}

/// The offset of physical x87 register `n`.
pub const fn fpreg(n: usize) -> usize {
    FPREGS + FPREG_SIZE * n
}

/// The offset of vector register `n`.
pub const fn zmm(n: usize) -> usize {
    XMM_REGS + ZMM_SIZE * n
}

/// The offset of control register `n` (0, 2, 3 or 4).
pub const fn cr(n: usize) -> usize {
    CR + 8 * n
}

/// The offset of debug register `n`.
pub const fn dr(n: usize) -> usize {
    DR + 8 * n
}

// The EFLAGS bits.

/// Carry.
pub const CC_C: u32 = 0x0001;
/// Parity.
pub const CC_P: u32 = 0x0004;
/// Adjust.
pub const CC_A: u32 = 0x0010;
/// Zero.
pub const CC_Z: u32 = 0x0040;
/// Sign.
pub const CC_S: u32 = 0x0080;
/// Overflow.
pub const CC_O: u32 = 0x0800;
/// The arithmetic flags.
pub const CC_MASK: u32 = CC_C | CC_P | CC_A | CC_Z | CC_S | CC_O;
/// Trap.
pub const TF_MASK: u32 = 0x0000_0100;
/// Interrupt enable.
pub const IF_MASK: u32 = 0x0000_0200;
/// Direction.
pub const DF_MASK: u32 = 0x0000_0400;
/// I/O privilege level.
pub const IOPL_MASK: u32 = 0x0000_3000;
/// The IOPL shift.
pub const IOPL_SHIFT: u32 = 12;
/// Nested task.
pub const NT_MASK: u32 = 0x0000_4000;
/// Resume.
pub const RF_MASK: u32 = 0x0001_0000;
/// Virtual 8086 mode.
pub const VM_MASK: u32 = 0x0002_0000;
/// Alignment check.
pub const AC_MASK: u32 = 0x0004_0000;
/// Virtual interrupt flag.
pub const VIF_MASK: u32 = 0x0008_0000;
/// Virtual interrupt pending.
pub const VIP_MASK: u32 = 0x0010_0000;
/// CPUID is available.
pub const ID_MASK: u32 = 0x0020_0000;

// The hflags bits not in `state`.

/// `HF_INHIBIT_IRQ_MASK`: the instruction after STI or MOV SS.
pub const HF_INHIBIT_IRQ_MASK: u32 = 1 << 3;
/// `HF_TF_MASK`, the same bit as TF.
pub const HF_TF_MASK: u32 = 1 << 8;
/// `HF_IOPL_MASK`, the same bits as IOPL.
pub const HF_IOPL_MASK: u32 = 3 << 12;
/// `HF_RF_MASK`, the same bit as RF.
pub const HF_RF_MASK: u32 = 1 << 16;
/// `HF_VM_MASK`, the same bit as VM.
pub const HF_VM_MASK: u32 = 1 << 17;
/// `HF_AC_MASK`, the same bit as AC.
pub const HF_AC_MASK: u32 = 1 << 18;
/// `HF_OSFXSR_MASK`.
pub const HF_OSFXSR_MASK: u32 = 1 << 22;
/// `HF_SMAP_MASK`.
pub const HF_SMAP_MASK: u32 = 1 << 23;
/// `HF_UMIP_MASK`.
pub const HF_UMIP_MASK: u32 = 1 << 27;
/// `HF_AVX_EN_MASK`: CR4.OSXSAVE is set and XCR0 enables SSE and AVX state.
pub const HF_AVX_EN_MASK: u32 = 1 << 28;
/// `HF2_HIF_MASK`.
pub const HF2_HIF_MASK: u32 = 1 << 1;

/// Read a little endian `u64`.
pub fn ld64(env: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(env[off..off + 8].try_into().expect("8 bytes"))
}

/// Write a little endian `u64`.
pub fn st64(env: &mut [u8], off: usize, v: u64) {
    env[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

/// Read a little endian `u32`.
pub fn ld32(env: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(env[off..off + 4].try_into().expect("4 bytes"))
}

/// Write a little endian `u32`.
pub fn st32(env: &mut [u8], off: usize, v: u32) {
    env[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

/// Read a segment cache at `off`.
pub fn ld_seg(env: &[u8], off: usize) -> SegmentCache {
    SegmentCache {
        selector: ld32(env, off + SEG_SELECTOR),
        base: ld64(env, off + SEG_BASE),
        limit: ld32(env, off + SEG_LIMIT),
        flags: ld32(env, off + SEG_FLAGS),
    }
}

/// Write a segment cache at `off`.
pub fn st_seg(env: &mut [u8], off: usize, s: &SegmentCache) {
    st32(env, off + SEG_SELECTOR, s.selector);
    st32(env, off + SEG_SELECTOR + 4, 0);
    st64(env, off + SEG_BASE, s.base);
    st32(env, off + SEG_LIMIT, s.limit);
    st32(env, off + SEG_FLAGS, s.flags);
}

/// Copy `s` into the buffer. The arithmetic flags go to `cc_src` with `cc_op` set to
/// `CC_OP_EFLAGS`, as `cpu_load_eflags()` leaves them.
pub fn load_state(env: &mut [u8], s: &X86CpuState) {
    for r in 0..CPU_NB_REGS {
        st64(env, reg(r), s.regs[r]);
    }
    st64(env, EIP, s.rip);
    let fl = s.rflags as u32;
    st64(env, CC_SRC, u64::from(fl & CC_MASK));
    st64(env, CC_DST, 0);
    st64(env, CC_SRC2, 0);
    st32(env, CC_OP, super::cc::CC_OP_EFLAGS);
    st32(env, DF, if fl & DF_MASK != 0 { -1i32 as u32 } else { 1 });
    st64(env, EFLAGS, u64::from(fl & !(CC_MASK | DF_MASK)) | 2);
    st32(env, HFLAGS, s.hflags);
    st32(env, HFLAGS2, s.hflags2);
    for i in 0..6 {
        st_seg(env, seg(i), &s.segs[i]);
    }
    st_seg(env, LDT, &s.ldt);
    st_seg(env, TR, &s.tr);
    st_seg(env, GDT, &s.gdt);
    st_seg(env, IDT, &s.idt);
    st64(env, cr(0), s.cr0);
    st64(env, cr(2), s.cr2);
    st64(env, cr(3), s.cr3);
    st64(env, cr(4), s.cr4);
    st64(env, CR8, s.cr8);
    st64(env, EFER, s.efer);
    st64(env, STAR, s.star);
    st64(env, LSTAR, s.lstar);
    st64(env, CSTAR, s.cstar);
    st64(env, FMASK, s.fmask);
    st64(env, KERNELGSBASE, s.kernelgsbase);
    st64(env, SYSENTER_CS, u64::from(s.sysenter_cs));
    st64(env, SYSENTER_ESP, s.sysenter_esp);
    st64(env, SYSENTER_EIP, s.sysenter_eip);
    for i in 0..8 {
        st64(env, dr(i), s.dr[i]);
    }
    st64(env, A20_MASK, s.a20_mask as i64 as u64);
    st32(env, ERROR_CODE, s.error_code);
    st32(env, EXCEPTION_IS_INT, 0);
    st64(env, EXCEPTION_NEXT_EIP, 0);
    st32(env, OLD_EXCEPTION, s.old_exception as u32);
    st64(env, TSC_OFFSET, 0);
    st64(env, PAT, s.pat);
    st64(env, APIC_BASE, s.apic_base);
    st64(env, TSC_AUX, s.tsc_aux);
    st64(env, MISC_ENABLE, s.msr_ia32_misc_enable);
    st64(env, XCR0, s.xcr0);
    st32(env, FPSTT, s.fpstt & 7);
    st32(env, FPUS, u32::from(s.fpus));
    st32(env, FPUC, u32::from(s.fpuc));
    st32(env, FPOP, u32::from(s.fpop));
    env[FPTAGS..FPTAGS + 8].copy_from_slice(&s.fptags);
    st64(env, FPIP, s.fpip);
    st64(env, FPDP, s.fpdp);
    st32(env, FPCS, u32::from(s.fpcs));
    st32(env, FPDS, u32::from(s.fpds));
    for i in 0..8 {
        st64(env, fpreg(i), s.fpregs[i][0]);
        st64(env, fpreg(i) + 8, s.fpregs[i][1] & 0xffff);
    }
    st32(env, MXCSR, s.mxcsr);
    st32(env, PKRU, s.pkru);
    for i in 0..32 {
        for j in 0..8 {
            st64(env, zmm(i) + 8 * j, s.xmm_regs[i][j]);
        }
    }
    // HF_AVX_EN_MASK is derived state, recomputed as cpu_sync_avx_hflag() does.
    let mut hf = s.hflags & !HF_AVX_EN_MASK;
    if avx_enabled(s.cr4, s.xcr0) {
        hf |= HF_AVX_EN_MASK;
    }
    st32(env, HFLAGS, hf);
}

/// The condition of `cpu_sync_avx_hflag()`.
pub fn avx_enabled(cr4: u64, xcr0: u64) -> bool {
    // CR4.OSXSAVE, and XCR0 enables both SSE and YMM state.
    cr4 & (1 << 18) != 0 && xcr0 & 6 == 6
}

/// Copy the buffer back into `s`, folding the lazy flags into `rflags` as
/// `cpu_compute_eflags()` does.
pub fn save_state(env: &[u8], s: &mut X86CpuState) {
    for r in 0..CPU_NB_REGS {
        s.regs[r] = ld64(env, reg(r));
    }
    s.rip = ld64(env, EIP);
    s.rflags = u64::from(compute_eflags(env));
    s.hflags = ld32(env, HFLAGS);
    s.hflags2 = ld32(env, HFLAGS2);
    for i in 0..6 {
        s.segs[i] = ld_seg(env, seg(i));
    }
    s.ldt = ld_seg(env, LDT);
    s.tr = ld_seg(env, TR);
    s.gdt = ld_seg(env, GDT);
    s.idt = ld_seg(env, IDT);
    s.cr0 = ld64(env, cr(0));
    s.cr2 = ld64(env, cr(2));
    s.cr3 = ld64(env, cr(3));
    s.cr4 = ld64(env, cr(4));
    s.cr8 = ld64(env, CR8);
    s.efer = ld64(env, EFER);
    s.star = ld64(env, STAR);
    s.lstar = ld64(env, LSTAR);
    s.cstar = ld64(env, CSTAR);
    s.fmask = ld64(env, FMASK);
    s.kernelgsbase = ld64(env, KERNELGSBASE);
    s.sysenter_cs = ld64(env, SYSENTER_CS) as u32;
    s.sysenter_esp = ld64(env, SYSENTER_ESP);
    s.sysenter_eip = ld64(env, SYSENTER_EIP);
    for i in 0..8 {
        s.dr[i] = ld64(env, dr(i));
    }
    s.a20_mask = ld64(env, A20_MASK) as i32;
    s.error_code = ld32(env, ERROR_CODE);
    s.old_exception = ld32(env, OLD_EXCEPTION) as i32;
    s.pat = ld64(env, PAT);
    s.apic_base = ld64(env, APIC_BASE);
    s.tsc_aux = ld64(env, TSC_AUX);
    s.msr_ia32_misc_enable = ld64(env, MISC_ENABLE);
    s.xcr0 = ld64(env, XCR0);
    s.fpstt = ld32(env, FPSTT) & 7;
    s.fpus = ld32(env, FPUS) as u16;
    s.fpuc = ld32(env, FPUC) as u16;
    s.fpop = ld32(env, FPOP) as u16;
    s.fptags.copy_from_slice(&env[FPTAGS..FPTAGS + 8]);
    s.fpip = ld64(env, FPIP);
    s.fpdp = ld64(env, FPDP);
    s.fpcs = ld32(env, FPCS) as u16;
    s.fpds = ld32(env, FPDS) as u16;
    for i in 0..8 {
        s.fpregs[i] = [ld64(env, fpreg(i)), ld64(env, fpreg(i) + 8) & 0xffff];
    }
    s.mxcsr = ld32(env, MXCSR);
    s.pkru = ld32(env, PKRU);
    for i in 0..32 {
        for j in 0..8 {
            s.xmm_regs[i][j] = ld64(env, zmm(i) + 8 * j);
        }
    }
}

/// `cpu_cc_compute_all()`: the arithmetic flags.
pub fn cc_compute_all(env: &[u8]) -> u32 {
    super::cc::compute_all(
        ld64(env, CC_DST),
        ld64(env, CC_SRC),
        ld64(env, CC_SRC2),
        ld32(env, CC_OP),
    )
}

/// `cpu_compute_eflags()`.
pub fn compute_eflags(env: &[u8]) -> u32 {
    let df = ld32(env, DF) as i32;
    let mut fl = ld64(env, EFLAGS) as u32 | cc_compute_all(env);
    if df < 0 {
        fl |= DF_MASK;
    }
    fl
}

/// `cpu_load_eflags()`: set EFLAGS to `eflags`, changing only the bits in `update_mask`.
pub fn load_eflags(env: &mut [u8], eflags: u32, update_mask: u32) {
    st64(env, CC_SRC, u64::from(eflags & CC_MASK));
    st32(env, CC_OP, super::cc::CC_OP_EFLAGS);
    st32(env, DF, if eflags & DF_MASK != 0 { -1i32 as u32 } else { 1 });
    let old = ld64(env, EFLAGS) as u32;
    let new = (old & !update_mask) | (eflags & update_mask) | 0x2;
    st64(env, EFLAGS, u64::from(new & !(CC_MASK | DF_MASK)));
}
