// SPDX-License-Identifier: GPL-2.0-or-later

//! Segmentation, privilege changes, exceptions and interrupts: the port of
//! `target/i386/tcg/seg_helper.c`, `excp_helper.c` and the system parts of them.
//!
//! The functions work on a [`Cpu`] and return `Err` with the [`CpuLoopExit`] of a guest
//! exception, which the helper wrappers in `helpers.rs` hand back to the runtime.

use ruvm_jit::cputlb::{cpu_ld_mmu, cpu_st_mmu};
use ruvm_jit::{Cpu, CpuLoopExit, Ra, excp, interrupt};
use ruvm_jit_core::{MemOp, MemOpIdx};

use super::env::{
    AC_MASK, CC_OP, CC_SRC, CC_Z, CSTAR, EFER, EFLAGS, EIP, ERROR_CODE, EXCEPTION_IS_INT,
    EXCEPTION_NEXT_EIP, FMASK, GDT, HFLAGS, HFLAGS2, ID_MASK, IDT, IF_MASK, IOPL_MASK, IOPL_SHIFT,
    LDT, LSTAR, NT_MASK, OLD_EXCEPTION, RF_MASK, SEG_BASE, SEG_FLAGS, SEG_SELECTOR, STAR,
    SYSENTER_CS, SYSENTER_EIP, SYSENTER_ESP, TF_MASK, TR, VM_MASK, cr, ld_seg, ld32, ld64, reg,
    seg, st_seg, st32, st64,
};
use super::{
    EXCP0A_TSS, EXCP0B_NOSEG, EXCP0C_STACK, EXCP0D_GPF, EXCP0E_PAGE, EXCP01_DB, EXCP03_INT3,
    EXCP04_INTO, EXCP06_ILLOP, EXCP08_DBLE, EXCP12_MCHK, X86, mmu_index_kernel, mmu_index_pl,
    x86_of,
};
use crate::state::{
    CR0_PE_MASK, DESC_A_MASK, DESC_B_MASK, DESC_B_SHIFT, DESC_C_MASK, DESC_CS_MASK, DESC_DPL_SHIFT,
    DESC_G_MASK, DESC_L_MASK, DESC_P_MASK, DESC_R_MASK, DESC_S_MASK, DESC_TYPE_SHIFT, DESC_W_MASK,
    HF_ADDSEG_MASK, HF_CPL_MASK, HF_CS32_MASK, HF_CS64_MASK, HF_LMA_MASK, HF_SS32_MASK,
    HF2_NMI_MASK, MSR_EFER_SCE, R_CS, R_DS, R_ECX, R_EDX, R_ES, R_ESP, R_FS, R_GS, R_SS,
    SegmentCache,
};

/// `DESC_TSS_BUSY_MASK`.
const DESC_TSS_BUSY_MASK: u32 = 1 << 9;

type R<T> = Result<T, CpuLoopExit>;

fn hflags(cpu: &Cpu<'_>) -> u32 {
    ld32(cpu.env, HFLAGS)
}

fn eflags(cpu: &Cpu<'_>) -> u32 {
    ld64(cpu.env, EFLAGS) as u32
}

fn set_eflags(cpu: &mut Cpu<'_>, v: u32) {
    st64(cpu.env, EFLAGS, u64::from(v));
}

fn cpl(cpu: &Cpu<'_>) -> u32 {
    hflags(cpu) & HF_CPL_MASK
}

fn regv(cpu: &Cpu<'_>, r: usize) -> u64 {
    ld64(cpu.env, reg(r))
}

fn set_reg(cpu: &mut Cpu<'_>, r: usize, v: u64) {
    st64(cpu.env, reg(r), v);
}

fn seg_of(cpu: &Cpu<'_>, s: usize) -> SegmentCache {
    ld_seg(cpu.env, seg(s))
}

// Guest memory accesses on behalf of helpers.

pub(crate) fn ld(cpu: &mut Cpu<'_>, addr: u64, mop: MemOp, idx: usize, ra: Ra) -> R<u64> {
    cpu_ld_mmu(cpu, addr, MemOpIdx::new(mop, idx as u32), ra)
}

pub(crate) fn st(cpu: &mut Cpu<'_>, addr: u64, v: u64, mop: MemOp, idx: usize, ra: Ra) -> R<()> {
    cpu_st_mmu(cpu, addr, v, MemOpIdx::new(mop, idx as u32), ra)
}

pub(crate) fn ldl_kernel(cpu: &mut Cpu<'_>, addr: u64, ra: Ra) -> R<u32> {
    let idx = mmu_index_kernel(cpu.env);
    ld(cpu, addr, MemOp::LEUL, idx, ra).map(|v| v as u32)
}

fn lduw_kernel(cpu: &mut Cpu<'_>, addr: u64, ra: Ra) -> R<u32> {
    let idx = mmu_index_kernel(cpu.env);
    ld(cpu, addr, MemOp::LEUW, idx, ra).map(|v| v as u32)
}

fn ldq_kernel(cpu: &mut Cpu<'_>, addr: u64, ra: Ra) -> R<u64> {
    let idx = mmu_index_kernel(cpu.env);
    ld(cpu, addr, MemOp::LEUQ, idx, ra)
}

fn stl_kernel(cpu: &mut Cpu<'_>, addr: u64, v: u32, ra: Ra) -> R<()> {
    let idx = mmu_index_kernel(cpu.env);
    st(cpu, addr, u64::from(v), MemOp::LEUL, idx, ra)
}

// Raising exceptions.

/// `check_exception()`: turn a fault during the delivery of another into a double fault, and
/// a fault during a double fault into a triple fault, which halts.
fn check_exception(cpu: &mut Cpu<'_>, intno: i32, error_code: &mut u32) -> i32 {
    let old = ld32(cpu.env, OLD_EXCEPTION) as i32;
    let first_contributory = old == 0 || (10..=13).contains(&old);
    let second_contributory = intno == 0 || (10..=13).contains(&intno);
    if old == EXCP08_DBLE {
        // "Triple fault": ask the platform for a system reset and halt until it comes.
        let ops = cpu.ops();
        x86_of(&ops).note_triple_fault();
        cpu.shared().halted.store(1, std::sync::atomic::Ordering::Release);
        return excp::HLT;
    }
    let mut intno = intno;
    if (first_contributory && second_contributory)
        || (old == EXCP0E_PAGE && (second_contributory || intno == EXCP0E_PAGE))
    {
        intno = EXCP08_DBLE;
        *error_code = 0;
    }
    if second_contributory || intno == EXCP0E_PAGE || intno == EXCP08_DBLE {
        st32(cpu.env, OLD_EXCEPTION, intno as u32);
    }
    intno
}

/// `raise_interrupt2()`: leave the block to deliver `intno`.
///
/// `is_int` is set for INT n, where `next_eip_addend` is the length of the instruction.
pub(crate) fn raise_interrupt2(
    cpu: &mut Cpu<'_>,
    intno: i32,
    is_int: bool,
    error_code: u32,
    next_eip_addend: u64,
    ra: Ra,
) -> CpuLoopExit {
    let mut error_code = error_code;
    let intno = if is_int { intno } else { check_exception(cpu, intno, &mut error_code) };
    cpu.core.exception_index = intno;
    st32(cpu.env, ERROR_CODE, error_code);
    st32(cpu.env, EXCEPTION_IS_INT, u32::from(is_int));
    let next = ld64(cpu.env, EIP).wrapping_add(next_eip_addend);
    st64(cpu.env, EXCEPTION_NEXT_EIP, next);
    cpu.cpu_loop_exit_restore(ra)
}

/// `raise_exception_err_ra()`.
pub(crate) fn raise_exception_err_ra(
    cpu: &mut Cpu<'_>,
    intno: i32,
    error_code: u32,
    ra: Ra,
) -> CpuLoopExit {
    raise_interrupt2(cpu, intno, false, error_code, 0, ra)
}

/// `raise_exception_ra()`.
pub(crate) fn raise_exception_ra(cpu: &mut Cpu<'_>, intno: i32, ra: Ra) -> CpuLoopExit {
    raise_interrupt2(cpu, intno, false, 0, 0, ra)
}

fn gpf<T>(cpu: &mut Cpu<'_>, code: u32, ra: Ra) -> R<T> {
    Err(raise_exception_err_ra(cpu, EXCP0D_GPF, code, ra))
}

fn excp_err<T>(cpu: &mut Cpu<'_>, intno: i32, code: u32, ra: Ra) -> R<T> {
    Err(raise_exception_err_ra(cpu, intno, code, ra))
}

// Descriptors.

/// `get_seg_base()`.
pub(crate) fn get_seg_base(e1: u32, e2: u32) -> u64 {
    u64::from((e1 >> 16) | ((e2 & 0xff) << 16) | (e2 & 0xff00_0000))
}

/// `get_seg_limit()`.
pub(crate) fn get_seg_limit(e1: u32, e2: u32) -> u32 {
    let limit = (e1 & 0xffff) | (e2 & 0x000f_0000);
    if e2 & DESC_G_MASK != 0 { (limit << 12) | 0xfff } else { limit }
}

/// `get_sp_mask()`.
fn get_sp_mask(e2: u32) -> u64 {
    if e2 & DESC_B_MASK != 0 { 0xffff_ffff } else { 0xffff }
}

/// `load_segment_ra()`: the two words of the descriptor `selector` names, or `None` past
/// the table limit.
fn load_segment(cpu: &mut Cpu<'_>, selector: u32, ra: Ra) -> R<Option<(u32, u32)>> {
    let dt = if selector & 4 != 0 { ld_seg(cpu.env, LDT) } else { ld_seg(cpu.env, GDT) };
    let index = u64::from(selector & !7);
    if index + 7 > u64::from(dt.limit) {
        return Ok(None);
    }
    let ptr = dt.base.wrapping_add(index);
    let e1 = ldl_kernel(cpu, ptr, ra)?;
    let e2 = ldl_kernel(cpu, ptr.wrapping_add(4), ra)?;
    Ok(Some((e1, e2)))
}

/// `cpu_x86_load_seg_cache()` on the state buffer, with the `hflags` update.
pub(crate) fn load_seg_cache(
    env: &mut [u8],
    seg_reg: usize,
    selector: u32,
    base: u64,
    limit: u32,
    flags: u32,
) {
    st_seg(env, seg(seg_reg), &SegmentCache { selector, base, limit, flags });
    let mut hf = ld32(env, HFLAGS);
    if seg_reg == R_CS {
        if hf & HF_LMA_MASK != 0 && flags & DESC_L_MASK != 0 {
            hf |= HF_CS32_MASK | HF_SS32_MASK | HF_CS64_MASK;
        } else {
            let cs32 = (flags & DESC_B_MASK) >> (DESC_B_SHIFT - 4);
            hf = (hf & !(HF_CS32_MASK | HF_CS64_MASK)) | cs32;
        }
    }
    if seg_reg == R_SS {
        let cpl = (flags >> DESC_DPL_SHIFT) & 3;
        hf = (hf & !HF_CPL_MASK) | cpl;
    }
    let ss_flags = ld32(env, seg(R_SS) + SEG_FLAGS);
    let mut new_hflags = (ss_flags & DESC_B_MASK) >> (DESC_B_SHIFT - 5);
    if hf & HF_CS64_MASK != 0 {
        // Zero base assumed for DS, ES and SS in long mode.
    } else if ld64(env, cr(0)) & CR0_PE_MASK == 0
        || ld64(env, EFLAGS) as u32 & VM_MASK != 0
        || hf & HF_CS32_MASK == 0
    {
        new_hflags |= HF_ADDSEG_MASK;
    } else {
        let any = ld64(env, seg(R_DS) + SEG_BASE)
            | ld64(env, seg(R_ES) + SEG_BASE)
            | ld64(env, seg(R_SS) + SEG_BASE);
        if any != 0 {
            new_hflags |= HF_ADDSEG_MASK;
        }
    }
    hf = (hf & !(HF_SS32_MASK | HF_ADDSEG_MASK)) | new_hflags;
    st32(env, HFLAGS, hf);
}

fn load_seg_raw(cpu: &mut Cpu<'_>, s: usize, selector: u32, e1: u32, e2: u32) {
    load_seg_cache(cpu.env, s, selector, get_seg_base(e1, e2), get_seg_limit(e1, e2), e2);
}

/// `helper_load_seg()`: MOV, POP or LxS to a segment register in protected mode.
pub(crate) fn helper_load_seg(cpu: &mut Cpu<'_>, seg_reg: usize, selector: u32, ra: Ra) -> R<()> {
    let selector = selector & 0xffff;
    let cpl = cpl(cpu);
    if selector & 0xfffc == 0 {
        // Null selector case.
        if seg_reg == R_SS && (hflags(cpu) & HF_CS64_MASK == 0 || cpl == 3) {
            return gpf(cpu, 0, ra);
        }
        load_seg_cache(cpu.env, seg_reg, selector, 0, 0, 0);
        return Ok(());
    }
    let dt = if selector & 4 != 0 { ld_seg(cpu.env, LDT) } else { ld_seg(cpu.env, GDT) };
    let index = u64::from(selector & !7);
    if index + 7 > u64::from(dt.limit) {
        return gpf(cpu, selector & 0xfffc, ra);
    }
    let ptr = dt.base.wrapping_add(index);
    let e1 = ldl_kernel(cpu, ptr, ra)?;
    let mut e2 = ldl_kernel(cpu, ptr.wrapping_add(4), ra)?;
    if e2 & DESC_S_MASK == 0 {
        return gpf(cpu, selector & 0xfffc, ra);
    }
    let rpl = selector & 3;
    let dpl = (e2 >> DESC_DPL_SHIFT) & 3;
    if seg_reg == R_SS {
        // Must be a writable data segment.
        if e2 & DESC_CS_MASK != 0 || e2 & DESC_W_MASK == 0 {
            return gpf(cpu, selector & 0xfffc, ra);
        }
        if rpl != cpl || dpl != cpl {
            return gpf(cpu, selector & 0xfffc, ra);
        }
    } else {
        // Must be a readable segment.
        if e2 & (DESC_CS_MASK | DESC_R_MASK) == DESC_CS_MASK {
            return gpf(cpu, selector & 0xfffc, ra);
        }
        if (e2 & DESC_CS_MASK == 0 || e2 & DESC_C_MASK == 0) && (dpl < cpl || dpl < rpl) {
            // Not conforming code: check the rights.
            return gpf(cpu, selector & 0xfffc, ra);
        }
    }
    if e2 & DESC_P_MASK == 0 {
        if seg_reg == R_SS {
            return excp_err(cpu, EXCP0C_STACK, selector & 0xfffc, ra);
        }
        return excp_err(cpu, EXCP0B_NOSEG, selector & 0xfffc, ra);
    }
    // Set the accessed bit if not already set.
    if e2 & DESC_A_MASK == 0 {
        e2 |= DESC_A_MASK;
        stl_kernel(cpu, ptr.wrapping_add(4), e2, ra)?;
    }
    load_seg_raw(cpu, seg_reg, selector, e1, e2);
    Ok(())
}

// Stack accesses for far transfers and interrupts.

struct StackAccess {
    sp: u64,
    sp_mask: u64,
    ss_base: u64,
    mmu_index: usize,
    ra: Ra,
}

impl StackAccess {
    fn addr(&self) -> u64 {
        self.ss_base.wrapping_add(self.sp & self.sp_mask)
    }

    fn push(&mut self, cpu: &mut Cpu<'_>, v: u64, mop: MemOp) -> R<()> {
        self.sp = self.sp.wrapping_sub(u64::from(mop.size_bytes()));
        let a = self.addr();
        st(cpu, a, v, mop, self.mmu_index, self.ra)
    }

    fn pop(&mut self, cpu: &mut Cpu<'_>, mop: MemOp) -> R<u64> {
        let a = self.addr();
        let v = ld(cpu, a, mop, self.mmu_index, self.ra)?;
        self.sp = self.sp.wrapping_add(u64::from(mop.size_bytes()));
        Ok(v)
    }

    fn pushw(&mut self, cpu: &mut Cpu<'_>, v: u64) -> R<()> {
        self.push(cpu, v, MemOp::LEUW)
    }
    fn pushl(&mut self, cpu: &mut Cpu<'_>, v: u64) -> R<()> {
        self.push(cpu, v, MemOp::LEUL)
    }
    fn pushq(&mut self, cpu: &mut Cpu<'_>, v: u64) -> R<()> {
        self.push(cpu, v, MemOp::LEUQ)
    }
    fn popw(&mut self, cpu: &mut Cpu<'_>) -> R<u64> {
        self.pop(cpu, MemOp::LEUW)
    }
    fn popl(&mut self, cpu: &mut Cpu<'_>) -> R<u64> {
        self.pop(cpu, MemOp::LEUL)
    }
    fn popq(&mut self, cpu: &mut Cpu<'_>) -> R<u64> {
        self.pop(cpu, MemOp::LEUQ)
    }
}

/// `SET_ESP()`.
fn set_esp(cpu: &mut Cpu<'_>, val: u64, sp_mask: u64) {
    let esp = regv(cpu, R_ESP);
    let v = match sp_mask {
        0xffff => (esp & !0xffff) | (val & 0xffff),
        0xffff_ffff => val & 0xffff_ffff,
        _ => val,
    };
    set_reg(cpu, R_ESP, v);
}

fn current_stack(cpu: &Cpu<'_>, ra: Ra) -> StackAccess {
    let ss = seg_of(cpu, R_SS);
    StackAccess {
        sp: regv(cpu, R_ESP),
        sp_mask: get_sp_mask(ss.flags),
        ss_base: ss.base,
        mmu_index: mmu_index_pl(cpu.env, cpl(cpu)),
        ra,
    }
}

/// `get_ss_esp_from_tss()`.
fn get_ss_esp_from_tss(cpu: &mut Cpu<'_>, dpl: u32, ra: Ra) -> R<(u32, u64)> {
    let tr = ld_seg(cpu.env, TR);
    let ty = (tr.flags >> DESC_TYPE_SHIFT) & 0xf;
    if tr.flags & DESC_P_MASK == 0 || ty & 7 != 1 {
        // QEMU aborts with "invalid tss" here.
        return excp_err(cpu, EXCP0A_TSS, tr.selector & 0xfffc, ra);
    }
    let shift = ty >> 3;
    let index = u64::from((dpl * 4 + 2) << shift);
    if index + (4 << shift) - 1 > u64::from(tr.limit) {
        return excp_err(cpu, EXCP0A_TSS, tr.selector & 0xfffc, ra);
    }
    let p = tr.base.wrapping_add(index);
    if shift == 0 {
        let esp = lduw_kernel(cpu, p, ra)?;
        let ss = lduw_kernel(cpu, p.wrapping_add(2), ra)?;
        Ok((ss, u64::from(esp)))
    } else {
        let esp = ldl_kernel(cpu, p, ra)?;
        let ss = lduw_kernel(cpu, p.wrapping_add(4), ra)?;
        Ok((ss, u64::from(esp)))
    }
}

/// `get_rsp_from_tss()`.
fn get_rsp_from_tss(cpu: &mut Cpu<'_>, level: u32) -> R<u64> {
    let tr = ld_seg(cpu.env, TR);
    if tr.flags & DESC_P_MASK == 0 {
        // QEMU aborts with "invalid tss" here.
        return excp_err(cpu, EXCP0A_TSS, tr.selector & 0xfffc, Ra::None);
    }
    let index = u64::from(8 * level + 4);
    if index + 7 > u64::from(tr.limit) {
        return excp_err(cpu, EXCP0A_TSS, tr.selector & 0xfffc, Ra::None);
    }
    ldq_kernel(cpu, tr.base.wrapping_add(index), Ra::None)
}

/// `exception_has_error_code()`.
fn exception_has_error_code(intno: i32) -> bool {
    matches!(intno, 8 | 10 | 11 | 12 | 13 | 14 | 17 | 21)
}

/// `exception_is_fault()`: whether RF is set in the pushed EFLAGS.
fn exception_is_fault(intno: i32) -> bool {
    // #DB can be both fault- and trap-like, but it never sets RF=1 in the RFLAGS value
    // pushed on the stack.
    !matches!(intno, EXCP01_DB | EXCP03_INT3 | EXCP04_INTO | EXCP08_DBLE | EXCP12_MCHK)
}

/// `do_interrupt_protected()`.
fn do_interrupt_protected(
    cpu: &mut Cpu<'_>,
    intno: i32,
    is_int: bool,
    error_code: u32,
    next_eip: u64,
    is_hw: bool,
) -> R<()> {
    let ra = Ra::None;
    let has_error_code = !is_int && !is_hw && exception_has_error_code(intno);
    let (old_eip, set_rf) =
        if is_int { (next_eip, false) } else { (ld64(cpu.env, EIP), exception_is_fault(intno)) };
    let vec = intno as u32;
    let dt = ld_seg(cpu.env, IDT);
    if u64::from(vec) * 8 + 7 > u64::from(dt.limit) {
        return gpf(cpu, vec * 8 + 2, ra);
    }
    let ptr = dt.base.wrapping_add(u64::from(vec) * 8);
    let e1 = ldl_kernel(cpu, ptr, ra)?;
    let e2 = ldl_kernel(cpu, ptr.wrapping_add(4), ra)?;
    let ty = (e2 >> DESC_TYPE_SHIFT) & 0x1f;
    match ty {
        // Task gates are not implemented.
        6 | 7 | 14 | 15 => {}
        _ => return gpf(cpu, vec * 8 + 2, ra),
    }
    let dpl = (e2 >> DESC_DPL_SHIFT) & 3;
    let cpl = cpl(cpu);
    // Check privilege if software int.
    if is_int && dpl < cpl {
        return gpf(cpu, vec * 8 + 2, ra);
    }
    // Check valid bit.
    if e2 & DESC_P_MASK == 0 {
        return excp_err(cpu, EXCP0B_NOSEG, vec * 8 + 2, ra);
    }
    let shift = ty >> 3;
    let selector = e1 >> 16;
    let offset = u64::from((e2 & 0xffff_0000) | (e1 & 0x0000_ffff));
    if selector & 0xfffc == 0 {
        return gpf(cpu, 0, ra);
    }
    let Some((e1, e2)) = load_segment(cpu, selector, ra)? else {
        return gpf(cpu, selector & 0xfffc, ra);
    };
    if e2 & DESC_S_MASK == 0 || e2 & DESC_CS_MASK == 0 {
        return gpf(cpu, selector & 0xfffc, ra);
    }
    let mut dpl = (e2 >> DESC_DPL_SHIFT) & 3;
    if dpl > cpl {
        return gpf(cpu, selector & 0xfffc, ra);
    }
    if e2 & DESC_P_MASK == 0 {
        return excp_err(cpu, EXCP0B_NOSEG, selector & 0xfffc, ra);
    }
    if e2 & DESC_C_MASK != 0 {
        dpl = cpl;
    }
    let vm86 = eflags(cpu) & VM_MASK != 0;
    let mut sa;
    let new_stack;
    let mut new_ss = (0, 0, 0);
    if dpl < cpl {
        // To inner privilege.
        let (ss, esp) = get_ss_esp_from_tss(cpu, dpl, ra)?;
        if ss & 0xfffc == 0 || ss & 3 != dpl {
            return excp_err(cpu, EXCP0A_TSS, ss & 0xfffc, ra);
        }
        let Some((ss_e1, ss_e2)) = load_segment(cpu, ss, ra)? else {
            return excp_err(cpu, EXCP0A_TSS, ss & 0xfffc, ra);
        };
        let ss_dpl = (ss_e2 >> DESC_DPL_SHIFT) & 3;
        if ss_dpl != dpl
            || ss_e2 & DESC_S_MASK == 0
            || ss_e2 & DESC_CS_MASK != 0
            || ss_e2 & DESC_W_MASK == 0
            || ss_e2 & DESC_P_MASK == 0
        {
            return excp_err(cpu, EXCP0A_TSS, ss & 0xfffc, ra);
        }
        new_stack = true;
        sa = StackAccess {
            sp: esp,
            sp_mask: get_sp_mask(ss_e2),
            ss_base: get_seg_base(ss_e1, ss_e2),
            mmu_index: mmu_index_pl(cpu.env, dpl),
            ra,
        };
        new_ss = (ss, ss_e1, ss_e2);
    } else {
        // To the same privilege.
        if vm86 {
            return gpf(cpu, selector & 0xfffc, ra);
        }
        new_stack = false;
        sa = current_stack(cpu, ra);
        sa.mmu_index = mmu_index_pl(cpu.env, dpl);
    }

    let mut fl = super::env::compute_eflags(cpu.env);
    if set_rf {
        fl |= RF_MASK;
    }
    let old_ss = u64::from(seg_of(cpu, R_SS).selector);
    let old_esp = regv(cpu, R_ESP);
    let old_cs = u64::from(seg_of(cpu, R_CS).selector);
    if shift == 1 {
        if new_stack {
            sa.pushl(cpu, old_ss)?;
            sa.pushl(cpu, old_esp)?;
        }
        sa.pushl(cpu, u64::from(fl))?;
        sa.pushl(cpu, old_cs)?;
        sa.pushl(cpu, old_eip)?;
        if has_error_code {
            sa.pushl(cpu, u64::from(error_code))?;
        }
    } else {
        if new_stack {
            sa.pushw(cpu, old_ss)?;
            sa.pushw(cpu, old_esp)?;
        }
        sa.pushw(cpu, u64::from(fl))?;
        sa.pushw(cpu, old_cs)?;
        sa.pushw(cpu, old_eip)?;
        if has_error_code {
            sa.pushw(cpu, u64::from(error_code))?;
        }
    }

    // An interrupt gate clears IF.
    let mut f = eflags(cpu);
    if ty & 1 == 0 {
        f &= !IF_MASK;
    }
    f &= !(TF_MASK | VM_MASK | RF_MASK | NT_MASK);
    set_eflags(cpu, f);

    if new_stack {
        let (ss, ss_e1, ss_e2) = new_ss;
        let ss = (ss & !3) | dpl;
        load_seg_cache(cpu.env, R_SS, ss, sa.ss_base, get_seg_limit(ss_e1, ss_e2), ss_e2);
    }
    set_esp(cpu, sa.sp, sa.sp_mask);

    let selector = (selector & !3) | dpl;
    load_seg_raw(cpu, R_CS, selector, e1, e2);
    st64(cpu.env, EIP, offset);
    Ok(())
}

/// `do_interrupt64()`.
fn do_interrupt64(
    cpu: &mut Cpu<'_>,
    intno: i32,
    is_int: bool,
    error_code: u32,
    next_eip: u64,
    is_hw: bool,
) -> R<()> {
    let ra = Ra::None;
    let has_error_code = !is_int && !is_hw && exception_has_error_code(intno);
    let (old_eip, set_rf) =
        if is_int { (next_eip, false) } else { (ld64(cpu.env, EIP), exception_is_fault(intno)) };
    let vec = intno as u32;
    let dt = ld_seg(cpu.env, IDT);
    if u64::from(vec) * 16 + 15 > u64::from(dt.limit) {
        return gpf(cpu, vec * 8 + 2, ra);
    }
    let ptr = dt.base.wrapping_add(u64::from(vec) * 16);
    let e1 = ldl_kernel(cpu, ptr, ra)?;
    let e2 = ldl_kernel(cpu, ptr.wrapping_add(4), ra)?;
    let e3 = ldl_kernel(cpu, ptr.wrapping_add(8), ra)?;
    let ty = (e2 >> DESC_TYPE_SHIFT) & 0x1f;
    match ty {
        14 | 15 => {}
        _ => return gpf(cpu, vec * 8 + 2, ra),
    }
    let dpl = (e2 >> DESC_DPL_SHIFT) & 3;
    let cpl = cpl(cpu);
    if is_int && dpl < cpl {
        return gpf(cpu, vec * 8 + 2, ra);
    }
    if e2 & DESC_P_MASK == 0 {
        return excp_err(cpu, EXCP0B_NOSEG, vec * 8 + 2, ra);
    }
    let selector = e1 >> 16;
    let offset = (u64::from(e3) << 32) | u64::from((e2 & 0xffff_0000) | (e1 & 0x0000_ffff));
    let ist = e2 & 7;
    if selector & 0xfffc == 0 {
        return gpf(cpu, 0, ra);
    }
    let Some((e1, e2)) = load_segment(cpu, selector, ra)? else {
        return gpf(cpu, selector & 0xfffc, ra);
    };
    if e2 & DESC_S_MASK == 0 || e2 & DESC_CS_MASK == 0 {
        return gpf(cpu, selector & 0xfffc, ra);
    }
    let mut dpl = (e2 >> DESC_DPL_SHIFT) & 3;
    if dpl > cpl {
        return gpf(cpu, selector & 0xfffc, ra);
    }
    if e2 & DESC_P_MASK == 0 {
        return excp_err(cpu, EXCP0B_NOSEG, selector & 0xfffc, ra);
    }
    if e2 & DESC_L_MASK == 0 || e2 & DESC_B_MASK != 0 {
        return gpf(cpu, selector & 0xfffc, ra);
    }
    if e2 & DESC_C_MASK != 0 {
        dpl = cpl;
    }
    let new_stack;
    let sp = if dpl < cpl || ist != 0 {
        // To inner privilege.
        new_stack = true;
        get_rsp_from_tss(cpu, if ist != 0 { ist + 3 } else { dpl })?
    } else {
        // To the same privilege.
        if eflags(cpu) & VM_MASK != 0 {
            return gpf(cpu, selector & 0xfffc, ra);
        }
        new_stack = false;
        regv(cpu, R_ESP)
    };
    let mut sa = StackAccess {
        // Align the stack.
        sp: sp & !0xf,
        sp_mask: u64::MAX,
        ss_base: 0,
        mmu_index: mmu_index_pl(cpu.env, dpl),
        ra,
    };

    let mut fl = super::env::compute_eflags(cpu.env);
    if set_rf {
        fl |= RF_MASK;
    }
    let old_ss = u64::from(seg_of(cpu, R_SS).selector);
    let old_esp = regv(cpu, R_ESP);
    let old_cs = u64::from(seg_of(cpu, R_CS).selector);
    sa.pushq(cpu, old_ss)?;
    sa.pushq(cpu, old_esp)?;
    sa.pushq(cpu, u64::from(fl))?;
    sa.pushq(cpu, old_cs)?;
    sa.pushq(cpu, old_eip)?;
    if has_error_code {
        sa.pushq(cpu, u64::from(error_code))?;
    }

    let mut f = eflags(cpu);
    if ty & 1 == 0 {
        f &= !IF_MASK;
    }
    f &= !(TF_MASK | VM_MASK | RF_MASK | NT_MASK);
    set_eflags(cpu, f);

    if new_stack {
        // SS is the null selector with RPL = the new CPL.
        load_seg_cache(cpu.env, R_SS, dpl, 0, 0, dpl << DESC_DPL_SHIFT);
    }
    set_reg(cpu, R_ESP, sa.sp);

    let selector = (selector & !3) | dpl;
    load_seg_raw(cpu, R_CS, selector, e1, e2);
    st64(cpu.env, EIP, offset);
    Ok(())
}

/// `do_interrupt_real()`.
fn do_interrupt_real(cpu: &mut Cpu<'_>, intno: i32, is_int: bool, next_eip: u64) -> R<()> {
    let ra = Ra::None;
    let vec = intno as u32;
    let dt = ld_seg(cpu.env, IDT);
    if u64::from(vec) * 4 + 3 > u64::from(dt.limit) {
        return gpf(cpu, vec * 8 + 2, ra);
    }
    let ptr = dt.base.wrapping_add(u64::from(vec) * 4);
    let offset = lduw_kernel(cpu, ptr, ra)?;
    let selector = lduw_kernel(cpu, ptr.wrapping_add(2), ra)?;
    let ss = seg_of(cpu, R_SS);
    let mut sa = StackAccess {
        sp: regv(cpu, R_ESP),
        sp_mask: 0xffff,
        ss_base: ss.base,
        mmu_index: mmu_index_pl(cpu.env, 0),
        ra,
    };
    let old_eip = if is_int { next_eip } else { ld64(cpu.env, EIP) };
    let old_cs = u64::from(seg_of(cpu, R_CS).selector);
    let fl = super::env::compute_eflags(cpu.env);
    sa.pushw(cpu, u64::from(fl))?;
    sa.pushw(cpu, old_cs)?;
    sa.pushw(cpu, old_eip)?;

    // Update the processor state.
    let esp = regv(cpu, R_ESP);
    set_reg(cpu, R_ESP, (esp & !sa.sp_mask) | (sa.sp & sa.sp_mask));
    st64(cpu.env, EIP, u64::from(offset));
    st32(cpu.env, seg(R_CS) + SEG_SELECTOR, selector);
    st64(cpu.env, seg(R_CS) + SEG_BASE, u64::from(selector) << 4);
    let f = eflags(cpu) & !(IF_MASK | TF_MASK | AC_MASK | RF_MASK);
    set_eflags(cpu, f);
    Ok(())
}

/// `do_interrupt_all()`.
fn do_interrupt_all(
    cpu: &mut Cpu<'_>,
    intno: i32,
    is_int: bool,
    error_code: u32,
    next_eip: u64,
    is_hw: bool,
) -> R<()> {
    if ld64(cpu.env, cr(0)) & CR0_PE_MASK != 0 {
        if hflags(cpu) & HF_LMA_MASK != 0 {
            do_interrupt64(cpu, intno, is_int, error_code, next_eip, is_hw)
        } else {
            do_interrupt_protected(cpu, intno, is_int, error_code, next_eip, is_hw)
        }
    } else {
        do_interrupt_real(cpu, intno, is_int, next_eip)
    }
}

/// Deliver the exception in `exception_index`, and any fault raised while delivering it,
/// as QEMU does by going around `cpu_exec()`'s exception loop again.
fn deliver_pending(cpu: &mut Cpu<'_>) {
    loop {
        if cpu.core.exception_index >= excp::INTERRUPT {
            // A triple fault: check_exception() already halted the vCPU.
            cpu.shared().cpu_interrupt(interrupt::HALT);
            return;
        }
        let intno = cpu.core.exception_index;
        let is_int = ld32(cpu.env, EXCEPTION_IS_INT) != 0;
        let error_code = ld32(cpu.env, ERROR_CODE);
        let next_eip = ld64(cpu.env, EXCEPTION_NEXT_EIP);
        if do_interrupt_all(cpu, intno, is_int, error_code, next_eip, false).is_ok() {
            // Successfully delivered.
            st32(cpu.env, OLD_EXCEPTION, u32::MAX);
            return;
        }
    }
}

/// `x86_cpu_do_interrupt()`.
pub(crate) fn x86_cpu_do_interrupt(cpu: &mut Cpu<'_>, _x: &X86) {
    deliver_pending(cpu);
}

/// `do_interrupt_x86_hardirq()`.
pub(crate) fn do_interrupt_x86_hardirq(cpu: &mut Cpu<'_>, _x: &X86, intno: i32, is_hw: bool) {
    if do_interrupt_all(cpu, intno, false, 0, 0, is_hw).is_err() {
        deliver_pending(cpu);
    }
    cpu.core.exception_index = -1;
}

/// The #DB that completes a HLT run with TF set, `do_interrupt_all(cpu, EXCP01_DB, 0, 0,
/// env->eip, 0)` in `x86_cpu_exec_halt()`.
pub(crate) fn hlt_single_step(cpu: &mut Cpu<'_>, _x: &X86) {
    let eip = ld64(cpu.env, EIP);
    if do_interrupt_all(cpu, EXCP01_DB, false, 0, eip, false).is_err() {
        deliver_pending(cpu);
    }
    cpu.core.exception_index = -1;
}

// Far transfers.

/// `helper_ljmp_protected()`.
pub(crate) fn helper_ljmp_protected(cpu: &mut Cpu<'_>, new_cs: u32, new_eip: u64, ra: Ra) -> R<()> {
    if new_cs & 0xfffc == 0 {
        return gpf(cpu, 0, ra);
    }
    let Some((e1, e2)) = load_segment(cpu, new_cs, ra)? else {
        return gpf(cpu, new_cs & 0xfffc, ra);
    };
    let cpl = cpl(cpu);
    if e2 & DESC_S_MASK == 0 {
        // Call gates, task gates and TSS descriptors are not implemented.
        return gpf(cpu, new_cs & 0xfffc, ra);
    }
    if e2 & DESC_CS_MASK == 0 {
        return gpf(cpu, new_cs & 0xfffc, ra);
    }
    let dpl = (e2 >> DESC_DPL_SHIFT) & 3;
    if e2 & DESC_C_MASK != 0 {
        // Conforming code segment.
        if dpl > cpl {
            return gpf(cpu, new_cs & 0xfffc, ra);
        }
    } else {
        // Non conforming code segment.
        let rpl = new_cs & 3;
        if rpl > cpl || dpl != cpl {
            return gpf(cpu, new_cs & 0xfffc, ra);
        }
    }
    if e2 & DESC_P_MASK == 0 {
        return excp_err(cpu, EXCP0B_NOSEG, new_cs & 0xfffc, ra);
    }
    let limit = get_seg_limit(e1, e2);
    if new_eip > u64::from(limit) && (hflags(cpu) & HF_LMA_MASK == 0 || e2 & DESC_L_MASK == 0) {
        return gpf(cpu, 0, ra);
    }
    load_seg_cache(cpu.env, R_CS, (new_cs & 0xfffc) | cpl, get_seg_base(e1, e2), limit, e2);
    st64(cpu.env, EIP, new_eip);
    Ok(())
}

/// `helper_lcall_real()`.
pub(crate) fn helper_lcall_real(
    cpu: &mut Cpu<'_>,
    new_cs: u32,
    new_eip: u64,
    shift: u32,
    next_eip: u64,
    ra: Ra,
) -> R<()> {
    let mut sa = current_stack(cpu, ra);
    let cs = u64::from(seg_of(cpu, R_CS).selector);
    if shift != 0 {
        sa.pushl(cpu, cs)?;
        sa.pushl(cpu, next_eip)?;
    } else {
        sa.pushw(cpu, cs)?;
        sa.pushw(cpu, next_eip)?;
    }
    set_esp(cpu, sa.sp, sa.sp_mask);
    st64(cpu.env, EIP, new_eip);
    st32(cpu.env, seg(R_CS) + SEG_SELECTOR, new_cs);
    st64(cpu.env, seg(R_CS) + SEG_BASE, u64::from(new_cs) << 4);
    Ok(())
}

/// `helper_lcall_protected()`.
pub(crate) fn helper_lcall_protected(
    cpu: &mut Cpu<'_>,
    new_cs: u32,
    new_eip: u64,
    shift: u32,
    next_eip_addend: u64,
    ra: Ra,
) -> R<()> {
    let next_eip = ld64(cpu.env, EIP).wrapping_add(next_eip_addend);
    if new_cs & 0xfffc == 0 {
        return gpf(cpu, 0, ra);
    }
    let Some((e1, e2)) = load_segment(cpu, new_cs, ra)? else {
        return gpf(cpu, new_cs & 0xfffc, ra);
    };
    let cpl = cpl(cpu);
    if e2 & DESC_S_MASK == 0 {
        // Call gates, task gates and TSS descriptors are not implemented.
        return gpf(cpu, new_cs & 0xfffc, ra);
    }
    if e2 & DESC_CS_MASK == 0 {
        return gpf(cpu, new_cs & 0xfffc, ra);
    }
    let dpl = (e2 >> DESC_DPL_SHIFT) & 3;
    if e2 & DESC_C_MASK != 0 {
        if dpl > cpl {
            return gpf(cpu, new_cs & 0xfffc, ra);
        }
    } else {
        let rpl = new_cs & 3;
        if rpl > cpl || dpl != cpl {
            return gpf(cpu, new_cs & 0xfffc, ra);
        }
    }
    if e2 & DESC_P_MASK == 0 {
        return excp_err(cpu, EXCP0B_NOSEG, new_cs & 0xfffc, ra);
    }
    let cs = u64::from(seg_of(cpu, R_CS).selector);
    if shift == 2 {
        // 64-bit case.
        let mut sa = StackAccess {
            sp: regv(cpu, R_ESP),
            sp_mask: u64::MAX,
            ss_base: 0,
            mmu_index: mmu_index_pl(cpu.env, cpl),
            ra,
        };
        sa.pushq(cpu, cs)?;
        sa.pushq(cpu, next_eip)?;
        // From this point, not restartable.
        set_reg(cpu, R_ESP, sa.sp);
        load_seg_raw(cpu, R_CS, (new_cs & 0xfffc) | cpl, e1, e2);
        st64(cpu.env, EIP, new_eip);
    } else {
        let mut sa = current_stack(cpu, ra);
        if shift != 0 {
            sa.pushl(cpu, cs)?;
            sa.pushl(cpu, next_eip)?;
        } else {
            sa.pushw(cpu, cs)?;
            sa.pushw(cpu, next_eip)?;
        }
        let limit = get_seg_limit(e1, e2);
        if new_eip > u64::from(limit) {
            return gpf(cpu, new_cs & 0xfffc, ra);
        }
        // From this point, not restartable.
        set_esp(cpu, sa.sp, sa.sp_mask);
        load_seg_cache(cpu.env, R_CS, (new_cs & 0xfffc) | cpl, get_seg_base(e1, e2), limit, e2);
        st64(cpu.env, EIP, new_eip);
    }
    Ok(())
}

/// `helper_iret_real()`.
pub(crate) fn helper_iret_real(cpu: &mut Cpu<'_>, shift: u32, ra: Ra) -> R<()> {
    let mut sa = current_stack(cpu, ra);
    // QEMU notes: "use SS segment size?"
    sa.sp_mask = 0xffff;
    let (new_eip, new_cs, new_eflags) = if shift == 1 {
        let e = sa.popl(cpu)?;
        let c = sa.popl(cpu)? & 0xffff;
        let f = sa.popl(cpu)?;
        (e, c, f)
    } else {
        let e = sa.popw(cpu)?;
        let c = sa.popw(cpu)?;
        let f = sa.popw(cpu)?;
        (e, c, f)
    };
    set_esp(cpu, sa.sp, sa.sp_mask);
    st32(cpu.env, seg(R_CS) + SEG_SELECTOR, new_cs as u32);
    st64(cpu.env, seg(R_CS) + SEG_BASE, new_cs << 4);
    st64(cpu.env, EIP, new_eip);
    let mut mask = if eflags(cpu) & VM_MASK != 0 {
        TF_MASK | AC_MASK | ID_MASK | IF_MASK | RF_MASK | NT_MASK
    } else {
        TF_MASK | AC_MASK | ID_MASK | IF_MASK | IOPL_MASK | RF_MASK | NT_MASK
    };
    if shift == 0 {
        mask &= 0xffff;
    }
    super::env::load_eflags(cpu.env, new_eflags as u32, mask);
    let hf2 = ld32(cpu.env, HFLAGS2) & !HF2_NMI_MASK;
    st32(cpu.env, HFLAGS2, hf2);
    Ok(())
}

/// `validate_seg()`: on a return to an outer level, null data segments the new level may not
/// use.
fn validate_seg(cpu: &mut Cpu<'_>, seg_reg: usize, cpl: u32) {
    let s = seg_of(cpu, seg_reg);
    // XXX in QEMU: on x86_64, FS and GS are not nullified because they may still contain a
    // valid base.
    if (seg_reg == R_FS || seg_reg == R_GS) && s.selector & 0xfffc == 0 {
        return;
    }
    let e2 = s.flags;
    let dpl = (e2 >> DESC_DPL_SHIFT) & 3;
    if (e2 & DESC_CS_MASK == 0 || e2 & DESC_C_MASK == 0) && dpl < cpl {
        // Data or non conforming code segment.
        load_seg_cache(cpu.env, seg_reg, 0, s.base, s.limit, s.flags & !DESC_P_MASK);
    }
}

/// `helper_ret_protected()`: far RET and IRET in protected and long mode.
fn helper_ret_protected(
    cpu: &mut Cpu<'_>,
    shift: u32,
    is_iret: bool,
    addend: u64,
    ra: Ra,
) -> R<()> {
    let mut sa = current_stack(cpu, ra);
    if shift == 2 {
        sa.sp_mask = u64::MAX;
    }
    let mut new_eflags = 0;
    let (new_eip, new_cs);
    if shift == 2 {
        new_eip = sa.popq(cpu)?;
        new_cs = (sa.popq(cpu)? & 0xffff) as u32;
        if is_iret {
            new_eflags = sa.popq(cpu)? as u32;
        }
    } else if shift == 1 {
        new_eip = sa.popl(cpu)?;
        new_cs = (sa.popl(cpu)? & 0xffff) as u32;
        if is_iret {
            new_eflags = sa.popl(cpu)? as u32;
            if new_eflags & VM_MASK != 0 {
                // Virtual 8086 mode is not supported.
                return gpf(cpu, 0, ra);
            }
        }
    } else {
        new_eip = sa.popw(cpu)?;
        new_cs = sa.popw(cpu)? as u32;
        if is_iret {
            new_eflags = sa.popw(cpu)? as u32;
        }
    }
    if new_cs & 0xfffc == 0 {
        return gpf(cpu, new_cs & 0xfffc, ra);
    }
    let Some((e1, e2)) = load_segment(cpu, new_cs, ra)? else {
        return gpf(cpu, new_cs & 0xfffc, ra);
    };
    if e2 & DESC_S_MASK == 0 || e2 & DESC_CS_MASK == 0 {
        return gpf(cpu, new_cs & 0xfffc, ra);
    }
    let cpl = cpl(cpu);
    let rpl = new_cs & 3;
    if rpl < cpl {
        return gpf(cpu, new_cs & 0xfffc, ra);
    }
    let dpl = (e2 >> DESC_DPL_SHIFT) & 3;
    if e2 & DESC_C_MASK != 0 {
        if dpl > rpl {
            return gpf(cpu, new_cs & 0xfffc, ra);
        }
    } else if dpl != rpl {
        return gpf(cpu, new_cs & 0xfffc, ra);
    }
    if e2 & DESC_P_MASK == 0 {
        return excp_err(cpu, EXCP0B_NOSEG, new_cs & 0xfffc, ra);
    }

    sa.sp = sa.sp.wrapping_add(addend);
    let cs64 = hflags(cpu) & HF_CS64_MASK != 0;
    if rpl == cpl && (!cs64 || !is_iret) {
        // Return to the same privilege level.
        load_seg_raw(cpu, R_CS, new_cs, e1, e2);
    } else {
        // Return to a different privilege level.
        let (new_esp, new_ss) = if shift == 2 {
            let e = sa.popq(cpu)?;
            (e, (sa.popq(cpu)? & 0xffff) as u32)
        } else if shift == 1 {
            let e = sa.popl(cpu)?;
            (e, (sa.popl(cpu)? & 0xffff) as u32)
        } else {
            let e = sa.popw(cpu)?;
            (e, sa.popw(cpu)? as u32)
        };
        let ss_e2;
        if new_ss & 0xfffc == 0 {
            // A null SS is allowed in long mode if the new CPL is not 3.
            if hflags(cpu) & HF_LMA_MASK != 0 && rpl != 3 {
                let fl = DESC_G_MASK
                    | DESC_B_MASK
                    | DESC_P_MASK
                    | DESC_S_MASK
                    | (rpl << DESC_DPL_SHIFT)
                    | DESC_W_MASK
                    | DESC_A_MASK;
                load_seg_cache(cpu.env, R_SS, new_ss, 0, 0xffff_ffff, fl);
                ss_e2 = DESC_B_MASK;
            } else {
                return gpf(cpu, 0, ra);
            }
        } else {
            if new_ss & 3 != rpl {
                return gpf(cpu, new_ss & 0xfffc, ra);
            }
            let Some((ss_e1, e2s)) = load_segment(cpu, new_ss, ra)? else {
                return gpf(cpu, new_ss & 0xfffc, ra);
            };
            if e2s & DESC_S_MASK == 0 || e2s & DESC_CS_MASK != 0 || e2s & DESC_W_MASK == 0 {
                return gpf(cpu, new_ss & 0xfffc, ra);
            }
            let dpl = (e2s >> DESC_DPL_SHIFT) & 3;
            if dpl != rpl {
                return gpf(cpu, new_ss & 0xfffc, ra);
            }
            if e2s & DESC_P_MASK == 0 {
                return excp_err(cpu, EXCP0B_NOSEG, new_ss & 0xfffc, ra);
            }
            load_seg_raw(cpu, R_SS, new_ss, ss_e1, e2s);
            ss_e2 = e2s;
        }
        load_seg_raw(cpu, R_CS, new_cs, e1, e2);
        sa.sp = new_esp;
        sa.sp_mask = if hflags(cpu) & HF_CS64_MASK != 0 { u64::MAX } else { get_sp_mask(ss_e2) };
        // Validate the data segments.
        validate_seg(cpu, R_ES, rpl);
        validate_seg(cpu, R_DS, rpl);
        validate_seg(cpu, R_FS, rpl);
        validate_seg(cpu, R_GS, rpl);
        sa.sp = sa.sp.wrapping_add(addend);
    }
    set_esp(cpu, sa.sp, sa.sp_mask);
    st64(cpu.env, EIP, new_eip);
    if is_iret {
        // `cpl` is the old CPL.
        let mut mask = TF_MASK | AC_MASK | ID_MASK | RF_MASK | NT_MASK;
        if cpl == 0 {
            mask |= IOPL_MASK;
        }
        let iopl = (eflags(cpu) >> IOPL_SHIFT) & 3;
        if cpl <= iopl {
            mask |= IF_MASK;
        }
        if shift == 0 {
            mask &= 0xffff;
        }
        super::env::load_eflags(cpu.env, new_eflags, mask);
    }
    Ok(())
}

/// `helper_iret_protected()`.
pub(crate) fn helper_iret_protected(cpu: &mut Cpu<'_>, shift: u32, ra: Ra) -> R<()> {
    // Task returns with NT set are not implemented.
    if eflags(cpu) & NT_MASK != 0 {
        return gpf(cpu, 0, ra);
    }
    helper_ret_protected(cpu, shift, true, 0, ra)?;
    let hf2 = ld32(cpu.env, HFLAGS2) & !HF2_NMI_MASK;
    st32(cpu.env, HFLAGS2, hf2);
    Ok(())
}

/// `helper_lret_protected()`.
pub(crate) fn helper_lret_protected(cpu: &mut Cpu<'_>, shift: u32, addend: u64, ra: Ra) -> R<()> {
    helper_ret_protected(cpu, shift, false, addend, ra)
}

// Fast system calls.

const FLAT_CODE: u32 =
    DESC_G_MASK | DESC_P_MASK | DESC_S_MASK | DESC_CS_MASK | DESC_R_MASK | DESC_A_MASK;
const FLAT_DATA: u32 =
    DESC_G_MASK | DESC_B_MASK | DESC_P_MASK | DESC_S_MASK | DESC_W_MASK | DESC_A_MASK;

/// `helper_syscall()`.
pub(crate) fn helper_syscall(cpu: &mut Cpu<'_>, next_eip_addend: u64, ra: Ra) -> R<()> {
    let ops = cpu.ops();
    if x86_of(&ops).is_user_mode() {
        return Err(super::user::helper_syscall_user(cpu, next_eip_addend));
    }
    if ld64(cpu.env, EFER) & MSR_EFER_SCE == 0 {
        return excp_err(cpu, EXCP06_ILLOP, 0, ra);
    }
    let selector = ((ld64(cpu.env, STAR) >> 32) & 0xffff) as u32;
    let next = ld64(cpu.env, EIP).wrapping_add(next_eip_addend);
    if hflags(cpu) & HF_LMA_MASK != 0 {
        set_reg(cpu, R_ECX, next);
        let fl = super::env::compute_eflags(cpu.env) & !RF_MASK;
        set_reg(cpu, 11, u64::from(fl));
        let code64 = hflags(cpu) & HF_CS64_MASK != 0;
        let fmask = ld64(cpu.env, FMASK) as u32;
        let f = eflags(cpu) & !(fmask | RF_MASK);
        set_eflags(cpu, f);
        super::env::load_eflags(cpu.env, f, 0);
        load_seg_cache(cpu.env, R_CS, selector & 0xfffc, 0, 0xffff_ffff, FLAT_CODE | DESC_L_MASK);
        load_seg_cache(cpu.env, R_SS, (selector + 8) & 0xfffc, 0, 0xffff_ffff, FLAT_DATA);
        let target = if code64 { ld64(cpu.env, LSTAR) } else { ld64(cpu.env, CSTAR) };
        st64(cpu.env, EIP, target);
    } else {
        set_reg(cpu, R_ECX, next & 0xffff_ffff);
        let f = eflags(cpu) & !(IF_MASK | RF_MASK | VM_MASK);
        set_eflags(cpu, f);
        load_seg_cache(cpu.env, R_CS, selector & 0xfffc, 0, 0xffff_ffff, FLAT_CODE | DESC_B_MASK);
        load_seg_cache(cpu.env, R_SS, (selector + 8) & 0xfffc, 0, 0xffff_ffff, FLAT_DATA);
        st64(cpu.env, EIP, ld64(cpu.env, STAR) & 0xffff_ffff);
    }
    Ok(())
}

/// `helper_sysret()`.
pub(crate) fn helper_sysret(cpu: &mut Cpu<'_>, dflag: u32, ra: Ra) -> R<()> {
    if ld64(cpu.env, EFER) & MSR_EFER_SCE == 0 {
        return excp_err(cpu, EXCP06_ILLOP, 0, ra);
    }
    let cpl = cpl(cpu);
    if ld64(cpu.env, cr(0)) & CR0_PE_MASK == 0 || cpl != 0 {
        return gpf(cpu, 0, ra);
    }
    let selector = ((ld64(cpu.env, STAR) >> 48) & 0xffff) as u32;
    let dpl3 = 3 << DESC_DPL_SHIFT;
    if hflags(cpu) & HF_LMA_MASK != 0 {
        if dflag == 2 {
            let new_rip = regv(cpu, R_ECX);
            let ops = cpu.ops();
            if x86_of(&ops).model().cpuid(0, 0)[1] == 0x756e_6547 {
                // An Intel CPU raises #GP for a non-canonical RCX.
                let shift = if super::mmu::la57(cpu.env) { 56 } else { 47 };
                let sext = (new_rip as i64) >> shift;
                if sext != 0 && sext != -1 {
                    return gpf(cpu, 0, ra);
                }
            }
            load_seg_cache(
                cpu.env,
                R_CS,
                (selector + 16) | 3,
                0,
                0xffff_ffff,
                FLAT_CODE | dpl3 | DESC_L_MASK,
            );
            st64(cpu.env, EIP, new_rip);
        } else {
            load_seg_cache(
                cpu.env,
                R_CS,
                selector | 3,
                0,
                0xffff_ffff,
                FLAT_CODE | DESC_B_MASK | dpl3,
            );
            st64(cpu.env, EIP, regv(cpu, R_ECX) & 0xffff_ffff);
        }
        load_seg_cache(cpu.env, R_SS, (selector + 8) | 3, 0, 0xffff_ffff, FLAT_DATA | dpl3);
        let r11 = regv(cpu, 11) as u32;
        super::env::load_eflags(
            cpu.env,
            r11,
            TF_MASK | AC_MASK | ID_MASK | IF_MASK | IOPL_MASK | VM_MASK | RF_MASK | NT_MASK,
        );
    } else {
        let f = eflags(cpu) | IF_MASK;
        set_eflags(cpu, f);
        load_seg_cache(cpu.env, R_CS, selector | 3, 0, 0xffff_ffff, FLAT_CODE | DESC_B_MASK | dpl3);
        st64(cpu.env, EIP, regv(cpu, R_ECX) & 0xffff_ffff);
        load_seg_cache(cpu.env, R_SS, (selector + 8) | 3, 0, 0xffff_ffff, FLAT_DATA | dpl3);
    }
    Ok(())
}

/// `helper_sysenter()`.
pub(crate) fn helper_sysenter(cpu: &mut Cpu<'_>, ra: Ra) -> R<()> {
    let cs = ld64(cpu.env, SYSENTER_CS) as u32;
    if cs == 0 {
        return gpf(cpu, 0, ra);
    }
    let f = eflags(cpu) & !(VM_MASK | IF_MASK | RF_MASK);
    set_eflags(cpu, f);
    let code = if hflags(cpu) & HF_LMA_MASK != 0 {
        FLAT_CODE | DESC_B_MASK | DESC_L_MASK
    } else {
        FLAT_CODE | DESC_B_MASK
    };
    load_seg_cache(cpu.env, R_CS, cs & 0xfffc, 0, 0xffff_ffff, code);
    load_seg_cache(cpu.env, R_SS, (cs + 8) & 0xfffc, 0, 0xffff_ffff, FLAT_DATA);
    let esp = ld64(cpu.env, SYSENTER_ESP);
    set_reg(cpu, R_ESP, esp);
    let eip = ld64(cpu.env, SYSENTER_EIP);
    st64(cpu.env, EIP, eip);
    Ok(())
}

/// `helper_sysexit()`.
pub(crate) fn helper_sysexit(cpu: &mut Cpu<'_>, dflag: u32, ra: Ra) -> R<()> {
    let cs = ld64(cpu.env, SYSENTER_CS) as u32;
    if cs == 0 || cpl(cpu) != 0 {
        return gpf(cpu, 0, ra);
    }
    let dpl3 = 3 << DESC_DPL_SHIFT;
    if dflag == 2 {
        load_seg_cache(
            cpu.env,
            R_CS,
            (cs + 32) | 3,
            0,
            0xffff_ffff,
            FLAT_CODE | DESC_B_MASK | DESC_L_MASK | dpl3,
        );
        load_seg_cache(cpu.env, R_SS, (cs + 40) | 3, 0, 0xffff_ffff, FLAT_DATA | dpl3);
    } else {
        load_seg_cache(
            cpu.env,
            R_CS,
            (cs + 16) | 3,
            0,
            0xffff_ffff,
            FLAT_CODE | DESC_B_MASK | dpl3,
        );
        load_seg_cache(cpu.env, R_SS, (cs + 24) | 3, 0, 0xffff_ffff, FLAT_DATA | dpl3);
    }
    let ecx = regv(cpu, R_ECX);
    set_reg(cpu, R_ESP, ecx);
    let edx = regv(cpu, R_EDX);
    st64(cpu.env, EIP, edx);
    Ok(())
}

// System descriptor registers.

fn load_system_descriptor(cpu: &mut Cpu<'_>, selector: u32, ra: Ra) -> R<(u64, u32, u32, u64)> {
    if selector & 4 != 0 {
        return gpf(cpu, selector & 0xfffc, ra);
    }
    let dt = ld_seg(cpu.env, GDT);
    let index = u64::from(selector & !7);
    let lma = hflags(cpu) & HF_LMA_MASK != 0;
    let entry_limit = if lma { 15 } else { 7 };
    if index + entry_limit > u64::from(dt.limit) {
        return gpf(cpu, selector & 0xfffc, ra);
    }
    let ptr = dt.base.wrapping_add(index);
    let e1 = ldl_kernel(cpu, ptr, ra)?;
    let e2 = ldl_kernel(cpu, ptr.wrapping_add(4), ra)?;
    Ok((ptr, e1, e2, entry_limit))
}

/// `helper_lldt()`.
pub(crate) fn helper_lldt(cpu: &mut Cpu<'_>, selector: u32, ra: Ra) -> R<()> {
    let selector = selector & 0xffff;
    let mut ldt = ld_seg(cpu.env, LDT);
    if selector & 0xfffc == 0 {
        // XXX in QEMU: the null selector case makes an invalid LDT.
        ldt.base = 0;
        ldt.limit = 0;
    } else {
        let (ptr, e1, e2, entry_limit) = load_system_descriptor(cpu, selector, ra)?;
        if e2 & DESC_S_MASK != 0 || (e2 >> DESC_TYPE_SHIFT) & 0xf != 2 {
            return gpf(cpu, selector & 0xfffc, ra);
        }
        if e2 & DESC_P_MASK == 0 {
            return excp_err(cpu, EXCP0B_NOSEG, selector & 0xfffc, ra);
        }
        ldt.base = get_seg_base(e1, e2);
        ldt.limit = get_seg_limit(e1, e2);
        ldt.flags = e2;
        if entry_limit == 15 {
            let e3 = ldl_kernel(cpu, ptr.wrapping_add(8), ra)?;
            ldt.base |= u64::from(e3) << 32;
        }
    }
    ldt.selector = selector;
    st_seg(cpu.env, LDT, &ldt);
    Ok(())
}

/// `helper_ltr()`.
pub(crate) fn helper_ltr(cpu: &mut Cpu<'_>, selector: u32, ra: Ra) -> R<()> {
    let selector = selector & 0xffff;
    let mut tr = ld_seg(cpu.env, TR);
    if selector & 0xfffc == 0 {
        // The null selector makes an invalid TR.
        tr.base = 0;
        tr.limit = 0;
        tr.flags = 0;
    } else {
        let (ptr, e1, mut e2, entry_limit) = load_system_descriptor(cpu, selector, ra)?;
        let ty = (e2 >> DESC_TYPE_SHIFT) & 0xf;
        if e2 & DESC_S_MASK != 0 || (ty != 1 && ty != 9) {
            return gpf(cpu, selector & 0xfffc, ra);
        }
        if e2 & DESC_P_MASK == 0 {
            return excp_err(cpu, EXCP0B_NOSEG, selector & 0xfffc, ra);
        }
        tr.base = get_seg_base(e1, e2);
        tr.limit = get_seg_limit(e1, e2);
        tr.flags = e2;
        if entry_limit == 15 {
            let e3 = ldl_kernel(cpu, ptr.wrapping_add(8), ra)?;
            let e4 = ldl_kernel(cpu, ptr.wrapping_add(12), ra)?;
            if (e4 >> DESC_TYPE_SHIFT) & 0xf != 0 {
                return gpf(cpu, selector & 0xfffc, ra);
            }
            tr.base |= u64::from(e3) << 32;
        }
        e2 |= DESC_TSS_BUSY_MASK;
        stl_kernel(cpu, ptr.wrapping_add(4), e2, ra)?;
    }
    tr.selector = selector;
    st_seg(cpu.env, TR, &tr);
    Ok(())
}

/// Common checks of LAR, LSL, VERR and VERW: the descriptor if the selector passes the
/// privilege checks for a segment, `None` (ZF clear) if not.
fn check_access(cpu: &mut Cpu<'_>, selector: u32, ra: Ra) -> R<Option<(u32, u32, u32, u32)>> {
    if selector & 0xfffc == 0 {
        return Ok(None);
    }
    let Some((e1, e2)) = load_segment(cpu, selector, ra)? else {
        return Ok(None);
    };
    let rpl = selector & 3;
    let dpl = (e2 >> DESC_DPL_SHIFT) & 3;
    Ok(Some((e1, e2, rpl, dpl)))
}

fn set_zf(cpu: &mut Cpu<'_>, eflags: u32, ok: bool) {
    let v = if ok { eflags | CC_Z } else { eflags & !CC_Z };
    st64(cpu.env, CC_SRC, u64::from(v));
    st32(cpu.env, CC_OP, super::cc::CC_OP_EFLAGS);
}

/// `helper_lsl()` and `helper_lar()`.
pub(crate) fn helper_lar_lsl(cpu: &mut Cpu<'_>, selector: u32, lsl: bool, ra: Ra) -> R<u64> {
    let selector = selector & 0xffff;
    let fl = super::env::cc_compute_all(cpu.env);
    let cpl = cpl(cpu);
    let Some((e1, e2, rpl, dpl)) = check_access(cpu, selector, ra)? else {
        set_zf(cpu, fl, false);
        return Ok(0);
    };
    let ok = if e2 & DESC_S_MASK != 0 {
        (e2 & DESC_CS_MASK != 0 && e2 & DESC_C_MASK != 0) || (dpl >= cpl && dpl >= rpl)
    } else {
        let ty = (e2 >> DESC_TYPE_SHIFT) & 0xf;
        let valid = if lsl {
            matches!(ty, 1 | 2 | 3 | 9 | 11)
        } else {
            matches!(ty, 1 | 2 | 3 | 4 | 5 | 9 | 11 | 12)
        };
        valid && dpl >= cpl && dpl >= rpl
    };
    set_zf(cpu, fl, ok);
    if !ok {
        return Ok(0);
    }
    Ok(if lsl { u64::from(get_seg_limit(e1, e2)) } else { u64::from(e2 & 0x00f0_ff00) })
}

/// `helper_verr()` and `helper_verw()`.
pub(crate) fn helper_verr_verw(cpu: &mut Cpu<'_>, selector: u32, write: bool, ra: Ra) -> R<()> {
    let selector = selector & 0xffff;
    let fl = super::env::cc_compute_all(cpu.env);
    let cpl = cpl(cpu);
    let Some((_e1, e2, rpl, dpl)) = check_access(cpu, selector, ra)? else {
        set_zf(cpu, fl, false);
        return Ok(());
    };
    let ok = if e2 & DESC_S_MASK == 0 {
        false
    } else if e2 & DESC_CS_MASK != 0 {
        if write {
            false
        } else {
            e2 & DESC_R_MASK != 0 && (e2 & DESC_C_MASK != 0 || (dpl >= cpl && dpl >= rpl))
        }
    } else {
        dpl >= cpl && dpl >= rpl && (!write || e2 & DESC_W_MASK != 0)
    };
    set_zf(cpu, fl, ok);
    Ok(())
}

/// `helper_check_io()`: the I/O permission bitmap check.
pub(crate) fn helper_check_io(cpu: &mut Cpu<'_>, addr: u32, size: u32, ra: Ra) -> R<()> {
    let tr = ld_seg(cpu.env, TR);
    let ok =
        tr.flags & DESC_P_MASK != 0 && (tr.flags >> DESC_TYPE_SHIFT) & 0xf == 9 && tr.limit >= 103;
    if ok {
        let mut io_offset = lduw_kernel(cpu, tr.base.wrapping_add(0x66), ra)?;
        io_offset += addr >> 3;
        // Note: the check needs two bytes.
        if io_offset < tr.limit {
            let val = lduw_kernel(cpu, tr.base.wrapping_add(u64::from(io_offset)), ra)?;
            let val = val >> (addr & 7);
            let mask = (1 << size) - 1;
            // All bits must be zero to allow the I/O.
            if val & mask == 0 {
                return Ok(());
            }
        }
    }
    gpf(cpu, 0, ra)
}
