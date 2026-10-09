// SPDX-License-Identifier: GPL-2.0-or-later

//! The `CONFIG_USER_ONLY` parts of the x86 front end: `target/i386/tcg/user/seg_helper.c` and
//! `excp_helper.c`.
//!
//! Under user mode emulation the guest kernel is the emulator. SYSCALL and every exception
//! leave the vCPU with `exception_index` set, and the emulator's cpu loop turns them into a
//! system call or a signal. An access to a page the guest has not mapped becomes a page fault
//! the cpu loop reports as SIGSEGV.

use ruvm_jit::{Cpu, CpuLoopExit, MmuAccessType, Ra};

use super::env::{
    EIP, ERROR_CODE, EXCEPTION_IS_INT, EXCEPTION_NEXT_EIP, HFLAGS, IDT, OLD_EXCEPTION, SEG_BASE,
    cr, ld32, ld64, st32, st64,
};
use super::mmu::{PG_ERROR_I_D_MASK, PG_ERROR_P_MASK, PG_ERROR_U_MASK, PG_ERROR_W_MASK};
use super::{EXCP_SYSCALL, EXCP0D_GPF, EXCP0E_PAGE};
use crate::state::{DESC_DPL_SHIFT, HF_CPL_MASK, HF_LMA_MASK};

/// `helper_syscall()` for user mode: hand the system call to the cpu loop, with EIP to move
/// past the instruction.
pub(crate) fn helper_syscall_user(cpu: &mut Cpu<'_>, next_eip_addend: u64) -> CpuLoopExit {
    cpu.core.exception_index = EXCP_SYSCALL;
    st32(cpu.env, EXCEPTION_IS_INT, 0);
    let next = ld64(cpu.env, EIP).wrapping_add(next_eip_addend);
    st64(cpu.env, EXCEPTION_NEXT_EIP, next);
    cpu.cpu_loop_exit()
}

/// `x86_cpu_do_interrupt()` for user mode, `do_interrupt_user()`: what the CPU still does with
/// an exception before the cpu loop sees it. A software interrupt through a gate the program
/// may not use becomes #GP, and INT n and SYSCALL step over their instruction.
pub(crate) fn do_interrupt_user(cpu: &mut Cpu<'_>) {
    let mut intno = cpu.core.exception_index;
    let mut is_int = ld32(cpu.env, EXCEPTION_IS_INT) != 0;
    if is_int {
        let shift = if ld32(cpu.env, HFLAGS) & HF_LMA_MASK != 0 { 4 } else { 3 };
        let ptr = ld64(cpu.env, IDT + SEG_BASE).wrapping_add((intno as u64) << shift);
        let e2 = super::seg::ldl_kernel(cpu, ptr + 4, Ra::None).unwrap_or(0);
        let dpl = (e2 >> DESC_DPL_SHIFT) & 3;
        let cpl = ld32(cpu.env, HFLAGS) & HF_CPL_MASK;
        if dpl < cpl {
            // raise_exception_err() goes around the exception loop once more with #GP.
            st32(cpu.env, ERROR_CODE, ((intno as u32) << shift) + 2);
            st32(cpu.env, EXCEPTION_IS_INT, 0);
            st64(cpu.env, EXCEPTION_NEXT_EIP, ld64(cpu.env, EIP));
            intno = EXCP0D_GPF;
            is_int = false;
            cpu.core.exception_index = intno;
        }
    }
    if is_int || intno == EXCP_SYSCALL {
        st64(cpu.env, EIP, ld64(cpu.env, EXCEPTION_NEXT_EIP));
    }
    st32(cpu.env, OLD_EXCEPTION, u32::MAX);
}

/// `x86_cpu_record_sigsegv()`: an access the guest's mappings do not allow becomes a page
/// fault at `addr`, with the error code a kernel would see. `maperr` is set when nothing is
/// mapped there at all.
pub fn record_sigsegv(
    cpu: &mut Cpu<'_>,
    addr: u64,
    access_type: MmuAccessType,
    maperr: bool,
    ra: Ra,
) -> CpuLoopExit {
    st64(cpu.env, cr(2), addr);
    let mut code = if maperr { 0 } else { PG_ERROR_P_MASK };
    if access_type == MmuAccessType::DataStore {
        code |= PG_ERROR_W_MASK;
    }
    code |= PG_ERROR_U_MASK;
    if access_type == MmuAccessType::InstFetch {
        code |= PG_ERROR_I_D_MASK;
    }
    st32(cpu.env, ERROR_CODE, code);
    cpu.core.exception_index = EXCP0E_PAGE;
    st32(cpu.env, EXCEPTION_IS_INT, 0);
    st64(cpu.env, EXCEPTION_NEXT_EIP, u64::MAX);
    cpu.cpu_loop_exit_restore(ra)
}
