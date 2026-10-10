// SPDX-License-Identifier: GPL-2.0-or-later

//! The execution loop, `accel/tcg/cpu-exec.c`: `cpu_exec()`, halt, exception, interrupt and
//! breakpoint handling, `cpu_exec_step_atomic()` and `helper_lookup_tb_ptr()`, with
//! `tcg_cpu_exec()` from `tcg-accel-ops.c`.
//!
//! Differences from QEMU:
//!
//! - `sigsetjmp` is a loop: a [`CpuLoopExit`] coming back from any step restarts
//!   `cpu_exec_loop()` from the exception check, as the longjmp does.
//! - There is no icount, no record and replay and no clock alignment. A `TB_EXIT_REQUESTED`
//!   exit without an exit request (QEMU's "instruction counter expired") just goes round the
//!   loop again.
//! - The `cpu_exec_interrupt` hook cannot leave the loop with `cpu_loop_exit()`; it returns
//!   whether it took an interrupt.
//! - x86's `CPU_INTERRUPT_INIT`, `CPU_INTERRUPT_POLL` and SMM handling are not here; the target
//!   handles them in `cpu_exec_interrupt`.
//! - A block that was translated after a `tb_flush` done by `tb_gen_code()` itself is not
//!   chained to the block that ran before, which no longer exists.

use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::backend::TbRet;
use crate::cpu::{Cpu, CpuLoopExit};
use crate::tb::{Tb, TbCpuState};
use crate::tb_maint::{jc_set, tb_lookup, tb_lookup_jump, tb_lookup_with};
use crate::translate::tb_gen_code;
use crate::{bp, cf, excp, interrupt};
use ruvm_jit_core::types::tb_exit;

/// `cpu_handle_halt()`: whether the CPU stays halted.
fn cpu_handle_halt(cpu: &mut Cpu<'_>) -> bool {
    let shared = cpu.shared();
    if shared.halted.load(Ordering::Acquire) != 0 {
        let ops = cpu.ops();
        let leave_halt = ops.cpu_exec_halt(cpu);
        if !leave_halt {
            return true;
        }
        shared.halted.store(0, Ordering::Release);
    }
    false
}

/// `cpu_handle_debug_exception()`.
fn cpu_handle_debug_exception(cpu: &mut Cpu<'_>) {
    if cpu.core.watchpoint_hit.is_none() {
        for wp in &mut cpu.core.watchpoints {
            wp.flags &= !bp::WATCHPOINT_HIT;
        }
    }
    let ops = cpu.ops();
    ops.debug_excp_handler(cpu);
}

/// `cpu_handle_exception()`: `Some(ret)` if `cpu_exec()` should return `ret`.
fn cpu_handle_exception(cpu: &mut Cpu<'_>) -> Option<i32> {
    if cpu.core.exception_index < 0 {
        return None;
    }
    if cpu.core.exception_index >= excp::INTERRUPT {
        // Exit request from the cpu execution loop.
        let ret = cpu.core.exception_index;
        if ret == excp::DEBUG {
            cpu_handle_debug_exception(cpu);
        }
        cpu.core.exception_index = -1;
        return Some(ret);
    }
    let ops = cpu.ops();
    if cpu.jit().config.user_only {
        // User mode hands every exception to the cpu loop of the emulator, which turns it into
        // a syscall or a signal.
        ops.fake_user_interrupt(cpu);
        let ret = cpu.core.exception_index;
        cpu.core.exception_index = -1;
        return Some(ret);
    }
    ops.do_interrupt(cpu);
    cpu.core.exception_index = -1;
    if cpu.core.singlestep_enabled {
        // After processing the exception, ensure an EXCP_DEBUG is raised when
        // single-stepping so that GDB doesn't miss the next instruction.
        cpu_handle_debug_exception(cpu);
        return Some(excp::DEBUG);
    }
    None
}

/// `cpu_handle_interrupt()`: whether the inner loop should stop.
fn cpu_handle_interrupt(cpu: &mut Cpu<'_>, last_tb: &mut Option<Arc<Tb>>) -> bool {
    // If we have requested custom cflags with CF_NOIRQ we should skip checking here. Any
    // pending interrupts will get picked up by the next TB we execute under normal cflags.
    if cpu.core.cflags_next_tb != u32::MAX && cpu.core.cflags_next_tb & cf::NOIRQ != 0 {
        return false;
    }
    let shared = cpu.shared();

    // Clear the interrupt flag now since we're processing interrupt_request and
    // exit_request.
    shared.icount_decr.fetch_and(0x0000_ffff, Ordering::SeqCst);

    if shared.interrupt_request() != 0 {
        let mut interrupt_request = shared.interrupt_request();
        if interrupt_request & interrupt::DEBUG != 0 {
            shared.reset_interrupt(interrupt::DEBUG);
            cpu.core.exception_index = excp::DEBUG;
            return true;
        }
        if interrupt_request & interrupt::HALT != 0 {
            shared.reset_interrupt(interrupt::HALT);
            shared.halted.store(1, Ordering::Release);
            cpu.core.exception_index = excp::HLT;
            return true;
        } else if interrupt_request & interrupt::RESET != 0 {
            let ops = cpu.ops();
            ops.cpu_exec_reset(cpu);
            return true;
        } else {
            // The target hook has 2 exit conditions: false when the interrupt isn't processed,
            // true when it is, and we should restart on a new TB.
            let ops = cpu.ops();
            if ops.cpu_exec_interrupt(cpu, interrupt_request) {
                // After processing the interrupt, ensure an EXCP_DEBUG is raised when
                // single-stepping so that GDB doesn't miss the next instruction.
                if cpu.core.singlestep_enabled {
                    cpu.core.exception_index = excp::DEBUG;
                    return true;
                }
                cpu.core.exception_index = -1;
                *last_tb = None;
            }
            // The target hook may have updated the interrupt_request value, so reload it
            // before checking for EXITTB below.
            interrupt_request = shared.interrupt_request();
        }
        if interrupt_request & interrupt::EXITTB != 0 {
            shared.reset_interrupt(interrupt::EXITTB);
            // Ensure that no TB jump will be modified as the program flow was changed.
            *last_tb = None;
        }
    }

    // Finally, check if we need to exit to the main loop.
    if shared.exit_request.load(Ordering::Acquire) {
        shared.exit_request.store(false, Ordering::Release);
        if cpu.core.exception_index == -1 {
            cpu.core.exception_index = excp::INTERRUPT;
        }
        return true;
    }
    false
}

/// `check_for_breakpoints()`: whether a breakpoint hit at `pc`. A breakpoint elsewhere on the
/// page makes the block a single instruction.
#[inline]
fn check_for_breakpoints(cpu: &mut Cpu<'_>, pc: u64, cflags: &mut u32) -> bool {
    !cpu.core.breakpoints.is_empty() && check_for_breakpoints_slow(cpu, pc, cflags)
}

/// [`check_for_breakpoints`] with at least one breakpoint set.
#[cold]
fn check_for_breakpoints_slow(cpu: &mut Cpu<'_>, pc: u64, cflags: &mut u32) -> bool {
    // Singlestep overrides breakpoints.
    if cpu.core.singlestep_enabled {
        return false;
    }
    let page_mask = cpu.core.jit.page_mask();
    let mut match_page = false;
    for i in 0..cpu.core.breakpoints.len() {
        let b = cpu.core.breakpoints[i];
        // If we have an exact pc match, trigger the breakpoint. Otherwise, note matches
        // within the page.
        if pc == b.pc {
            let mut match_bp = false;
            if b.flags & bp::GDB != 0 {
                match_bp = true;
            } else if b.flags & bp::CPU != 0 {
                let ops = cpu.ops();
                match_bp = ops.debug_check_breakpoint(cpu);
            }
            if match_bp {
                cpu.core.exception_index = excp::DEBUG;
                return true;
            }
        } else if (pc ^ b.pc) & page_mask == 0 {
            match_page = true;
        }
    }
    // Within the same page as a breakpoint, single-step, returning to helper_lookup_tb_ptr
    // after each insn looking for the actual breakpoint.
    if match_page {
        *cflags = (*cflags & !cf::COUNT_MASK) | cf::NO_GOTO_TB | cf::BP_PAGE | 1;
    }
    false
}

/// `helper_lookup_tb_ptr()`: the block to continue with, for `lookup_and_goto_ptr`.
pub(crate) fn helper_lookup_tb_ptr(cpu: &mut Cpu<'_>) -> Result<Option<Arc<Tb>>, CpuLoopExit> {
    let mut s = lookup_tb_ptr_state(cpu);
    lookup_tb_ptr_cflags(cpu, &mut s)?;
    tb_lookup_with(cpu, s, Arc::clone)
}

/// [`helper_lookup_tb_ptr`] for native code, which gets the host address to jump to as
/// [`tb_lookup_jump`] finds it, or the block when it cannot jump there.
#[inline]
pub(crate) fn helper_lookup_tb_ptr_jump(
    cpu: &mut Cpu<'_>,
    key: [u64; 2],
    jump: impl FnOnce(&Arc<Tb>) -> Option<u64>,
) -> Result<Option<Result<u64, Arc<Tb>>>, CpuLoopExit> {
    // The state stays where the target wrote it and is passed by reference: moving it as a
    // whole reads the two 32-bit fields just written with one 64-bit load, which the host
    // cannot forward from the stores.
    let mut s = lookup_tb_ptr_state(cpu);
    lookup_tb_ptr_cflags(cpu, &mut s)?;
    tb_lookup_jump(cpu, &s, key, jump)
}

/// The start of `helper_lookup_tb_ptr()`: the state to look the next block up by, without its
/// compile flags.
#[inline]
fn lookup_tb_ptr_state(cpu: &mut Cpu<'_>) -> TbCpuState {
    // By definition we've just finished a TB, so I/O is OK. Avoid the possibility of calling
    // cpu_io_recompile() if a page table walk triggered by tb_lookup() calling
    // probe_access_internal() happens to touch an MMIO device. The next TB, if we chain to
    // it, will clear the flag again.
    cpu.set_can_do_io(true);
    let c: &Cpu<'_> = cpu;
    c.core.ops.get_tb_cpu_state(c)
}

/// The compile flags of `s` and the breakpoint check of `helper_lookup_tb_ptr()`.
#[inline]
fn lookup_tb_ptr_cflags(cpu: &mut Cpu<'_>, s: &mut TbCpuState) -> Result<(), CpuLoopExit> {
    s.cflags = cpu.curr_cflags();
    if check_for_breakpoints(cpu, s.pc, &mut s.cflags) {
        return Err(cpu.cpu_loop_exit());
    }
    Ok(())
}

/// `cpu_tb_exec()`: run `itb` and what it chains to.
fn cpu_tb_exec(cpu: &mut Cpu<'_>, itb: &Arc<Tb>) -> Result<TbRet, CpuLoopExit> {
    let jit = cpu.jit();
    let ret = jit.backend.exec(cpu, itb);
    crate::plugin::disable_mem_helpers();
    let ret = ret?;
    cpu.set_can_do_io(true);
    if ret.exit > tb_exit::IDX1 {
        // We didn't start executing this TB (eg because the instruction counter hit zero); we
        // must restore the guest PC to the address of the start of the TB.
        let last = ret.last_tb.clone().expect("TB_EXIT_REQUESTED without a block");
        let ops = cpu.ops();
        ops.synchronize_from_tb(cpu, &last);
    }
    // If gdb single-step, and we haven't raised another exception, raise a debug exception.
    // Single-step with another exception is handled in cpu_handle_exception.
    if cpu.core.singlestep_enabled && cpu.core.exception_index == -1 {
        cpu.core.exception_index = excp::DEBUG;
        return Err(cpu.cpu_loop_exit());
    }
    Ok(ret)
}

/// `cpu_loop_exec_tb()`.
fn cpu_loop_exec_tb(
    cpu: &mut Cpu<'_>,
    tb: &Arc<Tb>,
    last_tb: &mut Option<Arc<Tb>>,
    tb_exit_out: &mut usize,
) -> Result<(), CpuLoopExit> {
    let ret = cpu_tb_exec(cpu, tb)?;
    *tb_exit_out = (ret.exit & tb_exit::MASK) as usize;
    if ret.exit != tb_exit::REQUESTED {
        *last_tb = ret.last_tb;
        return Ok(());
    }
    // Something asked us to stop executing chained TBs; just continue round the main loop.
    // Whatever requested the exit will also have set something else (eg exit_request or
    // interrupt_request) which will be handled by cpu_handle_interrupt. Without icount there
    // is no other reason for this exit.
    *last_tb = None;
    Ok(())
}

/// One pass of `cpu_exec_loop()`, from the exception check until it returns or something
/// leaves with `cpu_loop_exit()`.
fn cpu_exec_loop(cpu: &mut Cpu<'_>) -> Result<i32, CpuLoopExit> {
    let jit = cpu.jit();
    let ops = cpu.ops();
    // If an exception is pending, we execute it here.
    loop {
        if let Some(ret) = cpu_handle_exception(cpu) {
            return Ok(ret);
        }
        let mut last_tb: Option<Arc<Tb>> = None;
        let mut tb_exit_n = 0usize;

        while !cpu_handle_interrupt(cpu, &mut last_tb) {
            let mut s: TbCpuState = ops.get_tb_cpu_state(cpu);
            s.cflags = cpu.core.cflags_next_tb;
            // When requested, use an exact setting for cflags for the next execution. This is
            // used for precise smc and stop-after-access watchpoints.
            if s.cflags == u32::MAX {
                s.cflags = cpu.curr_cflags();
            } else {
                cpu.core.cflags_next_tb = u32::MAX;
            }

            if check_for_breakpoints(cpu, s.pc, &mut s.cflags) {
                break;
            }

            let tb = match tb_lookup(cpu, s)? {
                Some(tb) => tb,
                None => {
                    let flushes = jit.tb_flush_count();
                    let tb = tb_gen_code(cpu, s)?;
                    if jit.tb_flush_count() != flushes {
                        last_tb = None;
                    }
                    // We add the TB in the virtual pc hash table for the fast lookup.
                    jc_set(cpu, s.pc, &tb);
                    tb
                }
            };

            // We don't take care of direct jumps when address mapping changes in system
            // emulation. So it's not safe to make a direct jump to a TB spanning two pages
            // because the mapping for the second page can change.
            if tb.page_addr[1] != u64::MAX {
                last_tb = None;
            }
            // See if we can patch the calling TB.
            if let Some(last) = &last_tb {
                jit.tb_add_jump(last, tb_exit_n, &tb);
            }

            cpu_loop_exec_tb(cpu, &tb, &mut last_tb, &mut tb_exit_n)?;
        }
    }
}

thread_local! {
    /// QEMU's `current_cpu`, as the index of the vCPU this thread runs guest code for.
    static CURRENT_CPU: Cell<Option<usize>> = const { Cell::new(None) };
}

/// The `cpu_index` of the vCPU whose guest code this thread is running, QEMU's
/// `current_cpu->cpu_index`. Devices that bank registers per CPU, like the GICv2, use it to
/// tell which CPU made an access. `None` outside [`cpu_exec`] and `cpu_exec_step_atomic()`.
pub fn current_cpu_index() -> Option<usize> {
    CURRENT_CPU.with(Cell::get)
}

/// Sets [`current_cpu_index`] for as long as it lives, then puts back the previous value.
struct CurrentCpu(Option<usize>);

impl CurrentCpu {
    fn enter(cpu: &Cpu<'_>) -> CurrentCpu {
        CurrentCpu(CURRENT_CPU.with(|c| c.replace(Some(cpu.shared().cpu_index))))
    }
}

impl Drop for CurrentCpu {
    fn drop(&mut self) {
        CURRENT_CPU.with(|c| c.set(self.0));
    }
}

/// `cpu_exec()`: run guest code until something makes the CPU stop. Returns an `EXCP_*` code:
/// [`excp::HALTED`], [`excp::INTERRUPT`], [`excp::HLT`], [`excp::DEBUG`], [`excp::YIELD`],
/// [`excp::ATOMIC`] or a target code at or above [`excp::INTERRUPT`].
pub fn cpu_exec(cpu: &mut Cpu<'_>) -> i32 {
    let _current = CurrentCpu::enter(cpu);
    if cpu_handle_halt(cpu) {
        return excp::HALTED;
    }
    let ops = cpu.ops();
    ops.cpu_exec_enter(cpu);
    let ret = loop {
        match cpu_exec_loop(cpu) {
            Ok(ret) => break ret,
            // The longjmp: cpu_exec_longjmp_cleanup() has nothing to do here.
            Err(_) => {
                cpu.core.current_tb = None;
                cpu.core.unwinding = None;
            }
        }
    };
    ops.cpu_exec_exit(cpu);
    ret
}

/// `cpu_exec_step_atomic()`: run one instruction with every other vCPU stopped and without
/// `CF_PARALLEL`, after a block left with [`excp::ATOMIC`].
pub fn cpu_exec_step_atomic(cpu: &mut Cpu<'_>) {
    let _current = CurrentCpu::enter(cpu);
    let jit = cpu.jit();
    let shared = cpu.shared();
    jit.start_exclusive();
    assert!(!shared.running.load(Ordering::SeqCst));
    shared.running.store(true, Ordering::SeqCst);

    let r: Result<(), CpuLoopExit> = (|| {
        let ops = cpu.ops();
        let mut s = ops.get_tb_cpu_state(cpu);
        s.cflags = cpu.curr_cflags();
        // Execute in a serial context.
        s.cflags &= !cf::PARALLEL;
        // After 1 insn, return and release the exclusive lock.
        s.cflags |= cf::NO_GOTO_TB | cf::NO_GOTO_PTR | 1;
        // No need to check_for_breakpoints here. We only arrive in cpu_exec_step_atomic
        // after beginning execution of an insn that includes an atomic operation we can't
        // handle. Any breakpoint for this insn will have been recognized earlier.
        let tb = match tb_lookup(cpu, s)? {
            Some(tb) => tb,
            None => tb_gen_code(cpu, s)?,
        };
        ops.cpu_exec_enter(cpu);
        // execute the generated code
        cpu_tb_exec(cpu, &tb)?;
        ops.cpu_exec_exit(cpu);
        Ok(())
    })();
    if r.is_err() {
        cpu.core.current_tb = None;
        cpu.core.unwinding = None;
    }

    // As we start the exclusive region before codegen we must still be in the region if we
    // longjump out of either the codegen or the execution.
    assert!(crate::cpu::in_exclusive_context());
    shared.running.store(false, Ordering::SeqCst);
    jit.end_exclusive();
}

/// `tcg_cpu_exec()`: [`cpu_exec`] between `cpu_exec_start()` and `cpu_exec_end()`.
pub fn tcg_cpu_exec(cpu: &mut Cpu<'_>) -> i32 {
    let jit = cpu.jit();
    let shared = cpu.shared();
    jit.cpu_exec_start(&shared);
    let ret = cpu_exec(cpu);
    jit.cpu_exec_end(&shared);
    ret
}
