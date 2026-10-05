// SPDX-License-Identifier: GPL-2.0-or-later

//! Building translation blocks, `tb_gen_code()` from `translate-all.c` with the code buffer
//! handling of `tcg/tcg.c`, and the state restore entry points `cpu_io_recompile()` and
//! `tb_check_watchpoint()`.
//!
//! Differences from QEMU:
//!
//! - The IR is built in a fresh [`Func`] per block instead of a reset per thread context.
//! - The return address used to restore state is the last `insn_start` of the running block
//!   (see [`crate::cpu::Ra`]), so a block is found from the vCPU, not from a host PC.
//! - A block that crosses into a second page holds no page locks while it is translated; see the
//!   crate docs.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use ruvm_jit_core::liveness::LogMask;
use ruvm_jit_core::{Func, FuncConfig, Opcode};

use crate::backend::GenCodeError;
use crate::cpu::{Cpu, CpuLoopExit, Ra};
use crate::cputlb;
use crate::jit::TB_HEADER_COST;
use crate::tb::{Tb, TbCpuState, lock};
use crate::tb_maint::queue_tb_flush;
use crate::{TCG_MAX_INSNS, cf, excp};

/// A block being translated: the IR and the fields of `TranslationBlock` the front end reads
/// and writes.
#[derive(Debug)]
pub struct TbBuild {
    /// The IR of the block.
    pub f: Func,
    /// The block's id, the value `exit_tb` names it by.
    pub id: u64,
    /// The guest PC of the first instruction.
    pub pc: u64,
    /// `cs_base`.
    pub cs_base: u64,
    /// Target flags.
    pub flags: u32,
    /// Compile flags.
    pub cflags: u32,
    /// The `ram_addr` of the first byte and of the second page; `u64::MAX` for none.
    pub page_addr: [u64; 2],
    /// Bytes of guest code covered, set by [`crate::translator_loop`].
    pub size: u32,
    /// Instructions translated, set by [`crate::translator_loop`].
    pub icount: u16,
    /// The most instructions the block may hold.
    pub max_insns: u32,
}

impl TbBuild {
    /// `tb_cflags()`.
    pub fn cflags(&self) -> u32 {
        self.cflags
    }
}

/// `tcg_tb_alloc()`: whether the code buffer has room for another block header.
fn tcg_tb_alloc(cpu: &Cpu<'_>) -> bool {
    let jit = &cpu.core.jit;
    let r = lock(&jit.region);
    !r.full && r.used + TB_HEADER_COST <= jit.config.code_gen_buffer_size
}

/// `tb_gen_code()`: translate the block at `s`, link it and return it. A block that is
/// already in the hash table (translated by another vCPU meanwhile) is returned instead.
pub fn tb_gen_code(cpu: &mut Cpu<'_>, s: TbCpuState) -> Result<Arc<Tb>, CpuLoopExit> {
    let jit = cpu.jit();
    let ops = cpu.ops();
    let mut cflags = s.cflags;
    assert!(cflags & cf::INVALID == 0);

    let phys_pc = cputlb::get_page_addr_code(cpu, s.pc)?;
    if phys_pc == u64::MAX {
        // Generate a one-shot TB with 1 insn in it.
        cflags = (cflags & !cf::COUNT_MASK) | 1;
    }
    let mut max_insns = cflags & cf::COUNT_MASK;
    if max_insns == 0 {
        max_insns = TCG_MAX_INSNS;
    }

    'buffer_overflow: loop {
        if !tcg_tb_alloc(cpu) {
            // Flush must be done.
            if cpu.in_serial_context() {
                jit.tb_flush_exclusive_or_serial();
                continue 'buffer_overflow;
            }
            queue_tb_flush(cpu);
            // Make the execution loop process the flush as soon as possible.
            cpu.core.exception_index = excp::INTERRUPT;
            return Err(cpu.cpu_loop_exit());
        }
        let id = jit.next_tb_id.fetch_add(4, Ordering::Relaxed);

        loop {
            // tb_overflow:
            let config = FuncConfig {
                parallel: cflags & cf::PARALLEL != 0,
                user_only: false,
                guest_mo: ops.guest_default_memory_order(),
                target_default_mo: jit.backend.target_default_mo(),
                addr_type: ops.addr_type(),
                no_goto_ptr: cflags & cf::NO_GOTO_PTR != 0,
                fence_mapping: jit.backend.fence_mapping(ops.guest_default_memory_order()),
                tlb_page_bits: Some(jit.config.page_bits),
            };
            let mut b = TbBuild {
                f: Func::new(config),
                id,
                pc: s.pc,
                cs_base: s.cs_base,
                flags: s.flags,
                cflags,
                page_addr: [phys_pc, u64::MAX],
                size: 0,
                icount: 0,
                max_insns,
            };
            // setjmp_gen_code()
            ops.translate_code(cpu, &mut b)?;
            assert!(b.size != 0);
            max_insns = u32::from(b.icount);
            b.f.gen_code(jit.config.optimize, LogMask::default());
            let mut goto_tb_used = [false; 2];
            for (_, op) in b.f.ops() {
                if op.opc == Opcode::GotoTb {
                    goto_tb_used[op.args[0] as usize & 1] = true;
                }
            }
            let (code, code_size) = match jit.backend.gen_code(b.f, id) {
                Ok(x) => x,
                // Overflow of code_gen_buffer, or the current slice of it.
                Err(GenCodeError::BufferFull) => {
                    lock(&jit.region).full = true;
                    continue 'buffer_overflow;
                }
                // The code generated for the TranslationBlock is too large. The maximum size
                // allowed by the unwind info is 64k. There may be stricter constraints from
                // relocations in the tcg backend.
                //
                // Try again with half as many insns as we attempted this time. If a single
                // insn overflows, there's a bug somewhere...
                Err(GenCodeError::TooLarge) => {
                    assert!(max_insns > 1, "a single instruction overflowed the code buffer");
                    max_insns /= 2;
                    continue;
                }
            };
            {
                let mut r = lock(&jit.region);
                let total = TB_HEADER_COST + code_size;
                if r.full || r.used + total > jit.config.code_gen_buffer_size {
                    r.full = true;
                    continue 'buffer_overflow;
                }
                r.used += total;
            }
            let state = TbCpuState { pc: s.pc, flags: s.flags, cflags, cs_base: s.cs_base };
            let tb = Arc::new(Tb::new(
                id,
                state,
                b.size,
                b.icount,
                b.page_addr,
                goto_tb_used,
                code,
                code_size,
            ));
            jit.backend.tb_created(&tb);
            // Init original jump addresses.
            for (n, used) in goto_tb_used.iter().enumerate() {
                if *used {
                    jit.backend.set_jmp_target(&tb, n, None);
                }
            }
            // tcg_tb_insert(): before publishing through the hash table.
            lock(&jit.region).tbs.insert(id, tb.clone());

            // If the TB is not associated with a physical RAM page then it must be a temporary
            // one-insn TB, executed at most once. Return early before attempting to link to
            // other TBs or add to the hash table.
            if tb.page_addr[0] == u64::MAX {
                return Ok(tb);
            }

            let existing = jit.tb_link_page(&tb);
            // If the TB already exists, discard what we just translated.
            if !Arc::ptr_eq(&existing, &tb) {
                let mut r = lock(&jit.region);
                if r.tbs.remove(&id).is_some() {
                    r.used -= TB_HEADER_COST + tb.code_size;
                }
                return Ok(existing);
            }
            return Ok(tb);
        }
    }
}

/// `cpu_io_recompile()`: an IO access happened in an instruction that was not the last of its
/// block. Run it again in a block of its own.
pub fn cpu_io_recompile(cpu: &mut Cpu<'_>, ra: Ra) -> CpuLoopExit {
    if ra == Ra::None || cpu.core.current_tb.is_none() {
        panic!("cpu_io_recompile: could not find TB for pc=0x0");
    }
    cpu.cpu_restore_state(ra);
    // Exit the loop and potentially generate a new TB executing the just the I/O insns. We
    // also limit instrumentation to memory operations only (which execute after completion)
    // so we don't double instrument the instruction. Also don't let an IRQ sneak in before we
    // execute it.
    cpu.core.cflags_next_tb = cpu.curr_cflags() | cf::MEMI_ONLY | cf::NOIRQ | 1;
    cpu.cpu_loop_exit_noexc()
}

/// `tb_check_watchpoint()`: a watchpoint hit; restore the state and drop the block so that the
/// access is translated again on its own.
pub fn tb_check_watchpoint(cpu: &mut Cpu<'_>, ra: Ra) {
    let jit = cpu.jit();
    let tb = if ra == Ra::Tb { cpu.core.current_tb.clone() } else { None };
    match tb {
        Some(tb) => {
            // We can use retranslation to find the PC.
            cpu.cpu_restore_state(ra);
            jit.tb_phys_invalidate(&tb, u64::MAX);
        }
        None => {
            // The exception probably happened in a helper. The CPU state should have been
            // saved before calling it. Fetch the PC from there.
            let ops = cpu.ops();
            let s = ops.get_tb_cpu_state(cpu);
            if let Ok(addr) = cputlb::get_page_addr_code(cpu, s.pc) {
                if addr != u64::MAX {
                    jit.tb_invalidate_phys_range(addr, addr);
                }
            }
        }
    }
}
