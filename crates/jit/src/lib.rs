// SPDX-License-Identifier: GPL-2.0-or-later

//! The JIT runtime, ported from QEMU's `accel/tcg`: `cpu-exec.c`, `translate-all.c`,
//! `tb-maint.c`, `cputlb.c`, `translator.c`, `watchpoint.c`, the `tcg-accel-ops*.c` vCPU
//! threads, `cpus-common.c`, and the translation block parts of `tcg/tcg.c`.
//!
//! The pieces:
//!
//! - [`backend::Backend`] turns a finished [`ruvm_jit_core::Func`] into something that runs, and
//!   patches `goto_tb` jumps. [`backend::InterpBackend`] does it with `ruvm-jit-interp`.
//! - [`translator`] is the guest front end contract, QEMU's `TranslatorOps` and
//!   `translator_loop()`, including `translator_ld*` and the page crossing rules.
//! - [`jit::Jit`] holds what QEMU keeps in `tb_ctx`, `tcg_ctx` and the CPU list: the translation
//!   block hash table, the code region, the per page lists of blocks, and the exclusive section
//!   state.
//! - [`translate`] is `tb_gen_code()` and the state restore paths; [`tb_maint`] is lookup,
//!   chaining, invalidation and `tb_flush`.
//! - [`cputlb`] is the softmmu TLB: fast and victim tables per MMU index, `tlb_set_page`, the
//!   flush family including the cross CPU forms, and the load and store slow paths with IO,
//!   notdirty and watchpoint handling.
//! - [`cpu`] is the vCPU (`CPUState`): interrupts, work items, exclusive sections, breakpoints and
//!   watchpoints. [`cpu_exec`] is the execution loop and [`accel`] the MTTCG and round robin
//!   vCPU threads.
//!
//! # The CPU state buffer
//!
//! Generated code runs against a byte buffer, `env`. Its first [`ENV_TARGET_OFFSET`] bytes are
//! the runtime's, standing in for QEMU's `CPUNegativeOffsetState`: `icount_decr` as a little
//! endian u32 at [`ENV_ICOUNT_DECR_OFFSET`] and `can_do_io` as a byte at
//! [`ENV_CAN_DO_IO_OFFSET`]. The target's own state follows. The interpreter has no negative
//! offsets, so the runtime state comes first instead of just before `env`.
//!
//! # Differences from QEMU
//!
//! - A `cpu_loop_exit()` longjmp is a Rust value: functions that can leave the execution loop
//!   return `Result<_, CpuLoopExit>` and the loop restarts where `sigsetjmp` would return. A
//!   [`cpu::CpuLoopExit`] can only be made by the calls that set up the exit
//!   (`cpu_loop_exit()` and friends), so the state change always happens first.
//! - The return address used to find the faulting instruction is [`cpu::Ra`]: either "not in
//!   generated code" or "in the current block". The instruction is the one whose `insn_start`
//!   ran last, which the interpreter reports through `GuestMemory::insn_start`, instead of a
//!   search of the block's unwind data by host PC.
//! - `icount_decr` lives in an atomic shared with other threads and is copied into `env` before
//!   every block, chained ones included, since generated code only sees `env`. `can_do_io` lives
//!   only in `env`.
//! - There is no big QEMU lock. Device callbacks (MMIO) are called without one; `ruvm-mem`
//!   devices do their own locking. Work items and `run_on_cpu()` wait on the runtime's own lock.
//! - Translation holds no page locks, and one mutex covers all the per page block lists. Two
//!   threads may translate the same block at once; the second to link it finds the first in the
//!   hash table and drops its own copy, as QEMU does.
//! - The code buffer is one region with a byte budget. Generated code is kept alive by the
//!   region until `tb_flush`, so a stale pointer to a block is always safe to follow.
//! - Physical pages of RAM are named by a runtime assigned `ram_addr`: each `RamBlock` gets a
//!   page aligned base the first time the TLB sees it. The TLB `addend` turns a guest virtual
//!   address into that `ram_addr` instead of a host pointer.
//! - Self modifying code tracking uses the runtime's own "page holds translated code" bit instead
//!   of the `DIRTY_MEMORY_CODE` bitmap. Writes that do not go through the softmmu, such as DMA,
//!   must call [`jit::Jit::tb_invalidate_phys_range`] themselves; QEMU does it from
//!   `invalidate_and_set_dirty()`.
//! - The runtime does not listen to memory map changes. A machine that changes the map must
//!   flush the TLBs, which is what QEMU's `tcg_commit()` does. IO accesses are dispatched through
//!   the CPU's address space by physical address instead of through a saved section.
//! - Ranges behind an IOMMU are treated as IO.
//! - Only the legacy `tlb_fill` hook is supported, so alignment is checked before paging, and the
//!   interpreter checks alignment before it calls the softmmu. `MO_ALIGN_TLB_ONLY` and
//!   `TLB_CHECK_ALIGNED` are therefore checked as plain alignment.
//! - Guest atomics under `CF_PARALLEL` are made indivisible by one global lock around the
//!   load and store pair, since the interpreter's atomic helpers are a load and a store.
//! - No icount, no `-d exec` style logging, no perf maps, and no user mode. TCG plugins are
//!   supported through [`plugin`], with the differences listed there.

#![forbid(unsafe_code)]

pub mod accel;
pub mod backend;
pub mod cpu;
pub mod cpu_exec;
pub mod cputlb;
pub mod jit;
pub mod plugin;
pub mod tb;
pub mod tb_maint;
pub mod translate;
pub mod translator;

pub use backend::{Backend, GenCodeError, InterpBackend, TbRet};
pub use cpu::{
    Breakpoint, Cpu, CpuCore, CpuLoopExit, CpuOps, CpuShared, MmuAccessType, Ra, Vcpu, Watchpoint,
};
pub use cputlb::{TlbEntryFull, TlbSection};
pub use jit::{Jit, JitConfig};
pub use tb::{Tb, TbCpuState};
pub use translator::{DisasContextBase, DisasJumpType, TranslatorOps, translator_loop};

/// Offset in `env` of `icount_decr`, a little endian u32. Its high half is the exit request:
/// `0xffff` there makes the value negative, which ends the block at its start check.
pub const ENV_ICOUNT_DECR_OFFSET: i64 = 0;
/// Offset in `env` of `can_do_io`, one byte.
pub const ENV_CAN_DO_IO_OFFSET: i64 = 4;
/// Offset in `env` where the target's own state starts.
pub const ENV_TARGET_OFFSET: usize = 16;

/// The largest number of guest instructions in one block, `TCG_MAX_INSNS`.
pub const TCG_MAX_INSNS: u32 = 512;

/// The op count at which the translator stops adding instructions, `OPC_MAX_SIZE` in
/// `tcg_op_buf_full()`.
pub const OPC_MAX_SIZE: usize = 4000;

/// `TB_JMP_CACHE_BITS`.
pub const TB_JMP_CACHE_BITS: u32 = 12;
/// `TB_JMP_CACHE_SIZE`.
pub const TB_JMP_CACHE_SIZE: usize = 1 << TB_JMP_CACHE_BITS;

/// The compile flags of a translation block, `CF_*`.
pub mod cf {
    /// The instruction count, zero meaning [`crate::TCG_MAX_INSNS`].
    pub const COUNT_MASK: u32 = 0x0000_01ff;
    /// Do not chain with `goto_tb`.
    pub const NO_GOTO_TB: u32 = 0x0000_0200;
    /// Do not chain with `goto_ptr`.
    pub const NO_GOTO_PTR: u32 = 0x0000_0400;
    /// gdbstub single step.
    pub const SINGLE_STEP: u32 = 0x0000_0800;
    /// Only instrument memory ops.
    pub const MEMI_ONLY: u32 = 0x0000_1000;
    /// icount is on.
    pub const USE_ICOUNT: u32 = 0x0000_2000;
    /// The block is stale.
    pub const INVALID: u32 = 0x0000_4000;
    /// Other vCPUs run in parallel.
    pub const PARALLEL: u32 = 0x0000_8000;
    /// Generate an uninterruptible block.
    pub const NOIRQ: u32 = 0x0001_0000;
    /// The code is position independent.
    pub const PCREL: u32 = 0x0002_0000;
    /// A breakpoint is on the block's page.
    pub const BP_PAGE: u32 = 0x0004_0000;
    /// The cluster index.
    pub const CLUSTER_MASK: u32 = 0xff00_0000;
    /// Shift of the cluster index.
    pub const CLUSTER_SHIFT: u32 = 24;
}

/// Exception indexes used by the runtime, `EXCP_*` from `cpu-all.h`.
pub mod excp {
    /// Async interruption.
    pub const INTERRUPT: i32 = 0x10000;
    /// The CPU halted.
    pub const HLT: i32 = 0x10001;
    /// cpu stopped after a breakpoint or single step.
    pub const DEBUG: i32 = 0x10002;
    /// cpu is halted (waiting for external event).
    pub const HALTED: i32 = 0x10003;
    /// cpu wants to yield timeslice to another.
    pub const YIELD: i32 = 0x10004;
    /// stop the world and emulate atomic.
    pub const ATOMIC: i32 = 0x10005;
}

/// `CPU_INTERRUPT_*` bits of `interrupt_request`.
pub mod interrupt {
    /// External hardware interrupt pending.
    pub const HARD: u32 = 0x0002;
    /// Exit the current block.
    pub const EXITTB: u32 = 0x0004;
    /// Halt the CPU.
    pub const HALT: u32 = 0x0020;
    /// Debug event pending.
    pub const DEBUG: u32 = 0x0080;
    /// Reset signal pending.
    pub const RESET: u32 = 0x0400;
    /// The first target specific bit, `CPU_INTERRUPT_TGT_EXT_0`.
    pub const TGT_EXT_0: u32 = 0x0008;
    /// `CPU_INTERRUPT_TGT_INT_0`.
    pub const TGT_INT_0: u32 = 0x0100;
}

/// Breakpoint and watchpoint flags, `BP_*`.
pub mod bp {
    /// Watch reads.
    pub const MEM_READ: u32 = 0x01;
    /// Watch writes.
    pub const MEM_WRITE: u32 = 0x02;
    /// Watch both.
    pub const MEM_ACCESS: u32 = MEM_READ | MEM_WRITE;
    /// Stop before the access instead of after it.
    pub const STOP_BEFORE_ACCESS: u32 = 0x04;
    /// Set by the gdbstub.
    pub const GDB: u32 = 0x10;
    /// Set by the CPU model.
    pub const CPU: u32 = 0x20;
    /// Any origin.
    pub const ANY: u32 = GDB | CPU;
    /// Shift of the hit bits.
    pub const HIT_SHIFT: u32 = 6;
    /// Read hit.
    pub const WATCHPOINT_HIT_READ: u32 = MEM_READ << HIT_SHIFT;
    /// Write hit.
    pub const WATCHPOINT_HIT_WRITE: u32 = MEM_WRITE << HIT_SHIFT;
    /// Any hit.
    pub const WATCHPOINT_HIT: u32 = MEM_ACCESS << HIT_SHIFT;
}

/// Page protection bits, `PAGE_*`.
pub mod page {
    /// Readable.
    pub const READ: u32 = 0x0001;
    /// Writable.
    pub const WRITE: u32 = 0x0002;
    /// Executable.
    pub const EXEC: u32 = 0x0004;
    /// Every access.
    pub const RWX: u32 = READ | WRITE | EXEC;
    /// Invalidate the TLB entry right after a write, `PAGE_WRITE_INV`.
    pub const WRITE_INV: u32 = 0x0020;
}

/// TLB entry flags, `TLB_*`, for system mode.
pub mod tlb {
    /// Swap the byte order of every access.
    pub const BSWAP: u32 = 1 << 0;
    /// A watchpoint is on the page.
    pub const WATCHPOINT: u32 = 1 << 1;
    /// Check alignment on every access.
    pub const CHECK_ALIGNED: u32 = 1 << 2;
    /// Drop writes.
    pub const DISCARD_WRITE: u32 = 1 << 3;
    /// Access through the device callbacks.
    pub const MMIO: u32 = 1 << 4;
    /// The flags kept in the full entry, `TLB_SLOW_FLAGS_MASK`.
    pub const SLOW_FLAGS_MASK: u32 = BSWAP | WATCHPOINT | CHECK_ALIGNED | DISCARD_WRITE | MMIO;
    /// The entry is invalid.
    pub const INVALID_MASK: u64 = 1 << 6;
    /// Writes must track dirty memory and self modifying code.
    pub const NOTDIRTY: u64 = 1 << 7;
    /// Look at the slow flags.
    pub const FORCE_SLOW: u64 = 1 << 8;
    /// The flags kept in the comparator, `TLB_FLAGS_MASK`.
    pub const FLAGS_MASK: u64 = INVALID_MASK | NOTDIRTY | FORCE_SLOW;
}

/// `TARGET_PAGE_BITS_MIN`: the smallest page size any target uses. The TLB flag bits sit
/// below it, so [`JitConfig::page_bits`](crate::jit::JitConfig) must be at least this.
pub const TARGET_PAGE_BITS_MIN: u32 = 9;

/// `CPU_VTLB_SIZE`.
pub const CPU_VTLB_SIZE: usize = 8;
/// `CPU_TLB_DYN_MIN_BITS`.
pub const CPU_TLB_DYN_MIN_BITS: u32 = 6;
/// `CPU_TLB_DYN_DEFAULT_BITS`.
pub const CPU_TLB_DYN_DEFAULT_BITS: u32 = 8;
/// `CPU_TLB_ENTRY_BITS`.
pub const CPU_TLB_ENTRY_BITS: u32 = 5;
