// SPDX-License-Identifier: GPL-2.0-or-later

//! The RISC-V 64 host backend for the ruvm JIT, QEMU's `tcg/riscv64`.
//!
//! [`CodeRegion::compile`] turns a finished [`ruvm_jit_core::Func`] into RV64GC machine code in
//! an executable buffer, and [`CompiledTb::run`] runs it with the same inputs and the same
//! results as [`ruvm_jit_interp::Machine::run`]: the CPU state buffer, guest memory and the
//! helper registry go in, an [`ruvm_jit_interp::Exit`] or [`ruvm_jit_interp::InterpError`] comes
//! out. The interpreter is the reference; the tests run random blocks through both and compare.
//!
//! Values live in host registers, placed by the register allocator of
//! [`ruvm_jit_core::regalloc`]. Every scalar op is covered with the base RV64IM instructions.
//! Zba, Zbb, Zbs and Zicond are used when [`HostFeatures`] has them, detected with the
//! `riscv_hwprobe` system call and a `SIGILL` probe as QEMU's `cpuinfo_init` does, and each
//! has a fallback. Vector ops are refused for now with [`GenCodeError::Unsupported`], so a
//! caller runs such blocks with the interpreter. Code can be generated on any host, but only
//! run on a riscv64 Linux one.
//!
//! [`CompiledTb::run_chained`] runs a block and the blocks it chains to without coming back in
//! between: [`CompiledTb::set_goto_tb_target`] patches `goto_tb` jumps to go straight to the
//! next block, and a [`Chain`] supplies the blocks `lookup_and_goto_ptr` jumps to.
//!
//! All unsafe code is in the code buffer (mapping and writing executable memory), in the
//! feature probe, in the two places the runtime crosses into and back out of generated code,
//! and where the service routine reads the metadata of the block that called it.

#[allow(
    dead_code,
    reason = "the assembler is a port of tcg-target.c.inc's encoders, including encodings this \
              backend does not use yet"
)]
mod asm;
mod buffer;
mod codegen;
mod features;
mod runtime;

pub use buffer::{BufferError, CodeBuffer};
pub use codegen::GenCodeError;
pub use features::HostFeatures;
pub use runtime::{
    Chain, ChainExit, CodeRegion, CompileOptions, CompiledTb, Found, MAX_SLOT_WORDS,
};
