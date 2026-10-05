// SPDX-License-Identifier: GPL-2.0-or-later

//! The x86-64 host backend for the ruvm JIT, QEMU's `tcg/x86_64`.
//!
//! [`CodeRegion::compile`] turns a finished [`ruvm_jit_core::Func`] into x86-64 machine code in
//! an executable buffer, and [`CompiledTb::run`] runs it with the same inputs and the same
//! results as [`ruvm_jit_interp::Machine::run`]: the CPU state buffer, guest memory and the
//! helper registry go in, an [`ruvm_jit_interp::Exit`] or [`ruvm_jit_interp::InterpError`] comes
//! out. The interpreter is the reference; the tests run random blocks through both and compare.
//!
//! Values live in host registers, placed by the register allocator of
//! [`ruvm_jit_core::regalloc`]. Every scalar op is covered, and every 64 and 128-bit vector op
//! with SSE2 to SSE4.2 or AVX; 256-bit vectors need AVX2. Which extensions the code uses is a
//! [`HostFeatures`] value, detected with `cpuid` by default and narrowed with
//! [`CodeRegion::with_features`] to test the fallbacks. Blocks the backend cannot compile are
//! refused with [`GenCodeError::Unsupported`], so a caller can fall back to the interpreter.
//! Code can be generated on any host, but only run on an x86-64 one. Both the System V and the
//! Win64 calling conventions are supported.
//!
//! [`CompiledTb::run_chained`] runs a block and the blocks it chains to without coming back in
//! between: [`CompiledTb::set_goto_tb_target`] patches `goto_tb` jumps to go straight to the
//! next block, and a [`Chain`] supplies the blocks `lookup_and_goto_ptr` jumps to.
//!
//! All unsafe code is in the code buffer (mapping and writing executable memory), in the two
//! places the runtime crosses into and back out of generated code, and where the service
//! routine reads the metadata of the block that called it.

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
