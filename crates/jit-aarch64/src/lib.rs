// SPDX-License-Identifier: GPL-2.0-or-later

//! The AArch64 host backend for the ruvm JIT, QEMU's `tcg/aarch64`.
//!
//! [`CodeRegion::compile`] turns a finished [`ruvm_jit_core::Func`] into A64 machine code in an
//! executable buffer, and [`CompiledTb::run`] runs it with the same inputs and the same results
//! as [`ruvm_jit_interp::Machine::run`]: the CPU state buffer, guest memory and the helper
//! registry go in, an [`ruvm_jit_interp::Exit`] or [`ruvm_jit_interp::InterpError`] comes out.
//! The interpreter is the reference; the tests run random blocks through both and compare.
//!
//! Values live in host registers, placed by the register allocator of
//! [`ruvm_jit_core::regalloc`], and every scalar op and every 64 and 128 bit vector op is
//! covered (NEON). Blocks with wider temps are refused with [`GenCodeError::Unsupported`], so a
//! caller can fall back to the interpreter for those.
//! Code can be generated on any host, but only run on an AArch64 one.
//!
//! [`memory_order`] lowers the fence mappings of [`ruvm_jit_core::memory_model`] (QEMU's,
//! Risotto's and the RCpc one after Arancini) to `dmb`, `ldapr` and `stlr`, detects FEAT_LRCPC
//! and picks the mapping for a guest on this host.
//!
//! [`CompiledTb::run_chained`] runs a block and the blocks it chains to without coming back in
//! between: [`CompiledTb::set_goto_tb_target`] patches `goto_tb` jumps to go straight to the
//! next block, and a [`Chain`] supplies the blocks `lookup_and_goto_ptr` jumps to.
//!
//! All unsafe code is in the code buffer (mapping, writing and flushing executable memory), in
//! the two places the runtime crosses into and back out of generated code, and where the
//! service routine reads the metadata of the block that called it.

#[allow(
    dead_code,
    reason = "the assembler is a whole port of tcg-target.c.inc's encoders, including parts \
              such as the softmmu fast path and the atomics that this backend does not use yet"
)]
mod asm;
mod buffer;
mod codegen;
pub mod memory_order;
mod runtime;

pub use buffer::{BufferError, CodeBuffer};
pub use codegen::{CodegenOptions, GenCodeError};
pub use memory_order::{HostFeatures, TARGET_DEFAULT_MO, select_fence_mapping};
pub use runtime::{
    Chain, ChainExit, CodeRegion, CompileOptions, CompiledTb, Found, HostWindow, MAX_SLOT_WORDS,
};
