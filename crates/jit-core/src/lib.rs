// SPDX-License-Identifier: MIT OR Apache-2.0

//! The JIT intermediate representation, a port of the generic half of QEMU's TCG.
//!
//! A [`ir::Func`] holds one translation block under construction: its temps (globals at fixed
//! offsets from `env`, TB temps, EBB temps and constants), its labels, the helpers it calls and
//! a doubly linked list of ops. The builder methods on `Func` follow `tcg-op.h`,
//! `tcg-op-ldst.c` and `tcg-op-vec.c` by name (`gen_add_i32` is `tcg_gen_add_i32`). The op set is
//! `tcg-opc.h` from QEMU 11.1, where integer ops carry their type on the op.
//!
//! The passes are:
//!
//! - [`ir::Func::verify`]: types, use before def and labels;
//! - [`ir::Func::optimize`]: the tier-1 optimizer of `tcg/optimize.c`;
//! - [`ir::Func::reachable_code_pass`], [`ir::Func::liveness_pass_0`],
//!   [`ir::Func::liveness_pass_1`] and [`ir::Func::liveness_pass_2`] from `tcg/tcg.c`;
//! - [`ir::Func::gen_code`], which runs them in QEMU's order and returns the `-d op,op_opt` log;
//! - [`ir::Func::dump_ops`], `tcg_dump_ops`;
//! - [`regalloc`], the target independent register allocator of `tcg/tcg.c`, which host
//!   backends drive through [`regalloc::Target`].
//!
//! Every op keeps QEMU's semantics bit for bit; `ruvm-jit-interp` is the reference interpreter.
//!
//! # Differences from QEMU
//!
//! - The target is a virtual host that implements every generic op, every vector op at every
//!   element size, the test conditions (`TCG_TARGET_HAS_tst`) and `deposit`, `extract` and
//!   `sextract` at every position and length. The builder therefore never emits the fallback
//!   expansions QEMU uses on hosts without them, and the optimizer never takes its lowering
//!   paths. One visible result: the optimizer turns `and` with any low mask of ones into
//!   `extract`, as QEMU does on hosts such as x86_64 and aarch64.
//! - The host is 64-bit and little endian and byte swapping memory operations are native, so
//!   `qemu_ld` and `qemu_st` keep `MO_BSWAP` and 128-bit accesses use `qemu_ld2` and `qemu_st2`.
//!   The out of line `ld_i128` and `st_i128` helpers are never called.
//! - Plugin ops are emitted as markers only; no memory callbacks are expanded.
//! - The generic vector expanders of `tcg-op-gvec.c` are not ported.
//! - Dumps never print `pref=` register preferences, which belong to a real register allocator.
//! - The optimizer keeps its env memory copies in a vector instead of an interval tree; the
//!   lookup order, and therefore the result, is the same.
//! - There is no single verifier in QEMU; [`verify`] gathers its debug assertions in one pass.
//! - Maps use [`hash::FastHasher`] rather than GLib's hashes.
//! - [`memory_model`] adds two fence mappings beside QEMU's `tcg_gen_req_mo` for guests such as
//!   x86 on weaker hosts. QEMU's mapping stays the default.

#![forbid(unsafe_code)]

pub mod dump;
pub mod hash;
pub mod helpers;
pub mod ir;
pub mod liveness;
pub mod memory_model;
pub mod opcode;
pub mod optimize;
pub mod regalloc;
pub mod tcg_op;
pub mod tcg_op_ldst;
pub mod tcg_op_vec;
pub mod types;
pub mod verify;

pub use ir::{Func, FuncConfig, HelperId, HelperInfo, HelperType, Label, Op, OpId, Temp};
pub use memory_model::FenceMapping;
pub use opcode::Opcode;
pub use types::{Cond, MemOp, MemOpIdx, TempKind, Type};
