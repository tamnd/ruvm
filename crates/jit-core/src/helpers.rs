// SPDX-License-Identifier: MIT OR Apache-2.0

//! Declarations of the runtime helpers that the builder itself calls, from
//! `accel/tcg/tcg-runtime.h` and `accel/tcg/atomic_template.h`.
//!
//! The names are the `DEF_HELPER` names, which is what QEMU prints in dumps. An interpreter or
//! backend provides the implementations; `ruvm-jit-interp` has all of them built in.

use crate::ir::HelperInfo;
use crate::ir::HelperType::{I32, I64, I128, Ptr, Void};
use crate::types::MemOp;
use crate::types::call_flags::{NO_RETURN, NO_WG, NO_WG_SE};

/// `lookup_tb_ptr(env) -> ptr`.
pub fn lookup_tb_ptr() -> HelperInfo {
    HelperInfo::new("lookup_tb_ptr", NO_WG_SE, Ptr, &[Ptr])
}

/// `exit_atomic(env)`, which never returns.
pub fn exit_atomic() -> HelperInfo {
    HelperInfo::new("exit_atomic", NO_WG | NO_RETURN, Void, &[Ptr])
}

/// The suffix QEMU appends for a size and byte order, such as `w_le` or `b`.
pub fn size_suffix(memop: MemOp) -> &'static str {
    match memop.0 & (MemOp::SIZE.0 | MemOp::BSWAP.0) {
        0x00 | 0x10 => "b",
        0x01 => "w_le",
        0x11 => "w_be",
        0x02 => "l_le",
        0x12 => "l_be",
        0x03 => "q_le",
        0x13 => "q_be",
        0x04 => "o_le",
        0x14 => "o_be",
        _ => panic!("no atomic helper for this size"),
    }
}

fn value_type(memop: MemOp) -> crate::ir::HelperType {
    match memop.size() {
        0..=2 => I32,
        3 => I64,
        _ => I128,
    }
}

/// `atomic_cmpxchg{b,w_le,...}(env, addr, cmpv, newv, oi)`.
pub fn atomic_cmpxchg(memop: MemOp) -> HelperInfo {
    let v = value_type(memop);
    let name = format!("atomic_cmpxchg{}", size_suffix(memop));
    HelperInfo::new(&name, NO_WG, v, &[Ptr, I64, v, v, I32])
}

/// `atomic_{name}{b,w_le,...}(env, addr, val, oi)` for the read-modify-write operations.
pub fn atomic_op(name: &str, memop: MemOp) -> HelperInfo {
    let v = value_type(memop);
    let name = format!("atomic_{name}{}", size_suffix(memop));
    HelperInfo::new(&name, NO_WG, v, &[Ptr, I64, v, I32])
}
